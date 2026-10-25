"""Harmonica training on Lightning Fabric.

Section 4.1: summed binary cross-entropy over onset, sustain, and offset (and
velocity times ``model.head.velocity_weight``, with the head the paper does
not have), 100k steps at batch 8, AdamW with a cosine schedule from 2e-4.
Every ``val_freq`` steps rank 0 transcribes the whole development set at fixed
thresholds and logs the paper's three scores; thresholds are swept only at
evaluation.
"""

import math
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from harmonica.config import (
    HarmonicaConfig,
    config_to_dict,
    resolve_num_workers,
    resolve_widths,
    run_dir,
)
from harmonica.corpus import DEV, TRAIN, load_manifest
from harmonica.data import ClipStream
from harmonica.decoding import mean_scores
from harmonica.inference import (
    CHECKPOINT_FORMAT_VERSION,
    PIPELINE_NAME,
    run_recordings,
    score_all,
)
from harmonica.model import HEADS, VELOCITY, HarmonicaModel, count_parameters
from harmonica.requests import HarmonicaTrainRequest

CHECKPOINT_PREFIX = "checkpoint_"
CHECKPOINT_LATEST = "checkpoint_latest.ckpt"


def lr_scale(step: int, config: HarmonicaConfig) -> float:
    """Linear warmup (if any), then cosine decay to the floor."""
    training = config.training
    if step < training.warmup_steps:
        return (step + 1) / training.warmup_steps
    floor = training.min_learning_rate / training.learning_rate
    decay = max(1, training.max_steps - training.warmup_steps)
    progress = min(1.0, (step - training.warmup_steps) / decay)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def loss_terms(
    logits: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    velocity_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """One binary cross-entropy per head, against ``labels[:, head]``.

    Velocity, when the model has the head, is Kong et al.'s: cross-entropy
    against ``velocity / 127``, averaged over the onset frames whose velocity
    is known, and zero in a batch that has none. ``velocity_weight`` scales
    that term before it is returned (the detection heads stay unweighted).
    """
    terms = {
        name: F.binary_cross_entropy_with_logits(logits[name].float(), labels[:, i])
        for i, name in enumerate(HEADS)
    }
    if VELOCITY in logits:
        target = labels[:, len(HEADS)]
        known = (target > 0).float()
        each = F.binary_cross_entropy_with_logits(
            logits[VELOCITY].float(), target, reduction="none"
        )
        terms[VELOCITY] = (
            velocity_weight * (each * known).sum() / known.sum().clamp(min=1.0)
        )
    return terms


def checkpoint_path(config: HarmonicaConfig, step: int) -> Path:
    return run_dir(config) / "checkpoints" / f"{CHECKPOINT_PREFIX}{step}.ckpt"


def find_latest_checkpoint(config: HarmonicaConfig) -> Path | None:
    directory = run_dir(config) / "checkpoints"
    if not directory.is_dir():
        return None
    steps = [
        (int(path.stem.removeprefix(CHECKPOINT_PREFIX)), path)
        for path in directory.glob(f"{CHECKPOINT_PREFIX}*.ckpt")
        if path.stem.removeprefix(CHECKPOINT_PREFIX).isdigit()
    ]
    return max(steps)[1] if steps else None


def _save(
    fabric: Fabric,
    config: HarmonicaConfig,
    model: torch.nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    step: int,
) -> None:
    path = checkpoint_path(config, step)
    fabric.print(f"Saving {path}")
    fabric.save(
        path,
        {
            "pipeline": PIPELINE_NAME,
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "config": config_to_dict(config),
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
    )
    if fabric.global_rank == 0:
        link = path.parent / CHECKPOINT_LATEST
        staged = path.parent / f"{CHECKPOINT_LATEST}.new"
        staged.unlink(missing_ok=True)
        staged.symlink_to(path.name)
        staged.replace(link)


def _resume(
    fabric: Fabric,
    resume: str | Path | None,
    config: HarmonicaConfig,
    model: torch.nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
) -> int:
    if resume is None:
        return 0
    path = find_latest_checkpoint(config) if str(resume) == "auto" else Path(resume)
    if path is None:
        return 0
    fabric.print(f"Resuming from {path}")
    remainder = fabric.load(path, {"model": model, "optimizer": optimizer})
    scheduler.load_state_dict(remainder["scheduler"])
    return int(remainder["step"])


def _cycle(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


def _validate(
    fabric: Fabric, config: HarmonicaConfig, model: HarmonicaModel, dev: list
) -> dict[str, float]:
    """Dev-set scores at the configured thresholds, on rank 0 only."""
    module = model.module if hasattr(model, "module") else model
    module.eval()
    activations = run_recordings(module, dev, config.evaluation, fabric.device)
    cqt = config.model.cqt
    scores = mean_scores(
        score_all(
            dev,
            activations,
            config.evaluation,
            cqt.sample_rate / cqt.hop_length,
            config.training.val_onset_threshold,
            config.training.val_frame_threshold,
        )
    )
    module.train()
    return {f"val/{key}": value for key, value in scores.as_dict().items()}


def _run(fabric: Fabric, request: HarmonicaTrainRequest) -> None:
    config = request.config
    training = config.training
    fabric.seed_everything(training.seed)

    splits = load_manifest(config)
    fabric.print(
        f"Corpus: {len(splits[TRAIN])} training recordings, "
        f"{len(splits[DEV])} development recordings"
    )
    model = HarmonicaModel(config.model)
    widths = resolve_widths(config.model)
    fabric.print(
        f"Harmonica {config.model.size}: c1={widths.frontend_channels} "
        f"c2={widths.channels} l={widths.layers}, "
        f"{count_parameters(model):,} parameters"
    )
    decayed = [p for p in model.parameters() if p.dim() >= 2]
    plain = [p for p in model.parameters() if p.dim() < 2]
    optimizer = AdamW(
        [
            {"params": decayed, "weight_decay": training.weight_decay},
            {"params": plain, "weight_decay": 0.0},
        ],
        lr=training.learning_rate,
        betas=(training.beta1, training.beta2),
    )
    if training.compile:
        # Fabric sees through the wrapper; checkpoints hold the plain module's keys.
        model = torch.compile(model)
    model, optimizer = fabric.setup(model, optimizer)
    scheduler = LambdaLR(optimizer, lambda step: lr_scale(step, config))
    step = _resume(fabric, request.resume, config, model, optimizer, scheduler)

    for logger_ in fabric.loggers:
        _ = logger_.experiment
        logger_.log_hyperparams(config_to_dict(config))

    cqt = config.model.cqt
    stream = ClipStream(
        splits[TRAIN],
        sample_rate=cqt.sample_rate,
        hop_length=cqt.hop_length,
        clip_seconds=config.data.clip_seconds,
        temperature=config.data.temperature,
        # A resumed run draws clips it has not drawn yet.
        seed=training.seed + step,
        rank=fabric.global_rank,
    )
    workers = resolve_num_workers(config)
    loader = DataLoader(
        stream,
        batch_size=training.batch_size // training.devices,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    batches = _cycle(fabric.setup_dataloaders(loader, move_to_device=True))

    model.train()
    began = perf_counter()
    while step < training.max_steps:
        audio, labels = next(batches)
        logits = model(audio)
        terms = loss_terms(
            logits, labels, velocity_weight=config.model.head.velocity_weight
        )
        loss = sum(terms.values())
        optimizer.zero_grad(set_to_none=True)
        fabric.backward(loss)
        if training.grad_clip is not None:
            fabric.clip_gradients(model, optimizer, max_norm=training.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % training.log_freq == 0:
            elapsed = perf_counter() - began
            metrics = {
                "train/loss": loss.item(),
                **{f"train/loss_{name}": term.item() for name, term in terms.items()},
                "train/lr": scheduler.get_last_lr()[0],
                "train/steps_per_second": training.log_freq / elapsed,
            }
            fabric.log_dict(metrics, step=step)
            fabric.print(
                f"step {step}: loss {metrics['train/loss']:.4f} "
                f"({metrics['train/steps_per_second']:.1f} it/s)"
            )
            began = perf_counter()

        if step % training.val_freq == 0 or step == training.max_steps:
            if fabric.global_rank == 0 and splits[DEV]:
                scores = _validate(fabric, config, model, splits[DEV])
                fabric.log_dict(scores, step=step)
                fabric.print(
                    f"step {step}: dev "
                    + " ".join(f"{k[4:]} {v:.3f}" for k, v in scores.items())
                )
            fabric.barrier()
            began = perf_counter()

        if step % training.save_freq == 0 or step == training.max_steps:
            _save(fabric, config, model, optimizer, scheduler, step)


def train(request: HarmonicaTrainRequest) -> None:
    """Execute one Harmonica training request."""
    config = request.config
    run_dir(config).mkdir(parents=True, exist_ok=True)
    training = config.training
    wandb_logger = WandbLogger(
        project=config.wandb.project,
        name=config.wandb.name or config.name,
        save_dir=str(run_dir(config)),
        mode=config.wandb.mode,
    )
    fabric = Fabric(
        accelerator=training.accelerator,
        devices=training.devices,
        strategy=training.strategy,
        precision=training.precision,
        loggers=[wandb_logger],
    )
    status = "failed"
    try:
        fabric.launch(_run, request)
        status = "success"
    finally:
        import wandb

        for logger_ in fabric.loggers:
            logger_.finalize(status)
        if wandb.run is not None:
            wandb.finish(exit_code=0 if status == "success" else 1)
    logger.info("Harmonica training finished: {}", run_dir(config))
