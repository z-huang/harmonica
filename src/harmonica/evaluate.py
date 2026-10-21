"""The paper's benchmark: sweep thresholds on dev, then score each test set.

Writes ``report.json`` (thresholds, the sweep, per-set and per-recording scores,
xRT) and ``report.md`` (Table 1's row for this model) under
``evaluation.output_dir / name / <checkpoint stem>``. A full sweep also saves
its thresholds beside the checkpoint, ``<checkpoint>.thresholds.json``, where
``infer`` and ``export`` read them.
"""

import json
import os
from pathlib import Path

import torch
from loguru import logger

from harmonica.config import (
    ALL_DEVICES,
    SOURCES,
    HarmonicaEvaluationConfig,
    evaluation_dir,
)
from harmonica.corpus import (
    DEV,
    TEST,
    Notes,
    load_manifest,
)
from harmonica.decoding import decode, mean_scores
from harmonica.inference import (
    Harmonica,
    load_checkpoint,
    run_recordings_on,
    save_thresholds,
    score_all,
    sweep_thresholds,
)
from harmonica.model import count_parameters
from harmonica.requests import (
    HarmonicaEvaluateRequest,
    HarmonicaInferRequest,
)


def _device(accelerator: str) -> torch.device:
    if accelerator in ("auto", "gpu", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _devices(evaluation: HarmonicaEvaluationConfig) -> list[torch.device]:
    """``evaluation.devices`` GPUs, or the CPU alone."""
    if _device(evaluation.accelerator).type == "cpu":
        return [torch.device("cpu")]
    available = torch.cuda.device_count()
    count = available if evaluation.devices == ALL_DEVICES else evaluation.devices
    if count > available:
        raise ValueError(
            f"evaluation.devices is {count}, but only {available} GPUs are visible"
        )
    return [torch.device("cuda", index) for index in range(count)]


def evaluate(request: HarmonicaEvaluateRequest) -> Path:
    config = request.config
    evaluation = config.evaluation
    given = request.onset_threshold is not None and request.frame_threshold is not None
    devices = _devices(evaluation)
    loaded = [load_checkpoint(request.checkpoint, device) for device in devices]
    model, trained, step = loaded[0]
    replicas = [(m, device) for (m, _, _), device in zip(loaded, devices, strict=True)]
    logger.info("Transcribing on {}", ", ".join(map(str, devices)))
    if trained.model != config.model:
        raise ValueError(
            f"{request.checkpoint}: trained with a different model block than "
            "the config given; evaluate with the run's own config"
        )
    cqt = config.model.cqt
    frames_per_second = cqt.sample_rate / cqt.hop_length
    out = request.output or (evaluation_dir(config) / request.checkpoint.stem)
    out.mkdir(parents=True, exist_ok=True)
    splits = load_manifest(config)

    def capped(recordings):
        limit = evaluation.max_recordings
        return recordings if limit is None else recordings[:limit]

    report: dict = {
        "checkpoint": str(request.checkpoint),
        "step": step,
        "parameters": count_parameters(model),
        "complete": False,
    }

    def write() -> None:
        # After every set, so a run cut short keeps the sets it finished.
        report["xRT"] = xrt(report["test"])
        write_report(out, report, config.name)

    if given:
        onset_threshold = request.onset_threshold
        frame_threshold = request.frame_threshold
        report["sweep"] = None
    dev = capped(splits[DEV])
    activations = run_recordings_on(replicas, dev, evaluation)
    if not given:
        logger.info("Sweeping thresholds on {} development recordings", len(dev))
        onset_threshold, frame_threshold, trace = sweep_thresholds(
            dev, activations, evaluation, frames_per_second
        )
        report["sweep"] = trace
        if evaluation.max_recordings is None:
            saved = save_thresholds(
                request.checkpoint, onset_threshold, frame_threshold
            )
            logger.info("Thresholds saved to {}", saved)
    # Table 2's numbers, at the thresholds the test sets are scored at.
    dev_scores = mean_scores(
        score_all(
            dev,
            activations,
            evaluation,
            frames_per_second,
            onset_threshold,
            frame_threshold,
        )
    )
    report["dev"] = dev_scores.as_dict()
    logger.info(
        "dev: OnP {:.3f} OnPOff {:.3f} Frm {:.3f}",
        dev_scores.onset,
        dev_scores.onset_offset,
        dev_scores.frame,
    )
    del activations
    report["thresholds"] = {"onset": onset_threshold, "frame": frame_threshold}
    logger.info("Thresholds: onset {} frame {}", onset_threshold, frame_threshold)

    report["test"] = {}
    for source in evaluation.test_sets:
        recordings = capped([r for r in splits[TEST] if r.source == source])
        if not recordings:
            logger.warning("{}: no test recordings prepared; skipped", source)
            continue
        logger.info("{}: {} test recordings", source, len(recordings))
        activations = run_recordings_on(replicas, recordings, evaluation)
        scores = score_all(
            recordings,
            activations,
            evaluation,
            frames_per_second,
            onset_threshold,
            frame_threshold,
        )
        mean = mean_scores(scores)
        logger.info(
            "{}: OnP {:.3f} OnPOff {:.3f} Frm {:.3f}",
            source,
            mean.onset,
            mean.onset_offset,
            mean.frame,
        )
        report["test"][source] = {
            **mean.as_dict(),
            "seconds": sum(a.seconds for a in activations),
            "audio_seconds": sum(a.audio_seconds for a in activations),
            "recordings": {
                r.key: s.as_dict() for r, s in zip(recordings, scores, strict=True)
            },
        }
        if evaluation.write_midi:
            directory = out / "midi" / source
            directory.mkdir(parents=True, exist_ok=True)
            for recording, result in zip(recordings, activations, strict=True):
                notes = decode(
                    result.onset,
                    result.sustain,
                    frames_per_second,
                    onset_threshold,
                    frame_threshold,
                )
                notes.save_midi(directory / f"{recording.key}.mid")
        write()
    report["complete"] = True
    write()
    logger.info("Wrote {}", out / "report.json")
    return out


def xrt(test: dict) -> float | None:
    """Audio seconds per second of CQT and network, over every set scored."""
    seconds = sum(row["seconds"] for row in test.values())
    audio_seconds = sum(row["audio_seconds"] for row in test.values())
    return audio_seconds / seconds if seconds else None


def write_report(out: Path, report: dict, name: str) -> None:
    # Through a temporary file, so a reader never sees half a report.
    for filename, text in (
        ("report.json", json.dumps(report, indent=1)),
        ("report.md", markdown_row(report, name)),
    ):
        partial = out / f".{filename}.partial"
        partial.write_text(text)
        os.replace(partial, out / filename)


def markdown_row(report: dict, name: str) -> str:
    """Table 1's columns: OnP and Frm per set; OnPOff in the second table."""
    sets = [s for s in SOURCES if s in report["test"]]
    header = (
        "| Model | #Param | xRT | "
        + " | ".join(f"{s} OnP | {s} OnPOff | {s} Frm" for s in sets)
        + " |"
    )
    rule = "|" + "---|" * (3 + 3 * len(sets))
    xrt = f"{report['xRT']:.1f}x" if report["xRT"] else "-"
    cells = [name, f"{report['parameters']:,}", xrt]
    for s in sets:
        row = report["test"][s]
        cells += [f"{row['OnP']:.3f}", f"{row['OnPOff']:.3f}", f"{row['Frm']:.3f}"]
    return "\n".join([header, rule, "| " + " | ".join(cells) + " |", ""])


def infer(request: HarmonicaInferRequest) -> Notes:
    """Transcribe one audio file to MIDI."""
    model = Harmonica(
        request.model,
        device=request.device,
        onset_threshold=request.onset_threshold,
        frame_threshold=request.frame_threshold,
        chunk_seconds=request.chunk_seconds,
        batch_size=request.batch_size,
    )
    notes = model.transcribe(request.audio)
    notes.save_midi(request.output)
    logger.info("Wrote {} notes to {}", len(notes), request.output)
    return notes
