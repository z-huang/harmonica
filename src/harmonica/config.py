"""Typed configuration for the Harmonica pipeline.

Harmonica is a frame-level transcriber, so nothing here is about tokens: the
data block names the recordings, the model block names a size from the paper's
table, and evaluation carries the paper's metric settings.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

ALL_CPUS = "all"
ALL_DEVICES = "all"

# The paper's pitch axis: the 88 piano keys, one output row each.
MIN_PITCH = 21
MAX_PITCH = 108
NUM_PITCHES = MAX_PITCH - MIN_PITCH + 1

# Every corpus the paper uses; a source left at null is skipped, and said so.
SOURCES = ("maestro", "maps", "guitarset", "gaps", "egdb", "goat", "urmp", "slakh")

# Section 4.1, "From nano to x-large, we set (c2, l) to ...", with c1 = c2 / 4.
SIZES: dict[str, tuple[int, int]] = {
    "nano": (16, 2),
    "small": (32, 3),
    "medium": (64, 4),
    "large": (128, 5),
    "x-large": (256, 6),
}

# Table 2's rows: what aggregates across harmonics, and where.
HARMONIC_KINDS = ("shift_aggregate", "hdconv", "stacking", "none")
HARMONIC_DEPTHS = ("multi", "single")


@dataclass
class HarmonicaSourcesConfig:
    """Where each corpus lives; null leaves it out of every split."""

    maestro: str | None = None
    maps: str | None = None
    guitarset: str | None = None
    gaps: str | None = None
    egdb: str | None = None
    goat: str | None = None
    urmp: str | None = None
    slakh: str | None = None


@dataclass
class HarmonicaDataConfig:
    """The corpus, its splits, and how training clips are drawn."""

    sources: HarmonicaSourcesConfig = field(default_factory=HarmonicaSourcesConfig)
    # Manifest, note arrays, and the 16 kHz audio extracted from zips.
    cache_dir: str = "data/cache"
    # Copy every recording not already 16 kHz mono into cache_dir as 16 kHz
    # FLAC, so training clips are read from there.
    cache_audio: bool = True
    # MAESTRO and MAPS hold notes down while the sustain pedal is.
    sustain_pedal: bool = True
    # Source i is drawn with probability proportional to (hours_i / total)^alpha.
    temperature: float = 0.5
    clip_seconds: float = 5.0
    # Held out of training under the same temperature sampling (Table 3: 4.3 h).
    dev_hours: float = 4.33
    dev_seed: int = 1234
    num_workers: int | str = 8


@dataclass
class HarmonicaCqtConfig:
    """The fixed transform: 16 kHz, three bins a semitone, 32 ms frames."""

    sample_rate: int = 16000
    hop_length: int = 512
    bins_per_semitone: int = 3


@dataclass
class HarmonicaHarmonicConfig:
    """How the trunk aggregates across harmonics (Table 2)."""

    kind: str = "shift_aggregate"
    depth: str = "multi"
    # Harmonics 1..n and subharmonics 2..n; 8 is the paper's 7 + 7.
    harmonics: int = 8


@dataclass
class HarmonicaHeadConfig:
    """One frequency-grouped LSTM per target, shared over pitches.

    Bidirectional, half the trunk's width a direction.
    """

    # Per direction; null is the trunk's width, halved when bidirectional.
    hidden_size: int | None = None
    bidirectional: bool = True


@dataclass
class HarmonicaModelConfig:
    """Size preset, optionally overridden one width at a time."""

    size: str = "medium"
    # c2; null takes the size's.
    channels: int | None = None
    # l; null takes the size's.
    layers: int | None = None
    # c1; null is channels / 4.
    frontend_channels: int | None = None
    cqt: HarmonicaCqtConfig = field(default_factory=HarmonicaCqtConfig)
    harmonic: HarmonicaHarmonicConfig = field(default_factory=HarmonicaHarmonicConfig)
    head: HarmonicaHeadConfig = field(default_factory=HarmonicaHeadConfig)


@dataclass
class HarmonicaTrainingConfig:
    output_dir: str = "outputs"
    seed: int = 1234
    accelerator: str = "auto"
    devices: int = 1
    strategy: str = "auto"
    precision: str = "bf16-mixed"
    # torch.compile the model before training: the same function, faster.
    compile: bool = True
    batch_size: int = 8
    max_steps: int = 100_000
    learning_rate: float = 2e-4
    min_learning_rate: float = 0.0
    warmup_steps: int = 0
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float | None = None
    log_freq: int = 50
    # Dev-set notes at fixed thresholds, every val_freq steps.
    val_freq: int = 5000
    val_onset_threshold: float = 0.5
    val_frame_threshold: float = 0.5
    save_freq: int = 5000


@dataclass
class HarmonicaThresholdGridConfig:
    """Appendix D: both thresholds over [0.1, 0.85] in steps of 0.05."""

    start: float = 0.1
    stop: float = 0.85
    step: float = 0.05


@dataclass
class HarmonicaEvaluationConfig:
    output_dir: str = "outputs/evaluations"
    test_sets: list[str] = field(default_factory=lambda: list(SOURCES))
    # Appendix D's "half" strategy: chunks this long, half of each kept.
    chunk_seconds: float = 5.0
    batch_size: int = 32
    threshold_grid: HarmonicaThresholdGridConfig = field(
        default_factory=HarmonicaThresholdGridConfig
    )
    # Frame F1 is read off the decoded notes at this frame size, for every model.
    frame_seconds: float = 0.025
    onset_tolerance: float = 0.05
    offset_ratio: float = 0.2
    offset_min_tolerance: float = 0.05
    max_recordings: int | None = None
    # Processes decoding and scoring recordings in the sweep and on each test
    # set: a count, or "all". mir_eval is single-threaded, and the sweep scores
    # every development recording once per threshold.
    workers: int | str = 16
    write_midi: bool = False
    accelerator: str = "auto"
    # GPUs transcribing recordings at once, one model on each: a count, or
    # "all". Decoding and scoring stay with ``workers``.
    devices: int | str = 1


@dataclass
class HarmonicaWandbConfig:
    project: str = "harmonica"
    name: str | None = None
    mode: str = "online"


@dataclass
class HarmonicaConfig:
    """Complete configuration for one Harmonica run.

    ``name`` is the directory every output is filed under; the checked-in
    config spells it from ``model.size``, so a size override renames the run
    with it.
    """

    name: str = "harmonica"
    data: HarmonicaDataConfig = field(default_factory=HarmonicaDataConfig)
    model: HarmonicaModelConfig = field(default_factory=HarmonicaModelConfig)
    training: HarmonicaTrainingConfig = field(default_factory=HarmonicaTrainingConfig)
    evaluation: HarmonicaEvaluationConfig = field(
        default_factory=HarmonicaEvaluationConfig
    )
    wandb: HarmonicaWandbConfig = field(default_factory=HarmonicaWandbConfig)


@dataclass(frozen=True)
class ResolvedWidths:
    """The three numbers a size stands for, after any override."""

    channels: int
    layers: int
    frontend_channels: int
    head_hidden: int


def resolve_widths(model: HarmonicaModelConfig) -> ResolvedWidths:
    """Read ``model.size`` and apply whichever widths the config overrides."""
    channels, layers = SIZES[model.size]
    if model.channels is not None:
        channels = model.channels
    if model.layers is not None:
        layers = model.layers
    frontend = (
        model.frontend_channels
        if model.frontend_channels is not None
        else max(1, channels // 4)
    )
    if model.head.hidden_size is not None:
        hidden = model.head.hidden_size
    else:
        hidden = max(1, channels // 2) if model.head.bidirectional else channels
    return ResolvedWidths(channels, layers, frontend, hidden)


def _positive(value: float, name: str, source: str | Path) -> None:
    if not value > 0:
        raise ValueError(f"{source}: {name} must be positive, got {value!r}")


def validate_config(config: HarmonicaConfig, source: str | Path) -> None:
    """Reject settings the pipeline cannot run, naming the field."""
    if not config.name or os.sep in config.name or config.name in (".", ".."):
        raise ValueError(f"{source}: name must be one path component")

    model = config.model
    if model.size not in SIZES:
        raise ValueError(
            f"{source}: model.size must be one of {', '.join(SIZES)}, "
            f"got {model.size!r}"
        )
    for name in ("channels", "layers", "frontend_channels"):
        value = getattr(model, name)
        if value is not None:
            _positive(value, f"model.{name}", source)
    if model.head.hidden_size is not None:
        _positive(model.head.hidden_size, "model.head.hidden_size", source)
    if model.harmonic.kind not in HARMONIC_KINDS:
        raise ValueError(
            f"{source}: model.harmonic.kind must be one of "
            f"{', '.join(HARMONIC_KINDS)}, got {model.harmonic.kind!r}"
        )
    if model.harmonic.depth not in HARMONIC_DEPTHS:
        raise ValueError(
            f"{source}: model.harmonic.depth must be one of "
            f"{', '.join(HARMONIC_DEPTHS)}, got {model.harmonic.depth!r}"
        )
    if model.harmonic.harmonics < 2:
        raise ValueError(f"{source}: model.harmonic.harmonics must be at least 2")
    cqt = model.cqt
    for name in ("sample_rate", "hop_length", "bins_per_semitone"):
        _positive(getattr(cqt, name), f"model.cqt.{name}", source)
    if cqt.bins_per_semitone % 2 == 0:
        # The strided conv keeps each semitone's middle bin at its centre.
        raise ValueError(f"{source}: model.cqt.bins_per_semitone must be odd")

    data = config.data
    _positive(data.clip_seconds, "data.clip_seconds", source)
    _positive(data.temperature, "data.temperature", source)
    if data.dev_hours < 0:
        raise ValueError(f"{source}: data.dev_hours must not be negative")
    if isinstance(data.num_workers, str) and data.num_workers != ALL_CPUS:
        raise ValueError(f"{source}: data.num_workers must be a count or {ALL_CPUS!r}")

    training = config.training
    for name in ("batch_size", "max_steps", "learning_rate", "log_freq", "save_freq"):
        _positive(getattr(training, name), f"training.{name}", source)
    _positive(training.val_freq, "training.val_freq", source)
    if training.batch_size % training.devices:
        raise ValueError(
            f"{source}: training.batch_size ({training.batch_size}) must divide "
            f"by training.devices ({training.devices})"
        )

    evaluation = config.evaluation
    unknown = [name for name in evaluation.test_sets if name not in SOURCES]
    if unknown:
        raise ValueError(
            f"{source}: evaluation.test_sets names unknown sources {unknown}; "
            f"choose from {', '.join(SOURCES)}"
        )
    grid = evaluation.threshold_grid
    if not 0 < grid.start <= grid.stop < 1 or not grid.step > 0:
        raise ValueError(f"{source}: evaluation.threshold_grid must lie in (0, 1)")
    for name in ("chunk_seconds", "batch_size", "frame_seconds"):
        _positive(getattr(evaluation, name), f"evaluation.{name}", source)
    if isinstance(evaluation.workers, str):
        if evaluation.workers != ALL_CPUS:
            raise ValueError(
                f"{source}: evaluation.workers must be a count or {ALL_CPUS!r}"
            )
    else:
        _positive(evaluation.workers, "evaluation.workers", source)
    if isinstance(evaluation.devices, str):
        if evaluation.devices != ALL_DEVICES:
            raise ValueError(
                f"{source}: evaluation.devices must be a count or {ALL_DEVICES!r}"
            )
    else:
        _positive(evaluation.devices, "evaluation.devices", source)


def load_config(
    config_path: str | Path, overrides: Sequence[str] = ()
) -> HarmonicaConfig:
    """Load, override, resolve, and validate one Harmonica configuration."""
    path = Path(config_path)
    try:
        merged = OmegaConf.merge(
            OmegaConf.structured(HarmonicaConfig), OmegaConf.load(path)
        )
        for override in overrides:
            key, separator, raw = override.partition("=")
            if not separator:
                raise ValueError(
                    f"{path}: override {override!r} is not a dotted.path=value pair"
                )
            OmegaConf.update(merged, key, OmegaConf.create(f"value: {raw}").value)
        OmegaConf.resolve(merged)
        config = OmegaConf.to_object(merged)
    except OmegaConfBaseException as error:
        raise ValueError(f"{path}: invalid Harmonica config: {error}") from error
    if not isinstance(config, HarmonicaConfig):
        raise TypeError(f"{path}: OmegaConf did not create HarmonicaConfig")
    validate_config(config, path)
    return config


def config_from_dict(stored: dict, source: str | Path) -> HarmonicaConfig:
    """Rebuild the config a checkpoint was trained under."""
    merged = OmegaConf.merge(OmegaConf.structured(HarmonicaConfig), stored)
    config = OmegaConf.to_object(merged)
    assert isinstance(config, HarmonicaConfig)
    validate_config(config, source)
    return config


def config_to_dict(config: HarmonicaConfig) -> dict:
    container = OmegaConf.to_container(OmegaConf.structured(config))
    assert isinstance(container, dict)
    return container


def run_dir(config: HarmonicaConfig) -> Path:
    """Where this run files its checkpoints."""
    return Path(config.training.output_dir) / config.name


def evaluation_dir(config: HarmonicaConfig) -> Path:
    """Where this run files its evaluation reports."""
    return Path(config.evaluation.output_dir) / config.name


def resolve_num_workers(config: HarmonicaConfig) -> int:
    return resolve_workers(config.data.num_workers)


def resolve_workers(workers: int | str) -> int:
    """A worker count, with "all" meaning every CPU this process may use."""
    if not isinstance(workers, str):
        return workers
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1
