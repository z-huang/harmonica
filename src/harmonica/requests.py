"""Typed orchestration requests for the Harmonica pipeline."""

from dataclasses import dataclass
from pathlib import Path

from harmonica.config import SIZES, HarmonicaConfig

# One slot per row of Table 1, each `harmonica export`ed with the thresholds
# its development-set sweep chose. A size is bundled once its file is here;
# all five are.
WEIGHTS = Path(__file__).with_name("weights")
PRETRAINED = {size: WEIGHTS / f"harmonica-{size}.pt" for size in SIZES}
DEFAULT_MODEL = "medium"


def bundled_models() -> list[str]:
    """The sizes whose weights ship with this package."""
    return [size for size, path in PRETRAINED.items() if path.is_file()]


def resolve_model(model: str | Path) -> Path:
    """A size from Table 1, or the path of a checkpoint, as a path."""
    if isinstance(model, str) and model in PRETRAINED:
        path = PRETRAINED[model]
        if not path.is_file():
            raise FileNotFoundError(
                f"{model}: not trained yet; bundled are "
                f"{', '.join(bundled_models())}, or give a checkpoint's path"
            )
        return path
    path = Path(model)
    if not path.is_file():
        raise FileNotFoundError(
            f"{model}: neither a checkpoint nor a size ({', '.join(PRETRAINED)})"
        )
    return path


@dataclass(frozen=True)
class HarmonicaPrepareRequest:
    """Build the manifest, note caches, and development split."""

    config: HarmonicaConfig
    workers: int


@dataclass(frozen=True)
class HarmonicaTrainRequest:
    config: HarmonicaConfig
    config_source: Path
    resume: str | Path | None = "auto"


@dataclass(frozen=True)
class HarmonicaInferRequest:
    """Transcribe one audio file to MIDI.

    ``model`` is a bundled model's name or a checkpoint's path. Thresholds left
    out are the ones its evaluation chose. ``device`` (``cpu``, ``cuda:1``, ...)
    defaults to the GPU when there is one.
    """

    audio: Path
    output: Path
    model: str | Path = DEFAULT_MODEL
    device: str | None = None
    onset_threshold: float | None = None
    frame_threshold: float | None = None
    chunk_seconds: float = 5.0
    batch_size: int = 32


@dataclass(frozen=True)
class HarmonicaEvaluateRequest:
    """Sweep thresholds on the development set, then score every test set.

    Thresholds given here skip the sweep, which is how a run is scored at the
    thresholds another run chose.
    """

    config: HarmonicaConfig
    checkpoint: Path
    output: Path | None = None
    onset_threshold: float | None = None
    frame_threshold: float | None = None
