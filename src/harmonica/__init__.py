"""Public interface for Harmonica.

A reproduction of *Harmonica: Accurate and Lightweight Instrument-Agnostic
Music Transcription* (Ou et al., arXiv:2609.04640): a frame-level transcriber
built on multi-depth harmonic convolution. Heavy modules load only when a
command runs.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from harmonica.config import (
    SIZES,
    SOURCES,
    HarmonicaConfig,
    HarmonicaModelConfig,
    load_config,
)
from harmonica.requests import (
    HarmonicaEvaluateRequest,
    HarmonicaInferRequest,
    HarmonicaPrepareRequest,
    HarmonicaTrainRequest,
)

if TYPE_CHECKING:
    from harmonica.corpus import Notes
    from harmonica.inference import Harmonica


def __getattr__(name: str):
    # Harmonica needs torch, which only a command or a caller should load.
    if name == "Harmonica":
        from harmonica.inference import Harmonica

        return Harmonica
    raise AttributeError(f"module 'harmonica' has no attribute {name!r}")


def prepare(request: HarmonicaPrepareRequest) -> Path:
    from harmonica.corpus import prepare as run

    return run(request.config, request.workers)


def train(request: HarmonicaTrainRequest) -> None:
    from harmonica.training import train as run

    run(request)


def infer(request: HarmonicaInferRequest) -> "Notes":
    from harmonica.evaluate import infer as run

    return run(request)


def evaluate(request: HarmonicaEvaluateRequest) -> Path:
    from harmonica.evaluate import evaluate as run

    return run(request)


__all__ = [
    "SIZES",
    "SOURCES",
    "Harmonica",
    "HarmonicaConfig",
    "HarmonicaEvaluateRequest",
    "HarmonicaInferRequest",
    "HarmonicaModelConfig",
    "HarmonicaPrepareRequest",
    "HarmonicaTrainRequest",
    "evaluate",
    "infer",
    "load_config",
    "prepare",
    "train",
]
