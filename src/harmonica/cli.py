"""The ``harmonica`` command: ``prepare``, ``train``, ``infer``, ``eval``, ``export``.

Installed bare, the package transcribes: ``prepare``, ``train``, and ``eval``
need the ``train`` extra as well.
"""

import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from harmonica import evaluate as run_evaluate
from harmonica import infer as run_infer
from harmonica import prepare as run_prepare
from harmonica import train as run_train
from harmonica.config import load_config
from harmonica.requests import (
    DEFAULT_MODEL,
    PRETRAINED,
    HarmonicaEvaluateRequest,
    HarmonicaInferRequest,
    HarmonicaPrepareRequest,
    HarmonicaTrainRequest,
    resolve_model,
)

DEFAULT_CONFIG = Path("config/harmonica.yaml")

app = typer.Typer(
    help="Harmonica: instrument-agnostic music transcription (arXiv:2609.04640).",
    no_args_is_help=True,
)


@app.callback()
def _log_format(ctx: typer.Context) -> None:
    """Log for a reader, not a debugger: no module, function, or line.

    ``infer`` finishes in seconds, so its lines go without a timestamp.
    """
    stamp = "" if ctx.invoked_subcommand == "infer" else "<dim>{time:HH:mm:ss}</dim> "

    def line(record: dict) -> str:
        level = record["level"].name
        if level == "WARNING":
            tag = "<yellow>warning:</yellow> "
        elif record["level"].no >= logger.level("ERROR").no:
            tag = "<red>error:</red> "
        else:
            tag = ""
        return stamp + tag + "{message}\n{exception}"

    logger.remove()
    logger.add(sys.stderr, format=line)


ConfigOption = Annotated[
    Path, typer.Option("--config", help="Pipeline-owned YAML configuration.")
]
OverrideOption = Annotated[
    list[str] | None,
    typer.Option(
        "--override", help="Override a config field as dotted.path=value. Repeatable."
    ),
]
SizeOption = Annotated[
    str | None,
    typer.Option(
        "--size",
        help="nano, small, medium, large, or x-large; overrides model.size.",
    ),
]


@contextmanager
def _train_extra() -> Iterator[None]:
    """Name the extra to install when a training dependency is missing."""
    try:
        yield
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"{error.name} is missing: this command needs the train extra, "
            "e.g. pip install 'harmonica-amt[train]' or uv sync"
        ) from error


def _overrides(override: list[str] | None, size: str | None) -> list[str]:
    overrides = list(override or ())
    if size is not None:
        overrides.append(f"model.size={size}")
    return overrides


@app.command("prepare")
def prepare_harmonica(
    config: ConfigOption = DEFAULT_CONFIG,
    override: OverrideOption = None,
    workers: Annotated[
        int, typer.Option("--workers", help="Processes decoding audio and MIDI.")
    ] = 16,
) -> None:
    """Index every configured corpus and hold out the development set."""
    with _train_extra():
        run_prepare(
            HarmonicaPrepareRequest(
                config=load_config(config, list(override or ())), workers=workers
            )
        )


@app.command("train")
def train_harmonica(
    config: ConfigOption = DEFAULT_CONFIG,
    override: OverrideOption = None,
    size: SizeOption = None,
    devices: Annotated[
        int | None,
        typer.Option("--devices", help="Devices to train on; overrides the config."),
    ] = None,
    max_steps: Annotated[
        str | None,
        typer.Option(
            "--max-steps",
            metavar="COUNT",
            help="Optimizer steps, as digits or 100k/1m; overrides the config.",
        ),
    ] = None,
    resume: Annotated[
        str,
        typer.Option(
            "--resume",
            help="'auto' for the newest checkpoint, 'none', or a checkpoint path.",
        ),
    ] = "auto",
) -> None:
    """Train Harmonica."""
    overrides = _overrides(override, size)
    if devices is not None:
        overrides.append(f"training.devices={devices}")
    if max_steps is not None:
        try:
            overrides.append(f"training.max_steps={parse_count(max_steps)}")
        except ValueError as error:
            raise typer.BadParameter(str(error), param_hint="--max-steps") from error
    with _train_extra():
        run_train(
            HarmonicaTrainRequest(
                config=load_config(config, overrides),
                config_source=config,
                resume=None if resume == "none" else resume,
            )
        )


@app.command("infer")
def infer_harmonica(
    audio: Annotated[
        Path,
        typer.Argument(
            help="Audio file to transcribe: WAV, FLAC, MP3, OGG.",
            exists=True,
            dir_okay=False,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Destination MIDI file; defaults to the input with a .mid suffix.",
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            metavar="[" + "|".join(PRETRAINED) + "|PATH]",
            help="A bundled size, or the path of a checkpoint you trained or exported.",
        ),
    ] = DEFAULT_MODEL,
    device: Annotated[
        str | None,
        typer.Option(
            "--device",
            "-d",
            help="Where to run the model: cpu, cuda, cuda:1, ...; defaults to "
            "the GPU when there is one.",
        ),
    ] = None,
    onset_threshold: Annotated[
        float | None,
        typer.Option(
            "--onset-threshold",
            help="Override the threshold the model's evaluation chose.",
        ),
    ] = None,
    frame_threshold: Annotated[
        float | None,
        typer.Option(
            "--frame-threshold",
            help="Override the threshold the model's evaluation chose.",
        ),
    ] = None,
) -> None:
    """Transcribe a recording of any pitched instrument to MIDI."""
    try:
        resolve_model(model)
    except FileNotFoundError as error:
        raise typer.BadParameter(str(error), param_hint="--model") from None
    if device is not None:
        import torch

        try:
            torch.device(device)
        except RuntimeError as error:
            raise typer.BadParameter(str(error), param_hint="--device") from None
    run_infer(
        HarmonicaInferRequest(
            audio=audio,
            output=output or audio.with_suffix(".mid"),
            model=model,
            device=device,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
        )
    )


@app.command("eval")
def evaluate_harmonica(
    checkpoint: Annotated[
        Path, typer.Option("--checkpoint", help="Trained Harmonica checkpoint.")
    ],
    config: ConfigOption = DEFAULT_CONFIG,
    override: OverrideOption = None,
    size: SizeOption = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Report directory; defaults under the config."),
    ] = None,
    onset_threshold: Annotated[
        float | None,
        typer.Option("--onset-threshold", help="Skip the sweep: use this onset."),
    ] = None,
    frame_threshold: Annotated[
        float | None,
        typer.Option("--frame-threshold", help="Skip the sweep: use this frame."),
    ] = None,
    workers: Annotated[
        str | None,
        typer.Option(
            "--workers",
            help="Processes decoding and scoring, a count or 'all'; overrides "
            "the config.",
        ),
    ] = None,
    devices: Annotated[
        str | None,
        typer.Option(
            "--devices",
            help="GPUs transcribing at once, a count or 'all'; overrides the config.",
        ),
    ] = None,
) -> None:
    """Sweep thresholds on dev, then score every test set as Table 1 does."""
    if (onset_threshold is None) != (frame_threshold is None):
        raise typer.BadParameter(
            "--onset-threshold and --frame-threshold go together, or neither"
        )
    overrides = _overrides(override, size)
    if workers is not None:
        overrides.append(f"evaluation.workers={workers}")
    if devices is not None:
        overrides.append(f"evaluation.devices={devices}")
    with _train_extra():
        run_evaluate(
            HarmonicaEvaluateRequest(
                config=load_config(config, overrides),
                checkpoint=checkpoint,
                output=output,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
            )
        )


@app.command("export")
def export_harmonica(
    checkpoint: Annotated[
        Path, typer.Option("--checkpoint", help="Training checkpoint to export.")
    ],
    output: Annotated[Path, typer.Option("--output", help="Destination weights.")],
    onset_threshold: Annotated[
        float | None,
        typer.Option(
            "--onset-threshold", help="Override the one harmonica eval saved."
        ),
    ] = None,
    frame_threshold: Annotated[
        float | None,
        typer.Option(
            "--frame-threshold", help="Override the one harmonica eval saved."
        ),
    ] = None,
) -> None:
    """Strip a checkpoint to its weights, with the thresholds infer decodes at."""
    from harmonica.inference import export_weights

    export_weights(checkpoint, output, onset_threshold, frame_threshold)


# A magnitude suffix, because 100000 and 1000000 differ by one easily dropped zero.
_COUNT = re.compile(r"^(\d+(?:\.\d+)?)([km])?$", re.IGNORECASE)
_MAGNITUDES = {"k": 1_000, "m": 1_000_000}


def parse_count(value: str) -> int:
    """``100000``, ``100k``, ``1m``, or ``1.5k``; refuse anything not whole."""
    matched = _COUNT.match(value.strip().replace("_", ""))
    if not matched:
        raise ValueError(f"{value!r} is not a count; write e.g. 100000, 100k, or 1m")
    mantissa, magnitude = matched.groups()
    scaled = Decimal(mantissa) * (_MAGNITUDES[magnitude.lower()] if magnitude else 1)
    if scaled != scaled.to_integral_value():
        raise ValueError(f"{value!r} is not a whole count")
    return int(scaled)
