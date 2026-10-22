"""Whole recordings through the model, and the thresholds that decode them.

Appendix D's "half" strategy: the song's CQT is cut into chunks as long as a
training clip that overlap by half, and only the middle half of each chunk is
kept, so every kept frame saw a quarter chunk of context on either side.
"""

import itertools
import json
import multiprocessing
import queue
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from harmonica.config import (
    HarmonicaConfig,
    HarmonicaEvaluationConfig,
    config_from_dict,
    resolve_workers,
)
from harmonica.corpus import Notes, Recording
from harmonica.data import read_audio, resample
from harmonica.decoding import Scores, decode, note_f1, score
from harmonica.model import SILENCE, HarmonicaModel
from harmonica.requests import DEFAULT_MODEL, resolve_model

# 2: the CQT is log10(1 + |X|); version 1 models read log(max(|X|, 1e-5)).
CHECKPOINT_FORMAT_VERSION = 2
PIPELINE_NAME = "harmonica"
# nnAudio's CQT kernels: rebuilt from the config, so exported weights omit them.
CQT_KERNELS = "cqt.cqt."


@dataclass(frozen=True)
class Activations:
    """Probabilities ``[88, T]`` for one recording, and how long it took."""

    onset: np.ndarray
    sustain: np.ndarray
    seconds: float
    audio_seconds: float


@torch.inference_mode()
def transcribe_audio(
    model: HarmonicaModel,
    audio: np.ndarray,
    *,
    chunk_seconds: float,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Onset and sustain probabilities for a whole recording."""
    cqt_config = model.config.cqt
    frames_per_second = cqt_config.sample_rate / cqt_config.hop_length
    # A multiple of four frames, so the kept half and both margins are whole.
    chunk = 4 * max(1, int(round(chunk_seconds * frames_per_second / 4)))
    keep = chunk // 2
    margin = chunk // 4

    waveform = torch.from_numpy(audio).to(device).unsqueeze(0)
    spectrogram = model.spectrogram(waveform)[0]  # [1, F, T]
    total = spectrogram.shape[-1]
    count = max(1, -(-total // keep))
    # Chunk k starts at k * keep and keeps original frames [k * keep, (k+1) * keep).
    # The padding is silence, as a training clip's zero-padded tail reads.
    padded = torch.nn.functional.pad(
        spectrogram, (margin, (count + 1) * keep - margin - total), value=SILENCE
    )
    windows = padded.unfold(-1, chunk, keep)[:, :, :count]  # [1, F, count, chunk]
    windows = windows.permute(2, 0, 1, 3)  # [count, 1, F, chunk]

    onsets, sustains = [], []
    for first in range(0, count, batch_size):
        batch = windows[first : first + batch_size]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model.forward_spectrogram(batch)
        middle = slice(margin, margin + keep)
        onsets.append(torch.sigmoid(logits["onset"][..., middle].float()))
        sustains.append(torch.sigmoid(logits["sustain"][..., middle].float()))

    def stitch(parts: list[torch.Tensor]) -> np.ndarray:
        joined = torch.cat(parts).permute(1, 0, 2).reshape(parts[0].shape[1], -1)
        return joined[:, :total].cpu().numpy()

    return stitch(onsets), stitch(sustains)


class Harmonica:
    """A model loaded once, turning recordings into notes.

    ``model`` is a bundled model's name (``PRETRAINED``) or a checkpoint's
    path. Thresholds left out are the ones the checkpoint's evaluation chose;
    failing those, 0.5. ``device`` defaults to the GPU when there is one.
    """

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        device: str | torch.device | None = None,
        onset_threshold: float | None = None,
        frame_threshold: float | None = None,
        chunk_seconds: float = 5.0,
        batch_size: int = 8,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        path = resolve_model(model)
        stored = read_checkpoint(path)
        self.model, self.config, _ = model_from_checkpoint(stored, path, self.device)
        if onset_threshold is None or frame_threshold is None:
            thresholds = checkpoint_thresholds(stored, path)
            if thresholds is None:
                logger.warning(
                    "{}: no thresholds stored with it or beside it; 0.5 unless "
                    "given. harmonica eval chooses and saves them",
                    path,
                )
                thresholds = {"onset": 0.5, "frame": 0.5}
            if onset_threshold is None:
                onset_threshold = thresholds["onset"]
            if frame_threshold is None:
                frame_threshold = thresholds["frame"]
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.chunk_seconds = chunk_seconds
        self.batch_size = batch_size

    @property
    def sample_rate(self) -> int:
        return self.model.config.cqt.sample_rate

    @property
    def frames_per_second(self) -> float:
        cqt = self.model.config.cqt
        return cqt.sample_rate / cqt.hop_length

    def activations(
        self, audio: str | Path | np.ndarray, sample_rate: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Onset and sustain probabilities ``[88, T]``, at ``frames_per_second``.

        ``audio`` is a file, or samples ``[samples]`` or ``[samples, channels]``
        at ``sample_rate``, which an array must give.
        """
        if isinstance(audio, np.ndarray):
            if sample_rate is None:
                raise ValueError("an audio array needs its sample_rate")
            mono = audio.mean(axis=1) if audio.ndim == 2 else audio
            samples = resample(mono.astype(np.float32), sample_rate, self.sample_rate)
        else:
            samples = read_audio(str(audio), self.sample_rate)
        return transcribe_audio(
            self.model,
            samples,
            chunk_seconds=self.chunk_seconds,
            batch_size=self.batch_size,
            device=self.device,
        )

    def transcribe(
        self, audio: str | Path | np.ndarray, sample_rate: int | None = None
    ) -> Notes:
        """The notes in one recording, as ``activations`` takes it."""
        onset, sustain = self.activations(audio, sample_rate)
        return decode(
            onset,
            sustain,
            self.frames_per_second,
            self.onset_threshold,
            self.frame_threshold,
        )


def run_recordings(
    model: HarmonicaModel,
    recordings: list[Recording],
    evaluation: HarmonicaEvaluationConfig,
    device: torch.device,
) -> list[Activations]:
    """Activations for every recording, timed from file to probabilities."""
    results = []
    for index, recording in enumerate(recordings):
        results.append(_run_recording(model, recording, evaluation, device))
        if (index + 1) % 50 == 0:
            logger.info("  {}/{} recordings", index + 1, len(recordings))
    return results


def run_recordings_on(
    replicas: list[tuple[HarmonicaModel, torch.device]],
    recordings: list[Recording],
    evaluation: HarmonicaEvaluationConfig,
) -> list[Activations]:
    """``run_recordings`` with one thread per device, in ``recordings``' order.

    The threads take recordings off one queue, longest first, so a device that
    finishes early picks up more and none is left with a long tail.
    """
    if len(replicas) == 1:
        model, device = replicas[0]
        return run_recordings(model, recordings, evaluation, device)
    pending: queue.SimpleQueue[int] = queue.SimpleQueue()
    for index in sorted(range(len(recordings)), key=lambda i: -recordings[i].duration):
        pending.put(index)
    results: list[Activations | None] = [None] * len(recordings)
    done = itertools.count(1)

    def work(model: HarmonicaModel, device: torch.device) -> None:
        while True:
            try:
                index = pending.get_nowait()
            except queue.Empty:
                return
            results[index] = _run_recording(
                model, recordings[index], evaluation, device
            )
            finished = next(done)
            if finished % 50 == 0:
                logger.info("  {}/{} recordings", finished, len(recordings))

    with ThreadPoolExecutor(len(replicas)) as executor:
        for future in [executor.submit(work, *replica) for replica in replicas]:
            future.result()
    return results  # type: ignore[return-value]


def _run_recording(
    model: HarmonicaModel,
    recording: Recording,
    evaluation: HarmonicaEvaluationConfig,
    device: torch.device,
) -> Activations:
    sample_rate = model.config.cqt.sample_rate
    audio = read_audio(recording.audio, sample_rate)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    began = time.perf_counter()
    onset, sustain = transcribe_audio(
        model,
        audio,
        chunk_seconds=evaluation.chunk_seconds,
        batch_size=evaluation.batch_size,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - began
    return Activations(onset, sustain, elapsed, len(audio) / sample_rate)


# The activations a scoring pool's workers decode. Set before the pool forks,
# so every worker inherits them instead of being sent them: one test set's
# probabilities run to gigabytes, and the sweep decodes them once a threshold.
_POOL_ACTIVATIONS: list[Activations] = []


class _Scorer:
    """``map`` over recordings, in forked workers or, with one, in this process."""

    def __init__(self, pool: Pool | None) -> None:
        self.pool = pool

    def map(self, function, *iterables) -> list:
        if self.pool is None:
            return list(map(function, *iterables))
        return self.pool.starmap(function, zip(*iterables, strict=True))

    def decode(
        self, count: int, frames_per_second: float, onset: float, frame: float
    ) -> list[Notes]:
        job = partial(_decode_pooled, frames_per_second, onset, frame)
        return self.map(job, range(count))


def _decode_pooled(
    frames_per_second: float, onset: float, frame: float, index: int
) -> Notes:
    result = _POOL_ACTIVATIONS[index]
    return decode(result.onset, result.sustain, frames_per_second, onset, frame)


@contextmanager
def scoring_pool(
    activations: list[Activations], evaluation: HarmonicaEvaluationConfig
) -> Iterator[_Scorer]:
    """``evaluation.workers`` processes decoding and scoring these activations.

    Forked, and they never touch this process's CUDA context.
    """
    global _POOL_ACTIVATIONS
    workers = min(resolve_workers(evaluation.workers), len(activations))
    _POOL_ACTIVATIONS = activations
    try:
        if workers <= 1:
            yield _Scorer(None)
            return
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            yield _Scorer(pool)
    finally:
        _POOL_ACTIVATIONS = []


def score_all(
    recordings: list[Recording],
    activations: list[Activations],
    evaluation: HarmonicaEvaluationConfig,
    frames_per_second: float,
    onset_threshold: float,
    frame_threshold: float,
) -> list[Scores]:
    references = [Notes.load(recording.notes) for recording in recordings]
    scorer_job = partial(
        score,
        onset_tolerance=evaluation.onset_tolerance,
        offset_ratio=evaluation.offset_ratio,
        offset_min_tolerance=evaluation.offset_min_tolerance,
        frame_seconds=evaluation.frame_seconds,
    )
    with scoring_pool(activations, evaluation) as scorer:
        estimates = scorer.decode(
            len(activations), frames_per_second, onset_threshold, frame_threshold
        )
        return scorer.map(scorer_job, references, estimates)


def threshold_grid(evaluation: HarmonicaEvaluationConfig) -> list[float]:
    grid = evaluation.threshold_grid
    count = int(round((grid.stop - grid.start) / grid.step)) + 1
    return [round(grid.start + i * grid.step, 6) for i in range(count)]


def sweep_thresholds(
    recordings: list[Recording],
    activations: list[Activations],
    evaluation: HarmonicaEvaluationConfig,
    frames_per_second: float,
) -> tuple[float, float, list[dict]]:
    """Appendix D's two stages: onset for OnP, then frame for OnPOff.

    Each stage scores only the F1 it maximises, and skips a threshold outright
    when even a perfect match of its notes could not beat the best so far:
    F1 is at most ``2 min(r, e) / (r + e)`` for ``r`` reference and ``e``
    estimated notes, and decoding is cheap where mir_eval's matching is not.
    A low threshold on a weak model decodes tens of thousands of notes, which
    is exactly the case the bound rules out. The grid is walked from the top,
    so the sparse, cheap thresholds set the bar first.
    """
    grid = sorted(threshold_grid(evaluation), reverse=True)
    references = [Notes.load(r.notes) for r in recordings]
    trace: list[dict] = []

    def stage(
        scorer: _Scorer,
        name: str,
        thresholds: list[tuple[float, float]],
        offsets: bool,
    ):
        f1 = partial(
            note_f1,
            onset_tolerance=evaluation.onset_tolerance,
            offset_ratio=evaluation.offset_ratio if offsets else None,
            offset_min_tolerance=evaluation.offset_min_tolerance,
        )
        best, best_value = thresholds[0], -1.0
        for on, frame in thresholds:
            estimates = scorer.decode(len(activations), frames_per_second, on, frame)
            bound = float(
                np.mean(
                    [
                        2 * min(len(r), len(e)) / (len(r) + len(e))
                        if len(r) + len(e)
                        else 1.0
                        for r, e in zip(references, estimates, strict=True)
                    ]
                )
            )
            if bound <= best_value:
                trace.append({"onset": on, "frame": frame, name: None, "bound": bound})
                continue
            value = float(np.mean(scorer.map(f1, references, estimates)))
            trace.append({"onset": on, "frame": frame, name: value})
            logger.info(
                "  onset {:.2f} frame {:.2f}: {} {:.3f}", on, frame, name, value
            )
            if value > best_value:
                best, best_value = (on, frame), value
        return best

    with scoring_pool(activations, evaluation) as scorer:
        # OnP does not read the frame threshold, so the first stage holds it at 0.5.
        best_onset, _ = stage(scorer, "OnP", [(on, 0.5) for on in grid], offsets=False)
        _, best_frame = stage(
            scorer, "OnPOff", [(best_onset, f) for f in grid], offsets=True
        )
    return best_onset, best_frame, trace


def read_checkpoint(path: Path) -> dict:
    """A checkpoint's contents, refused unless it is Harmonica's current format."""
    stored = torch.load(path, map_location="cpu", weights_only=False)
    if stored.get("pipeline") != PIPELINE_NAME:
        raise ValueError(f"{path}: not a Harmonica checkpoint")
    if stored.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"{path}: checkpoint format {stored.get('checkpoint_format_version')}, "
            f"expected {CHECKPOINT_FORMAT_VERSION}"
        )
    return stored


def thresholds_path(checkpoint: Path) -> Path:
    """Where ``harmonica eval`` leaves a checkpoint's thresholds: beside it.

    Symlinks are followed, so ``checkpoint_latest`` finds its target's.
    """
    return checkpoint.resolve().with_suffix(".thresholds.json")


def save_thresholds(checkpoint: Path, onset: float, frame: float) -> Path:
    path = thresholds_path(checkpoint)
    path.write_text(json.dumps({"onset": onset, "frame": frame}) + "\n")
    return path


def checkpoint_thresholds(stored: dict, path: Path) -> dict | None:
    """``{"onset", "frame"}`` the weights carry, else those saved beside them."""
    if "thresholds" in stored:
        return stored["thresholds"]
    beside = thresholds_path(path)
    if beside.is_file():
        return json.loads(beside.read_text())
    return None


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[HarmonicaModel, HarmonicaConfig, int]:
    """The model a checkpoint holds, the config it trained under, and its step."""
    return model_from_checkpoint(read_checkpoint(path), path, device)


def model_from_checkpoint(
    stored: dict, path: Path, device: torch.device
) -> tuple[HarmonicaModel, HarmonicaConfig, int]:
    """``load_checkpoint`` for contents already read.

    Exported weights leave out nnAudio's CQT kernels, which every model builds
    for itself from the config; every other tensor must be there.
    """
    config = config_from_dict(stored["config"], path)
    model = HarmonicaModel(config.model)
    state = {
        key.removeprefix("_orig_mod."): value for key, value in stored["model"].items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if not key.startswith(CQT_KERNELS)]
    if missing or unexpected:
        raise ValueError(
            f"{path}: weights do not fit the model; missing {missing}, "
            f"unexpected {unexpected}"
        )
    return model.to(device).eval(), config, int(stored["step"])


def export_weights(
    checkpoint: Path,
    output: Path,
    onset_threshold: float | None = None,
    frame_threshold: float | None = None,
) -> None:
    """A training checkpoint as weights alone, with the thresholds to decode at.

    No optimizer, no scheduler, and no CQT kernels: a medium model comes to
    2.6 MB from 77, small enough to ship inside the package. Thresholds left
    out are the ones ``harmonica eval`` saved beside the checkpoint.
    """
    stored = read_checkpoint(checkpoint)
    if onset_threshold is None or frame_threshold is None:
        saved = checkpoint_thresholds(stored, checkpoint)
        if saved is None:
            raise ValueError(
                f"{checkpoint}: no thresholds saved beside it; run harmonica "
                "eval on it first, or give both"
            )
        if onset_threshold is None:
            onset_threshold = saved["onset"]
        if frame_threshold is None:
            frame_threshold = saved["frame"]
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pipeline": PIPELINE_NAME,
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "config": stored["config"],
            "model": {
                key.removeprefix("_orig_mod."): value
                for key, value in stored["model"].items()
                if not key.removeprefix("_orig_mod.").startswith(CQT_KERNELS)
            },
            "step": stored["step"],
            "thresholds": {"onset": onset_threshold, "frame": frame_threshold},
        },
        output,
    )
