"""Training clips: temperature-sampled sources, random 5 s windows, frame labels.

Following MT3 as the paper does, each clip first draws a source with
probability proportional to ``(hours_i / total) ** alpha``, then a recording
from it uniformly, then a window uniformly within that recording. Labels are
built on the CQT's own frame grid, so frame ``t`` of a label is the frame the
model's CQT centres at ``t * hop / sr``.
"""

import math
import random
from collections.abc import Iterator
from functools import lru_cache

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import IterableDataset, get_worker_info

from harmonica.config import MIN_PITCH, NUM_PITCHES
from harmonica.corpus import Notes, Recording, source_weights
from harmonica.model import num_frames


def read_audio(
    path: str, sample_rate: int, start: float = 0.0, duration: float | None = None
) -> np.ndarray:
    """Mono float32 at ``sample_rate``, from ``start`` for ``duration`` seconds.

    Seeks in the file's own rate and resamples only the window read, so a
    44.1 kHz Slakh stem never has to be decoded whole for one clip. A window
    running past the end is zero-padded to its full length.
    """
    with sf.SoundFile(path) as handle:
        rate = handle.samplerate
        first = int(round(start * rate))
        handle.seek(min(first, handle.frames))
        count = -1 if duration is None else int(round(duration * rate))
        audio = handle.read(count, dtype="float32", always_2d=True)
    audio = resample(audio.mean(axis=1), rate, sample_rate)
    if duration is not None:
        wanted = int(round(duration * sample_rate))
        if len(audio) < wanted:
            audio = np.pad(audio, (0, wanted - len(audio)))
        audio = audio[:wanted]
    return audio


def resample(audio: np.ndarray, rate: int, sample_rate: int) -> np.ndarray:
    """Float32 ``audio`` at ``rate`` brought to ``sample_rate``."""
    from scipy.signal import resample_poly

    if rate == sample_rate:
        return audio.astype(np.float32, copy=False)
    g = math.gcd(rate, sample_rate)
    return resample_poly(audio, sample_rate // g, rate // g).astype(np.float32)


def frame_labels(
    notes: Notes, start: float, frames: int, frames_per_second: float
) -> np.ndarray:
    """``[4, 88, frames]`` onset, sustain, offset, and velocity targets.

    A note sustains from its onset frame up to, not including, its offset
    frame, and always for at least one frame. An onset or offset outside the
    window is not labelled, so a note entering mid-sustain carries none. The
    velocity, as ``velocity / 127``, sits at the onset frame; 0 there is
    unknown.
    """
    labels = np.zeros((4, NUM_PITCHES, frames), dtype=np.float32)
    end = start + frames / frames_per_second
    active = (notes.offsets > start) & (notes.onsets < end)
    for onset, offset, pitch, velocity in zip(
        notes.onsets[active],
        notes.offsets[active],
        notes.pitches[active],
        notes.velocities[active],
        strict=True,
    ):
        row = int(pitch) - MIN_PITCH
        on = int(round((onset - start) * frames_per_second))
        off = int(round((offset - start) * frames_per_second))
        off = max(off, on + 1)
        if 0 <= on < frames:
            labels[0, row, on] = 1.0
            labels[3, row, on] = velocity / 127
        labels[1, row, max(on, 0) : min(off, frames)] = 1.0
        if 0 <= off < frames:
            labels[2, row, off] = 1.0
    return labels


@lru_cache(maxsize=4096)
def _notes(path: str) -> Notes:
    return Notes.load(path)


class ClipStream(IterableDataset):
    """An endless stream of ``(audio [samples], labels [4, 88, T])`` clips."""

    def __init__(
        self,
        recordings: list[Recording],
        *,
        sample_rate: int,
        hop_length: int,
        clip_seconds: float,
        temperature: float,
        seed: int,
        rank: int = 0,
    ) -> None:
        super().__init__()
        self.by_source: dict[str, list[Recording]] = {}
        hours: dict[str, float] = {}
        for r in recordings:
            self.by_source.setdefault(r.source, []).append(r)
            hours[r.source] = hours.get(r.source, 0.0) + r.hours
        self.weights = source_weights(hours, temperature)
        self.sources = sorted(self.weights)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.clip_seconds = clip_seconds
        self.samples = int(round(clip_seconds * sample_rate))
        self.frames = num_frames(self.samples, hop_length)
        self.seed = seed
        self.rank = rank

    def clip(self, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
        source = rng.choices(self.sources, [self.weights[s] for s in self.sources])[0]
        recording = rng.choice(self.by_source[source])
        latest = max(0.0, recording.duration - self.clip_seconds)
        start = rng.uniform(0.0, latest)
        audio = read_audio(recording.audio, self.sample_rate, start, self.clip_seconds)
        labels = frame_labels(
            _notes(recording.notes),
            start,
            self.frames,
            self.sample_rate / self.hop_length,
        )
        return torch.from_numpy(audio), torch.from_numpy(labels)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = random.Random(f"{self.seed}/{self.rank}/{worker_id}")
        while True:
            yield self.clip(rng)
