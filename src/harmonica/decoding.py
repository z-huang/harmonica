"""Frame activations to notes, and the paper's three F1 scores.

Decoding is the paper's ``g``: an onset is a peak of the onset activation at or
above its threshold, and the note it opens lasts while the sustain activation
stays at or above a separate threshold. The offset activation is training
signal only and plays no part here. A velocity activation, from a model with
that head, gives each note the velocity read at its onset frame.
"""

from dataclasses import dataclass

import numpy as np

from harmonica.config import MIN_PITCH
from harmonica.corpus import Notes


def decode(
    onset: np.ndarray,
    sustain: np.ndarray,
    frames_per_second: float,
    onset_threshold: float,
    frame_threshold: float,
    velocity: np.ndarray | None = None,
) -> Notes:
    """``[88, T]`` probabilities to notes.

    A note opened at frame ``t`` ends at the first later frame whose sustain
    falls below ``frame_threshold``, or at the key's next onset, whichever
    comes first, and always lasts at least one frame. Without ``velocity``,
    every note's velocity is unknown.
    """
    pitches, frames = onset.shape
    rows: list[tuple[float, float, int, int]] = []
    previous = np.pad(onset, ((0, 0), (1, 0)), constant_values=-1.0)[:, :-1]
    following = np.pad(onset, ((0, 0), (0, 1)), constant_values=-1.0)[:, 1:]
    peaks = (onset >= onset_threshold) & (onset >= previous) & (onset > following)
    below = sustain < frame_threshold
    index = np.arange(frames)
    for row in np.flatnonzero(peaks.any(axis=1)):
        starts = np.flatnonzero(peaks[row])
        # For every frame, the first frame at or after it whose sustain is low.
        low = np.where(below[row], index, frames)
        next_low = np.minimum.accumulate(low[::-1])[::-1]
        next_low = np.append(next_low, frames)
        for here, start in enumerate(starts):
            end = next_low[start + 1]
            if here + 1 < len(starts):
                end = min(end, starts[here + 1])
            end = max(end, start + 1)
            loudness = 0
            if velocity is not None:
                loudness = int(np.clip(round(velocity[row, start] * 127), 1, 127))
            rows.append(
                (
                    start / frames_per_second,
                    end / frames_per_second,
                    MIN_PITCH + row,
                    loudness,
                )
            )
    return Notes.from_rows(rows)


@dataclass(frozen=True)
class Scores:
    onset: float
    onset_offset: float
    frame: float

    def as_dict(self) -> dict[str, float]:
        return {"OnP": self.onset, "OnPOff": self.onset_offset, "Frm": self.frame}


def _hz(pitches: np.ndarray) -> np.ndarray:
    return 440.0 * 2.0 ** ((pitches.astype(np.float64) - 69) / 12)


def _intervals(notes: Notes) -> np.ndarray:
    if len(notes) == 0:
        return np.zeros((0, 2))
    return np.stack([notes.onsets, notes.offsets], axis=1)


def note_f1(
    reference: Notes,
    estimate: Notes,
    *,
    onset_tolerance: float,
    offset_ratio: float | None,
    offset_min_tolerance: float,
) -> float:
    """mir_eval's note F1, with or without the offset criterion.

    Matched one pitch at a time. mir_eval pairs notes within 50 cents, and
    the keys are 100 apart, so only notes on one key can match: the maximum
    matching of the whole is the sum of each key's, and the F1 is the same.
    mir_eval builds a dense reference-by-estimate graph, so on a 23-minute
    MAESTRO piece (15k notes) this is hundreds of times faster.
    """
    import mir_eval

    if len(reference) == 0 and len(estimate) == 0:
        return 1.0
    if len(reference) == 0 or len(estimate) == 0:
        return 0.0
    reference_intervals, estimate_intervals = (
        _intervals(reference),
        _intervals(estimate),
    )
    mir_eval.transcription.validate(
        reference_intervals,
        _hz(reference.pitches),
        estimate_intervals,
        _hz(estimate.pitches),
    )
    matched = 0
    for pitch in np.intersect1d(reference.pitches, estimate.pitches):
        on_reference = reference.pitches == pitch
        on_estimate = estimate.pitches == pitch
        matched += len(
            mir_eval.transcription.match_notes(
                reference_intervals[on_reference],
                _hz(reference.pitches[on_reference]),
                estimate_intervals[on_estimate],
                _hz(estimate.pitches[on_estimate]),
                onset_tolerance=onset_tolerance,
                offset_ratio=offset_ratio,
                offset_min_tolerance=offset_min_tolerance,
            )
        )
    precision = matched / len(estimate)
    recall = matched / len(reference)
    return float(mir_eval.util.f_measure(precision, recall))


def piano_roll(notes: Notes, frame_seconds: float, frames: int) -> np.ndarray:
    """``[128, frames]`` booleans: a note covers the frames whose centres it spans."""
    roll = np.zeros((128, frames), dtype=bool)
    for onset, offset, pitch in zip(
        notes.onsets, notes.offsets, notes.pitches, strict=True
    ):
        first = int(np.ceil(onset / frame_seconds - 0.5))
        last = int(np.ceil(offset / frame_seconds - 0.5))
        roll[int(pitch), max(first, 0) : min(last, frames)] = True
    return roll


def frame_f1(reference: Notes, estimate: Notes, frame_seconds: float) -> float:
    """Frame F1 from notes on a fixed grid, the same for every model."""
    end = max(
        float(reference.offsets.max()) if len(reference) else 0.0,
        float(estimate.offsets.max()) if len(estimate) else 0.0,
    )
    frames = int(np.ceil(end / frame_seconds)) + 1
    ref = piano_roll(reference, frame_seconds, frames)
    est = piano_roll(estimate, frame_seconds, frames)
    hits = int((ref & est).sum())
    total = int(ref.sum()) + int(est.sum())
    if total == 0:
        return 1.0
    return 2 * hits / total


def score(
    reference: Notes,
    estimate: Notes,
    *,
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
    offset_min_tolerance: float = 0.05,
    frame_seconds: float = 0.025,
) -> Scores:
    """OnP, OnPOff, and Frm for one recording."""
    return Scores(
        onset=note_f1(
            reference,
            estimate,
            onset_tolerance=onset_tolerance,
            offset_ratio=None,
            offset_min_tolerance=offset_min_tolerance,
        ),
        onset_offset=note_f1(
            reference,
            estimate,
            onset_tolerance=onset_tolerance,
            offset_ratio=offset_ratio,
            offset_min_tolerance=offset_min_tolerance,
        ),
        frame=frame_f1(reference, estimate, frame_seconds),
    )


def mean_scores(scores: list[Scores]) -> Scores:
    """The mean over recordings of each score."""
    if not scores:
        return Scores(float("nan"), float("nan"), float("nan"))
    return Scores(
        onset=float(np.mean([s.onset for s in scores])),
        onset_offset=float(np.mean([s.onset_offset for s in scores])),
        frame=float(np.mean([s.frame for s in scores])),
    )
