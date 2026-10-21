"""The paper's eight corpora as one manifest of audio--note pairs.

``prepare`` walks every configured source once and writes, under
``data.cache_dir``:

* ``notes/<source>/<key>.npz`` -- onset, offset (seconds), and pitch arrays;
* ``audio/<source>/<key>.flac`` -- 16 kHz mono, for sources whose audio is only
  readable inside a zip (MAPS here) or that need mixing first
  (GuitarSet's pickup, summed from its six hexaphonic strings), and, with
  ``data.cache_audio``, for every recording not already 16 kHz mono;
* ``manifest.jsonl`` -- one row per recording, with its split;
* ``dev.json`` -- the training recordings held out as the development set.

A row is one audio--MIDI pair, which is what the paper counts in Table 3, so
URMP and Slakh contribute one row per stem and GuitarSet two per take.
"""

import io
import json
import math
import random
import re
import subprocess
import zipfile
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger

from harmonica.config import (
    MAX_PITCH,
    MIN_PITCH,
    SOURCES,
    HarmonicaConfig,
)
from harmonica.slakh_redux import REDUX_OMITTED_TRACKS

MANIFEST_NAME = "manifest.jsonl"
DEV_NAME = "dev.json"
TRAIN, TEST = "train", "test"
DEV = "dev"
CACHE_SAMPLE_RATE = 16000

# MT3's URMP validation pieces, which the paper evaluates on.
URMP_TEST_PIECES = frozenset({1, 2, 12, 13, 24, 25, 31, 38, 39})
# Slakh's sound-effects family is never rendered.
SLAKH_EXCLUDED_CLASSES = frozenset({"Sound Effects", "Drums"})


@dataclass(frozen=True)
class Recording:
    """One audio--note pair in the manifest."""

    source: str
    key: str
    split: str
    audio: str
    notes: str
    duration: float

    @property
    def hours(self) -> float:
        return self.duration / 3600


@dataclass(frozen=True)
class Notes:
    """Notes as three parallel arrays, in seconds and MIDI pitch."""

    onsets: np.ndarray
    offsets: np.ndarray
    pitches: np.ndarray

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            onsets=self.onsets.astype(np.float64),
            offsets=self.offsets.astype(np.float64),
            pitches=self.pitches.astype(np.int16),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Notes":
        with np.load(path) as stored:
            return cls(stored["onsets"], stored["offsets"], stored["pitches"])

    @classmethod
    def from_rows(cls, rows: list[tuple[float, float, int]]) -> "Notes":
        """Keep the notes on the 88 keys that last a positive time, sorted."""
        kept = sorted(
            (on, off, p)
            for on, off, p in rows
            if MIN_PITCH <= p <= MAX_PITCH and off > on
        )
        if not kept:
            return cls(np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int16))
        on, off, pitch = zip(*kept, strict=True)
        return cls(np.array(on), np.array(off), np.array(pitch, dtype=np.int16))

    def __len__(self) -> int:
        return len(self.pitches)

    def save_midi(self, path: str | Path) -> None:
        """One piano track at 120 BPM, every note at velocity 80."""
        import symusic

        score = symusic.Score(480, ttype="second")
        score.tempos.append(symusic.Tempo(0.0, 120, ttype="second"))
        track = symusic.Track(program=0, is_drum=False, ttype="second")
        for onset, offset, pitch in zip(
            self.onsets, self.offsets, self.pitches, strict=True
        ):
            track.notes.append(
                symusic.Note(
                    float(onset), float(offset - onset), int(pitch), 80, ttype="second"
                )
            )
        score.tracks.append(track)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        score.to("tick", min_dur=1).dump_midi(str(path))


# --------------------------------------------------------------------------- MIDI


def notes_from_midi(data: bytes, sustain_pedal: bool) -> Notes:
    """Every pitched note in a MIDI file, pedal-extended if asked.

    Read through symusic with its controllers, since the pedal is one.
    Extension follows Onsets and Frames:
    a note released under a held pedal sounds until the pedal lifts or the same
    key is struck again, whichever is first.
    """
    import symusic

    score = symusic.Score.from_midi(data).to("second")
    rows: list[tuple[float, float, int]] = []
    for track in score.tracks:
        if track.is_drum:
            continue
        notes = [(n.time, n.time + n.duration, n.pitch) for n in track.notes]
        if sustain_pedal:
            pedal = [(c.time, c.value >= 64) for c in track.controls if c.number == 64]
            notes = extend_by_pedal(notes, pedal)
        rows.extend(notes)
    return Notes.from_rows(rows)


def extend_by_pedal(
    notes: list[tuple[float, float, int]], pedal: list[tuple[float, bool]]
) -> list[tuple[float, float, int]]:
    """Hold each note's release until the pedal lifts or its key is restruck."""
    downs: list[tuple[float, float]] = []
    start: float | None = None
    for time, down in sorted(pedal):
        if down and start is None:
            start = time
        elif not down and start is not None:
            downs.append((start, time))
            start = None
    if start is not None:
        downs.append((start, math.inf))
    if not downs:
        return notes

    starts = np.array([s for s, _ in downs])
    extended: list[tuple[float, float, int]] = []
    for onset, offset, pitch in notes:
        index = int(np.searchsorted(starts, offset, side="right")) - 1
        if index >= 0 and downs[index][0] <= offset < downs[index][1]:
            offset = downs[index][1]
        extended.append((onset, offset, pitch))

    # A restruck key cuts the note before it short; an endless pedal ends
    # with the last thing the key does.
    by_pitch: dict[int, list[int]] = {}
    for index, (_, _, pitch) in enumerate(extended):
        by_pitch.setdefault(pitch, []).append(index)
    result = list(extended)
    for indices in by_pitch.values():
        indices.sort(key=lambda i: extended[i][0])
        for here, after in zip(indices, indices[1:], strict=False):
            onset, offset, pitch = result[here]
            next_onset = extended[after][0]
            if offset > next_onset:
                result[here] = (onset, max(next_onset, onset + 1e-3), pitch)
        last = indices[-1]
        onset, offset, pitch = result[last]
        if math.isinf(offset):
            original = notes[last][1]
            result[last] = (onset, original, pitch)
    return result


def hz_to_midi(frequency: float) -> int:
    return round(69 + 12 * math.log2(frequency / 440.0))


# ------------------------------------------------------------------ audio caching


def write_flac_16k(audio: np.ndarray, sample_rate: int, path: Path) -> float:
    """Mix to mono, resample to 16 kHz, write FLAC; return the duration."""
    from scipy.signal import resample_poly

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != CACHE_SAMPLE_RATE:
        g = math.gcd(sample_rate, CACHE_SAMPLE_RATE)
        audio = resample_poly(audio, CACHE_SAMPLE_RATE // g, sample_rate // g)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(".tmp.flac")
    sf.write(staged, audio, CACHE_SAMPLE_RATE, subtype="PCM_16")
    staged.replace(path)
    return len(audio) / CACHE_SAMPLE_RATE


def audio_duration(path: str | Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


# ------------------------------------------------------------------------ readers
#
# A reader lists ``Job``s: plain data naming one manifest row and how to build
# it. ``build_job`` runs in a worker process, where most of the time goes to
# decoding and resampling audio.


@dataclass(frozen=True)
class Job:
    source: str
    key: str
    split: str
    kind: str
    args: tuple


def _zip_member(archive: str, member: str) -> bytes:
    """One member's bytes; through ``unzip`` when ``zipfile`` refuses the archive.

    GuitarSet's zip records a Zip64 offset four gigabytes off, which Python
    reads as overlapping entries; Info-ZIP's ``unzip`` corrects for it.
    """
    try:
        with zipfile.ZipFile(archive) as z:
            return z.read(member)
    except zipfile.BadZipFile:
        result = subprocess.run(
            ["unzip", "-p", archive, member], capture_output=True, check=False
        )
        if not result.stdout:
            raise
        return result.stdout


def _cached_audio(
    cache: Path, source: str, key: str, read: Callable[[], tuple[np.ndarray, int]]
) -> tuple[str, float]:
    target = cache / "audio" / source / f"{key}.flac"
    if target.exists():
        return str(target), audio_duration(target)
    audio, rate = read()
    return str(target), write_flac_16k(audio, rate, target)


def _local_audio(cache: Path, source: str, key: str, audio: str) -> tuple[str, float]:
    """``audio`` itself when it is already 16 kHz mono, else its cached copy.

    A training clip is a random 5 s window, and reading one from 44.1 kHz FLAC
    or over network storage costs tens of milliseconds; from a local 16 kHz
    FLAC it costs two.
    """
    info = sf.info(audio)
    if info.samplerate == CACHE_SAMPLE_RATE and info.channels == 1:
        return audio, info.frames / info.samplerate
    return _cached_audio(
        cache, source, key, lambda: sf.read(audio, dtype="float32", always_2d=True)
    )


def _read_zipped(archive: str, member: str) -> tuple[np.ndarray, int]:
    return sf.read(
        io.BytesIO(_zip_member(archive, member)), dtype="float32", always_2d=True
    )


def _read_hex_sum(archive: str, member: str) -> tuple[np.ndarray, int]:
    """Six strings summed, not averaged, so the mix is as loud as one guitar."""
    audio, rate = _read_zipped(archive, member)
    mono = audio.sum(axis=1, keepdims=True)
    peak = float(np.abs(mono).max()) or 1.0
    return (mono / peak if peak > 1.0 else mono), rate


def build_job(
    job: Job, cache: Path, cache_audio: bool = False
) -> tuple[str, float, Notes]:
    """Produce one row's audio path, duration, and notes."""

    def local(audio: str) -> tuple[str, float]:
        if cache_audio:
            return _local_audio(cache, job.source, job.key, audio)
        return audio, audio_duration(audio)

    if job.kind == "maestro":
        audio, midi, sustain = job.args
        notes = notes_from_midi(Path(midi).read_bytes(), sustain)
        return *local(audio), notes
    if job.kind == "maps":
        archive, member = job.args
        text = _zip_member(archive, member[: -len(".wav")] + ".txt").decode()
        notes = maps_notes(text)
        path, duration = _cached_audio(
            cache, job.source, job.key, lambda: _read_zipped(archive, member)
        )
        return path, duration, notes
    if job.kind == "guitarset":
        archive, jams_member, audio_member, summed = job.args
        notes = guitarset_notes(_zip_member(archive, jams_member))
        read = _read_hex_sum if summed else _read_zipped
        path, duration = _cached_audio(
            cache, job.source, job.key, lambda: read(archive, audio_member)
        )
        return path, duration, notes
    if job.kind == "guitarset_official":
        jams_archive, jams_member, audio_archive, audio_member = job.args
        notes = guitarset_notes(_zip_member(jams_archive, jams_member))
        path, duration = _cached_audio(
            cache,
            job.source,
            job.key,
            lambda: _read_zipped(audio_archive, audio_member),
        )
        return path, duration, notes
    if job.kind == "urmp":
        audio, notes_path = job.args
        return *local(audio), urmp_notes(Path(notes_path).read_text())
    if job.kind == "egdb":
        audio, midi = job.args
        notes = notes_from_midi(Path(midi).read_bytes(), sustain_pedal=False)
        return *local(audio), notes
    if job.kind == "gaps":
        audio, midi = job.args
        notes = notes_from_midi(Path(midi).read_bytes(), sustain_pedal=False)
        return *local(audio), notes
    if job.kind == "goat":
        audio, midi = job.args
        notes = notes_from_midi(Path(midi).read_bytes(), sustain_pedal=False)
        return *local(audio), notes
    if job.kind == "slakh":
        audio, midi = job.args
        notes = notes_from_midi(Path(midi).read_bytes(), sustain_pedal=False)
        return *local(audio), notes
    raise ValueError(f"unknown job kind {job.kind!r}")


def maestro_jobs(root: Path, sustain: bool) -> Iterator[Job]:
    """MAESTRO v3, extracted: ``maestro-v3.0.0.json`` and its year folders.

    Train and validation train; test tests (Table 3).
    """
    rows = json.loads((root / "maestro-v3.0.0.json").read_text())
    for i in rows["split"]:
        split = TEST if rows["split"][i] == "test" else TRAIN
        audio = root / rows["audio_filename"][i]
        midi = root / rows["midi_filename"][i]
        yield Job(
            "maestro",
            audio.stem,
            split,
            "maestro",
            (str(audio), str(midi), sustain),
        )


def maps_jobs(root: Path) -> Iterator[Job]:
    """MAPS's two Disklavier pianos, full pieces only: test only."""
    for archive in sorted((root / "Source").glob("MAPS_ENSTDk*.zip")):
        with zipfile.ZipFile(archive) as z:
            members = [n for n in z.namelist() if "/MUS/" in n and n.endswith(".wav")]
        for member in members:
            yield Job("maps", Path(member).stem, TEST, "maps", (str(archive), member))


def maps_notes(text: str) -> Notes:
    """``OnsetTime OffsetTime MidiPitch``, one note a line under a header."""
    rows = []
    for line in text.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 3:
            rows.append((float(fields[0]), float(fields[1]), int(fields[2])))
    return Notes.from_rows(rows)


_GUITARSET_TAKE = re.compile(r"^(\d\d)_([A-Za-z]+)(\d)-(\d+)-([A-G]#?b?)_(comp|solo)$")


def guitarset_jobs(root: Path) -> Iterator[Job]:
    """GuitarSet's microphone and pickup, as two pairs per take.

    The release here has no ``audio_mono-pickup_mix``; the pickup is rebuilt by
    summing the six debleeded hexaphonic strings, which is what that mix is.
    Progressions 1 and 2 train, 3 tests, as in MT3.
    """
    if (root / "annotation.zip").exists():
        yield from _guitarset_official_jobs(root)
        return
    archive = str(root / "GuitarSet.zip")
    with zipfile.ZipFile(archive) as z:
        jams = sorted(
            n for n in z.namelist() if n.endswith(".jams") and "__MACOSX" not in n
        )
    for member in jams:
        take = Path(member).stem
        match = _GUITARSET_TAKE.match(take)
        if match is None:
            logger.warning("GuitarSet: skipping unrecognised take {}", take)
            continue
        split = TEST if match.group(3) == "3" else TRAIN
        kind = match.group(6)
        mic = f"GuitarSet/audio/audio_mic/{kind}_mic/{take}_mic.wav"
        hexaphonic = (
            f"GuitarSet/audio/audio_hex-pickup_debleeded/{kind}_hex_cln/"
            f"{take}_hex_cln.wav"
        )
        yield Job(
            "guitarset",
            f"{take}_mic",
            split,
            "guitarset",
            (archive, member, mic, False),
        )
        yield Job(
            "guitarset",
            f"{take}_pickup",
            split,
            "guitarset",
            (archive, member, hexaphonic, True),
        )


def _guitarset_official_jobs(root: Path) -> Iterator[Job]:
    """Zenodo's release: ``annotation.zip``, ``audio_mono-mic.zip``, and
    ``audio_mono-pickup_mix.zip``, flat inside, so both pairs are as recorded."""
    with zipfile.ZipFile(root / "annotation.zip") as z:
        jams = sorted(n for n in z.namelist() if n.endswith(".jams"))
    for member in jams:
        take = Path(member).stem
        match = _GUITARSET_TAKE.match(take)
        if match is None:
            logger.warning("GuitarSet: skipping unrecognised take {}", take)
            continue
        split = TEST if match.group(3) == "3" else TRAIN
        for variant, archive, audio in (
            ("mic", "audio_mono-mic.zip", f"{take}_mic.wav"),
            ("pickup", "audio_mono-pickup_mix.zip", f"{take}_mix.wav"),
        ):
            yield Job(
                "guitarset",
                f"{take}_{variant}",
                split,
                "guitarset_official",
                (str(root / "annotation.zip"), member, str(root / archive), audio),
            )


def guitarset_notes(data: bytes) -> Notes:
    """The six ``note_midi`` annotations of a JAMS file, merged, via REMI-z.

    The paper converts the JAMS to MIDI with REMI-z, so this does too: the
    notes become a ``NoteStream`` (times rounded to the millisecond, at least
    1 ms long), are written as a MIDI file, and are read back from it -- where
    two overlapping notes on one key are paired up again as MIDI pairs them.
    """
    import tempfile

    from remi_z import NoteStream

    jams = json.loads(data)
    rows = []
    for annotation in jams["annotations"]:
        if annotation["namespace"] != "note_midi":
            continue
        for note in annotation["data"]:
            onset = float(note["time"])
            rows.append([onset, onset + float(note["duration"]), round(note["value"])])
    if not rows:
        return Notes.from_rows([])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "guitarset.mid"
        NoteStream.from_triplet_list(rows).to_midi(str(path))
        return notes_from_midi(path.read_bytes(), sustain_pedal=False)


def urmp_jobs(root: Path) -> Iterator[Job]:
    """URMP's separated stems with their note files, piece by piece."""
    for piece in sorted((root / "Dataset").iterdir()):
        match = re.match(r"^(\d\d)_", piece.name)
        if match is None or not piece.is_dir():
            continue
        split = TEST if int(match.group(1)) in URMP_TEST_PIECES else TRAIN
        for audio in sorted(piece.glob("AuSep_*.wav")):
            notes = piece / ("Notes_" + audio.stem[len("AuSep_") :] + ".txt")
            yield Job("urmp", audio.stem, split, "urmp", (str(audio), str(notes)))


def urmp_notes(text: str) -> Notes:
    """``onset frequency duration`` a line, in seconds and Hz."""
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 3:
            onset, frequency, duration = map(float, fields[:3])
            rows.append((onset, onset + duration, hz_to_midi(frequency)))
    return Notes.from_rows(rows)


def slakh_jobs(root: Path) -> Iterator[Job]:
    """Slakh2100-redux's pitched stems, each with its own MIDI: no drums, no mixes.

    Read from the original layout, skipping the tracks redux omits; a copy
    already resplit to redux moved them to ``omitted/``, which is never read.
    Table 3 counts the drum stems too (16,379 and 1,539 rows with them); the
    text excludes them, and so does this.
    """
    import yaml

    for split_dir, split in (("train", TRAIN), ("validation", TRAIN), ("test", TEST)):
        for track in sorted((root / split_dir).glob("Track*")):
            if int(track.name.removeprefix("Track")) in REDUX_OMITTED_TRACKS:
                continue
            metadata = yaml.safe_load((track / "metadata.yaml").read_text())
            for stem, info in metadata["stems"].items():
                if (
                    info.get("is_drum")
                    or not info.get("audio_rendered")
                    or not info.get("midi_saved")
                    or info.get("inst_class") in SLAKH_EXCLUDED_CLASSES
                ):
                    continue
                audio = track / "stems" / f"{stem}.flac"
                midi = track / "MIDI" / f"{stem}.mid"
                if audio.exists() and midi.exists():
                    yield Job(
                        "slakh",
                        f"{track.name}_{stem}",
                        split,
                        "slakh",
                        (str(audio), str(midi)),
                    )


EGDB_TONES = ("DI", "Ftwin", "JCjazz", "Marshall", "Mesa", "Plexi")
# EGDB's README: "we choose clips from '216' to '241' to match a 9/1 splitting".
EGDB_FIRST_TEST_CLIP = 216


def egdb_jobs(root: Path) -> Iterator[Job]:
    """EGDB's 240 clips in all six tones, one row per clip and tone.

    The README's clip split, applied to every tone: 25 test clips, 150 test
    rows, which is Table 3's count. Its alternative timbre split (JCjazz and
    Plexi held out) is not the paper's. ``RealData`` is left out.
    """
    for tone in EGDB_TONES:
        for audio in sorted((root / f"audio_{tone}").glob("*.wav")):
            clip = audio.stem
            midi = root / "audio_label" / f"{clip}.midi"
            if not clip.isdigit() or not midi.exists():
                continue
            split = TEST if int(clip) >= EGDB_FIRST_TEST_CLIP else TRAIN
            yield Job("egdb", f"{tone}_{clip}", split, "egdb", (str(audio), str(midi)))


def gaps_jobs(root: Path) -> Iterator[Job]:
    """GAPS v1.1 from Hugging Face (``xavriley/GAPS``), which ships its audio.

    The metadata's own split column: 270 train and 30 test pieces, Table 3's
    counts. The hundred-odd rows it leaves unsplit are not used.
    """
    import csv

    with (root / "gaps_metadata_with_splits.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["split"] not in (TRAIN, TEST):
            continue
        audio, midi = root / row["audio_path"], root / row["midi_path"]
        if audio.exists() and midi.exists():
            yield Job("gaps", row["id"], row["split"], "gaps", (str(audio), str(midi)))


# GOAT's recordings of each item: the DI signal and its five amp renders. The
# Guitar Pro synthesis (``gp_audio_path``) is not a recording and is left out.
GOAT_VARIANTS = {
    "di": "di_audio_path",
    **{f"amp_{n}": f"amp_audio_path_{n}" for n in range(1, 6)},
}


def goat_jobs(root: Path) -> Iterator[Job]:
    """GOAT's six recordings an item, each with the item's fine-aligned MIDI.

    The metadata's own split. Items without a fine-aligned MIDI (20, all in
    train) are left out, which leaves Table 3's counts exactly: 145 train items
    for 870 rows and 31.3 h, 7 test items for 42 rows and 1.4 h. Paths in
    ``metadata.csv`` start ``GOAT/``, which on disk is ``data/``.
    """
    import csv

    def local(path: str) -> Path:
        return root / "data" / Path(path).relative_to("GOAT")

    with (root / "metadata.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["split"] not in (TRAIN, TEST) or not row["finealigned_midi_path"]:
            continue
        midi = local(row["finealigned_midi_path"])
        if not midi.exists():
            continue
        for variant, column in GOAT_VARIANTS.items():
            audio = local(row[column])
            if audio.exists():
                yield Job(
                    "goat",
                    f"{row['item']}_{variant}",
                    row["split"],
                    "goat",
                    (str(audio), str(midi)),
                )


def jobs_for(config: HarmonicaConfig) -> Iterator[Job]:
    sources = config.data.sources
    readers: dict[str, Callable[[Path], Iterator[Job]]] = {
        "maestro": lambda r: maestro_jobs(r, config.data.sustain_pedal),
        "maps": maps_jobs,
        "guitarset": guitarset_jobs,
        "gaps": gaps_jobs,
        "egdb": egdb_jobs,
        "goat": goat_jobs,
        "urmp": urmp_jobs,
        "slakh": slakh_jobs,
    }
    for name in SOURCES:
        root = getattr(sources, name)
        if root is None:
            logger.warning("Harmonica: data.sources.{} is null; skipping it", name)
            continue
        # Absolute, so the manifest reads the same from any working directory.
        yield from readers[name](Path(root).absolute())


# ---------------------------------------------------------------------- manifest


def _run_job(job: Job, cache: Path, cache_audio: bool) -> dict | None:
    notes_path = cache / "notes" / job.source / f"{job.key}.npz"
    try:
        audio, duration, notes = build_job(job, cache, cache_audio)
    except Exception as error:  # noqa: BLE001 -- one bad file must not stop a corpus
        logger.warning("{}/{}: skipped ({})", job.source, job.key, error)
        return None
    if len(notes) == 0:
        logger.warning("{}/{}: no notes on the 88 keys; skipped", job.source, job.key)
        return None
    notes.save(notes_path)
    return asdict(
        Recording(job.source, job.key, job.split, audio, str(notes_path), duration)
    )


def _run_job_star(args: tuple[Job, Path, bool]) -> dict | None:
    return _run_job(*args)


def prepare(config: HarmonicaConfig, workers: int) -> Path:
    """Build the manifest and the development split; return the manifest."""
    cache = Path(config.data.cache_dir).absolute()
    cache.mkdir(parents=True, exist_ok=True)
    jobs = list(jobs_for(config))
    logger.info(
        "Harmonica: preparing {} recordings with {} workers", len(jobs), workers
    )

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, row in enumerate(
            pool.map(
                _run_job_star,
                ((job, cache, config.data.cache_audio) for job in jobs),
                chunksize=4,
            )
        ):
            if row is not None:
                rows.append(row)
            if (index + 1) % 500 == 0:
                logger.info("Harmonica: {}/{} prepared", index + 1, len(jobs))

    manifest = cache / MANIFEST_NAME
    staged = manifest.with_suffix(".tmp")
    with staged.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    staged.replace(manifest)

    recordings = [Recording(**row) for row in rows]
    dev = select_dev(recordings, config)
    (cache / DEV_NAME).write_text(json.dumps(sorted(dev), indent=1))
    summarize(recordings, dev)
    return manifest


def source_weights(hours: dict[str, float], temperature: float) -> dict[str, float]:
    """``(n_i / sum_j n_j) ** alpha``, normalized: the paper's sampling."""
    total = sum(hours.values())
    raw = {name: (h / total) ** temperature for name, h in hours.items() if h > 0}
    norm = sum(raw.values())
    return {name: w / norm for name, w in raw.items()}


def select_dev(recordings: list[Recording], config: HarmonicaConfig) -> set[str]:
    """Hold out ``dev_hours`` of training audio, shared out by source weight.

    Each source gives its share of the hours -- the temperature-sampled
    proportion -- as whole recordings drawn at random, so the development set
    is distributed the way training batches are (Table 3).
    """
    train = [r for r in recordings if r.split == TRAIN]
    hours: dict[str, float] = {}
    for r in train:
        hours[r.source] = hours.get(r.source, 0.0) + r.hours
    if not hours:
        return set()
    weights = source_weights(hours, config.data.temperature)
    rng = random.Random(config.data.dev_seed)
    chosen: set[str] = set()
    for source, weight in sorted(weights.items()):
        budget = weight * config.data.dev_hours
        pool = sorted((r for r in train if r.source == source), key=lambda r: r.key)
        rng.shuffle(pool)
        taken = 0.0
        for r in pool:
            if taken >= budget:
                break
            chosen.add(record_id(r))
            taken += r.hours
    return chosen


def record_id(recording: Recording) -> str:
    return f"{recording.source}/{recording.key}"


def summarize(recordings: list[Recording], dev: set[str]) -> None:
    table: dict[tuple[str, str], list[float]] = {}
    for r in recordings:
        split = DEV if record_id(r) in dev else r.split
        cell = table.setdefault((r.source, split), [0, 0.0])
        cell[0] += 1
        cell[1] += r.hours
    for (source, split), (count, hours) in sorted(table.items()):
        logger.info(
            "  {:<10} {:<5} {:>6} recordings {:>8.1f} h", source, split, count, hours
        )


def load_manifest(config: HarmonicaConfig) -> dict[str, list[Recording]]:
    """The prepared corpus by split: ``train`` without dev, ``dev``, ``test``."""
    cache = Path(config.data.cache_dir)
    manifest = cache / MANIFEST_NAME
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} does not exist; run `harmonica prepare` first"
        )
    dev = set(json.loads((cache / DEV_NAME).read_text()))
    configured = {name for name in SOURCES if getattr(config.data.sources, name)}
    splits: dict[str, list[Recording]] = {TRAIN: [], DEV: [], TEST: []}
    with manifest.open() as handle:
        for line in handle:
            r = Recording(**json.loads(line))
            if r.source not in configured:
                continue
            split = DEV if record_id(r) in dev else r.split
            splits[split].append(r)
    return splits
