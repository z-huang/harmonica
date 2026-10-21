"""Harmonica: the paper's numbers, and the pieces that must agree with each other."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from harmonica.config import (
    MIN_PITCH,
    NUM_PITCHES,
    SIZES,
    HarmonicaConfig,
    HarmonicaHarmonicConfig,
    HarmonicaModelConfig,
    load_config,
)
from harmonica.corpus import (
    Notes,
    Recording,
    extend_by_pedal,
    guitarset_notes,
    maps_notes,
    select_dev,
    source_weights,
    urmp_notes,
)
from harmonica.data import frame_labels
from harmonica.decoding import decode, frame_f1, score
from harmonica.inference import transcribe_audio
from harmonica.model import (
    SILENCE,
    HarmonicaModel,
    ShiftAggregate,
    count_parameters,
    harmonic_layer_parameters,
    harmonic_offsets,
    num_frames,
    shift_frequency,
)

CONFIG = Path("config/harmonica.yaml")

# Table 1's parameter counts.
PAPER_PARAMETERS = {
    "nano": 26.3e3,
    "small": 137e3,
    "medium": 679e3,
    "large": 3.2e6,
    "x-large": 15.1e6,
}


# ------------------------------------------------------------------------- model


def test_offsets_are_the_papers_seven_harmonics_and_seven_subharmonics() -> None:
    upper = [12, 19, 24, 28, 31, 34, 36]
    assert harmonic_offsets(8) == sorted([0, *upper, *(-o for o in upper)])


def test_medium_has_654_291_parameters() -> None:
    model = HarmonicaModel(HarmonicaModelConfig(size="medium"))
    assert count_parameters(model) == 654_291
    assert count_parameters(model.stem) == 6_928


@pytest.mark.parametrize("size", list(SIZES))
def test_every_size_is_a_little_under_table_1(size: str) -> None:
    model = HarmonicaModel(HarmonicaModelConfig(size=size))
    count = count_parameters(model)
    assert 0.93 * PAPER_PARAMETERS[size] < count < PAPER_PARAMETERS[size]


def test_shift_aggregate_saves_what_section_3_4_says_over_hdconv() -> None:
    ours = harmonic_layer_parameters(HarmonicaModel(HarmonicaModelConfig()))
    hdconv = harmonic_layer_parameters(
        HarmonicaModel(
            HarmonicaModelConfig(harmonic=HarmonicaHarmonicConfig(kind="hdconv"))
        )
    )
    # "reduces the parameter count of the harmonic layers by 28.5%", and
    # Table 2's HD-Conv has "39.9% more parameters per harmonic layer".
    assert round(100 * (1 - ours / hdconv), 1) == 28.5
    assert round(100 * (hdconv / ours - 1), 1) == 39.9


def test_shift_aggregate_is_shift_concatenate_and_one_1x1_conv() -> None:
    torch.manual_seed(0)
    offsets = harmonic_offsets(8)
    layer = ShiftAggregate(3, offsets)
    x = torch.randn(2, 3, NUM_PITCHES, 5)
    stacked = torch.cat([shift_frequency(x, o) for o in offsets], dim=1)
    # The weight's columns are offset-major, as the concatenation is.
    weight = layer.weight.view(3, len(offsets), 3).reshape(3, -1, 1, 1)
    expected = torch.nn.functional.conv2d(stacked, weight, layer.bias)
    torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-5)


def test_shift_reads_the_bin_offset_above() -> None:
    x = torch.arange(6.0).view(1, 1, 6, 1)
    assert shift_frequency(x, 2).flatten().tolist() == [2, 3, 4, 5, 0, 0]
    assert shift_frequency(x, -2).flatten().tolist() == [0, 0, 0, 1, 2, 3]
    assert shift_frequency(x, 7).abs().sum() == 0


@pytest.mark.parametrize(
    ("kind", "depth"),
    [
        ("shift_aggregate", "multi"),
        ("shift_aggregate", "single"),
        ("hdconv", "multi"),
        ("stacking", "multi"),
        ("none", "multi"),
    ],
)
def test_every_table_2_variant_maps_audio_to_88_keys(kind: str, depth: str) -> None:
    model = HarmonicaModel(
        HarmonicaModelConfig(
            size="nano", harmonic=HarmonicaHarmonicConfig(kind=kind, depth=depth)
        )
    )
    samples = 16000
    out = model(torch.randn(2, samples))
    frames = num_frames(samples, 512)
    assert set(out) == {"onset", "sustain", "offset"}
    for logits in out.values():
        assert logits.shape == (2, NUM_PITCHES, frames)


def test_cqt_is_three_bins_a_semitone_over_the_88_keys() -> None:
    model = HarmonicaModel(HarmonicaModelConfig(size="nano"))
    spectrogram = model.spectrogram(torch.zeros(1, 8000))
    assert spectrogram.shape == (1, 1, 3 * NUM_PITCHES, num_frames(8000, 512))


def test_middle_bin_of_each_semitone_hears_its_key() -> None:
    """A4's sine lands in A4's middle bin, so the stride-3 conv centres on keys."""
    model = HarmonicaModel(HarmonicaModelConfig(size="nano"))
    t = torch.arange(16000) / 16000
    spectrogram = model.spectrogram(torch.sin(2 * math.pi * 440.0 * t)[None])
    loudest = int(spectrogram[0, 0, :, 10:-10].mean(dim=-1).argmax())
    assert loudest == 3 * (69 - MIN_PITCH) + 1


# ---------------------------------------------------------------- labels/decoding


def _notes(*rows: tuple[float, float, int]) -> Notes:
    return Notes.from_rows(list(rows))


def test_labels_mark_onset_sustain_and_offset_frames() -> None:
    labels = frame_labels(_notes((1.0, 2.0, 60)), 0.0, 100, 10.0)
    row = 60 - MIN_PITCH
    assert np.flatnonzero(labels[0, row]).tolist() == [10]
    assert np.flatnonzero(labels[1, row]).tolist() == list(range(10, 20))
    assert np.flatnonzero(labels[2, row]).tolist() == [20]
    assert labels.sum() == 1 + 10 + 1


def test_a_note_entering_the_window_sustains_without_an_onset() -> None:
    labels = frame_labels(_notes((0.5, 1.5, 60)), 1.0, 20, 10.0)
    row = 60 - MIN_PITCH
    assert labels[0].sum() == 0
    assert np.flatnonzero(labels[1, row]).tolist() == list(range(5))
    assert np.flatnonzero(labels[2, row]).tolist() == [5]


def test_decoding_the_labels_returns_the_notes() -> None:
    notes = _notes((0.5, 1.2, 60), (0.5, 0.9, 64), (1.5, 2.0, 60))
    fps = 10.0
    labels = frame_labels(notes, 0.0, 30, fps)
    decoded = decode(labels[0], labels[1], fps, 0.5, 0.5)
    assert sorted(
        zip(decoded.onsets, decoded.offsets, decoded.pitches, strict=True)
    ) == sorted(zip(notes.onsets, notes.offsets, notes.pitches, strict=True))


def test_a_repeated_onset_ends_the_note_before_it() -> None:
    onset = np.zeros((NUM_PITCHES, 10))
    sustain = np.zeros((NUM_PITCHES, 10))
    row = 0
    onset[row, [1, 4]] = 1.0
    sustain[row, 1:9] = 1.0
    decoded = decode(onset, sustain, 1.0, 0.5, 0.5)
    assert decoded.onsets.tolist() == [1.0, 4.0]
    assert decoded.offsets.tolist() == [4.0, 9.0]


def test_perfect_transcription_scores_one() -> None:
    notes = _notes((0.5, 1.2, 60), (1.0, 3.0, 67))
    result = score(notes, notes)
    assert (result.onset, result.onset_offset, result.frame) == (1.0, 1.0, 1.0)


def test_frame_f1_counts_frames_on_the_25_ms_grid() -> None:
    reference = _notes((0.0, 0.1, 60))  # four frames
    estimate = _notes((0.05, 0.15, 60))  # two of them, two more
    assert frame_f1(reference, estimate, 0.025) == pytest.approx(0.5)


# ------------------------------------------------------------------- inference


class _FrameIndexModel(torch.nn.Module):
    """A stand-in whose onset logit at each frame encodes the frame's index.

    ``index % 7 - 3``: small integers survive bf16 autocast exactly, and 7
    shares no factor with the 78 frames a chunk keeps, so a chunk placed even
    one frame off shows.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = HarmonicaModelConfig()

    def spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        frames = num_frames(audio.shape[-1], 512)
        index = torch.arange(frames, dtype=torch.float32)
        return index.view(1, 1, 1, -1).expand(1, 1, 3 * NUM_PITCHES, frames)

    def forward_spectrogram(self, cqt: torch.Tensor) -> dict[str, torch.Tensor]:
        pitched = torch.remainder(cqt[:, 0, ::3, :], 7) - 3
        return {"onset": pitched, "sustain": pitched, "offset": pitched}


@pytest.mark.parametrize("seconds", [1.0, 5.0, 12.3])
def test_half_strategy_stitches_every_frame_once_in_order(seconds: float) -> None:
    samples = int(seconds * 16000)
    onset, _ = transcribe_audio(
        _FrameIndexModel(),
        np.zeros(samples, dtype=np.float32),
        chunk_seconds=5.0,
        batch_size=3,
        device=torch.device("cpu"),
    )
    frames = num_frames(samples, 512)
    assert onset.shape == (NUM_PITCHES, frames)
    logits = np.log(onset[0] / (1 - onset[0]))
    expected = np.arange(frames) % 7 - 3
    np.testing.assert_allclose(logits, expected, atol=1e-3)


def test_half_strategy_pads_with_silence() -> None:
    """Every padded frame reads silence, as a silent clip does."""

    class _PaddingModel(_FrameIndexModel):
        def spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
            frames = num_frames(audio.shape[-1], 512)
            return torch.full((1, 1, 3 * NUM_PITCHES, frames), SILENCE)

        def forward_spectrogram(self, cqt: torch.Tensor) -> dict[str, torch.Tensor]:
            # Any frame of a chunk off the floor lights every kept frame.
            off = (cqt[:, 0, ::3, :] - SILENCE).abs().amax(dim=-1, keepdim=True)
            logits = off.expand(-1, -1, cqt.shape[-1])
            return {"onset": logits, "sustain": logits, "offset": logits}

    onset, _ = transcribe_audio(
        _PaddingModel(),
        np.zeros(int(7.3 * 16000), dtype=np.float32),
        chunk_seconds=5.0,
        batch_size=2,
        device=torch.device("cpu"),
    )
    np.testing.assert_allclose(onset, 0.5, atol=1e-3)


def test_recordings_on_many_devices_come_back_in_order(tmp_path: Path) -> None:
    import soundfile as sf

    from harmonica.config import HarmonicaEvaluationConfig
    from harmonica.inference import run_recordings, run_recordings_on

    rng = np.random.default_rng(0)
    recordings = []
    for k, seconds in enumerate((1.0, 6.5, 2.0, 4.0, 0.5)):
        path = tmp_path / f"{k}.flac"
        sf.write(path, rng.uniform(-0.5, 0.5, int(seconds * 16000)), 16000)
        recordings.append(Recording("x", str(k), "test", str(path), "", seconds))
    model = HarmonicaModel(HarmonicaModelConfig(size="nano")).eval()
    evaluation = HarmonicaEvaluationConfig(batch_size=2)
    cpu = torch.device("cpu")
    alone = run_recordings(model, recordings, evaluation, cpu)
    together = run_recordings_on([(model, cpu)] * 3, recordings, evaluation)
    for one, other in zip(alone, together, strict=True):
        np.testing.assert_array_equal(one.onset, other.onset)
        np.testing.assert_array_equal(one.sustain, other.sustain)
        assert one.audio_seconds == other.audio_seconds


# ---------------------------------------------------------------------- corpus


def test_pedal_holds_a_release_until_it_lifts() -> None:
    notes = [(0.0, 0.5, 60), (0.2, 0.4, 64)]
    pedal = [(0.3, True), (1.0, False)]
    assert extend_by_pedal(notes, pedal) == [(0.0, 1.0, 60), (0.2, 1.0, 64)]


def test_pedal_extension_stops_at_a_restruck_key() -> None:
    notes = [(0.0, 0.2, 60), (0.6, 0.8, 60)]
    pedal = [(0.1, True), (2.0, False)]
    assert extend_by_pedal(notes, pedal) == [(0.0, 0.6, 60), (0.6, 2.0, 60)]


def test_temperature_half_is_square_root_of_hours() -> None:
    weights = source_weights({"a": 100.0, "b": 1.0}, 0.5)
    assert weights["a"] / weights["b"] == pytest.approx(10.0)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_dev_split_takes_each_source_its_weighted_share() -> None:
    recordings = [
        Recording("big", f"b{i}", "train", "", "", 3600.0) for i in range(100)
    ] + [Recording("small", f"s{i}", "train", "", "", 360.0) for i in range(100)]
    config = HarmonicaConfig()
    config.data.dev_hours = 4.0
    dev = select_dev(recordings, config)
    hours = {"big": 0.0, "small": 0.0}
    for r in recordings:
        if f"{r.source}/{r.key}" in dev:
            hours[r.source] += r.hours
    weights = source_weights({"big": 100.0, "small": 10.0}, 0.5)
    assert hours["big"] == pytest.approx(4.0 * weights["big"], abs=1.0)
    assert hours["small"] == pytest.approx(4.0 * weights["small"], abs=0.1)


def test_maps_text_is_onset_offset_pitch() -> None:
    text = "OnsetTime\tOffsetTime\tMidiPitch\n0.5\t1.0\t60\n0.7\t0.9\t20\n"
    notes = maps_notes(text)
    assert (notes.onsets.tolist(), notes.offsets.tolist(), notes.pitches.tolist()) == (
        [0.5],
        [1.0],
        [60],
    )


def test_urmp_notes_are_onset_hz_duration() -> None:
    notes = urmp_notes("1.0\t440.0\t0.5\n")
    assert notes.pitches.tolist() == [69]
    assert notes.offsets.tolist() == [1.5]


def test_guitarset_reads_only_note_midi_annotations() -> None:
    jams = {
        "annotations": [
            {
                "namespace": "pitch_contour",
                "data": [{"time": 0, "duration": 1, "value": 1}],
            },
            {
                "namespace": "note_midi",
                "data": [{"time": 1.0, "duration": 0.5, "value": 52.3}],
            },
        ]
    }
    notes = guitarset_notes(json.dumps(jams).encode())
    assert notes.pitches.tolist() == [52]
    assert notes.offsets.tolist() == [1.5]


# ---------------------------------------------------------------------- config


def test_checked_in_config_is_the_paper_and_names_itself_by_size() -> None:
    config = load_config(CONFIG)
    assert config.name == "harmonica-medium"
    assert (config.training.batch_size, config.training.max_steps) == (8, 100_000)
    assert config.training.learning_rate == 2e-4
    assert config.data.temperature == 0.5
    assert load_config(CONFIG, ["model.size=nano"]).name == "harmonica-nano"


def test_unknown_size_is_refused() -> None:
    with pytest.raises(ValueError, match="model.size"):
        load_config(CONFIG, ["model.size=huge"])


def test_sweep_finds_a_threshold_between_noise_and_notes(tmp_path: Path) -> None:
    from harmonica.config import HarmonicaEvaluationConfig
    from harmonica.inference import Activations, sweep_thresholds

    fps = 10.0
    notes = _notes((1.0, 2.0, 60), (3.0, 4.0, 64))
    notes.save(tmp_path / "n.npz")
    labels = frame_labels(notes, 0.0, 60, fps)
    onset = labels[0] * 0.7
    onset[10:30:2, 15:25] = 0.35  # isolated spurious peaks on other keys
    activations = [Activations(onset, labels[1] * 0.9, 1.0, 6.0)]
    recording = Recording("x", "k", "dev", "", str(tmp_path / "n.npz"), 6.0)
    on, frame, trace = sweep_thresholds(
        [recording], activations, HarmonicaEvaluationConfig(), fps
    )
    assert 0.35 < on <= 0.7
    assert frame <= 0.85
    assert any(row.get("OnP", 0) is None for row in trace)  # pruned, not scored


def test_scoring_in_workers_matches_scoring_here(tmp_path: Path) -> None:
    from harmonica.config import HarmonicaEvaluationConfig
    from harmonica.inference import (
        Activations,
        score_all,
        sweep_thresholds,
    )

    fps = 10.0
    rng = np.random.default_rng(0)
    recordings, activations = [], []
    for k in range(5):
        notes = _notes((1.0, 2.0, 60 + k), (3.0, 4.5, 64), (4.0, 5.0, 40 + k))
        notes.save(tmp_path / f"{k}.npz")
        labels = frame_labels(notes, 0.0, 60, fps)
        noise = rng.uniform(0, 0.6, labels[0].shape)
        activations.append(
            Activations(np.maximum(labels[0] * 0.8, noise), labels[1] * 0.9, 1.0, 6.0)
        )
        recordings.append(
            Recording("x", str(k), "dev", "", str(tmp_path / f"{k}.npz"), 6.0)
        )

    def run(workers: int):
        evaluation = HarmonicaEvaluationConfig(workers=workers)
        swept = sweep_thresholds(recordings, activations, evaluation, fps)
        scores = score_all(recordings, activations, evaluation, fps, 0.5, 0.5)
        return swept, scores

    assert run(3) == run(1)


def test_egdb_holds_out_clips_216_on_in_every_tone(tmp_path: Path) -> None:
    from harmonica.corpus import EGDB_TONES, egdb_jobs

    (tmp_path / "audio_label").mkdir()
    for clip in (1, 215, 216, 240):
        (tmp_path / "audio_label" / f"{clip}.midi").touch()
        for tone in EGDB_TONES:
            (tmp_path / f"audio_{tone}").mkdir(exist_ok=True)
            (tmp_path / f"audio_{tone}" / f"{clip}.wav").touch()
    jobs = list(egdb_jobs(tmp_path))
    assert len(jobs) == 4 * len(EGDB_TONES)
    test = {j.key for j in jobs if j.split == "test"}
    assert test == {f"{t}_{c}" for t in EGDB_TONES for c in (216, 240)}


def test_gaps_uses_its_own_split_column_and_drops_unsplit_rows(
    tmp_path: Path,
) -> None:
    from harmonica.corpus import gaps_jobs

    rows = ["id,audio_path,midi_path,split"]
    for name, split in (("a", "train"), ("b", "test"), ("c", "")):
        (tmp_path / f"{name}.wav").touch()
        (tmp_path / f"{name}.mid").touch()
        rows.append(f"{name},{name}.wav,{name}.mid,{split}")
    (tmp_path / "gaps_metadata_with_splits.csv").write_text("\n".join(rows))
    assert [(j.key, j.split) for j in gaps_jobs(tmp_path)] == [
        ("a", "train"),
        ("b", "test"),
    ]


def test_slakh_reads_redux_only(tmp_path: Path) -> None:
    import yaml

    from harmonica.corpus import slakh_jobs
    from harmonica.slakh_redux import REDUX_OMITTED_TRACKS

    assert len(REDUX_OMITTED_TRACKS) == 2100 - 1710
    stems = {
        "S00": {"inst_class": "Piano", "audio_rendered": True, "midi_saved": True},
        "S01": {"inst_class": "Drums", "is_drum": True, "audio_rendered": True},
        "S02": {"inst_class": "Bass", "audio_rendered": False, "midi_saved": True},
    }
    # Track00049 is the first redux omits.
    for split, number in (("train", 1), ("train", 49), ("test", 1501)):
        track = tmp_path / split / f"Track{number:05d}"
        (track / "stems").mkdir(parents=True)
        (track / "MIDI").mkdir()
        (track / "metadata.yaml").write_text(yaml.safe_dump({"stems": stems}))
        for stem in stems:
            (track / "stems" / f"{stem}.flac").touch()
            (track / "MIDI" / f"{stem}.mid").touch()
    assert [(j.key, j.split) for j in slakh_jobs(tmp_path)] == [
        ("Track00001_S00", "train"),
        ("Track01501_S00", "test"),
    ]


# ------------------------------------------------------------------------- cli


def test_notes_round_trip_through_midi(tmp_path: Path) -> None:
    from harmonica.corpus import notes_from_midi

    notes = _notes((0.5, 1.25, 60), (1.0, 3.0, 67))
    notes.save_midi(tmp_path / "out" / "n.mid")
    back = notes_from_midi((tmp_path / "out" / "n.mid").read_bytes(), False)
    assert back.pitches.tolist() == [60, 67]
    np.testing.assert_allclose(back.onsets, [0.5, 1.0], atol=2e-3)
    np.testing.assert_allclose(back.offsets, [1.25, 3.0], atol=2e-3)


def test_counts_take_a_magnitude() -> None:
    from harmonica.cli import parse_count

    assert [parse_count(v) for v in ("100000", "100k", "1M", "1.5k", "1_000")] == [
        100_000,
        100_000,
        1_000_000,
        1_500,
        1_000,
    ]
    for bad in ("1.5", "1.0001k", "ten"):
        with pytest.raises(ValueError):
            parse_count(bad)


def test_cli_has_prepare_train_infer_eval_export() -> None:
    from typer.testing import CliRunner

    from harmonica.cli import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("prepare", "train", "infer", "eval", "export"):
        assert command in result.output


def test_exported_weights_are_the_checkpoint_without_cqt_kernels(
    tmp_path: Path,
) -> None:
    from harmonica.config import config_to_dict, load_config
    from harmonica.inference import (
        CHECKPOINT_FORMAT_VERSION,
        CQT_KERNELS,
        PIPELINE_NAME,
        export_weights,
        load_checkpoint,
        read_checkpoint,
    )

    config = load_config(CONFIG, ["model.size=nano"])
    model = HarmonicaModel(config.model)
    torch.save(
        {
            "pipeline": PIPELINE_NAME,
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "config": config_to_dict(config),
            "model": {f"_orig_mod.{k}": v for k, v in model.state_dict().items()},
            "optimizer": {},
            "step": 7,
        },
        tmp_path / "full.ckpt",
    )
    export_weights(tmp_path / "full.ckpt", tmp_path / "w.pt", 0.3, 0.4)

    stored = read_checkpoint(tmp_path / "w.pt")
    assert stored["thresholds"] == {"onset": 0.3, "frame": 0.4}
    assert not any(key.startswith(CQT_KERNELS) for key in stored["model"])
    loaded, _, step = load_checkpoint(tmp_path / "w.pt", torch.device("cpu"))
    assert step == 7
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_harmonica_reads_a_file_as_it_reads_the_array(tmp_path: Path) -> None:
    import soundfile as sf

    from harmonica import Harmonica

    rate = 22050
    t = np.arange(rate) / rate
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo = np.stack([audio, audio], axis=1)
    sf.write(tmp_path / "a.wav", stereo, rate, subtype="FLOAT")

    model = Harmonica(device="cpu")
    assert (model.onset_threshold, model.frame_threshold) == (0.25, 0.45)
    from_file = model.activations(tmp_path / "a.wav")
    from_array = model.activations(audio, rate)
    for a, b in zip(from_file, from_array, strict=True):
        assert a.shape == (88, num_frames(16000, 512))
        np.testing.assert_allclose(a, b, atol=1e-3)
    with pytest.raises(ValueError, match="sample_rate"):
        model.activations(audio)


def test_every_bundled_size_is_the_size_it_is_named_for() -> None:
    from harmonica.inference import read_checkpoint
    from harmonica.requests import PRETRAINED, bundled_models

    assert list(PRETRAINED) == list(SIZES)
    assert bundled_models() == list(SIZES)
    for size in bundled_models():
        stored = read_checkpoint(PRETRAINED[size])
        assert stored["config"]["model"]["size"] == size
        assert set(stored["thresholds"]) == {"onset", "frame"}
    medium = read_checkpoint(PRETRAINED["medium"])
    assert medium["config"]["model"]["head"]["bidirectional"] is True
    assert medium["thresholds"] == {"onset": 0.25, "frame": 0.45}
    assert read_checkpoint(PRETRAINED["nano"])["thresholds"] == {
        "onset": 0.2,
        "frame": 0.45,
    }
    assert read_checkpoint(PRETRAINED["small"])["thresholds"] == {
        "onset": 0.2,
        "frame": 0.45,
    }
    assert read_checkpoint(PRETRAINED["large"])["thresholds"] == {
        "onset": 0.25,
        "frame": 0.45,
    }
    assert read_checkpoint(PRETRAINED["x-large"])["thresholds"] == {
        "onset": 0.25,
        "frame": 0.45,
    }


def test_a_model_is_a_size_or_a_file(tmp_path: Path, monkeypatch) -> None:
    from harmonica import requests

    assert requests.resolve_model("medium") == requests.PRETRAINED["medium"]
    (tmp_path / "mine.ckpt").touch()
    assert requests.resolve_model(str(tmp_path / "mine.ckpt")) == tmp_path / "mine.ckpt"
    with pytest.raises(FileNotFoundError, match="x-large"):
        requests.resolve_model("huge")
    # A size not trained yet keeps its slot, and says so.
    monkeypatch.setitem(
        requests.PRETRAINED, "x-large", tmp_path / "harmonica-x-large.pt"
    )
    with pytest.raises(FileNotFoundError, match="x-large: not trained yet"):
        requests.resolve_model("x-large")


def test_thresholds_saved_beside_a_checkpoint_are_found_through_its_link(
    tmp_path: Path,
) -> None:
    from harmonica.config import config_to_dict, load_config
    from harmonica.inference import (
        CHECKPOINT_FORMAT_VERSION,
        PIPELINE_NAME,
        Harmonica,
        export_weights,
        read_checkpoint,
        save_thresholds,
    )

    config = load_config(CONFIG, ["model.size=nano"])
    torch.save(
        {
            "pipeline": PIPELINE_NAME,
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "config": config_to_dict(config),
            "model": HarmonicaModel(config.model).state_dict(),
            "step": 3,
        },
        tmp_path / "checkpoint_3.ckpt",
    )
    (tmp_path / "checkpoint_latest.ckpt").symlink_to("checkpoint_3.ckpt")
    latest = tmp_path / "checkpoint_latest.ckpt"

    assert Harmonica(latest, device="cpu").onset_threshold == 0.5
    with pytest.raises(ValueError, match="harmonica eval"):
        export_weights(latest, tmp_path / "w.pt")

    saved = save_thresholds(latest, 0.3, 0.35)
    assert saved == tmp_path / "checkpoint_3.thresholds.json"
    model = Harmonica(latest, device="cpu", frame_threshold=0.6)
    assert (model.onset_threshold, model.frame_threshold) == (0.3, 0.6)
    export_weights(latest, tmp_path / "w.pt")
    assert read_checkpoint(tmp_path / "w.pt")["thresholds"] == {
        "onset": 0.3,
        "frame": 0.35,
    }


def test_maestro_reads_the_extracted_folder(tmp_path: Path) -> None:
    from harmonica.corpus import maestro_jobs

    rows = {
        "split": {"0": "train", "1": "validation", "2": "test"},
        "audio_filename": {str(i): f"2004/piece{i}.wav" for i in range(3)},
        "midi_filename": {str(i): f"2004/piece{i}.midi" for i in range(3)},
    }
    (tmp_path / "maestro-v3.0.0.json").write_text(json.dumps(rows))
    jobs = list(maestro_jobs(tmp_path, sustain=True))
    assert [(j.key, j.split) for j in jobs] == [
        ("piece0", "train"),
        ("piece1", "train"),
        ("piece2", "test"),
    ]
    assert jobs[2].args == (
        str(tmp_path / "2004/piece2.wav"),
        str(tmp_path / "2004/piece2.midi"),
        True,
    )


def test_goat_takes_di_and_amps_of_aligned_items_only(tmp_path: Path) -> None:
    from harmonica.corpus import GOAT_VARIANTS, goat_jobs

    header = ["item", "split", "finealigned_midi_path", *GOAT_VARIANTS.values()]
    header.append("gp_audio_path")
    lines = [",".join(header)]
    for item, split, aligned in (("a", "train", True), ("b", "test", True)) + (
        ("c", "train", False),
    ):
        folder = tmp_path / "data" / item
        folder.mkdir(parents=True)
        midi = f"GOAT/{item}/{item}_fine_aligned.mid" if aligned else ""
        if aligned:
            (folder / f"{item}_fine_aligned.mid").touch()
        audio = []
        for variant in [*GOAT_VARIANTS, "gp"]:
            (folder / f"{item}_{variant}.wav").touch()
            audio.append(f"GOAT/{item}/{item}_{variant}.wav")
        lines.append(",".join([item, split, midi, *audio]))
    (tmp_path / "metadata.csv").write_text("\n".join(lines))
    jobs = list(goat_jobs(tmp_path))
    assert [(j.key, j.split) for j in jobs] == [
        (f"{item}_{variant}", split)
        for item, split in (("a", "train"), ("b", "test"))
        for variant in GOAT_VARIANTS
    ]
    assert jobs[0].args == (
        str(tmp_path / "data/a/a_di.wav"),
        str(tmp_path / "data/a/a_fine_aligned.mid"),
    )
