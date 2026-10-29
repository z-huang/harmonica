# Harmonica

[![PyPI](https://img.shields.io/pypi/v/harmonica-amt)](https://pypi.org/project/harmonica-amt/)
[![arXiv](https://img.shields.io/badge/arXiv-2609.04640-b31b1b)](https://arxiv.org/abs/2609.04640)

An independent reproduction of *[Harmonica: Accurate and Lightweight
Instrument-Agnostic Music Transcription](https://arxiv.org/abs/2609.04640)* (Ou,
Martel, Hennessy-Priest, Cho; arXiv:2609.04640). Audio of one pitched instrument
goes in and its notes come out as MIDI, with no instrument label anywhere.

This is not the authors' code. Everything here was rebuilt from the paper.
Where the paper leaves a detail unspecified, we filled it in with our own
assumptions, so this implementation is not guaranteed to match the authors'
exactly. One deliberate addition beyond the paper is a **velocity head**: the
bundled models predict MIDI velocity at each onset, so the transcribed MIDI
carries dynamics instead of a flat default.

## Transcribe a recording

```bash
uvx harmonica-amt song.wav                   # writes song.mid
uvx harmonica-amt song.flac -o out.mid       # choose where the MIDI goes
uvx harmonica-amt song.wav --model x-large   # the most accurate size; medium by default
```

Input can be WAV, FLAC, MP3, or OGG. Inference runs on a GPU if there is one,
otherwise on the CPU.

## Models

`--model` takes one of the sizes below or the path of a checkpoint. Each size
ships with the decoding thresholds its development-set sweep chose. Scores are
the mean **OnP / OnPOff / Frm** over the paper's eight test sets:


| Variant          | Params | Ours               | Paper              |
| ---------------- | ------ | ------------------ | ------------------ |
| nano             | 26.3K  | .800 / .530 / .772 | .808 / .547 / .773 |
| small            | 137K   | .841 / .617 / .809 | .854 / .638 / .815 |
| medium (default) | 679K   | .864 / .661 / .833 | .879 / .687 / .838 |
| large            | 3.2M   | .878 / .687 / .840 | .891 / .713 / .848 |
| x-large          | 15.1M  | .879 / .695 / .846 | .895 / .721 / .852 |


[docs/results.md](docs/results.md) breaks the scores down by test set.

## Python

```python
from harmonica import Harmonica

model = Harmonica()  # the bundled medium, on the GPU if there is one
notes = model.transcribe("song.wav")
notes.save_midi("song.mid")

for onset, offset, pitch in zip(notes.onsets, notes.offsets, notes.pitches):
    print(f"{onset:.2f}-{offset:.2f}s  MIDI {pitch}")
```

The model loads once, so reuse a `Harmonica` for many recordings. It also
takes a NumPy array with its sample rate, `model.transcribe(audio, 44100)`,
and `Harmonica(model, device=..., onset_threshold=..., frame_threshold=...)`
picks another model, device, or thresholds.

## Reproduce the paper

```bash
uv sync                                      # Python 3.12+, a CUDA GPU
source .venv/bin/activate                    # puts harmonica on PATH
harmonica prepare --workers 16               # index the eight corpora
harmonica train --size medium                # one row of the table above
harmonica eval --size medium \
    --checkpoint outputs/harmonica-medium/checkpoints/checkpoint_latest.ckpt
```

- [docs/data.md](docs/data.md): where each corpus goes, the cache, and the splits.
- [docs/training.md](docs/training.md): configuration, ablations, resuming, and
  exporting weights.
- [docs/evaluation.md](docs/evaluation.md): the threshold sweep, the metrics, and
  evaluating on many GPUs.

## Citation

This repository reproduces the following paper; please cite it:

```bibtex
@article{ou2026harmonica,
  title   = {Harmonica: Accurate and Lightweight Instrument-Agnostic Music Transcription},
  author  = {Longshen Ou and H{\'e}ctor Martel and Joe Hennessy-Priest and Taemin Cho},
  journal = {arXiv preprint arXiv:2609.04640},
  year    = {2026}
}
```
