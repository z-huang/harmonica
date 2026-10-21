# Harmonica

An independent reproduction of *[Harmonica: Accurate and Lightweight
Instrument-Agnostic Music Transcription](https://arxiv.org/abs/2609.04640)* (Ou,
Martel, Hennessy-Priest, Cho; arXiv:2609.04640). Audio of one pitched instrument
goes in and its notes come out as MIDI, with no instrument label anywhere.

This is not the authors' code. Everything here was rebuilt from the paper.
Where the paper leaves a detail unspecified, we filled it in with our own
assumptions, so this implementation is not guaranteed to match the authors'
exactly.

## Transcribe a recording

```bash
uvx harmonica-amt infer song.wav                 # writes song.mid
uvx harmonica-amt infer song.flac -o out.mid     # choose where the MIDI goes
uvx harmonica-amt infer song.wav --model my-run/checkpoint_latest.ckpt  # your own model
```

Input can be WAV, FLAC, MP3, or OGG. Inference runs on a GPU if there is one,
otherwise on the CPU. Until the package is on PyPI, run it from GitHub:
`uvx --from git+https://github.com/z-huang/harmonica harmonica infer song.wav`.
(PyPI's `harmonica` is a geophysics package, hence `harmonica-amt`; once
installed, the command is available as both.)

## Models

`--model` takes one of Table 1's sizes or the path of a checkpoint. Each size
ships with the decoding thresholds its development-set sweep chose. Scores are
the mean **OnP / OnPOff / Frm** over the paper's eight test sets:


| Variant          | Params | Ours               | Paper              |
| ---------------- | ------ | ------------------ | ------------------ |
| nano             | 25K    | .791 / .524 / .763 | .808 / .547 / .773 |
| small            | 131K   | .835 / .603 / .804 | .854 / .638 / .815 |
| medium (default) | 654K   | .862 / .660 / .827 | .879 / .687 / .838 |
| large            | 3.1M   | .877 / .688 / .843 | .891 / .713 / .848 |
| x-large          | 14.7M  | .882 / .694 / .843 | .895 / .721 / .852 |


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
harmonica prepare --workers 16               # index the eight corpora
harmonica train --size medium                # one row of Table 1
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
  author  = {Ou, Longshen and Martel, H{\'e}ctor and Hennessy-Priest, Joe and Cho, Taemin},
  journal = {arXiv preprint arXiv:2609.04640},
  year    = {2026}
}
```
