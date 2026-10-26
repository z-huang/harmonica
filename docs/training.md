# Training Harmonica

## Install

```bash
uv sync                              # everything, for training and evaluation
pip install 'harmonica-amt[train]'   # the same, from PyPI
uv run pytest -n auto                # the test suite, CPU only
```

Python 3.12 or later. The bare package only transcribes. `prepare`, `train`,
and `eval` also need the `train` extra, which `uv sync` installs. Training and
evaluation expect a CUDA GPU.

## From corpora to bundled weights

```bash
# Once: index the corpora, cache their audio as 16 kHz FLAC, hold out dev.
harmonica prepare --workers 16

# Train one row of Table 1 (medium by default). The run is named harmonica-<size>.
harmonica train --size nano
harmonica train --size x-large --devices 2

# Sweep thresholds on dev, then score every test set; writes report.json/.md
# and saves the thresholds beside the checkpoint, as checkpoint_N.thresholds.json.
harmonica eval --size nano \
    --checkpoint outputs/harmonica-nano/checkpoints/checkpoint_latest.ckpt

# Transcribe one file, at the thresholds eval saved.
harmonica infer song.wav \
    --model outputs/harmonica-nano/checkpoints/checkpoint_latest.ckpt

# Strip a checkpoint to about 2.6 MB of weights that carry their thresholds;
# this is how src/harmonica/weights/harmonica-medium.pt was made.
harmonica export --checkpoint outputs/harmonica-medium/checkpoints/checkpoint_100000.ckpt \
    --output src/harmonica/weights/harmonica-medium.pt
```

[data.md](data.md) says where each corpus goes and what `prepare` writes, and
[evaluation.md](evaluation.md) covers `harmonica eval` in full: the threshold
sweep, the metrics, the report, and splitting a run across GPUs.

`infer` needs no threshold flags for a checkpoint that `eval` has scored, or
for weights from `export`: it reads the thresholds saved beside them.
`--onset-threshold` and `--frame-threshold` override them, and a checkpoint
that was never evaluated falls back to 0.5, with a warning.

Every size has a slot in `src/harmonica/weights/`: export a trained size to
`harmonica-<size>.pt` there and `--model <size>` picks it up. No code changes.

## Configuration

Every command reads [config/harmonica.yaml](../config/harmonica.yaml).
`--config` picks another file, and `--override dotted.path=value` (repeatable)
changes one field. `--size` is shorthand for `--override model.size=...`.
Table 2's ablations are overrides too:
`--override model.harmonic.kind=hdconv|stacking|none` and
`--override model.harmonic.depth=single`. Give an ablation its own name as well,
for example `--override name=harmonica-medium-hdconv`. Otherwise it saves its
checkpoints over the baseline's.

`harmonica train` resumes from the newest checkpoint of its run by default.
`--resume none` starts fresh, and `--resume PATH` resumes from a given file.
`--max-steps` accepts `100k` or `1m`.

## Velocity

Not in the paper: `--override model.head.velocity=true` adds a fourth head,
the same frequency-grouped LSTM as the other three, that predicts each onset's
MIDI velocity (25K more parameters for medium, 4%). It trains as Kong et al.'s
does, with cross-entropy against `velocity / 127` at onset frames only, then
multiplied by `model.head.velocity_weight` (default 0.01) before it joins the
summed loss—without that scale the sparse-onset mean stays ~0.6 while the
frame-averaged detection losses drop to ~0.01 and the trunk is flooded. Notes
whose velocity is unknown are left out of that loss. A note's velocity is
unknown when its corpus gives none (MAPS's text, GuitarSet, URMP, and the
extra corpora but Guitar-TECHS), or when a MIDI file gives every note the
same one (GAPS, GOAT, and about half of EGDB). That leaves MAESTRO, Slakh,
the rest of EGDB, and Guitar-TECHS. The transcribed MIDI then carries the
velocity read at each onset, where it is otherwise 80 throughout.

Velocities reach the cache's note files only from `harmonica prepare`, so a
cache prepared before them must be prepared again. That rereads the
annotations but keeps the cached audio. Velocity is not scored. Training
still defaults it off; the bundled weights in `src/harmonica/weights/` are
the velocity runs reported in [results.md](results.md).

## The recipe

Training follows Section 4.1 and Appendix D. Each 5 s clip is drawn in three
steps. First a source is picked with probability proportional to
`(hours / total) ** 0.5`. Then a recording from that source is picked
uniformly, and then a window within it, also uniformly. Batches hold 8 clips,
and training runs 100k steps of AdamW at 2e-4 with a cosine schedule to zero.
Every 5k steps the development set is transcribed at thresholds 0.5/0.5, and
the scores are logged to Weights & Biases as `val/OnP`, `val/OnPOff`, and
`val/Frm`. Set `--override wandb.mode=offline` to keep the logs local.
