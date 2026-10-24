# Harmonica data

The paper uses eight corpora. Put each one under `data/` (symlinks work),
or point `data.sources.<name>` at it:


| Source          | `data.sources`        | Expected layout                                                                     | Train / test rows | Paper (Table 3)                 |
| --------------- | --------------------- | ----------------------------------------------------------------------------------- | ----------------- | ------------------------------- |
| MAESTRO v3      | `data/maestro-v3.0.0` | extracted: `maestro-v3.0.0.json` and the year folders                               | 1099 / 177        | 1099 / 177                      |
| MAPS            | `data/MAPS`           | `Source/MAPS_ENSTDk*.zip`; ENSTDkAm + ENSTDkCl `MUS` only                           | 0 / 60            | 0 / 60                          |
| GuitarSet       | `data/GuitarSet`      | Zenodo 3371780: `annotation.zip`, `audio_mono-mic.zip`, `audio_mono-pickup_mix.zip` | 480 / 240         | 480 / 240                       |
| GAPS v1.1       | `data/GAPS`           | Hugging Face `xavriley/GAPS`, with `gaps_metadata_with_splits.csv`                  | 270 / 30          | 270 / 30                        |
| EGDB            | `data/EGDB`           | `audio_<tone>/*.wav` for the six tones, `audio_label/*.midi`                        | 1290 / 150        | 1290 / 150                      |
| GOAT            | `data/GOAT`           | Zenodo 15690894 (by request): `metadata.csv`, `data/item_*/`                        | 870 / 42          | 870 / 42                        |
| URMP            | `data/URMP`           | `Dataset/<NN>_<piece>/AuSep_*.wav` + `Notes_*.txt`                                  | 117 / 32          | 117 / 32                        |
| Slakh2100-redux | `data/slakh2100_flac` | the original `train/`, `validation/`, `test/` layout                                | 14,818 / 1,388    | 16,379 / 1,539 (drums included) |


The train counts are before the development set is held out, which is how
Table 3 counts them. A source left at `null` is skipped with a warning.
GOAT uses the DI and five amp renders of every item with a fine-aligned MIDI,
which reproduces Table 3 exactly. Slakh leaves out drum stems, as the paper's
text says, although Table 3 counts them.

## The cache

`harmonica prepare --workers 16` writes its output to `data.cache_dir`
(`data/cache`, about 42 GB, nearly all of it audio: Slakh 28 GB, MAESTRO
9.6 GB):

- `manifest.jsonl`: one row per audio-note pair, with its split;
- `notes/<source>/<key>.npz`: the notes, with their MIDI velocities where the
  annotations carry any (see [training.md](training.md#velocity));
- `audio/<source>/<key>.flac`: 16 kHz mono copies of every recording not
  already 16 kHz mono. MAPS and GuitarSet ship their audio inside zip files,
  so it is always extracted here. The rest are copied because
  `data.cache_audio` is on: training reads random 5 s windows, which costs
  tens of milliseconds a clip from 44.1 kHz FLAC or network storage and two
  from a local 16 kHz FLAC. Put the cache on a local disk;
- `dev.json`: the development set, training recordings held out with the
  same temperature sampling that training uses, to a 4.33 h budget (Table 3).
  Recordings are taken whole, so it comes out slightly over: 110 recordings
  and 4.5 h here, against the paper's 98 and 4.3 h.

## Splits

Splits follow the paper:

- MAESTRO uses its own split, and its validation recordings join training.
- MAPS is test only.
- GuitarSet tests on progression 3, as MT3 does.
- GAPS uses its metadata's split column.
- EGDB tests on clips 216-240 in every tone, following its README.
- URMP tests on MT3's nine pieces.
- Slakh uses redux's split.

GuitarSet's JAMS annotations become MIDI through REMI-z, as in the paper.
Slakh tracks that redux omits are listed in
[slakh_redux.py](../src/harmonica/slakh_redux.py), taken from slakh-utils'
`redux.json`.
