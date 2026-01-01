# Data

This directory holds data *locations* and small, redistributable sample subsets.
It is never committed with the full datasets.

```
data/
├── README.md              # this file
├── samples/               # committed: small faithful sample CSVs, one per dataset
│   ├── datacon_website_mini.csv      (420 flows, 100 classes)
│   └── nudt_mobile_mini.csv          (1260 flows, 300 classes)
├── mini/                  # (generated, gitignored) runnable mini caches
│   ├── datacon_website/   # built from datacon_website_mini.csv
│   └── nudt_mobile/       # built from nudt_mobile_mini.csv
└── full/                  # (gitignored) reserved: full caches downloaded from
                           #   Google Drive, one sub-directory per dataset
```

## Layout of the sample CSVs

Each CSV is a *faithful slice* of the corresponding full data cache. Columns:

| Column | Meaning |
|---|---|
| `sample_id` | Unique flow id (== id in the full cache). |
| `split` | `train` / `val` / `test` (kept from the original dataset split). |
| `label` | Integer class id. |
| `label_name` | Informational class identifier carried through from the source data for traceability; pass `--label-map <label_map.json>` to the exporter to supply human-readable names. It is *not* necessarily equal to `label`: `label` is the contiguous `0..N-1` class index the pipeline requires, whereas `label_name` preserves the source dataset's own identifier (numeric in both shipped samples). The class used for training and metrics always comes from `label`. |
| `num_packets` | Number of valid packets in the flow. |
| `packets` | Per-packet features `"c0,c1;c0,c1;..."`: channel 0 = `signed_log1p(packet_size)`, channel 1 = `log1p(inter-arrival_time)`. |
| `fwd_bytes` / `bwd_bytes` | Payload-byte tokens (byte `b` stored as `b+1`; `0` = padding), comma separated, per direction. |

The mini caches built from these CSVs reproduce the corresponding rows of the
full caches **byte-exactly** (sequences and payload bytes), so a passing mini
run confirms the code and configuration are correct.

## One-click small-sample check

```bash
# Linux / macOS
bash scripts/run_mini.sh
# Windows
scripts\run_mini.bat
```

This converts the two sample CSVs into `data/mini/<dataset>/` caches and
trains + tests each with `configs/mini_*.yaml` (model identical to the full
recipe; short schedule). Expect a few minutes on a GPU.

## Full datasets (Google Drive)

Download the full data caches from Google Drive into `data/full/`, one
sub-directory per dataset (see the main README, section *Getting the datasets*,
for the link and exact file layout).  The shipped full configs
(`configs/datacon_website.yaml`, `configs/nudt_mobile.yaml`) point at
`data/full/<dataset>/` with paths relative
to the repository root: place each downloaded dataset sub-directory under
`data/full/` (gitignored) and run from the repository root.  Each sub-directory
must contain the files listed under *Data cache format* in the main README.  To
keep the caches elsewhere, edit the `data.*` paths in the config (still
relative).

`scripts/export_mini_samples.py` regenerates the sample CSVs (or larger custom
slices) from a downloaded full cache when you want a different sample size.
`--include-bytes` writes the payload-byte columns (the shipped samples were
exported this way); `--label-map <label_map.json>` is optional and only swaps
the `label_name` column for human-readable names -- without it `label_name`
falls back to the numeric label:

```bash
python scripts/export_mini_samples.py \
    --cache-dir data/full/datacon_website --split data/full/datacon_website/split.json \
    --out data/samples/datacon_website_mini.csv --include-bytes
```
