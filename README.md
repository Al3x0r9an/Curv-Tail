# CURV-TAIL

**Curvature-Adaptive Lorentz Temporal Network for Long-Tailed Encrypted Traffic Classification**

CURV-TAIL is a double-branch (multi-view) temporal convolutional network whose per-view
features live on the **Lorentz hyperboloid**. It classifies encrypted TLS/QUIC flows from
their packet-length / inter-arrival sequence *and* their bidirectional payload bytes, and it is
specifically designed to be robust to **long-tailed class distributions**.

This repository contains the complete, self-contained, open-source implementation of the method
only. Benchmark/pre-processing pipelines and unrelated baseline architectures are not included.

---

## Table of contents

1. [Scope](#1-scope)
2. [Repository layout](#2-repository-layout)
3. [Installation](#3-installation)
4. [Data cache format](#4-data-cache-format)
5. [Configuration reference](#5-configuration-reference)
6. [Usage](#6-usage)
7. [Run outputs](#7-run-outputs)
8. [Reproducibility](#8-reproducibility)
9. [Code / API overview](#9-code--api-overview)
10. [Frequently asked questions](#10-frequently-asked-questions)
11. [Getting the datasets (Google Drive) & small-sample checks](#11-getting-the-datasets-google-drive--small-sample-checks)

---

## 1. Scope

This repository contains the open-source implementation of **CURV-TAIL**, a double-branch
temporal convolutional network for long-tailed encrypted-traffic classification (short
summary in the header above).  It ships only this method: no baseline architectures and no
dataset preprocessing pipelines are included.  The network, its discrete length tokenizer and
its Euclidean ablation (`geometry: euclidean`) are defined in `curv_tail/models.py`; the
public API is listed in section 9.

---


## 2. Repository layout

```
curv-tail/
├── pyproject.toml            # packaging; console script `curv-train`
├── requirements.txt          # runtime dependencies
├── README.md
├── LICENSE                   # Apache-2.0
├── .gitignore                # ignores outputs/, demo_cache/, data/mini/, data/full/
├── .gitattributes
├── configs/                  # example YAML configurations
│   ├── datacon_website.yaml  #   full dataset recipes (identical model/training)
│   ├── nudt_mobile.yaml
│   ├── mini_datacon_website.yaml   #   small-sample check recipes
│   ├── mini_nudt_mobile.yaml
│   └── demo.yaml             # tiny synthetic smoke-test recipe
├── data/
│   ├── README.md             # data layout, mini checks, Google Drive link
│   ├── samples/              # committed small faithful sample CSVs (2 datasets)
│   ├── mini/                 # (generated) runnable mini caches (see run_mini)
│   └── full/                 # (reserved) full datasets downloaded from Drive
├── scripts/
│   ├── build_demo_cache.py   # generates the synthetic demo data cache
│   ├── run_demo.sh / run_demo.bat      # one-click synthetic demo
│   ├── export_mini_samples.py          # slice a full cache into a sample CSV
│   ├── sample_csv_to_cache.py          # sample CSV -> runnable mini cache
│   └── run_mini.sh / run_mini.bat      # one-click 2-dataset small-sample check
└── curv_tail/                # the package
    ├── __init__.py
    ├── lorentz.py            # hyperbolic primitives (expmap/logmap/distance/curvature)
    ├── models.py             # CURV-TAIL network + discrete tokenizer
    ├── data.py               # cache readers, datasets, long-tail grouping
    ├── train.py              # training/evaluation CLI (entry point)
    ├── metrics.py            # accuracy / F1 / long-tail group metrics
    ├── config.py             # YAML loading, validation, content hash
    └── utils.py              # RNG, JSONL logging, checkpoint I/O helpers
```

---

## 3. Installation

Requirements: **Python ≥ 3.10** and **PyTorch ≥ 2.1**. Training benefits from an NVIDIA GPU
(`bf16` mixed precision requires an Ampere-generation GPU or newer); everything also runs on CPU
(recommended only for the demo or small-scale experiments).

```bash
# 1) (recommended) create a virtual environment
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
#   .venv\Scripts\activate

# 2) install PyTorch (CPU: omit the --index-url line)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3) install the remaining dependencies and the package itself
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` exposes the `curv-train` console command. If you prefer not to install, all
examples below also work from the repository root with `python -m curv_tail.train` in place of
`curv-train`.

**Windows note.** The loaders default to `num_workers: 0` (synchronous main-process loading),
which is the safe setting on Windows and changes no results: ordering is fixed by the seeded
`EpochRandomSampler`. On Linux/macOS you may raise it (e.g. `num_workers: 8`).

---

## 4. Data cache format

CURV-TAIL consumes **pre-tokenized feature caches**: a directory of NumPy/PyTorch-ready files,
one row per flow. The demo builder (`scripts/build_demo_cache.py`) is a readable reference
implementation of this exact schema.

| File | Contents |
|---|---|
| `cache_manifest.json` | Metadata: `num_samples`, `num_classes`, `max_packets`, `max_bytes_per_direction`, `features`. |
| `manifest.jsonl` | One JSON object per flow: `sample_id`, `label` (int), `label_name`, `sequence_path`. `sample_id` must be globally unique. |
| `split.json` | `{"train": [ids...], "val": [ids...], "test": [ids...]}`. The three sets must be disjoint and, together, equal the manifest exactly. |
| `sequences.npy` | `float32 [N, T, 2]` packet features. Channel 0 = `signed_log1p(packet_size)` (sign = direction), channel 1 = `log1p(inter-arrival_time)`. Rows past the flow's length are `0`. |
| `lengths.npy` | `int64 [N]` number of valid packets per flow (`1 ≤ length ≤ T`). |
| `fwd_bytes.npy` | `uint16 [N, M]` forward payload bytes. Byte `b` stored as `b + 1`; `0` = padding (vocabulary `1..256`). |
| `bwd_bytes.npy` | `uint16 [N, M]` backward payload bytes, same encoding. |
| `labels.npy` | `int64 [N]` integer class of each flow. |
| `sample_ids.txt` | One `sample_id` per line, ordered exactly as the arrays above. |

Constraints enforced at load time:

* **Closed-set training split**: every class must appear at least once in `train`.
* **Label agreement**: cache `labels.npy` and manifest `label` must agree for every flow.
* **Padding convention**: the discrete length token is re-derived from the raw `signed_log1p`
  channel, so sequences are consumed *unstandardized*; do not feed normalized features.

### 4.1 Pointing a config at your caches

All `data.*` paths are **relative to the repository root** (run the commands from there, as every
example in this README does).  The two shipped full-recipe configs point at
`data/full/<dataset>/`, the reserved location described in §11.  Each dataset directory must
contain all files listed above (the manifest and split may live in the same directory as the
cache arrays; the config simply names their paths).  To keep caches elsewhere, edit
`data.cache_dir` / `data.manifest` / `data.split` in the YAML.

---

## 5. Configuration reference

A configuration is one YAML file with the sections `experiment`, `data`, `model`, `training`.
The supervised loss is always cross-entropy (optionally joined by the masked length-token
reconstruction of `model.rec_weight`) and the optimizer is always AdamW; neither is
configurable in this release. The two released recipes share byte-identical `model` and
`training` blocks; only `experiment.name/seed` and `data` change.

### 5.1 `experiment`

| Key | Meaning |
|---|---|
| `name` | Run name (used in auto-generated output dirs). |
| `seed` | Master seed; override on the command line with `--seed`. |
| `output_root` | Directory under which auto-named runs are created. |

### 5.2 `data`

| Key | Meaning |
|---|---|
| `cache_dir` / `manifest` / `split` | Paths from §4, relative to the repository root. |
| `dataset_name` | Informational dataset identifier, recorded in the resolved config. |
| `input_view` | Must be `packet_bytes_multiview` (the double-branch view). |
| `expected_num_samples` | Exact manifest row count; a cheap integrity check. |
| `expected_num_classes` | Number of classes (closed set). |
| `max_packets` | Sequence width `T` (drives the positional embedding size). If omitted, it is read off the cache. |

### 5.3 `model`  (CURV-TAIL hyper-parameters)

| Key | Meaning |
|---|---|
| `family` / `temporal` / `view` | Must be `lorentz_origin_multiview` / `conv` / `multiview` (identity guards). |
| `geometry` | `hyperbolic` = CURV-TAIL (method); `euclidean` = the Euclidean twin ablation. |
| `length_embed` | Must be `true` (exact discrete length tokens). |
| `length_vocab`, `length_block`, `coarse_scale` | Tokenizer: vocab size, byte quantization step, R4 coarse grid scale. Released recipe: `1535 / 1 / 1424`. |
| `dim`, `depth`, `kernel_size`, `dilation_cycle` | Width, block count, kernel width and dilation pattern (`[1, 2, 4]`). |
| `input_dim` | Number of packet-sequence channels (`2`). |
| `dropout` | Dropout probability. |
| `length_embed_dim`, `use_direction`, `use_position` | Length/direction embedding options. |
| `rec_weight` | Weight of the masked length-token reconstruction auxiliary loss (`0` disables it). |
| `byte_embed_dim`, `byte_patch_size`, `max_bytes_per_direction` | Byte-branch stem. `max_bytes_per_direction` must be divisible by `byte_patch_size`. |
| `packet_pooling` | `attention` (learned query pooling) or `mean`. |
| `curvature`, `learnable_curvature`, `min_curvature`, `max_curvature` | Initial value and learnable bounds of `c`. |
| `lift_scale`, `max_tangent_norm` | Origin lift magnitude and the tangent-norm cap used by `expmap_origin`. |
| `prototype_temperature` | Initial temperature of the geodesic classifier. |

### 5.4 `training`

| Key | Meaning |
|---|---|
| `epochs` | Number of epochs (released recipe: `50`). |
| `batch_size`, `eval_batch_size` | Train / eval batch sizes (`1024` / `2048`). |
| `gradient_accumulation` | Gradients accumulated over this many micro-batches. |
| `num_workers` | DataLoader workers (`0` = Windows-safe default). |
| `pin_memory` | Pin host memory in the DataLoaders (default `true`). |
| `persistent_workers`, `prefetch_factor` | Worker lifetime and prefetch depth; only consulted when `num_workers > 0`. |
| `learning_rate` / `weight_decay` | Optimizer (always AdamW) hyper-parameters (`5e-4`, `1e-2`). |
| `warmup_epochs`, `min_learning_rate` | Linear warm-up and cosine end LR (`3`, `5e-6`). |
| `grad_clip_norm` | Gradient-norm clipping (`1.0`). |
| `amp` | Mixed precision: `bf16` (default on GPU), `fp16`, or `none`. |
| `early_stop_patience` | Stop after this many epochs without a better validation macro-F1. |
| `keep_latest` | Number of rolling `latest_epoch_*.pt` snapshots to retain. |

---

## 6. Usage

### 6.1 One-click end-to-end demo (smoke test)

Builds a tiny synthetic cache, trains a small model on CPU for 3 epochs and evaluates it:

```bash
# Linux / macOS
bash scripts/run_demo.sh

# Windows
scripts\run_demo.bat
```

Expect the whole run to finish in about a minute on a laptop CPU.

### 6.2 Training

```bash
# Installed console script (the config path is still repository-relative)
curv-train --config configs/nudt_mobile.yaml --seed 42

# Or, from the repository root without installing:
python -m curv_tail.train --config configs/nudt_mobile.yaml --seed 42

# A fixed output location (instead of the auto-named timestamped directory):
curv-train --config configs/demo.yaml --run-dir outputs/demo_run
```

Training logs one JSON line per epoch to `<run>/logs/epochs.jsonl` (train/val metrics, LR, wall
time) and checkpoints every epoch. Two *selection* checkpoints are kept:
`best_macro_f1.pt` (best validation macro-F1) and `best_tail_macro_f1.pt` (best validation tail
macro-F1).

### 6.3 Evaluation (test mode)

```bash
curv-train --config configs/nudt_mobile.yaml --mode test \
    --checkpoint outputs/<run>/checkpoints/best_macro_f1.pt
```

Test mode runs the checkpoint on the held-out split once and writes
`<run>/test_once/metrics.json` (+ `predictions.npz`, `per_class.csv`). When no `--run-dir` is
given, the run directory is inferred from the checkpoint path, so the run's artifacts are reused.

### 6.4 Resuming and tagging

```bash
# Resume training; the run directory is inferred from the checkpoint path.
curv-train --config configs/demo.yaml --resume outputs/demo_run/checkpoints/latest.pt

# Append a suffix to the auto-generated run name (letters/digits/_/- only).
curv-train --config configs/demo.yaml --tag ablation1
```

A resume continues from the checkpoint's epoch with the optimizer, scheduler and RNG states
restored, so it is equivalent to an uninterrupted run.

### 6.5 Long-tail metrics

Classes are ranked by **training** count and grouped as:

* **head**: the most frequent 50% of classes,
* **body**: the middle 30%,
* **tail**: the least frequent 20%.

Every evaluation reports overall accuracy / balanced accuracy / macro-F1 plus per-group macro-F1
and accuracy, tail precision/recall, tail→head / tail→body / tail→frequent error rates (and the
converse frequent→tail rate) and top-1/3/5/10 hit rates (including tail-only
top-k hit rates and per-k predicted-class coverage). See `curv_tail/metrics.py`.

---

## 7. Run outputs

A training run creates a directory tree:

```
outputs/<name>_seed<S>_<UTC>/
├── resolved_config.yaml        # fully resolved configuration
├── config.sha256               # content hash of the config
├── model_info.json             # device, parameter counts
├── artifacts/
│   ├── train_normalization.json
│   ├── class_counts.json       # per-class training counts
│   ├── class_groups.json       # head/body/tail group per class
│   └── best_macro_val_per_class.csv
├── logs/epochs.jsonl           # per-epoch train/val metrics (JSONL)
├── checkpoints/
│   ├── latest.pt               # latest epoch
│   ├── best_macro_f1.pt        # best validation macro-F1
│   ├── best_tail_macro_f1.pt   # best validation tail macro-F1
│   └── latest_history/latest_epoch_XXXX.pt   # rolling snapshots
└── test_once/                  # written by --mode test
    ├── metrics.json
    ├── predictions.npz         # labels / argmax / top-k arrays
    └── per_class.csv
```

Checkpoints store the full model/optimizer/scheduler state **and** the configuration content hash
plus all RNG states, so a checkpoint can only be resumed or tested against an identical
configuration (a mismatch raises a clear error).

---

## 8. Reproducibility

Every released dataset is trained with the **identical** model and schedule: 50 epochs,
cosine LR `5e-4 → 5e-6`, warm-up 3 epochs, batch 1024, weight decay `1e-2`, gradient
clipping `1.0`, early-stop patience 12, `bf16`.  Three seeds are used per dataset,
`{42, 2027, 3407}` (`--seed S`).

Per-epoch shuffling is deterministic (`EpochRandomSampler`, keyed by `seed + epoch`) and a
resumed run picks up the same random stream (RNG states are stored in the checkpoint), so the
same config, cache and seed reproduce the same data order and trajectory.  Two caveats: the
seed only covers child processes if it is exported before interpreter start, and
`cudnn.benchmark` is enabled, so cuDNN autotuning can select different kernels between runs
and GPU results need not be bit-exact.  Train and then test each seed, e.g. with the
NUDT-Mobile cache at `data/full/nudt_mobile/`:

```bash
for seed in 42 2027 3407; do
  curv-train --config configs/nudt_mobile.yaml --seed "$seed"
  curv-train --config configs/nudt_mobile.yaml --mode test --seed "$seed" \
      --checkpoint "outputs/curv_tail_nudt_mobile_seed${seed}_"*/checkpoints/best_macro_f1.pt
done
```

---

## 9. Code / API overview

| Module | Public surface | Role |
|---|---|---|
| `curv_tail.lorentz` | `minkowski_dot`, `origin_like`, `project_to_tangent`, `tangent_norm`, `clip_tangent`, `expmap`, `logmap`, `distance`, `expmap_origin`, `logmap_origin`, `manifold_error`, `reset_masked_to_origin`, `Curvature` | Hyperbolic primitives on the Lorentz model H^d_c. |
| `curv_tail.models` | `discretize_length`, `PacketStem`, `BytePatchStem`, `OriginTangentTemporalBlock`, `EuclideanTemporalBlock`, `LorentzPrototypeClassifier`, `masked_mean`, `MultiViewLorentzTrafficCNN`, `build_model` | The CURV-TAIL network and discrete tokenizer. |
| `curv_tail.data` | `Record`, `SequenceCache`, `MultiViewCache`, `load_manifest`, `load_split`, `records_by_split`, `load_multiview_cache`, `compute_train_normalization`, `TrafficSequenceDataset`, `MultiViewTrafficDataset`, `training_class_counts`, `cumulative_frequency_groups` | Cache readers, dataset classes and long-tail grouping. |
| `curv_tail.train` | `main`, `run`, `run_dir_for_checkpoint` | Training / evaluation CLI. |
| `curv_tail.metrics` | `compute_metrics`, `per_class_table`, `json_ready` | Metrics incl. head/body/tail aggregates. |
| `curv_tail.config` | `load_config`, `validate_config`, `config_hash`, `public_config` | Config loading, integrity hashing, private-key stripping. |
| `curv_tail.utils` | `seed_everything`, `append_jsonl`, `utc_stamp`, `count_parameters`, `capture_rng_state`, `restore_rng_state`, `atomic_torch_save`, `torch_load_checkpoint`, `rotate_epoch_snapshots` | Shared helpers. |

Each module's `__all__` is authoritative; the table above mirrors it.

---

## 10. Frequently asked questions

* **CPU-only?** Yes: set `amp: none` (the shipped `demo.yaml` does) or simply run without a GPU;
  `bf16` is auto-ignored on CPU by the trainer.
* **My checkpoint will not load.** The configuration content hash must match. Ensure you test or
  resume with the *same* config file used for training.
* **`expected_num_samples` mismatch.** The config declares the exact manifest size as an integrity
  check; update it only if you intentionally use a different cache.
* **What is the runtime / memory?** NUDT-Mobile (~1.3 M flows, 300 classes, dim 128) trains in a
  few hours on a 24 GB GPU with `bf16`; the demo runs in ~1 minute on CPU.

---

## 11. Getting the datasets (Google Drive) & small-sample checks

The two encrypted-traffic benchmarks are public datasets and are distributed
as *pre-built CURV-TAIL caches* from Google Drive (the caches use the exact
on-disk format of §4; their source CSVs and licenses are those of the original
datasets):

* **DataCon-Website-zeek**: `datacon_website/` (85,146 flows, 100 classes)
* **NUDT-Mobile**: `nudt_mobile/` (1,282,557 flows, 300 classes)

### Google Drive link

> Download the full data caches:
> **`https://drive.google.com/...`** (folder shared at release time)
>
> Each dataset is one sub-directory containing `sequences.npy`, `lengths.npy`,
> `fwd_bytes.npy`, `bwd_bytes.npy`, `labels.npy`, `sample_ids.txt`,
> `manifest.jsonl`, `split.json` and `cache_manifest.json`.

Place each downloaded dataset folder at `data/full/<dataset>/` (reserved
path, gitignored) and run from the repository root.  The shipped full
configs already point at these relative paths, so no setup is needed:

```bash
# 1) create the reserved directory and unzip each dataset into data/full/<dataset>/
mkdir -p data/full
#     data/full/datacon_website/{sequences.npy,lengths.npy,fwd_bytes.npy,
#                                bwd_bytes.npy,labels.npy,sample_ids.txt,
#                                manifest.jsonl,split.json,cache_manifest.json}
#     data/full/nudt_mobile/...

# 2) run from the repository root
curv-train --config configs/nudt_mobile.yaml --seed 42
```

To keep the caches in a different location, edit `data.cache_dir` /
`data.manifest` / `data.split` in the YAML to point at them (still relative
paths).

### Small-sample check before downloading the full data

The repository ships small **faithful sample slices** of each dataset as CSVs
(`data/samples/*_mini.csv`, ~420/1260 flows).  A mini cache built from one
of these CSVs reproduces the corresponding rows of the full cache
**byte-exactly** (packet features *and* payload bytes), so a passing run
confirms the code and configuration are correct:

```bash
# one click: build the two mini caches, train + test each (configs/mini_*.yaml)
bash scripts/run_mini.sh            # Windows: scripts\run_mini.bat

# or step by step for one dataset
python scripts/sample_csv_to_cache.py \
    --csv data/samples/datacon_website_mini.csv \
    --out data/mini/datacon_website --name datacon_website_mini
curv-train --config configs/mini_datacon_website.yaml --mode train \
    --run-dir outputs/mini_datacon_website
curv-train --config configs/mini_datacon_website.yaml --mode test \
    --checkpoint outputs/mini_datacon_website/checkpoints/best_macro_f1.pt
```

The mini recipes use the *identical* `model` block as the full recipe (dim 128,
depth 3, R1+R4 discrete tokenization, hyperbolic geometry).  The `training`
blocks differ in scale only -- `epochs` 10 vs 50, `batch_size` 512 vs 1024,
`eval_batch_size` 512 vs 2048, `warmup_epochs` 1 vs 3 and
`early_stop_patience` 4 vs 12.  The `data` blocks keep the same
`expected_num_classes` as their full counterparts (100 and 300) and differ only
in `expected_num_samples`, `dataset_name` and the three cache paths.  Metrics
from mini runs are for smoke-testing only and are not comparable to the released
(full-recipe) numbers.

Sample-CSV schema and how to regenerate a different slice from a downloaded
full cache (`export_mini_samples.py`) are documented in `data/README.md`.

## License and citation

Distributed under the Apache-2.0 license (see `LICENSE`).

A citation entry will be added here once the accompanying paper is publicly available.
