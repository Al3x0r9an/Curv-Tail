#!/usr/bin/env bash
# One-click small-sample check of CURV-TAIL on faithful slices of the two
# real datasets (DataCon-Website, NUDT-Mobile) (Linux / macOS).
#
# For each dataset it
#   * converts the committed feature+payload sample CSV into a runnable mini
#     cache under data/mini/<dataset> (skipped if already present),
#   * trains up to 10 epochs (configs/mini_*.yaml -- model identical to the full
#     recipe, only shorter; it early-stops after 4 epochs without improvement),
#   * evaluates the best-validation checkpoint on the mini test split, and
#   * prints the test metrics.
#
# The mini caches reproduce the corresponding rows of the full caches
# byte-exactly (sequences + payload bytes), so a passing run confirms that the
# code and configurations are correct before downloading the full datasets.
# Runtime is a few minutes on a GPU (a few tens of minutes on CPU).
set -euo pipefail
cd "$(dirname "$0")/.."

build_cache () {
  local csv="$1" out="$2" name="$3"
  if [ ! -f "$out/cache_manifest.json" ]; then
    echo "-- building $out from $csv"
    python scripts/sample_csv_to_cache.py --csv "$csv" --out "$out" --name "$name"
  else
    echo "-- cache already present: $out (delete to rebuild)"
  fi
}

run_dataset () {
  local key="$1"
  build_cache "data/samples/${key}_mini.csv" "data/mini/$key" "${key}_mini"
  echo "== training mini $key =="
  python -m curv_tail.train --config "configs/mini_${key}.yaml" --mode train \
      --run-dir "outputs/mini_${key}"
  echo "== testing mini $key =="
  python -m curv_tail.train --config "configs/mini_${key}.yaml" --mode test \
      --checkpoint "outputs/mini_${key}/checkpoints/best_macro_f1.pt"
}

run_dataset "datacon_website"
run_dataset "nudt_mobile"

echo
echo "Both mini checks finished.  Test metrics under outputs/mini_*/test_once/metrics.json"
