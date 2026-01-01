#!/usr/bin/env bash
# One-click end-to-end smoke test of CURV-TAIL (Linux / macOS).
#
#   * generates the tiny synthetic cache (./demo_cache),
#   * trains 3 epochs on CPU (configs/demo.yaml),
#   * evaluates the best-validation checkpoint on the held-out split,
#   * prints the test metrics.
#
# Run from anywhere; it changes into the repository root automatically.
# No `pip install` is required: `python -m curv_tail.train` resolves the
# package from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== [1/3] building the synthetic demo cache =="
python scripts/build_demo_cache.py --out demo_cache --flows 600 --classes 12

echo "== [2/3] training (3 epochs, ~1 min on CPU) =="
python -m curv_tail.train --config configs/demo.yaml --mode train --run-dir outputs/demo_run

echo "== [3/3] testing the best-validation checkpoint =="
python -m curv_tail.train --config configs/demo.yaml --mode test \
    --checkpoint outputs/demo_run/checkpoints/best_macro_f1.pt

echo
echo "Smoke test finished.  Test metrics:"
cat outputs/demo_run/test_once/metrics.json
echo
echo "Logs & checkpoints live under outputs/demo_run/"
