@echo off
rem One-click end-to-end smoke test of CURV-TAIL (Windows).
rem
rem   * generates the tiny synthetic cache (.\demo_cache),
rem   * trains 3 epochs (configs\demo.yaml; ~1 min on CPU, faster on a GPU),
rem   * evaluates the best-validation checkpoint on the held-out split,
rem   * prints the test metrics.
rem
rem Run from anywhere; the script changes into the repository root.
rem No `pip install` is required: `python -m curv_tail.train` resolves the
rem package from the repository root.
setlocal
cd /d "%~dp0.."

echo == [1/3] building the synthetic demo cache ==
python scripts\build_demo_cache.py --out demo_cache --flows 600 --classes 12
if errorlevel 1 exit /b %errorlevel%

echo == [2/3] training (3 epochs, ~1 min on CPU) ==
python -m curv_tail.train --config configs\demo.yaml --mode train --run-dir outputs\demo_run
if errorlevel 1 exit /b %errorlevel%

echo == [3/3] testing the best-validation checkpoint ==
python -m curv_tail.train --config configs\demo.yaml --mode test --checkpoint outputs\demo_run\checkpoints\best_macro_f1.pt
if errorlevel 1 exit /b %errorlevel%

echo.
echo Smoke test finished.  Test metrics:
type outputs\demo_run\test_once\metrics.json
echo.
echo Logs ^& checkpoints live under outputs\demo_run\
endlocal
