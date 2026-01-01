@echo off
rem One-click small-sample check of CURV-TAIL on faithful slices of the two
rem real datasets (DataCon-Website, NUDT-Mobile) (Windows).
rem
rem For each dataset it
rem   * converts the committed feature+payload sample CSV into a runnable mini
rem     cache under data\mini\<dataset> (skipped if already present),
rem   * trains up to 10 epochs (configs\mini_*.yaml -- model identical to the
rem     full recipe, only shorter; it early-stops after 4 epochs without
rem     improvement),
rem   * evaluates the best-validation checkpoint on the mini test split, and
rem   * prints the test metrics.
rem
rem The mini caches reproduce the corresponding rows of the full caches
rem byte-exactly, so a passing run confirms the code and configurations are
rem correct before downloading the full datasets.
setlocal
cd /d "%~dp0.."

if not exist "data\mini\datacon_website\cache_manifest.json" (
  echo -- building data\mini\datacon_website
  python scripts\sample_csv_to_cache.py --csv data\samples\datacon_website_mini.csv --out data\mini\datacon_website --name datacon_website_mini
  if errorlevel 1 exit /b %errorlevel%
)
if not exist "data\mini\nudt_mobile\cache_manifest.json" (
  echo -- building data\mini\nudt_mobile
  python scripts\sample_csv_to_cache.py --csv data\samples\nudt_mobile_mini.csv --out data\mini\nudt_mobile --name nudt_mobile_mini
  if errorlevel 1 exit /b %errorlevel%
)

for %%D in (datacon_website nudt_mobile) do (
  echo == training mini %%D ==
  python -m curv_tail.train --config configs\mini_%%D.yaml --mode train --run-dir outputs\mini_%%D
  if errorlevel 1 exit /b %errorlevel%
  echo == testing mini %%D ==
  python -m curv_tail.train --config configs\mini_%%D.yaml --mode test --checkpoint outputs\mini_%%D\checkpoints\best_macro_f1.pt
  if errorlevel 1 exit /b %errorlevel%
)

echo.
echo Both mini checks finished. Test metrics under outputs\mini_*\test_once\metrics.json
endlocal
