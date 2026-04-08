@echo off
setlocal enabledelayedexpansion

REM build_materials.bat
REM
REM Default (full): run the full offline pipeline (heavy).
REM Optional: pass "quick" to rebuild paper/public materials from what is already computed.
REM Optional: pass "quick-nodocx" to rebuild HTML/PDF only (alias of quick).
REM Note: outputs are written under output\private\summary by default.
REM Note: DOCX auto-generation is disabled by policy; this script rebuilds HTML/PDF only.
REM Note: all modes emit LaTeX (.tex) and PDF for available paper profiles.
REM
REM Usage:
REM   build_materials.bat
REM   build_materials.bat full
REM   build_materials.bat quick
REM   build_materials.bat quick-nodocx
REM   build_materials.bat 1  (Part I only)
REM   build_materials.bat 2  (Part II only)
REM   build_materials.bat 3  (Part III-A + III-B)
REM   build_materials.bat 3a (Part III-A only)
REM   build_materials.bat 3b (Part III-B only)
REM   build_materials.bat 4  (Part IV only)
REM   build_materials.bat 5  (Part V only)
REM
REM Console log:
REM   - Every run writes full console output to:
REM     output\private\summary\logs\build_materials_console_YYYYmmdd_HHMMSS.log

if /I "%~1"=="__run__" (
  shift
  goto main
)

for %%I in ("%~f0") do set "SELF_DIR_BOOT=%%~dpI"
for %%I in ("%SELF_DIR_BOOT%..\\..") do set "ROOT_BOOT=%%~fI"
if not exist "%ROOT_BOOT%\\scripts\\summary\\paper_build.py" (
  for %%I in ("%CD%") do set "ROOT_BOOT=%%~fI"
)
if not exist "%ROOT_BOOT%\\scripts\\summary\\paper_build.py" (
  echo [err] Cannot resolve repo root from "%~f0"
  exit /b 1
)
set "LOG_DIR=%ROOT_BOOT%\output\private\summary\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1
for /f %%I in ('powershell -NoProfile -Command "(Get-Date).ToUniversalTime().ToString(\"yyyyMMdd_HHmmss\")"') do set "LOG_TS=%%I"
set "LOG_FILE=%LOG_DIR%\build_materials_console_%LOG_TS%.log"
set "CMDLINE=""%~f0"" __run__ %*"

echo [info] full console log: "%LOG_FILE%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Continue'; & cmd /d /v:on /c $env:CMDLINE 2>&1 | Tee-Object -FilePath $env:LOG_FILE; exit $LASTEXITCODE"
set "RC=%ERRORLEVEL%"
echo [info] build_materials exit code: %RC%
exit /b %RC%

:main

if defined ROOT_BOOT if exist "%ROOT_BOOT%\\scripts\\summary\\paper_build.py" (
  set "ROOT=%ROOT_BOOT%"
) else (
  for %%I in ("%~f0") do set "SELF_DIR=%%~dpI"
  for %%I in ("%SELF_DIR%..\\..") do set "ROOT=%%~fI"
)
if not exist "%ROOT%\\scripts\\summary\\paper_build.py" (
  echo [err] Cannot resolve repo root: "%ROOT%"
  exit /b 1
)
pushd "%ROOT%" >nul 2>&1
if errorlevel 1 (
  echo [err] Cannot cd to repo root: "%ROOT%"
  exit /b 1
)

set "MODE=%~1"
if "%MODE%"=="" set "MODE=full"

set "DOCX_TIMEOUT=600"
set "PDF_COMMON_ARGS=--engine auto --require-engine --sync-papers --papers-dir papers"

echo.
echo === sync_public_readme ===
python -B scripts\summary\sync_public_readme.py --direction root-to-public
if errorlevel 1 goto fail

set "PROFILE="
set "HTML_NAME="
set "DOCX_NAME="
set "PB_EXTRA_ARGS="

if "%MODE%"=="1" goto mode1
if "%MODE%"=="2" goto mode2
if /I "%MODE%"=="3" goto mode3family
if /I "%MODE%"=="3a" goto mode3a
if /I "%MODE%"=="3b" goto mode3b
if "%MODE%"=="4" goto mode4
if "%MODE%"=="5" goto mode5
goto dispatch

:mode1
set "PROFILE=paper"
set "HTML_NAME=pmodel_paper.html"
set "DOCX_NAME=pmodel_paper.docx"
goto single_profile

:mode2
set "PROFILE=part2_astrophysics"
set "HTML_NAME=pmodel_paper_part2_astrophysics.html"
set "DOCX_NAME=pmodel_paper_part2_astrophysics.docx"
set "DOCX_HTML_NAME=pmodel_paper_part2_astrophysics_docx.html"
goto single_profile

:mode3a
set "PROFILE=part3a_quantum_foundations"
set "HTML_NAME=pmodel_paper_part3a_quantum_foundations.html"
set "DOCX_NAME=pmodel_paper_part3a_quantum_foundations.docx"
set "DOCX_HTML_NAME=pmodel_paper_part3a_quantum_foundations_docx.html"
goto single_profile

:mode3b
set "PROFILE=part3b_quantum_verification"
set "HTML_NAME=pmodel_paper_part3b_quantum_verification.html"
set "DOCX_NAME=pmodel_paper_part3b_quantum_verification.docx"
set "DOCX_HTML_NAME=pmodel_paper_part3b_quantum_verification_docx.html"
goto single_profile

:mode4
set "PROFILE=part4_verification"
set "HTML_NAME=pmodel_paper_part4_verification.html"
set "DOCX_NAME=pmodel_paper_part4_verification.docx"
goto single_profile

:mode5
set "PROFILE=part5_future_predictions"
set "HTML_NAME=pmodel_paper_part5_future_predictions.html"
set "DOCX_NAME=pmodel_paper_part5_future_predictions.docx"
goto single_profile

:dispatch

if /I "%MODE%"=="full" goto full
if /I "%MODE%"=="quick" goto quick
if /I "%MODE%"=="quick-nodocx" goto quick_nodocx

echo [err] Unknown mode: "%MODE%"
echo [hint] Usage: build_materials.bat [full^|quick^|quick-nodocx]
echo [hint]        build_materials.bat [1^|2^|3^|3a^|3b^|4^|5]
goto fail

:single_profile
echo [info] Mode=%MODE% (single paper)
echo [info] PROFILE=%PROFILE%
echo [info] ROOT=%ROOT%

if /I "%PROFILE%"=="part4_verification" goto single_profile_part4_direct

echo.
echo === paper_build (%PROFILE%) ===
python -B scripts\summary\paper_build.py --profile %PROFILE% --mode publish --outdir output\private\summary --skip-docx --skip-lint %PB_EXTRA_ARGS%
if errorlevel 1 goto fail

echo.
echo === latex_paper (%PROFILE%) ===
python -B scripts\summary\paper_latex.py --profile %PROFILE% --outdir output\private\summary
if errorlevel 1 goto fail
echo.
echo === paper_tex_audit (%PROFILE%) ===
python -B scripts\summary\paper_tex_audit.py --profile %PROFILE% --outdir output\private\summary
if errorlevel 1 goto fail
echo.
echo === paper_pdf (%PROFILE%) ===
python -B scripts\summary\paper_pdf.py --profile %PROFILE% --outdir output\private\summary %PDF_COMMON_ARGS%
if errorlevel 1 goto fail

set "DOCX_HTML_IN=output\private\summary\%HTML_NAME%"
echo.
echo [info] DOCX auto-generation is disabled by policy. Skipping DOCX export for %PROFILE%.
:single_profile_after_docx

echo.
echo [ok] Done (profile=%PROFILE%)
goto ok

:single_profile_part4_direct
echo.
echo === part4 direct rerender default ===
echo [info] Part IV is rerendered from existing canonical artifacts by default.
echo [info] Use paper_build.py explicitly only when upstream refresh is intended.
set "WAVEP_FIGURE_LANG=ja"
set "WAVEP_MPL_FORCE_JA_TEXT=1"

echo.
echo === paper_lint (%PROFILE%) ===
python -B scripts\summary\paper_lint.py --manuscript doc\paper\13_part4_verification.md
if errorlevel 1 goto fail

echo.
echo === paper_html (%PROFILE%) ===
python -B scripts\summary\paper_html.py --profile %PROFILE% --mode publish --outdir output\private\summary
if errorlevel 1 goto fail

echo.
echo === paper_latex (%PROFILE%) ===
python -B scripts\summary\paper_latex.py --profile %PROFILE% --outdir output\private\summary
if errorlevel 1 goto fail

echo.
echo === paper_tex_audit (%PROFILE%) ===
python -B scripts\summary\paper_tex_audit.py --profile %PROFILE% --outdir output\private\summary
if errorlevel 1 goto fail

echo.
echo === paper_pdf (%PROFILE%) ===
python -B scripts\summary\paper_pdf.py --profile %PROFILE% --outdir output\private\summary %PDF_COMMON_ARGS%
if errorlevel 1 goto fail

echo.
echo [ok] Done (profile=%PROFILE%, route=direct-rerender)
goto ok

:mode3family
echo [info] Mode=3 (Part III-A + Part III-B)
echo [info] ROOT=%ROOT%

echo.
echo === paper_build (part3a_quantum_foundations) ===
python -B scripts\summary\paper_build.py --profile part3a_quantum_foundations --figure-lang ja --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 goto fail

echo.
echo === paper_build (part3b_quantum_verification) ===
python -B scripts\summary\paper_build.py --profile part3b_quantum_verification --figure-lang ja --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 (
  echo [warn] paper_build part3b_quantum_verification failed; retrying with --skip-tables
  python -B scripts\summary\paper_build.py --profile part3b_quantum_verification --figure-lang ja --mode publish --outdir output\private\summary --skip-docx --skip-lint --skip-tables
  if errorlevel 1 goto fail
)

echo.
echo === latex_paper (part3a_quantum_foundations / part3b_quantum_verification) ===
python -B scripts\summary\paper_latex.py --profile part3a_quantum_foundations --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part3b_quantum_verification --outdir output\private\summary
if errorlevel 1 goto fail
echo.
echo === paper_tex_audit (part3a_quantum_foundations / part3b_quantum_verification) ===
python -B scripts\summary\paper_tex_audit.py --profile part3a_quantum_foundations --profile part3b_quantum_verification --outdir output\private\summary
if errorlevel 1 goto fail
echo.
echo === paper_pdf (part3a_quantum_foundations / part3b_quantum_verification) ===
python -B scripts\summary\paper_pdf.py --profile part3a_quantum_foundations --profile part3b_quantum_verification --outdir output\private\summary %PDF_COMMON_ARGS%
if errorlevel 1 goto fail

echo.
echo [ok] Done (Part III-A + Part III-B)
goto ok

:full
echo [info] Mode=full (run_all offline; heavy)
echo [info] ROOT=%ROOT%

echo.
REM Warm-up: LLR time-tag auto selection needs Horizons cache; otherwise Part II figures become placeholders.
if not exist output\private\llr\horizons_cache\horizons_vectors_301_*.csv (
  echo === llr_batch_eval (online warm-cache for Horizons) ===
  python -B scripts\llr\llr_batch_eval.py --time-tag-mode auto --min-points 30 --chunk 50
  if errorlevel 1 (
    echo [warn] llr_batch_eval warm-cache failed; continuing...
  )
  echo.
)

echo.
echo === run_all (offline) ===
python -B scripts\summary\run_all.py --offline --jobs 2
set "RUN_ALL_RC=!ERRORLEVEL!"
if not "!RUN_ALL_RC!"=="0" (
  echo [warn] run_all returned rc=!RUN_ALL_RC! ; common causes: paper_lint or missing outputs. Continuing...
)

echo.
echo === paper_build (paper) ===
python -B scripts\summary\paper_build.py --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 goto fail

echo.
echo === validation_scoreboard ===
python -B scripts\summary\validation_scoreboard.py --target-fig-h-in 9.2
if errorlevel 1 goto fail

echo.
echo === quantum_scoreboard ===
python -B scripts\summary\quantum_scoreboard.py
if errorlevel 1 goto fail

echo.
echo === paper_build (part2_astrophysics) ===
python -B scripts\summary\paper_build.py --profile part2_astrophysics --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 (
  echo [warn] paper_build part2_astrophysics failed; retrying with --skip-tables
  python -B scripts\summary\paper_build.py --profile part2_astrophysics --mode publish --outdir output\private\summary --skip-docx --skip-lint --skip-tables
  if errorlevel 1 goto fail
)

echo.
echo === paper_build (part3a_quantum_foundations) ===
python -B scripts\summary\paper_build.py --profile part3a_quantum_foundations --figure-lang ja --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 (
  echo [warn] paper_build part3a_quantum_foundations failed; retrying with --skip-tables
  python -B scripts\summary\paper_build.py --profile part3a_quantum_foundations --figure-lang ja --mode publish --outdir output\private\summary --skip-docx --skip-lint --skip-tables
  if errorlevel 1 goto fail
)

echo.
echo === paper_build (part3b_quantum_verification) ===
python -B scripts\summary\paper_build.py --profile part3b_quantum_verification --figure-lang ja --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 (
  echo [warn] paper_build part3b_quantum_verification failed; retrying with --skip-tables
  python -B scripts\summary\paper_build.py --profile part3b_quantum_verification --figure-lang ja --mode publish --outdir output\private\summary --skip-docx --skip-lint --skip-tables
  if errorlevel 1 goto fail
)

echo.
echo === paper_build (part4_verification) ===
python -B scripts\summary\paper_build.py --profile part4_verification --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 goto fail

echo.
echo === paper_build (part5_future_predictions) ===
python -B scripts\summary\paper_build.py --profile part5_future_predictions --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 goto fail

echo.
echo === latex_paper (paper/part2/part3a/part3b/part4/part5) ===
python -B scripts\summary\paper_latex.py --profile paper --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part2_astrophysics --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part3a_quantum_foundations --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part3b_quantum_verification --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part4_verification --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part5_future_predictions --outdir output\private\summary
if errorlevel 1 goto fail
echo.
echo === paper_tex_audit (paper/part2/part3a/part3b/part4/part5) ===
python -B scripts\summary\paper_tex_audit.py --profile paper --profile part2_astrophysics --profile part3a_quantum_foundations --profile part3b_quantum_verification --profile part4_verification --profile part5_future_predictions --outdir output\private\summary
if errorlevel 1 goto fail
echo.
echo === paper_pdf (paper/part2/part3a/part3b/part4/part5) ===
python -B scripts\summary\paper_pdf.py --profile paper --profile part2_astrophysics --profile part3a_quantum_foundations --profile part3b_quantum_verification --profile part4_verification --profile part5_future_predictions --outdir output\private\summary %PDF_COMMON_ARGS%
if errorlevel 1 goto fail

echo.
echo [info] DOCX auto-generation is disabled by policy. Skipping DOCX exports (full mode).
:full_after_docx

echo.
echo [ok] Done (full)
goto ok

:quick
echo [info] Mode=quick (paper_build only; assumes outputs already computed)
echo [info] ROOT=%ROOT%

echo.
echo === paper_build (paper) ===
python -B scripts\summary\paper_build.py --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 goto fail

echo.
echo === paper_build (part2_astrophysics) ===
python -B scripts\summary\paper_build.py --profile part2_astrophysics --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 (
  echo [warn] paper_build part2_astrophysics failed; retrying with --skip-tables
  python -B scripts\summary\paper_build.py --profile part2_astrophysics --mode publish --outdir output\private\summary --skip-docx --skip-lint --skip-tables
  if errorlevel 1 goto fail
)

echo.
echo === paper_build (part3a_quantum_foundations) ===
python -B scripts\summary\paper_build.py --profile part3a_quantum_foundations --figure-lang ja --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 (
  echo [warn] paper_build part3a_quantum_foundations failed; retrying with --skip-tables
  python -B scripts\summary\paper_build.py --profile part3a_quantum_foundations --figure-lang ja --mode publish --outdir output\private\summary --skip-docx --skip-lint --skip-tables
  if errorlevel 1 goto fail
)

echo.
echo === paper_build (part3b_quantum_verification) ===
python -B scripts\summary\paper_build.py --profile part3b_quantum_verification --figure-lang ja --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 (
  echo [warn] paper_build part3b_quantum_verification failed; retrying with --skip-tables
  python -B scripts\summary\paper_build.py --profile part3b_quantum_verification --figure-lang ja --mode publish --outdir output\private\summary --skip-docx --skip-lint --skip-tables
  if errorlevel 1 goto fail
)

echo.
echo === paper_build (part4_verification) ===
python -B scripts\summary\paper_build.py --profile part4_verification --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 goto fail

echo.
echo === paper_build (part5_future_predictions) ===
python -B scripts\summary\paper_build.py --profile part5_future_predictions --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 goto fail

echo.
echo === latex_paper (paper/part2/part3a/part3b/part4/part5) ===
python -B scripts\summary\paper_latex.py --profile paper --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part2_astrophysics --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part3a_quantum_foundations --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part3b_quantum_verification --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part4_verification --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part5_future_predictions --outdir output\private\summary
if errorlevel 1 goto fail
echo.
echo === paper_tex_audit (paper/part2/part3a/part3b/part4/part5) ===
python -B scripts\summary\paper_tex_audit.py --profile paper --profile part2_astrophysics --profile part3a_quantum_foundations --profile part3b_quantum_verification --profile part4_verification --profile part5_future_predictions --outdir output\private\summary
if errorlevel 1 goto fail
echo.
echo === paper_pdf (paper/part2/part3a/part3b/part4/part5) ===
python -B scripts\summary\paper_pdf.py --profile paper --profile part2_astrophysics --profile part3a_quantum_foundations --profile part3b_quantum_verification --profile part4_verification --profile part5_future_predictions --outdir output\private\summary %PDF_COMMON_ARGS%
if errorlevel 1 goto fail

echo.
echo [info] DOCX auto-generation is disabled by policy. Skipping DOCX exports (quick mode).
:quick_after_docx

echo.
echo [ok] Done (quick)
goto ok

:quick_nodocx
echo [info] Mode=quick-nodocx (paper_build only; HTML/PDF only)
echo [info] ROOT=%ROOT%

echo.
echo === paper_build (paper) ===
python -B scripts\summary\paper_build.py --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 goto fail

echo.
echo === paper_build (part2_astrophysics) ===
python -B scripts\summary\paper_build.py --profile part2_astrophysics --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 goto fail

echo.
echo === paper_build (part3a_quantum_foundations) ===
python -B scripts\summary\paper_build.py --profile part3a_quantum_foundations --figure-lang ja --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 goto fail

echo.
echo === paper_build (part3b_quantum_verification) ===
python -B scripts\summary\paper_build.py --profile part3b_quantum_verification --figure-lang ja --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 goto fail

echo.
echo === paper_build (part4_verification) ===
python -B scripts\summary\paper_build.py --profile part4_verification --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 goto fail

echo.
echo === paper_build (part5_future_predictions) ===
python -B scripts\summary\paper_build.py --profile part5_future_predictions --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 goto fail

echo.
echo === latex_paper (paper/part2/part3a/part3b/part4/part5) ===
python -B scripts\summary\paper_latex.py --profile paper --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part2_astrophysics --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part3a_quantum_foundations --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part3b_quantum_verification --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part4_verification --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part5_future_predictions --outdir output\private\summary
if errorlevel 1 goto fail

echo.
echo === paper_tex_audit (paper/part2/part3a/part3b/part4/part5) ===
python -B scripts\summary\paper_tex_audit.py --profile paper --profile part2_astrophysics --profile part3a_quantum_foundations --profile part3b_quantum_verification --profile part4_verification --profile part5_future_predictions --outdir output\private\summary
if errorlevel 1 goto fail

echo.
echo === paper_pdf (paper/part2/part3a/part3b/part4/part5) ===
python -B scripts\summary\paper_pdf.py --profile paper --profile part2_astrophysics --profile part3a_quantum_foundations --profile part3b_quantum_verification --profile part4_verification --profile part5_future_predictions --outdir output\private\summary %PDF_COMMON_ARGS%
if errorlevel 1 goto fail

echo.
echo [ok] Done (quick-nodocx)
goto ok

:fail
popd >nul 2>&1
echo.
echo [err] Failed.
exit /b 1

:ok
popd >nul 2>&1
exit /b 0
