@echo off
setlocal enabledelayedexpansion

REM build_materials.bat
REM
REM Default (full): run the full offline pipeline (heavy).
REM Optional: pass "quick" to rebuild paper/public materials from what is already computed.
REM Optional: pass "quick-nodocx" to rebuild HTML only (skip DOCX exports).
REM Note: outputs are written under output\private\summary by default.
REM Note: full also rebuilds all paper HTML/DOCX (pmodel_paper + part2/part3 + part4).
REM Note: all modes also emit LaTeX (.tex) for available paper profiles.
REM
REM Usage:
REM   build_materials.bat
REM   build_materials.bat full
REM   build_materials.bat quick
REM   build_materials.bat quick-nodocx
REM   build_materials.bat 1  (Part I only)
REM   build_materials.bat 2  (Part II only)
REM   build_materials.bat 3  (Part III only)
REM   build_materials.bat 4  (Part IV only)
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

echo.
echo === sync_public_readme ===
python -B scripts\summary\sync_public_readme.py --direction public-to-root --bootstrap
if errorlevel 1 goto fail

set "PROFILE="
set "HTML_NAME="
set "DOCX_NAME="
set "PB_EXTRA_ARGS="

if "%MODE%"=="1" goto mode1
if "%MODE%"=="2" goto mode2
if "%MODE%"=="3" goto mode3
if "%MODE%"=="4" goto mode4
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

:mode3
set "PROFILE=part3_quantum"
set "HTML_NAME=pmodel_paper_part3_quantum.html"
set "DOCX_NAME=pmodel_paper_part3_quantum.docx"
set "DOCX_HTML_NAME=pmodel_paper_part3_quantum_docx.html"
goto single_profile

:mode4
set "PROFILE=part4_verification"
set "HTML_NAME=pmodel_paper_part4_verification.html"
set "DOCX_NAME=pmodel_paper_part4_verification.docx"
goto single_profile

:dispatch

if /I "%MODE%"=="full" goto full
if /I "%MODE%"=="quick" goto quick
if /I "%MODE%"=="quick-nodocx" goto quick_nodocx

echo [err] Unknown mode: "%MODE%"
echo [hint] Usage: build_materials.bat [full^|quick^|quick-nodocx]
echo [hint]        build_materials.bat [1^|2^|3^|4]
goto fail

:single_profile
echo [info] Mode=%MODE% (single paper)
echo [info] PROFILE=%PROFILE%
echo [info] ROOT=%ROOT%

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

set "DOCX_HTML_IN=output\private\summary\%HTML_NAME%"
if /I "%PROFILE%"=="part2_astrophysics" (
  echo.
  echo === paper_html docx-friendly %PROFILE% ===
  python -B scripts\summary\paper_html.py --profile %PROFILE% --mode publish --outdir output\private\summary --out-name %DOCX_HTML_NAME% --no-embed-images
  set "DOCX_HTML_IN=output\private\summary\%DOCX_HTML_NAME%"
)
if /I "%PROFILE%"=="part3_quantum" (
  echo.
  echo === paper_html docx-friendly %PROFILE% ===
  python -B scripts\summary\paper_html.py --profile %PROFILE% --mode publish --outdir output\private\summary --out-name %DOCX_HTML_NAME% --no-embed-images
  set "DOCX_HTML_IN=output\private\summary\%DOCX_HTML_NAME%"
)

echo.
echo === docx_paper (%PROFILE%) ===
python -B scripts\summary\html_to_docx.py --in %DOCX_HTML_IN% --out output\private\summary\%DOCX_NAME% --paper-equations --orientation landscape --margin-mm 7 --timeout-s %DOCX_TIMEOUT%
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto :single_docx_done
if "%RC%"=="3" (
  echo [warn] DOCX export skipped: Microsoft Word not available.
  goto :single_docx_done
)
echo [warn] DOCX export failed (%PROFILE%). Continuing with HTML only.
:single_docx_done
if exist output\private\summary\pmodel_paper_part3_quantum__tmp.docx (
  echo [note] pmodel_paper_part3_quantum.docx was locked; updated file is: output\private\summary\pmodel_paper_part3_quantum__tmp.docx
)

echo.
echo [ok] Done (profile=%PROFILE%)
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
python -B scripts\summary\validation_scoreboard.py
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
echo === paper_build (part3_quantum) ===
python -B scripts\summary\paper_build.py --profile part3_quantum --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 (
  echo [warn] paper_build part3_quantum failed; retrying with --skip-tables
  python -B scripts\summary\paper_build.py --profile part3_quantum --mode publish --outdir output\private\summary --skip-docx --skip-lint --skip-tables
  if errorlevel 1 goto fail
)

echo.
echo === paper_build (part4_verification) ===
python -B scripts\summary\paper_build.py --profile part4_verification --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 goto fail

echo.
echo === latex_paper (paper/part2/part3/part4) ===
python -B scripts\summary\paper_latex.py --profile paper --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part2_astrophysics --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part3_quantum --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part4_verification --outdir output\private\summary
if errorlevel 1 goto fail
echo.
echo === paper_tex_audit (paper/part2/part3/part4) ===
python -B scripts\summary\paper_tex_audit.py --profile paper --profile part2_astrophysics --profile part3_quantum --profile part4_verification --outdir output\private\summary
if errorlevel 1 goto fail

echo.
echo === docx_paper (paper) ===
python -B scripts\summary\html_to_docx.py --in output\private\summary\pmodel_paper.html --out output\private\summary\pmodel_paper.docx --paper-equations --orientation landscape --margin-mm 7 --timeout-s %DOCX_TIMEOUT%
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto :full_paper_docx_done
if "%RC%"=="3" (
  echo [warn] DOCX export skipped: Microsoft Word not available.
  goto :full_paper_docx_done
)
echo [warn] DOCX export failed (paper). Continuing with HTML only.
:full_paper_docx_done

echo.
echo === docx_paper (part2_astrophysics) ===
echo === paper_html docx-friendly part2_astrophysics ===
python -B scripts\summary\paper_html.py --profile part2_astrophysics --mode publish --outdir output\private\summary --out-name pmodel_paper_part2_astrophysics_docx.html --no-embed-images
python -B scripts\summary\html_to_docx.py --in output\private\summary\pmodel_paper_part2_astrophysics_docx.html --out output\private\summary\pmodel_paper_part2_astrophysics.docx --paper-equations --orientation landscape --margin-mm 7 --timeout-s %DOCX_TIMEOUT%
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto :full_part2_docx_done
if "%RC%"=="3" (
  echo [warn] DOCX export skipped: Microsoft Word not available.
  goto :full_part2_docx_done
)
echo [warn] DOCX export failed (part2_astrophysics). Continuing with HTML only.
:full_part2_docx_done

echo.
echo === docx_paper (part3_quantum) ===
echo === paper_html docx-friendly part3_quantum ===
python -B scripts\summary\paper_html.py --profile part3_quantum --mode publish --outdir output\private\summary --out-name pmodel_paper_part3_quantum_docx.html --no-embed-images
python -B scripts\summary\html_to_docx.py --in output\private\summary\pmodel_paper_part3_quantum_docx.html --out output\private\summary\pmodel_paper_part3_quantum.docx --paper-equations --orientation landscape --margin-mm 7 --timeout-s %DOCX_TIMEOUT%
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto :full_part3_docx_done
if "%RC%"=="3" (
  echo [warn] DOCX export skipped: Microsoft Word not available.
  goto :full_part3_docx_done
)
echo [warn] DOCX export failed (part3_quantum). Continuing with HTML only.
:full_part3_docx_done
if exist output\private\summary\pmodel_paper_part3_quantum__tmp.docx (
  echo [note] pmodel_paper_part3_quantum.docx was locked; updated file is: output\private\summary\pmodel_paper_part3_quantum__tmp.docx
)

echo.
echo === docx_paper (part4_verification) ===
python -B scripts\summary\html_to_docx.py --in output\private\summary\pmodel_paper_part4_verification.html --out output\private\summary\pmodel_paper_part4_verification.docx --paper-equations --orientation landscape --margin-mm 7 --timeout-s %DOCX_TIMEOUT%
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto :full_part4_docx_done
if "%RC%"=="3" (
  echo [warn] DOCX export skipped: Microsoft Word not available.
  goto :full_part4_docx_done
)
echo [warn] DOCX export failed (part4_verification). Continuing with HTML only.
:full_part4_docx_done

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
echo === paper_build (part3_quantum) ===
python -B scripts\summary\paper_build.py --profile part3_quantum --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 (
  echo [warn] paper_build part3_quantum failed; retrying with --skip-tables
  python -B scripts\summary\paper_build.py --profile part3_quantum --mode publish --outdir output\private\summary --skip-docx --skip-lint --skip-tables
  if errorlevel 1 goto fail
)

echo.
echo === paper_build (part4_verification) ===
python -B scripts\summary\paper_build.py --profile part4_verification --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 goto fail

echo.
echo === latex_paper (paper/part2/part3/part4) ===
python -B scripts\summary\paper_latex.py --profile paper --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part2_astrophysics --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part3_quantum --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part4_verification --outdir output\private\summary
if errorlevel 1 goto fail
echo.
echo === paper_tex_audit (paper/part2/part3/part4) ===
python -B scripts\summary\paper_tex_audit.py --profile paper --profile part2_astrophysics --profile part3_quantum --profile part4_verification --outdir output\private\summary
if errorlevel 1 goto fail

echo.
echo === docx_paper (paper) ===
python -B scripts\summary\html_to_docx.py --in output\private\summary\pmodel_paper.html --out output\private\summary\pmodel_paper.docx --paper-equations --orientation landscape --margin-mm 7 --timeout-s %DOCX_TIMEOUT%
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto :quick_paper_docx_done
if "%RC%"=="3" (
  echo [warn] DOCX export skipped: Microsoft Word not available.
  goto :quick_paper_docx_done
)
echo [warn] DOCX export failed (paper). Continuing with HTML only.
:quick_paper_docx_done

echo.
echo === docx_paper (part2_astrophysics) ===
echo === paper_html docx-friendly part2_astrophysics ===
python -B scripts\summary\paper_html.py --profile part2_astrophysics --mode publish --outdir output\private\summary --out-name pmodel_paper_part2_astrophysics_docx.html --no-embed-images
python -B scripts\summary\html_to_docx.py --in output\private\summary\pmodel_paper_part2_astrophysics_docx.html --out output\private\summary\pmodel_paper_part2_astrophysics.docx --paper-equations --orientation landscape --margin-mm 7 --timeout-s %DOCX_TIMEOUT%
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto :quick_part2_docx_done
if "%RC%"=="3" (
  echo [warn] DOCX export skipped: Microsoft Word not available.
  goto :quick_part2_docx_done
)
echo [warn] DOCX export failed (part2_astrophysics). Continuing with HTML only.
:quick_part2_docx_done

echo.
echo === docx_paper (part3_quantum) ===
echo === paper_html docx-friendly part3_quantum ===
python -B scripts\summary\paper_html.py --profile part3_quantum --mode publish --outdir output\private\summary --out-name pmodel_paper_part3_quantum_docx.html --no-embed-images
python -B scripts\summary\html_to_docx.py --in output\private\summary\pmodel_paper_part3_quantum_docx.html --out output\private\summary\pmodel_paper_part3_quantum.docx --paper-equations --orientation landscape --margin-mm 7 --timeout-s %DOCX_TIMEOUT%
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto :quick_part3_docx_done
if "%RC%"=="3" (
  echo [warn] DOCX export skipped: Microsoft Word not available.
  goto :quick_part3_docx_done
)
echo [warn] DOCX export failed (part3_quantum). Continuing with HTML only.
:quick_part3_docx_done
if exist output\private\summary\pmodel_paper_part3_quantum__tmp.docx (
  echo [note] pmodel_paper_part3_quantum.docx was locked; updated file is: output\private\summary\pmodel_paper_part3_quantum__tmp.docx
)

echo.
echo === docx_paper (part4_verification) ===
python -B scripts\summary\html_to_docx.py --in output\private\summary\pmodel_paper_part4_verification.html --out output\private\summary\pmodel_paper_part4_verification.docx --paper-equations --orientation landscape --margin-mm 7 --timeout-s %DOCX_TIMEOUT%
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto :quick_part4_docx_done
if "%RC%"=="3" (
  echo [warn] DOCX export skipped: Microsoft Word not available.
  goto :quick_part4_docx_done
)
echo [warn] DOCX export failed (part4_verification). Continuing with HTML only.
:quick_part4_docx_done

echo.
echo [ok] Done (quick)
goto ok

:quick_nodocx
echo [info] Mode=quick-nodocx (paper_build only; HTML only)
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
echo === paper_build (part3_quantum) ===
python -B scripts\summary\paper_build.py --profile part3_quantum --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 goto fail

echo.
echo === paper_build (part4_verification) ===
python -B scripts\summary\paper_build.py --profile part4_verification --mode publish --outdir output\private\summary --skip-docx --skip-lint
if errorlevel 1 goto fail

echo.
echo === latex_paper (paper/part2/part3/part4) ===
python -B scripts\summary\paper_latex.py --profile paper --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part2_astrophysics --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part3_quantum --outdir output\private\summary
if errorlevel 1 goto fail
python -B scripts\summary\paper_latex.py --profile part4_verification --outdir output\private\summary
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
