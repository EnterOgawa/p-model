@echo off
setlocal enabledelayedexpansion

REM build_materials.bat
REM
REM Default (full): run the full offline pipeline (heavy).
REM Optional: pass "quick" to rebuild paper/public materials from what is already computed.
REM Optional: pass "quick-nodocx" to rebuild HTML only (skip DOCX exports).
REM Note: outputs are written under output\private\summary by default.
REM Note: full also rebuilds all paper HTML/DOCX (pmodel_paper + part2/part3 + part4).
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

for %%I in ("%~dp0..\\..") do set "ROOT=%%~fI"
pushd "%ROOT%" >nul 2>&1
if errorlevel 1 (
  echo [err] Cannot cd to repo root: "%ROOT%"
  exit /b 1
)

set "MODE=%~1"
if "%MODE%"=="" set "MODE=full"

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
set "PB_EXTRA_ARGS=--skip-tables"
goto single_profile

:mode3
set "PROFILE=part3_quantum"
set "HTML_NAME=pmodel_paper_part3_quantum.html"
set "DOCX_NAME=pmodel_paper_part3_quantum.docx"
set "PB_EXTRA_ARGS=--skip-tables"
goto single_profile

:mode4
set "PROFILE=part4_verification"
set "HTML_NAME=pmodel_paper_part4_verification.html"
set "DOCX_NAME=pmodel_paper_part4_verification.docx"
set "PB_EXTRA_ARGS=--skip-tables"
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
echo === docx_paper (%PROFILE%) ===
python -B scripts\summary\html_to_docx.py --in output\private\summary\%HTML_NAME% --out output\private\summary\%DOCX_NAME% --paper-equations --orientation landscape --margin-mm 7
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
echo === run_all (offline) ===
python -B scripts\summary\run_all.py --offline --jobs 2
if errorlevel 1 goto fail

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
python -B scripts\summary\paper_build.py --profile part2_astrophysics --mode publish --outdir output\private\summary --skip-docx --skip-tables --skip-lint
if errorlevel 1 goto fail

echo.
echo === paper_build (part3_quantum) ===
python -B scripts\summary\paper_build.py --profile part3_quantum --mode publish --outdir output\private\summary --skip-docx --skip-tables --skip-lint
if errorlevel 1 goto fail

echo.
echo === paper_build (part4_verification) ===
python -B scripts\summary\paper_build.py --profile part4_verification --mode publish --outdir output\private\summary --skip-docx --skip-tables --skip-lint
if errorlevel 1 goto fail

echo.
echo === docx_paper (paper) ===
python -B scripts\summary\html_to_docx.py --in output\private\summary\pmodel_paper.html --out output\private\summary\pmodel_paper.docx --paper-equations --orientation landscape --margin-mm 7
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
python -B scripts\summary\html_to_docx.py --in output\private\summary\pmodel_paper_part2_astrophysics.html --out output\private\summary\pmodel_paper_part2_astrophysics.docx --paper-equations --orientation landscape --margin-mm 7
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
python -B scripts\summary\html_to_docx.py --in output\private\summary\pmodel_paper_part3_quantum.html --out output\private\summary\pmodel_paper_part3_quantum.docx --paper-equations --orientation landscape --margin-mm 7
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
python -B scripts\summary\html_to_docx.py --in output\private\summary\pmodel_paper_part4_verification.html --out output\private\summary\pmodel_paper_part4_verification.docx --paper-equations --orientation landscape --margin-mm 7
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
python -B scripts\summary\paper_build.py --profile part2_astrophysics --mode publish --outdir output\private\summary --skip-docx --skip-tables --skip-lint
if errorlevel 1 goto fail

echo.
echo === paper_build (part3_quantum) ===
python -B scripts\summary\paper_build.py --profile part3_quantum --mode publish --outdir output\private\summary --skip-docx --skip-tables --skip-lint
if errorlevel 1 goto fail

echo.
echo === paper_build (part4_verification) ===
python -B scripts\summary\paper_build.py --profile part4_verification --mode publish --outdir output\private\summary --skip-docx --skip-tables --skip-lint
if errorlevel 1 goto fail

echo.
echo === docx_paper (paper) ===
python -B scripts\summary\html_to_docx.py --in output\private\summary\pmodel_paper.html --out output\private\summary\pmodel_paper.docx --paper-equations --orientation landscape --margin-mm 7
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
python -B scripts\summary\html_to_docx.py --in output\private\summary\pmodel_paper_part2_astrophysics.html --out output\private\summary\pmodel_paper_part2_astrophysics.docx --paper-equations --orientation landscape --margin-mm 7
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
python -B scripts\summary\html_to_docx.py --in output\private\summary\pmodel_paper_part3_quantum.html --out output\private\summary\pmodel_paper_part3_quantum.docx --paper-equations --orientation landscape --margin-mm 7
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
python -B scripts\summary\html_to_docx.py --in output\private\summary\pmodel_paper_part4_verification.html --out output\private\summary\pmodel_paper_part4_verification.docx --paper-equations --orientation landscape --margin-mm 7
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
python -B scripts\summary\paper_build.py --profile part2_astrophysics --mode publish --outdir output\private\summary --skip-docx --skip-tables --skip-lint
if errorlevel 1 goto fail

echo.
echo === paper_build (part3_quantum) ===
python -B scripts\summary\paper_build.py --profile part3_quantum --mode publish --outdir output\private\summary --skip-docx --skip-tables --skip-lint
if errorlevel 1 goto fail

echo.
echo === paper_build (part4_verification) ===
python -B scripts\summary\paper_build.py --profile part4_verification --mode publish --outdir output\private\summary --skip-docx --skip-tables --skip-lint
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
