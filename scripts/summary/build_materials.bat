@echo off
setlocal enabledelayedexpansion

REM build_materials.bat
REM
REM Default (full): run the full offline pipeline (heavy).
REM Optional: pass "quick" to rebuild paper/public materials from what is already computed.
REM Optional: pass "quick-nodocx" to rebuild HTML only (skip DOCX exports).
REM Note: quick/quick-nodocx also rebuild the short note HTML (Phase 8 / Step 8.1).
REM Note: full also rebuilds all paper HTML/DOCX (pmodel_paper + part2/part3 + short_note).
REM
REM Usage:
REM   build_materials.bat
REM   build_materials.bat full
REM   build_materials.bat quick
REM   build_materials.bat quick-nodocx

for %%I in ("%~dp0..\\..") do set "ROOT=%%~fI"
pushd "%ROOT%" >nul 2>&1
if errorlevel 1 (
  echo [err] Cannot cd to repo root: "%ROOT%"
  exit /b 1
)

set "MODE=%~1"
if "%MODE%"=="" set "MODE=full"

if /I "%MODE%"=="full" goto full
if /I "%MODE%"=="quick" goto quick
if /I "%MODE%"=="quick-nodocx" goto quick_nodocx

echo [err] Unknown mode: "%MODE%"
echo [hint] Usage: build_materials.bat [full^|quick^|quick-nodocx]
goto fail

:full
echo [info] Mode=full (run_all offline; heavy)
echo [info] ROOT=%ROOT%

echo.
echo === run_all (offline) ===
python -B scripts\summary\run_all.py --offline --jobs 2
if errorlevel 1 goto fail

echo.
echo === paper_build (paper) ===
python -B scripts\summary\paper_build.py --mode publish --outdir output\summary --skip-docx
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
python -B scripts\summary\paper_build.py --profile part2_astrophysics --mode publish --outdir output\summary --skip-docx --skip-tables
if errorlevel 1 goto fail

echo.
echo === paper_build (part3_quantum) ===
python -B scripts\summary\paper_build.py --profile part3_quantum --mode publish --outdir output\summary --skip-docx --skip-tables
if errorlevel 1 goto fail

echo.
echo === paper_build (short_note) ===
python -B scripts\summary\paper_build.py --profile short_note --mode publish --outdir output\summary --skip-docx --skip-tables
if errorlevel 1 goto fail

echo.
echo === docx_paper (paper) ===
python -B scripts\summary\html_to_docx.py --in output\summary\pmodel_paper.html --out output\summary\pmodel_paper.docx --paper-equations --orientation landscape --margin-mm 7
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
python -B scripts\summary\html_to_docx.py --in output\summary\pmodel_paper_part2_astrophysics.html --out output\summary\pmodel_paper_part2_astrophysics.docx --paper-equations --orientation landscape --margin-mm 7
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
python -B scripts\summary\html_to_docx.py --in output\summary\pmodel_paper_part3_quantum.html --out output\summary\pmodel_paper_part3_quantum.docx --paper-equations --orientation landscape --margin-mm 7
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto :full_part3_docx_done
if "%RC%"=="3" (
  echo [warn] DOCX export skipped: Microsoft Word not available.
  goto :full_part3_docx_done
)
echo [warn] DOCX export failed (part3_quantum). Continuing with HTML only.
:full_part3_docx_done
if exist output\summary\pmodel_paper_part3_quantum__tmp.docx (
  echo [note] pmodel_paper_part3_quantum.docx was locked; updated file is: output\summary\pmodel_paper_part3_quantum__tmp.docx
)

echo.
echo === docx_short_note ===
python -B scripts\summary\html_to_docx.py --in output\summary\pmodel_short_note.html --out output\summary\pmodel_short_note.docx --paper-equations --orientation landscape --margin-mm 7
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto :full_short_note_docx_done
if "%RC%"=="3" (
  echo [warn] DOCX export skipped: Microsoft Word not available.
  goto :full_short_note_docx_done
)
echo [warn] DOCX export failed (short_note). Continuing with HTML only.
:full_short_note_docx_done

echo.
echo [ok] Done (full)
goto ok

:quick
echo [info] Mode=quick (paper_build only; assumes outputs already computed)
echo [info] ROOT=%ROOT%

echo.
echo === paper_build (paper) ===
python -B scripts\summary\paper_build.py --mode publish --outdir output\summary --skip-docx
if errorlevel 1 goto fail

echo.
echo === paper_build (part2_astrophysics) ===
python -B scripts\summary\paper_build.py --profile part2_astrophysics --mode publish --outdir output\summary --skip-docx --skip-tables
if errorlevel 1 goto fail

echo.
echo === paper_build (part3_quantum) ===
python -B scripts\summary\paper_build.py --profile part3_quantum --mode publish --outdir output\summary --skip-docx --skip-tables
if errorlevel 1 goto fail

echo.
echo === paper_build (short_note) ===
python -B scripts\summary\paper_build.py --profile short_note --mode publish --outdir output\summary --skip-docx --skip-tables
if errorlevel 1 goto fail

echo.
echo === docx_paper (paper) ===
python -B scripts\summary\html_to_docx.py --in output\summary\pmodel_paper.html --out output\summary\pmodel_paper.docx --paper-equations --orientation landscape --margin-mm 7
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
python -B scripts\summary\html_to_docx.py --in output\summary\pmodel_paper_part2_astrophysics.html --out output\summary\pmodel_paper_part2_astrophysics.docx --paper-equations --orientation landscape --margin-mm 7
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
python -B scripts\summary\html_to_docx.py --in output\summary\pmodel_paper_part3_quantum.html --out output\summary\pmodel_paper_part3_quantum.docx --paper-equations --orientation landscape --margin-mm 7
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto :quick_part3_docx_done
if "%RC%"=="3" (
  echo [warn] DOCX export skipped: Microsoft Word not available.
  goto :quick_part3_docx_done
)
echo [warn] DOCX export failed (part3_quantum). Continuing with HTML only.
:quick_part3_docx_done
if exist output\summary\pmodel_paper_part3_quantum__tmp.docx (
  echo [note] pmodel_paper_part3_quantum.docx was locked; updated file is: output\summary\pmodel_paper_part3_quantum__tmp.docx
)

echo.
echo === docx_short_note ===
python -B scripts\summary\html_to_docx.py --in output\summary\pmodel_short_note.html --out output\summary\pmodel_short_note.docx --paper-equations --orientation landscape --margin-mm 7
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto :quick_short_note_docx_done
if "%RC%"=="3" (
  echo [warn] DOCX export skipped: Microsoft Word not available.
  goto :quick_short_note_docx_done
)
echo [warn] DOCX export failed (short_note). Continuing with HTML only.
:quick_short_note_docx_done

echo.
echo [ok] Done (quick)
goto ok

:quick_nodocx
echo [info] Mode=quick-nodocx (paper_build only; HTML only)
echo [info] ROOT=%ROOT%

echo.
echo === paper_build (paper) ===
python -B scripts\summary\paper_build.py --mode publish --outdir output\summary --skip-docx
if errorlevel 1 goto fail

echo.
echo === paper_build (part2_astrophysics) ===
python -B scripts\summary\paper_build.py --profile part2_astrophysics --mode publish --outdir output\summary --skip-docx --skip-tables
if errorlevel 1 goto fail

echo.
echo === paper_build (part3_quantum) ===
python -B scripts\summary\paper_build.py --profile part3_quantum --mode publish --outdir output\summary --skip-docx --skip-tables
if errorlevel 1 goto fail

echo.
echo === paper_build (short_note) ===
python -B scripts\summary\paper_build.py --profile short_note --mode publish --outdir output\summary --skip-docx --skip-tables
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
