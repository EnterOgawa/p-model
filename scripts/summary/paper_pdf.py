#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_pdf.py

Build PDF files from generated TeX files, apply safe auto-fixes for common
compile warnings/errors, and optionally sync PDFs into local `papers/`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
# Condition: ensure repository root is importable.
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog

_PROFILE_TO_TEX: Dict[str, str] = {
    "paper": "pmodel_paper.tex",
    "part2_astrophysics": "pmodel_paper_part2_astrophysics.tex",
    "part3_quantum": "pmodel_paper_part3_quantum.tex",
    "part4_verification": "pmodel_paper_part4_verification.tex",
    "part5_future_predictions": "pmodel_paper_part5_future_predictions.tex",
}

_PROFILE_TO_PDF: Dict[str, str] = {
    "paper": "pmodel_paper.pdf",
    "part2_astrophysics": "pmodel_paper_part2_astrophysics.pdf",
    "part3_quantum": "pmodel_paper_part3_quantum.pdf",
    "part4_verification": "pmodel_paper_part4_verification.pdf",
    "part5_future_predictions": "pmodel_paper_part5_future_predictions.pdf",
}

_FATAL_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"Undefined control sequence"),
    re.compile(r"Missing \$ inserted"),
    re.compile(r"Double subscript"),
    re.compile(r"Extra }, or forgotten \$"),
    re.compile(r"Missing } inserted"),
    re.compile(r"Runaway argument\?"),
    re.compile(r"! LaTeX Error:"),
    re.compile(r"Emergency stop"),
    re.compile(r"Fatal error"),
]

_LABEL_MULTI_WARN_RE = re.compile(r"Label `[^`]+` multiply defined")
_OVERFULL_WARN_RE = re.compile(r"Overfull \\hbox")
_UNDEFINED_REFS_RE = re.compile(r"There were undefined references")
_UNDEFINED_CIT_RE = re.compile(r"Citation `[^`]+` .* undefined")
_RERUN_RE = re.compile(r"Rerun to get cross-references right")
_LABEL_CHANGED_RE = re.compile(r"Label\(s\) may have changed")
_LABEL_RE = re.compile(r"\\label\{([^{}]+)\}")


# Function: return current UTC timestamp in ISO8601.
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: resolve latex engine binary path from PATH/env/known install dirs.

def _find_engine_binary(engine: str) -> str | None:
    from_path = shutil.which(engine)
    if from_path:
        return from_path

    env_candidates: List[str] = []
    for key in (
        f"WAVEP_{engine.upper()}_PATH",
        f"TEX_{engine.upper()}_PATH",
        "WAVEP_TEX_BIN",
        "TEXLIVE_BIN",
        "MIKTEX_BIN",
    ):
        value = os.environ.get(key, "").strip()
        if not value:
            continue

        p = Path(value)
        if p.suffix.lower() == ".exe":
            env_candidates.append(str(p))
        else:
            env_candidates.append(str(p / f"{engine}.exe"))

    local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
    common_candidates = [
        Path(rf"C:\Program Files\MiKTeX\miktex\bin\x64\{engine}.exe"),
        Path(rf"C:\Program Files\MiKTeX\miktex\bin\{engine}.exe"),
        Path(rf"C:\Program Files (x86)\MiKTeX\miktex\bin\x64\{engine}.exe"),
    ]
    if local_appdata:
        common_candidates.append(Path(local_appdata) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64" / f"{engine}.exe")

    common_candidates.extend(sorted(Path(r"C:\texlive").glob(rf"*\bin\windows\{engine}.exe"), reverse=True))
    common_candidates.extend(sorted(Path(r"C:\Program Files\texlive").glob(rf"*\bin\windows\{engine}.exe"), reverse=True))

    for candidate in env_candidates + [str(p) for p in common_candidates]:
        try:
            cp = Path(candidate)
            if cp.exists():
                return str(cp)
        except Exception:
            continue

    return None


# Function: pick latex engine binary according to CLI choice.

def _pick_engine(choice: str) -> Tuple[str | None, str]:
    if choice != "auto":
        found = _find_engine_binary(choice)
        return (found, "requested" if found else "missing")

    for eng in ("lualatex", "xelatex", "pdflatex"):
        found = _find_engine_binary(eng)
        if found:
            return found, "auto"

    return None, "missing"


# Function: extract compile errors/warnings from a latex log text.

def _collect_issues(log_text: str, returncode: int, *, include_overfull_warning: bool) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    for pat in _FATAL_PATTERNS:
        if pat.search(log_text):
            errors.append(f"compile:{pat.pattern}")

    if _LABEL_MULTI_WARN_RE.search(log_text):
        warnings.append("warn:label_multiply_defined")

    if include_overfull_warning and _OVERFULL_WARN_RE.search(log_text):
        warnings.append("warn:overfull_hbox")

    if _UNDEFINED_REFS_RE.search(log_text):
        warnings.append("warn:undefined_references")

    if _UNDEFINED_CIT_RE.search(log_text):
        warnings.append("warn:undefined_citations")

    if returncode != 0:
        errors.append(f"compile_rc={returncode}")

    return sorted(set(errors)), sorted(set(warnings))


# Function: determine whether another latex pass is required.

def _needs_rerun(log_text: str) -> bool:
    if _RERUN_RE.search(log_text):
        return True

    if _LABEL_CHANGED_RE.search(log_text):
        return True

    if _UNDEFINED_REFS_RE.search(log_text):
        return True

    if _UNDEFINED_CIT_RE.search(log_text):
        return True

    return False


# Function: run one latex compile pass and return log + produced PDF path.

def _run_latex_once(
    *,
    engine: str,
    tex_path: Path,
    build_dir: Path,
    logs_dir: Path,
    profile: str,
    round_idx: int,
    pass_idx: int,
) -> Dict[str, Any]:
    build_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        engine,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "-synctex=0",
        f"-output-directory={str(build_dir)}",
        str(tex_path.resolve()),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(tex_path.parent.resolve()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_text = proc.stdout or ""
    log_path = logs_dir / f"paper_pdf_{profile}_r{round_idx}_p{pass_idx}.log"
    log_path.write_text(log_text, encoding="utf-8")
    out_pdf = build_dir / f"{tex_path.stem}.pdf"
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "log_text": log_text,
        "log_path": str(log_path),
        "pdf_path": str(out_pdf),
        "pdf_exists": bool(out_pdf.exists()),
    }


# Function: run up to max_passes to settle cross-references/citations.

def _compile_with_reruns(
    *,
    engine: str,
    tex_path: Path,
    build_dir: Path,
    logs_dir: Path,
    profile: str,
    round_idx: int,
    max_passes: int,
    include_overfull_warning: bool,
) -> Dict[str, Any]:
    last: Dict[str, Any] = {}
    passes: List[Dict[str, Any]] = []
    for pass_idx in range(1, max_passes + 1):
        one = _run_latex_once(
            engine=engine,
            tex_path=tex_path,
            build_dir=build_dir,
            logs_dir=logs_dir,
            profile=profile,
            round_idx=round_idx,
            pass_idx=pass_idx,
        )
        passes.append(one)
        last = one
        if int(one.get("returncode", 1)) != 0:
            break

        if _needs_rerun(str(one.get("log_text", ""))) and pass_idx < max_passes:
            continue

        break

    errors, warnings = _collect_issues(
        str(last.get("log_text", "")),
        int(last.get("returncode", 1)),
        include_overfull_warning=include_overfull_warning,
    )
    return {
        "passes": passes,
        "last": last,
        "errors": errors,
        "warnings": warnings,
    }


# Function: rename duplicate labels in TeX to remove multiply-defined warnings.

def _fix_duplicate_labels(tex_text: str) -> Tuple[str, List[Tuple[str, str]]]:
    seen: Dict[str, int] = {}
    replacements: List[Tuple[str, str]] = []

    # 関数: `_repl` の入出力契約と処理意図を定義する。
    def _repl(m: re.Match[str]) -> str:
        label = m.group(1)
        n = int(seen.get(label, 0)) + 1
        seen[label] = n
        if n == 1:
            return m.group(0)

        new_label = f"{label}-dup{n}"
        replacements.append((label, new_label))
        return rf"\label{{{new_label}}}"

    new_text = _LABEL_RE.sub(_repl, tex_text)
    return new_text, replacements


# Function: fix common escape artifacts that break TeX semantics.

def _fix_escape_artifacts(tex_text: str) -> Tuple[str, List[str]]:
    rules: List[Tuple[str, str]] = [
        (r"\textbackslash{}texttt\{", r"\texttt{"),
        (r"\textbackslash{}\%", r"\%"),
        (r"\textbackslash{}%", r"\%"),
        (r"\textbackslash{}\_", r"\_"),
        (r"\textbackslash{}_", r"\_"),
        (r"\textasciicircum{}", "^"),
    ]
    notes: List[str] = []
    out = tex_text
    for old, new in rules:
        if old in out:
            out = out.replace(old, new)
            notes.append(f"escape_fix:{old}->{new}")

    return out, notes


# Function: inject emergencystretch once to mitigate persistent overfull warnings.

def _inject_emergencystretch(tex_text: str) -> Tuple[str, bool]:
    if r"\emergencystretch=" in tex_text:
        return tex_text, False

    marker = r"\begin{document}"
    idx = tex_text.find(marker)
    if idx < 0:
        return tex_text, False

    inject = "\\setlength{\\emergencystretch}{2em}\n"
    return tex_text[:idx] + inject + tex_text[idx:], True


# Function: apply one auto-fix round and write updated TeX if changed.

def _apply_autofix_round(
    *,
    tex_path: Path,
    warnings: List[str],
    fail_on_overfull: bool,
) -> Tuple[bool, List[str]]:
    original = tex_path.read_text(encoding="utf-8", errors="replace")
    changed = False
    notes: List[str] = []
    updated = original

    if "warn:label_multiply_defined" in warnings:
        dedup_text, replacements = _fix_duplicate_labels(updated)
        if replacements:
            updated = dedup_text
            changed = True
            notes.append(f"label_dedup:{len(replacements)}")

    escape_fixed, escape_notes = _fix_escape_artifacts(updated)
    if escape_notes:
        updated = escape_fixed
        changed = True
        notes.extend(escape_notes)

    if fail_on_overfull and ("warn:overfull_hbox" in warnings):
        stretched, done = _inject_emergencystretch(updated)
        if done:
            updated = stretched
            changed = True
            notes.append("inject_emergencystretch")

    if changed and (updated != original):
        tex_path.write_text(updated, encoding="utf-8")
        return True, notes

    return False, notes


# Function: keep papers/ limited to canonical paper PDFs only.

def _prune_papers_dir(*, papers_dir: Path) -> List[str]:
    papers_dir.mkdir(parents=True, exist_ok=True)
    allowed_names = set(_PROFILE_TO_PDF.values())
    removed: List[str] = []
    for entry in list(papers_dir.iterdir()):
        if entry.is_file() and entry.name in allowed_names and entry.suffix.lower() == ".pdf":
            continue

        if entry.is_file():
            entry.unlink()
            removed.append(entry.name)
            continue

        if entry.is_dir():
            shutil.rmtree(entry)
            removed.append(f"{entry.name}/")

    return removed


# Function: copy generated PDF into papers/ with canonical file names.

def _sync_to_papers(*, pdf_src: Path, papers_dir: Path, profile: str) -> Path:
    papers_dir.mkdir(parents=True, exist_ok=True)
    out_name = _PROFILE_TO_PDF[profile]
    out_path = papers_dir / out_name
    shutil.copy2(str(pdf_src), str(out_path))
    return out_path


# Function: process one profile end-to-end (compile, autofix, optional sync).

def _build_profile_pdf(
    *,
    profile: str,
    outdir: Path,
    logs_dir: Path,
    engine_bin: str,
    max_passes: int,
    max_autofix_rounds: int,
    fail_on_overfull: bool,
    sync_papers: bool,
    papers_dir: Path,
) -> Dict[str, Any]:
    tex_name = _PROFILE_TO_TEX[profile]
    pdf_name = _PROFILE_TO_PDF[profile]
    tex_path = outdir / tex_name
    pdf_out = outdir / pdf_name
    build_dir = (outdir / "_tex_pdf_tmp" / profile).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {
        "profile": profile,
        "tex_path": str(tex_path),
        "pdf_path": str(pdf_out),
        "ok": False,
        "rounds": [],
        "errors": [],
        "warnings": [],
        "synced_papers_pdf": None,
    }

    if not tex_path.exists():
        result["errors"] = [f"tex_missing:{tex_path}"]
        return result

    for round_idx in range(0, max_autofix_rounds + 1):
        comp = _compile_with_reruns(
            engine=engine_bin,
            tex_path=tex_path,
            build_dir=build_dir,
            logs_dir=logs_dir,
            profile=profile,
            round_idx=round_idx,
            max_passes=max_passes,
            include_overfull_warning=bool(fail_on_overfull),
        )
        result["rounds"].append(comp)
        errors = list(comp.get("errors") or [])
        warnings = list(comp.get("warnings") or [])
        result["errors"] = errors
        result["warnings"] = warnings
        blocking_warning = ("warn:label_multiply_defined" in warnings) or (
            fail_on_overfull and ("warn:overfull_hbox" in warnings)
        )

        if (not errors) and (not blocking_warning):
            last_pdf = Path(str((comp.get("last") or {}).get("pdf_path") or ""))
            if not last_pdf.exists():
                result["errors"] = [f"pdf_missing:{last_pdf}"]
                break

            shutil.copy2(str(last_pdf), str(pdf_out))
            if sync_papers:
                synced = _sync_to_papers(pdf_src=pdf_out, papers_dir=papers_dir, profile=profile)
                result["synced_papers_pdf"] = str(synced)

            result["ok"] = True
            return result

        if round_idx >= max_autofix_rounds:
            break

        fixed, notes = _apply_autofix_round(
            tex_path=tex_path,
            warnings=warnings,
            fail_on_overfull=fail_on_overfull,
        )
        result["rounds"][-1]["autofix"] = {"applied": bool(fixed), "notes": notes}
        if not fixed:
            break

    return result


# Function: CLI entrypoint.

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build paper PDFs from TeX with safe auto-fixes.")
    ap.add_argument(
        "--profile",
        choices=list(_PROFILE_TO_TEX.keys()),
        action="append",
        help="target profile (repeatable). default: all profiles",
    )
    ap.add_argument("--outdir", default=str(_ROOT / "output" / "private" / "summary"))
    ap.add_argument("--logs-dir", default=str(_ROOT / "output" / "private" / "summary" / "logs"))
    ap.add_argument("--json-out", default=None, help="output json path (default: outdir/paper_pdf_build.json)")
    ap.add_argument("--engine", choices=["auto", "lualatex", "xelatex", "pdflatex"], default="auto")
    ap.add_argument("--require-engine", action="store_true", help="fail when no TeX engine is available")
    ap.add_argument("--max-passes", type=int, default=3, help="max compile passes per round (default: 3)")
    ap.add_argument(
        "--max-autofix-rounds",
        type=int,
        default=2,
        help="max auto-fix rounds after initial compile (default: 2)",
    )
    ap.add_argument("--fail-on-overfull", action="store_true", help="treat overfull hbox warnings as blocking")
    ap.add_argument("--sync-papers", action="store_true", help="(互換) papers同期を明示。現在は常時有効。")
    ap.add_argument("--papers-dir", default=str(_ROOT / "papers"), help="destination directory for --sync-papers")
    args = ap.parse_args(list(argv) if argv is not None else None)

    profiles = args.profile or list(_PROFILE_TO_TEX.keys())
    outdir = Path(str(args.outdir))
    logs_dir = Path(str(args.logs_dir))
    logs_dir.mkdir(parents=True, exist_ok=True)
    papers_dir = Path(str(args.papers_dir))
    # 運用固定: TeX更新時のPDFは必ず papers/ に同期する。
    sync_papers = True
    if not bool(args.sync_papers):
        print("[info] --sync-papers 未指定でも運用ルールにより papers 同期を強制します。")

    removed_papers_entries: List[str] = []
    if sync_papers:
        removed_papers_entries = _prune_papers_dir(papers_dir=papers_dir)
        for name in removed_papers_entries:
            print(f"[info] removed from papers/: {name}")

    max_passes = max(1, int(args.max_passes))
    max_autofix_rounds = max(0, int(args.max_autofix_rounds))

    engine_bin, engine_mode = _pick_engine(str(args.engine))
    if engine_bin is None:
        payload = {
            "generated_utc": _utc_now(),
            "ok": False,
            "reason": "latex_engine_unavailable",
            "engine_mode": engine_mode,
            "engine_request": str(args.engine),
            "profiles": profiles,
        }
        json_out = Path(str(args.json_out)) if args.json_out else (outdir / "paper_pdf_build.json")
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[ok] wrote: {json_out}")
        if args.require_engine:
            print("paper_pdf: ok=False (latex_engine_unavailable)")
            return 1

        print("paper_pdf: ok=True (engine unavailable; skipped)")
        return 0

    results: Dict[str, Any] = {}
    all_ok = True
    for profile in profiles:
        one = _build_profile_pdf(
            profile=profile,
            outdir=outdir,
            logs_dir=logs_dir,
            engine_bin=str(engine_bin),
            max_passes=max_passes,
            max_autofix_rounds=max_autofix_rounds,
            fail_on_overfull=bool(args.fail_on_overfull),
            sync_papers=sync_papers,
            papers_dir=papers_dir,
        )
        results[profile] = one
        all_ok = all_ok and bool(one.get("ok"))

    payload = {
        "generated_utc": _utc_now(),
        "ok": all_ok,
        "engine": str(engine_bin),
        "engine_mode": engine_mode,
        "profiles": profiles,
        "results": results,
        "params": {
            "max_passes": max_passes,
            "max_autofix_rounds": max_autofix_rounds,
            "fail_on_overfull": bool(args.fail_on_overfull),
            "sync_papers": sync_papers,
            "papers_dir": str(papers_dir),
            "removed_papers_entries": removed_papers_entries,
        },
    }
    json_out = Path(str(args.json_out)) if args.json_out else (outdir / "paper_pdf_build.json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {json_out}")
    print(f"paper_pdf: ok={all_ok}")
    for profile in profiles:
        rec = results.get(profile) or {}
        print(
            f"- {profile}: ok={bool(rec.get('ok'))} "
            f"errors={len(rec.get('errors') or [])} "
            f"warnings={len(rec.get('warnings') or [])}"
        )

    try:
        worklog.append_event(
            {
                "event_type": "paper_pdf",
                "profiles": profiles,
                "ok": all_ok,
                "json_out": json_out,
                "sync_papers": sync_papers,
                "papers_dir": papers_dir,
            }
        )
    except Exception:
        pass

    return 0 if all_ok else 1


# Condition: run CLI when executed as a script.

if __name__ == "__main__":
    raise SystemExit(main())
