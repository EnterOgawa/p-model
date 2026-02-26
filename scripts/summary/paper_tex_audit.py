#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_tex_audit.py

生成済みTeX（Part1-4）に対する厳格監査。

目的:
- LaTeXコンパイル由来の崩れ（Undefined control sequence / Missing $ / Double subscript等）を
  生成直後に検出し、配布前に失敗させる。
- コンパイラが無い環境でも、既知の高リスク表記を静的監査して早期検知する。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog

_PROFILE_TO_TEX = {
    "paper": "pmodel_paper.tex",
    "part2_astrophysics": "pmodel_paper_part2_astrophysics.tex",
    "part3_quantum": "pmodel_paper_part3_quantum.tex",
    "part4_verification": "pmodel_paper_part4_verification.tex",
}

_COMPILE_FATAL_PATTERNS = [
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

_COMPILE_WARNING_PATTERNS = [
    re.compile(r"Label `[^`]+` multiply defined"),
    re.compile(r"Overfull \\hbox"),
]

_SUSPICIOUS_LITERAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("glued_command_approxx", re.compile(r"\\approxx\b")),
    ("glued_command_proptocos", re.compile(r"\\proptocos\b")),
    ("glued_command_proptosin", re.compile(r"\\proptosin\b")),
    ("glued_command_toz", re.compile(r"\\toz\b")),
    ("glued_command_thetaP", re.compile(r"\\thetaP\b")),
    ("glued_command_Boxu", re.compile(r"\\Boxu\b")),
    ("glued_command_nabla", re.compile(r"\\nabla(?=[A-Za-z])")),
    ("glued_command_ln", re.compile(r"\\ln(?=[A-Za-z])")),
    ("glued_command_exp", re.compile(r"\\exp(?=[A-Za-z])")),
    ("glued_command_sqrt", re.compile(r"\\sqrt(?=[A-Za-z])")),
    ("glued_partial_subscript_p", re.compile(r"\\partial_(?:\\[A-Za-z]+|[A-Za-z])P\b")),
]

_MATH_SEGMENT_RE = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$", flags=re.DOTALL)
_SNAKE_IN_MATH_RE = re.compile(r"(?<!\\)\b[A-Za-z][A-Za-z0-9.]*_(?:[A-Za-z0-9.]+_)+[A-Za-z0-9.]+\b")
_DANGEROUS_TEXT_UNDERSCORE_RE = re.compile(r"(?<!\\)\b[A-Za-z]+_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b")
_LABEL_RE = re.compile(r"\\label\{([^{}]+)\}")
_TEXTTT_RE = re.compile(r"\\texttt\{([^{}]*)\}")
_INLINE_MATH_RE = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$")
_PLAIN_DV_VECTOR_RE = re.compile(r"\bdv\s*=\s*\[\s*ξ0\s*,\s*ξ2\s*\]\s*\+\s*cov\b")
_PLAIN_Z_SCORE_RANGE_RE = re.compile(
    r"\bz_score_combined\s*(?:≈|=)\s*[-+]?\d+(?:\.\d+)?\s*(?:\.{2,}|…)\s*[-+]?\d+(?:\.\d+)?"
)
_PLAIN_EPSILON_ASSIGN_RE = re.compile(r"[εϵ]\s*=\s*[+-]?\d+(?:\.\d+)?")
_PLAIN_AP_Z_TRIPLET_RE = re.compile(r"\bz\s*(?:≈|=)\s*\d+(?:\.\d+)?/\d+(?:\.\d+)?/\d+(?:\.\d+)?")
_DISTANCE_FORMULA_IN_TEXTTT_RE = re.compile(r"\\texttt\{[^{}]*(?:D\\_\{A\}|D\\_\{V\}|d\\_\{A\}|d\\_\{M\})[^{}]*\}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_engine_binary(engine: str) -> str | None:
    from_path = shutil.which(engine)
    # 条件分岐: `from_path` を満たす経路を評価する。
    if from_path:
        return from_path

    env_candidates: list[str] = []
    for key in (
        f"WAVEP_{engine.upper()}_PATH",
        f"TEX_{engine.upper()}_PATH",
        "WAVEP_TEX_BIN",
        "TEXLIVE_BIN",
        "MIKTEX_BIN",
    ):
        value = os.environ.get(key, "").strip()
        # 条件分岐: `not value` を満たす経路を評価する。
        if not value:
            continue

        p = Path(value)
        # 条件分岐: `p.suffix.lower() == ".exe"` を満たす経路を評価する。
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
    # 条件分岐: `local_appdata` を満たす経路を評価する。
    if local_appdata:
        common_candidates.append(Path(local_appdata) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64" / f"{engine}.exe")

    common_candidates.extend(sorted(Path(r"C:\texlive").glob(rf"*\bin\windows\{engine}.exe"), reverse=True))
    common_candidates.extend(sorted(Path(r"C:\Program Files\texlive").glob(rf"*\bin\windows\{engine}.exe"), reverse=True))

    for candidate in env_candidates + [str(p) for p in common_candidates]:
        try:
            cp = Path(candidate)
            # 条件分岐: `cp.exists()` を満たす経路を評価する。
            if cp.exists():
                return str(cp)
        except Exception:
            continue

    return None


def _pick_engine(choice: str) -> Tuple[str | None, str]:
    # 条件分岐: `choice == "none"` を満たす経路を評価する。
    if choice == "none":
        return None, "disabled"

    # 条件分岐: `choice != "auto"` を満たす経路を評価する。

    if choice != "auto":
        found = _find_engine_binary(choice)
        return (found, "requested" if found else "missing")

    for eng in ("lualatex", "xelatex", "pdflatex"):
        found = _find_engine_binary(eng)
        # 条件分岐: `found` を満たす経路を評価する。
        if found:
            return found, "auto"

    return None, "missing"


def _strip_comment(line: str) -> str:
    out: list[str] = []
    escaped = False
    for ch in line:
        # 条件分岐: `escaped` を満たす経路を評価する。
        if escaped:
            out.append(ch)
            escaped = False
            continue

        # 条件分岐: `ch == "\\"` を満たす経路を評価する。

        if ch == "\\":
            out.append(ch)
            escaped = True
            continue

        # 条件分岐: `ch == "%"` を満たす経路を評価する。

        if ch == "%":
            break

        out.append(ch)

    return "".join(out)


def _strip_inline_math(text: str) -> str:
    return _INLINE_MATH_RE.sub("", text)


def _texttt_unescape(payload: str) -> str:
    out = payload
    out = out.replace(r"\_", "_")
    out = out.replace(r"\textasciicircum{}", "^")
    out = out.replace(r"\{", "{").replace(r"\}", "}")
    out = out.replace(r"\%", "%").replace(r"\&", "&").replace(r"\$", "$")
    return out.strip()


def _is_pathlike_token(text: str) -> bool:
    low = text.lower()
    # 条件分岐: `"://" in low` を満たす経路を評価する。
    if "://" in low:
        return True

    # 条件分岐: `re.match(r"^[a-z]:[\\/]", text, flags=re.IGNORECASE)` を満たす経路を評価する。

    if re.match(r"^[a-z]:[\\/]", text, flags=re.IGNORECASE):
        return True

    # 条件分岐: `low.startswith(("output/", "scripts/", "data/", "doc/", "../", "./", "..\\",...` を満たす経路を評価する。

    if low.startswith(("output/", "scripts/", "data/", "doc/", "../", "./", "..\\", ".\\")):
        return True

    # 条件分岐: `re.search(r"\.(json|csv|png|jpg|jpeg|pdf|svg|bmp|webp|txt|md|tex|html|docx|py...` を満たす経路を評価する。

    if re.search(r"\.(json|csv|png|jpg|jpeg|pdf|svg|bmp|webp|txt|md|tex|html|docx|py|bat|log)\b", low):
        return True

    # 条件分岐: `"/" in text or "\\" in text` を満たす経路を評価する。

    if "/" in text or "\\" in text:
        return True

    return False


def _looks_like_pseudo_math_texttt(text: str) -> bool:
    # 条件分岐: `not text` を満たす経路を評価する。
    if not text:
        return False

    # 条件分岐: `_is_pathlike_token(text)` を満たす経路を評価する。

    if _is_pathlike_token(text):
        return False

    if re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_]*(?:\s*(?:=|<=|>=|<|>)\s*[A-Za-z0-9_.+-]+)+",
        text,
    ):
        return False

    has_var = bool(re.search(r"[A-Za-zα-ωΑ-Ω](?:_[A-Za-z0-9{}]+)?", text))
    has_cmp = bool(re.search(r"(<=|>=|<|>)", text))
    has_math_ops = bool(re.search(r"(\^|/|\(|\)|\+|\*|_[{(]|\{|\}|\\frac|\\sqrt)", text))
    has_equal = "=" in text
    # 条件分岐: `has_equal and (not has_cmp) and (not has_math_ops)` を満たす経路を評価する。
    if has_equal and (not has_cmp) and (not has_math_ops):
        return False

    return has_var and (has_cmp or has_math_ops or has_equal)


def _static_audit(tex_text: str) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    labels = _LABEL_RE.findall(tex_text)
    dup = {k: v for k, v in Counter(labels).items() if v > 1}
    # 条件分岐: `dup` を満たす経路を評価する。
    if dup:
        errors.append("duplicate_labels: " + ", ".join(f"{k}x{v}" for k, v in sorted(dup.items())))

    for key, pattern in _SUSPICIOUS_LITERAL_PATTERNS:
        # 条件分岐: `pattern.search(tex_text)` を満たす経路を評価する。
        if pattern.search(tex_text):
            errors.append(f"{key}_detected")

    for m in _MATH_SEGMENT_RE.finditer(tex_text):
        seg = m.group(1)
        bad = _SNAKE_IN_MATH_RE.search(seg)
        # 条件分岐: `bad` を満たす経路を評価する。
        if bad:
            errors.append(f"double_subscript_risk_in_math: {bad.group(0)}")
            break

    for m in _TEXTTT_RE.finditer(tex_text):
        raw_payload = m.group(1)
        payload = _texttt_unescape(raw_payload)
        # 条件分岐: `_looks_like_pseudo_math_texttt(payload)` を満たす経路を評価する。
        if _looks_like_pseudo_math_texttt(payload):
            errors.append(f"pseudo_math_in_texttt_detected: {payload[:80]}")
            break

    # 条件分岐: `_DISTANCE_FORMULA_IN_TEXTTT_RE.search(tex_text)` を満たす経路を評価する。

    if _DISTANCE_FORMULA_IN_TEXTTT_RE.search(tex_text):
        errors.append("distance_formula_in_texttt_detected")

    # text領域での snake_case 多重連結（運用ログ混入等）を検出

    in_display_math = False
    for lineno, raw in enumerate(tex_text.splitlines(), start=1):
        line = _strip_comment(raw)
        # 条件分岐: `not line.strip()` を満たす経路を評価する。
        if not line.strip():
            continue

        # 条件分岐: `"\\[" in line` を満たす経路を評価する。

        if "\\[" in line:
            in_display_math = True

        # 条件分岐: `in_display_math` を満たす経路を評価する。

        if in_display_math:
            # 条件分岐: `"\\]" in line` を満たす経路を評価する。
            if "\\]" in line:
                in_display_math = False

            continue

        line_nomath = _strip_inline_math(line)
        # 条件分岐: `re.search(r"\|z\|\s*(?:>=|<=|>|<)\s*\d", line_nomath)` を満たす経路を評価する。
        if re.search(r"\|z\|\s*(?:>=|<=|>|<)\s*\d", line_nomath):
            errors.append(f"plain_comparator_outside_math: line {lineno} |z| comparator")
            break

        # 条件分岐: `_PLAIN_DV_VECTOR_RE.search(line_nomath)` を満たす経路を評価する。

        if _PLAIN_DV_VECTOR_RE.search(line_nomath):
            errors.append(f"plain_dv_vector_outside_math: line {lineno}")
            break

        # 条件分岐: `_PLAIN_Z_SCORE_RANGE_RE.search(line_nomath)` を満たす経路を評価する。

        if _PLAIN_Z_SCORE_RANGE_RE.search(line_nomath):
            errors.append(f"plain_z_score_range_outside_math: line {lineno}")
            break

        # 条件分岐: `_PLAIN_EPSILON_ASSIGN_RE.search(line_nomath)` を満たす経路を評価する。

        if _PLAIN_EPSILON_ASSIGN_RE.search(line_nomath):
            errors.append(f"plain_epsilon_assignment_outside_math: line {lineno}")
            break

        # 条件分岐: `"AP" in line_nomath and _PLAIN_AP_Z_TRIPLET_RE.search(line_nomath)` を満たす経路を評価する。

        if "AP" in line_nomath and _PLAIN_AP_Z_TRIPLET_RE.search(line_nomath):
            errors.append(f"plain_ap_z_triplet_outside_math: line {lineno}")
            break

        # 条件分岐: `"\\texttt{" in line or "\\url{" in line or "\\href{" in line` を満たす経路を評価する。

        if "\\texttt{" in line or "\\url{" in line or "\\href{" in line:
            continue

        # 条件分岐: `_DANGEROUS_TEXT_UNDERSCORE_RE.search(line)` を満たす経路を評価する。

        if _DANGEROUS_TEXT_UNDERSCORE_RE.search(line):
            warnings.append(f"line {lineno}: snake_case token outside protected context")
            # 条件分岐: `len(warnings) >= 20` を満たす経路を評価する。
            if len(warnings) >= 20:
                break

    # 生の .html ローカルリンク混入

    if re.search(r"\\href\{[^{}]+\.html?(?:[#?][^{}]*)?\}", tex_text, flags=re.IGNORECASE):
        errors.append("local_html_href_detected")

    return {"errors": errors, "warnings": warnings, "duplicate_labels": dup}


def _run_compile(
    *,
    engine: str,
    tex_path: Path,
    logs_dir: Path,
    fail_on_overfull: bool,
) -> Dict[str, Any]:
    build_dir = (tex_path.parent / "_tex_audit_tmp" / tex_path.stem).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        engine,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
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
    log_file = logs_dir / f"paper_tex_audit_{tex_path.stem}.log"
    log_file.write_text(log_text, encoding="utf-8")

    errors: List[str] = []
    warnings: List[str] = []
    for pat in _COMPILE_FATAL_PATTERNS:
        # 条件分岐: `pat.search(log_text)` を満たす経路を評価する。
        if pat.search(log_text):
            errors.append(f"compile:{pat.pattern}")

    for pat in _COMPILE_WARNING_PATTERNS:
        # 条件分岐: `pat.search(log_text)` を満たす経路を評価する。
        if pat.search(log_text):
            msg = f"compile_warn:{pat.pattern}"
            # 条件分岐: `fail_on_overfull and "Overfull" in pat.pattern` を満たす経路を評価する。
            if fail_on_overfull and "Overfull" in pat.pattern:
                errors.append(msg)
            else:
                warnings.append(msg)

    # 条件分岐: `proc.returncode != 0` を満たす経路を評価する。

    if proc.returncode != 0:
        errors.append(f"compile_rc={proc.returncode}")

    return {
        "engine": engine,
        "cmd": cmd,
        "returncode": proc.returncode,
        "log_file": str(log_file),
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }


def _audit_profile(
    *,
    profile: str,
    outdir: Path,
    logs_dir: Path,
    engine_choice: str,
    require_engine: bool,
    fail_on_overfull: bool,
) -> Dict[str, Any]:
    tex_name = _PROFILE_TO_TEX[profile]
    tex_path = outdir / tex_name
    result: Dict[str, Any] = {
        "profile": profile,
        "tex_path": str(tex_path),
        "exists": tex_path.exists(),
        "static": {"errors": [], "warnings": [], "duplicate_labels": {}},
        "compile": None,
        "ok": False,
    }

    # 条件分岐: `not tex_path.exists()` を満たす経路を評価する。
    if not tex_path.exists():
        result["static"]["errors"] = [f"tex_missing:{tex_path}"]
        return result

    tex_text = tex_path.read_text(encoding="utf-8", errors="replace")
    static = _static_audit(tex_text)
    result["static"] = static

    engine, engine_mode = _pick_engine(engine_choice)
    result["engine_mode"] = engine_mode
    # 条件分岐: `engine is None` を満たす経路を評価する。
    if engine is None:
        msg = "latex_engine_unavailable"
        # 条件分岐: `require_engine` を満たす経路を評価する。
        if require_engine:
            static["errors"].append(msg)
        else:
            static["warnings"].append(msg)
    else:
        result["compile"] = _run_compile(
            engine=engine,
            tex_path=tex_path,
            logs_dir=logs_dir,
            fail_on_overfull=fail_on_overfull,
        )

    compile_errors = result["compile"]["errors"] if result.get("compile") else []
    static_errors = static.get("errors", [])
    result["ok"] = (len(static_errors) == 0) and (len(compile_errors) == 0)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Strict post-build TeX audit for paper profiles.")
    ap.add_argument(
        "--profile",
        choices=list(_PROFILE_TO_TEX.keys()),
        action="append",
        help="target profile (repeatable). default: all profiles",
    )
    ap.add_argument("--outdir", default=str(_ROOT / "output" / "private" / "summary"))
    ap.add_argument("--logs-dir", default=str(_ROOT / "output" / "private" / "summary" / "logs"))
    ap.add_argument("--engine", choices=["auto", "lualatex", "xelatex", "pdflatex", "none"], default="auto")
    ap.add_argument("--require-engine", action="store_true", help="fail when no TeX compiler is available")
    ap.add_argument("--fail-on-overfull", action="store_true", help="treat Overfull \\hbox as error")
    ap.add_argument("--json-out", default=None, help="output json path (default: outdir/paper_tex_audit.json)")
    args = ap.parse_args(argv)

    profiles = args.profile or list(_PROFILE_TO_TEX.keys())
    outdir = Path(args.outdir)
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    json_out = Path(args.json_out) if args.json_out else (outdir / "paper_tex_audit.json")

    per_profile: Dict[str, Any] = {}
    all_ok = True
    for profile in profiles:
        res = _audit_profile(
            profile=profile,
            outdir=outdir,
            logs_dir=logs_dir,
            engine_choice=str(args.engine),
            require_engine=bool(args.require_engine),
            fail_on_overfull=bool(args.fail_on_overfull),
        )
        per_profile[profile] = res
        all_ok = all_ok and bool(res.get("ok"))

    payload = {
        "generated_utc": _utc_now(),
        "outdir": str(outdir),
        "profiles": profiles,
        "ok": all_ok,
        "results": per_profile,
    }
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote: {json_out}")
    print(f"paper_tex_audit: ok={all_ok}")
    for p in profiles:
        r = per_profile[p]
        s_err = len(r["static"]["errors"])
        c_err = len((r.get("compile") or {}).get("errors", []))
        print(f"- {p}: ok={r['ok']} static_errors={s_err} compile_errors={c_err}")

    try:
        worklog.append_event(
            {
                "event_type": "paper_tex_audit",
                "profiles": profiles,
                "ok": all_ok,
                "json_out": json_out,
            }
        )
    except Exception:
        pass

    return 0 if all_ok else 1


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
