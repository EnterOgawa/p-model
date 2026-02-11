#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_qc.py

Phase 8 / Step 8.2（本文整合：最終確認）向けの軽量QC。

目的：
- `output/private/summary/pmodel_paper.html`（publish）を対象に、機械的に検出できる崩れ/混入を潰す。
- 8.4（図番号昇順・可読性）と 5.5（LaTeX表記混入防止）の運用チェックを一つにまとめる。
 - Part IV（検証資料）が存在する場合は、Part IV HTMLも同様に崩れ/混入を検出する。
 - （Wordが使える環境では）`output/private/summary/pmodel_paper.docx` の体裁崩れ（要請ブロックの記号/コロン、A.0表の枠線有無）も機械的に検出する。

チェック項目：
- paper_lint strict（引用キー/図表インデックス/参照PNG）
- pmodel_paper.html に `\\`（2連バックスラッシュ）が残っていない（LaTeX文字列露出の検出）
- 数式画像（equation-img）の alt が `数式` になっている
- 図番号（fig-001..）と caption（図1..）が連番・単調増加
 - paper DOCX（存在する場合）：
   - 2.1 要請ブロック：`要請（P内部）→帰結` が存在し、直後のコロンが fullwidth `：` になっている
   - A.0 自由度台帳：最初の表（自由度台帳直下）が「枠線なし」になっている（tblBorders があっても val=none/nil）

出力：
- `output/private/summary/paper_qc.json`
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import paper_lint as _paper_lint  # noqa: E402
from scripts.summary import worklog  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class _Check:
    ok: bool
    details: Dict[str, Any]


def _check_paper_lint_strict() -> _Check:
    root = _ROOT
    result = _paper_lint._lint(  # noqa: SLF001
        root=root,
        manuscript_paths=[root / "doc" / "paper" / "10_part1_core_theory.md"],
        references_path=root / "doc" / "paper" / "30_references.md",
        figures_index_path=root / "doc" / "paper" / "01_figures_index.md",
    )
    ok = (len(result.errors) == 0) and (len(result.warnings) == 0)
    return _Check(ok=ok, details={"errors": result.errors, "warnings": result.warnings})


def _check_part4_lint_strict() -> _Check:
    root = _ROOT
    result = _paper_lint._lint(  # noqa: SLF001
        root=root,
        manuscript_paths=[root / "doc" / "paper" / "13_part4_verification.md"],
        references_path=root / "doc" / "paper" / "30_references.md",
        figures_index_path=root / "doc" / "paper" / "01_figures_index.md",
    )
    ok = (len(result.errors) == 0) and (len(result.warnings) == 0)
    return _Check(ok=ok, details={"errors": result.errors, "warnings": result.warnings})


def _check_part2_lint_strict() -> _Check:
    root = _ROOT
    result = _paper_lint._lint(  # noqa: SLF001
        root=root,
        manuscript_paths=[root / "doc" / "paper" / "11_part2_astrophysics.md"],
        references_path=root / "doc" / "paper" / "30_references.md",
        figures_index_path=root / "doc" / "paper" / "01_figures_index.md",
    )
    ok = (len(result.errors) == 0) and (len(result.warnings) == 0)
    return _Check(ok=ok, details={"errors": result.errors, "warnings": result.warnings})


def _check_part3_lint_strict() -> _Check:
    root = _ROOT
    result = _paper_lint._lint(  # noqa: SLF001
        root=root,
        manuscript_paths=[
            root / "doc" / "paper" / "12_part3_quantum.md",
            root / "doc" / "paper" / "12_part3_quantum_appendix_a.md",
        ],
        references_path=root / "doc" / "paper" / "30_references.md",
        figures_index_path=root / "doc" / "paper" / "01_figures_index.md",
    )
    ok = (len(result.errors) == 0) and (len(result.warnings) == 0)
    return _Check(ok=ok, details={"errors": result.errors, "warnings": result.warnings})


_MD_PROSE_LATEX_ESCAPE_RE = re.compile(r"\\\\|\\[A-Za-z]+")


def _check_markdown_prose_no_latex_escapes(*, paper_dir: Path) -> _Check:
    """
    Enforce repo rule (5.5): prose must not expose LaTeX escapes like '\\gamma' or '\\\\'.

    We intentionally ignore:
    - math blocks delimited by $$ ... $$ (including multi-line blocks),
    - fenced code blocks ``` ... ```,
    - inline code spans `...` (Windows paths etc.).
    """

    hits: List[Dict[str, Any]] = []
    md_paths = sorted(paper_dir.glob("*.md"))
    for path in md_paths:
        lines = _read_text(path).splitlines()
        in_code_fence = False
        in_math = False
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()

            if stripped.startswith("```"):
                in_code_fence = not in_code_fence
                continue
            if in_code_fence:
                continue

            # Split by $$ to isolate math/prose segments and toggle state across lines.
            parts = line.split("$$")
            cur_in_math = in_math
            for idx, seg in enumerate(parts):
                seg_in_math = cur_in_math
                if not seg_in_math:
                    seg_no_code = re.sub(r"`[^`]*`", "", seg)
                    m = _MD_PROSE_LATEX_ESCAPE_RE.search(seg_no_code)
                    if m:
                        hits.append(
                            {
                                "file": str(path.relative_to(_ROOT)).replace("\\", "/"),
                                "line": int(lineno),
                                "token": m.group(0),
                                "text": stripped,
                            }
                        )
                        break
                if idx != len(parts) - 1:
                    cur_in_math = not cur_in_math
            in_math = cur_in_math

    return _Check(ok=(len(hits) == 0), details={"files": int(len(md_paths)), "hits": hits[:50], "hits_total": int(len(hits))})


_MD_HEADING_NUM_RE = re.compile(r"^#{1,6}\s+([A-Z]\.\d+(?:\.\d+)*|\d+(?:\.\d+)*)(?:\.)?\b")
_MD_SECTION_REF_SINGLE_RE = re.compile(r"([A-Z]\.\d+(?:\.\d+)*|\d+(?:\.\d+)*)節")
_MD_SECTION_REF_RANGE_RE = re.compile(r"(\d+(?:\.\d+)*)\s*[〜~\-]\s*(\d+(?:\.\d+)*)節")
_MD_SECTION_REF_SLASH_RE = re.compile(r"(\d+(?:\.\d+)*)\s*/\s*(\d+(?:\.\d+)*)節")
_MD_FIG_REF_RE = re.compile(r"図\s*(\d{1,4})")


_HTML_CODE_RE = re.compile(r"<code>([^<]{1,1000})</code>")


def _check_publish_html_no_repo_paths(*, html_text: str) -> _Check:
    """
    Enforce paper design policy:
    publish HTML must not expose repo-internal file paths or concrete run commands.

    (Those details are moved to Verification Materials.)
    """
    banned_prefixes = ("output/", "scripts/", "data/", "doc/")
    banned_command_prefixes = ("python -B ", "cmd /c ", "wsl ", "bash -lc ")

    hits: List[Dict[str, Any]] = []
    for m in _HTML_CODE_RE.finditer(html_text):
        code = html.unescape(m.group(1)).strip().replace("\\", "/")
        if not code:
            continue
        reason = None
        if any(code.startswith(p) for p in banned_prefixes):
            reason = "repo_path"
        elif any(code.startswith(p) for p in banned_command_prefixes):
            reason = "run_command"
        elif code.startswith("python ") and ("scripts/" in code):
            reason = "run_command"
        if reason:
            hits.append({"reason": reason, "code": code[:300]})

    return _Check(ok=(len(hits) == 0), details={"hits_total": int(len(hits)), "hits": hits[:50]})


def _iter_markdown_prose_segments(md_text: str):
    """
    Iterate prose segments of Markdown while ignoring:
    - fenced code blocks ``` ... ```
    - math blocks $$ ... $$
    - inline code spans `...`
    """
    lines = md_text.splitlines()
    in_code_fence = False
    in_math = False
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            continue

        parts = line.split("$$")
        cur_in_math = in_math
        for idx, seg in enumerate(parts):
            if not cur_in_math:
                seg_no_code = re.sub(r"`[^`]*`", "", seg)
                yield int(lineno), seg_no_code, stripped
            if idx != len(parts) - 1:
                cur_in_math = not cur_in_math
        in_math = cur_in_math


def _check_markdown_section_references(*, paper_dir: Path) -> _Check:
    """
    Check that references like '2.6節' point to an existing numbered heading
    somewhere under doc/paper/*.md (cross-document references are allowed).
    """
    md_paths = sorted(paper_dir.glob("*.md"))
    texts: Dict[Path, str] = {}
    headings: set[str] = set()

    for path in md_paths:
        txt = _read_text(path)
        texts[path] = txt
        in_code_fence = False
        for line in txt.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_fence = not in_code_fence
                continue
            if in_code_fence:
                continue
            m = _MD_HEADING_NUM_RE.match(line)
            if m:
                headings.add(m.group(1))

    missing: List[Dict[str, Any]] = []
    for path, txt in texts.items():
        for lineno, seg, stripped in _iter_markdown_prose_segments(txt):
            refs: set[str] = set()
            for a, b in _MD_SECTION_REF_SLASH_RE.findall(seg):
                refs.add(a)
                refs.add(b)
            for a, b in _MD_SECTION_REF_RANGE_RE.findall(seg):
                refs.add(a)
                refs.add(b)
            for n in _MD_SECTION_REF_SINGLE_RE.findall(seg):
                refs.add(n)
            for n in sorted(refs):
                if n in headings:
                    continue
                missing.append(
                    {
                        "file": str(path.relative_to(_ROOT)).replace("\\", "/"),
                        "line": int(lineno),
                        "ref": n,
                        "text": stripped,
                    }
                )

    return _Check(
        ok=(len(missing) == 0),
        details={
            "files": int(len(md_paths)),
            "headings_total": int(len(headings)),
            "missing": missing[:50],
            "missing_total": int(len(missing)),
        },
    )


def _extract_figure_numbers_from_html(text: str) -> set[int]:
    caps = re.findall(r"<figcaption><strong>図(\d+):", text)
    out: set[int] = set()
    for x in caps:
        try:
            out.add(int(x))
        except Exception:
            continue
    return out


def _check_markdown_figure_number_references(*, manuscript_md: Path, html_text: str) -> _Check:
    """
    Check that references like '図16' in a manuscript do not exceed / fall outside
    the actual figure numbers present in the corresponding publish HTML.
    """
    if not manuscript_md.exists():
        return _Check(ok=False, details={"missing": True, "manuscript": str(manuscript_md)})
    md_text = _read_text(manuscript_md)
    fig_nums = _extract_figure_numbers_from_html(html_text)

    bad: List[Dict[str, Any]] = []
    for lineno, seg, stripped in _iter_markdown_prose_segments(md_text):
        for m in _MD_FIG_REF_RE.finditer(seg):
            try:
                n = int(m.group(1))
            except Exception:
                continue
            if (not fig_nums) or (n not in fig_nums):
                bad.append(
                    {
                        "file": str(manuscript_md.relative_to(_ROOT)).replace("\\", "/"),
                        "line": int(lineno),
                        "ref": int(n),
                        "text": stripped,
                    }
                )

    return _Check(
        ok=(len(bad) == 0),
        details={
            "manuscript": str(manuscript_md.relative_to(_ROOT)).replace("\\", "/"),
            "figures_in_html": int(len(fig_nums)),
            "bad_refs": bad[:50],
            "bad_refs_total": int(len(bad)),
        },
    )


def _check_no_double_backslash(text: str) -> _Check:
    # Ignore occurrences inside code/pre blocks so Windows paths like
    # `output\\private\\summary\\...` do not trigger false positives.
    scrubbed = re.sub(r"<pre[^>]*>.*?</pre>", "", text, flags=re.DOTALL)
    scrubbed = re.sub(r"<code[^>]*>.*?</code>", "", scrubbed, flags=re.DOTALL)

    idx = scrubbed.find("\\\\")
    ok = idx < 0
    ctx: Optional[str] = None
    if not ok:
        start = max(0, idx - 60)
        end = min(len(scrubbed), idx + 140)
        ctx = scrubbed[start:end]
    return _Check(ok=ok, details={"found_index": (None if ok else int(idx)), "context": ctx})


def _check_no_substrings(text: str, substrings: Sequence[str]) -> _Check:
    """
    Prevent accidental "draft" labels leaking into publish artifacts.

    This is intentionally a blunt check: we only use it for strings that should
    not appear anywhere in the publish HTML (e.g., "草稿").
    """

    found: List[Dict[str, Any]] = []
    for s in substrings:
        idx = text.find(s)
        if idx < 0:
            continue
        start = max(0, idx - 80)
        end = min(len(text), idx + 240)
        found.append({"substring": s, "found_index": int(idx), "context": text[start:end]})
    return _Check(ok=(len(found) == 0), details={"found": found[:10], "found_total": int(len(found))})


_HTML_ID_RE = re.compile(r"\bid=['\"]([^'\"]+)['\"]", re.IGNORECASE)
_HTML_HREF_ANCHOR_RE = re.compile(r"\bhref=['\"]#([^'\"]+)['\"]", re.IGNORECASE)


def _check_internal_anchor_links(text: str) -> _Check:
    """
    Check that in-page links (href="#...") resolve to an existing id="...".
    """
    ids = set(_HTML_ID_RE.findall(text))
    hrefs = _HTML_HREF_ANCHOR_RE.findall(text)
    missing = sorted({h for h in hrefs if h not in ids})
    return _Check(
        ok=(len(missing) == 0),
        details={
            "ids_count": int(len(ids)),
            "href_anchor_count": int(len(hrefs)),
            "missing": missing[:50],
            "missing_total": int(len(missing)),
        },
    )


_EQ_IMG_ALT_RE = re.compile(
    r"<img[^>]*\bclass=['\"]equation-img['\"][^>]*\balt=['\"]([^'\"]*)['\"][^>]*>",
    re.IGNORECASE,
)


def _check_equation_alt(text: str) -> _Check:
    alts = _EQ_IMG_ALT_RE.findall(text)
    bad = [a for a in alts if a != "数式"]
    return _Check(ok=(len(bad) == 0), details={"count": int(len(alts)), "bad_examples": bad[:10]})


def _check_figure_numbering(text: str) -> _Check:
    # id='fig-001' numbering
    ids = re.findall(r"id=['\"]fig-([0-9]{3})['\"]", text)
    id_nums = [int(x) for x in ids]

    # caption: <strong>図1:</strong>
    caps = re.findall(r"<figcaption><strong>図(\d+):", text)
    cap_nums = [int(x) for x in caps]

    details: Dict[str, Any] = {
        "fig_id_count": int(len(id_nums)),
        "caption_count": int(len(cap_nums)),
        "fig_id_dups": sorted([k for k, v in Counter(id_nums).items() if v > 1])[:20],
        "caption_dups": sorted([k for k, v in Counter(cap_nums).items() if v > 1])[:20],
        "fig_id_range": None,
        "caption_range": None,
        "fig_id_missing_in_range": [],
        "caption_missing_in_range": [],
        "fig_id_non_monotone_pairs": 0,
        "caption_non_monotone_pairs": 0,
    }

    def _range_and_missing(nums: List[int]) -> tuple[Optional[list[int]], List[int]]:
        if not nums:
            return None, []
        lo, hi = min(nums), max(nums)
        missing = [n for n in range(lo, hi + 1) if n not in set(nums)]
        return [lo, hi], missing

    fig_range, fig_missing = _range_and_missing(id_nums)
    cap_range, cap_missing = _range_and_missing(cap_nums)
    details["fig_id_range"] = fig_range
    details["caption_range"] = cap_range
    details["fig_id_missing_in_range"] = fig_missing[:50]
    details["caption_missing_in_range"] = cap_missing[:50]

    details["fig_id_non_monotone_pairs"] = int(sum(1 for a, b in zip(id_nums, id_nums[1:]) if b <= a))
    details["caption_non_monotone_pairs"] = int(sum(1 for a, b in zip(cap_nums, cap_nums[1:]) if b <= a))

    # If the document has no figures at all, treat numbering as trivially OK.
    if (details["fig_id_count"] == 0) and (details["caption_count"] == 0):
        ok = True
    else:
        ok = (
            (len(details["fig_id_dups"]) == 0)
            and (len(details["caption_dups"]) == 0)
            and (details["fig_id_missing_in_range"] == [])
            and (details["caption_missing_in_range"] == [])
            and (details["fig_id_non_monotone_pairs"] == 0)
            and (details["caption_non_monotone_pairs"] == 0)
            and (details["fig_id_count"] > 0)
            and (details["caption_count"] > 0)
        )
    return _Check(ok=ok, details=details)


def _read_docx_document_xml(docx_path: Path) -> Optional[str]:
    if not docx_path.exists():
        return None
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml = z.read("word/document.xml")
        return xml.decode("utf-8", errors="ignore")
    except Exception:
        return None


def _check_docx_callout_punctuation(*, docx_xml: Optional[str]) -> _Check:
    if not docx_xml:
        return _Check(ok=True, details={"skipped": True, "reason": "missing docx or unreadable word/document.xml"})

    marker = "要請（P内部）→帰結"
    idx = docx_xml.find(marker)
    if idx < 0:
        return _Check(ok=False, details={"marker": marker, "found": False})

    window = docx_xml[idx : min(len(docx_xml), idx + 1500)]
    has_fullwidth_colon = ("：" in window) or ("\uff1a" in window)
    has_ascii_colon = (":</w:t>" in window) or (">:</w:t>" in window)
    ok = bool(has_fullwidth_colon) and (not has_ascii_colon)

    ctx = window[:300]
    return _Check(
        ok=ok,
        details={
            "marker": marker,
            "found": True,
            "has_fullwidth_colon_in_window": bool(has_fullwidth_colon),
            "has_ascii_colon_in_window": bool(has_ascii_colon),
            "context_ascii": ctx.encode("ascii", "backslashreplace").decode("ascii"),
        },
    )


_BORDER_VAL_RE = re.compile(
    r"<w:(?:top|bottom|left|right|insideH|insideV)\\b[^>]*\\bw:val=\"([^\"]+)\"",
    flags=re.IGNORECASE,
)


def _check_docx_a0_table_no_borders(*, docx_xml: Optional[str]) -> _Check:
    if not docx_xml:
        return _Check(ok=True, details={"skipped": True, "reason": "missing docx or unreadable word/document.xml"})

    marker = "自由度台帳"
    idx = docx_xml.find(marker)
    if idx < 0:
        return _Check(ok=False, details={"marker": marker, "found": False})

    tbl_idx = docx_xml.find("<w:tbl", idx)
    if tbl_idx < 0:
        return _Check(ok=False, details={"marker": marker, "found": True, "table_found": False})

    window = docx_xml[tbl_idx : min(len(docx_xml), tbl_idx + 6000)]
    vals = [v.lower() for v in _BORDER_VAL_RE.findall(window)]
    bad_vals = [v for v in vals if v not in ("none", "nil")]
    ok = len(bad_vals) == 0
    return _Check(
        ok=bool(ok),
        details={
            "marker": marker,
            "found": True,
            "table_found": True,
            "border_vals_found": vals,
            "bad_border_vals": bad_vals,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Paper publish QC (Phase 8 / Step 8.2).")
    ap.add_argument(
        "--paper-html",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "pmodel_paper.html"),
        help="publish paper html path (default: output/private/summary/pmodel_paper.html)",
    )
    ap.add_argument(
        "--paper-docx",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "pmodel_paper.docx"),
        help="publish paper docx path (default: output/private/summary/pmodel_paper.docx)",
    )
    ap.add_argument(
        "--part2-html",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "pmodel_paper_part2_astrophysics.html"),
        help="publish Part II html path (optional; default: output/private/summary/pmodel_paper_part2_astrophysics.html)",
    )
    ap.add_argument(
        "--part3-html",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "pmodel_paper_part3_quantum.html"),
        help="publish Part III html path (optional; default: output/private/summary/pmodel_paper_part3_quantum.html)",
    )
    ap.add_argument(
        "--part4-html",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "pmodel_paper_part4_verification.html"),
        help="publish Part IV html path (default: output/private/summary/pmodel_paper_part4_verification.html)",
    )
    ap.add_argument(
        "--public-html",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "pmodel_public_report.html"),
        help="public report html path (default: output/private/summary/pmodel_public_report.html)",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "paper_qc.json"),
        help="output json path (default: output/private/summary/paper_qc.json)",
    )
    ap.add_argument("--no-log", action="store_true", help="do not append worklog event")
    args = ap.parse_args(list(argv) if argv is not None else None)

    paper_html = Path(str(args.paper_html))
    if not paper_html.is_absolute():
        paper_html = (_ROOT / paper_html).resolve()
    paper_docx = Path(str(args.paper_docx))
    if not paper_docx.is_absolute():
        paper_docx = (_ROOT / paper_docx).resolve()
    part2_html = Path(str(args.part2_html))
    if not part2_html.is_absolute():
        part2_html = (_ROOT / part2_html).resolve()
    part3_html = Path(str(args.part3_html))
    if not part3_html.is_absolute():
        part3_html = (_ROOT / part3_html).resolve()
    part4_html = Path(str(args.part4_html))
    if not part4_html.is_absolute():
        part4_html = (_ROOT / part4_html).resolve()
    public_html = Path(str(args.public_html))
    if not public_html.is_absolute():
        public_html = (_ROOT / public_html).resolve()
    out_json = Path(str(args.out_json))
    if not out_json.is_absolute():
        out_json = (_ROOT / out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if not paper_html.exists():
        print(f"[err] missing: {paper_html}")
        return 2
    if not part4_html.exists():
        print(f"[err] missing: {part4_html}")
        return 2

    paper_text = _read_text(paper_html)
    part2_text = _read_text(part2_html) if part2_html.exists() else None
    part3_text = _read_text(part3_html) if part3_html.exists() else None
    part4_text = _read_text(part4_html)
    public_text = _read_text(public_html) if public_html.exists() else None
    docx_xml = _read_docx_document_xml(paper_docx)

    checks: Dict[str, _Check] = {
        "paper_lint_strict": _check_paper_lint_strict(),
        "part2_lint_strict": _check_part2_lint_strict(),
        "part3_lint_strict": _check_part3_lint_strict(),
        "part4_lint_strict": _check_part4_lint_strict(),
        "paper_md_section_references_resolve": _check_markdown_section_references(paper_dir=_ROOT / "doc" / "paper"),
        "paper_md_prose_no_latex_escapes": _check_markdown_prose_no_latex_escapes(paper_dir=_ROOT / "doc" / "paper"),
        "paper_html_no_double_backslash": _check_no_double_backslash(paper_text),
        "paper_html_no_draft_labels": _check_no_substrings(paper_text, ["草稿", "ドラフト"]),
        "part4_html_no_double_backslash": _check_no_double_backslash(part4_text),
        "part4_html_no_draft_labels": _check_no_substrings(part4_text, ["草稿", "ドラフト"]),
        "paper_html_no_repo_paths": _check_publish_html_no_repo_paths(html_text=paper_text),
        "paper_html_internal_anchor_links": _check_internal_anchor_links(paper_text),
        "part4_html_internal_anchor_links": _check_internal_anchor_links(part4_text),
        "paper_equation_alt": _check_equation_alt(paper_text),
        "part4_equation_alt": _check_equation_alt(part4_text),
        "paper_figure_numbering": _check_figure_numbering(paper_text),
        "part4_figure_numbering": _check_figure_numbering(part4_text),
        "paper_md_figure_number_references": _check_markdown_figure_number_references(
            manuscript_md=_ROOT / "doc" / "paper" / "10_part1_core_theory.md",
            html_text=paper_text,
        ),
        "part4_md_figure_number_references": _check_markdown_figure_number_references(
            manuscript_md=_ROOT / "doc" / "paper" / "13_part4_verification.md",
            html_text=part4_text,
        ),
        "paper_docx_callout_punctuation": _check_docx_callout_punctuation(docx_xml=docx_xml),
        "paper_docx_a0_table_no_borders": _check_docx_a0_table_no_borders(docx_xml=docx_xml),
    }

    if isinstance(part2_text, str):
        checks.update(
            {
                "part2_html_no_double_backslash": _check_no_double_backslash(part2_text),
                "part2_html_no_draft_labels": _check_no_substrings(part2_text, ["草稿", "ドラフト"]),
                "part2_html_no_repo_paths": _check_publish_html_no_repo_paths(html_text=part2_text),
                "part2_html_internal_anchor_links": _check_internal_anchor_links(part2_text),
                "part2_equation_alt": _check_equation_alt(part2_text),
                "part2_figure_numbering": _check_figure_numbering(part2_text),
                "part2_md_figure_number_references": _check_markdown_figure_number_references(
                    manuscript_md=_ROOT / "doc" / "paper" / "11_part2_astrophysics.md",
                    html_text=part2_text,
                ),
            }
        )

    if isinstance(part3_text, str):
        checks.update(
            {
                "part3_html_no_double_backslash": _check_no_double_backslash(part3_text),
                "part3_html_no_draft_labels": _check_no_substrings(part3_text, ["草稿", "ドラフト"]),
                "part3_html_no_repo_paths": _check_publish_html_no_repo_paths(html_text=part3_text),
                "part3_html_internal_anchor_links": _check_internal_anchor_links(part3_text),
                "part3_equation_alt": _check_equation_alt(part3_text),
                "part3_figure_numbering": _check_figure_numbering(part3_text),
                "part3_md_figure_number_references": _check_markdown_figure_number_references(
                    manuscript_md=_ROOT / "doc" / "paper" / "12_part3_quantum.md",
                    html_text=part3_text,
                ),
            }
        )

    if isinstance(public_text, str):
        checks.update(
            {
                "public_html_no_double_backslash": _check_no_double_backslash(public_text),
                "public_html_no_draft_labels": _check_no_substrings(public_text, ["草稿", "ドラフト"]),
                "public_html_internal_anchor_links": _check_internal_anchor_links(public_text),
                "public_figure_numbering": _check_figure_numbering(public_text),
            }
        )

    ok = all(c.ok for c in checks.values())
    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "domain": "summary",
        "step": "8.2 (paper publish qc)",
        "inputs": {
            "paper_html": str(paper_html.relative_to(_ROOT)).replace("\\", "/"),
            "part2_html": (str(part2_html.relative_to(_ROOT)).replace("\\", "/") if part2_html.exists() else None),
            "part3_html": (str(part3_html.relative_to(_ROOT)).replace("\\", "/") if part3_html.exists() else None),
            "part4_html": str(part4_html.relative_to(_ROOT)).replace("\\", "/"),
            "public_html": (str(public_html.relative_to(_ROOT)).replace("\\", "/") if public_html.exists() else None),
        },
        "ok": bool(ok),
        "checks": {k: {"ok": bool(v.ok), **v.details} for k, v in checks.items()},
        "outputs": {"json": str(out_json.relative_to(_ROOT)).replace("\\", "/")},
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("paper_qc:")
    print(f"- ok: {payload['ok']}")
    for k, v in checks.items():
        print(f"- {k}: {v.ok}")
    print(f"[ok] json: {out_json}")

    if not bool(args.no_log):
        worklog.append_event(
            {
                "tool": "paper_qc",
                "inputs": [p for p in [paper_html, part2_html, part3_html, part4_html, public_html] if p.exists()],
                "outputs": [out_json],
                "result": {"ok": bool(ok)},
            }
        )

    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
