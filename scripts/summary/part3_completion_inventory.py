#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
part3_completion_inventory.py

Phase 7 / Step 7.18.1:
Part III（量子）論文（doc/paper/12_part3_quantum.md）の各セクションについて、
Input / Frozen / Statistic / Reject / Output が揃っているかを棚卸しし、
欠けている行（=欠落項目がある行）だけを固定出力する。

出力（固定）:
  - output/public/summary/part3_completion_inventory.json
  - output/public/summary/part3_completion_inventory.md

注意:
- publish モードに合わせ、INTERNAL_ONLY ブロックは検査対象から除外する。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relpath(p: Path) -> str:
    try:
        return str(p.relative_to(_ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


_RE_HEADING = re.compile(r"^(#{2,6})\s+(?P<title>.+?)\s*$")
_RE_SECTION_NUM = re.compile(r"^(?P<num>\d+(?:\.\d+){0,5})\s+(?P<title>.+?)\s*$")
_RE_BOLD_FIELD = re.compile(r"^\*\*(?P<name>[^*]+)\*\*：\s*(?P<rest>.*)$")

_INTERNAL_ONLY_START = "<!-- INTERNAL_ONLY_START -->"
_INTERNAL_ONLY_END = "<!-- INTERNAL_ONLY_END -->"


@dataclass(frozen=True)
class MdSection:
    level: int
    heading_line_no: int  # 1-based
    heading_line_idx: int  # 0-based
    section_num: Optional[str]
    title: str
    body_start_idx: int  # 0-based
    body_end_idx: int  # 0-based, exclusive


def _strip_internal_lines(lines: Sequence[str]) -> List[str]:
    out: List[str] = []
    in_internal = False
    for line in lines:
        if _INTERNAL_ONLY_START in line:
            in_internal = True
            continue
        if _INTERNAL_ONLY_END in line:
            in_internal = False
            continue
        if in_internal:
            continue
        out.append(line)
    return out


def _parse_sections(lines: Sequence[str]) -> List[MdSection]:
    headings: List[Tuple[int, int, int, Optional[str], str]] = []
    # (level, line_no, line_idx, section_num, title)
    for i, raw in enumerate(lines):
        m = _RE_HEADING.match(raw)
        if not m:
            continue
        level = len(m.group(1))
        title_full = m.group("title").strip()
        sec_num = None
        sec_title = title_full
        mn = _RE_SECTION_NUM.match(title_full)
        if mn:
            sec_num = mn.group("num")
            sec_title = mn.group("title").strip()
        headings.append((level, i + 1, i, sec_num, sec_title))

    sections: List[MdSection] = []
    for idx, (level, line_no, line_idx, sec_num, sec_title) in enumerate(headings):
        end_idx_excl = len(lines)
        for j in range(idx + 1, len(headings)):
            next_level, _, next_line_idx, _, _ = headings[j]
            if next_level <= level:
                end_idx_excl = next_line_idx
                break
        sections.append(
            MdSection(
                level=level,
                heading_line_no=line_no,
                heading_line_idx=line_idx,
                section_num=sec_num,
                title=sec_title,
                body_start_idx=line_idx + 1,
                body_end_idx=end_idx_excl,
            )
        )
    return sections


def _is_candidate_section(sec: MdSection) -> bool:
    if not sec.section_num:
        return False
    return sec.section_num.startswith("4.2.") or sec.section_num.startswith("5.")


def _is_container(sec: MdSection, *, all_secs: Sequence[MdSection]) -> bool:
    if not sec.section_num:
        return False
    prefix = sec.section_num + "."
    for other in all_secs:
        if other.heading_line_idx <= sec.heading_line_idx:
            continue
        if other.heading_line_idx >= sec.body_end_idx:
            continue
        if other.section_num and other.section_num.startswith(prefix):
            return True
    return False


def _category_for(sec_num: str, title: str) -> str:
    # Primary mapping by section number (stable within Part III).
    if sec_num == "4.2.1" or sec_num.startswith("4.2.1.") or sec_num == "5.1" or sec_num.startswith("5.1."):
        return "Bell"
    if (
        sec_num == "5.2"
        or sec_num.startswith("5.2.")
        or sec_num == "4.2.2"
        or sec_num.startswith("4.2.2.")
        or sec_num == "4.2.3"
        or sec_num.startswith("4.2.3.")
        or sec_num == "4.2.4"
        or sec_num.startswith("4.2.4.")
        or sec_num == "4.2.5"
        or sec_num.startswith("4.2.5.")
    ):
        return "干渉"
    if (
        sec_num == "5.3"
        or sec_num.startswith("5.3.")
        or sec_num == "5.4"
        or sec_num.startswith("5.4.")
        or sec_num == "4.2.7"
        or sec_num.startswith("4.2.7.")
    ):
        return "核"
    if (
        sec_num == "4.2.15"
        or sec_num.startswith("4.2.15.")
        or sec_num == "4.2.16"
        or sec_num.startswith("4.2.16.")
    ):
        return "熱"
    if (
        sec_num == "4.2.6"
        or sec_num.startswith("4.2.6.")
        or sec_num == "4.2.8"
        or sec_num.startswith("4.2.8.")
    ):
        return "物性"
    if sec_num.startswith("4.2.") and re.match(r"^4\.2\.(9|10|11|12|13|14)(?:\.|$)", sec_num):
        return "物性"

    # Fallback by keywords (for future extension).
    t = title
    if "Bell" in t or "ベル" in t:
        return "Bell"
    if "干渉" in t or "interference" in t.lower() or "de Broglie" in t:
        return "干渉"
    if "核" in t or "deuteron" in t.lower():
        return "核"
    if "熱" in t or "thermo" in t.lower() or "黒体" in t:
        return "熱"
    if "物性" in t or "凝縮" in t or "QED" in t:
        return "物性"
    return "未分類"


def _extract_bold_field(section_lines: Sequence[str], field_name: str) -> Optional[str]:
    for i, line in enumerate(section_lines):
        m = _RE_BOLD_FIELD.match(line)
        if not m:
            continue
        name = (m.group("name") or "").strip()
        if name != field_name:
            continue
        buf: List[str] = [m.group("rest") or ""]
        for j in range(i + 1, len(section_lines)):
            if _RE_BOLD_FIELD.match(section_lines[j]) or _RE_HEADING.match(section_lines[j]):
                break
            buf.append(section_lines[j])
        return "\n".join(buf)
    return None


def _is_meaningful(text: Optional[str]) -> bool:
    if text is None:
        return False
    t = text.strip()
    if not t:
        return False
    t_norm = t.replace("`", "").strip().lower()
    if t_norm in {"tbd", "n/a", "na", "—", "-", "なし", "未定"}:
        return False
    if t_norm.replace("-", "").strip() == "":
        return False
    if re.fullmatch(r"[—\-\s]+", t):
        return False
    return True


def _has_output_reference(text: str) -> bool:
    return "output/" in text.replace("\\", "/")


def _has_frozen_marker(text: str) -> bool:
    t = text
    if "凍結" in t or "凍結値" in t:
        return True
    if "固定" in t:
        return True
    if "固定値" in t:
        return True
    tl = t.lower()
    if "frozen_parameters" in tl or "frozen parameter" in tl or "freeze" in tl:
        return True
    return False


def _has_reject_marker(text: str) -> bool:
    t = text
    if "棄却条件" in t:
        return True
    if "棄却" in t:
        return True
    tl = t.lower()
    if "reject" in tl or "no-go" in tl or "pass/fail" in tl:
        return True
    return False


def build_inventory(*, paper_md: Path) -> Dict[str, Any]:
    raw_lines = paper_md.read_text(encoding="utf-8").splitlines()
    all_sections = _parse_sections(raw_lines)

    candidate_sections = [s for s in all_sections if _is_candidate_section(s)]
    leaf_sections = [s for s in candidate_sections if not _is_container(s, all_secs=candidate_sections)]

    missing_by_cat: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ("Bell", "核", "物性", "熱", "干渉", "未分類")}
    scanned_n = 0

    for sec in leaf_sections:
        scanned_n += 1
        body_lines = raw_lines[sec.body_start_idx : sec.body_end_idx]
        body_lines_pub = _strip_internal_lines(body_lines)
        body_text = "\n".join(body_lines_pub)

        input_text = _extract_bold_field(body_lines_pub, "入力")
        stat_text = _extract_bold_field(body_lines_pub, "指標")
        output_text = _extract_bold_field(body_lines_pub, "出力")

        found = {
            "Input": _is_meaningful(input_text),
            "Frozen": _has_frozen_marker(body_text),
            "Statistic": _is_meaningful(stat_text),
            "Reject": _has_reject_marker(body_text),
            # Prefer prose-based Output (per user request: avoid file names in-body),
            # but keep backward-compatibility with legacy sections that only listed output paths.
            "Output": _is_meaningful(output_text) or _has_output_reference(body_text),
        }
        missing = [k for k, ok in found.items() if not ok]
        if not missing:
            continue

        sec_num = sec.section_num or ""
        cat = _category_for(sec_num, sec.title)
        if cat not in missing_by_cat:
            cat = "未分類"

        missing_by_cat[cat].append(
            {
                "section": sec_num,
                "title": sec.title,
                "heading_line": sec.heading_line_no,
                "category": cat,
                "missing": missing,
                "found": found,
            }
        )

    missing_sections_n = sum(len(v) for v in missing_by_cat.values())
    payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "source": {"paper_md": _relpath(paper_md)},
        "required": ["Input", "Frozen", "Statistic", "Reject", "Output"],
        "rules": {
            "Input": "Detect **入力**： and require non-empty/non-placeholder content (publish view).",
            "Frozen": "Detect markers: 凍結/凍結値/固定/固定値 or frozen_parameters/freeze (publish view).",
            "Statistic": "Detect **指標**： and require non-empty/non-placeholder content (publish view).",
            "Reject": "Detect markers: 棄却条件/棄却/reject/no-go/pass-fail (publish view).",
            "Output": "Detect **出力**： (publish view). Fallback: accept an 'output/' reference (legacy).",
            "note": "INTERNAL_ONLY blocks are excluded (publish mode behavior).",
        },
        "summary": {"sections_scanned": scanned_n, "missing_sections": missing_sections_n},
        "missing_by_category": {k: v for k, v in missing_by_cat.items() if v},
    }
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_md(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Part III 完成条件棚卸し（欠落のみ）")
    lines.append("")
    lines.append(f"- generated_utc: {payload.get('generated_utc')}")
    lines.append(f"- source: `{payload.get('source', {}).get('paper_md')}`")
    lines.append(f"- sections_scanned: {payload.get('summary', {}).get('sections_scanned')}")
    lines.append(f"- missing_sections: {payload.get('summary', {}).get('missing_sections')}")
    lines.append("")
    lines.append("## ルール（検出）")
    lines.append("")
    rules = payload.get("rules", {})
    for k in ("Input", "Frozen", "Statistic", "Reject", "Output"):
        if k in rules:
            lines.append(f"- {k}: {rules.get(k)}")
    if "note" in rules:
        lines.append(f"- note: {rules.get('note')}")
    lines.append("")

    missing_by_cat = payload.get("missing_by_category", {}) or {}
    order = ["Bell", "干渉", "核", "物性", "熱", "未分類"]
    for cat in order:
        items = missing_by_cat.get(cat)
        if not items:
            continue
        lines.append(f"## {cat}")
        lines.append("")
        for it in items:
            sec = it.get("section")
            title = it.get("title")
            ln = it.get("heading_line")
            missing = it.get("missing") or []
            loc = f"doc/paper/12_part3_quantum.md:{ln}"
            lines.append(f"- {sec} {title}（{loc}） missing={', '.join(missing)}")
        lines.append("")

    if payload.get("summary", {}).get("missing_sections", 0) == 0:
        lines.append("- 欠落は検出されなかった。")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Inventory Part III completeness (Input/Frozen/Statistic/Reject/Output) and emit missing-only list.")
    ap.add_argument(
        "--paper",
        type=str,
        default=str(_ROOT / "doc" / "paper" / "12_part3_quantum.md"),
        help="Path to Part III paper markdown (default: doc/paper/12_part3_quantum.md).",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(_ROOT / "output" / "public" / "summary"),
        help="Output directory (default: output/public/summary).",
    )
    args = ap.parse_args(argv)

    paper_md = Path(args.paper)
    if not paper_md.is_absolute():
        paper_md = (_ROOT / paper_md).resolve()
    if not paper_md.exists():
        raise SystemExit(f"[error] paper not found: {paper_md}")

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = (_ROOT / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    payload = build_inventory(paper_md=paper_md)

    out_json = outdir / "part3_completion_inventory.json"
    out_md = outdir / "part3_completion_inventory.md"
    _write_json(out_json, payload)
    _write_md(out_md, payload)

    print(f"[ok] wrote: {_relpath(out_json)}")
    print(f"[ok] wrote: {_relpath(out_md)}")
    try:
        worklog.append_event(
            {
                "event_type": "part3_completion_inventory",
                "phase": "7.18.1",
                "inputs": {"paper_md": _relpath(paper_md)},
                "outputs": {"json": _relpath(out_json), "md": _relpath(out_md)},
                "summary": payload.get("summary"),
            }
        )
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
