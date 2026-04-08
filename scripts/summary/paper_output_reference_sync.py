#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_output_reference_sync.py

Normalize manuscript-side output references so `doc/paper/*.md` stays aligned
with the current source-of-truth policy.

What it does:
- rewrite legacy `output/<topic>/...` references to `output/public/<topic>/...`
- prefer sibling `.pdf` assets over `.png` when a public PDF exists
- emit an internal audit report under `output/private/summary/`

This script intentionally edits only markdown manuscripts. It does not build
HTML/TeX/PDF by itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import paper_profile_content as profile_content, worklog  # noqa: E402

_CODE_OUTPUT_PATH_RE = re.compile(r"`(output/[^`\r\n]+)`")


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_default_manuscripts` の入出力契約と処理意図を定義する。

def _default_manuscripts() -> List[Path]:
    profiles = (
        profile_content.PART3A_PROFILE,
        profile_content.PART3B_PROFILE,
        "part4_verification",
    )
    return [profile_content.resolve_manuscript_path(_ROOT, profile) for profile in profiles]


# 関数: `_normalize_output_path` の入出力契約と処理意図を定義する。

def _normalize_output_path(raw_path: str) -> str:
    return raw_path.replace("\\", "/").strip()


# 関数: `_rewrite_legacy_output_root` の入出力契約と処理意図を定義する。

def _rewrite_legacy_output_root(path_text: str) -> tuple[str, bool]:
    normalized = _normalize_output_path(path_text)
    trimmed = normalized.rstrip("/")
    if Path(trimmed).suffix == "":
        return normalized, False

    parts = Path(normalized).parts
    if len(parts) < 2:
        return normalized, False

    if parts[0] != "output":
        return normalized, False

    if parts[1] in ("public", "private"):
        return normalized, False

    rewritten = str(Path("output") / "public" / Path(*parts[1:])).replace("\\", "/")
    return rewritten, rewritten != normalized


# 関数: `_prefer_pdf_if_present` の入出力契約と処理意図を定義する。

def _prefer_pdf_if_present(path_text: str) -> tuple[str, bool]:
    normalized = _normalize_output_path(path_text)
    if not normalized.lower().endswith(".png"):
        return normalized, False

    pdf_rel = normalized[:-4] + ".pdf"
    pdf_path = _ROOT / Path(pdf_rel)
    if not pdf_path.exists():
        return normalized, False

    return pdf_rel, True


# クラス: `_RewriteRecord` の責務と境界条件を定義する。

@dataclass(frozen=True)
class _RewriteRecord:
    manuscript: str
    original: str
    updated: str
    changed_legacy_root: bool
    changed_png_to_pdf: bool


# 関数: `_rewrite_one_manuscript` の入出力契約と処理意図を定義する。

def _rewrite_one_manuscript(path: Path, *, apply: bool) -> Dict[str, Any]:
    original_text = path.read_text(encoding="utf-8")
    records: List[_RewriteRecord] = []

    # 関数: `repl` の入出力契約と処理意図を定義する。
    def repl(match: re.Match[str]) -> str:
        raw = match.group(1)
        rewritten, changed_legacy = _rewrite_legacy_output_root(raw)
        preferred, changed_pdf = _prefer_pdf_if_present(rewritten)
        if (preferred != raw) or changed_legacy or changed_pdf:
            records.append(
                _RewriteRecord(
                    manuscript=str(path.relative_to(_ROOT)).replace("\\", "/"),
                    original=raw,
                    updated=preferred,
                    changed_legacy_root=changed_legacy,
                    changed_png_to_pdf=changed_pdf,
                )
            )

        return f"`{preferred}`"

    updated_text = _CODE_OUTPUT_PATH_RE.sub(repl, original_text)
    changed = updated_text != original_text
    if changed and apply:
        path.write_text(updated_text, encoding="utf-8")

    remaining_legacy = []
    for ref in _CODE_OUTPUT_PATH_RE.findall(updated_text):
        normalized = _normalize_output_path(ref)
        trimmed = normalized.rstrip("/")
        if Path(trimmed).suffix == "":
            continue
        if not normalized.startswith("output/"):
            continue
        if Path(normalized).parts[1] in ("public", "private"):
            continue
        remaining_legacy.append(normalized)

    remaining_png_with_pdf: List[str] = []
    for ref in _CODE_OUTPUT_PATH_RE.findall(updated_text):
        normalized = _normalize_output_path(ref)
        if not normalized.lower().endswith(".png"):
            continue

        pdf_rel = normalized[:-4] + ".pdf"
        if (_ROOT / Path(pdf_rel)).exists():
            remaining_png_with_pdf.append(normalized)

    return {
        "manuscript": str(path.relative_to(_ROOT)).replace("\\", "/"),
        "changed": changed,
        "rewrite_count": len(records),
        "legacy_root_rewrite_count": int(sum(1 for record in records if record.changed_legacy_root)),
        "png_to_pdf_rewrite_count": int(sum(1 for record in records if record.changed_png_to_pdf)),
        "remaining_legacy_root_refs": remaining_legacy,
        "remaining_png_refs_with_pdf_sibling": remaining_png_with_pdf,
        "records": [record.__dict__ for record in records],
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows_list = list(rows)
    if not rows_list:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows_list[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Normalize manuscript output references to current canonical policy.")
    ap.add_argument("--manuscript", action="append", help="target manuscript path (repeatable)")
    ap.add_argument("--outdir", default=str(_ROOT / "output" / "private" / "summary"))
    ap.add_argument("--check", action="store_true", help="report only; do not modify manuscripts")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manuscripts = [Path(p) for p in args.manuscript] if args.manuscript else _default_manuscripts()
    manuscripts = [path if path.is_absolute() else (_ROOT / path) for path in manuscripts]
    results: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []
    all_ok = True
    apply = not bool(args.check)

    for manuscript in manuscripts:
        result = _rewrite_one_manuscript(manuscript, apply=apply)
        results.append(result)
        all_ok = all_ok and (not result["remaining_legacy_root_refs"]) and (not result["remaining_png_refs_with_pdf_sibling"])
        flat_rows.extend(result["records"])

    payload = {
        "generated_utc": _utc_now(),
        "apply_mode": "check" if args.check else "rewrite",
        "ok": all_ok,
        "manuscripts": [str(path.relative_to(_ROOT)).replace("\\", "/") for path in manuscripts],
        "results": results,
    }

    json_out = outdir / "paper_output_reference_sync.json"
    csv_out = outdir / "paper_output_reference_sync.csv"
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_out, flat_rows)

    print(f"[ok] wrote: {json_out}")
    print(f"[ok] wrote: {csv_out}")
    print(f"paper_output_reference_sync: ok={all_ok}")
    for result in results:
        print(
            f"- {result['manuscript']}: changed={result['changed']} rewrites={result['rewrite_count']} "
            f"remaining_legacy={len(result['remaining_legacy_root_refs'])} "
            f"remaining_png_with_pdf={len(result['remaining_png_refs_with_pdf_sibling'])}"
        )

    try:
        worklog.append_event(
            {
                "event_type": "paper_output_reference_sync",
                "ok": all_ok,
                "apply_mode": "check" if args.check else "rewrite",
                "manuscripts": manuscripts,
                "json_out": json_out,
                "csv_out": csv_out,
            }
        )
    except Exception:
        pass

    return 0 if all_ok else 1


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
