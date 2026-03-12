#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_page_audit.py

論文PDFの「該当ページに本当に最新図が入っているか」を確認するための補助監査。

目的:
- 図単体の PDF だけでなく、最終的な `papers/*.pdf` / `output/private/summary/*.pdf`
  の該当ページを PNG にレンダリングして、stale 参照や viewer cache 誤認を減らす。
- caption 断片や固定の文字列からページを特定し、同一ターンで確認用 PNG を残す。

出力:
- 既定: `output/private/summary/page_audit/*.png`
- 監査メタ: `output/private/summary/page_audit/paper_page_audit.json`
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = ROOT / "output" / "private" / "summary" / "page_audit"
PDFTOTEXT = Path(r"C:\texlive\2024\bin\windows\pdftotext.exe")
PDFTOPPM = Path(r"C:\texlive\2024\bin\windows\pdftoppm.exe")


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。
def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_run` の入出力契約と処理意図を定義する。
def _run(argv: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")


# 関数: `_extract_pdf_page_text` の入出力契約と処理意図を定義する。
def _extract_pdf_page_text(pdf_path: Path, *, page_num: int) -> str:
    completed = _run(
        [
            str(PDFTOTEXT),
            "-f",
            str(page_num),
            "-l",
            str(page_num),
            "-layout",
            str(pdf_path),
            "-",
        ]
    )
    return completed.stdout.replace("\r\n", "\n")


# 関数: `_find_matching_pages` の入出力契約と処理意図を定義する。
def _find_matching_pages(pdf_path: Path, *, page_count: int, pattern: str) -> List[int]:
    regex = re.compile(pattern, flags=re.IGNORECASE)
    matches: List[int] = []
    for idx in range(1, page_count + 1):
        text = _extract_pdf_page_text(pdf_path, page_num=idx)
        # 条件分岐: `regex.search(text)` を満たす経路を評価する。
        if regex.search(text):
            matches.append(idx)

    return matches


# 関数: `_pdf_page_count` の入出力契約と処理意図を定義する。
def _pdf_page_count(pdf_path: Path) -> int:
    return len(PdfReader(str(pdf_path)).pages)


# 関数: `_render_page_png` の入出力契約と処理意図を定義する。
def _render_page_png(pdf_path: Path, *, page_num: int, out_prefix: Path) -> Path:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(PDFTOPPM),
            "-f",
            str(page_num),
            "-l",
            str(page_num),
            "-png",
            str(pdf_path),
            str(out_prefix),
        ]
    )

    candidates = sorted(out_prefix.parent.glob(f"{out_prefix.name}-*.png"))
    # 条件分岐: `not candidates` を満たす経路を評価する。
    if not candidates:
        raise FileNotFoundError(f"render output missing for page {page_num}: {out_prefix}")

    return candidates[-1]


# 関数: `main` の入出力契約と処理意図を定義する。
def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render matching paper PDF pages to PNG for stale-reference audits.")
    ap.add_argument("--pdf", required=True, help="Target paper PDF path.")
    ap.add_argument(
        "--pattern",
        required=True,
        help="Regex pattern searched in pdftotext page text (example: '図 24|GW150914').",
    )
    ap.add_argument("--tag", default="page_audit", help="Filename stem tag for rendered PNGs.")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output directory for rendered pages.")
    ap.add_argument(
        "--max-matches",
        type=int,
        default=0,
        help="Maximum number of matched pages to render (0 = no limit).",
    )
    args = ap.parse_args(argv)

    pdf_path = Path(args.pdf)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 条件分岐: `not PDFTOTEXT.exists()` を満たす経路を評価する。
    if not PDFTOTEXT.exists():
        print(f"[err] missing pdftotext: {PDFTOTEXT}")
        return 2

    # 条件分岐: `not PDFTOPPM.exists()` を満たす経路を評価する。
    if not PDFTOPPM.exists():
        print(f"[err] missing pdftoppm: {PDFTOPPM}")
        return 2

    # 条件分岐: `not pdf_path.exists()` を満たす経路を評価する。
    if not pdf_path.exists():
        print(f"[err] missing pdf: {pdf_path}")
        return 2

    page_count = _pdf_page_count(pdf_path)
    matched_pages = _find_matching_pages(pdf_path, page_count=page_count, pattern=str(args.pattern))
    if int(args.max_matches) > 0:
        matched_pages = matched_pages[: int(args.max_matches)]
    # 条件分岐: `not matched_pages` を満たす経路を評価する。
    if not matched_pages:
        print(f"[err] no pages matched pattern: {args.pattern}")
        return 1

    rendered: List[Dict[str, Any]] = []
    for page_num in matched_pages:
        prefix = outdir / f"{args.tag}_page{page_num:03d}"
        png_path = _render_page_png(pdf_path, page_num=page_num, out_prefix=prefix)
        rendered.append(
            {
                "page_num": page_num,
                "png": str(png_path).replace("\\", "/"),
            }
        )
        print(f"[ok] page {page_num}: {png_path}")

    audit_json = outdir / "paper_page_audit.json"
    payload = {
        "generated_utc": _iso_utc_now(),
        "pdf": str(pdf_path).replace("\\", "/"),
        "pattern": str(args.pattern),
        "matched_pages": matched_pages,
        "rendered": rendered,
    }
    audit_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] audit: {audit_json}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。
if __name__ == "__main__":
    raise SystemExit(main())
