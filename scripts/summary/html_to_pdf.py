#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
html_to_pdf.py

Convert a local HTML file to PDF (for sharing) using a headless browser.

Default strategy:
  - Windows: Microsoft Edge (msedge) headless printing

Outputs:
  - output PDF file (path given by --out)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog


# クラス: `Browser` の責務と境界条件を定義する。
@dataclass(frozen=True)
class Browser:
    name: str
    exe: Path


# 関数: `_repo_root` の入出力契約と処理意図を定義する。

def _repo_root() -> Path:
    return _ROOT


# 関数: `_find_edge_windows` の入出力契約と処理意図を定義する。

def _find_edge_windows() -> Optional[Browser]:
    which = shutil.which("msedge") or shutil.which("msedge.exe")
    # 条件分岐: `which and Path(which).exists()` を満たす経路を評価する。
    if which and Path(which).exists():
        return Browser(name="edge", exe=Path(which))

    candidates: list[Path] = []
    for env in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env)
        # 条件分岐: `base` を満たす経路を評価する。
        if base:
            candidates.append(Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe")

    # Fallback hard-coded paths (common on Windows)

    candidates.extend(
        [
            Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        ]
    )

    for p in candidates:
        # 条件分岐: `p.exists()` を満たす経路を評価する。
        if p.exists():
            return Browser(name="edge", exe=p)

    return None


# 関数: `_find_browser` の入出力契約と処理意図を定義する。

def _find_browser(kind: str) -> Optional[Browser]:
    kind = (kind or "").strip().lower()

    # 条件分岐: `os.name == "nt"` を満たす経路を評価する。
    if os.name == "nt":
        edge = _find_edge_windows()
        # 条件分岐: `kind in ("auto", "edge")` を満たす経路を評価する。
        if kind in ("auto", "edge"):
            return edge

        return None

    # Non-Windows: not currently supported in this repo (best effort).

    return None


# 関数: `_to_file_uri` の入出力契約と処理意図を定義する。

def _to_file_uri(path: Path) -> str:
    return path.resolve().as_uri()


# 関数: `_run_edge_print` の入出力契約と処理意図を定義する。

def _run_edge_print(browser: Browser, html_in: Path, pdf_out: Path, *, timeout_s: int) -> None:
    pdf_out.parent.mkdir(parents=True, exist_ok=True)

    # Chromium/Edge expects an absolute output path.
    out_abs = str(pdf_out.resolve())
    in_uri = _to_file_uri(html_in)

    # NOTE: On some Windows builds, `--headless` does not reliably create PDFs, while `--headless=new` does.
    # Try the newer mode first, then fall back for older Edge versions.
    headless_args = ["--headless=new", "--headless"]
    last_err = ""
    for headless in headless_args:
        pre_size: Optional[int] = None
        pre_mtime_ns: Optional[int] = None
        # 条件分岐: `pdf_out.exists()` を満たす経路を評価する。
        if pdf_out.exists():
            try:
                st0 = pdf_out.stat()
                pre_size = int(st0.st_size)
                pre_mtime_ns = int(st0.st_mtime_ns)
            except Exception:
                pre_size = None
                pre_mtime_ns = None

            try:
                pdf_out.unlink()
            except Exception:
                # If the file is locked, keep it and later require that it actually changes.
                pass

        cmd = [
            str(browser.exe),
            headless,
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--allow-file-access-from-files",
            f"--print-to-pdf={out_abs}",
            in_uri,
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )

        # Edge may return before the PDF is fully materialized on disk.
        # Poll when rc==0 to avoid false negatives (PDF creation can lag on large pages).
        ok = False
        # 条件分岐: `proc.returncode == 0` を満たす経路を評価する。
        if proc.returncode == 0:
            for _ in range(300):  # up to ~30s
                # 条件分岐: `pdf_out.exists()` を満たす経路を評価する。
                if pdf_out.exists():
                    try:
                        st = pdf_out.stat()
                        # 条件分岐: `st.st_size > 0` を満たす経路を評価する。
                        if st.st_size > 0:
                            # If the previous file couldn't be deleted, ensure it actually changed.
                            if pre_mtime_ns is None:
                                ok = True
                                break

                            # 条件分岐: `int(st.st_mtime_ns) != pre_mtime_ns or int(st.st_size) != (pre_size or 0)` を満たす経路を評価する。

                            if int(st.st_mtime_ns) != pre_mtime_ns or int(st.st_size) != (pre_size or 0):
                                ok = True
                                break
                    except Exception:
                        pass

                time.sleep(0.1)

        # 条件分岐: `ok` を満たす経路を評価する。

        if ok:
            return

        last_err = f"headless={headless} rc={proc.returncode}\n{proc.stdout}"

    raise RuntimeError(f"edge print failed:\n{last_err}")


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Convert a local HTML file to PDF (headless Edge).")
    ap.add_argument("--in", dest="html_in", required=True, help="Input HTML file path.")
    ap.add_argument("--out", dest="pdf_out", required=True, help="Output PDF file path.")
    ap.add_argument(
        "--browser",
        choices=["auto", "edge"],
        default="auto",
        help="Browser backend (default: auto).",
    )
    ap.add_argument("--timeout-s", type=int, default=120, help="Timeout seconds (default: 120).")
    args = ap.parse_args(argv)

    root = _repo_root()
    html_in = Path(args.html_in)
    pdf_out = Path(args.pdf_out)

    # 条件分岐: `not html_in.is_absolute()` を満たす経路を評価する。
    if not html_in.is_absolute():
        html_in = (root / html_in).resolve()

    # 条件分岐: `not pdf_out.is_absolute()` を満たす経路を評価する。

    if not pdf_out.is_absolute():
        pdf_out = (root / pdf_out).resolve()

    # 条件分岐: `not html_in.exists()` を満たす経路を評価する。

    if not html_in.exists():
        print(f"[err] input HTML not found: {html_in}", file=sys.stderr)
        return 2

    browser = _find_browser(str(args.browser))
    # 条件分岐: `not browser` を満たす経路を評価する。
    if not browser:
        print("[warn] no supported browser found for HTML→PDF conversion; skipping.", file=sys.stderr)
        return 3

    try:
        _run_edge_print(browser, html_in, pdf_out, timeout_s=int(args.timeout_s))
    except Exception as e:
        print(f"[err] HTML→PDF failed: {e}", file=sys.stderr)
        return 1

    try:
        worklog.append_event(
            {
                "event_type": "html_to_pdf",
                "inputs": {"html": html_in},
                "params": {"browser": browser.name, "timeout_s": int(args.timeout_s)},
                "outputs": {"pdf": pdf_out},
            }
        )
    except Exception:
        pass

    print(f"[ok] pdf: {pdf_out}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
