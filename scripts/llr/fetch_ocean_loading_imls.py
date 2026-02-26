#!/usr/bin/env python3
"""
fetch_ocean_loading_imls.py

目的:
  LLR（月レーザー測距）の ns級モデル化で必要になる「海洋潮汐荷重（tidal ocean loading; TOC）」の
  一次ソース候補（NASA/IMLS: International Mass Loading Service）を取得し、
  data/llr/ocean_loading/ に固定ファイル名でキャッシュする。

入力（ネットワーク）:
  - IMLS HARPOS list (TOC / FES2014b):
    https://massloading.smce.nasa.gov/imls/load_har_list/toc/fes2014b/toc_fes2014b_harmod.hps

出力（固定）:
  - data/llr/ocean_loading/toc_fes2014b_harmod.hps
  - data/llr/ocean_loading/toc_fes2014b_harmod.meta.json

備考:
  - 2回目以降はオフライン再現（キャッシュ利用）を前提とするため、
    取得したバイナリの sha256 と取得時刻（UTC）をメタJSONに保存する。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


IMLS_TOC_FES2014B_URL = (
    "https://massloading.smce.nasa.gov/imls/load_har_list/toc/fes2014b/toc_fes2014b_harmod.hps"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def _download(url: str, dst: Path, *, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `dst.exists() and not force` を満たす経路を評価する。
    if dst.exists() and not force:
        print(f"[skip] exists: {dst}")
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[dl] {url}")
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)

    tmp.replace(dst)
    print(f"[ok] saved: {dst} ({dst.stat().st_size} bytes)")


def main() -> int:
    root = _repo_root()

    ap = argparse.ArgumentParser(description="Fetch IMLS tidal ocean loading HARPOS (TOC/FES2014b) for LLR.")
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(root / "data" / "llr" / "ocean_loading"),
        help="Output directory (default: data/llr/ocean_loading).",
    )
    ap.add_argument("--force", action="store_true", help="Re-download even if cached file exists.")
    ap.add_argument(
        "--url",
        type=str,
        default=IMLS_TOC_FES2014B_URL,
        help="Override download URL (default: IMLS TOC FES2014b).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    # 条件分岐: `not out_dir.is_absolute()` を満たす経路を評価する。
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    dst = out_dir / "toc_fes2014b_harmod.hps"
    _download(str(args.url), dst, force=bool(args.force))

    meta = {
        "source": "NASA/IMLS load_har_list (HARPOS)",
        "kind": "TOC (tidal ocean loading) harmonics",
        "model": "FES2014b",
        "url": str(args.url),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "bytes": int(dst.stat().st_size),
        "sha256": _sha256(dst),
    }
    meta_path = out_dir / "toc_fes2014b_harmod.meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] meta: {meta_path}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())

