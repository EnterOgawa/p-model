#!/usr/bin/env python3
"""
fetch_moon_kernels_naif.py

LLR の ns 級モデルに必要な NAIF/SPICE カーネル（Moon DE421 PA）を取得して
data/llr/kernels/naif/ にキャッシュする。

目的:
  - 月の回転（MOON_PA_DE421 / MOON_ME_DE421）を高精度に扱う
  - 2回目以降は data/ 配下のカーネルでオフライン再現する

出力（固定）
  - data/llr/kernels/naif/naif0012.tls
  - data/llr/kernels/naif/pck00010.tpc
  - data/llr/kernels/naif/moon_080317.tf
  - data/llr/kernels/naif/moon_pa_de421_1900-2050.bpc
  - data/llr/kernels/naif/moon_pa_de421_1900-2050.cmt
  - data/llr/kernels/naif/kernels_meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Kernel:
    name: str
    url: str


KERNELS: list[Kernel] = [
    Kernel(
        name="naif0012.tls",
        url="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
    ),
    Kernel(
        name="pck00010.tpc",
        url="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc",
    ),
    Kernel(
        name="moon_080317.tf",
        url="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/moon_080317.tf",
    ),
    Kernel(
        name="moon_pa_de421_1900-2050.bpc",
        url="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/moon_pa_de421_1900-2050.bpc",
    ),
    Kernel(
        name="moon_pa_de421_1900-2050.cmt",
        url="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/moon_pa_de421_1900-2050.cmt",
    ),
]


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
    if dst.exists() and not force:
        print(f"[skip] exists: {dst}")
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[dl] {url}")
    with urllib.request.urlopen(url, timeout=180) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)
    tmp.replace(dst)
    print(f"[ok] saved: {dst} ({dst.stat().st_size} bytes)")


def main() -> int:
    root = _repo_root()
    out_dir = root / "data" / "llr" / "kernels" / "naif"

    ap = argparse.ArgumentParser(description="Fetch NAIF kernels needed for Moon DE421 PA orientation.")
    ap.add_argument("--offline", action="store_true", help="Do not use network; only succeed if all kernels exist.")
    ap.add_argument("--force", action="store_true", help="Re-download and overwrite cached kernels.")
    args = ap.parse_args()

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.offline:
        missing = [k.name for k in KERNELS if not (out_dir / k.name).exists()]
        if missing:
            print("[err] offline and missing kernels:")
            for m in missing:
                print("  -", m)
            return 2
        print(f"[ok] offline: {out_dir}")
        return 0

    for k in KERNELS:
        _download(k.url, out_dir / k.name, force=bool(args.force))

    meta: dict = {
        "source": "NAIF generic_kernels",
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "kernels": [],
    }
    for k in KERNELS:
        p = out_dir / k.name
        meta["kernels"].append(
            {
                "name": k.name,
                "url": k.url,
                "bytes": int(p.stat().st_size) if p.exists() else None,
                "sha256": _sha256(p) if p.exists() else None,
            }
        )

    meta_path = out_dir / "kernels_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] meta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

