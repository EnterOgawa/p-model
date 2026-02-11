#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_boss_dr12_ross2016_corrfunc.py

Phase 16（宇宙論）/ Step 16.4：
BAOの「圧縮出力（D_M/r_d, D_H/r_d）」ではなく、一次に近い観測統計として
Ross et al. (2016) の post-reconstruction 相関関数 multipoles（ξ0, ξ2）と共分散を取得し、
P-model側で距離変換（AP/異方歪み）を入れ替えて再導出できるようにする。

取得元（SAS; SDSS DR12 papers / clustering）:
  - Ross_etal_2016_COMBINEDDR12_corrfunc.zip

出力（固定）:
  - data/cosmology/ross_2016_combineddr12_corrfunc/（展開先）
  - data/cosmology/sources/Ross_etal_2016_COMBINEDDR12_corrfunc.zip（キャッシュ）
  - data/cosmology/ross_2016_combineddr12_corrfunc/_download_meta.json（取得メタ）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


_URL = "https://dr12.sdss3.org/sas/dr12/boss/papers/clustering/Ross_etal_2016_COMBINEDDR12_corrfunc.zip"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, out_path: Path, *, timeout_s: int = 60) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "waveP/1.0"})
    with urlopen(req, timeout=timeout_s) as resp:
        headers = {k.lower(): v for k, v in (resp.headers.items() if resp.headers else [])}
        body = resp.read()
    out_path.write_bytes(body)
    return {
        "url": url,
        "bytes": int(out_path.stat().st_size),
        "sha256": _sha256(out_path),
        "etag": headers.get("etag"),
        "last_modified": headers.get("last-modified"),
        "content_type": headers.get("content-type"),
    }


def _extract_zip(zip_path: Path, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    skipped = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = str(info.filename).replace("\\", "/")
            if not name or name.endswith("/"):
                continue
            if name.startswith("__MACOSX/") or "/__MACOSX/" in name:
                skipped += 1
                continue
            if name.endswith(".DS_Store") or "/.DS_Store" in name:
                skipped += 1
                continue

            # The zip contains a top-level folder; flatten into out_dir.
            parts = [p for p in name.split("/") if p and p not in (".", "..")]
            if len(parts) >= 2 and parts[0].lower().startswith("ross_2016"):
                rel = Path(*parts[1:])
            else:
                rel = Path(*parts)
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, dest.open("wb") as dst:
                dst.write(src.read())
            extracted += 1
    return {"files_extracted": extracted, "files_skipped": skipped}


def _looks_ready(out_dir: Path) -> bool:
    # Minimal check: one representative file from zbin3/bincent0.
    return (out_dir / "Ross_2016_COMBINEDDR12_zbin3_correlation_function_monopole_post_recon_bincent0.dat").exists()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", type=str, default=_URL)
    ap.add_argument("--force", action="store_true", help="Overwrite existing extracted files.")
    args = ap.parse_args(argv)

    data_dir = _ROOT / "data" / "cosmology"
    out_dir = data_dir / "ross_2016_combineddr12_corrfunc"
    zip_cache = data_dir / "sources" / "Ross_etal_2016_COMBINEDDR12_corrfunc.zip"
    meta_path = out_dir / "_download_meta.json"

    if out_dir.exists() and _looks_ready(out_dir) and not bool(args.force):
        print(f"[skip] already present: {out_dir}")
        return 0
    if out_dir.exists() and bool(args.force):
        # Remove only known data files; keep directory.
        for p in out_dir.glob("*"):
            if p.name.startswith("Ross_2016_COMBINEDDR12_") or p.name in ("README", "_download_meta.json"):
                try:
                    p.unlink()
                except Exception:
                    pass

    dl = _download(str(args.url), zip_cache)
    ex = _extract_zip(zip_cache, out_dir)

    meta: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "download": dl,
        "extract": ex,
        "outputs": {
            "zip": str(zip_cache.relative_to(_ROOT)).replace("\\", "/"),
            "dir": str(out_dir.relative_to(_ROOT)).replace("\\", "/"),
        },
        "notes": [
            "SAS上のzipにはMac用のメタファイルが含まれるため、__MACOSX/.DS_Store は除外して展開する。",
            "本データはRoss et al. (2016) のBAO解析で用いた post-recon ξℓ（ℓ=0,2）と共分散。",
        ],
    }
    _write_json(meta_path, meta)

    print(f"[ok] zip : {zip_cache}")
    print(f"[ok] dir : {out_dir}")
    print(f"[ok] meta: {meta_path}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_fetch_boss_dr12_ross2016_corrfunc",
                "argv": sys.argv,
                "inputs": {"url": str(args.url)},
                "outputs": {"dir": out_dir, "zip": zip_cache, "meta_json": meta_path},
            }
        )
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
