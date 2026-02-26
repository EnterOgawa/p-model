#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_boss_dr12_satpathy2016_corrfunc_multipoles.py

Phase 16（宇宙論）/ Step 16.4（BAO一次統計の再導出）:
Satpathy et al. (2016) の pre-reconstruction 相関関数 multipoles（ξ0, ξ2）と共分散を取得する。

取得元（SAS; SDSS DR12 papers / clustering）:
  - Satpathy_etal_2016_COMBINEDDR12_full_shape_corrfunc_multipoles.zip

出力（固定）:
  - data/cosmology/satpathy_2016_combineddr12_fs_corrfunc_multipoles/（展開先）
  - data/cosmology/sources/Satpathy_etal_2016_COMBINEDDR12_full_shape_corrfunc_multipoles.zip（キャッシュ）
  - data/cosmology/satpathy_2016_combineddr12_fs_corrfunc_multipoles/_download_meta.json（取得メタ）
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
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


_URL = "https://dr12.sdss3.org/sas/dr12/boss/papers/clustering/Satpathy_etal_2016_COMBINEDDR12_full_shape_corrfunc_multipoles.zip"

_KEEP_FILES = {
    "Satpathy_2016_COMBINEDDR12_Bin1_Monopole_pre_recon.dat",
    "Satpathy_2016_COMBINEDDR12_Bin1_Quadrupole_pre_recon.dat",
    "Satpathy_2016_COMBINEDDR12_Bin1_Covariance_pre_recon.txt",
    "Satpathy_2016_COMBINEDDR12_Bin2_Monopole_pre_recon.dat",
    "Satpathy_2016_COMBINEDDR12_Bin2_Quadrupole_pre_recon.dat",
    "Satpathy_2016_COMBINEDDR12_Bin2_Covariance_pre_recon.txt",
    "Satpathy_2016_COMBINEDDR12_Bin3_Monopole_pre_recon.dat",
    "Satpathy_2016_COMBINEDDR12_Bin3_Quadrupole_pre_recon.dat",
    "Satpathy_2016_COMBINEDDR12_Bin3_CovarianceMatrix_pre_recon.txt",
}


# 関数: `_sha256` の入出力契約と処理意図を定義する。
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


# 関数: `_download` の入出力契約と処理意図を定義する。

def _download(url: str, out_path: Path, *, timeout_s: int = 90) -> Dict[str, Any]:
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


# 関数: `_extract_zip` の入出力契約と処理意図を定義する。

def _extract_zip(zip_path: Path, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    skipped = 0
    kept = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = str(info.filename).replace("\\", "/")
            # 条件分岐: `not name or name.endswith("/")` を満たす経路を評価する。
            if not name or name.endswith("/"):
                continue

            # 条件分岐: `name.startswith("__MACOSX/") or "/__MACOSX/" in name` を満たす経路を評価する。

            if name.startswith("__MACOSX/") or "/__MACOSX/" in name:
                skipped += 1
                continue

            # 条件分岐: `name.endswith(".DS_Store") or "/.DS_Store" in name` を満たす経路を評価する。

            if name.endswith(".DS_Store") or "/.DS_Store" in name:
                skipped += 1
                continue

            base = name.split("/")[-1]
            # 条件分岐: `base not in _KEEP_FILES` を満たす経路を評価する。
            if base not in _KEEP_FILES:
                skipped += 1
                continue

            dest = out_dir / base
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, dest.open("wb") as dst:
                dst.write(src.read())

            extracted += 1
            kept += 1

    return {"files_extracted": extracted, "files_kept": kept, "files_skipped": skipped}


# 関数: `_looks_ready` の入出力契約と処理意図を定義する。

def _looks_ready(out_dir: Path) -> bool:
    return (out_dir / "Satpathy_2016_COMBINEDDR12_Bin1_Monopole_pre_recon.dat").exists() and (
        out_dir / "Satpathy_2016_COMBINEDDR12_Bin3_CovarianceMatrix_pre_recon.txt"
    ).exists()


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", type=str, default=_URL)
    ap.add_argument("--force", action="store_true", help="Overwrite existing extracted files.")
    args = ap.parse_args(argv)

    data_dir = _ROOT / "data" / "cosmology"
    out_dir = data_dir / "satpathy_2016_combineddr12_fs_corrfunc_multipoles"
    zip_cache = data_dir / "sources" / "Satpathy_etal_2016_COMBINEDDR12_full_shape_corrfunc_multipoles.zip"
    meta_path = out_dir / "_download_meta.json"

    # 条件分岐: `out_dir.exists() and _looks_ready(out_dir) and not bool(args.force)` を満たす経路を評価する。
    if out_dir.exists() and _looks_ready(out_dir) and not bool(args.force):
        print(f"[skip] already present: {out_dir}")
        return 0

    # 条件分岐: `out_dir.exists() and bool(args.force)` を満たす経路を評価する。

    if out_dir.exists() and bool(args.force):
        for p in out_dir.glob("*"):
            # 条件分岐: `p.name in _KEEP_FILES or p.name in ("_download_meta.json",)` を満たす経路を評価する。
            if p.name in _KEEP_FILES or p.name in ("_download_meta.json",):
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
            "本スクリプトは pre-recon の ξ0/ξ2 と共分散（48x48）に必要な最小ファイルのみを抽出する。",
        ],
    }
    _write_json(meta_path, meta)

    print(f"[ok] zip : {zip_cache}")
    print(f"[ok] dir : {out_dir}")
    print(f"[ok] meta: {meta_path}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_fetch_boss_dr12_satpathy2016_corrfunc_multipoles",
                "argv": sys.argv,
                "inputs": {"url": str(args.url)},
                "outputs": {"dir": out_dir, "zip": zip_cache, "meta_json": meta_path},
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
