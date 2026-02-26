#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_boss_dr12_beutler2016_bao_powspec.py

Phase 16（宇宙論）/ Step 16.4（BAO一次統計の再導出）:
BOSS DR12 の BAO power spectrum multipoles（Pℓ; ℓ=0,2）と共分散を取得する。

背景:
- BAO の圧縮出力（D_M/r_d, D_H/r_d）は、観測統計（角度×赤方偏移）→距離変換→テンプレートfit を含む推定量であり、
  前提が異なれば値が動き得る。
- P-model 側で距離変換を入れ替えて比較するため、一次統計により近い公開パッケージ（P(k) multipoles）をキャッシュする。

取得元（SAS; SDSS DR12 / BOSS / papers / clustering）:
  - Beutler_etal_DR12COMBINED_BAO_powspec.tar.gz

出力（固定）:
  - data/cosmology/beutler_2016_combineddr12_bao_powspec/（展開先; 必要最小ファイルのみ抽出）
  - data/cosmology/sources/Beutler_etal_DR12COMBINED_BAO_powspec.tar.gz（キャッシュ）
  - data/cosmology/beutler_2016_combineddr12_bao_powspec/_download_meta.json（取得メタ）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


_URL = "https://dr12.sdss3.org/sas/dr12/boss/papers/clustering/Beutler_etal_DR12COMBINED_BAO_powspec.tar.gz"

_KEEP_FILES = {
    # P(k) multipoles (NGC/SGC; z1..z3; pre/post recon)
    *{
        f"Beutleretal_pk_{ell}_DR12_{cap}_z{z}_{recon}_120.dat"
        for ell in ("monopole", "quadrupole")
        for cap in ("NGC", "SGC")
        for z in (1, 2, 3)
        for recon in ("postrecon", "prerecon")
    },
    # Window multipoles (RRℓ; used to convolve model with survey geometry)
    *{f"Beutleretal_window_z{z}_{cap}.dat" for cap in ("NGC", "SGC") for z in (1, 2, 3)},
    # Covariance (PATCHY mocks; pre/post recon)
    "Beutleretal_cov_patchy_z1_NGC_postrecon_1_30_1_30_1_1_996_60.dat",
    "Beutleretal_cov_patchy_z1_NGC_prerecon_1_30_1_30_1_1_2045_60.dat",
    "Beutleretal_cov_patchy_z1_SGC_postrecon_1_30_1_30_1_1_999_60.dat",
    "Beutleretal_cov_patchy_z1_SGC_prerecon_1_30_1_30_1_1_2048_60.dat",
    "Beutleretal_cov_patchy_z2_NGC_postrecon_1_30_1_30_1_1_996_60.dat",
    "Beutleretal_cov_patchy_z2_NGC_prerecon_1_30_1_30_1_1_2045_60.dat",
    "Beutleretal_cov_patchy_z2_SGC_postrecon_1_30_1_30_1_1_999_60.dat",
    "Beutleretal_cov_patchy_z2_SGC_prerecon_1_30_1_30_1_1_2048_60.dat",
    "Beutleretal_cov_patchy_z3_NGC_postrecon_1_30_1_30_1_1_996_60.dat",
    "Beutleretal_cov_patchy_z3_NGC_prerecon_1_30_1_30_1_1_2045_60.dat",
    "Beutleretal_cov_patchy_z3_SGC_postrecon_1_30_1_30_1_1_999_60.dat",
    "Beutleretal_cov_patchy_z3_SGC_prerecon_1_30_1_30_1_1_2048_60.dat",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def _download(url: str, out_path: Path, *, timeout_s: int = 120) -> Dict[str, Any]:
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


def _extract_tar(tar_gz: Path, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    skipped = 0
    kept = 0
    with tarfile.open(tar_gz, mode="r:gz") as tf:
        for m in tf.getmembers():
            name = str(m.name).replace("\\", "/")
            # 条件分岐: `not name or name.endswith("/")` を満たす経路を評価する。
            if not name or name.endswith("/"):
                skipped += 1
                continue

            base = name.split("/")[-1]
            # 条件分岐: `base.startswith("._") or base == ".DS_Store"` を満たす経路を評価する。
            if base.startswith("._") or base == ".DS_Store":
                skipped += 1
                continue

            # 条件分岐: `base not in _KEEP_FILES` を満たす経路を評価する。

            if base not in _KEEP_FILES:
                skipped += 1
                continue

            src = tf.extractfile(m)
            # 条件分岐: `src is None` を満たす経路を評価する。
            if src is None:
                skipped += 1
                continue

            dest = out_dir / base
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read())
            extracted += 1
            kept += 1

    return {"files_extracted": extracted, "files_kept": kept, "files_skipped": skipped}


def _looks_ready(out_dir: Path) -> bool:
    return (out_dir / "Beutleretal_pk_monopole_DR12_NGC_z1_postrecon_120.dat").exists() and (
        out_dir / "Beutleretal_cov_patchy_z1_NGC_postrecon_1_30_1_30_1_1_996_60.dat"
    ).exists() and (out_dir / "Beutleretal_window_z1_NGC.dat").exists()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", type=str, default=_URL)
    ap.add_argument("--force", action="store_true", help="Overwrite existing extracted files.")
    args = ap.parse_args(argv)

    data_dir = _ROOT / "data" / "cosmology"
    out_dir = data_dir / "beutler_2016_combineddr12_bao_powspec"
    tar_cache = data_dir / "sources" / "Beutler_etal_DR12COMBINED_BAO_powspec.tar.gz"
    meta_path = out_dir / "_download_meta.json"

    # 条件分岐: `out_dir.exists() and _looks_ready(out_dir) and not bool(args.force)` を満たす経路を評価する。
    if out_dir.exists() and _looks_ready(out_dir) and not bool(args.force):
        print(f"[skip] already present: {out_dir}")
        return 0

    # 条件分岐: `out_dir.exists() and bool(args.force)` を満たす経路を評価する。

    if out_dir.exists() and bool(args.force):
        for p in out_dir.glob("*"):
            # 条件分岐: `p.name in _KEEP_FILES or p.name == "_download_meta.json"` を満たす経路を評価する。
            if p.name in _KEEP_FILES or p.name == "_download_meta.json":
                try:
                    p.unlink()
                except Exception:
                    pass

    dl = _download(str(args.url), tar_cache)
    ex = _extract_tar(tar_cache, out_dir)

    meta: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "download": dl,
        "extract": ex,
        "outputs": {
            "tar_gz": str(tar_cache.relative_to(_ROOT)).replace("\\", "/"),
            "dir": str(out_dir.relative_to(_ROOT)).replace("\\", "/"),
        },
        "notes": [
            "公開tar.gzには paper_figures などの補助ファイルが含まれるため、本リポジトリでは一次統計（P0/P2 と共分散）に必要な最小ファイルのみ抽出する。",
            "本データは P(k) の multipoles、PATCHY mocks 由来の共分散（z1..z3, NGC/SGC, pre/post recon）、および窓関数（RR multipoles）を含む。",
        ],
    }
    _write_json(meta_path, meta)

    print(f"[ok] tar : {tar_cache}")
    print(f"[ok] dir : {out_dir}")
    print(f"[ok] meta: {meta_path}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_fetch_boss_dr12_beutler2016_bao_powspec",
                "argv": sys.argv,
                "inputs": {"url": str(args.url)},
                "outputs": {"dir": out_dir, "tar_gz": tar_cache, "meta_json": meta_path},
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
