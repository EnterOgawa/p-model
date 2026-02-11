#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_desi_dr1_bao_bao_data.py

Phase 4（宇宙論）/ Step 4.5B.21.4.4.1.2（DESI: 公開 cov へ寄せる）の補助:
DESI DR1 BAO の「公開 mean/cov（Gaussian BAO likelihood）」を
公式ドキュメントが参照する `CobayaSampler/bao_data` から取得し、
ローカルにキャッシュする。

背景:
- DESI の BAO（距離指標）結果は、`CobayaSampler/bao_data` の `desi_2024_gaussian_bao_*`
  に mean ベクトルと共分散が公開されている（DESI data portal のドキュメントからリンク）。
- ここでは **LRG1/LRG2 の z-bin** を含む DR1 の主要トレーサを固定ファイル名で取得し、
  `data/cosmology/desi_dr1_bao_bao_data.json` にまとめる。

入出力（固定）:
- 取得元（GitHub raw）:
  - https://raw.githubusercontent.com/CobayaSampler/bao_data/master/
- 生ファイル保存先:
  - data/cosmology/sources/bao_data/<filename>
- 解析用JSON:
  - data/cosmology/desi_dr1_bao_bao_data.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402


RAW_BASE = "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/"

# DR1 BAO（距離指標）: mean/cov（Gaussian）として公開されている主要ファイル（固定）
FILES = [
    # Combined (ALL)
    "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt",
    "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt",
    # BGS
    "desi_2024_gaussian_bao_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_mean.txt",
    "desi_2024_gaussian_bao_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_cov.txt",
    # LRG (LRG1/LRG2)
    "desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6_mean.txt",
    "desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6_cov.txt",
    "desi_2024_gaussian_bao_LRG_GCcomb_z0.6-0.8_mean.txt",
    "desi_2024_gaussian_bao_LRG_GCcomb_z0.6-0.8_cov.txt",
    # LRG+ELG (LRG3+ELG1)
    "desi_2024_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_mean.txt",
    "desi_2024_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_cov.txt",
    # ELG
    "desi_2024_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_mean.txt",
    "desi_2024_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_cov.txt",
    # QSO
    "desi_2024_gaussian_bao_QSO_GCcomb_z0.8-2.1_mean.txt",
    "desi_2024_gaussian_bao_QSO_GCcomb_z0.8-2.1_cov.txt",
    # Lyα (auto + cross combined)
    "desi_2024_gaussian_bao_Lya_GCcomb_mean.txt",
    "desi_2024_gaussian_bao_Lya_GCcomb_cov.txt",
]


@dataclass(frozen=True)
class BaoDataset:
    name: str
    mean_path: Path
    cov_path: Path
    z_eff: Optional[float]
    z_values: List[float]
    quantities: List[str]
    mean: List[float]
    cov: List[List[float]]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_text(url: str, *, timeout_sec: int) -> str:
    if requests is None:
        raise RuntimeError("requests is required (pip install requests)")
    r = requests.get(url, timeout=timeout_sec)
    r.raise_for_status()
    return r.text


def _parse_mean(text: str) -> Tuple[Optional[float], List[float], List[str], List[float]]:
    """
    mean.txt:
      # [z] [value at z] [quantity]
      0.510 13.62 DM_over_rs
      0.510 20.98 DH_over_rs
    """
    z_vals: List[float] = []
    qs: List[str] = []
    vals: List[float] = []
    for line in text.splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        parts = re.split(r"\s+", t)
        if len(parts) < 3:
            continue
        z = float(parts[0])
        v = float(parts[1])
        q = str(parts[2]).strip()
        z_vals.append(z)
        vals.append(v)
        qs.append(q)
    if not qs:
        raise ValueError("failed to parse mean file (no rows)")
    # Some files (e.g. desi_2024_gaussian_bao_ALL_*) include multiple z points.
    z0 = float(z_vals[0])
    z_eff: Optional[float] = z0
    for z in z_vals[1:]:
        if abs(float(z) - z0) > 1e-9:
            z_eff = None
            break
    return z_eff, z_vals, qs, vals


def _parse_cov(text: str) -> List[List[float]]:
    rows: List[List[float]] = []
    for line in text.splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        parts = re.split(r"\s+", t)
        rows.append([float(x) for x in parts if x])
    if not rows:
        raise ValueError("failed to parse cov file (no rows)")
    n = len(rows)
    if any(len(r) != n for r in rows):
        raise ValueError(f"cov is not square: {[len(r) for r in rows]}")
    return rows


def _load_dataset(name: str, mean_path: Path, cov_path: Path) -> BaoDataset:
    mean_text = mean_path.read_text(encoding="utf-8", errors="replace")
    cov_text = cov_path.read_text(encoding="utf-8", errors="replace")
    z_eff, z_vals, qs, vals = _parse_mean(mean_text)
    cov = _parse_cov(cov_text)
    if len(vals) != len(cov):
        raise ValueError(f"mean/cov size mismatch for {name}: len(mean)={len(vals)} len(cov)={len(cov)}")
    return BaoDataset(
        name=name,
        mean_path=mean_path,
        cov_path=cov_path,
        z_eff=(None if z_eff is None else float(z_eff)),
        z_values=[float(z) for z in z_vals],
        quantities=list(qs),
        mean=list(vals),
        cov=cov,
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch DESI DR1 BAO (Gaussian) mean/cov from CobayaSampler/bao_data.")
    ap.add_argument(
        "--raw-base",
        type=str,
        default=RAW_BASE,
        help="GitHub raw base URL (default: CobayaSampler/bao_data master)",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "data" / "cosmology" / "desi_dr1_bao_bao_data.json"),
        help="Output JSON path (default: data/cosmology/desi_dr1_bao_bao_data.json)",
    )
    ap.add_argument(
        "--out-raw-dir",
        type=str,
        default=str(ROOT / "data" / "cosmology" / "sources" / "bao_data"),
        help="Directory to store raw mean/cov txt files (default: data/cosmology/sources/bao_data/)",
    )
    ap.add_argument("--timeout-sec", type=int, default=30, help="HTTP timeout seconds (default: 30)")
    ap.add_argument("--no-network", action="store_true", help="Do not fetch; only parse existing cached files.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if requests is None and not bool(args.no_network):
        raise SystemExit("requests is required to fetch from GitHub (pip install requests)")

    raw_base = str(args.raw_base).strip()
    if not raw_base.endswith("/"):
        raw_base += "/"

    out_raw_dir = Path(str(args.out_raw_dir)).resolve()
    out_raw_dir.mkdir(parents=True, exist_ok=True)

    # Fetch (or reuse cached)
    fetched: Dict[str, str] = {}
    for fn in FILES:
        dst = out_raw_dir / fn
        if dst.exists():
            continue
        if bool(args.no_network):
            raise SystemExit(f"missing cached file (no-network): {dst}")
        url = raw_base + fn
        txt = _fetch_text(url, timeout_sec=int(args.timeout_sec))
        dst.write_text(txt, encoding="utf-8")
        fetched[fn] = url

    # Pair up mean/cov datasets
    datasets: List[BaoDataset] = []
    for fn in FILES:
        if not fn.endswith("_mean.txt"):
            continue
        base = fn[: -len("_mean.txt")]
        cov_fn = base + "_cov.txt"
        mean_path = out_raw_dir / fn
        cov_path = out_raw_dir / cov_fn
        if not mean_path.exists() or not cov_path.exists():
            raise SystemExit(f"missing mean/cov pair: {mean_path} / {cov_path}")
        datasets.append(_load_dataset(base, mean_path, cov_path))

    out_json = Path(str(args.out_json)).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "generated_utc": _now_utc(),
        "source": {
            "repo": "https://github.com/CobayaSampler/bao_data",
            "raw_base": raw_base,
            "note": "DESI portal doc (DR1 BAO cosmology results) points to this repo for BAO likelihood mean/cov.",
        },
        "cache": {
            "raw_dir": str(out_raw_dir),
            "downloaded": fetched,
        },
        "datasets": [
            {
                "name": d.name,
                "z_eff": d.z_eff,
                "z_values": d.z_values,
                "quantities": d.quantities,
                "mean": d.mean,
                "cov": d.cov,
                "paths": {"mean": str(d.mean_path), "cov": str(d.cov_path)},
            }
            for d in sorted(datasets, key=lambda x: x.name)
        ],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_json}")
    print(f"[ok] datasets={len(payload['datasets'])} raw_dir={out_raw_dir}")

    try:
        worklog.append_event(
            {
                "domain": "cosmology",
                "action": "fetch_desi_dr1_bao_bao_data",
                "argv": sys.argv,
                "inputs": [],
                "outputs": [str(out_json)],
                "params": {"raw_base": raw_base, "files": FILES},
            }
        )
    except Exception:
        pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
