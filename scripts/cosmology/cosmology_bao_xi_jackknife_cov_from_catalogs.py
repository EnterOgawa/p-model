#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_xi_jackknife_cov_from_catalogs.py

Phase 4（宇宙論）/ Step 4.5B.21.4.4.4（DESI cov 代替策）:
`cosmology_bao_xi_from_catalogs.py` で計算した「銀河+random→ξℓ」の case に対して、
sky jackknife（leave-one-out）で dv=[xi0, xi2] の共分散を推定し、peakfit を dv+cov に更新する入口を作る。

前提：
- Corrfunc を使用するため **WSL（Linux）で実行**する。
- 24スレッド運用が原則（`--threads 24` / `OMP_NUM_THREADS=24`）。

入力：
- `output/private/cosmology/cosmology_bao_xi_from_catalogs_*_metrics.json`（1ケース分を指定）
  - この metrics には、入力NPZ（galaxy/random）のパスと、z範囲・重み等の仕様が含まれる。

出力（固定名）：
- `output/private/cosmology/cosmology_bao_xi_from_catalogs_{tag}__jk_cov.npz`（既定）
  - `s`（xi の s bin center; xi出力と一致）
  - `cov`（shape=(2*n_s, 2*n_s); dv順序は y=[xi0, xi2]）
  - `y_jk`（shape=(n_jk, 2*n_s); leave-one-out の推定値）
  - `ra_edges_deg`（jackknife領域境界; 分位点; ra_quantile のみ）
- `output/private/cosmology/cosmology_bao_xi_from_catalogs_{tag}__jk_cov_metrics.json`（既定）

備考：
- 現状の region 分割は RA 分位点（決定的・seed不要）。
  （将来：healpix/tiles などの分割に拡張可能）
  - `--jk-mode ra_quantile_per_cap` は cap（north/south 等）ごとに分位点を作り、
    region id を cap ごとにオフセットして「cap内の連結領域」を jackknife 単位とする（n_regions は合計）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cosmology import cosmology_bao_xi_from_catalogs as _xi  # noqa: E402
from scripts.summary import worklog  # noqa: E402


# 関数: `_infer_cap_label` の入出力契約と処理意図を定義する。
def _infer_cap_label(path: Path, *, fallback: str) -> str:
    s = path.name.lower()
    # 条件分岐: `"ngc" in s or "north" in s` を満たす経路を評価する。
    if "ngc" in s or "north" in s:
        return "north"

    # 条件分岐: `"sgc" in s or "south" in s` を満たす経路を評価する。

    if "sgc" in s or "south" in s:
        return "south"

    return fallback


# 関数: `_ra_quantile_edges` の入出力契約と処理意図を定義する。

def _ra_quantile_edges(ra_deg_all: np.ndarray, n_regions: int) -> np.ndarray:
    ra = np.asarray(ra_deg_all, dtype=float).reshape(-1)
    ra = ra[np.isfinite(ra)]
    # 条件分岐: `ra.size < max(10, n_regions * 2)` を満たす経路を評価する。
    if ra.size < max(10, n_regions * 2):
        raise ValueError(f"too few RA samples for jackknife: n={ra.size}")

    ra = np.mod(ra, 360.0)
    ra_sorted = np.sort(ra)
    edges = np.empty(int(n_regions) + 1, dtype=float)
    edges[0] = 0.0
    for i in range(1, int(n_regions)):
        idx = int(math.floor(float(i) * float(ra_sorted.size) / float(n_regions)))
        idx = min(max(idx, 0), int(ra_sorted.size - 1))
        edges[i] = float(ra_sorted[idx])

    edges[int(n_regions)] = 360.0
    # Ensure strict monotonicity to avoid empty bins due to repeated quantiles.
    for i in range(1, edges.size):
        # 条件分岐: `not (edges[i] > edges[i - 1])` を満たす経路を評価する。
        if not (edges[i] > edges[i - 1]):
            edges[i] = min(360.0, float(edges[i - 1]) + 1e-6)

    return edges


# 関数: `_assign_region_ra` の入出力契約と処理意図を定義する。

def _assign_region_ra(ra_deg: np.ndarray, edges: np.ndarray) -> np.ndarray:
    ra = np.mod(np.asarray(ra_deg, dtype=float), 360.0)
    # right edge belongs to previous bin except the last edge (360).
    idx = np.searchsorted(edges, ra, side="right") - 1
    idx = np.clip(idx, 0, int(edges.size - 2))
    return idx.astype(np.int32, copy=False)


# 関数: `_quantile_edges` の入出力契約と処理意図を定義する。

def _quantile_edges(x: np.ndarray, n_regions: int) -> np.ndarray:
    v = np.asarray(x, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    # 条件分岐: `v.size < max(10, n_regions * 2)` を満たす経路を評価する。
    if v.size < max(10, n_regions * 2):
        raise ValueError(f"too few samples for quantiles: n={v.size}")

    v_sorted = np.sort(v)
    edges = np.empty(int(n_regions) + 1, dtype=float)
    edges[0] = float(v_sorted[0])
    for i in range(1, int(n_regions)):
        idx = int(math.floor(float(i) * float(v_sorted.size) / float(n_regions)))
        idx = min(max(idx, 0), int(v_sorted.size - 1))
        edges[i] = float(v_sorted[idx])

    edges[int(n_regions)] = float(v_sorted[-1])
    for i in range(1, edges.size):
        # 条件分岐: `not (edges[i] > edges[i - 1])` を満たす経路を評価する。
        if not (edges[i] > edges[i - 1]):
            edges[i] = float(edges[i - 1]) + 1e-9

    return edges


# 関数: `_assign_region_1d` の入出力契約と処理意図を定義する。

def _assign_region_1d(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=float).reshape(-1)
    idx = np.searchsorted(edges, v, side="right") - 1
    idx = np.clip(idx, 0, int(edges.size - 2))
    return idx.astype(np.int32, copy=False)


# 関数: `_choose_grid` の入出力契約と処理意図を定義する。

def _choose_grid(n_regions: int) -> Tuple[int, int]:
    """
    Choose (n_dec, n_ra) such that n_dec*n_ra = n_regions, preferring n_dec near sqrt(n_regions).
    """
    n = int(n_regions)
    # 条件分岐: `n <= 0` を満たす経路を評価する。
    if n <= 0:
        raise ValueError("n_regions must be positive")

    root = math.sqrt(float(n))
    best = 1
    for d in range(1, n + 1):
        # 条件分岐: `(n % d) != 0` を満たす経路を評価する。
        if (n % d) != 0:
            continue

        # 条件分岐: `float(d) <= root` を満たす経路を評価する。

        if float(d) <= root:
            best = d

    return int(best), int(n // best)


# 関数: `_ra_shift_to_largest_gap` の入出力契約と処理意図を定義する。

def _ra_shift_to_largest_gap(ra_deg_all: np.ndarray) -> float:
    ra = np.asarray(ra_deg_all, dtype=float).reshape(-1)
    ra = ra[np.isfinite(ra)]
    # 条件分岐: `ra.size < 2` を満たす経路を評価する。
    if ra.size < 2:
        return 0.0

    ra = np.mod(ra, 360.0)
    ra_sorted = np.sort(ra)
    gaps = np.diff(ra_sorted)
    wrap_gap = float(ra_sorted[0] + 360.0 - ra_sorted[-1])
    gaps_all = np.concatenate([gaps, np.asarray([wrap_gap], dtype=float)], axis=0)
    k = int(np.argmax(gaps_all))
    # 条件分岐: `k >= int(ra_sorted.size - 1)` を満たす経路を評価する。
    if k >= int(ra_sorted.size - 1):
        # Wrap gap is the largest: pick the minimum RA as the cut.
        return float(ra_sorted[0])
    # Cut at the start of the largest internal gap.

    return float(ra_sorted[k + 1])


# 関数: `_ra_quantile_edges_unwrapped` の入出力契約と処理意図を定義する。

def _ra_quantile_edges_unwrapped(ra_deg_all: np.ndarray, n_regions: int) -> Tuple[np.ndarray, float]:
    shift = _ra_shift_to_largest_gap(ra_deg_all)
    ra_shifted = np.mod(np.asarray(ra_deg_all, dtype=float) - shift, 360.0)
    edges = _ra_quantile_edges(ra_shifted, n_regions=n_regions)
    return edges, float(shift)


# 関数: `_weighted_totals` の入出力契約と処理意図を定義する。

def _weighted_totals(w_g: np.ndarray, w_r: np.ndarray) -> Dict[str, float]:
    w_g = np.asarray(w_g, dtype=float).reshape(-1)
    w_r = np.asarray(w_r, dtype=float).reshape(-1)
    sum_wg = float(np.sum(w_g))
    sum_wr = float(np.sum(w_r))
    sum_wg2 = float(np.sum(w_g * w_g))
    sum_wr2 = float(np.sum(w_r * w_r))
    # Corrfunc autocorr counts ordered pairs (i != j), so totals use the same convention.
    dd_tot = sum_wg * sum_wg - sum_wg2
    rr_tot = sum_wr * sum_wr - sum_wr2
    dr_tot = sum_wg * sum_wr
    # 条件分岐: `not (dd_tot > 0.0 and rr_tot > 0.0 and dr_tot > 0.0)` を満たす経路を評価する。
    if not (dd_tot > 0.0 and rr_tot > 0.0 and dr_tot > 0.0):
        raise ValueError("invalid total weights (non-positive)")

    return {
        "sum_w_gal": sum_wg,
        "sum_w2_gal": sum_wg2,
        "sum_w_rnd": sum_wr,
        "sum_w2_rnd": sum_wr2,
        "dd_tot": dd_tot,
        "dr_tot": dr_tot,
        "rr_tot": rr_tot,
    }


# 関数: `_paircounts_for_subset` の入出力契約と処理意図を定義する。

def _paircounts_for_subset(
    *,
    ra_g: np.ndarray,
    dec_g: np.ndarray,
    d_g: np.ndarray,
    w_g: np.ndarray,
    ra_r: np.ndarray,
    dec_r: np.ndarray,
    d_r: np.ndarray,
    w_r: np.ndarray,
    s_bins_file: Path,
    mu_max: float,
    nmu: int,
    nthreads: int,
) -> Dict[str, np.ndarray]:
    dd_w = _xi._corrfunc_paircounts_smu(
        ra1=ra_g,
        dec1=dec_g,
        dist1=d_g,
        w1=w_g,
        ra2=None,
        dec2=None,
        dist2=None,
        w2=None,
        s_bins_file=s_bins_file,
        mu_max=mu_max,
        nmu=nmu,
        nthreads=nthreads,
        autocorr=1,
    )
    dr_w = _xi._corrfunc_paircounts_smu(
        ra1=ra_g,
        dec1=dec_g,
        dist1=d_g,
        w1=w_g,
        ra2=ra_r,
        dec2=dec_r,
        dist2=d_r,
        w2=w_r,
        s_bins_file=s_bins_file,
        mu_max=mu_max,
        nmu=nmu,
        nthreads=nthreads,
        autocorr=0,
    )
    rr_w = _xi._corrfunc_paircounts_smu(
        ra1=ra_r,
        dec1=dec_r,
        dist1=d_r,
        w1=w_r,
        ra2=None,
        dec2=None,
        dist2=None,
        w2=None,
        s_bins_file=s_bins_file,
        mu_max=mu_max,
        nmu=nmu,
        nthreads=nthreads,
        autocorr=1,
    )
    return {"DD_w": dd_w, "DR_w": dr_w, "RR_w": rr_w}


# 関数: `_load_cap_inputs` の入出力契約と処理意図を定義する。

def _load_cap_inputs(
    *,
    gal_npz: Path,
    rnd_npz: Path,
    z_source: str,
    weight_scheme: str,
    z_min: float | None,
    z_max: float | None,
    dist_model: str,
    omega_m: float,
    lcdm_n_grid: int,
    lcdm_z_grid_max: float | None,
) -> Dict[str, Any]:
    gal = _xi._load_npz(gal_npz)
    rnd = _xi._load_npz(rnd_npz)

    ra_g0 = np.asarray(gal["RA"], dtype=np.float64)
    dec_g0 = np.asarray(gal["DEC"], dtype=np.float64)
    z_g0, _ = _xi._select_redshift(gal, z_source=str(z_source))
    w_g0, _ = _xi._weights_galaxy(gal, scheme=str(weight_scheme))

    ra_r0 = np.asarray(rnd["RA"], dtype=np.float64)
    dec_r0 = np.asarray(rnd["DEC"], dtype=np.float64)
    z_r0, _ = _xi._select_redshift(rnd, z_source=str(z_source))
    w_r0, _ = _xi._weights_random(rnd, scheme=str(weight_scheme))

    mg = np.isfinite(z_g0) & (z_g0 > 0.0) & np.isfinite(w_g0)
    mr = np.isfinite(z_r0) & (z_r0 > 0.0) & np.isfinite(w_r0)
    # 条件分岐: `z_min is not None` を満たす経路を評価する。
    if z_min is not None:
        mg = mg & (z_g0 >= float(z_min))
        mr = mr & (z_r0 >= float(z_min))

    # 条件分岐: `z_max is not None` を満たす経路を評価する。

    if z_max is not None:
        mg = mg & (z_g0 < float(z_max))
        mr = mr & (z_r0 < float(z_max))

    ra_g = ra_g0[mg]
    dec_g = dec_g0[mg]
    z_g = np.asarray(z_g0[mg], dtype=np.float64)
    w_g = np.asarray(w_g0[mg], dtype=np.float64)

    ra_r = ra_r0[mr]
    dec_r = dec_r0[mr]
    z_r = np.asarray(z_r0[mr], dtype=np.float64)
    w_r = np.asarray(w_r0[mr], dtype=np.float64)

    z_all = np.concatenate([z_g, z_r], axis=0).astype(np.float64, copy=False)
    d_all, dist_meta = _xi._comoving_distance_mpc_over_h(
        z_all,
        model=str(dist_model),
        lcdm_omega_m=float(omega_m),
        lcdm_n_grid=int(lcdm_n_grid),
        lcdm_z_grid_max=lcdm_z_grid_max,
    )
    d_all = np.asarray(d_all, dtype=np.float64)
    d_g = d_all[: int(z_g.size)]
    d_r = d_all[int(z_g.size) :]

    return {
        "gal": {"ra": ra_g, "dec": dec_g, "z": z_g, "d": d_g, "w": w_g},
        "rnd": {"ra": ra_r, "dec": dec_r, "z": z_r, "d": d_r, "w": w_r},
        "dist_meta": dist_meta,
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cosmology: jackknife covariance for catalog-based xi multipoles (xi0/xi2).")
    ap.add_argument(
        "--xi-metrics-json",
        required=True,
        help="Path to a single cosmology_bao_xi_from_catalogs_*_metrics.json (the case definition).",
    )
    ap.add_argument("--jk-n", type=int, default=8, help="Number of jackknife regions (default: 8)")
    ap.add_argument(
        "--jk-mode",
        choices=[
            "ra_quantile",
            "ra_quantile_unwrapped",
            "ra_quantile_per_cap",
            "ra_quantile_per_cap_unwrapped",
            "ra_dec_quantile_per_cap_unwrapped",
        ],
        default="ra_quantile",
        help="Jackknife region assignment (default: ra_quantile).",
    )
    ap.add_argument("--jk-dec-bands", type=int, default=0, help="For ra_dec_quantile*: number of Dec bands (0=auto)")
    ap.add_argument("--jk-ra-bins", type=int, default=0, help="For ra_dec_quantile*: number of RA bins per Dec band (0=auto)")
    ap.add_argument(
        "--output-suffix",
        type=str,
        default="jk_cov",
        help=(
            "Output suffix for covariance files (default: jk_cov => "
            "...__jk_cov.npz and ...__jk_cov_metrics.json)."
        ),
    )
    ap.add_argument("--threads", type=int, default=24, help="Corrfunc threads (default: 24)")
    ap.add_argument("--rcond", type=float, default=1e-12, help="rcond for pseudo-inverse (metrics only; default: 1e-12)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    # 条件分岐: `os.name == "nt"` を満たす経路を評価する。
    if os.name == "nt":
        raise SystemExit("Corrfunc is not supported on Windows; run this under WSL (Ubuntu-24.04) as per AGENTS.md.")

    metrics_path = _xi._resolve_manifest_path(str(args.xi_metrics_json))
    # 条件分岐: `not metrics_path.exists()` を満たす経路を評価する。
    if not metrics_path.exists():
        raise SystemExit(f"missing metrics json: {metrics_path}")

    base = json.loads(metrics_path.read_text(encoding="utf-8"))

    params = base.get("params", {}) or {}
    inputs = base.get("inputs", {}) or {}
    outputs = base.get("outputs", {}) or {}
    # 条件分岐: `"npz" not in outputs` を満たす経路を評価する。
    if "npz" not in outputs:
        raise SystemExit("metrics json missing outputs.npz")

    xi_npz_path = _xi._resolve_manifest_path(str(outputs["npz"]))
    # 条件分岐: `not xi_npz_path.exists()` を満たす経路を評価する。
    if not xi_npz_path.exists():
        raise SystemExit(f"missing xi npz (run xi-from-catalogs first): {xi_npz_path}")

    stem = xi_npz_path.stem
    prefix = "cosmology_bao_xi_from_catalogs_"
    # 条件分岐: `not stem.startswith(prefix)` を満たす経路を評価する。
    if not stem.startswith(prefix):
        raise SystemExit(f"unexpected xi npz name: {xi_npz_path.name}")

    tag = stem[len(prefix) :]

    out_suffix = str(args.output_suffix).strip()
    # 条件分岐: `not out_suffix` を満たす経路を評価する。
    if not out_suffix:
        raise SystemExit("--output-suffix must be non-empty")

    out_cov_npz = xi_npz_path.with_name(f"{prefix}{tag}__{out_suffix}.npz")
    out_cov_json = xi_npz_path.with_name(f"{prefix}{tag}__{out_suffix}_metrics.json")

    gal_list = inputs.get("galaxy_npz", [])
    rnd_list = inputs.get("random_npz", [])
    # 条件分岐: `not (isinstance(gal_list, list) and isinstance(rnd_list, list) and gal_list a...` を満たす経路を評価する。
    if not (isinstance(gal_list, list) and isinstance(rnd_list, list) and gal_list and rnd_list):
        raise SystemExit("metrics json missing inputs.galaxy_npz/random_npz lists")

    # 条件分岐: `len(gal_list) != len(rnd_list)` を満たす経路を評価する。

    if len(gal_list) != len(rnd_list):
        raise SystemExit("inputs.galaxy_npz/random_npz length mismatch")

    z_cut = params.get("z_cut", {}) or {}
    z_min = z_cut.get("z_min", None)
    z_max = z_cut.get("z_max", None)
    # 条件分岐: `z_min is not None` を満たす経路を評価する。
    if z_min is not None:
        z_min = float(z_min)

    # 条件分岐: `z_max is not None` を満たす経路を評価する。

    if z_max is not None:
        z_max = float(z_max)

    s_bins = params.get("s_bins", {}) or {}
    s_min = float(s_bins.get("min", 30.0))
    s_max = float(s_bins.get("max", 150.0))
    s_step = float(s_bins.get("step", 5.0))
    # 条件分岐: `not (s_step > 0 and math.isfinite(s_step))` を満たす経路を評価する。
    if not (s_step > 0 and math.isfinite(s_step)):
        raise SystemExit("invalid s_step in metrics params")

    mu_bins = params.get("mu_bins", {}) or {}
    mu_max = float(mu_bins.get("mu_max", 1.0))
    nmu = int(mu_bins.get("nmu", 120))
    # 条件分岐: `not (nmu > 0)` を満たす経路を評価する。
    if not (nmu > 0):
        raise SystemExit("invalid nmu in metrics params")

    omega_m = float(params.get("lcdm_omega_m", 0.315))
    lcdm_n_grid = int(params.get("lcdm_n_grid", 6000))
    lcdm_z_grid_max = float(params.get("lcdm_z_grid_max", 3.0))
    dist_model = str(params.get("distance_model", "lcdm"))
    z_source = str(params.get("z_source", "obs"))
    weight_scheme = str(params.get("weight_scheme", "boss_default"))

    out_dir = out_cov_npz.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    s_bins_file, edges = _xi._make_s_bins_file(out_dir, s_min=s_min, s_max=s_max, s_step=s_step)

    n_jk = int(args.jk_n)
    # 条件分岐: `n_jk < 3` を満たす経路を評価する。
    if n_jk < 3:
        raise SystemExit("--jk-n must be >= 3")

    # Load per-cap inputs (already z-filtered; distances computed).

    cap_inputs: List[Dict[str, Any]] = []
    for i, (gp, rp) in enumerate(zip(gal_list, rnd_list)):
        gpath = _xi._resolve_manifest_path(str(gp))
        rpath = _xi._resolve_manifest_path(str(rp))
        # 条件分岐: `not gpath.exists()` を満たす経路を評価する。
        if not gpath.exists():
            raise SystemExit(f"missing galaxy npz: {gpath}")

        # 条件分岐: `not rpath.exists()` を満たす経路を評価する。

        if not rpath.exists():
            raise SystemExit(f"missing random npz: {rpath}")

        cap = _infer_cap_label(gpath, fallback=f"cap{i}")
        pack = _load_cap_inputs(
            gal_npz=gpath,
            rnd_npz=rpath,
            z_source=z_source,
            weight_scheme=weight_scheme,
            z_min=z_min,
            z_max=z_max,
            dist_model=dist_model,
            omega_m=omega_m,
            lcdm_n_grid=lcdm_n_grid,
            lcdm_z_grid_max=lcdm_z_grid_max,
        )
        pack["cap"] = cap
        cap_inputs.append(pack)

    # Define deterministic jackknife regions based on the (z-cut) galaxy RA distribution.
    # Note: per-cap mode offsets region ids so that each jackknife region is contiguous within a cap.

    jk_mode = str(args.jk_mode)
    ra_edges: np.ndarray = np.zeros(0, dtype=float)
    ra_shift_deg: float | None = None
    ra_edges_by_cap: Dict[str, np.ndarray] = {}
    ra_shift_by_cap: Dict[str, float] = {}
    dec_edges_by_cap: Dict[str, np.ndarray] = {}
    ra_edges_by_cap_by_dec: Dict[str, List[np.ndarray]] = {}
    ra_shift_by_cap_by_dec: Dict[str, List[float]] = {}
    grid_meta: Dict[str, int] | None = None
    # 条件分岐: `jk_mode == "ra_quantile"` を満たす経路を評価する。
    if jk_mode == "ra_quantile":
        ra_all = np.concatenate([np.asarray(p["gal"]["ra"], dtype=float).reshape(-1) for p in cap_inputs], axis=0)
        ra_edges = _ra_quantile_edges(ra_all, n_regions=n_jk)
        for p in cap_inputs:
            p["gal"]["jk_id"] = _assign_region_ra(p["gal"]["ra"], ra_edges)
            p["rnd"]["jk_id"] = _assign_region_ra(p["rnd"]["ra"], ra_edges)
    # 条件分岐: 前段条件が不成立で、`jk_mode == "ra_quantile_unwrapped"` を追加評価する。
    elif jk_mode == "ra_quantile_unwrapped":
        ra_all = np.concatenate([np.asarray(p["gal"]["ra"], dtype=float).reshape(-1) for p in cap_inputs], axis=0)
        ra_edges, ra_shift_deg = _ra_quantile_edges_unwrapped(ra_all, n_regions=n_jk)
        for p in cap_inputs:
            p["gal"]["jk_id"] = _assign_region_ra(np.mod(np.asarray(p["gal"]["ra"], dtype=float) - float(ra_shift_deg), 360.0), ra_edges)
            p["rnd"]["jk_id"] = _assign_region_ra(np.mod(np.asarray(p["rnd"]["ra"], dtype=float) - float(ra_shift_deg), 360.0), ra_edges)
    # 条件分岐: 前段条件が不成立で、`jk_mode == "ra_quantile_per_cap"` を追加評価する。
    elif jk_mode == "ra_quantile_per_cap":
        caps_sorted = sorted({str(p.get("cap", "")) for p in cap_inputs})
        # 条件分岐: `not caps_sorted or any(not c for c in caps_sorted)` を満たす経路を評価する。
        if not caps_sorted or any(not c for c in caps_sorted):
            raise SystemExit("invalid cap labels for --jk-mode ra_quantile_per_cap")

        n_caps = int(len(caps_sorted))
        # 条件分岐: `n_caps == 1` を満たす経路を評価する。
        if n_caps == 1:
            ra_all = np.concatenate([np.asarray(p["gal"]["ra"], dtype=float).reshape(-1) for p in cap_inputs], axis=0)
            ra_edges = _ra_quantile_edges(ra_all, n_regions=n_jk)
            for p in cap_inputs:
                p["gal"]["jk_id"] = _assign_region_ra(p["gal"]["ra"], ra_edges)
                p["rnd"]["jk_id"] = _assign_region_ra(p["rnd"]["ra"], ra_edges)
        else:
            # 条件分岐: `(n_jk % n_caps) != 0` を満たす経路を評価する。
            if (n_jk % n_caps) != 0:
                raise SystemExit(
                    f"--jk-n must be divisible by n_caps for ra_quantile_per_cap "
                    f"(jk_n={n_jk}, n_caps={n_caps}, caps={caps_sorted})"
                )

            n_per_cap = int(n_jk // n_caps)
            # 条件分岐: `n_per_cap < 3` を満たす経路を評価する。
            if n_per_cap < 3:
                raise SystemExit(
                    f"--jk-n too small for ra_quantile_per_cap "
                    f"(need >= {3*n_caps}; got jk_n={n_jk}, n_caps={n_caps})"
                )

            cap_to_idx = {cap: i for i, cap in enumerate(caps_sorted)}
            for cap in caps_sorted:
                ra_cap = np.concatenate(
                    [np.asarray(p["gal"]["ra"], dtype=float).reshape(-1) for p in cap_inputs if str(p.get("cap")) == cap],
                    axis=0,
                )
                ra_edges_by_cap[cap] = _ra_quantile_edges(ra_cap, n_regions=n_per_cap)

            for p in cap_inputs:
                cap = str(p.get("cap"))
                offset = int(cap_to_idx[cap]) * int(n_per_cap)
                edges_cap = ra_edges_by_cap[cap]
                p["gal"]["jk_id"] = offset + _assign_region_ra(p["gal"]["ra"], edges_cap)
                p["rnd"]["jk_id"] = offset + _assign_region_ra(p["rnd"]["ra"], edges_cap)
    # 条件分岐: 前段条件が不成立で、`jk_mode == "ra_quantile_per_cap_unwrapped"` を追加評価する。
    elif jk_mode == "ra_quantile_per_cap_unwrapped":
        caps_sorted = sorted({str(p.get("cap", "")) for p in cap_inputs})
        # 条件分岐: `not caps_sorted or any(not c for c in caps_sorted)` を満たす経路を評価する。
        if not caps_sorted or any(not c for c in caps_sorted):
            raise SystemExit("invalid cap labels for --jk-mode ra_quantile_per_cap_unwrapped")

        n_caps = int(len(caps_sorted))
        # 条件分岐: `n_caps == 1` を満たす経路を評価する。
        if n_caps == 1:
            ra_all = np.concatenate([np.asarray(p["gal"]["ra"], dtype=float).reshape(-1) for p in cap_inputs], axis=0)
            ra_edges, ra_shift_deg = _ra_quantile_edges_unwrapped(ra_all, n_regions=n_jk)
            for p in cap_inputs:
                p["gal"]["jk_id"] = _assign_region_ra(
                    np.mod(np.asarray(p["gal"]["ra"], dtype=float) - float(ra_shift_deg), 360.0), ra_edges
                )
                p["rnd"]["jk_id"] = _assign_region_ra(
                    np.mod(np.asarray(p["rnd"]["ra"], dtype=float) - float(ra_shift_deg), 360.0), ra_edges
                )
        else:
            # 条件分岐: `(n_jk % n_caps) != 0` を満たす経路を評価する。
            if (n_jk % n_caps) != 0:
                raise SystemExit(
                    f"--jk-n must be divisible by n_caps for ra_quantile_per_cap_unwrapped "
                    f"(jk_n={n_jk}, n_caps={n_caps}, caps={caps_sorted})"
                )

            n_per_cap = int(n_jk // n_caps)
            # 条件分岐: `n_per_cap < 3` を満たす経路を評価する。
            if n_per_cap < 3:
                raise SystemExit(
                    f"--jk-n too small for ra_quantile_per_cap_unwrapped "
                    f"(need >= {3*n_caps}; got jk_n={n_jk}, n_caps={n_caps})"
                )

            cap_to_idx = {cap: i for i, cap in enumerate(caps_sorted)}
            for cap in caps_sorted:
                ra_cap = np.concatenate(
                    [np.asarray(p["gal"]["ra"], dtype=float).reshape(-1) for p in cap_inputs if str(p.get("cap")) == cap],
                    axis=0,
                )
                edges_cap, shift_cap = _ra_quantile_edges_unwrapped(ra_cap, n_regions=n_per_cap)
                ra_edges_by_cap[cap] = edges_cap
                ra_shift_by_cap[cap] = float(shift_cap)

            for p in cap_inputs:
                cap = str(p.get("cap"))
                offset = int(cap_to_idx[cap]) * int(n_per_cap)
                edges_cap = ra_edges_by_cap[cap]
                shift_cap = float(ra_shift_by_cap[cap])
                p["gal"]["jk_id"] = offset + _assign_region_ra(
                    np.mod(np.asarray(p["gal"]["ra"], dtype=float) - shift_cap, 360.0), edges_cap
                )
                p["rnd"]["jk_id"] = offset + _assign_region_ra(
                    np.mod(np.asarray(p["rnd"]["ra"], dtype=float) - shift_cap, 360.0), edges_cap
                )
    # 条件分岐: 前段条件が不成立で、`jk_mode == "ra_dec_quantile_per_cap_unwrapped"` を追加評価する。
    elif jk_mode == "ra_dec_quantile_per_cap_unwrapped":
        caps_sorted = sorted({str(p.get("cap", "")) for p in cap_inputs})
        # 条件分岐: `not caps_sorted or any(not c for c in caps_sorted)` を満たす経路を評価する。
        if not caps_sorted or any(not c for c in caps_sorted):
            raise SystemExit("invalid cap labels for --jk-mode ra_dec_quantile_per_cap_unwrapped")

        n_caps = int(len(caps_sorted))
        # 条件分岐: `n_caps < 1` を満たす経路を評価する。
        if n_caps < 1:
            raise SystemExit("no caps found")

        # 条件分岐: `(n_jk % n_caps) != 0` を満たす経路を評価する。

        if (n_jk % n_caps) != 0:
            raise SystemExit(
                f"--jk-n must be divisible by n_caps for ra_dec_quantile_per_cap_unwrapped "
                f"(jk_n={n_jk}, n_caps={n_caps}, caps={caps_sorted})"
            )

        n_per_cap = int(n_jk // n_caps)
        # 条件分岐: `n_per_cap < 3` を満たす経路を評価する。
        if n_per_cap < 3:
            raise SystemExit(f"--jk-n too small per cap (jk_n={n_jk}, n_caps={n_caps})")

        n_dec = int(args.jk_dec_bands)
        n_ra = int(args.jk_ra_bins)
        # 条件分岐: `n_dec <= 0 and n_ra <= 0` を満たす経路を評価する。
        if n_dec <= 0 and n_ra <= 0:
            n_dec, n_ra = _choose_grid(n_per_cap)
        # 条件分岐: 前段条件が不成立で、`n_dec <= 0 and n_ra > 0` を追加評価する。
        elif n_dec <= 0 and n_ra > 0:
            # 条件分岐: `(n_per_cap % n_ra) != 0` を満たす経路を評価する。
            if (n_per_cap % n_ra) != 0:
                raise SystemExit(f"--jk-ra-bins must divide n_per_cap (n_per_cap={n_per_cap}, jk_ra_bins={n_ra})")

            n_dec = int(n_per_cap // n_ra)
        # 条件分岐: 前段条件が不成立で、`n_dec > 0 and n_ra <= 0` を追加評価する。
        elif n_dec > 0 and n_ra <= 0:
            # 条件分岐: `(n_per_cap % n_dec) != 0` を満たす経路を評価する。
            if (n_per_cap % n_dec) != 0:
                raise SystemExit(f"--jk-dec-bands must divide n_per_cap (n_per_cap={n_per_cap}, jk_dec_bands={n_dec})")

            n_ra = int(n_per_cap // n_dec)
        else:
            # 条件分岐: `int(n_dec * n_ra) != int(n_per_cap)` を満たす経路を評価する。
            if int(n_dec * n_ra) != int(n_per_cap):
                raise SystemExit(
                    f"--jk-dec-bands * --jk-ra-bins must equal n_per_cap "
                    f"(n_per_cap={n_per_cap}, jk_dec_bands={n_dec}, jk_ra_bins={n_ra})"
                )

        # 条件分岐: `n_dec < 1 or n_ra < 1` を満たす経路を評価する。

        if n_dec < 1 or n_ra < 1:
            raise SystemExit(f"invalid grid (n_dec={n_dec}, n_ra={n_ra})")

        grid_meta = {"n_dec": int(n_dec), "n_ra": int(n_ra), "n_per_cap": int(n_per_cap), "n_caps": int(n_caps)}

        cap_to_idx = {cap: i for i, cap in enumerate(caps_sorted)}
        for cap in caps_sorted:
            # Build dec edges per cap from galaxy distribution.
            dec_cap = np.concatenate(
                [np.asarray(p["gal"]["dec"], dtype=float).reshape(-1) for p in cap_inputs if str(p.get("cap")) == cap], axis=0
            )
            dec_edges = _quantile_edges(dec_cap, n_regions=int(n_dec))
            dec_edges_by_cap[cap] = dec_edges

        for p in cap_inputs:
            cap = str(p.get("cap"))
            offset = int(cap_to_idx[cap]) * int(n_per_cap)
            dec_edges = dec_edges_by_cap[cap]
            gal_dec = np.asarray(p["gal"]["dec"], dtype=float)
            rnd_dec = np.asarray(p["rnd"]["dec"], dtype=float)
            gal_dec_id = _assign_region_1d(gal_dec, dec_edges)
            rnd_dec_id = _assign_region_1d(rnd_dec, dec_edges)

            gal_ra = np.asarray(p["gal"]["ra"], dtype=float)
            rnd_ra = np.asarray(p["rnd"]["ra"], dtype=float)

            jk_g = np.empty(int(gal_ra.size), dtype=np.int32)
            jk_r = np.empty(int(rnd_ra.size), dtype=np.int32)

            ra_edges_list: List[np.ndarray] = []
            ra_shift_list: List[float] = []
            for j in range(int(n_dec)):
                mg = gal_dec_id == int(j)
                # 条件分岐: `not np.any(mg)` を満たす経路を評価する。
                if not np.any(mg):
                    raise SystemExit(f"empty dec band in galaxies (cap={cap}, band={j}/{n_dec})")

                edges_ra, shift_ra = _ra_quantile_edges_unwrapped(gal_ra[mg], n_regions=int(n_ra))
                ra_edges_list.append(edges_ra)
                ra_shift_list.append(float(shift_ra))

                # Galaxies in band.
                ra_g_shifted = np.mod(gal_ra[mg] - float(shift_ra), 360.0)
                ra_id_g = _assign_region_ra(ra_g_shifted, edges_ra)
                jk_g[mg] = offset + int(j) * int(n_ra) + ra_id_g

                # Randoms in band.
                mr = rnd_dec_id == int(j)
                # 条件分岐: `np.any(mr)` を満たす経路を評価する。
                if np.any(mr):
                    ra_r_shifted = np.mod(rnd_ra[mr] - float(shift_ra), 360.0)
                    ra_id_r = _assign_region_ra(ra_r_shifted, edges_ra)
                    jk_r[mr] = offset + int(j) * int(n_ra) + ra_id_r

            p["gal"]["jk_id"] = jk_g
            p["rnd"]["jk_id"] = jk_r
            ra_edges_by_cap_by_dec[cap] = ra_edges_list
            ra_shift_by_cap_by_dec[cap] = ra_shift_list
    else:
        raise SystemExit(f"unsupported --jk-mode: {jk_mode}")

    y_jk: List[np.ndarray] = []
    t0 = time.time()
    for jk_i in range(n_jk):
        per_cap: List[Dict[str, Any]] = []
        for p in cap_inputs:
            mg = np.asarray(p["gal"]["jk_id"], dtype=np.int32) != int(jk_i)
            mr = np.asarray(p["rnd"]["jk_id"], dtype=np.int32) != int(jk_i)
            gal = p["gal"]
            rnd = p["rnd"]
            ra_g = np.asarray(gal["ra"], dtype=np.float64)[mg]
            dec_g = np.asarray(gal["dec"], dtype=np.float64)[mg]
            d_g = np.asarray(gal["d"], dtype=np.float64)[mg]
            w_g = np.asarray(gal["w"], dtype=np.float64)[mg]
            ra_r = np.asarray(rnd["ra"], dtype=np.float64)[mr]
            dec_r = np.asarray(rnd["dec"], dtype=np.float64)[mr]
            d_r = np.asarray(rnd["d"], dtype=np.float64)[mr]
            w_r = np.asarray(rnd["w"], dtype=np.float64)[mr]

            totals = _weighted_totals(w_g, w_r)
            counts = _paircounts_for_subset(
                ra_g=ra_g,
                dec_g=dec_g,
                d_g=d_g,
                w_g=w_g,
                ra_r=ra_r,
                dec_r=dec_r,
                d_r=d_r,
                w_r=w_r,
                s_bins_file=s_bins_file,
                mu_max=mu_max,
                nmu=nmu,
                nthreads=int(args.threads),
            )
            per_cap.append(
                {
                    "cap": p["cap"],
                    "totals": totals,
                    "counts": counts,
                }
            )

        # Combine caps in the same way as xi-from-catalogs (random weight rescale to align gal/rnd ratios).

        dd_w = sum(np.asarray(pp["counts"]["DD_w"], dtype=np.float64) for pp in per_cap)
        sum_w_gal_total = sum(float(pp["totals"]["sum_w_gal"]) for pp in per_cap)
        sum_w_rnd_total = sum(float(pp["totals"]["sum_w_rnd"]) for pp in per_cap)
        ratio_target = sum_w_gal_total / max(1e-30, sum_w_rnd_total)

        dr_w = 0.0
        rr_w = 0.0
        dr_tot = 0.0
        rr_tot = 0.0
        for pp in per_cap:
            sum_w_gal_i = float(pp["totals"]["sum_w_gal"])
            sum_w_rnd_i = float(pp["totals"]["sum_w_rnd"])
            f = (sum_w_gal_i / max(1e-30, sum_w_rnd_i)) / max(1e-30, ratio_target)
            dr_w = dr_w + float(f) * np.asarray(pp["counts"]["DR_w"], dtype=np.float64)
            rr_w = rr_w + float(f * f) * np.asarray(pp["counts"]["RR_w"], dtype=np.float64)
            dr_tot = dr_tot + float(f) * float(pp["totals"]["dr_tot"])
            rr_tot = rr_tot + float(f * f) * float(pp["totals"]["rr_tot"])

        dd_tot = sum(float(pp["totals"]["dd_tot"]) for pp in per_cap)
        out_xi = _xi._xi_multipoles_from_paircounts(
            dd_w=dd_w,
            dr_w=dr_w,
            rr_w=rr_w,
            edges=edges,
            mu_max=float(mu_max),
            nmu=int(nmu),
            dd_tot=float(dd_tot),
            dr_tot=float(dr_tot),
            rr_tot=float(rr_tot),
        )
        xi0 = np.asarray(out_xi["xi0"], dtype=np.float64).reshape(-1)
        xi2 = np.asarray(out_xi["xi2"], dtype=np.float64).reshape(-1)
        y_jk.append(np.concatenate([xi0, xi2], axis=0))
        dt = time.time() - t0
        print(f"[jk] {jk_i+1}/{n_jk} done  (elapsed={dt:.1f}s)")

    y = np.stack(y_jk, axis=0)  # (n_jk, 2*n_s)
    y_mean = np.mean(y, axis=0)
    dy = y - y_mean[None, :]
    cov = (float(n_jk - 1) / float(n_jk)) * (dy.T @ dy)

    # Sanity: load s from the reference xi output (must match).
    with np.load(xi_npz_path) as z:
        s_ref = np.asarray(z["s"], dtype=np.float64).reshape(-1)

    s_here = np.asarray(out_xi["s"], dtype=np.float64).reshape(-1)
    # 条件分岐: `s_ref.shape != s_here.shape or not np.allclose(s_ref, s_here, rtol=0.0, atol=...` を満たす経路を評価する。
    if s_ref.shape != s_here.shape or not np.allclose(s_ref, s_here, rtol=0.0, atol=1e-12):
        raise SystemExit("s bins mismatch vs reference xi output (tag mismatch or different binning)")

    np.savez_compressed(
        out_cov_npz,
        s=s_ref,
        cov=cov,
        y_jk=y,
        ra_edges_deg=ra_edges.astype(np.float64),
    )

    jk_meta: Dict[str, Any] = {"mode": str(jk_mode), "n_regions": int(n_jk)}
    # 条件分岐: `grid_meta is not None` を満たす経路を評価する。
    if grid_meta is not None:
        jk_meta["grid"] = dict(grid_meta)

    # 条件分岐: `jk_mode == "ra_quantile"` を満たす経路を評価する。

    if jk_mode == "ra_quantile":
        jk_meta["ra_edges_deg"] = ra_edges.tolist()
    # 条件分岐: 前段条件が不成立で、`jk_mode == "ra_quantile_unwrapped"` を追加評価する。
    elif jk_mode == "ra_quantile_unwrapped":
        jk_meta["ra_edges_deg_shifted"] = ra_edges.tolist()
        jk_meta["ra_shift_deg"] = float(ra_shift_deg) if ra_shift_deg is not None else 0.0
    # 条件分岐: 前段条件が不成立で、`jk_mode == "ra_quantile_per_cap"` を追加評価する。
    elif jk_mode == "ra_quantile_per_cap":
        # 条件分岐: `ra_edges_by_cap` を満たす経路を評価する。
        if ra_edges_by_cap:
            jk_meta["ra_edges_by_cap_deg"] = {k: v.tolist() for k, v in ra_edges_by_cap.items()}
    # 条件分岐: 前段条件が不成立で、`jk_mode == "ra_quantile_per_cap_unwrapped"` を追加評価する。
    elif jk_mode == "ra_quantile_per_cap_unwrapped":
        # 条件分岐: `ra_edges_by_cap` を満たす経路を評価する。
        if ra_edges_by_cap:
            jk_meta["ra_edges_by_cap_deg_shifted"] = {k: v.tolist() for k, v in ra_edges_by_cap.items()}
            jk_meta["ra_shift_by_cap_deg"] = {k: float(v) for k, v in ra_shift_by_cap.items()}
    # 条件分岐: 前段条件が不成立で、`jk_mode == "ra_dec_quantile_per_cap_unwrapped"` を追加評価する。
    elif jk_mode == "ra_dec_quantile_per_cap_unwrapped":
        # 条件分岐: `dec_edges_by_cap` を満たす経路を評価する。
        if dec_edges_by_cap:
            jk_meta["dec_edges_by_cap_deg"] = {k: v.tolist() for k, v in dec_edges_by_cap.items()}

        # 条件分岐: `ra_edges_by_cap_by_dec` を満たす経路を評価する。

        if ra_edges_by_cap_by_dec:
            jk_meta["ra_edges_by_cap_by_dec_deg_shifted"] = {
                cap: [np.asarray(e, dtype=float).tolist() for e in edges_list] for cap, edges_list in ra_edges_by_cap_by_dec.items()
            }

        # 条件分岐: `ra_shift_by_cap_by_dec` を満たす経路を評価する。

        if ra_shift_by_cap_by_dec:
            jk_meta["ra_shift_by_cap_by_dec_deg"] = {cap: [float(x) for x in xs] for cap, xs in ra_shift_by_cap_by_dec.items()}

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "4.5B.21.4.4.4 (jackknife covariance for catalog-based xi multipoles)",
        "inputs": {
            "xi_metrics_json": str(metrics_path),
            "xi_npz": str(xi_npz_path),
            "galaxy_npz": [str(_xi._resolve_manifest_path(str(x))) for x in gal_list],
            "random_npz": [str(_xi._resolve_manifest_path(str(x))) for x in rnd_list],
        },
        "params": {
            "tag": tag,
            "distance_model": dist_model,
            "z_source": z_source,
            "weight_scheme": weight_scheme,
            "z_min": z_min,
            "z_max": z_max,
            "s_bins": {"min": s_min, "max": s_max, "step": s_step},
            "mu_bins": {"mu_max": mu_max, "nmu": int(nmu)},
            "corrfunc_threads": int(args.threads),
            "jackknife": jk_meta,
            "pinv_rcond_metrics_only": float(args.rcond),
        },
        "outputs": {"cov_npz": str(out_cov_npz), "metrics_json": str(out_cov_json)},
        "summary": {
            "n_jk": int(n_jk),
            "dv_dim": int(y.shape[1]),
        },
    }
    out_cov_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_bao_xi_jackknife_cov_from_catalogs",
                "argv": sys.argv,
                "inputs": {"xi_metrics_json": metrics_path, "xi_npz": xi_npz_path},
                "outputs": {"cov_npz": out_cov_npz, "metrics_json": out_cov_json},
                "metrics": {"tag": tag, "jk_n": int(n_jk), "threads": int(args.threads), "jk_mode": str(args.jk_mode)},
            }
        )
    except Exception:
        pass

    print(f"[ok] cov npz : {out_cov_npz}")
    print(f"[ok] metrics : {out_cov_json}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
