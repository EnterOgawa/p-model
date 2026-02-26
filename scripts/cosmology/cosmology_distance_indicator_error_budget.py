#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_distance_indicator_error_budget.py

Step 14.2.15（距離指標の“誤差予算”を一次ソースで設定）:
距離指標（SNe Ia / BAO）の観測不確かさを一次データから z 依存で見積もり、
静的背景P最小（DDR: ε0=-1）が“隠れ得る範囲”を定量化する。

ここでの誤差予算は「距離指標（D_L, D_A）の観測不確かさ（目安）」であり、
距離指標の構成（標準化・校正・系統）そのものの再導出が必要である点は別途扱う。

入力（固定）:
  - data/cosmology/pantheon_lcparam_full_long.txt（Pantheon; lcparam）
  - data/cosmology/pantheon_sys_full_long.txt（Pantheon; systematics covariance for distance moduli）
  - data/cosmology/alcock_paczynski_constraints.json（BOSS DR12; BAO距離プロダクト）
  - data/cosmology/boss_dr12_baofs_consensus_reduced_covariance_cij.json（BOSS DR12; BAO+FS reduced covariance c_ij）
  - data/cosmology/bao_sound_horizon_constraints.json（Planck r_drag; BAO校正スケール）
  - （任意）output/private/cosmology/cosmology_distance_indicator_reach_limit_metrics.json（DDR→必要補正 Δμ）

出力（固定名）:
  - output/private/cosmology/cosmology_distance_indicator_error_budget.png
  - output/private/cosmology/cosmology_distance_indicator_error_budget_metrics.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = [
            "Yu Gothic",
            "Meiryo",
            "BIZ UDGothic",
            "MS Gothic",
            "Yu Mincho",
            "MS Mincho",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def _fmt_float(x: Optional[float], *, digits: int = 6) -> str:
    # 条件分岐: `x is None` を満たす経路を評価する。
    if x is None:
        return ""

    # 条件分岐: `x == 0.0` を満たす経路を評価する。

    if x == 0.0:
        return "0"

    ax = abs(x)
    # 条件分岐: `ax >= 1e4 or ax < 1e-3` を満たす経路を評価する。
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"

    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _safe_float(x: Any) -> Optional[float]:
    try:
        # 条件分岐: `x is None` を満たす経路を評価する。
        if x is None:
            return None

        return float(x)
    except Exception:
        return None


def _load_pantheon_lcparam(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    # 条件分岐: `"#name" in df.columns` を満たす経路を評価する。
    if "#name" in df.columns:
        df = df.rename(columns={"#name": "name"})

    return df


def _load_pantheon_sys_cov(path: Path) -> np.ndarray:
    """
    Pantheon public release provides a systematics covariance matrix (sys_full_long.txt).

    File format (Pantheon repo):
      - First line: N
      - Then N*N floats, typically one per line (variance/covariance in mag^2).
    """
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        raise FileNotFoundError(str(path))

    first = path.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()
    try:
        n = int(first)
    except Exception as e:
        raise ValueError(f"Invalid first line for Pantheon sys covariance (expected N int): {first!r}") from e

    data = np.loadtxt(path, dtype=float, skiprows=1)
    # 条件分岐: `data.size != int(n) * int(n)` を満たす経路を評価する。
    if data.size != int(n) * int(n):
        raise ValueError(f"Pantheon sys covariance size mismatch: got {data.size}, expected {n*n}")

    return data.reshape((n, n))


def _cov_sem_weighted(cov: np.ndarray) -> Optional[float]:
    """
    Generalized SEM for correlated errors:
      Var(mu_hat) = 1 / (1^T C^{-1} 1)
    which matches inverse-variance SEM for diagonal C.
    """
    try:
        n = int(cov.shape[0])
    except Exception:
        return None

    # 条件分岐: `n <= 0 or cov.shape != (n, n)` を満たす経路を評価する。

    if n <= 0 or cov.shape != (n, n):
        return None

    ones = np.ones(n, dtype=float)
    try:
        x = np.linalg.solve(cov, ones)
    except Exception:
        return None

    denom = float(ones @ x)
    # 条件分岐: `not (denom > 0.0) or not math.isfinite(denom)` を満たす経路を評価する。
    if not (denom > 0.0) or not math.isfinite(denom):
        return None

    v = math.sqrt(1.0 / denom)
    return v if math.isfinite(v) else None


def _compute_sn_binned_budget(
    df: pd.DataFrame,
    *,
    z_max: float,
    bin_width: float,
    min_points: int,
    sys_cov: Optional[np.ndarray],
) -> List[Dict[str, Any]]:
    # 条件分岐: `"zcmb" not in df.columns or "dmb" not in df.columns` を満たす経路を評価する。
    if "zcmb" not in df.columns or "dmb" not in df.columns:
        raise ValueError("Pantheon lcparam must contain columns: zcmb, dmb")

    bins = np.arange(0.0, float(z_max) + float(bin_width) + 1e-9, float(bin_width))
    out: List[Dict[str, Any]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub0 = df[(df["zcmb"] >= lo) & (df["zcmb"] < hi)]
        n0 = int(len(sub0))
        z_center = 0.5 * (float(lo) + float(hi))

        sub = sub0[np.isfinite(sub0["dmb"]) & (sub0["dmb"] > 0)].copy()
        n = int(len(sub))
        # 条件分岐: `n < int(min_points)` を満たす経路を評価する。
        if n < int(min_points):
            out.append(
                {
                    "z_center": z_center,
                    "z_lo": float(lo),
                    "z_hi": float(hi),
                    "n": n0,
                    "n_valid": n,
                    "dmb_median": None,
                    "dmb_sem_weighted": None,
                    "dmb_median_total": None,
                    "dmb_sem_total_cov": None,
                }
            )
            continue

        sig = sub["dmb"].to_numpy(dtype=float)
        idx = sub.index.to_numpy(dtype=int)

        dmb_median = float(np.nanmedian(sig))
        w = 1.0 / (sig * sig)
        sem = float(math.sqrt(1.0 / float(w.sum())))

        dmb_median_total: Optional[float] = None
        sem_total_cov: Optional[float] = None
        # 条件分岐: `sys_cov is not None` を満たす経路を評価する。
        if sys_cov is not None:
            try:
                cov_sys = sys_cov[np.ix_(idx, idx)].astype(float, copy=False)
                cov_total = cov_sys + np.diag(sig * sig)
                sem_total_cov = _cov_sem_weighted(cov_total)

                diag_total = np.diag(cov_total)
                diag_total = diag_total[np.isfinite(diag_total) & (diag_total >= 0.0)]
                # 条件分岐: `diag_total.size` を満たす経路を評価する。
                if diag_total.size:
                    dmb_median_total = float(np.nanmedian(np.sqrt(diag_total)))
            except Exception:
                sem_total_cov = None
                dmb_median_total = None

        out.append(
            {
                "z_center": z_center,
                "z_lo": float(lo),
                "z_hi": float(hi),
                "n": n0,
                "n_valid": n,
                "dmb_median": dmb_median,
                "dmb_sem_weighted": sem,
                "dmb_median_total": dmb_median_total,
                "dmb_sem_total_cov": sem_total_cov,
            }
        )

    return out


def _bao_mu_sigma_mag_from_boss_ap(path: Path) -> List[Dict[str, Any]]:
    src = _read_json(path)
    rows = src.get("constraints") or []
    out: List[Dict[str, Any]] = []
    for r in rows:
        z = _safe_float(r.get("z_eff"))
        dm = _safe_float(r.get("DM_scaled_mpc"))
        sig = _safe_float(r.get("DM_scaled_sigma_mpc"))
        # 条件分岐: `z is None or dm is None or sig is None or not (dm > 0) or not (sig > 0)` を満たす経路を評価する。
        if z is None or dm is None or sig is None or not (dm > 0) or not (sig > 0):
            continue

        frac = float(sig) / float(dm)
        mu = 5.0 * math.log10(1.0 + frac)
        out.append(
            {
                "id": str(r.get("id") or ""),
                "short_label": str(r.get("short_label") or r.get("id") or ""),
                "z_eff": float(z),
                "DM_scaled_mpc": float(dm),
                "DM_scaled_sigma_mpc": float(sig),
                "frac_sigma": float(frac),
                "mu_sigma_mag_equiv": float(mu),
                "source": r.get("source") or {},
            }
        )

    out.sort(key=lambda x: x["z_eff"])
    return out


def _r_drag_mu_sigma_mag(path: Path) -> Optional[Dict[str, Any]]:
    src = _read_json(path)
    rows = src.get("constraints") or []
    # 条件分岐: `not rows` を満たす経路を評価する。
    if not rows:
        return None
    # pick the smallest sigma as "primary"

    rows = sorted(rows, key=lambda r: float(r.get("r_drag_sigma_mpc", float("inf"))))
    r = rows[0]
    val = _safe_float(r.get("r_drag_mpc"))
    sig = _safe_float(r.get("r_drag_sigma_mpc"))
    # 条件分岐: `val is None or sig is None or not (val > 0) or not (sig > 0)` を満たす経路を評価する。
    if val is None or sig is None or not (val > 0) or not (sig > 0):
        return None

    frac = float(sig) / float(val)
    mu = 5.0 * math.log10(1.0 + frac)
    return {
        "id": str(r.get("id") or ""),
        "short_label": str(r.get("short_label") or r.get("id") or ""),
        "r_drag_mpc": float(val),
        "r_drag_sigma_mpc": float(sig),
        "frac_sigma": float(frac),
        "mu_sigma_mag_equiv": float(mu),
        "sigma_note": str(r.get("sigma_note") or ""),
        "source": r.get("source") or {},
    }


def _load_boss_baofs_reduced_cov_cij(path: Path) -> Dict[str, Any]:
    src = _read_json(path)
    params = src.get("parameters") or []
    cij = src.get("cij_1e4")
    # 条件分岐: `not isinstance(params, list) or not params` を満たす経路を評価する。
    if not isinstance(params, list) or not params:
        raise ValueError("BOSS reduced covariance JSON must contain non-empty 'parameters' list")

    # 条件分岐: `not isinstance(cij, list) or not cij` を満たす経路を評価する。

    if not isinstance(cij, list) or not cij:
        raise ValueError("BOSS reduced covariance JSON must contain 'cij_1e4' matrix")

    mat = np.array(cij, dtype=float)
    # 条件分岐: `mat.ndim != 2 or mat.shape[0] != mat.shape[1]` を満たす経路を評価する。
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Invalid 'cij_1e4' shape: {mat.shape}")

    # 条件分岐: `mat.shape[0] != len(params)` を満たす経路を評価する。

    if mat.shape[0] != len(params):
        raise ValueError(f"'cij_1e4' size {mat.shape[0]} does not match parameters {len(params)}")

    return {"raw": src, "parameters": params, "cij_1e4": mat}


def _match_boss_dm_indices(
    boss_params: List[Dict[str, Any]], bao_points: List[Dict[str, Any]], *, z_tol: float = 5e-4
) -> Tuple[List[int], List[Dict[str, Any]]]:
    dm_params = [(i, p) for i, p in enumerate(boss_params) if str(p.get("kind") or "") == "DM_scaled_mpc"]
    # 条件分岐: `not dm_params` を満たす経路を評価する。
    if not dm_params:
        raise ValueError("BOSS reduced covariance JSON has no DM_scaled_mpc parameters")

    out_idx: List[int] = []
    out_param: List[Dict[str, Any]] = []
    for bp in bao_points:
        z = float(bp["z_eff"])
        best = None
        best_d = float("inf")
        for i, p in dm_params:
            zz = _safe_float(p.get("z_eff"))
            # 条件分岐: `zz is None` を満たす経路を評価する。
            if zz is None:
                continue

            d = abs(float(zz) - float(z))
            # 条件分岐: `d < best_d` を満たす経路を評価する。
            if d < best_d:
                best_d = d
                best = (i, p)

        # 条件分岐: `best is None or best_d > float(z_tol)` を満たす経路を評価する。

        if best is None or best_d > float(z_tol):
            raise ValueError(f"Cannot match BAO z={z} to BOSS DM parameter (best_d={best_d})")

        out_idx.append(int(best[0]))
        out_param.append(dict(best[1]))

    return out_idx, out_param


def _bao_mu_sigma_mag_from_dm_cov_interpolated(
    *,
    z_grid: np.ndarray,
    z_points: np.ndarray,
    dm_points: np.ndarray,
    dm_cov: np.ndarray,
) -> np.ndarray:
    """
    Build an approximate BAO uncertainty curve σ_mu(z) from D_M(z) points with covariance.

    We use piecewise-linear interpolation for D_M(z), and propagate variance using the
    covariance of the two adjacent anchor points (weights 2-point only).
    """
    # 条件分岐: `z_grid.ndim != 1` を満たす経路を評価する。
    if z_grid.ndim != 1:
        raise ValueError("z_grid must be 1D")

    # 条件分岐: `z_points.size < 2` を満たす経路を評価する。

    if z_points.size < 2:
        # fallback: constant uncertainty from the single point
        dm0 = float(dm_points[0])
        sig0 = float(math.sqrt(float(dm_cov[0, 0])))
        frac = sig0 / dm0 if dm0 > 0 else 0.0
        return np.full_like(z_grid, 5.0 * np.log10(1.0 + frac), dtype=float)

    # Sort by z.

    order = np.argsort(z_points)
    z_pts = z_points[order]
    dm_pts = dm_points[order]
    cov_pts = dm_cov[np.ix_(order, order)]

    out_mu = np.zeros_like(z_grid, dtype=float)
    # Left of first point: hold constant.
    dm0 = float(dm_pts[0])
    var0 = float(cov_pts[0, 0])
    sig0 = math.sqrt(max(0.0, var0))
    frac0 = sig0 / dm0 if dm0 > 0 else 0.0
    out_mu[z_grid <= z_pts[0]] = 5.0 * np.log10(1.0 + frac0)

    # Right of last point: hold constant.
    dmn = float(dm_pts[-1])
    varn = float(cov_pts[-1, -1])
    sign = math.sqrt(max(0.0, varn))
    fracn = sign / dmn if dmn > 0 else 0.0
    out_mu[z_grid >= z_pts[-1]] = 5.0 * np.log10(1.0 + fracn)

    # Interior: piecewise between adjacent points.
    for i in range(len(z_pts) - 1):
        z1 = float(z_pts[i])
        z2 = float(z_pts[i + 1])
        # 条件分岐: `not (z2 > z1)` を満たす経路を評価する。
        if not (z2 > z1):
            continue

        mask = (z_grid > z1) & (z_grid < z2)
        # 条件分岐: `not mask.any()` を満たす経路を評価する。
        if not mask.any():
            continue

        t = (z_grid[mask] - z1) / (z2 - z1)
        dm_i = float(dm_pts[i])
        dm_j = float(dm_pts[i + 1])
        # D_M linear interpolation (proxy).
        dm_z = (1.0 - t) * dm_i + t * dm_j

        var_i = float(cov_pts[i, i])
        var_j = float(cov_pts[i + 1, i + 1])
        cov_ij = float(cov_pts[i, i + 1])
        var_z = (1.0 - t) ** 2 * var_i + t**2 * var_j + 2.0 * t * (1.0 - t) * cov_ij
        var_z = np.maximum(0.0, var_z)
        sig_z = np.sqrt(var_z)
        frac_z = np.where(dm_z > 0.0, sig_z / dm_z, 0.0)
        out_mu[mask] = 5.0 * np.log10(1.0 + frac_z)

    return out_mu


def _load_reach_representatives(reach_metrics_path: Path) -> Dict[str, Any]:
    src = _read_json(reach_metrics_path)
    reps = src.get("representatives") or {}
    reach = src.get("reach") or {}

    def extract(key: str) -> Optional[Dict[str, Any]]:
        rr = reach.get(key)
        # 条件分岐: `not isinstance(rr, dict)` を満たす経路を評価する。
        if not isinstance(rr, dict):
            return None

        return {
            "label": str(key),
            "id": str(rr.get("id") or ""),
            "short_label": str(rr.get("short_label") or rr.get("id") or ""),
            "uses_bao": bool(rr.get("uses_bao", False)),
            "delta_eps_needed": float(rr.get("delta_eps_needed", float("nan"))),
            "epsilon0_obs": float(rr.get("epsilon0_obs", float("nan"))),
            "epsilon0_sigma": float(rr.get("epsilon0_sigma", float("nan"))),
        }

    out = {
        "representatives": reps,
        "reach": {
            "bao": extract("bao"),
            "no_bao": extract("no_bao"),
        },
    }
    return out


def _interp_on_grid(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    # 条件分岐: `x.size == 0 or y.size == 0` を満たす経路を評価する。
    if x.size == 0 or y.size == 0:
        return np.full_like(x_grid, np.nan, dtype=float)
    # Fill NaNs by nearest-neighbor on available points.

    mask = np.isfinite(x) & np.isfinite(y)
    # 条件分岐: `not mask.any()` を満たす経路を評価する。
    if not mask.any():
        return np.full_like(x_grid, np.nan, dtype=float)

    xs = x[mask]
    ys = y[mask]
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    return np.interp(x_grid, xs, ys, left=float(ys[0]), right=float(ys[-1]))


def _required_delta_mu_mag(delta_eps_needed: float, z: np.ndarray) -> np.ndarray:
    op = 1.0 + z
    log10_op = np.log10(op)
    return np.abs(5.0 * float(delta_eps_needed) * log10_op)


def _z_limit(required: np.ndarray, budget: np.ndarray, z: np.ndarray, *, sigma_multiplier: float) -> Optional[float]:
    ok = np.isfinite(required) & np.isfinite(budget) & (budget >= 0.0) & (z >= 0.0)
    # 条件分岐: `not ok.any()` を満たす経路を評価する。
    if not ok.any():
        return None

    rr = required[ok]
    bb = budget[ok] * float(sigma_multiplier)
    zz = z[ok]
    # Find first index where required exceeds budget.
    exceed = rr > bb
    # 条件分岐: `not exceed.any()` を満たす経路を評価する。
    if not exceed.any():
        return float(zz[-1])

    idx = int(np.argmax(exceed))
    # 条件分岐: `idx <= 0` を満たす経路を評価する。
    if idx <= 0:
        return float(zz[0])

    return float(zz[idx - 1])


def _plot(
    *,
    out_png: Path,
    reach_reps: Dict[str, Any],
    sn_bins: List[Dict[str, Any]],
    bao_points: List[Dict[str, Any]],
    boss_baofs_cov: Optional[Dict[str, Any]],
    r_drag_budget: Optional[Dict[str, Any]],
    z_max: float,
    sn_min_points: int,
    sigma_multipliers: Sequence[float],
) -> Dict[str, Any]:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    # Prepare SN arrays.
    sn_z = np.array([float(r["z_center"]) for r in sn_bins], dtype=float)
    sn_med = np.array(
        [float(r["dmb_median"]) if r.get("dmb_median") is not None else float("nan") for r in sn_bins],
        dtype=float,
    )
    sn_sem = np.array(
        [float(r["dmb_sem_weighted"]) if r.get("dmb_sem_weighted") is not None else float("nan") for r in sn_bins],
        dtype=float,
    )
    sn_sem_total = np.array(
        [float(r["dmb_sem_total_cov"]) if r.get("dmb_sem_total_cov") is not None else float("nan") for r in sn_bins],
        dtype=float,
    )
    sn_med_total = np.array(
        [float(r["dmb_median_total"]) if r.get("dmb_median_total") is not None else float("nan") for r in sn_bins],
        dtype=float,
    )

    # BAO arrays (mu-equivalent).
    bao_z = np.array([float(r["z_eff"]) for r in bao_points], dtype=float)
    bao_mu = np.array([float(r["mu_sigma_mag_equiv"]) for r in bao_points], dtype=float)
    bao_dm = np.array([float(r["DM_scaled_mpc"]) for r in bao_points], dtype=float)

    sn_valid = np.isfinite(sn_z) & (np.isfinite(sn_med) | np.isfinite(sn_sem))
    sn_max_z = float(np.nanmax(sn_z[sn_valid])) if sn_valid.any() else None
    bao_max_z = float(np.nanmax(bao_z)) if bao_z.size else None

    # Build smooth grids for comparing to required correction.
    z_grid = np.linspace(0.0, float(z_max), 1200)
    sn_sem_grid = _interp_on_grid(sn_z, sn_sem, z_grid)
    sn_sem_total_grid = _interp_on_grid(sn_z, sn_sem_total, z_grid)
    sn_med_grid = _interp_on_grid(sn_z, sn_med, z_grid)
    sn_med_total_grid = _interp_on_grid(sn_z, sn_med_total, z_grid)
    bao_mu_grid_naive = _interp_on_grid(bao_z, bao_mu, z_grid) if bao_z.size else np.zeros_like(z_grid)
    bao_mu_grid_cov = np.array(bao_mu_grid_naive, copy=True)
    rdrag_mu = float(r_drag_budget["mu_sigma_mag_equiv"]) if isinstance(r_drag_budget, dict) else 0.0

    boss_dm_cov = None
    boss_dm_corr = None
    boss_dm_sigmas = None
    boss_dm_match = []
    # 条件分岐: `boss_baofs_cov is not None and bao_z.size` を満たす経路を評価する。
    if boss_baofs_cov is not None and bao_z.size:
        try:
            boss_params = list(boss_baofs_cov["parameters"])
            cij_1e4 = np.array(boss_baofs_cov["cij_1e4"], dtype=float)
            dm_idx, dm_params = _match_boss_dm_indices(boss_params, bao_points)
            boss_dm_match = [{"z_eff": float(p.get("z_eff")), "id": str(p.get("id") or ""), "sigma": float(p.get("sigma"))} for p in dm_params]
            dm_corr = (cij_1e4[np.ix_(dm_idx, dm_idx)] / 10000.0).astype(float)
            dm_sigma = np.array([float(p.get("sigma")) for p in dm_params], dtype=float)
            dm_cov = dm_corr * np.outer(dm_sigma, dm_sigma)
            # Ensure symmetry.
            dm_cov = 0.5 * (dm_cov + dm_cov.T)

            boss_dm_cov = dm_cov
            boss_dm_corr = dm_corr
            boss_dm_sigmas = dm_sigma

            bao_mu_grid_cov = _bao_mu_sigma_mag_from_dm_cov_interpolated(
                z_grid=z_grid,
                z_points=bao_z,
                dm_points=bao_dm,
                dm_cov=dm_cov,
            )
        except Exception:
            boss_dm_cov = None
            boss_dm_corr = None
            boss_dm_sigmas = None
            bao_mu_grid_cov = np.array(bao_mu_grid_naive, copy=True)

    # Combined budgets (quadrature; observational uncertainty proxy).
    # Lower bound: stat-only (SNe) + naive BAO interpolation.

    budget_opt = np.sqrt(np.maximum(0.0, sn_sem_grid) ** 2 + np.maximum(0.0, bao_mu_grid_naive) ** 2 + rdrag_mu**2)
    budget_cons = np.sqrt(np.maximum(0.0, sn_med_grid) ** 2 + np.maximum(0.0, bao_mu_grid_naive) ** 2 + rdrag_mu**2)
    # Upper bound: stat+sys (SNe) + BAO covariance-interpolated curve.
    budget_opt_total = np.sqrt(
        np.maximum(0.0, sn_sem_total_grid) ** 2 + np.maximum(0.0, bao_mu_grid_cov) ** 2 + rdrag_mu**2
    )
    budget_cons_total = np.sqrt(
        np.maximum(0.0, sn_med_total_grid) ** 2 + np.maximum(0.0, bao_mu_grid_cov) ** 2 + rdrag_mu**2
    )

    reps = reach_reps.get("reach") or {}
    rep_bao = reps.get("bao") or {}
    rep_no = reps.get("no_bao") or {}

    dm_req_bao = _required_delta_mu_mag(float(rep_bao.get("delta_eps_needed", float("nan"))), z_grid) if rep_bao else None
    dm_req_no = _required_delta_mu_mag(float(rep_no.get("delta_eps_needed", float("nan"))), z_grid) if rep_no else None

    # z-limits for reporting.
    limits: Dict[str, Any] = {"opt": {}, "cons": {}}
    limits_total: Dict[str, Any] = {"opt": {}, "cons": {}}
    for label, dm_req in [("bao", dm_req_bao), ("no_bao", dm_req_no)]:
        # 条件分岐: `dm_req is None` を満たす経路を評価する。
        if dm_req is None:
            continue

        limits["opt"][label] = {}
        limits["cons"][label] = {}
        limits_total["opt"][label] = {}
        limits_total["cons"][label] = {}
        for k, bud, bud_total in [
            ("budget_opt", budget_opt, budget_opt_total),
            ("budget_cons", budget_cons, budget_cons_total),
        ]:
            for m in sigma_multipliers:
                z_lim = _z_limit(dm_req, bud, z_grid, sigma_multiplier=float(m))
                z_lim_total = _z_limit(dm_req, bud_total, z_grid, sigma_multiplier=float(m))
                tgt = limits["opt"][label] if k == "budget_opt" else limits["cons"][label]
                tgt[f"{m}sigma"] = z_lim
                tgt2 = limits_total["opt"][label] if k == "budget_opt" else limits_total["cons"][label]
                tgt2[f"{m}sigma"] = z_lim_total

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.8))

    # Left: observational budget components.
    ax1.plot(sn_z, sn_sem, color="#2ca02c", linewidth=2.0, label="SNe Ia: bin平均の誤差（SEM; stat; Pantheon）")
    # 条件分岐: `np.isfinite(sn_sem_total).any()` を満たす経路を評価する。
    if np.isfinite(sn_sem_total).any():
        ax1.plot(
            sn_z,
            sn_sem_total,
            color="#2ca02c",
            linestyle="--",
            linewidth=2.0,
            alpha=0.9,
            label="SNe Ia: bin平均の誤差（SEM; stat+sys共分散; Pantheon）",
        )

    ax1.scatter(sn_z, sn_med, color="#1f77b4", s=18, alpha=0.75, label="SNe Ia: 1点の誤差（median dmb; Pantheon）")
    # 条件分岐: `bao_z.size` を満たす経路を評価する。
    if bao_z.size:
        ax1.plot(bao_z, bao_mu, color="#9467bd", marker="s", linewidth=1.8, label="BAO: 距離の誤差（BOSS DR12; D_M）")
        # 条件分岐: `boss_dm_cov is not None and np.isfinite(bao_mu_grid_cov).any()` を満たす経路を評価する。
        if boss_dm_cov is not None and np.isfinite(bao_mu_grid_cov).any():
            zmin = float(np.nanmin(bao_z))
            zmax_ = float(np.nanmax(bao_z))
            msk = (z_grid >= zmin) & (z_grid <= zmax_)
            # 条件分岐: `msk.any()` を満たす経路を評価する。
            if msk.any():
                ax1.plot(
                    z_grid[msk],
                    bao_mu_grid_cov[msk],
                    color="#9467bd",
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.65,
                    label="BAO: 共分散を用いた補間（目安）",
                )

    # 条件分岐: `isinstance(r_drag_budget, dict)` を満たす経路を評価する。

    if isinstance(r_drag_budget, dict):
        ax1.axhline(rdrag_mu, color="#777777", linestyle="--", linewidth=1.2, alpha=0.7, label="r_drag校正（Planck; 1σ）")

    ax1.set_title("距離指標の誤差予算（1σ; 観測の不確かさの目安）", fontsize=12)
    ax1.set_xlabel("赤方偏移 z", fontsize=11)
    ax1.set_ylabel("等価 |Δμ| [mag]", fontsize=11)
    ax1.set_xlim(0.0, float(z_max))
    ax1.set_ylim(0.0, max(0.35, float(np.nanmax(sn_med[np.isfinite(sn_med)]) if np.isfinite(sn_med).any() else 0.3) * 1.2))
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(fontsize=9, loc="upper left")

    # Right: required correction vs budgets.
    if dm_req_bao is not None and rep_bao:
        ax2.plot(z_grid, dm_req_bao, color="#1f77b4", linewidth=2.2, label=f"必要補正 |Δμ|（DDR; {rep_bao.get('short_label','')})")

    # 条件分岐: `dm_req_no is not None and rep_no` を満たす経路を評価する。

    if dm_req_no is not None and rep_no:
        ax2.plot(z_grid, dm_req_no, color="#ff7f0e", linewidth=2.2, label=f"必要補正 |Δμ|（DDR; {rep_no.get('short_label','')})")

    # Error-budget band: stat-only → stat+sys covariance (Pantheon) for bin-mean SEM case.

    if np.isfinite(budget_opt_total).any():
        for m in sigma_multipliers:
            lo = float(m) * budget_opt
            hi = float(m) * budget_opt_total
            # 条件分岐: `not (np.isfinite(lo).any() and np.isfinite(hi).any())` を満たす経路を評価する。
            if not (np.isfinite(lo).any() and np.isfinite(hi).any()):
                continue
            # Ensure lo<=hi elementwise for fill.

            y1 = np.minimum(lo, hi)
            y2 = np.maximum(lo, hi)
            a = 0.14 if float(m) == 1.0 else 0.06
            label = None
            # 条件分岐: `float(m) == 1.0` を満たす経路を評価する。
            if float(m) == 1.0:
                label = "誤差予算（SEM⊕BAO; 1σ; stat-only↔stat+sys+BAO共分散）"
            # 条件分岐: 前段条件が不成立で、`float(m) == 3.0` を追加評価する。
            elif float(m) == 3.0:
                label = "誤差予算（SEM⊕BAO; 3σ; stat-only↔stat+sys+BAO共分散）"

            ax2.fill_between(z_grid, y1, y2, color="#000000", alpha=a, label=label)

    # Conservative line (single-point proxy).

    ax2.plot(z_grid, budget_cons, color="#444444", linewidth=2.0, alpha=0.9, label="誤差予算（median dmb⊕BAO; 1σ）")
    # 条件分岐: `np.isfinite(budget_cons_total).any()` を満たす経路を評価する。
    if np.isfinite(budget_cons_total).any():
        ax2.plot(
            z_grid,
            budget_cons_total,
            color="#444444",
            linestyle="--",
            linewidth=1.6,
            alpha=0.7,
            label="誤差予算（median dmb⊕BAO; 1σ; +sys(対角)）",
        )

    # 条件分岐: `bao_z.size` を満たす経路を評価する。

    if bao_z.size:
        ax2.axvline(float(np.nanmax(bao_z)), color="#333333", linewidth=1.2, alpha=0.25)
        ax2.text(
            float(np.nanmax(bao_z)) + 0.02,
            0.02,
            f"BAO点の最大z≈{_fmt_float(float(np.nanmax(bao_z)), digits=3)}",
            fontsize=9,
            color="#333333",
            alpha=0.8,
        )

    # 条件分岐: `sn_max_z is not None` を満たす経路を評価する。

    if sn_max_z is not None:
        ax2.axvline(float(sn_max_z), color="#2ca02c", linewidth=1.2, alpha=0.18)
        ax2.text(
            float(sn_max_z) + 0.02,
            0.08,
            f"SNeの有効bin最大z≈{_fmt_float(float(sn_max_z), digits=3)}（n≥{int(sn_min_points)}）",
            fontsize=9,
            color="#2ca02c",
            alpha=0.8,
        )

    ax2.set_title("静的背景P最小を“隠す”のに必要な補正 vs 誤差予算", fontsize=12)
    ax2.set_xlabel("赤方偏移 z", fontsize=11)
    ax2.set_ylabel("等価 |Δμ| [mag]", fontsize=11)
    ax2.set_xlim(0.0, float(z_max))
    ax2.set_ylim(0.0, max(0.8, float(np.nanmax(dm_req_bao) if dm_req_bao is not None else 0.8) * 1.05))
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(fontsize=8, loc="upper left")

    fig.suptitle("宇宙論（距離指標の誤差予算）：観測の不確かさと“到達限界”の接続", fontsize=14)
    fig.text(
        0.5,
        0.012,
        f"Pantheon: lcparam の dmb（統計）＋ sys共分散（系統）を使用（SNeはn≥{int(sn_min_points)}のbin）。"
        " BAO: BOSS DR12 の D_M を使用（3点; Table 8 の reduced covariance で補間誤差の目安も併記）。",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    # Summary values at z_ref.
    z_refs = [0.1, 0.5, 1.0, 2.0]
    def sample(arr: np.ndarray, zref: float) -> Optional[float]:
        # 条件分岐: `not (0.0 <= zref <= float(z_max))` を満たす経路を評価する。
        if not (0.0 <= zref <= float(z_max)):
            return None

        idx = int(np.argmin(np.abs(z_grid - float(zref))))
        v = float(arr[idx])
        return v if math.isfinite(v) else None

    budgets_at_z = []
    for zref in z_refs:
        budgets_at_z.append(
            {
                "z": float(zref),
                "sn_sem_mag": sample(sn_sem_grid, zref),
                "sn_sem_total_mag": sample(sn_sem_total_grid, zref),
                "sn_median_mag": sample(sn_med_grid, zref),
                "sn_median_total_mag": sample(sn_med_total_grid, zref),
                "bao_mu_mag_naive": sample(bao_mu_grid_naive, zref),
                "bao_mu_mag_cov": sample(bao_mu_grid_cov, zref),
                "r_drag_mu_mag": (rdrag_mu if rdrag_mu > 0 else None),
                "budget_opt_mag": sample(budget_opt, zref),
                "budget_opt_total_mag": sample(budget_opt_total, zref),
                "budget_cons_mag": sample(budget_cons, zref),
                "budget_cons_total_mag": sample(budget_cons_total, zref),
                "sn_extrapolated": (bool(sn_max_z is not None and float(zref) > float(sn_max_z) + 1e-9)),
                "bao_extrapolated": (bool(bao_max_z is not None and float(zref) > float(bao_max_z) + 1e-9)),
            }
        )

    return {
        "coverage": {
            "sn_min_points": int(sn_min_points),
            "sn_max_z_with_budget": sn_max_z,
            "bao_max_z": bao_max_z,
            "bao_covariance_used": bool(boss_dm_cov is not None),
        },
        "bao_covariance": (
            {
                "dm_z_eff": [float(z) for z in bao_z.tolist()],
                "dm_sigma_mpc": [float(x) for x in (boss_dm_sigmas.tolist() if boss_dm_sigmas is not None else [])],
                "dm_corr": (boss_dm_corr.tolist() if boss_dm_corr is not None else None),
                "dm_cov_mpc2": (boss_dm_cov.tolist() if boss_dm_cov is not None else None),
                "dm_match": boss_dm_match,
            }
            if boss_dm_cov is not None
            else None
        ),
        "z_refs": z_refs,
        "budgets_at_z": budgets_at_z,
        "reach_limits": limits,
        "reach_limits_total": limits_total,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cosmology: distance-indicator error budget (Pantheon + BOSS DR12).")
    ap.add_argument(
        "--pantheon",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "pantheon_lcparam_full_long.txt"),
        help="Pantheon lcparam file (default: data/cosmology/pantheon_lcparam_full_long.txt).",
    )
    ap.add_argument(
        "--pantheon-sys-cov",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "pantheon_sys_full_long.txt"),
        help="Pantheon systematics covariance (default: data/cosmology/pantheon_sys_full_long.txt).",
    )
    ap.add_argument(
        "--boss-ap",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "alcock_paczynski_constraints.json"),
        help="BOSS DR12 consensus JSON (default: data/cosmology/alcock_paczynski_constraints.json).",
    )
    ap.add_argument(
        "--boss-baofs-cov-cij",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "boss_dr12_baofs_consensus_reduced_covariance_cij.json"),
        help="BOSS DR12 BAO+FS reduced covariance c_ij JSON (default: data/cosmology/boss_dr12_baofs_consensus_reduced_covariance_cij.json).",
    )
    ap.add_argument(
        "--bao-rdrag",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "bao_sound_horizon_constraints.json"),
        help="BAO calibration (r_drag) constraints (default: data/cosmology/bao_sound_horizon_constraints.json).",
    )
    ap.add_argument(
        "--reach-metrics",
        type=str,
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_distance_indicator_reach_limit_metrics.json"),
        help="Reach-limit metrics JSON (default: output/private/cosmology/cosmology_distance_indicator_reach_limit_metrics.json).",
    )
    ap.add_argument(
        "--z-max",
        type=float,
        default=2.3,
        help="Max redshift for plotting (default: 2.3).",
    )
    ap.add_argument(
        "--bin-width",
        type=float,
        default=0.1,
        help="Pantheon z bin width for budgets (default: 0.1).",
    )
    ap.add_argument(
        "--sn-min-points",
        type=int,
        default=15,
        help="Minimum points per Pantheon z-bin to compute budgets (default: 15).",
    )
    ap.add_argument(
        "--sigma-multipliers",
        type=str,
        default="1,3",
        help="Sigma multipliers to report (default: '1,3').",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    z_max = float(args.z_max)
    # 条件分岐: `not (z_max > 0.0)` を満たす経路を評価する。
    if not (z_max > 0.0):
        raise ValueError("--z-max must be > 0")

    bin_width = float(args.bin_width)
    # 条件分岐: `not (bin_width > 0.0)` を満たす経路を評価する。
    if not (bin_width > 0.0):
        raise ValueError("--bin-width must be > 0")

    sn_min_points = int(args.sn_min_points)
    # 条件分岐: `sn_min_points < 1` を満たす経路を評価する。
    if sn_min_points < 1:
        raise ValueError("--sn-min-points must be >= 1")

    sigma_multipliers: List[float] = []
    for part in str(args.sigma_multipliers).split(","):
        part = part.strip()
        # 条件分岐: `not part` を満たす経路を評価する。
        if not part:
            continue

        sigma_multipliers.append(float(part))

    # 条件分岐: `not sigma_multipliers` を満たす経路を評価する。

    if not sigma_multipliers:
        sigma_multipliers = [1.0, 3.0]

    pantheon_path = Path(args.pantheon)
    pantheon_sys_path = Path(args.pantheon_sys_cov)
    boss_path = Path(args.boss_ap)
    boss_cov_path = Path(args.boss_baofs_cov_cij)
    rdrag_path = Path(args.bao_rdrag)
    reach_path = Path(args.reach_metrics)

    df = _load_pantheon_lcparam(pantheon_path)
    sys_cov: Optional[np.ndarray] = None
    # 条件分岐: `pantheon_sys_path.exists()` を満たす経路を評価する。
    if pantheon_sys_path.exists():
        sys_cov = _load_pantheon_sys_cov(pantheon_sys_path)
        # 条件分岐: `sys_cov.shape[0] != int(len(df))` を満たす経路を評価する。
        if sys_cov.shape[0] != int(len(df)):
            raise ValueError(f"Pantheon sys covariance N={sys_cov.shape[0]} does not match lcparam rows={len(df)}")

    sn_bins = _compute_sn_binned_budget(
        df,
        z_max=z_max,
        bin_width=bin_width,
        min_points=sn_min_points,
        sys_cov=sys_cov,
    )
    bao_points = _bao_mu_sigma_mag_from_boss_ap(boss_path)
    rdrag_budget = _r_drag_mu_sigma_mag(rdrag_path)
    boss_cov: Optional[Dict[str, Any]] = None
    # 条件分岐: `boss_cov_path.exists()` を満たす経路を評価する。
    if boss_cov_path.exists():
        try:
            boss_cov = _load_boss_baofs_reduced_cov_cij(boss_cov_path)
        except Exception:
            boss_cov = None

    reach_reps: Dict[str, Any] = {"reach": {}}
    # 条件分岐: `reach_path.exists()` を満たす経路を評価する。
    if reach_path.exists():
        reach_reps = _load_reach_representatives(reach_path)

    out_dir = _ROOT / "output" / "private" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "cosmology_distance_indicator_error_budget.png"
    out_json = out_dir / "cosmology_distance_indicator_error_budget_metrics.json"

    plot_metrics = _plot(
        out_png=out_png,
        reach_reps=reach_reps,
        sn_bins=sn_bins,
        bao_points=bao_points,
        boss_baofs_cov=boss_cov,
        r_drag_budget=rdrag_budget,
        z_max=z_max,
        sn_min_points=sn_min_points,
        sigma_multipliers=sigma_multipliers,
    )

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "sn_error_proxy": "Pantheon lcparam の dmb（標準化後のm_B誤差）を、距離モジュラス誤差の代理として扱う（絶対等級の定数オフセットはDDRのz依存には影響しない）。",
            "sn_binning": "z binごとに dmb の median と、逆分散重み付き平均の SEM=1/sqrt(sum(1/σ^2)) を算出（相関は無視）。",
            "sn_systematics_cov": "Pantheon 公開の系統共分散（sys_full_long）を読み込み、全共分散 C_total = diag(dmb^2) + C_sys として SEM = 1/sqrt(1^T C_total^{-1} 1) を算出（bin平均の誤差の上限側の目安）。",
            "bao_error_proxy": "BOSS DR12 consensus の D_M の相対誤差 σ(D_M)/D_M を等価Δμ=5 log10(1+σ/D) に変換。",
            "bao_covariance": "BOSS DR12 BAO+FS（consensus）の reduced covariance c_ij（Table 8）から D_M の相関を取り出し、2点線形補間 D_M(z)=w D_Mi+(1-w) D_Mj の誤差伝播で σ(D_M(z)) を推定して等価Δμへ変換（目安）。",
            "reach_required_delta_mu": "DDRの必要補正: Δμ(z)=5 log10((1+z)^{Δε_needed}) = 5 Δε_needed log10(1+z)。",
        },
        "inputs": {
            "pantheon_lcparam": str(pantheon_path).replace("\\", "/"),
            "pantheon_lcparam_sha256": _sha256(pantheon_path),
            "pantheon_source_url": "https://raw.githubusercontent.com/dscolnic/Pantheon/master/lcparam_full_long.txt",
            "pantheon_sys_cov": (str(pantheon_sys_path).replace("\\", "/") if pantheon_sys_path.exists() else None),
            "pantheon_sys_cov_sha256": (_sha256(pantheon_sys_path) if pantheon_sys_path.exists() else None),
            "pantheon_sys_cov_source_url": "https://raw.githubusercontent.com/dscolnic/Pantheon/master/sys_full_long.txt",
            "boss_dr12_consensus": str(boss_path).replace("\\", "/"),
            "boss_dr12_baofs_cov_cij": (str(boss_cov_path).replace("\\", "/") if boss_cov_path.exists() else None),
            "boss_dr12_baofs_cov_cij_sha256": (_sha256(boss_cov_path) if boss_cov_path.exists() else None),
            "bao_r_drag_constraints": str(rdrag_path).replace("\\", "/"),
            "reach_metrics": (str(reach_path).replace("\\", "/") if reach_path.exists() else None),
        },
        "reach_representatives": reach_reps,
        "sn_bins": sn_bins,
        "bao_points": bao_points,
        "r_drag_budget": rdrag_budget,
        "params": {
            "z_max": z_max,
            "bin_width": bin_width,
            "sn_min_points": sn_min_points,
            "sigma_multipliers": sigma_multipliers,
        },
        "outputs": {"png": str(out_png).replace("\\", "/"), "metrics_json": str(out_json).replace("\\", "/")},
        "summary": plot_metrics,
        "notes": [
            "これは“観測の不確かさ（目安）”であり、距離指標の定義（標準化/校正）の前提そのもの（静的無限空間での再導出）は別途の論点。",
            "Pantheon の dmb は統計誤差＋標準化の残差（実装依存）を含むため、厳密な解析の共分散モデル（相関/サンプル定義）とは一致しない可能性がある。",
            "BAO の共分散は BOSS DR12（Alam+2017）の Table 8（BAO+FS consensus）に基づく reduced covariance を使用。3点のため z 依存は単純補間の目安。",
        ],
    }
    _write_json(out_json, payload)

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_distance_indicator_error_budget",
                "argv": list(sys.argv),
                "inputs": {
                    "pantheon": pantheon_path,
                    "pantheon_sys_cov": (pantheon_sys_path if pantheon_sys_path.exists() else None),
                    "boss_ap": boss_path,
                    "boss_baofs_cov_cij": (boss_cov_path if boss_cov_path.exists() else None),
                    "bao_rdrag": rdrag_path,
                    "reach_metrics": (reach_path if reach_path.exists() else None),
                },
                "outputs": {"png": out_png, "metrics_json": out_json},
                "metrics": {
                    "z_max": z_max,
                    "bin_width": bin_width,
                    "sn_min_points": sn_min_points,
                    "pantheon_sys_cov_used": bool(pantheon_sys_path.exists()),
                },
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
