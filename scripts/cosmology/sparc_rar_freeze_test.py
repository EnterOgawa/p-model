#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparc_rar_freeze_test.py

Phase 6 / Step 6.5（SPARC：RAR）:
RAR（g_obs vs g_bar）の「fit→freeze→holdout」の最小枠組みを固定する。

目的：
- シナリオ(B)（弱場修正：g_obs=f(g_bar,P)）を主張する場合に、自由度/凍結手順/反証条件を
  機械的に固定できる I/F を先に整える。
- 現段階では P-model の弱場修正式が未確定のため、比較用 baseline として
  McGaugh+2016 の経験式（a0 1パラメータ）を同じ枠組みで評価する（P-model予測ではない）。

入力：
- output/cosmology/sparc_rar_reconstruction.csv

出力（固定）：
- output/cosmology/sparc_rar_freeze_test_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cosmology.sparc_falsification_pack import _fit_log10_a0_grid, _rar_mcgaugh2016_log10_pred  # noqa: E402

try:
    from scripts.summary import worklog  # type: ignore # noqa: E402
except Exception:  # pragma: no cover
    worklog = None

C_LIGHT_M_S = 299_792_458.0
MPC_TO_M = 3.0856775814913673e22
DEFAULT_PBG_KAPPA = 1.0 / (2.0 * math.pi)
KPC_TO_M = 3.0856775814913673e19
KM_TO_M = 1.0e3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_grid(start: float, stop: float, step: float) -> List[float]:
    if not (np.isfinite(start) and np.isfinite(stop) and np.isfinite(step) and float(step) > 0):
        raise ValueError("invalid grid params")
    if float(stop) < float(start):
        raise ValueError("stop < start")
    n = int(math.floor((float(stop) - float(start)) / float(step) + 0.5)) + 1
    vv = float(start) + float(step) * np.arange(n, dtype=float)
    vv = vv[(vv >= float(start) - 1e-12) & (vv <= float(stop) + 1e-12)]
    return [float(x) for x in vv.tolist()]


def _h0p_si_from_km_s_mpc(h0_km_s_mpc: float) -> float:
    return float(h0_km_s_mpc) * 1.0e3 / MPC_TO_M


def _get_h0p_si(
    *,
    h0p_metrics: Path,
    h0p_km_s_mpc_override: Optional[float],
) -> Tuple[float, Dict[str, Any]]:
    if h0p_km_s_mpc_override is not None and np.isfinite(h0p_km_s_mpc_override) and float(h0p_km_s_mpc_override) > 0:
        h0_si = _h0p_si_from_km_s_mpc(float(h0p_km_s_mpc_override))
        return h0_si, {"mode": "override", "H0P_km_s_Mpc": float(h0p_km_s_mpc_override), "H0P_SI_s^-1": h0_si}

    d = _read_json(h0p_metrics)
    derived = d.get("derived", {}) if isinstance(d.get("derived", {}), dict) else {}
    params = d.get("params", {}) if isinstance(d.get("params", {}), dict) else {}
    h0_si = derived.get("H0P_SI_s^-1")
    if isinstance(h0_si, (int, float)) and np.isfinite(h0_si) and float(h0_si) > 0:
        return float(h0_si), {"mode": "metrics", "path": _rel(h0p_metrics), "H0P_SI_s^-1": float(h0_si)}

    h0_km_s_mpc = params.get("H0P_km_s_Mpc")
    if isinstance(h0_km_s_mpc, (int, float)) and np.isfinite(h0_km_s_mpc) and float(h0_km_s_mpc) > 0:
        h0_si2 = _h0p_si_from_km_s_mpc(float(h0_km_s_mpc))
        return h0_si2, {"mode": "metrics_params", "path": _rel(h0p_metrics), "H0P_km_s_Mpc": float(h0_km_s_mpc), "H0P_SI_s^-1": h0_si2}

    raise RuntimeError(f"failed to read H0^(P) from metrics: {h0p_metrics}")


@dataclass(frozen=True)
class Point:
    galaxy: str
    g_bar: float
    g_obs: float
    sg_obs: float
    # Optional components for g_bar recomputation:
    # g_bar = g_gas + Υ_disk * g_disk_u + Υ_bulge * g_bul_u
    g_gas: float = float("nan")
    g_disk_u: float = float("nan")
    g_bul_u: float = float("nan")


def _read_points(csv_path: Path) -> List[Point]:
    pts: List[Point] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            gal = str(row.get("galaxy") or "").strip()
            if not gal:
                continue
            try:
                g_bar = float(row.get("g_bar_m_s2") or "nan")
                g_obs = float(row.get("g_obs_m_s2") or "nan")
                sg_obs = float(row.get("g_obs_sigma_m_s2") or "nan")
            except Exception:
                continue
            if not (np.isfinite(g_bar) and np.isfinite(g_obs)) or g_bar <= 0.0 or g_obs <= 0.0:
                continue
            # Optional components: vgas/vdisk/vbul are Rotmod components (Υ=1). g_bar can be recomputed as:
            # g_gas = vgas^2/r, g_disk_u = vdisk^2/r, g_bul_u = vbul^2/r.
            # Then g_bar(Υ)=g_gas + Υ_disk*g_disk_u + Υ_bulge*g_bul_u.
            g_gas = float("nan")
            g_disk_u = float("nan")
            g_bul_u = float("nan")
            try:
                rr_kpc = float(row.get("r_kpc") or "nan")
                vgas_km_s = float(row.get("vgas_km_s") or "nan")
                vdisk_km_s = float(row.get("vdisk_km_s") or "nan")
                vbul_km_s = float(row.get("vbul_km_s") or "nan")
                rr_m = float(rr_kpc) * KPC_TO_M
                if np.isfinite(rr_m) and rr_m > 0 and np.isfinite(vgas_km_s) and np.isfinite(vdisk_km_s) and np.isfinite(vbul_km_s):
                    vgas = float(vgas_km_s) * KM_TO_M
                    vdisk = float(vdisk_km_s) * KM_TO_M
                    vbul = float(vbul_km_s) * KM_TO_M
                    g_gas = float((vgas * vgas) / rr_m)
                    g_disk_u = float((vdisk * vdisk) / rr_m)
                    g_bul_u = float((vbul * vbul) / rr_m)
            except Exception:
                pass

            pts.append(Point(galaxy=gal, g_bar=g_bar, g_obs=g_obs, sg_obs=sg_obs, g_gas=g_gas, g_disk_u=g_disk_u, g_bul_u=g_bul_u))
    return pts


def _fit_upsilon_global_grid(
    train: Sequence[Point],
    *,
    log10_a0: float,
    sigma_floor_dex: float,
    low_accel_cut_log10_gbar: float,
    objective: str,
    upsilon_disk_grid: Sequence[float],
    upsilon_bulge_grid: Sequence[float],
) -> Dict[str, Any]:
    # Fit global (Υ_disk, Υ_bulge) on train by grid search (min chi2 with sigma_y).
    if not upsilon_disk_grid or not upsilon_bulge_grid:
        return {"status": "skipped", "reason": "empty_upsilon_grid"}

    g_gas = np.asarray([p.g_gas for p in train], dtype=float)
    g_d = np.asarray([p.g_disk_u for p in train], dtype=float)
    g_b = np.asarray([p.g_bul_u for p in train], dtype=float)
    g_obs = np.asarray([p.g_obs for p in train], dtype=float)
    sg_obs = np.asarray([p.sg_obs for p in train], dtype=float)

    # Require component-based recomputation to be available for at least 90% of points.
    m_comp = np.isfinite(g_gas) & np.isfinite(g_d) & np.isfinite(g_b)
    if int(np.count_nonzero(m_comp)) < int(0.9 * float(len(train))):
        return {"status": "skipped", "reason": "missing_components", "n_total": int(len(train)), "n_with_components": int(np.count_nonzero(m_comp))}

    g_gas = g_gas[m_comp]
    g_d = g_d[m_comp]
    g_b = g_b[m_comp]
    g_obs = g_obs[m_comp]
    sg_obs = sg_obs[m_comp]

    y = np.log10(g_obs)
    sigma_y = _sigma_log10_gobs(g_obs, sg_obs, floor_dex=float(sigma_floor_dex))
    m = np.isfinite(y) & np.isfinite(sigma_y) & (sigma_y > 0.0)
    if int(np.count_nonzero(m)) < 50:
        return {"status": "skipped", "reason": "not_enough_points", "n_used": int(np.count_nonzero(m))}

    g_gas = g_gas[m]
    g_d = g_d[m]
    g_b = g_b[m]
    y = y[m]
    sigma_y = sigma_y[m]

    ud = np.asarray([float(x) for x in upsilon_disk_grid if np.isfinite(x) and float(x) >= 0.0], dtype=float)
    ub = np.asarray([float(x) for x in upsilon_bulge_grid if np.isfinite(x) and float(x) >= 0.0], dtype=float)
    if ud.size == 0 or ub.size == 0:
        return {"status": "skipped", "reason": "no_valid_upsilon_grid"}

    # Flatten 2D grid into vectors
    udv = np.repeat(ud, ub.size)
    ubv = np.tile(ub, ud.size)
    n_combo = int(udv.size)

    # g_bar(Υ) = g_gas + Υd*g_d + Υb*g_b
    gbar = g_gas[:, None] + g_d[:, None] * udv[None, :] + g_b[:, None] * ubv[None, :]
    # guard against non-positive values
    gbar = np.where(np.isfinite(gbar) & (gbar > 0.0), gbar, np.nan)

    # y_pred for McGaugh+2016 RAR at fixed a0
    a0 = float(10.0 ** float(log10_a0))
    x = gbar / a0
    x = np.where(np.isfinite(x) & (x > 0.0), x, np.nan)
    s = np.sqrt(x)
    denom = -np.expm1(-s)
    denom = np.where(np.isfinite(denom) & (denom > 0.0), denom, np.nan)
    g_pred = gbar / denom
    y_pred = np.log10(g_pred)

    r = y[:, None] - y_pred
    # chi2 per combo
    w = 1.0 / (sigma_y * sigma_y)
    if str(objective) == "chi2_all":
        chi2 = np.nansum((r * r) * w[:, None], axis=0)
        score = chi2
        meta: Dict[str, Any] = {"objective": "chi2_all"}
    elif str(objective) == "chi2_low_accel":
        # Evaluate only low-accel points, where the candidate is expected to differ most.
        log10_gbar = np.log10(gbar)
        m_low = np.isfinite(log10_gbar) & (log10_gbar < float(low_accel_cut_log10_gbar)) & np.isfinite(r)
        n_low = np.sum(m_low, axis=0)
        # chi2_dof on low-accel points (avoid bias to small n_low)
        chi2_low = np.nansum(((r * r) * w[:, None]) * m_low, axis=0)
        chi2 = chi2_low
        dof = n_low - 1
        score = np.where(dof >= 10, chi2_low / dof, np.nan)
        meta = {"objective": "chi2_low_accel", "n_low_min_required": 11}
    elif str(objective) == "chi2_high_accel":
        # Calibrate Υ on higher-accel points (baryon-dominated), then test the low-accel regime on holdout.
        log10_gbar = np.log10(gbar)
        m_high = np.isfinite(log10_gbar) & (log10_gbar >= float(low_accel_cut_log10_gbar)) & np.isfinite(r)
        n_high = np.sum(m_high, axis=0)
        chi2_high = np.nansum(((r * r) * w[:, None]) * m_high, axis=0)
        chi2 = chi2_high
        dof = n_high - 1
        score = np.where(dof >= 10, chi2_high / dof, np.nan)
        meta = {"objective": "chi2_high_accel", "n_high_min_required": 11}
    else:
        return {"status": "skipped", "reason": f"unknown_objective:{objective}"}

    if not np.isfinite(chi2).any():
        return {"status": "failed", "reason": "chi2_all_nan", "n_used": int(y.size), "n_combo": n_combo}

    if not np.isfinite(score).any():
        return {"status": "failed", "reason": "score_all_nan", "n_used": int(y.size), "n_combo": n_combo, "objective": str(objective)}

    i_best = int(np.nanargmin(score))
    return {
        "status": "ok",
        "n_used": int(y.size),
        "n_combo": n_combo,
        "log10_a0_m_s2": float(log10_a0),
        "upsilon_disk_best": float(udv[i_best]),
        "upsilon_bulge_best": float(ubv[i_best]),
        "chi2_min": float(chi2[i_best]),
        "score_min": float(score[i_best]),
        **meta,
    }


def _sigma_log10_gobs(g_obs: np.ndarray, sg_obs: np.ndarray, *, floor_dex: float) -> np.ndarray:
    sigma_y = sg_obs / (g_obs * math.log(10.0))
    sigma_floor = float(max(floor_dex, 1e-6))
    sigma_y = np.where(np.isfinite(sigma_y) & (sigma_y > 0.0), sigma_y, np.nan)
    sigma_y = np.where(np.isfinite(sigma_y) & (sigma_y < sigma_floor), sigma_floor, sigma_y)
    return sigma_y


def _solve_sigma_int(resid: np.ndarray, sigma_y: np.ndarray, *, dof: int) -> float:
    # Find sigma_int >= 0 such that chi2/dof ~= 1 for chi2=sum(r^2/(sy^2+si^2)).
    # If already <=1 at si=0, return 0.
    m = np.isfinite(resid) & np.isfinite(sigma_y) & (sigma_y > 0.0)
    r = resid[m]
    sy = sigma_y[m]
    if r.size < 10 or dof <= 1:
        return 0.0
    chi2_0 = float(np.sum((r / sy) ** 2))
    if chi2_0 <= float(dof):
        return 0.0

    lo = 0.0
    hi = 5.0  # dex, very conservative upper bound
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        denom = np.sqrt(sy * sy + mid * mid)
        chi2 = float(np.sum((r / denom) ** 2))
        if chi2 > float(dof):
            lo = mid
        else:
            hi = mid
    return float(hi)


def _eval_model(
    name: str,
    pts: Sequence[Point],
    *,
    y_pred: np.ndarray,
    sigma_floor_dex: float,
    low_accel_cut_log10_gbar: float,
    sigma_int_dex: float = 0.0,
    min_points_per_galaxy: int = 3,
    galaxy_clipping_method: str = "none",
    galaxy_clipping_k: float = 3.5,
) -> Dict[str, Any]:
    g_bar = np.asarray([p.g_bar for p in pts], dtype=float)
    g_obs = np.asarray([p.g_obs for p in pts], dtype=float)
    sg_obs = np.asarray([p.sg_obs for p in pts], dtype=float)
    x = np.log10(g_bar)
    y = np.log10(g_obs)
    sigma_y = _sigma_log10_gobs(g_obs, sg_obs, floor_dex=float(sigma_floor_dex))
    if sigma_int_dex > 0:
        sigma_y = np.sqrt(sigma_y * sigma_y + float(sigma_int_dex) * float(sigma_int_dex))

    r = y - y_pred
    m = np.isfinite(r) & np.isfinite(sigma_y) & (sigma_y > 0.0)
    dof = int(np.count_nonzero(m)) - 1
    chi2 = float(np.sum((r[m] / sigma_y[m]) ** 2)) if dof > 0 else float("nan")
    chi2_dof = float(chi2 / float(dof)) if dof > 0 else float("nan")

    # low-accel weighted mean residual
    m_low = m & (x < float(low_accel_cut_log10_gbar))
    mean_low = float("nan")
    sem_low = float("nan")
    z_low = float("nan")
    if int(np.count_nonzero(m_low)) >= 10:
        rr = r[m_low]
        w = 1.0 / (sigma_y[m_low] ** 2)
        wsum = float(np.sum(w))
        mean_low = float(np.sum(w * rr) / wsum) if wsum > 0 else float(np.mean(rr))
        sem_low = float(math.sqrt(1.0 / wsum)) if wsum > 0 else float(np.std(rr) / math.sqrt(max(len(rr), 1)))
        z_low = mean_low / sem_low if sem_low > 0 else float("inf")

    # low-accel residual aggregated by galaxy (treat galaxies as independent units)
    galaxies = [p.galaxy for p in pts]
    by_gal: Dict[str, List[float]] = {}
    for i, gal in enumerate(galaxies):
        if not bool(m_low[i]):
            continue
        by_gal.setdefault(str(gal), []).append(float(r[i]))

    mpg = int(min_points_per_galaxy)
    if mpg <= 0:
        mpg = 1
    gal_means_all = np.asarray([float(np.mean(v)) for v in by_gal.values() if len(v) >= mpg], dtype=float)
    gal_means_all = gal_means_all[np.isfinite(gal_means_all)]

    def _gal_stats(vv: np.ndarray) -> Dict[str, Any]:
        if vv.size < 3 or not np.isfinite(vv).all():
            return {"n": int(vv.size), "mean_residual_dex": float("nan"), "sem_dex": float("nan"), "z": float("nan"), "residual_quantiles_dex": {"q16": float("nan"), "q50": float("nan"), "q84": float("nan")}}
        mean = float(np.mean(vv))
        std = float(np.std(vv, ddof=1)) if vv.size >= 2 else float("nan")
        sem = float(std / math.sqrt(float(vv.size))) if np.isfinite(std) and std > 0 else float("nan")
        z = float(mean / sem) if np.isfinite(sem) and sem > 0 else float("nan")
        return {
            "n": int(vv.size),
            "mean_residual_dex": mean,
            "sem_dex": sem,
            "z": z,
            "residual_quantiles_dex": {
                "q16": float(np.quantile(vv, 0.16)),
                "q50": float(np.quantile(vv, 0.50)),
                "q84": float(np.quantile(vv, 0.84)),
            },
        }

    raw_stats = _gal_stats(gal_means_all)

    method = str(galaxy_clipping_method or "none").strip().lower()
    clip_k = float(galaxy_clipping_k)
    used = gal_means_all
    clip_info: Dict[str, Any] = {"method": method, "status": "skipped"}
    if method == "none":
        clip_info = {"method": "none", "status": "skipped"}
    elif method == "mad":
        if not (np.isfinite(clip_k) and clip_k > 0):
            clip_info = {"method": "mad", "status": "skipped", "reason": "invalid_k", "k": float(galaxy_clipping_k)}
        elif gal_means_all.size >= 5:
            med = float(np.median(gal_means_all))
            mad = float(np.median(np.abs(gal_means_all - med)))
            rsig = float(1.4826 * mad)
            if np.isfinite(rsig) and rsig > 0:
                keep = np.abs(gal_means_all - med) <= float(clip_k) * rsig
                vv2 = gal_means_all[keep]
                if vv2.size >= 3:
                    used = vv2
                    clip_info = {
                        "method": "mad",
                        "status": "ok",
                        "k": float(clip_k),
                        "median": med,
                        "mad": mad,
                        "robust_sigma": rsig,
                        "n_total": int(gal_means_all.size),
                        "n_used": int(vv2.size),
                        "n_clipped": int(gal_means_all.size - vv2.size),
                    }
                else:
                    clip_info = {
                        "method": "mad",
                        "status": "skipped",
                        "reason": "too_few_after_clipping",
                        "k": float(clip_k),
                        "median": med,
                        "mad": mad,
                        "robust_sigma": rsig,
                        "n_total": int(gal_means_all.size),
                        "n_used": int(vv2.size),
                        "n_clipped": int(gal_means_all.size - vv2.size),
                    }
            else:
                clip_info = {"method": "mad", "status": "skipped", "reason": "robust_sigma_zero", "k": float(clip_k), "median": med, "mad": mad, "robust_sigma": rsig, "n_total": int(gal_means_all.size)}
        else:
            clip_info = {"method": "mad", "status": "skipped", "reason": "too_few_total", "k": float(clip_k), "n_total": int(gal_means_all.size)}
    else:
        clip_info = {"method": method, "status": "skipped", "reason": "unknown_method"}

    used_stats = _gal_stats(used)
    gal_mean = float(used_stats.get("mean_residual_dex") or float("nan"))
    gal_sem = float(used_stats.get("sem_dex") or float("nan"))
    gal_z = float(used_stats.get("z") or float("nan"))

    return {
        "model": name,
        "counts": {"n_points": int(len(pts)), "n_used": int(np.count_nonzero(m)), "n_low_accel": int(np.count_nonzero(m_low))},
        "sigma_int_dex": float(sigma_int_dex),
        "chi2_dof": chi2_dof,
        "residual_stats_dex": {"median": float(np.median(r[m])), "rms": float(np.sqrt(np.mean(r[m] * r[m])))},
        "low_accel": {
            "cut_log10_gbar": float(low_accel_cut_log10_gbar),
            "weighted_mean_residual_dex": mean_low,
            "sem_dex": sem_low,
            "z": z_low,
        },
        "low_accel_galaxy": {
            "cut_log10_gbar": float(low_accel_cut_log10_gbar),
            "min_points_per_galaxy": int(mpg),
            "n_galaxies_low_accel": int(used_stats.get("n") or 0),
            "n_galaxies_low_accel_total": int(raw_stats.get("n") or 0),
            "mean_residual_dex": gal_mean,
            "sem_dex": gal_sem,
            "z": gal_z,
            "residual_quantiles_dex": used_stats.get("residual_quantiles_dex"),
            "clipping": clip_info,
            "raw": {
                "n": raw_stats.get("n"),
                "mean_residual_dex": raw_stats.get("mean_residual_dex"),
                "sem_dex": raw_stats.get("sem_dex"),
                "z": raw_stats.get("z"),
                "residual_quantiles_dex": raw_stats.get("residual_quantiles_dex"),
            },
        },
    }


def _split_by_galaxy(pts: Sequence[Point], *, seed: int, train_frac: float) -> Tuple[List[Point], List[Point]]:
    galaxies = sorted({p.galaxy for p in pts})
    rng = random.Random(int(seed))
    rng.shuffle(galaxies)
    n_train = int(round(float(train_frac) * float(len(galaxies))))
    train_set = set(galaxies[:n_train])
    train = [p for p in pts if p.galaxy in train_set]
    test = [p for p in pts if p.galaxy not in train_set]
    return train, test


def _summarize_sweep(values: Sequence[float], *, threshold: float = 3.0) -> Dict[str, Any]:
    vv = np.asarray([float(x) for x in values if np.isfinite(x)], dtype=float)
    if vv.size == 0:
        return {"n": 0, "min": float("nan"), "max": float("nan"), "median": float("nan"), "p16": float("nan"), "p84": float("nan"), "pass_rate_abs_lt_threshold": float("nan")}
    pass_rate = float(np.mean(np.abs(vv) < float(threshold)))
    return {
        "n": int(vv.size),
        "min": float(np.min(vv)),
        "max": float(np.max(vv)),
        "median": float(np.median(vv)),
        "p16": float(np.quantile(vv, 0.16)),
        "p84": float(np.quantile(vv, 0.84)),
        "pass_rate_abs_lt_threshold": pass_rate,
        "threshold_abs_z": float(threshold),
    }


def _run_once(
    pts: Sequence[Point],
    *,
    seed: int,
    train_frac: float,
    h0p_metrics: Path,
    h0p_km_s_mpc_override: Optional[float],
    pbg_kappa: float,
    sigma_floor_dex: float,
    low_accel_cut_log10_gbar: float,
    min_points_per_galaxy: int = 3,
    galaxy_clipping_method: str = "none",
    galaxy_clipping_k: float = 3.5,
    models: Optional[Sequence[str]] = None,
    fit_upsilon_global: bool = False,
    upsilon_disk_grid: Optional[Sequence[float]] = None,
    upsilon_bulge_grid: Optional[Sequence[float]] = None,
    upsilon_fit_objective: str = "chi2_low_accel",
) -> Dict[str, Any]:
    train, test = _split_by_galaxy(pts, seed=int(seed), train_frac=float(train_frac))

    # Prepare arrays for fitting on train
    g_bar_tr = np.asarray([p.g_bar for p in train], dtype=float)
    g_obs_tr = np.asarray([p.g_obs for p in train], dtype=float)
    sg_obs_tr = np.asarray([p.sg_obs for p in train], dtype=float)
    y_tr = np.log10(g_obs_tr)
    sigma_y_tr = _sigma_log10_gobs(g_obs_tr, sg_obs_tr, floor_dex=float(sigma_floor_dex))

    wanted = {str(x) for x in (models or []) if str(x)}
    if not wanted:
        wanted = {
            "baryons_only",
            "baseline_rar_mcgaugh2016_fit_a0",
            "candidate_rar_pbg_a0_fixed_kappa",
            "candidate_rar_pbg_fit_kappa",
        }

    # Model 0: baryons-only (no fit)
    sigma_int_bary = float("nan")
    if "baryons_only" in wanted:
        ypred_bary_tr = np.log10(g_bar_tr)
        r_bary_tr = y_tr - ypred_bary_tr
        sigma_int_bary = _solve_sigma_int(r_bary_tr, sigma_y_tr, dof=int(np.isfinite(r_bary_tr).sum()) - 1)

    # Model 1: McGaugh+2016 empirical RAR (fit a0 on train)
    rar_fit: Dict[str, Any] = {}
    la0_best = float("nan")
    sigma_int_rar = float("nan")
    if "baseline_rar_mcgaugh2016_fit_a0" in wanted or "candidate_rar_pbg_fit_kappa" in wanted:
        rar_fit = _fit_log10_a0_grid(g_bar_tr, y_tr, sigma_y_tr)
        la0_best = float(rar_fit.get("log10_a0_best_m_s2") or float("nan"))
        ypred_rar_tr = _rar_mcgaugh2016_log10_pred(g_bar_tr, log10_a0=la0_best) if np.isfinite(la0_best) else np.full_like(y_tr, np.nan)
        r_rar_tr = y_tr - ypred_rar_tr
        sigma_int_rar = _solve_sigma_int(r_rar_tr, sigma_y_tr, dof=int(np.isfinite(r_rar_tr).sum()) - 1)

    # Candidate: tie a0 to cosmology via a0 = kappa * c * H0^(P)
    h0p_si, h0p_src = _get_h0p_si(h0p_metrics=h0p_metrics, h0p_km_s_mpc_override=h0p_km_s_mpc_override)
    kappa = float(pbg_kappa)
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("--pbg-kappa must be positive")
    a0_pbg = float(kappa) * float(C_LIGHT_M_S) * float(h0p_si)
    if not np.isfinite(a0_pbg) or a0_pbg <= 0:
        raise ValueError("invalid a0 computed from kappa*c*H0^(P)")
    la0_pbg = float(math.log10(a0_pbg))
    sigma_int_pbg = float("nan")
    if "candidate_rar_pbg_a0_fixed_kappa" in wanted or bool(fit_upsilon_global):
        ypred_pbg_tr = _rar_mcgaugh2016_log10_pred(g_bar_tr, log10_a0=la0_pbg)
        r_pbg_tr = y_tr - ypred_pbg_tr
        sigma_int_pbg = _solve_sigma_int(r_pbg_tr, sigma_y_tr, dof=int(np.isfinite(r_pbg_tr).sum()) - 1)

    # Candidate: fit kappa on train (1 parameter), then freeze and evaluate on holdout.
    # Note: since a0 = kappa*c*H0^(P) is one-to-one, fitting kappa is equivalent to fitting a0.
    # We express the fitted a0 as kappa for bookkeeping.
    kappa_fit: Dict[str, Any] = {"status": "skipped"}
    la0_kfit = float("nan")
    sigma_int_kfit = float("nan")
    if "candidate_rar_pbg_fit_kappa" in wanted and np.isfinite(la0_best):
        denom = float(C_LIGHT_M_S) * float(h0p_si)
        if np.isfinite(denom) and denom > 0:
            a0_best = float(10.0 ** la0_best)
            kappa_best = float(a0_best / denom) if denom > 0 else float("nan")
            la0_kfit = float(math.log10(a0_best)) if a0_best > 0 else float("nan")
            sigma_int_kfit = float(sigma_int_rar)
            kappa_fit = {
                "status": "ok",
                "H0P_source": h0p_src,
                "kappa_best": float(kappa_best),
                "log10_kappa_best": float(math.log10(kappa_best)) if np.isfinite(kappa_best) and kappa_best > 0 else float("nan"),
                "a0_best_m_s2": float(a0_best),
                "log10_a0_best_m_s2": float(la0_best),
                "note": "Equivalent to baseline a0 fit, expressed as kappa=a0/(c H0^(P)).",
            }

    # Evaluate on train/test
    out_models: List[Dict[str, Any]] = []
    for label, ypred_fn, sig_int in [
        ("baryons_only", lambda gb: np.log10(gb), sigma_int_bary),
        ("baseline_rar_mcgaugh2016_fit_a0", lambda gb: _rar_mcgaugh2016_log10_pred(gb, log10_a0=la0_best), sigma_int_rar),
        ("candidate_rar_pbg_a0_fixed_kappa", lambda gb: _rar_mcgaugh2016_log10_pred(gb, log10_a0=la0_pbg), sigma_int_pbg),
        ("candidate_rar_pbg_fit_kappa", lambda gb: _rar_mcgaugh2016_log10_pred(gb, log10_a0=la0_kfit), sigma_int_kfit),
    ]:
        if label not in wanted:
            continue
        # train
        gb_tr = np.asarray([p.g_bar for p in train], dtype=float)
        ypred_tr = ypred_fn(gb_tr)
        tr_raw = _eval_model(
            label,
            train,
            y_pred=ypred_tr,
            sigma_floor_dex=float(sigma_floor_dex),
            low_accel_cut_log10_gbar=float(low_accel_cut_log10_gbar),
            sigma_int_dex=0.0,
            min_points_per_galaxy=int(min_points_per_galaxy),
            galaxy_clipping_method=str(galaxy_clipping_method),
            galaxy_clipping_k=float(galaxy_clipping_k),
        )
        tr_int = _eval_model(
            label,
            train,
            y_pred=ypred_tr,
            sigma_floor_dex=float(sigma_floor_dex),
            low_accel_cut_log10_gbar=float(low_accel_cut_log10_gbar),
            sigma_int_dex=float(sig_int),
            min_points_per_galaxy=int(min_points_per_galaxy),
            galaxy_clipping_method=str(galaxy_clipping_method),
            galaxy_clipping_k=float(galaxy_clipping_k),
        )

        # test
        gb_te = np.asarray([p.g_bar for p in test], dtype=float)
        ypred_te = ypred_fn(gb_te)
        te_raw = _eval_model(
            label,
            test,
            y_pred=ypred_te,
            sigma_floor_dex=float(sigma_floor_dex),
            low_accel_cut_log10_gbar=float(low_accel_cut_log10_gbar),
            sigma_int_dex=0.0,
            min_points_per_galaxy=int(min_points_per_galaxy),
            galaxy_clipping_method=str(galaxy_clipping_method),
            galaxy_clipping_k=float(galaxy_clipping_k),
        )
        te_int = _eval_model(
            label,
            test,
            y_pred=ypred_te,
            sigma_floor_dex=float(sigma_floor_dex),
            low_accel_cut_log10_gbar=float(low_accel_cut_log10_gbar),
            sigma_int_dex=float(sig_int),
            min_points_per_galaxy=int(min_points_per_galaxy),
            galaxy_clipping_method=str(galaxy_clipping_method),
            galaxy_clipping_k=float(galaxy_clipping_k),
        )

        out_models.append(
            {
                "name": label,
                "fit": {
                    "sigma_int_dex": float(sig_int),
                    "rar_fit": rar_fit if label.startswith("baseline_rar") else {},
                    "pbg_a0": (
                        {
                            "H0P_source": h0p_src,
                            "kappa": float(kappa),
                            "a0_m_s2": float(a0_pbg),
                            "log10_a0_m_s2": float(la0_pbg),
                        }
                        if label == "candidate_rar_pbg_a0_fixed_kappa"
                        else {}
                    ),
                    "pbg_kappa_fit": (kappa_fit if label == "candidate_rar_pbg_fit_kappa" else {}),
                },
                "train": {"raw": tr_raw, "with_sigma_int": tr_int},
                "test": {"raw": te_raw, "with_sigma_int": te_int},
            }
        )

    # Optional: fit global Υ_disk/Υ_bulge on train (nuisance) and evaluate candidate on holdout.
    if bool(fit_upsilon_global):
        ud_grid = list(upsilon_disk_grid or [])
        ub_grid = list(upsilon_bulge_grid or [])
        ups_fit = _fit_upsilon_global_grid(
            train,
            log10_a0=float(la0_pbg),
            sigma_floor_dex=float(sigma_floor_dex),
            low_accel_cut_log10_gbar=float(low_accel_cut_log10_gbar),
            objective=str(upsilon_fit_objective),
            upsilon_disk_grid=ud_grid,
            upsilon_bulge_grid=ub_grid,
        )
        if ups_fit.get("status") == "ok":
            ud_best = float(ups_fit["upsilon_disk_best"])
            ub_best = float(ups_fit["upsilon_bulge_best"])

            def _gbar_from_u(p: Point) -> float:
                if np.isfinite(p.g_gas) and np.isfinite(p.g_disk_u) and np.isfinite(p.g_bul_u):
                    gb = float(p.g_gas + ud_best * p.g_disk_u + ub_best * p.g_bul_u)
                    return gb
                return float(p.g_bar)

            train_u = [Point(galaxy=p.galaxy, g_bar=_gbar_from_u(p), g_obs=p.g_obs, sg_obs=p.sg_obs) for p in train]
            test_u = [Point(galaxy=p.galaxy, g_bar=_gbar_from_u(p), g_obs=p.g_obs, sg_obs=p.sg_obs) for p in test]

            # sigma_int on train under this nuisance fit
            gb_tr_u = np.asarray([p.g_bar for p in train_u], dtype=float)
            go_tr_u = np.asarray([p.g_obs for p in train_u], dtype=float)
            sgo_tr_u = np.asarray([p.sg_obs for p in train_u], dtype=float)
            y_tr_u = np.log10(go_tr_u)
            sy_tr_u = _sigma_log10_gobs(go_tr_u, sgo_tr_u, floor_dex=float(sigma_floor_dex))
            ypred_tr_u = _rar_mcgaugh2016_log10_pred(gb_tr_u, log10_a0=float(la0_pbg))
            r_tr_u = y_tr_u - ypred_tr_u
            sigma_int_u = _solve_sigma_int(r_tr_u, sy_tr_u, dof=int(np.isfinite(r_tr_u).sum()) - 1)

            # Evaluate train/test
            tr_raw = _eval_model(
                "candidate_rar_pbg_a0_fixed_kappa_fit_upsilon_global",
                train_u,
                y_pred=ypred_tr_u,
                sigma_floor_dex=float(sigma_floor_dex),
                low_accel_cut_log10_gbar=float(low_accel_cut_log10_gbar),
                sigma_int_dex=0.0,
                min_points_per_galaxy=int(min_points_per_galaxy),
                galaxy_clipping_method=str(galaxy_clipping_method),
                galaxy_clipping_k=float(galaxy_clipping_k),
            )
            tr_int = _eval_model(
                "candidate_rar_pbg_a0_fixed_kappa_fit_upsilon_global",
                train_u,
                y_pred=ypred_tr_u,
                sigma_floor_dex=float(sigma_floor_dex),
                low_accel_cut_log10_gbar=float(low_accel_cut_log10_gbar),
                sigma_int_dex=float(sigma_int_u),
                min_points_per_galaxy=int(min_points_per_galaxy),
                galaxy_clipping_method=str(galaxy_clipping_method),
                galaxy_clipping_k=float(galaxy_clipping_k),
            )

            gb_te_u = np.asarray([p.g_bar for p in test_u], dtype=float)
            ypred_te_u = _rar_mcgaugh2016_log10_pred(gb_te_u, log10_a0=float(la0_pbg))
            te_raw = _eval_model(
                "candidate_rar_pbg_a0_fixed_kappa_fit_upsilon_global",
                test_u,
                y_pred=ypred_te_u,
                sigma_floor_dex=float(sigma_floor_dex),
                low_accel_cut_log10_gbar=float(low_accel_cut_log10_gbar),
                sigma_int_dex=0.0,
                min_points_per_galaxy=int(min_points_per_galaxy),
                galaxy_clipping_method=str(galaxy_clipping_method),
                galaxy_clipping_k=float(galaxy_clipping_k),
            )
            te_int = _eval_model(
                "candidate_rar_pbg_a0_fixed_kappa_fit_upsilon_global",
                test_u,
                y_pred=ypred_te_u,
                sigma_floor_dex=float(sigma_floor_dex),
                low_accel_cut_log10_gbar=float(low_accel_cut_log10_gbar),
                sigma_int_dex=float(sigma_int_u),
                min_points_per_galaxy=int(min_points_per_galaxy),
                galaxy_clipping_method=str(galaxy_clipping_method),
                galaxy_clipping_k=float(galaxy_clipping_k),
            )

            out_models.append(
                {
                    "name": "candidate_rar_pbg_a0_fixed_kappa_fit_upsilon_global",
                    "fit": {
                        "sigma_int_dex": float(sigma_int_u),
                        "pbg_a0": {
                            "H0P_source": h0p_src,
                            "kappa": float(kappa),
                            "a0_m_s2": float(a0_pbg),
                            "log10_a0_m_s2": float(la0_pbg),
                        },
                        "upsilon_fit": {
                            "status": "ok",
                            "upsilon_disk_best": float(ud_best),
                            "upsilon_bulge_best": float(ub_best),
                            "grid": {"upsilon_disk": ud_grid, "upsilon_bulge": ub_grid},
                            "chi2_min": float(ups_fit.get("chi2_min") or float("nan")),
                            "objective": str(ups_fit.get("objective") or ""),
                            "score_min": float(ups_fit.get("score_min") or float("nan")),
                            "n_used": int(ups_fit.get("n_used") or 0),
                            "n_combo": int(ups_fit.get("n_combo") or 0),
                            "note": "Fit Υ on train only (nuisance), then freeze (Υ, a0) and evaluate on holdout.",
                        },
                    },
                    "train": {"raw": tr_raw, "with_sigma_int": tr_int},
                    "test": {"raw": te_raw, "with_sigma_int": te_int},
                }
            )
        else:
            out_models.append(
                {
                    "name": "candidate_rar_pbg_a0_fixed_kappa_fit_upsilon_global",
                    "fit": {"upsilon_fit": ups_fit},
                    "train": {},
                    "test": {},
                }
            )

    return {
        "seed": int(seed),
        "train_frac": float(train_frac),
        "counts": {"n_points_total": int(len(pts)), "n_points_train": int(len(train)), "n_points_test": int(len(test))},
        "models": out_models,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rar-csv",
        default=str(_ROOT / "output" / "cosmology" / "sparc_rar_reconstruction.csv"),
        help="RAR reconstruction CSV (default: output/cosmology/sparc_rar_reconstruction.csv)",
    )
    p.add_argument(
        "--h0p-metrics",
        default=str(_ROOT / "output" / "cosmology" / "cosmology_redshift_pbg_metrics.json"),
        help="Path to cosmology_redshift_pbg_metrics.json (default: output/cosmology/cosmology_redshift_pbg_metrics.json)",
    )
    p.add_argument("--h0p-km-s-mpc", type=float, default=None, help="Override H0^(P) in km/s/Mpc (optional)")
    p.add_argument("--pbg-kappa", type=float, default=DEFAULT_PBG_KAPPA, help="a0 = kappa * c * H0^(P) (default: 1/(2π))")
    p.add_argument("--out", default=str(_ROOT / "output" / "cosmology" / "sparc_rar_freeze_test_metrics.json"), help="Output JSON path")
    p.add_argument("--sigma-floor-dex", type=float, default=0.01, help="Floor for sigma(log10 g_obs) in dex")
    p.add_argument("--low-accel-cut", type=float, default=-10.5, help="Low-acceleration cut on log10(g_bar)")
    p.add_argument(
        "--min-points-per-galaxy",
        type=int,
        default=3,
        help="Minimum radial points per galaxy in the low-accel galaxy-level aggregation (default: 3)",
    )
    p.add_argument(
        "--galaxy-clipping-method",
        choices=["none", "mad"],
        default="none",
        help="Optional clipping method for low-accel galaxy means (default: none)",
    )
    p.add_argument("--galaxy-clipping-k", type=float, default=3.5, help="Clipping threshold k (used when method=mad; default: 3.5)")
    p.add_argument("--fit-upsilon-global", action="store_true", help="Fit global (Υ_disk,Υ_bulge) on train for the candidate (nuisance), then freeze and evaluate on holdout.")
    p.add_argument(
        "--upsilon-fit-objective",
        choices=["chi2_all", "chi2_low_accel", "chi2_high_accel"],
        default="chi2_low_accel",
        help="Objective for Υ fit on train (default: chi2_low_accel)",
    )
    p.add_argument("--upsilon-disk-start", type=float, default=0.3, help="Υ_disk grid start (default: 0.3)")
    p.add_argument("--upsilon-disk-stop", type=float, default=0.7, help="Υ_disk grid stop (default: 0.7)")
    p.add_argument("--upsilon-disk-step", type=float, default=0.05, help="Υ_disk grid step (default: 0.05)")
    p.add_argument("--upsilon-bulge-start", type=float, default=0.5, help="Υ_bulge grid start (default: 0.5)")
    p.add_argument("--upsilon-bulge-stop", type=float, default=0.9, help="Υ_bulge grid stop (default: 0.9)")
    p.add_argument("--upsilon-bulge-step", type=float, default=0.05, help="Υ_bulge grid step (default: 0.05)")
    p.add_argument("--train-frac", type=float, default=0.7, help="Training fraction by galaxy (default: 0.7)")
    p.add_argument("--train-fracs", action="append", type=float, default=[], help="Train fractions to sweep (repeatable). Overrides --train-frac.")
    p.add_argument("--train-frac-start", type=float, default=None, help="Train fraction sweep start (optional)")
    p.add_argument("--train-frac-stop", type=float, default=None, help="Train fraction sweep stop (optional)")
    p.add_argument("--train-frac-step", type=float, default=None, help="Train fraction sweep step (optional)")
    p.add_argument("--seed", type=int, default=20260129, help="Random seed for galaxy split")
    p.add_argument("--seeds", action="append", type=int, default=[], help="Seeds to sweep (repeatable). Overrides --seed.")
    p.add_argument("--seed-start", type=int, default=None, help="Seed sweep start (optional)")
    p.add_argument("--seed-count", type=int, default=None, help="Seed sweep count (optional)")
    args = p.parse_args(list(argv) if argv is not None else None)

    rar_csv = Path(args.rar_csv)
    if not rar_csv.exists():
        raise FileNotFoundError(f"missing rar csv: {rar_csv}")

    pts = _read_points(rar_csv)
    if not pts:
        raise RuntimeError("no valid points in rar csv")

    if args.seeds:
        seeds = [int(x) for x in args.seeds]
    elif args.seed_start is not None or args.seed_count is not None:
        if args.seed_start is None or args.seed_count is None:
            raise ValueError("--seed-start and --seed-count must be provided together")
        if int(args.seed_count) <= 0:
            raise ValueError("--seed-count must be positive")
        seeds = [int(args.seed_start) + i for i in range(int(args.seed_count))]
    else:
        seeds = [int(args.seed)]

    if args.train_fracs:
        train_fracs = [float(x) for x in args.train_fracs]
    elif args.train_frac_start is not None or args.train_frac_stop is not None or args.train_frac_step is not None:
        if args.train_frac_start is None or args.train_frac_stop is None or args.train_frac_step is None:
            raise ValueError("--train-frac-start/--train-frac-stop/--train-frac-step must be provided together")
        start = float(args.train_frac_start)
        stop = float(args.train_frac_stop)
        step = float(args.train_frac_step)
        if not (np.isfinite(start) and np.isfinite(stop) and np.isfinite(step)) or step <= 0:
            raise ValueError("invalid train-frac sweep parameters")
        if stop < start:
            raise ValueError("--train-frac-stop must be >= --train-frac-start")
        # Generate inclusive grid, rounding to avoid float accumulation.
        n = int(math.floor((stop - start) / step + 1.0 + 1e-12))
        train_fracs = [float(round(start + i * step, 12)) for i in range(n)]
    else:
        train_fracs = [float(args.train_frac)]

    train_fracs = [float(x) for x in train_fracs if np.isfinite(x)]
    for tf in train_fracs:
        if not (0.1 <= float(tf) <= 0.95):
            raise ValueError("--train-frac(s) must be in [0.1,0.95]")

    runs: List[Dict[str, Any]] = []
    args_upsilon_fit_objective = str(args.upsilon_fit_objective)
    ups_disk_grid = _parse_grid(float(args.upsilon_disk_start), float(args.upsilon_disk_stop), float(args.upsilon_disk_step))
    ups_bul_grid = _parse_grid(float(args.upsilon_bulge_start), float(args.upsilon_bulge_stop), float(args.upsilon_bulge_step))
    for tf in train_fracs:
        for sd in seeds:
            runs.append(
                _run_once(
                    pts,
                    seed=int(sd),
                    train_frac=float(tf),
                    h0p_metrics=Path(args.h0p_metrics),
                    h0p_km_s_mpc_override=args.h0p_km_s_mpc,
                    pbg_kappa=float(args.pbg_kappa),
                    sigma_floor_dex=float(args.sigma_floor_dex),
                    low_accel_cut_log10_gbar=float(args.low_accel_cut),
                    min_points_per_galaxy=int(args.min_points_per_galaxy),
                    galaxy_clipping_method=str(args.galaxy_clipping_method),
                    galaxy_clipping_k=float(args.galaxy_clipping_k),
                    fit_upsilon_global=bool(args.fit_upsilon_global),
                    upsilon_fit_objective=str(args_upsilon_fit_objective),
                    upsilon_disk_grid=ups_disk_grid,
                    upsilon_bulge_grid=ups_bul_grid,
                )
            )

    # Representative run for backward-compat fields: prefer (seed, train_frac) = (--seed, --train-frac) if included in sweep.
    rep = None
    want_seed = int(args.seed)
    want_tf = float(args.train_frac)
    for r in runs:
        if not isinstance(r, dict):
            continue
        if int(r.get("seed") or -1) == want_seed and np.isfinite(float(r.get("train_frac") or float("nan"))) and abs(float(r.get("train_frac")) - want_tf) < 1e-12:
            rep = r
            break
    if rep is None:
        rep = runs[0] if runs else {"seed": want_seed, "train_frac": want_tf, "counts": {}, "models": []}

    # Sweep summary (holdout low-accel z, with sigma_int)
    # - sweep_summary: point-level (radial points treated as independent)
    # - sweep_summary_galaxy: galaxy-level (galaxies treated as independent units)
    by_model: Dict[str, List[float]] = {}
    by_model_galaxy: Dict[str, List[float]] = {}
    for run in runs:
        for m in run.get("models", []) if isinstance(run.get("models"), list) else []:
            if not isinstance(m, dict):
                continue
            name = str(m.get("name") or "")
            z = (((m.get("test") or {}).get("with_sigma_int") or {}).get("low_accel") or {}).get("z")
            if isinstance(z, (int, float)) and np.isfinite(z):
                by_model.setdefault(name, []).append(float(z))
            z_gal = (((m.get("test") or {}).get("with_sigma_int") or {}).get("low_accel_galaxy") or {}).get("z")
            if isinstance(z_gal, (int, float)) and np.isfinite(z_gal):
                by_model_galaxy.setdefault(name, []).append(float(z_gal))

    sweep_summary = {k: _summarize_sweep(v) for k, v in sorted(by_model.items())}
    sweep_summary_galaxy = {k: _summarize_sweep(v) for k, v in sorted(by_model_galaxy.items())}

    out = Path(args.out)
    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "inputs": {
            "rar_csv": _rel(rar_csv),
            "h0p_metrics": _rel(Path(args.h0p_metrics)),
            "h0p_km_s_mpc": float(args.h0p_km_s_mpc) if args.h0p_km_s_mpc is not None else None,
            "pbg_kappa": float(args.pbg_kappa),
            "seeds": [int(x) for x in seeds],
            "train_fracs": [float(x) for x in train_fracs],
            "sigma_floor_dex": float(args.sigma_floor_dex),
            "low_accel_cut_log10_gbar": float(args.low_accel_cut),
            "min_points_per_galaxy": int(args.min_points_per_galaxy),
            "galaxy_clipping": {"method": str(args.galaxy_clipping_method), "k": float(args.galaxy_clipping_k)},
            "fit_upsilon_global": bool(args.fit_upsilon_global),
            "upsilon_fit_objective": str(args_upsilon_fit_objective),
            "upsilon_grid": {"disk": ups_disk_grid, "bulge": ups_bul_grid},
            "note": "Split is by galaxy (not by radial points) to reduce leakage; baselines/candidates here are not treated as established P-model predictions.",
        },
        "counts": rep.get("counts", {}) if isinstance(rep.get("counts"), dict) else {},
        "models": rep.get("models", []) if isinstance(rep.get("models"), list) else [],
        "runs": runs,
        "sweep_summary": sweep_summary,
        "sweep_summary_galaxy": sweep_summary_galaxy,
        "outputs": {"metrics": _rel(out)},
    }
    _write_json(out, payload)

    if worklog is not None:
        try:
            worklog.append_event("cosmology.sparc_rar_freeze_test", {"metrics": _rel(out)})
        except Exception:
            pass

    print(json.dumps({"metrics": _rel(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
