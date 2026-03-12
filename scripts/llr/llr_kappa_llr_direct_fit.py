#!/usr/bin/env python3
"""
llr_kappa_llr_direct_fit.py

Step 8.7.47 (LLR kappa direct fit):
- 8.7.47.1: Build a beta=1 normalized solar-Shapiro template from LLR batch points.
- 8.7.47.2: Fit kappa_LLR directly with nuisance terms (station/reflector/year trend).
- 8.7.47.3: Evaluate |z| gate against kappa_LLR=1 and map beta := kappa_LLR.
- 8.7.47.4: Audit year/station consistency by weighted-mean and chi2/dof.
- 8.7.47.10: Integrate template-contamination split (Shapiro + orthogonal nuisance simultaneous fit).

Input:
- output/private/llr/batch/llr_batch_points.csv

Outputs (default: output/private/llr, and synced to output/public/llr):
- llr_kappa_llr_template_points.csv
- llr_kappa_llr_fit_mode_summary.csv
- llr_kappa_llr_fit_selected_points.csv
- llr_kappa_llr_year_consistency.csv
- llr_kappa_llr_station_consistency.csv
- llr_kappa_llr_imbalance_policy_summary.csv
- llr_kappa_llr_station_stratified_refit.csv
- llr_kappa_llr_target_stratified_refit.csv
- llr_kappa_llr_template_nulltest_summary.csv
- llr_kappa_llr_template_decontamination_summary.csv
- llr_kappa_llr_template_decontamination_projection.csv
- llr_kappa_llr_metrics.json
- llr_kappa_llr_fit.pdf
- llr_kappa_llr_fit.png
- llr_kappa_llr_imbalance_audit.pdf
- llr_kappa_llr_imbalance_audit.png
- llr_kappa_llr_template_nulltest.pdf
- llr_kappa_llr_template_nulltest.png
- llr_kappa_llr_template_decontamination.pdf
- llr_kappa_llr_template_decontamination.png
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_ROOT = Path(__file__).resolve().parents[2]


# 関数: `_safe_rel` の入出力契約と処理意図を定義する。
def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


# 関数: `_status_from_abs_z` の入出力契約と処理意図を定義する。

def _status_from_abs_z(abs_z: float) -> str:
    if not np.isfinite(abs_z):
        return "reject"

    if abs_z <= 2.0:
        return "pass"

    if abs_z <= 3.0:
        return "watch"

    return "reject"


# 関数: `_consistency_status_from_chi2_dof` の入出力契約と処理意図を定義する。

def _consistency_status_from_chi2_dof(chi2_dof: float) -> str:
    if not np.isfinite(chi2_dof):
        return "reject"

    if chi2_dof <= 2.0:
        return "pass"

    if chi2_dof <= 5.0:
        return "watch"

    return "reject"


# 関数: `_combine_gate_status` の入出力契約と処理意図を定義する。

def _combine_gate_status(statuses: Sequence[str]) -> str:
    norm = [str(s or "").strip().lower() for s in statuses if str(s or "").strip()]
    if not norm:
        return "reject"

    if any(s == "reject" for s in norm):
        return "reject"

    if all(s == "pass" for s in norm):
        return "pass"

    return "watch"


# クラス: `FitResult` の責務と境界条件を定義する。

@dataclass
class FitResult:
    mode: str
    n_points: int
    n_params: int
    dof: int
    rss: float
    rmse_ns: float
    aic_like: float
    bic_like: float
    delta_kappa: float
    kappa_est: float
    kappa_sigma: float
    z_value: float
    abs_z: float
    status: str
    y_hat: np.ndarray
    fit_residual: np.ndarray


# 関数: `_parse_epoch_utc_series` の入出力契約と処理意図を定義する。

def _parse_epoch_utc_series(values: pd.Series) -> pd.Series:
    """Parse mixed timestamp strings robustly into UTC-aware datetimes.

    pandas can fail on mixed second/fractional-second formats when format is not explicit.
    We therefore attempt strict parsers first (ISO8601/mixed) and only fallback row-wise
    on unresolved entries.
    """
    s = values.astype(str)
    parsed = pd.to_datetime(s, utc=True, errors="coerce")
    if parsed.notna().all():
        return parsed

    for fmt in ("ISO8601", "mixed"):
        try:
            parsed_try = pd.to_datetime(s, utc=True, errors="coerce", format=fmt)
        except TypeError:
            # Older pandas may not support format="mixed"/"ISO8601".
            continue

        if parsed_try.notna().sum() > parsed.notna().sum():
            parsed = parsed_try

        if parsed.notna().all():
            return parsed

    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = s.loc[mask].apply(lambda x: pd.to_datetime(x, utc=True, errors="coerce"))

    return parsed


# 関数: `_read_points` の入出力契約と処理意図を定義する。

def _read_points(points_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(points_csv)
    required = [
        "epoch_utc",
        "station",
        "target",
        "inlier_best",
        "residual_sr_tropo_tide_ns",
        "dt_sun_shapiro_ns",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns in {points_csv}: {missing}")

    df["epoch_utc"] = _parse_epoch_utc_series(df["epoch_utc"])
    df["station"] = df["station"].astype(str)
    df["target"] = df["target"].astype(str)
    df["inlier_best"] = df["inlier_best"].astype(bool)
    df["residual_sr_tropo_tide_ns"] = pd.to_numeric(df["residual_sr_tropo_tide_ns"], errors="coerce")
    df["dt_sun_shapiro_ns"] = pd.to_numeric(df["dt_sun_shapiro_ns"], errors="coerce")
    df["year"] = df["epoch_utc"].dt.year.astype("Int64")

    year_float = (
        df["epoch_utc"].dt.year
        + (df["epoch_utc"].dt.dayofyear - 1.0) / 365.25
        + df["epoch_utc"].dt.hour / (24.0 * 365.25)
    )
    df["year_float"] = pd.to_numeric(year_float, errors="coerce")

    ok = (
        df["inlier_best"].astype(bool)
        & df["epoch_utc"].notna()
        & np.isfinite(df["residual_sr_tropo_tide_ns"].to_numpy(dtype=float))
        & np.isfinite(df["dt_sun_shapiro_ns"].to_numpy(dtype=float))
    )
    df = df.loc[ok].copy().reset_index(drop=True)
    return df


# 関数: `_one_hot_drop_first` の入出力契約と処理意図を定義する。

def _one_hot_drop_first(values: Sequence[str], prefix: str) -> Tuple[np.ndarray, List[str]]:
    cats = sorted(set(str(v) for v in values))
    if not cats:
        return np.zeros((len(values), 0), dtype=float), []

    use = cats[1:]
    if not use:
        return np.zeros((len(values), 0), dtype=float), []

    arr = np.zeros((len(values), len(use)), dtype=float)
    idx_map = {c: i for i, c in enumerate(use)}
    for r, v in enumerate(values):
        j = idx_map.get(str(v))
        if j is not None:
            arr[r, j] = 1.0

    names = [f"{prefix}_{c}" for c in use]
    return arr, names


# 関数: `_build_design_matrix` の入出力契約と処理意図を定義する。

def _build_design_matrix(df: pd.DataFrame, mode: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    y = df["residual_sr_tropo_tide_ns"].to_numpy(dtype=float)
    x_template = df["dt_sun_shapiro_ns"].to_numpy(dtype=float)
    year = df["year_float"].to_numpy(dtype=float)
    year_centered = year - float(np.nanmean(year))

    station = df["station"].astype(str).tolist()
    target = df["target"].astype(str).tolist()
    group = [f"{s}|{t}" for s, t in zip(station, target)]

    cols: List[np.ndarray] = []
    names: List[str] = []

    # Always fit delta_kappa against the beta=1 solar-Shapiro template.
    cols.append(x_template)
    names.append("delta_kappa_slope")

    # Global intercept.
    cols.append(np.ones_like(x_template))
    names.append("intercept")

    if mode == "none":
        pass
    elif mode == "station":
        oh, oh_names = _one_hot_drop_first(station, prefix="st")
        for j in range(oh.shape[1]):
            cols.append(oh[:, j])
            names.append(oh_names[j])
    elif mode == "station_target":
        oh, oh_names = _one_hot_drop_first(group, prefix="stt")
        for j in range(oh.shape[1]):
            cols.append(oh[:, j])
            names.append(oh_names[j])
    elif mode == "station_target_year":
        oh_g, oh_g_names = _one_hot_drop_first(group, prefix="stt")
        for j in range(oh_g.shape[1]):
            cols.append(oh_g[:, j])
            names.append(oh_g_names[j])

        # Station-specific linear year trends.

        stations_unique = sorted(set(station))
        st_index = {s: i for i, s in enumerate(stations_unique)}
        trend = np.zeros((len(station), len(stations_unique)), dtype=float)
        for i, s in enumerate(station):
            trend[i, st_index[s]] = year_centered[i]

        # Drop first station trend to avoid rank issues with intercept/group dummies.

        if trend.shape[1] > 1:
            trend = trend[:, 1:]
            for j in range(trend.shape[1]):
                cols.append(trend[:, j])
                names.append(f"trend_st_{stations_unique[j + 1]}")
    else:
        raise ValueError(f"unknown mode: {mode}")

    x = np.column_stack(cols)
    return x, y, names


# 関数: `_normalize_sample_weight` の入出力契約と処理意図を定義する。

def _normalize_sample_weight(sample_weight: Optional[np.ndarray], n_rows: int) -> Optional[np.ndarray]:
    if sample_weight is None:
        return None

    w = np.asarray(sample_weight, dtype=float).reshape(-1)
    if len(w) != int(n_rows):
        raise ValueError(f"sample_weight length mismatch: len(w)={len(w)} n_rows={n_rows}")

    w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
    ok = np.isfinite(w)
    if not np.any(ok):
        raise ValueError("all sample weights are invalid")

    mean_w = float(np.nanmean(w[ok]))
    if not np.isfinite(mean_w) or mean_w <= 0:
        raise ValueError("invalid sample weight mean")

    w = np.where(ok, w / mean_w, 1.0)
    return w


# 関数: `_fit_ols` の入出力契約と処理意図を定義する。

def _fit_ols(mode: str, x: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> FitResult:
    w = _normalize_sample_weight(sample_weight, n_rows=len(y))
    if w is None:
        x_fit = x
        y_fit = y
    else:
        sw = np.sqrt(w)
        x_fit = x * sw[:, None]
        y_fit = y * sw

    beta_hat, _, _, _ = np.linalg.lstsq(x_fit, y_fit, rcond=None)
    y_hat = x @ beta_hat
    fit_residual = y - y_hat
    if w is None:
        rss = float(np.sum(fit_residual * fit_residual))
        n_eff = float(len(y))
    else:
        rss = float(np.sum(w * fit_residual * fit_residual))
        n_eff = float(np.sum(w))

    n = int(len(y))
    p = int(x.shape[1])
    dof = int(max(n - p, 1))
    rmse = float(math.sqrt(rss / max(n_eff, 1.0)))
    sigma2 = float(rss / dof)

    xtx = x_fit.T @ x_fit
    cov = sigma2 * np.linalg.pinv(xtx)
    delta = float(beta_hat[0])
    delta_sigma = float(math.sqrt(max(cov[0, 0], 0.0)))
    kappa = 1.0 + delta
    z = float(delta / delta_sigma) if delta_sigma > 0 and np.isfinite(delta_sigma) else float("nan")
    abs_z = float(abs(z)) if np.isfinite(z) else float("nan")

    aic = float(n * math.log(max(rss / max(n_eff, 1.0), 1e-30)) + 2.0 * p)
    bic = float(n * math.log(max(rss / max(n_eff, 1.0), 1e-30)) + p * math.log(max(n, 2)))
    status = _status_from_abs_z(abs_z)

    return FitResult(
        mode=mode,
        n_points=n,
        n_params=p,
        dof=dof,
        rss=rss,
        rmse_ns=rmse,
        aic_like=aic,
        bic_like=bic,
        delta_kappa=delta,
        kappa_est=kappa,
        kappa_sigma=delta_sigma,
        z_value=z,
        abs_z=abs_z,
        status=status,
        y_hat=y_hat,
        fit_residual=fit_residual,
    )


# 関数: `_pick_best_mode` の入出力契約と処理意図を定義する。

def _pick_best_mode(rows: Sequence[FitResult]) -> FitResult:
    valid = [r for r in rows if np.isfinite(r.aic_like)]
    if valid:
        valid.sort(key=lambda r: (r.aic_like, r.rmse_ns))
        return valid[0]

    fallback = sorted(rows, key=lambda r: r.rmse_ns)
    return fallback[0]


# 関数: `_fit_subset_kappa` の入出力契約と処理意図を定義する。

def _fit_subset_kappa(
    df_sub: pd.DataFrame,
    mode: str,
    sample_weight: Optional[np.ndarray] = None,
) -> Optional[FitResult]:
    if df_sub.empty:
        return None

    try:
        x, y, _ = _build_design_matrix(df_sub, mode=mode)
        return _fit_ols(mode=mode, x=x, y=y, sample_weight=sample_weight)
    except Exception:
        return None


# 関数: `_weighted_mean_and_chi2` の入出力契約と処理意図を定義する。

def _weighted_mean_and_chi2(values: np.ndarray, sigma: np.ndarray) -> Dict[str, float]:
    ok = np.isfinite(values) & np.isfinite(sigma) & (sigma > 0)
    v = values[ok]
    s = sigma[ok]
    if len(v) < 2:
        return {
            "n_valid": int(len(v)),
            "weighted_mean": float("nan"),
            "weighted_sigma": float("nan"),
            "chi2": float("nan"),
            "dof": float("nan"),
            "chi2_dof": float("nan"),
        }

    w = 1.0 / (s * s)
    mu = float(np.sum(w * v) / np.sum(w))
    sig = float(math.sqrt(1.0 / np.sum(w)))
    chi2 = float(np.sum(((v - mu) / s) ** 2))
    dof = float(len(v) - 1)
    chi2_dof = float(chi2 / max(dof, 1.0))
    return {
        "n_valid": int(len(v)),
        "weighted_mean": mu,
        "weighted_sigma": sig,
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": chi2_dof,
    }


# 関数: `_make_consistency_tables` の入出力契約と処理意図を定義する。

def _make_consistency_tables(
    df: pd.DataFrame,
    year_fit_mode: str,
    station_fit_mode: str,
    min_points_year: int,
    min_points_station: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    year_rows: List[Dict[str, Any]] = []
    station_rows: List[Dict[str, Any]] = []

    year_vals = sorted(set(int(v) for v in df["year"].dropna().astype(int).tolist()))
    for y in year_vals:
        sub = df[df["year"].astype("Int64") == int(y)].copy()
        n = int(len(sub))
        if n < int(min_points_year):
            year_rows.append(
                {
                    "year": int(y),
                    "n_points": n,
                    "fit_ok": False,
                    "reason": f"n<{int(min_points_year)}",
                    "kappa_est": float("nan"),
                    "kappa_sigma": float("nan"),
                    "abs_z": float("nan"),
                    "status": "reject",
                }
            )
            continue

        fr = _fit_subset_kappa(sub, mode=year_fit_mode)
        if fr is None:
            year_rows.append(
                {
                    "year": int(y),
                    "n_points": n,
                    "fit_ok": False,
                    "reason": "fit_failed",
                    "kappa_est": float("nan"),
                    "kappa_sigma": float("nan"),
                    "abs_z": float("nan"),
                    "status": "reject",
                }
            )
            continue

        year_rows.append(
            {
                "year": int(y),
                "n_points": n,
                "fit_ok": True,
                "reason": "",
                "kappa_est": fr.kappa_est,
                "kappa_sigma": fr.kappa_sigma,
                "abs_z": fr.abs_z,
                "status": fr.status,
            }
        )

    station_vals = sorted(set(str(v) for v in df["station"].astype(str).tolist()))
    for st in station_vals:
        sub = df[df["station"].astype(str) == st].copy()
        n = int(len(sub))
        if n < int(min_points_station):
            station_rows.append(
                {
                    "station": st,
                    "n_points": n,
                    "fit_ok": False,
                    "reason": f"n<{int(min_points_station)}",
                    "kappa_est": float("nan"),
                    "kappa_sigma": float("nan"),
                    "abs_z": float("nan"),
                    "status": "reject",
                }
            )
            continue

        fr = _fit_subset_kappa(sub, mode=station_fit_mode)
        if fr is None:
            station_rows.append(
                {
                    "station": st,
                    "n_points": n,
                    "fit_ok": False,
                    "reason": "fit_failed",
                    "kappa_est": float("nan"),
                    "kappa_sigma": float("nan"),
                    "abs_z": float("nan"),
                    "status": "reject",
                }
            )
            continue

        station_rows.append(
            {
                "station": st,
                "n_points": n,
                "fit_ok": True,
                "reason": "",
                "kappa_est": fr.kappa_est,
                "kappa_sigma": fr.kappa_sigma,
                "abs_z": fr.abs_z,
                "status": fr.status,
            }
        )

    year_df = pd.DataFrame(year_rows).sort_values(["year"]).reset_index(drop=True)
    station_df = pd.DataFrame(station_rows).sort_values(["station"]).reset_index(drop=True)

    year_stats = _weighted_mean_and_chi2(
        pd.to_numeric(year_df.get("kappa_est"), errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(year_df.get("kappa_sigma"), errors="coerce").to_numpy(dtype=float),
    )
    station_stats = _weighted_mean_and_chi2(
        pd.to_numeric(station_df.get("kappa_est"), errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(station_df.get("kappa_sigma"), errors="coerce").to_numpy(dtype=float),
    )

    year_status = _consistency_status_from_chi2_dof(float(year_stats.get("chi2_dof", float("nan"))))
    station_status = _consistency_status_from_chi2_dof(float(station_stats.get("chi2_dof", float("nan"))))

    consistency = {
        "year_fit_mode": year_fit_mode,
        "station_fit_mode": station_fit_mode,
        "year": {**year_stats, "status": year_status},
        "station": {**station_stats, "status": station_status},
    }
    return year_df, station_df, consistency


# 関数: `_make_weight_from_inverse_counts` の入出力契約と処理意図を定義する。

def _make_weight_from_inverse_counts(keys: Sequence[str], min_count_floor: int = 1) -> np.ndarray:
    s = pd.Series([str(v) for v in keys], dtype="string")
    counts = s.value_counts(dropna=False).to_dict()
    floor = float(max(int(min_count_floor), 1))
    w = np.array([1.0 / max(float(counts.get(str(v), 1.0)), floor) for v in s.tolist()], dtype=float)
    return w


# 関数: `_build_imbalance_weight` の入出力契約と処理意図を定義する。

def _build_imbalance_weight(
    df: pd.DataFrame,
    scheme: str,
    floor_station: int,
    floor_target: int,
    floor_station_target: int,
    max_weight_cap: float,
) -> np.ndarray:
    station = df["station"].astype(str).tolist()
    target = df["target"].astype(str).tolist()
    station_target = [f"{s}|{t}" for s, t in zip(station, target)]

    if scheme == "uniform":
        w = np.ones(len(df), dtype=float)
    elif scheme == "inv_station":
        w = _make_weight_from_inverse_counts(station, min_count_floor=floor_station)
    elif scheme == "inv_target":
        w = _make_weight_from_inverse_counts(target, min_count_floor=floor_target)
    elif scheme == "inv_station_target":
        w = _make_weight_from_inverse_counts(station_target, min_count_floor=floor_station_target)
    elif scheme == "station_cap_p95":
        st_counts = pd.Series(station, dtype="string").value_counts(dropna=False)
        cap = float(np.percentile(st_counts.to_numpy(dtype=float), 95))
        cap = max(cap, 1.0)
        m = st_counts.to_dict()
        w = np.array([min(1.0, cap / max(float(m.get(str(s), 1.0)), 1.0)) for s in station], dtype=float)
    else:
        raise ValueError(f"unknown imbalance weight scheme: {scheme}")

    w_norm = _normalize_sample_weight(w, n_rows=len(df))
    if w_norm is None:
        return np.ones(len(df), dtype=float)

    cap = float(max_weight_cap)
    if np.isfinite(cap) and cap > 0:
        w_norm = np.minimum(w_norm, cap)

    return w_norm


# 関数: `_summarize_policy_robustness` の入出力契約と処理意図を定義する。

def _summarize_policy_robustness(policy_df: pd.DataFrame) -> Dict[str, Any]:
    if policy_df.empty:
        return {
            "n_valid_policies": 0,
            "kappa_min": float("nan"),
            "kappa_max": float("nan"),
            "kappa_span": float("nan"),
            "kappa_median": float("nan"),
            "max_abs_z_vs_uniform": float("nan"),
            "status": "reject",
        }

    valid = policy_df[np.isfinite(pd.to_numeric(policy_df.get("kappa_est"), errors="coerce"))].copy()
    if valid.empty:
        return {
            "n_valid_policies": 0,
            "kappa_min": float("nan"),
            "kappa_max": float("nan"),
            "kappa_span": float("nan"),
            "kappa_median": float("nan"),
            "max_abs_z_vs_uniform": float("nan"),
            "status": "reject",
        }

    k = pd.to_numeric(valid["kappa_est"], errors="coerce").to_numpy(dtype=float)
    k_sigma = pd.to_numeric(valid["kappa_sigma"], errors="coerce").to_numpy(dtype=float)
    k_min = float(np.nanmin(k))
    k_max = float(np.nanmax(k))
    k_span = float(k_max - k_min)
    k_med = float(np.nanmedian(k))

    if "uniform" in set(valid["weight_scheme"].astype(str).tolist()):
        ref = valid[valid["weight_scheme"].astype(str) == "uniform"].iloc[0]
    else:
        ref = valid.iloc[0]

    ref_k = float(ref["kappa_est"])
    ref_s = float(ref["kappa_sigma"])
    z_vals: List[float] = []
    for row in valid.to_dict(orient="records"):
        kk = float(row.get("kappa_est", float("nan")))
        ss = float(row.get("kappa_sigma", float("nan")))
        denom = math.sqrt(max(ref_s * ref_s + ss * ss, 1e-30))
        if np.isfinite(kk) and np.isfinite(denom) and denom > 0:
            z_vals.append(abs((kk - ref_k) / denom))

    max_abs_z = float(np.nanmax(z_vals)) if z_vals else float("nan")
    status = _status_from_abs_z(max_abs_z)

    return {
        "n_valid_policies": int(len(valid)),
        "kappa_min": k_min,
        "kappa_max": k_max,
        "kappa_span": k_span,
        "kappa_median": k_med,
        "max_abs_z_vs_uniform": max_abs_z,
        "status": status,
    }


# 関数: `_build_group_consistency_table` の入出力契約と処理意図を定義する。

def _build_group_consistency_table(
    df: pd.DataFrame,
    group_col: str,
    fit_mode: str,
    min_points: int,
    sample_weight: Optional[np.ndarray],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    vals = sorted(set(str(v) for v in df[group_col].astype(str).tolist()))
    for g in vals:
        mask = df[group_col].astype(str) == g
        sub = df.loc[mask].copy()
        n = int(len(sub))
        if n < int(min_points):
            rows.append(
                {
                    group_col: g,
                    "n_points": n,
                    "fit_ok": False,
                    "reason": f"n<{int(min_points)}",
                    "kappa_est": float("nan"),
                    "kappa_sigma": float("nan"),
                    "abs_z": float("nan"),
                    "status": "reject",
                }
            )
            continue

        if sample_weight is None:
            w_sub = None
        else:
            idx = np.flatnonzero(mask.to_numpy(dtype=bool))
            w_sub = np.asarray(sample_weight, dtype=float)[idx]

        fr = _fit_subset_kappa(sub, mode=fit_mode, sample_weight=w_sub)
        if fr is None:
            rows.append(
                {
                    group_col: g,
                    "n_points": n,
                    "fit_ok": False,
                    "reason": "fit_failed",
                    "kappa_est": float("nan"),
                    "kappa_sigma": float("nan"),
                    "abs_z": float("nan"),
                    "status": "reject",
                }
            )
            continue

        rows.append(
            {
                group_col: g,
                "n_points": n,
                "fit_ok": True,
                "reason": "",
                "kappa_est": fr.kappa_est,
                "kappa_sigma": fr.kappa_sigma,
                "abs_z": fr.abs_z,
                "status": fr.status,
            }
        )

    out_df = pd.DataFrame(rows).sort_values([group_col]).reset_index(drop=True)
    stats = _weighted_mean_and_chi2(
        pd.to_numeric(out_df.get("kappa_est"), errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(out_df.get("kappa_sigma"), errors="coerce").to_numpy(dtype=float),
    )
    stats["status"] = _consistency_status_from_chi2_dof(float(stats.get("chi2_dof", float("nan"))))
    stats["fit_mode"] = fit_mode
    stats["min_points"] = int(min_points)
    return out_df, stats


# 関数: `_run_imbalance_stratified_audit` の入出力契約と処理意図を定義する。

def _run_imbalance_stratified_audit(
    df: pd.DataFrame,
    imbalance_fit_mode: str,
    imbalance_schemes: Sequence[str],
    station_fit_mode: str,
    target_fit_mode: str,
    min_points_station: int,
    min_points_target: int,
    stratified_weight_scheme: str,
    floor_station: int,
    floor_target: int,
    floor_station_target: int,
    max_weight_cap: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    x_base, y_base, _ = _build_design_matrix(df, mode=imbalance_fit_mode)
    scheme_weights: Dict[str, np.ndarray] = {}
    policy_rows: List[Dict[str, Any]] = []
    for scheme in imbalance_schemes:
        w = _build_imbalance_weight(
            df,
            scheme=scheme,
            floor_station=floor_station,
            floor_target=floor_target,
            floor_station_target=floor_station_target,
            max_weight_cap=max_weight_cap,
        )
        scheme_weights[scheme] = w
        fr = _fit_ols(mode=f"{imbalance_fit_mode}|{scheme}", x=x_base, y=y_base, sample_weight=w)
        policy_rows.append(
            {
                "weight_scheme": scheme,
                "fit_mode": imbalance_fit_mode,
                "n_points": fr.n_points,
                "n_params": fr.n_params,
                "dof": fr.dof,
                "rmse_ns": fr.rmse_ns,
                "aic_like": fr.aic_like,
                "bic_like": fr.bic_like,
                "kappa_est": fr.kappa_est,
                "kappa_sigma": fr.kappa_sigma,
                "delta_kappa": fr.delta_kappa,
                "z_value": fr.z_value,
                "abs_z": fr.abs_z,
                "status": fr.status,
                "weight_min": float(np.nanmin(w)),
                "weight_max": float(np.nanmax(w)),
            }
        )

    policy_df = pd.DataFrame(policy_rows).sort_values(["aic_like", "rmse_ns"]).reset_index(drop=True)
    robustness = _summarize_policy_robustness(policy_df)

    w_strat = scheme_weights.get(stratified_weight_scheme)
    if w_strat is None:
        w_strat = _build_imbalance_weight(
            df,
            scheme="uniform",
            floor_station=floor_station,
            floor_target=floor_target,
            floor_station_target=floor_station_target,
            max_weight_cap=max_weight_cap,
        )

    station_df, station_stats = _build_group_consistency_table(
        df=df,
        group_col="station",
        fit_mode=station_fit_mode,
        min_points=min_points_station,
        sample_weight=w_strat,
    )
    target_df, target_stats = _build_group_consistency_table(
        df=df,
        group_col="target",
        fit_mode=target_fit_mode,
        min_points=min_points_target,
        sample_weight=w_strat,
    )

    summary = {
        "imbalance_fit_mode": imbalance_fit_mode,
        "imbalance_schemes": [str(s) for s in imbalance_schemes],
        "stratified_weight_scheme": stratified_weight_scheme,
        "weight_floor": {
            "station": int(floor_station),
            "target": int(floor_target),
            "station_target": int(floor_station_target),
        },
        "max_weight_cap": float(max_weight_cap),
        "robustness_envelope": robustness,
        "station_stratified": station_stats,
        "target_stratified": target_stats,
    }
    return policy_df, station_df, target_df, summary


# 関数: `_run_template_null_test` の入出力契約と処理意図を定義する。

def _run_template_null_test(
    df: pd.DataFrame,
    fit_mode: str,
    scales: Sequence[float],
    sample_weight: Optional[np.ndarray],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scale in scales:
        d = df.copy()
        d["dt_sun_shapiro_ns"] = pd.to_numeric(d["dt_sun_shapiro_ns"], errors="coerce") * float(scale)
        x, y, _ = _build_design_matrix(d, mode=fit_mode)
        fr = _fit_ols(mode=f"{fit_mode}|scale={scale:+.3f}", x=x, y=y, sample_weight=sample_weight)
        rows.append(
            {
                "template_scale": float(scale),
                "fit_mode": fit_mode,
                "kappa_est": fr.kappa_est,
                "kappa_sigma": fr.kappa_sigma,
                "delta_kappa": fr.delta_kappa,
                "z_value": fr.z_value,
                "abs_z": fr.abs_z,
                "status": fr.status,
                "rmse_ns": fr.rmse_ns,
                "aic_like": fr.aic_like,
                "bic_like": fr.bic_like,
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["template_scale"]).reset_index(drop=True)
    if out_df.empty:
        return out_df, {"status": "reject"}

    if np.any(np.isclose(pd.to_numeric(out_df["template_scale"], errors="coerce").to_numpy(dtype=float), 1.0)):
        ref = out_df[np.isclose(pd.to_numeric(out_df["template_scale"], errors="coerce").to_numpy(dtype=float), 1.0)].iloc[0]
    else:
        ref = out_df.iloc[0]

    ref_k = float(ref.get("kappa_est", float("nan")))
    ref_s = float(ref.get("kappa_sigma", float("nan")))
    z_shift: List[float] = []
    for row in out_df.to_dict(orient="records"):
        kk = float(row.get("kappa_est", float("nan")))
        ss = float(row.get("kappa_sigma", float("nan")))
        denom = math.sqrt(max(ref_s * ref_s + ss * ss, 1e-30))
        if np.isfinite(kk) and np.isfinite(denom) and denom > 0:
            z_shift.append(abs((kk - ref_k) / denom))

    max_abs_z = float(np.nanmax(z_shift)) if z_shift else float("nan")
    status = _status_from_abs_z(max_abs_z)
    summary = {
        "fit_mode": fit_mode,
        "scales": [float(v) for v in out_df["template_scale"].tolist()],
        "ref_scale": float(ref.get("template_scale", float("nan"))),
        "ref_kappa_est": ref_k,
        "ref_kappa_sigma": ref_s,
        "max_abs_z_vs_ref": max_abs_z,
        "status": status,
    }
    return out_df, summary


# 関数: `_weighted_corr` の入出力契約と処理意図を定義する。

def _weighted_corr(x: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray]) -> float:
    xx = np.asarray(x, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    if len(xx) != len(yy):
        return float("nan")

    ok = np.isfinite(xx) & np.isfinite(yy)
    if sample_weight is None:
        ww = np.ones(len(xx), dtype=float)
    else:
        ww = np.asarray(sample_weight, dtype=float).reshape(-1)
        if len(ww) != len(xx):
            return float("nan")

        ok = ok & np.isfinite(ww) & (ww > 0)

    if int(np.sum(ok)) < 3:
        return float("nan")

    xx = xx[ok]
    yy = yy[ok]
    ww = ww[ok]
    ww_sum = float(np.sum(ww))
    if not np.isfinite(ww_sum) or ww_sum <= 0:
        return float("nan")

    cov = float(np.sum(ww * xx * yy) / ww_sum)
    vx = float(np.sum(ww * xx * xx) / ww_sum)
    vy = float(np.sum(ww * yy * yy) / ww_sum)
    if vx <= 0 or vy <= 0:
        return float("nan")

    return float(cov / math.sqrt(vx * vy))


# 関数: `_orthogonalize_against_template` の入出力契約と処理意図を定義する。

def _orthogonalize_against_template(
    x: np.ndarray,
    names: Sequence[str],
    sample_weight: Optional[np.ndarray],
    min_std: float,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    if x.ndim != 2 or x.shape[1] < 2:
        return x, [str(v) for v in names], pd.DataFrame()

    n_rows = int(x.shape[0])
    w = _normalize_sample_weight(sample_weight, n_rows=n_rows)
    if w is None:
        w = np.ones(n_rows, dtype=float)

    t = np.asarray(x[:, 0], dtype=float)
    t_w = w * t
    t_norm2 = float(np.sum(t_w * t))
    if not np.isfinite(t_norm2) or t_norm2 <= 0:
        return x, [str(v) for v in names], pd.DataFrame()

    cols: List[np.ndarray] = [t.copy()]
    out_names: List[str] = [str(names[0]) if names else "delta_kappa_slope"]
    rows: List[Dict[str, Any]] = []
    for j in range(1, int(x.shape[1])):
        name = str(names[j]) if j < len(names) else f"nuis_{j}"
        col = np.asarray(x[:, j], dtype=float)
        corr_before = _weighted_corr(col, t, w)
        proj_coef = float(np.sum((w * col) * t) / t_norm2)
        col_orth = col - (proj_coef * t)
        corr_after = _weighted_corr(col_orth, t, w)
        std_after = float(np.nanstd(col_orth))
        keep = bool(np.isfinite(std_after) and (std_after >= float(min_std)))
        if keep:
            cols.append(col_orth)
            out_names.append(f"{name}__orth")

        rows.append(
            {
                "column_name": name,
                "projection_coef_on_template": proj_coef,
                "corr_before": corr_before,
                "corr_after": corr_after,
                "std_after": std_after,
                "kept": keep,
            }
        )

    x_orth = np.column_stack(cols)
    proj_df = pd.DataFrame(rows).sort_values(["kept", "column_name"], ascending=[False, True]).reset_index(drop=True)
    return x_orth, out_names, proj_df


# 関数: `_run_template_decontamination_audit` の入出力契約と処理意図を定義する。

def _run_template_decontamination_audit(
    df: pd.DataFrame,
    fit_mode: str,
    sample_weight: Optional[np.ndarray],
    min_std: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    x_base, y, names = _build_design_matrix(df, mode=fit_mode)
    fr_base = _fit_ols(mode=f"{fit_mode}|baseline", x=x_base, y=y, sample_weight=sample_weight)
    x_orth, names_orth, proj_df = _orthogonalize_against_template(
        x=x_base,
        names=names,
        sample_weight=sample_weight,
        min_std=float(min_std),
    )
    fr_decont = _fit_ols(mode=f"{fit_mode}|decontaminated", x=x_orth, y=y, sample_weight=sample_weight)

    shift = float(fr_decont.kappa_est - fr_base.kappa_est)
    denom = math.sqrt(max((fr_base.kappa_sigma ** 2) + (fr_decont.kappa_sigma ** 2), 1e-30))
    abs_z_shift = float(abs(shift) / denom) if denom > 0 and np.isfinite(denom) else float("nan")
    shift_status = _status_from_abs_z(abs_z_shift)

    kappa_gate_status = _status_from_abs_z(fr_decont.abs_z)
    if proj_df.empty:
        max_abs_corr_before = float("nan")
        max_abs_corr_after = float("nan")
        kept_count = 0
        dropped_count = 0
    else:
        c_before = pd.to_numeric(proj_df.get("corr_before"), errors="coerce").to_numpy(dtype=float)
        c_after = pd.to_numeric(proj_df.get("corr_after"), errors="coerce").to_numpy(dtype=float)
        keep_series = proj_df.get("kept", pd.Series(dtype=bool)).astype(bool)
        max_abs_corr_before = float(np.nanmax(np.abs(c_before))) if len(c_before) > 0 else float("nan")
        max_abs_corr_after = float(np.nanmax(np.abs(c_after))) if len(c_after) > 0 else float("nan")
        kept_count = int(np.sum(keep_series))
        dropped_count = int(len(keep_series) - kept_count)

    summary = {
        "fit_mode": fit_mode,
        "baseline_mode": fr_base.mode,
        "decontaminated_mode": fr_decont.mode,
        "baseline_kappa_est": fr_base.kappa_est,
        "baseline_kappa_sigma": fr_base.kappa_sigma,
        "baseline_abs_z": fr_base.abs_z,
        "decontaminated_kappa_est": fr_decont.kappa_est,
        "decontaminated_kappa_sigma": fr_decont.kappa_sigma,
        "decontaminated_abs_z": fr_decont.abs_z,
        "kappa_minus_1_status": kappa_gate_status,
        "kappa_shift_decont_minus_base": shift,
        "abs_z_shift": abs_z_shift,
        "status": shift_status,
        "nuisance_input_count": int(max(x_base.shape[1] - 1, 0)),
        "nuisance_orth_kept_count": kept_count,
        "nuisance_orth_dropped_count": dropped_count,
        "max_abs_corr_before": max_abs_corr_before,
        "max_abs_corr_after": max_abs_corr_after,
        "min_orth_std": float(min_std),
        "orth_design_n_params": int(len(names_orth)),
    }
    return proj_df, summary


# 関数: `_write_template_null_plot` の入出力契約と処理意図を定義する。

def _write_template_null_plot(
    null_df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    x = pd.to_numeric(null_df.get("template_scale"), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(null_df.get("kappa_est"), errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(null_df.get("kappa_sigma"), errors="coerce").to_numpy(dtype=float)
    ax.errorbar(x, y, yerr=e, fmt="o-", color="#9467bd", ecolor="#9467bd", capsize=4, linewidth=1.6)
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.2)
    ax.axvline(1.0, color="#777777", linestyle=":", linewidth=1.0)
    ax.set_xlabel("template scale")
    ax.set_ylabel("kappa_LLR")
    ax.set_title("Template null/sign test")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# 関数: `_write_template_decontamination_plot` の入出力契約と処理意図を定義する。

def _write_template_decontamination_plot(
    summary: Dict[str, Any],
    proj_df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.6), height_ratios=[1.0, 1.1])
    ax0 = axes[0]
    labels = ["baseline", "decontaminated"]
    y = np.array(
        [
            float(summary.get("baseline_kappa_est", float("nan"))),
            float(summary.get("decontaminated_kappa_est", float("nan"))),
        ],
        dtype=float,
    )
    e = np.array(
        [
            float(summary.get("baseline_kappa_sigma", float("nan"))),
            float(summary.get("decontaminated_kappa_sigma", float("nan"))),
        ],
        dtype=float,
    )
    x = np.arange(len(labels), dtype=float)
    ax0.errorbar(x, y, yerr=e, fmt="o", color="#1f77b4", ecolor="#1f77b4", capsize=4)
    ax0.axhline(1.0, color="#444444", linestyle="--", linewidth=1.2)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylabel("kappa_LLR")
    ax0.set_title("Template contamination split: kappa comparison")
    ax0.grid(alpha=0.25)
    shift = float(summary.get("kappa_shift_decont_minus_base", float("nan")))
    abs_z_shift = float(summary.get("abs_z_shift", float("nan")))
    ax0.text(
        0.02,
        0.02,
        f"delta_kappa={shift:+.6f}, |z_shift|={abs_z_shift:.3f}",
        transform=ax0.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.0,
    )

    ax1 = axes[1]
    if proj_df.empty:
        ax1.text(0.5, 0.5, "no nuisance columns", transform=ax1.transAxes, ha="center", va="center")
        ax1.set_axis_off()
    else:
        p = proj_df.copy()
        p = p[p.get("kept", pd.Series(dtype=bool)).astype(bool)].copy()
        p["abs_corr_before"] = np.abs(pd.to_numeric(p.get("corr_before"), errors="coerce"))
        p["abs_corr_after"] = np.abs(pd.to_numeric(p.get("corr_after"), errors="coerce"))
        p = p.sort_values(["abs_corr_before"], ascending=[False]).head(12).copy()
        idx = np.arange(len(p), dtype=float)
        b = pd.to_numeric(p.get("abs_corr_before"), errors="coerce").to_numpy(dtype=float)
        a = pd.to_numeric(p.get("abs_corr_after"), errors="coerce").to_numpy(dtype=float)
        labels_p = p.get("column_name", pd.Series(dtype=str)).astype(str).tolist()
        ax1.barh(idx - 0.17, b, height=0.32, color="#ff7f0e", alpha=0.75, label="|corr| before")
        ax1.barh(idx + 0.17, a, height=0.32, color="#2ca02c", alpha=0.75, label="|corr| after")
        ax1.set_yticks(idx)
        ax1.set_yticklabels(labels_p)
        ax1.invert_yaxis()
        ax1.set_xlabel("weighted |corr(template, nuisance)|")
        ax1.set_title("Top nuisance-template correlations (kept columns)")
        ax1.grid(axis="x", alpha=0.22)
        ax1.legend(loc="lower right")

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# 関数: `_write_plot` の入出力契約と処理意図を定義する。

def _write_plot(
    mode_df: pd.DataFrame,
    selected_points_df: pd.DataFrame,
    year_df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
) -> None:
    fig = plt.figure(figsize=(13.8, 12.2))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.2, 1.0], hspace=0.38)

    ax0 = fig.add_subplot(grid[0, 0])
    x0 = np.arange(len(mode_df), dtype=float)
    y0 = pd.to_numeric(mode_df["kappa_est"], errors="coerce").to_numpy(dtype=float)
    e0 = pd.to_numeric(mode_df["kappa_sigma"], errors="coerce").to_numpy(dtype=float)
    ax0.errorbar(x0, y0, yerr=e0, fmt="o", color="#1f77b4", ecolor="#1f77b4", capsize=4)
    ax0.axhline(1.0, color="#444444", linestyle="--", linewidth=1.2)
    ax0.set_xticks(x0)
    ax0.set_xticklabels(mode_df["mode"].astype(str).tolist(), rotation=20, ha="right")
    ax0.set_ylabel("kappa_LLR")
    ax0.set_title("kappa_LLR fit by nuisance mode")
    ax0.grid(alpha=0.25)

    ax1 = fig.add_subplot(grid[1, 0])
    xx = pd.to_numeric(selected_points_df["template_dt_sun_shapiro_ns"], errors="coerce").to_numpy(dtype=float)
    yy = pd.to_numeric(selected_points_df["residual_sr_tropo_tide_ns"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[ok]
    yy = yy[ok]
    hb = ax1.hexbin(xx, yy, gridsize=55, mincnt=1, cmap="viridis")
    cb = fig.colorbar(hb, ax=ax1, pad=0.012)
    cb.set_label("count")
    if len(xx) >= 2:
        slope, intercept = np.polyfit(xx, yy, 1)
        x_line = np.linspace(float(np.nanmin(xx)), float(np.nanmax(xx)), 200)
        y_line = slope * x_line + intercept
        ax1.plot(x_line, y_line, color="#ff7f0e", linewidth=2.0, label=f"OLS slope={slope:.4f}")
        ax1.legend(loc="upper left")

    ax1.set_xlabel("template dt_sun_shapiro_ns (beta=1 normalized)")
    ax1.set_ylabel("residual_sr_tropo_tide_ns")
    ax1.set_title("LLR residual vs solar-Shapiro template")
    ax1.grid(alpha=0.20)

    ax2 = fig.add_subplot(grid[2, 0])
    y_year = year_df[year_df.get("fit_ok").astype(bool)].copy()
    if not y_year.empty:
        xs = pd.to_numeric(y_year["year"], errors="coerce").to_numpy(dtype=float)
        ys = pd.to_numeric(y_year["kappa_est"], errors="coerce").to_numpy(dtype=float)
        es = pd.to_numeric(y_year["kappa_sigma"], errors="coerce").to_numpy(dtype=float)
        ax2.errorbar(xs, ys, yerr=es, fmt="o", color="#2ca02c", ecolor="#2ca02c", capsize=3)
        ax2.plot(xs, ys, color="#2ca02c", alpha=0.65, linewidth=1.2)

    ax2.axhline(1.0, color="#444444", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("year")
    ax2.set_ylabel("kappa_LLR")
    ax2.set_title("Year consistency (independent yearly fits)")
    ax2.grid(alpha=0.25)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# 関数: `_write_imbalance_plot` の入出力契約と処理意図を定義する。

def _write_imbalance_plot(
    policy_df: pd.DataFrame,
    station_df: pd.DataFrame,
    target_df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14.2, 13.0))

    ax0 = axes[0]
    x0 = np.arange(len(policy_df), dtype=float)
    y0 = pd.to_numeric(policy_df.get("kappa_est"), errors="coerce").to_numpy(dtype=float)
    e0 = pd.to_numeric(policy_df.get("kappa_sigma"), errors="coerce").to_numpy(dtype=float)
    labels0 = policy_df.get("weight_scheme", pd.Series(dtype="string")).astype(str).tolist()
    ax0.errorbar(x0, y0, yerr=e0, fmt="o", color="#1f77b4", ecolor="#1f77b4", capsize=4)
    ax0.axhline(1.0, color="#444444", linestyle="--", linewidth=1.2)
    ax0.set_xticks(x0)
    ax0.set_xticklabels(labels0, rotation=25, ha="right")
    ax0.set_ylabel("kappa_LLR")
    ax0.set_title("Imbalance weighting audit")
    ax0.grid(alpha=0.25)

    ax1 = axes[1]
    st_ok = station_df[station_df.get("fit_ok").astype(bool)].copy()
    if not st_ok.empty:
        x1 = np.arange(len(st_ok), dtype=float)
        y1 = pd.to_numeric(st_ok.get("kappa_est"), errors="coerce").to_numpy(dtype=float)
        e1 = pd.to_numeric(st_ok.get("kappa_sigma"), errors="coerce").to_numpy(dtype=float)
        l1 = st_ok.get("station", pd.Series(dtype="string")).astype(str).tolist()
        ax1.errorbar(x1, y1, yerr=e1, fmt="o", color="#ff7f0e", ecolor="#ff7f0e", capsize=4)
        ax1.set_xticks(x1)
        ax1.set_xticklabels(l1, rotation=20, ha="right")

    ax1.axhline(1.0, color="#444444", linestyle="--", linewidth=1.2)
    ax1.set_ylabel("kappa_LLR")
    ax1.set_title("Station-stratified refit")
    ax1.grid(alpha=0.25)

    ax2 = axes[2]
    tg_ok = target_df[target_df.get("fit_ok").astype(bool)].copy()
    if not tg_ok.empty:
        x2 = np.arange(len(tg_ok), dtype=float)
        y2 = pd.to_numeric(tg_ok.get("kappa_est"), errors="coerce").to_numpy(dtype=float)
        e2 = pd.to_numeric(tg_ok.get("kappa_sigma"), errors="coerce").to_numpy(dtype=float)
        l2 = tg_ok.get("target", pd.Series(dtype="string")).astype(str).tolist()
        ax2.errorbar(x2, y2, yerr=e2, fmt="o", color="#2ca02c", ecolor="#2ca02c", capsize=4)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(l2, rotation=20, ha="right")

    ax2.axhline(1.0, color="#444444", linestyle="--", linewidth=1.2)
    ax2.set_ylabel("kappa_LLR")
    ax2.set_title("Target-stratified refit")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# 関数: `_sync_outputs_to_public` の入出力契約と処理意図を定義する。

def _sync_outputs_to_public(paths: Iterable[Path], private_root: Path, public_root: Path) -> List[str]:
    public_root.mkdir(parents=True, exist_ok=True)
    synced: List[str] = []
    for p in paths:
        try:
            rel = p.resolve().relative_to(private_root.resolve())
        except Exception:
            rel = Path(p.name)

        dst = public_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        synced.append(_safe_rel(dst, _ROOT))

    return synced


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    ap = argparse.ArgumentParser(description="Direct kappa_LLR fit from LLR batch residuals and beta=1 solar template.")
    ap.add_argument(
        "--points-csv",
        type=str,
        default=str(_ROOT / "output" / "private" / "llr" / "batch" / "llr_batch_points.csv"),
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(_ROOT / "output" / "private" / "llr"),
    )
    ap.add_argument(
        "--public-dir",
        type=str,
        default=str(_ROOT / "output" / "public" / "llr"),
    )
    ap.add_argument(
        "--modes",
        type=str,
        default="none,station,station_target,station_target_year",
        help="Comma-separated nuisance modes.",
    )
    ap.add_argument("--min-points-year", type=int, default=120)
    ap.add_argument("--min-points-station", type=int, default=180)
    ap.add_argument("--min-points-target", type=int, default=180)
    ap.add_argument("--year-fit-mode", type=str, default="station_target")
    ap.add_argument("--station-fit-mode", type=str, default="station_target_year")
    ap.add_argument("--target-fit-mode", type=str, default="station_target_year")
    ap.add_argument("--imbalance-fit-mode", type=str, default="station_target_year")
    ap.add_argument(
        "--imbalance-schemes",
        type=str,
        default="uniform,inv_station,inv_target,inv_station_target,station_cap_p95",
        help="Comma-separated imbalance weight schemes.",
    )
    ap.add_argument(
        "--stratified-weight-scheme",
        type=str,
        default="inv_station_target",
        help="Weight scheme used for station/target stratified refit.",
    )
    ap.add_argument("--weight-floor-station", type=int, default=180)
    ap.add_argument("--weight-floor-target", type=int, default=180)
    ap.add_argument("--weight-floor-station-target", type=int, default=120)
    ap.add_argument("--max-weight-cap", type=float, default=8.0)
    ap.add_argument("--template-null-scales", type=str, default="1,0,-1")
    ap.add_argument("--template-null-fit-mode", type=str, default="station_target_year")
    ap.add_argument("--template-null-weight-scheme", type=str, default="uniform")
    ap.add_argument("--decontamination-fit-mode", type=str, default="station_target_year")
    ap.add_argument("--decontamination-weight-scheme", type=str, default="inv_station_target")
    ap.add_argument("--decontamination-min-orth-std", type=float, default=1e-6)
    args = ap.parse_args()

    points_csv = Path(str(args.points_csv))
    out_dir = Path(str(args.out_dir))
    public_dir = Path(str(args.public_dir))
    if not points_csv.is_absolute():
        points_csv = (_ROOT / points_csv).resolve()

    if not out_dir.is_absolute():
        out_dir = (_ROOT / out_dir).resolve()

    if not public_dir.is_absolute():
        public_dir = (_ROOT / public_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_points(points_csv)
    if df.empty:
        raise RuntimeError(f"no valid inlier rows from {points_csv}")

    template_csv = out_dir / "llr_kappa_llr_template_points.csv"
    template_df = df[
        [
            "epoch_utc",
            "station",
            "target",
            "year",
            "year_float",
            "residual_sr_tropo_tide_ns",
            "dt_sun_shapiro_ns",
        ]
    ].copy()
    template_df = template_df.rename(columns={"dt_sun_shapiro_ns": "template_dt_sun_shapiro_ns"})
    template_df.to_csv(template_csv, index=False)

    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    fit_rows: List[FitResult] = []
    selected_cache: Dict[str, Any] = {}
    for mode in modes:
        x, y, _ = _build_design_matrix(df, mode=mode)
        fr = _fit_ols(mode=mode, x=x, y=y)
        fit_rows.append(fr)
        selected_cache[mode] = {"y_hat": fr.y_hat, "fit_residual": fr.fit_residual}

    best = _pick_best_mode(fit_rows)

    summary_rows: List[Dict[str, Any]] = []
    for fr in fit_rows:
        summary_rows.append(
            {
                "mode": fr.mode,
                "n_points": fr.n_points,
                "n_params": fr.n_params,
                "dof": fr.dof,
                "rmse_ns": fr.rmse_ns,
                "aic_like": fr.aic_like,
                "bic_like": fr.bic_like,
                "kappa_est": fr.kappa_est,
                "kappa_sigma": fr.kappa_sigma,
                "delta_kappa": fr.delta_kappa,
                "z_value": fr.z_value,
                "abs_z": fr.abs_z,
                "status": fr.status,
                "is_selected": fr.mode == best.mode,
            }
        )

    mode_df = pd.DataFrame(summary_rows).sort_values(["aic_like", "rmse_ns"]).reset_index(drop=True)
    mode_csv = out_dir / "llr_kappa_llr_fit_mode_summary.csv"
    mode_df.to_csv(mode_csv, index=False)

    selected_points_df = template_df.copy()
    selected_points_df["fit_mode"] = best.mode
    selected_points_df["y_hat_ns"] = selected_cache[best.mode]["y_hat"]
    selected_points_df["fit_residual_ns"] = selected_cache[best.mode]["fit_residual"]
    selected_points_csv = out_dir / "llr_kappa_llr_fit_selected_points.csv"
    selected_points_df.to_csv(selected_points_csv, index=False)

    year_df, station_df, consistency = _make_consistency_tables(
        df=df,
        year_fit_mode=str(args.year_fit_mode),
        station_fit_mode=str(args.station_fit_mode),
        min_points_year=int(args.min_points_year),
        min_points_station=int(args.min_points_station),
    )
    year_csv = out_dir / "llr_kappa_llr_year_consistency.csv"
    station_csv = out_dir / "llr_kappa_llr_station_consistency.csv"
    year_df.to_csv(year_csv, index=False)
    station_df.to_csv(station_csv, index=False)

    plot_pdf = out_dir / "llr_kappa_llr_fit.pdf"
    plot_png = out_dir / "llr_kappa_llr_fit.png"
    _write_plot(
        mode_df=mode_df,
        selected_points_df=selected_points_df,
        year_df=year_df,
        out_pdf=plot_pdf,
        out_png=plot_png,
    )

    imbalance_schemes = [s.strip() for s in str(args.imbalance_schemes).split(",") if s.strip()]
    policy_df, station_strat_df, target_strat_df, imbalance_summary = _run_imbalance_stratified_audit(
        df=df,
        imbalance_fit_mode=str(args.imbalance_fit_mode),
        imbalance_schemes=imbalance_schemes,
        station_fit_mode=str(args.station_fit_mode),
        target_fit_mode=str(args.target_fit_mode),
        min_points_station=int(args.min_points_station),
        min_points_target=int(args.min_points_target),
        stratified_weight_scheme=str(args.stratified_weight_scheme),
        floor_station=int(args.weight_floor_station),
        floor_target=int(args.weight_floor_target),
        floor_station_target=int(args.weight_floor_station_target),
        max_weight_cap=float(args.max_weight_cap),
    )
    imbalance_policy_csv = out_dir / "llr_kappa_llr_imbalance_policy_summary.csv"
    station_strat_csv = out_dir / "llr_kappa_llr_station_stratified_refit.csv"
    target_strat_csv = out_dir / "llr_kappa_llr_target_stratified_refit.csv"
    imbalance_policy_df = policy_df.copy()
    imbalance_policy_df.to_csv(imbalance_policy_csv, index=False)
    station_strat_df.to_csv(station_strat_csv, index=False)
    target_strat_df.to_csv(target_strat_csv, index=False)
    imbalance_plot_pdf = out_dir / "llr_kappa_llr_imbalance_audit.pdf"
    imbalance_plot_png = out_dir / "llr_kappa_llr_imbalance_audit.png"
    _write_imbalance_plot(
        policy_df=policy_df,
        station_df=station_strat_df,
        target_df=target_strat_df,
        out_pdf=imbalance_plot_pdf,
        out_png=imbalance_plot_png,
    )

    null_scales: List[float] = []
    for tok in str(args.template_null_scales).split(","):
        t = tok.strip()
        if not t:
            continue

        try:
            null_scales.append(float(t))
        except ValueError:
            continue

    if not null_scales:
        null_scales = [1.0, 0.0, -1.0]

    null_weight = _build_imbalance_weight(
        df,
        scheme=str(args.template_null_weight_scheme),
        floor_station=int(args.weight_floor_station),
        floor_target=int(args.weight_floor_target),
        floor_station_target=int(args.weight_floor_station_target),
        max_weight_cap=float(args.max_weight_cap),
    )
    template_null_df, template_null_summary = _run_template_null_test(
        df=df,
        fit_mode=str(args.template_null_fit_mode),
        scales=null_scales,
        sample_weight=null_weight,
    )
    template_null_csv = out_dir / "llr_kappa_llr_template_nulltest_summary.csv"
    template_null_df.to_csv(template_null_csv, index=False)
    template_null_pdf = out_dir / "llr_kappa_llr_template_nulltest.pdf"
    template_null_png = out_dir / "llr_kappa_llr_template_nulltest.png"
    _write_template_null_plot(
        null_df=template_null_df,
        out_pdf=template_null_pdf,
        out_png=template_null_png,
    )

    decont_weight = _build_imbalance_weight(
        df,
        scheme=str(args.decontamination_weight_scheme),
        floor_station=int(args.weight_floor_station),
        floor_target=int(args.weight_floor_target),
        floor_station_target=int(args.weight_floor_station_target),
        max_weight_cap=float(args.max_weight_cap),
    )
    decont_proj_df, decont_summary = _run_template_decontamination_audit(
        df=df,
        fit_mode=str(args.decontamination_fit_mode),
        sample_weight=decont_weight,
        min_std=float(args.decontamination_min_orth_std),
    )
    decont_summary_csv = out_dir / "llr_kappa_llr_template_decontamination_summary.csv"
    decont_proj_csv = out_dir / "llr_kappa_llr_template_decontamination_projection.csv"
    pd.DataFrame([decont_summary]).to_csv(decont_summary_csv, index=False)
    decont_proj_df.to_csv(decont_proj_csv, index=False)
    decont_pdf = out_dir / "llr_kappa_llr_template_decontamination.pdf"
    decont_png = out_dir / "llr_kappa_llr_template_decontamination.png"
    _write_template_decontamination_plot(
        summary=decont_summary,
        proj_df=decont_proj_df,
        out_pdf=decont_pdf,
        out_png=decont_png,
    )

    year_status = str(consistency.get("year", {}).get("status", "reject"))
    station_status = str(consistency.get("station", {}).get("status", "reject"))
    decont_kappa_est = float(decont_summary.get("decontaminated_kappa_est", best.kappa_est))
    decont_kappa_sigma = float(decont_summary.get("decontaminated_kappa_sigma", best.kappa_sigma))
    decont_abs_z = float(decont_summary.get("decontaminated_abs_z", best.abs_z))
    global_status = str(decont_summary.get("kappa_minus_1_status", _status_from_abs_z(decont_abs_z)))
    imbalance_status = str(imbalance_summary.get("robustness_envelope", {}).get("status", "reject"))
    station_strat_status = str(imbalance_summary.get("station_stratified", {}).get("status", "reject"))
    target_strat_status = str(imbalance_summary.get("target_stratified", {}).get("status", "reject"))
    template_null_status = str(template_null_summary.get("status", "reject"))
    template_decont_status = str(decont_summary.get("status", "reject"))
    llr_bias_components = {
        "imbalance_policy_gate": imbalance_status,
        "station_stratified_gate": station_strat_status,
        "target_stratified_gate": target_strat_status,
        "template_null_gate": template_null_status,
        "template_decontamination_gate": template_decont_status,
    }
    llr_bias_status = _combine_gate_status(list(llr_bias_components.values()))
    all_status = [
        global_status,
        year_status,
        station_status,
        llr_bias_status,
    ]
    if all(s == "pass" for s in all_status):
        overall = "pass"
    elif "reject" in all_status:
        overall = "reject"
    else:
        overall = "watch"

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": {"step": "8.7.47.1-8.7.47.10"},
        "input": {
            "points_csv": _safe_rel(points_csv, _ROOT),
            "n_inlier_points": int(len(df)),
            "n_station": int(df["station"].nunique()),
            "n_target": int(df["target"].nunique()),
            "year_range": [int(df["year"].min()), int(df["year"].max())],
        },
        "template": {
            "description": "delta_rho_P template from beta=1 normalized solar-Shapiro contribution",
            "field_name": "template_dt_sun_shapiro_ns",
            "template_points_csv": _safe_rel(template_csv, _ROOT),
        },
        "fit": {
            "modes": mode_df.to_dict(orient="records"),
            "selected_mode": best.mode,
            "selected_kappa_est": decont_kappa_est,
            "selected_kappa_sigma": decont_kappa_sigma,
            "selected_abs_z": decont_abs_z,
            "selected_status": global_status,
            "beta_mapping": {
                "rule": "beta := kappa_LLR (template-decontaminated fit)",
                "source": "template_decontaminated",
                "beta_est": decont_kappa_est,
                "beta_sigma": decont_kappa_sigma,
                "abs_z_beta_minus_1": decont_abs_z,
                "baseline_beta_est": best.kappa_est,
                "baseline_beta_sigma": best.kappa_sigma,
                "baseline_abs_z_beta_minus_1": best.abs_z,
            },
        },
        "consistency": consistency,
        "imbalance_audit": {
            **imbalance_summary,
            "outputs": {
                "imbalance_policy_csv": _safe_rel(imbalance_policy_csv, _ROOT),
                "station_stratified_csv": _safe_rel(station_strat_csv, _ROOT),
                "target_stratified_csv": _safe_rel(target_strat_csv, _ROOT),
                "imbalance_plot_pdf": _safe_rel(imbalance_plot_pdf, _ROOT),
                "imbalance_plot_png": _safe_rel(imbalance_plot_png, _ROOT),
            },
        },
        "template_null_test": {
            **template_null_summary,
            "weight_scheme": str(args.template_null_weight_scheme),
            "outputs": {
                "template_null_csv": _safe_rel(template_null_csv, _ROOT),
                "template_null_pdf": _safe_rel(template_null_pdf, _ROOT),
                "template_null_png": _safe_rel(template_null_png, _ROOT),
            },
        },
        "template_decontamination": {
            **decont_summary,
            "fit_weight_scheme": str(args.decontamination_weight_scheme),
            "outputs": {
                "decontamination_summary_csv": _safe_rel(decont_summary_csv, _ROOT),
                "decontamination_projection_csv": _safe_rel(decont_proj_csv, _ROOT),
                "decontamination_pdf": _safe_rel(decont_pdf, _ROOT),
                "decontamination_png": _safe_rel(decont_png, _ROOT),
            },
        },
        "bias_audit": {
            "components": llr_bias_components,
            "status": llr_bias_status,
        },
        "gate": {
            "kappa_gate": {"rule": "|z(kappa-1)| <= 2(pass), <=3(watch), >3(reject)", "status": global_status},
            "year_consistency_gate": {"rule": "chi2/dof <=2(pass), <=5(watch), >5(reject)", "status": year_status},
            "station_consistency_gate": {"rule": "chi2/dof <=2(pass), <=5(watch), >5(reject)", "status": station_status},
            "imbalance_policy_gate": {"rule": "max |z(policy-uniform)| <=2(pass), <=3(watch), >3(reject)", "status": imbalance_status},
            "station_stratified_gate": {"rule": "chi2/dof <=2(pass), <=5(watch), >5(reject)", "status": station_strat_status},
            "target_stratified_gate": {"rule": "chi2/dof <=2(pass), <=5(watch), >5(reject)", "status": target_strat_status},
            "template_null_gate": {"rule": "max |z(scale-ref)| <=2(pass), <=3(watch), >3(reject)", "status": template_null_status},
            "template_decontamination_gate": {"rule": "|z(kappa_decont-kappa_base)| <=2(pass), <=3(watch), >3(reject)", "status": template_decont_status},
            "bias_audit_gate": {"rule": "all bias components pass/watch with no reject", "status": llr_bias_status},
        },
        "overall_status": overall,
        "outputs": {
            "template_points_csv": _safe_rel(template_csv, _ROOT),
            "mode_summary_csv": _safe_rel(mode_csv, _ROOT),
            "selected_points_csv": _safe_rel(selected_points_csv, _ROOT),
            "year_consistency_csv": _safe_rel(year_csv, _ROOT),
            "station_consistency_csv": _safe_rel(station_csv, _ROOT),
            "plot_pdf": _safe_rel(plot_pdf, _ROOT),
            "plot_png": _safe_rel(plot_png, _ROOT),
            "imbalance_policy_csv": _safe_rel(imbalance_policy_csv, _ROOT),
            "station_stratified_csv": _safe_rel(station_strat_csv, _ROOT),
            "target_stratified_csv": _safe_rel(target_strat_csv, _ROOT),
            "imbalance_plot_pdf": _safe_rel(imbalance_plot_pdf, _ROOT),
            "imbalance_plot_png": _safe_rel(imbalance_plot_png, _ROOT),
            "template_null_csv": _safe_rel(template_null_csv, _ROOT),
            "template_null_pdf": _safe_rel(template_null_pdf, _ROOT),
            "template_null_png": _safe_rel(template_null_png, _ROOT),
            "decontamination_summary_csv": _safe_rel(decont_summary_csv, _ROOT),
            "decontamination_projection_csv": _safe_rel(decont_proj_csv, _ROOT),
            "decontamination_pdf": _safe_rel(decont_pdf, _ROOT),
            "decontamination_png": _safe_rel(decont_png, _ROOT),
        },
    }

    metrics_json = out_dir / "llr_kappa_llr_metrics.json"
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    produced = [
        template_csv,
        mode_csv,
        selected_points_csv,
        year_csv,
        station_csv,
        imbalance_policy_csv,
        station_strat_csv,
        target_strat_csv,
        template_null_csv,
        decont_summary_csv,
        decont_proj_csv,
        metrics_json,
        plot_pdf,
        plot_png,
        imbalance_plot_pdf,
        imbalance_plot_png,
        template_null_pdf,
        template_null_png,
        decont_pdf,
        decont_png,
    ]
    synced = _sync_outputs_to_public(produced, private_root=out_dir, public_root=public_dir)

    print(f"[ok] wrote: {template_csv}")
    print(f"[ok] wrote: {mode_csv}")
    print(f"[ok] wrote: {selected_points_csv}")
    print(f"[ok] wrote: {year_csv}")
    print(f"[ok] wrote: {station_csv}")
    print(f"[ok] wrote: {imbalance_policy_csv}")
    print(f"[ok] wrote: {station_strat_csv}")
    print(f"[ok] wrote: {target_strat_csv}")
    print(f"[ok] wrote: {template_null_csv}")
    print(f"[ok] wrote: {decont_summary_csv}")
    print(f"[ok] wrote: {decont_proj_csv}")
    print(f"[ok] wrote: {metrics_json}")
    print(f"[ok] wrote: {plot_pdf}")
    print(f"[ok] wrote: {plot_png}")
    print(f"[ok] wrote: {imbalance_plot_pdf}")
    print(f"[ok] wrote: {imbalance_plot_png}")
    print(f"[ok] wrote: {template_null_pdf}")
    print(f"[ok] wrote: {template_null_png}")
    print(f"[ok] wrote: {decont_pdf}")
    print(f"[ok] wrote: {decont_png}")
    print(f"[ok] synced_to_public: {len(synced)} files")
    print(
        f"[summary] mode={best.mode} "
        f"kappa(beta-map)={decont_kappa_est:.6f}+/-{decont_kappa_sigma:.6f} "
        f"baseline_kappa={best.kappa_est:.6f}+/-{best.kappa_sigma:.6f} "
        f"imbalance_status={imbalance_status} "
        f"template_null_status={template_null_status} "
        f"template_decont_status={template_decont_status} "
        f"|z|={decont_abs_z:.3f} "
        f"overall={overall}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
