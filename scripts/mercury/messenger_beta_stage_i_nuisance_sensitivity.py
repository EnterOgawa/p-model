#!/usr/bin/env python3
"""
messenger_beta_stage_i_nuisance_sensitivity.py

Roadmap Step 8.7.48.9 (high-correlation nuisance sensitivity audit) implementation.

Purpose:
- Quantify beta_dyn sensitivity against nuisance modeling choices using the
  same Stage D/E joint-fit interface.
- Keep outputs machine-readable and synced to output/public.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from scripts.mercury.messenger_beta_stage_d_joint_fit import (
    _build_design_matrix,
    _fit_joint,
    _parse_epoch_series,
    _sync_to_public,
)
from scripts.summary.worklog import append_event


# Class: Defines one nuisance scenario result row.
@dataclass
class ScenarioRow:
    branch: str
    scenario_id: str
    scenario_group: str
    station_bias_cap: int
    use_sun_quad_proxy: bool
    use_transponder_quad: bool
    use_plasma_proxy: bool
    use_srp_proxy: bool
    n_rows: int
    n_range_rows: int
    n_doppler_rows: int
    n_params: int
    beta_dyn: float
    beta_sigma: float
    beta_z_from_1: float
    beta_lt: float
    beta_lt_sigma: float
    beta_lt_z_from_1: float
    beta_split_mode: str
    beta_dyn_lt_consistency_status: str
    beta_dyn_lt_template_overlap: float
    beta_dyn_lt_overlap_status: str
    beta_dyn_lt_delta_status: str
    z_delta_vs_base: float
    fit_status: str
    sigma_status: str
    delta_status: str
    overall_status: str
    note: str


# Class: Defines one branch-level aggregate consistency row.

@dataclass
class BranchSummaryRow:
    branch: str
    n_scenarios: int
    n_core_scenarios: int
    n_diagnostic_scenarios: int
    beta_base: float
    beta_sigma_base: float
    beta_min: float
    beta_max: float
    beta_span: float
    max_abs_z_delta_core: float
    max_abs_z_delta_all: float
    max_abs_z_delta: float
    median_beta_sigma: float
    status: str
    sigma_status_diagnostic: str
    status_all_diagnostic: str
    note: str


# Function: Returns repository-relative path when possible.

def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


# Function: Resolves possibly-relative path against repository root.

def _resolve_path(path_str: str, root: Path) -> Path:
    p = Path(str(path_str))
    # 条件分岐: `p.is_absolute()` を満たす経路を評価する。
    if p.is_absolute():
        return p

    return (root / p).resolve()


# Function: Combines statuses with reject > watch > pass priority.

def _combine_status(values: Iterable[str]) -> str:
    norm = [str(v or "").strip().lower() for v in values if str(v or "").strip()]
    # 条件分岐: `len(norm) <= 0` を満たす経路を評価する。
    if len(norm) <= 0:
        return "reject"

    # 条件分岐: `any(v == "reject" for v in norm)` を満たす経路を評価する。

    if any(v == "reject" for v in norm):
        return "reject"

    # 条件分岐: `all(v == "pass" for v in norm)` を満たす経路を評価する。

    if all(v == "pass" for v in norm):
        return "pass"

    return "watch"


# Function: Returns pass/watch/reject from absolute thresholding.

def _status_from_abs(value: float, pass_thr: float, watch_thr: float) -> str:
    # 条件分岐: `not math.isfinite(value)` を満たす経路を評価する。
    if not math.isfinite(value):
        return "reject"

    # 条件分岐: `float(value) <= float(pass_thr)` を満たす経路を評価する。

    if float(value) <= float(pass_thr):
        return "pass"

    # 条件分岐: `float(value) <= float(watch_thr)` を満たす経路を評価する。

    if float(value) <= float(watch_thr):
        return "watch"

    return "reject"


# Function: Loads one branch/channel CSV and normalizes key columns.

def _load_branch_channel_csv(path: Path, channel: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 条件分岐: `"epoch_utc" not in df.columns` を満たす経路を評価する。
    if "epoch_utc" not in df.columns:
        raise ValueError(f"{path}: missing epoch_utc")

    # 条件分岐: `"observable_value" not in df.columns` を満たす経路を評価する。

    if "observable_value" not in df.columns:
        raise ValueError(f"{path}: missing observable_value")

    work = pd.DataFrame()
    work["epoch_utc"] = _parse_epoch_series(df["epoch_utc"])
    work["observable_value"] = pd.to_numeric(df["observable_value"], errors="coerce")
    work["channel"] = str(channel)
    # 条件分岐: `"station_id" in df.columns` を満たす経路を評価する。
    if "station_id" in df.columns:
        work["station_id"] = df["station_id"].astype(str)
    else:
        work["station_id"] = "unknown"

    work = work.dropna(subset=["epoch_utc", "observable_value"]).reset_index(drop=True)
    return work


# Function: Aggregates rows while preserving station/channel columns.

def _aggregate_channel(df: pd.DataFrame, bin_minutes: int) -> pd.DataFrame:
    work = df.copy()
    # 条件分岐: `int(bin_minutes) > 0` を満たす経路を評価する。
    if int(bin_minutes) > 0:
        work["epoch_bin"] = work["epoch_utc"].dt.floor(f"{int(bin_minutes)}min")
    else:
        work["epoch_bin"] = work["epoch_utc"]

    out = (
        work.groupby(["epoch_bin", "station_id", "channel"], as_index=False)
        .agg(observable_value=("observable_value", "median"))
        .sort_values("epoch_bin")
        .reset_index(drop=True)
    )
    out["epoch_utc"] = out["epoch_bin"]
    out = out.drop(columns=["epoch_bin"])
    return out


# Function: Returns z-difference between scenario beta and base beta.

def _z_delta_beta(beta: float, sigma: float, beta_ref: float, sigma_ref: float) -> float:
    # 条件分岐: `(not math.isfinite(beta)) or (not math.isfinite(sigma))` を満たす経路を評価する。
    if (not math.isfinite(beta)) or (not math.isfinite(sigma)):
        return float("nan")

    # 条件分岐: `(not math.isfinite(beta_ref)) or (not math.isfinite(sigma_ref))` を満たす経路を評価する。

    if (not math.isfinite(beta_ref)) or (not math.isfinite(sigma_ref)):
        return float("nan")

    denom = float(math.sqrt(max(0.0, sigma * sigma + sigma_ref * sigma_ref)))
    # 条件分岐: `denom <= 0.0` を満たす経路を評価する。
    if denom <= 0.0:
        return float("nan")

    return float(abs(beta - beta_ref) / denom)


# Function: Demeans an array per channel to reduce intercept collinearity.

def _demean_per_channel(values: np.ndarray, channels: np.ndarray) -> np.ndarray:
    out = values.astype(float).copy()
    for key in ("range", "doppler"):
        mask = channels == key
        # 条件分岐: `int(np.sum(mask)) <= 0` を満たす経路を評価する。
        if int(np.sum(mask)) <= 0:
            continue

        out[mask] = out[mask] - float(np.mean(out[mask]))

    return out


# Function: Normalizes one vector by robust RMS while preserving zeros.

def _normalize_vector(values: np.ndarray) -> np.ndarray:
    vec = values.astype(float).copy()
    rms = float(np.sqrt(np.mean(np.square(vec)))) if len(vec) > 0 else 0.0
    # 条件分岐: `(not math.isfinite(rms)) or rms <= 0.0` を満たす経路を評価する。
    if (not math.isfinite(rms)) or rms <= 0.0:
        return vec

    return vec / rms


# Function: Builds extra nuisance proxy columns for one scenario.

def _build_proxy_columns(
    work: pd.DataFrame,
    orbital_period_days: float,
    use_sun_quad_proxy: bool,
    use_transponder_quad: bool,
    use_plasma_proxy: bool,
    use_srp_proxy: bool,
) -> Tuple[List[np.ndarray], List[str]]:
    out_cols: List[np.ndarray] = []
    out_labels: List[str] = []
    channels = work["channel"].astype(str).to_numpy()
    is_range = channels == "range"
    is_dop = channels == "doppler"
    t_days = work["t_days"].to_numpy(dtype=float)
    theta = 2.0 * np.pi * t_days / float(orbital_period_days)

    # 条件分岐: `bool(use_sun_quad_proxy)` を満たす経路を評価する。
    if bool(use_sun_quad_proxy):
        quad_wave = np.where(is_range, np.cos(2.0 * theta), -np.sin(2.0 * theta))
        quad_wave = _demean_per_channel(quad_wave, channels)
        quad_wave = _normalize_vector(quad_wave)
        out_cols.append(quad_wave)
        out_labels.append("sun_quadrupole_like_proxy_norm")

    # 条件分岐: `bool(use_transponder_quad)` を満たす経路を評価する。

    if bool(use_transponder_quad):
        t_center = t_days - float(np.mean(t_days))
        q = np.square(t_center)
        q_r = np.where(is_range, q, 0.0)
        q_d = np.where(is_dop, q, 0.0)
        q_r = _demean_per_channel(q_r, channels)
        q_d = _demean_per_channel(q_d, channels)
        q_r = _normalize_vector(q_r)
        q_d = _normalize_vector(q_d)
        out_cols.append(q_r)
        out_labels.append("transponder_quad_range_norm_per_day2")
        out_cols.append(q_d)
        out_labels.append("transponder_quad_doppler_norm_per_day2")

    # 条件分岐: `bool(use_plasma_proxy)` を満たす経路を評価する。

    if bool(use_plasma_proxy):
        phase_y = 2.0 * np.pi * t_days / 365.25
        p_cos = np.where(is_range, np.cos(phase_y), np.cos(phase_y))
        p_sin = np.where(is_range, np.sin(phase_y), np.sin(phase_y))
        p_cos = _demean_per_channel(p_cos, channels)
        p_sin = _demean_per_channel(p_sin, channels)
        p_cos = _normalize_vector(p_cos)
        p_sin = _normalize_vector(p_sin)
        out_cols.append(p_cos)
        out_labels.append("plasma_proxy_cos_annual_norm")
        out_cols.append(p_sin)
        out_labels.append("plasma_proxy_sin_annual_norm")

    # 条件分岐: `bool(use_srp_proxy)` を満たす経路を評価する。

    if bool(use_srp_proxy):
        # SRP-like branch: orbital-phase nuisance at 1-cycle harmonic.
        srp_cos = np.where(is_range, np.cos(theta), np.cos(theta))
        srp_sin = np.where(is_range, np.sin(theta), np.sin(theta))
        srp_cos = _demean_per_channel(srp_cos, channels)
        srp_sin = _demean_per_channel(srp_sin, channels)
        srp_cos = _normalize_vector(srp_cos)
        srp_sin = _normalize_vector(srp_sin)
        out_cols.append(srp_cos)
        out_labels.append("srp_like_proxy_cos_orbital_norm")
        out_cols.append(srp_sin)
        out_labels.append("srp_like_proxy_sin_orbital_norm")

    return (out_cols, out_labels)


# Function: Runs one scenario fit with optional nuisance proxies.

def _run_scenario_fit(
    df_joint: pd.DataFrame,
    orbital_period_days: float,
    station_bias_cap: int,
    min_joint_rows: int,
    sigma_watch_threshold: float,
    split_beta_lt: bool,
    use_sun_quad_proxy: bool,
    use_transponder_quad: bool,
    use_plasma_proxy: bool,
    use_srp_proxy: bool,
) -> Dict[str, object]:
    X, y_norm, y_obs, labels, _meta, work = _build_design_matrix(
        df_joint[["epoch_utc", "observable_value", "station_id", "channel"]].copy(),
        orbital_period_days=float(orbital_period_days),
        max_station_bias_per_channel=int(station_bias_cap),
        split_beta_lt=bool(split_beta_lt),
    )
    extra_cols, extra_labels = _build_proxy_columns(
        work=work,
        orbital_period_days=float(orbital_period_days),
        use_sun_quad_proxy=bool(use_sun_quad_proxy),
        use_transponder_quad=bool(use_transponder_quad),
        use_plasma_proxy=bool(use_plasma_proxy),
        use_srp_proxy=bool(use_srp_proxy),
    )
    # 条件分岐: `len(extra_cols) > 0` を満たす経路を評価する。
    if len(extra_cols) > 0:
        X = np.column_stack([X] + extra_cols)
        labels = list(labels) + list(extra_labels)

    channels = work["channel"].astype(str).to_numpy()
    fit, _coef, _fit_norm, _residual_norm = _fit_joint(
        X=X,
        y_norm=y_norm,
        y_obs=y_obs,
        scale_by_row=work["scale_by_row"].to_numpy(dtype=float),
        labels=labels,
        channels=channels,
        min_rows=int(min_joint_rows),
        sigma_watch_threshold=float(sigma_watch_threshold),
    )
    return {
        "n_rows": int(fit.n_rows),
        "n_range_rows": int(fit.n_range_rows),
        "n_doppler_rows": int(fit.n_doppler_rows),
        "n_params": int(X.shape[1]),
        "beta_dyn": float(fit.beta_dyn),
        "beta_sigma": float(fit.beta_sigma),
        "beta_z_from_1": float(fit.beta_z_from_1),
        "beta_lt": float(fit.beta_lt),
        "beta_lt_sigma": float(fit.beta_lt_sigma),
        "beta_lt_z_from_1": float(fit.beta_lt_z_from_1),
        "beta_split_mode": str(fit.beta_split_mode),
        "beta_dyn_lt_consistency_status": str(fit.beta_dyn_lt_consistency_status),
        "beta_dyn_lt_template_overlap": float(fit.beta_dyn_lt_template_overlap),
        "beta_dyn_lt_overlap_status": str(fit.beta_dyn_lt_overlap_status),
        "beta_dyn_lt_delta_status": str(fit.beta_dyn_lt_delta_status),
        "fit_status": str(fit.status_data),
        "sigma_status": str(fit.status_sigma),
        "overall_status_fit": str(fit.overall_status),
    }


# Function: Writes scenario rows to CSV.

def _write_scenarios_csv(path: Path, rows: Sequence[ScenarioRow]) -> None:
    fields = [
        "branch",
        "scenario_id",
        "scenario_group",
        "station_bias_cap",
        "use_sun_quad_proxy",
        "use_transponder_quad",
        "use_plasma_proxy",
        "use_srp_proxy",
        "n_rows",
        "n_range_rows",
        "n_doppler_rows",
        "n_params",
        "beta_dyn",
        "beta_sigma",
        "beta_z_from_1",
        "beta_lt",
        "beta_lt_sigma",
        "beta_lt_z_from_1",
        "beta_split_mode",
        "beta_dyn_lt_consistency_status",
        "beta_dyn_lt_template_overlap",
        "beta_dyn_lt_overlap_status",
        "beta_dyn_lt_delta_status",
        "z_delta_vs_base",
        "fit_status",
        "sigma_status",
        "delta_status",
        "overall_status",
        "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "branch": r.branch,
                    "scenario_id": r.scenario_id,
                    "scenario_group": r.scenario_group,
                    "station_bias_cap": int(r.station_bias_cap),
                    "use_sun_quad_proxy": bool(r.use_sun_quad_proxy),
                    "use_transponder_quad": bool(r.use_transponder_quad),
                    "use_plasma_proxy": bool(r.use_plasma_proxy),
                    "use_srp_proxy": bool(r.use_srp_proxy),
                    "n_rows": int(r.n_rows),
                    "n_range_rows": int(r.n_range_rows),
                    "n_doppler_rows": int(r.n_doppler_rows),
                    "n_params": int(r.n_params),
                    "beta_dyn": float(r.beta_dyn),
                    "beta_sigma": float(r.beta_sigma),
                    "beta_z_from_1": float(r.beta_z_from_1),
                    "beta_lt": float(r.beta_lt),
                    "beta_lt_sigma": float(r.beta_lt_sigma),
                    "beta_lt_z_from_1": float(r.beta_lt_z_from_1),
                    "beta_split_mode": str(r.beta_split_mode),
                    "beta_dyn_lt_consistency_status": str(r.beta_dyn_lt_consistency_status),
                    "beta_dyn_lt_template_overlap": float(r.beta_dyn_lt_template_overlap),
                    "beta_dyn_lt_overlap_status": str(r.beta_dyn_lt_overlap_status),
                    "beta_dyn_lt_delta_status": str(r.beta_dyn_lt_delta_status),
                    "z_delta_vs_base": float(r.z_delta_vs_base),
                    "fit_status": r.fit_status,
                    "sigma_status": r.sigma_status,
                    "delta_status": r.delta_status,
                    "overall_status": r.overall_status,
                    "note": r.note,
                }
            )


# Function: Writes branch summary rows to CSV.

def _write_branch_summary_csv(path: Path, rows: Sequence[BranchSummaryRow]) -> None:
    fields = [
        "branch",
        "n_scenarios",
        "n_core_scenarios",
        "n_diagnostic_scenarios",
        "beta_base",
        "beta_sigma_base",
        "beta_min",
        "beta_max",
        "beta_span",
        "max_abs_z_delta_core",
        "max_abs_z_delta_all",
        "max_abs_z_delta",
        "median_beta_sigma",
        "status",
        "sigma_status_diagnostic",
        "status_all_diagnostic",
        "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "branch": r.branch,
                    "n_scenarios": int(r.n_scenarios),
                    "n_core_scenarios": int(r.n_core_scenarios),
                    "n_diagnostic_scenarios": int(r.n_diagnostic_scenarios),
                    "beta_base": float(r.beta_base),
                    "beta_sigma_base": float(r.beta_sigma_base),
                    "beta_min": float(r.beta_min),
                    "beta_max": float(r.beta_max),
                    "beta_span": float(r.beta_span),
                    "max_abs_z_delta_core": float(r.max_abs_z_delta_core),
                    "max_abs_z_delta_all": float(r.max_abs_z_delta_all),
                    "max_abs_z_delta": float(r.max_abs_z_delta),
                    "median_beta_sigma": float(r.median_beta_sigma),
                    "status": r.status,
                    "sigma_status_diagnostic": r.sigma_status_diagnostic,
                    "status_all_diagnostic": r.status_all_diagnostic,
                    "note": r.note,
                }
            )


# Function: Builds branch-level summary metrics with core-vs-diagnostic split.

def _build_branch_summary(rows: Sequence[ScenarioRow], core_scenario_ids: Sequence[str]) -> List[BranchSummaryRow]:
    out: List[BranchSummaryRow] = []
    # 条件分岐: `len(rows) <= 0` を満たす経路を評価する。
    if len(rows) <= 0:
        return out

    core_ids = {str(x).strip() for x in core_scenario_ids if str(x).strip()}
    df = pd.DataFrame([r.__dict__ for r in rows])
    for branch in sorted(df["branch"].astype(str).unique().tolist()):
        sub = df.loc[df["branch"].astype(str) == str(branch)].copy()
        # 条件分岐: `len(sub) <= 0` を満たす経路を評価する。
        if len(sub) <= 0:
            continue

        base = sub.loc[sub["scenario_id"].astype(str) == "baseline"].copy()
        # 条件分岐: `len(base) <= 0` を満たす経路を評価する。
        if len(base) <= 0:
            beta_base = float("nan")
            beta_sigma_base = float("nan")
            note = "baseline_missing"
        else:
            beta_base = float(pd.to_numeric(base["beta_dyn"], errors="coerce").iloc[0])
            beta_sigma_base = float(pd.to_numeric(base["beta_sigma"], errors="coerce").iloc[0])
            note = "ok"

        sub["is_core"] = sub["scenario_id"].astype(str).isin(core_ids)
        sub_core = sub.loc[sub["is_core"]].copy()
        # 条件分岐: `len(sub_core) <= 0` を満たす経路を評価する。
        if len(sub_core) <= 0:
            sub_core = sub.copy()
            note = f"{note}|core_scenarios_missing_fallback_to_all"

        beta_vals = pd.to_numeric(sub["beta_dyn"], errors="coerce").to_numpy(dtype=float)
        beta_sigmas = pd.to_numeric(sub["beta_sigma"], errors="coerce").to_numpy(dtype=float)
        z_delta_all = pd.to_numeric(sub["z_delta_vs_base"], errors="coerce").to_numpy(dtype=float)
        z_all_valid = z_delta_all[np.isfinite(z_delta_all)]
        max_abs_z_all = float(np.max(np.abs(z_all_valid))) if len(z_all_valid) > 0 else float("nan")
        status_delta_all = _status_from_abs(max_abs_z_all, pass_thr=2.0, watch_thr=5.0)

        z_delta_core = pd.to_numeric(sub_core["z_delta_vs_base"], errors="coerce").to_numpy(dtype=float)
        z_core_valid = z_delta_core[np.isfinite(z_delta_core)]
        max_abs_z_core = float(np.max(np.abs(z_core_valid))) if len(z_core_valid) > 0 else float("nan")
        status_delta_core = _status_from_abs(max_abs_z_core, pass_thr=2.0, watch_thr=5.0)

        fit_status = _combine_status(sub_core["fit_status"].astype(str).tolist())
        sigma_status = _combine_status(sub["sigma_status"].astype(str).tolist())
        status = _combine_status([fit_status, status_delta_core])

        beta_finite = beta_vals[np.isfinite(beta_vals)]
        sigma_finite = beta_sigmas[np.isfinite(beta_sigmas)]
        beta_min = float(np.min(beta_finite)) if len(beta_finite) > 0 else float("nan")
        beta_max = float(np.max(beta_finite)) if len(beta_finite) > 0 else float("nan")
        beta_span = float(beta_max - beta_min) if len(beta_finite) > 0 else float("nan")
        sigma_med = float(np.median(sigma_finite)) if len(sigma_finite) > 0 else float("nan")

        out.append(
            BranchSummaryRow(
                branch=str(branch),
                n_scenarios=int(len(sub)),
                n_core_scenarios=int(len(sub_core)),
                n_diagnostic_scenarios=int(max(0, len(sub) - len(sub_core))),
                beta_base=float(beta_base),
                beta_sigma_base=float(beta_sigma_base),
                beta_min=float(beta_min),
                beta_max=float(beta_max),
                beta_span=float(beta_span),
                max_abs_z_delta_core=float(max_abs_z_core),
                max_abs_z_delta_all=float(max_abs_z_all),
                max_abs_z_delta=float(max_abs_z_core),
                median_beta_sigma=float(sigma_med),
                status=str(status),
                sigma_status_diagnostic=str(sigma_status),
                status_all_diagnostic=str(status_delta_all),
                note=str(note),
            )
        )

    return out


# Function: Builds beta_dyn deviation decomposition per branch for diagnostics.

def _build_beta_dyn_decomposition(rows: Sequence[ScenarioRow]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    # 条件分岐: `len(rows) <= 0` を満たす経路を評価する。
    if len(rows) <= 0:
        return out


    df = pd.DataFrame([r.__dict__ for r in rows])
    for branch in sorted(df["branch"].astype(str).unique().tolist()):
        sub = df.loc[df["branch"].astype(str) == str(branch)].copy()
        # 条件分岐: `len(sub) <= 0` を満たす経路を評価する。
        if len(sub) <= 0:
            continue

        sub["abs_z_beta_minus_1"] = np.abs(
            (pd.to_numeric(sub["beta_dyn"], errors="coerce") - 1.0)
            / pd.to_numeric(sub["beta_sigma"], errors="coerce")
        )
        base = sub.loc[sub["scenario_id"].astype(str) == "baseline"].copy()
        # 条件分岐: `len(base) <= 0` を満たす経路を評価する。
        if len(base) <= 0:
            continue

        best_idx = sub["abs_z_beta_minus_1"].idxmin()
        best_row = sub.loc[int(best_idx)]
        max_shift_idx = pd.to_numeric(sub["z_delta_vs_base"], errors="coerce").abs().idxmax()
        max_shift_row = sub.loc[int(max_shift_idx)]
        base_row = base.iloc[0]
        out.append(
            {
                "branch": str(branch),
                "baseline": {
                    "scenario_id": str(base_row["scenario_id"]),
                    "beta_dyn": float(base_row["beta_dyn"]),
                    "beta_sigma": float(base_row["beta_sigma"]),
                    "abs_z_beta_minus_1": float(base_row["abs_z_beta_minus_1"]),
                },
                "best_abs_z": {
                    "scenario_id": str(best_row["scenario_id"]),
                    "scenario_group": str(best_row["scenario_group"]),
                    "beta_dyn": float(best_row["beta_dyn"]),
                    "beta_sigma": float(best_row["beta_sigma"]),
                    "abs_z_beta_minus_1": float(best_row["abs_z_beta_minus_1"]),
                    "delta_vs_baseline_z": float(best_row["z_delta_vs_base"]),
                },
                "max_shift_vs_baseline": {
                    "scenario_id": str(max_shift_row["scenario_id"]),
                    "scenario_group": str(max_shift_row["scenario_group"]),
                    "beta_dyn": float(max_shift_row["beta_dyn"]),
                    "beta_sigma": float(max_shift_row["beta_sigma"]),
                    "abs_z_beta_minus_1": float(max_shift_row["abs_z_beta_minus_1"]),
                    "delta_vs_baseline_z": float(max_shift_row["z_delta_vs_base"]),
                },
            }
        )

    return out


# Function: Creates nuisance sensitivity audit plot.

def _make_plot(
    rows: Sequence[ScenarioRow],
    out_pdf: Path,
    out_png: Path,
) -> Optional[str]:
    # 条件分岐: `plt is None` を満たす経路を評価する。
    if plt is None:
        return "matplotlib_unavailable"

    df = pd.DataFrame([r.__dict__ for r in rows])
    # 条件分岐: `len(df) <= 0` を満たす経路を評価する。
    if len(df) <= 0:
        return "no_data"

    branches = sorted(df["branch"].astype(str).unique().tolist())
    fig, axes = plt.subplots(len(branches), 1, figsize=(13.5, 4.4 * max(1, len(branches))), constrained_layout=True)
    # 条件分岐: `len(branches) == 1` を満たす経路を評価する。
    if len(branches) == 1:
        axes = [axes]

    for ax, branch in zip(axes, branches):
        sub = df.loc[df["branch"].astype(str) == str(branch)].copy()
        sub = sub.sort_values(["scenario_group", "scenario_id"]).reset_index(drop=True)
        x = np.arange(len(sub), dtype=float)
        y = pd.to_numeric(sub["beta_dyn"], errors="coerce").to_numpy(dtype=float)
        s = pd.to_numeric(sub["beta_sigma"], errors="coerce").to_numpy(dtype=float)
        labels = sub["scenario_id"].astype(str).tolist()
        colors: List[str] = []
        for sid in labels:
            # 条件分岐: `sid == "baseline"` を満たす経路を評価する。
            if sid == "baseline":
                colors.append("#1f77b4")
            # 条件分岐: 前段条件が不成立で、`"station" in sid` を追加評価する。
            elif "station" in sid:
                colors.append("#ff7f0e")
            # 条件分岐: 前段条件が不成立で、`"sunq" in sid` を追加評価する。
            elif "sunq" in sid:
                colors.append("#2ca02c")
            # 条件分岐: 前段条件が不成立で、`"transponder" in sid` を追加評価する。
            elif "transponder" in sid:
                colors.append("#d62728")
            # 条件分岐: 前段条件が不成立で、`"plasma" in sid` を追加評価する。
            elif "plasma" in sid:
                colors.append("#9467bd")
            # 条件分岐: 前段条件が不成立で、`"srp" in sid` を追加評価する。
            elif "srp" in sid:
                colors.append("#8c564b")
            else:
                colors.append("#7f7f7f")

        ax.axhline(1.0, color="#7A7A7A", linestyle="--", linewidth=1.1)
        for i in range(len(sub)):
            ax.errorbar(x[i], y[i], yerr=s[i], fmt="o", capsize=3, color=colors[i], alpha=0.9)

        ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("beta_dyn")
        ax.set_title(f"{branch}: nuisance sensitivity scenarios")
        ax.grid(alpha=0.28)

    axes[0].set_title("Roadmap 8.7.48.9: Nuisance sensitivity audit (ODF/TNF)")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return None


# Function: Main entrypoint for roadmap step 8.7.48.9.

def main() -> int:
    ap = argparse.ArgumentParser(description="Roadmap 8.7.48.9: nuisance sensitivity audit.")
    ap.add_argument("--data-root", type=str, default=str(_ROOT / "data" / "mercury" / "messenger"))
    ap.add_argument("--out-dir", type=str, default=str(_ROOT / "output" / "private" / "mercury"))
    ap.add_argument("--public-dir", type=str, default=str(_ROOT / "output" / "public" / "mercury"))
    ap.add_argument("--odf-doppler-bin-minutes", type=int, default=60)
    ap.add_argument("--odf-range-bin-minutes", type=int, default=60)
    ap.add_argument("--tnf-doppler-bin-minutes", type=int, default=30)
    ap.add_argument("--tnf-range-bin-minutes", type=int, default=30)
    ap.add_argument("--odf-min-joint-rows", type=int, default=1000)
    ap.add_argument("--tnf-min-joint-rows", type=int, default=300)
    ap.add_argument("--sigma-watch-threshold", type=float, default=0.1)
    ap.add_argument("--orbital-period-days", type=float, default=87.9691)
    ap.add_argument("--base-station-bias-cap", type=int, default=8)
    ap.add_argument(
        "--beta-split-mode",
        type=str,
        choices=("coupled", "split"),
        default="coupled",
        help="Stage I nuisance fit beta mode. Use split for beta_dyn/beta_lt separated audit.",
    )
    ap.add_argument(
        "--core-scenario-ids",
        type=str,
        default="baseline,station_bias_lowcap,station_bias_highcap,sunq_proxy_on",
    )
    args = ap.parse_args()

    data_root = _resolve_path(args.data_root, _ROOT)
    out_dir = _resolve_path(args.out_dir, _ROOT)
    public_dir = _resolve_path(args.public_dir, _ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_summary_csv = out_dir / "messenger_beta_stage_i_nuisance_sensitivity_summary.csv"
    out_branch_csv = out_dir / "messenger_beta_stage_i_nuisance_sensitivity_branch_summary.csv"
    out_metrics_json = out_dir / "messenger_beta_stage_i_nuisance_sensitivity_metrics.json"
    out_plot_pdf = out_dir / "messenger_beta_stage_i_nuisance_sensitivity_audit.pdf"
    out_plot_png = out_dir / "messenger_beta_stage_i_nuisance_sensitivity_audit.png"

    # Branch definitions: same ODF/TNF split used in Stage H.
    branches = [
        {
            "branch": "odf",
            "doppler_csv": data_root / "derived" / "odf_doppler_observations.csv",
            "range_csv": data_root / "derived" / "odf_range_observations.csv",
            "doppler_bin": int(args.odf_doppler_bin_minutes),
            "range_bin": int(args.odf_range_bin_minutes),
            "min_rows": int(args.odf_min_joint_rows),
        },
        {
            "branch": "tnf",
            "doppler_csv": data_root / "derived" / "tnf_doppler_observations.csv",
            "range_csv": data_root / "derived" / "tnf_range_observations.csv",
            "doppler_bin": int(args.tnf_doppler_bin_minutes),
            "range_bin": int(args.tnf_range_bin_minutes),
            "min_rows": int(args.tnf_min_joint_rows),
        },
    ]

    scenarios = [
        {
            "scenario_id": "baseline",
            "scenario_group": "baseline",
            "station_bias_cap": int(args.base_station_bias_cap),
            "use_sun_quad_proxy": False,
            "use_transponder_quad": False,
            "use_plasma_proxy": False,
            "use_srp_proxy": False,
            "note": "stage_d_style_baseline",
        },
        {
            "scenario_id": "station_bias_lowcap",
            "scenario_group": "station_bias",
            "station_bias_cap": 2,
            "use_sun_quad_proxy": False,
            "use_transponder_quad": False,
            "use_plasma_proxy": False,
            "use_srp_proxy": False,
            "note": "station_bias_cap_2",
        },
        {
            "scenario_id": "station_bias_highcap",
            "scenario_group": "station_bias",
            "station_bias_cap": 12,
            "use_sun_quad_proxy": False,
            "use_transponder_quad": False,
            "use_plasma_proxy": False,
            "use_srp_proxy": False,
            "note": "station_bias_cap_12",
        },
        {
            "scenario_id": "sunq_proxy_on",
            "scenario_group": "sun_quadrupole",
            "station_bias_cap": int(args.base_station_bias_cap),
            "use_sun_quad_proxy": True,
            "use_transponder_quad": False,
            "use_plasma_proxy": False,
            "use_srp_proxy": False,
            "note": "add_sun_quadrupole_like_proxy",
        },
        {
            "scenario_id": "transponder_quad_on",
            "scenario_group": "transponder_drift",
            "station_bias_cap": int(args.base_station_bias_cap),
            "use_sun_quad_proxy": False,
            "use_transponder_quad": True,
            "use_plasma_proxy": False,
            "use_srp_proxy": False,
            "note": "add_quadratic_transponder_drift_proxy",
        },
        {
            "scenario_id": "plasma_proxy_on",
            "scenario_group": "plasma_residual",
            "station_bias_cap": int(args.base_station_bias_cap),
            "use_sun_quad_proxy": False,
            "use_transponder_quad": False,
            "use_plasma_proxy": True,
            "use_srp_proxy": False,
            "note": "add_annual_plasma_proxy",
        },
        {
            "scenario_id": "srp_proxy_on",
            "scenario_group": "srp_like",
            "station_bias_cap": int(args.base_station_bias_cap),
            "use_sun_quad_proxy": False,
            "use_transponder_quad": False,
            "use_plasma_proxy": False,
            "use_srp_proxy": True,
            "note": "add_srp_like_proxy_orbital_phase",
        },
        {
            "scenario_id": "all_proxies_on",
            "scenario_group": "combined",
            "station_bias_cap": int(args.base_station_bias_cap),
            "use_sun_quad_proxy": True,
            "use_transponder_quad": True,
            "use_plasma_proxy": True,
            "use_srp_proxy": True,
            "note": "combined_high_correlation_nuisance",
        },
    ]

    missing_inputs: List[str] = []
    for cfg in branches:
        # 条件分岐: `(not Path(cfg["doppler_csv"]).exists()) or (not Path(cfg["range_csv"]).exists())` を満たす経路を評価する。
        if (not Path(cfg["doppler_csv"]).exists()) or (not Path(cfg["range_csv"]).exists()):
            missing_inputs.append(str(cfg["branch"]))

    # 条件分岐: `len(missing_inputs) > 0` を満たす経路を評価する。

    if len(missing_inputs) > 0:
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "phase_step": "8.7.48.9",
            "overall_status": "reject",
            "reason": "missing_branch_inputs",
            "missing_branches": missing_inputs,
        }
        out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        synced = _sync_to_public([out_metrics_json], private_root=out_dir, public_root=public_dir)
        append_event(
            {
                "event": "run_script",
                "script": "scripts/mercury/messenger_beta_stage_i_nuisance_sensitivity.py",
                "phase_step": "8.7.48.9",
                "status": "reject",
                "input": str(data_root),
                "outputs": [_safe_rel(out_metrics_json, _ROOT)],
                "metrics": {"reason": "missing_branch_inputs", "missing": missing_inputs},
            }
        )
        print(f"[warn] Stage I skipped: missing inputs for branches={missing_inputs}")
        print(f"[ok] wrote: {out_metrics_json}")
        print(f"[ok] synced_to_public={len(synced)}")
        return 0

    rows: List[ScenarioRow] = []
    branch_meta: Dict[str, Dict[str, object]] = {}
    base_by_branch: Dict[str, Dict[str, float]] = {}

    for cfg in branches:
        branch = str(cfg["branch"])
        doppler = _load_branch_channel_csv(Path(cfg["doppler_csv"]), channel="doppler")
        rng = _load_branch_channel_csv(Path(cfg["range_csv"]), channel="range")
        doppler_agg = _aggregate_channel(doppler, bin_minutes=int(cfg["doppler_bin"]))
        range_agg = _aggregate_channel(rng, bin_minutes=int(cfg["range_bin"]))
        joint = pd.concat([range_agg, doppler_agg], ignore_index=True).sort_values("epoch_utc").reset_index(drop=True)
        branch_meta[branch] = {
            "n_rows_joint": int(len(joint)),
            "n_rows_range": int(np.sum(joint["channel"].astype(str).to_numpy() == "range")),
            "n_rows_doppler": int(np.sum(joint["channel"].astype(str).to_numpy() == "doppler")),
            "n_station_unique": int(joint["station_id"].astype(str).nunique()),
        }

        branch_rows: List[ScenarioRow] = []
        baseline_fit: Optional[Dict[str, object]] = None
        for sc in scenarios:
            fit = _run_scenario_fit(
                df_joint=joint,
                orbital_period_days=float(args.orbital_period_days),
                station_bias_cap=int(sc["station_bias_cap"]),
                min_joint_rows=int(cfg["min_rows"]),
                sigma_watch_threshold=float(args.sigma_watch_threshold),
                split_beta_lt=(str(args.beta_split_mode).strip().lower() == "split"),
                use_sun_quad_proxy=bool(sc["use_sun_quad_proxy"]),
                use_transponder_quad=bool(sc["use_transponder_quad"]),
                use_plasma_proxy=bool(sc["use_plasma_proxy"]),
                use_srp_proxy=bool(sc["use_srp_proxy"]),
            )
            # 条件分岐: `str(sc["scenario_id"]) == "baseline"` を満たす経路を評価する。
            if str(sc["scenario_id"]) == "baseline":
                baseline_fit = fit

            # 条件分岐: `baseline_fit is None` を満たす経路を評価する。

            if baseline_fit is None:
                z_delta = float("nan")
            else:
                z_delta = _z_delta_beta(
                    beta=float(fit["beta_dyn"]),
                    sigma=float(fit["beta_sigma"]),
                    beta_ref=float(baseline_fit["beta_dyn"]),
                    sigma_ref=float(baseline_fit["beta_sigma"]),
                )

            delta_status = _status_from_abs(z_delta, pass_thr=2.0, watch_thr=5.0)
            overall_scenario = _combine_status([str(fit["fit_status"]), str(delta_status)])
            row = ScenarioRow(
                branch=branch,
                scenario_id=str(sc["scenario_id"]),
                scenario_group=str(sc["scenario_group"]),
                station_bias_cap=int(sc["station_bias_cap"]),
                use_sun_quad_proxy=bool(sc["use_sun_quad_proxy"]),
                use_transponder_quad=bool(sc["use_transponder_quad"]),
                use_plasma_proxy=bool(sc["use_plasma_proxy"]),
                use_srp_proxy=bool(sc["use_srp_proxy"]),
                n_rows=int(fit["n_rows"]),
                n_range_rows=int(fit["n_range_rows"]),
                n_doppler_rows=int(fit["n_doppler_rows"]),
                n_params=int(fit["n_params"]),
                beta_dyn=float(fit["beta_dyn"]),
                beta_sigma=float(fit["beta_sigma"]),
                beta_z_from_1=float(fit["beta_z_from_1"]),
                beta_lt=float(fit["beta_lt"]),
                beta_lt_sigma=float(fit["beta_lt_sigma"]),
                beta_lt_z_from_1=float(fit["beta_lt_z_from_1"]),
                beta_split_mode=str(fit["beta_split_mode"]),
                beta_dyn_lt_consistency_status=str(fit["beta_dyn_lt_consistency_status"]),
                beta_dyn_lt_template_overlap=float(fit["beta_dyn_lt_template_overlap"]),
                beta_dyn_lt_overlap_status=str(fit["beta_dyn_lt_overlap_status"]),
                beta_dyn_lt_delta_status=str(fit["beta_dyn_lt_delta_status"]),
                z_delta_vs_base=float(z_delta),
                fit_status=str(fit["fit_status"]),
                sigma_status=str(fit["sigma_status"]),
                delta_status=str(delta_status),
                overall_status=str(overall_scenario),
                note=str(sc["note"]),
            )
            branch_rows.append(row)
            rows.append(row)

        # 条件分岐: `baseline_fit is None` を満たす経路を評価する。

        if baseline_fit is None:
            base_by_branch[branch] = {"beta_dyn": float("nan"), "beta_sigma": float("nan")}
        else:
            base_by_branch[branch] = {
                "beta_dyn": float(baseline_fit["beta_dyn"]),
                "beta_sigma": float(baseline_fit["beta_sigma"]),
            }

    core_scenario_ids = [s.strip() for s in str(args.core_scenario_ids).split(",") if s.strip()]
    branch_summary_rows = _build_branch_summary(rows, core_scenario_ids=core_scenario_ids)
    beta_dyn_decomposition = _build_beta_dyn_decomposition(rows)
    branch_status = {str(r.branch): str(r.status) for r in branch_summary_rows}
    overall = _combine_status([r.status for r in branch_summary_rows])
    _write_scenarios_csv(out_summary_csv, rows)
    _write_branch_summary_csv(out_branch_csv, branch_summary_rows)
    plot_note = _make_plot(rows=rows, out_pdf=out_plot_pdf, out_png=out_plot_png)

    produced: List[Path] = [out_summary_csv, out_branch_csv, out_metrics_json]
    # 条件分岐: `plot_note is None` を満たす経路を評価する。
    if plot_note is None:
        produced.extend([out_plot_pdf, out_plot_png])

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase_step": "8.7.48.9",
        "overall_status": overall,
        "branch_status": branch_status,
        "base_by_branch": base_by_branch,
        "branch_meta": branch_meta,
        "branch_summary": [r.__dict__ for r in branch_summary_rows],
        "beta_dyn_decomposition": beta_dyn_decomposition,
        "scenario_policy": {
            "delta_pass_abs_z": 2.0,
            "delta_watch_abs_z": 5.0,
            "sigma_watch_threshold": float(args.sigma_watch_threshold),
            "sigma_gate_role": "diagnostic_only",
            "beta_split_mode": str(args.beta_split_mode),
            "core_scenario_ids": core_scenario_ids,
            "diagnostic_scenario_ids": [
                str(sc["scenario_id"]) for sc in scenarios if str(sc["scenario_id"]) not in set(core_scenario_ids)
            ],
            "scenarios": scenarios,
        },
        "counts": {
            "n_rows": int(len(rows)),
            "n_branches": int(len(branches)),
            "n_scenarios_per_branch": int(len(scenarios)),
        },
        "plot": "generated" if plot_note is None else str(plot_note),
        "outputs_private": [_safe_rel(p, _ROOT) for p in produced if p != out_metrics_json],
    }
    out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    synced = _sync_to_public(produced, private_root=out_dir, public_root=public_dir)
    payload["outputs_public"] = [_safe_rel(p, _ROOT) for p in synced if p.name != out_metrics_json.name]
    out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _sync_to_public([out_metrics_json], private_root=out_dir, public_root=public_dir)

    append_event(
        {
            "event": "run_script",
            "script": "scripts/mercury/messenger_beta_stage_i_nuisance_sensitivity.py",
            "phase_step": "8.7.48.9",
            "status": overall,
            "input": str(data_root),
            "outputs": [_safe_rel(p, _ROOT) for p in produced],
            "metrics": {
                "branch_status": branch_status,
                "n_rows": int(len(rows)),
                "n_scenarios_per_branch": int(len(scenarios)),
            },
        }
    )

    print(f"[ok] stage_i_overall={overall}")
    print(f"[ok] branch_status={branch_status}")
    print(f"[ok] wrote: {out_summary_csv}")
    print(f"[ok] wrote: {out_metrics_json}")
    # 条件分岐: `plot_note is None` を満たす経路を評価する。
    if plot_note is None:
        print(f"[ok] wrote: {out_plot_pdf}")
        print(f"[ok] wrote: {out_plot_png}")
    else:
        print(f"[warn] plot skipped: {plot_note}")

    print(f"[ok] synced_to_public={len(synced)}")
    return 0


# Condition: Executes CLI main routine.

if __name__ == "__main__":
    raise SystemExit(main())
