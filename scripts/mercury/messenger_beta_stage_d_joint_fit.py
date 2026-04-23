#!/usr/bin/env python3
"""
messenger_beta_stage_d_joint_fit.py

Roadmap Step 8.7.48.4 (Stage D joint fit: range + Doppler) の実装。

目的:
- Stage B/C で生成した ODF 正規化観測を同時に扱い、beta_dyn（必要時は beta_lt も）
  と channel/station nuisance を joint に推定する。
- Stage E（TNF replay）の比較基準となる machine-readable 指標を固定する。

注意:
- 本実装は Stage D の最小I/F（pilot）であり、full ephemeris 置換前段。
- beta template は固定周期（Mercury 公転）を用いた感度確認であり、
  結果判定は原則 watch 上限とする。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from scripts.summary.worklog import append_event


# クラス: `JointFitResult` の責務と境界条件を定義する。
@dataclass
class JointFitResult:
    beta_dyn: float
    beta_sigma: float
    beta_z_from_1: float
    beta_lt: float
    beta_lt_sigma: float
    beta_lt_z_from_1: float
    beta_split_mode: str
    beta_dyn_lt_delta: float
    beta_dyn_lt_consistency_z: float
    beta_dyn_lt_consistency_status: str
    beta_dyn_lt_template_overlap: float
    beta_dyn_lt_overlap_status: str
    beta_dyn_lt_delta_status: str
    rss_norm: float
    dof: int
    n_rows: int
    n_range_rows: int
    n_doppler_rows: int
    rms_range: float
    rms_doppler: float
    status_data: str
    status_sigma: str
    status_model: str
    overall_status: str


# 関数: `_safe_rel` の入出力契約と処理意図を定義する。

def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


# 関数: `_resolve_path` の入出力契約と処理意図を定義する。

def _resolve_path(path_str: str, root: Path) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p

    return (root / p).resolve()


# 関数: `_parse_epoch_series` の入出力契約と処理意図を定義する。

def _parse_epoch_series(series: pd.Series) -> pd.Series:
    parsed_default = pd.to_datetime(series, utc=True, errors="coerce")
    nonnull_default = int(parsed_default.notna().sum())
    total = int(len(parsed_default))
    if total <= 0:
        return parsed_default

    if nonnull_default >= int(0.95 * total):
        return parsed_default

    parsed_best = parsed_default
    best_nonnull = nonnull_default
    for fmt in ("ISO8601", "mixed"):
        try:
            parsed_try = pd.to_datetime(series, utc=True, errors="coerce", format=fmt)
        except Exception:
            continue

        nonnull_try = int(parsed_try.notna().sum())
        if nonnull_try > best_nonnull:
            parsed_best = parsed_try
            best_nonnull = nonnull_try

    return parsed_best


# 関数: `_detect_column` の入出力契約と処理意図を定義する。

def _detect_column(columns: Sequence[str], keys: Sequence[str]) -> Optional[str]:
    lowers = {str(c).lower(): str(c) for c in columns}
    for key in keys:
        if str(key).lower() in lowers:
            return lowers[str(key).lower()]

    return None


# 関数: `_load_channel_csv` の入出力契約と処理意図を定義する。

def _load_channel_csv(path: Path, channel: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    epoch_col = _detect_column(df.columns.tolist(), ("epoch_utc", "time_utc", "epoch", "time", "utc"))
    if epoch_col is None:
        raise ValueError(f"{channel}: epoch column not found.")

    if str(channel) == "doppler":
        value_col = _detect_column(df.columns.tolist(), ("doppler_hz", "observable_value", "doppler"))
    else:
        value_col = _detect_column(df.columns.tolist(), ("range_value", "observable_value", "range", "range_obs"))

    if value_col is None:
        raise ValueError(f"{channel}: observable column not found.")

    work = pd.DataFrame()
    work["epoch_utc"] = _parse_epoch_series(df[epoch_col])
    work["observable_value"] = pd.to_numeric(df[value_col], errors="coerce")
    if "station_id" in df.columns:
        work["station_id"] = df["station_id"].astype(str)
    else:
        work["station_id"] = "unknown"

    work["channel"] = str(channel)
    work = work.dropna(subset=["epoch_utc", "observable_value"]).reset_index(drop=True)
    return work


# 関数: `_aggregate_channel` の入出力契約と処理意図を定義する。

def _aggregate_channel(df: pd.DataFrame, bin_minutes: int) -> pd.DataFrame:
    if int(bin_minutes) <= 0:
        return df.copy()

    work = df.copy()
    work["epoch_bin"] = work["epoch_utc"].dt.floor(f"{int(bin_minutes)}min")
    out = (
        work.groupby(["epoch_bin", "station_id", "channel"], as_index=False)
        .agg(observable_value=("observable_value", "median"))
        .sort_values("epoch_bin")
        .reset_index(drop=True)
    )
    out["epoch_utc"] = out["epoch_bin"]
    out = out.drop(columns=["epoch_bin"])
    return out


# 関数: `_robust_scale` の入出力契約と処理意図を定義する。

def _robust_scale(values: np.ndarray) -> float:
    if len(values) <= 0:
        return 1.0

    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad > 0.0:
        return float(1.4826 * mad)

    std = float(np.std(values))
    if std > 0.0:
        return std

    return 1.0


# 関数: `_select_top_station_ids` の入出力契約と処理意図を定義する。

def _select_top_station_ids(df: pd.DataFrame, channel: str, max_count: int) -> List[str]:
    sub = df.loc[df["channel"] == str(channel)].copy()
    if len(sub) <= 0:
        return []

    counts = (
        sub.groupby("station_id", as_index=False)
        .size()
        .rename(columns={"size": "n_rows"})
        .sort_values("n_rows", ascending=False)
        .reset_index(drop=True)
    )
    ids = counts["station_id"].astype(str).tolist()
    if int(max_count) > 0:
        ids = ids[: int(max_count)]

    return ids


# 関数: `_build_design_matrix` の入出力契約と処理意図を定義する。

def _build_design_matrix(
    df_joint: pd.DataFrame,
    orbital_period_days: float,
    max_station_bias_per_channel: int,
    split_beta_lt: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, object], pd.DataFrame]:
    work = df_joint.copy().sort_values("epoch_utc").reset_index(drop=True)
    t0 = work["epoch_utc"].iloc[0]
    t_days = (work["epoch_utc"] - t0).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    theta = 2.0 * np.pi * t_days / float(orbital_period_days)
    is_range = (work["channel"].to_numpy(dtype=str) == "range")
    is_doppler = ~is_range

    template_range = np.cos(theta)
    template_doppler = -np.sin(theta)
    if int(np.sum(is_range)) > 0:
        template_range[is_range] = template_range[is_range] - float(np.mean(template_range[is_range]))

    if int(np.sum(is_doppler)) > 0:
        template_doppler[is_doppler] = template_doppler[is_doppler] - float(np.mean(template_doppler[is_doppler]))

    if bool(split_beta_lt):
        # split時は beta_dyn を doppler 側、beta_lt を range 側へ分離して同定退化を回避する。
        beta_dyn_template = np.where(is_doppler, template_doppler, 0.0)
        beta_lt_template = np.where(is_range, template_range, 0.0)
    else:
        beta_dyn_template = np.where(is_range, template_range, template_doppler)
        beta_lt_template = np.where(is_range, template_range, 0.0)

    t_center = t_days - float(np.mean(t_days))
    y_obs = work["observable_value"].to_numpy(dtype=float)
    scale_range = _robust_scale(y_obs[is_range]) if int(np.sum(is_range)) > 0 else 1.0
    scale_doppler = _robust_scale(y_obs[is_doppler]) if int(np.sum(is_doppler)) > 0 else 1.0
    scale_by_row = np.where(is_range, float(scale_range), float(scale_doppler))
    y_norm = y_obs / scale_by_row

    c_intercept_range = np.where(is_range, 1.0, 0.0)
    c_intercept_dop = np.where(is_doppler, 1.0, 0.0)
    c_drift_range = np.where(is_range, t_center, 0.0)
    c_drift_dop = np.where(is_doppler, t_center, 0.0)

    if bool(split_beta_lt):
        design_cols = [
            beta_dyn_template,
            beta_lt_template,
            c_intercept_range,
            c_intercept_dop,
            c_drift_range,
            c_drift_dop,
        ]
        labels = [
            "beta_dyn_minus_1",
            "beta_lt_minus_1",
            "intercept_range_norm",
            "intercept_doppler_norm",
            "drift_range_norm_per_day",
            "drift_doppler_norm_per_day",
        ]
    else:
        design_cols = [
            beta_dyn_template,
            c_intercept_range,
            c_intercept_dop,
            c_drift_range,
            c_drift_dop,
        ]
        labels = [
            "beta_dyn_minus_1",
            "intercept_range_norm",
            "intercept_doppler_norm",
            "drift_range_norm_per_day",
            "drift_doppler_norm_per_day",
        ]

    station_ids_range = _select_top_station_ids(work, channel="range", max_count=int(max_station_bias_per_channel))
    station_ids_dop = _select_top_station_ids(work, channel="doppler", max_count=int(max_station_bias_per_channel))

    base_range = station_ids_range[0] if len(station_ids_range) > 0 else "unknown"
    base_dop = station_ids_dop[0] if len(station_ids_dop) > 0 else "unknown"

    for sid in station_ids_range[1:]:
        col = np.where(is_range & (work["station_id"].astype(str).to_numpy() == str(sid)), 1.0, 0.0)
        design_cols.append(col)
        labels.append(f"station_range_{sid}_norm")

    for sid in station_ids_dop[1:]:
        col = np.where(is_doppler & (work["station_id"].astype(str).to_numpy() == str(sid)), 1.0, 0.0)
        design_cols.append(col)
        labels.append(f"station_doppler_{sid}_norm")

    X = np.column_stack(design_cols)
    meta = {
        "t0_utc": pd.Timestamp(t0).isoformat(),
        "orbital_period_days": float(orbital_period_days),
        "scale_range": float(scale_range),
        "scale_doppler": float(scale_doppler),
        "station_ids_range_top": station_ids_range,
        "station_ids_doppler_top": station_ids_dop,
        "station_base_range": str(base_range),
        "station_base_doppler": str(base_dop),
        "beta_split_mode": "split" if bool(split_beta_lt) else "coupled",
        "beta_split_template_variant": "channel_separable" if bool(split_beta_lt) else "joint_template",
    }
    work["t_days"] = t_days
    work["beta_dyn_template"] = beta_dyn_template
    work["beta_lt_template"] = beta_lt_template
    work["scale_by_row"] = scale_by_row
    return (X, y_norm, y_obs, labels, meta, work)


# 関数: `_fit_joint` の入出力契約と処理意図を定義する。

def _fit_joint(
    X: np.ndarray,
    y_norm: np.ndarray,
    y_obs: np.ndarray,
    scale_by_row: np.ndarray,
    labels: Sequence[str],
    channels: np.ndarray,
    min_rows: int,
    sigma_watch_threshold: float,
) -> Tuple[JointFitResult, np.ndarray, np.ndarray, np.ndarray]:
    n_rows = int(X.shape[0])
    n_params = int(X.shape[1])
    n_range = int(np.sum(channels == "range"))
    n_dop = int(np.sum(channels == "doppler"))
    if n_rows < max(int(min_rows), n_params + 2):
        fit = JointFitResult(
            beta_dyn=float("nan"),
            beta_sigma=float("nan"),
            beta_z_from_1=float("nan"),
            beta_lt=float("nan"),
            beta_lt_sigma=float("nan"),
            beta_lt_z_from_1=float("nan"),
            beta_split_mode="split" if "beta_lt_minus_1" in labels else "coupled",
            beta_dyn_lt_delta=float("nan"),
            beta_dyn_lt_consistency_z=float("nan"),
            beta_dyn_lt_consistency_status="reject",
            beta_dyn_lt_template_overlap=float("nan"),
            beta_dyn_lt_overlap_status="reject",
            beta_dyn_lt_delta_status="reject",
            rss_norm=float("nan"),
            dof=max(0, n_rows - n_params),
            n_rows=n_rows,
            n_range_rows=n_range,
            n_doppler_rows=n_dop,
            rms_range=float("nan"),
            rms_doppler=float("nan"),
            status_data="reject",
            status_sigma="reject",
            status_model="watch",
            overall_status="reject",
        )
        return (fit, np.zeros(n_params), np.full_like(y_norm, np.nan), np.full_like(y_obs, np.nan))

    coef, _, _, _ = np.linalg.lstsq(X, y_norm, rcond=None)
    y_hat_norm = X @ coef
    residual_norm = y_norm - y_hat_norm
    rss_norm = float(np.sum(residual_norm**2))
    dof = int(max(1, n_rows - n_params))
    sigma2 = rss_norm / float(dof)
    cov = np.linalg.pinv(X.T @ X) * sigma2
    idx_beta_dyn = int(list(labels).index("beta_dyn_minus_1"))
    beta_delta_dyn = float(coef[idx_beta_dyn])
    beta_sigma_dyn = float(np.sqrt(max(0.0, float(cov[idx_beta_dyn, idx_beta_dyn]))))
    beta_dyn = 1.0 + beta_delta_dyn
    beta_z_dyn = float(abs(beta_dyn - 1.0) / beta_sigma_dyn) if beta_sigma_dyn > 0.0 else float("inf")

    if "beta_lt_minus_1" in labels:
        idx_beta_lt = int(list(labels).index("beta_lt_minus_1"))
        beta_delta_lt = float(coef[idx_beta_lt])
        beta_sigma_lt = float(np.sqrt(max(0.0, float(cov[idx_beta_lt, idx_beta_lt]))))
        beta_lt = 1.0 + beta_delta_lt
        beta_z_lt = float(abs(beta_lt - 1.0) / beta_sigma_lt) if beta_sigma_lt > 0.0 else float("inf")
        delta_dyn_lt = float(beta_dyn - beta_lt)
        denom_dyn_lt = float(math.sqrt(max(0.0, beta_sigma_dyn * beta_sigma_dyn + beta_sigma_lt * beta_sigma_lt)))
        z_dyn_lt = float(abs(delta_dyn_lt) / denom_dyn_lt) if denom_dyn_lt > 0.0 else float("inf")
        template_dyn = X[:, idx_beta_dyn]
        template_lt = X[:, idx_beta_lt]
        norm_dyn = float(np.linalg.norm(template_dyn))
        norm_lt = float(np.linalg.norm(template_lt))
        if norm_dyn > 0.0 and norm_lt > 0.0:
            overlap_dyn_lt = float(abs(float(np.dot(template_dyn, template_lt))) / (norm_dyn * norm_lt))
        else:
            overlap_dyn_lt = float("nan")

        if not math.isfinite(overlap_dyn_lt):
            status_overlap = "reject"
        elif overlap_dyn_lt <= 0.10:
            status_overlap = "pass"
        elif overlap_dyn_lt <= 0.35:
            status_overlap = "watch"
        else:
            status_overlap = "reject"

        if not math.isfinite(z_dyn_lt):
            status_delta = "reject"
        elif z_dyn_lt <= 2.0:
            status_delta = "pass"
        elif z_dyn_lt <= 5.0:
            status_delta = "watch"
        else:
            status_delta = "reject"

        if status_overlap == "reject":
            status_dyn_lt = "reject"
        elif status_overlap == "watch":
            status_dyn_lt = "watch"
        else:
            status_dyn_lt = "pass"

        split_mode = "split"
    else:
        beta_lt = float("nan")
        beta_sigma_lt = float("nan")
        beta_z_lt = float("nan")
        delta_dyn_lt = float("nan")
        z_dyn_lt = float("nan")
        overlap_dyn_lt = float("nan")
        status_overlap = "not_applicable"
        status_delta = "not_applicable"
        status_dyn_lt = "not_applicable"
        split_mode = "coupled"

    y_hat_obs = y_hat_norm * scale_by_row
    residual_obs = y_obs - y_hat_obs
    rms_range = float(np.sqrt(np.mean((residual_obs[channels == "range"]) ** 2))) if n_range > 0 else float("nan")
    rms_dop = float(np.sqrt(np.mean((residual_obs[channels == "doppler"]) ** 2))) if n_dop > 0 else float("nan")

    status_data = "pass"
    status_sigma = "pass" if beta_sigma_dyn <= float(sigma_watch_threshold) else "watch"
    status_model = "watch"
    overall = "watch"
    if status_data == "reject":
        overall = "reject"

    fit = JointFitResult(
        beta_dyn=beta_dyn,
        beta_sigma=beta_sigma_dyn,
        beta_z_from_1=beta_z_dyn,
        beta_lt=beta_lt,
        beta_lt_sigma=beta_sigma_lt,
        beta_lt_z_from_1=beta_z_lt,
        beta_split_mode=split_mode,
        beta_dyn_lt_delta=delta_dyn_lt,
        beta_dyn_lt_consistency_z=z_dyn_lt,
        beta_dyn_lt_consistency_status=status_dyn_lt,
        beta_dyn_lt_template_overlap=overlap_dyn_lt,
        beta_dyn_lt_overlap_status=status_overlap,
        beta_dyn_lt_delta_status=status_delta,
        rss_norm=rss_norm,
        dof=dof,
        n_rows=n_rows,
        n_range_rows=n_range,
        n_doppler_rows=n_dop,
        rms_range=rms_range,
        rms_doppler=rms_dop,
        status_data=status_data,
        status_sigma=status_sigma,
        status_model=status_model,
        overall_status=overall,
    )
    return (fit, coef, y_hat_norm, residual_norm)


# 関数: `_write_coeff_csv` の入出力契約と処理意図を定義する。

def _write_coeff_csv(path: Path, labels: Sequence[str], coef: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["parameter", "value"])
        writer.writeheader()
        for i, name in enumerate(labels):
            writer.writerow({"parameter": str(name), "value": float(coef[i])})


# 関数: `_make_plot` の入出力契約と処理意図を定義する。

def _make_plot(df: pd.DataFrame, out_pdf: Path, out_png: Path, sample_max: int = 60000) -> Optional[str]:
    if plt is None:
        return "matplotlib_unavailable"

    if len(df) <= 0:
        return "no_data"

    n = int(len(df))
    if n <= int(sample_max):
        idx = np.arange(n, dtype=int)
    else:
        idx = np.linspace(0, n - 1, int(sample_max), dtype=int)

    sub = df.iloc[idx].copy()
    t = pd.to_datetime(sub["epoch_utc"])
    ch = sub["channel"].astype(str)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.2, 7.4), sharex=True)
    for channel, color in (("range", "#1f77b4"), ("doppler", "#d62728")):
        mask = (ch == channel).to_numpy()
        if int(np.sum(mask)) <= 0:
            continue

        ax1.plot(t[mask], sub.loc[mask, "value_scaled"], ".", ms=1.2, alpha=0.35, color=color, label=f"{channel} obs")
        ax1.plot(
            t[mask],
            sub.loc[mask, "fit_scaled"],
            ".",
            ms=1.2,
            alpha=0.35,
            color=color,
            markeredgewidth=0,
            label=f"{channel} fit",
        )
        ax2.plot(t[mask], sub.loc[mask, "residual_norm"], ".", ms=1.2, alpha=0.4, color=color, label=f"{channel} resid")

    ax1.set_ylabel("Scaled observable")
    ax1.set_title("Roadmap 8.7.48.4: Stage D joint fit (range + doppler)")
    ax1.grid(alpha=0.22)
    ax1.legend(loc="upper right", fontsize=8)

    ax2.set_ylabel("Residual (norm)")
    ax2.set_xlabel("Epoch (UTC)")
    ax2.grid(alpha=0.22)
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return None


# 関数: `_sync_to_public` の入出力契約と処理意図を定義する。

def _sync_to_public(paths: Iterable[Path], private_root: Path, public_root: Path) -> List[Path]:
    public_root.mkdir(parents=True, exist_ok=True)
    synced: List[Path] = []
    for src in paths:
        try:
            rel = src.resolve().relative_to(private_root.resolve())
        except Exception:
            rel = Path(src.name)

        dst = public_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        synced.append(dst)

    return synced


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    ap = argparse.ArgumentParser(description="Roadmap 8.7.48.4: Stage D joint fit (range + doppler).")
    ap.add_argument(
        "--data-root",
        type=str,
        default=str(_ROOT / "data" / "mercury" / "messenger"),
        help="MESSENGER data root.",
    )
    ap.add_argument(
        "--doppler-csv",
        type=str,
        default="",
        help="Doppler CSV path; default <data-root>/derived/odf_doppler_observations.csv.",
    )
    ap.add_argument(
        "--range-csv",
        type=str,
        default="",
        help="Range CSV path; default <data-root>/derived/odf_range_observations.csv.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(_ROOT / "output" / "private" / "mercury"),
        help="Private output directory.",
    )
    ap.add_argument(
        "--public-dir",
        type=str,
        default=str(_ROOT / "output" / "public" / "mercury"),
        help="Public output directory.",
    )
    ap.add_argument("--doppler-bin-minutes", type=int, default=60, help="Doppler aggregation bin (minutes).")
    ap.add_argument("--range-bin-minutes", type=int, default=60, help="Range aggregation bin (minutes).")
    ap.add_argument("--min-joint-rows", type=int, default=1000, help="Minimum rows for joint fit gate.")
    ap.add_argument("--max-station-bias-per-channel", type=int, default=8, help="Max station bias columns per channel.")
    ap.add_argument("--orbital-period-days", type=float, default=87.9691, help="Fixed orbital period for beta template.")
    ap.add_argument("--sigma-watch-threshold", type=float, default=0.1, help="beta sigma watch threshold.")
    ap.add_argument(
        "--beta-split-mode",
        type=str,
        choices=("coupled", "split"),
        default="coupled",
        help="Beta parameterization mode. 'split' adds beta_lt_minus_1 term in Stage D.",
    )
    args = ap.parse_args()

    data_root = _resolve_path(args.data_root, _ROOT)
    doppler_csv = _resolve_path(args.doppler_csv, _ROOT) if str(args.doppler_csv).strip() else (
        data_root / "derived" / "odf_doppler_observations.csv"
    )
    range_csv = _resolve_path(args.range_csv, _ROOT) if str(args.range_csv).strip() else (
        data_root / "derived" / "odf_range_observations.csv"
    )
    out_dir = _resolve_path(args.out_dir, _ROOT)
    public_dir = _resolve_path(args.public_dir, _ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_summary_csv = out_dir / "messenger_beta_stage_d_joint_summary.csv"
    out_coeff_csv = out_dir / "messenger_beta_stage_d_joint_coefficients.csv"
    out_resid_csv = out_dir / "messenger_beta_stage_d_joint_residuals.csv"
    out_scale_csv = out_dir / "messenger_beta_stage_d_joint_channel_scales.csv"
    out_metrics_json = out_dir / "messenger_beta_stage_d_joint_metrics.json"
    out_plot_pdf = out_dir / "messenger_beta_stage_d_joint_fit.pdf"
    out_plot_png = out_dir / "messenger_beta_stage_d_joint_fit.png"

    if (not doppler_csv.exists()) or (not range_csv.exists()):
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "phase_step": "8.7.48.4",
            "overall_status": "reject",
            "reason": "joint_input_missing",
            "doppler_csv": _safe_rel(doppler_csv, _ROOT),
            "range_csv": _safe_rel(range_csv, _ROOT),
        }
        out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        synced = _sync_to_public([out_metrics_json], private_root=out_dir, public_root=public_dir)
        append_event(
            {
                "event": "run_script",
                "script": "scripts/mercury/messenger_beta_stage_d_joint_fit.py",
                "phase_step": "8.7.48.4",
                "status": "reject",
                "input": f"{_safe_rel(doppler_csv, _ROOT)}|{_safe_rel(range_csv, _ROOT)}",
                "outputs": [_safe_rel(out_metrics_json, _ROOT)],
                "metrics": {"reason": "joint_input_missing"},
            }
        )
        print("[warn] Stage D skipped: required inputs missing.")
        print(f"[ok] wrote: {out_metrics_json}")
        print(f"[ok] synced_to_public={len(synced)}")
        return 0

    doppler_df = _load_channel_csv(doppler_csv, channel="doppler")
    range_df = _load_channel_csv(range_csv, channel="range")
    doppler_agg = _aggregate_channel(doppler_df, bin_minutes=int(args.doppler_bin_minutes))
    range_agg = _aggregate_channel(range_df, bin_minutes=int(args.range_bin_minutes))
    joint_df = pd.concat([range_agg, doppler_agg], ignore_index=True).sort_values("epoch_utc").reset_index(drop=True)

    X, y_norm, y_obs, labels, meta, work = _build_design_matrix(
        joint_df,
        orbital_period_days=float(args.orbital_period_days),
        max_station_bias_per_channel=int(args.max_station_bias_per_channel),
        split_beta_lt=(str(args.beta_split_mode).strip().lower() == "split"),
    )
    channels = work["channel"].astype(str).to_numpy()
    fit, coef, fit_norm, residual_norm = _fit_joint(
        X=X,
        y_norm=y_norm,
        y_obs=y_obs,
        scale_by_row=work["scale_by_row"].to_numpy(dtype=float),
        labels=labels,
        channels=channels,
        min_rows=int(args.min_joint_rows),
        sigma_watch_threshold=float(args.sigma_watch_threshold),
    )

    work["fit_scaled"] = fit_norm
    work["value_scaled"] = y_norm
    work["residual_norm"] = residual_norm
    work.to_csv(out_resid_csv, index=False)
    _write_coeff_csv(out_coeff_csv, labels=labels, coef=coef)

    scale_df = pd.DataFrame(
        [
            {"channel": "range", "scale": float(meta.get("scale_range", 1.0))},
            {"channel": "doppler", "scale": float(meta.get("scale_doppler", 1.0))},
        ]
    )
    scale_df.to_csv(out_scale_csv, index=False)

    summary = pd.DataFrame(
        [
            {
                "phase_step": "8.7.48.4",
                "overall_status": fit.overall_status,
                "beta_dyn": fit.beta_dyn,
                "beta_sigma": fit.beta_sigma,
                "beta_z_from_1": fit.beta_z_from_1,
                "beta_lt": fit.beta_lt,
                "beta_lt_sigma": fit.beta_lt_sigma,
                "beta_lt_z_from_1": fit.beta_lt_z_from_1,
                "beta_split_mode": fit.beta_split_mode,
                "beta_dyn_lt_delta": fit.beta_dyn_lt_delta,
                "beta_dyn_lt_consistency_z": fit.beta_dyn_lt_consistency_z,
                "beta_dyn_lt_consistency_status": fit.beta_dyn_lt_consistency_status,
                "beta_dyn_lt_template_overlap": fit.beta_dyn_lt_template_overlap,
                "beta_dyn_lt_overlap_status": fit.beta_dyn_lt_overlap_status,
                "beta_dyn_lt_delta_status": fit.beta_dyn_lt_delta_status,
                "rss_norm": fit.rss_norm,
                "dof": fit.dof,
                "n_rows": fit.n_rows,
                "n_range_rows": fit.n_range_rows,
                "n_doppler_rows": fit.n_doppler_rows,
                "rms_range": fit.rms_range,
                "rms_doppler": fit.rms_doppler,
                "status_data": fit.status_data,
                "status_sigma": fit.status_sigma,
                "status_model": fit.status_model,
                "doppler_bin_minutes": int(args.doppler_bin_minutes),
                "range_bin_minutes": int(args.range_bin_minutes),
            }
        ]
    )
    summary.to_csv(out_summary_csv, index=False)

    plot_note = _make_plot(work, out_pdf=out_plot_pdf, out_png=out_plot_png)
    produced: List[Path] = [out_summary_csv, out_coeff_csv, out_resid_csv, out_scale_csv, out_metrics_json]
    if plot_note is None:
        produced.extend([out_plot_pdf, out_plot_png])

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase_step": "8.7.48.4",
        "overall_status": fit.overall_status,
        "data_root": _safe_rel(data_root, _ROOT),
        "doppler_csv": _safe_rel(doppler_csv, _ROOT),
        "range_csv": _safe_rel(range_csv, _ROOT),
        "n_rows_joint": fit.n_rows,
        "n_rows_range": fit.n_range_rows,
        "n_rows_doppler": fit.n_doppler_rows,
        "beta_dyn_estimate": fit.beta_dyn,
        "beta_sigma": fit.beta_sigma,
        "beta_z_from_1": fit.beta_z_from_1,
        "beta_lt_estimate": fit.beta_lt,
        "beta_lt_sigma": fit.beta_lt_sigma,
        "beta_lt_z_from_1": fit.beta_lt_z_from_1,
        "beta_split_mode": fit.beta_split_mode,
        "beta_dyn_lt_delta": fit.beta_dyn_lt_delta,
        "beta_dyn_lt_consistency_z": fit.beta_dyn_lt_consistency_z,
        "beta_dyn_lt_consistency_status": fit.beta_dyn_lt_consistency_status,
        "beta_dyn_lt_template_overlap": fit.beta_dyn_lt_template_overlap,
        "beta_dyn_lt_overlap_status": fit.beta_dyn_lt_overlap_status,
        "beta_dyn_lt_delta_status": fit.beta_dyn_lt_delta_status,
        "rss_norm": fit.rss_norm,
        "dof": int(fit.dof),
        "rms_range": fit.rms_range,
        "rms_doppler": fit.rms_doppler,
        "status_components": {
            "data": fit.status_data,
            "sigma": fit.status_sigma,
            "model": fit.status_model,
        },
        "joint_meta": meta,
        "gating_policy": {
            "min_joint_rows": int(args.min_joint_rows),
            "sigma_watch_threshold": float(args.sigma_watch_threshold),
            "model_status_cap": "watch_until_stage_e",
            "beta_split_mode": fit.beta_split_mode,
        },
        "plot": "generated" if plot_note is None else plot_note,
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
            "script": "scripts/mercury/messenger_beta_stage_d_joint_fit.py",
            "phase_step": "8.7.48.4",
            "status": fit.overall_status,
            "input": f"{_safe_rel(doppler_csv, _ROOT)}|{_safe_rel(range_csv, _ROOT)}",
            "outputs": [_safe_rel(p, _ROOT) for p in produced],
            "metrics": {
                "n_rows_joint": fit.n_rows,
                "beta_dyn": fit.beta_dyn,
                "beta_sigma": fit.beta_sigma,
                "beta_z_from_1": fit.beta_z_from_1,
                "beta_lt": fit.beta_lt,
                "beta_lt_sigma": fit.beta_lt_sigma,
                "beta_lt_z_from_1": fit.beta_lt_z_from_1,
                "beta_split_mode": fit.beta_split_mode,
                "beta_dyn_lt_delta": fit.beta_dyn_lt_delta,
                "beta_dyn_lt_consistency_z": fit.beta_dyn_lt_consistency_z,
                "beta_dyn_lt_consistency_status": fit.beta_dyn_lt_consistency_status,
                "beta_dyn_lt_template_overlap": fit.beta_dyn_lt_template_overlap,
                "beta_dyn_lt_overlap_status": fit.beta_dyn_lt_overlap_status,
                "beta_dyn_lt_delta_status": fit.beta_dyn_lt_delta_status,
                "rms_range": fit.rms_range,
                "rms_doppler": fit.rms_doppler,
            },
        }
    )

    print(f"[ok] stage_d_overall={fit.overall_status}")
    print(f"[ok] n_rows_joint={fit.n_rows} (range={fit.n_range_rows}, doppler={fit.n_doppler_rows})")
    print(f"[ok] beta_dyn={fit.beta_dyn:.8f} sigma={fit.beta_sigma:.8f} z={fit.beta_z_from_1:.4f}")
    if fit.beta_split_mode == "split":
        print(
            f"[ok] beta_lt={fit.beta_lt:.8f} sigma={fit.beta_lt_sigma:.8f} "
            f"z={fit.beta_lt_z_from_1:.4f} dyn_minus_lt={fit.beta_dyn_lt_delta:.8f} "
            f"status={fit.beta_dyn_lt_consistency_status} "
            f"overlap={fit.beta_dyn_lt_template_overlap:.4f} "
            f"overlap_status={fit.beta_dyn_lt_overlap_status} "
            f"delta_status={fit.beta_dyn_lt_delta_status}"
        )

    print(f"[ok] wrote: {out_summary_csv}")
    print(f"[ok] wrote: {out_metrics_json}")
    if plot_note is None:
        print(f"[ok] wrote: {out_plot_pdf}")
        print(f"[ok] wrote: {out_plot_png}")
    else:
        print(f"[warn] plot skipped: {plot_note}")

    print(f"[ok] synced_to_public={len(synced)}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
