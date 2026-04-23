#!/usr/bin/env python3
"""
messenger_beta_stage_c_range_global_pilot.py

Roadmap Step 8.7.48.3 (Stage C range-only global beta pilot) の実装。

目的:
- ODF 由来 range 観測を使い、global nuisance と beta_dyn を同時推定する
  Stage C の最小I/Fを固定する。
- Stage D（joint fit）へ接続するため、パラメータ感度と残差統計を
  machine-readable 出力で保存する。

注意:
- 本実装は Stage C の pilot であり、full ephemeris 数値積分の代替ではない。
- beta template は Mercury 公転周期の固定形（cos）を用いた感度確認モード。
  したがって判定は最大で watch とし、Pass は Stage D/E の統合後に判定する。
"""

from __future__ import annotations

import argparse
import csv
import json
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


# クラス: `FitResult` の責務と境界条件を定義する。
@dataclass
class FitResult:
    beta_dyn: float
    beta_sigma: float
    beta_z_from_1: float
    rss: float
    dof: int
    rms_observable: float
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


# 関数: `_detect_epoch_column` の入出力契約と処理意図を定義する。

def _detect_epoch_column(columns: Sequence[str]) -> Optional[str]:
    lowers = {c.lower(): c for c in columns}
    for key in ("epoch_utc", "time_utc", "epoch", "time", "utc"):
        if key in lowers:
            return lowers[key]

    return None


# 関数: `_detect_observable_column` の入出力契約と処理意図を定義する。

def _detect_observable_column(columns: Sequence[str]) -> Optional[str]:
    lowers = {c.lower(): c for c in columns}
    for key in ("range_value", "observable_value", "range_obs", "range_ns", "range"):
        if key in lowers:
            return lowers[key]

    return None


# 関数: `_select_range_dtype` の入出力契約と処理意図を定義する。

def _select_range_dtype(
    df: pd.DataFrame,
    prefer_dtype: int,
    min_dtype_rows: int,
) -> Tuple[pd.DataFrame, int, Dict[str, int]]:
    if "dtype_id" not in df.columns:
        return (df, -1, {})

    counts: Dict[str, int] = {}
    for dtype in sorted(df["dtype_id"].dropna().astype(int).unique().tolist()):
        n = int((df["dtype_id"].astype(int) == int(dtype)).sum())
        counts[str(int(dtype))] = n

    if str(int(prefer_dtype)) in counts and int(counts[str(int(prefer_dtype))]) >= int(min_dtype_rows):
        out = df.loc[df["dtype_id"].astype(int) == int(prefer_dtype)].copy()
        return (out, int(prefer_dtype), counts)

    if str(41) in counts and int(counts[str(41)]) >= int(min_dtype_rows):
        out = df.loc[df["dtype_id"].astype(int) == 41].copy()
        return (out, 41, counts)

    if str(37) in counts and int(counts[str(37)]) >= int(min_dtype_rows):
        out = df.loc[df["dtype_id"].astype(int) == 37].copy()
        return (out, 37, counts)

    if len(counts) <= 0:
        return (df, -1, counts)

    sorted_counts = sorted(counts.items(), key=lambda kv: int(kv[1]), reverse=True)
    selected_dtype = int(sorted_counts[0][0])
    out = df.loc[df["dtype_id"].astype(int) == selected_dtype].copy()
    return (out, selected_dtype, counts)


# 関数: `_aggregate_bins` の入出力契約と処理意図を定義する。

def _aggregate_bins(df: pd.DataFrame, bin_minutes: int) -> pd.DataFrame:
    if int(bin_minutes) <= 0:
        return df.copy()

    work = df.copy()
    work["epoch_bin"] = work["epoch_utc"].dt.floor(f"{int(bin_minutes)}min")
    g = (
        work.groupby(["epoch_bin", "station_id"], as_index=False)
        .agg(observable_value=("observable_value", "median"), dtype_id=("dtype_id", "median"))
        .sort_values("epoch_bin")
        .reset_index(drop=True)
    )
    g["epoch_utc"] = g["epoch_bin"]
    g = g.drop(columns=["epoch_bin"])
    return g


# 関数: `_build_design_matrix` の入出力契約と処理意図を定義する。

def _build_design_matrix(
    df: pd.DataFrame,
    orbital_period_days: float,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    work = df.copy().sort_values("epoch_utc").reset_index(drop=True)
    t0 = work["epoch_utc"].iloc[0]
    t_days = (work["epoch_utc"] - t0).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    y = work["observable_value"].to_numpy(dtype=float)

    beta_template = np.cos(2.0 * np.pi * t_days / float(orbital_period_days))
    beta_template = beta_template - float(np.mean(beta_template))
    drift = t_days - float(np.mean(t_days))

    station_ids = sorted(work["station_id"].astype(str).unique().tolist())
    station_base = station_ids[0] if len(station_ids) > 0 else "unknown"
    station_cols: List[np.ndarray] = []
    station_labels: List[str] = []
    for sid in station_ids[1:]:
        mask = (work["station_id"].astype(str).to_numpy() == str(sid)).astype(float)
        station_cols.append(mask)
        station_labels.append(f"station_bias_{sid}")

    design_cols = [np.ones_like(t_days), beta_template, drift]
    param_labels = ["intercept", "beta_dyn_minus_1", "drift_per_day"]
    if len(station_cols) > 0:
        design_cols.extend(station_cols)
        param_labels.extend(station_labels)

    X = np.column_stack(design_cols)
    work["t_days"] = t_days
    work["beta_template"] = beta_template
    work["station_base"] = station_base
    return (X, y, param_labels, work)


# 関数: `_fit_linear_model` の入出力契約と処理意図を定義する。

def _fit_linear_model(
    X: np.ndarray,
    y: np.ndarray,
    param_labels: Sequence[str],
    min_rows: int,
    sigma_watch_threshold: float,
) -> Tuple[FitResult, np.ndarray, np.ndarray]:
    n_rows = int(X.shape[0])
    n_params = int(X.shape[1])
    if n_rows < max(int(min_rows), n_params + 2):
        fit = FitResult(
            beta_dyn=float("nan"),
            beta_sigma=float("nan"),
            beta_z_from_1=float("nan"),
            rss=float("nan"),
            dof=max(0, n_rows - n_params),
            rms_observable=float("nan"),
            status_data="reject",
            status_sigma="reject",
            status_model="watch",
            overall_status="reject",
        )
        return (fit, np.zeros(n_params), np.full_like(y, np.nan, dtype=float))

    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef
    residual = y - y_hat
    rss = float(np.sum(residual**2))
    dof = int(max(1, n_rows - n_params))
    sigma2 = rss / float(dof)
    xtx = X.T @ X
    cov = np.linalg.pinv(xtx) * sigma2
    idx_beta = int(list(param_labels).index("beta_dyn_minus_1"))
    beta_delta = float(coef[idx_beta])
    beta_sigma = float(np.sqrt(max(0.0, float(cov[idx_beta, idx_beta]))))
    beta_dyn = 1.0 + beta_delta
    beta_z = float(abs(beta_dyn - 1.0) / beta_sigma) if beta_sigma > 0.0 else float("inf")
    rms_obs = float(np.sqrt(np.mean(residual**2)))

    status_data = "pass"
    status_sigma = "pass" if beta_sigma <= float(sigma_watch_threshold) else "watch"
    status_model = "watch"
    overall_status = "watch"
    if status_data == "reject":
        overall_status = "reject"

    fit = FitResult(
        beta_dyn=beta_dyn,
        beta_sigma=beta_sigma,
        beta_z_from_1=beta_z,
        rss=rss,
        dof=dof,
        rms_observable=rms_obs,
        status_data=status_data,
        status_sigma=status_sigma,
        status_model=status_model,
        overall_status=overall_status,
    )
    return (fit, coef, residual)


# 関数: `_write_coeff_csv` の入出力契約と処理意図を定義する。

def _write_coeff_csv(path: Path, param_labels: Sequence[str], coef: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["parameter", "value"])
        writer.writeheader()
        for i, name in enumerate(param_labels):
            writer.writerow({"parameter": str(name), "value": float(coef[i])})


# 関数: `_make_plot` の入出力契約と処理意図を定義する。

def _make_plot(
    fit_df: pd.DataFrame,
    residual: np.ndarray,
    out_pdf: Path,
    out_png: Path,
    sample_max: int = 30000,
) -> Optional[str]:
    if plt is None:
        return "matplotlib_unavailable"

    n = int(len(fit_df))
    if n <= 0:
        return "no_data"

    if n <= int(sample_max):
        idx = np.arange(n, dtype=int)
    else:
        idx = np.linspace(0, n - 1, int(sample_max), dtype=int)

    t = pd.to_datetime(fit_df["epoch_utc"]).iloc[idx]
    y = fit_df["observable_value"].to_numpy(dtype=float)[idx]
    y_hat = fit_df["fit_value"].to_numpy(dtype=float)[idx]
    r = residual[idx]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.0, 7.2), sharex=True)
    ax1.plot(t, y, ".", ms=1.0, alpha=0.35, color="#1f77b4", label="observed")
    ax1.plot(t, y_hat, ".", ms=1.0, alpha=0.35, color="#d62728", label="model")
    ax1.set_ylabel("Range observable")
    ax1.set_title("Roadmap 8.7.48.3: Stage C range-only global beta pilot")
    ax1.grid(alpha=0.22)
    ax1.legend(loc="upper right", fontsize=8)

    ax2.plot(t, r, ".", ms=1.0, alpha=0.4, color="#2ca02c")
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("Epoch (UTC)")
    ax2.grid(alpha=0.22)

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
    ap = argparse.ArgumentParser(description="Roadmap 8.7.48.3: MESSENGER Stage C range-only global beta pilot.")
    ap.add_argument(
        "--data-root",
        type=str,
        default=str(_ROOT / "data" / "mercury" / "messenger"),
        help="MESSENGER data root.",
    )
    ap.add_argument(
        "--range-csv",
        type=str,
        default="",
        help="Range observation CSV; default <data-root>/derived/odf_range_observations.csv.",
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
    ap.add_argument("--prefer-dtype", type=int, default=41, help="Preferred range dtype (41 or 37).")
    ap.add_argument("--min-dtype-rows", type=int, default=200, help="Minimum rows to keep preferred dtype.")
    ap.add_argument("--bin-minutes", type=int, default=60, help="Aggregation bin width in minutes.")
    ap.add_argument("--min-rows", type=int, default=500, help="Minimum rows for fit gate.")
    ap.add_argument(
        "--orbital-period-days",
        type=float,
        default=87.9691,
        help="Fixed orbital period for beta sensitivity template.",
    )
    ap.add_argument(
        "--sigma-watch-threshold",
        type=float,
        default=0.1,
        help="beta sigma threshold for pass/watch split (pilot gate).",
    )
    args = ap.parse_args()

    data_root = _resolve_path(args.data_root, _ROOT)
    out_dir = _resolve_path(args.out_dir, _ROOT)
    public_dir = _resolve_path(args.public_dir, _ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    range_csv = _resolve_path(args.range_csv, _ROOT) if str(args.range_csv).strip() else (
        data_root / "derived" / "odf_range_observations.csv"
    )

    out_station_csv = out_dir / "messenger_beta_stage_c_range_station_counts.csv"
    out_coeff_csv = out_dir / "messenger_beta_stage_c_range_coefficients.csv"
    out_resid_csv = out_dir / "messenger_beta_stage_c_range_residuals.csv"
    out_summary_csv = out_dir / "messenger_beta_stage_c_range_summary.csv"
    out_metrics_json = out_dir / "messenger_beta_stage_c_range_metrics.json"
    out_plot_pdf = out_dir / "messenger_beta_stage_c_range_global_pilot.pdf"
    out_plot_png = out_dir / "messenger_beta_stage_c_range_global_pilot.png"

    if not range_csv.exists():
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "phase_step": "8.7.48.3",
            "overall_status": "reject",
            "reason": "range_csv_missing",
            "range_csv": _safe_rel(range_csv, _ROOT),
            "next": "Run messenger_odf_to_doppler_csv.py --observable-mode range first.",
        }
        out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        synced = _sync_to_public([out_metrics_json], private_root=out_dir, public_root=public_dir)
        append_event(
            {
                "event": "run_script",
                "script": "scripts/mercury/messenger_beta_stage_c_range_global_pilot.py",
                "phase_step": "8.7.48.3",
                "status": "reject",
                "input": _safe_rel(range_csv, _ROOT),
                "outputs": [_safe_rel(out_metrics_json, _ROOT)],
                "metrics": {"reason": "range_csv_missing"},
            }
        )
        print("[warn] Stage C skipped: range CSV missing.")
        print(f"[ok] wrote: {out_metrics_json}")
        print(f"[ok] synced_to_public={len(synced)}")
        return 0

    df = pd.read_csv(range_csv)
    epoch_col = _detect_epoch_column(df.columns.tolist())
    obs_col = _detect_observable_column(df.columns.tolist())
    if epoch_col is None or obs_col is None:
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "phase_step": "8.7.48.3",
            "overall_status": "reject",
            "reason": "required_columns_missing",
            "range_csv": _safe_rel(range_csv, _ROOT),
            "columns": df.columns.tolist(),
        }
        out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        synced = _sync_to_public([out_metrics_json], private_root=out_dir, public_root=public_dir)
        append_event(
            {
                "event": "run_script",
                "script": "scripts/mercury/messenger_beta_stage_c_range_global_pilot.py",
                "phase_step": "8.7.48.3",
                "status": "reject",
                "input": _safe_rel(range_csv, _ROOT),
                "outputs": [_safe_rel(out_metrics_json, _ROOT)],
                "metrics": {"reason": "required_columns_missing"},
            }
        )
        print("[warn] Stage C parse failed: required columns missing.")
        print(f"[ok] wrote: {out_metrics_json}")
        print(f"[ok] synced_to_public={len(synced)}")
        return 0

    work = pd.DataFrame()
    work["epoch_utc"] = _parse_epoch_series(df[epoch_col])
    work["observable_value"] = pd.to_numeric(df[obs_col], errors="coerce")
    if "dtype_id" in df.columns:
        work["dtype_id"] = pd.to_numeric(df["dtype_id"], errors="coerce")
    else:
        work["dtype_id"] = -1

    if "station_id" in df.columns:
        work["station_id"] = df["station_id"].astype(str)
    else:
        work["station_id"] = "unknown"

    work = work.dropna(subset=["epoch_utc", "observable_value"]).reset_index(drop=True)
    work_selected, dtype_selected, dtype_counts = _select_range_dtype(
        work,
        prefer_dtype=int(args.prefer_dtype),
        min_dtype_rows=int(args.min_dtype_rows),
    )
    work_binned = _aggregate_bins(work_selected, bin_minutes=int(args.bin_minutes))
    station_counts = (
        work_binned.groupby("station_id", as_index=False)
        .size()
        .rename(columns={"size": "n_rows"})
        .sort_values("n_rows", ascending=False)
        .reset_index(drop=True)
    )
    station_counts.to_csv(out_station_csv, index=False)

    X, y, param_labels, fit_df = _build_design_matrix(
        work_binned,
        orbital_period_days=float(args.orbital_period_days),
    )
    fit, coef, residual = _fit_linear_model(
        X=X,
        y=y,
        param_labels=param_labels,
        min_rows=int(args.min_rows),
        sigma_watch_threshold=float(args.sigma_watch_threshold),
    )
    y_hat = X @ coef if len(coef) > 0 else np.full_like(y, np.nan, dtype=float)
    fit_df["fit_value"] = y_hat
    fit_df["residual"] = residual
    fit_df.to_csv(out_resid_csv, index=False)
    _write_coeff_csv(out_coeff_csv, param_labels=param_labels, coef=coef)

    summary = pd.DataFrame(
        [
            {
                "phase_step": "8.7.48.3",
                "overall_status": fit.overall_status,
                "beta_dyn": fit.beta_dyn,
                "beta_sigma": fit.beta_sigma,
                "beta_z_from_1": fit.beta_z_from_1,
                "rss": fit.rss,
                "dof": fit.dof,
                "rms_observable": fit.rms_observable,
                "status_data": fit.status_data,
                "status_sigma": fit.status_sigma,
                "status_model": fit.status_model,
                "dtype_selected": dtype_selected,
                "n_rows_raw": int(len(work)),
                "n_rows_selected": int(len(work_selected)),
                "n_rows_binned": int(len(work_binned)),
                "n_station": int(station_counts["station_id"].nunique()) if len(station_counts) > 0 else 0,
            }
        ]
    )
    summary.to_csv(out_summary_csv, index=False)

    plot_note = _make_plot(
        fit_df=fit_df,
        residual=residual,
        out_pdf=out_plot_pdf,
        out_png=out_plot_png,
    )
    produced: List[Path] = [
        out_station_csv,
        out_coeff_csv,
        out_resid_csv,
        out_summary_csv,
        out_metrics_json,
    ]
    if plot_note is None:
        produced.extend([out_plot_pdf, out_plot_png])

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase_step": "8.7.48.3",
        "overall_status": fit.overall_status,
        "data_root": _safe_rel(data_root, _ROOT),
        "range_csv": _safe_rel(range_csv, _ROOT),
        "dtype_selected": int(dtype_selected),
        "dtype_counts": dtype_counts,
        "n_rows_raw": int(len(work)),
        "n_rows_selected": int(len(work_selected)),
        "n_rows_binned": int(len(work_binned)),
        "n_station": int(station_counts["station_id"].nunique()) if len(station_counts) > 0 else 0,
        "beta_dyn_estimate": fit.beta_dyn,
        "beta_sigma": fit.beta_sigma,
        "beta_z_from_1": fit.beta_z_from_1,
        "rss": fit.rss,
        "dof": int(fit.dof),
        "rms_observable": fit.rms_observable,
        "status_components": {
            "data": fit.status_data,
            "sigma": fit.status_sigma,
            "model": fit.status_model,
        },
        "gating_policy": {
            "data_min_rows": int(args.min_rows),
            "sigma_watch_threshold": float(args.sigma_watch_threshold),
            "model_status_cap": "watch_until_stage_d",
        },
        "pilot_model": {
            "beta_template": "cos(2*pi*t/87.9691d) fixed-phase proxy",
            "notes": "Stage Dでfull forward modelへ置換予定",
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
            "script": "scripts/mercury/messenger_beta_stage_c_range_global_pilot.py",
            "phase_step": "8.7.48.3",
            "status": fit.overall_status,
            "input": _safe_rel(range_csv, _ROOT),
            "outputs": [_safe_rel(p, _ROOT) for p in produced],
            "metrics": {
                "dtype_selected": int(dtype_selected),
                "n_rows_binned": int(len(work_binned)),
                "beta_dyn": float(fit.beta_dyn),
                "beta_sigma": float(fit.beta_sigma),
                "beta_z_from_1": float(fit.beta_z_from_1),
            },
        }
    )

    print(f"[ok] stage_c_overall={fit.overall_status}")
    print(f"[ok] dtype_selected={dtype_selected}")
    print(f"[ok] n_rows_binned={len(work_binned)}")
    print(f"[ok] beta_dyn={fit.beta_dyn:.8f} sigma={fit.beta_sigma:.8f} z={fit.beta_z_from_1:.4f}")
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
