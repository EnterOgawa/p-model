#!/usr/bin/env python3
"""
llr_kappa_llr_injection_recovery_audit.py

Step 8.7.47 (LLR beta robustness roadmap; Step 1):
- Synthetic injection / recovery audit for kappa (beta-mapped amplitude).
- Reuses the same decontaminated fit core used in llr_kappa_llr_direct_fit.py.
- Preserves cadence / missingness / weight pattern from real LLR points.

Input:
- output/private/llr/batch/llr_batch_points.csv

Outputs (default: output/private/llr, synced to output/public/llr):
- llr_kappa_llr_injection_recovery_summary.csv
- llr_kappa_llr_injection_recovery_fitline.csv
- llr_kappa_llr_injection_recovery_metrics.json
- llr_kappa_llr_injection_recovery.pdf
- llr_kappa_llr_injection_recovery.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


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


# 関数: `_combine_status` の入出力契約と処理意図を定義する。

def _combine_status(values: Iterable[str]) -> str:
    norm = [str(v or "").strip().lower() for v in values if str(v or "").strip()]
    if not norm:
        return "reject"

    if any(v == "reject" for v in norm):
        return "reject"

    if all(v == "pass" for v in norm):
        return "pass"

    return "watch"


# 関数: `_parse_grid` の入出力契約と処理意図を定義する。

def _parse_grid(text: str) -> List[float]:
    vals: List[float] = []
    for tok in str(text).split(","):
        t = tok.strip()
        if not t:
            continue

        try:
            vals.append(float(t))
        except ValueError:
            continue

    return vals


# 関数: `_load_core_module` の入出力契約と処理意図を定義する。

def _load_core_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("llr_kappa_llr_direct_fit_core", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load core module spec: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# 関数: `_weighted_line_fit` の入出力契約と処理意図を定義する。

def _weighted_line_fit(x: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> Dict[str, float]:
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0.0)
    xv = x[ok]
    yv = y[ok]
    sv = sigma[ok]
    n = int(xv.size)
    if n < 2:
        return {
            "n_valid": n,
            "intercept": float("nan"),
            "slope": float("nan"),
            "intercept_sigma": float("nan"),
            "slope_sigma": float("nan"),
            "z_slope_minus_1": float("nan"),
            "z_intercept": float("nan"),
            "status": "reject",
        }

    w = 1.0 / np.square(sv)
    sw = float(np.sum(w))
    sx = float(np.sum(w * xv))
    sy = float(np.sum(w * yv))
    sxx = float(np.sum(w * xv * xv))
    sxy = float(np.sum(w * xv * yv))
    det = (sw * sxx) - (sx * sx)
    if not np.isfinite(det) or abs(det) <= 0.0:
        return {
            "n_valid": n,
            "intercept": float("nan"),
            "slope": float("nan"),
            "intercept_sigma": float("nan"),
            "slope_sigma": float("nan"),
            "z_slope_minus_1": float("nan"),
            "z_intercept": float("nan"),
            "status": "reject",
        }

    intercept = ((sxx * sy) - (sx * sxy)) / det
    slope = ((sw * sxy) - (sx * sy)) / det
    var_intercept = sxx / det
    var_slope = sw / det
    sig_intercept = math.sqrt(max(var_intercept, 0.0))
    sig_slope = math.sqrt(max(var_slope, 0.0))
    z_slope = (slope - 1.0) / sig_slope if sig_slope > 0.0 else float("nan")
    z_intercept = intercept / sig_intercept if sig_intercept > 0.0 else float("nan")
    status = _combine_status(
        [
            _status_from_abs_z(abs(z_slope)) if np.isfinite(z_slope) else "reject",
            _status_from_abs_z(abs(z_intercept)) if np.isfinite(z_intercept) else "reject",
        ]
    )
    return {
        "n_valid": n,
        "intercept": float(intercept),
        "slope": float(slope),
        "intercept_sigma": float(sig_intercept),
        "slope_sigma": float(sig_slope),
        "z_slope_minus_1": float(z_slope),
        "z_intercept": float(z_intercept),
        "status": status,
    }


# 関数: `_load_station_scatter` の入出力契約と処理意図を定義する。

def _load_station_scatter(primary_csv: Path, fallback_csv: Path) -> Dict[str, float]:
    csv_path = primary_csv if primary_csv.exists() else fallback_csv
    if not csv_path.exists():
        return {
            "path": "",
            "n_valid": 0,
            "std_kappa": float("nan"),
            "status": "watch",
        }

    df = pd.read_csv(csv_path)
    if "kappa_est" not in df.columns:
        return {
            "path": str(csv_path),
            "n_valid": 0,
            "std_kappa": float("nan"),
            "status": "watch",
        }

    vals = pd.to_numeric(df["kappa_est"], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        std = float("nan")
    else:
        std = float(np.nanstd(vals, ddof=1))

    return {
        "path": str(csv_path),
        "n_valid": int(vals.size),
        "std_kappa": std,
        "status": "pass" if np.isfinite(std) and std > 0.0 else "watch",
    }


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(rows_df: pd.DataFrame, fit: Dict[str, float], out_pdf: Path, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)
    ax0, ax1 = axes

    x = pd.to_numeric(rows_df["kappa_true"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(rows_df["kappa_recovered"], errors="coerce").to_numpy(dtype=float)
    s = pd.to_numeric(rows_df["kappa_recovered_sigma"], errors="coerce").to_numpy(dtype=float)
    err = pd.to_numeric(rows_df["kappa_error"], errors="coerce").to_numpy(dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x_ok = x[ok]
    y_ok = y[ok]
    s_ok = s[ok]
    err_ok = err[ok]

    ax0.errorbar(x_ok, y_ok, yerr=s_ok, fmt="o", capsize=4, color="#2A6EA6", label="recovered")
    if x_ok.size >= 2:
        xmin = float(np.nanmin(x_ok))
        xmax = float(np.nanmax(x_ok))
    else:
        xmin, xmax = 0.0, 2.0

    xx = np.linspace(xmin, xmax, 120)
    ax0.plot(xx, xx, linestyle="--", color="#7A7A7A", linewidth=1.2, label="ideal: y=x")
    if np.isfinite(float(fit.get("intercept", math.nan))) and np.isfinite(float(fit.get("slope", math.nan))):
        yy = float(fit["intercept"]) + (float(fit["slope"]) * xx)
        ax0.plot(xx, yy, color="#C23B22", linewidth=1.2, label="weighted fit")

    ax0.set_xlabel("injected kappa_true")
    ax0.set_ylabel("recovered kappa")
    ax0.set_title("Injection/Recovery Linearity")
    ax0.grid(alpha=0.3)
    ax0.legend(frameon=False, fontsize=9)

    ax1.axhline(0.0, linestyle="--", color="#7A7A7A", linewidth=1.2)
    ax1.errorbar(x_ok, err_ok, yerr=s_ok, fmt="o", capsize=4, color="#008A5E")
    ax1.set_xlabel("injected kappa_true")
    ax1.set_ylabel("kappa_recovered - kappa_true")
    ax1.set_title("Recovery Error")
    ax1.grid(alpha=0.3)

    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


# 関数: `_sync_to_public` の入出力契約と処理意図を定義する。

def _sync_to_public(paths: Iterable[Path], private_root: Path, public_root: Path) -> List[str]:
    synced: List[str] = []
    for src in paths:
        rel = src.resolve().relative_to(private_root.resolve())
        dst = public_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        synced.append(str(dst))

    return synced


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    ap = argparse.ArgumentParser(description="LLR kappa synthetic injection/recovery audit.")
    ap.add_argument(
        "--points-csv",
        type=str,
        default=str(ROOT / "output" / "private" / "llr" / "batch" / "llr_batch_points.csv"),
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "output" / "private" / "llr"),
    )
    ap.add_argument(
        "--public-dir",
        type=str,
        default=str(ROOT / "output" / "public" / "llr"),
    )
    ap.add_argument(
        "--core-script",
        type=str,
        default=str(ROOT / "scripts" / "llr" / "llr_kappa_llr_direct_fit.py"),
    )
    ap.add_argument("--kappa-grid", type=str, default="0.5,0.8,1.0,1.2,1.5")
    ap.add_argument("--fit-mode", type=str, default="station_target_year")
    ap.add_argument("--weight-scheme", type=str, default="inv_station_target")
    ap.add_argument("--weight-floor-station", type=int, default=180)
    ap.add_argument("--weight-floor-target", type=int, default=180)
    ap.add_argument("--weight-floor-station-target", type=int, default=120)
    ap.add_argument("--max-weight-cap", type=float, default=8.0)
    ap.add_argument("--min-orth-std", type=float, default=1e-6)
    args = ap.parse_args()

    points_csv = Path(str(args.points_csv))
    out_dir = Path(str(args.out_dir))
    public_dir = Path(str(args.public_dir))
    core_script = Path(str(args.core_script))
    if not points_csv.is_absolute():
        points_csv = (ROOT / points_csv).resolve()

    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()

    if not public_dir.is_absolute():
        public_dir = (ROOT / public_dir).resolve()

    if not core_script.is_absolute():
        core_script = (ROOT / core_script).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)

    grid = _parse_grid(str(args.kappa_grid))
    if not grid:
        raise RuntimeError("empty kappa grid")

    core = _load_core_module(core_script)
    df = core._read_points(points_csv)
    if df.empty:
        raise RuntimeError(f"no rows from {points_csv}")

    sample_weight = core._build_imbalance_weight(
        df,
        scheme=str(args.weight_scheme),
        floor_station=int(args.weight_floor_station),
        floor_target=int(args.weight_floor_target),
        floor_station_target=int(args.weight_floor_station_target),
        max_weight_cap=float(args.max_weight_cap),
    )
    _, base_summary = core._run_template_decontamination_audit(
        df=df,
        fit_mode=str(args.fit_mode),
        sample_weight=sample_weight,
        min_std=float(args.min_orth_std),
    )
    baseline_kappa = float(base_summary.get("decontaminated_kappa_est", float("nan")))
    baseline_sigma = float(base_summary.get("decontaminated_kappa_sigma", float("nan")))

    template = pd.to_numeric(df["dt_sun_shapiro_ns"], errors="coerce").to_numpy(dtype=float)
    residual = pd.to_numeric(df["residual_sr_tropo_tide_ns"], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(baseline_kappa) and np.isfinite(template).all() and np.isfinite(residual).all()):
        raise RuntimeError("invalid baseline/template/residual values")

    # Null residual keeps real cadence/noise/nuisance while removing the fitted Shapiro amplitude.

    residual_null = residual - (baseline_kappa * template)

    rows: List[Dict[str, Any]] = []
    for k_true in grid:
        df_syn = df.copy()
        df_syn["residual_sr_tropo_tide_ns"] = residual_null + (float(k_true) * template)
        _, syn_summary = core._run_template_decontamination_audit(
            df=df_syn,
            fit_mode=str(args.fit_mode),
            sample_weight=sample_weight,
            min_std=float(args.min_orth_std),
        )
        k_rec = float(syn_summary.get("decontaminated_kappa_est", float("nan")))
        k_sig = float(syn_summary.get("decontaminated_kappa_sigma", float("nan")))
        k_err = float(k_rec - float(k_true)) if np.isfinite(k_rec) else float("nan")
        z_err = float(k_err / k_sig) if np.isfinite(k_err) and np.isfinite(k_sig) and (k_sig > 0.0) else float("nan")
        rows.append(
            {
                "kappa_true": float(k_true),
                "kappa_recovered": k_rec,
                "kappa_recovered_sigma": k_sig,
                "kappa_error": k_err,
                "z_error": z_err,
                "status": _status_from_abs_z(abs(z_err)) if np.isfinite(z_err) else "reject",
            }
        )

    rows_df = pd.DataFrame(rows)
    k_true_arr = pd.to_numeric(rows_df["kappa_true"], errors="coerce").to_numpy(dtype=float)
    k_rec_arr = pd.to_numeric(rows_df["kappa_recovered"], errors="coerce").to_numpy(dtype=float)
    k_sig_arr = pd.to_numeric(rows_df["kappa_recovered_sigma"], errors="coerce").to_numpy(dtype=float)
    k_err_arr = pd.to_numeric(rows_df["kappa_error"], errors="coerce").to_numpy(dtype=float)
    fit_line = _weighted_line_fit(k_true_arr, k_rec_arr, k_sig_arr)

    eps_rms = float(np.sqrt(np.nanmean(np.square(k_err_arr)))) if np.isfinite(k_err_arr).any() else float("nan")
    eps_abs_max = float(np.nanmax(np.abs(k_err_arr))) if np.isfinite(k_err_arr).any() else float("nan")

    station_scatter = _load_station_scatter(
        primary_csv=out_dir / "llr_kappa_llr_station_stratified_refit.csv",
        fallback_csv=public_dir / "llr_kappa_llr_station_stratified_refit.csv",
    )
    std_station = float(station_scatter.get("std_kappa", float("nan")))
    eps_vs_station_ratio = float(eps_rms / std_station) if np.isfinite(eps_rms) and np.isfinite(std_station) and std_station > 0.0 else float("nan")
    eps_vs_station_status = "pass" if np.isfinite(eps_vs_station_ratio) and eps_vs_station_ratio <= 1.0 else ("watch" if np.isfinite(eps_vs_station_ratio) and eps_vs_station_ratio <= 2.0 else "reject")

    point_status = _combine_status(rows_df["status"].astype(str).tolist())
    linearity_status = str(fit_line.get("status", "reject"))
    overall = _combine_status([point_status, linearity_status, eps_vs_station_status])

    summary_csv = out_dir / "llr_kappa_llr_injection_recovery_summary.csv"
    fitline_csv = out_dir / "llr_kappa_llr_injection_recovery_fitline.csv"
    metrics_json = out_dir / "llr_kappa_llr_injection_recovery_metrics.json"
    plot_pdf = out_dir / "llr_kappa_llr_injection_recovery.pdf"
    plot_png = out_dir / "llr_kappa_llr_injection_recovery.png"

    rows_df.to_csv(summary_csv, index=False)
    pd.DataFrame([fit_line]).to_csv(fitline_csv, index=False)
    _plot(rows_df=rows_df, fit=fit_line, out_pdf=plot_pdf, out_png=plot_png)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": {"step": "8.7.47.step1_injection_recovery"},
        "input": {
            "points_csv": _safe_rel(points_csv, ROOT),
            "n_points": int(len(df)),
            "fit_mode": str(args.fit_mode),
            "weight_scheme": str(args.weight_scheme),
            "kappa_grid": [float(v) for v in grid],
        },
        "baseline": {
            "kappa_est": baseline_kappa,
            "kappa_sigma": baseline_sigma,
            "method": "template_decontaminated",
        },
        "recovery": {
            "rows": rows,
            "point_status": point_status,
            "linearity_fit": fit_line,
            "linearity_status": linearity_status,
            "epsilon_rms": eps_rms,
            "epsilon_abs_max": eps_abs_max,
            "station_scatter_reference": station_scatter,
            "epsilon_vs_station_ratio": eps_vs_station_ratio,
            "epsilon_vs_station_status": eps_vs_station_status,
            "overall_status": overall,
        },
        "outputs": {
            "summary_csv": _safe_rel(summary_csv, ROOT),
            "fitline_csv": _safe_rel(fitline_csv, ROOT),
            "metrics_json": _safe_rel(metrics_json, ROOT),
            "plot_pdf": _safe_rel(plot_pdf, ROOT),
            "plot_png": _safe_rel(plot_png, ROOT),
        },
    }

    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    synced = _sync_to_public(
        paths=[summary_csv, fitline_csv, metrics_json, plot_pdf, plot_png],
        private_root=out_dir,
        public_root=public_dir,
    )
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {fitline_csv}")
    print(f"Wrote: {metrics_json}")
    print(f"Wrote: {plot_pdf}")
    print(f"Wrote: {plot_png}")
    print(f"Synced: {len(synced)} files")
    print(f"Status: {overall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
