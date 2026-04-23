#!/usr/bin/env python3
"""
messenger_beta_stage_f_injection_recovery.py

Roadmap Step 8.7.48.6 (synthetic injection / recovery) の実装。

目的:
- Stage D/E と同一の joint-fit I/F（range + doppler）を使い、
  beta_true を注入したときの beta_recovered 回収性を定量監査する。
- cadence / missingness / station mix は実データのまま維持し、
  パイプライン線形性（slope / intercept）を gate 化する。
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

from scripts.mercury.messenger_beta_stage_d_joint_fit import (
    _aggregate_channel,
    _build_design_matrix,
    _fit_joint,
    _load_channel_csv,
    _sync_to_public,
)
from scripts.summary.worklog import append_event


# クラス: `InjectionRow` の責務と境界条件を定義する。
@dataclass
class InjectionRow:
    beta_true: float
    beta_recovered: float
    beta_sigma: float
    beta_error: float
    z_error: float
    fit_status: str


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


# 関数: `_parse_grid` の入出力契約と処理意図を定義する。

def _parse_grid(text: str) -> List[float]:
    vals: List[float] = []
    for tok in str(text).split(","):
        t = tok.strip()
        if len(t) <= 0:
            continue

        try:
            vals.append(float(t))
        except Exception:
            continue

    return vals


# 関数: `_combine_status` の入出力契約と処理意図を定義する。

def _combine_status(values: Iterable[str]) -> str:
    norm = [str(v or "").strip().lower() for v in values if str(v or "").strip()]
    if len(norm) <= 0:
        return "reject"

    if any(v == "reject" for v in norm):
        return "reject"

    if all(v == "pass" for v in norm):
        return "pass"

    return "watch"


# 関数: `_status_from_abs` の入出力契約と処理意図を定義する。

def _status_from_abs(value: float, pass_thr: float, watch_thr: float) -> str:
    if not math.isfinite(value):
        return "reject"

    if float(value) <= float(pass_thr):
        return "pass"

    if float(value) <= float(watch_thr):
        return "watch"

    return "reject"


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
    if (not math.isfinite(det)) or abs(det) <= 0.0:
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

    abs_slope_delta = abs(float(slope - 1.0))
    abs_intercept = abs(float(intercept))
    status = _combine_status(
        [
            _status_from_abs(abs_slope_delta, pass_thr=0.05, watch_thr=0.10),
            _status_from_abs(abs_intercept, pass_thr=0.05, watch_thr=0.10),
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
        "abs_slope_delta": abs_slope_delta,
        "abs_intercept": abs_intercept,
        "status": status,
    }


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(rows_df: pd.DataFrame, fit: Dict[str, float], out_pdf: Path, out_png: Path) -> Optional[str]:
    if plt is None:
        return "matplotlib_unavailable"

    if len(rows_df) <= 0:
        return "no_data"

    x = pd.to_numeric(rows_df["beta_true"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(rows_df["beta_recovered"], errors="coerce").to_numpy(dtype=float)
    s = pd.to_numeric(rows_df["beta_sigma"], errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(rows_df["beta_error"], errors="coerce").to_numpy(dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x_ok = x[ok]
    y_ok = y[ok]
    s_ok = s[ok]
    e_ok = e[ok]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.8, 5.4), constrained_layout=True)
    ax0.errorbar(x_ok, y_ok, yerr=s_ok, fmt="o", capsize=4, color="#2A6EA6", label="recovered")
    if len(x_ok) >= 2:
        xmin = float(np.nanmin(x_ok))
        xmax = float(np.nanmax(x_ok))
    else:
        xmin = 0.0
        xmax = 2.0

    xx = np.linspace(xmin, xmax, 120)
    ax0.plot(xx, xx, linestyle="--", color="#7A7A7A", linewidth=1.2, label="ideal: y=x")
    if np.isfinite(float(fit.get("intercept", math.nan))) and np.isfinite(float(fit.get("slope", math.nan))):
        yy = float(fit["intercept"]) + (float(fit["slope"]) * xx)
        ax0.plot(xx, yy, color="#C23B22", linewidth=1.2, label="weighted fit")

    ax0.set_xlabel("injected beta_true")
    ax0.set_ylabel("recovered beta")
    ax0.set_title("Roadmap 8.7.48.6: Injection/Recovery Linearity")
    ax0.grid(alpha=0.3)
    ax0.legend(frameon=False, fontsize=9)

    ax1.axhline(0.0, linestyle="--", color="#7A7A7A", linewidth=1.2)
    ax1.errorbar(x_ok, e_ok, yerr=s_ok, fmt="o", capsize=4, color="#008A5E")
    ax1.set_xlabel("injected beta_true")
    ax1.set_ylabel("beta_recovered - beta_true")
    ax1.set_title("Recovery Error")
    ax1.grid(alpha=0.3)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    return None


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    ap = argparse.ArgumentParser(description="Roadmap 8.7.48.6: synthetic injection/recovery audit for beta.")
    ap.add_argument("--data-root", type=str, default=str(_ROOT / "data" / "mercury" / "messenger"))
    ap.add_argument("--doppler-csv", type=str, default="")
    ap.add_argument("--range-csv", type=str, default="")
    ap.add_argument("--source-branch", type=str, default="tnf", choices=("tnf", "odf"))
    ap.add_argument("--out-dir", type=str, default=str(_ROOT / "output" / "private" / "mercury"))
    ap.add_argument("--public-dir", type=str, default=str(_ROOT / "output" / "public" / "mercury"))
    ap.add_argument("--doppler-bin-minutes", type=int, default=30)
    ap.add_argument("--range-bin-minutes", type=int, default=30)
    ap.add_argument("--min-joint-rows", type=int, default=300)
    ap.add_argument("--max-station-bias-per-channel", type=int, default=8)
    ap.add_argument("--orbital-period-days", type=float, default=87.9691)
    ap.add_argument("--sigma-watch-threshold", type=float, default=0.1)
    ap.add_argument("--beta-grid", type=str, default="0.0,0.5,1.0,1.5,2.0")
    args = ap.parse_args()

    data_root = _resolve_path(args.data_root, _ROOT)
    if str(args.doppler_csv).strip():
        doppler_csv = _resolve_path(args.doppler_csv, _ROOT)
    else:
        if str(args.source_branch) == "odf":
            doppler_csv = data_root / "derived" / "odf_doppler_observations.csv"
        else:
            doppler_csv = data_root / "derived" / "tnf_doppler_observations.csv"

    if str(args.range_csv).strip():
        range_csv = _resolve_path(args.range_csv, _ROOT)
    else:
        if str(args.source_branch) == "odf":
            range_csv = data_root / "derived" / "odf_range_observations.csv"
        else:
            range_csv = data_root / "derived" / "tnf_range_observations.csv"

    out_dir = _resolve_path(args.out_dir, _ROOT)
    public_dir = _resolve_path(args.public_dir, _ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_summary_csv = out_dir / "messenger_beta_stage_f_injection_recovery_summary.csv"
    out_fitline_csv = out_dir / "messenger_beta_stage_f_injection_recovery_fitline.csv"
    out_metrics_json = out_dir / "messenger_beta_stage_f_injection_recovery_metrics.json"
    out_plot_pdf = out_dir / "messenger_beta_stage_f_injection_recovery.pdf"
    out_plot_png = out_dir / "messenger_beta_stage_f_injection_recovery.png"

    if (not doppler_csv.exists()) or (not range_csv.exists()):
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "phase_step": "8.7.48.6",
            "overall_status": "reject",
            "reason": "input_missing",
            "source_branch": str(args.source_branch),
            "doppler_csv": _safe_rel(doppler_csv, _ROOT),
            "range_csv": _safe_rel(range_csv, _ROOT),
        }
        out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        synced = _sync_to_public([out_metrics_json], private_root=out_dir, public_root=public_dir)
        append_event(
            {
                "event": "run_script",
                "script": "scripts/mercury/messenger_beta_stage_f_injection_recovery.py",
                "phase_step": "8.7.48.6",
                "status": "reject",
                "input": f"{_safe_rel(doppler_csv, _ROOT)}|{_safe_rel(range_csv, _ROOT)}",
                "outputs": [_safe_rel(out_metrics_json, _ROOT)],
                "metrics": {"reason": "input_missing"},
            }
        )
        print("[warn] Stage F skipped: required input CSV missing.")
        print(f"[ok] wrote: {out_metrics_json}")
        print(f"[ok] synced_to_public={len(synced)}")
        return 0

    beta_grid = _parse_grid(args.beta_grid)
    if len(beta_grid) <= 0:
        beta_grid = [0.0, 0.5, 1.0, 1.5, 2.0]

    doppler_df = _load_channel_csv(doppler_csv, channel="doppler")
    range_df = _load_channel_csv(range_csv, channel="range")
    doppler_agg = _aggregate_channel(doppler_df, bin_minutes=int(args.doppler_bin_minutes))
    range_agg = _aggregate_channel(range_df, bin_minutes=int(args.range_bin_minutes))
    joint_df = pd.concat([range_agg, doppler_agg], ignore_index=True).sort_values("epoch_utc").reset_index(drop=True)

    X, y_norm, y_obs, labels, meta, work = _build_design_matrix(
        joint_df,
        orbital_period_days=float(args.orbital_period_days),
        max_station_bias_per_channel=int(args.max_station_bias_per_channel),
    )
    channels = work["channel"].astype(str).to_numpy()
    fit_base, coef_base, fit_norm_base, residual_base = _fit_joint(
        X=X,
        y_norm=y_norm,
        y_obs=y_obs,
        scale_by_row=work["scale_by_row"].to_numpy(dtype=float),
        labels=labels,
        channels=channels,
        min_rows=int(args.min_joint_rows),
        sigma_watch_threshold=float(args.sigma_watch_threshold),
    )
    if fit_base.status_data == "reject":
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "phase_step": "8.7.48.6",
            "overall_status": "reject",
            "reason": "base_fit_reject",
            "source_branch": str(args.source_branch),
            "n_rows_joint": int(X.shape[0]),
            "base_status": fit_base.status_data,
            "doppler_csv": _safe_rel(doppler_csv, _ROOT),
            "range_csv": _safe_rel(range_csv, _ROOT),
        }
        out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        synced = _sync_to_public([out_metrics_json], private_root=out_dir, public_root=public_dir)
        append_event(
            {
                "event": "run_script",
                "script": "scripts/mercury/messenger_beta_stage_f_injection_recovery.py",
                "phase_step": "8.7.48.6",
                "status": "reject",
                "input": f"{_safe_rel(doppler_csv, _ROOT)}|{_safe_rel(range_csv, _ROOT)}",
                "outputs": [_safe_rel(out_metrics_json, _ROOT)],
                "metrics": {"reason": "base_fit_reject"},
            }
        )
        print("[warn] Stage F skipped: base fit rejected for data gate.")
        print(f"[ok] wrote: {out_metrics_json}")
        print(f"[ok] synced_to_public={len(synced)}")
        return 0

    try:
        idx_beta = int(list(labels).index("beta_dyn_minus_1"))
    except Exception:
        idx_beta = 0

    rows: List[InjectionRow] = []
    scale_by_row = work["scale_by_row"].to_numpy(dtype=float)
    for beta_true in beta_grid:
        coef_true = np.array(coef_base, dtype=float)
        coef_true[idx_beta] = float(beta_true - 1.0)
        y_inj_norm = (X @ coef_true) + residual_base
        y_inj_obs = y_inj_norm * scale_by_row
        fit_i, _coef_i, _fit_norm_i, _residual_i = _fit_joint(
            X=X,
            y_norm=y_inj_norm,
            y_obs=y_inj_obs,
            scale_by_row=scale_by_row,
            labels=labels,
            channels=channels,
            min_rows=int(args.min_joint_rows),
            sigma_watch_threshold=float(args.sigma_watch_threshold),
        )
        recovered = float(fit_i.beta_dyn)
        sigma = float(fit_i.beta_sigma)
        err = float(recovered - beta_true) if math.isfinite(recovered) else float("nan")
        z = float(err / sigma) if (math.isfinite(err) and sigma > 0.0 and math.isfinite(sigma)) else float("nan")
        rows.append(
            InjectionRow(
                beta_true=float(beta_true),
                beta_recovered=recovered,
                beta_sigma=sigma,
                beta_error=err,
                z_error=z,
                fit_status=str(fit_i.overall_status),
            )
        )

    rows_df = pd.DataFrame(
        [
            {
                "beta_true": r.beta_true,
                "beta_recovered": r.beta_recovered,
                "beta_sigma": r.beta_sigma,
                "beta_error": r.beta_error,
                "z_error": r.z_error,
                "fit_status": r.fit_status,
            }
            for r in rows
        ]
    )
    rows_df.to_csv(out_summary_csv, index=False)

    fitline = _weighted_line_fit(
        x=pd.to_numeric(rows_df["beta_true"], errors="coerce").to_numpy(dtype=float),
        y=pd.to_numeric(rows_df["beta_recovered"], errors="coerce").to_numpy(dtype=float),
        sigma=pd.to_numeric(rows_df["beta_sigma"], errors="coerce").to_numpy(dtype=float),
    )

    max_abs_z = float(
        np.nanmax(np.abs(pd.to_numeric(rows_df["z_error"], errors="coerce").to_numpy(dtype=float)))
    ) if len(rows_df) > 0 else float("nan")
    status_point = _status_from_abs(max_abs_z, pass_thr=2.0, watch_thr=3.0)
    status_line = str(fitline.get("status", "reject"))
    overall = _combine_status([status_line, status_point])

    fitline_csv_df = pd.DataFrame(
        [
            {
                "phase_step": "8.7.48.6",
                "source_branch": str(args.source_branch),
                "n_valid": int(fitline.get("n_valid", 0)),
                "intercept": fitline.get("intercept"),
                "slope": fitline.get("slope"),
                "intercept_sigma": fitline.get("intercept_sigma"),
                "slope_sigma": fitline.get("slope_sigma"),
                "z_slope_minus_1": fitline.get("z_slope_minus_1"),
                "z_intercept": fitline.get("z_intercept"),
                "abs_slope_delta": fitline.get("abs_slope_delta"),
                "abs_intercept": fitline.get("abs_intercept"),
                "max_abs_z_error": max_abs_z,
                "status_linearity": status_line,
                "status_pointwise": status_point,
                "status_base_data": fit_base.status_data,
                "status_base_sigma": fit_base.status_sigma,
                "overall_status": overall,
            }
        ]
    )
    fitline_csv_df.to_csv(out_fitline_csv, index=False)

    plot_note = _plot(rows_df=rows_df, fit=fitline, out_pdf=out_plot_pdf, out_png=out_plot_png)
    produced: List[Path] = [out_summary_csv, out_fitline_csv, out_metrics_json]
    if plot_note is None:
        produced.extend([out_plot_pdf, out_plot_png])

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase_step": "8.7.48.6",
        "overall_status": overall,
        "source_branch": str(args.source_branch),
        "doppler_csv": _safe_rel(doppler_csv, _ROOT),
        "range_csv": _safe_rel(range_csv, _ROOT),
        "n_rows_joint": int(X.shape[0]),
        "n_rows_range": int(np.sum(channels == "range")),
        "n_rows_doppler": int(np.sum(channels == "doppler")),
        "beta_grid": [float(v) for v in beta_grid],
        "linearity_fit": fitline,
        "max_abs_z_error": max_abs_z,
        "status_components": {
            "linearity": status_line,
            "pointwise": status_point,
            "base_data": fit_base.status_data,
            "base_sigma": fit_base.status_sigma,
            "base_model": fit_base.status_model,
        },
        "base_fit": {
            "beta_dyn": fit_base.beta_dyn,
            "beta_sigma": fit_base.beta_sigma,
            "beta_z_from_1": fit_base.beta_z_from_1,
            "rss_norm": fit_base.rss_norm,
            "dof": int(fit_base.dof),
        },
        "joint_meta": meta,
        "gating_policy": {
            "linearity_abs_slope_delta_pass": 0.05,
            "linearity_abs_slope_delta_watch": 0.10,
            "linearity_abs_intercept_pass": 0.05,
            "linearity_abs_intercept_watch": 0.10,
            "pointwise_max_abs_z_pass": 2.0,
            "pointwise_max_abs_z_watch": 3.0,
            "min_joint_rows": int(args.min_joint_rows),
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
            "script": "scripts/mercury/messenger_beta_stage_f_injection_recovery.py",
            "phase_step": "8.7.48.6",
            "status": overall,
            "input": f"{_safe_rel(doppler_csv, _ROOT)}|{_safe_rel(range_csv, _ROOT)}",
            "outputs": [_safe_rel(p, _ROOT) for p in produced],
            "metrics": {
                "n_rows_joint": int(X.shape[0]),
                "linearity_slope": fitline.get("slope"),
                "linearity_intercept": fitline.get("intercept"),
                "max_abs_z_error": max_abs_z,
            },
        }
    )

    print(f"[ok] stage_f_overall={overall}")
    print(f"[ok] source_branch={args.source_branch} n_rows_joint={int(X.shape[0])}")
    print(
        "[ok] linearity slope={:.6f} intercept={:.6f} max|z|={:.4f}".format(
            float(fitline.get("slope", float("nan"))),
            float(fitline.get("intercept", float("nan"))),
            float(max_abs_z),
        )
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


if __name__ == "__main__":
    raise SystemExit(main())
