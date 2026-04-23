#!/usr/bin/env python3
"""
vlbi_beta_nuisance_sensitivity.py

Run nuisance-model sensitivity checks for VLBI beta direct fit.

Purpose:
- Evaluate how beta estimates move across nuisance modes under the same
  primary-data configuration.

Input:
- Extracted vgosDb directory
- Optional source filter (comma separated)

Output:
- output/vlbi/vlbi_<session>_beta_nuisance_sensitivity_summary.csv
- output/vlbi/vlbi_<session>_beta_nuisance_sensitivity_metrics.json
- output/vlbi/vlbi_<session>_beta_nuisance_sensitivity.pdf
- output/vlbi/vlbi_<session>_beta_nuisance_sensitivity.png
- Synced copies under output/public/vlbi/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


DEFAULT_MODES = ["none", "baseline_intercept", "baseline_intercept_linear"]


# Function: Resolve repository root from this script location.

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Function: Detect whether the current figure surface is English-localized.
def _is_en_figure() -> bool:
    return str(os.getenv("WAVEP_FIGURE_LANG", "ja")).strip().lower().startswith("en")


# Function: Resolve public sync destination for the active locale.
def _public_output_dir(root: Path) -> Path:
    base = root / "output" / "public" / "vlbi"
    if _is_en_figure():
        return base / "locales" / "en"

    return base


# Function: Normalize labels for stable output filenames.

def _slugify(text: str) -> str:
    value = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip().lower())
    return value or "session"


# Function: Parse comma-separated nuisance mode list.

def _parse_modes(text: str) -> List[str]:
    tokens = [t.strip() for t in str(text).replace(";", ",").split(",")]
    out = [t for t in tokens if t]
    if not out:
        out = list(DEFAULT_MODES)

    return out


# Function: Run one direct-fit command for a specific nuisance mode.

def _run_direct_fit(
    root: Path,
    session_label: str,
    input_root: Path,
    nuisance_mode: str,
    source_include: str,
    observable_series: str,
    disable_iono_free: bool,
    uniform_weight: bool,
) -> Path:
    mode_slug = nuisance_mode.lower().replace("-", "_")
    session_mode = f"{session_label}_{mode_slug}"
    cmd = [
        sys.executable,
        "-B",
        str(root / "scripts" / "vlbi" / "vlbi_beta_direct_fit_from_vgosdb.py"),
        "--session",
        session_mode,
        "--input-root",
        str(input_root),
        "--nuisance-mode",
        nuisance_mode,
        "--observable-series",
        observable_series,
    ]
    if source_include.strip():
        cmd.extend(["--source-include", source_include.strip()])

    if disable_iono_free:
        cmd.append("--disable-iono-free")

    if uniform_weight:
        cmd.append("--uniform-weight")

    subprocess.run(cmd, check=True)
    metric_path = root / "output" / "vlbi" / f"vlbi_{_slugify(session_mode)}_beta_direct_fit_metrics.json"
    if not metric_path.exists():
        raise FileNotFoundError(f"metrics not found after run: {metric_path}")

    return metric_path


# Function: Compute AIC-like score from fit metadata for mode comparison.

def _aic_like(chi2: float, n_params: int) -> float:
    if not math.isfinite(chi2):
        return math.nan

    return float(chi2 + (2.0 * max(0, int(n_params))))


# Function: Generate a compact sensitivity figure in vector PDF.

def _plot_summary(
    pdf_path: Path,
    png_path: Path,
    modes: List[str],
    beta_est: np.ndarray,
    beta_sig: np.ndarray,
    wrmse_s: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    font_scale = 1.22 if _is_en_figure() else 1.0
    title_font = 14.0 * font_scale
    axis_font = 12.8 * font_scale
    tick_font = 11.4 * font_scale
    legend_font = 11.2 * font_scale
    x = np.arange(len(modes), dtype=np.float64)
    fig, ax0 = plt.subplots(figsize=(11.5, 6.8))
    ax0.errorbar(
        x,
        beta_est,
        yerr=beta_sig,
        fmt="o",
        capsize=4,
        linewidth=1.4,
        color="tab:blue",
        ecolor="tab:blue",
        label="beta estimate (+/-1sigma)",
    )
    ax0.axhline(1.0, color="tab:gray", linestyle="--", linewidth=1.2, label="beta=1")
    ax0.set_xticks(x)
    ax0.set_xticklabels(modes, rotation=0, fontsize=tick_font)
    ax0.set_ylabel("beta estimate", fontsize=axis_font)
    ax0.grid(True, alpha=0.28)
    ax0.set_title("VLBI beta direct-fit nuisance sensitivity", fontsize=title_font)
    ax1 = ax0.twinx()
    ax1.plot(x, wrmse_s * 1.0e12, "s-", color="tab:red", linewidth=1.5, label="weighted RMSE [ps]")
    ax1.set_ylabel("weighted RMSE [ps]", fontsize=axis_font)
    h0, l0 = ax0.get_legend_handles_labels()
    h1, l1 = ax1.get_legend_handles_labels()
    ax0.legend(h0 + h1, l0 + l1, loc="best", fontsize=legend_font)
    ax0.tick_params(labelsize=tick_font)
    ax1.tick_params(labelsize=tick_font)
    for axis in (ax0, ax1):
        for tick in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
            tick.set_fontsize(tick_font)

    fig.tight_layout()
    fig.savefig(str(pdf_path))
    fig.savefig(str(png_path), dpi=170)
    plt.close(fig)


# Function: Copy generated artifacts to output/public/vlbi.

def _sync_public(root: Path, outputs: Sequence[Path]) -> None:
    dst = _public_output_dir(root)
    dst.mkdir(parents=True, exist_ok=True)
    for path in outputs:
        if path.exists():
            shutil.copy2(path, dst / path.name)


# Function: Main entrypoint for nuisance sensitivity audit.

def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Run nuisance sensitivity checks for VLBI beta direct fit.")
    ap.add_argument("--session", type=str, default="17MAY01XA", help="Base session label for outputs.")
    ap.add_argument(
        "--input-root",
        type=Path,
        default=root / "data" / "vlbi" / "sources" / "vgosdb" / "17MAY01XA" / "extracted",
        help="Extracted vgosDb root directory.",
    )
    ap.add_argument(
        "--modes",
        type=str,
        default="none,baseline_intercept,baseline_intercept_linear",
        help="Comma-separated nuisance modes.",
    )
    ap.add_argument(
        "--source-include",
        type=str,
        default="",
        help="Comma-separated source allowlist passed through to direct-fit script.",
    )
    ap.add_argument(
        "--observable-series",
        type=str,
        default="full",
        choices=["full", "fringe"],
        help="Observable series passed through to direct-fit script.",
    )
    ap.add_argument("--disable-iono-free", action="store_true", help="Disable ionosphere-free combination.")
    ap.add_argument("--uniform-weight", action="store_true", help="Use uniform weights for all runs.")
    args = ap.parse_args()

    session = str(args.session).strip()
    session_slug = _slugify(session)
    input_root = args.input_root.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"input root not found: {input_root}")

    modes = _parse_modes(args.modes)
    rows: List[Dict[str, float | int | str]] = []
    metric_paths: Dict[str, str] = {}
    for mode in modes:
        metric_path = _run_direct_fit(
            root=root,
            session_label=session,
            input_root=input_root,
            nuisance_mode=mode,
            source_include=str(args.source_include),
            observable_series=str(args.observable_series),
            disable_iono_free=bool(args.disable_iono_free),
            uniform_weight=bool(args.uniform_weight),
        )
        metric_paths[mode] = str(metric_path)
        payload = json.loads(metric_path.read_text(encoding="utf-8"))
        fit = payload.get("fit_result", {}) if isinstance(payload.get("fit_result"), dict) else {}
        chi2 = float(fit.get("chi2", math.nan))
        n_params = int(fit.get("n_params", 0))
        row: Dict[str, float | int | str] = {
            "mode": mode,
            "n_points": int(fit.get("n_points", 0)),
            "n_params": n_params,
            "beta_est": float(fit.get("beta_est", math.nan)),
            "beta_sigma": float(fit.get("beta_sigma", math.nan)),
            "delta_beta": float(fit.get("delta_beta", math.nan)),
            "gamma_est": float(fit.get("gamma_est", math.nan)),
            "gamma_sigma": float(fit.get("gamma_sigma", math.nan)),
            "chi2": chi2,
            "dof": int(fit.get("dof", 0)),
            "weighted_rmse_s": float(fit.get("weighted_rmse_s", math.nan)),
            "aic_like": _aic_like(chi2=chi2, n_params=n_params),
        }
        rows.append(row)

    baseline_mode = "baseline_intercept" if "baseline_intercept" in modes else modes[0]
    aic_base = math.nan
    for row in rows:
        if str(row["mode"]) == baseline_mode:
            aic_base = float(row["aic_like"])
            break

    for row in rows:
        aic_value = float(row["aic_like"])
        row["delta_aic_base_minus_mode"] = (aic_base - aic_value) if (math.isfinite(aic_base) and math.isfinite(aic_value)) else math.nan

    beta_vals = np.asarray([float(row["beta_est"]) for row in rows], dtype=np.float64)
    beta_sig = np.asarray([float(row["beta_sigma"]) for row in rows], dtype=np.float64)
    wrmse = np.asarray([float(row["weighted_rmse_s"]) for row in rows], dtype=np.float64)
    beta_spread = float(np.nanmax(beta_vals) - np.nanmin(beta_vals)) if beta_vals.size > 0 else math.nan
    mode_best_rmse = str(rows[int(np.nanargmin(wrmse))]["mode"]) if wrmse.size > 0 else ""
    mode_best_aic = ""
    if len(rows) > 0:
        aic_arr = np.asarray([float(r["aic_like"]) for r in rows], dtype=np.float64)
        if np.isfinite(aic_arr).any():
            mode_best_aic = str(rows[int(np.nanargmin(aic_arr))]["mode"])

    out_dir = root / "output" / "vlbi"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / f"vlbi_{session_slug}_beta_nuisance_sensitivity_summary.csv"
    metrics_json = out_dir / f"vlbi_{session_slug}_beta_nuisance_sensitivity_metrics.json"
    plot_pdf = out_dir / f"vlbi_{session_slug}_beta_nuisance_sensitivity.pdf"
    plot_png = out_dir / f"vlbi_{session_slug}_beta_nuisance_sensitivity.png"

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "mode",
                "n_points",
                "n_params",
                "beta_est",
                "beta_sigma",
                "delta_beta",
                "gamma_est",
                "gamma_sigma",
                "chi2",
                "dof",
                "weighted_rmse_s",
                "aic_like",
                "delta_aic_base_minus_mode",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    row["mode"],
                    row["n_points"],
                    row["n_params"],
                    f"{float(row['beta_est']):.16e}",
                    f"{float(row['beta_sigma']):.16e}",
                    f"{float(row['delta_beta']):.16e}",
                    f"{float(row['gamma_est']):.16e}",
                    f"{float(row['gamma_sigma']):.16e}",
                    f"{float(row['chi2']):.16e}",
                    row["dof"],
                    f"{float(row['weighted_rmse_s']):.16e}",
                    f"{float(row['aic_like']):.16e}",
                    f"{float(row['delta_aic_base_minus_mode']):.16e}",
                ]
            )

    _plot_summary(
        pdf_path=plot_pdf,
        png_path=plot_png,
        modes=[str(r["mode"]) for r in rows],
        beta_est=beta_vals,
        beta_sig=beta_sig,
        wrmse_s=wrmse,
    )

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "session": session,
        "input_root": str(input_root),
        "modes": modes,
        "source_include": str(args.source_include),
        "observable_series": str(args.observable_series),
        "ionosphere_free_enabled": not bool(args.disable_iono_free),
        "uniform_weight_enabled": bool(args.uniform_weight),
        "baseline_mode_for_delta_aic": baseline_mode,
        "summary": {
            "beta_spread_abs": beta_spread,
            "mode_best_weighted_rmse": mode_best_rmse,
            "mode_best_aic_like": mode_best_aic,
        },
        "rows": rows,
        "mode_metric_paths": metric_paths,
        "outputs": {
            "summary_csv": str(summary_csv),
            "metrics_json": str(metrics_json),
            "plot_pdf": str(plot_pdf),
            "plot_png": str(plot_png),
        },
    }
    metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _sync_public(root, [summary_csv, metrics_json, plot_pdf, plot_png])
    print("Wrote:", summary_csv)
    print("Wrote:", metrics_json)
    print("Wrote:", plot_pdf)
    print("Wrote:", plot_png)
    print("Synced:", root / "output" / "public" / "vlbi")
    return 0


# Branch: Execute CLI entrypoint when this file is invoked directly.

if __name__ == "__main__":
    raise SystemExit(main())

