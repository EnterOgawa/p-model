#!/usr/bin/env python3
"""
vlbi_beta_source_filter_sensitivity.py

Compare VLBI beta nuisance-sensitivity outputs between source-filter settings.

Purpose:
- Quantify source-selection dependence (all sources vs selected near-Sun sources)
  under the same fitting pipeline.

Input:
- Two nuisance-sensitivity metrics JSON files produced by
  scripts/vlbi/vlbi_beta_nuisance_sensitivity.py

Output:
- output/vlbi/vlbi_<session>_beta_source_filter_sensitivity_summary.csv
- output/vlbi/vlbi_<session>_beta_source_filter_sensitivity_metrics.json
- output/vlbi/vlbi_<session>_beta_source_filter_sensitivity.pdf
- output/vlbi/vlbi_<session>_beta_source_filter_sensitivity.png
- Synced copies under output/public/vlbi/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


# Function: Resolve repository root from this script location.

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Function: Normalize labels for stable output filenames.

def _slugify(text: str) -> str:
    value = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip().lower())
    return value or "session"


# Function: Load nuisance sensitivity JSON and index rows by mode.

def _load_mode_map(path: Path) -> Dict[str, Dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows_raw = payload.get("rows", [])
    rows: Dict[str, Dict[str, float]] = {}
    if isinstance(rows_raw, list):
        for row in rows_raw:
            if not isinstance(row, dict):
                continue

            mode = str(row.get("mode") or "").strip()
            if not mode:
                continue

            rows[mode] = {
                "beta_est": float(row.get("beta_est", math.nan)),
                "beta_sigma": float(row.get("beta_sigma", math.nan)),
                "weighted_rmse_s": float(row.get("weighted_rmse_s", math.nan)),
            }

    return rows


# Function: Build mode-wise comparison records.

def _build_rows(
    selected_map: Dict[str, Dict[str, float]],
    all_map: Dict[str, Dict[str, float]],
) -> List[Dict[str, float | str]]:
    out: List[Dict[str, float | str]] = []
    shared_modes = sorted(set(selected_map.keys()) & set(all_map.keys()))
    for mode in shared_modes:
        sel = selected_map[mode]
        alls = all_map[mode]
        delta_beta = float(alls["beta_est"] - sel["beta_est"])
        sigma_comb = float(math.sqrt(max(0.0, (sel["beta_sigma"] ** 2) + (alls["beta_sigma"] ** 2))))
        z_abs = float(abs(delta_beta) / sigma_comb) if sigma_comb > 0.0 else math.nan
        out.append(
            {
                "mode": mode,
                "beta_selected": float(sel["beta_est"]),
                "beta_selected_sigma": float(sel["beta_sigma"]),
                "beta_all": float(alls["beta_est"]),
                "beta_all_sigma": float(alls["beta_sigma"]),
                "delta_beta_all_minus_selected": delta_beta,
                "sigma_combined": sigma_comb,
                "abs_z_delta_beta": z_abs,
                "wrmse_selected_s": float(sel["weighted_rmse_s"]),
                "wrmse_all_s": float(alls["weighted_rmse_s"]),
            }
        )

    return out


# Function: Determine a coarse status from best-mode filter dependence.

def _best_mode_status(best_row: Dict[str, float | str]) -> str:
    z_abs = float(best_row.get("abs_z_delta_beta", math.nan))
    if not math.isfinite(z_abs):
        return "watch"

    if z_abs <= 2.0:
        return "pass"

    if z_abs <= 3.0:
        return "watch"

    return "reject"


# Function: Generate a compact filter-sensitivity figure in vector PDF.

def _plot_rows(
    pdf_path: Path,
    png_path: Path,
    rows: List[Dict[str, float | str]],
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    if not rows:
        return

    modes = [str(r["mode"]) for r in rows]
    x = np.arange(len(modes), dtype=np.float64)
    beta_sel = np.asarray([float(r["beta_selected"]) for r in rows], dtype=np.float64)
    sig_sel = np.asarray([float(r["beta_selected_sigma"]) for r in rows], dtype=np.float64)
    beta_all = np.asarray([float(r["beta_all"]) for r in rows], dtype=np.float64)
    sig_all = np.asarray([float(r["beta_all_sigma"]) for r in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    ax.errorbar(x - 0.08, beta_sel, yerr=sig_sel, fmt="o", capsize=4, color="tab:blue", label="selected sources")
    ax.errorbar(x + 0.08, beta_all, yerr=sig_all, fmt="s", capsize=4, color="tab:orange", label="all sources")
    ax.axhline(1.0, color="tab:gray", linestyle="--", linewidth=1.2, label="beta=1")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("beta estimate")
    ax.set_title("VLBI beta source-filter sensitivity (all vs selected)")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(pdf_path))
    fig.savefig(str(png_path), dpi=170)
    plt.close(fig)


# Function: Copy generated artifacts to output/public/vlbi.

def _sync_public(root: Path, outputs: Sequence[Path]) -> None:
    dst = root / "output" / "public" / "vlbi"
    dst.mkdir(parents=True, exist_ok=True)
    for path in outputs:
        if path.exists():
            shutil.copy2(path, dst / path.name)


# Function: Main entrypoint for source-filter sensitivity comparison.

def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Compare VLBI beta sensitivity between source-filter settings.")
    ap.add_argument("--session", type=str, default="17MAY01XA", help="Base session label for output filenames.")
    ap.add_argument(
        "--selected-metrics",
        type=Path,
        default=root / "output" / "public" / "vlbi" / "vlbi_17may01xa_beta_nuisance_sensitivity_metrics.json",
        help="Metrics JSON for selected-source run.",
    )
    ap.add_argument(
        "--all-metrics",
        type=Path,
        default=root / "output" / "public" / "vlbi" / "vlbi_17may01xa_all_beta_nuisance_sensitivity_metrics.json",
        help="Metrics JSON for all-source run.",
    )
    ap.add_argument(
        "--best-mode",
        type=str,
        default="baseline_intercept_linear",
        help="Mode used for final watch/pass status evaluation.",
    )
    args = ap.parse_args()

    session = str(args.session).strip()
    session_slug = _slugify(session)
    selected_metrics = args.selected_metrics.resolve()
    all_metrics = args.all_metrics.resolve()
    if not selected_metrics.exists():
        raise FileNotFoundError(f"selected metrics not found: {selected_metrics}")

    if not all_metrics.exists():
        raise FileNotFoundError(f"all-source metrics not found: {all_metrics}")

    selected_map = _load_mode_map(selected_metrics)
    all_map = _load_mode_map(all_metrics)
    rows = _build_rows(selected_map=selected_map, all_map=all_map)
    if not rows:
        raise RuntimeError("no shared nuisance modes between selected and all metrics")

    best_mode = str(args.best_mode).strip()
    best_row = next((r for r in rows if str(r.get("mode")) == best_mode), rows[0])
    best_status = _best_mode_status(best_row)

    out_dir = root / "output" / "vlbi"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / f"vlbi_{session_slug}_beta_source_filter_sensitivity_summary.csv"
    metrics_json = out_dir / f"vlbi_{session_slug}_beta_source_filter_sensitivity_metrics.json"
    plot_pdf = out_dir / f"vlbi_{session_slug}_beta_source_filter_sensitivity.pdf"
    plot_png = out_dir / f"vlbi_{session_slug}_beta_source_filter_sensitivity.png"

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "mode",
                "beta_selected",
                "beta_selected_sigma",
                "beta_all",
                "beta_all_sigma",
                "delta_beta_all_minus_selected",
                "sigma_combined",
                "abs_z_delta_beta",
                "wrmse_selected_s",
                "wrmse_all_s",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    row["mode"],
                    f"{float(row['beta_selected']):.16e}",
                    f"{float(row['beta_selected_sigma']):.16e}",
                    f"{float(row['beta_all']):.16e}",
                    f"{float(row['beta_all_sigma']):.16e}",
                    f"{float(row['delta_beta_all_minus_selected']):.16e}",
                    f"{float(row['sigma_combined']):.16e}",
                    f"{float(row['abs_z_delta_beta']):.16e}",
                    f"{float(row['wrmse_selected_s']):.16e}",
                    f"{float(row['wrmse_all_s']):.16e}",
                ]
            )

    _plot_rows(pdf_path=plot_pdf, png_path=plot_png, rows=rows)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "session": session,
        "inputs": {
            "selected_metrics": str(selected_metrics),
            "all_metrics": str(all_metrics),
            "best_mode": best_mode,
        },
        "summary": {
            "shared_modes": [str(r["mode"]) for r in rows],
            "best_mode_status": best_status,
            "best_mode_abs_z_delta_beta": float(best_row["abs_z_delta_beta"]),
            "best_mode_delta_beta_all_minus_selected": float(best_row["delta_beta_all_minus_selected"]),
            "best_mode_sigma_combined": float(best_row["sigma_combined"]),
        },
        "rows": rows,
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

