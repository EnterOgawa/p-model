#!/usr/bin/env python3
"""
vlbi_beta_high_sensitivity_factor_decomposition.py

Run high-sensitivity all-sky factor decomposition across sessions.

Purpose:
- Focus on sessions with strong solar gravity template amplitude
  (max|Cal-BendSun| above a threshold).
- Decompose beta variability drivers within each session using
  drop-one refits on:
  1) source
  2) baseline
  3) time quartile
- Aggregate per-session maxima for cross-session diagnostics.

Inputs:
- output/public/vlbi/vlbi_allsky_beta_consistency_summary.csv
- data/vlbi/sources/vgosdb/<SESSION>/extracted (primary vgosDb netCDF)

Outputs:
- output/vlbi/vlbi_high_sensitivity_factor_decomposition_summary.csv
- output/vlbi/vlbi_high_sensitivity_factor_decomposition_components.csv
- output/vlbi/vlbi_high_sensitivity_factor_decomposition_metrics.json
- output/vlbi/vlbi_high_sensitivity_factor_decomposition.pdf
- output/vlbi/vlbi_high_sensitivity_factor_decomposition.png
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
from typing import Dict, List, Optional, Sequence

import numpy as np

import vlbi_beta_source_filter_decomposition as decomp


# Function: Resolve repository root from this script location.
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Function: Read all-sky summary rows from CSV.

def _read_allsky_summary(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "session": str(row.get("session") or "").strip(),
                    "beta_est": float(row.get("beta_est", "nan")),
                    "beta_sigma": float(row.get("beta_sigma", "nan")),
                    "max_abs_bendsun_ns": float(row.get("max_abs_bendsun_ns", "nan")),
                    "n_points": int(float(row.get("n_points", "0"))),
                }
            )

    return rows


# Function: Score one drop-group impact against all-source baseline fit.

def _drop_impact_row(
    session: str,
    group_type: str,
    label: str,
    n_removed: int,
    fit_all: Dict[str, object],
    fit_drop: Optional[Dict[str, object]],
) -> Optional[Dict[str, object]]:
    if fit_drop is None:
        return None

    beta_all = float(fit_all["beta_est"])
    sig_all = float(fit_all["beta_sigma"])
    beta_drop = float(fit_drop["beta_est"])
    sig_drop = float(fit_drop["beta_sigma"])
    impact = float(beta_all - beta_drop)
    sigma_comb = float(math.sqrt(max(0.0, (sig_all * sig_all) + (sig_drop * sig_drop))))
    abs_z = float(abs(impact) / sigma_comb) if sigma_comb > 0.0 else math.nan
    return {
        "session": session,
        "group_type": group_type,
        "group_label": label,
        "n_removed": int(n_removed),
        "beta_all": beta_all,
        "beta_all_sigma": sig_all,
        "beta_drop": beta_drop,
        "beta_drop_sigma": sig_drop,
        "impact_beta_all_minus_drop": impact,
        "sigma_combined": sigma_comb,
        "abs_z_impact": abs_z,
    }


# Function: Select top labels by removable point count.

def _top_labels(values: np.ndarray, mask: np.ndarray, limit: int) -> List[str]:
    vm = np.asarray(values, dtype=object)
    mm = np.asarray(mask, dtype=bool)
    if int(np.sum(mm)) <= 0:
        return []

    uniq, cnt = np.unique(np.asarray([str(v) for v in vm[mm]], dtype=object), return_counts=True)
    order = np.argsort(-cnt.astype(np.int64))
    out: List[str] = []
    for idx in order[: int(max(1, limit))]:
        out.append(str(uniq[idx]))

    return out


# Function: Compute weighted-mean consistency for session beta values.

def _weighted_consistency(beta: np.ndarray, sigma: np.ndarray) -> Dict[str, float]:
    b = np.asarray(beta, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    mask = np.isfinite(b) & np.isfinite(s) & (s > 0.0)
    n = int(np.sum(mask))
    if n < 2:
        return {
            "n_valid": n,
            "beta_weighted_mean": math.nan,
            "beta_weighted_sigma": math.nan,
            "chi2": math.nan,
            "dof": math.nan,
            "chi2_dof": math.nan,
        }

    b = b[mask]
    s = s[mask]
    w = 1.0 / np.square(s)
    wsum = float(np.sum(w))
    bbar = float(np.sum(w * b) / wsum)
    sig_bar = float(math.sqrt(1.0 / wsum))
    chi2 = float(np.sum(np.square((b - bbar) / s)))
    dof = float(max(1, int(b.size - 1)))
    chi2_dof = float(chi2 / dof)
    return {
        "n_valid": int(b.size),
        "beta_weighted_mean": bbar,
        "beta_weighted_sigma": sig_bar,
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": chi2_dof,
    }


# Function: Render compact decomposition summary figure.

def _plot_summary(
    pdf_path: Path,
    png_path: Path,
    summary_rows: List[Dict[str, object]],
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    if not summary_rows:
        return

    labels = [str(r["session"]) for r in summary_rows]
    src = np.asarray([float(r.get("max_abs_z_source", math.nan)) for r in summary_rows], dtype=np.float64)
    bl = np.asarray([float(r.get("max_abs_z_baseline", math.nan)) for r in summary_rows], dtype=np.float64)
    tm = np.asarray([float(r.get("max_abs_z_time_quartile", math.nan)) for r in summary_rows], dtype=np.float64)
    sens = np.asarray([float(r.get("max_abs_bendsun_ns", math.nan)) for r in summary_rows], dtype=np.float64)
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.26

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13.2, 8.8), gridspec_kw={"height_ratios": [2.0, 1.2]})
    ax0.bar(x - width, src, width=width, label="source", color="tab:blue", alpha=0.9)
    ax0.bar(x, bl, width=width, label="baseline", color="tab:orange", alpha=0.9)
    ax0.bar(x + width, tm, width=width, label="time_quartile", color="tab:green", alpha=0.9)
    ax0.axhline(2.0, color="tab:gray", linestyle="--", linewidth=1.1, label="|z|=2 gate")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=35, ha="right")
    ax0.set_ylabel("max |z impact|")
    ax0.set_title("High-sensitivity VLBI factor decomposition (drop-one)")
    ax0.grid(True, axis="y", alpha=0.28)
    ax0.legend(loc="best")

    ax1.bar(x, sens, color="tab:red", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.set_ylabel("max |Cal-BendSun| [ns]")
    ax1.set_xlabel("Session")
    ax1.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(pdf_path))
    fig.savefig(str(png_path), dpi=170)
    plt.close(fig)


# Function: Copy generated artifacts to output/public/vlbi.

def _sync_public(root: Path, outputs: Sequence[Path]) -> None:
    dst = root / "output" / "public" / "vlbi"
    dst.mkdir(parents=True, exist_ok=True)
    for p in outputs:
        if p.exists():
            shutil.copy2(p, dst / p.name)


# Function: Main entrypoint for high-sensitivity factor decomposition.

def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="High-sensitivity VLBI factor decomposition across sessions.")
    ap.add_argument(
        "--allsky-summary",
        type=Path,
        default=root / "output" / "public" / "vlbi" / "vlbi_allsky_beta_consistency_summary.csv",
        help="All-sky session summary CSV.",
    )
    ap.add_argument(
        "--session-root",
        type=Path,
        default=root / "data" / "vlbi" / "sources" / "vgosdb",
        help="Root containing per-session extracted vgosDb data.",
    )
    ap.add_argument(
        "--min-sensitivity-ns",
        type=float,
        default=10.0,
        help="Minimum max|Cal-BendSun| [ns] to include a session.",
    )
    ap.add_argument(
        "--nuisance-mode",
        type=str,
        default="baseline_intercept_linear",
        choices=["none", "baseline_intercept", "baseline_intercept_linear"],
        help="Nuisance mode used for all refits.",
    )
    ap.add_argument(
        "--observable-series",
        type=str,
        default="full",
        choices=["full", "fringe"],
        help="Observable series used in decomposition reconstruction.",
    )
    ap.add_argument(
        "--min-template-abs",
        type=float,
        default=1.0e-14,
        help="Absolute threshold on |Cal-BendSun template| [s].",
    )
    ap.add_argument(
        "--top-sources",
        type=int,
        default=12,
        help="Maximum number of source groups per session.",
    )
    ap.add_argument(
        "--top-baselines",
        type=int,
        default=12,
        help="Maximum number of baseline groups per session.",
    )
    args = ap.parse_args()

    allsky_rows = _read_allsky_summary(args.allsky_summary.resolve())
    if not allsky_rows:
        raise FileNotFoundError(f"all-sky summary not found or empty: {args.allsky_summary}")

    selected = [
        r
        for r in allsky_rows
        if math.isfinite(float(r.get("max_abs_bendsun_ns", math.nan)))
        and float(r.get("max_abs_bendsun_ns", math.nan)) >= float(args.min_sensitivity_ns)
    ]
    if not selected:
        raise RuntimeError(
            f"no session satisfies min-sensitivity-ns={args.min_sensitivity_ns} "
            f"in {args.allsky_summary}"
        )

    session_root = args.session_root.resolve()
    summary_rows: List[Dict[str, object]] = []
    component_rows: List[Dict[str, object]] = []
    for row in selected:
        session = str(row["session"])
        input_root = session_root / session / "extracted"
        if not input_root.exists():
            continue

        index = decomp.core._scan_netcdf_variables(input_root)
        if not index:
            continue

        prepared = decomp._prepare_vectors(
            index=index,
            input_root=input_root,
            band_index=0,
            observable_series=str(args.observable_series),
            threshold_s=float(args.min_template_abs),
            disable_flag_filter=False,
        )
        base_mask = np.asarray(prepared["mask_base"], dtype=bool)
        if int(np.sum(base_mask)) < 3:
            continue

        fit_all = decomp._fit_with_mask(prepared=prepared, mask=base_mask, nuisance_mode=str(args.nuisance_mode))
        if fit_all is None:
            continue

        source_vec = np.asarray(prepared["source_vec"], dtype=object)
        baseline_vec = np.asarray(prepared["baseline_vec"], dtype=object)
        time_seconds = np.asarray(prepared["time_seconds"], dtype=np.float64)
        source_rows: List[Dict[str, object]] = []
        baseline_rows: List[Dict[str, object]] = []
        time_rows: List[Dict[str, object]] = []

        for src in _top_labels(source_vec, base_mask, int(args.top_sources)):
            rm_mask = base_mask & (np.asarray([str(v) for v in source_vec], dtype=object) == str(src))
            n_removed = int(np.sum(rm_mask))
            if n_removed < 3:
                continue

            fit_drop = decomp._fit_with_mask(
                prepared=prepared,
                mask=base_mask & (~rm_mask),
                nuisance_mode=str(args.nuisance_mode),
            )
            dr = _drop_impact_row(
                session=session,
                group_type="source",
                label=str(src),
                n_removed=n_removed,
                fit_all=fit_all,
                fit_drop=fit_drop,
            )
            if dr is not None:
                source_rows.append(dr)
                component_rows.append(dr)

        for bl in _top_labels(baseline_vec, base_mask, int(args.top_baselines)):
            rm_mask = base_mask & (np.asarray([str(v) for v in baseline_vec], dtype=object) == str(bl))
            n_removed = int(np.sum(rm_mask))
            if n_removed < 3:
                continue

            fit_drop = decomp._fit_with_mask(
                prepared=prepared,
                mask=base_mask & (~rm_mask),
                nuisance_mode=str(args.nuisance_mode),
            )
            dr = _drop_impact_row(
                session=session,
                group_type="baseline",
                label=str(bl),
                n_removed=n_removed,
                fit_all=fit_all,
                fit_drop=fit_drop,
            )
            if dr is not None:
                baseline_rows.append(dr)
                component_rows.append(dr)

        for q_label, q_mask in decomp._time_quartile_masks(time_seconds=time_seconds, base_mask=base_mask):
            rm_mask = np.asarray(q_mask, dtype=bool)
            n_removed = int(np.sum(rm_mask))
            if n_removed < 3:
                continue

            fit_drop = decomp._fit_with_mask(
                prepared=prepared,
                mask=base_mask & (~rm_mask),
                nuisance_mode=str(args.nuisance_mode),
            )
            dr = _drop_impact_row(
                session=session,
                group_type="time_quartile",
                label=str(q_label),
                n_removed=n_removed,
                fit_all=fit_all,
                fit_drop=fit_drop,
            )
            if dr is not None:
                time_rows.append(dr)
                component_rows.append(dr)

        summary_rows.append(
            {
                "session": session,
                "max_abs_bendsun_ns": float(row["max_abs_bendsun_ns"]),
                "n_points": int(fit_all["n_points"]),
                "beta_all": float(fit_all["beta_est"]),
                "beta_all_sigma": float(fit_all["beta_sigma"]),
                "max_abs_z_source": max(
                    [float(v["abs_z_impact"]) for v in source_rows if math.isfinite(float(v["abs_z_impact"]))],
                    default=math.nan,
                ),
                "max_abs_z_baseline": max(
                    [float(v["abs_z_impact"]) for v in baseline_rows if math.isfinite(float(v["abs_z_impact"]))],
                    default=math.nan,
                ),
                "max_abs_z_time_quartile": max(
                    [float(v["abs_z_impact"]) for v in time_rows if math.isfinite(float(v["abs_z_impact"]))],
                    default=math.nan,
                ),
                "top_source_group": (
                    ""
                    if not source_rows
                    else str(
                        sorted(source_rows, key=lambda v: abs(float(v["impact_beta_all_minus_drop"])), reverse=True)[0][
                            "group_label"
                        ]
                    )
                ),
                "top_baseline_group": (
                    ""
                    if not baseline_rows
                    else str(
                        sorted(baseline_rows, key=lambda v: abs(float(v["impact_beta_all_minus_drop"])), reverse=True)[0][
                            "group_label"
                        ]
                    )
                ),
                "top_time_quartile_group": (
                    ""
                    if not time_rows
                    else str(
                        sorted(time_rows, key=lambda v: abs(float(v["impact_beta_all_minus_drop"])), reverse=True)[0][
                            "group_label"
                        ]
                    )
                ),
            }
        )

    summary_rows = sorted(summary_rows, key=lambda r: float(r.get("max_abs_bendsun_ns", math.nan)), reverse=True)
    component_rows = sorted(
        component_rows,
        key=lambda r: (str(r["session"]), str(r["group_type"]), -abs(float(r["impact_beta_all_minus_drop"]))),
    )
    if not summary_rows:
        raise RuntimeError("no high-sensitivity sessions produced decomposition outputs.")

    beta_arr = np.asarray([float(r["beta_all"]) for r in summary_rows], dtype=np.float64)
    sigma_arr = np.asarray([float(r["beta_all_sigma"]) for r in summary_rows], dtype=np.float64)
    consistency = _weighted_consistency(beta=beta_arr, sigma=sigma_arr)
    out_dir = root / "output" / "vlbi"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "vlbi_high_sensitivity_factor_decomposition_summary.csv"
    components_csv = out_dir / "vlbi_high_sensitivity_factor_decomposition_components.csv"
    metrics_json = out_dir / "vlbi_high_sensitivity_factor_decomposition_metrics.json"
    plot_pdf = out_dir / "vlbi_high_sensitivity_factor_decomposition.pdf"
    plot_png = out_dir / "vlbi_high_sensitivity_factor_decomposition.png"

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "session",
            "max_abs_bendsun_ns",
            "n_points",
            "beta_all",
            "beta_all_sigma",
            "max_abs_z_source",
            "max_abs_z_baseline",
            "max_abs_z_time_quartile",
            "top_source_group",
            "top_baseline_group",
            "top_time_quartile_group",
        ]
        w = csv.writer(f)
        w.writerow(cols)
        for r in summary_rows:
            out: List[object] = []
            for c in cols:
                val = r.get(c, "")
                if isinstance(val, float):
                    out.append(f"{val:.16e}" if math.isfinite(val) else "nan")
                else:
                    out.append(val)

            w.writerow(out)

    with components_csv.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "session",
            "group_type",
            "group_label",
            "n_removed",
            "beta_all",
            "beta_all_sigma",
            "beta_drop",
            "beta_drop_sigma",
            "impact_beta_all_minus_drop",
            "sigma_combined",
            "abs_z_impact",
        ]
        w = csv.writer(f)
        w.writerow(cols)
        for r in component_rows:
            out: List[object] = []
            for c in cols:
                val = r.get(c, "")
                if isinstance(val, float):
                    out.append(f"{val:.16e}" if math.isfinite(val) else "nan")
                else:
                    out.append(val)

            w.writerow(out)

    _plot_summary(pdf_path=plot_pdf, png_path=plot_png, summary_rows=summary_rows)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "description": "all-sky drop-one factor decomposition on high-sensitivity sessions",
            "nuisance_mode": str(args.nuisance_mode),
            "observable_series": str(args.observable_series),
            "min_sensitivity_ns": float(args.min_sensitivity_ns),
        },
        "input": {
            "allsky_summary_csv": str(args.allsky_summary.resolve()),
            "session_root": str(session_root),
            "n_sessions_selected": int(len(summary_rows)),
        },
        "high_sensitivity_consistency": consistency,
        "summary_rows": summary_rows,
        "outputs": {
            "summary_csv": str(summary_csv),
            "components_csv": str(components_csv),
            "metrics_json": str(metrics_json),
            "plot_pdf": str(plot_pdf),
            "plot_png": str(plot_png),
        },
    }
    metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _sync_public(root, [summary_csv, components_csv, metrics_json, plot_pdf, plot_png])
    print("Wrote:", summary_csv)
    print("Wrote:", components_csv)
    print("Wrote:", metrics_json)
    print("Wrote:", plot_pdf)
    print("Wrote:", plot_png)
    print("Synced:", root / "output" / "public" / "vlbi")
    return 0


# Branch: Execute CLI entrypoint when this file is invoked directly.

if __name__ == "__main__":
    raise SystemExit(main())
