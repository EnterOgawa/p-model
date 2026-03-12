#!/usr/bin/env python3
"""
vlbi_beta_session_watchpack_audit.py

Session-level watchpack audit for a designated high-sensitivity session.

Purpose:
- Audit source/time decomposition signals for one target session.
- Quantify the effect of excluding the target session on high-sensitivity
  cross-session consistency.
- Fix a machine-readable watchpack policy (exclude or keep).

Inputs:
- output/public/vlbi/vlbi_allsky_beta_consistency_summary.csv
- output/public/vlbi/vlbi_high_sensitivity_factor_decomposition_summary.csv
- output/public/vlbi/vlbi_high_sensitivity_factor_decomposition_components.csv

Outputs:
- output/vlbi/vlbi_beta_session_watchpack_audit_leave_one_out.csv
- output/vlbi/vlbi_beta_session_watchpack_audit_target_components.csv
- output/vlbi/vlbi_beta_session_watchpack_audit_metrics.json
- output/vlbi/vlbi_beta_session_watchpack_audit.pdf
- output/vlbi/vlbi_beta_session_watchpack_audit.png
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


# Function: Resolve repository root from script location.
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Function: Read all-sky summary rows from CSV.

def _read_allsky_summary(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "session": str(row.get("session") or "").strip(),
                    "beta_est": float(row.get("beta_est", "nan")),
                    "beta_sigma": float(row.get("beta_sigma", "nan")),
                    "max_abs_bendsun_ns": float(row.get("max_abs_bendsun_ns", "nan")),
                }
            )

    return rows


# Function: Read factor-decomposition summary rows.

def _read_factor_summary(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "session": str(row.get("session") or "").strip(),
                    "max_abs_bendsun_ns": float(row.get("max_abs_bendsun_ns", "nan")),
                    "n_points": int(float(row.get("n_points", "0"))),
                    "beta_all": float(row.get("beta_all", "nan")),
                    "beta_all_sigma": float(row.get("beta_all_sigma", "nan")),
                    "max_abs_z_source": float(row.get("max_abs_z_source", "nan")),
                    "max_abs_z_baseline": float(row.get("max_abs_z_baseline", "nan")),
                    "max_abs_z_time_quartile": float(row.get("max_abs_z_time_quartile", "nan")),
                    "top_source_group": str(row.get("top_source_group") or "").strip(),
                    "top_baseline_group": str(row.get("top_baseline_group") or "").strip(),
                    "top_time_quartile_group": str(row.get("top_time_quartile_group") or "").strip(),
                }
            )

    return rows


# Function: Read factor-decomposition component rows.

def _read_factor_components(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "session": str(row.get("session") or "").strip(),
                    "group_type": str(row.get("group_type") or "").strip(),
                    "group_label": str(row.get("group_label") or "").strip(),
                    "n_removed": int(float(row.get("n_removed", "0"))),
                    "impact_beta_all_minus_drop": float(row.get("impact_beta_all_minus_drop", "nan")),
                    "abs_z_impact": float(row.get("abs_z_impact", "nan")),
                }
            )

    return rows


# Function: Compute weighted consistency metrics.

def _weighted_consistency(rows: List[Dict[str, object]]) -> Dict[str, float]:
    beta = np.asarray([float(r["beta_est"]) for r in rows], dtype=np.float64)
    sigma = np.asarray([float(r["beta_sigma"]) for r in rows], dtype=np.float64)
    mask = np.isfinite(beta) & np.isfinite(sigma) & (sigma > 0.0)
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

    beta = beta[mask]
    sigma = sigma[mask]
    w = 1.0 / np.square(sigma)
    wsum = float(np.sum(w))
    mean = float(np.sum(w * beta) / wsum)
    sig_mean = float(math.sqrt(1.0 / wsum))
    chi2 = float(np.sum(np.square((beta - mean) / sigma)))
    dof = float(max(1, int(beta.size - 1)))
    return {
        "n_valid": int(beta.size),
        "beta_weighted_mean": mean,
        "beta_weighted_sigma": sig_mean,
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": float(chi2 / dof),
    }


# Function: Build leave-one-out rows for designated high-sensitivity subset.

def _build_leave_one_out(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    base = _weighted_consistency(rows)
    out.append(
        {
            "mode": "base",
            "drop_session": "",
            "n_sessions": int(len(rows)),
            "n_valid": int(base["n_valid"]),
            "beta_weighted_mean": float(base["beta_weighted_mean"]),
            "beta_weighted_sigma": float(base["beta_weighted_sigma"]),
            "chi2": float(base["chi2"]),
            "dof": float(base["dof"]),
            "chi2_dof": float(base["chi2_dof"]),
        }
    )
    for row in rows:
        sess = str(row["session"])
        sub = [r for r in rows if str(r["session"]) != sess]
        cc = _weighted_consistency(sub)
        out.append(
            {
                "mode": "drop_one",
                "drop_session": sess,
                "n_sessions": int(len(sub)),
                "n_valid": int(cc["n_valid"]),
                "beta_weighted_mean": float(cc["beta_weighted_mean"]),
                "beta_weighted_sigma": float(cc["beta_weighted_sigma"]),
                "chi2": float(cc["chi2"]),
                "dof": float(cc["dof"]),
                "chi2_dof": float(cc["chi2_dof"]),
            }
        )

    return out


# Function: Render leave-one-out chi2/dof comparison figure.

def _plot_leave_one_out(pdf_path: Path, png_path: Path, rows: List[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    drop_rows = [r for r in rows if str(r["mode"]) == "drop_one"]
    if not drop_rows:
        return

    base_rows = [r for r in rows if str(r["mode"]) == "base"]
    base_chi = float(base_rows[0]["chi2_dof"]) if base_rows else math.nan
    labels = [str(r["drop_session"]) for r in drop_rows]
    vals = np.asarray([float(r["chi2_dof"]) for r in drop_rows], dtype=np.float64)
    x = np.arange(len(labels), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.bar(x, vals, color="tab:blue", alpha=0.85)
    if math.isfinite(base_chi):
        ax.axhline(base_chi, color="tab:red", linestyle="--", linewidth=1.2, label="base")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("chi2/dof")
    ax.set_title("High-sensitivity leave-one-out consistency")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(pdf_path))
    fig.savefig(str(png_path), dpi=170)
    plt.close(fig)


# Function: Sync outputs to output/public/vlbi.

def _sync_public(root: Path, outputs: Sequence[Path]) -> None:
    dst = root / "output" / "public" / "vlbi"
    dst.mkdir(parents=True, exist_ok=True)
    for p in outputs:
        if p.exists():
            shutil.copy2(p, dst / p.name)


# Function: Main entrypoint for session watchpack audit.

def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Session-level watchpack audit for high-sensitivity VLBI consistency.")
    ap.add_argument(
        "--allsky-summary",
        type=Path,
        default=root / "output" / "public" / "vlbi" / "vlbi_allsky_beta_consistency_summary.csv",
        help="All-sky summary CSV.",
    )
    ap.add_argument(
        "--factor-summary",
        type=Path,
        default=root / "output" / "public" / "vlbi" / "vlbi_high_sensitivity_factor_decomposition_summary.csv",
        help="Factor decomposition summary CSV.",
    )
    ap.add_argument(
        "--factor-components",
        type=Path,
        default=root / "output" / "public" / "vlbi" / "vlbi_high_sensitivity_factor_decomposition_components.csv",
        help="Factor decomposition component CSV.",
    )
    ap.add_argument(
        "--target-session",
        type=str,
        default="22MAY09XA",
        help="Session label to audit.",
    )
    ap.add_argument(
        "--min-sensitivity-ns",
        type=float,
        default=10.0,
        help="High-sensitivity cutoff for leave-one-out subset.",
    )
    ap.add_argument(
        "--min-relative-improvement",
        type=float,
        default=0.05,
        help="Minimum relative chi2/dof improvement required to recommend exclusion.",
    )
    args = ap.parse_args()

    allsky_rows = _read_allsky_summary(args.allsky_summary.resolve())
    if not allsky_rows:
        raise FileNotFoundError(f"all-sky summary not found or empty: {args.allsky_summary}")

    factor_summary = _read_factor_summary(args.factor_summary.resolve())
    if not factor_summary:
        raise FileNotFoundError(f"factor summary not found or empty: {args.factor_summary}")

    factor_components = _read_factor_components(args.factor_components.resolve())
    if not factor_components:
        raise FileNotFoundError(f"factor components not found or empty: {args.factor_components}")

    hs_rows = [
        r
        for r in allsky_rows
        if math.isfinite(float(r["max_abs_bendsun_ns"])) and float(r["max_abs_bendsun_ns"]) >= float(args.min_sensitivity_ns)
    ]
    if len(hs_rows) < 2:
        raise RuntimeError("insufficient high-sensitivity rows for leave-one-out audit.")

    leave_one_out_rows = _build_leave_one_out(hs_rows)
    base_row = [r for r in leave_one_out_rows if str(r["mode"]) == "base"][0]
    target = str(args.target_session).strip()
    drop_target_rows = [
        r for r in leave_one_out_rows if str(r["mode"]) == "drop_one" and str(r["drop_session"]) == target
    ]
    if not drop_target_rows:
        raise RuntimeError(f"target session not found in high-sensitivity subset: {target}")

    drop_target = drop_target_rows[0]
    base_chi = float(base_row["chi2_dof"])
    drop_chi = float(drop_target["chi2_dof"])
    rel_improvement = float((base_chi - drop_chi) / base_chi) if (math.isfinite(base_chi) and base_chi > 0.0) else math.nan
    recommend_exclude = bool(
        math.isfinite(rel_improvement) and (rel_improvement >= float(args.min_relative_improvement))
    )
    ranking = sorted(
        [r for r in leave_one_out_rows if str(r["mode"]) == "drop_one"],
        key=lambda r: float(r["chi2_dof"]) if math.isfinite(float(r["chi2_dof"])) else float("inf"),
    )

    target_summary = [r for r in factor_summary if str(r["session"]) == target]
    if not target_summary:
        raise RuntimeError(f"target session not found in factor summary: {target}")

    target_summary_row = target_summary[0]
    target_components = [r for r in factor_components if str(r["session"]) == target]
    target_components = sorted(
        target_components,
        key=lambda r: (str(r["group_type"]), -abs(float(r["impact_beta_all_minus_drop"]))),
    )

    out_dir = root / "output" / "vlbi"
    out_dir.mkdir(parents=True, exist_ok=True)
    loo_csv = out_dir / "vlbi_beta_session_watchpack_audit_leave_one_out.csv"
    comp_csv = out_dir / "vlbi_beta_session_watchpack_audit_target_components.csv"
    metrics_json = out_dir / "vlbi_beta_session_watchpack_audit_metrics.json"
    plot_pdf = out_dir / "vlbi_beta_session_watchpack_audit.pdf"
    plot_png = out_dir / "vlbi_beta_session_watchpack_audit.png"

    with loo_csv.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "mode",
            "drop_session",
            "n_sessions",
            "n_valid",
            "beta_weighted_mean",
            "beta_weighted_sigma",
            "chi2",
            "dof",
            "chi2_dof",
        ]
        w = csv.writer(f)
        w.writerow(cols)
        for r in leave_one_out_rows:
            out: List[object] = []
            for c in cols:
                val = r.get(c, "")
                if isinstance(val, float):
                    out.append(f"{val:.16e}" if math.isfinite(val) else "nan")
                else:
                    out.append(val)

            w.writerow(out)

    with comp_csv.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "session",
            "group_type",
            "group_label",
            "n_removed",
            "impact_beta_all_minus_drop",
            "abs_z_impact",
        ]
        w = csv.writer(f)
        w.writerow(cols)
        for r in target_components:
            out: List[object] = []
            for c in cols:
                val = r.get(c, "")
                if isinstance(val, float):
                    out.append(f"{val:.16e}" if math.isfinite(val) else "nan")
                else:
                    out.append(val)

            w.writerow(out)

    _plot_leave_one_out(pdf_path=plot_pdf, png_path=plot_png, rows=leave_one_out_rows)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "description": "session-level watchpack audit from high-sensitivity leave-one-out and factor decomposition",
            "target_session": target,
            "min_sensitivity_ns": float(args.min_sensitivity_ns),
            "min_relative_improvement": float(args.min_relative_improvement),
        },
        "input": {
            "allsky_summary_csv": str(args.allsky_summary.resolve()),
            "factor_summary_csv": str(args.factor_summary.resolve()),
            "factor_components_csv": str(args.factor_components.resolve()),
            "n_high_sensitivity_sessions": int(len(hs_rows)),
        },
        "target_summary": target_summary_row,
        "target_components": target_components,
        "leave_one_out": leave_one_out_rows,
        "leave_one_out_ranking": ranking,
        "watchpack_decision": {
            "target_session": target,
            "base_chi2_dof": base_chi,
            "drop_target_chi2_dof": drop_chi,
            "relative_improvement": rel_improvement,
            "recommend_exclude": recommend_exclude,
            "reason": (
                "exclude_improves_consistency"
                if recommend_exclude
                else "exclude_not_improving_consistency"
            ),
            "fixed_policy": (
                f"exclude_session:{target}"
                if recommend_exclude
                else f"keep_session:{target}"
            ),
        },
        "outputs": {
            "leave_one_out_csv": str(loo_csv),
            "target_components_csv": str(comp_csv),
            "metrics_json": str(metrics_json),
            "plot_pdf": str(plot_pdf),
            "plot_png": str(plot_png),
        },
    }
    metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _sync_public(root, [loo_csv, comp_csv, metrics_json, plot_pdf, plot_png])
    print("Wrote:", loo_csv)
    print("Wrote:", comp_csv)
    print("Wrote:", metrics_json)
    print("Wrote:", plot_pdf)
    print("Wrote:", plot_png)
    print("Synced:", root / "output" / "public" / "vlbi")
    return 0


# Branch: Execute CLI entrypoint when invoked directly.

if __name__ == "__main__":
    raise SystemExit(main())
