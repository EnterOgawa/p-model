#!/usr/bin/env python3
"""
vlbi_beta_high_sensitivity_threshold_sweep.py

Sensitivity sweep for high-sensitivity session threshold (min_sensitivity_ns).

Purpose:
- Re-evaluate cross-session beta consistency as a function of the
  high-sensitivity gate on max|Cal-BendSun| [ns].
- Provide a machine-readable recommendation for operational threshold.

Input:
- output/public/vlbi/vlbi_allsky_beta_consistency_summary.csv

Outputs:
- output/vlbi/vlbi_high_sensitivity_threshold_sweep.csv
- output/vlbi/vlbi_high_sensitivity_threshold_sweep_metrics.json
- output/vlbi/vlbi_high_sensitivity_threshold_sweep.pdf
- output/vlbi/vlbi_high_sensitivity_threshold_sweep.png
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


# Function: Read all-sky summary rows.

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
            "status": "watch",
        }

    beta = beta[mask]
    sigma = sigma[mask]
    w = 1.0 / np.square(sigma)
    wsum = float(np.sum(w))
    mean = float(np.sum(w * beta) / wsum)
    sig_mean = float(math.sqrt(1.0 / wsum))
    chi2 = float(np.sum(np.square((beta - mean) / sigma)))
    dof = float(max(1, int(beta.size - 1)))
    chi2_dof = float(chi2 / dof)
    status = "reject"
    if chi2_dof <= 2.0:
        status = "pass"
    elif chi2_dof <= 5.0:
        status = "watch"

    return {
        "n_valid": int(beta.size),
        "beta_weighted_mean": mean,
        "beta_weighted_sigma": sig_mean,
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": chi2_dof,
        "status": status,
    }


# Function: Select recommendation from threshold sweep rows.

def _select_recommendation(rows: List[Dict[str, object]], min_sessions_operational: int) -> Dict[str, object]:
    candidates = [r for r in rows if int(r["n_sessions"]) >= int(min_sessions_operational)]
    pool = candidates if candidates else rows
    if not pool:
        return {"recommended_threshold_ns": math.nan, "reason": "no_rows"}

    ranked = sorted(
        pool,
        key=lambda r: (
            float(r["chi2_dof"]) if math.isfinite(float(r["chi2_dof"])) else float("inf"),
            -int(r["n_sessions"]),
            float(r["threshold_ns"]),
        ),
    )
    best = ranked[0]
    return {
        "recommended_threshold_ns": float(best["threshold_ns"]),
        "recommended_n_sessions": int(best["n_sessions"]),
        "recommended_chi2_dof": float(best["chi2_dof"]),
        "recommended_status": str(best["status"]),
        "candidate_count": int(len(pool)),
        "min_sessions_operational": int(min_sessions_operational),
        "reason": ("operational_pool" if candidates else "fallback_all_rows"),
    }


# Function: Plot threshold sweep diagnostics.

def _plot_sweep(pdf_path: Path, png_path: Path, rows: List[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    if not rows:
        return

    x = np.asarray([float(r["threshold_ns"]) for r in rows], dtype=np.float64)
    y = np.asarray([float(r["chi2_dof"]) for r in rows], dtype=np.float64)
    n = np.asarray([int(r["n_sessions"]) for r in rows], dtype=np.int64)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11.8, 7.8), gridspec_kw={"height_ratios": [2.0, 1.2]})
    ax0.plot(x, y, marker="o", color="tab:blue", linewidth=1.7)
    ax0.axhline(2.0, color="tab:gray", linestyle="--", linewidth=1.0, label="pass gate")
    ax0.axhline(5.0, color="tab:gray", linestyle=":", linewidth=1.0, label="watch/reject gate")
    ax0.set_ylabel("chi2/dof")
    ax0.set_title("High-sensitivity threshold sweep")
    ax0.grid(True, axis="both", alpha=0.25)
    ax0.legend(loc="best")

    ax1.bar(x, n, width=0.9, color="tab:orange", alpha=0.85)
    ax1.set_xlabel("min_sensitivity_ns")
    ax1.set_ylabel("n_sessions")
    ax1.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(pdf_path))
    fig.savefig(str(png_path), dpi=170)
    plt.close(fig)


# Function: Sync generated outputs to public VLBI directory.

def _sync_public(root: Path, outputs: Sequence[Path]) -> None:
    dst = root / "output" / "public" / "vlbi"
    dst.mkdir(parents=True, exist_ok=True)
    for p in outputs:
        if p.exists():
            shutil.copy2(p, dst / p.name)


# Function: Main entrypoint for threshold sweep.

def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Sweep high-sensitivity threshold for VLBI all-sky consistency.")
    ap.add_argument(
        "--allsky-summary",
        type=Path,
        default=root / "output" / "public" / "vlbi" / "vlbi_allsky_beta_consistency_summary.csv",
        help="All-sky summary CSV generated by vlbi_beta_allsky_consistency.py.",
    )
    ap.add_argument(
        "--thresholds",
        type=str,
        default="10,12,15,20",
        help="Comma-separated min_sensitivity_ns values.",
    )
    ap.add_argument(
        "--min-sessions-operational",
        type=int,
        default=3,
        help="Minimum sessions for operational recommendation pool.",
    )
    args = ap.parse_args()

    rows = _read_allsky_summary(args.allsky_summary.resolve())
    if not rows:
        raise FileNotFoundError(f"all-sky summary not found or empty: {args.allsky_summary}")

    thresholds = []
    for token in str(args.thresholds).split(","):
        t = token.strip()
        if not t:
            continue

        thresholds.append(float(t))

    if not thresholds:
        raise RuntimeError("threshold list is empty.")

    thresholds = sorted(list({float(v) for v in thresholds}))
    sweep_rows: List[Dict[str, object]] = []
    for th in thresholds:
        sub = [r for r in rows if math.isfinite(float(r["max_abs_bendsun_ns"])) and float(r["max_abs_bendsun_ns"]) >= th]
        cc = _weighted_consistency(sub)
        sweep_rows.append(
            {
                "threshold_ns": float(th),
                "n_sessions": int(len(sub)),
                "n_valid": int(cc["n_valid"]),
                "beta_weighted_mean": float(cc["beta_weighted_mean"]),
                "beta_weighted_sigma": float(cc["beta_weighted_sigma"]),
                "chi2": float(cc["chi2"]),
                "dof": float(cc["dof"]),
                "chi2_dof": float(cc["chi2_dof"]),
                "status": str(cc["status"]),
                "sessions": [str(r["session"]) for r in sorted(sub, key=lambda x: float(x["max_abs_bendsun_ns"]), reverse=True)],
            }
        )

    recommendation = _select_recommendation(
        rows=sweep_rows,
        min_sessions_operational=int(args.min_sessions_operational),
    )
    out_dir = root / "output" / "vlbi"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "vlbi_high_sensitivity_threshold_sweep.csv"
    metrics_path = out_dir / "vlbi_high_sensitivity_threshold_sweep_metrics.json"
    plot_pdf = out_dir / "vlbi_high_sensitivity_threshold_sweep.pdf"
    plot_png = out_dir / "vlbi_high_sensitivity_threshold_sweep.png"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "threshold_ns",
            "n_sessions",
            "n_valid",
            "beta_weighted_mean",
            "beta_weighted_sigma",
            "chi2",
            "dof",
            "chi2_dof",
            "status",
            "sessions",
        ]
        w = csv.writer(f)
        w.writerow(cols)
        for r in sweep_rows:
            out: List[object] = []
            for c in cols:
                if c == "sessions":
                    out.append(";".join([str(v) for v in r["sessions"]]))
                    continue

                val = r.get(c, "")
                if isinstance(val, float):
                    out.append(f"{val:.16e}" if math.isfinite(val) else "nan")
                else:
                    out.append(val)

            w.writerow(out)

    _plot_sweep(pdf_path=plot_pdf, png_path=plot_png, rows=sweep_rows)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "description": "high-sensitivity threshold sweep on all-sky summary",
            "thresholds": thresholds,
            "min_sessions_operational": int(args.min_sessions_operational),
        },
        "input": {
            "allsky_summary_csv": str(args.allsky_summary.resolve()),
            "n_allsky_rows": int(len(rows)),
        },
        "sweep_rows": sweep_rows,
        "recommendation": recommendation,
        "outputs": {
            "csv": str(csv_path),
            "metrics_json": str(metrics_path),
            "plot_pdf": str(plot_pdf),
            "plot_png": str(plot_png),
        },
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _sync_public(root, [csv_path, metrics_path, plot_pdf, plot_png])
    print("Wrote:", csv_path)
    print("Wrote:", metrics_path)
    print("Wrote:", plot_pdf)
    print("Wrote:", plot_png)
    print("Synced:", root / "output" / "public" / "vlbi")
    return 0


# Branch: Execute CLI entrypoint when invoked directly.

if __name__ == "__main__":
    raise SystemExit(main())
