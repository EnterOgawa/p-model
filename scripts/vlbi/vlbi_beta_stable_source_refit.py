#!/usr/bin/env python3
"""
vlbi_beta_stable_source_refit.py

Stable-source constrained beta refit on high-sensitivity VLBI sessions.

Purpose:
- Build a robust subset by selecting sources that are stable across sessions
  (based on source-session matrix diagnostics).
- Refit beta per high-sensitivity session using only that stable-source subset.
- Evaluate cross-session consistency via weighted mean and chi2/dof.

Inputs:
- output/public/vlbi/vlbi_beta_source_session_matrix_source_summary.csv
- output/public/vlbi/vlbi_allsky_beta_consistency_summary.csv
- data/vlbi/sources/vgosdb/<SESSION>/extracted

Outputs:
- output/vlbi/vlbi_beta_stable_source_refit_summary.csv
- output/vlbi/vlbi_beta_stable_source_refit_source_presence.csv
- output/vlbi/vlbi_beta_stable_source_refit_metrics.json
- output/vlbi/vlbi_beta_stable_source_refit.pdf
- output/vlbi/vlbi_beta_stable_source_refit.png
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

import vlbi_beta_source_filter_decomposition as decomp


# Function: Resolve repository root from script location.
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Function: Read all-sky consistency summary rows.

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
                    "n_points": int(float(row.get("n_points", "0"))),
                    "status": str(row.get("status") or "").strip(),
                }
            )

    return rows


# Function: Read source-session matrix source summary rows.

def _read_source_summary(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "source": str(row.get("source") or "").strip(),
                    "n_sessions": int(float(row.get("n_sessions", "0"))),
                    "total_points": int(float(row.get("total_points", "0"))),
                    "beta_weighted_mean": float(row.get("beta_weighted_mean", "nan")),
                    "beta_weighted_sigma": float(row.get("beta_weighted_sigma", "nan")),
                    "chi2_dof": float(row.get("chi2_dof", "nan")),
                    "status": str(row.get("status") or "").strip(),
                }
            )

    return rows


# Function: Compute weighted consistency metrics.

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
    return {
        "n_valid": int(b.size),
        "beta_weighted_mean": bbar,
        "beta_weighted_sigma": sig_bar,
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": float(chi2 / dof),
    }


# Function: Convert chi2/dof value to pass/watch/reject status.

def _status_from_chi2_dof(value: float) -> str:
    if not math.isfinite(value):
        return "watch"

    if value <= 2.0:
        return "pass"

    if value <= 5.0:
        return "watch"

    return "reject"


# Function: Plot stable-source refit overview.

def _plot_summary(
    pdf_path: Path,
    png_path: Path,
    rows: List[Dict[str, object]],
    consistency: Dict[str, float],
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    if not rows:
        return

    labels = [str(r["session"]) for r in rows]
    x = np.arange(len(labels), dtype=np.float64)
    beta = np.asarray([float(r["beta_est_stable"]) for r in rows], dtype=np.float64)
    sigma = np.asarray([float(r["beta_sigma_stable"]) for r in rows], dtype=np.float64)
    npts = np.asarray([float(r["n_points_stable"]) for r in rows], dtype=np.float64)
    nsrc = np.asarray([float(r["n_stable_sources_present"]) for r in rows], dtype=np.float64)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12.8, 8.8), gridspec_kw={"height_ratios": [2.2, 1.2]})
    ax0.errorbar(x, beta, yerr=sigma, fmt="o", color="tab:blue", ecolor="tab:blue", capsize=3)
    if math.isfinite(float(consistency.get("beta_weighted_mean", math.nan))):
        ax0.axhline(
            float(consistency["beta_weighted_mean"]),
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label="weighted mean",
        )

    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=35, ha="right")
    ax0.set_ylabel("beta (stable-source refit)")
    ax0.set_title("Stable-source constrained beta refit on high-sensitivity sessions")
    ax0.grid(True, axis="y", alpha=0.25)
    ax0.legend(loc="best")

    width = 0.36
    ax1.bar(x - (width / 2.0), npts, width=width, color="tab:green", alpha=0.85, label="n_points_stable")
    ax1.bar(x + (width / 2.0), nsrc, width=width, color="tab:orange", alpha=0.85, label="n_stable_sources_present")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.set_ylabel("count")
    ax1.set_xlabel("Session")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="best")

    fig.tight_layout()
    fig.savefig(str(pdf_path))
    fig.savefig(str(png_path), dpi=170)
    plt.close(fig)


# Function: Sync generated artifacts to public VLBI output.

def _sync_public(root: Path, outputs: Sequence[Path]) -> None:
    dst = root / "output" / "public" / "vlbi"
    dst.mkdir(parents=True, exist_ok=True)
    for p in outputs:
        if p.exists():
            shutil.copy2(p, dst / p.name)


# Function: Main entrypoint for stable-source constrained refit.

def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Stable-source constrained VLBI beta refit on high-sensitivity sessions.")
    ap.add_argument(
        "--source-summary",
        type=Path,
        default=root / "output" / "public" / "vlbi" / "vlbi_beta_source_session_matrix_source_summary.csv",
        help="Source summary CSV from source-session matrix step.",
    )
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
        help="Minimum max|Cal-BendSun| [ns] used to select sessions.",
    )
    ap.add_argument(
        "--min-source-sessions",
        type=int,
        default=3,
        help="Stable-source gate: minimum number of sessions in source summary.",
    )
    ap.add_argument(
        "--max-source-chi2-dof",
        type=float,
        default=2.0,
        help="Stable-source gate: maximum source chi2/dof in source summary.",
    )
    ap.add_argument(
        "--min-source-total-points",
        type=int,
        default=0,
        help="Stable-source gate: minimum total points in source summary.",
    )
    ap.add_argument(
        "--require-source-status",
        type=str,
        default="pass",
        choices=["", "pass", "watch", "reject"],
        help="Optional source-status gate from source summary. Empty string disables status gate.",
    )
    ap.add_argument(
        "--min-source-points-per-session",
        type=int,
        default=20,
        help="Per-session source gate: minimum points to include a stable source in that session fit.",
    )
    ap.add_argument(
        "--nuisance-mode",
        type=str,
        default="baseline_intercept_linear",
        choices=["none", "baseline_intercept", "baseline_intercept_linear"],
        help="Nuisance mode for session-level refits.",
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
    args = ap.parse_args()

    source_rows = _read_source_summary(args.source_summary.resolve())
    if not source_rows:
        raise FileNotFoundError(f"source summary not found or empty: {args.source_summary}")

    stable_source_rows = [
        r
        for r in source_rows
        if int(r["n_sessions"]) >= int(args.min_source_sessions)
        and math.isfinite(float(r["chi2_dof"]))
        and float(r["chi2_dof"]) <= float(args.max_source_chi2_dof)
        and int(r["total_points"]) >= int(args.min_source_total_points)
        and (
            (str(args.require_source_status) == "")
            or (str(r.get("status") or "").strip() == str(args.require_source_status))
        )
    ]
    stable_sources = sorted(list({str(r["source"]) for r in stable_source_rows}))
    if not stable_sources:
        raise RuntimeError("stable-source gate selected zero sources.")

    allsky_rows = _read_allsky_summary(args.allsky_summary.resolve())
    if not allsky_rows:
        raise FileNotFoundError(f"all-sky summary not found or empty: {args.allsky_summary}")

    selected_sessions = [
        r
        for r in allsky_rows
        if math.isfinite(float(r["max_abs_bendsun_ns"]))
        and float(r["max_abs_bendsun_ns"]) >= float(args.min_sensitivity_ns)
    ]
    selected_sessions = sorted(selected_sessions, key=lambda r: float(r["max_abs_bendsun_ns"]), reverse=True)
    if not selected_sessions:
        raise RuntimeError(f"no session satisfies min-sensitivity-ns={args.min_sensitivity_ns}")

    session_root = args.session_root.resolve()
    summary_rows: List[Dict[str, object]] = []
    source_presence_rows: List[Dict[str, object]] = []
    stable_source_set = set(stable_sources)
    for sr in selected_sessions:
        session = str(sr["session"])
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

        source_vec = np.asarray([str(v) for v in np.asarray(prepared["source_vec"], dtype=object)], dtype=object)
        stable_mask_seed = base_mask & np.asarray([(s in stable_source_set) for s in source_vec], dtype=bool)
        uniq_seed, cnt_seed = np.unique(source_vec[stable_mask_seed], return_counts=True)
        allowed_source_set = {
            str(s)
            for s, n in zip(uniq_seed.tolist(), cnt_seed.tolist())
            if int(n) >= int(args.min_source_points_per_session)
        }
        stable_mask = base_mask & np.asarray([(s in allowed_source_set) for s in source_vec], dtype=bool)
        n_stable_points = int(np.sum(stable_mask))
        if n_stable_points < 3:
            continue

        fit_stable = decomp._fit_with_mask(prepared=prepared, mask=stable_mask, nuisance_mode=str(args.nuisance_mode))
        if fit_stable is None:
            continue

        fit_all = decomp._fit_with_mask(prepared=prepared, mask=base_mask, nuisance_mode=str(args.nuisance_mode))
        if fit_all is None:
            continue

        uniq, cnt = np.unique(source_vec[stable_mask], return_counts=True)
        source_counts = sorted(
            [{"source": str(s), "n_points": int(n)} for s, n in zip(uniq.tolist(), cnt.tolist())],
            key=lambda v: int(v["n_points"]),
            reverse=True,
        )
        for row in source_counts:
            source_presence_rows.append(
                {"session": session, "source": str(row["source"]), "n_points": int(row["n_points"])}
            )

        beta_stable = float(fit_stable["beta_est"])
        sig_stable = float(fit_stable["beta_sigma"])
        beta_all = float(fit_all["beta_est"])
        sig_all = float(fit_all["beta_sigma"])
        delta = float(beta_stable - beta_all)
        sigma_comb = float(math.sqrt(max(0.0, (sig_stable * sig_stable) + (sig_all * sig_all))))
        z_abs = float(abs(delta) / sigma_comb) if sigma_comb > 0.0 else math.nan
        summary_rows.append(
            {
                "session": session,
                "max_abs_bendsun_ns": float(sr["max_abs_bendsun_ns"]),
                "n_points_all": int(np.sum(base_mask)),
                "n_points_stable": n_stable_points,
                "n_stable_sources_present": int(len(source_counts)),
                "beta_est_stable": beta_stable,
                "beta_sigma_stable": sig_stable,
                "beta_est_all": beta_all,
                "beta_sigma_all": sig_all,
                "delta_beta_stable_minus_all": delta,
                "abs_z_stable_vs_all": z_abs,
                "top_source_by_points": (str(source_counts[0]["source"]) if source_counts else ""),
            }
        )

    if not summary_rows:
        raise RuntimeError("stable-source refit produced zero valid session rows.")

    summary_rows = sorted(summary_rows, key=lambda r: float(r["max_abs_bendsun_ns"]), reverse=True)
    source_presence_rows = sorted(
        source_presence_rows,
        key=lambda r: (str(r["session"]), -int(r["n_points"]), str(r["source"])),
    )
    consistency = _weighted_consistency(
        beta=np.asarray([float(r["beta_est_stable"]) for r in summary_rows], dtype=np.float64),
        sigma=np.asarray([float(r["beta_sigma_stable"]) for r in summary_rows], dtype=np.float64),
    )
    consistency_status = _status_from_chi2_dof(float(consistency["chi2_dof"]))
    out_dir = root / "output" / "vlbi"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "vlbi_beta_stable_source_refit_summary.csv"
    source_presence_csv = out_dir / "vlbi_beta_stable_source_refit_source_presence.csv"
    metrics_json = out_dir / "vlbi_beta_stable_source_refit_metrics.json"
    plot_pdf = out_dir / "vlbi_beta_stable_source_refit.pdf"
    plot_png = out_dir / "vlbi_beta_stable_source_refit.png"

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "session",
            "max_abs_bendsun_ns",
            "n_points_all",
            "n_points_stable",
            "n_stable_sources_present",
            "beta_est_stable",
            "beta_sigma_stable",
            "beta_est_all",
            "beta_sigma_all",
            "delta_beta_stable_minus_all",
            "abs_z_stable_vs_all",
            "top_source_by_points",
        ]
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in summary_rows:
            out: List[object] = []
            for col in cols:
                val = row.get(col, "")
                if isinstance(val, float):
                    out.append(f"{val:.16e}" if math.isfinite(val) else "nan")
                else:
                    out.append(val)

            writer.writerow(out)

    with source_presence_csv.open("w", encoding="utf-8", newline="") as f:
        cols = ["session", "source", "n_points"]
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in source_presence_rows:
            writer.writerow([row["session"], row["source"], row["n_points"]])

    _plot_summary(pdf_path=plot_pdf, png_path=plot_png, rows=summary_rows, consistency=consistency)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "description": "stable-source constrained refit on high-sensitivity sessions",
            "nuisance_mode": str(args.nuisance_mode),
            "observable_series": str(args.observable_series),
            "min_sensitivity_ns": float(args.min_sensitivity_ns),
            "stable_source_gate": {
                "min_source_sessions": int(args.min_source_sessions),
                "max_source_chi2_dof": float(args.max_source_chi2_dof),
                "min_source_total_points": int(args.min_source_total_points),
                "require_source_status": str(args.require_source_status),
                "min_source_points_per_session": int(args.min_source_points_per_session),
            },
        },
        "input": {
            "source_summary_csv": str(args.source_summary.resolve()),
            "allsky_summary_csv": str(args.allsky_summary.resolve()),
            "session_root": str(session_root),
            "n_sessions_selected": int(len(selected_sessions)),
            "n_sessions_valid": int(len(summary_rows)),
            "n_stable_sources": int(len(stable_sources)),
        },
        "stable_sources": stable_sources,
        "stable_source_rows": stable_source_rows,
        "session_summary": summary_rows,
        "consistency": {
            **consistency,
            "status": consistency_status,
        },
        "outputs": {
            "summary_csv": str(summary_csv),
            "source_presence_csv": str(source_presence_csv),
            "metrics_json": str(metrics_json),
            "plot_pdf": str(plot_pdf),
            "plot_png": str(plot_png),
        },
    }
    metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _sync_public(root, [summary_csv, source_presence_csv, metrics_json, plot_pdf, plot_png])
    print("Wrote:", summary_csv)
    print("Wrote:", source_presence_csv)
    print("Wrote:", metrics_json)
    print("Wrote:", plot_pdf)
    print("Wrote:", plot_png)
    print("Synced:", root / "output" / "public" / "vlbi")
    return 0


# Branch: Execute CLI entrypoint when invoked directly.

if __name__ == "__main__":
    raise SystemExit(main())
