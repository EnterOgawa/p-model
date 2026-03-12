#!/usr/bin/env python3
"""
vlbi_beta_source_session_matrix.py

Build a source x session matrix of VLBI beta estimates on high-sensitivity sessions.

Goal:
- Quantify whether beta remains stable within the same source across sessions.
- Separate source-structure-driven variability from time/baseline systematic effects.

Definitions:
- beta_source: direct-fit beta estimated on one source subset in one session.
- delta_beta_from_1 = beta_source - 1
- delta_beta_vs_session_all = beta_source - beta_all(session)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import vlbi_beta_source_filter_decomposition as decomp


# Function: Resolve repository root from script path.
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Function: Read all-sky session summary CSV rows.

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
                }
            )

    return rows


# Function: Compute weighted mean and chi2/dof for beta values.

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


# Function: Classify chi2/dof into pass/watch/reject status.

def _status_from_chi2_dof(value: float) -> str:
    if not math.isfinite(value):
        return "watch"

    if value <= 2.0:
        return "pass"

    if value <= 5.0:
        return "watch"

    return "reject"


# Function: Render source x session heatmap for delta_beta_from_1.

def _plot_matrix(
    pdf_path: Path,
    png_path: Path,
    matrix: np.ndarray,
    sources: List[str],
    sessions: List[str],
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    if matrix.size == 0:
        return

    mask = np.isfinite(matrix)
    if not np.any(mask):
        return

    m = np.array(matrix, dtype=np.float64)
    m_plot = np.where(mask, m, np.nan)
    fig, ax = plt.subplots(figsize=(max(10.0, 0.55 * len(sessions) + 3.0), max(6.0, 0.42 * len(sources) + 2.4)))
    im = ax.imshow(m_plot, aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(sessions)))
    ax.set_xticklabels(sessions, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(sources)))
    ax.set_yticklabels(sources)
    ax.set_xlabel("Session")
    ax.set_ylabel("Source")
    ax.set_title("VLBI source x session matrix: delta_beta_from_1")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("delta_beta_from_1")
    fig.tight_layout()
    fig.savefig(str(pdf_path))
    fig.savefig(str(png_path), dpi=170)
    plt.close(fig)


# Function: Copy outputs to output/public/vlbi.

def _sync_public(root: Path, outputs: Sequence[Path]) -> None:
    dst = root / "output" / "public" / "vlbi"
    dst.mkdir(parents=True, exist_ok=True)
    for p in outputs:
        if p.exists():
            shutil.copy2(p, dst / p.name)


# Function: Main entrypoint for source-session matrix analysis.

def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Build source x session delta-beta matrix on high-sensitivity VLBI sessions.")
    ap.add_argument(
        "--allsky-summary",
        type=Path,
        default=root / "output" / "public" / "vlbi" / "vlbi_allsky_beta_consistency_summary.csv",
        help="All-sky summary CSV path.",
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
        help="Session selection threshold on max|Cal-BendSun| [ns].",
    )
    ap.add_argument(
        "--min-source-points",
        type=int,
        default=120,
        help="Minimum points per source in a session to run source-level fit.",
    )
    ap.add_argument(
        "--nuisance-mode",
        type=str,
        default="baseline_intercept_linear",
        choices=["none", "baseline_intercept", "baseline_intercept_linear"],
        help="Nuisance mode used in all source-level refits.",
    )
    ap.add_argument(
        "--observable-series",
        type=str,
        default="full",
        choices=["full", "fringe"],
        help="Observable series passed to decomposition reconstruction.",
    )
    ap.add_argument(
        "--min-template-abs",
        type=float,
        default=1.0e-14,
        help="Absolute threshold on |Cal-BendSun template| [s].",
    )
    args = ap.parse_args()

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
        raise RuntimeError(f"no session passes min-sensitivity-ns={args.min_sensitivity_ns}")

    session_root = args.session_root.resolve()
    source_session_rows: List[Dict[str, object]] = []
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

        fit_all = decomp._fit_with_mask(prepared=prepared, mask=base_mask, nuisance_mode=str(args.nuisance_mode))
        if fit_all is None:
            continue

        source_vec = np.asarray([str(v) for v in np.asarray(prepared["source_vec"], dtype=object)], dtype=object)
        uniq, cnt = np.unique(source_vec[base_mask], return_counts=True)
        for src, npt in zip(uniq.tolist(), cnt.tolist()):
            n_points = int(npt)
            if n_points < int(args.min_source_points):
                continue

            src_mask = base_mask & (source_vec == str(src))
            fit_src = decomp._fit_with_mask(prepared=prepared, mask=src_mask, nuisance_mode=str(args.nuisance_mode))
            if fit_src is None:
                continue

            beta_src = float(fit_src["beta_est"])
            beta_src_sig = float(fit_src["beta_sigma"])
            beta_all = float(fit_all["beta_est"])
            beta_all_sig = float(fit_all["beta_sigma"])
            source_session_rows.append(
                {
                    "session": session,
                    "source": str(src),
                    "n_points": n_points,
                    "max_abs_bendsun_ns": float(sr["max_abs_bendsun_ns"]),
                    "beta_source": beta_src,
                    "beta_source_sigma": beta_src_sig,
                    "delta_beta_from_1": float(beta_src - 1.0),
                    "beta_all_session": beta_all,
                    "beta_all_session_sigma": beta_all_sig,
                    "delta_beta_vs_session_all": float(beta_src - beta_all),
                }
            )

    if not source_session_rows:
        raise RuntimeError("no source-session rows available after filters.")

    sessions = sorted(list({str(r["session"]) for r in source_session_rows}))
    sources = sorted(list({str(r["source"]) for r in source_session_rows}))
    session_to_idx = {s: i for i, s in enumerate(sessions)}
    source_to_idx = {s: i for i, s in enumerate(sources)}
    matrix = np.full((len(sources), len(sessions)), np.nan, dtype=np.float64)
    for r in source_session_rows:
        i = source_to_idx[str(r["source"])]
        j = session_to_idx[str(r["session"])]
        matrix[i, j] = float(r["delta_beta_from_1"])

    source_summary_rows: List[Dict[str, object]] = []
    for src in sources:
        rows_src = [r for r in source_session_rows if str(r["source"]) == src]
        beta = np.asarray([float(r["beta_source"]) for r in rows_src], dtype=np.float64)
        sigma = np.asarray([float(r["beta_source_sigma"]) for r in rows_src], dtype=np.float64)
        stat = _weighted_consistency(beta=beta, sigma=sigma)
        source_summary_rows.append(
            {
                "source": src,
                "n_sessions": int(len(rows_src)),
                "total_points": int(np.sum([int(r["n_points"]) for r in rows_src])),
                "beta_weighted_mean": float(stat["beta_weighted_mean"]),
                "beta_weighted_sigma": float(stat["beta_weighted_sigma"]),
                "delta_beta_weighted_mean": float(stat["beta_weighted_mean"] - 1.0)
                if math.isfinite(float(stat["beta_weighted_mean"]))
                else math.nan,
                "chi2_dof": float(stat["chi2_dof"]),
                "status": _status_from_chi2_dof(float(stat["chi2_dof"])),
            }
        )

    source_summary_rows = sorted(
        source_summary_rows,
        key=lambda r: (str(r["status"]), -int(r["n_sessions"]), -int(r["total_points"]), str(r["source"])),
    )
    session_consistency = _weighted_consistency(
        beta=np.asarray([float(r["beta_all_session"]) for r in source_session_rows], dtype=np.float64),
        sigma=np.asarray([float(r["beta_all_session_sigma"]) for r in source_session_rows], dtype=np.float64),
    )
    out_dir = root / "output" / "vlbi"
    out_dir.mkdir(parents=True, exist_ok=True)
    details_csv = out_dir / "vlbi_beta_source_session_matrix_details.csv"
    source_summary_csv = out_dir / "vlbi_beta_source_session_matrix_source_summary.csv"
    metrics_json = out_dir / "vlbi_beta_source_session_matrix_metrics.json"
    plot_pdf = out_dir / "vlbi_beta_source_session_matrix.pdf"
    plot_png = out_dir / "vlbi_beta_source_session_matrix.png"

    with details_csv.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "session",
            "source",
            "n_points",
            "max_abs_bendsun_ns",
            "beta_source",
            "beta_source_sigma",
            "delta_beta_from_1",
            "beta_all_session",
            "beta_all_session_sigma",
            "delta_beta_vs_session_all",
        ]
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(source_session_rows, key=lambda v: (str(v["source"]), str(v["session"]))):
            out: List[object] = []
            for c in cols:
                val = r.get(c, "")
                if isinstance(val, float):
                    out.append(f"{val:.16e}" if math.isfinite(val) else "nan")
                else:
                    out.append(val)

            w.writerow(out)

    with source_summary_csv.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "source",
            "n_sessions",
            "total_points",
            "beta_weighted_mean",
            "beta_weighted_sigma",
            "delta_beta_weighted_mean",
            "chi2_dof",
            "status",
        ]
        w = csv.writer(f)
        w.writerow(cols)
        for r in source_summary_rows:
            out: List[object] = []
            for c in cols:
                val = r.get(c, "")
                if isinstance(val, float):
                    out.append(f"{val:.16e}" if math.isfinite(val) else "nan")
                else:
                    out.append(val)

            w.writerow(out)

    _plot_matrix(pdf_path=plot_pdf, png_path=plot_png, matrix=matrix, sources=sources, sessions=sessions)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "description": "source x session matrix from high-sensitivity VLBI sessions",
            "nuisance_mode": str(args.nuisance_mode),
            "observable_series": str(args.observable_series),
            "min_sensitivity_ns": float(args.min_sensitivity_ns),
            "min_source_points": int(args.min_source_points),
            "delta_beta_definition": "beta_source - 1",
        },
        "input": {
            "allsky_summary_csv": str(args.allsky_summary.resolve()),
            "session_root": str(session_root),
            "n_sessions_selected": int(len(selected_sessions)),
            "n_source_session_rows": int(len(source_session_rows)),
        },
        "sessions": sessions,
        "sources": sources,
        "source_summary": source_summary_rows,
        "session_consistency_proxy": session_consistency,
        "outputs": {
            "details_csv": str(details_csv),
            "source_summary_csv": str(source_summary_csv),
            "metrics_json": str(metrics_json),
            "plot_pdf": str(plot_pdf),
            "plot_png": str(plot_png),
        },
    }
    metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _sync_public(root, [details_csv, source_summary_csv, metrics_json, plot_pdf, plot_png])
    print("Wrote:", details_csv)
    print("Wrote:", source_summary_csv)
    print("Wrote:", metrics_json)
    print("Wrote:", plot_pdf)
    print("Wrote:", plot_png)
    print("Synced:", root / "output" / "public" / "vlbi")
    return 0


# Branch: Execute CLI entrypoint when this file is invoked directly.

if __name__ == "__main__":
    raise SystemExit(main())
