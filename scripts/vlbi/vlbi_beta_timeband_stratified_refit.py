#!/usr/bin/env python3
"""
vlbi_beta_timeband_stratified_refit.py

Time-band (quartile) stratified beta refit on high-sensitivity VLBI sessions.

Purpose:
- Refit beta on high-sensitivity sessions using stable-source subsets.
- Split each session into time quartiles and run per-quartile refits.
- Test whether cross-session consistency improves when comparing
  like-for-like time bands (Q1..Q4).

Inputs:
- output/public/vlbi/vlbi_beta_source_session_matrix_source_summary.csv
- output/public/vlbi/vlbi_allsky_beta_consistency_summary.csv
- data/vlbi/sources/vgosdb/<SESSION>/extracted

Outputs:
- output/vlbi/vlbi_beta_timeband_stratified_refit_details.csv
- output/vlbi/vlbi_beta_timeband_stratified_refit_session_summary.csv
- output/vlbi/vlbi_beta_timeband_stratified_refit_quartile_consistency.csv
- output/vlbi/vlbi_beta_timeband_stratified_refit_metrics.json
- output/vlbi/vlbi_beta_timeband_stratified_refit.pdf
- output/vlbi/vlbi_beta_timeband_stratified_refit.png
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


# Function: Compute max pairwise z-score among quartiles in one session.

def _max_pairwise_z(rows: List[Dict[str, object]]) -> float:
    if len(rows) < 2:
        return math.nan

    max_abs_z = -1.0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            b0 = float(rows[i]["beta_est_quartile"])
            s0 = float(rows[i]["beta_sigma_quartile"])
            b1 = float(rows[j]["beta_est_quartile"])
            s1 = float(rows[j]["beta_sigma_quartile"])
            sigma_comb = float(math.sqrt(max(0.0, (s0 * s0) + (s1 * s1))))
            if sigma_comb <= 0.0:
                continue

            z_abs = float(abs(b0 - b1) / sigma_comb)
            if z_abs > max_abs_z:
                max_abs_z = z_abs

    if max_abs_z < 0.0:
        return math.nan

    return float(max_abs_z)


# Function: Build quartile-level weighted consistency rows from detail rows.

def _build_quartile_consistency_rows(detail_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    quartiles = sorted(list({str(r["quartile"]) for r in detail_rows}))
    rows: List[Dict[str, object]] = []
    for q in quartiles:
        rq = [r for r in detail_rows if str(r["quartile"]) == q]
        cc = _weighted_consistency(
            beta=np.asarray([float(r["beta_est_quartile"]) for r in rq], dtype=np.float64),
            sigma=np.asarray([float(r["beta_sigma_quartile"]) for r in rq], dtype=np.float64),
        )
        rows.append(
            {
                "quartile": q,
                "n_sessions": int(len(rq)),
                "n_points_total": int(np.sum(np.asarray([int(r["n_points_quartile"]) for r in rq], dtype=np.int64))),
                "beta_weighted_mean": float(cc["beta_weighted_mean"]),
                "beta_weighted_sigma": float(cc["beta_weighted_sigma"]),
                "chi2": float(cc["chi2"]),
                "dof": float(cc["dof"]),
                "chi2_dof": float(cc["chi2_dof"]),
                "status": _status_from_chi2_dof(float(cc["chi2_dof"])),
            }
        )

    return rows


# Function: Draw summary figure for quartile-stratified refit.

def _plot_summary(
    pdf_path: Path,
    png_path: Path,
    detail_rows: List[Dict[str, object]],
    quartile_rows_raw: List[Dict[str, object]],
    quartile_rows_gated: List[Dict[str, object]],
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    if not detail_rows:
        return

    session_labels = sorted(list({str(r["session"]) for r in detail_rows}))
    q_labels = ["Q1", "Q2", "Q3", "Q4"]
    x = np.arange(len(session_labels), dtype=np.float64)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13.2, 9.0), gridspec_kw={"height_ratios": [2.0, 1.2]})
    colors = {"Q1": "tab:blue", "Q2": "tab:orange", "Q3": "tab:green", "Q4": "tab:red"}
    offsets = {"Q1": -0.24, "Q2": -0.08, "Q3": 0.08, "Q4": 0.24}

    for q in q_labels:
        y = []
        e = []
        for sess in session_labels:
            rr = [r for r in detail_rows if str(r["session"]) == sess and str(r["quartile"]) == q]
            if rr:
                y.append(float(rr[0]["beta_est_quartile"]))
                e.append(float(rr[0]["beta_sigma_quartile"]))
            else:
                y.append(math.nan)
                e.append(math.nan)

        yv = np.asarray(y, dtype=np.float64)
        ev = np.asarray(e, dtype=np.float64)
        m = np.isfinite(yv) & np.isfinite(ev) & (ev > 0.0)
        if int(np.sum(m)) > 0:
            ax0.errorbar(
                x[m] + float(offsets[q]),
                yv[m],
                yerr=ev[m],
                fmt="o",
                color=str(colors[q]),
                ecolor=str(colors[q]),
                capsize=3,
                label=q,
            )

    ax0.set_xticks(x)
    ax0.set_xticklabels(session_labels, rotation=35, ha="right")
    ax0.set_ylabel("beta (quartile-stratified)")
    ax0.set_title("High-sensitivity stable-source refit by time quartile")
    ax0.grid(True, axis="y", alpha=0.25)
    ax0.legend(loc="best")

    qx = np.arange(len(q_labels), dtype=np.float64)
    qchi_raw = []
    qchi_gated = []
    for q in q_labels:
        rr = [r for r in quartile_rows_raw if str(r["quartile"]) == q]
        qchi_raw.append(float(rr[0]["chi2_dof"]) if rr else math.nan)
        rg = [r for r in quartile_rows_gated if str(r["quartile"]) == q]
        qchi_gated.append(float(rg[0]["chi2_dof"]) if rg else math.nan)

    qv_raw = np.asarray(qchi_raw, dtype=np.float64)
    qv_gated = np.asarray(qchi_gated, dtype=np.float64)
    width = 0.36
    ax1.bar(qx - (width / 2.0), qv_raw, width=width, color="tab:gray", alpha=0.55, label="raw")
    ax1.bar(qx + (width / 2.0), qv_gated, width=width, color="tab:purple", alpha=0.85, label="quality-gated")
    ax1.axhline(2.0, color="tab:gray", linestyle="--", linewidth=1.0, label="pass gate")
    ax1.axhline(5.0, color="tab:gray", linestyle=":", linewidth=1.0, label="watch/reject gate")
    ax1.set_xticks(qx)
    ax1.set_xticklabels(q_labels)
    ax1.set_ylabel("chi2/dof across sessions")
    ax1.set_xlabel("Time quartile")
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


# Function: Main entrypoint for time-band stratified refit.

def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(
        description="Time-band quartile stratified beta refit on high-sensitivity VLBI sessions."
    )
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
        help="Per-session source gate: minimum points to include a stable source.",
    )
    ap.add_argument(
        "--min-quartile-points",
        type=int,
        default=8,
        help="Minimum points required in each time quartile refit.",
    )
    ap.add_argument(
        "--min-quartile-sigma",
        type=float,
        default=1.0e-2,
        help="Quality gate: minimum allowed beta_sigma in quartile refit.",
    )
    ap.add_argument(
        "--max-session-pairwise-z",
        type=float,
        default=20.0,
        help="Quality gate: maximum allowed session-level pairwise quartile z.",
    )
    ap.add_argument(
        "--min-valid-quartiles-per-session",
        type=int,
        default=2,
        help="Quality gate: minimum quartiles required after row-quality filtering.",
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
    stable_source_set = set(stable_sources)
    detail_rows: List[Dict[str, object]] = []
    session_rows: List[Dict[str, object]] = []
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

        time_seconds = np.asarray(prepared["time_seconds"], dtype=np.float64)
        q_rows_this_session: List[Dict[str, object]] = []
        for q_label, q_mask in decomp._time_quartile_masks(time_seconds=time_seconds, base_mask=stable_mask):
            qq = np.asarray(q_mask, dtype=bool)
            n_q = int(np.sum(qq))
            if n_q < int(args.min_quartile_points):
                continue

            fit_q = decomp._fit_with_mask(prepared=prepared, mask=qq, nuisance_mode=str(args.nuisance_mode))
            if fit_q is None:
                continue

            beta_q = float(fit_q["beta_est"])
            sigma_q = float(fit_q["beta_sigma"])
            beta_s = float(fit_stable["beta_est"])
            sigma_s = float(fit_stable["beta_sigma"])
            sigma_comb = float(math.sqrt(max(0.0, (sigma_q * sigma_q) + (sigma_s * sigma_s))))
            abs_z_vs_session = float(abs(beta_q - beta_s) / sigma_comb) if sigma_comb > 0.0 else math.nan
            row_pass_sigma = bool(math.isfinite(sigma_q) and (sigma_q >= float(args.min_quartile_sigma)))
            row_pass_points = bool(n_q >= int(args.min_quartile_points))
            row_quality_pass = bool(row_pass_sigma and row_pass_points)
            row_quality_reason = (
                "pass"
                if row_quality_pass
                else (
                    "sigma_below_floor"
                    if (not row_pass_sigma)
                    else "points_below_floor"
                )
            )
            row = {
                "session": session,
                "quartile": str(q_label),
                "max_abs_bendsun_ns": float(sr["max_abs_bendsun_ns"]),
                "n_points_stable_session": n_stable_points,
                "n_stable_sources_present": int(len(allowed_source_set)),
                "n_points_quartile": n_q,
                "beta_est_quartile": beta_q,
                "beta_sigma_quartile": sigma_q,
                "beta_est_session_stable": beta_s,
                "beta_sigma_session_stable": sigma_s,
                "delta_beta_quartile_minus_session": float(beta_q - beta_s),
                "abs_z_quartile_vs_session": abs_z_vs_session,
                "quality_row_gate_pass": row_quality_pass,
                "quality_row_gate_reason": row_quality_reason,
            }
            q_rows_this_session.append(row)
            detail_rows.append(row)

        if not q_rows_this_session:
            continue

        max_abs_z_pairwise = _max_pairwise_z(q_rows_this_session)
        beta_spread = float(
            max([float(r["beta_est_quartile"]) for r in q_rows_this_session])
            - min([float(r["beta_est_quartile"]) for r in q_rows_this_session])
        )
        session_rows.append(
            {
                "session": session,
                "max_abs_bendsun_ns": float(sr["max_abs_bendsun_ns"]),
                "n_points_stable_session": n_stable_points,
                "n_stable_sources_present": int(len(allowed_source_set)),
                "n_quartiles_valid": int(len(q_rows_this_session)),
                "beta_est_session_stable": float(fit_stable["beta_est"]),
                "beta_sigma_session_stable": float(fit_stable["beta_sigma"]),
                "beta_spread_quartiles": beta_spread,
                "max_abs_z_pairwise_quartiles": max_abs_z_pairwise,
            }
        )

    if not detail_rows:
        raise RuntimeError("time-band stratified refit produced zero quartile rows.")

    detail_rows = sorted(detail_rows, key=lambda r: (str(r["session"]), str(r["quartile"])))
    session_rows = sorted(session_rows, key=lambda r: float(r["max_abs_bendsun_ns"]), reverse=True)
    quartile_rows_raw = _build_quartile_consistency_rows(detail_rows)
    detail_rows_quality = [r for r in detail_rows if bool(r.get("quality_row_gate_pass", False))]
    session_quality_rows: List[Dict[str, object]] = []
    allowed_sessions_quality: set[str] = set()
    for sess in sorted(list({str(r["session"]) for r in detail_rows})):
        rq = [r for r in detail_rows_quality if str(r["session"]) == sess]
        n_valid = int(len(rq))
        if n_valid < int(args.min_valid_quartiles_per_session):
            session_quality_rows.append(
                {
                    "session": sess,
                    "n_valid_quartiles_quality": n_valid,
                    "max_abs_z_pairwise_quality": math.nan,
                    "quality_session_gate_pass": False,
                    "quality_session_gate_reason": "insufficient_valid_quartiles",
                }
            )
            continue

        pairwise_quality = _max_pairwise_z(rq)
        pairwise_pass = bool(math.isfinite(pairwise_quality) and (pairwise_quality <= float(args.max_session_pairwise_z)))
        session_quality_rows.append(
            {
                "session": sess,
                "n_valid_quartiles_quality": n_valid,
                "max_abs_z_pairwise_quality": float(pairwise_quality),
                "quality_session_gate_pass": bool(pairwise_pass),
                "quality_session_gate_reason": (
                    "pass"
                    if pairwise_pass
                    else (
                        "pairwise_z_exceeds_gate"
                        if math.isfinite(pairwise_quality)
                        else "pairwise_z_not_finite"
                    )
                ),
            }
        )
        if pairwise_pass:
            allowed_sessions_quality.add(sess)

    quality_lookup: Dict[str, Dict[str, object]] = {
        str(r["session"]): r for r in session_quality_rows
    }
    for row in session_rows:
        sess = str(row["session"])
        qr = quality_lookup.get(sess)
        row["n_valid_quartiles_quality"] = int(qr["n_valid_quartiles_quality"]) if qr else 0
        row["max_abs_z_pairwise_quality"] = float(qr["max_abs_z_pairwise_quality"]) if qr else math.nan
        row["quality_session_gate_pass"] = bool(qr["quality_session_gate_pass"]) if qr else False
        row["quality_session_gate_reason"] = str(qr["quality_session_gate_reason"]) if qr else "not_evaluated"

    detail_rows_gated = [r for r in detail_rows_quality if str(r["session"]) in allowed_sessions_quality]
    quartile_rows_gated = _build_quartile_consistency_rows(detail_rows_gated)

    session_consistency = _weighted_consistency(
        beta=np.asarray([float(r["beta_est_session_stable"]) for r in session_rows], dtype=np.float64),
        sigma=np.asarray([float(r["beta_sigma_session_stable"]) for r in session_rows], dtype=np.float64),
    )
    session_rows_gated = [r for r in session_rows if bool(r.get("quality_session_gate_pass", False))]
    session_consistency_gated = _weighted_consistency(
        beta=np.asarray([float(r["beta_est_session_stable"]) for r in session_rows_gated], dtype=np.float64),
        sigma=np.asarray([float(r["beta_sigma_session_stable"]) for r in session_rows_gated], dtype=np.float64),
    )
    q_chi_raw = np.asarray([float(r["chi2_dof"]) for r in quartile_rows_raw], dtype=np.float64)
    q_chi_gated = np.asarray([float(r["chi2_dof"]) for r in quartile_rows_gated], dtype=np.float64)
    q_chi_raw_finite = q_chi_raw[np.isfinite(q_chi_raw)]
    q_chi_gated_finite = q_chi_gated[np.isfinite(q_chi_gated)]
    quartile_chi2_median_raw = float(np.median(q_chi_raw_finite)) if q_chi_raw_finite.size > 0 else math.nan
    quartile_chi2_median_gated = (
        float(np.median(q_chi_gated_finite)) if q_chi_gated_finite.size > 0 else math.nan
    )
    separation_hint = "time_band_not_sufficient"
    if math.isfinite(quartile_chi2_median_gated) and math.isfinite(float(session_consistency_gated["chi2_dof"])):
        if quartile_chi2_median_gated < float(session_consistency_gated["chi2_dof"]):
            separation_hint = "time_band_explains_part_of_variance"

    out_dir = root / "output" / "vlbi"
    out_dir.mkdir(parents=True, exist_ok=True)
    details_csv = out_dir / "vlbi_beta_timeband_stratified_refit_details.csv"
    session_csv = out_dir / "vlbi_beta_timeband_stratified_refit_session_summary.csv"
    quartile_csv = out_dir / "vlbi_beta_timeband_stratified_refit_quartile_consistency.csv"
    metrics_json = out_dir / "vlbi_beta_timeband_stratified_refit_metrics.json"
    plot_pdf = out_dir / "vlbi_beta_timeband_stratified_refit.pdf"
    plot_png = out_dir / "vlbi_beta_timeband_stratified_refit.png"

    with details_csv.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "session",
            "quartile",
            "max_abs_bendsun_ns",
            "n_points_stable_session",
            "n_stable_sources_present",
            "n_points_quartile",
            "beta_est_quartile",
            "beta_sigma_quartile",
            "beta_est_session_stable",
            "beta_sigma_session_stable",
            "delta_beta_quartile_minus_session",
            "abs_z_quartile_vs_session",
            "quality_row_gate_pass",
            "quality_row_gate_reason",
        ]
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in detail_rows:
            out: List[object] = []
            for col in cols:
                val = row.get(col, "")
                if isinstance(val, float):
                    out.append(f"{val:.16e}" if math.isfinite(val) else "nan")
                else:
                    out.append(val)

            writer.writerow(out)

    with session_csv.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "session",
            "max_abs_bendsun_ns",
            "n_points_stable_session",
            "n_stable_sources_present",
            "n_quartiles_valid",
            "beta_est_session_stable",
            "beta_sigma_session_stable",
            "beta_spread_quartiles",
            "max_abs_z_pairwise_quartiles",
            "n_valid_quartiles_quality",
            "max_abs_z_pairwise_quality",
            "quality_session_gate_pass",
            "quality_session_gate_reason",
        ]
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in session_rows:
            out: List[object] = []
            for col in cols:
                val = row.get(col, "")
                if isinstance(val, float):
                    out.append(f"{val:.16e}" if math.isfinite(val) else "nan")
                else:
                    out.append(val)

            writer.writerow(out)

    with quartile_csv.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "mode",
            "quartile",
            "n_sessions",
            "n_points_total",
            "beta_weighted_mean",
            "beta_weighted_sigma",
            "chi2",
            "dof",
            "chi2_dof",
            "status",
        ]
        writer = csv.writer(f)
        writer.writerow(cols)
        for mode, rows in [("raw", quartile_rows_raw), ("quality_gated", quartile_rows_gated)]:
            for row in rows:
                out: List[object] = []
                for col in cols:
                    if col == "mode":
                        out.append(mode)
                        continue

                    val = row.get(col, "")
                    if isinstance(val, float):
                        out.append(f"{val:.16e}" if math.isfinite(val) else "nan")
                    else:
                        out.append(val)

                writer.writerow(out)

    _plot_summary(
        pdf_path=plot_pdf,
        png_path=plot_png,
        detail_rows=detail_rows,
        quartile_rows_raw=quartile_rows_raw,
        quartile_rows_gated=quartile_rows_gated,
    )
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "description": "high-sensitivity stable-source refit stratified by time quartiles",
            "nuisance_mode": str(args.nuisance_mode),
            "observable_series": str(args.observable_series),
            "min_sensitivity_ns": float(args.min_sensitivity_ns),
            "min_quartile_points": int(args.min_quartile_points),
            "min_quartile_sigma": float(args.min_quartile_sigma),
            "max_session_pairwise_z": float(args.max_session_pairwise_z),
            "min_valid_quartiles_per_session": int(args.min_valid_quartiles_per_session),
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
            "n_sessions_valid": int(len(session_rows)),
            "n_sessions_quality_gated": int(len(session_rows_gated)),
            "n_stable_sources": int(len(stable_sources)),
            "n_quartile_rows_raw": int(len(detail_rows)),
            "n_quartile_rows_after_row_gate": int(len(detail_rows_quality)),
            "n_quartile_rows_after_session_gate": int(len(detail_rows_gated)),
        },
        "stable_sources": stable_sources,
        "stable_source_rows": stable_source_rows,
        "session_summary": session_rows,
        "quartile_detail_rows": detail_rows,
        "quartile_detail_rows_quality": detail_rows_quality,
        "quartile_detail_rows_gated": detail_rows_gated,
        "quartile_consistency_raw": quartile_rows_raw,
        "quartile_consistency_gated": quartile_rows_gated,
        "session_consistency_stable": {
            **session_consistency,
            "status": _status_from_chi2_dof(float(session_consistency["chi2_dof"])),
        },
        "session_consistency_quality_gated": {
            **session_consistency_gated,
            "status": _status_from_chi2_dof(float(session_consistency_gated["chi2_dof"])),
        },
        "separation_diagnostics": {
            "quartile_chi2_dof_median_raw": quartile_chi2_median_raw,
            "quartile_chi2_dof_median_gated": quartile_chi2_median_gated,
            "session_chi2_dof": float(session_consistency["chi2_dof"]),
            "session_chi2_dof_gated": float(session_consistency_gated["chi2_dof"]),
            "separation_hint": separation_hint,
        },
        "outputs": {
            "details_csv": str(details_csv),
            "session_csv": str(session_csv),
            "quartile_csv": str(quartile_csv),
            "metrics_json": str(metrics_json),
            "plot_pdf": str(plot_pdf),
            "plot_png": str(plot_png),
        },
    }
    metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _sync_public(root, [details_csv, session_csv, quartile_csv, metrics_json, plot_pdf, plot_png])
    print("Wrote:", details_csv)
    print("Wrote:", session_csv)
    print("Wrote:", quartile_csv)
    print("Wrote:", metrics_json)
    print("Wrote:", plot_pdf)
    print("Wrote:", plot_png)
    print("Synced:", root / "output" / "public" / "vlbi")
    return 0


# Branch: Execute CLI entrypoint when invoked directly.

if __name__ == "__main__":
    raise SystemExit(main())
