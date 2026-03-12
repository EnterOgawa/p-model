#!/usr/bin/env python3
"""
vlbi_beta_local_mask_two_stage_sweep.py

Local-mask two-stage sweep for a target session.

Goal:
- Avoid full-session exclusion by testing source/time local masks.
- Evaluate each candidate by two-stage gate:
  1) all-sky improvement (threshold sweep chi2/dof)
  2) stable/timeband non-regression
- Select a minimum-intervention candidate if any pass both gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import vlbi_beta_source_filter_decomposition as decomp


# Function: Resolve repository root from script path.
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Function: Normalize a label to a stable scenario suffix token.

def _slugify(text: str) -> str:
    value = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip().lower())
    return value or "x"


# Function: Read all-sky summary rows preserving column order.

def _read_allsky_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"all-sky summary not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        cols = list(r.fieldnames or [])
        rows = [dict(row) for row in r]

    if not cols or not rows:
        raise RuntimeError(f"all-sky summary is empty: {path}")

    return cols, rows


# Function: Write all-sky rows with given field order.

def _write_allsky_rows(path: Path, cols: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in cols})


# Function: Convert string-like values to float safely.

def _to_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
        return out
    except Exception:
        return float(default)


# Function: Parse decomposition components and extract top labels.

def _top_groups_from_components(
    path: Path,
    session: str,
    group_type: str,
    top_n: int,
) -> List[str]:
    if not path.exists():
        return []

    rows: List[Tuple[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if str(row.get("session") or "").strip() != str(session):
                continue

            if str(row.get("group_type") or "").strip() != str(group_type):
                continue

            label = str(row.get("group_label") or "").strip()
            impact = abs(_to_float(row.get("impact_beta_all_minus_drop"), default=math.nan))
            if not label or not math.isfinite(impact):
                continue

            rows.append((label, impact))

    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    out: List[str] = []
    for label, _ in rows[: int(max(0, top_n))]:
        if label not in out:
            out.append(label)

    return out


# Function: Run subprocess command and return structured result.

def _run(cmd: List[str], cwd: Path) -> Dict[str, object]:
    cp = subprocess.run(cmd, cwd=str(cwd), check=False, capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": int(cp.returncode),
        "ok": bool(cp.returncode == 0),
        "stdout_tail": "\n".join((cp.stdout or "").splitlines()[-10:]),
        "stderr_tail": "\n".join((cp.stderr or "").splitlines()[-10:]),
    }


# Function: Copy canonical outputs to scenario-suffixed files.

def _snapshot_outputs(base_public: Path, scenario: str) -> None:
    pairs = [
        ("vlbi_beta_source_session_matrix_details.csv", f"vlbi_beta_source_session_matrix_details_{scenario}.csv"),
        (
            "vlbi_beta_source_session_matrix_source_summary.csv",
            f"vlbi_beta_source_session_matrix_source_summary_{scenario}.csv",
        ),
        ("vlbi_beta_source_session_matrix_metrics.json", f"vlbi_beta_source_session_matrix_metrics_{scenario}.json"),
        ("vlbi_beta_source_session_matrix.pdf", f"vlbi_beta_source_session_matrix_{scenario}.pdf"),
        ("vlbi_beta_source_session_matrix.png", f"vlbi_beta_source_session_matrix_{scenario}.png"),
        ("vlbi_beta_stable_source_refit_summary.csv", f"vlbi_beta_stable_source_refit_summary_{scenario}.csv"),
        (
            "vlbi_beta_stable_source_refit_source_presence.csv",
            f"vlbi_beta_stable_source_refit_source_presence_{scenario}.csv",
        ),
        ("vlbi_beta_stable_source_refit_metrics.json", f"vlbi_beta_stable_source_refit_metrics_{scenario}.json"),
        ("vlbi_beta_stable_source_refit.pdf", f"vlbi_beta_stable_source_refit_{scenario}.pdf"),
        ("vlbi_beta_stable_source_refit.png", f"vlbi_beta_stable_source_refit_{scenario}.png"),
        (
            "vlbi_beta_timeband_stratified_refit_details.csv",
            f"vlbi_beta_timeband_stratified_refit_details_{scenario}.csv",
        ),
        (
            "vlbi_beta_timeband_stratified_refit_session_summary.csv",
            f"vlbi_beta_timeband_stratified_refit_session_summary_{scenario}.csv",
        ),
        (
            "vlbi_beta_timeband_stratified_refit_quartile_consistency.csv",
            f"vlbi_beta_timeband_stratified_refit_quartile_consistency_{scenario}.csv",
        ),
        (
            "vlbi_beta_timeband_stratified_refit_metrics.json",
            f"vlbi_beta_timeband_stratified_refit_metrics_{scenario}.json",
        ),
        ("vlbi_beta_timeband_stratified_refit.pdf", f"vlbi_beta_timeband_stratified_refit_{scenario}.pdf"),
        ("vlbi_beta_timeband_stratified_refit.png", f"vlbi_beta_timeband_stratified_refit_{scenario}.png"),
        ("vlbi_high_sensitivity_threshold_sweep.csv", f"vlbi_high_sensitivity_threshold_sweep_{scenario}.csv"),
        (
            "vlbi_high_sensitivity_threshold_sweep_metrics.json",
            f"vlbi_high_sensitivity_threshold_sweep_metrics_{scenario}.json",
        ),
        ("vlbi_high_sensitivity_threshold_sweep.pdf", f"vlbi_high_sensitivity_threshold_sweep_{scenario}.pdf"),
        ("vlbi_high_sensitivity_threshold_sweep.png", f"vlbi_high_sensitivity_threshold_sweep_{scenario}.png"),
    ]
    for src_name, dst_name in pairs:
        src = base_public / src_name
        if src.exists():
            shutil.copy2(src, base_public / dst_name)


# Function: Sync files from private output/vlbi to public.

def _sync_public(root: Path, outputs: Sequence[Path]) -> None:
    dst = root / "output" / "public" / "vlbi"
    dst.mkdir(parents=True, exist_ok=True)
    for p in outputs:
        if p.exists():
            shutil.copy2(p, dst / p.name)


# Function: Read chain metrics from canonical outputs.

def _read_chain_metrics(base_public: Path) -> Dict[str, object]:
    stable = json.loads((base_public / "vlbi_beta_stable_source_refit_metrics.json").read_text(encoding="utf-8"))
    timeband = json.loads((base_public / "vlbi_beta_timeband_stratified_refit_metrics.json").read_text(encoding="utf-8"))
    threshold = json.loads((base_public / "vlbi_high_sensitivity_threshold_sweep_metrics.json").read_text(encoding="utf-8"))
    return {
        "stable_chi2_dof": _to_float(stable.get("consistency", {}).get("chi2_dof")),
        "stable_status": str(stable.get("consistency", {}).get("status") or ""),
        "timeband_chi2_dof": _to_float(timeband.get("session_consistency_stable", {}).get("chi2_dof")),
        "timeband_status": str(timeband.get("session_consistency_stable", {}).get("status") or ""),
        "threshold_chi2_dof": _to_float(threshold.get("recommendation", {}).get("recommended_chi2_dof")),
        "threshold_status": str(threshold.get("recommendation", {}).get("recommended_status") or ""),
        "threshold_ns": _to_float(threshold.get("recommendation", {}).get("recommended_threshold_ns")),
    }


# Function: Build candidate masks for local intervention.

def _build_candidates(
    source_labels: Sequence[str],
    quartile_labels: Sequence[str],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    out.append({"scenario": "keep_all", "kind": "keep", "source": "", "quartile": ""})
    for src in source_labels:
        out.append(
            {
                "scenario": f"local_source_{_slugify(src)}",
                "kind": "source",
                "source": str(src),
                "quartile": "",
            }
        )

    for q in quartile_labels:
        out.append(
            {
                "scenario": f"local_time_{_slugify(q)}",
                "kind": "time",
                "source": "",
                "quartile": str(q),
            }
        )

    for src in source_labels:
        for q in quartile_labels:
            out.append(
                {
                    "scenario": f"local_source_{_slugify(src)}_or_time_{_slugify(q)}",
                    "kind": "source_or_time",
                    "source": str(src),
                    "quartile": str(q),
                }
            )

    uniq: List[Dict[str, object]] = []
    seen = set()
    for row in out:
        sc = str(row["scenario"])
        if sc in seen:
            continue

        seen.add(sc)
        uniq.append(row)

    return uniq


# Function: Main entrypoint for local-mask two-stage sweep.

def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Local source/time mask sweep with two-stage gate for VLBI watchpack.")
    ap.add_argument(
        "--base-allsky-summary",
        type=Path,
        default=root / "output" / "vlbi" / "vlbi_allsky_beta_consistency_summary_keep_all.csv",
        help="Base all-sky summary CSV for keep_all scenario.",
    )
    ap.add_argument(
        "--components-csv",
        type=Path,
        default=root / "output" / "public" / "vlbi" / "vlbi_high_sensitivity_factor_decomposition_components.csv",
        help="High-sensitivity decomposition components CSV.",
    )
    ap.add_argument(
        "--target-session",
        type=str,
        default="20MAY04XA",
        help="Target session label for local mask interventions.",
    )
    ap.add_argument(
        "--input-root",
        type=Path,
        default=root / "data" / "vlbi" / "sources" / "vgosdb" / "20MAY04XA" / "extracted",
        help="Extracted vgosDb directory of the target session.",
    )
    ap.add_argument(
        "--session-root",
        type=Path,
        default=root / "data" / "vlbi" / "sources" / "vgosdb",
        help="Root containing per-session extracted vgosDb data.",
    )
    ap.add_argument(
        "--top-sources",
        type=int,
        default=3,
        help="Number of source labels (by impact rank) to include in local mask candidates.",
    )
    ap.add_argument(
        "--top-time-quartiles",
        type=int,
        default=2,
        help="Number of time-quartile labels (by impact rank) to include in local mask candidates.",
    )
    ap.add_argument(
        "--min-allsky-relative-improvement",
        type=float,
        default=0.05,
        help="Stage-1 gate minimum relative improvement in threshold chi2/dof.",
    )
    ap.add_argument(
        "--max-stable-delta-chi2-dof",
        type=float,
        default=0.0,
        help="Stage-2 gate maximum allowed stable chi2/dof delta vs keep_all.",
    )
    ap.add_argument(
        "--max-timeband-delta-chi2-dof",
        type=float,
        default=0.0,
        help="Stage-2 gate maximum allowed timeband chi2/dof delta vs keep_all.",
    )
    ap.add_argument(
        "--nuisance-mode",
        type=str,
        default="baseline_intercept_linear",
        choices=["none", "baseline_intercept", "baseline_intercept_linear"],
        help="Nuisance mode for all refits.",
    )
    ap.add_argument(
        "--observable-series",
        type=str,
        default="full",
        choices=["full", "fringe"],
        help="Observable series for reconstruction.",
    )
    ap.add_argument(
        "--min-template-abs",
        type=float,
        default=1.0e-14,
        help="Absolute threshold on |Cal-BendSun template| [s].",
    )
    args = ap.parse_args()

    target_session = str(args.target_session).strip()
    input_root = args.input_root.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"target input root not found: {input_root}")

    cols, base_rows = _read_allsky_rows(args.base_allsky_summary.resolve())
    target_rows = [r for r in base_rows if str(r.get("session") or "").strip() == target_session]
    if not target_rows:
        raise RuntimeError(f"target session row not found in base all-sky summary: {target_session}")

    top_sources = _top_groups_from_components(
        path=args.components_csv.resolve(),
        session=target_session,
        group_type="source",
        top_n=int(max(0, args.top_sources)),
    )
    top_quartiles = _top_groups_from_components(
        path=args.components_csv.resolve(),
        session=target_session,
        group_type="time_quartile",
        top_n=int(max(0, args.top_time_quartiles)),
    )
    if not top_sources:
        top_sources = ["0059+581"]

    if not top_quartiles:
        top_quartiles = ["Q2"]

    candidates = _build_candidates(source_labels=top_sources, quartile_labels=top_quartiles)

    index = decomp.core._scan_netcdf_variables(input_root)
    if not index:
        raise RuntimeError(f"failed to scan netCDF variables: {input_root}")

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
        raise RuntimeError("base mask has fewer than 3 points.")

    fit_all = decomp._fit_with_mask(prepared=prepared, mask=base_mask, nuisance_mode=str(args.nuisance_mode))
    if fit_all is None:
        raise RuntimeError("base fit failed on target session.")

    source_vec = np.asarray([str(v) for v in np.asarray(prepared["source_vec"], dtype=object)], dtype=object)
    time_seconds = np.asarray(prepared["time_seconds"], dtype=np.float64)
    quartile_masks = {q: np.asarray(m, dtype=bool) for q, m in decomp._time_quartile_masks(time_seconds, base_mask)}
    template = np.asarray(prepared["template"], dtype=np.float64)
    base_points = int(np.sum(base_mask))

    base_public = root / "output" / "public" / "vlbi"
    out_dir = root / "output" / "vlbi"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows: List[Dict[str, object]] = []
    runs: List[Dict[str, object]] = []

    keep_ref: Optional[Dict[str, object]] = None
    py = sys.executable
    for cand in candidates:
        scenario = str(cand["scenario"])
        kind = str(cand["kind"])
        src = str(cand["source"])
        q = str(cand["quartile"])
        rm_mask = np.zeros_like(base_mask, dtype=bool)
        if kind == "source":
            rm_mask = base_mask & (source_vec == src)
        elif kind == "time":
            rm_mask = np.asarray(quartile_masks.get(q, np.zeros_like(base_mask, dtype=bool)), dtype=bool)
        elif kind == "source_or_time":
            src_mask = base_mask & (source_vec == src)
            q_mask = np.asarray(quartile_masks.get(q, np.zeros_like(base_mask, dtype=bool)), dtype=bool)
            rm_mask = src_mask | q_mask

        keep_mask = base_mask & (~rm_mask)
        n_removed = int(np.sum(base_mask & rm_mask))
        if int(np.sum(keep_mask)) < 3:
            candidate_rows.append(
                {
                    "scenario": scenario,
                    "kind": kind,
                    "source": src,
                    "quartile": q,
                    "n_removed": n_removed,
                    "removed_fraction": float(n_removed / base_points) if base_points > 0 else math.nan,
                    "fit_ok": False,
                    "fit_reason": "too_few_points_after_mask",
                }
            )
            continue

        fit = decomp._fit_with_mask(prepared=prepared, mask=keep_mask, nuisance_mode=str(args.nuisance_mode))
        if fit is None:
            candidate_rows.append(
                {
                    "scenario": scenario,
                    "kind": kind,
                    "source": src,
                    "quartile": q,
                    "n_removed": n_removed,
                    "removed_fraction": float(n_removed / base_points) if base_points > 0 else math.nan,
                    "fit_ok": False,
                    "fit_reason": "fit_failed",
                }
            )
            continue

        kept_template = template[keep_mask]
        max_abs_bendsun_ns = float(np.max(np.abs(kept_template)) * 1.0e9) if kept_template.size > 0 else math.nan
        allsky_rows = [dict(r) for r in base_rows]
        for row in allsky_rows:
            if str(row.get("session") or "").strip() != target_session:
                continue

            row["n_points"] = str(int(fit["n_points"]))
            row["beta_est"] = f"{float(fit['beta_est']):.16e}"
            row["beta_sigma"] = f"{float(fit['beta_sigma']):.16e}"
            row["delta_beta"] = f"{float(fit['delta_beta']):.16e}"
            row["max_abs_bendsun_ns"] = f"{max_abs_bendsun_ns:.16e}"

        allsky_path = out_dir / f"vlbi_allsky_beta_consistency_summary_{scenario}.csv"
        _write_allsky_rows(allsky_path, cols, allsky_rows)
        _sync_public(root, [allsky_path])

        commands = [
            [
                py,
                "-B",
                str((root / "scripts" / "vlbi" / "vlbi_beta_source_session_matrix.py").resolve()),
                "--allsky-summary",
                str(allsky_path),
                "--session-root",
                str(args.session_root.resolve()),
                "--min-sensitivity-ns",
                "10",
                "--min-source-points",
                "20",
                "--nuisance-mode",
                str(args.nuisance_mode),
                "--observable-series",
                str(args.observable_series),
            ],
            [
                py,
                "-B",
                str((root / "scripts" / "vlbi" / "vlbi_beta_stable_source_refit.py").resolve()),
                "--allsky-summary",
                str(allsky_path),
                "--source-summary",
                str(base_public / "vlbi_beta_source_session_matrix_source_summary.csv"),
                "--session-root",
                str(args.session_root.resolve()),
                "--min-sensitivity-ns",
                "10",
                "--min-source-sessions",
                "3",
                "--max-source-chi2-dof",
                "2",
                "--require-source-status",
                "pass",
                "--min-source-points-per-session",
                "20",
                "--nuisance-mode",
                str(args.nuisance_mode),
                "--observable-series",
                str(args.observable_series),
            ],
            [
                py,
                "-B",
                str((root / "scripts" / "vlbi" / "vlbi_beta_timeband_stratified_refit.py").resolve()),
                "--allsky-summary",
                str(allsky_path),
                "--source-summary",
                str(base_public / "vlbi_beta_source_session_matrix_source_summary.csv"),
                "--session-root",
                str(args.session_root.resolve()),
                "--min-sensitivity-ns",
                "10",
                "--min-source-sessions",
                "3",
                "--max-source-chi2-dof",
                "2",
                "--require-source-status",
                "pass",
                "--min-source-points-per-session",
                "20",
                "--min-quartile-points",
                "8",
                "--min-quartile-sigma",
                "0.01",
                "--max-session-pairwise-z",
                "20",
                "--min-valid-quartiles-per-session",
                "2",
                "--nuisance-mode",
                str(args.nuisance_mode),
                "--observable-series",
                str(args.observable_series),
            ],
            [
                py,
                "-B",
                str((root / "scripts" / "vlbi" / "vlbi_beta_high_sensitivity_threshold_sweep.py").resolve()),
                "--allsky-summary",
                str(allsky_path),
                "--thresholds",
                "10,12,15,20",
                "--min-sessions-operational",
                "3",
            ],
        ]
        cmd_results: List[Dict[str, object]] = []
        ok = True
        for cmd in commands:
            res = _run(cmd, cwd=root)
            cmd_results.append(res)
            if not bool(res["ok"]):
                ok = False
                break

        runs.append({"scenario": scenario, "commands": cmd_results})
        if not ok:
            candidate_rows.append(
                {
                    "scenario": scenario,
                    "kind": kind,
                    "source": src,
                    "quartile": q,
                    "n_removed": n_removed,
                    "removed_fraction": float(n_removed / base_points) if base_points > 0 else math.nan,
                    "fit_ok": True,
                    "fit_reason": "chain_failed",
                }
            )
            continue

        _snapshot_outputs(base_public=base_public, scenario=scenario)
        chain = _read_chain_metrics(base_public=base_public)
        row = {
            "scenario": scenario,
            "kind": kind,
            "source": src,
            "quartile": q,
            "allsky_summary_csv": str(allsky_path.resolve()),
            "n_removed": int(n_removed),
            "removed_fraction": float(n_removed / base_points) if base_points > 0 else math.nan,
            "fit_ok": True,
            "target_beta_est": float(fit["beta_est"]),
            "target_beta_sigma": float(fit["beta_sigma"]),
            "target_max_abs_bendsun_ns": float(max_abs_bendsun_ns),
            "stable_chi2_dof": float(chain["stable_chi2_dof"]),
            "stable_status": str(chain["stable_status"]),
            "timeband_chi2_dof": float(chain["timeband_chi2_dof"]),
            "timeband_status": str(chain["timeband_status"]),
            "threshold_chi2_dof": float(chain["threshold_chi2_dof"]),
            "threshold_status": str(chain["threshold_status"]),
            "threshold_ns": float(chain["threshold_ns"]),
        }
        candidate_rows.append(row)
        if scenario == "keep_all":
            keep_ref = row

    if keep_ref is None:
        raise RuntimeError("keep_all scenario failed; cannot evaluate two-stage local masks.")

    for row in candidate_rows:
        if not bool(row.get("fit_ok")) or ("stable_chi2_dof" not in row):
            row["allsky_relative_improvement"] = math.nan
            row["stable_delta_chi2_dof"] = math.nan
            row["timeband_delta_chi2_dof"] = math.nan
            row["stage1_pass"] = False
            row["stage2_pass"] = False
            row["status_regression"] = False
            row["overall_pass"] = False
            continue

        thr_keep = float(keep_ref["threshold_chi2_dof"])
        thr_now = float(row["threshold_chi2_dof"])
        allsky_rel = float((thr_keep - thr_now) / thr_keep) if (math.isfinite(thr_keep) and thr_keep > 0.0) else math.nan
        stable_delta = float(row["stable_chi2_dof"]) - float(keep_ref["stable_chi2_dof"])
        time_delta = float(row["timeband_chi2_dof"]) - float(keep_ref["timeband_chi2_dof"])
        stage1_pass = bool(
            math.isfinite(allsky_rel) and (allsky_rel >= float(args.min_allsky_relative_improvement))
        )
        stage2_pass = bool(
            math.isfinite(stable_delta)
            and math.isfinite(time_delta)
            and (stable_delta <= float(args.max_stable_delta_chi2_dof))
            and (time_delta <= float(args.max_timeband_delta_chi2_dof))
        )
        keep_st = str(keep_ref["stable_status"])
        keep_tb = str(keep_ref["timeband_status"])
        now_st = str(row["stable_status"])
        now_tb = str(row["timeband_status"])
        status_reg = bool(
            (keep_st in {"pass", "watch"} and now_st == "reject")
            or (keep_tb in {"pass", "watch"} and now_tb == "reject")
        )
        row["allsky_relative_improvement"] = float(allsky_rel)
        row["stable_delta_chi2_dof"] = float(stable_delta)
        row["timeband_delta_chi2_dof"] = float(time_delta)
        row["stage1_pass"] = bool(stage1_pass)
        row["stage2_pass"] = bool(stage2_pass)
        row["status_regression"] = bool(status_reg)
        row["overall_pass"] = bool(stage1_pass and stage2_pass and (not status_reg))

    pass_rows = [
        r
        for r in candidate_rows
        if bool(r.get("fit_ok"))
        and bool(r.get("overall_pass"))
        and str(r.get("scenario") or "") != "keep_all"
    ]
    if pass_rows:
        pass_rows = sorted(
            pass_rows,
            key=lambda r: (
                float(r.get("removed_fraction", math.inf)),
                -float(r.get("allsky_relative_improvement", -math.inf)),
                float(r.get("stable_delta_chi2_dof", math.inf)) + float(r.get("timeband_delta_chi2_dof", math.inf)),
            ),
        )
        selected = pass_rows[0]
        selected_reason = "selected_minimum_intervention_candidate_passing_two_stage_gate"
    else:
        selected = keep_ref
        selected_reason = "no_local_mask_candidate_passed_two_stage_gate_keep_all"

    selected_scenario = str(selected.get("scenario") or "keep_all")
    selected_policy = (
        "keep_all" if selected_scenario == "keep_all" else f"local_mask:{selected_scenario}"
    )
    selected_allsky_csv = str(selected.get("allsky_summary_csv") or str(args.base_allsky_summary.resolve()))
    selected_threshold_ns = _to_float(selected.get("threshold_ns"), default=10.0)

    summary_csv = out_dir / "vlbi_beta_local_mask_two_stage_sweep_summary.csv"
    metrics_json = out_dir / "vlbi_beta_local_mask_two_stage_sweep_metrics.json"
    plot_pdf = out_dir / "vlbi_beta_local_mask_two_stage_sweep.pdf"
    plot_png = out_dir / "vlbi_beta_local_mask_two_stage_sweep.png"
    cols_out = [
        "scenario",
        "kind",
        "source",
        "quartile",
        "allsky_summary_csv",
        "n_removed",
        "removed_fraction",
        "fit_ok",
        "fit_reason",
        "target_beta_est",
        "target_beta_sigma",
        "target_max_abs_bendsun_ns",
        "stable_chi2_dof",
        "stable_status",
        "timeband_chi2_dof",
        "timeband_status",
        "threshold_chi2_dof",
        "threshold_status",
        "threshold_ns",
        "allsky_relative_improvement",
        "stable_delta_chi2_dof",
        "timeband_delta_chi2_dof",
        "stage1_pass",
        "stage2_pass",
        "status_regression",
        "overall_pass",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols_out)
        w.writeheader()
        for row in candidate_rows:
            w.writerow({k: row.get(k, "") for k in cols_out})

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "step": "8.7.46.25",
        "target_session": target_session,
        "candidate_generation": {
            "top_sources": top_sources,
            "top_time_quartiles": top_quartiles,
            "n_candidates": int(len(candidates)),
        },
        "gate": {
            "min_allsky_relative_improvement": float(args.min_allsky_relative_improvement),
            "max_stable_delta_chi2_dof": float(args.max_stable_delta_chi2_dof),
            "max_timeband_delta_chi2_dof": float(args.max_timeband_delta_chi2_dof),
        },
        "rows": candidate_rows,
        "decision": {
            "selected_scenario": selected_scenario,
            "selected_policy": selected_policy,
            "selected_allsky_summary_csv": selected_allsky_csv,
            "selected_threshold_ns": float(selected_threshold_ns),
            "reason": selected_reason,
        },
        "runs": runs,
        "outputs": {
            "summary_csv": str(summary_csv.resolve()),
            "metrics_json": str(metrics_json.resolve()),
            "plot_pdf": str(plot_pdf.resolve()),
            "plot_png": str(plot_png.resolve()),
        },
    }
    metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(9.6, 5.0))
        x_labels: List[str] = []
        y_vals: List[float] = []
        colors: List[str] = []
        for row in candidate_rows:
            if not bool(row.get("fit_ok")) or ("allsky_relative_improvement" not in row):
                continue

            x_labels.append(str(row["scenario"]))
            y_vals.append(float(row["allsky_relative_improvement"]))
            colors.append("tab:green" if bool(row.get("overall_pass")) else "tab:blue")

        if x_labels:
            x = np.arange(len(x_labels), dtype=np.float64)
            ax.bar(x, y_vals, color=colors, alpha=0.9)
            ax.axhline(float(args.min_allsky_relative_improvement), color="tab:red", linestyle="--", linewidth=1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=30, ha="right")
            ax.set_ylabel("all-sky relative improvement")
            ax.set_title("Local mask candidates (stage-1 metric)")
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(str(plot_pdf))
            fig.savefig(str(plot_png), dpi=170)
            plt.close(fig)
    except Exception:
        pass

    outputs = [summary_csv, metrics_json, plot_pdf, plot_png]
    _sync_public(root, outputs)
    for p in outputs:
        print(f"Wrote: {p}")

    print("Synced:", root / "output" / "public" / "vlbi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

