#!/usr/bin/env python3
"""
Step 8.7.46.28:
Evaluate two-session simultaneous local-source interventions under the two-stage gate.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import vlbi_beta_source_filter_decomposition as decomp


# Function: Resolve repository root from script location.
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Function: Normalize a token to stable scenario-safe slug.

def _slugify(text: str) -> str:
    value = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip().lower())
    return value or "x"


# Function: Parse optional session=source override string.

def _parse_overrides(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for token in [t.strip() for t in str(text).split(",") if t.strip()]:
        if "=" not in token:
            continue

        k, v = token.split("=", 1)
        ks = str(k).strip().upper()
        vs = str(v).strip()
        if ks and vs:
            out[ks] = vs

    return out


# Function: Safe float parse helper.

def _to_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)


# Function: Read all-sky CSV preserving field order.

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


# Function: Write all-sky rows in deterministic column order.

def _write_allsky_rows(path: Path, cols: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in cols})


# Function: Read high-sensitivity factor summary and map session -> top source.

def _session_top_source(
    factor_summary_csv: Path,
    min_sensitivity_ns: float,
) -> Dict[str, str]:
    if not factor_summary_csv.exists():
        raise FileNotFoundError(f"factor summary CSV not found: {factor_summary_csv}")

    out: Dict[str, str] = {}
    with factor_summary_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            session = str(row.get("session") or "").strip().upper()
            top_source = str(row.get("top_source_group") or "").strip()
            sens = _to_float(row.get("max_abs_bendsun_ns"), default=math.nan)
            if not session or not top_source or (not math.isfinite(sens)):
                continue

            if sens < float(min_sensitivity_ns):
                continue

            out[session] = top_source

    return out


# Function: Execute one command and return structured diagnostics.

def _run(cmd: List[str], cwd: Path) -> Dict[str, object]:
    cp = subprocess.run(cmd, cwd=str(cwd), check=False, capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": int(cp.returncode),
        "ok": bool(cp.returncode == 0),
        "stdout_tail": "\n".join((cp.stdout or "").splitlines()[-10:]),
        "stderr_tail": "\n".join((cp.stderr or "").splitlines()[-10:]),
    }


# Function: Sync files from private output/vlbi to public output/public/vlbi.

def _sync_public(root: Path, outputs: Sequence[Path]) -> None:
    dst = root / "output" / "public" / "vlbi"
    dst.mkdir(parents=True, exist_ok=True)
    for p in outputs:
        if p.exists():
            shutil.copy2(p, dst / p.name)


# Function: Read chain metrics from canonical output names.

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


# Function: Copy canonical outputs to scenario-suffixed snapshots.

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


# Function: Build prepared vectors cache for one session.

def _prepare_session_cache(
    session_root: Path,
    session: str,
    observable_series: str,
    min_template_abs: float,
) -> Dict[str, object]:
    input_root = session_root / session / "extracted"
    if not input_root.exists():
        raise FileNotFoundError(f"extracted session dir missing: {input_root}")

    index = decomp.core._scan_netcdf_variables(input_root)
    if not index:
        raise RuntimeError(f"failed to scan netCDF variables: {input_root}")

    prepared = decomp._prepare_vectors(
        index=index,
        input_root=input_root,
        band_index=0,
        observable_series=str(observable_series),
        threshold_s=float(min_template_abs),
        disable_flag_filter=False,
    )
    base_mask = np.asarray(prepared["mask_base"], dtype=bool)
    if int(np.sum(base_mask)) < 3:
        raise RuntimeError(f"base mask too small: session={session} n={int(np.sum(base_mask))}")

    return {
        "prepared": prepared,
        "base_mask": base_mask,
        "source_vec": np.asarray([str(v) for v in np.asarray(prepared["source_vec"], dtype=object)], dtype=object),
        "template": np.asarray(prepared["template"], dtype=np.float64),
    }


# Function: Build candidate scenario list from high-sensitivity sessions.

def _build_candidates(
    session_to_source: Dict[str, str],
    pair_size: int,
) -> List[Dict[str, object]]:
    sessions = sorted(session_to_source.keys())
    out: List[Dict[str, object]] = [{"scenario": "keep_current", "kind": "keep", "masks": {}}]
    if len(sessions) < int(pair_size):
        return out

    for combo in itertools.combinations(sessions, int(pair_size)):
        masks = {s: session_to_source[s] for s in combo}
        parts: List[str] = []
        for s in combo:
            parts.append(f"{s.lower()}_{_slugify(session_to_source[s])}")

        sc = "pair_source_" + "__".join(parts)
        out.append({"scenario": sc, "kind": "pair_source", "masks": masks})

    return out


# Function: Main entrypoint for multi-session local-mask two-stage sweep.

def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Two-session local-source sweep with two-stage gate (8.7.46.28).")
    ap.add_argument(
        "--base-allsky-summary",
        type=Path,
        default=root / "output" / "vlbi" / "vlbi_allsky_beta_consistency_summary_local_source_2000_472.csv",
        help="Baseline all-sky summary CSV (current canonical local mask).",
    )
    ap.add_argument(
        "--factor-summary",
        type=Path,
        default=root / "output" / "public" / "vlbi" / "vlbi_high_sensitivity_factor_decomposition_summary.csv",
        help="High-sensitivity factor decomposition summary CSV.",
    )
    ap.add_argument(
        "--session-root",
        type=Path,
        default=root / "data" / "vlbi" / "sources" / "vgosdb",
        help="Root containing per-session extracted vgosDb data.",
    )
    ap.add_argument(
        "--pair-size",
        type=int,
        default=2,
        help="Number of sessions to intervene simultaneously.",
    )
    ap.add_argument(
        "--min-sensitivity-ns",
        type=float,
        default=10.0,
        help="High-sensitivity threshold for candidate sessions.",
    )
    ap.add_argument(
        "--session-source-overrides",
        type=str,
        default="20MAY04XA=2000+472",
        help="Comma-separated session=source overrides (applied over factor-summary top source).",
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
        help="Stage-2 gate maximum allowed stable chi2/dof delta vs keep_current.",
    )
    ap.add_argument(
        "--max-timeband-delta-chi2-dof",
        type=float,
        default=0.0,
        help="Stage-2 gate maximum allowed timeband chi2/dof delta vs keep_current.",
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
    ap.add_argument(
        "--apply-selected",
        action="store_true",
        help="Apply selected decision to canonical chain using watchpack_apply_chain.py.",
    )
    args = ap.parse_args()

    if int(args.pair_size) < 2:
        raise ValueError("--pair-size must be >= 2 for multi-session sweep.")

    cols, base_rows = _read_allsky_rows(args.base_allsky_summary.resolve())
    factor_map = _session_top_source(
        factor_summary_csv=args.factor_summary.resolve(),
        min_sensitivity_ns=float(args.min_sensitivity_ns),
    )
    overrides = _parse_overrides(args.session_source_overrides)
    for session, source in overrides.items():
        factor_map[str(session).strip().upper()] = str(source).strip()

    high_sens_sessions: List[str] = []
    for row in base_rows:
        session = str(row.get("session") or "").strip().upper()
        sens = _to_float(row.get("max_abs_bendsun_ns"), default=math.nan)
        if not session or (not math.isfinite(sens)):
            continue

        if sens < float(args.min_sensitivity_ns):
            continue

        if session in factor_map:
            high_sens_sessions.append(session)

    high_sens_sessions = sorted(set(high_sens_sessions))
    session_to_source = {s: factor_map[s] for s in high_sens_sessions}
    candidates = _build_candidates(session_to_source=session_to_source, pair_size=int(args.pair_size))

    session_root = args.session_root.resolve()
    session_cache: Dict[str, Dict[str, object]] = {}
    for session in high_sens_sessions:
        session_cache[session] = _prepare_session_cache(
            session_root=session_root,
            session=session,
            observable_series=str(args.observable_series),
            min_template_abs=float(args.min_template_abs),
        )

    base_public = root / "output" / "public" / "vlbi"
    out_dir = root / "output" / "vlbi"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, object]] = []
    candidate_rows: List[Dict[str, object]] = []
    keep_ref: Dict[str, object] | None = None
    py = sys.executable
    for cand in candidates:
        scenario = str(cand["scenario"])
        masks = dict(cand.get("masks") or {})
        modified_rows = [dict(r) for r in base_rows]

        session_fit_rows: List[Dict[str, object]] = []
        total_removed = 0
        total_base_points = 0
        fit_failed = False
        fail_reason = ""
        for session, source_label in masks.items():
            cache = session_cache.get(str(session).strip().upper())
            if cache is None:
                fit_failed = True
                fail_reason = f"session_cache_missing:{session}"
                break

            prepared = cache["prepared"]
            base_mask = np.asarray(cache["base_mask"], dtype=bool)
            source_vec = np.asarray(cache["source_vec"], dtype=object)
            template = np.asarray(cache["template"], dtype=np.float64)

            rm_mask = base_mask & (source_vec == str(source_label))
            keep_mask = base_mask & (~rm_mask)
            n_removed = int(np.sum(base_mask & rm_mask))
            n_base = int(np.sum(base_mask))
            if int(np.sum(keep_mask)) < 3:
                fit_failed = True
                fail_reason = f"too_few_points_after_mask:{session}"
                break

            fit = decomp._fit_with_mask(prepared=prepared, mask=keep_mask, nuisance_mode=str(args.nuisance_mode))
            if fit is None:
                fit_failed = True
                fail_reason = f"fit_failed:{session}"
                break

            kept_template = template[keep_mask]
            max_abs_bendsun_ns = float(np.max(np.abs(kept_template)) * 1.0e9) if kept_template.size > 0 else math.nan
            session_fit_rows.append(
                {
                    "session": session,
                    "source": str(source_label),
                    "n_removed": int(n_removed),
                    "n_base": int(n_base),
                    "n_points": int(fit["n_points"]),
                    "beta_est": float(fit["beta_est"]),
                    "beta_sigma": float(fit["beta_sigma"]),
                    "delta_beta": float(fit["delta_beta"]),
                    "max_abs_bendsun_ns": float(max_abs_bendsun_ns),
                }
            )
            total_removed += int(n_removed)
            total_base_points += int(n_base)

            for row in modified_rows:
                if str(row.get("session") or "").strip().upper() != str(session):
                    continue

                row["n_points"] = str(int(fit["n_points"]))
                row["beta_est"] = f"{float(fit['beta_est']):.16e}"
                row["beta_sigma"] = f"{float(fit['beta_sigma']):.16e}"
                row["delta_beta"] = f"{float(fit['delta_beta']):.16e}"
                row["max_abs_bendsun_ns"] = f"{max_abs_bendsun_ns:.16e}"

        if fit_failed:
            candidate_rows.append(
                {
                    "scenario": scenario,
                    "kind": str(cand.get("kind") or ""),
                    "mask_spec": json.dumps(masks, ensure_ascii=False),
                    "fit_ok": False,
                    "fit_reason": fail_reason,
                    "n_sessions_masked": int(len(masks)),
                    "n_removed_total": int(total_removed),
                    "removed_fraction_total": (
                        float(total_removed / total_base_points) if total_base_points > 0 else math.nan
                    ),
                }
            )
            continue

        allsky_path = out_dir / f"vlbi_allsky_beta_consistency_summary_{scenario}.csv"
        _write_allsky_rows(allsky_path, cols, modified_rows)
        _sync_public(root, [allsky_path])

        commands = [
            [
                py,
                "-B",
                str((root / "scripts" / "vlbi" / "vlbi_beta_source_session_matrix.py").resolve()),
                "--allsky-summary",
                str(allsky_path),
                "--session-root",
                str(session_root),
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
                str(session_root),
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
                str(session_root),
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
                    "kind": str(cand.get("kind") or ""),
                    "mask_spec": json.dumps(masks, ensure_ascii=False),
                    "allsky_summary_csv": str(allsky_path.resolve()),
                    "fit_ok": True,
                    "fit_reason": "chain_failed",
                    "n_sessions_masked": int(len(masks)),
                    "n_removed_total": int(total_removed),
                    "removed_fraction_total": (
                        float(total_removed / total_base_points) if total_base_points > 0 else math.nan
                    ),
                }
            )
            continue

        _snapshot_outputs(base_public=base_public, scenario=scenario)
        chain = _read_chain_metrics(base_public=base_public)
        row = {
            "scenario": scenario,
            "kind": str(cand.get("kind") or ""),
            "mask_spec": json.dumps(masks, ensure_ascii=False),
            "allsky_summary_csv": str(allsky_path.resolve()),
            "fit_ok": True,
            "fit_reason": "",
            "n_sessions_masked": int(len(masks)),
            "n_removed_total": int(total_removed),
            "removed_fraction_total": (
                float(total_removed / total_base_points) if total_base_points > 0 else math.nan
            ),
            "stable_chi2_dof": float(chain["stable_chi2_dof"]),
            "stable_status": str(chain["stable_status"]),
            "timeband_chi2_dof": float(chain["timeband_chi2_dof"]),
            "timeband_status": str(chain["timeband_status"]),
            "threshold_chi2_dof": float(chain["threshold_chi2_dof"]),
            "threshold_status": str(chain["threshold_status"]),
            "threshold_ns": float(chain["threshold_ns"]),
            "session_fit_rows": session_fit_rows,
        }
        candidate_rows.append(row)
        if scenario == "keep_current":
            keep_ref = row

    if keep_ref is None:
        raise RuntimeError("keep_current scenario failed; cannot evaluate two-stage gate.")

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
        stage1_pass = bool(math.isfinite(allsky_rel) and (allsky_rel >= float(args.min_allsky_relative_improvement)))
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
        and str(r.get("scenario") or "") != "keep_current"
    ]
    if pass_rows:
        pass_rows = sorted(
            pass_rows,
            key=lambda r: (
                float(r.get("removed_fraction_total", math.inf)),
                -float(r.get("allsky_relative_improvement", -math.inf)),
                float(r.get("stable_delta_chi2_dof", math.inf)) + float(r.get("timeband_delta_chi2_dof", math.inf)),
            ),
        )
        selected = pass_rows[0]
        selected_reason = "selected_minimum_intervention_candidate_passing_two_stage_gate"
    else:
        selected = keep_ref
        selected_reason = "no_multi_session_candidate_passed_two_stage_gate_keep_current"

    selected_scenario = str(selected.get("scenario") or "keep_current")
    selected_policy = "keep_current" if selected_scenario == "keep_current" else f"multi_local_mask:{selected_scenario}"
    selected_allsky_csv = str(selected.get("allsky_summary_csv") or str(args.base_allsky_summary.resolve()))
    selected_threshold_ns = _to_float(selected.get("threshold_ns"), default=10.0)

    summary_csv = out_dir / "vlbi_beta_multisession_local_mask_two_stage_sweep_summary.csv"
    metrics_json = out_dir / "vlbi_beta_multisession_local_mask_two_stage_sweep_metrics.json"
    plot_pdf = out_dir / "vlbi_beta_multisession_local_mask_two_stage_sweep.pdf"
    plot_png = out_dir / "vlbi_beta_multisession_local_mask_two_stage_sweep.png"
    cols_out = [
        "scenario",
        "kind",
        "mask_spec",
        "allsky_summary_csv",
        "fit_ok",
        "fit_reason",
        "n_sessions_masked",
        "n_removed_total",
        "removed_fraction_total",
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
        "step": "8.7.46.28",
        "baseline": {
            "base_allsky_summary_csv": str(args.base_allsky_summary.resolve()),
            "keep_current_scenario": "keep_current",
        },
        "candidate_generation": {
            "min_sensitivity_ns": float(args.min_sensitivity_ns),
            "pair_size": int(args.pair_size),
            "session_to_source": session_to_source,
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

        fig, ax = plt.subplots(figsize=(11.0, 5.2))
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
            ax.set_xticklabels(x_labels, rotation=35, ha="right")
            ax.set_ylabel("all-sky relative improvement")
            ax.set_title("Multi-session local mask candidates (stage-1 metric)")
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

    if args.apply_selected:
        cmd_apply = [
            sys.executable,
            "-B",
            str((root / "scripts" / "vlbi" / "vlbi_beta_watchpack_apply_chain.py").resolve()),
            "--gate-metrics",
            str(metrics_json.resolve()),
        ]
        res_apply = _run(cmd_apply, cwd=root)
        print("Apply selected:", res_apply.get("returncode"))
        if not bool(res_apply.get("ok")):
            print(res_apply.get("stdout_tail"))
            print(res_apply.get("stderr_tail"))

    return 0


# Branch: Execute CLI entrypoint when invoked directly.

if __name__ == "__main__":
    raise SystemExit(main())
