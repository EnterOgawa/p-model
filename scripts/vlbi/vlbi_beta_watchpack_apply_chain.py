#!/usr/bin/env python3
"""
vlbi_beta_watchpack_apply_chain.py

Apply two-stage watchpack policy to the VLBI consistency chain.

This script:
1) Reads selected scenario from two-stage gate metrics.
2) Runs source-session -> stable-source -> timeband -> threshold in order.
3) Freezes selected scenario outputs into canonical (no-suffix) filenames.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence


# Function: Resolve repository root.
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Function: Execute one command and return structured result.

def _run(cmd: List[str], cwd: Path) -> Dict[str, object]:
    cp = subprocess.run(cmd, cwd=str(cwd), check=False, capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": int(cp.returncode),
        "ok": bool(cp.returncode == 0),
        "stdout_tail": "\n".join((cp.stdout or "").splitlines()[-10:]),
        "stderr_tail": "\n".join((cp.stderr or "").splitlines()[-10:]),
    }


# Function: Copy files to canonical names in both output/vlbi and output/public/vlbi.

def _copy_to_canonical(base_public: Path, base_private: Path, pairs: Sequence[tuple[str, str]]) -> None:
    for src_name, dst_name in pairs:
        src_public = base_public / src_name
        if not src_public.exists():
            raise FileNotFoundError(f"source artifact missing: {src_public}")

        dst_public = base_public / dst_name
        if src_public.resolve() != dst_public.resolve():
            shutil.copy2(src_public, dst_public)

        shutil.copy2(src_public, base_private / dst_name)


# Function: Main entrypoint for chain application.

def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Apply watchpack two-stage gate decision to VLBI analysis chain.")
    ap.add_argument(
        "--gate-metrics",
        type=Path,
        default=root / "output" / "public" / "vlbi" / "vlbi_beta_watchpack_two_stage_gate_metrics.json",
        help="Two-stage gate metrics JSON.",
    )
    ap.add_argument(
        "--source-summary",
        type=Path,
        default=root / "output" / "public" / "vlbi" / "vlbi_beta_source_session_matrix_source_summary.csv",
        help="Source-session matrix summary CSV.",
    )
    ap.add_argument(
        "--session-root",
        type=Path,
        default=root / "data" / "vlbi" / "sources" / "vgosdb",
        help="Per-session extracted vgosDb root.",
    )
    ap.add_argument(
        "--min-sensitivity-ns",
        type=float,
        default=10.0,
        help="Minimum max|Cal-BendSun| [ns] for high-sensitivity subset.",
    )
    ap.add_argument(
        "--min-source-sessions",
        type=int,
        default=3,
        help="Stable-source gate: minimum sessions per source.",
    )
    ap.add_argument(
        "--max-source-chi2-dof",
        type=float,
        default=2.0,
        help="Stable-source gate: maximum source chi2/dof.",
    )
    ap.add_argument(
        "--require-source-status",
        type=str,
        default="pass",
        choices=["", "pass", "watch", "reject"],
        help="Stable-source gate: required source status.",
    )
    ap.add_argument(
        "--min-source-points-per-session",
        type=int,
        default=20,
        help="Stable-source gate: minimum source points per session.",
    )
    ap.add_argument(
        "--min-quartile-points",
        type=int,
        default=8,
        help="Timeband gate: minimum quartile points.",
    )
    ap.add_argument(
        "--min-quartile-sigma",
        type=float,
        default=0.01,
        help="Timeband gate: quartile sigma floor.",
    )
    ap.add_argument(
        "--max-session-pairwise-z",
        type=float,
        default=20.0,
        help="Timeband gate: max pairwise z per session.",
    )
    ap.add_argument(
        "--min-valid-quartiles-per-session",
        type=int,
        default=2,
        help="Timeband gate: minimum valid quartiles after quality gate.",
    )
    ap.add_argument(
        "--nuisance-mode",
        type=str,
        default="baseline_intercept_linear",
        choices=["none", "baseline_intercept", "baseline_intercept_linear"],
        help="Nuisance mode passed to chain scripts.",
    )
    ap.add_argument(
        "--observable-series",
        type=str,
        default="full",
        choices=["full", "fringe"],
        help="Observable series passed to chain scripts.",
    )
    args = ap.parse_args()

    gate_payload = json.loads(args.gate_metrics.resolve().read_text(encoding="utf-8"))
    decision = dict(gate_payload.get("decision") or {})
    selected_scenario = str(decision.get("selected_scenario") or "").strip()
    selected_allsky_csv = str(decision.get("selected_allsky_summary_csv") or "").strip()
    selected_threshold_ns = float(decision.get("selected_threshold_ns") or args.min_sensitivity_ns)
    selected_policy = str(decision.get("selected_policy") or "").strip()
    if not selected_scenario or not selected_allsky_csv:
        raise RuntimeError("invalid gate decision payload: selected scenario/allsky summary missing.")

    allsky_summary = Path(selected_allsky_csv).resolve()
    if not allsky_summary.exists():
        raise FileNotFoundError(f"selected all-sky summary not found: {allsky_summary}")

    py = sys.executable
    commands = [
        [
            py,
            "-B",
            str((root / "scripts" / "vlbi" / "vlbi_beta_source_session_matrix.py").resolve()),
            "--allsky-summary",
            str(allsky_summary),
            "--session-root",
            str(args.session_root.resolve()),
            "--min-sensitivity-ns",
            str(args.min_sensitivity_ns),
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
            str(allsky_summary),
            "--source-summary",
            str(args.source_summary.resolve()),
            "--session-root",
            str(args.session_root.resolve()),
            "--min-sensitivity-ns",
            str(args.min_sensitivity_ns),
            "--min-source-sessions",
            str(args.min_source_sessions),
            "--max-source-chi2-dof",
            str(args.max_source_chi2_dof),
            "--require-source-status",
            str(args.require_source_status),
            "--min-source-points-per-session",
            str(args.min_source_points_per_session),
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
            str(allsky_summary),
            "--source-summary",
            str(args.source_summary.resolve()),
            "--session-root",
            str(args.session_root.resolve()),
            "--min-sensitivity-ns",
            str(args.min_sensitivity_ns),
            "--min-source-sessions",
            str(args.min_source_sessions),
            "--max-source-chi2-dof",
            str(args.max_source_chi2_dof),
            "--require-source-status",
            str(args.require_source_status),
            "--min-source-points-per-session",
            str(args.min_source_points_per_session),
            "--min-quartile-points",
            str(args.min_quartile_points),
            "--min-quartile-sigma",
            str(args.min_quartile_sigma),
            "--max-session-pairwise-z",
            str(args.max_session_pairwise_z),
            "--min-valid-quartiles-per-session",
            str(args.min_valid_quartiles_per_session),
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
            str(allsky_summary),
            "--thresholds",
            "10,12,15,20",
            "--min-sessions-operational",
            "3",
        ],
    ]

    run_results: List[Dict[str, object]] = []
    for cmd in commands:
        res = _run(cmd, cwd=root)
        run_results.append(res)
        if not bool(res["ok"]):
            break

    if not all(bool(r["ok"]) for r in run_results):
        raise RuntimeError("chain execution failed; inspect apply-chain metrics stderr/stdout tails.")

    base_public = root / "output" / "public" / "vlbi"
    base_private = root / "output" / "vlbi"
    scenario_suffix = selected_scenario
    copy_pairs = [
        (f"vlbi_beta_source_session_matrix_details.csv", f"vlbi_beta_source_session_matrix_details.csv"),
        (f"vlbi_beta_source_session_matrix_source_summary.csv", f"vlbi_beta_source_session_matrix_source_summary.csv"),
        (f"vlbi_beta_source_session_matrix_metrics.json", f"vlbi_beta_source_session_matrix_metrics.json"),
        (f"vlbi_beta_source_session_matrix.pdf", f"vlbi_beta_source_session_matrix.pdf"),
        (f"vlbi_beta_source_session_matrix.png", f"vlbi_beta_source_session_matrix.png"),
        (
            f"vlbi_beta_stable_source_refit_summary_{scenario_suffix}.csv",
            "vlbi_beta_stable_source_refit_summary.csv",
        ),
        (
            f"vlbi_beta_stable_source_refit_source_presence_{scenario_suffix}.csv",
            "vlbi_beta_stable_source_refit_source_presence.csv",
        ),
        (
            f"vlbi_beta_stable_source_refit_metrics_{scenario_suffix}.json",
            "vlbi_beta_stable_source_refit_metrics.json",
        ),
        (
            f"vlbi_beta_stable_source_refit_{scenario_suffix}.pdf",
            "vlbi_beta_stable_source_refit.pdf",
        ),
        (
            f"vlbi_beta_stable_source_refit_{scenario_suffix}.png",
            "vlbi_beta_stable_source_refit.png",
        ),
        (
            f"vlbi_beta_timeband_stratified_refit_details_{scenario_suffix}.csv",
            "vlbi_beta_timeband_stratified_refit_details.csv",
        ),
        (
            f"vlbi_beta_timeband_stratified_refit_session_summary_{scenario_suffix}.csv",
            "vlbi_beta_timeband_stratified_refit_session_summary.csv",
        ),
        (
            f"vlbi_beta_timeband_stratified_refit_quartile_consistency_{scenario_suffix}.csv",
            "vlbi_beta_timeband_stratified_refit_quartile_consistency.csv",
        ),
        (
            f"vlbi_beta_timeband_stratified_refit_metrics_{scenario_suffix}.json",
            "vlbi_beta_timeband_stratified_refit_metrics.json",
        ),
        (
            f"vlbi_beta_timeband_stratified_refit_{scenario_suffix}.pdf",
            "vlbi_beta_timeband_stratified_refit.pdf",
        ),
        (
            f"vlbi_beta_timeband_stratified_refit_{scenario_suffix}.png",
            "vlbi_beta_timeband_stratified_refit.png",
        ),
        (
            f"vlbi_high_sensitivity_threshold_sweep_{scenario_suffix}.csv",
            "vlbi_high_sensitivity_threshold_sweep.csv",
        ),
        (
            f"vlbi_high_sensitivity_threshold_sweep_metrics_{scenario_suffix}.json",
            "vlbi_high_sensitivity_threshold_sweep_metrics.json",
        ),
        (
            f"vlbi_high_sensitivity_threshold_sweep_{scenario_suffix}.pdf",
            "vlbi_high_sensitivity_threshold_sweep.pdf",
        ),
        (
            f"vlbi_high_sensitivity_threshold_sweep_{scenario_suffix}.png",
            "vlbi_high_sensitivity_threshold_sweep.png",
        ),
    ]
    _copy_to_canonical(base_public=base_public, base_private=base_private, pairs=copy_pairs)

    canonical_allsky_public = base_public / "vlbi_allsky_beta_consistency_summary.csv"
    canonical_allsky_private = base_private / "vlbi_allsky_beta_consistency_summary.csv"
    shutil.copy2(allsky_summary, canonical_allsky_public)
    shutil.copy2(allsky_summary, canonical_allsky_private)

    out_dir = root / "output" / "vlbi"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "vlbi_beta_watchpack_apply_chain_metrics.json"
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "step": "8.7.46.24",
        "selected_scenario": selected_scenario,
        "selected_policy": selected_policy,
        "selected_allsky_summary_csv": str(allsky_summary),
        "selected_threshold_ns": float(selected_threshold_ns),
        "commands": run_results,
        "canonical_sync": {
            "allsky_summary_public": str(canonical_allsky_public.resolve()),
            "allsky_summary_private": str(canonical_allsky_private.resolve()),
            "copied_pairs_n": int(len(copy_pairs)),
        },
        "outputs": {
            "metrics_json": str(metrics_path.resolve()),
        },
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(metrics_path, (base_public / metrics_path.name))
    print(f"Wrote: {metrics_path}")
    print("Synced:", base_public / metrics_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
