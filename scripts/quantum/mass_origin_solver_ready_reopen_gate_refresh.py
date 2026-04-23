#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_solver_ready_reopen_gate_refresh.py

Step 8.7.55.2.82:
Refresh the solver-ready row and mass-origin reopen gate after the new
single-shape closure and positive particle-sector promotion outputs.

Inputs:
  - output/public/quantum/mass_origin_single_public_vpp_shape_closure_metrics.json
  - output/public/quantum/mass_origin_positive_particle_sector_chi_to_vpp_metrics.json
  - output/public/quantum/mass_origin_same_sector_vpp_shape_gate_metrics.json
  - output/public/quantum/mass_origin_blocked_state_reopen_metrics.json

Outputs:
  - output/public/quantum/mass_origin_solver_ready_reopen_gate_refresh_metrics.json
  - output/public/quantum/mass_origin_solver_ready_reopen_gate_refresh_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

CLOSURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_single_public_vpp_shape_closure_metrics.json"
PROMOTION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_positive_particle_sector_chi_to_vpp_metrics.json"
GATE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_vpp_shape_gate_metrics.json"
REOPEN_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_blocked_state_reopen_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_solver_ready_reopen_gate_refresh_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_solver_ready_reopen_gate_refresh_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.82"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh solver-ready and reopen-gate state for the mass-origin branch.",
    )
    parser.add_argument(
        "--step-tag",
        default=DEFAULT_STEP_TAG,
        help="Roadmap step tag to stamp into the output payload.",
    )
    return parser.parse_args()


# 関数: `_require_path` の入出力契約と処理意図を定義する。

def _require_path(path: Path) -> None:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: `_relative_str` の入出力契約と処理意図を定義する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (CLOSURE_JSON, PROMOTION_JSON, GATE_JSON, REOPEN_JSON):
        _require_path(path)

    closure = _read_json(CLOSURE_JSON)
    promotion = _read_json(PROMOTION_JSON)
    gate = _read_json(GATE_JSON)
    reopen = _read_json(REOPEN_JSON)

    closure_summary = closure.get("summary", {})
    promotion_summary = promotion.get("summary", {})
    gate_summary = gate.get("summary", {})
    reopen_decision = reopen.get("decision", {})

    positive_particle_sector_public_artifact_available = bool(
        promotion_summary.get("positive_particle_sector_chi_p_to_vpp_public_artifact_available", False)
    )
    single_public_boundary_family_fixed = bool(gate_summary.get("single_public_boundary_family_fixed", False))
    single_public_vpp_shape_available = bool(closure_summary.get("single_public_vpp_shape_available", False))
    solver_ready_row_promoted_to_pass = (
        positive_particle_sector_public_artifact_available
        and single_public_boundary_family_fixed
        and single_public_vpp_shape_available
    )
    mass_origin_branch_reopen_ready = solver_ready_row_promoted_to_pass
    proceed_to_no_free_parameter_mass_solver = mass_origin_branch_reopen_ready
    proceed_to_dark_matter_branch = False

    next_required_artifacts: List[str] = []

    # 条件分岐: `not positive_particle_sector_public_artifact_available` を満たす経路を評価する。
    if not positive_particle_sector_public_artifact_available:
        next_required_artifacts.append("positive_particle_sector_chi_p_to_vpp_public_artifact")

    # 条件分岐: `not single_public_vpp_shape_available` を満たす経路を評価する。

    if not single_public_vpp_shape_available:
        next_required_artifacts.append("single_public_vpp_shape")

    # 条件分岐: `not solver_ready_row_promoted_to_pass` を満たす経路を評価する。

    if not solver_ready_row_promoted_to_pass:
        next_required_artifacts.append("solver_ready_row_promoted_to_pass")

    rows = [
        {
            "row_id": "solver_ready_reopen_refresh_complete",
            "status": "pass",
            "metric": "solver-ready / reopen refresh complete",
            "value": 1.0,
            "note": "The refresh step re-evaluates solver-ready and reopen readiness after the positive same-sector promotion and single-shape closure outputs.",
        },
        {
            "row_id": "solver_ready_positive_same_sector_public_artifact",
            "status": "pass" if positive_particle_sector_public_artifact_available else "reject",
            "metric": "positive particle-sector chi_P -> V''(|P|_*) public artifact available",
            "value": 1.0 if positive_particle_sector_public_artifact_available else 0.0,
            "note": f"Promotion status is {promotion_summary.get('promotion_status')}.",
        },
        {
            "row_id": "solver_ready_single_public_boundary_family",
            "status": "pass" if single_public_boundary_family_fixed else "reject",
            "metric": "single public boundary family fixed",
            "value": 1.0 if single_public_boundary_family_fixed else 0.0,
            "note": "Shell quantization remains the sole public boundary family.",
        },
        {
            "row_id": "solver_ready_single_public_vpp_shape",
            "status": "pass" if single_public_vpp_shape_available else "reject",
            "metric": "single public V(|P|) shape available",
            "value": 1.0 if single_public_vpp_shape_available else 0.0,
            "note": (
                f"Single-shape closure selected {closure_summary.get('selected_candidate_id_or_none')}."
                if single_public_vpp_shape_available
                else f"Closure remains non-closing: {closure_summary.get('nonclosure_reason_or_none')}."
            ),
        },
        {
            "row_id": "solver_ready_row_promoted_to_pass",
            "status": "pass" if solver_ready_row_promoted_to_pass else "reject",
            "metric": "solver-ready row promoted to pass",
            "value": 1.0 if solver_ready_row_promoted_to_pass else 0.0,
            "note": (
                "All reopen prerequisites are satisfied."
                if solver_ready_row_promoted_to_pass
                else f"Solver-ready remains blocked by {next_required_artifacts}."
            ),
        },
        {
            "row_id": "mass_origin_branch_reopen_ready",
            "status": "pass" if mass_origin_branch_reopen_ready else "reject",
            "metric": "mass-origin branch reopen ready",
            "value": 1.0 if mass_origin_branch_reopen_ready else 0.0,
            "note": (
                "The branch can advance to the no-free-parameter mass-spectrum pilot."
                if mass_origin_branch_reopen_ready
                else "The branch remains blocked because the solver-ready row did not promote."
            ),
        },
        {
            "row_id": "proceed_to_no_free_parameter_mass_solver",
            "status": "pass" if proceed_to_no_free_parameter_mass_solver else "reject",
            "metric": "allowed to start the mass-spectrum pilot",
            "value": 1.0 if proceed_to_no_free_parameter_mass_solver else 0.0,
            "note": "This row follows the reopen gate directly and controls whether 8.7.55.2.83 can run.",
        },
        {
            "row_id": "proceed_to_dark_matter_branch",
            "status": "blocked",
            "metric": "allowed to start 8.7.55.3",
            "value": 0.0,
            "note": (
                "Even when reopen becomes true, 8.7.55.3 waits until the mass-origin branch closes through the spectrum / ratio steps."
                if mass_origin_branch_reopen_ready
                else "8.7.55.3 stays blocked because the mass-origin branch is not yet reopen ready."
            ),
        },
        {
            "row_id": "next_required_artifact_count",
            "status": "inventory",
            "metric": "next required artifacts count",
            "value": float(len(next_required_artifacts)),
            "note": f"Next required artifacts: {', '.join(next_required_artifacts)}." if next_required_artifacts else "No required artifacts remain before the spectrum pilot.",
        },
    ]

    overall_status = (
        "solver_ready_reopen_gate_refreshed_reopen_ready"
        if mass_origin_branch_reopen_ready
        else "solver_ready_reopen_gate_refreshed_still_blocked"
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "solver-ready row / reopen gate refresh",
        },
        "inputs": {
            "mass_origin_single_public_vpp_shape_closure_json": _relative_str(CLOSURE_JSON),
            "mass_origin_positive_particle_sector_chi_to_vpp_json": _relative_str(PROMOTION_JSON),
            "mass_origin_same_sector_vpp_shape_gate_json": _relative_str(GATE_JSON),
            "mass_origin_blocked_state_reopen_json": _relative_str(REOPEN_JSON),
        },
        "intent": "Refresh solver-ready and reopen readiness using the new same-sector closure and promotion artifacts.",
        "formulas": {
            "solver_ready_rule": "solver_ready_row_promoted_to_pass iff positive same-sector public artifact + single public boundary family + single public V(|P|) shape are all true",
            "reopen_rule": "mass_origin_branch_reopen_ready follows solver_ready_row_promoted_to_pass",
            "dark_matter_rule": "8.7.55.3 remains false here because the mass-origin branch still needs the spectrum / ratio closure after reopen",
        },
        "rows": rows,
        "summary": {
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": positive_particle_sector_public_artifact_available,
            "single_public_boundary_family_fixed": single_public_boundary_family_fixed,
            "single_public_vpp_shape_available": single_public_vpp_shape_available,
            "solver_ready_row_promoted_to_pass": solver_ready_row_promoted_to_pass,
            "mass_origin_branch_reopen_ready": mass_origin_branch_reopen_ready,
            "proceed_to_no_free_parameter_mass_solver": proceed_to_no_free_parameter_mass_solver,
            "proceed_to_dark_matter_branch": proceed_to_dark_matter_branch,
            "next_required_artifacts": next_required_artifacts,
        },
        "decision": {
            "overall_status": overall_status,
            "blocked_state_detail": str(reopen_decision.get("blocked_state_detail", "")),
            "mass_origin_branch_blocked": not mass_origin_branch_reopen_ready,
            "mass_origin_branch_reopen_ready": mass_origin_branch_reopen_ready,
            "solver_ready_row_promoted_to_pass": solver_ready_row_promoted_to_pass,
            "proceed_to_no_free_parameter_mass_solver": proceed_to_no_free_parameter_mass_solver,
            "proceed_to_dark_matter_branch": proceed_to_dark_matter_branch,
            "next_required_artifacts": next_required_artifacts,
        },
        "evidence": {
            "closure_summary": closure_summary,
            "promotion_summary": promotion_summary,
            "gate_summary": gate_summary,
            "prior_reopen_decision": reopen_decision,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    args = _parse_args()
    payload = _build_payload(args.step_tag)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["rows"])
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
