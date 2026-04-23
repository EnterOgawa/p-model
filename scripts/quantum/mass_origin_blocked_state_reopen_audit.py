#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_blocked_state_reopen_audit.py

Step 8.7.55.2.4:
Freeze the blocked-state and reopen criteria for the mass-origin branch, using
only already-frozen public canonical evidence from Steps 8.7.55.2.1-.3.

Inputs:
  - output/public/quantum/mass_origin_readiness_gate_metrics.json
  - output/public/quantum/mass_origin_curvature_boundary_metrics.json
  - output/public/quantum/mass_origin_solver_spec_gate_metrics.json
  - output/public/quantum/mass_origin_solver_family_elimination_metrics.json
  - output/public/quantum/mass_origin_same_sector_vpp_shape_gate_metrics.json
  - output/public/quantum/mass_origin_latent_reopen_route_inventory_metrics.json

Outputs:
  - output/public/quantum/mass_origin_blocked_state_reopen_metrics.json
  - output/public/quantum/mass_origin_blocked_state_reopen_rows.csv

Assumptions:
  - Reopen requires a positive particle-sector chi_P -> V''(|P|_*) public
    artifact, a unique public boundary family, and a unique public V(|P|)
    shape, all fixed without new free parameters.
  - Until those conditions are met, the mass-origin branch remains blocked and
    the dark-matter branch cannot start.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

READINESS_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_readiness_gate_metrics.json"
CURVATURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_curvature_boundary_metrics.json"
SOLVER_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_solver_spec_gate_metrics.json"
ELIMINATION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_solver_family_elimination_metrics.json"
SPECIFIC_GATE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_vpp_shape_gate_metrics.json"
LATENT_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_latent_reopen_route_inventory_metrics.json"
OUT_DIR = ROOT / "output" / "public" / "quantum"
OUT_JSON = OUT_DIR / "mass_origin_blocked_state_reopen_metrics.json"
OUT_CSV = OUT_DIR / "mass_origin_blocked_state_reopen_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.4"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync the blocked-state and reopen criteria for the mass-origin branch.",
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


# 関数: `_find_row_by_id` の入出力契約と処理意図を定義する。

def _find_row_by_id(rows: List[Dict[str, Any]], row_id: str) -> Dict[str, Any]:
    for row in rows:
        # 条件分岐: `str(row.get("row_id")) == row_id` を満たす経路を評価する。
        if str(row.get("row_id")) == row_id:
            return row

    raise KeyError(f"missing row_id: {row_id}")


# 関数: `_positive_status` の入出力契約と処理意図を定義する。

def _positive_status(status: str) -> bool:
    return status not in {
        "missing",
        "reject",
        "watch",
        "entry_only",
        "blocked",
        "candidate_doc_only",
        "candidate_script_only",
        "candidate_public",
        "candidate_public_interface",
        "inventory",
        "fixed_target",
        "interface_fixed",
    }


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows(
    readiness: Dict[str, Any],
    curvature: Dict[str, Any],
    solver: Dict[str, Any],
    elimination: Dict[str, Any],
    specific_gate: Dict[str, Any],
    latent_inventory: Dict[str, Any],
) -> List[Dict[str, Any]]:
    readiness_rows = readiness.get("rows", [])
    curvature_rows = curvature.get("rows", [])
    solver_rows = solver.get("rows", [])
    elimination_rows = elimination.get("rows", [])
    specific_rows = specific_gate.get("rows", [])
    latent_rows = latent_inventory.get("rows", [])

    # 条件分岐: `not isinstance(readiness_rows, list)` を満たす経路を評価する。
    if not isinstance(readiness_rows, list):
        raise SystemExit(f"[fail] invalid rows in {READINESS_JSON}")

    # 条件分岐: `not isinstance(curvature_rows, list)` を満たす経路を評価する。

    if not isinstance(curvature_rows, list):
        raise SystemExit(f"[fail] invalid rows in {CURVATURE_JSON}")

    # 条件分岐: `not isinstance(solver_rows, list)` を満たす経路を評価する。

    if not isinstance(solver_rows, list):
        raise SystemExit(f"[fail] invalid rows in {SOLVER_JSON}")

    if not isinstance(elimination_rows, list):
        raise SystemExit(f"[fail] invalid rows in {ELIMINATION_JSON}")

    if not isinstance(specific_rows, list):
        raise SystemExit(f"[fail] invalid rows in {SPECIFIC_GATE_JSON}")

    if not isinstance(latent_rows, list):
        raise SystemExit(f"[fail] invalid rows in {LATENT_INVENTORY_JSON}")

    readiness_same_sector = _find_row_by_id(readiness_rows, "same_sector_chi_p_to_vpp_mapping")
    curvature_same_sector = _find_row_by_id(curvature_rows, "same_sector_curvature_mapping_particle_sector")
    unique_potential_row = _find_row_by_id(curvature_rows, "single_vpp_shape_unique")
    solver_ready_row = _find_row_by_id(curvature_rows, "no_free_parameter_mass_solver_spec_ready")
    reopen_row = _find_row_by_id(solver_rows, "mass_origin_branch_reopen_requires_new_public_artifact")
    positive_mapping_row = _find_row_by_id(solver_rows, "positive_same_sector_mapping_public_artifact_count")
    elimination_boundary_row = _find_row_by_id(elimination_rows, "single_public_boundary_family_remaining")
    specific_same_sector_row = _find_row_by_id(specific_rows, "same_sector_public_artifact_still_missing")
    specific_vpp_row = _find_row_by_id(specific_rows, "single_public_vpp_shape_still_missing")
    specific_fixed_state_row = _find_row_by_id(specific_rows, "specific_missing_artifacts_fixed_state")
    specific_solver_dependency_row = _find_row_by_id(specific_rows, "solver_ready_row_still_depends_on_two_named_artifacts")
    latent_same_sector_row = _find_row_by_id(latent_rows, "latent_positive_same_sector_public_rows")
    latent_vpp_row = _find_row_by_id(latent_rows, "effective_potential_nonphenomenological_public_count")
    latent_exhausted_row = _find_row_by_id(latent_rows, "latent_reopen_route_inventory_exhausted")

    same_sector_ready = _positive_status(str(positive_mapping_row.get("status", "")))
    unique_boundary_ready = _positive_status(str(elimination_boundary_row.get("status", "")))
    unique_potential_ready = _positive_status(str(unique_potential_row.get("status", "")))
    solver_ready = _positive_status(str(solver_ready_row.get("status", "")))
    reopen_ready = same_sector_ready and unique_boundary_ready and unique_potential_ready and solver_ready
    specific_blocked_detail_fixed = _positive_status(str(specific_fixed_state_row.get("status", "")))
    blocked_note = (
        "8.7.55.2 remains blocked because the remaining blockers are now frozen to "
        "`positive_particle_sector_chi_p_to_vpp_public_artifact` and "
        "`single_public_vpp_shape`, plus the dependent solver-ready row."
    )

    if not specific_blocked_detail_fixed:
        blocked_note = (
            "8.7.55.2 remains blocked because public canonical evidence still lacks "
            "a positive same-sector curvature map and a unique V(|P|) shape, even "
            "after family elimination reduced the public boundary family to one."
        )

    if str(latent_exhausted_row.get("status", "")) == "pass":
        blocked_note = (
            "8.7.55.2 remains blocked because the remaining blockers are frozen to "
            "`positive_particle_sector_chi_p_to_vpp_public_artifact` and `single_public_vpp_shape`, "
            "and the repo-wide latent-route inventory is also exhausted."
        )

    return [
        {
            "row_id": "mass_origin_branch_blocked_state",
            "status": "blocked",
            "metric": "mass-origin branch currently blocked",
            "value": 1.0,
            "note": blocked_note,
        },
        {
            "row_id": "reopen_requires_positive_same_sector_public_artifact",
            "status": "pass" if same_sector_ready else "reject",
            "metric": "positive particle-sector chi_P -> V''(|P|_*) public artifact",
            "value": 1.0 if same_sector_ready else 0.0,
            "note": str(specific_same_sector_row.get("note", "")),
        },
        {
            "row_id": "reopen_requires_single_boundary_family",
            "status": "pass" if unique_boundary_ready else "reject",
            "metric": "single public boundary / quantization family fixed",
            "value": 1.0 if unique_boundary_ready else 0.0,
            "note": str(elimination_boundary_row.get("note", "")),
        },
        {
            "row_id": "reopen_requires_single_vpp_shape",
            "status": "pass" if unique_potential_ready else "reject",
            "metric": "single public V(|P|) shape fixed",
            "value": 1.0 if unique_potential_ready else 0.0,
            "note": str(specific_vpp_row.get("note", "")),
        },
        {
            "row_id": "reopen_requires_solver_ready_row",
            "status": "pass" if solver_ready else "reject",
            "metric": "no-free-parameter solver spec ready",
            "value": 1.0 if solver_ready else 0.0,
            "note": str(specific_solver_dependency_row.get("note", "")),
        },
        {
            "row_id": "reopen_gate_all_conditions",
            "status": "pass" if reopen_ready else "blocked",
            "metric": "all reopen conditions satisfied simultaneously",
            "value": 1.0 if reopen_ready else 0.0,
            "note": "Reopen requires all four conditions at once: positive same-sector mapping, unique boundary family, unique V(|P|) shape, and solver-ready row.",
        },
        {
            "row_id": "dark_matter_branch_still_blocked_by_mass_origin",
            "status": "blocked" if not reopen_ready else "pass",
            "metric": "8.7.55.3 allowed to start",
            "value": 0.0 if not reopen_ready else 1.0,
            "note": "Do not start 8.7.55.3 until the mass-origin branch reopens and closes, because its prerequisites remain unsatisfied.",
        },
        {
            "row_id": "blocked_state_reopen_wording_synced",
            "status": "pass",
            "metric": "blocked-state and reopen wording synchronized",
            "value": 1.0,
            "note": str(reopen_row.get("note", "")),
        },
        {
            "row_id": "specific_missing_artifacts_fixed_state",
            "status": str(specific_fixed_state_row.get("status", "unknown")),
            "metric": str(specific_fixed_state_row.get("metric", "")),
            "value": float(specific_fixed_state_row.get("value", 0.0)),
            "note": str(specific_fixed_state_row.get("note", "")),
        },
        {
            "row_id": "latent_same_sector_public_candidate_count",
            "status": str(latent_same_sector_row.get("status", "unknown")),
            "metric": str(latent_same_sector_row.get("metric", "")),
            "value": float(latent_same_sector_row.get("value", 0.0)),
            "note": str(latent_same_sector_row.get("note", "")),
        },
        {
            "row_id": "latent_nonphenomenological_vpp_public_candidate_count",
            "status": str(latent_vpp_row.get("status", "unknown")),
            "metric": str(latent_vpp_row.get("metric", "")),
            "value": float(latent_vpp_row.get("value", 0.0)),
            "note": str(latent_vpp_row.get("note", "")),
        },
        {
            "row_id": "latent_reopen_route_inventory_exhausted",
            "status": str(latent_exhausted_row.get("status", "unknown")),
            "metric": str(latent_exhausted_row.get("metric", "")),
            "value": float(latent_exhausted_row.get("value", 0.0)),
            "note": str(latent_exhausted_row.get("note", "")),
        },
        {
            "row_id": "readiness_same_sector_row_still_missing",
            "status": str(readiness_same_sector.get("status", "unknown")),
            "metric": str(readiness_same_sector.get("metric", "")),
            "value": float(readiness_same_sector.get("value", 0.0)),
            "note": str(readiness_same_sector.get("note", "")),
        },
        {
            "row_id": "curvature_same_sector_row_still_missing",
            "status": str(curvature_same_sector.get("status", "unknown")),
            "metric": str(curvature_same_sector.get("metric", "")),
            "value": float(curvature_same_sector.get("value", 0.0)),
            "note": str(curvature_same_sector.get("note", "")),
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (READINESS_JSON, CURVATURE_JSON, SOLVER_JSON, ELIMINATION_JSON, SPECIFIC_GATE_JSON, LATENT_INVENTORY_JSON):
        _require_path(path)

    readiness = _read_json(READINESS_JSON)
    curvature = _read_json(CURVATURE_JSON)
    solver = _read_json(SOLVER_JSON)
    elimination = _read_json(ELIMINATION_JSON)
    specific_gate = _read_json(SPECIFIC_GATE_JSON)
    latent_inventory = _read_json(LATENT_INVENTORY_JSON)
    rows = _build_rows(readiness, curvature, solver, elimination, specific_gate, latent_inventory)

    reopen_ready = bool(_find_row_by_id(rows, "reopen_gate_all_conditions").get("value"))
    blocked = not reopen_ready
    unsatisfied_requirements = [
        row["row_id"]
        for row in rows
        if row["row_id"].startswith("reopen_requires_") and float(row["value"]) <= 0.0
    ]
    next_required_artifacts_map = {
        "reopen_requires_positive_same_sector_public_artifact": "positive_particle_sector_chi_p_to_vpp_public_artifact",
        "reopen_requires_single_boundary_family": "single_public_boundary_family",
        "reopen_requires_single_vpp_shape": "single_public_vpp_shape",
        "reopen_requires_solver_ready_row": "solver_ready_row_promoted_to_pass",
    }
    next_required_artifacts = [next_required_artifacts_map[row_id] for row_id in unsatisfied_requirements]
    specific_gate_decision = specific_gate.get("decision", {})
    latent_inventory_decision = latent_inventory.get("decision", {})
    specific_blocked_detail = str(specific_gate_decision.get("blocked_state_detail", "generic_blocked_state"))
    overall_status = str(specific_gate_decision.get("overall_status", "blocked_state_frozen_waiting_for_new_public_artifact"))

    if reopen_ready:
        overall_status = "reopen_ready"
        specific_blocked_detail = "none"

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "mass-origin blocked-state and reopen criteria sync",
        },
        "inputs": {
            "mass_origin_readiness_gate_json": _relative_str(READINESS_JSON),
            "mass_origin_curvature_boundary_json": _relative_str(CURVATURE_JSON),
            "mass_origin_solver_spec_gate_json": _relative_str(SOLVER_JSON),
            "mass_origin_solver_family_elimination_json": _relative_str(ELIMINATION_JSON),
            "mass_origin_same_sector_vpp_shape_gate_json": _relative_str(SPECIFIC_GATE_JSON),
            "mass_origin_latent_reopen_route_inventory_json": _relative_str(LATENT_INVENTORY_JSON),
        },
        "intent": "Freeze the mass-origin branch as blocked in public canonical form, and state exactly which new public artifacts are required before the branch can reopen.",
        "formulas": {
            "reopen_gate": "reopen only if same-sector chi_P -> V''(|P|_*) public artifact + unique public boundary family + unique public V(|P|) shape + solver-ready row are all fixed",
            "blocked_policy": "while reopen gate is false, keep 8.7.55.2 blocked and do not proceed to 8.7.55.3",
        },
        "rows": rows,
        "summary": {
            "blocked_state_fixed": blocked,
            "reopen_ready": reopen_ready,
            "blocked_state_detail": specific_blocked_detail,
            "latent_reopen_routes_exhausted": bool(_find_row_by_id(rows, "latent_reopen_route_inventory_exhausted").get("value")),
            "reopen_requirement_count": 4,
            "reopen_requirement_satisfied_count": 4 - len(unsatisfied_requirements),
            "reopen_requirement_unsatisfied_count": len(unsatisfied_requirements),
            "unsatisfied_requirements": unsatisfied_requirements,
        },
        "decision": {
            "overall_status": overall_status,
            "blocked_state_detail": specific_blocked_detail,
            "mass_origin_branch_blocked": blocked,
            "mass_origin_branch_reopen_ready": reopen_ready,
            "proceed_to_no_free_parameter_mass_solver": reopen_ready,
            "proceed_to_dark_matter_branch": False,
            "next_required_artifacts": next_required_artifacts,
        },
        "evidence": {
            "readiness_decision": readiness.get("decision", {}),
            "curvature_decision": curvature.get("decision", {}),
            "solver_decision": solver.get("decision", {}),
            "elimination_decision": elimination.get("decision", {}),
            "specific_gate_decision": specific_gate_decision,
            "latent_inventory_decision": latent_inventory_decision,
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
    payload = _build_payload(str(args.step_tag))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["rows"])
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
