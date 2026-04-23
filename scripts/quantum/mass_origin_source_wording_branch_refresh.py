#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_source_wording_branch_refresh.py

Step 8.7.55.2.108:
Re-inject the source-wording closure result into the blocked second-route
stack and refreeze whether handoff to 8.7.55.2.83-.84 is now allowed.

Inputs:
  - output/public/quantum/mass_origin_split_source_branch_refresh_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_semantic_bridge_contract_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_semantic_bridge_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_literal_fragment_inventory_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_literal_lift_metrics.json
  - output/public/quantum/mass_origin_source_wording_closure_metrics.json

Outputs:
  - output/public/quantum/mass_origin_source_wording_branch_refresh_metrics.json
  - output/public/quantum/mass_origin_source_wording_branch_refresh_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]

PRIOR_REFRESH_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_split_source_branch_refresh_metrics.json"
SHELL_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_semantic_bridge_contract_metrics.json"
SHELL_BRIDGE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_semantic_bridge_metrics.json"
LITERAL_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_literal_fragment_inventory_metrics.json"
LITERAL_LIFT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_literal_lift_metrics.json"
SOURCE_WORDING_CLOSURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_source_wording_closure_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_source_wording_branch_refresh_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_source_wording_branch_refresh_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.108"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh blocked second-route handoff eligibility after source-wording closure.",
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


# 関数: `_ordered_unique` の入出力契約と処理意図を定義する。

def _ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []

    for value in values:
        # 条件分岐: `value and value not in seen` を満たす経路を評価する。
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)

    return ordered


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (
        PRIOR_REFRESH_JSON,
        SHELL_CONTRACT_JSON,
        SHELL_BRIDGE_JSON,
        LITERAL_INVENTORY_JSON,
        LITERAL_LIFT_JSON,
        SOURCE_WORDING_CLOSURE_JSON,
    ):
        _require_path(path)

    prior_refresh = _read_json(PRIOR_REFRESH_JSON)
    shell_contract = _read_json(SHELL_CONTRACT_JSON)
    shell_bridge = _read_json(SHELL_BRIDGE_JSON)
    literal_inventory = _read_json(LITERAL_INVENTORY_JSON)
    literal_lift = _read_json(LITERAL_LIFT_JSON)
    source_wording_closure = _read_json(SOURCE_WORDING_CLOSURE_JSON)

    prior_refresh_summary = prior_refresh.get("summary", {})
    shell_contract_summary = shell_contract.get("summary", {})
    shell_bridge_summary = shell_bridge.get("summary", {})
    literal_inventory_summary = literal_inventory.get("summary", {})
    literal_lift_summary = literal_lift.get("summary", {})
    source_wording_summary = source_wording_closure.get("summary", {})

    same_sector_tiebreak_target_value_available = bool(
        source_wording_summary.get("same_sector_tiebreak_target_value_available", False)
    )
    target_source_kind_or_none = source_wording_summary.get("target_source_kind_or_none")
    target_value_bridge_without_new_free_parameters = bool(
        source_wording_summary.get("bridge_without_new_free_parameters", False)
    )
    single_public_boundary_family_fixed = bool(prior_refresh_summary.get("single_public_boundary_family_fixed", False))
    single_public_vpp_shape_available = bool(prior_refresh_summary.get("single_public_vpp_shape_available", False))
    selected_candidate_id_or_none = prior_refresh_summary.get("selected_candidate_id_or_none")
    positive_particle_sector_chi_p_to_vpp_public_artifact_available = bool(
        prior_refresh_summary.get("positive_particle_sector_chi_p_to_vpp_public_artifact_available", False)
    )
    solver_ready_row_promoted_to_pass = bool(
        single_public_boundary_family_fixed
        and same_sector_tiebreak_target_value_available
        and single_public_vpp_shape_available
        and positive_particle_sector_chi_p_to_vpp_public_artifact_available
    )
    mass_origin_branch_reopen_ready = solver_ready_row_promoted_to_pass
    hand_off_to_8_7_55_2_83 = bool(mass_origin_branch_reopen_ready)

    remaining_missing_artifacts: List[str] = []

    # 条件分岐: `not same_sector_tiebreak_target_value_available` を満たす経路を評価する。
    if not same_sector_tiebreak_target_value_available:
        remaining_missing_artifacts.append("same_sector_tiebreak_target_value")

    # 条件分岐: `not single_public_vpp_shape_available` を満たす経路を評価する。

    if not single_public_vpp_shape_available:
        remaining_missing_artifacts.append("single_public_vpp_shape")

    # 条件分岐: `not positive_particle_sector_chi_p_to_vpp_public_artifact_available` を満たす経路を評価する。

    if not positive_particle_sector_chi_p_to_vpp_public_artifact_available:
        remaining_missing_artifacts.append("positive_particle_sector_chi_p_to_vpp_public_artifact")

    # 条件分岐: `not solver_ready_row_promoted_to_pass` を満たす経路を評価する。

    if not solver_ready_row_promoted_to_pass:
        remaining_missing_artifacts.append("solver_ready_row_promoted_to_pass")

    remaining_missing_artifacts = _ordered_unique(remaining_missing_artifacts)
    remaining_source_level_blockers = _ordered_unique(
        [str(item) for item in source_wording_summary.get("remaining_source_level_blockers", [])]
    )

    rows = [
        {
            "row_id": "source_wording_branch_refresh_complete",
            "status": "pass",
            "metric": "source-wording branch refresh complete",
            "value": 1.0,
            "note": "This refresh reinjects the source-wording closure result into the blocked second-route stack and refreezes .83-.84 handoff eligibility.",
        },
        {
            "row_id": "source_wording_branch_same_sector_tiebreak_target_value",
            "status": "pass" if same_sector_tiebreak_target_value_available else "watch",
            "metric": "same-sector tie-break target value available after source-wording refresh",
            "value": 1.0 if same_sector_tiebreak_target_value_available else 0.0,
            "note": (
                f"Target source kind is {target_source_kind_or_none}."
                if same_sector_tiebreak_target_value_available
                else f"Source-wording refresh remains blocked by {remaining_source_level_blockers}."
            ),
        },
        {
            "row_id": "source_wording_branch_single_public_vpp_shape",
            "status": "pass" if single_public_vpp_shape_available else "reject",
            "metric": "single public V(|P|) shape available after source-wording refresh",
            "value": 1.0 if single_public_vpp_shape_available else 0.0,
            "note": (
                f"Selected candidate is {selected_candidate_id_or_none}."
                if single_public_vpp_shape_available
                else "Single public V(|P|) shape remains absent because no target-driven unique candidate has reopened the branch."
            ),
        },
        {
            "row_id": "source_wording_branch_positive_same_sector_public_artifact",
            "status": "pass" if positive_particle_sector_chi_p_to_vpp_public_artifact_available else "reject",
            "metric": "positive particle-sector chi_P -> V''(|P|_*) public artifact available after source-wording refresh",
            "value": 1.0 if positive_particle_sector_chi_p_to_vpp_public_artifact_available else 0.0,
            "note": (
                "The named same-sector public artifact is already available."
                if positive_particle_sector_chi_p_to_vpp_public_artifact_available
                else "Promotion remains absent because the explicit mapping route and single-shape closure did not reopen together."
            ),
        },
        {
            "row_id": "solver_ready_row_promoted_to_pass",
            "status": "pass" if solver_ready_row_promoted_to_pass else "reject",
            "metric": "solver-ready row promoted to pass after source-wording refresh",
            "value": 1.0 if solver_ready_row_promoted_to_pass else 0.0,
            "note": (
                "All reopen prerequisites now close simultaneously after source-wording reinjection."
                if solver_ready_row_promoted_to_pass
                else f"Solver-ready remains blocked by {remaining_missing_artifacts}."
            ),
        },
        {
            "row_id": "mass_origin_branch_reopen_ready",
            "status": "pass" if mass_origin_branch_reopen_ready else "reject",
            "metric": "mass-origin branch reopen ready after source-wording refresh",
            "value": 1.0 if mass_origin_branch_reopen_ready else 0.0,
            "note": (
                "The branch can hand off to the spectrum pilot."
                if mass_origin_branch_reopen_ready
                else "The branch remains blocked because solver-ready did not promote."
            ),
        },
        {
            "row_id": "hand_off_to_8_7_55_2_83",
            "status": "pass" if hand_off_to_8_7_55_2_83 else "reject",
            "metric": "handoff to 8.7.55.2.83-.84 allowed after source-wording refresh",
            "value": 1.0 if hand_off_to_8_7_55_2_83 else 0.0,
            "note": (
                "Handoff to the discrete-spectrum pilot is now allowed."
                if hand_off_to_8_7_55_2_83
                else "Handoff remains blocked because source-wording reinjection did not reopen the branch."
            ),
        },
        {
            "row_id": "source_wording_branch_source_level_blocker_count",
            "status": "inventory",
            "metric": "remaining source-level blocker count after source-wording refresh",
            "value": float(len(remaining_source_level_blockers)),
            "note": (
                f"Remaining source-level blockers are {remaining_source_level_blockers}."
                if remaining_source_level_blockers
                else "No source-level blockers remain after source-wording refresh."
            ),
        },
        {
            "row_id": "source_wording_branch_remaining_artifact_count",
            "status": "inventory",
            "metric": "remaining artifact-level missing count after source-wording refresh",
            "value": float(len(remaining_missing_artifacts)),
            "note": (
                f"Remaining artifact-level missing items are {remaining_missing_artifacts}."
                if remaining_missing_artifacts
                else "No artifact-level blockers remain after source-wording refresh."
            ),
        },
    ]

    overall_status = (
        "source_wording_branch_refresh_reopen_ready"
        if hand_off_to_8_7_55_2_83
        else "source_wording_branch_refresh_still_blocked"
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "source-wording branch refresh / handoff",
        },
        "inputs": {
            "mass_origin_split_source_branch_refresh_json": _relative_str(PRIOR_REFRESH_JSON),
            "mass_origin_shell_anchor_semantic_bridge_contract_json": _relative_str(SHELL_CONTRACT_JSON),
            "mass_origin_shell_anchor_semantic_bridge_json": _relative_str(SHELL_BRIDGE_JSON),
            "mass_origin_explicit_mapping_literal_fragment_inventory_json": _relative_str(LITERAL_INVENTORY_JSON),
            "mass_origin_explicit_mapping_literal_lift_json": _relative_str(LITERAL_LIFT_JSON),
            "mass_origin_source_wording_closure_json": _relative_str(SOURCE_WORDING_CLOSURE_JSON),
        },
        "intent": "Refresh the blocked second-route stack after source-wording closure and refreeze whether the branch can hand off to .83-.84.",
        "formulas": {
            "target_value_rule": "same_sector_tiebreak_target_value_available iff source_wording_closure now closes one admissible source-wording route",
            "single_shape_rule": "single_public_vpp_shape_available remains tied to the existing refreshed candidate selection unless a later branch changes it",
            "handoff_rule": "hand_off_to_8_7_55_2_83 iff same_sector_tiebreak_target_value + single_public_vpp_shape + positive same-sector public artifact + solver-ready reopen all close together",
        },
        "rows": rows,
        "summary": {
            "same_sector_tiebreak_target_value_available": same_sector_tiebreak_target_value_available,
            "target_source_kind_or_none": target_source_kind_or_none,
            "target_value_bridge_without_new_free_parameters": target_value_bridge_without_new_free_parameters,
            "single_public_boundary_family_fixed": single_public_boundary_family_fixed,
            "single_public_vpp_shape_available": single_public_vpp_shape_available,
            "selected_candidate_id_or_none": selected_candidate_id_or_none,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": positive_particle_sector_chi_p_to_vpp_public_artifact_available,
            "solver_ready_row_promoted_to_pass": solver_ready_row_promoted_to_pass,
            "mass_origin_branch_reopen_ready": mass_origin_branch_reopen_ready,
            "hand_off_to_8_7_55_2_83": hand_off_to_8_7_55_2_83,
            "remaining_missing_artifacts": remaining_missing_artifacts,
            "remaining_source_level_blockers": remaining_source_level_blockers,
        },
        "decision": {
            "overall_status": overall_status,
            "keep_mass_origin_branch_blocked": not hand_off_to_8_7_55_2_83,
            "same_sector_tiebreak_target_value_available": same_sector_tiebreak_target_value_available,
            "single_public_vpp_shape_available": single_public_vpp_shape_available,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": positive_particle_sector_chi_p_to_vpp_public_artifact_available,
            "mass_origin_branch_reopen_ready": mass_origin_branch_reopen_ready,
            "hand_off_to_8_7_55_2_83": hand_off_to_8_7_55_2_83,
            "remaining_missing_artifacts": remaining_missing_artifacts,
            "remaining_source_level_blockers": remaining_source_level_blockers,
        },
        "evidence": {
            "prior_refresh_summary": prior_refresh_summary,
            "shell_anchor_semantic_bridge_contract_summary": shell_contract_summary,
            "shell_anchor_semantic_bridge_summary": shell_bridge_summary,
            "explicit_mapping_literal_fragment_inventory_summary": literal_inventory_summary,
            "explicit_mapping_literal_lift_summary": literal_lift_summary,
            "source_wording_closure_summary": source_wording_summary,
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
