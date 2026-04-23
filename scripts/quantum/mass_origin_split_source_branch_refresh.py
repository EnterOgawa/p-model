#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_split_source_branch_refresh.py

Step 8.7.55.2.102:
Re-inject the split-source branch results into the blocked mass-origin stack
and re-freeze whether the second route can hand off to 8.7.55.2.83-.84.

Inputs:
  - output/public/quantum/mass_origin_target_source_branch_refresh_metrics.json
  - output/public/quantum/mass_origin_target_source_blocker_split_contract_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_target_synthesis_contract_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_target_synthesis_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_equation_lift_contract_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_equation_lift_metrics.json

Outputs:
  - output/public/quantum/mass_origin_split_source_branch_refresh_metrics.json
  - output/public/quantum/mass_origin_split_source_branch_refresh_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]

PRIOR_BRANCH_REFRESH_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_target_source_branch_refresh_metrics.json"
BLOCKER_SPLIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_target_source_blocker_split_contract_metrics.json"
SHELL_SYNTHESIS_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_target_synthesis_contract_metrics.json"
SHELL_SYNTHESIS_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_target_synthesis_metrics.json"
MAPPING_LIFT_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_equation_lift_contract_metrics.json"
MAPPING_LIFT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_equation_lift_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_split_source_branch_refresh_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_split_source_branch_refresh_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.102"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the split-source branch and refreeze .83 handoff eligibility.",
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
        PRIOR_BRANCH_REFRESH_JSON,
        BLOCKER_SPLIT_JSON,
        SHELL_SYNTHESIS_CONTRACT_JSON,
        SHELL_SYNTHESIS_JSON,
        MAPPING_LIFT_CONTRACT_JSON,
        MAPPING_LIFT_JSON,
    ):
        _require_path(path)

    prior_branch_refresh = _read_json(PRIOR_BRANCH_REFRESH_JSON)
    blocker_split = _read_json(BLOCKER_SPLIT_JSON)
    shell_synthesis_contract = _read_json(SHELL_SYNTHESIS_CONTRACT_JSON)
    shell_synthesis = _read_json(SHELL_SYNTHESIS_JSON)
    mapping_lift_contract = _read_json(MAPPING_LIFT_CONTRACT_JSON)
    mapping_lift = _read_json(MAPPING_LIFT_JSON)

    prior_branch_refresh_summary = prior_branch_refresh.get("summary", {})
    blocker_split_summary = blocker_split.get("summary", {})
    shell_synthesis_contract_summary = shell_synthesis_contract.get("summary", {})
    shell_synthesis_summary = shell_synthesis.get("summary", {})
    mapping_lift_contract_summary = mapping_lift_contract.get("summary", {})
    mapping_lift_summary = mapping_lift.get("summary", {})
    prior_reopen_retry_summary = prior_branch_refresh.get("evidence", {}).get("reopen_retry_summary", {})
    prior_disposition_summary = prior_branch_refresh.get("evidence", {}).get("disposition_summary", {})

    shell_anchor_target_value_available = bool(shell_synthesis_summary.get("shell_anchor_target_value_available", False))
    explicit_mapping_equation_available = bool(mapping_lift_summary.get("explicit_mapping_equation_available", False))

    # 条件分岐: `shell_anchor_target_value_available` を満たす経路を評価する。
    if shell_anchor_target_value_available:
        target_source_kind_or_none: str | None = "surviving_shell_anchor_pack"
        target_value_candidate_match_count = int(shell_synthesis_summary.get("candidate_match_count", 0))
        target_value_matching_candidate_ids = [
            str(item) for item in shell_synthesis_summary.get("matching_candidate_ids", [])
        ]
        target_value_bridge_without_new_free_parameters = bool(
            shell_synthesis_summary.get("bridge_without_new_free_parameters", False)
        )

    # 条件分岐: `explicit_mapping_equation_available` を満たす経路を評価する。
    elif explicit_mapping_equation_available:
        target_source_kind_or_none = "explicit_mapping_equation"
        target_value_candidate_match_count = 0
        target_value_matching_candidate_ids = []
        target_value_bridge_without_new_free_parameters = bool(
            mapping_lift_summary.get("mapping_without_new_free_parameters", False)
        )

    else:
        target_source_kind_or_none = None
        target_value_candidate_match_count = 0
        target_value_matching_candidate_ids = []
        target_value_bridge_without_new_free_parameters = False

    same_sector_tiebreak_target_value_available = target_source_kind_or_none is not None
    tie_break_route_available = bool(prior_disposition_summary.get("tie_break_route_available", False))
    single_public_boundary_family_fixed = bool(
        prior_reopen_retry_summary.get("single_public_boundary_family_fixed", False)
    )
    selected_candidate_id_or_none = (
        target_value_matching_candidate_ids[0]
        if same_sector_tiebreak_target_value_available and target_value_candidate_match_count == 1
        else None
    )
    single_public_vpp_shape_available = bool(selected_candidate_id_or_none is not None)
    positive_particle_sector_chi_p_to_vpp_public_artifact_available = bool(
        explicit_mapping_equation_available and single_public_vpp_shape_available
    )
    solver_ready_row_promoted_to_pass = bool(
        single_public_boundary_family_fixed
        and same_sector_tiebreak_target_value_available
        and single_public_vpp_shape_available
        and positive_particle_sector_chi_p_to_vpp_public_artifact_available
    )
    mass_origin_branch_reopen_ready = solver_ready_row_promoted_to_pass
    hand_off_to_8_7_55_2_83 = bool(
        mass_origin_branch_reopen_ready and tie_break_route_available
    )

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
    remaining_source_level_blockers: List[str] = []

    # 条件分岐: `not shell_anchor_target_value_available` を満たす経路を評価する。
    if not shell_anchor_target_value_available:
        remaining_source_level_blockers.append("shell_anchor_semantic_bridge_absent")

    # 条件分岐: `not explicit_mapping_equation_available` を満たす経路を評価する。

    if not explicit_mapping_equation_available:
        remaining_source_level_blockers.append("explicit_mapping_equation_lift_absent")

    remaining_source_level_blockers = _ordered_unique(remaining_source_level_blockers)

    rows = [
        {
            "row_id": "split_source_branch_refresh_complete",
            "status": "pass",
            "metric": "split-source branch refresh complete",
            "value": 1.0,
            "note": "This refresh reinjects the shell-anchor synthesis and explicit mapping-equation lift audits into the blocked second-route stack.",
        },
        {
            "row_id": "split_source_branch_same_sector_tiebreak_target_value",
            "status": "pass" if same_sector_tiebreak_target_value_available else "watch",
            "metric": "same-sector tie-break target value available after split-source refresh",
            "value": 1.0 if same_sector_tiebreak_target_value_available else 0.0,
            "note": (
                f"Target source kind is {target_source_kind_or_none}."
                if same_sector_tiebreak_target_value_available
                else f"Split-source refresh remains blocked by {remaining_source_level_blockers}."
            ),
        },
        {
            "row_id": "split_source_branch_single_public_vpp_shape",
            "status": "pass" if single_public_vpp_shape_available else "reject",
            "metric": "single public V(|P|) shape available after split-source refresh",
            "value": 1.0 if single_public_vpp_shape_available else 0.0,
            "note": (
                f"Selected candidate is {selected_candidate_id_or_none}."
                if single_public_vpp_shape_available
                else "No same-sector target value closes uniquely onto one surviving candidate."
            ),
        },
        {
            "row_id": "split_source_branch_positive_same_sector_public_artifact",
            "status": "pass" if positive_particle_sector_chi_p_to_vpp_public_artifact_available else "reject",
            "metric": "positive particle-sector chi_P -> V''(|P|_*) public artifact available after split-source refresh",
            "value": 1.0 if positive_particle_sector_chi_p_to_vpp_public_artifact_available else 0.0,
            "note": (
                "The explicit same-sector mapping equation and single public V(|P|) shape now close together."
                if positive_particle_sector_chi_p_to_vpp_public_artifact_available
                else "Promotion cannot close because the explicit mapping equation lift and/or single-shape closure remain absent."
            ),
        },
        {
            "row_id": "solver_ready_row_promoted_to_pass",
            "status": "pass" if solver_ready_row_promoted_to_pass else "reject",
            "metric": "solver-ready row promoted to pass after split-source refresh",
            "value": 1.0 if solver_ready_row_promoted_to_pass else 0.0,
            "note": (
                "All reopen prerequisites are now jointly satisfied."
                if solver_ready_row_promoted_to_pass
                else f"Solver-ready remains blocked by {remaining_missing_artifacts}."
            ),
        },
        {
            "row_id": "mass_origin_branch_reopen_ready",
            "status": "pass" if mass_origin_branch_reopen_ready else "reject",
            "metric": "mass-origin branch reopen ready after split-source refresh",
            "value": 1.0 if mass_origin_branch_reopen_ready else 0.0,
            "note": (
                "The branch may reopen into the no-free-parameter mass-solver path."
                if mass_origin_branch_reopen_ready
                else "The branch remains blocked because solver-ready did not promote."
            ),
        },
        {
            "row_id": "hand_off_to_8_7_55_2_83",
            "status": "pass" if hand_off_to_8_7_55_2_83 else "reject",
            "metric": "handoff to 8.7.55.2.83-.84 allowed after split-source refresh",
            "value": 1.0 if hand_off_to_8_7_55_2_83 else 0.0,
            "note": (
                "The branch may continue into the discrete-spectrum pilot."
                if hand_off_to_8_7_55_2_83
                else "Handoff remains blocked because the split-source refresh did not reopen the branch."
            ),
        },
        {
            "row_id": "split_source_branch_source_level_blocker_count",
            "status": "inventory",
            "metric": "remaining source-level blocker count after split-source refresh",
            "value": float(len(remaining_source_level_blockers)),
            "note": (
                f"Remaining source-level blockers are {remaining_source_level_blockers}."
                if remaining_source_level_blockers
                else "No source-level blockers remain."
            ),
        },
        {
            "row_id": "split_source_branch_remaining_artifact_count",
            "status": "inventory",
            "metric": "remaining artifact-level missing count after split-source refresh",
            "value": float(len(remaining_missing_artifacts)),
            "note": (
                f"Remaining artifact-level missing items are {remaining_missing_artifacts}."
                if remaining_missing_artifacts
                else "No artifact-level missing items remain."
            ),
        },
    ]

    overall_status = (
        "split_source_branch_refresh_handoff_ready"
        if hand_off_to_8_7_55_2_83
        else "split_source_branch_refresh_still_blocked"
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "split-source branch refresh / handoff",
        },
        "inputs": {
            "mass_origin_target_source_branch_refresh_json": _relative_str(PRIOR_BRANCH_REFRESH_JSON),
            "mass_origin_target_source_blocker_split_contract_json": _relative_str(BLOCKER_SPLIT_JSON),
            "mass_origin_shell_anchor_target_synthesis_contract_json": _relative_str(SHELL_SYNTHESIS_CONTRACT_JSON),
            "mass_origin_shell_anchor_target_synthesis_json": _relative_str(SHELL_SYNTHESIS_JSON),
            "mass_origin_explicit_mapping_equation_lift_contract_json": _relative_str(MAPPING_LIFT_CONTRACT_JSON),
            "mass_origin_explicit_mapping_equation_lift_json": _relative_str(MAPPING_LIFT_JSON),
        },
        "intent": "Refresh the blocked second-route stack after the split-source branch and refreeze whether .83-.84 handoff is now allowed.",
        "formulas": {
            "target_value_rule": "same_sector_tiebreak_target_value_available iff either the shell-anchor synthesis audit or the explicit mapping-equation lift audit closes inside the frozen same-sector contract",
            "single_shape_rule": "single_public_vpp_shape_available iff the refreshed target value selects exactly one surviving candidate under the existing tie-break route",
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
            "prior_branch_refresh_summary": prior_branch_refresh_summary,
            "blocker_split_summary": blocker_split_summary,
            "shell_synthesis_contract_summary": shell_synthesis_contract_summary,
            "shell_synthesis_summary": shell_synthesis_summary,
            "mapping_lift_contract_summary": mapping_lift_contract_summary,
            "mapping_lift_summary": mapping_lift_summary,
            "prior_reopen_retry_summary": prior_reopen_retry_summary,
            "prior_disposition_summary": prior_disposition_summary,
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
