#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_target_source_branch_refresh.py

Step 8.7.55.2.96:
Re-inject the target-source branch closure result into the retry branch and
freeze whether the mass-origin route can hand off to 8.7.55.2.83-.84.

Inputs:
  - output/public/quantum/mass_origin_single_public_vpp_shape_closure_retry_metrics.json
  - output/public/quantum/mass_origin_positive_particle_sector_chi_to_vpp_retry_metrics.json
  - output/public/quantum/mass_origin_solver_ready_reopen_gate_retry_metrics.json
  - output/public/quantum/mass_origin_tiebreak_branch_disposition_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_contract_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_inventory_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_shell_anchor_metrics.json
  - output/public/quantum/mass_origin_same_sector_mapping_equation_source_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_value_closure_metrics.json

Outputs:
  - output/public/quantum/mass_origin_target_source_branch_refresh_metrics.json
  - output/public/quantum/mass_origin_target_source_branch_refresh_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]

CLOSURE_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_single_public_vpp_shape_closure_retry_metrics.json"
PROMOTION_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_positive_particle_sector_chi_to_vpp_retry_metrics.json"
REOPEN_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_solver_ready_reopen_gate_retry_metrics.json"
DISPOSITION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_tiebreak_branch_disposition_metrics.json"
SOURCE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_source_contract_metrics.json"
SOURCE_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_source_inventory_metrics.json"
SHELL_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_shell_anchor_metrics.json"
MAPPING_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_mapping_equation_source_metrics.json"
TARGET_VALUE_CLOSURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_value_closure_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_target_source_branch_refresh_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_target_source_branch_refresh_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.96"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the target-source branch and freeze .83 handoff eligibility.",
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
        CLOSURE_RETRY_JSON,
        PROMOTION_RETRY_JSON,
        REOPEN_RETRY_JSON,
        DISPOSITION_JSON,
        SOURCE_CONTRACT_JSON,
        SOURCE_INVENTORY_JSON,
        SHELL_AUDIT_JSON,
        MAPPING_AUDIT_JSON,
        TARGET_VALUE_CLOSURE_JSON,
    ):
        _require_path(path)

    closure_retry = _read_json(CLOSURE_RETRY_JSON)
    promotion_retry = _read_json(PROMOTION_RETRY_JSON)
    reopen_retry = _read_json(REOPEN_RETRY_JSON)
    disposition = _read_json(DISPOSITION_JSON)
    source_contract = _read_json(SOURCE_CONTRACT_JSON)
    source_inventory = _read_json(SOURCE_INVENTORY_JSON)
    shell_audit = _read_json(SHELL_AUDIT_JSON)
    mapping_audit = _read_json(MAPPING_AUDIT_JSON)
    target_value_closure = _read_json(TARGET_VALUE_CLOSURE_JSON)

    closure_retry_summary = closure_retry.get("summary", {})
    promotion_retry_summary = promotion_retry.get("summary", {})
    reopen_retry_summary = reopen_retry.get("summary", {})
    disposition_summary = disposition.get("summary", {})
    source_contract_summary = source_contract.get("summary", {})
    source_inventory_summary = source_inventory.get("summary", {})
    shell_summary = shell_audit.get("summary", {})
    mapping_summary = mapping_audit.get("summary", {})
    target_value_summary = target_value_closure.get("summary", {})

    target_value_available = bool(target_value_summary.get("target_value_available", False))
    single_public_vpp_shape_available = bool(closure_retry_summary.get("single_public_vpp_shape_available", False))
    positive_particle_sector_public_artifact_available = bool(
        promotion_retry_summary.get("positive_particle_sector_chi_p_to_vpp_public_artifact_available", False)
    )

    solver_ready_row_promoted_to_pass = bool(
        target_value_available
        and single_public_vpp_shape_available
        and positive_particle_sector_public_artifact_available
    )
    mass_origin_branch_reopen_ready = solver_ready_row_promoted_to_pass
    hand_off_to_8_7_55_2_83 = bool(
        mass_origin_branch_reopen_ready
        and disposition_summary.get("tie_break_route_available", False)
        and target_value_available
    )

    artifact_level_missing = _ordered_unique(
        list(target_value_closure.get("decision", {}).get("nonclosure_reason_or_none", []) if False else [])
    )
    artifact_level_missing = _ordered_unique(
        list(reopen_retry_summary.get("next_required_artifacts", []))
        + list(closure_retry.get("decision", {}).get("next_required_artifacts", []))
        + list(promotion_retry.get("decision", {}).get("next_required_artifacts", []))
    )

    # 条件分岐: `not target_value_available` を満たす経路を評価する。
    if not target_value_available:
        artifact_level_missing = _ordered_unique(["same_sector_tiebreak_target_value"] + artifact_level_missing)

    # 条件分岐: `not single_public_vpp_shape_available` を満たす経路を評価する。

    if not single_public_vpp_shape_available:
        artifact_level_missing = _ordered_unique(["single_public_vpp_shape"] + artifact_level_missing)

    # 条件分岐: `not positive_particle_sector_public_artifact_available` を満たす経路を評価する。

    if not positive_particle_sector_public_artifact_available:
        artifact_level_missing = _ordered_unique(
            ["positive_particle_sector_chi_p_to_vpp_public_artifact"] + artifact_level_missing
        )

    # 条件分岐: `not solver_ready_row_promoted_to_pass` を満たす経路を評価する。

    if not solver_ready_row_promoted_to_pass:
        artifact_level_missing = _ordered_unique(["solver_ready_row_promoted_to_pass"] + artifact_level_missing)

    source_level_blockers: List[str] = []

    # 条件分岐: `not shell_summary.get("shell_anchor_target_value_available", False)` を満たす経路を評価する。
    if not shell_summary.get("shell_anchor_target_value_available", False):
        source_level_blockers.append("shell_anchor_target_value_missing")

    # 条件分岐: `not mapping_summary.get("explicit_mapping_equation_available", False)` を満たす経路を評価する。

    if not mapping_summary.get("explicit_mapping_equation_available", False):
        source_level_blockers.append("explicit_mapping_equation_absent")

    source_level_blockers = _ordered_unique(source_level_blockers)

    current_target_source_kind_or_none = target_value_summary.get("target_source_kind_or_none")
    current_target_value_available = bool(source_contract_summary.get("current_target_value_available", False))

    rows = [
        {
            "row_id": "target_source_branch_refresh_complete",
            "status": "pass",
            "metric": "target-source branch refresh complete",
            "value": 1.0,
            "note": "This refresh reinjects the target-source closure result into the retry branch and re-freezes .83 handoff eligibility.",
        },
        {
            "row_id": "target_source_branch_target_value_available",
            "status": "pass" if target_value_available else "watch",
            "metric": "same-sector tiebreak target value available after closure injection",
            "value": 1.0 if target_value_available else 0.0,
            "note": (
                f"Target value source kind is {current_target_source_kind_or_none}."
                if target_value_available
                else f"Target value remains absent because {target_value_summary.get('nonclosure_reason_or_none')}."
            ),
        },
        {
            "row_id": "target_source_branch_single_public_vpp_shape",
            "status": "pass" if single_public_vpp_shape_available else "reject",
            "metric": "single public V(|P|) shape available after target-source refresh",
            "value": 1.0 if single_public_vpp_shape_available else 0.0,
            "note": (
                f"Selected candidate is {closure_retry_summary.get('selected_candidate_id_or_none')}."
                if single_public_vpp_shape_available
                else f"Single-shape closure remains blocked by {closure_retry_summary.get('nonclosure_reason_or_none')}."
            ),
        },
        {
            "row_id": "target_source_branch_positive_same_sector_public_artifact",
            "status": "pass" if positive_particle_sector_public_artifact_available else "reject",
            "metric": "positive particle-sector chi_P -> V''(|P|_*) public artifact available after refresh",
            "value": 1.0 if positive_particle_sector_public_artifact_available else 0.0,
            "note": (
                "The named artifact is now available in public canonical form."
                if positive_particle_sector_public_artifact_available
                else f"Promotion remains blocked by {promotion_retry_summary.get('missing_promotion_requirements')}."
            ),
        },
        {
            "row_id": "solver_ready_row_promoted_to_pass",
            "status": "pass" if solver_ready_row_promoted_to_pass else "reject",
            "metric": "solver-ready row promoted to pass after target-source refresh",
            "value": 1.0 if solver_ready_row_promoted_to_pass else 0.0,
            "note": (
                "Target-source reinjection closes the reopen prerequisites."
                if solver_ready_row_promoted_to_pass
                else f"Solver-ready stays blocked by {artifact_level_missing}."
            ),
        },
        {
            "row_id": "mass_origin_branch_reopen_ready",
            "status": "pass" if mass_origin_branch_reopen_ready else "reject",
            "metric": "mass-origin branch reopen ready after target-source refresh",
            "value": 1.0 if mass_origin_branch_reopen_ready else 0.0,
            "note": (
                "The branch can hand off to the no-free-parameter mass solver."
                if mass_origin_branch_reopen_ready
                else "The branch remains blocked because solver-ready did not promote."
            ),
        },
        {
            "row_id": "hand_off_to_8_7_55_2_83",
            "status": "pass" if hand_off_to_8_7_55_2_83 else "reject",
            "metric": "handoff to 8.7.55.2.83-.84 allowed after target-source refresh",
            "value": 1.0 if hand_off_to_8_7_55_2_83 else 0.0,
            "note": (
                "The branch may continue into the no-free-parameter mass-spectrum pilot."
                if hand_off_to_8_7_55_2_83
                else "Handoff remains blocked because target-source reinjection did not reopen the branch."
            ),
        },
        {
            "row_id": "target_source_branch_source_level_blocker_count",
            "status": "inventory",
            "metric": "remaining source-level blocker count",
            "value": float(len(source_level_blockers)),
            "note": (
                f"Remaining source-level blockers: {', '.join(source_level_blockers)}."
                if source_level_blockers
                else "No source-level blockers remain."
            ),
        },
        {
            "row_id": "target_source_branch_artifact_level_missing_count",
            "status": "inventory",
            "metric": "remaining artifact-level missing count",
            "value": float(len(artifact_level_missing)),
            "note": (
                f"Remaining artifact-level missing items: {', '.join(artifact_level_missing)}."
                if artifact_level_missing
                else "No artifact-level missing items remain."
            ),
        },
    ]

    overall_status = (
        "target_source_branch_refresh_handoff_ready"
        if hand_off_to_8_7_55_2_83
        else "target_source_branch_refresh_still_blocked"
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "target-source branch refresh / handoff",
        },
        "inputs": {
            "mass_origin_single_public_vpp_shape_closure_retry_json": _relative_str(CLOSURE_RETRY_JSON),
            "mass_origin_positive_particle_sector_chi_to_vpp_retry_json": _relative_str(PROMOTION_RETRY_JSON),
            "mass_origin_solver_ready_reopen_gate_retry_json": _relative_str(REOPEN_RETRY_JSON),
            "mass_origin_tiebreak_branch_disposition_json": _relative_str(DISPOSITION_JSON),
            "mass_origin_same_sector_tiebreak_target_source_contract_json": _relative_str(SOURCE_CONTRACT_JSON),
            "mass_origin_same_sector_tiebreak_target_source_inventory_json": _relative_str(SOURCE_INVENTORY_JSON),
            "mass_origin_same_sector_tiebreak_shell_anchor_json": _relative_str(SHELL_AUDIT_JSON),
            "mass_origin_same_sector_mapping_equation_source_json": _relative_str(MAPPING_AUDIT_JSON),
            "mass_origin_same_sector_tiebreak_target_value_closure_json": _relative_str(TARGET_VALUE_CLOSURE_JSON),
        },
        "intent": "Refresh the target-source branch and refreeze whether the second route can hand off to .83-.84.",
        "formulas": {
            "refresh_rule": "single_public_vpp_shape, positive same-sector public artifact, and solver-ready reopen state are re-evaluated after injecting the target-source closure result",
            "handoff_rule": "hand_off_to_8_7_55_2_83 iff target_value_available + single_public_vpp_shape_available + positive_particle_sector_chi_p_to_vpp_public_artifact_available jointly close and solver_ready_row_promoted_to_pass becomes true",
            "source_blocker_rule": "when target value closure remains absent, the residual source-level blocker set is reduced to shell_anchor_target_value_missing and/or explicit_mapping_equation_absent",
        },
        "rows": rows,
        "summary": {
            "target_value_available": target_value_available,
            "target_source_kind_or_none": current_target_source_kind_or_none,
            "single_public_vpp_shape_available": single_public_vpp_shape_available,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": positive_particle_sector_public_artifact_available,
            "solver_ready_row_promoted_to_pass": solver_ready_row_promoted_to_pass,
            "mass_origin_branch_reopen_ready": mass_origin_branch_reopen_ready,
            "hand_off_to_8_7_55_2_83": hand_off_to_8_7_55_2_83,
            "remaining_missing_artifacts": artifact_level_missing,
            "remaining_source_level_blockers": source_level_blockers,
            "current_target_value_available_from_contract": current_target_value_available,
            "candidate_source_count": source_inventory_summary.get("candidate_source_count"),
        },
        "decision": {
            "overall_status": overall_status,
            "keep_mass_origin_branch_blocked": not hand_off_to_8_7_55_2_83,
            "mass_origin_branch_reopen_ready": mass_origin_branch_reopen_ready,
            "hand_off_to_8_7_55_2_83": hand_off_to_8_7_55_2_83,
            "remaining_missing_artifacts": artifact_level_missing,
            "remaining_source_level_blockers": source_level_blockers,
        },
        "evidence": {
            "closure_retry_summary": closure_retry_summary,
            "promotion_retry_summary": promotion_retry_summary,
            "reopen_retry_summary": reopen_retry_summary,
            "disposition_summary": disposition_summary,
            "target_source_contract_summary": source_contract_summary,
            "target_source_inventory_summary": source_inventory_summary,
            "shell_anchor_summary": shell_summary,
            "mapping_equation_summary": mapping_summary,
            "target_value_closure_summary": target_value_summary,
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
