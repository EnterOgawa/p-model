#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_explicit_mapping_equation_lift_audit.py

Step 8.7.55.2.101:
Audit whether the current public canonical pack already satisfies the frozen
explicit mapping-equation lift contract without introducing new free
parameters.

Inputs:
  - output/public/quantum/mass_origin_explicit_mapping_equation_lift_contract_metrics.json
  - output/public/quantum/mass_origin_same_sector_chi_to_vpp_contract_metrics.json
  - output/public/quantum/mass_origin_positive_particle_sector_chi_to_vpp_metrics.json
  - output/public/quantum/mass_origin_same_sector_mapping_equation_source_metrics.json
  - output/public/quantum/mass_origin_target_source_blocker_split_contract_metrics.json

Outputs:
  - output/public/quantum/mass_origin_explicit_mapping_equation_lift_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_equation_lift_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

LIFT_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_equation_lift_contract_metrics.json"
CHI_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_chi_to_vpp_contract_metrics.json"
PROMOTION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_positive_particle_sector_chi_to_vpp_metrics.json"
MAPPING_SOURCE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_mapping_equation_source_metrics.json"
BLOCKER_SPLIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_target_source_blocker_split_contract_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_equation_lift_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_equation_lift_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.101"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether the explicit same-sector mapping equation can already be lifted into public canonical form.",
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


# 関数: `_count_true` の入出力契約と処理意図を定義する。

def _count_true(flags: Dict[str, bool]) -> int:
    return sum(1 for value in flags.values() if value)


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (
        LIFT_CONTRACT_JSON,
        CHI_CONTRACT_JSON,
        PROMOTION_JSON,
        MAPPING_SOURCE_JSON,
        BLOCKER_SPLIT_JSON,
    ):
        _require_path(path)

    lift_contract = _read_json(LIFT_CONTRACT_JSON)
    chi_contract = _read_json(CHI_CONTRACT_JSON)
    promotion = _read_json(PROMOTION_JSON)
    mapping_source = _read_json(MAPPING_SOURCE_JSON)
    blocker_split = _read_json(BLOCKER_SPLIT_JSON)

    lift_contract_summary = lift_contract.get("summary", {})
    chi_contract_summary = chi_contract.get("summary", {})
    promotion_summary = promotion.get("summary", {})
    mapping_source_summary = mapping_source.get("summary", {})
    blocker_split_summary = blocker_split.get("summary", {})

    required_mapping_row_fields = [str(item) for item in lift_contract_summary.get("required_mapping_row_fields", [])]
    required_mapping_equation_slots = [str(item) for item in lift_contract_summary.get("required_mapping_equation_slots", [])]
    forbidden_placeholder_substitutions = [
        str(item) for item in lift_contract_summary.get("forbidden_placeholder_substitutions", [])
    ]

    field_presence = {
        "same_particle_sector_only": bool(chi_contract_summary.get("same_particle_sector_only", False)),
        "explicit_mapping_equation": bool(mapping_source_summary.get("explicit_mapping_equation_available", False)),
        "sign_statement": bool(chi_contract_summary.get("sign_convention_frozen", False)),
        "unit_statement": bool(chi_contract_summary.get("unit_contract_frozen", False)),
        "reference_point_symbol_absP_star": str(chi_contract_summary.get("reference_point_symbol")) == "|P|_*",
        "public_shell_family_numerical_rows": bool(promotion_summary.get("shell_family_numerical_rows_present", False)),
        "no_new_free_parameter_note": bool(promotion_summary.get("no_new_free_parameter_note_present", False)),
        "source_kind_explicit_mapping_equation": bool(mapping_source_summary.get("explicit_mapping_source_kind_allowed", False)),
    }
    present_mapping_row_field_count = _count_true(field_presence)

    equation_slot_presence = {
        "lhs_observable_chi_P_or_declared_same_sector_shell_equivalent": str(chi_contract_summary.get("chi_symbol")) == "chi_P",
        "rhs_curvature_symbol_Vpp_absP_star": str(chi_contract_summary.get("curvature_symbol")) == "V''(|P|_*)",
        "reference_point_absP_star": str(chi_contract_summary.get("reference_point_symbol")) == "|P|_*",
        "same_sector_equivalence_statement_or_none": bool(mapping_source_summary.get("explicit_mapping_equation_available", False)),
        "mapping_operator_or_relation": bool(mapping_source_summary.get("explicit_mapping_equation_available", False)),
    }
    present_mapping_equation_slot_count = _count_true(equation_slot_presence)

    explicit_mapping_candidate_listed = bool(mapping_source_summary.get("explicit_mapping_candidate_listed", False))
    explicit_mapping_source_kind_allowed = bool(mapping_source_summary.get("explicit_mapping_source_kind_allowed", False))
    explicit_mapping_equation_available = bool(
        mapping_source_summary.get("explicit_mapping_equation_available", False)
        and present_mapping_row_field_count == len(required_mapping_row_fields)
        and present_mapping_equation_slot_count == len(required_mapping_equation_slots)
    )
    lifted_mapping_equation_kind_or_none = (
        "explicit_same_sector_chi_to_vpp_row" if explicit_mapping_equation_available else None
    )
    mapping_without_new_free_parameters = bool(
        explicit_mapping_equation_available
        and promotion_summary.get("no_new_free_parameter_note_present", False)
        and chi_contract_summary.get("same_particle_sector_only", False)
    )
    downstream_single_public_vpp_shape_available = bool(
        promotion_summary.get("single_public_vpp_shape_available", False)
    )

    missing_lift_requirements: List[str] = []

    # 条件分岐: `not explicit_mapping_source_kind_allowed` を満たす経路を評価する。
    if not explicit_mapping_source_kind_allowed:
        missing_lift_requirements.append("explicit_mapping_source_kind_not_allowed")

    # 条件分岐: `not explicit_mapping_candidate_listed` を満たす経路を評価する。

    if not explicit_mapping_candidate_listed:
        missing_lift_requirements.append("explicit_mapping_equation_not_in_inventory")

    # 条件分岐: `not field_presence["explicit_mapping_equation"]` を満たす経路を評価する。

    if not field_presence["explicit_mapping_equation"]:
        missing_lift_requirements.append("explicit_mapping_equation")

    rows = [
        {
            "row_id": "explicit_mapping_equation_lift_audit_complete",
            "status": "pass",
            "metric": "explicit mapping-equation lift audit complete",
            "value": 1.0,
            "note": "This audit checks whether the frozen explicit same-sector mapping row can already be lifted from the current public canonical pack.",
        },
        {
            "row_id": "explicit_mapping_equation_lift_source_kind_allowed",
            "status": "pass" if explicit_mapping_source_kind_allowed else "reject",
            "metric": "explicit mapping-equation source kind remains allowed",
            "value": 1.0 if explicit_mapping_source_kind_allowed else 0.0,
            "note": "The lift route remains admissible under the frozen same-sector source contract.",
        },
        {
            "row_id": "explicit_mapping_equation_lift_candidate_listed",
            "status": "pass" if explicit_mapping_candidate_listed else "reject",
            "metric": "explicit mapping-equation candidate remains listed",
            "value": 1.0 if explicit_mapping_candidate_listed else 0.0,
            "note": "The target-source inventory still carries the explicit mapping-equation placeholder row.",
        },
        {
            "row_id": "explicit_mapping_equation_lift_present_field_count",
            "status": "inventory",
            "metric": "present required mapping row field count",
            "value": float(present_mapping_row_field_count),
            "note": f"Present row fields are {sorted([key for key, value in field_presence.items() if value])}.",
        },
        {
            "row_id": "explicit_mapping_equation_lift_present_equation_slot_count",
            "status": "inventory",
            "metric": "present required mapping-equation slot count",
            "value": float(present_mapping_equation_slot_count),
            "note": f"Present equation slots are {sorted([key for key, value in equation_slot_presence.items() if value])}.",
        },
        {
            "row_id": "explicit_mapping_equation_lift_forbidden_placeholder_count",
            "status": "inventory",
            "metric": "forbidden placeholder substitution count",
            "value": float(len(forbidden_placeholder_substitutions)),
            "note": f"Forbidden placeholder substitutions are {forbidden_placeholder_substitutions}.",
        },
        {
            "row_id": "explicit_mapping_equation_lift_available",
            "status": "pass" if explicit_mapping_equation_available else "watch",
            "metric": "explicit mapping equation already liftable into public canonical form",
            "value": 1.0 if explicit_mapping_equation_available else 0.0,
            "note": (
                f"Lifted mapping equation kind is {lifted_mapping_equation_kind_or_none}."
                if explicit_mapping_equation_available
                else f"The lift remains absent because the missing requirements are {missing_lift_requirements}."
            ),
        },
        {
            "row_id": "explicit_mapping_equation_lift_without_new_free_parameters",
            "status": "pass" if mapping_without_new_free_parameters else "reject",
            "metric": "explicit mapping equation lift closes without new free parameters",
            "value": 1.0 if mapping_without_new_free_parameters else 0.0,
            "note": (
                "The lifted mapping row stays inside the frozen same-sector no-new-free-parameter contract."
                if mapping_without_new_free_parameters
                else "The lifted mapping row is not yet available, so the no-new-free-parameter closure cannot be claimed."
            ),
        },
        {
            "row_id": "explicit_mapping_equation_lift_single_shape_still_downstream",
            "status": "watch" if not downstream_single_public_vpp_shape_available else "pass",
            "metric": "single public V(|P|) shape remains downstream of explicit mapping-equation lift",
            "value": 1.0 if downstream_single_public_vpp_shape_available else 0.0,
            "note": (
                "Single public V(|P|) shape is already available downstream."
                if downstream_single_public_vpp_shape_available
                else "Single public V(|P|) shape is still downstream; the lift audit freezes the mapping row independently of final single-shape closure."
            ),
        },
        {
            "row_id": "explicit_mapping_equation_lift_route_still_admissible",
            "status": "pass" if blocker_split_summary.get("explicit_mapping_route_still_admissible", False) else "reject",
            "metric": "explicit mapping-equation route remains admissible inside split-source branch",
            "value": 1.0 if blocker_split_summary.get("explicit_mapping_route_still_admissible", False) else 0.0,
            "note": "The split-source branch still keeps explicit mapping-equation lift as an admissible same-sector route.",
        },
    ]

    overall_status = (
        "explicit_mapping_equation_lift_available"
        if explicit_mapping_equation_available
        else "explicit_mapping_equation_lift_frozen_absent"
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "explicit mapping-equation lift audit",
        },
        "inputs": {
            "mass_origin_explicit_mapping_equation_lift_contract_json": _relative_str(LIFT_CONTRACT_JSON),
            "mass_origin_same_sector_chi_to_vpp_contract_json": _relative_str(CHI_CONTRACT_JSON),
            "mass_origin_positive_particle_sector_chi_to_vpp_json": _relative_str(PROMOTION_JSON),
            "mass_origin_same_sector_mapping_equation_source_json": _relative_str(MAPPING_SOURCE_JSON),
            "mass_origin_target_source_blocker_split_contract_json": _relative_str(BLOCKER_SPLIT_JSON),
        },
        "intent": "Audit whether the current public canonical pack already satisfies the frozen explicit mapping-equation lift contract without new free parameters.",
        "formulas": {
            "lift_rule": "explicit_mapping_equation_available iff the required mapping row fields, equation slots, and allowed source-kind route are simultaneously present in current public canonical form",
            "no_new_parameter_rule": "mapping_without_new_free_parameters iff the lifted mapping row is available and remains inside the frozen same-sector no-new-free-parameter contract",
        },
        "rows": rows,
        "summary": {
            "explicit_mapping_equation_available": explicit_mapping_equation_available,
            "lifted_mapping_equation_kind_or_none": lifted_mapping_equation_kind_or_none,
            "mapping_without_new_free_parameters": mapping_without_new_free_parameters,
            "required_mapping_row_field_count": len(required_mapping_row_fields),
            "present_mapping_row_field_count": present_mapping_row_field_count,
            "required_mapping_equation_slot_count": len(required_mapping_equation_slots),
            "present_mapping_equation_slot_count": present_mapping_equation_slot_count,
            "missing_lift_requirements": missing_lift_requirements,
            "tiebreak_invariant_name": lift_contract_summary.get("tiebreak_invariant_name"),
        },
        "decision": {
            "overall_status": overall_status,
            "keep_mass_origin_branch_blocked": True,
            "explicit_mapping_equation_available": explicit_mapping_equation_available,
            "lifted_mapping_equation_kind_or_none": lifted_mapping_equation_kind_or_none,
            "mapping_without_new_free_parameters": mapping_without_new_free_parameters,
            "missing_lift_requirements": missing_lift_requirements,
        },
        "evidence": {
            "lift_contract_summary": lift_contract_summary,
            "chi_contract_summary": chi_contract_summary,
            "promotion_summary": promotion_summary,
            "mapping_source_summary": mapping_source_summary,
            "blocker_split_summary": blocker_split_summary,
            "field_presence": field_presence,
            "equation_slot_presence": equation_slot_presence,
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
