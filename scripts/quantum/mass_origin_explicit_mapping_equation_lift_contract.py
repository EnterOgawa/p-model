#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_explicit_mapping_equation_lift_contract.py

Step 8.7.55.2.100:
Freeze the minimum contract required to lift an explicit same-sector
chi_P -> V''(|P|_*) mapping equation row into the public canonical pack.

Inputs:
  - output/public/quantum/mass_origin_same_sector_chi_to_vpp_contract_metrics.json
  - output/public/quantum/mass_origin_positive_particle_sector_chi_to_vpp_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_contract_metrics.json
  - output/public/quantum/mass_origin_same_sector_mapping_equation_source_metrics.json
  - output/public/quantum/mass_origin_target_source_blocker_split_contract_metrics.json

Outputs:
  - output/public/quantum/mass_origin_explicit_mapping_equation_lift_contract_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_equation_lift_contract_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

CHI_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_chi_to_vpp_contract_metrics.json"
PROMOTION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_positive_particle_sector_chi_to_vpp_metrics.json"
TARGET_SOURCE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_source_contract_metrics.json"
MAPPING_SOURCE_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_mapping_equation_source_metrics.json"
BLOCKER_SPLIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_target_source_blocker_split_contract_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_equation_lift_contract_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_equation_lift_contract_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.100"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the explicit mapping-equation lift contract.",
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
    for path in (
        CHI_CONTRACT_JSON,
        PROMOTION_JSON,
        TARGET_SOURCE_CONTRACT_JSON,
        MAPPING_SOURCE_AUDIT_JSON,
        BLOCKER_SPLIT_JSON,
    ):
        _require_path(path)

    chi_contract = _read_json(CHI_CONTRACT_JSON)
    promotion = _read_json(PROMOTION_JSON)
    target_source_contract = _read_json(TARGET_SOURCE_CONTRACT_JSON)
    mapping_source_audit = _read_json(MAPPING_SOURCE_AUDIT_JSON)
    blocker_split = _read_json(BLOCKER_SPLIT_JSON)

    chi_contract_summary = chi_contract.get("summary", {})
    promotion_summary = promotion.get("summary", {})
    target_source_contract_summary = target_source_contract.get("summary", {})
    mapping_source_summary = mapping_source_audit.get("summary", {})
    blocker_split_summary = blocker_split.get("summary", {})

    required_mapping_row_fields = [
        "same_particle_sector_only",
        "explicit_mapping_equation",
        "sign_statement",
        "unit_statement",
        "reference_point_symbol_absP_star",
        "public_shell_family_numerical_rows",
        "no_new_free_parameter_note",
        "source_kind_explicit_mapping_equation",
    ]
    required_mapping_equation_slots = [
        "lhs_observable_chi_P_or_declared_same_sector_shell_equivalent",
        "rhs_curvature_symbol_Vpp_absP_star",
        "reference_point_absP_star",
        "same_sector_equivalence_statement_or_none",
        "mapping_operator_or_relation",
    ]
    forbidden_placeholder_substitutions = [
        "cross_sector_proxy_substitution",
        "interface_only_spread_substitution",
        "phenomenological_backsolve",
        "shell_anchor_target_value_placeholder_substitution",
        "single_public_vpp_shape_placeholder_substitution",
        "undeclared_shell_equivalent_substitution",
    ]

    mapping_equation_source_kind_allowed = "explicit_mapping_equation" in target_source_contract_summary.get(
        "allowed_source_kind_ids", []
    )
    mapping_candidate_listed = bool(mapping_source_summary.get("explicit_mapping_candidate_listed", False))
    mapping_route_still_admissible = bool(blocker_split_summary.get("explicit_mapping_route_still_admissible", False))
    sign_units_reference_point_present = bool(promotion_summary.get("sign_units_reference_point_present", False))
    shell_family_numerical_rows_present = bool(promotion_summary.get("shell_family_numerical_rows_present", False))
    no_new_free_parameter_note_present = bool(promotion_summary.get("no_new_free_parameter_note_present", False))
    explicit_mapping_equation_available_now = bool(mapping_source_summary.get("explicit_mapping_equation_available", False))
    single_public_vpp_shape_available_now = bool(promotion_summary.get("single_public_vpp_shape_available", False))

    mapping_equation_lift_ready = bool(
        chi_contract_summary.get("chi_to_vpp_mapping_contract_frozen", False)
        and mapping_equation_source_kind_allowed
        and mapping_candidate_listed
        and mapping_route_still_admissible
        and sign_units_reference_point_present
        and shell_family_numerical_rows_present
        and no_new_free_parameter_note_present
    )

    rows = [
        {
            "row_id": "explicit_mapping_equation_lift_contract_complete",
            "status": "pass",
            "metric": "explicit mapping-equation lift contract complete",
            "value": 1.0,
            "note": "This step freezes the minimum same-sector contract needed to lift an explicit chi_P -> V''(|P|_*) row into public canonical form.",
        },
        {
            "row_id": "explicit_mapping_equation_lift_same_particle_sector_only",
            "status": "pass" if chi_contract_summary.get("same_particle_sector_only", False) else "reject",
            "metric": "lift contract stays inside same particle sector only",
            "value": 1.0 if chi_contract_summary.get("same_particle_sector_only", False) else 0.0,
            "note": "The future lift may not rely on cross-sector or interface-only substitutes.",
        },
        {
            "row_id": "explicit_mapping_equation_lift_source_kind_allowed",
            "status": "pass" if mapping_equation_source_kind_allowed else "reject",
            "metric": "explicit mapping-equation source kind remains allowed",
            "value": 1.0 if mapping_equation_source_kind_allowed else 0.0,
            "note": f"Allowed source kinds are {target_source_contract_summary.get('allowed_source_kind_ids', [])}.",
        },
        {
            "row_id": "explicit_mapping_equation_lift_candidate_listed",
            "status": "pass" if mapping_candidate_listed else "reject",
            "metric": "explicit mapping-equation candidate remains listed",
            "value": 1.0 if mapping_candidate_listed else 0.0,
            "note": "The target-source inventory still carries the explicit mapping-equation placeholder row.",
        },
        {
            "row_id": "explicit_mapping_equation_lift_required_field_count",
            "status": "inventory",
            "metric": "required mapping row field count",
            "value": float(len(required_mapping_row_fields)),
            "note": f"Required mapping row fields are {required_mapping_row_fields}.",
        },
        {
            "row_id": "explicit_mapping_equation_lift_required_equation_slot_count",
            "status": "inventory",
            "metric": "required mapping-equation slot count",
            "value": float(len(required_mapping_equation_slots)),
            "note": f"Required mapping-equation slots are {required_mapping_equation_slots}.",
        },
        {
            "row_id": "explicit_mapping_equation_lift_forbidden_placeholder_count",
            "status": "inventory",
            "metric": "forbidden placeholder substitution count",
            "value": float(len(forbidden_placeholder_substitutions)),
            "note": f"Forbidden placeholder substitutions are {forbidden_placeholder_substitutions}.",
        },
        {
            "row_id": "explicit_mapping_equation_lift_route_still_admissible",
            "status": "pass" if mapping_route_still_admissible else "reject",
            "metric": "explicit mapping-equation route remains admissible",
            "value": 1.0 if mapping_route_still_admissible else 0.0,
            "note": "The split-source branch still keeps the explicit mapping-equation route alive as an allowed same-sector route.",
        },
        {
            "row_id": "explicit_mapping_equation_lift_supporting_public_rows_present",
            "status": "pass"
            if sign_units_reference_point_present and shell_family_numerical_rows_present and no_new_free_parameter_note_present
            else "reject",
            "metric": "supporting sign/units/reference-point and shell-family rows already present",
            "value": 1.0
            if sign_units_reference_point_present and shell_family_numerical_rows_present and no_new_free_parameter_note_present
            else 0.0,
            "note": "The old promotion stack already keeps sign, units, |P|_* reference point, shell-family numerical rows, and no-new-free-parameter wording available for reuse.",
        },
        {
            "row_id": "explicit_mapping_equation_lift_current_equation_absent",
            "status": "watch" if not explicit_mapping_equation_available_now else "pass",
            "metric": "explicit mapping equation currently still absent",
            "value": 0.0 if not explicit_mapping_equation_available_now else 1.0,
            "note": (
                "The current public canonical pack still has no lifted explicit chi_P -> V''(|P|_*) row."
                if not explicit_mapping_equation_available_now
                else "The explicit mapping equation is already available before the lift audit."
            ),
        },
        {
            "row_id": "explicit_mapping_equation_lift_single_shape_still_downstream",
            "status": "watch" if not single_public_vpp_shape_available_now else "pass",
            "metric": "single public V(|P|) shape remains downstream dependency",
            "value": 0.0 if not single_public_vpp_shape_available_now else 1.0,
            "note": (
                "The lift contract freezes the mapping row specification first; single_public_vpp_shape remains a downstream closure dependency rather than a prerequisite for lift readiness."
                if not single_public_vpp_shape_available_now
                else "single_public_vpp_shape is already available."
            ),
        },
        {
            "row_id": "explicit_mapping_equation_lift_ready",
            "status": "pass" if mapping_equation_lift_ready else "reject",
            "metric": "explicit mapping-equation lift contract ready for audit",
            "value": 1.0 if mapping_equation_lift_ready else 0.0,
            "note": (
                "The next step may audit whether the current public canonical pack already satisfies the frozen explicit mapping-equation lift contract."
                if mapping_equation_lift_ready
                else "The explicit mapping-equation lift prerequisites are not stable enough for audit."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "explicit mapping-equation lift contract",
        },
        "inputs": {
            "mass_origin_same_sector_chi_to_vpp_contract_json": _relative_str(CHI_CONTRACT_JSON),
            "mass_origin_positive_particle_sector_chi_to_vpp_json": _relative_str(PROMOTION_JSON),
            "mass_origin_same_sector_tiebreak_target_source_contract_json": _relative_str(TARGET_SOURCE_CONTRACT_JSON),
            "mass_origin_same_sector_mapping_equation_source_json": _relative_str(MAPPING_SOURCE_AUDIT_JSON),
            "mass_origin_target_source_blocker_split_contract_json": _relative_str(BLOCKER_SPLIT_JSON),
        },
        "intent": "Freeze the minimum contract needed to lift an explicit same-sector chi_P -> V''(|P|_*) mapping row into public canonical form.",
        "formulas": {
            "lift_contract_rule": "mapping_equation_lift_ready iff the same-sector contract, explicit source-kind contract, source inventory listing, supporting public rows, and split-source admissibility all remain available together",
            "placeholder_rule": "the lift may not be satisfied by cross-sector, interface-only, phenomenological, shell-anchor-target, single-shape-placeholder, or undeclared-shell-equivalent substitutions",
        },
        "rows": rows,
        "summary": {
            "required_mapping_row_fields": required_mapping_row_fields,
            "required_mapping_equation_slots": required_mapping_equation_slots,
            "forbidden_placeholder_substitutions": forbidden_placeholder_substitutions,
            "mapping_equation_source_kind_allowed": mapping_equation_source_kind_allowed,
            "mapping_candidate_listed": mapping_candidate_listed,
            "mapping_route_still_admissible": mapping_route_still_admissible,
            "explicit_mapping_equation_available_now": explicit_mapping_equation_available_now,
            "single_public_vpp_shape_available_now": single_public_vpp_shape_available_now,
            "mapping_equation_lift_ready": mapping_equation_lift_ready,
            "tiebreak_invariant_name": target_source_contract_summary.get("tiebreak_invariant_name"),
        },
        "decision": {
            "overall_status": "explicit_mapping_equation_lift_contract_frozen",
            "keep_mass_origin_branch_blocked": True,
            "required_mapping_row_fields": required_mapping_row_fields,
            "required_mapping_equation_slots": required_mapping_equation_slots,
            "forbidden_placeholder_substitutions": forbidden_placeholder_substitutions,
            "mapping_equation_lift_ready": mapping_equation_lift_ready,
        },
        "evidence": {
            "chi_contract_summary": chi_contract_summary,
            "promotion_summary": promotion_summary,
            "target_source_contract_summary": target_source_contract_summary,
            "mapping_source_summary": mapping_source_summary,
            "blocker_split_summary": blocker_split_summary,
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
