#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_positive_particle_sector_chi_to_vpp_promotion.py

Step 8.7.55.2.81:
Freeze whether the missing positive-particle-sector chi_P -> V''(|P|_*)
artifact can be promoted into public canonical form from the currently frozen
same-sector contract, derivative slots, and single-shape closure state.

Inputs:
  - output/public/quantum/mass_origin_same_sector_chi_to_vpp_contract_metrics.json
  - output/public/quantum/mass_origin_single_vpp_candidate_derivative_metrics.json
  - output/public/quantum/mass_origin_single_public_vpp_shape_closure_metrics.json

Outputs:
  - output/public/quantum/mass_origin_positive_particle_sector_chi_to_vpp_metrics.json
  - output/public/quantum/mass_origin_positive_particle_sector_chi_to_vpp_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_chi_to_vpp_contract_metrics.json"
DERIVATIVE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_single_vpp_candidate_derivative_metrics.json"
CLOSURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_single_public_vpp_shape_closure_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_positive_particle_sector_chi_to_vpp_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_positive_particle_sector_chi_to_vpp_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.81"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze promotion status for the positive-particle-sector chi_P -> V''(|P|_*) public artifact.",
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


# 関数: `_promotion_status` の入出力契約と処理意図を定義する。

def _promotion_status(missing_requirements: List[str]) -> str:
    missing_set = set(missing_requirements)

    # 条件分岐: `not missing_requirements` を満たす経路を評価する。
    if not missing_requirements:
        return "promoted_positive_particle_sector_chi_to_vpp_public_artifact"

    # 条件分岐: `missing_set == {"explicit_mapping_equation", "single_public_vpp_shape"}` を満たす経路を評価する。

    if missing_set == {"explicit_mapping_equation", "single_public_vpp_shape"}:
        return "watch_nonpromoted_mapping_equation_absent_and_single_shape_nonclosing"

    # 条件分岐: `missing_set == {"explicit_mapping_equation"}` を満たす経路を評価する。

    if missing_set == {"explicit_mapping_equation"}:
        return "watch_nonpromoted_mapping_equation_absent"

    # 条件分岐: `missing_set == {"single_public_vpp_shape"}` を満たす経路を評価する。

    if missing_set == {"single_public_vpp_shape"}:
        return "watch_nonpromoted_single_shape_nonclosing"

    return "watch_nonpromoted_required_fields_missing"


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (CONTRACT_JSON, DERIVATIVE_JSON, CLOSURE_JSON):
        _require_path(path)

    contract = _read_json(CONTRACT_JSON)
    derivative = _read_json(DERIVATIVE_JSON)
    closure = _read_json(CLOSURE_JSON)

    contract_rows = contract.get("rows", [])
    contract_summary = contract.get("summary", {})
    contract_decision = contract.get("decision", {})
    derivative_summary = derivative.get("summary", {})
    closure_summary = closure.get("summary", {})
    closure_decision = closure.get("decision", {})

    # 条件分岐: `not isinstance(contract_rows, list)` を満たす経路を評価する。
    if not isinstance(contract_rows, list):
        raise SystemExit(f"[fail] invalid rows in {CONTRACT_JSON}")

    mapping_row = _find_row_by_id(contract_rows, "contract_explicit_mapping_equation_still_missing")
    shell_rows_row = _find_row_by_id(contract_rows, "contract_public_shell_family_rows_required")
    no_new_parameter_row = _find_row_by_id(contract_rows, "contract_no_new_free_parameter_note_required")

    same_particle_sector_only = bool(contract_summary.get("same_particle_sector_only", False))
    explicit_mapping_equation_present = bool(contract_summary.get("explicit_mapping_equation_available", False))
    sign_units_reference_point_present = all(
        [
            bool(contract_summary.get("sign_convention_frozen", False)),
            bool(contract_summary.get("unit_contract_frozen", False)),
            bool(str(contract_summary.get("reference_point_symbol", ""))),
        ]
    )
    shell_family_row_ids = [str(item) for item in contract_summary.get("existing_shell_family_row_ids", [])]
    shell_family_numerical_rows_present = bool(contract_summary.get("shell_family_contract_consistent", False)) and bool(
        shell_family_row_ids
    )
    no_new_free_parameter_note_present = "no_new_free_parameter_note" in [
        str(item) for item in contract_summary.get("required_contract_annotations", [])
    ]
    same_sector_curvature_slot_ready = bool(derivative_summary.get("same_sector_curvature_slot_ready", False))
    three_wave_slot_ready = bool(derivative_summary.get("three_wave_slot_ready", False))
    single_public_vpp_shape_available = bool(closure_summary.get("single_public_vpp_shape_available", False))
    selected_candidate_id_or_none = closure_summary.get("selected_candidate_id_or_none")

    missing_requirements: List[str] = []

    # 条件分岐: `not explicit_mapping_equation_present` を満たす経路を評価する。
    if not explicit_mapping_equation_present:
        missing_requirements.append("explicit_mapping_equation")

    # 条件分岐: `not sign_units_reference_point_present` を満たす経路を評価する。

    if not sign_units_reference_point_present:
        missing_requirements.append("sign_units_reference_point")

    # 条件分岐: `not shell_family_numerical_rows_present` を満たす経路を評価する。

    if not shell_family_numerical_rows_present:
        missing_requirements.append("public_shell_family_numerical_rows")

    # 条件分岐: `not no_new_free_parameter_note_present` を満たす経路を評価する。

    if not no_new_free_parameter_note_present:
        missing_requirements.append("no_new_free_parameter_note")

    # 条件分岐: `not same_sector_curvature_slot_ready` を満たす経路を評価する。

    if not same_sector_curvature_slot_ready:
        missing_requirements.append("same_sector_curvature_slot")

    # 条件分岐: `not three_wave_slot_ready` を満たす経路を評価する。

    if not three_wave_slot_ready:
        missing_requirements.append("three_wave_slot")

    # 条件分岐: `not single_public_vpp_shape_available` を満たす経路を評価する。

    if not single_public_vpp_shape_available:
        missing_requirements.append("single_public_vpp_shape")

    positive_particle_sector_chi_p_to_vpp_public_artifact_available = (
        same_particle_sector_only and not missing_requirements
    )
    public_row_count = 4
    present_public_row_count = sum(
        1
        for value in (
            explicit_mapping_equation_present,
            sign_units_reference_point_present,
            shell_family_numerical_rows_present,
            no_new_free_parameter_note_present,
        )
        if value
    )
    promotion_status = _promotion_status(missing_requirements)

    rows = [
        {
            "row_id": "positive_particle_sector_same_particle_sector_only",
            "status": "pass" if same_particle_sector_only else "reject",
            "metric": "promotion stays inside positive particle sector only",
            "value": 1.0 if same_particle_sector_only else 0.0,
            "note": "The named artifact remains constrained to the same particle sector and may not be satisfied by cross-sector or interface-only substitutes.",
        },
        {
            "row_id": "positive_particle_sector_explicit_mapping_equation",
            "status": "pass" if explicit_mapping_equation_present else "missing",
            "metric": "explicit chi_P -> V''(|P|_*) mapping equation present",
            "value": 1.0 if explicit_mapping_equation_present else 0.0,
            "note": str(mapping_row.get("note", "")),
        },
        {
            "row_id": "positive_particle_sector_sign_units_reference_point",
            "status": "pass" if sign_units_reference_point_present else "reject",
            "metric": "sign, units, and |P|_* reference point present together",
            "value": 1.0 if sign_units_reference_point_present else 0.0,
            "note": (
                "The contract still freezes sign, units, and reference point together; the future public rows must expose them alongside the mapping."
            ),
        },
        {
            "row_id": "positive_particle_sector_public_shell_family_numerical_rows",
            "status": "pass" if shell_family_numerical_rows_present else "reject",
            "metric": "surviving shell-family numerical rows available for promotion",
            "value": float(len(shell_family_row_ids)) if shell_family_numerical_rows_present else 0.0,
            "note": str(shell_rows_row.get("note", "")),
        },
        {
            "row_id": "positive_particle_sector_no_new_free_parameter_note",
            "status": "pass" if no_new_free_parameter_note_present else "reject",
            "metric": "no-new-free-parameter note available for promotion",
            "value": 1.0 if no_new_free_parameter_note_present else 0.0,
            "note": str(no_new_parameter_row.get("note", "")),
        },
        {
            "row_id": "positive_particle_sector_same_sector_curvature_slot_ready",
            "status": "pass" if same_sector_curvature_slot_ready else "reject",
            "metric": "same-sector V''(|P|_*) slot ready across candidate audit",
            "value": 1.0 if same_sector_curvature_slot_ready else 0.0,
            "note": "Derivative audit keeps a same-sector curvature slot available for the surviving candidate families.",
        },
        {
            "row_id": "positive_particle_sector_three_wave_slot_ready",
            "status": "pass" if three_wave_slot_ready else "reject",
            "metric": "same-sector V'''(|P|_*) slot ready across candidate audit",
            "value": 1.0 if three_wave_slot_ready else 0.0,
            "note": "Derivative audit keeps the three-wave derivative slot available for the surviving candidate families.",
        },
        {
            "row_id": "positive_particle_sector_single_public_vpp_shape_available",
            "status": "pass" if single_public_vpp_shape_available else "watch",
            "metric": "single public V(|P|) shape already available",
            "value": 1.0 if single_public_vpp_shape_available else 0.0,
            "note": (
                f"Single shape is fixed to {selected_candidate_id_or_none}."
                if single_public_vpp_shape_available
                else f"Single-shape closure is still open: {closure_summary.get('nonclosure_reason_or_none')}."
            ),
        },
        {
            "row_id": "positive_particle_sector_public_artifact_available",
            "status": "pass" if positive_particle_sector_chi_p_to_vpp_public_artifact_available else "watch",
            "metric": "named positive-particle-sector chi_P -> V''(|P|_*) public artifact available",
            "value": 1.0 if positive_particle_sector_chi_p_to_vpp_public_artifact_available else 0.0,
            "note": (
                "Promotion is complete."
                if positive_particle_sector_chi_p_to_vpp_public_artifact_available
                else f"Promotion stays non-closing because the remaining missing requirements are {missing_requirements}."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "positive_particle_sector_chi_p_to_vpp_public_artifact promotion",
        },
        "inputs": {
            "mass_origin_same_sector_chi_to_vpp_contract_json": _relative_str(CONTRACT_JSON),
            "mass_origin_single_vpp_candidate_derivative_json": _relative_str(DERIVATIVE_JSON),
            "mass_origin_single_public_vpp_shape_closure_json": _relative_str(CLOSURE_JSON),
        },
        "intent": "Freeze whether the named same-sector chi_P -> V''(|P|_*) artifact can already be promoted into public canonical form.",
        "formulas": {
            "promotion_rule": "artifact available iff explicit mapping equation, sign/units/reference point, shell-family numerical rows, no-new-free-parameter note, same-sector V'' and V''' slots, and single_public_vpp_shape are all simultaneously present",
            "current_dependency": "single_public_vpp_shape remains part of promotion because the public mapping must land on one same-sector V''(|P|_*) target rather than a two-family ambiguity",
        },
        "rows": rows,
        "summary": {
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": positive_particle_sector_chi_p_to_vpp_public_artifact_available,
            "public_row_count": public_row_count,
            "present_public_row_count": present_public_row_count,
            "explicit_mapping_equation_present": explicit_mapping_equation_present,
            "sign_units_reference_point_present": sign_units_reference_point_present,
            "shell_family_numerical_rows_present": shell_family_numerical_rows_present,
            "no_new_free_parameter_note_present": no_new_free_parameter_note_present,
            "same_sector_curvature_slot_ready": same_sector_curvature_slot_ready,
            "three_wave_slot_ready": three_wave_slot_ready,
            "single_public_vpp_shape_available": single_public_vpp_shape_available,
            "selected_candidate_id_or_none": selected_candidate_id_or_none,
            "missing_promotion_requirements": missing_requirements,
            "promotion_status": promotion_status,
        },
        "decision": {
            "overall_status": "positive_particle_sector_chi_to_vpp_promotion_frozen",
            "keep_mass_origin_branch_blocked": True,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": positive_particle_sector_chi_p_to_vpp_public_artifact_available,
            "public_row_count": public_row_count,
            "present_public_row_count": present_public_row_count,
            "explicit_mapping_equation_present": explicit_mapping_equation_present,
            "shell_family_numerical_rows_present": shell_family_numerical_rows_present,
            "promotion_status": promotion_status,
            "blocked_state_detail": str(contract_decision.get("blocked_state_detail", "")),
            "next_required_artifacts": contract_decision.get(
                "next_required_artifacts",
                [
                    "positive_particle_sector_chi_p_to_vpp_public_artifact",
                    "single_public_vpp_shape",
                    "solver_ready_row_promoted_to_pass",
                ],
            ),
        },
        "evidence": {
            "contract_summary": contract_summary,
            "derivative_summary": derivative_summary,
            "closure_summary": closure_summary,
            "closure_decision": closure_decision,
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
