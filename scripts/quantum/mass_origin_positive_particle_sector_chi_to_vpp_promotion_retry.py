#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_positive_particle_sector_chi_to_vpp_promotion_retry.py

Step 8.7.55.2.88:
Retry promotion of the positive-particle-sector chi_P -> V''(|P|_*) public
artifact after the tie-break bridge and single-shape closure retry.

Inputs:
  - output/public/quantum/mass_origin_positive_particle_sector_chi_to_vpp_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_bridge_metrics.json
  - output/public/quantum/mass_origin_single_public_vpp_shape_closure_retry_metrics.json

Outputs:
  - output/public/quantum/mass_origin_positive_particle_sector_chi_to_vpp_retry_metrics.json
  - output/public/quantum/mass_origin_positive_particle_sector_chi_to_vpp_retry_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

PROMOTION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_positive_particle_sector_chi_to_vpp_metrics.json"
BRIDGE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_bridge_metrics.json"
CLOSURE_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_single_public_vpp_shape_closure_retry_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_positive_particle_sector_chi_to_vpp_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_positive_particle_sector_chi_to_vpp_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.88"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry promotion of the positive-particle-sector chi_P -> V''(|P|_*) public artifact.",
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


# 関数: `_promotion_retry_status` の入出力契約と処理意図を定義する。

def _promotion_retry_status(
    positive_particle_sector_chi_p_to_vpp_public_artifact_available: bool,
    explicit_mapping_equation_present: bool,
    single_public_vpp_shape_available: bool,
    target_value_available: bool,
) -> str:
    # 条件分岐: `positive_particle_sector_chi_p_to_vpp_public_artifact_available` を満たす経路を評価する。
    if positive_particle_sector_chi_p_to_vpp_public_artifact_available:
        return "promoted_positive_particle_sector_chi_to_vpp_public_artifact_after_retry"

    # 条件分岐: `not explicit_mapping_equation_present and not target_value_available` を満たす経路を評価する。

    if not explicit_mapping_equation_present and not target_value_available:
        return "watch_nonpromoted_mapping_equation_absent_and_tiebreak_target_value_missing"

    # 条件分岐: `not explicit_mapping_equation_present and not single_public_vpp_shape_available` を満たす経路を評価する。

    if not explicit_mapping_equation_present and not single_public_vpp_shape_available:
        return "watch_nonpromoted_mapping_equation_absent_and_single_shape_retry_nonclosing"

    # 条件分岐: `not explicit_mapping_equation_present` を満たす経路を評価する。

    if not explicit_mapping_equation_present:
        return "watch_nonpromoted_mapping_equation_absent"

    # 条件分岐: `not target_value_available` を満たす経路を評価する。

    if not target_value_available:
        return "watch_nonpromoted_tiebreak_target_value_missing"

    # 条件分岐: `not single_public_vpp_shape_available` を満たす経路を評価する。

    if not single_public_vpp_shape_available:
        return "watch_nonpromoted_single_shape_retry_nonclosing"

    return "watch_nonpromoted_retry_requirements_missing"


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (PROMOTION_JSON, BRIDGE_JSON, CLOSURE_RETRY_JSON):
        _require_path(path)

    promotion = _read_json(PROMOTION_JSON)
    bridge = _read_json(BRIDGE_JSON)
    closure_retry = _read_json(CLOSURE_RETRY_JSON)

    promotion_rows = promotion.get("rows", [])
    promotion_summary = promotion.get("summary", {})
    promotion_decision = promotion.get("decision", {})
    bridge_summary = bridge.get("summary", {})
    bridge_decision = bridge.get("decision", {})
    closure_retry_summary = closure_retry.get("summary", {})
    closure_retry_decision = closure_retry.get("decision", {})

    # 条件分岐: `not isinstance(promotion_rows, list)` を満たす経路を評価する。
    if not isinstance(promotion_rows, list):
        raise SystemExit(f"[fail] invalid rows in {PROMOTION_JSON}")

    same_particle_sector_row = _find_row_by_id(
        promotion_rows,
        "positive_particle_sector_same_particle_sector_only",
    )
    sign_units_row = _find_row_by_id(
        promotion_rows,
        "positive_particle_sector_sign_units_reference_point",
    )
    shell_family_row = _find_row_by_id(
        promotion_rows,
        "positive_particle_sector_public_shell_family_numerical_rows",
    )
    no_new_parameter_row = _find_row_by_id(
        promotion_rows,
        "positive_particle_sector_no_new_free_parameter_note",
    )

    same_particle_sector_only = str(same_particle_sector_row.get("status", "")) == "pass"
    explicit_mapping_equation_present = bool(promotion_summary.get("explicit_mapping_equation_present", False))
    sign_units_reference_point_present = bool(promotion_summary.get("sign_units_reference_point_present", False))
    shell_family_numerical_rows_present = bool(promotion_summary.get("shell_family_numerical_rows_present", False))
    no_new_free_parameter_note_present = bool(promotion_summary.get("no_new_free_parameter_note_present", False))
    same_sector_curvature_slot_ready = bool(promotion_summary.get("same_sector_curvature_slot_ready", False))
    three_wave_slot_ready = bool(promotion_summary.get("three_wave_slot_ready", False))
    target_value_available = bool(bridge_summary.get("target_value_available", False))
    target_source_kind_or_none = bridge_summary.get("target_source_kind_or_none")
    bridge_without_new_free_parameters = bool(bridge_summary.get("bridge_without_new_free_parameters", False))
    single_public_vpp_shape_available = bool(closure_retry_summary.get("single_public_vpp_shape_available", False))
    selected_candidate_id_or_none = closure_retry_summary.get("selected_candidate_id_or_none")
    closure_retry_status = str(closure_retry_summary.get("closure_retry_status", ""))
    retry_nonclosure_reason_or_none = closure_retry_summary.get("nonclosure_reason_or_none")

    missing_promotion_requirements: List[str] = []

    # 条件分岐: `not explicit_mapping_equation_present` を満たす経路を評価する。
    if not explicit_mapping_equation_present:
        missing_promotion_requirements.append("explicit_mapping_equation")

    # 条件分岐: `not sign_units_reference_point_present` を満たす経路を評価する。

    if not sign_units_reference_point_present:
        missing_promotion_requirements.append("sign_units_reference_point")

    # 条件分岐: `not shell_family_numerical_rows_present` を満たす経路を評価する。

    if not shell_family_numerical_rows_present:
        missing_promotion_requirements.append("public_shell_family_numerical_rows")

    # 条件分岐: `not no_new_free_parameter_note_present` を満たす経路を評価する。

    if not no_new_free_parameter_note_present:
        missing_promotion_requirements.append("no_new_free_parameter_note")

    # 条件分岐: `not same_sector_curvature_slot_ready` を満たす経路を評価する。

    if not same_sector_curvature_slot_ready:
        missing_promotion_requirements.append("same_sector_curvature_slot")

    # 条件分岐: `not three_wave_slot_ready` を満たす経路を評価する。

    if not three_wave_slot_ready:
        missing_promotion_requirements.append("three_wave_slot")

    # 条件分岐: `not target_value_available` を満たす経路を評価する。

    if not target_value_available:
        missing_promotion_requirements.append("same_sector_tiebreak_target_value")

    # 条件分岐: `not single_public_vpp_shape_available` を満たす経路を評価する。

    if not single_public_vpp_shape_available:
        missing_promotion_requirements.append("single_public_vpp_shape")

    positive_particle_sector_chi_p_to_vpp_public_artifact_available = (
        same_particle_sector_only and not missing_promotion_requirements
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
    promotion_retry_status = _promotion_retry_status(
        positive_particle_sector_chi_p_to_vpp_public_artifact_available,
        explicit_mapping_equation_present,
        single_public_vpp_shape_available,
        target_value_available,
    )

    rows = [
        {
            "row_id": "positive_particle_sector_retry_same_particle_sector_only",
            "status": "pass" if same_particle_sector_only else "reject",
            "metric": "retry promotion stays inside positive particle sector only",
            "value": 1.0 if same_particle_sector_only else 0.0,
            "note": str(same_particle_sector_row.get("note", "")),
        },
        {
            "row_id": "positive_particle_sector_retry_explicit_mapping_equation",
            "status": "pass" if explicit_mapping_equation_present else "missing",
            "metric": "explicit chi_P -> V''(|P|_*) mapping equation present after retry",
            "value": 1.0 if explicit_mapping_equation_present else 0.0,
            "note": (
                "The retry branch still cannot promote the named artifact because the explicit mapping equation is absent."
                if not explicit_mapping_equation_present
                else "The explicit mapping equation is present and can support retry promotion."
            ),
        },
        {
            "row_id": "positive_particle_sector_retry_sign_units_reference_point",
            "status": "pass" if sign_units_reference_point_present else "reject",
            "metric": "sign, units, and |P|_* reference point remain present after retry",
            "value": 1.0 if sign_units_reference_point_present else 0.0,
            "note": str(sign_units_row.get("note", "")),
        },
        {
            "row_id": "positive_particle_sector_retry_public_shell_family_rows",
            "status": "pass" if shell_family_numerical_rows_present else "reject",
            "metric": "surviving shell-family numerical rows remain present after retry",
            "value": float(shell_family_row.get("value", 0.0)) if shell_family_numerical_rows_present else 0.0,
            "note": str(shell_family_row.get("note", "")),
        },
        {
            "row_id": "positive_particle_sector_retry_no_new_free_parameter_note",
            "status": "pass" if no_new_free_parameter_note_present else "reject",
            "metric": "no-new-free-parameter note remains present after retry",
            "value": 1.0 if no_new_free_parameter_note_present else 0.0,
            "note": str(no_new_parameter_row.get("note", "")),
        },
        {
            "row_id": "positive_particle_sector_retry_tiebreak_target_value_available",
            "status": "pass" if target_value_available else "watch",
            "metric": "same-sector tie-break target value available for retry promotion",
            "value": 1.0 if target_value_available else 0.0,
            "note": (
                f"Target source kind is {target_source_kind_or_none}."
                if target_value_available
                else "The retry branch still lacks a public canonical target value for the derivative-ratio discriminant."
            ),
        },
        {
            "row_id": "positive_particle_sector_retry_bridge_without_new_free_parameters",
            "status": "pass" if bridge_without_new_free_parameters else "reject",
            "metric": "retry promotion bridge closes without new free parameters",
            "value": 1.0 if bridge_without_new_free_parameters else 0.0,
            "note": (
                "The retry promotion uses only already-frozen same-sector ingredients."
                if bridge_without_new_free_parameters
                else "The retry promotion cannot close the tie-break bridge without new free parameters because the target value is still missing."
            ),
        },
        {
            "row_id": "positive_particle_sector_retry_single_public_vpp_shape_available",
            "status": "pass" if single_public_vpp_shape_available else "watch",
            "metric": "single public V(|P|) shape available after retry",
            "value": 1.0 if single_public_vpp_shape_available else 0.0,
            "note": (
                f"Single shape is fixed to {selected_candidate_id_or_none}."
                if single_public_vpp_shape_available
                else f"Single-shape retry is still open: {retry_nonclosure_reason_or_none}."
            ),
        },
        {
            "row_id": "positive_particle_sector_retry_public_artifact_available",
            "status": "pass" if positive_particle_sector_chi_p_to_vpp_public_artifact_available else "watch",
            "metric": "named positive-particle-sector chi_P -> V''(|P|_*) public artifact available after retry",
            "value": 1.0 if positive_particle_sector_chi_p_to_vpp_public_artifact_available else 0.0,
            "note": (
                "Retry promotion is complete."
                if positive_particle_sector_chi_p_to_vpp_public_artifact_available
                else f"Retry promotion stays non-closing because the remaining missing requirements are {missing_promotion_requirements}."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "positive_particle_sector_chi_p_to_vpp_public_artifact promotion retry",
        },
        "inputs": {
            "mass_origin_positive_particle_sector_chi_to_vpp_json": _relative_str(PROMOTION_JSON),
            "mass_origin_same_sector_tiebreak_target_bridge_json": _relative_str(BRIDGE_JSON),
            "mass_origin_single_public_vpp_shape_closure_retry_json": _relative_str(CLOSURE_RETRY_JSON),
        },
        "intent": "Retry promotion of the named same-sector chi_P -> V''(|P|_*) artifact after the tie-break branch updates.",
        "formulas": {
            "promotion_retry_rule": "artifact available iff explicit mapping equation, sign/units/reference point, shell-family numerical rows, no-new-free-parameter note, same-sector V'' and V''' slots, same_sector_tiebreak_target_value, and single_public_vpp_shape are all simultaneously present",
            "retry_dependency": "the retry branch narrows the promotion blocker from a generic two-family ambiguity to the absence of a public canonical tie-break target value",
        },
        "rows": rows,
        "summary": {
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": positive_particle_sector_chi_p_to_vpp_public_artifact_available,
            "explicit_mapping_equation_present": explicit_mapping_equation_present,
            "single_public_vpp_shape_available": single_public_vpp_shape_available,
            "promotion_retry_status": promotion_retry_status,
            "public_row_count": public_row_count,
            "present_public_row_count": present_public_row_count,
            "target_value_available": target_value_available,
            "target_source_kind_or_none": target_source_kind_or_none,
            "bridge_without_new_free_parameters": bridge_without_new_free_parameters,
            "closure_retry_status": closure_retry_status,
            "missing_promotion_requirements": missing_promotion_requirements,
            "selected_candidate_id_or_none": selected_candidate_id_or_none,
        },
        "decision": {
            "overall_status": "positive_particle_sector_chi_to_vpp_promotion_retry_frozen",
            "keep_mass_origin_branch_blocked": True,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": positive_particle_sector_chi_p_to_vpp_public_artifact_available,
            "explicit_mapping_equation_present": explicit_mapping_equation_present,
            "single_public_vpp_shape_available": single_public_vpp_shape_available,
            "promotion_retry_status": promotion_retry_status,
            "blocked_state_detail": str(bridge_decision.get("blocked_state_detail", promotion_decision.get("blocked_state_detail", closure_retry_decision.get("blocked_state_detail", "")))),
            "next_required_artifacts": bridge_decision.get(
                "next_required_artifacts",
                [
                    "same_sector_tiebreak_target_value",
                    "single_public_vpp_shape",
                    "positive_particle_sector_chi_p_to_vpp_public_artifact",
                    "solver_ready_row_promoted_to_pass",
                ],
            ),
        },
        "evidence": {
            "promotion_summary": promotion_summary,
            "bridge_summary": bridge_summary,
            "closure_retry_summary": closure_retry_summary,
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
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["rows"])
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
