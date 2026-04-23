#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_anchor_target_synthesis_audit.py

Step 8.7.55.2.99:
Audit whether the currently surviving shell-anchor rows alone can synthesize
the same-sector tie-break target value without introducing new free parameters.

Inputs:
  - output/public/quantum/mass_origin_shell_anchor_target_synthesis_contract_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_inventory_rows.csv
  - output/public/quantum/mass_origin_shell_quantization_canonicalization_rows.csv
  - output/public/quantum/mass_origin_same_sector_vpp_tiebreak_invariant_metrics.json

Outputs:
  - output/public/quantum/mass_origin_shell_anchor_target_synthesis_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_target_synthesis_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_target_synthesis_contract_metrics.json"
SOURCE_INVENTORY_ROWS_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_source_inventory_rows.csv"
SHELL_CANONICAL_ROWS_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_canonicalization_rows.csv"
TIEBREAK_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_vpp_tiebreak_invariant_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_target_synthesis_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_target_synthesis_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.99"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether surviving shell-anchor rows synthesize the same-sector tie-break target value.",
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


# 関数: `_read_csv_by_row_id` の入出力契約と処理意図を定義する。

def _read_csv_by_row_id(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {str(row["row_id"]): {key: str(value) for key, value in row.items()} for row in reader}


# 関数: `_read_source_inventory_by_source_row_id` の入出力契約と処理意図を定義する。

def _read_source_inventory_by_source_row_id(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        inventory_rows: Dict[str, Dict[str, str]] = {}

        for row in reader:
            normalized_row = {key: str(value) for key, value in row.items()}
            source_kind = normalized_row.get("source_kind", "")
            source_row_id = normalized_row.get("source_row_id", "")

            # 条件分岐: `source_kind == "surviving_shell_anchor_pack" and source_row_id` を満たす経路を評価する。
            if source_kind == "surviving_shell_anchor_pack" and source_row_id:
                inventory_rows[source_row_id] = normalized_row

        return inventory_rows


# 関数: `_relative_str` の入出力契約と処理意図を定義する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: `_float_or_none` の入出力契約と処理意図を定義する。

def _float_or_none(value: str | None) -> float | None:
    # 条件分岐: `value is None` を満たす経路を評価する。
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# 関数: `_has_required_fields` の入出力契約と処理意図を定義する。

def _has_required_fields(row: Dict[str, str], required_fields: List[str]) -> bool:
    for field_name in required_fields:
        # 条件分岐: `field_name not in row or row[field_name] == ""` を満たす経路を評価する。
        if field_name not in row or row[field_name] == "":
            return False

    return True


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (CONTRACT_JSON, SOURCE_INVENTORY_ROWS_CSV, SHELL_CANONICAL_ROWS_CSV, TIEBREAK_JSON):
        _require_path(path)

    contract = _read_json(CONTRACT_JSON)
    source_inventory_rows = _read_source_inventory_by_source_row_id(SOURCE_INVENTORY_ROWS_CSV)
    shell_canonical_rows = _read_csv_by_row_id(SHELL_CANONICAL_ROWS_CSV)
    tiebreak = _read_json(TIEBREAK_JSON)

    contract_summary = contract.get("summary", {})
    contract_decision = contract.get("decision", {})
    tiebreak_summary = tiebreak.get("summary", {})

    required_shell_anchor_row_ids = [str(item) for item in contract_summary.get("required_shell_anchor_row_ids", [])]
    required_shell_anchor_row_fields = [str(item) for item in contract_summary.get("required_shell_anchor_row_fields", [])]
    target_synthesis_formula_kind_or_none = contract_summary.get("target_synthesis_formula_kind_or_none")
    tiebreak_invariant_name = str(tiebreak_summary.get("tiebreak_invariant_name"))
    candidate_invariant_values = {
        str(candidate_id): float(value)
        for candidate_id, value in tiebreak_summary.get("surviving_candidate_invariant_values", {}).items()
    }

    inventory_rows_present_count = 0
    inventory_rows_with_required_fields_count = 0
    canonical_numeric_rows_present_count = 0
    shell_anchor_row_values: Dict[str, float] = {}
    missing_inventory_rows: List[str] = []
    missing_canonical_rows: List[str] = []

    for row_id in required_shell_anchor_row_ids:
        inventory_row = source_inventory_rows.get(row_id)
        canonical_row = shell_canonical_rows.get(row_id)

        # 条件分岐: `inventory_row is None` を満たす経路を評価する。
        if inventory_row is None:
            missing_inventory_rows.append(row_id)

        else:
            inventory_rows_present_count += 1

            # 条件分岐: `_has_required_fields(inventory_row, required_shell_anchor_row_fields)` を満たす経路を評価する。
            if _has_required_fields(inventory_row, required_shell_anchor_row_fields):
                inventory_rows_with_required_fields_count += 1

        # 条件分岐: `canonical_row is None` を満たす経路を評価する。

        if canonical_row is None:
            missing_canonical_rows.append(row_id)

        else:
            numeric_value = _float_or_none(canonical_row.get("value"))

            # 条件分岐: `numeric_value is not None` を満たす経路を評価する。
            if numeric_value is not None:
                canonical_numeric_rows_present_count += 1
                shell_anchor_row_values[row_id] = numeric_value

    shell_anchor_pair_complete = (
        inventory_rows_present_count == len(required_shell_anchor_row_ids)
        and inventory_rows_with_required_fields_count == len(required_shell_anchor_row_ids)
        and canonical_numeric_rows_present_count == len(required_shell_anchor_row_ids)
    )
    synthesis_formula_recognized = target_synthesis_formula_kind_or_none == "dimensionless_two_anchor_pair_synthesis"

    # shell-anchor rows currently expose only shell-gap correction amplitudes / ratios.
    # They do not yet publish a same-sector row that identifies either anchor or any
    # algebraic combination of them as the tie-break invariant R3.
    semantic_target_bridge_present = False

    synthesized_target_value_or_none: float | None = None

    # 条件分岐: `shell_anchor_pair_complete and synthesis_formula_recognized and semantic_target_bridge_present` を満たす経路を評価する。
    if shell_anchor_pair_complete and synthesis_formula_recognized and semantic_target_bridge_present:
        synthesized_target_value_or_none = 0.0

    shell_anchor_target_value_available = synthesized_target_value_or_none is not None
    bridge_without_new_free_parameters = bool(
        shell_anchor_target_value_available and contract_decision.get("shell_anchor_target_synthesis_ready", False)
    )
    matching_candidate_ids = [
        candidate_id
        for candidate_id, invariant_value in candidate_invariant_values.items()
        if synthesized_target_value_or_none is not None and invariant_value == synthesized_target_value_or_none
    ]
    candidate_match_count = len(matching_candidate_ids)
    synthesis_nonclosure_reason_or_none = None

    # 条件分岐: `not shell_anchor_target_value_available` を満たす経路を評価する。
    if not shell_anchor_target_value_available:
        synthesis_nonclosure_reason_or_none = "shell_anchor_pair_has_no_public_same_sector_target_bridge"

    rows = [
        {
            "row_id": "shell_anchor_target_synthesis_audit_complete",
            "status": "pass",
            "metric": "shell-anchor target synthesis audit complete",
            "value": 1.0,
            "note": "This step audits whether the frozen shell-anchor row pair can actually synthesize the same-sector tie-break target value.",
        },
        {
            "row_id": "shell_anchor_target_synthesis_required_inventory_rows_present",
            "status": "pass" if inventory_rows_present_count == len(required_shell_anchor_row_ids) else "reject",
            "metric": "required shell-anchor inventory rows present",
            "value": float(inventory_rows_present_count),
            "note": (
                f"Required inventory rows are present for {required_shell_anchor_row_ids}."
                if inventory_rows_present_count == len(required_shell_anchor_row_ids)
                else f"Missing inventory rows are {missing_inventory_rows}."
            ),
        },
        {
            "row_id": "shell_anchor_target_synthesis_required_field_rows_present",
            "status": "pass" if inventory_rows_with_required_fields_count == len(required_shell_anchor_row_ids) else "reject",
            "metric": "required shell-anchor rows satisfy field contract",
            "value": float(inventory_rows_with_required_fields_count),
            "note": (
                f"Required fields are {required_shell_anchor_row_fields}."
                if inventory_rows_with_required_fields_count == len(required_shell_anchor_row_ids)
                else "At least one required shell-anchor inventory row does not satisfy the frozen field contract."
            ),
        },
        {
            "row_id": "shell_anchor_target_synthesis_canonical_numeric_rows_present",
            "status": "pass" if canonical_numeric_rows_present_count == len(required_shell_anchor_row_ids) else "reject",
            "metric": "required shell-anchor canonical numeric rows present",
            "value": float(canonical_numeric_rows_present_count),
            "note": (
                f"Canonical numeric rows exist for {required_shell_anchor_row_ids}."
                if canonical_numeric_rows_present_count == len(required_shell_anchor_row_ids)
                else f"Missing canonical numeric rows are {missing_canonical_rows}."
            ),
        },
        {
            "row_id": "shell_anchor_target_synthesis_formula_recognized",
            "status": "pass" if synthesis_formula_recognized else "reject",
            "metric": "shell-anchor synthesis formula kind recognized",
            "value": 1.0 if synthesis_formula_recognized else 0.0,
            "note": f"Frozen synthesis formula kind is {target_synthesis_formula_kind_or_none}.",
        },
        {
            "row_id": "shell_anchor_target_synthesis_semantic_bridge_present",
            "status": "pass" if semantic_target_bridge_present else "watch",
            "metric": "shell-anchor pair already publishes a same-sector target bridge",
            "value": 1.0 if semantic_target_bridge_present else 0.0,
            "note": (
                f"The shell-anchor pair already closes {tiebreak_invariant_name} in public canonical form."
                if semantic_target_bridge_present
                else f"The shell-anchor rows remain shell-gap anchors only; no public row or note equates them, or any frozen pair synthesis, to {tiebreak_invariant_name}."
            ),
        },
        {
            "row_id": "shell_anchor_target_value_available",
            "status": "pass" if shell_anchor_target_value_available else "watch",
            "metric": "shell-anchor route yields a same-sector target value",
            "value": 1.0 if shell_anchor_target_value_available else 0.0,
            "note": (
                f"Synthesized target value is {synthesized_target_value_or_none}."
                if shell_anchor_target_value_available
                else f"No synthesized target value is available; nonclosure reason is {synthesis_nonclosure_reason_or_none}."
            ),
        },
        {
            "row_id": "shell_anchor_target_synthesis_bridge_without_new_free_parameters",
            "status": "pass" if bridge_without_new_free_parameters else "reject",
            "metric": "shell-anchor route closes without new free parameters",
            "value": 1.0 if bridge_without_new_free_parameters else 0.0,
            "note": (
                "The shell-anchor row pair closes the same-sector target value within the frozen contract."
                if bridge_without_new_free_parameters
                else "The shell-anchor pair does not yet close the target value, so the no-new-free-parameter bridge remains unavailable."
            ),
        },
        {
            "row_id": "shell_anchor_target_synthesis_candidate_match_count",
            "status": "inventory",
            "metric": "candidate count selected by shell-anchor target synthesis",
            "value": float(candidate_match_count),
            "note": (
                f"Matching candidate ids are {matching_candidate_ids}."
                if matching_candidate_ids
                else "No candidate ids are selected because the shell-anchor route still does not synthesize the target value."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "shell-anchor target synthesis audit",
        },
        "inputs": {
            "mass_origin_shell_anchor_target_synthesis_contract_json": _relative_str(CONTRACT_JSON),
            "mass_origin_same_sector_tiebreak_target_source_inventory_rows_csv": _relative_str(SOURCE_INVENTORY_ROWS_CSV),
            "mass_origin_shell_quantization_canonicalization_rows_csv": _relative_str(SHELL_CANONICAL_ROWS_CSV),
            "mass_origin_same_sector_vpp_tiebreak_invariant_json": _relative_str(TIEBREAK_JSON),
        },
        "intent": "Audit whether the frozen shell-anchor row pair actually synthesizes the same-sector target value of the derivative-ratio tie-break invariant.",
        "formulas": {
            "synthesis_rule": "shell_anchor_target_value_available iff the frozen shell-anchor pair both exists and already publishes a same-sector semantic bridge from the pair synthesis to the target invariant",
            "bridge_rule": "bridge_without_new_free_parameters iff the shell-anchor synthesis closes within the frozen no-new-free-parameter contract",
        },
        "rows": rows,
        "summary": {
            "required_shell_anchor_row_ids": required_shell_anchor_row_ids,
            "required_shell_anchor_row_fields": required_shell_anchor_row_fields,
            "shell_anchor_pair_complete": shell_anchor_pair_complete,
            "shell_anchor_row_values": shell_anchor_row_values,
            "shell_anchor_target_value_available": shell_anchor_target_value_available,
            "synthesized_target_value_or_none": synthesized_target_value_or_none,
            "bridge_without_new_free_parameters": bridge_without_new_free_parameters,
            "matching_candidate_ids": matching_candidate_ids,
            "candidate_match_count": candidate_match_count,
            "synthesis_nonclosure_reason_or_none": synthesis_nonclosure_reason_or_none,
            "tiebreak_invariant_name": tiebreak_invariant_name,
        },
        "decision": {
            "overall_status": (
                "shell_anchor_target_synthesis_available"
                if shell_anchor_target_value_available
                else "shell_anchor_target_synthesis_frozen_target_missing"
            ),
            "keep_mass_origin_branch_blocked": not shell_anchor_target_value_available,
            "shell_anchor_target_value_available": shell_anchor_target_value_available,
            "synthesized_target_value_or_none": synthesized_target_value_or_none,
            "bridge_without_new_free_parameters": bridge_without_new_free_parameters,
            "matching_candidate_ids": matching_candidate_ids,
            "candidate_match_count": candidate_match_count,
            "synthesis_nonclosure_reason_or_none": synthesis_nonclosure_reason_or_none,
        },
        "evidence": {
            "contract_summary": contract_summary,
            "shell_anchor_inventory_rows": {
                row_id: source_inventory_rows.get(row_id)
                for row_id in required_shell_anchor_row_ids
            },
            "shell_anchor_canonical_rows": {
                row_id: shell_canonical_rows.get(row_id)
                for row_id in required_shell_anchor_row_ids
            },
            "tiebreak_summary": tiebreak_summary,
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
