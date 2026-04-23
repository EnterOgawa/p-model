#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_same_sector_candidate_admissibility_gate.py

Step 8.7.55.2.79:
Apply the first same-sector admissibility gate to the minimal V(|P|) candidate
families frozen in 8.7.55.2.76-.78.

This step still does not close the branch. It filters the candidate set using
the already-frozen contract and derivative slots:

  - same-sector only
  - shell-family compatibility
  - chi_P -> V''(|P|_*) contract readiness
  - three-wave V'''(|P|_*) contract readiness
  - no-new-free-parameter compliance

Inputs:
  - output/public/quantum/mass_origin_same_sector_chi_to_vpp_contract_metrics.json
  - output/public/quantum/mass_origin_single_vpp_candidate_inventory_metrics.json
  - output/public/quantum/mass_origin_single_vpp_candidate_derivative_metrics.json

Outputs:
  - output/public/quantum/mass_origin_same_sector_candidate_admissibility_metrics.json
  - output/public/quantum/mass_origin_same_sector_candidate_admissibility_rows.csv
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
INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_single_vpp_candidate_inventory_metrics.json"
DERIVATIVE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_single_vpp_candidate_derivative_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_candidate_admissibility_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_candidate_admissibility_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.79"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter admissible same-sector V(|P|) candidates for the mass-origin route.",
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


# 関数: `_candidate_families` の入出力契約と処理意図を定義する。

def _candidate_families() -> List[str]:
    return [
        "mexican_hat",
        "logarithmic",
        "even_polynomial",
    ]


# 関数: `_build_candidate_row` の入出力契約と処理意図を定義する。

def _build_candidate_row(
    family: str,
    coefficient_count: int,
    shell_anchor_count: int,
    contract_summary: Dict[str, Any],
    inventory_summary: Dict[str, Any],
    derivative_summary: Dict[str, Any],
    derivative_row: Dict[str, Any],
) -> Dict[str, Any]:
    same_sector_only = (
        bool(contract_summary.get("same_particle_sector_only", False))
        and bool(inventory_summary.get("same_sector_only", False))
        and bool(derivative_summary.get("same_sector_only", False))
    )
    chi_contract_satisfied = (
        bool(contract_summary.get("chi_to_vpp_mapping_contract_frozen", False))
        and bool(derivative_row.get("same_sector_curvature_slot_available", False))
    )
    three_wave_contract_satisfied = bool(derivative_row.get("three_wave_slot_available", False))
    no_new_free_parameter_required = "no_new_free_parameter_note" in contract_summary.get(
        "required_contract_annotations",
        [],
    )
    no_new_free_parameter_violation = no_new_free_parameter_required and coefficient_count > shell_anchor_count
    shell_family_compatible = (
        bool(contract_summary.get("shell_family_contract_consistent", False))
        and coefficient_count <= shell_anchor_count
    )
    admissible = (
        same_sector_only
        and shell_family_compatible
        and chi_contract_satisfied
        and three_wave_contract_satisfied
        and not no_new_free_parameter_violation
    )

    family_display = family.replace("_", " ")

    # 条件分岐: `admissible` を満たす経路を評価する。
    if admissible:
        note = (
            f"{family_display} keeps a same-sector V'' / V''' slot and can be anchored by the "
            f"{shell_anchor_count} surviving shell-family rows without adding a new free coefficient family."
        )
        status = "survive"
        value = 1.0

    else:
        note = (
            f"{family_display} keeps the derivative slots, but its minimal coefficient count {coefficient_count} "
            f"exceeds the {shell_anchor_count} surviving shell anchors, so the no-new-free-parameter contract would be broken."
        )
        status = "reject"
        value = 0.0

    return {
        "row_id": f"candidate_admissibility_{family}",
        "status": status,
        "family": family,
        "metric": f"same-sector admissibility for {family} family",
        "value": value,
        "same_sector_only": same_sector_only,
        "shell_family_compatible": shell_family_compatible,
        "chi_contract_satisfied": chi_contract_satisfied,
        "three_wave_contract_satisfied": three_wave_contract_satisfied,
        "no_new_free_parameter_violation": no_new_free_parameter_violation,
        "coefficient_count": coefficient_count,
        "shell_anchor_count": shell_anchor_count,
        "admissible": admissible,
        "note": note,
    }


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (CONTRACT_JSON, INVENTORY_JSON, DERIVATIVE_JSON):
        _require_path(path)

    contract = _read_json(CONTRACT_JSON)
    inventory = _read_json(INVENTORY_JSON)
    derivative = _read_json(DERIVATIVE_JSON)

    contract_summary = contract.get("summary", {})
    contract_decision = contract.get("decision", {})
    inventory_summary = inventory.get("summary", {})
    derivative_summary = derivative.get("summary", {})
    inventory_rows = inventory.get("rows", [])
    derivative_rows = derivative.get("rows", [])

    # 条件分岐: `not isinstance(inventory_rows, list)` を満たす経路を評価する。
    if not isinstance(inventory_rows, list):
        raise SystemExit(f"[fail] invalid rows in {INVENTORY_JSON}")

    # 条件分岐: `not isinstance(derivative_rows, list)` を満たす経路を評価する。

    if not isinstance(derivative_rows, list):
        raise SystemExit(f"[fail] invalid rows in {DERIVATIVE_JSON}")

    shell_anchor_count = len(contract_summary.get("existing_shell_family_row_ids", []))
    candidate_rows: List[Dict[str, Any]] = []
    surviving_candidate_ids: List[str] = []
    rejected_candidate_ids: List[str] = []

    for family in _candidate_families():
        inventory_row = _find_row_by_id(inventory_rows, f"candidate_family_{family}")
        derivative_row = _find_row_by_id(derivative_rows, f"candidate_derivative_{family}")
        coefficient_count = int(float(inventory_row.get("value", 0.0)))
        candidate_row = _build_candidate_row(
            family=family,
            coefficient_count=coefficient_count,
            shell_anchor_count=shell_anchor_count,
            contract_summary=contract_summary,
            inventory_summary=inventory_summary,
            derivative_summary=derivative_summary,
            derivative_row=derivative_row,
        )
        candidate_rows.append(candidate_row)

        # 条件分岐: `bool(candidate_row["admissible"])` を満たす経路を評価する。
        if bool(candidate_row["admissible"]):
            surviving_candidate_ids.append(family)

        else:
            rejected_candidate_ids.append(family)

    surviving_candidate_count = len(surviving_candidate_ids)
    single_shape_ready = surviving_candidate_count == 1
    admissibility_nonclosure_reason = (
        "single_candidate_selected"
        if single_shape_ready
        else "two_minimal_same_sector_families_remain"
        if surviving_candidate_count == 2
        else "no_same_sector_candidate_survives"
    )

    rows = candidate_rows + [
        {
            "row_id": "candidate_admissibility_audit_complete",
            "status": "pass",
            "family": "aggregate",
            "metric": "same-sector admissibility audit complete",
            "value": 1.0,
            "same_sector_only": True,
            "shell_family_compatible": None,
            "chi_contract_satisfied": None,
            "three_wave_contract_satisfied": None,
            "no_new_free_parameter_violation": None,
            "coefficient_count": len(candidate_rows),
            "shell_anchor_count": shell_anchor_count,
            "admissible": True,
            "note": "All candidate families were evaluated against the same-sector, shell-family, and no-new-free-parameter gate.",
        },
        {
            "row_id": "candidate_admissibility_surviving_candidate_count",
            "status": "watch" if not single_shape_ready else "pass",
            "family": "aggregate",
            "metric": "surviving admissible candidate count",
            "value": float(surviving_candidate_count),
            "same_sector_only": True,
            "shell_family_compatible": None,
            "chi_contract_satisfied": None,
            "three_wave_contract_satisfied": None,
            "no_new_free_parameter_violation": None,
            "coefficient_count": len(candidate_rows),
            "shell_anchor_count": shell_anchor_count,
            "admissible": single_shape_ready,
            "note": (
                f"Surviving candidates are {surviving_candidate_ids}. "
                "A single public shape is only ready when exactly one family survives."
            ),
        },
        {
            "row_id": "candidate_admissibility_single_shape_ready",
            "status": "pass" if single_shape_ready else "watch",
            "family": "aggregate",
            "metric": "single_public_vpp_shape can be closed from admissibility gate",
            "value": 1.0 if single_shape_ready else 0.0,
            "same_sector_only": True,
            "shell_family_compatible": None,
            "chi_contract_satisfied": None,
            "three_wave_contract_satisfied": None,
            "no_new_free_parameter_violation": None,
            "coefficient_count": len(candidate_rows),
            "shell_anchor_count": shell_anchor_count,
            "admissible": single_shape_ready,
            "note": (
                "The gate remains non-closing because the surviving set is still wider than one family."
                if not single_shape_ready
                else "Exactly one family survives, so the next step may close single_public_vpp_shape."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "same-sector admissibility gate",
        },
        "inputs": {
            "mass_origin_same_sector_chi_to_vpp_contract_json": _relative_str(CONTRACT_JSON),
            "mass_origin_single_vpp_candidate_inventory_json": _relative_str(INVENTORY_JSON),
            "mass_origin_single_vpp_candidate_derivative_json": _relative_str(DERIVATIVE_JSON),
        },
        "intent": "Freeze the surviving same-sector V(|P|) candidate set under shell-family compatibility and no-new-free-parameter constraints.",
        "formulas": {
            "admissibility_rule": "candidate survives iff it stays same-sector, keeps chi_P -> V''(|P|_*) and V'''(|P|_*) slots, and does not require more free coefficients than the surviving shell anchor pack can fix",
            "shell_anchor_rule": "surviving shell-family anchor count = len(existing_shell_family_row_ids) = 2",
            "nonclosure_rule": "single_public_vpp_shape remains unavailable while surviving_candidate_count != 1",
        },
        "rows": rows,
        "summary": {
            "candidate_family_count": len(candidate_rows),
            "candidate_family_ids": _candidate_families(),
            "admissibility_audit_complete": True,
            "shell_anchor_count": shell_anchor_count,
            "surviving_candidate_count": surviving_candidate_count,
            "surviving_candidate_ids": surviving_candidate_ids,
            "rejected_candidate_ids": rejected_candidate_ids,
            "single_shape_ready": single_shape_ready,
            "admissibility_nonclosure_reason": admissibility_nonclosure_reason,
        },
        "decision": {
            "overall_status": "same_sector_candidate_admissibility_gate_frozen",
            "keep_mass_origin_branch_blocked": True,
            "candidate_family_count": len(candidate_rows),
            "admissibility_audit_complete": True,
            "surviving_candidate_count": surviving_candidate_count,
            "surviving_candidate_ids": surviving_candidate_ids,
            "single_shape_ready": single_shape_ready,
            "single_public_vpp_shape_available": single_shape_ready,
            "admissibility_nonclosure_reason": admissibility_nonclosure_reason,
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
            "inventory_summary": inventory_summary,
            "derivative_summary": derivative_summary,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_id",
                "status",
                "family",
                "metric",
                "value",
                "same_sector_only",
                "shell_family_compatible",
                "chi_contract_satisfied",
                "three_wave_contract_satisfied",
                "no_new_free_parameter_violation",
                "coefficient_count",
                "shell_anchor_count",
                "admissible",
                "note",
            ],
        )
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
