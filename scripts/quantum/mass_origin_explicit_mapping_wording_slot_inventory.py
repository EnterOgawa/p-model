#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_explicit_mapping_wording_slot_inventory.py

Step 8.7.55.2.112:
Freeze the required / present / missing wording slots for the explicit
same-sector mapping route after the source-wording blocker split isolates the
explicit mapping branch.

Inputs:
  - output/public/quantum/mass_origin_explicit_mapping_equation_lift_contract_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_literal_fragment_inventory_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_literal_lift_metrics.json
  - output/public/quantum/mass_origin_source_wording_blocker_split_contract_metrics.json

Outputs:
  - output/public/quantum/mass_origin_explicit_mapping_wording_slot_inventory_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_wording_slot_inventory_rows.csv
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
LITERAL_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_literal_fragment_inventory_metrics.json"
LITERAL_LIFT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_literal_lift_metrics.json"
SPLIT_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_source_wording_blocker_split_contract_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_wording_slot_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_wording_slot_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.112"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory required / present / missing wording slots for the explicit mapping route.",
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


# 関数: `_required_wording_slots` の入出力契約と処理意図を定義する。

def _required_wording_slots() -> List[str]:
    return [
        "lhs_observable_chi_P_or_declared_same_sector_shell_equivalent",
        "rhs_curvature_symbol_Vpp_absP_star",
        "reference_point_absP_star",
        "explicit_mapping_equation_literal",
        "same_sector_equivalence_statement_or_none",
        "mapping_operator_or_relation",
    ]


# 関数: `_slot_notes` の入出力契約と処理意図を定義する。

def _slot_notes() -> Dict[str, str]:
    return {
        "lhs_observable_chi_P_or_declared_same_sector_shell_equivalent": (
            "The lhs observable wording slot is already frozen via chi_P or a declared same-sector shell-equivalent observable."
        ),
        "rhs_curvature_symbol_Vpp_absP_star": (
            "The rhs curvature wording slot is already frozen to V''(|P|_*)."
        ),
        "reference_point_absP_star": "The |P|_* reference-point wording slot is already present.",
        "explicit_mapping_equation_literal": (
            "The explicit chi_P -> V''(|P|_*) equation literal itself is still absent from the public canonical pack."
        ),
        "same_sector_equivalence_statement_or_none": (
            "A same-sector equivalence statement linking the observable side to the curvature side is still absent."
        ),
        "mapping_operator_or_relation": (
            "A literal mapping operator / relation between chi_P and V''(|P|_*) is still absent."
        ),
    }


# 関数: `_build_wording_presence` の入出力契約と処理意図を定義する。

def _build_wording_presence(
    equation_slot_presence: Dict[str, Any],
    missing_literal_fragments: List[str],
) -> Dict[str, bool]:
    return {
        "lhs_observable_chi_P_or_declared_same_sector_shell_equivalent": bool(
            equation_slot_presence.get("lhs_observable_chi_P_or_declared_same_sector_shell_equivalent", False)
        ),
        "rhs_curvature_symbol_Vpp_absP_star": bool(
            equation_slot_presence.get("rhs_curvature_symbol_Vpp_absP_star", False)
        ),
        "reference_point_absP_star": bool(
            equation_slot_presence.get("reference_point_absP_star", False)
        ),
        "explicit_mapping_equation_literal": "explicit_mapping_equation_literal" not in missing_literal_fragments,
        "same_sector_equivalence_statement_or_none": bool(
            equation_slot_presence.get("same_sector_equivalence_statement_or_none", False)
        ),
        "mapping_operator_or_relation": bool(
            equation_slot_presence.get("mapping_operator_or_relation", False)
        ),
    }


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (LIFT_CONTRACT_JSON, LITERAL_INVENTORY_JSON, LITERAL_LIFT_JSON, SPLIT_CONTRACT_JSON):
        _require_path(path)

    lift_contract = _read_json(LIFT_CONTRACT_JSON)
    literal_inventory = _read_json(LITERAL_INVENTORY_JSON)
    literal_lift = _read_json(LITERAL_LIFT_JSON)
    split_contract = _read_json(SPLIT_CONTRACT_JSON)

    lift_contract_summary = lift_contract.get("summary", {})
    literal_inventory_summary = literal_inventory.get("summary", {})
    literal_lift_summary = literal_lift.get("summary", {})
    split_contract_summary = split_contract.get("summary", {})
    equation_slot_presence = literal_lift.get("evidence", {}).get("equation_slot_presence", {})

    required_explicit_mapping_wording_slots = _required_wording_slots()
    missing_literal_fragments = [
        str(item) for item in literal_inventory_summary.get("missing_literal_fragments", [])
    ]
    wording_presence = _build_wording_presence(
        equation_slot_presence=equation_slot_presence,
        missing_literal_fragments=missing_literal_fragments,
    )
    present_explicit_mapping_wording_slots = [
        slot_id
        for slot_id in required_explicit_mapping_wording_slots
        if wording_presence.get(slot_id, False)
    ]
    missing_explicit_mapping_wording_slots = [
        slot_id
        for slot_id in required_explicit_mapping_wording_slots
        if not wording_presence.get(slot_id, False)
    ]
    explicit_mapping_wording_route_still_admissible = bool(
        split_contract_summary.get("explicit_mapping_wording_route_still_admissible", False)
    )
    hand_off_to_8_7_55_2_83 = bool(split_contract_summary.get("hand_off_to_8_7_55_2_83", False))
    explicit_mapping_wording_inventory_ready = bool(
        explicit_mapping_wording_route_still_admissible
        and not hand_off_to_8_7_55_2_83
        and len(required_explicit_mapping_wording_slots)
        == len(present_explicit_mapping_wording_slots) + len(missing_explicit_mapping_wording_slots)
    )
    slot_notes = _slot_notes()

    rows: List[Dict[str, Any]] = [
        {
            "row_id": "explicit_mapping_wording_slot_inventory_complete",
            "status": "pass",
            "metric": "explicit mapping wording slot inventory complete",
            "value": 1.0,
            "note": "This step freezes which explicit-mapping wording slots are already present and which remain missing after the source-wording blocker split.",
        },
        {
            "row_id": "explicit_mapping_wording_route_still_admissible",
            "status": "pass" if explicit_mapping_wording_route_still_admissible else "reject",
            "metric": "explicit mapping wording route remains admissible after the blocker split",
            "value": 1.0 if explicit_mapping_wording_route_still_admissible else 0.0,
            "note": (
                "The explicit mapping wording route remains admissible for closure retry."
                if explicit_mapping_wording_route_still_admissible
                else "The explicit mapping wording route is no longer admissible, so the split branch cannot continue."
            ),
        },
        {
            "row_id": "explicit_mapping_wording_slot_required_count",
            "status": "inventory",
            "metric": "required explicit mapping wording slot count",
            "value": float(len(required_explicit_mapping_wording_slots)),
            "note": f"Required explicit mapping wording slots are {required_explicit_mapping_wording_slots}.",
        },
        {
            "row_id": "explicit_mapping_wording_slot_present_count",
            "status": "inventory",
            "metric": "present explicit mapping wording slot count",
            "value": float(len(present_explicit_mapping_wording_slots)),
            "note": f"Present explicit mapping wording slots are {present_explicit_mapping_wording_slots}.",
        },
        {
            "row_id": "explicit_mapping_wording_slot_missing_count",
            "status": "inventory",
            "metric": "missing explicit mapping wording slot count",
            "value": float(len(missing_explicit_mapping_wording_slots)),
            "note": f"Missing explicit mapping wording slots are {missing_explicit_mapping_wording_slots}.",
        },
    ]

    for slot_id in required_explicit_mapping_wording_slots:
        is_present = slot_id in present_explicit_mapping_wording_slots
        rows.append(
            {
                "row_id": f"explicit_mapping_wording_slot_{slot_id}",
                "status": "pass" if is_present else "watch",
                "metric": f"explicit mapping wording slot {slot_id} present in the current public canonical pack",
                "value": 1.0 if is_present else 0.0,
                "note": slot_notes.get(slot_id, ""),
            }
        )

    rows.append(
        {
            "row_id": "explicit_mapping_wording_inventory_ready",
            "status": "pass" if explicit_mapping_wording_inventory_ready else "reject",
            "metric": "explicit mapping wording inventory ready for closure retry",
            "value": 1.0 if explicit_mapping_wording_inventory_ready else 0.0,
            "note": (
                "The next step may attempt explicit mapping wording closure retry using the frozen required/present/missing slot split."
                if explicit_mapping_wording_inventory_ready
                else "The explicit mapping wording slot inventory is not internally complete enough to support closure retry."
            ),
        }
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "explicit mapping wording slot inventory",
        },
        "inputs": {
            "mass_origin_explicit_mapping_equation_lift_contract_json": _relative_str(LIFT_CONTRACT_JSON),
            "mass_origin_explicit_mapping_literal_fragment_inventory_json": _relative_str(LITERAL_INVENTORY_JSON),
            "mass_origin_explicit_mapping_literal_lift_json": _relative_str(LITERAL_LIFT_JSON),
            "mass_origin_source_wording_blocker_split_contract_json": _relative_str(SPLIT_CONTRACT_JSON),
        },
        "intent": "Freeze the wording-slot partition for the explicit mapping route so literal-lift absence is reduced to wording-level missing slots.",
        "formulas": {
            "wording_inventory_rule": "required_explicit_mapping_wording_slots are partitioned into present_explicit_mapping_wording_slots and missing_explicit_mapping_wording_slots using only the current public canonical pack",
            "closure_readiness_rule": "explicit_mapping_wording_inventory_ready iff the explicit mapping wording route remains admissible after the split and every required wording slot is explicitly inventoried as present or missing",
        },
        "rows": rows,
        "summary": {
            "required_explicit_mapping_wording_slots": required_explicit_mapping_wording_slots,
            "present_explicit_mapping_wording_slots": present_explicit_mapping_wording_slots,
            "missing_explicit_mapping_wording_slots": missing_explicit_mapping_wording_slots,
            "required_explicit_mapping_wording_slot_count": len(required_explicit_mapping_wording_slots),
            "present_explicit_mapping_wording_slot_count": len(present_explicit_mapping_wording_slots),
            "missing_explicit_mapping_wording_slot_count": len(missing_explicit_mapping_wording_slots),
            "explicit_mapping_wording_route_still_admissible": explicit_mapping_wording_route_still_admissible,
            "explicit_mapping_wording_inventory_ready": explicit_mapping_wording_inventory_ready,
            "prior_literal_lift_nonclosure_reason_or_none": literal_lift_summary.get(
                "literal_lift_nonclosure_reason_or_none"
            ),
            "mapping_equation_lift_ready": lift_contract_summary.get("mapping_equation_lift_ready"),
        },
        "decision": {
            "overall_status": "explicit_mapping_wording_slot_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "required_explicit_mapping_wording_slots": required_explicit_mapping_wording_slots,
            "present_explicit_mapping_wording_slots": present_explicit_mapping_wording_slots,
            "missing_explicit_mapping_wording_slots": missing_explicit_mapping_wording_slots,
            "explicit_mapping_wording_inventory_ready": explicit_mapping_wording_inventory_ready,
        },
        "evidence": {
            "explicit_mapping_equation_lift_contract_summary": lift_contract_summary,
            "explicit_mapping_literal_fragment_inventory_summary": literal_inventory_summary,
            "explicit_mapping_literal_lift_summary": literal_lift_summary,
            "source_wording_blocker_split_contract_summary": split_contract_summary,
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
    payload = _build_payload(step_tag=str(args.step_tag))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["rows"])
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
