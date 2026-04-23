#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_explicit_mapping_literal_lift_closure.py

Step 8.7.55.2.106:
Retry explicit mapping-equation lift closure after freezing the required /
present / missing literal fragments for the same-sector chi_P -> V''(|P|_*)
route.

Inputs:
  - output/public/quantum/mass_origin_explicit_mapping_equation_lift_contract_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_equation_lift_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_literal_fragment_inventory_metrics.json

Outputs:
  - output/public/quantum/mass_origin_explicit_mapping_literal_lift_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_literal_lift_rows.csv
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
LIFT_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_equation_lift_metrics.json"
LITERAL_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_literal_fragment_inventory_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_literal_lift_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_literal_lift_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.106"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Close or non-close explicit mapping-equation literal lift after fragment inventory.",
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


# 関数: `_same_sector_relation_fragments` の入出力契約と処理意図を定義する。

def _same_sector_relation_fragments() -> List[str]:
    return [
        "same_sector_equivalence_statement",
        "mapping_operator_or_relation",
    ]


# 関数: `_literal_lift_nonclosure_reason` の入出力契約と処理意図を定義する。

def _literal_lift_nonclosure_reason(
    missing_literal_fragments: List[str],
    missing_lift_requirements: List[str],
) -> str | None:
    relation_fragments = _same_sector_relation_fragments()
    missing_relation_fragments = [
        fragment_id for fragment_id in relation_fragments if fragment_id in missing_literal_fragments
    ]
    explicit_mapping_equation_literal_missing = "explicit_mapping_equation_literal" in missing_literal_fragments

    # 条件分岐: `not missing_literal_fragments and not missing_lift_requirements` を満たす経路を評価する。
    if not missing_literal_fragments and not missing_lift_requirements:
        return None

    # 条件分岐: `explicit_mapping_equation_literal_missing and missing_relation_fragments` を満たす経路を評価する。

    if explicit_mapping_equation_literal_missing and missing_relation_fragments:
        return "explicit_mapping_equation_literal_and_same_sector_relation_wording_absent"

    # 条件分岐: `explicit_mapping_equation_literal_missing` を満たす経路を評価する。

    if explicit_mapping_equation_literal_missing:
        return "explicit_mapping_equation_literal_absent"

    # 条件分岐: `missing_relation_fragments` を満たす経路を評価する。

    if missing_relation_fragments:
        return "same_sector_relation_wording_absent"

    # 条件分岐: `missing_lift_requirements` を満たす経路を評価する。

    if missing_lift_requirements:
        return "explicit_mapping_equation_lift_requirements_unsatisfied"

    return "explicit_mapping_literal_fragments_missing"


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (LIFT_CONTRACT_JSON, LIFT_AUDIT_JSON, LITERAL_INVENTORY_JSON):
        _require_path(path)

    lift_contract = _read_json(LIFT_CONTRACT_JSON)
    lift_audit = _read_json(LIFT_AUDIT_JSON)
    literal_inventory = _read_json(LITERAL_INVENTORY_JSON)

    lift_contract_summary = lift_contract.get("summary", {})
    lift_audit_summary = lift_audit.get("summary", {})
    lift_audit_evidence = lift_audit.get("evidence", {})
    literal_inventory_summary = literal_inventory.get("summary", {})
    literal_inventory_decision = literal_inventory.get("decision", {})

    required_literal_fragments = [
        str(item) for item in literal_inventory_summary.get("required_literal_fragments", [])
    ]
    present_literal_fragments = [
        str(item) for item in literal_inventory_summary.get("present_literal_fragments", [])
    ]
    missing_literal_fragments = [
        str(item) for item in literal_inventory_summary.get("missing_literal_fragments", [])
    ]
    missing_lift_requirements = [
        str(item) for item in lift_audit_summary.get("missing_lift_requirements", [])
    ]
    relation_fragments = _same_sector_relation_fragments()
    missing_relation_fragments = [
        fragment_id for fragment_id in relation_fragments if fragment_id in missing_literal_fragments
    ]
    explicit_mapping_equation_literal_missing = "explicit_mapping_equation_literal" in missing_literal_fragments
    literal_fragment_inventory_ready = bool(literal_inventory_summary.get("literal_fragment_inventory_ready", False))
    mapping_equation_lift_ready = bool(lift_contract_summary.get("mapping_equation_lift_ready", False))

    explicit_mapping_equation_available = bool(
        lift_audit_summary.get("explicit_mapping_equation_available", False)
        and literal_fragment_inventory_ready
        and not missing_literal_fragments
    )
    lifted_mapping_equation_kind_or_none = (
        "explicit_same_sector_chi_to_vpp_row" if explicit_mapping_equation_available else None
    )
    mapping_without_new_free_parameters = bool(
        explicit_mapping_equation_available
        and lift_audit_summary.get("mapping_without_new_free_parameters", False)
    )
    literal_lift_nonclosure_reason_or_none = _literal_lift_nonclosure_reason(
        missing_literal_fragments=missing_literal_fragments,
        missing_lift_requirements=missing_lift_requirements,
    )

    rows = [
        {
            "row_id": "explicit_mapping_literal_lift_closure_complete",
            "status": "pass",
            "metric": "explicit mapping-equation literal lift closure complete",
            "value": 1.0,
            "note": "This closure artifact re-evaluates whether the explicit same-sector chi_P -> V''(|P|_*) row can be lifted after freezing the literal fragment inventory.",
        },
        {
            "row_id": "explicit_mapping_literal_lift_contract_ready",
            "status": "pass" if mapping_equation_lift_ready and literal_fragment_inventory_ready else "reject",
            "metric": "explicit mapping-equation literal lift prerequisites remain ready",
            "value": 1.0 if mapping_equation_lift_ready and literal_fragment_inventory_ready else 0.0,
            "note": (
                "Both the frozen lift contract and the literal fragment inventory are ready for closure."
                if mapping_equation_lift_ready and literal_fragment_inventory_ready
                else "The literal lift closure cannot proceed unless both the lift contract and literal fragment inventory remain ready."
            ),
        },
        {
            "row_id": "explicit_mapping_literal_lift_present_fragment_count",
            "status": "inventory",
            "metric": "present literal fragment count seen by literal lift closure",
            "value": float(len(present_literal_fragments)),
            "note": f"Present literal fragments are {present_literal_fragments}.",
        },
        {
            "row_id": "explicit_mapping_literal_lift_missing_fragment_count",
            "status": "inventory",
            "metric": "missing literal fragment count seen by literal lift closure",
            "value": float(len(missing_literal_fragments)),
            "note": f"Missing literal fragments are {missing_literal_fragments}.",
        },
        {
            "row_id": "explicit_mapping_literal_lift_equation_row_literal_present",
            "status": "pass" if not explicit_mapping_equation_literal_missing else "watch",
            "metric": "explicit mapping equation row literal present in current public canonical pack",
            "value": 0.0 if explicit_mapping_equation_literal_missing else 1.0,
            "note": (
                "The explicit chi_P -> V''(|P|_*) equation row literal is present."
                if not explicit_mapping_equation_literal_missing
                else "The explicit chi_P -> V''(|P|_*) equation row literal is still absent."
            ),
        },
        {
            "row_id": "explicit_mapping_literal_lift_same_sector_relation_wording_present",
            "status": "pass" if not missing_relation_fragments else "watch",
            "metric": "same-sector relation wording is present for explicit mapping lift",
            "value": 0.0 if missing_relation_fragments else 1.0,
            "note": (
                "Both the same-sector equivalence statement and the mapping operator / relation wording are present."
                if not missing_relation_fragments
                else f"Missing same-sector relation wording fragments are {missing_relation_fragments}."
            ),
        },
        {
            "row_id": "explicit_mapping_literal_lift_available",
            "status": "pass" if explicit_mapping_equation_available else "watch",
            "metric": "explicit mapping equation is literal-liftable into public canonical form",
            "value": 1.0 if explicit_mapping_equation_available else 0.0,
            "note": (
                f"Lifted mapping equation kind is {lifted_mapping_equation_kind_or_none}."
                if explicit_mapping_equation_available
                else f"Literal lift remains non-closing because {literal_lift_nonclosure_reason_or_none}."
            ),
        },
        {
            "row_id": "explicit_mapping_literal_lift_without_new_free_parameters",
            "status": "pass" if mapping_without_new_free_parameters else "reject",
            "metric": "explicit mapping equation literal lift closes without new free parameters",
            "value": 1.0 if mapping_without_new_free_parameters else 0.0,
            "note": (
                "The explicit mapping equation row is now liftable using only already-frozen same-sector ingredients."
                if mapping_without_new_free_parameters
                else "The literal lift does not yet close, so the no-new-free-parameter claim cannot be promoted."
            ),
        },
        {
            "row_id": "explicit_mapping_literal_lift_nonclosure_reason_fixed",
            "status": "pass" if literal_lift_nonclosure_reason_or_none is None else "watch",
            "metric": "explicit mapping-equation literal lift nonclosure reason fixed",
            "value": 1.0 if literal_lift_nonclosure_reason_or_none is None else 0.0,
            "note": (
                "No nonclosure reason remains because the literal lift is closed."
                if literal_lift_nonclosure_reason_or_none is None
                else f"Literal lift nonclosure reason is {literal_lift_nonclosure_reason_or_none}."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "explicit mapping-equation literal lift closure",
        },
        "inputs": {
            "mass_origin_explicit_mapping_equation_lift_contract_json": _relative_str(LIFT_CONTRACT_JSON),
            "mass_origin_explicit_mapping_equation_lift_json": _relative_str(LIFT_AUDIT_JSON),
            "mass_origin_explicit_mapping_literal_fragment_inventory_json": _relative_str(LITERAL_INVENTORY_JSON),
        },
        "intent": "Re-close whether the explicit same-sector chi_P -> V''(|P|_*) equation row can be literal-lifted from the current public canonical pack.",
        "formulas": {
            "literal_lift_rule": "explicit_mapping_equation_available iff the frozen lift audit is satisfied and the literal fragment inventory carries no missing fragments",
            "nonclosure_rule": "literal_lift_nonclosure_reason_or_none is fixed by the remaining missing literal fragments and any residual frozen lift requirements",
        },
        "rows": rows,
        "summary": {
            "explicit_mapping_equation_available": explicit_mapping_equation_available,
            "lifted_mapping_equation_kind_or_none": lifted_mapping_equation_kind_or_none,
            "mapping_without_new_free_parameters": mapping_without_new_free_parameters,
            "literal_lift_nonclosure_reason_or_none": literal_lift_nonclosure_reason_or_none,
            "required_literal_fragment_count": len(required_literal_fragments),
            "present_literal_fragment_count": len(present_literal_fragments),
            "missing_literal_fragment_count": len(missing_literal_fragments),
            "missing_literal_fragments": missing_literal_fragments,
            "missing_relation_fragments": missing_relation_fragments,
            "missing_lift_requirements": missing_lift_requirements,
        },
        "decision": {
            "overall_status": (
                "explicit_mapping_literal_lift_closure_closed"
                if explicit_mapping_equation_available
                else "explicit_mapping_literal_lift_closure_frozen_absent"
            ),
            "keep_mass_origin_branch_blocked": True,
            "explicit_mapping_equation_available": explicit_mapping_equation_available,
            "lifted_mapping_equation_kind_or_none": lifted_mapping_equation_kind_or_none,
            "mapping_without_new_free_parameters": mapping_without_new_free_parameters,
            "literal_lift_nonclosure_reason_or_none": literal_lift_nonclosure_reason_or_none,
        },
        "evidence": {
            "lift_contract_summary": lift_contract_summary,
            "lift_audit_summary": lift_audit_summary,
            "literal_inventory_summary": literal_inventory_summary,
            "literal_inventory_decision": literal_inventory_decision,
            "field_presence": lift_audit_evidence.get("field_presence", {}),
            "equation_slot_presence": lift_audit_evidence.get("equation_slot_presence", {}),
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "status", "metric", "value", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    args = _parse_args()
    payload = _build_payload(step_tag=args.step_tag)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with OUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    _write_csv(payload["rows"])
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
