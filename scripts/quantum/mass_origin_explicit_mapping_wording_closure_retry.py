#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_explicit_mapping_wording_closure_retry.py

Step 8.7.55.2.113:
Retry explicit mapping wording closure using the frozen wording-slot
inventory produced after the source-wording blocker split.

Inputs:
  - output/public/quantum/mass_origin_explicit_mapping_literal_lift_metrics.json
  - output/public/quantum/mass_origin_source_wording_blocker_split_contract_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_wording_slot_inventory_metrics.json

Outputs:
  - output/public/quantum/mass_origin_explicit_mapping_wording_closure_retry_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_wording_closure_retry_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

LITERAL_LIFT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_literal_lift_metrics.json"
SPLIT_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_source_wording_blocker_split_contract_metrics.json"
WORDING_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_wording_slot_inventory_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_wording_closure_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_wording_closure_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.113"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry explicit mapping wording closure using the frozen slot inventory.",
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
    for path in (LITERAL_LIFT_JSON, SPLIT_CONTRACT_JSON, WORDING_INVENTORY_JSON):
        _require_path(path)

    literal_lift = _read_json(LITERAL_LIFT_JSON)
    split_contract = _read_json(SPLIT_CONTRACT_JSON)
    wording_inventory = _read_json(WORDING_INVENTORY_JSON)

    literal_lift_summary = literal_lift.get("summary", {})
    split_contract_summary = split_contract.get("summary", {})
    wording_inventory_summary = wording_inventory.get("summary", {})

    explicit_mapping_wording_route_still_admissible = bool(
        split_contract_summary.get("explicit_mapping_wording_route_still_admissible", False)
    )
    explicit_mapping_wording_inventory_ready = bool(
        wording_inventory_summary.get("explicit_mapping_wording_inventory_ready", False)
    )
    required_explicit_mapping_wording_slots = [
        str(item) for item in wording_inventory_summary.get("required_explicit_mapping_wording_slots", [])
    ]
    present_explicit_mapping_wording_slots = [
        str(item) for item in wording_inventory_summary.get("present_explicit_mapping_wording_slots", [])
    ]
    missing_explicit_mapping_wording_slots = [
        str(item) for item in wording_inventory_summary.get("missing_explicit_mapping_wording_slots", [])
    ]
    explicit_mapping_equation_available = bool(
        explicit_mapping_wording_route_still_admissible
        and explicit_mapping_wording_inventory_ready
        and not missing_explicit_mapping_wording_slots
    )
    lifted_mapping_equation_kind_or_none = (
        "explicit_same_sector_chi_to_vpp_row"
        if explicit_mapping_equation_available
        else None
    )
    mapping_without_new_free_parameters = bool(explicit_mapping_equation_available)
    explicit_mapping_wording_nonclosure_reason_or_none = None

    # 条件分岐: `not explicit_mapping_equation_available` を満たす経路を評価する。
    if not explicit_mapping_equation_available:
        explicit_mapping_wording_nonclosure_reason_or_none = "explicit_mapping_wording_slots_still_missing"

    rows: List[Dict[str, Any]] = [
        {
            "row_id": "explicit_mapping_wording_closure_retry_complete",
            "status": "pass",
            "metric": "explicit mapping wording closure retry complete",
            "value": 1.0,
            "note": "This step retries explicit mapping equation closure using the frozen required/present/missing wording-slot inventory.",
        },
        {
            "row_id": "explicit_mapping_wording_closure_retry_route_admissible",
            "status": "pass" if explicit_mapping_wording_route_still_admissible else "reject",
            "metric": "explicit mapping wording route remains admissible for closure retry",
            "value": 1.0 if explicit_mapping_wording_route_still_admissible else 0.0,
            "note": (
                "The split branch still permits an explicit mapping wording closure retry."
                if explicit_mapping_wording_route_still_admissible
                else "The explicit mapping wording route is no longer admissible, so closure retry is not meaningful."
            ),
        },
        {
            "row_id": "explicit_mapping_wording_closure_retry_inventory_ready",
            "status": "pass" if explicit_mapping_wording_inventory_ready else "reject",
            "metric": "explicit mapping wording inventory is ready for closure retry",
            "value": 1.0 if explicit_mapping_wording_inventory_ready else 0.0,
            "note": (
                "The wording-slot inventory is internally complete and can support closure retry."
                if explicit_mapping_wording_inventory_ready
                else "The wording-slot inventory is not complete enough to support closure retry."
            ),
        },
        {
            "row_id": "explicit_mapping_wording_closure_retry_missing_slot_count",
            "status": "inventory",
            "metric": "explicit mapping wording closure retry missing slot count",
            "value": float(len(missing_explicit_mapping_wording_slots)),
            "note": f"Missing wording slots at retry are {missing_explicit_mapping_wording_slots}.",
        },
        {
            "row_id": "explicit_mapping_wording_closure_retry_equation_available",
            "status": "pass" if explicit_mapping_equation_available else "watch",
            "metric": "explicit mapping equation available after wording closure retry",
            "value": 1.0 if explicit_mapping_equation_available else 0.0,
            "note": (
                f"Lifted mapping equation kind is {lifted_mapping_equation_kind_or_none}."
                if explicit_mapping_equation_available
                else "Closure retry remains non-closing because required wording slots are still missing."
            ),
        },
        {
            "row_id": "explicit_mapping_wording_closure_retry_no_new_free_parameters",
            "status": "pass" if mapping_without_new_free_parameters else "reject",
            "metric": "explicit mapping wording closure retry closes without new free parameters",
            "value": 1.0 if mapping_without_new_free_parameters else 0.0,
            "note": (
                "Closure retry yields an explicit same-sector mapping row without introducing new free parameters."
                if mapping_without_new_free_parameters
                else "Closure retry did not yield a public explicit mapping row with a no-new-free-parameter closure."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "explicit mapping wording closure retry",
        },
        "inputs": {
            "mass_origin_explicit_mapping_literal_lift_json": _relative_str(LITERAL_LIFT_JSON),
            "mass_origin_source_wording_blocker_split_contract_json": _relative_str(SPLIT_CONTRACT_JSON),
            "mass_origin_explicit_mapping_wording_slot_inventory_json": _relative_str(WORDING_INVENTORY_JSON),
        },
        "intent": "Retry explicit mapping equation closure using the frozen wording-slot partition from the split source-wording branch.",
        "formulas": {
            "closure_retry_rule": "explicit_mapping_equation_available iff the explicit mapping wording route remains admissible, the frozen wording inventory is ready, and no required wording slots remain missing",
            "no_new_parameter_rule": "mapping_without_new_free_parameters iff the explicit mapping equation is available after wording closure retry",
        },
        "rows": rows,
        "summary": {
            "required_explicit_mapping_wording_slots": required_explicit_mapping_wording_slots,
            "present_explicit_mapping_wording_slots": present_explicit_mapping_wording_slots,
            "missing_explicit_mapping_wording_slots": missing_explicit_mapping_wording_slots,
            "explicit_mapping_wording_route_still_admissible": explicit_mapping_wording_route_still_admissible,
            "explicit_mapping_wording_inventory_ready": explicit_mapping_wording_inventory_ready,
            "explicit_mapping_equation_available": explicit_mapping_equation_available,
            "lifted_mapping_equation_kind_or_none": lifted_mapping_equation_kind_or_none,
            "mapping_without_new_free_parameters": mapping_without_new_free_parameters,
            "explicit_mapping_wording_nonclosure_reason_or_none": explicit_mapping_wording_nonclosure_reason_or_none,
            "prior_literal_lift_nonclosure_reason_or_none": literal_lift_summary.get(
                "literal_lift_nonclosure_reason_or_none"
            ),
        },
        "decision": {
            "overall_status": (
                "explicit_mapping_wording_closure_retry_available"
                if explicit_mapping_equation_available
                else "explicit_mapping_wording_closure_retry_frozen_absent"
            ),
            "keep_mass_origin_branch_blocked": True,
            "explicit_mapping_equation_available": explicit_mapping_equation_available,
            "lifted_mapping_equation_kind_or_none": lifted_mapping_equation_kind_or_none,
            "mapping_without_new_free_parameters": mapping_without_new_free_parameters,
            "explicit_mapping_wording_nonclosure_reason_or_none": explicit_mapping_wording_nonclosure_reason_or_none,
            "missing_explicit_mapping_wording_slots": missing_explicit_mapping_wording_slots,
        },
        "evidence": {
            "explicit_mapping_literal_lift_summary": literal_lift_summary,
            "source_wording_blocker_split_contract_summary": split_contract_summary,
            "explicit_mapping_wording_slot_inventory_summary": wording_inventory_summary,
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
