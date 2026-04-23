#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_explicit_mapping_missing_slot_closure_retry.py

Step 8.7.55.2.119:
Retry explicit-mapping missing-slot closure using the frozen source inventory
produced after the wording-slot residual split contract.

Inputs:
  - output/public/quantum/mass_origin_explicit_mapping_wording_closure_retry_metrics.json
  - output/public/quantum/mass_origin_wording_slot_residual_split_contract_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_missing_slot_source_inventory_metrics.json

Outputs:
  - output/public/quantum/mass_origin_explicit_mapping_missing_slot_closure_retry_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_missing_slot_closure_retry_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

WORDING_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_wording_closure_retry_metrics.json"
SPLIT_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_wording_slot_residual_split_contract_metrics.json"
SOURCE_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_missing_slot_source_inventory_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_missing_slot_closure_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_missing_slot_closure_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.119"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry explicit-mapping missing-slot closure using the frozen source inventory.",
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
    for path in (WORDING_RETRY_JSON, SPLIT_CONTRACT_JSON, SOURCE_INVENTORY_JSON):
        _require_path(path)

    wording_retry = _read_json(WORDING_RETRY_JSON)
    split_contract = _read_json(SPLIT_CONTRACT_JSON)
    source_inventory = _read_json(SOURCE_INVENTORY_JSON)

    wording_retry_summary = wording_retry.get("summary", {})
    split_contract_summary = split_contract.get("summary", {})
    source_inventory_summary = source_inventory.get("summary", {})

    explicit_mapping_missing_slot_route_still_admissible = bool(
        split_contract_summary.get("explicit_mapping_wording_slot_route_still_admissible", False)
    )
    explicit_mapping_missing_slot_source_inventory_ready = bool(
        source_inventory_summary.get("explicit_mapping_missing_slot_source_inventory_ready", False)
    )
    required_explicit_mapping_missing_wording_slots = [
        str(item) for item in source_inventory_summary.get("required_explicit_mapping_missing_wording_slots", [])
    ]
    present_explicit_mapping_missing_wording_sources = [
        str(item) for item in source_inventory_summary.get("present_explicit_mapping_missing_wording_sources", [])
    ]
    missing_explicit_mapping_missing_wording_sources = [
        str(item) for item in source_inventory_summary.get("missing_explicit_mapping_missing_wording_sources", [])
    ]
    explicit_mapping_equation_available = bool(
        explicit_mapping_missing_slot_route_still_admissible
        and explicit_mapping_missing_slot_source_inventory_ready
        and not missing_explicit_mapping_missing_wording_sources
    )
    lifted_mapping_equation_kind_or_none = (
        "explicit_same_sector_chi_to_vpp_row"
        if explicit_mapping_equation_available
        else None
    )
    mapping_without_new_free_parameters = bool(explicit_mapping_equation_available)
    explicit_mapping_missing_slot_nonclosure_reason_or_none = None

    # 条件分岐: `not explicit_mapping_equation_available` を満たす経路を評価する。
    if not explicit_mapping_equation_available:
        explicit_mapping_missing_slot_nonclosure_reason_or_none = "explicit_mapping_missing_slot_sources_still_missing"

    rows: List[Dict[str, Any]] = [
        {
            "row_id": "explicit_mapping_missing_slot_closure_retry_complete",
            "status": "pass",
            "metric": "explicit-mapping missing-slot closure retry complete",
            "value": 1.0,
            "note": "This step retries explicit-mapping closure using the frozen missing-slot source inventory.",
        },
        {
            "row_id": "explicit_mapping_missing_slot_closure_retry_route_admissible",
            "status": "pass" if explicit_mapping_missing_slot_route_still_admissible else "reject",
            "metric": "explicit-mapping missing-slot route remains admissible for closure retry",
            "value": 1.0 if explicit_mapping_missing_slot_route_still_admissible else 0.0,
            "note": (
                "The explicit-mapping residual route remains admissible after the wording-slot residual split."
                if explicit_mapping_missing_slot_route_still_admissible
                else "The explicit-mapping residual route is no longer admissible, so closure retry is not meaningful."
            ),
        },
        {
            "row_id": "explicit_mapping_missing_slot_closure_retry_inventory_ready",
            "status": "pass" if explicit_mapping_missing_slot_source_inventory_ready else "reject",
            "metric": "explicit-mapping missing-slot source inventory is ready for closure retry",
            "value": 1.0 if explicit_mapping_missing_slot_source_inventory_ready else 0.0,
            "note": (
                "The missing-slot source inventory is internally complete and can support closure retry."
                if explicit_mapping_missing_slot_source_inventory_ready
                else "The missing-slot source inventory is not complete enough to support closure retry."
            ),
        },
        {
            "row_id": "explicit_mapping_missing_slot_closure_retry_missing_source_count",
            "status": "inventory",
            "metric": "explicit-mapping missing-slot closure retry missing source count",
            "value": float(len(missing_explicit_mapping_missing_wording_sources)),
            "note": f"Missing sources at retry are {missing_explicit_mapping_missing_wording_sources}.",
        },
        {
            "row_id": "explicit_mapping_missing_slot_closure_retry_equation_available",
            "status": "pass" if explicit_mapping_equation_available else "watch",
            "metric": "explicit mapping equation available after missing-slot closure retry",
            "value": 1.0 if explicit_mapping_equation_available else 0.0,
            "note": (
                f"Lifted mapping equation kind is {lifted_mapping_equation_kind_or_none}."
                if explicit_mapping_equation_available
                else "Closure retry remains non-closing because required explicit-mapping source literals or relations are still missing."
            ),
        },
        {
            "row_id": "explicit_mapping_missing_slot_closure_retry_no_new_free_parameters",
            "status": "pass" if mapping_without_new_free_parameters else "reject",
            "metric": "explicit-mapping missing-slot closure retry closes without new free parameters",
            "value": 1.0 if mapping_without_new_free_parameters else 0.0,
            "note": (
                "Closure retry yields an explicit same-sector mapping row without introducing new free parameters."
                if mapping_without_new_free_parameters
                else "Closure retry did not yield a no-new-free-parameter explicit same-sector mapping row."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "explicit mapping missing-slot closure retry",
        },
        "inputs": {
            "mass_origin_explicit_mapping_wording_closure_retry_json": _relative_str(WORDING_RETRY_JSON),
            "mass_origin_wording_slot_residual_split_contract_json": _relative_str(SPLIT_CONTRACT_JSON),
            "mass_origin_explicit_mapping_missing_slot_source_inventory_json": _relative_str(SOURCE_INVENTORY_JSON),
        },
        "intent": "Retry explicit-mapping closure using the frozen missing-slot source inventory from the slot-level residual branch.",
        "formulas": {
            "closure_retry_rule": "explicit_mapping_equation_available iff the explicit-mapping residual route remains admissible, the frozen missing-slot source inventory is ready, and no required explicit-mapping source candidates remain missing",
            "no_new_parameter_rule": "mapping_without_new_free_parameters iff the explicit mapping equation is available after missing-slot closure retry",
        },
        "rows": rows,
        "summary": {
            "required_explicit_mapping_missing_wording_slots": required_explicit_mapping_missing_wording_slots,
            "present_explicit_mapping_missing_wording_sources": present_explicit_mapping_missing_wording_sources,
            "missing_explicit_mapping_missing_wording_sources": missing_explicit_mapping_missing_wording_sources,
            "explicit_mapping_missing_slot_route_still_admissible": explicit_mapping_missing_slot_route_still_admissible,
            "explicit_mapping_missing_slot_source_inventory_ready": explicit_mapping_missing_slot_source_inventory_ready,
            "explicit_mapping_equation_available": explicit_mapping_equation_available,
            "lifted_mapping_equation_kind_or_none": lifted_mapping_equation_kind_or_none,
            "mapping_without_new_free_parameters": mapping_without_new_free_parameters,
            "explicit_mapping_missing_slot_nonclosure_reason_or_none": explicit_mapping_missing_slot_nonclosure_reason_or_none,
            "prior_explicit_mapping_wording_nonclosure_reason_or_none": wording_retry_summary.get(
                "explicit_mapping_wording_nonclosure_reason_or_none"
            ),
        },
        "decision": {
            "overall_status": (
                "explicit_mapping_missing_slot_closure_retry_available"
                if explicit_mapping_equation_available
                else "explicit_mapping_missing_slot_closure_retry_frozen_absent"
            ),
            "keep_mass_origin_branch_blocked": True,
            "explicit_mapping_equation_available": explicit_mapping_equation_available,
            "lifted_mapping_equation_kind_or_none": lifted_mapping_equation_kind_or_none,
            "mapping_without_new_free_parameters": mapping_without_new_free_parameters,
            "explicit_mapping_missing_slot_nonclosure_reason_or_none": explicit_mapping_missing_slot_nonclosure_reason_or_none,
            "missing_explicit_mapping_missing_wording_sources": missing_explicit_mapping_missing_wording_sources,
        },
        "evidence": {
            "explicit_mapping_wording_closure_retry_summary": wording_retry_summary,
            "wording_slot_residual_split_contract_summary": split_contract_summary,
            "explicit_mapping_missing_slot_source_inventory_summary": source_inventory_summary,
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
