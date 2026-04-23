#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_explicit_mapping_missing_slot_source_inventory.py

Step 8.7.55.2.118:
Inventory source candidates for the explicit-mapping wording slots that
remain missing after the wording-slot residual split contract.

Inputs:
  - output/public/quantum/mass_origin_explicit_mapping_wording_slot_inventory_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_wording_closure_retry_metrics.json
  - output/public/quantum/mass_origin_wording_slot_residual_split_contract_metrics.json

Outputs:
  - output/public/quantum/mass_origin_explicit_mapping_missing_slot_source_inventory_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_missing_slot_source_inventory_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

WORDING_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_wording_slot_inventory_metrics.json"
WORDING_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_wording_closure_retry_metrics.json"
SPLIT_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_wording_slot_residual_split_contract_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_missing_slot_source_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_missing_slot_source_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.118"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory present and missing source candidates for explicit-mapping missing wording slots.",
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


# 関数: `_unique_preserve_order` の入出力契約と処理意図を定義する。

def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []

    for item in items:
        # 条件分岐: `item not in seen` を満たす経路を評価する。
        if item not in seen:
            seen.add(item)
            ordered.append(item)

    return ordered


# 関数: `_source_plan` の入出力契約と処理意図を定義する。

def _source_plan(present_slots: List[str]) -> Dict[str, Dict[str, List[str] | str]]:
    lhs_present = "lhs_observable_chi_P_or_declared_same_sector_shell_equivalent" in present_slots
    rhs_present = "rhs_curvature_symbol_Vpp_absP_star" in present_slots
    reference_present = "reference_point_absP_star" in present_slots

    equation_context_sources = []

    if lhs_present:
        equation_context_sources.append("lhs_observable_context")

    if rhs_present:
        equation_context_sources.append("rhs_curvature_context")

    if reference_present:
        equation_context_sources.append("reference_point_context")

    relation_context_sources = []

    if lhs_present:
        relation_context_sources.append("lhs_observable_context")

    if rhs_present:
        relation_context_sources.append("rhs_curvature_context")

    same_sector_context_sources = []

    if lhs_present:
        same_sector_context_sources.append("lhs_same_sector_context")

    return {
        "explicit_mapping_equation_literal": {
            "present_sources": equation_context_sources,
            "missing_sources": ["explicit_mapping_equation_literal"],
            "note": "The lhs observable, rhs curvature symbol, and |P|_* reference point are already fixed, but the explicit mapping equation literal itself is still absent.",
        },
        "same_sector_equivalence_statement_or_none": {
            "present_sources": same_sector_context_sources,
            "missing_sources": ["same_sector_equivalence_statement_literal"],
            "note": "The observable side is already declared in same-sector form, but the explicit same-sector equivalence statement that bridges lhs and rhs is still absent.",
        },
        "mapping_operator_or_relation": {
            "present_sources": relation_context_sources,
            "missing_sources": ["mapping_operator_or_relation_literal"],
            "note": "The lhs and rhs symbols are both present, but the mapping operator / relation that ties them together is still absent.",
        },
    }


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (WORDING_INVENTORY_JSON, WORDING_RETRY_JSON, SPLIT_CONTRACT_JSON):
        _require_path(path)

    wording_inventory = _read_json(WORDING_INVENTORY_JSON)
    wording_retry = _read_json(WORDING_RETRY_JSON)
    split_contract = _read_json(SPLIT_CONTRACT_JSON)

    wording_inventory_summary = wording_inventory.get("summary", {})
    wording_retry_summary = wording_retry.get("summary", {})
    split_contract_summary = split_contract.get("summary", {})

    required_explicit_mapping_missing_wording_slots = [
        str(item) for item in split_contract_summary.get("explicit_mapping_missing_wording_slots", [])
    ]
    present_explicit_mapping_wording_slots = [
        str(item) for item in wording_inventory_summary.get("present_explicit_mapping_wording_slots", [])
    ]
    explicit_mapping_wording_route_still_admissible = bool(
        split_contract_summary.get("explicit_mapping_wording_slot_route_still_admissible", False)
    )
    explicit_mapping_wording_inventory_ready = bool(
        wording_retry_summary.get("explicit_mapping_wording_inventory_ready", False)
    )

    source_plan = _source_plan(present_slots=present_explicit_mapping_wording_slots)
    present_sources_by_slot = {
        slot_id: list(source_plan.get(slot_id, {}).get("present_sources", []))
        for slot_id in required_explicit_mapping_missing_wording_slots
    }
    missing_sources_by_slot = {
        slot_id: list(source_plan.get(slot_id, {}).get("missing_sources", []))
        for slot_id in required_explicit_mapping_missing_wording_slots
    }
    present_explicit_mapping_missing_wording_sources = _unique_preserve_order(
        [
            source_id
            for slot_id in required_explicit_mapping_missing_wording_slots
            for source_id in present_sources_by_slot.get(slot_id, [])
        ]
    )
    missing_explicit_mapping_missing_wording_sources = _unique_preserve_order(
        [
            source_id
            for slot_id in required_explicit_mapping_missing_wording_slots
            for source_id in missing_sources_by_slot.get(slot_id, [])
        ]
    )
    explicit_mapping_missing_slot_source_inventory_ready = bool(
        explicit_mapping_wording_route_still_admissible
        and explicit_mapping_wording_inventory_ready
        and set(required_explicit_mapping_missing_wording_slots).issubset(set(source_plan.keys()))
    )

    rows: List[Dict[str, Any]] = [
        {
            "row_id": "explicit_mapping_missing_slot_source_inventory_complete",
            "status": "pass",
            "metric": "explicit-mapping missing-slot source inventory complete",
            "value": 1.0,
            "slot_id": "aggregate",
            "source_id": "aggregate",
            "note": "This step freezes present and missing source candidates for the explicit-mapping wording slots that are still absent.",
        },
        {
            "row_id": "explicit_mapping_missing_slot_source_route_admissible",
            "status": "pass" if explicit_mapping_wording_route_still_admissible else "reject",
            "metric": "explicit-mapping missing-slot source route remains admissible",
            "value": 1.0 if explicit_mapping_wording_route_still_admissible else 0.0,
            "slot_id": "aggregate",
            "source_id": "aggregate",
            "note": (
                "The explicit-mapping wording route remains admissible after the residual split."
                if explicit_mapping_wording_route_still_admissible
                else "The explicit-mapping wording route is no longer admissible, so missing-slot source inventory cannot support closure retry."
            ),
        },
        {
            "row_id": "explicit_mapping_missing_slot_source_inventory_ready",
            "status": "pass" if explicit_mapping_missing_slot_source_inventory_ready else "reject",
            "metric": "explicit-mapping missing-slot source inventory ready for closure retry",
            "value": 1.0 if explicit_mapping_missing_slot_source_inventory_ready else 0.0,
            "slot_id": "aggregate",
            "source_id": "aggregate",
            "note": (
                "Each missing explicit-mapping wording slot now has an explicit present/missing source decomposition."
                if explicit_mapping_missing_slot_source_inventory_ready
                else "The current public canonical pack still cannot provide a stable source decomposition for every explicit-mapping missing wording slot."
            ),
        },
        {
            "row_id": "explicit_mapping_missing_slot_source_present_count",
            "status": "inventory",
            "metric": "present explicit-mapping missing-slot source count",
            "value": float(len(present_explicit_mapping_missing_wording_sources)),
            "slot_id": "aggregate",
            "source_id": "aggregate",
            "note": f"Present explicit-mapping missing-slot sources are {present_explicit_mapping_missing_wording_sources}.",
        },
        {
            "row_id": "explicit_mapping_missing_slot_source_missing_count",
            "status": "inventory",
            "metric": "missing explicit-mapping missing-slot source count",
            "value": float(len(missing_explicit_mapping_missing_wording_sources)),
            "slot_id": "aggregate",
            "source_id": "aggregate",
            "note": f"Missing explicit-mapping missing-slot sources are {missing_explicit_mapping_missing_wording_sources}.",
        },
    ]

    for slot_id in required_explicit_mapping_missing_wording_slots:
        slot_present_sources = present_sources_by_slot.get(slot_id, [])
        slot_missing_sources = missing_sources_by_slot.get(slot_id, [])
        slot_note = str(source_plan.get(slot_id, {}).get("note", ""))
        slot_closable_now = not slot_missing_sources
        rows.append(
            {
                "row_id": f"explicit_mapping_missing_slot_source_{slot_id}",
                "status": "pass" if slot_closable_now else "watch",
                "metric": f"source decomposition for explicit-mapping missing wording slot {slot_id}",
                "value": float(len(slot_present_sources)),
                "slot_id": slot_id,
                "source_id": "aggregate",
                "note": f"Present sources: {slot_present_sources}. Missing sources: {slot_missing_sources}. {slot_note}",
            }
        )

        for source_id in slot_present_sources:
            rows.append(
                {
                    "row_id": f"explicit_mapping_missing_slot_present_source_{slot_id}_{source_id}",
                    "status": "inventory",
                    "metric": f"present source candidate for explicit-mapping missing wording slot {slot_id}",
                    "value": 1.0,
                    "slot_id": slot_id,
                    "source_id": source_id,
                    "note": f"{source_id} is already present in the current public canonical pack as context for {slot_id}.",
                }
            )

        for source_id in slot_missing_sources:
            rows.append(
                {
                    "row_id": f"explicit_mapping_missing_slot_missing_source_{slot_id}_{source_id}",
                    "status": "watch",
                    "metric": f"missing source candidate for explicit-mapping missing wording slot {slot_id}",
                    "value": 0.0,
                    "slot_id": slot_id,
                    "source_id": source_id,
                    "note": f"{source_id} is still absent from the current public canonical pack for {slot_id}.",
                }
            )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "explicit mapping missing-slot source inventory",
        },
        "inputs": {
            "mass_origin_explicit_mapping_wording_slot_inventory_json": _relative_str(WORDING_INVENTORY_JSON),
            "mass_origin_explicit_mapping_wording_closure_retry_json": _relative_str(WORDING_RETRY_JSON),
            "mass_origin_wording_slot_residual_split_contract_json": _relative_str(SPLIT_CONTRACT_JSON),
        },
        "intent": "Decompose the remaining explicit-mapping wording slots into present and missing source candidates using only the current public canonical pack.",
        "formulas": {
            "slot_source_inventory_rule": "each explicit-mapping missing wording slot is decomposed into contextual present sources already in the public pack and literal/relation sources that remain absent",
            "inventory_ready_rule": "explicit_mapping_missing_slot_source_inventory_ready iff the explicit-mapping wording route remains admissible, the prior wording inventory remains ready, and every residual missing slot is explicitly decomposed into present/missing source candidates",
        },
        "rows": rows,
        "summary": {
            "required_explicit_mapping_missing_wording_slots": required_explicit_mapping_missing_wording_slots,
            "present_explicit_mapping_missing_wording_sources": present_explicit_mapping_missing_wording_sources,
            "missing_explicit_mapping_missing_wording_sources": missing_explicit_mapping_missing_wording_sources,
            "present_explicit_mapping_missing_sources_by_slot": present_sources_by_slot,
            "missing_explicit_mapping_missing_sources_by_slot": missing_sources_by_slot,
            "explicit_mapping_wording_route_still_admissible": explicit_mapping_wording_route_still_admissible,
            "explicit_mapping_wording_inventory_ready": explicit_mapping_wording_inventory_ready,
            "explicit_mapping_missing_slot_source_inventory_ready": explicit_mapping_missing_slot_source_inventory_ready,
            "prior_explicit_mapping_wording_nonclosure_reason_or_none": wording_retry_summary.get(
                "explicit_mapping_wording_nonclosure_reason_or_none"
            ),
        },
        "decision": {
            "overall_status": "explicit_mapping_missing_slot_source_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "required_explicit_mapping_missing_wording_slots": required_explicit_mapping_missing_wording_slots,
            "present_explicit_mapping_missing_wording_sources": present_explicit_mapping_missing_wording_sources,
            "missing_explicit_mapping_missing_wording_sources": missing_explicit_mapping_missing_wording_sources,
            "explicit_mapping_missing_slot_source_inventory_ready": explicit_mapping_missing_slot_source_inventory_ready,
        },
        "evidence": {
            "explicit_mapping_wording_slot_inventory_summary": wording_inventory_summary,
            "explicit_mapping_wording_closure_retry_summary": wording_retry_summary,
            "wording_slot_residual_split_contract_summary": split_contract_summary,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "status", "metric", "value", "slot_id", "source_id", "note"],
        )
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
