#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_anchor_missing_slot_source_inventory.py

Step 8.7.55.2.116:
Inventory source candidates for the three shell-anchor wording slots that
remain missing after the wording-slot residual split contract.

Inputs:
  - output/public/quantum/mass_origin_shell_anchor_wording_slot_inventory_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_wording_closure_retry_metrics.json
  - output/public/quantum/mass_origin_wording_slot_residual_split_contract_metrics.json

Outputs:
  - output/public/quantum/mass_origin_shell_anchor_missing_slot_source_inventory_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_missing_slot_source_inventory_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

WORDING_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_wording_slot_inventory_metrics.json"
WORDING_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_wording_closure_retry_metrics.json"
SPLIT_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_wording_slot_residual_split_contract_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_missing_slot_source_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_missing_slot_source_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.116"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory present and missing source candidates for shell-anchor missing wording slots.",
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

def _source_plan(
    present_slots: List[str],
    required_bridge_target_symbol: str,
) -> Dict[str, Dict[str, List[str] | str]]:
    target_symbol_source = f"bridge_target_symbol::{required_bridge_target_symbol}"
    pair_reference_present = "shell_anchor_pair_reference" in present_slots
    same_sector_present = "same_sector_statement" in present_slots

    return {
        "pair_to_target_relation": {
            "present_sources": (
                ["shell_anchor_pair_reference_context"]
                if pair_reference_present
                else []
            ),
            "missing_sources": ["pair_to_target_relation_literal"],
            "note": (
                f"The shell-anchor pair is already named, but the current public canonical pack still lacks an explicit relation from that pair to {required_bridge_target_symbol}."
            ),
        },
        "dimensionless_target_note": {
            "present_sources": [target_symbol_source] if required_bridge_target_symbol else [],
            "missing_sources": ["dimensionless_target_note_literal"],
            "note": (
                f"The target symbol {required_bridge_target_symbol} is already fixed, but no public note states that the target is dimensionless."
            ),
        },
        "no_new_free_parameter_note": {
            "present_sources": (
                ["same_sector_statement_context"]
                if same_sector_present
                else []
            ),
            "missing_sources": ["no_new_free_parameter_note_literal"],
            "note": "The same-sector route remains declared, but the public pack still lacks an explicit no-new-free-parameter note for the shell-anchor wording route.",
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
    wording_inventory_evidence = wording_inventory.get("evidence", {})
    wording_retry_summary = wording_retry.get("summary", {})
    split_contract_summary = split_contract.get("summary", {})

    required_shell_anchor_missing_wording_slots = [
        str(item) for item in split_contract_summary.get("shell_anchor_missing_wording_slots", [])
    ]
    present_shell_anchor_wording_slots = [
        str(item) for item in wording_inventory_summary.get("present_shell_anchor_wording_slots", [])
    ]
    shell_anchor_wording_route_still_admissible = bool(
        split_contract_summary.get("shell_anchor_wording_slot_route_still_admissible", False)
    )
    shell_anchor_wording_inventory_ready = bool(
        wording_retry_summary.get("shell_anchor_wording_inventory_ready", False)
    )
    required_bridge_target_symbol = str(
        wording_inventory_evidence
        .get("shell_anchor_semantic_bridge_contract_summary", {})
        .get("required_bridge_target_symbol", "absP_star_times_vppp_over_vpp")
    )

    source_plan = _source_plan(
        present_slots=present_shell_anchor_wording_slots,
        required_bridge_target_symbol=required_bridge_target_symbol,
    )
    present_sources_by_slot = {
        slot_id: list(source_plan.get(slot_id, {}).get("present_sources", []))
        for slot_id in required_shell_anchor_missing_wording_slots
    }
    missing_sources_by_slot = {
        slot_id: list(source_plan.get(slot_id, {}).get("missing_sources", []))
        for slot_id in required_shell_anchor_missing_wording_slots
    }
    present_shell_anchor_missing_wording_sources = _unique_preserve_order(
        [
            source_id
            for slot_id in required_shell_anchor_missing_wording_slots
            for source_id in present_sources_by_slot.get(slot_id, [])
        ]
    )
    missing_shell_anchor_missing_wording_sources = _unique_preserve_order(
        [
            source_id
            for slot_id in required_shell_anchor_missing_wording_slots
            for source_id in missing_sources_by_slot.get(slot_id, [])
        ]
    )
    shell_anchor_missing_slot_source_inventory_ready = bool(
        shell_anchor_wording_route_still_admissible
        and shell_anchor_wording_inventory_ready
        and set(required_shell_anchor_missing_wording_slots).issubset(set(source_plan.keys()))
    )

    rows: List[Dict[str, Any]] = [
        {
            "row_id": "shell_anchor_missing_slot_source_inventory_complete",
            "status": "pass",
            "metric": "shell-anchor missing-slot source inventory complete",
            "value": 1.0,
            "slot_id": "aggregate",
            "source_id": "aggregate",
            "note": "This step freezes present and missing source candidates for the shell-anchor wording slots that are still absent.",
        },
        {
            "row_id": "shell_anchor_missing_slot_source_route_admissible",
            "status": "pass" if shell_anchor_wording_route_still_admissible else "reject",
            "metric": "shell-anchor missing-slot source route remains admissible",
            "value": 1.0 if shell_anchor_wording_route_still_admissible else 0.0,
            "slot_id": "aggregate",
            "source_id": "aggregate",
            "note": (
                "The shell-anchor wording route remains admissible after the residual split."
                if shell_anchor_wording_route_still_admissible
                else "The shell-anchor wording route is no longer admissible, so missing-slot source inventory cannot support closure retry."
            ),
        },
        {
            "row_id": "shell_anchor_missing_slot_source_inventory_ready",
            "status": "pass" if shell_anchor_missing_slot_source_inventory_ready else "reject",
            "metric": "shell-anchor missing-slot source inventory ready for closure retry",
            "value": 1.0 if shell_anchor_missing_slot_source_inventory_ready else 0.0,
            "slot_id": "aggregate",
            "source_id": "aggregate",
            "note": (
                "Each missing shell-anchor wording slot now has an explicit present/missing source decomposition."
                if shell_anchor_missing_slot_source_inventory_ready
                else "The current public canonical pack still cannot provide a stable source decomposition for every shell-anchor missing wording slot."
            ),
        },
        {
            "row_id": "shell_anchor_missing_slot_source_present_count",
            "status": "inventory",
            "metric": "present shell-anchor missing-slot source count",
            "value": float(len(present_shell_anchor_missing_wording_sources)),
            "slot_id": "aggregate",
            "source_id": "aggregate",
            "note": f"Present shell-anchor missing-slot sources are {present_shell_anchor_missing_wording_sources}.",
        },
        {
            "row_id": "shell_anchor_missing_slot_source_missing_count",
            "status": "inventory",
            "metric": "missing shell-anchor missing-slot source count",
            "value": float(len(missing_shell_anchor_missing_wording_sources)),
            "slot_id": "aggregate",
            "source_id": "aggregate",
            "note": f"Missing shell-anchor missing-slot sources are {missing_shell_anchor_missing_wording_sources}.",
        },
    ]

    for slot_id in required_shell_anchor_missing_wording_slots:
        slot_present_sources = present_sources_by_slot.get(slot_id, [])
        slot_missing_sources = missing_sources_by_slot.get(slot_id, [])
        slot_note = str(source_plan.get(slot_id, {}).get("note", ""))
        slot_closable_now = not slot_missing_sources
        rows.append(
            {
                "row_id": f"shell_anchor_missing_slot_source_{slot_id}",
                "status": "pass" if slot_closable_now else "watch",
                "metric": f"source decomposition for shell-anchor missing wording slot {slot_id}",
                "value": float(len(slot_present_sources)),
                "slot_id": slot_id,
                "source_id": "aggregate",
                "note": (
                    f"Present sources: {slot_present_sources}. Missing sources: {slot_missing_sources}. {slot_note}"
                ),
            }
        )

        for source_id in slot_present_sources:
            rows.append(
                {
                    "row_id": f"shell_anchor_missing_slot_present_source_{slot_id}_{source_id}",
                    "status": "inventory",
                    "metric": f"present source candidate for shell-anchor missing wording slot {slot_id}",
                    "value": 1.0,
                    "slot_id": slot_id,
                    "source_id": source_id,
                    "note": f"{source_id} is already present in the current public canonical pack as context for {slot_id}.",
                }
            )

        for source_id in slot_missing_sources:
            rows.append(
                {
                    "row_id": f"shell_anchor_missing_slot_missing_source_{slot_id}_{source_id}",
                    "status": "watch",
                    "metric": f"missing source candidate for shell-anchor missing wording slot {slot_id}",
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
            "name": "shell-anchor missing-slot source inventory",
        },
        "inputs": {
            "mass_origin_shell_anchor_wording_slot_inventory_json": _relative_str(WORDING_INVENTORY_JSON),
            "mass_origin_shell_anchor_wording_closure_retry_json": _relative_str(WORDING_RETRY_JSON),
            "mass_origin_wording_slot_residual_split_contract_json": _relative_str(SPLIT_CONTRACT_JSON),
        },
        "intent": "Decompose the remaining shell-anchor wording slots into present and missing source candidates using only the current public canonical pack.",
        "formulas": {
            "slot_source_inventory_rule": "each shell-anchor missing wording slot is decomposed into contextual present sources already in the public pack and literal/note sources that remain absent",
            "inventory_ready_rule": "shell_anchor_missing_slot_source_inventory_ready iff the shell-anchor wording route remains admissible, the prior wording inventory remains ready, and every residual missing slot is explicitly decomposed into present/missing source candidates",
        },
        "rows": rows,
        "summary": {
            "required_shell_anchor_missing_wording_slots": required_shell_anchor_missing_wording_slots,
            "present_shell_anchor_missing_wording_sources": present_shell_anchor_missing_wording_sources,
            "missing_shell_anchor_missing_wording_sources": missing_shell_anchor_missing_wording_sources,
            "present_shell_anchor_missing_sources_by_slot": present_sources_by_slot,
            "missing_shell_anchor_missing_sources_by_slot": missing_sources_by_slot,
            "shell_anchor_wording_route_still_admissible": shell_anchor_wording_route_still_admissible,
            "shell_anchor_wording_inventory_ready": shell_anchor_wording_inventory_ready,
            "shell_anchor_missing_slot_source_inventory_ready": shell_anchor_missing_slot_source_inventory_ready,
            "required_bridge_target_symbol": required_bridge_target_symbol,
            "prior_shell_anchor_wording_nonclosure_reason_or_none": wording_retry_summary.get(
                "shell_anchor_wording_nonclosure_reason_or_none"
            ),
        },
        "decision": {
            "overall_status": "shell_anchor_missing_slot_source_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "required_shell_anchor_missing_wording_slots": required_shell_anchor_missing_wording_slots,
            "present_shell_anchor_missing_wording_sources": present_shell_anchor_missing_wording_sources,
            "missing_shell_anchor_missing_wording_sources": missing_shell_anchor_missing_wording_sources,
            "shell_anchor_missing_slot_source_inventory_ready": shell_anchor_missing_slot_source_inventory_ready,
        },
        "evidence": {
            "shell_anchor_wording_slot_inventory_summary": wording_inventory_summary,
            "shell_anchor_wording_closure_retry_summary": wording_retry_summary,
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
