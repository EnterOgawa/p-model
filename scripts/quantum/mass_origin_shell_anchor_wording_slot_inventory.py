#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_anchor_wording_slot_inventory.py

Step 8.7.55.2.110:
Freeze the required / present / missing shell-anchor wording slots after
the source-wording blocker split contract isolates the shell-anchor route.

Inputs:
  - output/public/quantum/mass_origin_shell_anchor_semantic_bridge_contract_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_semantic_bridge_metrics.json
  - output/public/quantum/mass_origin_source_wording_blocker_split_contract_metrics.json

Outputs:
  - output/public/quantum/mass_origin_shell_anchor_wording_slot_inventory_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_wording_slot_inventory_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

SHELL_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_semantic_bridge_contract_metrics.json"
SHELL_BRIDGE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_semantic_bridge_metrics.json"
SPLIT_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_source_wording_blocker_split_contract_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_wording_slot_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_wording_slot_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.110"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory required / present / missing wording slots for the shell-anchor semantic-bridge route.",
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


# 関数: `_slot_notes` の入出力契約と処理意図を定義する。

def _slot_notes(required_bridge_target_symbol: str) -> Dict[str, str]:
    return {
        "shell_anchor_pair_reference": "The shell-anchor pair rows remain publicly complete and continue to anchor the wording route.",
        "pair_to_target_relation": (
            f"A direct same-sector relation from the shell-anchor pair to {required_bridge_target_symbol} is still absent."
        ),
        "same_sector_statement": "Same-sector wording is already attached to the shell-anchor row corpus.",
        "dimensionless_target_note": "A note that the target quantity is dimensionless is still absent from the shell-anchor wording route.",
        "no_new_free_parameter_note": "The shell-anchor wording route still lacks an explicit no-new-free-parameter note.",
    }


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (SHELL_CONTRACT_JSON, SHELL_BRIDGE_JSON, SPLIT_CONTRACT_JSON):
        _require_path(path)

    shell_contract = _read_json(SHELL_CONTRACT_JSON)
    shell_bridge = _read_json(SHELL_BRIDGE_JSON)
    split_contract = _read_json(SPLIT_CONTRACT_JSON)

    shell_contract_summary = shell_contract.get("summary", {})
    shell_bridge_summary = shell_bridge.get("summary", {})
    split_contract_summary = split_contract.get("summary", {})

    required_shell_anchor_wording_slots = [
        str(item) for item in shell_contract_summary.get("required_bridge_relation_slots", [])
    ]
    missing_shell_anchor_wording_slots = [
        str(item) for item in shell_bridge_summary.get("missing_relation_slots", [])
    ]
    present_shell_anchor_wording_slots = [
        slot_id
        for slot_id in required_shell_anchor_wording_slots
        if slot_id not in missing_shell_anchor_wording_slots
    ]
    required_shell_anchor_wording_slot_count = len(required_shell_anchor_wording_slots)
    present_shell_anchor_wording_slot_count = len(present_shell_anchor_wording_slots)
    missing_shell_anchor_wording_slot_count = len(missing_shell_anchor_wording_slots)
    shell_anchor_wording_route_still_admissible = bool(
        split_contract_summary.get("shell_anchor_wording_route_still_admissible", False)
    )
    hand_off_to_8_7_55_2_83 = bool(split_contract_summary.get("hand_off_to_8_7_55_2_83", False))
    shell_anchor_wording_inventory_ready = bool(
        shell_anchor_wording_route_still_admissible
        and not hand_off_to_8_7_55_2_83
        and required_shell_anchor_wording_slot_count
        == present_shell_anchor_wording_slot_count + missing_shell_anchor_wording_slot_count
    )
    required_bridge_target_symbol = str(shell_contract_summary.get("required_bridge_target_symbol"))
    slot_notes = _slot_notes(required_bridge_target_symbol)

    rows: List[Dict[str, Any]] = [
        {
            "row_id": "shell_anchor_wording_slot_inventory_complete",
            "status": "pass",
            "metric": "shell-anchor wording slot inventory complete",
            "value": 1.0,
            "note": "This step freezes which shell-anchor wording slots are already present and which remain missing for the split wording route.",
        },
        {
            "row_id": "shell_anchor_wording_route_still_admissible",
            "status": "pass" if shell_anchor_wording_route_still_admissible else "reject",
            "metric": "shell-anchor wording route remains admissible after the blocker split",
            "value": 1.0 if shell_anchor_wording_route_still_admissible else 0.0,
            "note": (
                "The shell-anchor wording route remains admissible for closure retry."
                if shell_anchor_wording_route_still_admissible
                else "The shell-anchor wording route is no longer admissible, so the split branch cannot continue."
            ),
        },
        {
            "row_id": "shell_anchor_wording_slot_required_count",
            "status": "inventory",
            "metric": "required shell-anchor wording slot count",
            "value": float(required_shell_anchor_wording_slot_count),
            "note": f"Required shell-anchor wording slots are {required_shell_anchor_wording_slots}.",
        },
        {
            "row_id": "shell_anchor_wording_slot_present_count",
            "status": "inventory",
            "metric": "present shell-anchor wording slot count",
            "value": float(present_shell_anchor_wording_slot_count),
            "note": f"Present shell-anchor wording slots are {present_shell_anchor_wording_slots}.",
        },
        {
            "row_id": "shell_anchor_wording_slot_missing_count",
            "status": "inventory",
            "metric": "missing shell-anchor wording slot count",
            "value": float(missing_shell_anchor_wording_slot_count),
            "note": f"Missing shell-anchor wording slots are {missing_shell_anchor_wording_slots}.",
        },
    ]

    for slot_id in required_shell_anchor_wording_slots:
        is_present = slot_id in present_shell_anchor_wording_slots
        rows.append(
            {
                "row_id": f"shell_anchor_wording_slot_{slot_id}",
                "status": "pass" if is_present else "watch",
                "metric": f"shell-anchor wording slot {slot_id} present in the current public canonical pack",
                "value": 1.0 if is_present else 0.0,
                "note": slot_notes.get(slot_id, ""),
            }
        )

    rows.append(
        {
            "row_id": "shell_anchor_wording_inventory_ready",
            "status": "pass" if shell_anchor_wording_inventory_ready else "reject",
            "metric": "shell-anchor wording inventory ready for closure retry",
            "value": 1.0 if shell_anchor_wording_inventory_ready else 0.0,
            "note": (
                "The next step may attempt shell-anchor wording closure retry using the frozen required/present/missing slot split."
                if shell_anchor_wording_inventory_ready
                else "The shell-anchor wording slot inventory is not internally complete enough to support closure retry."
            ),
        }
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "shell-anchor wording slot inventory",
        },
        "inputs": {
            "mass_origin_shell_anchor_semantic_bridge_contract_json": _relative_str(SHELL_CONTRACT_JSON),
            "mass_origin_shell_anchor_semantic_bridge_json": _relative_str(SHELL_BRIDGE_JSON),
            "mass_origin_source_wording_blocker_split_contract_json": _relative_str(SPLIT_CONTRACT_JSON),
        },
        "intent": "Freeze the wording-slot partition for the shell-anchor route before shell-anchor wording closure retry.",
        "formulas": {
            "wording_inventory_rule": "required_shell_anchor_wording_slots are partitioned into present_shell_anchor_wording_slots and missing_shell_anchor_wording_slots using only the current public canonical pack",
            "closure_readiness_rule": "shell_anchor_wording_inventory_ready iff the shell-anchor route remains admissible after the split and every required wording slot is explicitly inventoried as present or missing",
        },
        "rows": rows,
        "summary": {
            "required_shell_anchor_wording_slots": required_shell_anchor_wording_slots,
            "present_shell_anchor_wording_slots": present_shell_anchor_wording_slots,
            "missing_shell_anchor_wording_slots": missing_shell_anchor_wording_slots,
            "required_shell_anchor_wording_slot_count": required_shell_anchor_wording_slot_count,
            "present_shell_anchor_wording_slot_count": present_shell_anchor_wording_slot_count,
            "missing_shell_anchor_wording_slot_count": missing_shell_anchor_wording_slot_count,
            "shell_anchor_wording_route_still_admissible": shell_anchor_wording_route_still_admissible,
            "shell_anchor_wording_inventory_ready": shell_anchor_wording_inventory_ready,
            "shell_anchor_semantic_bridge_available_now": shell_bridge_summary.get("semantic_bridge_available"),
            "shell_anchor_semantic_bridge_nonclosure_reason_or_none": shell_bridge_summary.get(
                "semantic_bridge_nonclosure_reason_or_none"
            ),
        },
        "decision": {
            "overall_status": "shell_anchor_wording_slot_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "required_shell_anchor_wording_slots": required_shell_anchor_wording_slots,
            "present_shell_anchor_wording_slots": present_shell_anchor_wording_slots,
            "missing_shell_anchor_wording_slots": missing_shell_anchor_wording_slots,
            "shell_anchor_wording_inventory_ready": shell_anchor_wording_inventory_ready,
        },
        "evidence": {
            "shell_anchor_semantic_bridge_contract_summary": shell_contract_summary,
            "shell_anchor_semantic_bridge_summary": shell_bridge_summary,
            "source_wording_blocker_split_contract_summary": split_contract_summary,
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
