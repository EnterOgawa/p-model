#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_same_sector_tiebreak_shell_anchor_audit.py

Step 8.7.55.2.93:
Audit whether the admissible shell-anchor route alone can close the missing
same-sector tie-break target value without introducing new free parameters.

Inputs:
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_bridge_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_contract_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_inventory_metrics.json

Outputs:
  - output/public/quantum/mass_origin_same_sector_tiebreak_shell_anchor_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_shell_anchor_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

TARGET_BRIDGE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_bridge_metrics.json"
SOURCE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_source_contract_metrics.json"
SOURCE_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_source_inventory_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_shell_anchor_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_shell_anchor_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.93"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether shell-anchor sources alone can close the same-sector tie-break target value.",
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
    for path in (TARGET_BRIDGE_JSON, SOURCE_CONTRACT_JSON, SOURCE_INVENTORY_JSON):
        _require_path(path)

    target_bridge = _read_json(TARGET_BRIDGE_JSON)
    source_contract = _read_json(SOURCE_CONTRACT_JSON)
    source_inventory = _read_json(SOURCE_INVENTORY_JSON)

    bridge_summary = target_bridge.get("summary", {})
    source_contract_summary = source_contract.get("summary", {})
    source_inventory_summary = source_inventory.get("summary", {})

    allowed_source_kind_ids = [str(item) for item in source_contract_summary.get("allowed_source_kind_ids", [])]
    candidate_kind_by_row_id = {
        str(row_id): str(kind_id)
        for row_id, kind_id in source_inventory_summary.get("candidate_source_kind_by_row_id", {}).items()
    }
    shell_anchor_candidate_row_ids = [
        row_id
        for row_id, kind_id in candidate_kind_by_row_id.items()
        if kind_id == "surviving_shell_anchor_pack"
    ]
    shell_anchor_candidate_count = len(shell_anchor_candidate_row_ids)
    shell_anchor_source_kind_allowed = "surviving_shell_anchor_pack" in allowed_source_kind_ids
    shell_anchor_target_value_available = bool(
        bridge_summary.get("target_source_kind_or_none") == "surviving_shell_anchor_pack"
    )
    shell_anchor_bridge_without_new_free_parameters = bool(
        shell_anchor_target_value_available
        and source_contract_summary.get("bridge_without_new_free_parameters_required", False)
    )
    candidate_match_count = int(bridge_summary.get("candidate_match_count", 0)) if shell_anchor_target_value_available else 0
    matching_candidate_ids = (
        [str(item) for item in bridge_summary.get("matching_candidate_ids", [])]
        if shell_anchor_target_value_available
        else []
    )
    inventory_status = str(source_inventory_summary.get("inventory_status", ""))
    overall_status = (
        "same_sector_tiebreak_shell_anchor_target_value_available"
        if shell_anchor_target_value_available
        else "same_sector_tiebreak_shell_anchor_frozen_target_missing"
    )

    rows = [
        {
            "row_id": "same_sector_tiebreak_shell_anchor_audit_complete",
            "status": "pass",
            "metric": "shell-anchor target derivability audit complete",
            "value": 1.0,
            "note": "This step isolates the shell-anchor route from the broader target-source inventory and audits whether it alone can close the tie-break target value.",
        },
        {
            "row_id": "same_sector_tiebreak_shell_anchor_source_kind_allowed",
            "status": "pass" if shell_anchor_source_kind_allowed else "reject",
            "metric": "surviving shell-anchor pack remains an allowed target-source kind",
            "value": 1.0 if shell_anchor_source_kind_allowed else 0.0,
            "note": f"Allowed source kinds are {allowed_source_kind_ids}.",
        },
        {
            "row_id": "same_sector_tiebreak_shell_anchor_candidate_count",
            "status": "inventory",
            "metric": "shell-anchor candidate source row count",
            "value": float(shell_anchor_candidate_count),
            "note": f"Shell-anchor candidate source rows are {shell_anchor_candidate_row_ids}.",
        },
        {
            "row_id": "same_sector_tiebreak_shell_anchor_inventory_status",
            "status": "watch" if not shell_anchor_target_value_available else "pass",
            "metric": "shell-anchor route status inside current inventory",
            "value": 0.0 if not shell_anchor_target_value_available else 1.0,
            "note": (
                f"Inventory status is {inventory_status}; shell-anchor source rows are present but the route is still non-closing."
                if not shell_anchor_target_value_available
                else f"Inventory status is {inventory_status}; the shell-anchor route now closes the target value."
            ),
        },
        {
            "row_id": "same_sector_tiebreak_shell_anchor_target_value_available",
            "status": "pass" if shell_anchor_target_value_available else "watch",
            "metric": "shell-anchor route alone yields a public target value",
            "value": 1.0 if shell_anchor_target_value_available else 0.0,
            "note": (
                "The surviving shell-anchor pack directly yields the target value for the tie-break invariant."
                if shell_anchor_target_value_available
                else "The surviving shell-anchor rows remain kappa-like anchors only; no public canonical shell-anchor row fixes the derivative-ratio target value."
            ),
        },
        {
            "row_id": "same_sector_tiebreak_shell_anchor_bridge_without_new_free_parameters",
            "status": "pass" if shell_anchor_bridge_without_new_free_parameters else "reject",
            "metric": "shell-anchor route closes without new free parameters",
            "value": 1.0 if shell_anchor_bridge_without_new_free_parameters else 0.0,
            "note": (
                "The shell-anchor route closes using already-frozen same-sector ingredients only."
                if shell_anchor_bridge_without_new_free_parameters
                else "The shell-anchor route cannot close the target value, so the no-new-free-parameter bridge remains unavailable."
            ),
        },
        {
            "row_id": "same_sector_tiebreak_shell_anchor_candidate_match_count",
            "status": "inventory",
            "metric": "candidate count selected by shell-anchor route",
            "value": float(candidate_match_count),
            "note": (
                f"Matching candidate ids are {matching_candidate_ids}."
                if matching_candidate_ids
                else "No candidate can be selected through the shell-anchor route because no shell-anchor target value is available."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "shell-anchor target derivability audit",
        },
        "inputs": {
            "mass_origin_same_sector_tiebreak_target_bridge_json": _relative_str(TARGET_BRIDGE_JSON),
            "mass_origin_same_sector_tiebreak_target_source_contract_json": _relative_str(SOURCE_CONTRACT_JSON),
            "mass_origin_same_sector_tiebreak_target_source_inventory_json": _relative_str(SOURCE_INVENTORY_JSON),
        },
        "intent": "Determine whether the shell-anchor route alone can close the missing same-sector tie-break target value without introducing new free parameters.",
        "formulas": {
            "shell_anchor_route_rule": "shell_anchor_target_value_available iff the target bridge reports source_kind == surviving_shell_anchor_pack",
            "shell_anchor_no_new_parameter_rule": "shell_anchor_bridge_without_new_free_parameters iff the shell-anchor route closes and the target-source contract still requires no new free parameters",
        },
        "rows": rows,
        "summary": {
            "shell_anchor_source_kind_allowed": shell_anchor_source_kind_allowed,
            "shell_anchor_candidate_row_ids": shell_anchor_candidate_row_ids,
            "shell_anchor_candidate_count": shell_anchor_candidate_count,
            "shell_anchor_target_value_available": shell_anchor_target_value_available,
            "shell_anchor_target_source_kind_or_none": (
                "surviving_shell_anchor_pack" if shell_anchor_target_value_available else None
            ),
            "shell_anchor_bridge_without_new_free_parameters": shell_anchor_bridge_without_new_free_parameters,
            "candidate_match_count": candidate_match_count,
            "matching_candidate_ids": matching_candidate_ids,
            "tiebreak_invariant_name": bridge_summary.get("tiebreak_invariant_name"),
        },
        "decision": {
            "overall_status": overall_status,
            "keep_mass_origin_branch_blocked": True,
            "shell_anchor_target_value_available": shell_anchor_target_value_available,
            "shell_anchor_bridge_without_new_free_parameters": shell_anchor_bridge_without_new_free_parameters,
            "candidate_match_count": candidate_match_count,
            "matching_candidate_ids": matching_candidate_ids,
        },
        "evidence": {
            "target_bridge_summary": bridge_summary,
            "target_source_contract_summary": source_contract_summary,
            "target_source_inventory_summary": source_inventory_summary,
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
