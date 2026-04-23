#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_anchor_target_synthesis_contract.py

Step 8.7.55.2.98:
Freeze the minimal contract for synthesizing the same-sector tie-break target
value from the surviving shell-anchor source rows.

Inputs:
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_contract_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_inventory_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_shell_anchor_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_value_closure_metrics.json
  - output/public/quantum/mass_origin_target_source_blocker_split_contract_metrics.json

Outputs:
  - output/public/quantum/mass_origin_shell_anchor_target_synthesis_contract_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_target_synthesis_contract_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

SOURCE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_source_contract_metrics.json"
SOURCE_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_source_inventory_metrics.json"
SHELL_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_shell_anchor_metrics.json"
TARGET_VALUE_CLOSURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_value_closure_metrics.json"
BLOCKER_SPLIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_target_source_blocker_split_contract_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_target_synthesis_contract_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_target_synthesis_contract_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.98"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the shell-anchor target synthesis contract.",
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
    for path in (
        SOURCE_CONTRACT_JSON,
        SOURCE_INVENTORY_JSON,
        SHELL_AUDIT_JSON,
        TARGET_VALUE_CLOSURE_JSON,
        BLOCKER_SPLIT_JSON,
    ):
        _require_path(path)

    source_contract = _read_json(SOURCE_CONTRACT_JSON)
    source_inventory = _read_json(SOURCE_INVENTORY_JSON)
    shell_audit = _read_json(SHELL_AUDIT_JSON)
    target_value_closure = _read_json(TARGET_VALUE_CLOSURE_JSON)
    blocker_split = _read_json(BLOCKER_SPLIT_JSON)

    source_contract_summary = source_contract.get("summary", {})
    source_inventory_summary = source_inventory.get("summary", {})
    shell_summary = shell_audit.get("summary", {})
    target_value_summary = target_value_closure.get("summary", {})
    blocker_split_summary = blocker_split.get("summary", {})

    required_shell_anchor_row_ids = [
        str(row_id) for row_id in shell_summary.get("shell_anchor_candidate_row_ids", [])
    ]
    required_shell_anchor_row_fields = [
        "row_id",
        "metric",
        "value",
        "source_kind",
        "source_row_id",
    ]
    target_synthesis_formula_kind_or_none = (
        "dimensionless_two_anchor_pair_synthesis"
        if required_shell_anchor_row_ids
        else None
    )
    forbidden_backsolve_operations = [
        "cross_sector_proxy_substitution",
        "interface_only_spread_substitution",
        "phenomenological_backsolve",
        "new_fit_parameter_injection",
    ]
    shell_anchor_target_synthesis_ready = bool(
        blocker_split_summary.get("split_contract_ready", False)
        and blocker_split_summary.get("shell_anchor_route_still_admissible", False)
        and len(required_shell_anchor_row_ids) == 2
        and target_synthesis_formula_kind_or_none is not None
    )

    rows = [
        {
            "row_id": "shell_anchor_target_synthesis_contract_complete",
            "status": "pass",
            "metric": "shell-anchor target synthesis contract complete",
            "value": 1.0,
            "note": "This step freezes the minimum contract needed to attempt a target-value synthesis using only surviving shell-anchor rows.",
        },
        {
            "row_id": "shell_anchor_target_synthesis_route_admissible",
            "status": "pass" if blocker_split_summary.get("shell_anchor_route_still_admissible", False) else "reject",
            "metric": "shell-anchor target synthesis route remains admissible",
            "value": 1.0 if blocker_split_summary.get("shell_anchor_route_still_admissible", False) else 0.0,
            "note": "The shell-anchor route remains allowed under the frozen same-sector / no-new-free-parameter contract.",
        },
        {
            "row_id": "shell_anchor_target_synthesis_required_row_count",
            "status": "inventory",
            "metric": "required shell-anchor source row count",
            "value": float(len(required_shell_anchor_row_ids)),
            "note": f"Required shell-anchor row ids are {required_shell_anchor_row_ids}.",
        },
        {
            "row_id": "shell_anchor_target_synthesis_required_field_count",
            "status": "inventory",
            "metric": "required shell-anchor row field count",
            "value": float(len(required_shell_anchor_row_fields)),
            "note": f"Required shell-anchor row fields are {required_shell_anchor_row_fields}.",
        },
        {
            "row_id": "shell_anchor_target_synthesis_formula_kind_frozen",
            "status": "pass" if target_synthesis_formula_kind_or_none is not None else "reject",
            "metric": "target synthesis formula kind frozen",
            "value": 1.0 if target_synthesis_formula_kind_or_none is not None else 0.0,
            "note": (
                f"Allowed synthesis formula kind is {target_synthesis_formula_kind_or_none}."
                if target_synthesis_formula_kind_or_none is not None
                else "No admissible shell-anchor synthesis formula kind is available."
            ),
        },
        {
            "row_id": "shell_anchor_target_synthesis_forbidden_backsolve_operation_count",
            "status": "inventory",
            "metric": "forbidden backsolve operation count",
            "value": float(len(forbidden_backsolve_operations)),
            "note": f"Forbidden backsolve operations are {forbidden_backsolve_operations}.",
        },
        {
            "row_id": "shell_anchor_target_synthesis_ready",
            "status": "pass" if shell_anchor_target_synthesis_ready else "reject",
            "metric": "shell-anchor target synthesis contract ready for audit",
            "value": 1.0 if shell_anchor_target_synthesis_ready else 0.0,
            "note": (
                "The next step may audit whether the required rows actually synthesize a dimensionless target value without new free parameters."
                if shell_anchor_target_synthesis_ready
                else "The shell-anchor synthesis prerequisites are not stable enough for the next audit."
            ),
        },
        {
            "row_id": "shell_anchor_target_synthesis_target_value_still_absent",
            "status": "watch" if not target_value_summary.get("target_value_available", False) else "pass",
            "metric": "current target value still absent before synthesis audit",
            "value": 0.0 if not target_value_summary.get("target_value_available", False) else 1.0,
            "note": (
                "The contract prepares the synthesis audit, but the current public canonical pack still has no closed target value."
                if not target_value_summary.get("target_value_available", False)
                else "A target value is already available before synthesis audit."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "shell-anchor target synthesis contract",
        },
        "inputs": {
            "mass_origin_same_sector_tiebreak_target_source_contract_json": _relative_str(SOURCE_CONTRACT_JSON),
            "mass_origin_same_sector_tiebreak_target_source_inventory_json": _relative_str(SOURCE_INVENTORY_JSON),
            "mass_origin_same_sector_tiebreak_shell_anchor_json": _relative_str(SHELL_AUDIT_JSON),
            "mass_origin_same_sector_tiebreak_target_value_closure_json": _relative_str(TARGET_VALUE_CLOSURE_JSON),
            "mass_origin_target_source_blocker_split_contract_json": _relative_str(BLOCKER_SPLIT_JSON),
        },
        "intent": "Freeze the minimum shell-anchor-only contract needed to attempt target-value synthesis in the next audit.",
        "formulas": {
            "synthesis_contract_rule": "shell-anchor target synthesis is admissible only if it uses the frozen surviving shell-anchor row pair and keeps the target value dimensionless with no new free parameters",
            "forbidden_backsolve_rule": "the shell-anchor route may not substitute cross-sector proxies, interface-only spreads, phenomenological backsolves, or new fit parameters",
        },
        "rows": rows,
        "summary": {
            "required_shell_anchor_row_ids": required_shell_anchor_row_ids,
            "required_shell_anchor_row_fields": required_shell_anchor_row_fields,
            "target_synthesis_formula_kind_or_none": target_synthesis_formula_kind_or_none,
            "forbidden_backsolve_operations": forbidden_backsolve_operations,
            "shell_anchor_target_synthesis_ready": shell_anchor_target_synthesis_ready,
            "shell_anchor_route_still_admissible": blocker_split_summary.get("shell_anchor_route_still_admissible", False),
            "target_value_available_before_synthesis_audit": target_value_summary.get("target_value_available", False),
            "candidate_source_count": source_inventory_summary.get("candidate_source_count"),
        },
        "decision": {
            "overall_status": "shell_anchor_target_synthesis_contract_frozen",
            "keep_mass_origin_branch_blocked": True,
            "required_shell_anchor_row_ids": required_shell_anchor_row_ids,
            "required_shell_anchor_row_fields": required_shell_anchor_row_fields,
            "target_synthesis_formula_kind_or_none": target_synthesis_formula_kind_or_none,
            "forbidden_backsolve_operations": forbidden_backsolve_operations,
            "shell_anchor_target_synthesis_ready": shell_anchor_target_synthesis_ready,
        },
        "evidence": {
            "target_source_contract_summary": source_contract_summary,
            "target_source_inventory_summary": source_inventory_summary,
            "shell_anchor_summary": shell_summary,
            "target_value_closure_summary": target_value_summary,
            "blocker_split_summary": blocker_split_summary,
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
