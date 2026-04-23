#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_same_sector_tiebreak_target_value_closure.py

Step 8.7.55.2.95:
Combine the shell-anchor audit and the explicit mapping-equation audit into a
single closure artifact for the missing same-sector tie-break target value.

Inputs:
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_contract_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_inventory_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_shell_anchor_metrics.json
  - output/public/quantum/mass_origin_same_sector_mapping_equation_source_metrics.json

Outputs:
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_value_closure_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_value_closure_rows.csv
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
MAPPING_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_mapping_equation_source_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_value_closure_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_value_closure_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.95"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Close whether a same-sector tie-break target value is available in public canonical form.",
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
    for path in (SOURCE_CONTRACT_JSON, SOURCE_INVENTORY_JSON, SHELL_AUDIT_JSON, MAPPING_AUDIT_JSON):
        _require_path(path)

    source_contract = _read_json(SOURCE_CONTRACT_JSON)
    source_inventory = _read_json(SOURCE_INVENTORY_JSON)
    shell_audit = _read_json(SHELL_AUDIT_JSON)
    mapping_audit = _read_json(MAPPING_AUDIT_JSON)

    source_contract_summary = source_contract.get("summary", {})
    source_inventory_summary = source_inventory.get("summary", {})
    shell_summary = shell_audit.get("summary", {})
    mapping_summary = mapping_audit.get("summary", {})

    shell_anchor_target_value_available = bool(shell_summary.get("shell_anchor_target_value_available", False))
    explicit_mapping_equation_available = bool(mapping_summary.get("explicit_mapping_equation_available", False))

    # 条件分岐: `shell_anchor_target_value_available` を満たす経路を評価する。
    if shell_anchor_target_value_available:
        target_source_kind_or_none: str | None = "surviving_shell_anchor_pack"

    # 条件分岐: `explicit_mapping_equation_available` を満たす経路を評価する。
    elif explicit_mapping_equation_available:
        target_source_kind_or_none = "explicit_mapping_equation"

    else:
        target_source_kind_or_none = None

    target_value_available = target_source_kind_or_none is not None
    bridge_without_new_free_parameters = bool(
        target_value_available and source_contract_summary.get("bridge_without_new_free_parameters_required", False)
    )

    # 条件分岐: `target_source_kind_or_none == "surviving_shell_anchor_pack"` を満たす経路を評価する。
    if target_source_kind_or_none == "surviving_shell_anchor_pack":
        matching_candidate_ids = [str(item) for item in shell_summary.get("matching_candidate_ids", [])]

    # 条件分岐: `target_source_kind_or_none == "explicit_mapping_equation"` を満たす経路を評価する。
    elif target_source_kind_or_none == "explicit_mapping_equation":
        matching_candidate_ids = []

    else:
        matching_candidate_ids = []

    candidate_match_count = len(matching_candidate_ids)
    available_source_kinds = [
        kind_id
        for kind_id, available in (
            ("surviving_shell_anchor_pack", shell_anchor_target_value_available),
            ("explicit_mapping_equation", explicit_mapping_equation_available),
        )
        if available
    ]

    nonclosure_reason_or_none = None

    # 条件分岐: `not target_value_available` を満たす経路を評価する。
    if not target_value_available:
        nonclosure_reason_or_none = "shell_anchor_target_value_missing_and_explicit_mapping_equation_absent"

    overall_status = (
        "same_sector_tiebreak_target_value_closure_available"
        if target_value_available
        else "same_sector_tiebreak_target_value_closure_frozen_absent"
    )

    rows = [
        {
            "row_id": "same_sector_tiebreak_target_value_closure_complete",
            "status": "pass",
            "metric": "same-sector tie-break target value closure complete",
            "value": 1.0,
            "note": "This step combines the shell-anchor and explicit-mapping audits into a single closure decision for the tie-break target value.",
        },
        {
            "row_id": "same_sector_tiebreak_target_value_shell_anchor_route",
            "status": "pass" if shell_anchor_target_value_available else "watch",
            "metric": "shell-anchor route closes target value",
            "value": 1.0 if shell_anchor_target_value_available else 0.0,
            "note": (
                "The shell-anchor route provides the public target value."
                if shell_anchor_target_value_available
                else "The shell-anchor route remains non-closing and contributes no public target value."
            ),
        },
        {
            "row_id": "same_sector_tiebreak_target_value_explicit_mapping_route",
            "status": "pass" if explicit_mapping_equation_available else "watch",
            "metric": "explicit mapping-equation route closes target value",
            "value": 1.0 if explicit_mapping_equation_available else 0.0,
            "note": (
                "The explicit same-sector mapping equation provides the public target value."
                if explicit_mapping_equation_available
                else "The explicit mapping-equation route remains non-closing and contributes no public target value."
            ),
        },
        {
            "row_id": "same_sector_tiebreak_target_value_available",
            "status": "pass" if target_value_available else "watch",
            "metric": "same-sector tie-break target value available",
            "value": 1.0 if target_value_available else 0.0,
            "note": (
                f"Target value source kind is {target_source_kind_or_none}."
                if target_value_available
                else f"No admissible source closes the target value; available source kinds are {available_source_kinds} and nonclosure reason is {nonclosure_reason_or_none}."
            ),
        },
        {
            "row_id": "same_sector_tiebreak_target_value_bridge_without_new_free_parameters",
            "status": "pass" if bridge_without_new_free_parameters else "reject",
            "metric": "target value bridge closes without new free parameters",
            "value": 1.0 if bridge_without_new_free_parameters else 0.0,
            "note": (
                "The chosen target-value source closes within the already-frozen same-sector no-new-free-parameter contract."
                if bridge_without_new_free_parameters
                else "No admissible source currently closes the target value, so the no-new-free-parameter bridge remains unavailable."
            ),
        },
        {
            "row_id": "same_sector_tiebreak_target_value_candidate_match_count",
            "status": "inventory",
            "metric": "candidate count selected by closed target value",
            "value": float(candidate_match_count),
            "note": (
                f"Matching candidate ids are {matching_candidate_ids}."
                if matching_candidate_ids
                else "No candidate ids are selected because the target value is still absent."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "same-sector target value closure",
        },
        "inputs": {
            "mass_origin_same_sector_tiebreak_target_source_contract_json": _relative_str(SOURCE_CONTRACT_JSON),
            "mass_origin_same_sector_tiebreak_target_source_inventory_json": _relative_str(SOURCE_INVENTORY_JSON),
            "mass_origin_same_sector_tiebreak_shell_anchor_json": _relative_str(SHELL_AUDIT_JSON),
            "mass_origin_same_sector_mapping_equation_source_json": _relative_str(MAPPING_AUDIT_JSON),
        },
        "intent": "Close whether a public canonical same-sector target value now exists for the derivative-ratio tie-break invariant, and if so from which source kind.",
        "formulas": {
            "closure_rule": "target_value_available iff either the shell-anchor audit or the explicit mapping-equation audit reports an available admissible source",
            "bridge_rule": "bridge_without_new_free_parameters iff the chosen source kind closes inside the already-frozen same-sector no-new-free-parameter contract",
        },
        "rows": rows,
        "summary": {
            "target_value_available": target_value_available,
            "target_source_kind_or_none": target_source_kind_or_none,
            "bridge_without_new_free_parameters": bridge_without_new_free_parameters,
            "matching_candidate_ids": matching_candidate_ids,
            "candidate_match_count": candidate_match_count,
            "available_source_kinds": available_source_kinds,
            "nonclosure_reason_or_none": nonclosure_reason_or_none,
            "tiebreak_invariant_name": source_contract_summary.get("tiebreak_invariant_name"),
        },
        "decision": {
            "overall_status": overall_status,
            "keep_mass_origin_branch_blocked": True,
            "target_value_available": target_value_available,
            "target_source_kind_or_none": target_source_kind_or_none,
            "bridge_without_new_free_parameters": bridge_without_new_free_parameters,
            "matching_candidate_ids": matching_candidate_ids,
            "candidate_match_count": candidate_match_count,
            "nonclosure_reason_or_none": nonclosure_reason_or_none,
        },
        "evidence": {
            "target_source_contract_summary": source_contract_summary,
            "target_source_inventory_summary": source_inventory_summary,
            "shell_anchor_summary": shell_summary,
            "mapping_equation_summary": mapping_summary,
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
