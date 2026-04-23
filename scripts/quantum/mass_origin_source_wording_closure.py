#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_source_wording_closure.py

Step 8.7.55.2.107:
Combine the shell-anchor semantic bridge audit and the explicit mapping
literal lift closure into a single source-wording closure artifact for the
same-sector tie-break target value.

Inputs:
  - output/public/quantum/mass_origin_shell_anchor_semantic_bridge_contract_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_semantic_bridge_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_literal_fragment_inventory_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_literal_lift_metrics.json

Outputs:
  - output/public/quantum/mass_origin_source_wording_closure_metrics.json
  - output/public/quantum/mass_origin_source_wording_closure_rows.csv
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
LITERAL_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_literal_fragment_inventory_metrics.json"
LITERAL_LIFT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_literal_lift_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_source_wording_closure_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_source_wording_closure_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.107"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Close whether the source-wording layer already yields a same-sector tie-break target value.",
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


# 関数: `_ordered_unique` の入出力契約と処理意図を定義する。

def _ordered_unique(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []

    for value in values:
        # 条件分岐: `value and value not in seen` を満たす経路を評価する。
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)

    return ordered


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (
        SHELL_CONTRACT_JSON,
        SHELL_BRIDGE_JSON,
        LITERAL_INVENTORY_JSON,
        LITERAL_LIFT_JSON,
    ):
        _require_path(path)

    shell_contract = _read_json(SHELL_CONTRACT_JSON)
    shell_bridge = _read_json(SHELL_BRIDGE_JSON)
    literal_inventory = _read_json(LITERAL_INVENTORY_JSON)
    literal_lift = _read_json(LITERAL_LIFT_JSON)

    shell_contract_summary = shell_contract.get("summary", {})
    shell_bridge_summary = shell_bridge.get("summary", {})
    literal_inventory_summary = literal_inventory.get("summary", {})
    literal_lift_summary = literal_lift.get("summary", {})

    shell_anchor_semantic_bridge_available = bool(shell_bridge_summary.get("semantic_bridge_available", False))
    explicit_mapping_literal_lift_available = bool(literal_lift_summary.get("explicit_mapping_equation_available", False))

    # 条件分岐: `shell_anchor_semantic_bridge_available` を満たす経路を評価する。
    if shell_anchor_semantic_bridge_available:
        target_source_kind_or_none: str | None = "surviving_shell_anchor_pack"
        bridge_without_new_free_parameters = bool(
            shell_bridge_summary.get("semantic_bridge_without_new_free_parameters", False)
        )

    # 条件分岐: `explicit_mapping_literal_lift_available` を満たす経路を評価する。
    elif explicit_mapping_literal_lift_available:
        target_source_kind_or_none = "explicit_mapping_equation"
        bridge_without_new_free_parameters = bool(
            literal_lift_summary.get("mapping_without_new_free_parameters", False)
        )

    else:
        target_source_kind_or_none = None
        bridge_without_new_free_parameters = False

    same_sector_tiebreak_target_value_available = target_source_kind_or_none is not None
    available_source_kinds = [
        source_kind
        for source_kind, is_available in (
            ("surviving_shell_anchor_pack", shell_anchor_semantic_bridge_available),
            ("explicit_mapping_equation", explicit_mapping_literal_lift_available),
        )
        if is_available
    ]

    route_nonclosure_reasons_by_source_kind = {
        "surviving_shell_anchor_pack": str(
            shell_bridge_summary.get("semantic_bridge_nonclosure_reason_or_none")
            or "shell_anchor_semantic_bridge_absent"
        ),
        "explicit_mapping_equation": str(
            literal_lift_summary.get("literal_lift_nonclosure_reason_or_none")
            or "explicit_mapping_literal_lift_absent"
        ),
    }

    remaining_source_level_blockers: List[str] = []

    # 条件分岐: `not shell_anchor_semantic_bridge_available` を満たす経路を評価する。
    if not shell_anchor_semantic_bridge_available:
        remaining_source_level_blockers.append("shell_anchor_semantic_bridge_absent")

    # 条件分岐: `not explicit_mapping_literal_lift_available` を満たす経路を評価する。

    if not explicit_mapping_literal_lift_available:
        remaining_source_level_blockers.append("explicit_mapping_literal_lift_absent")

    remaining_source_level_blockers = _ordered_unique(remaining_source_level_blockers)

    rows = [
        {
            "row_id": "source_wording_closure_complete",
            "status": "pass",
            "metric": "source-wording closure complete",
            "value": 1.0,
            "note": "This closure artifact combines the shell-anchor semantic bridge route and explicit mapping literal lift route into one source-wording decision for the same-sector tie-break target value.",
        },
        {
            "row_id": "source_wording_shell_anchor_route",
            "status": "pass" if shell_anchor_semantic_bridge_available else "watch",
            "metric": "shell-anchor semantic bridge route closes source-wording target value",
            "value": 1.0 if shell_anchor_semantic_bridge_available else 0.0,
            "note": (
                "The shell-anchor route now provides the public canonical source-wording bridge."
                if shell_anchor_semantic_bridge_available
                else f"Shell-anchor route remains non-closing because {route_nonclosure_reasons_by_source_kind['surviving_shell_anchor_pack']}."
            ),
        },
        {
            "row_id": "source_wording_explicit_mapping_route",
            "status": "pass" if explicit_mapping_literal_lift_available else "watch",
            "metric": "explicit mapping literal-lift route closes source-wording target value",
            "value": 1.0 if explicit_mapping_literal_lift_available else 0.0,
            "note": (
                "The explicit same-sector mapping equation now provides the public canonical source-wording bridge."
                if explicit_mapping_literal_lift_available
                else f"Explicit mapping route remains non-closing because {route_nonclosure_reasons_by_source_kind['explicit_mapping_equation']}."
            ),
        },
        {
            "row_id": "same_sector_tiebreak_target_value_available",
            "status": "pass" if same_sector_tiebreak_target_value_available else "watch",
            "metric": "same-sector tie-break target value available after source-wording closure",
            "value": 1.0 if same_sector_tiebreak_target_value_available else 0.0,
            "note": (
                f"Target source kind is {target_source_kind_or_none}."
                if same_sector_tiebreak_target_value_available
                else f"Source-wording closure remains blocked by {remaining_source_level_blockers}; available source kinds are {available_source_kinds}."
            ),
        },
        {
            "row_id": "source_wording_bridge_without_new_free_parameters",
            "status": "pass" if bridge_without_new_free_parameters else "reject",
            "metric": "source-wording closure bridges without new free parameters",
            "value": 1.0 if bridge_without_new_free_parameters else 0.0,
            "note": (
                "The chosen source-wording route closes within the already-frozen same-sector no-new-free-parameter contract."
                if bridge_without_new_free_parameters
                else "No source-wording route currently closes the target value without introducing new free parameters."
            ),
        },
        {
            "row_id": "source_wording_remaining_source_level_blocker_count",
            "status": "inventory",
            "metric": "remaining source-level blocker count after source-wording closure",
            "value": float(len(remaining_source_level_blockers)),
            "note": (
                f"Remaining source-level blockers are {remaining_source_level_blockers}."
                if remaining_source_level_blockers
                else "No source-level blockers remain after source-wording closure."
            ),
        },
    ]

    overall_status = (
        "source_wording_closure_available"
        if same_sector_tiebreak_target_value_available
        else "source_wording_closure_frozen_absent"
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "source-wording closure",
        },
        "inputs": {
            "mass_origin_shell_anchor_semantic_bridge_contract_json": _relative_str(SHELL_CONTRACT_JSON),
            "mass_origin_shell_anchor_semantic_bridge_json": _relative_str(SHELL_BRIDGE_JSON),
            "mass_origin_explicit_mapping_literal_fragment_inventory_json": _relative_str(LITERAL_INVENTORY_JSON),
            "mass_origin_explicit_mapping_literal_lift_json": _relative_str(LITERAL_LIFT_JSON),
        },
        "intent": "Close whether the current public canonical pack already carries enough source-wording to support a same-sector tie-break target value.",
        "formulas": {
            "closure_rule": "same_sector_tiebreak_target_value_available iff either the shell-anchor semantic bridge route or the explicit mapping literal-lift route closes in public canonical form",
            "bridge_rule": "bridge_without_new_free_parameters iff the chosen source-wording route closes without violating the frozen same-sector no-new-free-parameter contract",
        },
        "rows": rows,
        "summary": {
            "same_sector_tiebreak_target_value_available": same_sector_tiebreak_target_value_available,
            "target_source_kind_or_none": target_source_kind_or_none,
            "bridge_without_new_free_parameters": bridge_without_new_free_parameters,
            "remaining_source_level_blockers": remaining_source_level_blockers,
            "available_source_kinds": available_source_kinds,
            "route_nonclosure_reasons_by_source_kind": route_nonclosure_reasons_by_source_kind,
            "missing_shell_anchor_relation_slots": shell_bridge_summary.get("missing_relation_slots", []),
            "missing_explicit_mapping_literal_fragments": literal_inventory_summary.get("missing_literal_fragments", []),
        },
        "decision": {
            "overall_status": overall_status,
            "keep_mass_origin_branch_blocked": True,
            "same_sector_tiebreak_target_value_available": same_sector_tiebreak_target_value_available,
            "target_source_kind_or_none": target_source_kind_or_none,
            "bridge_without_new_free_parameters": bridge_without_new_free_parameters,
            "remaining_source_level_blockers": remaining_source_level_blockers,
        },
        "evidence": {
            "shell_anchor_semantic_bridge_contract_summary": shell_contract_summary,
            "shell_anchor_semantic_bridge_summary": shell_bridge_summary,
            "explicit_mapping_literal_fragment_inventory_summary": literal_inventory_summary,
            "explicit_mapping_literal_lift_summary": literal_lift_summary,
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
