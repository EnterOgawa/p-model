#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_target_source_blocker_split_contract.py

Step 8.7.55.2.97:
Freeze the reduced source-level blocker split after the target-source branch
refresh stays blocked at 8.7.55.2.96.

Inputs:
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_contract_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_source_inventory_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_shell_anchor_metrics.json
  - output/public/quantum/mass_origin_same_sector_mapping_equation_source_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_value_closure_metrics.json
  - output/public/quantum/mass_origin_target_source_branch_refresh_metrics.json

Outputs:
  - output/public/quantum/mass_origin_target_source_blocker_split_contract_metrics.json
  - output/public/quantum/mass_origin_target_source_blocker_split_contract_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]

SOURCE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_source_contract_metrics.json"
SOURCE_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_source_inventory_metrics.json"
SHELL_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_shell_anchor_metrics.json"
MAPPING_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_mapping_equation_source_metrics.json"
TARGET_VALUE_CLOSURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_value_closure_metrics.json"
BRANCH_REFRESH_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_target_source_branch_refresh_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_target_source_blocker_split_contract_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_target_source_blocker_split_contract_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.97"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the reduced source-level blocker split after .96 remains blocked.",
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

def _ordered_unique(values: Iterable[str]) -> List[str]:
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
        SOURCE_CONTRACT_JSON,
        SOURCE_INVENTORY_JSON,
        SHELL_AUDIT_JSON,
        MAPPING_AUDIT_JSON,
        TARGET_VALUE_CLOSURE_JSON,
        BRANCH_REFRESH_JSON,
    ):
        _require_path(path)

    source_contract = _read_json(SOURCE_CONTRACT_JSON)
    source_inventory = _read_json(SOURCE_INVENTORY_JSON)
    shell_audit = _read_json(SHELL_AUDIT_JSON)
    mapping_audit = _read_json(MAPPING_AUDIT_JSON)
    target_value_closure = _read_json(TARGET_VALUE_CLOSURE_JSON)
    branch_refresh = _read_json(BRANCH_REFRESH_JSON)

    source_contract_summary = source_contract.get("summary", {})
    source_inventory_summary = source_inventory.get("summary", {})
    shell_summary = shell_audit.get("summary", {})
    mapping_summary = mapping_audit.get("summary", {})
    target_value_summary = target_value_closure.get("summary", {})
    branch_refresh_summary = branch_refresh.get("summary", {})

    remaining_source_level_blockers = _ordered_unique(branch_refresh_summary.get("remaining_source_level_blockers", []))
    remaining_missing_artifacts = _ordered_unique(branch_refresh_summary.get("remaining_missing_artifacts", []))
    shell_anchor_route_still_admissible = bool(shell_summary.get("shell_anchor_source_kind_allowed", False))
    explicit_mapping_route_still_admissible = bool(mapping_summary.get("explicit_mapping_source_kind_allowed", False))
    hand_off_to_8_7_55_2_83 = bool(branch_refresh_summary.get("hand_off_to_8_7_55_2_83", False))
    split_contract_ready = (
        not hand_off_to_8_7_55_2_83
        and len(remaining_source_level_blockers) == 2
        and shell_anchor_route_still_admissible
        and explicit_mapping_route_still_admissible
    )

    rows = [
        {
            "row_id": "target_source_blocker_split_contract_complete",
            "status": "pass",
            "metric": "target-source blocker split contract complete",
            "value": 1.0,
            "note": "This step freezes the reduced blocker split that remains after the .96 refresh stayed blocked.",
        },
        {
            "row_id": "target_source_blocker_split_handoff_still_blocked",
            "status": "pass" if not hand_off_to_8_7_55_2_83 else "reject",
            "metric": "handoff to 8.7.55.2.83 still blocked",
            "value": 1.0 if not hand_off_to_8_7_55_2_83 else 0.0,
            "note": "The new branch is only needed because .96 still returned hand_off_to_8_7_55_2_83=false.",
        },
        {
            "row_id": "target_source_blocker_split_shell_anchor_route_admissible",
            "status": "pass" if shell_anchor_route_still_admissible else "reject",
            "metric": "shell-anchor source route remains admissible",
            "value": 1.0 if shell_anchor_route_still_admissible else 0.0,
            "note": "The surviving shell-anchor pack remains an allowed source class even though it still lacks a target value.",
        },
        {
            "row_id": "target_source_blocker_split_mapping_route_admissible",
            "status": "pass" if explicit_mapping_route_still_admissible else "reject",
            "metric": "explicit mapping-equation route remains admissible",
            "value": 1.0 if explicit_mapping_route_still_admissible else 0.0,
            "note": "The explicit mapping-equation source class remains allowed even though the public equation row is still absent.",
        },
        {
            "row_id": "target_source_blocker_split_source_level_blocker_count",
            "status": "inventory",
            "metric": "remaining source-level blocker count",
            "value": float(len(remaining_source_level_blockers)),
            "note": f"Remaining source-level blockers: {', '.join(remaining_source_level_blockers)}.",
        },
        {
            "row_id": "target_source_blocker_split_artifact_level_missing_count",
            "status": "inventory",
            "metric": "remaining artifact-level missing count",
            "value": float(len(remaining_missing_artifacts)),
            "note": f"Remaining artifact-level missing items: {', '.join(remaining_missing_artifacts)}.",
        },
        {
            "row_id": "target_source_blocker_split_contract_ready",
            "status": "pass" if split_contract_ready else "reject",
            "metric": "reduced blocker split contract ready for next branch",
            "value": 1.0 if split_contract_ready else 0.0,
            "note": (
                "The next branch may now treat shell-anchor target synthesis and explicit mapping-equation lift as separate source-level routes."
                if split_contract_ready
                else "The reduced blocker split is not stable enough to launch the next branch."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "target-source blocker split contract",
        },
        "inputs": {
            "mass_origin_same_sector_tiebreak_target_source_contract_json": _relative_str(SOURCE_CONTRACT_JSON),
            "mass_origin_same_sector_tiebreak_target_source_inventory_json": _relative_str(SOURCE_INVENTORY_JSON),
            "mass_origin_same_sector_tiebreak_shell_anchor_json": _relative_str(SHELL_AUDIT_JSON),
            "mass_origin_same_sector_mapping_equation_source_json": _relative_str(MAPPING_AUDIT_JSON),
            "mass_origin_same_sector_tiebreak_target_value_closure_json": _relative_str(TARGET_VALUE_CLOSURE_JSON),
            "mass_origin_target_source_branch_refresh_json": _relative_str(BRANCH_REFRESH_JSON),
        },
        "intent": "Freeze the reduced source-level blocker split so the next branch can attack shell-anchor target synthesis and explicit mapping-equation lift separately.",
        "formulas": {
            "split_rule": "the next branch may split iff .96 still blocks handoff and the residual source-level blocker set is exactly {shell_anchor_target_value_missing, explicit_mapping_equation_absent}",
            "route_rule": "both split routes must remain admissible under the frozen same-sector / no-new-free-parameter contract",
        },
        "rows": rows,
        "summary": {
            "remaining_source_level_blockers": remaining_source_level_blockers,
            "remaining_missing_artifacts": remaining_missing_artifacts,
            "shell_anchor_route_still_admissible": shell_anchor_route_still_admissible,
            "explicit_mapping_route_still_admissible": explicit_mapping_route_still_admissible,
            "hand_off_to_8_7_55_2_83": hand_off_to_8_7_55_2_83,
            "split_contract_ready": split_contract_ready,
            "candidate_source_count": source_inventory_summary.get("candidate_source_count"),
            "target_value_available": target_value_summary.get("target_value_available"),
            "current_target_value_available_from_contract": source_contract_summary.get("current_target_value_available"),
        },
        "decision": {
            "overall_status": "target_source_blocker_split_contract_frozen",
            "keep_mass_origin_branch_blocked": True,
            "split_contract_ready": split_contract_ready,
            "hand_off_to_8_7_55_2_83": hand_off_to_8_7_55_2_83,
            "remaining_source_level_blockers": remaining_source_level_blockers,
            "remaining_missing_artifacts": remaining_missing_artifacts,
        },
        "evidence": {
            "target_source_contract_summary": source_contract_summary,
            "target_source_inventory_summary": source_inventory_summary,
            "shell_anchor_summary": shell_summary,
            "mapping_equation_summary": mapping_summary,
            "target_value_closure_summary": target_value_summary,
            "branch_refresh_summary": branch_refresh_summary,
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
