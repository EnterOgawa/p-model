#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_source_wording_blocker_split_contract.py

Step 8.7.55.2.109:
Freeze the reduced source-wording blocker split after the source-wording
branch refresh stays blocked at 8.7.55.2.108.

Inputs:
  - output/public/quantum/mass_origin_shell_anchor_semantic_bridge_contract_metrics.json
  - output/public/quantum/mass_origin_shell_anchor_semantic_bridge_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_equation_lift_contract_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_literal_lift_metrics.json
  - output/public/quantum/mass_origin_source_wording_closure_metrics.json
  - output/public/quantum/mass_origin_source_wording_branch_refresh_metrics.json

Outputs:
  - output/public/quantum/mass_origin_source_wording_blocker_split_contract_metrics.json
  - output/public/quantum/mass_origin_source_wording_blocker_split_contract_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]

SHELL_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_semantic_bridge_contract_metrics.json"
SHELL_BRIDGE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_semantic_bridge_metrics.json"
MAPPING_LIFT_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_equation_lift_contract_metrics.json"
LITERAL_LIFT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_literal_lift_metrics.json"
SOURCE_WORDING_CLOSURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_source_wording_closure_metrics.json"
BRANCH_REFRESH_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_source_wording_branch_refresh_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_source_wording_blocker_split_contract_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_source_wording_blocker_split_contract_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.109"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the reduced source-wording blocker split after .108 remains blocked.",
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
        SHELL_CONTRACT_JSON,
        SHELL_BRIDGE_JSON,
        MAPPING_LIFT_CONTRACT_JSON,
        LITERAL_LIFT_JSON,
        SOURCE_WORDING_CLOSURE_JSON,
        BRANCH_REFRESH_JSON,
    ):
        _require_path(path)

    shell_contract = _read_json(SHELL_CONTRACT_JSON)
    shell_bridge = _read_json(SHELL_BRIDGE_JSON)
    mapping_lift_contract = _read_json(MAPPING_LIFT_CONTRACT_JSON)
    literal_lift = _read_json(LITERAL_LIFT_JSON)
    source_wording_closure = _read_json(SOURCE_WORDING_CLOSURE_JSON)
    branch_refresh = _read_json(BRANCH_REFRESH_JSON)

    shell_contract_summary = shell_contract.get("summary", {})
    shell_bridge_summary = shell_bridge.get("summary", {})
    mapping_lift_contract_summary = mapping_lift_contract.get("summary", {})
    literal_lift_summary = literal_lift.get("summary", {})
    source_wording_closure_summary = source_wording_closure.get("summary", {})
    branch_refresh_summary = branch_refresh.get("summary", {})

    remaining_source_level_blockers = _ordered_unique(
        [str(item) for item in branch_refresh_summary.get("remaining_source_level_blockers", [])]
    )
    remaining_missing_artifacts = _ordered_unique(
        [str(item) for item in branch_refresh_summary.get("remaining_missing_artifacts", [])]
    )
    shell_anchor_wording_route_still_admissible = bool(
        shell_contract_summary.get("semantic_bridge_route_admissible", False)
    )
    explicit_mapping_wording_route_still_admissible = bool(
        mapping_lift_contract_summary.get("mapping_equation_lift_ready", False)
    )
    hand_off_to_8_7_55_2_83 = bool(branch_refresh_summary.get("hand_off_to_8_7_55_2_83", False))
    split_contract_ready = bool(
        not hand_off_to_8_7_55_2_83
        and remaining_source_level_blockers == [
            "shell_anchor_semantic_bridge_absent",
            "explicit_mapping_literal_lift_absent",
        ]
        and shell_anchor_wording_route_still_admissible
        and explicit_mapping_wording_route_still_admissible
    )

    rows = [
        {
            "row_id": "source_wording_blocker_split_contract_complete",
            "status": "pass",
            "metric": "source-wording blocker split contract complete",
            "value": 1.0,
            "note": "This step freezes the reduced source-wording blocker split that remains after the .108 refresh stayed blocked.",
        },
        {
            "row_id": "source_wording_blocker_split_handoff_still_blocked",
            "status": "pass" if not hand_off_to_8_7_55_2_83 else "reject",
            "metric": "handoff to 8.7.55.2.83 still blocked after source-wording refresh",
            "value": 1.0 if not hand_off_to_8_7_55_2_83 else 0.0,
            "note": "The new branch is only needed because .108 still returned hand_off_to_8_7_55_2_83=false.",
        },
        {
            "row_id": "source_wording_blocker_split_shell_anchor_route_admissible",
            "status": "pass" if shell_anchor_wording_route_still_admissible else "reject",
            "metric": "shell-anchor wording route remains admissible",
            "value": 1.0 if shell_anchor_wording_route_still_admissible else 0.0,
            "note": (
                "The shell-anchor semantic bridge route remains admissible even though its wording bridge is still absent."
                if shell_anchor_wording_route_still_admissible
                else "The shell-anchor semantic bridge route is no longer admissible for the next branch."
            ),
        },
        {
            "row_id": "source_wording_blocker_split_mapping_route_admissible",
            "status": "pass" if explicit_mapping_wording_route_still_admissible else "reject",
            "metric": "explicit mapping wording route remains admissible",
            "value": 1.0 if explicit_mapping_wording_route_still_admissible else 0.0,
            "note": (
                "The explicit mapping literal-lift route remains admissible even though its equation wording is still absent."
                if explicit_mapping_wording_route_still_admissible
                else "The explicit mapping literal-lift route is no longer admissible for the next branch."
            ),
        },
        {
            "row_id": "source_wording_blocker_split_source_level_blocker_count",
            "status": "inventory",
            "metric": "remaining source-level blocker count",
            "value": float(len(remaining_source_level_blockers)),
            "note": f"Remaining source-level blockers: {', '.join(remaining_source_level_blockers)}.",
        },
        {
            "row_id": "source_wording_blocker_split_artifact_level_missing_count",
            "status": "inventory",
            "metric": "remaining artifact-level missing count",
            "value": float(len(remaining_missing_artifacts)),
            "note": f"Remaining artifact-level missing items: {', '.join(remaining_missing_artifacts)}.",
        },
        {
            "row_id": "source_wording_blocker_split_contract_ready",
            "status": "pass" if split_contract_ready else "reject",
            "metric": "reduced source-wording blocker split contract ready for next branch",
            "value": 1.0 if split_contract_ready else 0.0,
            "note": (
                "The next branch may now treat shell-anchor wording closure and explicit mapping wording closure as separate source-level routes."
                if split_contract_ready
                else "The reduced source-wording blocker split is not stable enough to launch the next branch."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "source-wording blocker split contract",
        },
        "inputs": {
            "mass_origin_shell_anchor_semantic_bridge_contract_json": _relative_str(SHELL_CONTRACT_JSON),
            "mass_origin_shell_anchor_semantic_bridge_json": _relative_str(SHELL_BRIDGE_JSON),
            "mass_origin_explicit_mapping_equation_lift_contract_json": _relative_str(MAPPING_LIFT_CONTRACT_JSON),
            "mass_origin_explicit_mapping_literal_lift_json": _relative_str(LITERAL_LIFT_JSON),
            "mass_origin_source_wording_closure_json": _relative_str(SOURCE_WORDING_CLOSURE_JSON),
            "mass_origin_source_wording_branch_refresh_json": _relative_str(BRANCH_REFRESH_JSON),
        },
        "intent": "Freeze the reduced source-wording blocker split so the next branch can attack shell-anchor wording closure and explicit mapping wording closure separately.",
        "formulas": {
            "split_rule": "the next branch may split iff .108 still blocks handoff and the residual source-level blocker set is exactly {shell_anchor_semantic_bridge_absent, explicit_mapping_literal_lift_absent}",
            "route_rule": "both split routes must remain admissible under the frozen same-sector / no-new-free-parameter contract",
        },
        "rows": rows,
        "summary": {
            "remaining_source_level_blockers": remaining_source_level_blockers,
            "remaining_missing_artifacts": remaining_missing_artifacts,
            "shell_anchor_wording_route_still_admissible": shell_anchor_wording_route_still_admissible,
            "explicit_mapping_wording_route_still_admissible": explicit_mapping_wording_route_still_admissible,
            "hand_off_to_8_7_55_2_83": hand_off_to_8_7_55_2_83,
            "split_contract_ready": split_contract_ready,
            "same_sector_tiebreak_target_value_available": source_wording_closure_summary.get("same_sector_tiebreak_target_value_available"),
            "target_source_kind_or_none": source_wording_closure_summary.get("target_source_kind_or_none"),
        },
        "decision": {
            "overall_status": "source_wording_blocker_split_contract_frozen",
            "keep_mass_origin_branch_blocked": True,
            "split_contract_ready": split_contract_ready,
            "hand_off_to_8_7_55_2_83": hand_off_to_8_7_55_2_83,
            "remaining_source_level_blockers": remaining_source_level_blockers,
            "remaining_missing_artifacts": remaining_missing_artifacts,
        },
        "evidence": {
            "shell_anchor_semantic_bridge_contract_summary": shell_contract_summary,
            "shell_anchor_semantic_bridge_summary": shell_bridge_summary,
            "explicit_mapping_equation_lift_contract_summary": mapping_lift_contract_summary,
            "explicit_mapping_literal_lift_summary": literal_lift_summary,
            "source_wording_closure_summary": source_wording_closure_summary,
            "source_wording_branch_refresh_summary": branch_refresh_summary,
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
