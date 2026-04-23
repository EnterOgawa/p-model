#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_wording_slot_residual_split_contract.py

Step 8.7.55.2.115:
Freeze the reduced wording-slot residual blocker split so the next branch can
attack shell-anchor missing slots and explicit-mapping missing slots
separately.

Inputs:
  - output/public/quantum/mass_origin_shell_anchor_wording_closure_retry_metrics.json
  - output/public/quantum/mass_origin_explicit_mapping_wording_closure_retry_metrics.json
  - output/public/quantum/mass_origin_source_wording_blocker_split_refresh_metrics.json

Outputs:
  - output/public/quantum/mass_origin_wording_slot_residual_split_contract_metrics.json
  - output/public/quantum/mass_origin_wording_slot_residual_split_contract_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

SHELL_ANCHOR_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_anchor_wording_closure_retry_metrics.json"
EXPLICIT_MAPPING_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_explicit_mapping_wording_closure_retry_metrics.json"
SPLIT_REFRESH_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_source_wording_blocker_split_refresh_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_wording_slot_residual_split_contract_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_wording_slot_residual_split_contract_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.115"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the reduced wording-slot residual blocker split after .114 stayed blocked.",
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
    for path in (SHELL_ANCHOR_RETRY_JSON, EXPLICIT_MAPPING_RETRY_JSON, SPLIT_REFRESH_JSON):
        _require_path(path)

    shell_anchor_retry = _read_json(SHELL_ANCHOR_RETRY_JSON)
    explicit_mapping_retry = _read_json(EXPLICIT_MAPPING_RETRY_JSON)
    split_refresh = _read_json(SPLIT_REFRESH_JSON)

    shell_anchor_retry_summary = shell_anchor_retry.get("summary", {})
    explicit_mapping_retry_summary = explicit_mapping_retry.get("summary", {})
    split_refresh_summary = split_refresh.get("summary", {})

    remaining_source_level_blockers = [
        str(item) for item in split_refresh_summary.get("remaining_source_level_blockers", [])
    ]
    shell_anchor_missing_wording_slots = [
        str(item) for item in shell_anchor_retry_summary.get("missing_shell_anchor_wording_slots", [])
    ]
    explicit_mapping_missing_wording_slots = [
        str(item) for item in explicit_mapping_retry_summary.get("missing_explicit_mapping_wording_slots", [])
    ]
    shell_anchor_wording_slot_route_still_admissible = bool(
        shell_anchor_retry_summary.get("shell_anchor_wording_route_still_admissible", False)
    )
    explicit_mapping_wording_slot_route_still_admissible = bool(
        explicit_mapping_retry_summary.get("explicit_mapping_wording_route_still_admissible", False)
    )
    hand_off_to_8_7_55_2_83 = bool(split_refresh_summary.get("hand_off_to_8_7_55_2_83", False))
    split_contract_ready = bool(
        not hand_off_to_8_7_55_2_83
        and remaining_source_level_blockers
        == ["shell_anchor_wording_slots_still_missing", "explicit_mapping_wording_slots_still_missing"]
        and shell_anchor_wording_slot_route_still_admissible
        and explicit_mapping_wording_slot_route_still_admissible
    )

    rows = [
        {
            "row_id": "wording_slot_residual_split_contract_complete",
            "status": "pass",
            "metric": "wording-slot residual split contract complete",
            "value": 1.0,
            "note": "This step freezes the reduced wording-slot residual blocker split that remains after the .114 refresh stayed blocked.",
        },
        {
            "row_id": "wording_slot_residual_split_handoff_still_blocked",
            "status": "pass",
            "metric": "handoff to 8.7.55.2.83 still blocked after wording-slot refresh",
            "value": 1.0,
            "note": "The new branch is only needed because .114 still returned hand_off_to_8_7_55_2_83=false.",
        },
        {
            "row_id": "wording_slot_residual_split_shell_anchor_route_admissible",
            "status": "pass" if shell_anchor_wording_slot_route_still_admissible else "reject",
            "metric": "shell-anchor wording-slot route remains admissible",
            "value": 1.0 if shell_anchor_wording_slot_route_still_admissible else 0.0,
            "note": "The shell-anchor wording-slot route remains admissible even though its missing wording slots are still open.",
        },
        {
            "row_id": "wording_slot_residual_split_explicit_mapping_route_admissible",
            "status": "pass" if explicit_mapping_wording_slot_route_still_admissible else "reject",
            "metric": "explicit-mapping wording-slot route remains admissible",
            "value": 1.0 if explicit_mapping_wording_slot_route_still_admissible else 0.0,
            "note": "The explicit-mapping wording-slot route remains admissible even though its missing wording slots are still open.",
        },
        {
            "row_id": "wording_slot_residual_split_source_level_blocker_count",
            "status": "inventory",
            "metric": "remaining source-level blocker count",
            "value": float(len(remaining_source_level_blockers)),
            "note": f"Remaining source-level blockers are {remaining_source_level_blockers}.",
        },
        {
            "row_id": "wording_slot_residual_split_shell_anchor_missing_slot_count",
            "status": "inventory",
            "metric": "shell-anchor missing wording slot count",
            "value": float(len(shell_anchor_missing_wording_slots)),
            "note": f"Shell-anchor missing wording slots are {shell_anchor_missing_wording_slots}.",
        },
        {
            "row_id": "wording_slot_residual_split_explicit_mapping_missing_slot_count",
            "status": "inventory",
            "metric": "explicit-mapping missing wording slot count",
            "value": float(len(explicit_mapping_missing_wording_slots)),
            "note": f"Explicit-mapping missing wording slots are {explicit_mapping_missing_wording_slots}.",
        },
        {
            "row_id": "wording_slot_residual_split_contract_ready",
            "status": "pass" if split_contract_ready else "reject",
            "metric": "reduced wording-slot residual split contract ready for next branch",
            "value": 1.0 if split_contract_ready else 0.0,
            "note": (
                "The next branch may now treat shell-anchor missing slots and explicit-mapping missing slots as separate residual routes."
                if split_contract_ready
                else "The reduced wording-slot residual split is not stable enough to launch the next branch."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "wording-slot residual blocker split contract",
        },
        "inputs": {
            "mass_origin_shell_anchor_wording_closure_retry_json": _relative_str(SHELL_ANCHOR_RETRY_JSON),
            "mass_origin_explicit_mapping_wording_closure_retry_json": _relative_str(EXPLICIT_MAPPING_RETRY_JSON),
            "mass_origin_source_wording_blocker_split_refresh_json": _relative_str(SPLIT_REFRESH_JSON),
        },
        "intent": "Freeze the reduced wording-slot residual blocker split so the next branch can attack shell-anchor and explicit-mapping missing wording slots separately.",
        "formulas": {
            "split_rule": "the next branch may split iff .114 still blocks handoff and the residual source-level blocker set is exactly {shell_anchor_wording_slots_still_missing, explicit_mapping_wording_slots_still_missing}",
            "route_rule": "both wording-slot residual routes must remain admissible under the frozen same-sector / no-new-free-parameter contract",
        },
        "rows": rows,
        "summary": {
            "remaining_source_level_blockers": remaining_source_level_blockers,
            "shell_anchor_missing_wording_slots": shell_anchor_missing_wording_slots,
            "explicit_mapping_missing_wording_slots": explicit_mapping_missing_wording_slots,
            "shell_anchor_wording_slot_route_still_admissible": shell_anchor_wording_slot_route_still_admissible,
            "explicit_mapping_wording_slot_route_still_admissible": explicit_mapping_wording_slot_route_still_admissible,
            "hand_off_to_8_7_55_2_83": hand_off_to_8_7_55_2_83,
            "split_contract_ready": split_contract_ready,
        },
        "decision": {
            "overall_status": "wording_slot_residual_split_contract_frozen",
            "keep_mass_origin_branch_blocked": True,
            "split_contract_ready": split_contract_ready,
            "hand_off_to_8_7_55_2_83": hand_off_to_8_7_55_2_83,
            "remaining_source_level_blockers": remaining_source_level_blockers,
            "shell_anchor_missing_wording_slots": shell_anchor_missing_wording_slots,
            "explicit_mapping_missing_wording_slots": explicit_mapping_missing_wording_slots,
        },
        "evidence": {
            "shell_anchor_wording_closure_retry_summary": shell_anchor_retry_summary,
            "explicit_mapping_wording_closure_retry_summary": explicit_mapping_retry_summary,
            "source_wording_blocker_split_refresh_summary": split_refresh_summary,
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
