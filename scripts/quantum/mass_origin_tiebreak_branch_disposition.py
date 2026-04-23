#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_tiebreak_branch_disposition.py

Step 8.7.55.2.90:
Freeze the disposition of the same-sector tie-break branch and decide whether
the mass-origin route can hand off to 8.7.55.2.83-.84 or must return to a
blocked hold.

Inputs:
  - output/public/quantum/mass_origin_same_sector_vpp_tiebreak_invariant_metrics.json
  - output/public/quantum/mass_origin_same_sector_tiebreak_target_bridge_metrics.json
  - output/public/quantum/mass_origin_single_public_vpp_shape_closure_retry_metrics.json
  - output/public/quantum/mass_origin_positive_particle_sector_chi_to_vpp_retry_metrics.json
  - output/public/quantum/mass_origin_solver_ready_reopen_gate_retry_metrics.json

Outputs:
  - output/public/quantum/mass_origin_tiebreak_branch_disposition_metrics.json
  - output/public/quantum/mass_origin_tiebreak_branch_disposition_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]

TIEBREAK_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_vpp_tiebreak_invariant_metrics.json"
BRIDGE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_tiebreak_target_bridge_metrics.json"
CLOSURE_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_single_public_vpp_shape_closure_retry_metrics.json"
PROMOTION_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_positive_particle_sector_chi_to_vpp_retry_metrics.json"
REOPEN_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_solver_ready_reopen_gate_retry_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_tiebreak_branch_disposition_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_tiebreak_branch_disposition_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.90"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the disposition of the same-sector tie-break branch.",
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
        TIEBREAK_JSON,
        BRIDGE_JSON,
        CLOSURE_RETRY_JSON,
        PROMOTION_RETRY_JSON,
        REOPEN_RETRY_JSON,
    ):
        _require_path(path)

    tiebreak = _read_json(TIEBREAK_JSON)
    bridge = _read_json(BRIDGE_JSON)
    closure_retry = _read_json(CLOSURE_RETRY_JSON)
    promotion_retry = _read_json(PROMOTION_RETRY_JSON)
    reopen_retry = _read_json(REOPEN_RETRY_JSON)

    tiebreak_summary = tiebreak.get("summary", {})
    bridge_summary = bridge.get("summary", {})
    closure_retry_summary = closure_retry.get("summary", {})
    promotion_retry_summary = promotion_retry.get("summary", {})
    reopen_retry_summary = reopen_retry.get("summary", {})
    reopen_retry_decision = reopen_retry.get("decision", {})

    tie_break_route_available = bool(tiebreak_summary.get("tie_break_route_available", False))
    selection_ready = bool(
        tiebreak_summary.get("selection_ready", False)
        and bridge_summary.get("target_value_available", False)
        and closure_retry_summary.get("single_public_vpp_shape_available", False)
    )
    hand_off_to_8_7_55_2_83 = bool(reopen_retry_summary.get("proceed_to_no_free_parameter_mass_solver", False))
    second_route_blocked_hold = not hand_off_to_8_7_55_2_83

    remaining_missing_artifacts = _ordered_unique(reopen_retry_decision.get("next_required_artifacts", []))

    rows = [
        {
            "row_id": "tiebreak_branch_disposition_complete",
            "status": "pass",
            "metric": "tie-break branch disposition complete",
            "value": 1.0,
            "note": "The branch disposition freezes whether the same-sector tie-break route hands off to the mass-spectrum pilot or returns to blocked hold.",
        },
        {
            "row_id": "tiebreak_branch_route_available",
            "status": "pass" if tie_break_route_available else "reject",
            "metric": "same-sector tie-break route available",
            "value": 1.0 if tie_break_route_available else 0.0,
            "note": "The derivative-ratio discriminant route exists because mexican_hat and logarithmic remain separated by the invariant R3.",
        },
        {
            "row_id": "tiebreak_branch_selection_ready",
            "status": "pass" if selection_ready else "reject",
            "metric": "tie-break branch selection ready",
            "value": 1.0 if selection_ready else 0.0,
            "note": (
                "A same-sector target value exists and the branch can isolate one family."
                if selection_ready
                else "Selection is not ready because the same-sector target value is still missing and no unique family is selected."
            ),
        },
        {
            "row_id": "tiebreak_branch_handoff_to_8_7_55_2_83",
            "status": "pass" if hand_off_to_8_7_55_2_83 else "reject",
            "metric": "handoff to 8.7.55.2.83-.84 allowed",
            "value": 1.0 if hand_off_to_8_7_55_2_83 else 0.0,
            "note": (
                "The branch may proceed to the mass-spectrum pilot."
                if hand_off_to_8_7_55_2_83
                else "The branch may not hand off because reopen is still false after the retry refresh."
            ),
        },
        {
            "row_id": "tiebreak_branch_second_route_blocked_hold",
            "status": "blocked" if second_route_blocked_hold else "pass",
            "metric": "second route returns to blocked hold",
            "value": 1.0 if second_route_blocked_hold else 0.0,
            "note": (
                f"The second route returns to blocked hold with remaining artifacts {remaining_missing_artifacts}."
                if second_route_blocked_hold
                else "Blocked hold is not needed because the branch handed off to the mass-spectrum pilot."
            ),
        },
        {
            "row_id": "tiebreak_branch_remaining_missing_artifact_count",
            "status": "inventory",
            "metric": "remaining missing artifacts count",
            "value": float(len(remaining_missing_artifacts)),
            "note": (
                f"Remaining missing artifacts: {', '.join(remaining_missing_artifacts)}."
                if remaining_missing_artifacts
                else "No missing artifacts remain."
            ),
        },
    ]

    overall_status = (
        "tiebreak_branch_handoff_ready"
        if hand_off_to_8_7_55_2_83
        else "tiebreak_branch_returns_to_blocked_hold"
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "tie-break branch disposition / handoff",
        },
        "inputs": {
            "mass_origin_same_sector_vpp_tiebreak_invariant_json": _relative_str(TIEBREAK_JSON),
            "mass_origin_same_sector_tiebreak_target_bridge_json": _relative_str(BRIDGE_JSON),
            "mass_origin_single_public_vpp_shape_closure_retry_json": _relative_str(CLOSURE_RETRY_JSON),
            "mass_origin_positive_particle_sector_chi_to_vpp_retry_json": _relative_str(PROMOTION_RETRY_JSON),
            "mass_origin_solver_ready_reopen_gate_retry_json": _relative_str(REOPEN_RETRY_JSON),
        },
        "intent": "Freeze the disposition of the same-sector tie-break branch after the retry refresh.",
        "formulas": {
            "handoff_rule": "hand_off_to_8_7_55_2_83 iff the retry refresh allows proceed_to_no_free_parameter_mass_solver",
            "blocked_hold_rule": "second_route_blocked_hold iff handoff is false and remaining missing artifacts remain",
            "selection_rule": "selection_ready requires the tie-break route, a public same-sector target value, and a unique selected V(|P|) family",
        },
        "rows": rows,
        "summary": {
            "tie_break_route_available": tie_break_route_available,
            "selection_ready": selection_ready,
            "hand_off_to_8_7_55_2_83": hand_off_to_8_7_55_2_83,
            "second_route_blocked_hold": second_route_blocked_hold,
            "remaining_missing_artifacts": remaining_missing_artifacts,
        },
        "decision": {
            "overall_status": overall_status,
            "keep_mass_origin_branch_blocked": second_route_blocked_hold,
            "tie_break_route_available": tie_break_route_available,
            "selection_ready": selection_ready,
            "hand_off_to_8_7_55_2_83": hand_off_to_8_7_55_2_83,
            "second_route_blocked_hold": second_route_blocked_hold,
            "remaining_missing_artifacts": remaining_missing_artifacts,
        },
        "evidence": {
            "tiebreak_summary": tiebreak_summary,
            "bridge_summary": bridge_summary,
            "closure_retry_summary": closure_retry_summary,
            "promotion_retry_summary": promotion_retry_summary,
            "reopen_retry_summary": reopen_retry_summary,
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
