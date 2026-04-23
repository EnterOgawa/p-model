#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_stack_operational_cycle.py

Step 8.7.55.2.64:
Freeze one current-state operational-cycle artifact for the mass-origin
blocked interface-stack control-plane pack stack control-plane pack stack
stack, so downstream readers can tell whether the top-level control-plane
pack stack stack continuity artifact and the top-level control-plane pack
stack stack rerun policy still agree on the present no-change-hold state.

Inputs:
  - output/public/quantum/mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_stack_continuity_metrics.json
  - output/public/quantum/mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_stack_rerun_policy.json

Outputs:
  - output/public/quantum/mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_stack_operational_cycle.json
  - output/public/quantum/mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_stack_operational_cycle_rows.csv

Assumptions:
  - The operational cycle is descriptive only; it does not execute the rerun
    chain and does not reopen the mass-origin branch.
  - When the continuity artifact reports no change and the rerun policy says
    rerun is not required, the operational cycle should remain on no-change
    hold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_QUANTUM_DIR = ROOT / "output" / "public" / "quantum"
CONTINUITY_JSON = (
    PUBLIC_QUANTUM_DIR
    / "mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_stack_continuity_metrics.json"
)
RERUN_POLICY_JSON = (
    PUBLIC_QUANTUM_DIR
    / "mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_stack_rerun_policy.json"
)
OUT_JSON = (
    PUBLIC_QUANTUM_DIR
    / "mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_stack_operational_cycle.json"
)
OUT_CSV = (
    PUBLIC_QUANTUM_DIR
    / "mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_stack_operational_cycle_rows.csv"
)
DEFAULT_STEP_TAG = "8.7.55.2.64"
TERMINAL_CONTROL_PLANE_PACK_STATUS = "watch:blocked_interface_stack_control_plane_pack_stack_locked_no_change_hold"
CONTROL_PLANE_STATUS = "watch:blocked_interface_stack_control_plane_pack_stack_control_plane_locked_no_change_hold"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the operational-cycle state for the mass-origin blocked interface-stack control-plane pack stack control-plane pack stack stack.",
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


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: `_as_dict` の入出力契約と処理意図を定義する。

def _as_dict(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = payload.get(key, {})
    return value if isinstance(value, dict) else {}


# 関数: `_extract_cycle_state` の入出力契約と処理意図を定義する。

def _extract_cycle_state(continuity: Dict[str, Any], rerun_policy: Dict[str, Any]) -> Dict[str, Any]:
    continuity_phase = _as_dict(continuity, "phase")
    continuity_summary = _as_dict(continuity, "summary")
    continuity_decision = _as_dict(continuity, "decision")
    rerun_phase = _as_dict(rerun_policy, "phase")
    rerun_summary = _as_dict(rerun_policy, "summary")
    rerun_decision = _as_dict(rerun_policy, "decision")

    next_required_artifacts = rerun_decision.get(
        "next_required_artifacts",
        continuity_decision.get("next_required_artifacts", []),
    )

    # 条件分岐: `not isinstance(next_required_artifacts, list)` を満たす経路を評価する。
    if not isinstance(next_required_artifacts, list):
        next_required_artifacts = []

    return {
        "continuity_phase_step": str(continuity_phase.get("step", "")).strip(),
        "continuity_decision": str(continuity_decision.get("decision", "")).strip(),
        "continuity_overall_status": str(continuity_decision.get("overall_status", "")).strip(),
        "continuity_state_changed": bool(continuity_summary.get("continuity_state_changed", False)),
        "continuity_event_counter": int(continuity_summary.get("event_counter", 0) or 0),
        "rerun_policy_phase_step": str(rerun_phase.get("step", "")).strip(),
        "rerun_policy_decision": str(rerun_decision.get("decision", "")).strip(),
        "rerun_policy_overall_status": str(rerun_decision.get("overall_status", "")).strip(),
        "rerun_required_now": bool(rerun_decision.get("rerun_required_now", False)),
        "rerun_trigger": str(rerun_decision.get("rerun_trigger", "")).strip(),
        "apply_chain_step_count": int(rerun_summary.get("apply_chain_step_count", 0) or 0),
        "apply_chain_signature_sha256": str(rerun_summary.get("apply_chain_signature_sha256", "")).strip().lower(),
        "blocked_state_detail": str(rerun_decision.get("blocked_state_detail", "")).strip(),
        "latent_reopen_routes_exhausted": bool(rerun_decision.get("latent_reopen_routes_exhausted", False)),
        "continuity_no_change_hold": bool(rerun_decision.get("continuity_no_change_hold", False)),
        "control_plane_pack_stack_control_plane_pack_stack_complete": bool(
            rerun_decision.get("control_plane_pack_stack_control_plane_pack_stack_complete", False)
        ),
        "keep_mass_origin_branch_blocked": bool(rerun_decision.get("keep_mass_origin_branch_blocked", True)),
        "terminal_control_plane_pack_status": str(
            continuity_summary.get(
                "terminal_control_plane_pack_status",
                rerun_summary.get("terminal_control_plane_pack_status", ""),
            )
        ).strip(),
        "control_plane_pack_signature_sha256": str(
            continuity_summary.get(
                "control_plane_pack_signature_sha256",
                rerun_summary.get("control_plane_pack_signature_sha256", ""),
            )
        ).strip().lower(),
        "control_plane_pack_stack_signature_sha256": str(
            continuity_summary.get(
                "control_plane_pack_stack_signature_sha256",
                rerun_summary.get("control_plane_pack_stack_signature_sha256", ""),
            )
        ).strip().lower(),
        "control_plane_status": str(
            continuity_summary.get("control_plane_status", rerun_summary.get("control_plane_status", ""))
        ).strip(),
        "next_action": str(rerun_decision.get("next_action", "")).strip(),
        "next_required_artifacts": [str(item) for item in next_required_artifacts],
    }


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows(cycle_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    next_required_artifacts = cycle_state.get("next_required_artifacts", [])
    next_required_note = ", ".join(str(item) for item in next_required_artifacts) if next_required_artifacts else "none"
    apply_chain_signature = str(cycle_state.get("apply_chain_signature_sha256", ""))
    terminal_control_plane_pack_status = str(cycle_state.get("terminal_control_plane_pack_status", ""))
    control_plane_pack_stack_signature = str(cycle_state.get("control_plane_pack_stack_signature_sha256", ""))
    control_plane_status = str(cycle_state.get("control_plane_status", ""))

    return [
        {
            "row_id": "continuity_decision_no_change_hold",
            "status": "pass" if cycle_state.get("continuity_decision") == "no_change_hold" else "watch",
            "metric": "control-plane pack stack control-plane pack stack stack continuity artifact remains on no-change hold",
            "value": 1.0 if cycle_state.get("continuity_decision") == "no_change_hold" else 0.0,
            "note": f"Current continuity decision = {cycle_state.get('continuity_decision')}.",
        },
        {
            "row_id": "rerun_policy_locked_no_change_hold",
            "status": "pass" if cycle_state.get("rerun_policy_decision") == "policy_locked_no_change_hold" else "watch",
            "metric": "control-plane pack stack control-plane pack stack stack rerun policy remains locked on no-change hold",
            "value": 1.0 if cycle_state.get("rerun_policy_decision") == "policy_locked_no_change_hold" else 0.0,
            "note": f"Current rerun policy decision = {cycle_state.get('rerun_policy_decision')}.",
        },
        {
            "row_id": "continuity_state_not_changed",
            "status": "pass" if not bool(cycle_state.get("continuity_state_changed", False)) else "watch",
            "metric": "control-plane pack stack control-plane pack stack stack continuity state is unchanged",
            "value": 1.0 if not bool(cycle_state.get("continuity_state_changed", False)) else 0.0,
            "note": "The rerun chain should stay idle until the control-plane pack stack control-plane pack stack stack continuity state changes.",
        },
        {
            "row_id": "rerun_required_now_false",
            "status": "pass" if not bool(cycle_state.get("rerun_required_now", False)) else "watch",
            "metric": "control-plane pack stack control-plane pack stack stack rerun policy does not currently require the chain to run",
            "value": 1.0 if not bool(cycle_state.get("rerun_required_now", False)) else 0.0,
            "note": "This row is watch only when the top-level control-plane policy already says the chain must run now.",
        },
        {
            "row_id": "rerun_trigger_is_control_plane_pack_stack_control_plane_pack_stack_stack_continuity_state_changed",
            "status": (
                "pass"
                if cycle_state.get("rerun_trigger")
                == "blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_stack_continuity_state_changed"
                else "reject"
            ),
            "metric": "rerun trigger matches the control-plane pack stack control-plane pack stack stack continuity trigger contract",
            "value": (
                1.0
                if cycle_state.get("rerun_trigger")
                == "blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_stack_continuity_state_changed"
                else 0.0
            ),
            "note": f"Current rerun trigger = {cycle_state.get('rerun_trigger')}.",
        },
        {
            "row_id": "apply_chain_step_count_expected",
            "status": "pass" if int(cycle_state.get("apply_chain_step_count", 0)) == 3 else "reject",
            "metric": "rerun/apply chain keeps the expected three-step order",
            "value": float(cycle_state.get("apply_chain_step_count", 0)),
            "note": "The control-plane pack stack control-plane pack stack stack operational cycle expects three steps: stack-stack manifest refresh, stack-stack continuity settle, and stack-stack rerun-policy refresh.",
        },
        {
            "row_id": "apply_chain_signature_present",
            "status": "pass" if apply_chain_signature else "reject",
            "metric": "control-plane pack stack control-plane pack stack stack rerun/apply chain signature is present",
            "value": 1.0 if apply_chain_signature else 0.0,
            "note": "The chain signature lets downstream readers detect top-level control-plane policy drift.",
        },
        {
            "row_id": "blocked_state_detail_specific_missing_artifacts_fixed",
            "status": "pass" if cycle_state.get("blocked_state_detail") == "specific_missing_artifacts_fixed" else "watch",
            "metric": "blocked-state detail stays specific-missing-artifacts-fixed",
            "value": 1.0 if cycle_state.get("blocked_state_detail") == "specific_missing_artifacts_fixed" else 0.0,
            "note": f"Current blocked_state_detail = {cycle_state.get('blocked_state_detail')}.",
        },
        {
            "row_id": "latent_reopen_routes_exhausted",
            "status": "pass" if bool(cycle_state.get("latent_reopen_routes_exhausted", False)) else "reject",
            "metric": "latent reopen routes remain exhausted",
            "value": 1.0 if bool(cycle_state.get("latent_reopen_routes_exhausted", False)) else 0.0,
            "note": "No repo-wide latent public route currently exists for the missing same-sector/Vpp artifacts.",
        },
        {
            "row_id": "control_plane_pack_stack_control_plane_pack_stack_complete",
            "status": "pass" if bool(cycle_state.get("control_plane_pack_stack_control_plane_pack_stack_complete", False)) else "reject",
            "metric": "control-plane pack stack control-plane pack stack remains complete",
            "value": 1.0 if bool(cycle_state.get("control_plane_pack_stack_control_plane_pack_stack_complete", False)) else 0.0,
            "note": "The operational cycle assumes the full blocked top-level control-plane pack stack control-plane pack stack remains complete.",
        },
        {
            "row_id": "control_plane_pack_stack_signature_present",
            "status": "pass" if control_plane_pack_stack_signature else "reject",
            "metric": "control-plane pack stack signature is present",
            "value": 1.0 if control_plane_pack_stack_signature else 0.0,
            "note": "The pack-stack signature lets downstream readers verify the operational-cycle contract against the top-level stack audit.",
        },
        {
            "row_id": "terminal_control_plane_pack_status_locked_no_change_hold",
            "status": "pass" if terminal_control_plane_pack_status == TERMINAL_CONTROL_PLANE_PACK_STATUS else "watch",
            "metric": "terminal control-plane pack status remains locked on no-change hold",
            "value": 1.0 if terminal_control_plane_pack_status == TERMINAL_CONTROL_PLANE_PACK_STATUS else 0.0,
            "note": f"Current terminal_control_plane_pack_status = {terminal_control_plane_pack_status}.",
        },
        {
            "row_id": "control_plane_status_locked_no_change_hold",
            "status": "pass" if control_plane_status == CONTROL_PLANE_STATUS else "watch",
            "metric": "top-level control-plane status remains locked on no-change hold",
            "value": 1.0 if control_plane_status == CONTROL_PLANE_STATUS else 0.0,
            "note": f"Current control_plane_status = {control_plane_status}.",
        },
        {
            "row_id": "mass_origin_branch_blocked",
            "status": "blocked" if bool(cycle_state.get("keep_mass_origin_branch_blocked", True)) else "pass",
            "metric": "mass-origin branch remains blocked in the control-plane pack stack control-plane pack stack stack operational cycle",
            "value": 1.0 if bool(cycle_state.get("keep_mass_origin_branch_blocked", True)) else 0.0,
            "note": "This operational cycle is about stable blocked-state maintenance, not reopen execution.",
        },
        {
            "row_id": "next_required_artifacts_count",
            "status": "inventory",
            "metric": "next required artifacts count",
            "value": float(len(next_required_artifacts)),
            "note": f"Next required artifacts: {next_required_note}.",
        },
    ]


# 関数: `_cycle_signature` の入出力契約と処理意図を定義する。

def _cycle_signature(cycle_state: Dict[str, Any]) -> str:
    packed = json.dumps(cycle_state, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(packed).hexdigest().lower()


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    _require_path(CONTINUITY_JSON)
    _require_path(RERUN_POLICY_JSON)
    continuity = _read_json(CONTINUITY_JSON)
    rerun_policy = _read_json(RERUN_POLICY_JSON)
    cycle_state = _extract_cycle_state(continuity, rerun_policy)
    rows = _build_rows(cycle_state)
    cycle_signature = _cycle_signature(cycle_state)
    has_reject = any(str(row.get("status")) == "reject" for row in rows)

    # 条件分岐: `has_reject` を満たす経路を評価する。
    if has_reject:
        overall_status = "reject"
        decision_text = "operational_cycle_broken"
        next_action = "repair_control_plane_pack_stack_control_plane_pack_stack_stack_trigger_or_apply_chain_contract"
    # 条件分岐: 前段条件が不成立で、`bool(cycle_state.get("continuity_state_changed", False)) or bool(cycle_state.get("rerun_required_now", False))` を追加評価する。
    elif bool(cycle_state.get("continuity_state_changed", False)) or bool(cycle_state.get("rerun_required_now", False)):
        overall_status = "watch"
        decision_text = "rerun_chain_required_now"
        next_action = "run_mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_stack_chain_in_order"
    # 条件分岐: 前段条件が不成立で、`bool(cycle_state.get("keep_mass_origin_branch_blocked", True))` を追加評価する。
    elif bool(cycle_state.get("keep_mass_origin_branch_blocked", True)):
        overall_status = "watch"
        decision_text = "operational_cycle_no_change_hold"
        next_action = "wait_for_new_public_artifact_then_rerun_control_plane_pack_stack_control_plane_pack_stack_stack_operational_cycle"
    else:
        overall_status = "pass"
        decision_text = "branch_unblocked_outside_cycle"
        next_action = "evaluate_reopen_transition_now"

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "mass-origin blocked interface-stack control-plane pack stack control-plane pack stack stack operational cycle",
        },
        "inputs": {
            "blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_stack_continuity_metrics_json": _rel(CONTINUITY_JSON),
            "blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_stack_rerun_policy_json": _rel(RERUN_POLICY_JSON),
        },
        "intent": "Freeze one current-state operational-cycle decision for the mass-origin blocked interface-stack control-plane pack stack control-plane pack stack stack without executing the rerun chain.",
        "summary": {
            "continuity_phase_step": str(cycle_state.get("continuity_phase_step", "")),
            "rerun_policy_phase_step": str(cycle_state.get("rerun_policy_phase_step", "")),
            "continuity_decision": str(cycle_state.get("continuity_decision", "")),
            "rerun_policy_decision": str(cycle_state.get("rerun_policy_decision", "")),
            "continuity_state_changed": bool(cycle_state.get("continuity_state_changed", False)),
            "rerun_required_now": bool(cycle_state.get("rerun_required_now", False)),
            "terminal_control_plane_pack_status": str(cycle_state.get("terminal_control_plane_pack_status", "")),
            "control_plane_pack_signature_sha256": str(cycle_state.get("control_plane_pack_signature_sha256", "")),
            "control_plane_pack_stack_signature_sha256": str(cycle_state.get("control_plane_pack_stack_signature_sha256", "")),
            "control_plane_status": str(cycle_state.get("control_plane_status", "")),
            "apply_chain_step_count": int(cycle_state.get("apply_chain_step_count", 0)),
            "apply_chain_signature_sha256": str(cycle_state.get("apply_chain_signature_sha256", "")),
            "cycle_signature_sha256": cycle_signature,
            "blocked_state_detail": str(cycle_state.get("blocked_state_detail", "")),
            "latent_reopen_routes_exhausted": bool(cycle_state.get("latent_reopen_routes_exhausted", False)),
            "continuity_no_change_hold": bool(cycle_state.get("continuity_no_change_hold", False)),
            "control_plane_pack_stack_control_plane_pack_stack_complete": bool(
                cycle_state.get("control_plane_pack_stack_control_plane_pack_stack_complete", False)
            ),
            "next_required_artifacts": cycle_state.get("next_required_artifacts", []),
        },
        "checks": rows,
        "decision": {
            "overall_status": overall_status,
            "decision": decision_text,
            "continuity_state_changed": bool(cycle_state.get("continuity_state_changed", False)),
            "rerun_required_now": bool(cycle_state.get("rerun_required_now", False)),
            "rerun_trigger": str(cycle_state.get("rerun_trigger", "")),
            "control_plane_pack_stack_control_plane_pack_stack_complete": bool(
                cycle_state.get("control_plane_pack_stack_control_plane_pack_stack_complete", False)
            ),
            "keep_mass_origin_branch_blocked": bool(cycle_state.get("keep_mass_origin_branch_blocked", True)),
            "blocked_state_detail": str(cycle_state.get("blocked_state_detail", "")),
            "latent_reopen_routes_exhausted": bool(cycle_state.get("latent_reopen_routes_exhausted", False)),
            "continuity_no_change_hold": bool(cycle_state.get("continuity_no_change_hold", False)),
            "proceed_to_dark_matter_branch": False,
            "next_required_artifacts": cycle_state.get("next_required_artifacts", []),
            "next_action": next_action,
        },
        "evidence": {
            "continuity_artifact": {
                "path": _rel(CONTINUITY_JSON),
                "phase_step": str(cycle_state.get("continuity_phase_step", "")),
                "decision": str(cycle_state.get("continuity_decision", "")),
            },
            "rerun_policy_artifact": {
                "path": _rel(RERUN_POLICY_JSON),
                "phase_step": str(cycle_state.get("rerun_policy_phase_step", "")),
                "decision": str(cycle_state.get("rerun_policy_decision", "")),
            },
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
    payload = _build_payload(str(args.step_tag))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["checks"])
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
