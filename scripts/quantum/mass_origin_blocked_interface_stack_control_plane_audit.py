#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_blocked_interface_stack_control_plane_audit.py

Step 8.7.55.2.30:
Freeze one top-level control-plane audit for the mass-origin blocked interface
stack, so downstream readers can verify that the manifest, continuity,
rerun-policy, and operational-cycle artifacts still expose the same blocked
detail, rerun trigger, and no-change-hold contract.

Inputs:
  - output/public/quantum/mass_origin_blocked_interface_stack_manifest.json
  - output/public/quantum/mass_origin_blocked_interface_stack_continuity_metrics.json
  - output/public/quantum/mass_origin_blocked_interface_stack_rerun_policy.json
  - output/public/quantum/mass_origin_blocked_interface_stack_operational_cycle.json

Outputs:
  - output/public/quantum/mass_origin_blocked_interface_stack_control_plane_metrics.json
  - output/public/quantum/mass_origin_blocked_interface_stack_control_plane_rows.csv

Assumptions:
  - This audit is descriptive only; it does not execute the rerun chain and
    does not reopen the mass-origin branch.
  - The control plane is considered settled only when all four artifacts agree
    on blocked detail, latent-route exhaustion, next required artifacts,
    rerun-trigger contract, apply-chain signature, and stable no-change-hold.
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
MANIFEST_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_manifest.json"
CONTINUITY_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_continuity_metrics.json"
RERUN_POLICY_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_rerun_policy.json"
OPERATIONAL_CYCLE_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_operational_cycle.json"
OUT_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_control_plane_metrics.json"
OUT_CSV = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_control_plane_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.30"
EXPECTED_BLOCKED_DETAIL = "specific_missing_artifacts_fixed"
EXPECTED_TERMINAL_STACK_STATUS = "watch:blocked_interface_stack_locked_no_change_hold"
EXPECTED_RERUN_TRIGGER = "blocked_interface_stack_continuity_state_changed"
EXPECTED_MANIFEST_STATUS = "blocked_interface_stack_manifest_frozen"
EXPECTED_CONTINUITY_DECISION = "no_change_hold"
EXPECTED_RERUN_POLICY_DECISION = "policy_locked_no_change_hold"
EXPECTED_OPERATIONAL_CYCLE_DECISION = "operational_cycle_no_change_hold"
EXPECTED_APPLY_CHAIN_STEP_COUNT = 11


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the top-level control-plane audit for the mass-origin blocked interface stack.",
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


# 関数: `_extract_str` の入出力契約と処理意図を定義する。

def _extract_str(payload: Dict[str, Any], path_keys: List[List[str]]) -> str:
    for path in path_keys:
        cursor: Any = payload
        missing = False
        for key in path:
            # 条件分岐: `not isinstance(cursor, dict) or key not in cursor` を満たす経路を評価する。
            if not isinstance(cursor, dict) or key not in cursor:
                missing = True
                break

            cursor = cursor[key]

        # 条件分岐: `not missing` を満たす経路を評価する。

        if not missing:
            return str(cursor).strip()

    return ""


# 関数: `_extract_bool` の入出力契約と処理意図を定義する。

def _extract_bool(payload: Dict[str, Any], path_keys: List[List[str]], default: bool = False) -> bool:
    for path in path_keys:
        cursor: Any = payload
        missing = False
        for key in path:
            # 条件分岐: `not isinstance(cursor, dict) or key not in cursor` を満たす経路を評価する。
            if not isinstance(cursor, dict) or key not in cursor:
                missing = True
                break

            cursor = cursor[key]

        # 条件分岐: `not missing` を満たす経路を評価する。

        if not missing:
            return bool(cursor)

    return default


# 関数: `_extract_list` の入出力契約と処理意図を定義する。

def _extract_list(payload: Dict[str, Any], path_keys: List[List[str]]) -> List[str]:
    for path in path_keys:
        cursor: Any = payload
        missing = False
        for key in path:
            # 条件分岐: `not isinstance(cursor, dict) or key not in cursor` を満たす経路を評価する。
            if not isinstance(cursor, dict) or key not in cursor:
                missing = True
                break

            cursor = cursor[key]

        # 条件分岐: `not missing and isinstance(cursor, list)` を満たす経路を評価する。

        if (not missing) and isinstance(cursor, list):
            return [str(item) for item in cursor]

    return []


# 関数: `_extract_control_plane_state` の入出力契約と処理意図を定義する。

def _extract_control_plane_state(
    manifest: Dict[str, Any],
    continuity: Dict[str, Any],
    rerun_policy: Dict[str, Any],
    operational_cycle: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "manifest_phase_step": _extract_str(manifest, [["phase", "step"]]),
        "continuity_phase_step": _extract_str(continuity, [["phase", "step"]]),
        "rerun_policy_phase_step": _extract_str(rerun_policy, [["phase", "step"]]),
        "operational_cycle_phase_step": _extract_str(operational_cycle, [["phase", "step"]]),
        "manifest_status": _extract_str(manifest, [["decision", "overall_status"]]),
        "continuity_decision": _extract_str(continuity, [["decision", "decision"]]),
        "rerun_policy_decision": _extract_str(rerun_policy, [["decision", "decision"]]),
        "operational_cycle_decision": _extract_str(operational_cycle, [["decision", "decision"]]),
        "manifest_blocked_state_detail": _extract_str(
            manifest,
            [["decision", "blocked_state_detail"], ["summary", "blocked_state_detail"]],
        ),
        "continuity_blocked_state_detail": _extract_str(
            continuity,
            [["decision", "blocked_state_detail"], ["summary", "blocked_state_detail"]],
        ),
        "rerun_policy_blocked_state_detail": _extract_str(
            rerun_policy,
            [["decision", "blocked_state_detail"], ["summary", "blocked_state_detail"]],
        ),
        "operational_cycle_blocked_state_detail": _extract_str(
            operational_cycle,
            [["decision", "blocked_state_detail"], ["summary", "blocked_state_detail"]],
        ),
        "manifest_latent_reopen_routes_exhausted": _extract_bool(
            manifest,
            [["decision", "latent_reopen_routes_exhausted"], ["summary", "latent_reopen_routes_exhausted"]],
        ),
        "continuity_latent_reopen_routes_exhausted": _extract_bool(
            continuity,
            [["decision", "latent_reopen_routes_exhausted"], ["summary", "latent_reopen_routes_exhausted"]],
        ),
        "rerun_policy_latent_reopen_routes_exhausted": _extract_bool(
            rerun_policy,
            [["decision", "latent_reopen_routes_exhausted"], ["summary", "latent_reopen_routes_exhausted"]],
        ),
        "operational_cycle_latent_reopen_routes_exhausted": _extract_bool(
            operational_cycle,
            [["decision", "latent_reopen_routes_exhausted"], ["summary", "latent_reopen_routes_exhausted"]],
        ),
        "manifest_next_required_artifacts": _extract_list(
            manifest,
            [["decision", "next_required_artifacts"], ["summary", "next_required_artifacts"]],
        ),
        "continuity_next_required_artifacts": _extract_list(
            continuity,
            [["decision", "next_required_artifacts"], ["summary", "next_required_artifacts"]],
        ),
        "rerun_policy_next_required_artifacts": _extract_list(
            rerun_policy,
            [["decision", "next_required_artifacts"], ["summary", "next_required_artifacts"]],
        ),
        "operational_cycle_next_required_artifacts": _extract_list(
            operational_cycle,
            [["decision", "next_required_artifacts"], ["summary", "next_required_artifacts"]],
        ),
        "manifest_keep_blocked": _extract_bool(manifest, [["decision", "keep_mass_origin_branch_blocked"]], True),
        "continuity_keep_blocked": _extract_bool(continuity, [["decision", "keep_mass_origin_branch_blocked"]], True),
        "rerun_policy_keep_blocked": _extract_bool(rerun_policy, [["decision", "keep_mass_origin_branch_blocked"]], True),
        "operational_cycle_keep_blocked": _extract_bool(
            operational_cycle,
            [["decision", "keep_mass_origin_branch_blocked"]],
            True,
        ),
        "manifest_interface_stack_complete": _extract_bool(manifest, [["decision", "interface_stack_complete"]]),
        "continuity_interface_stack_complete": _extract_bool(continuity, [["decision", "interface_stack_complete"]]),
        "rerun_policy_interface_stack_complete": _extract_bool(rerun_policy, [["decision", "interface_stack_complete"]]),
        "operational_cycle_interface_stack_complete": _extract_bool(
            operational_cycle,
            [["decision", "interface_stack_complete"]],
        ),
        "terminal_stack_status_manifest": _extract_str(manifest, [["summary", "terminal_stack_status"]]),
        "terminal_stack_status_continuity": _extract_str(continuity, [["summary", "terminal_stack_status"]]),
        "terminal_stack_status_rerun_policy": _extract_str(rerun_policy, [["summary", "terminal_stack_status"]]),
        "terminal_stack_status_operational_cycle": _extract_str(
            operational_cycle,
            [["summary", "terminal_stack_status"]],
        ),
        "continuity_state_changed": _extract_bool(
            continuity,
            [["summary", "continuity_state_changed"], ["decision", "continuity_state_changed"]],
        ),
        "rerun_required_now_policy": _extract_bool(rerun_policy, [["decision", "rerun_required_now"]]),
        "rerun_required_now_cycle": _extract_bool(operational_cycle, [["decision", "rerun_required_now"]]),
        "rerun_trigger_policy": _extract_str(rerun_policy, [["decision", "rerun_trigger"]]),
        "rerun_trigger_cycle": _extract_str(operational_cycle, [["decision", "rerun_trigger"]]),
        "apply_chain_step_count_policy": int(_extract_str(rerun_policy, [["summary", "apply_chain_step_count"]]) or 0),
        "apply_chain_step_count_cycle": int(
            _extract_str(operational_cycle, [["summary", "apply_chain_step_count"]]) or 0
        ),
        "apply_chain_signature_policy": _extract_str(
            rerun_policy,
            [["summary", "apply_chain_signature_sha256"]],
        ).lower(),
        "apply_chain_signature_cycle": _extract_str(
            operational_cycle,
            [["summary", "apply_chain_signature_sha256"]],
        ).lower(),
        "manifest_signature_sha256": _extract_str(manifest, [["summary", "manifest_signature_sha256"]]).lower(),
        "continuity_state_signature_sha256": _extract_str(
            continuity,
            [["summary", "continuity_state_signature_sha256"]],
        ).lower(),
        "terminal_stack_signature_sha256": _extract_str(
            continuity,
            [["summary", "terminal_stack_signature_sha256"]],
        ).lower(),
        "operational_cycle_signature_sha256": _extract_str(
            operational_cycle,
            [["summary", "cycle_signature_sha256"]],
        ).lower(),
    }


# 関数: `_control_plane_signature` の入出力契約と処理意図を定義する。

def _control_plane_signature(control_plane_state: Dict[str, Any]) -> str:
    packed = json.dumps(control_plane_state, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(packed).hexdigest().lower()


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows(control_plane_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    detail_values = [
        control_plane_state.get("manifest_blocked_state_detail", ""),
        control_plane_state.get("continuity_blocked_state_detail", ""),
        control_plane_state.get("rerun_policy_blocked_state_detail", ""),
        control_plane_state.get("operational_cycle_blocked_state_detail", ""),
    ]
    latent_flags = [
        bool(control_plane_state.get("manifest_latent_reopen_routes_exhausted", False)),
        bool(control_plane_state.get("continuity_latent_reopen_routes_exhausted", False)),
        bool(control_plane_state.get("rerun_policy_latent_reopen_routes_exhausted", False)),
        bool(control_plane_state.get("operational_cycle_latent_reopen_routes_exhausted", False)),
    ]
    next_required_lists = [
        control_plane_state.get("manifest_next_required_artifacts", []),
        control_plane_state.get("continuity_next_required_artifacts", []),
        control_plane_state.get("rerun_policy_next_required_artifacts", []),
        control_plane_state.get("operational_cycle_next_required_artifacts", []),
    ]
    blocked_flags = [
        bool(control_plane_state.get("manifest_keep_blocked", True)),
        bool(control_plane_state.get("continuity_keep_blocked", True)),
        bool(control_plane_state.get("rerun_policy_keep_blocked", True)),
        bool(control_plane_state.get("operational_cycle_keep_blocked", True)),
    ]
    interface_stack_complete_flags = [
        bool(control_plane_state.get("manifest_interface_stack_complete", False)),
        bool(control_plane_state.get("continuity_interface_stack_complete", False)),
        bool(control_plane_state.get("rerun_policy_interface_stack_complete", False)),
        bool(control_plane_state.get("operational_cycle_interface_stack_complete", False)),
    ]
    terminal_statuses = [
        control_plane_state.get("terminal_stack_status_manifest", ""),
        control_plane_state.get("terminal_stack_status_continuity", ""),
        control_plane_state.get("terminal_stack_status_rerun_policy", ""),
        control_plane_state.get("terminal_stack_status_operational_cycle", ""),
    ]
    canonical_next_required = list(next_required_lists[0])
    next_required_consistent = all(list(items) == canonical_next_required for items in next_required_lists)
    detail_consistent = all(value == EXPECTED_BLOCKED_DETAIL for value in detail_values)
    latent_consistent = all(value is True for value in latent_flags)
    blocked_consistent = all(blocked_flags)
    interface_stack_complete_consistent = all(interface_stack_complete_flags)
    terminal_status_consistent = all(value == EXPECTED_TERMINAL_STACK_STATUS for value in terminal_statuses)
    rerun_trigger_contract_consistent = (
        control_plane_state.get("rerun_trigger_policy") == EXPECTED_RERUN_TRIGGER
        and control_plane_state.get("rerun_trigger_cycle") == EXPECTED_RERUN_TRIGGER
    )
    apply_chain_signature_consistent = (
        bool(control_plane_state.get("apply_chain_signature_policy", ""))
        and control_plane_state.get("apply_chain_signature_policy")
        == control_plane_state.get("apply_chain_signature_cycle")
    )
    apply_chain_step_count_consistent = (
        int(control_plane_state.get("apply_chain_step_count_policy", 0)) == EXPECTED_APPLY_CHAIN_STEP_COUNT
        and int(control_plane_state.get("apply_chain_step_count_cycle", 0)) == EXPECTED_APPLY_CHAIN_STEP_COUNT
    )
    rerun_required_now_false_consistent = (
        not bool(control_plane_state.get("rerun_required_now_policy", False))
        and not bool(control_plane_state.get("rerun_required_now_cycle", False))
    )

    return [
        {
            "row_id": "blocked_state_detail_consistent",
            "status": "pass" if detail_consistent else "reject",
            "metric": "blocked-state detail is consistent across the interface-stack control plane",
            "value": 1.0 if detail_consistent else 0.0,
            "note": f"Current detail set = {detail_values}.",
        },
        {
            "row_id": "latent_reopen_routes_exhausted_consistent",
            "status": "pass" if latent_consistent else "reject",
            "metric": "latent reopen-route exhaustion is consistent across the control plane",
            "value": 1.0 if latent_consistent else 0.0,
            "note": f"Current latent-route flags = {latent_flags}.",
        },
        {
            "row_id": "next_required_artifacts_consistent",
            "status": "pass" if next_required_consistent else "reject",
            "metric": "next required artifacts are consistent across the control plane",
            "value": 1.0 if next_required_consistent else 0.0,
            "note": f"Canonical next required artifacts = {canonical_next_required}.",
        },
        {
            "row_id": "rerun_trigger_contract_consistent",
            "status": "pass" if rerun_trigger_contract_consistent else "reject",
            "metric": "rerun-trigger contract is consistent across policy and operational cycle",
            "value": 1.0 if rerun_trigger_contract_consistent else 0.0,
            "note": (
                "Current rerun triggers = "
                f"{control_plane_state.get('rerun_trigger_policy')} / {control_plane_state.get('rerun_trigger_cycle')}."
            ),
        },
        {
            "row_id": "apply_chain_signature_consistent",
            "status": "pass" if apply_chain_signature_consistent else "reject",
            "metric": "apply-chain signature is consistent across policy and operational cycle",
            "value": 1.0 if apply_chain_signature_consistent else 0.0,
            "note": (
                "Current signatures = "
                f"{control_plane_state.get('apply_chain_signature_policy')} / "
                f"{control_plane_state.get('apply_chain_signature_cycle')}."
            ),
        },
        {
            "row_id": "apply_chain_step_count_consistent",
            "status": "pass" if apply_chain_step_count_consistent else "reject",
            "metric": "apply-chain step count remains the expected eleven-step order",
            "value": float(control_plane_state.get("apply_chain_step_count_policy", 0)),
            "note": (
                "Current apply-chain step counts = "
                f"{control_plane_state.get('apply_chain_step_count_policy')} / "
                f"{control_plane_state.get('apply_chain_step_count_cycle')}."
            ),
        },
        {
            "row_id": "manifest_frozen",
            "status": "pass" if control_plane_state.get("manifest_status") == EXPECTED_MANIFEST_STATUS else "watch",
            "metric": "manifest artifact remains frozen",
            "value": 1.0 if control_plane_state.get("manifest_status") == EXPECTED_MANIFEST_STATUS else 0.0,
            "note": f"Current manifest status = {control_plane_state.get('manifest_status')}.",
        },
        {
            "row_id": "continuity_no_change_hold",
            "status": "pass" if control_plane_state.get("continuity_decision") == EXPECTED_CONTINUITY_DECISION else "watch",
            "metric": "continuity artifact remains on no-change hold",
            "value": 1.0 if control_plane_state.get("continuity_decision") == EXPECTED_CONTINUITY_DECISION else 0.0,
            "note": f"Current continuity decision = {control_plane_state.get('continuity_decision')}.",
        },
        {
            "row_id": "rerun_policy_locked_no_change_hold",
            "status": "pass" if control_plane_state.get("rerun_policy_decision") == EXPECTED_RERUN_POLICY_DECISION else "watch",
            "metric": "rerun policy remains locked on no-change hold",
            "value": 1.0 if control_plane_state.get("rerun_policy_decision") == EXPECTED_RERUN_POLICY_DECISION else 0.0,
            "note": f"Current rerun-policy decision = {control_plane_state.get('rerun_policy_decision')}.",
        },
        {
            "row_id": "operational_cycle_no_change_hold",
            "status": "pass" if control_plane_state.get("operational_cycle_decision") == EXPECTED_OPERATIONAL_CYCLE_DECISION else "watch",
            "metric": "operational cycle remains on no-change hold",
            "value": 1.0 if control_plane_state.get("operational_cycle_decision") == EXPECTED_OPERATIONAL_CYCLE_DECISION else 0.0,
            "note": f"Current operational-cycle decision = {control_plane_state.get('operational_cycle_decision')}.",
        },
        {
            "row_id": "continuity_state_not_changed",
            "status": "pass" if not bool(control_plane_state.get("continuity_state_changed", False)) else "watch",
            "metric": "continuity state remains unchanged at the control-plane level",
            "value": 1.0 if not bool(control_plane_state.get("continuity_state_changed", False)) else 0.0,
            "note": "The control plane stays on hold only while the continuity state remains unchanged.",
        },
        {
            "row_id": "rerun_required_now_false_consistent",
            "status": "pass" if rerun_required_now_false_consistent else "reject",
            "metric": "rerun-required-now remains false across policy and operational cycle",
            "value": 1.0 if rerun_required_now_false_consistent else 0.0,
            "note": (
                "Current rerun-required flags = "
                f"{control_plane_state.get('rerun_required_now_policy')} / "
                f"{control_plane_state.get('rerun_required_now_cycle')}."
            ),
        },
        {
            "row_id": "terminal_stack_status_consistent",
            "status": "pass" if terminal_status_consistent else "reject",
            "metric": "terminal stack status is consistent across the control plane",
            "value": 1.0 if terminal_status_consistent else 0.0,
            "note": f"Current terminal stack statuses = {terminal_statuses}.",
        },
        {
            "row_id": "interface_stack_complete_all_layers",
            "status": "pass" if interface_stack_complete_consistent else "reject",
            "metric": "interface stack remains complete across all control-plane layers",
            "value": 1.0 if interface_stack_complete_consistent else 0.0,
            "note": f"Current interface-stack-complete flags = {interface_stack_complete_flags}.",
        },
        {
            "row_id": "mass_origin_branch_blocked_all_layers",
            "status": "blocked" if blocked_consistent else "reject",
            "metric": "mass-origin branch remains blocked across all control-plane layers",
            "value": 1.0 if blocked_consistent else 0.0,
            "note": f"Current keep-blocked flags = {blocked_flags}.",
        },
        {
            "row_id": "next_required_artifacts_count",
            "status": "inventory",
            "metric": "next required artifacts count",
            "value": float(len(canonical_next_required)),
            "note": f"Next required artifacts: {', '.join(canonical_next_required) if canonical_next_required else 'none'}.",
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (MANIFEST_JSON, CONTINUITY_JSON, RERUN_POLICY_JSON, OPERATIONAL_CYCLE_JSON):
        _require_path(path)

    manifest = _read_json(MANIFEST_JSON)
    continuity = _read_json(CONTINUITY_JSON)
    rerun_policy = _read_json(RERUN_POLICY_JSON)
    operational_cycle = _read_json(OPERATIONAL_CYCLE_JSON)
    control_plane_state = _extract_control_plane_state(
        manifest,
        continuity,
        rerun_policy,
        operational_cycle,
    )
    rows = _build_rows(control_plane_state)
    control_plane_signature = _control_plane_signature(control_plane_state)
    has_reject = any(str(row.get("status")) == "reject" for row in rows)
    rerun_required_now = bool(control_plane_state.get("rerun_required_now_policy", False)) or bool(
        control_plane_state.get("rerun_required_now_cycle", False)
    )
    keep_blocked = all(
        [
            bool(control_plane_state.get("manifest_keep_blocked", True)),
            bool(control_plane_state.get("continuity_keep_blocked", True)),
            bool(control_plane_state.get("rerun_policy_keep_blocked", True)),
            bool(control_plane_state.get("operational_cycle_keep_blocked", True)),
        ]
    )

    # 条件分岐: `has_reject` を満たす経路を評価する。
    if has_reject:
        overall_status = "reject"
        decision_text = "blocked_interface_stack_control_plane_inconsistent"
        next_action = "repair_control_plane_contract_before_any_refresh"
    # 条件分岐: 前段条件が不成立で、`rerun_required_now` を追加評価する。
    elif rerun_required_now:
        overall_status = "watch"
        decision_text = "blocked_interface_stack_control_plane_rerun_required"
        next_action = "run_mass_origin_blocked_interface_stack_chain_in_order"
    # 条件分岐: 前段条件が不成立で、`keep_blocked` を追加評価する。
    elif keep_blocked:
        overall_status = "watch"
        decision_text = "blocked_interface_stack_control_plane_locked_no_change_hold"
        next_action = "wait_for_new_public_artifact_then_rerun_interface_stack_control_plane_audit"
    else:
        overall_status = "pass"
        decision_text = "blocked_interface_stack_control_plane_unblocked_outside_audit"
        next_action = "evaluate_reopen_transition_now"

    next_required_artifacts = control_plane_state.get("manifest_next_required_artifacts", [])
    interface_stack_complete = all(
        [
            bool(control_plane_state.get("manifest_interface_stack_complete", False)),
            bool(control_plane_state.get("continuity_interface_stack_complete", False)),
            bool(control_plane_state.get("rerun_policy_interface_stack_complete", False)),
            bool(control_plane_state.get("operational_cycle_interface_stack_complete", False)),
        ]
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "mass-origin blocked interface-stack control-plane audit",
        },
        "inputs": {
            "blocked_interface_stack_manifest_json": _rel(MANIFEST_JSON),
            "blocked_interface_stack_continuity_metrics_json": _rel(CONTINUITY_JSON),
            "blocked_interface_stack_rerun_policy_json": _rel(RERUN_POLICY_JSON),
            "blocked_interface_stack_operational_cycle_json": _rel(OPERATIONAL_CYCLE_JSON),
        },
        "intent": "Freeze one top-level control-plane audit for the mass-origin blocked interface stack and verify that the current no-change-hold contract stays aligned.",
        "summary": {
            "manifest_phase_step": str(control_plane_state.get("manifest_phase_step", "")),
            "continuity_phase_step": str(control_plane_state.get("continuity_phase_step", "")),
            "rerun_policy_phase_step": str(control_plane_state.get("rerun_policy_phase_step", "")),
            "operational_cycle_phase_step": str(control_plane_state.get("operational_cycle_phase_step", "")),
            "manifest_status": str(control_plane_state.get("manifest_status", "")),
            "continuity_decision": str(control_plane_state.get("continuity_decision", "")),
            "rerun_policy_decision": str(control_plane_state.get("rerun_policy_decision", "")),
            "operational_cycle_decision": str(control_plane_state.get("operational_cycle_decision", "")),
            "blocked_state_detail": str(control_plane_state.get("manifest_blocked_state_detail", "")),
            "latent_reopen_routes_exhausted": bool(
                control_plane_state.get("manifest_latent_reopen_routes_exhausted", False)
            ),
            "rerun_trigger": str(control_plane_state.get("rerun_trigger_policy", "")),
            "apply_chain_step_count": int(control_plane_state.get("apply_chain_step_count_policy", 0)),
            "apply_chain_signature_sha256": str(control_plane_state.get("apply_chain_signature_policy", "")),
            "manifest_signature_sha256": str(control_plane_state.get("manifest_signature_sha256", "")),
            "continuity_state_signature_sha256": str(control_plane_state.get("continuity_state_signature_sha256", "")),
            "terminal_stack_signature_sha256": str(control_plane_state.get("terminal_stack_signature_sha256", "")),
            "operational_cycle_signature_sha256": str(control_plane_state.get("operational_cycle_signature_sha256", "")),
            "control_plane_signature_sha256": control_plane_signature,
            "terminal_stack_status": str(control_plane_state.get("terminal_stack_status_manifest", "")),
            "continuity_state_changed": bool(control_plane_state.get("continuity_state_changed", False)),
            "rerun_required_now": rerun_required_now,
            "next_required_artifacts": next_required_artifacts,
        },
        "rows": rows,
        "decision": {
            "overall_status": overall_status,
            "decision": decision_text,
            "keep_mass_origin_branch_blocked": keep_blocked,
            "blocked_state_detail": str(control_plane_state.get("manifest_blocked_state_detail", "")),
            "latent_reopen_routes_exhausted": bool(
                control_plane_state.get("manifest_latent_reopen_routes_exhausted", False)
            ),
            "rerun_required_now": rerun_required_now,
            "rerun_trigger": str(control_plane_state.get("rerun_trigger_policy", "")),
            "interface_stack_complete": interface_stack_complete,
            "proceed_to_dark_matter_branch": False,
            "next_required_artifacts": next_required_artifacts,
            "next_action": next_action,
        },
        "evidence": {
            "manifest_artifact": {
                "path": _rel(MANIFEST_JSON),
                "phase_step": str(control_plane_state.get("manifest_phase_step", "")),
                "status": str(control_plane_state.get("manifest_status", "")),
            },
            "continuity_artifact": {
                "path": _rel(CONTINUITY_JSON),
                "phase_step": str(control_plane_state.get("continuity_phase_step", "")),
                "status": str(control_plane_state.get("continuity_decision", "")),
            },
            "rerun_policy_artifact": {
                "path": _rel(RERUN_POLICY_JSON),
                "phase_step": str(control_plane_state.get("rerun_policy_phase_step", "")),
                "status": str(control_plane_state.get("rerun_policy_decision", "")),
            },
            "operational_cycle_artifact": {
                "path": _rel(OPERATIONAL_CYCLE_JSON),
                "phase_step": str(control_plane_state.get("operational_cycle_phase_step", "")),
                "status": str(control_plane_state.get("operational_cycle_decision", "")),
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
    _write_csv(payload["rows"])
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
