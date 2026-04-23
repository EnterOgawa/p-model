#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_blocked_interface_stack_audit.py

Step 8.7.55.2.25:
Freeze one stack-level consistency audit for the current public blocked
interfaces of the mass-origin branch, so downstream readers can confirm that
blocked-state, hold-monitor, manifest, continuity, rerun policy, and
operational-cycle artifacts all agree on the present no-change-hold state.

Inputs:
  - output/public/quantum/mass_origin_blocked_state_reopen_metrics.json
  - output/public/quantum/mass_origin_blocked_hold_monitor_metrics.json
  - output/public/quantum/mass_origin_blocked_evidence_pack_manifest.json
  - output/public/quantum/mass_origin_blocked_evidence_pack_continuity_metrics.json
  - output/public/quantum/mass_origin_blocked_evidence_pack_rerun_policy.json
  - output/public/quantum/mass_origin_blocked_evidence_pack_operational_cycle.json

Outputs:
  - output/public/quantum/mass_origin_blocked_interface_stack_metrics.json
  - output/public/quantum/mass_origin_blocked_interface_stack_rows.csv

Assumptions:
  - This audit is descriptive only; it does not execute the rerun chain and
    does not reopen the mass-origin branch.
  - The stack is considered consistent only if blocked-state detail, latent
    reopen-route exhaustion, next required artifacts, and blocked / no-change
    hold decisions remain aligned across all public blocked interfaces.
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
BLOCKED_STATE_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_state_reopen_metrics.json"
HOLD_MONITOR_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_hold_monitor_metrics.json"
PACK_MANIFEST_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_evidence_pack_manifest.json"
CONTINUITY_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_evidence_pack_continuity_metrics.json"
RERUN_POLICY_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_evidence_pack_rerun_policy.json"
OPERATIONAL_CYCLE_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_evidence_pack_operational_cycle.json"
OUT_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_metrics.json"
OUT_CSV = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.25"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the stack-level consistency audit for the mass-origin blocked public interfaces.",
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


# 関数: `_extract_phase_step` の入出力契約と処理意図を定義する。

def _extract_phase_step(payload: Dict[str, Any]) -> str:
    phase = _as_dict(payload, "phase")
    return str(phase.get("step", "")).strip()


# 関数: `_extract_decision_text` の入出力契約と処理意図を定義する。

def _extract_decision_text(payload: Dict[str, Any]) -> str:
    decision = _as_dict(payload, "decision")
    decision_text = str(decision.get("decision", "")).strip()

    # 条件分岐: `decision_text` を満たす経路を評価する。
    if decision_text:
        return decision_text

    return str(decision.get("overall_status", "")).strip()


# 関数: `_extract_blocked_state_detail` の入出力契約と処理意図を定義する。

def _extract_blocked_state_detail(payload: Dict[str, Any]) -> str:
    decision = _as_dict(payload, "decision")
    summary = _as_dict(payload, "summary")
    blocked_state = _as_dict(payload, "blocked_state")

    for source in (decision, summary, blocked_state):
        value = str(source.get("blocked_state_detail", "")).strip()

        # 条件分岐: `value` を満たす経路を評価する。
        if value:
            return value

    return ""


# 関数: `_extract_latent_reopen_routes_exhausted` の入出力契約と処理意図を定義する。

def _extract_latent_reopen_routes_exhausted(payload: Dict[str, Any]) -> bool:
    decision = _as_dict(payload, "decision")
    summary = _as_dict(payload, "summary")
    blocked_state = _as_dict(payload, "blocked_state")

    for source in (decision, summary, blocked_state):
        # 条件分岐: `"latent_reopen_routes_exhausted" in source` を満たす経路を評価する。
        if "latent_reopen_routes_exhausted" in source:
            return bool(source.get("latent_reopen_routes_exhausted", False))

    return False


# 関数: `_extract_next_required_artifacts` の入出力契約と処理意図を定義する。

def _extract_next_required_artifacts(payload: Dict[str, Any]) -> List[str]:
    decision = _as_dict(payload, "decision")
    summary = _as_dict(payload, "summary")
    blocked_state = _as_dict(payload, "blocked_state")

    for source in (decision, summary, blocked_state):
        value = source.get("next_required_artifacts", [])

        # 条件分岐: `isinstance(value, list)` を満たす経路を評価する。
        if isinstance(value, list):
            return [str(item) for item in value]

    return []


# 関数: `_extract_keep_blocked` の入出力契約と処理意図を定義する。

def _extract_keep_blocked(payload: Dict[str, Any]) -> bool:
    decision = _as_dict(payload, "decision")
    blocked_state = _as_dict(payload, "blocked_state")

    for key in ("keep_mass_origin_branch_blocked", "mass_origin_branch_blocked"):
        # 条件分岐: `key in decision` を満たす経路を評価する。
        if key in decision:
            return bool(decision.get(key))

        # 条件分岐: `key in blocked_state` を満たす経路を評価する。

        if key in blocked_state:
            return bool(blocked_state.get(key))

    return True


# 関数: `_extract_hold_monitor_decision` の入出力契約と処理意図を定義する。

def _extract_hold_monitor_decision(payload: Dict[str, Any]) -> str:
    decision = _as_dict(payload, "decision")
    overall_status = str(decision.get("overall_status", "")).strip()
    decision_text = str(decision.get("decision", "")).strip()

    # 条件分岐: `overall_status and decision_text` を満たす経路を評価する。
    if overall_status and decision_text:
        return f"{overall_status}:{decision_text}"

    return decision_text or overall_status


# 関数: `_extract_manifest_status` の入出力契約と処理意図を定義する。

def _extract_manifest_status(payload: Dict[str, Any]) -> str:
    decision = _as_dict(payload, "decision")
    return str(decision.get("overall_status", "")).strip()


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


# 関数: `_extract_stack_state` の入出力契約と処理意図を定義する。

def _extract_stack_state(
    blocked_state: Dict[str, Any],
    hold_monitor: Dict[str, Any],
    manifest: Dict[str, Any],
    continuity: Dict[str, Any],
    rerun_policy: Dict[str, Any],
    operational_cycle: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "blocked_state_phase_step": _extract_phase_step(blocked_state),
        "hold_monitor_phase_step": _extract_phase_step(hold_monitor),
        "manifest_phase_step": _extract_phase_step(manifest),
        "continuity_phase_step": _extract_phase_step(continuity),
        "rerun_policy_phase_step": _extract_phase_step(rerun_policy),
        "operational_cycle_phase_step": _extract_phase_step(operational_cycle),
        "blocked_state_status": _extract_decision_text(blocked_state),
        "hold_monitor_status": _extract_hold_monitor_decision(hold_monitor),
        "manifest_status": _extract_manifest_status(manifest),
        "continuity_decision": _extract_decision_text(continuity),
        "rerun_policy_decision": _extract_decision_text(rerun_policy),
        "operational_cycle_decision": _extract_decision_text(operational_cycle),
        "blocked_state_detail": _extract_blocked_state_detail(blocked_state),
        "hold_monitor_blocked_state_detail": _extract_blocked_state_detail(hold_monitor),
        "manifest_blocked_state_detail": _extract_blocked_state_detail(manifest),
        "continuity_blocked_state_detail": _extract_blocked_state_detail(continuity),
        "rerun_policy_blocked_state_detail": _extract_blocked_state_detail(rerun_policy),
        "operational_cycle_blocked_state_detail": _extract_blocked_state_detail(operational_cycle),
        "blocked_state_latent_reopen_routes_exhausted": _extract_latent_reopen_routes_exhausted(blocked_state),
        "hold_monitor_latent_reopen_routes_exhausted": _extract_latent_reopen_routes_exhausted(hold_monitor),
        "manifest_latent_reopen_routes_exhausted": _extract_latent_reopen_routes_exhausted(manifest),
        "continuity_latent_reopen_routes_exhausted": _extract_latent_reopen_routes_exhausted(continuity),
        "rerun_policy_latent_reopen_routes_exhausted": _extract_latent_reopen_routes_exhausted(rerun_policy),
        "operational_cycle_latent_reopen_routes_exhausted": _extract_latent_reopen_routes_exhausted(operational_cycle),
        "blocked_state_next_required_artifacts": _extract_next_required_artifacts(blocked_state),
        "hold_monitor_next_required_artifacts": _extract_next_required_artifacts(hold_monitor),
        "manifest_next_required_artifacts": _extract_next_required_artifacts(manifest),
        "continuity_next_required_artifacts": _extract_next_required_artifacts(continuity),
        "rerun_policy_next_required_artifacts": _extract_next_required_artifacts(rerun_policy),
        "operational_cycle_next_required_artifacts": _extract_next_required_artifacts(operational_cycle),
        "blocked_state_keep_blocked": _extract_keep_blocked(blocked_state),
        "hold_monitor_keep_blocked": _extract_keep_blocked(hold_monitor),
        "manifest_keep_blocked": _extract_keep_blocked(manifest),
        "continuity_keep_blocked": _extract_keep_blocked(continuity),
        "rerun_policy_keep_blocked": _extract_keep_blocked(rerun_policy),
        "operational_cycle_keep_blocked": _extract_keep_blocked(operational_cycle),
        "continuity_state_changed": _extract_bool(
            continuity,
            [["summary", "continuity_state_changed"], ["decision", "continuity_state_changed"]],
            default=False,
        ),
        "rerun_required_now_policy": _extract_bool(rerun_policy, [["decision", "rerun_required_now"]], default=False),
        "rerun_required_now_cycle": _extract_bool(operational_cycle, [["decision", "rerun_required_now"]], default=False),
        "apply_chain_step_count_policy": int(_extract_str(rerun_policy, [["summary", "apply_chain_step_count"]]) or 0),
        "apply_chain_step_count_cycle": int(_extract_str(operational_cycle, [["summary", "apply_chain_step_count"]]) or 0),
        "apply_chain_signature_policy": _extract_str(rerun_policy, [["summary", "apply_chain_signature_sha256"]]).lower(),
        "apply_chain_signature_cycle": _extract_str(
            operational_cycle,
            [["summary", "apply_chain_signature_sha256"]],
        ).lower(),
        "cycle_signature_sha256": _extract_str(operational_cycle, [["summary", "cycle_signature_sha256"]]).lower(),
        "manifest_signature_sha256": _extract_str(manifest, [["summary", "manifest_signature_sha256"]]).lower(),
        "continuity_state_signature_sha256": _extract_str(
            continuity,
            [["summary", "continuity_state_signature_sha256"]],
        ).lower(),
    }


# 関数: `_stack_signature` の入出力契約と処理意図を定義する。

def _stack_signature(stack_state: Dict[str, Any]) -> str:
    packed = json.dumps(stack_state, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(packed).hexdigest().lower()


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows(stack_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    detail_values = [
        stack_state.get("blocked_state_detail", ""),
        stack_state.get("hold_monitor_blocked_state_detail", ""),
        stack_state.get("manifest_blocked_state_detail", ""),
        stack_state.get("continuity_blocked_state_detail", ""),
        stack_state.get("rerun_policy_blocked_state_detail", ""),
        stack_state.get("operational_cycle_blocked_state_detail", ""),
    ]
    latent_values = [
        bool(stack_state.get("blocked_state_latent_reopen_routes_exhausted", False)),
        bool(stack_state.get("hold_monitor_latent_reopen_routes_exhausted", False)),
        bool(stack_state.get("manifest_latent_reopen_routes_exhausted", False)),
        bool(stack_state.get("continuity_latent_reopen_routes_exhausted", False)),
        bool(stack_state.get("rerun_policy_latent_reopen_routes_exhausted", False)),
        bool(stack_state.get("operational_cycle_latent_reopen_routes_exhausted", False)),
    ]
    blocked_flags = [
        bool(stack_state.get("blocked_state_keep_blocked", True)),
        bool(stack_state.get("hold_monitor_keep_blocked", True)),
        bool(stack_state.get("manifest_keep_blocked", True)),
        bool(stack_state.get("continuity_keep_blocked", True)),
        bool(stack_state.get("rerun_policy_keep_blocked", True)),
        bool(stack_state.get("operational_cycle_keep_blocked", True)),
    ]
    next_required_lists = [
        stack_state.get("blocked_state_next_required_artifacts", []),
        stack_state.get("hold_monitor_next_required_artifacts", []),
        stack_state.get("manifest_next_required_artifacts", []),
        stack_state.get("continuity_next_required_artifacts", []),
        stack_state.get("rerun_policy_next_required_artifacts", []),
        stack_state.get("operational_cycle_next_required_artifacts", []),
    ]
    canonical_next_required = list(next_required_lists[0])
    next_required_consistent = all(list(items) == canonical_next_required for items in next_required_lists)
    detail_consistent = all(value == detail_values[0] for value in detail_values)
    latent_consistent = all(value == latent_values[0] for value in latent_values)
    blocked_consistent = all(value for value in blocked_flags)
    apply_chain_signature_match = (
        bool(stack_state.get("apply_chain_signature_policy", ""))
        and stack_state.get("apply_chain_signature_policy") == stack_state.get("apply_chain_signature_cycle")
    )
    rerun_required_now_consistent = (
        not bool(stack_state.get("rerun_required_now_policy", False))
        and not bool(stack_state.get("rerun_required_now_cycle", False))
    )

    return [
        {
            "row_id": "blocked_state_detail_consistent",
            "status": "pass" if detail_consistent else "reject",
            "metric": "blocked-state detail is consistent across blocked interfaces",
            "value": 1.0 if detail_consistent else 0.0,
            "note": f"Current detail set = {detail_values}.",
        },
        {
            "row_id": "next_required_artifacts_consistent",
            "status": "pass" if next_required_consistent else "reject",
            "metric": "next required artifacts are consistent across blocked interfaces",
            "value": 1.0 if next_required_consistent else 0.0,
            "note": f"Canonical next required artifacts = {canonical_next_required}.",
        },
        {
            "row_id": "latent_reopen_routes_exhausted_consistent",
            "status": "pass" if latent_consistent and latent_values[0] else "reject",
            "metric": "latent reopen-route exhaustion is consistent across blocked interfaces",
            "value": 1.0 if latent_consistent and latent_values[0] else 0.0,
            "note": f"Current latent-route flags = {latent_values}.",
        },
        {
            "row_id": "mass_origin_branch_blocked_all_layers",
            "status": "blocked" if blocked_consistent else "reject",
            "metric": "mass-origin branch remains blocked across all blocked interfaces",
            "value": 1.0 if blocked_consistent else 0.0,
            "note": f"Current keep-blocked flags = {blocked_flags}.",
        },
        {
            "row_id": "blocked_state_artifact_still_blocked",
            "status": "pass" if stack_state.get("blocked_state_status") == "specific_missing_artifacts_fixed_still_blocked" else "watch",
            "metric": "blocked-state artifact keeps the specific-missing-artifacts-fixed block",
            "value": 1.0 if stack_state.get("blocked_state_status") == "specific_missing_artifacts_fixed_still_blocked" else 0.0,
            "note": f"Current blocked-state status = {stack_state.get('blocked_state_status')}.",
        },
        {
            "row_id": "hold_monitor_no_change_hold",
            "status": "pass" if stack_state.get("hold_monitor_status") == "watch:no_change_hold" else "watch",
            "metric": "hold monitor remains settled at no_change_hold",
            "value": 1.0 if stack_state.get("hold_monitor_status") == "watch:no_change_hold" else 0.0,
            "note": f"Current hold-monitor status = {stack_state.get('hold_monitor_status')}.",
        },
        {
            "row_id": "blocked_evidence_pack_manifest_frozen",
            "status": "pass" if stack_state.get("manifest_status") == "blocked_evidence_pack_manifest_frozen" else "watch",
            "metric": "blocked evidence-pack manifest remains frozen",
            "value": 1.0 if stack_state.get("manifest_status") == "blocked_evidence_pack_manifest_frozen" else 0.0,
            "note": f"Current manifest status = {stack_state.get('manifest_status')}.",
        },
        {
            "row_id": "continuity_no_change_hold",
            "status": "pass" if stack_state.get("continuity_decision") == "no_change_hold" else "watch",
            "metric": "continuity artifact remains on no-change hold",
            "value": 1.0 if stack_state.get("continuity_decision") == "no_change_hold" else 0.0,
            "note": f"Current continuity decision = {stack_state.get('continuity_decision')}.",
        },
        {
            "row_id": "rerun_policy_locked_no_change_hold",
            "status": "pass" if stack_state.get("rerun_policy_decision") == "policy_locked_no_change_hold" else "watch",
            "metric": "rerun policy remains locked on no-change hold",
            "value": 1.0 if stack_state.get("rerun_policy_decision") == "policy_locked_no_change_hold" else 0.0,
            "note": f"Current rerun-policy decision = {stack_state.get('rerun_policy_decision')}.",
        },
        {
            "row_id": "operational_cycle_no_change_hold",
            "status": "pass" if stack_state.get("operational_cycle_decision") == "operational_cycle_no_change_hold" else "watch",
            "metric": "operational cycle remains on no-change hold",
            "value": 1.0 if stack_state.get("operational_cycle_decision") == "operational_cycle_no_change_hold" else 0.0,
            "note": f"Current operational-cycle decision = {stack_state.get('operational_cycle_decision')}.",
        },
        {
            "row_id": "apply_chain_signature_consistent",
            "status": "pass" if apply_chain_signature_match else "reject",
            "metric": "rerun/apply chain signature matches between policy and operational cycle",
            "value": 1.0 if apply_chain_signature_match else 0.0,
            "note": (
                "Current signatures = "
                f"{stack_state.get('apply_chain_signature_policy')} / {stack_state.get('apply_chain_signature_cycle')}."
            ),
        },
        {
            "row_id": "rerun_required_now_false_consistent",
            "status": "pass" if rerun_required_now_consistent else "reject",
            "metric": "rerun-required-now remains false across reader layers",
            "value": 1.0 if rerun_required_now_consistent else 0.0,
            "note": (
                "Current rerun-required flags = "
                f"{stack_state.get('rerun_required_now_policy')} / {stack_state.get('rerun_required_now_cycle')}."
            ),
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
    for path in (
        BLOCKED_STATE_JSON,
        HOLD_MONITOR_JSON,
        PACK_MANIFEST_JSON,
        CONTINUITY_JSON,
        RERUN_POLICY_JSON,
        OPERATIONAL_CYCLE_JSON,
    ):
        _require_path(path)

    blocked_state = _read_json(BLOCKED_STATE_JSON)
    hold_monitor = _read_json(HOLD_MONITOR_JSON)
    manifest = _read_json(PACK_MANIFEST_JSON)
    continuity = _read_json(CONTINUITY_JSON)
    rerun_policy = _read_json(RERUN_POLICY_JSON)
    operational_cycle = _read_json(OPERATIONAL_CYCLE_JSON)

    stack_state = _extract_stack_state(
        blocked_state,
        hold_monitor,
        manifest,
        continuity,
        rerun_policy,
        operational_cycle,
    )
    rows = _build_rows(stack_state)
    stack_signature = _stack_signature(stack_state)
    has_reject = any(str(row.get("status")) == "reject" for row in rows)
    rerun_required_now = bool(stack_state.get("rerun_required_now_policy", False)) or bool(
        stack_state.get("rerun_required_now_cycle", False)
    )
    keep_blocked = bool(stack_state.get("blocked_state_keep_blocked", True))

    # 条件分岐: `has_reject` を満たす経路を評価する。
    if has_reject:
        overall_status = "reject"
        decision_text = "blocked_interface_stack_inconsistent"
        next_action = "repair_stack_inconsistency_before_any_rerun"
    # 条件分岐: 前段条件が不成立で、`rerun_required_now` を追加評価する。
    elif rerun_required_now:
        overall_status = "watch"
        decision_text = "blocked_interface_stack_rerun_required"
        next_action = "run_mass_origin_blocked_evidence_pack_chain_in_order"
    # 条件分岐: 前段条件が不成立で、`keep_blocked` を追加評価する。
    elif keep_blocked:
        overall_status = "watch"
        decision_text = "blocked_interface_stack_locked_no_change_hold"
        next_action = "wait_for_new_public_artifact_then_rerun_interface_stack_audit"
    else:
        overall_status = "pass"
        decision_text = "blocked_interface_stack_unblocked_outside_audit"
        next_action = "evaluate_reopen_transition_now"

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "mass-origin blocked interface stack audit",
        },
        "inputs": {
            "mass_origin_blocked_state_reopen_json": _rel(BLOCKED_STATE_JSON),
            "mass_origin_blocked_hold_monitor_json": _rel(HOLD_MONITOR_JSON),
            "mass_origin_blocked_evidence_pack_manifest_json": _rel(PACK_MANIFEST_JSON),
            "mass_origin_blocked_evidence_pack_continuity_metrics_json": _rel(CONTINUITY_JSON),
            "mass_origin_blocked_evidence_pack_rerun_policy_json": _rel(RERUN_POLICY_JSON),
            "mass_origin_blocked_evidence_pack_operational_cycle_json": _rel(OPERATIONAL_CYCLE_JSON),
        },
        "intent": "Freeze one stack-level consistency audit for the current public blocked interfaces of the mass-origin branch.",
        "summary": {
            "blocked_state_phase_step": str(stack_state.get("blocked_state_phase_step", "")),
            "hold_monitor_phase_step": str(stack_state.get("hold_monitor_phase_step", "")),
            "manifest_phase_step": str(stack_state.get("manifest_phase_step", "")),
            "continuity_phase_step": str(stack_state.get("continuity_phase_step", "")),
            "rerun_policy_phase_step": str(stack_state.get("rerun_policy_phase_step", "")),
            "operational_cycle_phase_step": str(stack_state.get("operational_cycle_phase_step", "")),
            "blocked_state_status": str(stack_state.get("blocked_state_status", "")),
            "hold_monitor_status": str(stack_state.get("hold_monitor_status", "")),
            "manifest_status": str(stack_state.get("manifest_status", "")),
            "continuity_decision": str(stack_state.get("continuity_decision", "")),
            "rerun_policy_decision": str(stack_state.get("rerun_policy_decision", "")),
            "operational_cycle_decision": str(stack_state.get("operational_cycle_decision", "")),
            "blocked_state_detail": str(stack_state.get("blocked_state_detail", "")),
            "latent_reopen_routes_exhausted": bool(stack_state.get("blocked_state_latent_reopen_routes_exhausted", False)),
            "apply_chain_step_count": int(stack_state.get("apply_chain_step_count_policy", 0)),
            "apply_chain_signature_sha256": str(stack_state.get("apply_chain_signature_policy", "")),
            "manifest_signature_sha256": str(stack_state.get("manifest_signature_sha256", "")),
            "continuity_state_signature_sha256": str(stack_state.get("continuity_state_signature_sha256", "")),
            "operational_cycle_signature_sha256": str(stack_state.get("cycle_signature_sha256", "")),
            "interface_stack_signature_sha256": stack_signature,
            "next_required_artifacts": stack_state.get("blocked_state_next_required_artifacts", []),
        },
        "rows": rows,
        "decision": {
            "overall_status": overall_status,
            "decision": decision_text,
            "keep_mass_origin_branch_blocked": keep_blocked,
            "blocked_state_detail": str(stack_state.get("blocked_state_detail", "")),
            "latent_reopen_routes_exhausted": bool(stack_state.get("blocked_state_latent_reopen_routes_exhausted", False)),
            "rerun_required_now": rerun_required_now,
            "proceed_to_dark_matter_branch": False,
            "next_required_artifacts": stack_state.get("blocked_state_next_required_artifacts", []),
            "next_action": next_action,
        },
        "evidence": {
            "blocked_state_artifact": {
                "path": _rel(BLOCKED_STATE_JSON),
                "phase_step": str(stack_state.get("blocked_state_phase_step", "")),
                "status": str(stack_state.get("blocked_state_status", "")),
            },
            "hold_monitor_artifact": {
                "path": _rel(HOLD_MONITOR_JSON),
                "phase_step": str(stack_state.get("hold_monitor_phase_step", "")),
                "status": str(stack_state.get("hold_monitor_status", "")),
            },
            "manifest_artifact": {
                "path": _rel(PACK_MANIFEST_JSON),
                "phase_step": str(stack_state.get("manifest_phase_step", "")),
                "status": str(stack_state.get("manifest_status", "")),
            },
            "continuity_artifact": {
                "path": _rel(CONTINUITY_JSON),
                "phase_step": str(stack_state.get("continuity_phase_step", "")),
                "status": str(stack_state.get("continuity_decision", "")),
            },
            "rerun_policy_artifact": {
                "path": _rel(RERUN_POLICY_JSON),
                "phase_step": str(stack_state.get("rerun_policy_phase_step", "")),
                "status": str(stack_state.get("rerun_policy_decision", "")),
            },
            "operational_cycle_artifact": {
                "path": _rel(OPERATIONAL_CYCLE_JSON),
                "phase_step": str(stack_state.get("operational_cycle_phase_step", "")),
                "status": str(stack_state.get("operational_cycle_decision", "")),
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
