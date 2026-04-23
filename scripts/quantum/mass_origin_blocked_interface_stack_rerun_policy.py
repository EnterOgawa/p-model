#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_blocked_interface_stack_rerun_policy.py

Step 8.7.55.2.28:
Freeze the rerun/apply policy for the mass-origin blocked interface stack, so
downstream operators know which public canonical scripts must be rerun, and in
what order, when the blocked interface-stack continuity state changes.

Inputs:
  - output/public/quantum/mass_origin_blocked_interface_stack_continuity_metrics.json
  - output/public/quantum/mass_origin_blocked_interface_stack_manifest.json

Outputs:
  - output/public/quantum/mass_origin_blocked_interface_stack_rerun_policy.json
  - output/public/quantum/mass_origin_blocked_interface_stack_rerun_policy_rows.csv

Assumptions:
  - The blocked interface-stack continuity audit is the sole trigger source for
    deciding whether the full blocked-interface reader chain should be rerun.
  - This policy is descriptive only; it does not execute the chain and does
    not reopen the mass-origin branch by itself.
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
CONTINUITY_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_continuity_metrics.json"
MANIFEST_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_manifest.json"
OUT_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_rerun_policy.json"
OUT_CSV = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_rerun_policy_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.28"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the rerun/apply policy for the mass-origin blocked interface stack.",
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


# 関数: `_command_tokens` の入出力契約と処理意図を定義する。

def _command_tokens(script_rel: str, step_tag: str) -> List[str]:
    return ["python", "-B", script_rel, "--step-tag", step_tag]


# 関数: `_command_text` の入出力契約と処理意図を定義する。

def _command_text(tokens: List[str]) -> str:
    return " ".join(tokens)


# 関数: `_extract_continuity_state` の入出力契約と処理意図を定義する。

def _extract_continuity_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = payload.get("summary", {})
    decision = payload.get("decision", {})
    diagnostics = payload.get("diagnostics", {})
    watchpack = diagnostics.get("continuity_watchpack", {}) if isinstance(diagnostics, dict) else {}

    # 条件分岐: `not isinstance(summary, dict)` を満たす経路を評価する。
    if not isinstance(summary, dict):
        summary = {}

    # 条件分岐: `not isinstance(decision, dict)` を満たす経路を評価する。

    if not isinstance(decision, dict):
        decision = {}

    # 条件分岐: `not isinstance(watchpack, dict)` を満たす経路を評価する。

    if not isinstance(watchpack, dict):
        watchpack = {}

    next_required_artifacts = decision.get("next_required_artifacts", summary.get("next_required_artifacts", []))
    # 条件分岐: `not isinstance(next_required_artifacts, list)` を満たす経路を評価する。
    if not isinstance(next_required_artifacts, list):
        next_required_artifacts = []

    return {
        "continuity_decision": str(decision.get("decision") or "").strip(),
        "continuity_overall_status": str(decision.get("overall_status") or "").strip(),
        "continuity_state_changed": bool(summary.get("continuity_state_changed", False)),
        "continuity_event_counter": int(summary.get("event_counter") or 0),
        "blocked_state_detail": str(decision.get("blocked_state_detail") or summary.get("blocked_state_detail") or "").strip(),
        "latent_reopen_routes_exhausted": bool(
            decision.get("latent_reopen_routes_exhausted", summary.get("latent_reopen_routes_exhausted", False))
        ),
        "keep_mass_origin_branch_blocked": bool(decision.get("keep_mass_origin_branch_blocked", True)),
        "hold_monitor_no_change_hold": bool(decision.get("hold_monitor_no_change_hold", False)),
        "interface_stack_complete": bool(decision.get("interface_stack_complete", False)),
        "next_required_artifacts": [str(item) for item in next_required_artifacts],
        "watchpack_update_event_type": str(watchpack.get("update_event_type") or "").strip(),
        "watchpack_next_action": str(watchpack.get("next_action") or "").strip(),
        "terminal_stack_status": str(summary.get("terminal_stack_status") or "").strip(),
        "terminal_stack_signature_sha256": str(summary.get("terminal_stack_signature_sha256") or "").strip().lower(),
    }


# 関数: `_build_apply_chain_rows` の入出力契約と処理意図を定義する。

def _build_apply_chain_rows(step_tag: str, rerun_required_now: bool) -> List[Dict[str, Any]]:
    chain_specs = [
        {
            "command_id": "refresh_latent_reopen_route_inventory",
            "script": "scripts/quantum/mass_origin_latent_reopen_route_inventory_audit.py",
            "expected_output": "output/public/quantum/mass_origin_latent_reopen_route_inventory_metrics.json",
            "run_when": "interface_stack_continuity_state_changed",
            "settle_required": False,
            "note": "Repo-wide latent-route scan must refresh before any same-sector/Vpp gate reevaluation.",
        },
        {
            "command_id": "refresh_same_sector_vpp_shape_gate",
            "script": "scripts/quantum/mass_origin_same_sector_vpp_shape_gate_audit.py",
            "expected_output": "output/public/quantum/mass_origin_same_sector_vpp_shape_gate_metrics.json",
            "run_when": "after_latent_inventory_refresh",
            "settle_required": False,
            "note": "Recompute named missing-artifact gates after latent inventory changes.",
        },
        {
            "command_id": "refresh_blocked_state_reopen",
            "script": "scripts/quantum/mass_origin_blocked_state_reopen_audit.py",
            "expected_output": "output/public/quantum/mass_origin_blocked_state_reopen_metrics.json",
            "run_when": "after_same_sector_gate_refresh",
            "settle_required": False,
            "note": "Rebuild the blocked-state decision from the refreshed blocker artifacts.",
        },
        {
            "command_id": "settle_blocked_hold_monitor",
            "script": "scripts/quantum/mass_origin_blocked_hold_monitor.py",
            "expected_output": "output/public/quantum/mass_origin_blocked_hold_monitor_metrics.json",
            "run_when": "after_blocked_state_refresh",
            "settle_required": True,
            "note": "Rerun until the hold monitor returns no_change_hold after new upstream hashes are absorbed.",
        },
        {
            "command_id": "refresh_blocked_evidence_pack_manifest",
            "script": "scripts/quantum/mass_origin_blocked_evidence_pack_manifest.py",
            "expected_output": "output/public/quantum/mass_origin_blocked_evidence_pack_manifest.json",
            "run_when": "after_hold_monitor_settles",
            "settle_required": False,
            "note": "Re-freeze the blocked evidence pack after monitor and blocker artifacts settle.",
        },
        {
            "command_id": "settle_blocked_evidence_pack_continuity",
            "script": "scripts/quantum/mass_origin_blocked_evidence_pack_continuity_audit.py",
            "expected_output": "output/public/quantum/mass_origin_blocked_evidence_pack_continuity_metrics.json",
            "run_when": "after_blocked_evidence_pack_manifest_refresh",
            "settle_required": True,
            "note": "Rerun until the evidence-pack continuity artifact returns no_change_hold against the refreshed manifest pack.",
        },
        {
            "command_id": "refresh_blocked_evidence_pack_rerun_policy",
            "script": "scripts/quantum/mass_origin_blocked_evidence_pack_rerun_policy.py",
            "expected_output": "output/public/quantum/mass_origin_blocked_evidence_pack_rerun_policy.json",
            "run_when": "after_blocked_evidence_pack_continuity_settles",
            "settle_required": False,
            "note": "Refresh the evidence-pack rerun policy so the chain signature stays aligned with the settled pack.",
        },
        {
            "command_id": "refresh_blocked_evidence_pack_operational_cycle",
            "script": "scripts/quantum/mass_origin_blocked_evidence_pack_operational_cycle.py",
            "expected_output": "output/public/quantum/mass_origin_blocked_evidence_pack_operational_cycle.json",
            "run_when": "after_blocked_evidence_pack_rerun_policy_refresh",
            "settle_required": False,
            "note": "Recompute the evidence-pack operational-cycle decision after the policy refresh.",
        },
        {
            "command_id": "refresh_blocked_interface_stack_audit",
            "script": "scripts/quantum/mass_origin_blocked_interface_stack_audit.py",
            "expected_output": "output/public/quantum/mass_origin_blocked_interface_stack_metrics.json",
            "run_when": "after_blocked_evidence_pack_operational_cycle_refresh",
            "settle_required": False,
            "note": "Rebuild the stack-level consistency audit from the refreshed blocked-interface layer.",
        },
        {
            "command_id": "refresh_blocked_interface_stack_manifest",
            "script": "scripts/quantum/mass_origin_blocked_interface_stack_manifest.py",
            "expected_output": "output/public/quantum/mass_origin_blocked_interface_stack_manifest.json",
            "run_when": "after_blocked_interface_stack_audit_refresh",
            "settle_required": False,
            "note": "Re-freeze the full blocked interface stack after the stack-level audit is refreshed.",
        },
        {
            "command_id": "settle_blocked_interface_stack_continuity",
            "script": "scripts/quantum/mass_origin_blocked_interface_stack_continuity_audit.py",
            "expected_output": "output/public/quantum/mass_origin_blocked_interface_stack_continuity_metrics.json",
            "run_when": "after_blocked_interface_stack_manifest_refresh",
            "settle_required": True,
            "note": "Rerun until the interface-stack continuity artifact returns no_change_hold against the refreshed stack manifest.",
        },
    ]

    rows: List[Dict[str, Any]] = []
    for index, spec in enumerate(chain_specs, start=1):
        command_tokens = _command_tokens(str(spec["script"]), step_tag)
        rows.append(
            {
                "chain_order": index,
                "command_id": str(spec["command_id"]),
                "status": "required" if rerun_required_now else "idle",
                "run_when": str(spec["run_when"]),
                "settle_required": bool(spec["settle_required"]),
                "command": _command_text(command_tokens),
                "expected_output": str(spec["expected_output"]),
                "note": str(spec["note"]),
            }
        )

    return rows


# 関数: `_chain_signature` の入出力契約と処理意図を定義する。

def _chain_signature(rows: List[Dict[str, Any]]) -> str:
    packed = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(packed).hexdigest().lower()


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    _require_path(CONTINUITY_JSON)
    _require_path(MANIFEST_JSON)
    continuity = _read_json(CONTINUITY_JSON)
    manifest = _read_json(MANIFEST_JSON)
    continuity_state = _extract_continuity_state(continuity)
    rerun_required_now = bool(continuity_state.get("continuity_state_changed", False))
    apply_chain = _build_apply_chain_rows(step_tag, rerun_required_now)
    chain_signature = _chain_signature(apply_chain)
    manifest_phase = manifest.get("phase", {})
    manifest_phase_step = str(manifest_phase.get("step", "")) if isinstance(manifest_phase, dict) else ""

    # 条件分岐: `rerun_required_now` を満たす経路を評価する。
    if rerun_required_now:
        decision_text = "rerun_chain_required_now"
        overall_status = "watch"
        next_action = "run_mass_origin_blocked_interface_stack_chain_in_order"
    # 条件分岐: 前段条件が不成立で、`continuity_state.get(\"continuity_decision\") == \"no_change_hold\"` を追加評価する。
    elif continuity_state.get("continuity_decision") == "no_change_hold":
        decision_text = "policy_locked_no_change_hold"
        overall_status = "watch"
        next_action = "wait_for_new_public_artifact_then_rerun_interface_stack_policy_chain"
    else:
        decision_text = "policy_locked_watch_state"
        overall_status = "watch"
        next_action = str(
            continuity_state.get("watchpack_next_action") or "keep_interface_stack_policy_active_until_continuity_changes"
        )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "mass-origin blocked interface-stack rerun policy",
        },
        "inputs": {
            "blocked_interface_stack_continuity_metrics_json": _rel(CONTINUITY_JSON),
            "blocked_interface_stack_manifest_json": _rel(MANIFEST_JSON),
        },
        "intent": "Freeze the rerun/apply chain that must be used when the mass-origin blocked interface-stack continuity state changes.",
        "summary": {
            "continuity_decision": str(continuity_state.get("continuity_decision", "")),
            "continuity_overall_status": str(continuity_state.get("continuity_overall_status", "")),
            "continuity_state_changed": bool(continuity_state.get("continuity_state_changed", False)),
            "continuity_event_counter": int(continuity_state.get("continuity_event_counter", 0)),
            "manifest_phase_step": manifest_phase_step,
            "terminal_stack_status": str(continuity_state.get("terminal_stack_status", "")),
            "terminal_stack_signature_sha256": str(continuity_state.get("terminal_stack_signature_sha256", "")),
            "apply_chain_step_count": len(apply_chain),
            "apply_chain_signature_sha256": chain_signature,
            "blocked_state_detail": str(continuity_state.get("blocked_state_detail", "")),
            "latent_reopen_routes_exhausted": bool(continuity_state.get("latent_reopen_routes_exhausted", False)),
            "hold_monitor_no_change_hold": bool(continuity_state.get("hold_monitor_no_change_hold", False)),
            "rerun_required_now": rerun_required_now,
            "next_required_artifacts": continuity_state.get("next_required_artifacts", []),
        },
        "apply_chain": apply_chain,
        "decision": {
            "overall_status": overall_status,
            "decision": decision_text,
            "rerun_required_now": rerun_required_now,
            "rerun_trigger": "blocked_interface_stack_continuity_state_changed",
            "interface_stack_complete": bool(continuity_state.get("interface_stack_complete", False)),
            "keep_mass_origin_branch_blocked": bool(continuity_state.get("keep_mass_origin_branch_blocked", True)),
            "blocked_state_detail": str(continuity_state.get("blocked_state_detail", "")),
            "latent_reopen_routes_exhausted": bool(continuity_state.get("latent_reopen_routes_exhausted", False)),
            "hold_monitor_no_change_hold": bool(continuity_state.get("hold_monitor_no_change_hold", False)),
            "proceed_to_dark_matter_branch": False,
            "next_required_artifacts": continuity_state.get("next_required_artifacts", []),
            "next_action": next_action,
        },
        "evidence": {
            "continuity_artifact": {
                "path": _rel(CONTINUITY_JSON),
                "phase_step": str((continuity.get("phase") or {}).get("step", "")),
                "decision": str((continuity.get("decision") or {}).get("decision", "")),
            },
            "manifest_artifact": {
                "path": _rel(MANIFEST_JSON),
                "phase_step": manifest_phase_step,
                "decision": str((manifest.get("decision") or {}).get("overall_status", "")),
            },
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "chain_order",
                "command_id",
                "status",
                "run_when",
                "settle_required",
                "command",
                "expected_output",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    args = _parse_args()
    payload = _build_payload(str(args.step_tag))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["apply_chain"])
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
