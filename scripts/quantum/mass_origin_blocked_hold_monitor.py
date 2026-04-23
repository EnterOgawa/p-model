#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_blocked_hold_monitor.py

Step 8.7.55.2.5:
Monitor the blocked-hold state of the mass-origin branch and rerun only when
the public canonical evidence pack changes.

Inputs:
  - output/public/quantum/mass_origin_readiness_gate_metrics.json
  - output/public/quantum/mass_origin_curvature_boundary_metrics.json
  - output/public/quantum/mass_origin_solver_spec_gate_metrics.json
  - output/public/quantum/mass_origin_blocked_state_reopen_metrics.json
  - output/public/quantum/mass_origin_same_sector_vpp_shape_gate_metrics.json
  - output/public/quantum/mass_origin_latent_reopen_route_inventory_metrics.json

Outputs:
  - output/public/quantum/mass_origin_blocked_hold_monitor_metrics.json
  - output/public/quantum/mass_origin_blocked_hold_monitor_rows.csv
  - output/public/quantum/mass_origin_blocked_hold_monitor_status_rows.csv

Assumptions:
  - The mass-origin branch remains blocked until the reopen criteria in
    `mass_origin_blocked_state_reopen_metrics.json` become satisfied.
  - This monitor is operational only; it does not change the blocked-state
    logic and only detects whether the evidence pack changed.
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

READINESS_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_readiness_gate_metrics.json"
CURVATURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_curvature_boundary_metrics.json"
SOLVER_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_solver_spec_gate_metrics.json"
REOPEN_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_blocked_state_reopen_metrics.json"
SPECIFIC_GATE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_vpp_shape_gate_metrics.json"
LATENT_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_latent_reopen_route_inventory_metrics.json"
OUT_DIR = ROOT / "output" / "public" / "quantum"
OUT_JSON = OUT_DIR / "mass_origin_blocked_hold_monitor_metrics.json"
OUT_CSV = OUT_DIR / "mass_origin_blocked_hold_monitor_rows.csv"
OUT_STATUS_CSV = OUT_DIR / "mass_origin_blocked_hold_monitor_status_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.5"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor the blocked-hold state of the mass-origin branch.",
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


# 関数: `_sha256_file` の入出力契約と処理意図を定義する。

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


# 関数: `_relative_str` の入出力契約と処理意図を定義する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: `_load_previous_expected_hashes` の入出力契約と処理意図を定義する。

def _load_previous_expected_hashes() -> Dict[str, str]:
    # 条件分岐: `not OUT_JSON.exists()` を満たす経路を評価する。
    if not OUT_JSON.exists():
        return {}

    payload = _read_json(OUT_JSON)
    rows = payload.get("input_watch_rows", [])

    # 条件分岐: `not isinstance(rows, list)` を満たす経路を評価する。
    if not isinstance(rows, list):
        return {}

    expected: Dict[str, str] = {}
    for row in rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        input_id = str(row.get("input_id", ""))
        sha256 = str(row.get("current_sha256", "")).lower()

        # 条件分岐: `input_id and sha256` を満たす経路を評価する。
        if input_id and sha256:
            expected[input_id] = sha256

    return expected


# 関数: `_collect_inputs` の入出力契約と処理意図を定義する。

def _collect_inputs() -> List[Dict[str, Any]]:
    return [
        {"input_id": "mass_origin_readiness_gate_json", "path": READINESS_JSON},
        {"input_id": "mass_origin_curvature_boundary_json", "path": CURVATURE_JSON},
        {"input_id": "mass_origin_solver_spec_gate_json", "path": SOLVER_JSON},
        {"input_id": "mass_origin_blocked_state_reopen_json", "path": REOPEN_JSON},
        {"input_id": "mass_origin_same_sector_vpp_shape_gate_json", "path": SPECIFIC_GATE_JSON},
        {"input_id": "mass_origin_latent_reopen_route_inventory_json", "path": LATENT_INVENTORY_JSON},
    ]


# 関数: `_build_input_watch_rows` の入出力契約と処理意図を定義する。

def _build_input_watch_rows(previous_expected: Dict[str, str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in _collect_inputs():
        input_id = str(item["input_id"])
        path = Path(item["path"])
        _require_path(path)
        current = _sha256_file(path)
        expected = previous_expected.get(input_id, current)
        changed = current != expected
        rows.append(
            {
                "input_id": input_id,
                "path": _relative_str(path),
                "exists": True,
                "expected_sha256": expected,
                "current_sha256": current,
                "hash_changed": changed,
            }
        )

    return rows


# 関数: `_build_status_rows` の入出力契約と処理意図を定義する。

def _build_status_rows(
    hash_changed: bool,
    blocked_state_detail: str,
    latent_reopen_routes_exhausted: bool,
    blocked_state: Dict[str, Any],
    next_required_artifacts: List[str],
) -> List[Dict[str, Any]]:
    unsatisfied_requirements = blocked_state.get("unsatisfied_requirements", [])
    unsatisfied_count = int(blocked_state.get("reopen_requirement_unsatisfied_count", 0))
    blocked = bool(blocked_state.get("mass_origin_branch_blocked", True))
    settled = not hash_changed
    next_required_note = ", ".join(str(item) for item in next_required_artifacts) if next_required_artifacts else "none"
    unsatisfied_note = ", ".join(str(item) for item in unsatisfied_requirements) if unsatisfied_requirements else "none"

    return [
        {
            "row_id": "mass_origin_branch_blocked_hold_state",
            "status": "blocked" if blocked else "pass",
            "metric": "mass-origin branch remains on blocked hold",
            "value": 1.0 if blocked else 0.0,
            "note": "Hold remains active while the reopen criteria stay unsatisfied.",
        },
        {
            "row_id": "monitor_decision_no_change_hold",
            "status": "pass" if settled else "watch",
            "metric": "blocked-hold monitor settled without new watched inputs",
            "value": 1.0 if settled else 0.0,
            "note": "The watchpack is stable and the monitor can keep the current blocked hold." if settled else "At least one watched input changed; rerun the gate stack before keeping the hold.",
        },
        {
            "row_id": "blocked_state_detail_specific_missing_artifacts_fixed",
            "status": "pass" if blocked_state_detail == "specific_missing_artifacts_fixed" else "watch",
            "metric": "blocked-state detail propagated to monitor",
            "value": 1.0 if blocked_state_detail == "specific_missing_artifacts_fixed" else 0.0,
            "note": f"Current blocked_state_detail = {blocked_state_detail}.",
        },
        {
            "row_id": "latent_reopen_routes_exhausted",
            "status": "pass" if latent_reopen_routes_exhausted else "reject",
            "metric": "repo-wide latent reopen routes exhausted",
            "value": 1.0 if latent_reopen_routes_exhausted else 0.0,
            "note": "No positive same-sector public row or non-phenomenological public V(|P|) ansatz has appeared in the current public canonical pack.",
        },
        {
            "row_id": "reopen_requirement_unsatisfied_count",
            "status": "blocked" if unsatisfied_count > 0 else "pass",
            "metric": "unsatisfied reopen requirements count",
            "value": float(unsatisfied_count),
            "note": f"Unsatisfied requirements: {unsatisfied_note}.",
        },
        {
            "row_id": "next_required_artifacts_count",
            "status": "inventory",
            "metric": "next required artifacts count",
            "value": float(len(next_required_artifacts)),
            "note": f"Next required artifacts: {next_required_note}.",
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    previous_expected = _load_previous_expected_hashes()
    watch_rows = _build_input_watch_rows(previous_expected)
    changed_inputs = [row["input_id"] for row in watch_rows if bool(row["hash_changed"])]
    hash_changed = bool(changed_inputs)
    update_event_type = "input_hash_changed" if hash_changed else "no_change"

    reopen_payload = _read_json(REOPEN_JSON)
    reopen_summary = reopen_payload.get("summary", {})
    reopen_decision = reopen_payload.get("decision", {})
    blocked_state_detail = str(
        reopen_summary.get(
            "blocked_state_detail",
            reopen_decision.get("blocked_state_detail", "generic_blocked_state"),
        )
    )
    latent_reopen_routes_exhausted = bool(reopen_summary.get("latent_reopen_routes_exhausted", False))
    next_required_artifacts = reopen_decision.get("next_required_artifacts", [])
    previous_payload = _read_json(OUT_JSON) if OUT_JSON.exists() else {}
    previous_counter = int(previous_payload.get("monitor", {}).get("event_counter", 0)) if isinstance(previous_payload, dict) else 0
    event_counter = previous_counter + 1 if hash_changed else previous_counter
    blocked_state = {
        "blocked_state_fixed": bool(reopen_summary.get("blocked_state_fixed", False)),
        "mass_origin_branch_blocked": bool(reopen_decision.get("mass_origin_branch_blocked", True)),
        "mass_origin_branch_reopen_ready": bool(reopen_decision.get("mass_origin_branch_reopen_ready", False)),
        "blocked_state_detail": blocked_state_detail,
        "latent_reopen_routes_exhausted": latent_reopen_routes_exhausted,
        "reopen_requirement_unsatisfied_count": int(reopen_summary.get("reopen_requirement_unsatisfied_count", 0)),
        "unsatisfied_requirements": reopen_summary.get("unsatisfied_requirements", []),
        "next_required_artifacts": next_required_artifacts,
    }
    status_rows = _build_status_rows(
        hash_changed=hash_changed,
        blocked_state_detail=blocked_state_detail,
        latent_reopen_routes_exhausted=latent_reopen_routes_exhausted,
        blocked_state=blocked_state,
        next_required_artifacts=next_required_artifacts,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "mass-origin blocked-hold monitor",
        },
        "inputs": {
            "mass_origin_readiness_gate_json": _relative_str(READINESS_JSON),
            "mass_origin_curvature_boundary_json": _relative_str(CURVATURE_JSON),
            "mass_origin_solver_spec_gate_json": _relative_str(SOLVER_JSON),
            "mass_origin_blocked_state_reopen_json": _relative_str(REOPEN_JSON),
            "mass_origin_same_sector_vpp_shape_gate_json": _relative_str(SPECIFIC_GATE_JSON),
            "mass_origin_latent_reopen_route_inventory_json": _relative_str(LATENT_INVENTORY_JSON),
        },
        "intent": "Keep the mass-origin branch on blocked hold until the public canonical evidence pack changes and the reopen criteria can be reevaluated.",
        "monitor": {
            "input_hash_changed": hash_changed,
            "changed_inputs_n": len(changed_inputs),
            "changed_inputs": changed_inputs,
            "update_event_type": update_event_type,
            "event_counter": event_counter,
            "rerun_required": hash_changed,
            "rerun_policy": "rerun_mass_origin_branch_only_if_input_hash_changed",
            "action_taken": "rerun_blocked_state_gate_pending" if hash_changed else "skip_rerun_keep_blocked_hold",
        },
        "input_watch_rows": watch_rows,
        "blocked_state": blocked_state,
        "status_rows": status_rows,
        "decision": {
            "overall_status": "watch",
            "decision": "input_hash_changed_rerun_required" if hash_changed else "no_change_hold",
            "mass_origin_branch_blocked": bool(reopen_decision.get("mass_origin_branch_blocked", True)),
            "mass_origin_branch_reopen_ready": bool(reopen_decision.get("mass_origin_branch_reopen_ready", False)),
            "blocked_state_detail": blocked_state_detail,
            "latent_reopen_routes_exhausted": latent_reopen_routes_exhausted,
            "proceed_to_dark_matter_branch": False,
            "next_required_artifacts": next_required_artifacts,
            "next_action": "rerun_8.7.55.2.1-.4_equivalent_gate_stack" if hash_changed else "wait_for_new_public_artifact_then_rerun_monitor",
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["input_id", "path", "exists", "expected_sha256", "current_sha256", "hash_changed"],
        )
        writer.writeheader()
        writer.writerows(rows)


# 関数: `_write_status_csv` の入出力契約と処理意図を定義する。

def _write_status_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_STATUS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    args = _parse_args()
    payload = _build_payload(str(args.step_tag))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["input_watch_rows"])
    _write_status_csv(payload["status_rows"])
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
