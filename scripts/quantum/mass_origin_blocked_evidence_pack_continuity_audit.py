#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_blocked_evidence_pack_continuity_audit.py

Step 8.7.55.2.22:
Freeze a downstream continuity audit for the mass-origin blocked evidence pack,
so later reopen attempts can detect whether the blocked pack changed as a
single unit instead of re-reading each upstream artifact manually.

Inputs:
  - output/public/quantum/mass_origin_blocked_evidence_pack_manifest.json
  - output/public/quantum/mass_origin_blocked_evidence_pack_manifest_rows.csv

Outputs:
  - output/public/quantum/mass_origin_blocked_evidence_pack_continuity_metrics.json
  - output/public/quantum/mass_origin_blocked_evidence_pack_continuity_rows.csv

Assumptions:
  - The blocked evidence-pack manifest is the single public canonical summary
    of the current mass-origin blocked state.
  - This audit is downstream only; it does not change blocked-state logic and
    only records whether the evidence pack changed as a whole.
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
MANIFEST_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_evidence_pack_manifest.json"
MANIFEST_ROWS_CSV = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_evidence_pack_manifest_rows.csv"
OUT_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_evidence_pack_continuity_metrics.json"
OUT_CSV = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_evidence_pack_continuity_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.22"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a downstream continuity audit for the mass-origin blocked evidence pack.",
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


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: `_file_signature` の入出力契約と処理意図を定義する。

def _file_signature(path: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "path": _rel(path),
        "exists": path.exists(),
        "size_bytes": None,
        "mtime_utc": None,
        "sha256": None,
    }

    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return payload

    stat = path.stat()
    payload["size_bytes"] = int(stat.st_size)
    payload["mtime_utc"] = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    payload["sha256"] = _sha256_file(path)
    return payload


# 関数: `_load_previous_watchpack` の入出力契約と処理意図を定義する。

def _load_previous_watchpack() -> Dict[str, Any]:
    # 条件分岐: `not OUT_JSON.exists()` を満たす経路を評価する。
    if not OUT_JSON.exists():
        return {}

    try:
        payload = _read_json(OUT_JSON)
    except Exception:
        return {}

    diagnostics = payload.get("diagnostics", {})
    # 条件分岐: `not isinstance(diagnostics, dict)` を満たす経路を評価する。
    if not isinstance(diagnostics, dict):
        return {}

    watchpack = diagnostics.get("continuity_watchpack", {})
    # 条件分岐: `not isinstance(watchpack, dict)` を満たす経路を評価する。
    if not isinstance(watchpack, dict):
        return {}

    return watchpack


# 関数: `_count_csv_rows` の入出力契約と処理意図を定義する。

def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


# 関数: `_extract_logic_state` の入出力契約と処理意図を定義する。

def _extract_logic_state(manifest: Dict[str, Any], rows_csv_count: int) -> Dict[str, Any]:
    summary = manifest.get("summary", {})
    decision = manifest.get("decision", {})

    # 条件分岐: `not isinstance(summary, dict)` を満たす経路を評価する。
    if not isinstance(summary, dict):
        summary = {}

    # 条件分岐: `not isinstance(decision, dict)` を満たす経路を評価する。

    if not isinstance(decision, dict):
        decision = {}

    next_required_artifacts = decision.get("next_required_artifacts", summary.get("next_required_artifacts", []))
    # 条件分岐: `not isinstance(next_required_artifacts, list)` を満たす経路を評価する。
    if not isinstance(next_required_artifacts, list):
        next_required_artifacts = []

    artifact_count = int(summary.get("artifact_count") or 0)
    return {
        "artifact_count": artifact_count,
        "rows_csv_artifact_count": int(rows_csv_count),
        "rows_csv_matches_manifest": bool(rows_csv_count == artifact_count),
        "manifest_signature_sha256": str(summary.get("manifest_signature_sha256") or "").strip().lower(),
        "blocked_state_detail": str(
            decision.get("blocked_state_detail", summary.get("blocked_state_detail", ""))
        ).strip(),
        "latent_reopen_routes_exhausted": bool(
            decision.get("latent_reopen_routes_exhausted", summary.get("latent_reopen_routes_exhausted", False))
        ),
        "hold_monitor_status": str(summary.get("hold_monitor_status") or "").strip(),
        "evidence_pack_complete": bool(decision.get("evidence_pack_complete", False)),
        "keep_mass_origin_branch_blocked": bool(decision.get("keep_mass_origin_branch_blocked", True)),
        "hold_monitor_no_change_hold": bool(decision.get("hold_monitor_no_change_hold", False)),
        "next_required_artifacts": [str(item) for item in next_required_artifacts],
    }


# 関数: `_state_signature` の入出力契約と処理意図を定義する。

def _state_signature(logic_state: Dict[str, Any]) -> str:
    packed = json.dumps(logic_state, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(packed).hexdigest().lower()


# 関数: `_derive_continuity_watchpack` の入出力契約と処理意図を定義する。

def _derive_continuity_watchpack(
    *,
    current_input_signatures: Dict[str, Dict[str, Any]],
    previous_watchpack: Dict[str, Any],
    logic_state: Dict[str, Any],
) -> Dict[str, Any]:
    previous_input_signatures = (
        previous_watchpack.get("input_signatures")
        if isinstance(previous_watchpack.get("input_signatures"), dict)
        else {}
    )
    previous_state_signature = str(previous_watchpack.get("continuity_state_signature_sha256") or "").strip().lower()
    current_state_signature = _state_signature(logic_state)

    input_hash_changed = False
    input_metadata_changed_without_hash_change = False
    changed_inputs: List[str] = []
    baseline_initialized_now = not previous_input_signatures

    for input_id, current_signature in current_input_signatures.items():
        previous_signature = (
            previous_input_signatures.get(input_id, {})
            if isinstance(previous_input_signatures, dict)
            else {}
        )
        current_exists = bool(current_signature.get("exists"))
        previous_exists = bool(previous_signature.get("exists"))
        current_sha = str(current_signature.get("sha256") or "").strip().lower()
        previous_sha = str(previous_signature.get("sha256") or "").strip().lower()
        hash_changed = current_exists and previous_exists and bool(current_sha) and bool(previous_sha) and current_sha != previous_sha

        # 条件分岐: `hash_changed` を満たす経路を評価する。
        if hash_changed:
            input_hash_changed = True
            changed_inputs.append(str(input_id))

        # 条件分岐: `current_exists and previous_exists and (not hash_changed) and current_sha == previous_sha` を満たす経路を評価する。

        if current_exists and previous_exists and (not hash_changed) and current_sha == previous_sha:
            current_mtime = str(current_signature.get("mtime_utc") or "").strip()
            previous_mtime = str(previous_signature.get("mtime_utc") or "").strip()
            current_size = current_signature.get("size_bytes")
            previous_size = previous_signature.get("size_bytes")
            if (current_mtime and previous_mtime and current_mtime != previous_mtime) or (current_size != previous_size):
                input_metadata_changed_without_hash_change = True

    continuity_state_changed = bool(previous_state_signature) and current_state_signature != previous_state_signature

    # 条件分岐: `baseline_initialized_now` を満たす経路を評価する。
    if baseline_initialized_now:
        update_event_type = "baseline_initialized"
    # 条件分岐: 前段条件が不成立で、`continuity_state_changed` を追加評価する。
    elif continuity_state_changed:
        update_event_type = "blocked_evidence_pack_state_changed"
    # 条件分岐: 前段条件が不成立で、`input_hash_changed` を追加評価する。
    elif input_hash_changed:
        update_event_type = "input_hash_changed_state_same"
    # 条件分岐: 前段条件が不成立で、`input_metadata_changed_without_hash_change` を追加評価する。
    elif input_metadata_changed_without_hash_change:
        update_event_type = "metadata_changed_hash_same"
    else:
        update_event_type = "no_change"

    update_event_detected = continuity_state_changed
    event_counter_prev = int(previous_watchpack.get("event_counter", 0)) if previous_watchpack else 0
    event_counter = event_counter_prev + 1 if update_event_detected else event_counter_prev

    # 条件分岐: `continuity_state_changed` を満たす経路を評価する。
    if continuity_state_changed:
        next_action = "rerun_blocked_evidence_pack_readers_now"
    # 条件分岐: 前段条件が不成立で、`logic_state.get(\"keep_mass_origin_branch_blocked\", True)` を追加評価する。
    elif logic_state.get("keep_mass_origin_branch_blocked", True):
        next_action = "wait_for_new_public_artifact_then_rerun_continuity"
    else:
        next_action = "evaluate_reopen_transition_now"

    return {
        "input_signatures": current_input_signatures,
        "previous_input_signatures": previous_input_signatures,
        "changed_inputs": changed_inputs,
        "changed_inputs_n": len(changed_inputs),
        "input_hash_changed": input_hash_changed,
        "input_metadata_changed_without_hash_change": input_metadata_changed_without_hash_change,
        "baseline_initialized_now": baseline_initialized_now,
        "continuity_state_signature_sha256": current_state_signature,
        "previous_continuity_state_signature_sha256": previous_state_signature or None,
        "continuity_state_changed": continuity_state_changed,
        "update_event_detected": update_event_detected,
        "update_event_type": update_event_type,
        "event_counter": event_counter,
        "next_action": next_action,
        "note": (
            "Event counter increments only when the logical blocked evidence-pack state changes. "
            "Hash-only refreshes are recorded without incrementing the counter."
        ),
    }


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows(logic_state: Dict[str, Any], watchpack: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocked_state_detail = str(logic_state.get("blocked_state_detail") or "")
    state_settled = not bool(watchpack.get("continuity_state_changed", False))
    next_required_artifacts = logic_state.get("next_required_artifacts", [])
    next_required_note = ", ".join(str(item) for item in next_required_artifacts) if next_required_artifacts else "none"

    return [
        {
            "row_id": "blocked_evidence_pack_manifest_signature_present",
            "status": "pass" if str(logic_state.get("manifest_signature_sha256") or "") else "reject",
            "metric": "blocked evidence-pack manifest signature is present",
            "value": 1.0 if str(logic_state.get("manifest_signature_sha256") or "") else 0.0,
            "note": "Continuity audit requires a stable manifest signature from the blocked evidence-pack manifest.",
        },
        {
            "row_id": "blocked_evidence_pack_rows_sync",
            "status": "pass" if bool(logic_state.get("rows_csv_matches_manifest", False)) else "reject",
            "metric": "manifest rows CSV count matches manifest summary count",
            "value": 1.0 if bool(logic_state.get("rows_csv_matches_manifest", False)) else 0.0,
            "note": (
                f"Manifest summary artifact_count = {logic_state.get('artifact_count')}, "
                f"rows CSV artifact_count = {logic_state.get('rows_csv_artifact_count')}."
            ),
        },
        {
            "row_id": "blocked_state_detail_specific_missing_artifacts_fixed",
            "status": "pass" if blocked_state_detail == "specific_missing_artifacts_fixed" else "watch",
            "metric": "blocked-state detail propagated into blocked evidence-pack continuity",
            "value": 1.0 if blocked_state_detail == "specific_missing_artifacts_fixed" else 0.0,
            "note": f"Current blocked_state_detail = {blocked_state_detail}.",
        },
        {
            "row_id": "blocked_evidence_pack_complete",
            "status": "pass" if bool(logic_state.get("evidence_pack_complete", False)) else "reject",
            "metric": "blocked evidence pack remains complete",
            "value": 1.0 if bool(logic_state.get("evidence_pack_complete", False)) else 0.0,
            "note": "Continuity audit assumes the blocked evidence pack remains complete before downstream readers rely on it.",
        },
        {
            "row_id": "mass_origin_branch_blocked_in_manifest",
            "status": "blocked" if bool(logic_state.get("keep_mass_origin_branch_blocked", True)) else "pass",
            "metric": "manifest keeps the mass-origin branch blocked",
            "value": 1.0 if bool(logic_state.get("keep_mass_origin_branch_blocked", True)) else 0.0,
            "note": "The continuity artifact mirrors the current blocked decision and does not reopen the branch on its own.",
        },
        {
            "row_id": "latent_reopen_routes_exhausted",
            "status": "pass" if bool(logic_state.get("latent_reopen_routes_exhausted", False)) else "reject",
            "metric": "repo-wide latent reopen routes remain exhausted",
            "value": 1.0 if bool(logic_state.get("latent_reopen_routes_exhausted", False)) else 0.0,
            "note": "No latent public route exists for same-sector chi_P -> V''(|P|_*) or for a non-phenomenological single public V(|P|) shape.",
        },
        {
            "row_id": "hold_monitor_no_change_hold",
            "status": "pass" if bool(logic_state.get("hold_monitor_no_change_hold", False)) else "watch",
            "metric": "hold monitor remains settled at no_change_hold",
            "value": 1.0 if bool(logic_state.get("hold_monitor_no_change_hold", False)) else 0.0,
            "note": f"Current hold_monitor_status = {logic_state.get('hold_monitor_status')}.",
        },
        {
            "row_id": "blocked_evidence_pack_continuity_state_settled",
            "status": "pass" if state_settled else "watch",
            "metric": "blocked evidence-pack logical state is settled",
            "value": 1.0 if state_settled else 0.0,
            "note": (
                "Continuity state is unchanged from the previous continuity snapshot."
                if state_settled
                else "Continuity state changed and downstream readers should be reevaluated."
            ),
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
    _require_path(MANIFEST_JSON)
    _require_path(MANIFEST_ROWS_CSV)

    manifest = _read_json(MANIFEST_JSON)
    rows_csv_count = _count_csv_rows(MANIFEST_ROWS_CSV)
    logic_state = _extract_logic_state(manifest, rows_csv_count)
    current_input_signatures = {
        "blocked_evidence_pack_manifest_json": _file_signature(MANIFEST_JSON),
        "blocked_evidence_pack_manifest_rows_csv": _file_signature(MANIFEST_ROWS_CSV),
    }
    previous_watchpack = _load_previous_watchpack()
    watchpack = _derive_continuity_watchpack(
        current_input_signatures=current_input_signatures,
        previous_watchpack=previous_watchpack,
        logic_state=logic_state,
    )
    rows = _build_rows(logic_state, watchpack)
    has_reject = any(str(row.get("status")) == "reject" for row in rows)

    # 条件分岐: `has_reject` を満たす経路を評価する。
    if has_reject:
        overall_status = "reject"
        decision_text = "blocked_evidence_pack_sync_broken"
    # 条件分岐: 前段条件が不成立で、`watchpack.get(\"update_event_type\") == \"baseline_initialized\"` を追加評価する。
    elif watchpack.get("update_event_type") == "baseline_initialized":
        overall_status = "watch"
        decision_text = "baseline_initialized"
    # 条件分岐: 前段条件が不成立で、`bool(watchpack.get(\"continuity_state_changed\", False))` を追加評価する。
    elif bool(watchpack.get("continuity_state_changed", False)):
        overall_status = "watch"
        decision_text = "blocked_evidence_pack_state_changed"
    # 条件分岐: 前段条件が不成立で、`bool(watchpack.get(\"input_hash_changed\", False)) or bool(watchpack.get(\"input_metadata_changed_without_hash_change\", False))` を追加評価する。
    elif bool(watchpack.get("input_hash_changed", False)) or bool(
        watchpack.get("input_metadata_changed_without_hash_change", False)
    ):
        overall_status = "watch"
        decision_text = "upstream_refresh_status_same"
    # 条件分岐: 前段条件が不成立で、`bool(logic_state.get(\"keep_mass_origin_branch_blocked\", True))` を追加評価する。
    elif bool(logic_state.get("keep_mass_origin_branch_blocked", True)):
        overall_status = "watch"
        decision_text = "no_change_hold"
    else:
        overall_status = "pass"
        decision_text = "reopen_transition_detected"

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "mass-origin blocked evidence-pack continuity audit",
        },
        "inputs": {
            "blocked_evidence_pack_manifest_json": _rel(MANIFEST_JSON),
            "blocked_evidence_pack_manifest_rows_csv": _rel(MANIFEST_ROWS_CSV),
        },
        "intent": "Track whether the mass-origin blocked evidence pack changes as a single public canonical unit and keep downstream readers aligned with the current blocked state.",
        "summary": {
            "artifact_count": int(logic_state.get("artifact_count", 0)),
            "rows_csv_artifact_count": int(logic_state.get("rows_csv_artifact_count", 0)),
            "rows_csv_matches_manifest": bool(logic_state.get("rows_csv_matches_manifest", False)),
            "manifest_signature_sha256": str(logic_state.get("manifest_signature_sha256", "")),
            "blocked_state_detail": str(logic_state.get("blocked_state_detail", "")),
            "latent_reopen_routes_exhausted": bool(logic_state.get("latent_reopen_routes_exhausted", False)),
            "hold_monitor_status": str(logic_state.get("hold_monitor_status", "")),
            "continuity_state_signature_sha256": str(watchpack.get("continuity_state_signature_sha256", "")),
            "continuity_state_changed": bool(watchpack.get("continuity_state_changed", False)),
            "event_counter": int(watchpack.get("event_counter", 0)),
            "next_required_artifacts": logic_state.get("next_required_artifacts", []),
        },
        "rows": rows,
        "diagnostics": {
            "input_signatures": current_input_signatures,
            "continuity_watchpack": watchpack,
        },
        "decision": {
            "overall_status": overall_status,
            "decision": decision_text,
            "evidence_pack_complete": bool(logic_state.get("evidence_pack_complete", False)),
            "keep_mass_origin_branch_blocked": bool(logic_state.get("keep_mass_origin_branch_blocked", True)),
            "blocked_state_detail": str(logic_state.get("blocked_state_detail", "")),
            "latent_reopen_routes_exhausted": bool(logic_state.get("latent_reopen_routes_exhausted", False)),
            "hold_monitor_no_change_hold": bool(logic_state.get("hold_monitor_no_change_hold", False)),
            "proceed_to_dark_matter_branch": False,
            "next_required_artifacts": logic_state.get("next_required_artifacts", []),
            "next_action": str(watchpack.get("next_action", "")),
        },
        "evidence": {
            "source_manifest": {
                "path": _rel(MANIFEST_JSON),
                "phase_step": str((manifest.get("phase") or {}).get("step", "")),
                "overall_status": str((manifest.get("decision") or {}).get("overall_status", "")),
            },
            "source_rows_csv": {
                "path": _rel(MANIFEST_ROWS_CSV),
                "row_count": int(rows_csv_count),
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
