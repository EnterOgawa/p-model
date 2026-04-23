#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_blocked_interface_stack_manifest.py

Step 8.7.55.2.26:
Freeze a single public manifest for the current blocked-interface stack of the
mass-origin branch, so downstream readers can inspect the complete blocked
interface layer without opening each blocked JSON artifact manually.

Inputs:
  - output/public/quantum/mass_origin_blocked_state_reopen_metrics.json
  - output/public/quantum/mass_origin_blocked_hold_monitor_metrics.json
  - output/public/quantum/mass_origin_blocked_evidence_pack_manifest.json
  - output/public/quantum/mass_origin_blocked_evidence_pack_continuity_metrics.json
  - output/public/quantum/mass_origin_blocked_evidence_pack_rerun_policy.json
  - output/public/quantum/mass_origin_blocked_evidence_pack_operational_cycle.json
  - output/public/quantum/mass_origin_blocked_interface_stack_metrics.json

Outputs:
  - output/public/quantum/mass_origin_blocked_interface_stack_manifest.json
  - output/public/quantum/mass_origin_blocked_interface_stack_manifest_rows.csv

Assumptions:
  - The blocked-interface stack is complete only if every current public
    blocked interface and the stack-level audit artifact exist together.
  - This manifest is descriptive only; it does not execute reruns and does not
    reopen the mass-origin branch.
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
STACK_AUDIT_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_metrics.json"
OUT_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_manifest.json"
OUT_CSV = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_manifest_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.26"

ARTIFACT_SPECS = [
    {
        "artifact_id": "mass_origin_blocked_state_reopen",
        "role": "blocked_interface",
        "path": BLOCKED_STATE_JSON,
        "note": "Blocked-state artifact for the mass-origin branch.",
    },
    {
        "artifact_id": "mass_origin_blocked_hold_monitor",
        "role": "blocked_interface",
        "path": HOLD_MONITOR_JSON,
        "note": "Operational blocked-hold monitor JSON.",
    },
    {
        "artifact_id": "mass_origin_blocked_evidence_pack_manifest",
        "role": "blocked_interface",
        "path": PACK_MANIFEST_JSON,
        "note": "Evidence-pack manifest for the blocked branch.",
    },
    {
        "artifact_id": "mass_origin_blocked_evidence_pack_continuity",
        "role": "blocked_interface",
        "path": CONTINUITY_JSON,
        "note": "Continuity audit for the blocked evidence pack.",
    },
    {
        "artifact_id": "mass_origin_blocked_evidence_pack_rerun_policy",
        "role": "blocked_interface",
        "path": RERUN_POLICY_JSON,
        "note": "Rerun/apply policy for the blocked evidence pack.",
    },
    {
        "artifact_id": "mass_origin_blocked_evidence_pack_operational_cycle",
        "role": "blocked_interface",
        "path": OPERATIONAL_CYCLE_JSON,
        "note": "Operational-cycle decision for the blocked evidence pack.",
    },
    {
        "artifact_id": "mass_origin_blocked_interface_stack_audit",
        "role": "stack_audit",
        "path": STACK_AUDIT_JSON,
        "note": "Stack-level consistency audit across all blocked interfaces.",
    },
]


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a single manifest for the current mass-origin blocked-interface stack.",
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


# 関数: `_as_dict` の入出力契約と処理意図を定義する。

def _as_dict(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = payload.get(key, {})
    return value if isinstance(value, dict) else {}


# 関数: `_extract_phase_step` の入出力契約と処理意図を定義する。

def _extract_phase_step(payload: Dict[str, Any]) -> str:
    phase = _as_dict(payload, "phase")
    return str(phase.get("step", "")).strip()


# 関数: `_extract_artifact_status` の入出力契約と処理意図を定義する。

def _extract_artifact_status(payload: Dict[str, Any]) -> str:
    decision = _as_dict(payload, "decision")
    overall_status = str(decision.get("overall_status", "")).strip()
    decision_text = str(decision.get("decision", "")).strip()

    # 条件分岐: `overall_status and decision_text` を満たす経路を評価する。
    if overall_status and decision_text:
        return f"{overall_status}:{decision_text}"

    # 条件分岐: `overall_status` を満たす経路を評価する。

    if overall_status:
        return overall_status

    return decision_text or "present"


# 関数: `_extract_blocked_state_detail` の入出力契約と処理意図を定義する。

def _extract_blocked_state_detail(payload: Dict[str, Any]) -> str:
    for source_name in ("decision", "summary", "blocked_state"):
        source = _as_dict(payload, source_name)
        value = str(source.get("blocked_state_detail", "")).strip()

        # 条件分岐: `value` を満たす経路を評価する。
        if value:
            return value

    return ""


# 関数: `_extract_latent_reopen_routes_exhausted` の入出力契約と処理意図を定義する。

def _extract_latent_reopen_routes_exhausted(payload: Dict[str, Any]) -> str:
    for source_name in ("decision", "summary", "blocked_state"):
        source = _as_dict(payload, source_name)

        # 条件分岐: `"latent_reopen_routes_exhausted" in source` を満たす経路を評価する。
        if "latent_reopen_routes_exhausted" in source:
            return "true" if bool(source.get("latent_reopen_routes_exhausted", False)) else "false"

    return ""


# 関数: `_extract_next_required_artifacts` の入出力契約と処理意図を定義する。

def _extract_next_required_artifacts(payload: Dict[str, Any]) -> List[str]:
    for source_name in ("decision", "summary", "blocked_state"):
        source = _as_dict(payload, source_name)
        value = source.get("next_required_artifacts", [])

        # 条件分岐: `isinstance(value, list)` を満たす経路を評価する。
        if isinstance(value, list):
            return [str(item) for item in value]

    return []


# 関数: `_extract_keep_blocked` の入出力契約と処理意図を定義する。

def _extract_keep_blocked(payload: Dict[str, Any]) -> bool:
    for source_name in ("decision", "blocked_state"):
        source = _as_dict(payload, source_name)
        for key in ("keep_mass_origin_branch_blocked", "mass_origin_branch_blocked"):
            # 条件分岐: `key in source` を満たす経路を評価する。
            if key in source:
                return bool(source.get(key))

    return True


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in ARTIFACT_SPECS:
        path = Path(spec["path"])
        _require_path(path)
        payload = _read_json(path)
        next_required_artifacts = _extract_next_required_artifacts(payload)
        rows.append(
            {
                "row_id": str(spec["artifact_id"]),
                "status": "pass",
                "metric": f"{spec['role']} artifact present in blocked interface stack",
                "value": 1.0,
                "artifact_path": _rel(path),
                "artifact_sha256": _sha256_file(path),
                "artifact_phase_step": _extract_phase_step(payload),
                "artifact_status": _extract_artifact_status(payload),
                "blocked_state_detail": _extract_blocked_state_detail(payload),
                "latent_reopen_routes_exhausted": _extract_latent_reopen_routes_exhausted(payload),
                "next_required_artifacts_count": len(next_required_artifacts),
                "note": str(spec["note"]),
            }
        )

    return rows


# 関数: `_manifest_signature` の入出力契約と処理意図を定義する。

def _manifest_signature(rows: List[Dict[str, Any]]) -> str:
    packed = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(packed).hexdigest().lower()


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    rows = _build_rows()
    manifest_signature = _manifest_signature(rows)
    terminal_row = next(row for row in rows if row["row_id"] == "mass_origin_blocked_interface_stack_audit")
    terminal_payload = _read_json(STACK_AUDIT_JSON)
    terminal_decision = _as_dict(terminal_payload, "decision")
    terminal_summary = _as_dict(terminal_payload, "summary")
    next_required_artifacts = _extract_next_required_artifacts(terminal_payload)
    keep_blocked = _extract_keep_blocked(terminal_payload)
    hold_monitor_payload = _read_json(HOLD_MONITOR_JSON)
    hold_monitor_decision = _as_dict(hold_monitor_payload, "decision")

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "mass-origin blocked interface stack manifest",
        },
        "inputs": {
            "public_quantum_dir": _rel(PUBLIC_QUANTUM_DIR),
            "artifact_count": len(rows),
        },
        "intent": "Freeze the current blocked-interface stack of the mass-origin branch as a single public canonical manifest.",
        "rows": rows,
        "summary": {
            "artifact_count": len(rows),
            "json_artifact_count": len(rows),
            "manifest_signature_sha256": manifest_signature,
            "blocked_state_detail": _extract_blocked_state_detail(terminal_payload),
            "latent_reopen_routes_exhausted": bool(_extract_latent_reopen_routes_exhausted(terminal_payload) == "true"),
            "terminal_stack_status": str(terminal_row.get("artifact_status", "")),
            "terminal_stack_signature_sha256": str(terminal_summary.get("interface_stack_signature_sha256", "")),
            "hold_monitor_status": str(_extract_artifact_status(hold_monitor_payload)),
            "next_required_artifacts": next_required_artifacts,
        },
        "decision": {
            "overall_status": "blocked_interface_stack_manifest_frozen",
            "interface_stack_complete": True,
            "keep_mass_origin_branch_blocked": keep_blocked,
            "blocked_state_detail": _extract_blocked_state_detail(terminal_payload),
            "latent_reopen_routes_exhausted": bool(_extract_latent_reopen_routes_exhausted(terminal_payload) == "true"),
            "hold_monitor_no_change_hold": str(hold_monitor_decision.get("decision", "")).strip() == "no_change_hold",
            "proceed_to_dark_matter_branch": False,
            "next_required_artifacts": next_required_artifacts,
            "next_action": "wait_for_new_public_artifact_then_refresh_interface_stack_manifest",
        },
        "evidence": {
            "terminal_stack_audit": {
                "path": _rel(STACK_AUDIT_JSON),
                "phase_step": _extract_phase_step(terminal_payload),
                "artifact_status": str(terminal_row.get("artifact_status", "")),
            },
            "hold_monitor_artifact": {
                "path": _rel(HOLD_MONITOR_JSON),
                "phase_step": _extract_phase_step(hold_monitor_payload),
                "artifact_status": _extract_artifact_status(hold_monitor_payload),
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
                "row_id",
                "status",
                "metric",
                "value",
                "artifact_path",
                "artifact_sha256",
                "artifact_phase_step",
                "artifact_status",
                "blocked_state_detail",
                "latent_reopen_routes_exhausted",
                "next_required_artifacts_count",
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
    _write_csv(payload["rows"])
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
