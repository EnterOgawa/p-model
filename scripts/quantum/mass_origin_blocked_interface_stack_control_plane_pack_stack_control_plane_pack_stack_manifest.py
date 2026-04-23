#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_manifest.py

Step 8.7.55.2.56:
Freeze a single public manifest for the current blocked interface-stack
control-plane pack stack control-plane pack stack of the mass-origin branch,
so downstream readers can inspect the complete top-level control-plane pack
stack without reopening each JSON artifact manually.

Inputs:
  - output/public/quantum/mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_manifest.json
  - output/public/quantum/mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_continuity_metrics.json
  - output/public/quantum/mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_rerun_policy.json
  - output/public/quantum/mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_operational_cycle.json
  - output/public/quantum/mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_metrics.json

Outputs:
  - output/public/quantum/mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_manifest.json
  - output/public/quantum/mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_manifest_rows.csv

Assumptions:
  - The control-plane pack stack is complete only if every current top-level
    control-plane pack artifact and the terminal control-plane pack-stack
    audit exist together.
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
CONTROL_PLANE_PACK_MANIFEST_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_manifest.json"
CONTROL_PLANE_PACK_CONTINUITY_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_continuity_metrics.json"
CONTROL_PLANE_PACK_RERUN_POLICY_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_rerun_policy.json"
CONTROL_PLANE_PACK_OPERATIONAL_CYCLE_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_operational_cycle.json"
CONTROL_PLANE_PACK_STACK_AUDIT_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_metrics.json"
OUT_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_manifest.json"
OUT_CSV = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_manifest_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.56"

ARTIFACT_SPECS = [
    {
        "artifact_id": "mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_manifest",
        "role": "control_plane_pack_stack_control_plane_pack_stack_input",
        "path": CONTROL_PLANE_PACK_MANIFEST_JSON,
        "note": "Frozen manifest for the current top-level control-plane pack layer.",
    },
    {
        "artifact_id": "mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_continuity",
        "role": "control_plane_pack_stack_control_plane_pack_stack_input",
        "path": CONTROL_PLANE_PACK_CONTINUITY_JSON,
        "note": "Continuity audit for the current top-level control-plane pack layer.",
    },
    {
        "artifact_id": "mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_rerun_policy",
        "role": "control_plane_pack_stack_control_plane_pack_stack_input",
        "path": CONTROL_PLANE_PACK_RERUN_POLICY_JSON,
        "note": "Rerun/apply policy for the current top-level control-plane pack layer.",
    },
    {
        "artifact_id": "mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_operational_cycle",
        "role": "control_plane_pack_stack_control_plane_pack_stack_input",
        "path": CONTROL_PLANE_PACK_OPERATIONAL_CYCLE_JSON,
        "note": "Operational-cycle decision for the current top-level control-plane pack layer.",
    },
    {
        "artifact_id": "mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_audit",
        "role": "control_plane_pack_stack_control_plane_pack_stack_terminal",
        "path": CONTROL_PLANE_PACK_STACK_AUDIT_JSON,
        "note": "Top-level control-plane pack stack audit across the current control-plane pack stack.",
    },
]


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a single manifest for the current mass-origin interface-stack control-plane pack stack control-plane pack stack.",
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
                "metric": f"{spec['role']} artifact present in control-plane pack stack manifest",
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
    terminal_row = next(
        row
        for row in rows
        if row["row_id"] == "mass_origin_blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_audit"
    )
    terminal_payload = _read_json(CONTROL_PLANE_PACK_STACK_AUDIT_JSON)
    terminal_summary = _as_dict(terminal_payload, "summary")
    next_required_artifacts = _extract_next_required_artifacts(terminal_payload)
    keep_blocked = _extract_keep_blocked(terminal_payload)
    continuity_payload = _read_json(CONTROL_PLANE_PACK_CONTINUITY_JSON)
    continuity_decision = _as_dict(continuity_payload, "decision")

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "mass-origin blocked interface-stack control-plane pack stack control-plane pack stack manifest",
        },
        "inputs": {
            "public_quantum_dir": _rel(PUBLIC_QUANTUM_DIR),
            "artifact_count": len(rows),
        },
        "intent": "Freeze the current blocked interface-stack control-plane pack stack control-plane pack stack of the mass-origin branch as a single public canonical manifest.",
        "rows": rows,
        "summary": {
            "artifact_count": len(rows),
            "json_artifact_count": len(rows),
            "manifest_signature_sha256": manifest_signature,
            "blocked_state_detail": _extract_blocked_state_detail(terminal_payload),
            "latent_reopen_routes_exhausted": bool(_extract_latent_reopen_routes_exhausted(terminal_payload) == "true"),
            "terminal_control_plane_pack_status": str(terminal_summary.get("terminal_control_plane_pack_status", "")),
            "control_plane_pack_signature_sha256": str(terminal_summary.get("control_plane_pack_signature_sha256", "")),
            "control_plane_pack_stack_signature_sha256": str(
                terminal_summary.get("control_plane_pack_stack_signature_sha256", "")
            ),
            "control_plane_status": str(terminal_summary.get("control_plane_status", "")),
            "continuity_status": _extract_artifact_status(continuity_payload),
            "pack_stack_audit_status": str(terminal_row.get("artifact_status", "")),
            "next_required_artifacts": next_required_artifacts,
        },
        "decision": {
            "overall_status": "blocked_interface_stack_control_plane_pack_stack_control_plane_pack_stack_manifest_frozen",
            "control_plane_pack_stack_control_plane_pack_complete": bool(
                terminal_summary.get("control_plane_pack_stack_control_plane_pack_complete", True)
            ),
            "keep_mass_origin_branch_blocked": keep_blocked,
            "blocked_state_detail": _extract_blocked_state_detail(terminal_payload),
            "latent_reopen_routes_exhausted": bool(_extract_latent_reopen_routes_exhausted(terminal_payload) == "true"),
            "continuity_no_change_hold": str(continuity_decision.get("decision", "")).strip() == "no_change_hold",
            "proceed_to_dark_matter_branch": False,
            "next_required_artifacts": next_required_artifacts,
            "next_action": "wait_for_new_public_artifact_then_refresh_control_plane_pack_stack_control_plane_pack_stack_manifest",
        },
        "evidence": {
            "terminal_control_plane_pack_stack_control_plane_pack_stack_audit": {
                "path": _rel(CONTROL_PLANE_PACK_STACK_AUDIT_JSON),
                "phase_step": _extract_phase_step(terminal_payload),
                "artifact_status": str(terminal_row.get("artifact_status", "")),
            },
            "continuity_artifact": {
                "path": _rel(CONTROL_PLANE_PACK_CONTINUITY_JSON),
                "phase_step": _extract_phase_step(continuity_payload),
                "artifact_status": _extract_artifact_status(continuity_payload),
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
