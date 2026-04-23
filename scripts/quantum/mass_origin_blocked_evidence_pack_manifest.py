#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_blocked_evidence_pack_manifest.py

Step 8.7.55.2.21:
Freeze a single public manifest for the blocked evidence pack of the
mass-origin branch, so downstream readers can inspect the complete blocked
state without walking each artifact manually.

Inputs:
  - output/public/quantum/mass_origin_readiness_gate_metrics.json
  - output/public/quantum/mass_origin_curvature_boundary_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_canonicalization_metrics.json
  - output/public/quantum/mass_origin_solver_family_elimination_metrics.json
  - output/public/quantum/mass_origin_shell_curvature_bridge_metrics.json
  - output/public/quantum/mass_origin_solver_spec_gate_metrics.json
  - output/public/quantum/mass_origin_same_sector_vpp_shape_gate_metrics.json
  - output/public/quantum/mass_origin_latent_reopen_route_inventory_metrics.json
  - output/public/quantum/mass_origin_blocked_state_reopen_metrics.json
  - output/public/quantum/mass_origin_blocked_hold_monitor_metrics.json
  - output/public/quantum/mass_origin_blocked_hold_monitor_rows.csv
  - output/public/quantum/mass_origin_blocked_hold_monitor_status_rows.csv

Outputs:
  - output/public/quantum/mass_origin_blocked_evidence_pack_manifest.json
  - output/public/quantum/mass_origin_blocked_evidence_pack_manifest_rows.csv

Assumptions:
  - The blocked evidence pack is complete only if the upstream gate artifacts,
    the blocker-specific artifacts, the blocked-state artifact, and the hold
    monitor artifacts all exist together in public canonical form.
  - This manifest is descriptive only; it does not change the blocked-state
    logic and only freezes the current evidence pack as a single source of
    truth.
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

READINESS_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_readiness_gate_metrics.json"
CURVATURE_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_curvature_boundary_metrics.json"
SHELL_CANON_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_shell_quantization_canonicalization_metrics.json"
ELIMINATION_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_solver_family_elimination_metrics.json"
SHELL_BRIDGE_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_shell_curvature_bridge_metrics.json"
SOLVER_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_solver_spec_gate_metrics.json"
SPECIFIC_GATE_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_same_sector_vpp_shape_gate_metrics.json"
LATENT_INVENTORY_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_latent_reopen_route_inventory_metrics.json"
BLOCKED_STATE_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_state_reopen_metrics.json"
HOLD_MONITOR_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_hold_monitor_metrics.json"
HOLD_MONITOR_WATCH_CSV = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_hold_monitor_rows.csv"
HOLD_MONITOR_STATUS_CSV = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_hold_monitor_status_rows.csv"
OUT_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_evidence_pack_manifest.json"
OUT_CSV = PUBLIC_QUANTUM_DIR / "mass_origin_blocked_evidence_pack_manifest_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.21"

ARTIFACT_SPECS = [
    {
        "artifact_id": "mass_origin_readiness_gate",
        "role": "upstream_gate",
        "path": READINESS_JSON,
        "note": "Entry gate for the mass-origin branch.",
    },
    {
        "artifact_id": "mass_origin_curvature_boundary",
        "role": "upstream_gate",
        "path": CURVATURE_JSON,
        "note": "Curvature and boundary-family uniqueness gate.",
    },
    {
        "artifact_id": "mass_origin_shell_quantization_canonicalization",
        "role": "upstream_gate",
        "path": SHELL_CANON_JSON,
        "note": "Shell-family public canonicalization artifact.",
    },
    {
        "artifact_id": "mass_origin_solver_family_elimination",
        "role": "upstream_gate",
        "path": ELIMINATION_JSON,
        "note": "Family-elimination artifact that leaves shell quantization public.",
    },
    {
        "artifact_id": "mass_origin_shell_curvature_bridge",
        "role": "upstream_gate",
        "path": SHELL_BRIDGE_JSON,
        "note": "Bridge check between shell observables and particle-sector curvature.",
    },
    {
        "artifact_id": "mass_origin_solver_spec_gate",
        "role": "upstream_gate",
        "path": SOLVER_JSON,
        "note": "Solver-spec gate that formalizes reopen requirements.",
    },
    {
        "artifact_id": "mass_origin_same_sector_vpp_shape_gate",
        "role": "blocker_gate",
        "path": SPECIFIC_GATE_JSON,
        "note": "Named-missing-artifact gate for same-sector curvature and single V(|P|) shape.",
    },
    {
        "artifact_id": "mass_origin_latent_reopen_route_inventory",
        "role": "blocker_gate",
        "path": LATENT_INVENTORY_JSON,
        "note": "Repo-wide latent reopen-route inventory.",
    },
    {
        "artifact_id": "mass_origin_blocked_state_reopen",
        "role": "blocked_state",
        "path": BLOCKED_STATE_JSON,
        "note": "Blocked-state artifact for the mass-origin branch.",
    },
    {
        "artifact_id": "mass_origin_blocked_hold_monitor",
        "role": "hold_monitor",
        "path": HOLD_MONITOR_JSON,
        "note": "Operational blocked-hold monitor JSON.",
    },
    {
        "artifact_id": "mass_origin_blocked_hold_monitor_watch_rows",
        "role": "hold_monitor_csv",
        "path": HOLD_MONITOR_WATCH_CSV,
        "linked_json": HOLD_MONITOR_JSON,
        "note": "Operational hash-watch CSV for the blocked-hold monitor.",
    },
    {
        "artifact_id": "mass_origin_blocked_hold_monitor_status_rows",
        "role": "hold_monitor_csv",
        "path": HOLD_MONITOR_STATUS_CSV,
        "linked_json": HOLD_MONITOR_JSON,
        "note": "Row-canonical blocked-hold status snapshot CSV.",
    },
]


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a single public manifest for the mass-origin blocked evidence pack.",
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


# 関数: `_extract_phase_step` の入出力契約と処理意図を定義する。

def _extract_phase_step(payload: Dict[str, Any]) -> str:
    phase = payload.get("phase", {})

    # 条件分岐: `isinstance(phase, dict)` を満たす経路を評価する。
    if isinstance(phase, dict):
        return str(phase.get("step", "unknown"))

    return "unknown"


# 関数: `_extract_decision_status` の入出力契約と処理意図を定義する。

def _extract_decision_status(payload: Dict[str, Any]) -> str:
    decision = payload.get("decision", {})

    # 条件分岐: `not isinstance(decision, dict)` を満たす経路を評価する。
    if not isinstance(decision, dict):
        return "present"

    overall_status = str(decision.get("overall_status", "")).strip()
    monitor_decision = str(decision.get("decision", "")).strip()

    # 条件分岐: `overall_status and monitor_decision` を満たす経路を評価する。
    if overall_status and monitor_decision:
        return f"{overall_status}:{monitor_decision}"

    # 条件分岐: `overall_status` を満たす経路を評価する。

    if overall_status:
        return overall_status

    # 条件分岐: `monitor_decision` を満たす経路を評価する。

    if monitor_decision:
        return monitor_decision

    return "present"


# 関数: `_extract_blocked_state_detail` の入出力契約と処理意図を定義する。

def _extract_blocked_state_detail(payload: Dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    decision = payload.get("decision", {})
    blocked_state = payload.get("blocked_state", {})

    for source in (summary, decision, blocked_state):
        # 条件分岐: `isinstance(source, dict) and source.get("blocked_state_detail")` を満たす経路を評価する。
        if isinstance(source, dict) and source.get("blocked_state_detail"):
            return str(source.get("blocked_state_detail"))

    return ""


# 関数: `_extract_latent_reopen_routes_exhausted` の入出力契約と処理意図を定義する。

def _extract_latent_reopen_routes_exhausted(payload: Dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    decision = payload.get("decision", {})
    blocked_state = payload.get("blocked_state", {})

    for source in (summary, decision, blocked_state):
        # 条件分岐: `isinstance(source, dict) and "latent_reopen_routes_exhausted" in source` を満たす経路を評価する。
        if isinstance(source, dict) and "latent_reopen_routes_exhausted" in source:
            return "true" if bool(source.get("latent_reopen_routes_exhausted")) else "false"

    return ""


# 関数: `_extract_next_required_artifacts` の入出力契約と処理意図を定義する。

def _extract_next_required_artifacts(payload: Dict[str, Any]) -> List[str]:
    decision = payload.get("decision", {})
    blocked_state = payload.get("blocked_state", {})

    for source in (decision, blocked_state):
        # 条件分岐: `isinstance(source, dict) and isinstance(source.get("next_required_artifacts"), list)` を満たす経路を評価する。
        if isinstance(source, dict) and isinstance(source.get("next_required_artifacts"), list):
            return [str(item) for item in source.get("next_required_artifacts", [])]

    return []


# 関数: `_linked_payload` の入出力契約と処理意図を定義する。

def _linked_payload(spec: Dict[str, Any], cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    path = Path(spec["path"])

    # 条件分岐: `path.suffix.lower() == ".json"` を満たす経路を評価する。
    if path.suffix.lower() == ".json":
        return cache[str(path)]

    linked_json = Path(spec["linked_json"])
    return cache[str(linked_json)]


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows() -> List[Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}

    for spec in ARTIFACT_SPECS:
        path = Path(spec["path"])
        _require_path(path)

        # 条件分岐: `path.suffix.lower() == ".json"` を満たす経路を評価する。
        if path.suffix.lower() == ".json":
            cache[str(path)] = _read_json(path)

        linked_json = spec.get("linked_json")

        # 条件分岐: `linked_json` を満たす経路を評価する。
        if linked_json:
            linked_path = Path(linked_json)
            _require_path(linked_path)

            # 条件分岐: `str(linked_path) not in cache` を満たす経路を評価する。
            if str(linked_path) not in cache:
                cache[str(linked_path)] = _read_json(linked_path)

    rows: List[Dict[str, Any]] = []
    for spec in ARTIFACT_SPECS:
        path = Path(spec["path"])
        payload = _linked_payload(spec, cache)
        next_required_artifacts = _extract_next_required_artifacts(payload)
        rows.append(
            {
                "row_id": str(spec["artifact_id"]),
                "status": "pass",
                "metric": f"{spec['role']} artifact present in blocked evidence pack",
                "value": 1.0,
                "artifact_path": _relative_str(path),
                "artifact_sha256": _sha256_file(path),
                "artifact_phase_step": _extract_phase_step(payload),
                "artifact_status": _extract_decision_status(payload),
                "blocked_state_detail": _extract_blocked_state_detail(payload),
                "latent_reopen_routes_exhausted": _extract_latent_reopen_routes_exhausted(payload),
                "next_required_artifacts_count": len(next_required_artifacts),
                "note": str(spec["note"]),
            }
        )

    return rows


# 関数: `_build_manifest_signature` の入出力契約と処理意図を定義する。

def _build_manifest_signature(rows: List[Dict[str, Any]]) -> str:
    serializable = [
        {
            "row_id": row["row_id"],
            "artifact_path": row["artifact_path"],
            "artifact_sha256": row["artifact_sha256"],
            "artifact_phase_step": row["artifact_phase_step"],
            "artifact_status": row["artifact_status"],
        }
        for row in rows
    ]
    packed = json.dumps(serializable, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(packed).hexdigest().lower()


# 関数: `_find_row` の入出力契約と処理意図を定義する。

def _find_row(rows: List[Dict[str, Any]], row_id: str) -> Dict[str, Any]:
    for row in rows:
        # 条件分岐: `str(row.get("row_id")) == row_id` を満たす経路を評価する。
        if str(row.get("row_id")) == row_id:
            return row

    raise KeyError(f"missing row_id: {row_id}")


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    rows = _build_rows()
    blocked_state_row = _find_row(rows, "mass_origin_blocked_state_reopen")
    hold_monitor_row = _find_row(rows, "mass_origin_blocked_hold_monitor")
    specific_gate_row = _find_row(rows, "mass_origin_same_sector_vpp_shape_gate")
    latent_inventory_row = _find_row(rows, "mass_origin_latent_reopen_route_inventory")
    manifest_signature = _build_manifest_signature(rows)
    next_required_artifacts = [
        "positive_particle_sector_chi_p_to_vpp_public_artifact",
        "single_public_vpp_shape",
        "solver_ready_row_promoted_to_pass",
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "mass-origin blocked evidence pack manifest",
        },
        "inputs": {
            "public_quantum_dir": _relative_str(PUBLIC_QUANTUM_DIR),
            "artifact_count": len(rows),
        },
        "intent": "Freeze the current blocked evidence pack of the mass-origin branch as a single public canonical manifest, including upstream gates, blocker-specific artifacts, the blocked-state artifact, and the hold-monitor artifacts.",
        "rows": rows,
        "summary": {
            "artifact_count": len(rows),
            "json_artifact_count": sum(1 for spec in ARTIFACT_SPECS if Path(spec["path"]).suffix.lower() == ".json"),
            "csv_artifact_count": sum(1 for spec in ARTIFACT_SPECS if Path(spec["path"]).suffix.lower() == ".csv"),
            "manifest_signature_sha256": manifest_signature,
            "blocked_state_detail": str(blocked_state_row.get("blocked_state_detail", "")),
            "latent_reopen_routes_exhausted": str(blocked_state_row.get("latent_reopen_routes_exhausted", "")) == "true",
            "hold_monitor_status": str(hold_monitor_row.get("artifact_status", "")),
            "specific_gate_status": str(specific_gate_row.get("artifact_status", "")),
            "latent_inventory_status": str(latent_inventory_row.get("artifact_status", "")),
            "next_required_artifacts": next_required_artifacts,
        },
        "decision": {
            "overall_status": "blocked_evidence_pack_manifest_frozen",
            "evidence_pack_complete": True,
            "keep_mass_origin_branch_blocked": True,
            "blocked_state_detail": str(blocked_state_row.get("blocked_state_detail", "")),
            "latent_reopen_routes_exhausted": str(blocked_state_row.get("latent_reopen_routes_exhausted", "")) == "true",
            "hold_monitor_no_change_hold": str(hold_monitor_row.get("artifact_status", "")).endswith("no_change_hold"),
            "proceed_to_dark_matter_branch": False,
            "next_required_artifacts": next_required_artifacts,
        },
        "evidence": {
            "blocked_state_artifact": {
                "path": str(blocked_state_row.get("artifact_path", "")),
                "phase_step": str(blocked_state_row.get("artifact_phase_step", "")),
                "artifact_status": str(blocked_state_row.get("artifact_status", "")),
            },
            "hold_monitor_artifact": {
                "path": str(hold_monitor_row.get("artifact_path", "")),
                "phase_step": str(hold_monitor_row.get("artifact_phase_step", "")),
                "artifact_status": str(hold_monitor_row.get("artifact_status", "")),
            },
            "specific_gate_artifact": {
                "path": str(specific_gate_row.get("artifact_path", "")),
                "phase_step": str(specific_gate_row.get("artifact_phase_step", "")),
                "artifact_status": str(specific_gate_row.get("artifact_status", "")),
            },
            "latent_inventory_artifact": {
                "path": str(latent_inventory_row.get("artifact_path", "")),
                "phase_step": str(latent_inventory_row.get("artifact_phase_step", "")),
                "artifact_status": str(latent_inventory_row.get("artifact_status", "")),
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
    PUBLIC_QUANTUM_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["rows"])
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
