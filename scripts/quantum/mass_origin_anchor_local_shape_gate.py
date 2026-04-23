#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_local_shape_gate.py

Step 8.7.55.2.246:
Apply the candidate-shape gate after the anchor-local shape jet freeze.

This step asks whether the anchor-local jet is already sufficient to select a
single same-sector V(|P|) family and reopen the mass-spectrum handoff into the
eigenvalue pilot (`8.7.55.2.83-.84`).

Inputs:
  - output/public/quantum/mass_origin_anchor_local_r3_registry_metrics.json
  - output/public/quantum/mass_origin_anchor_local_shape_jet_metrics.json
  - output/public/quantum/mass_origin_same_sector_vpp_shape_gate_metrics.json
  - output/public/quantum/mass_origin_readiness_gate_metrics.json

Outputs:
  - output/public/quantum/mass_origin_anchor_local_shape_gate_metrics.json
  - output/public/quantum/mass_origin_anchor_local_shape_gate_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

R3_REGISTRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_r3_registry_metrics.json"
SHAPE_JET_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_jet_metrics.json"
SAME_SECTOR_GATE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_vpp_shape_gate_metrics.json"
READINESS_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_readiness_gate_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.246"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the anchor-local candidate-shape gate for the mass-origin route.",
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


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows(
    *,
    single_public_boundary_family_fixed: bool,
    candidate_family_ids: List[str],
    candidate_family_r3_values: Dict[str, Any],
    r3_target_available: bool,
    single_public_vpp_shape_available: bool,
    positive_artifact_available: bool,
    eigenvalue_handoff_ready: bool,
    handoff: bool,
    nonclosure_reason: str | None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = [
        {
            "row_id": "anchor_local_shape_gate_complete",
            "status": "pass",
            "metric": "anchor-local candidate-shape gate complete",
            "value": 1.0,
            "note": "This step evaluates whether the anchor-local jet can already choose a unique same-sector V(|P|) class and reopen the discrete-spectrum handoff.",
        },
        {
            "row_id": "anchor_local_shape_gate_boundary_family_fixed",
            "status": "pass" if single_public_boundary_family_fixed else "reject",
            "metric": "single public boundary family fixed",
            "value": 1.0 if single_public_boundary_family_fixed else 0.0,
            "note": (
                "Shell quantization remains the sole public boundary family during the anchor-local gate."
                if single_public_boundary_family_fixed
                else "The anchor-local gate cannot close because the boundary family is no longer unique."
            ),
        },
        {
            "row_id": "anchor_local_shape_gate_r3_target_available",
            "status": "pass" if r3_target_available else "reject",
            "metric": "R_3 target available for candidate discrimination",
            "value": 1.0 if r3_target_available else 0.0,
            "note": (
                "The public canonical pack now carries R_3_target, so the candidate registry can collapse to one family."
                if r3_target_available
                else "The candidate registry remains two-valued because the public canonical pack still lacks R_3_target."
            ),
        },
    ]

    for family in candidate_family_ids:
        rows.append(
            {
                "row_id": f"anchor_local_shape_gate_candidate_{family}",
                "status": "watch" if not r3_target_available else "pass",
                "metric": f"{family} remains eligible at the anchor-local gate",
                "value": 1.0,
                "note": (
                    f"{family} stays live with registered R_3 = {candidate_family_r3_values.get(family)} until a public canonical target value is supplied."
                    if not r3_target_available
                    else f"{family} has been checked against the public canonical R_3 target."
                ),
            }
        )

    rows.extend(
        [
            {
                "row_id": "anchor_local_shape_gate_single_public_vpp_shape",
                "status": "pass" if single_public_vpp_shape_available else "reject",
                "metric": "single public V(|P|) shape available after anchor-local gate",
                "value": 1.0 if single_public_vpp_shape_available else 0.0,
                "note": (
                    "The anchor-local jet is sufficient to promote a single public V(|P|) shape."
                    if single_public_vpp_shape_available
                    else f"The anchor-local gate is non-closing: {nonclosure_reason}."
                ),
            },
            {
                "row_id": "anchor_local_shape_gate_positive_same_sector_artifact",
                "status": "pass" if positive_artifact_available else "reject",
                "metric": "positive particle-sector chi_P -> V''(|P|_*) artifact available",
                "value": 1.0 if positive_artifact_available else 0.0,
                "note": (
                    "The positive same-sector public artifact is already promotable from the anchor-local gate."
                    if positive_artifact_available
                    else "The positive same-sector public artifact cannot promote before the shape gate closes."
                ),
            },
            {
                "row_id": "anchor_local_shape_gate_eigenvalue_handoff_ready",
                "status": "pass" if eigenvalue_handoff_ready else "reject",
                "metric": "eigenvalue handoff ready",
                "value": 1.0 if eigenvalue_handoff_ready else 0.0,
                "note": (
                    "The branch is ready to hand off into the discrete-spectrum pilot."
                    if eigenvalue_handoff_ready
                    else "The branch is not ready for the discrete-spectrum pilot because the single-shape gate is still open."
                ),
            },
            {
                "row_id": "hand_off_to_8_7_55_2_83",
                "status": "pass" if handoff else "reject",
                "metric": "handoff to 8.7.55.2.83-.84 allowed after anchor-local gate",
                "value": 1.0 if handoff else 0.0,
                "note": (
                    "Handoff to the eigenvalue pilot is now allowed."
                    if handoff
                    else "Handoff remains blocked because the anchor-local gate did not reduce the candidate set to a single public shape."
                ),
            },
        ]
    )
    return rows


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (R3_REGISTRY_JSON, SHAPE_JET_JSON, SAME_SECTOR_GATE_JSON, READINESS_JSON):
        _require_path(path)

    r3_registry = _read_json(R3_REGISTRY_JSON)
    shape_jet = _read_json(SHAPE_JET_JSON)
    same_sector_gate = _read_json(SAME_SECTOR_GATE_JSON)
    readiness = _read_json(READINESS_JSON)

    r3_registry_summary = r3_registry.get("summary", {})
    shape_jet_summary = shape_jet.get("summary", {})
    shape_jet_decision = shape_jet.get("decision", {})
    same_sector_gate_summary = same_sector_gate.get("summary", {})
    readiness_summary = readiness.get("summary", {})

    candidate_family_ids = [str(item) for item in r3_registry_summary.get("candidate_family_ids", [])]
    candidate_family_r3_values = dict(r3_registry_summary.get("candidate_family_r3_values", {}))
    r3_target_available = bool(shape_jet_summary.get("r3_target_available", False))
    single_public_boundary_family_fixed = bool(same_sector_gate_summary.get("single_public_boundary_family_fixed", False))
    single_public_vpp_shape_available = bool(r3_target_available and len(candidate_family_ids) == 1)
    positive_artifact_available = False
    eigenvalue_handoff_ready = bool(
        single_public_boundary_family_fixed
        and single_public_vpp_shape_available
        and positive_artifact_available
        and readiness_summary.get("no_free_parameter_mass_solver_ready", False)
    )
    handoff = eigenvalue_handoff_ready
    nonclosure_reason = None if single_public_vpp_shape_available else "r3_target_unavailable"

    rows = _build_rows(
        single_public_boundary_family_fixed=single_public_boundary_family_fixed,
        candidate_family_ids=candidate_family_ids,
        candidate_family_r3_values=candidate_family_r3_values,
        r3_target_available=r3_target_available,
        single_public_vpp_shape_available=single_public_vpp_shape_available,
        positive_artifact_available=positive_artifact_available,
        eigenvalue_handoff_ready=eigenvalue_handoff_ready,
        handoff=handoff,
        nonclosure_reason=nonclosure_reason,
    )

    surviving_candidate_family_ids = [str(item) for item in candidate_family_ids]
    next_required_artifacts = [
        "anchor_normalized_g3w_public_value",
        "r3_target",
        "single_public_vpp_shape",
        "positive_particle_sector_chi_p_to_vpp_public_artifact",
        "solver_ready_row_promoted_to_pass",
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "candidate-shape gate and eigenvalue handoff",
        },
        "inputs": {
            "mass_origin_anchor_local_r3_registry_json": _relative_str(R3_REGISTRY_JSON),
            "mass_origin_anchor_local_shape_jet_json": _relative_str(SHAPE_JET_JSON),
            "mass_origin_same_sector_vpp_shape_gate_json": _relative_str(SAME_SECTOR_GATE_JSON),
            "mass_origin_readiness_gate_json": _relative_str(READINESS_JSON),
        },
        "intent": "Freeze whether the anchor-local jet already selects a single same-sector V(|P|) family and reopens the eigenvalue handoff.",
        "formulas": {
            "candidate_gate_rule": "single_public_vpp_shape_available iff the anchor-local jet fixes R_3_target and collapses the surviving candidate registry to one family",
            "handoff_rule": "hand_off_to_8_7_55_2_83 iff single public boundary family + single public V(|P|) shape + positive same-sector artifact + solver-ready readiness all close together",
        },
        "rows": rows,
        "summary": {
            "single_public_boundary_family_fixed": single_public_boundary_family_fixed,
            "surviving_candidate_family_ids": surviving_candidate_family_ids,
            "candidate_family_r3_values": candidate_family_r3_values,
            "single_public_vpp_shape_available": single_public_vpp_shape_available,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": positive_artifact_available,
            "shape_gate_nonclosure_reason_or_none": nonclosure_reason,
            "eigenvalue_handoff_ready": eigenvalue_handoff_ready,
            "hand_off_to_8_7_55_2_83": handoff,
        },
        "decision": {
            "overall_status": "anchor_local_shape_gate_frozen_r3_target_pending",
            "keep_mass_origin_branch_blocked": True,
            "single_public_boundary_family_fixed": single_public_boundary_family_fixed,
            "surviving_candidate_family_ids": surviving_candidate_family_ids,
            "single_public_vpp_shape_available": single_public_vpp_shape_available,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": positive_artifact_available,
            "shape_gate_nonclosure_reason_or_none": nonclosure_reason,
            "eigenvalue_handoff_ready": eigenvalue_handoff_ready,
            "hand_off_to_8_7_55_2_83": handoff,
            "blocked_state_detail": str(shape_jet_decision.get("blocked_state_detail", "specific_missing_artifacts_fixed")),
            "next_required_artifacts": next_required_artifacts,
        },
        "evidence": {
            "r3_registry_summary": r3_registry_summary,
            "shape_jet_summary": shape_jet_summary,
            "same_sector_gate_summary": same_sector_gate_summary,
            "readiness_summary": readiness_summary,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_id", "status", "metric", "value", "note"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    args = _parse_args()
    payload = _build_payload(args.step_tag)
    _write_json(OUT_JSON, payload)
    _write_csv(OUT_CSV, payload["rows"])
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
