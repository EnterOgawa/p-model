#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_local_shape_jet.py

Step 8.7.55.2.245:
Freeze the anchor-local shape jet

  {V'(rho_*), rho_*^2 V''(rho_*), R_3_target}

after the curvature bridge and R_3 target-source audit.

This step records that the local jet is already closed up to the missing
R_3 target: the stationary anchor condition and the anchor-local curvature
identity are fixed without a new free parameter, while the global shape
selection remains blocked until a public canonical R_3 target is promoted.

Inputs:
  - output/public/quantum/mass_origin_anchor_local_curvature_bridge_metrics.json
  - output/public/quantum/mass_origin_anchor_local_r3_registry_metrics.json
  - output/public/quantum/mass_origin_r3_target_source_audit_metrics.json

Outputs:
  - output/public/quantum/mass_origin_anchor_local_shape_jet_metrics.json
  - output/public/quantum/mass_origin_anchor_local_shape_jet_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

CURVATURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_curvature_bridge_metrics.json"
R3_REGISTRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_r3_registry_metrics.json"
R3_ROUTE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_r3_target_source_audit_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_jet_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_jet_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.245"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the anchor-local shape jet for the mass-origin route.",
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
    vp_anchor_zero: bool,
    rho2_vpp_anchor_value: str | None,
    r3_target_available: bool,
    preferred_r3_route_or_none: str | None,
    remaining_r3_missing_datum_or_none: str | None,
    shape_jet_without_new_free_parameters: bool,
    global_vpp_shape_fixed: bool,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "anchor_local_shape_jet_freeze_complete",
            "status": "pass",
            "metric": "anchor-local shape jet freeze complete",
            "value": 1.0,
            "note": "This step freezes how much of the anchor-local V(|P|) jet is already fixed before a public canonical R_3 target exists.",
        },
        {
            "row_id": "anchor_local_shape_jet_vp_anchor_zero",
            "status": "pass" if vp_anchor_zero else "reject",
            "metric": "anchor stationary condition V'(rho_*) = 0 fixed",
            "value": 1.0 if vp_anchor_zero else 0.0,
            "note": (
                "The anchor is frozen as a stationary background point, so V'(rho_*) = 0 is part of the shape jet."
                if vp_anchor_zero
                else "The anchor-stationary condition is not yet frozen, so the local jet cannot be closed."
            ),
        },
        {
            "row_id": "anchor_local_shape_jet_rho2_vpp_anchor_value",
            "status": "pass" if rho2_vpp_anchor_value else "reject",
            "metric": "anchor-local rho_*^2 V''(rho_*) value fixed",
            "value": 1.0 if rho2_vpp_anchor_value else 0.0,
            "note": (
                f"rho_*^2 V''(rho_*) is frozen symbolically as {rho2_vpp_anchor_value}."
                if rho2_vpp_anchor_value
                else "The anchor-local curvature value is not yet frozen."
            ),
        },
        {
            "row_id": "anchor_local_shape_jet_r3_target_available",
            "status": "pass" if r3_target_available else "watch",
            "metric": "public canonical R_3 target available",
            "value": 1.0 if r3_target_available else 0.0,
            "note": (
                "The public canonical pack already fixes R_3_target, so the anchor-local jet can close fully."
                if r3_target_available
                else "The local jet remains open only at R_3_target; this is the sole unresolved anchor-local jet component."
            ),
        },
        {
            "row_id": "anchor_local_shape_jet_preferred_r3_route",
            "status": "watch" if preferred_r3_route_or_none else "reject",
            "metric": "preferred route carried forward for missing R_3 target",
            "value": 1.0 if preferred_r3_route_or_none else 0.0,
            "note": (
                f"The preferred next-closing route is {preferred_r3_route_or_none}."
                if preferred_r3_route_or_none
                else "No preferred route is currently available for the missing R_3 target."
            ),
        },
        {
            "row_id": "anchor_local_shape_jet_remaining_missing_datum",
            "status": "missing" if remaining_r3_missing_datum_or_none else "pass",
            "metric": "remaining missing datum for R_3 target",
            "value": 0.0 if remaining_r3_missing_datum_or_none else 1.0,
            "note": (
                f"The remaining datum is {remaining_r3_missing_datum_or_none}."
                if remaining_r3_missing_datum_or_none
                else "No remaining datum is needed for the R_3 target."
            ),
        },
        {
            "row_id": "anchor_local_shape_jet_without_new_free_parameters",
            "status": "pass" if shape_jet_without_new_free_parameters else "reject",
            "metric": "anchor-local jet closed without new free parameters",
            "value": 1.0 if shape_jet_without_new_free_parameters else 0.0,
            "note": (
                "V'(rho_*) and rho_*^2 V''(rho_*) are already fixed without introducing a new free parameter."
                if shape_jet_without_new_free_parameters
                else "The current anchor-local jet still depends on a new free parameter."
            ),
        },
        {
            "row_id": "anchor_local_shape_jet_global_shape_fixed",
            "status": "pass" if global_vpp_shape_fixed else "watch",
            "metric": "global V(|P|) shape fixed by the anchor-local jet",
            "value": 1.0 if global_vpp_shape_fixed else 0.0,
            "note": (
                "The anchor-local jet is sufficient to select a unique global same-sector shape."
                if global_vpp_shape_fixed
                else "Global V(|P|) remains unfixed because the public canonical pack still lacks R_3_target."
            ),
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (CURVATURE_JSON, R3_REGISTRY_JSON, R3_ROUTE_JSON):
        _require_path(path)

    curvature = _read_json(CURVATURE_JSON)
    r3_registry = _read_json(R3_REGISTRY_JSON)
    r3_route = _read_json(R3_ROUTE_JSON)

    curvature_summary = curvature.get("summary", {})
    curvature_decision = curvature.get("decision", {})
    r3_registry_summary = r3_registry.get("summary", {})
    r3_route_summary = r3_route.get("summary", {})

    vp_anchor_zero = bool(r3_registry_summary.get("anchor_stationary_condition_vp_zero", False))
    rho2_vpp_anchor_value = None

    # 条件分岐: `curvature_summary.get("vpp_oscillation_path_ready", False)` を満たす経路を評価する。
    if curvature_summary.get("vpp_oscillation_path_ready", False):
        rho2_vpp_anchor_value = "M_chi^2 omega_*^2 = rho_*^2 g_P Z_P / chi_P"

    r3_target_available = bool(r3_route_summary.get("r3_target_available", False))
    r3_target_value_or_none = r3_route_summary.get("r3_target_value_or_none")
    preferred_r3_route_or_none = r3_route_summary.get("preferred_r3_route_or_none")
    remaining_r3_missing_datum_or_none = r3_route_summary.get("remaining_r3_missing_datum_or_none")
    shape_jet_without_new_free_parameters = bool(curvature_summary.get("vpp_closed_without_new_free_parameters", False)) and vp_anchor_zero
    global_vpp_shape_fixed = bool(vp_anchor_zero and rho2_vpp_anchor_value and r3_target_available)
    local_shape_jet_closed_up_to_r3_target = bool(vp_anchor_zero and rho2_vpp_anchor_value and not r3_target_available)

    rows = _build_rows(
        vp_anchor_zero=vp_anchor_zero,
        rho2_vpp_anchor_value=rho2_vpp_anchor_value,
        r3_target_available=r3_target_available,
        preferred_r3_route_or_none=preferred_r3_route_or_none,
        remaining_r3_missing_datum_or_none=remaining_r3_missing_datum_or_none,
        shape_jet_without_new_free_parameters=shape_jet_without_new_free_parameters,
        global_vpp_shape_fixed=global_vpp_shape_fixed,
    )

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
            "name": "anchor-local shape jet freeze",
        },
        "inputs": {
            "mass_origin_anchor_local_curvature_bridge_json": _relative_str(CURVATURE_JSON),
            "mass_origin_anchor_local_r3_registry_json": _relative_str(R3_REGISTRY_JSON),
            "mass_origin_r3_target_source_audit_json": _relative_str(R3_ROUTE_JSON),
        },
        "intent": "Freeze the anchor-local V(|P|) jet {V'_*, rho_*^2 V''_*, R_3_target} and record that only the public canonical R_3 target is still missing.",
        "formulas": {
            "vp_anchor_condition": "V'(rho_*) = 0",
            "rho2_vpp_anchor_identity": "rho_*^2 V''(rho_*) = M_chi^2 omega_*^2 = rho_*^2 g_P Z_P / chi_P",
            "r3_target_slot": "R_3_target = rho_* V'''(rho_*) / V''(rho_*)",
            "global_shape_rule": "global V(|P|) remains unfixed until a public canonical R_3_target chooses between the surviving same-sector candidates",
        },
        "rows": rows,
        "summary": {
            "vp_anchor_zero": vp_anchor_zero,
            "rho2_vpp_anchor_value": rho2_vpp_anchor_value,
            "r3_target_available": r3_target_available,
            "r3_target_value_or_none": r3_target_value_or_none,
            "preferred_r3_route_or_none": preferred_r3_route_or_none,
            "remaining_r3_missing_datum_or_none": remaining_r3_missing_datum_or_none,
            "shape_jet_without_new_free_parameters": shape_jet_without_new_free_parameters,
            "local_shape_jet_closed_up_to_r3_target": local_shape_jet_closed_up_to_r3_target,
            "global_vpp_shape_fixed": global_vpp_shape_fixed,
        },
        "decision": {
            "overall_status": "anchor_local_shape_jet_frozen_r3_target_pending",
            "keep_mass_origin_branch_blocked": True,
            "vp_anchor_zero": vp_anchor_zero,
            "rho2_vpp_anchor_value": rho2_vpp_anchor_value,
            "r3_target_available": r3_target_available,
            "r3_target_value_or_none": r3_target_value_or_none,
            "preferred_r3_route_or_none": preferred_r3_route_or_none,
            "remaining_r3_missing_datum_or_none": remaining_r3_missing_datum_or_none,
            "shape_jet_without_new_free_parameters": shape_jet_without_new_free_parameters,
            "local_shape_jet_closed_up_to_r3_target": local_shape_jet_closed_up_to_r3_target,
            "global_vpp_shape_fixed": global_vpp_shape_fixed,
            "blocked_state_detail": str(curvature_decision.get("blocked_state_detail", "specific_missing_artifacts_fixed")),
            "next_required_artifacts": next_required_artifacts,
        },
        "evidence": {
            "curvature_summary": curvature_summary,
            "curvature_decision": curvature_decision,
            "r3_registry_summary": r3_registry_summary,
            "r3_route_summary": r3_route_summary,
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
