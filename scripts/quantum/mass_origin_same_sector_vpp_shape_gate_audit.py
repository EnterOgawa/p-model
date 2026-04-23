#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_same_sector_vpp_shape_gate_audit.py

Step 8.7.55.2.15:
With shell quantization already frozen as the sole surviving public solver
family, formalize the minimum public-canonical specification for the two
remaining blockers:

  1. a positive particle-sector chi_P -> V''(|P|_*) artifact
  2. a single public V(|P|) shape

This step does not create either missing artifact. It freezes what the missing
artifacts must contain, whether any current public anchors exist, and whether a
concrete reopen route is already visible. If not, the branch remains blocked in
the more specific state "specific missing artifacts fixed" rather than falling
back to a generic blocked-hold description.

Inputs:
  - output/public/quantum/mass_origin_readiness_gate_metrics.json
  - output/public/quantum/mass_origin_curvature_boundary_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_canonicalization_metrics.json
  - output/public/quantum/mass_origin_solver_family_elimination_metrics.json
  - output/public/quantum/mass_origin_shell_curvature_bridge_metrics.json
  - output/public/quantum/mass_origin_blocked_state_reopen_metrics.json
  - output/public/quantum/action_principle_el_derivation_audit.json
  - output/public/quantum/nuclear_effective_potential_canonical_metrics.json
  - output/public/quantum/nuclear_binding_energy_frequency_mapping_interface_metrics.json
  - output/public/quantum/mass_origin_latent_reopen_route_inventory_metrics.json

Outputs:
  - output/public/quantum/mass_origin_same_sector_vpp_shape_gate_metrics.json
  - output/public/quantum/mass_origin_same_sector_vpp_shape_gate_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

READINESS_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_readiness_gate_metrics.json"
CURVATURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_curvature_boundary_metrics.json"
SHELL_CANON_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_canonicalization_metrics.json"
ELIMINATION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_solver_family_elimination_metrics.json"
SHELL_BRIDGE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_curvature_bridge_metrics.json"
REOPEN_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_blocked_state_reopen_metrics.json"
ACTION_JSON = ROOT / "output" / "public" / "quantum" / "action_principle_el_derivation_audit.json"
EFFECTIVE_POTENTIAL_JSON = ROOT / "output" / "public" / "quantum" / "nuclear_effective_potential_canonical_metrics.json"
INTERFACE_JSON = ROOT / "output" / "public" / "quantum" / "nuclear_binding_energy_frequency_mapping_interface_metrics.json"
LATENT_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_latent_reopen_route_inventory_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_vpp_shape_gate_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_vpp_shape_gate_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.15"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the same-sector / V(|P|) blocker gate for the mass-origin branch.",
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


# 関数: `_find_row_by_id` の入出力契約と処理意図を定義する。

def _find_row_by_id(rows: List[Dict[str, Any]], row_id: str) -> Dict[str, Any]:
    for row in rows:
        # 条件分岐: `str(row.get("row_id")) == row_id` を満たす経路を評価する。
        if str(row.get("row_id")) == row_id:
            return row

    raise KeyError(f"missing row_id: {row_id}")


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (
        READINESS_JSON,
        CURVATURE_JSON,
        SHELL_CANON_JSON,
        ELIMINATION_JSON,
        SHELL_BRIDGE_JSON,
        REOPEN_JSON,
        ACTION_JSON,
        EFFECTIVE_POTENTIAL_JSON,
        INTERFACE_JSON,
        LATENT_INVENTORY_JSON,
    ):
        _require_path(path)

    readiness = _read_json(READINESS_JSON)
    curvature = _read_json(CURVATURE_JSON)
    shell_canon = _read_json(SHELL_CANON_JSON)
    elimination = _read_json(ELIMINATION_JSON)
    shell_bridge = _read_json(SHELL_BRIDGE_JSON)
    reopen = _read_json(REOPEN_JSON)
    action = _read_json(ACTION_JSON)
    effective_potential = _read_json(EFFECTIVE_POTENTIAL_JSON)
    interface = _read_json(INTERFACE_JSON)
    latent_inventory = _read_json(LATENT_INVENTORY_JSON)

    readiness_rows = readiness.get("rows", [])
    curvature_rows = curvature.get("rows", [])
    shell_rows = shell_canon.get("rows", [])
    elimination_rows = elimination.get("rows", [])
    bridge_rows = shell_bridge.get("rows", [])
    latent_rows = latent_inventory.get("rows", [])

    # 条件分岐: `not isinstance(readiness_rows, list)` を満たす経路を評価する。
    if not isinstance(readiness_rows, list):
        raise SystemExit(f"[fail] invalid rows in {READINESS_JSON}")

    # 条件分岐: `not isinstance(curvature_rows, list)` を満たす経路を評価する。

    if not isinstance(curvature_rows, list):
        raise SystemExit(f"[fail] invalid rows in {CURVATURE_JSON}")

    # 条件分岐: `not isinstance(shell_rows, list)` を満たす経路を評価する。

    if not isinstance(shell_rows, list):
        raise SystemExit(f"[fail] invalid rows in {SHELL_CANON_JSON}")

    # 条件分岐: `not isinstance(elimination_rows, list)` を満たす経路を評価する。

    if not isinstance(elimination_rows, list):
        raise SystemExit(f"[fail] invalid rows in {ELIMINATION_JSON}")

    # 条件分岐: `not isinstance(bridge_rows, list)` を満たす経路を評価する。

    if not isinstance(bridge_rows, list):
        raise SystemExit(f"[fail] invalid rows in {SHELL_BRIDGE_JSON}")

    if not isinstance(latent_rows, list):
        raise SystemExit(f"[fail] invalid rows in {LATENT_INVENTORY_JSON}")

    readiness_same_sector = _find_row_by_id(readiness_rows, "same_sector_chi_p_to_vpp_mapping")
    curvature_single_vpp = _find_row_by_id(curvature_rows, "single_vpp_shape_unique")
    shell_family_row = _find_row_by_id(shell_rows, "shell_quantization_family_public_candidate")
    shell_kappa_row = _find_row_by_id(shell_rows, "shell_quantization_fit_kappa")
    shell_kz_row = _find_row_by_id(shell_rows, "shell_quantization_fit_kz_over_kn")
    elimination_single_family = _find_row_by_id(elimination_rows, "single_public_boundary_family_remaining")
    shell_bridge_row = _find_row_by_id(bridge_rows, "shell_to_curvature_bridge_ready")
    shell_coefficients_row = _find_row_by_id(bridge_rows, "shell_quantization_coefficients_not_vpp")
    abstract_action_row = _find_row_by_id(bridge_rows, "abstract_action_not_enough_for_vpp_coefficients")
    latent_same_sector_row = _find_row_by_id(latent_rows, "latent_positive_same_sector_public_rows")
    latent_vpp_row = _find_row_by_id(latent_rows, "effective_potential_nonphenomenological_public_count")
    latent_inventory_exhausted_row = _find_row_by_id(latent_rows, "latent_reopen_route_inventory_exhausted")

    shell_family_public = str(shell_family_row.get("status", "")) == "candidate_public"
    single_boundary_fixed = str(elimination_single_family.get("status", "")) == "pass"
    action_fixed = str(action.get("decision", {}).get("route_a_el_derivation_gate", "")) == "pass"
    same_sector_available = float(readiness_same_sector.get("value", 0.0)) > 0.0
    single_vpp_available = float(curvature_single_vpp.get("value", 0.0)) > 0.0
    shell_bridge_available = float(shell_bridge_row.get("value", 0.0)) > 0.0
    latent_same_sector_count = float(latent_same_sector_row.get("value", 0.0))
    latent_vpp_count = float(latent_vpp_row.get("value", 0.0))
    latent_inventory_exhausted = str(latent_inventory_exhausted_row.get("status", "")) == "pass"

    effective_positioning = effective_potential.get("model", {}).get("positioning", [])
    effective_positioning_text = " ".join(str(item) for item in effective_positioning)
    effective_is_phenomenological = "Phenomenological" in effective_positioning_text or "not a first-principles" in effective_positioning_text
    interface_rows = interface.get("rows", [])
    interface_omega0_values = [
        float(row.get("omega0_eff_per_s", 0.0))
        for row in interface_rows
        if isinstance(row, dict) and float(row.get("omega0_eff_per_s", 0.0)) > 0.0
    ]
    interface_spread = 0.0

    # 条件分岐: `interface_omega0_values` を満たす経路を評価する。
    if interface_omega0_values:
        omega0_mid = sorted(interface_omega0_values)[len(interface_omega0_values) // 2]

        # 条件分岐: `omega0_mid > 0.0` を満たす経路を評価する。
        if omega0_mid > 0.0:
            interface_spread = (max(interface_omega0_values) - min(interface_omega0_values)) / omega0_mid

    lagrangian = str(action.get("equations", {}).get("lagrangian_density", ""))

    same_sector_anchor_available = shell_family_public and single_boundary_fixed and action_fixed
    same_sector_concrete_route = same_sector_available or shell_bridge_available or latent_same_sector_count > 0.0
    single_vpp_concrete_route = (single_vpp_available and not effective_is_phenomenological) or latent_vpp_count > 0.0

    rows = [
        {
            "row_id": "single_public_boundary_family_already_fixed",
            "status": "pass" if single_boundary_fixed else "reject",
            "metric": "single public boundary family already fixed",
            "value": 1.0 if single_boundary_fixed else 0.0,
            "note": str(elimination_single_family.get("note", "")),
        },
        {
            "row_id": "same_sector_public_artifact_still_missing",
            "status": str(readiness_same_sector.get("status", "missing")),
            "metric": "positive particle-sector chi_P -> V''(|P|_*) artifact available",
            "value": float(readiness_same_sector.get("value", 0.0)),
            "note": str(readiness_same_sector.get("note", "")),
        },
        {
            "row_id": "same_sector_public_artifact_minimal_spec_frozen",
            "status": "pass",
            "metric": "minimum public spec for chi_P -> V''(|P|_*) artifact",
            "value": 5.0,
            "note": (
                "Required elements are now frozen: same particle sector only; explicit mapping equation from chi_P or an equivalent shell-family observable to "
                "V''(|P|_*); units/sign/reference point for |P|_*; public numerical rows that tie the mapping to the surviving shell family; and a no-new-free-parameter note."
            ),
        },
        {
            "row_id": "same_sector_artifact_must_reuse_shell_family_only",
            "status": "pass",
            "metric": "same-sector artifact must reuse surviving shell family",
            "value": 1.0,
            "note": (
                "The artifact must be anchored to the surviving public shell family and may not promote the cross-sector structural parity proxy or the interface-only "
                "omega0_eff spread into a particle-sector curvature substitute."
            ),
        },
        {
            "row_id": "same_sector_artifact_anchor_pack_available",
            "status": "pass" if same_sector_anchor_available else "reject",
            "metric": "public anchors exist for a future same-sector artifact",
            "value": 1.0 if same_sector_anchor_available else 0.0,
            "note": (
                f"Current anchors are shell family status={shell_family_row.get('status', '')}, kappa={shell_kappa_row.get('value')}, "
                f"kZ_over_kN={shell_kz_row.get('value')}, and action `{lagrangian}`."
            ),
        },
        {
            "row_id": "same_sector_artifact_concrete_reopen_route",
            "status": "pass" if same_sector_concrete_route else "reject",
            "metric": "concrete reopen route for chi_P -> V''(|P|_*) artifact",
            "value": 1.0 if same_sector_concrete_route else 0.0,
            "note": (
                "A positive route would require an existing public row that already maps shell-family observables to particle-sector curvature. "
                f"Current bridge row is {shell_bridge_row.get('status', '')}, repo-wide latent candidate count is {latent_same_sector_count}, "
                f"and the shell-coefficient row still says: {shell_coefficients_row.get('note', '')}"
            ),
        },
        {
            "row_id": "latent_same_sector_public_candidate_count",
            "status": "pass" if latent_same_sector_count > 0.0 else "reject",
            "metric": "repo-wide latent same-sector public candidate count",
            "value": latent_same_sector_count,
            "note": str(latent_same_sector_row.get("note", "")),
        },
        {
            "row_id": "single_public_vpp_shape_still_missing",
            "status": str(curvature_single_vpp.get("status", "reject")),
            "metric": str(curvature_single_vpp.get("metric", "")),
            "value": float(curvature_single_vpp.get("value", 0.0)),
            "note": str(curvature_single_vpp.get("note", "")),
        },
        {
            "row_id": "single_vpp_shape_promotion_spec_frozen",
            "status": "pass",
            "metric": "promotion conditions for single public V(|P|) shape",
            "value": 4.0,
            "note": (
                "Promotion now requires: one same-sector ansatz only; direct compatibility with the surviving shell family; curvature coefficients fixed by the same-sector artifact; "
                "and explicit no-new-free-parameter wording that promotes the solver-ready row."
            ),
        },
        {
            "row_id": "effective_potential_public_candidate_stays_noncanonical",
            "status": "watch" if effective_is_phenomenological else "pass",
            "metric": "current public V(|P|) candidate is already first-principles and same-sector",
            "value": 0.0 if effective_is_phenomenological else 1.0,
            "note": (
                "The current public effective-potential branch cannot satisfy the single-V(|P|) gate because its own positioning remains phenomenological: "
                f"{effective_positioning_text}"
            ),
        },
        {
            "row_id": "single_vpp_shape_concrete_reopen_route",
            "status": "pass" if single_vpp_concrete_route else "reject",
            "metric": "concrete reopen route for single public V(|P|) shape",
            "value": 1.0 if single_vpp_concrete_route else 0.0,
            "note": (
                "A concrete route would require a non-phenomenological same-sector public ansatz already tied to the shell-family curvature map. "
                f"Current abstract-action row remains `{abstract_action_row.get('status', '')}`, repo-wide non-phenomenological public ansatz count is {latent_vpp_count}, "
                f"and the nuclear interface spread stays at {interface_spread}."
            ),
        },
        {
            "row_id": "latent_nonphenomenological_vpp_public_candidate_count",
            "status": "pass" if latent_vpp_count > 0.0 else "reject",
            "metric": "repo-wide non-phenomenological public V(|P|) ansatz count",
            "value": latent_vpp_count,
            "note": str(latent_vpp_row.get("note", "")),
        },
        {
            "row_id": "latent_reopen_route_inventory_exhausted",
            "status": "pass" if latent_inventory_exhausted else "reject",
            "metric": "repo-wide latent reopen route inventory exhausted",
            "value": 1.0 if latent_inventory_exhausted else 0.0,
            "note": str(latent_inventory_exhausted_row.get("note", "")),
        },
        {
            "row_id": "specific_missing_artifacts_fixed_state",
            "status": "pass",
            "metric": "blocked state refined to specific missing artifacts fixed",
            "value": 1.0,
            "note": (
                "The blocked state is no longer generic. The remaining blockers are frozen to the two named artifacts "
                "`positive_particle_sector_chi_p_to_vpp_public_artifact` and `single_public_vpp_shape`, plus the dependent solver-ready row."
            ),
        },
        {
            "row_id": "solver_ready_row_still_depends_on_two_named_artifacts",
            "status": "blocked",
            "metric": "solver-ready promotion blocked by named missing artifacts",
            "value": 0.0,
            "note": (
                "The solver-ready row cannot promote until both missing artifacts are realized in public canonical form. "
                f"Current reopen gate remains `{reopen.get('decision', {}).get('overall_status', '')}`."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "same-sector chi_P -> V'' artifact and single V(|P|) shape gate",
        },
        "inputs": {
            "mass_origin_readiness_gate_json": _relative_str(READINESS_JSON),
            "mass_origin_curvature_boundary_json": _relative_str(CURVATURE_JSON),
            "mass_origin_shell_quantization_canonicalization_json": _relative_str(SHELL_CANON_JSON),
            "mass_origin_solver_family_elimination_json": _relative_str(ELIMINATION_JSON),
            "mass_origin_shell_curvature_bridge_json": _relative_str(SHELL_BRIDGE_JSON),
            "mass_origin_blocked_state_reopen_json": _relative_str(REOPEN_JSON),
            "action_principle_el_derivation_audit_json": _relative_str(ACTION_JSON),
            "nuclear_effective_potential_canonical_json": _relative_str(EFFECTIVE_POTENTIAL_JSON),
            "nuclear_binding_energy_frequency_mapping_interface_json": _relative_str(INTERFACE_JSON),
            "mass_origin_latent_reopen_route_inventory_json": _relative_str(LATENT_INVENTORY_JSON),
        },
        "intent": "Freeze the minimum specification and reopen-route status for the two remaining mass-origin blockers now that the public solver family has collapsed to shell quantization only.",
        "formulas": {
            "same_sector_artifact_requirement": "same-sector chi_P or equivalent shell observable -> V''(|P|_*) with explicit sign, units, and |P|_* reference point",
            "single_vpp_shape_requirement": "one same-sector V(|P|) ansatz only, tied to the surviving shell family and fixed without new free parameters",
            "solver_ready_dependency": "same-sector artifact + single public V(|P|) shape + surviving shell family -> solver_ready_row_promoted_to_pass",
        },
        "rows": rows,
        "summary": {
            "single_public_boundary_family_fixed": single_boundary_fixed,
            "same_sector_public_artifact_available": same_sector_available,
            "same_sector_anchor_pack_available": same_sector_anchor_available,
            "same_sector_concrete_reopen_route": same_sector_concrete_route,
            "single_public_vpp_shape_available": single_vpp_available,
            "single_vpp_shape_concrete_reopen_route": single_vpp_concrete_route,
            "latent_same_sector_public_candidate_count": latent_same_sector_count,
            "latent_nonphenomenological_vpp_public_candidate_count": latent_vpp_count,
            "latent_reopen_route_inventory_exhausted": latent_inventory_exhausted,
            "remaining_missing_artifacts": [
                "positive_particle_sector_chi_p_to_vpp_public_artifact",
                "single_public_vpp_shape",
                "solver_ready_row_promoted_to_pass",
            ],
            "same_sector_minimal_spec_items": [
                "same_particle_sector_only",
                "explicit_mapping_equation",
                "sign_units_and_reference_point",
                "public_shell_family_numerical_rows",
                "no_new_free_parameter_note",
            ],
            "single_vpp_shape_promotion_items": [
                "single_same_sector_ansatz",
                "compatible_with_surviving_shell_family",
                "curvature_coefficients_fixed_by_same_sector_artifact",
                "solver_ready_row_promotion_wording",
            ],
        },
        "decision": {
            "overall_status": "specific_missing_artifacts_fixed_still_blocked",
            "keep_mass_origin_branch_blocked": True,
            "blocked_state_detail": "specific_missing_artifacts_fixed",
            "same_sector_public_artifact_available": same_sector_available,
            "same_sector_concrete_reopen_route": same_sector_concrete_route,
            "single_public_vpp_shape_available": single_vpp_available,
            "single_vpp_shape_concrete_reopen_route": single_vpp_concrete_route,
            "proceed_to_no_free_parameter_mass_solver": False,
            "proceed_to_dark_matter_branch": False,
            "next_required_artifacts": [
                "positive_particle_sector_chi_p_to_vpp_public_artifact",
                "single_public_vpp_shape",
                "solver_ready_row_promoted_to_pass",
            ],
        },
        "evidence": {
            "shell_family_row": shell_family_row,
            "shell_kappa_row": shell_kappa_row,
            "shell_kz_over_kn_row": shell_kz_row,
            "same_sector_readiness_row": readiness_same_sector,
            "single_vpp_shape_row": curvature_single_vpp,
            "shell_bridge_row": shell_bridge_row,
            "abstract_action_row": abstract_action_row,
            "effective_potential_positioning": effective_positioning,
            "latent_inventory_summary": latent_inventory.get("summary", {}),
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
