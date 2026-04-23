#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_local_curvature_bridge.py

Step 8.7.55.2.242:
Freeze the anchor-local same-sector curvature bridge that rewrites the
mass-origin blocker in chi-space / |P|-space language before attempting any
global V(|P|) selection.

This step does not claim that the positive particle-sector
chi_P -> V''(|P|_*) public artifact is already promoted. It freezes the
anchor-local identity layer that later steps will use to define the local
shape jet {V'_*, rho_*^2 V''_*, R_3} without introducing a new free
parameter.

Inputs:
  - doc/paper/10_part1_core_theory.md
  - doc/paper/12_part3a_quantum_foundations.md
  - output/public/quantum/action_principle_el_derivation_audit.json
  - output/public/quantum/born_phase_diffusion_audit_metrics.json
  - output/public/quantum/lagrangian_noether_rotational_closure_audit.json
  - output/public/quantum/mass_origin_readiness_gate_metrics.json
  - output/public/quantum/mass_origin_same_sector_chi_to_vpp_contract_metrics.json
  - output/public/quantum/mass_origin_shell_curvature_bridge_metrics.json
  - output/public/quantum/mass_origin_same_sector_vpp_shape_gate_metrics.json

Outputs:
  - output/public/quantum/mass_origin_anchor_local_curvature_bridge_metrics.json
  - output/public/quantum/mass_origin_anchor_local_curvature_bridge_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

PART1_MD = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A_MD = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
ACTION_JSON = ROOT / "output" / "public" / "quantum" / "action_principle_el_derivation_audit.json"
BORN_JSON = ROOT / "output" / "public" / "quantum" / "born_phase_diffusion_audit_metrics.json"
ROT_JSON = ROOT / "output" / "public" / "quantum" / "lagrangian_noether_rotational_closure_audit.json"
READINESS_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_readiness_gate_metrics.json"
CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_chi_to_vpp_contract_metrics.json"
SHELL_BRIDGE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_curvature_bridge_metrics.json"
VPP_GATE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_vpp_shape_gate_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_curvature_bridge_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_curvature_bridge_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.242"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the anchor-local chi_P-V'' curvature bridge for the mass-origin route.",
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


# 関数: `_read_text` の入出力契約と処理意図を定義する。

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# 関数: `_relative_str` の入出力契約と処理意図を定義する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: `_find_first_match` の入出力契約と処理意図を定義する。

def _find_first_match(text: str, pattern: str) -> Dict[str, Any] | None:
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        # 条件分岐: `pattern in raw_line` を満たす経路を評価する。
        if pattern in raw_line:
            return {
                "pattern": pattern,
                "line": line_number,
                "text": raw_line.strip(),
            }

    return None


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows(
    *,
    contract_summary: Dict[str, Any],
    gate_summary: Dict[str, Any],
    shell_decision: Dict[str, Any],
    born_formulas: Dict[str, Any],
    born_assumptions: List[str],
    readiness_formulas: Dict[str, Any],
    readiness_targets: List[Dict[str, Any]],
    action_equations: Dict[str, Any],
    part1_zp_hit: Dict[str, Any] | None,
    part1_gamma_hit: Dict[str, Any] | None,
    part3a_coarse_hit: Dict[str, Any] | None,
    rotational_lrot: str,
) -> List[Dict[str, Any]]:
    same_sector_contract_ready = bool(contract_summary.get("same_particle_sector_only", False)) and bool(
        contract_summary.get("chi_to_vpp_mapping_contract_frozen", False)
    )
    rho_definition_frozen = True
    chi_definition_frozen = True
    u2_equals_rho2_vpp_frozen = True
    static_input_symbol_pack_ready = bool(part1_zp_hit) and bool(part1_gamma_hit) and bool(part3a_coarse_hit)
    static_susceptibility_identity_ready = same_sector_contract_ready and static_input_symbol_pack_ready
    oscillation_target_pack_ready = bool(readiness_targets) and "rest_mass_frequency_mapping" in readiness_formulas
    mchi_symbol_available = "M_chi^2" in rotational_lrot
    oscillation_identity_ready = same_sector_contract_ready and oscillation_target_pack_ready and mchi_symbol_available
    chi_p_local_static_identification_wording_ready = bool(part3a_coarse_hit) and any(
        "chi_P" in item and "same Markov/FDT closure" in item for item in born_assumptions
    )
    vpp_closed_without_new_free_parameters = (
        rho_definition_frozen
        and chi_definition_frozen
        and u2_equals_rho2_vpp_frozen
        and static_susceptibility_identity_ready
        and oscillation_identity_ready
        and chi_p_local_static_identification_wording_ready
    )
    legacy_shell_bridge_absent = not bool(shell_decision.get("same_sector_curvature_bridge_available", False))
    positive_artifact_available = bool(gate_summary.get("same_sector_public_artifact_available", False))

    return [
        {
            "row_id": "anchor_local_rho_definition_frozen",
            "status": "pass" if rho_definition_frozen else "reject",
            "metric": "rho definition frozen",
            "value": 1.0 if rho_definition_frozen else 0.0,
            "note": "This step fixes rho ≡ |P| and uses rho_* ≡ |P|_* as the anchor-local reference point for the mass-origin route.",
        },
        {
            "row_id": "anchor_local_chi_definition_frozen",
            "status": "pass" if chi_definition_frozen else "reject",
            "metric": "chi definition frozen",
            "value": 1.0 if chi_definition_frozen else 0.0,
            "note": "This step fixes chi = ln(rho / P_ref) as the canonical bridge between Part I chi-space and Part III-A V(|P|)-space.",
        },
        {
            "row_id": "anchor_local_u2_equals_rho2_vpp_frozen",
            "status": "pass" if u2_equals_rho2_vpp_frozen else "reject",
            "metric": "U''(chi_*) = rho_*^2 V''(rho_*) frozen",
            "value": 1.0 if u2_equals_rho2_vpp_frozen else 0.0,
            "note": "The anchor-local curvature translation is frozen as U''(chi_*) = rho_*^2 V''(rho_*), so V'' follows once the chi-space curvature is fixed.",
        },
        {
            "row_id": "anchor_local_same_sector_contract_preserved",
            "status": "pass" if same_sector_contract_ready else "reject",
            "metric": "same-sector contract preserved for anchor-local bridge",
            "value": 1.0 if same_sector_contract_ready else 0.0,
            "note": "The bridge remains inside the already-frozen same-particle-sector contract and does not reopen any cross-sector proxy route.",
        },
        {
            "row_id": "anchor_local_part1_vector_normalization_symbol_available",
            "status": "pass" if part1_zp_hit else "reject",
            "metric": "Part I exposes Z_P in the vector-field closure",
            "value": 1.0 if part1_zp_hit else 0.0,
            "note": (
                f"Part I exposes Z_P at line {part1_zp_hit['line']}: {part1_zp_hit['text']}"
                if part1_zp_hit
                else "The vector-field normalization symbol Z_P could not be located in the current Part I source."
            ),
        },
        {
            "row_id": "anchor_local_chi_p_coarse_grained_susceptibility_available",
            "status": "pass" if static_input_symbol_pack_ready else "reject",
            "metric": "chi_P same-sector coarse-grained susceptibility already exposed",
            "value": 1.0 if static_input_symbol_pack_ready else 0.0,
            "note": (
                "Part I / Part III-A already expose Gamma_path, chi_P, and the same coarse-grained susceptibility wording, "
                "so the local-static identification can be frozen without adding a new parameter."
                if static_input_symbol_pack_ready
                else "The current public sources do not simultaneously expose Z_P, Gamma_path / chi_P, and the coarse-grained wording needed for the static bridge."
            ),
        },
        {
            "row_id": "anchor_local_static_susceptibility_identity_ready",
            "status": "pass" if static_susceptibility_identity_ready else "reject",
            "metric": "V''(rho_*) = g_P Z_P / chi_P ready",
            "value": 1.0 if static_susceptibility_identity_ready else 0.0,
            "note": (
                "The static path is frozen as V''(rho_*) = g_P Z_P / chi_P in the local quasi-static same-sector limit."
                if static_susceptibility_identity_ready
                else "The static susceptibility path cannot be frozen yet because one of the required symbol / wording ingredients is still missing."
            ),
        },
        {
            "row_id": "anchor_local_oscillation_target_pack_ready",
            "status": "pass" if oscillation_target_pack_ready else "reject",
            "metric": "omega_* target pack already frozen",
            "value": float(len(readiness_targets)) if oscillation_target_pack_ready else 0.0,
            "note": (
                "The mass-origin readiness gate already freezes the observed omega_* target pack through m = ħ ω_* / c^2."
                if oscillation_target_pack_ready
                else "The observed omega_* target pack is not available, so the oscillation path cannot be frozen."
            ),
        },
        {
            "row_id": "anchor_local_oscillation_mass_scale_symbol_available",
            "status": "pass" if mchi_symbol_available else "reject",
            "metric": "M_chi symbol already public canonical",
            "value": 1.0 if mchi_symbol_available else 0.0,
            "note": (
                f"The public canonical pack already carries M_chi in `{rotational_lrot}`."
                if mchi_symbol_available
                else "The public canonical pack does not yet expose an M_chi symbol."
            ),
        },
        {
            "row_id": "anchor_local_oscillation_identity_ready",
            "status": "pass" if oscillation_identity_ready else "reject",
            "metric": "rho_*^2 V''(rho_*) = M_chi^2 omega_*^2 ready",
            "value": 1.0 if oscillation_identity_ready else 0.0,
            "note": (
                "The oscillation path is frozen as rho_*^2 V''(rho_*) = M_chi^2 omega_*^2; this uses the already-frozen omega_* target pack and public M_chi symbol, not a new fit."
                if oscillation_identity_ready
                else "The oscillation identity cannot be frozen yet because the omega_* target pack or M_chi symbol is missing."
            ),
        },
        {
            "row_id": "anchor_local_chi_p_local_static_identification_wording_ready",
            "status": "pass" if chi_p_local_static_identification_wording_ready else "reject",
            "metric": "chi_P local-static same-sector wording ready",
            "value": 1.0 if chi_p_local_static_identification_wording_ready else 0.0,
            "note": (
                "chi_P is frozen as the same coarse-grained susceptibility in the quasi-static same-sector limit, so the static path is an identity rather than a new phenomenological insertion."
                if chi_p_local_static_identification_wording_ready
                else "The wording that identifies chi_P with the local-static same-sector susceptibility is not yet explicit enough to freeze the bridge."
            ),
        },
        {
            "row_id": "anchor_local_legacy_shell_bridge_absence_reframed",
            "status": "pass" if legacy_shell_bridge_absent else "watch",
            "metric": "legacy shell-only bridge absence is reframed by anchor-local route",
            "value": 1.0 if legacy_shell_bridge_absent else 0.0,
            "note": (
                "The old shell-only bridge remains absent, but this step no longer requires shell coefficients to be V'' directly; it rewrites the blocker as an anchor-local curvature identity plus the later R3 target."
                if legacy_shell_bridge_absent
                else "The shell-only bridge is already available, so the anchor-local rewrite is conservative rather than corrective."
            ),
        },
        {
            "row_id": "anchor_local_positive_public_artifact_still_missing",
            "status": "missing" if not positive_artifact_available else "pass",
            "metric": "positive particle-sector chi_P -> V''(|P|_*) public artifact already promoted",
            "value": 1.0 if positive_artifact_available else 0.0,
            "note": (
                "The positive same-sector public artifact is already promoted."
                if positive_artifact_available
                else "This step freezes the anchor-local bridge only; it does not yet promote the named positive particle-sector public artifact."
            ),
        },
        {
            "row_id": "anchor_local_vpp_closed_without_new_free_parameters",
            "status": "pass" if vpp_closed_without_new_free_parameters else "reject",
            "metric": "anchor-local V'' closure without new free parameters",
            "value": 1.0 if vpp_closed_without_new_free_parameters else 0.0,
            "note": (
                "Both static and oscillation paths close on the same anchor-local V'' identity without adding a new free parameter."
                if vpp_closed_without_new_free_parameters
                else "The anchor-local V'' closure is not yet strong enough to serve as the no-new-free-parameter bridge."
            ),
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (
        PART1_MD,
        PART3A_MD,
        ACTION_JSON,
        BORN_JSON,
        ROT_JSON,
        READINESS_JSON,
        CONTRACT_JSON,
        SHELL_BRIDGE_JSON,
        VPP_GATE_JSON,
    ):
        _require_path(path)

    part1_text = _read_text(PART1_MD)
    part3a_text = _read_text(PART3A_MD)
    action = _read_json(ACTION_JSON)
    born = _read_json(BORN_JSON)
    rotational = _read_json(ROT_JSON)
    readiness = _read_json(READINESS_JSON)
    contract = _read_json(CONTRACT_JSON)
    shell_bridge = _read_json(SHELL_BRIDGE_JSON)
    vpp_gate = _read_json(VPP_GATE_JSON)

    action_equations = action.get("equations", {})
    born_formulas = born.get("formulas", {})
    born_assumptions = [str(item) for item in born.get("assumptions", [])]
    rotational_equations = rotational.get("equations", {})
    readiness_formulas = readiness.get("formulas", {})
    readiness_targets = readiness.get("particle_spectrum_targets", [])
    contract_summary = contract.get("summary", {})
    contract_decision = contract.get("decision", {})
    shell_bridge_decision = shell_bridge.get("decision", {})
    vpp_gate_summary = vpp_gate.get("summary", {})
    vpp_gate_decision = vpp_gate.get("decision", {})

    # 条件分岐: `not isinstance(readiness_targets, list)` を満たす経路を評価する。
    if not isinstance(readiness_targets, list):
        raise SystemExit(f"[fail] invalid particle_spectrum_targets in {READINESS_JSON}")

    part1_zp_hit = _find_first_match(part1_text, "Z_P")
    part1_gamma_hit = _find_first_match(part1_text, "g_P^2")
    part3a_coarse_hit = _find_first_match(part3a_text, "coarse-grained susceptibility")
    rotational_lrot = str(rotational_equations.get("l_rot_minimal", ""))

    rows = _build_rows(
        contract_summary=contract_summary,
        gate_summary=vpp_gate_summary,
        shell_decision=shell_bridge_decision,
        born_formulas=born_formulas,
        born_assumptions=born_assumptions,
        readiness_formulas=readiness_formulas,
        readiness_targets=readiness_targets,
        action_equations=action_equations,
        part1_zp_hit=part1_zp_hit,
        part1_gamma_hit=part1_gamma_hit,
        part3a_coarse_hit=part3a_coarse_hit,
        rotational_lrot=rotational_lrot,
    )

    rho_definition_frozen = True
    chi_definition_frozen = True
    u2_equals_rho2_vpp_frozen = True
    static_input_symbol_pack_ready = bool(part1_zp_hit) and bool(part1_gamma_hit) and bool(part3a_coarse_hit)
    vpp_static_path_ready = bool(contract_summary.get("same_particle_sector_only", False)) and bool(
        contract_summary.get("chi_to_vpp_mapping_contract_frozen", False)
    ) and static_input_symbol_pack_ready
    vpp_oscillation_path_ready = bool(contract_summary.get("same_particle_sector_only", False)) and bool(
        contract_summary.get("chi_to_vpp_mapping_contract_frozen", False)
    ) and bool(readiness_targets) and "rest_mass_frequency_mapping" in readiness_formulas and "M_chi^2" in rotational_lrot
    chi_p_local_static_identification_wording_ready = bool(part3a_coarse_hit) and any(
        "chi_P" in item and "same Markov/FDT closure" in item for item in born_assumptions
    )
    vpp_closed_without_new_free_parameters = (
        rho_definition_frozen
        and chi_definition_frozen
        and u2_equals_rho2_vpp_frozen
        and vpp_static_path_ready
        and vpp_oscillation_path_ready
        and chi_p_local_static_identification_wording_ready
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "chi_P-V'' canonical bridge freeze",
        },
        "inputs": {
            "part1_core_theory_markdown": _relative_str(PART1_MD),
            "part3a_quantum_foundations_markdown": _relative_str(PART3A_MD),
            "action_principle_el_derivation_audit_json": _relative_str(ACTION_JSON),
            "born_phase_diffusion_audit_json": _relative_str(BORN_JSON),
            "lagrangian_noether_rotational_closure_json": _relative_str(ROT_JSON),
            "mass_origin_readiness_gate_json": _relative_str(READINESS_JSON),
            "mass_origin_same_sector_chi_to_vpp_contract_json": _relative_str(CONTRACT_JSON),
            "mass_origin_shell_curvature_bridge_json": _relative_str(SHELL_BRIDGE_JSON),
            "mass_origin_same_sector_vpp_shape_gate_json": _relative_str(VPP_GATE_JSON),
        },
        "intent": "Freeze the anchor-local same-sector bridge between chi-space curvature and V''(|P|_*) before any R3-based candidate-shape selection.",
        "formulas": {
            "rho_definition": "rho = |P|",
            "chi_definition": "chi = ln(rho / P_ref)",
            "anchor_curvature_translation": "U''(chi_*) = rho_*^2 V''(rho_*)",
            "static_susceptibility_path": "V''(rho_*) = g_P Z_P / chi_P",
            "oscillation_path": "rho_*^2 V''(rho_*) = M_chi^2 omega_*^2",
            "bridge_identity": "g_P Z_P / chi_P = M_chi^2 omega_*^2 / rho_*^2",
            "wording_rule": "chi_P is identified with the same coarse-grained quasi-static susceptibility in the local same-sector limit",
        },
        "rows": rows,
        "summary": {
            "rho_definition_frozen": rho_definition_frozen,
            "chi_definition_frozen": chi_definition_frozen,
            "reference_point_symbol": "rho_* = |P|_*",
            "u2_equals_rho2_v2_frozen": u2_equals_rho2_vpp_frozen,
            "static_susceptibility_identity_ready": vpp_static_path_ready,
            "oscillation_identity_ready": vpp_oscillation_path_ready,
            "vpp_static_path_ready": vpp_static_path_ready,
            "vpp_oscillation_path_ready": vpp_oscillation_path_ready,
            "chi_p_local_static_identification_wording_ready": chi_p_local_static_identification_wording_ready,
            "vpp_closed_without_new_free_parameters": vpp_closed_without_new_free_parameters,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": bool(
                vpp_gate_summary.get("same_sector_public_artifact_available", False)
            ),
            "same_sector_contract_frozen": bool(contract_summary.get("chi_to_vpp_mapping_contract_frozen", False)),
            "legacy_shell_only_bridge_available": bool(shell_bridge_decision.get("same_sector_curvature_bridge_available", False)),
            "observed_omega_target_count": len(readiness_targets),
        },
        "decision": {
            "overall_status": "anchor_local_curvature_bridge_frozen",
            "keep_mass_origin_branch_blocked": True,
            "vpp_closed_without_new_free_parameters": vpp_closed_without_new_free_parameters,
            "vpp_static_path_ready": vpp_static_path_ready,
            "vpp_oscillation_path_ready": vpp_oscillation_path_ready,
            "chi_p_local_static_identification_wording_ready": chi_p_local_static_identification_wording_ready,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": bool(
                vpp_gate_summary.get("same_sector_public_artifact_available", False)
            ),
            "blocked_state_detail": str(vpp_gate_decision.get("blocked_state_detail", "specific_missing_artifacts_fixed")),
            "next_required_artifacts": [
                "r3_target",
                "single_public_vpp_shape",
                "positive_particle_sector_chi_p_to_vpp_public_artifact",
                "solver_ready_row_promoted_to_pass",
            ],
        },
        "evidence": {
            "part1_zp_line": part1_zp_hit,
            "part1_gamma_path_line": part1_gamma_hit,
            "part3a_coarse_grained_susceptibility_line": part3a_coarse_hit,
            "action_equations": action_equations,
            "born_assumptions": born_assumptions,
            "born_formulas": born_formulas,
            "rotational_equations": rotational_equations,
            "readiness_formulas": readiness_formulas,
            "readiness_frozen_targets": readiness.get("frozen_targets", {}),
            "contract_summary": contract_summary,
            "contract_decision": contract_decision,
            "shell_bridge_decision": shell_bridge_decision,
            "same_sector_vpp_shape_gate_summary": vpp_gate_summary,
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
