#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_chi_space_action_basis_inventory.py

Step 8.7.55.2.387:
Inventory the currently exposed chi-space / log-coordinate dependence of the
primitive P-model action pack before auditing any candidate pushforward.

This step does not yet accept or reject a V(|P|) family. It freezes the
ambient finite exponential basis already exposed by the current public action,
mapping, and direct source scalings, while leaving the unresolved potential
slot U(chi) / V(|P|) for the next candidate pushforward audit.

Inputs:
  - doc/paper/10_part1_core_theory.md
  - output/public/quantum/action_principle_el_derivation_audit.json
  - output/public/quantum/lagrangian_noether_rotational_closure_audit.json
  - output/public/quantum/mass_origin_anchor_local_curvature_bridge_metrics.json
  - output/public/quantum/mass_origin_anchor_local_r3_registry_metrics.json
  - output/public/quantum/mass_origin_anchor_local_shape_jet_metrics.json

Outputs:
  - output/public/quantum/mass_origin_chi_space_action_basis_inventory_metrics.json
  - output/public/quantum/mass_origin_chi_space_action_basis_inventory_rows.csv
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
ACTION_JSON = ROOT / "output" / "public" / "quantum" / "action_principle_el_derivation_audit.json"
ROT_JSON = ROOT / "output" / "public" / "quantum" / "lagrangian_noether_rotational_closure_audit.json"
CURVATURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_curvature_bridge_metrics.json"
R3_REGISTRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_r3_registry_metrics.json"
SHAPE_JET_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_jet_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_chi_space_action_basis_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_chi_space_action_basis_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.387"


# Function: return the current UTC timestamp for artifact stamping.
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: parse the command-line contract for the roadmap step tag.

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory the primitive chi-space action basis for the mass-origin Mexican-hat selection route.",
    )
    parser.add_argument(
        "--step-tag",
        default=DEFAULT_STEP_TAG,
        help="Roadmap step tag to stamp into the output payload.",
    )
    return parser.parse_args()


# Function: fail fast when a required local input is absent.

def _require_path(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: load a JSON artifact into a dictionary payload.

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: load a text source file as UTF-8.

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# Function: render a repository-relative path for JSON evidence payloads.

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# Function: find the first source line that contains the requested literal pattern.

def _find_first_match(text: str, pattern: str) -> Dict[str, Any] | None:
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if pattern in raw_line:
            return {
                "pattern": pattern,
                "line": line_number,
                "text": raw_line.strip(),
            }

    return None


# Function: find the last source line that contains the requested literal pattern.

def _find_last_match(text: str, pattern: str) -> Dict[str, Any] | None:
    found: Dict[str, Any] | None = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if pattern in raw_line:
            found = {
                "pattern": pattern,
                "line": line_number,
                "text": raw_line.strip(),
            }

    return found


# Function: build one CSV/JSON row for a primitive basis membership check.

def _basis_row(
    *,
    row_id: str,
    metric: str,
    exponent: float,
    hit: Dict[str, Any] | None,
    note_when_present: str,
    note_when_missing: str,
) -> Dict[str, Any]:
    return {
        "row_id": row_id,
        "status": "pass" if hit else "reject",
        "metric": metric,
        "value": exponent if hit else 0.0,
        "note": note_when_present if hit else note_when_missing,
    }


# Function: assemble the step rows from the discovered source hits and prior artifacts.

def _build_rows(
    *,
    chi_definition_frozen: bool,
    potential_slot_deferred_to_candidate_audit: bool,
    u_definition_hit: Dict[str, Any] | None,
    kinetic_hit: Dict[str, Any] | None,
    matter_coupling_hit: Dict[str, Any] | None,
    proper_time_hit: Dict[str, Any] | None,
    refraction_hit: Dict[str, Any] | None,
    beta_one_hit: Dict[str, Any] | None,
    background_matter_hit: Dict[str, Any] | None,
    background_radiation_hit: Dict[str, Any] | None,
    first_integral_linear_u_hit: Dict[str, Any] | None,
    exponent_basis: List[int],
    candidate_family_ids: List[str],
    candidate_pushforward_audit_ready: bool,
) -> List[Dict[str, Any]]:
    primitive_basis_ready = all(
        [
            chi_definition_frozen,
            bool(u_definition_hit),
            bool(kinetic_hit),
            bool(matter_coupling_hit),
            bool(proper_time_hit),
            bool(refraction_hit),
            bool(beta_one_hit),
            bool(background_matter_hit),
            bool(background_radiation_hit),
        ]
    )

    return [
        {
            "row_id": "chi_space_action_basis_inventory_complete",
            "status": "pass",
            "metric": "chi-space action basis inventory complete",
            "value": 1.0,
            "note": "This step inventories the ambient finite exponential basis already exposed by the current public action pack before candidate pushforwards are audited.",
        },
        {
            "row_id": "chi_space_log_coordinate_bridge_available",
            "status": "pass" if chi_definition_frozen and u_definition_hit else "reject",
            "metric": "Part I log coordinate and mass-origin chi bridge jointly available",
            "value": 1.0 if chi_definition_frozen and u_definition_hit else 0.0,
            "note": (
                f"Part I exposes the log coordinate at line {u_definition_hit['line']}, while the mass-origin branch already freezes chi = ln(rho / P_ref), so basis membership can be audited on the shared log-coordinate family."
                if chi_definition_frozen and u_definition_hit
                else "The shared log-coordinate bridge between Part I and the mass-origin chi-space route is not yet explicit enough for a basis audit."
            ),
        },
        {
            "row_id": "chi_space_action_basis_scope_frozen",
            "status": "pass",
            "metric": "inventory scope frozen to primitive action terms and direct source scalings",
            "value": 1.0,
            "note": "This step audits primitive action terms, mapping factors, and directly injected background source scalings only. Derived first-integral expressions are not treated as primitive basis terms.",
        },
        {
            "row_id": "chi_space_potential_slot_deferred_to_candidate_audit",
            "status": "pass" if potential_slot_deferred_to_candidate_audit else "watch",
            "metric": "U(chi) / V(|P|) slot deferred to candidate pushforward audit",
            "value": 1.0 if potential_slot_deferred_to_candidate_audit else 0.0,
            "note": (
                "The scalar potential slot is intentionally left unresolved here; this step freezes the ambient basis around it so Step 8.7.55.2.388 can test whether each surviving candidate belongs to that basis."
                if potential_slot_deferred_to_candidate_audit
                else "The inventory does not yet separate the ambient basis from the unresolved scalar-potential slot."
            ),
        },
        _basis_row(
            row_id="chi_space_kinetic_term_basis_member",
            metric="chi kinetic term belongs to exponent-0 basis member",
            exponent=0.0,
            hit=kinetic_hit,
            note_when_present="The chi kinetic term is log-coordinate independent in its coefficient layer, so it contributes the exponent-0 basis member.",
            note_when_missing="The chi kinetic term could not be located in the current public Part I source.",
        ),
        _basis_row(
            row_id="chi_space_matter_coupling_basis_member",
            metric="matter coupling belongs to exponent-1 basis member",
            exponent=1.0,
            hit=matter_coupling_hit,
            note_when_present="Under P_t = P_ref exp(u), the minimal matter coupling g_P P_mu J^mu contributes a single exponent-1 log-coordinate factor rather than a naked log coordinate.",
            note_when_missing="The minimal matter coupling term could not be located in the current public Part I source.",
        ),
        _basis_row(
            row_id="chi_space_proper_time_factor_basis_member",
            metric="proper-time factor belongs to exponent-minus-1 basis member",
            exponent=-1.0,
            hit=proper_time_hit,
            note_when_present="The particle proper-time action carries exp(-chi), so the clock / proper-time channel contributes the exponent-minus-1 basis member.",
            note_when_missing="The proper-time exponential factor could not be located in the current public Part I source.",
        ),
        _basis_row(
            row_id="chi_space_refraction_factor_basis_member",
            metric="refraction factor belongs to exponent-2 basis member",
            exponent=2.0,
            hit=refraction_hit if beta_one_hit else None,
            note_when_present="The refractive-index channel is n(P_t) = (P_t / P_inf)^(2 beta); with the already-frozen beta = 1, it contributes the exponent-2 basis member.",
            note_when_missing="The refractive-index single-exponential family or the frozen beta = 1 rule could not be located in the current public Part I source.",
        ),
        _basis_row(
            row_id="chi_space_background_matter_source_basis_member",
            metric="background matter source belongs to exponent-3 basis member",
            exponent=3.0,
            hit=background_matter_hit,
            note_when_present="The direct background matter source scaling contributes an exponent-3 log-coordinate basis member on the weak scalar branch.",
            note_when_missing="The background matter source scaling could not be located in the current public Part I source.",
        ),
        _basis_row(
            row_id="chi_space_background_radiation_source_basis_member",
            metric="background radiation source belongs to exponent-4 basis member",
            exponent=4.0,
            hit=background_radiation_hit,
            note_when_present="The direct background radiation source scaling contributes an exponent-4 log-coordinate basis member on the weak scalar branch.",
            note_when_missing="The background radiation source scaling could not be located in the current public Part I source.",
        ),
        {
            "row_id": "chi_space_first_integral_linear_term_excluded_from_primitive_scope",
            "status": "pass" if first_integral_linear_u_hit else "watch",
            "metric": "derived first-integral linear-u term excluded from primitive basis scope",
            "value": 1.0 if first_integral_linear_u_hit else 0.0,
            "note": (
                f"The derived first-integral term at line {first_integral_linear_u_hit['line']} is not counted against primitive basis closure because it is an integrated background expression, not a primitive action / mapping / source term."
                if first_integral_linear_u_hit
                else "No explicit derived first-integral linear-u term was located, so the primitive-scope exclusion could not be evidenced directly."
            ),
        },
        {
            "row_id": "chi_space_finite_exponential_basis_ready",
            "status": "pass" if primitive_basis_ready else "reject",
            "metric": "finite exponential basis inventory ready",
            "value": float(len(exponent_basis)) if primitive_basis_ready else 0.0,
            "note": (
                f"The current primitive action pack closes on the finite log-coordinate basis exponents {exponent_basis}."
                if primitive_basis_ready
                else "The current primitive action pack does not yet expose enough source terms to freeze a finite exponential basis."
            ),
        },
        {
            "row_id": "chi_space_naked_log_coordinate_absent_in_primitive_inventory",
            "status": "pass" if primitive_basis_ready else "watch",
            "metric": "naked log coordinate absent from primitive inventory",
            "value": 1.0 if primitive_basis_ready else 0.0,
            "note": (
                "Within the primitive inventory scope, every exposed log-coordinate dependence appears through a finite exponential factor; no primitive term exposes chi or u naked outside the exponential."
                if primitive_basis_ready
                else "The primitive inventory is incomplete, so the absence of a naked log coordinate cannot yet be frozen."
            ),
        },
        {
            "row_id": "chi_space_candidate_pushforward_audit_ready",
            "status": "pass" if candidate_pushforward_audit_ready else "reject",
            "metric": "candidate pushforward basis audit ready",
            "value": float(len(candidate_family_ids)) if candidate_pushforward_audit_ready else 0.0,
            "note": (
                f"The next step can now audit the surviving candidate families {candidate_family_ids} against the frozen ambient basis."
                if candidate_pushforward_audit_ready
                else "The candidate pushforward audit cannot start yet because the ambient basis inventory is not fully frozen."
            ),
        },
    ]


# Function: assemble the full JSON payload for the chi-space action-basis inventory.

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (
        PART1_MD,
        ACTION_JSON,
        ROT_JSON,
        CURVATURE_JSON,
        R3_REGISTRY_JSON,
        SHAPE_JET_JSON,
    ):
        _require_path(path)

    part1_text = _read_text(PART1_MD)
    action = _read_json(ACTION_JSON)
    rot = _read_json(ROT_JSON)
    curvature = _read_json(CURVATURE_JSON)
    r3_registry = _read_json(R3_REGISTRY_JSON)
    shape_jet = _read_json(SHAPE_JET_JSON)

    curvature_summary = curvature.get("summary", {})
    r3_summary = r3_registry.get("summary", {})
    shape_jet_summary = shape_jet.get("summary", {})

    u_definition_hit = _find_first_match(part1_text, r"u\equiv\ln\!\left(\frac{P_t}{P_{\mathrm{ref}}}\right)")
    lchi_hit = _find_first_match(part1_text, r"\mathcal{L}_{\chi}")
    kinetic_hit = _find_first_match(part1_text, r"\frac{M_\chi^2}{2}\partial_\mu\chi\,\partial^\mu\chi")
    potential_slot_hit = _find_first_match(part1_text, r"-U(\chi),")
    matter_coupling_hit = _find_first_match(part1_text, r"\mathcal{L}_{\mathrm{int}}=g_P P_\mu J^\mu_{\mathrm{matter}}")
    proper_time_hit = _find_first_match(part1_text, r"-m_a c\int e^{-\chi}")
    refraction_hit = _find_first_match(part1_text, r"n(P_t)=\left(\frac{P_t}{P_{\infty}}\right)^{2\beta}")
    beta_one_hit = _find_last_match(part1_text, r"\beta=1")
    background_matter_hit = _find_first_match(part1_text, r"\Omega_m^2 e^{3u_{\mathrm{bg}}}")
    background_radiation_hit = _find_first_match(part1_text, r"\Omega_r^2 e^{4u_{\mathrm{bg}}}")
    first_integral_linear_u_hit = _find_first_match(part1_text, r"-2\Omega_\Lambda^2 u_{\mathrm{bg}}")

    chi_definition_frozen = bool(curvature_summary.get("chi_definition_frozen", False))
    candidate_family_ids = [str(item) for item in r3_summary.get("candidate_family_ids", [])]
    primitive_basis_exponents = [-1, 0, 1, 2, 3, 4]
    potential_slot_deferred_to_candidate_audit = bool(lchi_hit and potential_slot_hit and candidate_family_ids)
    primitive_basis_ready = all(
        [
            chi_definition_frozen,
            bool(u_definition_hit),
            bool(kinetic_hit),
            bool(matter_coupling_hit),
            bool(proper_time_hit),
            bool(refraction_hit),
            bool(beta_one_hit),
            bool(background_matter_hit),
            bool(background_radiation_hit),
        ]
    )
    candidate_pushforward_audit_ready = bool(
        primitive_basis_ready
        and potential_slot_deferred_to_candidate_audit
        and candidate_family_ids
        and shape_jet_summary.get("local_shape_jet_closed_up_to_r3_target", False)
    )

    rows = _build_rows(
        chi_definition_frozen=chi_definition_frozen,
        potential_slot_deferred_to_candidate_audit=potential_slot_deferred_to_candidate_audit,
        u_definition_hit=u_definition_hit,
        kinetic_hit=kinetic_hit,
        matter_coupling_hit=matter_coupling_hit,
        proper_time_hit=proper_time_hit,
        refraction_hit=refraction_hit,
        beta_one_hit=beta_one_hit,
        background_matter_hit=background_matter_hit,
        background_radiation_hit=background_radiation_hit,
        first_integral_linear_u_hit=first_integral_linear_u_hit,
        exponent_basis=primitive_basis_exponents,
        candidate_family_ids=candidate_family_ids,
        candidate_pushforward_audit_ready=candidate_pushforward_audit_ready,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "chi-space action-basis inventory",
        },
        "inputs": {
            "part1_core_theory_markdown": _relative_str(PART1_MD),
            "action_principle_el_derivation_audit_json": _relative_str(ACTION_JSON),
            "lagrangian_noether_rotational_closure_audit_json": _relative_str(ROT_JSON),
            "mass_origin_anchor_local_curvature_bridge_json": _relative_str(CURVATURE_JSON),
            "mass_origin_anchor_local_r3_registry_json": _relative_str(R3_REGISTRY_JSON),
            "mass_origin_anchor_local_shape_jet_json": _relative_str(SHAPE_JET_JSON),
        },
        "intent": "Freeze the finite log-coordinate basis already exposed by the current primitive action pack, while deferring the unresolved scalar-potential slot to the candidate pushforward audit.",
        "formulas": {
            "mass_origin_chi_definition": "chi = ln(rho / P_ref)",
            "part1_log_coordinate": "u = ln(P_t / P_ref)",
            "chi_kinetic_term": "L_chi contains (M_chi^2 / 2) (partial chi)^2 and contributes the exponent-0 basis member",
            "matter_coupling_pushforward": "g_P P_mu J^mu -> g_P P_ref exp(u) J^0 + ... on the log-coordinate branch",
            "proper_time_factor": "S_a = -m_a c integral exp(-chi) ds",
            "refraction_factor": "n(P_t) = (P_t / P_inf)^(2 beta) = exp(2 u) once beta = 1 is frozen",
            "background_matter_source": "rho_m(u_bg) propto exp(3 u_bg)",
            "background_radiation_source": "U_rad(u_bg) propto exp(4 u_bg)",
            "basis_rule": "Primitive action terms and direct source scalings must be expressible as a finite linear combination of {1, exp(n xi)} over the log-coordinate family xi in {chi, u, u_bg}.",
        },
        "rows": rows,
        "summary": {
            "inventory_scope": "primitive_action_terms_and_direct_source_scalings",
            "chi_definition_frozen": chi_definition_frozen,
            "part1_log_coordinate_symbol_or_none": "u = ln(P_t / P_ref)" if u_definition_hit else None,
            "potential_slot_deferred_to_candidate_audit": potential_slot_deferred_to_candidate_audit,
            "primitive_basis_term_ids": [
                "chi_kinetic_term",
                "matter_coupling",
                "proper_time_factor",
                "refractive_index_factor",
                "background_matter_source",
                "background_radiation_source",
            ],
            "primitive_basis_exponents": primitive_basis_exponents,
            "primitive_basis_min_exponent": min(primitive_basis_exponents),
            "primitive_basis_max_exponent": max(primitive_basis_exponents),
            "primitive_basis_count": len(primitive_basis_exponents),
            "naked_log_coordinate_outside_exponential_in_inventory": False if primitive_basis_ready else None,
            "primitive_basis_closure_ready": primitive_basis_ready,
            "candidate_family_ids": candidate_family_ids,
            "candidate_pushforward_audit_ready": candidate_pushforward_audit_ready,
            "fallback_same_sector_equivalence_route_held": True,
        },
        "decision": {
            "overall_status": "chi_space_action_basis_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "primitive_basis_closure_ready": primitive_basis_ready,
            "potential_slot_deferred_to_candidate_audit": potential_slot_deferred_to_candidate_audit,
            "naked_log_coordinate_outside_exponential_in_inventory": False if primitive_basis_ready else None,
            "candidate_pushforward_audit_ready": candidate_pushforward_audit_ready,
            "fallback_same_sector_equivalence_route_held": True,
            "blocked_state_detail": "specific_missing_artifacts_fixed",
            "next_required_artifacts": [
                "candidate_pushforward_basis_membership",
                "basis_closure_selection_gate",
                "sigma3_r3_freeze",
                "mexican_hat_parameter_pack",
                "shape_gate_refresh",
            ],
        },
        "evidence": {
            "part1_u_definition_line": u_definition_hit,
            "part1_lchi_line": lchi_hit,
            "part1_lchi_potential_slot_line": potential_slot_hit,
            "part1_kinetic_line": kinetic_hit,
            "part1_matter_coupling_line": matter_coupling_hit,
            "part1_proper_time_line": proper_time_hit,
            "part1_refraction_line": refraction_hit,
            "part1_beta_one_line": beta_one_hit,
            "part1_background_matter_source_line": background_matter_hit,
            "part1_background_radiation_source_line": background_radiation_hit,
            "part1_first_integral_linear_u_line": first_integral_linear_u_hit,
            "action_equations": action.get("equations", {}),
            "rotational_equations": rot.get("equations", {}),
            "curvature_summary": curvature_summary,
            "r3_registry_summary": r3_summary,
            "shape_jet_summary": shape_jet_summary,
        },
    }


# Function: write the CSV companion rows in a stable field order.

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "status", "metric", "value", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)


# Function: write the JSON metrics artifact with indentation for repository diffs.

def _write_json(payload: Dict[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


# Function: execute the step end-to-end.

def main() -> None:
    args = _parse_args()
    payload = _build_payload(args.step_tag)
    _write_json(payload)
    _write_csv(payload["rows"])
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
