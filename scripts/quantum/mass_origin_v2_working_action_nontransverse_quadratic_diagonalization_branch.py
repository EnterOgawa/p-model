#!/usr/bin/env python3
"""
Generate working-action nontransverse quadratic-diagonalization artifacts for 8.7.56.186-.189.

This branch deepens the post-breakthrough vector rebuild residual route after
the temporal/longitudinal/Stueckelberg-basis audit. Current canon already
exposes the gauge-invariant hint Pi_mu, the Stückelberg-completed action, the
propagator mix terms, and the ghost-free guard. The unresolved question is
whether those ingredients are ever rewritten as an explicit component-level
quadratic form for the nontransverse complement once the transverse photon
branch has been removed.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

TRIAL1_BREAKTHROUGH_MAXWELL = OUT / "mass_origin_v2_trial1_breakthrough_maxwell_coupling_audit_metrics.json"
TRIAL2_DECLARATION = OUT / "mass_origin_v2_trial2_declaration_gate_metrics.json"
TEMP_BASIS_SOURCE = OUT / "mass_origin_v2_working_action_temporal_longitudinal_stueckelberg_basis_source_inventory_metrics.json"
TEMP_BASIS_AUDIT = OUT / "mass_origin_v2_working_action_temporal_longitudinal_stueckelberg_basis_identification_audit_metrics.json"
REOPEN_FIFTH = OUT / "mass_origin_v2_vector_rebuild_reopen_fifth_gate_metrics.json"
NONTRANSVERSE_ROUTE = OUT / "mass_origin_v2_working_action_nontransverse_quadratic_diagonalization_route_contract_metrics.json"
VECTOR_REDUCED_SOLVER = OUT / "mass_origin_v2_vector_reduced_solver_pilot_metrics.json"
VECTOR_ANCHOR = OUT / "mass_origin_v2_vector_anchor_refresh_metrics.json"
TRIAL3_FALLBACK_ROUTE = OUT / "mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_metrics.json"


# Function: return the current UTC timestamp in ISO 8601 format.
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: stop execution if a required input path is missing.

def req(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: load a UTF-8 JSON artifact.

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: load a UTF-8 text source.

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# Function: convert an absolute path into a repository-relative path.

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# Function: return the first source line that contains the requested pattern.

def hit(text: str, pattern: str) -> dict | None:
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: build a standard result row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: build a standard payload object.

def payload(
    step: str,
    name: str,
    inputs: dict,
    intent: str,
    formulas: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "intent": intent,
        "formulas": formulas,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# Function: save a JSON artifact and its row table.

def write_artifact(stem: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: build a standard inventory target record.

def target_record(file_key: str, path: Path, text: str, pattern: str, note: str) -> dict:
    target_hit = hit(text, pattern)
    return {
        "file_key": file_key,
        "file": rel(path),
        "pattern": pattern,
        "present": target_hit is not None,
        "note": note,
        "evidence": target_hit,
    }


# Function: execute the working-action nontransverse quadratic-diagonalization residual branch.

def main() -> None:
    for path in (
        PART1,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        TRIAL1_BREAKTHROUGH_MAXWELL,
        TRIAL2_DECLARATION,
        TEMP_BASIS_SOURCE,
        TEMP_BASIS_AUDIT,
        REOPEN_FIFTH,
        NONTRANSVERSE_ROUTE,
        VECTOR_REDUCED_SOLVER,
        VECTOR_ANCHOR,
        TRIAL3_FALLBACK_ROUTE,
    ):
        req(path)

    part1 = read_text(PART1)
    status = read_text(STATUS)
    roadmap = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)

    breakthrough_maxwell = read_json(TRIAL1_BREAKTHROUGH_MAXWELL)
    trial2_declaration = read_json(TRIAL2_DECLARATION)
    temp_basis_source = read_json(TEMP_BASIS_SOURCE)
    temp_basis_audit = read_json(TEMP_BASIS_AUDIT)
    reopen_fifth = read_json(REOPEN_FIFTH)
    nontransverse_route = read_json(NONTRANSVERSE_ROUTE)
    vector_reduced_solver = read_json(VECTOR_REDUCED_SOLVER)
    vector_anchor = read_json(VECTOR_ANCHOR)
    trial3_fallback_route = read_json(TRIAL3_FALLBACK_ROUTE)

    common_inputs = {
        "part1_markdown": rel(PART1),
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_trial1_breakthrough_maxwell_coupling_audit_json": rel(TRIAL1_BREAKTHROUGH_MAXWELL),
        "mass_origin_v2_trial2_declaration_gate_json": rel(TRIAL2_DECLARATION),
        "mass_origin_v2_working_action_temporal_longitudinal_stueckelberg_basis_source_inventory_json": rel(TEMP_BASIS_SOURCE),
        "mass_origin_v2_working_action_temporal_longitudinal_stueckelberg_basis_identification_audit_json": rel(TEMP_BASIS_AUDIT),
        "mass_origin_v2_vector_rebuild_reopen_fifth_gate_json": rel(REOPEN_FIFTH),
        "mass_origin_v2_working_action_nontransverse_quadratic_diagonalization_route_contract_json": rel(NONTRANSVERSE_ROUTE),
        "mass_origin_v2_vector_reduced_solver_pilot_json": rel(VECTOR_REDUCED_SOLVER),
        "mass_origin_v2_vector_anchor_refresh_json": rel(VECTOR_ANCHOR),
        "mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_json": rel(TRIAL3_FALLBACK_ROUTE),
    }

    source_targets = [
        target_record(
            "part1_full_action_line",
            PART1,
            part1,
            "\\mathcal{L}_{P,\\mathrm{full}}",
            "Part I still freezes the Lorentz-covariant Stückelberg-completed action used as the source hint for any nontransverse reduction.",
        ),
        target_record(
            "part1_stueckelberg_mass_line",
            PART1,
            part1,
            "+\\frac{m_P^2}{2}\\left(P_\\mu-\\frac{1}{m_P}\\partial_\\mu\\pi\\right)",
            "The explicit Pi_mu Pi^mu-type mass source still exists as a source hint inside the compact action.",
        ),
        target_record(
            "part1_gauge_fixing_line",
            PART1,
            part1,
            "-\\frac{1}{2\\xi_g}\\left(\\partial_\\mu P^\\mu+\\xi_g m_P\\pi\\right)^2",
            "The gauge-fixing term still exposes the temporal/longitudinal/Stueckelberg mix pack.",
        ),
        target_record(
            "part1_propagator_massive_pole_line",
            PART1,
            part1,
            "\\frac{\\eta_{\\mu\\nu}-k_\\mu k_\\nu/m_P^2}{k^2-m_P^2+i0}",
            "The old propagator still exposes the massive pole structure.",
        ),
        target_record(
            "part1_propagator_gauge_piece_line",
            PART1,
            part1,
            "\\frac{\\xi_g\\,k_\\mu k_\\nu/m_P^2}{k^2-\\xi_g m_P^2+i0}",
            "The old propagator still exposes the gauge-dependent longitudinal/Stueckelberg pole piece.",
        ),
        target_record(
            "part1_pi_mu_definition_line",
            PART1,
            part1,
            "\\Pi_\\mu:=P_\\mu-\\partial_\\mu\\pi/m_P",
            "Part I still defines the gauge-invariant Pi_mu combination.",
        ),
        target_record(
            "part1_ghost_free_line",
            PART1,
            part1,
            "負ノルム（ghost）モードは出現しない。",
            "Any component-level diagonalization must preserve the ghost-free closure.",
        ),
        target_record(
            "status_next_step_anchor",
            STATUS,
            status,
            "current official next step は `8.7.56.186`",
            "STATUS must already expose the nontransverse-diagonalization branch as the next official route.",
        ),
        target_record(
            "roadmap_branch_anchor",
            ROADMAP,
            roadmap,
            "`8.7.56.185-.188`",
            "ROADMAP must already expose the current nontransverse-diagonalization residual branch.",
        ),
    ]
    source_pack_ready = all(item["present"] for item in source_targets)

    transverse_photon_branch_ready = bool(
        temp_basis_source["summary"]["transverse_photon_branch_ready"]
    )
    transverse_photon_subtraction_ready = bool(
        temp_basis_source["summary"]["transverse_photon_subtraction_ready"]
    )
    pi_mu_hint_ready = source_targets[5]["present"]
    stueckelberg_full_quadratic_pack_ready = (
        source_targets[0]["present"] and source_targets[1]["present"] and source_targets[2]["present"]
    )
    propagator_mix_pack_ready = source_targets[3]["present"] and source_targets[4]["present"]
    ghost_free_guard_ready = source_targets[6]["present"]

    source_inventory = payload(
        "8.7.56.186",
        "Working-action nontransverse quadratic diagonalization source inventory",
        common_inputs,
        "Freeze the source pack needed to decide whether the current canon rewrites the nontransverse complement as an explicit component-level quadratic form after the transverse photon branch is removed.",
        {
            "working_action_rule": "the breakthrough working action keeps A_mu = delta P_mu^T / sqrt(Z_P) as the transverse photon branch",
            "complement_rule": "the remaining nontransverse sector must be extracted from the compact Stückelberg-completed action and propagator mix pack",
            "diagonalization_requirement": "a current-canon diagonalization needs an explicit component-level quadratic form for the nontransverse complement, not only a compact Lorentz-covariant action and a propagator hint",
            "ghost_rule": "any successful reduction must preserve the ghost-free closure already fixed in Part I",
        },
        [
            row(
                "working_action_nontransverse_quadratic_diagonalization_source_inventory_complete",
                "pass",
                "working-action nontransverse quadratic diagonalization source inventory complete",
                1,
                "The source pack for the nontransverse diagonalization branch is frozen.",
            ),
            row(
                "working_action_nontransverse_quadratic_diagonalization_source_pack_ready",
                "pass" if source_pack_ready else "reject",
                "working-action nontransverse quadratic diagonalization source pack ready",
                1 if source_pack_ready else 0,
                "The residual branch has the required canonical anchors.",
            ),
            row(
                "working_action_nontransverse_quadratic_diagonalization_photon_branch_ready",
                "pass" if transverse_photon_branch_ready and transverse_photon_subtraction_ready else "reject",
                "transverse photon branch ready before nontransverse diagonalization",
                1 if transverse_photon_branch_ready and transverse_photon_subtraction_ready else 0,
                "The nontransverse branch begins only after the photon branch is identified and subtracted.",
            ),
            row(
                "working_action_nontransverse_quadratic_diagonalization_pi_mu_hint_ready",
                "pass" if pi_mu_hint_ready else "reject",
                "Pi_mu hint ready",
                1 if pi_mu_hint_ready else 0,
                "The gauge-invariant Pi_mu combination remains visible as a source hint.",
            ),
            row(
                "working_action_nontransverse_quadratic_diagonalization_stueckelberg_pack_ready",
                "pass" if stueckelberg_full_quadratic_pack_ready else "reject",
                "Stueckelberg full quadratic pack ready",
                1 if stueckelberg_full_quadratic_pack_ready else 0,
                "The compact action plus gauge-fixing still exposes the unresolved mix pack.",
            ),
            row(
                "working_action_nontransverse_quadratic_diagonalization_propagator_mix_pack_ready",
                "pass" if propagator_mix_pack_ready else "reject",
                "propagator mix pack ready",
                1 if propagator_mix_pack_ready else 0,
                "The old propagator still exposes the massive and gauge-dependent pole pieces.",
            ),
            row(
                "working_action_nontransverse_quadratic_diagonalization_ghost_guard_ready",
                "pass" if ghost_free_guard_ready else "reject",
                "ghost-free guard ready",
                1 if ghost_free_guard_ready else 0,
                "Any future diagonalization must preserve the ghost-free closure.",
            ),
        ],
        {
            "working_action_nontransverse_quadratic_diagonalization_source_pack_ready": source_pack_ready,
            "transverse_photon_branch_ready": transverse_photon_branch_ready,
            "transverse_photon_subtraction_ready": transverse_photon_subtraction_ready,
            "pi_mu_gauge_invariant_hint_ready": pi_mu_hint_ready,
            "stueckelberg_full_quadratic_pack_ready": stueckelberg_full_quadratic_pack_ready,
            "propagator_mix_pack_ready": propagator_mix_pack_ready,
            "ghost_free_guard_ready": ghost_free_guard_ready,
            "first_route_to_close_or_none": "working_action_nontransverse_quadratic_diagonalization_identification_audit",
        },
        {
            "overall_status": "working_action_nontransverse_quadratic_diagonalization_source_inventory_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_187": True,
            "next_required_artifacts": ["working_action_nontransverse_quadratic_diagonalization_identification_audit"],
        },
        {
            "inventory_targets": source_targets,
            "temp_basis_source_summary": temp_basis_source["summary"],
            "temp_basis_audit_summary": temp_basis_audit["summary"],
            "breakthrough_maxwell_summary": breakthrough_maxwell["summary"],
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    nontransverse_component_quadratic_form_available = False
    nontransverse_quadratic_diagonalization_available = False
    massive_sector_projector_available = False

    identification_audit = payload(
        "8.7.56.187",
        "Working-action nontransverse quadratic diagonalization identification audit",
        common_inputs,
        "Audit whether the current canon actually rewrites the nontransverse complement into an explicit component-level quadratic form that can then be diagonalized after photon extraction.",
        {
            "compact_action_rule": "the compact full action is a source hint, but it is not yet a component-level diagonalized statement for the nontransverse complement",
            "propagator_rule": "the propagator mix pack exposes massive and gauge-dependent poles, but it still does not specify the canonical complement basis after the transverse photon is removed",
            "component_rule": "a successful identification needs an explicit quadratic form for the nontransverse component fields before a projector or radial eigenoperator can be claimed",
        },
        [
            row(
                "working_action_nontransverse_quadratic_diagonalization_pi_mu_hint_available",
                "pass" if pi_mu_hint_ready else "reject",
                "Pi_mu gauge-invariant hint available",
                1 if pi_mu_hint_ready else 0,
                "Current canon does expose Pi_mu as a gauge-invariant hint.",
            ),
            row(
                "working_action_nontransverse_quadratic_diagonalization_stueckelberg_pack_available",
                "pass" if stueckelberg_full_quadratic_pack_ready else "reject",
                "Stueckelberg full quadratic pack available",
                1 if stueckelberg_full_quadratic_pack_ready else 0,
                "Current canon still exposes the compact full action and gauge-fixing mix pack.",
            ),
            row(
                "working_action_nontransverse_component_quadratic_form_available",
                "reject",
                "working-action nontransverse component quadratic form available",
                0,
                "Current canon does not yet rewrite the nontransverse complement as an explicit component-level quadratic form after the transverse photon is removed.",
            ),
            row(
                "working_action_nontransverse_quadratic_diagonalization_available",
                "reject",
                "working-action nontransverse quadratic diagonalization available",
                0,
                "Without the missing component-level quadratic form, no diagonalization can be claimed.",
            ),
            row(
                "working_action_massive_sector_projector_available_after_nontransverse_audit",
                "reject",
                "working-action massive-sector projector available after nontransverse audit",
                0,
                "The projector remains blocked by the absent nontransverse component quadratic form.",
            ),
        ],
        {
            "pi_mu_gauge_invariant_hint_available": pi_mu_hint_ready,
            "stueckelberg_full_quadratic_pack_available": stueckelberg_full_quadratic_pack_ready,
            "propagator_mix_pack_available": propagator_mix_pack_ready,
            "working_action_nontransverse_component_quadratic_form_available": nontransverse_component_quadratic_form_available,
            "working_action_nontransverse_quadratic_diagonalization_available": nontransverse_quadratic_diagonalization_available,
            "working_action_massive_sector_projector_available": massive_sector_projector_available,
            "identification_nonclosure_reason_or_none": "working_action_nontransverse_component_quadratic_form_absent",
            "first_route_to_close_or_none": "working_action_vector_rebuild_reopen_sixth_gate",
        },
        {
            "overall_status": "working_action_nontransverse_quadratic_diagonalization_identification_nonclosure_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_188": True,
            "next_required_artifacts": ["working_action_vector_rebuild_reopen_sixth_gate"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "part1_full_action_line": hit(part1, "\\mathcal{L}_{P,\\mathrm{full}}"),
            "part1_gauge_fixing_line": hit(part1, "-\\frac{1}{2\\xi_g}\\left(\\partial_\\mu P^\\mu+\\xi_g m_P\\pi\\right)^2"),
            "part1_propagator_massive_line": hit(part1, "\\frac{\\eta_{\\mu\\nu}-k_\\mu k_\\nu/m_P^2}{k^2-m_P^2+i0}"),
            "part1_propagator_gauge_line": hit(part1, "\\frac{\\xi_g\\,k_\\mu k_\\nu/m_P^2}{k^2-\\xi_g m_P^2+i0}"),
            "part1_pi_mu_line": hit(part1, "\\Pi_\\mu:=P_\\mu-\\partial_\\mu\\pi/m_P"),
            "breakthrough_photon_formula": breakthrough_maxwell["summary"]["photon_definition_formula"],
        },
    )

    reopen_gate = payload(
        "8.7.56.188",
        "Working-action vector rebuild reopen sixth gate / Trial-3 fallback sixth refresh",
        common_inputs,
        "Integrate the nontransverse-diagonalization audit and decide whether the vector rebuild can reopen or whether a deeper component-form route must be selected.",
        {
            "reopen_rule": "the vector rebuild reopens only if the current canon freezes a nontransverse component-level quadratic form and its diagonalization",
            "anchor_rule": "anchor refresh remains blocked while the projector and radial eigenoperator still depend on the absent component form",
            "fallback_rule": "Trial-3 remains on fallback hold while the same nontransverse component-form issue stays unresolved",
        },
        [
            row(
                "working_action_vector_rebuild_sixth_gate_source_inventory_ready",
                "pass" if source_inventory["summary"]["working_action_nontransverse_quadratic_diagonalization_source_pack_ready"] else "reject",
                "working-action nontransverse diagonalization source inventory ready at sixth gate",
                1 if source_inventory["summary"]["working_action_nontransverse_quadratic_diagonalization_source_pack_ready"] else 0,
                "The nontransverse diagonalization residual branch has its source pack frozen.",
            ),
            row(
                "working_action_vector_rebuild_sixth_gate_identification_ready",
                "reject",
                "working-action nontransverse diagonalization identification ready at sixth gate",
                0,
                "The nontransverse component quadratic form is still missing.",
            ),
            row(
                "working_action_vector_rebuild_sixth_gate_anchor_refresh_ready",
                "reject",
                "working-action vector anchor refresh ready at sixth gate",
                0,
                "Anchor refresh remains blocked until the projector and radial eigenoperator become current-canon ready.",
            ),
            row(
                "working_action_vector_rebuild_sixth_gate_trial3_hold_retained",
                "pass",
                "Trial-3 fallback hold retained at sixth gate",
                1,
                "The weak-sector branch remains downstream of the unresolved nontransverse component-form route.",
            ),
        ],
        {
            "working_action_vector_rebuild_reopen_ready": False,
            "working_action_nontransverse_quadratic_diagonalization_identification_ready": False,
            "working_action_massive_sector_projector_identification_ready": False,
            "working_action_vector_mass_spectrum_reduced_solver_numeric_ready": False,
            "working_action_vector_mass_spectrum_anchor_refresh_ready": False,
            "trial2_paper_side_sync_deferred_until_vector_anchor_refresh": True,
            "trial3_fallback_hold_retained": True,
            "trial3_fallback_hold_release_ready": False,
            "recommended_next_route_or_none": "8.7.56.189",
        },
        {
            "overall_status": "working_action_vector_rebuild_sixth_gate_nonclosure_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_189": True,
            "next_required_artifacts": ["working_action_nontransverse_component_quadratic_form_identification"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "trial2_declaration_summary": trial2_declaration["summary"],
            "vector_reduced_solver_summary": vector_reduced_solver["summary"],
            "vector_anchor_summary": vector_anchor["summary"],
            "trial3_fallback_route_summary": trial3_fallback_route["summary"],
            "reopen_fifth_summary": reopen_fifth["summary"],
        },
    )

    route_contract = payload(
        "8.7.56.189",
        "Working-action nontransverse component quadratic-form route contract",
        common_inputs,
        "Freeze the deeper residual route suggested by the nontransverse-diagonalization audit: identify the explicit component-level quadratic form of the nontransverse complement after transverse photon extraction.",
        {
            "selected_residual_route": "working_action_nontransverse_component_quadratic_form_identification",
            "blocking_rule": "Pi_mu, the compact action, and the propagator mix hint at the complement sector, but current canon still lacks the explicit component-level quadratic form needed before any diagonalization can be claimed",
            "dependency_rule": "projector, radial eigenoperator, anchor refresh, Trial-2 paper-side sync, and Trial-3 fallback release all remain downstream of the same missing component form",
        },
        [
            row(
                "working_action_nontransverse_component_quadratic_form_route_contract_complete",
                "pass",
                "working-action nontransverse component quadratic-form route contract complete",
                1,
                "The next deeper residual route is frozen after the nontransverse-diagonalization non-closure.",
            ),
            row(
                "working_action_nontransverse_component_quadratic_form_missing",
                "pass",
                "working-action nontransverse component quadratic form missing",
                1,
                "The missing artifact is the explicit component-level quadratic form of the nontransverse complement.",
            ),
            row(
                "working_action_nontransverse_component_quadratic_form_trial3_dependency_blocked",
                "pass",
                "Trial-3 dependency still blocked by nontransverse component quadratic form",
                1,
                "The weak-sector branch remains blocked by the unresolved component-form route.",
            ),
            row(
                "working_action_nontransverse_component_quadratic_form_paper_sync_deferred",
                "pass",
                "Trial-2 paper-side sync still deferred by nontransverse component quadratic form",
                1,
                "Paper-side sync remains deferred until vector anchor refresh becomes current-canon ready.",
            ),
        ],
        {
            "selected_residual_route": "working_action_nontransverse_component_quadratic_form_identification",
            "missing_v2_artifact": "working_action_nontransverse_component_quadratic_form",
            "trial3_dependency_state": "blocked_by_working_action_nontransverse_component_quadratic_form",
            "trial2_paper_side_sync_state": "deferred_until_working_action_vector_anchor_refresh",
            "split_contract_ready": True,
            "recommended_next_route_or_none": "8.7.56.190",
        },
        {
            "overall_status": "working_action_nontransverse_component_quadratic_form_route_contract_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_190": True,
            "next_required_artifacts": [
                "working_action_nontransverse_component_quadratic_form_source_inventory",
                "working_action_nontransverse_component_quadratic_form_identification_audit",
                "working_action_vector_rebuild_reopen_seventh_gate",
            ],
        },
        {
            "reopen_gate_summary": reopen_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "nontransverse_route_summary": nontransverse_route["summary"],
            "temp_basis_audit_summary": temp_basis_audit["summary"],
            "part1_full_action_line": hit(part1, "\\mathcal{L}_{P,\\mathrm{full}}"),
        },
    )

    write_artifact("mass_origin_v2_working_action_nontransverse_quadratic_diagonalization_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_working_action_nontransverse_quadratic_diagonalization_identification_audit", identification_audit)
    write_artifact("mass_origin_v2_vector_rebuild_reopen_sixth_gate", reopen_gate)
    write_artifact("mass_origin_v2_working_action_nontransverse_component_quadratic_form_route_contract", route_contract)

    print("[ok] wrote:")
    print(" - mass_origin_v2_working_action_nontransverse_quadratic_diagonalization_source_inventory_metrics.json")
    print(" - mass_origin_v2_working_action_nontransverse_quadratic_diagonalization_identification_audit_metrics.json")
    print(" - mass_origin_v2_vector_rebuild_reopen_sixth_gate_metrics.json")
    print(" - mass_origin_v2_working_action_nontransverse_component_quadratic_form_route_contract_metrics.json")


# Function: run the working-action nontransverse quadratic-diagonalization branch from the CLI.

if __name__ == "__main__":
    main()
