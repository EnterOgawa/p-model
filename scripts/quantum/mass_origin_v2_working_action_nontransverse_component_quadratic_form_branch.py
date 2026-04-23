#!/usr/bin/env python3
"""
Generate working-action nontransverse component-quadratic-form artifacts for 8.7.56.190-.193.

This branch deepens the post-breakthrough vector rebuild residual route after
the nontransverse quadratic-diagonalization audit. Current canon still exposes
the old three-sector split and the compact Stückelberg-completed action, but it
does not yet restate the nontransverse complement as an explicit component
decomposition and quadratic form under the breakthrough working action.
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

VEV_QUADRATIC = OUT / "mass_origin_v2_vev_quadratic_mode_decomposition_metrics.json"
TRIAL1_BREAKTHROUGH_VEV = OUT / "mass_origin_v2_trial1_breakthrough_modified_vev_decomposition_metrics.json"
TRIAL1_BREAKTHROUGH_MAXWELL = OUT / "mass_origin_v2_trial1_breakthrough_maxwell_coupling_audit_metrics.json"
TRIAL2_DECLARATION = OUT / "mass_origin_v2_trial2_declaration_gate_metrics.json"
TEMP_BASIS_SOURCE = OUT / "mass_origin_v2_working_action_temporal_longitudinal_stueckelberg_basis_source_inventory_metrics.json"
TEMP_BASIS_AUDIT = OUT / "mass_origin_v2_working_action_temporal_longitudinal_stueckelberg_basis_identification_audit_metrics.json"
NONTRANSVERSE_SOURCE = OUT / "mass_origin_v2_working_action_nontransverse_quadratic_diagonalization_source_inventory_metrics.json"
NONTRANSVERSE_AUDIT = OUT / "mass_origin_v2_working_action_nontransverse_quadratic_diagonalization_identification_audit_metrics.json"
REOPEN_SIXTH = OUT / "mass_origin_v2_vector_rebuild_reopen_sixth_gate_metrics.json"
COMPONENT_ROUTE = OUT / "mass_origin_v2_working_action_nontransverse_component_quadratic_form_route_contract_metrics.json"
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


# Function: execute the working-action nontransverse component-quadratic-form residual branch.

def main() -> None:
    for path in (
        PART1,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        VEV_QUADRATIC,
        TRIAL1_BREAKTHROUGH_VEV,
        TRIAL1_BREAKTHROUGH_MAXWELL,
        TRIAL2_DECLARATION,
        TEMP_BASIS_SOURCE,
        TEMP_BASIS_AUDIT,
        NONTRANSVERSE_SOURCE,
        NONTRANSVERSE_AUDIT,
        REOPEN_SIXTH,
        COMPONENT_ROUTE,
        VECTOR_REDUCED_SOLVER,
        VECTOR_ANCHOR,
        TRIAL3_FALLBACK_ROUTE,
    ):
        req(path)

    part1 = read_text(PART1)
    status = read_text(STATUS)
    roadmap = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)

    vev_quadratic = read_json(VEV_QUADRATIC)
    breakthrough_vev = read_json(TRIAL1_BREAKTHROUGH_VEV)
    breakthrough_maxwell = read_json(TRIAL1_BREAKTHROUGH_MAXWELL)
    trial2_declaration = read_json(TRIAL2_DECLARATION)
    temp_basis_source = read_json(TEMP_BASIS_SOURCE)
    temp_basis_audit = read_json(TEMP_BASIS_AUDIT)
    nontransverse_source = read_json(NONTRANSVERSE_SOURCE)
    nontransverse_audit = read_json(NONTRANSVERSE_AUDIT)
    reopen_sixth = read_json(REOPEN_SIXTH)
    component_route = read_json(COMPONENT_ROUTE)
    vector_reduced_solver = read_json(VECTOR_REDUCED_SOLVER)
    vector_anchor = read_json(VECTOR_ANCHOR)
    trial3_fallback_route = read_json(TRIAL3_FALLBACK_ROUTE)

    common_inputs = {
        "part1_markdown": rel(PART1),
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_vev_quadratic_mode_decomposition_json": rel(VEV_QUADRATIC),
        "mass_origin_v2_trial1_breakthrough_modified_vev_decomposition_json": rel(TRIAL1_BREAKTHROUGH_VEV),
        "mass_origin_v2_trial1_breakthrough_maxwell_coupling_audit_json": rel(TRIAL1_BREAKTHROUGH_MAXWELL),
        "mass_origin_v2_trial2_declaration_gate_json": rel(TRIAL2_DECLARATION),
        "mass_origin_v2_working_action_temporal_longitudinal_stueckelberg_basis_source_inventory_json": rel(TEMP_BASIS_SOURCE),
        "mass_origin_v2_working_action_temporal_longitudinal_stueckelberg_basis_identification_audit_json": rel(TEMP_BASIS_AUDIT),
        "mass_origin_v2_working_action_nontransverse_quadratic_diagonalization_source_inventory_json": rel(NONTRANSVERSE_SOURCE),
        "mass_origin_v2_working_action_nontransverse_quadratic_diagonalization_identification_audit_json": rel(NONTRANSVERSE_AUDIT),
        "mass_origin_v2_vector_rebuild_reopen_sixth_gate_json": rel(REOPEN_SIXTH),
        "mass_origin_v2_working_action_nontransverse_component_quadratic_form_route_contract_json": rel(COMPONENT_ROUTE),
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
            "Part I still freezes the compact Stückelberg-completed full action.",
        ),
        target_record(
            "part1_stueckelberg_mass_line",
            PART1,
            part1,
            "+\\frac{m_P^2}{2}\\left(P_\\mu-\\frac{1}{m_P}\\partial_\\mu\\pi\\right)",
            "The compact mass source still exposes the unresolved nontransverse mix pack.",
        ),
        target_record(
            "part1_gauge_fixing_line",
            PART1,
            part1,
            "-\\frac{1}{2\\xi_g}\\left(\\partial_\\mu P^\\mu+\\xi_g m_P\\pi\\right)^2",
            "The gauge-fixing term still exposes the temporal/longitudinal/Stueckelberg mix.",
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
            "Any component-form reconstruction must preserve the ghost-free closure.",
        ),
        target_record(
            "status_next_step_anchor",
            STATUS,
            status,
            "current official next step は `8.7.56.190`",
            "STATUS must already expose the component-quadratic-form branch as the next official route.",
        ),
        target_record(
            "roadmap_branch_anchor",
            ROADMAP,
            roadmap,
            "`8.7.56.189-.192`",
            "ROADMAP must already expose the current component-quadratic-form residual branch.",
        ),
    ]
    source_pack_ready = all(item["present"] for item in source_targets)

    old_three_sector_split_ready = bool(vev_quadratic["summary"]["three_sector_split_available"])
    old_transverse_decoupling_ready = bool(
        vev_quadratic["summary"]["transverse_sector_decoupled_at_quadratic_order"]
    )
    breakthrough_transverse_massless_ready = bool(
        breakthrough_vev["summary"]["transverse_mode_massless_under_breakthrough_action"]
    )
    photon_definition_ready = bool(
        breakthrough_maxwell["summary"]["transverse_maxwell_reduction_available"]
    )
    pi_mu_hint_ready = source_targets[3]["present"]
    compact_action_ready = (
        source_targets[0]["present"] and source_targets[1]["present"] and source_targets[2]["present"]
    )
    ghost_free_guard_ready = source_targets[4]["present"]

    source_inventory = payload(
        "8.7.56.190",
        "Working-action nontransverse component quadratic-form source inventory",
        common_inputs,
        "Freeze the source pack needed to decide whether the current canon restates the nontransverse complement as an explicit component-field decomposition and quadratic form under the breakthrough working action.",
        {
            "old_split_rule": "the old VEV quadratic pack already separates delta P_0, delta P^L, and delta P^T at quadratic order",
            "breakthrough_rule": "the breakthrough working action promotes delta P^T to the photon branch and leaves the nontransverse complement to be reinterpreted",
            "component_form_requirement": "a current-canon component quadratic form needs an explicit post-photon decomposition of the remaining fields, not only the compact full action",
            "ghost_rule": "any successful component reconstruction must preserve the ghost-free closure already fixed in Part I",
        },
        [
            row(
                "working_action_nontransverse_component_quadratic_form_source_inventory_complete",
                "pass",
                "working-action nontransverse component quadratic-form source inventory complete",
                1,
                "The source pack for the component-form residual branch is frozen.",
            ),
            row(
                "working_action_nontransverse_component_quadratic_form_source_pack_ready",
                "pass" if source_pack_ready else "reject",
                "working-action nontransverse component quadratic-form source pack ready",
                1 if source_pack_ready else 0,
                "The residual branch has the required canonical anchors.",
            ),
            row(
                "working_action_nontransverse_component_old_three_sector_split_ready",
                "pass" if old_three_sector_split_ready and old_transverse_decoupling_ready else "reject",
                "old three-sector split ready",
                1 if old_three_sector_split_ready and old_transverse_decoupling_ready else 0,
                "The old VEV quadratic pack still exposes the pre-breakthrough component split as a source hint.",
            ),
            row(
                "working_action_nontransverse_component_breakthrough_photon_ready",
                "pass" if breakthrough_transverse_massless_ready and photon_definition_ready else "reject",
                "breakthrough photon branch ready before component-form inventory",
                1 if breakthrough_transverse_massless_ready and photon_definition_ready else 0,
                "The component-form route starts only after the transverse photon branch is frozen.",
            ),
            row(
                "working_action_nontransverse_component_compact_action_ready",
                "pass" if compact_action_ready else "reject",
                "compact full action ready",
                1 if compact_action_ready else 0,
                "The compact Stückelberg full action remains the starting source hint for the unresolved nontransverse complement.",
            ),
            row(
                "working_action_nontransverse_component_pi_mu_hint_ready",
                "pass" if pi_mu_hint_ready else "reject",
                "Pi_mu hint ready",
                1 if pi_mu_hint_ready else 0,
                "The gauge-invariant Pi_mu combination remains visible as a source hint.",
            ),
            row(
                "working_action_nontransverse_component_ghost_guard_ready",
                "pass" if ghost_free_guard_ready else "reject",
                "ghost-free guard ready",
                1 if ghost_free_guard_ready else 0,
                "Any future component-form reconstruction must preserve ghost-free closure.",
            ),
        ],
        {
            "working_action_nontransverse_component_quadratic_form_source_pack_ready": source_pack_ready,
            "old_three_sector_split_ready": old_three_sector_split_ready,
            "old_transverse_decoupling_ready": old_transverse_decoupling_ready,
            "breakthrough_transverse_massless_ready": breakthrough_transverse_massless_ready,
            "photon_definition_ready": photon_definition_ready,
            "compact_full_action_ready": compact_action_ready,
            "pi_mu_gauge_invariant_hint_ready": pi_mu_hint_ready,
            "ghost_free_guard_ready": ghost_free_guard_ready,
            "first_route_to_close_or_none": "working_action_nontransverse_component_quadratic_form_identification_audit",
        },
        {
            "overall_status": "working_action_nontransverse_component_quadratic_form_source_inventory_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_191": True,
            "next_required_artifacts": ["working_action_nontransverse_component_quadratic_form_identification_audit"],
        },
        {
            "inventory_targets": source_targets,
            "vev_quadratic_summary": vev_quadratic["summary"],
            "breakthrough_vev_summary": breakthrough_vev["summary"],
            "breakthrough_maxwell_summary": breakthrough_maxwell["summary"],
            "temp_basis_audit_summary": temp_basis_audit["summary"],
            "nontransverse_audit_summary": nontransverse_audit["summary"],
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    component_field_decomposition_available = False
    component_quadratic_form_available = False
    nontransverse_quadratic_diagonalization_available = False
    massive_sector_projector_available = False

    identification_audit = payload(
        "8.7.56.191",
        "Working-action nontransverse component quadratic-form identification audit",
        common_inputs,
        "Audit whether the current canon actually restates the nontransverse complement as an explicit component-field decomposition and quadratic form after the transverse photon branch is removed.",
        {
            "old_split_rule": "the old VEV split is only a source hint and does not by itself become the breakthrough working-action component basis",
            "component_decomposition_rule": "a current-canon component quadratic form first needs an explicit post-photon decomposition of the nontransverse complement fields",
            "projector_rule": "without that decomposition and its quadratic form, no projector or radial eigenoperator can be claimed",
        },
        [
            row(
                "working_action_nontransverse_component_old_split_available",
                "pass" if old_three_sector_split_ready else "reject",
                "old three-sector split available as source hint",
                1 if old_three_sector_split_ready else 0,
                "The old VEV decomposition remains available only as hint-level evidence.",
            ),
            row(
                "working_action_nontransverse_component_pi_mu_hint_available",
                "pass" if pi_mu_hint_ready else "reject",
                "Pi_mu gauge-invariant hint available",
                1 if pi_mu_hint_ready else 0,
                "Current canon still exposes Pi_mu as a gauge-invariant hint.",
            ),
            row(
                "working_action_nontransverse_component_field_decomposition_available",
                "reject",
                "working-action nontransverse component field decomposition available",
                0,
                "Current canon does not yet restate the post-photon complement as an explicit component decomposition of the remaining fields.",
            ),
            row(
                "working_action_nontransverse_component_quadratic_form_available",
                "reject",
                "working-action nontransverse component quadratic form available",
                0,
                "Without the missing component decomposition, no explicit working-action quadratic form can be written.",
            ),
            row(
                "working_action_nontransverse_quadratic_diagonalization_available_after_component_audit",
                "reject",
                "working-action nontransverse quadratic diagonalization available after component audit",
                0,
                "Diagonalization remains blocked by the absent component decomposition and quadratic form.",
            ),
        ],
        {
            "old_three_sector_split_available_as_source_hint": old_three_sector_split_ready,
            "pi_mu_gauge_invariant_hint_available": pi_mu_hint_ready,
            "working_action_nontransverse_component_field_decomposition_available": component_field_decomposition_available,
            "working_action_nontransverse_component_quadratic_form_available": component_quadratic_form_available,
            "working_action_nontransverse_quadratic_diagonalization_available": nontransverse_quadratic_diagonalization_available,
            "working_action_massive_sector_projector_available": massive_sector_projector_available,
            "identification_nonclosure_reason_or_none": "working_action_nontransverse_component_field_decomposition_absent",
            "first_route_to_close_or_none": "working_action_vector_rebuild_reopen_seventh_gate",
        },
        {
            "overall_status": "working_action_nontransverse_component_quadratic_form_identification_nonclosure_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_192": True,
            "next_required_artifacts": ["working_action_vector_rebuild_reopen_seventh_gate"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "vev_quadratic_formulas": vev_quadratic["formulas"],
            "breakthrough_vev_formulas": breakthrough_vev["formulas"],
            "part1_full_action_line": hit(part1, "\\mathcal{L}_{P,\\mathrm{full}}"),
            "part1_pi_mu_line": hit(part1, "\\Pi_\\mu:=P_\\mu-\\partial_\\mu\\pi/m_P"),
            "part1_gauge_fixing_line": hit(part1, "-\\frac{1}{2\\xi_g}\\left(\\partial_\\mu P^\\mu+\\xi_g m_P\\pi\\right)^2"),
        },
    )

    reopen_gate = payload(
        "8.7.56.192",
        "Working-action vector rebuild reopen seventh gate / Trial-3 fallback seventh refresh",
        common_inputs,
        "Integrate the component-form audit and decide whether the vector rebuild can reopen or whether a deeper component-decomposition route must be selected.",
        {
            "reopen_rule": "the vector rebuild reopens only if the current canon freezes a post-photon component decomposition and its quadratic form for the nontransverse complement",
            "anchor_rule": "anchor refresh remains blocked while projector and radial eigenoperator still depend on the absent component decomposition",
            "fallback_rule": "Trial-3 remains on fallback hold while the same component-decomposition issue stays unresolved",
        },
        [
            row(
                "working_action_vector_rebuild_seventh_gate_source_inventory_ready",
                "pass" if source_inventory["summary"]["working_action_nontransverse_component_quadratic_form_source_pack_ready"] else "reject",
                "working-action component-form source inventory ready at seventh gate",
                1 if source_inventory["summary"]["working_action_nontransverse_component_quadratic_form_source_pack_ready"] else 0,
                "The component-form residual branch has its source pack frozen.",
            ),
            row(
                "working_action_vector_rebuild_seventh_gate_identification_ready",
                "reject",
                "working-action component-form identification ready at seventh gate",
                0,
                "The post-photon component decomposition is still missing.",
            ),
            row(
                "working_action_vector_rebuild_seventh_gate_anchor_refresh_ready",
                "reject",
                "working-action vector anchor refresh ready at seventh gate",
                0,
                "Anchor refresh remains blocked until the projector and radial eigenoperator become current-canon ready.",
            ),
            row(
                "working_action_vector_rebuild_seventh_gate_trial3_hold_retained",
                "pass",
                "Trial-3 fallback hold retained at seventh gate",
                1,
                "The weak-sector branch remains downstream of the unresolved component-decomposition route.",
            ),
        ],
        {
            "working_action_vector_rebuild_reopen_ready": False,
            "working_action_nontransverse_component_quadratic_form_identification_ready": False,
            "working_action_massive_sector_projector_identification_ready": False,
            "working_action_vector_mass_spectrum_reduced_solver_numeric_ready": False,
            "working_action_vector_mass_spectrum_anchor_refresh_ready": False,
            "trial2_paper_side_sync_deferred_until_vector_anchor_refresh": True,
            "trial3_fallback_hold_retained": True,
            "trial3_fallback_hold_release_ready": False,
            "recommended_next_route_or_none": "8.7.56.193",
        },
        {
            "overall_status": "working_action_vector_rebuild_seventh_gate_nonclosure_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_193": True,
            "next_required_artifacts": ["working_action_nontransverse_component_field_decomposition_identification"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "trial2_declaration_summary": trial2_declaration["summary"],
            "vector_reduced_solver_summary": vector_reduced_solver["summary"],
            "vector_anchor_summary": vector_anchor["summary"],
            "trial3_fallback_route_summary": trial3_fallback_route["summary"],
            "reopen_sixth_summary": reopen_sixth["summary"],
        },
    )

    route_contract = payload(
        "8.7.56.193",
        "Working-action nontransverse component-field-decomposition route contract",
        common_inputs,
        "Freeze the deeper residual route suggested by the component-form audit: identify the explicit post-photon component decomposition of the nontransverse complement before any quadratic form or diagonalization can be claimed.",
        {
            "selected_residual_route": "working_action_nontransverse_component_field_decomposition_identification",
            "blocking_rule": "the old three-sector split and compact action remain hint-level evidence, but current canon still lacks the explicit post-photon component decomposition needed before a working-action quadratic form can be written",
            "dependency_rule": "quadratic form, projector, radial eigenoperator, anchor refresh, Trial-2 paper-side sync, and Trial-3 fallback release all remain downstream of the same missing component decomposition",
        },
        [
            row(
                "working_action_nontransverse_component_field_decomposition_route_contract_complete",
                "pass",
                "working-action nontransverse component-field-decomposition route contract complete",
                1,
                "The next deeper residual route is frozen after the component-form non-closure.",
            ),
            row(
                "working_action_nontransverse_component_field_decomposition_missing",
                "pass",
                "working-action nontransverse component field decomposition missing",
                1,
                "The missing artifact is the explicit post-photon component decomposition of the nontransverse complement.",
            ),
            row(
                "working_action_nontransverse_component_field_decomposition_trial3_dependency_blocked",
                "pass",
                "Trial-3 dependency still blocked by nontransverse component field decomposition",
                1,
                "The weak-sector branch remains blocked by the unresolved component-decomposition route.",
            ),
            row(
                "working_action_nontransverse_component_field_decomposition_paper_sync_deferred",
                "pass",
                "Trial-2 paper-side sync still deferred by nontransverse component field decomposition",
                1,
                "Paper-side sync remains deferred until vector anchor refresh becomes current-canon ready.",
            ),
        ],
        {
            "selected_residual_route": "working_action_nontransverse_component_field_decomposition_identification",
            "missing_v2_artifact": "working_action_nontransverse_component_field_decomposition",
            "trial3_dependency_state": "blocked_by_working_action_nontransverse_component_field_decomposition",
            "trial2_paper_side_sync_state": "deferred_until_working_action_vector_anchor_refresh",
            "split_contract_ready": True,
            "recommended_next_route_or_none": "8.7.56.194",
        },
        {
            "overall_status": "working_action_nontransverse_component_field_decomposition_route_contract_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_194": True,
            "next_required_artifacts": [
                "working_action_nontransverse_component_field_decomposition_source_inventory",
                "working_action_nontransverse_component_field_decomposition_identification_audit",
                "working_action_vector_rebuild_reopen_eighth_gate",
            ],
        },
        {
            "reopen_gate_summary": reopen_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "component_route_summary": component_route["summary"],
            "nontransverse_audit_summary": nontransverse_audit["summary"],
            "part1_full_action_line": hit(part1, "\\mathcal{L}_{P,\\mathrm{full}}"),
        },
    )

    write_artifact("mass_origin_v2_working_action_nontransverse_component_quadratic_form_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_working_action_nontransverse_component_quadratic_form_identification_audit", identification_audit)
    write_artifact("mass_origin_v2_vector_rebuild_reopen_seventh_gate", reopen_gate)
    write_artifact("mass_origin_v2_working_action_nontransverse_component_field_decomposition_route_contract", route_contract)

    print("[ok] wrote:")
    print(" - mass_origin_v2_working_action_nontransverse_component_quadratic_form_source_inventory_metrics.json")
    print(" - mass_origin_v2_working_action_nontransverse_component_quadratic_form_identification_audit_metrics.json")
    print(" - mass_origin_v2_vector_rebuild_reopen_seventh_gate_metrics.json")
    print(" - mass_origin_v2_working_action_nontransverse_component_field_decomposition_route_contract_metrics.json")


# Function: run the working-action nontransverse component-quadratic-form branch from the CLI.

if __name__ == "__main__":
    main()
