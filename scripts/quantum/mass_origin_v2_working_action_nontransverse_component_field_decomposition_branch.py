#!/usr/bin/env python3
"""
Generate working-action nontransverse component-field-decomposition artifacts for 8.7.56.194-.197.

This branch deepens the post-breakthrough vector rebuild residual route after
the component-quadratic-form audit. Current canon still exposes the old
three-sector split and the breakthrough photon identification, but it does not
yet map those two statements into an explicit post-photon decomposition of the
remaining nontransverse complement.
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
COMPONENT_SOURCE = OUT / "mass_origin_v2_working_action_nontransverse_component_quadratic_form_source_inventory_metrics.json"
COMPONENT_AUDIT = OUT / "mass_origin_v2_working_action_nontransverse_component_quadratic_form_identification_audit_metrics.json"
REOPEN_SEVENTH = OUT / "mass_origin_v2_vector_rebuild_reopen_seventh_gate_metrics.json"
COMPONENT_ROUTE = OUT / "mass_origin_v2_working_action_nontransverse_component_field_decomposition_route_contract_metrics.json"
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


# Function: execute the working-action nontransverse component-field-decomposition residual branch.

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
        COMPONENT_SOURCE,
        COMPONENT_AUDIT,
        REOPEN_SEVENTH,
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
    component_source = read_json(COMPONENT_SOURCE)
    component_audit = read_json(COMPONENT_AUDIT)
    reopen_seventh = read_json(REOPEN_SEVENTH)
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
        "mass_origin_v2_working_action_nontransverse_component_quadratic_form_source_inventory_json": rel(COMPONENT_SOURCE),
        "mass_origin_v2_working_action_nontransverse_component_quadratic_form_identification_audit_json": rel(COMPONENT_AUDIT),
        "mass_origin_v2_vector_rebuild_reopen_seventh_gate_json": rel(REOPEN_SEVENTH),
        "mass_origin_v2_working_action_nontransverse_component_field_decomposition_route_contract_json": rel(COMPONENT_ROUTE),
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
            "Part I still freezes the compact Stückelberg-completed action that underlies any post-photon mapping.",
        ),
        target_record(
            "part1_pi_mu_definition_line",
            PART1,
            part1,
            "\\Pi_\\mu:=P_\\mu-\\partial_\\mu\\pi/m_P",
            "Part I still defines the gauge-invariant Pi_mu combination.",
        ),
        target_record(
            "part1_gauge_transform_line",
            PART1,
            part1,
            "P_\\mu\\to P_\\mu+\\partial_\\mu\\alpha,",
            "Part I still records the local gauge redundancy paired with the Stückelberg scalar.",
        ),
        target_record(
            "part1_ghost_free_line",
            PART1,
            part1,
            "負ノルム（ghost）モードは出現しない。",
            "Any post-photon mapping must preserve the ghost-free closure.",
        ),
        target_record(
            "status_next_step_anchor",
            STATUS,
            status,
            "current official next step は `8.7.56.194`",
            "STATUS must already expose the component-field-decomposition branch as the next official route.",
        ),
        target_record(
            "roadmap_branch_anchor",
            ROADMAP,
            roadmap,
            "`8.7.56.193-.196`",
            "ROADMAP must already expose the current component-field-decomposition residual branch.",
        ),
    ]
    source_pack_ready = all(item["present"] for item in source_targets)

    old_three_sector_split_ready = bool(component_source["summary"]["old_three_sector_split_ready"])
    photon_definition_ready = bool(component_source["summary"]["photon_definition_ready"])
    breakthrough_transverse_massless_ready = bool(
        component_source["summary"]["breakthrough_transverse_massless_ready"]
    )
    compact_action_ready = bool(component_source["summary"]["compact_full_action_ready"])
    pi_mu_hint_ready = bool(component_source["summary"]["pi_mu_gauge_invariant_hint_ready"])
    ghost_free_guard_ready = bool(component_source["summary"]["ghost_free_guard_ready"])

    source_inventory = payload(
        "8.7.56.194",
        "Working-action nontransverse component-field-decomposition source inventory",
        common_inputs,
        "Freeze the source pack needed to decide whether the current canon maps the old three-sector split and the breakthrough photon branch into an explicit post-photon decomposition of the nontransverse complement.",
        {
            "old_split_rule": "the old VEV pack separates delta P_0, delta P^L, and delta P^T only before the breakthrough reinterpretation",
            "photon_rule": "the breakthrough route identifies A_mu = delta P_mu^T / sqrt(Z_P) as the transverse photon branch",
            "mapping_requirement": "a current-canon post-photon decomposition needs an explicit statement of what remains once the photon branch is removed from the old split",
            "ghost_rule": "any successful mapping must preserve the ghost-free closure already fixed in Part I",
        },
        [
            row(
                "working_action_nontransverse_component_field_decomposition_source_inventory_complete",
                "pass",
                "working-action nontransverse component-field-decomposition source inventory complete",
                1,
                "The source pack for the component-decomposition residual branch is frozen.",
            ),
            row(
                "working_action_nontransverse_component_field_decomposition_source_pack_ready",
                "pass" if source_pack_ready else "reject",
                "working-action nontransverse component-field-decomposition source pack ready",
                1 if source_pack_ready else 0,
                "The residual branch has the required canonical anchors.",
            ),
            row(
                "working_action_nontransverse_component_field_decomposition_old_split_ready",
                "pass" if old_three_sector_split_ready else "reject",
                "old three-sector split ready as mapping source",
                1 if old_three_sector_split_ready else 0,
                "The old split remains available as source material for the post-photon mapping.",
            ),
            row(
                "working_action_nontransverse_component_field_decomposition_photon_branch_ready",
                "pass" if photon_definition_ready and breakthrough_transverse_massless_ready else "reject",
                "photon branch ready before component mapping",
                1 if photon_definition_ready and breakthrough_transverse_massless_ready else 0,
                "The mapping route begins only after the photon branch is frozen.",
            ),
            row(
                "working_action_nontransverse_component_field_decomposition_compact_action_ready",
                "pass" if compact_action_ready else "reject",
                "compact full action ready",
                1 if compact_action_ready else 0,
                "The compact Stückelberg full action remains available as source hint for the nontransverse remainder.",
            ),
            row(
                "working_action_nontransverse_component_field_decomposition_pi_mu_hint_ready",
                "pass" if pi_mu_hint_ready else "reject",
                "Pi_mu hint ready",
                1 if pi_mu_hint_ready else 0,
                "Pi_mu remains visible as a gauge-invariant hint during the mapping route.",
            ),
            row(
                "working_action_nontransverse_component_field_decomposition_ghost_guard_ready",
                "pass" if ghost_free_guard_ready else "reject",
                "ghost-free guard ready",
                1 if ghost_free_guard_ready else 0,
                "Any future mapping must preserve the ghost-free closure.",
            ),
        ],
        {
            "working_action_nontransverse_component_field_decomposition_source_pack_ready": source_pack_ready,
            "old_three_sector_split_ready": old_three_sector_split_ready,
            "photon_definition_ready": photon_definition_ready,
            "breakthrough_transverse_massless_ready": breakthrough_transverse_massless_ready,
            "compact_full_action_ready": compact_action_ready,
            "pi_mu_gauge_invariant_hint_ready": pi_mu_hint_ready,
            "ghost_free_guard_ready": ghost_free_guard_ready,
            "first_route_to_close_or_none": "working_action_nontransverse_component_field_decomposition_identification_audit",
        },
        {
            "overall_status": "working_action_nontransverse_component_field_decomposition_source_inventory_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_195": True,
            "next_required_artifacts": ["working_action_nontransverse_component_field_decomposition_identification_audit"],
        },
        {
            "inventory_targets": source_targets,
            "vev_quadratic_summary": vev_quadratic["summary"],
            "breakthrough_vev_summary": breakthrough_vev["summary"],
            "breakthrough_maxwell_summary": breakthrough_maxwell["summary"],
            "component_source_summary": component_source["summary"],
            "component_audit_summary": component_audit["summary"],
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    post_photon_component_mapping_available = False
    component_field_decomposition_available = False
    component_quadratic_form_available = False
    massive_sector_projector_available = False

    identification_audit = payload(
        "8.7.56.195",
        "Working-action nontransverse component-field-decomposition identification audit",
        common_inputs,
        "Audit whether the current canon actually maps the old three-sector split into an explicit post-photon decomposition of the remaining nontransverse complement.",
        {
            "mapping_rule": "a successful route needs an explicit statement of how the old delta P_0 and delta P^L pieces, together with the Stückelberg scalar, remain after the photon branch delta P^T is promoted to A_mu",
            "hint_rule": "the old split and Pi_mu are source hints only until a post-photon mapping statement is written in the current canon",
            "projector_rule": "without that mapping, no component decomposition, quadratic form, or projector can be claimed",
        },
        [
            row(
                "working_action_nontransverse_component_field_decomposition_old_split_available",
                "pass" if old_three_sector_split_ready else "reject",
                "old three-sector split available as source hint",
                1 if old_three_sector_split_ready else 0,
                "The old VEV split remains available as hint-level evidence.",
            ),
            row(
                "working_action_nontransverse_component_field_decomposition_photon_definition_available",
                "pass" if photon_definition_ready else "reject",
                "photon definition available before mapping audit",
                1 if photon_definition_ready else 0,
                "The transverse photon branch is already frozen under the breakthrough route.",
            ),
            row(
                "working_action_post_photon_nontransverse_component_mapping_available",
                "reject",
                "working-action post-photon nontransverse component mapping available",
                0,
                "Current canon does not yet state how the nontransverse remainder is mapped after the photon branch is removed from the old split.",
            ),
            row(
                "working_action_nontransverse_component_field_decomposition_available",
                "reject",
                "working-action nontransverse component field decomposition available",
                0,
                "Without the missing post-photon mapping, no explicit component decomposition can be claimed.",
            ),
            row(
                "working_action_nontransverse_component_quadratic_form_available_after_mapping_audit",
                "reject",
                "working-action nontransverse component quadratic form available after mapping audit",
                0,
                "The component quadratic form remains blocked by the absent post-photon mapping.",
            ),
        ],
        {
            "old_three_sector_split_available_as_source_hint": old_three_sector_split_ready,
            "photon_definition_available": photon_definition_ready,
            "pi_mu_gauge_invariant_hint_available": pi_mu_hint_ready,
            "working_action_post_photon_nontransverse_component_mapping_available": post_photon_component_mapping_available,
            "working_action_nontransverse_component_field_decomposition_available": component_field_decomposition_available,
            "working_action_nontransverse_component_quadratic_form_available": component_quadratic_form_available,
            "working_action_massive_sector_projector_available": massive_sector_projector_available,
            "identification_nonclosure_reason_or_none": "working_action_post_photon_nontransverse_component_mapping_absent",
            "first_route_to_close_or_none": "working_action_vector_rebuild_reopen_eighth_gate",
        },
        {
            "overall_status": "working_action_nontransverse_component_field_decomposition_identification_nonclosure_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_196": True,
            "next_required_artifacts": ["working_action_vector_rebuild_reopen_eighth_gate"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "vev_quadratic_formulas": vev_quadratic["formulas"],
            "breakthrough_vev_formulas": breakthrough_vev["formulas"],
            "breakthrough_photon_formula": breakthrough_maxwell["summary"]["photon_definition_formula"],
            "part1_full_action_line": hit(part1, "\\mathcal{L}_{P,\\mathrm{full}}"),
            "part1_pi_mu_line": hit(part1, "\\Pi_\\mu:=P_\\mu-\\partial_\\mu\\pi/m_P"),
        },
    )

    reopen_gate = payload(
        "8.7.56.196",
        "Working-action vector rebuild reopen eighth gate / Trial-3 fallback eighth refresh",
        common_inputs,
        "Integrate the component-decomposition audit and decide whether the vector rebuild can reopen or whether a deeper post-photon mapping route must be selected.",
        {
            "reopen_rule": "the vector rebuild reopens only if the current canon freezes an explicit post-photon mapping of the nontransverse complement",
            "anchor_rule": "anchor refresh remains blocked while projector and radial eigenoperator still depend on the absent post-photon mapping",
            "fallback_rule": "Trial-3 remains on fallback hold while the same mapping issue stays unresolved",
        },
        [
            row(
                "working_action_vector_rebuild_eighth_gate_source_inventory_ready",
                "pass" if source_inventory["summary"]["working_action_nontransverse_component_field_decomposition_source_pack_ready"] else "reject",
                "working-action component-field-decomposition source inventory ready at eighth gate",
                1 if source_inventory["summary"]["working_action_nontransverse_component_field_decomposition_source_pack_ready"] else 0,
                "The component-decomposition residual branch has its source pack frozen.",
            ),
            row(
                "working_action_vector_rebuild_eighth_gate_identification_ready",
                "reject",
                "working-action component-field-decomposition identification ready at eighth gate",
                0,
                "The explicit post-photon mapping is still missing.",
            ),
            row(
                "working_action_vector_rebuild_eighth_gate_anchor_refresh_ready",
                "reject",
                "working-action vector anchor refresh ready at eighth gate",
                0,
                "Anchor refresh remains blocked until the projector and radial eigenoperator become current-canon ready.",
            ),
            row(
                "working_action_vector_rebuild_eighth_gate_trial3_hold_retained",
                "pass",
                "Trial-3 fallback hold retained at eighth gate",
                1,
                "The weak-sector branch remains downstream of the unresolved post-photon mapping route.",
            ),
        ],
        {
            "working_action_vector_rebuild_reopen_ready": False,
            "working_action_nontransverse_component_field_decomposition_identification_ready": False,
            "working_action_massive_sector_projector_identification_ready": False,
            "working_action_vector_mass_spectrum_reduced_solver_numeric_ready": False,
            "working_action_vector_mass_spectrum_anchor_refresh_ready": False,
            "trial2_paper_side_sync_deferred_until_vector_anchor_refresh": True,
            "trial3_fallback_hold_retained": True,
            "trial3_fallback_hold_release_ready": False,
            "recommended_next_route_or_none": "8.7.56.197",
        },
        {
            "overall_status": "working_action_vector_rebuild_eighth_gate_nonclosure_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_197": True,
            "next_required_artifacts": ["working_action_post_photon_nontransverse_component_mapping_identification"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "trial2_declaration_summary": trial2_declaration["summary"],
            "vector_reduced_solver_summary": vector_reduced_solver["summary"],
            "vector_anchor_summary": vector_anchor["summary"],
            "trial3_fallback_route_summary": trial3_fallback_route["summary"],
            "reopen_seventh_summary": reopen_seventh["summary"],
        },
    )

    route_contract = payload(
        "8.7.56.197",
        "Working-action post-photon nontransverse component-mapping route contract",
        common_inputs,
        "Freeze the deeper residual route suggested by the component-decomposition audit: identify the explicit post-photon mapping from the old split to the remaining nontransverse complement.",
        {
            "selected_residual_route": "working_action_post_photon_nontransverse_component_mapping_identification",
            "blocking_rule": "the old three-sector split and breakthrough photon formula are both present, but current canon still lacks the explicit statement that maps what remains after the photon branch is removed",
            "dependency_rule": "component decomposition, component quadratic form, projector, radial eigenoperator, anchor refresh, Trial-2 paper-side sync, and Trial-3 fallback release all remain downstream of the same missing post-photon mapping",
        },
        [
            row(
                "working_action_post_photon_nontransverse_component_mapping_route_contract_complete",
                "pass",
                "working-action post-photon nontransverse component-mapping route contract complete",
                1,
                "The next deeper residual route is frozen after the component-decomposition non-closure.",
            ),
            row(
                "working_action_post_photon_nontransverse_component_mapping_missing",
                "pass",
                "working-action post-photon nontransverse component mapping missing",
                1,
                "The missing artifact is the explicit mapping of the nontransverse remainder after the photon branch is removed.",
            ),
            row(
                "working_action_post_photon_nontransverse_component_mapping_trial3_dependency_blocked",
                "pass",
                "Trial-3 dependency still blocked by post-photon nontransverse component mapping",
                1,
                "The weak-sector branch remains blocked by the unresolved post-photon mapping route.",
            ),
            row(
                "working_action_post_photon_nontransverse_component_mapping_paper_sync_deferred",
                "pass",
                "Trial-2 paper-side sync still deferred by post-photon nontransverse component mapping",
                1,
                "Paper-side sync remains deferred until vector anchor refresh becomes current-canon ready.",
            ),
        ],
        {
            "selected_residual_route": "working_action_post_photon_nontransverse_component_mapping_identification",
            "missing_v2_artifact": "working_action_post_photon_nontransverse_component_mapping",
            "trial3_dependency_state": "blocked_by_working_action_post_photon_nontransverse_component_mapping",
            "trial2_paper_side_sync_state": "deferred_until_working_action_vector_anchor_refresh",
            "split_contract_ready": True,
            "recommended_next_route_or_none": "8.7.56.198",
        },
        {
            "overall_status": "working_action_post_photon_nontransverse_component_mapping_route_contract_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_198": True,
            "next_required_artifacts": [
                "working_action_post_photon_nontransverse_component_mapping_source_inventory",
                "working_action_post_photon_nontransverse_component_mapping_identification_audit",
                "working_action_vector_rebuild_reopen_ninth_gate",
            ],
        },
        {
            "reopen_gate_summary": reopen_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "component_route_summary": component_route["summary"],
            "part1_full_action_line": hit(part1, "\\mathcal{L}_{P,\\mathrm{full}}"),
            "breakthrough_photon_formula": breakthrough_maxwell["summary"]["photon_definition_formula"],
        },
    )

    write_artifact("mass_origin_v2_working_action_nontransverse_component_field_decomposition_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_working_action_nontransverse_component_field_decomposition_identification_audit", identification_audit)
    write_artifact("mass_origin_v2_vector_rebuild_reopen_eighth_gate", reopen_gate)
    write_artifact("mass_origin_v2_working_action_post_photon_nontransverse_component_mapping_route_contract", route_contract)

    print("[ok] wrote:")
    print(" - mass_origin_v2_working_action_nontransverse_component_field_decomposition_source_inventory_metrics.json")
    print(" - mass_origin_v2_working_action_nontransverse_component_field_decomposition_identification_audit_metrics.json")
    print(" - mass_origin_v2_vector_rebuild_reopen_eighth_gate_metrics.json")
    print(" - mass_origin_v2_working_action_post_photon_nontransverse_component_mapping_route_contract_metrics.json")


# Function: run the working-action nontransverse component-field-decomposition branch from the CLI.

if __name__ == "__main__":
    main()
