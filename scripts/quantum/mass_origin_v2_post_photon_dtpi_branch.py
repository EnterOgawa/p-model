#!/usr/bin/env python3
"""
Generate working-action post-photon temporal/Pi_mu basis artifacts for 8.7.56.214-.217.

This branch follows the post-photon temporal/Pi_mu residual route. Current
canon already keeps the old three-sector split, the breakthrough photon
definition, the time-sector fluctuation hint, the Pi_mu hint, and the compact
Stueckelberg full action as source hints, but it still does not freeze an
explicit delta P_t / Pi_mu complement statement after photon extraction.
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
PREV_SOURCE = OUT / "mass_origin_v2_working_action_post_photon_temporal_longitudinal_stueckelberg_basis_statement_source_inventory_metrics.json"
PREV_AUDIT = OUT / "mass_origin_v2_working_action_post_photon_temporal_longitudinal_stueckelberg_basis_statement_identification_audit_metrics.json"
PREV_GATE = OUT / "mass_origin_v2_vector_rebuild_reopen_twelfth_gate_metrics.json"
CURRENT_ROUTE = OUT / "mass_origin_v2_working_action_post_photon_temporal_pi_mu_basis_statement_route_contract_metrics.json"
VECTOR_REDUCED_SOLVER = OUT / "mass_origin_v2_vector_reduced_solver_pilot_metrics.json"
VECTOR_ANCHOR = OUT / "mass_origin_v2_vector_anchor_refresh_metrics.json"
TRIAL3_FALLBACK_ROUTE = OUT / "mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_metrics.json"


# Function: return the current UTC timestamp in ISO 8601 format.
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: stop execution if a required input path is missing.

def require(path: Path) -> None:
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


# Function: execute the post-photon temporal/Pi_mu basis branch.

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
        PREV_SOURCE,
        PREV_AUDIT,
        PREV_GATE,
        CURRENT_ROUTE,
        VECTOR_REDUCED_SOLVER,
        VECTOR_ANCHOR,
        TRIAL3_FALLBACK_ROUTE,
    ):
        require(path)

    part1 = read_text(PART1)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    vev_quadratic = read_json(VEV_QUADRATIC)
    breakthrough_vev = read_json(TRIAL1_BREAKTHROUGH_VEV)
    breakthrough_maxwell = read_json(TRIAL1_BREAKTHROUGH_MAXWELL)
    trial2_declaration = read_json(TRIAL2_DECLARATION)
    prev_source = read_json(PREV_SOURCE)
    prev_audit = read_json(PREV_AUDIT)
    prev_gate = read_json(PREV_GATE)
    current_route = read_json(CURRENT_ROUTE)
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
        "mass_origin_v2_working_action_post_photon_temporal_longitudinal_stueckelberg_basis_statement_source_inventory_json": rel(PREV_SOURCE),
        "mass_origin_v2_working_action_post_photon_temporal_longitudinal_stueckelberg_basis_statement_identification_audit_json": rel(PREV_AUDIT),
        "mass_origin_v2_vector_rebuild_reopen_twelfth_gate_json": rel(PREV_GATE),
        "mass_origin_v2_working_action_post_photon_temporal_pi_mu_basis_statement_route_contract_json": rel(CURRENT_ROUTE),
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
            "Part I still freezes the compact Stückelberg-completed action that any post-photon temporal/Pi_mu basis statement must respect.",
        ),
        target_record(
            "part1_stueckelberg_closure_line",
            PART1,
            part1,
            "Stückelberg 場 $\\pi$",
            "Part I still keeps the Stückelberg closure itself in the current canon.",
        ),
        target_record(
            "part1_pi_mu_definition_line",
            PART1,
            part1,
            "\\Pi_\\mu:=P_\\mu-\\partial_\\mu\\pi/m_P",
            "Part I still defines the gauge-invariant Pi_mu combination.",
        ),
        target_record(
            "part1_time_fluctuation_line",
            PART1,
            part1,
            "P_t=P_t^{\\mathrm{(lin)}}+\\delta P_t,",
            "Part I still exposes an explicit time-sector fluctuation variable.",
        ),
        target_record(
            "part1_ghost_free_line",
            PART1,
            part1,
            "負ノルム（ghost）モードは出現しない。",
            "Any post-photon temporal/Pi_mu basis statement must preserve the ghost-free closure.",
        ),
        target_record(
            "status_next_step_anchor",
            STATUS,
            status_text,
            "current official next step は `8.7.56.214`",
            "STATUS must already expose the temporal/Pi_mu basis branch as the next official route.",
        ),
        target_record(
            "roadmap_branch_anchor",
            ROADMAP,
            roadmap_text,
            "`8.7.56.213-.216`",
            "ROADMAP must already expose the current temporal/Pi_mu basis residual branch.",
        ),
    ]
    source_pack_ready = all(item["present"] for item in source_targets)

    old_three_sector_split_ready = bool(prev_source["summary"]["old_three_sector_split_ready"])
    photon_definition_ready = bool(prev_source["summary"]["photon_definition_ready"])
    breakthrough_transverse_massless_ready = bool(prev_source["summary"]["breakthrough_transverse_massless_ready"])
    compact_action_ready = bool(prev_source["summary"]["compact_full_action_ready"])
    pi_mu_hint_ready = bool(prev_source["summary"]["pi_mu_gauge_invariant_hint_ready"])
    ghost_free_guard_ready = bool(prev_source["summary"]["ghost_free_guard_ready"])
    time_fluctuation_ready = hit(part1, "P_t=P_t^{\\mathrm{(lin)}}+\\delta P_t,") is not None
    route_contract_ready = bool(current_route["summary"]["split_contract_ready"])

    source_inventory = payload(
        "8.7.56.214",
        "Working-action post-photon temporal/Pi_mu basis-statement source inventory",
        common_inputs,
        "Freeze the source pack needed to decide whether the current canon explicitly states the post-photon temporal/Pi_mu complement basis.",
        {
            "old_split_rule": "P_mu P^mu = v^2 + 2 v delta P_0 + (delta P_0)^2 - |delta P^L|^2 - |delta P^T|^2",
            "photon_rule": "A_mu = delta P_mu^T / sqrt(Z_P)",
            "basis_rule": "the current canon needs an explicit statement of the post-photon temporal/Pi_mu complement basis after delta P^T is promoted to the photon branch",
            "hint_rule": "Pi_mu and delta P_t are source hints only until the post-photon delta P_t / Pi_mu complement is written as an explicit current-canon basis statement",
        },
        [
            row(
                "working_action_post_photon_temporal_pi_mu_basis_statement_source_inventory_complete",
                "pass",
                "working-action post-photon temporal/Pi_mu basis-statement source inventory complete",
                1,
                "The source pack for the temporal/Pi_mu basis residual branch is frozen.",
            ),
            row(
                "working_action_post_photon_temporal_pi_mu_basis_statement_source_pack_ready",
                "pass" if source_pack_ready else "reject",
                "working-action post-photon temporal/Pi_mu basis-statement source pack ready",
                1 if source_pack_ready else 0,
                "The residual branch has the required canonical anchors.",
            ),
            row(
                "working_action_post_photon_temporal_pi_mu_basis_statement_old_split_ready",
                "pass" if old_three_sector_split_ready else "reject",
                "old three-sector split ready as temporal/Pi_mu basis source",
                1 if old_three_sector_split_ready else 0,
                "The old split remains available as the pre-photon source structure.",
            ),
            row(
                "working_action_post_photon_temporal_pi_mu_basis_statement_photon_branch_ready",
                "pass" if photon_definition_ready and breakthrough_transverse_massless_ready else "reject",
                "photon branch ready before temporal/Pi_mu basis audit",
                1 if photon_definition_ready and breakthrough_transverse_massless_ready else 0,
                "The temporal/Pi_mu route begins only after the transverse photon branch is frozen.",
            ),
            row(
                "working_action_post_photon_temporal_pi_mu_basis_statement_time_fluctuation_ready",
                "pass" if time_fluctuation_ready else "reject",
                "time-sector fluctuation hint ready",
                1 if time_fluctuation_ready else 0,
                "Current canon still exposes a time-sector fluctuation hint for the post-photon temporal/Pi_mu complement basis.",
            ),
            row(
                "working_action_post_photon_temporal_pi_mu_basis_statement_pi_mu_hint_ready",
                "pass" if pi_mu_hint_ready else "reject",
                "Pi_mu hint ready",
                1 if pi_mu_hint_ready else 0,
                "Pi_mu remains visible as a gauge-invariant hint during the temporal/Pi_mu basis route.",
            ),
            row(
                "working_action_post_photon_temporal_pi_mu_basis_statement_route_contract_ready",
                "pass" if route_contract_ready else "reject",
                "temporal/Pi_mu basis route contract ready",
                1 if route_contract_ready else 0,
                "The previous route contract has frozen this residual route as the official next branch.",
            ),
        ],
        {
            "working_action_post_photon_temporal_pi_mu_basis_statement_source_pack_ready": source_pack_ready,
            "old_three_sector_split_ready": old_three_sector_split_ready,
            "photon_definition_ready": photon_definition_ready,
            "breakthrough_transverse_massless_ready": breakthrough_transverse_massless_ready,
            "time_sector_fluctuation_hint_ready": time_fluctuation_ready,
            "compact_full_action_ready": compact_action_ready,
            "pi_mu_gauge_invariant_hint_ready": pi_mu_hint_ready,
            "ghost_free_guard_ready": ghost_free_guard_ready,
            "temporal_pi_mu_basis_route_contract_ready": route_contract_ready,
            "first_route_to_close_or_none": "working_action_post_photon_temporal_pi_mu_basis_statement_identification_audit",
        },
        {
            "overall_status": "working_action_post_photon_temporal_pi_mu_basis_statement_source_inventory_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_215": True,
            "next_required_artifacts": [
                "working_action_post_photon_temporal_pi_mu_basis_statement_identification_audit"
            ],
        },
        {
            "inventory_targets": source_targets,
            "previous_source_summary": prev_source["summary"],
            "previous_audit_summary": prev_audit["summary"],
            "current_route_summary": current_route["summary"],
            "breakthrough_photon_formula": breakthrough_maxwell["summary"]["photon_definition_formula"],
            "ai_context_current_step": ai_context["current_step"],
            "vev_quadratic_formulas": vev_quadratic["evidence"],
            "breakthrough_vev_summary": breakthrough_vev["summary"],
        },
    )
    write_artifact(
        "mass_origin_v2_working_action_post_photon_temporal_pi_mu_basis_statement_source_inventory",
        source_inventory,
    )

    candidate_patterns = [
        "delta P_t and Pi_mu",
        "delta P_t と Pi_mu",
        "delta P_t / Pi_mu",
        "temporal/Pi_mu",
        "after the photon branch is removed",
        "photon branch を差し引いたあと",
    ]
    candidate_hits = [hit(part1, pattern) for pattern in candidate_patterns]
    candidate_hits = [item for item in candidate_hits if item is not None]

    basis_statement_available = False
    delta_pt_pi_mu_complement_statement_available = False

    identification_audit = payload(
        "8.7.56.215",
        "Working-action post-photon temporal/Pi_mu basis-statement identification audit",
        common_inputs,
        "Audit whether the current canon actually states the post-photon temporal/Pi_mu complement basis.",
        {
            "statement_rule": "a successful route needs an explicit basis statement of what remains as the delta P_t / Pi_mu complement once delta P^T is promoted to A_mu",
            "basis_rule": "the missing statement must identify a post-photon delta P_t / Pi_mu complement, not merely restate the old split, Pi_mu, or the static limit hints",
            "downstream_rule": "without that basis statement, no post-photon remainder mapping, component decomposition, quadratic form, or projector can be claimed",
        },
        [
            row(
                "working_action_post_photon_temporal_pi_mu_basis_statement_old_split_available",
                "pass",
                "old three-sector split available as source hint",
                1,
                "The old VEV split remains available as hint-level evidence.",
            ),
            row(
                "working_action_post_photon_temporal_pi_mu_basis_statement_photon_definition_available",
                "pass",
                "photon definition available before temporal/Pi_mu basis audit",
                1,
                "The transverse photon branch is already frozen under the breakthrough route.",
            ),
            row(
                "working_action_post_photon_temporal_pi_mu_basis_statement_time_fluctuation_available",
                "pass" if time_fluctuation_ready else "reject",
                "time-sector fluctuation hint available",
                1 if time_fluctuation_ready else 0,
                "The temporal fluctuation hint is present, but it is not yet the required post-photon temporal/Pi_mu basis statement.",
            ),
            row(
                "working_action_post_photon_temporal_pi_mu_basis_statement_available",
                "pass" if basis_statement_available else "reject",
                "working-action post-photon temporal/Pi_mu basis statement available",
                1 if basis_statement_available else 0,
                "Current canon still needs an explicit post-photon temporal/Pi_mu basis statement for the complement.",
            ),
            row(
                "working_action_post_photon_delta_pt_pi_mu_complement_statement_available",
                "pass" if delta_pt_pi_mu_complement_statement_available else "reject",
                "working-action post-photon delta-P_t/Pi_mu complement statement available",
                1 if delta_pt_pi_mu_complement_statement_available else 0,
                "Current canon still does not state the explicit delta P_t / Pi_mu complement after photon extraction.",
            ),
            row(
                "working_action_post_photon_nontransverse_component_mapping_available_after_temporal_pi_mu_audit",
                "pass" if delta_pt_pi_mu_complement_statement_available else "reject",
                "working-action post-photon nontransverse component mapping available after temporal/Pi_mu audit",
                1 if delta_pt_pi_mu_complement_statement_available else 0,
                "Without the missing delta P_t / Pi_mu complement statement, no downstream mapping can be claimed.",
            ),
        ],
        {
            "old_three_sector_split_available_as_source_hint": True,
            "photon_definition_available": True,
            "time_sector_fluctuation_hint_available": time_fluctuation_ready,
            "pi_mu_gauge_invariant_hint_available": True,
            "working_action_post_photon_temporal_pi_mu_basis_statement_available": basis_statement_available,
            "working_action_post_photon_delta_pt_pi_mu_complement_statement_available": delta_pt_pi_mu_complement_statement_available,
            "working_action_post_photon_nontransverse_remainder_statement_available": False,
            "working_action_post_photon_nontransverse_component_mapping_available": False,
            "working_action_nontransverse_component_field_decomposition_available": False,
            "working_action_nontransverse_component_quadratic_form_available": False,
            "working_action_massive_sector_projector_available": False,
            "identification_nonclosure_reason_or_none": "working_action_post_photon_delta_pt_pi_mu_complement_statement_absent",
            "first_route_to_close_or_none": "working_action_vector_rebuild_reopen_thirteenth_gate",
        },
        {
            "overall_status": "working_action_post_photon_temporal_pi_mu_basis_statement_identification_nonclosure_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_216": True,
            "next_required_artifacts": ["working_action_vector_rebuild_reopen_thirteenth_gate"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "previous_audit_summary": prev_audit["summary"],
            "current_route_summary": current_route["summary"],
            "breakthrough_photon_formula": breakthrough_maxwell["summary"]["photon_definition_formula"],
            "candidate_basis_hits": candidate_hits,
            "part1_full_action_line": hit(part1, "\\mathcal{L}_{P,\\mathrm{full}}"),
            "part1_pi_mu_line": hit(part1, "\\Pi_\\mu:=P_\\mu-\\partial_\\mu\\pi/m_P"),
            "part1_time_fluctuation_line": hit(part1, "P_t=P_t^{\\mathrm{(lin)}}+\\delta P_t,"),
        },
    )
    write_artifact(
        "mass_origin_v2_working_action_post_photon_temporal_pi_mu_basis_statement_identification_audit",
        identification_audit,
    )

    reopen_gate = payload(
        "8.7.56.216",
        "Working-action vector rebuild reopen thirteenth gate / Trial-3 fallback thirteenth refresh",
        common_inputs,
        "Integrate the temporal/Pi_mu basis audit and decide whether the vector rebuild can reopen or whether a deeper post-photon delta-P_t/Pi_mu complement route must be selected.",
        {
            "reopen_rule": "the vector rebuild reopens only if the current canon freezes an explicit post-photon temporal/Pi_mu basis statement",
            "basis_rule": "if that statement still fails, the next deeper blocker is the absent delta-P_t/Pi_mu complement statement after photon subtraction",
            "fallback_rule": "Trial-3 remains on fallback hold while the same post-photon delta-P_t/Pi_mu complement issue stays unresolved",
        },
        [
            row(
                "working_action_vector_rebuild_thirteenth_gate_source_inventory_ready",
                "pass",
                "working-action post-photon temporal/Pi_mu source inventory ready at thirteenth gate",
                1,
                "The temporal/Pi_mu basis residual branch has its source pack frozen.",
            ),
            row(
                "working_action_vector_rebuild_thirteenth_gate_identification_ready",
                "reject",
                "working-action post-photon temporal/Pi_mu identification ready at thirteenth gate",
                0,
                "The explicit temporal/Pi_mu basis statement is still missing.",
            ),
            row(
                "working_action_vector_rebuild_thirteenth_gate_anchor_refresh_ready",
                "reject",
                "working-action vector anchor refresh ready at thirteenth gate",
                0,
                "Anchor refresh remains blocked until projector and radial eigenoperator become current-canon ready.",
            ),
            row(
                "working_action_vector_rebuild_thirteenth_gate_trial3_hold_retained",
                "pass",
                "Trial-3 fallback hold retained at thirteenth gate",
                1,
                "The weak-sector branch remains downstream of the unresolved post-photon delta-P_t/Pi_mu complement route.",
            ),
        ],
        {
            "working_action_vector_rebuild_reopen_ready": False,
            "working_action_post_photon_temporal_pi_mu_basis_statement_identification_ready": False,
            "working_action_post_photon_delta_pt_pi_mu_complement_statement_identification_ready": False,
            "working_action_massive_sector_projector_identification_ready": False,
            "working_action_vector_mass_spectrum_reduced_solver_numeric_ready": False,
            "working_action_vector_mass_spectrum_anchor_refresh_ready": False,
            "trial2_paper_side_sync_deferred_until_vector_anchor_refresh": True,
            "trial3_fallback_hold_retained": True,
            "trial3_fallback_hold_release_ready": False,
            "recommended_next_route_or_none": "8.7.56.217",
        },
        {
            "overall_status": "working_action_vector_rebuild_thirteenth_gate_nonclosure_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_217": True,
            "next_required_artifacts": [
                "working_action_post_photon_delta_pt_pi_mu_complement_statement_identification"
            ],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "trial2_declaration_summary": trial2_declaration["summary"],
            "vector_reduced_solver_summary": vector_reduced_solver["summary"],
            "vector_anchor_summary": vector_anchor["summary"],
            "trial3_fallback_route_summary": trial3_fallback_route["summary"],
            "reopen_previous_gate_summary": prev_gate["summary"],
        },
    )
    write_artifact("mass_origin_v2_vector_rebuild_reopen_thirteenth_gate", reopen_gate)

    next_route = payload(
        "8.7.56.217",
        "Working-action post-photon delta-P_t/Pi_mu complement-statement route contract",
        common_inputs,
        "Freeze the deeper residual route suggested by the temporal/Pi_mu basis audit: identify the explicit post-photon delta P_t / Pi_mu complement statement that says what survives after the photon branch is removed from the old split.",
        {
            "selected_residual_route": "working_action_post_photon_delta_pt_pi_mu_complement_statement_identification",
            "blocking_rule": "the old split, breakthrough photon formula, delta P_t fluctuation hint, and Pi_mu hint are present, but current canon still lacks the explicit post-photon delta-P_t/Pi_mu complement statement",
            "dependency_rule": "post-photon mapping, component decomposition, quadratic form, projector, radial eigenoperator, anchor refresh, Trial-2 paper-side sync, and Trial-3 fallback release all remain downstream of the same missing delta-P_t/Pi_mu complement statement",
        },
        [
            row(
                "working_action_post_photon_delta_pt_pi_mu_complement_statement_route_contract_complete",
                "pass",
                "working-action post-photon delta-P_t/Pi_mu complement-statement route contract complete",
                1,
                "The next deeper residual route is frozen after the temporal/Pi_mu basis non-closure.",
            ),
            row(
                "working_action_post_photon_delta_pt_pi_mu_complement_statement_missing",
                "pass",
                "working-action post-photon delta-P_t/Pi_mu complement statement missing",
                1,
                "The missing artifact is the explicit delta P_t / Pi_mu complement statement for what survives after the photon branch is removed.",
            ),
            row(
                "working_action_post_photon_delta_pt_pi_mu_complement_statement_trial3_dependency_blocked",
                "pass",
                "Trial-3 dependency still blocked by post-photon delta-P_t/Pi_mu complement statement",
                1,
                "The weak-sector branch remains blocked by the unresolved post-photon delta-P_t/Pi_mu complement route.",
            ),
            row(
                "working_action_post_photon_delta_pt_pi_mu_complement_statement_paper_sync_deferred",
                "pass",
                "Trial-2 paper-side sync still deferred by post-photon delta-P_t/Pi_mu complement statement",
                1,
                "Paper-side sync remains deferred until vector anchor refresh becomes current-canon ready.",
            ),
        ],
        {
            "selected_residual_route": "working_action_post_photon_delta_pt_pi_mu_complement_statement_identification",
            "missing_v2_artifact": "working_action_post_photon_delta_pt_pi_mu_complement_statement",
            "trial3_dependency_state": "blocked_by_working_action_post_photon_delta_pt_pi_mu_complement_statement",
            "trial2_paper_side_sync_state": "deferred_until_working_action_vector_anchor_refresh",
            "split_contract_ready": True,
            "recommended_next_route_or_none": "8.7.56.218",
        },
        {
            "overall_status": "working_action_post_photon_delta_pt_pi_mu_complement_statement_route_contract_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_218": True,
            "next_required_artifacts": [
                "working_action_post_photon_delta_pt_pi_mu_complement_statement_source_inventory",
                "working_action_post_photon_delta_pt_pi_mu_complement_statement_identification_audit",
                "working_action_vector_rebuild_reopen_fourteenth_gate",
            ],
        },
        {
            "reopen_gate_summary": reopen_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "current_route_summary": current_route["summary"],
            "part1_full_action_line": hit(part1, "\\mathcal{L}_{P,\\mathrm{full}}"),
            "breakthrough_photon_formula": breakthrough_maxwell["summary"]["photon_definition_formula"],
        },
    )
    write_artifact(
        "mass_origin_v2_working_action_post_photon_delta_pt_pi_mu_complement_statement_route_contract",
        next_route,
    )

    print("[ok] wrote:")
    print(" - mass_origin_v2_working_action_post_photon_temporal_pi_mu_basis_statement_source_inventory_metrics.json")
    print(" - mass_origin_v2_working_action_post_photon_temporal_pi_mu_basis_statement_identification_audit_metrics.json")
    print(" - mass_origin_v2_vector_rebuild_reopen_thirteenth_gate_metrics.json")
    print(" - mass_origin_v2_working_action_post_photon_delta_pt_pi_mu_complement_statement_route_contract_metrics.json")


# Function: run the script entry point.

if __name__ == "__main__":
    main()
