#!/usr/bin/env python3
"""
Generate working-action massive-sector projector artifacts for 8.7.56.178-.181.

This branch sharpens the post-breakthrough vector rebuild blocker after the
radial-eigenoperator route failed. The question is no longer whether the
transverse photon branch exists; that is already frozen under the breakthrough
working action. The remaining question is whether current canon identifies the
complementary massive nontransverse sector strongly enough to define a current-
canon projector and reopen the rebuilt vector mass-spectrum solver.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

VEV_QUADRATIC = OUT / "mass_origin_v2_vev_quadratic_mode_decomposition_metrics.json"
TRIAL1_BREAKTHROUGH_VEV = OUT / "mass_origin_v2_trial1_breakthrough_modified_vev_decomposition_metrics.json"
TRIAL1_BREAKTHROUGH_MAXWELL = OUT / "mass_origin_v2_trial1_breakthrough_maxwell_coupling_audit_metrics.json"
TRIAL2_DECLARATION = OUT / "mass_origin_v2_trial2_declaration_gate_metrics.json"
VECTOR_REBUILD_SOURCE = OUT / "mass_origin_v2_vector_rebuild_source_inventory_metrics.json"
VECTOR_REDUCED_SOLVER = OUT / "mass_origin_v2_vector_reduced_solver_pilot_metrics.json"
VECTOR_ANCHOR = OUT / "mass_origin_v2_vector_anchor_refresh_metrics.json"
VECTOR_RADIAL_AUDIT = OUT / "mass_origin_v2_vector_radial_eigenoperator_identification_audit_metrics.json"
VECTOR_REOPEN_GATE = OUT / "mass_origin_v2_vector_rebuild_reopen_gate_metrics.json"
VECTOR_ROUTE_CONTRACT = OUT / "mass_origin_v2_working_action_massive_sector_projector_route_contract_metrics.json"
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


# Function: execute the working-action massive-sector projector residual branch.

def main() -> None:
    for path in (
        PART1,
        PART3A,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        VEV_QUADRATIC,
        TRIAL1_BREAKTHROUGH_VEV,
        TRIAL1_BREAKTHROUGH_MAXWELL,
        TRIAL2_DECLARATION,
        VECTOR_REBUILD_SOURCE,
        VECTOR_REDUCED_SOLVER,
        VECTOR_ANCHOR,
        VECTOR_RADIAL_AUDIT,
        VECTOR_REOPEN_GATE,
        VECTOR_ROUTE_CONTRACT,
        TRIAL3_FALLBACK_ROUTE,
    ):
        req(path)

    part1 = read_text(PART1)
    part3a = read_text(PART3A)
    status = read_text(STATUS)
    roadmap = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)

    vev_quadratic = read_json(VEV_QUADRATIC)
    breakthrough_vev = read_json(TRIAL1_BREAKTHROUGH_VEV)
    breakthrough_maxwell = read_json(TRIAL1_BREAKTHROUGH_MAXWELL)
    trial2_declaration = read_json(TRIAL2_DECLARATION)
    vector_rebuild_source = read_json(VECTOR_REBUILD_SOURCE)
    vector_reduced_solver = read_json(VECTOR_REDUCED_SOLVER)
    vector_anchor = read_json(VECTOR_ANCHOR)
    vector_radial_audit = read_json(VECTOR_RADIAL_AUDIT)
    vector_reopen_gate = read_json(VECTOR_REOPEN_GATE)
    vector_route_contract = read_json(VECTOR_ROUTE_CONTRACT)
    trial3_fallback_route = read_json(TRIAL3_FALLBACK_ROUTE)

    common_inputs = {
        "part1_markdown": rel(PART1),
        "part3a_markdown": rel(PART3A),
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_vev_quadratic_mode_decomposition_json": rel(VEV_QUADRATIC),
        "mass_origin_v2_trial1_breakthrough_modified_vev_decomposition_json": rel(TRIAL1_BREAKTHROUGH_VEV),
        "mass_origin_v2_trial1_breakthrough_maxwell_coupling_audit_json": rel(TRIAL1_BREAKTHROUGH_MAXWELL),
        "mass_origin_v2_trial2_declaration_gate_json": rel(TRIAL2_DECLARATION),
        "mass_origin_v2_vector_rebuild_source_inventory_json": rel(VECTOR_REBUILD_SOURCE),
        "mass_origin_v2_vector_reduced_solver_pilot_json": rel(VECTOR_REDUCED_SOLVER),
        "mass_origin_v2_vector_anchor_refresh_json": rel(VECTOR_ANCHOR),
        "mass_origin_v2_vector_radial_eigenoperator_identification_audit_json": rel(VECTOR_RADIAL_AUDIT),
        "mass_origin_v2_vector_rebuild_reopen_gate_json": rel(VECTOR_REOPEN_GATE),
        "mass_origin_v2_working_action_massive_sector_projector_route_contract_json": rel(VECTOR_ROUTE_CONTRACT),
        "mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_json": rel(TRIAL3_FALLBACK_ROUTE),
    }

    source_targets = [
        target_record(
            "part1_full_action_anchor",
            PART1,
            part1,
            "\\mathcal{L}_{P,\\mathrm{full}}",
            "Part I still anchors the full vector-P action whose fluctuation content must be reinterpreted under the breakthrough working action.",
        ),
        target_record(
            "part1_stueckelberg_mass_line",
            PART1,
            part1,
            "+\\frac{m_P^2}{2}\\left(P_\\mu-\\frac{1}{m_P}\\partial_\\mu\\pi\\right)",
            "The old canon still records the temporal/longitudinal/Stueckelberg mixing source that must be revisited after photon extraction.",
        ),
        target_record(
            "part1_gauge_fixing_line",
            PART1,
            part1,
            "-\\frac{1}{2\\xi_g}\\left(\\partial_\\mu P^\\mu+\\xi_g m_P\\pi\\right)^2",
            "Part I still exposes the gauge-fixing term that controls how longitudinal and Stueckelberg modes mix.",
        ),
        target_record(
            "part1_gauge_transform_line",
            PART1,
            part1,
            "P_\\mu\\to P_\\mu+\\partial_\\mu\\alpha,",
            "The local gauge structure remains an input to any post-photon sector split.",
        ),
        target_record(
            "status_next_step_anchor",
            STATUS,
            status,
            "current official next step は `8.7.56.178`",
            "STATUS must already expose the massive-sector projector branch as the next official route.",
        ),
        target_record(
            "roadmap_projector_branch_anchor",
            ROADMAP,
            roadmap,
            "`8.7.56.177-.180`",
            "ROADMAP must already expose the current projector residual branch.",
        ),
    ]
    source_pack_ready = all(item["present"] for item in source_targets)

    old_three_sector_split_ready = bool(vev_quadratic["summary"]["three_sector_split_available"])
    breakthrough_transverse_massless = bool(
        breakthrough_vev["summary"]["transverse_mode_massless_under_breakthrough_action"]
    )
    photon_definition_ready = bool(breakthrough_maxwell["summary"]["transverse_maxwell_reduction_available"])
    photon_formula = breakthrough_maxwell["summary"]["photon_definition_formula"]
    stueckelberg_mix_source_ready = (
        source_targets[1]["present"] and source_targets[2]["present"] and source_targets[3]["present"]
    )
    historic_template_pack_ready = bool(vector_rebuild_source["summary"]["old_vector_benchmark_pack_retained"])

    source_inventory = payload(
        "8.7.56.178",
        "Working-action massive-sector projector source inventory",
        common_inputs,
        "Freeze the source pack needed to decide whether the current canon identifies the massive sector that remains after the transverse photon branch is extracted.",
        {
            "old_split_rule": "the pre-breakthrough VEV quadratic pack already separates delta P_0, delta P^L, and delta P^T at quadratic order",
            "breakthrough_rule": "the breakthrough working action makes delta P^T massless and identifies A_mu = delta P_mu^T / sqrt(Z_P)",
            "projector_requirement": "to reopen the vector rebuild, current canon must also identify the complementary massive sector built from the remaining nontransverse fluctuations",
            "candidate_complement": "delta P_mu^(M) ?= complement of delta P_mu^T inside (delta P_0, delta P_i^L, pi)",
        },
        [
            row(
                "working_action_massive_sector_projector_source_inventory_complete",
                "pass",
                "working-action massive-sector projector source inventory complete",
                1,
                "The source pack for the projector residual branch is frozen.",
            ),
            row(
                "working_action_old_three_sector_split_ready",
                "pass" if old_three_sector_split_ready else "reject",
                "old VEV three-sector split ready",
                1 if old_three_sector_split_ready else 0,
                "The old VEV expansion still separates time/radial, longitudinal, and transverse sectors.",
            ),
            row(
                "working_action_breakthrough_transverse_massless_ready",
                "pass" if breakthrough_transverse_massless else "reject",
                "breakthrough transverse massless route ready",
                1 if breakthrough_transverse_massless else 0,
                "The breakthrough working action keeps the transverse branch massless.",
            ),
            row(
                "working_action_photon_definition_ready",
                "pass" if photon_definition_ready else "reject",
                "photon definition ready",
                1 if photon_definition_ready else 0,
                "The photon candidate is already frozen as the transverse fluctuation.",
            ),
            row(
                "working_action_stueckelberg_mix_source_ready",
                "pass" if stueckelberg_mix_source_ready else "reject",
                "temporal/longitudinal/Stueckelberg mix source ready",
                1 if stueckelberg_mix_source_ready else 0,
                "Part I still exposes the mass/gauge-fixing structure whose complement sector must be identified.",
            ),
            row(
                "working_action_historic_template_pack_ready",
                "pass" if historic_template_pack_ready else "reject",
                "historic vector rebuild template pack ready",
                1 if historic_template_pack_ready else 0,
                "Historic vector rebuild bookkeeping remains available as benchmark template only.",
            ),
        ],
        {
            "working_action_massive_sector_projector_source_pack_ready": source_pack_ready,
            "old_three_sector_split_ready": old_three_sector_split_ready,
            "breakthrough_transverse_massless_ready": breakthrough_transverse_massless,
            "photon_definition_ready": photon_definition_ready,
            "stueckelberg_mix_source_ready": stueckelberg_mix_source_ready,
            "historic_vector_template_pack_ready": historic_template_pack_ready,
            "first_route_to_close_or_none": "working_action_massive_sector_projector_identification_audit",
        },
        {
            "overall_status": "working_action_massive_sector_projector_source_inventory_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_179": True,
            "next_required_artifacts": ["working_action_massive_sector_projector_identification_audit"],
        },
        {
            "inventory_targets": source_targets,
            "vev_quadratic_summary": vev_quadratic["summary"],
            "breakthrough_vev_summary": breakthrough_vev["summary"],
            "breakthrough_maxwell_summary": breakthrough_maxwell["summary"],
            "vector_rebuild_source_summary": vector_rebuild_source["summary"],
            "vector_route_contract_summary": vector_route_contract["summary"],
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    transverse_photon_branch_identification_available = old_three_sector_split_ready and photon_definition_ready
    massive_nontransverse_sector_basis_available = False
    massive_sector_projector_available = False
    radial_eigenoperator_ready_after_projector = False

    identification_audit = payload(
        "8.7.56.179",
        "Working-action massive-sector projector identification audit",
        common_inputs,
        "Audit whether current canon already identifies the complement of the transverse photon branch strongly enough to define a massive-sector projector for the rebuilt vector spectrum.",
        {
            "photon_branch_rule": "A_mu = delta P_mu^T / sqrt(Z_P) already canonizes the transverse photon branch under the breakthrough working action",
            "projector_rule": "a massive-sector projector needs a canonical complement basis for the remaining nontransverse fluctuations, not just the knowledge that a photon branch exists",
            "basis_rule": "the missing basis must resolve how delta P_0, delta P_i^L, and the Stueckelberg scalar pi combine after the transverse branch is removed",
        },
        [
            row(
                "working_action_transverse_photon_branch_identification_available",
                "pass" if transverse_photon_branch_identification_available else "reject",
                "transverse photon branch identification available",
                1 if transverse_photon_branch_identification_available else 0,
                "The current working canon does identify the transverse photon candidate.",
            ),
            row(
                "working_action_transverse_photon_subtraction_partially_ready",
                "pass" if transverse_photon_branch_identification_available else "reject",
                "transverse photon subtraction partially ready",
                1 if transverse_photon_branch_identification_available else 0,
                "The transverse branch can be named, but its complement is not yet canonized as a single massive sector.",
            ),
            row(
                "working_action_massive_nontransverse_basis_available",
                "reject",
                "massive nontransverse sector basis available",
                0,
                "Current canon does not yet define how the temporal, longitudinal, and Stueckelberg fluctuations combine after photon extraction.",
            ),
            row(
                "working_action_massive_sector_projector_available",
                "reject",
                "working-action massive-sector projector available",
                0,
                "Without that complement basis, no current-canon projector onto the massive sector can be written.",
            ),
            row(
                "working_action_vector_radial_eigenoperator_ready_after_projector_audit",
                "reject",
                "working-action vector radial eigenoperator ready after projector audit",
                0,
                "The radial eigenoperator remains blocked by the missing nontransverse complement basis.",
            ),
        ],
        {
            "transverse_photon_branch_identification_available": transverse_photon_branch_identification_available,
            "transverse_photon_subtraction_ready": transverse_photon_branch_identification_available,
            "working_action_massive_nontransverse_sector_basis_available": massive_nontransverse_sector_basis_available,
            "working_action_massive_sector_projector_available": massive_sector_projector_available,
            "working_action_vector_radial_eigenoperator_identification_available": radial_eigenoperator_ready_after_projector,
            "identification_nonclosure_reason_or_none": "working_action_temporal_longitudinal_stueckelberg_mode_basis_absent",
            "first_route_to_close_or_none": "working_action_vector_rebuild_reopen_fourth_gate",
        },
        {
            "overall_status": "working_action_massive_sector_projector_identification_nonclosure_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_180": True,
            "next_required_artifacts": ["working_action_vector_rebuild_reopen_fourth_gate"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "vev_quadratic_formulas": vev_quadratic["formulas"],
            "breakthrough_vev_formulas": breakthrough_vev["formulas"],
            "breakthrough_maxwell_formulas": breakthrough_maxwell["formulas"],
            "part1_full_action_line": hit(part1, "\\mathcal{L}_{P,\\mathrm{full}}"),
            "part1_gauge_transform_line": hit(part1, "P_\\mu\\to P_\\mu+\\partial_\\mu\\alpha,"),
            "photon_formula": photon_formula,
        },
    )

    reopen_gate = payload(
        "8.7.56.180",
        "Working-action vector rebuild reopen fourth gate / Trial-3 fallback fourth refresh",
        common_inputs,
        "Integrate the projector audit and decide whether the working-action vector rebuild can reopen or whether a deeper residual route is required.",
        {
            "reopen_rule": "the vector rebuild reopens only if the current canon identifies a massive-sector projector after photon extraction",
            "anchor_rule": "anchor refresh remains downstream of the same current-canon projector because no numeric mass proxy exists without it",
            "fallback_rule": "Trial-3 stays on fallback hold while the nontransverse complement basis is still absent",
        },
        [
            row(
                "working_action_vector_rebuild_fourth_gate_source_inventory_ready",
                "pass" if source_inventory["summary"]["working_action_massive_sector_projector_source_pack_ready"] else "reject",
                "working-action massive-sector projector source inventory ready at fourth gate",
                1 if source_inventory["summary"]["working_action_massive_sector_projector_source_pack_ready"] else 0,
                "The projector residual branch has its source pack frozen.",
            ),
            row(
                "working_action_vector_rebuild_fourth_gate_projector_identification_ready",
                "reject",
                "working-action massive-sector projector identification ready at fourth gate",
                0,
                "The complement basis for the massive sector is still missing.",
            ),
            row(
                "working_action_vector_rebuild_fourth_gate_anchor_refresh_ready",
                "reject",
                "working-action vector anchor refresh ready at fourth gate",
                0,
                "Anchor refresh remains blocked until the projector and radial eigenoperator become current-canon ready.",
            ),
            row(
                "working_action_vector_rebuild_fourth_gate_trial3_hold_retained",
                "pass",
                "Trial-3 fallback hold retained at fourth gate",
                1,
                "The weak-sector branch remains downstream of the unresolved massive-sector complement basis.",
            ),
        ],
        {
            "working_action_vector_rebuild_reopen_ready": False,
            "working_action_massive_sector_projector_identification_ready": False,
            "working_action_vector_mass_spectrum_reduced_solver_numeric_ready": False,
            "working_action_vector_mass_spectrum_anchor_refresh_ready": False,
            "trial2_paper_side_sync_deferred_until_vector_anchor_refresh": True,
            "trial3_fallback_hold_retained": True,
            "trial3_fallback_hold_release_ready": False,
            "recommended_next_route_or_none": "8.7.56.181",
        },
        {
            "overall_status": "working_action_vector_rebuild_fourth_gate_nonclosure_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_181": True,
            "next_required_artifacts": ["working_action_temporal_longitudinal_stueckelberg_mode_basis_identification"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "trial2_declaration_summary": trial2_declaration["summary"],
            "vector_reduced_solver_summary": vector_reduced_solver["summary"],
            "vector_anchor_summary": vector_anchor["summary"],
            "trial3_fallback_route_summary": trial3_fallback_route["summary"],
            "vector_reopen_gate_summary": vector_reopen_gate["summary"],
        },
    )

    route_contract = payload(
        "8.7.56.181",
        "Working-action temporal/longitudinal/Stueckelberg basis route contract",
        common_inputs,
        "Freeze the deeper residual route suggested by the projector audit: identify the canonical complement basis that remains after the transverse photon branch is extracted.",
        {
            "selected_residual_route": "working_action_temporal_longitudinal_stueckelberg_mode_basis_identification",
            "blocking_rule": "without a canonical basis for the nontransverse complement, no current-canon massive-sector projector can be claimed",
            "dependency_rule": "vector rebuild, anchor refresh, Trial-2 paper-side sync, and Trial-3 fallback release all remain downstream of the same missing basis",
        },
        [
            row(
                "working_action_nontransverse_basis_route_contract_complete",
                "pass",
                "working-action nontransverse basis route contract complete",
                1,
                "The next deeper residual route is frozen after the projector non-closure.",
            ),
            row(
                "working_action_nontransverse_basis_missing",
                "pass",
                "working-action temporal/longitudinal/Stueckelberg basis missing",
                1,
                "The missing artifact is the canonical complement basis after the transverse photon branch is removed.",
            ),
            row(
                "working_action_nontransverse_basis_trial3_dependency_blocked",
                "pass",
                "Trial-3 dependency still blocked by nontransverse basis",
                1,
                "The weak-sector branch remains blocked by the unresolved complement basis.",
            ),
            row(
                "working_action_nontransverse_basis_paper_sync_deferred",
                "pass",
                "Trial-2 paper-side sync still deferred by nontransverse basis",
                1,
                "Paper-side sync remains deferred until vector anchor refresh becomes current-canon ready.",
            ),
        ],
        {
            "selected_residual_route": "working_action_temporal_longitudinal_stueckelberg_mode_basis_identification",
            "missing_v2_artifact": "working_action_temporal_longitudinal_stueckelberg_mode_basis",
            "trial3_dependency_state": "blocked_by_working_action_temporal_longitudinal_stueckelberg_mode_basis",
            "trial2_paper_side_sync_state": "deferred_until_working_action_vector_anchor_refresh",
            "split_contract_ready": True,
            "recommended_next_route_or_none": "8.7.56.182",
        },
        {
            "overall_status": "working_action_nontransverse_basis_route_contract_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_182": True,
            "next_required_artifacts": [
                "working_action_temporal_longitudinal_stueckelberg_mode_basis_source_inventory",
                "working_action_temporal_longitudinal_stueckelberg_mode_basis_identification_audit",
                "working_action_vector_rebuild_reopen_fifth_gate",
            ],
        },
        {
            "reopen_gate_summary": reopen_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "vector_route_contract_summary": vector_route_contract["summary"],
            "trial3_fallback_route_summary": trial3_fallback_route["summary"],
            "part3a_case_b_line": hit(part3a, "A棄却、B採用"),
        },
    )

    write_artifact("mass_origin_v2_working_action_massive_sector_projector_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_working_action_massive_sector_projector_identification_audit", identification_audit)
    write_artifact("mass_origin_v2_vector_rebuild_reopen_fourth_gate", reopen_gate)
    write_artifact("mass_origin_v2_working_action_temporal_longitudinal_stueckelberg_basis_route_contract", route_contract)

    print("[ok] wrote:")
    print(" - mass_origin_v2_working_action_massive_sector_projector_source_inventory_metrics.json")
    print(" - mass_origin_v2_working_action_massive_sector_projector_identification_audit_metrics.json")
    print(" - mass_origin_v2_vector_rebuild_reopen_fourth_gate_metrics.json")
    print(" - mass_origin_v2_working_action_temporal_longitudinal_stueckelberg_basis_route_contract_metrics.json")


# Function: run the working-action massive-sector projector branch from the CLI.

if __name__ == "__main__":
    main()
