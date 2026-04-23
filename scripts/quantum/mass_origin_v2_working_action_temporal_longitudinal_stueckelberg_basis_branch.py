#!/usr/bin/env python3
"""
Generate working-action temporal/longitudinal/Stueckelberg basis artifacts for 8.7.56.182-.185.

This branch deepens the post-breakthrough vector rebuild residual route after
the massive-sector projector audit. The photon branch is already identified as
the transverse fluctuation. What remains unresolved is whether the current
canon provides a canonical complement basis for the temporal, longitudinal, and
Stueckelberg fluctuations, and whether that complement is diagonalized strongly
enough to support a projector and rebuilt radial eigenoperator.
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
PROJECTOR_SOURCE = OUT / "mass_origin_v2_working_action_massive_sector_projector_source_inventory_metrics.json"
PROJECTOR_AUDIT = OUT / "mass_origin_v2_working_action_massive_sector_projector_identification_audit_metrics.json"
REOPEN_FOURTH = OUT / "mass_origin_v2_vector_rebuild_reopen_fourth_gate_metrics.json"
PROJECTOR_ROUTE = OUT / "mass_origin_v2_working_action_temporal_longitudinal_stueckelberg_basis_route_contract_metrics.json"
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


# Function: execute the temporal/longitudinal/Stueckelberg basis residual branch.

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
        PROJECTOR_SOURCE,
        PROJECTOR_AUDIT,
        REOPEN_FOURTH,
        PROJECTOR_ROUTE,
        VECTOR_REDUCED_SOLVER,
        VECTOR_ANCHOR,
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
    projector_source = read_json(PROJECTOR_SOURCE)
    projector_audit = read_json(PROJECTOR_AUDIT)
    reopen_fourth = read_json(REOPEN_FOURTH)
    projector_route = read_json(PROJECTOR_ROUTE)
    vector_reduced_solver = read_json(VECTOR_REDUCED_SOLVER)
    vector_anchor = read_json(VECTOR_ANCHOR)
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
        "mass_origin_v2_working_action_massive_sector_projector_source_inventory_json": rel(PROJECTOR_SOURCE),
        "mass_origin_v2_working_action_massive_sector_projector_identification_audit_json": rel(PROJECTOR_AUDIT),
        "mass_origin_v2_vector_rebuild_reopen_fourth_gate_json": rel(REOPEN_FOURTH),
        "mass_origin_v2_working_action_temporal_longitudinal_stueckelberg_basis_route_contract_json": rel(PROJECTOR_ROUTE),
        "mass_origin_v2_vector_reduced_solver_pilot_json": rel(VECTOR_REDUCED_SOLVER),
        "mass_origin_v2_vector_anchor_refresh_json": rel(VECTOR_ANCHOR),
        "mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_json": rel(TRIAL3_FALLBACK_ROUTE),
    }

    source_targets = [
        target_record(
            "part1_stueckelberg_mass_line",
            PART1,
            part1,
            "+\\frac{m_P^2}{2}\\left(P_\\mu-\\frac{1}{m_P}\\partial_\\mu\\pi\\right)",
            "Part I still exposes the Stückelberg-completed massive vector term.",
        ),
        target_record(
            "part1_gauge_fixing_line",
            PART1,
            part1,
            "-\\frac{1}{2\\xi_g}\\left(\\partial_\\mu P^\\mu+\\xi_g m_P\\pi\\right)^2",
            "Part I still exposes the gauge-fixing structure that mixes longitudinal and Stückelberg modes.",
        ),
        target_record(
            "part1_propagator_massive_pole_line",
            PART1,
            part1,
            "\\frac{\\eta_{\\mu\\nu}-k_\\mu k_\\nu/m_P^2}{k^2-m_P^2+i0}",
            "Part I still exposes the massive vector pole before the breakthrough reinterpretation.",
        ),
        target_record(
            "part1_propagator_gauge_piece_line",
            PART1,
            part1,
            "\\frac{\\xi_g\\,k_\\mu k_\\nu/m_P^2}{k^2-\\xi_g m_P^2+i0}",
            "Part I still exposes the gauge-dependent longitudinal/Stückelberg propagator piece.",
        ),
        target_record(
            "part1_pi_definition_line",
            PART1,
            part1,
            "\\Pi_\\mu:=P_\\mu-\\partial_\\mu\\pi/m_P",
            "Part I still defines the gauge-invariant Stückelberg combination Pi_mu.",
        ),
        target_record(
            "part1_ghost_free_clause",
            PART1,
            part1,
            "負ノルム（ghost）モードは出現しない。",
            "Any complement-basis identification must preserve the ghost-free closure.",
        ),
        target_record(
            "status_next_step_anchor",
            STATUS,
            status,
            "current official next step は `8.7.56.182`",
            "STATUS must already expose the basis residual branch as the next official route.",
        ),
        target_record(
            "roadmap_basis_branch_anchor",
            ROADMAP,
            roadmap,
            "`8.7.56.181-.184`",
            "ROADMAP must already expose the current complement-basis residual branch.",
        ),
    ]
    source_pack_ready = all(item["present"] for item in source_targets)

    photon_branch_ready = bool(projector_audit["summary"]["transverse_photon_branch_identification_available"])
    photon_subtraction_ready = bool(projector_audit["summary"]["transverse_photon_subtraction_ready"])
    pi_mu_definition_ready = source_targets[4]["present"]
    propagator_mix_pack_ready = source_targets[2]["present"] and source_targets[3]["present"]
    ghost_free_guard_ready = source_targets[5]["present"]
    old_three_sector_split_ready = bool(vev_quadratic["summary"]["three_sector_split_available"])

    source_inventory = payload(
        "8.7.56.182",
        "Working-action temporal/longitudinal/Stueckelberg basis source inventory",
        common_inputs,
        "Freeze the source pack needed to decide whether current canon canonizes the nontransverse complement basis after the transverse photon branch is removed.",
        {
            "photon_branch_rule": "A_mu = delta P_mu^T / sqrt(Z_P) freezes the transverse photon branch under the breakthrough working action",
            "complement_rule": "the remaining basis must be built from delta P_0, delta P_i^L, and the Stückelberg scalar pi",
            "pi_mu_rule": "Pi_mu := P_mu - partial_mu pi / m_P is the natural gauge-invariant hint, but a complement basis still requires an explicit mode-level interpretation after photon extraction",
            "propagator_rule": "the old propagator and gauge-fixing pieces may be reused only as source hints, not as already-reinterpreted working-action basis statements",
        },
        [
            row(
                "working_action_nontransverse_basis_source_inventory_complete",
                "pass",
                "working-action temporal/longitudinal/Stueckelberg basis source inventory complete",
                1,
                "The complement-basis source pack is frozen.",
            ),
            row(
                "working_action_nontransverse_basis_source_pack_ready",
                "pass" if source_pack_ready else "reject",
                "working-action nontransverse basis source pack ready",
                1 if source_pack_ready else 0,
                "The residual branch has the required canonical anchors.",
            ),
            row(
                "working_action_nontransverse_basis_photon_branch_ready",
                "pass" if photon_branch_ready else "reject",
                "transverse photon branch ready before complement-basis inventory",
                1 if photon_branch_ready else 0,
                "The complement-basis route starts only after the photon branch is frozen.",
            ),
            row(
                "working_action_nontransverse_basis_pi_mu_definition_ready",
                "pass" if pi_mu_definition_ready else "reject",
                "Pi_mu definition ready",
                1 if pi_mu_definition_ready else 0,
                "Part I still exposes the gauge-invariant Stückelberg combination as a source hint.",
            ),
            row(
                "working_action_nontransverse_basis_propagator_mix_pack_ready",
                "pass" if propagator_mix_pack_ready else "reject",
                "propagator mix pack ready",
                1 if propagator_mix_pack_ready else 0,
                "The old propagator still exposes the massive and gauge-dependent k_mu k_nu pieces.",
            ),
            row(
                "working_action_nontransverse_basis_ghost_free_guard_ready",
                "pass" if ghost_free_guard_ready else "reject",
                "ghost-free guard ready",
                1 if ghost_free_guard_ready else 0,
                "Any future complement basis must preserve the ghost-free closure.",
            ),
        ],
        {
            "working_action_temporal_longitudinal_stueckelberg_basis_source_pack_ready": source_pack_ready,
            "transverse_photon_branch_ready": photon_branch_ready,
            "transverse_photon_subtraction_ready": photon_subtraction_ready,
            "pi_mu_definition_ready": pi_mu_definition_ready,
            "propagator_mix_pack_ready": propagator_mix_pack_ready,
            "old_three_sector_split_ready": old_three_sector_split_ready,
            "ghost_free_guard_ready": ghost_free_guard_ready,
            "first_route_to_close_or_none": "working_action_temporal_longitudinal_stueckelberg_basis_identification_audit",
        },
        {
            "overall_status": "working_action_nontransverse_basis_source_inventory_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_183": True,
            "next_required_artifacts": ["working_action_temporal_longitudinal_stueckelberg_basis_identification_audit"],
        },
        {
            "inventory_targets": source_targets,
            "projector_source_summary": projector_source["summary"],
            "projector_audit_summary": projector_audit["summary"],
            "breakthrough_vev_summary": breakthrough_vev["summary"],
            "breakthrough_maxwell_summary": breakthrough_maxwell["summary"],
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    pi_mu_hint_available = pi_mu_definition_ready
    complement_basis_available = False
    nontransverse_quadratic_diagonalization_available = False
    massive_sector_projector_available = False

    identification_audit = payload(
        "8.7.56.183",
        "Working-action temporal/longitudinal/Stueckelberg basis identification audit",
        common_inputs,
        "Audit whether current canon actually diagonalizes or canonically identifies the nontransverse complement basis after photon extraction.",
        {
            "pi_mu_hint_rule": "Pi_mu supplies a gauge-invariant hint but does not by itself specify the complement basis after the transverse branch has been removed",
            "diagonalization_rule": "a canonical complement basis needs an explicit statement of how temporal, longitudinal, and Stückelberg fluctuations diagonalize or recombine under the working action",
            "projector_rule": "without that diagonalized complement basis, no current-canon massive-sector projector can be written",
        },
        [
            row(
                "working_action_nontransverse_basis_pi_mu_hint_available",
                "pass" if pi_mu_hint_available else "reject",
                "Pi_mu gauge-invariant hint available",
                1 if pi_mu_hint_available else 0,
                "Current canon does expose Pi_mu as a gauge-invariant combination.",
            ),
            row(
                "working_action_nontransverse_basis_photon_branch_ready_after_audit",
                "pass" if photon_branch_ready and photon_subtraction_ready else "reject",
                "photon branch ready before complement-basis audit",
                1 if photon_branch_ready and photon_subtraction_ready else 0,
                "The complement basis is being audited only after the photon branch has been identified and subtracted.",
            ),
            row(
                "working_action_nontransverse_basis_identification_available",
                "reject",
                "working-action temporal/longitudinal/Stueckelberg complement basis available",
                0,
                "Current canon still does not identify a canonical complement basis built from temporal, longitudinal, and Stückelberg fluctuations.",
            ),
            row(
                "working_action_nontransverse_quadratic_diagonalization_available",
                "reject",
                "working-action nontransverse quadratic diagonalization available",
                0,
                "No current-canon statement diagonalizes the nontransverse complement after the transverse photon is removed.",
            ),
            row(
                "working_action_massive_sector_projector_available_after_basis_audit",
                "reject",
                "working-action massive-sector projector available after basis audit",
                0,
                "Without the nontransverse diagonalization, the projector still cannot be written.",
            ),
        ],
        {
            "pi_mu_gauge_invariant_hint_available": pi_mu_hint_available,
            "transverse_photon_branch_ready": photon_branch_ready,
            "transverse_photon_subtraction_ready": photon_subtraction_ready,
            "working_action_temporal_longitudinal_stueckelberg_mode_basis_available": complement_basis_available,
            "working_action_nontransverse_quadratic_diagonalization_available": nontransverse_quadratic_diagonalization_available,
            "working_action_massive_sector_projector_available": massive_sector_projector_available,
            "identification_nonclosure_reason_or_none": "working_action_nontransverse_quadratic_diagonalization_absent",
            "first_route_to_close_or_none": "working_action_vector_rebuild_reopen_fifth_gate",
        },
        {
            "overall_status": "working_action_nontransverse_basis_identification_nonclosure_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_184": True,
            "next_required_artifacts": ["working_action_vector_rebuild_reopen_fifth_gate"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "part1_pi_mu_line": hit(part1, "\\Pi_\\mu:=P_\\mu-\\partial_\\mu\\pi/m_P"),
            "part1_propagator_massive_line": hit(part1, "\\frac{\\eta_{\\mu\\nu}-k_\\mu k_\\nu/m_P^2}{k^2-m_P^2+i0}"),
            "part1_propagator_gauge_line": hit(part1, "\\frac{\\xi_g\\,k_\\mu k_\\nu/m_P^2}{k^2-\\xi_g m_P^2+i0}"),
            "part1_ghost_free_line": hit(part1, "負ノルム（ghost）モードは出現しない。"),
            "part3a_case_b_line": hit(part3a, "A棄却、B採用"),
            "breakthrough_maxwell_formulas": breakthrough_maxwell["formulas"],
        },
    )

    reopen_gate = payload(
        "8.7.56.184",
        "Working-action vector rebuild reopen fifth gate / Trial-3 fallback fifth refresh",
        common_inputs,
        "Integrate the complement-basis audit and decide whether the working-action vector rebuild can reopen or whether a deeper nontransverse diagonalization route is required.",
        {
            "reopen_rule": "the vector rebuild reopens only if the nontransverse complement basis is canonically identified and diagonalized",
            "anchor_rule": "anchor refresh remains blocked while the projector and radial eigenoperator still depend on the absent diagonalization",
            "fallback_rule": "Trial-3 remains on fallback hold while the same nontransverse diagonalization issue stays unresolved",
        },
        [
            row(
                "working_action_vector_rebuild_fifth_gate_source_inventory_ready",
                "pass" if source_inventory["summary"]["working_action_temporal_longitudinal_stueckelberg_basis_source_pack_ready"] else "reject",
                "working-action complement-basis source inventory ready at fifth gate",
                1 if source_inventory["summary"]["working_action_temporal_longitudinal_stueckelberg_basis_source_pack_ready"] else 0,
                "The complement-basis residual branch has its source pack frozen.",
            ),
            row(
                "working_action_vector_rebuild_fifth_gate_basis_identification_ready",
                "reject",
                "working-action complement-basis identification ready at fifth gate",
                0,
                "The canonical complement basis is still missing.",
            ),
            row(
                "working_action_vector_rebuild_fifth_gate_anchor_refresh_ready",
                "reject",
                "working-action vector anchor refresh ready at fifth gate",
                0,
                "Anchor refresh remains blocked until the nontransverse diagonalization is current-canon ready.",
            ),
            row(
                "working_action_vector_rebuild_fifth_gate_trial3_hold_retained",
                "pass",
                "Trial-3 fallback hold retained at fifth gate",
                1,
                "The weak-sector branch remains downstream of the unresolved nontransverse diagonalization.",
            ),
        ],
        {
            "working_action_vector_rebuild_reopen_ready": False,
            "working_action_temporal_longitudinal_stueckelberg_basis_identification_ready": False,
            "working_action_massive_sector_projector_identification_ready": False,
            "working_action_vector_mass_spectrum_reduced_solver_numeric_ready": False,
            "working_action_vector_mass_spectrum_anchor_refresh_ready": False,
            "trial2_paper_side_sync_deferred_until_vector_anchor_refresh": True,
            "trial3_fallback_hold_retained": True,
            "trial3_fallback_hold_release_ready": False,
            "recommended_next_route_or_none": "8.7.56.185",
        },
        {
            "overall_status": "working_action_vector_rebuild_fifth_gate_nonclosure_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_185": True,
            "next_required_artifacts": ["working_action_nontransverse_quadratic_diagonalization_identification"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "trial2_declaration_summary": trial2_declaration["summary"],
            "vector_reduced_solver_summary": vector_reduced_solver["summary"],
            "vector_anchor_summary": vector_anchor["summary"],
            "trial3_fallback_route_summary": trial3_fallback_route["summary"],
            "reopen_fourth_summary": reopen_fourth["summary"],
        },
    )

    route_contract = payload(
        "8.7.56.185",
        "Working-action nontransverse quadratic diagonalization route contract",
        common_inputs,
        "Freeze the deeper residual route suggested by the complement-basis audit: identify the diagonalized nontransverse sector after the transverse photon branch has been removed.",
        {
            "selected_residual_route": "working_action_nontransverse_quadratic_diagonalization_identification",
            "blocking_rule": "Pi_mu and the old propagator hint at the complement sector, but current canon still lacks the diagonalized nontransverse basis needed for a projector",
            "dependency_rule": "vector rebuild, anchor refresh, Trial-2 paper-side sync, and Trial-3 fallback release all remain downstream of the same missing diagonalization",
        },
        [
            row(
                "working_action_nontransverse_diagonalization_route_contract_complete",
                "pass",
                "working-action nontransverse quadratic diagonalization route contract complete",
                1,
                "The next deeper residual route is frozen after the complement-basis non-closure.",
            ),
            row(
                "working_action_nontransverse_diagonalization_missing",
                "pass",
                "working-action nontransverse quadratic diagonalization missing",
                1,
                "The missing artifact is the canonical diagonalization of the temporal/longitudinal/Stueckelberg complement sector.",
            ),
            row(
                "working_action_nontransverse_diagonalization_trial3_dependency_blocked",
                "pass",
                "Trial-3 dependency still blocked by nontransverse diagonalization",
                1,
                "The weak-sector branch remains blocked by the unresolved nontransverse diagonalization.",
            ),
            row(
                "working_action_nontransverse_diagonalization_paper_sync_deferred",
                "pass",
                "Trial-2 paper-side sync still deferred by nontransverse diagonalization",
                1,
                "Paper-side sync remains deferred until vector anchor refresh becomes current-canon ready.",
            ),
        ],
        {
            "selected_residual_route": "working_action_nontransverse_quadratic_diagonalization_identification",
            "missing_v2_artifact": "working_action_nontransverse_quadratic_diagonalization",
            "trial3_dependency_state": "blocked_by_working_action_nontransverse_quadratic_diagonalization",
            "trial2_paper_side_sync_state": "deferred_until_working_action_vector_anchor_refresh",
            "split_contract_ready": True,
            "recommended_next_route_or_none": "8.7.56.186",
        },
        {
            "overall_status": "working_action_nontransverse_quadratic_diagonalization_route_contract_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_186": True,
            "next_required_artifacts": [
                "working_action_nontransverse_quadratic_diagonalization_source_inventory",
                "working_action_nontransverse_quadratic_diagonalization_identification_audit",
                "working_action_vector_rebuild_reopen_sixth_gate",
            ],
        },
        {
            "reopen_gate_summary": reopen_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "projector_route_summary": projector_route["summary"],
            "trial3_fallback_route_summary": trial3_fallback_route["summary"],
            "part1_pi_mu_line": hit(part1, "\\Pi_\\mu:=P_\\mu-\\partial_\\mu\\pi/m_P"),
        },
    )

    write_artifact("mass_origin_v2_working_action_temporal_longitudinal_stueckelberg_basis_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_working_action_temporal_longitudinal_stueckelberg_basis_identification_audit", identification_audit)
    write_artifact("mass_origin_v2_vector_rebuild_reopen_fifth_gate", reopen_gate)
    write_artifact("mass_origin_v2_working_action_nontransverse_quadratic_diagonalization_route_contract", route_contract)

    print("[ok] wrote:")
    print(" - mass_origin_v2_working_action_temporal_longitudinal_stueckelberg_basis_source_inventory_metrics.json")
    print(" - mass_origin_v2_working_action_temporal_longitudinal_stueckelberg_basis_identification_audit_metrics.json")
    print(" - mass_origin_v2_vector_rebuild_reopen_fifth_gate_metrics.json")
    print(" - mass_origin_v2_working_action_nontransverse_quadratic_diagonalization_route_contract_metrics.json")


# Function: run the temporal/longitudinal/Stueckelberg basis branch from the CLI.

if __name__ == "__main__":
    main()
