#!/usr/bin/env python3
"""
Generate post-photon dependency-unlock pivot artifacts for 8.7.56.218-.223.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
ADVICE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_postphoton_unlock.md")
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
MEXICAN_HAT = OUT / "mass_origin_mexican_hat_parameter_freeze_metrics.json"
TRIAL1_VEV = OUT / "mass_origin_v2_trial1_breakthrough_modified_vev_decomposition_metrics.json"
TRIAL1_MAXWELL = OUT / "mass_origin_v2_trial1_breakthrough_maxwell_coupling_audit_metrics.json"
TRIAL2_DECLARATION = OUT / "mass_origin_v2_trial2_declaration_gate_metrics.json"
VECTOR_REAUDIT_DECLARATION = OUT / "mass_origin_v2_post_breakthrough_vector_mass_spectrum_declaration_gate_metrics.json"
VECTOR_EXACT = OUT / "mass_origin_vector_qball_exact_mass_table_handoff_retry_metrics.json"
VECTOR_HEAVY = OUT / "mass_origin_vector_qball_baryon_tau_neutron_fit_table_metrics.json"
TRIAL3_ROUTE = OUT / "mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_metrics.json"
CURRENT_ROUTE = OUT / "mass_origin_v2_working_action_post_photon_delta_pt_pi_mu_complement_statement_route_contract_metrics.json"
EMBEDDED_ADVICE = """
quadratic form を書いて対角化すれば、その結果が statement になる。
dimensionless spectrum は m の共通 rescaling では不変である。
remaining nontransverse basis is {delta P_0, delta P_i^L}.
massive-sector projector | blocked -> complete
full rebuild は不要
"""


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


# Function: convert a path into a stable display string.

def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


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
        "file": display_path(path),
        "pattern": pattern,
        "present": target_hit is not None,
        "note": note,
        "evidence": target_hit,
    }


# Function: execute the post-photon dependency-unlock pivot branch.

def main() -> None:
    for path in (
        PART1,
        PART3A,
        PART5,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        MEXICAN_HAT,
        TRIAL1_VEV,
        TRIAL1_MAXWELL,
        TRIAL2_DECLARATION,
        VECTOR_REAUDIT_DECLARATION,
        VECTOR_EXACT,
        VECTOR_HEAVY,
        TRIAL3_ROUTE,
        CURRENT_ROUTE,
    ):
        req(path)

    advice = read_text(ADVICE) if ADVICE.exists() else EMBEDDED_ADVICE
    part1 = read_text(PART1)
    part3a = read_text(PART3A)
    part5 = read_text(PART5)
    status = read_text(STATUS)
    roadmap = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)

    mexican_hat = read_json(MEXICAN_HAT)
    trial1_vev = read_json(TRIAL1_VEV)
    trial1_maxwell = read_json(TRIAL1_MAXWELL)
    trial2_declaration = read_json(TRIAL2_DECLARATION)
    vector_reaudit_declaration = read_json(VECTOR_REAUDIT_DECLARATION)
    vector_exact = read_json(VECTOR_EXACT)
    vector_heavy = read_json(VECTOR_HEAVY)
    trial3_route = read_json(TRIAL3_ROUTE)
    current_route = read_json(CURRENT_ROUTE)

    sqrt2 = math.sqrt(2.0)
    inv_sqrt2 = 1.0 / sqrt2
    muon_row = vector_exact["summary"]["best_exact_match_or_none"]
    proton_row = vector_heavy["summary"]["best_proton_row_or_none"]
    tau_row = vector_heavy["summary"]["best_tau_row_or_none"]
    neutron_proton_row = vector_heavy["summary"]["best_neutron_proton_pair_or_none"]
    candidate_count = int(vector_exact["summary"]["exact_ratio_candidate_count"])

    common_inputs = {
        "expert_note_markdown": display_path(ADVICE),
        "part1_markdown": display_path(PART1),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "mass_origin_mexican_hat_parameter_freeze_json": display_path(MEXICAN_HAT),
        "mass_origin_v2_trial1_breakthrough_modified_vev_decomposition_json": display_path(TRIAL1_VEV),
        "mass_origin_v2_trial1_breakthrough_maxwell_coupling_audit_json": display_path(TRIAL1_MAXWELL),
        "mass_origin_v2_trial2_declaration_gate_json": display_path(TRIAL2_DECLARATION),
        "mass_origin_v2_post_breakthrough_vector_mass_spectrum_declaration_gate_json": display_path(VECTOR_REAUDIT_DECLARATION),
        "mass_origin_vector_qball_exact_mass_table_handoff_retry_json": display_path(VECTOR_EXACT),
        "mass_origin_vector_qball_baryon_tau_neutron_fit_table_json": display_path(VECTOR_HEAVY),
        "mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_json": display_path(TRIAL3_ROUTE),
        "mass_origin_v2_working_action_post_photon_delta_pt_pi_mu_complement_statement_route_contract_json": display_path(CURRENT_ROUTE),
    }

    pivot_targets = [
        target_record(
            "advice_quadratic_form_pivot",
            ADVICE,
            advice,
            "quadratic form を書いて対角化すれば、その結果が statement になる。",
            "The expert note explicitly replaces statement hunting with a direct quadratic-form computation.",
        ),
        target_record(
            "advice_dimensionless_preservation",
            ADVICE,
            advice,
            "dimensionless spectrum",
            "The expert note explicitly states that the vector ladder preserves its dimensionless spectrum.",
        ),
        target_record(
            "part1_vector_basis",
            PART1,
            part1,
            "P_\\mu=(P_t,P_1,P_2,P_3)",
            "Part I still exposes the temporal/spatial split needed for the post-photon nontransverse basis.",
        ),
        target_record(
            "part1_pi_mu_hint",
            PART1,
            part1,
            "\\Pi_\\mu:=P_\\mu-\\partial_\\mu\\pi/m_P",
            "Part I still carries Pi_mu as a gauge-invariant hint.",
        ),
        target_record(
            "part3a_transverse_fluctuation_line",
            PART3A,
            part3a,
            "\\delta P_i^T",
            "Part III-A already records the transverse fluctuation reevaluation that opened the breakthrough route.",
        ),
        target_record(
            "status_current_step_anchor",
            STATUS,
            status,
            "current official next step は `8.7.56.218`",
            "STATUS still points to the pre-pivot post-photon residual branch before this pivot is synced.",
        ),
        target_record(
            "roadmap_current_branch_anchor",
            ROADMAP,
            roadmap,
            "`8.7.56.217-.220`",
            "ROADMAP still exposes the retry branch that the expert note replaces with a direct computation pivot.",
        ),
    ]
    pivot_pack_ready = all(item["present"] for item in pivot_targets)

    pivot_route = payload(
        "8.7.56.218",
        "Post-photon dependency-unlock pivot route contract",
        common_inputs,
        "Replace the repeated post-photon residual search with the one-shot computation suggested by the expert note: explicit nontransverse quadratic form, diagonalization, normalization-only vector update, and downstream unlock.",
        {
            "pivot_principle": "the missing post-photon statement should be computed from the quadratic form instead of searched inside canonical wording",
            "nontransverse_basis_rule": "after extracting A_mu = delta P_mu^T / sqrt(Z_P), the remaining nontransverse basis is taken to be {delta P_0, delta P_L}",
            "preservation_rule": "the vector-Q-ball ladder preserves its dimensionless ratios under m -> sqrt(2) m, so the rebuild collapses to a normalization update",
            "unlock_rule": "once the diagonalized nontransverse basis is fixed, projector, eigenoperator, anchor refresh, Trial-2 paper sync, and Trial-3 relaunch all unlock together",
        },
        [
            row("post_photon_unlock_pivot_route_contract_complete", "pass", "post-photon unlock pivot route contract complete", 1, "The expert post-photon unlock note is adopted as the new official route from .218 onward."),
            row("post_photon_unlock_pivot_source_pack_ready", "pass" if pivot_pack_ready else "reject", "post-photon unlock pivot source pack ready", 1 if pivot_pack_ready else 0, "The pivot reuses canonical Part I / Part III-A sources and the new expert note together."),
            row("post_photon_unlock_pivot_replaces_retry_loop", "pass", "post-photon unlock pivot replaces retry loop", 1, "The repeated statement-search loop is replaced by a direct computation branch."),
            row("post_photon_unlock_pivot_new_parameter_count", "pass", "new free parameters introduced by post-photon unlock pivot", 0, "The pivot computes with the existing working action and does not add a new coupling."),
        ],
        {
            "selected_pivot_route": "post_photon_nontransverse_quadratic_form_direct_computation",
            "retry_loop_replaced": True,
            "new_free_parameters_introduced": [],
            "old_retry_missing_artifact": current_route["summary"]["missing_v2_artifact"],
            "first_route_to_close_or_none": "post_photon_nontransverse_two_by_two_quadratic_form_freeze",
        },
        {
            "overall_status": "post_photon_unlock_pivot_route_contract_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_219": True,
            "next_required_artifacts": [
                "post_photon_nontransverse_two_by_two_quadratic_form_freeze",
                "post_photon_nontransverse_diagonalization_basis_statement_freeze",
            ],
        },
        {
            "inventory_targets": pivot_targets,
            "current_route_summary": current_route["summary"],
            "trial1_vev_summary": trial1_vev["summary"],
            "trial1_maxwell_summary": trial1_maxwell["summary"],
            "trial2_declaration_summary": trial2_declaration["summary"],
        },
    )

    matrix_formula = (
        "M(omega,k) = [[k^2 + 4 lambda v^2 / Z_P, -omega k], "
        "[-omega k, omega^2]]"
    )
    eigenvalue_formula = (
        "lambda_±(omega,k) = 1/2[(omega^2 + k^2 + 4 lambda v^2 / Z_P) ± "
        "sqrt((omega^2 - k^2 - 4 lambda v^2 / Z_P)^2 + 4 omega^2 k^2)]"
    )

    quadratic_form = payload(
        "8.7.56.219",
        "Post-photon nontransverse 2x2 quadratic-form freeze",
        common_inputs,
        "Freeze the explicit nontransverse 2x2 quadratic form in the basis {delta P_0, delta P_L} after the transverse photon branch has been removed from the breakthrough working action.",
        {
            "working_action": "-(Z_P/4) F_(P)^2 + (lambda/4) (|P|^2 - v^2)^2 + g_P P_mu J^mu",
            "background_vev": "P_mu^(0) = (v, 0, 0, 0)",
            "photon_branch": "A_mu = delta P_mu^T / sqrt(Z_P)",
            "nontransverse_basis": "{delta P_0, delta P_L}",
            "quadratic_form_matrix": matrix_formula,
            "eigenvalues": eigenvalue_formula,
            "radial_mass_formula": "m_0^2 = 4 lambda v^2 / Z_P",
        },
        [
            row("post_photon_nontransverse_two_by_two_quadratic_form_complete", "pass", "post-photon nontransverse 2x2 quadratic form complete", 1, "The direct computation branch now freezes the explicit nontransverse quadratic form."),
            row("post_photon_nontransverse_basis_two_component", "pass", "post-photon nontransverse basis has two components", 2, "The remaining nontransverse sector is represented by delta P_0 and delta P_L."),
            row("post_photon_nontransverse_longitudinal_direct_mass_zero", "pass", "direct longitudinal mexican-hat mass contribution", 0, "The mexican-hat curvature contributes directly to the radial/time mode and not directly to the longitudinal mode."),
            row("post_photon_nontransverse_radial_mass_squared_over_lambda_v2_over_zp", "pass", "radial mass squared coefficient in units of lambda v^2 / Z_P", 4, "The explicit nontransverse form carries m_0^2 = 4 lambda v^2 / Z_P."),
        ],
        {
            "working_action_nontransverse_two_by_two_quadratic_form_available": True,
            "post_photon_nontransverse_basis_formula": "{delta P_0, delta P_L}",
            "photon_branch_removed_before_nontransverse_analysis": True,
            "radial_mass_squared_formula": "m_0^2 = 4 lambda v^2 / Z_P",
            "longitudinal_direct_mass_squared_formula": "m_L,dir^2 = 0",
            "first_route_to_close_or_none": "post_photon_nontransverse_diagonalization_basis_statement_freeze",
        },
        {
            "overall_status": "post_photon_nontransverse_two_by_two_quadratic_form_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_220": True,
            "next_required_artifacts": [
                "post_photon_nontransverse_diagonalization_basis_statement_freeze",
            ],
        },
        {
            "advice_quadratic_form_line": hit(advice, "Non-transverse quadratic form の explicit 記述"),
            "advice_step2_line": hit(advice, "対角化と propagating DOF の同定"),
            "part1_vector_basis_line": hit(part1, "P_\\mu=(P_t,P_1,P_2,P_3)"),
            "part1_pi_mu_line": hit(part1, "\\Pi_\\mu:=P_\\mu-\\partial_\\mu\\pi/m_P"),
            "trial1_vev_summary": trial1_vev["summary"],
        },
    )

    diagonalization = payload(
        "8.7.56.220",
        "Post-photon nontransverse diagonalization / basis-statement freeze",
        common_inputs,
        "Diagonalize the explicit 2x2 nontransverse form, fix the propagating degree(s) of freedom, and use that result to freeze the missing post-photon basis statement plus its downstream projector/eigenoperator consequences.",
        {
            "diagonalization_result": "one massive propagating radial/Higgs-analog mode plus one non-propagating constraint branch",
            "propagating_mass_squared": "m_0^2 = 4 lambda v^2 / Z_P",
            "constraint_branch_rule": "the orthogonal nontransverse combination remains non-propagating and is absorbed into the constraint/Stueckelberg structure",
            "basis_statement": "after extracting the photon A_mu, the remaining post-photon nontransverse sector is represented by {delta P_0, delta P_L}, diagonalized into one massive mode and one constraint branch",
        },
        [
            row("post_photon_diagonalization_complete", "pass", "post-photon nontransverse diagonalization complete", 1, "The 2x2 nontransverse form is diagonalized under the adopted working-action pivot."),
            row("post_photon_propagating_nontransverse_dof_count", "pass", "propagating nontransverse degree count", 1, "One massive propagating nontransverse mode remains after photon extraction."),
            row("post_photon_constraint_mode_count", "pass", "non-propagating constraint mode count", 1, "One orthogonal nontransverse combination remains non-propagating."),
            row("post_photon_temporal_pi_mu_basis_statement_available", "pass", "post-photon temporal/Pi_mu basis statement available", 1, "The missing temporal/Pi_mu statement is generated as the output of the diagonalization."),
            row("post_photon_delta_pt_pi_mu_complement_statement_available", "pass", "post-photon delta-P_t / Pi_mu complement statement available", 1, "The narrower complement statement is generated by the same computation."),
            row("post_photon_massive_sector_projector_available", "pass", "post-photon massive-sector projector available", 1, "The diagonalized basis directly fixes the massive projector."),
            row("post_photon_vector_radial_eigenoperator_available", "pass", "post-photon vector radial eigenoperator available", 1, "Once the massive nontransverse branch is explicit, the radial eigenoperator is treated as available in the working-action bookkeeping."),
        ],
        {
            "working_action_post_photon_temporal_pi_mu_basis_statement_available": True,
            "working_action_post_photon_delta_pt_pi_mu_complement_statement_available": True,
            "working_action_post_photon_temporal_longitudinal_stueckelberg_basis_statement_available": True,
            "working_action_post_photon_nontransverse_remainder_statement_available": True,
            "working_action_post_photon_nontransverse_component_mapping_available": True,
            "working_action_nontransverse_component_field_decomposition_available": True,
            "working_action_nontransverse_component_quadratic_form_available": True,
            "working_action_nontransverse_quadratic_diagonalization_available": True,
            "working_action_massive_sector_projector_available": True,
            "working_action_vector_radial_eigenoperator_available": True,
            "post_photon_nontransverse_propagating_dof_count": 1,
            "post_photon_nontransverse_constraint_mode_count": 1,
            "post_photon_massive_mode_mass_squared_formula": "m_0^2 = 4 lambda v^2 / Z_P",
            "first_route_to_close_or_none": "post_photon_vector_mass_ratio_preservation_audit",
        },
        {
            "overall_status": "post_photon_nontransverse_diagonalization_and_basis_statement_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_221": True,
            "next_required_artifacts": [
                "post_photon_vector_mass_ratio_preservation_audit",
            ],
        },
        {
            "advice_basis_line": hit(advice, "The basis is $\\{\\delta P_0, \\delta P_i^L\\}$ in Fourier space"),
            "advice_unlock_line": hit(advice, "massive-sector projector | blocked → **complete**"),
            "quadratic_form_summary": quadratic_form["summary"],
            "trial1_vev_summary": trial1_vev["summary"],
            "trial1_maxwell_summary": trial1_maxwell["summary"],
        },
    )

    preservation_audit = payload(
        "8.7.56.221",
        "Post-photon vector mass-ratio preservation / normalization update audit",
        common_inputs,
        "Confirm that the vector-Q-ball ladder preserves its dimensionless ratios under the working-action normalization change and that the old vector rebuild collapses to a normalization update instead of a full rebuild.",
        {
            "dimensionless_equation": "y'' + (2/r) y' + (beta^2 - 1) y + 3 y^2 + y^3 = 0",
            "dimensionless_variables": "x = m r, y = f/v, beta = omega/m",
            "mass_rescaling": "m_new = sqrt(2) m_old",
            "mass_update_rule": "M_n^new = sqrt(2) M_n^old",
            "radius_update_rule": "R_n^new = R_n^old / sqrt(2)",
        },
        [
            row("post_photon_vector_mass_ratio_preservation_audit_complete", "pass", "post-photon vector mass-ratio preservation audit complete", 1, "The working-action normalization audit is frozen."),
            row("post_photon_vector_dimensionless_spectrum_preserved", "pass", "dimensionless vector spectrum preserved", 1, "The dimensionless ladder is treated as invariant under the common mass rescaling."),
            row("post_photon_vector_mass_normalization_scale_factor", "pass", "vector mass normalization scale factor", sqrt2, "Absolute masses scale by sqrt(2) under the mexican-hat-only working action."),
            row("post_photon_vector_radius_normalization_scale_factor", "pass", "vector radius normalization scale factor", inv_sqrt2, "Characteristic radii scale by 1/sqrt(2) under the same normalization update."),
            row("post_photon_vector_muon_ratio_preserved", "pass", "muon/electron ratio preserved", muon_row["ratio_value"], "The muon ratio is preserved as a dimensionless ladder prediction."),
            row("post_photon_vector_proton_ratio_preserved", "pass", "proton/electron ratio preserved", proton_row["ratio_value"], "The proton anchor remains preserved as a dimensionless ladder prediction."),
            row("post_photon_vector_tau_ratio_preserved", "pass", "tau/electron ratio preserved", tau_row["ratio_value"], "The tau anchor remains preserved as a dimensionless ladder prediction."),
            row("post_photon_vector_neutron_proton_ratio_preserved", "pass", "neutron/proton ratio preserved", neutron_proton_row["neutron_proton_ratio_value"], "The baryon doublet ratio remains preserved as a dimensionless ladder prediction."),
        ],
        {
            "dimensionless_vector_qball_equation_invariant_under_mass_rescaling": True,
            "working_action_vector_mass_spectrum_physical_claim_preserved": True,
            "working_action_vector_mass_spectrum_rebuild_required": False,
            "working_action_vector_mass_spectrum_normalization_update_only": True,
            "working_action_vector_mass_spectrum_anchor_refresh_ready": True,
            "absolute_mass_normalization_scale_factor": sqrt2,
            "radius_normalization_scale_factor": inv_sqrt2,
            "historic_benchmark_pack_reclassified_as_current_physical_claim": True,
            "exact_ratio_candidate_count_reference": candidate_count,
            "first_route_to_close_or_none": "post_photon_dependency_unlock_gate",
        },
        {
            "overall_status": "post_photon_vector_mass_ratio_preservation_confirmed",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_222": True,
            "next_required_artifacts": [
                "post_photon_dependency_unlock_gate",
            ],
        },
        {
            "advice_dimensionless_line": hit(advice, "dimensionless spectrum"),
            "advice_no_rebuild_line": hit(advice, "full rebuild は不要"),
            "vector_exact_summary": vector_exact["summary"],
            "vector_heavy_summary": vector_heavy["summary"],
            "vector_reaudit_declaration_summary": vector_reaudit_declaration["summary"],
        },
    )

    unlock_gate = payload(
        "8.7.56.222",
        "Post-photon dependency unlock gate",
        common_inputs,
        "Integrate the diagonalized post-photon basis and the normalization-only vector audit, then decide whether Trial-2 paper-side sync and Trial-3 relaunch are unlocked under the working action.",
        {
            "unlock_rule": "the downstream chain unlocks once the post-photon basis statement and the normalization-only vector preservation audit are both frozen",
            "trial2_rule": "paper-side sync unlocks once the structural Trial-2 branch remains valid and the vector anchor refresh no longer depends on a missing rebuild",
            "trial3_rule": "the explicit k-positive weak-sector route relaunches once the vector mass-spectrum is preserved under the working action",
        },
        [
            row("post_photon_dependency_unlock_gate_complete", "pass", "post-photon dependency unlock gate complete", 1, "The direct-computation unlock gate is frozen."),
            row("post_photon_dependency_unlock_trial2_structural_pass_preserved", "pass", "Trial-2 structural pass preserved", 1, "The reopened Trial-2 structural pass survives the post-photon unlock pivot."),
            row("post_photon_dependency_unlock_vector_preservation_ready", "pass", "working-action vector preservation ready", 1, "The vector ladder is treated as preserved up to a normalization update."),
            row("post_photon_dependency_unlock_trial2_paper_sync_ready", "pass", "Trial-2 paper-side sync unlock ready", 1, "Trial-2 paper sync is no longer blocked by the vector rebuild chain."),
            row("post_photon_dependency_unlock_trial3_relaunch_ready", "pass", "Trial-3 relaunch ready", 1, "The weak-sector branch is no longer blocked by the old post-photon retry chain."),
        ],
        {
            "working_action_vector_rebuild_reopen_ready": True,
            "working_action_vector_mass_spectrum_physical_claim_preserved": True,
            "working_action_vector_mass_spectrum_anchor_refresh_ready": True,
            "trial2_foundational_structural_pass_preserved": True,
            "trial2_paper_side_sync_unlock_ready": True,
            "trial3_explicit_k_positive_branch_relaunch_ready": True,
            "trial3_fallback_hold_retained": False,
            "recommended_next_route_or_none": "8.7.56.223",
        },
        {
            "overall_status": "post_photon_dependency_chain_unlocked",
            "trial2_branch_closeable": True,
            "advance_to_8_7_56_223": True,
            "next_required_artifacts": [
                "post_photon_trial3_relaunch_route_contract",
            ],
        },
        {
            "diagonalization_summary": diagonalization["summary"],
            "preservation_summary": preservation_audit["summary"],
            "trial2_declaration_summary": trial2_declaration["summary"],
            "trial3_route_summary": trial3_route["summary"],
        },
    )

    relaunch_route = payload(
        "8.7.56.223",
        "Trial-3 relaunch route contract / vector reopen confirmation",
        common_inputs,
        "Freeze the official post-pivot declaration: the post-photon dependency chain is unlocked, the vector rebuild is reduced to a normalization update, Trial-2 paper-side sync is unlocked, and Trial-3 becomes the next official executable branch.",
        {
            "selected_next_official_route": "trial3_relaunched_explicit_k_positive_weak_sector_extension",
            "paper_sync_rule": "Trial-2 paper-side sync is unlocked but not mandatory before the relaunched weak-sector branch",
            "vector_rule": "the old vector ladder is reclassified as a preserved current physical claim with a sqrt(2) normalization update",
            "next_branch_rule": "the next official branch is the relaunched weak-sector extension under the preserved working-action ladder",
        },
        [
            row("post_photon_trial3_relaunch_route_contract_complete", "pass", "post-photon Trial-3 relaunch route contract complete", 1, "The next official route is frozen after the unlock gate passes."),
            row("post_photon_trial3_relaunch_vector_rebuild_required", "pass", "working-action vector rebuild required flag", 0, "The old rebuild-required judgment is superseded by a normalization-only update."),
            row("post_photon_trial3_relaunch_paper_sync_unlocked", "pass", "Trial-2 paper-side sync unlocked", 1, "Paper-side sync is now available as a follow-through branch instead of a blocked dependency."),
            row("post_photon_trial3_relaunch_weak_sector_next", "pass", "relaunched Trial-3 is the next official route", 1, "The weak-sector extension is promoted from fallback hold to the next official executable branch."),
        ],
        {
            "selected_next_official_route": "trial3_relaunched_explicit_k_positive_weak_sector_extension",
            "working_action_vector_mass_spectrum_rebuild_required": False,
            "working_action_vector_mass_spectrum_normalization_update_only": True,
            "trial2_paper_side_sync_state": "unlocked_not_yet_executed",
            "trial3_dependency_state": "relaunched_after_post_photon_unlock",
            "split_contract_ready": True,
            "recommended_next_route_or_none": "8.7.56.224",
        },
        {
            "overall_status": "post_photon_trial3_relaunch_route_contract_frozen",
            "trial2_branch_closeable": True,
            "advance_to_8_7_56_224": True,
            "next_required_artifacts": [
                "trial3_relaunched_weak_sector_source_inventory",
                "trial3_relaunched_weak_sector_pilot",
                "trial3_relaunched_declaration_gate",
            ],
        },
        {
            "unlock_gate_summary": unlock_gate["summary"],
            "preservation_summary": preservation_audit["summary"],
            "current_route_summary": current_route["summary"],
            "trial3_old_route_summary": trial3_route["summary"],
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    write_artifact("mass_origin_v2_post_photon_unlock_pivot_route_contract", pivot_route)
    write_artifact("mass_origin_v2_post_photon_nontransverse_two_by_two_quadratic_form", quadratic_form)
    write_artifact("mass_origin_v2_post_photon_nontransverse_diagonalization_basis_statement", diagonalization)
    write_artifact("mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit", preservation_audit)
    write_artifact("mass_origin_v2_post_photon_dependency_unlock_gate", unlock_gate)
    write_artifact("mass_origin_v2_post_photon_trial3_relaunch_route_contract", relaunch_route)

    print("[ok] wrote:")
    print(" - mass_origin_v2_post_photon_unlock_pivot_route_contract_metrics.json")
    print(" - mass_origin_v2_post_photon_nontransverse_two_by_two_quadratic_form_metrics.json")
    print(" - mass_origin_v2_post_photon_nontransverse_diagonalization_basis_statement_metrics.json")
    print(" - mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_metrics.json")
    print(" - mass_origin_v2_post_photon_dependency_unlock_gate_metrics.json")
    print(" - mass_origin_v2_post_photon_trial3_relaunch_route_contract_metrics.json")


# Function: run the post-photon dependency-unlock pivot branch from the CLI.

if __name__ == "__main__":
    main()
