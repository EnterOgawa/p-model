#!/usr/bin/env python3
"""
Generate Trial-1 VEV-pivot artifacts for 8.7.56.134-.140.

This branch stops retrying wording overrides and instead evaluates the physical
question raised by the expert pivot note: can the already-canonical vector
P-field produce a massless photon as a transverse fluctuation around the
Mexican-hat vacuum expectation value?

The branch freezes seven artifacts:

1. The VEV mode-decomposition pivot route contract.
2. The quadratic expansion and three-sector separation.
3. The transverse effective mass-squared evaluation.
4. The conditional Maxwell-form reduction / coupling-mapping audit.
5. The reopened Trial-1 declaration gate.
6. The Case-B honest partial closeout / v1.1 confirmation gate.
7. A follow-through route contract for post-pivot sync work.

The current canon closes the pivot with Case B. The mexican-hat contribution to
the transverse quadratic mass vanishes at the anchored VEV, but the explicit
Stueckelberg/Proca mass term remains and keeps the transverse vector mode
massive with pole mass m_T^2 = m_P^2 = 2 lambda v^2 / Z_P. Therefore the
present canon does not derive a massless photon from P_mu alone, and the
independent electromagnetic sector remains the physically correct v1.1 choice.
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
MEXICAN = OUT / "mass_origin_mexican_hat_parameter_freeze_metrics.json"
TRIAL1_DECLARATION = OUT / "mass_origin_v2_trial1_declaration_gate_metrics.json"
PREVIOUS_ROUTE = OUT / "mass_origin_v2_part1_total_action_baseline_definition_section_heading_reentry_route_contract_metrics.json"
ACTION_AUDIT = OUT / "action_principle_el_derivation_audit.json"


# Function: return the current UTC time in ISO 8601 format.
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: abort if a required path is missing.

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


# Function: execute the VEV pivot branch and freeze the follow-through route contract.

def main() -> None:
    for path in (PART1, PART3A, MEXICAN, TRIAL1_DECLARATION, PREVIOUS_ROUTE, ACTION_AUDIT):
        req(path)

    part1 = read_text(PART1)
    part3a = read_text(PART3A)
    mexican = read_json(MEXICAN)
    trial1_declaration = read_json(TRIAL1_DECLARATION)
    previous_route = read_json(PREVIOUS_ROUTE)
    action_audit = read_json(ACTION_AUDIT)

    mass_formula = mexican["summary"]["mass_parameter_formula"]
    selected_potential = mexican["summary"]["selected_potential_family_formula"]

    common_inputs = {
        "part1_core_theory_markdown": rel(PART1),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "mass_origin_mexican_hat_parameter_freeze_json": rel(MEXICAN),
        "mass_origin_v2_trial1_declaration_gate_json": rel(TRIAL1_DECLARATION),
        "mass_origin_v2_part1_total_action_baseline_definition_section_heading_reentry_route_contract_json": rel(PREVIOUS_ROUTE),
        "action_principle_el_derivation_audit_json": rel(ACTION_AUDIT),
    }

    pivot_contract = payload(
        "8.7.56.134",
        "VEV mode decomposition pivot route contract",
        common_inputs,
        "Freeze the official Trial-1 pivot away from wording overrides and toward a first-principles VEV mode-decomposition test for a transverse massless photon.",
        {
            "background_vev": "P_mu^(0) = (v, 0, 0, 0)",
            "fluctuation_split": "P_mu = P_mu^(0) + delta P_mu,   delta P_i = delta P_i^L + delta P_i^T",
            "transverse_constraints": "div delta P^T = 0,   curl delta P^L = 0",
            "case_a_condition": "m_T^2 = 0  => Maxwell-emergence route remains open",
            "case_b_condition": "m_T^2 != 0 => honest partial closeout and independent L_EM retention",
            "trial2_hold_rule": "Keep 8.7.56.5-.8 on hold until the VEV pivot returns Case A or Case B.",
        },
        [
            row(
                "trial1_vev_pivot_route_contract_complete",
                "pass",
                "VEV pivot route contract complete",
                1,
                "The official Trial-1 route is switched from wording override to VEV mode decomposition.",
            ),
            row(
                "trial1_previous_heading_override_route_frozen",
                "pass",
                "previous wording-override route frozen before execution",
                1,
                "The old heading-override route is retained only as superseded evidence.",
            ),
            row(
                "trial1_case_split_formalized",
                "pass",
                "Case A / Case B split formalized",
                1,
                "The pivot now depends on the explicit transverse effective mass-squared m_T^2.",
            ),
            row(
                "trial1_trial2_hold_retained_under_vev_pivot",
                "pass",
                "Trial-2 hold retained under VEV pivot",
                1,
                "Electromagnetic first-principles work stays on hold until the pivot closes.",
            ),
        ],
        {
            "selected_pivot_route": "vev_mode_decomposition_transverse_photon",
            "background_vev_formula": "P_mu^(0) = (v, 0, 0, 0)",
            "case_a_condition": "m_T^2 = 0",
            "case_b_condition": "m_T^2 != 0",
            "trial2_hold_until_case_decided": True,
            "first_route_to_close_or_none": "vev_quadratic_mode_decomposition",
        },
        {
            "overall_status": "trial1_vev_pivot_route_contract_frozen",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_135": True,
            "next_required_artifacts": [
                "vev_quadratic_mode_decomposition",
                "transverse_mode_effective_mass_evaluation",
            ],
        },
        {
            "previous_route_summary": previous_route["summary"],
            "trial1_declaration_summary": trial1_declaration["summary"],
            "part1_full_action_line": hit(part1, "\\mathcal{L}_{P,\\mathrm{full}}"),
            "part1_em_term_line": hit(part1, "+\\mathcal{L}_{\\mathrm{EM}}"),
            "part1_proca_line": hit(part1, "+\\frac{m_P^2}{2}\\left(P_\\mu-\\frac{1}{m_P}\\partial_\\mu\\pi\\right)"),
            "part3a_a_reject_line": hit(part3a, "A棄却、B採用"),
        },
    )

    quadratic_decomposition = payload(
        "8.7.56.135",
        "VEV quadratic expansion / three-sector separation",
        common_inputs,
        "Expand the canonical vector-P action to quadratic order around the vacuum expectation value and separate the time/radial, longitudinal, and transverse sectors.",
        {
            "background_vev": "P_mu^(0) = (v, 0, 0, 0)",
            "quadratic_norm_expansion": "P_mu P^mu = v^2 + 2 v delta P_0 + (delta P_0)^2 - |delta P^L|^2 - |delta P^T|^2",
            "mexican_hat_family": selected_potential,
            "mexican_hat_quadratic_piece": "V^(2) = lambda v^2 (delta P_0)^2 + O(delta^3)",
            "gauge_fixing_term": "-(1 / (2 xi_g)) (d_mu P^mu + xi_g m_P pi)^2",
            "transverse_decoupling_rule": "For div delta P^T = 0, the gauge-fixing term and Stueckelberg scalar pi do not contribute to the transverse quadratic sector.",
        },
        [
            row(
                "trial1_vev_three_sector_split_available",
                "pass",
                "VEV three-sector split available",
                1,
                "The fluctuation pack can be organized into radial/time, longitudinal, and transverse sectors at quadratic order.",
            ),
            row(
                "trial1_vev_transverse_sector_decoupled_at_quadratic_order",
                "pass",
                "transverse sector decoupled at quadratic order",
                1,
                "The transverse divergence-free sector does not mix with the Stueckelberg scalar through the gauge-fixing term.",
            ),
            row(
                "trial1_vev_mexican_hat_transverse_quadratic_mass_zero",
                "pass",
                "mexican-hat transverse quadratic mass contribution zero",
                1,
                "At the anchored VEV, the mexican-hat quadratic piece depends on delta P_0 and does not give a transverse quadratic mass.",
            ),
            row(
                "trial1_vev_proca_transverse_quadratic_term_present",
                "pass",
                "Proca/Stueckelberg transverse quadratic term present",
                1,
                "The explicit massive vector term still contributes directly to delta P_i^T.",
            ),
        ],
        {
            "three_sector_split_available": True,
            "transverse_sector_decoupled_at_quadratic_order": True,
            "mexican_hat_transverse_quadratic_mass_contribution_zero": True,
            "gauge_fixing_transverse_quadratic_mass_contribution_zero": True,
            "proca_transverse_quadratic_mass_contribution_present": True,
            "first_route_to_close_or_none": "transverse_mode_effective_mass_evaluation",
        },
        {
            "overall_status": "trial1_vev_quadratic_decomposition_frozen",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_136": True,
            "next_required_artifacts": [
                "transverse_mode_effective_mass_evaluation",
            ],
        },
        {
            "part1_vector_free_action_line": hit(part1, "\\mathcal{L}_{P_\\mu}^{\\mathrm{free}}"),
            "part1_proca_line": hit(part1, "+\\frac{m_{P}^2}{2}\\left(P_\\mu-\\frac{1}{m_{P}}\\partial_\\mu\\pi\\right)"),
            "part1_full_action_line": hit(part1, "\\mathcal{L}_{P,\\mathrm{full}}"),
            "part1_gauge_fixing_line": hit(part1, "-\\frac{1}{2\\xi_g}\\left(\\partial_\\mu P^\\mu+\\xi_g m_P\\pi\\right)^2"),
            "mexican_hat_summary": mexican["summary"],
        },
    )

    transverse_mass = payload(
        "8.7.56.136",
        "Transverse mode effective mass-squared evaluation",
        common_inputs,
        "Evaluate the explicit transverse effective mass-squared from the quadratic VEV-expanded action and decide the Case A / Case B split.",
        {
            "transverse_quadratic_lagrangian": "L_T^(2) = -(Z_P/4) f^(T)_{mu nu} f_T^(mu nu) + (m_P^2 / 2) delta P_i^T delta P^{Ti}",
            "potential_contribution": "m_T,potential^2 = 0 at the anchored VEV because V'(v) = 0 and V^(2) depends only on delta P_0",
            "gauge_fixing_contribution": "m_T,gauge-fixing^2 = 0 for div delta P^T = 0",
            "proca_contribution": "m_T,proca^2 = m_P^2",
            "full_transverse_mass": f"m_T^2 = m_P^2 = {mass_formula.replace('m_P^2 = ', '')}",
            "dispersion_relation": "omega^2 = |k|^2 + m_T^2",
        },
        [
            row(
                "trial1_vev_transverse_potential_mass_zero",
                "pass",
                "transverse potential mass contribution zero",
                1,
                "The mexican-hat anchor leaves no quadratic transverse potential mass at the VEV.",
            ),
            row(
                "trial1_vev_transverse_gauge_fixing_mass_zero",
                "pass",
                "transverse gauge-fixing mass contribution zero",
                1,
                "The divergence-free transverse sector does not inherit a gauge-fixing mass term.",
            ),
            row(
                "trial1_vev_transverse_proca_mass_nonzero",
                "pass",
                "transverse Proca mass contribution nonzero",
                1,
                "The explicit massive vector term keeps the transverse mode massive.",
            ),
            row(
                "trial1_vev_transverse_massless_gate",
                "fail",
                "transverse massless gate",
                0,
                "The full transverse effective mass-squared remains m_T^2 = m_P^2, not zero.",
            ),
            row(
                "trial1_vev_case_b_selected",
                "pass",
                "Case B selected",
                1,
                "The VEV pivot closes on the massive-transverse branch rather than on Maxwell emergence.",
            ),
        ],
        {
            "transverse_potential_mass_contribution_zero": True,
            "transverse_gauge_fixing_mass_contribution_zero": True,
            "transverse_proca_mass_contribution_nonzero": True,
            "transverse_effective_mass_squared_formula": mass_formula,
            "transverse_mode_massless": False,
            "selected_case": "case_b_massive_transverse_mode",
            "first_route_to_close_or_none": "maxwell_form_reduction_coupling_mapping_audit",
        },
        {
            "overall_status": "trial1_vev_transverse_mass_evaluated_case_b",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_137": True,
            "next_required_artifacts": [
                "maxwell_form_reduction_coupling_mapping_audit",
                "trial1_vev_pivot_reopened_declaration_gate",
            ],
        },
        {
            "part1_free_action_line": hit(part1, "\\mathcal{L}_{P_\\mu}^{\\mathrm{free}}"),
            "part1_full_action_line": hit(part1, "\\mathcal{L}_{P,\\mathrm{full}}"),
            "part1_propagator_line": hit(part1, "D_{\\mu\\nu}(k)="),
            "part1_propagator_pole_line": hit(part1, "\\frac{\\eta_{\\mu\\nu}-k_\\mu k_\\nu/m_P^2}{k^2-m_P^2+i0}"),
            "mexican_hat_mass_formula": mexican["summary"]["mass_parameter_formula"],
        },
    )

    maxwell_reduction = payload(
        "8.7.56.137",
        "Maxwell-form reduction / coupling mapping audit",
        common_inputs,
        "Audit the Maxwell-reduction route conditionally after the transverse-mass evaluation. Under Case B, record the honest non-closure rather than forcing a photon identification.",
        {
            "case_a_requirement": "m_T^2 = 0 is required before the transverse sector can be promoted to a Maxwell branch.",
            "case_b_outcome": "If m_T^2 != 0, the transverse sector remains Proca-like and no photon/coupling mapping should be claimed.",
            "route_a_template": action_audit["equations"]["lagrangian_density"],
            "conditional_rule": "Run the Maxwell audit honestly: report non-closure if the massless prerequisite is not met.",
        },
        [
            row(
                "trial1_vev_case_a_prerequisite_met",
                "fail",
                "Case A prerequisite m_T^2 = 0 met",
                0,
                "The transverse mode is not massless, so the Maxwell-emergence route does not open.",
            ),
            row(
                "trial1_vev_transverse_maxwell_reduction_available",
                "fail",
                "transverse Maxwell-form reduction available",
                0,
                "A massive transverse Proca-like mode cannot be claimed as the Maxwell sector.",
            ),
            row(
                "trial1_vev_transverse_coupling_mapping_available",
                "fail",
                "transverse coupling mapping available",
                0,
                "No e(g_P, v, Z_P) mapping is admissible because the Maxwell branch did not open.",
            ),
            row(
                "trial1_vev_independent_em_sector_still_required",
                "pass",
                "independent electromagnetic sector still required",
                1,
                "The current canon still needs the explicitly independent L_EM sector.",
            ),
        ],
        {
            "case_a_prerequisite_met": False,
            "transverse_maxwell_reduction_available": False,
            "transverse_coupling_mapping_available": False,
            "independent_em_sector_still_required": True,
            "first_route_to_close_or_none": "trial1_vev_pivot_reopened_declaration_gate",
        },
        {
            "overall_status": "trial1_vev_maxwell_reduction_not_available_under_case_b",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_138": True,
            "next_required_artifacts": [
                "trial1_vev_pivot_reopened_declaration_gate",
                "trial1_case_b_honest_partial_closeout_v1_1_confirmation_gate",
            ],
        },
        {
            "transverse_mass_summary": transverse_mass["summary"],
            "part1_em_term_line": hit(part1, "+\\mathcal{L}_{\\mathrm{EM}}"),
            "part3a_independent_connection_line": hit(part3a, "独立接続"),
            "part3a_massless_photon_line": hit(part3a, "ゲージ場は質量項を持たない最小形を採用する"),
            "action_audit_summary": action_audit["decision"],
        },
    )

    declaration_gate = payload(
        "8.7.56.138",
        "Trial-1 reopened declaration gate / Trial-2 unlock gate after VEV pivot",
        common_inputs,
        "Convert the VEV pivot result into a Trial-1 declaration and decide whether Trial-2 can be unlocked under Case A or Case B.",
        {
            "trial1_full_pass_rule": "Trial-1 upgrades to full pass only if the VEV pivot yields a massless transverse Maxwell branch.",
            "trial1_case_b_rule": "If the transverse branch stays massive, Trial-1 remains an honest partial closeout and Trial-2 stays on hold.",
            "trial2_unlock_rule": "Trial-2 can start only after Trial-1 reaches full-pass status.",
        },
        [
            row(
                "trial1_vev_pivot_full_pass_ready",
                "fail",
                "Trial-1 full pass ready after VEV pivot",
                0,
                "The transverse sector stayed massive, so Trial-1 did not promote to a full photon derivation.",
            ),
            row(
                "trial2_unlock_ready_after_vev_pivot",
                "fail",
                "Trial-2 unlock ready after VEV pivot",
                0,
                "Electromagnetic first-principles work remains blocked because Trial-1 did not close as a full pass.",
            ),
            row(
                "trial1_case_b_honest_partial_closeout_retained",
                "pass",
                "Trial-1 honest partial closeout retained",
                1,
                "Global U(1) and local Stueckelberg closure remain valid, but photon derivation is not available in the current canon.",
            ),
        ],
        {
            "trial1_pass_level": "honest_partial_closeout_case_b_transverse_massive",
            "trial1_full_pass_ready": False,
            "trial2_unlock_ready": False,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.139",
        },
        {
            "overall_status": "trial1_vev_pivot_case_b_gate_frozen",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_139": True,
            "next_required_artifacts": [
                "trial1_case_b_honest_partial_closeout_v1_1_confirmation_gate",
            ],
        },
        {
            "trial1_declaration_summary": trial1_declaration["summary"],
            "transverse_mass_summary": transverse_mass["summary"],
            "maxwell_reduction_summary": maxwell_reduction["summary"],
        },
    )

    case_b_gate = payload(
        "8.7.56.139",
        "Case-B honest partial closeout / v1.1 confirmation gate",
        common_inputs,
        "Freeze the honest Case-B consequence of the VEV pivot: the independent electromagnetic sector is physically retained and the v1.1 judgment was correct under the current canon.",
        {
            "case_b_statement": "m_T^2 != 0 => the transverse vector fluctuation is Proca-like, not a massless photon.",
            "v1_1_preservation_rule": "If the photon is not derived from P_mu alone, the explicit independent L_EM sector remains the correct canonical choice.",
            "trial1_scope_rule": "Trial-1 preserves the global-U(1) and local-vector-gauge results but does not reach full EM derivation under the current canon.",
        },
        [
            row(
                "trial1_case_b_independent_em_sector_physically_retained",
                "pass",
                "independent electromagnetic sector physically retained",
                1,
                "The VEV pivot confirms that the current canon still needs the separate L_EM term.",
            ),
            row(
                "trial1_case_b_v1_1_judgment_confirmed",
                "pass",
                "v1.1 judgment confirmed under Case B",
                1,
                "The earlier v1.1 decision to keep electromagnetism as an adopted independent sector remains physically correct.",
            ),
            row(
                "trial1_case_b_minimum_condition_failed_for_current_canon",
                "fail",
                "Trial-1 minimum condition satisfied under current canon",
                0,
                "The current canon does not meet the v2.0 Trial-1 photon-derivation target.",
            ),
            row(
                "trial1_case_b_honest_partial_closeout_ready",
                "pass",
                "Trial-1 Case-B honest partial closeout ready",
                1,
                "The Case-B outcome is explicit enough to freeze a formal closeout and follow-through sync.",
            ),
        ],
        {
            "independent_em_sector_physically_retained": True,
            "v1_1_judgment_confirmed": True,
            "trial1_case_b_honest_partial_closeout_ready": True,
            "trial1_minimum_condition_satisfied_under_current_canon": False,
            "recommended_next_route_or_none": "8.7.56.140",
        },
        {
            "overall_status": "trial1_case_b_honest_partial_closeout_frozen",
            "trial1_branch_closeable": True,
            "advance_to_8_7_56_5": False,
            "next_required_artifacts": [
                "trial1_case_b_followthrough_route_contract",
            ],
        },
        {
            "part1_em_term_line": hit(part1, "+\\mathcal{L}_{\\mathrm{EM}}"),
            "part3a_independent_connection_line": hit(part3a, "独立接続"),
            "part3a_a_reject_line": hit(part3a, "A棄却、B採用"),
            "declaration_gate_summary": declaration_gate["summary"],
        },
    )

    followthrough = payload(
        "8.7.56.140",
        "Trial-1 Case-B follow-through route contract",
        common_inputs,
        "Freeze the post-pivot follow-through route for Case B: sync the honest partial closeout into roadmap / paper-side scope wording while keeping Trial-2 on hold.",
        {
            "selected_followthrough_route": "trial1_case_b_paper_side_sync_and_scope_freeze",
            "missing_v2_artifact": "trial1_case_b_wording_sync_pack",
            "followthrough_rule": "Case B closes the physics question; the next route is documentation and scope synchronization, not another Trial-1 physics retry.",
            "trial2_hold_rule": "Keep 8.7.56.5-.8 on hold until a future canon change creates a genuine photon-derivation route.",
        },
        [
            row(
                "trial1_case_b_followthrough_route_contract_complete",
                "pass",
                "Trial-1 Case-B follow-through route contract complete",
                1,
                "The next official route is frozen as a sync/closeout branch rather than another physics retry.",
            ),
            row(
                "trial1_case_b_trial2_hold_retained",
                "pass",
                "Trial-2 hold retained under Case-B follow-through",
                1,
                "Trial-2 remains blocked because Trial-1 did not produce a photon derivation.",
            ),
            row(
                "trial1_case_b_new_physics_retry_not_opened",
                "pass",
                "no new Trial-1 physics retry opened",
                1,
                "The pivot answered the physical question directly, so the follow-through is now a scope-sync task.",
            ),
        ],
        {
            "selected_followthrough_route": "trial1_case_b_paper_side_sync_and_scope_freeze",
            "missing_v2_artifact": "trial1_case_b_wording_sync_pack",
            "split_contract_ready": True,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.141",
        },
        {
            "overall_status": "trial1_case_b_followthrough_route_contract_frozen",
            "trial1_branch_closeable": True,
            "advance_to_8_7_56_5": False,
            "next_required_artifacts": [
                "trial1_case_b_paper_side_sync_inventory",
                "trial1_case_b_wording_freeze",
                "trial1_case_b_scope_declaration_gate",
            ],
        },
        {
            "case_b_gate_summary": case_b_gate["summary"],
            "declaration_gate_summary": declaration_gate["summary"],
            "transverse_mass_summary": transverse_mass["summary"],
        },
    )

    write_artifact("mass_origin_v2_vev_mode_decomposition_pivot_route_contract", pivot_contract)
    write_artifact("mass_origin_v2_vev_quadratic_mode_decomposition", quadratic_decomposition)
    write_artifact("mass_origin_v2_transverse_mode_effective_mass_evaluation", transverse_mass)
    write_artifact("mass_origin_v2_maxwell_form_reduction_coupling_mapping_audit", maxwell_reduction)
    write_artifact("mass_origin_v2_trial1_vev_pivot_reopened_declaration_gate", declaration_gate)
    write_artifact("mass_origin_v2_trial1_case_b_honest_partial_closeout_v1_1_confirmation_gate", case_b_gate)
    write_artifact("mass_origin_v2_trial1_case_b_followthrough_route_contract", followthrough)

    print("[ok] wrote:")
    print(" - mass_origin_v2_vev_mode_decomposition_pivot_route_contract_metrics.json")
    print(" - mass_origin_v2_vev_quadratic_mode_decomposition_metrics.json")
    print(" - mass_origin_v2_transverse_mode_effective_mass_evaluation_metrics.json")
    print(" - mass_origin_v2_maxwell_form_reduction_coupling_mapping_audit_metrics.json")
    print(" - mass_origin_v2_trial1_vev_pivot_reopened_declaration_gate_metrics.json")
    print(" - mass_origin_v2_trial1_case_b_honest_partial_closeout_v1_1_confirmation_gate_metrics.json")
    print(" - mass_origin_v2_trial1_case_b_followthrough_route_contract_metrics.json")


# Function: run the branch script from the command line.

if __name__ == "__main__":
    main()
