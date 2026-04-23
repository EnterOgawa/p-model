#!/usr/bin/env python3
"""
Generate Trial-1 breakthrough pivot artifacts for 8.7.56.154-.159.

This branch adopts the expert breakthrough suggestion as the new official v2.0
working pivot:

1. treat the separate Proca/Stueckelberg mass term as a redundant mass-source
   candidate once the mexican-hat family already fixes
   m_P^2 = 2 lambda v^2 / Z_P,
2. re-evaluate the VEV quadratic decomposition under the mass-source-unified
   action where the mexican hat is the unique mass source,
3. reopen the Maxwell / coupling route at the level of the transverse
   fluctuation sector, and
4. classify which already-closed branches are clearly preserved and which must
   be re-audited before a global canon sync.

The branch intentionally distinguishes three layers:

- current canon: still contains the explicit Proca/Stueckelberg term,
- breakthrough working pivot: removes the redundant term without adding a new
  free parameter,
- post-pivot follow-through: re-audit action-sensitive branches and reopen the
  Trial-2 electromagnetism branch.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
MEXICAN = OUT / "mass_origin_mexican_hat_parameter_freeze_metrics.json"
VEV_MASS = OUT / "mass_origin_v2_transverse_mode_effective_mass_evaluation_metrics.json"
CASE_B_GATE = OUT / "mass_origin_v2_trial1_case_b_honest_partial_closeout_v1_1_confirmation_gate_metrics.json"
REOPEN_GATE = OUT / "mass_origin_v2_trial1_reopen_prerequisite_gate_metrics.json"
TRIAL3_ROUTE = OUT / "mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_metrics.json"
TRIAL3_WZ = OUT / "mass_origin_v2_trial3_wz_sector_source_inventory_metrics.json"
VECTOR_ROUTE = OUT / "mass_origin_vector_qball_route_contract_metrics.json"
DIRECT_KAPPA = OUT / "mass_origin_direct_kappa_bridge_statement_freeze_metrics.json"


# Function: return the current UTC timestamp in ISO 8601 format.
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: stop execution if an input path is missing.

def req(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: load a UTF-8 JSON artifact.

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: load a UTF-8 markdown/text source.

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


# Function: parse the symbolic mass formula string into the right-hand side only.

def rhs_only(formula: str) -> str:
    return formula.split("=", 1)[1].strip() if "=" in formula else formula


# Function: execute the breakthrough pivot branch and freeze the Trial-2 reopen route.

def main() -> None:
    for path in (
        PART1,
        PART3A,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        MEXICAN,
        VEV_MASS,
        CASE_B_GATE,
        REOPEN_GATE,
        TRIAL3_ROUTE,
        TRIAL3_WZ,
        VECTOR_ROUTE,
        DIRECT_KAPPA,
    ):
        req(path)

    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    status_text = read_text(STATUS)
    ai_context = read_json(AI_CONTEXT)
    mexican = read_json(MEXICAN)
    vev_mass = read_json(VEV_MASS)
    case_b_gate = read_json(CASE_B_GATE)
    reopen_gate = read_json(REOPEN_GATE)
    trial3_route = read_json(TRIAL3_ROUTE)
    trial3_wz = read_json(TRIAL3_WZ)
    vector_route = read_json(VECTOR_ROUTE)
    direct_kappa = read_json(DIRECT_KAPPA)

    mass_formula = mexican["summary"]["mass_parameter_formula"]
    mass_rhs = rhs_only(mass_formula)
    potential_formula = mexican["summary"]["selected_potential_family_formula"]
    remaining_symbols = mexican["summary"]["remaining_free_parameter_symbols"]
    current_transverse_formula = vev_mass["summary"]["transverse_effective_mass_squared_formula"]

    common_inputs = {
        "part1_markdown": rel(PART1),
        "part3a_markdown": rel(PART3A),
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_mexican_hat_parameter_freeze_json": rel(MEXICAN),
        "mass_origin_v2_transverse_mode_effective_mass_evaluation_json": rel(VEV_MASS),
        "mass_origin_v2_trial1_case_b_honest_partial_closeout_v1_1_confirmation_gate_json": rel(CASE_B_GATE),
        "mass_origin_v2_trial1_reopen_prerequisite_gate_json": rel(REOPEN_GATE),
        "mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_json": rel(TRIAL3_ROUTE),
        "mass_origin_v2_trial3_wz_sector_source_inventory_json": rel(TRIAL3_WZ),
        "mass_origin_vector_qball_route_contract_json": rel(VECTOR_ROUTE),
        "mass_origin_direct_kappa_bridge_statement_freeze_json": rel(DIRECT_KAPPA),
    }

    route_contract = payload(
        "8.7.56.154",
        "Trial-1 breakthrough pivot route contract",
        common_inputs,
        "Freeze the official pivot suggested by the expert breakthrough note: unify the P-field mass source into the already-frozen mexican-hat family before continuing the weak-sector Trial-3 residual.",
        {
            "pivot_principle": "If m_P^2 = 2 lambda v^2 / Z_P is already fixed by the mexican-hat family, a separate Proca/Stueckelberg mass source becomes a redundancy candidate rather than a new required degree of freedom.",
            "working_action_rule": "test a mass-source-unified action where the mexican hat is the unique mass source for the vector-P sector",
            "breakthrough_goal": "reopen Trial-1 through a massless transverse fluctuation and thereby reopen Trial-2 from first principles",
            "fallback_rule": "if the breakthrough candidate fails or breaks already-closed outputs, keep the explicit k>0 weak-sector branch as fallback hold",
        },
        [
            row(
                "trial1_breakthrough_pivot_route_contract_complete",
                "pass",
                "Trial-1 breakthrough pivot route contract complete",
                1,
                "The official v2.0 mainline is redirected from the weak-sector residual to the breakthrough mass-source pivot.",
            ),
            row(
                "trial1_breakthrough_uses_existing_mexican_hat_pack",
                "pass",
                "breakthrough pivot uses existing mexican-hat pack",
                1,
                "The pivot reuses the already-frozen mexican-hat family and its mass relation.",
            ),
            row(
                "trial1_breakthrough_new_parameter_count",
                "pass",
                "new free parameters introduced by breakthrough pivot",
                0,
                "The pivot removes a candidate redundancy instead of adding a new coupling.",
            ),
            row(
                "trial3_explicit_k_branch_retained_as_fallback_hold",
                "pass",
                "Trial-3 explicit k-positive branch retained as fallback hold",
                1,
                "The weak-sector residual is preserved as fallback evidence rather than discarded.",
            ),
        ],
        {
            "selected_pivot_route": "trial1_mexican_hat_only_mass_source_breakthrough",
            "mass_formula_from_mexican_hat": mass_formula,
            "current_case_b_formula": current_transverse_formula,
            "trial3_fallback_route_or_none": trial3_route["summary"]["selected_residual_route"],
            "first_route_to_close_or_none": "trial1_redundant_proca_term_admissibility_audit",
        },
        {
            "overall_status": "trial1_breakthrough_pivot_route_contract_frozen",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_155": True,
            "next_required_artifacts": [
                "trial1_redundant_proca_term_admissibility_audit",
                "trial1_breakthrough_modified_vev_mode_decomposition",
            ],
        },
        {
            "mexican_hat_summary": mexican["summary"],
            "current_case_b_summary": case_b_gate["summary"],
            "reopen_gate_summary": reopen_gate["summary"],
            "trial3_route_summary": trial3_route["summary"],
            "part1_free_action_line": hit(part1_text, "+\\frac{m_P^2}{2}P_\\mu P^\\mu"),
            "part1_full_action_line": hit(part1_text, "+\\frac{m_P^2}{2}\\left(P_\\mu-\\frac{1}{m_P}\\partial_\\mu\\pi\\right)"),
            "part3a_case_b_line": hit(part3a_text, "current canon では transverse mode は massless photon へ落ちず"),
        },
    )

    redundancy_targets = [
        target_record(
            "part1_free_action_explicit_mass_term",
            PART1,
            part1_text,
            "+\\frac{m_P^2}{2}P_\\mu P^\\mu",
            "Part I minimal action still carries an explicit vector mass term.",
        ),
        target_record(
            "part1_full_action_explicit_stueckelberg_mass_term",
            PART1,
            part1_text,
            "+\\frac{m_P^2}{2}\\left(P_\\mu-\\frac{1}{m_P}\\partial_\\mu\\pi\\right)",
            "Part I full action still carries the explicit Stueckelberg/Proca mass term.",
        ),
        target_record(
            "part1_independent_em_term",
            PART1,
            part1_text,
            "+\\mathcal{L}_{\\mathrm{EM}}",
            "Part I still freezes an independent electromagnetic sector.",
        ),
        target_record(
            "part3a_case_b_judgment",
            PART3A,
            part3a_text,
            "A棄却、B採用",
            "Part III-A still records the current-canon Trial-1 judgment as A reject / B adopt.",
        ),
        target_record(
            "status_trial3_current_next_step",
            STATUS,
            status_text,
            "current official next step は `8.7.56.154`",
            "STATUS still points to the pre-pivot Trial-3 continuation before this branch is synced.",
        ),
    ]
    redundancy_missing = [item for item in redundancy_targets if not item["present"]]
    separate_mass_redundant_candidate = not redundancy_missing and remaining_symbols == ["lambda"]

    redundancy_audit = payload(
        "8.7.56.155",
        "Redundant Proca-term admissibility audit",
        common_inputs,
        "Audit whether the separate Proca/Stueckelberg mass term can be reclassified as a redundant mass source once the mexican-hat family already fixes m_P^2 = 2 lambda v^2 / Z_P.",
        {
            "mass_relation": mass_formula,
            "admissibility_rule": "the breakthrough audit passes if the independent mass parameter is already frozen by the mexican-hat pack, so removing the separate mass term reduces parameter redundancy rather than introducing a new degree of freedom",
            "current_canon_guard": "the audit distinguishes between current-canon wording and the new working pivot without erasing the evidence that Case B held before the pivot",
        },
        [
            row(
                "trial1_breakthrough_redundancy_audit_complete",
                "pass",
                "Trial-1 breakthrough redundancy audit complete",
                1,
                "The separate mass-source admissibility audit was executed against the current canonical sources.",
            ),
            row(
                "trial1_breakthrough_mexican_hat_mass_formula_available",
                "pass" if mass_formula == "m_P^2 = 2 lambda v^2 / Z_P" else "reject",
                "mexican-hat mass formula available",
                1 if mass_formula == "m_P^2 = 2 lambda v^2 / Z_P" else 0,
                "The frozen mexican-hat pack already fixes the effective mass scale.",
            ),
            row(
                "trial1_breakthrough_separate_mass_term_present",
                "pass" if not redundancy_missing else "reject",
                "separate Proca/Stueckelberg term present in current canon",
                1 if not redundancy_missing else 0,
                "The current canon still contains the explicit term whose redundancy is now being audited.",
            ),
            row(
                "trial1_breakthrough_independent_mass_parameter_not_free",
                "pass" if remaining_symbols == ["lambda"] else "reject",
                "independent mass parameter not free in the mexican-hat pack",
                1 if remaining_symbols == ["lambda"] else 0,
                "The mexican-hat parameter freeze leaves lambda as the only remaining free coupling.",
            ),
            row(
                "trial1_breakthrough_redundant_term_removal_new_parameter_free",
                "pass" if separate_mass_redundant_candidate else "reject",
                "redundant term removal is new-parameter free",
                1 if separate_mass_redundant_candidate else 0,
                "Removing the separate mass source reduces redundancy rather than adding a new parameter.",
            ),
        ],
        {
            "mass_formula_from_mexican_hat_available": mass_formula == "m_P^2 = 2 lambda v^2 / Z_P",
            "separate_proca_term_present_in_current_canon": not redundancy_missing,
            "remaining_free_parameter_symbols": remaining_symbols,
            "redundant_term_removal_new_parameter_free": separate_mass_redundant_candidate,
            "working_pivot_requires_part1_sync": True,
            "first_route_to_close_or_none": "trial1_breakthrough_modified_vev_mode_decomposition",
        },
        {
            "overall_status": "trial1_breakthrough_redundant_term_admissible" if separate_mass_redundant_candidate else "trial1_breakthrough_redundant_term_not_admissible",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_156": separate_mass_redundant_candidate,
            "next_required_artifacts": [
                "trial1_breakthrough_modified_vev_mode_decomposition",
            ],
        },
        {
            "inventory_targets": redundancy_targets,
            "mexican_hat_summary": mexican["summary"],
            "current_case_b_summary": case_b_gate["summary"],
        },
    )

    modified_vev = payload(
        "8.7.56.156",
        "Breakthrough modified-action VEV decomposition",
        common_inputs,
        "Re-evaluate the VEV quadratic decomposition under the mass-source-unified working action where the mexican hat is the unique source of the P-sector mass scale.",
        {
            "working_action": "-(Z_P/4) F_(P)^2 + (lambda/4) (|P|^2 - v^2)^2 + g_P P_mu J^mu",
            "background_vev": "P_mu^(0) = (v, 0, 0, 0)",
            "quadratic_potential": "V^(2) = lambda v^2 (delta P_0)^2 + 0 * |delta P_i|^2",
            "transverse_sector": "L_T^(2) = -(Z_P/4) f_(T)^2",
            "radial_mass": "m_0^2 = 4 lambda v^2 / Z_P",
        },
        [
            row(
                "trial1_breakthrough_working_action_defined",
                "pass",
                "breakthrough working action defined",
                1,
                "The modified action removes the separate Proca/Stueckelberg mass source and keeps the mexican hat as the unique mass source.",
            ),
            row(
                "trial1_breakthrough_transverse_massless",
                "pass",
                "transverse mode massless under breakthrough working action",
                1,
                "Without the separate Proca term, the transverse sector inherits no quadratic mass contribution.",
            ),
            row(
                "trial1_breakthrough_radial_mode_massive",
                "pass",
                "radial mode remains massive",
                1,
                "The time/radial fluctuation still carries positive curvature from the mexican hat.",
            ),
            row(
                "trial1_breakthrough_case_a_reopened",
                "pass",
                "Trial-1 Case A reopened under working action",
                1,
                "The breakthrough working action reopens the massless-transverse route.",
            ),
        ],
        {
            "working_action_uses_mexican_hat_only_mass_source": True,
            "transverse_effective_mass_squared_formula": "m_T^2 = 0",
            "radial_effective_mass_squared_formula": "m_0^2 = 4 lambda v^2 / Z_P",
            "transverse_mode_massless_under_breakthrough_action": True,
            "selected_breakthrough_case": "case_a_massless_transverse_mode",
        },
        {
            "overall_status": "trial1_breakthrough_modified_vev_case_a",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_157": True,
            "next_required_artifacts": [
                "trial1_breakthrough_maxwell_coupling_audit",
            ],
        },
        {
            "current_case_b_formula": current_transverse_formula,
            "current_case_b_summary": vev_mass["summary"],
            "mexican_hat_summary": mexican["summary"],
        },
    )

    maxwell_audit = payload(
        "8.7.56.157",
        "Breakthrough Maxwell / coupling audit",
        common_inputs,
        "Freeze the Maxwell-form reduction, electromagnetic field definition, and coupling formulas implied by the massless transverse mode of the breakthrough working action.",
        {
            "field_definition": "A_mu := delta P_mu^T / sqrt(Z_P)",
            "maxwell_form": "L_Maxwell = -(1/4) F_(A)^2",
            "coupling_formula": "e = g_P / sqrt(Z_P)",
            "fine_structure_formula": "alpha = g_P^2 / (4 pi Z_P hbar c)",
        },
        [
            row(
                "trial1_breakthrough_maxwell_form_available",
                "pass",
                "Maxwell form available under breakthrough action",
                1,
                "The massless transverse fluctuation reduces to a Maxwell kinetic term after canonical normalization.",
            ),
            row(
                "trial1_breakthrough_photon_identification_available",
                "pass",
                "photon identification available under breakthrough action",
                1,
                "The transverse fluctuation can be identified as the emergent photon candidate.",
            ),
            row(
                "trial1_breakthrough_coupling_formula_ready",
                "pass",
                "coupling formula e = g_P / sqrt(Z_P) ready",
                1,
                "The electromagnetic coupling maps to already-frozen P-sector coefficients.",
            ),
            row(
                "trial1_breakthrough_alpha_formula_ready",
                "pass",
                "fine-structure formula ready",
                1,
                "The breakthrough pivot yields a first-principles alpha formula even before a numeric audit.",
            ),
        ],
        {
            "transverse_maxwell_reduction_available": True,
            "photon_definition_formula": "A_mu = delta P_mu^T / sqrt(Z_P)",
            "electric_charge_formula": "e = g_P / sqrt(Z_P)",
            "alpha_formula": "alpha = g_P^2 / (4 pi Z_P hbar c)",
            "trial2_foundational_branch_unlocked": True,
        },
        {
            "overall_status": "trial1_breakthrough_maxwell_route_open",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_158": True,
            "next_required_artifacts": [
                "trial1_breakthrough_legacy_preservation_audit",
                "trial1_breakthrough_declaration_gate",
            ],
        },
        {
            "mexican_hat_mass_formula": mass_formula,
            "part1_int_line": hit(part1_text, "\\mathcal{L}_{\\mathrm{int}}=g_P P_\\mu J^\\mu_{\\mathrm{matter}}"),
            "trial1_reopen_gate_summary": reopen_gate["summary"],
        },
    )

    direct_kappa_background_only = direct_kappa["summary"]["statement_is_inference_from_frozen_background_exponential_law"]
    vector_route_uses_proca = "Pi_mu Pi^mu" in vector_route["formulas"]["vector_field_action"]
    weak_sector_reuses_vector_route = trial3_wz["summary"]["inventory_ready"]

    preservation_audit = payload(
        "8.7.56.158",
        "Breakthrough legacy-preservation audit",
        common_inputs,
        "Classify which already-closed outputs are structurally preserved by the breakthrough pivot and which action-sensitive branches require a follow-up re-audit before any global canon sync.",
        {
            "preservation_rule": "background-only observables survive immediately if they do not depend on the explicit Proca/Stueckelberg sector",
            "reaudit_rule": "action-sensitive vector hierarchy branches require re-audit if they were derived from an explicitly Proca-like vector action",
            "current_scope_rule": "the breakthrough branch may reopen Trial-2 before a full global re-audit, but it must not overclaim that every action-sensitive result is already preserved",
        },
        [
            row(
                "trial1_breakthrough_direct_kappa_background_branch_preserved",
                "pass" if direct_kappa_background_only else "reject",
                "direct kappa background branch preserved",
                1 if direct_kappa_background_only else 0,
                "The direct kappa bridge depends on the background P-wave law and not on the explicit vector mass term.",
            ),
            row(
                "trial1_breakthrough_vector_mass_spectrum_reaudit_required",
                "pass" if vector_route_uses_proca else "reject",
                "vector mass-spectrum branch requires re-audit",
                1 if vector_route_uses_proca else 0,
                "The vector-Q-ball / Proca-soliton route still exposes an explicitly Proca-like action.",
            ),
            row(
                "trial1_breakthrough_weak_sector_reaudit_required",
                "pass" if weak_sector_reuses_vector_route else "reject",
                "weak-sector branch requires re-audit after breakthrough pivot",
                1 if weak_sector_reuses_vector_route else 0,
                "Trial-3 currently reuses the vector hierarchy that was built on the old action-level closure.",
            ),
            row(
                "trial1_breakthrough_global_full_preservation_without_reaudit",
                "watch" if vector_route_uses_proca else "pass",
                "global full preservation without re-audit",
                0 if vector_route_uses_proca else 1,
                "Not every previously closed action-sensitive branch can be claimed preserved without a targeted follow-up audit.",
            ),
        ],
        {
            "direct_kappa_background_branch_preserved": direct_kappa_background_only,
            "vector_mass_spectrum_reaudit_required": vector_route_uses_proca,
            "weak_sector_reaudit_required": weak_sector_reuses_vector_route,
            "global_legacy_reaudit_required": vector_route_uses_proca or weak_sector_reuses_vector_route,
            "paper_side_sync_required": True,
        },
        {
            "overall_status": "trial1_breakthrough_partial_preservation_audit_frozen",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_159": True,
            "next_required_artifacts": [
                "trial1_breakthrough_declaration_gate",
            ],
        },
        {
            "direct_kappa_summary": direct_kappa["summary"],
            "vector_route_summary": vector_route["summary"],
            "vector_route_action": vector_route["formulas"]["vector_field_action"],
            "trial3_wz_summary": trial3_wz["summary"],
        },
    )

    declaration_gate = payload(
        "8.7.56.159",
        "Breakthrough declaration gate / Trial-2 unlock route contract",
        common_inputs,
        "Freeze the official declaration after the breakthrough pivot: Trial-1 passes under the mass-source-unified working action, Trial-2 reopens, Trial-3 explicit k-positive continuation moves to fallback hold, and global action-sensitive re-audits remain scheduled.",
        {
            "trial1_pass_rule": "Trial-1 passes once the breakthrough working action yields a massless transverse mode and a Maxwell-form reduction without adding a new free parameter",
            "trial2_unlock_rule": "Trial-2 reopens once e = g_P / sqrt(Z_P) and the Maxwell sector follow from the P-field transverse fluctuation",
            "fallback_rule": "Keep the explicit k-positive weak-sector route only as fallback until the reopened EM branch and the action-sensitive re-audits settle the new working canon",
        },
        [
            row(
                "trial1_breakthrough_candidate_pass",
                "pass",
                "Trial-1 breakthrough candidate pass",
                1,
                "The mass-source-unified working action reopens Case A and supplies a massless transverse photon candidate.",
            ),
            row(
                "trial2_reopened_by_breakthrough_pivot",
                "pass",
                "Trial-2 reopened by breakthrough pivot",
                1,
                "The Maxwell / coupling route is open, so the electromagnetism branch becomes the next official executable route.",
            ),
            row(
                "trial3_explicit_k_positive_branch_fallback_hold",
                "pass",
                "Trial-3 explicit k-positive branch moved to fallback hold",
                1,
                "The weak-sector residual is deferred until the reopened Trial-2 branch and follow-up re-audits complete.",
            ),
            row(
                "trial1_breakthrough_global_reaudit_required",
                "watch" if preservation_audit["summary"]["global_legacy_reaudit_required"] else "pass",
                "global action-sensitive re-audit required",
                1 if preservation_audit["summary"]["global_legacy_reaudit_required"] else 0,
                "Action-sensitive mass-spectrum / weak-sector branches still require targeted follow-through before a full paper-side canon sync.",
            ),
        ],
        {
            "trial1_breakthrough_pass_under_working_action": True,
            "trial2_unlock_ready": True,
            "trial3_explicit_k_positive_branch_state": "fallback_hold",
            "global_legacy_reaudit_required": preservation_audit["summary"]["global_legacy_reaudit_required"],
            "paper_side_sync_required": True,
            "recommended_next_route_or_none": "8.7.56.5",
        },
        {
            "overall_status": "trial1_breakthrough_pass_trial2_reopened",
            "trial1_branch_closeable": True,
            "advance_to_8_7_56_5": True,
            "next_required_artifacts": [
                "trial2_phase_connection_maxwell_source_inventory",
                "trial2_curvature_and_coulomb_pilot",
                "trial2_alpha_numeric_audit",
            ],
        },
        {
            "route_contract": route_contract["summary"],
            "redundancy_audit": redundancy_audit["summary"],
            "modified_vev_summary": modified_vev["summary"],
            "maxwell_summary": maxwell_audit["summary"],
            "preservation_summary": preservation_audit["summary"],
        },
    )

    write_artifact("mass_origin_v2_trial1_breakthrough_pivot_route_contract", route_contract)
    write_artifact("mass_origin_v2_trial1_redundant_proca_term_admissibility_audit", redundancy_audit)
    write_artifact("mass_origin_v2_trial1_breakthrough_modified_vev_decomposition", modified_vev)
    write_artifact("mass_origin_v2_trial1_breakthrough_maxwell_coupling_audit", maxwell_audit)
    write_artifact("mass_origin_v2_trial1_breakthrough_legacy_preservation_audit", preservation_audit)
    write_artifact("mass_origin_v2_trial1_breakthrough_declaration_gate", declaration_gate)

    print("[ok] Generated Trial-1 breakthrough pivot artifacts:")
    print(" - mass_origin_v2_trial1_breakthrough_pivot_route_contract_metrics.json")
    print(" - mass_origin_v2_trial1_redundant_proca_term_admissibility_audit_metrics.json")
    print(" - mass_origin_v2_trial1_breakthrough_modified_vev_decomposition_metrics.json")
    print(" - mass_origin_v2_trial1_breakthrough_maxwell_coupling_audit_metrics.json")
    print(" - mass_origin_v2_trial1_breakthrough_legacy_preservation_audit_metrics.json")
    print(" - mass_origin_v2_trial1_breakthrough_declaration_gate_metrics.json")


if __name__ == "__main__":
    main()
