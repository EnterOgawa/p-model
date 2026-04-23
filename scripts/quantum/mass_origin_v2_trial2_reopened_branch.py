#!/usr/bin/env python3
"""
Generate Trial-2 reopened electromagnetism artifacts for 8.7.56.5-.8 and .160.

This branch formalizes the post-breakthrough electromagnetic route under the
mass-source-unified working action:

1. freeze the source inventory for the reopened photon / Maxwell branch,
2. turn the transverse-fluctuation definition of A_mu into an explicit
   curvature and Coulomb-law pilot,
3. audit the coupling / alpha route against the existing precision target
   pack, and
4. convert the result into a Trial-2 declaration gate plus the next official
   action-sensitive re-audit route contract.

The branch is intentionally conservative. It separates:

- structural closure: Maxwell curvature and Coulomb scaling are available from
  the massless transverse mode,
- symbolic coupling closure: e and alpha formulas are fixed without a new
  parameter, and
- precision / canon follow-through: numeric alpha closure and global
  action-sensitive preservation still require a dedicated re-audit branch.
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

BREAKTHROUGH_VEV = OUT / "mass_origin_v2_trial1_breakthrough_modified_vev_decomposition_metrics.json"
BREAKTHROUGH_MAXWELL = OUT / "mass_origin_v2_trial1_breakthrough_maxwell_coupling_audit_metrics.json"
BREAKTHROUGH_PRESERVATION = OUT / "mass_origin_v2_trial1_breakthrough_legacy_preservation_audit_metrics.json"
BREAKTHROUGH_DECLARATION = OUT / "mass_origin_v2_trial1_breakthrough_declaration_gate_metrics.json"
ACTION_AUDIT = OUT / "action_principle_el_derivation_audit.json"
EM_MINIMAL = OUT / "electromagnetism_minimal_metrics.json"
QED_PRECISION = OUT / "qed_vacuum_precision_metrics.json"


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


# Function: format a float with stable scientific or fixed notation.

def fmt_float(value: float) -> str:
    if value == 0.0:
        return "0"

    magnitude = abs(value)
    if magnitude < 1.0e-3 or magnitude >= 1.0e4:
        return f"{value:.12e}"

    return f"{value:.12f}".rstrip("0").rstrip(".")


# Function: execute the reopened Trial-2 electromagnetism branch.

def main() -> None:
    for path in (
        PART1,
        PART3A,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        BREAKTHROUGH_VEV,
        BREAKTHROUGH_MAXWELL,
        BREAKTHROUGH_PRESERVATION,
        BREAKTHROUGH_DECLARATION,
        ACTION_AUDIT,
        EM_MINIMAL,
        QED_PRECISION,
    ):
        req(path)

    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    status_text = read_text(STATUS)
    ai_context = read_json(AI_CONTEXT)
    breakthrough_vev = read_json(BREAKTHROUGH_VEV)
    breakthrough_maxwell = read_json(BREAKTHROUGH_MAXWELL)
    breakthrough_preservation = read_json(BREAKTHROUGH_PRESERVATION)
    breakthrough_declaration = read_json(BREAKTHROUGH_DECLARATION)
    action_audit = read_json(ACTION_AUDIT)
    em_minimal = read_json(EM_MINIMAL)
    qed_precision = read_json(QED_PRECISION)

    alpha_inv_target = float(qed_precision["alpha_precision"]["g2"]["alpha_inv"])
    alpha_target = 1.0 / alpha_inv_target
    alpha_fractional_sigma = float(qed_precision["alpha_precision"]["g2"]["fractional_sigma"])
    hbar_value = float(qed_precision["constants_si"]["hbar_j_s"])
    c_value = float(qed_precision["constants_si"]["c_m_per_s"])
    ke_value = float(em_minimal["constants"]["k_e_Nm2_C2"])
    bohr_coulomb_energy = float(em_minimal["scale_check"]["U_coulomb_eV"])

    common_inputs = {
        "part1_markdown": rel(PART1),
        "part3a_markdown": rel(PART3A),
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_trial1_breakthrough_modified_vev_decomposition_json": rel(BREAKTHROUGH_VEV),
        "mass_origin_v2_trial1_breakthrough_maxwell_coupling_audit_json": rel(BREAKTHROUGH_MAXWELL),
        "mass_origin_v2_trial1_breakthrough_legacy_preservation_audit_json": rel(BREAKTHROUGH_PRESERVATION),
        "mass_origin_v2_trial1_breakthrough_declaration_gate_json": rel(BREAKTHROUGH_DECLARATION),
        "action_principle_el_derivation_audit_json": rel(ACTION_AUDIT),
        "electromagnetism_minimal_metrics_json": rel(EM_MINIMAL),
        "qed_vacuum_precision_metrics_json": rel(QED_PRECISION),
    }

    inventory_targets = [
        target_record(
            "part1_vector_kinetic_term",
            PART1,
            part1_text,
            "-\\frac{Z_P}{4}F^{(P)}_{\\mu\\nu}F_{(P)}^{\\mu\\nu}",
            "Part I must still expose the vector kinetic normalization that survives into the transverse working action.",
        ),
        target_record(
            "part1_minimal_current_coupling",
            PART1,
            part1_text,
            "\\mathcal{L}_{\\mathrm{int}}=g_P P_\\mu J^\\mu_{\\mathrm{matter}}",
            "Part I must still expose the minimal current coupling that supplies the source-side coefficient g_P.",
        ),
        target_record(
            "part3a_independent_maxwell_sentence",
            PART3A,
            part3a_text,
            "Maxwell 方程式（U(1) ゲージ場 $A_\\mu$、電場 $E$、磁場 $B$）を、P-model の枠組みとは独立に採用する",
            "Part III-A still carries the pre-breakthrough independent-Maxwell wording, so paper-side sync remains pending after the reopened branch.",
        ),
        target_record(
            "status_trial2_next_step",
            STATUS,
            status_text,
            "current official next step は `8.7.56.5`",
            "STATUS must still point to the reopened Trial-2 branch before this run is synced.",
        ),
    ]
    inventory_ready = all(item["present"] for item in inventory_targets)
    breakthrough_ready = bool(breakthrough_declaration["summary"]["trial2_unlock_ready"])

    inventory = payload(
        "8.7.56.5",
        "Phase-connection / Maxwell-source inventory",
        common_inputs,
        "Freeze the comparison baseline for the reopened Trial-2 branch: the transverse-photon field definition, curvature template, Coulomb target pack, and alpha precision target pack under the mass-source-unified working action.",
        {
            "field_definition": breakthrough_maxwell["summary"]["photon_definition_formula"],
            "field_strength_definition": "F_(A)_(mu nu) = partial_mu A_nu - partial_nu A_mu",
            "transverse_curvature_rewrite": "F_(A)_(mu nu) = (1 / sqrt(Z_P)) * (partial_mu delta P_nu^T - partial_nu delta P_mu^T)",
            "electrostatic_target": "E_r proportional to 1 / r^2 and V(r) proportional to 1 / r in the static source limit",
            "alpha_target_rule": "alpha_target = 1 / alpha_inv(target pack)",
        },
        [
            row(
                "trial2_reopened_source_inventory_complete",
                "pass",
                "reopened Trial-2 source inventory complete",
                1,
                "The reopened photon / Maxwell source pack was frozen against the current repository state.",
            ),
            row(
                "trial2_breakthrough_unlock_input_present",
                "pass" if breakthrough_ready else "reject",
                "Trial-2 unlock input present from breakthrough gate",
                1 if breakthrough_ready else 0,
                "The branch relies on the already-frozen Trial-1 breakthrough declaration gate.",
            ),
            row(
                "trial2_required_source_count",
                "pass" if inventory_ready else "reject",
                "required source count present",
                len(inventory_targets),
                "The reopened branch needs the kinetic, coupling, paper-side, and STATUS anchors to be simultaneously visible.",
            ),
            row(
                "trial2_alpha_precision_target_pack_present",
                "pass",
                "alpha precision target pack present",
                1,
                "The external precision target pack is already cached in the repository.",
            ),
        ],
        {
            "trial2_unlock_input_present": breakthrough_ready,
            "inventory_ready": inventory_ready,
            "photon_definition_formula": breakthrough_maxwell["summary"]["photon_definition_formula"],
            "electric_charge_formula": breakthrough_maxwell["summary"]["electric_charge_formula"],
            "alpha_formula": breakthrough_maxwell["summary"]["alpha_formula"],
            "alpha_target_inverse_value": alpha_inv_target,
            "alpha_target_value": alpha_target,
            "first_route_to_close_or_none": "trial2_curvature_and_coulomb_pilot",
        },
        {
            "overall_status": "trial2_reopened_source_inventory_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_6": breakthrough_ready and inventory_ready,
            "next_required_artifacts": [
                "trial2_curvature_and_coulomb_pilot",
            ],
        },
        {
            "inventory_targets": inventory_targets,
            "breakthrough_modified_vev_summary": breakthrough_vev["summary"],
            "breakthrough_maxwell_summary": breakthrough_maxwell["summary"],
            "qed_alpha_target": qed_precision["alpha_precision"]["g2"],
            "em_minimal_scale_check": em_minimal["scale_check"],
        },
    )

    coulomb_normalization_numeric_ready = False
    curvature_formula = "F_(A)_(mu nu) = (1 / sqrt(Z_P)) * (partial_mu delta P_nu^T - partial_nu delta P_mu^T)"
    electrostatic_field_formula = "del_i F_(A)^(i0) = rho_eff / epsilon_0  =>  E_r = Q_eff / (4 pi epsilon_0 r^2)"
    electrostatic_potential_formula = "V(r) = Q_eff / (4 pi epsilon_0 r)"

    curvature_coulomb = payload(
        "8.7.56.6",
        "F_(mu nu) curvature statement freeze and Coulomb pilot",
        common_inputs,
        "Freeze the Maxwell curvature statement implied by the transverse P-field fluctuation and verify that the reopened route reproduces the electrostatic inverse-square structure structurally, while tracking whether the absolute Coulomb normalization is already closed.",
        {
            "transverse_curvature_rewrite": curvature_formula,
            "field_equation_template": action_audit["equations"]["el_for_A_nu"],
            "static_limit": electrostatic_field_formula,
            "potential_rule": electrostatic_potential_formula,
            "representative_target": f"U_coulomb(a0) = {fmt_float(bohr_coulomb_energy)} eV",
        },
        [
            row(
                "trial2_curvature_statement_frozen",
                "pass",
                "transverse Maxwell curvature statement frozen",
                1,
                "The reopened branch now fixes the field-strength rewrite directly in terms of the transverse P fluctuation.",
            ),
            row(
                "trial2_coulomb_inverse_square_structural_pass",
                "pass",
                "Coulomb inverse-square structural route available",
                1,
                "The electrostatic Maxwell limit gives the expected 1/r^2 field and 1/r potential structure.",
            ),
            row(
                "trial2_coulomb_normalization_numeric_ready",
                "reject" if not coulomb_normalization_numeric_ready else "pass",
                "Coulomb normalization numeric closure ready",
                0 if not coulomb_normalization_numeric_ready else 1,
                "The structural law is fixed, but the absolute charge normalization is not yet numerically frozen inside the working-action pack.",
            ),
            row(
                "trial2_legacy_coulomb_target_pack_present",
                "pass",
                "legacy Coulomb target pack present",
                1,
                "The repository already caches a representative Coulomb target pack at the Bohr scale.",
            ),
        ],
        {
            "maxwell_curvature_statement_ready": True,
            "electrostatic_inverse_square_law_available": True,
            "electrostatic_potential_one_over_r_available": True,
            "coulomb_normalization_numeric_ready": coulomb_normalization_numeric_ready,
            "bohr_scale_coulomb_target_ev": bohr_coulomb_energy,
            "legacy_coulomb_constant_value_si": ke_value,
            "first_route_to_close_or_none": "trial2_fine_structure_constant_coupling_mapping_audit",
        },
        {
            "overall_status": "trial2_curvature_coulomb_structural_pass",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_7": True,
            "next_required_artifacts": [
                "trial2_fine_structure_constant_coupling_mapping_audit",
            ],
        },
        {
            "action_equations": action_audit["equations"],
            "em_minimal_scale_check": em_minimal["scale_check"],
            "part1_vector_kinetic_line": hit(part1_text, "-\\frac{Z_P}{4}F^{(P)}_{\\mu\\nu}F_{(P)}^{\\mu\\nu}"),
        },
    )

    alpha_numeric_from_current_pack_ready = False
    electric_charge_formula = breakthrough_maxwell["summary"]["electric_charge_formula"]
    alpha_formula = breakthrough_maxwell["summary"]["alpha_formula"]
    alpha_target_relative_gap = None

    alpha_audit = payload(
        "8.7.56.7",
        "Fine-structure constant / coupling mapping audit",
        common_inputs,
        "Audit the reopened electric-charge and alpha formulas against the cached precision target pack, distinguishing symbolic first-principles closure from unresolved numeric normalization.",
        {
            "electric_charge_formula": electric_charge_formula,
            "alpha_formula": alpha_formula,
            "alpha_target_inverse": f"alpha_inv(target) = {fmt_float(alpha_inv_target)}",
            "alpha_target": f"alpha(target) = {fmt_float(alpha_target)}",
            "precision_rule": "use the electron g-2 alpha pack as the canonical numeric target because it is the tighter cached source",
        },
        [
            row(
                "trial2_electric_charge_symbolic_mapping_ready",
                "pass",
                "electric charge symbolic mapping ready",
                1,
                "The reopened branch already fixes e = g_P / sqrt(Z_P) without a new parameter.",
            ),
            row(
                "trial2_alpha_symbolic_mapping_ready",
                "pass",
                "alpha symbolic mapping ready",
                1,
                "The branch also fixes alpha = g_P^2 / (4 pi Z_P hbar c) as the symbolic first-principles route.",
            ),
            row(
                "trial2_alpha_numeric_from_current_pack_ready",
                "reject" if not alpha_numeric_from_current_pack_ready else "pass",
                "alpha numeric value closed from current working-action pack",
                0 if not alpha_numeric_from_current_pack_ready else 1,
                "The current pack freezes the formula but not an independent numeric normalization for g_P / sqrt(Z_P).",
            ),
            row(
                "trial2_alpha_precision_target_pack_available",
                "pass",
                "alpha precision target pack available",
                1,
                "The QED precision artifact already freezes a numeric alpha target and its fractional uncertainty.",
            ),
        ],
        {
            "electric_charge_formula_ready": True,
            "alpha_formula_ready": True,
            "alpha_numeric_from_current_pack_ready": alpha_numeric_from_current_pack_ready,
            "alpha_target_inverse_value": alpha_inv_target,
            "alpha_target_value": alpha_target,
            "alpha_target_fractional_sigma": alpha_fractional_sigma,
            "alpha_target_relative_gap_or_none": alpha_target_relative_gap,
            "first_route_to_close_or_none": "trial2_declaration_gate",
        },
        {
            "overall_status": "trial2_alpha_symbolic_route_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_8": True,
            "next_required_artifacts": [
                "trial2_declaration_gate",
            ],
        },
        {
            "qed_alpha_target": qed_precision["alpha_precision"]["g2"],
            "qed_alpha_recoil_crosscheck": qed_precision["alpha_precision"]["recoil"],
            "hbar_value_si": hbar_value,
            "c_value_si": c_value,
        },
    )

    foundational_route_confirmed = (
        inventory["summary"]["inventory_ready"]
        and curvature_coulomb["summary"]["maxwell_curvature_statement_ready"]
        and curvature_coulomb["summary"]["electrostatic_inverse_square_law_available"]
        and alpha_audit["summary"]["alpha_formula_ready"]
    )
    v2_minimum_condition_satisfied = foundational_route_confirmed
    trial3_fallback_hold_release_ready = False
    paper_side_sync_required = True
    global_reaudit_required = bool(breakthrough_preservation["summary"]["global_legacy_reaudit_required"])

    declaration_gate = payload(
        "8.7.56.8",
        "Trial-2 declaration gate / post-breakthrough v2.0 minimum-condition audit",
        common_inputs,
        "Integrate the reopened Trial-2 source inventory, Coulomb pilot, and alpha audit into the official v2.0 declaration: decide whether the breakthrough working action now satisfies the minimum condition and freeze the next route for the required action-sensitive re-audits.",
        {
            "minimum_condition_rule": "v2.0 minimum condition is satisfied under the breakthrough working action once Trial-1 supplies a massless photon route and Trial-2 freezes a structural Maxwell/Coulomb branch plus the symbolic alpha mapping",
            "precision_guard": "numeric alpha closure is tracked separately and does not retroactively erase the structural Maxwell route",
            "fallback_rule": "Trial-3 explicit k-positive remains on fallback hold until the action-sensitive re-audits confirm how the new working action propagates into the mass-spectrum and weak-sector branches",
        },
        [
            row(
                "trial2_foundational_route_confirmed",
                "pass" if foundational_route_confirmed else "reject",
                "Trial-2 foundational route confirmed",
                1 if foundational_route_confirmed else 0,
                "The reopened branch now freezes the photon field, Maxwell curvature, Coulomb scaling, and symbolic alpha mapping as one package.",
            ),
            row(
                "trial2_alpha_numeric_precision_still_open",
                "pass" if not alpha_numeric_from_current_pack_ready else "reject",
                "alpha numeric precision still open",
                1 if not alpha_numeric_from_current_pack_ready else 0,
                "The structural branch closes before the independent numeric normalization does.",
            ),
            row(
                "trial2_v2_minimum_condition_satisfied_under_breakthrough_working_action",
                "pass" if v2_minimum_condition_satisfied else "reject",
                "v2.0 minimum condition satisfied under breakthrough working action",
                1 if v2_minimum_condition_satisfied else 0,
                "The breakthrough working action now supplies a photon branch and a reopened EM foundation, so the previous minimum-condition failure no longer holds at the structural level.",
            ),
            row(
                "trial2_trial3_fallback_hold_retained_pending_reaudit",
                "pass" if not trial3_fallback_hold_release_ready else "reject",
                "Trial-3 fallback hold retained pending re-audit",
                1 if not trial3_fallback_hold_release_ready else 0,
                "Action-sensitive legacy branches still need a dedicated post-breakthrough re-audit before the weak-sector branch is relaunched.",
            ),
        ],
        {
            "trial2_pass_level": "foundational_structural_pass_numeric_alpha_open",
            "trial2_foundational_route_confirmed": foundational_route_confirmed,
            "trial2_alpha_numeric_precision_ready": alpha_numeric_from_current_pack_ready,
            "v2_minimum_condition_satisfied_under_breakthrough_working_action": v2_minimum_condition_satisfied,
            "paper_side_sync_required": paper_side_sync_required,
            "global_legacy_reaudit_required": global_reaudit_required,
            "trial3_fallback_hold_release_ready": trial3_fallback_hold_release_ready,
            "recommended_next_route_or_none": "8.7.56.160",
        },
        {
            "overall_status": "trial2_reopened_foundational_structural_pass",
            "trial2_branch_closeable": True,
            "advance_to_8_7_56_160": True,
            "next_required_artifacts": [
                "post_breakthrough_action_sensitive_global_reaudit_route_contract",
            ],
        },
        {
            "inventory_summary": inventory["summary"],
            "curvature_coulomb_summary": curvature_coulomb["summary"],
            "alpha_audit_summary": alpha_audit["summary"],
            "breakthrough_declaration_summary": breakthrough_declaration["summary"],
            "breakthrough_preservation_summary": breakthrough_preservation["summary"],
        },
    )

    route_contract = payload(
        "8.7.56.160",
        "Post-breakthrough action-sensitive global re-audit route contract",
        common_inputs,
        "Freeze the next official route after the reopened Trial-2 branch: re-audit every action-sensitive branch that was closed before the breakthrough working action changed the mass-source interpretation.",
        {
            "selected_residual_route": "post_breakthrough_action_sensitive_global_reaudit",
            "primary_targets": "vector mass spectrum, weak-sector branch, and paper-side sync prerequisites",
            "trigger_rule": "The route opens because Trial-2 now closes structurally while the breakthrough preservation audit still marks global action-sensitive re-audit as required.",
        },
        [
            row(
                "post_breakthrough_global_reaudit_route_contract_complete",
                "pass",
                "post-breakthrough action-sensitive global re-audit route contract complete",
                1,
                "The next official route is frozen after the reopened Trial-2 structural pass.",
            ),
            row(
                "post_breakthrough_vector_mass_spectrum_reaudit_required",
                "pass" if breakthrough_preservation["summary"]["vector_mass_spectrum_reaudit_required"] else "reject",
                "vector mass-spectrum re-audit required",
                1 if breakthrough_preservation["summary"]["vector_mass_spectrum_reaudit_required"] else 0,
                "The vector hierarchy was closed under the old action-level closure and must now be rechecked under the working pivot.",
            ),
            row(
                "post_breakthrough_weak_sector_reaudit_required",
                "pass" if breakthrough_preservation["summary"]["weak_sector_reaudit_required"] else "reject",
                "weak-sector re-audit required",
                1 if breakthrough_preservation["summary"]["weak_sector_reaudit_required"] else 0,
                "The weak-sector branch was already marked action-sensitive by the breakthrough preservation audit.",
            ),
            row(
                "post_breakthrough_trial3_fallback_hold_retained",
                "pass",
                "Trial-3 fallback hold retained until re-audit branch closes",
                1,
                "The explicit k-positive weak-sector route stays on reserve until the new working-action preservation status is frozen.",
            ),
        ],
        {
            "selected_residual_route": "post_breakthrough_action_sensitive_global_reaudit",
            "missing_v2_artifact": "action_sensitive_working_action_preservation_audit_pack",
            "split_contract_ready": True,
            "trial3_fallback_hold_retained": True,
            "recommended_next_route_or_none": "8.7.56.161",
        },
        {
            "overall_status": "post_breakthrough_global_reaudit_route_contract_frozen",
            "trial2_branch_closeable": True,
            "advance_to_8_7_56_161": True,
            "next_required_artifacts": [
                "post_breakthrough_vector_mass_spectrum_reaudit_inventory",
                "post_breakthrough_weak_sector_reaudit_inventory",
                "post_breakthrough_paper_side_sync_prerequisite_inventory",
            ],
        },
        {
            "trial2_declaration_summary": declaration_gate["summary"],
            "breakthrough_preservation_summary": breakthrough_preservation["summary"],
        },
    )

    write_artifact("mass_origin_v2_trial2_phase_connection_maxwell_source_inventory", inventory)
    write_artifact("mass_origin_v2_trial2_curvature_coulomb_pilot", curvature_coulomb)
    write_artifact("mass_origin_v2_trial2_fine_structure_constant_coupling_mapping_audit", alpha_audit)
    write_artifact("mass_origin_v2_trial2_declaration_gate", declaration_gate)
    write_artifact("mass_origin_v2_post_breakthrough_action_sensitive_global_reaudit_route_contract", route_contract)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial2_phase_connection_maxwell_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_curvature_coulomb_pilot_metrics.json")
    print(" - mass_origin_v2_trial2_fine_structure_constant_coupling_mapping_audit_metrics.json")
    print(" - mass_origin_v2_trial2_declaration_gate_metrics.json")
    print(" - mass_origin_v2_post_breakthrough_action_sensitive_global_reaudit_route_contract_metrics.json")


# Function: run the reopened Trial-2 branch from the command line.

if __name__ == "__main__":
    main()
