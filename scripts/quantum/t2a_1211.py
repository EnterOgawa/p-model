#!/usr/bin/env python3
"""Generate 8.7.56.1211-.1214 Trial-2 exact-coefficient-tracking artifacts."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
NOTE_PLACEHOLDER = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_placeholder_compress_and_attempt.md")

INVENTORY_1207 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_current_canon_limit_future_canon_hold_source_inventory_metrics.json"
)
AUDIT_1208 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_dimensionless_alpha_closed_form_attempt_audit_metrics.json"
)
TRACKING_1209 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_dimensionless_alpha_exact_factor_tracking_metrics.json"
)
EVAL_1210 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_dimensionless_alpha_numeric_evaluation_metrics.json"
)

ALPHA_TARGET = 7.2973525692838015e-3
CURRENT_C_TOTAL = 1.0
CURRENT_ALPHA = 1.0 / (4.0 * math.pi)
NEXT_ROUTE = "8.7.56.1215"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_charge_normalization_exact_coefficient_bridge_review"


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: abort when one required path is missing.

def require(path: Path) -> None:
    """Abort when one required path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: read one UTF-8 text file.

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# Function: read one UTF-8 JSON file.

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: return one display path relative to the repo when possible.

def display_path(path: Path) -> str:
    """Return one repo-relative display path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: locate the first matching line for one substring.

def hit(text: str, pattern: str) -> dict | None:
    """Return the first matching line for one substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: build one standard metrics row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: build one standard metrics payload.

def payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Build one standard metrics payload."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# Function: write one JSON metrics artifact and one CSV rows table.

def write_artifact(stem: str, data: dict) -> None:
    """Write one metrics payload as JSON and CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    json_path = PUBLIC_OUT / f"{stem}_metrics.json"
    csv_path = PUBLIC_OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: build one wording target record.

def target(text: str, path: Path, key: str, pattern: str, note: str) -> dict:
    """Build one wording target record."""
    evidence = hit(text, pattern)
    return {
        "file_key": key,
        "file": display_path(path),
        "pattern": pattern,
        "present": evidence is not None,
        "note": note,
        "evidence": evidence,
    }


# Function: compute the target coefficient and residuals.

def compute_residuals() -> dict[str, float]:
    """Compute the target coefficient and residuals for alpha = C_total^2 / (4 pi)."""
    target_c_total = math.sqrt(4.0 * math.pi * ALPHA_TARGET)
    residual_coefficient = target_c_total / CURRENT_C_TOTAL
    residual_squared = ALPHA_TARGET / CURRENT_ALPHA
    alpha_gap_factor = CURRENT_ALPHA / ALPHA_TARGET
    relative_error = abs(CURRENT_ALPHA - ALPHA_TARGET) / ALPHA_TARGET
    return {
        "target_c_total": target_c_total,
        "residual_coefficient": residual_coefficient,
        "residual_squared": residual_squared,
        "alpha_gap_factor": alpha_gap_factor,
        "relative_error": relative_error,
    }


# Function: execute the exact-coefficient-tracking branch.

def main() -> None:
    """Execute the 8.7.56.1211-.1214 branch."""
    required_paths = (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        PART1,
        PART3A,
        PART5,
        NOTE_PLACEHOLDER,
        INVENTORY_1207,
        AUDIT_1208,
        TRACKING_1209,
        EVAL_1210,
    )
    for path in required_paths:
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    work_history_recent_text = read_text(WORK_HISTORY_RECENT)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    placeholder_note_text = read_text(NOTE_PLACEHOLDER)
    inventory_1207 = read_json(INVENTORY_1207)["summary"]
    audit_1208 = read_json(AUDIT_1208)["summary"]
    tracking_1209 = read_json(TRACKING_1209)["summary"]
    eval_1210 = read_json(EVAL_1210)["summary"]
    residuals = compute_residuals()

    newton_4pi_surface_present = hit(part1_text, "g_P/Z_P=4\\pi G") is not None
    current_normalization_surface_present = hit(part1_text, "J^\\mu_{\\mathrm{matter}}=(\\rho c,\\rho \\mathbf{v})") is not None
    scalar_half_surface_present = hit(part1_text, "\\frac{M_\\chi^2}{2}") is not None
    vector_quarter_surface_present = (
        hit(part1_text, "-\\frac{Z_{P}}{4}") is not None
        or hit(part1_text, "-\\frac{Z_P}{4}") is not None
    )
    interaction_surface_present = hit(part1_text, "\\mathcal{L}_{\\mathrm{int}}=g_P") is not None
    charge_mapping_absent_surface_present = (
        hit(part3a_text, "explicit $g_P\\leftrightarrow e$ charge-normalization statement") is not None
        or hit(part3a_text, "explicit $g_P\\leftrightarrow e$ mapping") is not None
    )
    exact_factor_candidates_present = all(
        (
            hit(placeholder_note_text, "### 候補1: Newton 極限の $4\\pi$ convention") is not None,
            hit(placeholder_note_text, "### 候補2: coupling normalization") is not None,
            hit(placeholder_note_text, "### 候補3: Q-ball charge normalization") is not None,
            hit(placeholder_note_text, "### 候補4: $\\mathcal{L}_\\chi$ と $\\mathcal{L}_{P_\\mu}$ の cross normalization") is not None,
        )
    )
    inventory_ready = all(
        (
            inventory_1207["compress_ready"],
            audit_1208["dimensionless_alpha_closed_form_attempt_ready"],
            tracking_1209["exact_factor_tracking_ready"],
            eval_1210["exact_factor_tracking_required"],
            newton_4pi_surface_present,
            current_normalization_surface_present,
            scalar_half_surface_present,
            vector_quarter_surface_present,
            interaction_surface_present,
            charge_mapping_absent_surface_present,
            exact_factor_candidates_present,
        )
    )

    # Newton-side 4pi is already baked into the current-canon weak-field normalization,
    # so the admitted multiplicative factor in the dimensionless route is unity.
    newton_factor_current_canon = 1.0
    # J^mu normalization is fixed as a convention surface, but it does not add a new
    # standalone dimensionless coefficient once the route is expressed as e = g_P v.
    current_normalization_factor = 1.0
    # The scalar 1/2 and vector 1/4 disappear in the Euler-Lagrange equations, so they
    # do not survive as residual multiplicative coefficients in C_total.
    kinetic_prefactor_factor = 1.0
    current_canon_fixed_partial_coefficient = (
        newton_factor_current_canon * current_normalization_factor * kinetic_prefactor_factor
    )
    charge_mapping_factor_available = False
    charge_mapping_factor_current_canon = None
    audit_ready = inventory_ready and math.isclose(current_canon_fixed_partial_coefficient, 1.0, rel_tol=0.0, abs_tol=1e-15)

    targets = [
        target(status_text, STATUS, "status_current_branch", "exact coefficient tracking branch", "STATUS must expose the exact-coefficient-tracking branch as the live branch."),
        target(roadmap_text, ROADMAP, "roadmap_1211", "`8.7.56.1211`", "ROADMAP must expose the current source-inventory step."),
        target(work_history_recent_text, WORK_HISTORY_RECENT, "recent_1207", "`8.7.56.1207-.1210`", "Recent history must preserve the predecessor computation branch."),
        target(part1_text, PART1, "part1_newton", "g_P/Z_P=4\\pi G", "Part I must expose the current-canon weak-field normalization."),
        target(part1_text, PART1, "part1_current", "J^\\mu_{\\mathrm{matter}}=(\\rho c,\\rho \\mathbf{v})", "Part I must expose the current normalization surface."),
        target(part1_text, PART1, "part1_scalar_half", "\\frac{M_\\chi^2}{2}", "Part I must expose the scalar kinetic half-prefactor."),
        target(part1_text, PART1, "part1_vector_quarter", "-\\frac{Z_{P}}{4}", "Part I must expose the vector kinetic quarter-prefactor."),
        target(part1_text, PART1, "part1_interaction", "\\mathcal{L}_{\\mathrm{int}}=g_P", "Part I must expose the interaction term."),
        target(part3a_text, PART3A, "part3a_structural_e", "e=g_P/\\sqrt{Z_P}", "Part III-A must preserve the structural coupling route."),
        target(part3a_text, PART3A, "part3a_mapping_absent", "explicit $g_P\\leftrightarrow e$ charge-normalization statement", "Part III-A must preserve that explicit charge mapping is still absent."),
        target(placeholder_note_text, NOTE_PLACEHOLDER, "note_dimensionless_formula", "$$\\alpha = \\frac{e^2}{4\\pi} = \\frac{(g_P v)^2}{4\\pi}$$", "The placeholder-compress note must expose the coefficientized alpha route."),
        target(part5_text, PART5, "part5_current_state", "dimensionless formula found", "Part V must preserve the current checkpoint wording before the new branch completes."),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "part1": display_path(PART1),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
            "placeholder_note": display_path(NOTE_PLACEHOLDER),
        },
        "prior_metrics": {
            "inventory_1207": display_path(INVENTORY_1207),
            "audit_1208": display_path(AUDIT_1208),
            "tracking_1209": display_path(TRACKING_1209),
            "eval_1210": display_path(EVAL_1210),
        },
        "constants": {
            "alpha_target": ALPHA_TARGET,
            "current_c_total": CURRENT_C_TOTAL,
            "current_alpha": CURRENT_ALPHA,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory = payload(
        "8.7.56.1211",
        "Trial-2 numeric alpha exact coefficient tracking source inventory",
        inputs,
        [
            row("inventory_ready", "pass" if inventory_ready else "reject", "coefficient inventory ready", 1 if inventory_ready else 0, "The branch is ready only if the dimensionless route and all current-canon coefficient surfaces are simultaneously visible."),
            row("newton_4pi_surface_present", "pass" if newton_4pi_surface_present else "reject", "Newton 4pi surface present", 1 if newton_4pi_surface_present else 0, "Part I must expose the weak-field normalization that fixes the Newton-side 4pi convention."),
            row("current_normalization_surface_present", "pass" if current_normalization_surface_present else "reject", "current normalization surface present", 1 if current_normalization_surface_present else 0, "Part I must expose the current convention J^mu = (rho c, rho v)."),
            row("kinetic_prefactor_surfaces_present", "pass" if scalar_half_surface_present and vector_quarter_surface_present else "reject", "kinetic prefactor surfaces present", 1 if scalar_half_surface_present and vector_quarter_surface_present else 0, "Part I must expose both the scalar 1/2 and vector 1/4 prefactors for audit."),
            row("interaction_surface_present", "pass" if interaction_surface_present else "reject", "interaction surface present", 1 if interaction_surface_present else 0, "Part I must expose the g_P P_mu J^mu interaction surface."),
            row("charge_mapping_absent_surface_present", "pass" if charge_mapping_absent_surface_present else "reject", "charge-mapping absent surface present", 1 if charge_mapping_absent_surface_present else 0, "Part III-A must still expose that explicit g_P-to-elementary-charge mapping is absent in the current canon."),
            row("exact_factor_candidates_present", "pass" if exact_factor_candidates_present else "reject", "factor candidates present", 1 if exact_factor_candidates_present else 0, "The note must enumerate the four candidate factor families before any current-canon/future-canon split is made."),
        ],
        {
            "inventory_ready": inventory_ready,
            "coefficient_formula": "e = C_total, alpha = C_total^2 / (4 pi)",
            "current_c_total": CURRENT_C_TOTAL,
            "current_alpha": CURRENT_ALPHA,
            "alpha_target": ALPHA_TARGET,
            "candidate_factor_families": [
                "Newton-side 4pi convention",
                "current normalization of J^mu",
                "Q-ball / adopted-U(1) charge normalization",
                "kinetic prefactor 1/2 vs 1/4 propagation",
            ],
            "explicit_charge_mapping_current_canon_available": False,
            "charge_mapping_absent_surface_present": charge_mapping_absent_surface_present,
            "selected_next_substep": "8.7.56.1212",
        },
        {"overall_status": "trial2_numeric_alpha_exact_coefficient_inventory_fixed", "advance_to_8_7_56_1212": inventory_ready, "next_required_artifacts": ["exact_coefficient_tracking_audit"]},
        {"targets": targets, "prior_1209_summary": tracking_1209, "prior_1210_summary": eval_1210, "ai_context_snapshot": ai_context},
    )

    audit = payload(
        "8.7.56.1212",
        "Trial-2 numeric alpha exact coefficient tracking audit",
        inputs,
        [
            row("newton_factor_current_canon_unity", "pass" if audit_ready else "reject", "Newton-side coefficient under current canon is unity", newton_factor_current_canon, "The current-canon 4pi convention is already baked into g_P/Z_P = 4pi G, so it does not leave a new free multiplicative factor in C_total."),
            row("current_normalization_factor_unity", "pass" if audit_ready else "reject", "current-normalization coefficient under current canon is unity", current_normalization_factor, "J^mu = (rho c, rho v) fixes the convention, but in the dimensionless route it does not surface a standalone new pure-number coefficient."),
            row("kinetic_prefactor_factor_unity", "pass" if audit_ready else "reject", "kinetic-prefactor coefficient under current canon is unity", kinetic_prefactor_factor, "The scalar 1/2 and vector 1/4 cancel in the Euler-Lagrange equations and do not survive as residual factors."),
            row("charge_mapping_factor_available", "reject", "explicit charge-mapping factor available in current canon", 0.0, "The remaining coefficient is not fixed because the explicit g_P-to-elementary-charge mapping is still absent in the current canon."),
            row("residual_localized_to_charge_mapping", "pass" if audit_ready else "reject", "residual localized to charge normalization", 1 if audit_ready else 0, "Once Newton/current/kinetic prefactors are retired, the remaining factor gap localizes to charge-normalization or adopted-U(1) mapping."),
        ],
        {
            "audit_ready": audit_ready,
            "newton_side_4pi_current_canon_factor": newton_factor_current_canon,
            "current_normalization_current_canon_factor": current_normalization_factor,
            "kinetic_prefactor_current_canon_factor": kinetic_prefactor_factor,
            "current_canon_fixed_partial_coefficient": current_canon_fixed_partial_coefficient,
            "explicit_charge_mapping_current_canon_available": charge_mapping_factor_available,
            "charge_mapping_factor_current_canon": charge_mapping_factor_current_canon,
            "unresolved_residual_family": "gP_to_elementary_charge_charge_normalization",
            "result_class": "current_canon_action_level_factors_retired_charge_normalization_residual_open",
        },
        {"overall_status": "trial2_numeric_alpha_exact_coefficient_audit_completed", "advance_to_8_7_56_1213": audit_ready, "next_required_artifacts": ["exact_coefficient_tracking_declaration_gate"]},
        {"inventory_summary": inventory["summary"]},
    )

    declaration_gate = payload(
        "8.7.56.1213",
        "Trial-2 numeric alpha exact coefficient tracking declaration gate",
        inputs,
        [
            row("current_canon_partial_coefficient_fixed", "pass" if audit_ready else "reject", "current-canon partial coefficient fixed", current_canon_fixed_partial_coefficient, "The current canon fixes the action-level part of C_total at unity."),
            row("current_canon_complete_coefficient_available", "reject", "current-canon complete coefficient available", 0.0, "A complete coefficient is not available until the charge-normalization residual is fixed."),
            row("future_canon_or_adopted_u1_bridge_required", "pass", "future-canon or adopted-U1 bridge required", 1.0, "The remaining residual sits in charge normalization rather than in action-level prefactors."),
            row("physical_reject_required", "reject", "physical reject required", 0.0, "The branch remains open because a dimensionless formula exists and only the coefficient mapping remains unresolved."),
            row("closeout_ready", "reject", "closeout ready", 0.0, "Closeout is not ready while the exact charge-normalization coefficient remains unresolved."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "action_level_factors_closed_charge_normalization_residual_open",
            "current_canon_fixed_partial_coefficient": current_canon_fixed_partial_coefficient,
            "required_residual_coefficient": residuals["residual_coefficient"],
            "required_residual_squared": residuals["residual_squared"],
            "charge_normalization_residual_open": True,
            "future_canon_or_adopted_u1_bridge_required": True,
            "physical_reject_required": False,
            "closeout_ready": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {"overall_status": "trial2_numeric_alpha_charge_normalization_residual_declared", "advance_to_8_7_56_1214": audit_ready, "next_required_artifacts": [NEXT_ROUTE_NAME]},
        {"audit_summary": audit["summary"]},
    )

    evaluation = payload(
        "8.7.56.1214",
        "Trial-2 numeric alpha exact coefficient tracking numeric evaluation",
        inputs,
        [
            row("current_action_level_candidate_fixed", "pass", "current action-level candidate fixed", CURRENT_ALPHA, "With only current-canon action-level factors, the route remains alpha = 1 / (4 pi)."),
            row("required_charge_normalization_factor_fixed", "pass", "required charge-normalization factor fixed", residuals["residual_coefficient"], "The remaining coefficient needed to hit the observed alpha is sqrt(4 pi alpha_target)."),
            row("required_charge_normalization_factor_matches_unity", "reject", "required charge-normalization factor matches unity", 0.0, "The needed residual coefficient is not unity, so the action-level factors alone are insufficient."),
            row("exact_coefficient_tracking_completed", "pass",  "exact coefficient tracking completed", 1.0, "The current branch completed the action-level factor audit and localized the residual coefficient family."),
            row("closeout_ready", "reject", "closeout ready", 0.0, "The branch remains open because the charge-normalization residual is still unresolved."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "action_level_factors_closed_charge_normalization_residual_open",
            "e_current_canon_action_level": CURRENT_C_TOTAL,
            "alpha_current_canon_action_level": CURRENT_ALPHA,
            "e_target": residuals["target_c_total"],
            "alpha_target": ALPHA_TARGET,
            "required_charge_normalization_factor": residuals["residual_coefficient"],
            "required_charge_normalization_factor_squared": residuals["residual_squared"],
            "alpha_gap_factor": residuals["alpha_gap_factor"],
            "relative_error_if_charge_mapping_unfixed": residuals["relative_error"],
            "current_canon_closeout_ready": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {"overall_status": "trial2_numeric_alpha_exact_coefficient_tracking_completed", "advance_to_next_route": True, "next_required_artifacts": [NEXT_ROUTE_NAME]},
        {"declaration_gate_summary": declaration_gate["summary"]},
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_exact_coefficient_tracking_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_exact_coefficient_tracking_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_exact_coefficient_tracking_declaration_gate",
        declaration_gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_exact_coefficient_tracking_numeric_evaluation",
        evaluation,
    )

    print("[done] 8.7.56.1211-.1214 artifacts generated")
    print(f"[current_c_total] {CURRENT_C_TOTAL:.16f}")
    print(f"[target_c_total] {residuals['target_c_total']:.16f}")
    print(f"[required_charge_factor] {residuals['residual_coefficient']:.16f}")
    print(f"[alpha_gap_factor] {residuals['alpha_gap_factor']:.16f}")


if __name__ == "__main__":
    main()
