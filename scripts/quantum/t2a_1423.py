#!/usr/bin/env python3
"""Generate unified-closure Phase 2 lambda_rot correction artifacts for 8.7.56.1423-.1426."""

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
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

UNIFIED_PLAN = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_unified_closure_plan_20260327.md")
NEXT_STEPS = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")
SOLVER_FIX = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_solver_fix_final.md")
PERTURBATIVE_NOTE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_perturbative_fL_correction.md")

PHASE1_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase1_exact_coupled_l0_solver_"
    "audit_metrics.json"
)
PHASE1_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase1_exact_coupled_l0_solver_"
    "numeric_evaluation_metrics.json"
)
LROT_AUDIT = PUBLIC_OUT / "lagrangian_noether_rotational_closure_audit.json"
SPIN_REUSE = PUBLIC_OUT / "mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json"
COUPLED_SOLVER_INVENTORY = PUBLIC_OUT / "mass_origin_vector_qball_coupled_solver_source_inventory_metrics.json"

ALPHA_TARGET = 1.0 / 137.035999084
TARGET_FORM_FACTOR = math.sqrt(4.0 * math.pi * ALPHA_TARGET)

PRIOR_CLASS = "vector_qball_form_factor_unified_closure_phase1_exact_coupled_l0_solver_phase2_required"
BRANCH_CLASS = "vector_qball_form_factor_unified_closure_phase2_lambda_rot_secondary_lane_no_go_phase3_required"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_blind_vector_observable_gate"
NEXT_ROUTE = "8.7.56.1427"
STEM = "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_lambda_rot_form_factor_correction"


# Function: return the current UTC timestamp string.
def now_iso() -> str:
    """Return the current UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


# Function: fail fast when one required path is missing.

def require(path: Path) -> None:
    """Fail fast when one required path is missing."""
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


# Function: convert one path into repo-relative display form when possible.

def display_path(path: Path) -> str:
    """Convert one path into repo-relative display form when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: return the first matching line for one substring.

def hit(text: str, pattern: str) -> dict | None:
    """Return the first matching line for one substring."""
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


# Function: write one JSON payload and CSV rows table.

def write_artifact(kind: str, data: dict) -> None:
    """Write one JSON payload and CSV rows table."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    json_path = PUBLIC_OUT / f"{STEM}_{kind}_metrics.json"
    csv_path = PUBLIC_OUT / f"{STEM}_{kind}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: build one standard payload object.

def payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Build one standard payload object."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# Function: compute one safe ratio or infinity.

def safe_ratio(numerator: float, denominator: float) -> float:
    """Compute one safe ratio or infinity."""
    if denominator == 0.0:
        return math.inf

    return float(numerator / denominator)


# Function: execute the unified-closure Phase 2 lambda_rot correction branch.

def main() -> None:
    """Execute the unified-closure Phase 2 lambda_rot correction branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PART1,
        PART3A,
        PART5,
        UNIFIED_PLAN,
        NEXT_STEPS,
        SOLVER_FIX,
        PERTURBATIVE_NOTE,
        PHASE1_AUDIT,
        PHASE1_EVAL,
        LROT_AUDIT,
        SPIN_REUSE,
        COUPLED_SOLVER_INVENTORY,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    unified_plan_text = read_text(UNIFIED_PLAN)
    next_steps_text = read_text(NEXT_STEPS)
    solver_fix_text = read_text(SOLVER_FIX)
    perturbative_note_text = read_text(PERTURBATIVE_NOTE)

    phase1_audit = read_json(PHASE1_AUDIT)["summary"]
    phase1_eval = read_json(PHASE1_EVAL)["summary"]
    lambda_audit = read_json(LROT_AUDIT)
    spin_reuse = read_json(SPIN_REUSE)["summary"]
    coupled_solver_inventory = read_json(COUPLED_SOLVER_INVENTORY)["summary"]

    if not bool(phase1_audit["phase2_required"]):
        raise SystemExit("[fail] Phase 2 branch was invoked while phase1_audit says phase2_required=false")

    phase1_best = phase1_eval["phase1_best_alpha_candidate"]
    phase1_f_signed = float(phase1_best["F_at_q_theory"])
    phase1_f_abs = abs(phase1_f_signed)
    phase1_alpha = float(phase1_best["alpha_at_q_theory"])

    lambda_rot = float(lambda_audit["calibration"]["lambda_rot"])
    lambda_sigma = float(lambda_audit["calibration"]["lambda_sigma"])
    lambda_abs_mean = abs(lambda_rot)
    lambda_abs_1sigma = lambda_abs_mean + lambda_sigma
    lambda_abs_3sigma = lambda_abs_mean + 3.0 * lambda_sigma

    ell0_direct_spin_factor = 1.0 + lambda_rot * 0.0 * 0.0
    required_abs_delta_f = TARGET_FORM_FACTOR - phase1_f_abs
    required_signed_delta_to_positive = TARGET_FORM_FACTOR - phase1_f_signed
    required_abs_multiplier = safe_ratio(TARGET_FORM_FACTOR, phase1_f_abs)
    required_alpha_multiplier = safe_ratio(ALPHA_TARGET, phase1_alpha)
    required_linear_multiplier_coeff_mean = safe_ratio(required_abs_multiplier - 1.0, lambda_abs_mean)
    required_linear_multiplier_coeff_1sigma = safe_ratio(required_abs_multiplier - 1.0, lambda_abs_1sigma)
    required_linear_multiplier_coeff_3sigma = safe_ratio(required_abs_multiplier - 1.0, lambda_abs_3sigma)
    required_additive_coeff_1sigma = safe_ratio(required_abs_delta_f, lambda_abs_1sigma)
    required_additive_coeff_3sigma = safe_ratio(required_abs_delta_f, lambda_abs_3sigma)

    best_naive_mult_3sigma_f = phase1_f_abs * (1.0 + lambda_abs_3sigma)
    best_naive_mult_3sigma_alpha = (best_naive_mult_3sigma_f * best_naive_mult_3sigma_f) / (4.0 * math.pi)
    best_naive_mult_3sigma_relerr = abs(best_naive_mult_3sigma_alpha - ALPHA_TARGET) / ALPHA_TARGET

    best_naive_add_3sigma_f = phase1_f_abs + lambda_abs_3sigma
    best_naive_add_3sigma_alpha = (best_naive_add_3sigma_f * best_naive_add_3sigma_f) / (4.0 * math.pi)
    best_naive_add_3sigma_relerr = abs(best_naive_add_3sigma_alpha - ALPHA_TARGET) / ALPHA_TARGET

    inventory_hits = [
        hit(status_text, "8.7.56.1423-.1426"),
        hit(roadmap_text, "8.7.56.1423-.1426"),
        hit(current_problem_text, "lambda_{\\rm rot}"),
        hit(current_status_text, "conditional Phase 2 lambda_rot correction"),
        hit(unified_roadmap_text, "lambda_rot form-factor correction"),
        hit(unified_plan_text, "Stage 2"),
        hit(next_steps_text, "Step C"),
        hit(solver_fix_text, "Minkowski sign"),
        hit(perturbative_note_text, "vector_form_factor_theorem_blocked = false"),
        hit(part1_text, "Pauli 型スピン結合"),
        hit(part3a_text, "effective source formula `J^\\mu_{\\rm eff}[P^{\\rm Qball}]` が still surface していない"),
        hit(part5_text, "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_lambda_rot_form_factor_correction"),
    ]
    inventory_ready = all(item is not None for item in inventory_hits)

    lambda_rot_surface_available = bool(
        lambda_audit["calibration"]["lambda_rot"] is not None and hit(part1_text, "Pauli 型スピン結合") is not None
    )
    spin_orbit_reuse_available = bool(spin_reuse["lambda_rot_reuse_available"])
    electron_identification_preserved = True
    ell0_direct_spin_factor_available = not math.isclose(ell0_direct_spin_factor, 1.0, rel_tol=0.0, abs_tol=1.0e-15)
    exact_j_eff_available = False
    phase2_naive_multiplicative_close_available = bool(best_naive_mult_3sigma_relerr < 0.01)
    phase2_naive_additive_close_available = bool(best_naive_add_3sigma_relerr < 0.01)
    phase2_secondary_lane_no_go = bool(
        lambda_rot_surface_available
        and spin_orbit_reuse_available
        and (
            (not ell0_direct_spin_factor_available and not exact_j_eff_available)
            or (not phase2_naive_multiplicative_close_available and not phase2_naive_additive_close_available)
        )
    )
    phase3_required = bool(phase2_secondary_lane_no_go)

    common_inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem_note": display_path(CURRENT_PROBLEM),
            "current_status_note": display_path(CURRENT_STATUS),
            "unified_closure_roadmap_note": display_path(UNIFIED_ROADMAP),
            "part1": display_path(PART1),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
            "unified_plan_note": display_path(UNIFIED_PLAN),
            "next_steps_note": display_path(NEXT_STEPS),
            "solver_fix_note": display_path(SOLVER_FIX),
            "perturbative_note": display_path(PERTURBATIVE_NOTE),
        },
        "source_metrics": {
            "phase1_audit": display_path(PHASE1_AUDIT),
            "phase1_eval": display_path(PHASE1_EVAL),
            "lambda_rot_audit": display_path(LROT_AUDIT),
            "spin_orbit_reuse": display_path(SPIN_REUSE),
            "coupled_solver_inventory": display_path(COUPLED_SOLVER_INVENTORY),
        },
        "constants": {
            "alpha_target": ALPHA_TARGET,
            "target_form_factor": TARGET_FORM_FACTOR,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory_payload = payload(
        "8.7.56.1423",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure Phase 2 lambda_rot source inventory",
        common_inputs,
        [
            row(
                "phase2_inventory_ready",
                "pass" if inventory_ready else "reject",
                "Phase 2 inventory ready",
                1 if inventory_ready else 0,
                "Phase 2 inventory is ready only if the Phase 1 outputs, lambda_rot prior surfaces, and current pack wording coexist in one source pack.",
            ),
            row(
                "lambda_rot_surface_available",
                "pass" if lambda_rot_surface_available else "reject",
                "lambda_rot surface available",
                1 if lambda_rot_surface_available else 0,
                "The conditional correction branch is admissible only if lambda_rot is already frozen in the current pack.",
            ),
            row(
                "spin_orbit_reuse_available",
                "pass" if spin_orbit_reuse_available else "reject",
                "spin-orbit reuse available",
                1 if spin_orbit_reuse_available else 0,
                "The same frozen lambda_rot must already be reusable in the vector Q-ball spin sector with no new free parameter.",
            ),
            row(
                "electron_identification_preserved",
                "pass" if electron_identification_preserved else "reject",
                "electron identification preserved",
                1 if electron_identification_preserved else 0,
                "Phase 2 stays conditional only if the electron identification (1,0,0,0) is preserved.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": PRIOR_CLASS,
            "phase2_inventory_ready": inventory_ready,
            "lambda_rot_surface_available": lambda_rot_surface_available,
            "spin_orbit_reuse_available": spin_orbit_reuse_available,
            "electron_identification_preserved": electron_identification_preserved,
            "selected_next_generation_route": NEXT_ROUTE_NAME if phase3_required else None,
            "recommended_next_route_or_none": NEXT_ROUTE if phase3_required else None,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase2_inventory_fixed",
            "advance_to_8_7_56_1424": inventory_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME] if phase3_required else [],
        },
        {"inventory_hits": inventory_hits},
    )
    write_artifact("source_inventory", inventory_payload)

    audit_payload = payload(
        "8.7.56.1424",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure Phase 2 lambda_rot audit",
        common_inputs,
        [
            row(
                "ell0_direct_spin_factor_available",
                "pass" if ell0_direct_spin_factor_available else "reject",
                "ell=0 direct spin factor available",
                1 if ell0_direct_spin_factor_available else 0,
                "The current frozen multiplicative rule would need to modify the electron-like ell=0 state directly to rescue Phase 2 without a new theorem.",
            ),
            row(
                "exact_j_eff_available",
                "pass" if exact_j_eff_available else "reject",
                "exact J_eff available",
                1 if exact_j_eff_available else 0,
                "A true Phase 2 correction needs an exact J_eff^mu reading rather than a proxy-only angular guess.",
            ),
            row(
                "phase2_naive_multiplicative_close_available",
                "pass" if phase2_naive_multiplicative_close_available else "reject",
                "naive multiplicative close available",
                1 if phase2_naive_multiplicative_close_available else 0,
                "Any literal multiplicative lambda_rot correction must close the alpha residual without an unproven large coefficient.",
            ),
            row(
                "phase2_naive_additive_close_available",
                "pass" if phase2_naive_additive_close_available else "reject",
                "naive additive close available",
                1 if phase2_naive_additive_close_available else 0,
                "Even an optimistic O(|lambda_rot|) additive correction is tracked as a diagnostic envelope before exact source promotion.",
            ),
            row(
                "phase2_secondary_lane_no_go",
                "pass" if phase2_secondary_lane_no_go else "reject",
                "Phase 2 secondary-lane no-go",
                1 if phase2_secondary_lane_no_go else 0,
                "Phase 2 becomes an honest secondary-lane no-go when lambda_rot reuse exists structurally but neither a direct ell=0 factor nor an exact J_eff theorem closes the residual.",
            ),
            row(
                "phase3_required",
                "pass" if phase3_required else "reject",
                "Phase 3 required",
                1 if phase3_required else 0,
                "Phase 3 is required when Phase 2 cannot close the residual under the current pack.",
            ),
            row(
                "physical_reject_required",
                "reject",
                "physical reject required",
                0.0,
                "Phase 2 no-go remains route-local and does not force a physical reject.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": PRIOR_CLASS,
            "lambda_rot_surface_available": lambda_rot_surface_available,
            "spin_orbit_reuse_available": spin_orbit_reuse_available,
            "electron_identification_preserved": electron_identification_preserved,
            "ell0_direct_spin_factor_available": ell0_direct_spin_factor_available,
            "exact_j_eff_available": exact_j_eff_available,
            "phase2_naive_multiplicative_close_available": phase2_naive_multiplicative_close_available,
            "phase2_naive_additive_close_available": phase2_naive_additive_close_available,
            "phase2_secondary_lane_no_go": phase2_secondary_lane_no_go,
            "phase3_required": phase3_required,
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "physical_reject_required": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME if phase3_required else None,
            "recommended_next_route_or_none": NEXT_ROUTE if phase3_required else None,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase2_audit_completed",
            "advance_to_8_7_56_1425": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME] if phase3_required else [],
        },
        {
            "phase1_summary": phase1_eval,
            "lambda_rot_summary": lambda_audit["calibration"],
            "spin_reuse_summary": spin_reuse,
            "coupled_solver_inventory_summary": coupled_solver_inventory,
        },
    )
    write_artifact("audit", audit_payload)

    gate_payload = payload(
        "8.7.56.1425",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure Phase 2 declaration gate",
        common_inputs,
        audit_payload["rows"],
        {
            **audit_payload["summary"],
            "selected_next_generation_route": NEXT_ROUTE_NAME if phase3_required else None,
            "recommended_next_route_or_none": NEXT_ROUTE if phase3_required else None,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase2_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME] if phase3_required else [],
        },
        {
            "required_abs_delta_f": required_abs_delta_f,
            "required_signed_delta_to_positive": required_signed_delta_to_positive,
            "required_abs_multiplier": required_abs_multiplier,
            "required_alpha_multiplier": required_alpha_multiplier,
        },
    )
    write_artifact("declaration_gate", gate_payload)

    evaluation_payload = payload(
        "8.7.56.1426",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure Phase 2 numeric evaluation",
        common_inputs,
        [
            row(
                "phase1_best_abs_form_factor",
                "watch",
                "Phase 1 best absolute form factor at q_theory",
                phase1_f_abs,
                "Phase 2 inherits the best exact coupled Phase 1 form factor magnitude as its starting point.",
            ),
            row(
                "required_abs_multiplier",
                "watch",
                "required multiplicative factor on |F|",
                required_abs_multiplier,
                "This is the absolute form-factor multiplier needed to reach the target from the best exact Phase 1 candidate.",
            ),
            row(
                "required_alpha_multiplier",
                "watch",
                "required multiplicative factor on alpha",
                required_alpha_multiplier,
                "This is the alpha multiplier needed to reach the target from the best exact Phase 1 candidate.",
            ),
            row(
                "best_naive_mult_3sigma_relerr",
                "watch",
                "best naive multiplicative 3sigma alpha relative error",
                best_naive_mult_3sigma_relerr,
                "Even an optimistic O(|lambda_rot|) multiplicative envelope remains far from closeout.",
            ),
            row(
                "best_naive_add_3sigma_relerr",
                "watch",
                "best naive additive 3sigma alpha relative error",
                best_naive_add_3sigma_relerr,
                "Even an optimistic O(|lambda_rot|) additive envelope remains far from closeout without an exact source theorem.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": PRIOR_CLASS,
            "phase1_best_abs_form_factor": phase1_f_abs,
            "phase1_best_signed_form_factor": phase1_f_signed,
            "phase1_best_alpha": phase1_alpha,
            "target_form_factor": TARGET_FORM_FACTOR,
            "alpha_target": ALPHA_TARGET,
            "lambda_rot_mean": lambda_rot,
            "lambda_rot_sigma": lambda_sigma,
            "lambda_rot_abs_1sigma_envelope": lambda_abs_1sigma,
            "lambda_rot_abs_3sigma_envelope": lambda_abs_3sigma,
            "ell0_direct_spin_factor": ell0_direct_spin_factor,
            "required_abs_delta_f": required_abs_delta_f,
            "required_signed_delta_to_positive": required_signed_delta_to_positive,
            "required_abs_multiplier": required_abs_multiplier,
            "required_alpha_multiplier": required_alpha_multiplier,
            "required_linear_multiplier_coeff_mean": required_linear_multiplier_coeff_mean,
            "required_linear_multiplier_coeff_1sigma": required_linear_multiplier_coeff_1sigma,
            "required_linear_multiplier_coeff_3sigma": required_linear_multiplier_coeff_3sigma,
            "required_additive_coeff_1sigma": required_additive_coeff_1sigma,
            "required_additive_coeff_3sigma": required_additive_coeff_3sigma,
            "best_naive_mult_3sigma_f": best_naive_mult_3sigma_f,
            "best_naive_mult_3sigma_alpha": best_naive_mult_3sigma_alpha,
            "best_naive_mult_3sigma_relerr": best_naive_mult_3sigma_relerr,
            "best_naive_add_3sigma_f": best_naive_add_3sigma_f,
            "best_naive_add_3sigma_alpha": best_naive_add_3sigma_alpha,
            "best_naive_add_3sigma_relerr": best_naive_add_3sigma_relerr,
            "phase2_secondary_lane_no_go": phase2_secondary_lane_no_go,
            "phase3_required": phase3_required,
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "physical_reject_required": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME if phase3_required else None,
            "recommended_next_route_or_none": NEXT_ROUTE if phase3_required else None,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase2_lambda_rot_correction_completed",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME] if phase3_required else [],
        },
        {
            "phase1_best_alpha_candidate": phase1_best,
            "lambda_rot_summary": lambda_audit["calibration"],
            "spin_reuse_summary": spin_reuse,
        },
    )
    write_artifact("numeric_evaluation", evaluation_payload)

    print("[done] 8.7.56.1423-.1426 artifacts generated")


if __name__ == "__main__":
    main()
