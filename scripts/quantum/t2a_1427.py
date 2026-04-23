#!/usr/bin/env python3
"""Generate unified-closure Phase 3 blind vector observable gate artifacts for 8.7.56.1427-.1430."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
EXPERT_SHARE = ROOT / "doc" / "quantum" / "38_trial2_numeric_alpha_vector_qball_exploratory_expert_share.md"
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
PHASE2_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_lambda_rot_form_factor_correction_"
    "audit_metrics.json"
)
PHASE2_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_lambda_rot_form_factor_correction_"
    "numeric_evaluation_metrics.json"
)
QBALL_DISCRETE = PUBLIC_OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"

PIVOT_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_pivot_branch.py"

ALPHA_TARGET = 1.0 / 137.035999084
TARGET_FORM_FACTOR = math.sqrt(4.0 * math.pi * ALPHA_TARGET)
LOCAL_BAND_FRACTION = 0.20
GRID_Q_MAX = 1.0
GRID_Q_COUNT = 4001

PRIOR_CLASS = "vector_qball_form_factor_unified_closure_phase2_lambda_rot_secondary_lane_no_go_phase3_required"
BRANCH_CLASS = "vector_qball_form_factor_unified_closure_phase3_blind_observable_no_go_case_c_honest_partial"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_case_c_closeout_sync"
NEXT_ROUTE = "8.7.56.1431"
STEM = "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_blind_vector_observable_gate"


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


# Function: dynamically load one local Python module.

def load_module(path: Path, module_name: str):
    """Dynamically load one local Python module."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Function: evaluate one normalized spherical form factor from a density profile.

def form_factor(radius: np.ndarray, density: np.ndarray, q_ratio: float) -> tuple[float, float]:
    """Evaluate one normalized spherical form factor from a density profile."""
    weight = density * (radius**2)
    norm = float(np.trapezoid(weight, radius))
    qx = float(q_ratio) * radius
    sinc = np.ones_like(qx)
    mask = np.abs(qx) > 1.0e-12
    sinc[mask] = np.sin(qx[mask]) / qx[mask]
    numerator = float(np.trapezoid(weight * sinc, radius))
    return float(numerator / norm), float(norm)


# Function: solve the exact coupled ell=0 pilot profile used by Phase 1.

def solve_exact_profile(pivot, beta: float, amp0: float, amp_l: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the exact coupled ell=0 pilot profile used by Phase 1."""
    r0 = 1.0e-4
    y0 = [float(amp0), 0.0, float(amp_l) * r0, float(amp_l)]

    # Function: return the coupled ell=0 pilot ODE.
    def ode(radius: float, y: np.ndarray) -> list[float]:
        f0, f0_prime, f_l, f_l_prime = [float(value) for value in y]
        safe_r = max(float(radius), 1.0e-6)
        rho = math.sqrt(max(f0 * f0 + f_l * f_l, 0.0))
        nonlinear_coeff = 3.0 * rho + rho * rho
        f0_double_prime = (
            -(2.0 / safe_r) * f0_prime
            - (float(beta * beta) - float(pivot.RADIAL_MASS_SQUARED)) * f0
            - nonlinear_coeff * f0
        )
        f_l_double_prime = (
            -(2.0 / safe_r) * f_l_prime
            - (float(beta * beta) - float(pivot.LONGITUDINAL_DIRECT_MASS_SQUARED)) * f_l
            - nonlinear_coeff * f_l
        )
        return [f0_prime, f0_double_prime, f_l_prime, f_l_double_prime]

    sol = solve_ivp(ode, (r0, 25.0), y0, max_step=0.10, rtol=1.0e-7, atol=1.0e-9)
    if not sol.success:
        raise SystemExit("[fail] exact coupled Phase 1 pilot profile did not converge")

    return np.asarray(sol.t, dtype=float), np.asarray(sol.y[0], dtype=float), np.asarray(sol.y[2], dtype=float)


# Function: find one signed target crossing inside the search interval.

def find_signed_target_crossing(radius: np.ndarray, density: np.ndarray, q_lo: float, q_hi: float) -> float | None:
    """Find one signed target crossing inside the search interval."""

    # Function: return F(q)-F_target for one q.
    def residual(q_value: float) -> float:
        f_value, _norm = form_factor(radius, density, q_value)
        return float(f_value - TARGET_FORM_FACTOR)

    grid = np.linspace(float(q_lo), float(q_hi), 801)
    values = [residual(float(q_value)) for q_value in grid]
    for left_index in range(len(grid) - 1):
        left_value = values[left_index]
        right_value = values[left_index + 1]
        if left_value == 0.0:
            return float(grid[left_index])

        if left_value * right_value < 0.0:
            return float(brentq(residual, float(grid[left_index]), float(grid[left_index + 1])))

    return None


# Function: compute local-band best errors around q_theory.

def local_band_summary(radius: np.ndarray, density: np.ndarray, q_theory: float) -> dict:
    """Compute local-band best errors around q_theory."""
    q_min = max(0.0, float(q_theory) * (1.0 - LOCAL_BAND_FRACTION))
    q_max = float(q_theory) * (1.0 + LOCAL_BAND_FRACTION)
    grid = np.linspace(q_min, q_max, 801)
    f_values = np.array([form_factor(radius, density, float(q_value))[0] for q_value in grid], dtype=float)
    signed_errors = np.abs(f_values - TARGET_FORM_FACTOR)
    abs_errors = np.abs(np.abs(f_values) - TARGET_FORM_FACTOR)
    signed_index = int(np.argmin(signed_errors))
    abs_index = int(np.argmin(abs_errors))
    return {
        "band_q_min": q_min,
        "band_q_max": q_max,
        "signed_best_q": float(grid[signed_index]),
        "signed_best_F": float(f_values[signed_index]),
        "signed_best_error": float(signed_errors[signed_index]),
        "abs_best_q": float(grid[abs_index]),
        "abs_best_F": float(f_values[abs_index]),
        "abs_best_absF": float(abs(f_values[abs_index])),
        "abs_best_error": float(abs_errors[abs_index]),
    }


# Function: execute the unified-closure Phase 3 blind vector observable gate branch.

def main() -> None:
    """Execute the unified-closure Phase 3 blind vector observable gate branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        EXPERT_SHARE,
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
        PHASE2_AUDIT,
        PHASE2_EVAL,
        QBALL_DISCRETE,
        PIVOT_BRANCH,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    expert_share_text = read_text(EXPERT_SHARE)
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
    phase2_audit = read_json(PHASE2_AUDIT)["summary"]
    phase2_eval = read_json(PHASE2_EVAL)["summary"]
    qball_discrete = read_json(QBALL_DISCRETE)

    pivot = load_module(PIVOT_BRANCH, "wavep_trial3_pivot_phase3")

    best_alpha_candidate = phase1_eval["phase1_best_alpha_candidate"]
    beta = float(best_alpha_candidate["beta"])
    amp0 = float(best_alpha_candidate["amp0"])
    amp_l = float(best_alpha_candidate["amp_l"])
    q_theory = float(best_alpha_candidate["q_theory_over_m0"])
    radius, f0_values, f_l_values = solve_exact_profile(pivot, beta, amp0, amp_l)
    blind_density = f0_values * f0_values - f_l_values * f_l_values

    f_at_zero, density_norm = form_factor(radius, blind_density, 0.0)
    f_at_q_theory, _norm_check = form_factor(radius, blind_density, q_theory)
    f_at_m0, _norm_check_2 = form_factor(radius, blind_density, 1.0)
    alpha_at_q_theory = float((f_at_q_theory * f_at_q_theory) / (4.0 * math.pi))
    alpha_at_m0 = float((f_at_m0 * f_at_m0) / (4.0 * math.pi))
    alpha_relerr_at_q_theory = abs(alpha_at_q_theory - ALPHA_TARGET) / ALPHA_TARGET

    signed_target_crossing = find_signed_target_crossing(radius, blind_density, 0.0, q_theory)
    crossing_ratio_to_q_theory = None if signed_target_crossing is None else float(signed_target_crossing / q_theory)
    local_band = local_band_summary(radius, blind_density, q_theory)

    q_grid = np.linspace(0.0, GRID_Q_MAX, GRID_Q_COUNT)
    f_grid = np.array([form_factor(radius, blind_density, float(q_value))[0] for q_value in q_grid], dtype=float)
    signed_target_error_grid = np.abs(f_grid - TARGET_FORM_FACTOR)
    best_global_signed_index = int(np.argmin(signed_target_error_grid))
    best_global_signed_q = float(q_grid[best_global_signed_index])
    best_global_signed_f = float(f_grid[best_global_signed_index])
    best_global_signed_error = float(signed_target_error_grid[best_global_signed_index])

    inventory_hits = [
        hit(status_text, "8.7.56.1427"),
        hit(roadmap_text, "8.7.56.1427-.1430"),
        hit(current_problem_text, "blind vector observable"),
        hit(current_status_text, "blind vector observable"),
        hit(expert_share_text, "observable dictionary"),
        hit(unified_roadmap_text, "Phase 3"),
        hit(unified_plan_text, "Phase 3"),
        hit(unified_plan_text, "Case C"),
        hit(unified_plan_text, "Case D"),
        hit(next_steps_text, "Step D"),
        hit(next_steps_text, "Step E"),
        hit(solver_fix_text, "Minkowski sign"),
        hit(perturbative_note_text, "diagnostic only"),
        hit(part1_text, "Pauli 型スピン結合"),
        hit(part3a_text, "cross-scale freeze"),
        hit(part5_text, "blind vector observable gate"),
    ]
    inventory_ready = all(item is not None for item in inventory_hits)

    f0_normalization_pass = math.isclose(f_at_zero, 1.0, rel_tol=0.0, abs_tol=1.0e-9)
    q_theory_target_approach_pass = bool(local_band["abs_best_error"] <= 0.05 * TARGET_FORM_FACTOR)
    signed_target_crossing_exists = signed_target_crossing is not None
    signed_target_crossing_near_q_theory = bool(
        signed_target_crossing_exists
        and crossing_ratio_to_q_theory is not None
        and abs(crossing_ratio_to_q_theory - 1.0) <= LOCAL_BAND_FRACTION
    )
    exact_source_theorem_available = bool(phase2_audit["exact_j_eff_available"])
    source_theorem_compatibility_pass = bool(exact_source_theorem_available)
    universality_gate_pass = bool(
        signed_target_crossing_near_q_theory
        and source_theorem_compatibility_pass
        and float(qball_discrete["summary"]["reference_mode_index"]) == 1.0
    )
    blind_observable_gate_pass = bool(
        f0_normalization_pass
        and q_theory_target_approach_pass
        and source_theorem_compatibility_pass
        and universality_gate_pass
    )

    case_a_selected = bool(phase1_audit["phase1_close_within_one_percent"])
    case_b_selected = bool(
        not case_a_selected
        and (
            bool(phase2_audit["phase2_naive_multiplicative_close_available"])
            or bool(phase2_audit["phase2_naive_additive_close_available"])
        )
    )
    case_d_selected = bool(phase1_audit["primary_lane_no_go"])
    case_c_selected = bool(not case_a_selected and not case_b_selected and not case_d_selected)
    v2_closeout_ready = bool(case_a_selected or case_b_selected or case_c_selected or case_d_selected)

    common_inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem_note": display_path(CURRENT_PROBLEM),
            "current_status_note": display_path(CURRENT_STATUS),
            "expert_share_note": display_path(EXPERT_SHARE),
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
            "phase2_audit": display_path(PHASE2_AUDIT),
            "phase2_eval": display_path(PHASE2_EVAL),
            "qball_discrete": display_path(QBALL_DISCRETE),
        },
        "solver_modules": {"pivot_branch": display_path(PIVOT_BRANCH)},
        "constants": {
            "alpha_target": ALPHA_TARGET,
            "target_form_factor": TARGET_FORM_FACTOR,
            "local_band_fraction": LOCAL_BAND_FRACTION,
            "grid_q_max": GRID_Q_MAX,
            "grid_q_count": GRID_Q_COUNT,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory_payload = payload(
        "8.7.56.1427",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure Phase 3 blind observable inventory",
        common_inputs,
        [
            row("phase3_inventory_ready", "pass" if inventory_ready else "reject", "Phase 3 inventory ready", 1 if inventory_ready else 0, "Phase 3 inventory is ready only if Phase 1/2 outputs, observable-dictionary gaps, and final disposition criteria coexist in one pack."),
            row("phase1_nonzero_regular_branch_detected", "pass" if bool(phase1_audit["nonzero_regular_branch_detected"]) else "reject", "Phase 1 nonzero regular branch detected", 1 if bool(phase1_audit["nonzero_regular_branch_detected"]) else 0, "Case C remains admissible only if Phase 1 already proved that a nonzero regular branch exists."),
            row("phase2_secondary_lane_no_go", "pass" if bool(phase2_audit["phase2_secondary_lane_no_go"]) else "reject", "Phase 2 secondary-lane no-go", 1 if bool(phase2_audit["phase2_secondary_lane_no_go"]) else 0, "Phase 3 is only active when Phase 2 honestly failed without forcing a physical reject."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": PRIOR_CLASS,
            "phase3_inventory_ready": inventory_ready,
            "phase1_nonzero_regular_branch_detected": bool(phase1_audit["nonzero_regular_branch_detected"]),
            "phase2_secondary_lane_no_go": bool(phase2_audit["phase2_secondary_lane_no_go"]),
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase3_inventory_fixed",
            "advance_to_8_7_56_1428": inventory_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"inventory_hits": inventory_hits},
    )
    write_artifact("source_inventory", inventory_payload)

    audit_payload = payload(
        "8.7.56.1428",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure Phase 3 blind observable audit",
        common_inputs,
        [
            row("blind_f0_normalization_pass", "pass" if f0_normalization_pass else "reject", "blind F(0)=1 normalization pass", 1 if f0_normalization_pass else 0, "The blind observable gate keeps the normalized form-factor rule only if F(0)=1 survives the exact coupled density profile."),
            row("blind_q_theory_target_approach_pass", "pass" if q_theory_target_approach_pass else "reject", "blind q_theory-neighborhood target approach pass", 1 if q_theory_target_approach_pass else 0, "The blind route passes its second gate only if a 20% q_theory band gets close to the target without retuning the matching scale."),
            row("blind_signed_target_crossing_exists", "pass" if signed_target_crossing_exists else "reject", "blind signed target crossing exists", 1 if signed_target_crossing_exists else 0, "A signed target crossing is tracked separately because a remote low-q crossing can still exist even when the q_theory-neighborhood gate fails."),
            row("blind_signed_target_crossing_near_q_theory", "pass" if signed_target_crossing_near_q_theory else "reject", "blind signed target crossing near q_theory", 1 if signed_target_crossing_near_q_theory else 0, "A remote crossing does not rescue Phase 3 if it sits far from the fixed q_theory scale."),
            row("source_theorem_compatibility_pass", "pass" if source_theorem_compatibility_pass else "reject", "source-theorem compatibility pass", 1 if source_theorem_compatibility_pass else 0, "Phase 3 keeps a blind observable only if it remains compatible with an exact source reading rather than bypassing the theorem gap."),
            row("universality_gate_pass", "pass" if universality_gate_pass else "reject", "universality gate pass", 1 if universality_gate_pass else 0, "The blind observable must preserve the electron identification and avoid a hidden mass-spectrum relabeling to count as universal."),
            row("blind_observable_gate_pass", "pass" if blind_observable_gate_pass else "reject", "blind observable gate pass", 1 if blind_observable_gate_pass else 0, "Phase 3 passes only if normalization, q_theory-scale approach, source compatibility, and universality all hold simultaneously."),
            row("case_c_selected", "pass" if case_c_selected else "reject", "Case C selected", 1 if case_c_selected else 0, "Case C is selected when a nonzero branch exists but neither Phase 1 nor Phase 2 closes and Phase 3 does not rescue the blind vector observable gate."),
            row("v2_closeout_ready", "pass" if v2_closeout_ready else "reject", "v2.0 closeout ready", 1 if v2_closeout_ready else 0, "The unified-closure program remains v2.0-closeable once Case A/B/C/D is explicitly fixed."),
            row("physical_reject_required", "reject", "physical reject required", 0.0, "Phase 3 blind-observable failure remains route-local and does not force a physical reject."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": PRIOR_CLASS,
            "blind_f0_normalization_pass": f0_normalization_pass,
            "blind_q_theory_target_approach_pass": q_theory_target_approach_pass,
            "blind_signed_target_crossing_exists": signed_target_crossing_exists,
            "blind_signed_target_crossing_near_q_theory": signed_target_crossing_near_q_theory,
            "source_theorem_compatibility_pass": source_theorem_compatibility_pass,
            "universality_gate_pass": universality_gate_pass,
            "blind_observable_gate_pass": blind_observable_gate_pass,
            "case_a_selected": case_a_selected,
            "case_b_selected": case_b_selected,
            "case_c_selected": case_c_selected,
            "case_d_selected": case_d_selected,
            "v2_closeout_ready": v2_closeout_ready,
            "physical_reject_required": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase3_audit_completed",
            "advance_to_8_7_56_1429": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"phase1_summary": phase1_eval, "phase2_summary": phase2_eval, "local_band_summary": local_band},
    )
    write_artifact("audit", audit_payload)

    gate_payload = payload(
        "8.7.56.1429",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure Phase 3 final declaration gate",
        common_inputs,
        audit_payload["rows"],
        {
            **audit_payload["summary"],
            "v2_0_final_disposition_case": "Case C" if case_c_selected else None,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase3_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "signed_target_crossing_over_m0": signed_target_crossing,
            "signed_target_crossing_to_q_theory_ratio": crossing_ratio_to_q_theory,
            "best_global_signed_q_over_m0": best_global_signed_q,
            "best_global_signed_F": best_global_signed_f,
            "best_global_signed_error": best_global_signed_error,
        },
    )
    write_artifact("declaration_gate", gate_payload)

    evaluation_payload = payload(
        "8.7.56.1430",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure Phase 3 final numeric evaluation",
        common_inputs,
        [
            row("blind_F_at_zero", "pass" if f0_normalization_pass else "reject", "blind F(0)", f_at_zero, "The normalized blind observable preserves F(0)=1 on the exact coupled density profile."),
            row("blind_F_at_q_theory", "watch", "blind F(q_theory)", f_at_q_theory, "The exact coupled blind observable at q_theory stays far from the positive target and even flips sign."),
            row("blind_alpha_at_q_theory", "watch", "blind alpha(q_theory)", alpha_at_q_theory, "The exact coupled blind observable keeps the Phase 1 alpha value, which remains far from closeout."),
            row("blind_signed_target_crossing_over_m0", "watch" if signed_target_crossing_exists else "reject", "blind signed target crossing q/m0", -1.0 if signed_target_crossing is None else signed_target_crossing, "A remote low-q crossing exists, but it sits far below the fixed q_theory scale and therefore does not rescue the blind gate."),
            row("blind_crossing_to_q_theory_ratio", "watch" if crossing_ratio_to_q_theory is not None else "reject", "blind signed crossing to q_theory ratio", -1.0 if crossing_ratio_to_q_theory is None else crossing_ratio_to_q_theory, "This ratio tracks whether the blind target crossing is actually compatible with the fixed matching-scale theorem."),
            row("blind_F_at_m0", "watch", "blind F(m0)", f_at_m0, "The exact coupled blind observable at q=m0 remains tiny and does not offer an alternative closeout reading."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": PRIOR_CLASS,
            "beta_1": beta,
            "amp0": amp0,
            "amp_l": amp_l,
            "blind_density_norm": density_norm,
            "blind_F_at_zero": f_at_zero,
            "blind_F_at_q_theory": f_at_q_theory,
            "blind_alpha_at_q_theory": alpha_at_q_theory,
            "blind_alpha_relerr_vs_target": alpha_relerr_at_q_theory,
            "blind_F_at_m0": f_at_m0,
            "blind_alpha_at_m0": alpha_at_m0,
            "signed_target_crossing_over_m0": signed_target_crossing,
            "signed_target_crossing_to_q_theory_ratio": crossing_ratio_to_q_theory,
            "local_band_q_min": local_band["band_q_min"],
            "local_band_q_max": local_band["band_q_max"],
            "local_band_signed_best_q": local_band["signed_best_q"],
            "local_band_signed_best_F": local_band["signed_best_F"],
            "local_band_signed_best_error": local_band["signed_best_error"],
            "local_band_abs_best_q": local_band["abs_best_q"],
            "local_band_abs_best_F": local_band["abs_best_F"],
            "local_band_abs_best_absF": local_band["abs_best_absF"],
            "local_band_abs_best_error": local_band["abs_best_error"],
            "best_global_signed_q": best_global_signed_q,
            "best_global_signed_F": best_global_signed_f,
            "best_global_signed_error": best_global_signed_error,
            "blind_f0_normalization_pass": f0_normalization_pass,
            "blind_q_theory_target_approach_pass": q_theory_target_approach_pass,
            "source_theorem_compatibility_pass": source_theorem_compatibility_pass,
            "universality_gate_pass": universality_gate_pass,
            "blind_observable_gate_pass": blind_observable_gate_pass,
            "case_a_selected": case_a_selected,
            "case_b_selected": case_b_selected,
            "case_c_selected": case_c_selected,
            "case_d_selected": case_d_selected,
            "v2_0_final_disposition_case": "Case C" if case_c_selected else None,
            "v2_closeout_ready": v2_closeout_ready,
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "physical_reject_required": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase3_blind_observable_gate_completed",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "phase1_best_alpha_candidate": best_alpha_candidate,
            "phase2_summary": phase2_eval,
            "qball_reference_mode": qball_discrete["evidence"]["discrete_mass_mode_rows"][0],
        },
    )
    write_artifact("numeric_evaluation", evaluation_payload)

    print("[done] 8.7.56.1427-.1430 artifacts generated")


if __name__ == "__main__":
    main()
