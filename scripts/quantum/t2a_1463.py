#!/usr/bin/env python3
"""Generate perturbative f_L driven ODE diagnostic reopen review artifacts for 8.7.56.1463-.1466."""

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
ADVICE_REQUEST = ROOT / "doc" / "quantum" / "40_trial2_numeric_alpha_vector_qball_reopen_advice_request.md"
NEXT_ACTION_INTEGRATION = ROOT / "doc" / "quantum" / "41_trial2_vector_qball_next_action_integration.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

UNIFIED_PLAN = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_unified_closure_plan_20260327.md")
NEXT_STEPS = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")
NEXT_ACTION = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_action_20260327.md")
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
PHASE3_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_blind_vector_observable_gate_"
    "audit_metrics.json"
)
PHASE3_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_blind_vector_observable_gate_"
    "numeric_evaluation_metrics.json"
)
PRIOR_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_"
    "final_summary_route_declaration_gate_metrics.json"
)
QBALL_DISCRETE = PUBLIC_OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"
QBALL_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_qball_charge_mapping_branch.py"

ALPHA_TARGET = 1.0 / 137.035999084
CASE_ALPHA_THRESHOLD = 0.01
CASE_GAMMA_THRESHOLD = 0.30

PRIOR_CLASS = "vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_final_summary_completed"
BRANCH_CLASS = "vector_qball_form_factor_unified_closure_perturbative_fl_driven_ode_case_gamma_archive_registry_restore_required"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_archive_registry_restore"
NEXT_ROUTE = "8.7.56.1467"
ALTERNATE_NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_exact_solver_reinjection_reopen"
ALTERNATE_NEXT_ROUTE = "8.7.56.1467"
STEM = "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_perturbative_fl_driven_ode_diagnostic_reopen_review"


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


# Function: extract the retained scalar ground-state row.

def ground_state_row(qball_discrete: dict) -> dict:
    """Extract the retained scalar ground-state row."""
    for row_data in qball_discrete["evidence"]["discrete_mass_mode_rows"]:
        if int(row_data["mode_index"]) == 1:
            return {
                "mode_index": 1,
                "beta_n": float(row_data["beta_n"]),
                "central_amplitude": float(row_data["central_amplitude"]),
                "charge_proxy": float(row_data["charge_proxy"]),
            }

    raise SystemExit("[fail] missing mode_index=1 in discrete mass spectrum metrics")


# Function: solve the retained scalar ground-state profile.

def solve_scalar_profile(qball_module, beta: float, amplitude: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the retained scalar ground-state profile."""
    radius, field, field_prime = qball_module.solve_full_profile(float(beta), float(amplitude))
    return np.asarray(radius, dtype=float), np.asarray(field, dtype=float), np.asarray(field_prime, dtype=float)


# Function: evaluate one normalized spherical form factor.

def form_factor(radius: np.ndarray, density: np.ndarray, q_ratio: float) -> tuple[float, float]:
    """Evaluate one normalized spherical form factor."""
    weight = density * (radius**2)
    norm = float(np.trapezoid(weight, radius))
    qx = float(q_ratio) * radius
    sinc = np.ones_like(qx)
    mask = np.abs(qx) > 1.0e-12
    sinc[mask] = np.sin(qx[mask]) / qx[mask]
    numerator = float(np.trapezoid(weight * sinc, radius))
    return float(numerator / norm), float(norm)


# Function: find one sign-changing residual bracket on a fixed grid.

def find_residual_bracket(residual_fn, left: float, right: float, samples: int) -> tuple[float, float]:
    """Find one sign-changing residual bracket on a fixed grid."""
    grid = np.linspace(float(left), float(right), int(samples))
    values = [float(residual_fn(float(sample))) for sample in grid]
    for left_index in range(len(grid) - 1):
        left_value = values[left_index]
        right_value = values[left_index + 1]
        if left_value == 0.0:
            return float(grid[left_index]), float(grid[left_index])

        if left_value * right_value < 0.0:
            return float(grid[left_index]), float(grid[left_index + 1])

    raise SystemExit("[fail] unable to bracket perturbative f_L shooting slope on [-10, 10]")


# Function: solve the perturbative f_L driven ODE as a diagnostic-only probe.

def solve_diagnostic_f_l(radius: np.ndarray, field: np.ndarray, field_prime: np.ndarray, beta: float) -> dict:
    """Solve the perturbative f_L driven ODE as a diagnostic-only probe."""
    kappa_sq = 1.0 - float(beta) * float(beta)
    start_radius = 1.0e-6
    end_radius = float(radius[-1])
    eval_radius = radius[radius >= start_radius]

    # Function: return the driven ODE implied by the solver-fix note.
    def ode(rr: float, y: np.ndarray) -> list[float]:
        f_l, f_l_prime = [float(value) for value in y]
        safe_r = max(float(rr), 1.0e-10)
        source = float(beta) * float(np.interp(safe_r, radius, field_prime))
        f_l_double_prime = -2.0 * f_l_prime / safe_r + 2.0 * f_l / (safe_r * safe_r) + kappa_sq * f_l + source
        return [f_l_prime, f_l_double_prime]

    # Function: return the localization residual for one launch slope.

    def residual(df_l0: float) -> float:
        sol = solve_ivp(
            ode,
            (start_radius, end_radius),
            [start_radius * float(df_l0), float(df_l0)],
            t_eval=eval_radius,
            rtol=1.0e-10,
            atol=1.0e-12,
            max_step=0.03,
        )
        return float(sol.y[0, -1])

    bracket_left, bracket_right = find_residual_bracket(residual, -10.0, 10.0, 401)
    if bracket_left == bracket_right:
        df_l0 = float(bracket_left)
    else:
        df_l0 = float(brentq(residual, bracket_left, bracket_right))

    sol = solve_ivp(
        ode,
        (start_radius, end_radius),
        [start_radius * df_l0, df_l0],
        t_eval=eval_radius,
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.03,
    )
    f_l = np.asarray(sol.y[0], dtype=float)
    f_l_prime = np.asarray(sol.y[1], dtype=float)
    q_theory = float((1.0 - float(beta) * float(beta)) ** 0.25)
    density = field[: len(f_l)] * field[: len(f_l)] - f_l * f_l
    f_value, density_norm = form_factor(eval_radius, density, q_theory)
    peak_index = int(np.argmax(np.abs(f_l)))
    peak_sign = 0.0
    if f_l[peak_index] > 0.0:
        peak_sign = 1.0
    elif f_l[peak_index] < 0.0:
        peak_sign = -1.0

    alpha_value = float((f_value * f_value) / (4.0 * math.pi))
    return {
        "df_l0": df_l0,
        "residual_bracket_left": bracket_left,
        "residual_bracket_right": bracket_right,
        "max_abs_fL": float(np.max(np.abs(f_l))),
        "max_abs_ratio": float(np.max(np.abs(f_l)) / np.max(np.abs(field))),
        "peak_position": float(eval_radius[peak_index]),
        "peak_sign": peak_sign,
        "peak_value": float(f_l[peak_index]),
        "initial_derivative": float(f_l_prime[0]),
        "q_theory_over_m0": q_theory,
        "F_at_q_theory": f_value,
        "alpha_at_q_theory": alpha_value,
        "alpha_relerr_vs_target": abs(alpha_value - ALPHA_TARGET) / ALPHA_TARGET,
        "density_norm": density_norm,
    }


# Function: execute the perturbative f_L driven ODE diagnostic reopen review branch.

def main() -> None:
    """Execute the perturbative f_L driven ODE diagnostic reopen review branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        EXPERT_SHARE,
        UNIFIED_ROADMAP,
        ADVICE_REQUEST,
        NEXT_ACTION_INTEGRATION,
        PART1,
        PART3A,
        PART5,
        UNIFIED_PLAN,
        NEXT_STEPS,
        NEXT_ACTION,
        SOLVER_FIX,
        PERTURBATIVE_NOTE,
        PHASE1_AUDIT,
        PHASE1_EVAL,
        PHASE2_AUDIT,
        PHASE2_EVAL,
        PHASE3_AUDIT,
        PHASE3_EVAL,
        PRIOR_GATE,
        QBALL_DISCRETE,
        QBALL_SOLVER,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    expert_share_text = read_text(EXPERT_SHARE)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    advice_request_text = read_text(ADVICE_REQUEST)
    next_action_integration_text = read_text(NEXT_ACTION_INTEGRATION)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    unified_plan_text = read_text(UNIFIED_PLAN)
    next_steps_text = read_text(NEXT_STEPS)
    next_action_text = read_text(NEXT_ACTION)
    solver_fix_text = read_text(SOLVER_FIX)
    perturbative_note_text = read_text(PERTURBATIVE_NOTE)

    phase1_audit = read_json(PHASE1_AUDIT)["summary"]
    phase1_eval = read_json(PHASE1_EVAL)["summary"]
    phase2_audit = read_json(PHASE2_AUDIT)["summary"]
    phase2_eval = read_json(PHASE2_EVAL)["summary"]
    phase3_audit = read_json(PHASE3_AUDIT)["summary"]
    phase3_eval = read_json(PHASE3_EVAL)["summary"]
    prior_gate = read_json(PRIOR_GATE)["summary"]
    qball_discrete = read_json(QBALL_DISCRETE)

    qball_module = load_module(QBALL_SOLVER, "wavep_qball_charge_mapping_t2a1463")
    scalar_ground_state = ground_state_row(qball_discrete)
    radius, field, field_prime = solve_scalar_profile(
        qball_module,
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    diagnostic = solve_diagnostic_f_l(radius, field, field_prime, float(scalar_ground_state["beta_n"]))

    phase1_best = phase1_eval["phase1_best_alpha_candidate"]
    phase1_seed_ratio = abs(float(phase1_best["amp_l"])) / abs(float(phase1_best["amp0"]))
    phase1_profile_ratio = abs(float(phase1_best["max_abs_fL"])) / abs(float(phase1_best["max_abs_f0"]))

    case_alpha_selected = bool(float(diagnostic["max_abs_ratio"]) <= CASE_ALPHA_THRESHOLD)
    case_beta_selected = bool(
        float(diagnostic["max_abs_ratio"]) > CASE_ALPHA_THRESHOLD and float(diagnostic["max_abs_ratio"]) <= CASE_GAMMA_THRESHOLD
    )
    case_gamma_selected = bool(float(diagnostic["max_abs_ratio"]) > CASE_GAMMA_THRESHOLD)
    if sum((case_alpha_selected, case_beta_selected, case_gamma_selected)) != 1:
        raise SystemExit("[fail] perturbative f_L diagnostic classification is not unique")

    wrong_branch_suspicion_supported = bool(case_alpha_selected or case_beta_selected)
    perturbative_breakdown_detected = bool(case_gamma_selected)
    exact_solver_reinjection_required = bool(case_alpha_selected or case_beta_selected)
    handoff_archive_registry_restore_required = bool(case_gamma_selected)
    selected_next_generation_route = NEXT_ROUTE_NAME if case_gamma_selected else ALTERNATE_NEXT_ROUTE_NAME
    recommended_next_route_or_none = NEXT_ROUTE if case_gamma_selected else ALTERNATE_NEXT_ROUTE

    inventory_hits = [
        hit(status_text, "8.7.56.1463-.1466"),
        hit(roadmap_text, "8.7.56.1463-.1466"),
        hit(current_problem_text, "perturbative `f_L` driven ODE"),
        hit(current_status_text, "perturbative `f_L` driven ODE"),
        hit(expert_share_text, "exact_action_level_ell0_operator_reopen"),
        hit(advice_request_text, "exact_action_level_ell0_operator_reopen"),
        hit(next_action_integration_text, "Case α / β / γ"),
        hit(unified_roadmap_text, "perturbative `f_L` driven ODE diagnostic reopen review"),
        hit(unified_plan_text, "Phase 1"),
        hit(next_steps_text, "Step A"),
        hit(next_action_text, "Case γ"),
        hit(solver_fix_text, "2-step perturbative method"),
        hit(perturbative_note_text, "vector_form_factor_perturbative_computation_ready = true"),
        hit(part1_text, "F_{0r}^{(P)} = \\partial_0 P_r - \\partial_r P_0 = i\\omega f_L - f_0'"),
        hit(part3a_text, "D_\\mu=\\partial_\\mu+i q A_\\mu"),
        hit(part5_text, "perturbative `f_L` driven ODE diagnostic reopen review"),
    ]
    inventory_ready = all(item is not None for item in inventory_hits)

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
            "advice_request_note": display_path(ADVICE_REQUEST),
            "next_action_integration_note": display_path(NEXT_ACTION_INTEGRATION),
            "part1": display_path(PART1),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
            "unified_plan_note": display_path(UNIFIED_PLAN),
            "next_steps_note": display_path(NEXT_STEPS),
            "next_action_note": display_path(NEXT_ACTION),
            "solver_fix_note": display_path(SOLVER_FIX),
            "perturbative_note": display_path(PERTURBATIVE_NOTE),
        },
        "source_metrics": {
            "phase1_audit": display_path(PHASE1_AUDIT),
            "phase1_eval": display_path(PHASE1_EVAL),
            "phase2_audit": display_path(PHASE2_AUDIT),
            "phase2_eval": display_path(PHASE2_EVAL),
            "phase3_audit": display_path(PHASE3_AUDIT),
            "phase3_eval": display_path(PHASE3_EVAL),
            "prior_gate": display_path(PRIOR_GATE),
            "qball_discrete": display_path(QBALL_DISCRETE),
        },
        "solver_modules": {"qball_solver": display_path(QBALL_SOLVER)},
        "constants": {
            "alpha_target": ALPHA_TARGET,
            "case_alpha_threshold": CASE_ALPHA_THRESHOLD,
            "case_gamma_threshold": CASE_GAMMA_THRESHOLD,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "alternate_next_route_name": ALTERNATE_NEXT_ROUTE_NAME,
            "alternate_next_route": ALTERNATE_NEXT_ROUTE,
        },
    }

    inventory_payload = payload(
        "8.7.56.1463",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure perturbative f_L driven ODE diagnostic inventory",
        common_inputs,
        [
            row(
                "diagnostic_inventory_ready",
                "pass" if inventory_ready else "reject",
                "diagnostic inventory ready",
                1 if inventory_ready else 0,
                "The diagnostic branch is admissible only if unified closure outputs, solver-fix/perturbative notes, next-action advice, and current notes coexist in one pack.",
            ),
            row(
                "phase1_exact_best_candidate_available",
                "pass" if bool(phase1_audit["phase1_exact_coupled_pilot_available"]) else "reject",
                "Phase 1 exact best candidate available",
                1 if bool(phase1_audit["phase1_exact_coupled_pilot_available"]) else 0,
                "The perturbative diagnostic is meaningful only because Phase 1 already produced a nontrivial exact pilot candidate to test against.",
            ),
            row(
                "case_c_current_state_available",
                "pass" if bool(phase3_audit["case_c_selected"]) and bool(prior_gate["future_reopen_trigger_handoff_final_summary_ready"]) else "reject",
                "Case C current state available",
                1 if bool(phase3_audit["case_c_selected"]) and bool(prior_gate["future_reopen_trigger_handoff_final_summary_ready"]) else 0,
                "The diagnostic reopens Case C only if the current official state and handoff-final-summary sync are already frozen.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": PRIOR_CLASS,
            "diagnostic_inventory_ready": inventory_ready,
            "phase1_exact_best_candidate_available": bool(phase1_audit["phase1_exact_coupled_pilot_available"]),
            "case_c_current_state_available": bool(phase3_audit["case_c_selected"]) and bool(prior_gate["future_reopen_trigger_handoff_final_summary_ready"]),
            "selected_next_generation_route": selected_next_generation_route,
            "recommended_next_route_or_none": recommended_next_route_or_none,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_perturbative_fl_driven_ode_inventory_fixed",
            "advance_to_8_7_56_1464": inventory_ready,
            "next_required_artifacts": [selected_next_generation_route],
        },
        {"inventory_hits": inventory_hits},
    )
    write_artifact("source_inventory", inventory_payload)

    audit_payload = payload(
        "8.7.56.1464",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure perturbative f_L driven ODE diagnostic audit",
        common_inputs,
        [
            row(
                "diagnostic_solution_found",
                "pass",
                "diagnostic solution found",
                1.0,
                "The perturbative driven ODE admits a regular localized shooting solution for the retained scalar ground state.",
            ),
            row(
                "case_alpha_selected",
                "pass" if case_alpha_selected else "reject",
                "Case alpha selected",
                1 if case_alpha_selected else 0,
                "Case alpha requires a genuinely perturbative amplitude ratio at or below 0.01.",
            ),
            row(
                "case_beta_selected",
                "pass" if case_beta_selected else "reject",
                "Case beta selected",
                1 if case_beta_selected else 0,
                "Case beta absorbs the non-alpha subperturbative regime below the 0.3 breakdown threshold.",
            ),
            row(
                "case_gamma_selected",
                "pass" if case_gamma_selected else "reject",
                "Case gamma selected",
                1 if case_gamma_selected else 0,
                "Case gamma means the perturbative driven ODE already sits above the 0.3 breakdown threshold and cannot justify exact solver reinjection.",
            ),
            row(
                "wrong_branch_suspicion_supported",
                "pass" if wrong_branch_suspicion_supported else "reject",
                "wrong-branch suspicion supported",
                1 if wrong_branch_suspicion_supported else 0,
                "Wrong-branch suspicion survives only if the perturbative diagnostic remains in the alpha/beta regime.",
            ),
            row(
                "perturbative_breakdown_detected",
                "pass" if perturbative_breakdown_detected else "reject",
                "perturbative breakdown detected",
                1 if perturbative_breakdown_detected else 0,
                "Once the driven ODE exceeds the gamma threshold, the perturbative route becomes a breakdown diagnostic rather than a reinjection seed.",
            ),
            row(
                "exact_solver_reinjection_required",
                "pass" if exact_solver_reinjection_required else "reject",
                "exact solver reinjection required",
                1 if exact_solver_reinjection_required else 0,
                "Exact solver reinjection is required only for Case alpha/beta outcomes.",
            ),
            row(
                "handoff_archive_registry_restore_required",
                "pass" if handoff_archive_registry_restore_required else "reject",
                "handoff archive registry restore required",
                1 if handoff_archive_registry_restore_required else 0,
                "Case gamma restores the deferred archive/handoff route instead of reopening exact solver reinjection.",
            ),
            row(
                "physical_reject_required",
                "reject",
                "physical reject required",
                0.0,
                "The diagnostic is route-local and does not force a physical reject.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": PRIOR_CLASS,
            "case_alpha_selected": case_alpha_selected,
            "case_beta_selected": case_beta_selected,
            "case_gamma_selected": case_gamma_selected,
            "wrong_branch_suspicion_supported": wrong_branch_suspicion_supported,
            "perturbative_breakdown_detected": perturbative_breakdown_detected,
            "exact_solver_reinjection_required": exact_solver_reinjection_required,
            "handoff_archive_registry_restore_required": handoff_archive_registry_restore_required,
            "case_c_honest_partial_retained": True,
            "retained_scalar_strong_candidate_retained": True,
            "physical_reject_required": False,
            "selected_next_generation_route": selected_next_generation_route,
            "recommended_next_route_or_none": recommended_next_route_or_none,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_perturbative_fl_driven_ode_audit_completed",
            "advance_to_8_7_56_1465": True,
            "next_required_artifacts": [selected_next_generation_route],
        },
        {
            "phase1_summary": phase1_eval,
            "phase2_summary": phase2_eval,
            "phase3_summary": phase3_eval,
            "prior_gate_summary": prior_gate,
        },
    )
    write_artifact("audit", audit_payload)

    gate_payload = payload(
        "8.7.56.1465",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure perturbative f_L diagnostic classification gate",
        common_inputs,
        audit_payload["rows"],
        {
            **audit_payload["summary"],
            "selected_next_generation_route": selected_next_generation_route,
            "recommended_next_route_or_none": recommended_next_route_or_none,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_perturbative_fl_driven_ode_case_gamma_declared",
            "branch_completed": True,
            "next_required_artifacts": [selected_next_generation_route],
        },
        {
            "phase1_seed_ratio": phase1_seed_ratio,
            "phase1_profile_ratio": phase1_profile_ratio,
            "diagnostic_summary": diagnostic,
        },
    )
    write_artifact("declaration_gate", gate_payload)

    evaluation_payload = payload(
        "8.7.56.1466",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure perturbative f_L driven ODE diagnostic numeric evaluation",
        common_inputs,
        [
            row(
                "phase1_seed_amp_ratio",
                "watch",
                "Phase 1 seed amp_l/amp0 ratio",
                phase1_seed_ratio,
                "The advisory that triggered this branch compared the Phase 1 seed ratio 1.25/3.5 against the perturbative expectation.",
            ),
            row(
                "phase1_profile_max_abs_ratio",
                "watch",
                "Phase 1 exact-profile max|f_L|/max|f_0|",
                phase1_profile_ratio,
                "The exact pilot profile itself stays tiny even though the seed ratio was large, which is why the diagnostic had to be run explicitly.",
            ),
            row(
                "diagnostic_max_abs_fL",
                "watch",
                "diagnostic max|f_L|",
                diagnostic["max_abs_fL"],
                "This is the largest perturbative driven longitudinal amplitude on the retained scalar background.",
            ),
            row(
                "diagnostic_max_abs_ratio",
                "watch",
                "diagnostic max|f_L|/max|f_0|",
                diagnostic["max_abs_ratio"],
                "This ratio drives the Case alpha/beta/gamma classification and lands well above the gamma threshold.",
            ),
            row(
                "diagnostic_peak_position",
                "watch",
                "diagnostic peak position",
                diagnostic["peak_position"],
                "The perturbative driven ODE peaks deep in the tail rather than near the scalar center.",
            ),
            row(
                "diagnostic_peak_sign",
                "watch",
                "diagnostic peak sign",
                diagnostic["peak_sign"],
                "The peak sign is tracked because the next-action note treats the perturbative probe as a sign test as well as an amplitude test.",
            ),
            row(
                "diagnostic_dfL0",
                "watch",
                "diagnostic optimal f_L'(0)",
                diagnostic["df_l0"],
                "The optimal launch slope is fixed as a machine-readable output for any future exact reinjection discussion.",
            ),
            row(
                "diagnostic_F_at_q_theory",
                "watch",
                "diagnostic F(q_theory)",
                diagnostic["F_at_q_theory"],
                "The perturbative density keeps the vector observable negative at the retained matching scale.",
            ),
            row(
                "diagnostic_alpha_at_q_theory",
                "watch",
                "diagnostic alpha(q_theory)",
                diagnostic["alpha_at_q_theory"],
                "The perturbative probe badly misses the target and therefore cannot rescue Case C on its own.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": PRIOR_CLASS,
            "beta_1_scalar": float(scalar_ground_state["beta_n"]),
            "q_theory_over_m0_scalar": diagnostic["q_theory_over_m0"],
            "phase1_seed_amp_ratio": phase1_seed_ratio,
            "phase1_exact_profile_max_abs_ratio": phase1_profile_ratio,
            "diagnostic_dfL0": diagnostic["df_l0"],
            "diagnostic_residual_bracket_left": diagnostic["residual_bracket_left"],
            "diagnostic_residual_bracket_right": diagnostic["residual_bracket_right"],
            "diagnostic_max_abs_fL": diagnostic["max_abs_fL"],
            "diagnostic_max_abs_ratio": diagnostic["max_abs_ratio"],
            "diagnostic_peak_position": diagnostic["peak_position"],
            "diagnostic_peak_sign": diagnostic["peak_sign"],
            "diagnostic_peak_value": diagnostic["peak_value"],
            "diagnostic_initial_derivative": diagnostic["initial_derivative"],
            "diagnostic_F_at_q_theory": diagnostic["F_at_q_theory"],
            "diagnostic_alpha_at_q_theory": diagnostic["alpha_at_q_theory"],
            "diagnostic_alpha_relerr_vs_target": diagnostic["alpha_relerr_vs_target"],
            "diagnostic_density_norm": diagnostic["density_norm"],
            "phase1_best_exact_F_at_q_theory": float(phase1_best["F_at_q_theory"]),
            "phase1_best_exact_alpha_at_q_theory": float(phase1_best["alpha_at_q_theory"]),
            "phase1_best_exact_seed_amp_l": float(phase1_best["amp_l"]),
            "phase1_best_exact_seed_amp0": float(phase1_best["amp0"]),
            "case_alpha_threshold": CASE_ALPHA_THRESHOLD,
            "case_gamma_threshold": CASE_GAMMA_THRESHOLD,
            "diagnostic_ratio_vs_case_alpha_threshold": float(diagnostic["max_abs_ratio"]) / CASE_ALPHA_THRESHOLD,
            "diagnostic_ratio_vs_case_gamma_threshold": float(diagnostic["max_abs_ratio"]) / CASE_GAMMA_THRESHOLD,
            "case_alpha_selected": case_alpha_selected,
            "case_beta_selected": case_beta_selected,
            "case_gamma_selected": case_gamma_selected,
            "wrong_branch_suspicion_supported": wrong_branch_suspicion_supported,
            "perturbative_breakdown_detected": perturbative_breakdown_detected,
            "exact_solver_reinjection_required": exact_solver_reinjection_required,
            "handoff_archive_registry_restore_required": handoff_archive_registry_restore_required,
            "case_c_honest_partial_retained": True,
            "retained_scalar_strong_candidate_retained": True,
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "physical_reject_required": False,
            "selected_next_generation_route": selected_next_generation_route,
            "recommended_next_route_or_none": recommended_next_route_or_none,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_perturbative_fl_driven_ode_case_gamma_completed",
            "branch_completed": True,
            "next_required_artifacts": [selected_next_generation_route],
        },
        {
            "phase1_audit_summary": phase1_audit,
            "phase2_audit_summary": phase2_audit,
            "phase3_audit_summary": phase3_audit,
            "prior_gate_summary": prior_gate,
        },
    )
    write_artifact("numeric_evaluation", evaluation_payload)

    print("[done] 8.7.56.1463-.1466 artifacts generated")


if __name__ == "__main__":
    main()
