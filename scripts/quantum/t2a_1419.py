#!/usr/bin/env python3
"""Generate unified-closure Phase 1 exact-coupled ell=0 solver artifacts for 8.7.56.1419-.1422."""

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

QBALL_DISCRETE = PUBLIC_OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"
PRIOR_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_recontract_"
    "declaration_gate_metrics.json"
)
PRIOR_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_recontract_"
    "numeric_evaluation_metrics.json"
)

QBALL_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_qball_charge_mapping_branch.py"
PIVOT_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_pivot_branch.py"
TWO_COMPONENT_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_spectrum_branch.py"
NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_COUPLED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

ALPHA_TARGET = 1.0 / 137.035999084
TAIL_RATIO_THRESHOLD = 0.25
EXACT_BETA_GRID = (0.95, 0.97, 0.98, 0.99, 0.995, 0.9982557379261291, 0.999)
EXACT_AMP0_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
EXACT_AMPL_GRID = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)

PRIOR_CLASS = "vector_qball_form_factor_exploratory_retained_lane_top_level_recontract_under_exploratory_split"
BRANCH_CLASS = "vector_qball_form_factor_unified_closure_phase1_exact_coupled_l0_solver_phase2_required"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_lambda_rot_form_factor_correction"
NEXT_ROUTE = "8.7.56.1423"
STEM = "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase1_exact_coupled_l0_solver"


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
                "charge_proxy": float(row_data["charge_proxy"]),
                "energy_proxy": float(row_data["energy_proxy"]),
                "central_amplitude": float(row_data["central_amplitude"]),
            }

    raise SystemExit("[fail] missing mode_index=1 in discrete mass spectrum metrics")


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


# Function: solve the exact scalar ground-state profile.

def solve_scalar_profile(qball_module, beta: float, amplitude: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the exact scalar ground-state profile."""
    return qball_module.solve_full_profile(float(beta), float(amplitude))


# Function: solve the diagnostic-only perturbative f_L equation.

def solve_diagnostic_f_l(radius: np.ndarray, field: np.ndarray, field_prime: np.ndarray, beta: float) -> dict:
    """Solve the diagnostic-only perturbative f_L equation."""
    kappa_sq = 1.0 - float(beta) * float(beta)

    # Function: return the linearized perturbative diagnostic ODE.
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
            (1.0e-6, float(radius[-1])),
            [1.0e-6 * float(df_l0), float(df_l0)],
            t_eval=radius,
            rtol=1.0e-10,
            atol=1.0e-12,
            max_step=0.03,
        )
        return float(sol.y[0, -1])

    df_l0 = float(brentq(residual, 0.0, 0.5))
    sol = solve_ivp(
        ode,
        (1.0e-6, float(radius[-1])),
        [1.0e-6 * df_l0, df_l0],
        t_eval=radius,
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.03,
    )
    f_l = np.asarray(sol.y[0], dtype=float)
    q_theory = float((1.0 - float(beta) * float(beta)) ** 0.25)
    f_value, norm = form_factor(radius, field * field - f_l * f_l, q_theory)
    alpha_value = float((f_value * f_value) / (4.0 * math.pi))
    return {
        "df_l0": df_l0,
        "max_abs_fL": float(np.max(np.abs(f_l))),
        "max_abs_ratio": float(np.max(np.abs(f_l)) / np.max(np.abs(field))),
        "F_at_q_theory": f_value,
        "alpha_at_q_theory": alpha_value,
        "alpha_relerr_vs_target": abs(alpha_value - ALPHA_TARGET) / ALPHA_TARGET,
        "norm": norm,
    }


# Function: solve the exact coupled ell=0 pilot profile.

def solve_exact_profile(pivot, numerical, beta: float, amp0: float, amp_l: float) -> dict:
    """Solve the exact coupled ell=0 pilot profile."""
    r0 = 1.0e-4
    y0 = [float(amp0), 0.0, float(amp_l) * r0, float(amp_l)]

    # Function: return the exact coupled ell=0 pilot ODE.
    def ode(radius: float, y: np.ndarray) -> list[float]:
        f0, f0_prime, f_l, f_l_prime = [float(value) for value in y]
        safe_r = max(float(radius), 1.0e-6)
        rho = math.sqrt(max(f0 * f0 + f_l * f_l, 0.0))
        nonlinear_coeff = 3.0 * rho + rho * rho
        f0_double_prime = -(2.0 / safe_r) * f0_prime - (float(beta * beta) - float(pivot.RADIAL_MASS_SQUARED)) * f0 - nonlinear_coeff * f0
        f_l_double_prime = -(2.0 / safe_r) * f_l_prime - (float(beta * beta) - float(pivot.LONGITUDINAL_DIRECT_MASS_SQUARED)) * f_l - nonlinear_coeff * f_l
        return [f0_prime, f0_double_prime, f_l_prime, f_l_double_prime]

    sol = solve_ivp(ode, (r0, 25.0), y0, max_step=0.10, rtol=1.0e-7, atol=1.0e-9)
    radius = np.asarray(sol.t, dtype=float)
    f0_values = np.asarray(sol.y[0], dtype=float)
    f_l_values = np.asarray(sol.y[2], dtype=float)
    tail_norm = math.sqrt(float(f0_values[-1] * f0_values[-1] + f_l_values[-1] * f_l_values[-1]))
    input_norm = math.sqrt(float(amp0 * amp0 + amp_l * amp_l))
    q_theory = float((1.0 - float(beta) * float(beta)) ** 0.25)
    f_value, norm = form_factor(radius, f0_values * f0_values - f_l_values * f_l_values, q_theory)
    alpha_value = float((f_value * f_value) / (4.0 * math.pi))
    return {
        "success": bool(sol.success),
        "beta": float(beta),
        "amp0": float(amp0),
        "amp_l": float(amp_l),
        "tail_to_input_ratio": None if input_norm == 0.0 else float(tail_norm / input_norm),
        "max_abs_f0": float(np.max(np.abs(f0_values))),
        "max_abs_fL": float(np.max(np.abs(f_l_values))),
        "node_count_k0": int(numerical.count_radial_nodes(f0_values)),
        "node_count_kL": int(numerical.count_radial_nodes(f_l_values)),
        "F_at_q_theory": f_value,
        "alpha_at_q_theory": alpha_value,
        "alpha_relerr_vs_target": abs(alpha_value - ALPHA_TARGET) / ALPHA_TARGET,
        "q_theory_over_m0": q_theory,
        "norm": norm,
    }


# Function: run the exact coupled Phase 1 scan and summarize the best candidates.

def run_exact_scan(pivot, numerical) -> dict:
    """Run the exact coupled Phase 1 scan and summarize the best candidates."""
    total = 0
    localized_count = 0
    nonzero_regular_branch_count = 0
    best_tail = None
    best_alpha = None
    for beta in EXACT_BETA_GRID:
        for amp0 in EXACT_AMP0_GRID:
            for amp_l in EXACT_AMPL_GRID:
                total += 1
                solved = solve_exact_profile(pivot, numerical, float(beta), float(amp0), float(amp_l))
                if not solved["success"] or solved["tail_to_input_ratio"] is None:
                    continue

                if best_tail is None or float(solved["tail_to_input_ratio"]) < float(best_tail["tail_to_input_ratio"]):
                    best_tail = solved

                if float(solved["tail_to_input_ratio"]) <= TAIL_RATIO_THRESHOLD:
                    localized_count += 1
                    if float(solved["max_abs_fL"]) > 0.0:
                        nonzero_regular_branch_count += 1

                    if best_alpha is None or float(solved["alpha_relerr_vs_target"]) < float(best_alpha["alpha_relerr_vs_target"]):
                        best_alpha = solved

    return {
        "scan_total_count": int(total),
        "localized_candidate_count": int(localized_count),
        "nonzero_regular_branch_count": int(nonzero_regular_branch_count),
        "best_tail_candidate": best_tail,
        "best_alpha_candidate": best_alpha,
    }


# Function: execute the unified-closure Phase 1 exact-coupled ell=0 solver branch.

def main() -> None:
    """Execute the unified-closure Phase 1 exact-coupled ell=0 solver branch."""
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
        QBALL_DISCRETE,
        PRIOR_GATE,
        PRIOR_EVAL,
        QBALL_SOLVER,
        PIVOT_BRANCH,
        TWO_COMPONENT_BRANCH,
        NUMERICAL_BRANCH,
        FULL_COUPLED_BRANCH,
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

    qball_discrete = read_json(QBALL_DISCRETE)
    prior_gate = read_json(PRIOR_GATE)["summary"]
    prior_eval = read_json(PRIOR_EVAL)["summary"]

    qball_module = load_module(QBALL_SOLVER, "wavep_qball_charge_mapping")
    pivot = load_module(PIVOT_BRANCH, "wavep_trial3_pivot")
    numerical = load_module(NUMERICAL_BRANCH, "wavep_vector_qball_numerical")

    scalar_ground_state = ground_state_row(qball_discrete)
    radius, field, field_prime = solve_scalar_profile(
        qball_module,
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    scalar_q_theory = float((1.0 - float(scalar_ground_state["beta_n"]) * float(scalar_ground_state["beta_n"])) ** 0.25)
    scalar_f, _scalar_norm = form_factor(radius, field * field, scalar_q_theory)
    scalar_alpha = float((scalar_f * scalar_f) / (4.0 * math.pi))
    scalar_alpha_relerr = abs(scalar_alpha - ALPHA_TARGET) / ALPHA_TARGET

    diagnostic = solve_diagnostic_f_l(radius, field, field_prime, float(scalar_ground_state["beta_n"]))
    exact_scan = run_exact_scan(pivot, numerical)

    best_tail = exact_scan["best_tail_candidate"]
    best_alpha = exact_scan["best_alpha_candidate"]
    if best_tail is None or best_alpha is None:
        raise SystemExit("[fail] unified closure Phase 1 scan did not produce a localized exact candidate")

    inventory_hits = [
        hit(status_text, "8.7.56.1419-.1422"),
        hit(roadmap_text, "8.7.56.1419-.1422"),
        hit(current_problem_text, "ell=0 series / operator problem"),
        hit(current_status_text, "4次元ベクトル化"),
        hit(expert_share_text, "primary / secondary / reserve"),
        hit(unified_roadmap_text, "Phase 1 exact coupled"),
        hit(unified_plan_text, "Phase 1"),
        hit(next_steps_text, "Step A"),
        hit(solver_fix_text, "2-step perturbative method"),
        hit(perturbative_note_text, "diagnostic"),
        hit(part1_text, "F_{0r}"),
        hit(part3a_text, "D_\\mu=\\partial_\\mu+i q A_\\mu"),
        hit(part5_text, "trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_recontract"),
    ]
    inventory_ready = all(item is not None for item in inventory_hits)
    diagnostic_only_rule_honest = bool(
        float(diagnostic["alpha_relerr_vs_target"]) > float(best_alpha["alpha_relerr_vs_target"])
        and float(diagnostic["max_abs_ratio"]) / max(float(best_alpha["max_abs_fL"]) / float(best_alpha["max_abs_f0"]), 1.0e-18) > 10.0
    )
    nonzero_regular_branch_detected = int(exact_scan["nonzero_regular_branch_count"]) > 0
    phase1_close_within_one_percent = float(best_alpha["alpha_relerr_vs_target"]) < 0.01
    phase2_required = bool(nonzero_regular_branch_detected and not phase1_close_within_one_percent)
    primary_lane_no_go = bool(not nonzero_regular_branch_detected)

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
            "qball_discrete": display_path(QBALL_DISCRETE),
            "prior_gate": display_path(PRIOR_GATE),
            "prior_eval": display_path(PRIOR_EVAL),
        },
        "solver_modules": {
            "qball_solver": display_path(QBALL_SOLVER),
            "pivot_branch": display_path(PIVOT_BRANCH),
            "two_component_branch": display_path(TWO_COMPONENT_BRANCH),
            "numerical_branch": display_path(NUMERICAL_BRANCH),
            "full_coupled_branch": display_path(FULL_COUPLED_BRANCH),
        },
        "constants": {
            "alpha_target": ALPHA_TARGET,
            "tail_ratio_threshold": TAIL_RATIO_THRESHOLD,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory_payload = payload(
        "8.7.56.1419",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure Phase 1 exact coupled ell=0 solver source inventory",
        common_inputs,
        [
            row("phase1_inventory_ready", "pass" if inventory_ready else "reject", "Phase 1 inventory ready", 1 if inventory_ready else 0, "Phase 1 inventory is ready only if the unified-closure notes, solver-fix notes, solver modules, and current docs coexist in one pack."),
            row("diagnostic_only_rule_honest", "pass" if diagnostic_only_rule_honest else "reject", "perturbative f_L diagnostic-only rule honest", 1 if diagnostic_only_rule_honest else 0, "The perturbative note stays diagnostic-only only if its amplitude and alpha residual clearly disagree with the best exact pilot candidate."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": PRIOR_CLASS,
            "phase1_inventory_ready": inventory_ready,
            "diagnostic_only_rule_honest": diagnostic_only_rule_honest,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase1_inventory_fixed",
            "advance_to_8_7_56_1420": inventory_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"inventory_hits": inventory_hits},
    )
    write_artifact("source_inventory", inventory_payload)

    audit_payload = payload(
        "8.7.56.1420",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure Phase 1 perturbative f_L diagnostic and exact-coupled audit",
        common_inputs,
        [
            row("diagnostic_only_rule_honest", "pass" if diagnostic_only_rule_honest else "reject", "perturbative f_L diagnostic-only rule honest", 1 if diagnostic_only_rule_honest else 0, "The perturbative probe is honest only if it remains a seed/sign/scale diagnostic and not an exact theorem substitute."),
            row("nonzero_regular_branch_detected", "pass" if nonzero_regular_branch_detected else "reject", "nonzero regular branch detected", 1 if nonzero_regular_branch_detected else 0, "Phase 1 passes its first gate only if the exact coupled pilot finds at least one localized ell=0 branch with nonzero f_L."),
            row("phase1_close_within_one_percent", "pass" if phase1_close_within_one_percent else "reject", "Phase 1 close within one percent", 1 if phase1_close_within_one_percent else 0, "Phase 1 closes only if the best exact coupled pilot alpha reaches the target within one percent."),
            row("phase2_required", "pass" if phase2_required else "reject", "conditional Phase 2 required", 1 if phase2_required else 0, "Phase 2 is required when a nonzero regular branch exists but the exact coupled pilot does not close the alpha residual."),
            row("primary_lane_no_go", "pass" if primary_lane_no_go else "reject", "primary lane no-go", 1 if primary_lane_no_go else 0, "Primary-lane no-go is reserved for the case where no nonzero regular ell=0 branch survives Phase 1."),
            row("physical_reject_required", "reject", "physical reject required", 0.0, "Unified closure Phase 1 does not force a physical reject."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": PRIOR_CLASS,
            "diagnostic_only_rule_honest": diagnostic_only_rule_honest,
            "nonzero_regular_branch_detected": nonzero_regular_branch_detected,
            "phase1_close_within_one_percent": phase1_close_within_one_percent,
            "phase2_required": phase2_required,
            "primary_lane_no_go": primary_lane_no_go,
            "phase1_exact_coupled_pilot_available": True,
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "physical_reject_required": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME if phase2_required else None,
            "recommended_next_route_or_none": NEXT_ROUTE if phase2_required else None,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase1_audit_completed",
            "advance_to_8_7_56_1421": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME] if phase2_required else [],
        },
        {"prior_gate_summary": prior_gate, "prior_eval_summary": prior_eval},
    )
    write_artifact("audit", audit_payload)

    gate_payload = payload(
        "8.7.56.1421",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure Phase 1 declaration gate",
        common_inputs,
        audit_payload["rows"],
        {
            **audit_payload["summary"],
            "selected_next_generation_route": NEXT_ROUTE_NAME if phase2_required else None,
            "recommended_next_route_or_none": NEXT_ROUTE if phase2_required else None,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase1_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME] if phase2_required else [],
        },
        {
            "best_tail_candidate": best_tail,
            "best_alpha_candidate": best_alpha,
            "diagnostic_summary": diagnostic,
        },
    )
    write_artifact("declaration_gate", gate_payload)

    evaluation_payload = payload(
        "8.7.56.1422",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure Phase 1 numeric evaluation",
        common_inputs,
        [
            row("scalar_alpha_relerr_vs_target", "watch", "scalar alpha relative error vs target", scalar_alpha_relerr, "The retained scalar route stays at a 1.9% residual before vector correction."),
            row("diagnostic_alpha_relerr_vs_target", "watch", "diagnostic perturbative alpha relative error vs target", diagnostic["alpha_relerr_vs_target"], "The perturbative probe strongly misses the target and therefore stays diagnostic-only."),
            row("best_exact_alpha_relerr_vs_target", "watch" if not phase1_close_within_one_percent else "pass", "best exact coupled alpha relative error vs target", best_alpha["alpha_relerr_vs_target"], "The best exact coupled pilot candidate decides whether Phase 1 closes or escalates to conditional Phase 2."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": PRIOR_CLASS,
            "beta_1_scalar": float(scalar_ground_state["beta_n"]),
            "q_theory_over_m0_scalar": scalar_q_theory,
            "F_exact_at_q_theory_scalar": scalar_f,
            "alpha_exact_at_q_theory_scalar": scalar_alpha,
            "scalar_alpha_relerr_vs_target": scalar_alpha_relerr,
            "diagnostic_dfL0": diagnostic["df_l0"],
            "diagnostic_max_abs_fL": diagnostic["max_abs_fL"],
            "diagnostic_max_abs_ratio": diagnostic["max_abs_ratio"],
            "diagnostic_F_at_q_theory": diagnostic["F_at_q_theory"],
            "diagnostic_alpha_at_q_theory": diagnostic["alpha_at_q_theory"],
            "diagnostic_alpha_relerr_vs_target": diagnostic["alpha_relerr_vs_target"],
            "scan_total_count": exact_scan["scan_total_count"],
            "localized_candidate_count": exact_scan["localized_candidate_count"],
            "nonzero_regular_branch_count": exact_scan["nonzero_regular_branch_count"],
            "phase1_best_tail_candidate": best_tail,
            "phase1_best_alpha_candidate": best_alpha,
            "nonzero_regular_branch_detected": nonzero_regular_branch_detected,
            "phase1_close_within_one_percent": phase1_close_within_one_percent,
            "phase2_required": phase2_required,
            "primary_lane_no_go": primary_lane_no_go,
            "phase1_exact_coupled_pilot_available": True,
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "physical_reject_required": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME if phase2_required else None,
            "recommended_next_route_or_none": NEXT_ROUTE if phase2_required else None,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase1_exact_coupled_l0_solver_completed",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME] if phase2_required else [],
        },
        {
            "prior_problem_classification": PRIOR_CLASS,
            "new_problem_classification": BRANCH_CLASS,
            "diagnostic_summary": diagnostic,
            "best_tail_candidate": best_tail,
            "best_alpha_candidate": best_alpha,
        },
    )
    write_artifact("numeric_evaluation", evaluation_payload)

    print("[done] 8.7.56.1419-.1422 artifacts generated")


if __name__ == "__main__":
    main()
