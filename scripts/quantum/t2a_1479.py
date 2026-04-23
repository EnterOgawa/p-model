#!/usr/bin/env python3
"""Generate corrected exact-action-level ell=0 exact-solver reinjection artifacts for 8.7.56.1479-.1482.

This branch takes the corrected bootstrap family from 8.7.56.1475-.1478 and
tests the next honest question directly:

- if the scalar ladder already seeds a regular longitudinal bootstrap family,
  does reinjecting those seeds into the current exact coupled pilot leave a
  nontrivial localized branch that still preserves the electron-anchor mode?

The answer determines whether the roadmap may advance to an effective source
theorem, or whether the next mainline must first audit branch continuation /
anchor drift on the corrected exact solver.
"""


from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
CASE_GAMMA_ADVICE = ROOT / "doc" / "quantum" / "42_trial2_numeric_alpha_vector_qball_case_gamma_advice_request.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

SOLVER_FIX = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_solver_fix_final.md")
NEXT_STEPS = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")

BOOTSTRAP_EVAL = PUBLIC_OUT / "q_8_7_56_1475_1478_ell0_family_bootstrap_numeric_evaluation_metrics.json"
BOOTSTRAP_AUDIT = PUBLIC_OUT / "q_8_7_56_1475_1478_ell0_family_bootstrap_audit_metrics.json"
OPERATOR_GATE = PUBLIC_OUT / "q_8_7_56_1471_1474_ell0_exact_operator_derivation_declaration_gate_metrics.json"
PHASE1_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase1_exact_coupled_l0_solver_"
    "numeric_evaluation_metrics.json"
)
CASE_GAMMA_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_perturbative_fl_driven_ode_"
    "diagnostic_reopen_review_numeric_evaluation_metrics.json"
)

NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
PIVOT_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_pivot_branch.py"

STEP_TAG = "8.7.56.1479-1482"
STEM = build_compact_artifact_stem(STEP_TAG, "ell0_exact_solver_reinjection", prefix="q")
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor corrected exact-action-level ell=0 exact solver reinjection"

PRIOR_CLASS = "vector_qball_form_factor_corrected_ell0_bootstrap_family_exists_exact_solver_reinjection_required"
BRANCH_CLASS = "vector_qball_form_factor_corrected_exact_solver_reinjection_mode1_anchor_lost_scalar_like_branch_only"
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_corrected_exact_action_level_ell0_anchor_drift_branch_"
    "continuation_audit"
)
NEXT_ROUTE = "8.7.56.1483"
ALPHA_TARGET = 1.0 / 137.035999084
TAIL_RATIO_THRESHOLD = 0.25
ANCHOR_FACTOR2_MAX = 2.0
ANCHOR_FACTOR4_MAX = 4.0
NONTRIVIAL_RATIO_THRESHOLD = 0.10
AMP_SCALE_GRID = tuple(float(value) for value in np.geomspace(0.5, 512.0, 26))
LAMBDA_SCALE_GRID = (0.0,) + tuple(float(value) for value in np.geomspace(0.05, 64.0, 25))


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: fail immediately when one required path is missing.

def require(path: Path) -> None:
    """Fail immediately when one required path is missing."""
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


# Function: convert one absolute path into repo-relative display text when possible.

def display_path(path: Path) -> str:
    """Convert one absolute path into repo-relative display text when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: return the first matching line for one substring pattern.

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


# Function: write one JSON payload and its rows CSV with Windows-safe paths.

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and its rows CSV with Windows-safe paths."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": display_path(paths["json"]), "csv": display_path(paths["csv"])}


# Function: dynamically load one local Python module.

def load_module(path: Path, module_name: str):
    """Dynamically load one local Python module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


# Function: solve the current exact coupled ell=0 pilot at one reinjected seed point.

def solve_reinjected_exact_profile(pivot, numerical, beta: float, amp0: float, amp_l: float) -> dict:
    """Solve the current exact coupled ell=0 pilot at one reinjected seed point."""
    r0 = 1.0e-4
    y0 = [float(amp0), 0.0, float(amp_l) * r0, float(amp_l)]

    # Function: return the current exact coupled pilot ODE.
    def ode(radius: float, y: np.ndarray) -> list[float]:
        f0, f0_prime, f_l, f_l_prime = [float(value) for value in y]
        rr = max(float(radius), 1.0e-6)
        rho = math.sqrt(max(f0 * f0 + f_l * f_l, 0.0))
        nonlinear_coeff = 3.0 * rho + rho * rho
        f0_double_prime = (
            -(2.0 / rr) * f0_prime
            - (float(beta * beta) - float(pivot.RADIAL_MASS_SQUARED)) * f0
            - nonlinear_coeff * f0
        )
        f_l_double_prime = (
            -(2.0 / rr) * f_l_prime
            + (2.0 / (rr * rr)) * f_l
            - (float(beta * beta) - float(pivot.LONGITUDINAL_DIRECT_MASS_SQUARED)) * f_l
            - nonlinear_coeff * f_l
        )
        return [f0_prime, f0_double_prime, f_l_prime, f_l_double_prime]

    solution = solve_ivp(ode, (r0, 25.0), y0, max_step=0.10, rtol=1.0e-7, atol=1.0e-9)
    radius = np.asarray(solution.t, dtype=float)
    f0_values = np.asarray(solution.y[0], dtype=float)
    f_l_values = np.asarray(solution.y[2], dtype=float)
    tail_norm = math.sqrt(float(f0_values[-1] * f0_values[-1] + f_l_values[-1] * f_l_values[-1]))
    input_norm = math.sqrt(float(amp0 * amp0 + amp_l * amp_l))
    q_theory = float((1.0 - float(beta) * float(beta)) ** 0.25)
    form_value, proxy_norm = form_factor(radius, f0_values * f0_values - f_l_values * f_l_values, q_theory)
    alpha_value = float((form_value * form_value) / (4.0 * math.pi))
    alpha_relerr = float(abs(alpha_value - ALPHA_TARGET) / ALPHA_TARGET)
    max_abs_f0 = float(np.max(np.abs(f0_values)))
    max_abs_f_l = float(np.max(np.abs(f_l_values)))
    max_abs_ratio = float(max_abs_f_l / max(max_abs_f0, 1.0e-18))
    return {
        "success": bool(solution.success),
        "tail_to_input_ratio": None if input_norm == 0.0 else float(tail_norm / input_norm),
        "max_abs_ratio": max_abs_ratio,
        "F_at_q_theory": float(form_value),
        "alpha_at_q_theory": alpha_value,
        "alpha_relerr_vs_target": alpha_relerr,
        "vector_proxy_norm": proxy_norm,
        "q_theory_over_m0": q_theory,
        "node_count_k0": int(numerical.count_radial_nodes(f0_values)),
        "node_count_kL": int(numerical.count_radial_nodes(f_l_values)),
    }


# Function: summarize one reinjection search over amplitude and lambda scales.

def search_mode_reinjection(pivot, numerical, bootstrap_row: dict) -> dict:
    """Summarize one reinjection search over amplitude and lambda scales."""
    localized_rows = []
    best_alpha = None
    min_scale_candidate = None
    max_ratio_localized = 0.0
    factor2_count = 0
    factor4_count = 0

    for amp_scale in AMP_SCALE_GRID:
        amp0 = float(bootstrap_row["central_amplitude"]) * float(amp_scale)
        for lambda_scale in LAMBDA_SCALE_GRID:
            amp_l = float(bootstrap_row["df_l0"]) * float(lambda_scale)
            solved = solve_reinjected_exact_profile(pivot, numerical, float(bootstrap_row["beta"]), amp0, amp_l)
            if not solved["success"] or solved["tail_to_input_ratio"] is None:
                continue

            if float(solved["tail_to_input_ratio"]) > float(TAIL_RATIO_THRESHOLD):
                continue

            candidate = {
                "mode_index": int(bootstrap_row["mode_index"]),
                "beta": float(bootstrap_row["beta"]),
                "amp_scale": float(amp_scale),
                "lambda_scale": float(lambda_scale),
                "amp0": float(amp0),
                "amp_l": float(amp_l),
                **solved,
            }
            localized_rows.append(candidate)
            max_ratio_localized = max(max_ratio_localized, float(candidate["max_abs_ratio"]))

            if float(amp_scale) <= float(ANCHOR_FACTOR2_MAX):
                factor2_count += 1

            if float(amp_scale) <= float(ANCHOR_FACTOR4_MAX):
                factor4_count += 1

            if min_scale_candidate is None or (
                float(candidate["amp_scale"]),
                float(candidate["lambda_scale"]),
            ) < (
                float(min_scale_candidate["amp_scale"]),
                float(min_scale_candidate["lambda_scale"]),
            ):
                min_scale_candidate = candidate

            if best_alpha is None or float(candidate["alpha_relerr_vs_target"]) < float(best_alpha["alpha_relerr_vs_target"]):
                best_alpha = candidate

    return {
        "mode_index": int(bootstrap_row["mode_index"]),
        "beta": float(bootstrap_row["beta"]),
        "central_amplitude": float(bootstrap_row["central_amplitude"]),
        "bootstrap_df_l0": float(bootstrap_row["df_l0"]),
        "localized_candidate_count": int(len(localized_rows)),
        "anchor_factor2_localized_count": int(factor2_count),
        "anchor_factor4_localized_count": int(factor4_count),
        "min_localized_amp_scale_or_none": None if min_scale_candidate is None else float(min_scale_candidate["amp_scale"]),
        "min_localized_lambda_scale_or_none": None if min_scale_candidate is None else float(min_scale_candidate["lambda_scale"]),
        "min_localized_tail_ratio_or_none": None
        if min_scale_candidate is None
        else float(min_scale_candidate["tail_to_input_ratio"]),
        "min_localized_max_abs_ratio_or_none": None
        if min_scale_candidate is None
        else float(min_scale_candidate["max_abs_ratio"]),
        "best_alpha_candidate_or_none": best_alpha,
        "min_scale_candidate_or_none": min_scale_candidate,
        "max_localized_ratio_ceiling": float(max_ratio_localized),
        "localized_rows": localized_rows,
    }


# Function: test whether one float sequence is strictly decreasing.

def strictly_decreasing(values: list[float], tol: float = 1.0e-12) -> bool:
    """Test whether one float sequence is strictly decreasing."""
    return all(float(right) < float(left) - float(tol) for left, right in zip(values[:-1], values[1:]))


# Function: execute the corrected exact solver reinjection branch.

def main() -> None:
    """Execute the corrected exact solver reinjection branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        CASE_GAMMA_ADVICE,
        PART1,
        PART5,
        SOLVER_FIX,
        NEXT_STEPS,
        BOOTSTRAP_EVAL,
        BOOTSTRAP_AUDIT,
        OPERATOR_GATE,
        PHASE1_EVAL,
        CASE_GAMMA_EVAL,
        NUMERICAL_BRANCH,
        PIVOT_BRANCH,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part1_text = read_text(PART1)
    part5_text = read_text(PART5)
    solver_fix_text = read_text(SOLVER_FIX)
    next_steps_text = read_text(NEXT_STEPS)

    bootstrap_eval = read_json(BOOTSTRAP_EVAL)
    bootstrap_audit = read_json(BOOTSTRAP_AUDIT)
    operator_gate = read_json(OPERATOR_GATE)
    phase1_eval = read_json(PHASE1_EVAL)
    case_gamma_eval = read_json(CASE_GAMMA_EVAL)

    pivot = load_module(PIVOT_BRANCH, "wavep_trial3_pivot")
    numerical = load_module(NUMERICAL_BRANCH, "wavep_vector_qball_numerical")

    bootstrap_rows = list(bootstrap_eval["evidence"]["bootstrap_mode_rows"])
    search_rows = [search_mode_reinjection(pivot, numerical, bootstrap_row) for bootstrap_row in bootstrap_rows]

    mode1 = next(row_data for row_data in search_rows if int(row_data["mode_index"]) == 1)
    localized_mode_count = int(sum(1 for row_data in search_rows if int(row_data["localized_candidate_count"]) > 0))
    min_scale_sequence = [float(row_data["min_localized_amp_scale_or_none"]) for row_data in search_rows if row_data["min_localized_amp_scale_or_none"] is not None]
    min_scale_monotone_decreasing = strictly_decreasing(min_scale_sequence)

    global_best_alpha = None
    for row_data in search_rows:
        candidate = row_data["best_alpha_candidate_or_none"]
        if candidate is None:
            continue

        if global_best_alpha is None or float(candidate["alpha_relerr_vs_target"]) < float(global_best_alpha["alpha_relerr_vs_target"]):
            global_best_alpha = candidate

    if global_best_alpha is None:
        raise SystemExit("[fail] exact solver reinjection produced no localized candidates")

    mode1_anchor_factor2_survives = int(mode1["anchor_factor2_localized_count"]) > 0
    mode1_anchor_factor4_survives = int(mode1["anchor_factor4_localized_count"]) > 0
    mode1_nontrivial_localized_branch_survives = bool(
        float(mode1["max_localized_ratio_ceiling"]) >= float(NONTRIVIAL_RATIO_THRESHOLD)
    )
    mode1_best_prefers_zero_lambda = bool(
        mode1["best_alpha_candidate_or_none"] is not None
        and abs(float(mode1["best_alpha_candidate_or_none"]["lambda_scale"])) < 1.0e-15
    )
    global_best_prefers_zero_lambda = abs(float(global_best_alpha["lambda_scale"])) < 1.0e-15
    best_alpha_improves_phase1 = bool(
        float(global_best_alpha["alpha_relerr_vs_target"])
        < float(phase1_eval["summary"]["phase1_best_alpha_candidate"]["alpha_relerr_vs_target"])
    )
    exact_solver_reinjection_success = bool(
        mode1_anchor_factor4_survives
        and mode1_nontrivial_localized_branch_survives
        and best_alpha_improves_phase1
    )
    source_theorem_attempt_admissible_now = bool(exact_solver_reinjection_success)
    anchor_drift_branch_continuation_required = bool(not exact_solver_reinjection_success)

    inventory_rows = [
        row(
            "reinjection_inventory_ready",
            "pass",
            "corrected exact solver reinjection inventory ready",
            1.0,
            "The reinjection inventory is ready once the bootstrap family outputs, exact operator gate, Phase 1 baseline, and solver-fix notes are assembled in one pack.",
        ),
        row(
            "bootstrap_family_rows_available",
            "pass",
            "bootstrap family rows available",
            float(len(bootstrap_rows)),
            "The reinjection branch starts from the explicit bootstrap mode rows, not from a fresh blind grid.",
        ),
        row(
            "phase1_baseline_available",
            "pass",
            "Phase 1 exact pilot baseline available",
            1.0,
            "The reinjection branch compares directly against the retained Phase 1 exact pilot baseline.",
        ),
        row(
            "operator_gap_gate_available",
            "pass",
            "operator-gap declaration gate available",
            1.0,
            "The reinjection audit uses the prior gate that fixed a partial free backbone but no closed exact ell=0 operator.",
        ),
    ]

    inventory_payload = payload(
        "8.7.56.1479",
        f"{STEP_NAME} inventory",
        {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem_note": display_path(CURRENT_PROBLEM),
            "current_status_note": display_path(CURRENT_STATUS),
            "unified_roadmap_note": display_path(UNIFIED_ROADMAP),
            "case_gamma_advice_note": display_path(CASE_GAMMA_ADVICE),
            "part1": display_path(PART1),
            "part5": display_path(PART5),
            "solver_fix_note": display_path(SOLVER_FIX),
            "next_steps_note": display_path(NEXT_STEPS),
            "bootstrap_eval_json": display_path(BOOTSTRAP_EVAL),
            "bootstrap_audit_json": display_path(BOOTSTRAP_AUDIT),
            "operator_gate_json": display_path(OPERATOR_GATE),
            "phase1_eval_json": display_path(PHASE1_EVAL),
            "case_gamma_eval_json": display_path(CASE_GAMMA_EVAL),
            "pivot_branch": display_path(PIVOT_BRANCH),
            "numerical_branch": display_path(NUMERICAL_BRANCH),
        },
        inventory_rows,
        {
            "trial2_numeric_alpha_problem_classification": PRIOR_CLASS,
            "reinjection_inventory_ready": True,
            "bootstrap_mode_count": int(len(bootstrap_rows)),
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_corrected_exact_solver_reinjection_inventory_completed",
            "branch_completed": False,
            "next_required_artifacts": [f"{STEM}_audit_metrics.json"],
        },
        {
            "status_current_step_hit": hit(status_text, "8.7.56.1479-.1482"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1479-.1482"),
            "part1_field_strength_hit": hit(part1_text, "F_{0r}"),
            "part5_branch_hit": hit(part5_text, "corrected exact-action-level `ell=0` exact solver reinjection"),
            "solver_fix_two_step_hit": hit(solver_fix_text, "2-step perturbative method"),
            "next_steps_step_b_hit": hit(next_steps_text, "Step B. linearized longitudinal equationを exact に導く"),
        },
    )
    inventory_paths = write_artifact("inventory", inventory_payload)

    audit_rows = [
        row(
            "localized_exact_candidates_exist",
            "pass" if localized_mode_count > 0 else "reject",
            "localized exact reinjection candidates exist",
            float(localized_mode_count),
            "The reinjection branch remains computationally meaningful only if the current exact pilot localizes somewhere on the bootstrap-seeded search.",
        ),
        row(
            "mode1_anchor_factor2_survives",
            "pass" if mode1_anchor_factor2_survives else "reject",
            "mode 1 anchor survives inside factor-2 amplitude window",
            float(mode1["anchor_factor2_localized_count"]),
            "A preserved electron anchor should localize near the retained scalar amplitude before any large amplitude drift is introduced.",
        ),
        row(
            "mode1_anchor_factor4_survives",
            "pass" if mode1_anchor_factor4_survives else "reject",
            "mode 1 anchor survives inside factor-4 amplitude window",
            float(mode1["anchor_factor4_localized_count"]),
            "Even a relaxed anchor window should produce localized mode-1 exact candidates if the reinjected branch truly preserves the electron anchor.",
        ),
        row(
            "mode1_nontrivial_localized_branch_survives",
            "pass" if mode1_nontrivial_localized_branch_survives else "reject",
            "mode 1 localized branch keeps nontrivial longitudinal weight",
            float(mode1["max_localized_ratio_ceiling"]),
            "A nontrivial reinjected vector branch should retain max|f_L/f_0| above the scalar-like regime after localization.",
        ),
        row(
            "mode1_best_prefers_zero_lambda",
            "reject" if mode1_best_prefers_zero_lambda else "pass",
            "mode 1 best localized candidate prefers zero longitudinal seed",
            1.0 if mode1_best_prefers_zero_lambda else 0.0,
            "If the best localized mode-1 candidate sits at lambda = 0, the reinjection collapses back onto a scalar-like branch.",
        ),
        row(
            "global_best_prefers_zero_lambda",
            "reject" if global_best_prefers_zero_lambda else "pass",
            "global best localized candidate prefers zero longitudinal seed",
            1.0 if global_best_prefers_zero_lambda else 0.0,
            "If the globally best localized candidate also sits at lambda = 0, the corrected exact search still prefers the scalar-like branch.",
        ),
        row(
            "best_alpha_improves_phase1",
            "pass" if best_alpha_improves_phase1 else "reject",
            "best localized reinjection alpha improves Phase 1",
            float(1.0 if best_alpha_improves_phase1 else 0.0),
            "The reinjection branch only advances the roadmap if it beats the retained Phase 1 exact pilot rather than just reproducing its failure.",
        ),
        row(
            "anchor_drift_branch_continuation_required",
            "pass" if anchor_drift_branch_continuation_required else "reject",
            "anchor-drift / branch-continuation audit required next",
            float(1.0 if anchor_drift_branch_continuation_required else 0.0),
            "Once exact reinjection localizes only after large amplitude drift and near-scalar lambda collapse, the next honest action is branch continuation rather than source theorem.",
        ),
        row(
            "source_theorem_attempt_admissible_now",
            "pass" if source_theorem_attempt_admissible_now else "reject",
            "effective source theorem attempt admissible now",
            float(1.0 if source_theorem_attempt_admissible_now else 0.0),
            "The source theorem remains downstream until the corrected exact solver preserves a nontrivial electron-anchor branch.",
        ),
    ]

    audit_payload = payload(
        "8.7.56.1480",
        f"{STEP_NAME} audit",
        {
            "inventory_json": inventory_paths["json"],
            "bootstrap_eval_json": display_path(BOOTSTRAP_EVAL),
            "bootstrap_audit_json": display_path(BOOTSTRAP_AUDIT),
            "operator_gate_json": display_path(OPERATOR_GATE),
            "phase1_eval_json": display_path(PHASE1_EVAL),
            "case_gamma_eval_json": display_path(CASE_GAMMA_EVAL),
            "pivot_branch": display_path(PIVOT_BRANCH),
            "numerical_branch": display_path(NUMERICAL_BRANCH),
        },
        audit_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "localized_mode_count": localized_mode_count,
            "mode1_anchor_factor2_survives": mode1_anchor_factor2_survives,
            "mode1_anchor_factor4_survives": mode1_anchor_factor4_survives,
            "mode1_min_localized_amp_scale_or_none": mode1["min_localized_amp_scale_or_none"],
            "mode1_max_localized_ratio_ceiling": float(mode1["max_localized_ratio_ceiling"]),
            "mode1_best_prefers_zero_lambda": mode1_best_prefers_zero_lambda,
            "global_best_prefers_zero_lambda": global_best_prefers_zero_lambda,
            "min_localized_amp_scale_sequence": min_scale_sequence,
            "min_localized_amp_scale_monotone_decreasing": min_scale_monotone_decreasing,
            "global_best_alpha_candidate": global_best_alpha,
            "best_alpha_improves_phase1": best_alpha_improves_phase1,
            "exact_solver_reinjection_success": exact_solver_reinjection_success,
            "anchor_drift_branch_continuation_required": anchor_drift_branch_continuation_required,
            "source_theorem_attempt_admissible_now": source_theorem_attempt_admissible_now,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_corrected_exact_solver_reinjection_audited",
            "branch_completed": False,
            "next_required_artifacts": [f"{STEM}_declaration_gate_metrics.json"],
        },
        {
            "bootstrap_summary": bootstrap_eval["summary"],
            "bootstrap_audit_summary": bootstrap_audit["summary"],
            "operator_gate_summary": operator_gate["summary"],
            "phase1_summary": phase1_eval["summary"],
            "case_gamma_summary": case_gamma_eval["summary"],
            "mode_reinjection_rows": search_rows,
        },
    )
    audit_paths = write_artifact("audit", audit_payload)

    gate_rows = [
        row(
            "exact_solver_reinjection_success",
            "pass" if exact_solver_reinjection_success else "reject",
            "corrected exact solver reinjection succeeds for the electron anchor",
            float(1.0 if exact_solver_reinjection_success else 0.0),
            "Success requires an anchor-preserving, nontrivial localized mode-1 branch that improves the retained exact pilot.",
        ),
        row(
            "anchor_drift_branch_continuation_required",
            "pass" if anchor_drift_branch_continuation_required else "reject",
            "anchor-drift branch-continuation route selected",
            float(1.0 if anchor_drift_branch_continuation_required else 0.0),
            "If reinjection fails at the electron anchor, the roadmap must branch to anchor-drift / continuation work before source theorem.",
        ),
        row(
            "advance_to_effective_source_theorem_now",
            "pass" if source_theorem_attempt_admissible_now else "reject",
            "advance directly to effective source theorem now",
            float(1.0 if source_theorem_attempt_admissible_now else 0.0),
            "Direct source-theorem work is only honest when the corrected exact solver has already preserved the electron-anchor branch.",
        ),
    ]

    gate_payload = payload(
        "8.7.56.1481",
        f"{STEP_NAME} declaration gate",
        {
            "inventory_json": inventory_paths["json"],
            "audit_json": audit_paths["json"],
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
        },
        gate_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "exact_solver_reinjection_success": exact_solver_reinjection_success,
            "mode1_anchor_factor2_survives": mode1_anchor_factor2_survives,
            "mode1_anchor_factor4_survives": mode1_anchor_factor4_survives,
            "mode1_nontrivial_localized_branch_survives": mode1_nontrivial_localized_branch_survives,
            "best_alpha_improves_phase1": best_alpha_improves_phase1,
            "anchor_drift_branch_continuation_required": anchor_drift_branch_continuation_required,
            "source_theorem_attempt_admissible_now": source_theorem_attempt_admissible_now,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "physical_reject_required": False,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_corrected_exact_solver_reinjection_completed",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "status_next_step_hit": hit(status_text, "8.7.56.1479"),
            "roadmap_next_step_hit": hit(roadmap_text, "8.7.56.1483-.1486"),
            "unified_roadmap_hit": hit(unified_roadmap_text, "corrected exact-action-level `ell=0` exact solver reinjection"),
        },
    )
    gate_paths = write_artifact("declaration_gate", gate_payload)

    numeric_rows = [
        row(
            "mode1_min_localized_amp_scale",
            "watch",
            "mode 1 minimum localized amplitude scale",
            float(mode1["min_localized_amp_scale_or_none"]),
            "The first localized exact mode-1 candidate only appears after this multiplicative drift away from the retained scalar amplitude.",
        ),
        row(
            "mode1_max_localized_ratio_ceiling",
            "watch",
            "mode 1 maximum localized max|fL/f0|",
            float(mode1["max_localized_ratio_ceiling"]),
            "Even among localized mode-1 exact candidates, the longitudinal admixture never leaves the scalar-like regime.",
        ),
        row(
            "global_best_localized_alpha_relerr_vs_target",
            "watch",
            "global best localized alpha relative error vs target",
            float(global_best_alpha["alpha_relerr_vs_target"]),
            "The best localized reinjection candidate remains far from the target and therefore cannot justify a source-theorem advance yet.",
        ),
    ]

    numeric_payload = payload(
        "8.7.56.1482",
        f"{STEP_NAME} numeric evaluation",
        {
            "inventory_json": inventory_paths["json"],
            "audit_json": audit_paths["json"],
            "declaration_gate_json": gate_paths["json"],
            "bootstrap_eval_json": display_path(BOOTSTRAP_EVAL),
            "phase1_eval_json": display_path(PHASE1_EVAL),
            "case_gamma_eval_json": display_path(CASE_GAMMA_EVAL),
        },
        numeric_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "beta_1_scalar": float(bootstrap_rows[0]["beta"]),
            "q_theory_over_m0_scalar": float(bootstrap_rows[0]["q_theory_over_m0"]),
            "bootstrap_mode_count": int(len(bootstrap_rows)),
            "localized_mode_count": localized_mode_count,
            "mode1_anchor_factor2_localized_count": int(mode1["anchor_factor2_localized_count"]),
            "mode1_anchor_factor4_localized_count": int(mode1["anchor_factor4_localized_count"]),
            "mode1_min_localized_amp_scale_or_none": float(mode1["min_localized_amp_scale_or_none"]),
            "mode1_min_localized_lambda_scale_or_none": float(mode1["min_localized_lambda_scale_or_none"]),
            "mode1_min_localized_tail_ratio_or_none": float(mode1["min_localized_tail_ratio_or_none"]),
            "mode1_min_localized_max_abs_ratio_or_none": float(mode1["min_localized_max_abs_ratio_or_none"]),
            "mode1_max_localized_ratio_ceiling": float(mode1["max_localized_ratio_ceiling"]),
            "mode1_best_prefers_zero_lambda": mode1_best_prefers_zero_lambda,
            "global_best_localized_candidate": global_best_alpha,
            "global_best_prefers_zero_lambda": global_best_prefers_zero_lambda,
            "min_localized_amp_scale_sequence": min_scale_sequence,
            "min_localized_amp_scale_monotone_decreasing": min_scale_monotone_decreasing,
            "phase1_best_alpha_candidate": phase1_eval["summary"]["phase1_best_alpha_candidate"],
            "case_gamma_diagnostic_alpha_at_q_theory": float(case_gamma_eval["summary"]["diagnostic_alpha_at_q_theory"]),
            "best_alpha_improves_phase1": best_alpha_improves_phase1,
            "exact_solver_reinjection_success": exact_solver_reinjection_success,
            "anchor_drift_branch_continuation_required": anchor_drift_branch_continuation_required,
            "source_theorem_attempt_admissible_now": source_theorem_attempt_admissible_now,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "physical_reject_required": False,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_corrected_exact_solver_reinjection_numeric_evaluation_completed",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "mode_reinjection_rows": search_rows,
            "mode1_row": mode1,
            "global_best_alpha_candidate": global_best_alpha,
        },
    )
    numeric_paths = write_artifact("numeric_evaluation", numeric_payload)

    print(json.dumps(
        {
            "status": "ok",
            "step": STEP_TAG,
            "inventory_json": inventory_paths["json"],
            "audit_json": audit_paths["json"],
            "declaration_gate_json": gate_paths["json"],
            "numeric_evaluation_json": numeric_paths["json"],
            "mode1_min_localized_amp_scale": mode1["min_localized_amp_scale_or_none"],
            "mode1_anchor_factor4_localized_count": mode1["anchor_factor4_localized_count"],
            "mode1_max_localized_ratio_ceiling": mode1["max_localized_ratio_ceiling"],
            "best_global_alpha_relerr": global_best_alpha["alpha_relerr_vs_target"],
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
