#!/usr/bin/env python3
"""Generate corrected exact-action-level ell=0 family-bootstrap artifacts for 8.7.56.1475-.1478.

This branch accepts the 8.7.56.1471-.1474 result:

- the public post-photon nontransverse free backbone is available
- the closed exact ell=0 operator is still unavailable
- the old generic family map is no longer admissible

The honest next step is therefore not a direct source-theorem attempt. It is a
bootstrap test: can the retained scalar discrete ladder seed a nonzero regular
ell=0 longitudinal family on the corrected free backbone strongly enough to
justify an exact-solver reinjection phase?
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


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

QBALL_DISCRETE = PUBLIC_OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"
OPERATOR_AUDIT = PUBLIC_OUT / "q_8_7_56_1471_1474_ell0_exact_operator_derivation_audit_metrics.json"
OPERATOR_GATE = PUBLIC_OUT / "q_8_7_56_1471_1474_ell0_exact_operator_derivation_declaration_gate_metrics.json"
PHASE1_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase1_exact_coupled_l0_solver_"
    "numeric_evaluation_metrics.json"
)
CASE_GAMMA_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_perturbative_fl_driven_ode_"
    "diagnostic_reopen_review_numeric_evaluation_metrics.json"
)

QBALL_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_qball_charge_mapping_branch.py"
NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
TRIAL3_FAMILY_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_spectrum_branch.py"

STEP_TAG = "8.7.56.1475-1478"
STEM = build_compact_artifact_stem(STEP_TAG, "ell0_family_bootstrap", prefix="q")
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor corrected exact-action-level ell=0 family-map bootstrap"

PRIOR_CLASS = "vector_qball_form_factor_exact_action_level_ell0_operator_derivation_partial_free_backbone_bootstrap_required"
BRANCH_CLASS = "vector_qball_form_factor_corrected_ell0_bootstrap_family_exists_exact_solver_reinjection_required"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_corrected_exact_action_level_ell0_exact_solver_reinjection"
NEXT_ROUTE = "8.7.56.1479"
ALPHA_TARGET = 1.0 / 137.035999084


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: fail immediately when a required path is missing.

def require(path: Path) -> None:
    """Fail immediately when a required path is missing."""
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


# Function: convert one absolute path to repo-relative display text when possible.

def display_path(path: Path) -> str:
    """Convert one absolute path to repo-relative display text when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: return the first matching source line for one substring pattern.

def hit(text: str, pattern: str) -> dict | None:
    """Return the first matching source line for one substring pattern."""
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


# Function: solve one diagnostic longitudinal bootstrap seed on the scalar ladder.

def solve_bootstrap_seed(
    qball,
    numerical,
    beta: float,
    amplitude: float,
) -> dict:
    """Solve one diagnostic longitudinal bootstrap seed on the retained scalar ladder."""
    radius, f0_values, f0_prime_values = qball.solve_full_profile(float(beta), float(amplitude))
    radius = np.asarray(radius, dtype=float)
    f0_values = np.asarray(f0_values, dtype=float)
    f0_prime_values = np.asarray(f0_prime_values, dtype=float)
    r0 = float(radius[0])
    kappa2 = max(1.0 - float(beta) * float(beta), 0.0)

    # Function: return the driven longitudinal bootstrap ODE right-hand side.
    def ode(current_radius: float, y: np.ndarray) -> list[float]:
        f_l, f_l_prime = [float(value) for value in y]
        safe_radius = max(float(current_radius), 1.0e-12)
        source = float(beta) * float(np.interp(safe_radius, radius, f0_prime_values))
        f_l_double_prime = (
            -(2.0 / safe_radius) * f_l_prime
            + (2.0 / (safe_radius * safe_radius)) * f_l
            + kappa2 * f_l
            + source
        )
        return [f_l_prime, f_l_double_prime]

    # Function: return the localization residual for one origin derivative guess.

    def residual(df_l0: float) -> float:
        solution = solve_ivp(
            ode,
            (r0, float(radius[-1])),
            [r0 * float(df_l0), float(df_l0)],
            t_eval=radius,
            max_step=0.03,
            rtol=1.0e-8,
            atol=1.0e-10,
        )
        if not solution.success:
            raise RuntimeError(f"bootstrap solve failed for beta={beta}: {solution.message}")

        return float(solution.y[0, -1])

    bracket = None
    for scale, count in ((0.5, 81), (5.0, 81), (25.0, 101)):
        guesses = np.linspace(-float(scale), float(scale), int(count))
        residuals = [residual(float(guess)) for guess in guesses]
        for left, right, res_left, res_right in zip(
            guesses[:-1],
            guesses[1:],
            residuals[:-1],
            residuals[1:],
        ):
            if not np.isfinite(res_left) or not np.isfinite(res_right):
                continue

            if abs(float(res_left)) < 1.0e-12:
                bracket = (float(left), float(left))
                break

            if float(res_left) * float(res_right) < 0.0:
                bracket = (float(left), float(right))
                break

        if bracket is not None:
            break

    if bracket is None:
        raise RuntimeError(f"unable to bracket bootstrap seed for beta={beta}")

    if abs(float(bracket[0]) - float(bracket[1])) < 1.0e-14:
        df_l0 = float(bracket[0])
    else:
        df_l0 = float(brentq(residual, float(bracket[0]), float(bracket[1])))

    solved = solve_ivp(
        ode,
        (r0, float(radius[-1])),
        [r0 * float(df_l0), float(df_l0)],
        t_eval=radius,
        max_step=0.03,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    if not solved.success:
        raise RuntimeError(f"final bootstrap solve failed for beta={beta}: {solved.message}")

    f_l_values = np.asarray(solved.y[0], dtype=float)
    max_abs_f0 = float(np.max(np.abs(f0_values)))
    max_abs_f_l = float(np.max(np.abs(f_l_values)))
    ratio = float(max_abs_f_l / max_abs_f0) if max_abs_f0 > 0.0 else math.inf
    q_ratio = float((1.0 - float(beta) * float(beta)) ** 0.25)
    rho_vector = f0_values * f0_values - f_l_values * f_l_values
    form_factor_at_q, proxy_norm = form_factor(radius, rho_vector, q_ratio)
    alpha_at_q = float((form_factor_at_q * form_factor_at_q) / (4.0 * math.pi))
    alpha_relerr = float(abs(alpha_at_q - ALPHA_TARGET) / ALPHA_TARGET)
    peak_index = int(np.argmax(np.abs(f_l_values)))
    tail_abs = float(abs(f_l_values[-1]))
    node_count = int(numerical.count_radial_nodes(np.asarray(f_l_values, dtype=float)))
    regular_localized = bool(tail_abs <= 1.0e-6)

    return {
        "beta": float(beta),
        "central_amplitude": float(amplitude),
        "df_l0": float(df_l0),
        "max_abs_f0": max_abs_f0,
        "max_abs_fL": max_abs_f_l,
        "max_abs_ratio": ratio,
        "peak_position": float(radius[peak_index]),
        "tail_abs": tail_abs,
        "node_count_fL": node_count,
        "regular_localized": regular_localized,
        "q_theory_over_m0": q_ratio,
        "F_at_q_theory": form_factor_at_q,
        "alpha_at_q_theory": alpha_at_q,
        "alpha_relerr_vs_target": alpha_relerr,
        "vector_proxy_norm": proxy_norm,
    }


# Function: test whether one float sequence is strictly increasing.

def strictly_increasing(values: list[float], tol: float = 1.0e-12) -> bool:
    """Test whether one float sequence is strictly increasing."""
    return all(float(right) > float(left) + float(tol) for left, right in zip(values[:-1], values[1:]))


# Function: test whether one float sequence is strictly decreasing.

def strictly_decreasing(values: list[float], tol: float = 1.0e-12) -> bool:
    """Test whether one float sequence is strictly decreasing."""
    return all(float(right) < float(left) - float(tol) for left, right in zip(values[:-1], values[1:]))


# Function: execute the corrected ell=0 family-bootstrap branch.

def main() -> None:
    """Execute the corrected ell=0 family-bootstrap branch."""
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
        QBALL_DISCRETE,
        OPERATOR_AUDIT,
        OPERATOR_GATE,
        PHASE1_EVAL,
        CASE_GAMMA_EVAL,
        QBALL_SOLVER,
        NUMERICAL_BRANCH,
        TRIAL3_FAMILY_BRANCH,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part1_text = read_text(PART1)
    part5_text = read_text(PART5)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    solver_fix_text = read_text(SOLVER_FIX)
    next_steps_text = read_text(NEXT_STEPS)
    family_script_text = read_text(TRIAL3_FAMILY_BRANCH)

    discrete = read_json(QBALL_DISCRETE)
    operator_audit = read_json(OPERATOR_AUDIT)
    operator_gate = read_json(OPERATOR_GATE)
    phase1_eval = read_json(PHASE1_EVAL)
    case_gamma_eval = read_json(CASE_GAMMA_EVAL)

    qball = load_module(QBALL_SOLVER, "qball_charge_mapping_branch")
    numerical = load_module(NUMERICAL_BRANCH, "vector_qball_numerical_branch")

    mode_rows = sorted(
        discrete["evidence"]["discrete_mass_mode_rows"],
        key=lambda entry: int(entry["mode_index"]),
    )
    bootstrap_rows = []
    for mode in mode_rows:
        solved = solve_bootstrap_seed(
            qball,
            numerical,
            float(mode["beta_n"]),
            float(mode["central_amplitude"]),
        )
        solved["mode_index"] = int(mode["mode_index"])
        solved["charge_proxy"] = float(mode["charge_proxy"])
        solved["energy_proxy"] = float(mode["energy_proxy"])
        bootstrap_rows.append(solved)

    df_l0_values = [float(entry["df_l0"]) for entry in bootstrap_rows]
    peak_positions = [float(entry["peak_position"]) for entry in bootstrap_rows]
    ratios = [float(entry["max_abs_ratio"]) for entry in bootstrap_rows]
    norms = [float(entry["vector_proxy_norm"]) for entry in bootstrap_rows]
    alpha_errors = [float(entry["alpha_relerr_vs_target"]) for entry in bootstrap_rows]
    regular_count = int(sum(1 for entry in bootstrap_rows if bool(entry["regular_localized"])))
    family_available = bool(regular_count == len(bootstrap_rows))
    monotone_df_l0 = strictly_increasing(df_l0_values)
    monotone_peak = strictly_decreasing(peak_positions)
    low_modes_nonperturbative = bool(
        len(ratios) >= 2 and all(float(value) > 1.0 for value in ratios[:2])
    )
    perturbative_entry_available = bool(any(float(value) < 1.0 for value in ratios))
    proxy_norm_sign_change = bool(min(norms) < 0.0 < max(norms))
    sign_pattern = [int(math.copysign(1.0, value)) if value != 0.0 else 0 for value in norms]
    best_alpha_index = int(np.argmin(alpha_errors))
    best_alpha_mode = bootstrap_rows[best_alpha_index]
    direct_source_attempt_admissible = False
    corrected_exact_solver_reinjection_required = bool(
        family_available and (low_modes_nonperturbative or proxy_norm_sign_change)
    )

    inventory_rows = [
        row(
            "inventory_ready",
            "pass",
            "corrected ell0 family bootstrap inventory ready",
            1.0,
            "The bootstrap inventory is ready once the scalar ladder, operator audit, Phase 1 rows, and retained solver notes are assembled in one compact pack.",
        ),
        row(
            "scalar_ladder_available",
            "pass",
            "retained scalar ladder available",
            float(len(mode_rows)),
            "The retained discrete mass spectrum provides the scalar baseline ladder used for corrected bootstrap seeding.",
        ),
        row(
            "operator_gap_available",
            "pass",
            "operator-gap audit available",
            1.0,
            "The corrected bootstrap only makes sense after the prior branch fixed that the exact closed ell=0 operator is still unavailable.",
        ),
        row(
            "phase1_case_gamma_rows_available",
            "pass",
            "Phase 1 and Case gamma rows available",
            1.0,
            "Both the exact pilot baseline and the perturbative diagnostic remain available for comparison against the corrected bootstrap family.",
        ),
        row(
            "solver_fix_note_available",
            "pass",
            "solver-fix note available",
            1.0,
            "The retained solver-fix note defines why the next step must test a longitudinal bootstrap rather than continue text-only handoff work.",
        ),
    ]

    inventory_payload = payload(
        "8.7.56.1475",
        f"{STEP_NAME} inventory",
        {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "current_problem_note": display_path(CURRENT_PROBLEM),
            "current_status_note": display_path(CURRENT_STATUS),
            "unified_roadmap_note": display_path(UNIFIED_ROADMAP),
            "part1": display_path(PART1),
            "part5": display_path(PART5),
            "solver_fix_note": display_path(SOLVER_FIX),
            "next_steps_note": display_path(NEXT_STEPS),
            "qball_discrete_json": display_path(QBALL_DISCRETE),
            "operator_audit_json": display_path(OPERATOR_AUDIT),
            "operator_gate_json": display_path(OPERATOR_GATE),
            "phase1_eval_json": display_path(PHASE1_EVAL),
            "case_gamma_eval_json": display_path(CASE_GAMMA_EVAL),
            "qball_solver": display_path(QBALL_SOLVER),
            "numerical_branch": display_path(NUMERICAL_BRANCH),
            "trial3_family_branch": display_path(TRIAL3_FAMILY_BRANCH),
        },
        inventory_rows,
        {
            "trial2_numeric_alpha_problem_classification": PRIOR_CLASS,
            "retained_scalar_mode_count": int(len(mode_rows)),
            "phase1_problem_classification": phase1_eval["summary"]["trial2_numeric_alpha_problem_classification"],
            "case_gamma_problem_classification": case_gamma_eval["summary"]["trial2_numeric_alpha_problem_classification"],
            "operator_problem_classification": operator_gate["summary"]["trial2_numeric_alpha_problem_classification"],
            "bootstrap_inventory_ready": True,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_corrected_ell0_bootstrap_inventory_completed",
            "branch_completed": False,
            "next_required_artifacts": [f"{STEM}_audit_metrics.json"],
        },
        {
            "status_current_step_hit": hit(status_text, "8.7.56.1475-.1478"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1475-.1478"),
            "solver_fix_computation_hit": hit(solver_fix_text, "text search / ordering formalization は即座に停止。"),
            "next_steps_step_b_hit": hit(next_steps_text, "Step B. linearized longitudinal equationを exact に導く"),
        },
    )
    inventory_paths = write_artifact("inventory", inventory_payload)

    audit_rows = [
        row(
            "bootstrap_seed_family_available",
            "pass" if family_available else "reject",
            "bootstrap seed family available",
            float(1.0 if family_available else 0.0),
            "The corrected bootstrap succeeds only if the retained scalar ladder seeds a regular localized longitudinal solution across the whole discrete family.",
        ),
        row(
            "monotone_df_l0_increasing",
            "pass" if monotone_df_l0 else "watch",
            "origin derivative seed increases along the scalar ladder",
            float(1.0 if monotone_df_l0 else 0.0),
            "A monotone df_L(0) ladder supports a continuous family picture rather than isolated accidental roots.",
        ),
        row(
            "monotone_peak_position_decreasing",
            "pass" if monotone_peak else "watch",
            "longitudinal peak position decreases along the scalar ladder",
            float(1.0 if monotone_peak else 0.0),
            "A monotone inward drift of the f_L peak indicates a coherent family continuation on the corrected backbone.",
        ),
        row(
            "low_modes_nonperturbative",
            "watch" if low_modes_nonperturbative else "reject",
            "low modes are already nonperturbative",
            float(1.0 if low_modes_nonperturbative else 0.0),
            "If the first scalar modes exceed |f_L/f_0| > 1, perturbative rescue is no longer an admissible continuation strategy.",
        ),
        row(
            "higher_mode_perturbative_entry_available",
            "pass" if perturbative_entry_available else "watch",
            "higher-mode perturbative entry remains available",
            float(1.0 if perturbative_entry_available else 0.0),
            "A later perturbative entry supports family existence even when the physically relevant low modes are already nonperturbative.",
        ),
        row(
            "vector_proxy_norm_sign_change_present",
            "watch" if proxy_norm_sign_change else "reject",
            "vector proxy norm sign change present",
            float(1.0 if proxy_norm_sign_change else 0.0),
            "A norm sign flip across the ladder shows that proxy observables change topology and should not be promoted before an exact solver reinjection phase.",
        ),
        row(
            "corrected_exact_solver_reinjection_required",
            "pass" if corrected_exact_solver_reinjection_required else "reject",
            "corrected exact solver reinjection required",
            float(1.0 if corrected_exact_solver_reinjection_required else 0.0),
            "Once the family exists but low modes are nonperturbative, the next honest branch is an exact solver reinjection on the corrected backbone.",
        ),
        row(
            "direct_source_theorem_attempt_admissible_now",
            "reject" if not direct_source_attempt_admissible else "pass",
            "direct source theorem attempt admissible now",
            float(1.0 if direct_source_attempt_admissible else 0.0),
            "Source-theorem work is premature when the corrected bootstrap family is present but the physically relevant low modes still require exact reinjection.",
        ),
    ]

    audit_payload = payload(
        "8.7.56.1476",
        f"{STEP_NAME} audit",
        {
            "inventory_json": inventory_paths["json"],
            "qball_discrete_json": display_path(QBALL_DISCRETE),
            "operator_audit_json": display_path(OPERATOR_AUDIT),
            "phase1_eval_json": display_path(PHASE1_EVAL),
            "case_gamma_eval_json": display_path(CASE_GAMMA_EVAL),
            "qball_solver": display_path(QBALL_SOLVER),
            "numerical_branch": display_path(NUMERICAL_BRANCH),
            "trial3_family_branch": display_path(TRIAL3_FAMILY_BRANCH),
        },
        audit_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "bootstrap_mode_count": int(len(bootstrap_rows)),
            "bootstrap_regular_mode_count": int(regular_count),
            "monotone_df_l0_increasing": monotone_df_l0,
            "monotone_peak_position_decreasing": monotone_peak,
            "low_modes_nonperturbative": low_modes_nonperturbative,
            "higher_mode_perturbative_entry_available": perturbative_entry_available,
            "vector_proxy_norm_sign_change_present": proxy_norm_sign_change,
            "corrected_exact_solver_reinjection_required": corrected_exact_solver_reinjection_required,
            "direct_source_theorem_attempt_admissible_now": direct_source_attempt_admissible,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_corrected_ell0_bootstrap_family_audited",
            "branch_completed": False,
            "next_required_artifacts": [f"{STEM}_declaration_gate_metrics.json"],
        },
        {
            "operator_gap_summary": operator_audit["summary"],
            "phase1_summary": phase1_eval["summary"],
            "case_gamma_summary": case_gamma_eval["summary"],
            "trial3_collapse_hit": hit(
                family_script_text,
                "k_proxy = math.sqrt(max(float(ell * (ell + 1)), 0.0)) / rr",
            ),
            "solver_fix_driven_ode_hit": hit(
                solver_fix_text,
                "f_L'' + \\frac{2}{r}f_L' - \\frac{2}{r^2}f_L - \\kappa^2 f_L = \\beta\\,y_0'(x)",
            ),
            "bootstrap_mode_rows": bootstrap_rows,
            "proxy_norm_sign_pattern": sign_pattern,
        },
    )
    audit_paths = write_artifact("audit", audit_payload)

    gate_rows = [
        row(
            "bootstrap_family_exists_selected",
            "pass" if family_available else "reject",
            "bootstrap family exists disposition selected",
            float(1.0 if family_available else 0.0),
            "The corrected bootstrap branch succeeds only if the scalar ladder seeds a regular longitudinal family rather than isolated accidental solutions.",
        ),
        row(
            "corrected_exact_solver_reinjection_required",
            "pass" if corrected_exact_solver_reinjection_required else "reject",
            "corrected exact solver reinjection required",
            float(1.0 if corrected_exact_solver_reinjection_required else 0.0),
            "Nonperturbative low modes force the next mainline toward exact solver reinjection on the corrected backbone.",
        ),
        row(
            "direct_source_theorem_attempt_deferred",
            "pass" if not direct_source_attempt_admissible else "watch",
            "direct source theorem attempt deferred",
            float(1.0 if not direct_source_attempt_admissible else 0.0),
            "The source theorem stays downstream of the corrected exact solver reinjection because the family audit still depends on proxy observables.",
        ),
    ]

    gate_payload = payload(
        "8.7.56.1477",
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
            "bootstrap_seed_family_available": family_available,
            "low_modes_nonperturbative": low_modes_nonperturbative,
            "vector_proxy_norm_sign_change_present": proxy_norm_sign_change,
            "corrected_exact_solver_reinjection_required": corrected_exact_solver_reinjection_required,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "physical_reject_required": False,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_corrected_ell0_bootstrap_family_completed",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1475-.1478"),
            "status_next_step_hit": hit(status_text, "8.7.56.1475"),
            "unified_roadmap_bootstrap_hit": hit(
                unified_roadmap_text,
                "corrected exact-action-level `ell=0` family-map bootstrap",
            ),
        },
    )
    gate_paths = write_artifact("declaration_gate", gate_payload)

    numeric_rows = [
        row(
            "mode1_max_abs_ratio",
            "watch" if float(bootstrap_rows[0]["max_abs_ratio"]) > 1.0 else "pass",
            "mode 1 max |fL/f0|",
            float(bootstrap_rows[0]["max_abs_ratio"]),
            "The electron-anchor mode is already nonperturbative if this exceeds unity.",
        ),
        row(
            "mode5_max_abs_ratio",
            "pass" if float(bootstrap_rows[-1]["max_abs_ratio"]) < 1.0 else "watch",
            "mode 5 max |fL/f0|",
            float(bootstrap_rows[-1]["max_abs_ratio"]),
            "Higher ladder modes demonstrate whether a perturbative entry survives somewhere on the corrected family.",
        ),
        row(
            "best_proxy_alpha_relerr_vs_target",
            "watch",
            "best bootstrap proxy alpha relative error vs target",
            float(best_alpha_mode["alpha_relerr_vs_target"]),
            "This is still a proxy-only diagnostic and therefore cannot close the route, but it measures how far the family gets before exact reinjection.",
        ),
    ]

    numeric_payload = payload(
        "8.7.56.1478",
        f"{STEP_NAME} numeric evaluation",
        {
            "inventory_json": inventory_paths["json"],
            "audit_json": audit_paths["json"],
            "declaration_gate_json": gate_paths["json"],
            "qball_discrete_json": display_path(QBALL_DISCRETE),
            "phase1_eval_json": display_path(PHASE1_EVAL),
            "case_gamma_eval_json": display_path(CASE_GAMMA_EVAL),
        },
        numeric_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "beta_1_scalar": float(mode_rows[0]["beta_n"]),
            "phase1_best_vector_F_at_q_theory": float(phase1_eval["summary"]["phase1_best_alpha_candidate"]["F_at_q_theory"]),
            "phase1_best_vector_alpha_at_q_theory": float(phase1_eval["summary"]["phase1_best_alpha_candidate"]["alpha_at_q_theory"]),
            "case_gamma_F_at_q_theory": float(case_gamma_eval["summary"]["diagnostic_F_at_q_theory"]),
            "case_gamma_alpha_at_q_theory": float(case_gamma_eval["summary"]["diagnostic_alpha_at_q_theory"]),
            "bootstrap_mode_count": int(len(bootstrap_rows)),
            "bootstrap_regular_mode_count": int(regular_count),
            "bootstrap_df_l0_values": df_l0_values,
            "bootstrap_peak_positions": peak_positions,
            "bootstrap_max_abs_ratio_values": ratios,
            "bootstrap_vector_proxy_norm_values": norms,
            "bootstrap_best_proxy_alpha_mode_index": int(best_alpha_mode["mode_index"]),
            "bootstrap_best_proxy_F_at_q_theory": float(best_alpha_mode["F_at_q_theory"]),
            "bootstrap_best_proxy_alpha_at_q_theory": float(best_alpha_mode["alpha_at_q_theory"]),
            "bootstrap_best_proxy_alpha_relerr_vs_target": float(best_alpha_mode["alpha_relerr_vs_target"]),
            "monotone_df_l0_increasing": monotone_df_l0,
            "monotone_peak_position_decreasing": monotone_peak,
            "low_modes_nonperturbative": low_modes_nonperturbative,
            "vector_proxy_norm_sign_change_present": proxy_norm_sign_change,
            "corrected_exact_solver_reinjection_required": corrected_exact_solver_reinjection_required,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "physical_reject_required": False,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_corrected_ell0_bootstrap_family_numeric_evaluation_completed",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "bootstrap_mode_rows": bootstrap_rows,
            "best_proxy_alpha_mode": best_alpha_mode,
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
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "bootstrap_regular_mode_count": regular_count,
            "low_modes_nonperturbative": low_modes_nonperturbative,
            "vector_proxy_norm_sign_change_present": proxy_norm_sign_change,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
