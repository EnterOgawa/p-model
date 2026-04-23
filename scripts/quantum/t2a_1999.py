#!/usr/bin/env python3
"""Generate 8.7.56.1999-.2002 boundary phase-carrier higher-q extension artifacts."""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
import scripts.quantum.t2a_1963 as asymp_base
import scripts.quantum.t2a_1975 as local_jet_base
import scripts.quantum.t2a_1983 as ext_base
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
LONG_ROADMAP = ROOT / "doc" / "quantum" / "55_trial2_numeric_alpha_vector_qball_long_horizon_roadmap.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

QBALL_BRANCH_REFRESH = PUBLIC_OUT / "mass_origin_qball_charge_mapping_branch_refresh_metrics.json"
PRIOR_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1995_1998_boundary_phase_carrier_gate_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1999-2002"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor boundary phase-carrier "
    "higher-q extension audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "boundary_phase_curvature_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_phase_carrier_window_40_to_120_retained_"
    "higher_q_generalization_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_phase_curvature_window_120_to_200_"
    "large_coefficient_partial_retain_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_boundary_phase_curvature_"
    "decision_gate_registry"
)
NEXT_ROUTE = "8.7.56.2003"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_boundary_phase_curvature_"
    "higher_q_extension_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2007"
RETAINED_Q_MIN = 120.0
RETAINED_Q_MAX = 200.0
HOLDOUT_Q_MAX = 260.0


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: JSON/CSV artifact を書き出す。

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one rows CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "status", "metric", "value", "note"],
        )
        writer.writeheader()
        writer.writerows(data["rows"])

    return {
        "json": sign_base.display_path(paths["json"]),
        "csv": sign_base.display_path(paths["csv"]),
    }


# 関数: custom phase-carrier zero equation を返す。

def shifted_zero_equation(
    q_ratio: float,
    h0: float,
    h1: float,
    h2: float,
    r_box: float,
    phi0: float,
    phi_inv: float,
    delta_r: float,
    phi_inv2: float,
) -> float:
    """Return the local-jet zero equation with a phase carrier."""
    phase = (
        q_ratio * r_box
        + phi0
        + (phi_inv / q_ratio)
        + (delta_r * q_ratio)
        + (phi_inv2 / (q_ratio * q_ratio))
    )
    return (
        (-h0 * q_ratio * q_ratio + h2) * math.cos(phase)
        + (h1 * q_ratio) * math.sin(phase)
    )


# 関数: one q interval で custom carrier の root lattice を返す。

def find_custom_zeros(
    q_min: float,
    q_max: float,
    equation,
    r_box: float,
) -> np.ndarray:
    """Locate all custom zero roots on one q interval."""
    roots: list[float] = []
    max_mode = int(math.ceil(q_max * r_box / math.pi)) + 4
    for mode_index in range(max_mode):
        intervals = [
            (
                (mode_index * math.pi / r_box) + 1.0e-6,
                ((mode_index + 0.5) * math.pi / r_box) - 1.0e-6,
            ),
            (
                ((mode_index + 0.5) * math.pi / r_box) + 1.0e-6,
                ((mode_index + 1.0) * math.pi / r_box) - 1.0e-6,
            ),
        ]
        for left, right in intervals:
            if right <= q_min or left >= q_max:
                continue

            f_left = equation(left)
            f_right = equation(right)
            if not (math.isfinite(f_left) and math.isfinite(f_right)):
                continue

            if f_left * f_right >= 0.0:
                continue

            root = float(brentq(equation, left, right))
            if not roots or abs(root - roots[-1]) > 1.0e-6:
                roots.append(root)

    return np.array(roots, dtype=float)


# 関数: higher-order phase-curvature carrier を fit する。

def fit_phase_curvature_carrier(
    exact_roots: np.ndarray,
    predicted_roots: np.ndarray,
    r_box: float,
) -> tuple[float, float, float, float]:
    """Fit a second-shot boundary phase-curvature carrier on one q window."""
    q_values = []
    delta_q = []
    for exact_value in exact_roots:
        nearest_index = int(np.argmin(np.abs(predicted_roots - exact_value)))
        q_values.append(float(exact_value))
        delta_q.append(float(exact_value - predicted_roots[nearest_index]))

    q_array = np.array(q_values, dtype=float)
    delta_array = np.array(delta_q, dtype=float)
    design = np.column_stack(
        [
            np.ones_like(q_array),
            1.0 / q_array,
            q_array,
            1.0 / (q_array * q_array),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design, delta_array, rcond=None)
    return (
        float(-r_box * coefficients[0]),
        float(-r_box * coefficients[1]),
        float(-r_box * coefficients[2]),
        float(-r_box * coefficients[3]),
    )


# 関数: one q window の rule metrics を返す。

def evaluate_window(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    exact_roots_all: np.ndarray,
    predicted_roots: np.ndarray,
    q_min: float,
    q_max: float,
) -> dict[str, float]:
    """Return zero-lattice and sign diagnostics on one q window."""
    exact_window = exact_roots_all[(exact_roots_all >= q_min) & (exact_roots_all <= q_max)]
    predicted_window = predicted_roots[
        (predicted_roots >= q_min) & (predicted_roots <= q_max)
    ]
    root_stats = ext_base.nearest_neighbor_stats(exact_window, predicted_window)
    q_scan = np.linspace(q_min, q_max, 120001)
    form_factor_scan = np.array(
        [sign_base.form_factor(radius, weight, norm, float(value)) for value in q_scan],
        dtype=float,
    )
    absolute_scan = np.abs(form_factor_scan)
    prior_zero_count = int(np.count_nonzero(exact_roots_all < q_min))
    reconstruction = local_jet_base.evaluate_rule_window(
        q_scan,
        form_factor_scan,
        absolute_scan,
        prior_zero_count,
        predicted_window,
    )
    return {
        "exact_zero_count": float(exact_window.size),
        "predicted_zero_count": float(predicted_window.size),
        **root_stats,
        "signed_reconstruction_max_abs_error": reconstruction["max_abs_error"],
        "signed_reconstruction_mean_abs_error": reconstruction["mean_abs_error"],
        "sign_mismatch_fraction": reconstruction["sign_mismatch_fraction"],
    }


# 関数: audit で使う公式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the higher-q extension audit."""
    return {
        "retained_phase_carrier": "phi_3(q)=phi0 + phi_-1/q + dR q",
        "phase_curvature_carrier": "phi_4(q)=phi0 + phi_-1/q + dR q + phi_-2/q^2",
        "closeout_read": "first test the retained 3-term carrier on 120<=q/m0<=200 and 200<=q/m0<=260, then test whether a 4-term phase-curvature carrier rescues the second finite window without becoming canonical",
    }


# 関数: `.1999-.2002` を実行する。

def main() -> None:
    """Execute the boundary phase-carrier higher-q extension audit."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LONG_ROADMAP,
        PART5,
        QBALL_BRANCH_REFRESH,
        PRIOR_GATE,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    inventory_ready = bool(prior_summary["higher_q_phase_carrier_generalization_admissible_now"])

    qball_branch_refresh = sign_base.read_json(QBALL_BRANCH_REFRESH)
    scalar_ground_state = sign_base.extract_scalar_ground_state(qball_branch_refresh)
    qball_module = sign_base.load_qball_module()
    radius, field, _field_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    weight = (field**2) * (radius**2)
    norm = float(np.trapezoid(weight, radius))
    r_box = float(radius[-1])
    h0, h1, h2 = local_jet_base.boundary_local_jet(radius, field)
    exact_roots_all = asymp_base.find_signed_zeros_interval(
        radius,
        weight,
        norm,
        HOLDOUT_Q_MAX,
    )
    baseline_roots = local_jet_base.find_local_jet_zeros(
        RETAINED_Q_MIN,
        HOLDOUT_Q_MAX,
        h0,
        h1,
        h2,
        r_box,
    )

    retained_phi0 = float(prior_summary["phase_carrier_phi0"])
    retained_phi_inv = float(prior_summary["phase_carrier_phi_inv"])
    retained_delta_r = float(prior_summary["phase_carrier_delta_r"])
    retained_carrier_roots = find_custom_zeros(
        RETAINED_Q_MIN,
        HOLDOUT_Q_MAX,
        lambda q_value: shifted_zero_equation(
            q_value,
            h0,
            h1,
            h2,
            r_box,
            retained_phi0,
            retained_phi_inv,
            retained_delta_r,
            0.0,
        ),
        r_box,
    )

    fit_exact_roots = exact_roots_all[
        (exact_roots_all >= RETAINED_Q_MIN) & (exact_roots_all <= RETAINED_Q_MAX)
    ]
    phi0_curv, phi_inv_curv, delta_r_curv, phi_inv2_curv = fit_phase_curvature_carrier(
        fit_exact_roots,
        baseline_roots,
        r_box,
    )
    curvature_roots = find_custom_zeros(
        RETAINED_Q_MIN,
        HOLDOUT_Q_MAX,
        lambda q_value: shifted_zero_equation(
            q_value,
            h0,
            h1,
            h2,
            r_box,
            phi0_curv,
            phi_inv_curv,
            delta_r_curv,
            phi_inv2_curv,
        ),
        r_box,
    )

    retained_fit = evaluate_window(
        radius,
        weight,
        norm,
        exact_roots_all,
        retained_carrier_roots,
        RETAINED_Q_MIN,
        RETAINED_Q_MAX,
    )
    retained_holdout = evaluate_window(
        radius,
        weight,
        norm,
        exact_roots_all,
        retained_carrier_roots,
        RETAINED_Q_MAX,
        HOLDOUT_Q_MAX,
    )
    curvature_fit = evaluate_window(
        radius,
        weight,
        norm,
        exact_roots_all,
        curvature_roots,
        RETAINED_Q_MIN,
        RETAINED_Q_MAX,
    )
    curvature_holdout = evaluate_window(
        radius,
        weight,
        norm,
        exact_roots_all,
        curvature_roots,
        RETAINED_Q_MAX,
        HOLDOUT_Q_MAX,
    )

    curvature_coeff_l2 = float(
        np.linalg.norm(
            np.array(
                [phi0_curv, phi_inv_curv, delta_r_curv, phi_inv2_curv],
                dtype=float,
            )
        )
    )
    curvature_coeff_linf = float(
        np.max(
            np.abs(
                np.array(
                    [phi0_curv, phi_inv_curv, delta_r_curv, phi_inv2_curv],
                    dtype=float,
                )
            )
        )
    )

    retained_three_term_higher_q_supported = bool(
        (retained_fit["sign_mismatch_fraction"] <= 0.05)
        and (retained_holdout["sign_mismatch_fraction"] <= 0.05)
    )
    phase_curvature_window_supported = bool(
        (curvature_fit["sign_mismatch_fraction"] <= 0.02)
        and (curvature_fit["exact_to_pred_max_abs_error"] <= 0.01)
    )
    phase_curvature_noncanonical_large_coefficients = bool(curvature_coeff_linf >= 1000.0)
    phase_curvature_higher_q_holdout_failed = bool(
        curvature_holdout["sign_mismatch_fraction"] >= 0.1
    )
    physical_reject_required = False

    rows = [
        sign_base.row("inventory_ready", "pass" if inventory_ready else "reject", "phase-carrier higher-q extension inventory ready", sign_base.truth(inventory_ready), "The higher-q audit starts only after the finite-window boundary phase carrier has been formally retained by the prior decision gate."),
        sign_base.row("retained_three_term_fit_sign_mismatch_fraction", "watch", "retained 3-term carrier sign mismatch fraction on 120<=q/m0<=200", retained_fit["sign_mismatch_fraction"], "This tests whether the retained finite-window carrier already survives the next later asymptotic window without any further theory update."),
        sign_base.row("retained_three_term_holdout_sign_mismatch_fraction", "watch", "retained 3-term carrier sign mismatch fraction on 200<=q/m0<=260", retained_holdout["sign_mismatch_fraction"], "This determines whether the retained 3-term carrier is merely finite-window or already asymptotically stable."),
        sign_base.row("retained_three_term_higher_q_supported", "reject" if not retained_three_term_higher_q_supported else "pass", "retained 3-term carrier higher-q supported", sign_base.truth(retained_three_term_higher_q_supported), "The old 3-term carrier is only retained as the mainline if it survives both the next window and the later holdout."),
        sign_base.row("phase_curvature_fit_sign_mismatch_fraction", "pass" if phase_curvature_window_supported else "watch", "4-term phase-curvature carrier sign mismatch fraction on 120<=q/m0<=200", curvature_fit["sign_mismatch_fraction"], "The new higher-order carrier is only worth promoting if it materially rescues the second finite asymptotic window that defeats the retained 3-term carrier."),
        sign_base.row("phase_curvature_fit_root_nn_max_abs_error", "watch", "4-term phase-curvature carrier max nearest-neighbor root error on 120<=q/m0<=200", curvature_fit["exact_to_pred_max_abs_error"], "The root lattice must stay locked, not just the raw sign pattern."),
        sign_base.row("phase_curvature_holdout_sign_mismatch_fraction", "watch", "4-term phase-curvature carrier sign mismatch fraction on 200<=q/m0<=260", curvature_holdout["sign_mismatch_fraction"], "The later holdout decides whether the new higher-order carrier is genuinely more asymptotic or still only a finite-window continuation."),
        sign_base.row("phase_curvature_coeff_linf", "watch", "4-term phase-curvature carrier coefficient infinity norm", curvature_coeff_linf, "Large coefficient size is a direct warning that the new carrier may be noncanonical even if it numerically rescues one finite window."),
        sign_base.row("phase_curvature_window_supported", "pass" if phase_curvature_window_supported else "reject", "4-term phase-curvature finite window supported", sign_base.truth(phase_curvature_window_supported), "The second-shot carrier only earns a partial retain if it materially improves 120<=q/m0<=200."),
        sign_base.row("phase_curvature_noncanonical_large_coefficients", "watch" if phase_curvature_noncanonical_large_coefficients else "pass", "4-term phase-curvature carrier has noncanonical large coefficients", sign_base.truth(phase_curvature_noncanonical_large_coefficients), "A large coefficient norm means the new carrier should stay at most Gate B partial even if its finite-window fit is good."),
        sign_base.row("phase_curvature_higher_q_holdout_failed", "watch" if phase_curvature_higher_q_holdout_failed else "pass", "4-term phase-curvature carrier higher-q holdout failed", sign_base.truth(phase_curvature_higher_q_holdout_failed), "A later holdout failure means the new carrier is not yet an asymptotic theorem, only a second finite-window continuation."),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_window_lower_over_m0": RETAINED_Q_MIN,
        "retained_window_upper_over_m0": RETAINED_Q_MAX,
        "holdout_window_upper_over_m0": HOLDOUT_Q_MAX,
        "retained_phase_carrier_phi0": retained_phi0,
        "retained_phase_carrier_phi_inv": retained_phi_inv,
        "retained_phase_carrier_delta_r": retained_delta_r,
        "retained_three_term_fit_sign_mismatch_fraction": retained_fit["sign_mismatch_fraction"],
        "retained_three_term_holdout_sign_mismatch_fraction": retained_holdout["sign_mismatch_fraction"],
        "retained_three_term_higher_q_supported": retained_three_term_higher_q_supported,
        "phase_curvature_phi0": phi0_curv,
        "phase_curvature_phi_inv": phi_inv_curv,
        "phase_curvature_delta_r": delta_r_curv,
        "phase_curvature_phi_inv2": phi_inv2_curv,
        "phase_curvature_coeff_l2": curvature_coeff_l2,
        "phase_curvature_coeff_linf": curvature_coeff_linf,
        "phase_curvature_fit_root_nn_max_abs_error": curvature_fit["exact_to_pred_max_abs_error"],
        "phase_curvature_fit_root_nn_mean_abs_error": curvature_fit["exact_to_pred_mean_abs_error"],
        "phase_curvature_fit_sign_mismatch_fraction": curvature_fit["sign_mismatch_fraction"],
        "phase_curvature_fit_signed_reconstruction_max_abs_error": curvature_fit["signed_reconstruction_max_abs_error"],
        "phase_curvature_holdout_root_nn_max_abs_error": curvature_holdout["exact_to_pred_max_abs_error"],
        "phase_curvature_holdout_root_nn_mean_abs_error": curvature_holdout["exact_to_pred_mean_abs_error"],
        "phase_curvature_holdout_sign_mismatch_fraction": curvature_holdout["sign_mismatch_fraction"],
        "phase_curvature_holdout_signed_reconstruction_max_abs_error": curvature_holdout["signed_reconstruction_max_abs_error"],
        "phase_curvature_window_supported": phase_curvature_window_supported,
        "phase_curvature_noncanonical_large_coefficients": phase_curvature_noncanonical_large_coefficients,
        "phase_curvature_higher_q_holdout_failed": phase_curvature_higher_q_holdout_failed,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2001",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "ai_context": sign_base.display_path(AI_CONTEXT),
                "work_history_recent": sign_base.display_path(WORK_HISTORY_RECENT),
                "current_problem": sign_base.display_path(CURRENT_PROBLEM),
                "current_status": sign_base.display_path(CURRENT_STATUS),
                "unified_roadmap": sign_base.display_path(UNIFIED_ROADMAP),
                "long_roadmap": sign_base.display_path(LONG_ROADMAP),
                "part5": sign_base.display_path(PART5),
                "qball_branch_refresh": sign_base.display_path(QBALL_BRANCH_REFRESH),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
            },
            "constants": {
                "retained_window_over_m0": [RETAINED_Q_MIN, RETAINED_Q_MAX],
                "holdout_window_upper_over_m0": HOLDOUT_Q_MAX,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_boundary_phase_carrier_higher_q_extension_audited",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.1999"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.1999-.2002"),
                "current_problem_hit": sign_base.hit(current_problem_text, "higher_q_phase_carrier_generalization_admissible_now"),
                "current_status_hit": sign_base.hit(current_status_text, "boundary phase-carrier higher-q extension audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".1999-.2002"),
                "long_roadmap_hit": sign_base.hit(long_text, "boundary phase-carrier higher-q extension audit"),
                "part5_hit": sign_base.hit(part5_text, ".1991-.1998"),
            },
        },
    )

    route_rows = [
        sign_base.row("retained_three_term_higher_q_supported", "reject" if not retained_three_term_higher_q_supported else "pass", "retained 3-term carrier higher-q supported", sign_base.truth(retained_three_term_higher_q_supported), "The old finite-window carrier does not itself survive the later asymptotic windows and therefore cannot remain the sole mainline theorem."),
        sign_base.row("phase_curvature_window_supported", "pass" if phase_curvature_window_supported else "reject", "4-term phase-curvature finite window supported", sign_base.truth(phase_curvature_window_supported), "The next route is justified only if the richer carrier opens a real second finite-window surface."),
        sign_base.row("phase_curvature_higher_q_holdout_failed", "watch" if phase_curvature_higher_q_holdout_failed else "pass", "4-term phase-curvature higher-q holdout failed", sign_base.truth(phase_curvature_higher_q_holdout_failed), "A later holdout failure means the next official question is generalization, not same-level retry of the old 3-term carrier."),
        sign_base.row("next_route_fixed", "pass", "next route fixed", 1.0, "The next official branch is the boundary phase-curvature decision gate / registry."),
    ]

    route_payload = sign_base.payload(
        "8.7.56.2002",
        STEP_NAME + " route sync",
        {
            "declaration_source": sign_base.display_path(
                build_metrics_paths(PUBLIC_OUT, STEM, "declaration_gate")["json"]
            ),
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "selected_next_generation_route_or_none": NEXT_ROUTE,
            "selected_followup_route": FOLLOWUP_ROUTE_NAME,
            "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        },
        route_rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_boundary_phase_carrier_higher_q_extension_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_next_hit": sign_base.hit(status_text, "8.7.56.1999"),
                "roadmap_next_hit": sign_base.hit(roadmap_text, "8.7.56.2003-.2006"),
                "current_problem_next_hit": sign_base.hit(current_problem_text, "same_level_one_parameter_retry_admissible"),
                "current_status_next_hit": sign_base.hit(current_status_text, "finite phase-carrier window retained"),
                "unified_roadmap_next_hit": sign_base.hit(unified_text, ".2003-.2006"),
                "long_roadmap_next_hit": sign_base.hit(long_text, "phase-carrier higher-q extension audit"),
                "part5_next_hit": sign_base.hit(part5_text, ".1991-.1998"),
            },
        },
    )

    declaration_paths = write_artifact("declaration_gate", declaration_payload)
    route_paths = write_artifact("route_sync", route_payload)
    print("[ok] 8.7.56.1999-.2002 boundary phase-carrier higher-q extension artifacts generated")
    print(f"[ok] declaration: {declaration_paths['json']}")
    print(f"[ok] route sync:   {route_paths['json']}")


if __name__ == "__main__":
    main()
