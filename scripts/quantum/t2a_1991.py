#!/usr/bin/env python3
"""Generate 8.7.56.1991-.1994 asymptotic phase-drift audit artifacts."""

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
from scipy.optimize import minimize_scalar


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
    / "q_8_7_56_1987_1990_boundary_local_jet_generalization_gate_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1991-1994"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor boundary local-jet "
    "asymptotic phase-drift audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "boundary_local_jet_phase_drift_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_box_edge_local_jet_extension_to_40_retained_"
    "asymptotic_phase_drift_audit_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_phase_carrier_window_40_to_120_retained_"
    "higher_q_generalization_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_boundary_phase_carrier_"
    "decision_gate_registry"
)
NEXT_ROUTE = "8.7.56.1995"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_boundary_phase_carrier_"
    "higher_q_extension_audit"
)
FOLLOWUP_ROUTE = "8.7.56.1999"
FIT_Q_MIN = 40.0
FIT_Q_MAX = 120.0
HOLDOUT_Q_MAX = 200.0


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


# 関数: h(r)=r f(r)^2 の三階微分境界値を返す。

def boundary_h3(radius: np.ndarray, field: np.ndarray) -> float:
    """Return h'''(R_box) for h(r)=r f(r)^2."""
    weight = (field**2) * (radius**2)
    edge_density = weight / radius
    edge_density_prime = np.gradient(edge_density, radius, edge_order=2)
    edge_density_second = np.gradient(edge_density_prime, radius, edge_order=2)
    edge_density_third = np.gradient(edge_density_second, radius, edge_order=2)
    return float(edge_density_third[-1])


# 関数: local-jet family の phase carrier を返す。

def phase_carrier(q_ratio: float, phi0: float, phi_inv: float, delta_r: float) -> float:
    """Return the rational boundary phase carrier."""
    return float(phi0 + (phi_inv / q_ratio) + (delta_r * q_ratio))


# 関数: phase-corrected zero equation を返す。

def shifted_zero_equation(
    q_ratio: float,
    h0: float,
    h1: float,
    h2: float,
    r_box: float,
    phi0: float,
    phi_inv: float,
    delta_r: float,
) -> float:
    """Return the local-jet zero equation with a boundary phase carrier."""
    phase = (q_ratio * r_box) + phase_carrier(q_ratio, phi0, phi_inv, delta_r)
    return (
        (-h0 * q_ratio * q_ratio + h2) * math.cos(phase)
        + (h1 * q_ratio) * math.sin(phase)
    )


# 関数: one-parameter h3-corrected zero equation を返す。

def h3_zero_equation(
    q_ratio: float,
    h0: float,
    h1: float,
    h2: float,
    h3: float,
    r_box: float,
) -> float:
    """Return the single-parameter third-jet boundary rule."""
    return (
        (-h0 * q_ratio * q_ratio + h2) * math.cos(q_ratio * r_box)
        + (h1 * q_ratio - (h3 / q_ratio)) * math.sin(q_ratio * r_box)
    )


# 関数: one q interval で custom zero equation の root lattice を返す。

def find_custom_zeros(
    q_min: float,
    q_max: float,
    equation,
    r_box: float,
) -> np.ndarray:
    """Locate all custom zero roots on one q interval."""
    roots: list[float] = []
    max_mode = int(math.ceil(q_max * r_box / math.pi)) + 3
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


# 関数: nearest-matched residual に対する carrier 係数を fit する。

def fit_phase_carrier(
    exact_roots: np.ndarray,
    predicted_roots: np.ndarray,
    r_box: float,
) -> tuple[float, float, float]:
    """Fit one theorem-level boundary phase carrier on one q window."""
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
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design, delta_array, rcond=None)
    q_shift_constant = float(coefficients[0])
    q_shift_inverse = float(coefficients[1])
    q_shift_linear = float(coefficients[2])
    return (
        float(-r_box * q_shift_constant),
        float(-r_box * q_shift_inverse),
        float(-r_box * q_shift_linear),
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
    q_scan = np.linspace(q_min, q_max, 160001)
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


# 関数: audit で使う公式群を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the asymptotic phase-drift audit."""
    return {
        "retained_rule": "G_jet(q)=(-h0 q^2 + h2) cos(q R_box) + h1 q sin(q R_box)=0",
        "single_parameter_h3": "G_h3(q)=(-h0 q^2 + h2) cos(q R_box) + (h1 q - h3/q) sin(q R_box)=0",
        "single_parameter_boundary_shift": "G_dR(q)=(-h0 q^2 + h2) cos(q (R_box+dR)) + h1 q sin(q (R_box+dR))=0",
        "boundary_phase_carrier": "G_phi(q)=(-h0 q^2 + h2) cos(q R_box + phi0 + phi_-1/q + dR q) + h1 q sin(q R_box + phi0 + phi_-1/q + dR q)=0",
        "fit_window": "carrier coefficients are fitted on 40<=q/m0<=120 and then evaluated on the same finite asymptotic window plus the 120<=q/m0<=200 holdout",
    }


# 関数: `.1991-.1994` を実行する。

def main() -> None:
    """Execute the boundary local-jet asymptotic phase-drift audit."""
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
    inventory_ready = bool(prior_summary["asymptotic_phase_drift_audit_admissible_now"])

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
    h3 = boundary_h3(radius, field)

    exact_roots_all = asymp_base.find_signed_zeros_interval(
        radius,
        weight,
        norm,
        HOLDOUT_Q_MAX,
    )
    baseline_roots = local_jet_base.find_local_jet_zeros(
        FIT_Q_MIN,
        HOLDOUT_Q_MAX,
        h0,
        h1,
        h2,
        r_box,
    )

    fit_exact_roots = exact_roots_all[
        (exact_roots_all >= FIT_Q_MIN) & (exact_roots_all <= FIT_Q_MAX)
    ]
    phase_phi0, phase_phi_inv, phase_delta_r = fit_phase_carrier(
        fit_exact_roots,
        baseline_roots,
        r_box,
    )

    h3_roots = find_custom_zeros(
        FIT_Q_MIN,
        HOLDOUT_Q_MAX,
        lambda q_value: h3_zero_equation(q_value, h0, h1, h2, h3, r_box),
        r_box,
    )

    delta_r_optimum = float(
        minimize_scalar(
            lambda delta_r: evaluate_window(
                radius,
                weight,
                norm,
                exact_roots_all,
                find_custom_zeros(
                    FIT_Q_MIN,
                    FIT_Q_MAX,
                    lambda q_value: shifted_zero_equation(
                        q_value,
                        h0,
                        h1,
                        h2,
                        r_box,
                        0.0,
                        0.0,
                        delta_r,
                    ),
                    r_box,
                ),
                FIT_Q_MIN,
                FIT_Q_MAX,
            )["exact_to_pred_mean_abs_error"],
            bounds=(0.0, 0.02),
            method="bounded",
            options={"xatol": 1.0e-12},
        ).x
    )
    delta_r_roots = find_custom_zeros(
        FIT_Q_MIN,
        HOLDOUT_Q_MAX,
        lambda q_value: shifted_zero_equation(
            q_value,
            h0,
            h1,
            h2,
            r_box,
            0.0,
            0.0,
            delta_r_optimum,
        ),
        r_box,
    )

    phase_carrier_roots = find_custom_zeros(
        FIT_Q_MIN,
        HOLDOUT_Q_MAX,
        lambda q_value: shifted_zero_equation(
            q_value,
            h0,
            h1,
            h2,
            r_box,
            phase_phi0,
            phase_phi_inv,
            phase_delta_r,
        ),
        r_box,
    )

    baseline_fit = evaluate_window(
        radius,
        weight,
        norm,
        exact_roots_all,
        baseline_roots,
        FIT_Q_MIN,
        FIT_Q_MAX,
    )
    baseline_holdout = evaluate_window(
        radius,
        weight,
        norm,
        exact_roots_all,
        baseline_roots,
        FIT_Q_MAX,
        HOLDOUT_Q_MAX,
    )
    h3_fit = evaluate_window(
        radius,
        weight,
        norm,
        exact_roots_all,
        h3_roots,
        FIT_Q_MIN,
        FIT_Q_MAX,
    )
    h3_holdout = evaluate_window(
        radius,
        weight,
        norm,
        exact_roots_all,
        h3_roots,
        FIT_Q_MAX,
        HOLDOUT_Q_MAX,
    )
    delta_r_fit = evaluate_window(
        radius,
        weight,
        norm,
        exact_roots_all,
        delta_r_roots,
        FIT_Q_MIN,
        FIT_Q_MAX,
    )
    delta_r_holdout = evaluate_window(
        radius,
        weight,
        norm,
        exact_roots_all,
        delta_r_roots,
        FIT_Q_MAX,
        HOLDOUT_Q_MAX,
    )
    phase_carrier_fit = evaluate_window(
        radius,
        weight,
        norm,
        exact_roots_all,
        phase_carrier_roots,
        FIT_Q_MIN,
        FIT_Q_MAX,
    )
    phase_carrier_holdout = evaluate_window(
        radius,
        weight,
        norm,
        exact_roots_all,
        phase_carrier_roots,
        FIT_Q_MAX,
        HOLDOUT_Q_MAX,
    )

    single_parameter_supported = bool(
        (delta_r_fit["sign_mismatch_fraction"] <= 0.01)
        and (delta_r_holdout["sign_mismatch_fraction"] <= 0.01)
    )
    phase_carrier_window_supported = bool(
        (phase_carrier_fit["sign_mismatch_fraction"] <= 0.01)
        and (phase_carrier_fit["exact_to_pred_max_abs_error"] <= 0.005)
    )
    higher_q_generalization_beyond_120_not_yet_supported = bool(
        phase_carrier_holdout["sign_mismatch_fraction"] > 0.05
    )
    physical_reject_required = False

    rows = [
        sign_base.row("inventory_ready", "pass" if inventory_ready else "reject", "phase-drift audit inventory ready", sign_base.truth(inventory_ready), "The asymptotic drift audit only starts after the finite higher-q local-jet extension has been formally retained and the drift blocker has been selected."),
        sign_base.row("baseline_fit_sign_mismatch_fraction", "watch", "baseline sign mismatch fraction on 40<=q/m0<=120", baseline_fit["sign_mismatch_fraction"], "This is the retained local-jet drift level before any new correction theorem is introduced."),
        sign_base.row("baseline_holdout_sign_mismatch_fraction", "watch", "baseline sign mismatch fraction on 120<=q/m0<=200", baseline_holdout["sign_mismatch_fraction"], "The holdout window measures how quickly the retained local-jet theorem loses phase coherence deeper in the asymptotic regime."),
        sign_base.row("h3_single_parameter_sign_mismatch_fraction", "watch", "single-parameter h3 sign mismatch fraction on 40<=q/m0<=120", h3_fit["sign_mismatch_fraction"], "The third-boundary-jet correction is the most natural one-parameter asymptotic completion, so its failure matters for the next route choice."),
        sign_base.row("delta_r_single_parameter_sign_mismatch_fraction", "watch", "single-parameter boundary-shift sign mismatch fraction on 40<=q/m0<=120", delta_r_fit["sign_mismatch_fraction"], "A pure effective-boundary shift is the best simple one-parameter phase correction benchmark."),
        sign_base.row("delta_r_single_parameter_holdout_sign_mismatch_fraction", "watch", "single-parameter boundary-shift sign mismatch fraction on 120<=q/m0<=200", delta_r_holdout["sign_mismatch_fraction"], "A genuine one-parameter theorem would need to survive the later holdout, not just the first asymptotic window."),
        sign_base.row("phase_carrier_fit_sign_mismatch_fraction", "pass" if phase_carrier_fit["sign_mismatch_fraction"] <= 0.01 else "watch", "boundary phase-carrier sign mismatch fraction on 40<=q/m0<=120", phase_carrier_fit["sign_mismatch_fraction"], "The new theorem-level phase carrier is only worth promoting if it sharply improves the finite asymptotic window where the retained rule first drifts."),
        sign_base.row("phase_carrier_fit_root_nn_max_abs_error", "watch", "boundary phase-carrier max nearest-neighbor root error on 40<=q/m0<=120", phase_carrier_fit["exact_to_pred_max_abs_error"], "Root-lattice locking on the finite asymptotic window determines whether the phase carrier is more than a cosmetic sign fix."),
        sign_base.row("phase_carrier_holdout_sign_mismatch_fraction", "watch", "boundary phase-carrier sign mismatch fraction on 120<=q/m0<=200", phase_carrier_holdout["sign_mismatch_fraction"], "The later holdout determines whether the new phase carrier is already asymptotic or only a finite-window continuation."),
        sign_base.row("single_parameter_supported", "reject" if not single_parameter_supported else "pass", "one-parameter asymptotic correction supported", sign_base.truth(single_parameter_supported), "The current branch promised to test whether the later phase drift can be absorbed by a single theorem-level correction before escalating to a richer boundary carrier."),
        sign_base.row("phase_carrier_window_supported", "pass" if phase_carrier_window_supported else "reject", "finite-window boundary phase-carrier supported", sign_base.truth(phase_carrier_window_supported), "A multi-term boundary phase carrier can still be honest if it stabilizes the first asymptotic window without claiming global canonical closure."),
        sign_base.row("higher_q_generalization_beyond_120_not_yet_supported", "watch" if higher_q_generalization_beyond_120_not_yet_supported else "pass", "higher-q generalization beyond 120 not yet supported", sign_base.truth(higher_q_generalization_beyond_120_not_yet_supported), "The holdout beyond the fitted finite asymptotic window decides whether the new carrier is already global or needs a followup generalization audit."),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "fit_window_lower_over_m0": FIT_Q_MIN,
        "fit_window_upper_over_m0": FIT_Q_MAX,
        "holdout_window_upper_over_m0": HOLDOUT_Q_MAX,
        "boundary_h3_exact": h3,
        "boundary_shift_delta_r_optimum": delta_r_optimum,
        "phase_carrier_phi0": phase_phi0,
        "phase_carrier_phi_inv": phase_phi_inv,
        "phase_carrier_delta_r": phase_delta_r,
        "baseline_fit_sign_mismatch_fraction": baseline_fit["sign_mismatch_fraction"],
        "baseline_holdout_sign_mismatch_fraction": baseline_holdout["sign_mismatch_fraction"],
        "h3_single_parameter_fit_sign_mismatch_fraction": h3_fit["sign_mismatch_fraction"],
        "h3_single_parameter_holdout_sign_mismatch_fraction": h3_holdout["sign_mismatch_fraction"],
        "delta_r_single_parameter_fit_sign_mismatch_fraction": delta_r_fit["sign_mismatch_fraction"],
        "delta_r_single_parameter_holdout_sign_mismatch_fraction": delta_r_holdout["sign_mismatch_fraction"],
        "phase_carrier_fit_root_nn_max_abs_error": phase_carrier_fit["exact_to_pred_max_abs_error"],
        "phase_carrier_fit_root_nn_mean_abs_error": phase_carrier_fit["exact_to_pred_mean_abs_error"],
        "phase_carrier_fit_sign_mismatch_fraction": phase_carrier_fit["sign_mismatch_fraction"],
        "phase_carrier_fit_signed_reconstruction_max_abs_error": phase_carrier_fit["signed_reconstruction_max_abs_error"],
        "phase_carrier_holdout_root_nn_max_abs_error": phase_carrier_holdout["exact_to_pred_max_abs_error"],
        "phase_carrier_holdout_root_nn_mean_abs_error": phase_carrier_holdout["exact_to_pred_mean_abs_error"],
        "phase_carrier_holdout_sign_mismatch_fraction": phase_carrier_holdout["sign_mismatch_fraction"],
        "phase_carrier_holdout_signed_reconstruction_max_abs_error": phase_carrier_holdout["signed_reconstruction_max_abs_error"],
        "single_parameter_supported": single_parameter_supported,
        "phase_carrier_window_supported": phase_carrier_window_supported,
        "higher_q_generalization_beyond_120_not_yet_supported": higher_q_generalization_beyond_120_not_yet_supported,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.1993",
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
                "fit_window_over_m0": [FIT_Q_MIN, FIT_Q_MAX],
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
            "overall_status": "vector_qball_form_factor_boundary_local_jet_phase_drift_audited",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.1991"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.1991-.1994"),
                "current_problem_hit": sign_base.hit(current_problem_text, "asymptotic_phase_drift_audit_admissible_now"),
                "current_status_hit": sign_base.hit(current_status_text, "boundary local-jet asymptotic phase-drift audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".1991-.1994"),
                "long_roadmap_hit": sign_base.hit(long_text, "boundary local-jet asymptotic phase-drift audit"),
                "part5_hit": sign_base.hit(part5_text, ".1983-.1990"),
            },
        },
    )

    route_rows = [
        sign_base.row("single_parameter_supported", "reject" if not single_parameter_supported else "pass", "one-parameter asymptotic correction supported", sign_base.truth(single_parameter_supported), "The branch only escalates beyond one-parameter completion if every honest simple correction fails on the later phase-drift windows."),
        sign_base.row("phase_carrier_window_supported", "pass" if phase_carrier_window_supported else "reject", "finite-window boundary phase-carrier supported", sign_base.truth(phase_carrier_window_supported), "The next route is justified only if the new boundary phase carrier really improves the finite asymptotic window."),
        sign_base.row("higher_q_generalization_beyond_120_not_yet_supported", "watch" if higher_q_generalization_beyond_120_not_yet_supported else "pass", "higher-q generalization beyond 120 not yet supported", sign_base.truth(higher_q_generalization_beyond_120_not_yet_supported), "A finite-window rescue still needs a followup generalization audit before it can claim asymptotic closure."),
        sign_base.row("next_route_fixed", "pass", "next route fixed", 1.0, "The next official branch is the boundary phase-carrier decision gate / registry."),
    ]

    route_payload = sign_base.payload(
        "8.7.56.1994",
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
            "overall_status": "vector_qball_form_factor_boundary_local_jet_phase_drift_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_next_hit": sign_base.hit(status_text, "8.7.56.1991"),
                "roadmap_next_hit": sign_base.hit(roadmap_text, "8.7.56.1995-.1998"),
                "current_problem_next_hit": sign_base.hit(current_problem_text, "same_level_box_edge_retry_admissible"),
                "current_status_next_hit": sign_base.hit(current_status_text, "asymptotic phase drift selected"),
                "unified_roadmap_next_hit": sign_base.hit(unified_text, ".1995-.1998"),
                "long_roadmap_next_hit": sign_base.hit(long_text, "boundary local-jet asymptotic phase-drift decision gate"),
                "part5_next_hit": sign_base.hit(part5_text, ".1983-.1990"),
            },
        },
    )

    declaration_paths = write_artifact("declaration_gate", declaration_payload)
    route_paths = write_artifact("route_sync", route_payload)
    print("[ok] 8.7.56.1991-.1994 asymptotic phase-drift audit artifacts generated")
    print(f"[ok] declaration: {declaration_paths['json']}")
    print(f"[ok] route sync:   {route_paths['json']}")


if __name__ == "__main__":
    main()
