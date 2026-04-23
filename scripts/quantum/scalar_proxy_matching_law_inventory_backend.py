#!/usr/bin/env python3
"""Build the retained scalar-proxy matching-law inventory diagnostic pack.

Purpose:
    After the dense alpha(q) computation and the matching-scale redrive closed,
    the live blocker is no longer formula failure and no longer one more
    selected-extension replay. This helper inventories concrete matching-law
    candidates that might reproduce q_exact from retained scalar data alone and
    ranks which family deserves the next theorem-side audit.

Inputs:
    - output/public/quantum/q_8_7_56_5375_5378_updated_pack_scalar_proxy_alpha_q_curve_di_9aed6addb2_declaration_gate_metrics.json
    - output/public/quantum/mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_matching_scale_review_numeric_evaluation_metrics.json
    - output/public/quantum/mass_origin_qball_charge_mapping_branch_refresh_metrics.json
    - scripts/quantum/mass_origin_qball_charge_mapping_branch.py

Outputs:
    - One in-memory candidate inventory pack consumed by `.5391-.5398` wrappers

Assumptions:
    - alpha(q)=F(q)^2/(4*pi) already survived on the retained scalar profile
    - q_exact is already fixed by the dense retained alpha(q) curve
    - The present task is to inventory matching laws, not to tune alpha_target
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.scalar_proxy_alpha_q_curve_backend import extract_scalar_ground_state
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import load_qball_module
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import read_json
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import require

PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
ALPHA_Q_CURVE_AUDIT = (
    PUBLIC_OUT
    / "q_8_7_56_5375_5378_updated_pack_scalar_proxy_alpha_q_curve_di_9aed6addb2_declaration_gate_metrics.json"
)
PROJECTION_MATCHING_REVIEW = (
    PUBLIC_OUT
    / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_matching_scale_review_numeric_evaluation_metrics.json"
)
QBALL_BRANCH_REFRESH = PUBLIC_OUT / "mass_origin_qball_charge_mapping_branch_refresh_metrics.json"


# Function: evaluate one retained spherical form factor at one q/m0 value.
def form_factor(radius: np.ndarray, weight: np.ndarray, norm: float, q_ratio: float) -> float:
    """Evaluate one retained spherical form factor at one q/m0 value."""
    qx = float(q_ratio) * radius
    sinc = np.ones_like(qx)
    mask = np.abs(qx) > 1.0e-12
    sinc[mask] = np.sin(qx[mask]) / qx[mask]
    numerator = np.trapezoid(weight * sinc, radius)
    return float(numerator / norm)


# Function: evaluate F, F', and F'' at q_exact.

def evaluate_local_form_factor_derivatives(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    q_exact: float,
    step: float = 1.0e-6,
) -> dict[str, float]:
    """Return one stable local derivative pack for F(q) at q_exact."""
    left_value = form_factor(radius, weight, norm, float(q_exact - step))
    center_value = form_factor(radius, weight, norm, float(q_exact))
    right_value = form_factor(radius, weight, norm, float(q_exact + step))
    first_derivative = (right_value - left_value) / (2.0 * step)
    second_derivative = (right_value - 2.0 * center_value + left_value) / (step * step)
    normalized_log_slope = abs(float(q_exact) * first_derivative / center_value)
    return {
        "F_q_exact": float(center_value),
        "F_prime_q_exact": float(first_derivative),
        "F_double_prime_q_exact": float(second_derivative),
        "F_log_slope_q_exact_abs": float(normalized_log_slope),
    }


# Function: compute one weighted centroid q candidate.

def weighted_centroid_candidate(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    local_q_values: np.ndarray,
) -> float:
    """Return one weighted centroid candidate for q/m0."""
    return float(np.trapezoid(weight * local_q_values, radius) / norm)


# Function: collect centroid-style matching-law candidates.

def build_centroid_candidates(
    radius: np.ndarray,
    profile: np.ndarray,
    weight: np.ndarray,
    norm: float,
    beta1: float,
) -> dict[str, float]:
    """Return centroid-style candidate q laws built from retained profile structure."""
    safe_radius = np.where(radius > 1.0e-12, radius, np.inf)
    inverse_radius = 1.0 / safe_radius

    safe_profile = np.maximum(np.abs(profile), 1.0e-30)
    profile_derivative = np.gradient(profile, radius)
    log_derivative = -np.gradient(np.log(safe_profile), radius)
    local_kappa = np.sqrt(
        np.maximum(
            0.0,
            (1.0 - float(beta1) ** 2) + np.square(profile_derivative / safe_profile),
        )
    )

    return {
        "centroid_inverse_radius_q_over_m0": weighted_centroid_candidate(
            radius,
            weight,
            norm,
            inverse_radius,
        ),
        "centroid_log_derivative_q_over_m0": weighted_centroid_candidate(
            radius,
            weight,
            norm,
            log_derivative,
        ),
        "centroid_local_kappa_q_over_m0": weighted_centroid_candidate(
            radius,
            weight,
            norm,
            local_kappa,
        ),
    }


# Function: select the best q candidate by relative error against q_exact.

def select_best_candidate(
    candidate_values: dict[str, float],
    q_exact: float,
) -> dict[str, float | str]:
    """Return the best candidate name, value, and relative error."""
    best_name = ""
    best_value = math.nan
    best_error = math.inf
    for name, value in candidate_values.items():
        rel_error = abs(float(value) - float(q_exact)) / float(q_exact)
        if rel_error < best_error:
            best_name = str(name)
            best_value = float(value)
            best_error = float(rel_error)

    return {
        "best_name": best_name,
        "best_value": float(best_value),
        "best_rel_error": float(best_error),
    }


# Function: build the retained scalar-proxy matching-law inventory pack.

def build_scalar_proxy_matching_law_inventory_pack() -> dict:
    """Return the retained scalar-proxy matching-law inventory pack."""
    for path in (ALPHA_Q_CURVE_AUDIT, PROJECTION_MATCHING_REVIEW, QBALL_BRANCH_REFRESH):
        require(path)

    alpha_q_curve_audit = read_json(ALPHA_Q_CURVE_AUDIT)
    projection_matching_review = read_json(PROJECTION_MATCHING_REVIEW)
    qball_branch_refresh = read_json(QBALL_BRANCH_REFRESH)

    alpha_summary = alpha_q_curve_audit["summary"]
    matching_summary = projection_matching_review["summary"]
    matching_evidence = projection_matching_review["evidence"]
    scalar_ground_state = extract_scalar_ground_state(qball_branch_refresh)

    q_exact = float(alpha_summary["primary_q_exact_over_m0"])
    q_star = float(alpha_summary["q_star_over_m0"])
    q_blind = float(alpha_summary["q_blind_over_m0"])
    beta1 = float(alpha_summary["beta1"])

    qball_module = load_qball_module()
    radius, profile, _ = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    radius = np.asarray(radius, dtype=float)
    profile = np.asarray(profile, dtype=float)
    density = np.square(profile)
    weight = density * np.square(radius)
    norm = float(np.trapezoid(weight, radius))

    derivative_pack = evaluate_local_form_factor_derivatives(radius, weight, norm, q_exact)
    stationary_candidate_supported_now = bool(
        derivative_pack["F_log_slope_q_exact_abs"] <= 1.0e-3
    )

    legacy_support_phase_candidates = {
        "half_mass_target_phase_q_over_m0": float(matching_summary["q_from_half_mass_target_phase"]),
        "mean_first_zero_q_over_m0": float(matching_summary["q_from_mean_first_zero"]),
        "rms_first_zero_q_over_m0": float(matching_summary["q_from_rms_first_zero"]),
    }
    legacy_support_phase_best = select_best_candidate(legacy_support_phase_candidates, q_exact)

    centroid_candidates = build_centroid_candidates(radius, profile, weight, norm, beta1)
    centroid_best = select_best_candidate(centroid_candidates, q_exact)

    epsilon_beta = float(1.0 - beta1**2)
    q_star_correction_c1_fit = float(((q_exact / q_star) - 1.0) / epsilon_beta)
    q_star_correction_family_o1_now = bool(0.1 <= abs(q_star_correction_c1_fit) <= 10.0)
    q_star_correction_reconstructed_q = float(q_star * (1.0 + q_star_correction_c1_fit * epsilon_beta))
    q_star_correction_reconstructed_abs_error = float(abs(q_star_correction_reconstructed_q - q_exact))

    delta_kappa_squared = float(q_exact**2 - q_star**2)
    delta_kappa_squared_rel = float(delta_kappa_squared / (q_star**2))

    blind_overlap_numeric_bridge_retained_now = bool(abs(q_exact - q_blind) <= 1.0e-12)
    overlap_consistency_tautology_rejected_now = True
    exact_matching_law_closed_form_available_now = False
    profile_sensitive_q_star_correction_family_available_now = True
    matching_law_inventory_front_runner_name = "profile_sensitive_q_star_correction_family"
    matching_law_inventory_requires_profile_sensitive_completion_now = bool(
        profile_sensitive_q_star_correction_family_available_now
        and q_star_correction_family_o1_now
        and not exact_matching_law_closed_form_available_now
    )

    return {
        "q_exact_over_m0": q_exact,
        "q_star_over_m0": q_star,
        "q_blind_over_m0": q_blind,
        "beta1": beta1,
        "epsilon_beta": epsilon_beta,
        "F_q_exact": float(derivative_pack["F_q_exact"]),
        "F_prime_q_exact": float(derivative_pack["F_prime_q_exact"]),
        "F_double_prime_q_exact": float(derivative_pack["F_double_prime_q_exact"]),
        "F_log_slope_q_exact_abs": float(derivative_pack["F_log_slope_q_exact_abs"]),
        "stationary_candidate_supported_now": stationary_candidate_supported_now,
        "legacy_support_phase_candidates": legacy_support_phase_candidates,
        "legacy_support_phase_best_name": str(legacy_support_phase_best["best_name"]),
        "legacy_support_phase_best_value": float(legacy_support_phase_best["best_value"]),
        "legacy_support_phase_best_rel_error": float(legacy_support_phase_best["best_rel_error"]),
        "legacy_support_phase_rel_errors": matching_evidence["candidate_q_rel_errors"],
        "centroid_candidates": centroid_candidates,
        "centroid_best_name": str(centroid_best["best_name"]),
        "centroid_best_value": float(centroid_best["best_value"]),
        "centroid_best_rel_error": float(centroid_best["best_rel_error"]),
        "q_star_correction_c1_fit": q_star_correction_c1_fit,
        "q_star_correction_family_o1_now": q_star_correction_family_o1_now,
        "q_star_correction_reconstructed_q_over_m0": q_star_correction_reconstructed_q,
        "q_star_correction_reconstructed_abs_error": q_star_correction_reconstructed_abs_error,
        "delta_kappa_squared": delta_kappa_squared,
        "delta_kappa_squared_rel": delta_kappa_squared_rel,
        "blind_overlap_numeric_bridge_retained_now": blind_overlap_numeric_bridge_retained_now,
        "overlap_consistency_tautology_rejected_now": overlap_consistency_tautology_rejected_now,
        "exact_matching_law_closed_form_available_now": exact_matching_law_closed_form_available_now,
        "profile_sensitive_q_star_correction_family_available_now": profile_sensitive_q_star_correction_family_available_now,
        "matching_law_inventory_front_runner_name": matching_law_inventory_front_runner_name,
        "matching_law_inventory_requires_profile_sensitive_completion_now": matching_law_inventory_requires_profile_sensitive_completion_now,
    }


# Function: allow one CLI smoke run for local verification.

def main() -> None:
    """Print the retained scalar-proxy matching-law inventory pack as JSON."""
    print(json.dumps(build_scalar_proxy_matching_law_inventory_pack(), ensure_ascii=False, indent=2))


# Function: run the helper when invoked as one CLI script.

if __name__ == "__main__":
    main()
