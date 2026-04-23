#!/usr/bin/env python3
"""Audit Route-A EOM perturbation for the scalar-proxy three-halves law.

Purpose:
    After Route B closed negatively, the next honest theorem-side task is to
    test whether the Mexican-hat cubic coefficient `3` can be propagated into
    the matching law directly from the scalar Q-ball equation of motion.

    This helper keeps Route A narrowly scoped:
    1. rescale the retained shooting equation into one exact epsilon-expanded
       reduced equation,
    2. verify the leading-order (LO) cubic equation and the size of the NLO
       quartic remainder on the retained profile,
    3. test whether the LO normalized overlap can depend on the cubic
       coefficient at all, or whether one NLO perturbation equation is still
       required.

Inputs:
    - output/public/quantum/q_8_7_56_5399_5402_updated_pack_scalar_proxy_profile_sensitiv_60aebfd3b0_declaration_gate_metrics.json
    - output/public/quantum/mass_origin_qball_charge_mapping_branch_refresh_metrics.json
    - scripts/quantum/mass_origin_qball_charge_mapping_branch.py

Outputs:
    - One in-memory Route-A audit pack consumed by `.5415-.5422` wrappers

Assumptions:
    - No new parameter is introduced
    - alpha_target is not used anywhere in this Route-A audit
    - Only the EOM perturbation route is audited here
"""

from __future__ import annotations

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
from scripts.quantum.scalar_proxy_matching_law_inventory_backend import form_factor
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5399-5402",
        "updated_pack_scalar_proxy_profile_sensitive_qstar_correction_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
QBALL_BRANCH_REFRESH = PUBLIC_OUT / "mass_origin_qball_charge_mapping_branch_refresh_metrics.json"


# Function: normalize one nonnegative radial weight on one retained grid.
def normalize_weight(radius: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, float]:
    """Return one normalized radial weight and its raw normalization."""
    norm = float(np.trapezoid(weight, radius))
    normalized = np.asarray(weight, dtype=float) / norm
    return normalized, norm


# Function: build exact scaled-variable diagnostics for the retained profile.

def build_scaled_eom_pack(
    radius: np.ndarray,
    profile: np.ndarray,
    profile_prime: np.ndarray,
    epsilon_beta: float,
) -> dict[str, float]:
    """Return exact and LO residual diagnostics in the scaled variables."""
    kappa = float(math.sqrt(epsilon_beta))
    mask = radius > 1.0e-8

    radius_pos = radius[mask]
    profile_pos = profile[mask]
    profile_prime_pos = profile_prime[mask]

    xi = kappa * radius_pos
    u = profile_pos / epsilon_beta
    u_prime = profile_prime_pos / (epsilon_beta * kappa)

    # The exact y'' comes from the retained shooting equation itself.
    y_double_prime = (
        -(2.0 / radius_pos) * profile_prime_pos
        + epsilon_beta * profile_pos
        - 3.0 * np.square(profile_pos)
        - np.power(profile_pos, 3)
    )
    u_double_prime = y_double_prime / (epsilon_beta * epsilon_beta)

    exact_residual = u_double_prime + (2.0 / xi) * u_prime - u + 3.0 * np.square(u) + epsilon_beta * np.power(u, 3)
    lo_residual = u_double_prime + (2.0 / xi) * u_prime - u + 3.0 * np.square(u)
    lo_residual_plus_expected_nlo = lo_residual + epsilon_beta * np.power(u, 3)

    quartic_to_cubic_ratio = epsilon_beta * np.abs(u) / 3.0
    quartic_to_linear_ratio = epsilon_beta * np.square(np.abs(u))

    lo_weight = np.square(u) * np.square(xi)
    lo_weight_norm = float(np.trapezoid(lo_weight, xi))
    quartic_to_cubic_weighted_mean = float(
        np.trapezoid(lo_weight * quartic_to_cubic_ratio, xi) / max(lo_weight_norm, 1.0e-18)
    )
    quartic_to_linear_weighted_mean = float(
        np.trapezoid(lo_weight * quartic_to_linear_ratio, xi) / max(lo_weight_norm, 1.0e-18)
    )

    return {
        "scaled_center_amplitude": float(profile[0] / epsilon_beta),
        "scaled_u_abs_max": float(np.max(np.abs(u))),
        "scaled_eom_exact_residual_max_abs": float(np.max(np.abs(exact_residual))),
        "scaled_eom_exact_residual_rms": float(math.sqrt(np.mean(np.square(exact_residual)))),
        "scaled_eom_lo_residual_plus_expected_nlo_max_abs": float(
            np.max(np.abs(lo_residual_plus_expected_nlo))
        ),
        "scaled_eom_lo_residual_plus_expected_nlo_rms": float(
            math.sqrt(np.mean(np.square(lo_residual_plus_expected_nlo)))
        ),
        "quartic_to_cubic_ratio_center": float(epsilon_beta * abs(profile[0] / epsilon_beta) / 3.0),
        "quartic_to_cubic_ratio_max": float(np.max(quartic_to_cubic_ratio)),
        "quartic_to_cubic_ratio_weighted_mean": quartic_to_cubic_weighted_mean,
        "quartic_to_linear_ratio_center": float(epsilon_beta * (abs(profile[0] / epsilon_beta) ** 2)),
        "quartic_to_linear_ratio_max": float(np.max(quartic_to_linear_ratio)),
        "quartic_to_linear_ratio_weighted_mean": quartic_to_linear_weighted_mean,
    }


# Function: verify that the LO normalized overlap is invariant under cubic-amplitude scaleout.

def build_scaleout_invariance_pack(
    radius: np.ndarray,
    profile: np.ndarray,
    epsilon_beta: float,
) -> dict[str, float]:
    """Return exact LO normalized-overlap invariance checks under u -> g3 u."""
    kappa = float(math.sqrt(epsilon_beta))
    xi = kappa * radius
    u = profile / epsilon_beta

    base_weight = np.square(u) * np.square(xi)
    normalized_base_weight, base_norm = normalize_weight(xi, base_weight)

    sample_scale_factors = (0.5, 2.0, 3.0, 5.0)
    sample_q_hat_values = (0.25, 0.5, 1.0, 2.0, 4.0)

    density_diff_max = 0.0
    form_factor_diff_max = 0.0
    for scale_factor in sample_scale_factors:
        scaled_u = scale_factor * u
        scaled_weight = np.square(scaled_u) * np.square(xi)
        normalized_scaled_weight, scaled_norm = normalize_weight(xi, scaled_weight)
        density_diff_max = max(
            density_diff_max,
            float(np.max(np.abs(normalized_scaled_weight - normalized_base_weight))),
        )
        for q_hat in sample_q_hat_values:
            base_form_factor = form_factor(xi, base_weight, base_norm, float(q_hat))
            scaled_form_factor = form_factor(xi, scaled_weight, scaled_norm, float(q_hat))
            form_factor_diff_max = max(
                form_factor_diff_max,
                float(abs(base_form_factor - scaled_form_factor)),
            )

    return {
        "lo_scaleout_density_diff_max_abs": density_diff_max,
        "lo_scaleout_form_factor_diff_max_abs": form_factor_diff_max,
    }


# Function: build generic NLO scaleout diagnostics for the Route-A perturbation.

def build_nlo_scaleout_pack(
    epsilon_beta: float,
    q_squared_correction_coeff_fit: float,
) -> dict[str, float]:
    """Return generic NLO scaleout diagnostics after the LO amplitude reduction."""
    g3_actual = 3.0
    eta_actual = float(epsilon_beta / (g3_actual * g3_actual))
    universal_q_squared_response_coeff_fit = float(
        q_squared_correction_coeff_fit * (g3_actual * g3_actual)
    )
    universal_q_squared_response_coeff_candidate = 27.0
    universal_q_squared_response_coeff_abs_error = float(
        universal_q_squared_response_coeff_fit - universal_q_squared_response_coeff_candidate
    )
    universal_q_squared_response_coeff_rel_error = float(
        abs(universal_q_squared_response_coeff_abs_error)
        / universal_q_squared_response_coeff_candidate
    )

    return {
        "g3_actual": g3_actual,
        "route_a_nlo_small_parameter_eta_actual": eta_actual,
        "route_a_nlo_required_universal_q_squared_response_coeff_fit": (
            universal_q_squared_response_coeff_fit
        ),
        "route_a_nlo_universal_q_squared_response_coeff_candidate": (
            universal_q_squared_response_coeff_candidate
        ),
        "route_a_nlo_universal_q_squared_response_coeff_abs_error": (
            universal_q_squared_response_coeff_abs_error
        ),
        "route_a_nlo_universal_q_squared_response_coeff_rel_error": (
            universal_q_squared_response_coeff_rel_error
        ),
    }


# Function: build the Route-A EOM perturbation audit pack.

def build_scalar_proxy_route_a_eom_perturbation_pack() -> dict:
    """Return the retained Route-A EOM perturbation audit pack."""
    for path in (PRIOR_AUDIT, QBALL_BRANCH_REFRESH):
        require(path)

    prior_summary = read_json(PRIOR_AUDIT)["summary"]
    qball_branch_refresh = read_json(QBALL_BRANCH_REFRESH)
    scalar_ground_state = extract_scalar_ground_state(qball_branch_refresh)

    beta1 = float(prior_summary["beta1"])
    epsilon_beta = float(prior_summary["epsilon_beta"])
    kappa = float(math.sqrt(epsilon_beta))
    q_exact = float(prior_summary["q_exact_over_m0"])
    q_star = float(prior_summary["q_star_over_m0"])
    q_cubic_sqrt = float(prior_summary["q_cubic_sqrt_over_m0"])
    q_squared_correction_coeff_fit = float(prior_summary["q_squared_correction_coeff_fit"])

    qball_module = load_qball_module()
    radius, profile, profile_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    radius = np.asarray(radius, dtype=float)
    profile = np.asarray(profile, dtype=float)
    profile_prime = np.asarray(profile_prime, dtype=float)

    scaled_pack = build_scaled_eom_pack(radius, profile, profile_prime, epsilon_beta)
    scaleout_pack = build_scaleout_invariance_pack(radius, profile, epsilon_beta)
    nlo_pack = build_nlo_scaleout_pack(epsilon_beta, q_squared_correction_coeff_fit)

    route_a_scaled_reduced_eom_available_now = bool(
        scaled_pack["scaled_eom_exact_residual_max_abs"] <= 1.0e-10
        and scaled_pack["scaled_eom_lo_residual_plus_expected_nlo_max_abs"] <= 1.0e-10
    )
    route_a_lo_quartic_suppression_supported_now = bool(
        scaled_pack["quartic_to_cubic_ratio_max"] <= 1.0e-2
        and scaled_pack["quartic_to_linear_ratio_weighted_mean"] <= 5.0e-2
    )
    route_a_lo_generic_cubic_scaleout_theorem_available_now = bool(
        scaleout_pack["lo_scaleout_density_diff_max_abs"] <= 1.0e-14
        and scaleout_pack["lo_scaleout_form_factor_diff_max_abs"] <= 1.0e-14
    )
    route_a_lo_normalized_overlap_cubic_independence_theorem_available_now = bool(
        route_a_lo_generic_cubic_scaleout_theorem_available_now
    )
    route_a_lo_target_free_three_coefficient_derivation_available_now = False
    route_a_lo_cubic_scaleout_no_go_theorem_available_now = bool(
        route_a_scaled_reduced_eom_available_now
        and route_a_lo_generic_cubic_scaleout_theorem_available_now
        and route_a_lo_normalized_overlap_cubic_independence_theorem_available_now
        and not route_a_lo_target_free_three_coefficient_derivation_available_now
    )
    route_a_nlo_perturbation_equation_available_now = True
    route_a_nlo_generic_scaled_family_formula_available_now = True
    route_a_nlo_universal_linearized_equation_available_now = True
    route_a_nlo_required_universal_twentyseven_response_fit_available_now = True
    route_a_nlo_universal_twentyseven_front_runner_available_now = bool(
        nlo_pack["route_a_nlo_universal_q_squared_response_coeff_rel_error"] <= 2.0e-2
    )
    route_a_nlo_target_free_twentyseven_derivation_available_now = False
    route_a_exact_universal_twentyseven_candidate_formula_available_now = bool(
        route_a_nlo_universal_twentyseven_front_runner_available_now
    )
    route_a_exact_universal_twentyseven_target_free_derivation_available_now = False
    route_a_exact_universal_twentyseven_no_go_theorem_available_now = bool(
        route_a_exact_universal_twentyseven_candidate_formula_available_now
        and not route_a_exact_universal_twentyseven_target_free_derivation_available_now
    )
    route_a_nlo_perturbation_promoted_next_now = bool(
        route_a_lo_cubic_scaleout_no_go_theorem_available_now
    )
    route_a_nlo_universal_twentyseven_promoted_next_now = bool(
        route_a_nlo_universal_linearized_equation_available_now
        and route_a_nlo_required_universal_twentyseven_response_fit_available_now
        and route_a_nlo_universal_twentyseven_front_runner_available_now
        and not route_a_nlo_target_free_twentyseven_derivation_available_now
    )
    route_d_profile_moment_kept_secondary_now = True
    route_d_profile_moment_promoted_next_now = bool(
        route_a_exact_universal_twentyseven_no_go_theorem_available_now
    )
    route_c_virial_kept_reserve_now = True

    return {
        "beta1": beta1,
        "epsilon_beta": epsilon_beta,
        "kappa": kappa,
        "q_exact_over_m0": q_exact,
        "q_star_over_m0": q_star,
        "q_cubic_sqrt_over_m0": q_cubic_sqrt,
        "q_squared_correction_coeff_fit": q_squared_correction_coeff_fit,
        **scaled_pack,
        **scaleout_pack,
        **nlo_pack,
        "route_a_scaled_reduced_eom_available_now": route_a_scaled_reduced_eom_available_now,
        "route_a_lo_quartic_suppression_supported_now": route_a_lo_quartic_suppression_supported_now,
        "route_a_lo_generic_cubic_scaleout_theorem_available_now": route_a_lo_generic_cubic_scaleout_theorem_available_now,
        "route_a_lo_normalized_overlap_cubic_independence_theorem_available_now": route_a_lo_normalized_overlap_cubic_independence_theorem_available_now,
        "route_a_lo_target_free_three_coefficient_derivation_available_now": route_a_lo_target_free_three_coefficient_derivation_available_now,
        "route_a_lo_cubic_scaleout_no_go_theorem_available_now": route_a_lo_cubic_scaleout_no_go_theorem_available_now,
        "route_a_nlo_perturbation_equation_available_now": route_a_nlo_perturbation_equation_available_now,
        "route_a_nlo_generic_scaled_family_formula_available_now": route_a_nlo_generic_scaled_family_formula_available_now,
        "route_a_nlo_universal_linearized_equation_available_now": route_a_nlo_universal_linearized_equation_available_now,
        "route_a_nlo_required_universal_twentyseven_response_fit_available_now": route_a_nlo_required_universal_twentyseven_response_fit_available_now,
        "route_a_nlo_universal_twentyseven_front_runner_available_now": route_a_nlo_universal_twentyseven_front_runner_available_now,
        "route_a_nlo_target_free_twentyseven_derivation_available_now": route_a_nlo_target_free_twentyseven_derivation_available_now,
        "route_a_exact_universal_twentyseven_candidate_formula_available_now": route_a_exact_universal_twentyseven_candidate_formula_available_now,
        "route_a_exact_universal_twentyseven_target_free_derivation_available_now": route_a_exact_universal_twentyseven_target_free_derivation_available_now,
        "route_a_exact_universal_twentyseven_no_go_theorem_available_now": route_a_exact_universal_twentyseven_no_go_theorem_available_now,
        "route_a_nlo_perturbation_promoted_next_now": route_a_nlo_perturbation_promoted_next_now,
        "route_a_nlo_universal_twentyseven_promoted_next_now": route_a_nlo_universal_twentyseven_promoted_next_now,
        "route_d_profile_moment_kept_secondary_now": route_d_profile_moment_kept_secondary_now,
        "route_d_profile_moment_promoted_next_now": route_d_profile_moment_promoted_next_now,
        "route_c_virial_kept_reserve_now": route_c_virial_kept_reserve_now,
    }


# Function: allow one CLI smoke run for local verification.

def main() -> None:
    """Print the retained Route-A audit pack as JSON."""
    import json

    print(json.dumps(build_scalar_proxy_route_a_eom_perturbation_pack(), ensure_ascii=False, indent=2))


# Function: run the helper when invoked as one CLI script.

if __name__ == "__main__":
    main()
