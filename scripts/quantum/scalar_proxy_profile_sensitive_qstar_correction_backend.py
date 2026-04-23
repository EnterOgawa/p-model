#!/usr/bin/env python3
"""Audit the retained profile-sensitive q_star correction around the 3/2 law.

Purpose:
    After `.5391-.5398` reduced the live blocker to the family

        q = q_star * (1 + c1 * (1 - beta1^2)),

    the next honest task is to determine whether the fitted coefficient is
    governed by the simple rational leading law c1 = -3/2 and which suggested
    first-principles routes are currently supported by retained scalar data.

Inputs:
    - output/public/quantum/q_8_7_56_5391_5394_updated_pack_scalar_proxy_matching_law_inv_048f2a3aa0_declaration_gate_metrics.json
    - output/public/quantum/mass_origin_qball_charge_mapping_branch_refresh_metrics.json
    - scripts/quantum/mass_origin_qball_charge_mapping_branch.py

Outputs:
    - One in-memory correction audit pack consumed by `.5399-.5406` wrappers

Assumptions:
    - q_exact, q_star, and c1_fit are already fixed by prior retained audits
    - No new parameter is introduced
    - The present task is to audit a leading law, not to retune alpha_target
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.scalar_proxy_alpha_q_curve_backend import ALPHA_TARGET
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
        "8.7.56.5391-5394",
        "updated_pack_scalar_proxy_matching_law_inventory_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
QBALL_BRANCH_REFRESH = PUBLIC_OUT / "mass_origin_qball_charge_mapping_branch_refresh_metrics.json"
FOUR_PI = 4.0 * math.pi


# Function: evaluate F, F', and F'' at one q value on the retained profile.
def evaluate_form_factor_derivatives(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    q_value: float,
    step: float = 1.0e-6,
) -> dict[str, float]:
    """Return one retained derivative pack for the spherical form factor."""
    left = form_factor(radius, weight, norm, float(q_value - step))
    center = form_factor(radius, weight, norm, float(q_value))
    right = form_factor(radius, weight, norm, float(q_value + step))
    first_derivative = (right - left) / (2.0 * step)
    second_derivative = (right - 2.0 * center + left) / (step * step)
    return {
        "F_q": float(center),
        "F_prime_q": float(first_derivative),
        "F_double_prime_q": float(second_derivative),
    }


# Function: solve the small quadratic correction implied by one local Taylor expansion.

def solve_small_quadratic_root(a_coeff: float, b_coeff: float, c_coeff: float) -> float:
    """Return the Taylor root closest to the origin."""
    if abs(a_coeff) <= 1.0e-18:
        return float(-c_coeff / b_coeff)

    discriminant = max(0.0, b_coeff * b_coeff - 4.0 * a_coeff * c_coeff)
    root_plus = (-b_coeff + math.sqrt(discriminant)) / (2.0 * a_coeff)
    root_minus = (-b_coeff - math.sqrt(discriminant)) / (2.0 * a_coeff)
    return float(root_plus if abs(root_plus) <= abs(root_minus) else root_minus)


# Function: fit one tail-family coefficient on multiple retained cutoffs.

def build_tail_fit_pack(
    radius: np.ndarray,
    profile: np.ndarray,
    kappa_ratio: float,
) -> dict[str, float | list[dict[str, float]]]:
    """Return one retained tail-fit pack for the 1/r asymptotic correction route."""
    kappa = float(math.sqrt(kappa_ratio))
    tail_windows: list[dict[str, float]] = []
    for cutoff in (15.0, 20.0, 25.0, 28.0):
        mask = radius > cutoff
        if int(np.count_nonzero(mask)) < 5:
            continue

        tail_radius = radius[mask]
        tail_profile = profile[mask]
        scaled_tail = tail_radius * np.exp(kappa * tail_radius) * tail_profile
        inverse_radius = 1.0 / tail_radius
        design = np.column_stack((np.ones_like(inverse_radius), inverse_radius))
        intercept, slope = np.linalg.lstsq(design, scaled_tail, rcond=None)[0]
        normalized_correction = float(slope / intercept) if abs(intercept) > 1.0e-18 else math.nan
        rel_std = float(np.std(scaled_tail) / max(abs(np.mean(scaled_tail)), 1.0e-18))
        tail_windows.append(
            {
                "cutoff": float(cutoff),
                "normalized_correction": normalized_correction,
                "scaled_tail_rel_std": rel_std,
            }
        )

    correction_values = np.array(
        [
            row_data["normalized_correction"]
            for row_data in tail_windows
            if math.isfinite(float(row_data["normalized_correction"]))
        ],
        dtype=float,
    )
    rel_std_values = np.array(
        [row_data["scaled_tail_rel_std"] for row_data in tail_windows],
        dtype=float,
    )
    if correction_values.size >= 2:
        correction_span = float(np.max(correction_values) - np.min(correction_values))
        mean_abs_correction = float(np.mean(np.abs(correction_values)))
        rel_span = float(correction_span / max(mean_abs_correction, 1.0e-18))
    else:
        correction_span = math.nan
        rel_span = math.inf

    return {
        "tail_windows": tail_windows,
        "tail_normalized_correction_span": correction_span,
        "tail_normalized_correction_rel_span": rel_span,
        "tail_scaled_rel_std_max": float(np.max(rel_std_values)) if rel_std_values.size else math.inf,
    }


# Function: build a compact virial-style ratio inventory from retained profile energies.

def build_energy_ratio_pack(
    radius: np.ndarray,
    profile: np.ndarray,
    profile_prime: np.ndarray,
    beta1: float,
) -> dict[str, float]:
    """Return one retained energy-ratio pack for the virial-style route."""
    epsilon_beta = float(1.0 - beta1**2)
    volume = 4.0 * math.pi * np.square(radius)
    gradient_energy = float(np.trapezoid(volume * 0.5 * np.square(profile_prime), radius))
    mass_gap_energy = float(np.trapezoid(volume * 0.5 * epsilon_beta * np.square(profile), radius))
    cubic_energy = float(np.trapezoid(volume * np.power(profile, 3), radius))
    quartic_energy = float(np.trapezoid(volume * 0.25 * np.power(profile, 4), radius))
    ratios = {
        "gradient_over_mass_gap": gradient_energy / mass_gap_energy,
        "cubic_over_mass_gap": cubic_energy / mass_gap_energy,
        "cubic_over_gradient": cubic_energy / gradient_energy,
        "cubic_over_gradient_plus_mass_gap": cubic_energy / (gradient_energy + mass_gap_energy),
        "gradient_plus_quartic_over_mass_gap": (gradient_energy + quartic_energy) / mass_gap_energy,
    }
    best_three_halves_abs_error = min(abs(value - 1.5) for value in ratios.values())
    return {
        "gradient_energy": gradient_energy,
        "mass_gap_energy": mass_gap_energy,
        "cubic_energy": cubic_energy,
        "quartic_energy": quartic_energy,
        "gradient_over_mass_gap": float(ratios["gradient_over_mass_gap"]),
        "cubic_over_mass_gap": float(ratios["cubic_over_mass_gap"]),
        "cubic_over_gradient": float(ratios["cubic_over_gradient"]),
        "cubic_over_gradient_plus_mass_gap": float(ratios["cubic_over_gradient_plus_mass_gap"]),
        "gradient_plus_quartic_over_mass_gap": float(ratios["gradient_plus_quartic_over_mass_gap"]),
        "virial_best_three_halves_abs_error": float(best_three_halves_abs_error),
    }


# Function: build the retained profile-sensitive q_star correction audit pack.

def build_scalar_proxy_profile_sensitive_qstar_correction_pack() -> dict:
    """Return one retained profile-sensitive q_star correction audit pack."""
    for path in (PRIOR_AUDIT, QBALL_BRANCH_REFRESH):
        require(path)

    prior_summary = read_json(PRIOR_AUDIT)["summary"]
    qball_branch_refresh = read_json(QBALL_BRANCH_REFRESH)
    scalar_ground_state = extract_scalar_ground_state(qball_branch_refresh)

    q_exact = float(prior_summary["q_exact_over_m0"])
    q_star = float(prior_summary["q_star_over_m0"])
    beta1 = float(prior_summary["beta1"])
    epsilon_beta = float(prior_summary["epsilon_beta"])
    c1_fit = float(prior_summary["q_star_correction_c1_fit"])
    delta_kappa_squared_rel_observed = float(prior_summary["delta_kappa_squared_rel"])
    q_blind = float(prior_summary["q_blind_over_m0"])

    qball_module = load_qball_module()
    radius, profile, profile_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    radius = np.asarray(radius, dtype=float)
    profile = np.asarray(profile, dtype=float)
    profile_prime = np.asarray(profile_prime, dtype=float)
    density = np.square(profile)
    weight = density * np.square(radius)
    norm = float(np.trapezoid(weight, radius))

    three_halves_linear_c1 = -1.5
    q_linear_three_halves = float(q_star * (1.0 + three_halves_linear_c1 * epsilon_beta))
    q_linear_three_halves_rel_error = float(abs(q_linear_three_halves - q_exact) / q_exact)
    f_linear_three_halves = float(form_factor(radius, weight, norm, q_linear_three_halves))
    alpha_linear_three_halves = float((f_linear_three_halves**2) / FOUR_PI)
    alpha_linear_three_halves_rel_error = float(abs(alpha_linear_three_halves - ALPHA_TARGET) / ALPHA_TARGET)

    cubic_q_squared_coefficient = 3.0
    q_cubic_sqrt = float(q_star * math.sqrt(max(0.0, 1.0 - cubic_q_squared_coefficient * epsilon_beta)))
    q_cubic_sqrt_rel_error = float(abs(q_cubic_sqrt - q_exact) / q_exact)
    f_cubic_sqrt = float(form_factor(radius, weight, norm, q_cubic_sqrt))
    alpha_cubic_sqrt = float((f_cubic_sqrt**2) / FOUR_PI)
    alpha_cubic_sqrt_rel_error = float(abs(alpha_cubic_sqrt - ALPHA_TARGET) / ALPHA_TARGET)

    delta_kappa_squared_rel_from_cubic = float(-cubic_q_squared_coefficient * epsilon_beta)
    q_squared_correction_coeff_fit = float(-delta_kappa_squared_rel_observed / epsilon_beta)
    q_squared_correction_coeff_rel_error_vs_cubic = float(
        abs(q_squared_correction_coeff_fit - cubic_q_squared_coefficient) / cubic_q_squared_coefficient
    )

    direct_fourier_pack = evaluate_form_factor_derivatives(radius, weight, norm, q_star)
    target_form_factor = float(math.sqrt(FOUR_PI * ALPHA_TARGET))
    dq_exact = float(q_exact - q_star)
    dq_direct_linear = float(
        (target_form_factor - direct_fourier_pack["F_q"]) / direct_fourier_pack["F_prime_q"]
    )
    dq_direct_quadratic = solve_small_quadratic_root(
        0.5 * direct_fourier_pack["F_double_prime_q"],
        direct_fourier_pack["F_prime_q"],
        direct_fourier_pack["F_q"] - target_form_factor,
    )
    c1_direct_linear = float(dq_direct_linear / (q_star * epsilon_beta))
    c1_direct_quadratic = float(dq_direct_quadratic / (q_star * epsilon_beta))

    tail_pack = build_tail_fit_pack(radius, profile, float(q_star**2))
    energy_pack = build_energy_ratio_pack(radius, profile, profile_prime, beta1)

    three_halves_linear_law_available_now = bool(q_linear_three_halves_rel_error <= 2.0e-4)
    cubic_sqrt_leading_law_available_now = bool(q_cubic_sqrt_rel_error <= 1.0e-4)
    practical_matching_law_available_now = bool(alpha_cubic_sqrt_rel_error <= 5.0e-4)
    mexican_hat_cubic_route_supported_now = bool(
        q_squared_correction_coeff_rel_error_vs_cubic <= 2.0e-2
    )
    direct_fourier_route_supported_now = bool(abs(c1_direct_quadratic - c1_fit) <= 5.0e-5)
    direct_fourier_route_target_dependent_now = True
    evanescent_tail_route_supported_now = bool(
        tail_pack["tail_normalized_correction_rel_span"] <= 1.0e-1
        and tail_pack["tail_scaled_rel_std_max"] <= 2.5e-1
    )
    virial_route_supported_now = bool(energy_pack["virial_best_three_halves_abs_error"] <= 1.0e-1)
    exact_three_halves_first_principles_derivation_available_now = False
    three_halves_nlo_gap_remaining_now = bool(abs(c1_fit - three_halves_linear_c1) > 1.0e-2)
    old_blind_overlap_bridge_still_exact_now = bool(abs(q_exact - q_blind) <= 1.0e-12)

    return {
        "q_exact_over_m0": q_exact,
        "q_star_over_m0": q_star,
        "q_blind_over_m0": q_blind,
        "beta1": beta1,
        "epsilon_beta": epsilon_beta,
        "c1_fit": c1_fit,
        "three_halves_linear_c1": three_halves_linear_c1,
        "c1_abs_error_vs_three_halves": float(abs(c1_fit - three_halves_linear_c1)),
        "c1_rel_error_vs_three_halves": float(abs(c1_fit - three_halves_linear_c1) / abs(c1_fit)),
        "q_linear_three_halves_over_m0": q_linear_three_halves,
        "q_linear_three_halves_rel_error": q_linear_three_halves_rel_error,
        "F_linear_three_halves": f_linear_three_halves,
        "alpha_linear_three_halves": alpha_linear_three_halves,
        "alpha_linear_three_halves_rel_error": alpha_linear_three_halves_rel_error,
        "q_cubic_sqrt_over_m0": q_cubic_sqrt,
        "q_cubic_sqrt_rel_error": q_cubic_sqrt_rel_error,
        "F_cubic_sqrt": f_cubic_sqrt,
        "alpha_cubic_sqrt": alpha_cubic_sqrt,
        "alpha_cubic_sqrt_rel_error": alpha_cubic_sqrt_rel_error,
        "cubic_q_squared_coefficient": cubic_q_squared_coefficient,
        "delta_kappa_squared_rel_observed": delta_kappa_squared_rel_observed,
        "delta_kappa_squared_rel_from_cubic": delta_kappa_squared_rel_from_cubic,
        "q_squared_correction_coeff_fit": q_squared_correction_coeff_fit,
        "q_squared_correction_coeff_rel_error_vs_cubic": q_squared_correction_coeff_rel_error_vs_cubic,
        "F_q_star": float(direct_fourier_pack["F_q"]),
        "F_prime_q_star": float(direct_fourier_pack["F_prime_q"]),
        "F_double_prime_q_star": float(direct_fourier_pack["F_double_prime_q"]),
        "target_form_factor": target_form_factor,
        "dq_exact": dq_exact,
        "dq_direct_linear": dq_direct_linear,
        "dq_direct_quadratic": dq_direct_quadratic,
        "c1_direct_linear": c1_direct_linear,
        "c1_direct_quadratic": c1_direct_quadratic,
        "c1_direct_quadratic_abs_error_vs_fit": float(abs(c1_direct_quadratic - c1_fit)),
        "c1_direct_quadratic_abs_error_vs_three_halves": float(abs(c1_direct_quadratic - three_halves_linear_c1)),
        "tail_windows": tail_pack["tail_windows"],
        "tail_normalized_correction_span": float(tail_pack["tail_normalized_correction_span"]),
        "tail_normalized_correction_rel_span": float(tail_pack["tail_normalized_correction_rel_span"]),
        "tail_scaled_rel_std_max": float(tail_pack["tail_scaled_rel_std_max"]),
        "gradient_energy": float(energy_pack["gradient_energy"]),
        "mass_gap_energy": float(energy_pack["mass_gap_energy"]),
        "cubic_energy": float(energy_pack["cubic_energy"]),
        "quartic_energy": float(energy_pack["quartic_energy"]),
        "gradient_over_mass_gap": float(energy_pack["gradient_over_mass_gap"]),
        "cubic_over_mass_gap": float(energy_pack["cubic_over_mass_gap"]),
        "cubic_over_gradient": float(energy_pack["cubic_over_gradient"]),
        "cubic_over_gradient_plus_mass_gap": float(energy_pack["cubic_over_gradient_plus_mass_gap"]),
        "gradient_plus_quartic_over_mass_gap": float(energy_pack["gradient_plus_quartic_over_mass_gap"]),
        "virial_best_three_halves_abs_error": float(energy_pack["virial_best_three_halves_abs_error"]),
        "three_halves_linear_law_available_now": three_halves_linear_law_available_now,
        "cubic_sqrt_leading_law_available_now": cubic_sqrt_leading_law_available_now,
        "practical_matching_law_available_now": practical_matching_law_available_now,
        "mexican_hat_cubic_route_supported_now": mexican_hat_cubic_route_supported_now,
        "direct_fourier_route_supported_now": direct_fourier_route_supported_now,
        "direct_fourier_route_target_dependent_now": direct_fourier_route_target_dependent_now,
        "evanescent_tail_route_supported_now": evanescent_tail_route_supported_now,
        "virial_route_supported_now": virial_route_supported_now,
        "exact_three_halves_first_principles_derivation_available_now": exact_three_halves_first_principles_derivation_available_now,
        "three_halves_nlo_gap_remaining_now": three_halves_nlo_gap_remaining_now,
        "old_blind_overlap_bridge_still_exact_now": old_blind_overlap_bridge_still_exact_now,
    }


# Function: allow one CLI smoke run for local verification.

def main() -> None:
    """Print the retained profile-sensitive q_star correction pack as JSON."""
    import json

    print(json.dumps(build_scalar_proxy_profile_sensitive_qstar_correction_pack(), ensure_ascii=False, indent=2))


# Function: run the helper when invoked as one CLI script.

if __name__ == "__main__":
    main()
