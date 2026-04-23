#!/usr/bin/env python3
"""Audit the Route-B `kappa_eff` derivation for the scalar-proxy three-halves law.

Purpose:
    The current active blocker is a target-free first-principles derivation of
    the retained leading law

        q_corrected = q_star * sqrt(1 - 3 * epsilon_beta).

    This helper runs only Route B from the expert instruction:
    track how the Mexican-hat cubic term modifies the large-x evanescent tail
    and test whether that modification can be reinterpreted as one constant
    decay-shift `kappa_eff`.

Inputs:
    - output/public/quantum/q_8_7_56_5399_5402_updated_pack_scalar_proxy_profile_sensitiv_60aebfd3b0_declaration_gate_metrics.json
    - output/public/quantum/mass_origin_qball_charge_mapping_branch_refresh_metrics.json
    - scripts/quantum/mass_origin_qball_charge_mapping_branch.py

Outputs:
    - One in-memory Route-B audit pack consumed by `.5407-.5414` wrappers

Assumptions:
    - No new parameter is introduced
    - `alpha_target` is not used anywhere in the derivation
    - Only the Route-B `kappa_eff` path is audited in this helper
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


# Function: choose one best tail window by one scalar quality key.
def choose_best_window(windows: list[dict[str, float]], key: str) -> dict[str, float]:
    """Return the numerically cleanest retained tail window."""
    return min(windows, key=lambda row_data: float(row_data[key]))


# Function: fit the leading free-tail amplitude on late-x windows.

def build_tail_amplitude_windows(
    radius: np.ndarray,
    profile: np.ndarray,
    kappa: float,
) -> list[dict[str, float]]:
    """Return free-tail amplitude fits for several retained late-x cutoffs."""
    windows: list[dict[str, float]] = []
    for cutoff in (15.0, 20.0, 25.0, 28.0):
        mask = radius > cutoff
        if int(np.count_nonzero(mask)) < 5:
            continue

        tail_radius = radius[mask]
        tail_profile = profile[mask]
        scaled_tail = tail_radius * np.exp(kappa * tail_radius) * tail_profile
        amplitude_fit = float(np.mean(scaled_tail))
        amplitude_rel_std = float(
            np.std(scaled_tail) / max(abs(amplitude_fit), 1.0e-18)
        )
        windows.append(
            {
                "cutoff": float(cutoff),
                "amplitude_fit": amplitude_fit,
                "amplitude_rel_std": amplitude_rel_std,
            }
        )

    return windows


# Function: fit an effective late-tail decay constant from log(x y).

def build_kappa_fit_windows(
    radius: np.ndarray,
    profile: np.ndarray,
) -> list[dict[str, float]]:
    """Return late-tail log fits for one constant effective decay rate."""
    windows: list[dict[str, float]] = []
    for cutoff in (15.0, 20.0, 25.0, 28.0):
        mask = (radius > cutoff) & (np.abs(profile) > 1.0e-30)
        if int(np.count_nonzero(mask)) < 5:
            continue

        tail_radius = radius[mask]
        tail_u = tail_radius * np.abs(profile[mask])
        log_tail_u = np.log(tail_u)
        design = np.column_stack((np.ones_like(tail_radius), -tail_radius))
        intercept, kappa_eff_fit = np.linalg.lstsq(design, log_tail_u, rcond=None)[0]
        residual = log_tail_u - (intercept - kappa_eff_fit * tail_radius)
        windows.append(
            {
                "cutoff": float(cutoff),
                "kappa_eff_fit": float(kappa_eff_fit),
                "log_fit_std": float(np.std(residual)),
            }
        )

    return windows


# Function: compare the retained residual tail against the Route-B particular basis.

def build_particular_windows(
    radius: np.ndarray,
    profile: np.ndarray,
    kappa: float,
    amplitude_fit: float,
    predicted_coeff: float,
) -> list[dict[str, float]]:
    """Return late-tail fits of the Route-B particular correction coefficient."""
    free_tail = amplitude_fit * np.exp(-kappa * radius) / radius
    residual = profile - free_tail
    windows: list[dict[str, float]] = []
    for cutoff in (15.0, 20.0, 25.0, 28.0):
        mask = radius > cutoff
        if int(np.count_nonzero(mask)) < 5:
            continue

        tail_radius = radius[mask]
        scaled_particular = residual[mask] * np.square(tail_radius) * np.exp(2.0 * kappa * tail_radius)
        coeff_fit = float(np.mean(scaled_particular))
        coeff_rel_std = float(
            np.std(scaled_particular) / max(abs(coeff_fit), 1.0e-18)
        )
        coeff_rel_error_vs_prediction = float(
            abs(coeff_fit - predicted_coeff) / max(abs(predicted_coeff), 1.0e-18)
        )
        windows.append(
            {
                "cutoff": float(cutoff),
                "particular_coeff_fit": coeff_fit,
                "particular_coeff_rel_std": coeff_rel_std,
                "particular_coeff_rel_error_vs_prediction": coeff_rel_error_vs_prediction,
            }
        )

    return windows


# Function: build the Route-B `kappa_eff` derivation pack.

def build_scalar_proxy_three_halves_route_b_pack() -> dict:
    """Return the retained Route-B `kappa_eff` derivation audit pack."""
    for path in (PRIOR_AUDIT, QBALL_BRANCH_REFRESH):
        require(path)

    prior_summary = read_json(PRIOR_AUDIT)["summary"]
    qball_branch_refresh = read_json(QBALL_BRANCH_REFRESH)
    scalar_ground_state = extract_scalar_ground_state(qball_branch_refresh)

    beta1 = float(prior_summary["beta1"])
    epsilon_beta = float(prior_summary["epsilon_beta"])
    q_exact = float(prior_summary["q_exact_over_m0"])
    q_star = float(prior_summary["q_star_over_m0"])
    q_cubic_sqrt = float(prior_summary["q_cubic_sqrt_over_m0"])
    kappa = float(math.sqrt(epsilon_beta))
    q_star_squared = float(q_star * q_star)
    q_exact_squared = float(q_exact * q_exact)
    q_cubic_sqrt_squared = float(q_cubic_sqrt * q_cubic_sqrt)

    qball_module = load_qball_module()
    radius, profile, _ = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    radius = np.asarray(radius, dtype=float)
    profile = np.asarray(profile, dtype=float)

    amplitude_windows = build_tail_amplitude_windows(radius, profile, kappa)
    best_amplitude_window = choose_best_window(amplitude_windows, "amplitude_rel_std")
    amplitude_fit = float(best_amplitude_window["amplitude_fit"])

    source_leading_prefactor = float(-3.0 * amplitude_fit * amplitude_fit)
    operator_basis_leading_prefactor = float(3.0 * kappa * kappa)
    particular_coeff_exact = float(source_leading_prefactor / operator_basis_leading_prefactor)

    particular_windows = build_particular_windows(
        radius,
        profile,
        kappa,
        amplitude_fit,
        particular_coeff_exact,
    )
    best_particular_window = choose_best_window(
        particular_windows,
        "particular_coeff_rel_error_vs_prediction",
    )

    kappa_fit_windows = build_kappa_fit_windows(radius, profile)
    best_kappa_fit_window = choose_best_window(kappa_fit_windows, "log_fit_std")
    kappa_eff_fit = float(best_kappa_fit_window["kappa_eff_fit"])
    delta_kappa_fit_abs = float(kappa_eff_fit - kappa)
    delta_kappa_fit_rel = float(delta_kappa_fit_abs / max(abs(kappa), 1.0e-18))

    delta_kappa_required_rel = float(-3.0 * epsilon_beta)
    delta_kappa_required_abs = float(kappa * delta_kappa_required_rel)
    delta_kappa_fit_rel_error_vs_required = float(
        abs(delta_kappa_fit_abs - delta_kappa_required_abs)
        / max(abs(delta_kappa_required_abs), 1.0e-18)
    )

    q_squared_residual_vs_cubic_sqrt = float(q_exact_squared - q_cubic_sqrt_squared)

    route_b_asymptotic_algebra_available_now = True
    route_b_particular_tail_cross_check_supported_now = bool(
        best_particular_window["particular_coeff_rel_error_vs_prediction"] <= 2.5e-1
        and best_particular_window["particular_coeff_rel_std"] <= 1.0
    )
    route_b_constant_kappa_shift_supported_now = bool(
        delta_kappa_fit_rel_error_vs_required <= 2.5e-1
        and best_kappa_fit_window["log_fit_std"] <= 5.0e-3
    )
    route_b_exponent_mismatch_no_go_theorem_available_now = True
    route_b_target_free_three_halves_derivation_available_now = False
    route_b_negative_closeout_available_now = bool(
        route_b_asymptotic_algebra_available_now
        and route_b_exponent_mismatch_no_go_theorem_available_now
        and not route_b_constant_kappa_shift_supported_now
    )
    route_a_eom_perturbation_promoted_next_now = bool(route_b_negative_closeout_available_now)
    route_d_profile_moment_kept_secondary_now = True
    route_c_virial_kept_reserve_now = True

    return {
        "beta1": beta1,
        "epsilon_beta": epsilon_beta,
        "kappa": kappa,
        "q_exact_over_m0": q_exact,
        "q_star_over_m0": q_star,
        "q_cubic_sqrt_over_m0": q_cubic_sqrt,
        "q_star_squared": q_star_squared,
        "q_exact_squared": q_exact_squared,
        "q_cubic_sqrt_squared": q_cubic_sqrt_squared,
        "q_squared_residual_vs_cubic_sqrt": q_squared_residual_vs_cubic_sqrt,
        "amplitude_windows": amplitude_windows,
        "best_amplitude_cutoff": float(best_amplitude_window["cutoff"]),
        "amplitude_fit": amplitude_fit,
        "source_leading_prefactor": source_leading_prefactor,
        "operator_basis_leading_prefactor": operator_basis_leading_prefactor,
        "particular_coeff_exact": particular_coeff_exact,
        "particular_windows": particular_windows,
        "best_particular_cutoff": float(best_particular_window["cutoff"]),
        "best_particular_coeff_fit": float(best_particular_window["particular_coeff_fit"]),
        "best_particular_coeff_rel_std": float(best_particular_window["particular_coeff_rel_std"]),
        "best_particular_coeff_rel_error_vs_prediction": float(
            best_particular_window["particular_coeff_rel_error_vs_prediction"]
        ),
        "kappa_fit_windows": kappa_fit_windows,
        "best_kappa_fit_cutoff": float(best_kappa_fit_window["cutoff"]),
        "kappa_eff_fit": kappa_eff_fit,
        "best_kappa_log_fit_std": float(best_kappa_fit_window["log_fit_std"]),
        "delta_kappa_fit_abs": delta_kappa_fit_abs,
        "delta_kappa_fit_rel": delta_kappa_fit_rel,
        "delta_kappa_required_abs": delta_kappa_required_abs,
        "delta_kappa_required_rel": delta_kappa_required_rel,
        "delta_kappa_fit_rel_error_vs_required": delta_kappa_fit_rel_error_vs_required,
        "route_b_asymptotic_algebra_available_now": route_b_asymptotic_algebra_available_now,
        "route_b_particular_tail_cross_check_supported_now": route_b_particular_tail_cross_check_supported_now,
        "route_b_constant_kappa_shift_supported_now": route_b_constant_kappa_shift_supported_now,
        "route_b_exponent_mismatch_no_go_theorem_available_now": route_b_exponent_mismatch_no_go_theorem_available_now,
        "route_b_target_free_three_halves_derivation_available_now": route_b_target_free_three_halves_derivation_available_now,
        "route_b_negative_closeout_available_now": route_b_negative_closeout_available_now,
        "route_a_eom_perturbation_promoted_next_now": route_a_eom_perturbation_promoted_next_now,
        "route_d_profile_moment_kept_secondary_now": route_d_profile_moment_kept_secondary_now,
        "route_c_virial_kept_reserve_now": route_c_virial_kept_reserve_now,
    }


# Function: allow one CLI smoke run for local verification.

def main() -> None:
    """Print the retained Route-B audit pack as JSON."""
    import json

    print(json.dumps(build_scalar_proxy_three_halves_route_b_pack(), ensure_ascii=False, indent=2))


# Function: run the helper when invoked as one CLI script.

if __name__ == "__main__":
    main()
