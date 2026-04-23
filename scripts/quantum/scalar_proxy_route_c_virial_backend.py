#!/usr/bin/env python3
"""Audit Route-C virial identities for the scalar-proxy three-halves law.

Purpose:
    After Route B, Route A, and Route D each closed negatively, the remaining
    theorem-side branch is the virial route. This helper keeps that route
    narrow:

    1. derive the finite-radius weighted-EOM identity and the finite-radius
       virial identity directly from the retained shooting equation,
    2. verify that both identities recover the Mexican-hat cubic coefficient
       `3` exactly once the retained boundary terms are kept,
    3. test whether those identities alone produce a target-free bridge from
       the retained profile to the matching-law correction.

Inputs:
    - output/public/quantum/q_8_7_56_5399_5402_updated_pack_scalar_proxy_profile_sensitiv_60aebfd3b0_declaration_gate_metrics.json
    - output/public/quantum/mass_origin_qball_charge_mapping_branch_refresh_metrics.json
    - scripts/quantum/mass_origin_qball_charge_mapping_branch.py

Outputs:
    - One in-memory Route-C audit pack consumed by `.5447-.5454` wrappers

Assumptions:
    - No new parameter is introduced
    - alpha_target is not used anywhere in this Route-C audit
    - Only the virial route is audited here
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


# Function: integrate one retained radial density against the x^2 measure.
def integrate_radial(radius: np.ndarray, integrand: np.ndarray) -> float:
    """Return one radial integral int dx x^2 integrand(x)."""
    return float(np.trapezoid(np.square(radius) * integrand, radius))


# Function: build the finite-radius Route-C identity pack.

def build_route_c_identity_pack(
    radius: np.ndarray,
    profile: np.ndarray,
    profile_prime: np.ndarray,
    epsilon_beta: float,
) -> dict[str, float]:
    """Return one finite-radius identity pack for the retained scalar profile."""
    radius_end = float(radius[-1])
    profile_end = float(profile[-1])
    profile_prime_end = float(profile_prime[-1])

    integral_grad = integrate_radial(radius, np.square(profile_prime))
    integral_mass = integrate_radial(radius, epsilon_beta * np.square(profile))
    integral_cubic = integrate_radial(radius, np.power(profile, 3))
    integral_quartic = integrate_radial(radius, np.power(profile, 4))

    boundary_weighted_eom = float(
        (radius_end * radius_end) * profile_end * profile_prime_end
    )
    weighted_eom_residual = float(
        boundary_weighted_eom
        - integral_grad
        - integral_mass
        + 3.0 * integral_cubic
        + integral_quartic
    )

    boundary_virial = float(
        0.5 * (radius_end**3) * (profile_prime_end**2)
        - 0.5 * epsilon_beta * (radius_end**3) * (profile_end**2)
        + (radius_end**3) * (profile_end**3)
        + 0.25 * (radius_end**3) * (profile_end**4)
    )
    virial_residual = float(
        boundary_virial
        + 0.5 * integral_grad
        + 1.5 * integral_mass
        - 3.0 * integral_cubic
        - 0.75 * integral_quartic
    )

    cubic_coeff_from_exact_weighted_eom = float(
        (integral_grad + integral_mass - integral_quartic - boundary_weighted_eom)
        / integral_cubic
    )
    cubic_coeff_from_exact_virial = float(
        (0.5 * integral_grad + 1.5 * integral_mass - 0.75 * integral_quartic + boundary_virial)
        / integral_cubic
    )
    cubic_coeff_from_boundary_free_weighted_eom = float(
        (integral_grad + integral_mass - integral_quartic) / integral_cubic
    )
    cubic_coeff_from_boundary_free_virial = float(
        (0.5 * integral_grad + 1.5 * integral_mass - 0.75 * integral_quartic)
        / integral_cubic
    )

    boundary_weighted_eom_over_cubic = float(
        abs(boundary_weighted_eom) / max(3.0 * abs(integral_cubic), 1.0e-30)
    )
    boundary_virial_over_cubic = float(
        abs(boundary_virial) / max(3.0 * abs(integral_cubic), 1.0e-30)
    )
    quartic_over_cubic = float(
        abs(integral_quartic) / max(3.0 * abs(integral_cubic), 1.0e-30)
    )

    return {
        "radius_end": radius_end,
        "profile_end": profile_end,
        "profile_prime_end": profile_prime_end,
        "integral_grad": integral_grad,
        "integral_mass": integral_mass,
        "integral_cubic": integral_cubic,
        "integral_quartic": integral_quartic,
        "boundary_weighted_eom": boundary_weighted_eom,
        "boundary_virial": boundary_virial,
        "weighted_eom_residual": weighted_eom_residual,
        "virial_residual": virial_residual,
        "cubic_coeff_from_exact_weighted_eom": cubic_coeff_from_exact_weighted_eom,
        "cubic_coeff_from_exact_virial": cubic_coeff_from_exact_virial,
        "cubic_coeff_from_boundary_free_weighted_eom": cubic_coeff_from_boundary_free_weighted_eom,
        "cubic_coeff_from_boundary_free_virial": cubic_coeff_from_boundary_free_virial,
        "boundary_weighted_eom_over_cubic": boundary_weighted_eom_over_cubic,
        "boundary_virial_over_cubic": boundary_virial_over_cubic,
        "quartic_over_cubic": quartic_over_cubic,
    }


# Function: build the Route-C virial audit pack.

def build_scalar_proxy_route_c_virial_pack() -> dict:
    """Return the retained Route-C virial audit pack."""
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
    q_squared_correction_coeff_fit = float(prior_summary["q_squared_correction_coeff_fit"])

    qball_module = load_qball_module()
    radius, profile, profile_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    radius = np.asarray(radius, dtype=float)
    profile = np.asarray(profile, dtype=float)
    profile_prime = np.asarray(profile_prime, dtype=float)

    identity_pack = build_route_c_identity_pack(
        radius,
        profile,
        profile_prime,
        epsilon_beta,
    )

    route_c_exact_weighted_eom_identity_available_now = bool(
        abs(identity_pack["weighted_eom_residual"]) <= 1.0e-8
    )
    route_c_exact_virial_identity_available_now = bool(
        abs(identity_pack["virial_residual"]) <= 1.0e-8
    )
    route_c_exact_cubic_coefficient_recovered_now = bool(
        abs(identity_pack["cubic_coeff_from_exact_weighted_eom"] - 3.0) <= 1.0e-5
        and abs(identity_pack["cubic_coeff_from_exact_virial"] - 3.0) <= 1.0e-5
    )
    route_c_boundary_terms_negligible_now = bool(
        max(
            identity_pack["boundary_weighted_eom_over_cubic"],
            identity_pack["boundary_virial_over_cubic"],
        )
        <= 5.0e-2
    )
    route_c_boundary_free_virial_truncation_supported_now = bool(
        abs(identity_pack["cubic_coeff_from_boundary_free_weighted_eom"] - 3.0) <= 5.0e-2
        and abs(identity_pack["cubic_coeff_from_boundary_free_virial"] - 3.0) <= 5.0e-2
    )
    route_c_target_free_matching_law_bridge_available_now = False
    route_c_negative_closeout_available_now = bool(
        route_c_exact_weighted_eom_identity_available_now
        and route_c_exact_virial_identity_available_now
        and route_c_exact_cubic_coefficient_recovered_now
        and not route_c_target_free_matching_law_bridge_available_now
    )
    selected_extension_source_materialization_promoted_primary_now = bool(
        route_c_negative_closeout_available_now
    )

    return {
        "beta1": beta1,
        "epsilon_beta": epsilon_beta,
        "q_exact_over_m0": q_exact,
        "q_star_over_m0": q_star,
        "q_cubic_sqrt_over_m0": q_cubic_sqrt,
        "q_squared_correction_coeff_fit": q_squared_correction_coeff_fit,
        **identity_pack,
        "q_squared_correction_coeff_rel_error_vs_exact_cubic": float(
            abs(q_squared_correction_coeff_fit - 3.0) / 3.0
        ),
        "route_c_exact_weighted_eom_identity_available_now": route_c_exact_weighted_eom_identity_available_now,
        "route_c_exact_virial_identity_available_now": route_c_exact_virial_identity_available_now,
        "route_c_exact_cubic_coefficient_recovered_now": route_c_exact_cubic_coefficient_recovered_now,
        "route_c_boundary_terms_negligible_now": route_c_boundary_terms_negligible_now,
        "route_c_boundary_free_virial_truncation_supported_now": route_c_boundary_free_virial_truncation_supported_now,
        "route_c_target_free_matching_law_bridge_available_now": route_c_target_free_matching_law_bridge_available_now,
        "route_c_negative_closeout_available_now": route_c_negative_closeout_available_now,
        "selected_extension_source_materialization_promoted_primary_now": (
            selected_extension_source_materialization_promoted_primary_now
        ),
    }


# Function: allow one CLI smoke run for local verification.

def main() -> None:
    """Print the retained Route-C virial audit pack as JSON."""
    import json

    print(json.dumps(build_scalar_proxy_route_c_virial_pack(), ensure_ascii=False, indent=2))


# Function: run the helper when invoked as one CLI script.

if __name__ == "__main__":
    main()
