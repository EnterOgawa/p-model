#!/usr/bin/env python3
"""Audit Route-D profile-moment derivation for the scalar-proxy three-halves law.

Purpose:
    After Route B and Route A both closed negatively, the remaining theorem-side
    branch is the direct profile-moment route:

        F(q) = 1 - q^2 <r^2>/6 + q^4 <r^4>/120 - q^6 <r^6>/5040 + ...

    This helper checks whether the retained matching scales live inside a
    controlled low-q moment domain and whether low-order moment truncations can
    honestly reproduce the retained exact form factor without importing any new
    parameter.

Inputs:
    - output/public/quantum/q_8_7_56_5399_5402_updated_pack_scalar_proxy_profile_sensitiv_60aebfd3b0_declaration_gate_metrics.json
    - output/public/quantum/mass_origin_qball_charge_mapping_branch_refresh_metrics.json
    - scripts/quantum/mass_origin_qball_charge_mapping_branch.py

Outputs:
    - One in-memory Route-D audit pack consumed by `.5439-.5446` wrappers

Assumptions:
    - No new parameter is introduced
    - alpha_target is not used to define the route-D no-go theorem
    - Only the direct profile-moment route is audited in this helper
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
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import form_factor
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


# Function: return one normalized even radial moment of the retained density.
def compute_even_moment(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    even_power: int,
) -> float:
    """Return one normalized even moment <r^n> with n=even_power."""
    return float(np.trapezoid(weight * np.power(radius, even_power), radius) / norm)


# Function: return one truncated Taylor approximation of the spherical form factor.

def truncated_form_factor(moment_map: dict[int, float], q_value: float, max_even_power: int) -> float:
    """Return one low-order spherical-Bessel moment truncation."""
    truncated_value = 1.0
    for moment_power in range(2, max_even_power + 1, 2):
        order_index = moment_power // 2
        coefficient = ((-1) ** order_index) / math.factorial(moment_power + 1)
        truncated_value += coefficient * (float(q_value) ** moment_power) * float(moment_map[moment_power])

    return float(truncated_value)


# Function: build one term-by-term profile-moment expansion table at one q value.

def build_term_pack(moment_map: dict[int, float], q_value: float) -> dict[str, float]:
    """Return one compact term table for the first retained moment orders."""
    term_q2 = float(-(q_value**2) * moment_map[2] / math.factorial(3))
    term_q4 = float((q_value**4) * moment_map[4] / math.factorial(5))
    term_q6 = float(-(q_value**6) * moment_map[6] / math.factorial(7))
    term_q8 = float((q_value**8) * moment_map[8] / math.factorial(9))
    return {
        "term_q2": term_q2,
        "term_q4": term_q4,
        "term_q6": term_q6,
        "term_q8": term_q8,
        "control_parameter_q2_abs": abs(term_q2),
        "largest_term_abs": float(max(abs(term_q2), abs(term_q4), abs(term_q6), abs(term_q8))),
    }


# Function: build one truncation-error pack at one q value.

def build_truncation_error_pack(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    moment_map: dict[int, float],
    q_value: float,
) -> dict[str, float]:
    """Return one exact-versus-truncated comparison at one retained q value."""
    exact_value = float(form_factor(radius, weight, norm, float(q_value)))
    truncated_q2 = truncated_form_factor(moment_map, q_value, 2)
    truncated_q4 = truncated_form_factor(moment_map, q_value, 4)
    truncated_q6 = truncated_form_factor(moment_map, q_value, 6)
    truncated_q8 = truncated_form_factor(moment_map, q_value, 8)
    return {
        "F_exact": exact_value,
        "F_trunc_q2": truncated_q2,
        "F_trunc_q4": truncated_q4,
        "F_trunc_q6": truncated_q6,
        "F_trunc_q8": truncated_q8,
        "F_trunc_q2_abs_error": float(abs(truncated_q2 - exact_value)),
        "F_trunc_q4_abs_error": float(abs(truncated_q4 - exact_value)),
        "F_trunc_q6_abs_error": float(abs(truncated_q6 - exact_value)),
        "F_trunc_q8_abs_error": float(abs(truncated_q8 - exact_value)),
        "best_truncation_abs_error": float(
            min(
                abs(truncated_q2 - exact_value),
                abs(truncated_q4 - exact_value),
                abs(truncated_q6 - exact_value),
                abs(truncated_q8 - exact_value),
            )
        ),
    }


# Function: build the Route-D profile-moment audit pack.

def build_scalar_proxy_route_d_profile_moment_pack() -> dict:
    """Return the retained Route-D profile-moment audit pack."""
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

    moment_map = {
        2: compute_even_moment(radius, weight, norm, 2),
        4: compute_even_moment(radius, weight, norm, 4),
        6: compute_even_moment(radius, weight, norm, 6),
        8: compute_even_moment(radius, weight, norm, 8),
    }
    scaled_moment_map = {
        2: float(epsilon_beta * moment_map[2]),
        4: float((epsilon_beta**2) * moment_map[4]),
        6: float((epsilon_beta**3) * moment_map[6]),
        8: float((epsilon_beta**4) * moment_map[8]),
    }

    q_star_term_pack = build_term_pack(moment_map, q_star)
    q_exact_term_pack = build_term_pack(moment_map, q_exact)
    q_cubic_sqrt_term_pack = build_term_pack(moment_map, q_cubic_sqrt)

    q_star_error_pack = build_truncation_error_pack(radius, weight, norm, moment_map, q_star)
    q_exact_error_pack = build_truncation_error_pack(radius, weight, norm, moment_map, q_exact)
    q_cubic_sqrt_error_pack = build_truncation_error_pack(radius, weight, norm, moment_map, q_cubic_sqrt)

    moment_scaling_ratio_m4_over_m2sq = float(moment_map[4] / max(moment_map[2] ** 2, 1.0e-30))
    moment_scaling_ratio_m6_over_m2cu = float(moment_map[6] / max(moment_map[2] ** 3, 1.0e-30))
    moment_scaling_ratio_m8_over_m2qu = float(moment_map[8] / max(moment_map[2] ** 4, 1.0e-30))

    route_d_profile_moment_scaling_formula_available_now = True
    route_d_q_star_inside_small_q_control_domain_now = bool(
        q_star_term_pack["control_parameter_q2_abs"] <= 1.0e-1
    )
    route_d_q_exact_inside_small_q_control_domain_now = bool(
        q_exact_term_pack["control_parameter_q2_abs"] <= 1.0e-1
    )
    route_d_low_order_profile_moment_truncation_supported_now = bool(
        q_star_error_pack["best_truncation_abs_error"] <= 5.0e-2
        and q_exact_error_pack["best_truncation_abs_error"] <= 5.0e-2
    )
    route_d_low_order_profile_moment_no_go_theorem_available_now = bool(
        route_d_profile_moment_scaling_formula_available_now
        and not route_d_q_star_inside_small_q_control_domain_now
        and not route_d_q_exact_inside_small_q_control_domain_now
        and not route_d_low_order_profile_moment_truncation_supported_now
    )
    route_d_target_free_exact_derivation_available_now = False
    route_c_virial_promoted_next_now = bool(route_d_low_order_profile_moment_no_go_theorem_available_now)
    source_materialization_kept_secondary_reserve_now = True

    return {
        "beta1": beta1,
        "epsilon_beta": epsilon_beta,
        "q_exact_over_m0": q_exact,
        "q_star_over_m0": q_star,
        "q_cubic_sqrt_over_m0": q_cubic_sqrt,
        "moment_r2": float(moment_map[2]),
        "moment_r4": float(moment_map[4]),
        "moment_r6": float(moment_map[6]),
        "moment_r8": float(moment_map[8]),
        "scaled_moment_r2": float(scaled_moment_map[2]),
        "scaled_moment_r4": float(scaled_moment_map[4]),
        "scaled_moment_r6": float(scaled_moment_map[6]),
        "scaled_moment_r8": float(scaled_moment_map[8]),
        "moment_scaling_ratio_m4_over_m2sq": moment_scaling_ratio_m4_over_m2sq,
        "moment_scaling_ratio_m6_over_m2cu": moment_scaling_ratio_m6_over_m2cu,
        "moment_scaling_ratio_m8_over_m2qu": moment_scaling_ratio_m8_over_m2qu,
        "q_star_term_q2": float(q_star_term_pack["term_q2"]),
        "q_star_term_q4": float(q_star_term_pack["term_q4"]),
        "q_star_term_q6": float(q_star_term_pack["term_q6"]),
        "q_star_term_q8": float(q_star_term_pack["term_q8"]),
        "q_star_control_parameter_q2_abs": float(q_star_term_pack["control_parameter_q2_abs"]),
        "q_star_largest_term_abs": float(q_star_term_pack["largest_term_abs"]),
        "q_exact_term_q2": float(q_exact_term_pack["term_q2"]),
        "q_exact_term_q4": float(q_exact_term_pack["term_q4"]),
        "q_exact_term_q6": float(q_exact_term_pack["term_q6"]),
        "q_exact_term_q8": float(q_exact_term_pack["term_q8"]),
        "q_exact_control_parameter_q2_abs": float(q_exact_term_pack["control_parameter_q2_abs"]),
        "q_exact_largest_term_abs": float(q_exact_term_pack["largest_term_abs"]),
        "q_cubic_sqrt_term_q2": float(q_cubic_sqrt_term_pack["term_q2"]),
        "q_cubic_sqrt_term_q4": float(q_cubic_sqrt_term_pack["term_q4"]),
        "q_cubic_sqrt_term_q6": float(q_cubic_sqrt_term_pack["term_q6"]),
        "q_cubic_sqrt_term_q8": float(q_cubic_sqrt_term_pack["term_q8"]),
        "q_cubic_sqrt_control_parameter_q2_abs": float(q_cubic_sqrt_term_pack["control_parameter_q2_abs"]),
        "q_cubic_sqrt_largest_term_abs": float(q_cubic_sqrt_term_pack["largest_term_abs"]),
        "q_star_F_exact": float(q_star_error_pack["F_exact"]),
        "q_star_F_trunc_q2": float(q_star_error_pack["F_trunc_q2"]),
        "q_star_F_trunc_q4": float(q_star_error_pack["F_trunc_q4"]),
        "q_star_F_trunc_q6": float(q_star_error_pack["F_trunc_q6"]),
        "q_star_F_trunc_q8": float(q_star_error_pack["F_trunc_q8"]),
        "q_star_F_trunc_q2_abs_error": float(q_star_error_pack["F_trunc_q2_abs_error"]),
        "q_star_F_trunc_q4_abs_error": float(q_star_error_pack["F_trunc_q4_abs_error"]),
        "q_star_F_trunc_q6_abs_error": float(q_star_error_pack["F_trunc_q6_abs_error"]),
        "q_star_F_trunc_q8_abs_error": float(q_star_error_pack["F_trunc_q8_abs_error"]),
        "q_star_best_truncation_abs_error": float(q_star_error_pack["best_truncation_abs_error"]),
        "q_exact_F_exact": float(q_exact_error_pack["F_exact"]),
        "q_exact_F_trunc_q2": float(q_exact_error_pack["F_trunc_q2"]),
        "q_exact_F_trunc_q4": float(q_exact_error_pack["F_trunc_q4"]),
        "q_exact_F_trunc_q6": float(q_exact_error_pack["F_trunc_q6"]),
        "q_exact_F_trunc_q8": float(q_exact_error_pack["F_trunc_q8"]),
        "q_exact_F_trunc_q2_abs_error": float(q_exact_error_pack["F_trunc_q2_abs_error"]),
        "q_exact_F_trunc_q4_abs_error": float(q_exact_error_pack["F_trunc_q4_abs_error"]),
        "q_exact_F_trunc_q6_abs_error": float(q_exact_error_pack["F_trunc_q6_abs_error"]),
        "q_exact_F_trunc_q8_abs_error": float(q_exact_error_pack["F_trunc_q8_abs_error"]),
        "q_exact_best_truncation_abs_error": float(q_exact_error_pack["best_truncation_abs_error"]),
        "q_cubic_sqrt_F_exact": float(q_cubic_sqrt_error_pack["F_exact"]),
        "q_cubic_sqrt_F_trunc_q2": float(q_cubic_sqrt_error_pack["F_trunc_q2"]),
        "q_cubic_sqrt_F_trunc_q4": float(q_cubic_sqrt_error_pack["F_trunc_q4"]),
        "q_cubic_sqrt_F_trunc_q6": float(q_cubic_sqrt_error_pack["F_trunc_q6"]),
        "q_cubic_sqrt_F_trunc_q8": float(q_cubic_sqrt_error_pack["F_trunc_q8"]),
        "q_cubic_sqrt_F_trunc_q2_abs_error": float(q_cubic_sqrt_error_pack["F_trunc_q2_abs_error"]),
        "q_cubic_sqrt_F_trunc_q4_abs_error": float(q_cubic_sqrt_error_pack["F_trunc_q4_abs_error"]),
        "q_cubic_sqrt_F_trunc_q6_abs_error": float(q_cubic_sqrt_error_pack["F_trunc_q6_abs_error"]),
        "q_cubic_sqrt_F_trunc_q8_abs_error": float(q_cubic_sqrt_error_pack["F_trunc_q8_abs_error"]),
        "q_cubic_sqrt_best_truncation_abs_error": float(q_cubic_sqrt_error_pack["best_truncation_abs_error"]),
        "route_d_profile_moment_scaling_formula_available_now": route_d_profile_moment_scaling_formula_available_now,
        "route_d_q_star_inside_small_q_control_domain_now": route_d_q_star_inside_small_q_control_domain_now,
        "route_d_q_exact_inside_small_q_control_domain_now": route_d_q_exact_inside_small_q_control_domain_now,
        "route_d_low_order_profile_moment_truncation_supported_now": route_d_low_order_profile_moment_truncation_supported_now,
        "route_d_low_order_profile_moment_no_go_theorem_available_now": route_d_low_order_profile_moment_no_go_theorem_available_now,
        "route_d_target_free_exact_derivation_available_now": route_d_target_free_exact_derivation_available_now,
        "route_c_virial_promoted_next_now": route_c_virial_promoted_next_now,
        "source_materialization_kept_secondary_reserve_now": source_materialization_kept_secondary_reserve_now,
    }


# Function: allow one CLI smoke run for local verification.

def main() -> None:
    """Print the retained Route-D audit pack as JSON."""
    import json

    print(json.dumps(build_scalar_proxy_route_d_profile_moment_pack(), ensure_ascii=False, indent=2))


# Function: run the helper when invoked as one CLI script.

if __name__ == "__main__":
    main()
