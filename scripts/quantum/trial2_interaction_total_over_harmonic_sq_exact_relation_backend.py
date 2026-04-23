#!/usr/bin/env python3
"""Audit the exact relation behind the interaction_total_over_harmonic_sq ratio.

Purpose:
    After the blind variant screen promoted

        R = E_int * E_total / E_harm^2

    as the best simple energy-partition candidate, this helper checks whether
    that ratio can be derived directly from the retained shooting equation
    rather than left as a screened heuristic.

    The central task is narrow:

    1. verify the finite-radius weighted-EOM identity on the retained scalar
       family,
    2. eliminate the cubic integral I3 so that the Mexican-hat cubic factor
       1/3 appears explicitly,
    3. test whether the resulting exact relation is already sufficient for one
       target-free closeout, or whether only a beta-root followup remains.

Inputs:
    - scripts/quantum/mass_origin_qball_charge_mapping_branch.py
    - scripts/quantum/scalar_proxy_alpha_q_curve_backend.py
    - scripts/quantum/trial2_energy_partition_ratio_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5615-.5622` wrappers

Assumptions:
    - No new parameter is introduced
    - alpha_target is used only as an audit comparator
    - The retained scalar family already fixed by the practical closeout is
      reused without reopening old q-selection routes
"""

from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.mass_origin_qball_charge_mapping_branch import (
    load_qball_module as load_qball_pivot_module,
)
from scripts.quantum.mass_origin_qball_charge_mapping_branch import solve_full_profile
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import ALPHA_TARGET
from scripts.quantum.trial2_energy_partition_ratio_backend import build_energy_partition_row


PRIOR_ALPHA_BETA_ROOT = 0.9982996989044647
ROOT_SCAN_HALF_WIDTH = 1.0e-3
ROOT_SCAN_COUNT = 17


# 関数: retained scalar profile を materialize する pivot solver を cached で返す。
@lru_cache(maxsize=1)
def get_qball_pivot_module():
    """Return the retained pivot solver module used for scalar profiles."""
    return load_qball_pivot_module()


# 関数: one beta row の exact-relation ingredients を返す。

@lru_cache(maxsize=None)
def build_exact_relation_row(beta: float) -> dict:
    """Return one exact-relation row for the promoted energy-partition variant."""
    beta = float(beta)
    qball_pivot = get_qball_pivot_module()
    amplitude = qball_pivot.find_amp(beta)
    if amplitude is None:
        raise SystemExit(f"[fail] localized scalar profile is unavailable for beta={beta}")

    radius, profile, profile_prime = solve_full_profile(beta, float(amplitude))
    radius = np.asarray(radius, dtype=float)
    profile = np.asarray(profile, dtype=float)
    profile_prime = np.asarray(profile_prime, dtype=float)

    i2 = float(np.trapezoid(np.square(profile) * np.square(radius), radius))
    ig = float(np.trapezoid(np.square(profile_prime) * np.square(radius), radius))
    i3 = float(np.trapezoid(np.power(profile, 3) * np.square(radius), radius))
    i4 = float(np.trapezoid(np.power(profile, 4) * np.square(radius), radius))
    boundary_weighted_eom = float(
        np.square(radius[-1]) * profile[-1] * profile_prime[-1]
    )

    epsilon_beta = float(1.0 - beta * beta)
    weighted_eom_residual = float(
        boundary_weighted_eom - ig - epsilon_beta * i2 + 3.0 * i3 + i4
    )
    i3_from_weighted_eom = float(
        (ig + epsilon_beta * i2 - i4 - boundary_weighted_eom) / 3.0
    )
    i3_from_weighted_eom_diff = float(i3 - i3_from_weighted_eom)

    harmonic_dimless = float(2.0 * (1.0 + beta * beta) * i2)
    interaction_dimless = float(4.0 * i3 + i4)
    total_dimless = float(harmonic_dimless + 2.0 * ig + 4.0 * i3 + i4)

    exact_relation_from_integrals = float(
        (interaction_dimless * total_dimless) / (harmonic_dimless * harmonic_dimless)
    )

    exact_cubic_numerator = float(4.0 * (ig + epsilon_beta * i2 - boundary_weighted_eom) - i4)
    exact_total_numerator = float(
        2.0 * (5.0 + beta * beta) * i2 + 10.0 * ig - i4 - 4.0 * boundary_weighted_eom
    )
    exact_relation_from_weighted_eom = float(
        (exact_cubic_numerator * exact_total_numerator)
        / (36.0 * np.square(1.0 + beta * beta) * i2 * i2)
    )
    exact_relation_weighted_eom_residual = float(
        exact_relation_from_weighted_eom - exact_relation_from_integrals
    )

    leading_cubic_numerator = float(4.0 * (ig + epsilon_beta * i2 - boundary_weighted_eom))
    leading_total_numerator = float(
        2.0 * (5.0 + beta * beta) * i2 + 10.0 * ig - 4.0 * boundary_weighted_eom
    )
    leading_relation_cubic_dominant = float(
        (leading_cubic_numerator * leading_total_numerator)
        / (36.0 * np.square(1.0 + beta * beta) * i2 * i2)
    )

    energy_row = build_energy_partition_row(beta)
    if energy_row is None:
        raise SystemExit(f"[fail] energy-partition row is unavailable for beta={beta}")

    screened_front_runner = float(
        (float(energy_row["energy_interaction"]) * float(energy_row["energy_total"]))
        / np.square(float(energy_row["energy_harmonic"]))
    )
    screened_exact_relation_residual = float(
        exact_relation_from_integrals - screened_front_runner
    )

    cubic_share_of_interaction = float(
        float(energy_row["energy_cubic"]) / max(float(energy_row["energy_interaction"]), 1.0e-30)
    )
    exact_relation_rel_error_vs_target = float(
        (exact_relation_from_integrals - ALPHA_TARGET) / ALPHA_TARGET
    )
    leading_relation_rel_error_vs_target = float(
        (leading_relation_cubic_dominant - ALPHA_TARGET) / ALPHA_TARGET
    )
    boundary_share_of_source = float(
        abs(boundary_weighted_eom) / max(abs(ig + epsilon_beta * i2), 1.0e-30)
    )
    quartic_share_of_source = float(abs(i4) / max(abs(ig + epsilon_beta * i2), 1.0e-30))

    return {
        "beta": beta,
        "epsilon_beta": epsilon_beta,
        "central_amplitude": float(amplitude),
        "i2": i2,
        "ig": ig,
        "i3": i3,
        "i4": i4,
        "boundary_weighted_eom": boundary_weighted_eom,
        "weighted_eom_residual": weighted_eom_residual,
        "i3_from_weighted_eom": i3_from_weighted_eom,
        "i3_from_weighted_eom_diff": i3_from_weighted_eom_diff,
        "harmonic_dimless": harmonic_dimless,
        "interaction_dimless": interaction_dimless,
        "total_dimless": total_dimless,
        "exact_cubic_numerator": exact_cubic_numerator,
        "exact_total_numerator": exact_total_numerator,
        "exact_relation_from_integrals": exact_relation_from_integrals,
        "exact_relation_from_weighted_eom": exact_relation_from_weighted_eom,
        "exact_relation_weighted_eom_residual": exact_relation_weighted_eom_residual,
        "leading_cubic_numerator": leading_cubic_numerator,
        "leading_total_numerator": leading_total_numerator,
        "leading_relation_cubic_dominant": leading_relation_cubic_dominant,
        "exact_relation_rel_error_vs_target": exact_relation_rel_error_vs_target,
        "leading_relation_rel_error_vs_target": leading_relation_rel_error_vs_target,
        "screened_front_runner": screened_front_runner,
        "screened_exact_relation_residual": screened_exact_relation_residual,
        "energy_kinetic": float(energy_row["energy_kinetic"]),
        "energy_mass": float(energy_row["energy_mass"]),
        "energy_gradient": float(energy_row["energy_gradient"]),
        "energy_cubic": float(energy_row["energy_cubic"]),
        "energy_quartic": float(energy_row["energy_quartic"]),
        "energy_interaction": float(energy_row["energy_interaction"]),
        "energy_harmonic": float(energy_row["energy_harmonic"]),
        "energy_total": float(energy_row["energy_total"]),
        "cubic_share_of_interaction": cubic_share_of_interaction,
        "boundary_share_of_source": boundary_share_of_source,
        "quartic_share_of_source": quartic_share_of_source,
    }


# 関数: exact relation family の local beta root を探索する。

def find_local_exact_relation_root(retained_beta: float) -> dict:
    """Return one local beta root pack for the exact-relation family."""
    retained_beta = float(retained_beta)
    scan_betas = np.linspace(
        retained_beta,
        min(0.9995, retained_beta + ROOT_SCAN_HALF_WIDTH),
        ROOT_SCAN_COUNT,
        dtype=float,
    )
    scan_rows = [build_exact_relation_row(float(beta)) for beta in scan_betas]
    scan_values = [
        float(row["exact_relation_from_integrals"] - ALPHA_TARGET) for row in scan_rows
    ]

    left_beta = math.nan
    right_beta = math.nan
    for left_row, right_row, left_value, right_value in zip(
        scan_rows[:-1],
        scan_rows[1:],
        scan_values[:-1],
        scan_values[1:],
    ):
        if left_value == 0.0:
            left_beta = float(left_row["beta"])
            right_beta = float(left_row["beta"])
            break

        if left_value * right_value < 0.0:
            left_beta = float(left_row["beta"])
            right_beta = float(right_row["beta"])
            break

    root_available_now = bool(not math.isnan(left_beta) and not math.isnan(right_beta))
    if not root_available_now:
        return {
            "local_beta_root_available_now": False,
            "scan_rows": scan_rows,
            "scan_left_beta": math.nan,
            "scan_right_beta": math.nan,
            "beta_root": math.nan,
            "beta_root_rel_shift_vs_retained": math.nan,
            "beta_root_rel_shift_vs_prior_alpha_beta": math.nan,
            "beta_root_exact_relation_value": math.nan,
            "beta_root_exact_relation_rel_error_vs_target": math.nan,
        }

    if left_beta == right_beta:
        beta_root = float(left_beta)
    else:
        beta_root = float(
            brentq(
                lambda beta: float(
                    build_exact_relation_row(float(beta))["exact_relation_from_integrals"]
                    - ALPHA_TARGET
                ),
                left_beta,
                right_beta,
            )
        )

    root_row = build_exact_relation_row(beta_root)
    return {
        "local_beta_root_available_now": True,
        "scan_rows": scan_rows,
        "scan_left_beta": float(left_beta),
        "scan_right_beta": float(right_beta),
        "beta_root": beta_root,
        "beta_root_rel_shift_vs_retained": float(
            (beta_root - retained_beta) / retained_beta
        ),
        "beta_root_rel_shift_vs_prior_alpha_beta": float(
            (beta_root - PRIOR_ALPHA_BETA_ROOT) / PRIOR_ALPHA_BETA_ROOT
        ),
        "beta_root_exact_relation_value": float(root_row["exact_relation_from_integrals"]),
        "beta_root_exact_relation_rel_error_vs_target": float(
            (float(root_row["exact_relation_from_integrals"]) - ALPHA_TARGET) / ALPHA_TARGET
        ),
    }


# 関数: exact-relation audit 全体を official pack に束ねる。

def build_trial2_interaction_total_over_harmonic_sq_exact_pack(
    retained_beta: float,
    nearest_beta: float,
) -> dict:
    """Return one exact-relation audit pack for the promoted variant ratio."""
    retained_row = build_exact_relation_row(float(retained_beta))
    near_row = build_exact_relation_row(float(nearest_beta))
    beta_root_pack = find_local_exact_relation_root(float(retained_beta))

    exact_weighted_eom_identity_available_now = bool(
        abs(float(retained_row["weighted_eom_residual"])) <= 1.0e-8
        and abs(float(near_row["weighted_eom_residual"])) <= 1.0e-8
    )
    exact_relation_available_now = bool(
        abs(float(retained_row["exact_relation_weighted_eom_residual"])) <= 1.0e-8
        and abs(float(near_row["exact_relation_weighted_eom_residual"])) <= 1.0e-8
        and abs(float(retained_row["screened_exact_relation_residual"])) <= 1.0e-12
        and abs(float(near_row["screened_exact_relation_residual"])) <= 1.0e-12
    )
    one_third_factor_explicit_now = True
    quartic_negligible_now = bool(
        float(retained_row["cubic_share_of_interaction"]) >= 0.995
        and float(near_row["cubic_share_of_interaction"]) >= 0.995
    )
    leading_relation_subpercent_now = bool(
        abs(float(retained_row["leading_relation_rel_error_vs_target"])) <= 1.0e-2
        and abs(float(near_row["leading_relation_rel_error_vs_target"])) <= 1.0e-2
    )
    leading_relation_point_one_percent_now = bool(
        abs(float(retained_row["leading_relation_rel_error_vs_target"])) <= 1.0e-3
        and abs(float(near_row["leading_relation_rel_error_vs_target"])) <= 1.0e-3
    )
    local_beta_root_available_now = bool(beta_root_pack["local_beta_root_available_now"])
    beta_root_consistent_with_prior_alpha_beta_now = bool(
        local_beta_root_available_now
        and abs(float(beta_root_pack["beta_root_rel_shift_vs_prior_alpha_beta"])) <= 1.0e-4
    )
    exact_target_free_closeout_available_now = False
    beta_root_followup_required_now = bool(
        exact_weighted_eom_identity_available_now
        and exact_relation_available_now
        and local_beta_root_available_now
        and not exact_target_free_closeout_available_now
    )
    conditional_hold_secondary_retained_now = bool(beta_root_followup_required_now)

    return {
        "alpha_target": float(ALPHA_TARGET),
        "retained_beta1": float(retained_beta),
        "nearest_alpha_beta_root_to_retained": float(nearest_beta),
        "prior_alpha_beta_root": float(PRIOR_ALPHA_BETA_ROOT),
        "retained_row": retained_row,
        "nearest_row": near_row,
        **beta_root_pack,
        "exact_weighted_eom_identity_available_now": exact_weighted_eom_identity_available_now,
        "exact_relation_available_now": exact_relation_available_now,
        "one_third_factor_explicit_now": one_third_factor_explicit_now,
        "quartic_negligible_now": quartic_negligible_now,
        "leading_relation_subpercent_now": leading_relation_subpercent_now,
        "leading_relation_point_one_percent_now": leading_relation_point_one_percent_now,
        "beta_root_consistent_with_prior_alpha_beta_now": (
            beta_root_consistent_with_prior_alpha_beta_now
        ),
        "exact_target_free_closeout_available_now": exact_target_free_closeout_available_now,
        "beta_root_followup_required_now": beta_root_followup_required_now,
        "conditional_hold_secondary_retained_now": conditional_hold_secondary_retained_now,
    }


# 関数: backend 単体実行時に retained summary を表示する。

def main() -> None:
    """Run the exact-relation audit directly and print one compact summary."""
    pack = build_trial2_interaction_total_over_harmonic_sq_exact_pack(
        retained_beta=0.9982557379261291,
        nearest_beta=0.9982996989044647,
    )
    retained = pack["retained_row"]
    print("[trial2_interaction_total_over_harmonic_sq_exact_relation_backend] retained:")
    print(f"  R_exact = {retained['exact_relation_from_integrals']:.15f}")
    print(
        f"  R_exact_from_weighted_eom = "
        f"{retained['exact_relation_from_weighted_eom']:.15f}"
    )
    print(
        f"  weighted-EOM residual = {retained['weighted_eom_residual']:.15e}"
    )
    print(
        f"  local beta root available = {pack['local_beta_root_available_now']}"
    )
    if pack["local_beta_root_available_now"]:
        print(f"  beta_root = {pack['beta_root']:.15f}")


if __name__ == "__main__":
    main()
