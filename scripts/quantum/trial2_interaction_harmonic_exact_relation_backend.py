#!/usr/bin/env python3
"""Audit the exact relation behind the interaction-over-harmonic front runner.

Purpose:
    After the simple energy-partition screening promoted

        R_int_harm = E_int / E_harm

    as the unique front runner, this helper checks whether that ratio can be
    elevated into one exact target-free relation rather than a heuristic.

    The audit combines the retained energy decomposition with the exact
    finite-radius weighted-EOM / virial identities already validated on the
    same scalar branch. The key question is whether R_int_harm collapses to one
    simple beta-only or beta-plus-gradient law, or whether an independent
    boundary remainder remains unavoidable.

Inputs:
    - scripts/quantum/trial2_energy_partition_ratio_backend.py
    - scripts/quantum/scalar_proxy_route_c_virial_backend.py
    - scripts/quantum/mass_origin_qball_charge_mapping_branch.py

Outputs:
    - One in-memory audit pack consumed by `.5591-.5598` wrappers

Assumptions:
    - No new parameter is introduced
    - alpha_target is used only as an audit comparator
    - The audit stays on the retained scalar Q-ball family already fixed
"""

from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.mass_origin_qball_charge_mapping_branch import (
    load_qball_module as load_qball_pivot_module,
)
from scripts.quantum.mass_origin_qball_charge_mapping_branch import solve_full_profile
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import ALPHA_TARGET
from scripts.quantum.scalar_proxy_route_c_virial_backend import (
    build_route_c_identity_pack,
)
from scripts.quantum.trial2_energy_partition_ratio_backend import (
    build_energy_partition_row,
)


FOUR_PI = 4.0 * math.pi


# 関数: retained scalar profile を materialize する pivot solver module を cached で返す。
@lru_cache(maxsize=1)
def get_qball_pivot_module():
    """Return the retained pivot solver module used for scalar profiles."""
    return load_qball_pivot_module()


# 関数: one beta row の exact interaction-over-harmonic decomposition を返す。

def build_exact_relation_row(beta: float) -> dict:
    """Return one exact decomposition row for the interaction-over-harmonic ratio."""
    beta = float(beta)
    qball_pivot = get_qball_pivot_module()
    amplitude = qball_pivot.find_amp(beta)
    if amplitude is None:
        raise SystemExit(f"[fail] localized scalar profile is unavailable for beta={beta}")

    radius, profile, profile_prime = solve_full_profile(beta, float(amplitude))
    radius = np.asarray(radius, dtype=float)
    profile = np.asarray(profile, dtype=float)
    profile_prime = np.asarray(profile_prime, dtype=float)

    energy_row = build_energy_partition_row(beta)
    if energy_row is None:
        raise SystemExit(f"[fail] energy-partition row is unavailable for beta={beta}")

    epsilon_beta = float(1.0 - beta * beta)
    identity_pack = build_route_c_identity_pack(radius, profile, profile_prime, epsilon_beta)

    energy_mass = float(energy_row["energy_mass"])
    energy_gradient = float(energy_row["energy_gradient"])
    energy_harmonic = float(energy_row["energy_harmonic"])
    energy_interaction = float(energy_row["energy_interaction"])
    interaction_over_harmonic = float(
        energy_row["candidate_ratios"]["interaction_over_harmonic"]
    )
    i2 = float(energy_mass / (2.0 * math.pi))

    beta_term = float(epsilon_beta / (1.0 + beta * beta))
    gradient_term = float(energy_gradient / (3.0 * energy_harmonic))
    boundary_term = float(
        ((FOUR_PI / 3.0) * float(identity_pack["boundary_virial"])) / energy_harmonic
    )
    exact_reconstruction = float(beta_term + gradient_term + boundary_term)
    exact_reconstruction_residual = float(exact_reconstruction - interaction_over_harmonic)

    beta_only_residual = float(beta_term - interaction_over_harmonic)
    beta_plus_gradient_residual = float(
        beta_term + gradient_term - interaction_over_harmonic
    )
    boundary_share_of_front_runner = float(
        boundary_term / max(interaction_over_harmonic, 1.0e-30)
    )
    gradient_share_of_front_runner = float(
        gradient_term / max(interaction_over_harmonic, 1.0e-30)
    )
    beta_share_of_front_runner = float(
        beta_term / max(interaction_over_harmonic, 1.0e-30)
    )
    boundary_over_i2 = float(identity_pack["boundary_virial"] / max(i2, 1.0e-30))

    return {
        "beta": beta,
        "epsilon_beta": epsilon_beta,
        "central_amplitude": float(amplitude),
        "energy_kinetic": float(energy_row["energy_kinetic"]),
        "energy_mass": energy_mass,
        "energy_gradient": energy_gradient,
        "energy_interaction": energy_interaction,
        "energy_harmonic": energy_harmonic,
        "interaction_over_harmonic": interaction_over_harmonic,
        "beta_term": beta_term,
        "gradient_term": gradient_term,
        "boundary_term": boundary_term,
        "exact_reconstruction": exact_reconstruction,
        "exact_reconstruction_residual": exact_reconstruction_residual,
        "beta_only_residual": beta_only_residual,
        "beta_plus_gradient_residual": beta_plus_gradient_residual,
        "beta_share_of_front_runner": beta_share_of_front_runner,
        "gradient_share_of_front_runner": gradient_share_of_front_runner,
        "boundary_share_of_front_runner": boundary_share_of_front_runner,
        "boundary_virial": float(identity_pack["boundary_virial"]),
        "boundary_virial_over_i2": boundary_over_i2,
    }


# 関数: interaction-over-harmonic exact-relation audit 全体を official pack に束ねる。

def build_trial2_interaction_harmonic_exact_pack(
    retained_beta: float,
    nearest_beta: float,
) -> dict:
    """Return one exact-relation audit pack for the front-runner ratio."""
    retained_row = build_exact_relation_row(float(retained_beta))
    near_row = build_exact_relation_row(float(nearest_beta))

    exact_relation_available_now = bool(
        abs(retained_row["exact_reconstruction_residual"]) <= 1.0e-8
        and abs(near_row["exact_reconstruction_residual"]) <= 1.0e-8
    )
    boundary_term_negligible_now = bool(
        abs(retained_row["boundary_share_of_front_runner"]) <= 5.0e-2
        and abs(near_row["boundary_share_of_front_runner"]) <= 5.0e-2
    )
    beta_only_collapse_supported_now = bool(
        abs(retained_row["beta_only_residual"] / retained_row["interaction_over_harmonic"]) <= 1.0e-3
        and abs(near_row["beta_only_residual"] / near_row["interaction_over_harmonic"]) <= 1.0e-3
    )
    beta_plus_gradient_collapse_supported_now = bool(
        abs(
            retained_row["beta_plus_gradient_residual"]
            / retained_row["interaction_over_harmonic"]
        )
        <= 1.0e-3
        and abs(
            near_row["beta_plus_gradient_residual"]
            / near_row["interaction_over_harmonic"]
        )
        <= 1.0e-3
    )
    interaction_over_harmonic_exact_route_available_now = bool(
        exact_relation_available_now
        and boundary_term_negligible_now
        and beta_plus_gradient_collapse_supported_now
    )
    interaction_over_harmonic_negative_closeout_available_now = bool(
        exact_relation_available_now
        and not interaction_over_harmonic_exact_route_available_now
    )
    entropy_promoted_primary_now = bool(interaction_over_harmonic_negative_closeout_available_now)

    return {
        "alpha_target": float(ALPHA_TARGET),
        "retained_beta1": float(retained_beta),
        "nearest_alpha_beta_root_to_retained": float(nearest_beta),
        "retained_row": retained_row,
        "nearest_row": near_row,
        "exact_relation_available_now": exact_relation_available_now,
        "boundary_term_negligible_now": boundary_term_negligible_now,
        "beta_only_collapse_supported_now": beta_only_collapse_supported_now,
        "beta_plus_gradient_collapse_supported_now": beta_plus_gradient_collapse_supported_now,
        "interaction_over_harmonic_exact_route_available_now": (
            interaction_over_harmonic_exact_route_available_now
        ),
        "interaction_over_harmonic_negative_closeout_available_now": (
            interaction_over_harmonic_negative_closeout_available_now
        ),
        "entropy_promoted_primary_now": entropy_promoted_primary_now,
    }


# 関数: backend 単体実行時に retained decomposition summary を表示する。

def main() -> None:
    """Run the interaction-over-harmonic exact-relation audit directly."""
    pack = build_trial2_interaction_harmonic_exact_pack(
        retained_beta=0.9982557379261291,
        nearest_beta=0.9982996989044647,
    )
    retained = pack["retained_row"]
    print("[trial2_interaction_harmonic_exact_relation_backend] retained decomposition:")
    print(f"  R_int_harm = {retained['interaction_over_harmonic']:.15f}")
    print(f"  beta_term = {retained['beta_term']:.15f}")
    print(f"  gradient_term = {retained['gradient_term']:.15f}")
    print(f"  boundary_term = {retained['boundary_term']:.15f}")
    print(f"  exact residual = {retained['exact_reconstruction_residual']:.15e}")
    print(
        f"  one-term exact route available = "
        f"{pack['interaction_over_harmonic_exact_route_available_now']}"
    )


if __name__ == "__main__":
    main()
