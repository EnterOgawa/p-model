#!/usr/bin/env python3
"""Audit simple energy-partition ratios on the retained scalar Q-ball branch.

Purpose:
    Evaluate the followup route promoted after the alpha(beta) family reduced
    the blocker to one local beta microshift. This route asks whether the
    retained microshift can be read from one simple dimensionless energy
    partition ratio built directly from the localized Q-ball profile, without
    reintroducing q-selection logic.

Inputs:
    - scripts/quantum/mass_origin_qball_charge_mapping_branch.py
    - scripts/quantum/scalar_proxy_alpha_q_curve_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5583-.5590` wrappers

Assumptions:
    - No new parameter is introduced
    - The audit uses the retained mode-1 beta and the nearest high-beta root
      already exposed by the alpha(beta) family route
    - Only simple partition ratios built from the on-shell scalar profile are
      screened here
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


FOUR_PI = 4.0 * math.pi


# 関数: energy-partition 評価に必要な pivot solver module を cached で返す。
@lru_cache(maxsize=1)
def get_qball_pivot_module():
    """Return the retained pivot solver module used to materialize amplitudes."""
    return load_qball_pivot_module()


# 関数: 1 つの beta に対する energy-component row を構築する。

@lru_cache(maxsize=None)
def build_energy_partition_row(beta: float) -> dict | None:
    """Return one retained scalar energy-partition row or None when absent."""
    beta = float(beta)
    qball_pivot = get_qball_pivot_module()
    amplitude = qball_pivot.find_amp(beta)
    if amplitude is None:
        return None

    radius, profile, profile_prime = solve_full_profile(beta, float(amplitude))
    radius = np.asarray(radius, dtype=float)
    profile = np.asarray(profile, dtype=float)
    profile_prime = np.asarray(profile_prime, dtype=float)
    shell_weight = 4.0 * math.pi * np.square(radius)

    energy_kinetic = float(
        np.trapezoid(
            shell_weight * (0.5 * beta * beta * np.square(profile)),
            radius,
        )
    )
    energy_mass = float(
        np.trapezoid(
            shell_weight * (0.5 * np.square(profile)),
            radius,
        )
    )
    energy_gradient = float(
        np.trapezoid(
            shell_weight * (0.5 * np.square(profile_prime)),
            radius,
        )
    )
    energy_cubic = float(
        np.trapezoid(
            shell_weight * np.power(profile, 3),
            radius,
        )
    )
    energy_quartic = float(
        np.trapezoid(
            shell_weight * (0.25 * np.power(profile, 4)),
            radius,
        )
    )

    energy_interaction = energy_cubic + energy_quartic
    energy_harmonic = energy_kinetic + energy_mass
    energy_nonharmonic = energy_gradient + energy_interaction
    energy_total = energy_harmonic + energy_nonharmonic

    candidate_ratios = {
        "interaction_over_harmonic": float(energy_interaction / energy_harmonic),
        "interaction_over_total": float(energy_interaction / energy_total),
        "gradient_over_total": float(energy_gradient / energy_total),
        "gradient_over_harmonic": float(energy_gradient / energy_harmonic),
        "nonharmonic_over_total": float(energy_nonharmonic / energy_total),
        "nonharmonic_over_harmonic": float(energy_nonharmonic / energy_harmonic),
        "interaction_over_harmonic_over_four_pi": float((energy_interaction / energy_harmonic) / FOUR_PI),
        "interaction_over_total_over_four_pi": float((energy_interaction / energy_total) / FOUR_PI),
    }

    return {
        "beta": beta,
        "central_amplitude": float(amplitude),
        "energy_kinetic": energy_kinetic,
        "energy_mass": energy_mass,
        "energy_gradient": energy_gradient,
        "energy_cubic": energy_cubic,
        "energy_quartic": energy_quartic,
        "energy_interaction": energy_interaction,
        "energy_harmonic": energy_harmonic,
        "energy_nonharmonic": energy_nonharmonic,
        "energy_total": energy_total,
        "candidate_ratios": candidate_ratios,
    }


# 関数: ratio family を retained / near-root comparison table に変換する。

def build_candidate_rows(retained_row: dict, near_row: dict) -> list[dict]:
    """Return ranked candidate rows for the screened simple energy ratios."""
    candidate_rows: list[dict] = []
    retained_ratios = dict(retained_row["candidate_ratios"])
    near_ratios = dict(near_row["candidate_ratios"])
    for candidate_name, retained_value in retained_ratios.items():
        near_value = float(near_ratios[candidate_name])
        retained_rel_error = float((float(retained_value) - ALPHA_TARGET) / ALPHA_TARGET)
        near_rel_error = float((near_value - ALPHA_TARGET) / ALPHA_TARGET)
        near_rel_shift = float((near_value - float(retained_value)) / float(retained_value))
        candidate_rows.append(
            {
                "candidate_name": candidate_name,
                "retained_value": float(retained_value),
                "retained_rel_error_vs_target": retained_rel_error,
                "retained_abs_rel_error_vs_target": float(abs(retained_rel_error)),
                "near_value": near_value,
                "near_rel_error_vs_target": near_rel_error,
                "near_rel_shift_vs_retained": near_rel_shift,
            }
        )

    return sorted(
        candidate_rows,
        key=lambda row: (
            float(row["retained_abs_rel_error_vs_target"]),
            abs(float(row["near_rel_shift_vs_retained"])),
            str(row["candidate_name"]),
        ),
    )


# 関数: energy-partition screening 全体を official pack に束ねる。

def build_trial2_energy_partition_pack(retained_beta: float, nearest_beta: float) -> dict:
    """Return one retained energy-partition screening pack."""
    retained_row = build_energy_partition_row(float(retained_beta))
    near_row = build_energy_partition_row(float(nearest_beta))
    if retained_row is None or near_row is None:
        raise SystemExit("[fail] retained or nearest-root energy row is unavailable")

    candidate_rows = build_candidate_rows(retained_row, near_row)
    front_runner = dict(candidate_rows[0])
    second_runner = dict(candidate_rows[1])
    alpha_beta_retained_rel_error = -0.019262702271264597

    front_runner_improves_alpha_beta_now = bool(
        float(front_runner["retained_abs_rel_error_vs_target"]) < abs(alpha_beta_retained_rel_error)
    )
    front_runner_exact_route_available_now = bool(
        float(front_runner["retained_abs_rel_error_vs_target"]) <= 1.0e-12
        and abs(float(front_runner["near_rel_shift_vs_retained"])) <= 1.0e-12
    )
    entropy_followup_required_now = bool(
        front_runner_improves_alpha_beta_now and not front_runner_exact_route_available_now
    )

    return {
        "alpha_target": float(ALPHA_TARGET),
        "retained_beta1": float(retained_beta),
        "nearest_alpha_beta_root_to_retained": float(nearest_beta),
        "nearest_beta_rel_shift_vs_retained": float(
            (float(nearest_beta) - float(retained_beta)) / float(retained_beta)
        ),
        "retained_energy_row": retained_row,
        "nearest_energy_row": near_row,
        "candidate_rows": candidate_rows,
        "energy_partition_front_runner_name": str(front_runner["candidate_name"]),
        "energy_partition_front_runner_retained_value": float(front_runner["retained_value"]),
        "energy_partition_front_runner_retained_rel_error_vs_target": float(
            front_runner["retained_rel_error_vs_target"]
        ),
        "energy_partition_front_runner_near_value": float(front_runner["near_value"]),
        "energy_partition_front_runner_near_rel_error_vs_target": float(
            front_runner["near_rel_error_vs_target"]
        ),
        "energy_partition_front_runner_near_rel_shift_vs_retained": float(
            front_runner["near_rel_shift_vs_retained"]
        ),
        "energy_partition_second_runner_name": str(second_runner["candidate_name"]),
        "energy_partition_second_runner_retained_abs_rel_error_vs_target": float(
            second_runner["retained_abs_rel_error_vs_target"]
        ),
        "energy_partition_front_runner_margin_vs_second": float(
            float(second_runner["retained_abs_rel_error_vs_target"])
            - float(front_runner["retained_abs_rel_error_vs_target"])
        ),
        "front_runner_improves_alpha_beta_now": front_runner_improves_alpha_beta_now,
        "front_runner_exact_route_available_now": front_runner_exact_route_available_now,
        "entropy_followup_required_now": entropy_followup_required_now,
    }


# 関数: backend 単体実行時に screening summary を表示する。

def main() -> None:
    """Run the retained energy-partition screening directly."""
    pack = build_trial2_energy_partition_pack(
        retained_beta=0.9982557379261291,
        nearest_beta=0.9982996989044647,
    )
    print("[trial2_energy_partition_ratio_backend] front runner:")
    print(
        f"  {pack['energy_partition_front_runner_name']} = "
        f"{pack['energy_partition_front_runner_retained_value']:.15f}"
    )
    print(
        f"  rel error vs alpha_target = "
        f"{pack['energy_partition_front_runner_retained_rel_error_vs_target']:.15f}"
    )
    print(
        f"  near-root shift = "
        f"{pack['energy_partition_front_runner_near_rel_shift_vs_retained']:.15f}"
    )
    print(
        f"  improves alpha(beta) retained residual = "
        f"{pack['front_runner_improves_alpha_beta_now']}"
    )


if __name__ == "__main__":
    main()
