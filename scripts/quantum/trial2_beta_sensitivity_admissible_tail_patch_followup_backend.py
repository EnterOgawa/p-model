#!/usr/bin/env python3
"""Audit admissible-tail requirements for the Trial-2 pure-analytic refinement.

Purpose:
    Continue theorem hardening after `.5703-.5710`, where first-principles
    direct-alpha closure is already completed but pure analytic operator-level
    continuum refinement is still deferred to v3.0.

    The critical question is now narrower:

        can the current refinement route safely reuse the raw extended profile
        tail produced by `solve_full_profile`, or must the theorem switch to an
        admissible positive-decay tail continuation before any pure continuum
        statement is credible?

    This backend does not replay the whole alpha derivation. It isolates the
    tail issue by comparing three objects on the retained common-root branch:

    1. the pivot shooting solve, which is explicitly tuned so that the tail
       vanishes at radius 22,
    2. the raw extended solve used by the broader profile utilities, which
       continues the same amplitude past that truncation radius and overshoots
       through zero, and
    3. a simple admissible positive-decay tail candidate from the linearized
       asymptotic equation.

    The output fixes whether the raw extended tail is admissible and whether a
    positive-decay tail patch must become the next theorem-hardening program.

Inputs:
    - scripts/quantum/mass_origin_qball_pivot_branch.py
    - scripts/quantum/mass_origin_qball_charge_mapping_branch.py
    - scripts/quantum/trial2_beta_sensitivity_equation_backend.py
    - scripts/quantum/trial2_beta_sensitivity_monotonicity_followup_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5711-.5718` wrappers

Assumptions:
    - The retained common-root selector remains fixed at beta_common_root
    - No new parameter is introduced
    - The route targets tail admissibility only; it does not yet claim the
      full pure analytic continuum theorem
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

from scripts.quantum.mass_origin_qball_charge_mapping_branch import load_qball_module
from scripts.quantum.mass_origin_qball_charge_mapping_branch import solve_full_profile
from scripts.quantum.trial2_beta_sensitivity_equation_backend import BETA_COMMON_ROOT
from scripts.quantum.trial2_beta_sensitivity_equation_backend import build_common_grid
from scripts.quantum.trial2_beta_sensitivity_equation_backend import build_profile_row
from scripts.quantum.trial2_beta_sensitivity_monotonicity_followup_backend import (
    locate_potential_zero_crossing,
)


PIVOT_TAIL_RADIUS = 22.0
TAIL_MATCH_X = 21.0
TAIL_SAMPLE_POINTS = (22.0, 25.0, 30.0)
TAIL_H_VALUE = 1.0e-6
PIVOT_TAIL_ABS_TOL = 1.0e-7
PIVOT_PROFILE_MIN_TOL = 1.0e-10


# 関数: pivot Q-ball module を cached で返す。
@lru_cache(maxsize=1)
def get_qball_pivot_module():
    """Return the retained pivot module used to determine the shooting amplitude."""
    return load_qball_module()


# 関数: 線形補間で最初の zero crossing を返す。
def locate_first_zero_crossing(radius: np.ndarray, values: np.ndarray) -> float | None:
    """Return the first zero crossing found by linear interpolation."""
    radius = np.asarray(radius, dtype=float)
    values = np.asarray(values, dtype=float)
    for left, right in zip(range(values.size - 1), range(1, values.size)):
        v0 = float(values[left])
        v1 = float(values[right])
        if v0 == 0.0:
            return float(radius[left])

        if v1 == 0.0:
            return float(radius[right])

        if v0 * v1 < 0.0:
            x0 = float(radius[left])
            x1 = float(radius[right])
            return float(x0 - v0 * (x1 - x0) / (v1 - v0))

    return None


# 関数: admissible positive-decay tail candidate を評価する。
def evaluate_positive_decay_tail_candidate(
    beta_value: float,
    x_match: float,
    y_match: float,
    x_values: tuple[float, ...],
) -> dict:
    """Return a value-matched positive-decay tail candidate."""
    beta_value = float(beta_value)
    x_match = float(x_match)
    y_match = float(y_match)
    kappa = float(math.sqrt(1.0 - beta_value * beta_value))
    if x_match <= 0.0:
        raise SystemExit("[fail] tail match point must be positive")

    amplitude = float(y_match * x_match * math.exp(kappa * x_match))
    values = {}
    for x_value in x_values:
        x_value = float(x_value)
        values[f"tail_candidate_value_at_{str(x_value).replace('.', '_')}"] = float(
            amplitude * math.exp(-kappa * x_value) / x_value
        )

    derivative_at_match = float(y_match * (-kappa - 1.0 / x_match))
    return {
        "kappa": kappa,
        "tail_candidate_amplitude": amplitude,
        "tail_candidate_derivative_at_match": derivative_at_match,
        **values,
    }


# 関数: admissible-tail followup 監査 pack を返す。
def build_trial2_beta_sensitivity_admissible_tail_patch_followup_pack() -> dict:
    """Return one admissible-tail audit pack for the theorem-hardening route."""
    beta_common_root = float(BETA_COMMON_ROOT)
    qball_pivot = get_qball_pivot_module()
    central_amplitude = float(qball_pivot.find_amp(beta_common_root))

    pivot_row = qball_pivot.solve_profile(beta_common_root, central_amplitude)
    extended_radius, extended_profile, extended_profile_prime = solve_full_profile(
        beta_common_root,
        central_amplitude,
    )
    extended_radius = np.asarray(extended_radius, dtype=float)
    extended_profile = np.asarray(extended_profile, dtype=float)
    extended_profile_prime = np.asarray(extended_profile_prime, dtype=float)

    profile_row_plus = build_profile_row(beta_common_root + TAIL_H_VALUE)
    profile_row_minus = build_profile_row(beta_common_root - TAIL_H_VALUE)
    center_row = {
        "radius": extended_radius,
        "profile": extended_profile,
        "profile_prime": extended_profile_prime,
    }
    common_grid = build_common_grid(center_row, profile_row_plus, profile_row_minus)
    profile_plus = np.interp(common_grid, profile_row_plus["radius"], profile_row_plus["profile"])
    profile_minus = np.interp(common_grid, profile_row_minus["radius"], profile_row_minus["profile"])
    u_beta = (profile_plus - profile_minus) / (2.0 * TAIL_H_VALUE)

    tail_mask = extended_radius >= PIVOT_TAIL_RADIUS
    tail_mask_common = common_grid >= PIVOT_TAIL_RADIUS
    extended_profile_zero_crossing_x = locate_first_zero_crossing(
        extended_radius,
        extended_profile,
    )
    u_beta_zero_crossing_x = locate_first_zero_crossing(common_grid, u_beta)
    potential_zero_crossing_x = float(locate_potential_zero_crossing())

    y_match = float(np.interp(TAIL_MATCH_X, extended_radius, extended_profile))
    yp_match_raw = float(np.interp(TAIL_MATCH_X, extended_radius, extended_profile_prime))
    tail_candidate = evaluate_positive_decay_tail_candidate(
        beta_common_root,
        TAIL_MATCH_X,
        y_match,
        TAIL_SAMPLE_POINTS,
    )
    tail_candidate_positive_now = bool(
        all(
            float(tail_candidate[f"tail_candidate_value_at_{str(x).replace('.', '_')}"]) > 0.0
            for x in TAIL_SAMPLE_POINTS
        )
    )
    tail_derivative_mismatch_rel = float(
        abs(
            yp_match_raw - float(tail_candidate["tail_candidate_derivative_at_match"])
        )
        / max(abs(yp_match_raw), 1.0e-30)
    )

    pivot_tail_abs_at_22 = float(abs(pivot_row["tail"]))
    pivot_positive_up_to_22_now = bool(
        float(pivot_row["fmin"]) >= -PIVOT_PROFILE_MIN_TOL
    )
    raw_extended_tail_negative_now = bool(float(np.min(extended_profile[tail_mask])) < 0.0)
    raw_extended_tail_artifact_detected_now = bool(
        pivot_tail_abs_at_22 <= PIVOT_TAIL_ABS_TOL
        and pivot_positive_up_to_22_now
        and raw_extended_tail_negative_now
        and extended_profile_zero_crossing_x is not None
        and float(extended_profile_zero_crossing_x) <= PIVOT_TAIL_RADIUS + 0.1
    )
    admissible_positive_decay_tail_patch_formula_available_now = bool(
        float(y_match) > 0.0
        and tail_candidate_positive_now
        and potential_zero_crossing_x < TAIL_MATCH_X
    )
    exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now = False
    updated_pack_trial2_admissible_positive_decay_tail_patch_followup_required_now = bool(
        raw_extended_tail_artifact_detected_now
        and admissible_positive_decay_tail_patch_formula_available_now
        and not exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now
    )

    return {
        "beta_common_root": beta_common_root,
        "central_amplitude_common": central_amplitude,
        "pivot_tail_radius": float(PIVOT_TAIL_RADIUS),
        "pivot_tail_abs_at_22": pivot_tail_abs_at_22,
        "pivot_profile_min_up_to_22": float(pivot_row["fmin"]),
        "pivot_profile_max_up_to_22": float(pivot_row["fmax"]),
        "pivot_positive_up_to_22_now": pivot_positive_up_to_22_now,
        "extended_profile_zero_crossing_x": (
            None
            if extended_profile_zero_crossing_x is None
            else float(extended_profile_zero_crossing_x)
        ),
        "extended_profile_min_after_22": float(np.min(extended_profile[tail_mask])),
        "extended_profile_max_after_22": float(np.max(extended_profile[tail_mask])),
        "u_beta_zero_crossing_x": (
            None if u_beta_zero_crossing_x is None else float(u_beta_zero_crossing_x)
        ),
        "potential_zero_crossing_x": potential_zero_crossing_x,
        "tail_match_x": float(TAIL_MATCH_X),
        "tail_match_y_value": y_match,
        "tail_match_raw_derivative": yp_match_raw,
        "tail_derivative_mismatch_rel": tail_derivative_mismatch_rel,
        "tail_linear_kappa": float(tail_candidate["kappa"]),
        "tail_candidate_amplitude": float(tail_candidate["tail_candidate_amplitude"]),
        "tail_candidate_derivative_at_match": float(
            tail_candidate["tail_candidate_derivative_at_match"]
        ),
        "tail_candidate_value_at_22_0": float(
            tail_candidate["tail_candidate_value_at_22_0"]
        ),
        "tail_candidate_value_at_25_0": float(
            tail_candidate["tail_candidate_value_at_25_0"]
        ),
        "tail_candidate_value_at_30_0": float(
            tail_candidate["tail_candidate_value_at_30_0"]
        ),
        "tail_candidate_positive_now": tail_candidate_positive_now,
        "raw_extended_tail_negative_now": raw_extended_tail_negative_now,
        "exact_trial2_beta_sensitivity_raw_extended_tail_artifact_detected_now": (
            raw_extended_tail_artifact_detected_now
        ),
        "exact_trial2_beta_sensitivity_admissible_positive_decay_tail_patch_formula_available_now": (
            admissible_positive_decay_tail_patch_formula_available_now
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now": (
            exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now
        ),
        "updated_pack_trial2_admissible_positive_decay_tail_patch_followup_required_now": (
            updated_pack_trial2_admissible_positive_decay_tail_patch_followup_required_now
        ),
    }


# 関数: backend 単体実行時に retained metrics を表示する。
def main() -> None:
    """Run the admissible-tail audit directly and print key retained values."""
    pack = build_trial2_beta_sensitivity_admissible_tail_patch_followup_pack()
    print("[trial2-beta-admissible-tail-patch-followup]")
    print(f"beta_common_root = {pack['beta_common_root']:.16f}")
    print(f"pivot_tail_abs_at_22 = {pack['pivot_tail_abs_at_22']:.16e}")
    print(
        "extended_profile_zero_crossing_x = "
        f"{pack['extended_profile_zero_crossing_x']}"
    )
    print(f"u_beta_zero_crossing_x = {pack['u_beta_zero_crossing_x']}")
    print(f"potential_zero_crossing_x = {pack['potential_zero_crossing_x']:.16f}")
    print(
        "raw_extended_tail_artifact_detected_now = "
        f"{pack['exact_trial2_beta_sensitivity_raw_extended_tail_artifact_detected_now']}"
    )
    print(
        "admissible_positive_decay_tail_patch_formula_available_now = "
        f"{pack['exact_trial2_beta_sensitivity_admissible_positive_decay_tail_patch_formula_available_now']}"
    )


if __name__ == "__main__":
    main()
