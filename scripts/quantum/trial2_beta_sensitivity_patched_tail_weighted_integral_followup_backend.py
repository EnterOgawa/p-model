#!/usr/bin/env python3
"""Audit full-domain weighted-integral sign support on the admissible patched tail.

Purpose:
    Continue the reopened pure-analytic refinement after `.5711-.5718`, where
    the raw post-22 extension was fixed as an inadmissible artifact and one
    value-matched positive-decay tail candidate became available.

    The next honest theorem question is narrower:

        once the raw tail is replaced by an admissible positive-decay patch,
        do the full-domain weighted beta-derivative integrals

            dI_n / d beta = n ∫ y_beta^(n-1) u_beta x^2 dx

        keep their negative sign, with the patched tail remainder too small to
        reverse the interior contribution?

    This backend does not claim the final pure analytic continuum theorem. It
    checks one support layer only:

    1. the patched full-domain weighted integrals stay negative for n = 2, 3, 4,
    2. the patched tail remainder remains nonreversing, and
    3. the result is stable once the tail cutoff is pushed into the asymptotic
       regime.

Inputs:
    - scripts/quantum/trial2_beta_sensitivity_equation_backend.py
    - scripts/quantum/trial2_beta_sensitivity_admissible_tail_patch_followup_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5719-.5726` wrappers

Assumptions:
    - The retained common-root selector stays fixed at beta_common_root
    - The tail is matched by value at x_match = 21.0
    - No new parameter is introduced
    - This route targets patched-tail weighted-integral support only; it does
      not yet claim the final pure analytic operator-level theorem
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

from scripts.quantum.trial2_beta_sensitivity_admissible_tail_patch_followup_backend import (
    TAIL_MATCH_X,
)
from scripts.quantum.trial2_beta_sensitivity_equation_backend import BETA_COMMON_ROOT
from scripts.quantum.trial2_beta_sensitivity_equation_backend import H_VALUES
from scripts.quantum.trial2_beta_sensitivity_equation_backend import build_profile_row


PATCHED_TAIL_X_MAX_VALUES = (60.0, 80.0, 100.0, 140.0)
PATCHED_TAIL_ORDERS = (2, 3, 4)
PATCHED_TAIL_POINT_COUNT = 6000
PATCHED_FULL_GRID_POINT_COUNT = 12000
TAIL_REMAINDER_FRACTION_TOL = 2.0e-2
TAIL_CUTOFF_REL_SPREAD_TOL = 2.0e-3


# 関数: admissible positive-decay tail を profile row へ貼り込む。
@lru_cache(maxsize=None)
def build_patched_profile_row(beta: float, x_max: float) -> dict:
    """Return one profile row whose far tail is replaced by the admissible patch."""
    beta = float(beta)
    x_max = float(x_max)
    if x_max <= float(TAIL_MATCH_X):
        raise SystemExit("[fail] patched tail cutoff must lie beyond the match point")

    base_row = build_profile_row(beta)
    radius = np.asarray(base_row["radius"], dtype=float)
    profile = np.asarray(base_row["profile"], dtype=float)
    profile_prime = np.asarray(base_row["profile_prime"], dtype=float)
    x_match = float(TAIL_MATCH_X)
    kappa = float(math.sqrt(1.0 - beta * beta))
    y_match = float(np.interp(x_match, radius, profile))
    if y_match <= 0.0:
        raise SystemExit("[fail] patched tail match value must stay positive")

    keep_mask = radius <= x_match
    tail_radius = np.linspace(x_match, x_max, PATCHED_TAIL_POINT_COUNT, dtype=float)
    tail_profile = y_match * (x_match / tail_radius) * np.exp(-kappa * (tail_radius - x_match))
    tail_profile_prime = tail_profile * (-kappa - 1.0 / tail_radius)

    patched_radius = np.concatenate((radius[keep_mask][:-1], tail_radius))
    patched_profile = np.concatenate((profile[keep_mask][:-1], tail_profile))
    patched_profile_prime = np.concatenate((profile_prime[keep_mask][:-1], tail_profile_prime))
    return {
        "beta": beta,
        "x_max": x_max,
        "radius": patched_radius,
        "profile": patched_profile,
        "profile_prime": patched_profile_prime,
        "y_match": y_match,
        "kappa": kappa,
        "patched_profile_min": float(np.min(patched_profile)),
        "patched_tail_min": float(np.min(tail_profile)),
    }


# 関数: patched rows 共通の integration grid を返す。

def build_patched_common_grid(center_row: dict, plus_row: dict, minus_row: dict) -> np.ndarray:
    """Return one overlap grid shared by the patched central / plus / minus rows."""
    lower = max(
        float(center_row["radius"][0]),
        float(plus_row["radius"][0]),
        float(minus_row["radius"][0]),
    )
    upper = min(
        float(center_row["radius"][-1]),
        float(plus_row["radius"][-1]),
        float(minus_row["radius"][-1]),
    )
    if upper <= lower:
        raise SystemExit("[fail] patched-tail overlap grid is empty")

    return np.linspace(lower, upper, PATCHED_FULL_GRID_POINT_COUNT, dtype=float)


# 関数: one `(h, x_max)` に対する patched weighted-integral row を返す。

def build_patched_weighted_integral_row(h_value: float, x_max: float) -> dict:
    """Return one patched full-domain weighted-integral support row."""
    beta_common_root = float(BETA_COMMON_ROOT)
    h_value = float(h_value)
    x_max = float(x_max)
    center_row = build_patched_profile_row(beta_common_root, x_max)
    plus_row = build_patched_profile_row(beta_common_root + h_value, x_max)
    minus_row = build_patched_profile_row(beta_common_root - h_value, x_max)
    grid = build_patched_common_grid(center_row, plus_row, minus_row)

    profile = np.interp(grid, center_row["radius"], center_row["profile"])
    profile_plus = np.interp(grid, plus_row["radius"], plus_row["profile"])
    profile_minus = np.interp(grid, minus_row["radius"], minus_row["profile"])
    u_beta = (profile_plus - profile_minus) / (2.0 * h_value)
    tail_mask = grid >= float(TAIL_MATCH_X)
    if not np.any(tail_mask):
        raise SystemExit("[fail] patched-tail mask is empty")

    order_rows = []
    for order in PATCHED_TAIL_ORDERS:
        integrand = np.asarray(
            order * np.power(profile, order - 1) * u_beta * np.square(grid),
            dtype=float,
        )
        total_integral = float(np.trapezoid(integrand, grid))
        tail_integral = float(np.trapezoid(integrand[tail_mask], grid[tail_mask]))
        tail_fraction = float(abs(tail_integral) / max(abs(total_integral), 1.0e-30))
        order_rows.append(
            {
                "order_n": int(order),
                "weighted_total_integral": total_integral,
                "weighted_tail_integral": tail_integral,
                "tail_remainder_abs_fraction": tail_fraction,
                "weighted_total_negative_now": bool(total_integral < 0.0),
                "weighted_tail_nonreversing_now": bool(
                    tail_fraction <= TAIL_REMAINDER_FRACTION_TOL
                ),
            }
        )

    return {
        "h": h_value,
        "x_max": x_max,
        "y_match": float(center_row["y_match"]),
        "kappa": float(center_row["kappa"]),
        "patched_profile_positive_now": bool(center_row["patched_profile_min"] > 0.0),
        "patched_tail_positive_now": bool(center_row["patched_tail_min"] > 0.0),
        "u_beta_tail_positive_fraction": float(np.mean(u_beta[tail_mask] > 0.0)),
        "u_beta_tail_min": float(np.min(u_beta[tail_mask])),
        "u_beta_tail_max": float(np.max(u_beta[tail_mask])),
        "order_rows": order_rows,
    }


# 関数: order ごとの sign / remainder summary を返す。

def summarize_patched_order_rows(weighted_rows: list[dict], order: int) -> dict:
    """Return one cross-cut summary for the selected order."""
    order_rows = [
        {
            "h": float(row["h"]),
            "x_max": float(row["x_max"]),
            **next(item for item in row["order_rows"] if item["order_n"] == order),
        }
        for row in weighted_rows
    ]
    by_xmax_rows = [
        row
        for row in order_rows
        if abs(float(row["x_max"]) - float(PATCHED_TAIL_X_MAX_VALUES[-1])) < 1.0e-12
    ]
    if not by_xmax_rows:
        raise SystemExit("[fail] patched-tail summary could not find the retained cutoff row")

    smallest_h = min(float(value) for value in H_VALUES)
    cutoff_rows = [
        row
        for row in order_rows
        if abs(float(row["h"]) - smallest_h) < 1.0e-15
    ]
    if not cutoff_rows:
        raise SystemExit("[fail] patched-tail summary could not find the smallest-h rows")

    cutoff_values = [float(row["weighted_total_integral"]) for row in cutoff_rows]
    cutoff_rel_spread = float(
        (max(cutoff_values) - min(cutoff_values))
        / max(abs(np.mean(cutoff_values)), 1.0e-30)
    )
    return {
        "order_n": int(order),
        "weighted_total_integral_min": float(
            min(float(row["weighted_total_integral"]) for row in order_rows)
        ),
        "weighted_total_integral_max": float(
            max(float(row["weighted_total_integral"]) for row in order_rows)
        ),
        "weighted_tail_integral_min": float(
            min(float(row["weighted_tail_integral"]) for row in order_rows)
        ),
        "weighted_tail_integral_max": float(
            max(float(row["weighted_tail_integral"]) for row in order_rows)
        ),
        "tail_remainder_abs_fraction_min": float(
            min(float(row["tail_remainder_abs_fraction"]) for row in order_rows)
        ),
        "tail_remainder_abs_fraction_max": float(
            max(float(row["tail_remainder_abs_fraction"]) for row in order_rows)
        ),
        "tail_cutoff_rel_spread": cutoff_rel_spread,
    }


# 関数: patched-tail weighted-integral followup の監査 pack を返す。

def build_trial2_beta_sensitivity_patched_tail_weighted_integral_followup_pack() -> dict:
    """Return one audit pack for the patched-tail weighted-integral route."""
    weighted_rows = [
        build_patched_weighted_integral_row(h_value, x_max)
        for x_max in PATCHED_TAIL_X_MAX_VALUES
        for h_value in H_VALUES
    ]
    order_summaries = {
        str(order): summarize_patched_order_rows(weighted_rows, order)
        for order in PATCHED_TAIL_ORDERS
    }

    patched_tail_profile_available_now = bool(
        all(
            row["patched_profile_positive_now"] and row["patched_tail_positive_now"]
            for row in weighted_rows
        )
    )
    patched_tail_weighted_integral_sign_support_available_now = bool(
        patched_tail_profile_available_now
        and all(
            order_row["weighted_total_negative_now"]
            for row in weighted_rows
            for order_row in row["order_rows"]
        )
    )
    patched_tail_remainder_nonreversing_now = bool(
        all(
            order_row["weighted_tail_nonreversing_now"]
            for row in weighted_rows
            for order_row in row["order_rows"]
        )
    )
    patched_tail_cutoff_stable_now = bool(
        all(
            float(order_summaries[str(order)]["tail_cutoff_rel_spread"])
            <= TAIL_CUTOFF_REL_SPREAD_TOL
            for order in PATCHED_TAIL_ORDERS
        )
    )
    exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now = False
    updated_pack_trial2_patched_tail_remainder_bound_followup_required_now = bool(
        patched_tail_weighted_integral_sign_support_available_now
        and patched_tail_remainder_nonreversing_now
        and patched_tail_cutoff_stable_now
        and not exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now
    )

    return {
        "beta_common_root": float(BETA_COMMON_ROOT),
        "tail_match_x": float(TAIL_MATCH_X),
        "x_max_values": [float(value) for value in PATCHED_TAIL_X_MAX_VALUES],
        "h_values": [float(value) for value in H_VALUES],
        "tail_remainder_fraction_tol": float(TAIL_REMAINDER_FRACTION_TOL),
        "tail_cutoff_rel_spread_tol": float(TAIL_CUTOFF_REL_SPREAD_TOL),
        "weighted_rows": weighted_rows,
        "order_summaries": order_summaries,
        "u_beta_tail_positive_fraction_min": float(
            min(float(row["u_beta_tail_positive_fraction"]) for row in weighted_rows)
        ),
        "u_beta_tail_positive_fraction_max": float(
            max(float(row["u_beta_tail_positive_fraction"]) for row in weighted_rows)
        ),
        "u_beta_tail_min": float(
            min(float(row["u_beta_tail_min"]) for row in weighted_rows)
        ),
        "u_beta_tail_max": float(
            max(float(row["u_beta_tail_max"]) for row in weighted_rows)
        ),
        "exact_trial2_beta_sensitivity_patched_tail_profile_available_now": (
            patched_tail_profile_available_now
        ),
        "exact_trial2_beta_sensitivity_patched_tail_weighted_integral_sign_support_available_now": (
            patched_tail_weighted_integral_sign_support_available_now
        ),
        "exact_trial2_beta_sensitivity_patched_tail_remainder_nonreversing_now": (
            patched_tail_remainder_nonreversing_now
        ),
        "exact_trial2_beta_sensitivity_patched_tail_cutoff_stable_now": (
            patched_tail_cutoff_stable_now
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now": (
            exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now
        ),
        "updated_pack_trial2_patched_tail_remainder_bound_followup_required_now": (
            updated_pack_trial2_patched_tail_remainder_bound_followup_required_now
        ),
    }


# 関数: backend 単体実行時に retained metrics を表示する。

def main() -> None:
    """Run the patched-tail weighted-integral backend directly."""
    pack = build_trial2_beta_sensitivity_patched_tail_weighted_integral_followup_pack()
    print("[trial2-beta-patched-tail-weighted-integral-followup]")
    print(f"beta_common_root = {pack['beta_common_root']:.16f}")
    print(f"tail_match_x = {pack['tail_match_x']:.1f}")
    print(
        "u_beta_tail_positive_fraction_range = "
        f"{pack['u_beta_tail_positive_fraction_min']:.12f} .. "
        f"{pack['u_beta_tail_positive_fraction_max']:.12f}"
    )
    for order in PATCHED_TAIL_ORDERS:
        summary = pack["order_summaries"][str(order)]
        print(
            f"dI{order}_total_range = "
            f"{summary['weighted_total_integral_min']:.16e} .. "
            f"{summary['weighted_total_integral_max']:.16e}"
        )
        print(
            f"dI{order}_tail_fraction_max = "
            f"{summary['tail_remainder_abs_fraction_max']:.16e}"
        )
        print(
            f"dI{order}_tail_cutoff_rel_spread = "
            f"{summary['tail_cutoff_rel_spread']:.16e}"
        )

    print(
        "patched_tail_weighted_integral_sign_support = "
        f"{pack['exact_trial2_beta_sensitivity_patched_tail_weighted_integral_sign_support_available_now']}"
    )
    print(
        "patched_tail_remainder_nonreversing = "
        f"{pack['exact_trial2_beta_sensitivity_patched_tail_remainder_nonreversing_now']}"
    )
    print(
        "patched_tail_cutoff_stable = "
        f"{pack['exact_trial2_beta_sensitivity_patched_tail_cutoff_stable_now']}"
    )


if __name__ == "__main__":
    main()
