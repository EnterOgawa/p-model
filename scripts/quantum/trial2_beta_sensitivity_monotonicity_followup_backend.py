#!/usr/bin/env python3
"""Audit the maximum-principle path inside the Trial-2 beta-sensitivity route.

Purpose:
    Continue the strict-theorem program after the exact beta-sensitivity
    equation has been materialized. This branch checks whether the transformed
    beta-sensitivity operator is inverse-positive on the canonical window, so a
    classical maximum-principle proof of monotonicity could close directly.

Inputs:
    - scripts/quantum/trial2_beta_sensitivity_equation_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5647-.5654` wrappers

Assumptions:
    - The practical common-root selector remains fixed at beta_common_root
    - No new parameter is introduced
    - The route tests only the maximum-principle path; it does not replay old
      q-selector, residue, or spectral heuristics
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import eigh_tridiagonal


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_beta_sensitivity_equation_backend import BETA_COMMON_ROOT
from scripts.quantum.trial2_beta_sensitivity_equation_backend import WINDOW_X_MAX
from scripts.quantum.trial2_beta_sensitivity_equation_backend import WINDOW_X_MIN
from scripts.quantum.trial2_beta_sensitivity_equation_backend import build_profile_row


WINDOW_POINT_COUNT = 6000
WINDOW_X_MAX_VALUES = (5.0, 10.0, 12.0, 15.0, 20.0)
EIGENVALUE_SIGN_FLIP_BRACKET = (10.0, 15.0)
EIGENVALUE_SIGN_FLIP_ITERATIONS = 48
POTENTIAL_ROOT_SCAN_COUNT = 50000
INNER_WINDOW_X_MAX = 10.0
CANONICAL_WINDOW_X_MAX = 20.0


# 関数: retained beta row の cubic interpolation を返す。
@lru_cache(maxsize=1)
def get_profile_interpolant():
    """Return the retained profile interpolant at beta_common_root."""
    row = build_profile_row(float(BETA_COMMON_ROOT))
    return interp1d(
        row["radius"],
        row["profile"],
        kind="cubic",
        bounds_error=False,
        fill_value=0.0,
    )


# 関数: transformed operator potential を評価する。

def evaluate_transformed_potential(grid: np.ndarray) -> np.ndarray:
    """Return V_beta(x) in the transformed equation w'' + V_beta w = -2 beta x y."""
    profile = get_profile_interpolant()(grid)
    beta = float(BETA_COMMON_ROOT)
    return beta * beta - 1.0 + 6.0 * profile + 3.0 * np.square(profile)


# 関数: transformed operator source を評価する。

def evaluate_transformed_source(grid: np.ndarray) -> np.ndarray:
    """Return the positive source 2 beta x y_beta in H_beta w = 2 beta x y_beta."""
    profile = get_profile_interpolant()(grid)
    beta = float(BETA_COMMON_ROOT)
    return 2.0 * beta * grid * profile


# 関数: Dirichlet principal eigenvalue を返す。

def compute_principal_dirichlet_eigenvalue(window_x_max: float) -> dict:
    """Return the principal Dirichlet eigenvalue of H_beta on [x_min, window_x_max]."""
    grid = np.linspace(WINDOW_X_MIN, float(window_x_max), WINDOW_POINT_COUNT, dtype=float)
    potential = evaluate_transformed_potential(grid)
    step = float(grid[1] - grid[0])
    diagonal = 2.0 / (step * step) - potential[1:-1]
    off_diagonal = np.full(diagonal.size - 1, -1.0 / (step * step), dtype=float)
    eigenvalue = float(
        eigh_tridiagonal(
            diagonal,
            off_diagonal,
            select="i",
            select_range=(0, 0),
            check_finite=False,
        )[0][0]
    )
    return {
        "window_x_max": float(window_x_max),
        "principal_dirichlet_eigenvalue": eigenvalue,
        "maximum_principle_available_now": bool(eigenvalue > 0.0),
    }


# 関数: principal eigenvalue の sign-flip root を返す。

def locate_principal_eigenvalue_sign_flip() -> float:
    """Return the x-window where the principal Dirichlet eigenvalue changes sign."""
    left, right = EIGENVALUE_SIGN_FLIP_BRACKET
    left_value = compute_principal_dirichlet_eigenvalue(left)["principal_dirichlet_eigenvalue"]
    right_value = compute_principal_dirichlet_eigenvalue(right)["principal_dirichlet_eigenvalue"]
    if not (left_value > 0.0 and right_value < 0.0):
        raise SystemExit("[fail] principal-eigenvalue sign-flip bracket is invalid")

    for _ in range(EIGENVALUE_SIGN_FLIP_ITERATIONS):
        middle = 0.5 * (left + right)
        middle_value = compute_principal_dirichlet_eigenvalue(middle)[
            "principal_dirichlet_eigenvalue"
        ]
        if middle_value > 0.0:
            left = middle
        else:
            right = middle

    return 0.5 * (left + right)


# 関数: transformed potential の zero crossing を返す。

def locate_potential_zero_crossing() -> float:
    """Return the first x where the transformed potential V_beta crosses zero."""
    row = build_profile_row(float(BETA_COMMON_ROOT))
    grid = np.linspace(
        WINDOW_X_MIN,
        float(row["radius"][-1]),
        POTENTIAL_ROOT_SCAN_COUNT,
        dtype=float,
    )
    potential = evaluate_transformed_potential(grid)
    crossing_indices = np.where(np.signbit(potential[:-1]) != np.signbit(potential[1:]))[0]
    if crossing_indices.size == 0:
        raise SystemExit("[fail] transformed potential never crosses zero")

    idx = int(crossing_indices[0])
    left_x = float(grid[idx])
    right_x = float(grid[idx + 1])
    left_value = float(potential[idx])
    right_value = float(potential[idx + 1])
    return left_x - left_value * (right_x - left_x) / (right_value - left_value)


# 関数: monotonicity followup の監査 pack を返す。

def build_trial2_beta_sensitivity_monotonicity_followup_pack() -> dict:
    """Return one audit pack for the maximum-principle followup branch."""
    eigenvalue_rows = [
        compute_principal_dirichlet_eigenvalue(window_x_max)
        for window_x_max in WINDOW_X_MAX_VALUES
    ]
    eigenvalue_map = {
        row["window_x_max"]: row["principal_dirichlet_eigenvalue"] for row in eigenvalue_rows
    }
    canonical_grid = np.linspace(
        WINDOW_X_MIN,
        CANONICAL_WINDOW_X_MAX,
        POTENTIAL_ROOT_SCAN_COUNT,
        dtype=float,
    )
    canonical_potential = evaluate_transformed_potential(canonical_grid)
    canonical_source = evaluate_transformed_source(canonical_grid)
    transformed_operator_formula = (
        "w_beta = x u_beta, "
        "H_beta w_beta = 2 beta x y_beta, "
        "H_beta = -d^2/dx^2 - (beta^2 - 1 + 6 y_beta + 3 y_beta^2)"
    )
    sign_flip_root = locate_principal_eigenvalue_sign_flip()
    potential_zero_root = locate_potential_zero_crossing()
    inner_window_eigenvalue = eigenvalue_map[INNER_WINDOW_X_MAX]
    canonical_window_eigenvalue = eigenvalue_map[CANONICAL_WINDOW_X_MAX]
    maximum_principle_available_on_inner_window_now = bool(inner_window_eigenvalue > 0.0)
    maximum_principle_available_on_canonical_window_now = bool(canonical_window_eigenvalue > 0.0)
    transformed_source_positive_on_canonical_window_now = bool(
        float(np.min(canonical_source)) > 0.0
    )
    transformed_potential_positive_on_canonical_window_now = bool(
        float(np.min(canonical_potential)) > 0.0
    )
    principal_eigenvalue_sign_flip_available_now = bool(
        inner_window_eigenvalue > 0.0 and canonical_window_eigenvalue < 0.0
    )
    maximum_principle_negative_closeout_available_now = bool(
        transformed_source_positive_on_canonical_window_now
        and transformed_potential_positive_on_canonical_window_now
        and not maximum_principle_available_on_canonical_window_now
        and principal_eigenvalue_sign_flip_available_now
    )
    green_kernel_followup_required_now = bool(
        maximum_principle_negative_closeout_available_now
    )
    return {
        "beta_common_root": float(BETA_COMMON_ROOT),
        "transformed_operator_formula": transformed_operator_formula,
        "eigenvalue_rows": eigenvalue_rows,
        "window5_principal_dirichlet_eigenvalue": float(eigenvalue_map[5.0]),
        "window10_principal_dirichlet_eigenvalue": float(eigenvalue_map[10.0]),
        "window12_principal_dirichlet_eigenvalue": float(eigenvalue_map[12.0]),
        "window15_principal_dirichlet_eigenvalue": float(eigenvalue_map[15.0]),
        "window20_principal_dirichlet_eigenvalue": float(eigenvalue_map[20.0]),
        "principal_dirichlet_sign_flip_root_x_max": float(sign_flip_root),
        "transformed_potential_zero_crossing_x": float(potential_zero_root),
        "canonical_window_potential_min": float(np.min(canonical_potential)),
        "canonical_window_potential_max": float(np.max(canonical_potential)),
        "canonical_window_source_min": float(np.min(canonical_source)),
        "canonical_window_source_max": float(np.max(canonical_source)),
        "exact_trial2_beta_sensitivity_transformed_operator_available_now": True,
        "transformed_source_positive_on_canonical_window_now": (
            transformed_source_positive_on_canonical_window_now
        ),
        "transformed_potential_positive_on_canonical_window_now": (
            transformed_potential_positive_on_canonical_window_now
        ),
        "maximum_principle_available_on_inner_window_now": (
            maximum_principle_available_on_inner_window_now
        ),
        "maximum_principle_available_on_canonical_window_now": (
            maximum_principle_available_on_canonical_window_now
        ),
        "principal_eigenvalue_sign_flip_available_now": (
            principal_eigenvalue_sign_flip_available_now
        ),
        "maximum_principle_negative_closeout_available_now": (
            maximum_principle_negative_closeout_available_now
        ),
        "green_kernel_followup_required_now": green_kernel_followup_required_now,
        "exact_trial2_common_root_monotonicity_theorem_available_now": False,
    }


# 関数: backend 単体実行時に retained metrics を表示する。

def main() -> None:
    """Run the monotonicity followup backend directly and print the retained metrics."""
    pack = build_trial2_beta_sensitivity_monotonicity_followup_pack()
    print("[trial2-beta-monotonicity-followup]")
    print(f"beta_common_root = {pack['beta_common_root']:.16f}")
    for row in pack["eigenvalue_rows"]:
        print(
            "window_x_max="
            f"{row['window_x_max']:.1f} "
            f"lambda1={row['principal_dirichlet_eigenvalue']:.16f}"
        )

    print(
        "principal_dirichlet_sign_flip_root_x_max = "
        f"{pack['principal_dirichlet_sign_flip_root_x_max']:.16f}"
    )
    print(
        "transformed_potential_zero_crossing_x = "
        f"{pack['transformed_potential_zero_crossing_x']:.16f}"
    )
    print(
        "maximum_principle_available_on_canonical_window_now = "
        f"{pack['maximum_principle_available_on_canonical_window_now']}"
    )
    print(
        "green_kernel_followup_required_now = "
        f"{pack['green_kernel_followup_required_now']}"
    )


if __name__ == "__main__":
    main()
