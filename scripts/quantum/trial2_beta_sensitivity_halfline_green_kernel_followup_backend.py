#!/usr/bin/env python3
"""Audit a patched half-line Green-kernel route for pure analytic refinement.

Purpose:
    Reopen the pure analytic operator-level continuum program on a genuinely
    new branch after `.5735-.5742`, where one patched weighted-integral
    pure-continuum layer is already synchronized into the v2 theorem wording.

    The honest remaining question is narrower:

        once the right Dirichlet wall is replaced by a decaying half-line
        boundary condition compatible with the admissible patched tail, does
        the beta-sensitivity equation admit one global one-sign Green kernel,
        or does the route still require a source-weighted comparison theorem?

    This backend therefore does not replay the direct-alpha closure. It checks:

    1. the half-line homogeneous fundamental pair and sampled Green columns,
    2. the actual source-weighted half-line BVP solution on the physical
       control window,
    3. the matching Green-convolution solution on that same control window,
    4. the naive positive-Yukawa contraction test, and
    5. whether the route closes negatively at the kernel level while still
       exposing one sharper source-weighted comparison followup.

Inputs:
    - scripts/quantum/trial2_beta_sensitivity_patched_tail_weighted_integral_followup_backend.py
    - scripts/quantum/trial2_beta_sensitivity_admissible_tail_patch_followup_backend.py
    - scripts/quantum/trial2_beta_sensitivity_equation_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5743-.5750` wrappers

Assumptions:
    - The retained common-root selector stays fixed at beta_common_root
    - The admissible positive-decay patched tail remains the only allowed
      half-line continuation
    - No new parameter is introduced
    - This branch targets one new operator-level continuum route only; it does
      not weaken the already-completed first-principles direct-alpha closure
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import solve_bvp
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_beta_sensitivity_admissible_tail_patch_followup_backend import (
    TAIL_MATCH_X,
)
from scripts.quantum.trial2_beta_sensitivity_admissible_tail_patch_followup_backend import (
    locate_first_zero_crossing,
)
from scripts.quantum.trial2_beta_sensitivity_equation_backend import BETA_COMMON_ROOT
from scripts.quantum.trial2_beta_sensitivity_patched_tail_weighted_integral_followup_backend import (
    build_patched_profile_row,
)


HALFLINE_X_MIN = 0.05
CONTROL_WINDOW_X_MIN = 0.10
CONTROL_WINDOW_X_MAX = 19.90
HALFLINE_LEFT_REGULAR_X = 1.0e-6
HALFLINE_X_MAX_VALUES = (80.0, 100.0, 140.0)
HALFLINE_GRID_POINT_COUNT = 1200
GREEN_PROBE_FRACTIONS = (0.25, 0.50, 0.75)
TAIL_POSITIVE_X_MIN = 21.0
TAIL_H_VALUE = 1.0e-6
IVP_RTOL = 1.0e-9
IVP_ATOL = 1.0e-11
BVP_TOL = 1.0e-5
GREEN_BVP_REL_LINF_TOL = 5.0e-2
GREEN_BVP_CORREL_TOL = 9.9e-1


# 関数: patched half-line operator row を返す。
def build_halfline_operator_row(x_max: float) -> dict:
    """Return one patched half-line operator row."""
    beta_common_root = float(BETA_COMMON_ROOT)
    x_max = float(x_max)
    patched_row = build_patched_profile_row(beta_common_root, x_max)
    grid = np.linspace(HALFLINE_X_MIN, x_max, HALFLINE_GRID_POINT_COUNT, dtype=float)
    profile = np.interp(grid, patched_row["radius"], patched_row["profile"])
    potential = (
        beta_common_root * beta_common_root
        - 1.0
        + 6.0 * profile
        + 3.0 * np.square(profile)
    )
    source = 2.0 * beta_common_root * grid * profile
    control_mask = (grid >= CONTROL_WINDOW_X_MIN) & (grid <= CONTROL_WINDOW_X_MAX)
    tail_mask = grid >= float(TAIL_POSITIVE_X_MIN)
    if not np.any(control_mask):
        raise SystemExit("[fail] half-line control-window mask is empty")

    return {
        "beta_common_root": beta_common_root,
        "x_max": x_max,
        "grid": grid,
        "profile": profile,
        "potential": potential,
        "source": source,
        "control_mask": control_mask,
        "tail_mask": tail_mask,
        "kappa": float(patched_row["kappa"]),
        "patched_radius": np.asarray(patched_row["radius"], dtype=float),
        "patched_profile": np.asarray(patched_row["profile"], dtype=float),
    }


# 関数: patched half-line 上の potential を評価する。

def evaluate_halfline_potential(x_values: np.ndarray, operator_row: dict) -> np.ndarray:
    """Return the transformed potential on the requested half-line points."""
    x_values = np.asarray(x_values, dtype=float)
    profile = np.interp(
        x_values,
        operator_row["patched_radius"],
        operator_row["patched_profile"],
    )
    beta_common_root = float(operator_row["beta_common_root"])
    return (
        beta_common_root * beta_common_root
        - 1.0
        + 6.0 * profile
        + 3.0 * np.square(profile)
    )


# 関数: patched half-line 上の source を評価する。

def evaluate_halfline_source(x_values: np.ndarray, operator_row: dict) -> np.ndarray:
    """Return the transformed source on the requested half-line points."""
    x_values = np.asarray(x_values, dtype=float)
    profile = np.interp(
        x_values,
        operator_row["patched_radius"],
        operator_row["patched_profile"],
    )
    beta_common_root = float(operator_row["beta_common_root"])
    return 2.0 * beta_common_root * x_values * profile


# 関数: homogeneous ODE RHS を返す。

def build_homogeneous_rhs(operator_row: dict):
    """Return the homogeneous first-order ODE for the half-line operator."""

    # 関数: homogeneous pair の RHS を評価する。
    def rhs(x_value: float, state: np.ndarray) -> np.ndarray:
        potential = float(
            evaluate_halfline_potential(np.asarray([x_value], dtype=float), operator_row)[0]
        )
        return np.asarray([state[1], -potential * state[0]], dtype=float)

    return rhs


# 関数: half-line homogeneous fundamental pair row を返す。

def build_homogeneous_pair_row(operator_row: dict) -> dict:
    """Return one half-line homogeneous fundamental-pair row."""
    grid = np.asarray(operator_row["grid"], dtype=float)
    rhs = build_homogeneous_rhs(operator_row)
    x_max = float(operator_row["x_max"])
    kappa = float(operator_row["kappa"])

    left_eval = np.concatenate(
        (np.asarray([HALFLINE_LEFT_REGULAR_X], dtype=float), grid),
    )
    left_solution = solve_ivp(
        rhs,
        (HALFLINE_LEFT_REGULAR_X, x_max),
        np.asarray([HALFLINE_LEFT_REGULAR_X, 1.0], dtype=float),
        t_eval=left_eval,
        rtol=IVP_RTOL,
        atol=IVP_ATOL,
    )
    if not left_solution.success:
        raise SystemExit("[fail] half-line left regular solve failed")

    right_boundary_value = math.exp(-kappa * x_max)
    right_solution = solve_ivp(
        rhs,
        (x_max, HALFLINE_LEFT_REGULAR_X),
        np.asarray([right_boundary_value, -kappa * right_boundary_value], dtype=float),
        t_eval=grid[::-1],
        rtol=IVP_RTOL,
        atol=IVP_ATOL,
    )
    if not right_solution.success:
        raise SystemExit("[fail] half-line right decaying solve failed")

    left_w = np.asarray(left_solution.y[0][1:], dtype=float)
    left_w_prime = np.asarray(left_solution.y[1][1:], dtype=float)
    right_w = np.asarray(right_solution.y[0][::-1], dtype=float)
    right_w_prime = np.asarray(right_solution.y[1][::-1], dtype=float)
    wronskian = left_w * right_w_prime - left_w_prime * right_w

    return {
        "left_zero_crossing_x": locate_first_zero_crossing(grid, left_w),
        "right_zero_crossing_x": locate_first_zero_crossing(grid, right_w),
        "left_w": left_w,
        "left_w_prime": left_w_prime,
        "right_w": right_w,
        "right_w_prime": right_w_prime,
        "wronskian_min": float(np.min(wronskian)),
        "wronskian_max": float(np.max(wronskian)),
        "wronskian_rel_spread": float(
            (float(np.max(wronskian)) - float(np.min(wronskian)))
            / max(abs(float(np.mean(wronskian))), 1.0e-30)
        ),
        "wronskian_values": wronskian,
    }


# 関数: sampled half-line Green column row を返す。

def build_green_column_row(
    operator_row: dict,
    homogeneous_row: dict,
    probe_fraction: float,
) -> dict:
    """Return one sampled half-line Green-column sign row."""
    grid = np.asarray(operator_row["grid"], dtype=float)
    left_w = np.asarray(homogeneous_row["left_w"], dtype=float)
    right_w = np.asarray(homogeneous_row["right_w"], dtype=float)
    wronskian = np.asarray(homogeneous_row["wronskian_values"], dtype=float)
    diagonal_size = int(grid.size)
    column_index = int(round(float(probe_fraction) * float(diagonal_size - 1)))
    xi = float(grid[column_index])
    kernel = np.empty_like(grid)
    left_mask = grid <= xi
    denominator = float(wronskian[column_index])
    kernel[left_mask] = left_w[left_mask] * right_w[column_index] / denominator
    kernel[~left_mask] = left_w[column_index] * right_w[~left_mask] / denominator
    return {
        "probe_fraction": float(probe_fraction),
        "column_index": column_index,
        "probe_x": xi,
        "column_min": float(np.min(kernel)),
        "column_max": float(np.max(kernel)),
        "column_negative_fraction": float(np.mean(kernel < 0.0)),
        "column_positive_fraction": float(np.mean(kernel > 0.0)),
        "column_sign_mixed_now": bool(
            float(np.min(kernel)) < 0.0 and float(np.max(kernel)) > 0.0
        ),
        "column_all_negative_now": bool(float(np.max(kernel)) < 0.0),
    }


# 関数: beta-sensitivity finite-difference guess row を返す。

def build_halfline_bvp_guess_row(operator_row: dict) -> dict:
    """Return one finite-difference initial guess for the half-line BVP."""
    beta_common_root = float(BETA_COMMON_ROOT)
    x_max = float(operator_row["x_max"])
    grid = np.asarray(operator_row["grid"], dtype=float)
    plus_row = build_patched_profile_row(beta_common_root + TAIL_H_VALUE, x_max)
    minus_row = build_patched_profile_row(beta_common_root - TAIL_H_VALUE, x_max)
    profile_plus = np.interp(grid, plus_row["radius"], plus_row["profile"])
    profile_minus = np.interp(grid, minus_row["radius"], minus_row["profile"])
    u_beta = (profile_plus - profile_minus) / (2.0 * TAIL_H_VALUE)
    w_guess = grid * u_beta
    return {
        "w_guess": w_guess,
        "w_guess_prime": np.gradient(w_guess, grid),
    }


# 関数: half-line source-weighted BVP row を返す。

def build_halfline_bvp_row(operator_row: dict) -> dict:
    """Return one source-weighted half-line BVP sign row."""
    grid = np.asarray(operator_row["grid"], dtype=float)
    kappa = float(operator_row["kappa"])
    guess_row = build_halfline_bvp_guess_row(operator_row)
    control_mask = np.asarray(operator_row["control_mask"], dtype=bool)
    tail_mask = np.asarray(operator_row["tail_mask"], dtype=bool)

    # 関数: BVP の微分方程式を返す。
    def ode(x_values: np.ndarray, values: np.ndarray) -> np.ndarray:
        potential = evaluate_halfline_potential(x_values, operator_row)
        source = evaluate_halfline_source(x_values, operator_row)
        return np.vstack((values[1], -potential * values[0] - source))

    # 関数: regular-left / decaying-right boundary condition を返す。

    def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        return np.asarray([ya[0], yb[1] + kappa * yb[0]], dtype=float)

    solution = solve_bvp(
        ode,
        bc,
        grid,
        np.vstack((guess_row["w_guess"], guess_row["w_guess_prime"])),
        tol=BVP_TOL,
        max_nodes=50000,
    )
    if not solution.success:
        raise SystemExit("[fail] half-line source-weighted BVP solve failed")

    w_solution = np.asarray(solution.sol(grid)[0], dtype=float)
    return {
        "bvp_success": bool(solution.success),
        "control_solution_min": float(np.min(w_solution[control_mask])),
        "control_solution_max": float(np.max(w_solution[control_mask])),
        "control_negative_fraction": float(np.mean(w_solution[control_mask] < 0.0)),
        "full_solution_min": float(np.min(w_solution)),
        "full_solution_max": float(np.max(w_solution)),
        "full_negative_fraction": float(np.mean(w_solution < 0.0)),
        "tail_positive_fraction": float(np.mean(w_solution[tail_mask] > 0.0)),
        "tail_positive_max": float(np.max(w_solution[tail_mask])),
        "solution_values": w_solution,
    }


# 関数: half-line Green-convolution row を返す。

def build_green_convolution_row(operator_row: dict, homogeneous_row: dict) -> dict:
    """Return one source-weighted Green-convolution sign row."""
    grid = np.asarray(operator_row["grid"], dtype=float)
    left_w = np.asarray(homogeneous_row["left_w"], dtype=float)
    right_w = np.asarray(homogeneous_row["right_w"], dtype=float)
    source = np.asarray(operator_row["source"], dtype=float)
    control_mask = np.asarray(operator_row["control_mask"], dtype=bool)
    wronskian_mean = float(np.mean(np.asarray(homogeneous_row["wronskian_values"], dtype=float)))
    left_source = left_w * source
    right_source = right_w * source
    left_integral = cumulative_trapezoid(left_source, grid, initial=0.0)
    right_integral_reverse = cumulative_trapezoid(
        right_source[::-1],
        grid[::-1],
        initial=0.0,
    )
    right_integral = -right_integral_reverse[::-1]
    solution = -(right_w * left_integral + left_w * right_integral) / wronskian_mean
    return {
        "control_solution_min": float(np.min(solution[control_mask])),
        "control_solution_max": float(np.max(solution[control_mask])),
        "control_negative_fraction": float(np.mean(solution[control_mask] < 0.0)),
        "solution_values": solution,
    }


# 関数: BVP と Green-convolution の control-window coherence row を返す。

def build_control_coherence_row(
    operator_row: dict,
    bvp_row: dict,
    green_row: dict,
) -> dict:
    """Return one control-window coherence row between BVP and Green solutions."""
    control_mask = np.asarray(operator_row["control_mask"], dtype=bool)
    bvp_values = np.asarray(bvp_row["solution_values"], dtype=float)[control_mask]
    green_values = np.asarray(green_row["solution_values"], dtype=float)[control_mask]
    numerator = float(np.max(np.abs(green_values - bvp_values)))
    denominator = float(max(np.max(np.abs(bvp_values)), 1.0e-30))
    correlation = float(np.corrcoef(green_values, bvp_values)[0, 1])
    return {
        "green_bvp_control_rel_linf": float(numerator / denominator),
        "green_bvp_control_corrcoef": correlation,
        "green_bvp_control_consistent_now": bool(
            float(numerator / denominator) <= GREEN_BVP_REL_LINF_TOL
            and correlation >= GREEN_BVP_CORREL_TOL
        ),
    }


# 関数: naive positive-Yukawa contraction row を返す。

def build_positive_yukawa_contraction_row(operator_row: dict) -> dict:
    """Return one naive positive-Yukawa contraction diagnostic row."""
    grid = np.asarray(operator_row["grid"], dtype=float)
    kappa = float(operator_row["kappa"])
    profile = np.asarray(operator_row["profile"], dtype=float)
    potential_positive = 6.0 * profile + 3.0 * np.square(profile)
    min_grid = np.minimum.outer(grid, grid)
    max_grid = np.maximum.outer(grid, grid)
    kernel = np.sinh(kappa * min_grid) * np.exp(-kappa * max_grid) / kappa
    contraction_curve = np.trapezoid(kernel * potential_positive[None, :], grid, axis=1)
    argmax_index = int(np.argmax(contraction_curve))
    contraction_sup = float(contraction_curve[argmax_index])
    return {
        "yukawa_contraction_sup": contraction_sup,
        "yukawa_contraction_argmax_x": float(grid[argmax_index]),
        "yukawa_contraction_available_now": bool(contraction_sup < 1.0),
    }


# 関数: one x_max row をまとめて返す。

def build_halfline_route_row(x_max: float) -> dict:
    """Return one patched half-line route row for the selected truncation."""
    operator_row = build_halfline_operator_row(x_max)
    homogeneous_row = build_homogeneous_pair_row(operator_row)
    sampled_columns = [
        build_green_column_row(operator_row, homogeneous_row, probe_fraction)
        for probe_fraction in GREEN_PROBE_FRACTIONS
    ]
    bvp_row = build_halfline_bvp_row(operator_row)
    green_row = build_green_convolution_row(operator_row, homogeneous_row)
    coherence_row = build_control_coherence_row(operator_row, bvp_row, green_row)
    yukawa_row = build_positive_yukawa_contraction_row(operator_row)
    return {
        "x_max": float(x_max),
        "kappa": float(operator_row["kappa"]),
        "left_zero_crossing_x": homogeneous_row["left_zero_crossing_x"],
        "right_zero_crossing_x": homogeneous_row["right_zero_crossing_x"],
        "wronskian_min": float(homogeneous_row["wronskian_min"]),
        "wronskian_max": float(homogeneous_row["wronskian_max"]),
        "wronskian_rel_spread": float(homogeneous_row["wronskian_rel_spread"]),
        "sampled_columns": sampled_columns,
        "bvp_row": {key: value for key, value in bvp_row.items() if key != "solution_values"},
        "green_row": {key: value for key, value in green_row.items() if key != "solution_values"},
        "coherence_row": coherence_row,
        "yukawa_row": yukawa_row,
        "kernel_sign_mixed_now": bool(any(row["column_sign_mixed_now"] for row in sampled_columns)),
        "control_solution_negative_now": bool(
            bvp_row["control_negative_fraction"] == 1.0
            and green_row["control_negative_fraction"] == 1.0
            and coherence_row["green_bvp_control_consistent_now"]
        ),
        "tail_positive_leak_now": bool(float(bvp_row["tail_positive_fraction"]) > 0.0),
    }


# 関数: half-line Green-kernel followup の監査 pack を返す。

def build_trial2_beta_sensitivity_halfline_green_kernel_followup_pack() -> dict:
    """Return one audit pack for the patched half-line Green-kernel route."""
    route_rows = [build_halfline_route_row(x_max) for x_max in HALFLINE_X_MAX_VALUES]
    retained_row = route_rows[-1]
    retained_probe_075 = next(
        row for row in retained_row["sampled_columns"] if abs(row["probe_fraction"] - 0.75) < 1.0e-12
    )
    halfline_green_kernel_available_now = True
    halfline_green_kernel_one_sign_available_now = bool(
        all(not row["kernel_sign_mixed_now"] for row in route_rows)
    )
    halfline_source_weighted_bvp_negative_control_window_now = bool(
        all(
            row["control_solution_negative_now"]
            and row["bvp_row"]["control_negative_fraction"] == 1.0
            and row["green_row"]["control_negative_fraction"] == 1.0
            for row in route_rows
        )
    )
    halfline_yukawa_contraction_available_now = bool(
        all(row["yukawa_row"]["yukawa_contraction_available_now"] for row in route_rows)
    )
    exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now = False
    halfline_green_kernel_negative_closeout_available_now = bool(
        halfline_green_kernel_available_now
        and not halfline_green_kernel_one_sign_available_now
        and halfline_source_weighted_bvp_negative_control_window_now
    )
    updated_pack_trial2_source_weighted_comparison_followup_required_now = bool(
        halfline_green_kernel_negative_closeout_available_now
        and not halfline_yukawa_contraction_available_now
        and not exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now
    )

    return {
        "beta_common_root": float(BETA_COMMON_ROOT),
        "tail_match_x": float(TAIL_MATCH_X),
        "halfline_x_min": float(HALFLINE_X_MIN),
        "control_window_x_min": float(CONTROL_WINDOW_X_MIN),
        "control_window_x_max": float(CONTROL_WINDOW_X_MAX),
        "route_rows": route_rows,
        "retained_x_max": float(retained_row["x_max"]),
        "retained_left_zero_crossing_x": retained_row["left_zero_crossing_x"],
        "retained_right_zero_crossing_x": retained_row["right_zero_crossing_x"],
        "retained_wronskian_min": float(retained_row["wronskian_min"]),
        "retained_wronskian_max": float(retained_row["wronskian_max"]),
        "retained_probe_075_negative_fraction": float(
            retained_probe_075["column_negative_fraction"]
        ),
        "retained_probe_075_positive_fraction": float(
            retained_probe_075["column_positive_fraction"]
        ),
        "retained_probe_075_kernel_min": float(retained_probe_075["column_min"]),
        "retained_probe_075_kernel_max": float(retained_probe_075["column_max"]),
        "retained_bvp_control_solution_min": float(
            retained_row["bvp_row"]["control_solution_min"]
        ),
        "retained_bvp_control_solution_max": float(
            retained_row["bvp_row"]["control_solution_max"]
        ),
        "retained_bvp_tail_positive_fraction": float(
            retained_row["bvp_row"]["tail_positive_fraction"]
        ),
        "retained_green_control_solution_min": float(
            retained_row["green_row"]["control_solution_min"]
        ),
        "retained_green_control_solution_max": float(
            retained_row["green_row"]["control_solution_max"]
        ),
        "retained_green_bvp_control_rel_linf": float(
            retained_row["coherence_row"]["green_bvp_control_rel_linf"]
        ),
        "retained_green_bvp_control_corrcoef": float(
            retained_row["coherence_row"]["green_bvp_control_corrcoef"]
        ),
        "retained_yukawa_contraction_sup": float(
            retained_row["yukawa_row"]["yukawa_contraction_sup"]
        ),
        "retained_yukawa_contraction_argmax_x": float(
            retained_row["yukawa_row"]["yukawa_contraction_argmax_x"]
        ),
        "exact_trial2_beta_sensitivity_halfline_green_kernel_available_now": (
            halfline_green_kernel_available_now
        ),
        "exact_trial2_beta_sensitivity_halfline_green_kernel_one_sign_available_now": (
            halfline_green_kernel_one_sign_available_now
        ),
        "exact_trial2_beta_sensitivity_halfline_source_weighted_bvp_negative_control_window_now": (
            halfline_source_weighted_bvp_negative_control_window_now
        ),
        "exact_trial2_beta_sensitivity_halfline_yukawa_contraction_available_now": (
            halfline_yukawa_contraction_available_now
        ),
        "exact_trial2_beta_sensitivity_halfline_green_kernel_negative_closeout_available_now": (
            halfline_green_kernel_negative_closeout_available_now
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now": (
            exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now
        ),
        "updated_pack_trial2_source_weighted_comparison_followup_required_now": (
            updated_pack_trial2_source_weighted_comparison_followup_required_now
        ),
    }


# 関数: backend 単体実行時に retained metrics を表示する。

def main() -> None:
    """Run the patched half-line Green-kernel backend directly."""
    pack = build_trial2_beta_sensitivity_halfline_green_kernel_followup_pack()
    print("[trial2-beta-halfline-green-kernel-followup]")
    print(f"beta_common_root = {pack['beta_common_root']:.16f}")
    print(f"retained_x_max = {pack['retained_x_max']:.1f}")
    print(f"retained_left_zero_crossing_x = {pack['retained_left_zero_crossing_x']}")
    print(f"retained_right_zero_crossing_x = {pack['retained_right_zero_crossing_x']}")
    print(
        "retained_probe_075_kernel = "
        f"[{pack['retained_probe_075_kernel_min']:.16f}, "
        f"{pack['retained_probe_075_kernel_max']:.16f}]"
    )
    print(
        "retained_bvp_control = "
        f"[{pack['retained_bvp_control_solution_min']:.16f}, "
        f"{pack['retained_bvp_control_solution_max']:.16f}]"
    )
    print(
        "retained_green_control = "
        f"[{pack['retained_green_control_solution_min']:.16f}, "
        f"{pack['retained_green_control_solution_max']:.16f}]"
    )
    print(
        "retained_green_bvp_control_rel_linf = "
        f"{pack['retained_green_bvp_control_rel_linf']:.16f}"
    )
    print(
        "retained_yukawa_contraction_sup = "
        f"{pack['retained_yukawa_contraction_sup']:.16f}"
    )
    print(
        "halfline_green_kernel_negative_closeout_available = "
        f"{pack['exact_trial2_beta_sensitivity_halfline_green_kernel_negative_closeout_available_now']}"
    )
    print(
        "source_weighted_comparison_followup_required = "
        f"{pack['updated_pack_trial2_source_weighted_comparison_followup_required_now']}"
    )


if __name__ == "__main__":
    main()
