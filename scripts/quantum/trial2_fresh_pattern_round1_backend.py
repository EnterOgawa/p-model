#!/usr/bin/env python3
"""Audit the expert-supplied fresh Round-1 Trial-2 q_exact patterns.

Purpose:
    Screen the expert's new pattern set without replaying exhausted Route A-D,
    blind-overlap theorem, spectral distinguished-scale, or effective coupling
    / residue branches. The helper evaluates only the low-cost Round-1
    patterns:

        epsilon: Bohr-radius / Compton-scale matching
        zeta: mean momentum transfer from |F(q)|^2 weighting
        gamma: F(q) characteristic-point conditions
        beta: nonlinear dispersion from global <y^2> correction

Inputs:
    - scripts/quantum/scalar_proxy_alpha_q_curve_backend.py
    - output/public/quantum/mass_origin_qball_charge_mapping_branch_refresh_metrics.json
    - scripts/quantum/mass_origin_qball_charge_mapping_branch.py

Outputs:
    - One in-memory diagnostic pack consumed by `.5527-.5534` wrappers

Assumptions:
    - q_exact is already fixed as a retained numerical fact
    - alpha_target is not used as an input parameter for the candidate laws
    - all candidates must be built from frozen-action data or retained profile
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.scalar_proxy_alpha_q_curve_backend import QBALL_BRANCH_REFRESH
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import alpha_from_form_factor
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import build_scalar_proxy_alpha_q_curve_pack
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import extract_scalar_ground_state
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import form_factor
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import load_qball_module
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import read_json


# 関数: retained scalar profile と charge-weighted density を読み込む。
def load_retained_profile_data() -> dict:
    """Return the retained scalar profile and its natural charge-weighted data."""
    qball_branch_refresh = read_json(QBALL_BRANCH_REFRESH)
    scalar_ground_state = extract_scalar_ground_state(qball_branch_refresh)
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
    return {
        "radius": radius,
        "profile": profile,
        "density": density,
        "weight": weight,
        "norm": norm,
        "scalar_ground_state": scalar_ground_state,
    }


# 関数: alpha(q)=alpha_candidate の root を retained interval で探索する。

def find_alpha_curve_roots(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    q_values: np.ndarray,
    alpha_curve: np.ndarray,
    alpha_candidate: float,
) -> list[float]:
    """Return all retained-interval roots of alpha(q) - alpha_candidate."""
    diff = alpha_curve - float(alpha_candidate)
    roots: list[float] = []
    for index in range(len(q_values) - 1):
        left_q = float(q_values[index])
        right_q = float(q_values[index + 1])
        left_diff = float(diff[index])
        right_diff = float(diff[index + 1])
        if left_diff == 0.0:
            roots.append(left_q)

        if left_diff * right_diff < 0.0:
            root = brentq(
                lambda q: alpha_from_form_factor(
                    form_factor(radius, weight, norm, float(q))
                )
                - float(alpha_candidate),
                left_q,
                right_q,
            )
            roots.append(float(root))

    if diff[-1] == 0.0:
        roots.append(float(q_values[-1]))

    unique_roots: list[float] = []
    for candidate in roots:
        if not unique_roots or abs(candidate - unique_roots[-1]) > 1.0e-12:
            unique_roots.append(candidate)

    return unique_roots


# 関数: q_exact 近傍の F(q) 微分量を local polynomial から安定に読む。

def build_local_form_factor_derivatives(
    q_values: np.ndarray,
    form_factor_curve: np.ndarray,
    q_exact: float,
) -> dict[str, float]:
    """Return stable local derivative diagnostics around q_exact."""
    center_index = int(np.searchsorted(q_values, q_exact))
    window = slice(max(0, center_index - 3), min(len(q_values), center_index + 4))
    local_q = q_values[window] - float(q_exact)
    local_f = form_factor_curve[window]
    polynomial = np.polyfit(local_q, local_f, 4)
    f0 = float(np.polyval(polynomial, 0.0))
    f_prime = float(polynomial[-2])
    f_double_prime = float(2.0 * polynomial[-3])
    return {
        "F_at_q_exact": f0,
        "F_prime_at_q_exact": f_prime,
        "F_double_prime_at_q_exact": f_double_prime,
        "log_derivative_at_q_exact": float(q_exact * f_prime / f0),
        "self_reference_ratio_at_q_exact": float((q_exact**2) * f_double_prime / f0),
    }


# 関数: F'' の sign change から inflection interval を返す。

def build_inflection_diagnostics(
    q_values: np.ndarray,
    form_factor_curve: np.ndarray,
    q_exact: float,
) -> dict[str, float | list[float]]:
    """Return nearest inflection diagnostics for the retained F(q) curve."""
    first_derivative = np.gradient(form_factor_curve, q_values)
    second_derivative = np.gradient(first_derivative, q_values)
    changes = np.where(np.diff(np.sign(second_derivative)) != 0)[0]
    inflection_candidates = [
        0.5 * (float(q_values[index]) + float(q_values[index + 1]))
        for index in changes
    ]
    nearest_inflection = min(
        inflection_candidates,
        key=lambda candidate: abs(candidate - float(q_exact)),
    )
    return {
        "nearest_inflection_over_m0": float(nearest_inflection),
        "nearest_inflection_distance_over_m0": float(abs(nearest_inflection - q_exact)),
        "inflection_candidates_over_m0": [float(value) for value in inflection_candidates[:8]],
        "stationary_selector_available_now": False,
    }


# 関数: q-window ごとの mean momentum transfer を返す。

def build_mean_momentum_diagnostics(
    q_values: np.ndarray,
    form_factor_curve: np.ndarray,
    q_exact: float,
) -> dict:
    """Return q-window dependent mean momentum transfer diagnostics."""
    window_maxima = (0.5, 1.0, 2.0, 5.0)
    rows: list[dict[str, float]] = []
    for q_max in window_maxima:
        mask = q_values <= float(q_max)
        weighted_power = np.square(form_factor_curve[mask]) * np.square(q_values[mask])
        numerator = np.trapezoid(q_values[mask] * weighted_power, q_values[mask])
        denominator = np.trapezoid(weighted_power, q_values[mask])
        mean_q = float(numerator / denominator)
        rows.append(
            {
                "q_window_max_over_m0": float(q_max),
                "mean_q_over_m0": mean_q,
                "relative_error_vs_q_exact": float((mean_q - q_exact) / q_exact),
            }
        )

    best_row = min(rows, key=lambda row: abs(row["relative_error_vs_q_exact"]))
    spread = max(row["mean_q_over_m0"] for row in rows) - min(
        row["mean_q_over_m0"] for row in rows
    )
    return {
        "rows": rows,
        "best_mean_q_over_m0": float(best_row["mean_q_over_m0"]),
        "best_mean_q_relative_error_vs_q_exact": float(
            best_row["relative_error_vs_q_exact"]
        ),
        "mean_q_window_spread_over_m0": float(spread),
    }


# 関数: expert fresh Round-1 patterns の diagnostic pack を構築する。

def build_trial2_fresh_pattern_round1_pack(
    q_min: float = 0.0,
    q_max: float = 5.0,
    q_count: int = 50001,
) -> dict:
    """Return the expert fresh-pattern Round-1 diagnostic pack."""
    alpha_pack = build_scalar_proxy_alpha_q_curve_pack(
        q_min=float(q_min),
        q_max=float(q_max),
        q_count=int(q_count),
    )
    profile_data = load_retained_profile_data()
    q_values = np.asarray(alpha_pack["q_values"], dtype=float)
    form_factor_curve = np.asarray(alpha_pack["form_factor_curve"], dtype=float)
    alpha_curve = np.asarray(alpha_pack["alpha_curve"], dtype=float)
    radius = profile_data["radius"]
    profile = profile_data["profile"]
    weight = profile_data["weight"]
    norm = float(profile_data["norm"])

    q_exact = float(alpha_pack["primary_q_exact_over_m0"])
    q_star = float(alpha_pack["q_star_over_m0"])
    beta1 = float(alpha_pack["scalar_ground_state"]["beta_n"])
    epsilon_beta = float(1.0 - beta1**2)
    q_star_relative_error = float((q_star - q_exact) / q_exact)

    alpha_exact = alpha_from_form_factor(form_factor(radius, weight, norm, q_exact))

    alpha_bohr_one_eighth = float(math.sqrt(epsilon_beta) / 8.0)
    alpha_bohr_roots = find_alpha_curve_roots(
        radius,
        weight,
        norm,
        q_values,
        alpha_curve,
        alpha_bohr_one_eighth,
    )
    q_bohr = float(alpha_bohr_roots[0]) if alpha_bohr_roots else math.nan
    q_bohr_relative_error = (
        float((q_bohr - q_exact) / q_exact) if alpha_bohr_roots else math.nan
    )
    alpha_bohr_relative_error = float((alpha_bohr_one_eighth - alpha_exact) / alpha_exact)

    mean_q_diagnostics = build_mean_momentum_diagnostics(q_values, form_factor_curve, q_exact)
    derivative_diagnostics = build_local_form_factor_derivatives(
        q_values,
        form_factor_curve,
        q_exact,
    )
    inflection_diagnostics = build_inflection_diagnostics(
        q_values,
        form_factor_curve,
        q_exact,
    )

    mean_y2_charge = float(
        np.trapezoid(np.square(profile) * weight, radius) / np.trapezoid(weight, radius)
    )
    kappa_nl_squared_charge = float(epsilon_beta + 6.0 * mean_y2_charge)
    q_nonlinear_dispersion = float(kappa_nl_squared_charge ** 0.25)
    q_nonlinear_dispersion_relative_error = float(
        (q_nonlinear_dispersion - q_exact) / q_exact
    )

    log_fraction = Fraction(abs(derivative_diagnostics["log_derivative_at_q_exact"])).limit_denominator(12)
    self_fraction = Fraction(abs(derivative_diagnostics["self_reference_ratio_at_q_exact"])).limit_denominator(12)

    pattern_epsilon_front_runner_now = bool(
        len(alpha_bohr_roots) == 1
        and abs(q_bohr_relative_error) < abs(q_star_relative_error)
    )
    pattern_zeta_negative_screen_now = bool(
        abs(mean_q_diagnostics["best_mean_q_relative_error_vs_q_exact"])
        > abs(q_star_relative_error)
    )
    pattern_gamma_negative_screen_now = bool(
        not inflection_diagnostics["stationary_selector_available_now"]
        and inflection_diagnostics["nearest_inflection_distance_over_m0"]
        > abs(q_star - q_exact)
    )
    pattern_beta_negative_screen_now = bool(
        q_nonlinear_dispersion > q_star > q_exact
    )

    return {
        "q_exact_over_m0": q_exact,
        "q_star_over_m0": q_star,
        "q_star_relative_error_vs_q_exact": q_star_relative_error,
        "beta1": beta1,
        "epsilon_beta": epsilon_beta,
        "alpha_exact_from_q_exact": alpha_exact,
        "alpha_bohr_one_eighth": alpha_bohr_one_eighth,
        "alpha_bohr_relative_error_vs_exact": alpha_bohr_relative_error,
        "alpha_bohr_root_count": int(len(alpha_bohr_roots)),
        "alpha_bohr_root_list_over_m0": [float(value) for value in alpha_bohr_roots],
        "q_bohr_over_m0": q_bohr,
        "q_bohr_relative_error_vs_q_exact": q_bohr_relative_error,
        "pattern_epsilon_front_runner_now": pattern_epsilon_front_runner_now,
        "mean_q_rows": mean_q_diagnostics["rows"],
        "best_mean_q_over_m0": mean_q_diagnostics["best_mean_q_over_m0"],
        "best_mean_q_relative_error_vs_q_exact": mean_q_diagnostics[
            "best_mean_q_relative_error_vs_q_exact"
        ],
        "mean_q_window_spread_over_m0": mean_q_diagnostics["mean_q_window_spread_over_m0"],
        "pattern_zeta_negative_screen_now": pattern_zeta_negative_screen_now,
        "F_at_q_exact": derivative_diagnostics["F_at_q_exact"],
        "F_prime_at_q_exact": derivative_diagnostics["F_prime_at_q_exact"],
        "F_double_prime_at_q_exact": derivative_diagnostics["F_double_prime_at_q_exact"],
        "log_derivative_at_q_exact": derivative_diagnostics["log_derivative_at_q_exact"],
        "log_derivative_best_small_fraction": f"{log_fraction.numerator}/{log_fraction.denominator}",
        "log_derivative_best_small_fraction_value": float(log_fraction),
        "log_derivative_small_fraction_relative_gap": float(
            abs(abs(derivative_diagnostics["log_derivative_at_q_exact"]) - float(log_fraction))
            / abs(derivative_diagnostics["log_derivative_at_q_exact"])
        ),
        "self_reference_ratio_at_q_exact": derivative_diagnostics["self_reference_ratio_at_q_exact"],
        "self_reference_best_small_fraction": f"{self_fraction.numerator}/{self_fraction.denominator}",
        "self_reference_best_small_fraction_value": float(self_fraction),
        "self_reference_small_fraction_relative_gap": float(
            abs(abs(derivative_diagnostics["self_reference_ratio_at_q_exact"]) - float(self_fraction))
            / abs(derivative_diagnostics["self_reference_ratio_at_q_exact"])
        ),
        "nearest_inflection_over_m0": inflection_diagnostics["nearest_inflection_over_m0"],
        "nearest_inflection_distance_over_m0": inflection_diagnostics[
            "nearest_inflection_distance_over_m0"
        ],
        "inflection_candidates_over_m0": inflection_diagnostics[
            "inflection_candidates_over_m0"
        ],
        "stationary_selector_available_now": inflection_diagnostics[
            "stationary_selector_available_now"
        ],
        "pattern_gamma_negative_screen_now": pattern_gamma_negative_screen_now,
        "mean_y2_charge": mean_y2_charge,
        "kappa_nl_squared_charge": kappa_nl_squared_charge,
        "q_nonlinear_dispersion_over_m0": q_nonlinear_dispersion,
        "q_nonlinear_dispersion_relative_error_vs_q_exact": q_nonlinear_dispersion_relative_error,
        "pattern_beta_negative_screen_now": pattern_beta_negative_screen_now,
        "fresh_round1_inventory_nonempty_now": True,
        "fresh_round1_primary_candidate": (
            "pattern_epsilon_bohr_radius_matching"
            if pattern_epsilon_front_runner_now
            else "none"
        ),
    }


# 関数: CLI 直実行時に compact summary を表示する。

def main() -> None:
    """Run the helper directly and print a compact JSON-ready summary."""
    pack = build_trial2_fresh_pattern_round1_pack()
    summary = {
        "q_exact_over_m0": pack["q_exact_over_m0"],
        "q_star_over_m0": pack["q_star_over_m0"],
        "q_bohr_over_m0": pack["q_bohr_over_m0"],
        "q_bohr_relative_error_vs_q_exact": pack["q_bohr_relative_error_vs_q_exact"],
        "best_mean_q_over_m0": pack["best_mean_q_over_m0"],
        "q_nonlinear_dispersion_over_m0": pack["q_nonlinear_dispersion_over_m0"],
        "pattern_epsilon_front_runner_now": pack["pattern_epsilon_front_runner_now"],
        "pattern_zeta_negative_screen_now": pack["pattern_zeta_negative_screen_now"],
        "pattern_gamma_negative_screen_now": pack["pattern_gamma_negative_screen_now"],
        "pattern_beta_negative_screen_now": pack["pattern_beta_negative_screen_now"],
    }
    import json

    print(json.dumps(summary, ensure_ascii=False, indent=2))


# 関数: CLI entrypoint から Round-1 screen を実行する。

if __name__ == "__main__":
    main()
