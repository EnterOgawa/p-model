#!/usr/bin/env python3
"""Audit the direct-alpha self-consistent equation q = alpha(q) on the retained pack.

Purpose:
    Evaluate the expert-supplied direct-alpha route

        q = alpha(q) = F(q)^2 / (4 pi)

    without using alpha_target as an input, and determine whether the resulting
    self-consistent finite-q scale coincides with the already fixed retained
    crossing q_exact.

Inputs:
    - scripts/quantum/scalar_proxy_alpha_q_curve_backend.py
    - output/public/quantum/mass_origin_qball_charge_mapping_branch_refresh_metrics.json
    - output/public/quantum/mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_numeric_evaluation_metrics.json
    - output/public/quantum/mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_coupled_tail_reconciliation_review_numeric_evaluation_metrics.json

Outputs:
    - One in-memory audit pack consumed by `.5567-.5574` wrappers

Assumptions:
    - The retained scalar profile remains the only input profile
    - No new parameter is introduced
    - The self-consistent condition is evaluated on the same normalized
      dimensionless q/m0 variable used by the retained alpha(q) diagnostics
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.scalar_proxy_alpha_q_curve_backend import (
    QBALL_BRANCH_REFRESH,
)
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import (
    build_scalar_proxy_alpha_q_curve_pack,
)
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import (
    extract_scalar_ground_state,
)
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import form_factor
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import load_qball_module
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import read_json

SELF_CONSISTENT_TOL = 1.0e-12


# 関数: q-grid 上の符号反転から self-consistent roots を集める。
def find_self_consistent_roots(
    q_values: np.ndarray,
    alpha_curve: np.ndarray,
) -> list[float]:
    """Return the retained-interval roots of q - alpha(q)."""
    roots: list[float] = []
    difference = q_values - alpha_curve
    for index in range(len(q_values) - 1):
        left_q = float(q_values[index])
        right_q = float(q_values[index + 1])
        left_difference = float(difference[index])
        right_difference = float(difference[index + 1])

        if abs(left_difference) <= 1.0e-14:
            roots.append(left_q)

        if left_difference * right_difference < 0.0:
            root = brentq(
                lambda q_value: float(q_value - np.interp(q_value, q_values, alpha_curve)),
                left_q,
                right_q,
            )
            roots.append(float(root))

    if abs(float(difference[-1])) <= 1.0e-14:
        roots.append(float(q_values[-1]))

    unique_roots: list[float] = []
    for candidate in sorted(roots):
        if not unique_roots or abs(candidate - unique_roots[-1]) > 1.0e-12:
            unique_roots.append(candidate)

    return unique_roots


# 関数: retained scalar profile 上で必要な moments を計算する。

def build_profile_moments(radius: np.ndarray, weight: np.ndarray, norm: float) -> dict:
    """Return one compact set of retained radial moments."""
    mean_radius = float(np.trapezoid(weight * radius, radius) / norm)
    mean_r2 = float(np.trapezoid(weight * np.square(radius), radius) / norm)
    inverse_mean_radius = float(1.0 / mean_radius)
    inverse_rms_radius = float(1.0 / math.sqrt(mean_r2))
    return {
        "mean_radius_over_m0_inv": mean_radius,
        "mean_r2_over_m0_inv_sq": mean_r2,
        "inverse_mean_radius_over_m0": inverse_mean_radius,
        "inverse_rms_radius_over_m0": inverse_rms_radius,
    }


# 関数: retained scalar profile と alpha(q) から self-consistent route pack を構築する。

def build_trial2_direct_alpha_self_consistent_pack() -> dict:
    """Return one retained direct-alpha self-consistent audit pack."""
    alpha_pack = build_scalar_proxy_alpha_q_curve_pack()
    q_values = np.asarray(alpha_pack["q_values"], dtype=float)
    alpha_curve = np.asarray(alpha_pack["alpha_curve"], dtype=float)
    q_exact = float(alpha_pack["primary_q_exact_over_m0"])
    q_blind = float(alpha_pack["q_blind_over_m0"])
    q_star = float(alpha_pack["q_star_over_m0"])

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

    self_consistent_roots = find_self_consistent_roots(q_values, alpha_curve)
    q_self_consistent = float(self_consistent_roots[0]) if self_consistent_roots else math.nan
    alpha_at_q_self_consistent = (
        float(np.interp(q_self_consistent, q_values, alpha_curve))
        if self_consistent_roots
        else math.nan
    )
    form_factor_at_q_self_consistent = (
        float(form_factor(radius, weight, norm, q_self_consistent))
        if self_consistent_roots
        else math.nan
    )

    alpha_at_q_exact = float(np.interp(q_exact, q_values, alpha_curve))
    form_factor_at_q_exact = float(np.interp(q_exact, q_values, alpha_pack["form_factor_curve"]))
    self_consistent_gap_at_q_exact = float(q_exact - alpha_at_q_exact)
    q_self_consistent_rel_error_vs_q_exact = (
        float((q_self_consistent - q_exact) / q_exact)
        if self_consistent_roots
        else math.nan
    )
    q_self_consistent_rel_error_vs_q_star = (
        float((q_self_consistent - q_star) / q_star)
        if self_consistent_roots
        else math.nan
    )
    q_self_consistent_matches_q_exact_now = bool(
        self_consistent_roots and abs(q_self_consistent - q_exact) <= SELF_CONSISTENT_TOL
    )
    q_self_consistent_matches_q_blind_now = bool(
        self_consistent_roots and abs(q_self_consistent - q_blind) <= SELF_CONSISTENT_TOL
    )

    profile_moments = build_profile_moments(radius, weight, norm)
    root_proximity_candidates = {
        "q_exact_over_m0": q_exact,
        "q_blind_over_m0": q_blind,
        "q_star_over_m0": q_star,
        "inverse_mean_radius_over_m0": profile_moments["inverse_mean_radius_over_m0"],
        "inverse_rms_radius_over_m0": profile_moments["inverse_rms_radius_over_m0"],
    }
    nearest_name, nearest_value = min(
        root_proximity_candidates.items(),
        key=lambda item: abs(float(item[1]) - q_self_consistent) if self_consistent_roots else math.inf,
    )

    return {
        "beta1": float(scalar_ground_state["beta_n"]),
        "alpha_target": float(alpha_pack["alpha_target"]),
        "q_exact_over_m0": q_exact,
        "q_blind_over_m0": q_blind,
        "q_star_over_m0": q_star,
        "self_consistent_root_list_over_m0": [float(value) for value in self_consistent_roots],
        "self_consistent_root_exists_now": bool(self_consistent_roots),
        "self_consistent_root_unique_now": len(self_consistent_roots) == 1,
        "primary_q_self_consistent_over_m0": q_self_consistent,
        "alpha_at_q_self_consistent": alpha_at_q_self_consistent,
        "form_factor_at_q_self_consistent": form_factor_at_q_self_consistent,
        "alpha_at_q_exact": alpha_at_q_exact,
        "form_factor_at_q_exact": form_factor_at_q_exact,
        "q_minus_alpha_at_q_exact": self_consistent_gap_at_q_exact,
        "q_self_consistent_rel_error_vs_q_exact": q_self_consistent_rel_error_vs_q_exact,
        "q_self_consistent_rel_error_vs_q_star": q_self_consistent_rel_error_vs_q_star,
        "q_self_consistent_matches_q_exact_now": q_self_consistent_matches_q_exact_now,
        "q_self_consistent_matches_q_blind_now": q_self_consistent_matches_q_blind_now,
        "nearest_root_proximity_label": nearest_name,
        "nearest_root_proximity_value_over_m0": float(nearest_value),
        "nearest_root_proximity_gap_over_m0": (
            float(abs(q_self_consistent - float(nearest_value)))
            if self_consistent_roots
            else math.nan
        ),
        "profile_moments": profile_moments,
        "target_free_self_consistent_alpha_route_available_now": bool(
            self_consistent_roots and q_self_consistent_matches_q_exact_now
        ),
        "target_free_self_consistent_alpha_route_negative_now": bool(
            self_consistent_roots and not q_self_consistent_matches_q_exact_now
        ),
    }


# 関数: helper を単体実行して compact summary を表示する。

def main() -> None:
    """Run the helper directly and print one compact JSON summary."""
    pack = build_trial2_direct_alpha_self_consistent_pack()
    summary = {
        "q_exact_over_m0": pack["q_exact_over_m0"],
        "q_blind_over_m0": pack["q_blind_over_m0"],
        "q_star_over_m0": pack["q_star_over_m0"],
        "self_consistent_root_list_over_m0": pack["self_consistent_root_list_over_m0"],
        "q_minus_alpha_at_q_exact": pack["q_minus_alpha_at_q_exact"],
        "q_self_consistent_rel_error_vs_q_exact": pack["q_self_consistent_rel_error_vs_q_exact"],
        "nearest_root_proximity_label": pack["nearest_root_proximity_label"],
        "target_free_self_consistent_alpha_route_available_now": (
            pack["target_free_self_consistent_alpha_route_available_now"]
        ),
        "target_free_self_consistent_alpha_route_negative_now": (
            pack["target_free_self_consistent_alpha_route_negative_now"]
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
