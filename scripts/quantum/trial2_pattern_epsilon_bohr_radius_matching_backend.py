#!/usr/bin/env python3
"""Audit the retained Bohr-radius / Compton matching front-runner.

Purpose:
    Test whether the surviving fresh Round-1 epsilon pattern is merely the
    numerically best low-complexity fit or whether the current pack already
    contains an honest target-free theorem candidate. The helper compares:

        1. canonical radius / scale choices for R / a0 = alpha * R
        2. nearby integer denominator family alpha_n = sqrt(1-beta^2) / n

    against the retained scalar profile and existing alpha(q) surface.

Inputs:
    - scripts/quantum/trial2_fresh_pattern_round1_backend.py
    - output/public/quantum/mass_origin_qball_charge_mapping_branch_refresh_metrics.json

Outputs:
    - One in-memory diagnostic pack consumed by `.5535-.5542` wrappers

Assumptions:
    - q_exact is already fixed numerically
    - alpha_target is not used as an input parameter
    - the route must remain distinct from exhausted A-D, blind-overlap,
      spectral distinguished-scale, and effective coupling / residue branches
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_fresh_pattern_round1_backend import build_trial2_fresh_pattern_round1_pack
from scripts.quantum.trial2_fresh_pattern_round1_backend import find_alpha_curve_roots
from scripts.quantum.trial2_fresh_pattern_round1_backend import load_retained_profile_data
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import build_scalar_proxy_alpha_q_curve_pack


# 関数: half-charge radius を cumulative trapezoid から返す。
def compute_half_charge_radius(radius: np.ndarray, weight: np.ndarray, norm: float) -> float:
    """Return the radius where the cumulative charge weight reaches one half."""
    cumulative = np.zeros_like(radius)
    for index in range(1, len(radius)):
        shell = 0.5 * (weight[index - 1] + weight[index]) * (radius[index] - radius[index - 1])
        cumulative[index] = cumulative[index - 1] + shell

    target = 0.5 * float(norm)
    index = int(np.searchsorted(cumulative, target))
    return float(radius[min(index, len(radius) - 1)])


# 関数: profile の e-fold radius を返す。

def compute_e_fold_radius(radius: np.ndarray, profile: np.ndarray) -> float:
    """Return the first radius where the profile falls below e^-1 of its center."""
    threshold = float(profile[0] / math.e)
    matches = np.where(profile <= threshold)[0]
    if len(matches) == 0:
        return float(radius[-1])

    return float(radius[int(matches[0])])


# 関数: simple rational approximation を返す。

def build_small_fraction(value: float, max_denominator: int = 16) -> dict[str, float | str]:
    """Return the nearest low-complexity rational approximation to one value."""
    fraction = Fraction(float(value)).limit_denominator(max_denominator)
    fraction_value = float(fraction)
    relative_gap = float(abs(fraction_value - value) / abs(value))
    return {
        "fraction_label": f"{fraction.numerator}/{fraction.denominator}",
        "fraction_value": fraction_value,
        "relative_gap": relative_gap,
    }


# 関数: canonical radius candidates を返す。

def build_canonical_radius_rows(
    radius: np.ndarray,
    profile: np.ndarray,
    weight: np.ndarray,
    norm: float,
    alpha_exact: float,
    epsilon_beta: float,
    q_exact: float,
    q_star: float,
) -> list[dict[str, float | str]]:
    """Return canonical radii/scales and their low-complexity ratio fits."""
    mean_radius = float(np.trapezoid(radius * weight, radius) / norm)
    rms_radius = float(math.sqrt(np.trapezoid(np.square(radius) * weight, radius) / norm))
    half_charge_radius = compute_half_charge_radius(radius, weight, norm)
    e_fold_radius = compute_e_fold_radius(radius, profile)
    candidates = [
        ("tail_radius_inv_kappa", 1.0 / math.sqrt(epsilon_beta)),
        ("mean_radius", mean_radius),
        ("rms_radius", rms_radius),
        ("half_charge_radius", half_charge_radius),
        ("e_fold_radius", e_fold_radius),
        ("inverse_q_star", 1.0 / q_star),
        ("inverse_q_exact", 1.0 / q_exact),
    ]
    rows: list[dict[str, float | str]] = []
    for radius_label, radius_value in candidates:
        ratio = float(alpha_exact * radius_value)
        fraction = build_small_fraction(ratio)
        rows.append(
            {
                "radius_label": radius_label,
                "radius_value_over_m0_inv": float(radius_value),
                "ratio_value": ratio,
                "best_small_fraction": str(fraction["fraction_label"]),
                "best_small_fraction_value": float(fraction["fraction_value"]),
                "best_small_fraction_relative_gap": float(fraction["relative_gap"]),
            }
        )

    return rows


# 関数: integer denominator family n=6..12 の root diagnostics を返す。

def build_integer_denominator_rows(
    epsilon_beta: float,
    q_values: np.ndarray,
    alpha_curve: np.ndarray,
    q_exact: float,
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
) -> list[dict[str, float]]:
    """Return retained q-roots for alpha_n = sqrt(epsilon_beta)/n."""
    rows: list[dict[str, float]] = []
    for denominator in range(6, 13):
        alpha_candidate = float(math.sqrt(epsilon_beta) / denominator)
        roots = find_alpha_curve_roots(
            radius,
            weight,
            norm,
            q_values,
            alpha_curve,
            alpha_candidate,
        )
        first_root = float(roots[0]) if roots else math.nan
        relative_error = float((first_root - q_exact) / q_exact) if roots else math.nan
        rows.append(
            {
                "denominator_n": int(denominator),
                "alpha_candidate": alpha_candidate,
                "root_count": int(len(roots)),
                "q_root_over_m0": first_root,
                "relative_error_vs_q_exact": relative_error,
            }
        )

    return rows


# 関数: Bohr/Compton matching audit pack を構築する。

def build_trial2_pattern_epsilon_bohr_matching_pack() -> dict:
    """Return the Bohr / Compton matching diagnostic pack."""
    round1_pack = build_trial2_fresh_pattern_round1_pack()
    alpha_pack = build_scalar_proxy_alpha_q_curve_pack(q_min=0.0, q_max=5.0, q_count=50001)
    profile_data = load_retained_profile_data()

    q_exact = float(round1_pack["q_exact_over_m0"])
    q_star = float(round1_pack["q_star_over_m0"])
    epsilon_beta = float(round1_pack["epsilon_beta"])
    alpha_exact = float(round1_pack["alpha_exact_from_q_exact"])

    q_values = np.asarray(alpha_pack["q_values"], dtype=float)
    alpha_curve = np.asarray(alpha_pack["alpha_curve"], dtype=float)
    radius = np.asarray(profile_data["radius"], dtype=float)
    profile = np.asarray(profile_data["profile"], dtype=float)
    weight = np.asarray(profile_data["weight"], dtype=float)
    norm = float(profile_data["norm"])

    canonical_radius_rows = build_canonical_radius_rows(
        radius,
        profile,
        weight,
        norm,
        alpha_exact,
        epsilon_beta,
        q_exact,
        q_star,
    )
    best_radius_row = min(
        canonical_radius_rows,
        key=lambda row: float(row["best_small_fraction_relative_gap"]),
    )

    integer_denominator_rows = build_integer_denominator_rows(
        epsilon_beta,
        q_values,
        alpha_curve,
        q_exact,
        radius,
        weight,
        norm,
    )
    best_integer_row = min(
        integer_denominator_rows,
        key=lambda row: abs(float(row["relative_error_vs_q_exact"])),
    )

    n_fit = float(math.sqrt(epsilon_beta) / alpha_exact)
    one_eighth_row = next(
        row for row in integer_denominator_rows if int(row["denominator_n"]) == 8
    )

    tail_radius_one_eighth_available_now = bool(
        best_radius_row["radius_label"] == "tail_radius_inv_kappa"
        and best_radius_row["best_small_fraction"] == "1/8"
    )
    one_eighth_integer_front_runner_now = bool(int(best_integer_row["denominator_n"]) == 8)
    target_free_theorem_available_now = False
    heuristic_front_runner_only_now = bool(
        tail_radius_one_eighth_available_now
        and one_eighth_integer_front_runner_now
        and not target_free_theorem_available_now
    )
    negative_closeout_available_now = bool(
        heuristic_front_runner_only_now
        and abs(float(one_eighth_row["relative_error_vs_q_exact"])) < abs(float(round1_pack["q_star_relative_error_vs_q_exact"]))
    )

    return {
        "q_exact_over_m0": q_exact,
        "q_star_over_m0": q_star,
        "epsilon_beta": epsilon_beta,
        "alpha_exact_from_q_exact": alpha_exact,
        "n_fit_from_exact_ratio": n_fit,
        "n_fit_relative_gap_vs_8": float((n_fit - 8.0) / 8.0),
        "canonical_radius_rows": canonical_radius_rows,
        "best_radius_label": str(best_radius_row["radius_label"]),
        "best_radius_ratio_value": float(best_radius_row["ratio_value"]),
        "best_radius_fraction_label": str(best_radius_row["best_small_fraction"]),
        "best_radius_fraction_relative_gap": float(
            best_radius_row["best_small_fraction_relative_gap"]
        ),
        "integer_denominator_rows": integer_denominator_rows,
        "best_integer_denominator": int(best_integer_row["denominator_n"]),
        "best_integer_relative_error_vs_q_exact": float(
            best_integer_row["relative_error_vs_q_exact"]
        ),
        "q_one_eighth_over_m0": float(one_eighth_row["q_root_over_m0"]),
        "q_one_eighth_relative_error_vs_q_exact": float(
            one_eighth_row["relative_error_vs_q_exact"]
        ),
        "tail_radius_one_eighth_available_now": tail_radius_one_eighth_available_now,
        "one_eighth_integer_front_runner_now": one_eighth_integer_front_runner_now,
        "target_free_theorem_available_now": target_free_theorem_available_now,
        "heuristic_front_runner_only_now": heuristic_front_runner_only_now,
        "negative_closeout_available_now": negative_closeout_available_now,
    }


# 関数: CLI 実行時に compact summary を返す。

def main() -> None:
    """Run the Bohr / Compton matching helper directly."""
    pack = build_trial2_pattern_epsilon_bohr_matching_pack()
    import json

    summary = {
        "best_radius_label": pack["best_radius_label"],
        "best_radius_fraction_label": pack["best_radius_fraction_label"],
        "n_fit_from_exact_ratio": pack["n_fit_from_exact_ratio"],
        "best_integer_denominator": pack["best_integer_denominator"],
        "q_one_eighth_over_m0": pack["q_one_eighth_over_m0"],
        "q_one_eighth_relative_error_vs_q_exact": pack["q_one_eighth_relative_error_vs_q_exact"],
        "heuristic_front_runner_only_now": pack["heuristic_front_runner_only_now"],
        "negative_closeout_available_now": pack["negative_closeout_available_now"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# 関数: CLI entrypoint から helper を実行する。

if __name__ == "__main__":
    main()
