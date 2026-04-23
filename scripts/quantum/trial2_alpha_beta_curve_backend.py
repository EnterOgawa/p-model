#!/usr/bin/env python3
"""Audit the direct-alpha family alpha(beta) on the retained scalar Q-ball branch.

Purpose:
    Evaluate the genuinely new followup route promoted after the direct-alpha
    self-consistent fixed point closed negatively. The route reads alpha
    directly from beta through the beta-native scale

        q_star(beta) = (1 - beta^2)^(1/4)
        alpha_beta(beta) = F_beta(q_star(beta))^2 / (4 pi)

    and checks whether the retained scalar family yields one globally unique
    beta readout for the physical alpha target, or whether the family only
    compresses the blocker to a local microshift near the retained mode-1 beta.

Inputs:
    - scripts/quantum/mass_origin_qball_charge_mapping_branch.py
    - scripts/quantum/scalar_proxy_alpha_q_curve_backend.py
    - output/public/quantum/mass_origin_qball_charge_mapping_branch_refresh_metrics.json

Outputs:
    - One in-memory audit pack consumed by `.5575-.5582` wrappers

Assumptions:
    - No new parameter is introduced
    - The route is audited on the same scalar Q-ball family already retained
    - The family is allowed to use alpha_target only as an audit comparator,
      not as an input to the definition of alpha_beta(beta)
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
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import QBALL_BRANCH_REFRESH
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import extract_scalar_ground_state
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import form_factor
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import read_json


FOUR_PI = 4.0 * math.pi
BETA_GRID_MIN = 0.94
BETA_GRID_MAX = 0.999
BETA_GRID_COUNT = 41
LOCAL_BRANCH_HALF_WIDTH = 0.01


# 関数: beta-native scalar readout に必要な pivot solver を cached で返す。
@lru_cache(maxsize=1)
def get_qball_pivot_module():
    """Return the retained pivot solver module used to find the central amplitude."""
    return load_qball_pivot_module()


# 関数: 1 つの beta に対応する scalar family row を構築する。

@lru_cache(maxsize=None)
def build_beta_family_row(beta: float) -> dict | None:
    """Return one alpha(beta) family row or None when no localized solution exists."""
    beta = float(beta)
    qball_pivot = get_qball_pivot_module()
    amplitude = qball_pivot.find_amp(beta)
    if amplitude is None:
        return None

    radius, profile, profile_prime = solve_full_profile(beta, float(amplitude))
    radius = np.asarray(radius, dtype=float)
    profile = np.asarray(profile, dtype=float)
    profile_prime = np.asarray(profile_prime, dtype=float)
    density = np.square(profile)
    weight = density * np.square(radius)
    norm = float(np.trapezoid(weight, radius))
    q_star = float((1.0 - beta * beta) ** 0.25)
    form_factor_at_q_star = float(form_factor(radius, weight, norm, q_star))
    alpha_at_q_star = float((form_factor_at_q_star**2) / FOUR_PI)
    charge_proxy = float(beta * np.trapezoid(4.0 * math.pi * weight, radius))
    energy_proxy = float(
        np.trapezoid(
            4.0
            * math.pi
            * np.square(radius)
            * (
                0.5 * np.square(profile_prime)
                + 0.5 * (1.0 + beta * beta) * np.square(profile)
                + np.power(profile, 3)
                + 0.25 * np.power(profile, 4)
            ),
            radius,
        )
    )
    return {
        "beta": beta,
        "central_amplitude": float(amplitude),
        "q_star_over_m0": q_star,
        "F_at_q_star": form_factor_at_q_star,
        "alpha_at_q_star": alpha_at_q_star,
        "charge_proxy": charge_proxy,
        "energy_proxy": energy_proxy,
        "tail_abs": float(abs(profile[-1])),
    }


# 関数: beta-grid 上で定義される retained family rows を構築する。

def build_beta_family_rows(
    beta_min: float = BETA_GRID_MIN,
    beta_max: float = BETA_GRID_MAX,
    beta_count: int = BETA_GRID_COUNT,
) -> list[dict]:
    """Return the retained localized rows for the alpha(beta) audit grid."""
    rows: list[dict] = []
    for beta in np.linspace(float(beta_min), float(beta_max), int(beta_count), dtype=float):
        family_row = build_beta_family_row(float(beta))
        if family_row is not None:
            rows.append(family_row)

    return rows


# 関数: alpha(beta) - alpha_target の global roots を集める。

def find_alpha_beta_roots(rows: list[dict]) -> list[float]:
    """Return the beta values where alpha(beta) crosses the physical alpha target."""
    roots: list[float] = []
    for left, right in zip(rows[:-1], rows[1:]):
        left_beta = float(left["beta"])
        right_beta = float(right["beta"])
        left_diff = float(left["alpha_at_q_star"]) - float(ALPHA_TARGET)
        right_diff = float(right["alpha_at_q_star"]) - float(ALPHA_TARGET)

        if abs(left_diff) <= 1.0e-14:
            roots.append(left_beta)

        if left_diff * right_diff < 0.0:
            root = brentq(
                lambda beta: float(build_beta_family_row(float(beta))["alpha_at_q_star"] - ALPHA_TARGET),
                left_beta,
                right_beta,
            )
            roots.append(float(root))

    last_diff = float(rows[-1]["alpha_at_q_star"]) - float(ALPHA_TARGET)
    if abs(last_diff) <= 1.0e-14:
        roots.append(float(rows[-1]["beta"]))

    unique_roots: list[float] = []
    for candidate in sorted(roots):
        if not unique_roots or abs(candidate - unique_roots[-1]) > 1.0e-12:
            unique_roots.append(candidate)

    return unique_roots


# 関数: compact な family sample table を返す。

def build_family_samples(rows: list[dict], extra_betas: list[float]) -> list[dict]:
    """Return one compact family checkpoint table for documentation and wrappers."""
    sampled_betas = sorted({float(row["beta"]) for row in rows[::5]} | {float(beta) for beta in extra_betas})
    samples: list[dict] = []
    for beta in sampled_betas:
        row = build_beta_family_row(float(beta))
        if row is not None:
            samples.append(row)

    return samples


# 関数: alpha(beta) family を official 監査 pack に束ねる。

def build_trial2_alpha_beta_curve_pack() -> dict:
    """Return one retained alpha(beta) audit pack."""
    qball_branch_refresh = read_json(QBALL_BRANCH_REFRESH)
    scalar_ground_state = extract_scalar_ground_state(qball_branch_refresh)
    retained_beta = float(scalar_ground_state["beta_n"])
    retained_row = build_beta_family_row(retained_beta)
    if retained_row is None:
        raise SystemExit("[fail] retained mode-1 beta does not yield one localized alpha(beta) row")

    family_rows = build_beta_family_rows()
    if len(family_rows) < 2:
        raise SystemExit("[fail] insufficient localized rows for alpha(beta) family audit")

    global_roots = find_alpha_beta_roots(family_rows)
    nearest_root = min(global_roots, key=lambda value: abs(float(value) - retained_beta)) if global_roots else math.nan
    local_branch_roots = [
        float(root)
        for root in global_roots
        if abs(float(root) - retained_beta) <= LOCAL_BRANCH_HALF_WIDTH
    ]
    nearest_root_row = build_beta_family_row(float(nearest_root)) if global_roots else None
    if global_roots and nearest_root_row is None:
        raise SystemExit("[fail] nearest alpha(beta) root row could not be materialized")

    alpha_values = np.asarray([float(row["alpha_at_q_star"]) for row in family_rows], dtype=float)
    beta_values = np.asarray([float(row["beta"]) for row in family_rows], dtype=float)
    min_alpha_index = int(np.argmin(alpha_values))
    max_alpha_index = int(np.argmax(alpha_values))

    retained_alpha = float(retained_row["alpha_at_q_star"])
    retained_residual_rel = float((retained_alpha - ALPHA_TARGET) / ALPHA_TARGET)
    nearest_root_rel_shift = (
        float((float(nearest_root) - retained_beta) / retained_beta) if global_roots else math.nan
    )
    nearest_root_charge_rel_error = (
        float(
            (float(nearest_root_row["charge_proxy"]) - float(retained_row["charge_proxy"]))
            / float(retained_row["charge_proxy"])
        )
        if nearest_root_row is not None
        else math.nan
    )
    nearest_root_energy_rel_error = (
        float(
            (float(nearest_root_row["energy_proxy"]) - float(retained_row["energy_proxy"]))
            / float(retained_row["energy_proxy"])
        )
        if nearest_root_row is not None
        else math.nan
    )

    alpha_beta_family_exact_route_available_now = bool(
        len(global_roots) == 1 and abs(float(nearest_root) - retained_beta) <= 1.0e-12
    )
    alpha_beta_local_microshift_available_now = bool(
        len(local_branch_roots) == 1 and abs(nearest_root_rel_shift) <= 1.0e-3
    )

    family_samples = build_family_samples(
        family_rows,
        [retained_beta, *global_roots],
    )
    return {
        "alpha_target": float(ALPHA_TARGET),
        "retained_beta1": retained_beta,
        "retained_charge_proxy": float(retained_row["charge_proxy"]),
        "retained_energy_proxy": float(retained_row["energy_proxy"]),
        "retained_q_star_over_m0": float(retained_row["q_star_over_m0"]),
        "retained_F_at_q_star": float(retained_row["F_at_q_star"]),
        "retained_alpha_at_q_star": retained_alpha,
        "retained_alpha_rel_error_vs_target": retained_residual_rel,
        "family_beta_min": float(beta_values[0]),
        "family_beta_max": float(beta_values[-1]),
        "family_row_count": int(len(family_rows)),
        "alpha_beta_min": float(alpha_values[min_alpha_index]),
        "alpha_beta_min_beta": float(beta_values[min_alpha_index]),
        "alpha_beta_max": float(alpha_values[max_alpha_index]),
        "alpha_beta_max_beta": float(beta_values[max_alpha_index]),
        "alpha_beta_global_root_list": [float(root) for root in global_roots],
        "alpha_beta_global_root_count": int(len(global_roots)),
        "alpha_beta_global_unique_now": len(global_roots) == 1,
        "alpha_beta_local_branch_root_list": [float(root) for root in local_branch_roots],
        "alpha_beta_local_branch_unique_now": len(local_branch_roots) == 1,
        "nearest_alpha_beta_root_to_retained": float(nearest_root) if global_roots else math.nan,
        "nearest_alpha_beta_root_rel_shift_vs_retained": nearest_root_rel_shift,
        "nearest_alpha_beta_root_charge_proxy": (
            float(nearest_root_row["charge_proxy"]) if nearest_root_row is not None else math.nan
        ),
        "nearest_alpha_beta_root_energy_proxy": (
            float(nearest_root_row["energy_proxy"]) if nearest_root_row is not None else math.nan
        ),
        "nearest_alpha_beta_root_charge_rel_error_vs_retained": nearest_root_charge_rel_error,
        "nearest_alpha_beta_root_energy_rel_error_vs_retained": nearest_root_energy_rel_error,
        "alpha_beta_family_exact_route_available_now": alpha_beta_family_exact_route_available_now,
        "alpha_beta_local_microshift_available_now": alpha_beta_local_microshift_available_now,
        "energy_partition_followup_required_now": bool(
            alpha_beta_local_microshift_available_now and not alpha_beta_family_exact_route_available_now
        ),
        "entropy_route_reserve_retained_now": True,
        "family_samples": family_samples,
    }


# 関数: helper 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the helper directly and print one compact JSON-like summary."""
    import json

    pack = build_trial2_alpha_beta_curve_pack()
    summary = {
        "retained_beta1": pack["retained_beta1"],
        "retained_alpha_at_q_star": pack["retained_alpha_at_q_star"],
        "retained_alpha_rel_error_vs_target": pack["retained_alpha_rel_error_vs_target"],
        "alpha_beta_global_root_list": pack["alpha_beta_global_root_list"],
        "nearest_alpha_beta_root_to_retained": pack["nearest_alpha_beta_root_to_retained"],
        "nearest_alpha_beta_root_rel_shift_vs_retained": (
            pack["nearest_alpha_beta_root_rel_shift_vs_retained"]
        ),
        "alpha_beta_global_unique_now": pack["alpha_beta_global_unique_now"],
        "alpha_beta_local_microshift_available_now": pack["alpha_beta_local_microshift_available_now"],
        "energy_partition_followup_required_now": pack["energy_partition_followup_required_now"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
