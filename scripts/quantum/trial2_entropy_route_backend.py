#!/usr/bin/env python3
"""Audit the direct-alpha entropy route on the retained scalar Q-ball branch.

Purpose:
    After the interaction-over-harmonic exact-relation route closes negatively,
    the next honest low-cost direct-alpha branch is the entropy route. This
    helper tests whether the retained charge-density shape can encode alpha
    directly through a simple Shannon-entropy formula.

Inputs:
    - scripts/quantum/mass_origin_qball_charge_mapping_branch.py
    - scripts/quantum/scalar_proxy_alpha_q_curve_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5599-.5606` wrappers

Assumptions:
    - No new parameter is introduced
    - alpha_target is only an audit comparator
    - Only the entropy route is audited here
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
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import form_factor


FOUR_PI = 4.0 * math.pi


# 関数: retained scalar profile を materialize する pivot solver module を cached で返す。
@lru_cache(maxsize=1)
def get_qball_pivot_module():
    """Return the retained pivot solver module used for scalar profiles."""
    return load_qball_pivot_module()


# 関数: 1 つの beta row について entropy readout を構築する。

def build_entropy_row(beta: float, q_exact: float) -> dict:
    """Return one entropy-route row for the supplied retained beta."""
    beta = float(beta)
    q_exact = float(q_exact)
    qball_pivot = get_qball_pivot_module()
    amplitude = qball_pivot.find_amp(beta)
    if amplitude is None:
        raise SystemExit(f"[fail] localized scalar profile is unavailable for beta={beta}")

    radius, profile, _ = solve_full_profile(beta, float(amplitude))
    radius = np.asarray(radius, dtype=float)
    profile = np.asarray(profile, dtype=float)
    weight = np.square(profile) * np.square(radius)
    norm = float(np.trapezoid(weight, radius))
    probability_density = np.clip(weight / max(norm, 1.0e-300), 1.0e-300, None)
    entropy = float(-np.trapezoid(probability_density * np.log(probability_density), radius))

    alpha_from_entropy = float(math.exp(-entropy) / FOUR_PI)
    form_factor_from_entropy = float(math.exp(-0.5 * entropy))
    form_factor_exact = float(form_factor(radius, weight, norm, q_exact))

    return {
        "beta": beta,
        "central_amplitude": float(amplitude),
        "shannon_entropy": entropy,
        "alpha_from_entropy": alpha_from_entropy,
        "alpha_from_entropy_rel_error_vs_target": float(
            (alpha_from_entropy - ALPHA_TARGET) / ALPHA_TARGET
        ),
        "form_factor_from_entropy": form_factor_from_entropy,
        "form_factor_exact": form_factor_exact,
        "form_factor_from_entropy_rel_error_vs_exact": float(
            (form_factor_from_entropy - form_factor_exact) / max(form_factor_exact, 1.0e-30)
        ),
    }


# 関数: entropy route 全体を official pack に束ねる。

def build_trial2_entropy_pack(
    retained_beta: float,
    nearest_beta: float,
    q_exact: float,
) -> dict:
    """Return one entropy-route audit pack."""
    retained_row = build_entropy_row(float(retained_beta), float(q_exact))
    near_row = build_entropy_row(float(nearest_beta), float(q_exact))

    entropy_alpha_exact_route_available_now = bool(
        abs(retained_row["alpha_from_entropy_rel_error_vs_target"]) <= 1.0e-3
        and abs(near_row["alpha_from_entropy_rel_error_vs_target"]) <= 1.0e-3
    )
    entropy_form_factor_exact_route_available_now = bool(
        abs(retained_row["form_factor_from_entropy_rel_error_vs_exact"]) <= 1.0e-3
        and abs(near_row["form_factor_from_entropy_rel_error_vs_exact"]) <= 1.0e-3
    )
    entropy_route_negative_closeout_available_now = bool(
        not entropy_alpha_exact_route_available_now
        and not entropy_form_factor_exact_route_available_now
    )

    return {
        "alpha_target": float(ALPHA_TARGET),
        "q_exact_over_m0": float(q_exact),
        "retained_beta1": float(retained_beta),
        "nearest_alpha_beta_root_to_retained": float(nearest_beta),
        "retained_row": retained_row,
        "nearest_row": near_row,
        "entropy_alpha_exact_route_available_now": entropy_alpha_exact_route_available_now,
        "entropy_form_factor_exact_route_available_now": entropy_form_factor_exact_route_available_now,
        "entropy_route_negative_closeout_available_now": (
            entropy_route_negative_closeout_available_now
        ),
    }


# 関数: backend 単体実行時に entropy summary を表示する。

def main() -> None:
    """Run the entropy route audit directly."""
    pack = build_trial2_entropy_pack(
        retained_beta=0.9982557379261291,
        nearest_beta=0.9982996989044647,
        q_exact=0.2416825755115744,
    )
    retained = pack["retained_row"]
    print("[trial2_entropy_route_backend] retained row:")
    print(f"  S = {retained['shannon_entropy']:.15f}")
    print(f"  alpha_entropy = {retained['alpha_from_entropy']:.15f}")
    print(
        f"  alpha rel error = "
        f"{retained['alpha_from_entropy_rel_error_vs_target']:.15f}"
    )
    print(
        f"  F_entropy rel error = "
        f"{retained['form_factor_from_entropy_rel_error_vs_exact']:.15f}"
    )


if __name__ == "__main__":
    main()
