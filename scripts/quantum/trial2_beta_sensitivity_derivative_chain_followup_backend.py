#!/usr/bin/env python3
"""Audit the derivative-chain support behind the Trial-2 common-root theorem.

Purpose:
    After `.5679-.5686`, the retained control window already fixes the signs of
    the weighted beta-derivative integrals `dI_n / d beta`. The remaining gap
    is narrower:

        can the two readout derivatives

            d alpha_qstar / d beta
            d R8 / d beta

        be decomposed into controlled channels strongly enough to promote the
        next uniqueness-anchor followup?

    This backend still does not overclaim the final strict theorem. It only
    turns the existing sign support into one explicit derivative-chain support
    layer:

    1. `alpha_qstar(beta)` is split into a fixed-q profile-response channel and
       one explicit `q_star(beta)` channel,
    2. `R8(beta)` is split into exact total-derivative channels coming from
       `I2`, `Ig`, `I4`, `B`, and the explicit beta dependence of the exact
       relation,
    3. the sign / dominance pattern of those channels is checked across the
       retained `h` family.

Inputs:
    - scripts/quantum/trial2_beta_sensitivity_operator_level_spectral_projection_followup_backend.py
    - scripts/quantum/trial2_beta_sensitivity_equation_backend.py
    - scripts/quantum/trial2_interaction_total_over_harmonic_sq_exact_relation_backend.py
    - scripts/quantum/trial2_alpha_beta_curve_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5687-.5694` wrappers

Assumptions:
    - No new parameter is introduced
    - The route remains local to the retained common-root branch
    - The final strict theorem is still deferred; this backend only promotes
      derivative-chain sign support
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_alpha_beta_curve_backend import build_beta_family_row
from scripts.quantum.trial2_beta_sensitivity_equation_backend import BETA_COMMON_ROOT
from scripts.quantum.trial2_beta_sensitivity_equation_backend import H_VALUES
from scripts.quantum.trial2_beta_sensitivity_equation_backend import build_common_grid
from scripts.quantum.trial2_beta_sensitivity_equation_backend import (
    build_integral_derivative_pack,
)
from scripts.quantum.trial2_beta_sensitivity_equation_backend import build_profile_row
from scripts.quantum.trial2_beta_sensitivity_operator_level_spectral_projection_followup_backend import (
    build_trial2_beta_sensitivity_operator_level_spectral_projection_followup_pack,
)
from scripts.quantum.trial2_interaction_total_over_harmonic_sq_exact_relation_backend import (
    build_exact_relation_row,
)


DELTA_DERIVATIVE_REL_SPREAD_TOL = 5.0e-3
R8_DERIVATIVE_REL_SPREAD_TOL = 5.0e-3
ALPHA_DERIVATIVE_REL_SPREAD_TOL = 5.0e-3


# 関数: sinc(qx) を返す。
def evaluate_sinc(q_value: float, radius: np.ndarray) -> np.ndarray:
    """Return `sin(qx)/(qx)` with the regular `q -> 0` limit."""
    qr = float(q_value) * np.asarray(radius, dtype=float)
    return np.where(np.abs(qr) > 1.0e-12, np.sin(qr) / qr, 1.0)


# 関数: `q` 微分した sinc kernel を返す。

def evaluate_sinc_q_derivative(q_value: float, radius: np.ndarray) -> np.ndarray:
    """Return `d/dq [sin(qx)/(qx)]` on the retained positive-radius grid."""
    radius = np.asarray(radius, dtype=float)
    q_value = float(q_value)
    qr = q_value * radius
    return np.where(
        np.abs(qr) > 1.0e-12,
        (qr * np.cos(qr) - np.sin(qr)) / (q_value * q_value * radius),
        0.0,
    )


# 関数: `q_star(beta)` の beta 微分を返す。

def evaluate_q_star_beta_derivative(beta_value: float) -> float:
    """Return the exact beta derivative of `q_star(beta) = (1-beta^2)^(1/4)`."""
    beta_value = float(beta_value)
    return float(
        -beta_value / (2.0 * np.power(1.0 - beta_value * beta_value, 0.75))
    )


# 関数: `alpha_qstar(beta)` の derivative-chain row を返す。

def build_alpha_qstar_chain_row(h_value: float) -> dict:
    """Return one derivative-chain row for the `alpha_qstar(beta)` readout."""
    beta_common_root = float(BETA_COMMON_ROOT)
    h_value = float(h_value)
    center_row = build_profile_row(beta_common_root)
    plus_row = build_profile_row(beta_common_root + h_value)
    minus_row = build_profile_row(beta_common_root - h_value)
    grid = build_common_grid(center_row, plus_row, minus_row)

    profile = np.interp(grid, center_row["radius"], center_row["profile"])
    profile_plus = np.interp(grid, plus_row["radius"], plus_row["profile"])
    profile_minus = np.interp(grid, minus_row["radius"], minus_row["profile"])
    u_beta = (profile_plus - profile_minus) / (2.0 * h_value)

    q_star_row = build_beta_family_row(beta_common_root)
    if q_star_row is None:
        raise SystemExit("[fail] alpha_qstar row is unavailable at beta_common_root")

    q_star = float(q_star_row["q_star_over_m0"])
    sinc = evaluate_sinc(q_star, grid)
    sinc_q_derivative = evaluate_sinc_q_derivative(q_star, grid)

    i2 = float(np.trapezoid(np.square(profile) * np.square(grid), grid))
    numerator = float(
        np.trapezoid(np.square(profile) * sinc * np.square(grid), grid)
    )
    form_factor_value = float(numerator / i2)
    d_i2_dbeta = float(np.trapezoid(2.0 * profile * u_beta * np.square(grid), grid))
    d_numerator_profile_dbeta = float(
        np.trapezoid(2.0 * profile * u_beta * sinc * np.square(grid), grid)
    )
    d_form_factor_profile_dbeta = float(
        d_numerator_profile_dbeta / i2
        - numerator * d_i2_dbeta / (i2 * i2)
    )
    d_form_factor_dq = float(
        np.trapezoid(np.square(profile) * sinc_q_derivative * np.square(grid), grid)
        / i2
    )
    dq_star_dbeta = evaluate_q_star_beta_derivative(beta_common_root)
    d_form_factor_qstar_channel_dbeta = float(d_form_factor_dq * dq_star_dbeta)
    d_form_factor_total_dbeta = float(
        d_form_factor_profile_dbeta + d_form_factor_qstar_channel_dbeta
    )

    alpha_prefactor = float(form_factor_value / (2.0 * math.pi))
    alpha_profile_channel = float(alpha_prefactor * d_form_factor_profile_dbeta)
    alpha_qstar_channel = float(alpha_prefactor * d_form_factor_qstar_channel_dbeta)
    alpha_total_derivative = float(alpha_prefactor * d_form_factor_total_dbeta)
    finite_difference_reference = float(
        (build_beta_family_row(beta_common_root + h_value)["alpha_at_q_star"] - build_beta_family_row(beta_common_root - h_value)["alpha_at_q_star"])
        / (2.0 * h_value)
    )

    return {
        "h": h_value,
        "q_star_over_m0": q_star,
        "dq_star_dbeta": dq_star_dbeta,
        "form_factor_at_q_star": form_factor_value,
        "d_i2_dbeta": d_i2_dbeta,
        "d_numerator_profile_dbeta": d_numerator_profile_dbeta,
        "d_form_factor_profile_dbeta": d_form_factor_profile_dbeta,
        "d_form_factor_dq_at_q_star": d_form_factor_dq,
        "d_form_factor_qstar_channel_dbeta": d_form_factor_qstar_channel_dbeta,
        "d_form_factor_total_dbeta": d_form_factor_total_dbeta,
        "alpha_profile_channel_dbeta": alpha_profile_channel,
        "alpha_qstar_channel_dbeta": alpha_qstar_channel,
        "alpha_total_derivative_dbeta": alpha_total_derivative,
        "alpha_total_derivative_fd_reference": finite_difference_reference,
        "profile_channel_negative_now": bool(alpha_profile_channel < 0.0),
        "qstar_channel_positive_now": bool(alpha_qstar_channel > 0.0),
        "qstar_channel_dominates_profile_now": bool(
            alpha_qstar_channel > abs(alpha_profile_channel)
        ),
        "alpha_total_derivative_positive_now": bool(alpha_total_derivative > 0.0),
    }


# 関数: `R8(beta)` の exact partial derivatives を返す。

def evaluate_r8_exact_partials(beta_value: float, i2: float, ig: float, i4: float, boundary_value: float) -> dict:
    """Return the closed-form partial derivatives of the exact `R8(beta)` law."""
    beta_value = float(beta_value)
    i2 = float(i2)
    ig = float(ig)
    i4 = float(i4)
    boundary_value = float(boundary_value)
    beta_sq = beta_value * beta_value
    common_den = float(18.0 * i2 * i2 * (1.0 + beta_sq) * (1.0 + beta_sq))
    common_num = float(4.0 * boundary_value + i2 * beta_sq - 7.0 * i2 + i4 - 7.0 * ig)

    partial_i2 = float(
        (
            -16.0 * boundary_value * boundary_value
            - 4.0 * boundary_value * i2 * beta_sq
            + 28.0 * boundary_value * i2
            - 8.0 * boundary_value * i4
            + 56.0 * boundary_value * ig
            - i2 * i4 * beta_sq
            + 7.0 * i2 * i4
            + 16.0 * i2 * ig * beta_sq
            - 40.0 * i2 * ig
            - i4 * i4
            + 14.0 * i4 * ig
            - 40.0 * ig * ig
        )
        / (18.0 * i2**3 * (1.0 + beta_sq) * (1.0 + beta_sq))
    )
    partial_ig = float(
        (
            -28.0 * boundary_value
            - 16.0 * i2 * beta_sq
            + 40.0 * i2
            - 7.0 * i4
            + 40.0 * ig
        )
        / common_den
    )
    partial_i4 = float(common_num / common_den)
    partial_boundary = float((4.0 * common_num) / common_den)
    partial_beta_explicit = float(
        beta_value
        * (
            i2
            * (1.0 + beta_sq)
            * (
                4.0 * boundary_value
                - 4.0 * i2 * (beta_sq - 1.0)
                - 4.0 * i2 * (beta_sq + 5.0)
                + i4
                - 16.0 * ig
            )
            - (
                4.0 * boundary_value
                + 4.0 * i2 * (beta_sq - 1.0)
                + i4
                - 4.0 * ig
            )
            * (
                4.0 * boundary_value
                - 2.0 * i2 * (beta_sq + 5.0)
                + i4
                - 10.0 * ig
            )
        )
        / (9.0 * i2 * i2 * np.power(1.0 + beta_sq, 3))
    )

    return {
        "partial_r8_wrt_i2": partial_i2,
        "partial_r8_wrt_ig": partial_ig,
        "partial_r8_wrt_i4": partial_i4,
        "partial_r8_wrt_boundary": partial_boundary,
        "partial_r8_explicit_beta_channel": partial_beta_explicit,
    }


# 関数: `R8(beta)` の derivative-chain row を返す。

def build_r8_chain_row(h_value: float) -> dict:
    """Return one derivative-chain row for the exact `R8(beta)` readout."""
    beta_common_root = float(BETA_COMMON_ROOT)
    exact_row = build_exact_relation_row(beta_common_root)
    derivative_row = build_integral_derivative_pack(beta_common_root, float(h_value))
    partials = evaluate_r8_exact_partials(
        beta_common_root,
        float(exact_row["i2"]),
        float(exact_row["ig"]),
        float(exact_row["i4"]),
        float(exact_row["boundary_weighted_eom"]),
    )

    i2_channel = float(partials["partial_r8_wrt_i2"] * derivative_row["d_i2_dbeta"])
    ig_channel = float(partials["partial_r8_wrt_ig"] * derivative_row["d_ig_dbeta"])
    i4_channel = float(partials["partial_r8_wrt_i4"] * derivative_row["d_i4_dbeta"])
    boundary_channel = float(
        partials["partial_r8_wrt_boundary"] * derivative_row["d_boundary_dbeta"]
    )
    beta_explicit_channel = float(partials["partial_r8_explicit_beta_channel"])
    negative_channels_sum = float(ig_channel + beta_explicit_channel)
    positive_channels_sum = float(i2_channel + i4_channel + boundary_channel)
    total_derivative = float(
        i2_channel + ig_channel + i4_channel + boundary_channel + beta_explicit_channel
    )

    return {
        "h": float(h_value),
        **partials,
        "i2_channel_dbeta": i2_channel,
        "ig_channel_dbeta": ig_channel,
        "i4_channel_dbeta": i4_channel,
        "boundary_channel_dbeta": boundary_channel,
        "beta_explicit_channel_dbeta": beta_explicit_channel,
        "negative_channels_sum_dbeta": negative_channels_sum,
        "positive_channels_sum_dbeta": positive_channels_sum,
        "r8_total_derivative_dbeta": total_derivative,
        "r8_total_derivative_fd_reference": float(derivative_row["d_alpha_r8_dbeta"]),
        "partial_i2_negative_now": bool(partials["partial_r8_wrt_i2"] < 0.0),
        "partial_ig_positive_now": bool(partials["partial_r8_wrt_ig"] > 0.0),
        "partial_i4_negative_now": bool(partials["partial_r8_wrt_i4"] < 0.0),
        "partial_boundary_negative_now": bool(partials["partial_r8_wrt_boundary"] < 0.0),
        "partial_beta_explicit_negative_now": bool(
            partials["partial_r8_explicit_beta_channel"] < 0.0
        ),
        "negative_channels_dominate_now": bool(
            abs(negative_channels_sum) > positive_channels_sum
        ),
        "r8_total_derivative_negative_now": bool(total_derivative < 0.0),
    }


# 関数: derivative-chain followup 監査 pack を返す。

def build_trial2_beta_sensitivity_derivative_chain_followup_pack() -> dict:
    """Return one derivative-chain audit pack for the strict-theorem route."""
    operator_pack = (
        build_trial2_beta_sensitivity_operator_level_spectral_projection_followup_pack()
    )
    alpha_rows = [build_alpha_qstar_chain_row(h_value) for h_value in H_VALUES]
    r8_rows = [build_r8_chain_row(h_value) for h_value in H_VALUES]

    alpha_values = [float(row["alpha_total_derivative_dbeta"]) for row in alpha_rows]
    r8_values = [float(row["r8_total_derivative_dbeta"]) for row in r8_rows]
    delta_values = [
        float(alpha_row["alpha_total_derivative_dbeta"] - r8_row["r8_total_derivative_dbeta"])
        for alpha_row, r8_row in zip(alpha_rows, r8_rows)
    ]

    alpha_rel_spread = float(
        (max(alpha_values) - min(alpha_values)) / max(abs(np.mean(alpha_values)), 1.0e-30)
    )
    r8_rel_spread = float(
        (max(r8_values) - min(r8_values)) / max(abs(np.mean(r8_values)), 1.0e-30)
    )
    delta_rel_spread = float(
        (max(delta_values) - min(delta_values)) / max(abs(np.mean(delta_values)), 1.0e-30)
    )

    alpha_qstar_derivative_chain_positive_local_support_available_now = bool(
        all(
            row["profile_channel_negative_now"]
            and row["qstar_channel_positive_now"]
            and row["qstar_channel_dominates_profile_now"]
            and row["alpha_total_derivative_positive_now"]
            for row in alpha_rows
        )
        and alpha_rel_spread <= ALPHA_DERIVATIVE_REL_SPREAD_TOL
    )
    r8_derivative_chain_negative_local_support_available_now = bool(
        all(
            row["partial_i2_negative_now"]
            and row["partial_ig_positive_now"]
            and row["partial_i4_negative_now"]
            and row["partial_boundary_negative_now"]
            and row["partial_beta_explicit_negative_now"]
            and row["negative_channels_dominate_now"]
            and row["r8_total_derivative_negative_now"]
            for row in r8_rows
        )
        and r8_rel_spread <= R8_DERIVATIVE_REL_SPREAD_TOL
    )
    delta_common_derivative_chain_positive_local_support_available_now = bool(
        alpha_qstar_derivative_chain_positive_local_support_available_now
        and r8_derivative_chain_negative_local_support_available_now
        and all(value > 0.0 for value in delta_values)
        and delta_rel_spread <= DELTA_DERIVATIVE_REL_SPREAD_TOL
    )
    exact_trial2_beta_sensitivity_derivative_chain_theorem_available_now = False
    updated_pack_trial2_beta_sensitivity_uniqueness_anchor_followup_required_now = bool(
        bool(operator_pack["exact_trial2_beta_sensitivity_weighted_integral_sign_support_available_now"])
        and alpha_qstar_derivative_chain_positive_local_support_available_now
        and r8_derivative_chain_negative_local_support_available_now
        and delta_common_derivative_chain_positive_local_support_available_now
        and not exact_trial2_beta_sensitivity_derivative_chain_theorem_available_now
    )

    return {
        "beta_common_root": float(BETA_COMMON_ROOT),
        "operator_pack": operator_pack,
        "alpha_chain_rows": alpha_rows,
        "r8_chain_rows": r8_rows,
        "alpha_total_derivative_min": float(min(alpha_values)),
        "alpha_total_derivative_max": float(max(alpha_values)),
        "alpha_total_derivative_rel_spread": alpha_rel_spread,
        "r8_total_derivative_min": float(min(r8_values)),
        "r8_total_derivative_max": float(max(r8_values)),
        "r8_total_derivative_rel_spread": r8_rel_spread,
        "delta_common_derivative_min": float(min(delta_values)),
        "delta_common_derivative_max": float(max(delta_values)),
        "delta_common_derivative_rel_spread": delta_rel_spread,
        "exact_trial2_alpha_qstar_derivative_chain_positive_local_support_available_now": (
            alpha_qstar_derivative_chain_positive_local_support_available_now
        ),
        "exact_trial2_r8_derivative_chain_negative_local_support_available_now": (
            r8_derivative_chain_negative_local_support_available_now
        ),
        "exact_trial2_delta_common_derivative_chain_positive_local_support_available_now": (
            delta_common_derivative_chain_positive_local_support_available_now
        ),
        "exact_trial2_beta_sensitivity_derivative_chain_theorem_available_now": (
            exact_trial2_beta_sensitivity_derivative_chain_theorem_available_now
        ),
        "updated_pack_trial2_beta_sensitivity_uniqueness_anchor_followup_required_now": (
            updated_pack_trial2_beta_sensitivity_uniqueness_anchor_followup_required_now
        ),
    }


# 関数: backend 単体実行時に retained metrics を表示する。

def main() -> None:
    """Run the derivative-chain backend directly and print key retained metrics."""
    pack = build_trial2_beta_sensitivity_derivative_chain_followup_pack()
    print("[trial2-beta-derivative-chain-followup]")
    print(f"beta_common_root = {pack['beta_common_root']:.16f}")
    print(
        "alpha_total_derivative = "
        f"{pack['alpha_total_derivative_min']:.16f} .. {pack['alpha_total_derivative_max']:.16f}"
    )
    print(
        "r8_total_derivative = "
        f"{pack['r8_total_derivative_min']:.16f} .. {pack['r8_total_derivative_max']:.16f}"
    )
    print(
        "delta_common_derivative = "
        f"{pack['delta_common_derivative_min']:.16f} .. {pack['delta_common_derivative_max']:.16f}"
    )
    print(
        "alpha_chain_positive = "
        f"{pack['exact_trial2_alpha_qstar_derivative_chain_positive_local_support_available_now']}"
    )
    print(
        "r8_chain_negative = "
        f"{pack['exact_trial2_r8_derivative_chain_negative_local_support_available_now']}"
    )
    print(
        "delta_chain_positive = "
        f"{pack['exact_trial2_delta_common_derivative_chain_positive_local_support_available_now']}"
    )


if __name__ == "__main__":
    main()
