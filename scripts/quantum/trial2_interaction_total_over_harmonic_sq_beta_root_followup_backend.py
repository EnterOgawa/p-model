#!/usr/bin/env python3
"""Audit one target-free beta selector for the exact R8 family.

Purpose:
    The exact relation

        R8(beta) = E_int(beta) * E_total(beta) / E_harm(beta)^2

    is already fixed by the weighted-EOM elimination, and one local beta root
    relative to alpha_target is known numerically. The remaining blocker is to
    select that beta without using alpha_target as comparator.

    This helper tests the cleanest target-free candidate:

        alpha_qstar(beta) = F_beta(q_star(beta))^2 / (4 pi)
        alpha_R8(beta) = R8_exact(beta)

        select beta from alpha_qstar(beta) = alpha_R8(beta)

    Both sides come from the frozen action on the retained scalar family, so
    the equality itself introduces no new parameter and no alpha_target input.

Inputs:
    - scripts/quantum/trial2_alpha_beta_curve_backend.py
    - scripts/quantum/trial2_interaction_total_over_harmonic_sq_exact_relation_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5623-.5630` wrappers

Assumptions:
    - alpha_target is used only as an external audit comparator
    - The selector is tested on the already retained localized beta family
    - The route is new relative to prior alpha_target-root audits because the
      root is selected by equality of two independent readouts instead
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.scalar_proxy_alpha_q_curve_backend import ALPHA_TARGET
from scripts.quantum.trial2_alpha_beta_curve_backend import build_beta_family_row
from scripts.quantum.trial2_interaction_total_over_harmonic_sq_exact_relation_backend import (
    build_exact_relation_row,
)


PRIOR_ALPHA_BETA_ROOT = 0.9982996989044647
PRIOR_R8_BETA_ROOT = 0.9983085664490956
PRIOR_Q_EXACT = 0.2416825755115744
PRIOR_RETAINED_BETA = 0.9982557379261291
SCAN_BETA_MIN = 0.94
SCAN_BETA_MAX = 0.999
SCAN_BETA_COUNT = 31


# 関数: two independent beta readouts の差を返す。
def build_common_root_difference(beta: float) -> float:
    """Return alpha_qstar(beta) - alpha_R8(beta) for one localized beta."""
    beta = float(beta)
    alpha_row = build_beta_family_row(beta)
    exact_row = build_exact_relation_row(beta)
    if alpha_row is None:
        raise SystemExit(f"[fail] localized alpha(beta) row is unavailable for beta={beta}")

    return float(alpha_row["alpha_at_q_star"]) - float(exact_row["exact_relation_from_integrals"])


# 関数: sampled beta family 上の common-root sign structure を返す。

def build_common_root_scan() -> dict:
    """Return one sampled scan pack for the target-free common-root candidate."""
    scan_betas = np.linspace(SCAN_BETA_MIN, SCAN_BETA_MAX, SCAN_BETA_COUNT, dtype=float)
    scan_rows: list[dict] = []
    sign_change_count = 0
    monotone_increasing_now = True
    previous_diff = math.nan

    for beta in scan_betas:
        alpha_row = build_beta_family_row(float(beta))
        if alpha_row is None:
            continue

        exact_row = build_exact_relation_row(float(beta))
        diff = float(alpha_row["alpha_at_q_star"] - exact_row["exact_relation_from_integrals"])
        scan_rows.append(
            {
                "beta": float(beta),
                "alpha_qstar": float(alpha_row["alpha_at_q_star"]),
                "alpha_r8_exact": float(exact_row["exact_relation_from_integrals"]),
                "difference": diff,
            }
        )

        if not math.isnan(previous_diff):
            monotone_increasing_now = bool(monotone_increasing_now and diff > previous_diff)
            if previous_diff == 0.0 or previous_diff * diff < 0.0:
                sign_change_count += 1

        previous_diff = diff

    if len(scan_rows) < 2:
        raise SystemExit("[fail] insufficient localized beta rows for common-root scan")

    return {
        "scan_rows": scan_rows,
        "scan_beta_min": float(scan_rows[0]["beta"]),
        "scan_beta_max": float(scan_rows[-1]["beta"]),
        "scan_row_count": int(len(scan_rows)),
        "difference_monotone_increasing_now": bool(monotone_increasing_now),
        "difference_sign_change_count": int(sign_change_count),
        "difference_first": float(scan_rows[0]["difference"]),
        "difference_last": float(scan_rows[-1]["difference"]),
    }


# 関数: sampled sign change から common root を materialize する。

def find_target_free_common_root(scan_pack: dict) -> dict:
    """Return the unique common root implied by the sampled equality scan."""
    scan_rows = list(scan_pack["scan_rows"])
    left_beta = math.nan
    right_beta = math.nan

    for left_row, right_row in zip(scan_rows[:-1], scan_rows[1:]):
        left_diff = float(left_row["difference"])
        right_diff = float(right_row["difference"])
        if left_diff == 0.0:
            left_beta = float(left_row["beta"])
            right_beta = float(left_row["beta"])
            break

        if left_diff * right_diff < 0.0:
            left_beta = float(left_row["beta"])
            right_beta = float(right_row["beta"])
            break

    if math.isnan(left_beta) or math.isnan(right_beta):
        return {
            "common_root_available_now": False,
            "common_root_left_beta": math.nan,
            "common_root_right_beta": math.nan,
            "beta_common_root": math.nan,
            "beta_common_root_rel_shift_vs_retained": math.nan,
            "beta_common_root_rel_shift_vs_prior_alpha_beta": math.nan,
            "beta_common_root_rel_shift_vs_prior_r8_beta_root": math.nan,
            "alpha_common_value": math.nan,
            "alpha_common_rel_error_vs_target": math.nan,
            "q_star_common_over_m0": math.nan,
            "q_star_common_rel_shift_vs_q_exact": math.nan,
        }

    if left_beta == right_beta:
        beta_common_root = float(left_beta)
    else:
        beta_common_root = float(
            brentq(build_common_root_difference, float(left_beta), float(right_beta))
        )

    alpha_row = build_beta_family_row(beta_common_root)
    exact_row = build_exact_relation_row(beta_common_root)
    if alpha_row is None:
        raise SystemExit("[fail] localized alpha(beta) row is unavailable at the common root")

    alpha_common = float(alpha_row["alpha_at_q_star"])
    q_star_common = float(alpha_row["q_star_over_m0"])
    if abs(alpha_common - float(exact_row["exact_relation_from_integrals"])) > 1.0e-12:
        raise SystemExit("[fail] common-root alpha readouts do not agree at the selected beta")

    return {
        "common_root_available_now": True,
        "common_root_left_beta": float(left_beta),
        "common_root_right_beta": float(right_beta),
        "beta_common_root": beta_common_root,
        "beta_common_root_rel_shift_vs_retained": float(
            (beta_common_root - PRIOR_RETAINED_BETA) / PRIOR_RETAINED_BETA
        ),
        "beta_common_root_rel_shift_vs_prior_alpha_beta": float(
            (beta_common_root - PRIOR_ALPHA_BETA_ROOT) / PRIOR_ALPHA_BETA_ROOT
        ),
        "beta_common_root_rel_shift_vs_prior_r8_beta_root": float(
            (beta_common_root - PRIOR_R8_BETA_ROOT) / PRIOR_R8_BETA_ROOT
        ),
        "alpha_common_value": alpha_common,
        "alpha_common_rel_error_vs_target": float(
            (alpha_common - ALPHA_TARGET) / ALPHA_TARGET
        ),
        "q_star_common_over_m0": q_star_common,
        "q_star_common_rel_shift_vs_q_exact": float(
            (q_star_common - PRIOR_Q_EXACT) / PRIOR_Q_EXACT
        ),
    }


# 関数: beta-root followup 監査 pack 全体を返す。

def build_trial2_interaction_total_over_harmonic_sq_beta_root_followup_pack() -> dict:
    """Return one target-free common-root audit pack for the exact R8 family."""
    scan_pack = build_common_root_scan()
    common_root_pack = find_target_free_common_root(scan_pack)
    common_root_available_now = bool(common_root_pack["common_root_available_now"])
    monotone_increasing_now = bool(scan_pack["difference_monotone_increasing_now"])
    sign_change_count = int(scan_pack["difference_sign_change_count"])

    target_free_beta_selector_available_now = bool(
        common_root_available_now
        and monotone_increasing_now
        and sign_change_count == 1
    )
    practical_direct_alpha_closeout_available_now = bool(
        target_free_beta_selector_available_now
        and abs(float(common_root_pack["alpha_common_rel_error_vs_target"])) <= 1.0e-3
    )
    strict_target_free_theorem_closeout_available_now = False
    strict_theorem_followup_required_now = bool(
        target_free_beta_selector_available_now
        and practical_direct_alpha_closeout_available_now
        and not strict_target_free_theorem_closeout_available_now
    )

    return {
        "alpha_target": float(ALPHA_TARGET),
        "prior_retained_beta": float(PRIOR_RETAINED_BETA),
        "prior_alpha_beta_root": float(PRIOR_ALPHA_BETA_ROOT),
        "prior_r8_beta_root": float(PRIOR_R8_BETA_ROOT),
        "prior_q_exact_over_m0": float(PRIOR_Q_EXACT),
        **scan_pack,
        **common_root_pack,
        "target_free_beta_selector_available_now": target_free_beta_selector_available_now,
        "practical_direct_alpha_closeout_available_now": (
            practical_direct_alpha_closeout_available_now
        ),
        "strict_target_free_theorem_closeout_available_now": (
            strict_target_free_theorem_closeout_available_now
        ),
        "strict_theorem_followup_required_now": strict_theorem_followup_required_now,
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the target-free common-root audit directly."""
    pack = build_trial2_interaction_total_over_harmonic_sq_beta_root_followup_pack()
    print("[trial2_interaction_total_over_harmonic_sq_beta_root_followup_backend]")
    print(f"  target_free_beta_selector = {pack['target_free_beta_selector_available_now']}")
    print(f"  beta_common_root = {pack['beta_common_root']:.15f}")
    print(f"  alpha_common = {pack['alpha_common_value']:.15f}")


if __name__ == "__main__":
    main()
