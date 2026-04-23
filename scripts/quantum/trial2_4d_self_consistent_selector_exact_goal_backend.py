#!/usr/bin/env python3
"""Audit the self-consistent 4D selector route for the exact goal.

Purpose:
    After the deterministic external-probe candidate closed the direct
    theorem-source route negatively, the next genuinely different computation
    route is to stop treating the 4D correction as a post-selected add-on and
    instead solve one 4D-corrected selector self-consistently.

    The current pack already carries two independent 3D readouts

        alpha_qstar(beta)
        alpha_R8(beta)

    and one retained leading 4D selector family built from charge and mass
    correction factors. This helper asks a narrower question than a full new
    theorem search:

        if the q-star readout and the energy-partition readout are both given
        deterministic 4D corrections inside the current selector family, does
        the corrected selector root beat the current best exact-goal residual?

Inputs:
    - scripts/quantum/trial2_alpha_beta_curve_backend.py
    - scripts/quantum/trial2_interaction_total_over_harmonic_sq_exact_relation_backend.py
    - scripts/quantum/trial2_4d_full_integral_external_probe_current_vertex_exactification_backend.py
    - scripts/quantum/mass_origin_vector_qball_full_coupled_solver_branch.py

Outputs:
    - One in-memory audit pack consumed by `.5879-.5886` wrappers

Assumptions:
    - The exact goal `1/137` is used only as a comparator
    - No new parameter is introduced
    - The route is computation-only and audits the current deterministic 4D
      family before any genuinely new source branch is attempted
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

from scripts.quantum.mass_origin_vector_qball_full_coupled_solver_branch import (
    coupled_charge_factor,
)
from scripts.quantum.mass_origin_vector_qball_full_coupled_solver_branch import (
    coupled_mass_factor,
)
from scripts.quantum.mass_origin_vector_qball_full_coupled_solver_branch import (
    polarization_weight,
)
from scripts.quantum.trial2_4d_full_integral_external_probe_current_vertex_exactification_backend import (
    build_trial2_4d_full_integral_external_probe_current_vertex_exactification_pack,
)
from scripts.quantum.trial2_alpha_beta_curve_backend import build_beta_family_row
from scripts.quantum.trial2_interaction_total_over_harmonic_sq_exact_relation_backend import (
    build_exact_relation_row,
)


EXACT_GOAL_ALPHA = 1.0 / 137.0
BETA_SCAN_MIN = 0.995
BETA_SCAN_MAX = 0.999
BETA_SCAN_COUNT = 17
PRIOR_COMMON_ROOT_BETA = 0.9983014161324819
SELECTOR_FAMILY = (
    (1, 1),
    (1, 0),
    (2, -1),
    (3, 1),
)
Q_SIDE_LABELS = (
    "probe_mass_sq",
    "probe_half_mix",
    "probe_vertex_mix",
    "probe_all_mix",
    "probe_charge_mass",
)
R_SIDE_LABELS = (
    "bulk_none",
    "bulk_mass_sq",
    "bulk_half_mix",
    "bulk_all_mix",
    "bulk_charge_mass",
)
EXPERT_MINIMAL_Q_LABEL = "probe_vertex_mix"
EXPERT_MINIMAL_R_LABEL = "bulk_mass_sq"


# 関数: 1 つの beta における current selector-family weights を返す。
@lru_cache(maxsize=None)
def build_weight_row(beta: float) -> dict:
    """Return deterministic leading-family weights and 4D factors on one beta."""
    beta = float(beta)
    weight_rows: list[tuple[int, int, float]] = []
    for ell, s in SELECTOR_FAMILY:
        weight_rows.append((ell, s, float(polarization_weight(beta, ell, s))))

    leading_weight = float(weight_rows[0][2])
    nonzero_time_denom = float(sum(weight for _, s, weight in weight_rows if s != 0))
    all_selector_denom = float(sum(weight for _, _, weight in weight_rows))
    charge_factor = float(coupled_charge_factor(beta, 1, 1))
    mass_factor = float(coupled_mass_factor(beta, 1, 1))
    return {
        "beta": beta,
        "leading_weight": leading_weight,
        "eta_vertex": float(leading_weight / nonzero_time_denom),
        "eta_all": float(leading_weight / all_selector_denom),
        "charge_factor": charge_factor,
        "mass_factor": mass_factor,
    }


# 関数: mixed normalization を返す。

def mixed_normalization(alpha_value: float, charge_factor: float, mass_factor: float, eta: float) -> float:
    """Return alpha divided by the mixed charge/mass 4D denominator."""
    return float(alpha_value / (charge_factor**eta * mass_factor ** (2.0 - eta)))


# 関数: q-star readout に deterministic 4D correction を掛ける。

@lru_cache(maxsize=None)
def corrected_alpha_qstar(beta: float, q_label: str) -> float:
    """Return one deterministic 4D-corrected q-star readout."""
    beta = float(beta)
    alpha_row = build_beta_family_row(beta)
    if alpha_row is None:
        raise SystemExit(f"[fail] alpha_qstar row is unavailable for beta={beta}")

    weight_row = build_weight_row(beta)
    alpha_value = float(alpha_row["alpha_at_q_star"])
    charge_factor = float(weight_row["charge_factor"])
    mass_factor = float(weight_row["mass_factor"])
    eta_vertex = float(weight_row["eta_vertex"])
    eta_all = float(weight_row["eta_all"])

    if q_label == "probe_mass_sq":
        return mixed_normalization(alpha_value, charge_factor, mass_factor, 0.0)

    if q_label == "probe_half_mix":
        return mixed_normalization(alpha_value, charge_factor, mass_factor, 0.5)

    if q_label == "probe_vertex_mix":
        return mixed_normalization(alpha_value, charge_factor, mass_factor, eta_vertex)

    if q_label == "probe_all_mix":
        return mixed_normalization(alpha_value, charge_factor, mass_factor, eta_all)

    if q_label == "probe_charge_mass":
        return mixed_normalization(alpha_value, charge_factor, mass_factor, 1.0)

    raise ValueError(f"unknown q-side label: {q_label}")


# 関数: energy-partition readout に deterministic 4D correction を掛ける。

@lru_cache(maxsize=None)
def corrected_alpha_r8(beta: float, r_label: str) -> float:
    """Return one deterministic 4D-corrected energy-partition readout."""
    beta = float(beta)
    exact_row = build_exact_relation_row(beta)
    weight_row = build_weight_row(beta)
    alpha_value = float(exact_row["exact_relation_from_integrals"])
    charge_factor = float(weight_row["charge_factor"])
    mass_factor = float(weight_row["mass_factor"])
    eta_all = float(weight_row["eta_all"])

    if r_label == "bulk_none":
        return alpha_value

    if r_label == "bulk_mass_sq":
        return mixed_normalization(alpha_value, charge_factor, mass_factor, 0.0)

    if r_label == "bulk_half_mix":
        return mixed_normalization(alpha_value, charge_factor, mass_factor, 0.5)

    if r_label == "bulk_all_mix":
        return mixed_normalization(alpha_value, charge_factor, mass_factor, eta_all)

    if r_label == "bulk_charge_mass":
        return mixed_normalization(alpha_value, charge_factor, mass_factor, 1.0)

    raise ValueError(f"unknown r-side label: {r_label}")


# 関数: corrected selector difference を返す。

def build_self_consistent_difference(beta: float, q_label: str, r_label: str) -> float:
    """Return alpha_qstar^(4D)(beta) - alpha_R8^(4D)(beta) for one pair."""
    return float(corrected_alpha_qstar(beta, q_label) - corrected_alpha_r8(beta, r_label))


# 関数: 1 組の deterministic pair を scan / root solve する。

def build_pair_row(q_label: str, r_label: str) -> dict:
    """Return one self-consistent selector row for a deterministic pair."""
    scan_betas = np.linspace(BETA_SCAN_MIN, BETA_SCAN_MAX, BETA_SCAN_COUNT, dtype=float)
    scan_rows: list[dict] = []
    sign_change_count = 0
    monotone_increasing_now = True
    previous_diff = math.nan

    for beta in scan_betas:
        diff_value = float(build_self_consistent_difference(float(beta), q_label, r_label))
        scan_rows.append({"beta": float(beta), "difference": diff_value})
        if not math.isnan(previous_diff):
            monotone_increasing_now = bool(monotone_increasing_now and diff_value > previous_diff)
            if previous_diff == 0.0 or previous_diff * diff_value < 0.0:
                sign_change_count += 1

        previous_diff = diff_value

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

    root_available_now = bool(not math.isnan(left_beta) and not math.isnan(right_beta))
    if not root_available_now:
        return {
            "q_label": q_label,
            "r_label": r_label,
            "scan_rows": scan_rows,
            "difference_monotone_increasing_now": bool(monotone_increasing_now),
            "difference_sign_change_count": int(sign_change_count),
            "root_available_now": False,
            "root_beta": math.nan,
            "alpha_q_corrected_at_root": math.nan,
            "alpha_r_corrected_at_root": math.nan,
            "alpha_self_consistent_at_root": math.nan,
            "alpha_self_consistent_rel_error_vs_exact_goal": math.nan,
        }

    if left_beta == right_beta:
        root_beta = float(left_beta)
    else:
        root_beta = float(
            brentq(
                lambda beta: float(build_self_consistent_difference(float(beta), q_label, r_label)),
                float(left_beta),
                float(right_beta),
            )
        )

    alpha_q = float(corrected_alpha_qstar(root_beta, q_label))
    alpha_r = float(corrected_alpha_r8(root_beta, r_label))
    if abs(alpha_q - alpha_r) > 1.0e-12:
        raise SystemExit(
            "[fail] corrected self-consistent root does not equalize the two readouts"
        )

    alpha_self = float(alpha_q)
    rel_error = float((alpha_self - EXACT_GOAL_ALPHA) / EXACT_GOAL_ALPHA)
    return {
        "q_label": q_label,
        "r_label": r_label,
        "scan_rows": scan_rows,
        "difference_monotone_increasing_now": bool(monotone_increasing_now),
        "difference_sign_change_count": int(sign_change_count),
        "root_available_now": True,
        "root_beta": root_beta,
        "root_beta_rel_shift_vs_prior_common_root": float(
            (root_beta - PRIOR_COMMON_ROOT_BETA) / max(abs(PRIOR_COMMON_ROOT_BETA), 1.0e-30)
        ),
        "alpha_q_corrected_at_root": alpha_q,
        "alpha_r_corrected_at_root": alpha_r,
        "alpha_self_consistent_at_root": alpha_self,
        "alpha_self_consistent_rel_error_vs_exact_goal": rel_error,
    }


# 関数: self-consistent 4D selector audit pack を返す。

def build_trial2_4d_self_consistent_selector_exact_goal_pack() -> dict:
    """Return the full deterministic self-consistent 4D selector audit pack."""
    prior_pack = build_trial2_4d_full_integral_external_probe_current_vertex_exactification_pack()
    candidate_rows = [
        build_pair_row(q_label, r_label)
        for q_label in Q_SIDE_LABELS
        for r_label in R_SIDE_LABELS
    ]
    available_rows = [row for row in candidate_rows if row["root_available_now"]]
    if not available_rows:
        raise SystemExit("[fail] no deterministic self-consistent 4D selector row materialized")

    best_row = min(
        available_rows,
        key=lambda row: abs(float(row["alpha_self_consistent_rel_error_vs_exact_goal"])),
    )
    expert_minimal_row = next(
        row
        for row in candidate_rows
        if row["q_label"] == EXPERT_MINIMAL_Q_LABEL and row["r_label"] == EXPERT_MINIMAL_R_LABEL
    )

    prior_rel_error = float(prior_pack["alpha_vertex_candidate_rel_error_vs_exact_goal"])
    canonical_rel_error = float(prior_pack["canonical_row"]["corrected_alpha_rel_error_vs_exact_goal"])
    best_rel_error = float(best_row["alpha_self_consistent_rel_error_vs_exact_goal"])
    expert_rel_error = float(expert_minimal_row["alpha_self_consistent_rel_error_vs_exact_goal"])

    family_available_now = True
    expert_minimal_pair_available_now = bool(expert_minimal_row["root_available_now"])
    best_pair_improves_canonical_now = bool(abs(best_rel_error) < abs(canonical_rel_error))
    best_pair_beats_current_best_now = bool(abs(best_rel_error) < abs(prior_rel_error))
    zero_residual_available_now = bool(abs(best_rel_error) <= 1.0e-14)
    positive_partial_now = bool(best_pair_improves_canonical_now)
    negative_closeout_now = bool(
        family_available_now and positive_partial_now and not best_pair_beats_current_best_now and not zero_residual_available_now
    )

    return {
        "alpha_goal_exact_one_over_137": float(EXACT_GOAL_ALPHA),
        "prior_best_alpha_vertex_candidate": float(prior_pack["alpha_vertex_candidate"]),
        "prior_best_rel_error_vs_exact_goal": prior_rel_error,
        "prior_canonical_rel_error_vs_exact_goal": canonical_rel_error,
        "candidate_rows": candidate_rows,
        "available_row_count": int(len(available_rows)),
        "best_row": best_row,
        "expert_minimal_row": expert_minimal_row,
        "best_pair_improvement_factor_vs_canonical": float(
            abs(canonical_rel_error) / max(abs(best_rel_error), 1.0e-30)
        ),
        "best_pair_improvement_factor_vs_current_best": float(
            abs(prior_rel_error) / max(abs(best_rel_error), 1.0e-30)
        ),
        "expert_minimal_improvement_factor_vs_current_best": float(
            abs(prior_rel_error) / max(abs(expert_rel_error), 1.0e-30)
        ),
        "exact_trial2_4d_self_consistent_selector_family_available_now": (
            family_available_now
        ),
        "exact_trial2_4d_self_consistent_selector_expert_minimal_pair_available_now": (
            expert_minimal_pair_available_now
        ),
        "exact_trial2_4d_self_consistent_selector_positive_partial_now": (
            positive_partial_now
        ),
        "exact_trial2_4d_self_consistent_selector_beats_current_best_now": (
            best_pair_beats_current_best_now
        ),
        "exact_trial2_4d_self_consistent_selector_zero_residual_exact_goal_available_now": (
            zero_residual_available_now
        ),
        "updated_pack_trial2_4d_self_consistent_selector_negative_closeout_now": (
            negative_closeout_now
        ),
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the self-consistent 4D selector audit directly."""
    pack = build_trial2_4d_self_consistent_selector_exact_goal_pack()
    best_row = pack["best_row"]
    expert_row = pack["expert_minimal_row"]
    print("[trial2_4d_self_consistent_selector_exact_goal_backend]")
    print(
        "  best_pair = "
        f"{best_row['q_label']} / {best_row['r_label']}"
    )
    print(f"  best_alpha = {best_row['alpha_self_consistent_at_root']:.15f}")
    print(
        "  best_rel_error_vs_exact_goal = "
        f"{best_row['alpha_self_consistent_rel_error_vs_exact_goal']:+.12e}"
    )
    print(
        "  expert_pair = "
        f"{expert_row['q_label']} / {expert_row['r_label']}"
    )
    print(
        "  expert_rel_error_vs_exact_goal = "
        f"{expert_row['alpha_self_consistent_rel_error_vs_exact_goal']:+.12e}"
    )


if __name__ == "__main__":
    main()
