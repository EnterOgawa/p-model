#!/usr/bin/env python3
"""Audit the spectral-projection followup after the Green-kernel split verdict.

Purpose:
    Continue the strict-theorem route after the canonical-window Green-kernel
    audit. The naive one-sign-kernel theorem is already closed negatively, but
    the actual source-weighted resolvent solution remains strictly negative on
    the same window. This backend asks whether that negativity can be upgraded
    into one exact finite spectral-projection theorem on the discretized
    Dirichlet operator:

        H_beta w_beta = 2 beta x y_beta.

    The route is intentionally narrower than the failed maximum-principle /
    Green-kernel paths:

    1. keep the exact transformed operator and source,
    2. use the full discrete spectral decomposition on the canonical window,
    3. test whether the principal negative mode dominates the absolute remainder
       pointwise, and
    4. separate that exact discrete theorem from the still-missing continuum
       theorem.

Inputs:
    - scripts/quantum/trial2_beta_sensitivity_green_kernel_followup_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5663-.5670` wrappers

Assumptions:
    - The canonical window remains [0.05, 20]
    - No new parameter is introduced
    - The route tests spectral projection only; it does not replay old
      q-selector / residue / entropy / Jost branches
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_beta_sensitivity_green_kernel_followup_backend import (
    BETA_COMMON_ROOT,
)
from scripts.quantum.trial2_beta_sensitivity_green_kernel_followup_backend import (
    WINDOW_POINT_COUNTS,
)
from scripts.quantum.trial2_beta_sensitivity_green_kernel_followup_backend import (
    build_dirichlet_operator_row,
)


RECONSTRUCTION_REL_TOL = 1.0e-8


# 関数: one resolution の full spectral row を返す。
def build_full_spectral_projection_row(point_count: int) -> dict:
    """Return one exact full-spectrum projection row on the canonical window."""
    operator_row = build_dirichlet_operator_row(int(point_count))
    eigenvalues, eigenvectors = eigh_tridiagonal(
        operator_row["diagonal"],
        operator_row["off_diagonal"],
        check_finite=False,
    )
    source = np.asarray(operator_row["source"][1:-1], dtype=float)
    overlaps = np.asarray(eigenvectors.T @ source, dtype=float)
    coefficients = np.asarray(overlaps / eigenvalues, dtype=float)
    principal_mode = np.asarray(eigenvectors[:, 0], dtype=float)
    principal_component = np.asarray(coefficients[0] * principal_mode, dtype=float)
    remainder_component = np.asarray(
        eigenvectors[:, 1:] @ coefficients[1:],
        dtype=float,
    )
    reconstructed_solution = np.asarray(
        principal_component + remainder_component,
        dtype=float,
    )
    direct_solution = np.asarray(operator_row["lu_factor"].solve(source), dtype=float)
    principal_abs = np.asarray(
        abs(float(coefficients[0])) * np.abs(principal_mode),
        dtype=float,
    )
    remainder_abs_sum = np.asarray(
        np.sum(np.abs(eigenvectors[:, 1:]) * np.abs(coefficients[1:]), axis=1),
        dtype=float,
    )
    pointwise_margin = np.asarray(principal_abs - remainder_abs_sum, dtype=float)
    reconstruction_abs_max = float(
        np.max(np.abs(reconstructed_solution - direct_solution))
    )
    direct_solution_abs_max = float(np.max(np.abs(direct_solution)))
    reconstruction_rel_linf = float(
        reconstruction_abs_max / max(direct_solution_abs_max, 1.0e-30)
    )
    principal_mode_one_sign_now = bool(
        float(np.max(principal_mode)) < 0.0 or float(np.min(principal_mode)) > 0.0
    )
    principal_component_negative_now = bool(float(np.max(principal_component)) < 0.0)
    discrete_pointwise_dominance_now = bool(float(np.min(pointwise_margin)) > 0.0)
    reconstructed_solution_all_negative_now = bool(
        float(np.max(reconstructed_solution)) < 0.0
    )
    exact_discrete_negativity_theorem_now = bool(
        principal_mode_one_sign_now
        and principal_component_negative_now
        and discrete_pointwise_dominance_now
        and reconstructed_solution_all_negative_now
        and reconstruction_rel_linf <= RECONSTRUCTION_REL_TOL
    )
    return {
        "point_count": int(point_count),
        "lambda_1": float(eigenvalues[0]),
        "lambda_2": float(eigenvalues[1]),
        "principal_mode_min": float(np.min(principal_mode)),
        "principal_mode_max": float(np.max(principal_mode)),
        "principal_mode_one_sign_now": principal_mode_one_sign_now,
        "principal_coefficient": float(coefficients[0]),
        "principal_component_min": float(np.min(principal_component)),
        "principal_component_max": float(np.max(principal_component)),
        "principal_component_negative_now": principal_component_negative_now,
        "remainder_component_min": float(np.min(remainder_component)),
        "remainder_component_max": float(np.max(remainder_component)),
        "pointwise_margin_min": float(np.min(pointwise_margin)),
        "pointwise_margin_max": float(np.max(pointwise_margin)),
        "discrete_pointwise_dominance_now": discrete_pointwise_dominance_now,
        "reconstructed_solution_min": float(np.min(reconstructed_solution)),
        "reconstructed_solution_max": float(np.max(reconstructed_solution)),
        "reconstructed_solution_negative_fraction": float(
            np.mean(reconstructed_solution < 0.0)
        ),
        "reconstructed_solution_all_negative_now": (
            reconstructed_solution_all_negative_now
        ),
        "direct_solution_min": float(np.min(direct_solution)),
        "direct_solution_max": float(np.max(direct_solution)),
        "reconstruction_abs_max": reconstruction_abs_max,
        "reconstruction_rel_linf": reconstruction_rel_linf,
        "exact_discrete_negativity_theorem_now": exact_discrete_negativity_theorem_now,
    }


# 関数: spectral-projection followup 監査 pack を返す。

def build_trial2_beta_sensitivity_spectral_projection_followup_pack() -> dict:
    """Return one audit pack for the full spectral-projection followup."""
    spectral_rows = [
        build_full_spectral_projection_row(point_count)
        for point_count in WINDOW_POINT_COUNTS
    ]
    exact_trial2_beta_sensitivity_discrete_spectral_projection_available_now = True
    exact_trial2_beta_sensitivity_principal_mode_one_sign_now = bool(
        all(row["principal_mode_one_sign_now"] for row in spectral_rows)
    )
    exact_trial2_beta_sensitivity_principal_component_negative_now = bool(
        all(row["principal_component_negative_now"] for row in spectral_rows)
    )
    exact_trial2_beta_sensitivity_discrete_pointwise_dominance_now = bool(
        all(row["discrete_pointwise_dominance_now"] for row in spectral_rows)
    )
    exact_trial2_beta_sensitivity_discrete_negativity_theorem_available_now = bool(
        all(row["exact_discrete_negativity_theorem_now"] for row in spectral_rows)
    )
    exact_trial2_common_root_monotonicity_theorem_available_now = False
    updated_pack_trial2_beta_sensitivity_continuum_spectral_projection_followup_required_now = bool(
        exact_trial2_beta_sensitivity_discrete_negativity_theorem_available_now
        and not exact_trial2_common_root_monotonicity_theorem_available_now
    )
    return {
        "beta_common_root": float(BETA_COMMON_ROOT),
        "spectral_rows": spectral_rows,
        "exact_trial2_beta_sensitivity_discrete_spectral_projection_available_now": (
            exact_trial2_beta_sensitivity_discrete_spectral_projection_available_now
        ),
        "exact_trial2_beta_sensitivity_principal_mode_one_sign_now": (
            exact_trial2_beta_sensitivity_principal_mode_one_sign_now
        ),
        "exact_trial2_beta_sensitivity_principal_component_negative_now": (
            exact_trial2_beta_sensitivity_principal_component_negative_now
        ),
        "exact_trial2_beta_sensitivity_discrete_pointwise_dominance_now": (
            exact_trial2_beta_sensitivity_discrete_pointwise_dominance_now
        ),
        "exact_trial2_beta_sensitivity_discrete_negativity_theorem_available_now": (
            exact_trial2_beta_sensitivity_discrete_negativity_theorem_available_now
        ),
        "exact_trial2_common_root_monotonicity_theorem_available_now": (
            exact_trial2_common_root_monotonicity_theorem_available_now
        ),
        "updated_pack_trial2_beta_sensitivity_continuum_spectral_projection_followup_required_now": (
            updated_pack_trial2_beta_sensitivity_continuum_spectral_projection_followup_required_now
        ),
        "discrete_pointwise_margin_min_global": float(
            min(row["pointwise_margin_min"] for row in spectral_rows)
        ),
        "discrete_pointwise_margin_max_global": float(
            max(row["pointwise_margin_max"] for row in spectral_rows)
        ),
        "reconstruction_rel_linf_max": float(
            max(row["reconstruction_rel_linf"] for row in spectral_rows)
        ),
        "spectral_lambda_1_min": float(min(row["lambda_1"] for row in spectral_rows)),
        "spectral_lambda_2_min": float(min(row["lambda_2"] for row in spectral_rows)),
    }


# 関数: backend 単体実行時に retained metrics を表示する。

def main() -> None:
    """Run the spectral-projection backend directly and print the retained metrics."""
    pack = build_trial2_beta_sensitivity_spectral_projection_followup_pack()
    print("[trial2-beta-spectral-projection-followup]")
    print(f"beta_common_root = {pack['beta_common_root']:.16f}")
    for row in pack["spectral_rows"]:
        print(
            "point_count="
            f"{row['point_count']} "
            f"lambda1={row['lambda_1']:.16f} "
            f"lambda2={row['lambda_2']:.16f} "
            f"margin_min={row['pointwise_margin_min']:.16f} "
            f"recon_rel={row['reconstruction_rel_linf']:.16e} "
            f"all_negative={row['reconstructed_solution_all_negative_now']}"
        )

    print(
        "discrete_theorem_available_now = "
        f"{pack['exact_trial2_beta_sensitivity_discrete_negativity_theorem_available_now']}"
    )
    print(
        "continuum_followup_required_now = "
        f"{pack['updated_pack_trial2_beta_sensitivity_continuum_spectral_projection_followup_required_now']}"
    )


if __name__ == "__main__":
    main()
