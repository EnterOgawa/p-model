#!/usr/bin/env python3
"""Audit the Green-kernel followup after the Trial-2 maximum-principle failure.

Purpose:
    Continue the strict-theorem route after the canonical-window maximum-
    principle path closed negatively. The transformed operator

        H_beta w_beta = 2 beta x y_beta

    is already exact and the principal Dirichlet eigenvalue is known to satisfy
    lambda_1 < 0 < lambda_2 on the canonical window. This backend checks two
    genuinely new facts:

    1. whether the Dirichlet Green kernel is globally one-sign, and
    2. whether the source-weighted resolvent solution still stays one-sign even
       when the kernel itself does not.

Inputs:
    - scripts/quantum/trial2_beta_sensitivity_monotonicity_followup_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5655-.5662` wrappers

Assumptions:
    - The canonical window remains [0.05, 20]
    - No new parameter is introduced
    - The branch tests only the transformed Green-kernel / resolvent route
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import diags
from scipy.sparse.linalg import splu


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_beta_sensitivity_monotonicity_followup_backend import (
    CANONICAL_WINDOW_X_MAX,
)
from scripts.quantum.trial2_beta_sensitivity_monotonicity_followup_backend import (
    WINDOW_X_MIN,
)
from scripts.quantum.trial2_beta_sensitivity_monotonicity_followup_backend import (
    evaluate_transformed_potential,
)
from scripts.quantum.trial2_beta_sensitivity_monotonicity_followup_backend import (
    evaluate_transformed_source,
)
from scripts.quantum.trial2_beta_sensitivity_monotonicity_followup_backend import (
    BETA_COMMON_ROOT,
)


WINDOW_POINT_COUNTS = (240, 600, 1200)
PROBE_FRACTIONS = (0.25, 0.50, 0.75)
SPECTRAL_MODE_COUNT = 50


# 関数: canonical Dirichlet operator row を組み立てる。
def build_dirichlet_operator_row(point_count: int) -> dict:
    """Return one canonical-window Dirichlet operator row."""
    point_count = int(point_count)
    grid = np.linspace(
        WINDOW_X_MIN,
        CANONICAL_WINDOW_X_MAX,
        point_count,
        dtype=float,
    )
    potential = evaluate_transformed_potential(grid)
    source = evaluate_transformed_source(grid)
    step = float(grid[1] - grid[0])
    diagonal = 2.0 / (step * step) - potential[1:-1]
    off_diagonal = np.full(diagonal.size - 1, -1.0 / (step * step), dtype=float)
    sparse_matrix = diags(
        [off_diagonal, diagonal, off_diagonal],
        offsets=[-1, 0, 1],
        format="csc",
    )
    lu_factor = splu(sparse_matrix)
    return {
        "point_count": point_count,
        "grid": grid,
        "step": step,
        "potential": potential,
        "source": source,
        "diagonal": diagonal,
        "off_diagonal": off_diagonal,
        "lu_factor": lu_factor,
    }


# 関数: sampled Green-kernel column の符号 row を返す。

def build_green_column_row(operator_row: dict, probe_fraction: float) -> dict:
    """Return one sampled Green-column sign row."""
    diagonal_size = int(operator_row["diagonal"].size)
    column_index = int(round(float(probe_fraction) * float(diagonal_size - 1)))
    rhs = np.zeros(diagonal_size, dtype=float)
    rhs[column_index] = 1.0
    column = operator_row["lu_factor"].solve(rhs)
    return {
        "probe_fraction": float(probe_fraction),
        "column_index": column_index,
        "column_min": float(np.min(column)),
        "column_max": float(np.max(column)),
        "column_negative_fraction": float(np.mean(column < 0.0)),
        "column_positive_fraction": float(np.mean(column > 0.0)),
        "column_sign_mixed_now": bool(
            float(np.min(column)) < 0.0 and float(np.max(column)) > 0.0
        ),
        "column_all_negative_now": bool(float(np.max(column)) < 0.0),
    }


# 関数: source-weighted resolvent solution row を返す。

def build_source_solution_row(operator_row: dict) -> dict:
    """Return one sign row for the source-weighted resolvent solution."""
    source_solution = operator_row["lu_factor"].solve(operator_row["source"][1:-1])
    return {
        "solution_min": float(np.min(source_solution)),
        "solution_max": float(np.max(source_solution)),
        "solution_negative_fraction": float(np.mean(source_solution < 0.0)),
        "solution_positive_fraction": float(np.mean(source_solution > 0.0)),
        "solution_all_negative_now": bool(float(np.max(source_solution)) < 0.0),
    }


# 関数: coarse full inverse の符号 row を返す。

def build_coarse_full_inverse_row(operator_row: dict) -> dict:
    """Return one coarse full-inverse sign row."""
    identity = np.eye(int(operator_row["diagonal"].size), dtype=float)
    inverse_matrix = operator_row["lu_factor"].solve(identity)
    return {
        "full_inverse_min": float(np.min(inverse_matrix)),
        "full_inverse_max": float(np.max(inverse_matrix)),
        "full_inverse_negative_fraction": float(np.mean(inverse_matrix < 0.0)),
        "full_inverse_positive_fraction": float(np.mean(inverse_matrix > 0.0)),
        "full_inverse_one_sign_now": bool(
            float(np.max(inverse_matrix)) < 0.0 or float(np.min(inverse_matrix)) > 0.0
        ),
    }


# 関数: spectral projection row を返す。

def build_spectral_projection_row(operator_row: dict) -> dict:
    """Return one spectral-projection dominance row on the canonical window."""
    eigenvalues, eigenvectors = eigh_tridiagonal(
        operator_row["diagonal"],
        operator_row["off_diagonal"],
        select="i",
        select_range=(0, SPECTRAL_MODE_COUNT - 1),
        check_finite=False,
    )
    source = operator_row["source"][1:-1]
    overlaps = np.asarray(eigenvectors.T @ source, dtype=float)
    coefficients = overlaps / eigenvalues
    first_abs = float(abs(coefficients[0]))
    remainder_abs_sum = float(np.sum(np.abs(coefficients[1:])))
    return {
        "lambda_1": float(eigenvalues[0]),
        "lambda_2": float(eigenvalues[1]),
        "lambda_3": float(eigenvalues[2]),
        "first_mode_overlap": float(overlaps[0]),
        "second_mode_overlap": float(overlaps[1]),
        "first_mode_coefficient": float(coefficients[0]),
        "second_mode_coefficient": float(coefficients[1]),
        "principal_mode_abs_coefficient": first_abs,
        "remainder_mode_abs_coefficient_sum": remainder_abs_sum,
        "principal_mode_dominance_ratio": float(
            first_abs / max(remainder_abs_sum, 1.0e-30)
        ),
        "one_negative_mode_then_positive_gap_now": bool(
            float(eigenvalues[0]) < 0.0 < float(eigenvalues[1])
        ),
        "principal_mode_dominates_remainder_now": bool(first_abs > remainder_abs_sum),
    }


# 関数: Green-kernel followup 監査 pack を返す。

def build_trial2_beta_sensitivity_green_kernel_followup_pack() -> dict:
    """Return one audit pack for the transformed Green-kernel followup."""
    operator_rows = [
        build_dirichlet_operator_row(point_count)
        for point_count in WINDOW_POINT_COUNTS
    ]
    coarse_row = operator_rows[0]
    fine_row = operator_rows[-1]
    coarse_inverse_row = build_coarse_full_inverse_row(coarse_row)
    sampled_column_rows = [
        {
            "point_count": int(row["point_count"]),
            "columns": [
                build_green_column_row(row, probe_fraction)
                for probe_fraction in PROBE_FRACTIONS
            ],
        }
        for row in operator_rows
    ]
    source_solution_rows = [
        {
            "point_count": int(row["point_count"]),
            **build_source_solution_row(row),
        }
        for row in operator_rows
    ]
    spectral_row = build_spectral_projection_row(fine_row)

    probe_075_sign_mixed_all_resolutions = bool(
        all(
            next(
                column["column_sign_mixed_now"]
                for column in row["columns"]
                if abs(column["probe_fraction"] - 0.75) < 1.0e-12
            )
            for row in sampled_column_rows
        )
    )
    source_solution_all_negative_all_resolutions = bool(
        all(row["solution_all_negative_now"] for row in source_solution_rows)
    )

    exact_trial2_beta_sensitivity_green_kernel_available_now = True
    exact_trial2_beta_sensitivity_green_kernel_one_sign_available_now = bool(
        coarse_inverse_row["full_inverse_one_sign_now"]
        and not probe_075_sign_mixed_all_resolutions
    )
    exact_trial2_beta_sensitivity_source_weighted_resolvent_negative_now = bool(
        source_solution_all_negative_all_resolutions
    )
    exact_trial2_beta_sensitivity_single_negative_mode_dominance_support_available_now = bool(
        spectral_row["one_negative_mode_then_positive_gap_now"]
        and spectral_row["principal_mode_dominates_remainder_now"]
    )
    exact_trial2_beta_sensitivity_green_kernel_negative_closeout_available_now = bool(
        exact_trial2_beta_sensitivity_green_kernel_available_now
        and not exact_trial2_beta_sensitivity_green_kernel_one_sign_available_now
        and probe_075_sign_mixed_all_resolutions
        and exact_trial2_beta_sensitivity_source_weighted_resolvent_negative_now
    )
    updated_pack_trial2_beta_sensitivity_spectral_projection_followup_required_now = bool(
        exact_trial2_beta_sensitivity_green_kernel_negative_closeout_available_now
        and exact_trial2_beta_sensitivity_single_negative_mode_dominance_support_available_now
    )

    return {
        "beta_common_root": float(BETA_COMMON_ROOT),
        "canonical_window_x_min": float(WINDOW_X_MIN),
        "canonical_window_x_max": float(CANONICAL_WINDOW_X_MAX),
        "coarse_full_inverse_row": coarse_inverse_row,
        "sampled_column_rows": sampled_column_rows,
        "source_solution_rows": source_solution_rows,
        "spectral_projection_row": spectral_row,
        "probe_075_sign_mixed_all_resolutions": probe_075_sign_mixed_all_resolutions,
        "source_solution_all_negative_all_resolutions": (
            source_solution_all_negative_all_resolutions
        ),
        "exact_trial2_beta_sensitivity_green_kernel_available_now": (
            exact_trial2_beta_sensitivity_green_kernel_available_now
        ),
        "exact_trial2_beta_sensitivity_green_kernel_one_sign_available_now": (
            exact_trial2_beta_sensitivity_green_kernel_one_sign_available_now
        ),
        "exact_trial2_beta_sensitivity_source_weighted_resolvent_negative_now": (
            exact_trial2_beta_sensitivity_source_weighted_resolvent_negative_now
        ),
        "exact_trial2_beta_sensitivity_single_negative_mode_dominance_support_available_now": (
            exact_trial2_beta_sensitivity_single_negative_mode_dominance_support_available_now
        ),
        "exact_trial2_beta_sensitivity_green_kernel_negative_closeout_available_now": (
            exact_trial2_beta_sensitivity_green_kernel_negative_closeout_available_now
        ),
        "updated_pack_trial2_beta_sensitivity_spectral_projection_followup_required_now": (
            updated_pack_trial2_beta_sensitivity_spectral_projection_followup_required_now
        ),
        "exact_trial2_common_root_monotonicity_theorem_available_now": False,
    }


# 関数: backend 単体実行時に retained metrics を表示する。

def main() -> None:
    """Run the Green-kernel followup backend directly and print retained metrics."""
    pack = build_trial2_beta_sensitivity_green_kernel_followup_pack()
    print("[trial2-beta-green-kernel-followup]")
    print(f"beta_common_root = {pack['beta_common_root']:.16f}")
    coarse = pack["coarse_full_inverse_row"]
    print(
        "coarse_full_inverse_sign = "
        f"min {coarse['full_inverse_min']:.16f} "
        f"max {coarse['full_inverse_max']:.16f} "
        f"neg_frac {coarse['full_inverse_negative_fraction']:.6f} "
        f"pos_frac {coarse['full_inverse_positive_fraction']:.6f}"
    )
    for row in pack["sampled_column_rows"]:
        probe_075 = next(
            column for column in row["columns"] if abs(column["probe_fraction"] - 0.75) < 1.0e-12
        )
        print(
            "point_count="
            f"{row['point_count']} "
            f"probe075_min={probe_075['column_min']:.16f} "
            f"probe075_max={probe_075['column_max']:.16f} "
            f"mixed={probe_075['column_sign_mixed_now']}"
        )

    for row in pack["source_solution_rows"]:
        print(
            "source_solution_point_count="
            f"{row['point_count']} "
            f"min={row['solution_min']:.16f} "
            f"max={row['solution_max']:.16f} "
            f"all_negative={row['solution_all_negative_now']}"
        )

    spectral = pack["spectral_projection_row"]
    print(
        "spectral_gap = "
        f"lambda1 {spectral['lambda_1']:.16f} "
        f"lambda2 {spectral['lambda_2']:.16f} "
        f"dominance_ratio {spectral['principal_mode_dominance_ratio']:.16f}"
    )
    print(
        "green_kernel_one_sign_available_now = "
        f"{pack['exact_trial2_beta_sensitivity_green_kernel_one_sign_available_now']}"
    )
    print(
        "source_weighted_resolvent_negative_now = "
        f"{pack['exact_trial2_beta_sensitivity_source_weighted_resolvent_negative_now']}"
    )
    print(
        "spectral_projection_followup_required_now = "
        f"{pack['updated_pack_trial2_beta_sensitivity_spectral_projection_followup_required_now']}"
    )


if __name__ == "__main__":
    main()
