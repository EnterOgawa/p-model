#!/usr/bin/env python3
"""Audit continuum-support stability after the discrete spectral-projection theorem.

Purpose:
    Continue the strict-theorem route after `.5663-.5670`, where the full
    finite spectral decomposition already proved source-weighted negativity on
    the canonical Dirichlet window. The honest next question is narrower:

        does that discrete theorem survive continuum refinement in a way that
        clearly separates boundary-layer vanishing from genuine interior sign
        loss?

    This backend therefore does not replay the discrete theorem. It checks:

    1. whether the shrinking global pointwise margin is explained by the
       Dirichlet boundary layer, and
    2. whether fixed open interior windows retain positive pointwise dominance
       under refinement.

Inputs:
    - scripts/quantum/trial2_beta_sensitivity_green_kernel_followup_backend.py
    - scripts/quantum/trial2_beta_sensitivity_spectral_projection_followup_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5671-.5678` wrappers

Assumptions:
    - The canonical window remains [0.05, 20]
    - No new parameter is introduced
    - The branch targets continuum-support diagnostics only; it does not claim
      the final operator-level theorem
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
    build_dirichlet_operator_row,
)


CONTINUUM_POINT_COUNTS = (600, 1200, 1800, 2400)
INTERIOR_WINDOWS = (
    (0.10, 19.90),
    (0.20, 19.80),
    (0.50, 19.50),
    (1.00, 19.00),
)
BOUNDARY_LAYER_REL_SPREAD_TOL = 5.0e-3
INTERIOR_SUPPORT_REL_SPREAD_TOL = 5.0e-2


# 関数: one resolution の continuum-support row を返す。
def build_continuum_support_row(point_count: int) -> dict:
    """Return one continuum-support row for the canonical Dirichlet operator."""
    operator_row = build_dirichlet_operator_row(int(point_count))
    eigenvalues, eigenvectors = eigh_tridiagonal(
        operator_row["diagonal"],
        operator_row["off_diagonal"],
        check_finite=False,
    )
    source = np.asarray(operator_row["source"][1:-1], dtype=float)
    overlaps = np.asarray(eigenvectors.T @ source, dtype=float)
    coefficients = np.asarray(overlaps / eigenvalues, dtype=float)
    principal_component = np.asarray(coefficients[0] * eigenvectors[:, 0], dtype=float)
    remainder_abs_sum = np.asarray(
        np.sum(np.abs(eigenvectors[:, 1:]) * np.abs(coefficients[1:]), axis=1),
        dtype=float,
    )
    pointwise_margin = np.asarray(
        np.abs(principal_component) - remainder_abs_sum,
        dtype=float,
    )
    interior_grid = np.asarray(operator_row["grid"][1:-1], dtype=float)
    interior_window_rows = []
    for x_min, x_max in INTERIOR_WINDOWS:
        mask = (interior_grid >= float(x_min)) & (interior_grid <= float(x_max))
        window_margin_min = float(np.min(pointwise_margin[mask]))
        interior_window_rows.append(
            {
                "x_min": float(x_min),
                "x_max": float(x_max),
                "pointwise_margin_min": window_margin_min,
                "pointwise_margin_positive_now": bool(window_margin_min > 0.0),
            }
        )

    global_margin_min = float(np.min(pointwise_margin))
    global_margin_over_step = float(global_margin_min / operator_row["step"])
    return {
        "point_count": int(point_count),
        "step": float(operator_row["step"]),
        "lambda_1": float(eigenvalues[0]),
        "lambda_2": float(eigenvalues[1]),
        "global_pointwise_margin_min": global_margin_min,
        "global_pointwise_margin_over_step": global_margin_over_step,
        "interior_window_rows": interior_window_rows,
    }


# 関数: one fixed interior window の refinement summary を返す。

def build_interior_window_summary(rows: list[dict], x_min: float, x_max: float) -> dict:
    """Return one refinement summary for a fixed open interior window."""
    values = []
    steps = []
    for row in rows:
        window_row = next(
            item
            for item in row["interior_window_rows"]
            if abs(item["x_min"] - x_min) < 1.0e-12
            and abs(item["x_max"] - x_max) < 1.0e-12
        )
        values.append(float(window_row["pointwise_margin_min"]))
        steps.append(float(row["step"]))

    coarse_step = float(steps[1])
    fine_step = float(steps[-1])
    ratio = float(coarse_step / fine_step)
    coarse_value = float(values[1])
    fine_value = float(values[-1])
    continuum_estimate = float(
        fine_value + (fine_value - coarse_value) / (ratio * ratio - 1.0)
    )
    last_rel_spread = float(abs(values[-1] - values[-2]) / max(abs(values[-1]), 1.0e-30))
    return {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "margin_values": [float(value) for value in values],
        "margin_positive_all_rows_now": bool(all(value > 0.0 for value in values)),
        "last_rel_spread": last_rel_spread,
        "continuum_margin_estimate": continuum_estimate,
        "continuum_margin_positive_now": bool(continuum_estimate > 0.0),
        "refinement_stable_now": bool(last_rel_spread <= INTERIOR_SUPPORT_REL_SPREAD_TOL),
    }


# 関数: continuum-support followup 監査 pack を返す。

def build_trial2_beta_sensitivity_continuum_spectral_projection_followup_pack() -> dict:
    """Return one audit pack for the continuum spectral-projection followup."""
    rows = [
        build_continuum_support_row(point_count)
        for point_count in CONTINUUM_POINT_COUNTS
    ]
    interior_window_summaries = [
        build_interior_window_summary(rows, x_min, x_max)
        for x_min, x_max in INTERIOR_WINDOWS
    ]
    global_boundary_layer_values = [
        float(row["global_pointwise_margin_over_step"]) for row in rows
    ]
    boundary_layer_rel_spread = float(
        (max(global_boundary_layer_values) - min(global_boundary_layer_values))
        / max(abs(np.mean(global_boundary_layer_values)), 1.0e-30)
    )
    lambda_1_values = [float(row["lambda_1"]) for row in rows]
    lambda_2_values = [float(row["lambda_2"]) for row in rows]
    coarse_step = float(rows[1]["step"])
    fine_step = float(rows[-1]["step"])
    ratio = float(coarse_step / fine_step)
    lambda_1_continuum_estimate = float(
        lambda_1_values[-1]
        + (lambda_1_values[-1] - lambda_1_values[1]) / (ratio * ratio - 1.0)
    )
    lambda_2_continuum_estimate = float(
        lambda_2_values[-1]
        + (lambda_2_values[-1] - lambda_2_values[1]) / (ratio * ratio - 1.0)
    )
    exact_trial2_beta_sensitivity_continuum_boundary_layer_support_available_now = bool(
        boundary_layer_rel_spread <= BOUNDARY_LAYER_REL_SPREAD_TOL
    )
    exact_trial2_beta_sensitivity_continuum_gap_support_available_now = bool(
        all(value < 0.0 for value in lambda_1_values)
        and all(value > 0.0 for value in lambda_2_values)
        and lambda_1_continuum_estimate < 0.0
        and lambda_2_continuum_estimate > 0.0
    )
    exact_trial2_beta_sensitivity_continuum_open_interval_support_available_now = bool(
        exact_trial2_beta_sensitivity_continuum_boundary_layer_support_available_now
        and exact_trial2_beta_sensitivity_continuum_gap_support_available_now
        and all(
            summary["margin_positive_all_rows_now"]
            and summary["continuum_margin_positive_now"]
            and summary["refinement_stable_now"]
            for summary in interior_window_summaries
        )
    )
    exact_trial2_beta_sensitivity_operator_level_spectral_projection_theorem_available_now = (
        False
    )
    updated_pack_trial2_beta_sensitivity_operator_level_spectral_projection_followup_required_now = bool(
        exact_trial2_beta_sensitivity_continuum_open_interval_support_available_now
        and not exact_trial2_beta_sensitivity_operator_level_spectral_projection_theorem_available_now
    )
    return {
        "beta_common_root": float(BETA_COMMON_ROOT),
        "continuum_rows": rows,
        "interior_window_summaries": interior_window_summaries,
        "global_boundary_layer_values": global_boundary_layer_values,
        "boundary_layer_rel_spread": boundary_layer_rel_spread,
        "lambda_1_values": lambda_1_values,
        "lambda_2_values": lambda_2_values,
        "lambda_1_continuum_estimate": lambda_1_continuum_estimate,
        "lambda_2_continuum_estimate": lambda_2_continuum_estimate,
        "exact_trial2_beta_sensitivity_continuum_boundary_layer_support_available_now": (
            exact_trial2_beta_sensitivity_continuum_boundary_layer_support_available_now
        ),
        "exact_trial2_beta_sensitivity_continuum_gap_support_available_now": (
            exact_trial2_beta_sensitivity_continuum_gap_support_available_now
        ),
        "exact_trial2_beta_sensitivity_continuum_open_interval_support_available_now": (
            exact_trial2_beta_sensitivity_continuum_open_interval_support_available_now
        ),
        "exact_trial2_beta_sensitivity_operator_level_spectral_projection_theorem_available_now": (
            exact_trial2_beta_sensitivity_operator_level_spectral_projection_theorem_available_now
        ),
        "updated_pack_trial2_beta_sensitivity_operator_level_spectral_projection_followup_required_now": (
            updated_pack_trial2_beta_sensitivity_operator_level_spectral_projection_followup_required_now
        ),
    }


# 関数: backend 単体実行時に retained metrics を表示する。

def main() -> None:
    """Run the continuum-support backend directly and print retained metrics."""
    pack = build_trial2_beta_sensitivity_continuum_spectral_projection_followup_pack()
    print("[trial2-beta-continuum-spectral-projection-followup]")
    print(f"beta_common_root = {pack['beta_common_root']:.16f}")
    for row in pack["continuum_rows"]:
        print(
            "point_count="
            f"{row['point_count']} "
            f"lambda1={row['lambda_1']:.16f} "
            f"lambda2={row['lambda_2']:.16f} "
            f"global_margin_min={row['global_pointwise_margin_min']:.16f} "
            f"global_margin_over_step={row['global_pointwise_margin_over_step']:.16f}"
        )

    for summary in pack["interior_window_summaries"]:
        print(
            "interior_window="
            f"[{summary['x_min']:.2f}, {summary['x_max']:.2f}] "
            f"last_rel_spread={summary['last_rel_spread']:.16f} "
            f"continuum_margin_estimate={summary['continuum_margin_estimate']:.16f}"
        )

    print(
        "continuum_open_interval_support_available_now = "
        f"{pack['exact_trial2_beta_sensitivity_continuum_open_interval_support_available_now']}"
    )
    print(
        "operator_level_followup_required_now = "
        f"{pack['updated_pack_trial2_beta_sensitivity_operator_level_spectral_projection_followup_required_now']}"
    )


if __name__ == "__main__":
    main()
