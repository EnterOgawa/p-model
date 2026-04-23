#!/usr/bin/env python3
"""Compute dense scalar-proxy alpha(q) diagnostics on the retained Q-ball profile.

Purpose:
    Run the computation gate recommended by expert advice after repeated
    theory-extension branches reproduced the same failure surface. The helper
    evaluates the retained scalar-proxy curve

        F(q) = int dr rho(r) r^2 sinc(q r) / int dr rho(r) r^2
        alpha(q) = F(q)^2 / (4 pi)

    and compares the target-crossing scale q_exact against the retained theory
    scale q_star.

Inputs:
    - output/public/quantum/mass_origin_qball_charge_mapping_branch_refresh_metrics.json
    - output/public/quantum/mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_numeric_evaluation_metrics.json
    - output/public/quantum/mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_coupled_tail_reconciliation_review_numeric_evaluation_metrics.json
    - scripts/quantum/mass_origin_qball_charge_mapping_branch.py

Outputs:
    - One in-memory dense curve pack consumed by `.5375-.5382` wrappers

Assumptions:
    - The retained scalar ground-state profile remains the scalar-proxy input
    - No new parameter is introduced
    - The formula under audit is alpha(q) = F(q)^2 / (4 pi)
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
QBALL_BRANCH_REFRESH = PUBLIC_OUT / "mass_origin_qball_charge_mapping_branch_refresh_metrics.json"
PROJECTION_OVERLAP_EVAL = (
    PUBLIC_OUT
    / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_numeric_evaluation_metrics.json"
)
COUPLED_TAIL_EVAL = (
    PUBLIC_OUT
    / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_coupled_tail_reconciliation_review_numeric_evaluation_metrics.json"
)
QBALL_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_qball_charge_mapping_branch.py"

ALPHA_TARGET = 1.0 / 137.035999084
FOUR_PI = 4.0 * math.pi


# Function: fail immediately when one required input is missing.
def require(path: Path) -> None:
    """Abort when one required path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: read one UTF-8 JSON artifact.

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON artifact into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


# Function: load the retained scalar Q-ball solver module.

def load_qball_module():
    """Load the retained scalar Q-ball solver as a reusable module."""
    spec = importlib.util.spec_from_file_location("wavep_qball_charge_mapping", QBALL_SOLVER)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to load module from {QBALL_SOLVER}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Function: extract the retained scalar ground-state row.

def extract_scalar_ground_state(qball_branch_refresh: dict) -> dict:
    """Extract the scalar ground-state row from retained branch-refresh metrics."""
    for row_data in qball_branch_refresh["evidence"]["discrete_mode_rows"]:
        if int(row_data["mode_index"]) == 1:
            return {
                "mode_index": int(row_data["mode_index"]),
                "beta_n": float(row_data["beta_n"]),
                "charge_proxy": float(row_data["charge_proxy"]),
                "energy_proxy": float(row_data["energy_proxy"]),
                "central_amplitude": float(row_data["central_amplitude"]),
                "mass_ratio_to_first": float(row_data["mass_ratio_to_first"]),
            }

    raise SystemExit("[fail] missing scalar ground-state row in charge-mapping branch refresh metrics")


# Function: evaluate the retained spherical form factor at one q/m0 value.

def form_factor(radius: np.ndarray, weight: np.ndarray, norm: float, q_ratio: float) -> float:
    """Evaluate one normalized spherical form factor on the retained profile."""
    qx = float(q_ratio) * radius
    sinc = np.ones_like(qx)
    mask = np.abs(qx) > 1.0e-12
    sinc[mask] = np.sin(qx[mask]) / qx[mask]
    numerator = np.trapezoid(weight * sinc, radius)
    return float(numerator / norm)


# Function: convert one form-factor value into the audited scalar-proxy alpha.

def alpha_from_form_factor(form_factor_value: float) -> float:
    """Return alpha(q) = F(q)^2 / (4 pi)."""
    return float((float(form_factor_value) ** 2) / FOUR_PI)


# Function: remove near-duplicate roots while preserving sorted order.

def deduplicate_roots(values: list[float], tolerance: float = 1.0e-12) -> list[float]:
    """Return one sorted list of unique roots up to the requested tolerance."""
    unique_values: list[float] = []
    for candidate in sorted(float(value) for value in values):
        if not unique_values or abs(candidate - unique_values[-1]) > tolerance:
            unique_values.append(candidate)

    return unique_values


# Function: locate every retained-interval root of alpha(q) - alpha_target.

def find_alpha_target_roots(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    q_values: np.ndarray,
    alpha_curve: np.ndarray,
    alpha_target: float,
) -> list[float]:
    """Locate every target-crossing root on the retained audit interval."""
    diff = alpha_curve - float(alpha_target)
    roots: list[float] = []
    for index in range(len(q_values) - 1):
        left_q = float(q_values[index])
        right_q = float(q_values[index + 1])
        left_diff = float(diff[index])
        right_diff = float(diff[index + 1])

        if abs(left_diff) <= 1.0e-14:
            roots.append(left_q)

        if left_diff * right_diff < 0.0:
            root = brentq(
                lambda q: alpha_from_form_factor(form_factor(radius, weight, norm, float(q))) - float(alpha_target),
                left_q,
                right_q,
            )
            roots.append(float(root))

    if abs(float(diff[-1])) <= 1.0e-14:
        roots.append(float(q_values[-1]))

    return deduplicate_roots(roots)


# Function: evaluate a compact set of sample checkpoints for the dense curve.

def build_curve_samples(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    q_values: list[float],
) -> list[dict[str, float]]:
    """Return one compact checkpoint table for the retained scalar-proxy curve."""
    samples: list[dict[str, float]] = []
    for q_value in sorted(q_values):
        form_factor_value = form_factor(radius, weight, norm, float(q_value))
        samples.append(
            {
                "q_over_m0": float(q_value),
                "F_q": float(form_factor_value),
                "alpha_q": alpha_from_form_factor(form_factor_value),
            }
        )

    return samples


# Function: build the dense scalar-proxy alpha(q) diagnostic pack.

def build_scalar_proxy_alpha_q_curve_pack(
    q_min: float = 0.0,
    q_max: float = 1.0,
    q_count: int = 10001,
) -> dict:
    """Return the dense retained scalar-proxy alpha(q) diagnostic pack."""
    for path in (QBALL_BRANCH_REFRESH, PROJECTION_OVERLAP_EVAL, COUPLED_TAIL_EVAL, QBALL_SOLVER):
        require(path)

    qball_branch_refresh = read_json(QBALL_BRANCH_REFRESH)
    projection_overlap_eval = read_json(PROJECTION_OVERLAP_EVAL)
    coupled_tail_eval = read_json(COUPLED_TAIL_EVAL)

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

    q_values = np.linspace(float(q_min), float(q_max), int(q_count), dtype=float)
    form_factor_curve = np.array(
        [form_factor(radius, weight, norm, float(q_value)) for q_value in q_values],
        dtype=float,
    )
    alpha_curve = np.square(form_factor_curve) / FOUR_PI
    q_exact_list = find_alpha_target_roots(
        radius,
        weight,
        norm,
        q_values,
        alpha_curve,
        ALPHA_TARGET,
    )

    q_star = float(coupled_tail_eval["summary"]["q_theory_over_m0"])
    q_blind = float(projection_overlap_eval["summary"]["first_target_matching_q_over_m0"])
    form_factor_at_q_star = form_factor(radius, weight, norm, q_star)
    alpha_at_q_star = alpha_from_form_factor(form_factor_at_q_star)
    relative_residual_at_q_star = abs(alpha_at_q_star - ALPHA_TARGET) / ALPHA_TARGET

    alpha_max_index = int(np.argmax(alpha_curve))
    alpha_max = float(alpha_curve[alpha_max_index])
    q_at_alpha_max = float(q_values[alpha_max_index])

    primary_q_exact = float(q_exact_list[0]) if q_exact_list else math.nan
    delta_q = float(primary_q_exact - q_star) if q_exact_list else math.nan
    delta_q_over_q_star = float(delta_q / q_star) if q_exact_list else math.nan

    if not q_exact_list:
        case_label = "case_b_formula_failure"
    elif len(q_exact_list) > 1:
        case_label = "case_c_multiple_target_crossings"
    elif abs(delta_q_over_q_star) >= 3.0e-3:
        case_label = "case_a1_order_percent_matching_scale_correction"
    else:
        case_label = "case_a2_subpercent_matching_scale_correction"

    sample_q_values = [
        0.0,
        0.05,
        0.10,
        0.15,
        0.20,
        q_blind,
        q_star,
        0.25,
        0.30,
        0.50,
        1.0,
    ]
    if q_exact_list:
        sample_q_values.extend(q_exact_list)

    curve_samples = build_curve_samples(
        radius,
        weight,
        norm,
        deduplicate_roots(sample_q_values, tolerance=1.0e-10),
    )

    return {
        "alpha_target": float(ALPHA_TARGET),
        "scalar_ground_state": scalar_ground_state,
        "q_values": q_values,
        "form_factor_curve": form_factor_curve,
        "alpha_curve": alpha_curve,
        "curve_samples": curve_samples,
        "q_min_over_m0": float(q_values[0]),
        "q_max_over_m0": float(q_values[-1]),
        "q_count": int(len(q_values)),
        "q_exact_list": [float(value) for value in q_exact_list],
        "q_exact_exists_now": bool(q_exact_list),
        "q_exact_unique_now": len(q_exact_list) == 1,
        "q_blind_over_m0": q_blind,
        "q_star_over_m0": q_star,
        "F_at_q_star": float(form_factor_at_q_star),
        "alpha_at_q_star": float(alpha_at_q_star),
        "relative_residual_at_q_star": float(relative_residual_at_q_star),
        "alpha_max": alpha_max,
        "q_at_alpha_max": q_at_alpha_max,
        "alpha_max_over_target": float(alpha_max / ALPHA_TARGET),
        "primary_q_exact_over_m0": primary_q_exact,
        "delta_q_over_m0": delta_q,
        "delta_q_over_q_star": delta_q_over_q_star,
        "q_exact_matches_prior_blind_crossing_abs_error": (
            float(abs(primary_q_exact - q_blind)) if q_exact_list else math.nan
        ),
        "formula_failure_now": not q_exact_list,
        "matching_scale_primary_now": bool(q_exact_list),
        "case_label": case_label,
    }


# Function: print one compact JSON summary when the helper is run directly.

def main() -> None:
    """Run the helper directly and print one compact JSON summary."""
    pack = build_scalar_proxy_alpha_q_curve_pack()
    summary = {
        "beta1": pack["scalar_ground_state"]["beta_n"],
        "q_star_over_m0": pack["q_star_over_m0"],
        "q_exact_list": pack["q_exact_list"],
        "delta_q_over_q_star": pack["delta_q_over_q_star"],
        "alpha_at_q_star": pack["alpha_at_q_star"],
        "relative_residual_at_q_star": pack["relative_residual_at_q_star"],
        "alpha_max": pack["alpha_max"],
        "alpha_max_over_target": pack["alpha_max_over_target"],
        "case_label": pack["case_label"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# Function: run the helper when invoked as one CLI script.

if __name__ == "__main__":
    main()
