#!/usr/bin/env python3
"""Build the retained scalar-proxy matching-scale redrive diagnostic pack.

Purpose:
    Reclassify the active blocker after the dense scalar-proxy alpha(q) audit.
    The helper combines the new retained crossing q_exact with the old
    projection-overlap support-band review and the coupled-tail q_star formula
    to quantify what must be redriven next.

Inputs:
    - output/public/quantum/q_8_7_56_5375_5378_updated_pack_scalar_proxy_alpha_q_curve_di_9aed6addb2_declaration_gate_metrics.json
    - output/public/quantum/mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_matching_scale_review_numeric_evaluation_metrics.json
    - output/public/quantum/mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_effective_support_scale_review_numeric_evaluation_metrics.json
    - output/public/quantum/mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_coupled_tail_reconciliation_review_numeric_evaluation_metrics.json

Outputs:
    - One in-memory diagnostic pack consumed by `.5383-.5390` wrappers

Assumptions:
    - alpha(q)=F(q)^2/(4*pi) already survived on the retained scalar profile
    - q_exact is already fixed from the dense alpha(q) curve
    - The current task is not another extra-q replay but one matching-law redrive
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
ALPHA_Q_CURVE_AUDIT = (
    PUBLIC_OUT
    / "q_8_7_56_5375_5378_updated_pack_scalar_proxy_alpha_q_curve_di_9aed6addb2_declaration_gate_metrics.json"
)
PROJECTION_MATCHING_REVIEW = (
    PUBLIC_OUT
    / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_matching_scale_review_numeric_evaluation_metrics.json"
)
PROJECTION_SCALE_REVIEW = (
    PUBLIC_OUT
    / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_effective_support_scale_review_numeric_evaluation_metrics.json"
)
COUPLED_TAIL_EVAL = (
    PUBLIC_OUT
    / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_coupled_tail_reconciliation_review_numeric_evaluation_metrics.json"
)


# Function: fail immediately when one required input is missing.
def require(path: Path) -> None:
    """Abort when one required path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: read one UTF-8 JSON artifact.

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON artifact into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


# Function: build the retained scalar-proxy matching-scale redrive pack.

def build_scalar_proxy_matching_scale_redrive_pack() -> dict:
    """Return the retained scalar-proxy matching-scale redrive diagnostic pack."""
    for path in (
        ALPHA_Q_CURVE_AUDIT,
        PROJECTION_MATCHING_REVIEW,
        PROJECTION_SCALE_REVIEW,
        COUPLED_TAIL_EVAL,
    ):
        require(path)

    alpha_q_curve_audit = read_json(ALPHA_Q_CURVE_AUDIT)
    projection_matching_review = read_json(PROJECTION_MATCHING_REVIEW)
    projection_scale_review = read_json(PROJECTION_SCALE_REVIEW)
    coupled_tail_eval = read_json(COUPLED_TAIL_EVAL)

    alpha_summary = alpha_q_curve_audit["summary"]
    matching_summary = projection_matching_review["summary"]
    scale_summary = projection_scale_review["summary"]
    scale_evidence = projection_scale_review["evidence"]
    coupled_tail_summary = coupled_tail_eval["summary"]

    q_exact = float(alpha_summary["primary_q_exact_over_m0"])
    q_star = float(alpha_summary["q_star_over_m0"])
    q_blind = float(alpha_summary["q_blind_over_m0"])
    beta1 = float(alpha_summary["beta1"])
    kappa_ratio = float(coupled_tail_summary["kappa_ratio"])

    q_correction_factor = float(q_exact / q_star)
    q_correction_delta = float(q_exact - q_star)
    q_correction_rel = float(q_correction_delta / q_star)
    kappa_redriven = float(q_exact**2)
    kappa_correction_factor = float(kappa_redriven / kappa_ratio)
    beta_effective_from_q_exact = float(math.sqrt(1.0 - q_exact**4))
    delta_beta_effective = float(beta_effective_from_q_exact - beta1)
    delta_beta_effective_rel = float(delta_beta_effective / beta1)

    q_exact_matches_prior_projection_crossing_abs_error = float(abs(q_exact - q_blind))
    q_exact_matches_prior_projection_crossing_now = (
        q_exact_matches_prior_projection_crossing_abs_error <= 1.0e-12
    )
    projection_overlap_support_band_prejustified_now = bool(
        matching_summary["finite_internal_scale_theory_side_justified"]
    )
    projection_overlap_exact_scale_open_current_canon_now = bool(
        scale_summary["candidate_ambiguity_significant"]
        and scale_evidence["current_public_nonuniqueness_surface_available"]
    )
    formula_survives_now = bool(
        alpha_summary["exact_scalar_proxy_alpha_q_curve_formula_available_now"]
        and not alpha_summary["exact_scalar_proxy_formula_failure_now"]
    )
    matching_scale_redrive_requires_new_law_now = bool(
        formula_survives_now
        and alpha_summary["exact_scalar_proxy_q_exact_unique_on_retained_interval_now"]
        and q_exact_matches_prior_projection_crossing_now
        and projection_overlap_support_band_prejustified_now
        and projection_overlap_exact_scale_open_current_canon_now
    )
    effective_beta_shift_secondary_only_now = bool(
        matching_scale_redrive_requires_new_law_now
        and abs(delta_beta_effective_rel) < abs(q_correction_rel)
    )

    return {
        "alpha_target": float(alpha_summary["alpha_target"]),
        "beta1": beta1,
        "q_exact_over_m0": q_exact,
        "q_star_over_m0": q_star,
        "q_blind_over_m0": q_blind,
        "q_correction_factor": q_correction_factor,
        "q_correction_delta_over_m0": q_correction_delta,
        "q_correction_rel": q_correction_rel,
        "kappa_ratio": kappa_ratio,
        "kappa_redriven": kappa_redriven,
        "kappa_correction_factor": kappa_correction_factor,
        "beta_effective_from_q_exact": beta_effective_from_q_exact,
        "delta_beta_effective": delta_beta_effective,
        "delta_beta_effective_rel": delta_beta_effective_rel,
        "alpha_at_q_star": float(alpha_summary["alpha_at_q_star"]),
        "relative_residual_at_q_star": float(alpha_summary["relative_residual_at_q_star"]),
        "q_exact_matches_prior_projection_crossing_abs_error": q_exact_matches_prior_projection_crossing_abs_error,
        "q_exact_matches_prior_projection_crossing_now": q_exact_matches_prior_projection_crossing_now,
        "projection_overlap_support_band_prejustified_now": projection_overlap_support_band_prejustified_now,
        "projection_overlap_exact_scale_open_current_canon_now": projection_overlap_exact_scale_open_current_canon_now,
        "formula_survives_now": formula_survives_now,
        "matching_scale_redrive_requires_new_law_now": matching_scale_redrive_requires_new_law_now,
        "effective_beta_shift_secondary_only_now": effective_beta_shift_secondary_only_now,
        "best_projection_scale_candidate_name": str(scale_summary["best_candidate_name"]),
        "best_projection_scale_candidate_error": float(scale_summary["best_candidate_error"]),
        "second_projection_scale_candidate_name": str(scale_summary["second_candidate_name"]),
        "second_projection_scale_candidate_error": float(scale_summary["second_candidate_error"]),
        "projection_scale_candidate_error_gap": float(scale_summary["candidate_error_gap"]),
        "projection_scale_candidate_error_spread": float(scale_summary["candidate_error_spread"]),
    }


# Function: allow one CLI smoke run for local verification.

def main() -> None:
    """Print the retained scalar-proxy matching-scale redrive pack as JSON."""
    print(json.dumps(build_scalar_proxy_matching_scale_redrive_pack(), ensure_ascii=False, indent=2))


# Function: run the helper when invoked as one CLI script.

if __name__ == "__main__":
    main()
