#!/usr/bin/env python3
"""Generate 8.7.56.5391-.5394 scalar-proxy matching-law inventory artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.scalar_proxy_matching_law_inventory_backend import (
    build_scalar_proxy_matching_law_inventory_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5387-5390",
        "updated_pack_scalar_proxy_matching_scale_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5391-5394"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor scalar-proxy "
    "matching-law inventory audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_scalar_proxy_matching_law_inventory_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_matching_scale_redrive_audited_matching_law_inventory_"
    "primary_effective_beta_shift_secondary_source_materialization_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_matching_law_inventory_diagnosed_profile_sensitive_q_star_"
    "correction_primary_effective_beta_shift_secondary_"
    "source_materialization_reserve_gate"
)


# Function: write one metrics payload as JSON and CSV.
def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one rows CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "status", "metric", "value", "note"],
        )
        writer.writeheader()
        writer.writerows(data["rows"])

    return {
        "json": sign_base.display_path(paths["json"]),
        "csv": sign_base.display_path(paths["csv"]),
    }


# Function: return formulas used by the matching-law inventory.

def build_formulae() -> dict[str, str]:
    """Return formulas used by the matching-law inventory."""
    return {
        "stationary_candidate": "Candidate 1: F'(q_exact) = 0",
        "support_phase_family": "Candidate 3/4 support phase: q = phase_ref / r_ref",
        "centroid_family": "Candidate 4 centroid: q_cent = <q_local(r)>_(rho r^2)",
        "correction_family": "Candidate 5: q = q_star * (1 + c1 * (1 - beta1^2))",
        "evanescent_shift": "delta_kappa^2 = q_exact^2 - q_star^2",
    }


# Function: execute `.5391-.5394`.

def main() -> None:
    """Execute the scalar-proxy matching-law inventory audit."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_scalar_proxy_matching_law_inventory_pack()

    matching_law_inventory_available_now = bool(
        prior_summary["gate_a_updated_pack_exact_scalar_proxy_matching_scale_redrive_available_now"]
        and pack["blind_overlap_numeric_bridge_retained_now"]
    )
    profile_sensitive_q_star_correction_front_runner_now = bool(
        pack["matching_law_inventory_requires_profile_sensitive_completion_now"]
    )
    effective_beta_shift_secondary_only_now = bool(
        prior_summary["scalar_proxy_effective_beta_shift_secondary_only_now"]
    )
    source_materialization_secondary_reserve_retained_now = bool(
        prior_summary["selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now"]
    )

    rows = [
        sign_base.row(
            "exact_scalar_proxy_matching_law_inventory_available_now",
            "pass" if matching_law_inventory_available_now else "reject",
            "exact scalar-proxy matching-law inventory available now",
            sign_base.truth(matching_law_inventory_available_now),
            "The retained scalar profile, q_exact verdict, and old projection-overlap branch now support one explicit matching-law candidate inventory.",
        ),
        sign_base.row(
            "scalar_proxy_stationary_candidate_supported_now",
            "pass" if pack["stationary_candidate_supported_now"] else "reject",
            "scalar-proxy stationary candidate supported now",
            sign_base.truth(pack["stationary_candidate_supported_now"]),
            "Candidate 1 would require F'(q_exact) to vanish or nearly vanish. The retained profile now checks that directly.",
        ),
        sign_base.row(
            "scalar_proxy_stationary_log_slope_q_exact_abs_fixed",
            "pass",
            "scalar-proxy stationary log slope at q_exact absolute fixed",
            pack["F_log_slope_q_exact_abs"],
            "The normalized local slope |q_exact F'(q_exact) / F(q_exact)| quantifies how far q_exact is from any stationary-point interpretation.",
        ),
        sign_base.row(
            "scalar_proxy_overlap_consistency_tautology_rejected_now",
            "pass" if pack["overlap_consistency_tautology_rejected_now"] else "reject",
            "scalar-proxy overlap consistency tautology rejected now",
            sign_base.truth(pack["overlap_consistency_tautology_rejected_now"]),
            "The naive overlap-consistency identity is only a rewriting of F(q) and does not determine q by itself, so it cannot count as an honest matching law.",
        ),
        sign_base.row(
            "scalar_proxy_support_phase_family_best_rel_error_fixed",
            "pass",
            "scalar-proxy support-phase family best relative error fixed",
            pack["legacy_support_phase_best_rel_error"],
            "The inherited half-mass / mean-radius / rms-radius phase laws remain available, but the best one still misses q_exact at the several-percent level.",
        ),
        sign_base.row(
            "scalar_proxy_centroid_family_best_rel_error_fixed",
            "pass",
            "scalar-proxy centroid family best relative error fixed",
            pack["centroid_best_rel_error"],
            "Simple centroid laws built from 1/r, local log-derivative, and local kappa are much too coarse to explain q_exact directly.",
        ),
        sign_base.row(
            "scalar_proxy_blind_overlap_numeric_bridge_retained_now",
            "pass" if pack["blind_overlap_numeric_bridge_retained_now"] else "reject",
            "scalar-proxy blind-overlap numeric bridge retained now",
            sign_base.truth(pack["blind_overlap_numeric_bridge_retained_now"]),
            "The dense alpha(q) crossing still matches the old blind projection-overlap crossing to machine precision, so that numeric bridge remains physically informative.",
        ),
        sign_base.row(
            "scalar_proxy_profile_sensitive_q_star_correction_family_available_now",
            "pass" if pack["profile_sensitive_q_star_correction_family_available_now"] else "reject",
            "scalar-proxy profile-sensitive q_star correction family available now",
            sign_base.truth(pack["profile_sensitive_q_star_correction_family_available_now"]),
            "A one-parameter correction family around q_star can reproduce q_exact exactly and therefore deserves the next theorem-side audit.",
        ),
        sign_base.row(
            "scalar_proxy_q_star_correction_c1_fit_fixed",
            "pass",
            "scalar-proxy q_star correction c1 fit fixed",
            pack["q_star_correction_c1_fit"],
            "The fitted O(1) coefficient c1 tells how large the leading profile-sensitive correction to q_star must be if this family is the right law class.",
        ),
        sign_base.row(
            "scalar_proxy_delta_kappa_squared_rel_fixed",
            "pass",
            "scalar-proxy delta kappa squared relative fixed",
            pack["delta_kappa_squared_rel"],
            "The same mismatch can be read as a roughly one-percent correction in kappa^2 space, which is consistent with matching-law redrive rather than formula failure.",
        ),
        sign_base.row(
            "scalar_proxy_exact_matching_law_closed_form_available_now",
            "pass" if pack["exact_matching_law_closed_form_available_now"] else "reject",
            "scalar-proxy exact matching-law closed form available now",
            sign_base.truth(pack["exact_matching_law_closed_form_available_now"]),
            "No alpha-target-free closed-form law has been derived yet; the current branch only identifies the best completion family.",
        ),
        sign_base.row(
            "scalar_proxy_profile_sensitive_q_star_correction_front_runner_now",
            "pass" if profile_sensitive_q_star_correction_front_runner_now else "reject",
            "scalar-proxy profile-sensitive q_star correction front-runner now",
            sign_base.truth(profile_sensitive_q_star_correction_front_runner_now),
            "Among the current candidates, the only family that stays consistent with the retained hierarchy is the profile-sensitive correction family around q_star.",
        ),
        sign_base.row(
            "scalar_proxy_effective_beta_shift_secondary_only_now",
            "pass" if effective_beta_shift_secondary_only_now else "reject",
            "scalar-proxy effective beta shift secondary only now",
            sign_base.truth(effective_beta_shift_secondary_only_now),
            "beta_eff remains a secondary sensitivity reparameterization, not the primary law that determines q_exact.",
        ),
        sign_base.row(
            "selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now",
            "pass" if source_materialization_secondary_reserve_retained_now else "reject",
            "selected-extension independent extra-q-range source-materialization secondary reserve retained now",
            sign_base.truth(source_materialization_secondary_reserve_retained_now),
            "The old extra-q branch stays reserve-only because the retained scalar proxy still offers the higher-value path to closure.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_exact_over_m0": float(pack["q_exact_over_m0"]),
        "q_star_over_m0": float(pack["q_star_over_m0"]),
        "q_blind_over_m0": float(pack["q_blind_over_m0"]),
        "beta1": float(pack["beta1"]),
        "epsilon_beta": float(pack["epsilon_beta"]),
        "F_q_exact": float(pack["F_q_exact"]),
        "F_prime_q_exact": float(pack["F_prime_q_exact"]),
        "F_double_prime_q_exact": float(pack["F_double_prime_q_exact"]),
        "F_log_slope_q_exact_abs": float(pack["F_log_slope_q_exact_abs"]),
        "stationary_candidate_supported_now": bool(pack["stationary_candidate_supported_now"]),
        "legacy_support_phase_best_name": str(pack["legacy_support_phase_best_name"]),
        "legacy_support_phase_best_value": float(pack["legacy_support_phase_best_value"]),
        "legacy_support_phase_best_rel_error": float(pack["legacy_support_phase_best_rel_error"]),
        "centroid_best_name": str(pack["centroid_best_name"]),
        "centroid_best_value": float(pack["centroid_best_value"]),
        "centroid_best_rel_error": float(pack["centroid_best_rel_error"]),
        "q_star_correction_c1_fit": float(pack["q_star_correction_c1_fit"]),
        "q_star_correction_family_o1_now": bool(pack["q_star_correction_family_o1_now"]),
        "q_star_correction_reconstructed_q_over_m0": float(pack["q_star_correction_reconstructed_q_over_m0"]),
        "q_star_correction_reconstructed_abs_error": float(pack["q_star_correction_reconstructed_abs_error"]),
        "delta_kappa_squared": float(pack["delta_kappa_squared"]),
        "delta_kappa_squared_rel": float(pack["delta_kappa_squared_rel"]),
        "blind_overlap_numeric_bridge_retained_now": bool(pack["blind_overlap_numeric_bridge_retained_now"]),
        "overlap_consistency_tautology_rejected_now": bool(pack["overlap_consistency_tautology_rejected_now"]),
        "exact_scalar_proxy_matching_law_inventory_available_now": matching_law_inventory_available_now,
        "exact_matching_law_closed_form_available_now": bool(pack["exact_matching_law_closed_form_available_now"]),
        "profile_sensitive_q_star_correction_family_available_now": bool(
            pack["profile_sensitive_q_star_correction_family_available_now"]
        ),
        "profile_sensitive_q_star_correction_front_runner_now": profile_sensitive_q_star_correction_front_runner_now,
        "matching_law_inventory_front_runner_name": str(pack["matching_law_inventory_front_runner_name"]),
        "matching_law_inventory_requires_profile_sensitive_completion_now": bool(
            pack["matching_law_inventory_requires_profile_sensitive_completion_now"]
        ),
        "scalar_proxy_effective_beta_shift_secondary_only_now": effective_beta_shift_secondary_only_now,
        "selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now": source_materialization_secondary_reserve_retained_now,
        "selected_primary_completion_lane": "updated_pack_scalar_proxy_profile_sensitive_q_star_correction_audit",
        "selected_secondary_completion_lane": "updated_pack_scalar_proxy_effective_beta_shift_sensitivity_review",
        "selected_reserve_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun",
        "selected_next_generation_route": "trial2_numeric_alpha_scalar_proxy_matching_law_gate",
        "recommended_next_route_or_none": "8.7.56.5395",
        "selected_followup_route": "trial2_numeric_alpha_scalar_proxy_profile_sensitive_q_star_correction_audit",
        "selected_followup_route_or_none": "8.7.56.5399",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5393",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5395",
                "followup_route": "8.7.56.5399",
            },
        },
        rows,
        summary,
        {
            "overall_status": "scalar_proxy_matching_law_inventory_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} scalar-proxy matching-law inventory completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# Function: run the audit when invoked as one CLI script.

if __name__ == "__main__":
    main()
