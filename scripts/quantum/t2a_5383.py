#!/usr/bin/env python3
"""Generate 8.7.56.5383-.5386 scalar-proxy matching-scale redrive artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.scalar_proxy_matching_scale_redrive_backend import (
    build_scalar_proxy_matching_scale_redrive_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5379-5382",
        "updated_pack_scalar_proxy_alpha_q_curve_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5383-5386"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor scalar-proxy "
    "matching-scale redrive audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_scalar_proxy_matching_scale_redrive_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_alpha_q_curve_audited_matching_scale_redrive_primary_"
    "source_materialization_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_matching_scale_redrive_diagnosed_matching_law_inventory_"
    "primary_effective_beta_shift_secondary_source_materialization_reserve_gate"
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


# Function: return formulas used by the redrive audit.

def build_formulae() -> dict[str, str]:
    """Return formulas used in the matching-scale redrive audit."""
    return {
        "old_q_star": "q_star/m0 = (1 - beta1^2)^(1/4)",
        "redrive_factor": "C_q = q_exact / q_star",
        "effective_beta": "beta_eff = sqrt(1 - q_exact^4)",
        "effective_beta_shift": "delta_beta = beta_eff - beta1",
    }


# Function: execute `.5383-.5386`.

def main() -> None:
    """Execute the scalar-proxy matching-scale redrive audit."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_scalar_proxy_matching_scale_redrive_pack()

    matching_scale_redrive_pack_available_now = bool(
        prior_summary["gate_a_updated_pack_exact_scalar_proxy_alpha_q_curve_diagnosis_available_now"]
        and pack["formula_survives_now"]
    )
    matching_scale_redrive_requires_new_law_now = bool(
        pack["matching_scale_redrive_requires_new_law_now"]
    )
    effective_beta_shift_secondary_only_now = bool(
        pack["effective_beta_shift_secondary_only_now"]
    )
    source_materialization_secondary_reserve_retained_now = bool(
        prior_summary["selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_demoted_to_secondary_now"]
    )
    same_schema_selected_extension_source_materialization_replay_selected_now = False

    rows = [
        sign_base.row(
            "exact_scalar_proxy_matching_scale_redrive_pack_available_now",
            "pass" if matching_scale_redrive_pack_available_now else "reject",
            "exact scalar-proxy matching-scale redrive pack available now",
            sign_base.truth(matching_scale_redrive_pack_available_now),
            "The current branch bundles the retained alpha(q) verdict, the old support-band review, and the coupled-tail q_star formula into one redrive pack.",
        ),
        sign_base.row(
            "scalar_proxy_q_exact_matches_prior_projection_crossing_now",
            "pass" if pack["q_exact_matches_prior_projection_crossing_now"] else "reject",
            "scalar-proxy q_exact matches prior projection crossing now",
            sign_base.truth(pack["q_exact_matches_prior_projection_crossing_now"]),
            "The new dense alpha(q) crossing is not a different numeric object; it reproduces the old blind projection-overlap crossing exactly.",
        ),
        sign_base.row(
            "scalar_proxy_projection_overlap_support_band_prejustified_now",
            "pass" if pack["projection_overlap_support_band_prejustified_now"] else "reject",
            "scalar-proxy projection-overlap support band prejustified now",
            sign_base.truth(pack["projection_overlap_support_band_prejustified_now"]),
            "The old support-band branch already justified a finite matching-scale band, so the current blocker is not whether a finite scale exists.",
        ),
        sign_base.row(
            "scalar_proxy_projection_overlap_exact_scale_open_current_canon_now",
            "pass" if pack["projection_overlap_exact_scale_open_current_canon_now"] else "reject",
            "scalar-proxy projection-overlap exact scale open under current canon now",
            sign_base.truth(pack["projection_overlap_exact_scale_open_current_canon_now"]),
            "The old support-band review also fixed that current canon still does not choose one exact support scale by itself.",
        ),
        sign_base.row(
            "scalar_proxy_matching_scale_correction_factor_fixed",
            "pass",
            "scalar-proxy matching-scale correction factor fixed",
            pack["q_correction_factor"],
            "The redrive factor C_q=q_exact/q_star quantifies how much the current coupled-tail matching scale must move to hit the retained scalar-proxy crossing.",
        ),
        sign_base.row(
            "scalar_proxy_matching_scale_delta_q_over_q_star_fixed",
            "pass",
            "scalar-proxy matching-scale delta q over q_star fixed",
            pack["q_correction_rel"],
            "The retained mismatch is order-percent in q, which is large enough to treat matching-scale redrive as the primary blocker.",
        ),
        sign_base.row(
            "scalar_proxy_matching_scale_kappa_correction_factor_fixed",
            "pass",
            "scalar-proxy matching-scale kappa correction factor fixed",
            pack["kappa_correction_factor"],
            "Because q^2 enters the coupled-tail kappa ratio, the required redrive also has an explicit kappa-space factor.",
        ),
        sign_base.row(
            "scalar_proxy_effective_beta_from_q_exact_fixed",
            "pass",
            "scalar-proxy effective beta from q_exact fixed",
            pack["beta_effective_from_q_exact"],
            "An equivalent beta_eff exists algebraically, but it is only the beta value that reproduces q_exact after the fact.",
        ),
        sign_base.row(
            "scalar_proxy_effective_beta_shift_relative_fixed",
            "pass",
            "scalar-proxy effective beta shift relative fixed",
            pack["delta_beta_effective_rel"],
            "The equivalent beta shift is tiny compared with the q mismatch, so beta-shift rereading is secondary rather than primary.",
        ),
        sign_base.row(
            "scalar_proxy_matching_scale_redrive_requires_new_law_now",
            "pass" if matching_scale_redrive_requires_new_law_now else "reject",
            "scalar-proxy matching-scale redrive requires new law now",
            sign_base.truth(matching_scale_redrive_requires_new_law_now),
            "Given q_exact, the prejustified support band, and the still-open exact support-scale selection, the current blocker is one new matching law, not one more rerun.",
        ),
        sign_base.row(
            "scalar_proxy_effective_beta_shift_secondary_only_now",
            "pass" if effective_beta_shift_secondary_only_now else "reject",
            "scalar-proxy effective beta shift secondary only now",
            sign_base.truth(effective_beta_shift_secondary_only_now),
            "The effective beta reexpression remains a useful sensitivity proxy, but it does not replace the need for a matching-law redrive.",
        ),
        sign_base.row(
            "selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now",
            "pass" if source_materialization_secondary_reserve_retained_now else "reject",
            "selected-extension independent extra-q-range source-materialization secondary reserve retained now",
            sign_base.truth(source_materialization_secondary_reserve_retained_now),
            "The old extra-q source-materialization replay stays demoted because the higher-value blocker is still inside the scalar proxy.",
        ),
        sign_base.row(
            "same_schema_selected_extension_source_materialization_replay_selected_now",
            "reject",
            "same-schema selected-extension source-materialization replay selected now",
            0.0,
            "False means this branch did not burn another turn on the already-demoted extra-q source-materialization replay.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "alpha_target": float(pack["alpha_target"]),
        "beta1": float(pack["beta1"]),
        "q_exact_over_m0": float(pack["q_exact_over_m0"]),
        "q_star_over_m0": float(pack["q_star_over_m0"]),
        "q_blind_over_m0": float(pack["q_blind_over_m0"]),
        "q_correction_factor": float(pack["q_correction_factor"]),
        "q_correction_delta_over_m0": float(pack["q_correction_delta_over_m0"]),
        "q_correction_rel": float(pack["q_correction_rel"]),
        "kappa_ratio": float(pack["kappa_ratio"]),
        "kappa_redriven": float(pack["kappa_redriven"]),
        "kappa_correction_factor": float(pack["kappa_correction_factor"]),
        "beta_effective_from_q_exact": float(pack["beta_effective_from_q_exact"]),
        "delta_beta_effective": float(pack["delta_beta_effective"]),
        "delta_beta_effective_rel": float(pack["delta_beta_effective_rel"]),
        "alpha_at_q_star": float(pack["alpha_at_q_star"]),
        "relative_residual_at_q_star": float(pack["relative_residual_at_q_star"]),
        "q_exact_matches_prior_projection_crossing_abs_error": float(
            pack["q_exact_matches_prior_projection_crossing_abs_error"]
        ),
        "projection_overlap_support_band_prejustified_now": bool(
            pack["projection_overlap_support_band_prejustified_now"]
        ),
        "projection_overlap_exact_scale_open_current_canon_now": bool(
            pack["projection_overlap_exact_scale_open_current_canon_now"]
        ),
        "best_projection_scale_candidate_name": str(pack["best_projection_scale_candidate_name"]),
        "best_projection_scale_candidate_error": float(pack["best_projection_scale_candidate_error"]),
        "second_projection_scale_candidate_name": str(pack["second_projection_scale_candidate_name"]),
        "second_projection_scale_candidate_error": float(pack["second_projection_scale_candidate_error"]),
        "projection_scale_candidate_error_gap": float(pack["projection_scale_candidate_error_gap"]),
        "projection_scale_candidate_error_spread": float(pack["projection_scale_candidate_error_spread"]),
        "exact_scalar_proxy_matching_scale_redrive_pack_available_now": matching_scale_redrive_pack_available_now,
        "scalar_proxy_matching_scale_redrive_requires_new_law_now": matching_scale_redrive_requires_new_law_now,
        "scalar_proxy_effective_beta_shift_secondary_only_now": effective_beta_shift_secondary_only_now,
        "selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now": source_materialization_secondary_reserve_retained_now,
        "same_schema_selected_extension_source_materialization_replay_selected_now": same_schema_selected_extension_source_materialization_replay_selected_now,
        "selected_primary_completion_lane": "updated_pack_scalar_proxy_matching_law_inventory_audit",
        "selected_secondary_completion_lane": "updated_pack_scalar_proxy_effective_beta_shift_sensitivity_review",
        "selected_reserve_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun",
        "selected_next_generation_route": "trial2_numeric_alpha_scalar_proxy_matching_scale_redrive_gate",
        "recommended_next_route_or_none": "8.7.56.5387",
        "selected_followup_route": "trial2_numeric_alpha_scalar_proxy_matching_law_inventory_audit",
        "selected_followup_route_or_none": "8.7.56.5391",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5385",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5387",
                "followup_route": "8.7.56.5391",
            },
        },
        rows,
        summary,
        {
            "overall_status": "scalar_proxy_matching_scale_redrive_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} scalar-proxy matching-scale redrive completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# Function: run the audit when invoked as one CLI script.

if __name__ == "__main__":
    main()
