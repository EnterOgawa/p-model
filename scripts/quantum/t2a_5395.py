#!/usr/bin/env python3
"""Generate 8.7.56.5395-.5398 scalar-proxy matching-law gate artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5391-5394",
        "updated_pack_scalar_proxy_matching_law_inventory_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5395-5398"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor scalar-proxy "
    "matching-law gate / effective-beta-shift refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_scalar_proxy_matching_law_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_matching_law_inventory_diagnosed_profile_sensitive_q_star_"
    "correction_primary_effective_beta_shift_secondary_"
    "source_materialization_reserve_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_matching_law_inventory_audited_profile_sensitive_q_star_"
    "correction_primary_effective_beta_shift_secondary_"
    "source_materialization_reserve_next"
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

    return {"json": sign_base.display_path(paths["json"])}


# Function: return formulas used by the gate refresh.

def build_formulae() -> dict[str, str]:
    """Return formulas used by the matching-law gate refresh."""
    return {
        "gate_a": "Gate A = scalar-proxy matching-law inventory available now",
        "gate_b": "Gate B = profile-sensitive q_star correction audit promoted next",
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# Function: execute `.5395-.5398`.

def main() -> None:
    """Execute the scalar-proxy matching-law gate / refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary["exact_scalar_proxy_matching_law_inventory_available_now"]
        and prior_summary["profile_sensitive_q_star_correction_front_runner_now"]
    )
    gate_b = bool(gate_a)
    gate_c = False
    exact_matching_law_closed_form_available_now = bool(
        prior_summary["exact_matching_law_closed_form_available_now"]
    )
    effective_beta_shift_secondary_only_now = bool(
        prior_summary["scalar_proxy_effective_beta_shift_secondary_only_now"]
    )
    source_materialization_secondary_reserve_retained_now = bool(
        prior_summary["selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now"]
    )
    pack_update_required_now = bool(gate_b)

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_scalar_proxy_matching_law_inventory_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack exact scalar-proxy matching-law inventory available now",
            sign_base.truth(gate_a),
            "The retained scalar profile now has one explicit matching-law inventory with a ranked front-runner family.",
        ),
        sign_base.row(
            "gate_b_updated_pack_scalar_proxy_profile_sensitive_q_star_correction_promoted_next",
            "pass" if gate_b else "reject",
            "gate B updated-pack scalar-proxy profile-sensitive q_star correction promoted next",
            sign_base.truth(gate_b),
            "The next honest blocker is to close one profile-sensitive correction law around q_star rather than reopen extension replay or beta-shift surrogate branches.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "reject",
            "gate C farther hybrid continuation reopen required now",
            0.0,
            "Farther hybrid continuation remains reserve-only because the live blocker is still the scalar-proxy matching law.",
        ),
        sign_base.row(
            "scalar_proxy_exact_matching_law_closed_form_still_missing_now",
            "reject" if not exact_matching_law_closed_form_available_now else "pass",
            "scalar-proxy exact matching-law closed form still missing now",
            sign_base.truth(exact_matching_law_closed_form_available_now),
            "The inventory now exists, but no alpha-target-free closed-form matching law has been derived yet.",
        ),
        sign_base.row(
            "scalar_proxy_effective_beta_shift_secondary_only_now",
            "pass" if effective_beta_shift_secondary_only_now else "reject",
            "scalar-proxy effective beta shift secondary only now",
            sign_base.truth(effective_beta_shift_secondary_only_now),
            "beta_eff remains secondary because the inventory now points at q-law completion, not eigenvalue re-reading, as the primary task.",
        ),
        sign_base.row(
            "selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now",
            "pass" if source_materialization_secondary_reserve_retained_now else "reject",
            "selected-extension independent extra-q-range source-materialization secondary reserve retained now",
            sign_base.truth(source_materialization_secondary_reserve_retained_now),
            "The old extra-q branch stays reserve-only unless the new matching-law completion family dead-ends.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive route update required now",
            sign_base.truth(pack_update_required_now),
            "A substantive route update has happened: the matching-law inventory is complete and the front-runner completion family is now the primary blocker.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_exact_over_m0": float(prior_summary["q_exact_over_m0"]),
        "q_star_over_m0": float(prior_summary["q_star_over_m0"]),
        "beta1": float(prior_summary["beta1"]),
        "legacy_support_phase_best_name": str(prior_summary["legacy_support_phase_best_name"]),
        "legacy_support_phase_best_rel_error": float(prior_summary["legacy_support_phase_best_rel_error"]),
        "centroid_best_name": str(prior_summary["centroid_best_name"]),
        "centroid_best_rel_error": float(prior_summary["centroid_best_rel_error"]),
        "q_star_correction_c1_fit": float(prior_summary["q_star_correction_c1_fit"]),
        "delta_kappa_squared_rel": float(prior_summary["delta_kappa_squared_rel"]),
        "gate_a_updated_pack_exact_scalar_proxy_matching_law_inventory_available_now": gate_a,
        "gate_b_updated_pack_scalar_proxy_profile_sensitive_q_star_correction_promoted_next": gate_b,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c,
        "exact_matching_law_closed_form_available_now": exact_matching_law_closed_form_available_now,
        "scalar_proxy_effective_beta_shift_secondary_only_now": effective_beta_shift_secondary_only_now,
        "selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now": source_materialization_secondary_reserve_retained_now,
        "pack_update_required_now": pack_update_required_now,
        "selected_primary_completion_lane": "updated_pack_scalar_proxy_profile_sensitive_q_star_correction_audit",
        "selected_secondary_completion_lane": "updated_pack_scalar_proxy_effective_beta_shift_sensitivity_review",
        "selected_reserve_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun",
        "selected_next_generation_route": "trial2_numeric_alpha_scalar_proxy_profile_sensitive_q_star_correction_audit",
        "recommended_next_route_or_none": "8.7.56.5399",
        "selected_followup_route": "trial2_numeric_alpha_scalar_proxy_profile_sensitive_q_star_correction_gate",
        "selected_followup_route_or_none": "8.7.56.5403",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5397",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5399",
                "followup_route": "8.7.56.5403",
            },
        },
        rows,
        summary,
        {
            "overall_status": "scalar_proxy_matching_law_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} scalar-proxy matching-law gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# Function: run the gate refresh when invoked as one CLI script.

if __name__ == "__main__":
    main()
