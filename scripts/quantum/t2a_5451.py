#!/usr/bin/env python3
"""Generate 8.7.56.5451-.5454 Route-C gate / source-materialization refresh artifacts."""

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
        "8.7.56.5447-5450",
        "updated_pack_scalar_proxy_route_c_virial_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5451-5454"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor scalar-proxy "
    "Route-C virial gate / source-materialization refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_scalar_proxy_route_c_virial_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_route_c_virial_exact_identity_available_target_free_"
    "matching_law_bridge_missing_source_materialization_primary_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_route_c_virial_negative_closeout_completed_"
    "source_materialization_primary_next"
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


# Function: return formulas used by the Route-C gate refresh.

def build_formulae() -> dict[str, str]:
    """Return formulas used by the Route-C gate refresh."""
    return {
        "gate_a": "Gate A = Route-C virial negative closeout available now",
        "gate_b": "Gate B = selected-extension source-materialization promoted primary now",
        "gate_c": "Gate C = selected-extension source-materialization reopen required now",
    }


# Function: execute `.5451-.5454`.

def main() -> None:
    """Execute the Route-C gate / source-materialization refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary["route_c_exact_weighted_eom_identity_available_now"]
        and prior_summary["route_c_exact_virial_identity_available_now"]
        and prior_summary["route_c_exact_cubic_coefficient_recovered_now"]
        and prior_summary["route_c_negative_closeout_available_now"]
        and not prior_summary["route_c_target_free_matching_law_bridge_available_now"]
    )
    gate_b = bool(gate_a and prior_summary["selected_extension_source_materialization_promoted_primary_now"])
    gate_c = bool(gate_b)
    pack_update_required_now = bool(gate_b)

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_scalar_proxy_route_c_virial_negative_closeout_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack exact scalar-proxy Route-C virial negative closeout available now",
            sign_base.truth(gate_a),
            "Route C now closes honestly as the last theorem-side branch: coefficient 3 is visible, but the matching-law bridge is still absent.",
        ),
        sign_base.row(
            "gate_b_updated_pack_selected_extension_source_materialization_promoted_primary_now",
            "pass" if gate_b else "reject",
            "gate B updated-pack selected-extension source-materialization promoted primary now",
            sign_base.truth(gate_b),
            "With Route C exhausted, the held source-materialization lane becomes the next honest primary computation branch.",
        ),
        sign_base.row(
            "gate_c_selected_extension_source_materialization_reopen_required_now",
            "pass" if gate_c else "reject",
            "gate C selected-extension source-materialization reopen required now",
            sign_base.truth(gate_c),
            "The reserve source-materialization lane now reopens because no theorem-side derivation route remains live.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive route update required now",
            sign_base.truth(pack_update_required_now),
            "A substantive route update has happened: the three-halves derivation lane closes negatively and source-materialization returns primary.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_squared_correction_coeff_fit": float(prior_summary["q_squared_correction_coeff_fit"]),
        "q_squared_correction_coeff_rel_error_vs_exact_cubic": float(
            prior_summary["q_squared_correction_coeff_rel_error_vs_exact_cubic"]
        ),
        "cubic_coeff_from_exact_weighted_eom": float(prior_summary["cubic_coeff_from_exact_weighted_eom"]),
        "cubic_coeff_from_exact_virial": float(prior_summary["cubic_coeff_from_exact_virial"]),
        "cubic_coeff_from_boundary_free_weighted_eom": float(
            prior_summary["cubic_coeff_from_boundary_free_weighted_eom"]
        ),
        "cubic_coeff_from_boundary_free_virial": float(
            prior_summary["cubic_coeff_from_boundary_free_virial"]
        ),
        "boundary_weighted_eom_over_cubic": float(prior_summary["boundary_weighted_eom_over_cubic"]),
        "boundary_virial_over_cubic": float(prior_summary["boundary_virial_over_cubic"]),
        "gate_a_updated_pack_exact_scalar_proxy_route_c_virial_negative_closeout_available_now": gate_a,
        "gate_b_updated_pack_selected_extension_source_materialization_promoted_primary_now": gate_b,
        "gate_c_selected_extension_source_materialization_reopen_required_now": gate_c,
        "route_c_negative_closeout_available_now": bool(prior_summary["route_c_negative_closeout_available_now"]),
        "selected_extension_source_materialization_promoted_primary_now": bool(
            prior_summary["selected_extension_source_materialization_promoted_primary_now"]
        ),
        "selected_primary_completion_lane": (
            "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun"
        ),
        "selected_secondary_completion_lane": "updated_pack_scalar_proxy_route_c_archive_review",
        "selected_reserve_completion_lane": "updated_pack_scalar_proxy_direct_fourier_nlo_gap_review",
        "selected_next_generation_route": (
            "trial2_numeric_alpha_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun"
        ),
        "recommended_next_route_or_none": "8.7.56.5455",
        "selected_followup_route": (
            "trial2_numeric_alpha_selected_extension_independent_extra_q_range_source_materialization_gate"
        ),
        "selected_followup_route_or_none": "8.7.56.5459",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5453",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5455",
                "followup_route": "8.7.56.5459",
            },
        },
        rows,
        summary,
        {
            "overall_status": "scalar_proxy_route_c_virial_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} scalar-proxy Route-C virial gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# Function: run the gate refresh when invoked as one CLI script.

if __name__ == "__main__":
    main()
