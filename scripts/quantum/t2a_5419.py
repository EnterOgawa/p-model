#!/usr/bin/env python3
"""Generate 8.7.56.5419-.5422 Route-A gate / Route-D secondary refresh artifacts."""

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
        "8.7.56.5415-5418",
        "updated_pack_scalar_proxy_route_a_eom_perturbation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5419-5422"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor scalar-proxy "
    "Route-A EOM perturbation gate / Route-D secondary refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_scalar_proxy_route_a_eom_perturbation_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_route_a_lo_cubic_scaleout_no_go_theorem_derived_"
    "nlo_perturbation_primary_route_d_secondary_"
    "source_materialization_reserve_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_route_a_lo_cubic_scaleout_negative_closeout_completed_"
    "nlo_perturbation_primary_route_d_secondary_"
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
    """Return formulas used by the Route-A gate refresh."""
    return {
        "gate_a": "Gate A = Route-A LO cubic-scaleout no-go theorem available now",
        "gate_b": "Gate B = Route-A NLO perturbation promoted next",
        "gate_c": "Gate C = selected-extension source-materialization reopen required now",
    }


# Function: execute `.5419-.5422`.

def main() -> None:
    """Execute the Route-A gate / Route-D secondary refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary["route_a_scaled_reduced_eom_available_now"]
        and prior_summary["route_a_lo_generic_cubic_scaleout_theorem_available_now"]
        and prior_summary["route_a_lo_normalized_overlap_cubic_independence_theorem_available_now"]
        and prior_summary["route_a_lo_cubic_scaleout_no_go_theorem_available_now"]
    )
    gate_b = bool(gate_a and prior_summary["route_a_nlo_perturbation_promoted_next_now"])
    gate_c = False
    route_d_profile_moment_kept_secondary_now = bool(prior_summary["route_d_profile_moment_kept_secondary_now"])
    route_c_virial_kept_reserve_now = bool(prior_summary["route_c_virial_kept_reserve_now"])
    source_materialization_secondary_reserve_retained_now = bool(
        prior_summary["selected_reserve_completion_lane"]
        == "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun"
    )
    pack_update_required_now = bool(gate_b)

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_scalar_proxy_route_a_lo_cubic_scaleout_no_go_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack exact scalar-proxy Route-A LO cubic scaleout no-go available now",
            sign_base.truth(gate_a),
            "The LO Route-A audit reached one honest theorem-side no-go rather than another algebra replay.",
        ),
        sign_base.row(
            "gate_b_updated_pack_scalar_proxy_route_a_nlo_perturbation_promoted_next",
            "pass" if gate_b else "reject",
            "gate B updated-pack scalar-proxy Route-A NLO perturbation promoted next",
            sign_base.truth(gate_b),
            "Because LO Route A closes negatively, the next honest primary lane is the NLO perturbation equation.",
        ),
        sign_base.row(
            "gate_c_selected_extension_source_materialization_reopen_required_now",
            "reject",
            "gate C selected-extension source-materialization reopen required now",
            0.0,
            "The reserve source-materialization lane still does not reopen while Route A and Route D remain live theorem-side options.",
        ),
        sign_base.row(
            "scalar_proxy_route_d_profile_moment_kept_secondary_now",
            "pass" if route_d_profile_moment_kept_secondary_now else "reject",
            "scalar-proxy Route-D profile moment kept secondary now",
            sign_base.truth(route_d_profile_moment_kept_secondary_now),
            "Route D stays secondary because it remains the best numerical cross-check against the Route-A theorem path.",
        ),
        sign_base.row(
            "scalar_proxy_route_c_virial_kept_reserve_now",
            "pass" if route_c_virial_kept_reserve_now else "reject",
            "scalar-proxy Route-C virial kept reserve now",
            sign_base.truth(route_c_virial_kept_reserve_now),
            "Virial remains reserve because neither Route B nor LO Route A produced coefficient-level support from it.",
        ),
        sign_base.row(
            "selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now",
            "pass" if source_materialization_secondary_reserve_retained_now else "reject",
            "selected-extension independent extra-q-range source-materialization secondary reserve retained now",
            sign_base.truth(source_materialization_secondary_reserve_retained_now),
            "Source-materialization remains reserve-only while Route A NLO is now the honest primary blocker.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive route update required now",
            sign_base.truth(pack_update_required_now),
            "A substantive route update has happened: Route A LO closes negatively and Route A NLO is now primary next.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_exact_over_m0": float(prior_summary["q_exact_over_m0"]),
        "q_star_over_m0": float(prior_summary["q_star_over_m0"]),
        "q_cubic_sqrt_over_m0": float(prior_summary["q_cubic_sqrt_over_m0"]),
        "scaled_center_amplitude": float(prior_summary["scaled_center_amplitude"]),
        "scaled_u_abs_max": float(prior_summary["scaled_u_abs_max"]),
        "quartic_to_cubic_ratio_max": float(prior_summary["quartic_to_cubic_ratio_max"]),
        "quartic_to_linear_ratio_weighted_mean": float(prior_summary["quartic_to_linear_ratio_weighted_mean"]),
        "lo_scaleout_density_diff_max_abs": float(prior_summary["lo_scaleout_density_diff_max_abs"]),
        "lo_scaleout_form_factor_diff_max_abs": float(prior_summary["lo_scaleout_form_factor_diff_max_abs"]),
        "gate_a_updated_pack_exact_scalar_proxy_route_a_lo_cubic_scaleout_no_go_available_now": gate_a,
        "gate_b_updated_pack_scalar_proxy_route_a_nlo_perturbation_promoted_next": gate_b,
        "gate_c_selected_extension_source_materialization_reopen_required_now": gate_c,
        "route_a_lo_cubic_scaleout_no_go_theorem_available_now": bool(
            prior_summary["route_a_lo_cubic_scaleout_no_go_theorem_available_now"]
        ),
        "route_a_nlo_perturbation_equation_available_now": bool(
            prior_summary["route_a_nlo_perturbation_equation_available_now"]
        ),
        "route_a_nlo_perturbation_promoted_next_now": bool(
            prior_summary["route_a_nlo_perturbation_promoted_next_now"]
        ),
        "route_d_profile_moment_kept_secondary_now": route_d_profile_moment_kept_secondary_now,
        "route_c_virial_kept_reserve_now": route_c_virial_kept_reserve_now,
        "selected_primary_completion_lane": "updated_pack_scalar_proxy_route_a_nlo_perturbation_audit",
        "selected_secondary_completion_lane": "updated_pack_scalar_proxy_route_d_profile_moment_audit",
        "selected_reserve_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun",
        "selected_next_generation_route": "trial2_numeric_alpha_scalar_proxy_route_a_nlo_perturbation_audit",
        "recommended_next_route_or_none": "8.7.56.5423",
        "selected_followup_route": "trial2_numeric_alpha_scalar_proxy_route_a_nlo_perturbation_gate",
        "selected_followup_route_or_none": "8.7.56.5427",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5421",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5423",
                "followup_route": "8.7.56.5427",
            },
        },
        rows,
        summary,
        {
            "overall_status": "scalar_proxy_route_a_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} scalar-proxy Route-A gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# Function: run the gate refresh when invoked as one CLI script.

if __name__ == "__main__":
    main()
