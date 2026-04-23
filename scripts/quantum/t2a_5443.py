#!/usr/bin/env python3
"""Generate 8.7.56.5443-.5446 Route-D gate / Route-C refresh artifacts."""

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
        "8.7.56.5439-5442",
        "updated_pack_scalar_proxy_route_d_profile_moment_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5443-5446"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor scalar-proxy "
    "Route-D profile moment gate / source-materialization secondary refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_scalar_proxy_route_d_profile_moment_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_route_d_profile_moment_low_order_truncation_no_go_theorem_derived_"
    "route_c_virial_primary_source_materialization_reserve_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_route_d_profile_moment_negative_closeout_completed_"
    "route_c_virial_primary_source_materialization_reserve_next"
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


# Function: return formulas used by the Route-D gate refresh.

def build_formulae() -> dict[str, str]:
    """Return formulas used by the Route-D gate refresh."""
    return {
        "gate_a": "Gate A = Route-D low-order profile-moment no-go theorem available now",
        "gate_b": "Gate B = Route-C virial promoted next",
        "gate_c": "Gate C = selected-extension source-materialization reopen required now",
    }


# Function: execute `.5443-.5446`.

def main() -> None:
    """Execute the Route-D gate refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary["route_d_profile_moment_scaling_formula_available_now"]
        and prior_summary["route_d_low_order_profile_moment_no_go_theorem_available_now"]
        and not prior_summary["route_d_target_free_exact_derivation_available_now"]
    )
    gate_b = bool(gate_a and prior_summary["route_c_virial_promoted_next_now"])
    gate_c = False
    pack_update_required_now = bool(gate_b)
    source_materialization_kept_secondary_reserve_now = True

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_scalar_proxy_route_d_profile_moment_no_go_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack exact scalar-proxy Route-D profile moment no-go available now",
            sign_base.truth(gate_a),
            "The direct low-order profile-moment route has now closed honestly as a no-go under the retained target-free derivation requirement.",
        ),
        sign_base.row(
            "gate_b_updated_pack_scalar_proxy_route_c_virial_promoted_next",
            "pass" if gate_b else "reject",
            "gate B updated-pack scalar-proxy Route-C virial promoted next",
            sign_base.truth(gate_b),
            "With Route D closed negatively, Route C virial becomes the only remaining theorem-side derivation branch.",
        ),
        sign_base.row(
            "gate_c_selected_extension_source_materialization_reopen_required_now",
            "reject",
            "gate C selected-extension source-materialization reopen required now",
            0.0,
            "The reserve source-materialization lane still does not reopen while Route C remains one honest theorem-side alternative.",
        ),
        sign_base.row(
            "selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now",
            "pass" if source_materialization_kept_secondary_reserve_now else "reject",
            "selected-extension independent extra-q-range source-materialization secondary reserve retained now",
            sign_base.truth(source_materialization_kept_secondary_reserve_now),
            "Source-materialization remains reserve-only while Route C is now the promoted honest blocker.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive route update required now",
            sign_base.truth(pack_update_required_now),
            "A substantive route update has happened: Route D closes negatively and Route C virial is promoted primary.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_star_control_parameter_q2_abs": float(prior_summary["q_star_control_parameter_q2_abs"]),
        "q_exact_control_parameter_q2_abs": float(prior_summary["q_exact_control_parameter_q2_abs"]),
        "q_star_best_truncation_abs_error": float(prior_summary["q_star_best_truncation_abs_error"]),
        "q_exact_best_truncation_abs_error": float(prior_summary["q_exact_best_truncation_abs_error"]),
        "gate_a_updated_pack_exact_scalar_proxy_route_d_profile_moment_no_go_available_now": gate_a,
        "gate_b_updated_pack_scalar_proxy_route_c_virial_promoted_next": gate_b,
        "gate_c_selected_extension_source_materialization_reopen_required_now": gate_c,
        "route_d_low_order_profile_moment_no_go_theorem_available_now": bool(
            prior_summary["route_d_low_order_profile_moment_no_go_theorem_available_now"]
        ),
        "route_c_virial_promoted_next_now": bool(prior_summary["route_c_virial_promoted_next_now"]),
        "selected_primary_completion_lane": "updated_pack_scalar_proxy_route_c_virial_audit",
        "selected_secondary_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun",
        "selected_reserve_completion_lane": "updated_pack_scalar_proxy_route_d_profile_moment_resummation_review",
        "selected_next_generation_route": "trial2_numeric_alpha_scalar_proxy_route_c_virial_audit",
        "recommended_next_route_or_none": "8.7.56.5447",
        "selected_followup_route": "trial2_numeric_alpha_scalar_proxy_route_c_virial_gate",
        "selected_followup_route_or_none": "8.7.56.5451",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5445",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5447",
                "followup_route": "8.7.56.5451",
            },
        },
        rows,
        summary,
        {
            "overall_status": "scalar_proxy_route_d_profile_moment_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} scalar-proxy Route-D profile moment gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# Function: run the gate refresh when invoked as one CLI script.

if __name__ == "__main__":
    main()
