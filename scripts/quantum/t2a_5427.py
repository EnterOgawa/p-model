#!/usr/bin/env python3
"""Generate 8.7.56.5427-.5430 Route-A NLO gate / Route-D secondary refresh artifacts."""

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
        "8.7.56.5423-5426",
        "updated_pack_scalar_proxy_route_a_nlo_perturbation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5427-5430"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor scalar-proxy "
    "Route-A NLO perturbation gate / Route-D secondary refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_scalar_proxy_route_a_nlo_perturbation_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_route_a_nlo_universal_twentyseven_response_front_runner_derived_"
    "route_d_secondary_source_materialization_reserve_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_route_a_nlo_universal_twentyseven_response_front_runner_audited_"
    "exact_derivation_primary_route_d_secondary_source_materialization_reserve_next"
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
    """Return formulas used by the Route-A NLO gate refresh."""
    return {
        "gate_a": "Gate A = Route-A NLO universal linearized equation plus required universal response fit available now",
        "gate_b": "Gate B = Route-A exact universal twenty-seven derivation promoted next",
        "gate_c": "Gate C = selected-extension source-materialization reopen required now",
    }


# Function: execute `.5427-.5430`.

def main() -> None:
    """Execute the Route-A NLO gate / Route-D secondary refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary["route_a_nlo_generic_scaled_family_formula_available_now"]
        and prior_summary["route_a_nlo_universal_linearized_equation_available_now"]
        and prior_summary["route_a_nlo_required_universal_twentyseven_response_fit_available_now"]
        and prior_summary["route_a_nlo_universal_twentyseven_front_runner_available_now"]
    )
    gate_b = bool(gate_a and prior_summary["route_a_nlo_universal_twentyseven_promoted_next_now"])
    gate_c = False
    route_d_profile_moment_kept_secondary_now = bool(prior_summary["route_d_profile_moment_kept_secondary_now"])
    source_materialization_secondary_reserve_retained_now = bool(
        prior_summary["selected_reserve_completion_lane"]
        == "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun"
    )
    pack_update_required_now = bool(gate_b)

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_scalar_proxy_route_a_nlo_universal_twentyseven_front_runner_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack exact scalar-proxy Route-A NLO universal twenty-seven front-runner available now",
            sign_base.truth(gate_a),
            "The NLO algebra is now reduced to one universal coefficient problem rather than one generic perturbation family.",
        ),
        sign_base.row(
            "gate_b_updated_pack_scalar_proxy_route_a_exact_universal_twentyseven_derivation_promoted_next",
            "pass" if gate_b else "reject",
            "gate B updated-pack scalar-proxy Route-A exact universal twenty-seven derivation promoted next",
            sign_base.truth(gate_b),
            "The next honest primary lane is the target-free derivation of the universal coefficient 27 itself.",
        ),
        sign_base.row(
            "gate_c_selected_extension_source_materialization_reopen_required_now",
            "reject",
            "gate C selected-extension source-materialization reopen required now",
            0.0,
            "The reserve source-materialization lane still does not reopen while Route A retains one compact exact coefficient blocker.",
        ),
        sign_base.row(
            "scalar_proxy_route_d_profile_moment_kept_secondary_now",
            "pass" if route_d_profile_moment_kept_secondary_now else "reject",
            "scalar-proxy Route-D profile moment kept secondary now",
            sign_base.truth(route_d_profile_moment_kept_secondary_now),
            "Route D stays secondary because it remains the strongest numerical cross-check while Route A exact-coefficient derivation is still alive.",
        ),
        sign_base.row(
            "selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now",
            "pass" if source_materialization_secondary_reserve_retained_now else "reject",
            "selected-extension independent extra-q-range source-materialization secondary reserve retained now",
            sign_base.truth(source_materialization_secondary_reserve_retained_now),
            "Source-materialization remains reserve-only while the scalar-proxy exact coefficient derivation still has one honest remaining branch.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive route update required now",
            sign_base.truth(pack_update_required_now),
            "A substantive route update has happened: generic NLO algebra is no longer the blocker, and the exact universal coefficient derivation is promoted next.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta1": float(prior_summary["beta1"]),
        "epsilon_beta": float(prior_summary["epsilon_beta"]),
        "g3_actual": float(prior_summary["g3_actual"]),
        "route_a_nlo_small_parameter_eta_actual": float(prior_summary["route_a_nlo_small_parameter_eta_actual"]),
        "route_a_nlo_required_universal_q_squared_response_coeff_fit": float(
            prior_summary["route_a_nlo_required_universal_q_squared_response_coeff_fit"]
        ),
        "route_a_nlo_universal_q_squared_response_coeff_candidate": float(
            prior_summary["route_a_nlo_universal_q_squared_response_coeff_candidate"]
        ),
        "route_a_nlo_universal_q_squared_response_coeff_rel_error": float(
            prior_summary["route_a_nlo_universal_q_squared_response_coeff_rel_error"]
        ),
        "gate_a_updated_pack_exact_scalar_proxy_route_a_nlo_universal_twentyseven_front_runner_available_now": gate_a,
        "gate_b_updated_pack_scalar_proxy_route_a_exact_universal_twentyseven_derivation_promoted_next": gate_b,
        "gate_c_selected_extension_source_materialization_reopen_required_now": gate_c,
        "route_a_nlo_generic_scaled_family_formula_available_now": bool(
            prior_summary["route_a_nlo_generic_scaled_family_formula_available_now"]
        ),
        "route_a_nlo_universal_linearized_equation_available_now": bool(
            prior_summary["route_a_nlo_universal_linearized_equation_available_now"]
        ),
        "route_a_nlo_required_universal_twentyseven_response_fit_available_now": bool(
            prior_summary["route_a_nlo_required_universal_twentyseven_response_fit_available_now"]
        ),
        "route_a_nlo_universal_twentyseven_front_runner_available_now": bool(
            prior_summary["route_a_nlo_universal_twentyseven_front_runner_available_now"]
        ),
        "route_a_nlo_target_free_twentyseven_derivation_available_now": bool(
            prior_summary["route_a_nlo_target_free_twentyseven_derivation_available_now"]
        ),
        "route_d_profile_moment_kept_secondary_now": route_d_profile_moment_kept_secondary_now,
        "selected_primary_completion_lane": "updated_pack_scalar_proxy_route_a_exact_universal_twentyseven_derivation_audit",
        "selected_secondary_completion_lane": "updated_pack_scalar_proxy_route_d_profile_moment_audit",
        "selected_reserve_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun",
        "selected_next_generation_route": "trial2_numeric_alpha_scalar_proxy_route_a_exact_universal_twentyseven_derivation_audit",
        "recommended_next_route_or_none": "8.7.56.5431",
        "selected_followup_route": "trial2_numeric_alpha_scalar_proxy_route_a_exact_universal_twentyseven_derivation_gate",
        "selected_followup_route_or_none": "8.7.56.5435",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5429",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5431",
                "followup_route": "8.7.56.5435",
            },
        },
        rows,
        summary,
        {
            "overall_status": "scalar_proxy_route_a_nlo_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} scalar-proxy Route-A NLO gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# Function: run the gate refresh when invoked as one CLI script.

if __name__ == "__main__":
    main()
