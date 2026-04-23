#!/usr/bin/env python3
"""Generate 8.7.56.5423-.5426 Route-A NLO perturbation audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.scalar_proxy_route_a_eom_perturbation_backend import (
    build_scalar_proxy_route_a_eom_perturbation_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5419-5422",
        "updated_pack_scalar_proxy_route_a_eom_perturbation_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5423-5426"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor scalar-proxy "
    "Route-A NLO perturbation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_scalar_proxy_route_a_nlo_perturbation_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_route_a_lo_cubic_scaleout_negative_closeout_completed_"
    "nlo_perturbation_primary_route_d_secondary_"
    "source_materialization_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_route_a_nlo_universal_twentyseven_response_front_runner_derived_"
    "route_d_secondary_source_materialization_reserve_gate"
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


# Function: return formulas used by the Route-A NLO audit.

def build_formulae() -> dict[str, str]:
    """Return the Route-A NLO formulas fixed by the audit."""
    return {
        "generic_exact_scaled_equation": (
            "For generic cubic coefficient g3, the exact reduced equation is "
            "u'' + (2/xi) u' - u + g3 u^2 + epsilon_beta u^3 = 0"
        ),
        "generic_nlo_scaleout": (
            "Set u = v / g3, then v'' + (2/xi) v' - v + v^2 + "
            "(epsilon_beta / g3^2) v^3 = 0"
        ),
        "nlo_linearized_equation": (
            "With eta = epsilon_beta / g3^2 and v = v0 + eta v1 + O(eta^2), "
            "the NLO linearized equation is v1'' + (2/xi) v1' - v1 + 2 v0 v1 = -v0^3"
        ),
        "required_universal_response": (
            "If delta(q^2)/q_star^2 = -C_univ eta, then C_univ(required) = "
            "q_squared_correction_coeff_fit * g3^2"
        ),
        "route_a_nlo_verdict": (
            "Route A NLO keeps one universal coefficient problem alive: after exact "
            "generic scaleout the remaining target is one target-free derivation of "
            "C_univ = 27, which would reproduce the observed coefficient 3 for g3 = 3"
        ),
    }


# Function: execute `.5423-.5426`.

def main() -> None:
    """Execute the Route-A NLO perturbation audit."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_scalar_proxy_route_a_eom_perturbation_pack()

    route_a_nlo_audit_available_now = bool(
        prior_summary["gate_a_updated_pack_exact_scalar_proxy_route_a_lo_cubic_scaleout_no_go_available_now"]
        and prior_summary["gate_b_updated_pack_scalar_proxy_route_a_nlo_perturbation_promoted_next"]
    )
    route_a_nlo_generic_scaled_family_formula_available_now = bool(
        pack["route_a_nlo_generic_scaled_family_formula_available_now"]
    )
    route_a_nlo_universal_linearized_equation_available_now = bool(
        pack["route_a_nlo_universal_linearized_equation_available_now"]
    )
    route_a_nlo_required_universal_twentyseven_response_fit_available_now = bool(
        pack["route_a_nlo_required_universal_twentyseven_response_fit_available_now"]
    )
    route_a_nlo_universal_twentyseven_front_runner_available_now = bool(
        pack["route_a_nlo_universal_twentyseven_front_runner_available_now"]
    )
    route_a_nlo_target_free_twentyseven_derivation_available_now = bool(
        pack["route_a_nlo_target_free_twentyseven_derivation_available_now"]
    )
    route_a_nlo_universal_twentyseven_promoted_next_now = bool(
        pack["route_a_nlo_universal_twentyseven_promoted_next_now"]
    )
    route_d_profile_moment_kept_secondary_now = bool(pack["route_d_profile_moment_kept_secondary_now"])
    source_materialization_secondary_reserve_retained_now = True

    rows = [
        sign_base.row(
            "exact_scalar_proxy_route_a_nlo_perturbation_audit_available_now",
            "pass" if route_a_nlo_audit_available_now else "reject",
            "exact scalar-proxy Route-A NLO perturbation audit available now",
            sign_base.truth(route_a_nlo_audit_available_now),
            "Route A LO has already closed negatively, so the honest next theorem-side branch is the NLO perturbation audit.",
        ),
        sign_base.row(
            "scalar_proxy_route_a_nlo_generic_scaled_family_formula_available_now",
            "pass" if route_a_nlo_generic_scaled_family_formula_available_now else "reject",
            "scalar-proxy Route-A NLO generic scaled family formula available now",
            sign_base.truth(route_a_nlo_generic_scaled_family_formula_available_now),
            "After u = v/g3 the exact generic cubic family keeps only eta = epsilon_beta/g3^2 as explicit g3 dependence.",
        ),
        sign_base.row(
            "scalar_proxy_route_a_nlo_universal_linearized_equation_available_now",
            "pass" if route_a_nlo_universal_linearized_equation_available_now else "reject",
            "scalar-proxy Route-A NLO universal linearized equation available now",
            sign_base.truth(route_a_nlo_universal_linearized_equation_available_now),
            "The first perturbative correction is governed by one universal linearized equation that no longer contains g3 explicitly.",
        ),
        sign_base.row(
            "scalar_proxy_route_a_nlo_required_universal_twentyseven_response_fit_available_now",
            "pass" if route_a_nlo_required_universal_twentyseven_response_fit_available_now else "reject",
            "scalar-proxy Route-A NLO required universal twenty-seven response fit available now",
            sign_base.truth(route_a_nlo_required_universal_twentyseven_response_fit_available_now),
            "The retained q-squared fit can be re-expressed as one required universal response coefficient C_univ on top of eta = epsilon_beta/g3^2.",
        ),
        sign_base.row(
            "scalar_proxy_route_a_nlo_universal_twentyseven_front_runner_available_now",
            "pass" if route_a_nlo_universal_twentyseven_front_runner_available_now else "reject",
            "scalar-proxy Route-A NLO universal twenty-seven front-runner available now",
            sign_base.truth(route_a_nlo_universal_twentyseven_front_runner_available_now),
            "The required universal response coefficient is numerically close enough to 27 to promote C_univ = 27 as the honest front-runner target.",
        ),
        sign_base.row(
            "scalar_proxy_route_a_nlo_target_free_twentyseven_derivation_available_now",
            "pass" if route_a_nlo_target_free_twentyseven_derivation_available_now else "reject",
            "scalar-proxy Route-A NLO target-free twenty-seven derivation available now",
            sign_base.truth(route_a_nlo_target_free_twentyseven_derivation_available_now),
            "This remains false until Route A NLO derives the universal coefficient 27 directly from the frozen-action EOM without importing q_exact.",
        ),
        sign_base.row(
            "scalar_proxy_route_a_nlo_universal_twentyseven_promoted_next_now",
            "pass" if route_a_nlo_universal_twentyseven_promoted_next_now else "reject",
            "scalar-proxy Route-A NLO universal twenty-seven promoted next now",
            sign_base.truth(route_a_nlo_universal_twentyseven_promoted_next_now),
            "The next honest blocker is no longer generic NLO algebra; it is the target-free derivation of the universal coefficient 27 itself.",
        ),
        sign_base.row(
            "scalar_proxy_route_d_profile_moment_kept_secondary_now",
            "pass" if route_d_profile_moment_kept_secondary_now else "reject",
            "scalar-proxy Route-D profile moment kept secondary now",
            sign_base.truth(route_d_profile_moment_kept_secondary_now),
            "Route D remains the strongest secondary numerical cross-check while Route A NLO stays primary.",
        ),
        sign_base.row(
            "selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now",
            "pass" if source_materialization_secondary_reserve_retained_now else "reject",
            "selected-extension independent extra-q-range source-materialization secondary reserve retained now",
            sign_base.truth(source_materialization_secondary_reserve_retained_now),
            "The reserve source-materialization lane still stays closed while theorem-side Route A has one concrete exact coefficient left to derive.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta1": float(pack["beta1"]),
        "epsilon_beta": float(pack["epsilon_beta"]),
        "g3_actual": float(pack["g3_actual"]),
        "route_a_nlo_small_parameter_eta_actual": float(pack["route_a_nlo_small_parameter_eta_actual"]),
        "q_star_over_m0": float(pack["q_star_over_m0"]),
        "q_exact_over_m0": float(pack["q_exact_over_m0"]),
        "q_cubic_sqrt_over_m0": float(pack["q_cubic_sqrt_over_m0"]),
        "q_squared_correction_coeff_fit": float(pack["q_squared_correction_coeff_fit"]),
        "route_a_nlo_required_universal_q_squared_response_coeff_fit": float(
            pack["route_a_nlo_required_universal_q_squared_response_coeff_fit"]
        ),
        "route_a_nlo_universal_q_squared_response_coeff_candidate": float(
            pack["route_a_nlo_universal_q_squared_response_coeff_candidate"]
        ),
        "route_a_nlo_universal_q_squared_response_coeff_abs_error": float(
            pack["route_a_nlo_universal_q_squared_response_coeff_abs_error"]
        ),
        "route_a_nlo_universal_q_squared_response_coeff_rel_error": float(
            pack["route_a_nlo_universal_q_squared_response_coeff_rel_error"]
        ),
        "route_a_nlo_generic_scaled_family_formula_available_now": (
            route_a_nlo_generic_scaled_family_formula_available_now
        ),
        "route_a_nlo_universal_linearized_equation_available_now": (
            route_a_nlo_universal_linearized_equation_available_now
        ),
        "route_a_nlo_required_universal_twentyseven_response_fit_available_now": (
            route_a_nlo_required_universal_twentyseven_response_fit_available_now
        ),
        "route_a_nlo_universal_twentyseven_front_runner_available_now": (
            route_a_nlo_universal_twentyseven_front_runner_available_now
        ),
        "route_a_nlo_target_free_twentyseven_derivation_available_now": (
            route_a_nlo_target_free_twentyseven_derivation_available_now
        ),
        "route_a_nlo_universal_twentyseven_promoted_next_now": (
            route_a_nlo_universal_twentyseven_promoted_next_now
        ),
        "route_d_profile_moment_kept_secondary_now": route_d_profile_moment_kept_secondary_now,
        "selected_primary_completion_lane": "updated_pack_scalar_proxy_route_a_exact_universal_twentyseven_derivation_audit",
        "selected_secondary_completion_lane": "updated_pack_scalar_proxy_route_d_profile_moment_audit",
        "selected_reserve_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun",
        "selected_next_generation_route": "trial2_numeric_alpha_scalar_proxy_route_a_exact_universal_twentyseven_derivation_audit",
        "recommended_next_route_or_none": "8.7.56.5427",
        "selected_followup_route": "trial2_numeric_alpha_scalar_proxy_route_a_exact_universal_twentyseven_derivation_gate",
        "selected_followup_route_or_none": "8.7.56.5431",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5425",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5427",
                "followup_route": "8.7.56.5431",
            },
        },
        rows,
        summary,
        {
            "overall_status": "scalar_proxy_route_a_nlo_perturbation_audit_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} scalar-proxy Route-A NLO audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# Function: run the audit when invoked as one CLI script.

if __name__ == "__main__":
    main()
