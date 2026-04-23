#!/usr/bin/env python3
"""Generate 8.7.56.5447-.5450 Route-C virial audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.scalar_proxy_route_c_virial_backend import (
    build_scalar_proxy_route_c_virial_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5443-5446",
        "updated_pack_scalar_proxy_route_d_profile_moment_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5447-5450"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor scalar-proxy "
    "Route-C virial audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_scalar_proxy_route_c_virial_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_route_d_profile_moment_negative_closeout_completed_"
    "route_c_virial_primary_source_materialization_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_route_c_virial_exact_identity_available_target_free_"
    "matching_law_bridge_missing_source_materialization_primary_gate"
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


# Function: return formulas fixed by the Route-C virial audit.

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the Route-C virial audit."""
    return {
        "weighted_eom_identity": (
            "R^2 y(R) y'(R) - int dx x^2 y'^2 - epsilon_beta int dx x^2 y^2 + "
            "3 int dx x^2 y^3 + int dx x^2 y^4 = 0"
        ),
        "virial_identity": (
            "B_virial(R) + (1/2) int dx x^2 y'^2 + (3/2) epsilon_beta int dx x^2 y^2 "
            "- 3 int dx x^2 y^3 - (3/4) int dx x^2 y^4 = 0"
        ),
        "virial_verdict": (
            "Route C recovers the Mexican-hat cubic coefficient 3 exactly only after "
            "retained finite-radius boundary terms are kept, but it still does not fix "
            "one target-free bridge from that coefficient to q_corrected."
        ),
    }


# Function: execute `.5447-.5450`.

def main() -> None:
    """Execute the Route-C virial audit."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_scalar_proxy_route_c_virial_pack()

    route_c_virial_audit_available_now = bool(
        prior_summary["gate_a_updated_pack_exact_scalar_proxy_route_d_profile_moment_no_go_available_now"]
        and prior_summary["gate_b_updated_pack_scalar_proxy_route_c_virial_promoted_next"]
    )
    route_c_exact_weighted_eom_identity_available_now = bool(
        pack["route_c_exact_weighted_eom_identity_available_now"]
    )
    route_c_exact_virial_identity_available_now = bool(
        pack["route_c_exact_virial_identity_available_now"]
    )
    route_c_exact_cubic_coefficient_recovered_now = bool(
        pack["route_c_exact_cubic_coefficient_recovered_now"]
    )
    route_c_boundary_terms_negligible_now = bool(pack["route_c_boundary_terms_negligible_now"])
    route_c_boundary_free_virial_truncation_supported_now = bool(
        pack["route_c_boundary_free_virial_truncation_supported_now"]
    )
    route_c_target_free_matching_law_bridge_available_now = bool(
        pack["route_c_target_free_matching_law_bridge_available_now"]
    )
    route_c_negative_closeout_available_now = bool(pack["route_c_negative_closeout_available_now"])
    selected_extension_source_materialization_promoted_primary_now = bool(
        pack["selected_extension_source_materialization_promoted_primary_now"]
    )

    rows = [
        sign_base.row(
            "exact_scalar_proxy_route_c_virial_audit_available_now",
            "pass" if route_c_virial_audit_available_now else "reject",
            "exact scalar-proxy Route-C virial audit available now",
            sign_base.truth(route_c_virial_audit_available_now),
            "With Route D closed negatively, the virial route is now the last honest theorem-side branch for the three-halves law.",
        ),
        sign_base.row(
            "scalar_proxy_route_c_exact_weighted_eom_identity_available_now",
            "pass" if route_c_exact_weighted_eom_identity_available_now else "reject",
            "scalar-proxy Route-C exact weighted-EOM identity available now",
            sign_base.truth(route_c_exact_weighted_eom_identity_available_now),
            "The retained finite-radius profile satisfies the x^2 y weighted identity once the boundary term at the truncation radius is kept.",
        ),
        sign_base.row(
            "scalar_proxy_route_c_exact_virial_identity_available_now",
            "pass" if route_c_exact_virial_identity_available_now else "reject",
            "scalar-proxy Route-C exact virial identity available now",
            sign_base.truth(route_c_exact_virial_identity_available_now),
            "The retained finite-radius virial identity also closes once the same boundary-sensitive bookkeeping is kept explicit.",
        ),
        sign_base.row(
            "scalar_proxy_route_c_exact_cubic_coefficient_recovered_now",
            "pass" if route_c_exact_cubic_coefficient_recovered_now else "reject",
            "scalar-proxy Route-C exact cubic coefficient recovered now",
            sign_base.truth(route_c_exact_cubic_coefficient_recovered_now),
            "Both exact identities return the Mexican-hat cubic coefficient 3 directly, so the theorem-side issue is not coefficient visibility but the missing bridge to q-space.",
        ),
        sign_base.row(
            "scalar_proxy_route_c_boundary_terms_negligible_now",
            "pass" if route_c_boundary_terms_negligible_now else "reject",
            "scalar-proxy Route-C boundary terms negligible now",
            sign_base.truth(route_c_boundary_terms_negligible_now),
            "This remains false: the retained finite-radius boundary terms are O(0.2-0.3) of the cubic piece and cannot be dropped honestly.",
        ),
        sign_base.row(
            "scalar_proxy_route_c_boundary_free_virial_truncation_supported_now",
            "pass" if route_c_boundary_free_virial_truncation_supported_now else "reject",
            "scalar-proxy Route-C boundary-free virial truncation supported now",
            sign_base.truth(route_c_boundary_free_virial_truncation_supported_now),
            "Boundary-free reductions drift to inconsistent effective coefficients instead of one stable exact 3, so the simple virial shortcut is not honest on the retained pack.",
        ),
        sign_base.row(
            "scalar_proxy_route_c_target_free_matching_law_bridge_available_now",
            "pass" if route_c_target_free_matching_law_bridge_available_now else "reject",
            "scalar-proxy Route-C target-free matching-law bridge available now",
            sign_base.truth(route_c_target_free_matching_law_bridge_available_now),
            "The virial identities expose coefficient 3, but they still do not determine q_corrected without importing a separate overlap or matching rule.",
        ),
        sign_base.row(
            "scalar_proxy_route_c_negative_closeout_available_now",
            "pass" if route_c_negative_closeout_available_now else "reject",
            "scalar-proxy Route-C negative closeout available now",
            sign_base.truth(route_c_negative_closeout_available_now),
            "Route C now closes honestly: exact finite-radius virial identities exist, but they do not complete the target-free matching-law derivation.",
        ),
        sign_base.row(
            "selected_extension_source_materialization_promoted_primary_now",
            "pass" if selected_extension_source_materialization_promoted_primary_now else "reject",
            "selected-extension source-materialization promoted primary now",
            sign_base.truth(selected_extension_source_materialization_promoted_primary_now),
            "With Route C also exhausted, the held source-materialization lane becomes the next honest primary computation branch.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta1": float(pack["beta1"]),
        "epsilon_beta": float(pack["epsilon_beta"]),
        "q_exact_over_m0": float(pack["q_exact_over_m0"]),
        "q_star_over_m0": float(pack["q_star_over_m0"]),
        "q_cubic_sqrt_over_m0": float(pack["q_cubic_sqrt_over_m0"]),
        "q_squared_correction_coeff_fit": float(pack["q_squared_correction_coeff_fit"]),
        "q_squared_correction_coeff_rel_error_vs_exact_cubic": float(
            pack["q_squared_correction_coeff_rel_error_vs_exact_cubic"]
        ),
        "radius_end": float(pack["radius_end"]),
        "profile_end": float(pack["profile_end"]),
        "profile_prime_end": float(pack["profile_prime_end"]),
        "integral_grad": float(pack["integral_grad"]),
        "integral_mass": float(pack["integral_mass"]),
        "integral_cubic": float(pack["integral_cubic"]),
        "integral_quartic": float(pack["integral_quartic"]),
        "boundary_weighted_eom": float(pack["boundary_weighted_eom"]),
        "boundary_virial": float(pack["boundary_virial"]),
        "weighted_eom_residual": float(pack["weighted_eom_residual"]),
        "virial_residual": float(pack["virial_residual"]),
        "cubic_coeff_from_exact_weighted_eom": float(pack["cubic_coeff_from_exact_weighted_eom"]),
        "cubic_coeff_from_exact_virial": float(pack["cubic_coeff_from_exact_virial"]),
        "cubic_coeff_from_boundary_free_weighted_eom": float(
            pack["cubic_coeff_from_boundary_free_weighted_eom"]
        ),
        "cubic_coeff_from_boundary_free_virial": float(pack["cubic_coeff_from_boundary_free_virial"]),
        "boundary_weighted_eom_over_cubic": float(pack["boundary_weighted_eom_over_cubic"]),
        "boundary_virial_over_cubic": float(pack["boundary_virial_over_cubic"]),
        "quartic_over_cubic": float(pack["quartic_over_cubic"]),
        "route_c_exact_weighted_eom_identity_available_now": route_c_exact_weighted_eom_identity_available_now,
        "route_c_exact_virial_identity_available_now": route_c_exact_virial_identity_available_now,
        "route_c_exact_cubic_coefficient_recovered_now": route_c_exact_cubic_coefficient_recovered_now,
        "route_c_boundary_terms_negligible_now": route_c_boundary_terms_negligible_now,
        "route_c_boundary_free_virial_truncation_supported_now": route_c_boundary_free_virial_truncation_supported_now,
        "route_c_target_free_matching_law_bridge_available_now": route_c_target_free_matching_law_bridge_available_now,
        "route_c_negative_closeout_available_now": route_c_negative_closeout_available_now,
        "selected_extension_source_materialization_promoted_primary_now": (
            selected_extension_source_materialization_promoted_primary_now
        ),
        "selected_primary_completion_lane": (
            "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun"
        ),
        "selected_secondary_completion_lane": "updated_pack_scalar_proxy_route_c_archive_review",
        "selected_reserve_completion_lane": "updated_pack_scalar_proxy_direct_fourier_nlo_gap_review",
        "selected_next_generation_route": (
            "trial2_numeric_alpha_scalar_proxy_route_c_virial_gate"
        ),
        "recommended_next_route_or_none": "8.7.56.5451",
        "selected_followup_route": (
            "trial2_numeric_alpha_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun"
        ),
        "selected_followup_route_or_none": "8.7.56.5455",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5449",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5451",
                "followup_route": "8.7.56.5455",
            },
        },
        rows,
        summary,
        {
            "overall_status": "scalar_proxy_route_c_virial_audit_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} scalar-proxy Route-C virial audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# Function: run the audit when invoked as one CLI script.

if __name__ == "__main__":
    main()
