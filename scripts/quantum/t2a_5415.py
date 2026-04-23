#!/usr/bin/env python3
"""Generate 8.7.56.5415-.5418 Route-A EOM perturbation audit artifacts."""

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
        "8.7.56.5411-5414",
        "updated_pack_scalar_proxy_route_b_kappa_eff_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5415-5418"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor scalar-proxy "
    "Route-A EOM perturbation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_scalar_proxy_route_a_eom_perturbation_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_three_halves_route_b_kappa_eff_negative_closeout_"
    "completed_route_a_primary_route_d_secondary_"
    "source_materialization_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_route_a_lo_cubic_scaleout_no_go_theorem_derived_"
    "nlo_perturbation_primary_route_d_secondary_"
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


# Function: return formulas used by the Route-A audit.

def build_formulae() -> dict[str, str]:
    """Return the Route-A formulas fixed by the audit."""
    return {
        "scaled_exact_equation": (
            "With epsilon_beta = 1 - beta1^2, kappa = sqrt(epsilon_beta), "
            "xi = kappa x, y = epsilon_beta u, the exact reduced equation is "
            "u'' + (2/xi) u' - u + 3 u^2 + epsilon_beta u^3 = 0"
        ),
        "lo_equation": "u'' + (2/xi) u' - u + 3 u^2 = 0",
        "generic_cubic_scaleout": (
            "For u'' + (2/xi) u' - u + g3 u^2 = 0, set u = v / g3 to get "
            "v'' + (2/xi) v' - v + v^2 = 0"
        ),
        "normalized_lo_density": "rho_hat_LO(xi) = u(xi)^2 xi^2 / integral[u(xi)^2 xi^2 dxi]",
        "normalized_lo_form_factor": (
            "F_hat_LO(q_hat) = integral[rho_hat_LO(xi) sinc(q_hat xi) dxi]"
        ),
        "route_a_verdict": (
            "Because rho_hat_LO and F_hat_LO are invariant under u -> g3 u, "
            "the LO normalized overlap cannot target-free derive cubic coefficient 3"
        ),
    }


# Function: execute `.5415-.5418`.

def main() -> None:
    """Execute the Route-A EOM perturbation audit."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_scalar_proxy_route_a_eom_perturbation_pack()

    route_a_audit_available_now = bool(
        prior_summary["gate_a_updated_pack_exact_scalar_proxy_route_b_kappa_eff_audit_available_now"]
        and prior_summary["gate_b_updated_pack_scalar_proxy_route_a_eom_perturbation_promoted_next"]
    )
    route_a_lo_cubic_scaleout_no_go_theorem_available_now = bool(
        pack["route_a_lo_cubic_scaleout_no_go_theorem_available_now"]
    )
    route_a_nlo_perturbation_equation_available_now = bool(
        pack["route_a_nlo_perturbation_equation_available_now"]
    )
    route_a_nlo_perturbation_promoted_next_now = bool(
        pack["route_a_nlo_perturbation_promoted_next_now"]
    )
    route_d_profile_moment_kept_secondary_now = bool(pack["route_d_profile_moment_kept_secondary_now"])
    route_c_virial_kept_reserve_now = bool(pack["route_c_virial_kept_reserve_now"])
    source_materialization_secondary_reserve_retained_now = bool(
        not prior_summary["gate_c_selected_extension_source_materialization_reopen_required_now"]
    )

    rows = [
        sign_base.row(
            "exact_scalar_proxy_route_a_eom_perturbation_audit_available_now",
            "pass" if route_a_audit_available_now else "reject",
            "exact scalar-proxy Route-A EOM perturbation audit available now",
            sign_base.truth(route_a_audit_available_now),
            "Route B closed negatively, so one theorem-side Route-A audit is now the primary honest branch.",
        ),
        sign_base.row(
            "scalar_proxy_route_a_scaled_reduced_eom_available_now",
            "pass" if pack["route_a_scaled_reduced_eom_available_now"] else "reject",
            "scalar-proxy Route-A scaled reduced EOM available now",
            sign_base.truth(pack["route_a_scaled_reduced_eom_available_now"]),
            "The retained shooting equation is rewritten exactly as one epsilon-expanded reduced equation without any target input.",
        ),
        sign_base.row(
            "scalar_proxy_route_a_lo_quartic_suppression_supported_now",
            "pass" if pack["route_a_lo_quartic_suppression_supported_now"] else "reject",
            "scalar-proxy Route-A LO quartic suppression supported now",
            sign_base.truth(pack["route_a_lo_quartic_suppression_supported_now"]),
            "On the retained profile the quartic remainder stays parametrically smaller than the LO cubic term, so an LO Route-A theorem is honest.",
        ),
        sign_base.row(
            "scalar_proxy_route_a_lo_generic_cubic_scaleout_theorem_available_now",
            "pass" if pack["route_a_lo_generic_cubic_scaleout_theorem_available_now"] else "reject",
            "scalar-proxy Route-A LO generic cubic scaleout theorem available now",
            sign_base.truth(pack["route_a_lo_generic_cubic_scaleout_theorem_available_now"]),
            "After amplitude rescaling the generic cubic coefficient g3 scales out of the LO reduced equation itself.",
        ),
        sign_base.row(
            "scalar_proxy_route_a_lo_normalized_overlap_cubic_independence_theorem_available_now",
            "pass" if pack["route_a_lo_normalized_overlap_cubic_independence_theorem_available_now"] else "reject",
            "scalar-proxy Route-A LO normalized overlap cubic independence theorem available now",
            sign_base.truth(pack["route_a_lo_normalized_overlap_cubic_independence_theorem_available_now"]),
            "The LO normalized density and LO normalized form factor are invariant under amplitude scaleout, so the cubic coefficient cannot enter at LO through normalized overlap alone.",
        ),
        sign_base.row(
            "scalar_proxy_route_a_lo_target_free_three_coefficient_derivation_available_now",
            "pass" if pack["route_a_lo_target_free_three_coefficient_derivation_available_now"] else "reject",
            "scalar-proxy Route-A LO target-free three coefficient derivation available now",
            sign_base.truth(pack["route_a_lo_target_free_three_coefficient_derivation_available_now"]),
            "This remains false unless the LO Route-A algebra alone produces the Mexican-hat cubic coefficient 3 in the matching law without target input.",
        ),
        sign_base.row(
            "scalar_proxy_route_a_lo_cubic_scaleout_no_go_theorem_available_now",
            "pass" if route_a_lo_cubic_scaleout_no_go_theorem_available_now else "reject",
            "scalar-proxy Route-A LO cubic scaleout no-go theorem available now",
            sign_base.truth(route_a_lo_cubic_scaleout_no_go_theorem_available_now),
            "LO Route A is honest but insufficient: the cubic coefficient scales out of the normalized overlap, so the exact three-halves coefficient cannot close at LO.",
        ),
        sign_base.row(
            "scalar_proxy_route_a_nlo_perturbation_equation_available_now",
            "pass" if route_a_nlo_perturbation_equation_available_now else "reject",
            "scalar-proxy Route-A NLO perturbation equation available now",
            sign_base.truth(route_a_nlo_perturbation_equation_available_now),
            "The reduced equation already isolates the honest NLO source epsilon_beta u^3 for the next perturbative step.",
        ),
        sign_base.row(
            "scalar_proxy_route_a_nlo_perturbation_promoted_next_now",
            "pass" if route_a_nlo_perturbation_promoted_next_now else "reject",
            "scalar-proxy Route-A NLO perturbation promoted next now",
            sign_base.truth(route_a_nlo_perturbation_promoted_next_now),
            "Because LO closes negatively, the next honest primary blocker is the NLO Route-A perturbation equation rather than another LO replay.",
        ),
        sign_base.row(
            "scalar_proxy_route_d_profile_moment_kept_secondary_now",
            "pass" if route_d_profile_moment_kept_secondary_now else "reject",
            "scalar-proxy Route-D profile moment kept secondary now",
            sign_base.truth(route_d_profile_moment_kept_secondary_now),
            "Route D remains the best secondary cross-check because it already matched the fitted coefficient numerically.",
        ),
        sign_base.row(
            "selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now",
            "pass" if source_materialization_secondary_reserve_retained_now else "reject",
            "selected-extension independent extra-q-range source-materialization secondary reserve retained now",
            sign_base.truth(source_materialization_secondary_reserve_retained_now),
            "The source-materialization lane stays reserve-only while theorem-side derivation routes A and D still have honest work left.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta1": float(pack["beta1"]),
        "epsilon_beta": float(pack["epsilon_beta"]),
        "kappa": float(pack["kappa"]),
        "q_exact_over_m0": float(pack["q_exact_over_m0"]),
        "q_star_over_m0": float(pack["q_star_over_m0"]),
        "q_cubic_sqrt_over_m0": float(pack["q_cubic_sqrt_over_m0"]),
        "q_squared_correction_coeff_fit": float(pack["q_squared_correction_coeff_fit"]),
        "scaled_center_amplitude": float(pack["scaled_center_amplitude"]),
        "scaled_u_abs_max": float(pack["scaled_u_abs_max"]),
        "scaled_eom_exact_residual_max_abs": float(pack["scaled_eom_exact_residual_max_abs"]),
        "scaled_eom_exact_residual_rms": float(pack["scaled_eom_exact_residual_rms"]),
        "scaled_eom_lo_residual_plus_expected_nlo_max_abs": float(
            pack["scaled_eom_lo_residual_plus_expected_nlo_max_abs"]
        ),
        "scaled_eom_lo_residual_plus_expected_nlo_rms": float(
            pack["scaled_eom_lo_residual_plus_expected_nlo_rms"]
        ),
        "quartic_to_cubic_ratio_center": float(pack["quartic_to_cubic_ratio_center"]),
        "quartic_to_cubic_ratio_max": float(pack["quartic_to_cubic_ratio_max"]),
        "quartic_to_cubic_ratio_weighted_mean": float(pack["quartic_to_cubic_ratio_weighted_mean"]),
        "quartic_to_linear_ratio_center": float(pack["quartic_to_linear_ratio_center"]),
        "quartic_to_linear_ratio_max": float(pack["quartic_to_linear_ratio_max"]),
        "quartic_to_linear_ratio_weighted_mean": float(pack["quartic_to_linear_ratio_weighted_mean"]),
        "lo_scaleout_density_diff_max_abs": float(pack["lo_scaleout_density_diff_max_abs"]),
        "lo_scaleout_form_factor_diff_max_abs": float(pack["lo_scaleout_form_factor_diff_max_abs"]),
        "route_a_scaled_reduced_eom_available_now": bool(pack["route_a_scaled_reduced_eom_available_now"]),
        "route_a_lo_quartic_suppression_supported_now": bool(pack["route_a_lo_quartic_suppression_supported_now"]),
        "route_a_lo_generic_cubic_scaleout_theorem_available_now": bool(
            pack["route_a_lo_generic_cubic_scaleout_theorem_available_now"]
        ),
        "route_a_lo_normalized_overlap_cubic_independence_theorem_available_now": bool(
            pack["route_a_lo_normalized_overlap_cubic_independence_theorem_available_now"]
        ),
        "route_a_lo_target_free_three_coefficient_derivation_available_now": bool(
            pack["route_a_lo_target_free_three_coefficient_derivation_available_now"]
        ),
        "route_a_lo_cubic_scaleout_no_go_theorem_available_now": route_a_lo_cubic_scaleout_no_go_theorem_available_now,
        "route_a_nlo_perturbation_equation_available_now": route_a_nlo_perturbation_equation_available_now,
        "route_a_nlo_perturbation_promoted_next_now": route_a_nlo_perturbation_promoted_next_now,
        "route_d_profile_moment_kept_secondary_now": route_d_profile_moment_kept_secondary_now,
        "route_c_virial_kept_reserve_now": route_c_virial_kept_reserve_now,
        "selected_primary_completion_lane": "updated_pack_scalar_proxy_route_a_nlo_perturbation_audit",
        "selected_secondary_completion_lane": "updated_pack_scalar_proxy_route_d_profile_moment_audit",
        "selected_reserve_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun",
        "selected_next_generation_route": "trial2_numeric_alpha_scalar_proxy_route_a_nlo_perturbation_audit",
        "recommended_next_route_or_none": "8.7.56.5419",
        "selected_followup_route": "trial2_numeric_alpha_scalar_proxy_route_a_nlo_perturbation_gate",
        "selected_followup_route_or_none": "8.7.56.5423",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5417",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5419",
                "followup_route": "8.7.56.5423",
            },
        },
        rows,
        summary,
        {
            "overall_status": "scalar_proxy_route_a_eom_perturbation_audit_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} scalar-proxy Route-A audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# Function: run the audit when invoked as one CLI script.

if __name__ == "__main__":
    main()
