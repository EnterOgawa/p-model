#!/usr/bin/env python3
"""Generate 8.7.56.5399-.5402 profile-sensitive q_star correction artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.scalar_proxy_profile_sensitive_qstar_correction_backend import (
    build_scalar_proxy_profile_sensitive_qstar_correction_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5395-5398",
        "updated_pack_scalar_proxy_matching_law_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5399-5402"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor scalar-proxy "
    "profile-sensitive q_star correction audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_scalar_proxy_profile_sensitive_qstar_correction_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_matching_law_inventory_audited_profile_sensitive_q_star_"
    "correction_primary_effective_beta_shift_secondary_"
    "source_materialization_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_profile_sensitive_q_star_correction_diagnosed_three_halves_"
    "leading_law_primary_cubic_sqrt_direct_fourier_secondary_"
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


# Function: return formulas used by the correction audit.

def build_formulae() -> dict[str, str]:
    """Return formulas used by the profile-sensitive correction audit."""
    return {
        "linear_three_halves": "q_(3/2) = q_star * (1 - (3/2) * epsilon_beta)",
        "cubic_sqrt": "q_(sqrt3) = q_star * sqrt(1 - 3 * epsilon_beta)",
        "direct_fourier": "F(q) ~= F(q_star) + F'(q_star) dq + (1/2) F''(q_star) dq^2",
        "cubic_q_squared": "-delta_kappa^2 / q_star^2 ~= 3 * epsilon_beta",
    }


# Function: execute `.5399-.5402`.

def main() -> None:
    """Execute the profile-sensitive q_star correction audit."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_scalar_proxy_profile_sensitive_qstar_correction_pack()

    correction_audit_available_now = bool(
        prior_summary["gate_a_updated_pack_exact_scalar_proxy_matching_law_inventory_available_now"]
        and prior_summary["gate_b_updated_pack_scalar_proxy_profile_sensitive_q_star_correction_promoted_next"]
        and pack["old_blind_overlap_bridge_still_exact_now"]
    )
    three_halves_leading_law_available_now = bool(pack["three_halves_linear_law_available_now"])
    cubic_sqrt_leading_law_available_now = bool(pack["cubic_sqrt_leading_law_available_now"])
    practical_matching_law_available_now = bool(pack["practical_matching_law_available_now"])
    mexican_hat_cubic_route_supported_now = bool(pack["mexican_hat_cubic_route_supported_now"])
    direct_fourier_route_supported_now = bool(pack["direct_fourier_route_supported_now"])
    evanescent_tail_route_supported_now = bool(pack["evanescent_tail_route_supported_now"])
    virial_route_supported_now = bool(pack["virial_route_supported_now"])
    exact_three_halves_first_principles_derivation_available_now = bool(
        pack["exact_three_halves_first_principles_derivation_available_now"]
    )
    source_materialization_secondary_reserve_retained_now = bool(
        prior_summary["selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now"]
    )

    rows = [
        sign_base.row(
            "exact_scalar_proxy_profile_sensitive_q_star_correction_audit_available_now",
            "pass" if correction_audit_available_now else "reject",
            "exact scalar-proxy profile-sensitive q_star correction audit available now",
            sign_base.truth(correction_audit_available_now),
            "The retained scalar profile, the dense alpha(q) crossing, and the previous matching-law inventory now support one honest correction-law audit.",
        ),
        sign_base.row(
            "scalar_proxy_three_halves_linear_law_available_now",
            "pass" if three_halves_leading_law_available_now else "reject",
            "scalar-proxy three-halves linear law available now",
            sign_base.truth(three_halves_leading_law_available_now),
            "The simple linear law q = q_star * (1 - 3 epsilon_beta / 2) already lands within the retained q_exact window at the 1e-4 level.",
        ),
        sign_base.row(
            "scalar_proxy_cubic_sqrt_leading_law_available_now",
            "pass" if cubic_sqrt_leading_law_available_now else "reject",
            "scalar-proxy cubic-sqrt leading law available now",
            sign_base.truth(cubic_sqrt_leading_law_available_now),
            "The cleaner q^2 law q^2 = q_star^2 * (1 - 3 epsilon_beta) improves the leading reconstruction further and is the current primary algebraic front-runner.",
        ),
        sign_base.row(
            "scalar_proxy_practical_matching_law_available_now",
            "pass" if practical_matching_law_available_now else "reject",
            "scalar-proxy practical matching law available now",
            sign_base.truth(practical_matching_law_available_now),
            "The cubic-sqrt leading law already reproduces alpha within a few 1e-4 relative, so the remaining gap is now clearly NLO rather than an O(1) failure.",
        ),
        sign_base.row(
            "scalar_proxy_c1_abs_error_vs_three_halves_fixed",
            "pass",
            "scalar-proxy c1 absolute error versus -3/2 fixed",
            pack["c1_abs_error_vs_three_halves"],
            "This measures how far the fitted correction coefficient still sits from the exact rational candidate -3/2.",
        ),
        sign_base.row(
            "scalar_proxy_q_squared_correction_coeff_fit_fixed",
            "pass",
            "scalar-proxy q-squared correction coefficient fit fixed",
            pack["q_squared_correction_coeff_fit"],
            "In q^2 space the observed correction coefficient can be compared directly against the Mexican-hat cubic coefficient 3.",
        ),
        sign_base.row(
            "scalar_proxy_mexican_hat_cubic_route_supported_now",
            "pass" if mexican_hat_cubic_route_supported_now else "reject",
            "scalar-proxy Mexican-hat cubic route supported now",
            sign_base.truth(mexican_hat_cubic_route_supported_now),
            "The observed q^2 correction coefficient is already close to 3, so the cubic coefficient route survives as a leading-order explanation.",
        ),
        sign_base.row(
            "scalar_proxy_direct_fourier_route_supported_now",
            "pass" if direct_fourier_route_supported_now else "reject",
            "scalar-proxy direct Fourier route supported now",
            sign_base.truth(direct_fourier_route_supported_now),
            "A local quadratic expansion of F(q) around q_star reproduces the fitted coefficient almost exactly, so direct Fourier analysis remains a valid diagnostic route.",
        ),
        sign_base.row(
            "scalar_proxy_direct_fourier_route_target_dependent_now",
            "reject",
            "scalar-proxy direct Fourier route target dependent now",
            sign_base.truth(pack["direct_fourier_route_target_dependent_now"]),
            "The local Fourier reconstruction still uses the target crossing value, so it supports the law numerically but does not yet count as a target-free first-principles theorem.",
        ),
        sign_base.row(
            "scalar_proxy_evanescent_tail_route_supported_now",
            "pass" if evanescent_tail_route_supported_now else "reject",
            "scalar-proxy evanescent tail route supported now",
            sign_base.truth(evanescent_tail_route_supported_now),
            "The retained 1/r tail-fit coefficients drift strongly across cutoffs, so the tail route is not yet stable enough to claim the 3/2 law.",
        ),
        sign_base.row(
            "scalar_proxy_virial_route_supported_now",
            "pass" if virial_route_supported_now else "reject",
            "scalar-proxy virial route supported now",
            sign_base.truth(virial_route_supported_now),
            "Simple energy-component ratios do not return a clean 3/2 on the retained profile, so the virial route stays secondary.",
        ),
        sign_base.row(
            "scalar_proxy_exact_three_halves_first_principles_derivation_available_now",
            "pass" if exact_three_halves_first_principles_derivation_available_now else "reject",
            "scalar-proxy exact three-halves first-principles derivation available now",
            sign_base.truth(exact_three_halves_first_principles_derivation_available_now),
            "The leading law is now numerically supported, but a target-free frozen-action derivation is still missing.",
        ),
        sign_base.row(
            "selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now",
            "pass" if source_materialization_secondary_reserve_retained_now else "reject",
            "selected-extension independent extra-q-range source-materialization secondary reserve retained now",
            sign_base.truth(source_materialization_secondary_reserve_retained_now),
            "The source-materialization lane stays reserve-only because the scalar proxy now offers the cleaner path to closure.",
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
        "c1_fit": float(pack["c1_fit"]),
        "three_halves_linear_c1": float(pack["three_halves_linear_c1"]),
        "c1_abs_error_vs_three_halves": float(pack["c1_abs_error_vs_three_halves"]),
        "c1_rel_error_vs_three_halves": float(pack["c1_rel_error_vs_three_halves"]),
        "q_linear_three_halves_over_m0": float(pack["q_linear_three_halves_over_m0"]),
        "q_linear_three_halves_rel_error": float(pack["q_linear_three_halves_rel_error"]),
        "alpha_linear_three_halves": float(pack["alpha_linear_three_halves"]),
        "alpha_linear_three_halves_rel_error": float(pack["alpha_linear_three_halves_rel_error"]),
        "q_cubic_sqrt_over_m0": float(pack["q_cubic_sqrt_over_m0"]),
        "q_cubic_sqrt_rel_error": float(pack["q_cubic_sqrt_rel_error"]),
        "alpha_cubic_sqrt": float(pack["alpha_cubic_sqrt"]),
        "alpha_cubic_sqrt_rel_error": float(pack["alpha_cubic_sqrt_rel_error"]),
        "cubic_q_squared_coefficient": float(pack["cubic_q_squared_coefficient"]),
        "delta_kappa_squared_rel_observed": float(pack["delta_kappa_squared_rel_observed"]),
        "delta_kappa_squared_rel_from_cubic": float(pack["delta_kappa_squared_rel_from_cubic"]),
        "q_squared_correction_coeff_fit": float(pack["q_squared_correction_coeff_fit"]),
        "q_squared_correction_coeff_rel_error_vs_cubic": float(
            pack["q_squared_correction_coeff_rel_error_vs_cubic"]
        ),
        "F_q_star": float(pack["F_q_star"]),
        "F_prime_q_star": float(pack["F_prime_q_star"]),
        "F_double_prime_q_star": float(pack["F_double_prime_q_star"]),
        "target_form_factor": float(pack["target_form_factor"]),
        "dq_exact": float(pack["dq_exact"]),
        "dq_direct_linear": float(pack["dq_direct_linear"]),
        "dq_direct_quadratic": float(pack["dq_direct_quadratic"]),
        "c1_direct_linear": float(pack["c1_direct_linear"]),
        "c1_direct_quadratic": float(pack["c1_direct_quadratic"]),
        "c1_direct_quadratic_abs_error_vs_fit": float(pack["c1_direct_quadratic_abs_error_vs_fit"]),
        "tail_normalized_correction_rel_span": float(pack["tail_normalized_correction_rel_span"]),
        "tail_scaled_rel_std_max": float(pack["tail_scaled_rel_std_max"]),
        "gradient_over_mass_gap": float(pack["gradient_over_mass_gap"]),
        "cubic_over_mass_gap": float(pack["cubic_over_mass_gap"]),
        "cubic_over_gradient": float(pack["cubic_over_gradient"]),
        "cubic_over_gradient_plus_mass_gap": float(pack["cubic_over_gradient_plus_mass_gap"]),
        "gradient_plus_quartic_over_mass_gap": float(pack["gradient_plus_quartic_over_mass_gap"]),
        "virial_best_three_halves_abs_error": float(pack["virial_best_three_halves_abs_error"]),
        "exact_scalar_proxy_profile_sensitive_q_star_correction_audit_available_now": correction_audit_available_now,
        "three_halves_leading_law_available_now": three_halves_leading_law_available_now,
        "cubic_sqrt_leading_law_available_now": cubic_sqrt_leading_law_available_now,
        "practical_matching_law_available_now": practical_matching_law_available_now,
        "mexican_hat_cubic_route_supported_now": mexican_hat_cubic_route_supported_now,
        "direct_fourier_route_supported_now": direct_fourier_route_supported_now,
        "direct_fourier_route_target_dependent_now": bool(pack["direct_fourier_route_target_dependent_now"]),
        "evanescent_tail_route_supported_now": evanescent_tail_route_supported_now,
        "virial_route_supported_now": virial_route_supported_now,
        "exact_three_halves_first_principles_derivation_available_now": exact_three_halves_first_principles_derivation_available_now,
        "three_halves_nlo_gap_remaining_now": bool(pack["three_halves_nlo_gap_remaining_now"]),
        "selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now": source_materialization_secondary_reserve_retained_now,
        "selected_primary_completion_lane": "updated_pack_scalar_proxy_three_halves_first_principles_derivation_audit",
        "selected_secondary_completion_lane": "updated_pack_scalar_proxy_direct_fourier_nlo_gap_review",
        "selected_reserve_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun",
        "selected_next_generation_route": "trial2_numeric_alpha_scalar_proxy_profile_sensitive_q_star_correction_gate",
        "recommended_next_route_or_none": "8.7.56.5403",
        "selected_followup_route": "trial2_numeric_alpha_scalar_proxy_three_halves_first_principles_derivation_audit",
        "selected_followup_route_or_none": "8.7.56.5407",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5401",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5403",
                "followup_route": "8.7.56.5407",
            },
        },
        rows,
        summary,
        {
            "overall_status": "scalar_proxy_profile_sensitive_qstar_correction_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "formulae": build_formulae(),
            "tail_windows": pack["tail_windows"],
        },
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} scalar-proxy profile-sensitive q_star correction completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# Function: run the audit when invoked as one CLI script.

if __name__ == "__main__":
    main()
