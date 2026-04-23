#!/usr/bin/env python3
"""Generate 8.7.56.5407-.5410 Route-B `kappa_eff` derivation artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.scalar_proxy_three_halves_first_principles_derivation_backend import (
    build_scalar_proxy_three_halves_route_b_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5403-5406",
        "updated_pack_scalar_proxy_profile_sensitive_qstar_correction_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5407-5410"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor scalar-proxy "
    "three-halves first-principles derivation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_scalar_proxy_three_halves_first_principles_derivation_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_profile_sensitive_q_star_correction_audited_three_halves_"
    "leading_law_primary_exact_derivation_secondary_"
    "source_materialization_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_three_halves_route_b_kappa_eff_negative_closeout_"
    "completed_route_a_primary_route_d_secondary_"
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


# Function: return formulas used by the Route-B audit.

def build_formulae() -> dict[str, str]:
    """Return the Route-B formulas fixed by the audit."""
    return {
        "u_reduction": "u(x) = x y(x), so u'' - kappa^2 u + 3 u^2 / x + u^3 / x^2 = 0",
        "free_tail": "u_0(x) = A exp(-kappa x), y_0(x) = A exp(-kappa x) / x",
        "linearized_route_b": "delta u'' - kappa^2 delta u = -3 A^2 exp(-2 kappa x) / x",
        "basis_response": "(d^2/dx^2 - kappa^2)[exp(-2 kappa x)/x] = exp(-2 kappa x) * (3 kappa^2 / x + 4 kappa / x^2 + 2 / x^3)",
        "particular_solution": "delta u_p(x) = -(A^2 / kappa^2) exp(-2 kappa x) / x + O(exp(-2 kappa x) / x^2)",
        "tail_correction": "delta y_p(x) = -(A^2 / kappa^2) exp(-2 kappa x) / x^2 + O(exp(-2 kappa x) / x^3)",
        "constant_shift_contrast": "y_shift(x) = A exp(-(kappa + delta_kappa) x) / x = y_0(x) - A delta_kappa exp(-kappa x) + O(delta_kappa^2)",
    }


# Function: execute `.5407-.5410`.

def main() -> None:
    """Execute the Route-B `kappa_eff` derivation audit."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_scalar_proxy_three_halves_route_b_pack()

    route_b_audit_available_now = bool(
        prior_summary["gate_a_updated_pack_exact_scalar_proxy_profile_sensitive_q_star_correction_audit_available_now"]
        and prior_summary["gate_b_updated_pack_scalar_proxy_three_halves_first_principles_derivation_promoted_next"]
    )
    route_b_negative_closeout_available_now = bool(pack["route_b_negative_closeout_available_now"])
    route_a_eom_perturbation_promoted_next_now = bool(pack["route_a_eom_perturbation_promoted_next_now"])
    route_d_profile_moment_kept_secondary_now = bool(pack["route_d_profile_moment_kept_secondary_now"])
    route_c_virial_kept_reserve_now = bool(pack["route_c_virial_kept_reserve_now"])
    source_materialization_secondary_reserve_retained_now = bool(
        not prior_summary["gate_c_selected_extension_source_materialization_reopen_required_now"]
    )

    rows = [
        sign_base.row(
            "exact_scalar_proxy_three_halves_route_b_kappa_eff_audit_available_now",
            "pass" if route_b_audit_available_now else "reject",
            "exact scalar-proxy three-halves Route-B kappa_eff audit available now",
            sign_base.truth(route_b_audit_available_now),
            "The previous three-halves gate now licenses one Route-B-only `kappa_eff` audit without reopening source-materialization or side routes.",
        ),
        sign_base.row(
            "scalar_proxy_route_b_asymptotic_algebra_available_now",
            "pass" if pack["route_b_asymptotic_algebra_available_now"] else "reject",
            "scalar-proxy Route-B asymptotic algebra available now",
            sign_base.truth(pack["route_b_asymptotic_algebra_available_now"]),
            "The large-x reduction to `u = x y` and the inhomogeneous tail equation are fixed analytically from the frozen-action shooting equation.",
        ),
        sign_base.row(
            "scalar_proxy_route_b_particular_tail_cross_check_supported_now",
            "pass" if pack["route_b_particular_tail_cross_check_supported_now"] else "reject",
            "scalar-proxy Route-B particular tail cross-check supported now",
            sign_base.truth(pack["route_b_particular_tail_cross_check_supported_now"]),
            "The retained tail residual is compared directly against the Route-B particular basis `exp(-2 kappa x) / x^2` rather than against a guessed `kappa_eff` shift.",
        ),
        sign_base.row(
            "scalar_proxy_route_b_particular_coeff_exact_fixed",
            "pass",
            "scalar-proxy Route-B particular coefficient exact fixed",
            pack["particular_coeff_exact"],
            "Matching the leading `exp(-2 kappa x) / x` source and operator response fixes the particular coefficient to `-A^2 / kappa^2`.",
        ),
        sign_base.row(
            "scalar_proxy_route_b_constant_kappa_shift_supported_now",
            "pass" if pack["route_b_constant_kappa_shift_supported_now"] else "reject",
            "scalar-proxy Route-B constant kappa shift supported now",
            sign_base.truth(pack["route_b_constant_kappa_shift_supported_now"]),
            "If Route B were enough by itself, the late-tail fit would show one stable `kappa_eff = kappa + delta_kappa` matching the required cubic-sqrt shift.",
        ),
        sign_base.row(
            "scalar_proxy_route_b_exponent_mismatch_no_go_theorem_available_now",
            "pass" if pack["route_b_exponent_mismatch_no_go_theorem_available_now"] else "reject",
            "scalar-proxy Route-B exponent-mismatch no-go theorem available now",
            sign_base.truth(pack["route_b_exponent_mismatch_no_go_theorem_available_now"]),
            "Route B produces `delta y_p ~ exp(-2 kappa x) / x^2`, while a constant `kappa_eff` shift would require an `exp(-kappa x)` correction. The asymptotic exponents do not match.",
        ),
        sign_base.row(
            "scalar_proxy_route_b_delta_kappa_fit_abs_fixed",
            "pass",
            "scalar-proxy Route-B delta kappa fit absolute value fixed",
            pack["delta_kappa_fit_abs"],
            "This is the best late-tail `kappa_eff - kappa` fit from the retained profile and can be compared directly against the cubic-sqrt-required shift.",
        ),
        sign_base.row(
            "scalar_proxy_route_b_delta_kappa_required_abs_fixed",
            "pass",
            "scalar-proxy Route-B delta kappa required absolute value fixed",
            pack["delta_kappa_required_abs"],
            "This is the constant shift one would need if the cubic-sqrt law really arose from one `kappa_eff = kappa + delta_kappa` reinterpretation.",
        ),
        sign_base.row(
            "scalar_proxy_route_b_target_free_three_halves_derivation_available_now",
            "pass" if pack["route_b_target_free_three_halves_derivation_available_now"] else "reject",
            "scalar-proxy Route-B target-free three-halves derivation available now",
            sign_base.truth(pack["route_b_target_free_three_halves_derivation_available_now"]),
            "This remains false unless Route B alone turns the cubic coefficient into the full three-halves / cubic-sqrt matching law without extra target input.",
        ),
        sign_base.row(
            "scalar_proxy_route_b_negative_closeout_available_now",
            "pass" if route_b_negative_closeout_available_now else "reject",
            "scalar-proxy Route-B negative closeout available now",
            sign_base.truth(route_b_negative_closeout_available_now),
            "The present Route-B-only audit supports one no-go: the cubic term does generate a subleading tail correction, but not one constant `kappa_eff` shift of the required form.",
        ),
        sign_base.row(
            "scalar_proxy_route_a_eom_perturbation_promoted_next_now",
            "pass" if route_a_eom_perturbation_promoted_next_now else "reject",
            "scalar-proxy Route-A EOM perturbation promoted next now",
            sign_base.truth(route_a_eom_perturbation_promoted_next_now),
            "Because Route B closes negatively, the next honest primary lane is the systematic EOM-perturbation route rather than another tail reinterpretation.",
        ),
        sign_base.row(
            "scalar_proxy_route_d_profile_moment_kept_secondary_now",
            "pass" if route_d_profile_moment_kept_secondary_now else "reject",
            "scalar-proxy Route-D profile moment kept secondary now",
            sign_base.truth(route_d_profile_moment_kept_secondary_now),
            "Profile moments remain the best secondary cross-check because direct Fourier evidence was already numerically strongest there.",
        ),
        sign_base.row(
            "selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now",
            "pass" if source_materialization_secondary_reserve_retained_now else "reject",
            "selected-extension independent extra-q-range source-materialization secondary reserve retained now",
            sign_base.truth(source_materialization_secondary_reserve_retained_now),
            "The source-materialization lane stays reserve-only because the scalar-proxy derivation lane still has active theorem-side work left.",
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
        "q_squared_residual_vs_cubic_sqrt": float(pack["q_squared_residual_vs_cubic_sqrt"]),
        "best_amplitude_cutoff": float(pack["best_amplitude_cutoff"]),
        "amplitude_fit": float(pack["amplitude_fit"]),
        "source_leading_prefactor": float(pack["source_leading_prefactor"]),
        "operator_basis_leading_prefactor": float(pack["operator_basis_leading_prefactor"]),
        "particular_coeff_exact": float(pack["particular_coeff_exact"]),
        "best_particular_cutoff": float(pack["best_particular_cutoff"]),
        "best_particular_coeff_fit": float(pack["best_particular_coeff_fit"]),
        "best_particular_coeff_rel_std": float(pack["best_particular_coeff_rel_std"]),
        "best_particular_coeff_rel_error_vs_prediction": float(
            pack["best_particular_coeff_rel_error_vs_prediction"]
        ),
        "best_kappa_fit_cutoff": float(pack["best_kappa_fit_cutoff"]),
        "kappa_eff_fit": float(pack["kappa_eff_fit"]),
        "best_kappa_log_fit_std": float(pack["best_kappa_log_fit_std"]),
        "delta_kappa_fit_abs": float(pack["delta_kappa_fit_abs"]),
        "delta_kappa_fit_rel": float(pack["delta_kappa_fit_rel"]),
        "delta_kappa_required_abs": float(pack["delta_kappa_required_abs"]),
        "delta_kappa_required_rel": float(pack["delta_kappa_required_rel"]),
        "delta_kappa_fit_rel_error_vs_required": float(pack["delta_kappa_fit_rel_error_vs_required"]),
        "route_b_asymptotic_algebra_available_now": bool(pack["route_b_asymptotic_algebra_available_now"]),
        "route_b_particular_tail_cross_check_supported_now": bool(
            pack["route_b_particular_tail_cross_check_supported_now"]
        ),
        "route_b_constant_kappa_shift_supported_now": bool(pack["route_b_constant_kappa_shift_supported_now"]),
        "route_b_exponent_mismatch_no_go_theorem_available_now": bool(
            pack["route_b_exponent_mismatch_no_go_theorem_available_now"]
        ),
        "route_b_target_free_three_halves_derivation_available_now": bool(
            pack["route_b_target_free_three_halves_derivation_available_now"]
        ),
        "route_b_negative_closeout_available_now": route_b_negative_closeout_available_now,
        "route_a_eom_perturbation_promoted_next_now": route_a_eom_perturbation_promoted_next_now,
        "route_d_profile_moment_kept_secondary_now": route_d_profile_moment_kept_secondary_now,
        "route_c_virial_kept_reserve_now": route_c_virial_kept_reserve_now,
        "selected_primary_completion_lane": "updated_pack_scalar_proxy_route_a_eom_perturbation_audit",
        "selected_secondary_completion_lane": "updated_pack_scalar_proxy_route_d_profile_moment_audit",
        "selected_reserve_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun",
        "selected_next_generation_route": "trial2_numeric_alpha_scalar_proxy_route_a_eom_perturbation_audit",
        "recommended_next_route_or_none": "8.7.56.5411",
        "selected_followup_route": "trial2_numeric_alpha_scalar_proxy_route_a_eom_perturbation_gate",
        "selected_followup_route_or_none": "8.7.56.5415",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5409",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5411",
                "followup_route": "8.7.56.5415",
            },
        },
        rows,
        summary,
        {
            "overall_status": "scalar_proxy_route_b_kappa_eff_audit_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} scalar-proxy Route-B kappa_eff audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# Function: run the audit when invoked as one CLI script.

if __name__ == "__main__":
    main()
