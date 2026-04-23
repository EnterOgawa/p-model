#!/usr/bin/env python3
"""Generate 8.7.56.5439-.5442 Route-D profile-moment audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.scalar_proxy_route_d_profile_moment_backend import (
    build_scalar_proxy_route_d_profile_moment_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5435-5438",
        "updated_pack_scalar_proxy_route_a_exact_universal_twentyseven_derivation_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5439-5442"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor scalar-proxy "
    "Route-D profile moment audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_scalar_proxy_route_d_profile_moment_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_route_a_exact_universal_twentyseven_negative_closeout_completed_"
    "route_d_profile_moment_primary_source_materialization_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_route_d_profile_moment_low_order_truncation_no_go_theorem_derived_"
    "route_c_virial_primary_source_materialization_reserve_gate"
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


# Function: return formulas fixed by the Route-D profile-moment audit.

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the Route-D profile-moment audit."""
    return {
        "moment_series": (
            "F(q) = 1 - q^2<r^2>/6 + q^4<r^4>/120 - q^6<r^6>/5040 + q^8<r^8>/362880 + ..."
        ),
        "scaled_moment_family": "<r^(2n)> = epsilon_beta^(-n) * M_(2n,scaled)",
        "control_parameter": "z(q) = q^2<r^2>/6",
        "route_d_verdict": (
            "At q ~ q_star the control parameter is O(1), low-order moment terms are also O(1), "
            "and finite truncations do not approach the retained exact F(q)."
        ),
    }


# Function: execute `.5439-.5442`.

def main() -> None:
    """Execute the Route-D profile-moment audit."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_scalar_proxy_route_d_profile_moment_pack()

    route_d_profile_moment_audit_available_now = bool(
        prior_summary["gate_a_updated_pack_exact_scalar_proxy_route_a_exact_universal_twentyseven_no_go_available_now"]
        and prior_summary["gate_b_updated_pack_scalar_proxy_route_d_profile_moment_promoted_next"]
    )
    route_d_profile_moment_scaling_formula_available_now = bool(
        pack["route_d_profile_moment_scaling_formula_available_now"]
    )
    route_d_q_star_inside_small_q_control_domain_now = bool(
        pack["route_d_q_star_inside_small_q_control_domain_now"]
    )
    route_d_low_order_profile_moment_truncation_supported_now = bool(
        pack["route_d_low_order_profile_moment_truncation_supported_now"]
    )
    route_d_low_order_profile_moment_no_go_theorem_available_now = bool(
        pack["route_d_low_order_profile_moment_no_go_theorem_available_now"]
    )
    route_d_target_free_exact_derivation_available_now = bool(
        pack["route_d_target_free_exact_derivation_available_now"]
    )
    route_c_virial_promoted_next_now = bool(pack["route_c_virial_promoted_next_now"])
    source_materialization_kept_secondary_reserve_now = bool(
        pack["source_materialization_kept_secondary_reserve_now"]
    )

    rows = [
        sign_base.row(
            "exact_scalar_proxy_route_d_profile_moment_audit_available_now",
            "pass" if route_d_profile_moment_audit_available_now else "reject",
            "exact scalar-proxy Route-D profile moment audit available now",
            sign_base.truth(route_d_profile_moment_audit_available_now),
            "Route A exact twenty-seven has already closed negatively, so the direct profile-moment route is now the honest primary theorem-side branch.",
        ),
        sign_base.row(
            "scalar_proxy_route_d_profile_moment_scaling_formula_available_now",
            "pass" if route_d_profile_moment_scaling_formula_available_now else "reject",
            "scalar-proxy Route-D profile moment scaling formula available now",
            sign_base.truth(route_d_profile_moment_scaling_formula_available_now),
            "The retained profile moments follow one explicit epsilon_beta scaling family inherited from the scaled scalar ground-state profile.",
        ),
        sign_base.row(
            "scalar_proxy_route_d_q_star_inside_small_q_control_domain_now",
            "pass" if route_d_q_star_inside_small_q_control_domain_now else "reject",
            "scalar-proxy Route-D q_star inside small-q control domain now",
            sign_base.truth(route_d_q_star_inside_small_q_control_domain_now),
            "A low-order moment truncation is only honest when z(q)=q^2<r^2>/6 is small; the retained q_star point fails that condition.",
        ),
        sign_base.row(
            "scalar_proxy_route_d_low_order_profile_moment_truncation_supported_now",
            "pass" if route_d_low_order_profile_moment_truncation_supported_now else "reject",
            "scalar-proxy Route-D low-order profile-moment truncation supported now",
            sign_base.truth(route_d_low_order_profile_moment_truncation_supported_now),
            "Finite q^2/q^4/q^6/q^8 moment truncations are tested directly against the retained exact form factor and only pass if they stay close without importing new parameters.",
        ),
        sign_base.row(
            "scalar_proxy_route_d_low_order_profile_moment_no_go_theorem_available_now",
            "pass" if route_d_low_order_profile_moment_no_go_theorem_available_now else "reject",
            "scalar-proxy Route-D low-order profile-moment no-go theorem available now",
            sign_base.truth(route_d_low_order_profile_moment_no_go_theorem_available_now),
            "The retained q_star and q_exact points sit outside the controlled low-q domain, and low-order truncations do not converge to the exact retained form factor there.",
        ),
        sign_base.row(
            "scalar_proxy_route_d_target_free_exact_derivation_available_now",
            "pass" if route_d_target_free_exact_derivation_available_now else "reject",
            "scalar-proxy Route-D target-free exact derivation available now",
            sign_base.truth(route_d_target_free_exact_derivation_available_now),
            "This remains false unless the profile-moment route alone derives the matching law without importing the retained crossing.",
        ),
        sign_base.row(
            "scalar_proxy_route_c_virial_promoted_next_now",
            "pass" if route_c_virial_promoted_next_now else "reject",
            "scalar-proxy Route-C virial promoted next now",
            sign_base.truth(route_c_virial_promoted_next_now),
            "With the direct profile-moment truncation closed negatively, the remaining theorem-side branch is Route C virial.",
        ),
        sign_base.row(
            "selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now",
            "pass" if source_materialization_kept_secondary_reserve_now else "reject",
            "selected-extension independent extra-q-range source-materialization secondary reserve retained now",
            sign_base.truth(source_materialization_kept_secondary_reserve_now),
            "Source-materialization stays reserve-only while Route C still exists as one theorem-side alternative.",
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
        "moment_r2": float(pack["moment_r2"]),
        "moment_r4": float(pack["moment_r4"]),
        "moment_r6": float(pack["moment_r6"]),
        "moment_r8": float(pack["moment_r8"]),
        "scaled_moment_r2": float(pack["scaled_moment_r2"]),
        "scaled_moment_r4": float(pack["scaled_moment_r4"]),
        "scaled_moment_r6": float(pack["scaled_moment_r6"]),
        "scaled_moment_r8": float(pack["scaled_moment_r8"]),
        "q_star_control_parameter_q2_abs": float(pack["q_star_control_parameter_q2_abs"]),
        "q_exact_control_parameter_q2_abs": float(pack["q_exact_control_parameter_q2_abs"]),
        "q_star_largest_term_abs": float(pack["q_star_largest_term_abs"]),
        "q_exact_largest_term_abs": float(pack["q_exact_largest_term_abs"]),
        "q_star_best_truncation_abs_error": float(pack["q_star_best_truncation_abs_error"]),
        "q_exact_best_truncation_abs_error": float(pack["q_exact_best_truncation_abs_error"]),
        "q_cubic_sqrt_best_truncation_abs_error": float(pack["q_cubic_sqrt_best_truncation_abs_error"]),
        "route_d_profile_moment_scaling_formula_available_now": route_d_profile_moment_scaling_formula_available_now,
        "route_d_q_star_inside_small_q_control_domain_now": route_d_q_star_inside_small_q_control_domain_now,
        "route_d_q_exact_inside_small_q_control_domain_now": bool(pack["route_d_q_exact_inside_small_q_control_domain_now"]),
        "route_d_low_order_profile_moment_truncation_supported_now": route_d_low_order_profile_moment_truncation_supported_now,
        "route_d_low_order_profile_moment_no_go_theorem_available_now": route_d_low_order_profile_moment_no_go_theorem_available_now,
        "route_d_target_free_exact_derivation_available_now": route_d_target_free_exact_derivation_available_now,
        "route_c_virial_promoted_next_now": route_c_virial_promoted_next_now,
        "selected_primary_completion_lane": "updated_pack_scalar_proxy_route_c_virial_audit",
        "selected_secondary_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun",
        "selected_reserve_completion_lane": "updated_pack_scalar_proxy_route_d_profile_moment_resummation_review",
        "selected_next_generation_route": "trial2_numeric_alpha_scalar_proxy_route_d_profile_moment_gate",
        "recommended_next_route_or_none": "8.7.56.5443",
        "selected_followup_route": "trial2_numeric_alpha_scalar_proxy_route_c_virial_audit",
        "selected_followup_route_or_none": "8.7.56.5447",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5441",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5443",
                "followup_route": "8.7.56.5447",
            },
        },
        rows,
        summary,
        {
            "overall_status": "scalar_proxy_route_d_profile_moment_audit_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} scalar-proxy Route-D profile moment audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# Function: run the audit when invoked as one CLI script.

if __name__ == "__main__":
    main()
