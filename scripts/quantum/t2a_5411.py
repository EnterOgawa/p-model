#!/usr/bin/env python3
"""Generate 8.7.56.5411-.5414 Route-B gate / route-refresh artifacts."""

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
        "8.7.56.5407-5410",
        "updated_pack_scalar_proxy_three_halves_first_principles_derivation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5411-5414"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor scalar-proxy "
    "Route-B gate / route refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_scalar_proxy_route_b_kappa_eff_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_three_halves_route_b_kappa_eff_negative_closeout_"
    "completed_route_a_primary_route_d_secondary_"
    "source_materialization_reserve_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_three_halves_route_b_kappa_eff_negative_closeout_"
    "completed_route_a_primary_route_d_secondary_"
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
    """Return formulas used by the Route-B gate refresh."""
    return {
        "gate_a": "Gate A = Route-B `kappa_eff` audit available now",
        "gate_b": "Gate B = Route-A EOM perturbation promoted next",
        "gate_c": "Gate C = source-materialization reopen required now",
    }


# Function: execute `.5411-.5414`.

def main() -> None:
    """Execute the Route-B gate / route refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary["route_b_asymptotic_algebra_available_now"]
        and prior_summary["route_b_exponent_mismatch_no_go_theorem_available_now"]
        and prior_summary["route_b_negative_closeout_available_now"]
    )
    gate_b = bool(gate_a and prior_summary["route_a_eom_perturbation_promoted_next_now"])
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
            "gate_a_updated_pack_exact_scalar_proxy_route_b_kappa_eff_audit_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack exact scalar-proxy Route-B kappa_eff audit available now",
            sign_base.truth(gate_a),
            "The Route-B-only audit reached one honest theorem-side verdict rather than another approximate tail replay.",
        ),
        sign_base.row(
            "gate_b_updated_pack_scalar_proxy_route_a_eom_perturbation_promoted_next",
            "pass" if gate_b else "reject",
            "gate B updated-pack scalar-proxy Route-A EOM perturbation promoted next",
            sign_base.truth(gate_b),
            "Because Route B closes negatively, the next honest primary lane is now Route A rather than a second Route-B variant.",
        ),
        sign_base.row(
            "gate_c_selected_extension_source_materialization_reopen_required_now",
            "reject",
            "gate C selected-extension source-materialization reopen required now",
            0.0,
            "The reserve source-materialization lane still does not become primary while theorem-side derivation routes remain open.",
        ),
        sign_base.row(
            "scalar_proxy_route_d_profile_moment_kept_secondary_now",
            "pass" if route_d_profile_moment_kept_secondary_now else "reject",
            "scalar-proxy Route-D profile moment kept secondary now",
            sign_base.truth(route_d_profile_moment_kept_secondary_now),
            "Profile moments remain the cleanest secondary cross-check after the Route-B tail reinterpretation is cut.",
        ),
        sign_base.row(
            "scalar_proxy_route_c_virial_kept_reserve_now",
            "pass" if route_c_virial_kept_reserve_now else "reject",
            "scalar-proxy Route-C virial kept reserve now",
            sign_base.truth(route_c_virial_kept_reserve_now),
            "The virial route stays reserve because it has not yet produced coefficient-level support comparable to Routes A/B/D.",
        ),
        sign_base.row(
            "selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now",
            "pass" if source_materialization_secondary_reserve_retained_now else "reject",
            "selected-extension independent extra-q-range source-materialization secondary reserve retained now",
            sign_base.truth(source_materialization_secondary_reserve_retained_now),
            "Source-materialization remains reserve-only while the scalar-proxy derivation lane still has a promoted primary route.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive route update required now",
            sign_base.truth(pack_update_required_now),
            "A substantive route update has happened: Route B is closed negatively and Route A is now the primary theorem lane.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_exact_over_m0": float(prior_summary["q_exact_over_m0"]),
        "q_star_over_m0": float(prior_summary["q_star_over_m0"]),
        "q_cubic_sqrt_over_m0": float(prior_summary["q_cubic_sqrt_over_m0"]),
        "delta_kappa_fit_abs": float(prior_summary["delta_kappa_fit_abs"]),
        "delta_kappa_required_abs": float(prior_summary["delta_kappa_required_abs"]),
        "delta_kappa_fit_rel_error_vs_required": float(prior_summary["delta_kappa_fit_rel_error_vs_required"]),
        "gate_a_updated_pack_exact_scalar_proxy_route_b_kappa_eff_audit_available_now": gate_a,
        "gate_b_updated_pack_scalar_proxy_route_a_eom_perturbation_promoted_next": gate_b,
        "gate_c_selected_extension_source_materialization_reopen_required_now": gate_c,
        "route_b_negative_closeout_available_now": bool(prior_summary["route_b_negative_closeout_available_now"]),
        "route_a_eom_perturbation_promoted_next_now": bool(prior_summary["route_a_eom_perturbation_promoted_next_now"]),
        "route_d_profile_moment_kept_secondary_now": route_d_profile_moment_kept_secondary_now,
        "route_c_virial_kept_reserve_now": route_c_virial_kept_reserve_now,
        "selected_primary_completion_lane": "updated_pack_scalar_proxy_route_a_eom_perturbation_audit",
        "selected_secondary_completion_lane": "updated_pack_scalar_proxy_route_d_profile_moment_audit",
        "selected_reserve_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun",
        "selected_next_generation_route": "trial2_numeric_alpha_scalar_proxy_route_a_eom_perturbation_audit",
        "recommended_next_route_or_none": "8.7.56.5415",
        "selected_followup_route": "trial2_numeric_alpha_scalar_proxy_route_a_eom_perturbation_gate",
        "selected_followup_route_or_none": "8.7.56.5419",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5413",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5415",
                "followup_route": "8.7.56.5419",
            },
        },
        rows,
        summary,
        {
            "overall_status": "scalar_proxy_route_b_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} scalar-proxy Route-B gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# Function: run the gate refresh when invoked as one CLI script.

if __name__ == "__main__":
    main()
