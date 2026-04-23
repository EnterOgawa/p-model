#!/usr/bin/env python3
"""Generate 8.7.56.5403-.5406 profile-sensitive q_star correction gate artifacts."""

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
        "8.7.56.5399-5402",
        "updated_pack_scalar_proxy_profile_sensitive_qstar_correction_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5403-5406"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor scalar-proxy "
    "profile-sensitive q_star correction gate / exact-overlap refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_scalar_proxy_profile_sensitive_qstar_correction_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_profile_sensitive_q_star_correction_diagnosed_three_halves_"
    "leading_law_primary_cubic_sqrt_direct_fourier_secondary_"
    "source_materialization_reserve_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_profile_sensitive_q_star_correction_audited_three_halves_"
    "leading_law_primary_exact_derivation_secondary_"
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
    """Return formulas used by the correction gate refresh."""
    return {
        "gate_a": "Gate A = retained three-halves leading-law audit available now",
        "gate_b": "Gate B = exact three-halves derivation promoted next",
        "gate_c": "Gate C = reserve source-materialization reopen required now",
    }


# Function: execute `.5403-.5406`.

def main() -> None:
    """Execute the profile-sensitive q_star correction gate / refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary["exact_scalar_proxy_profile_sensitive_q_star_correction_audit_available_now"]
        and prior_summary["three_halves_leading_law_available_now"]
        and prior_summary["cubic_sqrt_leading_law_available_now"]
        and prior_summary["mexican_hat_cubic_route_supported_now"]
        and prior_summary["direct_fourier_route_supported_now"]
    )
    gate_b = bool(gate_a and not prior_summary["exact_three_halves_first_principles_derivation_available_now"])
    gate_c = False
    practical_matching_law_available_now = bool(prior_summary["practical_matching_law_available_now"])
    evanescent_tail_route_supported_now = bool(prior_summary["evanescent_tail_route_supported_now"])
    virial_route_supported_now = bool(prior_summary["virial_route_supported_now"])
    source_materialization_secondary_reserve_retained_now = bool(
        prior_summary["selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now"]
    )
    pack_update_required_now = bool(gate_b)

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_scalar_proxy_profile_sensitive_q_star_correction_audit_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack exact scalar-proxy profile-sensitive q_star correction audit available now",
            sign_base.truth(gate_a),
            "The retained scalar profile now supports a concrete leading-law audit around c = -3/2 and the cubic q^2 route.",
        ),
        sign_base.row(
            "gate_b_updated_pack_scalar_proxy_three_halves_first_principles_derivation_promoted_next",
            "pass" if gate_b else "reject",
            "gate B updated-pack scalar-proxy three-halves first-principles derivation promoted next",
            sign_base.truth(gate_b),
            "The next honest blocker is no longer generic correction-family fitting, but a target-free derivation of the three-halves / cubic-sqrt law.",
        ),
        sign_base.row(
            "gate_c_selected_extension_source_materialization_reopen_required_now",
            "reject",
            "gate C selected-extension source-materialization reopen required now",
            0.0,
            "The reserve source-materialization lane remains secondary because the scalar proxy now has a concrete leading law to finish first.",
        ),
        sign_base.row(
            "scalar_proxy_practical_matching_law_available_now",
            "pass" if practical_matching_law_available_now else "reject",
            "scalar-proxy practical matching law available now",
            sign_base.truth(practical_matching_law_available_now),
            "The retained cubic-sqrt leading law already reaches sub-1e-3 relative alpha accuracy, so only the exact derivation and NLO gap remain live.",
        ),
        sign_base.row(
            "scalar_proxy_evanescent_tail_route_supported_now",
            "pass" if evanescent_tail_route_supported_now else "reject",
            "scalar-proxy evanescent tail route supported now",
            sign_base.truth(evanescent_tail_route_supported_now),
            "The tail route stays secondary unless its window drift collapses in a later audit.",
        ),
        sign_base.row(
            "scalar_proxy_virial_route_supported_now",
            "pass" if virial_route_supported_now else "reject",
            "scalar-proxy virial route supported now",
            sign_base.truth(virial_route_supported_now),
            "The virial route stays secondary unless a clean 3/2 ratio is derived rather than guessed from current retained energies.",
        ),
        sign_base.row(
            "selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now",
            "pass" if source_materialization_secondary_reserve_retained_now else "reject",
            "selected-extension independent extra-q-range source-materialization secondary reserve retained now",
            sign_base.truth(source_materialization_secondary_reserve_retained_now),
            "The reserve source-materialization lane is kept alive but stays secondary while the scalar proxy still has a live algebraic closure path.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive route update required now",
            sign_base.truth(pack_update_required_now),
            "A substantive route update has happened: the active blocker is now the exact derivation of the three-halves leading law.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_exact_over_m0": float(prior_summary["q_exact_over_m0"]),
        "q_star_over_m0": float(prior_summary["q_star_over_m0"]),
        "c1_fit": float(prior_summary["c1_fit"]),
        "three_halves_linear_c1": float(prior_summary["three_halves_linear_c1"]),
        "q_cubic_sqrt_over_m0": float(prior_summary["q_cubic_sqrt_over_m0"]),
        "q_cubic_sqrt_rel_error": float(prior_summary["q_cubic_sqrt_rel_error"]),
        "alpha_cubic_sqrt": float(prior_summary["alpha_cubic_sqrt"]),
        "alpha_cubic_sqrt_rel_error": float(prior_summary["alpha_cubic_sqrt_rel_error"]),
        "q_squared_correction_coeff_fit": float(prior_summary["q_squared_correction_coeff_fit"]),
        "q_squared_correction_coeff_rel_error_vs_cubic": float(
            prior_summary["q_squared_correction_coeff_rel_error_vs_cubic"]
        ),
        "gate_a_updated_pack_exact_scalar_proxy_profile_sensitive_q_star_correction_audit_available_now": gate_a,
        "gate_b_updated_pack_scalar_proxy_three_halves_first_principles_derivation_promoted_next": gate_b,
        "gate_c_selected_extension_source_materialization_reopen_required_now": gate_c,
        "practical_matching_law_available_now": practical_matching_law_available_now,
        "mexican_hat_cubic_route_supported_now": bool(prior_summary["mexican_hat_cubic_route_supported_now"]),
        "direct_fourier_route_supported_now": bool(prior_summary["direct_fourier_route_supported_now"]),
        "evanescent_tail_route_supported_now": evanescent_tail_route_supported_now,
        "virial_route_supported_now": virial_route_supported_now,
        "exact_three_halves_first_principles_derivation_available_now": bool(
            prior_summary["exact_three_halves_first_principles_derivation_available_now"]
        ),
        "selected_primary_completion_lane": "updated_pack_scalar_proxy_three_halves_first_principles_derivation_audit",
        "selected_secondary_completion_lane": "updated_pack_scalar_proxy_direct_fourier_nlo_gap_review",
        "selected_reserve_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun",
        "selected_next_generation_route": "trial2_numeric_alpha_scalar_proxy_three_halves_first_principles_derivation_audit",
        "recommended_next_route_or_none": "8.7.56.5407",
        "selected_followup_route": "trial2_numeric_alpha_scalar_proxy_three_halves_first_principles_derivation_gate",
        "selected_followup_route_or_none": "8.7.56.5411",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5405",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5407",
                "followup_route": "8.7.56.5411",
            },
        },
        rows,
        summary,
        {
            "overall_status": "scalar_proxy_profile_sensitive_qstar_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} scalar-proxy profile-sensitive q_star gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# Function: run the gate refresh when invoked as one CLI script.

if __name__ == "__main__":
    main()
