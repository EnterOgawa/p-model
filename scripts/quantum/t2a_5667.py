#!/usr/bin/env python3
"""Generate 8.7.56.5667-.5670 Trial-2 beta-sensitivity spectral-projection gate artifacts."""

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
        "8.7.56.5663-5666",
        "updated_pack_trial2_beta_sensitivity_spectral_projection_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5667-5670"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "beta-sensitivity spectral-projection gate / conditional-hold secondary refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_spectral_projection_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_spectral_projection_audited_"
    "discrete_pointwise_dominance_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_discrete_spectral_projection_theorem_completed_"
    "continuum_followup_primary_conditional_hold_secondary_next"
)


# 関数: JSON/CSV artifact を書き出す。
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


# 関数: gate で使う式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the spectral-projection gate."""
    return {
        "gate_a": "Gate A = full finite spectral decomposition is available now",
        "gate_b": "Gate B = principal mode dominates the absolute remainder pointwise and proves discrete negativity now",
        "gate_c": "Gate C = continuum spectral-projection followup is promoted while conditional hold stays secondary",
    }


# 関数: `.5667-.5670` を実行する。

def main() -> None:
    """Execute the Trial-2 beta-sensitivity spectral-projection gate / refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_trial2_beta_sensitivity_discrete_spectral_projection_available_now"
        ]
        and prior_summary["exact_trial2_beta_sensitivity_principal_mode_one_sign_now"]
        and prior_summary["exact_trial2_beta_sensitivity_principal_component_negative_now"]
    )
    gate_b = bool(
        gate_a
        and prior_summary[
            "exact_trial2_beta_sensitivity_discrete_pointwise_dominance_now"
        ]
        and prior_summary[
            "exact_trial2_beta_sensitivity_discrete_negativity_theorem_available_now"
        ]
    )
    gate_c = bool(
        gate_b
        and prior_summary[
            "updated_pack_trial2_beta_sensitivity_continuum_spectral_projection_followup_required_now"
        ]
    )

    trial2_beta_sensitivity_spectral_projection_followup_lane_completed_now = bool(
        gate_b
    )
    trial2_beta_sensitivity_continuum_spectral_projection_followup_promoted_now = bool(
        gate_c
    )
    trial2_conditional_hold_secondary_now = bool(gate_c)

    rows = [
        sign_base.row(
            "gate_a_trial2_beta_sensitivity_discrete_spectral_projection_available_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 beta-sensitivity discrete spectral projection available now",
            sign_base.truth(gate_a),
            "The spectral-projection route only becomes official once the full finite decomposition, principal-mode one-sign property, and principal-component negativity are simultaneously machine-readable.",
        ),
        sign_base.row(
            "gate_b_trial2_beta_sensitivity_discrete_negativity_theorem_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 beta-sensitivity discrete negativity theorem now",
            sign_base.truth(gate_b),
            "The branch closes positively once the principal component dominates the absolute remainder pointwise and reconstructs the negative source-weighted resolvent exactly.",
        ),
        sign_base.row(
            "gate_c_trial2_beta_sensitivity_continuum_followup_promoted_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 beta-sensitivity continuum followup promoted now",
            sign_base.truth(gate_c),
            "Once the finite theorem is fixed, the honest next blocker is continuum / operator-level promotion rather than conditional hold or discrete replay.",
        ),
        sign_base.row(
            "trial2_beta_sensitivity_spectral_projection_followup_lane_completed_now",
            "pass"
            if trial2_beta_sensitivity_spectral_projection_followup_lane_completed_now
            else "reject",
            "Trial-2 beta-sensitivity spectral-projection followup lane completed now",
            sign_base.truth(
                trial2_beta_sensitivity_spectral_projection_followup_lane_completed_now
            ),
            "This lane is complete once the source-weighted spectral-projection theorem is either proven or honestly closed; here the discrete pointwise-dominance theorem is proven.",
        ),
        sign_base.row(
            "trial2_beta_sensitivity_continuum_spectral_projection_followup_promoted_now",
            "pass"
            if trial2_beta_sensitivity_continuum_spectral_projection_followup_promoted_now
            else "reject",
            "Trial-2 beta-sensitivity continuum spectral-projection followup promoted now",
            sign_base.truth(
                trial2_beta_sensitivity_continuum_spectral_projection_followup_promoted_now
            ),
            "The next strict-theorem blocker is no longer finite spectral support; it is the continuum / operator-level theorem that would remove the remaining discretization surface.",
        ),
        sign_base.row(
            "trial2_conditional_hold_secondary_now",
            "pass" if trial2_conditional_hold_secondary_now else "reject",
            "Trial-2 conditional hold secondary now",
            sign_base.truth(trial2_conditional_hold_secondary_now),
            "Conditional hold remains only as the fallback if the continuum spectral-projection route dead-ends honestly.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "interaction_total_over_harmonic_sq_beta_common_root": float(
            prior_summary["interaction_total_over_harmonic_sq_beta_common_root"]
        ),
        "interaction_total_over_harmonic_sq_alpha_common_value": float(
            prior_summary["interaction_total_over_harmonic_sq_alpha_common_value"]
        ),
        "interaction_total_over_harmonic_sq_alpha_common_rel_error_vs_target": float(
            prior_summary["interaction_total_over_harmonic_sq_alpha_common_rel_error_vs_target"]
        ),
        "exact_trial2_beta_sensitivity_discrete_spectral_projection_available_now": bool(
            prior_summary[
                "exact_trial2_beta_sensitivity_discrete_spectral_projection_available_now"
            ]
        ),
        "exact_trial2_beta_sensitivity_principal_mode_one_sign_now": bool(
            prior_summary["exact_trial2_beta_sensitivity_principal_mode_one_sign_now"]
        ),
        "exact_trial2_beta_sensitivity_principal_component_negative_now": bool(
            prior_summary[
                "exact_trial2_beta_sensitivity_principal_component_negative_now"
            ]
        ),
        "exact_trial2_beta_sensitivity_discrete_pointwise_dominance_now": bool(
            prior_summary[
                "exact_trial2_beta_sensitivity_discrete_pointwise_dominance_now"
            ]
        ),
        "exact_trial2_beta_sensitivity_discrete_negativity_theorem_available_now": bool(
            prior_summary[
                "exact_trial2_beta_sensitivity_discrete_negativity_theorem_available_now"
            ]
        ),
        "updated_pack_trial2_beta_sensitivity_continuum_spectral_projection_followup_required_now": bool(
            prior_summary[
                "updated_pack_trial2_beta_sensitivity_continuum_spectral_projection_followup_required_now"
            ]
        ),
        "trial2_beta_sensitivity_spectral_projection_followup_lane_completed_now": (
            trial2_beta_sensitivity_spectral_projection_followup_lane_completed_now
        ),
        "trial2_beta_sensitivity_continuum_spectral_projection_followup_promoted_now": (
            trial2_beta_sensitivity_continuum_spectral_projection_followup_promoted_now
        ),
        "trial2_conditional_hold_secondary_now": trial2_conditional_hold_secondary_now,
        "discrete_pointwise_margin_min_global": float(
            prior_summary["discrete_pointwise_margin_min_global"]
        ),
        "reconstruction_rel_linf_max": float(
            prior_summary["reconstruction_rel_linf_max"]
        ),
        "fine_pointwise_margin_min": float(prior_summary["fine_pointwise_margin_min"]),
        "fine_reconstructed_solution_max": float(
            prior_summary["fine_reconstructed_solution_max"]
        ),
        "selected_primary_completion_lane": (
            "trial2_beta_sensitivity_continuum_spectral_projection_followup"
        ),
        "selected_secondary_completion_lane": "conditional_hold_only",
        "selected_reserve_completion_lane": "conditional_hold_only",
        "selected_next_generation_route": (
            "trial2_beta_sensitivity_continuum_spectral_projection_followup"
        ),
        "recommended_next_route_or_none": (
            "trial2_beta_sensitivity_continuum_spectral_projection_followup"
        ),
        "selected_followup_route": (
            "trial2_beta_sensitivity_continuum_spectral_projection_followup"
        ),
        "selected_followup_route_or_none": (
            "trial2_beta_sensitivity_continuum_spectral_projection_followup"
        ),
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5669",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_beta_sensitivity_spectral_projection_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": gate_b,
            "physical_reject_required": False,
        },
        {
            "discrete_pointwise_margin_min_global": float(
                prior_summary["discrete_pointwise_margin_min_global"]
            ),
            "reconstruction_rel_linf_max": float(
                prior_summary["reconstruction_rel_linf_max"]
            ),
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print(
        "[done] 8.7.56.5667-5670 Trial-2 beta-sensitivity spectral-projection gate completed"
    )
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
