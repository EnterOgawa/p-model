#!/usr/bin/env python3
"""Generate 8.7.56.5659-.5662 Trial-2 beta-sensitivity Green-kernel gate artifacts."""

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
        "8.7.56.5655-5658",
        "updated_pack_trial2_beta_sensitivity_green_kernel_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5659-5662"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "beta-sensitivity Green-kernel gate / conditional-hold refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_green_kernel_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_green_kernel_audited_spectral_projection_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_green_kernel_negative_closeout_completed_"
    "spectral_projection_followup_primary_conditional_hold_secondary_next"
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
    """Return formulas used by the Green-kernel gate."""
    return {
        "gate_a": "Gate A = transformed Green-kernel / resolvent surface is available now",
        "gate_b": "Gate B = one-sign Green-kernel path closes negatively but source-weighted resolvent remains negative now",
        "gate_c": "Gate C = spectral-projection followup is promoted while conditional hold stays secondary",
    }


# 関数: `.5659-.5662` を実行する。

def main() -> None:
    """Execute the Trial-2 beta-sensitivity Green-kernel gate / refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary["exact_trial2_beta_sensitivity_green_kernel_available_now"]
        and prior_summary["exact_trial2_beta_sensitivity_source_weighted_resolvent_negative_now"]
    )
    gate_b = bool(
        prior_summary["exact_trial2_beta_sensitivity_green_kernel_negative_closeout_available_now"]
        and not prior_summary["exact_trial2_beta_sensitivity_green_kernel_one_sign_available_now"]
    )
    gate_c = bool(
        gate_a
        and gate_b
        and prior_summary[
            "updated_pack_trial2_beta_sensitivity_spectral_projection_followup_required_now"
        ]
    )

    trial2_beta_sensitivity_green_kernel_followup_lane_completed_now = bool(gate_b)
    trial2_beta_sensitivity_spectral_projection_followup_promoted_now = bool(gate_c)
    trial2_conditional_hold_secondary_now = bool(gate_c)

    rows = [
        sign_base.row(
            "gate_a_trial2_beta_sensitivity_green_kernel_available_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 beta-sensitivity Green-kernel surface available now",
            sign_base.truth(gate_a),
            "The Green-kernel branch only becomes official once the transformed resolvent surface and its source-weighted sign read are machine-readable.",
        ),
        sign_base.row(
            "gate_b_trial2_beta_sensitivity_green_kernel_negative_closeout_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 beta-sensitivity Green-kernel negative closeout now",
            sign_base.truth(gate_b),
            "The naive one-sign Green-kernel proof path closes negatively once mixed-sign kernel support is fixed on the canonical window.",
        ),
        sign_base.row(
            "gate_c_trial2_beta_sensitivity_spectral_projection_followup_promoted_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 beta-sensitivity spectral-projection followup promoted now",
            sign_base.truth(gate_c),
            "Once the one-sign kernel route fails, the honest next theorem route is source-weighted spectral projection / principal-mode dominance.",
        ),
        sign_base.row(
            "trial2_beta_sensitivity_green_kernel_followup_lane_completed_now",
            "pass" if trial2_beta_sensitivity_green_kernel_followup_lane_completed_now else "reject",
            "Trial-2 beta-sensitivity Green-kernel followup lane completed now",
            sign_base.truth(trial2_beta_sensitivity_green_kernel_followup_lane_completed_now),
            "This lane is complete once the Green-kernel route is either proven or honestly closed; here the naive one-sign-kernel path closes negatively.",
        ),
        sign_base.row(
            "trial2_beta_sensitivity_spectral_projection_followup_promoted_now",
            "pass" if trial2_beta_sensitivity_spectral_projection_followup_promoted_now else "reject",
            "Trial-2 beta-sensitivity spectral-projection followup promoted now",
            sign_base.truth(trial2_beta_sensitivity_spectral_projection_followup_promoted_now),
            "The next strict-theorem blocker is no longer kernel sign itself, but source-weighted spectral projection / principal-mode dominance.",
        ),
        sign_base.row(
            "trial2_conditional_hold_secondary_now",
            "pass" if trial2_conditional_hold_secondary_now else "reject",
            "Trial-2 conditional hold secondary now",
            sign_base.truth(trial2_conditional_hold_secondary_now),
            "Conditional hold remains only as the fallback if the spectral-projection route dead-ends honestly.",
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
        "exact_trial2_beta_sensitivity_green_kernel_available_now": bool(
            prior_summary["exact_trial2_beta_sensitivity_green_kernel_available_now"]
        ),
        "exact_trial2_beta_sensitivity_green_kernel_one_sign_available_now": bool(
            prior_summary["exact_trial2_beta_sensitivity_green_kernel_one_sign_available_now"]
        ),
        "exact_trial2_beta_sensitivity_source_weighted_resolvent_negative_now": bool(
            prior_summary["exact_trial2_beta_sensitivity_source_weighted_resolvent_negative_now"]
        ),
        "exact_trial2_beta_sensitivity_single_negative_mode_dominance_support_available_now": bool(
            prior_summary[
                "exact_trial2_beta_sensitivity_single_negative_mode_dominance_support_available_now"
            ]
        ),
        "exact_trial2_beta_sensitivity_green_kernel_negative_closeout_available_now": bool(
            prior_summary["exact_trial2_beta_sensitivity_green_kernel_negative_closeout_available_now"]
        ),
        "updated_pack_trial2_beta_sensitivity_spectral_projection_followup_required_now": bool(
            prior_summary[
                "updated_pack_trial2_beta_sensitivity_spectral_projection_followup_required_now"
            ]
        ),
        "trial2_beta_sensitivity_green_kernel_followup_lane_completed_now": (
            trial2_beta_sensitivity_green_kernel_followup_lane_completed_now
        ),
        "trial2_beta_sensitivity_spectral_projection_followup_promoted_now": (
            trial2_beta_sensitivity_spectral_projection_followup_promoted_now
        ),
        "trial2_conditional_hold_secondary_now": trial2_conditional_hold_secondary_now,
        "selected_primary_completion_lane": (
            "trial2_beta_sensitivity_spectral_projection_followup"
        ),
        "selected_secondary_completion_lane": "conditional_hold_only",
        "selected_reserve_completion_lane": "conditional_hold_only",
        "selected_next_generation_route": (
            "trial2_beta_sensitivity_spectral_projection_followup"
        ),
        "recommended_next_route_or_none": (
            "trial2_beta_sensitivity_spectral_projection_followup"
        ),
        "selected_followup_route": "trial2_beta_sensitivity_spectral_projection_followup",
        "selected_followup_route_or_none": (
            "trial2_beta_sensitivity_spectral_projection_followup"
        ),
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5661",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_beta_sensitivity_green_kernel_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": gate_b,
            "physical_reject_required": False,
        },
        {
            "coarse_full_inverse_negative_fraction": float(
                prior_summary["coarse_full_inverse_negative_fraction"]
            ),
            "coarse_full_inverse_positive_fraction": float(
                prior_summary["coarse_full_inverse_positive_fraction"]
            ),
            "fine_probe_075_negative_fraction": float(
                prior_summary["fine_probe_075_negative_fraction"]
            ),
            "fine_probe_075_positive_fraction": float(
                prior_summary["fine_probe_075_positive_fraction"]
            ),
            "fine_source_solution_min": float(prior_summary["fine_source_solution_min"]),
            "fine_source_solution_max": float(prior_summary["fine_source_solution_max"]),
            "spectral_lambda_1": float(prior_summary["spectral_lambda_1"]),
            "spectral_lambda_2": float(prior_summary["spectral_lambda_2"]),
            "principal_mode_dominance_ratio": float(
                prior_summary["principal_mode_dominance_ratio"]
            ),
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5659-5662 Trial-2 beta-sensitivity Green-kernel gate completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
