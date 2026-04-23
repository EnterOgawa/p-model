#!/usr/bin/env python3
"""Generate 8.7.56.5755-.5758 source-weighted comparison gate artifacts."""

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
        "8.7.56.5751-5754",
        "updated_pack_trial2_beta_sensitivity_source_weighted_comparison_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5755-5758"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "source-weighted comparison gate / conditional-reopen secondary refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_source_weighted_comparison_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_source_weighted_comparison_audited_"
    "pure_continuum_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_source_weighted_comparison_sign_support_completed_"
    "pure_continuum_followup_primary_conditional_reopen_secondary_next"
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
    """Return formulas used by the source-weighted comparison gate."""
    return {
        "gate_a": "Gate A = source-weighted comparison support is available now",
        "gate_b": "Gate B = positive source-weighted dominance stays stable and exact on the retained control window",
        "gate_c": "Gate C = pure-continuum followup is promoted while conditional reopen stays secondary now",
    }


# 関数: `.5755-.5758` を実行する。
def main() -> None:
    """Execute the source-weighted comparison gate / refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary["source_weighted_comparison_surface_available_now"]
        and prior_summary[
            "exact_trial2_beta_sensitivity_source_weighted_comparison_support_available_now"
        ]
    )
    gate_b = bool(
        prior_summary[
            "exact_trial2_beta_sensitivity_source_weighted_positive_dominance_control_window_now"
        ]
        and prior_summary[
            "exact_trial2_beta_sensitivity_source_weighted_comparison_identity_available_now"
        ]
        and prior_summary[
            "exact_trial2_beta_sensitivity_source_weighted_comparison_green_bvp_coherent_now"
        ]
        and prior_summary[
            "exact_trial2_beta_sensitivity_source_weighted_comparison_stable_now"
        ]
    )
    gate_c = bool(
        gate_a
        and gate_b
        and prior_summary[
            "updated_pack_trial2_source_weighted_comparison_pure_continuum_followup_required_now"
        ]
    )

    trial2_source_weighted_comparison_followup_lane_completed_now = bool(gate_b)
    trial2_source_weighted_comparison_pure_continuum_followup_promoted_now = bool(
        gate_c
    )
    trial2_conditional_reopen_secondary_now = bool(gate_c)

    rows = [
        sign_base.row(
            "gate_a_trial2_source_weighted_comparison_support_available_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 source-weighted comparison support available now",
            sign_base.truth(gate_a),
            "The route only becomes official once the exact source-weighted comparison identity and its control-window dominance are machine-readable.",
        ),
        sign_base.row(
            "gate_b_trial2_source_weighted_comparison_sign_support_completed_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 source-weighted comparison sign support completed now",
            sign_base.truth(gate_b),
            "The honest result is that negativity on the retained control window is now carried by stable source-weighted comparison, not by a one-sign kernel.",
        ),
        sign_base.row(
            "gate_c_trial2_source_weighted_comparison_pure_continuum_followup_promoted_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 source-weighted comparison pure-continuum followup promoted now",
            sign_base.truth(gate_c),
            "Once control-window comparison support is fixed, the next blocker is its pure-continuum promotion rather than any further kernel replay.",
        ),
        sign_base.row(
            "trial2_source_weighted_comparison_followup_lane_completed_now",
            "pass" if trial2_source_weighted_comparison_followup_lane_completed_now else "reject",
            "Trial-2 source-weighted comparison followup lane completed now",
            sign_base.truth(trial2_source_weighted_comparison_followup_lane_completed_now),
            "This lane is complete once source-weighted comparison either proves or honestly rejects the retained control-window negativity. Here it proves the comparison support.",
        ),
        sign_base.row(
            "trial2_source_weighted_comparison_pure_continuum_followup_promoted_now",
            "pass" if trial2_source_weighted_comparison_pure_continuum_followup_promoted_now else "reject",
            "Trial-2 source-weighted comparison pure-continuum followup promoted now",
            sign_base.truth(trial2_source_weighted_comparison_pure_continuum_followup_promoted_now),
            "The next live blocker is now the pure-continuum promotion of the comparison theorem, because control-window sign support is no longer the issue.",
        ),
        sign_base.row(
            "trial2_conditional_reopen_secondary_now",
            "pass" if trial2_conditional_reopen_secondary_now else "reject",
            "Trial-2 conditional reopen secondary now",
            sign_base.truth(trial2_conditional_reopen_secondary_now),
            "Conditional reopen remains only as fallback if the pure-continuum comparison promotion dead-ends honestly.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(prior_summary["beta_common_root"]),
        "alpha_common_value": float(prior_summary["alpha_common_value"]),
        "alpha_common_rel_error_vs_target": float(
            prior_summary["alpha_common_rel_error_vs_target"]
        ),
        "source_weighted_comparison_surface_available_now": bool(
            prior_summary["source_weighted_comparison_surface_available_now"]
        ),
        "exact_trial2_beta_sensitivity_source_weighted_positive_dominance_control_window_now": bool(
            prior_summary[
                "exact_trial2_beta_sensitivity_source_weighted_positive_dominance_control_window_now"
            ]
        ),
        "exact_trial2_beta_sensitivity_source_weighted_comparison_identity_available_now": bool(
            prior_summary[
                "exact_trial2_beta_sensitivity_source_weighted_comparison_identity_available_now"
            ]
        ),
        "exact_trial2_beta_sensitivity_source_weighted_comparison_green_bvp_coherent_now": bool(
            prior_summary[
                "exact_trial2_beta_sensitivity_source_weighted_comparison_green_bvp_coherent_now"
            ]
        ),
        "exact_trial2_beta_sensitivity_source_weighted_comparison_stable_now": bool(
            prior_summary[
                "exact_trial2_beta_sensitivity_source_weighted_comparison_stable_now"
            ]
        ),
        "exact_trial2_beta_sensitivity_source_weighted_comparison_support_available_now": bool(
            prior_summary[
                "exact_trial2_beta_sensitivity_source_weighted_comparison_support_available_now"
            ]
        ),
        "updated_pack_trial2_source_weighted_comparison_pure_continuum_followup_required_now": bool(
            prior_summary[
                "updated_pack_trial2_source_weighted_comparison_pure_continuum_followup_required_now"
            ]
        ),
        "trial2_source_weighted_comparison_followup_lane_completed_now": (
            trial2_source_weighted_comparison_followup_lane_completed_now
        ),
        "trial2_source_weighted_comparison_pure_continuum_followup_promoted_now": (
            trial2_source_weighted_comparison_pure_continuum_followup_promoted_now
        ),
        "trial2_conditional_reopen_secondary_now": trial2_conditional_reopen_secondary_now,
        "retained_x_max": float(prior_summary["retained_x_max"]),
        "retained_min_comparison_ratio": float(
            prior_summary["retained_min_comparison_ratio"]
        ),
        "retained_min_comparison_relative_gap": float(
            prior_summary["retained_min_comparison_relative_gap"]
        ),
        "retained_min_comparison_margin": float(
            prior_summary["retained_min_comparison_margin"]
        ),
        "retained_max_identity_rel_error": float(
            prior_summary["retained_max_identity_rel_error"]
        ),
        "comparison_ratio_rel_spread": float(prior_summary["comparison_ratio_rel_spread"]),
        "comparison_relative_gap_rel_spread": float(
            prior_summary["comparison_relative_gap_rel_spread"]
        ),
        "selected_primary_completion_lane": (
            "trial2_beta_sensitivity_source_weighted_comparison_pure_continuum_followup"
        ),
        "selected_secondary_completion_lane": "conditional_reopen_only",
        "selected_reserve_completion_lane": "conditional_reopen_only",
        "selected_next_generation_route": (
            "trial2_beta_sensitivity_source_weighted_comparison_pure_continuum_followup"
        ),
        "recommended_next_route_or_none": (
            "trial2_beta_sensitivity_source_weighted_comparison_pure_continuum_followup"
        ),
        "selected_followup_route": (
            "trial2_beta_sensitivity_source_weighted_comparison_pure_continuum_followup"
        ),
        "selected_followup_route_or_none": (
            "trial2_beta_sensitivity_source_weighted_comparison_pure_continuum_followup"
        ),
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5757",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_source_weighted_comparison_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": gate_b,
            "physical_reject_required": False,
        },
        {
            "retained_x_max": float(prior_summary["retained_x_max"]),
            "retained_min_comparison_ratio": float(
                prior_summary["retained_min_comparison_ratio"]
            ),
            "retained_min_comparison_relative_gap": float(
                prior_summary["retained_min_comparison_relative_gap"]
            ),
            "retained_min_comparison_margin": float(
                prior_summary["retained_min_comparison_margin"]
            ),
            "retained_max_identity_rel_error": float(
                prior_summary["retained_max_identity_rel_error"]
            ),
            "comparison_ratio_rel_spread": float(
                prior_summary["comparison_ratio_rel_spread"]
            ),
            "comparison_relative_gap_rel_spread": float(
                prior_summary["comparison_relative_gap_rel_spread"]
            ),
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5755-5758 Trial-2 source-weighted comparison gate completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
