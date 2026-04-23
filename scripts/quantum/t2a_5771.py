#!/usr/bin/env python3
"""Generate 8.7.56.5771-.5774 source-weighted operator-level continuum gate artifacts."""

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
        "8.7.56.5767-5770",
        "updated_pack_trial2_beta_sensitivity_source_weighted_operator_level_continuum_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5771-5774"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "source-weighted operator-level continuum gate / conditional-reopen refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_source_weighted_operator_level_continuum_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_source_weighted_operator_level_control_window_continuum_audited_"
    "gate_sync_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_source_weighted_operator_level_control_window_continuum_closure_completed_"
    "global_kernel_refinement_deferred_v3_conditional_reopen_only_next"
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
    """Return formulas used by the operator-level continuum gate."""
    return {
        "gate_a": "Gate A = source-weighted operator-level control-window continuum closure is completed now",
        "gate_b": "Gate B = pure analytic global one-sign kernel refinement remains deferred to v3 now",
        "gate_c": "Gate C = no unconditional next official branch remains now",
    }


# 関数: `.5771-.5774` を実行する。

def main() -> None:
    """Execute the source-weighted operator-level continuum gate / refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_trial2_source_weighted_operator_level_control_window_continuum_closure_available_now"
        ]
    )
    gate_b = bool(
        gate_a
        and not prior_summary[
            "exact_trial2_pure_analytic_global_one_sign_kernel_theorem_available_now"
        ]
    )
    gate_c = bool(
        gate_b
        and prior_summary[
            "updated_pack_trial2_source_weighted_operator_level_continuum_gate_required_now"
        ]
    )

    rows = [
        sign_base.row(
            "gate_a_trial2_source_weighted_operator_level_control_window_continuum_closure_completed_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 source-weighted operator-level control-window continuum closure completed now",
            sign_base.truth(gate_a),
            "The exact comparison identity and the explicit omitted-tail nonreversal bound now close one honest operator-level continuum theorem on the physical control window.",
        ),
        sign_base.row(
            "gate_b_trial2_global_kernel_refinement_deferred_v3_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 global kernel refinement deferred to v3 now",
            sign_base.truth(gate_b),
            "This gate keeps the wording honest: the control-window operator-level theorem is completed, while the stronger global one-sign kernel refinement remains deferred.",
        ),
        sign_base.row(
            "gate_c_trial2_no_unconditional_next_branch_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 no unconditional next branch now",
            sign_base.truth(gate_c),
            "Once the operator-level control-window closure is synced, the current pack returns to conditional reopen only rather than to another forced replay.",
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
        "retained_x_cutoff": float(prior_summary["retained_x_cutoff"]),
        "retained_omitted_negative_tail_upper_bound": float(
            prior_summary["retained_omitted_negative_tail_upper_bound"]
        ),
        "retained_comparison_margin_lower_bound": float(
            prior_summary["retained_comparison_margin_lower_bound"]
        ),
        "family_comparison_margin_lower_bound_min": float(
            prior_summary["family_comparison_margin_lower_bound_min"]
        ),
        "source_weighted_operator_level_control_window_continuum_closure_completed_now": bool(
            gate_a
        ),
        "exact_trial2_pure_analytic_global_one_sign_kernel_theorem_available_now": False,
        "exact_trial2_pure_analytic_global_one_sign_kernel_refinement_deferred_to_v3_now": bool(
            gate_b
        ),
        "no_unconditional_next_official_branch_now": bool(gate_c),
        "selected_next_generation_route": None,
        "recommended_next_route_or_none": None,
        "selected_followup_route": None,
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5773",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": (
                "trial2_source_weighted_operator_level_continuum_gate_completed"
            ),
            "branch_completed": True,
            "breakthrough_passed_now": gate_a,
            "physical_reject_required": False,
        },
        {
            "retained_x_cutoff": float(prior_summary["retained_x_cutoff"]),
            "retained_omitted_negative_tail_upper_bound": float(
                prior_summary["retained_omitted_negative_tail_upper_bound"]
            ),
            "retained_comparison_margin_lower_bound": float(
                prior_summary["retained_comparison_margin_lower_bound"]
            ),
            "family_comparison_margin_lower_bound_min": float(
                prior_summary["family_comparison_margin_lower_bound_min"]
            ),
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5771-5774 Trial-2 source-weighted operator-level continuum gate completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
