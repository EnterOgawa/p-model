#!/usr/bin/env python3
"""Generate 8.7.56.5747-.5750 patched half-line Green-kernel gate artifacts."""

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
        "8.7.56.5743-5746",
        "updated_pack_trial2_beta_sensitivity_halfline_green_kernel_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5747-5750"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "patched half-line Green-kernel gate / conditional-reopen refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_halfline_green_kernel_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_patched_halfline_green_kernel_audited_"
    "source_weighted_comparison_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_patched_halfline_green_kernel_negative_closeout_completed_"
    "source_weighted_comparison_followup_primary_conditional_reopen_secondary_next"
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
    """Return formulas used by the patched half-line Green-kernel gate."""
    return {
        "gate_a": "Gate A = patched half-line Green-kernel surface is available now",
        "gate_b": "Gate B = one-sign half-line kernel closes negatively while control-window source-weighted negativity remains available now",
        "gate_c": "Gate C = source-weighted comparison followup is promoted while conditional reopen stays secondary now",
    }


# 関数: `.5747-.5750` を実行する。

def main() -> None:
    """Execute the patched half-line Green-kernel gate / refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary["exact_trial2_beta_sensitivity_halfline_green_kernel_available_now"]
        and prior_summary[
            "exact_trial2_beta_sensitivity_halfline_source_weighted_bvp_negative_control_window_now"
        ]
    )
    gate_b = bool(
        prior_summary[
            "exact_trial2_beta_sensitivity_halfline_green_kernel_negative_closeout_available_now"
        ]
        and not prior_summary[
            "exact_trial2_beta_sensitivity_halfline_green_kernel_one_sign_available_now"
        ]
        and not prior_summary[
            "exact_trial2_beta_sensitivity_halfline_yukawa_contraction_available_now"
        ]
    )
    gate_c = bool(
        gate_a
        and gate_b
        and prior_summary["updated_pack_trial2_source_weighted_comparison_followup_required_now"]
    )

    trial2_patched_halfline_green_kernel_followup_lane_completed_now = bool(gate_b)
    trial2_source_weighted_comparison_followup_promoted_now = bool(gate_c)
    trial2_conditional_reopen_secondary_now = bool(gate_c)

    rows = [
        sign_base.row(
            "gate_a_trial2_patched_halfline_green_kernel_surface_available_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 patched half-line Green-kernel surface available now",
            sign_base.truth(gate_a),
            "The reopened route only becomes official once the admissible half-line kernel surface and the source-weighted control-window sign read are both machine-readable.",
        ),
        sign_base.row(
            "gate_b_trial2_patched_halfline_green_kernel_negative_closeout_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 patched half-line Green-kernel negative closeout now",
            sign_base.truth(gate_b),
            "The honest verdict is that one-sign kernel control and naive Yukawa contraction both fail, even though the source-weighted control-window solution stays negative.",
        ),
        sign_base.row(
            "gate_c_trial2_source_weighted_comparison_followup_promoted_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 source-weighted comparison followup promoted now",
            sign_base.truth(gate_c),
            "Once the kernel-level route closes negatively, the honest next theorem object is a source-weighted comparison theorem, not another kernel replay.",
        ),
        sign_base.row(
            "trial2_patched_halfline_green_kernel_followup_lane_completed_now",
            "pass" if trial2_patched_halfline_green_kernel_followup_lane_completed_now else "reject",
            "Trial-2 patched half-line Green-kernel followup lane completed now",
            sign_base.truth(trial2_patched_halfline_green_kernel_followup_lane_completed_now),
            "This lane is complete once the one-sign half-line kernel route is either proven or honestly closed; here it closes negatively.",
        ),
        sign_base.row(
            "trial2_source_weighted_comparison_followup_promoted_now",
            "pass" if trial2_source_weighted_comparison_followup_promoted_now else "reject",
            "Trial-2 source-weighted comparison followup promoted now",
            sign_base.truth(trial2_source_weighted_comparison_followup_promoted_now),
            "The next live blocker is now source-weighted comparison, because the half-line kernel itself no longer carries the proof burden.",
        ),
        sign_base.row(
            "trial2_conditional_reopen_secondary_now",
            "pass" if trial2_conditional_reopen_secondary_now else "reject",
            "Trial-2 conditional reopen secondary now",
            sign_base.truth(trial2_conditional_reopen_secondary_now),
            "Conditional reopen remains only as fallback if the source-weighted comparison route dead-ends honestly.",
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
        "exact_trial2_beta_sensitivity_halfline_green_kernel_available_now": bool(
            prior_summary["exact_trial2_beta_sensitivity_halfline_green_kernel_available_now"]
        ),
        "exact_trial2_beta_sensitivity_halfline_green_kernel_one_sign_available_now": bool(
            prior_summary["exact_trial2_beta_sensitivity_halfline_green_kernel_one_sign_available_now"]
        ),
        "exact_trial2_beta_sensitivity_halfline_source_weighted_bvp_negative_control_window_now": bool(
            prior_summary[
                "exact_trial2_beta_sensitivity_halfline_source_weighted_bvp_negative_control_window_now"
            ]
        ),
        "exact_trial2_beta_sensitivity_halfline_yukawa_contraction_available_now": bool(
            prior_summary["exact_trial2_beta_sensitivity_halfline_yukawa_contraction_available_now"]
        ),
        "exact_trial2_beta_sensitivity_halfline_green_kernel_negative_closeout_available_now": bool(
            prior_summary[
                "exact_trial2_beta_sensitivity_halfline_green_kernel_negative_closeout_available_now"
            ]
        ),
        "updated_pack_trial2_source_weighted_comparison_followup_required_now": bool(
            prior_summary["updated_pack_trial2_source_weighted_comparison_followup_required_now"]
        ),
        "trial2_patched_halfline_green_kernel_followup_lane_completed_now": (
            trial2_patched_halfline_green_kernel_followup_lane_completed_now
        ),
        "trial2_source_weighted_comparison_followup_promoted_now": (
            trial2_source_weighted_comparison_followup_promoted_now
        ),
        "trial2_conditional_reopen_secondary_now": trial2_conditional_reopen_secondary_now,
        "retained_x_max": float(prior_summary["retained_x_max"]),
        "retained_probe_075_negative_fraction": float(
            prior_summary["retained_probe_075_negative_fraction"]
        ),
        "retained_probe_075_positive_fraction": float(
            prior_summary["retained_probe_075_positive_fraction"]
        ),
        "retained_green_bvp_control_rel_linf": float(
            prior_summary["retained_green_bvp_control_rel_linf"]
        ),
        "retained_green_bvp_control_corrcoef": float(
            prior_summary["retained_green_bvp_control_corrcoef"]
        ),
        "retained_yukawa_contraction_sup": float(
            prior_summary["retained_yukawa_contraction_sup"]
        ),
        "selected_primary_completion_lane": (
            "trial2_beta_sensitivity_source_weighted_comparison_followup"
        ),
        "selected_secondary_completion_lane": "conditional_reopen_only",
        "selected_reserve_completion_lane": "conditional_reopen_only",
        "selected_next_generation_route": (
            "trial2_beta_sensitivity_source_weighted_comparison_followup"
        ),
        "recommended_next_route_or_none": (
            "trial2_beta_sensitivity_source_weighted_comparison_followup"
        ),
        "selected_followup_route": "trial2_beta_sensitivity_source_weighted_comparison_followup",
        "selected_followup_route_or_none": (
            "trial2_beta_sensitivity_source_weighted_comparison_followup"
        ),
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5749",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_patched_halfline_green_kernel_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": gate_b,
            "physical_reject_required": False,
        },
        {
            "retained_x_max": float(prior_summary["retained_x_max"]),
            "retained_probe_075_negative_fraction": float(
                prior_summary["retained_probe_075_negative_fraction"]
            ),
            "retained_probe_075_positive_fraction": float(
                prior_summary["retained_probe_075_positive_fraction"]
            ),
            "retained_green_bvp_control_rel_linf": float(
                prior_summary["retained_green_bvp_control_rel_linf"]
            ),
            "retained_green_bvp_control_corrcoef": float(
                prior_summary["retained_green_bvp_control_corrcoef"]
            ),
            "retained_yukawa_contraction_sup": float(
                prior_summary["retained_yukawa_contraction_sup"]
            ),
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5747-5750 Trial-2 patched half-line Green-kernel gate completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
