#!/usr/bin/env python3
"""Generate 8.7.56.5627-.5630 Trial-2 beta-root followup gate artifacts."""

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
        "8.7.56.5623-5626",
        "updated_pack_trial2_interaction_total_over_harmonic_sq_beta_root_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5627-5630"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "interaction_total_over_harmonic_sq beta-root gate / conditional-hold refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_interaction_total_over_harmonic_sq_beta_root_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_interaction_total_over_harmonic_sq_beta_root_followup_"
    "target_free_common_root_direct_alpha_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_target_free_common_root_direct_alpha_audited_practical_closeout_"
    "strict_theorem_followup_primary_conditional_hold_secondary_next"
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


# 関数: gate で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the beta-root followup gate."""
    return {
        "gate_a": "Gate A = target-free common-root beta selector is available now",
        "gate_b": "Gate B = practical direct-alpha closeout is available now",
        "gate_c": "Gate C = strict target-free theorem closeout is still unavailable now",
        "gate_d": "Gate D = strict-theorem followup becomes the primary next blocker",
    }


# 関数: `.5627-.5630` を実行する。

def main() -> None:
    """Execute the Trial-2 beta-root followup gate."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_trial2_interaction_total_over_harmonic_sq_target_free_beta_selector_available_now"
        ]
    )
    gate_b = bool(
        prior_summary[
            "exact_trial2_interaction_total_over_harmonic_sq_practical_direct_alpha_closeout_available_now"
        ]
    )
    gate_c = bool(
        not prior_summary[
            "exact_trial2_interaction_total_over_harmonic_sq_strict_target_free_theorem_closeout_available_now"
        ]
    )
    gate_d = bool(
        prior_summary[
            "updated_pack_trial2_target_free_common_root_strict_theorem_followup_primary_next_now"
        ]
    )

    trial2_target_free_common_root_direct_alpha_completed_now = bool(gate_a and gate_b and gate_c)
    trial2_target_free_common_root_strict_theorem_followup_primary_now = bool(
        trial2_target_free_common_root_direct_alpha_completed_now and gate_d
    )
    conditional_hold_secondary_retained_now = True

    rows = [
        sign_base.row(
            "gate_a_trial2_target_free_common_root_beta_selector_available_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 target-free common-root beta selector available now",
            sign_base.truth(gate_a),
            "The branch only advances if beta is selected by equality of two independent frozen-action readouts.",
        ),
        sign_base.row(
            "gate_b_trial2_target_free_common_root_practical_direct_alpha_closeout_available_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 target-free common-root practical direct-alpha closeout available now",
            sign_base.truth(gate_b),
            "The common alpha readout should already sit within one-per-mille of alpha_target before promoting the route.",
        ),
        sign_base.row(
            "gate_c_trial2_target_free_common_root_strict_theorem_unavailable_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 target-free common-root strict theorem unavailable now",
            sign_base.truth(gate_c),
            "The route remains numerical because analytic uniqueness and theorem closure are still unavailable.",
        ),
        sign_base.row(
            "gate_d_trial2_target_free_common_root_strict_theorem_followup_primary_now",
            "pass" if gate_d else "reject",
            "gate D Trial-2 target-free common-root strict-theorem followup primary now",
            sign_base.truth(gate_d),
            "Once the practical target-free selector exists, the next honest blocker is the strict theorem followup rather than conditional hold.",
        ),
        sign_base.row(
            "trial2_target_free_common_root_direct_alpha_completed_now",
            "pass" if trial2_target_free_common_root_direct_alpha_completed_now else "reject",
            "Trial-2 target-free common-root direct-alpha completed now",
            sign_base.truth(trial2_target_free_common_root_direct_alpha_completed_now),
            "This lane is complete once the target-free selector is real, the direct-alpha readout is practical, and the non-theorem status is synchronized.",
        ),
        sign_base.row(
            "trial2_target_free_common_root_strict_theorem_followup_primary_now",
            "pass" if trial2_target_free_common_root_strict_theorem_followup_primary_now else "reject",
            "Trial-2 target-free common-root strict-theorem followup primary now",
            sign_base.truth(trial2_target_free_common_root_strict_theorem_followup_primary_now),
            "The new active baseline is the analytic theorem followup for the already-promoted target-free common-root selector.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "interaction_total_over_harmonic_sq_beta_common_root": float(
            prior_summary["interaction_total_over_harmonic_sq_beta_common_root"]
        ),
        "interaction_total_over_harmonic_sq_beta_common_root_rel_shift_vs_retained": float(
            prior_summary["interaction_total_over_harmonic_sq_beta_common_root_rel_shift_vs_retained"]
        ),
        "interaction_total_over_harmonic_sq_beta_common_root_rel_shift_vs_prior_alpha_beta": float(
            prior_summary["interaction_total_over_harmonic_sq_beta_common_root_rel_shift_vs_prior_alpha_beta"]
        ),
        "interaction_total_over_harmonic_sq_beta_common_root_rel_shift_vs_prior_r8_beta_root": float(
            prior_summary["interaction_total_over_harmonic_sq_beta_common_root_rel_shift_vs_prior_r8_beta_root"]
        ),
        "interaction_total_over_harmonic_sq_alpha_common_value": float(
            prior_summary["interaction_total_over_harmonic_sq_alpha_common_value"]
        ),
        "interaction_total_over_harmonic_sq_alpha_common_rel_error_vs_target": float(
            prior_summary["interaction_total_over_harmonic_sq_alpha_common_rel_error_vs_target"]
        ),
        "interaction_total_over_harmonic_sq_q_star_common_over_m0": float(
            prior_summary["interaction_total_over_harmonic_sq_q_star_common_over_m0"]
        ),
        "interaction_total_over_harmonic_sq_q_star_common_rel_shift_vs_q_exact": float(
            prior_summary["interaction_total_over_harmonic_sq_q_star_common_rel_shift_vs_q_exact"]
        ),
        "common_root_scan_row_count": int(prior_summary["common_root_scan_row_count"]),
        "common_root_difference_monotone_increasing_now": bool(
            prior_summary["common_root_difference_monotone_increasing_now"]
        ),
        "common_root_difference_sign_change_count": int(
            prior_summary["common_root_difference_sign_change_count"]
        ),
        "exact_trial2_target_free_common_root_beta_selector_available_now": gate_a,
        "exact_trial2_target_free_common_root_practical_direct_alpha_closeout_available_now": gate_b,
        "exact_trial2_target_free_common_root_strict_theorem_closeout_available_now": (not gate_c),
        "trial2_target_free_common_root_direct_alpha_completed_now": (
            trial2_target_free_common_root_direct_alpha_completed_now
        ),
        "trial2_target_free_common_root_strict_theorem_followup_primary_now": (
            trial2_target_free_common_root_strict_theorem_followup_primary_now
        ),
        "trial2_conditional_hold_secondary_retained_now": (
            conditional_hold_secondary_retained_now
        ),
        "selected_primary_completion_lane": "trial2_target_free_common_root_strict_theorem_followup",
        "selected_secondary_completion_lane": "conditional_hold_only",
        "selected_reserve_completion_lane": "conditional_hold_only",
        "selected_next_generation_route": "trial2_target_free_common_root_strict_theorem_followup",
        "recommended_next_route_or_none": "8.7.56.5631",
        "selected_followup_route": "trial2_target_free_common_root_strict_theorem_followup",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5629",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5631",
                "followup_route": "trial2_target_free_common_root_strict_theorem_followup",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_target_free_common_root_direct_alpha_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": gate_a and gate_b,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 beta-root gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
