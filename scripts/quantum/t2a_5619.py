#!/usr/bin/env python3
"""Generate 8.7.56.5619-.5622 Trial-2 exact-relation gate artifacts."""

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
        "8.7.56.5615-5618",
        "updated_pack_trial2_interaction_total_over_harmonic_sq_exact_relation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5619-5622"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "interaction_total_over_harmonic_sq gate / conditional-hold refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_interaction_total_over_harmonic_sq_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_interaction_total_over_harmonic_sq_exact_relation_weighted_eom_"
    "one_third_factor_local_beta_root_followup_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_interaction_total_over_harmonic_sq_exact_relation_audited_"
    "local_beta_root_followup_primary_conditional_hold_secondary_next"
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
    """Return formulas used by the exact-relation gate."""
    return {
        "gate_a": "Gate A = exact weighted-EOM relation is available now",
        "gate_b": "Gate B = strict target-free closeout is still unavailable now",
        "gate_c": "Gate C = local beta-root followup becomes the primary next blocker",
        "gate_d": "Gate D = conditional hold remains the secondary fallback",
    }


# 関数: `.5619-.5622` を実行する。

def main() -> None:
    """Execute the Trial-2 interaction_total_over_harmonic_sq gate."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_trial2_interaction_total_over_harmonic_sq_exact_relation_available_now"
        ]
    )
    gate_b = bool(
        not prior_summary[
            "exact_trial2_interaction_total_over_harmonic_sq_target_free_closeout_available_now"
        ]
    )
    gate_c = bool(
        prior_summary[
            "updated_pack_trial2_interaction_total_over_harmonic_sq_beta_root_followup_primary_next_now"
        ]
    )
    gate_d = True

    trial2_interaction_total_over_harmonic_sq_exact_relation_completed_now = bool(
        gate_a and gate_b
    )
    trial2_interaction_total_over_harmonic_sq_beta_root_followup_primary_now = bool(
        trial2_interaction_total_over_harmonic_sq_exact_relation_completed_now and gate_c
    )
    trial2_conditional_hold_secondary_retained_now = bool(gate_d)

    rows = [
        sign_base.row(
            "gate_a_trial2_interaction_total_over_harmonic_sq_exact_relation_available_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 interaction_total_over_harmonic_sq exact relation available now",
            sign_base.truth(gate_a),
            "The branch only advances if the weighted-EOM elimination truly reconstructs the screened ratio as one exact relation.",
        ),
        sign_base.row(
            "gate_b_trial2_interaction_total_over_harmonic_sq_target_free_closeout_unavailable_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 interaction_total_over_harmonic_sq target-free closeout unavailable now",
            sign_base.truth(gate_b),
            "The exact relation is not yet a theorem closeout because the beta-root selection still needs an independent target-free law.",
        ),
        sign_base.row(
            "gate_c_trial2_interaction_total_over_harmonic_sq_beta_root_followup_primary_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 interaction_total_over_harmonic_sq beta-root followup primary now",
            sign_base.truth(gate_c),
            "Because the exact relation is real and a local beta root is available, the next honest blocker is the beta-root followup rather than conditional hold.",
        ),
        sign_base.row(
            "gate_d_trial2_conditional_hold_secondary_retained_now",
            "pass" if gate_d else "reject",
            "gate D Trial-2 conditional hold secondary retained now",
            sign_base.truth(gate_d),
            "Conditional hold remains the fallback if the beta-root followup dead-ends honestly.",
        ),
        sign_base.row(
            "trial2_interaction_total_over_harmonic_sq_exact_relation_completed_now",
            "pass" if trial2_interaction_total_over_harmonic_sq_exact_relation_completed_now else "reject",
            "Trial-2 interaction_total_over_harmonic_sq exact relation completed now",
            sign_base.truth(trial2_interaction_total_over_harmonic_sq_exact_relation_completed_now),
            "This lane is complete once the exact relation is fixed and its non-closeout status is synchronized.",
        ),
        sign_base.row(
            "trial2_interaction_total_over_harmonic_sq_beta_root_followup_primary_now",
            "pass" if trial2_interaction_total_over_harmonic_sq_beta_root_followup_primary_now else "reject",
            "Trial-2 interaction_total_over_harmonic_sq beta-root followup primary now",
            sign_base.truth(trial2_interaction_total_over_harmonic_sq_beta_root_followup_primary_now),
            "The new active baseline is the law that would select the exact-relation beta root without using alpha_target as comparator.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_beta1": float(prior_summary["retained_beta1"]),
        "prior_alpha_beta_root": float(prior_summary["prior_alpha_beta_root"]),
        "interaction_total_over_harmonic_sq_beta_root": float(
            prior_summary["interaction_total_over_harmonic_sq_beta_root"]
        ),
        "interaction_total_over_harmonic_sq_beta_root_rel_shift_vs_retained": float(
            prior_summary["interaction_total_over_harmonic_sq_beta_root_rel_shift_vs_retained"]
        ),
        "interaction_total_over_harmonic_sq_beta_root_rel_shift_vs_prior_alpha_beta": float(
            prior_summary["interaction_total_over_harmonic_sq_beta_root_rel_shift_vs_prior_alpha_beta"]
        ),
        "retained_exact_relation_value": float(prior_summary["retained_exact_relation_value"]),
        "retained_exact_relation_rel_error_vs_target": float(
            prior_summary["retained_exact_relation_rel_error_vs_target"]
        ),
        "retained_leading_relation_cubic_dominant": float(
            prior_summary["retained_leading_relation_cubic_dominant"]
        ),
        "retained_leading_relation_rel_error_vs_target": float(
            prior_summary["retained_leading_relation_rel_error_vs_target"]
        ),
        "near_exact_relation_value": float(prior_summary["near_exact_relation_value"]),
        "near_exact_relation_rel_error_vs_target": float(
            prior_summary["near_exact_relation_rel_error_vs_target"]
        ),
        "exact_trial2_interaction_total_over_harmonic_sq_exact_relation_available_now": gate_a,
        "exact_trial2_interaction_total_over_harmonic_sq_target_free_closeout_available_now": (
            not gate_b
        ),
        "trial2_interaction_total_over_harmonic_sq_exact_relation_completed_now": (
            trial2_interaction_total_over_harmonic_sq_exact_relation_completed_now
        ),
        "trial2_interaction_total_over_harmonic_sq_beta_root_followup_primary_now": (
            trial2_interaction_total_over_harmonic_sq_beta_root_followup_primary_now
        ),
        "trial2_conditional_hold_secondary_retained_now": (
            trial2_conditional_hold_secondary_retained_now
        ),
        "selected_primary_completion_lane": (
            "trial2_interaction_total_over_harmonic_sq_beta_root_followup"
        ),
        "selected_secondary_completion_lane": "conditional_hold_only",
        "selected_reserve_completion_lane": "conditional_hold_only",
        "selected_next_generation_route": (
            "trial2_interaction_total_over_harmonic_sq_beta_root_followup"
        ),
        "recommended_next_route_or_none": "8.7.56.5623",
        "selected_followup_route": "trial2_interaction_total_over_harmonic_sq_beta_root_followup",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5621",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5623",
                "followup_route": "trial2_interaction_total_over_harmonic_sq_beta_root_followup",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_interaction_total_over_harmonic_sq_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 interaction_total_over_harmonic_sq gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
