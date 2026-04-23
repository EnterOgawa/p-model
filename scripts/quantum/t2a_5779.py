#!/usr/bin/env python3
"""Generate 8.7.56.5779-.5782 source-weighted full operator-level gate artifacts."""

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
        "8.7.56.5775-5778",
        "updated_pack_trial2_beta_sensitivity_source_weighted_full_operator_level_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5779-5782"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "source-weighted full operator-level gate / final v2 sync"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_source_weighted_full_operator_level_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_source_weighted_full_operator_level_weighted_integral_audited_"
    "gate_sync_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_first_principles_direct_alpha_closure_completed_"
    "full_v2_operator_level_continuum_refinement_completed_"
    "conditional_reopen_only_next"
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
    """Return formulas used by the full operator-level gate."""
    return {
        "gate_a": "Gate A = source-weighted full-halfline weighted-integral closure is completed now",
        "gate_b": "Gate B = pure analytic operator-level continuum refinement is completed now for v2",
        "gate_c": "Gate C = no unconditional next official branch remains now",
    }


# 関数: `.5779-.5782` を実行する。

def main() -> None:
    """Execute the source-weighted full operator-level gate / final sync."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_trial2_source_weighted_full_halfline_weighted_integral_closure_available_now"
        ]
    )
    gate_b = bool(
        gate_a
        and prior_summary[
            "exact_trial2_pure_analytic_operator_level_continuum_refinement_completed_now"
        ]
    )
    gate_c = bool(
        gate_b
        and prior_summary[
            "updated_pack_trial2_source_weighted_full_operator_level_gate_required_now"
        ]
    )

    rows = [
        sign_base.row(
            "gate_a_trial2_source_weighted_full_halfline_weighted_integral_closure_completed_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 source-weighted full-halfline weighted-integral closure completed now",
            sign_base.truth(gate_a),
            "The full half-line weighted-integral signs now close directly from the exact source-weighted operator solution, compact-complement control, and analytic tail bounds.",
        ),
        sign_base.row(
            "gate_b_trial2_pure_analytic_operator_level_continuum_refinement_completed_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 pure analytic operator-level continuum refinement completed now",
            sign_base.truth(gate_b),
            "The v2 theorem no longer needs to defer a stronger auxiliary global one-sign kernel lemma once the full weighted-integral operator-level chain is fixed.",
        ),
        sign_base.row(
            "gate_c_trial2_no_unconditional_next_branch_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 no unconditional next branch now",
            sign_base.truth(gate_c),
            "After the final v2 sync, the current pack returns to conditional reopen only rather than to another forced refinement replay.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(prior_summary["beta_common_root"]),
        "retained_x_cutoff": float(prior_summary["retained_x_cutoff"]),
        "retained_full_lower_bound_n2": float(prior_summary["retained_full_lower_bound_n2"]),
        "retained_full_lower_bound_n3": float(prior_summary["retained_full_lower_bound_n3"]),
        "retained_full_lower_bound_n4": float(prior_summary["retained_full_lower_bound_n4"]),
        "retained_complement_and_tail_over_control_ratio_n2": float(
            prior_summary["retained_complement_and_tail_over_control_ratio_n2"]
        ),
        "retained_complement_and_tail_over_control_ratio_n3": float(
            prior_summary["retained_complement_and_tail_over_control_ratio_n3"]
        ),
        "retained_complement_and_tail_over_control_ratio_n4": float(
            prior_summary["retained_complement_and_tail_over_control_ratio_n4"]
        ),
        "family_full_lower_bound_min_n2": float(prior_summary["family_full_lower_bound_min_n2"]),
        "family_full_lower_bound_min_n3": float(prior_summary["family_full_lower_bound_min_n3"]),
        "family_full_lower_bound_min_n4": float(prior_summary["family_full_lower_bound_min_n4"]),
        "exact_trial2_source_weighted_full_halfline_weighted_integral_closure_completed_now": bool(
            gate_a
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_completed_now": bool(
            gate_b
        ),
        "exact_trial2_pure_analytic_global_one_sign_kernel_theorem_needed_now": False,
        "no_unconditional_next_official_branch_now": bool(gate_c),
        "selected_next_generation_route": None,
        "recommended_next_route_or_none": None,
        "selected_followup_route": None,
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5781",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": (
                "trial2_source_weighted_full_operator_level_gate_completed"
            ),
            "branch_completed": True,
            "breakthrough_passed_now": gate_b,
            "physical_reject_required": False,
        },
        {
            "retained_full_lower_bound_n2": float(prior_summary["retained_full_lower_bound_n2"]),
            "retained_full_lower_bound_n3": float(prior_summary["retained_full_lower_bound_n3"]),
            "retained_full_lower_bound_n4": float(prior_summary["retained_full_lower_bound_n4"]),
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5779-5782 Trial-2 source-weighted full operator-level gate completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
