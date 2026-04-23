#!/usr/bin/env python3
"""Generate 8.7.56.5707-.5710 Trial-2 final closure gate artifacts."""

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
        "8.7.56.5703-5706",
        "updated_pack_trial2_beta_sensitivity_final_closure_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5707-5710"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "beta-sensitivity final closure gate / v3-refinement defer refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_final_closure_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_final_closure_audited_"
    "first_principles_direct_alpha_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_first_principles_direct_alpha_closure_completed_"
    "pure_analytic_refinement_deferred_v3_conditional_reopen_only_next"
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
    """Return formulas used by the final closure gate."""
    return {
        "gate_a": "Gate A = first-principles direct-alpha closure is completed now",
        "gate_b": "Gate B = pure analytic operator-level continuum refinement is deferred to v3 now",
        "gate_c": "Gate C = there is no unconditional next official branch now",
    }


# 関数: `.5707-.5710` を実行する。

def main() -> None:
    """Execute the Trial-2 final closure gate / refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary["exact_trial2_first_principles_direct_alpha_closure_completed_now"]
    )
    gate_b = bool(
        gate_a
        and prior_summary[
            "exact_trial2_pure_analytic_operator_level_continuum_refinement_deferred_to_v3_now"
        ]
    )
    gate_c = bool(gate_b)

    trial2_beta_sensitivity_final_closure_followup_lane_completed_now = bool(gate_a)
    trial2_first_principles_direct_alpha_closure_completed_now = bool(gate_a)
    no_unconditional_next_official_branch_now = bool(gate_c)

    rows = [
        sign_base.row(
            "gate_a_trial2_first_principles_direct_alpha_closure_completed_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 first-principles direct-alpha closure completed now",
            sign_base.truth(gate_a),
            "The official declaration passes once the synchronized theorem-support chain is judged strong enough to treat alpha_* as completed from the frozen action.",
        ),
        sign_base.row(
            "gate_b_trial2_pure_analytic_operator_level_continuum_refinement_deferred_to_v3_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 pure analytic operator-level continuum refinement deferred to v3 now",
            sign_base.truth(gate_b),
            "The remaining operator-level continuum sharpening is retained as mathematical refinement, not as a blocker for the direct-alpha closure claim.",
        ),
        sign_base.row(
            "gate_c_trial2_no_unconditional_next_official_branch_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 no unconditional next official branch now",
            sign_base.truth(gate_c),
            "After closure, there is no unconditional next branch inside the current pack; only conditional reopen for genuinely new theorem or computation routes remains.",
        ),
        sign_base.row(
            "trial2_beta_sensitivity_final_closure_followup_lane_completed_now",
            "pass"
            if trial2_beta_sensitivity_final_closure_followup_lane_completed_now
            else "reject",
            "Trial-2 beta-sensitivity final closure followup lane completed now",
            sign_base.truth(
                trial2_beta_sensitivity_final_closure_followup_lane_completed_now
            ),
            "This lane is complete once the current pack reaches an honest final closure verdict instead of another support-level replay.",
        ),
        sign_base.row(
            "trial2_first_principles_direct_alpha_closure_completed_now",
            "pass"
            if trial2_first_principles_direct_alpha_closure_completed_now
            else "reject",
            "Trial-2 first-principles direct-alpha closure completed now",
            sign_base.truth(
                trial2_first_principles_direct_alpha_closure_completed_now
            ),
            "The frozen-action derivation chain now closes at alpha_* = alpha_qstar(beta_*) = R8(beta_*), with only v3-level mathematical refinement still open.",
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
        "exact_trial2_first_principles_direct_alpha_closure_completed_now": bool(
            trial2_first_principles_direct_alpha_closure_completed_now
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now": bool(
            prior_summary[
                "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now"
            ]
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_deferred_to_v3_now": bool(
            gate_b
        ),
        "trial2_beta_sensitivity_final_closure_followup_lane_completed_now": bool(
            trial2_beta_sensitivity_final_closure_followup_lane_completed_now
        ),
        "no_unconditional_next_official_branch_now": bool(
            no_unconditional_next_official_branch_now
        ),
        "selected_next_generation_route": None,
        "recommended_next_route_or_none": None,
    }

    payload = {
        "step_tag": STEP_TAG,
        "step_name": STEP_NAME,
        "summary": summary,
        "rows": rows,
        "formulae": build_formulae(),
        "notes": {
            "gate_meaning": (
                "First-principles direct-alpha closure is now complete, while "
                "pure analytic operator-level continuum refinement is deferred "
                "to v3.0."
            ),
        },
    }
    written = write_artifact("declaration_gate", payload)
    print(json.dumps({"ok": True, "written": written}, ensure_ascii=False))


if __name__ == "__main__":
    main()
