#!/usr/bin/env python3
"""Generate 8.7.56.5699-.5702 Trial-2 uniqueness-anchor gate artifacts."""

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
        "8.7.56.5695-5698",
        "updated_pack_trial2_beta_sensitivity_uniqueness_anchor_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5699-5702"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "beta-sensitivity uniqueness-anchor gate / conditional-hold secondary refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_uniqueness_anchor_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_uniqueness_anchor_audited_"
    "final_closure_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_uniqueness_anchor_sign_support_completed_"
    "final_closure_followup_primary_conditional_hold_secondary_next"
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
    """Return formulas used by the uniqueness-anchor gate."""
    return {
        "gate_a": "Gate A = uniqueness-anchor support is available now",
        "gate_b": "Gate B = final closure followup is promoted next",
        "gate_c": "Gate C = conditional hold stays secondary fallback",
    }


# 関数: `.5699-.5702` を実行する。

def main() -> None:
    """Execute the Trial-2 beta-sensitivity uniqueness-anchor gate / refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_trial2_beta_sensitivity_uniqueness_anchor_support_available_now"
        ]
    )
    gate_b = bool(
        gate_a
        and prior_summary[
            "updated_pack_trial2_beta_sensitivity_final_closure_followup_required_now"
        ]
    )
    gate_c = bool(gate_b)

    trial2_beta_sensitivity_uniqueness_anchor_followup_lane_completed_now = bool(
        gate_a
    )
    trial2_beta_sensitivity_final_closure_followup_promoted_now = bool(gate_b)
    trial2_conditional_hold_secondary_now = bool(gate_c)

    rows = [
        sign_base.row(
            "gate_a_trial2_beta_sensitivity_uniqueness_anchor_support_available_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 beta-sensitivity uniqueness-anchor support available now",
            sign_base.truth(gate_a),
            "The uniqueness-anchor layer becomes official only once lower/upper anchors, the retained common root, the sampled selector, and local transversality all live on one synchronized support surface.",
        ),
        sign_base.row(
            "gate_b_trial2_beta_sensitivity_final_closure_followup_promoted_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 beta-sensitivity final closure followup promoted now",
            sign_base.truth(gate_b),
            "Once uniqueness-anchor support is fixed honestly, the next blocker is the final closure verdict rather than another support-level replay.",
        ),
        sign_base.row(
            "gate_c_trial2_conditional_hold_secondary_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 conditional hold secondary now",
            sign_base.truth(gate_c),
            "Conditional hold remains only as fallback while the final closure route is live.",
        ),
        sign_base.row(
            "trial2_beta_sensitivity_uniqueness_anchor_followup_lane_completed_now",
            "pass"
            if trial2_beta_sensitivity_uniqueness_anchor_followup_lane_completed_now
            else "reject",
            "Trial-2 beta-sensitivity uniqueness-anchor followup lane completed now",
            sign_base.truth(
                trial2_beta_sensitivity_uniqueness_anchor_followup_lane_completed_now
            ),
            "This lane is complete once the branch honestly decides whether the retained anchors can be synchronized with the derivative-chain support; here they can.",
        ),
        sign_base.row(
            "trial2_beta_sensitivity_final_closure_followup_promoted_now",
            "pass"
            if trial2_beta_sensitivity_final_closure_followup_promoted_now
            else "reject",
            "Trial-2 beta-sensitivity final closure followup promoted now",
            sign_base.truth(
                trial2_beta_sensitivity_final_closure_followup_promoted_now
            ),
            "The live blocker is now the final closure verdict, not the uniqueness-anchor support itself.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_anchor_lower": float(prior_summary["beta_anchor_lower"]),
        "beta_anchor_upper": float(prior_summary["beta_anchor_upper"]),
        "delta_common_lower_anchor": float(prior_summary["delta_common_lower_anchor"]),
        "delta_common_upper_anchor": float(prior_summary["delta_common_upper_anchor"]),
        "sampled_anchor_gap_span": float(prior_summary["sampled_anchor_gap_span"]),
        "lower_anchor_abs_margin": float(prior_summary["lower_anchor_abs_margin"]),
        "upper_anchor_abs_margin": float(prior_summary["upper_anchor_abs_margin"]),
        "beta_common_root": float(prior_summary["beta_common_root"]),
        "alpha_common_value": float(prior_summary["alpha_common_value"]),
        "alpha_common_rel_error_vs_target": float(
            prior_summary["alpha_common_rel_error_vs_target"]
        ),
        "derivative_transversality_min": float(
            prior_summary["derivative_transversality_min"]
        ),
        "derivative_transversality_max": float(
            prior_summary["derivative_transversality_max"]
        ),
        "trial2_beta_sensitivity_uniqueness_anchor_followup_lane_completed_now": (
            trial2_beta_sensitivity_uniqueness_anchor_followup_lane_completed_now
        ),
        "trial2_beta_sensitivity_final_closure_followup_promoted_now": (
            trial2_beta_sensitivity_final_closure_followup_promoted_now
        ),
        "trial2_conditional_hold_secondary_now": trial2_conditional_hold_secondary_now,
        "selected_next_generation_route": (
            "trial2_beta_sensitivity_final_closure_followup"
        ),
        "recommended_next_route_or_none": (
            "trial2_beta_sensitivity_final_closure_followup"
        ),
    }

    payload = {
        "step_tag": STEP_TAG,
        "step_name": STEP_NAME,
        "summary": summary,
        "rows": rows,
        "formulae": build_formulae(),
        "notes": {
            "gate_meaning": (
                "Uniqueness-anchor support is now retained, but the final closure "
                "verdict is still missing."
            ),
        },
    }
    written = write_artifact("declaration_gate", payload)
    print(json.dumps({"ok": True, "written": written}, ensure_ascii=False))


if __name__ == "__main__":
    main()
