#!/usr/bin/env python3
"""Generate 8.7.56.5691-.5694 Trial-2 derivative-chain gate artifacts."""

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
        "8.7.56.5687-5690",
        "updated_pack_trial2_beta_sensitivity_derivative_chain_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5691-5694"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "beta-sensitivity derivative-chain gate / conditional-hold secondary refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_derivative_chain_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_derivative_chain_audited_"
    "uniqueness_anchor_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_derivative_chain_sign_support_completed_"
    "uniqueness_anchor_followup_primary_conditional_hold_secondary_next"
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
    """Return formulas used by the derivative-chain gate."""
    return {
        "gate_a": "Gate A = derivative-chain sign support is available now",
        "gate_b": "Gate B = uniqueness-anchor followup is promoted next",
        "gate_c": "Gate C = conditional hold stays secondary fallback",
    }


# 関数: `.5691-.5694` を実行する。

def main() -> None:
    """Execute the Trial-2 beta-sensitivity derivative-chain gate / refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_trial2_alpha_qstar_derivative_chain_positive_local_support_available_now"
        ]
        and prior_summary[
            "exact_trial2_r8_derivative_chain_negative_local_support_available_now"
        ]
        and prior_summary[
            "exact_trial2_delta_common_derivative_chain_positive_local_support_available_now"
        ]
    )
    gate_b = bool(
        gate_a
        and prior_summary[
            "updated_pack_trial2_beta_sensitivity_uniqueness_anchor_followup_required_now"
        ]
    )
    gate_c = bool(gate_b)

    trial2_beta_sensitivity_derivative_chain_followup_lane_completed_now = bool(gate_a)
    trial2_beta_sensitivity_uniqueness_anchor_followup_promoted_now = bool(gate_b)
    trial2_conditional_hold_secondary_now = bool(gate_c)

    rows = [
        sign_base.row(
            "gate_a_trial2_beta_sensitivity_derivative_chain_sign_support_available_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 beta-sensitivity derivative-chain sign support available now",
            sign_base.truth(gate_a),
            "The derivative-chain route only becomes official once alpha_qstar, R8, and Delta_common all keep the retained sign pattern across the full local h family.",
        ),
        sign_base.row(
            "gate_b_trial2_beta_sensitivity_uniqueness_anchor_followup_promoted_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 beta-sensitivity uniqueness-anchor followup promoted now",
            sign_base.truth(gate_b),
            "Once derivative-chain signs are fixed honestly, the next blocker is the anchor theorem that turns local transversality into the unique common-root statement.",
        ),
        sign_base.row(
            "gate_c_trial2_conditional_hold_secondary_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 conditional hold secondary now",
            sign_base.truth(gate_c),
            "Conditional hold remains only as fallback while the uniqueness-anchor theorem route is live.",
        ),
        sign_base.row(
            "trial2_beta_sensitivity_derivative_chain_followup_lane_completed_now",
            "pass"
            if trial2_beta_sensitivity_derivative_chain_followup_lane_completed_now
            else "reject",
            "Trial-2 beta-sensitivity derivative-chain followup lane completed now",
            sign_base.truth(
                trial2_beta_sensitivity_derivative_chain_followup_lane_completed_now
            ),
            "This lane is complete once the derivative-chain support is either promoted or honestly closed; here it is promoted.",
        ),
        sign_base.row(
            "trial2_beta_sensitivity_uniqueness_anchor_followup_promoted_now",
            "pass"
            if trial2_beta_sensitivity_uniqueness_anchor_followup_promoted_now
            else "reject",
            "Trial-2 beta-sensitivity uniqueness-anchor followup promoted now",
            sign_base.truth(
                trial2_beta_sensitivity_uniqueness_anchor_followup_promoted_now
            ),
            "The final blocker is no longer sign support itself, but the anchor statement that yields existence and uniqueness of the common root.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(prior_summary["beta_common_root"]),
        "alpha_total_derivative_min": float(prior_summary["alpha_total_derivative_min"]),
        "alpha_total_derivative_max": float(prior_summary["alpha_total_derivative_max"]),
        "r8_total_derivative_min": float(prior_summary["r8_total_derivative_min"]),
        "r8_total_derivative_max": float(prior_summary["r8_total_derivative_max"]),
        "delta_common_derivative_min": float(
            prior_summary["delta_common_derivative_min"]
        ),
        "delta_common_derivative_max": float(
            prior_summary["delta_common_derivative_max"]
        ),
        "trial2_beta_sensitivity_derivative_chain_followup_lane_completed_now": (
            trial2_beta_sensitivity_derivative_chain_followup_lane_completed_now
        ),
        "trial2_beta_sensitivity_uniqueness_anchor_followup_promoted_now": (
            trial2_beta_sensitivity_uniqueness_anchor_followup_promoted_now
        ),
        "trial2_conditional_hold_secondary_now": trial2_conditional_hold_secondary_now,
        "selected_next_generation_route": (
            "trial2_beta_sensitivity_uniqueness_anchor_followup"
        ),
        "recommended_next_route_or_none": (
            "trial2_beta_sensitivity_uniqueness_anchor_followup"
        ),
        "selected_followup_route": "trial2_beta_sensitivity_uniqueness_anchor_followup",
        "selected_followup_route_or_none": (
            "trial2_beta_sensitivity_uniqueness_anchor_followup"
        ),
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5693",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_beta_sensitivity_derivative_chain_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": gate_b,
            "physical_reject_required": False,
        },
        {
            "alpha_total_derivative_rel_spread": float(
                prior_summary["alpha_total_derivative_rel_spread"]
            ),
            "r8_total_derivative_rel_spread": float(
                prior_summary["r8_total_derivative_rel_spread"]
            ),
            "delta_common_derivative_rel_spread": float(
                prior_summary["delta_common_derivative_rel_spread"]
            ),
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print(
        "[done] 8.7.56.5691-5694 Trial-2 beta-sensitivity derivative-chain gate completed"
    )
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
