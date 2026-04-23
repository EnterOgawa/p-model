#!/usr/bin/env python3
"""Generate 8.7.56.5683-.5686 Trial-2 operator-level followup gate artifacts."""

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
        "8.7.56.5679-5682",
        "updated_pack_trial2_beta_sensitivity_operator_level_spectral_projection_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5683-5686"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "beta-sensitivity operator-level spectral-projection gate / "
    "conditional-hold secondary refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_operator_level_spectral_projection_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_operator_level_spectral_projection_audited_"
    "weighted_integral_sign_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_weighted_integral_sign_support_completed_"
    "derivative_chain_followup_primary_conditional_hold_secondary_next"
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
    """Return formulas used by the operator-level gate."""
    return {
        "gate_a": "Gate A = weighted-integral sign support is available now",
        "gate_b": "Gate B = derivative-chain followup is promoted next",
        "gate_c": "Gate C = conditional hold stays secondary fallback",
    }


# 関数: `.5683-.5686` を実行する。

def main() -> None:
    """Execute the Trial-2 operator-level spectral-projection gate / refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_trial2_beta_sensitivity_weighted_integral_sign_support_available_now"
        ]
    )
    gate_b = bool(
        gate_a
        and prior_summary[
            "updated_pack_trial2_beta_sensitivity_derivative_chain_followup_required_now"
        ]
    )
    gate_c = bool(gate_b)

    trial2_beta_sensitivity_operator_level_spectral_projection_followup_lane_completed_now = bool(
        gate_a
    )
    trial2_beta_sensitivity_derivative_chain_followup_promoted_now = bool(gate_b)
    trial2_conditional_hold_secondary_now = bool(gate_c)

    rows = [
        sign_base.row(
            "gate_a_trial2_beta_sensitivity_weighted_integral_sign_support_available_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 beta-sensitivity weighted-integral sign support available now",
            sign_base.truth(gate_a),
            "The operator-level followup becomes official only once the retained control window fixes the signs of dI_2/dbeta, dI_3/dbeta, and dI_4/dbeta without boundary-complement sign reversal.",
        ),
        sign_base.row(
            "gate_b_trial2_beta_sensitivity_derivative_chain_followup_promoted_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 beta-sensitivity derivative-chain followup promoted now",
            sign_base.truth(gate_b),
            "Once weighted-integral signs are controlled, the next blocker is the derivative-chain theorem that must produce Delta_common'(beta) > 0 and then uniqueness of beta_*.",
        ),
        sign_base.row(
            "gate_c_trial2_conditional_hold_secondary_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 conditional hold secondary now",
            sign_base.truth(gate_c),
            "Conditional hold remains only as fallback while the derivative-chain route is live.",
        ),
        sign_base.row(
            "trial2_beta_sensitivity_operator_level_spectral_projection_followup_lane_completed_now",
            "pass"
            if trial2_beta_sensitivity_operator_level_spectral_projection_followup_lane_completed_now
            else "reject",
            "Trial-2 beta-sensitivity operator-level spectral-projection followup lane completed now",
            sign_base.truth(
                trial2_beta_sensitivity_operator_level_spectral_projection_followup_lane_completed_now
            ),
            "This lane is complete once the branch honestly decides whether the operator-level route at least fixes the weighted-integral signs; here it does.",
        ),
        sign_base.row(
            "trial2_beta_sensitivity_derivative_chain_followup_promoted_now",
            "pass"
            if trial2_beta_sensitivity_derivative_chain_followup_promoted_now
            else "reject",
            "Trial-2 beta-sensitivity derivative-chain followup promoted now",
            sign_base.truth(
                trial2_beta_sensitivity_derivative_chain_followup_promoted_now
            ),
            "The live blocker is now the exact derivative chain from weighted-integral signs to Delta_common'(beta) > 0, not the operator-level support itself.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(prior_summary["beta_common_root"]),
        "control_window_x_min": float(prior_summary["control_window_x_min"]),
        "control_window_x_max": float(prior_summary["control_window_x_max"]),
        "control_window_continuum_margin_estimate": float(
            prior_summary["control_window_continuum_margin_estimate"]
        ),
        "control_window_last_rel_spread": float(
            prior_summary["control_window_last_rel_spread"]
        ),
        "d_i2_dbeta_min": float(prior_summary["d_i2_dbeta_min"]),
        "d_i2_dbeta_max": float(prior_summary["d_i2_dbeta_max"]),
        "d_i3_dbeta_min": float(prior_summary["d_i3_dbeta_min"]),
        "d_i3_dbeta_max": float(prior_summary["d_i3_dbeta_max"]),
        "d_i4_dbeta_min": float(prior_summary["d_i4_dbeta_min"]),
        "d_i4_dbeta_max": float(prior_summary["d_i4_dbeta_max"]),
        "boundary_complement_abs_fraction_max_n2": float(
            prior_summary["boundary_complement_abs_fraction_max_n2"]
        ),
        "boundary_complement_abs_fraction_max_n3": float(
            prior_summary["boundary_complement_abs_fraction_max_n3"]
        ),
        "boundary_complement_abs_fraction_max_n4": float(
            prior_summary["boundary_complement_abs_fraction_max_n4"]
        ),
        "delta_common_derivative_min": float(
            prior_summary["delta_common_derivative_min"]
        ),
        "delta_common_derivative_max": float(
            prior_summary["delta_common_derivative_max"]
        ),
        "delta_common_derivative_rel_spread": float(
            prior_summary["delta_common_derivative_rel_spread"]
        ),
        "trial2_beta_sensitivity_operator_level_spectral_projection_followup_lane_completed_now": (
            trial2_beta_sensitivity_operator_level_spectral_projection_followup_lane_completed_now
        ),
        "trial2_beta_sensitivity_derivative_chain_followup_promoted_now": (
            trial2_beta_sensitivity_derivative_chain_followup_promoted_now
        ),
        "trial2_conditional_hold_secondary_now": trial2_conditional_hold_secondary_now,
    }

    payload = {
        "step_tag": STEP_TAG,
        "step_name": STEP_NAME,
        "summary": summary,
        "rows": rows,
        "formulae": build_formulae(),
        "notes": {
            "gate_meaning": (
                "Weighted-integral sign support is now retained, but the exact "
                "derivative-chain theorem is still missing."
            ),
        },
    }
    written = write_artifact("declaration_gate", payload)
    print(json.dumps({"ok": True, "written": written}, ensure_ascii=False))


if __name__ == "__main__":
    main()
