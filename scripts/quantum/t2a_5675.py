#!/usr/bin/env python3
"""Generate 8.7.56.5675-.5678 Trial-2 continuum spectral-projection gate artifacts."""

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
        "8.7.56.5671-5674",
        "updated_pack_trial2_beta_sensitivity_continuum_spectral_projection_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5675-5678"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "beta-sensitivity continuum spectral-projection gate / conditional-hold secondary refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_continuum_spectral_projection_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_continuum_open_interval_support_audited_"
    "operator_level_theorem_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_continuum_open_interval_support_completed_"
    "operator_level_spectral_projection_followup_primary_"
    "conditional_hold_secondary_next"
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
    """Return formulas used by the continuum-support gate."""
    return {
        "gate_a": "Gate A = continuum open-interval support is available now",
        "gate_b": "Gate B = operator-level spectral-projection followup is promoted next",
        "gate_c": "Gate C = conditional hold stays secondary fallback",
    }


# 関数: `.5675-.5678` を実行する。

def main() -> None:
    """Execute the Trial-2 continuum spectral-projection gate / refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_trial2_beta_sensitivity_continuum_boundary_layer_support_available_now"
        ]
        and prior_summary[
            "exact_trial2_beta_sensitivity_continuum_gap_support_available_now"
        ]
        and prior_summary[
            "exact_trial2_beta_sensitivity_continuum_open_interval_support_available_now"
        ]
    )
    gate_b = bool(
        gate_a
        and prior_summary[
            "updated_pack_trial2_beta_sensitivity_operator_level_spectral_projection_followup_required_now"
        ]
    )
    gate_c = bool(gate_b)

    trial2_beta_sensitivity_continuum_spectral_projection_followup_lane_completed_now = bool(
        gate_a
    )
    trial2_beta_sensitivity_operator_level_spectral_projection_followup_promoted_now = bool(
        gate_b
    )
    trial2_conditional_hold_secondary_now = bool(gate_c)

    rows = [
        sign_base.row(
            "gate_a_trial2_beta_sensitivity_continuum_open_interval_support_available_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 beta-sensitivity continuum open-interval support available now",
            sign_base.truth(gate_a),
            "The continuum followup only becomes official once the boundary-layer explanation, spectral gap support, and fixed interior-window positivity all survive refinement together.",
        ),
        sign_base.row(
            "gate_b_trial2_beta_sensitivity_operator_level_spectral_projection_followup_promoted_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 beta-sensitivity operator-level spectral-projection followup promoted now",
            sign_base.truth(gate_b),
            "Once continuum-support numerics are fixed honestly, the next blocker is the operator-level theorem that removes the remaining discretization surface.",
        ),
        sign_base.row(
            "gate_c_trial2_conditional_hold_secondary_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 conditional hold secondary now",
            sign_base.truth(gate_c),
            "Conditional hold remains only as fallback while the operator-level spectral-projection route is live.",
        ),
        sign_base.row(
            "trial2_beta_sensitivity_continuum_spectral_projection_followup_lane_completed_now",
            "pass"
            if trial2_beta_sensitivity_continuum_spectral_projection_followup_lane_completed_now
            else "reject",
            "Trial-2 beta-sensitivity continuum spectral-projection followup lane completed now",
            sign_base.truth(
                trial2_beta_sensitivity_continuum_spectral_projection_followup_lane_completed_now
            ),
            "This lane is complete once the branch honestly decides whether continuum-support numerics survive beyond the discrete theorem; here they do.",
        ),
        sign_base.row(
            "trial2_beta_sensitivity_operator_level_spectral_projection_followup_promoted_now",
            "pass"
            if trial2_beta_sensitivity_operator_level_spectral_projection_followup_promoted_now
            else "reject",
            "Trial-2 beta-sensitivity operator-level spectral-projection followup promoted now",
            sign_base.truth(
                trial2_beta_sensitivity_operator_level_spectral_projection_followup_promoted_now
            ),
            "The discrete theorem and continuum-support numerics are no longer the blocker; the live blocker is the operator-level theorem itself.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(prior_summary["beta_common_root"]),
        "boundary_layer_rel_spread": float(prior_summary["boundary_layer_rel_spread"]),
        "continuum_row_2400_global_margin_over_step": float(
            prior_summary["continuum_row_2400_global_margin_over_step"]
        ),
        "continuum_row_2400_lambda_1": float(
            prior_summary["continuum_row_2400_lambda_1"]
        ),
        "continuum_row_2400_lambda_2": float(
            prior_summary["continuum_row_2400_lambda_2"]
        ),
        "lambda_1_continuum_estimate": float(
            prior_summary["lambda_1_continuum_estimate"]
        ),
        "lambda_2_continuum_estimate": float(
            prior_summary["lambda_2_continuum_estimate"]
        ),
        "smallest_interior_window_continuum_margin_estimate": float(
            prior_summary["smallest_interior_window_continuum_margin_estimate"]
        ),
        "smallest_interior_window_last_rel_spread": float(
            prior_summary["smallest_interior_window_last_rel_spread"]
        ),
        "largest_interior_window_continuum_margin_estimate": float(
            prior_summary["largest_interior_window_continuum_margin_estimate"]
        ),
        "largest_interior_window_last_rel_spread": float(
            prior_summary["largest_interior_window_last_rel_spread"]
        ),
        "trial2_beta_sensitivity_continuum_spectral_projection_followup_lane_completed_now": (
            trial2_beta_sensitivity_continuum_spectral_projection_followup_lane_completed_now
        ),
        "trial2_beta_sensitivity_operator_level_spectral_projection_followup_promoted_now": (
            trial2_beta_sensitivity_operator_level_spectral_projection_followup_promoted_now
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
                "Continuum-support numerics are now retained, but the final "
                "operator-level theorem is still missing."
            ),
        },
    }
    written = write_artifact("declaration_gate", payload)
    print(json.dumps({"ok": True, "written": written}, ensure_ascii=False))


if __name__ == "__main__":
    main()
