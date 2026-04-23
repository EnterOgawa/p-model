#!/usr/bin/env python3
"""Generate 8.7.56.5155-.5158 blind-vector numeric-evaluation gate artifacts."""

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
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5151-5154",
        "updated_pack_blind_vector_numeric_evaluation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5155-5158"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack blind-vector "
    "numeric evaluation gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_blind_vector_numeric_evaluation_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_numeric_evaluation_inherited_pilot_hs_checkpoint_failed_"
    "improvement_primary_pack_refresh_secondary_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_numeric_evaluation_audited_residual_origin_verdict_primary_"
    "hybrid_reserve_secondary_next"
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
    """Return formulas used in the blind-vector numeric-evaluation gate."""
    return {
        "gate_a": "Gate A = selected-extension blind-vector numeric checkpoints fixed now",
        "gate_b": "Gate B = blind-vector residual-origin verdict promoted next",
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.5155-.5158` を実行する。

def main() -> None:
    """Execute the blind-vector numeric-evaluation decision gate."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    gate_a_updated_pack_blind_vector_numeric_evaluation_available_now = bool(
        prior_summary["exact_blind_vector_selected_extension_numeric_evaluation_available_now"]
        and prior_summary[
            "selected_extension_numeric_first_shot_uses_retained_pilot_hs_checkpoint_now"
        ]
    )
    gate_b_updated_pack_blind_vector_residual_origin_verdict_promoted_next = bool(
        prior_summary["updated_pack_blind_vector_residual_origin_verdict_followup_required"]
        and not prior_summary["selected_extension_numeric_closeout_ready_now"]
    )
    gate_c_farther_hybrid_continuation_reopen_required_now = bool(
        prior_summary["farther_hybrid_continuation_reopen_required_now"]
    )
    exact_concrete_selected_extension_available_now = True
    direct_blind_vector_computation_primary_admissible_now = True

    rows = [
        sign_base.row(
            "gate_a_updated_pack_blind_vector_numeric_evaluation_available_now",
            "pass" if gate_a_updated_pack_blind_vector_numeric_evaluation_available_now else "reject",
            "Gate A updated-pack blind-vector numeric evaluation available now",
            sign_base.truth(gate_a_updated_pack_blind_vector_numeric_evaluation_available_now),
            "The selected extension now has actual q-checkpoint numbers attached to the blind-vector computation contract.",
        ),
        sign_base.row(
            "gate_b_updated_pack_blind_vector_residual_origin_verdict_promoted_next",
            "pass" if gate_b_updated_pack_blind_vector_residual_origin_verdict_promoted_next else "reject",
            "Gate B updated-pack blind-vector residual-origin verdict promoted next",
            sign_base.truth(gate_b_updated_pack_blind_vector_residual_origin_verdict_promoted_next),
            "Because the first-shot selected-extension numeric evaluation still misses the exact scalar target, the honest next blocker is the residual-origin verdict rather than another theorem-family restatement.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "pass" if gate_c_farther_hybrid_continuation_reopen_required_now else "reject",
            "Gate C farther hybrid continuation reopen required now",
            sign_base.truth(gate_c_farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains reserve-only because the next live task is residual-origin interpretation of the already computed selected-extension checkpoints.",
        ),
        sign_base.row(
            "exact_concrete_selected_extension_available_now",
            "pass" if exact_concrete_selected_extension_available_now else "reject",
            "exact concrete selected extension available now",
            sign_base.truth(exact_concrete_selected_extension_available_now),
            "The numeric-evaluation gate inherits the already closed selected extension Sigma_*^(pilot-HS) and does not reopen selector ambiguity.",
        ),
        sign_base.row(
            "direct_blind_vector_computation_primary_admissible_now",
            "pass" if direct_blind_vector_computation_primary_admissible_now else "reject",
            "direct blind-vector computation primary admissible now",
            sign_base.truth(direct_blind_vector_computation_primary_admissible_now),
            "The computation lane remains open; the issue is now verdict interpretation rather than admissibility.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_theory_over_m0": float(prior_summary["q_theory_over_m0"]),
        "alpha_exact_at_q_theory": float(prior_summary["alpha_exact_at_q_theory"]),
        "blind_alpha_at_q_theory": float(prior_summary["blind_alpha_at_q_theory"]),
        "delta_alpha_sel_exact": float(prior_summary["delta_alpha_sel_exact"]),
        "relative_exact_residual": float(prior_summary["relative_exact_residual"]),
        "gate_a_updated_pack_blind_vector_numeric_evaluation_available_now": gate_a_updated_pack_blind_vector_numeric_evaluation_available_now,
        "gate_b_updated_pack_blind_vector_residual_origin_verdict_promoted_next": gate_b_updated_pack_blind_vector_residual_origin_verdict_promoted_next,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c_farther_hybrid_continuation_reopen_required_now,
        "exact_concrete_selected_extension_available_now": exact_concrete_selected_extension_available_now,
        "direct_blind_vector_computation_primary_admissible_now": direct_blind_vector_computation_primary_admissible_now,
        "selected_primary_completion_lane": "updated_pack_blind_vector_residual_origin_verdict_audit",
        "selected_secondary_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_reserve_completion_lane": "selected_extension_numeric_solver_deformation_reopen_only_if_needed",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_residual_origin_verdict_audit",
        "recommended_next_route_or_none": "8.7.56.5159",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_residual_origin_verdict_gate",
        "selected_followup_route_or_none": "8.7.56.5163",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5157",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_audit": sign_base.display_path(PRIOR_GATE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5159",
                "followup_route": "8.7.56.5163",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_blind_vector_numeric_evaluation_gate_declared",
            "branch_completed": True,
            "residual_origin_verdict_promoted_next": gate_b_updated_pack_blind_vector_residual_origin_verdict_promoted_next,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} blind-vector numeric evaluation gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
