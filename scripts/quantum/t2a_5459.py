#!/usr/bin/env python3
"""Generate 8.7.56.5459-.5462 source-materialization gate / verdict refresh artifacts."""

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
        "8.7.56.5455-5458",
        "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5459-5462"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack selected-"
    "extension independent extra-q-range source-materialization gate / "
    "verdict refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_selected_extension_independent_extra_q_range_source_materialization_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_independent_extra_q_range_source_materialization_"
    "numeric_rerun_legacy_phase3_sideband_carryover_derived_negative_verdict_"
    "gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_independent_extra_q_range_source_materialization_"
    "negative_closeout_completed_trial2_numerical_closeout_inventory_"
    "primary_next"
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


# 関数: gate / verdict refresh で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the source-materialization gate / verdict refresh."""
    return {
        "gate_a": (
            "Gate A = selected-extension independent extra-q-range source-"
            "materialization numeric rerun negative verdict available now"
        ),
        "gate_b": (
            "Gate B = Trial-2 numerical closeout inventory promoted next"
        ),
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.5459-.5462` を実行する。

def main() -> None:
    """Execute the source-materialization gate / verdict refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_formula_available_now"
        ]
        and prior_summary[
            "exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_materialized_surface_available_now"
        ]
        and prior_summary[
            "exact_selected_extension_independent_extra_q_range_source_materialization_q_theory_failure_preserved_theorem_available_now"
        ]
        and prior_summary[
            "exact_selected_extension_independent_extra_q_range_source_materialization_legacy_phase3_sideband_carryover_theorem_available_now"
        ]
        and prior_summary[
            "exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_negative_verdict_available_now"
        ]
    )
    gate_b = bool(
        prior_summary[
            "updated_pack_selected_extension_independent_extra_q_range_source_materialization_negative_closeout_followup_required"
        ]
    )
    gate_c = bool(prior_summary["farther_hybrid_continuation_reopen_required_now"])
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    same_schema_replay_detected = bool(
        prior_summary[
            "updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_replay_detected_now"
        ]
    )
    pack_update_required_now = bool(gate_b)

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_selected_extension_independent_extra_q_range_source_materialization_negative_closeout_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack exact selected-extension independent extra-q-range source-materialization negative closeout available now",
            sign_base.truth(gate_a),
            "The helper-backed extra-q rerun now has an honest verdict: it preserves the q_theory failure and only materializes legacy Phase-3 sidebands.",
        ),
        sign_base.row(
            "gate_b_updated_pack_trial2_numerical_closeout_inventory_promoted_next",
            "pass" if gate_b else "reject",
            "gate B updated-pack Trial-2 numerical closeout inventory promoted next",
            sign_base.truth(gate_b),
            "Because source-materialization closes negatively, the honest next blocker is no longer extra-q route replay but Trial-2 numerical closeout inventory.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "pass" if gate_c else "reject",
            "gate C farther hybrid continuation reopen required now",
            sign_base.truth(gate_c),
            "Farther hybrid continuation stays reserve-only because the current extra-q surface did not materialize a new canonical rescue.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The route refresh follows actual helper-backed computation and an explicit verdict, not theorem-family replay.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Closing the source-materialization lane does not reopen exhausted selector, recompute, or deformation branches.",
        ),
        sign_base.row(
            "updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_replay_detected_now",
            "pass" if same_schema_replay_detected else "reject",
            "updated-pack same-schema selected-extension independent extra-q-range source-materialization numeric rerun replay detected now",
            sign_base.truth(same_schema_replay_detected),
            "False means the blocker has genuinely moved from missing extra-q verdict to a new closeout interpretation lane.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "A substantive lane shift happened here: selected-extension source-materialization closes negatively and Trial-2 numerical closeout becomes next.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_extension_label": prior_summary["selected_extension_label"],
        "solver_side_deformation_label": prior_summary["solver_side_deformation_label"],
        "source_materialization_label": prior_summary["source_materialization_label"],
        "q_theory_over_m0": float(prior_summary["q_theory_over_m0"]),
        "scalar_q_exact_over_m0": float(prior_summary["scalar_q_exact_over_m0"]),
        "alpha_exact_at_q_theory": float(prior_summary["alpha_exact_at_q_theory"]),
        "alpha_target": float(prior_summary["alpha_target"]),
        "best_extra_label_vs_alpha_exact": prior_summary["best_extra_label_vs_alpha_exact"],
        "best_extra_label_vs_alpha_target": prior_summary["best_extra_label_vs_alpha_target"],
        "best_extra_label_vs_q_exact": prior_summary["best_extra_label_vs_q_exact"],
        "best_extra_alpha_exact_residual": float(
            prior_summary["best_extra_alpha_exact_residual"]
        ),
        "best_extra_alpha_target_residual": float(
            prior_summary["best_extra_alpha_target_residual"]
        ),
        "best_extra_q_exact_gap": float(prior_summary["best_extra_q_exact_gap"]),
        "best_extra_label_diagnostic": prior_summary["best_extra_label_diagnostic"],
        "q_theory_diagnostic": prior_summary["q_theory_diagnostic"],
        "gate_a_updated_pack_exact_selected_extension_independent_extra_q_range_source_materialization_negative_closeout_available_now": gate_a,
        "gate_b_updated_pack_trial2_numerical_closeout_inventory_promoted_next": gate_b,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_replay_detected_now": same_schema_replay_detected,
        "pack_update_required_now": pack_update_required_now,
        "selected_primary_completion_lane": "trial2_numeric_alpha_numerical_closeout_inventory_audit",
        "selected_secondary_completion_lane": "trial2_numeric_alpha_numerical_closeout_gate",
        "selected_reserve_completion_lane": "farther_hybrid_reserve_only_until_new_independent_source_exists",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_trial2_numerical_closeout_inventory_audit",
        "recommended_next_route_or_none": "8.7.56.5463",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_trial2_numerical_closeout_gate",
        "selected_followup_route_or_none": "8.7.56.5467",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5461",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5463",
                "followup_route": "8.7.56.5467",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_selected_extension_source_materialization_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} selected-extension source-materialization gate completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から gate を実行する。

if __name__ == "__main__":
    main()
