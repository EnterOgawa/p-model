#!/usr/bin/env python3
"""Generate 8.7.56.5159-.5162 blind-vector residual-origin verdict artifacts."""

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
        "8.7.56.5155-5158",
        "updated_pack_blind_vector_numeric_evaluation_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5151-5154",
        "updated_pack_blind_vector_numeric_evaluation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5159-5162"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack blind-vector "
    "residual-origin verdict audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_blind_vector_residual_origin_verdict_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_numeric_evaluation_audited_residual_origin_verdict_primary_"
    "hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_residual_origin_selector_ambiguity_cleared_solver_"
    "deformation_required_primary_pack_refresh_secondary_gate"
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


# 関数: residual-origin verdict で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the blind-vector residual-origin verdict audit."""
    return {
        "retained_replay": (
            "{F_blind^(pilot-HS), alpha_blind^(pilot-HS)}_first_shot = "
            "{F_blind, alpha_blind}_retained_phase3"
        ),
        "selector_clearance": (
            "Concrete selected extension + retained replay + wrong-sign/no-improvement "
            "=> residual origin is not selector ambiguity"
        ),
        "negative_closeout_guard": (
            "negative closeout on the selected extension is honest only after an "
            "actual solver-side deformation attempt beyond retained replay"
        ),
        "solver_followup": (
            "D_solver^(pilot-HS) := actual recomputation/deformation of "
            "K_eff^(pilot-HS), Z_eff^(pilot-HS,T), F_blind^(pilot-HS), "
            "alpha_blind^(pilot-HS) under the fixed selected extension"
        ),
    }


# 関数: `.5159-.5162` を実行する。

def main() -> None:
    """Execute the blind-vector residual-origin verdict audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_blind_vector_residual_origin_verdict_promoted_next"
        ]
        and prior_gate_summary[
            "gate_a_updated_pack_blind_vector_numeric_evaluation_available_now"
        ]
    )
    retry_mode = bool(prior_audit_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_audit_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    selected_extension_available = bool(
        prior_gate_summary["exact_concrete_selected_extension_available_now"]
    )
    direct_computation_admissible = bool(
        prior_gate_summary["direct_blind_vector_computation_primary_admissible_now"]
    )
    first_shot_matches_retained_phase3 = bool(
        prior_audit_summary["selected_extension_numeric_matches_retained_phase3_blind_now"]
    )
    first_shot_failed_improvement = bool(
        not prior_audit_summary["selected_extension_numeric_improves_exact_scalar_now"]
    )
    first_shot_wrong_sign = bool(
        not prior_audit_summary["selected_extension_numeric_same_sign_as_exact_now"]
    )
    first_shot_closeout_ready = bool(
        prior_audit_summary["selected_extension_numeric_closeout_ready_now"]
    )

    selector_ambiguity_cleared = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selected_extension_available
        and direct_computation_admissible
        and first_shot_matches_retained_phase3
        and first_shot_failed_improvement
        and first_shot_wrong_sign
        and not first_shot_closeout_ready
    )
    exact_blind_vector_selected_extension_retained_replay_theorem_available_now = bool(
        selector_ambiguity_cleared
    )
    exact_blind_vector_residual_origin_not_selector_choice_theorem_available_now = bool(
        selector_ambiguity_cleared
    )
    exact_blind_vector_selected_extension_wrong_sign_no_improvement_theorem_available_now = bool(
        selector_ambiguity_cleared
    )
    exact_blind_vector_selected_extension_negative_closeout_available_now = False
    updated_pack_blind_vector_solver_side_deformation_followup_required = bool(
        selector_ambiguity_cleared
    )
    updated_pack_blind_vector_residual_origin_negative_closeout_completed_now = False
    updated_pack_same_schema_blind_vector_residual_origin_verdict_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = False

    blind_F_at_q_theory = float(prior_audit_summary["blind_F_at_q_theory"])
    blind_alpha_at_q_theory = float(prior_audit_summary["blind_alpha_at_q_theory"])
    alpha_exact_at_q_theory = float(prior_audit_summary["alpha_exact_at_q_theory"])
    delta_alpha_sel_exact = float(prior_audit_summary["delta_alpha_sel_exact"])
    relative_exact_residual = float(prior_audit_summary["relative_exact_residual"])

    rows = [
        sign_base.row(
            "updated_pack_blind_vector_residual_origin_verdict_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack blind-vector residual-origin verdict audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after selected-extension first-shot numeric checkpoints are already fixed and the live blocker is their interpretation.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The verdict stays on the computation lane rather than reopening theorem-family recursion.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Residual-origin interpretation stays honest only if exhausted surrogate and same-schema replay routes remain closed.",
        ),
        sign_base.row(
            "exact_concrete_selected_extension_available_now",
            "pass" if selected_extension_available else "reject",
            "exact concrete selected extension available now",
            sign_base.truth(selected_extension_available),
            "This verdict is meaningful only because one concrete selected extension Sigma_*^(pilot-HS) is already official.",
        ),
        sign_base.row(
            "selected_extension_numeric_matches_retained_phase3_blind_now",
            "pass" if first_shot_matches_retained_phase3 else "reject",
            "selected-extension numeric matches retained Phase 3 blind now",
            sign_base.truth(first_shot_matches_retained_phase3),
            "The first-shot selected-extension evaluation reproduces the retained blind checkpoint instead of generating a new numeric sector.",
        ),
        sign_base.row(
            "selected_extension_numeric_failed_improvement_now",
            "pass" if first_shot_failed_improvement else "reject",
            "selected-extension numeric failed improvement now",
            sign_base.truth(first_shot_failed_improvement),
            "The selected-extension first shot still fails to improve on the retained exact scalar target at q_theory.",
        ),
        sign_base.row(
            "selected_extension_numeric_wrong_sign_now",
            "pass" if first_shot_wrong_sign else "reject",
            "selected-extension numeric wrong sign now",
            sign_base.truth(first_shot_wrong_sign),
            "The selected-extension first shot remains in the wrong-sign sector at q_theory.",
        ),
        sign_base.row(
            "blind_F_at_q_theory",
            "watch",
            "selected-extension blind F(q_theory)",
            blind_F_at_q_theory,
            "The first-shot selected-extension checkpoint remains negative at q_theory.",
        ),
        sign_base.row(
            "blind_alpha_at_q_theory",
            "watch",
            "selected-extension blind alpha(q_theory)",
            blind_alpha_at_q_theory,
            "The first-shot selected-extension alpha remains far below the retained exact scalar target.",
        ),
        sign_base.row(
            "alpha_exact_at_q_theory",
            "pass",
            "retained exact scalar alpha(q_theory)",
            alpha_exact_at_q_theory,
            "This is the retained scalar comparison target already fixed before the selected-extension lane reopened blind-vector computation.",
        ),
        sign_base.row(
            "delta_alpha_sel_exact",
            "watch",
            "selected-extension delta alpha vs exact scalar target",
            delta_alpha_sel_exact,
            "The selected-extension first shot stays far below the retained exact scalar alpha at q_theory.",
        ),
        sign_base.row(
            "relative_exact_residual",
            "watch",
            "selected-extension relative residual vs exact scalar target",
            relative_exact_residual,
            "The first-shot selected-extension mismatch remains about 91.6%.",
        ),
        sign_base.row(
            "exact_blind_vector_selected_extension_retained_replay_theorem_available_now",
            "pass"
            if exact_blind_vector_selected_extension_retained_replay_theorem_available_now
            else "reject",
            "exact blind-vector selected-extension retained-replay theorem available now",
            sign_base.truth(
                exact_blind_vector_selected_extension_retained_replay_theorem_available_now
            ),
            "The selected-extension first shot is now theorem-side fixed as a retained replay of the old blind checkpoint rather than a new numeric branch.",
        ),
        sign_base.row(
            "exact_blind_vector_residual_origin_not_selector_choice_theorem_available_now",
            "pass"
            if exact_blind_vector_residual_origin_not_selector_choice_theorem_available_now
            else "reject",
            "exact blind-vector residual-origin not selector-choice theorem available now",
            sign_base.truth(
                exact_blind_vector_residual_origin_not_selector_choice_theorem_available_now
            ),
            "Because one concrete selected extension exists yet the first shot merely reproduces the old blind mismatch, selector ambiguity is no longer the live residual-origin explanation.",
        ),
        sign_base.row(
            "exact_blind_vector_selected_extension_wrong_sign_no_improvement_theorem_available_now",
            "pass"
            if exact_blind_vector_selected_extension_wrong_sign_no_improvement_theorem_available_now
            else "reject",
            "exact blind-vector selected-extension wrong-sign/no-improvement theorem available now",
            sign_base.truth(
                exact_blind_vector_selected_extension_wrong_sign_no_improvement_theorem_available_now
            ),
            "The fixed selected extension still lands in the wrong-sign / low-alpha sector on its inherited first shot.",
        ),
        sign_base.row(
            "exact_blind_vector_selected_extension_negative_closeout_available_now",
            "pass"
            if exact_blind_vector_selected_extension_negative_closeout_available_now
            else "reject",
            "exact blind-vector selected-extension negative closeout available now",
            sign_base.truth(
                exact_blind_vector_selected_extension_negative_closeout_available_now
            ),
            "Negative closeout on the selected extension itself is not yet honest because the current verdict still rests on inherited replay rather than an actual solver-side deformation attempt.",
        ),
        sign_base.row(
            "updated_pack_blind_vector_solver_side_deformation_followup_required",
            "pass"
            if updated_pack_blind_vector_solver_side_deformation_followup_required
            else "reject",
            "updated-pack blind-vector solver-side deformation followup required",
            sign_base.truth(
                updated_pack_blind_vector_solver_side_deformation_followup_required
            ),
            "The honest next blocker is now a solver-side deformation/recomputation lane under the fixed selected extension.",
        ),
        sign_base.row(
            "updated_pack_blind_vector_residual_origin_negative_closeout_completed_now",
            "pass"
            if updated_pack_blind_vector_residual_origin_negative_closeout_completed_now
            else "reject",
            "updated-pack blind-vector residual-origin negative closeout completed now",
            sign_base.truth(
                updated_pack_blind_vector_residual_origin_negative_closeout_completed_now
            ),
            "This branch clears selector ambiguity but does not yet complete a final negative closeout on the selected extension itself.",
        ),
        sign_base.row(
            "updated_pack_same_schema_blind_vector_residual_origin_verdict_replay_detected_now",
            "pass"
            if updated_pack_same_schema_blind_vector_residual_origin_verdict_replay_detected_now
            else "reject",
            "updated-pack same-schema blind-vector residual-origin verdict replay detected now",
            sign_base.truth(
                updated_pack_same_schema_blind_vector_residual_origin_verdict_replay_detected_now
            ),
            "False means this turn cut a new verdict about residual origin rather than replaying the same theorem schema.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range remains reserve-only because the current blocker has moved to solver-side deformation under the fixed selected extension.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_theory_over_m0": float(prior_audit_summary["q_theory_over_m0"]),
        "blind_F_at_q_theory": blind_F_at_q_theory,
        "blind_alpha_at_q_theory": blind_alpha_at_q_theory,
        "alpha_exact_at_q_theory": alpha_exact_at_q_theory,
        "delta_alpha_sel_exact": delta_alpha_sel_exact,
        "relative_exact_residual": relative_exact_residual,
        "selected_extension_numeric_matches_retained_phase3_blind_now": first_shot_matches_retained_phase3,
        "selected_extension_numeric_failed_improvement_now": first_shot_failed_improvement,
        "selected_extension_numeric_wrong_sign_now": first_shot_wrong_sign,
        "exact_blind_vector_selected_extension_retained_replay_theorem_available_now": exact_blind_vector_selected_extension_retained_replay_theorem_available_now,
        "exact_blind_vector_residual_origin_not_selector_choice_theorem_available_now": exact_blind_vector_residual_origin_not_selector_choice_theorem_available_now,
        "exact_blind_vector_selected_extension_wrong_sign_no_improvement_theorem_available_now": exact_blind_vector_selected_extension_wrong_sign_no_improvement_theorem_available_now,
        "exact_blind_vector_selected_extension_negative_closeout_available_now": exact_blind_vector_selected_extension_negative_closeout_available_now,
        "updated_pack_blind_vector_solver_side_deformation_followup_required": updated_pack_blind_vector_solver_side_deformation_followup_required,
        "updated_pack_blind_vector_residual_origin_negative_closeout_completed_now": updated_pack_blind_vector_residual_origin_negative_closeout_completed_now,
        "updated_pack_same_schema_blind_vector_residual_origin_verdict_replay_detected_now": updated_pack_same_schema_blind_vector_residual_origin_verdict_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_completion_lane": "updated_pack_blind_vector_solver_side_deformation_inventory_audit",
        "selected_secondary_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_reserve_completion_lane": "selected_extension_negative_closeout_only_after_solver_deformation_check",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_residual_origin_verdict_gate",
        "recommended_next_route_or_none": "8.7.56.5163",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_solver_side_deformation_inventory_audit",
        "selected_followup_route_or_none": "8.7.56.5167",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5161",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5163",
                "followup_route": "8.7.56.5167",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_blind_vector_residual_origin_verdict_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} blind-vector residual-origin verdict audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
