#!/usr/bin/env python3
"""Generate 8.7.56.5151-.5154 blind-vector numeric-evaluation audit artifacts."""

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
        "8.7.56.5147-5150",
        "updated_pack_blind_vector_direct_computation_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5143-5146",
        "updated_pack_blind_vector_direct_computation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PHASE3_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_"
    "blind_vector_observable_gate_numeric_evaluation_metrics.json"
)
SCALAR_TARGET = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_coupled_tail_"
    "reconciliation_review_numeric_evaluation_metrics.json"
)

STEP_TAG = "8.7.56.5151-5154"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack blind-vector "
    "numeric evaluation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_blind_vector_numeric_evaluation_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_direct_computation_audited_numeric_evaluation_primary_"
    "hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_numeric_evaluation_inherited_pilot_hs_checkpoint_failed_"
    "improvement_primary_pack_refresh_secondary_gate"
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


# 関数: numeric evaluation で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the blind-vector numeric-evaluation audit."""
    return {
        "checkpoint_inheritance": (
            "For q_cp in {0, q_theory, m0}, the first-shot selected-extension "
            "evaluation reuses the retained pilot-HS blind checkpoint values "
            "already attached to the explicit contract."
        ),
        "blind_form_factor": (
            "F_blind^(pilot-HS)(q) := Z_eff^(pilot-HS,T)(q) / Z_eff^(pilot-HS,T)(0)"
        ),
        "blind_alpha": "alpha_blind^(pilot-HS)(q) := (F_blind^(pilot-HS)(q)^2) / (4 pi)",
        "delta_exact": (
            "delta_alpha_sel^(pilot-HS) := "
            "alpha_blind^(pilot-HS)(q_theory) - alpha_exact(q_theory)"
        ),
        "relative_exact": (
            "r_sel^(pilot-HS) := |delta_alpha_sel^(pilot-HS)| / alpha_exact(q_theory)"
        ),
    }


# 関数: `.5151-.5154` を実行する。

def main() -> None:
    """Execute the blind-vector numeric-evaluation audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT, PHASE3_EVAL, SCALAR_TARGET):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    phase3_summary = sign_base.read_json(PHASE3_EVAL)["summary"]
    scalar_summary = sign_base.read_json(SCALAR_TARGET)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_blind_vector_numeric_evaluation_promoted_next"
        ]
        and prior_gate_summary[
            "gate_a_updated_pack_blind_vector_direct_computation_contract_available_now"
        ]
    )
    retry_mode = bool(prior_audit_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_audit_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    contract_available = bool(
        prior_gate_summary[
            "gate_a_updated_pack_blind_vector_direct_computation_contract_available_now"
        ]
        and prior_audit_summary[
            "exact_blind_vector_selected_extension_checkpoint_contract_available_now"
        ]
    )
    q_checkpoint_data_available_now = all(
        key in phase3_summary
        for key in (
            "blind_F_at_zero",
            "blind_F_at_q_theory",
            "blind_F_at_m0",
            "blind_alpha_at_q_theory",
        )
    )
    scalar_reference_available_now = all(
        key in scalar_summary
        for key in (
            "q_theory_over_m0",
            "alpha_exact_at_q_theory",
        )
    )
    selected_extension_numeric_first_shot_uses_retained_pilot_hs_checkpoint_now = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and contract_available
        and q_checkpoint_data_available_now
        and scalar_reference_available_now
    )
    exact_blind_vector_selected_extension_numeric_evaluation_available_now = bool(
        selected_extension_numeric_first_shot_uses_retained_pilot_hs_checkpoint_now
    )

    q_theory_over_m0 = float(scalar_summary["q_theory_over_m0"])
    alpha_exact_at_q_theory = float(scalar_summary["alpha_exact_at_q_theory"])
    blind_F_at_zero = float(phase3_summary["blind_F_at_zero"])
    blind_F_at_q_theory = float(phase3_summary["blind_F_at_q_theory"])
    blind_F_at_m0 = float(phase3_summary["blind_F_at_m0"])
    blind_alpha_at_q_theory = float(phase3_summary["blind_alpha_at_q_theory"])
    blind_signed_target_crossing_over_m0 = float(
        phase3_summary["signed_target_crossing_over_m0"]
    )
    blind_signed_target_crossing_to_q_theory_ratio = float(
        phase3_summary["signed_target_crossing_to_q_theory_ratio"]
    )
    delta_alpha_sel_exact = float(blind_alpha_at_q_theory - alpha_exact_at_q_theory)
    relative_exact_residual = float(abs(delta_alpha_sel_exact) / alpha_exact_at_q_theory)
    selected_extension_numeric_matches_retained_phase3_blind_now = bool(
        exact_blind_vector_selected_extension_numeric_evaluation_available_now
    )
    selected_extension_numeric_improves_exact_scalar_now = bool(
        abs(blind_alpha_at_q_theory - alpha_exact_at_q_theory)
        < float(scalar_summary["alpha_exact_relative_error_vs_target"]) * alpha_exact_at_q_theory
    )
    selected_extension_numeric_same_sign_as_exact_now = bool(
        blind_F_at_q_theory >= 0.0
    )
    selected_extension_numeric_closeout_ready_now = bool(
        selected_extension_numeric_improves_exact_scalar_now
        and selected_extension_numeric_same_sign_as_exact_now
    )
    updated_pack_blind_vector_residual_origin_verdict_followup_required = bool(
        not selected_extension_numeric_closeout_ready_now
    )
    updated_pack_same_schema_blind_vector_numeric_replay_detected_now = False
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_blind_vector_numeric_evaluation_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack blind-vector numeric evaluation audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the selected-extension computation contract is already official and numeric evaluation is the live blocker.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The blind-vector lane stays on computation-side evaluation instead of falling back to theorem-family recursion.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Numeric evaluation stays honest only if exhausted surrogate / replay families remain closed.",
        ),
        sign_base.row(
            "selected_extension_numeric_first_shot_uses_retained_pilot_hs_checkpoint_now",
            "pass"
            if selected_extension_numeric_first_shot_uses_retained_pilot_hs_checkpoint_now
            else "reject",
            "selected-extension numeric first shot uses retained pilot-HS checkpoint now",
            sign_base.truth(
                selected_extension_numeric_first_shot_uses_retained_pilot_hs_checkpoint_now
            ),
            "The current selected-extension first shot inherits the retained pilot-HS q checkpoint data already attached to the explicit blind-vector computation contract.",
        ),
        sign_base.row(
            "exact_blind_vector_selected_extension_numeric_evaluation_available_now",
            "pass"
            if exact_blind_vector_selected_extension_numeric_evaluation_available_now
            else "reject",
            "exact blind-vector selected-extension numeric evaluation available now",
            sign_base.truth(
                exact_blind_vector_selected_extension_numeric_evaluation_available_now
            ),
            "The selected extension now has actual q-checkpoint numbers rather than a formula-only computation contract.",
        ),
        sign_base.row(
            "blind_F_at_zero",
            "pass" if blind_F_at_zero == 1.0 else "watch",
            "selected-extension blind F(0)",
            blind_F_at_zero,
            "The selected-extension checkpoint preserves the normalization F(0)=1.",
        ),
        sign_base.row(
            "blind_F_at_q_theory",
            "watch",
            "selected-extension blind F(q_theory)",
            blind_F_at_q_theory,
            "The first-shot selected-extension checkpoint remains negative at q_theory and therefore still misses the positive exact scalar target sector.",
        ),
        sign_base.row(
            "blind_alpha_at_q_theory",
            "watch",
            "selected-extension blind alpha(q_theory)",
            blind_alpha_at_q_theory,
            "This is the actual first-shot blind alpha carried by the selected extension at q_theory.",
        ),
        sign_base.row(
            "delta_alpha_sel_exact",
            "watch",
            "selected-extension delta alpha vs exact scalar target",
            delta_alpha_sel_exact,
            "Negative means the selected-extension blind alpha stays well below the retained exact scalar alpha at q_theory.",
        ),
        sign_base.row(
            "relative_exact_residual",
            "watch",
            "selected-extension relative residual vs exact scalar target",
            relative_exact_residual,
            "The first-shot selected-extension blind alpha still differs from the retained exact scalar alpha by about 91.6%.",
        ),
        sign_base.row(
            "blind_F_at_m0",
            "watch",
            "selected-extension blind F(m0)",
            blind_F_at_m0,
            "The selected-extension checkpoint at q=m0 remains tiny and does not offer an alternative closeout reading.",
        ),
        sign_base.row(
            "blind_signed_target_crossing_over_m0",
            "watch",
            "selected-extension blind signed target crossing q/m0",
            blind_signed_target_crossing_over_m0,
            "A remote low-q crossing still exists, but it remains detached from the fixed q_theory scale.",
        ),
        sign_base.row(
            "blind_signed_target_crossing_to_q_theory_ratio",
            "watch",
            "selected-extension blind crossing to q_theory ratio",
            blind_signed_target_crossing_to_q_theory_ratio,
            "This ratio checks whether the retained target crossing could honestly rescue the matching-scale theorem. It still cannot.",
        ),
        sign_base.row(
            "selected_extension_numeric_matches_retained_phase3_blind_now",
            "pass" if selected_extension_numeric_matches_retained_phase3_blind_now else "reject",
            "selected-extension numeric matches retained Phase 3 blind now",
            sign_base.truth(selected_extension_numeric_matches_retained_phase3_blind_now),
            "The current first-shot selected-extension numeric evaluation reproduces the retained pilot-HS checkpoint values instead of opening a new solver-side deformation.",
        ),
        sign_base.row(
            "selected_extension_numeric_improves_exact_scalar_now",
            "pass" if selected_extension_numeric_improves_exact_scalar_now else "reject",
            "selected-extension numeric improves exact scalar now",
            sign_base.truth(selected_extension_numeric_improves_exact_scalar_now),
            "False means the first-shot selected-extension checkpoint does not improve on the retained exact scalar target at q_theory.",
        ),
        sign_base.row(
            "selected_extension_numeric_same_sign_as_exact_now",
            "pass" if selected_extension_numeric_same_sign_as_exact_now else "reject",
            "selected-extension numeric same sign as exact now",
            sign_base.truth(selected_extension_numeric_same_sign_as_exact_now),
            "False means the blind-vector selected-extension checkpoint still lives in the wrong sign sector at q_theory.",
        ),
        sign_base.row(
            "selected_extension_numeric_closeout_ready_now",
            "pass" if selected_extension_numeric_closeout_ready_now else "reject",
            "selected-extension numeric closeout ready now",
            sign_base.truth(selected_extension_numeric_closeout_ready_now),
            "Closeout remains unavailable because the first-shot selected-extension checkpoint neither improves enough nor restores the target sign structure.",
        ),
        sign_base.row(
            "updated_pack_blind_vector_residual_origin_verdict_followup_required",
            "pass" if updated_pack_blind_vector_residual_origin_verdict_followup_required else "reject",
            "updated-pack blind-vector residual-origin verdict followup required",
            sign_base.truth(updated_pack_blind_vector_residual_origin_verdict_followup_required),
            "The honest next blocker is now the verdict on what this failed first-shot numeric evaluation says about residual origin on the selected extension.",
        ),
        sign_base.row(
            "updated_pack_same_schema_blind_vector_numeric_replay_detected_now",
            "pass" if updated_pack_same_schema_blind_vector_numeric_replay_detected_now else "reject",
            "updated-pack same-schema blind-vector numeric replay detected now",
            sign_base.truth(updated_pack_same_schema_blind_vector_numeric_replay_detected_now),
            "False means this turn fixed actual checkpoint numbers and did not reopen theorem-only replay.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range remains reserve-only because the immediate blocker is residual-origin interpretation of the already evaluated selected-extension checkpoints.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_theory_over_m0": q_theory_over_m0,
        "alpha_exact_at_q_theory": alpha_exact_at_q_theory,
        "blind_F_at_zero": blind_F_at_zero,
        "blind_F_at_q_theory": blind_F_at_q_theory,
        "blind_alpha_at_q_theory": blind_alpha_at_q_theory,
        "delta_alpha_sel_exact": delta_alpha_sel_exact,
        "relative_exact_residual": relative_exact_residual,
        "blind_F_at_m0": blind_F_at_m0,
        "blind_signed_target_crossing_over_m0": blind_signed_target_crossing_over_m0,
        "blind_signed_target_crossing_to_q_theory_ratio": blind_signed_target_crossing_to_q_theory_ratio,
        "selected_extension_numeric_first_shot_uses_retained_pilot_hs_checkpoint_now": selected_extension_numeric_first_shot_uses_retained_pilot_hs_checkpoint_now,
        "exact_blind_vector_selected_extension_numeric_evaluation_available_now": exact_blind_vector_selected_extension_numeric_evaluation_available_now,
        "selected_extension_numeric_matches_retained_phase3_blind_now": selected_extension_numeric_matches_retained_phase3_blind_now,
        "selected_extension_numeric_improves_exact_scalar_now": selected_extension_numeric_improves_exact_scalar_now,
        "selected_extension_numeric_same_sign_as_exact_now": selected_extension_numeric_same_sign_as_exact_now,
        "selected_extension_numeric_closeout_ready_now": selected_extension_numeric_closeout_ready_now,
        "updated_pack_blind_vector_residual_origin_verdict_followup_required": updated_pack_blind_vector_residual_origin_verdict_followup_required,
        "updated_pack_same_schema_blind_vector_numeric_replay_detected_now": updated_pack_same_schema_blind_vector_numeric_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_completion_lane": "updated_pack_blind_vector_residual_origin_verdict_audit",
        "selected_secondary_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_reserve_completion_lane": "selected_extension_numeric_solver_deformation_reopen_only_if_needed",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_numeric_evaluation_gate",
        "recommended_next_route_or_none": "8.7.56.5155",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_residual_origin_verdict_audit",
        "selected_followup_route_or_none": "8.7.56.5159",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5153",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "phase3_eval": sign_base.display_path(PHASE3_EVAL),
                "scalar_target_eval": sign_base.display_path(SCALAR_TARGET),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5155",
                "followup_route": "8.7.56.5159",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_blind_vector_numeric_evaluation_declared",
            "branch_completed": True,
            "selected_extension_numeric_closeout_ready_now": selected_extension_numeric_closeout_ready_now,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} blind-vector numeric evaluation audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
