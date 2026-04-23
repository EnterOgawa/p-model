#!/usr/bin/env python3
"""Generate 8.7.56.5255-.5258 selected-extension solver-recompute residual-origin refresh artifacts."""

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
        "8.7.56.5251-5254",
        "updated_pack_selected_extension_solver_recompute_retained_q_rerun_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5247-5250",
        "updated_pack_selected_extension_solver_recompute_retained_q_rerun_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5255-5258"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack selected-"
    "extension solver-recompute residual-origin refresh audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_selected_extension_solver_recompute_residual_origin_refresh_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_recompute_retained_q_rerun_audited_residual_"
    "origin_refresh_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_recompute_retained_q_rerun_preserves_phase3_"
    "failure_theorem_derived_solver_deformation_required_primary_pack_refresh_"
    "secondary_gate"
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


# 関数: residual-origin refresh で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the selected-extension solver-recompute residual-origin refresh audit."""
    return {
        "retained_replay": (
            "{F_blind^(pilot-HS,recomp), alpha_blind^(pilot-HS,recomp)}_Qret = "
            "{F_blind, alpha_blind}_Phase3"
        ),
        "helper_clearance": (
            "Concrete selected-extension recompute helper + retained replay + "
            "wrong-sign/no-improvement => residual origin is not helper-side "
            "implementation ambiguity"
        ),
        "negative_closeout_guard": (
            "negative closeout on the selected-extension solver-recompute lane is "
            "honest only after an actual solver-side deformation attempt beyond "
            "the preserved retained-q replay"
        ),
        "solver_followup": (
            "D_solver_sel^(pilot-HS) := actual deformation / recomputation of "
            "K_eff^(pilot-HS,recomp), Z_eff^(pilot-HS,recomp,T), "
            "F_blind^(pilot-HS,recomp), alpha_blind^(pilot-HS,recomp) under the "
            "fixed selected extension"
        ),
    }


# 関数: `.5255-.5258` を実行する。

def main() -> None:
    """Execute the selected-extension solver-recompute residual-origin refresh audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_selected_extension_solver_recompute_residual_origin_refresh_promoted_next"
        ]
        and prior_gate_summary[
            "gate_a_updated_pack_exact_selected_extension_solver_recompute_retained_q_rerun_available_now"
        ]
    )
    retry_mode = bool(prior_audit_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_audit_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    selected_extension_label_matches_now = bool(
        prior_audit_summary["selected_extension_label"] == "Sigma_*^(pilot-HS)"
    )
    retained_q_rerun_available_now = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_selected_extension_solver_recompute_retained_q_rerun_available_now"
        ]
        and prior_audit_summary[
            "exact_selected_extension_solver_recompute_retained_q_rerun_available_now"
        ]
    )
    retained_q_checkpoint_preserved_now = bool(
        prior_audit_summary[
            "exact_selected_extension_solver_recompute_retained_q_checkpoint_preservation_theorem_available_now"
        ]
    )
    wrong_sign_persists_now = bool(prior_audit_summary["wrong_sign_persists_now"])
    low_alpha_persists_now = bool(prior_audit_summary["low_alpha_persists_now"])

    refresh_selected = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selected_extension_label_matches_now
        and retained_q_rerun_available_now
        and retained_q_checkpoint_preserved_now
        and wrong_sign_persists_now
        and low_alpha_persists_now
    )
    exact_selected_extension_solver_recompute_rerun_preserves_phase3_failure_theorem_available_now = bool(
        refresh_selected
    )
    exact_selected_extension_solver_recompute_helper_not_residual_origin_theorem_available_now = bool(
        refresh_selected
    )
    exact_selected_extension_solver_side_deformation_lane_required_theorem_available_now = bool(
        refresh_selected
    )
    exact_selected_extension_solver_recompute_lane_negative_closeout_available_now = bool(
        refresh_selected
    )
    updated_pack_selected_extension_solver_side_deformation_followup_required = bool(
        refresh_selected
    )
    updated_pack_selected_extension_solver_recompute_negative_closeout_completed_now = bool(
        refresh_selected
    )
    updated_pack_same_schema_selected_extension_solver_recompute_residual_refresh_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = False

    blind_F_recomp_at_q_theory = float(prior_audit_summary["blind_F_recomp_at_q_theory"])
    blind_alpha_recomp_at_q_theory = float(
        prior_audit_summary["blind_alpha_recomp_at_q_theory"]
    )
    alpha_exact_at_q_theory = float(prior_audit_summary["alpha_exact_at_q_theory"])
    delta_alpha_sel_recomp_exact = float(
        prior_audit_summary["delta_alpha_sel_recomp_exact"]
    )
    relative_exact_residual_recomp = float(
        prior_audit_summary["relative_exact_residual_recomp"]
    )

    rows = [
        sign_base.row(
            "updated_pack_selected_extension_solver_recompute_residual_origin_refresh_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack selected-extension solver-recompute residual-origin refresh audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the selected-extension retained-q rerun itself is already official and the live blocker is its residual-origin interpretation.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The selected-extension recompute lane stays on computation-side blocker reduction rather than reopening selector-family recursion.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Residual-origin refresh stays honest only if exhausted surrogate and same-schema replay routes remain closed.",
        ),
        sign_base.row(
            "selected_extension_label_matches_now",
            "pass" if selected_extension_label_matches_now else "reject",
            "selected-extension label matches now",
            sign_base.truth(selected_extension_label_matches_now),
            "The residual-origin theorem is meaningful only while the recompute helper still materializes the adopted selected extension Sigma_*^(pilot-HS).",
        ),
        sign_base.row(
            "retained_q_rerun_available_now",
            "pass" if retained_q_rerun_available_now else "reject",
            "retained-q rerun available now",
            sign_base.truth(retained_q_rerun_available_now),
            "Residual-origin refresh only becomes honest after one actual retained-q rerun surface exists on the selected extension.",
        ),
        sign_base.row(
            "retained_q_checkpoint_preserved_now",
            "pass" if retained_q_checkpoint_preserved_now else "reject",
            "retained-q checkpoint preserved now",
            sign_base.truth(retained_q_checkpoint_preserved_now),
            "The selected-extension retained-q rerun still reproduces the retained Phase 3 blind checkpoint values exactly.",
        ),
        sign_base.row(
            "wrong_sign_persists_now",
            "pass" if wrong_sign_persists_now else "reject",
            "wrong sign persists now",
            sign_base.truth(wrong_sign_persists_now),
            "The selected-extension recompute surface still lives in the wrong-sign sector at q_theory.",
        ),
        sign_base.row(
            "low_alpha_persists_now",
            "pass" if low_alpha_persists_now else "reject",
            "low alpha persists now",
            sign_base.truth(low_alpha_persists_now),
            "The selected-extension recompute surface still sits below the retained exact scalar alpha at q_theory.",
        ),
        sign_base.row(
            "blind_F_recomp_at_q_theory",
            "watch",
            "selected-extension recomputed blind F(q_theory)",
            blind_F_recomp_at_q_theory,
            "The selected-extension retained-q rerun remains negative at q_theory.",
        ),
        sign_base.row(
            "blind_alpha_recomp_at_q_theory",
            "watch",
            "selected-extension recomputed blind alpha(q_theory)",
            blind_alpha_recomp_at_q_theory,
            "The selected-extension retained-q rerun remains far below the retained exact scalar target.",
        ),
        sign_base.row(
            "alpha_exact_at_q_theory",
            "pass",
            "retained exact scalar alpha(q_theory)",
            alpha_exact_at_q_theory,
            "This retained scalar target remains the comparison baseline after selector ambiguity and recompute implementation are both cleared.",
        ),
        sign_base.row(
            "delta_alpha_sel_recomp_exact",
            "watch",
            "selected-extension recomputed delta alpha vs exact scalar target",
            delta_alpha_sel_recomp_exact,
            "Negative means the selected-extension retained-q rerun still undershoots the retained exact scalar alpha at q_theory.",
        ),
        sign_base.row(
            "relative_exact_residual_recomp",
            "watch",
            "selected-extension recomputed relative residual vs exact scalar target",
            relative_exact_residual_recomp,
            "The selected-extension retained-q rerun still differs from the retained exact scalar alpha by about 91.6%.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_recompute_rerun_preserves_phase3_failure_theorem_available_now",
            "pass"
            if exact_selected_extension_solver_recompute_rerun_preserves_phase3_failure_theorem_available_now
            else "reject",
            "exact selected-extension solver-recompute rerun preserves Phase 3 failure theorem available now",
            sign_base.truth(
                exact_selected_extension_solver_recompute_rerun_preserves_phase3_failure_theorem_available_now
            ),
            "The selected-extension retained-q rerun is now theorem-side fixed as preserving the old wrong-sign / low-alpha Phase 3 failure surface.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_recompute_helper_not_residual_origin_theorem_available_now",
            "pass"
            if exact_selected_extension_solver_recompute_helper_not_residual_origin_theorem_available_now
            else "reject",
            "exact selected-extension solver-recompute helper not residual-origin theorem available now",
            sign_base.truth(
                exact_selected_extension_solver_recompute_helper_not_residual_origin_theorem_available_now
            ),
            "Because one actual retained-q rerun exists yet preserves the old failed surface, the recompute helper itself is no longer an honest residual-origin explanation.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_lane_required_theorem_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_lane_required_theorem_available_now
            else "reject",
            "exact selected-extension solver-side deformation lane required theorem available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_lane_required_theorem_available_now
            ),
            "The honest remaining blocker is now actual solver-side deformation / recomputation beyond the preserved retained-q replay.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_recompute_lane_negative_closeout_available_now",
            "pass"
            if exact_selected_extension_solver_recompute_lane_negative_closeout_available_now
            else "reject",
            "exact selected-extension solver-recompute lane negative closeout available now",
            sign_base.truth(
                exact_selected_extension_solver_recompute_lane_negative_closeout_available_now
            ),
            "The selected-extension solver-recompute lane itself now closes negatively: retained-q rerun preserves the old failed surface and does not resolve residual origin.",
        ),
        sign_base.row(
            "updated_pack_selected_extension_solver_side_deformation_followup_required",
            "pass"
            if updated_pack_selected_extension_solver_side_deformation_followup_required
            else "reject",
            "updated-pack selected-extension solver-side deformation followup required",
            sign_base.truth(
                updated_pack_selected_extension_solver_side_deformation_followup_required
            ),
            "The honest next blocker is now solver-side deformation inventory under the fixed selected extension.",
        ),
        sign_base.row(
            "updated_pack_selected_extension_solver_recompute_negative_closeout_completed_now",
            "pass"
            if updated_pack_selected_extension_solver_recompute_negative_closeout_completed_now
            else "reject",
            "updated-pack selected-extension solver-recompute negative closeout completed now",
            sign_base.truth(
                updated_pack_selected_extension_solver_recompute_negative_closeout_completed_now
            ),
            "This branch completes a final negative closeout on the selected-extension solver-recompute lane itself.",
        ),
        sign_base.row(
            "updated_pack_same_schema_selected_extension_solver_recompute_residual_refresh_replay_detected_now",
            "pass"
            if updated_pack_same_schema_selected_extension_solver_recompute_residual_refresh_replay_detected_now
            else "reject",
            "updated-pack same-schema selected-extension solver-recompute residual refresh replay detected now",
            sign_base.truth(
                updated_pack_same_schema_selected_extension_solver_recompute_residual_refresh_replay_detected_now
            ),
            "False means this turn produced a new lane closeout verdict instead of replaying the earlier retained-q availability schema.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range remains reserve-only because the next honest blocker is solver-side deformation inventory under the fixed selected extension.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_extension_label": prior_audit_summary["selected_extension_label"],
        "q_theory_over_m0": float(prior_audit_summary["q_theory_over_m0"]),
        "blind_F_recomp_at_q_theory": blind_F_recomp_at_q_theory,
        "blind_alpha_recomp_at_q_theory": blind_alpha_recomp_at_q_theory,
        "alpha_exact_at_q_theory": alpha_exact_at_q_theory,
        "delta_alpha_sel_recomp_exact": delta_alpha_sel_recomp_exact,
        "relative_exact_residual_recomp": relative_exact_residual_recomp,
        "exact_selected_extension_solver_recompute_rerun_preserves_phase3_failure_theorem_available_now": exact_selected_extension_solver_recompute_rerun_preserves_phase3_failure_theorem_available_now,
        "exact_selected_extension_solver_recompute_helper_not_residual_origin_theorem_available_now": exact_selected_extension_solver_recompute_helper_not_residual_origin_theorem_available_now,
        "exact_selected_extension_solver_side_deformation_lane_required_theorem_available_now": exact_selected_extension_solver_side_deformation_lane_required_theorem_available_now,
        "exact_selected_extension_solver_recompute_lane_negative_closeout_available_now": exact_selected_extension_solver_recompute_lane_negative_closeout_available_now,
        "updated_pack_selected_extension_solver_side_deformation_followup_required": updated_pack_selected_extension_solver_side_deformation_followup_required,
        "updated_pack_selected_extension_solver_recompute_negative_closeout_completed_now": updated_pack_selected_extension_solver_recompute_negative_closeout_completed_now,
        "updated_pack_same_schema_selected_extension_solver_recompute_residual_refresh_replay_detected_now": updated_pack_same_schema_selected_extension_solver_recompute_residual_refresh_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": updated_pack_selected_extension_solver_side_deformation_followup_required,
        "selected_primary_completion_lane": "updated_pack_selected_extension_solver_side_deformation_inventory_audit",
        "selected_secondary_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_reserve_completion_lane": "selected_extension_solver_recompute_lane_closed_negative",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_solver_recompute_residual_origin_refresh_gate",
        "recommended_next_route_or_none": "8.7.56.5259",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_solver_side_deformation_inventory_audit",
        "selected_followup_route_or_none": "8.7.56.5263",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5257",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "recompute_helper": sign_base.display_path(
                    ROOT
                    / "scripts"
                    / "quantum"
                    / "selected_extension_solver_recompute_backend.py"
                ),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5259",
                "followup_route": "8.7.56.5263",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_selected_extension_solver_recompute_residual_origin_refresh_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} selected-extension solver-recompute residual-origin refresh completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
