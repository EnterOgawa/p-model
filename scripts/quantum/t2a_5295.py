#!/usr/bin/env python3
"""Generate 8.7.56.5295-.5298 selected-extension solver-side deformation residual-origin refresh artifacts."""

from __future__ import annotations

import csv
import json
import math
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
        "8.7.56.5291-5294",
        "updated_pack_selected_extension_solver_side_deformation_numeric_rerun_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5287-5290",
        "updated_pack_selected_extension_solver_side_deformation_numeric_rerun_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_RECOMP_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5247-5250",
        "updated_pack_selected_extension_solver_recompute_retained_q_rerun_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5295-5298"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack selected-"
    "extension solver-side deformation residual-origin refresh audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_selected_extension_solver_side_deformation_residual_origin_refresh_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_side_deformation_numeric_rerun_audited_residual_"
    "origin_refresh_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_side_deformation_retained_q_replay_preserves_"
    "phase3_failure_theorem_derived_extra_q_range_required_primary_pack_"
    "refresh_secondary_gate"
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
    """Return formulas used in the selected-extension solver-side deformation residual-origin refresh audit."""
    return {
        "retained_replay": (
            "{F_blind^(pilot-HS,deform), alpha_blind^(pilot-HS,deform)}_Qret = "
            "{F_blind^(pilot-HS,recomp), alpha_blind^(pilot-HS,recomp)}_Qret = "
            "{F_blind, alpha_blind}_Phase3"
        ),
        "front_runner_clearance": (
            "Concrete selected extension + retained-q deformation replay + "
            "wrong-sign/no-improvement => residual origin is not the retained-q "
            "front-runner deformation contract itself"
        ),
        "extra_q_guard": (
            "If retained-q deformation still preserves the old failed surface, the "
            "only honest remaining solver-side deformation candidate inside "
            "Inv_solver_sel^(pilot-HS) is D_solver_sel^(Qext)"
        ),
        "reserve_followup": (
            "D_solver_sel^(Qext)[Sigma_*^(pilot-HS)] := reopen extra q-range only "
            "after retained-q deformation replay has been exhausted as a residual-"
            "origin explanation"
        ),
    }


# 関数: 浮動小数を厳密比較に近い形で照合する。

def close_now(lhs: float, rhs: float) -> bool:
    """Return whether two checkpoint values match within the fixed retained-q tolerance."""
    return math.isclose(lhs, rhs, rel_tol=0.0, abs_tol=1.0e-12)


# 関数: `.5295-.5298` を実行する。

def main() -> None:
    """Execute the selected-extension solver-side deformation residual-origin refresh audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT, PRIOR_RECOMP_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    prior_recomp_summary = sign_base.read_json(PRIOR_RECOMP_AUDIT)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_selected_extension_solver_side_deformation_residual_origin_refresh_promoted_next"
        ]
        and prior_gate_summary[
            "gate_a_updated_pack_exact_selected_extension_solver_side_deformation_numeric_rerun_available_now"
        ]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    selected_extension_label_matches_now = bool(
        prior_gate_summary["selected_extension_label"] == "Sigma_*^(pilot-HS)"
        and prior_audit_summary["selected_extension_label"] == "Sigma_*^(pilot-HS)"
        and prior_recomp_summary["selected_extension_label"] == "Sigma_*^(pilot-HS)"
    )
    solver_side_deformation_label_matches_now = bool(
        prior_gate_summary["solver_side_deformation_label"]
        == "D_solver_sel^(pilot-HS,recompute-retained)"
        and prior_audit_summary["solver_side_deformation_label"]
        == "D_solver_sel^(pilot-HS,recompute-retained)"
    )
    retained_q_numeric_rerun_available_now = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_selected_extension_solver_side_deformation_numeric_rerun_available_now"
        ]
        and prior_audit_summary[
            "exact_selected_extension_solver_side_deformation_numeric_rerun_available_now"
        ]
    )
    retained_q_checkpoint_preserved_now = bool(
        prior_audit_summary[
            "exact_selected_extension_solver_side_deformation_numeric_checkpoint_preservation_theorem_available_now"
        ]
    )
    wrong_sign_persists_now = bool(prior_audit_summary["wrong_sign_persists_now"])
    low_alpha_persists_now = bool(prior_audit_summary["low_alpha_persists_now"])

    blind_F_deform_at_zero = float(prior_audit_summary["blind_F_deform_at_zero"])
    blind_F_deform_at_q_theory = float(prior_audit_summary["blind_F_deform_at_q_theory"])
    blind_F_deform_at_m0 = float(prior_audit_summary["blind_F_deform_at_m0"])
    blind_alpha_deform_at_q_theory = float(
        prior_audit_summary["blind_alpha_deform_at_q_theory"]
    )
    delta_alpha_sel_deform_exact = float(
        prior_audit_summary["delta_alpha_sel_deform_exact"]
    )
    relative_exact_residual_deform = float(
        prior_audit_summary["relative_exact_residual_deform"]
    )
    alpha_exact_at_q_theory = float(prior_audit_summary["alpha_exact_at_q_theory"])

    blind_F_recomp_at_zero = float(prior_recomp_summary["blind_F_recomp_at_zero"])
    blind_F_recomp_at_q_theory = float(prior_recomp_summary["blind_F_recomp_at_q_theory"])
    blind_F_recomp_at_m0 = float(prior_recomp_summary["blind_F_recomp_at_m0"])
    blind_alpha_recomp_at_q_theory = float(
        prior_recomp_summary["blind_alpha_recomp_at_q_theory"]
    )
    delta_alpha_sel_recomp_exact = float(
        prior_recomp_summary["delta_alpha_sel_recomp_exact"]
    )
    relative_exact_residual_recomp = float(
        prior_recomp_summary["relative_exact_residual_recomp"]
    )

    deformation_matches_recompute_surface_now = bool(
        close_now(blind_F_deform_at_zero, blind_F_recomp_at_zero)
        and close_now(blind_F_deform_at_q_theory, blind_F_recomp_at_q_theory)
        and close_now(blind_F_deform_at_m0, blind_F_recomp_at_m0)
        and close_now(blind_alpha_deform_at_q_theory, blind_alpha_recomp_at_q_theory)
        and close_now(delta_alpha_sel_deform_exact, delta_alpha_sel_recomp_exact)
        and close_now(
            relative_exact_residual_deform,
            relative_exact_residual_recomp,
        )
    )

    refresh_selected = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selected_extension_label_matches_now
        and solver_side_deformation_label_matches_now
        and retained_q_numeric_rerun_available_now
        and retained_q_checkpoint_preserved_now
        and deformation_matches_recompute_surface_now
        and wrong_sign_persists_now
        and low_alpha_persists_now
    )
    exact_selected_extension_solver_side_deformation_rerun_preserves_phase3_failure_theorem_available_now = bool(
        refresh_selected
    )
    exact_selected_extension_solver_side_deformation_front_runner_not_residual_origin_theorem_available_now = bool(
        refresh_selected
    )
    exact_selected_extension_solver_side_extra_q_range_reserve_lane_required_theorem_available_now = bool(
        refresh_selected
    )
    exact_selected_extension_solver_side_deformation_lane_negative_closeout_available_now = (
        False
    )
    updated_pack_selected_extension_solver_side_extra_q_range_reserve_followup_required = bool(
        refresh_selected
    )
    updated_pack_selected_extension_solver_side_deformation_negative_closeout_completed_now = (
        False
    )
    updated_pack_same_schema_selected_extension_solver_side_deformation_residual_refresh_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_selected_extension_solver_side_deformation_residual_origin_refresh_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack selected-extension solver-side deformation residual-origin refresh audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after one concrete retained-q deformation rerun surface is already official and the live blocker is its residual-origin interpretation.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The selected-extension deformation lane stays on computation-side blocker reduction rather than reopening theorem-family recursion or same-tag replay.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Residual-origin refresh stays honest only if exhausted surrogate and replay routes remain closed.",
        ),
        sign_base.row(
            "selected_extension_label_matches_now",
            "pass" if selected_extension_label_matches_now else "reject",
            "selected-extension label matches now",
            sign_base.truth(selected_extension_label_matches_now),
            "The residual-origin theorem is meaningful only while the fixed selected extension Sigma_*^(pilot-HS) remains unchanged across the rerun chain.",
        ),
        sign_base.row(
            "solver_side_deformation_label_matches_now",
            "pass" if solver_side_deformation_label_matches_now else "reject",
            "solver-side deformation label matches now",
            sign_base.truth(solver_side_deformation_label_matches_now),
            "This refresh must stay pinned to the promoted retained-q front-runner D_solver_sel^(pilot-HS,recompute-retained).",
        ),
        sign_base.row(
            "retained_q_numeric_rerun_available_now",
            "pass" if retained_q_numeric_rerun_available_now else "reject",
            "retained-q numeric rerun available now",
            sign_base.truth(retained_q_numeric_rerun_available_now),
            "Residual-origin refresh only becomes honest after one actual retained-q deformation rerun surface exists on the fixed selected extension.",
        ),
        sign_base.row(
            "retained_q_checkpoint_preserved_now",
            "pass" if retained_q_checkpoint_preserved_now else "reject",
            "retained-q checkpoint preserved now",
            sign_base.truth(retained_q_checkpoint_preserved_now),
            "The selected-extension retained-q deformation rerun still reproduces the retained checkpoint values exactly.",
        ),
        sign_base.row(
            "deformation_matches_recompute_surface_now",
            "pass" if deformation_matches_recompute_surface_now else "reject",
            "deformation matches recompute surface now",
            sign_base.truth(deformation_matches_recompute_surface_now),
            "The promoted retained-q deformation front-runner does not generate a new surface; it preserves the already materialized selected-extension recompute surface pointwise on Q_ret.",
        ),
        sign_base.row(
            "wrong_sign_persists_now",
            "pass" if wrong_sign_persists_now else "reject",
            "wrong sign persists now",
            sign_base.truth(wrong_sign_persists_now),
            "The retained-q deformation front-runner still lives in the wrong-sign sector at q_theory.",
        ),
        sign_base.row(
            "low_alpha_persists_now",
            "pass" if low_alpha_persists_now else "reject",
            "low alpha persists now",
            sign_base.truth(low_alpha_persists_now),
            "The retained-q deformation front-runner still sits below the retained exact scalar alpha at q_theory.",
        ),
        sign_base.row(
            "blind_F_deform_at_q_theory",
            "watch",
            "selected-extension deformed blind F(q_theory)",
            blind_F_deform_at_q_theory,
            "The retained-q deformation front-runner remains negative at q_theory.",
        ),
        sign_base.row(
            "blind_alpha_deform_at_q_theory",
            "watch",
            "selected-extension deformed blind alpha(q_theory)",
            blind_alpha_deform_at_q_theory,
            "The retained-q deformation front-runner remains far below the exact scalar target.",
        ),
        sign_base.row(
            "alpha_exact_at_q_theory",
            "pass",
            "retained exact scalar alpha(q_theory)",
            alpha_exact_at_q_theory,
            "This is the fixed exact scalar target used throughout the selected-extension residual-origin chain.",
        ),
        sign_base.row(
            "delta_alpha_sel_deform_exact",
            "watch",
            "selected-extension deformed delta alpha vs exact scalar target",
            delta_alpha_sel_deform_exact,
            "The retained-q deformation front-runner preserves the same negative delta alpha seen on the recompute surface.",
        ),
        sign_base.row(
            "relative_exact_residual_deform",
            "watch",
            "selected-extension deformed relative exact residual",
            relative_exact_residual_deform,
            "The retained-q deformation front-runner preserves the same large exact residual.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_rerun_preserves_phase3_failure_theorem_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_rerun_preserves_phase3_failure_theorem_available_now
            else "reject",
            "exact selected-extension solver-side deformation rerun preserves Phase 3 failure theorem available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_rerun_preserves_phase3_failure_theorem_available_now
            ),
            "The retained-q deformation rerun still preserves the retained Phase 3 wrong-sign / low-alpha failure surface.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_front_runner_not_residual_origin_theorem_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_front_runner_not_residual_origin_theorem_available_now
            else "reject",
            "exact selected-extension solver-side deformation front-runner not residual origin theorem available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_front_runner_not_residual_origin_theorem_available_now
            ),
            "Because the retained-q deformation front-runner only preserves the old failed surface, it is not itself the residual-origin explanation.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_extra_q_range_reserve_lane_required_theorem_available_now",
            "pass"
            if exact_selected_extension_solver_side_extra_q_range_reserve_lane_required_theorem_available_now
            else "reject",
            "exact selected-extension solver-side extra q-range reserve lane required theorem available now",
            sign_base.truth(
                exact_selected_extension_solver_side_extra_q_range_reserve_lane_required_theorem_available_now
            ),
            "The remaining honest candidate inside the fixed solver-side deformation inventory is the extra-q-range reserve branch D_solver_sel^(Qext).",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_lane_negative_closeout_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_lane_negative_closeout_available_now
            else "reject",
            "exact selected-extension solver-side deformation lane negative closeout available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_lane_negative_closeout_available_now
            ),
            "Negative closeout on the whole solver-side deformation lane is still unavailable because the extra-q-range reserve candidate has not yet been audited.",
        ),
        sign_base.row(
            "updated_pack_selected_extension_solver_side_extra_q_range_reserve_followup_required",
            "pass"
            if updated_pack_selected_extension_solver_side_extra_q_range_reserve_followup_required
            else "reject",
            "updated-pack selected-extension solver-side extra q-range reserve followup required",
            sign_base.truth(
                updated_pack_selected_extension_solver_side_extra_q_range_reserve_followup_required
            ),
            "The honest next blocker is the selected-extension solver-side extra-q-range reserve lane, not another retained-q deformation replay.",
        ),
        sign_base.row(
            "updated_pack_same_schema_selected_extension_solver_side_deformation_residual_refresh_replay_detected_now",
            "pass"
            if updated_pack_same_schema_selected_extension_solver_side_deformation_residual_refresh_replay_detected_now
            else "reject",
            "updated-pack same-schema selected-extension solver-side deformation residual refresh replay detected now",
            sign_base.truth(
                updated_pack_same_schema_selected_extension_solver_side_deformation_residual_refresh_replay_detected_now
            ),
            "False means this branch genuinely compressed the blocker from retained-q deformation replay to the extra-q-range reserve candidate.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Farther hybrid continuation stays reserve-only because the next honest step is still an internal selected-extension extra-q-range branch.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_extension_label": "Sigma_*^(pilot-HS)",
        "solver_side_deformation_label": "D_solver_sel^(pilot-HS,recompute-retained)",
        "q_theory_over_m0": float(prior_audit_summary["q_theory_over_m0"]),
        "blind_F_deform_at_zero": blind_F_deform_at_zero,
        "blind_F_deform_at_q_theory": blind_F_deform_at_q_theory,
        "blind_F_deform_at_m0": blind_F_deform_at_m0,
        "blind_alpha_deform_at_q_theory": blind_alpha_deform_at_q_theory,
        "alpha_exact_at_q_theory": alpha_exact_at_q_theory,
        "delta_alpha_sel_deform_exact": delta_alpha_sel_deform_exact,
        "relative_exact_residual_deform": relative_exact_residual_deform,
        "deformation_matches_recompute_surface_now": deformation_matches_recompute_surface_now,
        "wrong_sign_persists_now": wrong_sign_persists_now,
        "low_alpha_persists_now": low_alpha_persists_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_selected_extension_solver_side_deformation_rerun_preserves_phase3_failure_theorem_available_now": exact_selected_extension_solver_side_deformation_rerun_preserves_phase3_failure_theorem_available_now,
        "exact_selected_extension_solver_side_deformation_front_runner_not_residual_origin_theorem_available_now": exact_selected_extension_solver_side_deformation_front_runner_not_residual_origin_theorem_available_now,
        "exact_selected_extension_solver_side_extra_q_range_reserve_lane_required_theorem_available_now": exact_selected_extension_solver_side_extra_q_range_reserve_lane_required_theorem_available_now,
        "exact_selected_extension_solver_side_deformation_lane_negative_closeout_available_now": exact_selected_extension_solver_side_deformation_lane_negative_closeout_available_now,
        "updated_pack_selected_extension_solver_side_extra_q_range_reserve_followup_required": updated_pack_selected_extension_solver_side_extra_q_range_reserve_followup_required,
        "updated_pack_selected_extension_solver_side_deformation_negative_closeout_completed_now": updated_pack_selected_extension_solver_side_deformation_negative_closeout_completed_now,
        "updated_pack_same_schema_selected_extension_solver_side_deformation_residual_refresh_replay_detected_now": updated_pack_same_schema_selected_extension_solver_side_deformation_residual_refresh_replay_detected_now,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_completion_lane": "updated_pack_selected_extension_solver_side_extra_q_range_reserve_audit",
        "selected_secondary_completion_lane": "updated_pack_selected_extension_solver_side_extra_q_range_reserve_gate",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only_after_selected_extension_solver_side_extra_q_check",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_solver_side_extra_q_range_reserve_audit",
        "recommended_next_route_or_none": "8.7.56.5303",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_solver_side_extra_q_range_reserve_gate",
        "selected_followup_route_or_none": "8.7.56.5307",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5297",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "prior_recompute_audit": sign_base.display_path(PRIOR_RECOMP_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5303",
                "followup_route": "8.7.56.5307",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_selected_extension_solver_side_deformation_residual_origin_refresh_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} selected-extension solver-side deformation residual-origin refresh completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
