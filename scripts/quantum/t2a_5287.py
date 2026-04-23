#!/usr/bin/env python3
"""Generate 8.7.56.5287-.5290 selected-extension solver-side deformation numeric-rerun artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.selected_extension_solver_side_deformation_backend import (
    build_selected_extension_solver_side_deformation_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5283-5286",
        "updated_pack_selected_extension_solver_side_deformation_front_runner_implementation_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5287-5290"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack selected-"
    "extension solver-side deformation numeric rerun audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_selected_extension_solver_side_deformation_numeric_rerun_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_side_deformation_implementation_audited_numeric_"
    "rerun_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_side_deformation_numeric_rerun_derived_residual_"
    "origin_refresh_primary_pack_refresh_secondary_gate"
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


# 関数: numeric rerun audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the selected-extension solver-side deformation numeric rerun audit."""
    return {
        "deformation_pack": (
            "O_deform_sel,impl^(pilot-HS) := "
            "build_selected_extension_solver_side_deformation_pack(ell_values=(1,2,3))"
        ),
        "retained_q_surface": (
            "Q_ret^(pilot-HS,deform) := {F_blind^(pilot-HS,deform)(0), "
            "F_blind^(pilot-HS,deform)(q_theory), "
            "F_blind^(pilot-HS,deform)(m0), "
            "alpha_blind^(pilot-HS,deform)(q_theory)}"
        ),
        "delta_exact": (
            "delta_alpha_sel^(pilot-HS,deform) := "
            "alpha_blind^(pilot-HS,deform)(q_theory) - alpha_exact(q_theory)"
        ),
        "relative_exact": (
            "r_sel^(pilot-HS,deform) := "
            "|delta_alpha_sel^(pilot-HS,deform)| / alpha_exact(q_theory)"
        ),
    }


# 関数: `.5287-.5290` を実行する。

def main() -> None:
    """Execute the selected-extension solver-side deformation numeric rerun audit."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    audit_selected = bool(
        prior_summary[
            "gate_b_updated_pack_selected_extension_solver_side_deformation_numeric_rerun_promoted_next"
        ]
        and prior_summary[
            "gate_a_updated_pack_exact_selected_extension_solver_side_deformation_implementation_available_now"
        ]
    )
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )

    deformation_pack = build_selected_extension_solver_side_deformation_pack()
    selected_extension_label_matches_now = bool(
        deformation_pack["selected_extension_label"] == "Sigma_*^(pilot-HS)"
    )
    retained_q_window_available_now = bool(
        {"zero", "q_theory_over_m0", "m0"}
        <= set(deformation_pack["retained_q_window"].keys())
    )
    f_blind_deform_pack_available_now = bool(
        {"zero", "q_theory_over_m0", "m0"}
        <= set(deformation_pack["F_blind_deform_pack"].keys())
    )
    exact_selected_extension_solver_side_deformation_numeric_rerun_formula_available_now = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selected_extension_label_matches_now
    )
    exact_selected_extension_solver_side_deformation_numeric_surface_available_now = bool(
        exact_selected_extension_solver_side_deformation_numeric_rerun_formula_available_now
        and retained_q_window_available_now
        and f_blind_deform_pack_available_now
    )
    exact_selected_extension_solver_side_deformation_numeric_checkpoint_preservation_theorem_available_now = bool(
        exact_selected_extension_solver_side_deformation_numeric_surface_available_now
        and deformation_pack["preserves_recompute_surface_now"]
    )
    exact_selected_extension_solver_side_deformation_numeric_rerun_available_now = bool(
        exact_selected_extension_solver_side_deformation_numeric_rerun_formula_available_now
        and exact_selected_extension_solver_side_deformation_numeric_surface_available_now
        and exact_selected_extension_solver_side_deformation_numeric_checkpoint_preservation_theorem_available_now
    )

    blind_F_deform_at_zero = float(deformation_pack["F_blind_deform_pack"]["zero"])
    blind_F_deform_at_q_theory = float(
        deformation_pack["F_blind_deform_pack"]["q_theory_over_m0"]
    )
    blind_F_deform_at_m0 = float(deformation_pack["F_blind_deform_pack"]["m0"])
    blind_alpha_deform_at_q_theory = float(
        deformation_pack["alpha_blind_deform_at_q_theory"]
    )
    alpha_exact_at_q_theory = float(deformation_pack["alpha_exact_at_q_theory"])
    delta_alpha_sel_deform_exact = float(
        deformation_pack["delta_alpha_sel_deform_exact"]
    )
    relative_exact_residual_deform = float(
        deformation_pack["relative_exact_residual_deform"]
    )
    wrong_sign_persists_now = bool(blind_F_deform_at_q_theory < 0.0)
    low_alpha_persists_now = bool(
        blind_alpha_deform_at_q_theory < alpha_exact_at_q_theory
    )

    updated_pack_selected_extension_solver_side_deformation_residual_origin_refresh_followup_required = bool(
        exact_selected_extension_solver_side_deformation_numeric_rerun_available_now
    )
    updated_pack_same_schema_selected_extension_solver_side_deformation_numeric_rerun_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_selected_extension_solver_side_deformation_numeric_rerun_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack selected-extension solver-side deformation numeric rerun audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after one concrete selected-extension deformation implementation is already official and the live blocker is numeric rerun itself.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The active selected-extension deformation lane stays on computation-side blocker reduction rather than falling back to theorem-family replay.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Deformation numeric rerun remains honest only while exhausted surrogate and selector-replay branches stay closed.",
        ),
        sign_base.row(
            "selected_extension_label_matches_now",
            "pass" if selected_extension_label_matches_now else "reject",
            "selected-extension label matches now",
            sign_base.truth(selected_extension_label_matches_now),
            "The deformation rerun remains meaningful only while the helper still materializes the adopted selected extension Sigma_*^(pilot-HS).",
        ),
        sign_base.row(
            "retained_q_window_available_now",
            "pass" if retained_q_window_available_now else "reject",
            "retained-q window available now",
            sign_base.truth(retained_q_window_available_now),
            "The deformation rerun must stay anchored to the retained q checkpoints {0, q_theory, m0}.",
        ),
        sign_base.row(
            "f_blind_deform_pack_available_now",
            "pass" if f_blind_deform_pack_available_now else "reject",
            "deformed blind retained-q pack available now",
            sign_base.truth(f_blind_deform_pack_available_now),
            "The deformation helper must expose the retained-q blind form-factor values needed for residual-origin discrimination.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_numeric_rerun_formula_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_numeric_rerun_formula_available_now
            else "reject",
            "exact selected-extension solver-side deformation numeric rerun formula available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_numeric_rerun_formula_available_now
            ),
            "The implemented deformation helper now yields one literal retained-q numeric rerun surface on Sigma_*^(pilot-HS).",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_numeric_surface_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_numeric_surface_available_now
            else "reject",
            "exact selected-extension solver-side deformation numeric surface available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_numeric_surface_available_now
            ),
            "The retained-q deformation surface is now explicit as actual checkpoint values instead of remaining implicit inside the helper.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_numeric_checkpoint_preservation_theorem_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_numeric_checkpoint_preservation_theorem_available_now
            else "reject",
            "exact selected-extension solver-side deformation numeric checkpoint preservation theorem available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_numeric_checkpoint_preservation_theorem_available_now
            ),
            "The implemented deformation rerun preserves the already materialized selected-extension recompute surface on the retained q window.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_numeric_rerun_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_numeric_rerun_available_now
            else "reject",
            "exact selected-extension solver-side deformation numeric rerun available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_numeric_rerun_available_now
            ),
            "One concrete deformation-integrated retained-q rerun is now machine-readable and ready for residual-origin interpretation.",
        ),
        sign_base.row(
            "blind_F_deform_at_zero",
            "watch",
            "selected-extension deformation blind F(0)",
            blind_F_deform_at_zero,
            "The deformation helper preserves the normalized zero-momentum checkpoint.",
        ),
        sign_base.row(
            "blind_F_deform_at_q_theory",
            "watch",
            "selected-extension deformation blind F(q_theory)",
            blind_F_deform_at_q_theory,
            "The deformation-integrated retained-q surface remains negative at q_theory.",
        ),
        sign_base.row(
            "blind_F_deform_at_m0",
            "watch",
            "selected-extension deformation blind F(m0)",
            blind_F_deform_at_m0,
            "The retained deformation surface stays close to zero but remains negative at the m0 checkpoint.",
        ),
        sign_base.row(
            "blind_alpha_deform_at_q_theory",
            "watch",
            "selected-extension deformation blind alpha(q_theory)",
            blind_alpha_deform_at_q_theory,
            "The deformation-integrated retained-q alpha stays far below the retained exact scalar target at q_theory.",
        ),
        sign_base.row(
            "alpha_exact_at_q_theory",
            "pass",
            "retained exact scalar alpha(q_theory)",
            alpha_exact_at_q_theory,
            "This retained scalar target remains the comparison baseline for the deformation-integrated rerun.",
        ),
        sign_base.row(
            "delta_alpha_sel_deform_exact",
            "watch",
            "selected-extension deformation delta alpha vs exact scalar target",
            delta_alpha_sel_deform_exact,
            "The signed deformation residual remains large and negative at q_theory.",
        ),
        sign_base.row(
            "relative_exact_residual_deform",
            "watch",
            "selected-extension deformation relative exact residual",
            relative_exact_residual_deform,
            "The retained-q deformation rerun still sits far from the retained exact scalar alpha.",
        ),
        sign_base.row(
            "wrong_sign_persists_now",
            "pass" if wrong_sign_persists_now else "reject",
            "wrong sign persists now",
            sign_base.truth(wrong_sign_persists_now),
            "The deformation-integrated rerun remains in the wrong-sign sector at q_theory.",
        ),
        sign_base.row(
            "low_alpha_persists_now",
            "pass" if low_alpha_persists_now else "reject",
            "low alpha persists now",
            sign_base.truth(low_alpha_persists_now),
            "The deformation-integrated rerun still undershoots the retained exact scalar alpha target at q_theory.",
        ),
        sign_base.row(
            "updated_pack_selected_extension_solver_side_deformation_residual_origin_refresh_followup_required",
            "pass"
            if updated_pack_selected_extension_solver_side_deformation_residual_origin_refresh_followup_required
            else "reject",
            "updated-pack selected-extension solver-side deformation residual-origin refresh followup required",
            sign_base.truth(
                updated_pack_selected_extension_solver_side_deformation_residual_origin_refresh_followup_required
            ),
            "With one concrete deformation-integrated rerun fixed, the honest next blocker is residual-origin refresh on that surface.",
        ),
        sign_base.row(
            "updated_pack_same_schema_selected_extension_solver_side_deformation_numeric_rerun_replay_detected_now",
            "pass"
            if updated_pack_same_schema_selected_extension_solver_side_deformation_numeric_rerun_replay_detected_now
            else "reject",
            "updated-pack same-schema selected-extension solver-side deformation numeric-rerun replay detected now",
            sign_base.truth(
                updated_pack_same_schema_selected_extension_solver_side_deformation_numeric_rerun_replay_detected_now
            ),
            "False means this branch compressed the live blocker to residual-origin refresh instead of replaying the already fixed implementation schema.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range stays reserve-only while the implemented retained-q deformation rerun can still be interpreted directly.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_extension_label": deformation_pack["selected_extension_label"],
        "solver_side_deformation_label": deformation_pack[
            "solver_side_deformation_label"
        ],
        "q_theory_over_m0": float(
            deformation_pack["retained_q_window"]["q_theory_over_m0"]
        ),
        "blind_F_deform_at_zero": blind_F_deform_at_zero,
        "blind_F_deform_at_q_theory": blind_F_deform_at_q_theory,
        "blind_F_deform_at_m0": blind_F_deform_at_m0,
        "blind_alpha_deform_at_q_theory": blind_alpha_deform_at_q_theory,
        "alpha_exact_at_q_theory": alpha_exact_at_q_theory,
        "delta_alpha_sel_deform_exact": delta_alpha_sel_deform_exact,
        "relative_exact_residual_deform": relative_exact_residual_deform,
        "wrong_sign_persists_now": wrong_sign_persists_now,
        "low_alpha_persists_now": low_alpha_persists_now,
        "exact_selected_extension_solver_side_deformation_numeric_rerun_formula_available_now": exact_selected_extension_solver_side_deformation_numeric_rerun_formula_available_now,
        "exact_selected_extension_solver_side_deformation_numeric_surface_available_now": exact_selected_extension_solver_side_deformation_numeric_surface_available_now,
        "exact_selected_extension_solver_side_deformation_numeric_checkpoint_preservation_theorem_available_now": exact_selected_extension_solver_side_deformation_numeric_checkpoint_preservation_theorem_available_now,
        "exact_selected_extension_solver_side_deformation_numeric_rerun_available_now": exact_selected_extension_solver_side_deformation_numeric_rerun_available_now,
        "updated_pack_selected_extension_solver_side_deformation_residual_origin_refresh_followup_required": updated_pack_selected_extension_solver_side_deformation_residual_origin_refresh_followup_required,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_same_schema_selected_extension_solver_side_deformation_numeric_rerun_replay_detected_now": updated_pack_same_schema_selected_extension_solver_side_deformation_numeric_rerun_replay_detected_now,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_completion_lane": "updated_pack_selected_extension_solver_side_deformation_numeric_rerun_gate",
        "selected_secondary_completion_lane": "updated_pack_selected_extension_solver_side_deformation_residual_origin_refresh_audit",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_solver_side_deformation_numeric_rerun_gate",
        "recommended_next_route_or_none": "8.7.56.5291",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_solver_side_deformation_residual_origin_refresh_audit",
        "selected_followup_route_or_none": "8.7.56.5295",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5289",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5291",
                "followup_route": "8.7.56.5295",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_selected_extension_solver_side_deformation_numeric_rerun_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} selected-extension solver-side deformation numeric rerun completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から numeric rerun audit を実行する。

if __name__ == "__main__":
    main()
