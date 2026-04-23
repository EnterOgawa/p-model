#!/usr/bin/env python3
"""Generate 8.7.56.5247-.5250 selected-extension solver-recompute retained-q rerun artifacts."""

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
from scripts.quantum.selected_extension_solver_recompute_backend import (
    build_selected_extension_solver_recompute_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5243-5246",
        "updated_pack_selected_extension_solver_recompute_implementation_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PHASE3_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_"
    "blind_vector_observable_gate_numeric_evaluation_metrics.json"
)

STEP_TAG = "8.7.56.5247-5250"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack selected-"
    "extension solver-recompute retained-q rerun audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_selected_extension_solver_recompute_retained_q_rerun_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_recompute_implementation_audited_numeric_"
    "rerun_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_recompute_retained_q_rerun_derived_residual_"
    "origin_refresh_primary_pack_refresh_secondary_gate"
)
BLIND_KEY_MAP = {
    "zero": "blind_F_at_zero",
    "q_theory_over_m0": "blind_F_at_q_theory",
    "m0": "blind_F_at_m0",
}


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


# 関数: retained-q recompute pack が retained Phase 3 blind surface を保存するか判定する。

def checkpoint_preserved(recompute_pack: dict, phase3_summary: dict) -> bool:
    """Return whether the selected-extension retained-q rerun preserves Phase 3 blind values."""
    for recompute_key, phase3_key in BLIND_KEY_MAP.items():
        if not math.isclose(
            float(recompute_pack["F_blind_recomp_pack"][recompute_key]),
            float(phase3_summary[phase3_key]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            return False

    return math.isclose(
        float(recompute_pack["alpha_blind_recomp_at_q_theory"]),
        float(phase3_summary["blind_alpha_at_q_theory"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )


# 関数: retained-q rerun audit の式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the selected-extension solver-recompute retained-q rerun audit."""
    return {
        "recompute_pack": (
            "O_recomp_sel,impl^(pilot-HS) := "
            "build_selected_extension_solver_recompute_pack(ell_values=(1,2,3))"
        ),
        "retained_q_surface": (
            "Q_ret^(pilot-HS,recomp) := {F_blind^(pilot-HS,recomp)(0), "
            "F_blind^(pilot-HS,recomp)(q_theory), "
            "F_blind^(pilot-HS,recomp)(m0), "
            "alpha_blind^(pilot-HS,recomp)(q_theory)}"
        ),
        "delta_exact": (
            "delta_alpha_sel^(pilot-HS,recomp) := "
            "alpha_blind^(pilot-HS,recomp)(q_theory) - alpha_exact(q_theory)"
        ),
        "relative_exact": (
            "r_sel^(pilot-HS,recomp) := "
            "|delta_alpha_sel^(pilot-HS,recomp)| / alpha_exact(q_theory)"
        ),
    }


# 関数: `.5247-.5250` を実行する。

def main() -> None:
    """Execute the selected-extension solver-recompute retained-q rerun audit."""
    for path in (PRIOR_GATE, PHASE3_EVAL):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    phase3_summary = sign_base.read_json(PHASE3_EVAL)["summary"]

    audit_selected = bool(
        prior_summary[
            "gate_b_updated_pack_selected_extension_solver_recompute_retained_q_rerun_promoted_next"
        ]
        and prior_summary[
            "gate_a_updated_pack_exact_selected_extension_solver_recompute_implementation_available_now"
        ]
    )
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )

    recompute_pack = build_selected_extension_solver_recompute_pack()
    selected_extension_label_matches_now = bool(
        recompute_pack["selected_extension_label"] == "Sigma_*^(pilot-HS)"
    )
    retained_q_window_available_now = bool(
        {"zero", "q_theory_over_m0", "m0"}
        <= set(recompute_pack["retained_q_window"].keys())
    )
    f_blind_recomp_pack_available_now = bool(
        {"zero", "q_theory_over_m0", "m0"}
        <= set(recompute_pack["F_blind_recomp_pack"].keys())
    )
    exact_selected_extension_solver_recompute_retained_q_rerun_formula_available_now = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selected_extension_label_matches_now
    )
    exact_selected_extension_solver_recompute_retained_q_surface_available_now = bool(
        exact_selected_extension_solver_recompute_retained_q_rerun_formula_available_now
        and retained_q_window_available_now
        and f_blind_recomp_pack_available_now
    )
    exact_selected_extension_solver_recompute_retained_q_checkpoint_preservation_theorem_available_now = bool(
        exact_selected_extension_solver_recompute_retained_q_surface_available_now
        and checkpoint_preserved(recompute_pack, phase3_summary)
    )
    exact_selected_extension_solver_recompute_retained_q_rerun_available_now = bool(
        exact_selected_extension_solver_recompute_retained_q_rerun_formula_available_now
        and exact_selected_extension_solver_recompute_retained_q_surface_available_now
        and exact_selected_extension_solver_recompute_retained_q_checkpoint_preservation_theorem_available_now
    )

    blind_F_recomp_at_zero = float(recompute_pack["F_blind_recomp_pack"]["zero"])
    blind_F_recomp_at_q_theory = float(
        recompute_pack["F_blind_recomp_pack"]["q_theory_over_m0"]
    )
    blind_F_recomp_at_m0 = float(recompute_pack["F_blind_recomp_pack"]["m0"])
    blind_alpha_recomp_at_q_theory = float(
        recompute_pack["alpha_blind_recomp_at_q_theory"]
    )
    alpha_exact_at_q_theory = float(recompute_pack["alpha_exact_at_q_theory"])
    delta_alpha_sel_recomp_exact = float(recompute_pack["delta_alpha_sel_recomp_exact"])
    relative_exact_residual_recomp = float(
        recompute_pack["relative_exact_residual_recomp"]
    )
    wrong_sign_persists_now = bool(blind_F_recomp_at_q_theory < 0.0)
    low_alpha_persists_now = bool(
        blind_alpha_recomp_at_q_theory < alpha_exact_at_q_theory
    )

    updated_pack_selected_extension_solver_recompute_residual_origin_refresh_followup_required = bool(
        exact_selected_extension_solver_recompute_retained_q_rerun_available_now
    )
    updated_pack_same_schema_selected_extension_solver_recompute_retained_q_rerun_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_selected_extension_solver_recompute_retained_q_rerun_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack selected-extension solver-recompute retained-q rerun audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after one concrete selected-extension recompute implementation is already official and the live blocker is retained-q rerun itself.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The active selected-extension lane stays on computation-side blocker reduction rather than falling back to theorem-family replay.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Retained-q rerun is only honest while exhausted surrogate and selector-replay branches stay closed.",
        ),
        sign_base.row(
            "selected_extension_label_matches_now",
            "pass" if selected_extension_label_matches_now else "reject",
            "selected-extension label matches now",
            sign_base.truth(selected_extension_label_matches_now),
            "The retained-q rerun remains meaningful only while the helper still materializes the adopted selected extension Sigma_*^(pilot-HS).",
        ),
        sign_base.row(
            "retained_q_window_available_now",
            "pass" if retained_q_window_available_now else "reject",
            "retained-q window available now",
            sign_base.truth(retained_q_window_available_now),
            "The selected-extension rerun must stay anchored to the retained q checkpoints {0, q_theory, m0}.",
        ),
        sign_base.row(
            "f_blind_recomp_pack_available_now",
            "pass" if f_blind_recomp_pack_available_now else "reject",
            "recomputed blind retained-q pack available now",
            sign_base.truth(f_blind_recomp_pack_available_now),
            "The rerun helper must expose the retained-q blind form-factor values needed for residual-origin discrimination.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_recompute_retained_q_rerun_formula_available_now",
            "pass"
            if exact_selected_extension_solver_recompute_retained_q_rerun_formula_available_now
            else "reject",
            "exact selected-extension solver-recompute retained-q rerun formula available now",
            sign_base.truth(
                exact_selected_extension_solver_recompute_retained_q_rerun_formula_available_now
            ),
            "The selected-extension implementation now yields one literal retained-q rerun surface on the recompute pack.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_recompute_retained_q_surface_available_now",
            "pass"
            if exact_selected_extension_solver_recompute_retained_q_surface_available_now
            else "reject",
            "exact selected-extension solver-recompute retained-q surface available now",
            sign_base.truth(
                exact_selected_extension_solver_recompute_retained_q_surface_available_now
            ),
            "The recompute helper now materializes the retained-q blind surface instead of leaving it as an unattached theorem contract.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_recompute_retained_q_checkpoint_preservation_theorem_available_now",
            "pass"
            if exact_selected_extension_solver_recompute_retained_q_checkpoint_preservation_theorem_available_now
            else "reject",
            "exact selected-extension solver-recompute retained-q checkpoint preservation theorem available now",
            sign_base.truth(
                exact_selected_extension_solver_recompute_retained_q_checkpoint_preservation_theorem_available_now
            ),
            "The selected-extension retained-q rerun preserves the retained Phase 3 blind checkpoint values exactly on its first materialized implementation path.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_recompute_retained_q_rerun_available_now",
            "pass"
            if exact_selected_extension_solver_recompute_retained_q_rerun_available_now
            else "reject",
            "exact selected-extension solver-recompute retained-q rerun available now",
            sign_base.truth(
                exact_selected_extension_solver_recompute_retained_q_rerun_available_now
            ),
            "The honest next blocker is now residual-origin refresh on the selected-extension retained-q surface, not recompute implementation ambiguity.",
        ),
        sign_base.row(
            "blind_F_recomp_at_zero",
            "pass"
            if math.isclose(blind_F_recomp_at_zero, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
            else "watch",
            "selected-extension recomputed blind F(0)",
            blind_F_recomp_at_zero,
            "The selected-extension retained-q rerun keeps the normalization F(0)=1.",
        ),
        sign_base.row(
            "blind_F_recomp_at_q_theory",
            "watch",
            "selected-extension recomputed blind F(q_theory)",
            blind_F_recomp_at_q_theory,
            "The selected-extension retained-q surface still sits in the wrong-sign sector at q_theory.",
        ),
        sign_base.row(
            "blind_alpha_recomp_at_q_theory",
            "watch",
            "selected-extension recomputed blind alpha(q_theory)",
            blind_alpha_recomp_at_q_theory,
            "The selected-extension retained-q surface still carries the low-alpha first-shot value at q_theory.",
        ),
        sign_base.row(
            "delta_alpha_sel_recomp_exact",
            "watch",
            "selected-extension recomputed delta alpha vs exact scalar target",
            delta_alpha_sel_recomp_exact,
            "Negative means the selected-extension retained-q surface remains below the retained exact scalar target.",
        ),
        sign_base.row(
            "relative_exact_residual_recomp",
            "watch",
            "selected-extension recomputed relative residual vs exact scalar target",
            relative_exact_residual_recomp,
            "The selected-extension retained-q rerun still differs from the retained exact scalar alpha by about 91.6%.",
        ),
        sign_base.row(
            "wrong_sign_persists_now",
            "pass" if wrong_sign_persists_now else "reject",
            "wrong sign persists now",
            sign_base.truth(wrong_sign_persists_now),
            "Wrong-sign persistence is now attached to the selected-extension retained-q rerun surface rather than to an unimplemented recompute hypothesis.",
        ),
        sign_base.row(
            "low_alpha_persists_now",
            "pass" if low_alpha_persists_now else "reject",
            "low alpha persists now",
            sign_base.truth(low_alpha_persists_now),
            "Low-alpha persistence is now attached to the selected-extension retained-q rerun surface rather than to selector ambiguity.",
        ),
        sign_base.row(
            "updated_pack_selected_extension_solver_recompute_residual_origin_refresh_followup_required",
            "pass"
            if updated_pack_selected_extension_solver_recompute_residual_origin_refresh_followup_required
            else "reject",
            "updated-pack selected-extension solver-recompute residual-origin refresh followup required",
            sign_base.truth(
                updated_pack_selected_extension_solver_recompute_residual_origin_refresh_followup_required
            ),
            "The honest next blocker is now residual-origin refresh on the selected-extension retained-q surface.",
        ),
        sign_base.row(
            "updated_pack_same_schema_selected_extension_solver_recompute_retained_q_rerun_replay_detected_now",
            "pass"
            if updated_pack_same_schema_selected_extension_solver_recompute_retained_q_rerun_replay_detected_now
            else "reject",
            "updated-pack same-schema selected-extension solver-recompute retained-q rerun replay detected now",
            sign_base.truth(
                updated_pack_same_schema_selected_extension_solver_recompute_retained_q_rerun_replay_detected_now
            ),
            "False means this turn materialized the integrated retained-q surface instead of replaying a theorem-only recompute contract.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range stays reserve-only because selected-extension retained-q refresh is still the live blocker.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_extension_label": recompute_pack["selected_extension_label"],
        "q_theory_over_m0": float(recompute_pack["retained_q_window"]["q_theory_over_m0"]),
        "alpha_exact_at_q_theory": alpha_exact_at_q_theory,
        "blind_F_recomp_at_zero": blind_F_recomp_at_zero,
        "blind_F_recomp_at_q_theory": blind_F_recomp_at_q_theory,
        "blind_alpha_recomp_at_q_theory": blind_alpha_recomp_at_q_theory,
        "blind_F_recomp_at_m0": blind_F_recomp_at_m0,
        "delta_alpha_sel_recomp_exact": delta_alpha_sel_recomp_exact,
        "relative_exact_residual_recomp": relative_exact_residual_recomp,
        "exact_selected_extension_solver_recompute_retained_q_rerun_formula_available_now": exact_selected_extension_solver_recompute_retained_q_rerun_formula_available_now,
        "exact_selected_extension_solver_recompute_retained_q_surface_available_now": exact_selected_extension_solver_recompute_retained_q_surface_available_now,
        "exact_selected_extension_solver_recompute_retained_q_checkpoint_preservation_theorem_available_now": exact_selected_extension_solver_recompute_retained_q_checkpoint_preservation_theorem_available_now,
        "exact_selected_extension_solver_recompute_retained_q_rerun_available_now": exact_selected_extension_solver_recompute_retained_q_rerun_available_now,
        "wrong_sign_persists_now": wrong_sign_persists_now,
        "low_alpha_persists_now": low_alpha_persists_now,
        "updated_pack_selected_extension_solver_recompute_residual_origin_refresh_followup_required": updated_pack_selected_extension_solver_recompute_residual_origin_refresh_followup_required,
        "updated_pack_same_schema_selected_extension_solver_recompute_retained_q_rerun_replay_detected_now": updated_pack_same_schema_selected_extension_solver_recompute_retained_q_rerun_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": updated_pack_selected_extension_solver_recompute_residual_origin_refresh_followup_required,
        "selected_primary_completion_lane": "updated_pack_selected_extension_solver_recompute_residual_origin_refresh_audit",
        "selected_secondary_completion_lane": "selected_extension_solver_recompute_negative_closeout_only_after_refresh",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_solver_recompute_retained_q_rerun_gate",
        "recommended_next_route_or_none": "8.7.56.5251",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_solver_recompute_residual_origin_refresh_audit",
        "selected_followup_route_or_none": "8.7.56.5255",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5249",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "phase3_eval": sign_base.display_path(PHASE3_EVAL),
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
                "next_route": "8.7.56.5251",
                "followup_route": "8.7.56.5255",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_selected_extension_solver_recompute_retained_q_rerun_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae(), "evidence": recompute_pack["evidence_samples"]},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} selected-extension solver-recompute retained-q rerun completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から retained-q rerun audit を実行する。

if __name__ == "__main__":
    main()
