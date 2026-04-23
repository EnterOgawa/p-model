#!/usr/bin/env python3
"""Generate 8.7.56.5215-.5218 backend-integrated retained-q rerun audit artifacts."""

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
from scripts.quantum.blind_vector_selected_extension_backend import (
    build_selected_extension_backend_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5211-5214",
        "updated_pack_blind_vector_solver_side_backend_implementation_gate",
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

STEP_TAG = "8.7.56.5215-5218"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack blind-vector "
    "backend-integrated retained-q rerun audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_blind_vector_backend_integrated_retained_q_rerun_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_solver_side_backend_implementation_audited_numeric_rerun_"
    "primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_backend_integrated_retained_q_rerun_audited_residual_origin_"
    "refresh_primary_hybrid_reserve_secondary_next"
)
BLIND_KEY_MAP = {
    "blind_F_at_zero": "blind_F_at_zero",
    "blind_F_at_q_theory": "blind_F_at_q_theory",
    "blind_F_at_m0": "blind_F_at_m0",
    "blind_alpha_at_q_theory": "blind_alpha_at_q_theory",
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


# 関数: blind target surface が retained Phase 3 checkpoint を保存しているか判定する。
def checkpoint_preserved(blind_keys: dict, phase3_summary: dict) -> bool:
    """Return whether the backend-integrated blind keys preserve the retained Phase 3 values."""
    for pack_key, phase3_key in BLIND_KEY_MAP.items():
        if not math.isclose(
            float(blind_keys[pack_key]),
            float(phase3_summary[phase3_key]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            return False

    return True


# 関数: backend-integrated retained-q rerun audit の式を返す。
def build_formulae() -> dict[str, str]:
    """Return formulas used in the backend-integrated retained-q rerun audit."""
    return {
        "backend_pack": (
            "O_adapter,impl^(pilot-HS,legacy-vq) := "
            "build_selected_extension_backend_pack(ell_values=(1,2,3))"
        ),
        "retained_q_surface": (
            "Q_ret^(pilot-HS,backend) := {F_blind^(pilot-HS,backend)(0), "
            "F_blind^(pilot-HS,backend)(q_theory), "
            "F_blind^(pilot-HS,backend)(m0), "
            "alpha_blind^(pilot-HS,backend)(q_theory)}"
        ),
        "delta_exact": (
            "delta_alpha_sel^(pilot-HS,backend) := "
            "alpha_blind^(pilot-HS,backend)(q_theory) - alpha_exact(q_theory)"
        ),
        "relative_exact": (
            "r_sel^(pilot-HS,backend) := "
            "|delta_alpha_sel^(pilot-HS,backend)| / alpha_exact(q_theory)"
        ),
    }


# 関数: `.5215-.5218` を実行する。
def main() -> None:
    """Execute the backend-integrated retained-q rerun audit."""
    for path in (PRIOR_GATE, PHASE3_EVAL, SCALAR_TARGET):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    phase3_summary = sign_base.read_json(PHASE3_EVAL)["summary"]
    scalar_summary = sign_base.read_json(SCALAR_TARGET)["summary"]

    audit_selected = bool(
        prior_summary[
            "gate_b_updated_pack_blind_vector_backend_integrated_retained_q_rerun_promoted_next"
        ]
        and prior_summary[
            "gate_a_updated_pack_exact_blind_vector_solver_side_backend_implementation_available_now"
        ]
    )
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )

    backend_pack = build_selected_extension_backend_pack()
    selected_extension_label_matches_now = bool(
        backend_pack["selected_extension_label"] == "Sigma_*^(pilot-HS)"
    )
    retained_q_window_available_now = bool(
        {"zero", "q_theory_over_m0", "m0"} <= set(backend_pack["retained_q_window"].keys())
    )
    blind_target_keys_available_now = bool(
        set(BLIND_KEY_MAP.keys()) | {"delta_alpha_sel_exact"}
        <= set(backend_pack["blind_target_keys"].keys())
    )
    exact_blind_vector_backend_integrated_retained_q_rerun_formula_available_now = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selected_extension_label_matches_now
    )
    exact_blind_vector_backend_integrated_retained_q_surface_available_now = bool(
        exact_blind_vector_backend_integrated_retained_q_rerun_formula_available_now
        and retained_q_window_available_now
        and blind_target_keys_available_now
    )
    exact_blind_vector_backend_integrated_retained_q_checkpoint_preservation_theorem_available_now = bool(
        exact_blind_vector_backend_integrated_retained_q_surface_available_now
        and checkpoint_preserved(backend_pack["blind_target_keys"], phase3_summary)
    )
    exact_blind_vector_backend_integrated_retained_q_rerun_available_now = bool(
        exact_blind_vector_backend_integrated_retained_q_rerun_formula_available_now
        and exact_blind_vector_backend_integrated_retained_q_surface_available_now
        and exact_blind_vector_backend_integrated_retained_q_checkpoint_preservation_theorem_available_now
    )

    q_theory_over_m0 = float(backend_pack["retained_q_window"]["q_theory_over_m0"])
    alpha_exact_at_q_theory = float(scalar_summary["alpha_exact_at_q_theory"])
    blind_F_at_zero = float(backend_pack["blind_target_keys"]["blind_F_at_zero"])
    blind_F_at_q_theory = float(backend_pack["blind_target_keys"]["blind_F_at_q_theory"])
    blind_F_at_m0 = float(backend_pack["blind_target_keys"]["blind_F_at_m0"])
    blind_alpha_at_q_theory = float(
        backend_pack["blind_target_keys"]["blind_alpha_at_q_theory"]
    )
    delta_alpha_sel_exact = float(blind_alpha_at_q_theory - alpha_exact_at_q_theory)
    relative_exact_residual = float(abs(delta_alpha_sel_exact) / alpha_exact_at_q_theory)
    wrong_sign_persists_now = bool(blind_F_at_q_theory < 0.0)
    low_alpha_persists_now = bool(blind_alpha_at_q_theory < alpha_exact_at_q_theory)

    updated_pack_blind_vector_residual_origin_refresh_followup_required = bool(
        exact_blind_vector_backend_integrated_retained_q_rerun_available_now
    )
    updated_pack_same_schema_blind_vector_backend_integrated_retained_q_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_blind_vector_backend_integrated_retained_q_rerun_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack blind-vector backend-integrated retained-q rerun audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after one concrete selected-extension backend implementation is already official and the live blocker is retained-q rerun itself.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The active blind-vector lane stays on computation-side blocker reduction rather than falling back to theorem-family replay.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Backend-integrated rerun is only honest while exhausted surrogate and selector-replay branches stay closed.",
        ),
        sign_base.row(
            "selected_extension_label_matches_now",
            "pass" if selected_extension_label_matches_now else "reject",
            "selected-extension label matches now",
            sign_base.truth(selected_extension_label_matches_now),
            "The backend-integrated rerun remains meaningful only while the helper still materializes the adopted selected extension Sigma_*^(pilot-HS).",
        ),
        sign_base.row(
            "retained_q_window_available_now",
            "pass" if retained_q_window_available_now else "reject",
            "retained-q window available now",
            sign_base.truth(retained_q_window_available_now),
            "The backend-integrated rerun must stay anchored to the retained q checkpoints {0, q_theory, m0}.",
        ),
        sign_base.row(
            "blind_target_keys_available_now",
            "pass" if blind_target_keys_available_now else "reject",
            "blind target keys available now",
            sign_base.truth(blind_target_keys_available_now),
            "The integrated backend pack must actually expose the blind checkpoint values needed for residual-origin discrimination.",
        ),
        sign_base.row(
            "exact_blind_vector_backend_integrated_retained_q_rerun_formula_available_now",
            "pass"
            if exact_blind_vector_backend_integrated_retained_q_rerun_formula_available_now
            else "reject",
            "exact blind-vector backend-integrated retained-q rerun formula available now",
            sign_base.truth(
                exact_blind_vector_backend_integrated_retained_q_rerun_formula_available_now
            ),
            "The selected-extension backend implementation now yields one literal retained-q rerun surface on the integrated backend pack.",
        ),
        sign_base.row(
            "exact_blind_vector_backend_integrated_retained_q_surface_available_now",
            "pass"
            if exact_blind_vector_backend_integrated_retained_q_surface_available_now
            else "reject",
            "exact blind-vector backend-integrated retained-q surface available now",
            sign_base.truth(
                exact_blind_vector_backend_integrated_retained_q_surface_available_now
            ),
            "The integrated backend pack now materializes the blind retained-q checkpoint surface instead of leaving it as an unattached contract.",
        ),
        sign_base.row(
            "exact_blind_vector_backend_integrated_retained_q_checkpoint_preservation_theorem_available_now",
            "pass"
            if exact_blind_vector_backend_integrated_retained_q_checkpoint_preservation_theorem_available_now
            else "reject",
            "exact blind-vector backend-integrated retained-q checkpoint preservation theorem available now",
            sign_base.truth(
                exact_blind_vector_backend_integrated_retained_q_checkpoint_preservation_theorem_available_now
            ),
            "The integrated backend rerun preserves the retained Phase 3 blind checkpoint values exactly; it does not invent a new hidden surface.",
        ),
        sign_base.row(
            "exact_blind_vector_backend_integrated_retained_q_rerun_available_now",
            "pass"
            if exact_blind_vector_backend_integrated_retained_q_rerun_available_now
            else "reject",
            "exact blind-vector backend-integrated retained-q rerun available now",
            sign_base.truth(
                exact_blind_vector_backend_integrated_retained_q_rerun_available_now
            ),
            "The honest next blocker is now residual-origin refresh on the integrated backend surface, not backend implementation ambiguity.",
        ),
        sign_base.row(
            "blind_F_at_zero",
            "pass" if math.isclose(blind_F_at_zero, 1.0, rel_tol=0.0, abs_tol=1.0e-12) else "watch",
            "backend-integrated blind F(0)",
            blind_F_at_zero,
            "The backend-integrated retained-q surface keeps the normalization F(0)=1.",
        ),
        sign_base.row(
            "blind_F_at_q_theory",
            "watch",
            "backend-integrated blind F(q_theory)",
            blind_F_at_q_theory,
            "The backend-integrated retained-q surface still sits in the wrong-sign sector at q_theory.",
        ),
        sign_base.row(
            "blind_alpha_at_q_theory",
            "watch",
            "backend-integrated blind alpha(q_theory)",
            blind_alpha_at_q_theory,
            "The backend-integrated retained-q surface still carries the low-alpha first-shot value at q_theory.",
        ),
        sign_base.row(
            "delta_alpha_sel_exact",
            "watch",
            "backend-integrated delta alpha vs exact scalar target",
            delta_alpha_sel_exact,
            "Negative means the backend-integrated retained-q surface remains below the retained exact scalar target.",
        ),
        sign_base.row(
            "relative_exact_residual",
            "watch",
            "backend-integrated relative residual vs exact scalar target",
            relative_exact_residual,
            "The integrated backend rerun still differs from the retained exact scalar alpha by about 91.6%.",
        ),
        sign_base.row(
            "wrong_sign_persists_now",
            "pass" if wrong_sign_persists_now else "reject",
            "wrong sign persists now",
            sign_base.truth(wrong_sign_persists_now),
            "Wrong-sign persistence is now attached to the integrated backend surface rather than to an unimplemented adapter hypothesis.",
        ),
        sign_base.row(
            "low_alpha_persists_now",
            "pass" if low_alpha_persists_now else "reject",
            "low alpha persists now",
            sign_base.truth(low_alpha_persists_now),
            "Low-alpha persistence is now attached to the integrated backend surface rather than to selector ambiguity.",
        ),
        sign_base.row(
            "updated_pack_blind_vector_residual_origin_refresh_followup_required",
            "pass"
            if updated_pack_blind_vector_residual_origin_refresh_followup_required
            else "reject",
            "updated-pack blind-vector residual-origin refresh followup required",
            sign_base.truth(
                updated_pack_blind_vector_residual_origin_refresh_followup_required
            ),
            "The honest next blocker is now residual-origin refresh on the backend-integrated retained-q surface.",
        ),
        sign_base.row(
            "updated_pack_same_schema_blind_vector_backend_integrated_retained_q_replay_detected_now",
            "pass"
            if updated_pack_same_schema_blind_vector_backend_integrated_retained_q_replay_detected_now
            else "reject",
            "updated-pack same-schema blind-vector backend-integrated retained-q replay detected now",
            sign_base.truth(
                updated_pack_same_schema_blind_vector_backend_integrated_retained_q_replay_detected_now
            ),
            "False means this turn materialized the integrated retained-q surface instead of replaying a theorem-only backend contract.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range stays reserve-only because backend-integrated retained-q refresh is still the live blocker.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "backend_pack_selected_extension_label": backend_pack["selected_extension_label"],
        "q_theory_over_m0": q_theory_over_m0,
        "alpha_exact_at_q_theory": alpha_exact_at_q_theory,
        "blind_F_at_zero": blind_F_at_zero,
        "blind_F_at_q_theory": blind_F_at_q_theory,
        "blind_alpha_at_q_theory": blind_alpha_at_q_theory,
        "blind_F_at_m0": blind_F_at_m0,
        "delta_alpha_sel_exact": delta_alpha_sel_exact,
        "relative_exact_residual": relative_exact_residual,
        "ell_scan_counts": backend_pack["ell_scan_counts"],
        "base_mode_counts": backend_pack["base_mode_counts"],
        "exact_ladder_row_count": int(backend_pack["exact_ladder_row_count"]),
        "comparison_row_count": int(backend_pack["comparison_row_count"]),
        "exact_blind_vector_backend_integrated_retained_q_rerun_formula_available_now": exact_blind_vector_backend_integrated_retained_q_rerun_formula_available_now,
        "exact_blind_vector_backend_integrated_retained_q_surface_available_now": exact_blind_vector_backend_integrated_retained_q_surface_available_now,
        "exact_blind_vector_backend_integrated_retained_q_checkpoint_preservation_theorem_available_now": exact_blind_vector_backend_integrated_retained_q_checkpoint_preservation_theorem_available_now,
        "exact_blind_vector_backend_integrated_retained_q_rerun_available_now": exact_blind_vector_backend_integrated_retained_q_rerun_available_now,
        "wrong_sign_persists_now": wrong_sign_persists_now,
        "low_alpha_persists_now": low_alpha_persists_now,
        "updated_pack_blind_vector_residual_origin_refresh_followup_required": updated_pack_blind_vector_residual_origin_refresh_followup_required,
        "updated_pack_same_schema_blind_vector_backend_integrated_retained_q_replay_detected_now": updated_pack_same_schema_blind_vector_backend_integrated_retained_q_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": updated_pack_blind_vector_residual_origin_refresh_followup_required,
        "selected_primary_completion_lane": "updated_pack_blind_vector_backend_integrated_residual_origin_refresh_audit",
        "selected_secondary_completion_lane": "updated_pack_selected_extension_solver_recompute_closeout",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_backend_integrated_retained_q_rerun_gate",
        "recommended_next_route_or_none": "8.7.56.5219",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_backend_integrated_residual_origin_refresh_audit",
        "selected_followup_route_or_none": "8.7.56.5223",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5217",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "phase3_eval": sign_base.display_path(PHASE3_EVAL),
                "scalar_target_eval": sign_base.display_path(SCALAR_TARGET),
                "backend_helper": sign_base.display_path(
                    ROOT / "scripts" / "quantum" / "blind_vector_selected_extension_backend.py"
                ),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5219",
                "followup_route": "8.7.56.5223",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_blind_vector_backend_integrated_retained_q_rerun_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} blind-vector backend-integrated retained-q rerun completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
