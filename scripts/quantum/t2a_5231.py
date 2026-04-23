#!/usr/bin/env python3
"""Generate 8.7.56.5231-.5234 selected-extension solver-recompute contract artifacts."""

from __future__ import annotations

import csv
import json
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
        "8.7.56.5227-5230",
        "updated_pack_blind_vector_backend_integrated_residual_origin_refresh_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
SELECTED_EXTENSION_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5139-5142",
        "updated_pack_external_rule_selector_selected_extension_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
BACKEND_IMPLEMENTATION_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5211-5214",
        "updated_pack_blind_vector_solver_side_backend_implementation_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5231-5234"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack selected-"
    "extension solver-recompute contract audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_selected_extension_solver_recompute_contract_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_backend_integrated_retained_q_rerun_preserves_phase3_failure_"
    "closeout_completed_selected_extension_solver_recompute_primary_hybrid_"
    "reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_recompute_contract_derived_implementation_"
    "primary_pack_refresh_secondary_gate"
)
REQUIRED_BLIND_KEYS = (
    "blind_F_at_zero",
    "blind_F_at_q_theory",
    "blind_F_at_m0",
    "blind_alpha_at_q_theory",
    "delta_alpha_sel_exact",
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


# 関数: solver-recompute contract の式を返す。
def build_formulae() -> dict[str, str]:
    """Return formulas used in the selected-extension solver-recompute contract audit."""
    return {
        "input_pack": (
            "I_recomp_sel^(pilot-HS) := {Sigma_*^(pilot-HS), "
            "B_backend^(pilot-HS), Q_ret = {0, q_theory, m0}, "
            "blind_checkpoint^(pilot-HS), alpha_exact(q_theory)}"
        ),
        "output_pack": (
            "O_recomp_sel^(pilot-HS) := {K_eff^(pilot-HS,recomp)[Q_ret], "
            "Z_eff^(pilot-HS,recomp,T)[Q_ret], F_blind^(pilot-HS,recomp)[Q_ret], "
            "alpha_blind^(pilot-HS,recomp)(q_theory), "
            "delta_alpha_sel^(pilot-HS,recomp)}"
        ),
        "recompute_contract": (
            "C_recomp_sel^(pilot-HS) : I_recomp_sel^(pilot-HS) -> "
            "O_recomp_sel^(pilot-HS)"
        ),
        "residual_discriminator": (
            "Delta_recomp^(pilot-HS) := compare("
            "O_recomp_sel^(pilot-HS), blind_checkpoint^(pilot-HS), alpha_exact)"
        ),
        "front_runner": (
            "C_recomp_sel,front^(pilot-HS) := "
            "(build_selected_extension_backend_pack, retained-q observable path)"
        ),
    }


# 関数: `.5231-.5234` を実行する。
def main() -> None:
    """Execute the selected-extension solver-recompute contract audit."""
    for path in (PRIOR_GATE, SELECTED_EXTENSION_GATE, BACKEND_IMPLEMENTATION_GATE):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    selected_summary = sign_base.read_json(SELECTED_EXTENSION_GATE)["summary"]
    backend_summary = sign_base.read_json(BACKEND_IMPLEMENTATION_GATE)["summary"]

    audit_selected = bool(
        prior_summary[
            "gate_b_updated_pack_selected_extension_solver_recompute_lane_promoted_next"
        ]
        and prior_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    selected_extension_available = bool(
        selected_summary[
            "gate_a_updated_pack_exact_external_rule_selector_selected_extension_available_now"
        ]
        and selected_summary[
            "exact_external_rule_selector_selected_extension_available_now"
        ]
    )
    backend_implementation_available = bool(
        backend_summary[
            "gate_a_updated_pack_exact_blind_vector_solver_side_backend_implementation_available_now"
        ]
    )

    backend_pack = build_selected_extension_backend_pack()
    retained_q_window_available = bool(
        all(
            key in backend_pack["retained_q_window"]
            for key in ("zero", "q_theory_over_m0", "m0")
        )
    )
    blind_checkpoint_keys_available = bool(
        all(key in backend_pack["blind_target_keys"] for key in REQUIRED_BLIND_KEYS)
    )
    backend_pack_nonempty_now = bool(
        backend_pack["ell_scan_counts"]
        and backend_pack["base_mode_counts"]
        and backend_pack["exact_ladder_row_count"] > 0
        and backend_pack["comparison_row_count"] > 0
    )
    retained_anchor_match_preserved_now = bool(
        backend_pack["best_exact_match"] == backend_pack["retained_anchor_row"]
    )

    exact_selected_extension_solver_recompute_input_pack_formula_available_now = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selected_extension_available
        and backend_implementation_available
        and retained_q_window_available
        and blind_checkpoint_keys_available
        and backend_pack_nonempty_now
    )
    exact_selected_extension_solver_recompute_output_pack_formula_available_now = bool(
        exact_selected_extension_solver_recompute_input_pack_formula_available_now
    )
    exact_selected_extension_solver_recompute_contract_formula_available_now = bool(
        exact_selected_extension_solver_recompute_input_pack_formula_available_now
        and retained_anchor_match_preserved_now
    )
    exact_selected_extension_solver_recompute_residual_discriminator_formula_available_now = bool(
        exact_selected_extension_solver_recompute_contract_formula_available_now
    )
    exact_selected_extension_solver_recompute_front_runner_compatibility_theorem_available_now = bool(
        exact_selected_extension_solver_recompute_contract_formula_available_now
    )
    updated_pack_selected_extension_solver_recompute_implementation_followup_required = bool(
        exact_selected_extension_solver_recompute_front_runner_compatibility_theorem_available_now
    )
    updated_pack_same_schema_selected_extension_solver_recompute_contract_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_selected_extension_solver_recompute_contract_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack selected-extension solver-recompute contract audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after blind-vector direct computation has closed negatively and the honest next blocker is selected-extension solver recomputation.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The new lane stays computation-first instead of reopening exhausted blind replay or selector-family branches.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Solver-recompute work remains honest only while exhausted surrogate and replay explanations stay closed.",
        ),
        sign_base.row(
            "selected_extension_available_now",
            "pass" if selected_extension_available else "reject",
            "selected extension available now",
            sign_base.truth(selected_extension_available),
            "The solver-recompute lane is meaningful only while one concrete selected extension Sigma_*^(pilot-HS) remains fixed.",
        ),
        sign_base.row(
            "backend_implementation_available_now",
            "pass" if backend_implementation_available else "reject",
            "backend implementation available now",
            sign_base.truth(backend_implementation_available),
            "The retained-q backend helper must already exist before it can be promoted as the front-runner input pack for solver recomputation.",
        ),
        sign_base.row(
            "retained_q_window_available_now",
            "pass" if retained_q_window_available else "reject",
            "retained-q window available now",
            sign_base.truth(retained_q_window_available),
            "The recomputation contract remains anchored to the retained q-window {0, q_theory, m0}.",
        ),
        sign_base.row(
            "blind_checkpoint_keys_available_now",
            "pass" if blind_checkpoint_keys_available else "reject",
            "blind checkpoint keys available now",
            sign_base.truth(blind_checkpoint_keys_available),
            "The solver-recompute contract must compare against the retained blind checkpoint rather than inventing a fresh target surface.",
        ),
        sign_base.row(
            "backend_pack_nonempty_now",
            "pass" if backend_pack_nonempty_now else "reject",
            "backend pack nonempty now",
            sign_base.truth(backend_pack_nonempty_now),
            "The materialized helper already yields nonempty ell scans, base modes, exact ladder rows, and comparison rows for the selected extension.",
        ),
        sign_base.row(
            "retained_anchor_match_preserved_now",
            "pass" if retained_anchor_match_preserved_now else "reject",
            "retained anchor match preserved now",
            sign_base.truth(retained_anchor_match_preserved_now),
            "The helper path still preserves the retained exact anchor row, so it is admissible as the recomputation front-runner input pack.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_recompute_input_pack_formula_available_now",
            "pass"
            if exact_selected_extension_solver_recompute_input_pack_formula_available_now
            else "reject",
            "exact selected-extension solver-recompute input-pack formula available now",
            sign_base.truth(
                exact_selected_extension_solver_recompute_input_pack_formula_available_now
            ),
            "The theorem stack now fixes one literal input pack for selected-extension solver recomputation instead of leaving the solver side as an unstructured future task.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_recompute_output_pack_formula_available_now",
            "pass"
            if exact_selected_extension_solver_recompute_output_pack_formula_available_now
            else "reject",
            "exact selected-extension solver-recompute output-pack formula available now",
            sign_base.truth(
                exact_selected_extension_solver_recompute_output_pack_formula_available_now
            ),
            "The recomputation lane now names the exact output objects it must produce before any new residual-origin verdict can be honest.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_recompute_contract_formula_available_now",
            "pass"
            if exact_selected_extension_solver_recompute_contract_formula_available_now
            else "reject",
            "exact selected-extension solver-recompute contract formula available now",
            sign_base.truth(
                exact_selected_extension_solver_recompute_contract_formula_available_now
            ),
            "The live blocker is now a concrete solver contract on Sigma_*^(pilot-HS), not a vague request for more computation.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_recompute_residual_discriminator_formula_available_now",
            "pass"
            if exact_selected_extension_solver_recompute_residual_discriminator_formula_available_now
            else "reject",
            "exact selected-extension solver-recompute residual discriminator formula available now",
            sign_base.truth(
                exact_selected_extension_solver_recompute_residual_discriminator_formula_available_now
            ),
            "The recomputation lane now closes the exact compare-against-checkpoint discriminator it must use to decide whether solver-side deformation is truly the residual origin.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_recompute_front_runner_compatibility_theorem_available_now",
            "pass"
            if exact_selected_extension_solver_recompute_front_runner_compatibility_theorem_available_now
            else "reject",
            "exact selected-extension solver-recompute front-runner compatibility theorem available now",
            sign_base.truth(
                exact_selected_extension_solver_recompute_front_runner_compatibility_theorem_available_now
            ),
            "The existing backend helper is now officially compatible with the new solver-recompute lane as its front-runner input pack.",
        ),
        sign_base.row(
            "updated_pack_selected_extension_solver_recompute_implementation_followup_required",
            "pass"
            if updated_pack_selected_extension_solver_recompute_implementation_followup_required
            else "reject",
            "updated-pack selected-extension solver-recompute implementation followup required",
            sign_base.truth(
                updated_pack_selected_extension_solver_recompute_implementation_followup_required
            ),
            "After the contract is fixed, the honest next blocker is actual implementation of the selected-extension solver-recompute path.",
        ),
        sign_base.row(
            "updated_pack_same_schema_selected_extension_solver_recompute_contract_replay_detected_now",
            "pass"
            if updated_pack_same_schema_selected_extension_solver_recompute_contract_replay_detected_now
            else "reject",
            "updated-pack same-schema selected-extension solver-recompute contract replay detected now",
            sign_base.truth(
                updated_pack_same_schema_selected_extension_solver_recompute_contract_replay_detected_now
            ),
            "False means this turn introduced a new concrete recomputation contract rather than replaying the already-closed backend-integrated blind rerun schema.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Farther hybrid continuation stays reserve-only while selected-extension solver recomputation has not been attempted.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_extension_label": backend_pack["selected_extension_label"],
        "q_theory_over_m0": float(backend_pack["retained_q_window"]["q_theory_over_m0"]),
        "blind_F_at_q_theory": float(backend_pack["blind_target_keys"]["blind_F_at_q_theory"]),
        "blind_alpha_at_q_theory": float(
            backend_pack["blind_target_keys"]["blind_alpha_at_q_theory"]
        ),
        "delta_alpha_sel_exact": float(
            backend_pack["blind_target_keys"]["delta_alpha_sel_exact"]
        ),
        "ell_scan_counts": backend_pack["ell_scan_counts"],
        "base_mode_counts": backend_pack["base_mode_counts"],
        "exact_ladder_row_count": int(backend_pack["exact_ladder_row_count"]),
        "comparison_row_count": int(backend_pack["comparison_row_count"]),
        "retained_anchor_match_preserved_now": retained_anchor_match_preserved_now,
        "exact_selected_extension_solver_recompute_input_pack_formula_available_now": exact_selected_extension_solver_recompute_input_pack_formula_available_now,
        "exact_selected_extension_solver_recompute_output_pack_formula_available_now": exact_selected_extension_solver_recompute_output_pack_formula_available_now,
        "exact_selected_extension_solver_recompute_contract_formula_available_now": exact_selected_extension_solver_recompute_contract_formula_available_now,
        "exact_selected_extension_solver_recompute_residual_discriminator_formula_available_now": exact_selected_extension_solver_recompute_residual_discriminator_formula_available_now,
        "exact_selected_extension_solver_recompute_front_runner_compatibility_theorem_available_now": exact_selected_extension_solver_recompute_front_runner_compatibility_theorem_available_now,
        "updated_pack_selected_extension_solver_recompute_implementation_followup_required": updated_pack_selected_extension_solver_recompute_implementation_followup_required,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_same_schema_selected_extension_solver_recompute_contract_replay_detected_now": updated_pack_same_schema_selected_extension_solver_recompute_contract_replay_detected_now,
        "pack_update_required_now": updated_pack_selected_extension_solver_recompute_implementation_followup_required,
        "selected_primary_completion_lane": "updated_pack_selected_extension_solver_recompute_implementation_audit",
        "selected_secondary_completion_lane": "updated_pack_selected_extension_solver_recompute_observable_rerun",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_solver_recompute_implementation_audit",
        "recommended_next_route_or_none": "8.7.56.5235",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_solver_recompute_implementation_gate",
        "selected_followup_route_or_none": "8.7.56.5239",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5233",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "selected_extension_gate": sign_base.display_path(SELECTED_EXTENSION_GATE),
                "backend_implementation_gate": sign_base.display_path(
                    BACKEND_IMPLEMENTATION_GATE
                ),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5235",
                "followup_route": "8.7.56.5239",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_selected_extension_solver_recompute_contract_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} selected-extension solver-recompute contract completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から contract audit を実行する。
if __name__ == "__main__":
    main()
