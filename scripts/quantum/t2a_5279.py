#!/usr/bin/env python3
"""Generate 8.7.56.5279-.5282 selected-extension solver-side deformation implementation artifacts."""

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
from scripts.quantum.selected_extension_solver_side_deformation_backend import (
    build_selected_extension_solver_side_deformation_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5275-5278",
        "updated_pack_selected_extension_solver_side_deformation_front_runner_gate",
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

STEP_TAG = "8.7.56.5279-5282"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack selected-"
    "extension solver-side deformation front-runner implementation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_selected_extension_solver_side_deformation_front_runner_implementation_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_side_deformation_front_runner_audited_"
    "implementation_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_side_deformation_implementation_derived_numeric_"
    "rerun_primary_pack_refresh_secondary_gate"
)
BEST_MATCH_FLOAT_KEYS = ("target_value", "ratio_value", "relative_error")
BEST_MATCH_LITERAL_KEYS = ("n", "k", "ell", "s", "target_label", "passes_threshold")


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


# 関数: rebuilt best match が retained anchor と一致するかを判定する。

def retained_anchor_match(best_match: dict, retained_anchor: dict) -> bool:
    """Return whether the rebuilt best exact row matches the retained anchor row."""
    for key in BEST_MATCH_LITERAL_KEYS:
        if best_match.get(key) != retained_anchor.get(key):
            return False

    for key in BEST_MATCH_FLOAT_KEYS:
        if not math.isclose(
            float(best_match.get(key, 0.0)),
            float(retained_anchor.get(key, 0.0)),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            return False

    return True


# 関数: implementation audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the selected-extension solver-side deformation implementation audit."""
    return {
        "implementation_call": (
            "C_deform_sel,impl^(pilot-HS) := "
            "build_selected_extension_solver_side_deformation_pack(ell_values=(1,2,3))"
        ),
        "materialized_output_pack": (
            "O_deform_sel,impl^(pilot-HS) := {K_eff^(pilot-HS,deform)[Q_ret], "
            "Z_eff^(pilot-HS,deform,T)[Q_ret], F_blind^(pilot-HS,deform)[Q_ret], "
            "alpha_blind^(pilot-HS,deform)(q_theory), "
            "delta_alpha_sel^(pilot-HS,deform)}"
        ),
        "recompute_surface_check": (
            "preserves_recompute_surface iff O_deform_sel,impl matches "
            "O_recomp_sel^(pilot-HS) on Q_ret"
        ),
        "retained_anchor_check": (
            "anchor_match iff best_exact_match_impl = retained_anchor_row"
        ),
    }


# 関数: `.5279-.5282` を実行する。

def main() -> None:
    """Execute the selected-extension solver-side deformation front-runner implementation audit."""
    for path in (PRIOR_GATE, SELECTED_EXTENSION_GATE):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    selected_summary = sign_base.read_json(SELECTED_EXTENSION_GATE)["summary"]

    audit_selected = bool(
        prior_summary[
            "gate_b_updated_pack_selected_extension_solver_side_deformation_implementation_promoted_next"
        ]
        and prior_summary[
            "gate_a_updated_pack_exact_selected_extension_solver_side_deformation_front_runner_contract_available_now"
        ]
    )
    selected_extension_available = bool(
        selected_summary[
            "gate_a_updated_pack_exact_external_rule_selector_selected_extension_available_now"
        ]
        and selected_summary[
            "exact_external_rule_selector_selected_extension_available_now"
        ]
    )
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )

    deformation_pack = build_selected_extension_solver_side_deformation_pack()
    helper_module_available_now = True
    k_eff_deform_materialized_now = bool(
        {"zero", "q_theory_over_m0", "m0"}
        <= set(deformation_pack["K_eff_deform_transverse_projector_pack"].keys())
    )
    z_eff_deform_materialized_now = bool(
        {"zero", "q_theory_over_m0", "m0"}
        <= set(deformation_pack["Z_eff_deform_transverse_scalar_pack"].keys())
    )
    f_blind_deform_materialized_now = bool(
        {"zero", "q_theory_over_m0", "m0"}
        <= set(deformation_pack["F_blind_deform_pack"].keys())
    )
    alpha_blind_deform_available_now = bool(
        "alpha_blind_deform_at_q_theory" in deformation_pack
    )
    delta_alpha_sel_deform_available_now = bool(
        "delta_alpha_sel_deform_exact" in deformation_pack
    )
    retained_anchor_match_now = retained_anchor_match(
        deformation_pack["best_exact_match_or_none"],
        deformation_pack["retained_anchor_row"],
    )
    preserves_recompute_surface_now = bool(
        deformation_pack["preserves_recompute_surface_now"]
    )

    exact_selected_extension_solver_side_deformation_implementation_formula_available_now = bool(
        audit_selected
        and selected_extension_available
        and retry_mode
        and non_surrogate_guard
    )
    exact_selected_extension_solver_side_deformation_materialized_output_pack_available_now = bool(
        helper_module_available_now
        and k_eff_deform_materialized_now
        and z_eff_deform_materialized_now
        and f_blind_deform_materialized_now
        and alpha_blind_deform_available_now
        and delta_alpha_sel_deform_available_now
    )
    exact_selected_extension_solver_side_deformation_recompute_surface_preservation_theorem_available_now = bool(
        exact_selected_extension_solver_side_deformation_materialized_output_pack_available_now
        and preserves_recompute_surface_now
    )
    exact_selected_extension_solver_side_deformation_retained_anchor_match_theorem_available_now = bool(
        exact_selected_extension_solver_side_deformation_materialized_output_pack_available_now
        and retained_anchor_match_now
    )
    exact_selected_extension_solver_side_deformation_implementation_available_now = bool(
        exact_selected_extension_solver_side_deformation_implementation_formula_available_now
        and exact_selected_extension_solver_side_deformation_materialized_output_pack_available_now
        and exact_selected_extension_solver_side_deformation_recompute_surface_preservation_theorem_available_now
        and exact_selected_extension_solver_side_deformation_retained_anchor_match_theorem_available_now
    )
    updated_pack_selected_extension_solver_side_deformation_numeric_rerun_followup_required = bool(
        exact_selected_extension_solver_side_deformation_implementation_available_now
    )
    updated_pack_same_schema_selected_extension_solver_side_deformation_implementation_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_selected_extension_solver_side_deformation_implementation_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack selected-extension solver-side deformation implementation audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after one literal deformation contract is already official and the live blocker is actual implementation.",
        ),
        sign_base.row(
            "selected_extension_available_now",
            "pass" if selected_extension_available else "reject",
            "selected extension available now",
            sign_base.truth(selected_extension_available),
            "Actual deformation implementation is meaningful only while the adopted selected extension Sigma_*^(pilot-HS) remains official.",
        ),
        sign_base.row(
            "helper_module_available_now",
            "pass" if helper_module_available_now else "reject",
            "solver-side deformation helper module available now",
            sign_base.truth(helper_module_available_now),
            "The actual implementation is realized as one reusable helper that lifts the retained recompute pack into the deformation-labeled output pack.",
        ),
        sign_base.row(
            "k_eff_deform_materialized_now",
            "pass" if k_eff_deform_materialized_now else "reject",
            "K_eff deformation pack materialized now",
            sign_base.truth(k_eff_deform_materialized_now),
            "The implementation now materializes one retained-q deformation effective-kernel pack instead of stopping at theorem-only contract syntax.",
        ),
        sign_base.row(
            "z_eff_deform_materialized_now",
            "pass" if z_eff_deform_materialized_now else "reject",
            "Z_eff deformation transverse-scalar pack materialized now",
            sign_base.truth(z_eff_deform_materialized_now),
            "The implementation now materializes the retained-q deformation scalarization needed by the live lane.",
        ),
        sign_base.row(
            "f_blind_deform_materialized_now",
            "pass" if f_blind_deform_materialized_now else "reject",
            "F_blind deformation pack materialized now",
            sign_base.truth(f_blind_deform_materialized_now),
            "The implementation now exposes the retained-q blind form-factor pack directly on the deformation surface.",
        ),
        sign_base.row(
            "alpha_blind_deform_available_now",
            "pass" if alpha_blind_deform_available_now else "reject",
            "alpha_blind deformation available now",
            sign_base.truth(alpha_blind_deform_available_now),
            "The implementation now materializes the q_theory blind alpha needed by the deformation residual discriminator.",
        ),
        sign_base.row(
            "delta_alpha_sel_deform_available_now",
            "pass" if delta_alpha_sel_deform_available_now else "reject",
            "delta alpha deformation available now",
            sign_base.truth(delta_alpha_sel_deform_available_now),
            "The implementation now materializes the selected-extension deformation delta alpha against the retained exact scalar target.",
        ),
        sign_base.row(
            "preserves_recompute_surface_now",
            "pass" if preserves_recompute_surface_now else "reject",
            "preserves recompute surface now",
            sign_base.truth(preserves_recompute_surface_now),
            "The first implementation keeps the retained recompute checkpoint surface explicit instead of silently changing the live selected-extension baseline.",
        ),
        sign_base.row(
            "retained_anchor_match_now",
            "pass" if retained_anchor_match_now else "reject",
            "retained anchor match now",
            sign_base.truth(retained_anchor_match_now),
            "The deformation helper preserves the retained exact anchor row, so implementation does not silently drift from the live selected-extension comparison surface.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_implementation_formula_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_implementation_formula_available_now
            else "reject",
            "exact selected-extension solver-side deformation implementation formula available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_implementation_formula_available_now
            ),
            "The actual implementation call is now explicit: one helper materializes the retained-q deformation pack on Sigma_*^(pilot-HS).",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_materialized_output_pack_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_materialized_output_pack_available_now
            else "reject",
            "exact selected-extension solver-side deformation materialized output pack available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_materialized_output_pack_available_now
            ),
            "The contract output pack is no longer abstract; K_eff, Z_eff, F_blind, alpha_blind, and delta alpha are materialized now.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_recompute_surface_preservation_theorem_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_recompute_surface_preservation_theorem_available_now
            else "reject",
            "exact selected-extension solver-side deformation recompute-surface preservation theorem available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_recompute_surface_preservation_theorem_available_now
            ),
            "The implementation preserves the already closed retained-q recompute surface rather than inventing a new hidden baseline.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_retained_anchor_match_theorem_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_retained_anchor_match_theorem_available_now
            else "reject",
            "exact selected-extension solver-side deformation retained-anchor match theorem available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_retained_anchor_match_theorem_available_now
            ),
            "The actual implementation preserves the retained exact anchor row needed by the live comparison surface.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_deformation_implementation_available_now",
            "pass"
            if exact_selected_extension_solver_side_deformation_implementation_available_now
            else "reject",
            "exact selected-extension solver-side deformation implementation available now",
            sign_base.truth(
                exact_selected_extension_solver_side_deformation_implementation_available_now
            ),
            "One helper-backed deformation implementation is now available on the selected extension.",
        ),
        sign_base.row(
            "updated_pack_same_schema_selected_extension_solver_side_deformation_implementation_replay_detected_now",
            "pass"
            if updated_pack_same_schema_selected_extension_solver_side_deformation_implementation_replay_detected_now
            else "reject",
            "updated-pack same-schema selected-extension solver-side deformation implementation replay detected now",
            sign_base.truth(
                updated_pack_same_schema_selected_extension_solver_side_deformation_implementation_replay_detected_now
            ),
            "False means the live blocker has genuinely moved from contract syntax to actual implementation.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_extension_label": deformation_pack["selected_extension_label"],
        "solver_side_deformation_label": deformation_pack[
            "solver_side_deformation_label"
        ],
        "q_theory_over_m0": float(deformation_pack["retained_q_window"]["q_theory_over_m0"]),
        "blind_F_deform_at_q_theory": float(
            deformation_pack["F_blind_deform_pack"]["q_theory_over_m0"]
        ),
        "blind_alpha_deform_at_q_theory": float(
            deformation_pack["alpha_blind_deform_at_q_theory"]
        ),
        "delta_alpha_sel_deform_exact": float(
            deformation_pack["delta_alpha_sel_deform_exact"]
        ),
        "relative_exact_residual_deform": float(
            deformation_pack["relative_exact_residual_deform"]
        ),
        "exact_selected_extension_solver_side_deformation_implementation_formula_available_now": exact_selected_extension_solver_side_deformation_implementation_formula_available_now,
        "exact_selected_extension_solver_side_deformation_materialized_output_pack_available_now": exact_selected_extension_solver_side_deformation_materialized_output_pack_available_now,
        "exact_selected_extension_solver_side_deformation_recompute_surface_preservation_theorem_available_now": exact_selected_extension_solver_side_deformation_recompute_surface_preservation_theorem_available_now,
        "exact_selected_extension_solver_side_deformation_retained_anchor_match_theorem_available_now": exact_selected_extension_solver_side_deformation_retained_anchor_match_theorem_available_now,
        "exact_selected_extension_solver_side_deformation_implementation_available_now": exact_selected_extension_solver_side_deformation_implementation_available_now,
        "updated_pack_selected_extension_solver_side_deformation_numeric_rerun_followup_required": updated_pack_selected_extension_solver_side_deformation_numeric_rerun_followup_required,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_same_schema_selected_extension_solver_side_deformation_implementation_replay_detected_now": updated_pack_same_schema_selected_extension_solver_side_deformation_implementation_replay_detected_now,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": updated_pack_selected_extension_solver_side_deformation_numeric_rerun_followup_required,
        "selected_primary_completion_lane": "updated_pack_selected_extension_solver_side_deformation_front_runner_implementation_gate",
        "selected_secondary_completion_lane": "updated_pack_selected_extension_solver_side_deformation_numeric_rerun_audit",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_solver_side_deformation_front_runner_implementation_gate",
        "recommended_next_route_or_none": "8.7.56.5283",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_solver_side_deformation_numeric_rerun_audit",
        "selected_followup_route_or_none": "8.7.56.5287",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5281",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "selected_extension_gate": sign_base.display_path(SELECTED_EXTENSION_GATE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5283",
                "followup_route": "8.7.56.5287",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_selected_extension_solver_side_deformation_implementation_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} selected-extension solver-side deformation implementation audit completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から implementation audit を実行する。

if __name__ == "__main__":
    main()
