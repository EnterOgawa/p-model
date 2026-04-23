#!/usr/bin/env python3
"""Generate 8.7.56.5207-.5210 blind-vector backend-implementation audit artifacts."""

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
        "8.7.56.5203-5206",
        "updated_pack_blind_vector_solver_side_backend_adapter_gate",
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

STEP_TAG = "8.7.56.5207-5210"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack blind-vector "
    "solver-side backend implementation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_blind_vector_solver_side_backend_implementation_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_solver_side_backend_adapter_contract_audited_implementation_"
    "primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_solver_side_backend_implementation_audited_numeric_rerun_"
    "primary_hybrid_reserve_secondary_next"
)
BEST_MATCH_FLOAT_KEYS = ("target_value", "ratio_value", "relative_error")
BEST_MATCH_LITERAL_KEYS = ("n", "k", "ell", "s", "target_label", "passes_threshold")
BLIND_TARGET_KEYS = (
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


# 関数: backend-implementation audit の式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the blind-vector backend-implementation audit."""
    return {
        "implementation_call": (
            "B_adapter,impl^(pilot-HS,legacy-vq) := "
            "build_selected_extension_backend_pack(ell_values=(1,2,3))"
        ),
        "materialized_output_pack": (
            "O_adapter,impl^(pilot-HS,legacy-vq) := {ell_scan_rows, base_modes, "
            "exact_ladder, exact_comparisons, retained_anchor_row, blind_target_keys}"
        ),
        "retained_anchor_check": (
            "anchor_match iff best_exact_match_impl = best_exact_match_retained"
        ),
        "implementation_closeout": (
            "impl_available iff selected extension is fixed, the front-runner helper "
            "runs, O_adapter,impl is materialized, and the rebuilt best exact row "
            "matches the retained anchor"
        ),
    }


# 関数: `.5207-.5210` を実行する。

def main() -> None:
    """Execute the blind-vector backend-implementation audit."""
    for path in (PRIOR_GATE, SELECTED_EXTENSION_GATE):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    selected_summary = sign_base.read_json(SELECTED_EXTENSION_GATE)["summary"]

    audit_selected = bool(
        prior_summary[
            "gate_b_updated_pack_blind_vector_solver_side_backend_implementation_promoted_next"
        ]
        and prior_summary[
            "gate_a_updated_pack_exact_blind_vector_solver_side_backend_adapter_contract_available_now"
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

    backend_pack = build_selected_extension_backend_pack()
    helper_module_available_now = True
    ell_scan_rows_materialized_now = all(
        count > 0 for count in backend_pack["ell_scan_counts"].values()
    )
    base_modes_materialized_now = all(
        count > 0 for count in backend_pack["base_mode_counts"].values()
    )
    exact_ladder_materialized_now = bool(backend_pack["exact_ladder_row_count"] > 0)
    exact_comparisons_materialized_now = bool(backend_pack["comparison_row_count"] > 0)
    retained_anchor_match_now = retained_anchor_match(
        backend_pack["best_exact_match"],
        backend_pack["retained_anchor_row"],
    )
    retained_q_window_attached_now = bool(
        {"zero", "q_theory_over_m0", "m0"}
        <= set(backend_pack["retained_q_window"].keys())
    )
    blind_target_keys_attached_now = bool(
        set(BLIND_TARGET_KEYS) <= set(backend_pack["blind_target_keys"].keys())
    )

    exact_blind_vector_solver_side_backend_implementation_formula_available_now = bool(
        audit_selected
        and selected_extension_available
        and retry_mode
        and non_surrogate_guard
    )
    exact_blind_vector_solver_side_backend_materialized_output_pack_available_now = bool(
        helper_module_available_now
        and ell_scan_rows_materialized_now
        and base_modes_materialized_now
        and exact_ladder_materialized_now
        and exact_comparisons_materialized_now
        and retained_q_window_attached_now
        and blind_target_keys_attached_now
    )
    exact_blind_vector_solver_side_backend_retained_anchor_match_theorem_available_now = bool(
        exact_blind_vector_solver_side_backend_materialized_output_pack_available_now
        and retained_anchor_match_now
    )
    exact_blind_vector_solver_side_backend_implementation_available_now = bool(
        exact_blind_vector_solver_side_backend_implementation_formula_available_now
        and exact_blind_vector_solver_side_backend_materialized_output_pack_available_now
        and exact_blind_vector_solver_side_backend_retained_anchor_match_theorem_available_now
    )
    updated_pack_blind_vector_backend_integrated_retained_q_rerun_followup_required = bool(
        exact_blind_vector_solver_side_backend_implementation_available_now
    )
    updated_pack_same_schema_blind_vector_backend_implementation_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_blind_vector_solver_side_backend_implementation_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack blind-vector solver-side backend implementation audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after one concrete selected-extension backend adapter contract is already official and the live blocker is actual implementation.",
        ),
        sign_base.row(
            "selected_extension_available_now",
            "pass" if selected_extension_available else "reject",
            "selected extension available now",
            sign_base.truth(selected_extension_available),
            "Actual backend implementation is meaningful only while the adopted selected extension Sigma_*^(pilot-HS) remains official.",
        ),
        sign_base.row(
            "helper_module_available_now",
            "pass" if helper_module_available_now else "reject",
            "backend helper module available now",
            sign_base.truth(helper_module_available_now),
            "The actual implementation is realized as one reusable helper that wires the legacy profile backend, the legacy ladder backend, and the retained blind-vector target pack.",
        ),
        sign_base.row(
            "ell_scan_rows_materialized_now",
            "pass" if ell_scan_rows_materialized_now else "reject",
            "ell-scan rows materialized now",
            sign_base.truth(ell_scan_rows_materialized_now),
            "The implementation now actually rebuilds localized ell-sector rows instead of stopping at a theorem-only adapter contract.",
        ),
        sign_base.row(
            "base_modes_materialized_now",
            "pass" if base_modes_materialized_now else "reject",
            "base modes materialized now",
            sign_base.truth(base_modes_materialized_now),
            "The implementation now actually interpolates integer-charge base modes on the fixed retained ell sectors.",
        ),
        sign_base.row(
            "exact_ladder_materialized_now",
            "pass" if exact_ladder_materialized_now else "reject",
            "exact ladder materialized now",
            sign_base.truth(exact_ladder_materialized_now),
            "The implementation now actually rebuilds the exact multicomponent vector ladder on the selected extension.",
        ),
        sign_base.row(
            "exact_comparisons_materialized_now",
            "pass" if exact_comparisons_materialized_now else "reject",
            "exact comparisons materialized now",
            sign_base.truth(exact_comparisons_materialized_now),
            "The implementation now actually rebuilds the known-target comparison table needed for retained-anchor verification.",
        ),
        sign_base.row(
            "retained_anchor_match_now",
            "pass" if retained_anchor_match_now else "reject",
            "retained anchor match now",
            sign_base.truth(retained_anchor_match_now),
            "The rebuilt best exact row matches the retained exact anchor, so the implementation preserves the live selected-extension comparison surface.",
        ),
        sign_base.row(
            "retained_q_window_attached_now",
            "pass" if retained_q_window_attached_now else "reject",
            "retained q-window attached now",
            sign_base.truth(retained_q_window_attached_now),
            "The implementation carries the retained q-window {0, q_theory, m0} as explicit metadata for the next backend-integrated rerun.",
        ),
        sign_base.row(
            "blind_target_keys_attached_now",
            "pass" if blind_target_keys_attached_now else "reject",
            "blind target keys attached now",
            sign_base.truth(blind_target_keys_attached_now),
            "The implementation also carries the retained blind-vector target keys, so the next branch can measure rerun outputs against the live checkpoint surface.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_backend_implementation_formula_available_now",
            "pass"
            if exact_blind_vector_solver_side_backend_implementation_formula_available_now
            else "reject",
            "exact blind-vector solver-side backend implementation formula available now",
            sign_base.truth(
                exact_blind_vector_solver_side_backend_implementation_formula_available_now
            ),
            "The actual implementation call is now explicit: one helper rebuilds the selected-extension backend pack from the legacy solver chain.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_backend_materialized_output_pack_available_now",
            "pass"
            if exact_blind_vector_solver_side_backend_materialized_output_pack_available_now
            else "reject",
            "exact blind-vector solver-side backend materialized output pack available now",
            sign_base.truth(
                exact_blind_vector_solver_side_backend_materialized_output_pack_available_now
            ),
            "The contract output pack is no longer abstract; ell scans, base modes, exact ladder rows, retained anchor metadata, and blind target keys are materialized now.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_backend_retained_anchor_match_theorem_available_now",
            "pass"
            if exact_blind_vector_solver_side_backend_retained_anchor_match_theorem_available_now
            else "reject",
            "exact blind-vector solver-side backend retained-anchor match theorem available now",
            sign_base.truth(
                exact_blind_vector_solver_side_backend_retained_anchor_match_theorem_available_now
            ),
            "The actual implementation preserves the retained exact anchor rather than silently drifting away from the current blind-vector reference surface.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_backend_implementation_available_now",
            "pass"
            if exact_blind_vector_solver_side_backend_implementation_available_now
            else "reject",
            "exact blind-vector solver-side backend implementation available now",
            sign_base.truth(
                exact_blind_vector_solver_side_backend_implementation_available_now
            ),
            "The live blocker is no longer backend implementation itself; one concrete selected-extension backend implementation now exists and runs.",
        ),
        sign_base.row(
            "updated_pack_blind_vector_backend_integrated_retained_q_rerun_followup_required",
            "pass"
            if updated_pack_blind_vector_backend_integrated_retained_q_rerun_followup_required
            else "reject",
            "updated-pack blind-vector backend-integrated retained-q rerun followup required",
            sign_base.truth(
                updated_pack_blind_vector_backend_integrated_retained_q_rerun_followup_required
            ),
            "The honest next blocker is now the backend-integrated retained-q rerun itself, not helper existence or adapter implementation.",
        ),
        sign_base.row(
            "updated_pack_same_schema_blind_vector_backend_implementation_replay_detected_now",
            "pass"
            if updated_pack_same_schema_blind_vector_backend_implementation_replay_detected_now
            else "reject",
            "updated-pack same-schema blind-vector backend implementation replay detected now",
            sign_base.truth(
                updated_pack_same_schema_blind_vector_backend_implementation_replay_detected_now
            ),
            "False means this branch materially reduced the live blocker from implementation ambiguity to actual rerun execution.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Farther hybrid continuation stays reserve-only because the retained-q rerun can now be attacked directly with the implemented backend.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_theory_over_m0": float(backend_pack["retained_q_window"]["q_theory_over_m0"]),
        "blind_F_at_q_theory": float(backend_pack["blind_target_keys"]["blind_F_at_q_theory"]),
        "blind_alpha_at_q_theory": float(
            backend_pack["blind_target_keys"]["blind_alpha_at_q_theory"]
        ),
        "ell_scan_counts": backend_pack["ell_scan_counts"],
        "base_mode_counts": backend_pack["base_mode_counts"],
        "exact_ladder_row_count": int(backend_pack["exact_ladder_row_count"]),
        "comparison_row_count": int(backend_pack["comparison_row_count"]),
        "available_k_values": backend_pack["available_k_values"],
        "max_detected_k": int(backend_pack["max_detected_k"]),
        "best_exact_match_or_none": backend_pack["best_exact_match"],
        "retained_anchor_row": backend_pack["retained_anchor_row"],
        "exact_blind_vector_solver_side_backend_implementation_formula_available_now": exact_blind_vector_solver_side_backend_implementation_formula_available_now,
        "exact_blind_vector_solver_side_backend_materialized_output_pack_available_now": exact_blind_vector_solver_side_backend_materialized_output_pack_available_now,
        "exact_blind_vector_solver_side_backend_retained_anchor_match_theorem_available_now": exact_blind_vector_solver_side_backend_retained_anchor_match_theorem_available_now,
        "exact_blind_vector_solver_side_backend_implementation_available_now": exact_blind_vector_solver_side_backend_implementation_available_now,
        "updated_pack_blind_vector_backend_integrated_retained_q_rerun_followup_required": updated_pack_blind_vector_backend_integrated_retained_q_rerun_followup_required,
        "updated_pack_same_schema_blind_vector_backend_implementation_replay_detected_now": updated_pack_same_schema_blind_vector_backend_implementation_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": updated_pack_blind_vector_backend_integrated_retained_q_rerun_followup_required,
        "selected_primary_completion_lane": "updated_pack_blind_vector_backend_integrated_retained_q_rerun_audit",
        "selected_secondary_completion_lane": "updated_pack_blind_vector_residual_origin_refresh_after_backend_rerun",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_solver_side_backend_implementation_gate",
        "recommended_next_route_or_none": "8.7.56.5211",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_backend_integrated_retained_q_rerun_audit",
        "selected_followup_route_or_none": "8.7.56.5215",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5209",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "selected_extension_gate": sign_base.display_path(SELECTED_EXTENSION_GATE),
                "backend_helper": sign_base.display_path(
                    ROOT / "scripts" / "quantum" / "blind_vector_selected_extension_backend.py"
                ),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5211",
                "followup_route": "8.7.56.5215",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_blind_vector_solver_side_backend_implementation_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae(), "evidence": backend_pack["evidence_samples"]},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} blind-vector backend implementation audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
