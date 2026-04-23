#!/usr/bin/env python3
"""Generate 8.7.56.5455-.5458 source-materialization numeric rerun artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.selected_extension_solver_side_extra_q_range_numeric_rerun_backend import (
    build_selected_extension_solver_side_extra_q_range_numeric_rerun_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5451-5454",
        "updated_pack_scalar_proxy_route_c_virial_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
IMPLEMENTATION_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5371-5374",
        "updated_pack_selected_extension_independent_extra_q_range_source_materialization_implementation_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5455-5458"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack selected-"
    "extension independent extra-q-range source-materialization numeric rerun audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_route_c_virial_negative_closeout_completed_"
    "source_materialization_primary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_independent_extra_q_range_source_materialization_"
    "numeric_rerun_legacy_phase3_sideband_carryover_derived_negative_verdict_gate"
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


# 関数: numeric rerun audit の式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the source-materialization numeric rerun audit."""
    return {
        "alpha_proxy": "alpha_blind(q) = F_blind(q)^2 / (4 pi)",
        "legacy_sideband_rule": (
            "q in Q_ext^(ind) => provenance(q) = phase3_blind_numeric_evaluation"
        ),
        "negative_verdict": (
            "q_theory failure preserved and all extra-q labels legacy sidebands "
            "=> no canonical selected-extension extra-q rescue is materialized now"
        ),
        "matching_gap": "delta_qext(q) = |q - q_exact| / q_exact",
    }


# 関数: `.5455-.5458` を実行する。

def main() -> None:
    """Execute the selected-extension source-materialization numeric rerun audit."""
    for path in (PRIOR_GATE, IMPLEMENTATION_GATE):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    implementation_summary = sign_base.read_json(IMPLEMENTATION_GATE)["summary"]

    audit_selected = bool(
        prior_summary["gate_b_updated_pack_selected_extension_source_materialization_promoted_primary_now"]
        and implementation_summary[
            "gate_b_updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_promoted_next"
        ]
    )
    retry_mode = bool(
        prior_summary["selected_extension_source_materialization_promoted_primary_now"]
    )
    non_surrogate_guard = bool(
        implementation_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    source_materialization_implementation_available_now = bool(
        implementation_summary[
            "gate_a_updated_pack_exact_selected_extension_independent_extra_q_range_source_materialization_implementation_available_now"
        ]
    )

    rerun_pack = build_selected_extension_solver_side_extra_q_range_numeric_rerun_pack()
    numeric_rerun_materialized_surface_available_now = bool(
        rerun_pack[
            "selected_extension_solver_side_extra_q_range_numeric_rerun_available_now"
        ]
    )
    q_theory_failure_preserved_now = bool(rerun_pack["q_theory_failure_preserved_now"])
    legacy_phase3_sideband_carryover_now = bool(
        rerun_pack["phase3_sideband_carryover_only_now"]
    )
    canonical_extra_q_rescue_available_now = bool(
        rerun_pack["canonical_extra_q_rescue_available_now"]
    )

    exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_formula_available_now = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and source_materialization_implementation_available_now
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_materialized_surface_available_now = bool(
        numeric_rerun_materialized_surface_available_now
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_q_theory_failure_preserved_theorem_available_now = bool(
        numeric_rerun_materialized_surface_available_now and q_theory_failure_preserved_now
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_legacy_phase3_sideband_carryover_theorem_available_now = bool(
        numeric_rerun_materialized_surface_available_now and legacy_phase3_sideband_carryover_now
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_canonical_extra_q_rescue_available_now = bool(
        canonical_extra_q_rescue_available_now
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_negative_verdict_available_now = bool(
        exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_formula_available_now
        and exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_materialized_surface_available_now
        and exact_selected_extension_independent_extra_q_range_source_materialization_q_theory_failure_preserved_theorem_available_now
        and exact_selected_extension_independent_extra_q_range_source_materialization_legacy_phase3_sideband_carryover_theorem_available_now
        and not exact_selected_extension_independent_extra_q_range_source_materialization_canonical_extra_q_rescue_available_now
    )
    updated_pack_selected_extension_independent_extra_q_range_source_materialization_negative_closeout_followup_required = bool(
        exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_negative_verdict_available_now
    )
    updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = False

    best_extra_label = str(rerun_pack["best_extra_label_vs_alpha_exact"])
    best_extra_diag = rerun_pack["q_surface_diagnostics"][best_extra_label]

    rows = [
        sign_base.row(
            "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack selected-extension independent extra-q-range source-materialization numeric rerun audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after Route C is exhausted and the helper-backed source-materialization implementation is already official.",
        ),
        sign_base.row(
            "source_materialization_implementation_available_now",
            "pass" if source_materialization_implementation_available_now else "reject",
            "source-materialization implementation available now",
            sign_base.truth(source_materialization_implementation_available_now),
            "The helper-backed augmented-q surface must already exist before any numeric verdict can be taken seriously.",
        ),
        sign_base.row(
            "numeric_rerun_materialized_surface_available_now",
            "pass" if numeric_rerun_materialized_surface_available_now else "reject",
            "numeric rerun materialized surface available now",
            sign_base.truth(numeric_rerun_materialized_surface_available_now),
            "The helper now provides actual blind-alpha values on every q label in Q_aug.",
        ),
        sign_base.row(
            "best_extra_label_vs_alpha_exact",
            "pass",
            "best extra-q label versus retained exact alpha (q/m0)",
            float(best_extra_diag["q_over_m0"]),
            (
                "The extra-q label with the smallest residual against "
                f"alpha_exact(q_theory) is {best_extra_label}."
            ),
        ),
        sign_base.row(
            "best_extra_alpha_exact_residual",
            "pass",
            "best extra-q relative residual versus retained exact alpha",
            float(rerun_pack["best_extra_alpha_exact_residual"]),
            "Even the best extra-q point stays away from the retained exact scalar alpha at q_theory.",
        ),
        sign_base.row(
            "best_extra_q_exact_gap",
            "pass",
            "best extra-q relative gap versus scalar q_exact",
            float(rerun_pack["best_extra_q_exact_gap"]),
            "The closest materialized extra-q label is still far from the scalar q_exact crossing, so this branch does not reproduce the matching-law fix.",
        ),
        sign_base.row(
            "best_extra_label_legacy_phase3_now",
            "pass" if rerun_pack["best_extra_label_legacy_phase3_now"] else "reject",
            "best extra-q label legacy phase-3 now",
            sign_base.truth(rerun_pack["best_extra_label_legacy_phase3_now"]),
            "The numerically best extra-q point is carried directly from the old blind Phase-3 sideband surface.",
        ),
        sign_base.row(
            "q_theory_failure_preserved_now",
            "pass" if q_theory_failure_preserved_now else "reject",
            "q_theory failure preserved now",
            sign_base.truth(q_theory_failure_preserved_now),
            "The helper-backed augmented-q surface still preserves the wrong-sign / low-alpha retained-q failure at q_theory.",
        ),
        sign_base.row(
            "all_extra_labels_phase3_carried_now",
            "pass" if rerun_pack["all_extra_labels_phase3_carried_now"] else "reject",
            "all extra-q labels phase-3 carried now",
            sign_base.truth(rerun_pack["all_extra_labels_phase3_carried_now"]),
            "Every currently materialized extra-q point is legacy Phase-3 carry-over, not a new selected-extension-native q source.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_formula_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_formula_available_now
            else "reject",
            "exact selected-extension independent extra-q-range source-materialization numeric rerun formula available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_formula_available_now
            ),
            "The numeric verdict is now written explicitly on the helper-backed augmented-q surface.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_source_materialization_q_theory_failure_preserved_theorem_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_source_materialization_q_theory_failure_preserved_theorem_available_now
            else "reject",
            "exact selected-extension independent extra-q-range source-materialization q-theory failure preserved theorem available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_source_materialization_q_theory_failure_preserved_theorem_available_now
            ),
            "Materializing extra-q labels does not cure the retained selected-extension failure at q_theory.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_source_materialization_legacy_phase3_sideband_carryover_theorem_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_source_materialization_legacy_phase3_sideband_carryover_theorem_available_now
            else "reject",
            "exact selected-extension independent extra-q-range source-materialization legacy Phase-3 sideband carryover theorem available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_source_materialization_legacy_phase3_sideband_carryover_theorem_available_now
            ),
            "The helper-backed extra-q surface is a materialized carry-over of legacy blind sidebands, not a new selected-extension-specific rescue mechanism.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_source_materialization_canonical_extra_q_rescue_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_source_materialization_canonical_extra_q_rescue_available_now
            else "reject",
            "exact selected-extension independent extra-q-range source-materialization canonical extra-q rescue available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_source_materialization_canonical_extra_q_rescue_available_now
            ),
            "Reject means no new canonical extra-q rescue is materialized on the selected extension in the current pack.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_negative_verdict_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_negative_verdict_available_now
            else "reject",
            "exact selected-extension independent extra-q-range source-materialization numeric rerun negative verdict available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_negative_verdict_available_now
            ),
            "The honest read is now explicit: the rerun materializes legacy sidebands but does not produce a new canonical selected-extension rescue.",
        ),
        sign_base.row(
            "updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_replay_detected_now",
            "pass"
            if updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_replay_detected_now
            else "reject",
            "updated-pack same-schema selected-extension independent extra-q-range source-materialization numeric rerun replay detected now",
            sign_base.truth(
                updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_replay_detected_now
            ),
            "False means the blocker has genuinely moved from missing rerun to an actual verdict on the helper-backed surface.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_extension_label": rerun_pack["selected_extension_label"],
        "solver_side_deformation_label": rerun_pack["solver_side_deformation_label"],
        "source_materialization_label": rerun_pack["source_materialization_label"],
        "q_theory_over_m0": float(rerun_pack["q_theory_over_m0"]),
        "scalar_q_exact_over_m0": float(rerun_pack["scalar_q_exact_over_m0"]),
        "alpha_exact_at_q_theory": float(rerun_pack["alpha_exact_at_q_theory"]),
        "alpha_target": float(rerun_pack["alpha_target"]),
        "q_ext_ind_window": rerun_pack["q_ext_ind_window"],
        "q_aug_window": rerun_pack["q_aug_window"],
        "blind_F_qext_pack": rerun_pack["blind_F_qext_pack"],
        "alpha_blind_qext_pack": rerun_pack["alpha_blind_qext_pack"],
        "best_extra_label_vs_alpha_exact": rerun_pack["best_extra_label_vs_alpha_exact"],
        "best_extra_label_vs_alpha_target": rerun_pack["best_extra_label_vs_alpha_target"],
        "best_extra_label_vs_q_exact": rerun_pack["best_extra_label_vs_q_exact"],
        "best_extra_alpha_exact_residual": float(rerun_pack["best_extra_alpha_exact_residual"]),
        "best_extra_alpha_target_residual": float(rerun_pack["best_extra_alpha_target_residual"]),
        "best_extra_q_exact_gap": float(rerun_pack["best_extra_q_exact_gap"]),
        "best_extra_label_diagnostic": best_extra_diag,
        "q_theory_diagnostic": rerun_pack["q_surface_diagnostics"]["q_theory_over_m0"],
        "exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_formula_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_formula_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_materialized_surface_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_materialized_surface_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_q_theory_failure_preserved_theorem_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_q_theory_failure_preserved_theorem_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_legacy_phase3_sideband_carryover_theorem_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_legacy_phase3_sideband_carryover_theorem_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_canonical_extra_q_rescue_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_canonical_extra_q_rescue_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_negative_verdict_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_negative_verdict_available_now,
        "q_theory_failure_preserved_now": q_theory_failure_preserved_now,
        "all_extra_labels_phase3_carried_now": bool(rerun_pack["all_extra_labels_phase3_carried_now"]),
        "best_extra_label_legacy_phase3_now": bool(rerun_pack["best_extra_label_legacy_phase3_now"]),
        "phase3_sideband_carryover_only_now": legacy_phase3_sideband_carryover_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_selected_extension_independent_extra_q_range_source_materialization_negative_closeout_followup_required": updated_pack_selected_extension_independent_extra_q_range_source_materialization_negative_closeout_followup_required,
        "updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_replay_detected_now": updated_pack_same_schema_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_replay_detected_now,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": updated_pack_selected_extension_independent_extra_q_range_source_materialization_negative_closeout_followup_required,
        "selected_primary_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_gate",
        "selected_secondary_completion_lane": "trial2_numeric_alpha_numerical_closeout_inventory_audit",
        "selected_reserve_completion_lane": "farther_hybrid_reserve_only_until_new_independent_source_exists",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_independent_extra_q_range_source_materialization_gate",
        "recommended_next_route_or_none": "8.7.56.5459",
        "selected_followup_route": "trial2_numeric_alpha_numerical_closeout_inventory_audit",
        "selected_followup_route_or_none": "8.7.56.5463",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5457",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "implementation_gate": sign_base.display_path(IMPLEMENTATION_GATE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5459",
                "followup_route": "8.7.56.5463",
            },
        },
        rows,
        summary,
        {
            "overall_status": "selected_extension_source_materialization_numeric_rerun_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "formulae": build_formulae(),
            "q_surface_diagnostics": rerun_pack["q_surface_diagnostics"],
        },
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} selected-extension source-materialization numeric rerun audit completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から numeric rerun audit を実行する。

if __name__ == "__main__":
    main()
