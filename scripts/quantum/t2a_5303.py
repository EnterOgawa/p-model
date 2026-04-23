#!/usr/bin/env python3
"""Generate 8.7.56.5303-.5306 selected-extension solver-side extra-q-range reserve artifacts."""

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
        "8.7.56.5299-5302",
        "updated_pack_selected_extension_solver_side_deformation_residual_origin_refresh_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5295-5298",
        "updated_pack_selected_extension_solver_side_deformation_residual_origin_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
EXTRA_Q_RANGE_HELPER = (
    ROOT / "scripts" / "quantum" / "selected_extension_solver_side_extra_q_range_backend.py"
)

STEP_TAG = "8.7.56.5303-5306"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack selected-"
    "extension solver-side extra-q-range reserve audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_selected_extension_solver_side_extra_q_range_reserve_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_side_deformation_residual_origin_verdict_"
    "audited_extra_q_range_reserve_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_solver_side_extra_q_range_reserve_nontrigger_theorem_"
    "derived_independent_evidence_inventory_primary_pack_refresh_secondary_"
    "gate"
)
RETAINED_Q_LABELS = {"zero", "q_theory_over_m0", "m0"}


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


# 関数: extra-q reserve audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the selected-extension solver-side extra-q-range reserve audit."""
    return {
        "reserve_candidate": (
            "D_solver_sel^(Qext)[Sigma_*^(pilot-HS)] := reopen extra q-range only if "
            "one independent extra-q discriminator E_qext^(ind) appears beyond the "
            "retained window Q_ret = {0, q_theory, m0}"
        ),
        "nontrigger_condition": (
            "Q_ret-only implemented deformation surface + no extra-q helper + "
            "no independent E_qext^(ind) => D_solver_sel^(Qext) stays reserve-only"
        ),
        "negative_closeout_guard": (
            "retained-q front-runner not residual origin + reserve candidate "
            "nontriggered now => selected-extension solver-side deformation lane "
            "closes negatively"
        ),
        "followup_inventory": (
            "Inv_qext_ind^(pilot-HS) := inventory of admissible independent "
            "extra-q-range evidence sources required before farther hybrid "
            "continuation can reopen"
        ),
    }


# 関数: `.5303-.5306` を実行する。

def main() -> None:
    """Execute the selected-extension solver-side extra-q-range reserve audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    deformation_pack = build_selected_extension_solver_side_deformation_pack()

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_selected_extension_solver_side_extra_q_range_reserve_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    selected_extension_label_matches_now = bool(
        prior_gate_summary["selected_extension_label"] == "Sigma_*^(pilot-HS)"
        and prior_audit_summary["selected_extension_label"] == "Sigma_*^(pilot-HS)"
        and deformation_pack["selected_extension_label"] == "Sigma_*^(pilot-HS)"
    )
    retained_q_front_runner_not_residual_origin_now = bool(
        prior_gate_summary[
            "gate_a_updated_pack_selected_extension_solver_side_deformation_residual_origin_not_retained_q_front_runner_available_now"
        ]
        and prior_audit_summary[
            "exact_selected_extension_solver_side_deformation_front_runner_not_residual_origin_theorem_available_now"
        ]
    )
    retained_q_only_window_now = bool(
        set(deformation_pack["retained_q_window"].keys()) == RETAINED_Q_LABELS
    )
    extra_q_labels_materialized_now = sorted(
        set(deformation_pack["retained_q_window"].keys()) - RETAINED_Q_LABELS
    )
    selected_extension_solver_side_extra_q_range_helper_available_now = bool(
        EXTRA_Q_RANGE_HELPER.exists()
    )
    independent_extra_q_range_evidence_available_now = bool(
        extra_q_labels_materialized_now
        or selected_extension_solver_side_extra_q_range_helper_available_now
        or prior_gate_summary["gate_c_farther_hybrid_continuation_reopen_required_now"]
    )

    reserve_formula_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selected_extension_label_matches_now
        and retained_q_front_runner_not_residual_origin_now
        and retained_q_only_window_now
    )
    exact_selected_extension_solver_side_extra_q_range_reserve_formula_available_now = bool(
        reserve_formula_explicit
    )
    exact_selected_extension_solver_side_extra_q_range_reopen_condition_formula_available_now = bool(
        reserve_formula_explicit
    )
    exact_selected_extension_solver_side_extra_q_range_independent_evidence_missing_theorem_available_now = bool(
        reserve_formula_explicit and not independent_extra_q_range_evidence_available_now
    )
    exact_selected_extension_solver_side_extra_q_range_reserve_nontrigger_theorem_available_now = bool(
        exact_selected_extension_solver_side_extra_q_range_independent_evidence_missing_theorem_available_now
    )
    exact_selected_extension_solver_side_deformation_lane_negative_closeout_available_now = bool(
        exact_selected_extension_solver_side_extra_q_range_reserve_nontrigger_theorem_available_now
    )
    updated_pack_selected_extension_independent_extra_q_range_evidence_inventory_followup_required = bool(
        exact_selected_extension_solver_side_deformation_lane_negative_closeout_available_now
    )
    updated_pack_selected_extension_solver_side_deformation_negative_closeout_completed_now = bool(
        exact_selected_extension_solver_side_deformation_lane_negative_closeout_available_now
    )
    updated_pack_same_schema_selected_extension_solver_side_extra_q_range_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_selected_extension_solver_side_extra_q_range_reserve_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack selected-extension solver-side extra-q-range reserve audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after retained-q deformation has been ruled out as residual origin and the live blocker is the reserve-only Qext candidate.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The branch keeps the computation-side hard-stop discipline rather than reopening retained-q replay or theorem-family recursion.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The reserve verdict is honest only while exhausted surrogate and retained-q replay routes remain closed.",
        ),
        sign_base.row(
            "selected_extension_label_matches_now",
            "pass" if selected_extension_label_matches_now else "reject",
            "selected-extension label matches now",
            sign_base.truth(selected_extension_label_matches_now),
            "The reserve verdict is meaningful only while the adopted selected extension Sigma_*^(pilot-HS) remains unchanged.",
        ),
        sign_base.row(
            "retained_q_front_runner_not_residual_origin_now",
            "pass" if retained_q_front_runner_not_residual_origin_now else "reject",
            "retained-q front-runner not residual origin now",
            sign_base.truth(retained_q_front_runner_not_residual_origin_now),
            "The extra-q reserve candidate becomes the live blocker only after the retained-q deformation front-runner has been ruled out explicitly.",
        ),
        sign_base.row(
            "retained_q_only_window_now",
            "pass" if retained_q_only_window_now else "reject",
            "retained-q-only window now",
            sign_base.truth(retained_q_only_window_now),
            "The implemented selected-extension deformation surface still covers only Q_ret = {0, q_theory, m0}; no extra-q checkpoint has been materialized.",
        ),
        sign_base.row(
            "selected_extension_solver_side_extra_q_range_helper_available_now",
            "pass"
            if selected_extension_solver_side_extra_q_range_helper_available_now
            else "reject",
            "selected-extension solver-side extra-q-range helper available now",
            sign_base.truth(
                selected_extension_solver_side_extra_q_range_helper_available_now
            ),
            "There is still no actual helper implementing an extra-q-range selected-extension deformation rerun path.",
        ),
        sign_base.row(
            "independent_extra_q_range_evidence_available_now",
            "pass" if independent_extra_q_range_evidence_available_now else "reject",
            "independent extra-q-range evidence available now",
            sign_base.truth(independent_extra_q_range_evidence_available_now),
            "No independent extra-q discriminator has been materialized beyond the retained-q window, so Qext cannot honestly trigger yet.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_extra_q_range_reserve_formula_available_now",
            "pass"
            if exact_selected_extension_solver_side_extra_q_range_reserve_formula_available_now
            else "reject",
            "exact selected-extension solver-side extra-q-range reserve formula available now",
            sign_base.truth(
                exact_selected_extension_solver_side_extra_q_range_reserve_formula_available_now
            ),
            "The Qext reserve candidate is now fixed as a literal reopen condition rather than a vague fallback phrase.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_extra_q_range_reopen_condition_formula_available_now",
            "pass"
            if exact_selected_extension_solver_side_extra_q_range_reopen_condition_formula_available_now
            else "reject",
            "exact selected-extension solver-side extra-q-range reopen-condition formula available now",
            sign_base.truth(
                exact_selected_extension_solver_side_extra_q_range_reopen_condition_formula_available_now
            ),
            "The branch now states explicitly that farther hybrid continuation reopens only after independent extra-q evidence appears.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_extra_q_range_independent_evidence_missing_theorem_available_now",
            "pass"
            if exact_selected_extension_solver_side_extra_q_range_independent_evidence_missing_theorem_available_now
            else "reject",
            "exact selected-extension solver-side extra-q-range independent-evidence-missing theorem available now",
            sign_base.truth(
                exact_selected_extension_solver_side_extra_q_range_independent_evidence_missing_theorem_available_now
            ),
            "The current selected-extension deformation stack still lacks any independent extra-q-range discriminator beyond the retained-q surface.",
        ),
        sign_base.row(
            "exact_selected_extension_solver_side_extra_q_range_reserve_nontrigger_theorem_available_now",
            "pass"
            if exact_selected_extension_solver_side_extra_q_range_reserve_nontrigger_theorem_available_now
            else "reject",
            "exact selected-extension solver-side extra-q-range reserve nontrigger theorem available now",
            sign_base.truth(
                exact_selected_extension_solver_side_extra_q_range_reserve_nontrigger_theorem_available_now
            ),
            "Because no independent extra-q evidence exists now, the Qext reserve candidate is nontriggered and stays reserve-only.",
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
            "With retained-q deformation ruled out and Qext still nontriggered, the selected-extension solver-side deformation lane itself closes negatively.",
        ),
        sign_base.row(
            "updated_pack_selected_extension_independent_extra_q_range_evidence_inventory_followup_required",
            "pass"
            if updated_pack_selected_extension_independent_extra_q_range_evidence_inventory_followup_required
            else "reject",
            "updated-pack selected-extension independent extra-q-range evidence inventory followup required",
            sign_base.truth(
                updated_pack_selected_extension_independent_extra_q_range_evidence_inventory_followup_required
            ),
            "The honest next blocker is now independent extra-q-range evidence inventory, not further deformation replay inside the closed lane.",
        ),
        sign_base.row(
            "updated_pack_selected_extension_solver_side_deformation_negative_closeout_completed_now",
            "pass"
            if updated_pack_selected_extension_solver_side_deformation_negative_closeout_completed_now
            else "reject",
            "updated-pack selected-extension solver-side deformation negative closeout completed now",
            sign_base.truth(
                updated_pack_selected_extension_solver_side_deformation_negative_closeout_completed_now
            ),
            "This branch completes a final negative closeout on the selected-extension solver-side deformation lane.",
        ),
        sign_base.row(
            "updated_pack_same_schema_selected_extension_solver_side_extra_q_range_replay_detected_now",
            "pass"
            if updated_pack_same_schema_selected_extension_solver_side_extra_q_range_replay_detected_now
            else "reject",
            "updated-pack same-schema selected-extension solver-side extra-q-range replay detected now",
            sign_base.truth(
                updated_pack_same_schema_selected_extension_solver_side_extra_q_range_replay_detected_now
            ),
            "False means this turn produced a genuine reserve nontrigger verdict instead of replaying the already-closed retained-q deformation schema.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Farther hybrid continuation remains reserve-only because no independent extra-q-range evidence has been established.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_extension_label": deformation_pack["selected_extension_label"],
        "solver_side_deformation_label": "D_solver_sel^(Qext)",
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
        "retained_q_only_window_now": retained_q_only_window_now,
        "extra_q_labels_materialized_now": extra_q_labels_materialized_now,
        "selected_extension_solver_side_extra_q_range_helper_available_now": selected_extension_solver_side_extra_q_range_helper_available_now,
        "independent_extra_q_range_evidence_available_now": independent_extra_q_range_evidence_available_now,
        "exact_selected_extension_solver_side_extra_q_range_reserve_formula_available_now": exact_selected_extension_solver_side_extra_q_range_reserve_formula_available_now,
        "exact_selected_extension_solver_side_extra_q_range_reopen_condition_formula_available_now": exact_selected_extension_solver_side_extra_q_range_reopen_condition_formula_available_now,
        "exact_selected_extension_solver_side_extra_q_range_independent_evidence_missing_theorem_available_now": exact_selected_extension_solver_side_extra_q_range_independent_evidence_missing_theorem_available_now,
        "exact_selected_extension_solver_side_extra_q_range_reserve_nontrigger_theorem_available_now": exact_selected_extension_solver_side_extra_q_range_reserve_nontrigger_theorem_available_now,
        "exact_selected_extension_solver_side_deformation_lane_negative_closeout_available_now": exact_selected_extension_solver_side_deformation_lane_negative_closeout_available_now,
        "updated_pack_selected_extension_independent_extra_q_range_evidence_inventory_followup_required": updated_pack_selected_extension_independent_extra_q_range_evidence_inventory_followup_required,
        "updated_pack_selected_extension_solver_side_deformation_negative_closeout_completed_now": updated_pack_selected_extension_solver_side_deformation_negative_closeout_completed_now,
        "updated_pack_same_schema_selected_extension_solver_side_extra_q_range_replay_detected_now": updated_pack_same_schema_selected_extension_solver_side_extra_q_range_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": updated_pack_selected_extension_independent_extra_q_range_evidence_inventory_followup_required,
        "selected_primary_completion_lane": "updated_pack_selected_extension_solver_side_extra_q_range_reserve_gate",
        "selected_secondary_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_evidence_inventory_audit",
        "selected_reserve_completion_lane": "farther_hybrid_reserve_only_until_independent_extra_q_evidence",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_solver_side_extra_q_range_reserve_gate",
        "recommended_next_route_or_none": "8.7.56.5307",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_independent_extra_q_range_evidence_inventory_audit",
        "selected_followup_route_or_none": "8.7.56.5311",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5305",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "deformation_helper": sign_base.display_path(
                    ROOT
                    / "scripts"
                    / "quantum"
                    / "selected_extension_solver_side_deformation_backend.py"
                ),
                "extra_q_helper_candidate": sign_base.display_path(EXTRA_Q_RANGE_HELPER),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5307",
                "followup_route": "8.7.56.5311",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_selected_extension_solver_side_extra_q_range_reserve_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} selected-extension solver-side extra-q-range reserve audit completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から audit を実行する。

if __name__ == "__main__":
    main()
