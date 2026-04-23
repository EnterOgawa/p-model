#!/usr/bin/env python3
"""Generate 8.7.56.5343-.5346 extra-q evidence negative closeout artifacts."""

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
PRIOR_HELPER = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5319-5322",
        "updated_pack_selected_extension_independent_extra_q_range_evidence_front_runner_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_PUBLIC = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5327-5330",
        "updated_pack_selected_extension_independent_extra_q_range_evidence_public_checkpoint_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_HYBRID = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5335-5338",
        "updated_pack_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5339-5342",
        "updated_pack_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5343-5346"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack selected-"
    "extension independent extra-q-range evidence negative closeout audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_independent_extra_q_range_evidence_hybrid_bridge_"
    "no_go_theorem_audited_negative_closeout_primary_hybrid_reserve_secondary_"
    "next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_independent_extra_q_range_evidence_negative_closeout_"
    "theorem_derived_source_materialization_inventory_primary_pack_refresh_"
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


# 関数: lane negative closeout の式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the lane negative-closeout audit."""
    return {
        "lane_negative_closeout": (
            "not E_qext^(helper), not E_qext^(pub), not E_qext^(hyb) "
            "=> no admissible materialized independent extra-q evidence source "
            "exists in the current selected-extension pack"
        ),
        "materialization_followup": (
            "next lane := inventory of actual source-materialization routes that "
            "could make one admissible independent extra-q evidence source real"
        ),
    }


# 関数: `.5343-.5346` を実行する。

def main() -> None:
    """Execute the selected-extension extra-q evidence negative-closeout audit."""
    for path in (PRIOR_HELPER, PRIOR_PUBLIC, PRIOR_HYBRID, PRIOR_GATE):
        sign_base.require(path)

    helper_summary = sign_base.read_json(PRIOR_HELPER)["summary"]
    public_summary = sign_base.read_json(PRIOR_PUBLIC)["summary"]
    hybrid_summary = sign_base.read_json(PRIOR_HYBRID)["summary"]
    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    helper_no_go = bool(
        helper_summary[
            "exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_no_go_theorem_available_now"
        ]
    )
    public_no_go = bool(
        public_summary[
            "exact_selected_extension_independent_extra_q_range_evidence_public_checkpoint_no_go_theorem_available_now"
        ]
    )
    hybrid_no_go = bool(
        hybrid_summary[
            "exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_no_go_theorem_available_now"
        ]
    )

    lane_negative_closeout_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and helper_no_go
        and public_no_go
        and hybrid_no_go
    )
    exact_selected_extension_independent_extra_q_range_evidence_lane_negative_closeout_available_now = bool(
        lane_negative_closeout_explicit
    )
    exact_selected_extension_independent_extra_q_range_source_materialization_lane_required_theorem_available_now = bool(
        lane_negative_closeout_explicit
    )
    exact_selected_extension_independent_extra_q_range_evidence_source_available_now = False
    updated_pack_selected_extension_independent_extra_q_range_source_materialization_inventory_followup_required = bool(
        exact_selected_extension_independent_extra_q_range_source_materialization_lane_required_theorem_available_now
    )
    updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_completed_now = bool(
        exact_selected_extension_independent_extra_q_range_evidence_lane_negative_closeout_available_now
    )
    updated_pack_same_schema_selected_extension_independent_extra_q_range_evidence_negative_closeout_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack selected-extension independent extra-q-range evidence negative-closeout audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after helper, public-checkpoint, and hybrid-bridge candidates are all closed negatively.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The branch summarizes the candidate stack theorem-first instead of inventing a fourth evidence candidate by recursion.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Lane negative closeout is honest only while reserve-only replay and same-tag candidate recursion remain closed.",
        ),
        sign_base.row(
            "helper_backed_no_go_available_now",
            "pass" if helper_no_go else "reject",
            "helper-backed no-go available now",
            sign_base.truth(helper_no_go),
            "The helper-backed candidate has already been rejected because no actual helper-backed extra-q materialization exists now.",
        ),
        sign_base.row(
            "public_checkpoint_no_go_available_now",
            "pass" if public_no_go else "reject",
            "public-checkpoint no-go available now",
            sign_base.truth(public_no_go),
            "The public-checkpoint candidate has already been rejected because no promoted public extra-q checkpoint pack exists now.",
        ),
        sign_base.row(
            "hybrid_bridge_no_go_available_now",
            "pass" if hybrid_no_go else "reject",
            "hybrid-bridge no-go available now",
            sign_base.truth(hybrid_no_go),
            "The hybrid-bridge candidate has already been rejected because farther hybrid continuation remains reserve-only.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_evidence_lane_negative_closeout_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_evidence_lane_negative_closeout_available_now
            else "reject",
            "exact selected-extension independent extra-q-range evidence lane negative closeout available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_evidence_lane_negative_closeout_available_now
            ),
            "The whole independent extra-q evidence lane now closes negatively because no admissible materialized source exists in the current pack.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_source_materialization_lane_required_theorem_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_source_materialization_lane_required_theorem_available_now
            else "reject",
            "exact selected-extension independent extra-q-range source-materialization lane required theorem available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_source_materialization_lane_required_theorem_available_now
            ),
            "The honest next blocker is no longer theorem-side candidate syntax, but actual source materialization routes that could make one admissible extra-q evidence source real.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_evidence_source_available_now",
            "pass" if exact_selected_extension_independent_extra_q_range_evidence_source_available_now else "reject",
            "exact selected-extension independent extra-q-range evidence source available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_evidence_source_available_now
            ),
            "Reject means the lane closes negatively without any admissible materialized evidence source.",
        ),
        sign_base.row(
            "updated_pack_selected_extension_independent_extra_q_range_source_materialization_inventory_followup_required",
            "pass"
            if updated_pack_selected_extension_independent_extra_q_range_source_materialization_inventory_followup_required
            else "reject",
            "updated-pack selected-extension independent extra-q-range source-materialization inventory followup required",
            sign_base.truth(
                updated_pack_selected_extension_independent_extra_q_range_source_materialization_inventory_followup_required
            ),
            "The honest next blocker is now source-materialization inventory, not another candidate-family replay.",
        ),
        sign_base.row(
            "updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_completed_now",
            "pass"
            if updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_completed_now
            else "reject",
            "updated-pack selected-extension independent extra-q-range evidence negative closeout completed now",
            sign_base.truth(
                updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_completed_now
            ),
            "This branch completes a final negative closeout on the independent extra-q evidence lane.",
        ),
        sign_base.row(
            "updated_pack_same_schema_selected_extension_independent_extra_q_range_evidence_negative_closeout_replay_detected_now",
            "pass"
            if updated_pack_same_schema_selected_extension_independent_extra_q_range_evidence_negative_closeout_replay_detected_now
            else "reject",
            "updated-pack same-schema selected-extension independent extra-q-range evidence negative-closeout replay detected now",
            sign_base.truth(
                updated_pack_same_schema_selected_extension_independent_extra_q_range_evidence_negative_closeout_replay_detected_now
            ),
            "False means this turn summarizes the whole lane honestly instead of inventing another recursive candidate layer.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_extension_label": helper_summary["selected_extension_label"],
        "solver_side_deformation_label": helper_summary["solver_side_deformation_label"],
        "q_theory_over_m0": float(helper_summary["q_theory_over_m0"]),
        "blind_F_deform_at_q_theory": float(
            helper_summary["blind_F_deform_at_q_theory"]
        ),
        "blind_alpha_deform_at_q_theory": float(
            helper_summary["blind_alpha_deform_at_q_theory"]
        ),
        "delta_alpha_sel_deform_exact": float(
            helper_summary["delta_alpha_sel_deform_exact"]
        ),
        "relative_exact_residual_deform": float(
            helper_summary["relative_exact_residual_deform"]
        ),
        "exact_selected_extension_independent_extra_q_range_evidence_lane_negative_closeout_available_now": exact_selected_extension_independent_extra_q_range_evidence_lane_negative_closeout_available_now,
        "exact_selected_extension_independent_extra_q_range_source_materialization_lane_required_theorem_available_now": exact_selected_extension_independent_extra_q_range_source_materialization_lane_required_theorem_available_now,
        "exact_selected_extension_independent_extra_q_range_evidence_source_available_now": exact_selected_extension_independent_extra_q_range_evidence_source_available_now,
        "updated_pack_selected_extension_independent_extra_q_range_source_materialization_inventory_followup_required": updated_pack_selected_extension_independent_extra_q_range_source_materialization_inventory_followup_required,
        "updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_completed_now": updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_completed_now,
        "updated_pack_same_schema_selected_extension_independent_extra_q_range_evidence_negative_closeout_replay_detected_now": updated_pack_same_schema_selected_extension_independent_extra_q_range_evidence_negative_closeout_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": updated_pack_selected_extension_independent_extra_q_range_source_materialization_inventory_followup_required,
        "selected_primary_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_gate",
        "selected_secondary_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_inventory_audit",
        "selected_reserve_completion_lane": "farther_hybrid_reserve_only_until_independent_extra_q_evidence_promoted",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_gate",
        "recommended_next_route_or_none": "8.7.56.5347",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_independent_extra_q_range_source_materialization_inventory_audit",
        "selected_followup_route_or_none": "8.7.56.5351",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5345",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_helper": sign_base.display_path(PRIOR_HELPER),
                "prior_public": sign_base.display_path(PRIOR_PUBLIC),
                "prior_hybrid": sign_base.display_path(PRIOR_HYBRID),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5347",
                "followup_route": "8.7.56.5351",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_selected_extension_independent_extra_q_negative_closeout_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} selected-extension independent extra-q evidence negative closeout audit completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から audit を実行する。

if __name__ == "__main__":
    main()
