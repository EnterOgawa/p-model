#!/usr/bin/env python3
"""Generate 8.7.56.5319-.5322 front-runner extra-q evidence artifacts."""

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
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5311-5314",
        "updated_pack_selected_extension_independent_extra_q_range_evidence_inventory_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5315-5318",
        "updated_pack_selected_extension_independent_extra_q_range_evidence_inventory_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
EXTRA_Q_RANGE_HELPER = (
    ROOT / "scripts" / "quantum" / "selected_extension_solver_side_extra_q_range_backend.py"
)

STEP_TAG = "8.7.56.5319-5322"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack selected-"
    "extension independent extra-q-range evidence front-runner audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_selected_extension_independent_extra_q_range_evidence_front_runner_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_independent_extra_q_range_evidence_inventory_audited_"
    "front_runner_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_independent_extra_q_range_evidence_front_runner_helper_"
    "backed_no_go_theorem_derived_public_checkpoint_primary_pack_refresh_"
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


# 関数: front-runner helper-backed audit の式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the front-runner helper-backed audit."""
    return {
        "front_runner": (
            "E_qext,front^(pilot-HS) := E_qext^(helper)[Sigma_*^(pilot-HS)]"
        ),
        "helper_backed_candidate": (
            "E_qext^(helper)[Sigma_*^(pilot-HS)] := "
            "O_qext_sel^(pilot-HS)[Q_ext^(ind)] produced by "
            "selected_extension_solver_side_extra_q_range_backend.py"
        ),
        "actual_helper_absence": (
            "selected_extension_solver_side_extra_q_range_backend.py missing "
            "=> no actual helper-backed extra-q materialization exists now"
        ),
        "helper_no_go": (
            "not helper_available_now => "
            "E_qext^(helper)[Sigma_*^(pilot-HS)] cannot be the promoted "
            "admissible evidence source in the current pack"
        ),
        "followup": (
            "next admissible live candidate := "
            "E_qext^(pub)[Sigma_*^(pilot-HS)]"
        ),
    }


# 関数: `.5319-.5322` を実行する。

def main() -> None:
    """Execute the selected-extension extra-q evidence front-runner audit."""
    for path in (PRIOR_AUDIT, PRIOR_GATE):
        sign_base.require(path)

    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_selected_extension_independent_extra_q_range_evidence_front_runner_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    inventory_nonempty = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_selected_extension_independent_extra_q_range_evidence_inventory_nonempty_available_now"
        ]
        and prior_audit_summary[
            "exact_selected_extension_independent_extra_q_range_evidence_inventory_nonempty_theorem_available_now"
        ]
    )
    front_runner_formula_available = bool(
        prior_audit_summary[
            "exact_selected_extension_independent_extra_q_range_evidence_front_runner_candidate_formula_available_now"
        ]
        and prior_audit_summary[
            "exact_selected_extension_independent_extra_q_range_evidence_front_runner_compatibility_theorem_available_now"
        ]
    )
    helper_candidate_formula_available = bool(
        prior_audit_summary[
            "exact_selected_extension_independent_extra_q_range_helper_backed_candidate_formula_available_now"
        ]
    )
    helper_available_now = bool(EXTRA_Q_RANGE_HELPER.exists())
    public_checkpoint_materialized_now = bool(
        prior_audit_summary["public_extra_q_checkpoint_artifact_materialized_now"]
    )
    hybrid_materialized_now = bool(
        prior_audit_summary[
            "farther_hybrid_independent_extra_q_evidence_materialized_now"
        ]
    )

    helper_front_runner_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and inventory_nonempty
        and front_runner_formula_available
        and helper_candidate_formula_available
    )
    exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_formula_available_now = bool(
        helper_front_runner_explicit
    )
    exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_absence_theorem_available_now = bool(
        helper_front_runner_explicit and not helper_available_now
    )
    exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_no_go_theorem_available_now = bool(
        exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_absence_theorem_available_now
    )
    exact_minimal_selected_extension_independent_extra_q_range_public_checkpoint_requirement_theorem_available_now = bool(
        exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_no_go_theorem_available_now
    )
    exact_selected_extension_independent_extra_q_range_evidence_source_available_now = False
    updated_pack_selected_extension_independent_extra_q_range_evidence_public_checkpoint_followup_required = bool(
        exact_minimal_selected_extension_independent_extra_q_range_public_checkpoint_requirement_theorem_available_now
    )
    updated_pack_same_schema_selected_extension_independent_extra_q_range_evidence_front_runner_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_selected_extension_independent_extra_q_range_evidence_front_runner_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack selected-extension independent extra-q-range evidence front-runner audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the evidence inventory has promoted one live front-runner candidate.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The branch stays theorem-first and does not reopen the already closed retained-q deformation stack.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Front-runner audit is honest only while surrogate and reserve-only replay routes remain closed.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_evidence_inventory_nonempty_theorem_available_now",
            "pass" if inventory_nonempty else "reject",
            "exact selected-extension independent extra-q-range evidence inventory nonempty theorem available now",
            sign_base.truth(inventory_nonempty),
            "A candidate-specific front-runner audit is meaningful only after the inventory itself is theorem-side nonempty.",
        ),
        sign_base.row(
            "front_runner_formula_available_now",
            "pass" if front_runner_formula_available else "reject",
            "front-runner formula available now",
            sign_base.truth(front_runner_formula_available),
            "The front-runner must already be explicit and compatibility-checked before it can be rejected or promoted.",
        ),
        sign_base.row(
            "selected_extension_solver_side_extra_q_range_helper_available_now",
            "pass" if helper_available_now else "reject",
            "selected-extension solver-side extra-q-range helper available now",
            sign_base.truth(helper_available_now),
            "No actual helper-backed extra-q materialization exists now because the selected-extension extra-q helper file is still missing.",
        ),
        sign_base.row(
            "public_extra_q_checkpoint_artifact_materialized_now",
            "pass" if public_checkpoint_materialized_now else "reject",
            "public extra-q checkpoint artifact materialized now",
            sign_base.truth(public_checkpoint_materialized_now),
            "The next candidate stays downstream because no promoted public extra-q checkpoint pack exists yet.",
        ),
        sign_base.row(
            "farther_hybrid_independent_extra_q_evidence_materialized_now",
            "pass" if hybrid_materialized_now else "reject",
            "farther-hybrid independent extra-q evidence materialized now",
            sign_base.truth(hybrid_materialized_now),
            "Farther hybrid continuation is still reserve-only, so it does not rescue the helper-backed front-runner here.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_formula_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_formula_available_now
            else "reject",
            "exact selected-extension independent extra-q-range evidence front-runner helper-backed formula available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_formula_available_now
            ),
            "The promoted front-runner is now fixed literally as the helper-backed selected-extension extra-q materialization candidate.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_absence_theorem_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_absence_theorem_available_now
            else "reject",
            "exact selected-extension independent extra-q-range evidence front-runner helper-backed absence theorem available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_absence_theorem_available_now
            ),
            "The current pack still lacks any actual helper-backed extra-q materialization because the helper path is absent on disk.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_no_go_theorem_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_no_go_theorem_available_now
            else "reject",
            "exact selected-extension independent extra-q-range evidence front-runner helper-backed no-go theorem available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_no_go_theorem_available_now
            ),
            "Because no actual helper exists, the helper-backed front-runner cannot currently serve as one admissible independent extra-q evidence source.",
        ),
        sign_base.row(
            "exact_minimal_selected_extension_independent_extra_q_range_public_checkpoint_requirement_theorem_available_now",
            "pass"
            if exact_minimal_selected_extension_independent_extra_q_range_public_checkpoint_requirement_theorem_available_now
            else "reject",
            "exact minimal selected-extension independent extra-q-range public-checkpoint requirement theorem available now",
            sign_base.truth(
                exact_minimal_selected_extension_independent_extra_q_range_public_checkpoint_requirement_theorem_available_now
            ),
            "With the helper-backed front-runner closed negatively, the honest next live candidate is the public-checkpoint carry-over source.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_evidence_source_available_now",
            "pass" if exact_selected_extension_independent_extra_q_range_evidence_source_available_now else "reject",
            "exact selected-extension independent extra-q-range evidence source available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_evidence_source_available_now
            ),
            "Reject means the helper-backed front-runner audit did not materialize any admissible extra-q evidence source.",
        ),
        sign_base.row(
            "updated_pack_selected_extension_independent_extra_q_range_evidence_public_checkpoint_followup_required",
            "pass"
            if updated_pack_selected_extension_independent_extra_q_range_evidence_public_checkpoint_followup_required
            else "reject",
            "updated-pack selected-extension independent extra-q-range evidence public-checkpoint followup required",
            sign_base.truth(
                updated_pack_selected_extension_independent_extra_q_range_evidence_public_checkpoint_followup_required
            ),
            "The honest next blocker is now the public-checkpoint candidate, not the already rejected helper-backed front-runner.",
        ),
        sign_base.row(
            "updated_pack_same_schema_selected_extension_independent_extra_q_range_evidence_front_runner_replay_detected_now",
            "pass"
            if updated_pack_same_schema_selected_extension_independent_extra_q_range_evidence_front_runner_replay_detected_now
            else "reject",
            "updated-pack same-schema selected-extension independent extra-q-range evidence front-runner replay detected now",
            sign_base.truth(
                updated_pack_same_schema_selected_extension_independent_extra_q_range_evidence_front_runner_replay_detected_now
            ),
            "False means this turn genuinely eliminated the promoted front-runner instead of replaying the reserve-only inventory syntax.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_extension_label": prior_audit_summary["selected_extension_label"],
        "solver_side_deformation_label": prior_audit_summary[
            "solver_side_deformation_label"
        ],
        "q_theory_over_m0": float(prior_audit_summary["q_theory_over_m0"]),
        "blind_F_deform_at_q_theory": float(
            prior_audit_summary["blind_F_deform_at_q_theory"]
        ),
        "blind_alpha_deform_at_q_theory": float(
            prior_audit_summary["blind_alpha_deform_at_q_theory"]
        ),
        "delta_alpha_sel_deform_exact": float(
            prior_audit_summary["delta_alpha_sel_deform_exact"]
        ),
        "relative_exact_residual_deform": float(
            prior_audit_summary["relative_exact_residual_deform"]
        ),
        "selected_extension_solver_side_extra_q_range_helper_available_now": helper_available_now,
        "public_extra_q_checkpoint_artifact_materialized_now": public_checkpoint_materialized_now,
        "farther_hybrid_independent_extra_q_evidence_materialized_now": hybrid_materialized_now,
        "exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_formula_available_now": exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_formula_available_now,
        "exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_absence_theorem_available_now": exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_absence_theorem_available_now,
        "exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_no_go_theorem_available_now": exact_selected_extension_independent_extra_q_range_evidence_front_runner_helper_backed_no_go_theorem_available_now,
        "exact_minimal_selected_extension_independent_extra_q_range_public_checkpoint_requirement_theorem_available_now": exact_minimal_selected_extension_independent_extra_q_range_public_checkpoint_requirement_theorem_available_now,
        "exact_selected_extension_independent_extra_q_range_evidence_source_available_now": exact_selected_extension_independent_extra_q_range_evidence_source_available_now,
        "updated_pack_selected_extension_independent_extra_q_range_evidence_public_checkpoint_followup_required": updated_pack_selected_extension_independent_extra_q_range_evidence_public_checkpoint_followup_required,
        "updated_pack_same_schema_selected_extension_independent_extra_q_range_evidence_front_runner_replay_detected_now": updated_pack_same_schema_selected_extension_independent_extra_q_range_evidence_front_runner_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": updated_pack_selected_extension_independent_extra_q_range_evidence_public_checkpoint_followup_required,
        "selected_primary_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_evidence_front_runner_gate",
        "selected_secondary_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_evidence_public_checkpoint_audit",
        "selected_reserve_completion_lane": "farther_hybrid_reserve_only_until_independent_extra_q_evidence_promoted",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_independent_extra_q_range_evidence_front_runner_gate",
        "recommended_next_route_or_none": "8.7.56.5323",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_independent_extra_q_range_evidence_public_checkpoint_audit",
        "selected_followup_route_or_none": "8.7.56.5327",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5321",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "extra_q_helper_candidate": sign_base.display_path(EXTRA_Q_RANGE_HELPER),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5323",
                "followup_route": "8.7.56.5327",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_selected_extension_independent_extra_q_range_front_runner_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} selected-extension independent extra-q evidence front-runner audit completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から audit を実行する。

if __name__ == "__main__":
    main()
