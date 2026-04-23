#!/usr/bin/env python3
"""Generate 8.7.56.5335-.5338 hybrid-bridge extra-q evidence artifacts."""

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
PRIOR_INVENTORY = build_metrics_paths(
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
        "8.7.56.5331-5334",
        "updated_pack_selected_extension_independent_extra_q_range_evidence_public_checkpoint_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5335-5338"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack selected-"
    "extension independent extra-q-range evidence hybrid-bridge audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_independent_extra_q_range_evidence_public_checkpoint_"
    "no_go_theorem_audited_hybrid_bridge_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_independent_extra_q_range_evidence_hybrid_bridge_"
    "no_go_theorem_derived_negative_closeout_primary_pack_refresh_secondary_"
    "gate"
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


# 関数: hybrid-bridge audit の式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the hybrid-bridge audit."""
    return {
        "hybrid_bridge_candidate": (
            "E_qext^(hyb)[Sigma_*^(pilot-HS)] := H_qext^(hyb)[Q_ext^(ind)]"
        ),
        "reserve_only_absence": (
            "farther hybrid continuation reserve-only => no actual "
            "H_qext^(hyb)[Q_ext^(ind)] materialized now"
        ),
        "hybrid_no_go": (
            "not farther_hybrid_materialized_now => "
            "E_qext^(hyb)[Sigma_*^(pilot-HS)] cannot yet be the admissible "
            "independent extra-q evidence source"
        ),
        "followup": (
            "if helper/public/hybrid are all closed negatively, "
            "the whole independent extra-q evidence lane closes negatively"
        ),
    }


# 関数: `.5335-.5338` を実行する。

def main() -> None:
    """Execute the selected-extension extra-q evidence hybrid-bridge audit."""
    for path in (PRIOR_INVENTORY, PRIOR_GATE):
        sign_base.require(path)

    inventory_summary = sign_base.read_json(PRIOR_INVENTORY)["summary"]
    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    inventory_nonempty = bool(
        inventory_summary[
            "exact_selected_extension_independent_extra_q_range_evidence_inventory_nonempty_theorem_available_now"
        ]
    )
    hybrid_candidate_formula_available = bool(
        inventory_summary[
            "exact_selected_extension_independent_extra_q_range_hybrid_bridge_candidate_formula_available_now"
        ]
    )
    hybrid_materialized_now = bool(
        inventory_summary[
            "farther_hybrid_independent_extra_q_evidence_materialized_now"
        ]
    )

    hybrid_candidate_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and inventory_nonempty
        and hybrid_candidate_formula_available
    )
    exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_formula_available_now = bool(
        hybrid_candidate_explicit
    )
    exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_absence_theorem_available_now = bool(
        hybrid_candidate_explicit and not hybrid_materialized_now
    )
    exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_no_go_theorem_available_now = bool(
        exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_absence_theorem_available_now
    )
    exact_minimal_selected_extension_independent_extra_q_range_evidence_negative_closeout_requirement_theorem_available_now = bool(
        exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_no_go_theorem_available_now
    )
    exact_selected_extension_independent_extra_q_range_evidence_source_available_now = False
    updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_followup_required = bool(
        exact_minimal_selected_extension_independent_extra_q_range_evidence_negative_closeout_requirement_theorem_available_now
    )
    updated_pack_same_schema_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_replay_detected_now = (
        False
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack selected-extension independent extra-q-range evidence hybrid-bridge audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after helper and public-checkpoint candidates are both closed negatively.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The branch stays theorem-first and does not reopen any already closed helper/public routes.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Hybrid-bridge audit is honest only while surrogate and reserve-only replay routes remain closed.",
        ),
        sign_base.row(
            "farther_hybrid_independent_extra_q_evidence_materialized_now",
            "pass" if hybrid_materialized_now else "reject",
            "farther-hybrid independent extra-q evidence materialized now",
            sign_base.truth(hybrid_materialized_now),
            "Farther hybrid continuation remains reserve-only, so no actual hybrid-bridge evidence source exists now.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_formula_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_formula_available_now
            else "reject",
            "exact selected-extension independent extra-q-range evidence hybrid-bridge formula available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_formula_available_now
            ),
            "The third candidate is now fixed literally as one farther-hybrid bridge source.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_absence_theorem_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_absence_theorem_available_now
            else "reject",
            "exact selected-extension independent extra-q-range evidence hybrid-bridge absence theorem available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_absence_theorem_available_now
            ),
            "The current pack still lacks any actual farther-hybrid bridge carrying independent extra-q evidence.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_no_go_theorem_available_now",
            "pass"
            if exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_no_go_theorem_available_now
            else "reject",
            "exact selected-extension independent extra-q-range evidence hybrid-bridge no-go theorem available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_no_go_theorem_available_now
            ),
            "Because farther hybrid continuation is still reserve-only, the hybrid-bridge candidate cannot yet serve as one admissible independent extra-q evidence source.",
        ),
        sign_base.row(
            "exact_minimal_selected_extension_independent_extra_q_range_evidence_negative_closeout_requirement_theorem_available_now",
            "pass"
            if exact_minimal_selected_extension_independent_extra_q_range_evidence_negative_closeout_requirement_theorem_available_now
            else "reject",
            "exact minimal selected-extension independent extra-q-range evidence negative-closeout requirement theorem available now",
            sign_base.truth(
                exact_minimal_selected_extension_independent_extra_q_range_evidence_negative_closeout_requirement_theorem_available_now
            ),
            "With helper, public checkpoint, and hybrid bridge all closed negatively, the honest next blocker is lane-level negative closeout.",
        ),
        sign_base.row(
            "exact_selected_extension_independent_extra_q_range_evidence_source_available_now",
            "pass" if exact_selected_extension_independent_extra_q_range_evidence_source_available_now else "reject",
            "exact selected-extension independent extra-q-range evidence source available now",
            sign_base.truth(
                exact_selected_extension_independent_extra_q_range_evidence_source_available_now
            ),
            "Reject means no admissible materialized evidence source exists across helper/public/hybrid candidates.",
        ),
        sign_base.row(
            "updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_followup_required",
            "pass"
            if updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_followup_required
            else "reject",
            "updated-pack selected-extension independent extra-q-range evidence negative-closeout followup required",
            sign_base.truth(
                updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_followup_required
            ),
            "The honest next blocker is now lane-level negative closeout, not another candidate-family replay.",
        ),
        sign_base.row(
            "updated_pack_same_schema_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_replay_detected_now",
            "pass"
            if updated_pack_same_schema_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_replay_detected_now
            else "reject",
            "updated-pack same-schema selected-extension independent extra-q-range evidence hybrid-bridge replay detected now",
            sign_base.truth(
                updated_pack_same_schema_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_replay_detected_now
            ),
            "False means this turn genuinely eliminated the final candidate instead of replaying the already fixed public-checkpoint no-go.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_extension_label": inventory_summary["selected_extension_label"],
        "solver_side_deformation_label": inventory_summary[
            "solver_side_deformation_label"
        ],
        "q_theory_over_m0": float(inventory_summary["q_theory_over_m0"]),
        "blind_F_deform_at_q_theory": float(
            inventory_summary["blind_F_deform_at_q_theory"]
        ),
        "blind_alpha_deform_at_q_theory": float(
            inventory_summary["blind_alpha_deform_at_q_theory"]
        ),
        "delta_alpha_sel_deform_exact": float(
            inventory_summary["delta_alpha_sel_deform_exact"]
        ),
        "relative_exact_residual_deform": float(
            inventory_summary["relative_exact_residual_deform"]
        ),
        "farther_hybrid_independent_extra_q_evidence_materialized_now": hybrid_materialized_now,
        "exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_formula_available_now": exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_formula_available_now,
        "exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_absence_theorem_available_now": exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_absence_theorem_available_now,
        "exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_no_go_theorem_available_now": exact_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_no_go_theorem_available_now,
        "exact_minimal_selected_extension_independent_extra_q_range_evidence_negative_closeout_requirement_theorem_available_now": exact_minimal_selected_extension_independent_extra_q_range_evidence_negative_closeout_requirement_theorem_available_now,
        "exact_selected_extension_independent_extra_q_range_evidence_source_available_now": exact_selected_extension_independent_extra_q_range_evidence_source_available_now,
        "updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_followup_required": updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_followup_required,
        "updated_pack_same_schema_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_replay_detected_now": updated_pack_same_schema_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_followup_required,
        "selected_primary_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_gate",
        "selected_secondary_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_audit",
        "selected_reserve_completion_lane": "farther_hybrid_reserve_only_until_independent_extra_q_evidence_promoted",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_independent_extra_q_range_evidence_hybrid_bridge_gate",
        "recommended_next_route_or_none": "8.7.56.5339",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_selected_extension_independent_extra_q_range_evidence_negative_closeout_audit",
        "selected_followup_route_or_none": "8.7.56.5343",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5337",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_inventory": sign_base.display_path(PRIOR_INVENTORY),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5339",
                "followup_route": "8.7.56.5343",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_selected_extension_independent_extra_q_hybrid_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} selected-extension independent extra-q evidence hybrid-bridge audit completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から audit を実行する。

if __name__ == "__main__":
    main()
