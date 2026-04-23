#!/usr/bin/env python3
"""Generate 8.7.56.5115-.5118 vacuum-anchor minimal-deformation gate artifacts."""

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
        "8.7.56.5111-5114",
        "updated_pack_external_rule_selector_vacuum_anchor_minimal_deformation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5115-5118"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack external "
    "rule-selector vacuum-anchor minimal-deformation gate / route refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_external_rule_selector_vacuum_anchor_minimal_deformation_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_rule_selector_vacuum_anchor_minimal_deformation_concrete_rule_"
    "no_go_theorem_derived_chart_measure_convention_primary_pack_refresh_"
    "secondary_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_rule_selector_vacuum_anchor_minimal_deformation_no_go_theorem_"
    "audited_chart_measure_convention_primary_hybrid_reserve_secondary_next"
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


# 関数: gate で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the vacuum-anchor minimal-deformation gate."""
    return {
        "gate_a": (
            "Gate A = exact external rule-selector vacuum-anchor "
            "minimal-deformation concrete-rule no-go theorem available now"
        ),
        "gate_b": (
            "Gate B = external rule-selector chart/measure convention "
            "inventory promoted next"
        ),
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.5115-.5118` を実行する。

def main() -> None:
    """Execute the vacuum-anchor minimal-deformation gate / route refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_external_rule_selector_vacuum_anchor_minimal_deformation_concrete_rule_no_go_theorem_available_now"
        ]
        and prior_summary[
            "exact_minimal_external_rule_selector_chart_measure_convention_requirement_theorem_available_now"
        ]
    )
    gate_b = bool(
        prior_summary[
            "updated_pack_external_rule_selector_chart_measure_convention_followup_required"
        ]
    )
    gate_c = False
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    selector_selected_now = bool(prior_summary["exact_external_rule_selector_selected_now"])
    same_schema_replay_detected = bool(
        prior_summary[
            "updated_pack_same_schema_external_rule_selector_vacuum_anchor_minimal_deformation_replay_detected_now"
        ]
    )
    blind_blocked = bool(prior_summary["blind_vector_observable_gate_still_blocked"])
    pack_update_required_now = bool(gate_b)

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_external_rule_selector_vacuum_anchor_minimal_deformation_no_go_available_now",
            "pass" if gate_a else "reject",
            "gate A exact external rule-selector vacuum-anchor minimal-deformation no-go available now",
            sign_base.truth(gate_a),
            "The promoted selector has now been audited negatively: it is not yet one concrete adopted rule.",
        ),
        sign_base.row(
            "gate_b_updated_pack_external_rule_selector_chart_measure_convention_inventory_promoted_next",
            "pass" if gate_b else "reject",
            "gate B updated-pack external rule-selector chart/measure convention inventory promoted next",
            sign_base.truth(gate_b),
            "The honest next blocker is chart/measure convention inventory for the promoted selector rather than generic selector existence.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "pass" if gate_c else "reject",
            "gate C farther hybrid continuation reopen required now",
            sign_base.truth(gate_c),
            "Farther hybrid continuation remains closed because selector adoption is still unresolved.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This route refresh follows a substantive selector-candidate audit rather than same-tag replay.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Promoting the selector no-go does not reopen exhausted internal or candidate-level recursion lanes.",
        ),
        sign_base.row(
            "exact_external_rule_selector_selected_now",
            "pass" if selector_selected_now else "reject",
            "exact external rule-selector selected now",
            sign_base.truth(selector_selected_now),
            "No adopted external selector exists yet after the promoted selector audit.",
        ),
        sign_base.row(
            "updated_pack_same_schema_external_rule_selector_vacuum_anchor_minimal_deformation_replay_detected_now",
            "pass" if same_schema_replay_detected else "reject",
            "updated-pack same-schema external rule-selector vacuum-anchor minimal-deformation replay detected now",
            sign_base.truth(same_schema_replay_detected),
            "False means this turn did not reopen recursive selector descent; it only promoted a newly isolated blocker.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on one adopted external selector and one concrete extension.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "A substantive lane shift happened here: the next blocker is chart/measure convention inventory for the promoted selector.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "gate_a_updated_pack_exact_external_rule_selector_vacuum_anchor_minimal_deformation_no_go_available_now": gate_a,
        "gate_b_updated_pack_external_rule_selector_chart_measure_convention_inventory_promoted_next": gate_b,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_external_rule_selector_selected_now": selector_selected_now,
        "updated_pack_same_schema_external_rule_selector_vacuum_anchor_minimal_deformation_replay_detected_now": same_schema_replay_detected,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "pack_update_required_now": pack_update_required_now,
        "selected_primary_completion_lane": "updated_pack_external_rule_selector_chart_measure_convention_inventory_theorem_audit",
        "selected_secondary_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_reserve_completion_lane": "promoted_selector_candidate_replay_closed",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_rule_selector_chart_measure_convention_inventory_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.5119",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_rule_selector_chart_measure_convention_inventory_gate",
        "selected_followup_route_or_none": "8.7.56.5123",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5117",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5119",
                "followup_route": "8.7.56.5123",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_external_rule_selector_vacuum_anchor_minimal_deformation_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} external rule-selector vacuum-anchor minimal-deformation gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
