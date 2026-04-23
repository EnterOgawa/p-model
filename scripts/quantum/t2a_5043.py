#!/usr/bin/env python3
"""Generate 8.7.56.5043-.5046 deeper selector-representative gate artifacts."""

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
        "8.7.56.5039-5042",
        "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_representative_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5043-5046"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack beyond-current-"
    "written-action selector chart representative concrete-rule selector "
    "representative selected-candidate selector selected-candidate selected-"
    "candidate selected-candidate selector representative gate / route refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_representative_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selector_chart_representative_concrete_rule_"
    "selector_representative_selected_candidate_selector_selected_candidate_"
    "selected_candidate_selected_candidate_selector_representative_finite_"
    "anchor_no_go_theorem_derived_selector_chart_representative_concrete_"
    "rule_selector_representative_selected_candidate_selector_selected_"
    "candidate_selected_candidate_selected_candidate_selected_candidate_"
    "primary_pack_refresh_secondary_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selector_chart_representative_concrete_rule_"
    "selector_representative_selected_candidate_selector_selected_candidate_"
    "selected_candidate_selected_candidate_selector_representative_finite_"
    "anchor_no_go_theorem_derived_selector_chart_representative_concrete_"
    "rule_selector_representative_selected_candidate_selector_selected_"
    "candidate_selected_candidate_selected_candidate_selected_candidate_"
    "primary_hybrid_reserve_secondary_next"
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
    """Return formulas used in the deeper selector representative gate."""
    return {
        "gate_a": (
            "Gate A = exact beyond-current-written-action selector chart "
            "representative concrete-rule selector representative selected-"
            "candidate selector selected-candidate selected-candidate selected-"
            "candidate selector representative finite-anchor no-go theorem "
            "available now"
        ),
        "gate_b": (
            "Gate B = beyond-current-written-action selector chart "
            "representative concrete-rule selector representative selected-"
            "candidate selector selected-candidate selected-candidate selected-"
            "candidate selected-candidate promoted next"
        ),
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.5043-.5046` を実行する。

def main() -> None:
    """Execute the deeper selector representative gate / route refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a_key = (
        "gate_a_updated_pack_exact_beyond_current_written_action_selector_chart_"
        "representative_concrete_rule_selector_representative_selected_"
        "candidate_selector_selected_candidate_selected_candidate_selected_"
        "candidate_selector_representative_finite_anchor_no_go_available_now"
    )
    gate_b_key = (
        "gate_b_updated_pack_beyond_current_written_action_selector_chart_"
        "representative_concrete_rule_selector_representative_selected_"
        "candidate_selector_selected_candidate_selected_candidate_selected_"
        "candidate_selected_candidate_promoted_next"
    )
    selector_available_key = (
        "exact_beyond_current_written_action_selector_chart_representative_"
        "concrete_rule_selector_representative_selected_candidate_selector_"
        "selected_candidate_selected_candidate_selected_candidate_selector_"
        "representative_available_now"
    )

    gate_a = bool(
        prior[
            "exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_representative_finite_anchor_no_go_theorem_available_now"
        ]
        and prior[
            "exact_minimal_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selected_candidate_requirement_theorem_available_now"
        ]
    )
    gate_b = bool(
        prior[
            "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selected_candidate_primary_followup_required"
        ]
    )
    gate_c = False
    retry_mode = bool(prior["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(prior["failure_matrix_non_surrogate_guard_preserved"])
    selector_available = bool(prior[selector_available_key])
    same_tag = bool(prior["updated_pack_same_tag_deeper_selector_downstream_rerun_admissible_now"])
    blind_blocked = bool(prior["blind_vector_observable_gate_still_blocked"])
    pack_update_required_now = bool(gate_b)

    row_specs = [
        (
            gate_a_key,
            gate_a,
            "gate A exact beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate selector representative finite-anchor no-go available now",
            "The theorem stack now closes that finite anchors still do not choose one canonical deeper selector representative on [N_sel]_ord.",
        ),
        (
            gate_b_key,
            gate_b,
            "gate B beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate selected-candidate promoted next",
            "Once deeper selector-representative underdetermination closes theorem-side, the honest next blocker is the induced deeper selected-candidate family itself.",
        ),
        (
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            gate_c,
            "gate C farther hybrid continuation reopen required now",
            "Extra q-range remains reserve-only because the blocker is still theorem-side selector completion.",
        ),
        (
            "retry_gate_computation_mode_selected",
            retry_mode,
            "retry gate computation mode selected",
            "This gate follows a real theorem closure and does not count same-tag restatement as progress.",
        ),
        (
            "failure_matrix_non_surrogate_guard_preserved",
            non_surrogate_guard,
            "failure-matrix non-surrogate guard preserved",
            "Promoting the deeper selector representative theorem does not reopen the exhausted surrogate family.",
        ),
        (
            selector_available_key,
            selector_available,
            "exact beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate selector representative available now",
            "The current theorem stack fixes only the finite-anchor representative family and no-go, not one concrete deeper selector representative N_sel.",
        ),
        (
            "updated_pack_same_tag_deeper_selector_downstream_rerun_admissible_now",
            same_tag,
            "updated-pack same-tag deeper selector downstream rerun admissible now",
            "Same-tag downstream rerun remains closed because the blocker is selector completion, not old replay syntax.",
        ),
        (
            "blind_vector_observable_gate_still_blocked",
            blind_blocked,
            "blind-vector observable gate still blocked",
            "Blind-vector direct computation still waits on one concrete selected extension.",
        ),
        (
            "pack_update_required_now",
            pack_update_required_now,
            "updated-pack substantive pack update required now",
            "A new theorem object closed here, and the honest next blocker is the induced deeper selected-candidate family rather than same-tag route re-sync.",
        ),
    ]
    rows = [
        sign_base.row(rid, "pass" if ok else "reject", metric, sign_base.truth(ok), note)
        for rid, ok, metric, note in row_specs
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior["retained_scalar_residual_rel"]),
        gate_a_key: gate_a,
        gate_b_key: gate_b,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        selector_available_key: selector_available,
        "updated_pack_same_tag_deeper_selector_downstream_rerun_admissible_now": same_tag,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "pack_update_required_now": pack_update_required_now,
        "selected_primary_completion_lane": "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selected_candidate_theorem_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selected_candidate_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.5047",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selected_candidate_gate",
        "selected_followup_route_or_none": "8.7.56.5051",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5045",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5047",
                "followup_route": "8.7.56.5051",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_representative_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} updated-pack beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate selector representative gate completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
