#!/usr/bin/env python3
"""Generate 8.7.56.5051-.5054 yet-deeper selected-candidate family gate artifacts."""

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
        "8.7.56.5047-5050",
        "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selected_candidate_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5051-5054"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack beyond-current-"
    "written-action selector chart representative concrete-rule selector "
    "representative selected-candidate selector selected-candidate selected-"
    "candidate selected-candidate selected-candidate gate / route refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selected_candidate_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selector_chart_representative_concrete_rule_"
    "selector_representative_selected_candidate_selector_selected_candidate_"
    "selected_candidate_selected_candidate_selected_candidate_no_go_theorem_"
    "derived_selector_chart_representative_concrete_rule_selector_"
    "representative_selected_candidate_selector_selected_candidate_"
    "selected_candidate_selected_candidate_selected_candidate_selector_"
    "primary_pack_refresh_secondary_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selector_chart_representative_concrete_rule_"
    "selector_representative_selected_candidate_selector_selected_candidate_"
    "selected_candidate_selected_candidate_selected_candidate_no_go_theorem_"
    "derived_selector_chart_representative_concrete_rule_selector_"
    "representative_selected_candidate_selector_selected_candidate_"
    "selected_candidate_selected_candidate_selected_candidate_selector_"
    "primary_hard_stop_trigger_secondary_next"
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


# 関数: hard-stop gate で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the hard-stop gate."""
    return {
        "gate_a": (
            "Gate A = exact beyond-current-written-action selector chart "
            "representative concrete-rule selector representative selected-"
            "candidate selector selected-candidate selected-candidate selected-"
            "candidate selected-candidate no-go theorem available now"
        ),
        "gate_b": (
            "Gate B = same-schema replay only without concrete selector-chart "
            "representative / representative-selected candidate / selected extension"
        ),
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
        "hard_stop": (
            "if only the same schema repeats and no concrete representative / "
            "selected candidate / selected extension closes, stop deeper selector "
            "descent and switch to the meta no-go lane"
        ),
    }


# 関数: `.5051-.5054` を実行する。

def main() -> None:
    """Execute the yet-deeper selected-candidate family gate / hard-stop refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a_key = (
        "gate_a_updated_pack_exact_beyond_current_written_action_selector_chart_"
        "representative_concrete_rule_selector_representative_selected_"
        "candidate_selector_selected_candidate_selected_candidate_selected_"
        "candidate_selected_candidate_no_go_available_now"
    )
    gate_b_key = "gate_b_same_schema_replay_only_without_concrete_closeout"
    gate_c_key = "gate_c_farther_hybrid_continuation_reopen_required_now"
    hard_stop_key = "updated_pack_deeper_selector_hard_stop_trigger_required_now"
    selector_available_key = (
        "exact_beyond_current_written_action_selector_chart_representative_"
        "concrete_rule_selector_representative_selected_candidate_selector_"
        "selected_candidate_selected_candidate_selected_candidate_selected_"
        "candidate_available_now"
    )
    concrete_rep_key = "exact_concrete_selector_chart_representative_available_now"
    concrete_cand_key = "exact_concrete_representative_selected_candidate_available_now"
    concrete_ext_key = "exact_concrete_selected_extension_available_now"

    gate_a = bool(
        prior_summary[
            "exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selected_candidate_no_go_theorem_available_now"
        ]
        and prior_summary[
            "exact_minimal_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selected_candidate_selector_requirement_theorem_available_now"
        ]
    )
    selector_available = bool(prior_summary[selector_available_key])
    concrete_rep = False
    concrete_cand = False
    concrete_ext = False
    gate_b = bool(gate_a and not selector_available and not concrete_rep and not concrete_cand and not concrete_ext)
    gate_c = False
    hard_stop_required = bool(gate_b)
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(prior_summary["failure_matrix_non_surrogate_guard_preserved"])
    same_tag_rerun = bool(
        prior_summary[
            "updated_pack_same_tag_selected_candidate_selected_candidate_selected_candidate_selector_downstream_rerun_admissible_now"
        ]
    )
    blind_blocked = bool(prior_summary["blind_vector_observable_gate_still_blocked"])
    pack_update_required_now = bool(hard_stop_required)

    row_specs = [
        (
            gate_a_key,
            gate_a,
            "gate A exact beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate selected-candidate no-go available now",
            "The theorem stack now closes that current theory fixes only Cand_cand_cand_cand_cand_sel_rule[...] and not one canonical selector-selected-candidate.",
        ),
        (
            gate_b_key,
            gate_b,
            "gate B same-schema replay only without concrete closeout",
            "This branch confirms that the result is again the same schema only, with no concrete selector-chart representative, representative-selected candidate, or selected extension closed.",
        ),
        (
            gate_c_key,
            gate_c,
            "gate C farther hybrid continuation reopen required now",
            "Extra q-range remains reserve-only because the blocker is still theorem-side selector completion.",
        ),
        (
            hard_stop_key,
            hard_stop_required,
            "updated-pack deeper selector hard stop trigger required now",
            "Because the same schema replayed again without a concrete closeout, deeper selector descent must stop and switch to the meta no-go lane.",
        ),
        (
            "retry_gate_computation_mode_selected",
            retry_mode,
            "retry gate computation mode selected",
            "This gate is meaningful only because it decides whether theorem-first descent should be stopped.",
        ),
        (
            "failure_matrix_non_surrogate_guard_preserved",
            non_surrogate_guard,
            "failure-matrix non-surrogate guard preserved",
            "Hard stop does not reopen the exhausted surrogate family.",
        ),
        (
            selector_available_key,
            selector_available,
            "exact beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate selected-candidate available now",
            "The current theorem stack still fixes only the family Cand_cand_cand_cand_cand_sel_rule and no-go, not one concrete selector-selected-candidate.",
        ),
        (
            concrete_rep_key,
            concrete_rep,
            "exact concrete selector-chart representative available now",
            "No concrete selector-chart representative has closed on this branch.",
        ),
        (
            concrete_cand_key,
            concrete_cand,
            "exact concrete representative-selected candidate available now",
            "No concrete representative-selected candidate has closed on this branch.",
        ),
        (
            concrete_ext_key,
            concrete_ext,
            "exact concrete selected extension available now",
            "No concrete selected extension has closed on this branch.",
        ),
        (
            "updated_pack_same_tag_deeper_selector_downstream_rerun_admissible_now",
            same_tag_rerun,
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
            "A substantive route decision happened here: theorem-first descent is stopped and the meta no-go lane is promoted.",
        ),
    ]
    rows = [
        sign_base.row(rid, "pass" if ok else "reject", metric, sign_base.truth(ok), note)
        for rid, ok, metric, note in row_specs
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        gate_a_key: gate_a,
        gate_b_key: gate_b,
        gate_c_key: gate_c,
        hard_stop_key: hard_stop_required,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        selector_available_key: selector_available,
        concrete_rep_key: concrete_rep,
        concrete_cand_key: concrete_cand,
        concrete_ext_key: concrete_ext,
        "updated_pack_same_tag_deeper_selector_downstream_rerun_admissible_now": same_tag_rerun,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "pack_update_required_now": pack_update_required_now,
        "selected_primary_completion_lane": "updated_pack_meta_no_go_current_theory_cannot_canonically_select_one_extension_theorem_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_meta_no_go_current_theory_cannot_canonically_select_one_extension_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.5055",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_meta_no_go_current_theory_cannot_canonically_select_one_extension_gate",
        "selected_followup_route_or_none": "8.7.56.5059",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5053",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5055",
                "followup_route": "8.7.56.5059",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selected_candidate_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} updated-pack beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate selected-candidate gate completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
