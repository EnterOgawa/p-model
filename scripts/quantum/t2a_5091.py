#!/usr/bin/env python3
"""Generate 8.7.56.5091-.5094 Schur-complement selector gate artifacts."""

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
        "8.7.56.5087-5090",
        "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_selector_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5091-5094"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack external "
    "selector candidate independent probe-slot Schur-complement selector gate "
    "/ route refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_selector_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_selector_candidate_independent_probe_slot_schur_complement_"
    "selector_no_go_theorem_derived_candidate_hard_stop_primary_pack_refresh_"
    "secondary_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_selector_candidate_independent_probe_slot_schur_complement_"
    "selector_no_go_theorem_audited_candidate_hard_stop_primary_hybrid_reserve_"
    "secondary_next"
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


# 関数: selector gate で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the Schur-complement selector gate."""
    return {
        "gate_a": (
            "Gate A = exact external selector candidate independent probe-slot "
            "Schur-complement selector no-go theorem available now"
        ),
        "gate_b": (
            "Gate B = same-schema replay only without one concrete rule on the "
            "front-runner candidate"
        ),
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.5091-.5094` を実行する。

def main() -> None:
    """Execute the Schur-complement selector gate / route refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_selector_no_go_theorem_available_now"
        ]
        and prior_summary[
            "exact_minimal_external_selector_candidate_independent_probe_slot_schur_complement_selector_representative_requirement_theorem_available_now"
        ]
    )
    gate_b = bool(
        prior_summary[
            "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_selector_same_schema_replay_detected_now"
        ]
        and not prior_summary[
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_selector_available_now"
        ]
    )
    gate_c = False
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    concrete_rule_available = bool(
        prior_summary.get(
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_available_now",
            False,
        )
    )
    candidate_level_hard_stop_trigger_required_now = bool(gate_a and gate_b)
    same_tag_candidate_replay_admissible = bool(
        prior_summary[
            "updated_pack_same_tag_external_selector_candidate_concrete_rule_replay_admissible_now"
        ]
    )
    blind_blocked = bool(prior_summary["blind_vector_observable_gate_still_blocked"])
    pack_update_required_now = bool(candidate_level_hard_stop_trigger_required_now)

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_external_selector_candidate_independent_probe_slot_schur_complement_selector_no_go_available_now",
            "pass" if gate_a else "reject",
            "gate A exact external selector candidate independent probe-slot Schur-complement selector no-go available now",
            sign_base.truth(gate_a),
            "The theorem stack now closes that the front-runner candidate still fixes only a selector family/order class and not one canonical selector.",
        ),
        sign_base.row(
            "gate_b_updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_same_schema_replay_only_without_concrete_rule",
            "pass" if gate_b else "reject",
            "gate B external selector candidate independent probe-slot Schur-complement same-schema replay only without concrete rule",
            sign_base.truth(gate_b),
            "This turn added only selector family/equivalence/no-go objects and still did not produce one concrete rule on the front-runner candidate.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "pass" if gate_c else "reject",
            "gate C farther hybrid continuation reopen required now",
            sign_base.truth(gate_c),
            "Farther hybrid continuation remains reserve-only because the blocker is still selector completion on the external candidate lane.",
        ),
        sign_base.row(
            "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_candidate_level_hard_stop_trigger_required_now",
            "pass" if candidate_level_hard_stop_trigger_required_now else "reject",
            "updated-pack external selector candidate independent probe-slot Schur-complement candidate-level hard stop trigger required now",
            sign_base.truth(candidate_level_hard_stop_trigger_required_now),
            "Because the selector layer reproduced the same underdetermination schema without closing a concrete rule, candidate-level hard stop review is now required.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This route refresh converts a formal selector theorem into a hard-stop decision rather than allowing uncontrolled recursive descent.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Promoting the selector no-go does not reopen exhausted surrogate or internal-lane routes.",
        ),
        sign_base.row(
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_available_now",
            "pass" if concrete_rule_available else "reject",
            "exact external selector candidate independent probe-slot Schur-complement concrete-rule available now",
            sign_base.truth(concrete_rule_available),
            "The front-runner candidate still does not contain one concrete rule.",
        ),
        sign_base.row(
            "updated_pack_same_tag_external_selector_candidate_concrete_rule_replay_admissible_now",
            "pass" if same_tag_candidate_replay_admissible else "reject",
            "updated-pack same-tag external selector candidate concrete-rule replay admissible now",
            sign_base.truth(same_tag_candidate_replay_admissible),
            "Same-tag candidate replay remains closed because the blocker is now the candidate-level hard stop itself.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on one concrete external selector and one concrete extension.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "A substantive lane shift happened here: front-runner candidate descent must now be judged by hard stop rather than continued recursively.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "gate_a_updated_pack_exact_external_selector_candidate_independent_probe_slot_schur_complement_selector_no_go_available_now": gate_a,
        "gate_b_updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_same_schema_replay_only_without_concrete_rule": gate_b,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c,
        "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_candidate_level_hard_stop_trigger_required_now": candidate_level_hard_stop_trigger_required_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_external_selector_candidate_independent_probe_slot_schur_complement_concrete_rule_available_now": concrete_rule_available,
        "updated_pack_same_tag_external_selector_candidate_concrete_rule_replay_admissible_now": same_tag_candidate_replay_admissible,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "pack_update_required_now": pack_update_required_now,
        "selected_primary_completion_lane": "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_candidate_level_hard_stop_meta_no_go_audit",
        "selected_secondary_completion_lane": "external_rule_selector_inventory_after_candidate_closeout",
        "selected_reserve_completion_lane": "same_tag_external_selector_candidate_concrete_rule_replay_closed",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_candidate_level_hard_stop_meta_no_go_audit",
        "recommended_next_route_or_none": "8.7.56.5095",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_candidate_level_hard_stop_meta_no_go_gate",
        "selected_followup_route_or_none": "8.7.56.5099",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5093",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5095",
                "followup_route": "8.7.56.5099",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_external_selector_candidate_independent_probe_slot_schur_complement_selector_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Schur-complement selector gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
