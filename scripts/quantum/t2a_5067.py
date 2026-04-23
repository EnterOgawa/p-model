#!/usr/bin/env python3
"""Generate 8.7.56.5067-.5070 external selector candidate inventory gate artifacts."""

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
        "8.7.56.5063-5066",
        "updated_pack_external_selector_candidate_inventory_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5067-5070"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack external "
    "selector axiom or convention candidate inventory gate / route refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_external_selector_candidate_inventory_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_selector_candidate_inventory_independent_probe_slot_schur_"
    "complement_extension_theorem_derived_candidate_selection_primary_"
    "pack_refresh_secondary_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_selector_candidate_inventory_independent_probe_slot_schur_"
    "complement_extension_audited_candidate_selection_primary_hybrid_reserve_"
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


# 関数: candidate inventory gate で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the external selector candidate inventory gate."""
    return {
        "gate_a": "Gate A = external selector candidate inventory nonempty theorem available now",
        "gate_b": (
            "Gate B = independent probe-slot Schur-complement candidate promoted next"
        ),
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.5067-.5070` を実行する。

def main() -> None:
    """Execute the external selector candidate inventory gate / route refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_external_selector_axiom_or_convention_candidate_inventory_nonempty_theorem_available_now"
        ]
    )
    gate_b = bool(
        prior_summary["updated_pack_external_selector_candidate_specific_followup_required"]
    )
    gate_c = False
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    candidate_selected_now = bool(
        prior_summary[
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_selected_now"
        ]
    )
    same_tag_internal_no_go_replay_admissible_now = bool(
        prior_summary["updated_pack_same_tag_internal_no_go_replay_admissible_now"]
    )
    blind_blocked = bool(prior_summary["blind_vector_observable_gate_still_blocked"])
    pack_update_required_now = bool(gate_b)

    row_specs = [
        (
            "gate_a_updated_pack_exact_external_selector_candidate_inventory_nonempty_available_now",
            gate_a,
            "gate A exact external selector candidate inventory nonempty available now",
            "The external-selector lane is now theorem-side nonempty rather than a placeholder.",
        ),
        (
            "gate_b_updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_promoted_next",
            gate_b,
            "gate B external selector candidate independent probe-slot Schur-complement promoted next",
            "The honest next blocker is whether this specific candidate can be made concrete and non-arbitrary.",
        ),
        (
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            gate_c,
            "gate C farther hybrid continuation reopen required now",
            "Hybrid continuation remains reserve-only because the blocker is still selector choice, not missing q-range coverage.",
        ),
        (
            "retry_gate_computation_mode_selected",
            retry_mode,
            "retry gate computation mode selected",
            "This gate records a substantive new-lane route refresh rather than replaying the closed internal no-go branch.",
        ),
        (
            "failure_matrix_non_surrogate_guard_preserved",
            non_surrogate_guard,
            "failure-matrix non-surrogate guard preserved",
            "The candidate inventory gate keeps the exhausted surrogate family shut.",
        ),
        (
            "exact_external_selector_candidate_independent_probe_slot_schur_complement_selected_now",
            candidate_selected_now,
            "exact external selector candidate independent probe-slot Schur-complement selected now",
            "The candidate is promoted for specific audit next, not yet adopted as the final selector principle.",
        ),
        (
            "updated_pack_same_tag_internal_no_go_replay_admissible_now",
            same_tag_internal_no_go_replay_admissible_now,
            "updated-pack same-tag internal no-go replay admissible now",
            "The closed internal lane remains closed while the external candidate lane advances.",
        ),
        (
            "blind_vector_observable_gate_still_blocked",
            blind_blocked,
            "blind-vector observable gate still blocked",
            "Blind-vector direct computation still waits until one candidate becomes a concrete selector and chooses one extension.",
        ),
        (
            "pack_update_required_now",
            pack_update_required_now,
            "updated-pack substantive pack update required now",
            "A substantive lane shift happened here: the external inventory is no longer abstract and now has one specific candidate to audit.",
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
        "gate_a_updated_pack_exact_external_selector_candidate_inventory_nonempty_available_now": gate_a,
        "gate_b_updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_promoted_next": gate_b,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_external_selector_candidate_independent_probe_slot_schur_complement_selected_now": candidate_selected_now,
        "updated_pack_same_tag_internal_no_go_replay_admissible_now": same_tag_internal_no_go_replay_admissible_now,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "pack_update_required_now": pack_update_required_now,
        "selected_primary_completion_lane": "updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_theorem_audit",
        "selected_secondary_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_reserve_completion_lane": "same_tag_internal_no_go_replay_closed",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.5071",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_selector_candidate_independent_probe_slot_schur_complement_gate",
        "selected_followup_route_or_none": "8.7.56.5075",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5069",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5071",
                "followup_route": "8.7.56.5075",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_external_selector_candidate_inventory_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} external selector candidate inventory gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
