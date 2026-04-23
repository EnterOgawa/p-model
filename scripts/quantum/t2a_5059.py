#!/usr/bin/env python3
"""Generate 8.7.56.5059-.5062 meta no-go closeout gate artifacts."""

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
        "8.7.56.5055-5058",
        "updated_pack_current_theory_cannot_canonically_select_one_extension_meta_no_go_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5059-5062"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack current "
    "theory internal canonical extension selection no-go closeout gate / route "
    "refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_current_theory_internal_extension_selection_no_go_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "current_theory_internal_canonical_extension_selection_no_go_closeout_"
    "theorem_derived_external_selector_axiom_or_convention_required_primary_"
    "wait_for_selector_secondary_hybrid_reserve_hold"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "current_theory_internal_extension_selection_no_go_closeout_completed_"
    "external_selector_axiom_or_convention_candidate_inventory_primary_"
    "hybrid_reserve_secondary_next"
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


# 関数: meta no-go gate で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the meta no-go closeout gate."""
    return {
        "gate_a": (
            "Gate A = exact current-theory internal canonical extension "
            "selection no-go theorem available now"
        ),
        "gate_b": (
            "Gate B = external selector axiom or convention candidate "
            "inventory promoted next"
        ),
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.5059-.5062` を実行する。

def main() -> None:
    """Execute the meta no-go closeout gate / route refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_current_theory_internal_canonical_extension_selection_no_go_theorem_available_now"
        ]
        and prior_summary[
            "exact_external_selector_axiom_or_convention_requirement_theorem_available_now"
        ]
        and prior_summary[
            "exact_current_theory_internal_canonical_extension_selection_no_go_closeout_available_now"
        ]
    )
    gate_b = bool(
        prior_summary[
            "updated_pack_external_selector_axiom_or_convention_primary_followup_required"
        ]
    )
    gate_c = False
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(prior_summary["failure_matrix_non_surrogate_guard_preserved"])
    internal_positive_selection_available = bool(
        prior_summary["exact_current_theory_internal_extension_selection_available_now"]
    )
    same_tag_deeper_selector_replay_admissible_now = bool(
        prior_summary["updated_pack_same_tag_deeper_selector_replay_admissible_now"]
    )
    blind_blocked = bool(prior_summary["blind_vector_observable_gate_still_blocked"])
    pack_update_required_now = bool(gate_b)

    row_specs = [
        (
            "gate_a_updated_pack_exact_current_theory_internal_canonical_extension_selection_no_go_available_now",
            gate_a,
            "gate A exact current-theory internal canonical extension selection no-go available now",
            "The current theory-alone lane is now theorem-side closed negatively.",
        ),
        (
            "gate_b_updated_pack_external_selector_axiom_or_convention_candidate_inventory_promoted_next",
            gate_b,
            "gate B external selector axiom or convention candidate inventory promoted next",
            "The honest next blocker is no longer internal selector descent but which external selector axiom or convention should be adopted.",
        ),
        (
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            gate_c,
            "gate C farther hybrid continuation reopen required now",
            "Extra q-range remains reserve-only because the blocker is selector choice, not missing hybrid data.",
        ),
        (
            "retry_gate_computation_mode_selected",
            retry_mode,
            "retry gate computation mode selected",
            "This gate records a substantive theorem-side route change rather than a replay bookkeeping move.",
        ),
        (
            "failure_matrix_non_surrogate_guard_preserved",
            non_surrogate_guard,
            "failure-matrix non-surrogate guard preserved",
            "The no-go closeout keeps the exhausted surrogate family shut.",
        ),
        (
            "exact_current_theory_internal_extension_selection_available_now",
            internal_positive_selection_available,
            "exact current-theory internal extension selection available now",
            "The internal lane closes negatively; it does not suddenly produce one concrete internal selector here.",
        ),
        (
            "updated_pack_same_tag_deeper_selector_replay_admissible_now",
            same_tag_deeper_selector_replay_admissible_now,
            "updated-pack same-tag deeper selector replay admissible now",
            "Same-tag deeper selector replay remains closed after the meta no-go closeout.",
        ),
        (
            "blind_vector_observable_gate_still_blocked",
            blind_blocked,
            "blind-vector observable gate still blocked",
            "Blind-vector direct computation still waits on an externally selected extension.",
        ),
        (
            "pack_update_required_now",
            pack_update_required_now,
            "updated-pack substantive pack update required now",
            "A substantive route reset happened here: the internal lane is closed and external selector inventory becomes the next honest mainline.",
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
        "gate_a_updated_pack_exact_current_theory_internal_canonical_extension_selection_no_go_available_now": gate_a,
        "gate_b_updated_pack_external_selector_axiom_or_convention_candidate_inventory_promoted_next": gate_b,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_current_theory_internal_extension_selection_available_now": internal_positive_selection_available,
        "updated_pack_same_tag_deeper_selector_replay_admissible_now": same_tag_deeper_selector_replay_admissible_now,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "pack_update_required_now": pack_update_required_now,
        "selected_primary_completion_lane": "updated_pack_external_selector_axiom_or_convention_candidate_inventory_theorem_audit",
        "selected_secondary_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_reserve_completion_lane": "same_tag_deeper_selector_replay_closed",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_selector_axiom_or_convention_candidate_inventory_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.5063",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_selector_axiom_or_convention_candidate_inventory_gate",
        "selected_followup_route_or_none": "8.7.56.5067",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5061",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5063",
                "followup_route": "8.7.56.5067",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_current_theory_internal_extension_selection_no_go_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} current-theory internal extension-selection no-go gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
