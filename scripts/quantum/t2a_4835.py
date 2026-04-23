#!/usr/bin/env python3
"""Generate 8.7.56.4835-.4838 selected-extension-convention-selector convention representative gate artifacts."""

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
        "8.7.56.4831-4834",
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_convention_candidate_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.4835-4838"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack beyond-current-"
    "written-action selected extension convention selector selected extension "
    "convention representative gate / route refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_convention_representative_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selected_extension_convention_selector_"
    "selected_extension_convention_candidate_reduction_theorem_derived_"
    "selected_extension_convention_selector_selected_extension_convention_"
    "representative_primary_pack_refresh_secondary_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selected_extension_convention_selector_"
    "selected_extension_convention_representative_requirement_theorem_derived_"
    "selected_extension_convention_selector_selected_extension_convention_"
    "representative_candidate_primary_hybrid_reserve_secondary_next"
)


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


def build_formulae() -> dict[str, str]:
    """Return formulas used in the selected-extension-convention-selector convention representative gate."""
    return {
        "gate_a": (
            "Gate A = exact beyond-current-written-action selected extension "
            "convention selector selected extension convention candidate no-go "
            "theorem available now"
        ),
        "gate_b": (
            "Gate B = beyond-current-written-action selected extension convention "
            "selector selected extension convention representative promoted next"
        ),
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


def main() -> None:
    """Execute the selected-extension-convention-selector convention representative gate."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_convention_candidate_no_go_theorem_available_now"
        ]
        and prior_summary[
            "exact_minimal_selected_extension_convention_selector_selected_extension_convention_representative_requirement_theorem_available_now"
        ]
    )
    gate_b = bool(
        prior_summary[
            "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_convention_representative_primary_followup_required"
        ]
    )
    gate_c = False
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(prior_summary["failure_matrix_non_surrogate_guard_preserved"])
    convention_candidate_available = bool(
        prior_summary[
            "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_convention_candidate_available_now"
        ]
    )
    same_tag_reentry = bool(
        prior_summary["updated_pack_same_tag_pack_refresh_reentry_admissible_now"]
    )
    blind_blocked = bool(prior_summary["blind_vector_observable_gate_still_blocked"])
    pack_update_required_now = bool(gate_b)

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_convention_candidate_no_go_available_now",
            "pass" if gate_a else "reject",
            "Gate A exact beyond-current-written-action selected extension convention selector selected extension convention candidate no-go available now",
            sign_base.truth(gate_a),
            "The theorem stack now closes that convention-candidate choice still reduces to unresolved selector representative choice A_conv_ext.",
        ),
        sign_base.row(
            "gate_b_updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_convention_representative_promoted_next",
            "pass" if gate_b else "reject",
            "Gate B beyond-current-written-action selected extension convention selector selected extension convention representative promoted next",
            sign_base.truth(gate_b),
            "Once convention-candidate reduction closes, the honest next blocker is which representative convention on A_conv_ext could be concretely fixed.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "pass" if gate_c else "reject",
            "Gate C farther hybrid continuation reopen required now",
            sign_base.truth(gate_c),
            "Extra q-range remains reserve-only because the blocker is still theorem-side convention-representative completion.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This gate follows a real theorem closure and does not count same-tag restatement as progress.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Promoting the convention-representative route does not reopen the exhausted surrogate family.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_convention_candidate_available_now",
            "pass" if convention_candidate_available else "reject",
            "exact beyond-current-written-action selected extension convention selector selected extension convention candidate available now",
            sign_base.truth(convention_candidate_available),
            "The current theorem stack fixes only the candidate family and its reduction theorem, not one concrete representative convention.",
        ),
        sign_base.row(
            "same_tag_pack_refresh_reentry_admissible_now",
            "pass" if same_tag_reentry else "reject",
            "same-tag pack-refresh reentry admissible now",
            sign_base.truth(same_tag_reentry),
            "Same-tag pack-refresh reentry remains closed because the blocker is concrete representative choice, not bookkeeping syntax.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on one concrete selected extension.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "A new theorem object closed here, and the honest next blocker is convention-representative completion rather than same-tag route re-sync.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "gate_a_updated_pack_exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_convention_candidate_no_go_available_now": gate_a,
        "gate_b_updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_convention_representative_promoted_next": gate_b,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_convention_candidate_available_now": convention_candidate_available,
        "same_tag_pack_refresh_reentry_admissible_now": same_tag_reentry,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "pack_update_required_now": pack_update_required_now,
        "selected_primary_completion_lane": "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_convention_representative_theorem_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_convention_representative_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.4839",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_convention_representative_gate",
        "selected_followup_route_or_none": "8.7.56.4843",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.4837",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.4839",
                "followup_route": "8.7.56.4843",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_convention_representative_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        json.dumps(
            {
                "json": declaration_paths["json"],
                "classification": BRANCH_CLASS,
                "breakthrough_passed_now": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
