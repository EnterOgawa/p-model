#!/usr/bin/env python3
"""Generate 8.7.56.4571-.4574 matter/rotation completion gate artifacts."""

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
        "8.7.56.4567-4570",
        "updated_pack_current_written_action_matter_rotation_completion_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
STEP_TAG = "8.7.56.4571-4574"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack current "
    "written-action matter/rotation completion gate / route refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_current_written_action_matter_rotation_completion_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "current_written_action_matter_rotation_completion_no_go_theorem_derived_"
    "beyond_current_written_action_primary_pack_refresh_secondary_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "current_written_action_matter_rotation_completion_no_go_theorem_derived_"
    "beyond_current_written_action_primary_hybrid_reserve_secondary_next"
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
    """Return formulas used in the matter/rotation completion gate."""
    return {
        "gate_a": "Gate A = current written-action matter/rotation completion no-go theorem available now",
        "gate_b": "Gate B = beyond-current-written-action probe extension promoted next",
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.4571-.4574` を実行する。

def main() -> None:
    """Execute the current written-action matter/rotation completion gate / route refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_current_written_action_matter_rotation_completion_no_go_theorem_available_now"
        ]
    )
    gate_b = bool(
        prior_summary[
            "updated_pack_beyond_current_written_action_probe_extension_primary_followup_required"
        ]
    )
    gate_c = False
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(prior_summary["failure_matrix_non_surrogate_guard_preserved"])
    blind_blocked = bool(prior_summary["blind_vector_observable_gate_still_blocked"])
    old_retry = False
    pack_update_required = bool(gate_b)
    exact_current_written_action_matter_rotation_completion_available_now = bool(
        prior_summary["exact_current_written_action_matter_rotation_completion_available_now"]
    )
    same_tag_pack_refresh_reentry_admissible_now = bool(
        prior_summary["updated_pack_same_tag_pack_refresh_reentry_admissible_now"]
    )

    rows = [
        sign_base.row(
            "gate_a_updated_pack_current_written_action_matter_rotation_completion_no_go_available_now",
            "pass" if gate_a else "reject",
            "Gate A current written-action matter/rotation completion no-go available now",
            sign_base.truth(gate_a),
            "The current written note now closes theorem-side as unable to rescue probe structure through its deferred matter/rotation sector.",
        ),
        sign_base.row(
            "gate_b_updated_pack_beyond_current_written_action_probe_extension_promoted_next",
            "pass" if gate_b else "reject",
            "Gate B beyond-current-written-action probe extension promoted next",
            sign_base.truth(gate_b),
            "Once the current written note is exhausted theorem-side, the honest next blocker is whether an extension beyond the present written action can supply the missing probe structure.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "pass" if gate_c else "reject",
            "Gate C farther hybrid continuation reopen required now",
            sign_base.truth(gate_c),
            "Extra q-range remains reserve-only because the blocker is still theorem-side action extension rather than continuation range.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This gate follows a real theorem closure and does not count same-tag route restatement as progress.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Promoting the beyond-current-written-action extension theorem does not reopen the exhausted surrogate family.",
        ),
        sign_base.row(
            "exact_current_written_action_matter_rotation_completion_available_now",
            "pass" if exact_current_written_action_matter_rotation_completion_available_now else "reject",
            "exact current written-action matter/rotation completion available now",
            sign_base.truth(exact_current_written_action_matter_rotation_completion_available_now),
            "The current written note still does not supply a successful matter/rotation completion theorem for probe structure.",
        ),
        sign_base.row(
            "same_tag_pack_refresh_reentry_admissible_now",
            "pass" if same_tag_pack_refresh_reentry_admissible_now else "reject",
            "same-tag pack-refresh reentry admissible now",
            sign_base.truth(same_tag_pack_refresh_reentry_admissible_now),
            "Pack-refresh reentry stays closed because the remaining blocker is beyond-current-note action extension, not bookkeeping repetition.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on a theorem-side route beyond the exhaustion of the current written note.",
        ),
        sign_base.row(
            "old_density_proxy_eigenvalue_retry_admissible_now",
            "pass" if old_retry else "reject",
            "old density/proxy/eigenvalue retry admissible now",
            sign_base.truth(old_retry),
            "The matter/rotation completion gate keeps the exhausted surrogate retry family closed.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required else "reject",
            "updated-pack substantive pack update required now",
            sign_base.truth(pack_update_required),
            "A real theorem object closed here, but the remaining blocker has moved beyond the current written note itself.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "gate_a_updated_pack_current_written_action_matter_rotation_completion_no_go_available_now": gate_a,
        "gate_b_updated_pack_beyond_current_written_action_probe_extension_promoted_next": gate_b,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_current_written_action_matter_rotation_completion_available_now": exact_current_written_action_matter_rotation_completion_available_now,
        "same_tag_pack_refresh_reentry_admissible_now": same_tag_pack_refresh_reentry_admissible_now,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "old_density_proxy_eigenvalue_retry_admissible_now": old_retry,
        "pack_update_required_now": pack_update_required,
        "selected_primary_completion_lane": "updated_pack_beyond_current_written_action_probe_extension_theorem_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_probe_extension_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.4575",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_probe_extension_gate",
        "selected_followup_route_or_none": "8.7.56.4579",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.4573",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.4575",
                "followup_route": "8.7.56.4579",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_current_written_action_matter_rotation_completion_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack current written-action matter/rotation completion gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
