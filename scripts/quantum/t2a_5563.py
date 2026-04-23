#!/usr/bin/env python3
"""Generate 8.7.56.5563-.5566 Trial-2 Ward/current-algebra gate artifacts."""

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
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5559-5562",
        "updated_pack_trial2_ward_current_algebra_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5563-5566"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "Ward / current algebra gate / conditional-reopen refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_ward_current_algebra_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_ward_current_algebra_target_free_readout_missing_"
    "conditional_reopen_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_ward_current_algebra_negative_closeout_completed_"
    "conditional_reopen_only_next"
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
    """Return formulas used by the Ward/current-algebra gate."""
    return {
        "gate_a": "Gate A = Ward/current-algebra background/current no-go stack available",
        "gate_b": "Gate B = Ward/current-algebra negative closeout completed now",
        "gate_c": "Gate C = conditional reopen restored now",
    }


# 関数: `.5563-.5566` を実行する。

def main() -> None:
    """Execute the Trial-2 Ward/current-algebra gate."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_trial2_ward_current_algebra_background_noether_current_available_now"
        ]
        and prior_summary[
            "exact_trial2_ward_current_algebra_same_field_source_no_go_available_now"
        ]
    )
    gate_b = bool(
        prior_summary[
            "exact_trial2_ward_current_algebra_negative_closeout_available_now"
        ]
    )
    gate_c = bool(
        prior_summary["updated_pack_trial2_conditional_reopen_refresh_required_now"]
    )
    trial2_ward_current_algebra_negative_closeout_completed_now = bool(gate_a and gate_b)
    conditional_reopen_only_next_now = bool(
        trial2_ward_current_algebra_negative_closeout_completed_now and gate_c
    )
    no_unconditional_next_official_branch_now = bool(conditional_reopen_only_next_now)

    rows = [
        sign_base.row(
            "gate_a_updated_pack_trial2_ward_current_algebra_current_stack_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack Trial-2 Ward/current-algebra current stack available now",
            sign_base.truth(gate_a),
            "The current pack still retains the old exact background Noether-current theorem and the same-field source no-go theorem.",
        ),
        sign_base.row(
            "gate_b_updated_pack_trial2_ward_current_algebra_negative_closeout_completed_now",
            "pass" if gate_b else "reject",
            "gate B updated-pack Trial-2 Ward/current-algebra negative closeout completed now",
            sign_base.truth(gate_b),
            "The current selected-extension pack still materializes no independent Ward/current-algebra alpha readout.",
        ),
        sign_base.row(
            "gate_c_updated_pack_trial2_conditional_reopen_restored_now",
            "pass" if gate_c else "reject",
            "gate C updated-pack Trial-2 conditional reopen restored now",
            sign_base.truth(gate_c),
            "Once Ward/current algebra also closes negatively, the honest next state is conditional reopen only.",
        ),
        sign_base.row(
            "trial2_ward_current_algebra_negative_closeout_completed_now",
            "pass" if trial2_ward_current_algebra_negative_closeout_completed_now else "reject",
            "Trial-2 Ward/current-algebra negative closeout completed now",
            sign_base.truth(trial2_ward_current_algebra_negative_closeout_completed_now),
            "The Ward/current-algebra route has now closed honestly under the current pack.",
        ),
        sign_base.row(
            "conditional_reopen_only_next_now",
            "pass" if conditional_reopen_only_next_now else "reject",
            "conditional reopen only next now",
            sign_base.truth(conditional_reopen_only_next_now),
            "The honest followup is conditional reopen only; no new current-pack route remains promoted.",
        ),
        sign_base.row(
            "no_unconditional_next_official_branch_now",
            "pass" if no_unconditional_next_official_branch_now else "reject",
            "no unconditional next official branch now",
            sign_base.truth(no_unconditional_next_official_branch_now),
            "All currently promoted reopen routes are exhausted, so the pack returns to hold rather than spawning a new official branch.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "soft_alpha_naive": float(prior_summary["soft_alpha_naive"]),
        "alpha_target": float(prior_summary["alpha_target"]),
        "soft_alpha_target_relative_mismatch": float(
            prior_summary["soft_alpha_target_relative_mismatch"]
        ),
        "gate_a_updated_pack_trial2_ward_current_algebra_current_stack_available_now": gate_a,
        "gate_b_updated_pack_trial2_ward_current_algebra_negative_closeout_completed_now": gate_b,
        "gate_c_updated_pack_trial2_conditional_reopen_restored_now": gate_c,
        "trial2_ward_current_algebra_negative_closeout_completed_now": (
            trial2_ward_current_algebra_negative_closeout_completed_now
        ),
        "conditional_reopen_only_next_now": conditional_reopen_only_next_now,
        "no_unconditional_next_official_branch_now": (
            no_unconditional_next_official_branch_now
        ),
        "selected_primary_completion_lane": "conditional_reopen_only",
        "selected_secondary_completion_lane": "new_target_free_theorem_route_only",
        "selected_reserve_completion_lane": "new_selected_extension_native_source_only",
        "selected_next_generation_route": "conditional_reopen_only",
        "recommended_next_route_or_none": None,
        "selected_followup_route": None,
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5565",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": None,
                "followup_route": "conditional_reopen_only",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_ward_current_algebra_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 Ward/current-algebra gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から Ward/current-algebra gate を実行する。

if __name__ == "__main__":
    main()
