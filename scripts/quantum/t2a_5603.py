#!/usr/bin/env python3
"""Generate 8.7.56.5603-.5606 Trial-2 entropy gate artifacts."""

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
        "8.7.56.5599-5602",
        "updated_pack_trial2_entropy_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5603-5606"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor Trial-2 entropy gate / conditional-hold refresh"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_entropy_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_entropy_negative_closeout_completed_conditional_reopen_only_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_entropy_negative_closeout_completed_all_promoted_direct_alpha_routes_"
    "exhausted_conditional_hold_next"
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
    """Return formulas used by the entropy gate."""
    return {
        "gate_a": "Gate A = entropy negative closeout available now",
        "gate_b": "Gate B = all promoted direct-alpha routes exhausted now",
        "gate_c": "Gate C = conditional hold is the honest current state",
    }


# 関数: `.5603-.5606` を実行する。

def main() -> None:
    """Execute the Trial-2 entropy gate / conditional-hold refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(prior_summary["exact_trial2_entropy_negative_closeout_available_now"])
    gate_b = bool(
        gate_a
        and not prior_summary["exact_trial2_entropy_alpha_exact_route_available_now"]
        and not prior_summary["exact_trial2_entropy_form_factor_exact_route_available_now"]
    )
    gate_c = bool(gate_b)
    trial2_entropy_route_completed_now = bool(gate_b)
    trial2_conditional_hold_reinstated_now = bool(gate_c)

    rows = [
        sign_base.row(
            "gate_a_updated_pack_trial2_entropy_negative_closeout_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack Trial-2 entropy negative closeout available now",
            sign_base.truth(gate_a),
            "The entropy route is complete once both alpha and form-factor entropy candidates are fixed as unavailable.",
        ),
        sign_base.row(
            "gate_b_updated_pack_trial2_all_promoted_direct_alpha_routes_exhausted_now",
            "pass" if gate_b else "reject",
            "gate B updated-pack Trial-2 all promoted direct-alpha routes exhausted now",
            sign_base.truth(gate_b),
            "With entropy also closed negatively, no promoted low-cost direct-alpha route remains live in the current pack.",
        ),
        sign_base.row(
            "gate_c_updated_pack_trial2_conditional_hold_reinstated_now",
            "pass" if gate_c else "reject",
            "gate C updated-pack Trial-2 conditional hold reinstated now",
            sign_base.truth(gate_c),
            "The honest state returns to conditional hold until a genuinely new theorem route or selected-extension-native computation branch is promoted.",
        ),
        sign_base.row(
            "trial2_entropy_route_completed_now",
            "pass" if trial2_entropy_route_completed_now else "reject",
            "Trial-2 entropy route completed now",
            sign_base.truth(trial2_entropy_route_completed_now),
            "The promoted entropy lane is now fully audited and synchronized.",
        ),
        sign_base.row(
            "trial2_conditional_hold_reinstated_now",
            "pass" if trial2_conditional_hold_reinstated_now else "reject",
            "Trial-2 conditional hold reinstated now",
            sign_base.truth(trial2_conditional_hold_reinstated_now),
            "No unconditional official branch remains once all promoted direct-alpha routes are exhausted again.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_shannon_entropy": float(prior_summary["retained_shannon_entropy"]),
        "retained_alpha_from_entropy_rel_error_vs_target": float(
            prior_summary["retained_alpha_from_entropy_rel_error_vs_target"]
        ),
        "retained_form_factor_from_entropy_rel_error_vs_exact": float(
            prior_summary["retained_form_factor_from_entropy_rel_error_vs_exact"]
        ),
        "gate_a_updated_pack_trial2_entropy_negative_closeout_available_now": gate_a,
        "gate_b_updated_pack_trial2_all_promoted_direct_alpha_routes_exhausted_now": gate_b,
        "gate_c_updated_pack_trial2_conditional_hold_reinstated_now": gate_c,
        "trial2_entropy_route_completed_now": trial2_entropy_route_completed_now,
        "trial2_conditional_hold_reinstated_now": trial2_conditional_hold_reinstated_now,
        "selected_primary_completion_lane": "conditional_reopen_only",
        "selected_secondary_completion_lane": "conditional_reopen_only",
        "selected_reserve_completion_lane": "conditional_reopen_only",
        "selected_next_generation_route": "conditional_reopen_only",
        "recommended_next_route_or_none": None,
        "selected_followup_route": "conditional_reopen_only",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5605",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
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
            "overall_status": "trial2_entropy_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 entropy gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
