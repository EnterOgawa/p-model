#!/usr/bin/env python3
"""Generate 8.7.56.5483-.5486 Trial-2 conditional reopen hold gate artifacts."""

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
        "8.7.56.5479-5482",
        "updated_pack_trial2_conditional_reopen_inventory_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5483-5486"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "conditional reopen hold gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_conditional_reopen_hold_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_conditional_reopen_inventory_audited_no_current_trigger_"
    "hold_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_conditional_reopen_inventory_audited_no_current_trigger_"
    "hold_next"
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


# 関数: hold gate で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the conditional reopen hold gate."""
    return {
        "gate_a": "Gate A = conditional reopen inventory available now",
        "gate_b": "Gate B = no currently admissible reopen trigger detected now",
        "gate_c": "Gate C = unconditional reopen required now",
    }


# 関数: `.5483-.5486` を実行する。

def main() -> None:
    """Execute the Trial-2 conditional reopen hold gate."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    gate_a = bool(prior_summary["exact_trial2_conditional_reopen_inventory_available_now"])
    gate_b = bool(prior_summary["trial2_conditional_reopen_no_current_trigger_detected_now"])
    gate_c = False
    conditional_hold_completed_now = bool(gate_a and gate_b)
    no_unconditional_next_route_now = bool(conditional_hold_completed_now)
    future_reopen_requires_new_target_free_theorem_route_or_new_independent_source_now = True

    rows = [
        sign_base.row(
            "gate_a_updated_pack_trial2_conditional_reopen_inventory_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack Trial-2 conditional reopen inventory available now",
            sign_base.truth(gate_a),
            "The conditional reopen inventory has already enumerated the only admissible trigger classes.",
        ),
        sign_base.row(
            "gate_b_updated_pack_trial2_no_current_reopen_trigger_detected_now",
            "pass" if gate_b else "reject",
            "gate B updated-pack Trial-2 no current reopen trigger detected now",
            sign_base.truth(gate_b),
            "The current pack contains no actually materialized admissible reopen trigger.",
        ),
        sign_base.row(
            "gate_c_unconditional_reopen_required_now",
            "reject",
            "gate C unconditional reopen required now",
            0.0,
            "No unconditional reopen is justified after the inventory audit.",
        ),
        sign_base.row(
            "trial2_conditional_reopen_hold_completed_now",
            "pass" if conditional_hold_completed_now else "reject",
            "Trial-2 conditional reopen hold completed now",
            sign_base.truth(conditional_hold_completed_now),
            "The honest final hold reading is now promoted: keep reserve-only until a genuinely new trigger appears.",
        ),
        sign_base.row(
            "no_unconditional_next_route_now",
            "pass" if no_unconditional_next_route_now else "reject",
            "no unconditional next route now",
            sign_base.truth(no_unconditional_next_route_now),
            "There is still no automatic next branch inside the current pack.",
        ),
        sign_base.row(
            "future_reopen_requires_new_target_free_theorem_route_or_new_independent_source_now",
            "pass"
            if future_reopen_requires_new_target_free_theorem_route_or_new_independent_source_now
            else "reject",
            "future reopen requires new target-free theorem route or new independent source now",
            sign_base.truth(
                future_reopen_requires_new_target_free_theorem_route_or_new_independent_source_now
            ),
            "Only genuinely new theorem or selected-extension-native source/computation routes can justify reopening.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_blind_over_m0": float(prior_summary["q_blind_over_m0"]),
        "q_exact_over_m0": float(prior_summary["q_exact_over_m0"]),
        "q_star_over_m0": float(prior_summary["q_star_over_m0"]),
        "delta_q_over_q_star": float(prior_summary["delta_q_over_q_star"]),
        "gate_a_updated_pack_trial2_conditional_reopen_inventory_available_now": gate_a,
        "gate_b_updated_pack_trial2_no_current_reopen_trigger_detected_now": gate_b,
        "gate_c_unconditional_reopen_required_now": gate_c,
        "trial2_conditional_reopen_hold_completed_now": conditional_hold_completed_now,
        "no_unconditional_next_route_now": no_unconditional_next_route_now,
        "future_reopen_requires_new_target_free_theorem_route_or_new_independent_source_now": (
            future_reopen_requires_new_target_free_theorem_route_or_new_independent_source_now
        ),
        "selected_primary_completion_lane": "conditional_reopen_only",
        "selected_secondary_completion_lane": (
            "new_target_free_theorem_route_or_new_independent_source"
        ),
        "selected_reserve_completion_lane": (
            "farther_hybrid_reserve_only_until_genuinely_new_route_exists"
        ),
        "selected_next_generation_route": None,
        "recommended_next_route_or_none": None,
        "selected_followup_route": "conditional_reopen_only",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5485",
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
            "overall_status": "trial2_conditional_reopen_hold_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 conditional reopen hold completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から conditional reopen hold gate を実行する。

if __name__ == "__main__":
    main()
