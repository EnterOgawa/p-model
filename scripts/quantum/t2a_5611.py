#!/usr/bin/env python3
"""Generate 8.7.56.5611-.5614 Trial-2 energy-partition variant gate artifacts."""

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
        "8.7.56.5607-5610",
        "updated_pack_trial2_energy_partition_variant_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5611-5614"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "energy-partition variant gate / conditional-hold refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_energy_partition_variant_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_energy_partition_variant_screen_interaction_total_over_harmonic_sq_"
    "front_runner_followup_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_energy_partition_variant_audited_interaction_total_over_harmonic_sq_"
    "front_runner_exact_relation_primary_conditional_hold_secondary_next"
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
    """Return formulas used by the energy-partition variant gate."""
    return {
        "gate_a": "Gate A = interaction_total_over_harmonic_sq remains the screened front runner",
        "gate_b": "Gate B = exact target-free route remains unavailable now",
        "gate_c": "Gate C = exact-relation audit is the honest next blocker",
    }


# 関数: `.5611-.5614` を実行する。

def main() -> None:
    """Execute the Trial-2 energy-partition variant gate."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(prior_summary["exact_trial2_energy_partition_variant_front_runner_selected_now"])
    gate_b = bool(not prior_summary["exact_trial2_energy_partition_variant_exact_route_available_now"])
    gate_c = bool(
        prior_summary[
            "updated_pack_trial2_energy_partition_variant_front_runner_exact_relation_primary_next_now"
        ]
    )
    trial2_energy_partition_variant_completed_now = bool(gate_a and gate_b)
    trial2_interaction_total_over_harmonic_sq_primary_next_now = bool(
        trial2_energy_partition_variant_completed_now and gate_c
    )
    trial2_conditional_hold_secondary_retained_now = True

    rows = [
        sign_base.row(
            "gate_a_updated_pack_trial2_energy_partition_variant_front_runner_selected_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack Trial-2 energy-partition variant front runner selected now",
            sign_base.truth(gate_a),
            "The reopen only stays honest if the blind screen keeps interaction_total_over_harmonic_sq as the unique best retained variant.",
        ),
        sign_base.row(
            "gate_b_updated_pack_trial2_energy_partition_variant_exact_route_unavailable_now",
            "pass" if gate_b else "reject",
            "gate B updated-pack Trial-2 energy-partition variant exact route unavailable now",
            sign_base.truth(gate_b),
            "The variant screen remains incomplete because the promoted front runner still has not been elevated into one exact target-free law.",
        ),
        sign_base.row(
            "gate_c_updated_pack_trial2_interaction_total_over_harmonic_sq_exact_relation_primary_next_now",
            "pass" if gate_c else "reject",
            "gate C updated-pack Trial-2 interaction_total_over_harmonic_sq exact relation primary next now",
            sign_base.truth(gate_c),
            "The next honest blocker is whether the promoted second-order interaction ratio can be written as one exact relation rather than a screened heuristic.",
        ),
        sign_base.row(
            "trial2_energy_partition_variant_completed_now",
            "pass" if trial2_energy_partition_variant_completed_now else "reject",
            "Trial-2 energy-partition variant completed now",
            sign_base.truth(trial2_energy_partition_variant_completed_now),
            "The blind variant screen is complete once the new front runner is fixed and the non-exact verdict is synchronized.",
        ),
        sign_base.row(
            "trial2_interaction_total_over_harmonic_sq_primary_next_now",
            "pass" if trial2_interaction_total_over_harmonic_sq_primary_next_now else "reject",
            "Trial-2 interaction_total_over_harmonic_sq primary next now",
            sign_base.truth(trial2_interaction_total_over_harmonic_sq_primary_next_now),
            "This front runner becomes the next active exact-relation lane because it improves the previous baseline materially.",
        ),
        sign_base.row(
            "trial2_conditional_hold_secondary_retained_now",
            "pass" if trial2_conditional_hold_secondary_retained_now else "reject",
            "Trial-2 conditional hold secondary retained now",
            sign_base.truth(trial2_conditional_hold_secondary_retained_now),
            "If the exact-relation audit dead-ends, the current pack should fall back to conditional hold rather than replaying exhausted routes.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "energy_partition_variant_front_runner_name": str(
            prior_summary["energy_partition_variant_front_runner_name"]
        ),
        "energy_partition_variant_front_runner_formula": str(
            prior_summary["energy_partition_variant_front_runner_formula"]
        ),
        "energy_partition_variant_front_runner_retained_value": float(
            prior_summary["energy_partition_variant_front_runner_retained_value"]
        ),
        "energy_partition_variant_front_runner_retained_rel_error_vs_target": float(
            prior_summary["energy_partition_variant_front_runner_retained_rel_error_vs_target"]
        ),
        "energy_partition_variant_front_runner_near_rel_error_vs_target": float(
            prior_summary["energy_partition_variant_front_runner_near_rel_error_vs_target"]
        ),
        "energy_partition_variant_front_runner_near_rel_shift_vs_retained": float(
            prior_summary["energy_partition_variant_front_runner_near_rel_shift_vs_retained"]
        ),
        "energy_partition_variant_front_runner_margin_vs_second": float(
            prior_summary["energy_partition_variant_front_runner_margin_vs_second"]
        ),
        "baseline_interaction_over_harmonic_retained_rel_error_vs_target": float(
            prior_summary["baseline_interaction_over_harmonic_retained_rel_error_vs_target"]
        ),
        "gate_a_updated_pack_trial2_energy_partition_variant_front_runner_selected_now": gate_a,
        "gate_b_updated_pack_trial2_energy_partition_variant_exact_route_unavailable_now": gate_b,
        "gate_c_updated_pack_trial2_interaction_total_over_harmonic_sq_exact_relation_primary_next_now": (
            gate_c
        ),
        "trial2_energy_partition_variant_completed_now": (
            trial2_energy_partition_variant_completed_now
        ),
        "trial2_interaction_total_over_harmonic_sq_primary_next_now": (
            trial2_interaction_total_over_harmonic_sq_primary_next_now
        ),
        "trial2_conditional_hold_secondary_retained_now": (
            trial2_conditional_hold_secondary_retained_now
        ),
        "selected_primary_completion_lane": (
            "trial2_interaction_total_over_harmonic_sq_exact_relation"
        ),
        "selected_secondary_completion_lane": "conditional_hold_only",
        "selected_reserve_completion_lane": "conditional_hold_only",
        "selected_next_generation_route": (
            "trial2_interaction_total_over_harmonic_sq_exact_relation"
        ),
        "recommended_next_route_or_none": "8.7.56.5615",
        "selected_followup_route": "trial2_interaction_total_over_harmonic_sq_exact_relation",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5613",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5615",
                "followup_route": "trial2_interaction_total_over_harmonic_sq_exact_relation",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_energy_partition_variant_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 energy-partition variant gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
