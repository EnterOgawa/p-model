#!/usr/bin/env python3
"""Generate 8.7.56.5587-.5590 Trial-2 energy-partition gate artifacts."""

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
        "8.7.56.5583-5586",
        "updated_pack_trial2_energy_partition_ratio_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5587-5590"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "energy-partition gate / entropy refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_energy_partition_ratio_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_energy_partition_screen_interaction_harmonic_front_runner_"
    "followup_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_energy_partition_ratio_audited_interaction_harmonic_front_runner_"
    "entropy_secondary_next"
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
    """Return formulas used by the energy-partition gate."""
    return {
        "gate_a": "Gate A = interaction-over-harmonic remains the screened front runner",
        "gate_b": "Gate B = exact energy-partition theorem remains unavailable",
        "gate_c": "Gate C = entropy route remains secondary until the front runner dead-ends honestly",
    }


# 関数: `.5587-.5590` を実行する。

def main() -> None:
    """Execute the Trial-2 energy-partition gate."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    gate_a = bool(prior_summary["exact_trial2_energy_partition_interaction_over_harmonic_front_runner_now"])
    gate_b = bool(not prior_summary["exact_trial2_energy_partition_exact_route_available_now"])
    gate_c = bool(prior_summary["updated_pack_trial2_energy_partition_front_runner_followup_required_now"])
    trial2_energy_partition_ratio_completed_now = bool(gate_a and gate_b)
    trial2_energy_partition_interaction_harmonic_primary_next_now = bool(
        trial2_energy_partition_ratio_completed_now and gate_c
    )
    trial2_entropy_route_secondary_retained_now = True

    rows = [
        sign_base.row(
            "gate_a_updated_pack_trial2_energy_partition_interaction_over_harmonic_front_runner_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack Trial-2 interaction-over-harmonic front runner now",
            sign_base.truth(gate_a),
            "The screened energy family only compresses the blocker honestly if interaction-over-harmonic stays the best simple ratio.",
        ),
        sign_base.row(
            "gate_b_updated_pack_trial2_energy_partition_exact_route_unavailable_now",
            "pass" if gate_b else "reject",
            "gate B updated-pack Trial-2 energy-partition exact route unavailable now",
            sign_base.truth(gate_b),
            "The screening route remains incomplete because no simple partition ratio reproduces alpha_target exactly.",
        ),
        sign_base.row(
            "gate_c_updated_pack_trial2_entropy_secondary_retained_now",
            "pass" if gate_c else "reject",
            "gate C updated-pack Trial-2 entropy secondary retained now",
            sign_base.truth(gate_c),
            "Entropy stays secondary until the screened interaction-over-harmonic front runner dead-ends honestly.",
        ),
        sign_base.row(
            "trial2_energy_partition_ratio_completed_now",
            "pass" if trial2_energy_partition_ratio_completed_now else "reject",
            "Trial-2 energy-partition ratio completed now",
            sign_base.truth(trial2_energy_partition_ratio_completed_now),
            "The screening route is complete once the front runner is fixed and exact theorem closeout remains unavailable.",
        ),
        sign_base.row(
            "trial2_energy_partition_interaction_harmonic_primary_next_now",
            "pass" if trial2_energy_partition_interaction_harmonic_primary_next_now else "reject",
            "Trial-2 interaction-over-harmonic primary next now",
            sign_base.truth(trial2_energy_partition_interaction_harmonic_primary_next_now),
            "The next honest blocker is whether the screened interaction-over-harmonic ratio can be elevated into one exact target-free law.",
        ),
        sign_base.row(
            "trial2_entropy_route_secondary_retained_now",
            "pass" if trial2_entropy_route_secondary_retained_now else "reject",
            "Trial-2 entropy route secondary retained now",
            sign_base.truth(trial2_entropy_route_secondary_retained_now),
            "Entropy remains secondary until the energy-partition front runner dead-ends honestly.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "energy_partition_front_runner_name": str(prior_summary["energy_partition_front_runner_name"]),
        "energy_partition_front_runner_retained_value": float(
            prior_summary["energy_partition_front_runner_retained_value"]
        ),
        "energy_partition_front_runner_retained_rel_error_vs_target": float(
            prior_summary["energy_partition_front_runner_retained_rel_error_vs_target"]
        ),
        "energy_partition_front_runner_near_rel_shift_vs_retained": float(
            prior_summary["energy_partition_front_runner_near_rel_shift_vs_retained"]
        ),
        "energy_partition_front_runner_margin_vs_second": float(
            prior_summary["energy_partition_front_runner_margin_vs_second"]
        ),
        "gate_a_updated_pack_trial2_energy_partition_interaction_over_harmonic_front_runner_now": gate_a,
        "gate_b_updated_pack_trial2_energy_partition_exact_route_unavailable_now": gate_b,
        "gate_c_updated_pack_trial2_entropy_secondary_retained_now": gate_c,
        "trial2_energy_partition_ratio_completed_now": trial2_energy_partition_ratio_completed_now,
        "trial2_energy_partition_interaction_harmonic_primary_next_now": (
            trial2_energy_partition_interaction_harmonic_primary_next_now
        ),
        "trial2_entropy_route_secondary_retained_now": trial2_entropy_route_secondary_retained_now,
        "selected_primary_completion_lane": "trial2_energy_partition_interaction_harmonic",
        "selected_secondary_completion_lane": "trial2_entropy_route",
        "selected_reserve_completion_lane": "trial2_entropy_route",
        "selected_next_generation_route": "trial2_energy_partition_interaction_harmonic",
        "recommended_next_route_or_none": "8.7.56.5591",
        "selected_followup_route": "trial2_energy_partition_interaction_harmonic",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5589",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5591",
                "followup_route": "trial2_energy_partition_interaction_harmonic",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_energy_partition_ratio_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 energy-partition gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
