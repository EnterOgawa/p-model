#!/usr/bin/env python3
"""Generate 8.7.56.5531-.5534 Trial-2 fresh-pattern Round-1 gate artifacts."""

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
        "8.7.56.5527-5530",
        "updated_pack_trial2_fresh_pattern_round1_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5531-5534"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor fresh pattern Round-1 gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_fresh_pattern_round1_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_fresh_pattern_round1_audited_bohr_radius_front_runner_"
    "full_spectral_jost_secondary_scattering_thomson_reserve_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_fresh_pattern_round1_audited_bohr_radius_front_runner_"
    "full_spectral_jost_secondary_scattering_thomson_reserve_next"
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


# 関数: fresh-pattern gate で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the fresh-pattern Round-1 gate."""
    return {
        "gate_a": "Gate A = fresh-pattern Round-1 inventory available now",
        "gate_b": "Gate B = pattern epsilon promoted as front-runner",
        "gate_c": "Gate C = unconditional Jost replay required now",
    }


# 関数: `.5531-.5534` を実行する。

def main() -> None:
    """Execute the Trial-2 fresh-pattern Round-1 gate."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    gate_a = bool(prior_summary["trial2_fresh_pattern_round1_inventory_nonempty_now"])
    gate_b = bool(prior_summary["trial2_pattern_epsilon_front_runner_now"])
    gate_c = False
    trial2_fresh_pattern_round1_gate_completed_now = bool(gate_a and gate_b)
    trial2_pattern_epsilon_primary_next_now = bool(
        trial2_fresh_pattern_round1_gate_completed_now
    )
    trial2_full_spectral_jost_secondary_retained_now = True
    trial2_scattering_thomson_reserve_retained_now = True
    trial2_ward_current_algebra_deeper_reserve_retained_now = True

    rows = [
        sign_base.row(
            "gate_a_updated_pack_trial2_fresh_pattern_round1_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack Trial-2 fresh-pattern Round-1 available now",
            sign_base.truth(gate_a),
            "The expert fresh-pattern screen is available and nonempty.",
        ),
        sign_base.row(
            "gate_b_updated_pack_trial2_pattern_epsilon_front_runner_promoted_now",
            "pass" if gate_b else "reject",
            "gate B updated-pack Trial-2 pattern epsilon front-runner promoted now",
            sign_base.truth(gate_b),
            "Bohr / Compton matching is the only Round-1 pattern that improves on q_star.",
        ),
        sign_base.row(
            "gate_c_unconditional_jost_replay_required_now",
            "reject",
            "gate C unconditional Jost replay required now",
            0.0,
            "The fresh epsilon pattern is screened first; Jost stays secondary rather than replayed immediately.",
        ),
        sign_base.row(
            "trial2_fresh_pattern_round1_gate_completed_now",
            "pass" if trial2_fresh_pattern_round1_gate_completed_now else "reject",
            "Trial-2 fresh-pattern Round-1 gate completed now",
            sign_base.truth(trial2_fresh_pattern_round1_gate_completed_now),
            "The fresh-pattern reprioritization is now official and machine-readable.",
        ),
        sign_base.row(
            "trial2_pattern_epsilon_primary_next_now",
            "pass" if trial2_pattern_epsilon_primary_next_now else "reject",
            "Trial-2 pattern epsilon primary next now",
            sign_base.truth(trial2_pattern_epsilon_primary_next_now),
            "The next honest blocker is why the 1/8 Bohr / Compton ratio should hold target-free.",
        ),
        sign_base.row(
            "trial2_full_spectral_jost_secondary_retained_now",
            "pass" if trial2_full_spectral_jost_secondary_retained_now else "reject",
            "Trial-2 full spectral / Jost secondary retained now",
            sign_base.truth(trial2_full_spectral_jost_secondary_retained_now),
            "The heavier Jost route is retained as secondary rather than discarded.",
        ),
        sign_base.row(
            "trial2_scattering_thomson_reserve_retained_now",
            "pass" if trial2_scattering_thomson_reserve_retained_now else "reject",
            "Trial-2 scattering / Thomson reserve retained now",
            sign_base.truth(trial2_scattering_thomson_reserve_retained_now),
            "The scattering route stays available but does not preempt the surviving low-cost fresh pattern.",
        ),
        sign_base.row(
            "trial2_ward_current_algebra_deeper_reserve_retained_now",
            "pass" if trial2_ward_current_algebra_deeper_reserve_retained_now else "reject",
            "Trial-2 Ward / current algebra deeper reserve retained now",
            sign_base.truth(trial2_ward_current_algebra_deeper_reserve_retained_now),
            "Ward/current-algebra remains deeper reserve until epsilon and Jost both dead-end.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_exact_over_m0": float(prior_summary["q_exact_over_m0"]),
        "q_star_over_m0": float(prior_summary["q_star_over_m0"]),
        "q_bohr_over_m0": float(prior_summary["q_bohr_over_m0"]),
        "q_bohr_relative_error_vs_q_exact": float(
            prior_summary["q_bohr_relative_error_vs_q_exact"]
        ),
        "gate_a_updated_pack_trial2_fresh_pattern_round1_available_now": gate_a,
        "gate_b_updated_pack_trial2_pattern_epsilon_front_runner_promoted_now": gate_b,
        "gate_c_unconditional_jost_replay_required_now": gate_c,
        "trial2_fresh_pattern_round1_gate_completed_now": (
            trial2_fresh_pattern_round1_gate_completed_now
        ),
        "trial2_pattern_epsilon_primary_next_now": trial2_pattern_epsilon_primary_next_now,
        "trial2_full_spectral_jost_secondary_retained_now": (
            trial2_full_spectral_jost_secondary_retained_now
        ),
        "trial2_scattering_thomson_reserve_retained_now": (
            trial2_scattering_thomson_reserve_retained_now
        ),
        "trial2_ward_current_algebra_deeper_reserve_retained_now": (
            trial2_ward_current_algebra_deeper_reserve_retained_now
        ),
        "selected_primary_completion_lane": "pattern_epsilon_bohr_radius_matching",
        "selected_secondary_completion_lane": "full_spectral_jost",
        "selected_reserve_completion_lane": "scattering_thomson_then_ward_current_algebra",
        "selected_next_generation_route": (
            "trial2_numeric_alpha_vector_qball_form_factor_pattern_epsilon_"
            "bohr_radius_matching_primary"
        ),
        "recommended_next_route_or_none": "8.7.56.5535",
        "selected_followup_route": "pattern_epsilon_bohr_radius_matching",
        "selected_followup_route_or_none": "8.7.56.5539",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5533",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5535",
                "followup_route": "8.7.56.5539",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_fresh_pattern_round1_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 fresh-pattern Round-1 gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から fresh-pattern Round-1 gate を実行する。

if __name__ == "__main__":
    main()
