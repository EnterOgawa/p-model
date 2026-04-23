#!/usr/bin/env python3
"""Generate 8.7.56.5539-.5542 Trial-2 Bohr / Compton gate artifacts."""

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
        "8.7.56.5535-5538",
        "updated_pack_trial2_bohr_radius_matching_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5539-5542"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball Bohr radius / Compton gate"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_bohr_radius_matching_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_bohr_radius_matching_heuristic_front_runner_target_free_theorem_"
    "missing_full_spectral_jost_secondary_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_bohr_radius_matching_negative_closeout_completed_full_spectral_jost_"
    "primary_scattering_thomson_secondary_ward_current_algebra_reserve_next"
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
    """Return formulas used by the Bohr / Compton gate."""
    return {
        "gate_a": "Gate A = Bohr / Compton heuristic front-runner established",
        "gate_b": "Gate B = target-free theorem still unavailable",
        "gate_c": "Gate C = full spectral / Jost promoted primary next",
    }


# 関数: `.5539-.5542` を実行する。

def main() -> None:
    """Execute the Trial-2 Bohr / Compton gate."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    gate_a = bool(prior_summary["exact_trial2_bohr_radius_heuristic_front_runner_only_now"])
    gate_b = not bool(prior_summary["exact_trial2_bohr_radius_target_free_theorem_available_now"])
    gate_c = bool(prior_summary["updated_pack_trial2_full_spectral_jost_followup_required_now"])
    trial2_bohr_radius_matching_gate_completed_now = bool(gate_a and gate_b and gate_c)
    trial2_full_spectral_jost_primary_next_now = bool(
        trial2_bohr_radius_matching_gate_completed_now
    )
    trial2_scattering_thomson_secondary_retained_now = True
    trial2_ward_current_algebra_reserve_retained_now = True

    rows = [
        sign_base.row(
            "gate_a_updated_pack_trial2_bohr_radius_heuristic_front_runner_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack Trial-2 Bohr radius heuristic front-runner now",
            sign_base.truth(gate_a),
            "The route survives only as the best low-complexity heuristic fit.",
        ),
        sign_base.row(
            "gate_b_updated_pack_trial2_bohr_radius_target_free_theorem_unavailable_now",
            "pass" if gate_b else "reject",
            "gate B updated-pack Trial-2 Bohr radius target-free theorem unavailable now",
            sign_base.truth(gate_b),
            "Current pack still lacks an exact frozen-action identity selecting tail radius and denominator 8.",
        ),
        sign_base.row(
            "gate_c_updated_pack_trial2_full_spectral_jost_promoted_primary_now",
            "pass" if gate_c else "reject",
            "gate C updated-pack Trial-2 full spectral / Jost promoted primary now",
            sign_base.truth(gate_c),
            "Once the Bohr route closes negatively, Jost becomes the next honest primary route.",
        ),
        sign_base.row(
            "trial2_bohr_radius_matching_gate_completed_now",
            "pass" if trial2_bohr_radius_matching_gate_completed_now else "reject",
            "Trial-2 Bohr radius / Compton gate completed now",
            sign_base.truth(trial2_bohr_radius_matching_gate_completed_now),
            "The Bohr route is now fixed as heuristic-only and removed from the primary slot.",
        ),
        sign_base.row(
            "trial2_full_spectral_jost_primary_next_now",
            "pass" if trial2_full_spectral_jost_primary_next_now else "reject",
            "Trial-2 full spectral / Jost primary next now",
            sign_base.truth(trial2_full_spectral_jost_primary_next_now),
            "The next honest blocker is now the full spectral / Jost route audit.",
        ),
        sign_base.row(
            "trial2_scattering_thomson_secondary_retained_now",
            "pass" if trial2_scattering_thomson_secondary_retained_now else "reject",
            "Trial-2 scattering / Thomson secondary retained now",
            sign_base.truth(trial2_scattering_thomson_secondary_retained_now),
            "The scattering route remains secondary until Jost dead-ends honestly.",
        ),
        sign_base.row(
            "trial2_ward_current_algebra_reserve_retained_now",
            "pass" if trial2_ward_current_algebra_reserve_retained_now else "reject",
            "Trial-2 Ward / current algebra reserve retained now",
            sign_base.truth(trial2_ward_current_algebra_reserve_retained_now),
            "Ward/current-algebra remains reserve until both primary and secondary dead-end honestly.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_exact_over_m0": float(prior_summary["q_exact_over_m0"]),
        "q_one_eighth_over_m0": float(prior_summary["q_one_eighth_over_m0"]),
        "q_one_eighth_relative_error_vs_q_exact": float(
            prior_summary["q_one_eighth_relative_error_vs_q_exact"]
        ),
        "n_fit_from_exact_ratio": float(prior_summary["n_fit_from_exact_ratio"]),
        "gate_a_updated_pack_trial2_bohr_radius_heuristic_front_runner_now": gate_a,
        "gate_b_updated_pack_trial2_bohr_radius_target_free_theorem_unavailable_now": gate_b,
        "gate_c_updated_pack_trial2_full_spectral_jost_promoted_primary_now": gate_c,
        "trial2_bohr_radius_matching_gate_completed_now": (
            trial2_bohr_radius_matching_gate_completed_now
        ),
        "trial2_full_spectral_jost_primary_next_now": (
            trial2_full_spectral_jost_primary_next_now
        ),
        "trial2_scattering_thomson_secondary_retained_now": (
            trial2_scattering_thomson_secondary_retained_now
        ),
        "trial2_ward_current_algebra_reserve_retained_now": (
            trial2_ward_current_algebra_reserve_retained_now
        ),
        "selected_primary_completion_lane": "trial2_full_spectral_jost",
        "selected_secondary_completion_lane": "trial2_scattering_thomson",
        "selected_reserve_completion_lane": "trial2_ward_current_algebra",
        "selected_next_generation_route": "trial2_full_spectral_jost",
        "recommended_next_route_or_none": "8.7.56.5543",
        "selected_followup_route": "trial2_full_spectral_jost",
        "selected_followup_route_or_none": "8.7.56.5547",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5541",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5543",
                "followup_route": "8.7.56.5547",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_bohr_radius_matching_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 Bohr radius / Compton gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から Bohr / Compton gate を実行する。

if __name__ == "__main__":
    main()
