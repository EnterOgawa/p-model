#!/usr/bin/env python3
"""Generate 8.7.56.5583-.5586 Trial-2 energy-partition audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_energy_partition_ratio_backend import (
    build_trial2_energy_partition_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5579-5582",
        "updated_pack_trial2_alpha_beta_curve_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "78_trial2_numeric_alpha_vector_qball_energy_partition_audit.md"
)

STEP_TAG = "8.7.56.5583-5586"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "energy partition ratio audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_energy_partition_ratio_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_alpha_beta_curve_audited_local_beta_microshift_completed_"
    "energy_partition_primary_entropy_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_energy_partition_screen_interaction_harmonic_front_runner_"
    "followup_gate_next"
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


# 関数: note が expected claims を含むか確認する。

def note_contains_audit(text: str) -> bool:
    """Return whether the energy-partition audit note carries the expected claims."""
    patterns = (
        "interaction_over_harmonic",
        "energy partition",
        "entropy",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the energy-partition audit."""
    return {
        "harmonic_energy": "E_harm = E_kin + E_mass",
        "interaction_energy": "E_int = E_cubic + E_quartic",
        "front_runner": "R_int_harm = E_int / E_harm",
        "screen_rule": "pick the retained ratio with the smallest absolute relative error vs alpha_target among the simple screened partition family",
    }


# 関数: `.5583-.5586` を実行する。

def main() -> None:
    """Execute the Trial-2 energy-partition ratio audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_energy_partition_pack(
        retained_beta=float(prior_summary["retained_beta1"]),
        nearest_beta=float(prior_summary["nearest_alpha_beta_root_to_retained"]),
    )

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    energy_partition_front_runner_available_now = bool(len(pack["candidate_rows"]) >= 1)
    interaction_over_harmonic_front_runner_now = (
        str(pack["energy_partition_front_runner_name"]) == "interaction_over_harmonic"
    )
    front_runner_improves_alpha_beta_now = bool(pack["front_runner_improves_alpha_beta_now"])
    energy_partition_exact_route_available_now = bool(pack["front_runner_exact_route_available_now"])
    updated_pack_trial2_energy_partition_front_runner_followup_required_now = bool(
        route_selected
        and note_available
        and energy_partition_front_runner_available_now
        and interaction_over_harmonic_front_runner_now
        and front_runner_improves_alpha_beta_now
        and not energy_partition_exact_route_available_now
        and bool(pack["entropy_followup_required_now"])
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_energy_partition_ratio_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 energy-partition ratio selected now",
            sign_base.truth(route_selected),
            "The alpha(beta) route compressed the blocker to one local beta microshift, so energy partition is the next honest direct-alpha followup.",
        ),
        sign_base.row(
            "exact_trial2_energy_partition_audit_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 energy-partition audit note available now",
            sign_base.truth(note_available),
            "The audit note records the screened partition family, the front runner, and the entropy handoff.",
        ),
        sign_base.row(
            "exact_trial2_energy_partition_front_runner_available_now",
            "pass" if energy_partition_front_runner_available_now else "reject",
            "exact Trial-2 energy-partition front runner available now",
            sign_base.truth(energy_partition_front_runner_available_now),
            "The route is only nontrivial if one simple ratio clearly dominates the screened partition family.",
        ),
        sign_base.row(
            "exact_trial2_energy_partition_interaction_over_harmonic_front_runner_now",
            "pass" if interaction_over_harmonic_front_runner_now else "reject",
            "exact Trial-2 interaction-over-harmonic front runner now",
            sign_base.truth(interaction_over_harmonic_front_runner_now),
            "The best screened ratio should be the nonlinear interaction share relative to the harmonic core if this route is genuinely compressive.",
        ),
        sign_base.row(
            "exact_trial2_energy_partition_front_runner_improves_alpha_beta_now",
            "pass" if front_runner_improves_alpha_beta_now else "reject",
            "exact Trial-2 energy-partition front runner improves alpha(beta) now",
            sign_base.truth(front_runner_improves_alpha_beta_now),
            "The screened ratio must improve on the retained alpha(beta) residual to count as a useful followup rather than a replay.",
        ),
        sign_base.row(
            "exact_trial2_energy_partition_exact_route_available_now",
            "pass" if energy_partition_exact_route_available_now else "reject",
            "exact Trial-2 energy-partition exact route available now",
            sign_base.truth(energy_partition_exact_route_available_now),
            "Pass would mean one screened ratio already reproduces alpha_target exactly without additional interpretation.",
        ),
        sign_base.row(
            "updated_pack_trial2_energy_partition_front_runner_followup_required_now",
            "pass" if updated_pack_trial2_energy_partition_front_runner_followup_required_now else "reject",
            "updated-pack Trial-2 energy-partition front-runner followup required now",
            sign_base.truth(updated_pack_trial2_energy_partition_front_runner_followup_required_now),
            "The route is useful but not exact, so the honest next blocker is whether interaction-over-harmonic can be elevated from heuristic front runner to target-free relation.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "alpha_target": float(pack["alpha_target"]),
        "retained_beta1": float(pack["retained_beta1"]),
        "nearest_alpha_beta_root_to_retained": float(pack["nearest_alpha_beta_root_to_retained"]),
        "nearest_beta_rel_shift_vs_retained": float(pack["nearest_beta_rel_shift_vs_retained"]),
        "retained_energy_kinetic": float(pack["retained_energy_row"]["energy_kinetic"]),
        "retained_energy_mass": float(pack["retained_energy_row"]["energy_mass"]),
        "retained_energy_gradient": float(pack["retained_energy_row"]["energy_gradient"]),
        "retained_energy_interaction": float(pack["retained_energy_row"]["energy_interaction"]),
        "retained_energy_total": float(pack["retained_energy_row"]["energy_total"]),
        "energy_partition_front_runner_name": str(pack["energy_partition_front_runner_name"]),
        "energy_partition_front_runner_retained_value": float(
            pack["energy_partition_front_runner_retained_value"]
        ),
        "energy_partition_front_runner_retained_rel_error_vs_target": float(
            pack["energy_partition_front_runner_retained_rel_error_vs_target"]
        ),
        "energy_partition_front_runner_near_value": float(
            pack["energy_partition_front_runner_near_value"]
        ),
        "energy_partition_front_runner_near_rel_error_vs_target": float(
            pack["energy_partition_front_runner_near_rel_error_vs_target"]
        ),
        "energy_partition_front_runner_near_rel_shift_vs_retained": float(
            pack["energy_partition_front_runner_near_rel_shift_vs_retained"]
        ),
        "energy_partition_second_runner_name": str(pack["energy_partition_second_runner_name"]),
        "energy_partition_second_runner_retained_abs_rel_error_vs_target": float(
            pack["energy_partition_second_runner_retained_abs_rel_error_vs_target"]
        ),
        "energy_partition_front_runner_margin_vs_second": float(
            pack["energy_partition_front_runner_margin_vs_second"]
        ),
        "exact_trial2_energy_partition_front_runner_available_now": (
            energy_partition_front_runner_available_now
        ),
        "exact_trial2_energy_partition_interaction_over_harmonic_front_runner_now": (
            interaction_over_harmonic_front_runner_now
        ),
        "exact_trial2_energy_partition_front_runner_improves_alpha_beta_now": (
            front_runner_improves_alpha_beta_now
        ),
        "exact_trial2_energy_partition_exact_route_available_now": (
            energy_partition_exact_route_available_now
        ),
        "updated_pack_trial2_energy_partition_front_runner_followup_required_now": (
            updated_pack_trial2_energy_partition_front_runner_followup_required_now
        ),
        "selected_primary_completion_lane": "trial2_energy_partition_interaction_harmonic",
        "selected_secondary_completion_lane": "trial2_entropy_route",
        "selected_reserve_completion_lane": "trial2_entropy_route",
        "selected_next_generation_route": "trial2_energy_partition_interaction_harmonic",
        "recommended_next_route_or_none": "8.7.56.5587",
        "selected_followup_route": "trial2_energy_partition_interaction_harmonic",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5585",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
                "backend_helper": sign_base.display_path(
                    ROOT / "scripts" / "quantum" / "trial2_energy_partition_ratio_backend.py"
                ),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5587",
                "followup_route": "trial2_energy_partition_interaction_harmonic",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_energy_partition_ratio_audit_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae(), "candidate_rows": pack["candidate_rows"]},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 energy-partition ratio audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
