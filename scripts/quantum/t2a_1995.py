#!/usr/bin/env python3
"""Generate 8.7.56.1995-.1998 phase-carrier decision gate artifacts."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
LONG_ROADMAP = ROOT / "doc" / "quantum" / "55_trial2_numeric_alpha_vector_qball_long_horizon_roadmap.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

PRIOR_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1991_1994_boundary_local_jet_phase_drift_audit_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1995-1998"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor boundary phase-carrier "
    "decision gate / registry"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "boundary_phase_carrier_gate",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_phase_carrier_window_40_to_120_retained_"
    "higher_q_generalization_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_phase_carrier_window_40_to_120_retained_"
    "higher_q_generalization_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_boundary_phase_carrier_"
    "higher_q_extension_audit"
)
NEXT_ROUTE = "8.7.56.1999"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_boundary_phase_carrier_"
    "generalization_decision_gate_registry"
)
FOLLOWUP_ROUTE = "8.7.56.2003"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


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

    return {
        "json": sign_base.display_path(paths["json"]),
        "csv": sign_base.display_path(paths["csv"]),
    }


# 関数: closeout 用の公式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the phase-carrier decision gate."""
    return {
        "retained_rule": "G_jet(q)=(-h0 q^2 + h2) cos(q R_box) + h1 q sin(q R_box)=0",
        "new_phase_carrier": "G_phi(q)=(-h0 q^2 + h2) cos(q R_box + phi0 + phi_-1/q + dR q) + h1 q sin(q R_box + phi0 + phi_-1/q + dR q)=0",
        "closeout_read": "retain the new carrier on 40<=q/m0<=120, but keep higher-q generalization open because the 120<=q/m0<=200 holdout still drifts",
    }


# 関数: `.1995-.1998` を実行する。

def main() -> None:
    """Execute the boundary phase-carrier decision gate."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LONG_ROADMAP,
        PART5,
        PRIOR_GATE,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    inventory_ready = bool(prior_summary["phase_carrier_window_supported"])

    gate_a_exact_asymptotic_promotion_selected = False
    gate_b_finite_phase_carrier_window_retained = bool(
        prior_summary["phase_carrier_window_supported"]
    )
    gate_c_current_rule_blocked = False
    same_level_one_parameter_retry_admissible = False
    higher_q_phase_carrier_generalization_admissible_now = True
    physical_reject_required = False

    rows = [
        sign_base.row("inventory_ready", "pass" if inventory_ready else "reject", "phase-carrier decision inventory ready", sign_base.truth(inventory_ready), "The registry sync starts only after the audit has separated failed one-parameter corrections from the retained finite-window phase carrier."),
        sign_base.row("gate_a_exact_asymptotic_promotion_selected", "reject", "Gate A exact asymptotic promotion selected", sign_base.truth(gate_a_exact_asymptotic_promotion_selected), "The 120<=q/m0<=200 holdout still drifts, so the new carrier cannot yet claim global asymptotic exactness."),
        sign_base.row("gate_b_finite_phase_carrier_window_retained", "pass" if gate_b_finite_phase_carrier_window_retained else "reject", "Gate B finite phase-carrier window retained", sign_base.truth(gate_b_finite_phase_carrier_window_retained), "The new boundary phase carrier is honest as a finite 40<=q/m0<=120 continuation because it sharply lowers the drift there without pretending to be global."),
        sign_base.row("gate_c_current_rule_blocked", "reject" if not gate_c_current_rule_blocked else "pass", "Gate C current rule blocked", sign_base.truth(gate_c_current_rule_blocked), "The new carrier is not globally rejected because it already rescues the first later asymptotic window that the retained local-jet theorem could not track."),
        sign_base.row("same_level_one_parameter_retry_admissible", "reject", "same-level one-parameter retry admissible", sign_base.truth(same_level_one_parameter_retry_admissible), "The audit already killed the honest single-parameter candidates, so the next route must work with the retained carrier rather than looping over the same simple corrections."),
        sign_base.row("higher_q_phase_carrier_generalization_admissible_now", "pass", "higher-q phase-carrier generalization admissible now", sign_base.truth(higher_q_phase_carrier_generalization_admissible_now), "The next official question is whether the retained finite-window carrier can be generalized farther than q/m0=120."),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "gate_a_exact_asymptotic_promotion_selected": gate_a_exact_asymptotic_promotion_selected,
        "gate_b_finite_phase_carrier_window_retained": gate_b_finite_phase_carrier_window_retained,
        "gate_c_current_rule_blocked": gate_c_current_rule_blocked,
        "same_level_one_parameter_retry_admissible": same_level_one_parameter_retry_admissible,
        "higher_q_phase_carrier_generalization_admissible_now": higher_q_phase_carrier_generalization_admissible_now,
        "phase_carrier_phi0": float(prior_summary["phase_carrier_phi0"]),
        "phase_carrier_phi_inv": float(prior_summary["phase_carrier_phi_inv"]),
        "phase_carrier_delta_r": float(prior_summary["phase_carrier_delta_r"]),
        "phase_carrier_fit_sign_mismatch_fraction": float(prior_summary["phase_carrier_fit_sign_mismatch_fraction"]),
        "phase_carrier_holdout_sign_mismatch_fraction": float(prior_summary["phase_carrier_holdout_sign_mismatch_fraction"]),
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.1997",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "ai_context": sign_base.display_path(AI_CONTEXT),
                "work_history_recent": sign_base.display_path(WORK_HISTORY_RECENT),
                "current_problem": sign_base.display_path(CURRENT_PROBLEM),
                "current_status": sign_base.display_path(CURRENT_STATUS),
                "unified_roadmap": sign_base.display_path(UNIFIED_ROADMAP),
                "long_roadmap": sign_base.display_path(LONG_ROADMAP),
                "part5": sign_base.display_path(PART5),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
            },
            "constants": {
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_boundary_phase_carrier_gate_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.1995"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.1995-.1998"),
                "current_problem_hit": sign_base.hit(current_problem_text, "asymptotic_phase_drift_audit_admissible_now"),
                "current_status_hit": sign_base.hit(current_status_text, "asymptotic phase drift selected"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".1995-.1998"),
                "long_roadmap_hit": sign_base.hit(long_text, "boundary local-jet asymptotic phase-drift decision gate"),
                "part5_hit": sign_base.hit(part5_text, ".1983-.1990"),
            },
        },
    )

    route_rows = [
        sign_base.row("gate_b_finite_phase_carrier_window_retained", "pass" if gate_b_finite_phase_carrier_window_retained else "reject", "Gate B finite phase-carrier window retained", sign_base.truth(gate_b_finite_phase_carrier_window_retained), "The finite 40<=q/m0<=120 carrier survives and therefore becomes the official new signed rule family."),
        sign_base.row("higher_q_phase_carrier_generalization_admissible_now", "pass", "higher-q phase-carrier generalization admissible now", sign_base.truth(higher_q_phase_carrier_generalization_admissible_now), "The next branch must test whether the retained carrier can move past the fitted finite asymptotic window."),
        sign_base.row("next_route_fixed", "pass", "next route fixed", 1.0, "The next official branch is the boundary phase-carrier higher-q extension audit."),
    ]

    route_payload = sign_base.payload(
        "8.7.56.1998",
        STEP_NAME + " route sync",
        {
            "declaration_source": sign_base.display_path(
                build_metrics_paths(PUBLIC_OUT, STEM, "declaration_gate")["json"]
            ),
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "selected_next_generation_route_or_none": NEXT_ROUTE,
            "selected_followup_route": FOLLOWUP_ROUTE_NAME,
            "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        },
        route_rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_boundary_phase_carrier_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_next_hit": sign_base.hit(status_text, "8.7.56.1995"),
                "roadmap_next_hit": sign_base.hit(roadmap_text, "8.7.56.1999"),
                "current_problem_next_hit": sign_base.hit(current_problem_text, "same_level_box_edge_retry_admissible"),
                "current_status_next_hit": sign_base.hit(current_status_text, "asymptotic phase-drift audit"),
                "unified_roadmap_next_hit": sign_base.hit(unified_text, ".1999-.2002"),
                "long_roadmap_next_hit": sign_base.hit(long_text, "higher-q phase-carrier generalization"),
                "part5_next_hit": sign_base.hit(part5_text, ".1983-.1990"),
            },
        },
    )

    declaration_paths = write_artifact("declaration_gate", declaration_payload)
    route_paths = write_artifact("route_sync", route_payload)
    print("[ok] 8.7.56.1995-.1998 phase-carrier decision gate artifacts generated")
    print(f"[ok] declaration: {declaration_paths['json']}")
    print(f"[ok] route sync:   {route_paths['json']}")


if __name__ == "__main__":
    main()
