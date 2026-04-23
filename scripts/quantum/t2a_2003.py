#!/usr/bin/env python3
"""Generate 8.7.56.2003-.2006 boundary phase-curvature decision gate artifacts."""

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
    / "q_8_7_56_1999_2002_boundary_phase_curvature_audit_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.2003-2006"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor boundary phase-curvature "
    "decision gate / registry"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "boundary_phase_curvature_gate",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_phase_curvature_window_120_to_200_"
    "large_coefficient_partial_retain_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_phase_curvature_window_120_to_200_"
    "large_coefficient_partial_retain_higher_q_extension_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_boundary_phase_curvature_"
    "higher_q_extension_audit"
)
NEXT_ROUTE = "8.7.56.2007"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_boundary_phase_curvature_"
    "generalization_decision_gate_registry"
)
FOLLOWUP_ROUTE = "8.7.56.2011"


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
    """Return formulas used in the phase-curvature decision gate."""
    return {
        "retained_failed_rule": "phi_3(q)=phi0 + phi_-1/q + dR q",
        "partial_retain_rule": "phi_4(q)=phi0 + phi_-1/q + dR q + phi_-2/q^2",
        "closeout_read": "close the retained 3-term carrier as a failed higher-q continuation, retain the 4-term curvature carrier only on 120<=q/m0<=200, and keep farther higher-q generalization open",
    }


# 関数: `.2003-.2006` を実行する。

def main() -> None:
    """Execute the boundary phase-curvature decision gate."""
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
    inventory_ready = bool(prior_summary["phase_curvature_window_supported"])

    gate_a_exact_higher_q_promotion_selected = False
    gate_b_phase_curvature_window_partial_retain = bool(
        prior_summary["phase_curvature_window_supported"]
    )
    gate_c_current_rule_blocked = False
    same_level_retained_three_term_retry_admissible = False
    same_level_phase_curvature_refit_admissible = False
    higher_q_phase_curvature_generalization_admissible_now = True
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "phase-curvature decision inventory ready",
            sign_base.truth(inventory_ready),
            "The registry sync starts only after the higher-q audit separates the failed retained 3-term carrier from the rescued second finite window.",
        ),
        sign_base.row(
            "gate_a_exact_higher_q_promotion_selected",
            "reject",
            "Gate A exact higher-q promotion selected",
            sign_base.truth(gate_a_exact_higher_q_promotion_selected),
            "The old 3-term carrier fails on 120<=q/m0<=200 and the 4-term curvature carrier still fails on 200<=q/m0<=260, so honest exact higher-q promotion is unavailable.",
        ),
        sign_base.row(
            "gate_b_phase_curvature_window_partial_retain",
            "pass" if gate_b_phase_curvature_window_partial_retain else "reject",
            "Gate B finite phase-curvature window partial retain",
            sign_base.truth(gate_b_phase_curvature_window_partial_retain),
            "The new 4-term phase-curvature carrier materially rescues 120<=q/m0<=200 and therefore deserves a partial retain as the second finite asymptotic window theorem.",
        ),
        sign_base.row(
            "gate_c_current_rule_blocked",
            "reject" if not gate_c_current_rule_blocked else "pass",
            "Gate C current rule blocked",
            sign_base.truth(gate_c_current_rule_blocked),
            "The new family is not globally rejected because it does open a new finite window beyond the retained 40<=q/m0<=120 carrier.",
        ),
        sign_base.row(
            "same_level_retained_three_term_retry_admissible",
            "reject",
            "same-level retained 3-term retry admissible",
            sign_base.truth(same_level_retained_three_term_retry_admissible),
            "The old 3-term carrier is already falsified on the next two later windows and should not be retried at the same level.",
        ),
        sign_base.row(
            "same_level_phase_curvature_refit_admissible",
            "reject",
            "same-level phase-curvature refit admissible",
            sign_base.truth(same_level_phase_curvature_refit_admissible),
            "Large coefficients plus the 200<=q/m0<=260 holdout failure mean the honest next step is farther-window extension of the retained 4-term rule, not more same-window refits.",
        ),
        sign_base.row(
            "higher_q_phase_curvature_generalization_admissible_now",
            "pass",
            "higher-q phase-curvature generalization admissible now",
            sign_base.truth(higher_q_phase_curvature_generalization_admissible_now),
            "The next official question is whether the partial-retain 4-term curvature family survives any farther than the rescued 120<=q/m0<=200 window.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "gate_a_exact_higher_q_promotion_selected": gate_a_exact_higher_q_promotion_selected,
        "gate_b_phase_curvature_window_partial_retain": gate_b_phase_curvature_window_partial_retain,
        "gate_c_current_rule_blocked": gate_c_current_rule_blocked,
        "same_level_retained_three_term_retry_admissible": same_level_retained_three_term_retry_admissible,
        "same_level_phase_curvature_refit_admissible": same_level_phase_curvature_refit_admissible,
        "higher_q_phase_curvature_generalization_admissible_now": higher_q_phase_curvature_generalization_admissible_now,
        "retained_three_term_fit_sign_mismatch_fraction": float(
            prior_summary["retained_three_term_fit_sign_mismatch_fraction"]
        ),
        "retained_three_term_holdout_sign_mismatch_fraction": float(
            prior_summary["retained_three_term_holdout_sign_mismatch_fraction"]
        ),
        "phase_curvature_phi0": float(prior_summary["phase_curvature_phi0"]),
        "phase_curvature_phi_inv": float(prior_summary["phase_curvature_phi_inv"]),
        "phase_curvature_delta_r": float(prior_summary["phase_curvature_delta_r"]),
        "phase_curvature_phi_inv2": float(prior_summary["phase_curvature_phi_inv2"]),
        "phase_curvature_coeff_linf": float(prior_summary["phase_curvature_coeff_linf"]),
        "phase_curvature_fit_sign_mismatch_fraction": float(
            prior_summary["phase_curvature_fit_sign_mismatch_fraction"]
        ),
        "phase_curvature_holdout_sign_mismatch_fraction": float(
            prior_summary["phase_curvature_holdout_sign_mismatch_fraction"]
        ),
        "phase_curvature_noncanonical_large_coefficients": bool(
            prior_summary["phase_curvature_noncanonical_large_coefficients"]
        ),
        "phase_curvature_higher_q_holdout_failed": bool(
            prior_summary["phase_curvature_higher_q_holdout_failed"]
        ),
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2005",
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
            "overall_status": "vector_qball_form_factor_boundary_phase_curvature_gate_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2003"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.2003-.2006"),
                "current_problem_hit": sign_base.hit(
                    current_problem_text,
                    "higher_q_phase_carrier_generalization_admissible_now",
                ),
                "current_status_hit": sign_base.hit(
                    current_status_text,
                    "boundary phase-carrier higher-q extension audit",
                ),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2003-.2006"),
                "long_roadmap_hit": sign_base.hit(
                    long_text,
                    "phase-carrier generalization decision gate / registry",
                ),
                "part5_hit": sign_base.hit(part5_text, ".1991-.1998"),
            },
        },
    )

    route_rows = [
        sign_base.row(
            "gate_b_phase_curvature_window_partial_retain",
            "pass" if gate_b_phase_curvature_window_partial_retain else "reject",
            "Gate B finite phase-curvature window partial retain",
            sign_base.truth(gate_b_phase_curvature_window_partial_retain),
            "The 4-term curvature family survives as a second finite-window theorem and therefore becomes the official next retained family.",
        ),
        sign_base.row(
            "phase_curvature_noncanonical_large_coefficients",
            "watch",
            "phase-curvature noncanonical large coefficients retained",
            sign_base.truth(summary["phase_curvature_noncanonical_large_coefficients"]),
            "The large coefficient obstruction is retained as a warning, not as a reason to suppress the next extension test.",
        ),
        sign_base.row(
            "phase_curvature_higher_q_holdout_failed",
            "watch",
            "phase-curvature higher-q holdout failed",
            sign_base.truth(summary["phase_curvature_higher_q_holdout_failed"]),
            "Because the 200<=q/m0<=260 holdout already fails, the next branch must ask whether any farther-window generalization is honest.",
        ),
        sign_base.row(
            "next_route_fixed",
            "pass",
            "next route fixed",
            1.0,
            "The next official branch is the boundary phase-curvature higher-q extension audit.",
        ),
    ]

    route_payload = sign_base.payload(
        "8.7.56.2006",
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
            "overall_status": "vector_qball_form_factor_boundary_phase_curvature_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_next_hit": sign_base.hit(status_text, "8.7.56.2003"),
                "roadmap_next_hit": sign_base.hit(roadmap_text, "8.7.56.2007"),
                "current_problem_next_hit": sign_base.hit(
                    current_problem_text,
                    "higher_q_phase_carrier_generalization_admissible_now",
                ),
                "current_status_next_hit": sign_base.hit(
                    current_status_text,
                    "boundary phase-carrier higher-q extension audit",
                ),
                "unified_roadmap_next_hit": sign_base.hit(unified_text, ".2007-.2010"),
                "long_roadmap_next_hit": sign_base.hit(
                    long_text,
                    "phase-curvature higher-q extension audit",
                ),
                "part5_next_hit": sign_base.hit(part5_text, ".1991-.1998"),
            },
        },
    )

    declaration_paths = write_artifact("declaration_gate", declaration_payload)
    route_paths = write_artifact("route_sync", route_payload)
    print("[ok] 8.7.56.2003-.2006 phase-curvature decision gate artifacts generated")
    print(f"[ok] declaration: {declaration_paths['json']}")
    print(f"[ok] route sync:   {route_paths['json']}")


if __name__ == "__main__":
    main()
