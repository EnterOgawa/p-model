#!/usr/bin/env python3
"""Generate 8.7.56.2011-.2014 farther high-q sign-root decision gate artifacts."""

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
    / "q_8_7_56_2007_2010_boundary_phase_curvature_higher_q_ext_audit_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.2011-2014"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor farther high-q sign-root "
    "decision gate / registry"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "farther_high_q_sign_root_gate",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_phase_curvature_farther_high_q_"
    "unresolved_sign_root_floor_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_resolved_high_q_sign_root_floor_envelope_microphase_"
    "reactivation_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_resolved_high_q_sign_root_"
    "floor_envelope_microphase_audit"
)
NEXT_ROUTE = "8.7.56.2015"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_resolved_high_q_sign_root_"
    "decision_gate_registry"
)
FOLLOWUP_ROUTE = "8.7.56.2019"


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
    """Return formulas used in the farther high-q sign-root decision gate."""
    return {
        "retained_failed_rule": "phi_4(q)=phi0 + phi_-1/q + dR q + phi_-2/q^2",
        "new_blocker_read": "sign-root floor plus envelope/microphase decoupling replaces same-level smooth phase-curvature fitting as the honest blocker",
        "closeout_read": "close same-level smooth carrier retry, retain the finite 120<=q/m0<=200 phase-curvature window, and promote high-q sign-root floor / microphase structure to the next official audit",
    }


# 関数: `.2011-.2014` を実行する。

def main() -> None:
    """Execute the farther high-q sign-root decision gate."""
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
    inventory_ready = bool(prior_summary["smooth_phase_curvature_family_exhausted"])

    gate_a_exact_farther_higher_q_promotion_selected = False
    gate_b_unresolved_sign_root_floor_selected = bool(
        prior_summary["unresolved_sign_root_floor_detected"]
    )
    gate_c_current_rule_blocked = False
    same_level_phase_curvature_retry_admissible = False
    same_level_farther_high_q_refit_admissible = False
    resolved_high_q_sign_root_floor_reactivation_admissible_now = True
    substantive_pack_update_required_now = False
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "farther high-q sign-root decision inventory ready",
            sign_base.truth(inventory_ready),
            "The registry sync starts only after the farther-window audit has separated smooth carrier failure from sign-root floor structure.",
        ),
        sign_base.row(
            "gate_a_exact_farther_higher_q_promotion_selected",
            "reject",
            "Gate A exact farther high-q promotion selected",
            sign_base.truth(gate_a_exact_farther_higher_q_promotion_selected),
            "The retained smooth 4-term carrier fails immediately on 200<=q/m0<=260 and therefore cannot be promoted as an exact farther high-q theorem.",
        ),
        sign_base.row(
            "gate_b_unresolved_sign_root_floor_selected",
            "pass" if gate_b_unresolved_sign_root_floor_selected else "reject",
            "Gate B unresolved high-q sign-root floor selected",
            sign_base.truth(gate_b_unresolved_sign_root_floor_selected),
            "The honest blocker is now the sign-root floor plus envelope/microphase split, not a same-level smooth phase-carvature coefficient miss.",
        ),
        sign_base.row(
            "gate_c_current_rule_blocked",
            "reject" if not gate_c_current_rule_blocked else "pass",
            "Gate C current rule blocked",
            sign_base.truth(gate_c_current_rule_blocked),
            "The current family is not globally rejected because it still opens the retained finite 120<=q/m0<=200 window.",
        ),
        sign_base.row(
            "same_level_phase_curvature_retry_admissible",
            "reject",
            "same-level phase-curvature retry admissible",
            sign_base.truth(same_level_phase_curvature_retry_admissible),
            "Once the blocker is sign-root floor, more same-window smooth carrier tuning is not the honest next move.",
        ),
        sign_base.row(
            "same_level_farther_high_q_refit_admissible",
            "reject",
            "same-level farther high-q refit admissible",
            sign_base.truth(same_level_farther_high_q_refit_admissible),
            "The farther-window failure has already been reclassified, so another same-level smooth refit would only repeat the exhausted family.",
        ),
        sign_base.row(
            "resolved_high_q_sign_root_floor_reactivation_admissible_now",
            "pass",
            "resolved high-q sign-root floor reactivation admissible now",
            sign_base.truth(resolved_high_q_sign_root_floor_reactivation_admissible_now),
            "The next official question is whether the sign-root floor can be resolved by a new signed-rule split between envelope and microphase sectors.",
        ),
        sign_base.row(
            "substantive_pack_update_required_now",
            "reject" if not substantive_pack_update_required_now else "pass",
            "substantive pack update required now",
            sign_base.truth(substantive_pack_update_required_now),
            "The next step can still be posed inside the retained pack as a sign-root theorem audit, without immediately demanding a new pack update.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "gate_a_exact_farther_higher_q_promotion_selected": gate_a_exact_farther_higher_q_promotion_selected,
        "gate_b_unresolved_sign_root_floor_selected": gate_b_unresolved_sign_root_floor_selected,
        "gate_c_current_rule_blocked": gate_c_current_rule_blocked,
        "same_level_phase_curvature_retry_admissible": same_level_phase_curvature_retry_admissible,
        "same_level_farther_high_q_refit_admissible": same_level_farther_high_q_refit_admissible,
        "resolved_high_q_sign_root_floor_reactivation_admissible_now": resolved_high_q_sign_root_floor_reactivation_admissible_now,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "fit_window_sign_mismatch_fraction": float(prior_summary["fit_window_sign_mismatch_fraction"]),
        "floor_window_sign_mismatch_fraction": float(prior_summary["floor_window_sign_mismatch_fraction"]),
        "micro_window_sign_mismatch_fraction": float(prior_summary["micro_window_sign_mismatch_fraction"]),
        "edge_window_sign_mismatch_fraction": float(prior_summary["edge_window_sign_mismatch_fraction"]),
        "fit_window_max_abs_form_factor": float(prior_summary["fit_window_max_abs_form_factor"]),
        "floor_window_max_abs_form_factor": float(prior_summary["floor_window_max_abs_form_factor"]),
        "micro_window_max_abs_form_factor": float(prior_summary["micro_window_max_abs_form_factor"]),
        "edge_window_max_abs_form_factor": float(prior_summary["edge_window_max_abs_form_factor"]),
        "unresolved_sign_root_floor_detected": bool(prior_summary["unresolved_sign_root_floor_detected"]),
        "envelope_microphase_decoupling_detected": bool(prior_summary["envelope_microphase_decoupling_detected"]),
        "edge_spike_window_detected": bool(prior_summary["edge_spike_window_detected"]),
        "smooth_phase_curvature_family_exhausted": bool(prior_summary["smooth_phase_curvature_family_exhausted"]),
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2013",
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
            "overall_status": "vector_qball_form_factor_farther_high_q_sign_root_gate_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2011-.2014"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.2011-.2014"),
                "current_problem_hit": sign_base.hit(
                    current_problem_text,
                    "higher_q_phase_curvature_generalization_admissible_now",
                ),
                "current_status_hit": sign_base.hit(
                    current_status_text,
                    "boundary phase-curvature higher-q extension audit",
                ),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2011-.2014"),
                "long_roadmap_hit": sign_base.hit(
                    long_text,
                    "phase-curvature generalization decision gate / registry",
                ),
                "part5_hit": sign_base.hit(part5_text, ".1999-.2006"),
            },
        },
    )

    route_rows = [
        sign_base.row(
            "gate_b_unresolved_sign_root_floor_selected",
            "pass" if gate_b_unresolved_sign_root_floor_selected else "reject",
            "Gate B unresolved high-q sign-root floor selected",
            sign_base.truth(gate_b_unresolved_sign_root_floor_selected),
            "The next official branch is justified only if the blocker is now honestly classified as a sign-root floor family.",
        ),
        sign_base.row(
            "same_level_phase_curvature_retry_admissible",
            "reject",
            "same-level phase-curvature retry admissible",
            sign_base.truth(same_level_phase_curvature_retry_admissible),
            "The smooth carrier family is already exhausted and should not be retried at the same level.",
        ),
        sign_base.row(
            "resolved_high_q_sign_root_floor_reactivation_admissible_now",
            "pass",
            "resolved high-q sign-root floor reactivation admissible now",
            sign_base.truth(resolved_high_q_sign_root_floor_reactivation_admissible_now),
            "The next official branch is the resolved high-q sign-root floor / envelope-microphase audit.",
        ),
        sign_base.row(
            "next_route_fixed",
            "pass",
            "next route fixed",
            1.0,
            "The next official branch is the resolved high-q sign-root floor / envelope-microphase audit.",
        ),
    ]

    route_payload = sign_base.payload(
        "8.7.56.2014",
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
            "overall_status": "vector_qball_form_factor_farther_high_q_sign_root_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_next_hit": sign_base.hit(status_text, "8.7.56.2011-.2014"),
                "roadmap_next_hit": sign_base.hit(roadmap_text, "8.7.56.2015-.2018"),
                "current_problem_next_hit": sign_base.hit(
                    current_problem_text,
                    "phase_curvature_higher_q_holdout_failed",
                ),
                "current_status_next_hit": sign_base.hit(
                    current_status_text,
                    "boundary phase-curvature higher-q extension audit",
                ),
                "unified_roadmap_next_hit": sign_base.hit(unified_text, ".2015-.2018"),
                "long_roadmap_next_hit": sign_base.hit(
                    long_text,
                    "resolved high-q sign-root floor / envelope-microphase audit",
                ),
                "part5_next_hit": sign_base.hit(part5_text, ".1999-.2006"),
            },
        },
    )

    declaration_paths = write_artifact("declaration_gate", declaration_payload)
    route_paths = write_artifact("route_sync", route_payload)
    print("[ok] 8.7.56.2011-.2014 farther high-q sign-root gate artifacts generated")
    print(f"[ok] declaration: {declaration_paths['json']}")
    print(f"[ok] route sync:   {route_paths['json']}")


if __name__ == "__main__":
    main()
