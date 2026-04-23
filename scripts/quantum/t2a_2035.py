#!/usr/bin/env python3
"""Generate 8.7.56.2035-.2038 alias-image decision-gate artifacts."""

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
    / "q_8_7_56_2031_2034_boundary_alias_image_reactivation_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.2035-2038"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor alias-image decision "
    "gate / registry"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "alias_image_phase_slip_gate",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_alias_image_shared_phase_slip_partial_retain_"
    "decision_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_alias_image_shared_phase_slip_partial_retain_"
    "exact_phase_slip_theorem_or_higher_q_generalization_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exact_boundary_phase_slip_theorem_"
    "or_alias_image_higher_q_generalization"
)
NEXT_ROUTE = "8.7.56.2039"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_alias_image_shared_phase_slip_"
    "closeout_registry"
)
FOLLOWUP_ROUTE = "8.7.56.2043"


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


# 関数: decision gate 用の公式群を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the alias-image decision gate."""
    return {
        "plain_alias_image_rule": "sigma_img^(n)(q)=(-1)^n sign(F_exact(|q_alias^(n)-q|))",
        "shared_phase_slip_rule": "sigma_img,delta^(n)(q)=(-1)^n sign(F_exact(|q_alias^(n)+(-1)^(n+1) delta_q-q|))",
        "gate_logic": "Gate A requires exact canonical closeout; Gate B retains the shared phase-slip family only as a finite-window partial theorem; Gate C would require a full pack reset",
    }


# 関数: `.2035-.2038` を実行する。

def main() -> None:
    """Execute the alias-image shared phase-slip decision gate."""
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
    inventory_ready = bool(prior_summary["shared_phase_slip_alias_family_supported"])

    gate_a_exact_alias_image_promotion_selected = bool(
        prior_summary["plain_alias_image_exact_available"]
        or prior_summary["shared_phase_slip_canonical_theorem_available"]
    )
    gate_b_shared_phase_slip_partial_selected = bool(
        prior_summary["shared_phase_slip_alias_family_supported"]
        and prior_summary["shared_phase_slip_partial_window_retained"]
        and not gate_a_exact_alias_image_promotion_selected
    )
    gate_c_current_rule_blocked = False
    exact_boundary_phase_slip_theorem_admissible_now = bool(
        prior_summary["exact_boundary_phase_slip_theorem_admissible_now"]
    )
    same_level_unshifted_alias_retry_admissible = bool(
        prior_summary["same_level_unshifted_alias_retry_admissible"]
    )
    substantive_pack_update_required_now = False
    physical_reject_required = False

    rows = [
        sign_base.row("inventory_ready", "pass" if inventory_ready else "reject", "alias-image decision inventory ready", sign_base.truth(inventory_ready), "The decision gate starts only after `.2031-.2034` has fixed whether the shared phase-slip family improves both retained alias windows."),
        sign_base.row("gate_a_exact_alias_image_promotion_selected", "reject" if not gate_a_exact_alias_image_promotion_selected else "pass", "Gate A exact alias-image promotion selected", sign_base.truth(gate_a_exact_alias_image_promotion_selected), "Neither the plain alias-image rule nor the shifted family closes the spike windows exactly at theorem level here."),
        sign_base.row("gate_b_shared_phase_slip_partial_selected", "pass" if gate_b_shared_phase_slip_partial_selected else "reject", "Gate B shared phase-slip partial retain selected", sign_base.truth(gate_b_shared_phase_slip_partial_selected), "The honest read is a finite-window partial retain once one shared phase-slip improves both windows but still lacks an exact theorem."),
        sign_base.row("gate_c_current_rule_blocked", "reject" if not gate_c_current_rule_blocked else "pass", "Gate C current rule blocked", sign_base.truth(gate_c_current_rule_blocked), "The retained box pack still contains a next theorem surface, so a full pack reset is not yet required."),
        sign_base.row("same_level_unshifted_alias_retry_admissible", "reject" if not same_level_unshifted_alias_retry_admissible else "pass", "same-level unshifted alias retry admissible", sign_base.truth(same_level_unshifted_alias_retry_admissible), "The unshifted alias-image family is superseded by the shared phase-slip family and should not be retried at the same level."),
        sign_base.row("exact_boundary_phase_slip_theorem_admissible_now", "pass" if exact_boundary_phase_slip_theorem_admissible_now else "reject", "exact boundary phase-slip theorem admissible now", sign_base.truth(exact_boundary_phase_slip_theorem_admissible_now), "The next honest branch is to ask whether delta_q* can be derived canonically or generalized beyond the current finite windows."),
        sign_base.row("substantive_pack_update_required_now", "reject" if not substantive_pack_update_required_now else "pass", "substantive pack update required now", sign_base.truth(substantive_pack_update_required_now), "A pack update remains reserve; the current retained pack still has an internal theorem question left."),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "shared_phase_slip_delta_q_star_over_m0": float(prior_summary["shared_phase_slip_delta_q_star_over_m0"]),
        "fit_window_phase_slip_sign_mismatch_fraction": float(prior_summary["fit_window_phase_slip_sign_mismatch_fraction"]),
        "edge_window_phase_slip_sign_mismatch_fraction": float(prior_summary["edge_window_phase_slip_sign_mismatch_fraction"]),
        "fit_window_phase_slip_sign_correlation": float(prior_summary["fit_window_phase_slip_sign_correlation"]),
        "edge_window_phase_slip_sign_correlation": float(prior_summary["edge_window_phase_slip_sign_correlation"]),
        "fit_window_phase_slip_signed_reconstruction_max_abs_error": float(prior_summary["fit_window_phase_slip_signed_reconstruction_max_abs_error"]),
        "edge_window_phase_slip_signed_reconstruction_max_abs_error": float(prior_summary["edge_window_phase_slip_signed_reconstruction_max_abs_error"]),
        "fit_mismatch_gain_over_plain_alias_image": float(prior_summary["fit_mismatch_gain_over_plain_alias_image"]),
        "edge_mismatch_gain_over_plain_alias_image": float(prior_summary["edge_mismatch_gain_over_plain_alias_image"]),
        "delta_q_rel_to_edge_gap_estimate": float(prior_summary["delta_q_rel_to_edge_gap_estimate"]),
        "delta_q_rel_to_pi_over_rbox": float(prior_summary["delta_q_rel_to_pi_over_rbox"]),
        "plain_alias_image_exact_available": bool(prior_summary["plain_alias_image_exact_available"]),
        "shared_phase_slip_alias_family_supported": bool(prior_summary["shared_phase_slip_alias_family_supported"]),
        "shared_phase_slip_partial_window_retained": bool(prior_summary["shared_phase_slip_partial_window_retained"]),
        "shared_phase_slip_canonical_theorem_available": bool(prior_summary["shared_phase_slip_canonical_theorem_available"]),
        "gate_a_exact_alias_image_promotion_selected": gate_a_exact_alias_image_promotion_selected,
        "gate_b_shared_phase_slip_partial_selected": gate_b_shared_phase_slip_partial_selected,
        "gate_c_current_rule_blocked": gate_c_current_rule_blocked,
        "same_level_unshifted_alias_retry_admissible": same_level_unshifted_alias_retry_admissible,
        "exact_boundary_phase_slip_theorem_admissible_now": exact_boundary_phase_slip_theorem_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2037",
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
            "overall_status": "vector_qball_form_factor_alias_image_decision_gate_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2035"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.2035-.2038"),
                "current_problem_hit": sign_base.hit(current_problem_text, "boundary alias-image signed rule reactivation"),
                "current_status_hit": sign_base.hit(current_status_text, "boundary alias-image signed rule reactivation"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2035-.2038"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2035-.2038"),
                "part5_hit": sign_base.hit(part5_text, ".2023-.2030"),
            },
        },
    )

    route_payload = sign_base.payload(
        "8.7.56.2038",
        STEP_NAME + " route sync",
        declaration_payload["inputs"],
        [
            sign_base.row("gate_b_shared_phase_slip_partial_selected", "pass" if gate_b_shared_phase_slip_partial_selected else "reject", "Gate B shared phase-slip partial retain selected", sign_base.truth(gate_b_shared_phase_slip_partial_selected), "The next official route is justified only if the current pack retains a better finite-window alias-image family without claiming exact closure."),
            sign_base.row("exact_boundary_phase_slip_theorem_admissible_now", "pass" if exact_boundary_phase_slip_theorem_admissible_now else "reject", "exact boundary phase-slip theorem admissible now", sign_base.truth(exact_boundary_phase_slip_theorem_admissible_now), "The follow-up question is now the theorem status of delta_q* or its higher-q generalization."),
            sign_base.row("next_route_fixed", "pass", "next route fixed", 1.0, "The next official branch is the exact boundary phase-slip theorem or alias-image higher-q generalization audit."),
        ],
        summary,
        {
            "overall_status": "vector_qball_form_factor_alias_image_decision_gate_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"formulas": build_formulae()},
    )

    declaration_paths = write_artifact("declaration_gate", declaration_payload)
    route_paths = write_artifact("route_sync", route_payload)
    print("[ok] 8.7.56.2035-.2038 alias-image decision-gate artifacts generated")
    print(f"[ok] declaration: {declaration_paths['json']}")
    print(f"[ok] route sync:   {route_paths['json']}")


if __name__ == "__main__":
    main()
