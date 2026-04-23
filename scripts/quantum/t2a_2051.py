#!/usr/bin/env python3
"""Generate 8.7.56.2051-.2054 alias-image phase-slip loading registry artifacts."""

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

PRIOR_GATE = PUBLIC_OUT / "q_8_7_56_2047_2050_q_dependent_phase_slip_loading_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.2051-2054"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor alias-image phase-slip loading closeout / registry"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "alias_image_phase_slip_loading_registry",
    prefix="q",
)

PRIOR_CLASS = "vector_qball_form_factor_minimal_q_dependent_boundary_loading_blocked_higher_harmonic_windowwise_signed_rule_gate_next"
BRANCH_CLASS = "vector_qball_form_factor_q_dependent_boundary_loading_closed_higher_harmonic_windowwise_signed_rule_reactivation_next"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_higher_harmonic_windowwise_phase_slip_signed_rule_reactivation"
NEXT_ROUTE = "8.7.56.2055"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_higher_harmonic_loading_decision_gate_registry"
FOLLOWUP_ROUTE = "8.7.56.2059"


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


# 関数: 使用公式を返す。
def build_formulae() -> dict[str, str]:
    """Return formulas used in the registry sync."""
    return {
        "minimal_q_dependent_family": "delta_q^(2)(q)=a0 + a1/q + a2/q^2",
        "parity_split_family": "delta_q^(oe)(q)=a_o0+a_o1/q (odd), a_e0+a_e1/q (even)",
        "next_surface": "delta_q,star^(n)=argmin_delta mismatch_n(delta) on each harmonic window",
    }


# 関数: `.2051-.2054` を実行する。
def main() -> None:
    """Execute the alias-image phase-slip loading closeout / registry sync."""
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
    inventory_ready = bool(prior_summary["higher_harmonic_windowwise_signed_rule_admissible"])

    gate_a_minimal_q_dependent_loading_selected = bool(prior_summary["minimal_q_dependent_boundary_loading_supported"])
    gate_b_higher_harmonic_windowwise_signed_rule_selected = bool(prior_summary["higher_harmonic_windowwise_signed_rule_admissible"])
    gate_c_substantive_pack_update_required = False
    same_level_constant_delta_retry_admissible = bool(prior_summary["same_level_constant_delta_retry_admissible"])
    same_level_minimal_q_dependent_retry_admissible = bool(prior_summary["same_level_minimal_q_dependent_retry_admissible"])
    higher_harmonic_windowwise_loading_admissible_now = gate_b_higher_harmonic_windowwise_signed_rule_selected
    substantive_pack_update_required_now = False
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "phase-slip loading registry inventory ready",
            sign_base.truth(inventory_ready),
            "The closeout starts only after `.2047-.2050` has fixed whether smooth q-dependent loading survives translated and holdout windows.",
        ),
        sign_base.row(
            "gate_a_minimal_q_dependent_loading_selected",
            "reject" if not gate_a_minimal_q_dependent_loading_selected else "pass",
            "Gate A minimal q-dependent loading selected",
            sign_base.truth(gate_a_minimal_q_dependent_loading_selected),
            "Gate A would require one smooth boundary-loading family to close the translated harmonic windows without reopening new signed-rule structure.",
        ),
        sign_base.row(
            "gate_b_higher_harmonic_windowwise_signed_rule_selected",
            "pass" if gate_b_higher_harmonic_windowwise_signed_rule_selected else "reject",
            "Gate B higher-harmonic windowwise signed rule selected",
            sign_base.truth(gate_b_higher_harmonic_windowwise_signed_rule_selected),
            "Once minimal q-dependent loading fails but windowwise harmonic loading stays inside the partial-retain envelope, the honest next route is a harmonic-index dependent signed rule.",
        ),
        sign_base.row(
            "gate_c_substantive_pack_update_required",
            "reject",
            "Gate C substantive pack update required",
            sign_base.truth(gate_c_substantive_pack_update_required),
            "The current retained pack still has one internal higher-harmonic signed-rule surface left, so a pack update remains reserve.",
        ),
        sign_base.row(
            "same_level_constant_delta_retry_admissible",
            "reject" if not same_level_constant_delta_retry_admissible else "pass",
            "same-level constant-delta retry admissible",
            sign_base.truth(same_level_constant_delta_retry_admissible),
            "The constant-slip theorem is already fixed and should remain closed.",
        ),
        sign_base.row(
            "same_level_minimal_q_dependent_retry_admissible",
            "reject" if not same_level_minimal_q_dependent_retry_admissible else "pass",
            "same-level minimal q-dependent retry admissible",
            sign_base.truth(same_level_minimal_q_dependent_retry_admissible),
            "Once the minimal smooth families fail on holdout windows, more same-level rational refits should remain closed.",
        ),
        sign_base.row(
            "higher_harmonic_windowwise_loading_admissible_now",
            "pass" if higher_harmonic_windowwise_loading_admissible_now else "reject",
            "higher-harmonic windowwise loading admissible now",
            sign_base.truth(higher_harmonic_windowwise_loading_admissible_now),
            "The next honest surface is a harmonic-index dependent signed rule, not another smooth q-dependent loading retry.",
        ),
        sign_base.row(
            "substantive_pack_update_required_now",
            "reject",
            "substantive pack update required now",
            sign_base.truth(substantive_pack_update_required_now),
            "A pack update remains reserve because the current retained pack still contains a direct higher-harmonic loading question.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "constant_delta_q_theorem_over_m0": float(prior_summary["constant_delta_q_theorem_over_m0"]),
        "q2_fit_window_max_mismatch_fraction": float(prior_summary["q2_fit_window_max_mismatch_fraction"]),
        "q2_holdout_window_max_mismatch_fraction": float(prior_summary["q2_holdout_window_max_mismatch_fraction"]),
        "parity_split_fit_window_max_mismatch_fraction": float(prior_summary["parity_split_fit_window_max_mismatch_fraction"]),
        "parity_split_holdout_window_max_mismatch_fraction": float(prior_summary["parity_split_holdout_window_max_mismatch_fraction"]),
        "independent_core_window_max_mismatch_fraction": float(prior_summary["independent_core_window_max_mismatch_fraction"]),
        "independent_extension_window_max_mismatch_fraction": float(prior_summary["independent_extension_window_max_mismatch_fraction"]),
        "harmonic_delta_monotone": bool(prior_summary["harmonic_delta_monotone"]),
        "q2_vs_independent_delta_rms": float(prior_summary["q2_vs_independent_delta_rms"]),
        "gate_a_minimal_q_dependent_loading_selected": gate_a_minimal_q_dependent_loading_selected,
        "gate_b_higher_harmonic_windowwise_signed_rule_selected": gate_b_higher_harmonic_windowwise_signed_rule_selected,
        "gate_c_substantive_pack_update_required": gate_c_substantive_pack_update_required,
        "same_level_constant_delta_retry_admissible": same_level_constant_delta_retry_admissible,
        "same_level_minimal_q_dependent_retry_admissible": same_level_minimal_q_dependent_retry_admissible,
        "higher_harmonic_windowwise_loading_admissible_now": higher_harmonic_windowwise_loading_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2053",
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
            "overall_status": "vector_qball_form_factor_phase_slip_loading_registry_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2051"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.2051-.2054"),
                "current_problem_hit": sign_base.hit(current_problem_text, "q-dependent or harmonic-index dependent loading"),
                "current_status_hit": sign_base.hit(current_status_text, "q-dependent or harmonic-index dependent loading"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2051-.2054"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2051-.2054"),
                "part5_hit": sign_base.hit(part5_text, ".2039-.2046"),
            },
        },
    )

    route_payload = sign_base.payload(
        "8.7.56.2054",
        STEP_NAME + " route sync",
        declaration_payload["inputs"],
        [
            sign_base.row(
                "gate_b_higher_harmonic_windowwise_signed_rule_selected",
                "pass" if gate_b_higher_harmonic_windowwise_signed_rule_selected else "reject",
                "Gate B higher-harmonic windowwise signed rule selected",
                sign_base.truth(gate_b_higher_harmonic_windowwise_signed_rule_selected),
                "The current retained pack now points to a harmonic-index dependent signed rule rather than to another smooth boundary-loading refit.",
            ),
            sign_base.row(
                "same_level_minimal_q_dependent_retry_admissible",
                "reject" if not same_level_minimal_q_dependent_retry_admissible else "pass",
                "same-level minimal q-dependent retry admissible",
                sign_base.truth(same_level_minimal_q_dependent_retry_admissible),
                "The smooth rational loading family is closed after the holdout failure.",
            ),
            sign_base.row(
                "next_route_fixed",
                "pass",
                "next route fixed",
                1.0,
                "The next official branch is the higher-harmonic windowwise phase-slip signed-rule reactivation.",
            ),
        ],
        summary,
        {
            "overall_status": "vector_qball_form_factor_phase_slip_loading_registry_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"formulas": build_formulae()},
    )

    declaration_paths = write_artifact("declaration_gate", declaration_payload)
    route_paths = write_artifact("route_sync", route_payload)
    print("[done] 8.7.56.2051-.2054 complete")
    print(f"[info] declaration gate: {declaration_paths['json']}")
    print(f"[info] route sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
