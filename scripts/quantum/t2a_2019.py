#!/usr/bin/env python3
"""Generate 8.7.56.2019-.2022 resolved high-q sign-root gate artifacts."""

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
    / "q_8_7_56_2015_2018_resolved_high_q_sign_floor_audit_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.2019-2022"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor resolved high-q sign-root "
    "decision gate / registry"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "resolved_high_q_sign_root_gate",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_high_q_sign_root_floor_resolved_alias_harmonic_spike_"
    "gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_alias_harmonic_spike_reactivation_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_boundary_alias_harmonic_"
    "spike_audit"
)
NEXT_ROUTE = "8.7.56.2023"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_alias_harmonic_spike_"
    "decision_gate_registry"
)
FOLLOWUP_ROUTE = "8.7.56.2027"


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


# 関数: decision gate 用の公式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the resolved high-q sign-root decision gate."""
    return {
        "resolved_floor_read": "replace raw ROOT_TOL zero bookkeeping by sign-change parity on a fixed q scan",
        "microphase_read": "once |F_exact| falls below a shared envelope floor candidate, the residual floor/micro mismatch is no longer an honest signed-observable blocker",
        "remaining_blocker": "the unresolved residual now lives on alias-harmonic spike windows q_alias^(n)=2 n pi / Delta r_max",
    }


# 関数: `.2019-.2022` を実行する。

def main() -> None:
    """Execute the resolved high-q sign-root decision gate."""
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
    inventory_ready = bool(prior_summary["sign_root_floor_resolved"])

    gate_a_exact_farther_high_q_promotion_selected = False
    gate_b_alias_harmonic_spike_selected = bool(
        prior_summary["remaining_blocker_is_alias_harmonic_spike"]
    )
    gate_c_current_rule_blocked = False
    same_level_root_floor_retry_admissible = False
    same_level_microphase_retry_admissible = False
    alias_harmonic_spike_audit_admissible_now = True
    substantive_pack_update_required_now = False
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "resolved high-q sign-root decision inventory ready",
            sign_base.truth(inventory_ready),
            "The gate starts only after the prior audit has reclassified the raw root explosion into bookkeeping plus alias-harmonic structure.",
        ),
        sign_base.row(
            "gate_a_exact_farther_high_q_promotion_selected",
            "reject",
            "Gate A exact farther high-q promotion selected",
            sign_base.truth(gate_a_exact_farther_high_q_promotion_selected),
            "The current retained 4-term family is still not an exact farther high-q theorem.",
        ),
        sign_base.row(
            "gate_b_alias_harmonic_spike_selected",
            "pass" if gate_b_alias_harmonic_spike_selected else "reject",
            "Gate B alias-harmonic spike selected",
            sign_base.truth(gate_b_alias_harmonic_spike_selected),
            "After resolving the root-floor bookkeeping, the honest residual blocker is the alias-harmonic spike family.",
        ),
        sign_base.row(
            "gate_c_current_rule_blocked",
            "reject" if not gate_c_current_rule_blocked else "pass",
            "Gate C current rule blocked",
            sign_base.truth(gate_c_current_rule_blocked),
            "The current retained pack is not globally blocked because the root-floor family itself is no longer unresolved.",
        ),
        sign_base.row(
            "same_level_root_floor_retry_admissible",
            "reject",
            "same-level root-floor retry admissible",
            sign_base.truth(same_level_root_floor_retry_admissible),
            "The raw root-floor family has been resolved and should not be retried at the same level.",
        ),
        sign_base.row(
            "same_level_microphase_retry_admissible",
            "reject",
            "same-level microphase retry admissible",
            sign_base.truth(same_level_microphase_retry_admissible),
            "Once the shared envelope-floor candidate is admitted, generic microphase retry is no longer the honest next move.",
        ),
        sign_base.row(
            "alias_harmonic_spike_audit_admissible_now",
            "pass",
            "alias-harmonic spike audit admissible now",
            sign_base.truth(alias_harmonic_spike_audit_admissible_now),
            "The next official branch is now the boundary alias-harmonic spike audit.",
        ),
        sign_base.row(
            "substantive_pack_update_required_now",
            "reject" if not substantive_pack_update_required_now else "pass",
            "substantive pack update required now",
            sign_base.truth(substantive_pack_update_required_now),
            "The next move is still an internal signed-rule theorem audit rather than an immediate pack update.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "gate_a_exact_farther_high_q_promotion_selected": gate_a_exact_farther_high_q_promotion_selected,
        "gate_b_alias_harmonic_spike_selected": gate_b_alias_harmonic_spike_selected,
        "gate_c_current_rule_blocked": gate_c_current_rule_blocked,
        "same_level_root_floor_retry_admissible": same_level_root_floor_retry_admissible,
        "same_level_microphase_retry_admissible": same_level_microphase_retry_admissible,
        "alias_harmonic_spike_audit_admissible_now": alias_harmonic_spike_audit_admissible_now,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "q_nyquist_box_over_m0": float(prior_summary["q_nyquist_box_over_m0"]),
        "first_alias_harmonic_over_m0": float(prior_summary["first_alias_harmonic_over_m0"]),
        "second_alias_harmonic_over_m0": float(prior_summary["second_alias_harmonic_over_m0"]),
        "fit_raw_root_duplication_ratio": float(prior_summary["fit_raw_root_duplication_ratio"]),
        "floor_raw_root_duplication_ratio": float(prior_summary["floor_raw_root_duplication_ratio"]),
        "micro_raw_root_duplication_ratio": float(prior_summary["micro_raw_root_duplication_ratio"]),
        "edge_raw_root_duplication_ratio": float(prior_summary["edge_raw_root_duplication_ratio"]),
        "fit_resolved_sign_mismatch_fraction": float(prior_summary["fit_resolved_sign_mismatch_fraction"]),
        "floor_resolved_sign_mismatch_fraction": float(prior_summary["floor_resolved_sign_mismatch_fraction"]),
        "micro_resolved_sign_mismatch_fraction": float(prior_summary["micro_resolved_sign_mismatch_fraction"]),
        "edge_resolved_sign_mismatch_fraction": float(prior_summary["edge_resolved_sign_mismatch_fraction"]),
        "best_envelope_floor_tau": float(prior_summary["best_envelope_floor_tau"]),
        "best_envelope_floor_combined_mismatch_fraction": float(prior_summary["best_envelope_floor_combined_mismatch_fraction"]),
        "best_envelope_floor_combined_keep_fraction": float(prior_summary["best_envelope_floor_combined_keep_fraction"]),
        "sign_root_floor_resolved": bool(prior_summary["sign_root_floor_resolved"]),
        "envelope_floor_candidate_admissible": bool(prior_summary["envelope_floor_candidate_admissible"]),
        "fit_alias_harmonic_window_detected": bool(prior_summary["fit_alias_harmonic_window_detected"]),
        "edge_alias_harmonic_window_detected": bool(prior_summary["edge_alias_harmonic_window_detected"]),
        "remaining_blocker_is_alias_harmonic_spike": bool(prior_summary["remaining_blocker_is_alias_harmonic_spike"]),
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2021",
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
            "overall_status": "vector_qball_form_factor_resolved_high_q_sign_root_gate_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2019-.2022"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.2019-.2022"),
                "current_problem_hit": sign_base.hit(
                    current_problem_text,
                    "resolved high-q sign-root floor / envelope-microphase split",
                ),
                "current_status_hit": sign_base.hit(
                    current_status_text,
                    "resolved high-q sign-root floor / envelope-microphase audit",
                ),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2019-.2022"),
                "long_roadmap_hit": sign_base.hit(
                    long_text,
                    "resolved high-q sign-root decision gate / registry",
                ),
                "part5_hit": sign_base.hit(part5_text, ".2007-.2014"),
            },
        },
    )

    route_rows = [
        sign_base.row(
            "gate_b_alias_harmonic_spike_selected",
            "pass" if gate_b_alias_harmonic_spike_selected else "reject",
            "Gate B alias-harmonic spike selected",
            sign_base.truth(gate_b_alias_harmonic_spike_selected),
            "The next official branch is justified only if the honest residual blocker has been narrowed to alias-harmonic spikes.",
        ),
        sign_base.row(
            "same_level_root_floor_retry_admissible",
            "reject",
            "same-level root-floor retry admissible",
            sign_base.truth(same_level_root_floor_retry_admissible),
            "The resolved root-floor family is closed and should not be reopened at the same level.",
        ),
        sign_base.row(
            "alias_harmonic_spike_audit_admissible_now",
            "pass",
            "alias-harmonic spike audit admissible now",
            sign_base.truth(alias_harmonic_spike_audit_admissible_now),
            "The next official branch is the boundary alias-harmonic spike audit.",
        ),
        sign_base.row(
            "next_route_fixed",
            "pass",
            "next route fixed",
            1.0,
            "The next official branch is the boundary alias-harmonic spike audit.",
        ),
    ]

    route_payload = sign_base.payload(
        "8.7.56.2022",
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
            "overall_status": "vector_qball_form_factor_resolved_high_q_sign_root_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_next_hit": sign_base.hit(status_text, "8.7.56.2019-.2022"),
                "roadmap_next_hit": sign_base.hit(roadmap_text, "8.7.56.2023-.2026"),
                "current_problem_next_hit": sign_base.hit(
                    current_problem_text,
                    "resolved high-q sign-root floor / envelope-microphase split",
                ),
                "current_status_next_hit": sign_base.hit(
                    current_status_text,
                    "resolved high-q sign-root floor / envelope-microphase audit",
                ),
                "unified_roadmap_next_hit": sign_base.hit(unified_text, ".2023-.2026"),
                "long_roadmap_next_hit": sign_base.hit(
                    long_text,
                    "boundary alias-harmonic spike audit",
                ),
                "part5_next_hit": sign_base.hit(part5_text, ".2007-.2014"),
            },
        },
    )

    declaration_paths = write_artifact("declaration_gate", declaration_payload)
    route_paths = write_artifact("route_sync", route_payload)
    print("[ok] 8.7.56.2019-.2022 resolved high-q sign-root gate artifacts generated")
    print(f"[ok] declaration: {declaration_paths['json']}")
    print(f"[ok] route sync:   {route_paths['json']}")


if __name__ == "__main__":
    main()
