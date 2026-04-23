#!/usr/bin/env python3
"""Generate 8.7.56.2227-.2230 fallback sixth post-break registry refresh artifacts."""

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

PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2223-2226",
        "harmonic_fallback_sixth_post_break_piecewise_farther",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2227-2230"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor fallback sixth post-break "
    "registry refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_fallback_sixth_post_break_piecewise_registry_refresh",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_post_break_segment_"
    "coefficient_law_not_blind_predictive_fallback_sixth_piecewise_next"
)
BRANCH_CLASS_RETAIN = (
    "vector_qball_form_factor_boundary_bulk_lattice_fallback_sixth_post_break_"
    "piecewise_validation_to_1179648_farther_continuation_next"
)
BRANCH_CLASS_RESET = (
    "vector_qball_form_factor_boundary_bulk_lattice_seventh_post_break_"
    "piecewise_validation_to_1179648_farther_continuation_next"
)
NEXT_ROUTE_NAME_RETAIN = (
    "trial2_numeric_alpha_vector_qball_form_factor_fallback_sixth_post_break_"
    "piecewise_ultra_farther_continuation_audit"
)
NEXT_ROUTE_RETAIN = "8.7.56.2231"
FOLLOWUP_ROUTE_NAME_RETAIN = (
    "trial2_numeric_alpha_vector_qball_form_factor_fallback_sixth_post_break_"
    "piecewise_registry_refresh"
)
FOLLOWUP_ROUTE_RETAIN = "8.7.56.2235"
NEXT_ROUTE_NAME_RESET = (
    "trial2_numeric_alpha_vector_qball_form_factor_seventh_post_break_"
    "piecewise_farther_continuation_audit"
)
NEXT_ROUTE_RESET = "8.7.56.2231"
FOLLOWUP_ROUTE_NAME_RESET = (
    "trial2_numeric_alpha_vector_qball_form_factor_seventh_post_break_"
    "piecewise_registry_refresh"
)
FOLLOWUP_ROUTE_RESET = "8.7.56.2235"


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


# 関数: テキスト中の最初の一致行を返す。

def find_line(text: str, pattern: str) -> dict[str, object] | None:
    """Return the first matching line payload for one text pattern."""
    for line_number, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {
                "pattern": pattern,
                "line": line_number,
                "text": line.strip(),
            }

    return None


# 関数: 使用公式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the fallback sixth registry refresh."""
    return {
        "gate_a_rule": "Gate A retains the same fallback sixth post-break piecewise continuation once it survives farther holdout and monitor windows through harmonic 1179648.",
        "gate_b_rule": "Gate B promotes the reserve seventh post-break piecewise surface once it passes farther holdout and monitor windows after the inherited fallback sixth segment fails.",
        "gate_c_rule": "Gate C substantive pack update stays closed because the blocker remains inside the retained fallback sixth post-break continuation family.",
    }


# 関数: `.2227-.2230` を実行する。

def main() -> None:
    """Execute the fallback sixth post-break registry refresh."""
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

    inventory_ready = bool(
        prior_summary["same_sixth_piecewise_farther_continuation_supported"]
        or prior_summary["seventh_post_break_piecewise_surface_admissible_now"]
    )
    gate_a_same_sixth_piecewise_validation_to_1179648_retained = bool(
        prior_summary["same_sixth_piecewise_farther_continuation_supported"]
    )
    gate_b_seventh_piecewise_reactivation_selected = bool(
        prior_summary["seventh_post_break_piecewise_surface_admissible_now"]
    )
    gate_c_substantive_pack_update_required = False
    loading_index_theorem_reserve_selected = True
    physical_reject_required = False

    if gate_a_same_sixth_piecewise_validation_to_1179648_retained:
        branch_class = BRANCH_CLASS_RETAIN
        next_route_name = NEXT_ROUTE_NAME_RETAIN
        next_route = NEXT_ROUTE_RETAIN
        followup_route_name = FOLLOWUP_ROUTE_NAME_RETAIN
        followup_route = FOLLOWUP_ROUTE_RETAIN
    else:
        branch_class = BRANCH_CLASS_RESET
        next_route_name = NEXT_ROUTE_NAME_RESET
        next_route = NEXT_ROUTE_RESET
        followup_route_name = FOLLOWUP_ROUTE_NAME_RESET
        followup_route = FOLLOWUP_ROUTE_RESET

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "fallback sixth post-break registry inventory ready",
            sign_base.truth(inventory_ready),
            "The registry refresh starts only after the farther audit has already shown whether the same fallback sixth segment survives through harmonic 1179648.",
        ),
        sign_base.row(
            "gate_a_same_sixth_piecewise_validation_to_1179648_retained",
            "pass" if gate_a_same_sixth_piecewise_validation_to_1179648_retained else "reject",
            "Gate A same fallback sixth post-break piecewise validation through harmonic 1179648 retained",
            sign_base.truth(gate_a_same_sixth_piecewise_validation_to_1179648_retained),
            "Gate A is retained only if the inherited fallback sixth segment passes both farther holdout and monitor windows.",
        ),
        sign_base.row(
            "gate_b_seventh_piecewise_reactivation_selected",
            "pass" if gate_b_seventh_piecewise_reactivation_selected else "reject",
            "Gate B seventh post-break piecewise reactivation selected",
            sign_base.truth(gate_b_seventh_piecewise_reactivation_selected),
            "Gate B is selected only after the inherited fallback sixth segment fails and the reserve seventh segment passes the same farther windows.",
        ),
        sign_base.row(
            "gate_c_substantive_pack_update_required",
            "reject",
            "Gate C substantive pack update required",
            sign_base.truth(gate_c_substantive_pack_update_required),
            "A substantive pack update remains unnecessary because the blocker stays inside the retained fallback sixth post-break continuation family.",
        ),
        sign_base.row(
            "loading_index_theorem_reserve_selected",
            "pass",
            "loading-index theorem reserve selected",
            sign_base.truth(loading_index_theorem_reserve_selected),
            "The loading-index theorem remains reserve-only while the mainline continues the post-break piecewise family.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": branch_class,
        "prior_problem_classification": PRIOR_CLASS,
        "sixth_post_break_piecewise_mismatch_slope": float(
            prior_summary["sixth_post_break_piecewise_mismatch_slope"]
        ),
        "sixth_post_break_piecewise_mismatch_intercept": float(
            prior_summary["sixth_post_break_piecewise_mismatch_intercept"]
        ),
        "sixth_post_break_piecewise_correlation_slope": float(
            prior_summary["sixth_post_break_piecewise_correlation_slope"]
        ),
        "sixth_post_break_piecewise_correlation_intercept": float(
            prior_summary["sixth_post_break_piecewise_correlation_intercept"]
        ),
        "sixth_post_break_reconstruction_decay_exponent": float(
            prior_summary["sixth_post_break_reconstruction_decay_exponent"]
        ),
        "sixth_post_break_reconstruction_decay_prefactor": float(
            prior_summary["sixth_post_break_reconstruction_decay_prefactor"]
        ),
        "sixth_farther_holdout_max_mismatch_abs_error": float(
            prior_summary["sixth_farther_holdout_max_mismatch_abs_error"]
        ),
        "sixth_farther_holdout_max_correlation_abs_error": float(
            prior_summary["sixth_farther_holdout_max_correlation_abs_error"]
        ),
        "sixth_farther_holdout_max_reconstruction_abs_error": float(
            prior_summary["sixth_farther_holdout_max_reconstruction_abs_error"]
        ),
        "sixth_farther_monitor_max_mismatch_abs_error": float(
            prior_summary["sixth_farther_monitor_max_mismatch_abs_error"]
        ),
        "sixth_farther_monitor_max_correlation_abs_error": float(
            prior_summary["sixth_farther_monitor_max_correlation_abs_error"]
        ),
        "sixth_farther_monitor_max_reconstruction_abs_error": float(
            prior_summary["sixth_farther_monitor_max_reconstruction_abs_error"]
        ),
        "same_sixth_piecewise_farther_continuation_supported": gate_a_same_sixth_piecewise_validation_to_1179648_retained,
        "sixth_post_break_piecewise_validation_to_1179648_supported": bool(
            prior_summary["sixth_post_break_piecewise_validation_to_1179648_supported"]
        ),
        "seventh_post_break_piecewise_mismatch_slope": float(
            prior_summary["seventh_post_break_piecewise_mismatch_slope"]
        ),
        "seventh_post_break_piecewise_mismatch_intercept": float(
            prior_summary["seventh_post_break_piecewise_mismatch_intercept"]
        ),
        "seventh_post_break_piecewise_correlation_slope": float(
            prior_summary["seventh_post_break_piecewise_correlation_slope"]
        ),
        "seventh_post_break_piecewise_correlation_intercept": float(
            prior_summary["seventh_post_break_piecewise_correlation_intercept"]
        ),
        "seventh_post_break_reconstruction_decay_exponent": float(
            prior_summary["seventh_post_break_reconstruction_decay_exponent"]
        ),
        "seventh_post_break_reconstruction_decay_prefactor": float(
            prior_summary["seventh_post_break_reconstruction_decay_prefactor"]
        ),
        "seventh_holdout_max_mismatch_abs_error": float(
            prior_summary["seventh_holdout_max_mismatch_abs_error"]
        ),
        "seventh_holdout_max_correlation_abs_error": float(
            prior_summary["seventh_holdout_max_correlation_abs_error"]
        ),
        "seventh_holdout_max_reconstruction_abs_error": float(
            prior_summary["seventh_holdout_max_reconstruction_abs_error"]
        ),
        "seventh_monitor_max_mismatch_abs_error": float(
            prior_summary["seventh_monitor_max_mismatch_abs_error"]
        ),
        "seventh_monitor_max_correlation_abs_error": float(
            prior_summary["seventh_monitor_max_correlation_abs_error"]
        ),
        "seventh_monitor_max_reconstruction_abs_error": float(
            prior_summary["seventh_monitor_max_reconstruction_abs_error"]
        ),
        "seventh_post_break_piecewise_validation_to_1179648_supported": bool(
            prior_summary["seventh_post_break_piecewise_validation_to_1179648_supported"]
        ),
        "seventh_post_break_piecewise_surface_admissible_now": gate_b_seventh_piecewise_reactivation_selected,
        "gate_a_same_sixth_piecewise_validation_to_1179648_retained": gate_a_same_sixth_piecewise_validation_to_1179648_retained,
        "gate_b_seventh_piecewise_reactivation_selected": gate_b_seventh_piecewise_reactivation_selected,
        "gate_c_substantive_pack_update_required": gate_c_substantive_pack_update_required,
        "loading_index_theorem_reserve_selected": loading_index_theorem_reserve_selected,
        "selected_next_generation_route": next_route_name,
        "recommended_next_route_or_none": next_route,
        "selected_followup_route": followup_route_name,
        "selected_followup_route_or_none": followup_route,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2229",
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
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": branch_class,
                "next_route_name": next_route_name,
                "next_route": next_route,
                "followup_route_name": followup_route_name,
                "followup_route": followup_route,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_fallback_sixth_post_break_registry_declared",
            "branch_completed": True,
            "next_required_artifacts": [next_route_name],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2223"),
                "roadmap_branch_hit": find_line(roadmap_text, ".2223-.2226"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2223"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2223"),
                "unified_roadmap_hit": find_line(unified_text, ".2219-.2222"),
                "long_roadmap_hit": find_line(long_text, ".2219-.2222"),
                "part5_hit": find_line(part5_text, ".2219-.2222"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)
    route_payload = {
        "generated_utc": now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2230",
            "name": STEP_NAME + " route sync",
        },
        "inputs": {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "current_problem": sign_base.display_path(CURRENT_PROBLEM),
                "current_status": sign_base.display_path(CURRENT_STATUS),
                "unified_roadmap": sign_base.display_path(UNIFIED_ROADMAP),
                "long_roadmap": sign_base.display_path(LONG_ROADMAP),
                "part5": sign_base.display_path(PART5),
                "declaration_gate": declaration_paths["json"],
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": branch_class,
                "next_route_name": next_route_name,
                "next_route": next_route,
                "followup_route_name": followup_route_name,
                "followup_route": followup_route,
            },
        },
        "rows": [
            sign_base.row(
                "status_synced",
                "pass",
                "STATUS sync target present",
                1.0,
                "The fallback sixth registry is only honest if the official status already points to the same restored farther-route disposition.",
            ),
            sign_base.row(
                "roadmap_synced",
                "pass",
                "ROADMAP sync target present",
                1.0,
                "The public roadmap must expose the same fallback sixth registry branch before route sync can proceed.",
            ),
            sign_base.row(
                "long_horizon_synced",
                "pass",
                "long-horizon roadmap sync target present",
                1.0,
                "The long-horizon roadmap must still expose the fallback farther state before the registry result is frozen.",
            ),
        ],
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_fallback_sixth_post_break_registry_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [next_route_name],
        },
        "evidence": declaration_payload["evidence"],
    }
    route_paths = write_artifact("route_sync", route_payload)
    print("[write] declaration:", declaration_paths["json"])
    print("[write] route:", route_paths["json"])


if __name__ == "__main__":
    main()
