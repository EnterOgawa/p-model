#!/usr/bin/env python3
"""Generate 8.7.56.2483-.2486 updated-pack exact source-theorem gate artifacts."""

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
        "8.7.56.2479-2482",
        "updated_pack_exact_source_theorem_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2483-2486"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor updated-pack exact source-theorem gate"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_exact_source_theorem_gate",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_exact_source_"
    "theorem_refresh_audited_background_expansion_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_background_"
    "expansion_primary_charge_current_secondary_blind_vector_reserve_next"
)
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_qball_background_expansion_audit"
NEXT_ROUTE = "8.7.56.2487"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_qball_background_expansion_gate_charge_current_refresh"
FOLLOWUP_ROUTE = "8.7.56.2491"


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


# 関数: gate で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the updated-pack exact source-theorem gate."""
    return {
        "gate_a": "Gate A = updated-pack exact source-theorem refresh surface explicit",
        "gate_b": "Gate B = updated-pack exact Q-ball background expansion primary selected",
        "gate_c": "Gate C = blind vector computation primary admissible now",
        "ordered_refresh": "background expansion -> charge-current closure -> low-order J_eff^0 synthesis -> blind vector refresh",
    }


# 関数: `.2483-.2486` を実行する。

def main() -> None:
    """Execute the updated-pack exact source-theorem decision gate."""
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

    gate_a_updated_pack_exact_source_theorem_refresh_surface_explicit = bool(
        prior_summary["updated_pack_exact_source_theorem_refresh_surface_explicit_now"]
    )
    gate_b_updated_pack_exact_qball_background_expansion_primary_selected = bool(
        prior_summary["updated_pack_background_expansion_primary_refresh_supported"]
        and prior_summary["updated_pack_exact_source_theorem_refresh_order_stable"]
        and not prior_summary["updated_pack_exact_source_theorem_derived_now"]
    )
    gate_c_blind_vector_computation_primary_admissible_now = bool(
        not prior_summary["blind_vector_observable_gate_still_blocked"]
    )
    exact_source_theorem_derived_now = bool(
        prior_summary["updated_pack_exact_source_theorem_derived_now"]
    )
    farther_hybrid_continuation_reopen_required_now = bool(
        prior_summary["farther_hybrid_continuation_reopen_required_now"]
    )
    old_density_proxy_eigenvalue_retry_admissible_now = False

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_source_theorem_refresh_surface_explicit",
            "pass" if gate_a_updated_pack_exact_source_theorem_refresh_surface_explicit else "reject",
            "Gate A updated-pack exact source-theorem refresh surface explicit",
            sign_base.truth(gate_a_updated_pack_exact_source_theorem_refresh_surface_explicit),
            "The refresh audit already compressed the theorem blocker into one explicit ordered surface.",
        ),
        sign_base.row(
            "gate_b_updated_pack_exact_qball_background_expansion_primary_selected",
            "pass" if gate_b_updated_pack_exact_qball_background_expansion_primary_selected else "reject",
            "Gate B updated-pack exact Q-ball background expansion primary selected",
            sign_base.truth(gate_b_updated_pack_exact_qball_background_expansion_primary_selected),
            "The ordered refresh lane now makes the Q-ball background expansion the first honest mainline object.",
        ),
        sign_base.row(
            "gate_c_blind_vector_computation_primary_admissible_now",
            "pass" if gate_c_blind_vector_computation_primary_admissible_now else "reject",
            "Gate C blind vector computation primary admissible now",
            sign_base.truth(gate_c_blind_vector_computation_primary_admissible_now),
            "Blind vector computation stays downstream until the background-expansion and charge-current prerequisites move first.",
        ),
        sign_base.row(
            "exact_source_theorem_derived_now",
            "pass" if exact_source_theorem_derived_now else "reject",
            "exact source theorem derived now",
            sign_base.truth(exact_source_theorem_derived_now),
            "The gate synchronizes that ordering the theorem refresh lane does not itself derive the exact theorem.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains reserve-only because the blocker is still on the theorem-side prerequisite stack.",
        ),
        sign_base.row(
            "old_density_proxy_eigenvalue_retry_admissible_now",
            "pass" if old_density_proxy_eigenvalue_retry_admissible_now else "reject",
            "old density/proxy/eigenvalue retry admissible now",
            sign_base.truth(old_density_proxy_eigenvalue_retry_admissible_now),
            "The ordered updated-pack route still does not reopen exhausted pre-update retry families.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "gate_a_updated_pack_exact_source_theorem_refresh_surface_explicit": gate_a_updated_pack_exact_source_theorem_refresh_surface_explicit,
        "gate_b_updated_pack_exact_qball_background_expansion_primary_selected": gate_b_updated_pack_exact_qball_background_expansion_primary_selected,
        "gate_c_blind_vector_computation_primary_admissible_now": gate_c_blind_vector_computation_primary_admissible_now,
        "exact_source_theorem_derived_now": exact_source_theorem_derived_now,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "old_density_proxy_eigenvalue_retry_admissible_now": old_density_proxy_eigenvalue_retry_admissible_now,
        "hybrid_supporting_evidence_reopen_required": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_exact_qball_background_expansion_audit",
        "selected_secondary_pack_update_surface": "updated_pack_exact_charge_current_noether_refresh",
        "selected_tertiary_pack_update_surface": "updated_pack_exact_low_order_jeff0_formula_synthesis",
        "selected_reserve_completion_lane": "blind_vector_after_background_expansion_and_charge_current_refresh",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2485",
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
                "current_route": BRANCH_CLASS,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_exact_source_theorem_gate_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, ".2483-.2486"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2483-.2486"),
                "current_problem_hit": sign_base.hit(current_problem_text, "updated-pack exact source-theorem gate / blind-vector refresh"),
                "current_status_hit": sign_base.hit(current_status_text, "updated-pack exact source-theorem gate / blind-vector refresh"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2483-.2486"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2483-.2486"),
                "part5_hit": sign_base.hit(part5_text, ".2483-.2486"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2486",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_exact_source_theorem_gate_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulae": build_formulae(),
            "disposition": {
                "exact_source_theorem_refresh_surface_explicit": gate_a_updated_pack_exact_source_theorem_refresh_surface_explicit,
                "background_expansion_primary_selected": gate_b_updated_pack_exact_qball_background_expansion_primary_selected,
                "blind_vector_still_downstream": not gate_c_blind_vector_computation_primary_admissible_now,
            },
        },
    }
    route_paths = write_artifact("route_sync", route_payload)

    print("[ok] updated-pack exact source-theorem gate artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
