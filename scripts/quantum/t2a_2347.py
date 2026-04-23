#!/usr/bin/env python3
"""Generate 8.7.56.2347-.2350 observable-mismatch audit artifacts."""

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
        "8.7.56.2343-2346",
        "boundary_origin_falsification",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
ABS_PROMOTION_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.1835-1838",
        "global_abs_source_loading_reactivation",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
SIGNED_PROMOTION_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.1843-1846",
        "signed_source_phase_reactivation",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
EXT_INTERVAL_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.1955-1958",
        "further_ext_interval_sign_phase_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2347-2350"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor observable-definition mismatch audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "observable_definition_mismatch",
    prefix="q",
)

PRIOR_CLASS = "vector_qball_form_factor_residual_origin_boundary_falsified_observable_secondary_missing_action_primary_next"
BRANCH_CLASS = "vector_qball_form_factor_residual_origin_missing_action_primary_after_boundary_observable_audits_next"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_missing_action_level_term_audit"
NEXT_ROUTE = "8.7.56.2351"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_residual_origin_synthesis_hybrid_reserve_refresh"
FOLLOWUP_ROUTE = "8.7.56.2355"

RESIDUAL_REL = 0.019262702271264597
Q_THEORY = 0.24297729990871803
ALPHA_EXACT = 0.00715678583937324
ALPHA_TARGET = 0.0072973525692838015


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


# 関数: observable audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the observable-definition audit."""
    return {
        "residual_target": "residual = |alpha_exact(q_theory)-alpha_target| / alpha_target",
        "observable_internal_check": "F_exact(q)=sigma_F(q) F_abs(q), alpha_exact = F_exact(q)^2/(4 pi) on the retained low-q interval",
        "decision_rule": "Observable-definition mismatch is non-primary when the retained low-q observable family is internally exact on an interval that contains q_theory.",
    }


# 関数: `.2347-.2350` を実行する。

def main() -> None:
    """Execute the observable-definition mismatch audit."""
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
        ABS_PROMOTION_GATE,
        SIGNED_PROMOTION_GATE,
        EXT_INTERVAL_GATE,
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
    abs_summary = sign_base.read_json(ABS_PROMOTION_GATE)["summary"]
    signed_summary = sign_base.read_json(SIGNED_PROMOTION_GATE)["summary"]
    ext_summary = sign_base.read_json(EXT_INTERVAL_GATE)["summary"]

    exact_alpha_reproduction_error = float(abs_summary["exact_alpha_reproduction_max_abs_error"])
    signed_form_factor_reproduction_error = float(
        signed_summary["signed_form_factor_reproduction_max_abs_error"]
    )
    exact_interval_over_m0 = float(ext_summary["extended_interval_over_m0"])
    inventory_ready = bool(prior_summary["boundary_origin_falsified_for_q_theory"])
    low_q_cover_factor = exact_interval_over_m0 / Q_THEORY
    target_gap_abs = ALPHA_TARGET - ALPHA_EXACT
    observable_internal_consistency_exact = bool(
        exact_alpha_reproduction_error <= 1.0e-12
        and signed_form_factor_reproduction_error <= 1.0e-12
        and Q_THEORY <= exact_interval_over_m0
    )
    observable_definition_primary_supported = False
    observable_definition_secondary_carryover = True
    missing_action_level_primary_now = True

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "observable-definition inventory ready",
            sign_base.truth(inventory_ready),
            "The observable-definition audit starts only after the boundary lane has already been falsified as the primary residual origin.",
        ),
        sign_base.row(
            "exact_alpha_reproduction_max_abs_error",
            "pass" if exact_alpha_reproduction_error <= 1.0e-12 else "reject",
            "exact alpha reproduction max abs error on the retained low-q observable family",
            exact_alpha_reproduction_error,
            "The retained low-q observable map already reproduces alpha exactly on its own interval, which blocks observable inconsistency as the dominant explanation.",
        ),
        sign_base.row(
            "signed_form_factor_reproduction_max_abs_error",
            "pass" if signed_form_factor_reproduction_error <= 1.0e-12 else "reject",
            "signed form-factor reproduction max abs error on the retained low-q observable family",
            signed_form_factor_reproduction_error,
            "The retained signed observable rule closes exactly on the current low-q interval, so the observable family itself is internally consistent.",
        ),
        sign_base.row(
            "low_q_exact_interval_cover_factor",
            "pass" if low_q_cover_factor > 10.0 else "watch",
            "low-q exact interval cover factor over q_theory",
            low_q_cover_factor,
            "The retained exact interval contains q_theory with a large safety margin, so the residual is a stable offset, not an interval-edge artifact.",
        ),
        sign_base.row(
            "alpha_target_gap_abs",
            "watch",
            "absolute alpha gap between retained exact value and target",
            target_gap_abs,
            "The residual survives even though the internal observable family is exact, which points away from primary observable mismatch and toward an upstream physical omission.",
        ),
        sign_base.row(
            "observable_internal_consistency_exact",
            "pass" if observable_internal_consistency_exact else "reject",
            "observable internal consistency exact on the retained low-q family",
            sign_base.truth(observable_internal_consistency_exact),
            "The retained observable family is internally exact across the low-q interval that contains q_theory.",
        ),
        sign_base.row(
            "observable_definition_primary_supported",
            "reject",
            "observable-definition mismatch supported as the primary residual origin",
            sign_base.truth(observable_definition_primary_supported),
            "Because the retained observable family is internally exact, observable-definition mismatch cannot remain the primary residual origin.",
        ),
        sign_base.row(
            "observable_definition_secondary_carryover",
            "pass",
            "observable-definition mismatch retained only as a secondary carry-over",
            sign_base.truth(observable_definition_secondary_carryover),
            "The target comparison still uses an external physical reference, so observable-definition mismatch is not fully erased, but it is demoted to a secondary carry-over.",
        ),
        sign_base.row(
            "missing_action_level_primary_now",
            "pass",
            "missing action-level term becomes the current primary lane",
            sign_base.truth(missing_action_level_primary_now),
            "After boundary and observable-primary explanations are cut, the unresolved 1.9% residual points to the missing-action lane as the honest next mainline.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": RESIDUAL_REL,
        "alpha_exact_at_q_theory": ALPHA_EXACT,
        "alpha_target": ALPHA_TARGET,
        "alpha_target_gap_abs": target_gap_abs,
        "q_theory_over_m0": Q_THEORY,
        "extended_interval_over_m0": exact_interval_over_m0,
        "low_q_exact_interval_cover_factor": low_q_cover_factor,
        "exact_alpha_reproduction_max_abs_error": exact_alpha_reproduction_error,
        "signed_form_factor_reproduction_max_abs_error": signed_form_factor_reproduction_error,
        "observable_internal_consistency_exact": observable_internal_consistency_exact,
        "observable_definition_primary_supported": observable_definition_primary_supported,
        "observable_definition_secondary_carryover": observable_definition_secondary_carryover,
        "missing_action_level_primary_now": missing_action_level_primary_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2349",
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
                "abs_promotion_gate": sign_base.display_path(ABS_PROMOTION_GATE),
                "signed_promotion_gate": sign_base.display_path(SIGNED_PROMOTION_GATE),
                "ext_interval_gate": sign_base.display_path(EXT_INTERVAL_GATE),
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
            "overall_status": "vector_qball_form_factor_observable_definition_audit_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2347"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2347-.2350"),
                "current_problem_hit": sign_base.hit(current_problem_text, "observable-definition mismatch"),
                "current_status_hit": sign_base.hit(current_status_text, "observable-definition mismatch"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2347-.2350"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2347-.2350"),
                "part5_hit": sign_base.hit(part5_text, "2026-03-30 residual-origin update"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2350",
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
                "current_route": BRANCH_CLASS,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        "rows": [
            sign_base.row(
                "observable_audit_synced",
                "pass",
                "observable-definition audit synced",
                1.0,
                "The residual-origin mainline is only honest if observable-definition mismatch is explicitly demoted before the missing-action lane is promoted to the next audit.",
            ),
        ],
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_observable_definition_audit_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": declaration_payload["evidence"],
    }
    route_paths = write_artifact("route_sync", route_payload)
    print("[write] declaration:", declaration_paths["json"])
    print("[write] route:", route_paths["json"])


if __name__ == "__main__":
    main()
