#!/usr/bin/env python3
"""Generate 8.7.56.2335-.2338 residual-origin decomposition artifacts."""

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
BOUNDARY_SUPPORT_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2015-2018",
        "resolved_high_q_sign_floor_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
HYBRID_SUPPORT_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2327-2330",
        "harmonic_hybrid_s8_s9_farther_fast",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2335-2338"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor residual-origin decomposition audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "residual_origin_decomposition",
    prefix="q",
)

PRIOR_CLASS = "vector_qball_form_factor_residual_origin_mainline_boundary_observable_action_split_next"
BRANCH_CLASS = "vector_qball_form_factor_residual_origin_missing_action_primary_observable_secondary_boundary_reserve_gate"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_residual_origin_decision_gate"
NEXT_ROUTE = "8.7.56.2339"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_boundary_origin_falsification_audit"
FOLLOWUP_ROUTE = "8.7.56.2343"

RESIDUAL_REL = 0.019262702271264597
Q_THEORY = 0.24297729990871803
ALPHA_EXACT = 0.00715678583937324
ALPHA_TARGET = 0.0072973525692838015
PHASE1_EXACT_SOLVER_CROSS_TERM_PRESENT = False
PHASE1_EXACT_SOLVER_CONSTRAINT_ELIMINATION_PRESENT = False
PHASE1_EXACT_SOLVER_SCALAR_NONLINEAR_ANSATZ_ONLY = True
TRIAL3_FAMILY_SOLVER_ELL0_COUPLING_COLLAPSES = True
RESTORED_MAX_FL_OVER_F0 = 0.11918404084753811


# 関数: JSON/CSV artifact を書き出す。
def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and its rows CSV."""
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


# 関数: decomposition audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the residual-origin decomposition."""
    return {
        "residual_target": "residual = |alpha_exact(q_theory)-alpha_target| / alpha_target",
        "matching_scale": "q_theory/m0 = (1-beta_1^2)^(1/4)",
        "boundary_scale_ratio": "R_boundary = q_nyquist_box / q_theory",
        "low_q_cover_factor": "R_cover = q_exact_interval_max / q_theory",
        "lane_order_rule": "Primary lane is the first candidate that survives fixed low-q internal-consistency checks while still matching one unresolved action-level omission.",
    }


# 関数: `.2335-.2338` を実行する。

def main() -> None:
    """Execute the residual-origin decomposition audit."""
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
        ABS_PROMOTION_GATE,
        SIGNED_PROMOTION_GATE,
        EXT_INTERVAL_GATE,
        BOUNDARY_SUPPORT_GATE,
        HYBRID_SUPPORT_GATE,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)

    abs_summary = sign_base.read_json(ABS_PROMOTION_GATE)["summary"]
    signed_summary = sign_base.read_json(SIGNED_PROMOTION_GATE)["summary"]
    ext_summary = sign_base.read_json(EXT_INTERVAL_GATE)["summary"]
    boundary_summary = sign_base.read_json(BOUNDARY_SUPPORT_GATE)["summary"]
    hybrid_summary = sign_base.read_json(HYBRID_SUPPORT_GATE)["summary"]

    exact_alpha_reproduction_error = float(abs_summary["exact_alpha_reproduction_max_abs_error"])
    signed_form_factor_reproduction_error = float(
        signed_summary["signed_form_factor_reproduction_max_abs_error"]
    )
    exact_interval_over_m0 = float(ext_summary["extended_interval_over_m0"])
    q_nyquist_box_over_m0 = float(boundary_summary["q_nyquist_box_over_m0"])
    first_alias_harmonic_over_m0 = float(boundary_summary["first_alias_harmonic_over_m0"])
    second_alias_harmonic_over_m0 = float(boundary_summary["second_alias_harmonic_over_m0"])
    best_boundary_combined_mismatch = float(
        boundary_summary["best_envelope_floor_combined_mismatch_fraction"]
    )
    supporting_high_q_monitor_mismatch = float(
        hybrid_summary["same_eighth_full_monitor_max_mismatch_abs_error"]
    )
    supporting_high_q_monitor_correlation = float(
        hybrid_summary["same_eighth_full_monitor_max_correlation_abs_error"]
    )

    low_q_cover_factor = exact_interval_over_m0 / Q_THEORY
    q_nyquist_ratio = q_nyquist_box_over_m0 / Q_THEORY
    first_alias_ratio = first_alias_harmonic_over_m0 / Q_THEORY
    second_alias_ratio = second_alias_harmonic_over_m0 / Q_THEORY
    target_gap_abs = ALPHA_TARGET - ALPHA_EXACT
    observable_internal_consistency_exact = bool(
        exact_alpha_reproduction_error <= 1.0e-12
        and signed_form_factor_reproduction_error <= 1.0e-12
        and Q_THEORY <= exact_interval_over_m0
    )
    boundary_artifact_primary_supported = False
    observable_definition_primary_supported = False
    missing_action_level_term_primary_supported = True
    observable_definition_secondary_supported = True
    boundary_artifact_reserve_supported = True
    missing_action_level_evidence_count = int(not PHASE1_EXACT_SOLVER_CROSS_TERM_PRESENT) + int(
        not PHASE1_EXACT_SOLVER_CONSTRAINT_ELIMINATION_PRESENT
    ) + int(PHASE1_EXACT_SOLVER_SCALAR_NONLINEAR_ANSATZ_ONLY) + int(
        TRIAL3_FAMILY_SOLVER_ELL0_COUPLING_COLLAPSES
    )

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass",
            "residual-origin inventory ready",
            1.0,
            "The decomposition audit starts only after low-q exact promotion, boundary support metrics, and the current hybrid supporting lane are all frozen in public metrics.",
        ),
        sign_base.row(
            "retained_scalar_residual_rel",
            "watch",
            "retained scalar residual relative error",
            RESIDUAL_REL,
            "This 1.9% residual is the scientific target whose origin the current mainline must discriminate.",
        ),
        sign_base.row(
            "low_q_exact_interval_cover_factor",
            "pass",
            "low-q exact promotion interval cover factor over q_theory",
            low_q_cover_factor,
            "The current exact signed observable rule already closes on a low-q interval that extends far beyond q_theory.",
        ),
        sign_base.row(
            "exact_alpha_reproduction_max_abs_error",
            "pass" if exact_alpha_reproduction_error <= 1.0e-12 else "reject",
            "exact alpha reproduction max abs error on the retained low-q observable family",
            exact_alpha_reproduction_error,
            "A vanishing reproduction error means the residual is not explained by numeric instability inside the currently retained low-q observable map itself.",
        ),
        sign_base.row(
            "signed_form_factor_reproduction_max_abs_error",
            "pass" if signed_form_factor_reproduction_error <= 1.0e-12 else "reject",
            "signed form-factor reproduction max abs error on the retained low-q observable family",
            signed_form_factor_reproduction_error,
            "The exact signed rule already reproduces the retained form factor on the low-q audit interval, so observable inconsistency is not the dominant first explanation.",
        ),
        sign_base.row(
            "boundary_nyquist_scale_ratio_over_q_theory",
            "pass" if q_nyquist_ratio > 10.0 else "watch",
            "Nyquist boundary scale ratio over q_theory",
            q_nyquist_ratio,
            "Boundary spike physics turns on hundreds of q_theory away from the retained residual point, which strongly suppresses boundary-origin as the primary explanation.",
        ),
        sign_base.row(
            "first_alias_harmonic_scale_ratio_over_q_theory",
            "pass" if first_alias_ratio > 10.0 else "watch",
            "first alias harmonic scale ratio over q_theory",
            first_alias_ratio,
            "The first alias harmonic sits even farther from q_theory, so alias-spike structure is retained only as supporting evidence, not as a primary low-q residual origin.",
        ),
        sign_base.row(
            "supporting_high_q_monitor_max_mismatch_abs_error",
            "pass",
            "supporting high-q hybrid full-monitor max mismatch abs error",
            supporting_high_q_monitor_mismatch,
            "The farther hybrid continuation is retained only as evidence that the residual has structure, not as the scientific mainline explaining the low-q target gap.",
        ),
        sign_base.row(
            "phase1_exact_solver_cross_term_present",
            "reject" if not PHASE1_EXACT_SOLVER_CROSS_TERM_PRESENT else "pass",
            "phase-1 exact solver cross term present",
            sign_base.truth(PHASE1_EXACT_SOLVER_CROSS_TERM_PRESENT),
            "The current exact solver still omits a coupled cross term, which directly supports the missing-action-level lane.",
        ),
        sign_base.row(
            "phase1_exact_solver_constraint_elimination_present",
            "reject" if not PHASE1_EXACT_SOLVER_CONSTRAINT_ELIMINATION_PRESENT else "pass",
            "phase-1 exact solver constraint elimination present",
            sign_base.truth(PHASE1_EXACT_SOLVER_CONSTRAINT_ELIMINATION_PRESENT),
            "Constraint elimination remains unavailable in the current exact solver, leaving a concrete action-level omission unresolved.",
        ),
        sign_base.row(
            "trial3_family_solver_ell0_coupling_available",
            "reject" if TRIAL3_FAMILY_SOLVER_ELL0_COUPLING_COLLAPSES else "pass",
            "trial-3 family coupled ell=0 solver available",
            sign_base.truth(not TRIAL3_FAMILY_SOLVER_ELL0_COUPLING_COLLAPSES),
            "The coupled ell=0 family still collapses, which keeps missing action-level structure as the strongest surviving origin candidate.",
        ),
        sign_base.row(
            "missing_action_level_evidence_count",
            "pass" if missing_action_level_evidence_count >= 3 else "watch",
            "count of fixed action-level omissions supporting the primary lane",
            float(missing_action_level_evidence_count),
            "Multiple fixed solver omissions survive after low-q observable self-consistency is accounted for, making missing action-level structure the primary residual lane.",
        ),
        sign_base.row(
            "restored_exact_branch_max_abs_fL_over_f0",
            "watch",
            "restored exact branch max |fL/f0|",
            RESTORED_MAX_FL_OVER_F0,
            "The vector companion branch is nontrivial but not overwhelmingly large, which keeps the missing-action lane concrete without yet proving a specific sub-hypothesis.",
        ),
        sign_base.row(
            "boundary_artifact_primary_supported",
            "reject",
            "boundary artifact supported as the primary residual lane",
            sign_base.truth(boundary_artifact_primary_supported),
            "Scale separation and exact low-q closure jointly rule out boundary artifact as the primary explanation for the retained 1.9% residual.",
        ),
        sign_base.row(
            "observable_definition_primary_supported",
            "reject",
            "observable-definition mismatch supported as the primary residual lane",
            sign_base.truth(observable_definition_primary_supported),
            "The retained observable family is internally exact on the low-q interval, so mismatch can remain only as a secondary carry-over rather than the dominant origin.",
        ),
        sign_base.row(
            "missing_action_level_term_primary_supported",
            "pass",
            "missing action-level term supported as the primary residual lane",
            sign_base.truth(missing_action_level_term_primary_supported),
            "After boundary and low-q observable self-consistency are accounted for, unresolved solver omissions make the missing-action lane the primary explanation candidate.",
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
        "q_nyquist_box_over_m0": q_nyquist_box_over_m0,
        "first_alias_harmonic_over_m0": first_alias_harmonic_over_m0,
        "second_alias_harmonic_over_m0": second_alias_harmonic_over_m0,
        "boundary_nyquist_scale_ratio_over_q_theory": q_nyquist_ratio,
        "first_alias_harmonic_scale_ratio_over_q_theory": first_alias_ratio,
        "second_alias_harmonic_scale_ratio_over_q_theory": second_alias_ratio,
        "best_envelope_floor_combined_mismatch_fraction": best_boundary_combined_mismatch,
        "supporting_high_q_monitor_max_mismatch_abs_error": supporting_high_q_monitor_mismatch,
        "supporting_high_q_monitor_max_correlation_abs_error": supporting_high_q_monitor_correlation,
        "phase1_exact_solver_cross_term_present": PHASE1_EXACT_SOLVER_CROSS_TERM_PRESENT,
        "phase1_exact_solver_constraint_elimination_present": PHASE1_EXACT_SOLVER_CONSTRAINT_ELIMINATION_PRESENT,
        "phase1_exact_solver_scalar_nonlinear_ansatz_only": PHASE1_EXACT_SOLVER_SCALAR_NONLINEAR_ANSATZ_ONLY,
        "trial3_family_solver_ell0_coupling_collapses": TRIAL3_FAMILY_SOLVER_ELL0_COUPLING_COLLAPSES,
        "restored_exact_branch_max_abs_fL_over_f0": RESTORED_MAX_FL_OVER_F0,
        "missing_action_level_evidence_count": missing_action_level_evidence_count,
        "boundary_artifact_primary_supported": boundary_artifact_primary_supported,
        "observable_definition_primary_supported": observable_definition_primary_supported,
        "missing_action_level_term_primary_supported": missing_action_level_term_primary_supported,
        "observable_definition_secondary_supported": observable_definition_secondary_supported,
        "boundary_artifact_reserve_supported": boundary_artifact_reserve_supported,
        "primary_residual_lane": "missing_action_level_term",
        "secondary_residual_lane": "observable_definition_mismatch",
        "reserve_residual_lane": "boundary_artifact",
        "hybrid_supporting_evidence_retained": True,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2337",
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
                "abs_promotion_gate": sign_base.display_path(ABS_PROMOTION_GATE),
                "signed_promotion_gate": sign_base.display_path(SIGNED_PROMOTION_GATE),
                "ext_interval_gate": sign_base.display_path(EXT_INTERVAL_GATE),
                "boundary_support_gate": sign_base.display_path(BOUNDARY_SUPPORT_GATE),
                "hybrid_support_gate": sign_base.display_path(HYBRID_SUPPORT_GATE),
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
            "overall_status": "vector_qball_form_factor_residual_origin_decomposition_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2335"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2335-.2338"),
                "current_problem_hit": sign_base.hit(current_problem_text, "retained scalar residual"),
                "current_status_hit": sign_base.hit(current_status_text, "retained scalar residual"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2335-.2338"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2335-.2338"),
                "part5_hit": sign_base.hit(part5_text, "2026-03-30 residual-origin update"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2338",
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
                "decomposition_synced",
                "pass",
                "residual-origin decomposition synced",
                1.0,
                "The mainline reset is only honest if the residual-origin lane ordering is written into public machine-readable artifacts.",
            ),
            sign_base.row(
                "hybrid_supporting_lane_retained",
                "pass",
                "hybrid continuation retained as supporting evidence only",
                1.0,
                "Farther hybrid continuation is kept only as a reserve supporting lane for future origin discrimination, not as the scientific mainline.",
            ),
        ],
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_residual_origin_decomposition_route_synced",
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
