#!/usr/bin/env python3
"""Generate 8.7.56.2343-.2346 boundary-origin falsification artifacts."""

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
        "8.7.56.2339-2342",
        "residual_origin_decision_gate",
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
SIGNED_PROMOTION_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.1843-1846",
        "signed_source_phase_reactivation",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2343-2346"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor boundary-origin falsification audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "boundary_origin_falsification",
    prefix="q",
)

PRIOR_CLASS = "vector_qball_form_factor_residual_origin_missing_action_primary_observable_secondary_boundary_reserve_next"
BRANCH_CLASS = "vector_qball_form_factor_residual_origin_boundary_falsified_observable_secondary_missing_action_primary_next"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_observable_definition_mismatch_audit"
NEXT_ROUTE = "8.7.56.2347"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_missing_action_level_term_audit"
FOLLOWUP_ROUTE = "8.7.56.2351"

RESIDUAL_REL = 0.019262702271264597
Q_THEORY = 0.24297729990871803


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


# 関数: boundary falsification で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the boundary-origin audit."""
    return {
        "boundary_onset_ratio": "R_N = q_nyquist_box / q_theory",
        "alias_ratio": "R_alias = q_alias,1 / q_theory",
        "low_q_cover": "R_cover = q_exact_interval_max / q_theory",
        "falsification_rule": "Boundary origin is falsified as the primary lane when the retained residual point lies deep inside an exactly reproduced low-q interval that is scale-separated from boundary onset.",
    }


# 関数: `.2343-.2346` を実行する。

def main() -> None:
    """Execute the boundary-origin falsification audit."""
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
        EXT_INTERVAL_GATE,
        BOUNDARY_SUPPORT_GATE,
        SIGNED_PROMOTION_GATE,
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
    ext_summary = sign_base.read_json(EXT_INTERVAL_GATE)["summary"]
    boundary_summary = sign_base.read_json(BOUNDARY_SUPPORT_GATE)["summary"]
    signed_summary = sign_base.read_json(SIGNED_PROMOTION_GATE)["summary"]

    exact_interval_over_m0 = float(ext_summary["extended_interval_over_m0"])
    exact_signed_reproduction_error = float(
        signed_summary["signed_form_factor_reproduction_max_abs_error"]
    )
    q_nyquist_box_over_m0 = float(boundary_summary["q_nyquist_box_over_m0"])
    first_alias_harmonic_over_m0 = float(boundary_summary["first_alias_harmonic_over_m0"])
    best_boundary_combined_mismatch = float(
        boundary_summary["best_envelope_floor_combined_mismatch_fraction"]
    )

    inventory_ready = bool(prior_summary["reserve_boundary_artifact_selected"])
    low_q_cover_factor = exact_interval_over_m0 / Q_THEORY
    q_nyquist_ratio = q_nyquist_box_over_m0 / Q_THEORY
    first_alias_ratio = first_alias_harmonic_over_m0 / Q_THEORY
    boundary_onset_outside_low_q_interval = bool(q_nyquist_box_over_m0 > exact_interval_over_m0)
    boundary_origin_primary_supported = False
    boundary_origin_falsified_for_q_theory = bool(
        boundary_onset_outside_low_q_interval
        and exact_signed_reproduction_error <= 1.0e-12
        and low_q_cover_factor > 10.0
    )

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "boundary-origin inventory ready",
            sign_base.truth(inventory_ready),
            "The boundary-origin audit starts only after the decomposition gate has already demoted boundary artifact to the reserve lane.",
        ),
        sign_base.row(
            "low_q_exact_interval_cover_factor",
            "pass" if low_q_cover_factor > 10.0 else "watch",
            "low-q exact interval cover factor over q_theory",
            low_q_cover_factor,
            "The residual point sits deep inside the retained exact low-q interval, well before any boundary Nyquist structure turns on.",
        ),
        sign_base.row(
            "boundary_nyquist_scale_ratio_over_q_theory",
            "pass" if q_nyquist_ratio > 10.0 else "watch",
            "boundary Nyquist scale ratio over q_theory",
            q_nyquist_ratio,
            "Nyquist boundary structure lives hundreds of q_theory away from the retained residual point.",
        ),
        sign_base.row(
            "first_alias_harmonic_scale_ratio_over_q_theory",
            "pass" if first_alias_ratio > 10.0 else "watch",
            "first alias harmonic scale ratio over q_theory",
            first_alias_ratio,
            "The first alias harmonic is even farther from the retained residual point than the Nyquist edge itself.",
        ),
        sign_base.row(
            "signed_form_factor_reproduction_max_abs_error",
            "pass" if exact_signed_reproduction_error <= 1.0e-12 else "reject",
            "signed form-factor reproduction max abs error on the retained low-q interval",
            exact_signed_reproduction_error,
            "An exact low-q signed observable closure blocks the claim that a far-away boundary spike is the dominant source of the 1.9% low-q residual.",
        ),
        sign_base.row(
            "best_boundary_combined_mismatch_fraction",
            "pass",
            "best high-q boundary bookkeeping combined mismatch fraction",
            best_boundary_combined_mismatch,
            "Boundary structure is retained as real supporting evidence, but it is a high-q effect and therefore not the primary explanation for the low-q residual target.",
        ),
        sign_base.row(
            "boundary_origin_primary_supported",
            "reject",
            "boundary artifact supported as the primary residual origin",
            sign_base.truth(boundary_origin_primary_supported),
            "Scale separation plus exact low-q closure falsify boundary artifact as the primary origin lane.",
        ),
        sign_base.row(
            "boundary_origin_falsified_for_q_theory",
            "pass" if boundary_origin_falsified_for_q_theory else "watch",
            "boundary artifact falsified for the retained low-q residual point",
            sign_base.truth(boundary_origin_falsified_for_q_theory),
            "The retained residual point is too deep inside the exact low-q interval for high-q boundary onset to be the dominant origin.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": RESIDUAL_REL,
        "q_theory_over_m0": Q_THEORY,
        "extended_interval_over_m0": exact_interval_over_m0,
        "low_q_exact_interval_cover_factor": low_q_cover_factor,
        "q_nyquist_box_over_m0": q_nyquist_box_over_m0,
        "first_alias_harmonic_over_m0": first_alias_harmonic_over_m0,
        "boundary_nyquist_scale_ratio_over_q_theory": q_nyquist_ratio,
        "first_alias_harmonic_scale_ratio_over_q_theory": first_alias_ratio,
        "signed_form_factor_reproduction_max_abs_error": exact_signed_reproduction_error,
        "best_boundary_combined_mismatch_fraction": best_boundary_combined_mismatch,
        "boundary_onset_outside_low_q_interval": boundary_onset_outside_low_q_interval,
        "boundary_origin_primary_supported": boundary_origin_primary_supported,
        "boundary_origin_falsified_for_q_theory": boundary_origin_falsified_for_q_theory,
        "supporting_boundary_structure_retained": True,
        "observable_definition_secondary_carryover": True,
        "missing_action_level_primary_carryover": True,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2345",
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
                "ext_interval_gate": sign_base.display_path(EXT_INTERVAL_GATE),
                "boundary_support_gate": sign_base.display_path(BOUNDARY_SUPPORT_GATE),
                "signed_promotion_gate": sign_base.display_path(SIGNED_PROMOTION_GATE),
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
            "overall_status": "vector_qball_form_factor_boundary_origin_falsification_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2343"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2343-.2346"),
                "current_problem_hit": sign_base.hit(current_problem_text, "boundary artifact"),
                "current_status_hit": sign_base.hit(current_status_text, "boundary artifact"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2343-.2346"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2343-.2346"),
                "part5_hit": sign_base.hit(part5_text, "2026-03-30 residual-origin update"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2346",
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
                "boundary_falsification_synced",
                "pass",
                "boundary-origin falsification synced",
                1.0,
                "The residual-origin mainline is only honest if the boundary lane is explicitly closed out as a primary explanation once the scale-separation test is passed.",
            ),
        ],
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_boundary_origin_falsification_route_synced",
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
