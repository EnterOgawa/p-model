#!/usr/bin/env python3
"""Generate 8.7.56.2363-.2366 profile-fixed eigenvalue-shift gate artifacts."""

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
        "8.7.56.2359-2362",
        "exact_coupled_eigshift_theorem",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2363-2366"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor profile-fixed eigenvalue-shift decision gate"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "profile_fixed_eigshift_gate",
    prefix="q",
)

PRIOR_CLASS = "vector_qball_form_factor_residual_origin_missing_action_exact_coupled_eigenvalue_shift_theorem_audit_profile_fixed_candidate_retained_gate"
BRANCH_CLASS = "vector_qball_form_factor_residual_origin_missing_action_profile_fixed_eigenvalue_shift_candidate_retained_exact_operator_completion_next"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_exact_action_level_operator_completion_audit"
NEXT_ROUTE = "8.7.56.2367"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_exact_operator_completion_decision_gate_hybrid_reserve_refresh"
FOLLOWUP_ROUTE = "8.7.56.2371"


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


# 関数: decision gate で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the decision gate."""
    return {
        "gate_a": "Gate A = exact coupled eigenvalue-shift theorem selected",
        "gate_b": "Gate B = profile-fixed eigenvalue-shift candidate retained pending exact operator completion",
        "gate_c": "Gate C = hybrid supporting-evidence reopen required now",
    }


# 関数: `.2363-.2366` を実行する。

def main() -> None:
    """Execute the profile-fixed eigenvalue-shift decision gate."""
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

    gate_a = bool(prior_summary["exact_coupled_theorem_derivable_under_current_pack"])
    gate_b = bool(prior_summary["profile_fixed_candidate_retained_pending_operator_completion"])
    gate_c = bool(prior_summary["hybrid_supporting_evidence_reopen_required"])
    exact_operator_completion_required = bool(gate_b and not gate_a)
    observable_secondary_carryover_retained = True
    boundary_primary_falsified_retained = True

    rows = [
        sign_base.row(
            "gate_a_exact_coupled_theorem_selected",
            "pass" if gate_a else "reject",
            "Gate A exact coupled eigenvalue-shift theorem selected",
            sign_base.truth(gate_a),
            "Gate A would require the current pack to derive the needed delta(beta_1) from a closed exact coupled ell=0 operator.",
        ),
        sign_base.row(
            "gate_b_profile_fixed_candidate_retained",
            "pass" if gate_b else "reject",
            "Gate B profile-fixed eigenvalue-shift candidate retained",
            sign_base.truth(gate_b),
            "The numerically modest profile-fixed candidate survives because the blocker is operator incompleteness rather than an oversized required shift.",
        ),
        sign_base.row(
            "gate_c_hybrid_supporting_evidence_reopen_required",
            "pass" if gate_c else "reject",
            "Gate C hybrid supporting-evidence reopen required now",
            sign_base.truth(gate_c),
            "Hybrid continuation would reopen only if extra q-range became necessary for residual-origin discrimination.",
        ),
        sign_base.row(
            "exact_operator_completion_required",
            "pass" if exact_operator_completion_required else "reject",
            "exact action-level operator completion required next",
            sign_base.truth(exact_operator_completion_required),
            "Because Gate A fails while Gate B survives, the next mainline move is to sharpen the missing-action lane to exact operator completion.",
        ),
        sign_base.row(
            "observable_secondary_carryover_retained",
            "pass",
            "observable-definition mismatch retained as secondary carry-over",
            sign_base.truth(observable_secondary_carryover_retained),
            "The observable lane stays secondary because the low-q observable family remains internally exact.",
        ),
        sign_base.row(
            "boundary_primary_falsified_retained",
            "pass",
            "boundary artifact primary falsification retained",
            sign_base.truth(boundary_primary_falsified_retained),
            "The boundary lane remains reserve-only after the low-q scale-separation falsification.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "delta_beta2_exact_profile_fixed": float(prior_summary["delta_beta2_exact_profile_fixed"]),
        "required_delta_beta2_fraction_of_beta_gap": float(
            prior_summary["required_delta_beta2_fraction_of_beta_gap"]
        ),
        "required_delta_beta2_vs_ceiling_sq": float(
            prior_summary["required_delta_beta2_vs_ceiling_sq"]
        ),
        "gate_a_exact_coupled_theorem_selected": gate_a,
        "gate_b_profile_fixed_candidate_retained": gate_b,
        "gate_c_hybrid_supporting_evidence_reopen_required": gate_c,
        "exact_operator_completion_required": exact_operator_completion_required,
        "observable_secondary_carryover_retained": observable_secondary_carryover_retained,
        "boundary_primary_falsified_retained": boundary_primary_falsified_retained,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2365",
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
            "overall_status": "vector_qball_form_factor_profile_fixed_eigenvalue_shift_gate_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2363"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2363-.2366"),
                "current_problem_hit": sign_base.hit(current_problem_text, "exact coupled eigenvalue-shift theorem audit"),
                "current_status_hit": sign_base.hit(current_status_text, "exact coupled eigenvalue-shift theorem audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2363-.2366"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2363-.2366"),
                "part5_hit": sign_base.hit(part5_text, "exact coupled eigenvalue-shift theorem audit"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2366",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_profile_fixed_eigenvalue_shift_gate_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "selected_route": {
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            }
        },
    }
    write_artifact("route_sync", route_payload)

    print(f"[done] {STEP_TAG} profile-fixed eigenvalue-shift decision gate completed")
    print(f"[info] declaration_gate_json={declaration_paths['json']}")
    print(f"[info] declaration_gate_csv={declaration_paths['csv']}")


if __name__ == "__main__":
    main()
