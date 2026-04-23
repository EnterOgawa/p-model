#!/usr/bin/env python3
"""Generate 8.7.56.2779-.2782 external-probe gate artifacts."""

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

PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2775-2778",
        "updated_pack_exact_external_probe_current_vertex_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2779-2782"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack external-"
    "probe gate / mixed-response refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_external_probe_gate",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_probe_current_vertex_audited_mixed_probe_response_primary_"
    "vacuum_subtraction_tertiary_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_probe_current_vertex_audited_mixed_probe_response_primary_"
    "vacuum_subtraction_secondary_hybrid_reserve_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_mixed_"
    "probe_response_kernel_audit"
)
NEXT_ROUTE = "8.7.56.2783"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_mixed_probe_"
    "response_gate_pack_refresh"
)
FOLLOWUP_ROUTE = "8.7.56.2787"


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
    """Return formulas used in the external-probe gate."""
    return {
        "gate_a": "Gate A = exact external-probe current vertex available now",
        "gate_b": "Gate B = exact mixed probe-response kernel promoted next",
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.2779-.2782` を実行する。

def main() -> None:
    """Execute the updated-pack external-probe gate / mixed-response refresh."""
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
        PRIOR_AUDIT,
    ):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a_updated_pack_exact_external_probe_current_vertex_available_now = bool(
        prior_summary["exact_external_probe_current_vertex_formula_available_now"]
    )
    gate_b_updated_pack_exact_mixed_probe_response_kernel_promoted_next = bool(
        prior_summary["updated_pack_mixed_probe_response_primary_followup_required"]
        and (not gate_a_updated_pack_exact_external_probe_current_vertex_available_now)
    )
    gate_c_farther_hybrid_continuation_reopen_required_now = False
    retry_gate_computation_mode_selected = bool(
        prior_summary["retry_gate_computation_mode_selected"]
    )
    failure_matrix_non_surrogate_guard_preserved = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    blind_vector_observable_gate_still_blocked = bool(
        prior_summary["blind_vector_observable_gate_still_blocked"]
    )
    old_density_proxy_eigenvalue_retry_admissible_now = False
    pack_update_required_now = bool(
        gate_b_updated_pack_exact_mixed_probe_response_kernel_promoted_next
    )

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_external_probe_current_vertex_available_now",
            "pass" if gate_a_updated_pack_exact_external_probe_current_vertex_available_now else "reject",
            "Gate A exact external-probe current vertex available now",
            sign_base.truth(gate_a_updated_pack_exact_external_probe_current_vertex_available_now),
            "The external-probe one-point current target is explicit, but the exact frozen-action formula is still absent.",
        ),
        sign_base.row(
            "gate_b_updated_pack_exact_mixed_probe_response_kernel_promoted_next",
            "pass" if gate_b_updated_pack_exact_mixed_probe_response_kernel_promoted_next else "reject",
            "Gate B exact mixed probe-response kernel promoted next",
            sign_base.truth(gate_b_updated_pack_exact_mixed_probe_response_kernel_promoted_next),
            "Because the one-point current is not yet literal, the honest fallback shifts to the mixed probe-response kernel.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "pass" if gate_c_farther_hybrid_continuation_reopen_required_now else "reject",
            "Gate C farther hybrid continuation reopen required now",
            sign_base.truth(gate_c_farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains reserve-only because the blocker is still the missing probe current/kernel object.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_gate_computation_mode_selected else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_gate_computation_mode_selected),
            "The route remains on the derivation side after the pure-derivation reset.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if failure_matrix_non_surrogate_guard_preserved else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(failure_matrix_non_surrogate_guard_preserved),
            "The gate keeps the exhausted density/proxy/eigenvalue family closed while the probe kernel lane is promoted.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Blind-vector direct computation still waits on the rank-matched external probe object and its theorem stack.",
        ),
        sign_base.row(
            "old_density_proxy_eigenvalue_retry_admissible_now",
            "pass" if old_density_proxy_eigenvalue_retry_admissible_now else "reject",
            "old density/proxy/eigenvalue retry admissible now",
            sign_base.truth(old_density_proxy_eigenvalue_retry_admissible_now),
            "The gate does not reopen any same-level surrogate retry family.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "The honest next move stays inside the pack-update theorem chain, now with mixed probe-response as the primary fallback object.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "gate_a_updated_pack_exact_external_probe_current_vertex_available_now": gate_a_updated_pack_exact_external_probe_current_vertex_available_now,
        "gate_b_updated_pack_exact_mixed_probe_response_kernel_promoted_next": gate_b_updated_pack_exact_mixed_probe_response_kernel_promoted_next,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c_farther_hybrid_continuation_reopen_required_now,
        "retry_gate_computation_mode_selected": retry_gate_computation_mode_selected,
        "failure_matrix_non_surrogate_guard_preserved": failure_matrix_non_surrogate_guard_preserved,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "old_density_proxy_eigenvalue_retry_admissible_now": old_density_proxy_eigenvalue_retry_admissible_now,
        "pack_update_required_now": pack_update_required_now,
        "selected_primary_completion_lane": "updated_pack_exact_mixed_probe_response_kernel_audit",
        "selected_secondary_completion_lane": "updated_pack_vacuum_subtraction_hold",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2781",
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
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
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
            "overall_status": "vector_qball_form_factor_updated_pack_external_probe_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )

    route_payload = sign_base.payload(
        "8.7.56.2782",
        STEP_NAME + " route sync",
        {
            "source_files": declaration_payload["inputs"]["source_files"],
            "declaration": {},
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_external_probe_gate_route_synced",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "formulas": build_formulae(),
            "notes": {
                "primary_transition": (
                    "The official next bottleneck becomes the exact mixed probe-"
                    "response kernel instead of repeating one-point current wording."
                ),
                "secondary_hold": (
                    "Vacuum subtraction remains downstream and cannot be treated as "
                    "canonical before the kernel object is fixed."
                ),
            },
        },
    )

    declaration_paths = write_artifact("declaration_gate", declaration_payload)
    write_artifact("route_sync", route_payload)

    print(f"[done] {STEP_TAG} updated-pack external-probe gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
