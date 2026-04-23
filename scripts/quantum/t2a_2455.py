#!/usr/bin/env python3
"""Generate 8.7.56.2455-.2458 updated-pack exact effective-source-theorem audit artifacts."""

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
        "8.7.56.2451-2454",
        "updated_pack_exact_ell0_series_operator_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
OLD_SOURCE_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.1487-1490",
        "effective_source_theorem",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

NEXT_STEPS = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")

STEP_TAG = "8.7.56.2455-2458"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor updated-pack exact effective source theorem audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_exact_effective_source_theorem_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_exact_ell0_"
    "series_operator_audited_effective_source_theorem_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_exact_effective_"
    "source_theorem_audited_source_rule_refresh_gate"
)
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_effective_source_theorem_gate_source_rule_refresh"
NEXT_ROUTE = "8.7.56.2459"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_source_rule_audit"
FOLLOWUP_ROUTE = "8.7.56.2463"


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


# 関数: updated-pack exact effective source theorem audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the updated-pack exact effective source theorem audit."""
    return {
        "effective_source_surface": "L \\supset a_mu J_eff^mu[P^Qball]",
        "proxy_rule": "J_eff^0 low-order == |f_0|^2 - |f_L|^2 => proxy strong support",
        "no_go_rule": "J_eff^0 low-order != |f_0|^2 - |f_L|^2 => proxy route no-go",
        "step_order": "exact ell=0 series/operator -> exact effective source theorem -> blind vector computation",
    }


# 関数: `.2455-.2458` を実行する。

def main() -> None:
    """Execute the updated-pack exact effective source theorem audit."""
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
        OLD_SOURCE_AUDIT,
        NEXT_STEPS,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)
    next_steps_text = sign_base.read_text(NEXT_STEPS)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    old_source_summary = sign_base.read_json(OLD_SOURCE_AUDIT)["summary"]

    updated_pack_exact_effective_source_theorem_audit_selected = bool(
        prior_summary["gate_b_updated_pack_exact_effective_source_theorem_promoted_next"]
        and not prior_summary["blind_numeric_recompute_as_primary_admissible_now"]
    )
    updated_pack_step_c_surface_explicit = bool(
        sign_base.hit(next_steps_text, "### Step C. photon に結合する exact source / current を導く") is not None
        and sign_base.hit(next_steps_text, "\\mathcal L \\supset a_\\mu\\,J^{\\mu}_{\\rm eff}[P^{\\rm Qball}]") is not None
    )
    updated_pack_proxy_support_rule_explicit = bool(
        sign_base.hit(next_steps_text, "もし `J_eff^0` が低次で `|f_0|^2 - |f_L|^2` に落ちるなら") is not None
    )
    updated_pack_proxy_no_go_rule_explicit = bool(
        sign_base.hit(next_steps_text, "そうならないなら、現在の proxy route は no-go") is not None
    )
    retained_explicit_matter_current_surface_available = bool(
        sign_base.hit(current_problem_text, "J^\\mu_{\\mathrm{matter}}=(\\rho c,\\rho \\mathbf{v})") is not None
        and sign_base.hit(current_problem_text, "\\mathcal{L}_{\\mathrm{int}}=g_P\\,P_\\mu J^\\mu_{\\mathrm{matter}}") is not None
    )
    retained_charge_noether_current_surface_available = bool(
        not old_source_summary["exact_charge_current_noether_closure_required"]
    )
    current_canon_explicit_qball_background_expansion_available = bool(
        old_source_summary["explicit_qball_background_expansion_available"]
    )
    current_canon_explicit_effective_source_formula_available = bool(
        old_source_summary["explicit_effective_source_formula_available"]
    )
    updated_pack_exact_effective_source_theorem_supported_now = bool(
        updated_pack_exact_effective_source_theorem_audit_selected
        and updated_pack_step_c_surface_explicit
        and updated_pack_proxy_support_rule_explicit
        and updated_pack_proxy_no_go_rule_explicit
        and retained_explicit_matter_current_surface_available
    )
    updated_pack_exact_effective_source_theorem_derived_now = bool(
        updated_pack_exact_effective_source_theorem_supported_now
        and current_canon_explicit_qball_background_expansion_available
        and current_canon_explicit_effective_source_formula_available
        and retained_charge_noether_current_surface_available
    )
    updated_pack_source_rule_refresh_required = bool(
        updated_pack_exact_effective_source_theorem_supported_now
        and not updated_pack_exact_effective_source_theorem_derived_now
    )
    blind_vector_computation_primary_admissible_now = bool(
        updated_pack_exact_effective_source_theorem_derived_now
        and old_source_summary["observable_dictionary_gate_admissible_now"]
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_exact_effective_source_theorem_audit_selected",
            "pass" if updated_pack_exact_effective_source_theorem_audit_selected else "reject",
            "updated-pack exact effective source theorem audit selected",
            sign_base.truth(updated_pack_exact_effective_source_theorem_audit_selected),
            "The updated-pack operator gate already promoted the exact effective source theorem as the next honest followup.",
        ),
        sign_base.row(
            "updated_pack_step_c_surface_explicit",
            "pass" if updated_pack_step_c_surface_explicit else "reject",
            "updated-pack Step C source/current surface explicit",
            sign_base.truth(updated_pack_step_c_surface_explicit),
            "The refreshed next-steps note explicitly fixes the exact source/current question as `L ⊃ a_mu J_eff^mu[P^Qball]`.",
        ),
        sign_base.row(
            "updated_pack_proxy_support_rule_explicit",
            "pass" if updated_pack_proxy_support_rule_explicit else "reject",
            "updated-pack proxy support rule explicit",
            sign_base.truth(updated_pack_proxy_support_rule_explicit),
            "The updated pack now states the exact strong-support discriminator for the current vector proxy.",
        ),
        sign_base.row(
            "updated_pack_proxy_no_go_rule_explicit",
            "pass" if updated_pack_proxy_no_go_rule_explicit else "reject",
            "updated-pack proxy no-go rule explicit",
            sign_base.truth(updated_pack_proxy_no_go_rule_explicit),
            "The updated pack also states the no-go branch explicitly rather than hiding it behind proxy rhetoric.",
        ),
        sign_base.row(
            "retained_explicit_matter_current_surface_available",
            "pass" if retained_explicit_matter_current_surface_available else "reject",
            "retained explicit matter-current interaction surface available",
            sign_base.truth(retained_explicit_matter_current_surface_available),
            "The old source-theorem audit already established that the matter-current interaction surface is present in the current canon.",
        ),
        sign_base.row(
            "current_canon_explicit_qball_background_expansion_available",
            "pass" if current_canon_explicit_qball_background_expansion_available else "reject",
            "current canon explicit Q-ball background expansion available",
            sign_base.truth(current_canon_explicit_qball_background_expansion_available),
            "The retained blocker remains that the canon still does not expose the exact Q-ball background expansion as a public surface.",
        ),
        sign_base.row(
            "current_canon_explicit_effective_source_formula_available",
            "pass" if current_canon_explicit_effective_source_formula_available else "reject",
            "current canon explicit effective source formula available",
            sign_base.truth(current_canon_explicit_effective_source_formula_available),
            "The updated pack still lacks an explicit first-principles `J_eff^mu[P^Qball]` formula in the retained public canon.",
        ),
        sign_base.row(
            "updated_pack_exact_effective_source_theorem_supported_now",
            "pass" if updated_pack_exact_effective_source_theorem_supported_now else "reject",
            "updated-pack exact effective source theorem supported now",
            sign_base.truth(updated_pack_exact_effective_source_theorem_supported_now),
            "The updated pack is explicit enough to ask the theorem question honestly without reopening blind numeric computation first.",
        ),
        sign_base.row(
            "updated_pack_exact_effective_source_theorem_derived_now",
            "pass" if updated_pack_exact_effective_source_theorem_derived_now else "reject",
            "updated-pack exact effective source theorem derived now",
            sign_base.truth(updated_pack_exact_effective_source_theorem_derived_now),
            "The theorem only passes if the explicit Q-ball background expansion, exact effective source formula, and charge-current closure all surface.",
        ),
        sign_base.row(
            "updated_pack_source_rule_refresh_required",
            "pass" if updated_pack_source_rule_refresh_required else "reject",
            "updated-pack source-rule refresh required",
            sign_base.truth(updated_pack_source_rule_refresh_required),
            "Because the exact theorem is still absent, the next honest move is to refresh the proxy-support / no-go source rule rather than jump to blind vector computation.",
        ),
        sign_base.row(
            "blind_vector_computation_primary_admissible_now",
            "pass" if blind_vector_computation_primary_admissible_now else "reject",
            "blind vector computation primary admissible now",
            sign_base.truth(blind_vector_computation_primary_admissible_now),
            "Blind vector evaluation remains downstream until the exact source theorem or its no-go source rule is synchronized honestly.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains reserve-only because the blocker is still at the source/current theorem surface.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "updated_pack_exact_effective_source_theorem_audit_selected": updated_pack_exact_effective_source_theorem_audit_selected,
        "updated_pack_step_c_surface_explicit": updated_pack_step_c_surface_explicit,
        "updated_pack_proxy_support_rule_explicit": updated_pack_proxy_support_rule_explicit,
        "updated_pack_proxy_no_go_rule_explicit": updated_pack_proxy_no_go_rule_explicit,
        "retained_explicit_matter_current_surface_available": retained_explicit_matter_current_surface_available,
        "retained_charge_noether_current_surface_available": retained_charge_noether_current_surface_available,
        "current_canon_explicit_qball_background_expansion_available": current_canon_explicit_qball_background_expansion_available,
        "current_canon_explicit_effective_source_formula_available": current_canon_explicit_effective_source_formula_available,
        "updated_pack_exact_effective_source_theorem_supported_now": updated_pack_exact_effective_source_theorem_supported_now,
        "updated_pack_exact_effective_source_theorem_derived_now": updated_pack_exact_effective_source_theorem_derived_now,
        "updated_pack_source_rule_refresh_required": updated_pack_source_rule_refresh_required,
        "blind_vector_computation_primary_admissible_now": blind_vector_computation_primary_admissible_now,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_proxy_support_or_no_go_source_rule",
        "selected_secondary_pack_update_surface": "blind_vector_computation_after_source_rule_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2457",
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
                "old_source_audit": sign_base.display_path(OLD_SOURCE_AUDIT),
                "next_steps": sign_base.display_path(NEXT_STEPS),
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
            "overall_status": "vector_qball_form_factor_updated_pack_exact_effective_source_theorem_audit_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2455"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2455-.2458"),
                "current_problem_hit": sign_base.hit(current_problem_text, "effective source theorem"),
                "current_status_hit": sign_base.hit(current_status_text, "source theorem current-canon fail"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2447-.2450"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2447-.2450"),
                "part5_hit": sign_base.hit(part5_text, "updated-pack exact effective source theorem"),
                "next_steps_step_c_hit": sign_base.hit(next_steps_text, "### Step C."),
                "next_steps_jeff_hit": sign_base.hit(next_steps_text, "\\mathcal L \\supset a_\\mu\\,J^{\\mu}_{\\rm eff}[P^{\\rm Qball}]"),
                "next_steps_proxy_support_hit": sign_base.hit(next_steps_text, "もし `J_eff^0` が低次で `|f_0|^2 - |f_L|^2` に落ちるなら"),
                "next_steps_proxy_no_go_hit": sign_base.hit(next_steps_text, "そうならないなら、現在の proxy route は no-go"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2458",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_exact_effective_source_theorem_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": declaration_payload["evidence"],
    }
    write_artifact("route_sync", route_payload)


if __name__ == "__main__":
    main()
