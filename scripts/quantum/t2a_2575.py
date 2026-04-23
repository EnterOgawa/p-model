#!/usr/bin/env python3
"""Generate 8.7.56.2575-.2578 residual-origin refresh reroute artifacts."""

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
        "8.7.56.2567-2570",
        "updated_pack_blind_vector_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2571-2574",
        "updated_pack_blind_vector_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
BASELINE_RESIDUAL = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2519-2522",
        "updated_pack_residual_origin_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2575-2578"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor updated-pack residual-origin refresh audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_residual_origin_refresh_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_blind_vector_"
    "refresh_audited_residual_origin_primary_theorem_refresh_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_residual_origin_"
    "refresh_audited_background_expansion_derivation_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_residual_origin_gate_"
    "background_expansion_derivation_refresh"
)
NEXT_ROUTE = "8.7.56.2579"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_background_"
    "expansion_derivation_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2583"


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


# 関数: residual-origin refresh reroute で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the residual-origin refresh reroute audit."""
    return {
        "focus": "use q=0, q=q_theory, q=m0, pass/no-go gates, and consistency guards to localize residual origin",
        "pass_gate": "blind F_vector(q_theory) improves the retained scalar residual while F_vector(0)=1 stays fixed",
        "no_go_gate": "exact source theorem gives no vector correction or the proxy density fails as an exact current",
        "reroute": "once the residual-origin surface is explicit, the immediate theorem-side move is exact background-expansion derivation",
    }


# 関数: `.2575-.2578` を実行する。

def main() -> None:
    """Execute the updated-pack residual-origin refresh reroute audit."""
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
        PRIOR_GATE,
        BASELINE_RESIDUAL,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)

    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    baseline_summary = sign_base.read_json(BASELINE_RESIDUAL)["summary"]

    updated_pack_residual_origin_refresh_audit_selected = bool(
        prior_gate_summary["gate_b_updated_pack_residual_origin_refresh_primary_selected"]
        and not prior_gate_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    residual_origin_q0_normalization_guard_explicit = bool(
        prior_audit_summary["blind_vector_q0_checkpoint_explicit"]
    )
    residual_origin_q_theory_improvement_discriminator_explicit = bool(
        prior_audit_summary["blind_vector_q_theory_checkpoint_explicit"]
        and prior_audit_summary["blind_vector_residual_improvement_target_explicit"]
        and baseline_summary["residual_origin_q_theory_improvement_discriminator_explicit"]
    )
    residual_origin_m0_tail_guard_explicit = bool(
        prior_audit_summary["blind_vector_m0_checkpoint_explicit"]
    )
    residual_origin_exact_source_theorem_no_go_explicit = bool(
        baseline_summary["residual_origin_exact_source_theorem_no_go_explicit"]
    )
    residual_origin_proxy_exact_current_no_go_explicit = bool(
        baseline_summary["residual_origin_proxy_exact_current_no_go_explicit"]
    )
    universality_consistency_step_explicit = bool(
        baseline_summary["universality_consistency_step_explicit"]
    )
    low_q_coulomb_guard_explicit = bool(
        baseline_summary["low_q_coulomb_guard_explicit"]
    )
    soft_photon_limit_guard_explicit = bool(
        baseline_summary["soft_photon_limit_guard_explicit"]
    )
    updated_pack_residual_origin_refresh_target_surface_explicit = bool(
        updated_pack_residual_origin_refresh_audit_selected
        and residual_origin_q0_normalization_guard_explicit
        and residual_origin_q_theory_improvement_discriminator_explicit
        and residual_origin_m0_tail_guard_explicit
        and residual_origin_exact_source_theorem_no_go_explicit
        and residual_origin_proxy_exact_current_no_go_explicit
        and universality_consistency_step_explicit
        and low_q_coulomb_guard_explicit
        and soft_photon_limit_guard_explicit
    )
    updated_pack_residual_origin_refresh_machine_readable_now = bool(
        updated_pack_residual_origin_refresh_target_surface_explicit
        and prior_gate_summary["selected_primary_pack_update_surface"] == "residual_origin_refresh_after_blind_vector"
    )
    direct_blind_vector_computation_primary_admissible_now = bool(
        prior_gate_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    residual_origin_refresh_supports_missing_action_primary_now = bool(
        updated_pack_residual_origin_refresh_machine_readable_now
        and not direct_blind_vector_computation_primary_admissible_now
    )
    updated_pack_exact_background_expansion_derivation_followup_required = bool(
        residual_origin_refresh_supports_missing_action_primary_now
        and not prior_gate_summary["exact_source_theorem_derived_now"]
    )
    updated_pack_residual_origin_refresh_closes_missing_action_blocker_now = False
    farther_hybrid_continuation_reopen_required_now = bool(
        prior_gate_summary["farther_hybrid_continuation_reopen_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_residual_origin_refresh_audit_selected",
            "pass" if updated_pack_residual_origin_refresh_audit_selected else "reject",
            "updated-pack residual-origin refresh audit selected",
            sign_base.truth(updated_pack_residual_origin_refresh_audit_selected),
            "Selected because blind-vector direct computation remains blocked under the current pack.",
        ),
        sign_base.row(
            "residual_origin_q0_normalization_guard_explicit",
            "pass" if residual_origin_q0_normalization_guard_explicit else "reject",
            "residual-origin q=0 normalization guard explicit",
            sign_base.truth(residual_origin_q0_normalization_guard_explicit),
            "The retained blind-vector refresh still preserves the normalization checkpoint at q=0.",
        ),
        sign_base.row(
            "residual_origin_q_theory_improvement_discriminator_explicit",
            "pass" if residual_origin_q_theory_improvement_discriminator_explicit else "reject",
            "residual-origin q_theory improvement discriminator explicit",
            sign_base.truth(residual_origin_q_theory_improvement_discriminator_explicit),
            "The residual-origin question remains tied to improvement over the retained 1.9% scalar residual at q_theory.",
        ),
        sign_base.row(
            "residual_origin_m0_tail_guard_explicit",
            "pass" if residual_origin_m0_tail_guard_explicit else "reject",
            "residual-origin q=m0 tail guard explicit",
            sign_base.truth(residual_origin_m0_tail_guard_explicit),
            "The q=m0 checkpoint remains part of the same discriminator surface.",
        ),
        sign_base.row(
            "residual_origin_exact_source_theorem_no_go_explicit",
            "pass" if residual_origin_exact_source_theorem_no_go_explicit else "reject",
            "exact source-theorem no-go explicit",
            sign_base.truth(residual_origin_exact_source_theorem_no_go_explicit),
            "The exact-source no-go branch is retained from the earlier residual-origin audit.",
        ),
        sign_base.row(
            "residual_origin_proxy_exact_current_no_go_explicit",
            "pass" if residual_origin_proxy_exact_current_no_go_explicit else "reject",
            "proxy-vs-exact-current no-go explicit",
            sign_base.truth(residual_origin_proxy_exact_current_no_go_explicit),
            "The proxy-current mismatch branch is retained as a theorem-side guard.",
        ),
        sign_base.row(
            "universality_consistency_step_explicit",
            "pass" if universality_consistency_step_explicit else "reject",
            "universality consistency step explicit",
            sign_base.truth(universality_consistency_step_explicit),
            "The residual-origin lane still sits under the retained universality / consistency step.",
        ),
        sign_base.row(
            "low_q_coulomb_guard_explicit",
            "pass" if low_q_coulomb_guard_explicit else "reject",
            "low-q Coulomb guard explicit",
            sign_base.truth(low_q_coulomb_guard_explicit),
            "The discriminator still keeps the low-q Coulomb guard explicit.",
        ),
        sign_base.row(
            "soft_photon_limit_guard_explicit",
            "pass" if soft_photon_limit_guard_explicit else "reject",
            "soft-photon limit guard explicit",
            sign_base.truth(soft_photon_limit_guard_explicit),
            "The discriminator still keeps the soft-photon / Thomson guard explicit.",
        ),
        sign_base.row(
            "updated_pack_residual_origin_refresh_target_surface_explicit",
            "pass" if updated_pack_residual_origin_refresh_target_surface_explicit else "reject",
            "updated-pack residual-origin target surface explicit",
            sign_base.truth(updated_pack_residual_origin_refresh_target_surface_explicit),
            "The discriminator surface remains explicit without relying on any external note.",
        ),
        sign_base.row(
            "updated_pack_residual_origin_refresh_machine_readable_now",
            "pass" if updated_pack_residual_origin_refresh_machine_readable_now else "reject",
            "updated-pack residual-origin machine-readable now",
            sign_base.truth(updated_pack_residual_origin_refresh_machine_readable_now),
            "The residual-origin lane is localized on a concrete surface inside the retained updated-pack artifacts.",
        ),
        sign_base.row(
            "residual_origin_refresh_supports_missing_action_primary_now",
            "pass" if residual_origin_refresh_supports_missing_action_primary_now else "reject",
            "residual-origin supports missing-action primary now",
            sign_base.truth(residual_origin_refresh_supports_missing_action_primary_now),
            "The lane still points back to the theorem-side blocker rather than to a fresh numeric continuation.",
        ),
        sign_base.row(
            "updated_pack_exact_background_expansion_derivation_followup_required",
            "pass" if updated_pack_exact_background_expansion_derivation_followup_required else "reject",
            "updated-pack exact background-expansion derivation followup required",
            sign_base.truth(updated_pack_exact_background_expansion_derivation_followup_required),
            "After residual-origin refresh is re-synced, the immediate theorem-side move is exact background-expansion derivation rather than another generic theorem-refresh placeholder.",
        ),
        sign_base.row(
            "updated_pack_residual_origin_refresh_closes_missing_action_blocker_now",
            "pass" if updated_pack_residual_origin_refresh_closes_missing_action_blocker_now else "reject",
            "updated-pack residual-origin closes missing-action blocker now",
            sign_base.truth(updated_pack_residual_origin_refresh_closes_missing_action_blocker_now),
            "Residual-origin refresh does not close the blocker by itself.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range remains reserve-only because the blocker is still theorem-side.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_residual_origin_refresh_audit_selected": updated_pack_residual_origin_refresh_audit_selected,
        "residual_origin_q0_normalization_guard_explicit": residual_origin_q0_normalization_guard_explicit,
        "residual_origin_q_theory_improvement_discriminator_explicit": residual_origin_q_theory_improvement_discriminator_explicit,
        "residual_origin_m0_tail_guard_explicit": residual_origin_m0_tail_guard_explicit,
        "residual_origin_exact_source_theorem_no_go_explicit": residual_origin_exact_source_theorem_no_go_explicit,
        "residual_origin_proxy_exact_current_no_go_explicit": residual_origin_proxy_exact_current_no_go_explicit,
        "universality_consistency_step_explicit": universality_consistency_step_explicit,
        "low_q_coulomb_guard_explicit": low_q_coulomb_guard_explicit,
        "soft_photon_limit_guard_explicit": soft_photon_limit_guard_explicit,
        "updated_pack_residual_origin_refresh_target_surface_explicit": updated_pack_residual_origin_refresh_target_surface_explicit,
        "updated_pack_residual_origin_refresh_machine_readable_now": updated_pack_residual_origin_refresh_machine_readable_now,
        "direct_blind_vector_computation_primary_admissible_now": direct_blind_vector_computation_primary_admissible_now,
        "residual_origin_refresh_supports_missing_action_primary_now": residual_origin_refresh_supports_missing_action_primary_now,
        "updated_pack_exact_background_expansion_derivation_followup_required": updated_pack_exact_background_expansion_derivation_followup_required,
        "updated_pack_residual_origin_refresh_closes_missing_action_blocker_now": updated_pack_residual_origin_refresh_closes_missing_action_blocker_now,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "residual_origin_refresh_after_blind_vector",
        "selected_secondary_pack_update_surface": "updated_pack_exact_background_expansion_derivation_after_residual_origin",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2577",
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
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "baseline_residual": sign_base.display_path(BASELINE_RESIDUAL),
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
            "overall_status": "vector_qball_form_factor_updated_pack_residual_origin_refresh_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2575"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2571-.2574"),
                "current_problem_hit": sign_base.hit(current_problem_text, "updated-pack residual-origin refresh audit"),
                "current_status_hit": sign_base.hit(current_status_text, "updated-pack residual-origin refresh audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2571-.2574"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2571-.2574"),
                "part5_hit": sign_base.hit(part5_text, ".2567-.2574"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2578",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_residual_origin_refresh_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulae": build_formulae(),
            "disposition": {
                "residual_origin_refresh_surface_explicit": updated_pack_residual_origin_refresh_target_surface_explicit,
                "residual_origin_refresh_machine_readable_now": updated_pack_residual_origin_refresh_machine_readable_now,
                "background_expansion_derivation_followup_required": updated_pack_exact_background_expansion_derivation_followup_required,
                "direct_blind_vector_still_blocked": not direct_blind_vector_computation_primary_admissible_now,
            },
        },
    }
    route_paths = write_artifact("route_sync", route_payload)

    print("[ok] updated-pack residual-origin refresh audit artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
