#!/usr/bin/env python3
"""Generate 8.7.56.2511-.2514 updated-pack blind-vector refresh audit artifacts."""

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
        "8.7.56.2507-2510",
        "updated_pack_low_order_jeff0_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

NEXT_STEPS = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")

STEP_TAG = "8.7.56.2511-2514"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor updated-pack blind-vector refresh audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_blind_vector_refresh_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_low_order_"
    "jeff0_audited_blind_vector_primary_residual_origin_refresh_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_blind_"
    "vector_audited_residual_origin_gate"
)
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_gate_residual_origin_refresh"
NEXT_ROUTE = "8.7.56.2515"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_residual_origin_refresh_audit"
FOLLOWUP_ROUTE = "8.7.56.2519"


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


# 関数: blind-vector refresh audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the updated-pack blind-vector refresh audit."""
    return {
        "step_d": "After A/B/C, evaluate F_vector(q) at q = 0, q = q_theory, and q = m_0",
        "alpha_vector": "alpha_vector(q) = F_vector(q)^2 / (4 pi)",
        "decision_focus": "preserve F_vector(0)=1, improve the 1.9% scalar residual at q_theory, and approach F_vector(m_0) ~= 0.30282",
        "refresh_order": "low-order J_eff^0 synthesis -> blind-vector refresh -> residual-origin refresh",
    }


# 関数: `.2511-.2514` を実行する。

def main() -> None:
    """Execute the updated-pack blind-vector refresh audit."""
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

    updated_pack_blind_vector_refresh_audit_selected = bool(
        prior_summary["gate_b_updated_pack_blind_vector_refresh_primary_selected"]
        and not prior_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    blind_vector_step_d_surface_explicit = bool(
        sign_base.hit(next_steps_text, "### Step D. blind vector computation を行う") is not None
    )
    blind_vector_precondition_explicit = bool(
        sign_base.hit(next_steps_text, "上の A/B/C が通ったら") is not None
    )
    blind_vector_q0_checkpoint_explicit = bool(
        sign_base.hit(next_steps_text, "- `q = 0`") is not None
    )
    blind_vector_q_theory_checkpoint_explicit = bool(
        sign_base.hit(next_steps_text, "- `q = q_theory`") is not None
    )
    blind_vector_m0_checkpoint_explicit = bool(
        sign_base.hit(next_steps_text, "- `q = m_0`") is not None
    )
    blind_vector_alpha_target_explicit = bool(
        sign_base.hit(next_steps_text, r"\alpha_{\rm vector}(q)=") is not None
    )
    blind_vector_residual_improvement_target_explicit = bool(
        sign_base.hit(next_steps_text, "scalar 残差 1.9% を改善するか") is not None
    )
    updated_pack_blind_vector_refresh_target_surface_explicit = bool(
        updated_pack_blind_vector_refresh_audit_selected
        and blind_vector_step_d_surface_explicit
        and blind_vector_precondition_explicit
        and blind_vector_q0_checkpoint_explicit
        and blind_vector_q_theory_checkpoint_explicit
        and blind_vector_m0_checkpoint_explicit
        and blind_vector_alpha_target_explicit
        and blind_vector_residual_improvement_target_explicit
        and prior_summary["selected_primary_pack_update_surface"] == "updated_pack_blind_vector_refresh"
    )
    updated_pack_blind_vector_refresh_machine_readable_now = bool(
        updated_pack_blind_vector_refresh_target_surface_explicit
        and prior_summary["selected_secondary_pack_update_surface"] == "residual_origin_refresh_after_blind_vector"
    )
    vector_form_factor_exact_computation_ready_under_current_pack = bool(
        prior_summary["gate_c_blind_vector_computation_primary_admissible_now"]
    )
    blind_vector_direct_evaluation_admissible_now = bool(
        vector_form_factor_exact_computation_ready_under_current_pack
    )
    blind_vector_observable_gate_still_blocked = bool(
        not blind_vector_direct_evaluation_admissible_now
    )
    residual_origin_refresh_followup_required = bool(
        updated_pack_blind_vector_refresh_machine_readable_now
        and blind_vector_observable_gate_still_blocked
    )
    updated_pack_blind_vector_refresh_closes_missing_action_blocker_now = False
    farther_hybrid_continuation_reopen_required_now = bool(
        prior_summary["farther_hybrid_continuation_reopen_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_blind_vector_refresh_audit_selected",
            "pass" if updated_pack_blind_vector_refresh_audit_selected else "reject",
            "updated-pack blind-vector refresh audit selected",
            sign_base.truth(updated_pack_blind_vector_refresh_audit_selected),
            "The low-order J_eff^0 gate already promoted blind-vector refresh as the next honest downstream lane.",
        ),
        sign_base.row(
            "blind_vector_step_d_surface_explicit",
            "pass" if blind_vector_step_d_surface_explicit else "reject",
            "blind-vector Step D surface explicit",
            sign_base.truth(blind_vector_step_d_surface_explicit),
            "The next-steps pack now states blind-vector computation as its own named Step D rather than an implicit future remark.",
        ),
        sign_base.row(
            "blind_vector_precondition_explicit",
            "pass" if blind_vector_precondition_explicit else "reject",
            "blind-vector precondition explicit",
            sign_base.truth(blind_vector_precondition_explicit),
            "The same note keeps blind evaluation conditional on the A/B/C theorem stack rather than pretending it is already open.",
        ),
        sign_base.row(
            "blind_vector_q0_checkpoint_explicit",
            "pass" if blind_vector_q0_checkpoint_explicit else "reject",
            "blind-vector q = 0 checkpoint explicit",
            sign_base.truth(blind_vector_q0_checkpoint_explicit),
            "The blind-vector target surface explicitly preserves the normalization checkpoint at q = 0.",
        ),
        sign_base.row(
            "blind_vector_q_theory_checkpoint_explicit",
            "pass" if blind_vector_q_theory_checkpoint_explicit else "reject",
            "blind-vector q = q_theory checkpoint explicit",
            sign_base.truth(blind_vector_q_theory_checkpoint_explicit),
            "The residual-origin discriminator remains tied to the retained q_theory evaluation point.",
        ),
        sign_base.row(
            "blind_vector_m0_checkpoint_explicit",
            "pass" if blind_vector_m0_checkpoint_explicit else "reject",
            "blind-vector q = m0 checkpoint explicit",
            sign_base.truth(blind_vector_m0_checkpoint_explicit),
            "The blind-vector target still requires a literal q = m0 checkpoint rather than only a q_theory comparison.",
        ),
        sign_base.row(
            "blind_vector_alpha_target_explicit",
            "pass" if blind_vector_alpha_target_explicit else "reject",
            "blind-vector alpha target explicit",
            sign_base.truth(blind_vector_alpha_target_explicit),
            "The note keeps alpha_vector(q) as the comparison quantity rather than a wording-only proxy.",
        ),
        sign_base.row(
            "blind_vector_residual_improvement_target_explicit",
            "pass" if blind_vector_residual_improvement_target_explicit else "reject",
            "blind-vector residual-improvement target explicit",
            sign_base.truth(blind_vector_residual_improvement_target_explicit),
            "The target remains improvement over the retained 1.9% scalar residual, not a target-fit rewrite.",
        ),
        sign_base.row(
            "updated_pack_blind_vector_refresh_target_surface_explicit",
            "pass" if updated_pack_blind_vector_refresh_target_surface_explicit else "reject",
            "updated-pack blind-vector refresh target surface explicit",
            sign_base.truth(updated_pack_blind_vector_refresh_target_surface_explicit),
            "Step D plus its q checkpoints and alpha comparison are enough to define the blind-vector refresh target surface explicitly.",
        ),
        sign_base.row(
            "updated_pack_blind_vector_refresh_machine_readable_now",
            "pass" if updated_pack_blind_vector_refresh_machine_readable_now else "reject",
            "updated-pack blind-vector refresh machine-readable now",
            sign_base.truth(updated_pack_blind_vector_refresh_machine_readable_now),
            "The blind-vector lane is now localized on an explicit downstream surface rather than a generic future-computation phrase.",
        ),
        sign_base.row(
            "vector_form_factor_exact_computation_ready_under_current_pack",
            "pass" if vector_form_factor_exact_computation_ready_under_current_pack else "reject",
            "vector form-factor exact computation ready under current pack",
            sign_base.truth(vector_form_factor_exact_computation_ready_under_current_pack),
            "The theorem stack still blocks an honest direct blind-vector computation under the current pack.",
        ),
        sign_base.row(
            "blind_vector_direct_evaluation_admissible_now",
            "pass" if blind_vector_direct_evaluation_admissible_now else "reject",
            "blind-vector direct evaluation admissible now",
            sign_base.truth(blind_vector_direct_evaluation_admissible_now),
            "Blind evaluation itself remains downstream because Step D is still conditional on the unresolved theorem prerequisites.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "The current lane can describe the blind-vector target surface honestly, but it cannot yet run the direct theorem-level computation.",
        ),
        sign_base.row(
            "residual_origin_refresh_followup_required",
            "pass" if residual_origin_refresh_followup_required else "reject",
            "residual-origin refresh followup required",
            sign_base.truth(residual_origin_refresh_followup_required),
            "Once the blind-vector lane is explicit and machine-readable, the honest downstream lane is residual-origin refresh.",
        ),
        sign_base.row(
            "updated_pack_blind_vector_refresh_closes_missing_action_blocker_now",
            "pass" if updated_pack_blind_vector_refresh_closes_missing_action_blocker_now else "reject",
            "updated-pack blind-vector refresh closes missing-action blocker now",
            sign_base.truth(updated_pack_blind_vector_refresh_closes_missing_action_blocker_now),
            "This audit clarifies the blind-vector lane but does not derive the missing theorem objects needed to close the blocker.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains unnecessary because the blocker is still theorem-side and now sharpened to the blind-vector lane.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "updated_pack_blind_vector_refresh_audit_selected": updated_pack_blind_vector_refresh_audit_selected,
        "blind_vector_step_d_surface_explicit": blind_vector_step_d_surface_explicit,
        "blind_vector_precondition_explicit": blind_vector_precondition_explicit,
        "blind_vector_q0_checkpoint_explicit": blind_vector_q0_checkpoint_explicit,
        "blind_vector_q_theory_checkpoint_explicit": blind_vector_q_theory_checkpoint_explicit,
        "blind_vector_m0_checkpoint_explicit": blind_vector_m0_checkpoint_explicit,
        "blind_vector_alpha_target_explicit": blind_vector_alpha_target_explicit,
        "blind_vector_residual_improvement_target_explicit": blind_vector_residual_improvement_target_explicit,
        "updated_pack_blind_vector_refresh_target_surface_explicit": updated_pack_blind_vector_refresh_target_surface_explicit,
        "updated_pack_blind_vector_refresh_machine_readable_now": updated_pack_blind_vector_refresh_machine_readable_now,
        "vector_form_factor_exact_computation_ready_under_current_pack": vector_form_factor_exact_computation_ready_under_current_pack,
        "blind_vector_direct_evaluation_admissible_now": blind_vector_direct_evaluation_admissible_now,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "residual_origin_refresh_followup_required": residual_origin_refresh_followup_required,
        "updated_pack_blind_vector_refresh_closes_missing_action_blocker_now": updated_pack_blind_vector_refresh_closes_missing_action_blocker_now,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_blind_vector_refresh",
        "selected_secondary_pack_update_surface": "residual_origin_refresh_after_blind_vector",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2513",
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
            "overall_status": "vector_qball_form_factor_updated_pack_blind_vector_refresh_audit_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2511"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2507-.2510"),
                "current_problem_hit": sign_base.hit(current_problem_text, "updated-pack blind-vector refresh audit"),
                "current_status_hit": sign_base.hit(current_status_text, "updated-pack blind-vector refresh audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2503-.2510"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2511-.2514"),
                "part5_hit": sign_base.hit(part5_text, ".2503-.2510"),
                "step_d_hit": sign_base.hit(next_steps_text, "### Step D. blind vector computation を行う"),
                "precondition_hit": sign_base.hit(next_steps_text, "上の A/B/C が通ったら"),
                "q0_hit": sign_base.hit(next_steps_text, "- `q = 0`"),
                "q_theory_hit": sign_base.hit(next_steps_text, "- `q = q_theory`"),
                "m0_hit": sign_base.hit(next_steps_text, "- `q = m_0`"),
                "alpha_hit": sign_base.hit(next_steps_text, r"\alpha_{\rm vector}(q)="),
                "residual_hit": sign_base.hit(next_steps_text, "scalar 残差 1.9% を改善するか"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2514",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_updated_pack_blind_vector_refresh_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": {
            "formulae": build_formulae(),
            "disposition": {
                "blind_vector_refresh_surface_explicit": updated_pack_blind_vector_refresh_target_surface_explicit,
                "blind_vector_refresh_machine_readable_now": updated_pack_blind_vector_refresh_machine_readable_now,
                "direct_blind_vector_still_blocked": blind_vector_observable_gate_still_blocked,
                "residual_origin_followup_required": residual_origin_refresh_followup_required,
            },
        },
    }
    route_paths = write_artifact("route_sync", route_payload)

    print("[ok] updated-pack blind-vector refresh audit artifacts written")
    print(f"  declaration_gate: {declaration_paths['json']}")
    print(f"  route_sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
