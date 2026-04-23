#!/usr/bin/env python3
"""Generate 8.7.56.2043-.2046 alias-image phase-slip registry artifacts."""

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

PRIOR_GATE = PUBLIC_OUT / "q_8_7_56_2039_2042_alias_image_phase_slip_theorem_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.2043-2046"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor alias-image shared phase-slip closeout / registry"
STEM = build_compact_artifact_stem(STEP_TAG, "alias_image_phase_slip_registry", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_boundary_alias_image_local_jet_phase_slip_theorem_derived_higher_harmonic_generalization_gate_next"
BRANCH_CLASS = "vector_qball_form_factor_boundary_alias_image_local_jet_phase_slip_theorem_retained_q_dependent_or_higher_harmonic_loading_reactivation_next"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_q_dependent_boundary_phase_slip_loading_or_higher_harmonic_signed_rule_reactivation"
NEXT_ROUTE = "8.7.56.2047"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_alias_image_phase_slip_loading_closeout_registry"
FOLLOWUP_ROUTE = "8.7.56.2051"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: JSON/CSV artifact を書き出す。

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one rows CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": sign_base.display_path(paths["json"]), "csv": sign_base.display_path(paths["csv"])}


# 関数: 使用公式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the registry sync."""
    return {
        "boundary_phase_slip_theorem": "delta_q,jet = (3/2) (h1 / h0)",
        "active_window_rule": "sigma_jet^(n)(q)=(-1)^n sign(F_exact(|q_alias^(n)+(-1)^(n+1) delta_q,jet-q|))",
        "next_surface": "q-dependent delta_q(q) or harmonic-index dependent loading is required once the active theorem closes but translated higher-harmonic templates fail",
    }


# 関数: `.2043-.2046` を実行する。

def main() -> None:
    """Execute the alias-image phase-slip closeout / registry sync."""
    for path in (
        STATUS, ROADMAP, AI_CONTEXT, WORK_HISTORY_RECENT, CURRENT_PROBLEM, CURRENT_STATUS,
        UNIFIED_ROADMAP, LONG_ROADMAP, PART5, PRIOR_GATE
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
    inventory_ready = bool(prior_summary["boundary_local_jet_phase_slip_theorem_derived"])

    gate_a_exact_boundary_phase_slip_theorem_retained = bool(prior_summary["boundary_local_jet_phase_slip_theorem_derived"])
    gate_b_higher_harmonic_generalization_blocked = bool(not prior_summary["higher_harmonic_generalization_supported"])
    gate_c_current_rule_blocked = False
    same_level_constant_delta_retry_admissible = bool(prior_summary["same_level_constant_delta_retry_admissible"])
    q_dependent_or_higher_harmonic_loading_admissible_now = bool(prior_summary["q_dependent_or_higher_harmonic_loading_admissible_now"])
    substantive_pack_update_required_now = False
    physical_reject_required = False

    rows = [
        sign_base.row("inventory_ready", "pass" if inventory_ready else "reject", "phase-slip registry inventory ready", sign_base.truth(inventory_ready), "The closeout registry starts only after `.2039-.2042` has fixed whether a boundary-only phase-slip theorem exists at all."),
        sign_base.row("gate_a_exact_boundary_phase_slip_theorem_retained", "pass" if gate_a_exact_boundary_phase_slip_theorem_retained else "reject", "Gate A exact boundary phase-slip theorem retained", sign_base.truth(gate_a_exact_boundary_phase_slip_theorem_retained), "The boundary local-jet theorem is retained because delta_q,jet reproduces the active-window shared phase-slip without a free fit."),
        sign_base.row("gate_b_higher_harmonic_generalization_blocked", "pass" if gate_b_higher_harmonic_generalization_blocked else "reject", "Gate B higher-harmonic generalization blocked", sign_base.truth(gate_b_higher_harmonic_generalization_blocked), "The same constant-slip family fails on translated higher-harmonic template windows, so the unresolved gap is no longer the active theorem itself."),
        sign_base.row("gate_c_current_rule_blocked", "reject" if not gate_c_current_rule_blocked else "pass", "Gate C current rule blocked", sign_base.truth(gate_c_current_rule_blocked), "The current retained pack still contains a clear next loading surface, so a full pack reset is not yet required."),
        sign_base.row("same_level_constant_delta_retry_admissible", "reject" if not same_level_constant_delta_retry_admissible else "pass", "same-level constant-delta retry admissible", sign_base.truth(same_level_constant_delta_retry_admissible), "Once the constant-slip theorem and its failure mode are fixed, same-level constant-delta refits should remain closed."),
        sign_base.row("q_dependent_or_higher_harmonic_loading_admissible_now", "pass" if q_dependent_or_higher_harmonic_loading_admissible_now else "reject", "q-dependent or higher-harmonic loading admissible now", sign_base.truth(q_dependent_or_higher_harmonic_loading_admissible_now), "The honest next surface is a q-dependent loading or harmonic-index dependent signed rule, not another constant-delta scan."),
        sign_base.row("substantive_pack_update_required_now", "reject" if not substantive_pack_update_required_now else "pass", "substantive pack update required now", sign_base.truth(substantive_pack_update_required_now), "A pack update remains reserve because the current retained pack still has one internal loading question left."),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "delta_q_theorem_over_m0": float(prior_summary["delta_q_theorem_over_m0"]),
        "delta_q_theorem_vs_window_optima_max_abs_gap": float(prior_summary["delta_q_theorem_vs_window_optima_max_abs_gap"]),
        "theorem_fit_window_sign_mismatch_fraction": float(prior_summary["theorem_fit_window_sign_mismatch_fraction"]),
        "theorem_edge_window_sign_mismatch_fraction": float(prior_summary["theorem_edge_window_sign_mismatch_fraction"]),
        "harmonic3_fit_window_sign_mismatch_fraction": float(prior_summary["harmonic3_fit_window_sign_mismatch_fraction"]),
        "harmonic4_edge_window_sign_mismatch_fraction": float(prior_summary["harmonic4_edge_window_sign_mismatch_fraction"]),
        "harmonic5_fit_holdout_sign_mismatch_fraction": float(prior_summary["harmonic5_fit_holdout_sign_mismatch_fraction"]),
        "harmonic6_edge_holdout_sign_mismatch_fraction": float(prior_summary["harmonic6_edge_holdout_sign_mismatch_fraction"]),
        "boundary_local_jet_phase_slip_theorem_derived": gate_a_exact_boundary_phase_slip_theorem_retained,
        "higher_harmonic_generalization_supported": bool(prior_summary["higher_harmonic_generalization_supported"]),
        "gate_a_exact_boundary_phase_slip_theorem_retained": gate_a_exact_boundary_phase_slip_theorem_retained,
        "gate_b_higher_harmonic_generalization_blocked": gate_b_higher_harmonic_generalization_blocked,
        "gate_c_current_rule_blocked": gate_c_current_rule_blocked,
        "same_level_constant_delta_retry_admissible": same_level_constant_delta_retry_admissible,
        "q_dependent_or_higher_harmonic_loading_admissible_now": q_dependent_or_higher_harmonic_loading_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2045",
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
            "constants": {
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {"overall_status": "vector_qball_form_factor_alias_image_phase_slip_registry_declared", "branch_completed": True, "next_required_artifacts": [NEXT_ROUTE_NAME]},
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2043"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.2043-.2046"),
                "current_problem_hit": sign_base.hit(current_problem_text, "phase-slip theorem"),
                "current_status_hit": sign_base.hit(current_status_text, "phase-slip theorem"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2043-.2046"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2043-.2046"),
                "part5_hit": sign_base.hit(part5_text, ".2031-.2038"),
            },
        },
    )

    route_payload = sign_base.payload(
        "8.7.56.2046",
        STEP_NAME + " route sync",
        declaration_payload["inputs"],
        [
            sign_base.row("gate_a_exact_boundary_phase_slip_theorem_retained", "pass" if gate_a_exact_boundary_phase_slip_theorem_retained else "reject", "Gate A exact boundary phase-slip theorem retained", sign_base.truth(gate_a_exact_boundary_phase_slip_theorem_retained), "The active-window boundary theorem is now part of the retained pack."),
            sign_base.row("gate_b_higher_harmonic_generalization_blocked", "pass" if gate_b_higher_harmonic_generalization_blocked else "reject", "Gate B higher-harmonic generalization blocked", sign_base.truth(gate_b_higher_harmonic_generalization_blocked), "The unresolved gap has moved from active-window theorem derivation to q-dependent or harmonic-index dependent loading."),
            sign_base.row("next_route_fixed", "pass", "next route fixed", 1.0, "The next official branch is the q-dependent boundary phase-slip loading or higher-harmonic signed-rule reactivation."),
        ],
        summary,
        {"overall_status": "vector_qball_form_factor_alias_image_phase_slip_registry_route_synced", "branch_completed": True, "next_required_artifacts": [NEXT_ROUTE_NAME]},
        {"formulas": build_formulae()},
    )

    declaration_paths = write_artifact("declaration_gate", declaration_payload)
    route_paths = write_artifact("route_sync", route_payload)
    print("[done] 8.7.56.2043-.2046 complete")
    print(f"[info] declaration gate: {declaration_paths['json']}")
    print(f"[info] route sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
