#!/usr/bin/env python3
"""Generate 8.7.56.2431-.2434 trial3 ell=0 closure reserve audit artifacts."""

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
        "8.7.56.2427-2430",
        "phase1_nonlinear_closure_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
EIGSHIFT_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2359-2362",
        "exact_coupled_eigshift_theorem",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
ELL0_OPERATOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.1471-1474",
        "ell0_exact_operator_derivation",
        prefix="q",
    ),
    "audit",
)["json"]

TRIAL3_PIVOT_ROUTE = PUBLIC_OUT / "mass_origin_v2_trial3_two_component_pivot_route_contract_metrics.json"
TRIAL3_SPECTRUM = PUBLIC_OUT / "mass_origin_v2_trial3_two_component_spectrum_computation_metrics.json"
TRIAL3_WZ = PUBLIC_OUT / "mass_origin_v2_trial3_two_component_wz_target_comparison_metrics.json"
TRIAL3_DECLARATION = PUBLIC_OUT / "mass_origin_v2_trial3_two_component_declaration_gate_metrics.json"

TRIAL3_PIVOT_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_pivot_branch.py"
TRIAL3_SPECTRUM_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_spectrum_branch.py"

STEP_TAG = "8.7.56.2431-2434"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor trial3 ell=0 closure reserve audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "trial3_ell0_closure_reserve_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_phase1_nonheuristic_two_component_"
    "nonlinear_closure_missing_trial3_ell0_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_trial3_ell0_reserve_scalarlike_"
    "inventory_only_pack_update_gate"
)
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_trial3_ell0_reserve_gate_pack_update_refresh"
NEXT_ROUTE = "8.7.56.2435"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_substantive_pack_update_audit"
FOLLOWUP_ROUTE = "8.7.56.2439"


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


# 関数: trial3 ell=0 reserve audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the trial3 ell=0 reserve audit."""
    return {
        "trial3_coupling_proxy": "C_ell(beta,r) = beta sqrt(ell(ell+1)) / r",
        "ell0_collapse_rule": "C_0(beta,r) = 0",
        "reserve_rule": "localized ell=0 inventory plus integer-mode carry-over may survive as reserve support even when the literal coupled ell=0 operator is absent",
        "pack_update_rule": "if the trial3 ell=0 reserve stays scalarlike and non-closing while the exact ell=0 operator remains open, the next honest move is substantive pack-update refresh",
    }


# 関数: ell=0 mode summary を集約する。

def collect_ell0_mode_summary(mode_summary: dict[str, dict]) -> tuple[dict[str, dict], int]:
    """Collect all ell=0 groups and their total integer-mode count."""
    ell0_groups = {
        key: value
        for key, value in mode_summary.items()
        if key.startswith("0:")
    }
    total = sum(int(value["integer_mode_count"]) for value in ell0_groups.values())
    return ell0_groups, int(total)


# 関数: `.2431-.2434` を実行する。

def main() -> None:
    """Execute the trial3 ell=0 closure reserve audit."""
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
        EIGSHIFT_AUDIT,
        ELL0_OPERATOR_AUDIT,
        TRIAL3_PIVOT_ROUTE,
        TRIAL3_SPECTRUM,
        TRIAL3_WZ,
        TRIAL3_DECLARATION,
        TRIAL3_PIVOT_BRANCH,
        TRIAL3_SPECTRUM_BRANCH,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)
    pivot_branch_text = sign_base.read_text(TRIAL3_PIVOT_BRANCH)
    spectrum_branch_text = sign_base.read_text(TRIAL3_SPECTRUM_BRANCH)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    eigshift_summary = sign_base.read_json(EIGSHIFT_AUDIT)["summary"]
    ell0_operator_summary = sign_base.read_json(ELL0_OPERATOR_AUDIT)["summary"]
    pivot_summary = sign_base.read_json(TRIAL3_PIVOT_ROUTE)["summary"]
    spectrum_payload = sign_base.read_json(TRIAL3_SPECTRUM)
    wz_summary = sign_base.read_json(TRIAL3_WZ)["summary"]
    declaration_summary = sign_base.read_json(TRIAL3_DECLARATION)["summary"]

    sector0_summary = spectrum_payload["evidence"]["sector_summary"]["0"]
    ell0_mode_groups, ell0_integer_mode_count = collect_ell0_mode_summary(
        spectrum_payload["evidence"]["mode_summary"]
    )
    best_ell0_candidate = sector0_summary["best_candidate_or_none"]
    best_ell0_component_ratio = float(best_ell0_candidate["max_abs_fL"]) / max(
        float(best_ell0_candidate["max_abs_f0"]),
        1.0e-18,
    )

    trial3_ell0_reserve_audit_selected = bool(
        prior_summary["gate_b_trial3_ell0_closure_reserve_refreshed_next"]
    )
    trial3_family_solver_ell0_coupling_collapses = bool(
        eigshift_summary["trial3_family_solver_ell0_coupling_collapses"]
        and "math.sqrt(max(float(ell * (ell + 1)), 0.0)) / rr" in pivot_branch_text
        and "math.sqrt(max(float(ell * (ell + 1)), 0.0)) / rr" in spectrum_branch_text
    )
    trial3_family_ell0_coupling_zero_rule_explicit = bool(
        "C_ell(beta,r) = beta sqrt(ell(ell+1)) / r" in pivot_branch_text
        and "ELL_VALUES = tuple(range(0, 31))" in spectrum_branch_text
        and trial3_family_solver_ell0_coupling_collapses
    )
    trial3_family_ell0_raw_localized_sector_present = bool(
        int(sector0_summary["localized_solution_count"]) > 0
    )
    trial3_family_ell0_base_mode_surface_present = bool(
        ell0_integer_mode_count > 0
    )
    trial3_family_ell0_scalarlike_support_only = bool(
        best_ell0_component_ratio < 0.01
    )
    exact_action_level_closed_ell0_operator_available = bool(
        ell0_operator_summary["exact_action_level_closed_ell0_operator_available"]
    )
    trial3_family_overall_branch_closeable_under_current_pack = bool(
        declaration_summary["trial3_current_branch_closeable"]
    )
    trial3_family_ell0_reserve_supporting_inventory_available = bool(
        trial3_family_ell0_raw_localized_sector_present
        and trial3_family_ell0_base_mode_surface_present
    )
    trial3_family_ell0_coupled_operator_reuse_available = bool(
        trial3_family_ell0_reserve_supporting_inventory_available
        and not trial3_family_solver_ell0_coupling_collapses
        and exact_action_level_closed_ell0_operator_available
    )
    trial3_family_ell0_primary_fix_available = bool(
        trial3_family_ell0_coupled_operator_reuse_available
        and trial3_family_overall_branch_closeable_under_current_pack
    )
    trial3_ell0_reserve_closes_current_missing_action_blocker_now = bool(
        trial3_family_ell0_primary_fix_available
    )
    substantive_pack_update_followup_supported = bool(
        trial3_ell0_reserve_audit_selected
        and trial3_family_ell0_reserve_supporting_inventory_available
        and not trial3_ell0_reserve_closes_current_missing_action_blocker_now
        and not exact_action_level_closed_ell0_operator_available
    )
    farther_hybrid_continuation_reopen_required_now = False
    pack_update_required_now = False

    rows = [
        sign_base.row(
            "trial3_ell0_reserve_audit_selected",
            "pass" if trial3_ell0_reserve_audit_selected else "reject",
            "trial3 ell=0 reserve audit selected",
            sign_base.truth(trial3_ell0_reserve_audit_selected),
            "This branch starts only after `.2427-.2430` refreshed the trial3 ell=0 surface as the last current-pack reserve audit.",
        ),
        sign_base.row(
            "trial3_family_solver_ell0_coupling_collapses",
            "watch" if trial3_family_solver_ell0_coupling_collapses else "pass",
            "trial3 family solver ell=0 coupling collapses",
            sign_base.truth(trial3_family_solver_ell0_coupling_collapses),
            "The old trial3 family inherits an ell-dependent coupling proxy, so the literal temporal/longitudinal mixing vanishes at ell=0.",
        ),
        sign_base.row(
            "trial3_family_ell0_coupling_zero_rule_explicit",
            "pass" if trial3_family_ell0_coupling_zero_rule_explicit else "reject",
            "trial3 ell=0 coupling-zero rule explicit",
            sign_base.truth(trial3_family_ell0_coupling_zero_rule_explicit),
            "The code makes the collapse explicit through `sqrt(ell(ell+1))`, which becomes zero at ell=0.",
        ),
        sign_base.row(
            "trial3_family_ell0_raw_localized_sector_present",
            "pass" if trial3_family_ell0_raw_localized_sector_present else "reject",
            "trial3 ell=0 raw localized sector present",
            sign_base.truth(trial3_family_ell0_raw_localized_sector_present),
            "The archived two-component trial3 scan still contains localized ell=0 sectors, so reserve inventory exists at the raw scan level.",
        ),
        sign_base.row(
            "trial3_family_ell0_base_mode_surface_present",
            "pass" if trial3_family_ell0_base_mode_surface_present else "reject",
            "trial3 ell=0 base-mode surface present",
            float(ell0_integer_mode_count),
            "The same archived scan interpolates ell=0 integer-charge base modes, so the reserve survives as a carry-over table rather than disappearing completely.",
        ),
        sign_base.row(
            "trial3_family_ell0_scalarlike_support_only",
            "watch" if trial3_family_ell0_scalarlike_support_only else "pass",
            "trial3 ell=0 reserve is scalarlike support only",
            float(best_ell0_component_ratio),
            "The best archived ell=0 candidate keeps `max|f_L|/max|f_0|` at a tiny level, which is consistent with scalarlike carry-over rather than a literal coupled closure fix.",
        ),
        sign_base.row(
            "trial3_family_overall_branch_closeable_under_current_pack",
            "pass" if trial3_family_overall_branch_closeable_under_current_pack else "reject",
            "trial3 family overall branch closeable under current pack",
            sign_base.truth(trial3_family_overall_branch_closeable_under_current_pack),
            "The archived trial3 family never closed under the current pack, so reserve reuse cannot be promoted just by revisiting the ell=0 slice.",
        ),
        sign_base.row(
            "trial3_family_ell0_reserve_supporting_inventory_available",
            "pass" if trial3_family_ell0_reserve_supporting_inventory_available else "reject",
            "trial3 ell=0 reserve supporting inventory available",
            sign_base.truth(trial3_family_ell0_reserve_supporting_inventory_available),
            "The honest reserve survives as archived localized sectors plus integer-mode carry-over, not as an empty branch.",
        ),
        sign_base.row(
            "trial3_family_ell0_coupled_operator_reuse_available",
            "pass" if trial3_family_ell0_coupled_operator_reuse_available else "reject",
            "trial3 ell=0 coupled-operator reuse available",
            sign_base.truth(trial3_family_ell0_coupled_operator_reuse_available),
            "Reserve inventory would become a coupled-operator reuse only if the ell=0 coupling survived and the exact action-level ell=0 operator were already closed, which is not the current pack.",
        ),
        sign_base.row(
            "trial3_family_ell0_primary_fix_available",
            "pass" if trial3_family_ell0_primary_fix_available else "reject",
            "trial3 ell=0 primary fix available",
            sign_base.truth(trial3_family_ell0_primary_fix_available),
            "The old trial3 family cannot be restored as the primary missing-action fix from the ell=0 reserve slice.",
        ),
        sign_base.row(
            "trial3_ell0_reserve_closes_current_missing_action_blocker_now",
            "pass" if trial3_ell0_reserve_closes_current_missing_action_blocker_now else "reject",
            "trial3 ell=0 reserve closes current missing-action blocker now",
            sign_base.truth(trial3_ell0_reserve_closes_current_missing_action_blocker_now),
            "The reserve does not close the current blocker because it stays decoupled/scalarlike while the exact ell=0 operator is still open.",
        ),
        sign_base.row(
            "substantive_pack_update_followup_supported",
            "pass" if substantive_pack_update_followup_supported else "reject",
            "substantive pack-update followup supported",
            sign_base.truth(substantive_pack_update_followup_supported),
            "Once the trial3 ell=0 reserve is fixed as support-only, the next honest move is a pack-update refresh rather than another same-level retry.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "The reserve audit itself only prepares the gate; the actual pack-update promotion is synchronized in the followup branch.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "trial3_ell0_reserve_audit_selected": trial3_ell0_reserve_audit_selected,
        "trial3_family_solver_ell0_coupling_collapses": trial3_family_solver_ell0_coupling_collapses,
        "trial3_family_ell0_coupling_zero_rule_explicit": trial3_family_ell0_coupling_zero_rule_explicit,
        "trial3_family_ell0_raw_localized_sector_present": trial3_family_ell0_raw_localized_sector_present,
        "trial3_family_ell0_base_mode_surface_present": trial3_family_ell0_base_mode_surface_present,
        "trial3_family_ell0_integer_mode_count": int(ell0_integer_mode_count),
        "trial3_family_ell0_best_component_ratio": float(best_ell0_component_ratio),
        "trial3_family_ell0_scalarlike_support_only": trial3_family_ell0_scalarlike_support_only,
        "exact_action_level_closed_ell0_operator_available": exact_action_level_closed_ell0_operator_available,
        "trial3_family_overall_branch_closeable_under_current_pack": trial3_family_overall_branch_closeable_under_current_pack,
        "trial3_family_ell0_reserve_supporting_inventory_available": trial3_family_ell0_reserve_supporting_inventory_available,
        "trial3_family_ell0_coupled_operator_reuse_available": trial3_family_ell0_coupled_operator_reuse_available,
        "trial3_family_ell0_primary_fix_available": trial3_family_ell0_primary_fix_available,
        "trial3_ell0_reserve_closes_current_missing_action_blocker_now": trial3_ell0_reserve_closes_current_missing_action_blocker_now,
        "substantive_pack_update_followup_supported": substantive_pack_update_followup_supported,
        "pack_update_required_now": pack_update_required_now,
        "hybrid_supporting_evidence_reopen_required": farther_hybrid_continuation_reopen_required_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2433",
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
                "eigshift_audit": sign_base.display_path(EIGSHIFT_AUDIT),
                "ell0_operator_audit": sign_base.display_path(ELL0_OPERATOR_AUDIT),
                "trial3_pivot_route": sign_base.display_path(TRIAL3_PIVOT_ROUTE),
                "trial3_spectrum": sign_base.display_path(TRIAL3_SPECTRUM),
                "trial3_wz": sign_base.display_path(TRIAL3_WZ),
                "trial3_declaration": sign_base.display_path(TRIAL3_DECLARATION),
                "trial3_pivot_branch": sign_base.display_path(TRIAL3_PIVOT_BRANCH),
                "trial3_spectrum_branch": sign_base.display_path(TRIAL3_SPECTRUM_BRANCH),
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
            "overall_status": "vector_qball_form_factor_trial3_ell0_reserve_audit_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2431"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2431-.2434"),
                "current_problem_hit": sign_base.hit(current_problem_text, "trial3 ell=0 closure reserve audit"),
                "current_status_hit": sign_base.hit(current_status_text, "trial3 ell=0 closure reserve audit"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2427-.2430"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2427-.2430"),
                "part5_hit": sign_base.hit(part5_text, "trial3 ell=0 closure reserve audit"),
                "pivot_coupling_hit": sign_base.hit(pivot_branch_text, "C_ell(beta,r) = beta sqrt(ell(ell+1)) / r"),
                "pivot_k_proxy_hit": sign_base.hit(pivot_branch_text, "math.sqrt(max(float(ell * (ell + 1)), 0.0)) / rr"),
                "spectrum_ell_grid_hit": sign_base.hit(spectrum_branch_text, "ELL_VALUES = tuple(range(0, 31))"),
                "spectrum_k_proxy_hit": sign_base.hit(spectrum_branch_text, "math.sqrt(max(float(ell * (ell + 1)), 0.0)) / rr"),
            },
            "ell0_sector_summary": sector0_summary,
            "ell0_mode_groups": ell0_mode_groups,
            "pivot_route_summary": pivot_summary,
            "wz_summary": wz_summary,
            "trial3_declaration_summary": declaration_summary,
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2434",
            "name": STEP_NAME + " route sync",
        },
        "inputs": declaration_paths,
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_trial3_ell0_reserve_route_synced",
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

    print(f"[done] {STEP_TAG} trial3 ell=0 closure reserve audit completed")


if __name__ == "__main__":
    main()
