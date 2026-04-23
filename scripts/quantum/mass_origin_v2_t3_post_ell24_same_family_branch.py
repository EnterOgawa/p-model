#!/usr/bin/env python3
"""
Generate Trial-3 refactored post-ell24 same-family re-audit artifacts for 8.7.56.293-.296.

The previous radial-domain branch already reopened localized exact-family states
through ell=24 and improved the same-family weak-sector frontier. The next
honest question is narrower: does that reopened ell=20..24 family already carry
enough structure to close the W/Z and Weinberg-angle pack, or is the remaining
blocker now the residual target gap above the current same-family ceiling?
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

HELPER_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_t3_post_ell18_amplitude_branch.py"
PREVIOUS_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_t3_post_ell19_radial_branch.py"

PRIOR_RADIAL_SOURCE = OUT / "mass_origin_v2_trial3_refactored_post_ell19_radial_domain_extension_source_inventory_metrics.json"
PRIOR_RADIAL_AUDIT = OUT / "mass_origin_v2_trial3_refactored_post_ell19_radial_domain_extension_audit_metrics.json"
PRIOR_DECLARATION = OUT / "mass_origin_v2_trial3_refactored_declaration_fourth_gate_metrics.json"
PRIOR_DISPOSITION = OUT / "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_sixteenth_refresh_metrics.json"
AMPLITUDE_AUDIT = OUT / "mass_origin_v2_trial3_refactored_post_ell18_central_amplitude_window_extension_audit_metrics.json"
REFACTORED_WEAK_AUDIT = OUT / "mass_origin_v2_trial3_refactored_high_mass_k_positive_extension_audit_metrics.json"
SOLVER_REFACTOR_WEAK = OUT / "mass_origin_v2_trial3_solver_refactor_weak_sector_reaudit_metrics.json"

PASS_THRESHOLD = 0.10
NEAR_PAIR_THRESHOLD = 0.15


# 関数: helper branch を動的 import する。
def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 関数: post-ell24 same-family re-audit branch を実行する。

def main() -> None:
    helper = load_module(HELPER_BRANCH, "trial3_post_ell24_same_family_helper")

    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        HELPER_BRANCH,
        PREVIOUS_BRANCH,
        PRIOR_RADIAL_SOURCE,
        PRIOR_RADIAL_AUDIT,
        PRIOR_DECLARATION,
        PRIOR_DISPOSITION,
        AMPLITUDE_AUDIT,
        REFACTORED_WEAK_AUDIT,
        SOLVER_REFACTOR_WEAK,
    ):
        helper.req(path)

    status_text = helper.read_text(STATUS)
    roadmap_text = helper.read_text(ROADMAP)
    ai_context = helper.read_json(AI_CONTEXT)
    previous_branch_text = helper.read_text(PREVIOUS_BRANCH)
    prior_source = helper.read_json(PRIOR_RADIAL_SOURCE)
    prior_audit = helper.read_json(PRIOR_RADIAL_AUDIT)
    prior_declaration = helper.read_json(PRIOR_DECLARATION)
    prior_disposition = helper.read_json(PRIOR_DISPOSITION)
    amplitude_audit = helper.read_json(AMPLITUDE_AUDIT)
    refactored_weak_audit = helper.read_json(REFACTORED_WEAK_AUDIT)
    solver_refactor_weak = helper.read_json(SOLVER_REFACTOR_WEAK)

    prior_summary = prior_audit["summary"]
    best_w = prior_summary["best_w_row_or_none"]
    best_z = prior_summary["best_z_row_or_none"]
    best_pair = prior_summary["best_pair_or_none"]
    localized_ell_values = [int(value) for value in prior_summary["localized_ell_values"]]
    localized_above_ell24 = max(localized_ell_values) >= 24 if localized_ell_values else False
    w_anchor_pass = bool(best_w and bool(best_w["passes_threshold"]))
    z_anchor_pass = bool(best_z and bool(best_z["passes_threshold"]))
    mw_mz_ratio_pass = bool(best_pair and float(best_pair["mw_mz_ratio_relative_error"]) <= PASS_THRESHOLD)
    sin2_theta_w_pass = bool(best_pair and float(best_pair["sin2_theta_w_relative_error"]) <= PASS_THRESHOLD)
    best_pair_near_pass = bool(best_pair and float(best_pair["mw_mz_ratio_relative_error"]) <= NEAR_PAIR_THRESHOLD)
    trial3_recommended_condition_satisfied = bool(prior_summary["trial3_recommended_condition_satisfied"])
    anchor_gap_present = bool(
        prior_summary["w_gap_factor_or_none"] is not None
        and prior_summary["z_gap_factor_or_none"] is not None
        and float(prior_summary["w_gap_factor_or_none"]) > 1.0
        and float(prior_summary["z_gap_factor_or_none"]) > 1.0
    )

    inventory_targets = [
        {
            "label": "status_points_to_8_7_56_293",
            "present": "current official next step は `8.7.56.293`" in status_text,
            "evidence": {"status_markdown": helper.rel(STATUS)},
        },
        {
            "label": "roadmap_post_ell24_same_family_branch_present",
            "present": "`8.7.56.293-.296` 試練3 refactored post-`ell=24` same-family re-audit residual branch" in roadmap_text,
            "evidence": {"roadmap_markdown": helper.rel(ROADMAP)},
        },
        {
            "label": "prior_radial_rebuild_present",
            "present": bool(prior_source["summary"]["required_source_count_present"] == prior_source["summary"]["required_source_count"]),
            "evidence": {
                "required_source_count_present": prior_source["summary"]["required_source_count_present"],
                "required_source_count": prior_source["summary"]["required_source_count"],
            },
        },
        {
            "label": "localized_ell20_24_family_present",
            "present": localized_above_ell24,
            "evidence": {
                "localized_ell_values": localized_ell_values,
                "post_ell19_localized_solution_count_total": prior_summary["post_ell19_localized_solution_count_total"],
                "post_ell19_integer_mode_count_total": prior_summary["post_ell19_integer_mode_count_total"],
            },
        },
        {
            "label": "same_family_anchor_pair_pack_present",
            "present": bool(best_w and best_z and best_pair),
            "evidence": {
                "best_w_row_or_none": best_w,
                "best_z_row_or_none": best_z,
                "best_pair_or_none": best_pair,
            },
        },
        {
            "label": "previous_branch_same_family_route_present",
            "present": helper.hit(previous_branch_text, "selected_residual_route = f\"trial3_relaunched_refactored_post_ell{highest_ell}_same_family_reaudit\"") is not None,
            "evidence": helper.hit(previous_branch_text, "selected_residual_route = f\"trial3_relaunched_refactored_post_ell{highest_ell}_same_family_reaudit\""),
        },
        {
            "label": "prior_declaration_points_to_same_family_reaudit",
            "present": prior_declaration["summary"]["selected_residual_route"] == "trial3_relaunched_refactored_post_ell24_same_family_reaudit",
            "evidence": prior_declaration["summary"],
        },
        {
            "label": "prior_disposition_keeps_trial2_reserve_and_trial4_deferred",
            "present": prior_disposition["summary"]["trial2_paper_side_sync_state"] == "unlocked_reserve_retained"
            and bool(prior_disposition["summary"]["trial4_deferred"]),
            "evidence": prior_disposition["summary"],
        },
        {
            "label": "same_family_remaining_gap_present",
            "present": anchor_gap_present,
            "evidence": {
                "w_gap_factor_or_none": prior_summary["w_gap_factor_or_none"],
                "z_gap_factor_or_none": prior_summary["z_gap_factor_or_none"],
            },
        },
    ]

    source_inventory = helper.payload(
        "8.7.56.293",
        "Trial-3 refactored post-ell24 same-family re-audit source inventory",
        {
            "status_markdown": helper.rel(STATUS),
            "roadmap_markdown": helper.rel(ROADMAP),
            "ai_context_json": helper.rel(AI_CONTEXT),
            "mass_origin_v2_trial3_refactored_post_ell19_radial_domain_extension_source_inventory_json": helper.rel(PRIOR_RADIAL_SOURCE),
            "mass_origin_v2_trial3_refactored_post_ell19_radial_domain_extension_audit_json": helper.rel(PRIOR_RADIAL_AUDIT),
            "mass_origin_v2_trial3_refactored_declaration_fourth_gate_json": helper.rel(PRIOR_DECLARATION),
            "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_sixteenth_refresh_json": helper.rel(PRIOR_DISPOSITION),
            "mass_origin_v2_trial3_refactored_post_ell18_central_amplitude_window_extension_audit_json": helper.rel(AMPLITUDE_AUDIT),
            "mass_origin_v2_trial3_refactored_high_mass_k_positive_extension_audit_json": helper.rel(REFACTORED_WEAK_AUDIT),
            "mass_origin_v2_trial3_solver_refactor_weak_sector_reaudit_json": helper.rel(SOLVER_REFACTOR_WEAK),
            "mass_origin_v2_t3_post_ell19_radial_branch_py": helper.rel(PREVIOUS_BRANCH),
        },
        "Freeze the reopened ell=20..24 exact-family state pack, same-family W/Z anchor and pair metrics, and the remaining closeout gap before deciding whether Trial-3 is now blocked by a residual target-gap extension rather than by localization or solver artifacts.",
        {
            "closeout_pack_rule": "the post-ell24 same-family pack closes Trial-3 only if the reopened family simultaneously passes W anchor, Z anchor, M_W/M_Z, and sin^2(theta_W)",
            "residual_rule": "if ell=20..24 localized families exist but the reopened exact-family ceiling still undershoots W/Z and the Weinberg-angle proxy fails, the next blocker is the same-family target gap above the current ceiling rather than localization, amplitude, or radial contracts",
        },
        [
            helper.row("trial3_refactored_post_ell24_same_family_source_inventory_complete", "pass", "Trial-3 refactored post-ell24 same-family re-audit source inventory complete", 1, "The reopened same-family evidence pack is frozen."),
            helper.row("trial3_refactored_post_ell24_localized_family_present", "pass" if localized_above_ell24 else "reject", "localized same-family evidence reaches ell=24", 1 if localized_above_ell24 else 0, "The same-family re-audit is only honest if the reopened family genuinely extends through ell=24."),
            helper.row("trial3_refactored_post_ell24_anchor_pair_pack_present", "pass" if best_w and best_z and best_pair else "reject", "same-family W/Z anchor and pair pack present", 1 if best_w and best_z and best_pair else 0, "The re-audit requires both anchor rows and the best-pair diagnostic in one pack."),
            helper.row("trial3_refactored_post_ell24_remaining_gap_present", "pass" if anchor_gap_present else "reject", "same-family remaining gap pack present", 1 if anchor_gap_present else 0, "The post-ell24 branch must explicitly freeze the remaining W/Z gap rather than hand-wave it."),
        ],
        {
            "required_sources_total": len(inventory_targets),
            "required_sources_present": sum(1 for item in inventory_targets if item["present"]),
            "localized_ell_values": localized_ell_values,
            "maximum_detected_ell": prior_summary["maximum_detected_ell"],
            "maximum_detected_ell_with_k_positive": prior_summary["maximum_detected_ell_with_k_positive"],
            "best_w_row_or_none": best_w,
            "best_z_row_or_none": best_z,
            "best_pair_or_none": best_pair,
            "w_gap_factor_or_none": prior_summary["w_gap_factor_or_none"],
            "z_gap_factor_or_none": prior_summary["z_gap_factor_or_none"],
            "status_current_step_before_branch": ai_context["current_step"],
        },
        {
            "overall_status": "trial3_refactored_post_ell24_same_family_source_inventory_frozen",
            "source_inventory_complete": True,
            "advance_to_8_7_56_294": True,
            "next_required_artifacts": ["trial3_refactored_post_ell24_same_family_reaudit"],
        },
        {
            "inventory_targets": inventory_targets,
            "prior_radial_audit_summary": prior_summary,
            "prior_declaration_summary": prior_declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
            "amplitude_audit_summary": amplitude_audit["summary"],
            "refactored_weak_audit_summary": refactored_weak_audit["summary"],
            "solver_refactor_weak_summary": solver_refactor_weak["summary"],
        },
    )

    selected_residual_route = None
    missing_v2_artifact = None
    if not trial3_recommended_condition_satisfied:
        selected_residual_route = "trial3_relaunched_refactored_post_ell24_same_family_target_gap_extension_identification"
        missing_v2_artifact = "trial3_relaunched_refactored_post_ell24_same_family_closeout_pack_above_current_ceiling"

    reaudit = helper.payload(
        "8.7.56.294",
        "Trial-3 refactored post-ell24 same-family re-audit",
        source_inventory["inputs"],
        "Re-audit whether the reopened ell=20..24 same-family exact table already closes the weak-sector pack or whether the remaining blocker is now the residual target gap above the current ceiling.",
        {
            "closeout_rule": "Trial-3 closes only if the reopened same-family pack passes both absolute W/Z anchors and the relative W/Z pair constraints",
            "residual_rule": "if the current same-family family improves the ceiling and pair structure but still leaves W/Z under target, the next blocker is a same-family target-gap extension rather than another localization retry",
        },
        [
            helper.row("trial3_refactored_post_ell24_same_family_reaudit_complete", "pass", "Trial-3 refactored post-ell24 same-family re-audit complete", 1, "The post-ell24 same-family re-audit is frozen."),
            helper.row("trial3_refactored_post_ell24_same_family_family_present", "pass" if localized_above_ell24 else "reject", "same-family exact family present through ell=24", 1 if localized_above_ell24 else 0, "The reopened same-family family must remain present through ell=24 before closeout is judged."),
            helper.row("trial3_refactored_post_ell24_same_family_w_anchor_pass", "pass" if w_anchor_pass else "reject", "same-family W/electron anchor passes", 1 if w_anchor_pass else 0, "The W anchor is mandatory for an honest weak-sector closeout."),
            helper.row("trial3_refactored_post_ell24_same_family_z_anchor_pass", "pass" if z_anchor_pass else "reject", "same-family Z/electron anchor passes", 1 if z_anchor_pass else 0, "The Z anchor must close together with the W anchor."),
            helper.row("trial3_refactored_post_ell24_same_family_mw_mz_ratio_pass", "pass" if mw_mz_ratio_pass else "reject", "same-family M_W/M_Z pair passes", 1 if mw_mz_ratio_pass else 0, "Relative pair structure cannot replace the missing absolute anchors."),
            helper.row("trial3_refactored_post_ell24_same_family_sin2_theta_w_pass", "pass" if sin2_theta_w_pass else "reject", "same-family sin^2(theta_W) passes", 1 if sin2_theta_w_pass else 0, "The Weinberg-angle proxy must close together with the pair and anchors."),
            helper.row("trial3_refactored_post_ell24_same_family_best_pair_near_pass", "pass" if best_pair_near_pass else "reject", "same-family M_W/M_Z pair is structurally near pass", 1 if best_pair_near_pass else 0, "The pair has become structurally close enough to justify moving the blocker from localization into the remaining target gap."),
            helper.row("trial3_refactored_post_ell24_same_family_target_gap_extension_required", "pass" if not trial3_recommended_condition_satisfied else "reject", "same-family target-gap extension required", 1 if not trial3_recommended_condition_satisfied else 0, "The reopened family still needs a higher-ceiling closeout pack when anchors remain sub-target."),
        ],
        {
            "trial3_recommended_condition_satisfied": trial3_recommended_condition_satisfied,
            "localized_ell_values": localized_ell_values,
            "best_anchor_ell_or_none": None if best_w is None else int(best_w["ell"]),
            "best_anchor_k_or_none": None if best_w is None else int(best_w["k"]),
            "best_w_row_or_none": best_w,
            "best_z_row_or_none": best_z,
            "best_pair_or_none": best_pair,
            "w_gap_factor_or_none": prior_summary["w_gap_factor_or_none"],
            "z_gap_factor_or_none": prior_summary["z_gap_factor_or_none"],
            "best_pair_near_pass": best_pair_near_pass,
            "same_family_closeout_pack_available": bool(trial3_recommended_condition_satisfied),
            "same_family_target_gap_extension_required": bool(not trial3_recommended_condition_satisfied),
        },
        {
            "overall_status": "trial3_refactored_post_ell24_same_family_reaudited",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_295": True,
            "next_required_artifacts": ["trial3_refactored_declaration_fifth_gate"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "prior_radial_audit_summary": prior_summary,
            "prior_declaration_summary": prior_declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
        },
    )

    declaration = helper.payload(
        "8.7.56.295",
        "Trial-3 refactored declaration fifth gate",
        source_inventory["inputs"],
        "Freeze whether the post-ell24 same-family family already closes Trial-3 or whether the next honest route is a residual target-gap extension above the current ceiling.",
        {
            "closeout_rule": "Trial-3 closes only if the post-ell24 same-family family passes W anchor, Z anchor, M_W/M_Z, and sin^2(theta_W) together",
            "residual_rule": "if the pair becomes structurally near-pass but the anchors still miss target, the next honest blocker is the same-family target gap rather than another shape or solver artifact",
        },
        [
            helper.row("trial3_refactored_declaration_fifth_gate_complete", "pass", "Trial-3 refactored declaration fifth gate complete", 1, "The fifth declaration gate is frozen."),
            helper.row("trial3_refactored_fifth_branch_closeable", "pass" if trial3_recommended_condition_satisfied else "reject", "refactored Trial-3 branch closeable after post-ell24 same-family re-audit", 1 if trial3_recommended_condition_satisfied else 0, "The branch closes only when the reopened same-family pack really closes the weak sector."),
            helper.row("trial3_refactored_fifth_residual_route_required", "reject" if trial3_recommended_condition_satisfied else "pass", "refactored Trial-3 residual route required after post-ell24 same-family re-audit", 0 if trial3_recommended_condition_satisfied else 1, "A new residual route is still required when the same-family family remains below target."),
            helper.row("trial3_refactored_execute_trial2_paper_sync_now_fifth_gate", "reject", "execute Trial-2 paper-side sync now after post-ell24 same-family re-audit", 0, "Trial-2 paper-side sync remains reserve work while Trial-3 still has an honest same-family extension route."),
        ],
        {
            "trial3_recommended_condition_satisfied": trial3_recommended_condition_satisfied,
            "trial3_current_branch_closeable": trial3_recommended_condition_satisfied,
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "recommended_next_route_or_none": None if trial3_recommended_condition_satisfied else "8.7.56.297",
        },
        {
            "overall_status": "trial3_refactored_declaration_fifth_gate_frozen",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_296": True,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["trial3_refactored_trial2_paper_sync_trial4_disposition_seventeenth_refresh"],
        },
        {
            "reaudit_summary": reaudit["summary"],
            "prior_declaration_summary": prior_declaration["summary"],
            "best_pair_or_none": best_pair,
        },
    )

    disposition = helper.payload(
        "8.7.56.296",
        "Trial-2 paper-side sync / Trial-4 disposition seventeenth refresh",
        source_inventory["inputs"],
        "Refresh the reserve/deferred ordering after the post-ell24 same-family re-audit and freeze the next official residual route.",
        {
            "trial2_rule": "Trial-2 paper-side sync stays unlocked reserve retained while the refactored Trial-3 route still has an honest same-family extension path",
            "trial4_rule": "Trial-4 remains deferred while Trial-3 continues to expose a current-canon weak-sector frontier",
        },
        [
            helper.row("trial3_refactored_trial2_trial4_seventeenth_refresh_complete", "pass", "Trial-2 paper-side sync / Trial-4 disposition seventeenth refresh complete", 1, "The reserve/deferred ordering is refreshed after the post-ell24 same-family re-audit."),
            helper.row("trial3_refactored_trial2_paper_side_sync_reserve_retained_seventeenth_refresh", "pass", "Trial-2 paper-side sync reserve retained", 1, "Trial-2 paper-side sync stays unlocked reserve work while the same-family Trial-3 route is still open."),
            helper.row("trial3_refactored_trial4_deferred_retained_seventeenth_refresh", "pass", "Trial-4 deferred retained", 1, "Trial-4 stays deferred while Trial-3 still has an honest same-family extension path."),
        ],
        {
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "trial2_paper_side_sync_state": "unlocked_reserve_retained",
            "trial4_deferred": True,
            "recommended_next_route_or_none": None if trial3_recommended_condition_satisfied else "8.7.56.297",
        },
        {
            "overall_status": "trial3_refactored_trial2_trial4_seventeenth_disposition_refreshed",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_13": False,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["8.7.56.297"],
        },
        {
            "declaration_summary": declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
            "solver_refactor_weak_summary": solver_refactor_weak["summary"],
        },
    )

    helper.write_artifact("mass_origin_v2_trial3_refactored_post_ell24_same_family_reaudit_source_inventory", source_inventory)
    helper.write_artifact("mass_origin_v2_trial3_refactored_post_ell24_same_family_reaudit", reaudit)
    helper.write_artifact("mass_origin_v2_trial3_refactored_declaration_fifth_gate", declaration)
    helper.write_artifact("mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_seventeenth_refresh", disposition)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_refactored_post_ell24_same_family_reaudit_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_post_ell24_same_family_reaudit_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_declaration_fifth_gate_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_seventeenth_refresh_metrics.json")


# 関数: CLI から post-ell24 same-family branch を起動する。

if __name__ == "__main__":
    main()
