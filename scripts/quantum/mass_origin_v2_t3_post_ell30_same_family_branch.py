#!/usr/bin/env python3
"""
Generate Trial-3 refactored post-ell30 same-family re-audit artifacts for
8.7.56.309-.312.

The higher-ell frontier extension already reopened localized same-family states
through ell=30. The next honest question is narrower: does that reopened
ell=25..30 family actually lift the same-family weak-sector ceiling, anchor,
or pair diagnostics, or has the blocker shifted into a higher-ceiling stall?
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
PREVIOUS_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_t3_post_ell24_frontier_branch.py"

PRIOR_FRONTIER_SOURCE = OUT / "mass_origin_v2_trial3_refactored_post_ell24_higher_ell_frontier_extension_source_inventory_metrics.json"
PRIOR_FRONTIER_AUDIT = OUT / "mass_origin_v2_trial3_refactored_post_ell24_higher_ell_frontier_extension_audit_metrics.json"
PRIOR_DECLARATION = OUT / "mass_origin_v2_trial3_refactored_declaration_eighth_gate_metrics.json"
PRIOR_DISPOSITION = OUT / "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_twentieth_refresh_metrics.json"
PRIOR_SAME_FAMILY_AUDIT = OUT / "mass_origin_v2_trial3_refactored_post_ell24_same_family_reaudit_metrics.json"
PRIOR_HIGHER_AUDIT = OUT / "mass_origin_v2_trial3_refactored_post_ell24_same_family_higher_ceiling_extension_audit_metrics.json"


# 関数: helper branch を動的 import する。
def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 関数: post-ell30 same-family re-audit branch を実行する。

def main() -> None:
    helper = load_module(HELPER_BRANCH, "trial3_post_ell30_same_family_helper")

    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        HELPER_BRANCH,
        PREVIOUS_BRANCH,
        PRIOR_FRONTIER_SOURCE,
        PRIOR_FRONTIER_AUDIT,
        PRIOR_DECLARATION,
        PRIOR_DISPOSITION,
        PRIOR_SAME_FAMILY_AUDIT,
        PRIOR_HIGHER_AUDIT,
    ):
        helper.req(path)

    status_text = helper.read_text(STATUS)
    roadmap_text = helper.read_text(ROADMAP)
    ai_context = helper.read_json(AI_CONTEXT)
    previous_branch_text = helper.read_text(PREVIOUS_BRANCH)
    prior_source = helper.read_json(PRIOR_FRONTIER_SOURCE)
    prior_audit = helper.read_json(PRIOR_FRONTIER_AUDIT)
    prior_declaration = helper.read_json(PRIOR_DECLARATION)
    prior_disposition = helper.read_json(PRIOR_DISPOSITION)
    prior_same_family_audit = helper.read_json(PRIOR_SAME_FAMILY_AUDIT)
    prior_higher_audit = helper.read_json(PRIOR_HIGHER_AUDIT)

    prior_summary = prior_audit["summary"]
    prior_same_family_summary = prior_same_family_audit["summary"]
    best_w = prior_summary["best_w_row_or_none"]
    best_z = prior_summary["best_z_row_or_none"]
    best_pair = prior_summary["best_pair_or_none"]
    localized_ell_values = [int(value) for value in prior_summary["localized_ell_values"]]
    frontier_reopened_to_30 = bool(localized_ell_values and max(localized_ell_values) >= 30)
    current_ceiling = float(prior_summary["rebuilt_verified_ceiling_to_electron"])
    pre_frontier_ceiling = float(prior_summary["prior_rebuilt_verified_ceiling_to_electron"])
    ceiling_improved = bool(current_ceiling > pre_frontier_ceiling)
    best_anchor_unchanged = bool(
        best_w is not None
        and prior_same_family_summary["best_w_row_or_none"] is not None
        and int(best_w["ell"]) == int(prior_same_family_summary["best_w_row_or_none"]["ell"])
        and int(best_w["k"]) == int(prior_same_family_summary["best_w_row_or_none"]["k"])
        and float(best_w["ratio_value"]) == float(prior_same_family_summary["best_w_row_or_none"]["ratio_value"])
    )
    best_pair_unchanged = bool(
        best_pair is not None
        and prior_same_family_summary["best_pair_or_none"] is not None
        and float(best_pair["mw_mz_ratio_value"]) == float(prior_same_family_summary["best_pair_or_none"]["mw_mz_ratio_value"])
        and float(best_pair["sin2_theta_w_value"]) == float(prior_same_family_summary["best_pair_or_none"]["sin2_theta_w_value"])
    )
    trial3_recommended_condition_satisfied = bool(prior_summary["trial3_recommended_condition_satisfied"])
    higher_ceiling_stall_dominant = bool(frontier_reopened_to_30 and (not ceiling_improved) and best_anchor_unchanged and best_pair_unchanged)

    inventory_targets = [
        {
            "label": "status_points_to_8_7_56_309",
            "present": "current official next step は `8.7.56.309`" in status_text,
            "evidence": {"status_markdown": helper.rel(STATUS)},
        },
        {
            "label": "roadmap_post_ell30_same_family_branch_present",
            "present": "`8.7.56.309-.312` 試練3 refactored post-`ell=30` same-family re-audit residual branch" in roadmap_text,
            "evidence": {"roadmap_markdown": helper.rel(ROADMAP)},
        },
        {
            "label": "prior_frontier_source_ready",
            "present": bool(prior_source["summary"]["required_sources_present"] == prior_source["summary"]["required_sources_total"]),
            "evidence": prior_source["summary"],
        },
        {
            "label": "reopened_ell25_30_family_present",
            "present": frontier_reopened_to_30,
            "evidence": {
                "localized_ell_values": localized_ell_values,
                "tail_localized_solution_count_total": prior_summary["tail_localized_solution_count_total"],
                "tail_integer_mode_count_total": prior_summary["tail_integer_mode_count_total"],
            },
        },
        {
            "label": "same_family_ceiling_still_present",
            "present": current_ceiling > 0.0,
            "evidence": {
                "pre_frontier_ceiling_to_electron": pre_frontier_ceiling,
                "current_ceiling_to_electron": current_ceiling,
                "ceiling_improved": ceiling_improved,
            },
        },
        {
            "label": "best_anchor_pack_present",
            "present": best_w is not None and best_z is not None,
            "evidence": {
                "best_w_row_or_none": best_w,
                "best_z_row_or_none": best_z,
                "best_anchor_unchanged": best_anchor_unchanged,
            },
        },
        {
            "label": "best_pair_pack_present",
            "present": best_pair is not None,
            "evidence": {
                "best_pair_or_none": best_pair,
                "best_pair_unchanged": best_pair_unchanged,
            },
        },
        {
            "label": "previous_branch_points_to_post_ell30_reaudit",
            "present": helper.hit(previous_branch_text, "selected_residual_route = \"trial3_relaunched_refactored_post_ell30_same_family_reaudit_identification\"") is not None,
            "evidence": helper.hit(previous_branch_text, "selected_residual_route = \"trial3_relaunched_refactored_post_ell30_same_family_reaudit_identification\""),
        },
        {
            "label": "prior_disposition_keeps_trial2_reserve_and_trial4_deferred",
            "present": prior_disposition["summary"]["trial2_paper_side_sync_state"] == "unlocked_reserve_retained"
            and bool(prior_disposition["summary"]["trial4_deferred"]),
            "evidence": prior_disposition["summary"],
        },
    ]

    source_inventory = helper.payload(
        "8.7.56.309",
        "Trial-3 refactored post-ell30 same-family re-audit source inventory",
        {
            "status_markdown": helper.rel(STATUS),
            "roadmap_markdown": helper.rel(ROADMAP),
            "ai_context_json": helper.rel(AI_CONTEXT),
            "mass_origin_v2_trial3_refactored_post_ell24_higher_ell_frontier_extension_source_inventory_json": helper.rel(PRIOR_FRONTIER_SOURCE),
            "mass_origin_v2_trial3_refactored_post_ell24_higher_ell_frontier_extension_audit_json": helper.rel(PRIOR_FRONTIER_AUDIT),
            "mass_origin_v2_trial3_refactored_declaration_eighth_gate_json": helper.rel(PRIOR_DECLARATION),
            "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_twentieth_refresh_json": helper.rel(PRIOR_DISPOSITION),
            "mass_origin_v2_trial3_refactored_post_ell24_same_family_reaudit_json": helper.rel(PRIOR_SAME_FAMILY_AUDIT),
            "mass_origin_v2_trial3_refactored_post_ell24_same_family_higher_ceiling_extension_audit_json": helper.rel(PRIOR_HIGHER_AUDIT),
            "mass_origin_v2_t3_post_ell24_frontier_branch_py": helper.rel(PREVIOUS_BRANCH),
        },
        "Freeze the rebuilt exact-family table, the reopened ell=25..30 localized family, the unchanged best anchor/pair diagnostics, and the remaining W/Z gap before deciding whether the blocker is now a higher-ceiling stall rather than frontier existence.",
        {
            "closeout_pack_rule": "the post-ell30 same-family pack closes Trial-3 only if the reopened family lifts the same-family ceiling and simultaneously closes W anchor, Z anchor, M_W/M_Z, and sin^2(theta_W)",
            "stall_rule": "if ell=25..30 families reopen but the same-family ceiling, best anchor, and best pair remain unchanged, the honest next blocker is a higher-ceiling stall rather than another frontier-existence question",
        },
        [
            helper.row("trial3_refactored_post_ell30_same_family_source_inventory_complete", "pass", "Trial-3 refactored post-ell30 same-family re-audit source inventory complete", 1, "The post-ell30 same-family evidence pack is frozen."),
            helper.row("trial3_refactored_post_ell30_reopened_family_present", "pass" if frontier_reopened_to_30 else "reject", "reopened same-family family present through ell=30", 1 if frontier_reopened_to_30 else 0, "The post-ell30 re-audit is only honest if the reopened family genuinely reaches ell=30."),
            helper.row("trial3_refactored_post_ell30_current_ceiling_present", "pass" if current_ceiling > 0.0 else "reject", "current same-family ceiling present", current_ceiling, "The re-audit must explicitly freeze the current same-family ceiling before judging whether the new frontier lifted it."),
            helper.row("trial3_refactored_post_ell30_anchor_pair_pack_present", "pass" if best_w is not None and best_z is not None and best_pair is not None else "reject", "best anchor and pair pack present", 1 if best_w is not None and best_z is not None and best_pair is not None else 0, "The re-audit requires the best anchors and pair metrics in one pack."),
        ],
        {
            "required_sources_total": len(inventory_targets),
            "required_sources_present": sum(1 for item in inventory_targets if item["present"]),
            "localized_ell_values": localized_ell_values,
            "maximum_detected_ell": prior_summary["maximum_detected_ell"],
            "maximum_detected_ell_with_k_positive": prior_summary["maximum_detected_ell_with_k_positive"],
            "pre_frontier_ceiling_to_electron": pre_frontier_ceiling,
            "current_ceiling_to_electron": current_ceiling,
            "best_w_row_or_none": best_w,
            "best_z_row_or_none": best_z,
            "best_pair_or_none": best_pair,
            "w_gap_factor_or_none": prior_summary["w_gap_factor_or_none"],
            "z_gap_factor_or_none": prior_summary["z_gap_factor_or_none"],
            "status_current_step_before_branch": ai_context["current_step"],
        },
        {
            "overall_status": "trial3_refactored_post_ell30_same_family_source_inventory_frozen",
            "source_inventory_complete": True,
            "advance_to_8_7_56_310": True,
            "next_required_artifacts": ["trial3_refactored_post_ell30_same_family_reaudit"],
        },
        {
            "inventory_targets": inventory_targets,
            "prior_frontier_audit_summary": prior_summary,
            "prior_same_family_audit_summary": prior_same_family_summary,
            "prior_higher_audit_summary": prior_higher_audit["summary"],
            "prior_declaration_summary": prior_declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
        },
    )

    selected_residual_route = None
    missing_v2_artifact = None
    if not trial3_recommended_condition_satisfied:
        selected_residual_route = "trial3_relaunched_refactored_post_ell30_same_family_higher_ceiling_stall_identification"
        missing_v2_artifact = "trial3_relaunched_refactored_post_ell30_same_family_ceiling_lifting_pack"

    reaudit = helper.payload(
        "8.7.56.310",
        "Trial-3 refactored post-ell30 same-family re-audit",
        source_inventory["inputs"],
        "Re-audit whether the reopened ell=25..30 same-family exact family actually improves the weak-sector closeout pack or whether the blocker has collapsed into a higher-ceiling stall.",
        {
            "closeout_rule": "Trial-3 closes only if the reopened ell=25..30 family lifts the ceiling and closes both anchors plus the pair-side observables together",
            "stall_rule": "if the frontier extends to ell=30 but the ceiling, anchors, and pair diagnostics remain unchanged, the next blocker is a same-family higher-ceiling stall",
        },
        [
            helper.row("trial3_refactored_post_ell30_same_family_reaudit_complete", "pass", "Trial-3 refactored post-ell30 same-family re-audit complete", 1, "The post-ell30 same-family re-audit is frozen."),
            helper.row("trial3_refactored_post_ell30_same_family_family_present", "pass" if frontier_reopened_to_30 else "reject", "same-family exact family present through ell=30", 1 if frontier_reopened_to_30 else 0, "The reopened same-family family must remain present through ell=30 before closeout is judged."),
            helper.row("trial3_refactored_post_ell30_same_family_ceiling_improved", "pass" if ceiling_improved else "reject", "same-family ceiling improves beyond pre-frontier baseline", 1 if ceiling_improved else 0, "The reopened tail family should raise the same-family ceiling if it is actually solving the remaining weak-sector gap."),
            helper.row("trial3_refactored_post_ell30_same_family_anchor_unchanged", "pass" if best_anchor_unchanged else "reject", "best same-family anchor remains unchanged", 1 if best_anchor_unchanged else 0, "The dominant blocker is ceiling-lifting only if the best anchor really stays pinned to the pre-frontier state."),
            helper.row("trial3_refactored_post_ell30_same_family_pair_unchanged", "pass" if best_pair_unchanged else "reject", "best same-family pair remains unchanged", 1 if best_pair_unchanged else 0, "The dominant blocker is ceiling-lifting only if the pair-side observables also remain pinned."),
            helper.row("trial3_refactored_post_ell30_same_family_w_anchor_pass", "pass" if best_w and best_w["passes_threshold"] else "reject", "same-family W/electron anchor passes", 1 if best_w and best_w["passes_threshold"] else 0, "The W anchor is mandatory for an honest weak-sector closeout."),
            helper.row("trial3_refactored_post_ell30_same_family_z_anchor_pass", "pass" if best_z and best_z["passes_threshold"] else "reject", "same-family Z/electron anchor passes", 1 if best_z and best_z["passes_threshold"] else 0, "The Z anchor must close together with the W anchor."),
            helper.row("trial3_refactored_post_ell30_same_family_higher_ceiling_stall_dominant", "pass" if higher_ceiling_stall_dominant else "reject", "same-family higher-ceiling stall dominant blocker", 1 if higher_ceiling_stall_dominant else 0, "The honest next blocker is a higher-ceiling stall when the frontier is present but no leading weak-sector diagnostic improves."),
        ],
        {
            "trial3_recommended_condition_satisfied": trial3_recommended_condition_satisfied,
            "localized_ell_values": localized_ell_values,
            "pre_frontier_ceiling_to_electron": pre_frontier_ceiling,
            "current_ceiling_to_electron": current_ceiling,
            "ceiling_improved": ceiling_improved,
            "best_anchor_unchanged": best_anchor_unchanged,
            "best_pair_unchanged": best_pair_unchanged,
            "best_w_row_or_none": best_w,
            "best_z_row_or_none": best_z,
            "best_pair_or_none": best_pair,
            "w_gap_factor_or_none": prior_summary["w_gap_factor_or_none"],
            "z_gap_factor_or_none": prior_summary["z_gap_factor_or_none"],
            "same_family_higher_ceiling_stall_dominant": higher_ceiling_stall_dominant,
        },
        {
            "overall_status": "trial3_refactored_post_ell30_same_family_reaudited",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_311": True,
            "next_required_artifacts": ["trial3_refactored_declaration_ninth_gate"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "prior_frontier_audit_summary": prior_summary,
            "prior_same_family_audit_summary": prior_same_family_summary,
            "prior_declaration_summary": prior_declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
        },
    )

    declaration = helper.payload(
        "8.7.56.311",
        "Trial-3 refactored declaration ninth gate",
        source_inventory["inputs"],
        "Freeze whether the post-ell30 same-family family already closes Trial-3 or whether the next honest route is a higher-ceiling stall investigation.",
        {
            "closeout_rule": "Trial-3 closes only if the post-ell30 same-family family passes W anchor, Z anchor, M_W/M_Z, and sin^2(theta_W) together",
            "residual_rule": "if the frontier reaches ell=30 but the same-family ceiling and anchor/pair pack stay unchanged, the next honest blocker is a higher-ceiling stall rather than another frontier-extension question",
        },
        [
            helper.row("trial3_refactored_declaration_ninth_gate_complete", "pass", "Trial-3 refactored declaration ninth gate complete", 1, "The ninth declaration gate is frozen."),
            helper.row("trial3_refactored_ninth_branch_closeable", "pass" if trial3_recommended_condition_satisfied else "reject", "refactored Trial-3 branch closeable after post-ell30 same-family re-audit", 1 if trial3_recommended_condition_satisfied else 0, "The branch closes only when the reopened same-family pack really closes the weak sector."),
            helper.row("trial3_refactored_ninth_residual_route_required", "reject" if trial3_recommended_condition_satisfied else "pass", "refactored Trial-3 residual route required after post-ell30 same-family re-audit", 0 if trial3_recommended_condition_satisfied else 1, "A new residual route is still required when the reopened same-family family remains below target."),
            helper.row("trial3_refactored_execute_trial2_paper_sync_now_ninth_gate", "reject", "execute Trial-2 paper-side sync now after post-ell30 same-family re-audit", 0, "Trial-2 paper-side sync remains reserve work while Trial-3 still has an honest same-family stall-resolution path."),
        ],
        {
            "trial3_recommended_condition_satisfied": trial3_recommended_condition_satisfied,
            "trial3_current_branch_closeable": trial3_recommended_condition_satisfied,
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "recommended_next_route_or_none": None if trial3_recommended_condition_satisfied else "8.7.56.313",
        },
        {
            "overall_status": "trial3_refactored_declaration_ninth_gate_frozen",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_312": True,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["trial3_refactored_trial2_paper_sync_trial4_disposition_twenty_first_refresh"],
        },
        {
            "reaudit_summary": reaudit["summary"],
            "prior_declaration_summary": prior_declaration["summary"],
            "best_pair_or_none": best_pair,
        },
    )

    disposition = helper.payload(
        "8.7.56.312",
        "Trial-2 paper-side sync / Trial-4 disposition twenty-first refresh",
        source_inventory["inputs"],
        "Refresh the reserve/deferred ordering after the post-ell30 same-family re-audit and freeze the next official higher-ceiling stall route.",
        {
            "trial2_rule": "Trial-2 paper-side sync stays unlocked reserve retained while the refactored Trial-3 route still has an honest same-family stall-resolution path",
            "trial4_rule": "Trial-4 remains deferred while Trial-3 continues to expose a current-canon weak-sector route",
        },
        [
            helper.row("trial3_refactored_trial2_trial4_twenty_first_refresh_complete", "pass", "Trial-2 paper-side sync / Trial-4 disposition twenty-first refresh complete", 1, "The reserve/deferred ordering is refreshed after the post-ell30 same-family re-audit."),
            helper.row("trial3_refactored_trial2_paper_side_sync_reserve_retained_twenty_first_refresh", "pass", "Trial-2 paper-side sync reserve retained", 1, "Trial-2 paper-side sync stays unlocked reserve work while the same-family Trial-3 route is still open."),
            helper.row("trial3_refactored_trial4_deferred_retained_twenty_first_refresh", "pass", "Trial-4 deferred retained", 1, "Trial-4 stays deferred while Trial-3 still has an honest same-family stall-resolution path."),
        ],
        {
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "trial2_paper_side_sync_state": "unlocked_reserve_retained",
            "trial4_deferred": True,
            "recommended_next_route_or_none": None if trial3_recommended_condition_satisfied else "8.7.56.313",
        },
        {
            "overall_status": "trial3_refactored_trial2_trial4_twenty_first_disposition_refreshed",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_13": False,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["8.7.56.313"],
        },
        {
            "declaration_summary": declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
        },
    )

    helper.write_artifact("mass_origin_v2_trial3_refactored_post_ell30_same_family_reaudit_source_inventory", source_inventory)
    helper.write_artifact("mass_origin_v2_trial3_refactored_post_ell30_same_family_reaudit", reaudit)
    helper.write_artifact("mass_origin_v2_trial3_refactored_declaration_ninth_gate", declaration)
    helper.write_artifact("mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_twenty_first_refresh", disposition)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_refactored_post_ell30_same_family_reaudit_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_post_ell30_same_family_reaudit_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_declaration_ninth_gate_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_twenty_first_refresh_metrics.json")


# 関数: CLI から post-ell30 same-family re-audit branch を起動する。

if __name__ == "__main__":
    main()
