#!/usr/bin/env python3
"""
Generate Trial-3 refactored post-ell30 same-family higher-ceiling stall
artifacts for 8.7.56.313-.316.

The post-ell30 same-family re-audit already fixed the honest qualitative state:
the ell=25..30 family is present, but the same-family ceiling, best anchor, and
best pair remain pinned to the pre-frontier pack. The next honest question is
therefore narrower: is the blocker now the missing prerequisite that would let
the reopened post-ell30 family overtake the incumbent ell=22 anchor?
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
PREVIOUS_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_t3_post_ell30_same_family_branch.py"

PRIOR_SOURCE = OUT / "mass_origin_v2_trial3_refactored_post_ell30_same_family_reaudit_source_inventory_metrics.json"
PRIOR_AUDIT = OUT / "mass_origin_v2_trial3_refactored_post_ell30_same_family_reaudit_metrics.json"
PRIOR_DECLARATION = OUT / "mass_origin_v2_trial3_refactored_declaration_ninth_gate_metrics.json"
PRIOR_DISPOSITION = OUT / "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_twenty_first_refresh_metrics.json"
PRIOR_FRONTIER_AUDIT = OUT / "mass_origin_v2_trial3_refactored_post_ell24_higher_ell_frontier_extension_audit_metrics.json"


# 関数: helper branch を動的 import する。
def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 関数: post-ell30 same-family higher-ceiling stall branch を実行する。

def main() -> None:
    helper = load_module(HELPER_BRANCH, "trial3_post_ell30_higher_ceiling_stall_helper")

    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        HELPER_BRANCH,
        PREVIOUS_BRANCH,
        PRIOR_SOURCE,
        PRIOR_AUDIT,
        PRIOR_DECLARATION,
        PRIOR_DISPOSITION,
        PRIOR_FRONTIER_AUDIT,
    ):
        helper.req(path)

    status_text = helper.read_text(STATUS)
    roadmap_text = helper.read_text(ROADMAP)
    ai_context = helper.read_json(AI_CONTEXT)
    previous_branch_text = helper.read_text(PREVIOUS_BRANCH)
    prior_source = helper.read_json(PRIOR_SOURCE)
    prior_audit = helper.read_json(PRIOR_AUDIT)
    prior_declaration = helper.read_json(PRIOR_DECLARATION)
    prior_disposition = helper.read_json(PRIOR_DISPOSITION)
    prior_frontier_audit = helper.read_json(PRIOR_FRONTIER_AUDIT)

    prior_source_summary = prior_source["summary"]
    prior_summary = prior_audit["summary"]
    declaration_summary = prior_declaration["summary"]
    disposition_summary = prior_disposition["summary"]
    frontier_summary = prior_frontier_audit["summary"]

    localized_ell_values = [int(value) for value in prior_summary["localized_ell_values"]]
    frontier_reopened_to_30 = bool(localized_ell_values and max(localized_ell_values) >= 30)
    current_ceiling = float(prior_summary["current_ceiling_to_electron"])
    pre_frontier_ceiling = float(prior_summary["pre_frontier_ceiling_to_electron"])
    ceiling_stalled = bool((not bool(prior_summary["ceiling_improved"])) and current_ceiling == pre_frontier_ceiling)
    best_w = prior_summary["best_w_row_or_none"]
    best_z = prior_summary["best_z_row_or_none"]
    best_pair = prior_summary["best_pair_or_none"]
    best_anchor_pinned_to_ell22 = bool(best_w is not None and int(best_w["ell"]) == 22 and int(best_w["k"]) == 0)
    best_pair_unchanged = bool(prior_summary["best_pair_unchanged"])
    same_family_ceiling_below_w = bool(current_ceiling < float(helper.W_TARGET))
    same_family_ceiling_below_z = bool(current_ceiling < float(helper.Z_TARGET))
    anchor_overtake_prerequisite_missing = bool(
        frontier_reopened_to_30
        and ceiling_stalled
        and best_anchor_pinned_to_ell22
        and best_pair_unchanged
        and same_family_ceiling_below_w
        and same_family_ceiling_below_z
    )
    trial3_recommended_condition_satisfied = bool(prior_summary["trial3_recommended_condition_satisfied"])

    inventory_targets = [
        {
            "label": "status_points_to_8_7_56_313",
            "present": "current official next step は `8.7.56.313`" in status_text,
            "evidence": {"status_markdown": helper.rel(STATUS)},
        },
        {
            "label": "roadmap_post_ell30_higher_ceiling_stall_branch_present",
            "present": "`8.7.56.313-.316` 試練3 refactored post-`ell=30` same-family higher-ceiling stall residual branch" in roadmap_text,
            "evidence": {"roadmap_markdown": helper.rel(ROADMAP)},
        },
        {
            "label": "prior_same_family_reaudit_pack_present",
            "present": bool(prior_source_summary["required_sources_present"] >= 8),
            "evidence": prior_source_summary,
        },
        {
            "label": "frontier_reopened_to_ell30_present",
            "present": frontier_reopened_to_30,
            "evidence": {
                "localized_ell_values": localized_ell_values,
                "maximum_detected_ell": frontier_summary["maximum_detected_ell"],
                "maximum_detected_ell_with_k_positive": frontier_summary["maximum_detected_ell_with_k_positive"],
            },
        },
        {
            "label": "same_family_ceiling_stall_present",
            "present": ceiling_stalled,
            "evidence": {
                "pre_frontier_ceiling_to_electron": pre_frontier_ceiling,
                "current_ceiling_to_electron": current_ceiling,
                "ceiling_improved": prior_summary["ceiling_improved"],
            },
        },
        {
            "label": "best_anchor_pack_pinned_to_ell22",
            "present": best_w is not None and best_z is not None and best_anchor_pinned_to_ell22,
            "evidence": {
                "best_w_row_or_none": best_w,
                "best_z_row_or_none": best_z,
                "best_anchor_unchanged": prior_summary["best_anchor_unchanged"],
            },
        },
        {
            "label": "best_pair_pack_unchanged",
            "present": best_pair is not None and best_pair_unchanged,
            "evidence": {
                "best_pair_or_none": best_pair,
                "best_pair_unchanged": best_pair_unchanged,
            },
        },
        {
            "label": "previous_branch_points_to_higher_ceiling_stall",
            "present": helper.hit(previous_branch_text, "selected_residual_route = \"trial3_relaunched_refactored_post_ell30_same_family_higher_ceiling_stall_identification\"") is not None,
            "evidence": helper.hit(previous_branch_text, "selected_residual_route = \"trial3_relaunched_refactored_post_ell30_same_family_higher_ceiling_stall_identification\""),
        },
        {
            "label": "prior_disposition_keeps_trial2_reserve_and_trial4_deferred",
            "present": disposition_summary["trial2_paper_side_sync_state"] == "unlocked_reserve_retained"
            and bool(disposition_summary["trial4_deferred"]),
            "evidence": disposition_summary,
        },
    ]

    source_inventory = helper.payload(
        "8.7.56.313",
        "Trial-3 refactored post-ell30 same-family higher-ceiling stall source inventory",
        {
            "status_markdown": helper.rel(STATUS),
            "roadmap_markdown": helper.rel(ROADMAP),
            "ai_context_json": helper.rel(AI_CONTEXT),
            "mass_origin_v2_trial3_refactored_post_ell30_same_family_reaudit_source_inventory_json": helper.rel(PRIOR_SOURCE),
            "mass_origin_v2_trial3_refactored_post_ell30_same_family_reaudit_json": helper.rel(PRIOR_AUDIT),
            "mass_origin_v2_trial3_refactored_declaration_ninth_gate_json": helper.rel(PRIOR_DECLARATION),
            "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_twenty_first_refresh_json": helper.rel(PRIOR_DISPOSITION),
            "mass_origin_v2_trial3_refactored_post_ell24_higher_ell_frontier_extension_audit_json": helper.rel(PRIOR_FRONTIER_AUDIT),
            "mass_origin_v2_t3_post_ell30_same_family_branch_py": helper.rel(PREVIOUS_BRANCH),
        },
        "Freeze the post-ell30 same-family ceiling stall pack before deciding whether the honest blocker is now the missing prerequisite that would let the reopened family overtake the incumbent ell=22 anchor.",
        {
            "stall_rule": "if the ell=25..30 family exists but the same-family ceiling, best anchor, and best pair stay pinned to the pre-frontier pack, the blocker is a higher-ceiling stall rather than another existence question",
            "overtake_rule": "if the incumbent best anchor remains the ell=22 state while the reopened family never lifts the ceiling, the honest next blocker is the missing prerequisite for anchor overtake rather than another generic higher-ell scan",
        },
        [
            helper.row("trial3_refactored_post_ell30_higher_ceiling_stall_source_inventory_complete", "pass", "Trial-3 refactored post-ell30 same-family higher-ceiling stall source inventory complete", 1, "The post-ell30 higher-ceiling stall evidence pack is frozen."),
            helper.row("trial3_refactored_post_ell30_frontier_present", "pass" if frontier_reopened_to_30 else "reject", "reopened same-family family present through ell=30", 1 if frontier_reopened_to_30 else 0, "The stall question is only honest if the post-ell30 family really remains present."),
            helper.row("trial3_refactored_post_ell30_ceiling_stall_present", "pass" if ceiling_stalled else "reject", "same-family ceiling stall present", 1 if ceiling_stalled else 0, "The stall branch requires the current ceiling to remain pinned to the pre-frontier value."),
            helper.row("trial3_refactored_post_ell30_anchor_pack_pinned", "pass" if best_anchor_pinned_to_ell22 else "reject", "best anchor pinned to incumbent ell=22 state", 1 if best_anchor_pinned_to_ell22 else 0, "The next blocker only shrinks if the incumbent ell=22 anchor still dominates the reopened family."),
        ],
        {
            "required_sources_total": len(inventory_targets),
            "required_sources_present": sum(1 for item in inventory_targets if item["present"]),
            "localized_ell_values": localized_ell_values,
            "current_ceiling_to_electron": current_ceiling,
            "pre_frontier_ceiling_to_electron": pre_frontier_ceiling,
            "ceiling_stalled": ceiling_stalled,
            "best_w_row_or_none": best_w,
            "best_z_row_or_none": best_z,
            "best_pair_or_none": best_pair,
            "w_gap_factor_or_none": prior_summary["w_gap_factor_or_none"],
            "z_gap_factor_or_none": prior_summary["z_gap_factor_or_none"],
            "status_current_step_before_branch": ai_context["current_step"],
        },
        {
            "overall_status": "trial3_refactored_post_ell30_higher_ceiling_stall_source_inventory_frozen",
            "source_inventory_complete": True,
            "advance_to_8_7_56_314": True,
            "next_required_artifacts": ["trial3_refactored_post_ell30_higher_ceiling_stall_audit"],
        },
        {
            "inventory_targets": inventory_targets,
            "prior_source_summary": prior_source_summary,
            "prior_audit_summary": prior_summary,
            "prior_declaration_summary": declaration_summary,
            "prior_disposition_summary": disposition_summary,
            "prior_frontier_audit_summary": frontier_summary,
        },
    )

    selected_residual_route = None
    missing_v2_artifact = None
    if not trial3_recommended_condition_satisfied:
        selected_residual_route = "trial3_relaunched_refactored_post_ell30_same_family_anchor_overtake_prerequisite_identification"
        missing_v2_artifact = "trial3_relaunched_refactored_post_ell30_same_family_anchor_overtake_prerequisite_pack"

    audit = helper.payload(
        "8.7.56.314",
        "Trial-3 refactored post-ell30 same-family higher-ceiling stall audit",
        source_inventory["inputs"],
        "Audit whether the post-ell30 same-family stall really collapses to a missing anchor-overtake prerequisite or whether another higher-ell existence question still remains open.",
        {
            "stall_rule": "if the reopened ell=25..30 family persists while the ceiling and pair stay unchanged, the weak-sector route is stalled at ceiling lifting rather than at family existence",
            "anchor_overtake_rule": "if the incumbent best anchor remains the ell=22 state and the ceiling never exceeds that state, the honest next blocker is the missing prerequisite for anchor overtake",
        },
        [
            helper.row("trial3_refactored_post_ell30_higher_ceiling_stall_audit_complete", "pass", "Trial-3 refactored post-ell30 same-family higher-ceiling stall audit complete", 1, "The post-ell30 higher-ceiling stall audit is frozen."),
            helper.row("trial3_refactored_post_ell30_frontier_exists", "pass" if frontier_reopened_to_30 else "reject", "post-ell30 same-family frontier exists", 1 if frontier_reopened_to_30 else 0, "The blocker can only shrink beyond existence questions if the frontier remains open."),
            helper.row("trial3_refactored_post_ell30_ceiling_stalled", "pass" if ceiling_stalled else "reject", "same-family ceiling remains stalled", 1 if ceiling_stalled else 0, "The reopened family should lift the ceiling if it is genuinely solving the remaining weak-sector gap."),
            helper.row("trial3_refactored_post_ell30_ceiling_below_w_threshold", "pass" if same_family_ceiling_below_w else "reject", "same-family ceiling remains below W threshold", 1 if same_family_ceiling_below_w else 0, "The W anchor remains upstream while the ceiling is still sub-threshold."),
            helper.row("trial3_refactored_post_ell30_ceiling_below_z_threshold", "pass" if same_family_ceiling_below_z else "reject", "same-family ceiling remains below Z threshold", 1 if same_family_ceiling_below_z else 0, "The Z anchor remains upstream while the ceiling is still sub-threshold."),
            helper.row("trial3_refactored_post_ell30_incumbent_anchor_still_pinned", "pass" if best_anchor_pinned_to_ell22 else "reject", "incumbent ell=22 anchor still pinned", 1 if best_anchor_pinned_to_ell22 else 0, "The next blocker only shrinks if the same ell=22 state still dominates the reopened family."),
            helper.row("trial3_refactored_post_ell30_anchor_overtake_prerequisite_missing", "pass" if anchor_overtake_prerequisite_missing else "reject", "anchor-overtake prerequisite missing", 1 if anchor_overtake_prerequisite_missing else 0, "The honest next blocker is the missing condition that would let post-ell30 states overtake the incumbent anchor."),
        ],
        {
            "trial3_recommended_condition_satisfied": trial3_recommended_condition_satisfied,
            "localized_ell_values": localized_ell_values,
            "current_ceiling_to_electron": current_ceiling,
            "pre_frontier_ceiling_to_electron": pre_frontier_ceiling,
            "ceiling_stalled": ceiling_stalled,
            "best_w_row_or_none": best_w,
            "best_z_row_or_none": best_z,
            "best_pair_or_none": best_pair,
            "w_target_to_electron": helper.W_TARGET,
            "z_target_to_electron": helper.Z_TARGET,
            "w_gap_factor_or_none": prior_summary["w_gap_factor_or_none"],
            "z_gap_factor_or_none": prior_summary["z_gap_factor_or_none"],
            "best_anchor_pinned_to_ell22": best_anchor_pinned_to_ell22,
            "best_pair_unchanged": best_pair_unchanged,
            "anchor_overtake_prerequisite_missing": anchor_overtake_prerequisite_missing,
        },
        {
            "overall_status": "trial3_refactored_post_ell30_higher_ceiling_stall_audited",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_315": True,
            "next_required_artifacts": ["trial3_refactored_declaration_tenth_gate"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "prior_audit_summary": prior_summary,
            "prior_declaration_summary": declaration_summary,
            "prior_disposition_summary": disposition_summary,
            "prior_frontier_audit_summary": frontier_summary,
        },
    )

    declaration = helper.payload(
        "8.7.56.315",
        "Trial-3 refactored declaration tenth gate",
        source_inventory["inputs"],
        "Freeze whether the post-ell30 same-family higher-ceiling stall branch already closes Trial-3 or whether the next honest route is an anchor-overtake prerequisite investigation.",
        {
            "closeout_rule": "Trial-3 closes only if the reopened same-family family now passes both anchors and the pair-side observables together",
            "residual_rule": "if the frontier exists but the incumbent ell=22 anchor still pins the ceiling, the next honest blocker is the missing prerequisite for anchor overtake",
        },
        [
            helper.row("trial3_refactored_declaration_tenth_gate_complete", "pass", "Trial-3 refactored declaration tenth gate complete", 1, "The tenth declaration gate is frozen."),
            helper.row("trial3_refactored_tenth_branch_closeable", "pass" if trial3_recommended_condition_satisfied else "reject", "refactored Trial-3 branch closeable after higher-ceiling stall audit", 1 if trial3_recommended_condition_satisfied else 0, "The branch closes only if the current same-family family already closes the weak-sector pack."),
            helper.row("trial3_refactored_tenth_residual_route_required", "reject" if trial3_recommended_condition_satisfied else "pass", "refactored Trial-3 residual route required after higher-ceiling stall audit", 0 if trial3_recommended_condition_satisfied else 1, "A new residual route is still required while the incumbent anchor keeps the ceiling pinned."),
            helper.row("trial3_refactored_execute_trial2_paper_sync_now_tenth_gate", "reject", "execute Trial-2 paper-side sync now after higher-ceiling stall audit", 0, "Trial-2 paper-side sync remains reserve work while Trial-3 still has an honest anchor-overtake path."),
        ],
        {
            "trial3_recommended_condition_satisfied": trial3_recommended_condition_satisfied,
            "trial3_current_branch_closeable": trial3_recommended_condition_satisfied,
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "recommended_next_route_or_none": None if trial3_recommended_condition_satisfied else "8.7.56.317",
        },
        {
            "overall_status": "trial3_refactored_declaration_tenth_gate_frozen",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_316": True,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["trial3_refactored_trial2_paper_sync_trial4_disposition_twenty_second_refresh"],
        },
        {
            "audit_summary": audit["summary"],
            "prior_declaration_summary": declaration_summary,
            "best_pair_or_none": best_pair,
        },
    )

    disposition = helper.payload(
        "8.7.56.316",
        "Trial-2 paper-side sync / Trial-4 disposition twenty-second refresh",
        source_inventory["inputs"],
        "Refresh the reserve/deferred ordering after the post-ell30 higher-ceiling stall audit and freeze the next official anchor-overtake prerequisite route.",
        {
            "trial2_rule": "Trial-2 paper-side sync stays unlocked reserve retained while the refactored Trial-3 route still has an honest anchor-overtake prerequisite path",
            "trial4_rule": "Trial-4 remains deferred while Trial-3 still exposes a current-canon same-family weak-sector route",
        },
        [
            helper.row("trial3_refactored_trial2_trial4_twenty_second_refresh_complete", "pass", "Trial-2 paper-side sync / Trial-4 disposition twenty-second refresh complete", 1, "The reserve/deferred ordering is refreshed after the higher-ceiling stall audit."),
            helper.row("trial3_refactored_trial2_paper_side_sync_reserve_retained_twenty_second_refresh", "pass", "Trial-2 paper-side sync reserve retained", 1, "Trial-2 paper-side sync stays unlocked reserve work while the anchor-overtake route is still open."),
            helper.row("trial3_refactored_trial4_deferred_retained_twenty_second_refresh", "pass", "Trial-4 deferred retained", 1, "Trial-4 stays deferred while Trial-3 still has an honest anchor-overtake prerequisite path."),
        ],
        {
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "trial2_paper_side_sync_state": "unlocked_reserve_retained",
            "trial4_deferred": True,
            "recommended_next_route_or_none": None if trial3_recommended_condition_satisfied else "8.7.56.317",
        },
        {
            "overall_status": "trial3_refactored_trial2_trial4_twenty_second_disposition_refreshed",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_13": False,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["8.7.56.317"],
        },
        {
            "declaration_summary": declaration["summary"],
            "prior_disposition_summary": disposition_summary,
        },
    )

    helper.write_artifact("mass_origin_v2_trial3_refactored_post_ell30_same_family_higher_ceiling_stall_source_inventory", source_inventory)
    helper.write_artifact("mass_origin_v2_trial3_refactored_post_ell30_same_family_higher_ceiling_stall_audit", audit)
    helper.write_artifact("mass_origin_v2_trial3_refactored_declaration_tenth_gate", declaration)
    helper.write_artifact("mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_twenty_second_refresh", disposition)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_refactored_post_ell30_same_family_higher_ceiling_stall_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_post_ell30_same_family_higher_ceiling_stall_audit_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_declaration_tenth_gate_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_twenty_second_refresh_metrics.json")


# 関数: CLI から post-ell30 same-family higher-ceiling stall branch を起動する。

if __name__ == "__main__":
    main()
