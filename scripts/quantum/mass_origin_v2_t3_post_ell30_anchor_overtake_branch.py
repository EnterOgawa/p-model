#!/usr/bin/env python3
"""
Generate Trial-3 refactored post-ell30 same-family anchor-overtake prerequisite
artifacts for 8.7.56.317-.320.

The higher-ceiling stall audit already fixed the honest qualitative state:
the ell=25..30 family exists, but the incumbent ell=22 anchor still pins the
same-family ceiling. The next honest question is narrower: does the blocker now
collapse to the missing condition that would let post-ell30 states actually
displace the incumbent anchor?
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
PREVIOUS_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_t3_post_ell30_higher_ceiling_stall_branch.py"

PRIOR_SOURCE = OUT / "mass_origin_v2_trial3_refactored_post_ell30_same_family_higher_ceiling_stall_source_inventory_metrics.json"
PRIOR_AUDIT = OUT / "mass_origin_v2_trial3_refactored_post_ell30_same_family_higher_ceiling_stall_audit_metrics.json"
PRIOR_DECLARATION = OUT / "mass_origin_v2_trial3_refactored_declaration_tenth_gate_metrics.json"
PRIOR_DISPOSITION = OUT / "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_twenty_second_refresh_metrics.json"
PRIOR_REAUDIT = OUT / "mass_origin_v2_trial3_refactored_post_ell30_same_family_reaudit_metrics.json"


# 関数: helper branch を動的 import する。
def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 関数: post-ell30 same-family anchor-overtake prerequisite branch を実行する。

def main() -> None:
    helper = load_module(HELPER_BRANCH, "trial3_post_ell30_anchor_overtake_helper")

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
        PRIOR_REAUDIT,
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
    prior_reaudit = helper.read_json(PRIOR_REAUDIT)

    source_summary = prior_source["summary"]
    audit_summary = prior_audit["summary"]
    declaration_summary = prior_declaration["summary"]
    disposition_summary = prior_disposition["summary"]
    reaudit_summary = prior_reaudit["summary"]

    localized_ell_values = [int(value) for value in audit_summary["localized_ell_values"]]
    frontier_reopened_to_30 = bool(localized_ell_values and max(localized_ell_values) >= 30)
    current_ceiling = float(audit_summary["current_ceiling_to_electron"])
    incumbent_anchor = audit_summary["best_w_row_or_none"]
    best_z = audit_summary["best_z_row_or_none"]
    best_pair = audit_summary["best_pair_or_none"]
    incumbent_anchor_pinned = bool(audit_summary["best_anchor_pinned_to_ell22"])
    best_pair_unchanged = bool(audit_summary["best_pair_unchanged"])
    ceiling_stalled = bool(audit_summary["ceiling_stalled"])
    same_family_displacement_missing = bool(
        frontier_reopened_to_30
        and incumbent_anchor_pinned
        and ceiling_stalled
        and best_pair_unchanged
    )
    trial3_recommended_condition_satisfied = bool(audit_summary["trial3_recommended_condition_satisfied"])

    inventory_targets = [
        {
            "label": "status_points_to_8_7_56_317",
            "present": "current official next step は `8.7.56.317`" in status_text,
            "evidence": {"status_markdown": helper.rel(STATUS)},
        },
        {
            "label": "roadmap_anchor_overtake_branch_present",
            "present": "`8.7.56.317-.320` 試練3 refactored post-`ell=30` same-family anchor-overtake prerequisite residual branch" in roadmap_text,
            "evidence": {"roadmap_markdown": helper.rel(ROADMAP)},
        },
        {
            "label": "prior_higher_ceiling_stall_pack_present",
            "present": bool(source_summary["required_sources_present"] == source_summary["required_sources_total"]),
            "evidence": source_summary,
        },
        {
            "label": "frontier_reopened_to_ell30_present",
            "present": frontier_reopened_to_30,
            "evidence": {"localized_ell_values": localized_ell_values},
        },
        {
            "label": "incumbent_anchor_pack_present",
            "present": incumbent_anchor is not None and incumbent_anchor_pinned,
            "evidence": {"best_w_row_or_none": incumbent_anchor},
        },
        {
            "label": "same_family_ceiling_stall_present",
            "present": ceiling_stalled,
            "evidence": {
                "current_ceiling_to_electron": current_ceiling,
                "pre_frontier_ceiling_to_electron": reaudit_summary["pre_frontier_ceiling_to_electron"],
            },
        },
        {
            "label": "best_pair_pack_unchanged",
            "present": best_pair is not None and best_pair_unchanged,
            "evidence": {"best_pair_or_none": best_pair},
        },
        {
            "label": "previous_branch_points_to_anchor_overtake_prerequisite",
            "present": helper.hit(previous_branch_text, "selected_residual_route = \"trial3_relaunched_refactored_post_ell30_same_family_anchor_overtake_prerequisite_identification\"") is not None,
            "evidence": helper.hit(previous_branch_text, "selected_residual_route = \"trial3_relaunched_refactored_post_ell30_same_family_anchor_overtake_prerequisite_identification\""),
        },
        {
            "label": "prior_disposition_keeps_trial2_reserve_and_trial4_deferred",
            "present": disposition_summary["trial2_paper_side_sync_state"] == "unlocked_reserve_retained"
            and bool(disposition_summary["trial4_deferred"]),
            "evidence": disposition_summary,
        },
    ]

    source_inventory = helper.payload(
        "8.7.56.317",
        "Trial-3 refactored post-ell30 same-family anchor-overtake prerequisite source inventory",
        {
            "status_markdown": helper.rel(STATUS),
            "roadmap_markdown": helper.rel(ROADMAP),
            "ai_context_json": helper.rel(AI_CONTEXT),
            "mass_origin_v2_trial3_refactored_post_ell30_same_family_higher_ceiling_stall_source_inventory_json": helper.rel(PRIOR_SOURCE),
            "mass_origin_v2_trial3_refactored_post_ell30_same_family_higher_ceiling_stall_audit_json": helper.rel(PRIOR_AUDIT),
            "mass_origin_v2_trial3_refactored_declaration_tenth_gate_json": helper.rel(PRIOR_DECLARATION),
            "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_twenty_second_refresh_json": helper.rel(PRIOR_DISPOSITION),
            "mass_origin_v2_trial3_refactored_post_ell30_same_family_reaudit_json": helper.rel(PRIOR_REAUDIT),
            "mass_origin_v2_t3_post_ell30_higher_ceiling_stall_branch_py": helper.rel(PREVIOUS_BRANCH),
        },
        "Freeze the incumbent-anchor pack, the stalled same-family ceiling, and the post-ell30 family evidence before deciding whether the honest blocker is now the missing condition for anchor displacement itself.",
        {
            "inventory_rule": "the anchor-overtake prerequisite pack must include the incumbent ell=22 anchor, the stalled same-family ceiling, the reopened post-ell30 family, and the unchanged pair-side diagnostics in one place",
            "displacement_rule": "if the post-ell30 family exists but the incumbent ell=22 anchor still defines the ceiling, the next honest blocker is the missing condition for anchor displacement rather than another ceiling-stall generality",
        },
        [
            helper.row("trial3_refactored_post_ell30_anchor_overtake_source_inventory_complete", "pass", "Trial-3 refactored post-ell30 anchor-overtake prerequisite source inventory complete", 1, "The anchor-overtake prerequisite evidence pack is frozen."),
            helper.row("trial3_refactored_post_ell30_frontier_present", "pass" if frontier_reopened_to_30 else "reject", "reopened post-ell30 family present", 1 if frontier_reopened_to_30 else 0, "The displacement question is only honest if the post-ell30 family remains present."),
            helper.row("trial3_refactored_post_ell30_incumbent_anchor_present", "pass" if incumbent_anchor_pinned else "reject", "incumbent ell=22 anchor still present", 1 if incumbent_anchor_pinned else 0, "The next blocker only shrinks if the incumbent anchor remains the controlling state."),
            helper.row("trial3_refactored_post_ell30_ceiling_stall_present", "pass" if ceiling_stalled else "reject", "same-family ceiling stall present", 1 if ceiling_stalled else 0, "The displacement route only opens if the ceiling remains pinned to the incumbent state."),
        ],
        {
            "required_sources_total": len(inventory_targets),
            "required_sources_present": sum(1 for item in inventory_targets if item["present"]),
            "localized_ell_values": localized_ell_values,
            "current_ceiling_to_electron": current_ceiling,
            "best_w_row_or_none": incumbent_anchor,
            "best_z_row_or_none": best_z,
            "best_pair_or_none": best_pair,
            "w_gap_factor_or_none": audit_summary["w_gap_factor_or_none"],
            "z_gap_factor_or_none": audit_summary["z_gap_factor_or_none"],
            "status_current_step_before_branch": ai_context["current_step"],
        },
        {
            "overall_status": "trial3_refactored_post_ell30_anchor_overtake_source_inventory_frozen",
            "source_inventory_complete": True,
            "advance_to_8_7_56_318": True,
            "next_required_artifacts": ["trial3_refactored_post_ell30_anchor_overtake_prerequisite_audit"],
        },
        {
            "inventory_targets": inventory_targets,
            "prior_source_summary": source_summary,
            "prior_audit_summary": audit_summary,
            "prior_declaration_summary": declaration_summary,
            "prior_disposition_summary": disposition_summary,
        },
    )

    selected_residual_route = None
    missing_v2_artifact = None
    if not trial3_recommended_condition_satisfied:
        selected_residual_route = "trial3_relaunched_refactored_post_ell30_same_family_incumbent_anchor_displacement_identification"
        missing_v2_artifact = "trial3_relaunched_refactored_post_ell30_same_family_incumbent_anchor_displacement_pack"

    audit = helper.payload(
        "8.7.56.318",
        "Trial-3 refactored post-ell30 same-family anchor-overtake prerequisite audit",
        source_inventory["inputs"],
        "Audit whether the post-ell30 family failure now honestly collapses to incumbent-anchor displacement itself or whether a broader prerequisite family is still needed.",
        {
            "displacement_rule": "if the incumbent ell=22 anchor still defines the same-family ceiling while the post-ell30 family exists, the honest next blocker is incumbent-anchor displacement",
            "broad_prerequisite_rule": "if the reopened family, stalled ceiling, and unchanged pair remain simultaneously true, the blocker has narrowed past general prerequisites into the displacement condition itself",
        },
        [
            helper.row("trial3_refactored_post_ell30_anchor_overtake_audit_complete", "pass", "Trial-3 refactored post-ell30 anchor-overtake prerequisite audit complete", 1, "The anchor-overtake prerequisite audit is frozen."),
            helper.row("trial3_refactored_post_ell30_frontier_exists", "pass" if frontier_reopened_to_30 else "reject", "post-ell30 family exists", 1 if frontier_reopened_to_30 else 0, "The displacement route is only honest if the post-ell30 family remains present."),
            helper.row("trial3_refactored_post_ell30_incumbent_anchor_still_defines_ceiling", "pass" if incumbent_anchor_pinned and ceiling_stalled else "reject", "incumbent anchor still defines same-family ceiling", 1 if incumbent_anchor_pinned and ceiling_stalled else 0, "The blocker narrows only if the incumbent anchor still sets the ceiling."),
            helper.row("trial3_refactored_post_ell30_pair_unchanged", "pass" if best_pair_unchanged else "reject", "best pair remains unchanged", 1 if best_pair_unchanged else 0, "The pair-side pack must stay pinned while the anchor continues to dominate."),
            helper.row("trial3_refactored_post_ell30_same_family_displacement_missing", "pass" if same_family_displacement_missing else "reject", "same-family incumbent-anchor displacement missing", 1 if same_family_displacement_missing else 0, "The honest next blocker is the missing condition that would let post-ell30 states displace the incumbent anchor."),
        ],
        {
            "trial3_recommended_condition_satisfied": trial3_recommended_condition_satisfied,
            "localized_ell_values": localized_ell_values,
            "current_ceiling_to_electron": current_ceiling,
            "best_w_row_or_none": incumbent_anchor,
            "best_z_row_or_none": best_z,
            "best_pair_or_none": best_pair,
            "w_gap_factor_or_none": audit_summary["w_gap_factor_or_none"],
            "z_gap_factor_or_none": audit_summary["z_gap_factor_or_none"],
            "incumbent_anchor_pinned": incumbent_anchor_pinned,
            "ceiling_stalled": ceiling_stalled,
            "best_pair_unchanged": best_pair_unchanged,
            "same_family_displacement_missing": same_family_displacement_missing,
        },
        {
            "overall_status": "trial3_refactored_post_ell30_anchor_overtake_audited",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_319": True,
            "next_required_artifacts": ["trial3_refactored_declaration_eleventh_gate"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "prior_audit_summary": audit_summary,
            "prior_declaration_summary": declaration_summary,
            "prior_disposition_summary": disposition_summary,
        },
    )

    declaration = helper.payload(
        "8.7.56.319",
        "Trial-3 refactored declaration eleventh gate",
        source_inventory["inputs"],
        "Freeze whether the post-ell30 anchor-overtake prerequisite branch already closes Trial-3 or whether the next honest route is incumbent-anchor displacement identification.",
        {
            "closeout_rule": "Trial-3 closes only if the current same-family family now displaces the incumbent anchor and closes both anchors plus the pair-side observables together",
            "residual_rule": "if the incumbent ell=22 anchor still defines the ceiling, the next honest blocker is incumbent-anchor displacement identification",
        },
        [
            helper.row("trial3_refactored_declaration_eleventh_gate_complete", "pass", "Trial-3 refactored declaration eleventh gate complete", 1, "The eleventh declaration gate is frozen."),
            helper.row("trial3_refactored_eleventh_branch_closeable", "pass" if trial3_recommended_condition_satisfied else "reject", "refactored Trial-3 branch closeable after anchor-overtake prerequisite audit", 1 if trial3_recommended_condition_satisfied else 0, "The branch closes only if the current same-family family already displaces the incumbent anchor and closes the weak-sector pack."),
            helper.row("trial3_refactored_eleventh_residual_route_required", "reject" if trial3_recommended_condition_satisfied else "pass", "refactored Trial-3 residual route required after anchor-overtake prerequisite audit", 0 if trial3_recommended_condition_satisfied else 1, "A new residual route is still required while incumbent-anchor displacement remains absent."),
            helper.row("trial3_refactored_execute_trial2_paper_sync_now_eleventh_gate", "reject", "execute Trial-2 paper-side sync now after anchor-overtake prerequisite audit", 0, "Trial-2 paper-side sync remains reserve work while Trial-3 still has an honest incumbent-anchor displacement path."),
        ],
        {
            "trial3_recommended_condition_satisfied": trial3_recommended_condition_satisfied,
            "trial3_current_branch_closeable": trial3_recommended_condition_satisfied,
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "recommended_next_route_or_none": None if trial3_recommended_condition_satisfied else "8.7.56.321",
        },
        {
            "overall_status": "trial3_refactored_declaration_eleventh_gate_frozen",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_320": True,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["trial3_refactored_trial2_paper_sync_trial4_disposition_twenty_third_refresh"],
        },
        {
            "audit_summary": audit["summary"],
            "prior_declaration_summary": declaration_summary,
            "best_pair_or_none": best_pair,
        },
    )

    disposition = helper.payload(
        "8.7.56.320",
        "Trial-2 paper-side sync / Trial-4 disposition twenty-third refresh",
        source_inventory["inputs"],
        "Refresh the reserve/deferred ordering after the post-ell30 anchor-overtake prerequisite audit and freeze the next official incumbent-anchor displacement route.",
        {
            "trial2_rule": "Trial-2 paper-side sync stays unlocked reserve retained while the refactored Trial-3 route still has an honest incumbent-anchor displacement path",
            "trial4_rule": "Trial-4 remains deferred while Trial-3 still exposes a current-canon same-family weak-sector route",
        },
        [
            helper.row("trial3_refactored_trial2_trial4_twenty_third_refresh_complete", "pass", "Trial-2 paper-side sync / Trial-4 disposition twenty-third refresh complete", 1, "The reserve/deferred ordering is refreshed after the anchor-overtake prerequisite audit."),
            helper.row("trial3_refactored_trial2_paper_side_sync_reserve_retained_twenty_third_refresh", "pass", "Trial-2 paper-side sync reserve retained", 1, "Trial-2 paper-side sync stays unlocked reserve work while the displacement route is still open."),
            helper.row("trial3_refactored_trial4_deferred_retained_twenty_third_refresh", "pass", "Trial-4 deferred retained", 1, "Trial-4 stays deferred while Trial-3 still has an honest incumbent-anchor displacement path."),
        ],
        {
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "trial2_paper_side_sync_state": "unlocked_reserve_retained",
            "trial4_deferred": True,
            "recommended_next_route_or_none": None if trial3_recommended_condition_satisfied else "8.7.56.321",
        },
        {
            "overall_status": "trial3_refactored_trial2_trial4_twenty_third_disposition_refreshed",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_13": False,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["8.7.56.321"],
        },
        {
            "declaration_summary": declaration["summary"],
            "prior_disposition_summary": disposition_summary,
        },
    )

    helper.write_artifact("mass_origin_v2_trial3_refactored_post_ell30_same_family_anchor_overtake_prerequisite_source_inventory", source_inventory)
    helper.write_artifact("mass_origin_v2_trial3_refactored_post_ell30_same_family_anchor_overtake_prerequisite_audit", audit)
    helper.write_artifact("mass_origin_v2_trial3_refactored_declaration_eleventh_gate", declaration)
    helper.write_artifact("mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_twenty_third_refresh", disposition)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_refactored_post_ell30_same_family_anchor_overtake_prerequisite_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_post_ell30_same_family_anchor_overtake_prerequisite_audit_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_declaration_eleventh_gate_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_twenty_third_refresh_metrics.json")


# 関数: CLI から post-ell30 same-family anchor-overtake prerequisite branch を起動する。

if __name__ == "__main__":
    main()
