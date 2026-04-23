#!/usr/bin/env python3
"""
Generate Trial-3 refactored post-ell30 same-family incumbent-anchor
displacement artifacts for 8.7.56.321-.324.

The anchor-overtake prerequisite audit already fixed the honest qualitative
state: the ell=25..30 family exists, but the incumbent ell=22 anchor still
pins the same-family ceiling. The next honest question is narrower: does the
current blocker now collapse to the missing witness that would actually show a
post-ell30 state displacing that incumbent anchor?
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
PREVIOUS_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_t3_post_ell30_anchor_overtake_branch.py"

PRIOR_SOURCE = OUT / "mass_origin_v2_trial3_refactored_post_ell30_same_family_anchor_overtake_prerequisite_source_inventory_metrics.json"
PRIOR_AUDIT = OUT / "mass_origin_v2_trial3_refactored_post_ell30_same_family_anchor_overtake_prerequisite_audit_metrics.json"
PRIOR_DECLARATION = OUT / "mass_origin_v2_trial3_refactored_declaration_eleventh_gate_metrics.json"
PRIOR_DISPOSITION = OUT / "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_twenty_third_refresh_metrics.json"


# 関数: helper branch を動的 import する。
def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 関数: post-ell30 same-family incumbent-anchor displacement branch を実行する。

def main() -> None:
    helper = load_module(HELPER_BRANCH, "trial3_post_ell30_anchor_displacement_helper")

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

    source_summary = prior_source["summary"]
    audit_summary = prior_audit["summary"]
    declaration_summary = prior_declaration["summary"]
    disposition_summary = prior_disposition["summary"]

    localized_ell_values = [int(value) for value in audit_summary["localized_ell_values"]]
    frontier_reopened_to_30 = bool(localized_ell_values and max(localized_ell_values) >= 30)
    current_ceiling = float(audit_summary["current_ceiling_to_electron"])
    incumbent_anchor = audit_summary["best_w_row_or_none"]
    best_z = audit_summary["best_z_row_or_none"]
    best_pair = audit_summary["best_pair_or_none"]
    incumbent_anchor_pinned = bool(audit_summary["incumbent_anchor_pinned"])
    ceiling_stalled = bool(audit_summary["ceiling_stalled"])
    best_pair_unchanged = bool(audit_summary["best_pair_unchanged"])
    same_family_displacement_missing = bool(audit_summary["same_family_displacement_missing"])
    displacement_witness_missing = bool(
        frontier_reopened_to_30
        and incumbent_anchor_pinned
        and ceiling_stalled
        and best_pair_unchanged
        and same_family_displacement_missing
    )
    trial3_recommended_condition_satisfied = bool(audit_summary["trial3_recommended_condition_satisfied"])

    inventory_targets = [
        {
            "label": "status_points_to_8_7_56_321",
            "present": "current official next step は `8.7.56.321`" in status_text,
            "evidence": {"status_markdown": helper.rel(STATUS)},
        },
        {
            "label": "roadmap_incumbent_anchor_displacement_branch_present",
            "present": "`8.7.56.321-.324` 試練3 refactored post-`ell=30` same-family incumbent-anchor displacement residual branch" in roadmap_text,
            "evidence": {"roadmap_markdown": helper.rel(ROADMAP)},
        },
        {
            "label": "prior_anchor_overtake_pack_present",
            "present": bool(source_summary["required_sources_present"] == source_summary["required_sources_total"]),
            "evidence": source_summary,
        },
        {
            "label": "frontier_reopened_to_ell30_present",
            "present": frontier_reopened_to_30,
            "evidence": {"localized_ell_values": localized_ell_values},
        },
        {
            "label": "incumbent_anchor_still_pinned_present",
            "present": incumbent_anchor is not None and incumbent_anchor_pinned,
            "evidence": {"best_w_row_or_none": incumbent_anchor},
        },
        {
            "label": "same_family_ceiling_stall_present",
            "present": ceiling_stalled,
            "evidence": {"current_ceiling_to_electron": current_ceiling},
        },
        {
            "label": "best_pair_pack_unchanged",
            "present": best_pair is not None and best_pair_unchanged,
            "evidence": {"best_pair_or_none": best_pair},
        },
        {
            "label": "previous_branch_points_to_incumbent_anchor_displacement",
            "present": helper.hit(previous_branch_text, "selected_residual_route = \"trial3_relaunched_refactored_post_ell30_same_family_incumbent_anchor_displacement_identification\"") is not None,
            "evidence": helper.hit(previous_branch_text, "selected_residual_route = \"trial3_relaunched_refactored_post_ell30_same_family_incumbent_anchor_displacement_identification\""),
        },
        {
            "label": "prior_disposition_keeps_trial2_reserve_and_trial4_deferred",
            "present": disposition_summary["trial2_paper_side_sync_state"] == "unlocked_reserve_retained"
            and bool(disposition_summary["trial4_deferred"]),
            "evidence": disposition_summary,
        },
    ]

    source_inventory = helper.payload(
        "8.7.56.321",
        "Trial-3 refactored post-ell30 same-family incumbent-anchor displacement source inventory",
        {
            "status_markdown": helper.rel(STATUS),
            "roadmap_markdown": helper.rel(ROADMAP),
            "ai_context_json": helper.rel(AI_CONTEXT),
            "mass_origin_v2_trial3_refactored_post_ell30_same_family_anchor_overtake_prerequisite_source_inventory_json": helper.rel(PRIOR_SOURCE),
            "mass_origin_v2_trial3_refactored_post_ell30_same_family_anchor_overtake_prerequisite_audit_json": helper.rel(PRIOR_AUDIT),
            "mass_origin_v2_trial3_refactored_declaration_eleventh_gate_json": helper.rel(PRIOR_DECLARATION),
            "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_twenty_third_refresh_json": helper.rel(PRIOR_DISPOSITION),
            "mass_origin_v2_t3_post_ell30_anchor_overtake_branch_py": helper.rel(PREVIOUS_BRANCH),
        },
        "Freeze the incumbent-anchor displacement pack before deciding whether the honest blocker now collapses to the missing witness that would show a post-ell30 state overtaking the incumbent ell=22 anchor.",
        {
            "displacement_rule": "if the post-ell30 family exists but the incumbent ell=22 anchor still defines the same-family ceiling, the honest blocker is incumbent-anchor displacement rather than another broad frontier question",
            "witness_rule": "if the ceiling, anchor, and pair all remain pinned, the next honest blocker is the missing witness that any post-ell30 state actually displaces the incumbent anchor",
        },
        [
            helper.row("trial3_refactored_post_ell30_incumbent_anchor_displacement_source_inventory_complete", "pass", "Trial-3 refactored post-ell30 incumbent-anchor displacement source inventory complete", 1, "The incumbent-anchor displacement evidence pack is frozen."),
            helper.row("trial3_refactored_post_ell30_frontier_present", "pass" if frontier_reopened_to_30 else "reject", "reopened post-ell30 family present", 1 if frontier_reopened_to_30 else 0, "The displacement question is only honest if the post-ell30 family remains present."),
            helper.row("trial3_refactored_post_ell30_incumbent_anchor_present", "pass" if incumbent_anchor_pinned else "reject", "incumbent ell=22 anchor still present", 1 if incumbent_anchor_pinned else 0, "The blocker only narrows if the incumbent anchor still controls the ceiling."),
            helper.row("trial3_refactored_post_ell30_ceiling_stall_present", "pass" if ceiling_stalled else "reject", "same-family ceiling stall present", 1 if ceiling_stalled else 0, "The displacement branch requires the same-family ceiling to remain pinned."),
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
            "overall_status": "trial3_refactored_post_ell30_incumbent_anchor_displacement_source_inventory_frozen",
            "source_inventory_complete": True,
            "advance_to_8_7_56_322": True,
            "next_required_artifacts": ["trial3_refactored_post_ell30_incumbent_anchor_displacement_audit"],
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
        selected_residual_route = "trial3_relaunched_refactored_post_ell30_same_family_incumbent_anchor_displacement_witness_identification"
        missing_v2_artifact = "trial3_relaunched_refactored_post_ell30_same_family_incumbent_anchor_displacement_witness_pack"

    audit = helper.payload(
        "8.7.56.322",
        "Trial-3 refactored post-ell30 same-family incumbent-anchor displacement audit",
        source_inventory["inputs"],
        "Audit whether the post-ell30 family failure now honestly collapses to the missing witness of incumbent-anchor displacement itself.",
        {
            "displacement_rule": "if the incumbent ell=22 anchor still defines the same-family ceiling while the post-ell30 family exists, the honest next blocker is incumbent-anchor displacement itself",
            "witness_rule": "if no post-ell30 state changes the incumbent anchor, the blocker shrinks further to the missing displacement witness rather than another generic displacement family",
        },
        [
            helper.row("trial3_refactored_post_ell30_incumbent_anchor_displacement_audit_complete", "pass", "Trial-3 refactored post-ell30 incumbent-anchor displacement audit complete", 1, "The incumbent-anchor displacement audit is frozen."),
            helper.row("trial3_refactored_post_ell30_frontier_exists", "pass" if frontier_reopened_to_30 else "reject", "post-ell30 family exists", 1 if frontier_reopened_to_30 else 0, "The witness route is only honest if the post-ell30 family remains present."),
            helper.row("trial3_refactored_post_ell30_incumbent_anchor_still_defines_ceiling", "pass" if incumbent_anchor_pinned and ceiling_stalled else "reject", "incumbent anchor still defines same-family ceiling", 1 if incumbent_anchor_pinned and ceiling_stalled else 0, "The blocker narrows only if the incumbent anchor still sets the ceiling."),
            helper.row("trial3_refactored_post_ell30_pair_unchanged", "pass" if best_pair_unchanged else "reject", "best pair remains unchanged", 1 if best_pair_unchanged else 0, "The pair-side pack must stay pinned while the anchor continues to dominate."),
            helper.row("trial3_refactored_post_ell30_displacement_witness_missing", "pass" if displacement_witness_missing else "reject", "incumbent-anchor displacement witness missing", 1 if displacement_witness_missing else 0, "The honest next blocker is the missing witness that a post-ell30 state actually displaces the incumbent anchor."),
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
            "displacement_witness_missing": displacement_witness_missing,
        },
        {
            "overall_status": "trial3_refactored_post_ell30_incumbent_anchor_displacement_audited",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_323": True,
            "next_required_artifacts": ["trial3_refactored_declaration_twelfth_gate"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "prior_audit_summary": audit_summary,
            "prior_declaration_summary": declaration_summary,
            "prior_disposition_summary": disposition_summary,
        },
    )

    declaration = helper.payload(
        "8.7.56.323",
        "Trial-3 refactored declaration twelfth gate",
        source_inventory["inputs"],
        "Freeze whether the incumbent-anchor displacement branch already closes Trial-3 or whether the next honest route is the missing displacement-witness investigation.",
        {
            "closeout_rule": "Trial-3 closes only if the current same-family family now displaces the incumbent anchor and closes both anchors plus pair-side observables together",
            "residual_rule": "if the incumbent ell=22 anchor still defines the ceiling, the next honest blocker is the missing displacement witness itself",
        },
        [
            helper.row("trial3_refactored_declaration_twelfth_gate_complete", "pass", "Trial-3 refactored declaration twelfth gate complete", 1, "The twelfth declaration gate is frozen."),
            helper.row("trial3_refactored_twelfth_branch_closeable", "pass" if trial3_recommended_condition_satisfied else "reject", "refactored Trial-3 branch closeable after displacement audit", 1 if trial3_recommended_condition_satisfied else 0, "The branch closes only if the current same-family family already displaces the incumbent anchor and closes the weak-sector pack."),
            helper.row("trial3_refactored_twelfth_residual_route_required", "reject" if trial3_recommended_condition_satisfied else "pass", "refactored Trial-3 residual route required after displacement audit", 0 if trial3_recommended_condition_satisfied else 1, "A new residual route is still required while the displacement witness remains absent."),
            helper.row("trial3_refactored_execute_trial2_paper_sync_now_twelfth_gate", "reject", "execute Trial-2 paper-side sync now after displacement audit", 0, "Trial-2 paper-side sync remains reserve work while Trial-3 still has an honest displacement-witness path."),
        ],
        {
            "trial3_recommended_condition_satisfied": trial3_recommended_condition_satisfied,
            "trial3_current_branch_closeable": trial3_recommended_condition_satisfied,
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "recommended_next_route_or_none": None if trial3_recommended_condition_satisfied else "8.7.56.325",
        },
        {
            "overall_status": "trial3_refactored_declaration_twelfth_gate_frozen",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_324": True,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["trial3_refactored_trial2_paper_sync_trial4_disposition_twenty_fourth_refresh"],
        },
        {
            "audit_summary": audit["summary"],
            "prior_declaration_summary": declaration_summary,
            "best_pair_or_none": best_pair,
        },
    )

    disposition = helper.payload(
        "8.7.56.324",
        "Trial-2 paper-side sync / Trial-4 disposition twenty-fourth refresh",
        source_inventory["inputs"],
        "Refresh the reserve/deferred ordering after the incumbent-anchor displacement audit and freeze the next official displacement-witness route.",
        {
            "trial2_rule": "Trial-2 paper-side sync stays unlocked reserve retained while the refactored Trial-3 route still has an honest displacement-witness path",
            "trial4_rule": "Trial-4 remains deferred while Trial-3 still exposes a current-canon same-family weak-sector route",
        },
        [
            helper.row("trial3_refactored_trial2_trial4_twenty_fourth_refresh_complete", "pass", "Trial-2 paper-side sync / Trial-4 disposition twenty-fourth refresh complete", 1, "The reserve/deferred ordering is refreshed after the displacement audit."),
            helper.row("trial3_refactored_trial2_paper_side_sync_reserve_retained_twenty_fourth_refresh", "pass", "Trial-2 paper-side sync reserve retained", 1, "Trial-2 paper-side sync stays unlocked reserve work while the displacement-witness route is still open."),
            helper.row("trial3_refactored_trial4_deferred_retained_twenty_fourth_refresh", "pass", "Trial-4 deferred retained", 1, "Trial-4 stays deferred while Trial-3 still has an honest displacement-witness path."),
        ],
        {
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "trial2_paper_side_sync_state": "unlocked_reserve_retained",
            "trial4_deferred": True,
            "recommended_next_route_or_none": None if trial3_recommended_condition_satisfied else "8.7.56.325",
        },
        {
            "overall_status": "trial3_refactored_trial2_trial4_twenty_fourth_disposition_refreshed",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_13": False,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["8.7.56.325"],
        },
        {
            "declaration_summary": declaration["summary"],
            "prior_disposition_summary": disposition_summary,
        },
    )

    helper.write_artifact("mass_origin_v2_trial3_refactored_post_ell30_same_family_incumbent_anchor_displacement_source_inventory", source_inventory)
    helper.write_artifact("mass_origin_v2_trial3_refactored_post_ell30_same_family_incumbent_anchor_displacement_audit", audit)
    helper.write_artifact("mass_origin_v2_trial3_refactored_declaration_twelfth_gate", declaration)
    helper.write_artifact("mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_twenty_fourth_refresh", disposition)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_refactored_post_ell30_same_family_incumbent_anchor_displacement_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_post_ell30_same_family_incumbent_anchor_displacement_audit_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_declaration_twelfth_gate_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_twenty_fourth_refresh_metrics.json")


# 関数: CLI から post-ell30 same-family incumbent-anchor displacement branch を起動する。

if __name__ == "__main__":
    main()
