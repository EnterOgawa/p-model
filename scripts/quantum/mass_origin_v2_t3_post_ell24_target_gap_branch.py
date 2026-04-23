#!/usr/bin/env python3
"""
Generate Trial-3 refactored post-ell24 same-family target-gap extension artifacts for 8.7.56.297-.300.

The reopened same-family family through ell=24 is now stable, but its current
ceiling still sits below the W/Z anchor thresholds. This branch freezes whether
the next honest route is a higher-ceiling extension above the current same-
family frontier rather than another localization or pair-shape retry.
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
PREVIOUS_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_t3_post_ell24_same_family_branch.py"

PRIOR_SOURCE = OUT / "mass_origin_v2_trial3_refactored_post_ell24_same_family_reaudit_source_inventory_metrics.json"
PRIOR_REAUDIT = OUT / "mass_origin_v2_trial3_refactored_post_ell24_same_family_reaudit_metrics.json"
PRIOR_DECLARATION = OUT / "mass_origin_v2_trial3_refactored_declaration_fifth_gate_metrics.json"
PRIOR_DISPOSITION = OUT / "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_seventeenth_refresh_metrics.json"
PRIOR_RADIAL_AUDIT = OUT / "mass_origin_v2_trial3_refactored_post_ell19_radial_domain_extension_audit_metrics.json"
SOLVER_REFACTOR_WEAK = OUT / "mass_origin_v2_trial3_solver_refactor_weak_sector_reaudit_metrics.json"

W_MASS_MEV = 80369.0
Z_MASS_MEV = 91187.6
ELECTRON_MASS_MEV = 0.51099895
W_TARGET = W_MASS_MEV / ELECTRON_MASS_MEV
Z_TARGET = Z_MASS_MEV / ELECTRON_MASS_MEV
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


# 関数: post-ell24 target-gap extension branch を実行する。

def main() -> None:
    helper = load_module(HELPER_BRANCH, "trial3_post_ell24_target_gap_helper")

    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        HELPER_BRANCH,
        PREVIOUS_BRANCH,
        PRIOR_SOURCE,
        PRIOR_REAUDIT,
        PRIOR_DECLARATION,
        PRIOR_DISPOSITION,
        PRIOR_RADIAL_AUDIT,
        SOLVER_REFACTOR_WEAK,
    ):
        helper.req(path)

    status_text = helper.read_text(STATUS)
    roadmap_text = helper.read_text(ROADMAP)
    ai_context = helper.read_json(AI_CONTEXT)
    previous_branch_text = helper.read_text(PREVIOUS_BRANCH)
    prior_source = helper.read_json(PRIOR_SOURCE)
    prior_reaudit = helper.read_json(PRIOR_REAUDIT)
    prior_declaration = helper.read_json(PRIOR_DECLARATION)
    prior_disposition = helper.read_json(PRIOR_DISPOSITION)
    prior_radial_audit = helper.read_json(PRIOR_RADIAL_AUDIT)
    solver_refactor_weak = helper.read_json(SOLVER_REFACTOR_WEAK)

    prior_summary = prior_reaudit["summary"]
    best_w = prior_summary["best_w_row_or_none"]
    best_z = prior_summary["best_z_row_or_none"]
    best_pair = prior_summary["best_pair_or_none"]
    current_ceiling = max(float(best_w["ratio_value"]), float(best_z["ratio_value"]))
    current_ceiling_gap_to_w = float(W_TARGET / current_ceiling)
    current_ceiling_gap_to_z = float(Z_TARGET / current_ceiling)
    ceiling_reaches_w = bool(current_ceiling >= W_TARGET)
    ceiling_reaches_z = bool(current_ceiling >= Z_TARGET)
    best_pair_near_pass = bool(prior_summary["best_pair_near_pass"])
    mw_mz_ratio_pass = bool(best_pair and float(best_pair["mw_mz_ratio_relative_error"]) <= PASS_THRESHOLD)
    sin2_theta_w_pass = bool(best_pair and float(best_pair["sin2_theta_w_relative_error"]) <= PASS_THRESHOLD)
    anchor_threshold_dominant = bool(best_pair_near_pass and not ceiling_reaches_w and not ceiling_reaches_z)
    higher_ceiling_extension_required = bool(not ceiling_reaches_w and not ceiling_reaches_z)
    trial3_recommended_condition_satisfied = False

    inventory_targets = [
        {
            "label": "status_points_to_8_7_56_297",
            "present": "current official next step は `8.7.56.297`" in status_text,
            "evidence": {"status_markdown": helper.rel(STATUS)},
        },
        {
            "label": "roadmap_target_gap_branch_present",
            "present": "`8.7.56.297-.300` 試練3 refactored post-`ell=24` same-family target-gap extension residual branch" in roadmap_text,
            "evidence": {"roadmap_markdown": helper.rel(ROADMAP)},
        },
        {
            "label": "prior_same_family_inventory_ready",
            "present": bool(prior_source["summary"]["required_sources_present"] == prior_source["summary"]["required_sources_total"]),
            "evidence": prior_source["summary"],
        },
        {
            "label": "prior_same_family_reaudit_ready",
            "present": bool(prior_reaudit["summary"]["same_family_target_gap_extension_required"]),
            "evidence": prior_reaudit["summary"],
        },
        {
            "label": "current_ceiling_present",
            "present": current_ceiling > 0.0,
            "evidence": {
                "current_same_family_ceiling_to_electron": current_ceiling,
                "best_w_row_or_none": best_w,
                "best_z_row_or_none": best_z,
            },
        },
        {
            "label": "near_pass_pair_evidence_present",
            "present": best_pair is not None and best_pair_near_pass,
            "evidence": {
                "best_pair_or_none": best_pair,
                "best_pair_near_pass": best_pair_near_pass,
            },
        },
        {
            "label": "remaining_wz_gap_evidence_present",
            "present": current_ceiling_gap_to_w > 1.0 and current_ceiling_gap_to_z > 1.0,
            "evidence": {
                "ceiling_gap_to_w": current_ceiling_gap_to_w,
                "ceiling_gap_to_z": current_ceiling_gap_to_z,
            },
        },
        {
            "label": "previous_branch_points_to_target_gap_route",
            "present": helper.hit(previous_branch_text, "selected_residual_route = \"trial3_relaunched_refactored_post_ell24_same_family_target_gap_extension_identification\"") is not None,
            "evidence": helper.hit(previous_branch_text, "selected_residual_route = \"trial3_relaunched_refactored_post_ell24_same_family_target_gap_extension_identification\""),
        },
        {
            "label": "prior_disposition_keeps_trial2_reserve_and_trial4_deferred",
            "present": prior_disposition["summary"]["trial2_paper_side_sync_state"] == "unlocked_reserve_retained"
            and bool(prior_disposition["summary"]["trial4_deferred"]),
            "evidence": prior_disposition["summary"],
        },
    ]

    source_inventory = helper.payload(
        "8.7.56.297",
        "Trial-3 refactored post-ell24 same-family target-gap extension source inventory",
        {
            "status_markdown": helper.rel(STATUS),
            "roadmap_markdown": helper.rel(ROADMAP),
            "ai_context_json": helper.rel(AI_CONTEXT),
            "mass_origin_v2_trial3_refactored_post_ell24_same_family_reaudit_source_inventory_json": helper.rel(PRIOR_SOURCE),
            "mass_origin_v2_trial3_refactored_post_ell24_same_family_reaudit_json": helper.rel(PRIOR_REAUDIT),
            "mass_origin_v2_trial3_refactored_declaration_fifth_gate_json": helper.rel(PRIOR_DECLARATION),
            "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_seventeenth_refresh_json": helper.rel(PRIOR_DISPOSITION),
            "mass_origin_v2_trial3_refactored_post_ell19_radial_domain_extension_audit_json": helper.rel(PRIOR_RADIAL_AUDIT),
            "mass_origin_v2_trial3_solver_refactor_weak_sector_reaudit_json": helper.rel(SOLVER_REFACTOR_WEAK),
            "mass_origin_v2_t3_post_ell24_same_family_branch_py": helper.rel(PREVIOUS_BRANCH),
        },
        "Freeze the post-ell24 same-family ceiling, the best anchor and pair metrics, and the remaining W/Z threshold gap before deciding whether the next honest route is a higher-ceiling same-family extension above the current frontier.",
        {
            "inventory_rule": "the target-gap extension pack must contain the current same-family ceiling, the best anchor and best pair metrics, the remaining W/Z threshold gap, and the reserve/deferred ordering before any higher-ceiling retry is justified",
            "target_gap_rule": "if the current same-family ceiling remains below both W and Z while the pair is already near-pass, the next blocker is the missing higher-ceiling family above the current frontier",
        },
        [
            helper.row("trial3_refactored_post_ell24_target_gap_source_inventory_complete", "pass", "Trial-3 refactored post-ell24 target-gap extension source inventory complete", 1, "The target-gap extension source pack is frozen."),
            helper.row("trial3_refactored_post_ell24_current_ceiling_present", "pass" if current_ceiling > 0.0 else "reject", "current same-family ceiling present", current_ceiling, "The higher-ceiling route must start from the already-frozen same-family frontier."),
            helper.row("trial3_refactored_post_ell24_best_pair_near_pass_inventory", "pass" if best_pair_near_pass else "reject", "best pair near-pass evidence present", 1 if best_pair_near_pass else 0, "The same-family target-gap route is only honest if the pair structure is already close enough to justify focusing on the absolute ceiling."),
            helper.row("trial3_refactored_post_ell24_remaining_gap_inventory", "pass" if current_ceiling_gap_to_w > 1.0 and current_ceiling_gap_to_z > 1.0 else "reject", "remaining W/Z gap evidence present", 1 if current_ceiling_gap_to_w > 1.0 and current_ceiling_gap_to_z > 1.0 else 0, "The target-gap route must explicitly freeze the remaining anchor gap above the current ceiling."),
        ],
        {
            "required_sources_total": len(inventory_targets),
            "required_sources_present": sum(1 for item in inventory_targets if item["present"]),
            "current_same_family_ceiling_to_electron": current_ceiling,
            "w_target_to_electron": W_TARGET,
            "z_target_to_electron": Z_TARGET,
            "current_ceiling_gap_to_w": current_ceiling_gap_to_w,
            "current_ceiling_gap_to_z": current_ceiling_gap_to_z,
            "best_pair_near_pass": best_pair_near_pass,
            "status_current_step_before_branch": ai_context["current_step"],
        },
        {
            "overall_status": "trial3_refactored_post_ell24_target_gap_source_inventory_frozen",
            "source_inventory_complete": True,
            "advance_to_8_7_56_298": True,
            "next_required_artifacts": ["trial3_refactored_post_ell24_target_gap_extension_audit"],
        },
        {
            "inventory_targets": inventory_targets,
            "prior_same_family_reaudit_summary": prior_summary,
            "prior_declaration_summary": prior_declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
            "prior_radial_audit_summary": prior_radial_audit["summary"],
            "solver_refactor_weak_summary": solver_refactor_weak["summary"],
        },
    )

    selected_residual_route = None
    missing_v2_artifact = None
    if higher_ceiling_extension_required:
        selected_residual_route = "trial3_relaunched_refactored_post_ell24_same_family_higher_ceiling_extension_identification"
        missing_v2_artifact = "trial3_relaunched_refactored_post_ell24_same_family_exact_family_table_above_wz_anchor_thresholds"

    audit = helper.payload(
        "8.7.56.298",
        "Trial-3 refactored post-ell24 same-family target-gap extension audit",
        source_inventory["inputs"],
        "Audit whether the remaining post-ell24 same-family blocker is now a missing higher-ceiling family above the current frontier or whether pair-side structure still dominates the weak-sector gap.",
        {
            "higher_ceiling_rule": "if the current same-family ceiling remains below both W and Z while the pair is already near-pass, the dominant blocker is the missing higher-ceiling family above the current frontier",
            "pair_rule": "pair-side residuals become dominant only after the same-family family reaches the W/Z anchor neighborhood; before that, the anchor threshold gap remains upstream",
        },
        [
            helper.row("trial3_refactored_post_ell24_target_gap_extension_audit_complete", "pass", "Trial-3 refactored post-ell24 target-gap extension audit complete", 1, "The target-gap extension audit is frozen."),
            helper.row("trial3_refactored_post_ell24_ceiling_reaches_w_threshold", "pass" if ceiling_reaches_w else "reject", "current same-family ceiling reaches W threshold", 1 if ceiling_reaches_w else 0, "The higher-ceiling route becomes unnecessary only after the current family already reaches the W scale."),
            helper.row("trial3_refactored_post_ell24_ceiling_reaches_z_threshold", "pass" if ceiling_reaches_z else "reject", "current same-family ceiling reaches Z threshold", 1 if ceiling_reaches_z else 0, "The same-family route still needs more headroom while the current frontier remains below the Z scale."),
            helper.row("trial3_refactored_post_ell24_best_pair_near_pass", "pass" if best_pair_near_pass else "reject", "best pair remains structurally near pass", 1 if best_pair_near_pass else 0, "Near-pass pair structure supports prioritizing the absolute ceiling gap first."),
            helper.row("trial3_refactored_post_ell24_anchor_threshold_dominant_blocker", "pass" if anchor_threshold_dominant else "reject", "anchor threshold gap is the dominant remaining blocker", 1 if anchor_threshold_dominant else 0, "The dominant blocker shifts to anchor thresholds when the pair is near-pass but the family ceiling is still sub-threshold."),
            helper.row("trial3_refactored_post_ell24_higher_ceiling_extension_required", "pass" if higher_ceiling_extension_required else "reject", "higher-ceiling same-family extension required", 1 if higher_ceiling_extension_required else 0, "The next honest route is a higher-ceiling extension while both W/Z thresholds remain above the current same-family frontier."),
            helper.row("trial3_refactored_post_ell24_pair_side_residual_secondary", "pass" if (best_pair is not None and not sin2_theta_w_pass) else "reject", "pair-side residual remains secondary and unresolved", 1 if (best_pair is not None and not sin2_theta_w_pass) else 0, "The pair side is still open, but it is not the upstream blocker while the absolute ceiling remains sub-threshold."),
        ],
        {
            "trial3_recommended_condition_satisfied": trial3_recommended_condition_satisfied,
            "current_same_family_ceiling_to_electron": current_ceiling,
            "w_target_to_electron": W_TARGET,
            "z_target_to_electron": Z_TARGET,
            "current_ceiling_gap_to_w": current_ceiling_gap_to_w,
            "current_ceiling_gap_to_z": current_ceiling_gap_to_z,
            "best_pair_or_none": best_pair,
            "best_pair_near_pass": best_pair_near_pass,
            "mw_mz_ratio_pass": mw_mz_ratio_pass,
            "sin2_theta_w_pass": sin2_theta_w_pass,
            "anchor_threshold_dominant_blocker": anchor_threshold_dominant,
            "higher_ceiling_extension_required": higher_ceiling_extension_required,
        },
        {
            "overall_status": "trial3_refactored_post_ell24_target_gap_extension_audited",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_299": True,
            "next_required_artifacts": ["trial3_refactored_declaration_sixth_gate"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "prior_same_family_reaudit_summary": prior_summary,
            "prior_declaration_summary": prior_declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
        },
    )

    declaration = helper.payload(
        "8.7.56.299",
        "Trial-3 refactored declaration sixth gate",
        source_inventory["inputs"],
        "Freeze whether the post-ell24 same-family target-gap extension branch already closes Trial-3 or whether the next honest route is a higher-ceiling family extension above the current frontier.",
        {
            "closeout_rule": "Trial-3 closes only if the current same-family family already reaches the W/Z anchors and closes the pair-side observables together",
            "residual_rule": "if the pair is near-pass but the current frontier remains below both W/Z thresholds, the next residual route is a higher-ceiling family extension above the current same-family ceiling",
        },
        [
            helper.row("trial3_refactored_declaration_sixth_gate_complete", "pass", "Trial-3 refactored declaration sixth gate complete", 1, "The sixth declaration gate is frozen."),
            helper.row("trial3_refactored_sixth_branch_closeable", "pass" if trial3_recommended_condition_satisfied else "reject", "refactored Trial-3 branch closeable after post-ell24 target-gap extension audit", 1 if trial3_recommended_condition_satisfied else 0, "The branch closes only if the current same-family frontier already closes the weak-sector pack."),
            helper.row("trial3_refactored_sixth_residual_route_required", "reject" if trial3_recommended_condition_satisfied else "pass", "refactored Trial-3 residual route required after target-gap extension audit", 0 if trial3_recommended_condition_satisfied else 1, "A new residual route is still required while the current ceiling remains below the W/Z thresholds."),
            helper.row("trial3_refactored_execute_trial2_paper_sync_now_sixth_gate", "reject", "execute Trial-2 paper-side sync now after target-gap extension audit", 0, "Trial-2 paper-side sync remains reserve work while the higher-ceiling Trial-3 route is still open."),
        ],
        {
            "trial3_recommended_condition_satisfied": trial3_recommended_condition_satisfied,
            "trial3_current_branch_closeable": trial3_recommended_condition_satisfied,
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "recommended_next_route_or_none": None if trial3_recommended_condition_satisfied else "8.7.56.301",
        },
        {
            "overall_status": "trial3_refactored_declaration_sixth_gate_frozen",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_300": True,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["trial3_refactored_trial2_paper_sync_trial4_disposition_eighteenth_refresh"],
        },
        {
            "audit_summary": audit["summary"],
            "prior_declaration_summary": prior_declaration["summary"],
            "best_pair_or_none": best_pair,
        },
    )

    disposition = helper.payload(
        "8.7.56.300",
        "Trial-2 paper-side sync / Trial-4 disposition eighteenth refresh",
        source_inventory["inputs"],
        "Refresh the reserve/deferred ordering after the post-ell24 target-gap extension audit and freeze the next official residual route.",
        {
            "trial2_rule": "Trial-2 paper-side sync stays unlocked reserve retained while the refactored Trial-3 route still has an honest higher-ceiling extension path",
            "trial4_rule": "Trial-4 remains deferred while Trial-3 still exposes a current-canon weak-sector frontier",
        },
        [
            helper.row("trial3_refactored_trial2_trial4_eighteenth_refresh_complete", "pass", "Trial-2 paper-side sync / Trial-4 disposition eighteenth refresh complete", 1, "The reserve/deferred ordering is refreshed after the target-gap extension audit."),
            helper.row("trial3_refactored_trial2_paper_side_sync_reserve_retained_eighteenth_refresh", "pass", "Trial-2 paper-side sync reserve retained", 1, "Trial-2 paper-side sync stays unlocked reserve work while the higher-ceiling Trial-3 route is still open."),
            helper.row("trial3_refactored_trial4_deferred_retained_eighteenth_refresh", "pass", "Trial-4 deferred retained", 1, "Trial-4 stays deferred while Trial-3 still has an honest higher-ceiling extension path."),
        ],
        {
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "trial2_paper_side_sync_state": "unlocked_reserve_retained",
            "trial4_deferred": True,
            "recommended_next_route_or_none": None if trial3_recommended_condition_satisfied else "8.7.56.301",
        },
        {
            "overall_status": "trial3_refactored_trial2_trial4_eighteenth_disposition_refreshed",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_13": False,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["8.7.56.301"],
        },
        {
            "declaration_summary": declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
            "solver_refactor_weak_summary": solver_refactor_weak["summary"],
        },
    )

    helper.write_artifact("mass_origin_v2_trial3_refactored_post_ell24_same_family_target_gap_extension_source_inventory", source_inventory)
    helper.write_artifact("mass_origin_v2_trial3_refactored_post_ell24_same_family_target_gap_extension_audit", audit)
    helper.write_artifact("mass_origin_v2_trial3_refactored_declaration_sixth_gate", declaration)
    helper.write_artifact("mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_eighteenth_refresh", disposition)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_refactored_post_ell24_same_family_target_gap_extension_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_post_ell24_same_family_target_gap_extension_audit_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_declaration_sixth_gate_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_eighteenth_refresh_metrics.json")


# 関数: CLI から post-ell24 target-gap branch を起動する。

if __name__ == "__main__":
    main()
