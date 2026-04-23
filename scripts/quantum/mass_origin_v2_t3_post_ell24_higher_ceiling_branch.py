#!/usr/bin/env python3
"""
Generate Trial-3 refactored post-ell24 same-family higher-ceiling artifacts for
8.7.56.301-.304.

The same-family frontier is reopened through ell=24 and the best W/Z pair is
already near-pass in shape, but the absolute ceiling still sits below both W/Z
anchor thresholds. This branch freezes whether the next honest route is now a
higher-ell frontier extension above ell=24 rather than another pair-side or
normalization retry.
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
PREVIOUS_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_t3_post_ell24_target_gap_branch.py"

POST_PHOTON_PRESERVATION = OUT / "mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_metrics.json"
SOLVER_REFACTOR_EXECUTION = OUT / "mass_origin_v2_trial3_solver_refactor_execution_audit_metrics.json"
PRIOR_RADIAL_AUDIT = OUT / "mass_origin_v2_trial3_refactored_post_ell19_radial_domain_extension_audit_metrics.json"
PRIOR_TARGET_GAP_SOURCE = OUT / "mass_origin_v2_trial3_refactored_post_ell24_same_family_target_gap_extension_source_inventory_metrics.json"
PRIOR_TARGET_GAP_AUDIT = OUT / "mass_origin_v2_trial3_refactored_post_ell24_same_family_target_gap_extension_audit_metrics.json"
PRIOR_DECLARATION = OUT / "mass_origin_v2_trial3_refactored_declaration_sixth_gate_metrics.json"
PRIOR_DISPOSITION = OUT / "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_eighteenth_refresh_metrics.json"

W_MASS_MEV = 80369.0
Z_MASS_MEV = 91187.6
ELECTRON_MASS_MEV = 0.51099895
W_TARGET = W_MASS_MEV / ELECTRON_MASS_MEV
Z_TARGET = Z_MASS_MEV / ELECTRON_MASS_MEV


# 関数: helper branch を動的 import する。
def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 関数: post-ell24 higher-ceiling branch を実行する。

def main() -> None:
    helper = load_module(HELPER_BRANCH, "trial3_post_ell24_higher_ceiling_helper")

    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        HELPER_BRANCH,
        PREVIOUS_BRANCH,
        POST_PHOTON_PRESERVATION,
        SOLVER_REFACTOR_EXECUTION,
        PRIOR_RADIAL_AUDIT,
        PRIOR_TARGET_GAP_SOURCE,
        PRIOR_TARGET_GAP_AUDIT,
        PRIOR_DECLARATION,
        PRIOR_DISPOSITION,
    ):
        helper.req(path)

    status_text = helper.read_text(STATUS)
    roadmap_text = helper.read_text(ROADMAP)
    ai_context = helper.read_json(AI_CONTEXT)
    previous_branch_text = helper.read_text(PREVIOUS_BRANCH)
    post_photon = helper.read_json(POST_PHOTON_PRESERVATION)
    solver_refactor = helper.read_json(SOLVER_REFACTOR_EXECUTION)
    prior_radial_audit = helper.read_json(PRIOR_RADIAL_AUDIT)
    prior_target_gap_source = helper.read_json(PRIOR_TARGET_GAP_SOURCE)
    prior_target_gap_audit = helper.read_json(PRIOR_TARGET_GAP_AUDIT)
    prior_declaration = helper.read_json(PRIOR_DECLARATION)
    prior_disposition = helper.read_json(PRIOR_DISPOSITION)

    radial_summary = prior_radial_audit["summary"]
    prior_summary = prior_target_gap_audit["summary"]
    best_pair = prior_summary["best_pair_or_none"]
    current_ceiling = float(prior_summary["current_same_family_ceiling_to_electron"])
    current_ceiling_gap_to_w = float(prior_summary["current_ceiling_gap_to_w"])
    current_ceiling_gap_to_z = float(prior_summary["current_ceiling_gap_to_z"])
    higher_ceiling_extension_required = bool(prior_summary["higher_ceiling_extension_required"])
    best_pair_near_pass = bool(prior_summary["best_pair_near_pass"])
    anchor_threshold_dominant = bool(prior_summary["anchor_threshold_dominant_blocker"])
    previous_rebuilt_ceiling = float(radial_summary["prior_rebuilt_verified_ceiling_to_electron"])
    current_rebuilt_ceiling = float(radial_summary["rebuilt_verified_ceiling_to_electron"])
    higher_ceiling_gain_factor = current_rebuilt_ceiling / previous_rebuilt_ceiling
    maximum_detected_ell = int(radial_summary["maximum_detected_ell"])
    maximum_detected_ell_with_k_positive = int(radial_summary["maximum_detected_ell_with_k_positive"])
    normalized_ratios_preserved = bool(post_photon["summary"]["working_action_vector_mass_spectrum_physical_claim_preserved"])
    normalization_update_only = bool(post_photon["summary"]["working_action_vector_mass_spectrum_normalization_update_only"])
    software_blocker_removed = bool(solver_refactor["summary"]["software_blocker_removed"])

    higher_ell_frontier_extension_preferred = bool(
        higher_ceiling_extension_required
        and best_pair_near_pass
        and anchor_threshold_dominant
        and maximum_detected_ell == 24
        and maximum_detected_ell_with_k_positive == 24
        and higher_ceiling_gain_factor > 1.0
        and software_blocker_removed
        and normalized_ratios_preserved
        and normalization_update_only
    )
    pair_side_retry_dominant = bool(best_pair is not None and not higher_ell_frontier_extension_preferred)
    trial3_recommended_condition_satisfied = False
    selected_residual_route = None
    missing_v2_artifact = None
    if higher_ell_frontier_extension_preferred:
        selected_residual_route = "trial3_relaunched_refactored_post_ell24_higher_ell_frontier_extension_identification"
        missing_v2_artifact = "trial3_relaunched_refactored_same_family_localized_exact_family_table_above_ell_24"

    inventory_targets = [
        {
            "label": "status_points_to_8_7_56_301",
            "present": "current official next step は `8.7.56.301`" in status_text,
            "evidence": {"status_markdown": helper.rel(STATUS)},
        },
        {
            "label": "roadmap_higher_ceiling_branch_present",
            "present": "`8.7.56.301-.304` 試練3 refactored post-`ell=24` same-family higher-ceiling extension residual branch" in roadmap_text,
            "evidence": {"roadmap_markdown": helper.rel(ROADMAP)},
        },
        {
            "label": "prior_target_gap_inventory_ready",
            "present": bool(prior_target_gap_source["summary"]["required_sources_present"] == prior_target_gap_source["summary"]["required_sources_total"]),
            "evidence": prior_target_gap_source["summary"],
        },
        {
            "label": "prior_target_gap_audit_ready",
            "present": higher_ceiling_extension_required,
            "evidence": prior_summary,
        },
        {
            "label": "same_family_ceiling_present",
            "present": current_ceiling > 0.0,
            "evidence": {
                "current_same_family_ceiling_to_electron": current_ceiling,
                "current_ceiling_gap_to_w": current_ceiling_gap_to_w,
                "current_ceiling_gap_to_z": current_ceiling_gap_to_z,
            },
        },
        {
            "label": "near_pass_pair_evidence_present",
            "present": best_pair is not None and best_pair_near_pass,
            "evidence": {"best_pair_or_none": best_pair, "best_pair_near_pass": best_pair_near_pass},
        },
        {
            "label": "post_ell24_frontier_present",
            "present": maximum_detected_ell == 24 and maximum_detected_ell_with_k_positive == 24,
            "evidence": {
                "maximum_detected_ell": maximum_detected_ell,
                "maximum_detected_ell_with_k_positive": maximum_detected_ell_with_k_positive,
                "localized_ell_values": radial_summary["localized_ell_values"],
            },
        },
        {
            "label": "ceiling_gain_history_present",
            "present": current_rebuilt_ceiling > previous_rebuilt_ceiling,
            "evidence": {
                "prior_rebuilt_verified_ceiling_to_electron": previous_rebuilt_ceiling,
                "current_rebuilt_verified_ceiling_to_electron": current_rebuilt_ceiling,
                "higher_ceiling_gain_factor": higher_ceiling_gain_factor,
            },
        },
        {
            "label": "normalization_retry_not_required",
            "present": normalized_ratios_preserved and normalization_update_only,
            "evidence": post_photon["summary"],
        },
        {
            "label": "solver_blocker_already_removed",
            "present": software_blocker_removed,
            "evidence": solver_refactor["summary"],
        },
        {
            "label": "previous_branch_points_to_higher_ceiling_route",
            "present": helper.hit(previous_branch_text, "selected_residual_route = \"trial3_relaunched_refactored_post_ell24_same_family_higher_ceiling_extension_identification\"") is not None,
            "evidence": helper.hit(previous_branch_text, "selected_residual_route = \"trial3_relaunched_refactored_post_ell24_same_family_higher_ceiling_extension_identification\""),
        },
        {
            "label": "prior_disposition_keeps_trial2_reserve_and_trial4_deferred",
            "present": prior_disposition["summary"]["trial2_paper_side_sync_state"] == "unlocked_reserve_retained"
            and bool(prior_disposition["summary"]["trial4_deferred"]),
            "evidence": prior_disposition["summary"],
        },
    ]

    source_inventory = helper.payload(
        "8.7.56.301",
        "Trial-3 refactored post-ell24 same-family higher-ceiling extension source inventory",
        {
            "status_markdown": helper.rel(STATUS),
            "roadmap_markdown": helper.rel(ROADMAP),
            "ai_context_json": helper.rel(AI_CONTEXT),
            "mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_json": helper.rel(POST_PHOTON_PRESERVATION),
            "mass_origin_v2_trial3_solver_refactor_execution_audit_json": helper.rel(SOLVER_REFACTOR_EXECUTION),
            "mass_origin_v2_trial3_refactored_post_ell19_radial_domain_extension_audit_json": helper.rel(PRIOR_RADIAL_AUDIT),
            "mass_origin_v2_trial3_refactored_post_ell24_same_family_target_gap_extension_source_inventory_json": helper.rel(PRIOR_TARGET_GAP_SOURCE),
            "mass_origin_v2_trial3_refactored_post_ell24_same_family_target_gap_extension_audit_json": helper.rel(PRIOR_TARGET_GAP_AUDIT),
            "mass_origin_v2_trial3_refactored_declaration_sixth_gate_json": helper.rel(PRIOR_DECLARATION),
            "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_eighteenth_refresh_json": helper.rel(PRIOR_DISPOSITION),
            "mass_origin_v2_t3_post_ell24_target_gap_branch_py": helper.rel(PREVIOUS_BRANCH),
        },
        "Freeze the current same-family ceiling, the remaining W/Z threshold gap, the near-pass pair evidence, and the post-ell24 frontier prerequisites before deciding whether the next honest route is a higher-ell extension above ell=24.",
        {
            "inventory_rule": "the higher-ceiling source pack must include the current ceiling, the remaining W/Z gap, the near-pass pair evidence, the current frontier limit, and the proof that normalization and software blockers are already retired",
            "frontier_rule": "if the frontier is reopened through ell=24, the pair is already near-pass, and both W/Z thresholds remain above the current ceiling, the next honest route is to push the localized same-family frontier above ell=24",
        },
        [
            helper.row("trial3_refactored_post_ell24_higher_ceiling_source_inventory_complete", "pass", "Trial-3 refactored post-ell24 higher-ceiling source inventory complete", 1, "The higher-ceiling source pack is frozen."),
            helper.row("trial3_refactored_post_ell24_current_ceiling_present", "pass" if current_ceiling > 0.0 else "reject", "current same-family ceiling present", current_ceiling, "The next frontier extension must start from the already-frozen ell<=24 same-family ceiling."),
            helper.row("trial3_refactored_post_ell24_near_pass_pair_present", "pass" if best_pair_near_pass else "reject", "near-pass pair evidence present", 1 if best_pair_near_pass else 0, "The higher-ceiling route is honest only if pair shape is already close enough that absolute ceiling becomes the dominant blocker."),
            helper.row("trial3_refactored_post_ell24_frontier_cap_present", "pass" if maximum_detected_ell == 24 else "reject", "current same-family frontier capped at ell=24", float(maximum_detected_ell), "The next route should only move to ell>24 after the current frontier cap is explicitly frozen."),
            helper.row("trial3_refactored_post_ell24_normalization_retry_not_required", "pass" if normalized_ratios_preserved and normalization_update_only else "reject", "normalization retry no longer required", 1 if normalized_ratios_preserved and normalization_update_only else 0, "The higher-ceiling route should not relitigate the already-preserved normalization update."),
            helper.row("trial3_refactored_post_ell24_solver_blocker_removed", "pass" if software_blocker_removed else "reject", "solver blocker already removed", 1 if software_blocker_removed else 0, "The next route should not return to software refactor work after explicit k>0 execution has reopened the exact handoff."),
        ],
        {
            "required_sources_total": len(inventory_targets),
            "required_sources_present": sum(1 for item in inventory_targets if item["present"]),
            "current_same_family_ceiling_to_electron": current_ceiling,
            "current_ceiling_gap_to_w": current_ceiling_gap_to_w,
            "current_ceiling_gap_to_z": current_ceiling_gap_to_z,
            "best_pair_near_pass": best_pair_near_pass,
            "maximum_detected_ell": maximum_detected_ell,
            "maximum_detected_ell_with_k_positive": maximum_detected_ell_with_k_positive,
            "higher_ceiling_gain_factor": higher_ceiling_gain_factor,
            "status_current_step_before_branch": ai_context["current_step"],
        },
        {
            "overall_status": "trial3_refactored_post_ell24_higher_ceiling_source_inventory_frozen",
            "source_inventory_complete": True,
            "advance_to_8_7_56_302": True,
            "next_required_artifacts": ["trial3_refactored_post_ell24_higher_ceiling_extension_audit"],
        },
        {
            "inventory_targets": inventory_targets,
            "prior_target_gap_audit_summary": prior_summary,
            "prior_radial_audit_summary": radial_summary,
            "prior_declaration_summary": prior_declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
        },
    )

    audit = helper.payload(
        "8.7.56.302",
        "Trial-3 refactored post-ell24 same-family higher-ceiling extension audit",
        source_inventory["inputs"],
        "Audit whether the honest next Trial-3 route is a higher-ell frontier extension above ell=24 or whether the weak-sector path is exhausted under the current canon.",
        {
            "higher_ceiling_rule": "if the current same-family frontier is capped at ell=24, the pair is near-pass, and both W/Z thresholds remain above the current ceiling, the dominant blocker is a missing higher-ell frontier above ell=24",
            "frontier_preference_rule": "a higher-ell frontier extension is preferred only if normalization and software blockers are already retired and the previous ell/radial extensions produced monotonic ceiling gains",
        },
        [
            helper.row("trial3_refactored_post_ell24_higher_ceiling_audit_complete", "pass", "Trial-3 refactored post-ell24 higher-ceiling audit complete", 1, "The higher-ceiling audit is frozen."),
            helper.row("trial3_refactored_post_ell24_anchor_threshold_dominant_blocker", "pass" if anchor_threshold_dominant else "reject", "anchor threshold gap remains the dominant blocker", 1 if anchor_threshold_dominant else 0, "The next route should keep the focus on crossing the W/Z thresholds rather than re-opening pair shape first."),
            helper.row("trial3_refactored_post_ell24_higher_ceiling_extension_required", "pass" if higher_ceiling_extension_required else "reject", "higher-ceiling extension remains required", 1 if higher_ceiling_extension_required else 0, "The current frontier still sits below both W/Z thresholds."),
            helper.row("trial3_refactored_post_ell24_higher_ell_frontier_extension_preferred", "pass" if higher_ell_frontier_extension_preferred else "reject", "higher-ell frontier extension preferred", 1 if higher_ell_frontier_extension_preferred else 0, "The honest next route is to push the localized same-family frontier above ell=24."),
            helper.row("trial3_refactored_post_ell24_pair_side_retry_dominant", "reject" if not pair_side_retry_dominant else "pass", "pair-side retry dominant", 1 if pair_side_retry_dominant else 0, "Pair-side retry is secondary while the absolute frontier remains sub-threshold."),
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
            "anchor_threshold_dominant_blocker": anchor_threshold_dominant,
            "higher_ceiling_extension_required": higher_ceiling_extension_required,
            "higher_ell_frontier_extension_preferred": higher_ell_frontier_extension_preferred,
            "pair_side_retry_dominant": pair_side_retry_dominant,
            "maximum_detected_ell": maximum_detected_ell,
            "maximum_detected_ell_with_k_positive": maximum_detected_ell_with_k_positive,
            "higher_ceiling_gain_factor": higher_ceiling_gain_factor,
        },
        {
            "overall_status": "trial3_refactored_post_ell24_higher_ceiling_audited",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_303": True,
            "next_required_artifacts": ["trial3_refactored_declaration_seventh_gate"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "prior_target_gap_audit_summary": prior_summary,
            "prior_radial_audit_summary": radial_summary,
            "solver_refactor_summary": solver_refactor["summary"],
            "post_photon_summary": post_photon["summary"],
        },
    )

    declaration = helper.payload(
        "8.7.56.303",
        "Trial-3 refactored declaration seventh gate",
        source_inventory["inputs"],
        "Freeze whether the post-ell24 higher-ceiling audit already closes Trial-3 or whether the next honest route is a higher-ell frontier extension above ell=24.",
        {
            "closeout_rule": "Trial-3 closes only if the current same-family family already crosses the W/Z thresholds and closes the pair-side observables together",
            "residual_rule": "if the frontier is still capped at ell=24 while the pair is near-pass and the anchor thresholds remain upstream, the next residual route is a higher-ell frontier extension",
        },
        [
            helper.row("trial3_refactored_declaration_seventh_gate_complete", "pass", "Trial-3 refactored declaration seventh gate complete", 1, "The seventh declaration gate is frozen."),
            helper.row("trial3_refactored_seventh_branch_closeable", "pass" if trial3_recommended_condition_satisfied else "reject", "refactored Trial-3 branch closeable after higher-ceiling audit", 1 if trial3_recommended_condition_satisfied else 0, "The branch closes only if the current same-family frontier already closes the weak-sector pack."),
            helper.row("trial3_refactored_seventh_residual_route_required", "reject" if trial3_recommended_condition_satisfied else "pass", "refactored Trial-3 residual route required after higher-ceiling audit", 0 if trial3_recommended_condition_satisfied else 1, "A new residual route is still required while the frontier remains capped at ell=24."),
            helper.row("trial3_refactored_execute_trial2_paper_sync_now_seventh_gate", "reject", "execute Trial-2 paper-side sync now after higher-ceiling audit", 0, "Trial-2 paper-side sync remains reserve work while the higher-ell frontier route is still open."),
        ],
        {
            "trial3_recommended_condition_satisfied": trial3_recommended_condition_satisfied,
            "trial3_current_branch_closeable": trial3_recommended_condition_satisfied,
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "recommended_next_route_or_none": None if trial3_recommended_condition_satisfied else "8.7.56.305",
        },
        {
            "overall_status": "trial3_refactored_declaration_seventh_gate_frozen",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_304": True,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["trial3_refactored_trial2_paper_sync_trial4_disposition_nineteenth_refresh"],
        },
        {
            "audit_summary": audit["summary"],
            "prior_declaration_summary": prior_declaration["summary"],
            "best_pair_or_none": best_pair,
        },
    )

    disposition = helper.payload(
        "8.7.56.304",
        "Trial-2 paper-side sync / Trial-4 disposition nineteenth refresh",
        source_inventory["inputs"],
        "Refresh the reserve/deferred ordering after the post-ell24 higher-ceiling audit and freeze the next official higher-ell frontier route.",
        {
            "trial2_rule": "Trial-2 paper-side sync stays unlocked reserve retained while the refactored Trial-3 route still has an honest higher-ell frontier path",
            "trial4_rule": "Trial-4 remains deferred while Trial-3 still exposes a current-canon higher-ell frontier",
        },
        [
            helper.row("trial3_refactored_trial2_trial4_nineteenth_refresh_complete", "pass", "Trial-2 paper-side sync / Trial-4 disposition nineteenth refresh complete", 1, "The reserve/deferred ordering is refreshed after the higher-ceiling audit."),
            helper.row("trial3_refactored_trial2_paper_side_sync_reserve_retained_nineteenth_refresh", "pass", "Trial-2 paper-side sync reserve retained", 1, "Trial-2 paper-side sync stays unlocked reserve work while the higher-ell frontier route is still open."),
            helper.row("trial3_refactored_trial4_deferred_retained_nineteenth_refresh", "pass", "Trial-4 deferred retained", 1, "Trial-4 stays deferred while Trial-3 still has an honest higher-ell frontier extension path."),
        ],
        {
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "trial2_paper_side_sync_state": "unlocked_reserve_retained",
            "trial4_deferred": True,
            "recommended_next_route_or_none": None if trial3_recommended_condition_satisfied else "8.7.56.305",
        },
        {
            "overall_status": "trial3_refactored_trial2_trial4_nineteenth_disposition_refreshed",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_13": False,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["8.7.56.305"],
        },
        {
            "declaration_summary": declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
            "solver_refactor_summary": solver_refactor["summary"],
        },
    )

    helper.write_artifact("mass_origin_v2_trial3_refactored_post_ell24_same_family_higher_ceiling_extension_source_inventory", source_inventory)
    helper.write_artifact("mass_origin_v2_trial3_refactored_post_ell24_same_family_higher_ceiling_extension_audit", audit)
    helper.write_artifact("mass_origin_v2_trial3_refactored_declaration_seventh_gate", declaration)
    helper.write_artifact("mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_nineteenth_refresh", disposition)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_refactored_post_ell24_same_family_higher_ceiling_extension_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_post_ell24_same_family_higher_ceiling_extension_audit_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_declaration_seventh_gate_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_nineteenth_refresh_metrics.json")


# 関数: CLI から post-ell24 higher-ceiling branch を起動する。

if __name__ == "__main__":
    main()
