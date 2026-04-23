#!/usr/bin/env python3
"""
Generate Trial-3 refactored post-ell24 higher-ell frontier artifacts for
8.7.56.305-.308.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

POST_PHOTON_PRESERVATION = OUT / "mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_metrics.json"
SOLVER_REFACTOR_EXECUTION = OUT / "mass_origin_v2_trial3_solver_refactor_execution_audit_metrics.json"
PRIOR_RADIAL_AUDIT = OUT / "mass_origin_v2_trial3_refactored_post_ell19_radial_domain_extension_audit_metrics.json"
PRIOR_HIGHER_SOURCE = OUT / "mass_origin_v2_trial3_refactored_post_ell24_same_family_higher_ceiling_extension_source_inventory_metrics.json"
PRIOR_HIGHER_AUDIT = OUT / "mass_origin_v2_trial3_refactored_post_ell24_same_family_higher_ceiling_extension_audit_metrics.json"
PRIOR_DECLARATION = OUT / "mass_origin_v2_trial3_refactored_declaration_seventh_gate_metrics.json"
PRIOR_DISPOSITION = OUT / "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_nineteenth_refresh_metrics.json"
VECTOR_SPIN = OUT / "mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json"
SCALAR_SPECTRUM = OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"
EXACT_HANDOFF = OUT / "mass_origin_vector_qball_exact_mass_table_handoff_retry_metrics.json"

HELPER_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_t3_post_ell18_amplitude_branch.py"
RADIAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_t3_post_ell19_radial_branch.py"
PREVIOUS_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_t3_post_ell24_higher_ceiling_branch.py"
NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_COUPLED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"
EXTENDED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_extended_hierarchy_branch.py"

W_MASS_MEV = 80369.0
Z_MASS_MEV = 91187.6
ELECTRON_MASS_MEV = 0.51099895
W_TARGET = W_MASS_MEV / ELECTRON_MASS_MEV
Z_TARGET = Z_MASS_MEV / ELECTRON_MASS_MEV
PASS_THRESHOLD = 0.10
TAIL_POST_ELL24_VALUES = tuple(range(25, 31))


# 関数: local Python module を動的に読む。
def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 関数: post-ell24 higher-ell frontier branch を実行する。

def main() -> None:
    helper = load_module(HELPER_BRANCH, "trial3_post_ell24_frontier_helper")
    radial = load_module(RADIAL_BRANCH, "trial3_post_ell24_frontier_radial")
    numerical = load_module(NUMERICAL_BRANCH, "trial3_post_ell24_frontier_num")
    full = load_module(FULL_COUPLED_BRANCH, "trial3_post_ell24_frontier_full")
    extended = load_module(EXTENDED_BRANCH, "trial3_post_ell24_frontier_ext")
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        POST_PHOTON_PRESERVATION,
        SOLVER_REFACTOR_EXECUTION,
        PRIOR_RADIAL_AUDIT,
        PRIOR_HIGHER_SOURCE,
        PRIOR_HIGHER_AUDIT,
        PRIOR_DECLARATION,
        PRIOR_DISPOSITION,
        VECTOR_SPIN,
        SCALAR_SPECTRUM,
        EXACT_HANDOFF,
        HELPER_BRANCH,
        RADIAL_BRANCH,
        PREVIOUS_BRANCH,
        NUMERICAL_BRANCH,
        FULL_COUPLED_BRANCH,
        EXTENDED_BRANCH,
    ):
        helper.req(path)

    status_text = helper.read_text(STATUS)
    roadmap_text = helper.read_text(ROADMAP)
    ai_context = helper.read_json(AI_CONTEXT)
    post_photon = helper.read_json(POST_PHOTON_PRESERVATION)
    solver_refactor = helper.read_json(SOLVER_REFACTOR_EXECUTION)
    prior_radial_audit = helper.read_json(PRIOR_RADIAL_AUDIT)
    prior_source = helper.read_json(PRIOR_HIGHER_SOURCE)
    prior_audit = helper.read_json(PRIOR_HIGHER_AUDIT)
    prior_declaration = helper.read_json(PRIOR_DECLARATION)
    prior_disposition = helper.read_json(PRIOR_DISPOSITION)
    vector_spin = helper.read_json(VECTOR_SPIN)
    scalar_spectrum = helper.read_json(SCALAR_SPECTRUM)
    exact_handoff = helper.read_json(EXACT_HANDOFF)

    numerical_text = helper.read_text(NUMERICAL_BRANCH)
    radial_text = helper.read_text(RADIAL_BRANCH)
    previous_branch_text = helper.read_text(PREVIOUS_BRANCH)
    helper_branch_text = helper.read_text(HELPER_BRANCH)
    full_text = helper.read_text(FULL_COUPLED_BRANCH)

    scalar_modes = list(scalar_spectrum["evidence"]["discrete_mass_mode_rows"])
    lambda_rot = float(vector_spin["summary"]["lambda_rot_value"])
    normalization_scale = float(post_photon["summary"]["absolute_mass_normalization_scale_factor"])
    current_ceiling = float(prior_audit["summary"]["current_same_family_ceiling_to_electron"])
    current_w_gap = float(prior_audit["summary"]["current_ceiling_gap_to_w"])
    current_z_gap = float(prior_audit["summary"]["current_ceiling_gap_to_z"])
    best_pair_near_pass = bool(prior_audit["summary"]["best_pair_near_pass"])
    prior_maximum_detected_ell = int(prior_audit["summary"]["maximum_detected_ell"])
    software_blocker_removed = bool(solver_refactor["summary"]["software_blocker_removed"])
    normalization_update_only = bool(post_photon["summary"]["working_action_vector_mass_spectrum_normalization_update_only"])
    physical_claim_preserved = bool(post_photon["summary"]["working_action_vector_mass_spectrum_physical_claim_preserved"])

    inventory_targets = [
        {
            "label": "status_points_to_8_7_56_305",
            "present": "current official next step は `8.7.56.305`" in status_text,
            "evidence": {"status_markdown": helper.rel(STATUS)},
        },
        {
            "label": "roadmap_higher_ell_frontier_branch_present",
            "present": "`8.7.56.305-.308` 試練3 refactored post-`ell=24` higher-ell frontier extension residual branch" in roadmap_text,
            "evidence": {"roadmap_markdown": helper.rel(ROADMAP)},
        },
        {
            "label": "prior_frontier_cap_fixed_at_ell24",
            "present": prior_maximum_detected_ell == 24,
            "evidence": prior_audit["summary"],
        },
        {
            "label": "current_same_family_ceiling_present",
            "present": current_ceiling > 0.0,
            "evidence": {
                "current_same_family_ceiling_to_electron": current_ceiling,
                "current_ceiling_gap_to_w": current_w_gap,
                "current_ceiling_gap_to_z": current_z_gap,
            },
        },
        {
            "label": "radial_scan_function_present",
            "present": helper.hit(radial_text, "def scan_radial_extended_sector(") is not None,
            "evidence": helper.hit(radial_text, "def scan_radial_extended_sector("),
        },
        {
            "label": "tail_beta_grid_present",
            "present": helper.hit(helper_branch_text, "TAIL_WIDENED_BETA_GRID = (") is not None,
            "evidence": helper.hit(helper_branch_text, "TAIL_WIDENED_BETA_GRID = ("),
        },
        {
            "label": "tail_amplitude_tranche_present",
            "present": helper.hit(helper_branch_text, "TAIL_EXTENDED_AMPLITUDES = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0)") is not None,
            "evidence": helper.hit(helper_branch_text, "TAIL_EXTENDED_AMPLITUDES = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0)"),
        },
        {
            "label": "low_ell_base_builder_present",
            "present": helper.hit(numerical_text, "def build_base_modes(ell_values: tuple[int, ...] = (1, 2, 3))") is not None,
            "evidence": helper.hit(numerical_text, "def build_base_modes(ell_values: tuple[int, ...] = (1, 2, 3))"),
        },
        {
            "label": "exact_ladder_builder_present",
            "present": helper.hit(full_text, "def build_exact_ladder(") is not None,
            "evidence": helper.hit(full_text, "def build_exact_ladder("),
        },
        {
            "label": "software_blocker_removed",
            "present": software_blocker_removed,
            "evidence": solver_refactor["summary"],
        },
        {
            "label": "normalization_update_only_preserved",
            "present": physical_claim_preserved and normalization_update_only,
            "evidence": post_photon["summary"],
        },
        {
            "label": "previous_branch_points_to_higher_ell_route",
            "present": helper.hit(previous_branch_text, "selected_residual_route = \"trial3_relaunched_refactored_post_ell24_higher_ell_frontier_extension_identification\"") is not None,
            "evidence": helper.hit(previous_branch_text, "selected_residual_route = \"trial3_relaunched_refactored_post_ell24_higher_ell_frontier_extension_identification\""),
        },
        {
            "label": "same_family_target_pack_present",
            "present": bool(exact_handoff["summary"]["hand_off_to_8_7_55_2_84"]),
            "evidence": {
                "best_exact_match_or_none": exact_handoff["summary"]["best_exact_match_or_none"],
                "w_target_to_electron": W_TARGET,
                "z_target_to_electron": Z_TARGET,
            },
        },
    ]

    source_inventory = helper.payload(
        "8.7.56.305",
        "Trial-3 refactored post-ell24 higher-ell frontier extension source inventory",
        {
            "status_markdown": helper.rel(STATUS),
            "roadmap_markdown": helper.rel(ROADMAP),
            "ai_context_json": helper.rel(AI_CONTEXT),
            "mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_json": helper.rel(POST_PHOTON_PRESERVATION),
            "mass_origin_v2_trial3_solver_refactor_execution_audit_json": helper.rel(SOLVER_REFACTOR_EXECUTION),
            "mass_origin_v2_trial3_refactored_post_ell19_radial_domain_extension_audit_json": helper.rel(PRIOR_RADIAL_AUDIT),
            "mass_origin_v2_trial3_refactored_post_ell24_same_family_higher_ceiling_extension_source_inventory_json": helper.rel(PRIOR_HIGHER_SOURCE),
            "mass_origin_v2_trial3_refactored_post_ell24_same_family_higher_ceiling_extension_audit_json": helper.rel(PRIOR_HIGHER_AUDIT),
            "mass_origin_v2_trial3_refactored_declaration_seventh_gate_json": helper.rel(PRIOR_DECLARATION),
            "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_nineteenth_refresh_json": helper.rel(PRIOR_DISPOSITION),
            "mass_origin_v2_t3_post_ell24_higher_ceiling_branch_py": helper.rel(PREVIOUS_BRANCH),
            "mass_origin_v2_t3_post_ell18_amplitude_branch_py": helper.rel(HELPER_BRANCH),
            "mass_origin_v2_t3_post_ell19_radial_branch_py": helper.rel(RADIAL_BRANCH),
            "mass_origin_vector_qball_numerical_solver_branch_py": helper.rel(NUMERICAL_BRANCH),
            "mass_origin_vector_qball_full_coupled_solver_branch_py": helper.rel(FULL_COUPLED_BRANCH),
            "mass_origin_vector_qball_extended_hierarchy_branch_py": helper.rel(EXTENDED_BRANCH),
        },
        "Freeze the ell=24 frontier cap, the preserved normalized same-family ceiling, and the tail-scan prerequisites before reopening the same-family exact-family table above ell=24.",
        {
            "inventory_rule": "the higher-ell frontier source pack must include the ell=24 ceiling/cap, the tail beta/amplitude windows, the preserved normalization rule, and the proof that solver-side blockers are already removed",
            "tail_scan_rule": f"for ell in {list(TAIL_POST_ELL24_VALUES)} keep the post-photon normalization fixed, preserve the primary radial contract {radial.PRIMARY_EXTENDED_RADIAL_CONTRACT}, and only extend the same-family frontier with the tail beta/amplitude tranche",
        },
        [
            helper.row("trial3_refactored_post_ell24_higher_ell_frontier_source_inventory_complete", "pass", "Trial-3 refactored post-ell24 higher-ell frontier source inventory complete", 1, "The higher-ell frontier source pack is frozen."),
            helper.row("trial3_refactored_post_ell24_current_ceiling_present", "pass" if current_ceiling > 0.0 else "reject", "current same-family ceiling present", current_ceiling, "The frontier extension starts from the already-frozen ell<=24 same-family ceiling."),
            helper.row("trial3_refactored_post_ell24_frontier_cap_present", "pass" if prior_maximum_detected_ell == 24 else "reject", "prior frontier cap fixed at ell=24", float(prior_maximum_detected_ell), "The next route must explicitly state that the current exact-family frontier is capped at ell=24 before the tail extension runs."),
            helper.row("trial3_refactored_post_ell24_tail_beta_grid_present", "pass", "tail beta grid present", float(len(helper.TAIL_WIDENED_BETA_GRID)), "The tail frontier extension uses the widened beta tranche already frozen in the amplitude branch."),
            helper.row("trial3_refactored_post_ell24_tail_amplitude_tranche_present", "pass", "tail amplitude tranche present", float(len(helper.TAIL_EXTENDED_AMPLITUDES)), "The tail frontier extension uses the high-amplitude tranche already frozen in the amplitude branch."),
            helper.row("trial3_refactored_post_ell24_solver_blocker_removed", "pass" if software_blocker_removed else "reject", "solver blocker removed before higher-ell frontier scan", 1 if software_blocker_removed else 0, "The frontier extension is honest only because the explicit k>0 solver refactor is already complete."),
            helper.row("trial3_refactored_post_ell24_normalization_update_only_preserved", "pass" if physical_claim_preserved and normalization_update_only else "reject", "normalization update only preserved", 1 if physical_claim_preserved and normalization_update_only else 0, "The higher-ell route should not reopen the already-settled normalization-only update."),
        ],
        {
            "required_sources_total": len(inventory_targets),
            "required_sources_present": sum(1 for item in inventory_targets if item["present"]),
            "current_same_family_ceiling_to_electron": current_ceiling,
            "current_ceiling_gap_to_w": current_w_gap,
            "current_ceiling_gap_to_z": current_z_gap,
            "prior_best_pair_near_pass": best_pair_near_pass,
            "prior_maximum_detected_ell": prior_maximum_detected_ell,
            "tail_post_ell24_values": list(TAIL_POST_ELL24_VALUES),
            "tail_widened_beta_grid": list(helper.TAIL_WIDENED_BETA_GRID),
            "tail_extended_amplitudes": list(helper.TAIL_EXTENDED_AMPLITUDES),
            "radial_contract": radial.PRIMARY_EXTENDED_RADIAL_CONTRACT,
            "status_current_step_before_branch": ai_context["current_step"],
        },
        {
            "overall_status": "trial3_refactored_post_ell24_higher_ell_frontier_source_inventory_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_306": True,
            "next_required_artifacts": ["trial3_refactored_post_ell24_higher_ell_frontier_extension_audit"],
        },
        {
            "inventory_targets": inventory_targets,
            "prior_higher_source_summary": prior_source["summary"],
            "prior_higher_audit_summary": prior_audit["summary"],
            "prior_declaration_summary": prior_declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
            "prior_radial_audit_summary": prior_radial_audit["summary"],
        },
    )

    ell_scan_rows, low_ell_base_modes = numerical.build_base_modes(radial.LOW_ELL_VALUES)
    base_modes = list(low_ell_base_modes)

    standard_scan_rows: dict[int, list[dict]] = {}
    standard_mode_rows: dict[int, list[dict]] = {}
    for ell in radial.STANDARD_EXTENSION_ELLS:
        scan_rows = extended.scan_extended_sector(numerical, int(ell))
        mode_rows = numerical.interpolate_integer_modes(scan_rows, int(ell))
        standard_scan_rows[int(ell)] = scan_rows
        standard_mode_rows[int(ell)] = mode_rows
        base_modes.extend(mode_rows)

    control_scan_rows, control_amplitude_values = helper.scan_amplitude_extended_sector(
        numerical,
        radial.CONTROL_ELL,
        radial.PRIMARY_WIDENED_BETA_GRID,
        radial.PRIMARY_EXTENDED_AMPLITUDES,
    )
    control_mode_rows = numerical.interpolate_integer_modes(control_scan_rows, radial.CONTROL_ELL)
    base_modes.extend(control_mode_rows)

    primary_scan_rows: dict[int, list[dict]] = {}
    primary_mode_rows: dict[int, list[dict]] = {}
    for ell in radial.PRIMARY_POST_ELL19_VALUES:
        scan_rows, _ = radial.scan_radial_extended_sector(
            numerical,
            helper,
            int(ell),
            radial.PRIMARY_WIDENED_BETA_GRID,
            radial.PRIMARY_EXTENDED_AMPLITUDES,
            radial.PRIMARY_EXTENDED_RADIAL_CONTRACT,
        )
        mode_rows = numerical.interpolate_integer_modes(scan_rows, int(ell))
        primary_scan_rows[int(ell)] = scan_rows
        primary_mode_rows[int(ell)] = mode_rows
        base_modes.extend(mode_rows)

    tail_scan_rows: dict[int, list[dict]] = {}
    tail_mode_rows: dict[int, list[dict]] = {}
    for ell in TAIL_POST_ELL24_VALUES:
        scan_rows, _ = radial.scan_radial_extended_sector(
            numerical,
            helper,
            int(ell),
            helper.TAIL_WIDENED_BETA_GRID,
            helper.TAIL_EXTENDED_AMPLITUDES,
            radial.PRIMARY_EXTENDED_RADIAL_CONTRACT,
        )
        mode_rows = numerical.interpolate_integer_modes(scan_rows, int(ell))
        tail_scan_rows[int(ell)] = scan_rows
        tail_mode_rows[int(ell)] = mode_rows
        base_modes.extend(mode_rows)

    base_modes = sorted(base_modes, key=lambda item: (int(item["ell"]), int(item["k"]), int(item["n"])))
    exact_rows = full.build_exact_ladder(scalar_modes, base_modes, lambda_rot)
    normalized_vector_rows = helper.normalize_vector_rows(
        [row_data for row_data in exact_rows if int(row_data["ell"]) > 0],
        normalization_scale,
    )

    best_w = helper.closest_state(normalized_vector_rows, W_TARGET)
    best_z = helper.closest_state(normalized_vector_rows, Z_TARGET)
    best_pair = helper.best_ratio_pair(normalized_vector_rows)
    max_row = max(normalized_vector_rows, key=lambda row_data: float(row_data["mass_ratio_to_scalar_base"]))
    k_positive_rows = [row_data for row_data in normalized_vector_rows if int(row_data["k"]) > 0]
    max_k_positive_row = None if not k_positive_rows else max(k_positive_rows, key=lambda row_data: float(row_data["mass_ratio_to_scalar_base"]))

    tail_localized_solution_count_total = sum(
        1
        for scan_rows in tail_scan_rows.values()
        for row_data in scan_rows
        if row_data.get("localized_solution_found")
    )
    tail_integer_mode_count_total = sum(len(mode_rows) for mode_rows in tail_mode_rows.values())
    localized_ell_values = sorted(
        {
            int(row_data["ell"])
            for scan_rows in tail_scan_rows.values()
            for row_data in scan_rows
            if row_data.get("localized_solution_found")
        }
    )
    first_new_localized_ell = None if not localized_ell_values else int(localized_ell_values[0])
    available_k_values = sorted(
        {
            int(mode["k"])
            for mode_rows in tail_mode_rows.values()
            for mode in mode_rows
        }
    )
    maximum_detected_k = None if not available_k_values else int(max(available_k_values))
    maximum_detected_ell = None if not localized_ell_values else int(max(localized_ell_values))
    maximum_detected_ell_with_k_positive = None if not tail_mode_rows else max(
        (
            int(mode["ell"])
            for mode_rows in tail_mode_rows.values()
            for mode in mode_rows
            if int(mode["k"]) > 0
        ),
        default=None,
    )
    rebuilt_ceiling = float(max_row["mass_ratio_to_scalar_base"])
    higher_ell_gain_factor = rebuilt_ceiling / current_ceiling
    w_gap_factor = None if best_w is None else W_TARGET / float(best_w["ratio_value"])
    z_gap_factor = None if best_z is None else Z_TARGET / float(best_z["ratio_value"])
    w_anchor_pass = bool(best_w and best_w["passes_threshold"])
    z_anchor_pass = bool(best_z and best_z["passes_threshold"])
    mw_mz_ratio_pass = bool(best_pair and best_pair["mw_mz_ratio_relative_error"] <= PASS_THRESHOLD)
    sin2_theta_w_pass = bool(best_pair and best_pair["sin2_theta_w_relative_error"] <= PASS_THRESHOLD)
    trial3_recommended_condition_satisfied = bool(w_anchor_pass and z_anchor_pass and mw_mz_ratio_pass and sin2_theta_w_pass)
    higher_ell_frontier_reopened = bool(tail_localized_solution_count_total > 0 and maximum_detected_ell is not None and maximum_detected_ell > 24)
    higher_ell_frontier_with_k_positive = bool(maximum_detected_ell_with_k_positive is not None and maximum_detected_ell_with_k_positive > 24)
    ceiling_improved = bool(rebuilt_ceiling > current_ceiling)
    remaining_anchor_gap_dominant = bool(
        not trial3_recommended_condition_satisfied
        and best_pair is not None
        and best_pair["mw_mz_ratio_relative_error"] <= prior_audit["summary"]["best_pair_or_none"]["mw_mz_ratio_relative_error"]
    )

    selected_residual_route = None
    missing_v2_artifact = None
    if not trial3_recommended_condition_satisfied:
        if higher_ell_frontier_reopened:
            selected_residual_route = f"trial3_relaunched_refactored_post_ell{maximum_detected_ell}_same_family_reaudit_identification"
            missing_v2_artifact = f"trial3_relaunched_refactored_post_ell{maximum_detected_ell}_same_family_closeout_pack"
        else:
            selected_residual_route = "trial3_relaunched_refactored_post_ell24_higher_ell_frontier_extension_identification"
            missing_v2_artifact = "trial3_relaunched_refactored_same_family_localized_exact_family_table_above_ell_24"

    audit = helper.payload(
        "8.7.56.306",
        "Trial-3 refactored post-ell24 higher-ell frontier extension audit",
        source_inventory["inputs"],
        "Rebuild the same-family exact table after explicitly extending the localized frontier above ell=24 and freeze whether the new tail family changes the weak-sector closeout state.",
        {
            "frontier_extension_rule": "if the tail scan reopens localized same-family sectors above ell=24 under the preserved normalization and radial contract, the blocker moves from frontier existence to a deeper same-family re-audit at the new ell ceiling",
            "closeout_rule": "Trial-3 closes only if the rebuilt same-family family crosses both W/Z anchors and the pair-side observables together",
        },
        [
            helper.row("trial3_refactored_post_ell24_higher_ell_frontier_extension_audit_complete", "pass", "Trial-3 refactored post-ell24 higher-ell frontier extension audit complete", 1, "The higher-ell frontier audit is frozen."),
            helper.row("trial3_refactored_post_ell24_tail_localized_solution_count", "pass" if tail_localized_solution_count_total > 0 else "reject", "localized solution count above ell=24", tail_localized_solution_count_total, "The frontier extension succeeds only if localized same-family sectors actually reopen above ell=24."),
            helper.row("trial3_refactored_post_ell24_tail_integer_mode_count", "pass" if tail_integer_mode_count_total > 0 else "reject", "integer mode count above ell=24", tail_integer_mode_count_total, "Localized sectors above ell=24 must also interpolate to integer modes before they can alter the exact-family table."),
            helper.row("trial3_refactored_post_ell24_frontier_reopened", "pass" if higher_ell_frontier_reopened else "reject", "frontier reopened above ell=24", 1 if higher_ell_frontier_reopened else 0, "The higher-ell route is honest only if the rebuilt same-family frontier actually moves above ell=24."),
            helper.row("trial3_refactored_post_ell24_rebuilt_ceiling_improved", "pass" if ceiling_improved else "reject", "rebuilt same-family ceiling improves beyond ell24 cap", 1 if ceiling_improved else 0, "The tail extension must raise the same-family ceiling beyond the ell<=24 rebuild."),
            helper.row("trial3_refactored_post_ell24_frontier_with_k_positive", "pass" if higher_ell_frontier_with_k_positive else "reject", "higher-ell frontier carries explicit k>0 support", 1 if higher_ell_frontier_with_k_positive else 0, "The refactored solver route remains strongest when the reopened frontier also carries explicit k>0 states."),
            helper.row("trial3_refactored_post_ell24_w_anchor_pass", "pass" if w_anchor_pass else "reject", "W/electron anchor passes after higher-ell frontier extension", 1 if w_anchor_pass else 0, "The higher-ell frontier closes Trial-3 only if it reaches the W scale."),
            helper.row("trial3_refactored_post_ell24_z_anchor_pass", "pass" if z_anchor_pass else "reject", "Z/electron anchor passes after higher-ell frontier extension", 1 if z_anchor_pass else 0, "The higher-ell frontier must also reach the Z scale."),
            helper.row("trial3_refactored_post_ell24_mw_mz_ratio_pass", "pass" if mw_mz_ratio_pass else "reject", "M_W/M_Z ratio passes after higher-ell frontier extension", 1 if mw_mz_ratio_pass else 0, "The same-family W/Z pair must remain viable after the tail extension."),
            helper.row("trial3_refactored_post_ell24_sin2_theta_w_pass", "pass" if sin2_theta_w_pass else "reject", "sin^2(theta_W) passes after higher-ell frontier extension", 1 if sin2_theta_w_pass else 0, "The Weinberg-angle proxy must close together with the W/Z pair."),
        ],
        {
            "normalization_scale_factor": normalization_scale,
            "prior_rebuilt_verified_ceiling_to_electron": current_ceiling,
            "rebuilt_verified_ceiling_to_electron": rebuilt_ceiling,
            "refactored_k_positive_ceiling_to_electron": None if max_k_positive_row is None else float(max_k_positive_row["mass_ratio_to_scalar_base"]),
            "tail_localized_solution_count_total": tail_localized_solution_count_total,
            "tail_integer_mode_count_total": tail_integer_mode_count_total,
            "localized_ell_values": localized_ell_values,
            "first_new_localized_ell_or_none": first_new_localized_ell,
            "available_k_values": available_k_values,
            "maximum_detected_k": maximum_detected_k,
            "maximum_detected_ell": maximum_detected_ell,
            "maximum_detected_ell_with_k_positive": maximum_detected_ell_with_k_positive,
            "best_w_row_or_none": best_w,
            "best_z_row_or_none": best_z,
            "best_pair_or_none": best_pair,
            "w_gap_factor_or_none": w_gap_factor,
            "z_gap_factor_or_none": z_gap_factor,
            "trial3_recommended_condition_satisfied": trial3_recommended_condition_satisfied,
            "remaining_anchor_gap_dominant": remaining_anchor_gap_dominant,
            "first_route_to_close_or_none": "trial3_refactored_declaration_eighth_gate",
        },
        {
            "overall_status": "trial3_refactored_post_ell24_higher_ell_frontier_audited",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_307": True,
            "next_required_artifacts": ["trial3_refactored_declaration_eighth_gate"],
        },
        {
            "low_ell_sector_summary": {
                str(ell): helper.sector_summary(
                    ell_scan_rows[int(ell)],
                    [mode for mode in low_ell_base_modes if int(mode["ell"]) == int(ell)],
                )
                for ell in radial.LOW_ELL_VALUES
            },
            "standard_extension_sector_summary": {
                str(ell): helper.sector_summary(standard_scan_rows[int(ell)], standard_mode_rows[int(ell)])
                for ell in radial.STANDARD_EXTENSION_ELLS
            },
            "control_ell_sector_summary": helper.sector_summary(control_scan_rows, control_mode_rows),
            "primary_post_ell19_sector_summary": {
                str(ell): helper.sector_summary(primary_scan_rows[int(ell)], primary_mode_rows[int(ell)])
                for ell in radial.PRIMARY_POST_ELL19_VALUES
            },
            "tail_post_ell24_sector_summary": {
                str(ell): helper.sector_summary(tail_scan_rows[int(ell)], tail_mode_rows[int(ell)])
                for ell in TAIL_POST_ELL24_VALUES
            },
            "prior_higher_audit_summary": prior_audit["summary"],
            "prior_radial_audit_summary": prior_radial_audit["summary"],
        },
    )

    declaration = helper.payload(
        "8.7.56.307",
        "Trial-3 refactored declaration eighth gate",
        source_inventory["inputs"],
        "Freeze whether the higher-ell frontier extension already closes Trial-3 or whether the next honest route is a same-family re-audit at the new ell ceiling.",
        {
            "closeout_rule": "Trial-3 closes only if the rebuilt same-family family simultaneously closes W, Z, M_W/M_Z, and sin^2(theta_W)",
            "residual_rule": "if the frontier moves above ell=24 but the weak-sector observables remain open, the next honest route is a same-family re-audit at the new ell ceiling rather than another generic frontier-existence audit",
        },
        [
            helper.row("trial3_refactored_declaration_eighth_gate_complete", "pass", "Trial-3 refactored declaration eighth gate complete", 1, "The eighth declaration gate is frozen."),
            helper.row("trial3_refactored_eighth_branch_closeable", "pass" if trial3_recommended_condition_satisfied else "reject", "refactored Trial-3 branch closeable after higher-ell frontier extension", 1 if trial3_recommended_condition_satisfied else 0, "The branch closes only if the rebuilt same-family family already closes the weak-sector pack."),
            helper.row("trial3_refactored_eighth_residual_route_required", "reject" if trial3_recommended_condition_satisfied else "pass", "refactored Trial-3 residual route required after higher-ell frontier extension", 0 if trial3_recommended_condition_satisfied else 1, "A new residual route is still required while the same-family weak-sector pack remains open."),
            helper.row("trial3_refactored_execute_trial2_paper_sync_now_eighth_gate", "reject", "execute Trial-2 paper-side sync now after higher-ell frontier extension", 0, "Trial-2 paper-side sync remains reserve work while the current Trial-3 same-family route is still live."),
        ],
        {
            "trial3_recommended_condition_satisfied": trial3_recommended_condition_satisfied,
            "trial3_current_branch_closeable": trial3_recommended_condition_satisfied,
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "recommended_next_route_or_none": None if trial3_recommended_condition_satisfied else "8.7.56.309",
        },
        {
            "overall_status": "trial3_refactored_declaration_eighth_gate_frozen",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_308": True,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["trial3_refactored_trial2_paper_sync_trial4_disposition_twentieth_refresh"],
        },
        {
            "audit_summary": audit["summary"],
            "prior_declaration_summary": prior_declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
        },
    )

    disposition = helper.payload(
        "8.7.56.308",
        "Trial-2 paper-side sync / Trial-4 disposition twentieth refresh",
        source_inventory["inputs"],
        "Refresh the reserve/deferred ordering after the higher-ell frontier extension and freeze the next official same-family re-audit route.",
        {
            "trial2_rule": "Trial-2 paper-side sync stays unlocked reserve retained while the refactored Trial-3 route still has an honest same-family closeout path",
            "trial4_rule": "Trial-4 remains deferred while Trial-3 still exposes a current-canon same-family weak-sector route",
        },
        [
            helper.row("trial3_refactored_trial2_trial4_twentieth_refresh_complete", "pass", "Trial-2 paper-side sync / Trial-4 disposition twentieth refresh complete", 1, "The reserve/deferred ordering is refreshed after the higher-ell frontier extension."),
            helper.row("trial3_refactored_trial2_paper_side_sync_reserve_retained_twentieth_refresh", "pass", "Trial-2 paper-side sync reserve retained", 1, "Trial-2 paper-side sync stays unlocked reserve work while the same-family re-audit route is still open."),
            helper.row("trial3_refactored_trial4_deferred_retained_twentieth_refresh", "pass", "Trial-4 deferred retained", 1, "Trial-4 stays deferred while Trial-3 still has an honest same-family weak-sector path."),
        ],
        {
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "trial2_paper_side_sync_state": "unlocked_reserve_retained",
            "trial4_deferred": True,
            "recommended_next_route_or_none": None if trial3_recommended_condition_satisfied else "8.7.56.309",
        },
        {
            "overall_status": "trial3_refactored_trial2_trial4_twentieth_disposition_refreshed",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_13": False,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["8.7.56.309"],
        },
        {
            "declaration_summary": declaration["summary"],
            "audit_summary": audit["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
        },
    )

    helper.write_artifact("mass_origin_v2_trial3_refactored_post_ell24_higher_ell_frontier_extension_source_inventory", source_inventory)
    helper.write_artifact("mass_origin_v2_trial3_refactored_post_ell24_higher_ell_frontier_extension_audit", audit)
    helper.write_artifact("mass_origin_v2_trial3_refactored_declaration_eighth_gate", declaration)
    helper.write_artifact("mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_twentieth_refresh", disposition)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_refactored_post_ell24_higher_ell_frontier_extension_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_post_ell24_higher_ell_frontier_extension_audit_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_declaration_eighth_gate_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_twentieth_refresh_metrics.json")


# 関数: CLI から post-ell24 higher-ell frontier branch を起動する。

if __name__ == "__main__":
    main()
