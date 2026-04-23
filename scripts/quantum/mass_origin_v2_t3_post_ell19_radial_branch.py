#!/usr/bin/env python3
"""
Generate Trial-3 refactored post-ell19 radial-domain artifacts for 8.7.56.289-.292.

The post-ell18 central-amplitude extension already reopened ell=19 and improved
the same-family weak-sector ceiling. The next honest question is narrower:
does the current radial integration contract still suppress higher-ell
localized families above ell=19 even after the amplitude domain was widened?
"""

from __future__ import annotations

import importlib.util
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

POST_PHOTON_PRESERVATION = OUT / "mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_metrics.json"
TRIAL3_REFACTORED_AUDIT = OUT / "mass_origin_v2_trial3_refactored_high_mass_k_positive_extension_audit_metrics.json"
TRIAL3_REFACTORED_GATE = OUT / "mass_origin_v2_trial3_refactored_declaration_gate_metrics.json"
TRIAL3_REFACTORED_DISPOSITION = OUT / "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_thirteenth_refresh_metrics.json"
TRIAL3_WINDOW_AUDIT = OUT / "mass_origin_v2_trial3_refactored_post_ell18_localization_window_extension_audit_metrics.json"
TRIAL3_WINDOW_GATE = OUT / "mass_origin_v2_trial3_refactored_declaration_second_gate_metrics.json"
TRIAL3_WINDOW_DISPOSITION = OUT / "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_fourteenth_refresh_metrics.json"
TRIAL3_AMPLITUDE_SOURCE = OUT / "mass_origin_v2_trial3_refactored_post_ell18_central_amplitude_window_extension_source_inventory_metrics.json"
TRIAL3_AMPLITUDE_AUDIT = OUT / "mass_origin_v2_trial3_refactored_post_ell18_central_amplitude_window_extension_audit_metrics.json"
TRIAL3_AMPLITUDE_GATE = OUT / "mass_origin_v2_trial3_refactored_declaration_third_gate_metrics.json"
TRIAL3_AMPLITUDE_DISPOSITION = OUT / "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_fifteenth_refresh_metrics.json"
TRIAL3_SOLVER_REFACTOR_EXECUTION = OUT / "mass_origin_v2_trial3_solver_refactor_execution_audit_metrics.json"
TRIAL3_SOLVER_REFACTOR_WEAK = OUT / "mass_origin_v2_trial3_solver_refactor_weak_sector_reaudit_metrics.json"
VECTOR_SPIN = OUT / "mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json"
SCALAR_SPECTRUM = OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"
EXACT_HANDOFF = OUT / "mass_origin_vector_qball_exact_mass_table_handoff_retry_metrics.json"

NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_COUPLED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"
EXTENDED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_extended_hierarchy_branch.py"
PREVIOUS_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_t3_post_ell18_amplitude_branch.py"

W_MASS_MEV = 80369.0
Z_MASS_MEV = 91187.6
ELECTRON_MASS_MEV = 0.51099895
W_TARGET = W_MASS_MEV / ELECTRON_MASS_MEV
Z_TARGET = Z_MASS_MEV / ELECTRON_MASS_MEV
PASS_THRESHOLD = 0.10

LOW_ELL_VALUES = (1, 2, 3)
STANDARD_EXTENSION_ELLS = tuple(range(4, 19))
CONTROL_ELL = 19
PRIMARY_POST_ELL19_VALUES = tuple(range(20, 25))
PRIMARY_WIDENED_BETA_GRID = (
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.08,
    0.10,
    0.12,
    0.16,
    0.20,
    0.26,
    0.34,
    0.44,
    0.56,
    0.70,
    0.84,
    0.94,
    0.98,
)
PRIMARY_EXTENDED_AMPLITUDES = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0)
CURRENT_RADIAL_CONTRACT = {
    "r0": 1.0e-4,
    "r_max": 30.0,
    "max_step": 0.05,
    "rtol": 1.0e-7,
    "atol": 1.0e-9,
}
PRIMARY_EXTENDED_RADIAL_CONTRACT = {
    "r0": 1.0e-4,
    "r_max": 60.0,
    "max_step": 0.03,
    "rtol": 1.0e-7,
    "atol": 1.0e-9,
}


# 関数: local Python module を動的に読む。
def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 関数: radial contract を差し替えた profile solver を実行する。

def solve_sector_profile_with_radial_contract(
    numerical,
    beta: float,
    amp: float,
    ell: int,
    radial_contract: dict[str, float],
) -> dict:
    r0 = float(radial_contract["r0"])
    r_max = float(radial_contract["r_max"])
    max_step = float(radial_contract["max_step"])
    rtol = float(radial_contract["rtol"])
    atol = float(radial_contract["atol"])
    if ell == 0:
        y0 = [float(amp), 0.0]
    else:
        y0 = [float(amp) * (r0**ell), float(amp) * ell * (r0 ** (ell - 1))]

    # 関数: radial contract を差し替えた effective single-profile ODE を返す。

    def ode(radius: float, y: np.ndarray) -> list[float]:
        field, field_prime = float(y[0]), float(y[1])
        damping = 2.0 * field_prime / radius if radius > 0.0 else 0.0
        barrier = ell * (ell + 1.0) * field / (radius * radius) if radius > 0.0 else 0.0
        field_double_prime = -damping + barrier - (beta * beta - 1.0) * field - 3.0 * field * field - field**3
        return [field_prime, field_double_prime]

    sol = solve_ivp(ode, (r0, r_max), y0, max_step=max_step, rtol=rtol, atol=atol)
    radius = sol.t
    field = sol.y[0]
    field_prime = sol.y[1]
    energy_density = (
        0.5 * field_prime * field_prime
        + 0.5 * (1.0 + beta * beta) * field * field
        + field**3
        + 0.25 * field**4
        + 0.5 * ell * (ell + 1.0) * field * field / np.maximum(radius * radius, 1.0e-12)
    )
    charge_proxy = float(beta * np.trapezoid(4.0 * math.pi * radius * radius * field * field, radius))
    energy_proxy = float(np.trapezoid(4.0 * math.pi * radius * radius * energy_density, radius))
    return {
        "tail": float(field[-1]),
        "tail_abs": float(abs(field[-1])),
        "charge_proxy": charge_proxy,
        "energy_proxy": energy_proxy,
        "central_amplitude": float(amp),
        "field_min": float(np.min(field)),
        "field_max": float(np.max(field)),
        "node_count_k": int(numerical.count_radial_nodes(field)),
        "radius_values": radius.tolist(),
        "field_values": field.tolist(),
    }


# 関数: extended radial contract で localized sector scan を実行する。

def scan_radial_extended_sector(
    numerical,
    helper,
    ell: int,
    beta_grid: tuple[float, ...],
    extra_amplitudes: tuple[float, ...],
    radial_contract: dict[str, float],
) -> tuple[list[dict], list[float]]:
    amplitude_values = helper.extended_amplitude_grid(numerical, int(ell), extra_amplitudes)

    # 関数: branch 内で同一 radial contract の profile を cache する。
    @lru_cache(maxsize=None)
    def cached_profile(beta_value: float, amp_value: float) -> dict:
        return solve_sector_profile_with_radial_contract(
            numerical,
            float(beta_value),
            float(amp_value),
            int(ell),
            radial_contract,
        )

    # 関数: branch 内で同一 radial contract の tail を cache する。

    @lru_cache(maxsize=None)
    def cached_tail(beta_value: float, amp_value: float) -> float:
        return float(cached_profile(float(beta_value), float(amp_value))["tail"])

    rows = []
    for beta in beta_grid:
        tails = []
        for amp in amplitude_values:
            try:
                tails.append(cached_tail(float(beta), float(amp)))
            except Exception:
                tails.append(float("nan"))

        candidates: list[dict] = []
        for amp_left, amp_right, tail_left, tail_right in zip(
            amplitude_values[:-1],
            amplitude_values[1:],
            tails[:-1],
            tails[1:],
        ):
            if not math.isfinite(tail_left) or not math.isfinite(tail_right):
                continue

            if tail_left == 0.0:
                root_amp = float(amp_left)
            elif tail_right == 0.0:
                root_amp = float(amp_right)
            elif tail_left * tail_right < 0.0:
                root_amp = float(
                    brentq(
                        lambda amp: cached_tail(float(beta), float(amp)),
                        float(amp_left),
                        float(amp_right),
                        maxiter=80,
                    )
                )
            else:
                continue

            solved = cached_profile(float(beta), float(root_amp))
            candidates.append(
                {
                    "central_amplitude": float(root_amp),
                    "profile": solved,
                    "node_count_k": int(solved["node_count_k"]),
                    "tail_abs": float(solved["tail_abs"]),
                }
            )

        best_by_k: dict[int, dict] = {}
        for candidate in candidates:
            k_value = int(candidate["node_count_k"])
            previous = best_by_k.get(k_value)
            if previous is None or float(candidate["tail_abs"]) < float(previous["tail_abs"]):
                best_by_k[k_value] = candidate

        localized_profiles = [best_by_k[k_value] for k_value in sorted(best_by_k)]
        if not localized_profiles:
            rows.append({"ell": int(ell), "beta": float(beta), "localized_solution_found": False})
            continue

        for branch_index, localized_profile in enumerate(localized_profiles, start=1):
            solved = localized_profile["profile"]
            rows.append(
                {
                    "ell": int(ell),
                    "beta": float(beta),
                    "localized_solution_found": True,
                    "central_amplitude": float(localized_profile["central_amplitude"]),
                    "charge_proxy": float(solved["charge_proxy"]),
                    "energy_proxy": float(solved["energy_proxy"]),
                    "tail_abs": float(solved["tail_abs"]),
                    "field_min": float(solved["field_min"]),
                    "field_max": float(solved["field_max"]),
                    "node_count_k": int(localized_profile["node_count_k"]),
                    "k": int(localized_profile["node_count_k"]),
                    "solution_branch_index": int(branch_index),
                }
            )

    return rows, amplitude_values


# 関数: Trial-3 refactored post-ell19 radial-domain branch を実行する。

def main() -> None:
    helper = load_module(PREVIOUS_BRANCH, "trial3_post_ell19_radial_helper")
    numerical = load_module(NUMERICAL_BRANCH, "trial3_post_ell19_radial_num")
    full = load_module(FULL_COUPLED_BRANCH, "trial3_post_ell19_radial_full")
    extended = load_module(EXTENDED_BRANCH, "trial3_post_ell19_radial_ext")

    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        POST_PHOTON_PRESERVATION,
        TRIAL3_REFACTORED_AUDIT,
        TRIAL3_REFACTORED_GATE,
        TRIAL3_REFACTORED_DISPOSITION,
        TRIAL3_WINDOW_AUDIT,
        TRIAL3_WINDOW_GATE,
        TRIAL3_WINDOW_DISPOSITION,
        TRIAL3_AMPLITUDE_SOURCE,
        TRIAL3_AMPLITUDE_AUDIT,
        TRIAL3_AMPLITUDE_GATE,
        TRIAL3_AMPLITUDE_DISPOSITION,
        TRIAL3_SOLVER_REFACTOR_EXECUTION,
        TRIAL3_SOLVER_REFACTOR_WEAK,
        VECTOR_SPIN,
        SCALAR_SPECTRUM,
        EXACT_HANDOFF,
        NUMERICAL_BRANCH,
        FULL_COUPLED_BRANCH,
        EXTENDED_BRANCH,
        PREVIOUS_BRANCH,
    ):
        helper.req(path)

    status_text = helper.read_text(STATUS)
    roadmap_text = helper.read_text(ROADMAP)
    ai_context = helper.read_json(AI_CONTEXT)
    post_photon = helper.read_json(POST_PHOTON_PRESERVATION)
    prior_refactored_audit = helper.read_json(TRIAL3_REFACTORED_AUDIT)
    prior_window_audit = helper.read_json(TRIAL3_WINDOW_AUDIT)
    prior_window_gate = helper.read_json(TRIAL3_WINDOW_GATE)
    prior_window_disposition = helper.read_json(TRIAL3_WINDOW_DISPOSITION)
    amplitude_source = helper.read_json(TRIAL3_AMPLITUDE_SOURCE)
    amplitude_audit = helper.read_json(TRIAL3_AMPLITUDE_AUDIT)
    amplitude_gate = helper.read_json(TRIAL3_AMPLITUDE_GATE)
    amplitude_disposition = helper.read_json(TRIAL3_AMPLITUDE_DISPOSITION)
    solver_refactor_execution = helper.read_json(TRIAL3_SOLVER_REFACTOR_EXECUTION)
    solver_refactor_weak = helper.read_json(TRIAL3_SOLVER_REFACTOR_WEAK)
    vector_spin = helper.read_json(VECTOR_SPIN)
    scalar_spectrum = helper.read_json(SCALAR_SPECTRUM)
    exact_handoff = helper.read_json(EXACT_HANDOFF)

    numerical_text = helper.read_text(NUMERICAL_BRANCH)
    full_text = helper.read_text(FULL_COUPLED_BRANCH)
    previous_branch_text = helper.read_text(PREVIOUS_BRANCH)

    scalar_modes = list(scalar_spectrum["evidence"]["discrete_mass_mode_rows"])
    lambda_rot = float(vector_spin["summary"]["lambda_rot_value"])
    normalization_scale = float(post_photon["summary"]["absolute_mass_normalization_scale_factor"])
    prior_ceiling = float(amplitude_audit["summary"]["rebuilt_verified_ceiling_to_electron"])
    prior_w_gap_factor = float(amplitude_audit["summary"]["w_gap_factor_or_none"])
    prior_z_gap_factor = float(amplitude_audit["summary"]["z_gap_factor_or_none"])
    zero_above_19 = all(
        int(amplitude_audit["evidence"]["primary_post_ell18_sector_summary"][str(ell)]["localized_solution_count"]) == 0
        for ell in PRIMARY_POST_ELL19_VALUES
    )

    inventory_targets = [
        {
            "label": "status_points_to_8_7_56_289",
            "present": "current official next step は `8.7.56.289`" in status_text,
            "evidence": {"status_markdown": helper.rel(STATUS)},
        },
        {
            "label": "roadmap_post_ell19_radial_branch_present",
            "present": "`8.7.56.289-.292` 試練3 refactored post-`ell=19` radial-domain extension residual branch" in roadmap_text,
            "evidence": {"roadmap_markdown": helper.rel(ROADMAP)},
        },
        {
            "label": "prior_ell19_reopening_fixed",
            "present": int(amplitude_audit["summary"]["first_localized_ell_or_none"]) == 19,
            "evidence": {
                "localized_ell_values": amplitude_audit["summary"]["localized_ell_values"],
                "post_ell18_localized_solution_count_total": amplitude_audit["summary"]["post_ell18_localized_solution_count_total"],
            },
        },
        {
            "label": "prior_ell20_plus_zero_localization_fixed",
            "present": zero_above_19,
            "evidence": {
                "primary_post_ell18_sector_summary": amplitude_audit["evidence"]["primary_post_ell18_sector_summary"],
            },
        },
        {
            "label": "numerical_solve_sector_profile_present",
            "present": helper.hit(numerical_text, "def solve_sector_profile(beta: float, amp: float, ell: int) -> dict:") is not None,
            "evidence": helper.hit(numerical_text, "def solve_sector_profile(beta: float, amp: float, ell: int) -> dict:"),
        },
        {
            "label": "numerical_current_radial_contract_present",
            "present": helper.hit(numerical_text, "sol = solve_ivp(ode, (r0, 30.0), y0, max_step=0.05, rtol=1.0e-7, atol=1.0e-9)") is not None,
            "evidence": helper.hit(numerical_text, "sol = solve_ivp(ode, (r0, 30.0), y0, max_step=0.05, rtol=1.0e-7, atol=1.0e-9)"),
        },
        {
            "label": "numerical_find_sector_amplitudes_present",
            "present": helper.hit(numerical_text, "def find_sector_amplitudes(beta: float, ell: int) -> list[dict]:") is not None,
            "evidence": helper.hit(numerical_text, "def find_sector_amplitudes(beta: float, ell: int) -> list[dict]:"),
        },
        {
            "label": "previous_amplitude_branch_present",
            "present": helper.hit(previous_branch_text, "PRIMARY_EXTENDED_AMPLITUDES = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0)") is not None,
            "evidence": helper.hit(previous_branch_text, "PRIMARY_EXTENDED_AMPLITUDES = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0)"),
        },
        {
            "label": "full_build_exact_ladder_present",
            "present": helper.hit(full_text, "def build_exact_ladder(") is not None,
            "evidence": helper.hit(full_text, "def build_exact_ladder("),
        },
        {
            "label": "same_family_weak_target_pack_present",
            "present": bool(exact_handoff["summary"]["hand_off_to_8_7_55_2_84"]),
            "evidence": {
                "best_exact_match_or_none": exact_handoff["summary"]["best_exact_match_or_none"],
                "w_target_to_electron": W_TARGET,
                "z_target_to_electron": Z_TARGET,
            },
        },
    ]

    source_inventory = helper.payload(
        "8.7.56.289",
        "Trial-3 refactored post-ell19 radial-domain extension source inventory",
        {
            "status_markdown": helper.rel(STATUS),
            "roadmap_markdown": helper.rel(ROADMAP),
            "ai_context_json": helper.rel(AI_CONTEXT),
            "mass_origin_v2_post_photon_vector_mass_ratio_preservation_audit_json": helper.rel(POST_PHOTON_PRESERVATION),
            "mass_origin_v2_trial3_refactored_high_mass_k_positive_extension_audit_json": helper.rel(TRIAL3_REFACTORED_AUDIT),
            "mass_origin_v2_trial3_refactored_post_ell18_localization_window_extension_audit_json": helper.rel(TRIAL3_WINDOW_AUDIT),
            "mass_origin_v2_trial3_refactored_declaration_second_gate_json": helper.rel(TRIAL3_WINDOW_GATE),
            "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_fourteenth_refresh_json": helper.rel(TRIAL3_WINDOW_DISPOSITION),
            "mass_origin_v2_trial3_refactored_post_ell18_central_amplitude_window_extension_source_inventory_json": helper.rel(TRIAL3_AMPLITUDE_SOURCE),
            "mass_origin_v2_trial3_refactored_post_ell18_central_amplitude_window_extension_audit_json": helper.rel(TRIAL3_AMPLITUDE_AUDIT),
            "mass_origin_v2_trial3_refactored_declaration_third_gate_json": helper.rel(TRIAL3_AMPLITUDE_GATE),
            "mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_fifteenth_refresh_json": helper.rel(TRIAL3_AMPLITUDE_DISPOSITION),
            "mass_origin_v2_trial3_solver_refactor_execution_audit_json": helper.rel(TRIAL3_SOLVER_REFACTOR_EXECUTION),
            "mass_origin_v2_trial3_solver_refactor_weak_sector_reaudit_json": helper.rel(TRIAL3_SOLVER_REFACTOR_WEAK),
            "mass_origin_vector_qball_spin_orbit_freeze_audit_json": helper.rel(VECTOR_SPIN),
            "mass_origin_qball_discrete_mass_spectrum_json": helper.rel(SCALAR_SPECTRUM),
            "mass_origin_vector_qball_exact_mass_table_handoff_retry_json": helper.rel(EXACT_HANDOFF),
            "mass_origin_vector_qball_numerical_solver_branch_py": helper.rel(NUMERICAL_BRANCH),
            "mass_origin_vector_qball_full_coupled_solver_branch_py": helper.rel(FULL_COUPLED_BRANCH),
            "mass_origin_vector_qball_extended_hierarchy_branch_py": helper.rel(EXTENDED_BRANCH),
            "mass_origin_v2_t3_post_ell18_amplitude_branch_py": helper.rel(PREVIOUS_BRANCH),
        },
        "Freeze the current radial integration contract and the ell=19 reopening evidence before reopening the same-family weak-sector route with an explicit post-ell19 radial-domain extension.",
        {
            "inventory_rule": "the source pack must contain the current radial-grid contract, the ell=19 reopening evidence, the ell>=20 zero-localization evidence, the preserved exact-family rebuild, and the same-family weak targets before the radial-domain audit can run honestly",
            "current_radial_contract_rule": "the current solver integrates from r0=1e-4 to r=30 with max_step=0.05, so the radial-domain audit must preserve the amplitude extension and change the radial span/density instead of modifying beta or amplitude again",
            "primary_radial_extension_rule": f"for ell in {list(PRIMARY_POST_ELL19_VALUES)} keep the widened beta grid and high-amplitude tranche fixed, then extend the radial contract to {PRIMARY_EXTENDED_RADIAL_CONTRACT}",
        },
        [
            helper.row("trial3_refactored_post_ell19_radial_domain_source_inventory_complete", "pass", "Trial-3 refactored post-ell19 radial-domain source inventory complete", 1, "The post-ell19 radial-domain source pack is frozen."),
            helper.row("trial3_refactored_post_ell19_radial_domain_required_source_count", "pass" if all(item["present"] for item in inventory_targets) else "reject", "required post-ell19 radial-domain source count", len(inventory_targets), "The radial-domain audit needs the current radial-grid contract, the ell=19 reopening evidence, the ell>=20 zero-localization evidence, the preserved exact-family rebuild, and the same-family W/Z targets in one source pack."),
            helper.row("trial3_refactored_current_solver_r_max", "pass", "current solver outer radius cutoff", CURRENT_RADIAL_CONTRACT["r_max"], "The current solver stops the radial integration at r=30 before the radial-domain extension is applied."),
            helper.row("trial3_refactored_current_solver_max_step", "pass", "current solver maximum radial step", CURRENT_RADIAL_CONTRACT["max_step"], "The current solver uses max_step=0.05 before the radial-domain extension is applied."),
            helper.row("trial3_refactored_prior_ell19_rebuilt_ceiling_to_electron", "pass", "prior ell19 rebuilt ceiling to electron", prior_ceiling, "The radial-domain audit starts from the ell=19 reopening already fixed by the amplitude extension."),
        ],
        {
            "required_source_count": len(inventory_targets),
            "required_source_count_present": sum(1 for item in inventory_targets if item["present"]),
            "inventory_ready": bool(all(item["present"] for item in inventory_targets)),
            "current_radial_contract": CURRENT_RADIAL_CONTRACT,
            "primary_extended_radial_contract": PRIMARY_EXTENDED_RADIAL_CONTRACT,
            "control_ell_value": CONTROL_ELL,
            "primary_post_ell19_values": list(PRIMARY_POST_ELL19_VALUES),
            "primary_widened_beta_grid": list(PRIMARY_WIDENED_BETA_GRID),
            "primary_extended_amplitudes": list(PRIMARY_EXTENDED_AMPLITUDES),
            "historic_preserved_verified_ceiling_to_electron": float(prior_window_audit["summary"]["historic_preserved_verified_ceiling_to_electron"]),
            "prior_rebuilt_verified_ceiling_to_electron": prior_ceiling,
            "prior_w_gap_factor": prior_w_gap_factor,
            "prior_z_gap_factor": prior_z_gap_factor,
            "first_route_to_close_or_none": "trial3_refactored_post_ell19_radial_domain_extension_audit",
        },
        {
            "overall_status": "trial3_refactored_post_ell19_radial_domain_source_inventory_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_290": True,
            "next_required_artifacts": ["trial3_refactored_post_ell19_radial_domain_extension_audit"],
        },
        {
            "inventory_targets": inventory_targets,
            "prior_window_audit_summary": prior_window_audit["summary"],
            "prior_window_gate_summary": prior_window_gate["summary"],
            "prior_window_disposition_summary": prior_window_disposition["summary"],
            "amplitude_source_summary": amplitude_source["summary"],
            "amplitude_audit_summary": amplitude_audit["summary"],
            "amplitude_gate_summary": amplitude_gate["summary"],
            "amplitude_disposition_summary": amplitude_disposition["summary"],
            "solver_refactor_execution_summary": solver_refactor_execution["summary"],
            "solver_refactor_weak_summary": solver_refactor_weak["summary"],
            "prior_refactored_audit_summary": prior_refactored_audit["summary"],
            "status_current_step_before_branch": ai_context["current_step"],
        },
    )

    ell_scan_rows, low_ell_base_modes = numerical.build_base_modes(LOW_ELL_VALUES)
    base_modes = list(low_ell_base_modes)

    standard_scan_rows = {}
    standard_mode_rows = {}
    for ell in STANDARD_EXTENSION_ELLS:
        scan_rows = extended.scan_extended_sector(numerical, int(ell))
        mode_rows = numerical.interpolate_integer_modes(scan_rows, int(ell))
        standard_scan_rows[int(ell)] = scan_rows
        standard_mode_rows[int(ell)] = mode_rows
        base_modes.extend(mode_rows)

    control_scan_rows, control_amplitude_values = helper.scan_amplitude_extended_sector(
        numerical,
        CONTROL_ELL,
        PRIMARY_WIDENED_BETA_GRID,
        PRIMARY_EXTENDED_AMPLITUDES,
    )
    control_mode_rows = numerical.interpolate_integer_modes(control_scan_rows, CONTROL_ELL)
    base_modes.extend(control_mode_rows)

    primary_scan_rows = {}
    primary_mode_rows = {}
    primary_amplitude_grid_map = {}
    for ell in PRIMARY_POST_ELL19_VALUES:
        scan_rows, amplitude_values = scan_radial_extended_sector(
            numerical,
            helper,
            int(ell),
            PRIMARY_WIDENED_BETA_GRID,
            PRIMARY_EXTENDED_AMPLITUDES,
            PRIMARY_EXTENDED_RADIAL_CONTRACT,
        )
        mode_rows = numerical.interpolate_integer_modes(scan_rows, int(ell))
        primary_scan_rows[int(ell)] = scan_rows
        primary_mode_rows[int(ell)] = mode_rows
        primary_amplitude_grid_map[int(ell)] = amplitude_values
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
    max_k_positive_row = max(k_positive_rows, key=lambda row_data: float(row_data["mass_ratio_to_scalar_base"]))

    ell_mode_summary = {}
    ell_mode_summary.update(
        {
            str(ell): helper.sector_summary(
                ell_scan_rows[int(ell)],
                [mode for mode in low_ell_base_modes if int(mode["ell"]) == int(ell)],
            )
            for ell in LOW_ELL_VALUES
        }
    )
    ell_mode_summary.update(
        {str(ell): helper.sector_summary(standard_scan_rows[int(ell)], standard_mode_rows[int(ell)]) for ell in STANDARD_EXTENSION_ELLS}
    )
    ell_mode_summary[str(CONTROL_ELL)] = helper.sector_summary(control_scan_rows, control_mode_rows)
    ell_mode_summary.update(
        {str(ell): helper.sector_summary(primary_scan_rows[int(ell)], primary_mode_rows[int(ell)]) for ell in PRIMARY_POST_ELL19_VALUES}
    )

    localized_ell_values = sorted(
        {
            int(row_data["ell"])
            for scan_rows in primary_scan_rows.values()
            for row_data in scan_rows
            if row_data.get("localized_solution_found")
        }
    )
    first_localized_ell = None if not localized_ell_values else int(localized_ell_values[0])
    available_k_values = sorted({int(mode["k"]) for mode in base_modes})
    maximum_detected_k = max(available_k_values) if available_k_values else 0
    maximum_detected_ell = max(int(mode["ell"]) for mode in base_modes)
    maximum_detected_ell_with_k_positive = max(int(mode["ell"]) for mode in base_modes if int(mode["k"]) > 0)
    post_ell19_localized_solution_count_total = sum(
        len([row_data for row_data in scan_rows if row_data.get("localized_solution_found")])
        for scan_rows in primary_scan_rows.values()
    )
    post_ell19_integer_mode_count_total = sum(len(mode_rows) for mode_rows in primary_mode_rows.values())
    rebuilt_max_ratio = float(max_row["mass_ratio_to_scalar_base"])
    ceiling_improved_beyond_prior = rebuilt_max_ratio > prior_ceiling
    w_anchor_pass = bool(best_w and best_w["passes_threshold"])
    z_anchor_pass = bool(best_z and best_z["passes_threshold"])
    mw_mz_ratio_pass = bool(best_pair and best_pair["mw_mz_ratio_relative_error"] <= PASS_THRESHOLD)
    sin2_theta_w_pass = bool(best_pair and best_pair["sin2_theta_w_relative_error"] <= PASS_THRESHOLD)
    trial3_recommended_condition_satisfied = bool(w_anchor_pass and z_anchor_pass and mw_mz_ratio_pass and sin2_theta_w_pass)

    audit = helper.payload(
        "8.7.56.290",
        "Trial-3 refactored post-ell19 radial-domain extension audit",
        source_inventory["inputs"],
        "Rebuild the same-family exact table after explicitly extending the post-ell19 radial domain and freeze whether higher-ell localized families reopen above ell=19.",
        {
            "radial_extension_rule": "keep the refactored low-ell, standard high-ell, and ell=19 amplitude-reopened families fixed, then extend the post-ell19 radial domain above the default r=30 / max_step=0.05 contract while preserving the widened beta and amplitude grids",
            "localization_rule": "if the radial extension creates localized sectors above ell=19, the blocker is no longer the radial contract at ell=19 itself",
            "residual_rule": "if localization reopens only at ell=20 and still fails to close W/Z, the next blocker must move away from the post-ell19 radial contract and into the deeper post-ell20 search domain",
        },
        [
            helper.row("trial3_refactored_post_ell19_radial_domain_extension_audit_complete", "pass", "Trial-3 refactored post-ell19 radial-domain extension audit complete", 1, "The post-ell19 radial-domain audit is frozen."),
            helper.row("trial3_refactored_post_ell19_localized_solution_count", "pass" if post_ell19_localized_solution_count_total > 0 else "reject", "localized solution count above ell=19 under radial extension", post_ell19_localized_solution_count_total, "The radial-domain extension must create localized sectors above ell=19 before the same-family weak-sector route can move honestly."),
            helper.row("trial3_refactored_post_ell19_integer_mode_count", "pass" if post_ell19_integer_mode_count_total > 0 else "reject", "integer mode count above ell=19 under radial extension", post_ell19_integer_mode_count_total, "Localized sectors above ell=19 must also interpolate to integer modes before they can change the exact-family table."),
            helper.row("trial3_refactored_post_ell19_ceiling_improved", "pass" if ceiling_improved_beyond_prior else "reject", "rebuilt ceiling improves beyond ell19 amplitude ceiling", 1 if ceiling_improved_beyond_prior else 0, "A successful radial-domain extension should push the same-family ceiling beyond the ell=19 amplitude-opened row."),
            helper.row("trial3_refactored_post_ell19_w_anchor_pass", "pass" if w_anchor_pass else "reject", "W/electron anchor passes after radial extension", 1 if w_anchor_pass else 0, "The radial-domain extension closes Trial-3 only if it reaches the W scale."),
            helper.row("trial3_refactored_post_ell19_z_anchor_pass", "pass" if z_anchor_pass else "reject", "Z/electron anchor passes after radial extension", 1 if z_anchor_pass else 0, "The radial-domain extension must also reach the Z scale."),
            helper.row("trial3_refactored_post_ell19_mw_mz_ratio_pass", "pass" if mw_mz_ratio_pass else "reject", "M_W/M_Z ratio passes after radial extension", 1 if mw_mz_ratio_pass else 0, "The same-family W/Z pair must remain viable after the radial extension."),
            helper.row("trial3_refactored_post_ell19_sin2_theta_w_pass", "pass" if sin2_theta_w_pass else "reject", "sin^2(theta_W) passes after radial extension", 1 if sin2_theta_w_pass else 0, "The Weinberg-angle proxy must close together with the W/Z pair rather than remaining only partially improved."),
        ],
        {
            "normalization_scale_factor": normalization_scale,
            "historic_preserved_verified_ceiling_to_electron": float(prior_window_audit["summary"]["historic_preserved_verified_ceiling_to_electron"]),
            "prior_rebuilt_verified_ceiling_to_electron": prior_ceiling,
            "rebuilt_verified_ceiling_to_electron": rebuilt_max_ratio,
            "refactored_k_positive_ceiling_to_electron": float(max_k_positive_row["mass_ratio_to_scalar_base"]),
            "control_ell_localized_solution_count_total": len([row_data for row_data in control_scan_rows if row_data.get("localized_solution_found")]),
            "control_ell_integer_mode_count_total": len(control_mode_rows),
            "post_ell19_localized_solution_count_total": post_ell19_localized_solution_count_total,
            "post_ell19_integer_mode_count_total": post_ell19_integer_mode_count_total,
            "localized_ell_values": localized_ell_values,
            "first_localized_ell_or_none": first_localized_ell,
            "available_k_values": available_k_values,
            "maximum_detected_k": maximum_detected_k,
            "maximum_detected_ell": maximum_detected_ell,
            "maximum_detected_ell_with_k_positive": maximum_detected_ell_with_k_positive,
            "best_w_row_or_none": best_w,
            "best_z_row_or_none": best_z,
            "best_pair_or_none": best_pair,
            "w_gap_factor_or_none": None if best_w is None else W_TARGET / float(best_w["ratio_value"]),
            "z_gap_factor_or_none": None if best_z is None else Z_TARGET / float(best_z["ratio_value"]),
            "trial3_recommended_condition_satisfied": trial3_recommended_condition_satisfied,
            "first_route_to_close_or_none": "trial3_refactored_declaration_fourth_gate",
        },
        {
            "overall_status": "trial3_refactored_post_ell19_radial_domain_audited",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_291": True,
            "next_required_artifacts": ["trial3_refactored_declaration_fourth_gate"],
        },
        {
            "low_ell_sector_summary": {str(ell): ell_mode_summary[str(ell)] for ell in LOW_ELL_VALUES},
            "standard_extension_sector_summary": {str(ell): ell_mode_summary[str(ell)] for ell in STANDARD_EXTENSION_ELLS},
            "control_ell_summary": {str(CONTROL_ELL): ell_mode_summary[str(CONTROL_ELL)]},
            "primary_post_ell19_sector_summary": {str(ell): ell_mode_summary[str(ell)] for ell in PRIMARY_POST_ELL19_VALUES},
            "control_amplitude_grid": control_amplitude_values,
            "primary_amplitude_grid_map": {str(ell): primary_amplitude_grid_map[int(ell)] for ell in PRIMARY_POST_ELL19_VALUES},
            "primary_extended_radial_contract": PRIMARY_EXTENDED_RADIAL_CONTRACT,
            "sampled_high_mass_rows": helper.sample(sorted(normalized_vector_rows, key=lambda item: float(item["mass_ratio_to_scalar_base"]), reverse=True), 16),
            "max_row_or_none": max_row,
            "max_k_positive_row_or_none": max_k_positive_row,
            "prior_amplitude_audit_summary": amplitude_audit["summary"],
        },
    )

    if post_ell19_localized_solution_count_total == 0:
        selected_residual_route = "trial3_relaunched_refactored_post_ell19_inner_radius_resolution_identification"
        missing_v2_artifact = "trial3_relaunched_refactored_localized_exact_family_table_above_ell19_with_refined_inner_radius_resolution"
    elif localized_ell_values and max(localized_ell_values) == 20 and not trial3_recommended_condition_satisfied:
        selected_residual_route = "trial3_relaunched_refactored_post_ell20_radial_domain_extension_identification"
        missing_v2_artifact = "trial3_relaunched_refactored_localized_exact_family_table_above_ell20_with_extended_radial_domain"
    elif localized_ell_values and not trial3_recommended_condition_satisfied:
        highest_ell = max(localized_ell_values)
        selected_residual_route = f"trial3_relaunched_refactored_post_ell{highest_ell}_same_family_reaudit"
        missing_v2_artifact = f"trial3_relaunched_refactored_post_ell{highest_ell}_same_family_closeout_pack"
    else:
        selected_residual_route = None
        missing_v2_artifact = None

    declaration = helper.payload(
        "8.7.56.291",
        "Trial-3 refactored declaration fourth gate",
        source_inventory["inputs"],
        "Freeze whether the explicit post-ell19 radial-domain extension is sufficient to close Trial-3 or whether the next residual route must move deeper into the post-ell20 solver domain.",
        {
            "closeout_rule": "Trial-3 closes only if the explicit post-ell19 radial-domain extension carries the same-family exact table through W/Z and the Weinberg-angle proxy",
            "residual_rule": "if the radial extension reopens only ell=20 or leaves higher ell dark while still failing W/Z, the next residual route must move beyond the post-ell19 radial contract",
        },
        [
            helper.row("trial3_refactored_declaration_fourth_gate_complete", "pass", "Trial-3 refactored declaration fourth gate complete", 1, "The fourth refactored declaration gate is frozen."),
            helper.row("trial3_refactored_fourth_branch_closeable", "pass" if trial3_recommended_condition_satisfied else "reject", "refactored Trial-3 branch closeable after post-ell19 radial extension", 1 if trial3_recommended_condition_satisfied else 0, "The branch closes only if the radial extension actually closes the weak-sector path."),
            helper.row("trial3_refactored_fourth_residual_route_required", "reject" if trial3_recommended_condition_satisfied else "pass", "refactored Trial-3 residual route required after post-ell19 radial extension", 0 if trial3_recommended_condition_satisfied else 1, "A new residual route is still required when the radial extension improves the ceiling but cannot yet close W/Z honestly."),
            helper.row("trial3_refactored_execute_trial2_paper_sync_now_fourth_gate", "reject", "execute Trial-2 paper-side sync now after post-ell19 radial extension", 0, "Trial-2 paper-side sync remains reserve work while the refactored Trial-3 scientific route stays open."),
        ],
        {
            "trial3_recommended_condition_satisfied": trial3_recommended_condition_satisfied,
            "trial3_current_branch_closeable": trial3_recommended_condition_satisfied,
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "recommended_next_route_or_none": None if trial3_recommended_condition_satisfied else "8.7.56.293",
        },
        {
            "overall_status": "trial3_refactored_declaration_fourth_gate_frozen",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_292": True,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["trial3_refactored_trial2_paper_sync_trial4_disposition_sixteenth_refresh"],
        },
        {
            "audit_summary": audit["summary"],
            "prior_amplitude_gate_summary": amplitude_gate["summary"],
            "best_pair_or_none": best_pair,
        },
    )

    disposition = helper.payload(
        "8.7.56.292",
        "Trial-2 paper-side sync / Trial-4 disposition sixteenth refresh",
        source_inventory["inputs"],
        "Refresh the reserve/deferred ordering after the post-ell19 radial-domain extension and freeze the next official residual route.",
        {
            "trial2_rule": "Trial-2 paper-side sync stays unlocked reserve retained until the refactored Trial-3 route loses all honest current-canon search axes",
            "trial4_rule": "Trial-4 remains deferred while the refactored Trial-3 route still has an honest same-family or solver-domain residual path",
        },
        [
            helper.row("trial3_refactored_trial2_trial4_sixteenth_refresh_complete", "pass", "Trial-2 paper-side sync / Trial-4 disposition sixteenth refresh complete", 1, "The reserve/deferred ordering is refreshed after the post-ell19 radial-domain audit."),
            helper.row("trial3_refactored_trial2_paper_side_sync_reserve_retained_sixteenth_refresh", "pass", "Trial-2 paper-side sync reserve retained", 1, "Trial-2 paper-side sync stays unlocked reserve work while Trial-3 remains scientifically open."),
            helper.row("trial3_refactored_trial4_deferred_retained_sixteenth_refresh", "pass", "Trial-4 deferred retained", 1, "Trial-4 stays deferred while the refactored Trial-3 route still has an honest current-canon path."),
        ],
        {
            "selected_residual_route": selected_residual_route,
            "missing_v2_artifact": missing_v2_artifact,
            "trial2_paper_side_sync_state": "unlocked_reserve_retained",
            "trial4_deferred": True,
            "recommended_next_route_or_none": None if trial3_recommended_condition_satisfied else "8.7.56.293",
        },
        {
            "overall_status": "trial3_refactored_trial2_trial4_sixteenth_disposition_refreshed",
            "trial3_branch_closeable": trial3_recommended_condition_satisfied,
            "advance_to_8_7_56_13": False,
            "next_required_artifacts": [] if trial3_recommended_condition_satisfied else ["8.7.56.293"],
        },
        {
            "declaration_summary": declaration["summary"],
            "prior_amplitude_disposition_summary": amplitude_disposition["summary"],
            "solver_refactor_weak_summary": solver_refactor_weak["summary"],
        },
    )

    helper.write_artifact("mass_origin_v2_trial3_refactored_post_ell19_radial_domain_extension_source_inventory", source_inventory)
    helper.write_artifact("mass_origin_v2_trial3_refactored_post_ell19_radial_domain_extension_audit", audit)
    helper.write_artifact("mass_origin_v2_trial3_refactored_declaration_fourth_gate", declaration)
    helper.write_artifact("mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_sixteenth_refresh", disposition)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_refactored_post_ell19_radial_domain_extension_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_post_ell19_radial_domain_extension_audit_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_declaration_fourth_gate_metrics.json")
    print(" - mass_origin_v2_trial3_refactored_paper_sync_trial4_disposition_sixteenth_refresh_metrics.json")


# 関数: CLI から refactored post-ell19 radial-domain branch を起動する。

if __name__ == "__main__":
    main()
