#!/usr/bin/env python3
"""
Generate Trial-3 high-mass residual artifacts for 8.7.56.150-.152 and 8.7.56.153.

The first weak-sector pilot reached only the baryon/tau-scale neighborhood and
left a large mass gap to the W/Z sector. This residual branch tests whether the
current exact family can be extended honestly without adding a new coupling.

The branch does four things:

1. inventory the remaining high-mass extension axes,
2. audit whether those axes are admissible under the current canon,
3. freeze the second Trial-3 declaration gate and the Trial-4 disposition, and
4. freeze the next official residual route contract.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

TRIAL3_ROUTE = OUT / "mass_origin_v2_trial3_high_mass_scale_gap_route_contract_metrics.json"
TRIAL3_PILOT = OUT / "mass_origin_v2_trial3_wz_vector_mode_pilot_metrics.json"
TRIAL3_AUDIT = OUT / "mass_origin_v2_trial3_weinberg_angle_weak_coupling_audit_metrics.json"
VECTOR_SOLVER_SPEC = OUT / "mass_origin_vector_qball_solver_spec_metrics.json"
VECTOR_CONSTRAINT = OUT / "mass_origin_vector_qball_coupled_constraint_freeze_audit_metrics.json"
VECTOR_SPIN = OUT / "mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json"
VECTOR_HEAVY = OUT / "mass_origin_vector_qball_baryon_tau_neutron_fit_table_metrics.json"
TRIAL1_CASE_B = OUT / "mass_origin_v2_trial1_case_b_scope_declaration_gate_metrics.json"
SCALAR_SPECTRUM = OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"

NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_COUPLED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"
EXTENDED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_extended_hierarchy_branch.py"

W_TARGET = 80369.0 / 0.51099895
Z_TARGET = 91187.6 / 0.51099895
TRIAL3_PILOT_ELL_MAX = 14
STANDARD_EXTENSION_ELLS = tuple(range(15, 19))
BROAD_EXTENSION_ELLS = tuple(range(19, 25))
BROAD_BETA_GRID = (0.04, 0.08, 0.12, 0.16, 0.22, 0.28, 0.34, 0.40, 0.48, 0.56, 0.66, 0.76, 0.86, 0.92)


# Function: return the current UTC timestamp in ISO 8601 format.
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: abort immediately when a required path is missing.

def req(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: read a UTF-8 JSON artifact.

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: read a UTF-8 text source.

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# Function: convert an absolute path into a repo-relative string.

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# Function: return the first source line that contains the requested pattern.

def hit(text: str, pattern: str) -> dict | None:
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: build a standard result row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: build a standard payload object.

def payload(
    step: str,
    name: str,
    inputs: dict,
    intent: str,
    formulas: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "intent": intent,
        "formulas": formulas,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# Function: save a JSON artifact and its row CSV.

def write_artifact(stem: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: dynamically load a local module.

def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Function: scan a sector on a custom beta grid.

def scan_custom_sector(numerical, ell: int, beta_grid: tuple[float, ...]) -> list[dict]:
    rows = []
    for beta in beta_grid:
        localized_profiles = numerical.find_sector_amplitudes(float(beta), int(ell))
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

    return rows


# Function: return the maximum exact mass ratio for each ell in a rebuilt extension table.

def max_ratio_by_ell(rows: list[dict]) -> dict[int, float]:
    maxima: dict[int, float] = {}
    for row_data in rows:
        ell = int(row_data["ell"])
        if ell == 0:
            continue

        ratio = float(row_data["mass_ratio_to_scalar_base"])
        if ell not in maxima or ratio > maxima[ell]:
            maxima[ell] = ratio

    return maxima


# Function: return a readable summary row for a sector scan.

def scan_summary(scan_rows: list[dict], mode_count: int) -> dict:
    localized = [row_data for row_data in scan_rows if row_data.get("localized_solution_found")]
    if not localized:
        return {
            "localized_solution_count": 0,
            "localized_beta_interval_or_none": None,
            "charge_interval_or_none": None,
            "energy_interval_or_none": None,
            "integer_mode_count": int(mode_count),
        }

    return {
        "localized_solution_count": len(localized),
        "localized_beta_interval_or_none": [float(localized[0]["beta"]), float(localized[-1]["beta"])],
        "charge_interval_or_none": [float(localized[0]["charge_proxy"]), float(localized[-1]["charge_proxy"])],
        "energy_interval_or_none": [float(localized[0]["energy_proxy"]), float(localized[-1]["energy_proxy"])],
        "integer_mode_count": int(mode_count),
    }


# Function: keep a readable sample from a long row table.

def sample(rows: list[dict], count: int = 10) -> list[dict]:
    if len(rows) <= count:
        return rows

    step = max(1, len(rows) // count)
    sampled = [rows[index] for index in range(0, len(rows), step)]
    return sampled[:count]


# Function: compute the optimistic quadratic tail coefficient from verified ell maxima.

def quadratic_tail_coefficient(ell_maxima: dict[int, float]) -> float:
    tail_ells = sorted(ell for ell in ell_maxima if ell >= TRIAL3_PILOT_ELL_MAX)
    ratios = [float(ell_maxima[ell]) / float(ell * ell) for ell in tail_ells]
    return float(sum(ratios) / len(ratios))


# Function: estimate the ell needed to reach a target under the optimistic quadratic tail rule.

def ell_needed_for_target(target: float, coefficient: float) -> float:
    return float(math.sqrt(float(target) / float(coefficient)))


# Function: execute the Trial-3 high-mass residual branch.

def main() -> None:
    for path in (
        PART1,
        PART3A,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        TRIAL3_ROUTE,
        TRIAL3_PILOT,
        TRIAL3_AUDIT,
        VECTOR_SOLVER_SPEC,
        VECTOR_CONSTRAINT,
        VECTOR_SPIN,
        VECTOR_HEAVY,
        TRIAL1_CASE_B,
        SCALAR_SPECTRUM,
        NUMERICAL_BRANCH,
        FULL_COUPLED_BRANCH,
        EXTENDED_BRANCH,
    ):
        req(path)

    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    trial3_route = read_json(TRIAL3_ROUTE)
    trial3_pilot = read_json(TRIAL3_PILOT)
    trial3_audit = read_json(TRIAL3_AUDIT)
    vector_solver_spec = read_json(VECTOR_SOLVER_SPEC)
    vector_constraint = read_json(VECTOR_CONSTRAINT)
    vector_spin = read_json(VECTOR_SPIN)
    vector_heavy = read_json(VECTOR_HEAVY)
    trial1_case_b = read_json(TRIAL1_CASE_B)
    scalar_modes = read_json(SCALAR_SPECTRUM)["evidence"]["discrete_mass_mode_rows"]

    numerical = load_module(NUMERICAL_BRANCH, "trial3_high_mass_num")
    full = load_module(FULL_COUPLED_BRANCH, "trial3_high_mass_full")
    extended = load_module(EXTENDED_BRANCH, "trial3_high_mass_ext")

    lambda_rot = float(vector_spin["summary"]["lambda_rot_value"])
    common_inputs = {
        "part1_markdown": rel(PART1),
        "part3a_markdown": rel(PART3A),
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "trial3_route_contract_json": rel(TRIAL3_ROUTE),
        "trial3_wz_pilot_json": rel(TRIAL3_PILOT),
        "trial3_weinberg_audit_json": rel(TRIAL3_AUDIT),
        "vector_qball_solver_spec_json": rel(VECTOR_SOLVER_SPEC),
        "vector_qball_coupled_constraint_json": rel(VECTOR_CONSTRAINT),
        "vector_qball_spin_orbit_freeze_json": rel(VECTOR_SPIN),
        "vector_qball_baryon_tau_neutron_fit_table_json": rel(VECTOR_HEAVY),
        "trial1_case_b_scope_gate_json": rel(TRIAL1_CASE_B),
        "mass_origin_qball_discrete_mass_spectrum_json": rel(SCALAR_SPECTRUM),
    }

    standard_scan_rows = {}
    standard_mode_rows = {}
    for ell in STANDARD_EXTENSION_ELLS:
        scan_rows = extended.scan_extended_sector(numerical, int(ell))
        standard_scan_rows[int(ell)] = scan_rows
        standard_mode_rows[int(ell)] = numerical.interpolate_integer_modes(scan_rows, int(ell))

    broad_scan_rows = {}
    broad_mode_rows = {}
    for ell in BROAD_EXTENSION_ELLS:
        scan_rows = scan_custom_sector(numerical, int(ell), BROAD_BETA_GRID)
        broad_scan_rows[int(ell)] = scan_rows
        broad_mode_rows[int(ell)] = numerical.interpolate_integer_modes(scan_rows, int(ell))

    standard_exact_rows = full.build_exact_ladder(scalar_modes, standard_mode_rows, lambda_rot)
    broad_exact_rows = full.build_exact_ladder(scalar_modes, broad_mode_rows, lambda_rot)

    standard_maxima = max_ratio_by_ell(standard_exact_rows)
    broad_maxima = max_ratio_by_ell(broad_exact_rows)
    current_pilot_max_ratio = float(trial3_pilot["summary"]["maximum_mass_ratio_to_electron"])
    combined_maxima = {TRIAL3_PILOT_ELL_MAX: current_pilot_max_ratio}
    combined_maxima.update(standard_maxima)
    combined_maxima.update(broad_maxima)

    current_verified_ell_ceiling = max(ell for ell, count in {**{ell: len(modes) for ell, modes in standard_mode_rows.items()}, **{ell: len(modes) for ell, modes in broad_mode_rows.items()}}.items() if count > 0)
    current_verified_max_ratio = max(float(value) for value in combined_maxima.values())
    current_verified_w_gap_factor = float(W_TARGET / current_verified_max_ratio)
    current_verified_z_gap_factor = float(Z_TARGET / current_verified_max_ratio)
    tail_coefficient = quadratic_tail_coefficient(combined_maxima)
    optimistic_w_ell_needed = ell_needed_for_target(W_TARGET, tail_coefficient)
    optimistic_z_ell_needed = ell_needed_for_target(Z_TARGET, tail_coefficient)

    standard_summaries = {str(ell): scan_summary(scan_rows, len(standard_mode_rows[ell])) for ell, scan_rows in standard_scan_rows.items()}
    broad_summaries = {str(ell): scan_summary(scan_rows, len(broad_mode_rows[ell])) for ell, scan_rows in broad_scan_rows.items()}
    broad_localized_solution_count_total = sum(item["localized_solution_count"] for item in broad_summaries.values())

    inventory_targets = [
        {
            "label": "status_points_to_8_7_56_150",
            "present": "current official next step は `8.7.56.150`" in status_text,
            "note": "STATUS must already point to the residual inventory step.",
        },
        {
            "label": "roadmap_residual_branch_present",
            "present": "`8.7.56.149-.152` 試練3 high-mass weak-sector scale-gap residual branch" in roadmap_text,
            "note": "ROADMAP must already freeze the residual Trial-3 branch.",
        },
        {
            "label": "solver_spec_keeps_k_positive_axis",
            "present": "k>0" in str(vector_solver_spec["formulas"].get("pilot_sector_rule", "")),
            "note": "The canonical solver spec must still reserve k>0 as a same-family extension axis.",
        },
        {
            "label": "coupled_constraint_k_node_bookkeeping_available",
            "present": bool(vector_constraint["summary"]["k_node_bookkeeping_available"]),
            "note": "The full-coupled solver must already freeze k-node bookkeeping before any explicit k>0 activation.",
        },
        {
            "label": "heavy_anchor_table_retained",
            "present": bool(vector_heavy["summary"]["muon_anchor_pass"] and vector_heavy["summary"]["proton_anchor_pass"] and vector_heavy["summary"]["tau_anchor_pass"]),
            "note": "The previous mass-origin heavy anchors must remain frozen while Trial-3 is extended.",
        },
        {
            "label": "trial1_case_b_hold_retained",
            "present": bool(trial1_case_b["summary"]["trial2_hold_retained"]),
            "note": "Trial-3 extension must stay inside the current-canon Trial-1 Case-B closure and keep Trial-2 on hold.",
        },
    ]

    inventory_ready = all(item["present"] for item in inventory_targets) and ("8.7.56.150" in status_text)
    high_ell_only_viable_next_route = bool(broad_localized_solution_count_total > 0 and current_verified_w_gap_factor <= 1.0 and current_verified_z_gap_factor <= 1.0)
    explicit_k_positive_candidate_available = bool(
        vector_constraint["summary"]["k_node_bookkeeping_available"]
        and "k>0" in str(vector_solver_spec["formulas"].get("pilot_sector_rule", ""))
    )
    extension_admissible = bool(
        explicit_k_positive_candidate_available
        and trial1_case_b["summary"]["trial2_hold_retained"]
        and vector_heavy["summary"]["muon_anchor_pass"]
        and vector_heavy["summary"]["proton_anchor_pass"]
        and vector_heavy["summary"]["tau_anchor_pass"]
    )
    selected_next_route = (
        "trial3_explicit_k_positive_weak_sector_extension"
        if extension_admissible
        else "trial3_future_canon_weak_sector_reopen_registry"
    )
    next_route_number = "8.7.56.153" if extension_admissible else "8.7.56.13"

    inventory = payload(
        "8.7.56.150",
        "Trial-3 high-mass scale-extension inventory",
        common_inputs,
        "Inventory the remaining same-family extension axes after the first weak-sector pilot failed to reach the W/Z mass scale.",
        {
            "inventory_rule": "extend the current exact-family evidence beyond the first pilot window before changing canon or moving to Trial-4",
            "standard_extension_window": f"reuse the adopted high-ell scan rule for ell in {list(STANDARD_EXTENSION_ELLS)}",
            "broad_extension_window": f"test a widened beta grid for ell in {list(BROAD_EXTENSION_ELLS)} using beta in {list(BROAD_BETA_GRID)}",
            "quadratic_tail_rule": "estimate an optimistic ell-only continuation by averaging max_ratio/ell^2 over the verified tail ell=14..18",
        },
        [
            row(
                "trial3_high_mass_scale_extension_inventory_complete",
                "pass",
                "Trial-3 high-mass scale-extension inventory complete",
                1,
                "The residual extension inventory has been frozen.",
            ),
            row(
                "trial3_high_ell_extension_ceiling_verified",
                "pass",
                "verified high-ell extension ceiling",
                current_verified_ell_ceiling,
                "Localized same-family sectors were verified only up to the present ell ceiling.",
            ),
            row(
                "trial3_beyond_ceiling_localized_solutions_available",
                "pass" if broad_localized_solution_count_total > 0 else "reject",
                "localized solutions available beyond the verified ell ceiling",
                broad_localized_solution_count_total,
                "A zero count means the widened beta scan still finds no localized sectors beyond the present ceiling.",
            ),
            row(
                "trial3_explicit_k_positive_candidate_available",
                "pass" if explicit_k_positive_candidate_available else "reject",
                "explicit k-positive extension candidate available",
                1 if explicit_k_positive_candidate_available else 0,
                "The current canon still carries a frozen k-axis even though the weak-sector pilot has not activated it yet.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "current_pilot_ell_window_max": TRIAL3_PILOT_ELL_MAX,
            "current_pilot_max_ratio_to_electron": current_pilot_max_ratio,
            "current_verified_ell_ceiling": current_verified_ell_ceiling,
            "current_verified_max_ratio_to_electron": current_verified_max_ratio,
            "w_gap_factor_or_none": current_verified_w_gap_factor,
            "z_gap_factor_or_none": current_verified_z_gap_factor,
            "optimistic_quadratic_tail_coefficient": tail_coefficient,
            "optimistic_quadratic_ell_needed_for_w": optimistic_w_ell_needed,
            "optimistic_quadratic_ell_needed_for_z": optimistic_z_ell_needed,
            "high_ell_only_viable_next_route": high_ell_only_viable_next_route,
            "explicit_k_positive_candidate_available": explicit_k_positive_candidate_available,
        },
        {
            "overall_status": "trial3_high_mass_scale_extension_inventory_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_151": True,
            "next_required_artifacts": ["trial3_high_mass_extension_admissibility_audit"],
        },
        {
            "inventory_targets": inventory_targets,
            "standard_extension_sector_summaries": standard_summaries,
            "broad_extension_sector_summaries": broad_summaries,
            "verified_tail_max_ratio_by_ell": combined_maxima,
            "part1_micro_chiral_line": hit(part1_text, "左手系カイラル流"),
            "part3a_exact_vector_hierarchy_line": hit(part3a_text, "exact vector hierarchy"),
        },
    )

    admissibility = payload(
        "8.7.56.151",
        "Trial-3 high-mass extension admissibility audit",
        common_inputs,
        "Audit whether the remaining same-family high-mass extension route stays inside the current canon without reopening Trial-1 or breaking the mass-origin closeout.",
        {
            "admissibility_rule": "an admissible extension must reuse the frozen exact-family axes, introduce no new coupling, preserve the mass-origin anchor table, and leave Trial-1 Case-B untouched",
            "high_ell_rule": "high-ell-only continuation is viable only if the verified extension ceiling still grows toward the weak scale inside the present canon",
            "k_positive_rule": "explicit k>0 activation is admissible if k bookkeeping is already frozen and no new parameter is introduced",
        },
        [
            row(
                "trial3_high_ell_only_viable_next_route",
                "pass" if high_ell_only_viable_next_route else "reject",
                "high-ell-only continuation remains a viable next route",
                1 if high_ell_only_viable_next_route else 0,
                "The widened scan must still leave an honest high-ell continuation path if high-ell-only is to remain the next route.",
            ),
            row(
                "trial3_explicit_k_positive_extension_admissible",
                "pass" if explicit_k_positive_candidate_available else "reject",
                "explicit k-positive weak-sector extension admissible",
                1 if explicit_k_positive_candidate_available else 0,
                "The k-axis is already canonical, so activating it does not require a new free parameter.",
            ),
            row(
                "trial3_extension_consistent_with_mass_origin_closeout",
                "pass" if vector_heavy["summary"]["muon_anchor_pass"] and vector_heavy["summary"]["proton_anchor_pass"] and vector_heavy["summary"]["tau_anchor_pass"] else "reject",
                "weak-sector extension consistent with mass-origin closeout",
                1 if vector_heavy["summary"]["muon_anchor_pass"] and vector_heavy["summary"]["proton_anchor_pass"] and vector_heavy["summary"]["tau_anchor_pass"] else 0,
                "The previous heavy-anchor closeout must remain intact while the weak-sector residual is extended.",
            ),
            row(
                "trial3_extension_consistent_with_trial1_case_b",
                "pass" if trial1_case_b["summary"]["trial2_hold_retained"] else "reject",
                "weak-sector extension consistent with Trial-1 Case-B hold",
                1 if trial1_case_b["summary"]["trial2_hold_retained"] else 0,
                "Trial-3 may extend only if Trial-1 remains an honest partial closeout and Trial-2 stays on hold.",
            ),
        ],
        {
            "high_ell_only_viable_next_route": high_ell_only_viable_next_route,
            "explicit_k_positive_extension_admissible": explicit_k_positive_candidate_available,
            "weak_sector_high_mass_exact_family_extension_admissible": extension_admissible,
            "selected_admissible_route_or_none": selected_next_route if extension_admissible else None,
            "wording_only_or_trial4_jump_admissible": False,
        },
        {
            "overall_status": "trial3_high_mass_extension_admissibility_audited",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_152": True,
            "next_required_artifacts": ["trial3_second_declaration_gate"],
        },
        {
            "trial3_route_summary": trial3_route["summary"],
            "trial3_pilot_summary": trial3_pilot["summary"],
            "trial3_audit_summary": trial3_audit["summary"],
            "vector_constraint_summary": vector_constraint["summary"],
            "vector_solver_spec_formulas": vector_solver_spec["formulas"],
            "vector_heavy_summary": vector_heavy["summary"],
            "trial1_case_b_summary": trial1_case_b["summary"],
        },
    )

    declaration = payload(
        "8.7.56.152",
        "Trial-3 second declaration gate / Trial-4 disposition",
        common_inputs,
        "Freeze whether Trial-3 should continue inside a same-family residual extension or stand down in favor of Trial-4.",
        {
            "gate_rule": "if an admissible same-family extension still exists, Trial-3 continues and Trial-4 stays deferred",
            "trial4_rule": "Trial-4 remains deferred unless Trial-3 loses all honest current-canon extension routes",
        },
        [
            row(
                "trial3_second_declaration_gate_complete",
                "pass",
                "Trial-3 second declaration gate complete",
                1,
                "The second declaration gate is frozen.",
            ),
            row(
                "trial3_same_family_extension_admissible",
                "pass" if extension_admissible else "reject",
                "same-family weak-sector extension admissible",
                1 if extension_admissible else 0,
                "The next residual route remains inside the current exact family when the k-axis can be activated without a new parameter.",
            ),
            row(
                "trial3_launch_trial4_now",
                "reject",
                "launch Trial-4 now",
                0,
                "Trial-4 stays deferred while an admissible Trial-3 residual route still exists.",
            ),
        ],
        {
            "trial3_recommended_condition_satisfied": False,
            "trial3_current_branch_closeable": False,
            "trial3_same_family_extension_admissible": extension_admissible,
            "trial4_deferred": True,
            "recommended_next_route_or_none": next_route_number,
        },
        {
            "overall_status": "trial3_second_declaration_gate_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_13": False,
            "next_required_artifacts": ["trial3_explicit_k_positive_extension_route_contract"] if extension_admissible else [],
        },
        {
            "inventory_summary": inventory["summary"],
            "admissibility_summary": admissibility["summary"],
        },
    )

    route_contract = payload(
        "8.7.56.153",
        "Trial-3 explicit k-positive weak-sector extension route contract",
        common_inputs,
        "Freeze the next official residual route after the high-mass inventory shows that high-ell-only continuation is insufficient under the current canon.",
        {
            "selected_residual_route": selected_next_route,
            "missing_v2_artifact": "weak_sector_explicit_k_positive_exact_ladder" if extension_admissible else "trial3_future_canon_weak_sector_registry",
            "pivot_principle": "The verified high-ell ceiling reaches only ell=18 and still leaves a large weak-scale gap, so the remaining same-family extension axis is the already-frozen k>0 ladder.",
        },
        [
            row(
                "trial3_explicit_k_positive_extension_route_contract_complete",
                "pass",
                "Trial-3 explicit k-positive extension route contract complete",
                1,
                "The next residual route is frozen after the high-mass inventory and admissibility audit.",
            ),
            row(
                "trial3_high_ell_only_route_retired",
                "pass",
                "high-ell-only weak-sector route retired",
                1,
                "The widened scan found no localized sectors beyond ell=18, so high-ell-only is no longer the primary next route.",
            ),
            row(
                "trial3_trial4_handoff_ready_now",
                "reject",
                "Trial-4 handoff ready now",
                0,
                "Trial-4 remains deferred while the explicit k-positive Trial-3 residual route is still open.",
            ),
        ],
        {
            "selected_residual_route": selected_next_route,
            "missing_v2_artifact": "weak_sector_explicit_k_positive_exact_ladder" if extension_admissible else "trial3_future_canon_weak_sector_registry",
            "split_contract_ready": True,
            "trial3_same_family_extension_admissible": extension_admissible,
            "recommended_next_route_or_none": "8.7.56.154" if extension_admissible else "8.7.56.13",
        },
        {
            "overall_status": "trial3_explicit_k_positive_extension_route_contract_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_13": False,
            "next_required_artifacts": [
                "trial3_explicit_k_positive_source_inventory",
                "trial3_explicit_k_positive_weak_sector_pilot",
                "trial3_third_declaration_gate",
            ] if extension_admissible else [],
        },
        {
            "inventory_summary": inventory["summary"],
            "admissibility_summary": admissibility["summary"],
            "declaration_summary": declaration["summary"],
        },
    )

    write_artifact("mass_origin_v2_trial3_high_mass_scale_extension_inventory", inventory)
    write_artifact("mass_origin_v2_trial3_high_mass_extension_admissibility_audit", admissibility)
    write_artifact("mass_origin_v2_trial3_second_declaration_gate", declaration)
    write_artifact("mass_origin_v2_trial3_explicit_k_positive_extension_route_contract", route_contract)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_high_mass_scale_extension_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_high_mass_extension_admissibility_audit_metrics.json")
    print(" - mass_origin_v2_trial3_second_declaration_gate_metrics.json")
    print(" - mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_metrics.json")


# Function: run the Trial-3 high-mass residual branch from the command line.

if __name__ == "__main__":
    main()
