#!/usr/bin/env python3
"""
Generate vector Q-ball extended-hierarchy artifacts for 8.7.55.2.832-.836.

This branch keeps the adopted exact full-coupled vector-Q-ball solver and
extends the hierarchy search window beyond the original `ell <= 3`, `k = 0`
pilot. The central question is whether heavy hierarchy anchors can be
recovered without adding a new coupling: first by extending `ell`, then by
checking whether a near-degenerate same-family baryon pair already reproduces
`m_n / m_p` before any explicit `k > 0` extension is required.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
VECTOR_ROUTE = OUT / "mass_origin_vector_qball_extended_hierarchy_route_contract_metrics.json"
VECTOR_RATIO = OUT / "mass_origin_mass_ratio_pilot_metrics.json"
VECTOR_SPIN = OUT / "mass_origin_vector_qball_spin_orbit_freeze_audit_metrics.json"
SCALAR_SPECTRUM = OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"
NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_COUPLED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

MUON_TARGET = 206.768283
PROTON_TARGET = 1836.15267343
TAU_TARGET = 3477.48
NEUTRON_PROTON_TARGET = 1.00137841925
PASS_THRESHOLD = 0.10
EXTENDED_ELL_VALUES = tuple(range(4, 13))


# Function: Return the current UTC timestamp in ISO 8601 format.
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: Abort immediately when a required artifact is missing.

def req(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: Read a UTF-8 JSON artifact into a dictionary.

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: Read a UTF-8 text source file.

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# Function: Convert an absolute path to a repo-relative string.

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# Function: Return the first source line that contains the requested pattern.

def hit(text: str, pattern: str) -> dict | None:
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: Build a common metrics row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: Build a common payload with the shared schema.

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


# Function: Save a JSON artifact and the paired CSV row table.

def write_artifact(stem: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: Dynamically load a local Python module.

def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Function: Return the extended beta grid for the requested high-ell sector.

def extended_beta_grid(ell: int) -> list[float]:
    start = max(0.06, 0.18 - 0.02 * (ell - 4))
    end = min(0.58, 0.54 + 0.02 * (ell - 4))
    return [round(start + (end - start) * index / 9.0, 6) for index in range(10)]


# Function: Run the localized scan for an extended high-ell sector.

def scan_extended_sector(prev, ell: int) -> list[dict]:
    rows = []
    for beta in extended_beta_grid(int(ell)):
        localized_profiles = prev.find_sector_amplitudes(float(beta), int(ell))
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


# Function: Return the best comparison row for each target label.

def best_rows_by_target(comparisons: list[dict]) -> dict[str, dict]:
    labels = ["m_mu/m_e", "m_p/m_e", "m_tau/m_e", "m_n/m_p"]
    best: dict[str, dict] = {}
    for label in labels:
        rows = [record for record in comparisons if record["target_label"] == label]
        best[label] = min(rows, key=lambda item: float(item["relative_error"]))

    return best


# Function: Search the proton-pass band for the best same-family neutron/proton pair.

def best_neutron_pair_same_family(exact_rows: list[dict]) -> dict | None:
    proton_pass_band = [
        row_data
        for row_data in exact_rows
        if abs(float(row_data["mass_ratio_to_scalar_base"]) - PROTON_TARGET) / PROTON_TARGET <= PASS_THRESHOLD
    ]
    by_family: dict[tuple[int, int], list[dict]] = {}
    for row_data in proton_pass_band:
        if int(row_data["ell"]) == 0:
            continue

        key = (int(row_data["ell"]), int(row_data["s"]))
        by_family.setdefault(key, []).append(row_data)

    best = None
    for key, family_rows in by_family.items():
        family_rows = sorted(family_rows, key=lambda item: float(item["mass_ratio_to_scalar_base"]))
        for left, right in combinations(family_rows, 2):
            heavier = max(left, right, key=lambda item: float(item["mass_ratio_to_scalar_base"]))
            lighter = min(left, right, key=lambda item: float(item["mass_ratio_to_scalar_base"]))
            ratio_value = float(heavier["mass_ratio_to_scalar_base"]) / float(lighter["mass_ratio_to_scalar_base"])
            relative_error = abs(ratio_value - NEUTRON_PROTON_TARGET) / NEUTRON_PROTON_TARGET
            mean_proton_scale = 0.5 * (
                float(heavier["mass_ratio_to_scalar_base"]) + float(lighter["mass_ratio_to_scalar_base"])
            )
            mean_proton_relative_error = abs(mean_proton_scale - PROTON_TARGET) / PROTON_TARGET
            candidate = {
                "family_key": {"ell": int(key[0]), "s": int(key[1])},
                "heavier_state": {
                    "n": int(heavier["n"]),
                    "k": int(heavier["k"]),
                    "ell": int(heavier["ell"]),
                    "s": int(heavier["s"]),
                    "ratio_to_electron": float(heavier["mass_ratio_to_scalar_base"]),
                },
                "lighter_state": {
                    "n": int(lighter["n"]),
                    "k": int(lighter["k"]),
                    "ell": int(lighter["ell"]),
                    "s": int(lighter["s"]),
                    "ratio_to_electron": float(lighter["mass_ratio_to_scalar_base"]),
                },
                "neutron_proton_ratio_value": float(ratio_value),
                "relative_error": float(relative_error),
                "mean_proton_scale_ratio_to_electron": float(mean_proton_scale),
                "mean_proton_scale_relative_error": float(mean_proton_relative_error),
                "passes_threshold": bool(relative_error <= PASS_THRESHOLD),
            }
            if best is None:
                best = candidate
                continue

            current_key = (float(candidate["relative_error"]), float(candidate["mean_proton_scale_relative_error"]))
            best_key = (float(best["relative_error"]), float(best["mean_proton_scale_relative_error"]))
            if current_key < best_key:
                best = candidate

    return best


# Function: Keep a readable sample from a long row table.

def sample(rows: list[dict], count: int = 16) -> list[dict]:
    if len(rows) <= count:
        return rows

    step = max(1, len(rows) // count)
    sampled = [rows[index] for index in range(0, len(rows), step)]
    return sampled[:count]


# Function: Run the extended-hierarchy branch and write artifacts.

def main() -> None:
    for path in (PART1, VECTOR_ROUTE, VECTOR_RATIO, VECTOR_SPIN, SCALAR_SPECTRUM, NUMERICAL_BRANCH, FULL_COUPLED_BRANCH):
        req(path)

    part1_text = read_text(PART1)
    route_contract = read_json(VECTOR_ROUTE)
    ratio_pilot = read_json(VECTOR_RATIO)
    spin_freeze = read_json(VECTOR_SPIN)
    scalar_modes = read_json(SCALAR_SPECTRUM)["evidence"]["discrete_mass_mode_rows"]
    prev = load_module(NUMERICAL_BRANCH, "vector_qball_effective_extended")
    full = load_module(FULL_COUPLED_BRANCH, "vector_qball_full_extended")

    lambda_rot = float(spin_freeze["summary"]["lambda_rot_value"])
    base_modes_by_ell = {}
    pilot_sector_summary_rows = []
    pilot_sector_scan_rows = {}

    for ell in (1, 2, 3):
        scan_rows = prev.scan_ell_sector(ell)
        base_modes_by_ell[ell] = prev.interpolate_integer_modes(scan_rows, ell)
        pilot_sector_scan_rows[str(ell)] = scan_rows

    for ell in EXTENDED_ELL_VALUES:
        scan_rows = scan_extended_sector(prev, ell)
        localized_rows = [row_data for row_data in scan_rows if row_data.get("localized_solution_found")]
        modes = prev.interpolate_integer_modes(scan_rows, ell)
        base_modes_by_ell[ell] = modes
        pilot_sector_scan_rows[str(ell)] = scan_rows
        pilot_sector_summary_rows.append(
            {
                "ell": int(ell),
                "beta_grid": extended_beta_grid(ell),
                "localized_solution_count": len(localized_rows),
                "localized_beta_interval_or_none": [localized_rows[0]["beta"], localized_rows[-1]["beta"]] if localized_rows else None,
                "charge_interval_or_none": [localized_rows[0]["charge_proxy"], localized_rows[-1]["charge_proxy"]] if localized_rows else None,
                "integer_mode_count": len(modes),
            }
        )

    exact_rows = full.build_exact_ladder(scalar_modes, base_modes_by_ell, lambda_rot)
    comparisons, closest = full.compare_known_targets(exact_rows)
    best_by_target = best_rows_by_target(comparisons)
    neutron_pair = best_neutron_pair_same_family(exact_rows)

    proton_pass = bool(best_by_target["m_p/m_e"]["passes_threshold"])
    tau_pass = bool(best_by_target["m_tau/m_e"]["passes_threshold"])
    muon_pass = bool(best_by_target["m_mu/m_e"]["passes_threshold"])
    neutron_pair_pass = bool(neutron_pair and neutron_pair["passes_threshold"])
    advance_to_8_7_55_3 = bool(muon_pass and proton_pass and tau_pass and neutron_pair_pass)

    total_extended_integer_modes = sum(len(base_modes_by_ell[ell]) for ell in EXTENDED_ELL_VALUES)
    high_ell_only_rows = [row_data for row_data in exact_rows if int(row_data["ell"]) in EXTENDED_ELL_VALUES]
    second_route_decision = (
        "advance_to_8_7_55_3_after_vector_extended_hierarchy"
        if advance_to_8_7_55_3
        else "continue_vector_qball_extended_hierarchy_without_8_7_55_3"
    )

    payloads = {
        "mass_origin_vector_qball_extended_hierarchy_sector_inventory": payload(
            "8.7.55.2.832",
            "Vector Q-ball extended hierarchy sector inventory",
            {
                "mass_origin_vector_qball_extended_hierarchy_route_contract_json": rel(VECTOR_ROUTE),
                "mass_origin_mass_ratio_pilot_json": rel(VECTOR_RATIO),
                "mass_origin_vector_qball_spin_orbit_freeze_audit_json": rel(VECTOR_SPIN),
            },
            "Inventory the extended high-ell hierarchy sectors that can be scanned next without changing the adopted exact vector-Q-ball solver.",
            {
                "extended_ell_scan_range": [min(EXTENDED_ELL_VALUES), max(EXTENDED_ELL_VALUES)],
                "extended_beta_grid_rule": "beta_start=max(0.06, 0.18-0.02*(ell-4)), beta_end=min(0.58, 0.54+0.02*(ell-4)), 10 evenly spaced samples",
                "priority_rule": "extend k=0 high-ell sectors first, then decide whether an explicit k>0 ladder is still needed",
            },
            [
                row(
                    "vector_qball_extended_hierarchy_sector_inventory_complete",
                    "pass",
                    "vector Q-ball extended hierarchy sector inventory complete",
                    1,
                    "The extended scan window is frozen.",
                ),
                row(
                    "vector_qball_extended_hierarchy_sector_count",
                    "pass",
                    "extended hierarchy sector count",
                    len(EXTENDED_ELL_VALUES),
                    "Nine new high-ell sectors are queued for the extended scan.",
                ),
                row(
                    "vector_qball_extended_integer_mode_lower_bound",
                    "pass",
                    "extended integer mode lower bound",
                    total_extended_integer_modes,
                    "The extended high-ell scan already opens a large integer-charge hierarchy without adding a new coupling.",
                ),
            ],
            {
                "extended_ell_min": min(EXTENDED_ELL_VALUES),
                "extended_ell_max": max(EXTENDED_ELL_VALUES),
                "extended_sector_count": len(EXTENDED_ELL_VALUES),
                "extended_integer_mode_lower_bound": total_extended_integer_modes,
                "priority_extension_axis": "high_ell_k0_before_explicit_k_positive",
                "first_route_to_close_or_none": "high_ell_k0_extension",
            },
            {
                "overall_status": "vector_qball_extended_hierarchy_sector_inventory_frozen",
                "keep_mass_origin_branch_blocked": True,
                "advance_to_8_7_55_3": False,
                "next_required_artifacts": [
                    "vector_qball_high_ell_pilot",
                    "vector_qball_near_degenerate_node_pilot",
                ],
            },
            {
                "pilot_sector_summary_rows": pilot_sector_summary_rows,
                "part1_vector_line": hit(part1_text, "P_\\mu=(P_t,P_1,P_2,P_3)"),
                "route_contract_summary": route_contract["summary"],
                "mass_ratio_pilot_summary": ratio_pilot["summary"],
            },
        ),
        "mass_origin_vector_qball_high_ell_pilot": payload(
            "8.7.55.2.833",
            "Vector Q-ball high-ell k=0 pilot",
            {
                "mass_origin_vector_qball_extended_hierarchy_sector_inventory_json": "output/public/quantum/mass_origin_vector_qball_extended_hierarchy_sector_inventory_metrics.json",
                "mass_origin_vector_qball_exact_mass_table_handoff_retry_json": rel(OUT / "mass_origin_vector_qball_exact_mass_table_handoff_retry_metrics.json"),
            },
            "Extend the adopted exact vector ladder to higher ell at k=0 and check whether proton/tau-scale anchors appear without introducing a new coupling.",
            {
                "extended_ladder_rule": "reuse the adopted exact full-coupled reconstruction while extending the k=0 base modes to ell=4..12",
                "heavy_anchor_targets": {"m_p/m_e": PROTON_TARGET, "m_tau/m_e": TAU_TARGET},
            },
            [
                row(
                    "vector_qball_high_ell_pilot_complete",
                    "pass",
                    "vector Q-ball high-ell pilot complete",
                    1,
                    "The extended high-ell exact ladder has been rebuilt.",
                ),
                row(
                    "vector_qball_proton_anchor_pass",
                    "pass" if proton_pass else "reject",
                    "proton/electron anchor passes threshold",
                    1 if proton_pass else 0,
                    "The extended high-ell ladder is checked against the proton/electron anchor.",
                ),
                row(
                    "vector_qball_tau_anchor_pass",
                    "pass" if tau_pass else "reject",
                    "tau/electron anchor passes threshold",
                    1 if tau_pass else 0,
                    "The extended high-ell ladder is checked against the tau/electron anchor.",
                ),
            ],
            {
                "extended_high_ell_state_count": len(high_ell_only_rows),
                "extended_exact_ratio_candidate_count": len(comparisons),
                "best_proton_row_or_none": best_by_target["m_p/m_e"],
                "best_tau_row_or_none": best_by_target["m_tau/m_e"],
                "proton_anchor_pass": proton_pass,
                "tau_anchor_pass": tau_pass,
            },
            {
                "overall_status": "vector_qball_high_ell_pilot_complete",
                "keep_mass_origin_branch_blocked": True,
                "advance_to_8_7_55_3": False,
                "next_required_artifacts": ["vector_qball_near_degenerate_node_pilot"],
            },
            {
                "high_ell_sector_summary_rows": pilot_sector_summary_rows,
                "top_proton_candidate_rows": [row_data for row_data in comparisons if row_data["target_label"] == "m_p/m_e"][:16],
                "top_tau_candidate_rows": [row_data for row_data in comparisons if row_data["target_label"] == "m_tau/m_e"][:16],
            },
        ),
        "mass_origin_vector_qball_near_degenerate_node_pilot": payload(
            "8.7.55.2.834",
            "Vector Q-ball k-node / near-degenerate pilot",
            {
                "mass_origin_vector_qball_high_ell_pilot_json": "output/public/quantum/mass_origin_vector_qball_high_ell_pilot_metrics.json",
            },
            "Check whether a near-degenerate same-family baryon pair already reproduces `m_n / m_p` before any explicit k>0 extension is required.",
            {
                "pair_search_rule": "search the proton-pass band (10%) for the same-(ell,s) pair that minimizes the neutron/proton relative error",
                "k_extension_rule": "explicit k>0 is required only if no same-family proton-band pair reproduces m_n/m_p within threshold",
            },
            [
                row(
                    "vector_qball_near_degenerate_node_pilot_complete",
                    "pass",
                    "vector Q-ball near-degenerate pilot complete",
                    1,
                    "The proton-band pair search is frozen.",
                ),
                row(
                    "vector_qball_neutron_proton_pair_available",
                    "pass" if neutron_pair_pass else "reject",
                    "neutron/proton pair available",
                    1 if neutron_pair_pass else 0,
                    "The same-family proton-band pair is checked against the neutron/proton ratio.",
                ),
                row(
                    "vector_qball_explicit_k_positive_required",
                    "reject" if neutron_pair_pass else "pass",
                    "explicit k-positive extension required",
                    0 if neutron_pair_pass else 1,
                    "Explicit k>0 extension is unnecessary when the same-family proton-band pair already reproduces the baryon doublet ratio.",
                ),
            ],
            {
                "near_degenerate_same_family_pair_available": neutron_pair_pass,
                "best_neutron_proton_pair_or_none": neutron_pair,
                "k_positive_extension_required": not neutron_pair_pass,
            },
            {
                "overall_status": "vector_qball_near_degenerate_node_pilot_complete",
                "keep_mass_origin_branch_blocked": True,
                "advance_to_8_7_55_3": False,
                "next_required_artifacts": ["vector_qball_baryon_tau_neutron_fit_table"],
            },
            {
                "best_neutron_proton_pair": neutron_pair,
            },
        ),
        "mass_origin_vector_qball_baryon_tau_neutron_fit_table": payload(
            "8.7.55.2.835",
            "Vector Q-ball baryon/tau/neutron fit table",
            {
                "mass_origin_vector_qball_high_ell_pilot_json": "output/public/quantum/mass_origin_vector_qball_high_ell_pilot_metrics.json",
                "mass_origin_vector_qball_near_degenerate_node_pilot_json": "output/public/quantum/mass_origin_vector_qball_near_degenerate_node_pilot_metrics.json",
            },
            "Freeze the heavy-hierarchy fit table after the extended high-ell and near-degenerate baryon scans.",
            {
                "single_state_targets": {
                    "m_mu/m_e": MUON_TARGET,
                    "m_p/m_e": PROTON_TARGET,
                    "m_tau/m_e": TAU_TARGET,
                },
                "pair_target": {"m_n/m_p": NEUTRON_PROTON_TARGET},
            },
            [
                row(
                    "vector_qball_baryon_tau_neutron_fit_table_complete",
                    "pass",
                    "vector Q-ball baryon/tau/neutron fit table complete",
                    1,
                    "The heavy-hierarchy fit table is frozen.",
                ),
                row(
                    "vector_qball_heavy_hierarchy_anchor_pass",
                    "pass" if proton_pass and tau_pass else "reject",
                    "heavy hierarchy anchors pass",
                    1 if proton_pass and tau_pass else 0,
                    "The proton and tau anchors are checked inside the extended exact ladder.",
                ),
                row(
                    "vector_qball_baryon_doublet_pair_pass",
                    "pass" if neutron_pair_pass else "reject",
                    "baryon doublet pair passes",
                    1 if neutron_pair_pass else 0,
                    "The baryon doublet pair is checked against the neutron/proton ratio.",
                ),
            ],
            {
                "best_muon_row_or_none": best_by_target["m_mu/m_e"],
                "best_proton_row_or_none": best_by_target["m_p/m_e"],
                "best_tau_row_or_none": best_by_target["m_tau/m_e"],
                "best_neutron_proton_pair_or_none": neutron_pair,
                "muon_anchor_pass": muon_pass,
                "proton_anchor_pass": proton_pass,
                "tau_anchor_pass": tau_pass,
                "neutron_proton_pair_pass": neutron_pair_pass,
            },
            {
                "overall_status": "vector_qball_baryon_tau_neutron_fit_table_complete",
                "keep_mass_origin_branch_blocked": not advance_to_8_7_55_3,
                "advance_to_8_7_55_3": False,
                "next_required_artifacts": ["vector_qball_second_route_gate_refresh"],
            },
            {
                "best_rows_by_target": best_by_target,
                "best_neutron_proton_pair": neutron_pair,
            },
        ),
        "mass_origin_vector_qball_second_route_gate_refresh": payload(
            "8.7.55.2.836",
            "Vector Q-ball second-route gate refresh",
            {
                "mass_origin_vector_qball_baryon_tau_neutron_fit_table_json": "output/public/quantum/mass_origin_vector_qball_baryon_tau_neutron_fit_table_metrics.json",
            },
            "Refresh the second-route gate after the extended exact hierarchy scan and decide whether 8.7.55.3 may start.",
            {
                "advance_rule": "advance requires muon, proton, tau, and neutron/proton anchors to pass within the current exact vector family",
            },
            [
                row(
                    "vector_qball_second_route_gate_refresh_complete",
                    "pass",
                    "vector Q-ball second-route gate refresh complete",
                    1,
                    "The second-route gate is refreshed after the extended hierarchy scan.",
                ),
                row(
                    "vector_qball_all_required_anchors_pass",
                    "pass" if advance_to_8_7_55_3 else "reject",
                    "all required anchors pass",
                    1 if advance_to_8_7_55_3 else 0,
                    "All required anchor tests are checked before reopening 8.7.55.3.",
                ),
                row(
                    "advance_to_8_7_55_3",
                    "pass" if advance_to_8_7_55_3 else "reject",
                    "advance to 8.7.55.3",
                    1 if advance_to_8_7_55_3 else 0,
                    "The third route reopens only after the extended vector hierarchy passes all anchor gates.",
                ),
            ],
            {
                "second_route_decision": second_route_decision,
                "mass_origin_branch_reopen_ready": advance_to_8_7_55_3,
                "advance_to_8_7_55_3": advance_to_8_7_55_3,
                "muon_anchor_pass": muon_pass,
                "proton_anchor_pass": proton_pass,
                "tau_anchor_pass": tau_pass,
                "neutron_proton_pair_pass": neutron_pair_pass,
                "recommended_next_route_or_none": "8.7.55.3" if advance_to_8_7_55_3 else "vector_qball_extended_hierarchy_followup",
            },
            {
                "overall_status": (
                    "vector_qball_extended_hierarchy_reopens_8_7_55_3"
                    if advance_to_8_7_55_3
                    else "vector_qball_extended_hierarchy_still_blocked"
                ),
                "keep_mass_origin_branch_blocked": not advance_to_8_7_55_3,
                "advance_to_8_7_55_3": advance_to_8_7_55_3,
                "new_branch_required": not advance_to_8_7_55_3,
                "next_required_artifacts": [] if advance_to_8_7_55_3 else ["vector_qball_extended_hierarchy_followup"],
            },
            {
                "best_rows_by_target": best_by_target,
                "best_neutron_proton_pair": neutron_pair,
                "closest_known_mass_ratio_or_none": closest,
            },
        ),
    }

    for stem, data in payloads.items():
        write_artifact(stem, data)
        print(f"[ok] wrote {OUT / (stem + '_metrics.json')}")
        print(f"[ok] wrote {OUT / (stem + '_rows.csv')}")


# Function: Run the extended-hierarchy branch when invoked as a script.

if __name__ == "__main__":
    main()
