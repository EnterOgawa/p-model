#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_readiness_gate_audit.py

Step 8.7.55.2.1:
Freeze the entry gate for the "mass origin" branch before attempting any
eigenvalue problem for V(|P|). This step does not solve the spectrum. It
collects the already-frozen observed targets, the currently available P-side
curvature proxy, and the missing same-sector conditions that still block a
no-free-parameter mass-spectrum closure.

Inputs:
  - output/public/quantum/gravity_quantum_differential_prediction_table_metrics.json
  - output/public/quantum/action_principle_el_derivation_audit.json
  - output/public/quantum/qcd_hadron_masses_baseline_metrics.json
  - output/public/quantum/nuclear_binding_energy_frequency_mapping_interface_metrics.json
  - data/quantum/sources/pdg_rpp_2024_mass_width/mass_width_2024.txt

Outputs:
  - output/public/quantum/mass_origin_readiness_gate_metrics.json
  - output/public/quantum/mass_origin_readiness_gate_rows.csv

Assumptions:
  - The observed target spectrum is frozen from PDG RPP 2024 masses.
  - The only currently available P-side curvature-related entry from 8.7.55.1
    is the structural parity row on (k_B T_env / chi_P); it is not yet a
    same-sector V''(|P|*) determination for particle masses.
  - The action-principle audit already fixes the minimal Lagrangian structure
    L = |D_mu P|^2 - V(|P|) - 1/4 F^2, but not the concrete shape of V(|P|).
  - The nuclear frequency-mass interface already freezes m = ħ ω_* / c^2 and
    Δm = ħ Δω / c^2 as an interface relation, not as a derivation of the
    discrete particle spectrum.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.qcd_hadron_masses_baseline import _mev, _parse_pdg_mcdata_mass_width  # noqa: E402


GRAVITY_DIFF_JSON = ROOT / "output" / "public" / "quantum" / "gravity_quantum_differential_prediction_table_metrics.json"
ACTION_JSON = ROOT / "output" / "public" / "quantum" / "action_principle_el_derivation_audit.json"
HADRON_JSON = ROOT / "output" / "public" / "quantum" / "qcd_hadron_masses_baseline_metrics.json"
NUCLEAR_INTERFACE_JSON = ROOT / "output" / "public" / "quantum" / "nuclear_binding_energy_frequency_mapping_interface_metrics.json"
PDG_FILE = ROOT / "data" / "quantum" / "sources" / "pdg_rpp_2024_mass_width" / "mass_width_2024.txt"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_readiness_gate_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_readiness_gate_rows.csv"

HBAR_J_S = 1.0545718176461565e-34
MEV_TO_J = 1.602176634e-13


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: `_require_path` の入出力契約と処理意図を定義する。

def _require_path(path: Path) -> None:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: `_omega_star_from_mass_mev` の入出力契約と処理意図を定義する。

def _omega_star_from_mass_mev(mass_mev: float) -> float:
    return float((mass_mev * MEV_TO_J) / HBAR_J_S)


# 関数: `_particle_row` の入出力契約と処理意図を定義する。

def _particle_row(*, label: str, mass_mev: float, electron_mass_mev: float, compton_lambda_fm: float | None) -> Dict[str, Any]:
    return {
        "label": label,
        "mass_mev": float(mass_mev),
        "omega_star_s_inv": _omega_star_from_mass_mev(float(mass_mev)),
        "ratio_to_electron": float(mass_mev / electron_mass_mev),
        "compton_lambda_fm": None if compton_lambda_fm is None else float(compton_lambda_fm),
    }


# 関数: `_find_row_by_id` の入出力契約と処理意図を定義する。

def _find_row_by_id(rows: List[Dict[str, Any]], row_id: str) -> Dict[str, Any]:
    for row in rows:
        # 条件分岐: `str(row.get("row_id")) == row_id` を満たす経路を評価する。
        if str(row.get("row_id")) == row_id:
            return row

    raise KeyError(f"missing row_id: {row_id}")


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload() -> Dict[str, Any]:
    for path in (GRAVITY_DIFF_JSON, ACTION_JSON, HADRON_JSON, NUCLEAR_INTERFACE_JSON, PDG_FILE):
        _require_path(path)

    gravity_diff = _read_json(GRAVITY_DIFF_JSON)
    action = _read_json(ACTION_JSON)
    hadron = _read_json(HADRON_JSON)
    nuclear = _read_json(NUCLEAR_INTERFACE_JSON)

    pdg_rows = _parse_pdg_mcdata_mass_width(PDG_FILE.read_text(encoding="utf-8", errors="replace").splitlines())
    selected_ids = {
        "electron": 11,
        "muon": 13,
        "tau": 15,
        "proton": 2212,
        "neutron": 2112,
        "pion_pm": 211,
        "kaon_pm": 321,
    }

    particle_masses: Dict[str, float] = {}
    for key, pid in selected_ids.items():
        row = pdg_rows.get(pid)
        # 条件分岐: `row is None or row.mass_gev is None` を満たす経路を評価する。
        if row is None or row.mass_gev is None:
            raise SystemExit(f"[fail] missing PDG mass for {key} ({pid})")

        particle_masses[key] = float(_mev(row.mass_gev))

    electron_mass_mev = particle_masses["electron"]
    hadron_rows = hadron.get("rows", [])
    hadron_lambda_by_label = {
        str(row.get("label")): row.get("compton_lambda_fm")
        for row in hadron_rows
        if isinstance(row, dict)
    }

    particle_rows = [
        _particle_row(label="electron", mass_mev=particle_masses["electron"], electron_mass_mev=electron_mass_mev, compton_lambda_fm=None),
        _particle_row(label="muon", mass_mev=particle_masses["muon"], electron_mass_mev=electron_mass_mev, compton_lambda_fm=None),
        _particle_row(label="tau", mass_mev=particle_masses["tau"], electron_mass_mev=electron_mass_mev, compton_lambda_fm=None),
        _particle_row(
            label="proton",
            mass_mev=particle_masses["proton"],
            electron_mass_mev=electron_mass_mev,
            compton_lambda_fm=float(hadron_lambda_by_label["p"]),
        ),
        _particle_row(
            label="neutron",
            mass_mev=particle_masses["neutron"],
            electron_mass_mev=electron_mass_mev,
            compton_lambda_fm=float(hadron_lambda_by_label["n"]),
        ),
        _particle_row(
            label="pion_pm",
            mass_mev=particle_masses["pion_pm"],
            electron_mass_mev=electron_mass_mev,
            compton_lambda_fm=float(hadron_lambda_by_label["π±"]),
        ),
        _particle_row(
            label="kaon_pm",
            mass_mev=particle_masses["kaon_pm"],
            electron_mass_mev=electron_mass_mev,
            compton_lambda_fm=float(hadron_lambda_by_label["K±"]),
        ),
    ]

    gravity_rows = gravity_diff.get("rows", [])
    # 条件分岐: `not isinstance(gravity_rows, list)` を満たす経路を評価する。
    if not isinstance(gravity_rows, list):
        raise SystemExit(f"[fail] invalid rows in {GRAVITY_DIFF_JSON}")

    structural_parity_row = _find_row_by_id(gravity_rows, "decoherence_structural_parity_entry")

    action_status = str(action.get("decision", {}).get("route_a_el_derivation_gate", "unknown"))

    nuclear_rows = nuclear.get("rows", [])
    # 条件分岐: `not isinstance(nuclear_rows, list) or not nuclear_rows` を満たす経路を評価する。
    if not isinstance(nuclear_rows, list) or not nuclear_rows:
        raise SystemExit(f"[fail] invalid rows in {NUCLEAR_INTERFACE_JSON}")

    omega0_values = [float(row["omega0_eff_per_s"]) for row in nuclear_rows if isinstance(row, dict)]
    omega0_median = sorted(omega0_values)[len(omega0_values) // 2]
    omega0_spread_fraction = (max(omega0_values) - min(omega0_values)) / omega0_median

    readiness_rows = [
        {
            "row_id": "minimal_action_fixed",
            "status": "pass" if action_status == "pass" else "watch",
            "metric": "route_a_el_derivation_gate",
            "value": action_status,
            "note": "The minimal Lagrangian structure is fixed, but V(|P|) itself is still not specified."
        },
        {
            "row_id": "observed_target_ratio_p_over_e",
            "status": "fixed_target",
            "metric": "m_p / m_e",
            "value": particle_masses["proton"] / particle_masses["electron"],
            "note": "This is the main target ratio for the mass-origin branch; reproducing ~1836.15 is the roadmap gate."
        },
        {
            "row_id": "observed_target_ratio_mu_over_e",
            "status": "fixed_target",
            "metric": "m_mu / m_e",
            "value": particle_masses["muon"] / particle_masses["electron"],
            "note": "Lepton-sector spacing is also frozen to prevent post-hoc target choice."
        },
        {
            "row_id": "light_nuclei_interface_omega0_spread",
            "status": "interface_fixed",
            "metric": "relative spread of omega0_eff per nucleon",
            "value": omega0_spread_fraction,
            "note": "The nuclear interface already freezes m = ħ ω_* / c^2, but it is an interface relation, not an eigenvalue derivation of discrete particle masses."
        },
        {
            "row_id": "structural_parity_entry_available",
            "status": "entry_only",
            "metric": "(k_B T_env / chi_P)_parity",
            "value": float(structural_parity_row["differential_prediction_value"]),
            "note": "This is the only currently frozen curvature-related proxy from 8.7.55.1, but it is not yet a same-sector V''(|P|*) determination for particle masses."
        },
        {
            "row_id": "same_sector_chi_p_to_vpp_mapping",
            "status": "missing",
            "metric": "chi_P -> V''(|P|*) in particle sector",
            "value": 0.0,
            "note": "The repository does not yet freeze a same-sector mapping from chi_P to the curvature of the mass-generating potential."
        },
        {
            "row_id": "single_potential_shape_fixed",
            "status": "missing",
            "metric": "V(|P|) shape fixed without new free parameters",
            "value": 0.0,
            "note": "The action audit fixes the form L = |D_mu P|^2 - V(|P|) - 1/4 F^2, but the concrete shape of V(|P|) is not frozen."
        },
        {
            "row_id": "mass_mode_boundary_condition_fixed",
            "status": "missing",
            "metric": "boundary / quantization rule for mass eigenmodes",
            "value": 0.0,
            "note": "The Schr/oscillon/Q-ball note still leaves multiple admissible quantization mechanisms open, so the eigenvalue problem is not unique yet."
        },
        {
            "row_id": "no_free_parameter_mass_solver_ready",
            "status": "reject",
            "metric": "readiness for no-free-parameter eigenvalue closure",
            "value": 0.0,
            "note": "Observed targets are frozen, but same-sector curvature, a unique potential shape, and a unique boundary condition are all still missing."
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.55.2.1",
            "name": "mass-origin readiness gate",
        },
        "inputs": {
            "gravity_quantum_differential_prediction_table_json": str(GRAVITY_DIFF_JSON.relative_to(ROOT)).replace("\\", "/"),
            "action_principle_el_derivation_audit_json": str(ACTION_JSON.relative_to(ROOT)).replace("\\", "/"),
            "qcd_hadron_masses_baseline_json": str(HADRON_JSON.relative_to(ROOT)).replace("\\", "/"),
            "nuclear_binding_energy_frequency_mapping_interface_json": str(NUCLEAR_INTERFACE_JSON.relative_to(ROOT)).replace("\\", "/"),
            "pdg_mass_width_file": str(PDG_FILE.relative_to(ROOT)).replace("\\", "/"),
        },
        "intent": "Freeze whether the mass-origin branch is actually ready for a no-free-parameter eigenvalue problem, using only already-frozen targets and already-frozen P-side curvature proxies.",
        "formulas": {
            "rest_mass_frequency_mapping": "m = ħ ω_* / c^2",
            "target_ratio_gate": "mass-origin branch continues only if a single V(|P|) eigenvalue problem reproduces m_p / m_e ≈ 1836.15 without new fit parameters",
            "current_curvature_proxy": "(k_B T_env / chi_P)_parity from 8.7.55.1.4 is only an entry proxy, not yet V''(|P|*) in the same particle sector",
        },
        "particle_spectrum_targets": particle_rows,
        "frozen_targets": {
            "proton_to_electron_ratio": particle_masses["proton"] / particle_masses["electron"],
            "neutron_to_proton_split_mev": particle_masses["neutron"] - particle_masses["proton"],
            "muon_to_electron_ratio": particle_masses["muon"] / particle_masses["electron"],
            "tau_to_electron_ratio": particle_masses["tau"] / particle_masses["electron"],
        },
        "interface_checks": {
            "light_nuclei_omega0_eff_per_nucleon_values": omega0_values,
            "light_nuclei_omega0_eff_spread_fraction": omega0_spread_fraction,
            "structural_parity_required_ratio": float(structural_parity_row["differential_prediction_value"]),
            "structural_parity_required_gate_1sigma_for_3sigma": float(structural_parity_row["required_precision_1sigma_for_3sigma"]),
        },
        "rows": readiness_rows,
        "summary": {
            "observed_target_rows": len(particle_rows),
            "action_principle_gate": action_status,
            "same_sector_curvature_fixed": False,
            "single_potential_shape_fixed": False,
            "boundary_condition_fixed": False,
            "no_free_parameter_mass_solver_ready": False,
        },
        "decision": {
            "overall_status": "entry_gate_fixed_not_ready_for_eigenvalue_closure",
            "new_free_parameters_introduced": False,
            "observed_target_spectrum_frozen": True,
            "same_sector_curvature_fixed": False,
            "single_potential_shape_fixed": False,
            "boundary_condition_fixed": False,
            "proceed_to_no_free_parameter_mass_solver": False,
            "next_required_steps": [
                "8.7.55.2.2",
                "8.7.54.23",
            ],
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["row_id", "status", "metric", "value", "note"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    payload = _build_payload()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_CSV, payload["rows"])
    print(f"[ok] json: {OUT_JSON}")
    print(f"[ok] csv : {OUT_CSV}")


# 条件分岐: `__name__ == \"__main__\"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
