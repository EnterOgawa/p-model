#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_curvature_bridge_audit.py

Step 8.7.55.2.14:
Check whether the surviving shell-quantization family already provides a
same-sector bridge from the particle-branch data to V''(|P|_*), or whether the
branch is still blocked at the curvature-mapping layer.

Inputs:
  - output/public/quantum/mass_origin_readiness_gate_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_canonicalization_metrics.json
  - output/public/quantum/mass_origin_solver_family_elimination_metrics.json
  - output/public/quantum/nuclear_binding_energy_frequency_mapping_interface_metrics.json
  - output/public/quantum/action_principle_el_derivation_audit.json

Outputs:
  - output/public/quantum/mass_origin_shell_curvature_bridge_metrics.json
  - output/public/quantum/mass_origin_shell_curvature_bridge_rows.csv
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

READINESS_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_readiness_gate_metrics.json"
SHELL_CANON_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_canonicalization_metrics.json"
ELIMINATION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_solver_family_elimination_metrics.json"
INTERFACE_JSON = ROOT / "output" / "public" / "quantum" / "nuclear_binding_energy_frequency_mapping_interface_metrics.json"
ACTION_JSON = ROOT / "output" / "public" / "quantum" / "action_principle_el_derivation_audit.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_curvature_bridge_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_curvature_bridge_rows.csv"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_require_path` の入出力契約と処理意図を定義する。

def _require_path(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: `_relative_str` の入出力契約と処理意図を定義する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: `_find_row_by_id` の入出力契約と処理意図を定義する。

def _find_row_by_id(rows: List[Dict[str, Any]], row_id: str) -> Dict[str, Any]:
    for row in rows:
        if str(row.get("row_id")) == row_id:
            return row

    raise KeyError(f"missing row_id: {row_id}")


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows(
    readiness: Dict[str, Any],
    shell_canon: Dict[str, Any],
    elimination: Dict[str, Any],
    interface: Dict[str, Any],
    action: Dict[str, Any],
) -> List[Dict[str, Any]]:
    readiness_rows = readiness.get("rows", [])
    elimination_rows = elimination.get("rows", [])
    if not isinstance(readiness_rows, list):
        raise SystemExit(f"[fail] invalid rows in {READINESS_JSON}")

    if not isinstance(elimination_rows, list):
        raise SystemExit(f"[fail] invalid rows in {ELIMINATION_JSON}")

    same_sector_row = _find_row_by_id(readiness_rows, "same_sector_chi_p_to_vpp_mapping")
    surviving_family_row = _find_row_by_id(elimination_rows, "single_public_boundary_family_remaining")
    shell_diag_row = _find_row_by_id(shell_canon.get("rows", []), "shell_quantization_fit_kappa")
    interface_spread = float(interface.get("interface_checks", {}).get("light_nuclei_omega0_eff_spread_fraction", 0.0))
    lagrangian = str(action.get("equations", {}).get("lagrangian_density", ""))
    shell_public = bool(shell_canon.get("decision", {}).get("shell_quantization_public_canonical", False))
    surviving_public = bool(elimination.get("decision", {}).get("single_public_boundary_family_remaining", False))
    return [
        {
            "row_id": "shell_quantization_public_family_survives",
            "status": "pass" if shell_public and surviving_public else "reject",
            "metric": "shell quantization remains the surviving public family",
            "value": 1.0 if shell_public and surviving_public else 0.0,
            "note": "Family elimination leaves shell quantization as the only public solver-family candidate.",
        },
        {
            "row_id": "nuclear_interface_stays_interface_only",
            "status": "interface_only",
            "metric": "light-nuclei omega0_eff spread is a consistency bridge only",
            "value": interface_spread,
            "note": "The nuclear interface freezes m = ħω0/c^2 consistency, but its own note says it does not yet choose a multi-body reduction or derive V''(|P|_*).",
        },
        {
            "row_id": "shell_quantization_coefficients_not_vpp",
            "status": "reject",
            "metric": "shell quantization coefficients directly define V''(|P|_*)",
            "value": float(shell_diag_row.get("value", 0.0)),
            "note": "kappa / kN / kZ are shell-gap correction amplitudes; no public row equates them to the particle-sector curvature V''(|P|_*).",
        },
        {
            "row_id": "same_sector_curvature_mapping_still_missing",
            "status": str(same_sector_row.get("status", "missing")),
            "metric": str(same_sector_row.get("metric", "")),
            "value": float(same_sector_row.get("value", 0.0)),
            "note": str(same_sector_row.get("note", "")),
        },
        {
            "row_id": "abstract_action_not_enough_for_vpp_coefficients",
            "status": "watch",
            "metric": "abstract action alone fixes particle-sector V(|P|) coefficients",
            "value": 0.0,
            "note": f"The action `{lagrangian}` fixes the abstract form only; it does not provide same-sector curvature coefficients for the surviving shell family.",
        },
        {
            "row_id": "shell_to_curvature_bridge_ready",
            "status": "reject",
            "metric": "same-sector shell-family bridge to V''(|P|_*) ready",
            "value": 0.0,
            "note": "A unique public solver family exists, but there is still no public canonical bridge from shell quantization observables to particle-sector V''(|P|_*).",
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload() -> Dict[str, Any]:
    for path in (READINESS_JSON, SHELL_CANON_JSON, ELIMINATION_JSON, INTERFACE_JSON, ACTION_JSON):
        _require_path(path)

    readiness = _read_json(READINESS_JSON)
    shell_canon = _read_json(SHELL_CANON_JSON)
    elimination = _read_json(ELIMINATION_JSON)
    interface = _read_json(INTERFACE_JSON)
    action = _read_json(ACTION_JSON)
    rows = _build_rows(readiness, shell_canon, elimination, interface, action)
    surviving_row = _find_row_by_id(rows, "shell_quantization_public_family_survives")
    bridge_row = _find_row_by_id(rows, "shell_to_curvature_bridge_ready")
    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.55.2.14",
            "name": "shell-family same-sector curvature bridge audit",
        },
        "inputs": {
            "mass_origin_readiness_gate_json": _relative_str(READINESS_JSON),
            "mass_origin_shell_quantization_canonicalization_json": _relative_str(SHELL_CANON_JSON),
            "mass_origin_solver_family_elimination_json": _relative_str(ELIMINATION_JSON),
            "nuclear_binding_energy_frequency_mapping_interface_json": _relative_str(INTERFACE_JSON),
            "action_principle_el_derivation_audit_json": _relative_str(ACTION_JSON),
        },
        "intent": "Decide whether the surviving shell-quantization family already supplies the same-sector curvature bridge needed for a no-free-parameter mass solver.",
        "rows": rows,
        "summary": {
            "single_public_family_remaining": float(surviving_row.get("value", 0.0)) > 0.0,
            "same_sector_curvature_bridge_available": float(bridge_row.get("value", 0.0)) > 0.0,
            "remaining_blockers": [
                "positive_particle_sector_chi_p_to_vpp_public_artifact",
                "single_public_vpp_shape",
            ],
        },
        "decision": {
            "overall_status": "unique_public_family_but_curvature_bridge_absent",
            "single_public_family_remaining": float(surviving_row.get("value", 0.0)) > 0.0,
            "same_sector_curvature_bridge_available": False,
            "proceed_to_single_vpp_shape_freeze": False,
            "proceed_to_no_free_parameter_mass_solver": False,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    payload = _build_payload()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["rows"])
    print(f"[ok] json: {OUT_JSON}")
    print(f"[ok] csv : {OUT_CSV}")


if __name__ == "__main__":
    main()
