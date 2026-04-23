#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gravity_quantum_differential_prediction_table_audit.py

Step 8.7.55.1.4:
Aggregate the already-frozen gravity-quantum differential entries from
8.7.55.1.1-.3 into one machine-readable table with explicit 3σ rejection gates.

Inputs:
  - output/public/quantum/gravity_induced_decoherence_pikovski_comparison_metrics.json
  - output/public/quantum/atom_interferometer_pikovski_phase_difference_metrics.json
  - output/public/quantum/optical_clock_differential_frequency_metrics.json

Outputs:
  - output/public/quantum/gravity_quantum_differential_prediction_table_metrics.json
  - output/public/quantum/gravity_quantum_differential_prediction_table_rows.csv

Assumptions:
  - No new fit parameter is introduced; every row must be reconstructed from the
    already-frozen outputs of 8.7.55.1.1-.3.
  - The structural decoherence row is a parity-entry row, not a current direct
    measurement row, so its detectability status is fixed as "entry-only".
  - The P-unique atom-interferometer row is the β/light local channel against the
    local arm-separation Pikovski proxy.
  - The optical-clock main row remains the Earth Δh≈399 m chronometric-leveling
    case, while the solar 0.05 AU ↔ 1 AU row is kept as a lever-arm future case.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]

# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DECOHERENCE_JSON = ROOT / "output" / "public" / "quantum" / "gravity_induced_decoherence_pikovski_comparison_metrics.json"
ATOM_JSON = ROOT / "output" / "public" / "quantum" / "atom_interferometer_pikovski_phase_difference_metrics.json"
CLOCK_JSON = ROOT / "output" / "public" / "quantum" / "optical_clock_differential_frequency_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "gravity_quantum_differential_prediction_table_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "gravity_quantum_differential_prediction_table_rows.csv"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: `_format_gate` の入出力契約と処理意図を定義する。

def _format_gate(*, observable_label: str, required_precision: float, precision_unit: str) -> str:
    return (
        f"Reject if the measured {observable_label} differs from the frozen differential prediction "
        f"by more than 3σ, i.e. if |residual| > {required_precision:.6e} {precision_unit}."
    )


# 関数: `_coerce_optional_float` の入出力契約と処理意図を定義する。

def _coerce_optional_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return float(value)


# 関数: `_find_phase_row` の入出力契約と処理意図を定義する。

def _find_phase_row(metrics: Dict[str, Any], *, pmodel_case_id: str, pikovski_case_id: str) -> Dict[str, Any]:
    for row in metrics["comparison_rows"]:
        if row["pmodel_case_id"] == pmodel_case_id and row["pikovski_case_id"] == pikovski_case_id:
            return row

    raise KeyError(f"Missing atom-interferometer row: {pmodel_case_id} vs {pikovski_case_id}")


# 関数: `_find_clock_case` の入出力契約と処理意図を定義する。

def _find_clock_case(metrics: Dict[str, Any], *, case_id: str) -> Dict[str, Any]:
    for row in metrics["case_rows"]:
        if row["case_id"] == case_id:
            return row

    raise KeyError(f"Missing optical-clock case row: {case_id}")


# 関数: `_find_precision_row` の入出力契約と処理意図を定義する。

def _find_precision_row(metrics: Dict[str, Any], *, precision_case_id: str) -> Dict[str, Any]:
    for row in metrics["precision_reference_rows"]:
        if row["precision_case_id"] == precision_case_id:
            return row

    raise KeyError(f"Missing precision row: {precision_case_id}")


# 関数: `_build_structural_row` の入出力契約と処理意図を定義する。

def _build_structural_row(metrics: Dict[str, Any]) -> Dict[str, Any]:
    largest_case = metrics["decision"]["largest_parity_requirement_case"]
    parity_ratio = float(largest_case["required_thermal_ratio_for_parity"])
    required_precision = abs(parity_ratio) / 3.0
    sigma_z_mm = float(largest_case["sigma_z_mm"])
    tau_free_s = float(largest_case["tau_free_s"])

    return {
        "row_id": "decoherence_structural_parity_entry",
        "source_step": "8.7.55.1.1",
        "channel_group": "gravity_induced_decoherence",
        "channel_role": "structural_parity_entry",
        "observable": "(k_B T_env / chi_P)_parity",
        "case_id": f"sigma_z_{sigma_z_mm:.1f}mm_tau_free_{tau_free_s:.0e}s",
        "differential_prediction_value": parity_ratio,
        "differential_prediction_unit": "dimensionless",
        "required_precision_1sigma_for_3sigma": required_precision,
        "precision_unit": "dimensionless",
        "current_precision": None,
        "current_over_required": None,
        "detectable_under_current": None,
        "reject_gate_3sigma": (
            "Reject structural parity if an inferred (k_B T_env / chi_P) in the same "
            f"sigma_z={sigma_z_mm:.1f} mm / tau_free={tau_free_s:.0e} s regime deviates from the "
            f"frozen parity ratio by more than {required_precision:.6e}."
        ),
        "status": "entry_fixed_no_current_measurement",
        "notes": (
            "This row fixes the largest parity requirement from 8.7.55.1.1 and remains an entry-only "
            "structural comparison until the same regime can be inferred experimentally."
        ),
    }


# 関数: `_build_atom_rows` の入出力契約と処理意図を定義する。

def _build_atom_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    p_unique_row = _find_phase_row(
        metrics,
        pmodel_case_id="pmodel_beta_light_local",
        pikovski_case_id="pikovski_arm_separation_proxy",
    )
    reference_row = _find_phase_row(
        metrics,
        pmodel_case_id="pmodel_reference_mapping",
        pikovski_case_id="pikovski_apex_height_envelope",
    )
    rows: List[Dict[str, Any]] = []

    rows.append(
        {
            "row_id": "atom_interferometer_p_unique_local_arm",
            "source_step": "8.7.55.1.2",
            "channel_group": "atom_interferometer",
            "channel_role": "p_unique_beta_light_local",
            "observable": "Delta phi_diff / phi_ref",
            "case_id": "beta_light_local_vs_pikovski_arm_proxy",
            "differential_prediction_value": float(p_unique_row["delta_phi_diff_rad"]),
            "differential_prediction_unit": "rad",
            "required_precision_1sigma_for_3sigma": float(
                p_unique_row["required_fractional_phase_precision_1sigma_for_3sigma"]
            ),
            "precision_unit": "fractional_phase",
            "current_precision": float(p_unique_row["current_fractional_phase_precision"]),
            "current_over_required": float(p_unique_row["current_over_required_precision"]),
            "detectable_under_current": bool(p_unique_row["detectable_under_current_precision"]),
            "reject_gate_3sigma": _format_gate(
                observable_label="fractional atom-interferometer phase residual",
                required_precision=float(p_unique_row["required_fractional_phase_precision_1sigma_for_3sigma"]),
                precision_unit="fractional phase",
            ),
            "status": "p_unique_requires_255x_precision_improvement",
            "notes": (
                "This is the P-unique β/light local channel against the local arm-separation Pikovski proxy. "
                "Current/reference fractional precision is still 255x above the 1σ requirement."
            ),
        }
    )

    rows.append(
        {
            "row_id": "atom_interferometer_reference_apex_envelope",
            "source_step": "8.7.55.1.2",
            "channel_group": "atom_interferometer",
            "channel_role": "reference_mapping_row",
            "observable": "Delta phi_diff / phi_ref",
            "case_id": "reference_mapping_vs_pikovski_apex_envelope",
            "differential_prediction_value": float(reference_row["delta_phi_diff_rad"]),
            "differential_prediction_unit": "rad",
            "required_precision_1sigma_for_3sigma": float(
                reference_row["required_fractional_phase_precision_1sigma_for_3sigma"]
            ),
            "precision_unit": "fractional_phase",
            "current_precision": float(reference_row["current_fractional_phase_precision"]),
            "current_over_required": float(reference_row["current_over_required_precision"]),
            "detectable_under_current": bool(reference_row["detectable_under_current_precision"]),
            "reject_gate_3sigma": _format_gate(
                observable_label="fractional atom-interferometer phase residual",
                required_precision=float(reference_row["required_fractional_phase_precision_1sigma_for_3sigma"]),
                precision_unit="fractional phase",
            ),
            "status": "reference_row_detectable_under_current_proxy",
            "notes": (
                "This row is retained as a non-P-unique reference channel. Under the frozen apex-height envelope, "
                "current precision is already inside the required 1σ band."
            ),
        }
    )

    return rows


# 関数: `_build_clock_rows` の入出力契約と処理意図を定義する。

def _build_clock_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    main_case = _find_clock_case(metrics, case_id="earth_399m_clock_leveling")
    lever_case = _find_clock_case(metrics, case_id="sun_0p05au_to_1au")
    current_precision_row = _find_precision_row(metrics, precision_case_id="current_optical_clock")
    current_precision = float(current_precision_row["current_precision_abs"])
    lever_current_over_required = current_precision / float(lever_case["required_abs_precision_exact_1sigma_for_3sigma"])
    rows: List[Dict[str, Any]] = []

    rows.append(
        {
            "row_id": "optical_clock_main_399m",
            "source_step": "8.7.55.1.3",
            "channel_group": "optical_clock",
            "channel_role": "earth_leveling_main",
            "observable": "delta(Delta f/f)",
            "case_id": "earth_399m_clock_leveling",
            "differential_prediction_value": float(main_case["delta_z_exact"]),
            "differential_prediction_unit": "delta(f/f)",
            "required_precision_1sigma_for_3sigma": float(main_case["required_abs_precision_exact_1sigma_for_3sigma"]),
            "precision_unit": "abs_delta(f/f)",
            "current_precision": current_precision,
            "current_over_required": float(current_precision_row["current_over_required"]),
            "detectable_under_current": bool(current_precision_row["detectable"]),
            "reject_gate_3sigma": _format_gate(
                observable_label="optical-clock differential redshift residual",
                required_precision=float(main_case["required_abs_precision_exact_1sigma_for_3sigma"]),
                precision_unit="abs delta(f/f)",
            ),
            "status": "current_optical_clock_far_from_detection",
            "notes": (
                "Main Earth Δh≈399 m chronometric-leveling row. Current precision is 1.43e6x above the "
                "1σ requirement and even the 1e-21 roadmap target remains 49.6x above it."
            ),
        }
    )

    rows.append(
        {
            "row_id": "optical_clock_solar_lever_arm",
            "source_step": "8.7.55.1.3",
            "channel_group": "optical_clock",
            "channel_role": "future_lever_arm_case",
            "observable": "delta(Delta f/f)",
            "case_id": "sun_0p05au_to_1au",
            "differential_prediction_value": float(lever_case["delta_z_exact"]),
            "differential_prediction_unit": "delta(f/f)",
            "required_precision_1sigma_for_3sigma": float(lever_case["required_abs_precision_exact_1sigma_for_3sigma"]),
            "precision_unit": "abs_delta(f/f)",
            "current_precision": current_precision,
            "current_over_required": lever_current_over_required,
            "detectable_under_current": bool(current_precision <= float(lever_case["required_abs_precision_exact_1sigma_for_3sigma"])),
            "reject_gate_3sigma": _format_gate(
                observable_label="optical-clock differential redshift residual",
                required_precision=float(lever_case["required_abs_precision_exact_1sigma_for_3sigma"]),
                precision_unit="abs delta(f/f)",
            ),
            "status": "lever_arm_geometry_detectable_with_current_clock_precision",
            "notes": (
                "This is the strongest frozen lever-arm case from 8.7.55.1.3. The geometry is not a current "
                "Earth-leveling setup, but the required precision is already looser than today's clock benchmark."
            ),
        }
    )

    return rows


# 関数: `_build_summary` の入出力契約と処理意図を定義する。

def _build_summary(
    rows: List[Dict[str, Any]],
    decoherence_metrics: Dict[str, Any],
    clock_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    measurable_rows = [row for row in rows if row["current_over_required"] is not None]
    detectable_rows = [row for row in measurable_rows if bool(row["detectable_under_current"])]
    undetectable_rows = [row for row in measurable_rows if not bool(row["detectable_under_current"])]
    hardest_row = max(measurable_rows, key=lambda row: float(row["current_over_required"]))
    easiest_row = min(measurable_rows, key=lambda row: float(row["current_over_required"]))

    return {
        "row_count": len(rows),
        "measurable_row_count": len(measurable_rows),
        "rows_without_current_measurement_count": len(rows) - len(measurable_rows),
        "current_detectable_count": len(detectable_rows),
        "current_undetectable_count": len(undetectable_rows),
        "current_detectable_row_ids": [str(row["row_id"]) for row in detectable_rows],
        "current_undetectable_row_ids": [str(row["row_id"]) for row in undetectable_rows],
        "hardest_current_margin_row_id": str(hardest_row["row_id"]),
        "hardest_current_margin_over_required": float(hardest_row["current_over_required"]),
        "best_current_margin_row_id": str(easiest_row["row_id"]),
        "best_current_margin_over_required": float(easiest_row["current_over_required"]),
        "structural_parity_case_required_ratio": float(
            decoherence_metrics["decision"]["largest_parity_requirement_case"]["required_thermal_ratio_for_parity"]
        ),
        "optical_clock_main_target_1e_19_over_required": float(clock_metrics["summary"]["roadmap_target_1e_19_over_required_exact"]),
        "optical_clock_main_target_1e_20_over_required": float(clock_metrics["summary"]["roadmap_target_1e_20_over_required_exact"]),
        "optical_clock_main_target_1e_21_over_required": float(clock_metrics["summary"]["roadmap_target_1e_21_over_required_exact"]),
        "best_future_case_id": str(clock_metrics["summary"]["most_accessible_case_id"]),
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "row_id",
        "source_step",
        "channel_group",
        "channel_role",
        "observable",
        "case_id",
        "differential_prediction_value",
        "differential_prediction_unit",
        "required_precision_1sigma_for_3sigma",
        "precision_unit",
        "current_precision",
        "current_over_required",
        "detectable_under_current",
        "reject_gate_3sigma",
        "status",
        "notes",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(
    decoherence_metrics: Dict[str, Any],
    atom_metrics: Dict[str, Any],
    clock_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    rows = [_build_structural_row(decoherence_metrics), *_build_atom_rows(atom_metrics), *_build_clock_rows(clock_metrics)]
    summary = _build_summary(rows, decoherence_metrics, clock_metrics)

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.55.1.4",
            "name": "gravity-quantum differential prediction table",
        },
        "inputs": {
            "gravity_induced_decoherence_pikovski_comparison_json": str(DECOHERENCE_JSON.relative_to(ROOT)).replace("\\", "/"),
            "atom_interferometer_pikovski_phase_difference_json": str(ATOM_JSON.relative_to(ROOT)).replace("\\", "/"),
            "optical_clock_differential_frequency_json": str(CLOCK_JSON.relative_to(ROOT)).replace("\\", "/"),
        },
        "intent": (
            "Aggregate the already-frozen differential predictions from 8.7.55.1.1-.3 into a single machine-readable "
            "table with explicit 3σ rejection gates and current detectability status."
        ),
        "assumptions": [
            "No new fit parameter is introduced; every row is reconstructed from the already-frozen outputs of 8.7.55.1.1-.3.",
            "The structural decoherence row remains an entry-only parity row because no direct current precision exists for inferred (k_B T_env / chi_P) in that same regime.",
            "The P-unique atom-interferometer row is the beta/light local channel against the arm-separation Pikovski proxy.",
            "The optical-clock main row is the Earth Δh≈399 m chronometric-leveling case, and the strongest frozen lever-arm future row is the 0.05 AU ↔ 1 AU solar case.",
        ],
        "formulas": {
            "reject_gate": "Reject if |residual| > observable_specific_required_precision_1sigma_for_3sigma.",
            "structural_parity_entry": "(k_B T_env / chi_P)_parity = Gamma_Pik / (omega_*^2 tau_free)",
            "atom_interferometer_differential": "Delta phi_diff = phi^(P) - phi^(Pik)",
            "optical_clock_differential": "delta(Delta f/f) = z_P - z_GR",
        },
        "rows": rows,
        "summary": summary,
        "decision": {
            "overall_status": "differential_prediction_table_fixed",
            "new_free_parameters_introduced": False,
            "p_unique_detectable_under_current": False,
            "reference_row_detectable_under_current": True,
            "optical_clock_main_detectable_under_current": False,
            "best_future_case_id": str(summary["best_future_case_id"]),
            "next_required_steps": [
                "8.7.55.2",
                "8.7.54.23",
            ],
        },
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    decoherence_metrics = _read_json(DECOHERENCE_JSON)
    atom_metrics = _read_json(ATOM_JSON)
    clock_metrics = _read_json(CLOCK_JSON)
    payload = _build_payload(decoherence_metrics, atom_metrics, clock_metrics)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    _write_csv(OUT_CSV, payload["rows"])


if __name__ == "__main__":
    main()
