#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gravity_induced_decoherence_pikovski_comparison_audit.py

Step 8.7.55.1.1:
Freeze the first-pass structural comparison between the P-model A1
phase-diffusion rate and the gravity-induced decoherence scaling used in the
Pikovski-type route.

Inputs:
  - output/public/quantum/gravity_induced_decoherence_metrics.json
  - output/public/quantum/born_phase_diffusion_audit_metrics.json
  - output/public/quantum/gravity_quantum_interference_delta_predictions.json

Outputs:
  - output/public/quantum/gravity_induced_decoherence_pikovski_comparison_metrics.json
  - output/public/quantum/gravity_induced_decoherence_pikovski_comparison_rows.csv

Assumptions:
  - The Part III 4.5 `t_half` values are treated as the operational Pikovski-side
    Gaussian half-visibility times, so Gamma_Pik = sqrt(2 ln 2) / t_half.
  - The P-model side uses only the already-frozen A1 rate
    Gamma_deph^(P) = omega_*^2 (k_B T_env / chi_P) tau_free.
  - No new chi_P fit is introduced; the comparison is reported through the
    parity condition on (k_B T_env / chi_P) and through probe ratios in the
    already-frozen A1 regime.
"""

from __future__ import annotations

import argparse
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


GRAVITY_JSON = ROOT / "output" / "public" / "quantum" / "gravity_induced_decoherence_metrics.json"
BORN_JSON = ROOT / "output" / "public" / "quantum" / "born_phase_diffusion_audit_metrics.json"
DELTA_JSON = ROOT / "output" / "public" / "quantum" / "gravity_quantum_interference_delta_predictions.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "gravity_induced_decoherence_pikovski_comparison_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "gravity_induced_decoherence_pikovski_comparison_rows.csv"

PROBE_THERMAL_RATIOS = (1.0e-24, 1.0e-21, 1.0e-18)
ROADMAP_CLOCK_TARGETS = (1.0e-19, 1.0e-21)


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: `_safe_log10` の入出力契約と処理意図を定義する。

def _safe_log10(value: float) -> float | None:
    if not math.isfinite(value) or value <= 0.0:
        return None

    return math.log10(value)


# 関数: `_pikovski_rows` の入出力契約と処理意図を定義する。

def _pikovski_rows(gravity_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    ensemble_rows = list(gravity_metrics["derived"]["ensemble"])
    rows: List[Dict[str, Any]] = []

    for row in ensemble_rows:
        sigma_z_m = float(row["sigma_z_m"])
        t_half_s = float(row["t_half_s"])
        gamma_pik_s_inv = math.sqrt(2.0 * math.log(2.0)) / t_half_s
        rows.append(
            {
                "sigma_case_id": f"sigma_z_{sigma_z_m:.1e}_m",
                "sigma_z_m": sigma_z_m,
                "sigma_z_mm": sigma_z_m * 1.0e3,
                "t_half_s": t_half_s,
                "gamma_pik_s_inv": gamma_pik_s_inv,
                "sigma_y_fractional": float(row["sigma_y"]),
            }
        )

    return rows


# 関数: `_markov_rows` の入出力契約と処理意図を定義する。

def _markov_rows(born_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(born_metrics["representative_markov_cases"])


# 関数: `_build_comparison_row` の入出力契約と処理意図を定義する。

def _build_comparison_row(
    sigma_row: Dict[str, Any],
    markov_row: Dict[str, Any],
    probe_thermal_ratios: tuple[float, ...],
) -> Dict[str, Any]:
    gamma_pik = float(sigma_row["gamma_pik_s_inv"])
    omega_star = float(markov_row["omega_star_s_inv"])
    tau_free = float(markov_row["tau_free_s"])
    tobs = float(markov_row["tobs_s"])
    denominator = (omega_star**2) * tau_free
    required_ratio = gamma_pik / denominator
    row: Dict[str, Any] = {
        "sigma_case_id": str(sigma_row["sigma_case_id"]),
        "sigma_z_m": float(sigma_row["sigma_z_m"]),
        "sigma_z_mm": float(sigma_row["sigma_z_mm"]),
        "t_half_s": float(sigma_row["t_half_s"]),
        "gamma_pik_s_inv": gamma_pik,
        "sigma_y_fractional": float(sigma_row["sigma_y_fractional"]),
        "pmodel_case_id": str(markov_row["case_id"]),
        "pmodel_case_label": str(markov_row["label"]),
        "omega_star_s_inv": omega_star,
        "tau_free_s": tau_free,
        "tobs_s": tobs,
        "naive_tau_corr_over_tau_carrier": float(markov_row["naive_tau_corr_over_tau_carrier"]),
        "coarse_grained_tau_corr_over_tobs": float(markov_row["coarse_grained_tau_corr_over_tobs"]),
        "gamma_path_times_tobs": float(markov_row["gamma_path_times_tobs"]),
        "required_thermal_ratio_for_parity": required_ratio,
        "log10_required_thermal_ratio_for_parity": _safe_log10(required_ratio),
        "required_chi_over_kbt_for_parity": (1.0 / required_ratio) if required_ratio > 0.0 else None,
        "differential_lever": float(sigma_row["sigma_z_mm"]) / tau_free,
    }

    for thermal_ratio in probe_thermal_ratios:
        key_tag = f"{thermal_ratio:.0e}".replace("+", "")
        gamma_p = denominator * thermal_ratio
        row[f"gamma_p_over_gamma_pik_at_{key_tag}"] = gamma_p / gamma_pik
        row[f"gamma_p_tobs_at_{key_tag}"] = gamma_p * tobs
        row[f"phase_mixing_expected_at_{key_tag}"] = bool((gamma_p * tobs) > 1.0)

    return row


# 関数: `_build_precision_rows` の入出力契約と処理意図を定義する。

def _build_precision_rows(delta_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    earth_rows = delta_metrics["comparisons_earth_field"]
    atom_row = earth_rows["atom_interferometer"]
    clock_row = earth_rows["optical_clock_leveling"]
    current_clock_sigma = float(delta_metrics["baselines"]["optical_clock_leveling"]["sigma_z_clock_abs"])
    clock_required = float(clock_row["required_abs_precision_for_3sigma_delta_f_over_f"])
    rows: List[Dict[str, Any]] = [
        {
            "channel_id": "atom_interferometer_current",
            "label": "atom interferometer (current reference)",
            "observable": "fractional phase precision",
            "current_precision": float(delta_metrics["baselines"]["atom_interferometer"]["current_phase_fractional_precision_assumed"]),
            "required_precision_3sigma": float(atom_row["required_fractional_precision_for_3sigma"]),
            "precision_margin_current_over_required": float(
                delta_metrics["baselines"]["atom_interferometer"]["current_phase_fractional_precision_assumed"]
            )
            / float(atom_row["required_fractional_precision_for_3sigma"]),
        },
        {
            "channel_id": "optical_clock_current",
            "label": "optical clock leveling (current reference)",
            "observable": "absolute delta(f/f)",
            "current_precision": current_clock_sigma,
            "required_precision_3sigma": clock_required,
            "precision_margin_current_over_required": current_clock_sigma / clock_required,
        },
    ]

    for target in ROADMAP_CLOCK_TARGETS:
        rows.append(
            {
                "channel_id": f"optical_clock_target_{target:.0e}".replace("+", ""),
                "label": f"optical clock roadmap target ({target:.0e})",
                "observable": "absolute delta(f/f)",
                "current_precision": target,
                "required_precision_3sigma": clock_required,
                "precision_margin_current_over_required": target / clock_required,
            }
        )

    return rows


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(
    gravity_metrics: Dict[str, Any],
    born_metrics: Dict[str, Any],
    delta_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    sigma_rows = _pikovski_rows(gravity_metrics)
    markov_rows = _markov_rows(born_metrics)
    comparison_rows: List[Dict[str, Any]] = []

    for sigma_row in sigma_rows:
        for markov_row in markov_rows:
            comparison_rows.append(_build_comparison_row(sigma_row, markov_row, PROBE_THERMAL_RATIOS))

    precision_rows = _build_precision_rows(delta_metrics)
    max_required_row = max(comparison_rows, key=lambda row: float(row["required_thermal_ratio_for_parity"]))
    min_required_row = min(comparison_rows, key=lambda row: float(row["required_thermal_ratio_for_parity"]))
    probe_tags = [f"{value:.0e}".replace("+", "") for value in PROBE_THERMAL_RATIOS]
    probe_summary: List[Dict[str, Any]] = []

    for thermal_ratio, probe_tag in zip(PROBE_THERMAL_RATIOS, probe_tags, strict=True):
        key_ratio = f"gamma_p_over_gamma_pik_at_{probe_tag}"
        key_gate = f"phase_mixing_expected_at_{probe_tag}"
        best_row = max(comparison_rows, key=lambda row: float(row[key_ratio]))
        probe_summary.append(
            {
                "thermal_ratio_kbt_over_chiP": thermal_ratio,
                "max_gamma_p_over_gamma_pik": float(best_row[key_ratio]),
                "maximizing_sigma_case_id": str(best_row["sigma_case_id"]),
                "maximizing_pmodel_case_id": str(best_row["pmodel_case_id"]),
                "phase_mixing_expected_count": sum(1 for row in comparison_rows if bool(row[key_gate])),
            }
        )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.55.1.1",
            "name": "Gamma_deph vs Pikovski structural comparison",
        },
        "inputs": {
            "gravity_induced_decoherence_json": str(GRAVITY_JSON.relative_to(ROOT)).replace("\\", "/"),
            "born_phase_diffusion_json": str(BORN_JSON.relative_to(ROOT)).replace("\\", "/"),
            "gravity_quantum_delta_json": str(DELTA_JSON.relative_to(ROOT)).replace("\\", "/"),
        },
        "intent": (
            "Freeze the first-pass differential comparison between the Part III 4.5 gravity-induced decoherence "
            "half-times and the A1 P-model dephasing rate without introducing any new fit parameter."
        ),
        "assumptions": [
            "Gamma_Pik is operationally identified from the Part III 4.5 half-visibility time through Gamma_Pik = sqrt(2 ln 2) / t_half.",
            "The P-model side keeps omega_* = 3e15 s^-1 and tau_free from the frozen A1 representative Markov cases.",
            "The comparison is reported through the parity requirement on k_B T_env / chi_P and through probe ratios 1e-24 / 1e-21 / 1e-18 spanning the frozen A1 regime.",
            "Current and roadmap precision references are read from the existing unified gravity×quantum delta audit and the roadmap-fixed future clock targets.",
        ],
        "formulas": {
            "pmodel": "Gamma_deph^(P) = omega_*^2 (k_B T_env / chi_P) tau_free",
            "pikovski_operational": "Gamma_Pik = sqrt(2 ln 2) / t_half",
            "parity_ratio": "(k_B T_env / chi_P)_parity = Gamma_Pik / (omega_*^2 tau_free)",
            "comparison_ratio": "Gamma_deph^(P) / Gamma_Pik = (k_B T_env / chi_P) / (k_B T_env / chi_P)_parity",
        },
        "probe_thermal_ratios_kbt_over_chiP": list(PROBE_THERMAL_RATIOS),
        "pikovski_rows": sigma_rows,
        "pmodel_markov_rows": markov_rows,
        "comparison_rows": comparison_rows,
        "precision_reference_rows": precision_rows,
        "summary": {
            "sigma_case_count": len(sigma_rows),
            "pmodel_case_count": len(markov_rows),
            "comparison_row_count": len(comparison_rows),
            "max_required_thermal_ratio_for_parity": float(max_required_row["required_thermal_ratio_for_parity"]),
            "max_required_thermal_ratio_case": {
                "sigma_case_id": str(max_required_row["sigma_case_id"]),
                "pmodel_case_id": str(max_required_row["pmodel_case_id"]),
            },
            "min_required_thermal_ratio_for_parity": float(min_required_row["required_thermal_ratio_for_parity"]),
            "min_required_thermal_ratio_case": {
                "sigma_case_id": str(min_required_row["sigma_case_id"]),
                "pmodel_case_id": str(min_required_row["pmodel_case_id"]),
            },
            "probe_ratio_summary": probe_summary,
            "atom_interferometer_margin_current_over_required": float(precision_rows[0]["precision_margin_current_over_required"]),
            "optical_clock_margin_current_over_required": float(precision_rows[1]["precision_margin_current_over_required"]),
            "optical_clock_margin_target_1e_19_over_required": float(precision_rows[2]["precision_margin_current_over_required"]),
            "optical_clock_margin_target_1e_21_over_required": float(precision_rows[3]["precision_margin_current_over_required"]),
        },
        "decision": {
            "overall_status": "structural_difference_envelope_fixed",
            "new_free_parameters_introduced": False,
            "differential_entry_exists": True,
            "largest_parity_requirement_case": {
                "sigma_z_mm": float(max_required_row["sigma_z_mm"]),
                "t_half_s": float(max_required_row["t_half_s"]),
                "tau_free_s": float(max_required_row["tau_free_s"]),
                "required_thermal_ratio_for_parity": float(max_required_row["required_thermal_ratio_for_parity"]),
            },
            "next_required_steps": ["8.7.55.1.2", "8.7.55.1.3", "8.7.55.1.4"],
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "sigma_case_id",
        "sigma_z_m",
        "sigma_z_mm",
        "t_half_s",
        "gamma_pik_s_inv",
        "pmodel_case_id",
        "pmodel_case_label",
        "omega_star_s_inv",
        "tau_free_s",
        "tobs_s",
        "naive_tau_corr_over_tau_carrier",
        "coarse_grained_tau_corr_over_tobs",
        "gamma_path_times_tobs",
        "required_thermal_ratio_for_parity",
        "log10_required_thermal_ratio_for_parity",
        "required_chi_over_kbt_for_parity",
        "gamma_p_over_gamma_pik_at_1e-24",
        "gamma_p_tobs_at_1e-24",
        "phase_mixing_expected_at_1e-24",
        "gamma_p_over_gamma_pik_at_1e-21",
        "gamma_p_tobs_at_1e-21",
        "phase_mixing_expected_at_1e-21",
        "gamma_p_over_gamma_pik_at_1e-18",
        "gamma_p_tobs_at_1e-18",
        "phase_mixing_expected_at_1e-18",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the structural comparison between the A1 P-model dephasing rate and the Pikovski gravity-induced decoherence route."
    )
    parser.add_argument("--gravity-json", default=str(GRAVITY_JSON), help="Part III 4.5 gravity-induced decoherence metrics JSON.")
    parser.add_argument("--born-json", default=str(BORN_JSON), help="A1 phase-diffusion audit metrics JSON.")
    parser.add_argument("--delta-json", default=str(DELTA_JSON), help="Unified gravity×quantum delta-prediction JSON.")
    parser.add_argument("--out-json", default=str(OUT_JSON), help="Output metrics JSON path.")
    parser.add_argument("--out-csv", default=str(OUT_CSV), help="Output CSV path.")
    args = parser.parse_args()

    gravity_json_path = Path(args.gravity_json)
    born_json_path = Path(args.born_json)
    delta_json_path = Path(args.delta_json)
    out_json_path = Path(args.out_json)
    out_csv_path = Path(args.out_csv)

    if not gravity_json_path.is_absolute():
        gravity_json_path = (ROOT / gravity_json_path).resolve()

    if not born_json_path.is_absolute():
        born_json_path = (ROOT / born_json_path).resolve()

    if not delta_json_path.is_absolute():
        delta_json_path = (ROOT / delta_json_path).resolve()

    if not out_json_path.is_absolute():
        out_json_path = (ROOT / out_json_path).resolve()

    if not out_csv_path.is_absolute():
        out_csv_path = (ROOT / out_csv_path).resolve()

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    gravity_metrics = _read_json(gravity_json_path)
    born_metrics = _read_json(born_json_path)
    delta_metrics = _read_json(delta_json_path)
    payload = _build_payload(gravity_metrics, born_metrics, delta_metrics)

    with out_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    _write_csv(out_csv_path, payload["comparison_rows"])


if __name__ == "__main__":
    main()
