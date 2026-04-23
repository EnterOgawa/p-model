#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
atom_interferometer_pikovski_phase_difference_audit.py

Step 8.7.55.1.2:
Freeze the atom-interferometer differential phase comparison between the
already-frozen P-model correction channels and a Pikovski-style
gravitational-time-dilation phase proxy.

Inputs:
  - output/public/quantum/atom_interferometer_gravimeter_phase_metrics.json
  - output/public/quantum/gravity_quantum_interference_delta_predictions.json

Outputs:
  - output/public/quantum/atom_interferometer_pikovski_phase_difference_metrics.json
  - output/public/quantum/atom_interferometer_pikovski_phase_difference_rows.csv

Assumptions:
  - The Part III 4.3 atom-interferometer baseline phase is frozen in
    `atom_interferometer_gravimeter_phase_metrics.json`.
  - The weak-field P-vs-GR reference row already frozen in
    `gravity_quantum_interference_delta_predictions.json` is kept as a
    reference channel, while the P-unique channel is the β/light-propagation
    correction from the chosen local-differential model.
  - The Pikovski-side phase is evaluated as an operational time-dilation
    proxy, `phi_Pik ≈ omega_carrier * g * delta_z * T / c^2`, using only
    already-frozen geometry/readout scales:
      - local arm-separation scale from the unified atom audit
      - apex-height envelope from the atom-fountain geometry
    No new carrier fit or internal-frequency parameter is introduced.
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


ATOM_JSON = ROOT / "output" / "public" / "quantum" / "atom_interferometer_gravimeter_phase_metrics.json"
DELTA_JSON = ROOT / "output" / "public" / "quantum" / "gravity_quantum_interference_delta_predictions.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "atom_interferometer_pikovski_phase_difference_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "atom_interferometer_pikovski_phase_difference_rows.csv"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: `_pikovski_phase_proxy` の入出力契約と処理意図を定義する。

def _pikovski_phase_proxy(*, omega_carrier_rad_s: float, g_m_per_s2: float, delta_z_m: float, t_s: float, c_m_per_s: float) -> float:
    return float(omega_carrier_rad_s * g_m_per_s2 * delta_z_m * t_s / (c_m_per_s**2))


# 関数: `_fractional_from_abs` の入出力契約と処理意図を定義する。

def _fractional_from_abs(*, abs_phase_rad: float, phi_ref_rad: float) -> float:
    return float(abs_phase_rad / phi_ref_rad)


# 関数: `_required_abs_precision_for_3sigma` の入出力契約と処理意図を定義する。

def _required_abs_precision_for_3sigma(delta_phase_rad: float) -> float:
    return float(abs(delta_phase_rad) / 3.0)


# 関数: `_build_row` の入出力契約と処理意図を定義する。

def _build_row(
    *,
    pmodel_case_id: str,
    pmodel_phase_rad: float,
    pikovski_case_id: str,
    pikovski_phase_rad: float,
    phi_ref_rad: float,
    current_abs_precision_rad: float,
    current_fractional_precision: float,
) -> Dict[str, Any]:
    delta_phi_diff = float(pmodel_phase_rad - pikovski_phase_rad)
    required_abs = _required_abs_precision_for_3sigma(delta_phi_diff)
    required_fractional = _fractional_from_abs(abs_phase_rad=required_abs, phi_ref_rad=phi_ref_rad)
    current_over_required = float(current_abs_precision_rad / required_abs) if required_abs > 0.0 else float("nan")
    return {
        "pmodel_case_id": pmodel_case_id,
        "pmodel_phase_rad": float(pmodel_phase_rad),
        "pikovski_case_id": pikovski_case_id,
        "pikovski_phase_rad": float(pikovski_phase_rad),
        "delta_phi_diff_rad": delta_phi_diff,
        "delta_phi_diff_over_phi_ref": float(delta_phi_diff / phi_ref_rad),
        "required_abs_phase_precision_1sigma_for_3sigma_rad": required_abs,
        "required_fractional_phase_precision_1sigma_for_3sigma": required_fractional,
        "current_abs_phase_precision_rad": float(current_abs_precision_rad),
        "current_fractional_phase_precision": float(current_fractional_precision),
        "current_over_required_precision": current_over_required,
        "detectable_under_current_precision": bool(current_abs_precision_rad <= required_abs),
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "pmodel_case_id",
        "pmodel_phase_rad",
        "pikovski_case_id",
        "pikovski_phase_rad",
        "delta_phi_diff_rad",
        "delta_phi_diff_over_phi_ref",
        "required_abs_phase_precision_1sigma_for_3sigma_rad",
        "required_fractional_phase_precision_1sigma_for_3sigma",
        "current_abs_phase_precision_rad",
        "current_fractional_phase_precision",
        "current_over_required_precision",
        "detectable_under_current_precision",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(atom_metrics: Dict[str, Any], delta_metrics: Dict[str, Any]) -> Dict[str, Any]:
    c_m_per_s = 299_792_458.0
    config = atom_metrics["config"]
    results = atom_metrics["results"]
    beta_block = results["beta_phase_dependence"]
    beta_model = beta_block["models"]["B_differential_within_interferometer"]
    baselines = delta_metrics["baselines"]
    earth_atom = delta_metrics["comparisons_earth_field"]["atom_interferometer"]

    lambda_m = float(config["lambda_m"])
    g_m_per_s2 = float(config["g_m_per_s2"])
    t_s = float(config["T_s"])
    phi_ref_rad = float(results["phi_ref_rad"])
    omega_carrier_rad_s = float(2.0 * math.pi * c_m_per_s / lambda_m)
    current_fractional_precision = float(baselines["atom_interferometer"]["current_phase_fractional_precision_assumed"])
    current_abs_precision_rad = float(phi_ref_rad * current_fractional_precision)

    arm_separation_m = float(baselines["atom_interferometer"]["arm_separation_scale_m"])
    apex_height_m = float(beta_model["assumptions"]["apex_height_m"])

    pikovski_arm_proxy = _pikovski_phase_proxy(
        omega_carrier_rad_s=omega_carrier_rad_s,
        g_m_per_s2=g_m_per_s2,
        delta_z_m=arm_separation_m,
        t_s=t_s,
        c_m_per_s=c_m_per_s,
    )
    pikovski_apex_envelope = _pikovski_phase_proxy(
        omega_carrier_rad_s=omega_carrier_rad_s,
        g_m_per_s2=g_m_per_s2,
        delta_z_m=apex_height_m,
        t_s=t_s,
        c_m_per_s=c_m_per_s,
    )

    pmodel_reference_phase = float(earth_atom["delta_phase_rad_est"])
    pmodel_beta_phase = float(beta_model["delta_phase_rad"])

    rows = [
        _build_row(
            pmodel_case_id="pmodel_reference_mapping",
            pmodel_phase_rad=pmodel_reference_phase,
            pikovski_case_id="pikovski_arm_separation_proxy",
            pikovski_phase_rad=pikovski_arm_proxy,
            phi_ref_rad=phi_ref_rad,
            current_abs_precision_rad=current_abs_precision_rad,
            current_fractional_precision=current_fractional_precision,
        ),
        _build_row(
            pmodel_case_id="pmodel_reference_mapping",
            pmodel_phase_rad=pmodel_reference_phase,
            pikovski_case_id="pikovski_apex_height_envelope",
            pikovski_phase_rad=pikovski_apex_envelope,
            phi_ref_rad=phi_ref_rad,
            current_abs_precision_rad=current_abs_precision_rad,
            current_fractional_precision=current_fractional_precision,
        ),
        _build_row(
            pmodel_case_id="pmodel_beta_light_local",
            pmodel_phase_rad=pmodel_beta_phase,
            pikovski_case_id="pikovski_arm_separation_proxy",
            pikovski_phase_rad=pikovski_arm_proxy,
            phi_ref_rad=phi_ref_rad,
            current_abs_precision_rad=current_abs_precision_rad,
            current_fractional_precision=current_fractional_precision,
        ),
        _build_row(
            pmodel_case_id="pmodel_beta_light_local",
            pmodel_phase_rad=pmodel_beta_phase,
            pikovski_case_id="pikovski_apex_height_envelope",
            pikovski_phase_rad=pikovski_apex_envelope,
            phi_ref_rad=phi_ref_rad,
            current_abs_precision_rad=current_abs_precision_rad,
            current_fractional_precision=current_fractional_precision,
        ),
    ]

    tightest_row = min(rows, key=lambda row: float(row["current_over_required_precision"]))
    loosest_row = max(rows, key=lambda row: float(row["current_over_required_precision"]))

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.55.1.2",
            "name": "atom interferometer differential phase vs Pikovski proxy",
        },
        "inputs": {
            "atom_interferometer_metrics_json": str(ATOM_JSON.relative_to(ROOT)).replace("\\", "/"),
            "gravity_quantum_delta_json": str(DELTA_JSON.relative_to(ROOT)).replace("\\", "/"),
        },
        "intent": (
            "Freeze the atom-interferometer differential phase comparison between the already-frozen "
            "P-model correction channels and a Pikovski-style time-dilation phase proxy without "
            "introducing any new carrier or geometry fit."
        ),
        "assumptions": [
            "The P-model side keeps two already-frozen channels: the Part III 4.3 weak-field mapping reference row and the β/light local differential row.",
            "The Pikovski side is evaluated as an operational phase proxy phi_Pik ≈ omega_carrier g delta_z T / c^2, using the Raman carrier frequency from lambda=852 nm.",
            "Two geometry choices are frozen: the local arm-separation scale from the unified atom audit and the apex-height envelope from the atom-fountain geometry.",
            "The current atom-interferometer precision reference remains the existing fractional phase benchmark 1e-9 from gravity_quantum_interference_delta_predictions.json.",
        ],
        "formulas": {
            "pmodel_reference_row": "delta_phi_ref = phi_ref * delta_z_over_z_GR  (already frozen in Part III 4.3 / unified audit)",
            "pmodel_beta_light_row": "delta_phi_beta ≈ (beta-1) * k_eff * g^3 * T^4 / c^2",
            "pikovski_proxy": "phi_Pik ≈ omega_carrier * g * delta_z * T / c^2",
            "differential": "Delta phi_diff = phi^(P) - phi^(Pik)",
            "required_precision": "sigma_phase,1sigma <= |Delta phi_diff| / 3",
        },
        "frozen_setup": {
            "lambda_m": lambda_m,
            "omega_carrier_rad_s": omega_carrier_rad_s,
            "g_m_per_s2": g_m_per_s2,
            "T_s": t_s,
            "phi_ref_rad": phi_ref_rad,
            "current_fractional_phase_precision": current_fractional_precision,
            "current_abs_phase_precision_rad": current_abs_precision_rad,
            "arm_separation_scale_m": arm_separation_m,
            "apex_height_m": apex_height_m,
            "pmodel_reference_phase_rad": pmodel_reference_phase,
            "pmodel_beta_light_phase_rad": pmodel_beta_phase,
            "pikovski_arm_separation_proxy_phase_rad": pikovski_arm_proxy,
            "pikovski_apex_height_envelope_phase_rad": pikovski_apex_envelope,
        },
        "comparison_rows": rows,
        "summary": {
            "row_count": len(rows),
            "tightest_current_over_required_precision": float(tightest_row["current_over_required_precision"]),
            "tightest_case": {
                "pmodel_case_id": str(tightest_row["pmodel_case_id"]),
                "pikovski_case_id": str(tightest_row["pikovski_case_id"]),
            },
            "loosest_current_over_required_precision": float(loosest_row["current_over_required_precision"]),
            "loosest_case": {
                "pmodel_case_id": str(loosest_row["pmodel_case_id"]),
                "pikovski_case_id": str(loosest_row["pikovski_case_id"]),
            },
            "beta_light_vs_arm_proxy_delta_phi_rad": float(rows[2]["delta_phi_diff_rad"]),
            "beta_light_vs_apex_envelope_delta_phi_rad": float(rows[3]["delta_phi_diff_rad"]),
            "reference_vs_arm_proxy_delta_phi_rad": float(rows[0]["delta_phi_diff_rad"]),
            "reference_vs_apex_envelope_delta_phi_rad": float(rows[1]["delta_phi_diff_rad"]),
        },
        "decision": {
            "overall_status": "atom_interferometer_differential_proxy_fixed",
            "new_free_parameters_introduced": False,
            "p_unique_channel_fixed": True,
            "legacy_reference_row_retained": True,
            "next_required_steps": ["8.7.55.1.3", "8.7.55.1.4"],
        },
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the atom-interferometer differential phase comparison against a Pikovski-style time-dilation proxy."
    )
    parser.add_argument("--atom-json", default=str(ATOM_JSON), help="Atom-interferometer gravimeter metrics JSON.")
    parser.add_argument("--delta-json", default=str(DELTA_JSON), help="Unified gravity×quantum delta audit JSON.")
    parser.add_argument("--out-json", default=str(OUT_JSON), help="Output metrics JSON path.")
    parser.add_argument("--out-csv", default=str(OUT_CSV), help="Output CSV path.")
    args = parser.parse_args()

    atom_json_path = Path(args.atom_json)
    delta_json_path = Path(args.delta_json)
    out_json_path = Path(args.out_json)
    out_csv_path = Path(args.out_csv)

    # 条件分岐: `not atom_json_path.is_absolute()` を満たす経路を評価する。
    if not atom_json_path.is_absolute():
        atom_json_path = (ROOT / atom_json_path).resolve()

    # 条件分岐: `not delta_json_path.is_absolute()` を満たす経路を評価する。

    if not delta_json_path.is_absolute():
        delta_json_path = (ROOT / delta_json_path).resolve()

    # 条件分岐: `not out_json_path.is_absolute()` を満たす経路を評価する。

    if not out_json_path.is_absolute():
        out_json_path = (ROOT / out_json_path).resolve()

    # 条件分岐: `not out_csv_path.is_absolute()` を満たす経路を評価する。

    if not out_csv_path.is_absolute():
        out_csv_path = (ROOT / out_csv_path).resolve()

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    atom_metrics = _read_json(atom_json_path)
    delta_metrics = _read_json(delta_json_path)
    payload = _build_payload(atom_metrics, delta_metrics)

    with out_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    _write_csv(out_csv_path, payload["comparison_rows"])


if __name__ == "__main__":
    main()
