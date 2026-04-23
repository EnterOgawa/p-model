#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optical_clock_differential_frequency_audit.py

Step 8.7.55.1.3:
Freeze the optical-clock differential observable between the P-model stationary
clock map and the GR Schwarzschild map, both exactly and through the weak-field
second-order expansion requested in the roadmap.

Inputs:
  - output/public/quantum/optical_clock_chronometric_leveling_metrics.json
  - output/public/quantum/gravity_quantum_interference_delta_predictions.json

Outputs:
  - output/public/quantum/optical_clock_differential_frequency_metrics.json
  - output/public/quantum/optical_clock_differential_frequency_rows.csv

Assumptions:
  - The current chronometric-leveling benchmark is the already-frozen
    Δh≈399 m / ΔU≈3915.88 m² s⁻² case from arXiv:2309.14953.
  - The exact weak-field comparison remains the stationary-clock map already
    frozen elsewhere in the repo:
      (dτ/dt)_P = exp(-x), (dτ/dt)_GR = sqrt(1-2x), x = GM/(c²r).
  - The roadmap request for a "2nd-order nonlinear expansion" is evaluated by
    truncating both stationary-clock rates to O(x²) before forming the ratio.
  - No new fit parameter or mission-specific extrapolation is introduced;
    roadmap targets 1e-19 / 1e-20 / 1e-21 are treated only as precision probes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]

# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CLOCK_JSON = ROOT / "output" / "public" / "quantum" / "optical_clock_chronometric_leveling_metrics.json"
DELTA_JSON = ROOT / "output" / "public" / "quantum" / "gravity_quantum_interference_delta_predictions.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "optical_clock_differential_frequency_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "optical_clock_differential_frequency_rows.csv"

ROADMAP_TARGETS = (1.0e-19, 1.0e-20, 1.0e-21)
DECIMAL_PRECISION = 80


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: `_decimal` の入出力契約と処理意図を定義する。

def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


# 関数: `_x` の入出力契約と処理意図を定義する。

def _x(*, gm_m3_s2: float, r_m: float, c_m_per_s: float) -> float:
    return float(gm_m3_s2 / ((c_m_per_s**2) * r_m))


# 関数: `_required_1sigma_for_3sigma` の入出力契約と処理意図を定義する。

def _required_1sigma_for_3sigma(delta_abs: float) -> float:
    return float(abs(delta_abs) / 3.0)


# 関数: `_stable_exact_redshift` の入出力契約と処理意図を定義する。

def _stable_exact_redshift(*, x_low: float, x_high: float) -> Dict[str, float]:
    with localcontext() as ctx:
        ctx.prec = DECIMAL_PRECISION
        x_low_dec = _decimal(x_low)
        x_high_dec = _decimal(x_high)
        delta_x_dec = x_low_dec - x_high_dec
        z_p_dec = delta_x_dec.exp() - Decimal(1)
        ratio_gr_dec = ((Decimal(1) - Decimal(2) * x_high_dec) / (Decimal(1) - Decimal(2) * x_low_dec)).sqrt()
        z_gr_dec = ratio_gr_dec - Decimal(1)
        delta_z_dec = z_p_dec - z_gr_dec

    return {
        "delta_x_linear": float(delta_x_dec),
        "z_p_exact": float(z_p_dec),
        "z_gr_exact": float(z_gr_dec),
        "delta_z_exact": float(delta_z_dec),
        "delta_z_exact_over_z_gr_exact": float(delta_z_dec / z_gr_dec) if z_gr_dec != 0 else float("nan"),
    }


# 関数: `_stable_second_order_redshift` の入出力契約と処理意図を定義する。

def _stable_second_order_redshift(*, x_low: float, x_high: float) -> Dict[str, float]:
    with localcontext() as ctx:
        ctx.prec = DECIMAL_PRECISION
        x_low_dec = _decimal(x_low)
        x_high_dec = _decimal(x_high)
        delta_x_dec = x_low_dec - x_high_dec

        rate_p_low = Decimal(1) - x_low_dec + (x_low_dec * x_low_dec) / Decimal(2)
        rate_p_high = Decimal(1) - x_high_dec + (x_high_dec * x_high_dec) / Decimal(2)
        rate_gr_low = Decimal(1) - x_low_dec - (x_low_dec * x_low_dec) / Decimal(2)
        rate_gr_high = Decimal(1) - x_high_dec - (x_high_dec * x_high_dec) / Decimal(2)

        z_p_dec = rate_p_high / rate_p_low - Decimal(1)
        z_gr_dec = rate_gr_high / rate_gr_low - Decimal(1)
        delta_z_dec = z_p_dec - z_gr_dec
        p_specific_dec = z_p_dec - delta_x_dec
        gr_specific_dec = z_gr_dec - delta_x_dec

    return {
        "z_p_second_order": float(z_p_dec),
        "z_gr_second_order": float(z_gr_dec),
        "delta_z_second_order": float(delta_z_dec),
        "p_specific_second_order_term": float(p_specific_dec),
        "gr_specific_second_order_term": float(gr_specific_dec),
        "delta_z_second_order_over_z_gr_second_order": float(delta_z_dec / z_gr_dec) if z_gr_dec != 0 else float("nan"),
    }


# 関数: `_build_case` の入出力契約と処理意図を定義する。

def _build_case(
    *,
    case_id: str,
    label: str,
    gm_m3_s2: float,
    r_low_m: float,
    r_high_m: float,
    c_m_per_s: float,
    frozen_exact: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    x_low = _x(gm_m3_s2=gm_m3_s2, r_m=r_low_m, c_m_per_s=c_m_per_s)
    x_high = _x(gm_m3_s2=gm_m3_s2, r_m=r_high_m, c_m_per_s=c_m_per_s)
    exact_row = _stable_exact_redshift(x_low=x_low, x_high=x_high)
    second_order_row = _stable_second_order_redshift(x_low=x_low, x_high=x_high)

    frozen_z_p = None
    frozen_z_gr = None
    frozen_delta_z = None
    frozen_rel = None
    exact_match_abs_error = None

    if isinstance(frozen_exact, dict):
        frozen_z_p = float(frozen_exact["z_p"])
        frozen_z_gr = float(frozen_exact["z_gr"])
        frozen_delta_z = float(frozen_exact["delta_z"])
        frozen_rel = float(frozen_exact["delta_z_over_z_gr"])
        exact_match_abs_error = abs(exact_row["delta_z_exact"] - frozen_delta_z)

    canonical_delta_z_exact = frozen_delta_z if frozen_delta_z is not None else float(exact_row["delta_z_exact"])
    canonical_z_p_exact = frozen_z_p if frozen_z_p is not None else float(exact_row["z_p_exact"])
    canonical_z_gr_exact = frozen_z_gr if frozen_z_gr is not None else float(exact_row["z_gr_exact"])
    canonical_rel = frozen_rel if frozen_rel is not None else float(exact_row["delta_z_exact_over_z_gr_exact"])

    return {
        "case_id": case_id,
        "label": label,
        "r_low_m": float(r_low_m),
        "r_high_m": float(r_high_m),
        "x_low": x_low,
        "x_high": x_high,
        "delta_x_linear": float(exact_row["delta_x_linear"]),
        "z_p_exact": canonical_z_p_exact,
        "z_gr_exact": canonical_z_gr_exact,
        "delta_z_exact": canonical_delta_z_exact,
        "delta_z_exact_recomputed": float(exact_row["delta_z_exact"]),
        "delta_z_exact_match_abs_error": exact_match_abs_error,
        "z_p_second_order": float(second_order_row["z_p_second_order"]),
        "z_gr_second_order": float(second_order_row["z_gr_second_order"]),
        "delta_z_second_order": float(second_order_row["delta_z_second_order"]),
        "p_specific_second_order_term": float(second_order_row["p_specific_second_order_term"]),
        "gr_specific_second_order_term": float(second_order_row["gr_specific_second_order_term"]),
        "delta_z_exact_over_z_gr_exact": canonical_rel,
        "delta_z_second_order_over_z_gr_second_order": float(second_order_row["delta_z_second_order_over_z_gr_second_order"]),
        "required_abs_precision_exact_1sigma_for_3sigma": _required_1sigma_for_3sigma(canonical_delta_z_exact),
        "required_abs_precision_second_order_1sigma_for_3sigma": _required_1sigma_for_3sigma(
            float(second_order_row["delta_z_second_order"])
        ),
    }


# 関数: `_build_precision_rows` の入出力契約と処理意図を定義する。

def _build_precision_rows(*, required_abs_precision: float, current_precision: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = [
        {
            "precision_case_id": "current_optical_clock",
            "label": "current optical clock leveling",
            "current_precision_abs": current_precision,
            "required_precision_abs": required_abs_precision,
            "current_over_required": float(current_precision / required_abs_precision),
            "detectable": bool(current_precision <= required_abs_precision),
        }
    ]

    for target in ROADMAP_TARGETS:
        rows.append(
            {
                "precision_case_id": f"roadmap_target_{target:.0e}".replace("+", ""),
                "label": f"roadmap target {target:.0e}",
                "current_precision_abs": target,
                "required_precision_abs": required_abs_precision,
                "current_over_required": float(target / required_abs_precision),
                "detectable": bool(target <= required_abs_precision),
            }
        )

    return rows


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "label",
        "x_low",
        "x_high",
        "delta_x_linear",
        "z_p_exact",
        "z_gr_exact",
        "delta_z_exact",
        "delta_z_exact_recomputed",
        "delta_z_exact_match_abs_error",
        "z_p_second_order",
        "z_gr_second_order",
        "delta_z_second_order",
        "p_specific_second_order_term",
        "gr_specific_second_order_term",
        "required_abs_precision_exact_1sigma_for_3sigma",
        "required_abs_precision_second_order_1sigma_for_3sigma",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(clock_metrics: Dict[str, Any], delta_metrics: Dict[str, Any]) -> Dict[str, Any]:
    constants = delta_metrics["constants"]
    baselines = delta_metrics["baselines"]
    earth_rows = delta_metrics["comparisons_earth_field"]
    future_examples = delta_metrics["future_regimes_examples"]

    c_m_per_s = float(constants["c_m_per_s"])
    gm_earth_m3_s2 = float(constants["gm_earth_m3_s2"])
    r_earth_m = float(constants["r_earth_m"])
    gm_sun_m3_s2 = float(constants["gm_sun_m3_s2"])
    au_m = float(constants["au_m"])

    clock_baseline = baselines["optical_clock_leveling"]
    height_scale_m = float(clock_baseline["height_scale_m"])
    current_sigma_abs = float(clock_baseline["sigma_z_clock_abs"])

    main_case = _build_case(
        case_id="earth_399m_clock_leveling",
        label="Earth field, Δh≈399 m chronometric leveling",
        gm_m3_s2=gm_earth_m3_s2,
        r_low_m=r_earth_m,
        r_high_m=r_earth_m + height_scale_m,
        c_m_per_s=c_m_per_s,
        frozen_exact=earth_rows["optical_clock_leveling"]["redshift"],
    )
    iss_case = _build_case(
        case_id="earth_400km_iss_like",
        label="Earth field, ground↔ISS-like 400 km",
        gm_m3_s2=gm_earth_m3_s2,
        r_low_m=r_earth_m,
        r_high_m=float(future_examples["iss_400km"]["r_high_m"]),
        c_m_per_s=c_m_per_s,
        frozen_exact=future_examples["iss_400km"]["redshift"],
    )
    sun_03_case = _build_case(
        case_id="sun_0p3au_to_1au",
        label="Solar potential lever arm, 0.3 AU ↔ 1 AU",
        gm_m3_s2=gm_sun_m3_s2,
        r_low_m=0.3 * au_m,
        r_high_m=au_m,
        c_m_per_s=c_m_per_s,
        frozen_exact=future_examples["sun_0p3au_to_1au"]["redshift"],
    )
    sun_005_case = _build_case(
        case_id="sun_0p05au_to_1au",
        label="Solar potential lever arm, 0.05 AU ↔ 1 AU",
        gm_m3_s2=gm_sun_m3_s2,
        r_low_m=0.05 * au_m,
        r_high_m=au_m,
        c_m_per_s=c_m_per_s,
        frozen_exact=future_examples["sun_0p05au_to_1au"]["redshift"],
    )

    case_rows = [main_case, iss_case, sun_03_case, sun_005_case]
    precision_rows = _build_precision_rows(
        required_abs_precision=float(main_case["required_abs_precision_exact_1sigma_for_3sigma"]),
        current_precision=current_sigma_abs,
    )

    most_accessible_case = max(case_rows, key=lambda row: float(row["required_abs_precision_exact_1sigma_for_3sigma"]))
    largest_exact_case = max(case_rows, key=lambda row: abs(float(row["delta_z_exact"])))
    second_order_gap = abs(float(main_case["delta_z_exact"]) - float(main_case["delta_z_second_order"]))

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.55.1.3",
            "name": "optical clock differential frequency audit",
        },
        "inputs": {
            "optical_clock_metrics_json": str(CLOCK_JSON.relative_to(ROOT)).replace("\\", "/"),
            "gravity_quantum_delta_json": str(DELTA_JSON.relative_to(ROOT)).replace("\\", "/"),
        },
        "intent": (
            "Freeze the P-model-vs-GR optical-clock differential observable at Δh≈399 m, "
            "including the weak-field second-order expansion requested in the roadmap, and "
            "compare it with current and roadmap precision targets."
        ),
        "assumptions": [
            "The main benchmark remains the already-frozen chronometric-leveling case with Δh≈399 m and σ(Δf/f)≈2.89e-17.",
            "The exact comparison uses the stationary-clock maps exp(-x) and sqrt(1-2x), with x = GM/(c^2 r).",
            "The second-order comparison is formed by truncating both stationary-clock rates to O(x^2) before taking the altitude ratio.",
            "Roadmap precision targets 1e-19 / 1e-20 / 1e-21 are used only as sensitivity probes, not as claimed achieved measurements.",
        ],
        "formulas": {
            "pmodel_exact": "(dτ/dt)_P = exp(-x)",
            "gr_exact": "(dτ/dt)_GR = sqrt(1-2x)",
            "pmodel_second_order": "(dτ/dt)_P ≈ 1 - x + x^2/2",
            "gr_second_order": "(dτ/dt)_GR ≈ 1 - x - x^2/2",
            "redshift": "1 + z = (dτ/dt)_high / (dτ/dt)_low",
            "differential": "delta(Δf/f) = z_P - z_GR",
            "required_precision": "sigma_abs,1sigma <= |delta(Δf/f)| / 3",
        },
        "baseline_clock_case": {
            "delta_u_geodetic_m2_s2": float(clock_metrics["source"]["abstract_values"]["delta_u_geodetic_m2_s2"]),
            "delta_u_clock_m2_s2": float(clock_metrics["source"]["abstract_values"]["delta_u_clock_m2_s2"]),
            "height_scale_m": height_scale_m,
            "current_sigma_abs_delta_f_over_f": current_sigma_abs,
        },
        "case_rows": case_rows,
        "precision_reference_rows": precision_rows,
        "summary": {
            "main_case_exact_delta_z": float(main_case["delta_z_exact"]),
            "main_case_second_order_delta_z": float(main_case["delta_z_second_order"]),
            "main_case_exact_vs_second_order_abs_gap": second_order_gap,
            "main_case_required_abs_precision_exact_1sigma_for_3sigma": float(main_case["required_abs_precision_exact_1sigma_for_3sigma"]),
            "main_case_required_abs_precision_second_order_1sigma_for_3sigma": float(
                main_case["required_abs_precision_second_order_1sigma_for_3sigma"]
            ),
            "current_margin_over_required_exact": float(precision_rows[0]["current_over_required"]),
            "roadmap_target_1e_19_over_required_exact": float(precision_rows[1]["current_over_required"]),
            "roadmap_target_1e_20_over_required_exact": float(precision_rows[2]["current_over_required"]),
            "roadmap_target_1e_21_over_required_exact": float(precision_rows[3]["current_over_required"]),
            "most_accessible_case_id": str(most_accessible_case["case_id"]),
            "most_accessible_required_abs_precision_exact_1sigma_for_3sigma": float(
                most_accessible_case["required_abs_precision_exact_1sigma_for_3sigma"]
            ),
            "largest_exact_difference_case_id": str(largest_exact_case["case_id"]),
            "largest_exact_difference_abs_delta_z": abs(float(largest_exact_case["delta_z_exact"])),
        },
        "decision": {
            "overall_status": "optical_clock_differential_observable_fixed",
            "new_free_parameters_introduced": False,
            "main_case_detectable_under_current": bool(precision_rows[0]["detectable"]),
            "main_case_detectable_under_1e_19": bool(precision_rows[1]["detectable"]),
            "main_case_detectable_under_1e_20": bool(precision_rows[2]["detectable"]),
            "main_case_detectable_under_1e_21": bool(precision_rows[3]["detectable"]),
            "next_required_steps": ["8.7.55.1.4"],
        },
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the optical-clock differential observable between the P-model and GR stationary-clock maps."
    )
    parser.add_argument("--clock-json", default=str(CLOCK_JSON), help="Optical clock chronometric leveling metrics JSON.")
    parser.add_argument("--delta-json", default=str(DELTA_JSON), help="Unified gravity×quantum delta audit JSON.")
    parser.add_argument("--out-json", default=str(OUT_JSON), help="Output metrics JSON path.")
    parser.add_argument("--out-csv", default=str(OUT_CSV), help="Output CSV path.")
    args = parser.parse_args()

    clock_json_path = Path(args.clock_json)
    delta_json_path = Path(args.delta_json)
    out_json_path = Path(args.out_json)
    out_csv_path = Path(args.out_csv)

    # 条件分岐: `not clock_json_path.is_absolute()` を満たす経路を評価する。
    if not clock_json_path.is_absolute():
        clock_json_path = (ROOT / clock_json_path).resolve()

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

    clock_metrics = _read_json(clock_json_path)
    delta_metrics = _read_json(delta_json_path)
    payload = _build_payload(clock_metrics, delta_metrics)

    with out_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    _write_csv(out_csv_path, payload["case_rows"])


if __name__ == "__main__":
    main()
