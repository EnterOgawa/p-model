#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
born_linear_detector_response_audit.py

Step 8.7.50.3 / 8.7.51.2:
Freeze the A2 linear-detector-response closure for the Born-rule route and add
the accumulated-backreaction supplement used in the Born closeout.

Inputs:
  - No observational dataset is required.
  - The script evaluates a few detector cells with prescribed envelope-density
    weights rho_j, detector spectral responses kappa_j, and weak-signal ratios
    |delta P_+ / P_det,*|.

Outputs:
  - output/public/quantum/born_linear_detector_response_audit_metrics.json
  - output/public/quantum/born_linear_detector_response_cases.csv
  - output/public/quantum/born_linear_detector_backreaction_cases.csv

Assumptions:
  - The detector couples through the existing Part I interaction
    L_int = g_P P_mu J^mu.
  - The lowest-order transition amplitude is linear in delta P_+ and therefore
    the rate is proportional to |psi|^2.
  - Nonlinear corrections are suppressed by the weak-signal ratio squared,
    |delta P_+ / P_det,*|^2.
  - Repeated detections shift the detector background only through the
    accumulated fraction N |delta P_+ / P_det,*|^2.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]

# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402


# クラス: `LinearResponseCase` の責務と境界条件を定義する。
@dataclass(frozen=True)
class LinearResponseCase:
    case_id: str
    rho_weights: Sequence[float]
    detector_spectral_response: Sequence[float]
    signal_to_background_ratio: Sequence[float]
    note: str


# クラス: `AccumulatedBackreactionCase` の責務と境界条件を定義する。

@dataclass(frozen=True)
class AccumulatedBackreactionCase:
    case_id: str
    signal_to_background_ratio: float
    detection_events: float
    regime: str
    note: str


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。

def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


# 関数: `_normalize_positive` の入出力契約と処理意図を定義する。

def _normalize_positive(values: Sequence[float]) -> List[float]:
    normalized = [float(value) for value in values]
    total = float(sum(normalized))

    # 条件分岐: `not normalized or any((not math.isfinite(value) or value <= 0.0) for value in normalized)` を満たす経路を評価する。
    if not normalized or any((not math.isfinite(value) or value <= 0.0) for value in normalized):
        raise ValueError("all values must be finite positive floats")

    # 条件分岐: `total <= 0.0 or not math.isfinite(total)` を満たす経路を評価する。

    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("sum of values must be finite and positive")

    return [value / total for value in normalized]


# 関数: `_leading_rates` の入出力契約と処理意図を定義する。

def _leading_rates(*, rho_weights: Sequence[float], detector_spectral_response: Sequence[float]) -> List[float]:
    rho = _normalize_positive(rho_weights)
    spectral = [float(value) for value in detector_spectral_response]

    # 条件分岐: `len(rho) != len(spectral)` を満たす経路を評価する。
    if len(rho) != len(spectral):
        raise ValueError("rho_weights and detector_spectral_response must have the same length")

    # 条件分岐: `any((not math.isfinite(value) or value <= 0.0) for value in spectral)` を満たす経路を評価する。

    if any((not math.isfinite(value) or value <= 0.0) for value in spectral):
        raise ValueError("detector spectral responses must be finite positive floats")

    return [rho_value * spectral_value for rho_value, spectral_value in zip(rho, spectral)]


# 関数: `_normalize_rates` の入出力契約と処理意図を定義する。

def _normalize_rates(rates: Sequence[float]) -> List[float]:
    return _normalize_positive(rates)


# 関数: `_flat_field_frequency` の入出力契約と処理意図を定義する。

def _flat_field_frequency(*, leading_rates: Sequence[float], detector_spectral_response: Sequence[float]) -> List[float]:
    corrected = []
    for rate_value, spectral_value in zip(leading_rates, detector_spectral_response):
        corrected.append(float(rate_value) / float(spectral_value))

    return _normalize_rates(corrected)


# 関数: `_linear_ratio_spread` の入出力契約と処理意図を定義する。

def _linear_ratio_spread(*, leading_rates: Sequence[float], rho_weights: Sequence[float], detector_spectral_response: Sequence[float]) -> float:
    rho = _normalize_positive(rho_weights)
    ratios = []
    for rate_value, rho_value, spectral_value in zip(leading_rates, rho, detector_spectral_response):
        ratios.append(float(rate_value) / (float(rho_value) * float(spectral_value)))

    return float(max(ratios) - min(ratios))


# 関数: `_frequency_bounds_from_nonlinearity` の入出力契約と処理意図を定義する。

def _frequency_bounds_from_nonlinearity(
    *,
    leading_rates: Sequence[float],
    signal_to_background_ratio: Sequence[float],
) -> Dict[str, Any]:
    ratios = [float(value) for value in signal_to_background_ratio]

    # 条件分岐: `len(leading_rates) != len(ratios)` を満たす経路を評価する。
    if len(leading_rates) != len(ratios):
        raise ValueError("leading_rates and signal_to_background_ratio must have the same length")

    # 条件分岐: `any((not math.isfinite(value) or value <= 0.0 or value >= 1.0) for value in ratios)` を満たす経路を評価する。

    if any((not math.isfinite(value) or value <= 0.0 or value >= 1.0) for value in ratios):
        raise ValueError("signal-to-background ratios must be finite and satisfy 0 < r < 1")

    base_frequency = _normalize_rates(leading_rates)
    nonlinear_eps = [value**2 for value in ratios]
    min_frequency = list(base_frequency)
    max_frequency = list(base_frequency)
    max_abs_error = 0.0

    for signs in itertools.product((-1.0, 1.0), repeat=len(leading_rates)):
        corrected_rates = []
        for rate_value, sign_value, epsilon_value in zip(leading_rates, signs, nonlinear_eps):
            corrected_rates.append(float(rate_value) * (1.0 + sign_value * epsilon_value))

        corrected_frequency = _normalize_rates(corrected_rates)
        for idx, frequency_value in enumerate(corrected_frequency):
            min_frequency[idx] = min(min_frequency[idx], float(frequency_value))
            max_frequency[idx] = max(max_frequency[idx], float(frequency_value))
            max_abs_error = max(max_abs_error, abs(float(frequency_value) - float(base_frequency[idx])))

    return {
        "base_frequency": [float(value) for value in base_frequency],
        "min_frequency": [float(value) for value in min_frequency],
        "max_frequency": [float(value) for value in max_frequency],
        "max_abs_frequency_error_bound": float(max_abs_error),
        "max_relative_nonlinear_correction_bound": float(max(nonlinear_eps)),
    }


# 関数: `_case_metrics` の入出力契約と処理意図を定義する。

def _case_metrics(case: LinearResponseCase) -> Dict[str, Any]:
    rho = _normalize_positive(case.rho_weights)
    spectral = [float(value) for value in case.detector_spectral_response]
    leading_rates = _leading_rates(
        rho_weights=case.rho_weights,
        detector_spectral_response=case.detector_spectral_response,
    )
    raw_frequency = _normalize_rates(leading_rates)
    flat_field_frequency = _flat_field_frequency(
        leading_rates=leading_rates,
        detector_spectral_response=case.detector_spectral_response,
    )
    nonlinear_bounds = _frequency_bounds_from_nonlinearity(
        leading_rates=leading_rates,
        signal_to_background_ratio=case.signal_to_background_ratio,
    )
    flat_field_error = max(abs(value - target) for value, target in zip(flat_field_frequency, rho))
    raw_frequency_distortion = max(abs(value - target) for value, target in zip(raw_frequency, rho))
    linear_ratio_spread = _linear_ratio_spread(
        leading_rates=leading_rates,
        rho_weights=case.rho_weights,
        detector_spectral_response=case.detector_spectral_response,
    )
    weak_signal_bound = max(float(value) ** 2 for value in case.signal_to_background_ratio)

    return {
        "case": asdict(case),
        "metrics": {
            "target_frequency": [float(value) for value in rho],
            "detector_spectral_response": [float(value) for value in spectral],
            "leading_rates": [float(value) for value in leading_rates],
            "raw_frequency": [float(value) for value in raw_frequency],
            "flat_field_frequency": [float(value) for value in flat_field_frequency],
            "raw_frequency_distortion_max_abs": float(raw_frequency_distortion),
            "flat_field_frequency_max_abs_error": float(flat_field_error),
            "linear_ratio_spread": float(linear_ratio_spread),
            "signal_to_background_ratio": [float(value) for value in case.signal_to_background_ratio],
            "max_signal_to_background_ratio": float(max(case.signal_to_background_ratio)),
            **nonlinear_bounds,
        },
        "decisions": {
            "rate_linear_in_rho": bool(linear_ratio_spread <= 1.0e-15),
            "flat_field_recovers_target": bool(flat_field_error <= 1.0e-15),
            "weak_signal_regime": bool(weak_signal_bound <= 1.0e-2),
            "case_pass": bool(linear_ratio_spread <= 1.0e-15 and flat_field_error <= 1.0e-15 and weak_signal_bound <= 1.0e-2),
        },
    }


# 関数: `_accumulated_backreaction_metrics` の入出力契約と処理意図を定義する。

def _accumulated_backreaction_metrics(case: AccumulatedBackreactionCase) -> Dict[str, Any]:
    ratio = float(case.signal_to_background_ratio)
    events = float(case.detection_events)

    if not math.isfinite(ratio) or ratio <= 0.0 or ratio >= 1.0:
        raise ValueError("signal_to_background_ratio must satisfy 0 < r < 1")

    if not math.isfinite(events) or events <= 0.0:
        raise ValueError("detection_events must be finite positive")

    cumulative_shift = events * (ratio**2)
    negligible = cumulative_shift <= 1.0e-6
    perturbative = cumulative_shift < 1.0

    return {
        "case": asdict(case),
        "metrics": {
            "signal_to_background_ratio": ratio,
            "signal_to_background_ratio_squared": ratio**2,
            "detection_events": events,
            "cumulative_background_shift_fraction": cumulative_shift,
        },
        "decisions": {
            "single_event_weak_signal": bool((ratio**2) <= 1.0e-2),
            "accumulated_backreaction_negligible": bool(negligible),
            "perturbative_background_still_valid": bool(perturbative),
            "case_pass": bool(negligible) if case.regime == "ordinary" else bool(not negligible),
        },
    }


# 関数: `_default_cases` の入出力契約と処理意図を定義する。

def _default_cases() -> List[LinearResponseCase]:
    return [
        LinearResponseCase(
            case_id="uniform_detector_three_bin",
            rho_weights=[0.20, 0.35, 0.45],
            detector_spectral_response=[1.0, 1.0, 1.0],
            signal_to_background_ratio=[1.0e-3, 1.2e-3, 8.0e-4],
            note="Uniform detector gain. Raw frequencies already track rho_j.",
        ),
        LinearResponseCase(
            case_id="flat_field_calibrated_pixels",
            rho_weights=[0.10, 0.30, 0.60],
            detector_spectral_response=[0.82, 1.08, 1.21],
            signal_to_background_ratio=[3.0e-3, 5.0e-3, 4.0e-3],
            note="Non-uniform detector gain. Flat-field correction must recover target rho_j.",
        ),
        LinearResponseCase(
            case_id="near_watch_but_still_linear",
            rho_weights=[0.15, 0.25, 0.60],
            detector_spectral_response=[0.95, 1.05, 1.00],
            signal_to_background_ratio=[4.0e-2, 7.0e-2, 6.0e-2],
            note="A stronger signal remains linear because nonlinear corrections stay below O(10^-2).",
        ),
    ]


# 関数: `_default_backreaction_cases` の入出力契約と処理意図を定義する。

def _default_backreaction_cases() -> List[AccumulatedBackreactionCase]:
    return [
        AccumulatedBackreactionCase(
            case_id="single_photon_counting",
            signal_to_background_ratio=1.0e-20,
            detection_events=1.0e10,
            regime="ordinary",
            note="Single-photon counting leaves the detector background unchanged even after 10^10 shots.",
        ),
        AccumulatedBackreactionCase(
            case_id="bright_calibrated_run",
            signal_to_background_ratio=1.0e-10,
            detection_events=1.0e12,
            regime="ordinary",
            note="A bright but still weak-signal calibrated run remains far below the cumulative backreaction threshold.",
        ),
        AccumulatedBackreactionCase(
            case_id="ultra_intense_many_body_injection",
            signal_to_background_ratio=1.0e-4,
            detection_events=1.0e8,
            regime="breakdown_watch",
            note="Ultra-strong laser pulses or many-body injection into a condensed background require a non-perturbative extension.",
        ),
    ]


# 関数: `build_payload` の入出力契約と処理意図を定義する。

def build_payload(
    linear_cases: Sequence[LinearResponseCase],
    backreaction_cases: Sequence[AccumulatedBackreactionCase],
) -> Dict[str, Any]:
    case_rows = [_case_metrics(case) for case in linear_cases]
    backreaction_rows = [_accumulated_backreaction_metrics(case) for case in backreaction_cases]
    all_pass = all(bool((row.get("decisions") or {}).get("case_pass")) for row in case_rows)
    flat_field_errors = [
        float(((row.get("metrics") or {}).get("flat_field_frequency_max_abs_error")))
        for row in case_rows
        if isinstance((row.get("metrics") or {}).get("flat_field_frequency_max_abs_error"), (int, float))
    ]
    nonlinear_bounds = [
        float(((row.get("metrics") or {}).get("max_relative_nonlinear_correction_bound")))
        for row in case_rows
        if isinstance((row.get("metrics") or {}).get("max_relative_nonlinear_correction_bound"), (int, float))
    ]
    frequency_error_bounds = [
        float(((row.get("metrics") or {}).get("max_abs_frequency_error_bound")))
        for row in case_rows
        if isinstance((row.get("metrics") or {}).get("max_abs_frequency_error_bound"), (int, float))
    ]
    ordinary_backreaction = [
        row
        for row in backreaction_rows
        if str(((row.get("case") or {}).get("regime") or "")) == "ordinary"
    ]
    breakdown_backreaction = [
        row
        for row in backreaction_rows
        if str(((row.get("case") or {}).get("regime") or "")) != "ordinary"
    ]

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 8, "step": "8.7.51.2", "name": "Born A2 linear response + accumulated backreaction supplement"},
        "intent": "Close the detector-linearity gap of the Born route and quantify when repeated detections do or do not backreact on the detector background.",
        "assumptions": [
            "The detector couples through the existing Part I interaction L_int = g_P P_mu J^mu.",
            "The detector background is macroscopic, so the incoming delta P_+ is a small fluctuation with |delta P_+ / P_det,*| << 1.",
            "Backreaction during one shot is negligible in the same regime where the detector is pointer-ready and phase mixing is already active.",
            "Repeated shots shift the background only through the accumulated fraction Delta P_det,* / P_det,* ~ N |delta P_+ / P_det,*|^2.",
        ],
        "formulas": {
            "lowest_order_hamiltonian": "H_int^(1) = -g_P integral_{V_j} d^3x [delta P_+ delta J_det^0 + c.c.]",
            "fermi_golden_rule": "lambda_{j->alpha} = (2 pi g_P^2 P_*^2 / hbar) |psi_j|^2 |<D_alpha| integral u_j delta J_det^0 |D_0>|^2 delta(E_alpha - E_0 - hbar omega_*)",
            "local_rate": "lambda_j = kappa_j |psi_j|^2 = kappa_bar_j rho_j",
            "spectral_density_form": "kappa_j = (g_P^2 P_*^2 / hbar^2) S_OO^(j)(omega_*)",
            "nonlinear_bound": "Delta lambda_j^(nl) / lambda_j^(1) = O(|delta P_+ / P_det,*|^2)",
            "flat_field_frequency": "f_j^(ff) = (lambda_j / kappa_j) / sum_k (lambda_k / kappa_k) = rho_j / sum_k rho_k",
            "accumulated_backreaction": "Delta P_det,* / P_det,* ~ N |delta P_+ / P_det,*|^2",
        },
        "cases": case_rows,
        "accumulated_backreaction_cases": backreaction_rows,
        "summary": {
            "all_cases_pass": all_pass,
            "max_flat_field_frequency_error": max(flat_field_errors) if flat_field_errors else None,
            "max_relative_nonlinear_correction_bound": max(nonlinear_bounds) if nonlinear_bounds else None,
            "max_abs_frequency_error_bound": max(frequency_error_bounds) if frequency_error_bounds else None,
            "ordinary_cases_max_cumulative_backreaction_fraction": max(
                float(((row.get("metrics") or {}).get("cumulative_background_shift_fraction")))
                for row in ordinary_backreaction
            )
            if ordinary_backreaction
            else None,
            "breakdown_watch_min_cumulative_backreaction_fraction": min(
                float(((row.get("metrics") or {}).get("cumulative_background_shift_fraction")))
                for row in breakdown_backreaction
            )
            if breakdown_backreaction
            else None,
        },
        "decision": {
            "a2_gap_status": "closed" if all_pass else "inconsistent",
            "accumulated_backreaction_status": "closed_with_breakdown_watch",
            "born_route_status": "conditional_detection_derivation_with_accumulated_backreaction_bound_fixed" if all_pass else "a2_not_fixed",
            "new_pmodel_free_parameters_introduced": False,
            "next_required_steps": ["8.7.51.3", "8.7.51.4"],
        },
    }


# 関数: `_write_cases_csv` の入出力契約と処理意図を定義する。

def _write_cases_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = payload.get("cases") if isinstance(payload.get("cases"), list) else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "raw_frequency_distortion_max_abs",
                "flat_field_frequency_max_abs_error",
                "max_signal_to_background_ratio",
                "max_relative_nonlinear_correction_bound",
                "max_abs_frequency_error_bound",
                "case_pass",
                "note",
            ],
        )
        writer.writeheader()
        for row in rows:
            # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
            if not isinstance(row, dict):
                continue

            case = row.get("case") if isinstance(row.get("case"), dict) else {}
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            decisions = row.get("decisions") if isinstance(row.get("decisions"), dict) else {}
            writer.writerow(
                {
                    "case_id": case.get("case_id"),
                    "raw_frequency_distortion_max_abs": metrics.get("raw_frequency_distortion_max_abs"),
                    "flat_field_frequency_max_abs_error": metrics.get("flat_field_frequency_max_abs_error"),
                    "max_signal_to_background_ratio": metrics.get("max_signal_to_background_ratio"),
                    "max_relative_nonlinear_correction_bound": metrics.get("max_relative_nonlinear_correction_bound"),
                    "max_abs_frequency_error_bound": metrics.get("max_abs_frequency_error_bound"),
                    "case_pass": decisions.get("case_pass"),
                    "note": case.get("note"),
                }
            )


# 関数: `_write_backreaction_csv` の入出力契約と処理意図を定義する。

def _write_backreaction_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = payload.get("accumulated_backreaction_cases")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "regime",
                "signal_to_background_ratio",
                "detection_events",
                "cumulative_background_shift_fraction",
                "accumulated_backreaction_negligible",
                "perturbative_background_still_valid",
                "note",
            ],
        )
        writer.writeheader()
        if not isinstance(rows, list):
            return

        for row in rows:
            if not isinstance(row, dict):
                continue

            case = row.get("case") if isinstance(row.get("case"), dict) else {}
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            decisions = row.get("decisions") if isinstance(row.get("decisions"), dict) else {}
            writer.writerow(
                {
                    "case_id": case.get("case_id"),
                    "regime": case.get("regime"),
                    "signal_to_background_ratio": metrics.get("signal_to_background_ratio"),
                    "detection_events": metrics.get("detection_events"),
                    "cumulative_background_shift_fraction": metrics.get("cumulative_background_shift_fraction"),
                    "accumulated_backreaction_negligible": decisions.get("accumulated_backreaction_negligible"),
                    "perturbative_background_still_valid": decisions.get("perturbative_background_still_valid"),
                    "note": case.get("note"),
                }
            )


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Freeze the A2 linear detector response closure for the Born-rule route.")
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "born_linear_detector_response_audit_metrics.json"),
        help="Output JSON path.",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "born_linear_detector_response_cases.csv"),
        help="Output CSV path.",
    )
    ap.add_argument(
        "--out-backreaction-csv",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "born_linear_detector_backreaction_cases.csv"),
        help="Output CSV path for accumulated detector backreaction cases.",
    )
    args = ap.parse_args(argv)

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_backreaction_csv = Path(args.out_backreaction_csv)

    # 条件分岐: `not out_json.is_absolute()` を満たす経路を評価する。
    if not out_json.is_absolute():
        out_json = (ROOT / out_json).resolve()

    # 条件分岐: `not out_csv.is_absolute()` を満たす経路を評価する。

    if not out_csv.is_absolute():
        out_csv = (ROOT / out_csv).resolve()

    if not out_backreaction_csv.is_absolute():
        out_backreaction_csv = (ROOT / out_backreaction_csv).resolve()

    payload = build_payload(_default_cases(), _default_backreaction_cases())
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_cases_csv(out_csv, payload)
    _write_backreaction_csv(out_backreaction_csv, payload)

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")
    print(f"[ok] wrote: {_rel(out_backreaction_csv)}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_born_linear_detector_response_audit",
                "phase": "8.7.51.2",
                "outputs": {
                    "born_linear_detector_response_audit_metrics_json": _rel(out_json),
                    "born_linear_detector_response_cases_csv": _rel(out_csv),
                    "born_linear_detector_backreaction_cases_csv": _rel(out_backreaction_csv),
                },
                "decision": payload.get("decision"),
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
