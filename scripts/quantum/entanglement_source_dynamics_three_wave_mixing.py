#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entanglement_source_dynamics_three_wave_mixing.py

Step 8.7.50.2:
Freeze the source dynamics of Xi(P_mu; x_A, x_B) as an adopted-U(1)
three-wave-mixing kernel.

Inputs:
  - No observational dataset is required.
  - The script evaluates a normalized joint spectral amplitude (JSA) on a small
    grid of external source settings: pump bandwidth, phase-matching scale, and
    group-velocity walkoff ratio.

Outputs:
  - output/public/quantum/entanglement_source_dynamics_three_wave_mixing_metrics.json
  - output/public/quantum/entanglement_source_dynamics_three_wave_mixing_cases.csv

Assumptions:
  - U(1) is already adopted as an effective sector and is not derived from P.
  - The pump phase is a shared source phase chi_s and therefore global.
  - External source settings are not new P-model parameters; the only model-side
    shorthand kept here is the existing nonlinear derivative g_3w ~ V_*^(3).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402


# クラス: `SourceCase` の責務と境界条件を定義する。
@dataclass(frozen=True)
class SourceCase:
    case_id: str
    sigma_sum: float
    phase_match_scale: float
    walkoff_ratio: float
    shared_phase_rad: float


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。

def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


# 関数: `_detuning_grid` の入出力契約と処理意図を定義する。

def _detuning_grid(*, grid_size: int, grid_max_abs: float) -> np.ndarray:
    # 条件分岐: `grid_size < 9 or grid_size % 2 == 0` を満たす経路を評価する。
    if grid_size < 9 or grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd integer >= 9")

    return np.linspace(-grid_max_abs, grid_max_abs, grid_size, dtype=float)


# 関数: `_source_kernel` の入出力契約と処理意図を定義する。

def _source_kernel(*, case: SourceCase, detunings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nu_s, nu_i = np.meshgrid(detunings, detunings, indexing="ij")
    sum_coord = nu_s + nu_i
    mismatch = case.phase_match_scale * (nu_s - case.walkoff_ratio * nu_i)
    pump = np.exp(-0.5 * (sum_coord / case.sigma_sum) ** 2)
    phase_matching = np.sinc(mismatch / math.pi) * np.exp(1.0j * mismatch)
    jsa = pump * phase_matching * np.exp(1.0j * case.shared_phase_rad)
    return jsa, sum_coord, mismatch


# 関数: `_normalize_kernel` の入出力契約と処理意図を定義する。

def _normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(kernel))
    # 条件分岐: `not math.isfinite(norm) or norm <= 0.0` を満たす経路を評価する。
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("kernel norm must be finite and positive")

    return kernel / norm


# 関数: `_schmidt_metrics` の入出力契約と処理意図を定義する。

def _schmidt_metrics(kernel: np.ndarray) -> Dict[str, Any]:
    u, singular_values, vh = np.linalg.svd(kernel, full_matrices=False)
    weights = singular_values**2
    weight_sum = float(np.sum(weights))

    # 条件分岐: `weight_sum <= 0.0 or not math.isfinite(weight_sum)` を満たす経路を評価する。
    if weight_sum <= 0.0 or not math.isfinite(weight_sum):
        raise ValueError("invalid singular value spectrum")

    weights /= weight_sum
    schmidt_rank = float(1.0 / np.sum(weights**2))
    leading_rank1 = singular_values[0] * np.outer(u[:, 0], vh[0, :])
    separability_residual = float(np.linalg.norm(kernel - leading_rank1) / np.linalg.norm(kernel))
    return {
        "effective_schmidt_rank": schmidt_rank,
        "rank1_separability_residual": separability_residual,
        "leading_weights": [float(value) for value in weights[:5]],
    }


# 関数: `_phase_invariance_error` の入出力契約と処理意図を定義する。

def _phase_invariance_error(kernel: np.ndarray, *, phase_shift_rad: float) -> float:
    shifted = kernel * np.exp(1.0j * phase_shift_rad)
    return float(np.max(np.abs(np.abs(kernel) ** 2 - np.abs(shifted) ** 2)))


# 関数: `_peak_metrics` の入出力契約と処理意図を定義する。

def _peak_metrics(kernel: np.ndarray, *, sum_coord: np.ndarray, mismatch: np.ndarray) -> Dict[str, float]:
    peak_index = np.unravel_index(int(np.argmax(np.abs(kernel) ** 2)), kernel.shape)
    peak_sum = float(abs(sum_coord[peak_index]))
    peak_mismatch = float(abs(mismatch[peak_index]))
    return {
        "peak_abs_energy_sum_detuning": peak_sum,
        "peak_abs_phase_mismatch": peak_mismatch,
    }


# 関数: `_case_metrics` の入出力契約と処理意図を定義する。

def _case_metrics(case: SourceCase, *, detunings: np.ndarray) -> Dict[str, Any]:
    raw_kernel, sum_coord, mismatch = _source_kernel(case=case, detunings=detunings)
    normalized_kernel = _normalize_kernel(raw_kernel)
    schmidt = _schmidt_metrics(normalized_kernel)
    peak = _peak_metrics(normalized_kernel, sum_coord=sum_coord, mismatch=mismatch)
    phase_error = _phase_invariance_error(normalized_kernel, phase_shift_rad=1.2345)
    grid_step = float(detunings[1] - detunings[0])
    nonseparable = bool(float(schmidt["effective_schmidt_rank"]) > 1.05)
    phase_matched = bool(peak["peak_abs_phase_mismatch"] <= grid_step)
    energy_locked = bool(peak["peak_abs_energy_sum_detuning"] <= grid_step)
    return {
        "case": asdict(case),
        "grid_step": grid_step,
        "metrics": {
            **peak,
            **schmidt,
            "global_phase_probability_invariance_max_error": phase_error,
        },
        "decisions": {
            "energy_conservation_locked": energy_locked,
            "phase_matching_locked": phase_matched,
            "nonseparable_kernel": nonseparable,
            "case_pass": bool(energy_locked and phase_matched and nonseparable and phase_error <= 1.0e-14),
        },
    }


# 関数: `_default_cases` の入出力契約と処理意図を定義する。

def _default_cases() -> List[SourceCase]:
    return [
        SourceCase(
            case_id="narrow_pump_long_crystal",
            sigma_sum=0.35,
            phase_match_scale=2.20,
            walkoff_ratio=1.00,
            shared_phase_rad=0.0,
        ),
        SourceCase(
            case_id="balanced_source",
            sigma_sum=0.80,
            phase_match_scale=1.20,
            walkoff_ratio=0.92,
            shared_phase_rad=0.6,
        ),
        SourceCase(
            case_id="broad_pump_short_crystal",
            sigma_sum=1.60,
            phase_match_scale=0.65,
            walkoff_ratio=0.85,
            shared_phase_rad=1.1,
        ),
    ]


# 関数: `build_payload` の入出力契約と処理意図を定義する。

def build_payload(*, grid_size: int, grid_max_abs: float, cases: Sequence[SourceCase]) -> Dict[str, Any]:
    detunings = _detuning_grid(grid_size=grid_size, grid_max_abs=grid_max_abs)
    case_rows = [_case_metrics(case, detunings=detunings) for case in cases]
    all_pass = all(bool((row.get("decisions") or {}).get("case_pass")) for row in case_rows)
    schmidt_ranks = [
        float(((row.get("metrics") or {}).get("effective_schmidt_rank")))
        for row in case_rows
        if isinstance((row.get("metrics") or {}).get("effective_schmidt_rank"), (int, float))
    ]
    separability_residuals = [
        float(((row.get("metrics") or {}).get("rank1_separability_residual")))
        for row in case_rows
        if isinstance((row.get("metrics") or {}).get("rank1_separability_residual"), (int, float))
    ]
    phase_errors = [
        float(((row.get("metrics") or {}).get("global_phase_probability_invariance_max_error")))
        for row in case_rows
        if isinstance((row.get("metrics") or {}).get("global_phase_probability_invariance_max_error"), (int, float))
    ]
    return {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 8, "step": "8.7.50.2", "name": "Entanglement source dynamics via three-wave mixing"},
        "intent": "Freeze Xi(P_mu; x_A, x_B) as a phase-matched three-wave-mixing source kernel on top of the adopted U(1) sector.",
        "assumptions": [
            "U(1) is adopted as an effective theory and is not derived from P-only local gauge redundancy.",
            "The pump phase chi_s is a global source phase and therefore cancels from probabilities.",
            "External source settings (bandwidth, crystal length/mismatch, walkoff) are laboratory configuration values, not new P-model parameters.",
        ],
        "formulas": {
            "three_wave_vertex": "L_3w = g_3w P_p P_s^* P_i^* exp[-i Delta_omega tau_s + i Delta_k x] + c.c.",
            "proper_time_phase": "chi_s = omega_p tau_s + arg(A_p0)",
            "joint_spectral_amplitude": "Xi(Omega_s, Omega_i) propto alpha_p(Omega_s + Omega_i) sinc[Delta_k(Omega_s, Omega_i) L / 2] exp[i Delta_k L / 2]",
            "nonseparability_test": "Xi(Omega_s, Omega_i) != xi_A(Omega_s) xi_B(Omega_i) unless the source is driven into a singular limit.",
        },
        "input_grid": {"grid_size": int(grid_size), "grid_max_abs": float(grid_max_abs)},
        "cases": case_rows,
        "summary": {
            "all_cases_pass": all_pass,
            "min_effective_schmidt_rank": min(schmidt_ranks) if schmidt_ranks else None,
            "max_rank1_separability_residual": max(separability_residuals) if separability_residuals else None,
            "max_global_phase_probability_invariance_error": max(phase_errors) if phase_errors else None,
        },
        "decision": {
            "b1_source_dynamics_status": "closed" if all_pass else "inconsistent",
            "new_pmodel_free_parameters_introduced": False,
            "entanglement_status": "source_dynamics_fixed_b2_b3_pending" if all_pass else "source_dynamics_not_fixed",
            "next_required_steps": ["8.7.50.3", "8.7.50.6"],
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
                "sigma_sum",
                "phase_match_scale",
                "walkoff_ratio",
                "shared_phase_rad",
                "peak_abs_energy_sum_detuning",
                "peak_abs_phase_mismatch",
                "effective_schmidt_rank",
                "rank1_separability_residual",
                "global_phase_probability_invariance_max_error",
                "case_pass",
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
                    "sigma_sum": case.get("sigma_sum"),
                    "phase_match_scale": case.get("phase_match_scale"),
                    "walkoff_ratio": case.get("walkoff_ratio"),
                    "shared_phase_rad": case.get("shared_phase_rad"),
                    "peak_abs_energy_sum_detuning": metrics.get("peak_abs_energy_sum_detuning"),
                    "peak_abs_phase_mismatch": metrics.get("peak_abs_phase_mismatch"),
                    "effective_schmidt_rank": metrics.get("effective_schmidt_rank"),
                    "rank1_separability_residual": metrics.get("rank1_separability_residual"),
                    "global_phase_probability_invariance_max_error": metrics.get("global_phase_probability_invariance_max_error"),
                    "case_pass": decisions.get("case_pass"),
                }
            )


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Freeze Xi(P_mu; x_A, x_B) as a three-wave-mixing source kernel.")
    ap.add_argument("--grid-size", type=int, default=81, help="Odd grid size for the detuning mesh.")
    ap.add_argument("--grid-max-abs", type=float, default=4.0, help="Absolute detuning range for the normalized grid.")
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "entanglement_source_dynamics_three_wave_mixing_metrics.json"),
        help="Output JSON path.",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "entanglement_source_dynamics_three_wave_mixing_cases.csv"),
        help="Output CSV path.",
    )
    args = ap.parse_args(argv)

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)

    # 条件分岐: `not out_json.is_absolute()` を満たす経路を評価する。
    if not out_json.is_absolute():
        out_json = (ROOT / out_json).resolve()

    # 条件分岐: `not out_csv.is_absolute()` を満たす経路を評価する。

    if not out_csv.is_absolute():
        out_csv = (ROOT / out_csv).resolve()

    payload = build_payload(grid_size=args.grid_size, grid_max_abs=args.grid_max_abs, cases=_default_cases())
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_cases_csv(out_csv, payload)

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_entanglement_source_dynamics_three_wave_mixing",
                "phase": "8.7.50.2",
                "outputs": {
                    "entanglement_source_dynamics_three_wave_mixing_metrics_json": _rel(out_json),
                    "entanglement_source_dynamics_three_wave_mixing_cases_csv": _rel(out_csv),
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
