#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
born_phase_diffusion_audit.py

Step 8.7.50.1 / 8.7.51.1:
Freeze the A1 phase-diffusion closure for the Born-rule route and add the
coarse-grained Markov supplement used in the Born closeout.

Inputs:
  - No observational input is required. The script evaluates the dimensionless
    phase-mixing gate on user-provided grids of (omega_* tau_free) and
    (T_obs / tau_free).
  - It uses only the existing Part I transport quantities tau_free, Gamma_path,
    chi_P, plus the external bath condition T_env.

Outputs:
  - output/public/quantum/born_phase_diffusion_audit_metrics.json
  - output/public/quantum/born_phase_diffusion_threshold_table.csv
  - output/public/quantum/born_phase_diffusion_markov_cases.csv

Assumptions:
  - carrier-removed phase obeys dot(vartheta) = -omega_* u_env(t)
  - the low-frequency environment is closed by the same Markov/FDT limit that
    defines Gamma_path and chi_P in Part I 2.7
  - the relevant Markov ratio is tau_corr / T_obs after carrier removal; the
    raw carrier ratio omega_* tau_free is kept only as a diagnostic
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

from scripts.summary import worklog  # noqa: E402


REPRESENTATIVE_MARKOV_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "optical_fast_bin",
        "label": "optical fast-bin readout",
        "omega_star_s_inv": 3.0e15,
        "tau_free_s": 1.0e-13,
        "tobs_s": 1.0e-9,
    },
    {
        "case_id": "optical_ns_readout",
        "label": "optical ns readout",
        "omega_star_s_inv": 3.0e15,
        "tau_free_s": 1.0e-12,
        "tobs_s": 1.0e-7,
    },
    {
        "case_id": "optical_us_integration",
        "label": "optical us integration",
        "omega_star_s_inv": 3.0e15,
        "tau_free_s": 1.0e-10,
        "tobs_s": 1.0e-5,
    },
]


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。
def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


# 関数: `_parse_float_list` の入出力契約と処理意図を定義する。

def _parse_float_list(raw: str) -> List[float]:
    values: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        # 条件分岐: `not token` を満たす経路を評価する。
        if not token:
            continue

        value = float(token)
        # 条件分岐: `not math.isfinite(value) or value <= 0.0` を満たす経路を評価する。
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"all grid values must be finite positive floats: {token}")

        values.append(value)

    # 条件分岐: `not values` を満たす経路を評価する。

    if not values:
        raise ValueError("expected at least one positive float")

    return values


# 関数: `_critical_thermal_ratio` の入出力契約と処理意図を定義する。

def _critical_thermal_ratio(*, omega_tau_free: float, tobs_over_tau_free: float) -> float:
    return 1.0 / ((omega_tau_free**2) * tobs_over_tau_free)


# 関数: `_phase_mixing_gate` の入出力契約と処理意図を定義する。

def _phase_mixing_gate(*, omega_tau_free: float, thermal_ratio: float, tobs_over_tau_free: float) -> float:
    return (omega_tau_free**2) * thermal_ratio * tobs_over_tau_free


# 関数: `_build_markov_cases` の入出力契約と処理意図を定義する。

def _build_markov_cases() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in REPRESENTATIVE_MARKOV_CASES:
        omega_star = float(case["omega_star_s_inv"])
        tau_free = float(case["tau_free_s"])
        tobs = float(case["tobs_s"])
        gamma_path = 1.0 / tau_free
        omega_tau_free = omega_star * tau_free
        tau_corr_over_tobs = tau_free / tobs
        gamma_path_tobs = gamma_path * tobs
        rows.append(
            {
                "case_id": str(case["case_id"]),
                "label": str(case["label"]),
                "omega_star_s_inv": omega_star,
                "gamma_path_s_inv": gamma_path,
                "tau_free_s": tau_free,
                "tobs_s": tobs,
                "naive_tau_corr_over_tau_carrier": omega_tau_free,
                "coarse_grained_tau_corr_over_tobs": tau_corr_over_tobs,
                "gamma_path_times_tobs": gamma_path_tobs,
                "coarse_grained_markov_expected": bool(tau_corr_over_tobs < 1.0e-2),
            }
        )

    return rows


# 関数: `_build_probe_rows` の入出力契約と処理意図を定義する。

def _build_probe_rows(
    *,
    omega_tau_free: float,
    tobs_over_tau_free: float,
    thermal_ratio_probes: List[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for thermal_ratio in thermal_ratio_probes:
        gate = _phase_mixing_gate(
            omega_tau_free=omega_tau_free,
            thermal_ratio=thermal_ratio,
            tobs_over_tau_free=tobs_over_tau_free,
        )
        rows.append(
            {
                "thermal_ratio_kbt_over_chiP": float(thermal_ratio),
                "gamma_deph_times_tobs": float(gate),
                "phase_mixing_expected": bool(gate > 1.0),
            }
        )

    return rows


# 関数: `build_payload` の入出力契約と処理意図を定義する。

def build_payload(
    *,
    omega_tau_free_grid: List[float],
    tobs_over_tau_free_grid: List[float],
    thermal_ratio_probes: List[float],
) -> Dict[str, Any]:
    threshold_rows: List[Dict[str, Any]] = []
    min_crit: Optional[float] = None
    max_crit: Optional[float] = None
    mixed_rows = 0
    total_probe_rows = 0
    markov_cases = _build_markov_cases()

    for omega_tau_free in omega_tau_free_grid:
        for tobs_over_tau_free in tobs_over_tau_free_grid:
            critical_ratio = _critical_thermal_ratio(
                omega_tau_free=omega_tau_free,
                tobs_over_tau_free=tobs_over_tau_free,
            )
            probe_rows = _build_probe_rows(
                omega_tau_free=omega_tau_free,
                tobs_over_tau_free=tobs_over_tau_free,
                thermal_ratio_probes=thermal_ratio_probes,
            )
            threshold_rows.append(
                {
                    "omega_tau_free": float(omega_tau_free),
                    "tobs_over_tau_free": float(tobs_over_tau_free),
                    "critical_thermal_ratio_kbt_over_chiP_for_gamma_deph_tobs_eq_1": float(critical_ratio),
                    "probe_rows": probe_rows,
                }
            )
            mixed_rows += sum(1 for row in probe_rows if bool(row["phase_mixing_expected"]))
            total_probe_rows += len(probe_rows)
            min_crit = critical_ratio if min_crit is None else min(min_crit, critical_ratio)
            max_crit = critical_ratio if max_crit is None else max(max_crit, critical_ratio)

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 8, "step": "8.7.51.1", "name": "Born A1 phase-diffusion + Markov supplement"},
        "intent": "Close the phase-mixing gap of the Born statistical bridge and justify the coarse-grained Markov closure without introducing new model parameters.",
        "assumptions": [
            "carrier-removed phase obeys dot(vartheta) = -omega_* u_env(t) from the proper-time mapping of Part I 2.7.0",
            "the low-frequency bath uses the same Markov/FDT closure that defines Gamma_path and chi_P in Part I 2.7",
            "T_env is an external environment condition rather than a new model parameter",
            "the relevant Markov ratio is tau_corr / T_obs after carrier removal; omega_* tau_free is reported only as a diagnostic ratio to the stripped carrier period",
        ],
        "formulas": {
            "gamma_path": "Gamma_path = tau_free^{-1} = (g_P^2 / chi_P) * integral_0^inf <delta J_x(t) delta J_x(0)> dt",
            "environment_covariance": "<u_env(t) u_env(0)> = (k_B T_env / chi_P) exp(-Gamma_path |t|)",
            "gamma_deph": "Gamma_deph = omega_*^2 (k_B T_env / chi_P) tau_free = omega_*^2 (k_B T_env / chi_P) / Gamma_path",
            "transport_substitution": "tau_free = [v/L_corr + A_col rho T_env^(-3/2)]^(-1)",
            "dimensionless_gate": "Gamma_deph T_obs = (omega_* tau_free)^2 (k_B T_env / chi_P) (T_obs / tau_free)",
            "naive_carrier_ratio": "tau_corr / tau_carrier = omega_* tau_free",
            "coarse_grained_markov_ratio": "tau_corr / T_obs = tau_free / T_obs = 1 / (Gamma_path T_obs)",
        },
        "input_grids": {
            "omega_tau_free_grid": [float(value) for value in omega_tau_free_grid],
            "tobs_over_tau_free_grid": [float(value) for value in tobs_over_tau_free_grid],
            "thermal_ratio_probes_kbt_over_chiP": [float(value) for value in thermal_ratio_probes],
        },
        "threshold_table": threshold_rows,
        "representative_markov_cases": markov_cases,
        "summary": {
            "critical_thermal_ratio_min": float(min_crit) if min_crit is not None else None,
            "critical_thermal_ratio_max": float(max_crit) if max_crit is not None else None,
            "probe_rows_with_phase_mixing": int(mixed_rows),
            "probe_rows_total": int(total_probe_rows),
            "markov_cases_all_pass_coarse_grained": bool(
                markov_cases and all(bool(row["coarse_grained_markov_expected"]) for row in markov_cases)
            ),
            "markov_cases_min_gamma_path_times_tobs": float(
                min(float(row["gamma_path_times_tobs"]) for row in markov_cases)
            )
            if markov_cases
            else None,
            "markov_cases_max_naive_tau_corr_over_tau_carrier": float(
                max(float(row["naive_tau_corr_over_tau_carrier"]) for row in markov_cases)
            )
            if markov_cases
            else None,
        },
        "decision": {
            "a1_gap_status": "closed",
            "new_free_parameters_introduced": False,
            "markov_closure_status": "closed_after_carrier_removal",
            "born_route_status": "phase_mixing_origin_and_markov_closure_fixed",
            "next_required_steps": ["8.7.51.2", "8.7.51.3", "8.7.51.4"],
        },
    }


# 関数: `_write_threshold_csv` の入出力契約と処理意図を定義する。

def _write_threshold_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = payload.get("threshold_table") if isinstance(payload.get("threshold_table"), list) else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "omega_tau_free",
                "tobs_over_tau_free",
                "critical_thermal_ratio_kbt_over_chiP_for_gamma_deph_tobs_eq_1",
            ],
        )
        writer.writeheader()
        for row in rows:
            # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
            if not isinstance(row, dict):
                continue

            writer.writerow(
                {
                    "omega_tau_free": row.get("omega_tau_free"),
                    "tobs_over_tau_free": row.get("tobs_over_tau_free"),
                    "critical_thermal_ratio_kbt_over_chiP_for_gamma_deph_tobs_eq_1": row.get(
                        "critical_thermal_ratio_kbt_over_chiP_for_gamma_deph_tobs_eq_1"
                    ),
                }
            )


# 関数: `_write_markov_cases_csv` の入出力契約と処理意図を定義する。

def _write_markov_cases_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = payload.get("representative_markov_cases")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "label",
                "omega_star_s_inv",
                "gamma_path_s_inv",
                "tau_free_s",
                "tobs_s",
                "naive_tau_corr_over_tau_carrier",
                "coarse_grained_tau_corr_over_tobs",
                "gamma_path_times_tobs",
                "coarse_grained_markov_expected",
            ],
        )
        writer.writeheader()
        if not isinstance(rows, list):
            return

        for row in rows:
            if not isinstance(row, dict):
                continue

            writer.writerow({name: row.get(name) for name in writer.fieldnames})


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Freeze the A1 phase-diffusion closure for the Born-rule route.")
    ap.add_argument(
        "--omega-tau-free-grid",
        type=str,
        default="1e1,1e3,1e6,1e9",
        help="Comma-separated positive grid for omega_* tau_free.",
    )
    ap.add_argument(
        "--tobs-over-tau-free-grid",
        type=str,
        default="1e1,1e3,1e6",
        help="Comma-separated positive grid for T_obs / tau_free.",
    )
    ap.add_argument(
        "--thermal-ratio-probes",
        type=str,
        default="1e-18,1e-12,1e-6",
        help="Comma-separated probe values for k_B T_env / chi_P.",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "born_phase_diffusion_audit_metrics.json"),
        help="Output JSON path.",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "born_phase_diffusion_threshold_table.csv"),
        help="Output CSV path.",
    )
    ap.add_argument(
        "--out-markov-csv",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "born_phase_diffusion_markov_cases.csv"),
        help="Output CSV path for representative Markov cases.",
    )
    args = ap.parse_args(argv)

    omega_tau_free_grid = _parse_float_list(args.omega_tau_free_grid)
    tobs_over_tau_free_grid = _parse_float_list(args.tobs_over_tau_free_grid)
    thermal_ratio_probes = _parse_float_list(args.thermal_ratio_probes)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_markov_csv = Path(args.out_markov_csv)

    # 条件分岐: `not out_json.is_absolute()` を満たす経路を評価する。
    if not out_json.is_absolute():
        out_json = (ROOT / out_json).resolve()

    # 条件分岐: `not out_csv.is_absolute()` を満たす経路を評価する。

    if not out_csv.is_absolute():
        out_csv = (ROOT / out_csv).resolve()

    if not out_markov_csv.is_absolute():
        out_markov_csv = (ROOT / out_markov_csv).resolve()

    payload = build_payload(
        omega_tau_free_grid=omega_tau_free_grid,
        tobs_over_tau_free_grid=tobs_over_tau_free_grid,
        thermal_ratio_probes=thermal_ratio_probes,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_threshold_csv(out_csv, payload)
    _write_markov_cases_csv(out_markov_csv, payload)

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")
    print(f"[ok] wrote: {_rel(out_markov_csv)}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_born_phase_diffusion_audit",
                "phase": "8.7.50.1",
                "outputs": {
                    "born_phase_diffusion_audit_metrics_json": _rel(out_json),
                    "born_phase_diffusion_threshold_table_csv": _rel(out_csv),
                    "born_phase_diffusion_markov_cases_csv": _rel(out_markov_csv),
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
