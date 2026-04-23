#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_measurement_pointer_basis_grounding_audit.py

Step 8.7.50.4:
Ground the measurement pointer basis as stable detector P-background modes.

Inputs:
  - output/public/quantum/quantum_measurement_dynamic_collapse_simulation_metrics.json

Outputs:
  - output/public/quantum/quantum_measurement_pointer_basis_grounding_audit_metrics.json
  - output/public/quantum/quantum_measurement_pointer_basis_grounding_channels.csv

Assumptions:
  - gamma_m in the reduced stochastic measurement equation is identified with
    the detector-side coarse-grained dephasing rate from A1.
  - Pointer channels are the low-frequency stable modes of the detector
    background P_det,* and obey an OU closure around the local free-energy
    minimum.
  - No new P-model parameters are introduced; the audit reuses the frozen
    simulation parameters Gamma_k, chi_k, eta_k, and sigma_xi,k.
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
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]

# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402


# クラス: `ChannelRow` の責務と境界条件を定義する。
@dataclass(frozen=True)
class ChannelRow:
    channel_index: int
    relax_s_inv: float
    gain_s_inv: float
    env_gain_s_inv: float
    noise_sqrt_s_inv: float
    fixed_point_plus: float
    env_offset_per_unit_env: float
    stationary_sigma: float
    separation_sigma_units: float
    static_sign_error_upper_bound: float


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。

def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_as_float_list` の入出力契約と処理意図を定義する。

def _as_float_list(payload: Dict[str, Any], key: str) -> List[float]:
    raw = payload.get(key)

    # 条件分岐: `not isinstance(raw, list)` を満たす経路を評価する。
    if not isinstance(raw, list):
        raise ValueError(f"expected list for {key}")

    values = [float(value) for value in raw]

    # 条件分岐: `any(not math.isfinite(value) for value in values)` を満たす経路を評価する。
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"all values in {key} must be finite")

    return values


# 関数: `_channel_rows` の入出力契約と処理意図を定義する。

def _channel_rows(*, params: Dict[str, Any]) -> List[ChannelRow]:
    relax = _as_float_list(params, "pointer_relax_s_inv")
    gain = _as_float_list(params, "pointer_gain_s_inv")
    env_gain = _as_float_list(params, "pointer_env_gain_s_inv")
    noise = _as_float_list(params, "pointer_noise_sqrt_s_inv")

    lengths = {len(relax), len(gain), len(env_gain), len(noise)}

    # 条件分岐: `len(lengths) != 1` を満たす経路を評価する。
    if len(lengths) != 1:
        raise ValueError("pointer parameter arrays must have the same length")

    rows: List[ChannelRow] = []
    for channel_index, (relax_value, gain_value, env_gain_value, noise_value) in enumerate(zip(relax, gain, env_gain, noise)):
        # 条件分岐: `relax_value <= 0.0 or noise_value <= 0.0` を満たす経路を評価する。
        if relax_value <= 0.0 or noise_value <= 0.0:
            raise ValueError("relaxation rates and noise scales must be positive")

        fixed_point_plus = gain_value / relax_value
        env_offset_per_unit_env = env_gain_value / relax_value
        stationary_sigma = noise_value / math.sqrt(2.0 * relax_value)
        separation = abs(fixed_point_plus) / stationary_sigma
        sign_error_upper_bound = 0.5 * math.erfc(separation / math.sqrt(2.0))
        rows.append(
            ChannelRow(
                channel_index=channel_index,
                relax_s_inv=relax_value,
                gain_s_inv=gain_value,
                env_gain_s_inv=env_gain_value,
                noise_sqrt_s_inv=noise_value,
                fixed_point_plus=fixed_point_plus,
                env_offset_per_unit_env=env_offset_per_unit_env,
                stationary_sigma=stationary_sigma,
                separation_sigma_units=separation,
                static_sign_error_upper_bound=sign_error_upper_bound,
            )
        )

    return rows


# 関数: `build_payload` の入出力契約と処理意図を定義する。

def build_payload(*, sim_metrics: Dict[str, Any], sim_metrics_path: Path) -> Dict[str, Any]:
    params = sim_metrics.get("parameters") if isinstance(sim_metrics.get("parameters"), dict) else {}
    summary = sim_metrics.get("summary") if isinstance(sim_metrics.get("summary"), dict) else {}
    channel_rows = _channel_rows(params=params)

    gamma_meas = float(params.get("gamma_meas_s_inv"))
    collapse_threshold_abs_z = float(params.get("collapse_threshold_abs_z"))
    tau50_s = float(summary.get("collapse_time_median_s"))
    pointer_consensus_fraction = float(summary.get("pointer_consensus_fraction"))
    branch_stable_fraction = float(summary.get("branch_stable_fraction"))
    pointer_channel_count = len(channel_rows)

    # 条件分岐: `pointer_channel_count <= 0` を満たす経路を評価する。
    if pointer_channel_count <= 0:
        raise ValueError("expected at least one pointer channel")

    gamma_pointer_feedback_mean = sum(row.gain_s_inv**2 / row.relax_s_inv for row in channel_rows) / float(pointer_channel_count)
    gamma_pointer_total = gamma_meas + gamma_pointer_feedback_mean
    coherence_initial_abs = 0.5
    coherence_threshold_abs = 0.5 * math.sqrt(max(0.0, 1.0 - collapse_threshold_abs_z**2))

    # 条件分岐: `coherence_threshold_abs <= 0.0 or gamma_pointer_total <= 0.0` を満たす経路を評価する。
    if coherence_threshold_abs <= 0.0 or gamma_pointer_total <= 0.0:
        raise ValueError("invalid threshold coherence or pointer rate")

    tau_d_s = math.log(coherence_initial_abs / coherence_threshold_abs) / (2.0 * gamma_pointer_total)
    tau50_over_tau_d = tau50_s / tau_d_s
    min_separation = min(row.separation_sigma_units for row in channel_rows)
    max_static_sign_error_upper_bound = max(row.static_sign_error_upper_bound for row in channel_rows)

    passes = {
        "pointer_modes_well_separated": bool(min_separation >= 5.0),
        "tau50_matches_tauD_order": bool(0.5 <= tau50_over_tau_d <= 2.0),
        "pointer_consensus_matches_fixed_modes": bool(pointer_consensus_fraction >= 0.95),
        "branch_stability_matches_fixed_modes": bool(branch_stable_fraction >= 0.90),
    }
    all_pass = all(passes.values())

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 8, "step": "8.7.50.4", "name": "Measurement pointer-basis grounding"},
        "inputs": {
            "dynamic_collapse_metrics_json": _rel(sim_metrics_path),
        },
        "intent": "Ground the pointer basis as stable detector P-background modes and connect gamma_m to the detector-side A1 dephasing rate.",
        "assumptions": [
            "gamma_m in the reduced stochastic measurement equation is identified with the detector-side coarse-grained dephasing rate Gamma_deph^(det).",
            "Pointer coordinates are stable low-frequency detector modes that follow an OU closure around the local free-energy minimum.",
            "The current audit reuses the frozen dynamic-collapse parameters and therefore introduces no new P-model free parameters.",
        ],
        "formulas": {
            "pointer_free_energy": "F_det^(m) = (1/2) sum_a Gamma_a q_a^2 - m sum_a chi_a q_a",
            "pointer_fixed_point": "q_a^(m) = m chi_a / Gamma_a",
            "stationary_width": "sigma_q^(inf) = sigma_xi / sqrt(2 Gamma)",
            "separation_index": "S_a = |q_a^(+)| / sigma_q^(inf)",
            "gamma_identification": "gamma_m = Gamma_deph^(det) = omega_*^2 (k_B T_det / chi_P,det) tau_free,det",
            "pointer_feedback": "Gamma_ptr = gamma_m + (1/N_ptr) sum_k chi_k^2 / Gamma_k",
            "decoherence_time": "tau_D = ln(|rho_01(0)| / |rho_01|_th) / (2 Gamma_ptr), |rho_01|_th = (1/2) sqrt(1 - z_th^2)",
        },
        "channels": [asdict(row) for row in channel_rows],
        "summary": {
            "gamma_meas_identified_as_gamma_deph_det_s_inv": float(gamma_meas),
            "gamma_pointer_feedback_mean_s_inv": float(gamma_pointer_feedback_mean),
            "gamma_pointer_total_s_inv": float(gamma_pointer_total),
            "coherence_initial_abs": float(coherence_initial_abs),
            "coherence_threshold_abs": float(coherence_threshold_abs),
            "tau_D_s": float(tau_d_s),
            "tau50_reference_s": float(tau50_s),
            "tau50_over_tauD": float(tau50_over_tau_d),
            "min_pointer_separation_sigma_units": float(min_separation),
            "max_static_sign_error_upper_bound": float(max_static_sign_error_upper_bound),
            "pointer_consensus_fraction_reference": float(pointer_consensus_fraction),
            "branch_stable_fraction_reference": float(branch_stable_fraction),
        },
        "decision": {
            "c1_pointer_basis_status": "closed" if all_pass else "inconsistent",
            "passes": passes,
            "new_pmodel_free_parameters_introduced": False,
            "measurement_status": "pointer_basis_grounded_c2_pending" if all_pass else "pointer_basis_not_grounded",
            "next_required_steps": ["8.7.50.5", "8.7.50.6"],
        },
    }


# 関数: `_write_channels_csv` の入出力契約と処理意図を定義する。

def _write_channels_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = payload.get("channels") if isinstance(payload.get("channels"), list) else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "channel_index",
                "relax_s_inv",
                "gain_s_inv",
                "env_gain_s_inv",
                "noise_sqrt_s_inv",
                "fixed_point_plus",
                "env_offset_per_unit_env",
                "stationary_sigma",
                "separation_sigma_units",
                "static_sign_error_upper_bound",
            ],
        )
        writer.writeheader()
        for row in rows:
            # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
            if not isinstance(row, dict):
                continue

            writer.writerow(row)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Ground the pointer basis as stable detector P-background modes.")
    ap.add_argument(
        "--sim-metrics",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_measurement_dynamic_collapse_simulation_metrics.json"),
        help="Input dynamic-collapse metrics JSON path.",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_measurement_pointer_basis_grounding_audit_metrics.json"),
        help="Output JSON path.",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_measurement_pointer_basis_grounding_channels.csv"),
        help="Output CSV path.",
    )
    args = ap.parse_args(argv)

    sim_metrics_path = Path(args.sim_metrics)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)

    # 条件分岐: `not sim_metrics_path.is_absolute()` を満たす経路を評価する。
    if not sim_metrics_path.is_absolute():
        sim_metrics_path = (ROOT / sim_metrics_path).resolve()

    # 条件分岐: `not out_json.is_absolute()` を満たす経路を評価する。

    if not out_json.is_absolute():
        out_json = (ROOT / out_json).resolve()

    # 条件分岐: `not out_csv.is_absolute()` を満たす経路を評価する。

    if not out_csv.is_absolute():
        out_csv = (ROOT / out_csv).resolve()

    sim_metrics = _read_json(sim_metrics_path)
    payload = build_payload(sim_metrics=sim_metrics, sim_metrics_path=sim_metrics_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_channels_csv(out_csv, payload)

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_measurement_pointer_basis_grounding_audit",
                "phase": "8.7.50.4",
                "inputs": {
                    "dynamic_collapse_metrics_json": _rel(sim_metrics_path),
                },
                "outputs": {
                    "quantum_measurement_pointer_basis_grounding_audit_metrics_json": _rel(out_json),
                    "quantum_measurement_pointer_basis_grounding_channels_csv": _rel(out_csv),
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
