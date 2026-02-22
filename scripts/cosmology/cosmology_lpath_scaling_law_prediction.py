#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_lpath_scaling_law_prediction.py

Step 8.7.25.21:
Part I の相互作用ラグランジアン L_int=g_P P_μ J^μ から、
Bullet 系オフセットで使う粗視化スケール L_path（=c_w tau_free）の
普遍スケーリング則を固定する。

固定出力:
  - output/public/cosmology/cosmology_lpath_scaling_law_prediction.json
  - output/public/cosmology/cosmology_lpath_scaling_law_prediction.csv
  - output/public/cosmology/cosmology_lpath_scaling_law_prediction.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.summary import worklog  # type: ignore  # noqa: E402
except Exception:
    worklog = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

KM_S_TO_KPC_GYR = 1.0227121650537077


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    label: str
    rho_ratio: float
    v_ratio: float
    temp_ratio: float
    note: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _render_png(path: Path, rows: Sequence[Dict[str, Any]], *, lpath0_kpc: float, tau0_gyr: float, pi0: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if plt is None:
        path.write_bytes(b"")
        return

    labels = [str(row["scenario_id"]) for row in rows]
    ratios = [float(row["lpath_ratio"]) for row in rows]
    tau_vals = [float(row["tau_free_pred_gyr"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=150)
    axes[0].bar(labels, ratios, color="#4c78a8")
    axes[0].axhline(1.0, color="#666666", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("L_path / L_path,0")
    axes[0].set_title("Step 8.7.25.21: universal scaling ratios")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(labels, tau_vals, color="#f58518")
    axes[1].axhline(float(tau0_gyr), color="#666666", linestyle="--", linewidth=1.0, label="baseline tau_free")
    axes[1].set_ylabel("tau_free predicted [Gyr]")
    axes[1].set_title("Coarse-grained relaxation time")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.suptitle("L_path scaling from L_int = g_P P_mu J^mu")
    fig.text(
        0.01,
        -0.02,
        (
            f"baseline: L_path,0={lpath0_kpc:.4f} kpc, tau_free,0={tau0_gyr:.6f} Gyr, "
            f"Pi0={pi0:.6f} (Pi0=tau_int,0/tau_free,0-1)"
        ),
        fontsize=8,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _default_scenarios() -> List[Scenario]:
    return [
        Scenario("baseline", "baseline", 1.0, 1.0, 1.0, "reference state"),
        Scenario("rho_x2", "density x2", 2.0, 1.0, 1.0, "rho→2rho"),
        Scenario("rho_x0_5", "density x0.5", 0.5, 1.0, 1.0, "rho→0.5rho"),
        Scenario("v_x2", "speed x2", 1.0, 2.0, 1.0, "v→2v"),
        Scenario("v_x0_5", "speed x0.5", 1.0, 0.5, 1.0, "v→0.5v"),
        Scenario("T_x2", "temperature x2", 1.0, 1.0, 2.0, "T→2T"),
        Scenario("T_x0_5", "temperature x0.5", 1.0, 1.0, 0.5, "T→0.5T"),
        Scenario("rho2_v2_T2", "rho,v,T all x2", 2.0, 2.0, 2.0, "combined hot+dense+fast"),
    ]


def _ratio_lpath(pi0: float, rho_ratio: float, v_ratio: float, temp_ratio: float, *, temp_power: float) -> float:
    rho_r = max(float(rho_ratio), 1.0e-12)
    v_r = max(float(v_ratio), 1.0e-12)
    t_r = max(float(temp_ratio), 1.0e-12)
    denom = v_r + pi0 * rho_r * (t_r ** temp_power)
    if not math.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float((1.0 + pi0) / denom)


def _load_reference(input_json: Path) -> Dict[str, Any]:
    if not input_json.exists():
        raise FileNotFoundError(f"missing input JSON: {input_json}")
    src = json.loads(input_json.read_text(encoding="utf-8"))
    tau_block = src.get("tau_origin_block", {})
    comp = tau_block.get("derived_components_gyr", {})
    tau_free = float(comp.get("tau_free", float("nan")))
    tau_int = float(comp.get("tau_int", float("nan")))

    retarded = src.get("assumptions", {}).get("retarded_branch", {})
    c_w = float(retarded.get("p_wave_speed_km_s", float("nan")))

    rows = src.get("cluster_rows", [])
    pi_values: List[float] = []
    for row in rows:
        try:
            tf = float(row.get("tau_free_gyr", float("nan")))
            ti = float(row.get("tau_int_gyr", float("nan")))
        except Exception:
            continue
        if (not math.isfinite(tf)) or (not math.isfinite(ti)) or tf <= 0.0 or ti <= 0.0:
            continue
        pi_values.append(max(ti / tf - 1.0, 0.0))

    if not pi_values and math.isfinite(tau_free) and math.isfinite(tau_int) and tau_free > 0.0 and tau_int > 0.0:
        pi_values = [max(tau_int / tau_free - 1.0, 0.0)]

    if (not math.isfinite(tau_free)) or tau_free <= 0.0:
        raise RuntimeError("invalid tau_free in input JSON")
    if (not math.isfinite(c_w)) or c_w <= 0.0:
        raise RuntimeError("invalid p_wave_speed_km_s in input JSON")
    if not pi_values:
        raise RuntimeError("unable to derive Pi0 from tau data")

    pi_arr = np.asarray(pi_values, dtype=float)
    pi0 = float(np.median(pi_arr))
    lpath0_kpc = float(tau_free * c_w * KM_S_TO_KPC_GYR)
    return {
        "source_json": src,
        "tau_free0_gyr": float(tau_free),
        "tau_int0_gyr": float(tau_int),
        "c_w_km_s": float(c_w),
        "lpath0_kpc": float(lpath0_kpc),
        "pi0_median": float(pi0),
        "pi0_mean": float(np.mean(pi_arr)),
        "pi0_std": float(np.std(pi_arr)),
        "n_pi_samples": int(pi_arr.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict universal L_path scaling from P_mu-J^mu derivation chain.")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=ROOT / "output/public/cosmology/cosmology_cluster_collision_pmu_jmu_separation_derivation.json",
        help="Input derivation JSON (tau_origin_block source).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "output/private/cosmology",
        help="Private output directory.",
    )
    parser.add_argument(
        "--public-outdir",
        type=Path,
        default=ROOT / "output/public/cosmology",
        help="Public output directory.",
    )
    parser.add_argument("--step-tag", default="8.7.25.21")
    parser.add_argument(
        "--temp-power",
        type=float,
        default=-1.5,
        help="Temperature exponent for collisional branch in Gamma_coll ~ rho * T^temp_power.",
    )
    args = parser.parse_args()

    ref = _load_reference(args.input_json)
    pi0 = float(ref["pi0_median"])
    tau0 = float(ref["tau_free0_gyr"])
    lpath0 = float(ref["lpath0_kpc"])
    c_w = float(ref["c_w_km_s"])
    temp_power = float(args.temp_power)

    scenarios = _default_scenarios()
    rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        ratio = _ratio_lpath(
            pi0,
            scenario.rho_ratio,
            scenario.v_ratio,
            scenario.temp_ratio,
            temp_power=temp_power,
        )
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "label": scenario.label,
                "rho_ratio": float(scenario.rho_ratio),
                "v_ratio": float(scenario.v_ratio),
                "temp_ratio": float(scenario.temp_ratio),
                "lpath_ratio": float(ratio),
                "lpath_pred_kpc": float(lpath0 * ratio),
                "tau_free_pred_gyr": float(tau0 * ratio),
                "note": scenario.note,
            }
        )

    row_map = {str(row["scenario_id"]): row for row in rows}
    ratio_rho_x2 = float(row_map["rho_x2"]["lpath_ratio"])
    ratio_v_x2 = float(row_map["v_x2"]["lpath_ratio"])
    ratio_t_x2 = float(row_map["T_x2"]["lpath_ratio"])

    checks = [
        {
            "check_id": "lpath_scaling::pi0_positive",
            "metric": "Pi0_median",
            "value": pi0,
            "expected": ">0",
            "hard_fail": (not math.isfinite(pi0)) or pi0 <= 0.0,
        },
        {
            "check_id": "lpath_scaling::density_monotonic",
            "metric": "L_path(2rho)/L_path(rho)",
            "value": ratio_rho_x2,
            "expected": "<1",
            "hard_fail": (not math.isfinite(ratio_rho_x2)) or ratio_rho_x2 >= 1.0,
        },
        {
            "check_id": "lpath_scaling::velocity_monotonic",
            "metric": "L_path(2v)/L_path(v)",
            "value": ratio_v_x2,
            "expected": "<1",
            "hard_fail": (not math.isfinite(ratio_v_x2)) or ratio_v_x2 >= 1.0,
        },
        {
            "check_id": "lpath_scaling::temperature_branch_sign",
            "metric": "L_path(2T)/L_path(T)",
            "value": ratio_t_x2,
            "expected": ">1 (Coulomb-like temp exponent)",
            "hard_fail": (not math.isfinite(ratio_t_x2)) or ratio_t_x2 <= 1.0,
        },
    ]

    hard_reject_n = int(sum(1 for check in checks if bool(check["hard_fail"])))
    overall_status = "pass" if hard_reject_n == 0 else "reject"
    decision = "lpath_scaling_law_fixed" if hard_reject_n == 0 else "lpath_scaling_law_reject"

    summary_lines = [
        f"overall_status={overall_status}, decision={decision}, hard_reject_n={hard_reject_n}",
        f"L_path(2rho)/L_path(rho)={ratio_rho_x2:.6f}",
        f"L_path(2v)/L_path(v)={ratio_v_x2:.6f}",
        f"L_path(2T)/L_path(T)={ratio_t_x2:.6f}",
    ]

    out_private_json = args.outdir / "cosmology_lpath_scaling_law_prediction.json"
    out_private_csv = args.outdir / "cosmology_lpath_scaling_law_prediction.csv"
    out_private_png = args.outdir / "cosmology_lpath_scaling_law_prediction.png"
    out_public_json = args.public_outdir / "cosmology_lpath_scaling_law_prediction.json"
    out_public_csv = args.public_outdir / "cosmology_lpath_scaling_law_prediction.csv"
    out_public_png = args.public_outdir / "cosmology_lpath_scaling_law_prediction.png"

    _write_csv(
        out_private_csv,
        rows,
        [
            "scenario_id",
            "label",
            "rho_ratio",
            "v_ratio",
            "temp_ratio",
            "lpath_ratio",
            "lpath_pred_kpc",
            "tau_free_pred_gyr",
            "note",
        ],
    )
    _write_csv(
        out_public_csv,
        rows,
        [
            "scenario_id",
            "label",
            "rho_ratio",
            "v_ratio",
            "temp_ratio",
            "lpath_ratio",
            "lpath_pred_kpc",
            "tau_free_pred_gyr",
            "note",
        ],
    )
    _render_png(out_private_png, rows, lpath0_kpc=lpath0, tau0_gyr=tau0, pi0=pi0)
    _render_png(out_public_png, rows, lpath0_kpc=lpath0, tau0_gyr=tau0, pi0=pi0)

    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "phase": {"phase": 8, "step": str(args.step_tag), "name": "lpath scaling law prediction"},
        "intent": (
            "Derive universal L_path scaling law for collision offsets from L_int=g_P P_mu J^mu "
            "and predict density/velocity/temperature response without ad-hoc injection."
        ),
        "inputs": {
            "source_tau_json": _rel(args.input_json),
            "temp_power_collisional_branch": temp_power,
        },
        "reference_state": {
            "tau_free_gyr": tau0,
            "tau_int_gyr": float(ref["tau_int0_gyr"]),
            "c_w_km_s": c_w,
            "lpath_kpc": lpath0,
            "pi0_median": pi0,
            "pi0_mean": float(ref["pi0_mean"]),
            "pi0_std": float(ref["pi0_std"]),
            "pi0_sample_count": int(ref["n_pi_samples"]),
            "pi0_from_tau_formula": "Pi0 = tau_int,0 / tau_free,0 - 1",
        },
        "prediction_equations": {
            "starting_lagrangian": "L_int = g_P P_mu J^mu",
            "memory_rate": "tau_free^{-1}=Gamma_path=(g_P^2/chi_P)∫_0^∞ <δJ_x(t)δJ_x(0)> dt",
            "transport_split": "Gamma_path=Gamma_adv+Gamma_coll",
            "advective_branch": "Gamma_adv=v/L_corr",
            "collisional_branch": "Gamma_coll=A_col * rho * T^{temp_power}",
            "lpath_balance": "L_path=c_w*tau_free=c_w/(v/L_corr + A_col*rho*T^{temp_power})",
            "universal_ratio": "L_path/L_path,0=(1+Pi0)/(r_v + Pi0*r_rho*r_T^{temp_power})",
            "ratio_symbols": {
                "r_rho": "rho/rho0",
                "r_v": "v/v0",
                "r_T": "T/T0",
                "Pi0": "A_col*rho0*T0^{temp_power}/(v0/L_corr)",
            },
        },
        "scenarios": rows,
        "checks": checks,
        "decision": {
            "overall_status": overall_status,
            "decision": decision,
            "hard_reject_n": hard_reject_n,
            "summary_lines": summary_lines,
        },
        "falsification_gate": {
            "reject_if": [
                "Pi0 <= 0 or non-finite.",
                "L_path(2rho)/L_path(rho) >= 1.",
                "L_path(2v)/L_path(v) >= 1.",
            ],
            "watch_if": [
                "L_path(2T)/L_path(T) <= 1 under Coulomb-like collisional branch.",
            ],
        },
        "outputs": {
            "private_json": _rel(out_private_json),
            "private_csv": _rel(out_private_csv),
            "private_png": _rel(out_private_png),
            "public_json": _rel(out_public_json),
            "public_csv": _rel(out_public_csv),
            "public_png": _rel(out_public_png),
        },
    }

    _write_json(out_private_json, payload)
    _write_json(out_public_json, payload)

    if worklog is not None:
        try:
            worklog.append_event(
                "cosmology.lpath_scaling_law_prediction",
                {
                    "status": overall_status,
                    "decision": decision,
                    "outputs": [_rel(out_public_json), _rel(out_public_csv), _rel(out_public_png)],
                },
            )
        except Exception:
            pass

    print(
        "[summary] status={0} decision={1} Pi0={2:.6f} "
        "L_path(2rho)/L_path(rho)={3:.6f} L_path(2v)/L_path(v)={4:.6f} L_path(2T)/L_path(T)={5:.6f}".format(
            overall_status,
            decision,
            pi0,
            ratio_rho_x2,
            ratio_v_x2,
            ratio_t_x2,
        )
    )
    print(f"[out] {out_public_json}")
    print(f"[out] {out_public_csv}")
    print(f"[out] {out_public_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
