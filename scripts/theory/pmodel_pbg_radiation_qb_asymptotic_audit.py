#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 8.7.22.16
Part I の作用原理（L_total）から得る背景波 P_bg の宇宙論的極限に、
黒体放射（p = u/3）を源として代入したときの漸近指数 q_B を監査する。

固定出力:
- output/public/theory/pmodel_pbg_radiation_qb_asymptotic_audit.json
- output/public/theory/pmodel_pbg_radiation_qb_asymptotic_audit.csv
- output/public/theory/pmodel_pbg_radiation_qb_asymptotic_audit.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.summary import worklog  # type: ignore
except Exception:  # pragma: no cover
    worklog = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.resolve().as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def _q_eff_exact(delta_t: float, c_val: float) -> float:
    if delta_t <= 0.0:
        raise ValueError("delta_t must be positive.")
    if abs(c_val) <= 1.0e-30:
        return 0.5
    if c_val > 0.0:
        x = 2.0 * math.sqrt(c_val) * delta_t
        return delta_t * math.sqrt(c_val) / math.tanh(x)
    x = 2.0 * math.sqrt(-c_val) * delta_t
    return delta_t * math.sqrt(-c_val) / math.tan(x)


def _build_payload(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    c_light = 299_792_458.0
    g_newton = 6.67430e-11
    sigma_sb = 5.670374419e-8
    a_rad = 4.0 * sigma_sb / c_light
    mev_to_k = 1.160451812e10

    t_b_sec = float(args.t_b_sec)
    t_b_mev = float(args.t_b_mev)
    t_b_kelvin = t_b_mev * mev_to_k
    eps_b = a_rad * (t_b_kelvin**4)

    # 背景波の宇宙論的極限（空間一様）:
    #   u_bg'' = (8πG/c^2) * eps_rad
    #   eps_rad = eps_B * exp[4(u_bg-u_B)]
    omega_sq = (8.0 * math.pi * g_newton / (c_light**2)) * eps_b
    omega = math.sqrt(max(omega_sq, 0.0))

    q_grid = np.linspace(float(args.q_min), float(args.q_max), int(args.q_count))
    exponent_delta = np.abs(2.0 - 4.0 * q_grid)
    best_idx = int(np.argmin(exponent_delta))
    q_best = float(q_grid[best_idx])
    q_target = 0.5
    q_err = abs(q_best - q_target)

    t_min = float(args.t_min_sec)
    t_max = float(args.t_max_sec)
    if not (t_min > 0.0 and t_max > t_min):
        raise ValueError("Require 0 < t_min < t_max for asymptotic diagnostic.")
    t_axis = np.geomspace(t_min, t_max, int(args.t_count))

    q_samples = [float(v) for v in args.q_samples.split(",") if str(v).strip()]
    q_samples = q_samples if q_samples else [0.4, 0.5, 0.6]

    sample_rows: list[dict[str, Any]] = []
    for q_val in q_samples:
        scaled = np.power(t_axis / t_b_sec, 2.0 - 4.0 * q_val)
        sample_rows.append(
            {
                "q_B": float(q_val),
                "scaled_factor_tmin": float(scaled[0]),
                "scaled_factor_tmax": float(scaled[-1]),
                "asymptotic_behavior_t_to_0": (
                    "finite_nonzero"
                    if abs(q_val - 0.5) <= 1.0e-12
                    else ("vanish_to_0" if q_val < 0.5 else "diverge_to_inf")
                ),
            }
        )

    cbar_samples = _parse_float_list(args.cbar_samples)
    if not cbar_samples:
        cbar_samples = [-0.2, 0.0, 0.2]
    dt_probe = np.geomspace(float(args.dt_probe_min_sec), float(args.dt_probe_max_sec), int(args.dt_probe_count))
    exact_rows: list[dict[str, Any]] = []
    exact_curve_rows: list[dict[str, Any]] = []
    for cbar in cbar_samples:
        c_val = float(cbar) * float(omega_sq)
        q_values = []
        for dt_val in dt_probe:
            q_eff = _q_eff_exact(float(dt_val), c_val)
            q_values.append(q_eff)
            exact_curve_rows.append(
                {
                    "cbar": float(cbar),
                    "delta_t_sec": float(dt_val),
                    "q_eff": float(q_eff),
                    "abs_q_eff_minus_half": float(abs(q_eff - 0.5)),
                }
            )
        exact_rows.append(
            {
                "cbar": float(cbar),
                "C0_per_s2": float(c_val),
                "q_eff_at_dt_min": float(q_values[0]),
                "q_eff_at_dt_max": float(q_values[-1]),
                "max_abs_q_eff_minus_half_on_probe": float(max(abs(v - 0.5) for v in q_values)),
                "asymptotic_limit": "q_eff -> 1/2",
            }
        )

    derivation_chain = [
        "L_total のスカラー弱場枝（u=ln(P/P_ref)）から、(1/c^2)ü-∇²u=(4πG/c^2)ρ_eff を採用する。",
        "完全流体の有効源は ρ_eff=ρ+3p/c^2。初期宇宙の黒体放射では p=u_rad/3, ρ=u_rad/c^2 より ρ_eff=2u_rad/c^2。",
        "空間一様背景（∇²u_bg=0）で ü_bg=(8πG/c^2)u_rad。",
        "黒体式 u_rad=a_rad T^4 と T∝P_bg, P_bg=P_ref exp(u_bg-u_ref) を使うと ü_bg=Ω_B^2 exp[4(u_bg-u_B)]。",
        "初期極限 t→0+ で u_bg=u_B-q_B ln(t/t_B)+O(1) を代入すると、左辺は t^-2、右辺は t^-4q_B で支配される。",
        "優勢項一致条件 2=4q_B から q_B=1/2 が一意に定まる。",
        "同じ結論は第一積分 (u̇_bg)^2-(Ω_B^2/2)exp[4(u_bg-u_B)]=C からも得られ、t→0+ では exp項が支配して u_bg=-1/2 ln t + O(1)。",
    ]

    payload: dict[str, Any] = {
        "generated_utc": _utc_now(),
        "phase": {"phase": 8, "step": str(args.step_tag), "name": "Background-P radiation asymptotic q_B derivation"},
        "inputs": {
            "t_B_sec": t_b_sec,
            "T_B_MeV": t_b_mev,
            "T_B_K": t_b_kelvin,
            "constants": {
                "c_m_per_s": c_light,
                "G_SI": g_newton,
                "sigma_SB_SI": sigma_sb,
                "a_rad_SI": a_rad,
            },
            "equation_of_state": "p = u_rad / 3",
            "scaling": "T ∝ P_bg, P_bg = P_ref * exp(u_bg-u_ref)",
        },
        "equations": {
            "background_wave_eq": "(1/c^2) d2u_bg/dt2 = (4πG/c^2) ρ_eff",
            "effective_source_perfect_fluid": "ρ_eff = ρ + 3p/c^2",
            "radiation_substitution": "ρ_eff = 2 u_rad / c^2 (from p=u_rad/3, ρ=u_rad/c^2)",
            "radiation_energy_density": "u_rad = a_rad T^4 = u_B * exp(4(u_bg-u_B))",
            "ode_background": "d2u_bg/dt2 = Ω_B^2 * exp(4(u_bg-u_B)), Ω_B^2=(8πG/c^2)u_B",
            "asymptotic_ansatz": "u_bg(t)=u_B-q_B ln(t/t_B)+O(1), t→0+",
            "dominant_balance": "t^-2 = t^(-4 q_B) => q_B = 1/2",
            "first_integral": "(du_bg/dt)^2 - (Ω_B^2/2) exp(4(u_bg-u_B)) = C",
            "first_integral_asymptotic": "u_bg(t) = -0.5 ln t + O(1) (cooling branch, t→0+)",
            "exact_reduction": "z=e^(-2(u_bg-u_B)) => (dz/dt)^2 = 2 Ω_B^2 + 4 C z^2",
            "exact_solution_C_gt_0": "z=(Ω_B/sqrt(2C)) sinh(2 sqrt(C) Δt), C>0",
            "exact_solution_C_eq_0": "z=sqrt(2) Ω_B Δt, C=0",
            "exact_solution_C_lt_0": "z=(Ω_B/sqrt(-2C)) sin(2 sqrt(-C) Δt), C<0",
            "universal_asymptotic": "z = sqrt(2) Ω_B Δt + O(Δt^3) => u_bg = u_B - 1/2 ln Δt + O(Δt^2)",
        },
        "derivation_chain": derivation_chain,
        "diagnostics": {
            "omega_B_sq_per_s2": float(omega_sq),
            "omega_B_per_s": float(omega),
            "q_scan": {
                "q_min": float(args.q_min),
                "q_max": float(args.q_max),
                "q_count": int(args.q_count),
                "best_q_from_exponent_match": q_best,
                "target_q": q_target,
                "abs_error": q_err,
            },
            "sample_scaled_behavior": sample_rows,
            "asymptotic_limits_rule": {
                "q<0.5": "scaled factor t^(2-4q) -> 0 as t->0+",
                "q=0.5": "scaled factor finite nonzero (dominant-balance closure)",
                "q>0.5": "scaled factor -> +∞ as t->0+",
            },
            "exact_solution_family_probe": {
                "dt_probe_min_sec": float(args.dt_probe_min_sec),
                "dt_probe_max_sec": float(args.dt_probe_max_sec),
                "dt_probe_count": int(args.dt_probe_count),
                "summary": exact_rows,
                "curves": exact_curve_rows,
            },
        },
        "decision": {
            "qB_asymptotic_value": 0.5,
            "qB_asymptotic_pass": bool(q_err <= float(args.q_tol)),
            "qB_exact_family_pass": bool(
                max(abs(row.get("q_eff_at_dt_min", 0.5) - 0.5) for row in exact_rows) <= float(args.q_exact_tol)
            ),
            "qB_exact_family_tolerance": float(args.q_exact_tol),
            "qB_tolerance": float(args.q_tol),
            "status": (
                "pass"
                if (
                    q_err <= float(args.q_tol)
                    and max(abs(row.get("q_eff_at_dt_min", 0.5) - 0.5) for row in exact_rows) <= float(args.q_exact_tol)
                )
                else "watch"
            ),
            "note": "q_B is fixed by dominant-balance exponent matching and exact-solution-family asymptotics in the sourced background-wave ODE.",
        },
    }

    csv_rows: list[dict[str, Any]] = []
    for q_val, d_val in zip(q_grid.tolist(), exponent_delta.tolist()):
        csv_rows.append(
            {
                "q_B": float(q_val),
                "exponent_mismatch_abs_2_minus_4q": float(d_val),
                "closure_flag": "pass" if abs(float(q_val) - 0.5) <= 1.0e-12 else "mismatch",
            }
        )
    return payload, csv_rows


def _plot(path: Path, payload: dict[str, Any]) -> None:
    if plt is None:
        return
    diag = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    q_scan = diag.get("q_scan") if isinstance(diag.get("q_scan"), dict) else {}
    q_min = float(q_scan.get("q_min", 0.2))
    q_max = float(q_scan.get("q_max", 0.8))
    q_count = int(q_scan.get("q_count", 501))
    q_axis = np.linspace(q_min, q_max, q_count)
    mismatch = np.abs(2.0 - 4.0 * q_axis)

    t_min = 1.0e-8
    t_max = 1.0
    t_axis = np.geomspace(t_min, t_max, 400)
    q_samples = [0.4, 0.5, 0.6]

    exact_probe = diag.get("exact_solution_family_probe") if isinstance(diag.get("exact_solution_family_probe"), dict) else {}
    exact_curves = exact_probe.get("curves") if isinstance(exact_probe.get("curves"), list) else []

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 4.8), dpi=180)
    fig.suptitle("Background-P radiation asymptotic audit: q_B closure from sourced wave equation", y=1.02)

    ax = axes[0]
    ax.plot(q_axis, mismatch, color="#1f77b4", lw=1.7, label="abs(2 - 4 q_B)")
    ax.axvline(0.5, color="#d62728", lw=1.2, ls="--", label="q_B = 0.5")
    ax.set_xlabel("q_B")
    ax.set_ylabel("dominant-balance exponent mismatch")
    ax.set_title("Exponent closure condition")
    ax.grid(True, ls=":", alpha=0.45)
    ax.legend(loc="upper right")

    ax = axes[1]
    for q in q_samples:
        scale = np.power(t_axis, 2.0 - 4.0 * q)
        ax.loglog(t_axis, scale, lw=1.6, label=f"q_B={q:.1f}")
    ax.set_xlabel("t / t_B")
    ax.set_ylabel("scaled source factor: (t/t_B)^(2-4 q_B)")
    ax.set_title("t -> 0+ asymptotic behavior")
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(loc="best")

    ax = axes[2]
    curve_by_cbar: dict[float, list[tuple[float, float]]] = {}
    for row in exact_curves:
        if not isinstance(row, dict):
            continue
        cbar = float(row.get("cbar", 0.0))
        dt_val = float(row.get("delta_t_sec", 0.0))
        q_eff = float(row.get("q_eff", 0.5))
        curve_by_cbar.setdefault(cbar, []).append((dt_val, q_eff))
    for cbar, rows in sorted(curve_by_cbar.items(), key=lambda item: item[0]):
        rows_sorted = sorted(rows, key=lambda item: item[0])
        xs = [item[0] for item in rows_sorted]
        ys = [item[1] for item in rows_sorted]
        ax.semilogx(xs, ys, lw=1.6, label=f"C/Ω²={cbar:+.2f}")
    ax.axhline(0.5, color="#d62728", lw=1.2, ls="--", label="q_eff=0.5")
    ax.set_xlabel("Δt [s]")
    ax.set_ylabel("q_eff(Δt) = -Δt du_bg/dt")
    ax.set_title("Exact-family convergence to 1/2")
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(loc="best")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit q_B=1/2 asymptotic closure from background-wave radiation source.")
    parser.add_argument("--step-tag", type=str, default="8.7.22.16")
    parser.add_argument("--t-b-sec", type=float, default=1.0, help="Reference BBN time t_B [s].")
    parser.add_argument("--t-b-mev", type=float, default=1.0, help="Reference BBN temperature T_B [MeV].")
    parser.add_argument("--q-min", type=float, default=0.2)
    parser.add_argument("--q-max", type=float, default=0.8)
    parser.add_argument("--q-count", type=int, default=601)
    parser.add_argument("--q-tol", type=float, default=1.0e-12)
    parser.add_argument("--t-min-sec", type=float, default=1.0e-8)
    parser.add_argument("--t-max-sec", type=float, default=1.0)
    parser.add_argument("--t-count", type=int, default=400)
    parser.add_argument("--q-samples", type=str, default="0.4,0.5,0.6")
    parser.add_argument("--cbar-samples", type=str, default="-0.2,0.0,0.2")
    parser.add_argument("--dt-probe-min-sec", type=float, default=1.0e-6)
    parser.add_argument("--dt-probe-max-sec", type=float, default=1.0e-2)
    parser.add_argument("--dt-probe-count", type=int, default=9)
    parser.add_argument("--q-exact-tol", type=float, default=1.0e-3)
    parser.add_argument("--outdir", type=Path, default=ROOT / "output" / "public" / "theory")
    args = parser.parse_args()

    out_dir = args.outdir
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "pmodel_pbg_radiation_qb_asymptotic_audit.json"
    out_csv = out_dir / "pmodel_pbg_radiation_qb_asymptotic_audit.csv"
    out_png = out_dir / "pmodel_pbg_radiation_qb_asymptotic_audit.png"

    payload, csv_rows = _build_payload(args)
    _write_json(out_json, payload)
    _write_csv(
        out_csv,
        csv_rows,
        fieldnames=["q_B", "exponent_mismatch_abs_2_minus_4q", "closure_flag"],
    )
    _plot(out_png, payload)

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")
    print(f"[ok] wrote: {_rel(out_png)}")

    if worklog is not None:
        try:
            worklog.append_event(
                {
                    "event_type": "theory_pbg_radiation_qb_asymptotic_audit",
                    "phase": str(args.step_tag),
                    "outputs": {
                        "json": _rel(out_json),
                        "csv": _rel(out_csv),
                        "png": _rel(out_png),
                    },
                    "decision": payload.get("decision"),
                }
            )
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
