#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pmodel_blackhole_interior_regularization_audit.py

Step 8.7.30:
静的球対称の P 場内部解を r→0 まで追跡し、
地平面形成有無・中心有限性・数値安定性を同一I/Fで監査する。

固定出力:
- output/public/theory/pmodel_blackhole_interior_regularization_audit.json
- output/public/theory/pmodel_blackhole_interior_regularization_audit.csv
- output/public/theory/pmodel_blackhole_interior_regularization_audit.png
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

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402

G = 6.67430e-11
C = 299792458.0
M_SUN = 1.98847e30


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


# 関数: `_fmt_float` の入出力契約と処理意図を定義する。

def _fmt_float(x: float, digits: int = 6) -> str:
    # 条件分岐: `x == 0.0` を満たす経路を評価する。
    if x == 0.0:
        return "0"

    ax = abs(x)
    # 条件分岐: `ax >= 1.0e4 or ax < 1.0e-3` を満たす経路を評価する。
    if ax >= 1.0e4 or ax < 1.0e-3:
        return f"{x:.{digits}g}"

    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。

def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = [
            "Yu Gothic",
            "Meiryo",
            "BIZ UDGothic",
            "MS Gothic",
            "Yu Mincho",
            "MS Mincho",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        return


# 関数: `_integrate_psi` の入出力契約と処理意図を定義する。

def _integrate_psi(x: np.ndarray, compactness_c: float) -> np.ndarray:
    integrand = compactness_c * x / (x**3 + 1.0)
    xr = x[::-1]
    fr = integrand[::-1]
    integ_rev = np.zeros_like(xr)
    dx = xr[:-1] - xr[1:]
    integ_rev[1:] = np.cumsum(0.5 * (fr[:-1] + fr[1:]) * dx)
    tail = compactness_c / max(float(x[-1]), 1.0)
    return integ_rev[::-1] + tail


# 関数: `_single_audit` の入出力契約と処理意図を定義する。

def _single_audit(
    *,
    compactness_c: float,
    mass_kg: float,
    horizon_gtt_eps: float,
    x_min: float,
    x_max: float,
    grid_points: int,
) -> Dict[str, Any]:
    x = np.geomspace(x_min, x_max, grid_points)
    psi = _integrate_psi(x, compactness_c)
    phi = -C * C * psi
    gtt = np.exp(np.clip(-2.0 * psi, -700.0, 700.0))

    mass_ratio = x**3 / (x**3 + 1.0)
    compactness_local = 2.0 * compactness_c * mass_ratio / x
    integrand = compactness_c * x / (x**3 + 1.0)
    dpsi_num = np.gradient(psi, x, edge_order=2)
    ode_residual = np.abs(dpsi_num + integrand)

    core_radius_m = (G * mass_kg) / (C * C * compactness_c)
    schwarzschild_radius_m = 2.0 * G * mass_kg / (C * C)

    central_density = 3.0 * mass_kg / (4.0 * math.pi * core_radius_m**3)
    curvature_center = 48.0 * (G * G) * (mass_kg * mass_kg) / ((C**4) * (core_radius_m**6))

    psi_center = float(psi[0])
    phi_center = float(phi[0])
    p_ratio_center = float(math.exp(min(psi_center, 700.0)))
    gtt_min = float(np.min(gtt))
    compactness_max = float(np.max(compactness_local))

    central_density_finite = bool(math.isfinite(central_density) and central_density > 0.0)
    curvature_invariant_finite = bool(math.isfinite(curvature_center) and curvature_center > 0.0)
    central_quantity_finite = bool(
        central_density_finite
        and curvature_invariant_finite
        and math.isfinite(phi_center)
        and math.isfinite(p_ratio_center)
    )

    finite_arrays = bool(
        np.all(np.isfinite(x))
        and np.all(np.isfinite(psi))
        and np.all(np.isfinite(gtt))
        and np.all(np.isfinite(compactness_local))
    )
    monotonic_mass = bool(np.all(np.diff(mass_ratio) >= -1.0e-12))
    monotonic_psi_outward = bool(np.all(np.diff(psi) <= 1.0e-8))
    ode_residual_max = float(np.max(ode_residual))
    ode_residual_p95 = float(np.percentile(ode_residual, 95.0))
    numerical_stability_pass = bool(
        finite_arrays and monotonic_mass and monotonic_psi_outward and ode_residual_max <= 2.0e-3
    )

    event_horizon_formed = bool(gtt_min <= horizon_gtt_eps)
    event_horizon_gr_proxy = bool(compactness_max >= 1.0)

    return {
        "compactness_c": float(compactness_c),
        "mass_kg": float(mass_kg),
        "core_radius_m": float(core_radius_m),
        "schwarzschild_radius_m": float(schwarzschild_radius_m),
        "radius_ratio_rs_over_core": float(schwarzschild_radius_m / core_radius_m),
        "event_horizon_formed": event_horizon_formed,
        "event_horizon_gr_proxy": event_horizon_gr_proxy,
        "horizon_metrics": {
            "gtt_min": gtt_min,
            "gtt_horizon_eps": float(horizon_gtt_eps),
            "compactness_max_2gm_over_rc2": compactness_max,
        },
        "central_values": {
            "psi_center": psi_center,
            "phi_center_m2_s2": phi_center,
            "p_ratio_center": p_ratio_center,
            "central_density_kg_m3": float(central_density),
            "curvature_kretschmann_center_m4": float(curvature_center),
        },
        "central_density_finite": central_density_finite,
        "curvature_invariant_finite": curvature_invariant_finite,
        "central_quantity_finite": central_quantity_finite,
        "numerical_stability": {
            "pass": numerical_stability_pass,
            "finite_arrays": finite_arrays,
            "monotonic_mass_enclosed": monotonic_mass,
            "monotonic_psi_outward": monotonic_psi_outward,
            "ode_residual_max": ode_residual_max,
            "ode_residual_p95": ode_residual_p95,
        },
        "profiles": {
            "x": x.tolist(),
            "psi": psi.tolist(),
            "gtt": gtt.tolist(),
            "compactness_local": compactness_local.tolist(),
        },
    }


# 関数: `_build_sweep` の入出力契約と処理意図を定義する。

def _build_sweep(
    *,
    mass_kg: float,
    c_min: float,
    c_max: float,
    n_points: int,
    horizon_gtt_eps: float,
    x_min: float,
    x_max: float,
    grid_points: int,
) -> Dict[str, Any]:
    c_grid = np.geomspace(c_min, c_max, n_points)
    rows: List[Dict[str, Any]] = []
    for c_val in c_grid:
        row = _single_audit(
            compactness_c=float(c_val),
            mass_kg=mass_kg,
            horizon_gtt_eps=horizon_gtt_eps,
            x_min=x_min,
            x_max=x_max,
            grid_points=grid_points,
        )
        rows.append(
            {
                "compactness_c": row["compactness_c"],
                "core_radius_m": row["core_radius_m"],
                "event_horizon_formed": row["event_horizon_formed"],
                "event_horizon_gr_proxy": row["event_horizon_gr_proxy"],
                "gtt_min": row["horizon_metrics"]["gtt_min"],
                "compactness_max_2gm_over_rc2": row["horizon_metrics"]["compactness_max_2gm_over_rc2"],
                "central_density_kg_m3": row["central_values"]["central_density_kg_m3"],
                "curvature_kretschmann_center_m4": row["central_values"]["curvature_kretschmann_center_m4"],
                "central_quantity_finite": row["central_quantity_finite"],
                "numerical_stability_pass": row["numerical_stability"]["pass"],
            }
        )

    first_p_horizon = next((r for r in rows if bool(r["event_horizon_formed"])), None)
    first_gr_horizon = next((r for r in rows if bool(r["event_horizon_gr_proxy"])), None)

    return {
        "rows": rows,
        "n_points": int(len(rows)),
        "first_event_horizon_compactness_c": None if first_p_horizon is None else first_p_horizon["compactness_c"],
        "first_gr_proxy_horizon_compactness_c": None if first_gr_horizon is None else first_gr_horizon["compactness_c"],
    }


# 関数: `_overall_decision` の入出力契約と処理意図を定義する。

def _overall_decision(baseline: Dict[str, Any]) -> Dict[str, Any]:
    central_finite = bool(baseline.get("central_quantity_finite"))
    stability = bool((baseline.get("numerical_stability") or {}).get("pass"))
    event_horizon_formed = bool(baseline.get("event_horizon_formed"))
    # 条件分岐: `central_finite and stability` を満たす経路を評価する。
    if central_finite and stability:
        status = "pass"
        decision = "interior_regularized"
    else:
        status = "reject"
        decision = "interior_breakdown"

    # 条件分岐: `not central_finite` を満たす経路を評価する。

    if not central_finite:
        reason = "central_quantity_not_finite"
    # 条件分岐: 前段条件が不成立で、`not stability` を追加評価する。
    elif not stability:
        reason = "numerical_stability_fail"
    else:
        reason = "none"

    return {
        "overall_status": status,
        "decision": decision,
        "event_horizon_formed": event_horizon_formed,
        "central_quantity_finite": central_finite,
        "numerical_stability_pass": stability,
        "reject_reason": reason,
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, sweep_rows: List[Dict[str, Any]], baseline_c: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "is_baseline",
                "compactness_c",
                "core_radius_m",
                "event_horizon_formed",
                "event_horizon_gr_proxy",
                "gtt_min",
                "compactness_max_2gm_over_rc2",
                "central_density_kg_m3",
                "curvature_kretschmann_center_m4",
                "central_quantity_finite",
                "numerical_stability_pass",
            ]
        )
        for row in sweep_rows:
            is_baseline = abs(float(row["compactness_c"]) - baseline_c) / baseline_c <= 5.0e-3
            w.writerow(
                [
                    "1" if is_baseline else "0",
                    _fmt_float(float(row["compactness_c"]), 8),
                    _fmt_float(float(row["core_radius_m"]), 8),
                    "1" if bool(row["event_horizon_formed"]) else "0",
                    "1" if bool(row["event_horizon_gr_proxy"]) else "0",
                    _fmt_float(float(row["gtt_min"]), 8),
                    _fmt_float(float(row["compactness_max_2gm_over_rc2"]), 8),
                    _fmt_float(float(row["central_density_kg_m3"]), 8),
                    _fmt_float(float(row["curvature_kretschmann_center_m4"]), 8),
                    "1" if bool(row["central_quantity_finite"]) else "0",
                    "1" if bool(row["numerical_stability_pass"]) else "0",
                ]
            )


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(
    *,
    out_png: Path,
    baseline: Dict[str, Any],
    sweep_rows: List[Dict[str, Any]],
    horizon_gtt_eps: float,
) -> None:
    _set_japanese_font()

    x = np.asarray((baseline.get("profiles") or {}).get("x") or [], dtype=float)
    gtt = np.asarray((baseline.get("profiles") or {}).get("gtt") or [], dtype=float)
    comp_local = np.asarray((baseline.get("profiles") or {}).get("compactness_local") or [], dtype=float)

    c_vals = np.asarray([float(r["compactness_c"]) for r in sweep_rows], dtype=float)
    gtt_min = np.asarray([float(r["gtt_min"]) for r in sweep_rows], dtype=float)
    comp_max = np.asarray([float(r["compactness_max_2gm_over_rc2"]) for r in sweep_rows], dtype=float)
    rho_center = np.asarray([float(r["central_density_kg_m3"]) for r in sweep_rows], dtype=float)
    k_center = np.asarray([float(r["curvature_kretschmann_center_m4"]) for r in sweep_rows], dtype=float)

    fig = plt.figure(figsize=(14.0, 5.4), dpi=180)
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    ax0.plot(x, gtt, color="#1f77b4", linewidth=2.0, label="P-model g_tt proxy")
    ax0.axhline(horizon_gtt_eps, color="#d62728", linestyle="--", linewidth=1.2, label="operational horizon gate")
    ax0_t = ax0.twinx()
    ax0_t.plot(x, comp_local, color="#ff7f0e", linestyle="-.", linewidth=1.6, label="2GM(r)/(rc²)")
    ax0_t.axhline(1.0, color="#444444", linestyle=":", linewidth=1.1)
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel("x = r / r_core")
    ax0.set_ylabel("g_tt proxy = exp(2φ/c²)")
    ax0_t.set_ylabel("compactness proxy")
    ax0.set_title("Baseline interior profile")
    ax0.grid(True, which="both", alpha=0.22)
    lines0, labels0 = ax0.get_legend_handles_labels()
    lines1, labels1 = ax0_t.get_legend_handles_labels()
    ax0.legend(lines0 + lines1, labels0 + labels1, loc="lower right", fontsize=8)

    ax1.plot(c_vals, gtt_min, color="#1f77b4", linewidth=2.0, label="min g_tt")
    ax1.axhline(horizon_gtt_eps, color="#d62728", linestyle="--", linewidth=1.2, label="horizon gate")
    ax1_t = ax1.twinx()
    ax1_t.plot(c_vals, comp_max, color="#ff7f0e", linestyle="-.", linewidth=1.6, label="max 2GM/(rc²)")
    ax1_t.axhline(1.0, color="#444444", linestyle=":", linewidth=1.1)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1_t.set_yscale("log")
    ax1.set_xlabel("compactness C = GM/(c² r_core)")
    ax1.set_ylabel("min g_tt")
    ax1_t.set_ylabel("max compactness proxy")
    ax1.set_title("Horizon formation sweep")
    ax1.grid(True, which="both", alpha=0.22)
    lines2, labels2 = ax1.get_legend_handles_labels()
    lines3, labels3 = ax1_t.get_legend_handles_labels()
    ax1.legend(lines2 + lines3, labels2 + labels3, loc="upper left", fontsize=8)

    ax2.plot(c_vals, rho_center, color="#2ca02c", linewidth=2.0, label="central density")
    ax2_t = ax2.twinx()
    ax2_t.plot(c_vals, k_center, color="#9467bd", linestyle="-.", linewidth=1.7, label="Kretschmann proxy")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2_t.set_yscale("log")
    ax2.set_xlabel("compactness C")
    ax2.set_ylabel("ρ(0) [kg/m³]")
    ax2_t.set_ylabel("K(0) proxy [m⁻⁴]")
    ax2.set_title("Central regularity scan")
    ax2.grid(True, which="both", alpha=0.22)
    lines4, labels4 = ax2.get_legend_handles_labels()
    lines5, labels5 = ax2_t.get_legend_handles_labels()
    ax2.legend(lines4 + lines5, labels4 + labels5, loc="upper left", fontsize=8)

    fig.suptitle("P-model blackhole interior regularization audit (Step 8.7.30)", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Step 8.7.30 blackhole interior regularization audit.")
    parser.add_argument("--mass-msun", type=float, default=6.5e9, help="Reference mass [solar mass].")
    parser.add_argument(
        "--core-radius-rs-frac",
        type=float,
        default=0.35,
        help="Core radius as fraction of Schwarzschild radius (r_core = frac * r_s).",
    )
    parser.add_argument("--scan-c-min", type=float, default=0.2, help="Compactness scan minimum.")
    parser.add_argument("--scan-c-max", type=float, default=20.0, help="Compactness scan maximum.")
    parser.add_argument("--scan-c-points", type=int, default=96, help="Compactness scan points.")
    parser.add_argument("--horizon-gtt-eps", type=float, default=1.0e-12, help="Operational horizon threshold on min(g_tt).")
    parser.add_argument("--x-min", type=float, default=1.0e-8, help="Minimum x=r/r_core for integration.")
    parser.add_argument("--x-max", type=float, default=1.0e5, help="Maximum x=r/r_core for integration.")
    parser.add_argument("--grid-points", type=int, default=6000, help="Grid points for radial integration.")
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "output" / "public" / "theory" / "pmodel_blackhole_interior_regularization_audit.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(ROOT / "output" / "public" / "theory" / "pmodel_blackhole_interior_regularization_audit.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default=str(ROOT / "output" / "public" / "theory" / "pmodel_blackhole_interior_regularization_audit.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args(argv)

    # 条件分岐: `args.mass_msun <= 0.0` を満たす経路を評価する。
    if args.mass_msun <= 0.0:
        raise SystemExit("--mass-msun must be > 0")

    # 条件分岐: `args.core_radius_rs_frac <= 0.0` を満たす経路を評価する。

    if args.core_radius_rs_frac <= 0.0:
        raise SystemExit("--core-radius-rs-frac must be > 0")

    # 条件分岐: `args.scan_c_min <= 0.0 or args.scan_c_max <= 0.0 or args.scan_c_max <= args.s...` を満たす経路を評価する。

    if args.scan_c_min <= 0.0 or args.scan_c_max <= 0.0 or args.scan_c_max <= args.scan_c_min:
        raise SystemExit("scan compactness bounds must satisfy 0 < min < max")

    # 条件分岐: `args.scan_c_points < 8` を満たす経路を評価する。

    if args.scan_c_points < 8:
        raise SystemExit("--scan-c-points must be >= 8")

    # 条件分岐: `args.x_min <= 0.0 or args.x_max <= args.x_min` を満たす経路を評価する。

    if args.x_min <= 0.0 or args.x_max <= args.x_min:
        raise SystemExit("integration range must satisfy 0 < x_min < x_max")

    # 条件分岐: `args.grid_points < 512` を満たす経路を評価する。

    if args.grid_points < 512:
        raise SystemExit("--grid-points must be >= 512")

    # 条件分岐: `args.horizon_gtt_eps <= 0.0 or args.horizon_gtt_eps >= 1.0` を満たす経路を評価する。

    if args.horizon_gtt_eps <= 0.0 or args.horizon_gtt_eps >= 1.0:
        raise SystemExit("--horizon-gtt-eps must be in (0,1)")

    out_json = Path(args.out_json).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_png = Path(args.out_png).resolve()

    mass_kg = float(args.mass_msun) * M_SUN
    compactness_baseline = 1.0 / (2.0 * float(args.core_radius_rs_frac))

    baseline = _single_audit(
        compactness_c=compactness_baseline,
        mass_kg=mass_kg,
        horizon_gtt_eps=float(args.horizon_gtt_eps),
        x_min=float(args.x_min),
        x_max=float(args.x_max),
        grid_points=int(args.grid_points),
    )
    sweep = _build_sweep(
        mass_kg=mass_kg,
        c_min=float(args.scan_c_min),
        c_max=float(args.scan_c_max),
        n_points=int(args.scan_c_points),
        horizon_gtt_eps=float(args.horizon_gtt_eps),
        x_min=float(args.x_min),
        x_max=float(args.x_max),
        grid_points=int(args.grid_points),
    )
    decision = _overall_decision(baseline)

    gr_reference = {
        "model": "Schwarzschild_point_mass_reference",
        "event_horizon_formed": True,
        "central_density_finite": False,
        "curvature_invariant_finite": False,
        "reason": "point-mass GR interior has singular center (r->0) while horizon exists at r_s=2GM/c^2",
    }

    payload: Dict[str, Any] = {
        "phase": {"id": "phase_8", "step": "8.7.30", "title": "blackhole_interior_regularization_audit"},
        "generated_utc": _utc_now(),
        "model": {
            "potential_definition": "phi = -c^2 ln(P/P0)",
            "mass_profile": "M(r)=M*r^3/(r^3+r_core^3)",
            "interior_equation": "d(ln(P/P0))/dr = -GM(r)/(c^2 r^2)",
            "event_horizon_proxy": "min(exp(2phi/c^2)) <= horizon_gtt_eps",
            "gr_proxy": "max(2GM(r)/(r c^2)) >= 1",
        },
        "inputs": {
            "mass_msun": float(args.mass_msun),
            "mass_kg": mass_kg,
            "core_radius_rs_frac": float(args.core_radius_rs_frac),
            "compactness_baseline_c": compactness_baseline,
            "scan_c_min": float(args.scan_c_min),
            "scan_c_max": float(args.scan_c_max),
            "scan_c_points": int(args.scan_c_points),
            "horizon_gtt_eps": float(args.horizon_gtt_eps),
            "x_min": float(args.x_min),
            "x_max": float(args.x_max),
            "grid_points": int(args.grid_points),
        },
        "baseline": {
            "compactness_c": baseline["compactness_c"],
            "core_radius_m": baseline["core_radius_m"],
            "schwarzschild_radius_m": baseline["schwarzschild_radius_m"],
            "radius_ratio_rs_over_core": baseline["radius_ratio_rs_over_core"],
            "event_horizon_formed": baseline["event_horizon_formed"],
            "event_horizon_gr_proxy": baseline["event_horizon_gr_proxy"],
            "central_density_finite": baseline["central_density_finite"],
            "curvature_invariant_finite": baseline["curvature_invariant_finite"],
            "central_quantity_finite": baseline["central_quantity_finite"],
            "numerical_stability_pass": baseline["numerical_stability"]["pass"],
            "horizon_metrics": baseline["horizon_metrics"],
            "central_values": baseline["central_values"],
            "numerical_stability": baseline["numerical_stability"],
        },
        "sweep": sweep,
        "gr_reference": gr_reference,
        "decision": decision,
        "falsification_gate": {
            "pass_if": [
                "central_quantity_finite=true",
                "numerical_stability_pass=true",
            ],
            "reject_if": [
                "central_quantity_finite=false",
                "numerical_stability_pass=false",
            ],
            "report_always": [
                "event_horizon_formed",
                "event_horizon_gr_proxy",
                "first_event_horizon_compactness_c",
                "first_gr_proxy_horizon_compactness_c",
            ],
        },
        "outputs": {
            "audit_json": _rel(out_json),
            "audit_csv": _rel(out_csv),
            "audit_png": _rel(out_png),
        },
    }

    _write_json(out_json, payload)
    _write_csv(out_csv, sweep["rows"], baseline_c=compactness_baseline)
    _plot(
        out_png=out_png,
        baseline=baseline,
        sweep_rows=sweep["rows"],
        horizon_gtt_eps=float(args.horizon_gtt_eps),
    )

    try:
        worklog.append_event(
            {
                "event_type": "theory_blackhole_interior_regularization_audit",
                "phase": "8.7.30",
                "inputs": payload.get("inputs"),
                "outputs": payload.get("outputs"),
                "decision": payload.get("decision"),
            }
        )
    except Exception:
        pass

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")
    print(f"[ok] wrote: {_rel(out_png)}")
    print(
        "[summary] status={0} horizon={1} central_finite={2} stability={3}".format(
            decision["overall_status"],
            decision["event_horizon_formed"],
            decision["central_quantity_finite"],
            decision["numerical_stability_pass"],
        )
    )
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
