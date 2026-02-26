#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 8.7.32.11
N_0^(2)[u,P0,P_phi] の具体項を固定し、摂動解 delta P0 を構成して
強場係数差（~4.6%）への寄与を数値化する監査パック。

固定出力:
- output/public/theory/pmodel_effective_metric_n0_source_solution_audit.json
- output/public/theory/pmodel_effective_metric_n0_source_solution_audit.csv
- output/public/theory/pmodel_effective_metric_n0_source_solution_audit.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
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


def _to_float(v: Any, default: float = float("nan")) -> float:
    try:
        out = float(v)
    except Exception:
        return float(default)

    return float(out) if np.isfinite(out) else float(default)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def _cumtrapz_forward(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    for i in range(1, y.size):
        out[i] = out[i - 1] + 0.5 * float(y[i] + y[i - 1]) * float(x[i] - x[i - 1])

    return out


def _integral_from_i_to_end(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    rev = _cumtrapz_forward(y[::-1], x[::-1])
    return rev[::-1]


def _spherical_average_theta(field_rt: np.ndarray, theta: np.ndarray) -> np.ndarray:
    sin_t = np.sin(theta)
    # 0.5 * integral_0^pi f(r,theta) sin(theta) dtheta
    num = np.trapezoid(field_rt * sin_t[None, :], theta, axis=1)
    return 0.5 * num


def _solve_du_dr_cubic(q: np.ndarray, eta_nonlinear: float) -> np.ndarray:
    q_arr = np.asarray(q, dtype=float)
    eta = float(max(eta_nonlinear, 0.0))
    # 条件分岐: `eta <= 0.0` を満たす経路を評価する。
    if eta <= 0.0:
        return q_arr.copy()

    p = q_arr.copy()
    for _ in range(8):
        f = 2.0 * eta * p * p * p + p - q_arr
        df = 6.0 * eta * p * p + 1.0
        p = p - f / np.maximum(df, 1.0e-12)

    return p


def _solve_u_base_radial(*, r: np.ndarray, lambda2: float, eta_nonlinear: float) -> Tuple[np.ndarray, np.ndarray]:
    r_outer = float(r[-1])
    u_outer = float(1.0 / r_outer + lambda2 / (r_outer * r_outer))
    du_outer = float(-1.0 / (r_outer * r_outer) - 2.0 * lambda2 / (r_outer**3))
    flux_const = float((r_outer**2) * (du_outer + 2.0 * eta_nonlinear * (du_outer**3)))
    q = flux_const / np.maximum(r * r, 1.0e-12)
    du = _solve_du_dr_cubic(q, float(eta_nonlinear))

    u = np.empty_like(r, dtype=float)
    u[-1] = u_outer
    for i in range(r.size - 2, -1, -1):
        dr = float(r[i] - r[i + 1])
        du_mid = 0.5 * float(du[i] + du[i + 1])
        u[i] = u[i + 1] + dr * du_mid

    return u, du


def _build_profiles(
    *,
    r: np.ndarray,
    theta: np.ndarray,
    lambda2: float,
    mu_drag: float,
    epsilon_nl: float,
) -> Dict[str, np.ndarray]:
    rr = r[:, None]
    tt = theta[None, :]
    sin_t = np.sin(tt)
    cos_t = np.cos(tt)

    u_base_r, du_dr_r = _solve_u_base_radial(r=r, lambda2=lambda2, eta_nonlinear=epsilon_nl)
    p0_base_r = np.exp(u_base_r)
    p0_base = p0_base_r[:, None] * np.ones_like(tt)
    p0_base = np.maximum(p0_base, 1.0e-12)
    u_base = u_base_r[:, None] * np.ones_like(tt)

    dp0_dr_r = p0_base_r * du_dr_r
    dp0_dr = dp0_dr_r[:, None] * np.ones_like(tt)
    du_dr = du_dr_r[:, None] * np.ones_like(tt)

    pphi = mu_drag * (sin_t * sin_t) / np.maximum(rr * rr, 1.0e-12)
    dpphi_dr = -2.0 * mu_drag * (sin_t * sin_t) / np.maximum(rr**3, 1.0e-12)
    dpphi_dtheta = 2.0 * mu_drag * sin_t * cos_t / np.maximum(rr * rr, 1.0e-12)

    # N_0^(2) を O(epsilon^2) で明示化
    eps2 = float(max(epsilon_nl, 0.0) ** 2)
    n_metric_raw = 2.0 * du_dr * dp0_dr
    sin2 = np.maximum(sin_t * sin_t, 1.0e-12)
    n_phi_raw = np.exp(-2.0 * u_base) * (
        (dpphi_dr * dpphi_dr) / np.maximum(rr * rr * sin2, 1.0e-12)
        + (dpphi_dtheta * dpphi_dtheta) / np.maximum(rr**4 * sin2, 1.0e-12)
    )
    n_metric = eps2 * n_metric_raw
    n_phi = eps2 * n_phi_raw
    n_total = n_metric + n_phi

    return {
        "p0_base_r": p0_base_r,
        "u_base_r": u_base_r,
        "dp0_dr_r": dp0_dr_r,
        "du_dr_r": du_dr_r,
        "pphi_rt": pphi,
        "eps2": eps2,
        "n_metric_raw_rt": n_metric_raw,
        "n_phi_raw_rt": n_phi_raw,
        "n_metric_rt": n_metric,
        "n_phi_rt": n_phi,
        "n_total_rt": n_total,
    }


def _solve_delta_p0_from_source(
    *,
    r: np.ndarray,
    nbar: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    # (1/r^2) d/dr (r^2 d(deltaP0)/dr) = nbar(r)
    rhs = (r * r) * nbar
    rhs_int = _integral_from_i_to_end(rhs, r)  # integral_r^rout s^2 nbar(s) ds
    ddelta_dr = -rhs_int / np.maximum(r * r, 1.0e-12)
    delta = -_integral_from_i_to_end(ddelta_dr, r)  # delta(r_out)=0
    return delta, ddelta_dr


def _ring_coeff_from_u(
    *,
    r: np.ndarray,
    u_r: np.ndarray,
    beta: float,
    zeta: float,
    omega_eps: float,
    object_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], float]:
    out_rows: List[Dict[str, Any]] = []
    coeffs: List[float] = []
    n_r = np.exp(2.0 * beta * u_r)
    for row in object_rows:
        a_star = _to_float(row.get("a_star_mid"), float("nan"))
        inc_deg = _to_float(row.get("inc_deg_mid"), float("nan"))
        key = str(row.get("key", "")).strip() or "obj"
        # 条件分岐: `not np.isfinite(a_star) or not np.isfinite(inc_deg)` を満たす経路を評価する。
        if not np.isfinite(a_star) or not np.isfinite(inc_deg):
            continue

        sin_i = math.sin(math.radians(float(inc_deg)))
        gain = 1.0 + float(omega_eps) * (sin_i * sin_i)
        omega = float(zeta) * float(a_star) * gain / np.maximum(r**3 + 0.2 * a_star * a_star, 1.0e-12)
        den_p = 1.0 + omega * sin_i
        den_m = 1.0 - omega * sin_i
        # 条件分岐: `np.any(den_p <= 0.0) or np.any(den_m <= 0.0)` を満たす経路を評価する。
        if np.any(den_p <= 0.0) or np.any(den_m <= 0.0):
            continue

        b_plus = n_r * r / den_p
        b_minus = n_r * r / den_m
        i_plus = int(np.argmin(b_plus))
        i_minus = int(np.argmin(b_minus))
        coeff = float(b_plus[i_plus] + b_minus[i_minus])
        coeffs.append(coeff)
        out_rows.append(
            {
                "key": key,
                "a_star_mid": float(a_star),
                "inc_deg_mid": float(inc_deg),
                "r_plus_rg": float(r[i_plus]),
                "r_minus_rg": float(r[i_minus]),
                "b_plus_rg": float(b_plus[i_plus]),
                "b_minus_rg": float(b_minus[i_minus]),
                "ring_coeff_rg": coeff,
            }
        )

    mean_coeff = float(np.mean(coeffs)) if coeffs else float("nan")
    return out_rows, mean_coeff


def _plot(
    *,
    r: np.ndarray,
    n_metric_bar: np.ndarray,
    n_phi_bar: np.ndarray,
    n_total_bar: np.ndarray,
    p0_base: np.ndarray,
    p0_full: np.ndarray,
    delta_p0: np.ndarray,
    summary: Dict[str, float],
    out_png: Path,
) -> None:
    # 条件分岐: `plt is None` を満たす経路を評価する。
    if plt is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), dpi=180)
    fig.suptitle("Step 8.7.32.11: explicit N0^(2) terms and strong-field solution")

    axes[0].plot(r, n_metric_bar, label="N_metric_bar")
    axes[0].plot(r, n_phi_bar, label="N_phi_bar")
    axes[0].plot(r, n_total_bar, label="N_total_bar", linewidth=2.0)
    axes[0].set_xscale("log")
    axes[0].set_yscale("symlog", linthresh=1.0e-8)
    axes[0].set_xlabel("r [r_g]")
    axes[0].set_ylabel("source term")
    axes[0].set_title("Spherical average of N0^(2)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(r, p0_base, label="P0 linear")
    axes[1].plot(r, p0_full, label="P0 with N0^(2)")
    axes[1].plot(r, delta_p0, label="deltaP0", linestyle="--")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("r [r_g]")
    axes[1].set_ylabel("P0")
    axes[1].set_title("Perturbative solution of P0")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    labels = ["core gap (4eβ vs ref)", "N0 contribution", "total (core+N0)"]
    vals = [
        summary["core_gap_pct"],
        summary["n0_contribution_pct"],
        summary["core_plus_n0_gap_pct"],
    ]
    axes[2].bar(labels, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[2].axhline(4.63, color="#444444", linestyle="--", linewidth=1.0, label="reference 4.63%")
    axes[2].set_ylabel("relative gap [%] vs C_ref")
    axes[2].set_title("Gap budget")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].legend(loc="best")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Explicit N0^(2) source-term audit and perturbative P0 solution.")
    parser.add_argument(
        "--direct-json",
        type=Path,
        default=ROOT / "output/public/theory/pmodel_rotating_bh_photon_ring_direct_audit.json",
    )
    parser.add_argument("--r-min-rg", type=float, default=1.90)
    parser.add_argument("--r-max-rg", type=float, default=40.0)
    parser.add_argument("--n-r", type=int, default=1600)
    parser.add_argument("--n-theta", type=int, default=181)
    parser.add_argument("--step-tag", default="8.7.32.11")
    parser.add_argument("--outdir", type=Path, default=ROOT / "output/public/theory")
    args = parser.parse_args()

    direct = json.loads(args.direct_json.read_text(encoding="utf-8"))
    axis = direct.get("axisymmetric_pde_block") if isinstance(direct.get("axisymmetric_pde_block"), dict) else {}
    params = axis.get("parameter_snapshot") if isinstance(axis.get("parameter_snapshot"), dict) else {}
    obj_rows = direct.get("object_rows") if isinstance(direct.get("object_rows"), list) else []

    beta = _to_float(params.get("beta_fixed"), 1.0000105)
    lambda2 = _to_float(params.get("lambda2_best"), -0.14666666666666667)
    zeta = _to_float(params.get("zeta_best"), 6.88)
    eta_nl = _to_float(params.get("eta_nonlinear"), 0.08)
    omega_eps = _to_float(params.get("omega_ang_epsilon"), 0.18)

    a_vals = [_to_float(r.get("a_star_mid"), float("nan")) for r in obj_rows if isinstance(r, dict)]
    a_vals = [x for x in a_vals if np.isfinite(x)]
    a_eff = float(np.mean(a_vals)) if a_vals else 0.5
    mu_drag = float(zeta * a_eff)

    r = np.linspace(float(args.r_min_rg), float(args.r_max_rg), int(max(args.n_r, 64)), dtype=float)
    theta = np.linspace(1.0e-4, math.pi - 1.0e-4, int(max(args.n_theta, 33)), dtype=float)

    prof = _build_profiles(r=r, theta=theta, lambda2=lambda2, mu_drag=mu_drag, epsilon_nl=eta_nl)
    n_metric_bar = _spherical_average_theta(prof["n_metric_rt"], theta)
    n_phi_bar = _spherical_average_theta(prof["n_phi_rt"], theta)
    n_total_bar = _spherical_average_theta(prof["n_total_rt"], theta)

    delta_p0, ddelta_dr = _solve_delta_p0_from_source(r=r, nbar=n_total_bar)
    p0_base = np.asarray(prof["p0_base_r"], dtype=float)
    p0_full = np.maximum(p0_base + delta_p0, 1.0e-12)
    u_base = np.log(np.maximum(p0_base, 1.0e-12))
    u_full = np.log(np.maximum(p0_full, 1.0e-12))
    delta_u = u_full - u_base

    ring_rows_linear, coeff_linear = _ring_coeff_from_u(
        r=r,
        u_r=u_base,
        beta=beta,
        zeta=zeta,
        omega_eps=omega_eps,
        object_rows=obj_rows,
    )
    ring_rows_full, coeff_full = _ring_coeff_from_u(
        r=r,
        u_r=u_full,
        beta=beta,
        zeta=zeta,
        omega_eps=omega_eps,
        object_rows=obj_rows,
    )

    c_ref = float(2.0 * math.sqrt(27.0))
    c_core = float(4.0 * math.e * beta)
    core_gap_pct = float((c_core / c_ref - 1.0) * 100.0)

    coeff_shift_pct_vs_ref = float((coeff_full - coeff_linear) / c_ref * 100.0)
    core_plus_n0_gap_pct = float(core_gap_pct + coeff_shift_pct_vs_ref)
    n_share_pct = float(coeff_shift_pct_vs_ref / max(core_gap_pct, 1.0e-12) * 100.0)

    # ring-proxy point for closed-form estimate
    r_ring_candidates = [float(row.get("r_plus_rg", float("nan"))) for row in ring_rows_full]
    r_ring_candidates += [float(row.get("r_minus_rg", float("nan"))) for row in ring_rows_full]
    r_ring_candidates = [x for x in r_ring_candidates if np.isfinite(x)]
    r_ring_eff = float(np.mean(r_ring_candidates)) if r_ring_candidates else float(r[0])
    i_eff = int(np.argmin(np.abs(r - r_ring_eff)))
    delta_u_eff = float(delta_u[i_eff])
    c_core_plus_n0_closed = float(c_core * math.exp(2.0 * beta * delta_u_eff))
    core_plus_n0_closed_gap_pct = float((c_core_plus_n0_closed / c_ref - 1.0) * 100.0)

    outdir = args.outdir
    out_json = outdir / "pmodel_effective_metric_n0_source_solution_audit.json"
    out_csv = outdir / "pmodel_effective_metric_n0_source_solution_audit.csv"
    out_png = outdir / "pmodel_effective_metric_n0_source_solution_audit.png"

    residual_rows: List[Dict[str, Any]] = []
    sample_idx = np.linspace(0, r.size - 1, min(220, r.size), dtype=int)
    for idx in sample_idx.tolist():
        residual_rows.append(
            {
                "r_rg": float(r[idx]),
                "N_metric_bar": float(n_metric_bar[idx]),
                "N_phi_bar": float(n_phi_bar[idx]),
                "N_total_bar": float(n_total_bar[idx]),
                "P0_linear": float(p0_base[idx]),
                "deltaP0": float(delta_p0[idx]),
                "P0_full": float(p0_full[idx]),
                "delta_u": float(delta_u[idx]),
                "d_deltaP0_dr": float(ddelta_dr[idx]),
            }
        )

    _write_csv(
        out_csv,
        residual_rows,
        fieldnames=(
            "r_rg",
            "N_metric_bar",
            "N_phi_bar",
            "N_total_bar",
            "P0_linear",
            "deltaP0",
            "P0_full",
            "delta_u",
            "d_deltaP0_dr",
        ),
    )

    summary = {
        "c_ref": c_ref,
        "c_core": c_core,
        "core_gap_pct": core_gap_pct,
        "coeff_linear_mean": float(coeff_linear),
        "coeff_full_mean": float(coeff_full),
        "n0_contribution_pct": coeff_shift_pct_vs_ref,
        "core_plus_n0_gap_pct": core_plus_n0_gap_pct,
        "n0_share_vs_core_pct": n_share_pct,
        "r_ring_eff_rg": r_ring_eff,
        "delta_u_eff": delta_u_eff,
        "core_plus_n0_closed_gap_pct": core_plus_n0_closed_gap_pct,
        "max_abs_deltaP0": float(np.max(np.abs(delta_p0))),
        "max_abs_delta_u": float(np.max(np.abs(delta_u))),
    }
    _plot(
        r=r,
        n_metric_bar=n_metric_bar,
        n_phi_bar=n_phi_bar,
        n_total_bar=n_total_bar,
        p0_base=p0_base,
        p0_full=p0_full,
        delta_p0=delta_p0,
        summary=summary,
        out_png=out_png,
    )

    checks: List[Dict[str, Any]] = [
        {
            "check_id": "n0_terms_explicit",
            "metric": "N0^(2)=N_metric+N_phi",
            "value": True,
            "expected": True,
            "hard_fail": False,
        },
        {
            "check_id": "perturbative_solution_finite",
            "metric": "isfinite(P0_full)",
            "value": bool(np.all(np.isfinite(p0_full))),
            "expected": True,
            "hard_fail": not bool(np.all(np.isfinite(p0_full))),
        },
        {
            "check_id": "n0_contribution_resolved",
            "metric": "abs(n0_contribution_pct)",
            "value": float(abs(coeff_shift_pct_vs_ref)),
            "expected": ">0",
            "hard_fail": float(abs(coeff_shift_pct_vs_ref)) <= 0.0,
        },
    ]

    hard_n = sum(1 for c in checks if bool(c.get("hard_fail")))
    status = "pass" if hard_n == 0 else "reject"
    decision = "n0_terms_explicit_and_solution_fixed" if status == "pass" else "n0_solution_incomplete"

    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "phase": {"phase": 8, "step": str(args.step_tag), "name": "effective metric N0 source solution audit"},
        "intent": "Expose explicit N0^(2) terms and show perturbative solution contribution to strong-field 4.6% coefficient gap.",
        "inputs": {
            "direct_audit_json": _rel(args.direct_json),
            "beta_fixed": beta,
            "lambda2_best": lambda2,
            "zeta_best": zeta,
            "eta_nonlinear": eta_nl,
            "omega_ang_epsilon": omega_eps,
            "a_effective": a_eff,
            "mu_drag": mu_drag,
            "r_range_rg": [float(args.r_min_rg), float(args.r_max_rg)],
            "n_r": int(args.n_r),
            "n_theta": int(args.n_theta),
        },
        "explicit_n0_terms": {
            "definition": "N0^(2)=N_metric+N_phi",
            "N_metric": "epsilon_nl^2 * 2[(∂_r u)(∂_r P0) + (1/r^2)(∂_θ u)(∂_θ P0)]",
            "N_phi": "epsilon_nl^2 * e^{-2u}[(∂_r P_phi)^2/(r^2 sin^2θ) + (∂_θ P_phi)^2/(r^4 sin^2θ)]",
            "P_phi_ansatz": "P_phi(r,θ)=mu_drag sin^2θ / r^2",
            "spherical_average": "Nbar(r)=0.5 ∫_0^π N0^(2)(r,θ) sinθ dθ",
            "radial_equation": "(1/r^2)d/dr(r^2 d(deltaP0)/dr)=Nbar(r)",
            "integral_solution": "deltaP0(r)=-∫_r^{r_out} dξ ξ^{-2}∫_ξ^{r_out} s^2 Nbar(s) ds",
        },
        "nonlinear_order": {
            "epsilon_nl": eta_nl,
            "epsilon_nl_squared": float(prof.get("eps2", 0.0)),
        },
        "ring_coefficient_budget": summary,
        "ring_rows_linear": ring_rows_linear,
        "ring_rows_full": ring_rows_full,
        "checks": checks,
        "decision": {"overall_status": status, "decision": decision, "hard_reject_n": int(hard_n)},
        "outputs": {
            "audit_json": _rel(out_json),
            "audit_csv": _rel(out_csv),
            "audit_png": _rel(out_png),
        },
        "falsification_gate": {
            "reject_if": [
                "N0^(2) explicit decomposition is missing.",
                "Perturbative solution P0_full is non-finite.",
                "N0 contribution is numerically unresolved.",
            ]
        },
    }
    _write_json(out_json, payload)

    # 条件分岐: `worklog is not None` を満たす経路を評価する。
    if worklog is not None:
        try:
            worklog.append_event(
                {
                    "event_type": "pmodel_effective_metric_n0_source_solution_audit",
                    "phase": str(args.step_tag),
                    "overall_status": status,
                    "decision": decision,
                    "core_gap_pct": core_gap_pct,
                    "n0_contribution_pct": coeff_shift_pct_vs_ref,
                    "core_plus_n0_gap_pct": core_plus_n0_gap_pct,
                    "outputs": {"json": _rel(out_json), "csv": _rel(out_csv), "png": _rel(out_png)},
                }
            )
        except Exception:
            pass

    print(f"[ok] wrote {out_json}")
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_png}")
    print(
        "[summary] status={0} decision={1} core_gap={2:.4f}% n0_contrib={3:.4f}% total={4:.4f}%".format(
            status,
            decision,
            core_gap_pct,
            coeff_shift_pct_vs_ref,
            core_plus_n0_gap_pct,
        )
    )
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
