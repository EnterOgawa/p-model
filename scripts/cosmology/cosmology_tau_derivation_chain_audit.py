#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_tau_derivation_chain_audit.py

Step 8.7.25.19:
`tau_origin_mode=derived_from_pmu_kernel` と `xi_mode=derived` の
本文追跡可能性を固定するため、導出鎖を機械監査する。

固定出力:
  - output/public/cosmology/cosmology_tau_derivation_chain_audit.json
  - output/public/cosmology/cosmology_tau_derivation_chain_audit.csv
  - output/public/cosmology/cosmology_tau_derivation_chain_audit.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.summary import worklog  # type: ignore  # noqa: E402
except Exception:
    worklog = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


# 関数: `_utc_now` の入出力契約と処理意図を定義する。

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


# 関数: `_f` の入出力契約と処理意図を定義する。

def _f(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)

    # 条件分岐: `not math.isfinite(out)` を満たす経路を評価する。

    if not math.isfinite(out):
        return float(default)

    return float(out)


# 関数: `_safe_rel` の入出力契約と処理意図を定義する。

def _safe_rel(a: float, b: float) -> float:
    denom = max(abs(b), 1.0e-12)
    return float(abs(a - b) / denom)


# 関数: `_render_png` の入出力契約と処理意図を定義する。

def _render_png(
    path: Path,
    *,
    tau_free: float,
    tau_int: float,
    tau_damp: float,
    tau_eff: float,
    tau_eff_harm: float,
    tau_from_kernel: float,
    xi_derived: float,
    xi_from_taus: float,
    summary_lines: List[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `plt is None` を満たす経路を評価する。
    if plt is None:
        path.write_bytes(b"")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=150)

    labels = ["tau_free", "tau_int", "tau_damp", "tau_eff", "tau_eff_harm", "tau_from_kernel"]
    values = [tau_free, tau_int, tau_damp, tau_eff, tau_eff_harm, tau_from_kernel]
    axes[0].bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"])
    axes[0].set_ylabel("timescale [Gyr]")
    axes[0].set_title("Step 8.7.25.19: tau derivation chain")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, axis="y", alpha=0.25)

    rel_errs = [
        _safe_rel(tau_eff_harm, tau_eff),
        _safe_rel(tau_from_kernel, tau_eff),
        _safe_rel(xi_from_taus, xi_derived),
    ]
    err_labels = ["rel(tau_eff_harm,tau_eff)", "rel(tau_kernel_chain,tau_eff)", "rel(xi_chain,xi_derived)"]
    axes[1].bar(err_labels, rel_errs, color=["#2ca02c", "#2ca02c", "#2ca02c"])
    axes[1].axhline(0.20, color="#d62728", linestyle="--", linewidth=1.0, label="tau reconstruction gate")
    axes[1].set_ylabel("relative error")
    axes[1].set_title("closure residuals")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.suptitle("Bullet tau/xi derivation-chain audit")
    fig.text(0.01, -0.02, "\n".join(summary_lines), fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    parser = argparse.ArgumentParser(description="Audit tau/xi derivation chain from Bullet separation derivation output.")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=ROOT / "output/public/cosmology/cosmology_cluster_collision_pmu_jmu_separation_derivation.json",
        help="Input derivation JSON.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "output/public/cosmology/cosmology_tau_derivation_chain_audit.json",
        help="Output audit JSON.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "output/public/cosmology/cosmology_tau_derivation_chain_audit.csv",
        help="Output audit CSV.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=ROOT / "output/public/cosmology/cosmology_tau_derivation_chain_audit.png",
        help="Output audit PNG.",
    )
    parser.add_argument("--step-tag", default="8.7.25.19")
    parser.add_argument("--tau-recon-threshold", type=float, default=0.20)
    parser.add_argument("--chain-rel-threshold", type=float, default=0.02)
    parser.add_argument("--epsilon-slow-threshold", type=float, default=0.25)
    args = parser.parse_args()

    # 条件分岐: `not args.input_json.exists()` を満たす経路を評価する。
    if not args.input_json.exists():
        raise FileNotFoundError(f"missing input JSON: {args.input_json}")

    src = json.loads(args.input_json.read_text(encoding="utf-8"))
    tau_block = src.get("tau_origin_block", {})
    fit_block = src.get("fit", {})

    mode = str(tau_block.get("mode", ""))
    closure_pass = bool(tau_block.get("tau_origin_closure_pass", False))
    manual_injection = bool(tau_block.get("tau_manual_injection", True))
    ad_hoc_count = int(tau_block.get("ad_hoc_parameter_count", -1))
    kernel_mode = str(tau_block.get("kernel_mode", ""))
    tau_recon_err = _f(tau_block.get("tau_reconstruction_rel_error"), float("nan"))
    xi_derived = _f(tau_block.get("derived_xi"), float("nan"))

    comps = tau_block.get("derived_components_gyr", {})
    tau_free = _f(comps.get("tau_free"), float("nan"))
    tau_int = _f(comps.get("tau_int"), float("nan"))
    tau_damp = _f(comps.get("tau_damp"), float("nan"))
    tau_eff = _f(comps.get("tau_eff"), float("nan"))
    tau_eff_harm = _f(comps.get("tau_eff_harmonic"), float("nan"))
    tau_kernel = _f(comps.get("tau_kernel"), float("nan"))
    eta = _f(tau_block.get("derived_eta_weight"), float("nan"))

    xi_mode = str(fit_block.get("xi_mode", ""))
    fit_xi = _f(fit_block.get("xi_coupling_best"), float("nan"))

    tau_eff_harm_calc = float(1.0 / max((1.0 / tau_free) + (1.0 / tau_int) + (1.0 / tau_damp), 1.0e-12))
    eta_calc = float(tau_free / max(tau_free + tau_int, 1.0e-12))
    tau_eff_from_kernel = float(tau_free + eta * tau_kernel)
    tau_eff_from_kernel_calc_eta = float(tau_free + eta_calc * tau_kernel)
    xi_from_taus = float((tau_int + tau_damp) / max(tau_eff, 1.0e-12))

    rel_tau_harm = _safe_rel(tau_eff_harm_calc, tau_eff_harm)
    rel_tau_kernel = _safe_rel(tau_eff_from_kernel, tau_eff)
    rel_tau_kernel_calc_eta = _safe_rel(tau_eff_from_kernel_calc_eta, tau_eff)
    rel_xi_chain = _safe_rel(xi_from_taus, xi_derived)
    rel_fit_vs_derived_xi = _safe_rel(fit_xi, xi_derived)
    tau_slow = float(min(tau_int, tau_damp))
    epsilon_slow = float(tau_free / max(tau_slow, 1.0e-12))
    diffusion_coefficient_norm = float(tau_free)  # c_w-normalized units (c_w=1)

    checks: List[Dict[str, Any]] = [
        {
            "check_id": "tau_chain::mode",
            "metric": "tau_origin_mode",
            "value": mode,
            "expected": "derived_from_pmu_kernel",
            "hard_fail": mode != "derived_from_pmu_kernel",
        },
        {
            "check_id": "tau_chain::closure_pass",
            "metric": "tau_origin_closure_pass",
            "value": closure_pass,
            "expected": True,
            "hard_fail": (not closure_pass),
        },
        {
            "check_id": "tau_chain::manual_injection",
            "metric": "tau_manual_injection",
            "value": manual_injection,
            "expected": False,
            "hard_fail": manual_injection,
        },
        {
            "check_id": "tau_chain::ad_hoc_count",
            "metric": "ad_hoc_parameter_count",
            "value": ad_hoc_count,
            "expected": 0,
            "hard_fail": ad_hoc_count != 0,
        },
        {
            "check_id": "tau_chain::tau_reconstruction",
            "metric": "tau_reconstruction_rel_error",
            "value": tau_recon_err,
            "expected": f"<= {args.tau_recon_threshold:.3f}",
            "hard_fail": (not math.isfinite(tau_recon_err)) or (tau_recon_err > float(args.tau_recon_threshold)),
        },
        {
            "check_id": "tau_chain::kernel_mode",
            "metric": "retarded_kernel_mode",
            "value": kernel_mode,
            "expected": "double_exp_derived",
            "hard_fail": kernel_mode != "double_exp_derived",
        },
        {
            "check_id": "tau_chain::xi_mode",
            "metric": "xi_mode",
            "value": xi_mode,
            "expected": "derived",
            "hard_fail": xi_mode != "derived",
        },
        {
            "check_id": "tau_chain::tau_harmonic_consistency",
            "metric": "rel(tau_eff_harm_calc, tau_eff_harm_stored)",
            "value": rel_tau_harm,
            "expected": f"<= {args.chain_rel_threshold:.3f}",
            "hard_fail": rel_tau_harm > float(args.chain_rel_threshold),
        },
        {
            "check_id": "tau_chain::tau_kernel_consistency",
            "metric": "rel(tau_free + eta*tau_kernel, tau_eff)",
            "value": rel_tau_kernel,
            "expected": f"<= {args.chain_rel_threshold:.3f}",
            "hard_fail": rel_tau_kernel > float(args.chain_rel_threshold),
        },
        {
            "check_id": "tau_chain::xi_consistency",
            "metric": "rel((tau_int+tau_damp)/tau_eff, xi_derived)",
            "value": rel_xi_chain,
            "expected": f"<= {args.chain_rel_threshold:.3f}",
            "hard_fail": rel_xi_chain > float(args.chain_rel_threshold),
        },
        {
            "check_id": "tau_chain::fit_vs_derived_xi",
            "metric": "rel(xi_fit, xi_derived)",
            "value": rel_fit_vs_derived_xi,
            "expected": f"<= {args.chain_rel_threshold:.3f}",
            "hard_fail": rel_fit_vs_derived_xi > float(args.chain_rel_threshold),
        },
        {
            "check_id": "tau_chain::wave_to_relax_slow_manifold",
            "metric": "epsilon_slow=tau_free/min(tau_int,tau_damp)",
            "value": epsilon_slow,
            "expected": f"<= {args.epsilon_slow_threshold:.3f}",
            "hard_fail": (not math.isfinite(epsilon_slow)) or (epsilon_slow > float(args.epsilon_slow_threshold)),
        },
    ]

    hard_fail_n = sum(1 for c in checks if bool(c.get("hard_fail")))
    # 条件分岐: `hard_fail_n > 0` を満たす経路を評価する。
    if hard_fail_n > 0:
        status = "reject"
        decision = "tau_derivation_chain_incomplete"
    # 条件分岐: 前段条件が不成立で、`tau_recon_err > 0.10` を追加評価する。
    elif tau_recon_err > 0.10:
        status = "watch"
        decision = "tau_derivation_chain_fixed_but_reconstruction_margin_watch"
    else:
        status = "pass"
        decision = "tau_derivation_chain_fixed"

    rows: List[Dict[str, Any]] = [
        {"component": "tau_free_gyr", "value": tau_free},
        {"component": "tau_int_gyr", "value": tau_int},
        {"component": "tau_damp_gyr", "value": tau_damp},
        {"component": "tau_eff_gyr", "value": tau_eff},
        {"component": "tau_eff_harmonic_gyr", "value": tau_eff_harm},
        {"component": "tau_eff_harmonic_calc_gyr", "value": tau_eff_harm_calc},
        {"component": "tau_kernel_gyr", "value": tau_kernel},
        {"component": "eta_stored", "value": eta},
        {"component": "eta_calc", "value": eta_calc},
        {"component": "tau_eff_from_kernel_gyr", "value": tau_eff_from_kernel},
        {"component": "tau_eff_from_kernel_calc_eta_gyr", "value": tau_eff_from_kernel_calc_eta},
        {"component": "xi_derived", "value": xi_derived},
        {"component": "xi_from_taus", "value": xi_from_taus},
        {"component": "xi_fit", "value": fit_xi},
        {"component": "tau_reconstruction_rel_error", "value": tau_recon_err},
        {"component": "rel_tau_harm", "value": rel_tau_harm},
        {"component": "rel_tau_kernel", "value": rel_tau_kernel},
        {"component": "rel_xi_chain", "value": rel_xi_chain},
        {"component": "rel_fit_vs_derived_xi", "value": rel_fit_vs_derived_xi},
        {"component": "tau_slow_gyr", "value": tau_slow},
        {"component": "epsilon_slow", "value": epsilon_slow},
        {"component": "D_norm_cw1_gyr", "value": diffusion_coefficient_norm},
    ]
    _write_csv(args.out_csv, rows, fieldnames=("component", "value"))

    summary_lines = [
        f"status={status}, decision={decision}, hard_fail_n={hard_fail_n}",
        f"tau_mode={mode}, kernel_mode={kernel_mode}, xi_mode={xi_mode}",
        f"tau_reconstruction_rel_error={tau_recon_err:.6f} (gate<={args.tau_recon_threshold:.3f})",
        f"epsilon_slow={epsilon_slow:.6f} (gate<={args.epsilon_slow_threshold:.3f})",
        f"rel_tau_kernel={rel_tau_kernel:.6f}, rel_xi_chain={rel_xi_chain:.6f}, rel_fit_xi={rel_fit_vs_derived_xi:.6f}",
    ]

    _render_png(
        args.out_png,
        tau_free=tau_free,
        tau_int=tau_int,
        tau_damp=tau_damp,
        tau_eff=tau_eff,
        tau_eff_harm=tau_eff_harm,
        tau_from_kernel=tau_eff_from_kernel,
        xi_derived=xi_derived,
        xi_from_taus=xi_from_taus,
        summary_lines=summary_lines,
    )

    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "phase": {"phase": 8, "step": str(args.step_tag), "name": "tau derivation chain audit"},
        "intent": (
            "Lock the equation chain from P_mu/J^mu kernel-derived tau components "
            "to first-moment response equation and xi(derived), without ad-hoc injection."
        ),
        "input": {"derivation_json": _rel(args.input_json)},
        "chain": {
            "equations": {
                "wave_projected": "(1/c_w^2)∂_tt u_L - ∂_xx u_L = S_L",
                "telegraph_with_memory_friction": "tau_free ∂_tt u_L + ∂_t u_L = D ∂_xx u_L + c_w^2 S_L, D=c_w^2 tau_free",
                "slow_manifold": "epsilon_slow=tau_free/min(tau_int,tau_damp) << 1 => ∂_t u_L = D ∂_xx u_L + c_w^2 S_L",
                "coarse_grained_balance": "∂_t P_L + ∂_x q = -(P_L-J^0)/tau_free",
                "flux_closure": "q=-D∂_xP_L-ξ[(1-η)J^x+ηJ^x(t-Δt)]",
                "tau_eff_harmonic": "1 / (1/tau_free + 1/tau_int + 1/tau_damp)",
                "eta": "tau_free / (tau_free + tau_int)",
                "tau_eff_kernel_first_moment": "tau_eff = tau_free + eta * tau_kernel",
                "xi_derived": "xi = (tau_int + tau_damp) / tau_eff",
            },
            "stored": {
                "tau_origin_mode": mode,
                "kernel_mode": kernel_mode,
                "xi_mode": xi_mode,
                "tau_free_gyr": tau_free,
                "tau_int_gyr": tau_int,
                "tau_damp_gyr": tau_damp,
                "tau_eff_gyr": tau_eff,
                "tau_eff_harmonic_gyr": tau_eff_harm,
                "tau_kernel_gyr": tau_kernel,
                "eta": eta,
                "xi_derived": xi_derived,
                "xi_fit": fit_xi,
                "tau_slow_gyr": tau_slow,
                "epsilon_slow": epsilon_slow,
                "D_norm_cw1_gyr": diffusion_coefficient_norm,
                "tau_reconstruction_rel_error": tau_recon_err,
                "tau_origin_closure_pass": closure_pass,
                "tau_manual_injection": manual_injection,
                "ad_hoc_parameter_count": ad_hoc_count,
            },
            "recomputed": {
                "tau_eff_harmonic_gyr": tau_eff_harm_calc,
                "eta": eta_calc,
                "tau_eff_from_kernel_gyr": tau_eff_from_kernel,
                "tau_eff_from_kernel_calc_eta_gyr": tau_eff_from_kernel_calc_eta,
                "xi_from_taus": xi_from_taus,
            },
            "relative_error": {
                "rel_tau_harm": rel_tau_harm,
                "rel_tau_kernel": rel_tau_kernel,
                "rel_tau_kernel_calc_eta": rel_tau_kernel_calc_eta,
                "rel_xi_chain": rel_xi_chain,
                "rel_fit_vs_derived_xi": rel_fit_vs_derived_xi,
            },
        },
        "checks": checks,
        "decision": {
            "overall_status": status,
            "decision": decision,
            "hard_reject_n": hard_fail_n,
            "tau_recon_threshold": float(args.tau_recon_threshold),
            "chain_rel_threshold": float(args.chain_rel_threshold),
            "epsilon_slow_threshold": float(args.epsilon_slow_threshold),
            "next_action": (
                "update_manuscript_chain_equations_and_references"
                if hard_fail_n == 0
                else "fix_derivation_chain_before_manuscript_lock"
            ),
        },
        "outputs": {
            "audit_json": _rel(args.out_json),
            "audit_csv": _rel(args.out_csv),
            "audit_png": _rel(args.out_png),
        },
        "falsification_gate": {
            "reject_if": [
                "tau_origin_mode != derived_from_pmu_kernel",
                "tau_origin_closure_pass == false",
                "tau_manual_injection == true",
                "ad_hoc_parameter_count != 0",
                f"tau_reconstruction_rel_error > {args.tau_recon_threshold:.3f}",
                "kernel_mode != double_exp_derived",
                "xi_mode != derived",
                f"epsilon_slow > {args.epsilon_slow_threshold:.3f}",
                f"rel_chain_errors > {args.chain_rel_threshold:.3f}",
            ]
        },
    }
    _write_json(args.out_json, payload)

    # 条件分岐: `worklog is not None` を満たす経路を評価する。
    if worklog is not None:
        try:
            worklog.append_event(
                {
                    "event_type": "cosmology_tau_derivation_chain_audit",
                    "phase": str(args.step_tag),
                    "overall_status": status,
                    "decision": decision,
                    "hard_reject_n": hard_fail_n,
                    "tau_reconstruction_rel_error": tau_recon_err,
                    "rel_tau_kernel": rel_tau_kernel,
                    "rel_xi_chain": rel_xi_chain,
                    "outputs": {
                        "json": _rel(args.out_json),
                        "csv": _rel(args.out_csv),
                        "png": _rel(args.out_png),
                    },
                }
            )
        except Exception:
            pass

    print(f"[ok] wrote {args.out_json}")
    print(f"[ok] wrote {args.out_csv}")
    print(f"[ok] wrote {args.out_png}")
    print(
        "[summary] status={0} decision={1} hard={2} tauErr={3:.6f} epsSlow={4:.6f} relTauKernel={5:.6f} relXi={6:.6f}".format(
            status,
            decision,
            hard_fail_n,
            tau_recon_err,
            epsilon_slow,
            rel_tau_kernel,
            rel_xi_chain,
        )
    )
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
