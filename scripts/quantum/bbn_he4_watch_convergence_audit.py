from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

import sys

# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _gate_status(*, z_abs: float, hard: float, watch: float) -> str:
    # 条件分岐: `z_abs <= watch` を満たす経路を評価する。
    if z_abs <= watch:
        return "pass"

    # 条件分岐: `z_abs <= hard` を満たす経路を評価する。

    if z_abs <= hard:
        return "watch"

    return "reject"


@dataclass
class _ParamSpec:
    name: str
    mu: float
    sigma: float
    lower: float | None
    group: str


def _freeze_tf(
    *,
    cf0: float,
    cf_alpha: float,
    q_b: float,
    t_b_sec: float,
    t_b_mev: float,
    a_w: float,
    cf_tref_mev: float,
) -> tuple[float, float]:
    n_eff = 5.0 - cf_alpha - (1.0 / q_b)
    # 条件分岐: `n_eff <= 1.0e-8` を満たす経路を評価する。
    if n_eff <= 1.0e-8:
        raise ValueError(f"invalid n_eff={n_eff:.6g}")

    base = (cf0 * (q_b / t_b_sec) * (cf_tref_mev ** (-cf_alpha)) * (t_b_mev ** (-1.0 / q_b))) / a_w
    # 条件分岐: `base <= 0.0` を満たす経路を評価する。
    if base <= 0.0:
        raise ValueError(f"invalid freeze base={base:.6g}")

    tf_mev = float(base ** (1.0 / n_eff))
    return tf_mev, float(n_eff)


def _cf_at_tf(*, tf_mev: float, cf0: float, cf_alpha: float, cf_tref_mev: float) -> float:
    return float(cf0 * (tf_mev / cf_tref_mev) ** cf_alpha)


def _he4_from_tf(*, tf_mev: float, delta_m_mev: float, delta_t_n_sec: float, tau_n_sec: float) -> float:
    np_f = math.exp(-delta_m_mev / tf_mev)
    np_n = np_f * math.exp(-delta_t_n_sec / tau_n_sec)
    return float(2.0 * np_n / (1.0 + np_n))


def _evaluate_state(*, params: dict[str, float], args: argparse.Namespace) -> dict[str, float]:
    tf_mev, n_eff = _freeze_tf(
        cf0=float(params["cf0"]),
        cf_alpha=float(params["cf_alpha"]),
        q_b=float(params["q_b"]),
        t_b_sec=float(params["t_b_sec"]),
        t_b_mev=float(params["t_b_mev"]),
        a_w=float(params["a_w"]),
        cf_tref_mev=float(args.cf_tref_mev),
    )
    cf_tf = _cf_at_tf(
        tf_mev=tf_mev,
        cf0=float(params["cf0"]),
        cf_alpha=float(params["cf_alpha"]),
        cf_tref_mev=float(args.cf_tref_mev),
    )
    y_pred = _he4_from_tf(
        tf_mev=tf_mev,
        delta_m_mev=float(args.delta_m_mev),
        delta_t_n_sec=float(args.delta_t_n_sec),
        tau_n_sec=float(args.tau_n_sec),
    )
    return {"tf_mev": tf_mev, "cf_tf": cf_tf, "y_pred": y_pred, "n_eff": n_eff}


def _numeric_gradient(
    *,
    target: str,
    specs: list[_ParamSpec],
    nominal_params: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, float]:
    grad: dict[str, float] = {}
    for spec in specs:
        mu = float(spec.mu)
        eps = max(abs(mu) * 1.0e-4, 1.0e-7)
        x_lo = mu - eps
        x_hi = mu + eps
        # 条件分岐: `spec.lower is not None and x_lo <= float(spec.lower)` を満たす経路を評価する。
        if spec.lower is not None and x_lo <= float(spec.lower):
            x_lo = max(float(spec.lower) * (1.0 + 1.0e-6), mu)

        p_lo = dict(nominal_params)
        p_hi = dict(nominal_params)
        p_lo[spec.name] = float(x_lo)
        p_hi[spec.name] = float(x_hi)
        try:
            y_hi = _evaluate_state(params=p_hi, args=args)[target]
            # 条件分岐: `x_lo == mu` を満たす経路を評価する。
            if x_lo == mu:
                p_mu = dict(nominal_params)
                p_mu[spec.name] = mu
                y_mu = _evaluate_state(params=p_mu, args=args)[target]
                grad[spec.name] = float((y_hi - y_mu) / (x_hi - mu))
            else:
                y_lo = _evaluate_state(params=p_lo, args=args)[target]
                grad[spec.name] = float((y_hi - y_lo) / (x_hi - x_lo))
        except Exception:
            grad[spec.name] = float("nan")

    return grad


def _build_specs(args: argparse.Namespace) -> list[_ParamSpec]:
    return [
        _ParamSpec(
            name="cf0",
            mu=float(args.cf0),
            sigma=abs(float(args.cf0)) * max(0.0, float(args.cf0_sigma_frac)),
            lower=1.0e-10,
            group="C_F(T)",
        ),
        _ParamSpec(
            name="cf_alpha",
            mu=float(args.cf_alpha),
            sigma=max(0.0, float(args.cf_alpha_sigma)),
            lower=None,
            group="C_F(T)",
        ),
        _ParamSpec(
            name="a_w",
            mu=float(args.a_w),
            sigma=abs(float(args.a_w)) * max(0.0, float(args.a_w_sigma_frac)),
            lower=1.0e-10,
            group="freeze",
        ),
        _ParamSpec(
            name="q_b",
            mu=float(args.q_b),
            sigma=max(0.0, float(args.q_b_sigma)),
            lower=1.0e-6,
            group="freeze",
        ),
        _ParamSpec(
            name="t_b_sec",
            mu=float(args.t_b_sec),
            sigma=abs(float(args.t_b_sec)) * max(0.0, float(args.t_b_sec_sigma_frac)),
            lower=1.0e-12,
            group="freeze",
        ),
        _ParamSpec(
            name="t_b_mev",
            mu=float(args.t_b_mev),
            sigma=abs(float(args.t_b_mev)) * max(0.0, float(args.t_b_mev_sigma_frac)),
            lower=1.0e-12,
            group="freeze",
        ),
    ]


def _summarize_samples(values: np.ndarray) -> dict[str, float]:
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _plot_audit(
    *,
    out_png: Path,
    y_samples: np.ndarray,
    tf_samples: np.ndarray,
    y_obs: float,
    y_pred: float,
    z_raw: float,
    z_linear: float,
    z_mc: float,
    watch_z_threshold: float,
    hard_z_threshold: float,
    contrib_rows: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.2), dpi=170)

    ax = axes[0, 0]
    ax.hist(y_samples, bins=40, color="#1f77b4", alpha=0.85)
    ax.axvline(y_obs, color="#111111", lw=1.4, ls="--", label="observed Y_p")
    ax.axvline(y_pred, color="#d62728", lw=1.2, ls="-", label="predicted Y_p")
    ax.set_title("He-4 Y_p Monte Carlo distribution")
    ax.set_xlabel("Y_p")
    ax.set_ylabel("count")
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[0, 1]
    ax.hist(tf_samples, bins=40, color="#2ca02c", alpha=0.85)
    ax.set_title("Freeze temperature T_F distribution")
    ax.set_xlabel("T_F [MeV]")
    ax.set_ylabel("count")
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)

    ax = axes[1, 0]
    labels = [str(row["parameter"]) for row in contrib_rows]
    vals = [float(row["sigma_contribution_abs"]) for row in contrib_rows]
    bars = ax.bar(labels, vals, color="#9467bd")
    ax.set_title("Linear σ(Y_p) contribution by parameter")
    ax.set_ylabel("abs(dY/dp) * sigma_p")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() * 0.5, v, f"{v:.2e}", ha="center", va="bottom", fontsize=7)

    ax = axes[1, 1]
    z_vals = [abs(z_raw), abs(z_linear), abs(z_mc)]
    z_labels = ["raw", "propagated-linear", "propagated-mc"]
    colors = []
    for z in z_vals:
        # 条件分岐: `z <= watch_z_threshold` を満たす経路を評価する。
        if z <= watch_z_threshold:
            colors.append("#2ca02c")
        # 条件分岐: 前段条件が不成立で、`z <= hard_z_threshold` を追加評価する。
        elif z <= hard_z_threshold:
            colors.append("#ffbf00")
        else:
            colors.append("#d62728")

    ax.bar(z_labels, z_vals, color=colors)
    ax.axhline(watch_z_threshold, color="#777777", ls="--", lw=1.0, label=f"watch |z|={watch_z_threshold:g}")
    ax.axhline(hard_z_threshold, color="#444444", ls=":", lw=1.0, label=f"hard |z|={hard_z_threshold:g}")
    ax.set_title("He-4 gate status before/after uncertainty propagation")
    ax.set_ylabel("|z|")
    ax.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Step 8.7.22.15: He-4 watch convergence audit")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 8.7.22.15: propagate freeze-chain derivation uncertainty for He-4 gate.")
    parser.add_argument("--outdir", type=Path, default=ROOT / "output" / "public" / "quantum")
    parser.add_argument("--step-tag", type=str, default="8.7.22.15")
    parser.add_argument("--hard-z-threshold", type=float, default=3.0)
    parser.add_argument("--watch-z-threshold", type=float, default=2.0)
    parser.add_argument("--he4-obs", type=float, default=0.245)
    parser.add_argument("--he4-sigma-obs", type=float, default=0.003)
    parser.add_argument("--delta-m-mev", type=float, default=1.293)
    parser.add_argument("--delta-t-n-sec", type=float, default=180.0)
    parser.add_argument("--tau-n-sec", type=float, default=880.0)
    parser.add_argument("--cf0", type=float, default=1.0)
    parser.add_argument("--cf0-sigma-frac", type=float, default=0.03)
    parser.add_argument("--cf-alpha", type=float, default=0.0)
    parser.add_argument("--cf-alpha-sigma", type=float, default=0.03)
    parser.add_argument("--cf-tref-mev", type=float, default=0.75)
    parser.add_argument("--q-b", type=float, default=0.5)
    parser.add_argument("--q-b-sigma", type=float, default=0.01)
    parser.add_argument("--t-b-sec", type=float, default=1.0)
    parser.add_argument("--t-b-sec-sigma-frac", type=float, default=0.02)
    parser.add_argument("--t-b-mev", type=float, default=1.0)
    parser.add_argument("--t-b-mev-sigma-frac", type=float, default=0.0)
    parser.add_argument("--a-w", type=float, default=1.1851851851851851)
    parser.add_argument("--a-w-sigma-frac", type=float, default=0.015)
    parser.add_argument("--mc-samples", type=int, default=30000)
    parser.add_argument("--mc-seed", type=int, default=872215)
    args = parser.parse_args()

    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = _build_specs(args)
    nominal_params = {spec.name: float(spec.mu) for spec in specs}
    nominal_state = _evaluate_state(params=nominal_params, args=args)

    grad_tf = _numeric_gradient(target="tf_mev", specs=specs, nominal_params=nominal_params, args=args)
    grad_cf = _numeric_gradient(target="cf_tf", specs=specs, nominal_params=nominal_params, args=args)
    grad_y = _numeric_gradient(target="y_pred", specs=specs, nominal_params=nominal_params, args=args)

    def _linear_sigma(grad: dict[str, float]) -> float:
        acc = 0.0
        for spec in specs:
            g = float(grad.get(spec.name, float("nan")))
            # 条件分岐: `not math.isfinite(g)` を満たす経路を評価する。
            if not math.isfinite(g):
                continue

            acc += (g * float(spec.sigma)) ** 2

        return float(math.sqrt(acc))

    sigma_tf_linear = _linear_sigma(grad_tf)
    sigma_cf_linear = _linear_sigma(grad_cf)
    sigma_y_linear = _linear_sigma(grad_y)

    rng = np.random.default_rng(int(args.mc_seed))
    sample_map: dict[str, np.ndarray] = {}
    for spec in specs:
        # 条件分岐: `spec.sigma > 0.0` を満たす経路を評価する。
        if spec.sigma > 0.0:
            arr = rng.normal(loc=float(spec.mu), scale=float(spec.sigma), size=int(args.mc_samples))
        else:
            arr = np.full(int(args.mc_samples), float(spec.mu), dtype=float)

        # 条件分岐: `spec.lower is not None` を満たす経路を評価する。

        if spec.lower is not None:
            arr = np.maximum(arr, float(spec.lower))

        sample_map[spec.name] = arr

    cf0_s = sample_map["cf0"]
    cf_alpha_s = sample_map["cf_alpha"]
    q_b_s = sample_map["q_b"]
    t_b_sec_s = sample_map["t_b_sec"]
    t_b_mev_s = sample_map["t_b_mev"]
    a_w_s = sample_map["a_w"]

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        n_eff_s = 5.0 - cf_alpha_s - (1.0 / q_b_s)
        base_s = (cf0_s * (q_b_s / t_b_sec_s) * (float(args.cf_tref_mev) ** (-cf_alpha_s)) * (t_b_mev_s ** (-1.0 / q_b_s))) / a_w_s
        valid = (
            np.isfinite(n_eff_s)
            & np.isfinite(base_s)
            & (n_eff_s > 1.0e-8)
            & (base_s > 0.0)
            & (q_b_s > 0.0)
            & (a_w_s > 0.0)
            & (t_b_sec_s > 0.0)
            & (t_b_mev_s > 0.0)
            & (cf0_s > 0.0)
        )
        tf_s = np.full(int(args.mc_samples), np.nan, dtype=float)
        tf_s[valid] = base_s[valid] ** (1.0 / n_eff_s[valid])
        cf_tf_s = np.full(int(args.mc_samples), np.nan, dtype=float)
        cf_tf_s[valid] = cf0_s[valid] * (tf_s[valid] / float(args.cf_tref_mev)) ** cf_alpha_s[valid]
        np_f_s = np.exp(-float(args.delta_m_mev) / tf_s[valid])
        np_n_s = np_f_s * np.exp(-float(args.delta_t_n_sec) / float(args.tau_n_sec))
        y_s = np.full(int(args.mc_samples), np.nan, dtype=float)
        y_s[valid] = 2.0 * np_n_s / (1.0 + np_n_s)

    tf_valid = tf_s[np.isfinite(tf_s)]
    cf_valid = cf_tf_s[np.isfinite(cf_tf_s)]
    y_valid = y_s[np.isfinite(y_s)]
    # 条件分岐: `tf_valid.size < 100 or cf_valid.size < 100 or y_valid.size < 100` を満たす経路を評価する。
    if tf_valid.size < 100 or cf_valid.size < 100 or y_valid.size < 100:
        raise RuntimeError("insufficient valid Monte Carlo samples for He-4 propagation")

    sigma_tf_mc = float(np.std(tf_valid, ddof=1))
    sigma_cf_mc = float(np.std(cf_valid, ddof=1))
    sigma_y_mc = float(np.std(y_valid, ddof=1))

    y_pred = float(nominal_state["y_pred"])
    y_obs = float(args.he4_obs)
    sigma_obs = float(args.he4_sigma_obs)
    delta_y = y_pred - y_obs

    z_raw = float(delta_y / sigma_obs)
    sigma_total_linear = float(math.sqrt(sigma_obs**2 + sigma_y_linear**2))
    sigma_total_mc = float(math.sqrt(sigma_obs**2 + sigma_y_mc**2))
    z_linear = float(delta_y / sigma_total_linear)
    z_mc = float(delta_y / sigma_total_mc)
    z_abs_conservative = float(max(abs(z_linear), abs(z_mc)))

    raw_status = _gate_status(z_abs=abs(z_raw), hard=float(args.hard_z_threshold), watch=float(args.watch_z_threshold))
    propagated_status = _gate_status(
        z_abs=z_abs_conservative,
        hard=float(args.hard_z_threshold),
        watch=float(args.watch_z_threshold),
    )

    # 条件分岐: `raw_status == "watch" and propagated_status == "pass"` を満たす経路を評価する。
    if raw_status == "watch" and propagated_status == "pass":
        convergence_outcome = "watch_to_pass_after_uncertainty_propagation"
    # 条件分岐: 前段条件が不成立で、`raw_status == propagated_status` を追加評価する。
    elif raw_status == propagated_status:
        convergence_outcome = "watch_fixed_after_uncertainty_propagation"
    else:
        convergence_outcome = f"{raw_status}_to_{propagated_status}_after_uncertainty_propagation"

    contrib_rows: list[dict[str, Any]] = []
    for spec in specs:
        g = float(grad_y.get(spec.name, float("nan")))
        sigma_term = abs(g * float(spec.sigma)) if math.isfinite(g) else float("nan")
        contrib_rows.append(
            {
                "parameter": spec.name,
                "group": spec.group,
                "mu": float(spec.mu),
                "sigma": float(spec.sigma),
                "dY_dp": g,
                "sigma_contribution_abs": sigma_term,
            }
        )

    contrib_rows.sort(key=lambda r: (0.0 if not math.isfinite(float(r["sigma_contribution_abs"])) else -float(r["sigma_contribution_abs"])))

    contrib_group: dict[str, float] = {}
    for row in contrib_rows:
        gname = str(row["group"])
        sval = float(row["sigma_contribution_abs"])
        # 条件分岐: `not math.isfinite(sval)` を満たす経路を評価する。
        if not math.isfinite(sval):
            continue

        contrib_group[gname] = contrib_group.get(gname, 0.0) + sval**2

    for key in list(contrib_group):
        contrib_group[key] = float(math.sqrt(contrib_group[key]))

    metrics_json = out_dir / "bbn_he4_watch_convergence_audit_metrics.json"
    summary_csv = out_dir / "bbn_he4_watch_convergence_audit_summary.csv"
    figure_png = out_dir / "bbn_he4_watch_convergence_audit.png"

    payload = {
        "generated_utc": _iso_now(),
        "phase": {"phase": 8, "step": str(args.step_tag), "name": "He-4 watch convergence with freeze uncertainty propagation"},
        "inputs": {
            "he4_obs": y_obs,
            "he4_sigma_obs": sigma_obs,
            "delta_m_mev": float(args.delta_m_mev),
            "delta_t_n_sec": float(args.delta_t_n_sec),
            "tau_n_sec": float(args.tau_n_sec),
            "cf_tref_mev": float(args.cf_tref_mev),
            "hard_z_threshold": float(args.hard_z_threshold),
            "watch_z_threshold": float(args.watch_z_threshold),
            "mc_samples": int(args.mc_samples),
            "mc_seed": int(args.mc_seed),
        },
        "freeze_nominal": {
            "tf_mev": float(nominal_state["tf_mev"]),
            "cf_tf": float(nominal_state["cf_tf"]),
            "n_eff": float(nominal_state["n_eff"]),
            "y_pred": y_pred,
            "delta_y": float(delta_y),
        },
        "uncertainty_parameters": [
            {
                "name": spec.name,
                "group": spec.group,
                "mu": float(spec.mu),
                "sigma": float(spec.sigma),
                "relative_sigma": float(spec.sigma / spec.mu) if spec.mu != 0.0 else None,
            }
            for spec in specs
        ],
        "linear_propagation": {
            "sigma_tf_mev": sigma_tf_linear,
            "sigma_cf_tf": sigma_cf_linear,
            "sigma_y_pred": sigma_y_linear,
            "sigma_y_by_group": contrib_group,
            "sigma_total_with_obs": sigma_total_linear,
            "z": z_linear,
            "z_abs": abs(z_linear),
            "status": _gate_status(
                z_abs=abs(z_linear),
                hard=float(args.hard_z_threshold),
                watch=float(args.watch_z_threshold),
            ),
            "gradient_tf": grad_tf,
            "gradient_cf_tf": grad_cf,
            "gradient_y_pred": grad_y,
            "contributions": contrib_rows,
        },
        "mc_propagation": {
            "valid_fraction": float(y_valid.size / int(args.mc_samples)),
            "tf": _summarize_samples(tf_valid),
            "cf_tf": _summarize_samples(cf_valid),
            "y_pred": _summarize_samples(y_valid),
            "sigma_tf_mev": sigma_tf_mc,
            "sigma_cf_tf": sigma_cf_mc,
            "sigma_y_pred": sigma_y_mc,
            "sigma_total_with_obs": sigma_total_mc,
            "z": z_mc,
            "z_abs": abs(z_mc),
            "status": _gate_status(
                z_abs=abs(z_mc),
                hard=float(args.hard_z_threshold),
                watch=float(args.watch_z_threshold),
            ),
        },
        "decision": {
            "raw_z": z_raw,
            "raw_z_abs": abs(z_raw),
            "raw_status": raw_status,
            "propagated_linear_z_abs": abs(z_linear),
            "propagated_mc_z_abs": abs(z_mc),
            "propagated_z_abs_conservative": z_abs_conservative,
            "overall_status": propagated_status,
            "criterion": "use max(|z_linear|, |z_mc|) after adding model sigma to observed sigma",
            "convergence_outcome": convergence_outcome,
        },
        "outputs": {
            "metrics_json": _rel(metrics_json),
            "summary_csv": _rel(summary_csv),
            "figure_png": _rel(figure_png),
        },
    }

    with metrics_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["step", str(args.step_tag)])
        writer.writerow(["tf_mev_nominal", nominal_state["tf_mev"]])
        writer.writerow(["cf_tf_nominal", nominal_state["cf_tf"]])
        writer.writerow(["y_pred_nominal", y_pred])
        writer.writerow(["y_obs", y_obs])
        writer.writerow(["delta_y", delta_y])
        writer.writerow(["z_raw_abs", abs(z_raw)])
        writer.writerow(["sigma_tf_linear", sigma_tf_linear])
        writer.writerow(["sigma_cf_linear", sigma_cf_linear])
        writer.writerow(["sigma_y_linear", sigma_y_linear])
        writer.writerow(["sigma_tf_mc", sigma_tf_mc])
        writer.writerow(["sigma_cf_mc", sigma_cf_mc])
        writer.writerow(["sigma_y_mc", sigma_y_mc])
        writer.writerow(["sigma_total_linear", sigma_total_linear])
        writer.writerow(["sigma_total_mc", sigma_total_mc])
        writer.writerow(["z_linear_abs", abs(z_linear)])
        writer.writerow(["z_mc_abs", abs(z_mc)])
        writer.writerow(["z_abs_conservative", z_abs_conservative])
        writer.writerow(["raw_status", raw_status])
        writer.writerow(["overall_status", propagated_status])
        writer.writerow(["convergence_outcome", convergence_outcome])
        writer.writerow(["mc_valid_fraction", float(y_valid.size / int(args.mc_samples))])
        for row in contrib_rows:
            writer.writerow([f"contrib::{row['parameter']}", row["sigma_contribution_abs"]])

    _plot_audit(
        out_png=figure_png,
        y_samples=y_valid,
        tf_samples=tf_valid,
        y_obs=y_obs,
        y_pred=y_pred,
        z_raw=z_raw,
        z_linear=z_linear,
        z_mc=z_mc,
        watch_z_threshold=float(args.watch_z_threshold),
        hard_z_threshold=float(args.hard_z_threshold),
        contrib_rows=contrib_rows,
    )

    worklog.append_event(
        {
            "event_type": "quantum_bbn_he4_watch_convergence_audit",
            "phase": str(args.step_tag),
            "params": {
                "watch_z_threshold": float(args.watch_z_threshold),
                "hard_z_threshold": float(args.hard_z_threshold),
                "mc_samples": int(args.mc_samples),
                "mc_seed": int(args.mc_seed),
            },
            "metrics": {
                "raw_z_abs": abs(z_raw),
                "z_linear_abs": abs(z_linear),
                "z_mc_abs": abs(z_mc),
                "z_abs_conservative": z_abs_conservative,
                "overall_status": propagated_status,
                "convergence_outcome": convergence_outcome,
            },
            "outputs": [metrics_json, summary_csv, figure_png],
        }
    )

    print(f"[ok] metrics: {metrics_json}")
    print(f"[ok] summary: {summary_csv}")
    print(f"[ok] figure : {figure_png}")
    print(
        "[info] raw |z|={0:.4f}, propagated |z| (linear/mc/conservative)={1:.4f}/{2:.4f}/{3:.4f}, status={4}".format(
            abs(z_raw),
            abs(z_linear),
            abs(z_mc),
            z_abs_conservative,
            propagated_status,
        )
    )
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
