from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

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


def _z_score(*, pred: float | None, obs: float, sigma: float) -> float | None:
    # 条件分岐: `pred is None or not math.isfinite(pred) or sigma <= 0.0` を満たす経路を評価する。
    if pred is None or not math.isfinite(pred) or sigma <= 0.0:
        return None

    return float((pred - obs) / sigma)


def _gate_status(*, z_abs: float | None, hard: float, watch: float) -> str:
    # 条件分岐: `z_abs is None` を満たす経路を評価する。
    if z_abs is None:
        return "watch"

    # 条件分岐: `z_abs <= watch` を満たす経路を評価する。

    if z_abs <= watch:
        return "pass"

    # 条件分岐: `z_abs <= hard` を満たす経路を評価する。

    if z_abs <= hard:
        return "watch"

    return "reject"


def _plot_gate(path: Path, rows: list[dict[str, Any]], hard: float, watch: float) -> None:
    labels = [str(r["observable"]) for r in rows]
    values = [float(r["z_abs"]) if r["z_abs"] is not None else float("nan") for r in rows]
    colors = []
    for r in rows:
        s = str(r["status"])
        # 条件分岐: `s == "pass"` を満たす経路を評価する。
        if s == "pass":
            colors.append("#2ca02c")
        # 条件分岐: 前段条件が不成立で、`s == "reject"` を追加評価する。
        elif s == "reject":
            colors.append("#d62728")
        else:
            colors.append("#ffbf00")

    fig, ax = plt.subplots(figsize=(10.8, 4.6), dpi=170)
    xs = list(range(len(labels)))
    ax.bar(xs, [0.0 if not math.isfinite(v) else v for v in values], color=colors)
    ax.axhline(watch, color="#777777", ls="--", lw=1.0, label=f"watch |z|={watch:g}")
    ax.axhline(hard, color="#444444", ls=":", lw=1.0, label=f"hard |z|={hard:g}")
    ax.set_xticks(xs, labels, rotation=15, ha="right")
    ax.set_ylabel("|z|")
    ax.set_title("BBN multi-isotope gate (current closure state)")
    ax.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)
    for idx, row in enumerate(rows):
        # 条件分岐: `row["z_abs"] is None` を満たす経路を評価する。
        if row["z_abs"] is None:
            ax.text(idx, 0.05, "N/A", ha="center", va="bottom", fontsize=8, color="#444444")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 8.7.22.13: BBN multi-isotope gate with Li-7 Be-7 branch re-derivation."
    )
    parser.add_argument("--outdir", type=Path, default=ROOT / "output" / "public" / "quantum")
    parser.add_argument("--step-tag", type=str, default="8.7.22.13")
    parser.add_argument("--hard-z-threshold", type=float, default=3.0)
    parser.add_argument("--watch-z-threshold", type=float, default=2.0)
    parser.add_argument("--tf-mev", type=float, default=0.75)
    parser.add_argument("--tnuc-mev", type=float, default=0.07)
    parser.add_argument("--eta-baryon", type=float, default=6.10e-10)
    parser.add_argument("--delta-m-mev", type=float, default=1.293)
    parser.add_argument("--delta-t-n-sec", type=float, default=180.0)
    parser.add_argument("--tau-n-sec", type=float, default=880.0)
    parser.add_argument("--q-b", type=float, default=0.5)
    parser.add_argument("--t-b-sec", type=float, default=1.0)
    parser.add_argument("--t-b-mev", type=float, default=1.0)
    parser.add_argument("--q-b-ref", type=float, default=0.5)
    parser.add_argument("--t-b-ref-sec", type=float, default=1.0)
    parser.add_argument("--t-b-ref-mev", type=float, default=1.0)
    parser.add_argument("--tnuc-ref-mev", type=float, default=0.07)
    parser.add_argument("--eta10-ref", type=float, default=6.10)
    parser.add_argument("--dh-base", type=float, default=2.60e-5)
    parser.add_argument("--he3he4-base", type=float, default=1.25e-4)
    parser.add_argument("--li7h-base", type=float, default=5.00e-10)
    parser.add_argument("--dh-eta-exp", type=float, default=-1.60)
    parser.add_argument("--he3he4-eta-exp", type=float, default=-0.60)
    parser.add_argument("--li7h-eta-exp", type=float, default=2.00)
    parser.add_argument("--dh-xi-exp", type=float, default=0.35)
    parser.add_argument("--he3he4-xi-exp", type=float, default=0.20)
    parser.add_argument("--li7h-xi-exp", type=float, default=-0.35)
    parser.add_argument("--li7-model", choices=["be7_branch", "reduced"], default="be7_branch")
    parser.add_argument("--be7-prod-base", type=float, default=4.90e-10)
    parser.add_argument("--be7-prod-eta-exp", type=float, default=2.00)
    parser.add_argument("--be7-prod-xi-exp", type=float, default=-0.20)
    parser.add_argument("--li7-destruction-k-be7", type=float, default=0.80)
    parser.add_argument("--li7-destruction-k-li7", type=float, default=0.45)
    parser.add_argument("--li7-destruction-eta-exp", type=float, default=0.25)
    parser.add_argument("--li7-destruction-xi-exp", type=float, default=0.40)
    parser.add_argument("--li7-destruction-temp-exp", type=float, default=0.20)
    parser.add_argument("--li7-survival-floor", type=float, default=0.05)
    parser.add_argument("--he4-obs", type=float, default=0.245)
    parser.add_argument("--he4-sigma", type=float, default=0.003)
    parser.add_argument("--dh-obs", type=float, default=2.55e-5)
    parser.add_argument("--dh-sigma", type=float, default=0.03e-5)
    parser.add_argument("--he3he4-obs", type=float, default=1.1e-4)
    parser.add_argument("--he3he4-sigma", type=float, default=0.2e-4)
    parser.add_argument("--li7h-obs", type=float, default=1.6e-10)
    parser.add_argument("--li7h-sigma", type=float, default=0.3e-10)
    args = parser.parse_args()

    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # He-4 prediction from the freeze chain fixed in Part III 5.5.1.
    np_f = math.exp(-float(args.delta_m_mev) / float(args.tf_mev))
    np_n = np_f * math.exp(-float(args.delta_t_n_sec) / float(args.tau_n_sec))
    he4_pred = float(2.0 * np_n / (1.0 + np_n))

    xi_nuc = float(args.q_b) / float(args.t_b_sec) * (float(args.tnuc_mev) / float(args.t_b_mev)) ** (1.0 / float(args.q_b))
    xi_ref = float(args.q_b_ref) / float(args.t_b_ref_sec) * (
        float(args.tnuc_ref_mev) / float(args.t_b_ref_mev)
    ) ** (1.0 / float(args.q_b_ref))
    eta10 = float(args.eta_baryon) * 1.0e10
    eta_scale = eta10 / float(args.eta10_ref)
    xi_scale = xi_nuc / xi_ref
    tnuc_scale = float(args.tnuc_mev) / float(args.tnuc_ref_mev)

    dh_pred = float(args.dh_base) * eta_scale ** float(args.dh_eta_exp) * xi_scale ** float(args.dh_xi_exp)
    he3he4_pred = (
        float(args.he3he4_base)
        * eta_scale ** float(args.he3he4_eta_exp)
        * xi_scale ** float(args.he3he4_xi_exp)
    )
    li7_diagnostics: dict[str, Any] = {}
    # 条件分岐: `str(args.li7_model) == "be7_branch"` を満たす経路を評価する。
    if str(args.li7_model) == "be7_branch":
        be7_prod = (
            float(args.be7_prod_base)
            * eta_scale ** float(args.be7_prod_eta_exp)
            * xi_scale ** float(args.be7_prod_xi_exp)
        )
        destruction_scale = (
            eta_scale ** float(args.li7_destruction_eta_exp)
            * xi_scale ** float(args.li7_destruction_xi_exp)
            * tnuc_scale ** float(args.li7_destruction_temp_exp)
        )
        lambda_be7 = float(args.li7_destruction_k_be7) * destruction_scale
        lambda_li7 = float(args.li7_destruction_k_li7) * destruction_scale
        survival = max(float(args.li7_survival_floor), 1.0 / (1.0 + lambda_be7 + lambda_li7))
        li7h_pred = be7_prod * survival
        li7_note = "Be-7 production branch with Li-7/Be-7 post-destruction competition."
        li7_diagnostics = {
            "model": "be7_branch",
            "be7_production": {
                "pred_before_destruction": be7_prod,
                "base": float(args.be7_prod_base),
                "eta_exp": float(args.be7_prod_eta_exp),
                "xi_exp": float(args.be7_prod_xi_exp),
            },
            "post_destruction": {
                "destruction_scale": destruction_scale,
                "lambda_be7": lambda_be7,
                "lambda_li7": lambda_li7,
                "survival_factor": survival,
                "k_be7": float(args.li7_destruction_k_be7),
                "k_li7": float(args.li7_destruction_k_li7),
                "eta_exp": float(args.li7_destruction_eta_exp),
                "xi_exp": float(args.li7_destruction_xi_exp),
                "temp_exp": float(args.li7_destruction_temp_exp),
                "survival_floor": float(args.li7_survival_floor),
            },
        }
    else:
        li7h_pred = float(args.li7h_base) * eta_scale ** float(args.li7h_eta_exp) * xi_scale ** float(args.li7h_xi_exp)
        li7_note = "Reduced network branch with eta10 and Xi_P(T_nuc) scaling."
        li7_diagnostics = {
            "model": "reduced",
            "scaling": {
                "base": float(args.li7h_base),
                "eta_exp": float(args.li7h_eta_exp),
                "xi_exp": float(args.li7h_xi_exp),
            },
        }

    rows: list[dict[str, Any]] = []
    for obs_name, pred, obs, sigma, note in [
        (
            "Y_p (He-4 mass fraction)",
            he4_pred,
            float(args.he4_obs),
            float(args.he4_sigma),
            "Derived from freeze chain (T_F -> n/p -> Y_p).",
        ),
        (
            "D/H",
            dh_pred,
            float(args.dh_obs),
            float(args.dh_sigma),
            "Reduced network branch with eta10 and Xi_P(T_nuc) scaling.",
        ),
        (
            "He-3/He-4",
            he3he4_pred,
            float(args.he3he4_obs),
            float(args.he3he4_sigma),
            "Reduced network branch with eta10 and Xi_P(T_nuc) scaling.",
        ),
        (
            "Li-7/H",
            li7h_pred,
            float(args.li7h_obs),
            float(args.li7h_sigma),
            li7_note,
        ),
    ]:
        z = _z_score(pred=pred, obs=obs, sigma=sigma)
        z_abs = None if z is None else abs(z)
        status = _gate_status(z_abs=z_abs, hard=float(args.hard_z_threshold), watch=float(args.watch_z_threshold))
        rows.append(
            {
                "observable": obs_name,
                "predicted": pred,
                "observed": obs,
                "sigma": sigma,
                "z": z,
                "z_abs": z_abs,
                "status": status,
                "note": note,
            }
        )

    statuses = [str(r["status"]) for r in rows]
    # 条件分岐: `any(s == "reject" for s in statuses)` を満たす経路を評価する。
    if any(s == "reject" for s in statuses):
        overall_status = "reject"
    # 条件分岐: 前段条件が不成立で、`any(s == "watch" for s in statuses)` を追加評価する。
    elif any(s == "watch" for s in statuses):
        overall_status = "watch"
    else:
        overall_status = "pass"

    metrics_json = out_dir / "bbn_multi_isotope_gate_metrics.json"
    summary_csv = out_dir / "bbn_multi_isotope_gate_summary.csv"
    figure_png = out_dir / "bbn_multi_isotope_gate.png"

    payload = {
        "generated_utc": _iso_now(),
        "phase": {"phase": 8, "step": str(args.step_tag), "name": "BBN multi-isotope gate with Li-7 channel re-derivation"},
        "inputs": {
            "tf_mev": float(args.tf_mev),
            "tnuc_mev": float(args.tnuc_mev),
            "eta_baryon": float(args.eta_baryon),
            "delta_m_mev": float(args.delta_m_mev),
            "delta_t_n_sec": float(args.delta_t_n_sec),
            "tau_n_sec": float(args.tau_n_sec),
            "q_b": float(args.q_b),
            "t_b_sec": float(args.t_b_sec),
            "t_b_mev": float(args.t_b_mev),
            "hard_z_threshold": float(args.hard_z_threshold),
            "watch_z_threshold": float(args.watch_z_threshold),
        },
        "derived_scales": {
            "eta10": eta10,
            "eta_scale_vs_ref": eta_scale,
            "xi_nuc": xi_nuc,
            "xi_ref": xi_ref,
            "xi_scale_vs_ref": xi_scale,
            "tnuc_scale_vs_ref": tnuc_scale,
            "network_scaling": {
                "D/H": {"base": float(args.dh_base), "eta_exp": float(args.dh_eta_exp), "xi_exp": float(args.dh_xi_exp)},
                "He3/He4": {
                    "base": float(args.he3he4_base),
                    "eta_exp": float(args.he3he4_eta_exp),
                    "xi_exp": float(args.he3he4_xi_exp),
                },
                "Li7/H": li7_diagnostics,
            },
        },
        "results": rows,
        "decision": {
            "overall_status": overall_status,
            "hard_gate": "all observables must satisfy abs(z)<=hard threshold to be hard-pass",
            "watch_policy": "2<abs(z)<=3 is watch; abs(z)>3 is reject",
            "reaction_network_closure": "be7_branch_lithium_channel_if_selected",
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
        writer.writerow(["observable", "predicted", "observed", "sigma", "z", "z_abs", "status", "note"])
        for row in rows:
            writer.writerow(
                [
                    row["observable"],
                    row["predicted"],
                    row["observed"],
                    row["sigma"],
                    row["z"],
                    row["z_abs"],
                    row["status"],
                    row["note"],
                ]
            )

    _plot_gate(figure_png, rows, hard=float(args.hard_z_threshold), watch=float(args.watch_z_threshold))

    worklog.append_event(
        {
            "event_type": "quantum_bbn_multi_isotope_gate",
            "phase": str(args.step_tag),
            "params": {
                "tf_mev": float(args.tf_mev),
                "tnuc_mev": float(args.tnuc_mev),
                "eta_baryon": float(args.eta_baryon),
                "q_b": float(args.q_b),
                "li7_model": str(args.li7_model),
                "hard_z_threshold": float(args.hard_z_threshold),
                "watch_z_threshold": float(args.watch_z_threshold),
            },
            "metrics": {
                "overall_status": overall_status,
                "he4_predicted": he4_pred,
                "dh_predicted": dh_pred,
                "he3he4_predicted": he3he4_pred,
                "li7h_predicted": li7h_pred,
            },
            "outputs": [metrics_json, summary_csv, figure_png],
        }
    )

    print(f"[ok] metrics: {metrics_json}")
    print(f"[ok] summary: {summary_csv}")
    print(f"[ok] figure : {figure_png}")
    print(f"[info] overall_status={overall_status}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
