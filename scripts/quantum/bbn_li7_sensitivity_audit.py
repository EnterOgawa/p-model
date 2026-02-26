from __future__ import annotations

import argparse
import csv
import json
import math
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


# 関数: `_iso_now` の入出力契約と処理意図を定義する。
def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


# 関数: `_gate_status` の入出力契約と処理意図を定義する。

def _gate_status(*, z_abs: float, hard: float, watch: float) -> str:
    # 条件分岐: `z_abs <= watch` を満たす経路を評価する。
    if z_abs <= watch:
        return "pass"

    # 条件分岐: `z_abs <= hard` を満たす経路を評価する。

    if z_abs <= hard:
        return "watch"

    return "reject"


# 関数: `_predict_li7h` の入出力契約と処理意図を定義する。

def _predict_li7h(
    *,
    eta_baryon: float,
    q_b: float,
    tnuc_mev: float,
    t_b_sec: float,
    t_b_mev: float,
    q_b_ref: float,
    t_b_ref_sec: float,
    t_b_ref_mev: float,
    tnuc_ref_mev: float,
    eta10_ref: float,
    be7_prod_base: float,
    be7_prod_eta_exp: float,
    be7_prod_xi_exp: float,
    li7_destruction_k_be7: float,
    li7_destruction_k_li7: float,
    li7_destruction_eta_exp: float,
    li7_destruction_xi_exp: float,
    li7_destruction_temp_exp: float,
    li7_survival_floor: float,
) -> dict[str, float]:
    xi_nuc = q_b / t_b_sec * (tnuc_mev / t_b_mev) ** (1.0 / q_b)
    xi_ref = q_b_ref / t_b_ref_sec * (tnuc_ref_mev / t_b_ref_mev) ** (1.0 / q_b_ref)
    eta10 = eta_baryon * 1.0e10
    eta_scale = eta10 / eta10_ref
    xi_scale = xi_nuc / xi_ref
    tnuc_scale = tnuc_mev / tnuc_ref_mev

    be7_prod = be7_prod_base * eta_scale**be7_prod_eta_exp * xi_scale**be7_prod_xi_exp
    destruction_scale = (
        eta_scale**li7_destruction_eta_exp
        * xi_scale**li7_destruction_xi_exp
        * tnuc_scale**li7_destruction_temp_exp
    )
    lambda_be7 = li7_destruction_k_be7 * destruction_scale
    lambda_li7 = li7_destruction_k_li7 * destruction_scale
    survival = max(li7_survival_floor, 1.0 / (1.0 + lambda_be7 + lambda_li7))
    li7h_pred = be7_prod * survival
    return {
        "xi_nuc": xi_nuc,
        "xi_scale": xi_scale,
        "eta10": eta10,
        "eta_scale": eta_scale,
        "tnuc_scale": tnuc_scale,
        "be7_prod": be7_prod,
        "destruction_scale": destruction_scale,
        "lambda_be7": lambda_be7,
        "lambda_li7": lambda_li7,
        "survival": survival,
        "li7h_pred": li7h_pred,
    }


# 関数: `_plot_sweep` の入出力契約と処理意図を定義する。

def _plot_sweep(
    *,
    out_png: Path,
    rows: list[dict[str, Any]],
    q_values: np.ndarray,
    eta_values: np.ndarray,
    tnuc_values: np.ndarray,
    pass_rate_map: np.ndarray,
    eta_ref_idx: int,
    q_ref_idx: int,
    watch_z_threshold: float,
    hard_z_threshold: float,
) -> None:
    z_abs = np.array([float(r["z_abs"]) for r in rows], dtype=float)
    tnuc = np.array([float(r["tnuc_mev"]) for r in rows], dtype=float)
    q_b = np.array([float(r["q_b"]) for r in rows], dtype=float)
    eta_b = np.array([float(r["eta_baryon"]) for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), dpi=170)

    ax = axes[0, 0]
    ax.hist(z_abs, bins=36, color="#1f77b4", alpha=0.85)
    ax.axvline(watch_z_threshold, color="#777777", ls="--", lw=1.0, label=f"watch |z|={watch_z_threshold:g}")
    ax.axvline(hard_z_threshold, color="#444444", ls=":", lw=1.0, label=f"hard |z|={hard_z_threshold:g}")
    ax.set_title("Li-7/H |z| distribution")
    ax.set_xlabel("|z|")
    ax.set_ylabel("count")
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[0, 1]
    image = ax.imshow(
        pass_rate_map,
        origin="lower",
        aspect="auto",
        extent=[float(eta_values[0]) * 1e10, float(eta_values[-1]) * 1e10, float(q_values[0]), float(q_values[-1])],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("pass fraction over T_N")
    ax.set_title("Li-7 pass-rate map")
    ax.set_xlabel("eta10")
    ax.set_ylabel("q_B")
    ax.grid(False)

    ax = axes[1, 0]
    mask_eta = np.isclose(eta_b, float(eta_values[eta_ref_idx]), rtol=0.0, atol=1e-18)
    tnuc_slice = tnuc[mask_eta]
    z_slice = z_abs[mask_eta]
    order = np.argsort(tnuc_slice)
    ax.plot(tnuc_slice[order], z_slice[order], lw=1.4, color="#ff7f0e")
    ax.axhline(watch_z_threshold, color="#777777", ls="--", lw=1.0)
    ax.axhline(hard_z_threshold, color="#444444", ls=":", lw=1.0)
    ax.set_title(f"eta10={float(eta_values[eta_ref_idx]) * 1e10:.3f} slice")
    ax.set_xlabel("T_N [MeV]")
    ax.set_ylabel("|z| (all q_B points)")
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)

    ax = axes[1, 1]
    mask_q = np.isclose(q_b, float(q_values[q_ref_idx]), rtol=0.0, atol=1e-18)
    eta_slice = eta_b[mask_q] * 1e10
    z_slice2 = z_abs[mask_q]
    order2 = np.argsort(eta_slice)
    ax.plot(eta_slice[order2], z_slice2[order2], lw=1.4, color="#2ca02c")
    ax.axhline(watch_z_threshold, color="#777777", ls="--", lw=1.0)
    ax.axhline(hard_z_threshold, color="#444444", ls=":", lw=1.0)
    ax.set_title(f"q_B={float(q_values[q_ref_idx]):.3f} slice")
    ax.set_xlabel("eta10")
    ax.set_ylabel("|z| (all T_N points)")
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)

    fig.suptitle("Step 8.7.22.14: Li-7 channel sensitivity audit")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    parser = argparse.ArgumentParser(description="Step 8.7.22.14: Li-7 channel sensitivity audit (T_N, q_B, eta_b sweep).")
    parser.add_argument("--outdir", type=Path, default=ROOT / "output" / "public" / "quantum")
    parser.add_argument("--step-tag", type=str, default="8.7.22.14")
    parser.add_argument("--watch-z-threshold", type=float, default=2.0)
    parser.add_argument("--hard-z-threshold", type=float, default=3.0)

    parser.add_argument("--tnuc-min-mev", type=float, default=0.05)
    parser.add_argument("--tnuc-max-mev", type=float, default=0.09)
    parser.add_argument("--tnuc-count", type=int, default=21)
    parser.add_argument("--qb-min", type=float, default=0.35)
    parser.add_argument("--qb-max", type=float, default=0.75)
    parser.add_argument("--qb-count", type=int, default=17)
    parser.add_argument("--eta-min", type=float, default=5.4e-10)
    parser.add_argument("--eta-max", type=float, default=6.8e-10)
    parser.add_argument("--eta-count", type=int, default=17)

    parser.add_argument("--tnuc-ref-mev", type=float, default=0.07)
    parser.add_argument("--q-b-ref", type=float, default=0.5)
    parser.add_argument("--eta-ref", type=float, default=6.10e-10)
    parser.add_argument("--t-b-sec", type=float, default=1.0)
    parser.add_argument("--t-b-mev", type=float, default=1.0)
    parser.add_argument("--t-b-ref-sec", type=float, default=1.0)
    parser.add_argument("--t-b-ref-mev", type=float, default=1.0)
    parser.add_argument("--eta10-ref", type=float, default=6.10)

    parser.add_argument("--be7-prod-base", type=float, default=4.90e-10)
    parser.add_argument("--be7-prod-eta-exp", type=float, default=2.00)
    parser.add_argument("--be7-prod-xi-exp", type=float, default=-0.20)
    parser.add_argument("--li7-destruction-k-be7", type=float, default=0.80)
    parser.add_argument("--li7-destruction-k-li7", type=float, default=0.45)
    parser.add_argument("--li7-destruction-eta-exp", type=float, default=0.25)
    parser.add_argument("--li7-destruction-xi-exp", type=float, default=0.40)
    parser.add_argument("--li7-destruction-temp-exp", type=float, default=0.20)
    parser.add_argument("--li7-survival-floor", type=float, default=0.05)

    parser.add_argument("--li7h-obs", type=float, default=1.6e-10)
    parser.add_argument("--li7h-sigma", type=float, default=0.3e-10)
    args = parser.parse_args()

    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    q_values = np.linspace(float(args.qb_min), float(args.qb_max), int(args.qb_count))
    eta_values = np.linspace(float(args.eta_min), float(args.eta_max), int(args.eta_count))
    tnuc_values = np.linspace(float(args.tnuc_min_mev), float(args.tnuc_max_mev), int(args.tnuc_count))
    eta_ref_idx = int(np.argmin(np.abs(eta_values - float(args.eta_ref))))
    q_ref_idx = int(np.argmin(np.abs(q_values - float(args.q_b_ref))))

    rows: list[dict[str, Any]] = []
    pass_rate_map = np.zeros((len(q_values), len(eta_values)), dtype=float)

    for q_idx, q_b in enumerate(q_values):
        for eta_idx, eta_b in enumerate(eta_values):
            pass_count = 0
            for tnuc_mev in tnuc_values:
                pred = _predict_li7h(
                    eta_baryon=float(eta_b),
                    q_b=float(q_b),
                    tnuc_mev=float(tnuc_mev),
                    t_b_sec=float(args.t_b_sec),
                    t_b_mev=float(args.t_b_mev),
                    q_b_ref=float(args.q_b_ref),
                    t_b_ref_sec=float(args.t_b_ref_sec),
                    t_b_ref_mev=float(args.t_b_ref_mev),
                    tnuc_ref_mev=float(args.tnuc_ref_mev),
                    eta10_ref=float(args.eta10_ref),
                    be7_prod_base=float(args.be7_prod_base),
                    be7_prod_eta_exp=float(args.be7_prod_eta_exp),
                    be7_prod_xi_exp=float(args.be7_prod_xi_exp),
                    li7_destruction_k_be7=float(args.li7_destruction_k_be7),
                    li7_destruction_k_li7=float(args.li7_destruction_k_li7),
                    li7_destruction_eta_exp=float(args.li7_destruction_eta_exp),
                    li7_destruction_xi_exp=float(args.li7_destruction_xi_exp),
                    li7_destruction_temp_exp=float(args.li7_destruction_temp_exp),
                    li7_survival_floor=float(args.li7_survival_floor),
                )
                z = (pred["li7h_pred"] - float(args.li7h_obs)) / float(args.li7h_sigma)
                z_abs = abs(z)
                status = _gate_status(z_abs=z_abs, hard=float(args.hard_z_threshold), watch=float(args.watch_z_threshold))
                # 条件分岐: `status == "pass"` を満たす経路を評価する。
                if status == "pass":
                    pass_count += 1

                rows.append(
                    {
                        "q_b": float(q_b),
                        "eta_baryon": float(eta_b),
                        "eta10": float(eta_b) * 1.0e10,
                        "tnuc_mev": float(tnuc_mev),
                        "li7h_pred": float(pred["li7h_pred"]),
                        "li7h_obs": float(args.li7h_obs),
                        "li7h_sigma": float(args.li7h_sigma),
                        "z": float(z),
                        "z_abs": float(z_abs),
                        "status": status,
                        "be7_prod": float(pred["be7_prod"]),
                        "destruction_scale": float(pred["destruction_scale"]),
                        "lambda_be7": float(pred["lambda_be7"]),
                        "lambda_li7": float(pred["lambda_li7"]),
                        "survival_factor": float(pred["survival"]),
                    }
                )

            pass_rate_map[q_idx, eta_idx] = pass_count / float(len(tnuc_values))

    status_counts = {"pass": 0, "watch": 0, "reject": 0}
    for row in rows:
        status_counts[str(row["status"])] += 1

    total_points = len(rows)
    z_abs_values = np.array([float(r["z_abs"]) for r in rows], dtype=float)

    center_pred = _predict_li7h(
        eta_baryon=float(args.eta_ref),
        q_b=float(args.q_b_ref),
        tnuc_mev=float(args.tnuc_ref_mev),
        t_b_sec=float(args.t_b_sec),
        t_b_mev=float(args.t_b_mev),
        q_b_ref=float(args.q_b_ref),
        t_b_ref_sec=float(args.t_b_ref_sec),
        t_b_ref_mev=float(args.t_b_ref_mev),
        tnuc_ref_mev=float(args.tnuc_ref_mev),
        eta10_ref=float(args.eta10_ref),
        be7_prod_base=float(args.be7_prod_base),
        be7_prod_eta_exp=float(args.be7_prod_eta_exp),
        be7_prod_xi_exp=float(args.be7_prod_xi_exp),
        li7_destruction_k_be7=float(args.li7_destruction_k_be7),
        li7_destruction_k_li7=float(args.li7_destruction_k_li7),
        li7_destruction_eta_exp=float(args.li7_destruction_eta_exp),
        li7_destruction_xi_exp=float(args.li7_destruction_xi_exp),
        li7_destruction_temp_exp=float(args.li7_destruction_temp_exp),
        li7_survival_floor=float(args.li7_survival_floor),
    )
    center_z = (center_pred["li7h_pred"] - float(args.li7h_obs)) / float(args.li7h_sigma)
    center_z_abs = abs(center_z)
    center_status = _gate_status(
        z_abs=center_z_abs,
        hard=float(args.hard_z_threshold),
        watch=float(args.watch_z_threshold),
    )

    pass_rows = [r for r in rows if str(r["status"]) == "pass"]
    watch_rows = [r for r in rows if str(r["status"]) == "watch"]
    reject_rows = [r for r in rows if str(r["status"]) == "reject"]

    # 関数: `_bounds` の入出力契約と処理意図を定義する。
    def _bounds(data: list[dict[str, Any]], key: str) -> dict[str, float] | None:
        # 条件分岐: `not data` を満たす経路を評価する。
        if not data:
            return None

        values = [float(d[key]) for d in data]
        return {"min": float(min(values)), "max": float(max(values))}

    # 条件分岐: `center_status == "pass"` を満たす経路を評価する。

    if center_status == "pass":
        overall_status = "pass"
    # 条件分岐: 前段条件が不成立で、`center_status == "watch"` を追加評価する。
    elif center_status == "watch":
        overall_status = "watch"
    else:
        overall_status = "reject"

    metrics_json = out_dir / "bbn_li7_sensitivity_audit_metrics.json"
    summary_csv = out_dir / "bbn_li7_sensitivity_audit_summary.csv"
    figure_png = out_dir / "bbn_li7_sensitivity_audit.png"

    payload = {
        "generated_utc": _iso_now(),
        "phase": {"phase": 8, "step": str(args.step_tag), "name": "Li-7 channel sensitivity audit"},
        "grid": {
            "tnuc_mev": {
                "min": float(args.tnuc_min_mev),
                "max": float(args.tnuc_max_mev),
                "count": int(args.tnuc_count),
            },
            "q_b": {"min": float(args.qb_min), "max": float(args.qb_max), "count": int(args.qb_count)},
            "eta_baryon": {
                "min": float(args.eta_min),
                "max": float(args.eta_max),
                "count": int(args.eta_count),
            },
        },
        "reference_point": {
            "tnuc_mev": float(args.tnuc_ref_mev),
            "q_b": float(args.q_b_ref),
            "eta_baryon": float(args.eta_ref),
            "li7h_pred": float(center_pred["li7h_pred"]),
            "z": float(center_z),
            "z_abs": float(center_z_abs),
            "status": center_status,
            "be7_prod": float(center_pred["be7_prod"]),
            "lambda_be7": float(center_pred["lambda_be7"]),
            "lambda_li7": float(center_pred["lambda_li7"]),
            "survival_factor": float(center_pred["survival"]),
        },
        "status_counts": status_counts,
        "status_fraction": {
            "pass": float(status_counts["pass"] / total_points),
            "watch": float(status_counts["watch"] / total_points),
            "reject": float(status_counts["reject"] / total_points),
        },
        "z_abs_distribution": {
            "min": float(np.min(z_abs_values)),
            "median": float(np.median(z_abs_values)),
            "p95": float(np.quantile(z_abs_values, 0.95)),
            "max": float(np.max(z_abs_values)),
        },
        "pass_region_bounds": {
            "pass": {
                "tnuc_mev": _bounds(pass_rows, "tnuc_mev"),
                "q_b": _bounds(pass_rows, "q_b"),
                "eta_baryon": _bounds(pass_rows, "eta_baryon"),
            },
            "watch": {
                "tnuc_mev": _bounds(watch_rows, "tnuc_mev"),
                "q_b": _bounds(watch_rows, "q_b"),
                "eta_baryon": _bounds(watch_rows, "eta_baryon"),
            },
            "reject": {
                "tnuc_mev": _bounds(reject_rows, "tnuc_mev"),
                "q_b": _bounds(reject_rows, "q_b"),
                "eta_baryon": _bounds(reject_rows, "eta_baryon"),
            },
        },
        "pass_rate_map": {
            "eta10_axis": [float(v * 1.0e10) for v in eta_values],
            "q_b_axis": [float(v) for v in q_values],
            "values": [[float(v) for v in row] for row in pass_rate_map],
            "boundary_cell_count": int(np.sum((pass_rate_map > 0.0) & (pass_rate_map < 1.0))),
        },
        "decision": {
            "overall_status": overall_status,
            "criterion": "reference-point Li-7 status (pass/watch/reject) with sweep boundary fixed",
            "hard_gate": "abs(z)<=3",
            "watch_gate": "2<abs(z)<=3",
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
        writer.writerow(
            [
                "q_b",
                "eta_baryon",
                "eta10",
                "tnuc_mev",
                "li7h_pred",
                "li7h_obs",
                "li7h_sigma",
                "z",
                "z_abs",
                "status",
                "be7_prod",
                "destruction_scale",
                "lambda_be7",
                "lambda_li7",
                "survival_factor",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["q_b"],
                    row["eta_baryon"],
                    row["eta10"],
                    row["tnuc_mev"],
                    row["li7h_pred"],
                    row["li7h_obs"],
                    row["li7h_sigma"],
                    row["z"],
                    row["z_abs"],
                    row["status"],
                    row["be7_prod"],
                    row["destruction_scale"],
                    row["lambda_be7"],
                    row["lambda_li7"],
                    row["survival_factor"],
                ]
            )

    _plot_sweep(
        out_png=figure_png,
        rows=rows,
        q_values=q_values,
        eta_values=eta_values,
        tnuc_values=tnuc_values,
        pass_rate_map=pass_rate_map,
        eta_ref_idx=eta_ref_idx,
        q_ref_idx=q_ref_idx,
        watch_z_threshold=float(args.watch_z_threshold),
        hard_z_threshold=float(args.hard_z_threshold),
    )

    worklog.append_event(
        {
            "event_type": "quantum_bbn_li7_sensitivity_audit",
            "phase": str(args.step_tag),
            "params": {
                "tnuc_min_mev": float(args.tnuc_min_mev),
                "tnuc_max_mev": float(args.tnuc_max_mev),
                "tnuc_count": int(args.tnuc_count),
                "qb_min": float(args.qb_min),
                "qb_max": float(args.qb_max),
                "qb_count": int(args.qb_count),
                "eta_min": float(args.eta_min),
                "eta_max": float(args.eta_max),
                "eta_count": int(args.eta_count),
            },
            "metrics": {
                "overall_status": overall_status,
                "reference_status": center_status,
                "reference_z_abs": float(center_z_abs),
                "pass_fraction": float(status_counts["pass"] / total_points),
                "watch_fraction": float(status_counts["watch"] / total_points),
                "reject_fraction": float(status_counts["reject"] / total_points),
            },
            "outputs": [metrics_json, summary_csv, figure_png],
        }
    )

    print(f"[ok] metrics: {metrics_json}")
    print(f"[ok] summary: {summary_csv}")
    print(f"[ok] figure : {figure_png}")
    print(
        "[info] status="
        f"{overall_status}; ref |z|={center_z_abs:.4f}; "
        f"pass/watch/reject={status_counts['pass']}/{status_counts['watch']}/{status_counts['reject']}"
    )
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())

