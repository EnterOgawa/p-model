#!/usr/bin/env python3
"""
llr_kappa_llr_decontamination_stability_sweep.py

Roadmap Step 8.7.47.18:
- Sweep LLR template-decontamination settings (fit mode / weight scheme / min orth std).
- Freeze robustness of beta_LLR against decontamination hyper-parameters.

Inputs:
- output/private/llr/batch/llr_batch_points.csv

Outputs (default: output/private/llr and synced to output/public/llr):
- llr_kappa_llr_decontamination_stability_sweep.csv
- llr_kappa_llr_decontamination_pairwise_z.csv
- llr_kappa_llr_decontamination_stability_sweep_metrics.json
- llr_kappa_llr_decontamination_stability_sweep.pdf
- llr_kappa_llr_decontamination_stability_sweep.png
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.llr import llr_kappa_llr_direct_fit as llrfit


# 関数: `_safe_rel` の入出力契約と処理意図を定義する。
def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


# 関数: `_status_from_abs_z` の入出力契約と処理意図を定義する。

def _status_from_abs_z(abs_z: Optional[float]) -> str:
    # 条件分岐: `abs_z is None or not np.isfinite(abs_z)` を満たす経路を評価する。
    if abs_z is None or not np.isfinite(abs_z):
        return "reject"

    # 条件分岐: `abs_z <= 2.0` を満たす経路を評価する。

    if abs_z <= 2.0:
        return "pass"

    # 条件分岐: `abs_z <= 3.0` を満たす経路を評価する。

    if abs_z <= 3.0:
        return "watch"

    return "reject"


# 関数: `_status_from_span` の入出力契約と処理意図を定義する。

def _status_from_span(span: Optional[float], pass_max: float, watch_max: float) -> str:
    # 条件分岐: `span is None or not np.isfinite(span)` を満たす経路を評価する。
    if span is None or not np.isfinite(span):
        return "reject"

    # 条件分岐: `span <= float(pass_max)` を満たす経路を評価する。

    if span <= float(pass_max):
        return "pass"

    # 条件分岐: `span <= float(watch_max)` を満たす経路を評価する。

    if span <= float(watch_max):
        return "watch"

    return "reject"


# 関数: `_combine_status` の入出力契約と処理意図を定義する。

def _combine_status(statuses: Sequence[str]) -> str:
    norm = [str(s or "").strip().lower() for s in statuses if str(s or "").strip()]
    # 条件分岐: `not norm` を満たす経路を評価する。
    if not norm:
        return "reject"

    # 条件分岐: `any(s == "reject" for s in norm)` を満たす経路を評価する。

    if any(s == "reject" for s in norm):
        return "reject"

    # 条件分岐: `all(s == "pass" for s in norm)` を満たす経路を評価する。

    if all(s == "pass" for s in norm):
        return "pass"

    return "watch"


# 関数: `_parse_str_list` の入出力契約と処理意図を定義する。

def _parse_str_list(csv_like: str) -> List[str]:
    return [v.strip() for v in str(csv_like).split(",") if v.strip()]


# 関数: `_parse_float_list` の入出力契約と処理意図を定義する。

def _parse_float_list(csv_like: str) -> List[float]:
    out: List[float] = []
    for tok in str(csv_like).split(","):
        t = tok.strip()
        # 条件分岐: `not t` を満たす経路を評価する。
        if not t:
            continue

        try:
            out.append(float(t))
        except ValueError:
            continue

    return out


# 関数: `_run_sweep` の入出力契約と処理意図を定義する。

def _run_sweep(
    df: pd.DataFrame,
    fit_modes: Sequence[str],
    weight_schemes: Sequence[str],
    min_orth_values: Sequence[float],
    floor_station: int,
    floor_target: int,
    floor_station_target: int,
    max_weight_cap: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    run_records: List[Dict[str, Any]] = []

    for fit_mode in fit_modes:
        for weight_scheme in weight_schemes:
            sample_weight = llrfit._build_imbalance_weight(
                df=df,
                scheme=weight_scheme,
                floor_station=int(floor_station),
                floor_target=int(floor_target),
                floor_station_target=int(floor_station_target),
                max_weight_cap=float(max_weight_cap),
            )
            for min_std in min_orth_values:
                proj_df, summary = llrfit._run_template_decontamination_audit(
                    df=df,
                    fit_mode=fit_mode,
                    sample_weight=sample_weight,
                    min_std=float(min_std),
                )
                label = f"{fit_mode}|{weight_scheme}|{min_std:.1e}"
                beta_est = float(summary.get("decontaminated_kappa_est", float("nan")))
                beta_sigma = float(summary.get("decontaminated_kappa_sigma", float("nan")))
                abs_z = float(summary.get("decontaminated_abs_z", float("nan")))
                run_records.append(
                    {
                        "label": label,
                        "fit_mode": fit_mode,
                        "weight_scheme": weight_scheme,
                        "min_orth_std": float(min_std),
                        "beta_est": beta_est,
                        "beta_sigma": beta_sigma,
                        "abs_z_beta_minus_1": abs_z,
                    }
                )
                rows.append(
                    {
                        "run_label": label,
                        "fit_mode": fit_mode,
                        "weight_scheme": weight_scheme,
                        "min_orth_std": float(min_std),
                        "decontaminated_kappa_est": beta_est,
                        "decontaminated_kappa_sigma": beta_sigma,
                        "decontaminated_abs_z": abs_z,
                        "kappa_minus_1_status": str(summary.get("kappa_minus_1_status", "reject")),
                        "decontamination_shift_status": str(summary.get("status", "reject")),
                        "kappa_shift_decont_minus_base": float(summary.get("kappa_shift_decont_minus_base", float("nan"))),
                        "abs_z_shift": float(summary.get("abs_z_shift", float("nan"))),
                        "max_abs_corr_before": float(summary.get("max_abs_corr_before", float("nan"))),
                        "max_abs_corr_after": float(summary.get("max_abs_corr_after", float("nan"))),
                        "nuisance_orth_kept_count": int(summary.get("nuisance_orth_kept_count", 0)),
                        "nuisance_orth_dropped_count": int(summary.get("nuisance_orth_dropped_count", 0)),
                        "projection_rows": int(len(proj_df)),
                    }
                )

    out_df = pd.DataFrame(rows).sort_values(["fit_mode", "weight_scheme", "min_orth_std"]).reset_index(drop=True)
    # 条件分岐: `not run_records` を満たす経路を評価する。
    if not run_records:
        return out_df, pd.DataFrame(), {"status": "reject"}

    for i in range(len(run_records)):
        for j in range(i + 1, len(run_records)):
            ri = run_records[i]
            rj = run_records[j]
            bi = float(ri["beta_est"])
            bj = float(rj["beta_est"])
            si = float(ri["beta_sigma"])
            sj = float(rj["beta_sigma"])
            # 条件分岐: `not (np.isfinite(bi) and np.isfinite(bj) and np.isfinite(si) and np.isfinite(...` を満たす経路を評価する。
            if not (np.isfinite(bi) and np.isfinite(bj) and np.isfinite(si) and np.isfinite(sj) and si > 0 and sj > 0):
                continue

            denom = math.sqrt(max((si * si) + (sj * sj), 1e-30))
            abs_z_pair = abs((bi - bj) / denom) if denom > 0 else float("nan")
            pair_rows.append(
                {
                    "run_i": str(ri["label"]),
                    "run_j": str(rj["label"]),
                    "beta_i": bi,
                    "beta_j": bj,
                    "sigma_i": si,
                    "sigma_j": sj,
                    "abs_z_pair": abs_z_pair,
                    "status": _status_from_abs_z(abs_z_pair),
                }
            )

    pair_df = pd.DataFrame(pair_rows).sort_values(["abs_z_pair"], ascending=[False]).reset_index(drop=True)

    beta_vals = pd.to_numeric(out_df.get("decontaminated_kappa_est"), errors="coerce").to_numpy(dtype=float)
    beta_sigmas = pd.to_numeric(out_df.get("decontaminated_kappa_sigma"), errors="coerce").to_numpy(dtype=float)
    abs_z_vals = pd.to_numeric(out_df.get("decontaminated_abs_z"), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(beta_vals) & np.isfinite(beta_sigmas) & (beta_sigmas > 0) & np.isfinite(abs_z_vals)
    beta_vals = beta_vals[ok]
    beta_sigmas = beta_sigmas[ok]
    abs_z_vals = abs_z_vals[ok]

    # 条件分岐: `len(beta_vals) == 0` を満たす経路を評価する。
    if len(beta_vals) == 0:
        return out_df, pair_df, {"status": "reject"}

    beta_min = float(np.min(beta_vals))
    beta_max = float(np.max(beta_vals))
    beta_span = float(beta_max - beta_min)
    beta_med = float(np.median(beta_vals))
    sigma_med = float(np.median(beta_sigmas))
    max_abs_z_beta_minus_1 = float(np.max(abs_z_vals))
    max_abs_z_pair = float(pd.to_numeric(pair_df.get("abs_z_pair"), errors="coerce").max()) if not pair_df.empty else float("nan")
    span_status = _status_from_span(beta_span, pass_max=0.005, watch_max=0.02)
    pair_status = _status_from_abs_z(max_abs_z_pair)
    beta_minus_1_status = _status_from_abs_z(max_abs_z_beta_minus_1)
    shift_status = _combine_status(
        [str(v) for v in out_df.get("decontamination_shift_status", pd.Series(dtype=str)).astype(str).tolist()]
    )
    stability_status = _combine_status([span_status, pair_status, beta_minus_1_status])
    overall_status = _combine_status([stability_status, shift_status])

    summary = {
        "n_runs": int(len(out_df)),
        "n_pairwise": int(len(pair_df)),
        "beta_min": beta_min,
        "beta_max": beta_max,
        "beta_span": beta_span,
        "beta_median": beta_med,
        "beta_sigma_median": sigma_med,
        "max_abs_z_beta_minus_1": max_abs_z_beta_minus_1,
        "max_abs_z_pairwise": max_abs_z_pair,
        "span_status": span_status,
        "pairwise_status": pair_status,
        "beta_minus_1_status": beta_minus_1_status,
        "decontamination_shift_status": shift_status,
        "stability_status": stability_status,
        "status": overall_status,
    }
    return out_df, pair_df, summary


# 関数: `_write_plot` の入出力契約と処理意図を定義する。

def _write_plot(sweep_df: pd.DataFrame, pair_df: pd.DataFrame, summary: Dict[str, Any], out_pdf: Path, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14.0, 10.6), height_ratios=[1.3, 1.0])
    ax0 = axes[0]
    # 条件分岐: `sweep_df.empty` を満たす経路を評価する。
    if sweep_df.empty:
        ax0.text(0.5, 0.5, "no sweep rows", transform=ax0.transAxes, ha="center", va="center")
        ax0.set_axis_off()
    else:
        labels = sweep_df["run_label"].astype(str).tolist()
        x = np.arange(len(labels), dtype=float)
        y = pd.to_numeric(sweep_df["decontaminated_kappa_est"], errors="coerce").to_numpy(dtype=float)
        e = pd.to_numeric(sweep_df["decontaminated_kappa_sigma"], errors="coerce").to_numpy(dtype=float)
        ax0.errorbar(x, y, yerr=e, fmt="o", color="#1f77b4", ecolor="#1f77b4", capsize=3)
        ax0.axhline(1.0, color="#444444", linestyle="--", linewidth=1.2)
        ax0.set_xticks(x)
        ax0.set_xticklabels(labels, rotation=65, ha="right", fontsize=8.0)
        ax0.set_ylabel("beta_LLR (decontaminated)")
        ax0.set_title("LLR decontamination stability sweep")
        ax0.grid(alpha=0.22)

    ax1 = axes[1]
    # 条件分岐: `pair_df.empty` を満たす経路を評価する。
    if pair_df.empty:
        ax1.text(0.5, 0.5, "no pairwise rows", transform=ax1.transAxes, ha="center", va="center")
        ax1.set_axis_off()
    else:
        q = pair_df.head(20).copy()
        labels = [f"{a} vs {b}" for a, b in zip(q["run_i"].astype(str), q["run_j"].astype(str))]
        x = np.arange(len(labels), dtype=float)
        z = pd.to_numeric(q["abs_z_pair"], errors="coerce").to_numpy(dtype=float)
        ax1.bar(x, z, color="#ff7f0e", alpha=0.8)
        ax1.axhline(2.0, color="#999999", linestyle="--", linewidth=1.0)
        ax1.axhline(3.0, color="#999999", linestyle="--", linewidth=1.0)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=70, ha="right", fontsize=7.2)
        ax1.set_ylabel("|z(beta_i-beta_j)|")
        ax1.set_title("Top pairwise differences")
        ax1.grid(axis="y", alpha=0.22)

    note = (
        f"status={summary.get('status')} / stability={summary.get('stability_status')} / "
        f"span={summary.get('beta_span'):.6g} / max_pairwise_z={summary.get('max_abs_z_pairwise'):.3f}"
    )
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=10.0)
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 1.0])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# 関数: `_sync_outputs_to_public` の入出力契約と処理意図を定義する。

def _sync_outputs_to_public(paths: Iterable[Path], private_root: Path, public_root: Path) -> List[str]:
    public_root.mkdir(parents=True, exist_ok=True)
    synced: List[str] = []
    for p in paths:
        try:
            rel = p.resolve().relative_to(private_root.resolve())
        except Exception:
            rel = Path(p.name)

        dst = public_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        synced.append(_safe_rel(dst, _ROOT))

    return synced


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    ap = argparse.ArgumentParser(description="LLR template-decontamination stability sweep.")
    ap.add_argument(
        "--points-csv",
        type=str,
        default=str(_ROOT / "output" / "private" / "llr" / "batch" / "llr_batch_points.csv"),
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(_ROOT / "output" / "private" / "llr"),
    )
    ap.add_argument(
        "--public-dir",
        type=str,
        default=str(_ROOT / "output" / "public" / "llr"),
    )
    ap.add_argument(
        "--fit-modes",
        type=str,
        default="station_target_year,station_target,station,none",
    )
    ap.add_argument(
        "--weight-schemes",
        type=str,
        default="uniform,inv_station,inv_target,inv_station_target,station_cap_p95",
    )
    ap.add_argument(
        "--min-orth-std-values",
        type=str,
        default="1e-8,1e-6,1e-4",
    )
    ap.add_argument("--weight-floor-station", type=int, default=180)
    ap.add_argument("--weight-floor-target", type=int, default=180)
    ap.add_argument("--weight-floor-station-target", type=int, default=120)
    ap.add_argument("--max-weight-cap", type=float, default=8.0)
    args = ap.parse_args()

    points_csv = Path(str(args.points_csv))
    out_dir = Path(str(args.out_dir))
    public_dir = Path(str(args.public_dir))
    # 条件分岐: `not points_csv.is_absolute()` を満たす経路を評価する。
    if not points_csv.is_absolute():
        points_csv = (_ROOT / points_csv).resolve()

    # 条件分岐: `not out_dir.is_absolute()` を満たす経路を評価する。

    if not out_dir.is_absolute():
        out_dir = (_ROOT / out_dir).resolve()

    # 条件分岐: `not public_dir.is_absolute()` を満たす経路を評価する。

    if not public_dir.is_absolute():
        public_dir = (_ROOT / public_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    fit_modes = _parse_str_list(str(args.fit_modes))
    weight_schemes = _parse_str_list(str(args.weight_schemes))
    min_orth_values = _parse_float_list(str(args.min_orth_std_values))
    # 条件分岐: `not fit_modes` を満たす経路を評価する。
    if not fit_modes:
        raise RuntimeError("empty fit-modes")

    # 条件分岐: `not weight_schemes` を満たす経路を評価する。

    if not weight_schemes:
        raise RuntimeError("empty weight-schemes")

    # 条件分岐: `not min_orth_values` を満たす経路を評価する。

    if not min_orth_values:
        raise RuntimeError("empty min-orth-std-values")

    df = llrfit._read_points(points_csv)
    # 条件分岐: `df.empty` を満たす経路を評価する。
    if df.empty:
        raise RuntimeError(f"no valid inlier rows from {points_csv}")

    sweep_df, pair_df, summary = _run_sweep(
        df=df,
        fit_modes=fit_modes,
        weight_schemes=weight_schemes,
        min_orth_values=min_orth_values,
        floor_station=int(args.weight_floor_station),
        floor_target=int(args.weight_floor_target),
        floor_station_target=int(args.weight_floor_station_target),
        max_weight_cap=float(args.max_weight_cap),
    )

    sweep_csv = out_dir / "llr_kappa_llr_decontamination_stability_sweep.csv"
    pair_csv = out_dir / "llr_kappa_llr_decontamination_pairwise_z.csv"
    metrics_json = out_dir / "llr_kappa_llr_decontamination_stability_sweep_metrics.json"
    plot_pdf = out_dir / "llr_kappa_llr_decontamination_stability_sweep.pdf"
    plot_png = out_dir / "llr_kappa_llr_decontamination_stability_sweep.png"
    sweep_df.to_csv(sweep_csv, index=False)
    pair_df.to_csv(pair_csv, index=False)
    _write_plot(sweep_df=sweep_df, pair_df=pair_df, summary=summary, out_pdf=plot_pdf, out_png=plot_png)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": {"step": "8.7.47.18"},
        "input": {
            "points_csv": _safe_rel(points_csv, _ROOT),
            "n_inlier_points": int(len(df)),
            "fit_modes": list(fit_modes),
            "weight_schemes": list(weight_schemes),
            "min_orth_std_values": [float(v) for v in min_orth_values],
            "weight_floor": {
                "station": int(args.weight_floor_station),
                "target": int(args.weight_floor_target),
                "station_target": int(args.weight_floor_station_target),
            },
            "max_weight_cap": float(args.max_weight_cap),
        },
        "summary": summary,
        "outputs": {
            "sweep_csv": _safe_rel(sweep_csv, _ROOT),
            "pairwise_csv": _safe_rel(pair_csv, _ROOT),
            "plot_pdf": _safe_rel(plot_pdf, _ROOT),
            "plot_png": _safe_rel(plot_png, _ROOT),
        },
    }
    metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    produced = [sweep_csv, pair_csv, metrics_json, plot_pdf, plot_png]
    synced = _sync_outputs_to_public(produced, private_root=out_dir, public_root=public_dir)
    print(f"[ok] wrote: {sweep_csv}")
    print(f"[ok] wrote: {pair_csv}")
    print(f"[ok] wrote: {metrics_json}")
    print(f"[ok] wrote: {plot_pdf}")
    print(f"[ok] wrote: {plot_png}")
    print(f"[ok] synced_to_public: {len(synced)} files")
    print(
        f"[summary] runs={summary.get('n_runs')} "
        f"beta_span={summary.get('beta_span')} "
        f"max_pairwise_z={summary.get('max_abs_z_pairwise')} "
        f"stability={summary.get('stability_status')} "
        f"status={summary.get('status')}"
    )
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
