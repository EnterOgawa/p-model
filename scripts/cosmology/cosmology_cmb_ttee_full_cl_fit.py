#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_cmb_ttee_full_cl_fit.py

Step 8.7.35.1（宇宙論 follow-up: CMB TT/TE/EE 同時 full C_ell 監査）

目的：
- Planck 2018 binned spectra（TT/TE/EE）を同一I/Fで同時評価し、
  TT 単独監査で見えた Reject の要因（形状差 vs AICペナルティ）を分解する。
- 判定規約は既存と同一（ΔAIC=AIC_baseline-AIC_P、正値でP優位）を使う。

注意：
- 本スクリプトは運用監査（summary-level）であり、Planck full-likelihood
  置換主張ではない。
- P-model 側は baseline best-fit テンプレートへの座標 remap
  （共有 alpha, delta_ell）と、各チャネルの線形調整（gain/offset）で評価する。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。
def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = ["Yu Gothic", "Meiryo", "BIZ UDGothic", "MS Gothic"]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


# 関数: `_read_channel` の入出力契約と処理意図を定義する。

def _read_channel(path: Path, *, ell_max: float) -> Dict[str, np.ndarray]:
    arr = np.loadtxt(path)
    # 条件分岐: `arr.ndim != 2 or arr.shape[1] < 5` を満たす経路を評価する。
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError(f"unexpected spectrum format: {path}")

    ell = arr[:, 0].astype(float)
    obs = arr[:, 1].astype(float)
    err_lo = arr[:, 2].astype(float)
    err_hi = arr[:, 3].astype(float)
    baseline = arr[:, 4].astype(float)
    sigma = np.maximum(0.5 * (np.abs(err_lo) + np.abs(err_hi)), 1.0e-12)
    mask = ell <= float(ell_max)
    return {
        "ell": ell[mask],
        "obs": obs[mask],
        "sigma": sigma[mask],
        "baseline": baseline[mask],
    }


# 関数: `_weighted_linear_fit` の入出力契約と処理意図を定義する。

def _weighted_linear_fit(template: np.ndarray, obs: np.ndarray, sigma: np.ndarray) -> Tuple[float, float, np.ndarray, float]:
    design = np.column_stack([template, np.ones_like(template, dtype=float)])
    w = 1.0 / np.maximum(sigma, 1.0e-12)
    xw = design * w[:, None]
    yw = obs * w
    beta = np.linalg.lstsq(xw, yw, rcond=None)[0]
    gain = float(beta[0])
    offset = float(beta[1])
    pred = gain * template + offset
    chi2 = float(np.sum(((obs - pred) / np.maximum(sigma, 1.0e-12)) ** 2))
    return gain, offset, pred, chi2


# 関数: `_iter_grid` の入出力契約と処理意図を定義する。

def _iter_grid(start: float, stop: float, step: float) -> Iterable[float]:
    # 条件分岐: `step <= 0.0` を満たす経路を評価する。
    if step <= 0.0:
        raise ValueError("step must be positive")

    n = int(np.floor((stop - start) / step + 0.5))
    for i in range(n + 1):
        yield float(start + i * step)


# 関数: `_search_flexible` の入出力契約と処理意図を定義する。

def _search_flexible(
    *,
    channels: Dict[str, Dict[str, np.ndarray]],
    alpha_values: Iterable[float],
    delta_values: Iterable[float],
) -> Dict[str, Any]:
    best_chi2 = float("inf")
    best: Dict[str, Any] = {}

    for alpha in alpha_values:
        for delta in delta_values:
            total = 0.0
            per: Dict[str, Dict[str, Any]] = {}
            for name, payload in channels.items():
                ell = payload["ell"]
                obs = payload["obs"]
                sigma = payload["sigma"]
                baseline = payload["baseline"]
                ell_map = np.clip(alpha * ell + delta, float(np.min(ell)), float(np.max(ell)))
                template = np.interp(ell_map, ell, baseline)
                gain, offset, pred, chi2 = _weighted_linear_fit(template, obs, sigma)
                total += float(chi2)
                per[name] = {
                    "gain": gain,
                    "offset": offset,
                    "chi2": float(chi2),
                    "pred": pred,
                }

            # 条件分岐: `total < best_chi2` を満たす経路を評価する。

            if total < best_chi2:
                best_chi2 = total
                best = {
                    "alpha": float(alpha),
                    "delta_ell": float(delta),
                    "chi2_total": float(total),
                    "per_channel": per,
                }

    # 条件分岐: `not best` を満たす経路を評価する。

    if not best:
        raise RuntimeError("flexible search failed")

    return best


# 関数: `_search_shape_only` の入出力契約と処理意図を定義する。

def _search_shape_only(
    *,
    channels: Dict[str, Dict[str, np.ndarray]],
    alpha_values: Iterable[float],
    delta_values: Iterable[float],
) -> Dict[str, Any]:
    best_chi2 = float("inf")
    best: Dict[str, Any] = {}

    for alpha in alpha_values:
        for delta in delta_values:
            total = 0.0
            per: Dict[str, float] = {}
            for name, payload in channels.items():
                ell = payload["ell"]
                obs = payload["obs"]
                sigma = payload["sigma"]
                baseline = payload["baseline"]
                ell_map = np.clip(alpha * ell + delta, float(np.min(ell)), float(np.max(ell)))
                pred = np.interp(ell_map, ell, baseline)
                chi2 = float(np.sum(((obs - pred) / np.maximum(sigma, 1.0e-12)) ** 2))
                total += chi2
                per[name] = chi2

            # 条件分岐: `total < best_chi2` を満たす経路を評価する。

            if total < best_chi2:
                best_chi2 = total
                best = {
                    "alpha": float(alpha),
                    "delta_ell": float(delta),
                    "chi2_total": float(total),
                    "per_channel": per,
                }

    # 条件分岐: `not best` を満たす経路を評価する。

    if not best:
        raise RuntimeError("shape-only search failed")

    return best


# 関数: `_decision_from_delta_aic` の入出力契約と処理意図を定義する。

def _decision_from_delta_aic(delta_aic: float) -> str:
    # 判定規約: ΔAIC=AIC_baseline-AIC_P（正値でP-model優位）
    # 条件分岐: `delta_aic > 2.0` を満たす経路を評価する。
    if delta_aic > 2.0:
        return "pass"

    # 条件分岐: `delta_aic > -2.0` を満たす経路を評価する。

    if delta_aic > -2.0:
        return "watch"

    return "reject"


# 関数: `_format_summary_multiline` の入出力契約と処理意図を定義する。

def _format_summary_multiline(summary_lines: List[str]) -> str:
    line_groups = [
        summary_lines[0:3],
        summary_lines[3:6],
        summary_lines[6:8],
    ]
    return "\n".join(" / ".join(group) for group in line_groups if group)


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(
    *,
    channels: Dict[str, Dict[str, np.ndarray]],
    flexible: Dict[str, Any],
    baseline_chi2: Dict[str, float],
    out_png: Path,
    out_pdf: Path,
    summary_lines: List[str],
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.2, 10.8))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.85, 1.85, 1.85, 1.55], hspace=0.44)

    order = [("TT", "CMB TT"), ("TE", "CMB TE"), ("EE", "CMB EE")]
    for idx, (name, title) in enumerate(order):
        ax = fig.add_subplot(gs[idx, 0])
        payload = channels[name]
        ell = payload["ell"]
        obs = payload["obs"]
        sigma = payload["sigma"]
        baseline = payload["baseline"]
        pred = np.asarray(flexible["per_channel"][name]["pred"], dtype=float)
        ax.errorbar(ell, obs, yerr=sigma, fmt=".", color="#808080", alpha=0.45, label=f"{title} observed")
        ax.plot(ell, baseline, color="#1f77b4", linewidth=1.5, label=f"{title} baseline")
        ax.plot(ell, pred, color="#d62728", linewidth=1.5, label=f"{title} P-model")
        ax.set_ylabel(r"$D_\ell$", fontsize=15.8)
        ax.set_title(title, fontsize=17.4)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=14.2, loc="upper right")
        ax.tick_params(labelsize=13.8)
        # 条件分岐: `idx == len(order) - 1` を満たす経路を評価する。
        if idx == len(order) - 1:
            ax.set_xlabel(r"Multipole $\ell$", fontsize=15.8)

    ax_bar = fig.add_subplot(gs[3, 0])
    labels = [x[0] for x in order]
    x = np.arange(len(labels), dtype=float)
    base_vals = [baseline_chi2[k] for k in labels]
    p_vals = [float(flexible["per_channel"][k]["chi2"]) for k in labels]
    width = 0.37
    ax_bar.bar(x - width / 2.0, base_vals, width, label="baseline χ²", color="#4e79a7")
    ax_bar.bar(x + width / 2.0, p_vals, width, label="P-model χ²", color="#e15759")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylabel("χ² contribution", fontsize=15.8)
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_bar.legend(fontsize=14.2, loc="upper right")
    ax_bar.tick_params(labelsize=13.8)

    fig.suptitle("CMB TT/TE/EE simultaneous full-range fit audit", fontsize=18.0)
    summary_text = _format_summary_multiline(summary_lines)
    fig.text(
        0.5,
        0.014,
        summary_text,
        ha="center",
        va="bottom",
        fontsize=12.8,
        linespacing=1.08,
    )
    fig.tight_layout(rect=(0.0, 0.108, 1.0, 0.965))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: Iterable[Tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for key, value in rows:
            writer.writerow([key, value])


# 関数: `_copy_to_public` の入出力契約と処理意図を定義する。

def _copy_to_public(*, private_files: Iterable[Path], public_dir: Path) -> None:
    public_dir.mkdir(parents=True, exist_ok=True)
    for src in private_files:
        shutil.copy2(src, public_dir / src.name)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    parser = argparse.ArgumentParser(description="CMB TT/TE/EE simultaneous full C_ell fit audit.")
    parser.add_argument(
        "--tt-data",
        default=str(ROOT / "data" / "cosmology" / "planck2018_com_power_spect_tt_binned_r3.01.txt"),
        help="Input TT binned spectrum.",
    )
    parser.add_argument(
        "--te-data",
        default=str(ROOT / "data" / "cosmology" / "planck2018_com_power_spect_te_binned_r3.02.txt"),
        help="Input TE binned spectrum.",
    )
    parser.add_argument(
        "--ee-data",
        default=str(ROOT / "data" / "cosmology" / "planck2018_com_power_spect_ee_binned_r3.02.txt"),
        help="Input EE binned spectrum.",
    )
    parser.add_argument("--ell-max", type=float, default=2500.0, help="Upper ell bound for channel loading.")
    parser.add_argument("--alpha-min", type=float, default=0.97, help="Grid minimum for alpha.")
    parser.add_argument("--alpha-max", type=float, default=1.03, help="Grid maximum for alpha.")
    parser.add_argument("--alpha-step", type=float, default=0.001, help="Grid step for alpha.")
    parser.add_argument("--delta-min", type=float, default=-80.0, help="Grid minimum for delta_ell.")
    parser.add_argument("--delta-max", type=float, default=80.0, help="Grid maximum for delta_ell.")
    parser.add_argument("--delta-step", type=float, default=1.0, help="Grid step for delta_ell.")
    parser.add_argument(
        "--out-private-dir",
        default=str(ROOT / "output" / "private" / "cosmology"),
        help="Output directory (private).",
    )
    parser.add_argument(
        "--out-public-dir",
        default=str(ROOT / "output" / "public" / "cosmology"),
        help="Output directory (public mirror).",
    )
    args = parser.parse_args()

    channels: Dict[str, Dict[str, np.ndarray]] = {
        "TT": _read_channel(Path(args.tt_data).resolve(), ell_max=float(args.ell_max)),
        "TE": _read_channel(Path(args.te_data).resolve(), ell_max=float(args.ell_max)),
        "EE": _read_channel(Path(args.ee_data).resolve(), ell_max=float(args.ell_max)),
    }
    alpha_values = tuple(_iter_grid(float(args.alpha_min), float(args.alpha_max), float(args.alpha_step)))
    delta_values = tuple(_iter_grid(float(args.delta_min), float(args.delta_max), float(args.delta_step)))

    baseline_chi2: Dict[str, float] = {}
    n_total = 0
    chi2_baseline_total = 0.0
    for name, payload in channels.items():
        obs = payload["obs"]
        sigma = payload["sigma"]
        baseline = payload["baseline"]
        chi2 = float(np.sum(((obs - baseline) / np.maximum(sigma, 1.0e-12)) ** 2))
        baseline_chi2[name] = chi2
        chi2_baseline_total += chi2
        n_total += int(len(obs))

    flexible = _search_flexible(channels=channels, alpha_values=alpha_values, delta_values=delta_values)
    shape_only = _search_shape_only(channels=channels, alpha_values=alpha_values, delta_values=delta_values)

    k_baseline = 0
    k_flexible = 8  # shared alpha/delta + 3*(gain/offset)
    aic_baseline = float(chi2_baseline_total + 2.0 * k_baseline)
    aic_flexible = float(flexible["chi2_total"] + 2.0 * k_flexible)
    delta_aic = float(aic_baseline - aic_flexible)
    status = _decision_from_delta_aic(delta_aic)

    dof_baseline = max(int(n_total - k_baseline), 1)
    dof_flexible = max(int(n_total - k_flexible), 1)

    shape_improvement = float(chi2_baseline_total - shape_only["chi2_total"])
    scaling_improvement = float(shape_only["chi2_total"] - flexible["chi2_total"])
    chi2_improvement_total = float(chi2_baseline_total - flexible["chi2_total"])
    aic_penalty = float(2.0 * k_flexible)

    out_private = Path(args.out_private_dir).resolve()
    out_public = Path(args.out_public_dir).resolve()
    out_png = out_private / "cosmology_cmb_ttee_full_cl_fit.png"
    out_pdf = out_private / "cosmology_cmb_ttee_full_cl_fit.pdf"
    out_json = out_private / "cosmology_cmb_ttee_full_cl_fit_metrics.json"
    out_csv = out_private / "cosmology_cmb_ttee_full_cl_fit_summary.csv"

    summary_lines = [
        f"n_bin_total={n_total}",
        f"chi2(ΛCDM/P)={chi2_baseline_total:.3f}/{flexible['chi2_total']:.3f}",
        f"AIC(ΛCDM/P)={aic_baseline:.3f}/{aic_flexible:.3f}",
        f"ΔAIC={delta_aic:.3f}  status={status}",
        f"shape_gain={shape_improvement:.3f}",
        f"channel_scaling_gain={scaling_improvement:.3f}",
        f"AIC_penalty={aic_penalty:.3f}",
        f"alpha={flexible['alpha']:.4f}, delta_ell={flexible['delta_ell']:+.2f}",
    ]
    _plot(
        channels=channels,
        flexible=flexible,
        baseline_chi2=baseline_chi2,
        out_png=out_png,
        out_pdf=out_pdf,
        summary_lines=summary_lines,
    )

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "8.7.35.1 (CMB TT/TE/EE simultaneous full C_ell fit audit)",
        "dataset": {
            "TT": "Planck 2018 TT binned (R3.01)",
            "TE": "Planck 2018 TE binned (R3.02)",
            "EE": "Planck 2018 EE binned (R3.02)",
        },
        "inputs": {
            "tt_data": str(Path(args.tt_data).resolve()).replace("\\", "/"),
            "te_data": str(Path(args.te_data).resolve()).replace("\\", "/"),
            "ee_data": str(Path(args.ee_data).resolve()).replace("\\", "/"),
            "ell_max": float(args.ell_max),
            "grid": {
                "alpha_min": float(args.alpha_min),
                "alpha_max": float(args.alpha_max),
                "alpha_step": float(args.alpha_step),
                "delta_min": float(args.delta_min),
                "delta_max": float(args.delta_max),
                "delta_step": float(args.delta_step),
                "n_alpha": len(alpha_values),
                "n_delta": len(delta_values),
            },
        },
        "fit": {
            "baseline_lcdm": {
                "n_bin_total": int(n_total),
                "chi2_total": float(chi2_baseline_total),
                "chi2_dof": float(chi2_baseline_total / dof_baseline),
                "aic": float(aic_baseline),
                "k_params": int(k_baseline),
                "per_channel_chi2": baseline_chi2,
            },
            "pmodel_flexible": {
                "chi2_total": float(flexible["chi2_total"]),
                "chi2_dof": float(flexible["chi2_total"] / dof_flexible),
                "aic": float(aic_flexible),
                "k_params": int(k_flexible),
                "alpha": float(flexible["alpha"]),
                "delta_ell": float(flexible["delta_ell"]),
                "per_channel": {
                    key: {
                        "chi2": float(value["chi2"]),
                        "gain": float(value["gain"]),
                        "offset": float(value["offset"]),
                    }
                    for key, value in flexible["per_channel"].items()
                },
            },
            "pmodel_shape_only": {
                "chi2_total": float(shape_only["chi2_total"]),
                "alpha": float(shape_only["alpha"]),
                "delta_ell": float(shape_only["delta_ell"]),
                "per_channel_chi2": shape_only["per_channel"],
            },
            "delta_aic_baseline_minus_pmodel": float(delta_aic),
            "winner": "pmodel" if delta_aic > 0.0 else "baseline_lcdm",
            "status": status,
            "decision_rule": "ΔAIC=AIC_baseline-AIC_P; >2:pass, (-2,2]:watch, <=-2:reject",
            "decomposition": {
                "chi2_improvement_total": float(chi2_improvement_total),
                "shape_improvement": float(shape_improvement),
                "channel_scaling_improvement": float(scaling_improvement),
                "aic_penalty": float(aic_penalty),
                "net_delta_aic": float(delta_aic),
            },
        },
        "notes": [
            "本監査は TT/TE/EE binned spectra の同時評価であり、Planck full-likelihood の代替ではない。",
            "P-model側は baseline テンプレートの共有 remap（alpha,delta）とチャネル毎の線形調整（gain/offset）で summary-level 適合度を評価する。",
        ],
        "outputs": {
            "png": str(out_png).replace("\\", "/"),
            "pdf": str(out_pdf).replace("\\", "/"),
            "metrics_json": str(out_json).replace("\\", "/"),
            "summary_csv": str(out_csv).replace("\\", "/"),
        },
    }
    _write_json(out_json, payload)
    _write_csv(
        out_csv,
        rows=(
            ("n_bin_total", float(n_total)),
            ("chi2_baseline_lcdm", float(chi2_baseline_total)),
            ("chi2_pmodel_flexible", float(flexible["chi2_total"])),
            ("chi2_pmodel_shape_only", float(shape_only["chi2_total"])),
            ("aic_baseline_lcdm", float(aic_baseline)),
            ("aic_pmodel_flexible", float(aic_flexible)),
            ("delta_aic_baseline_minus_pmodel", float(delta_aic)),
            ("shape_improvement", float(shape_improvement)),
            ("channel_scaling_improvement", float(scaling_improvement)),
            ("aic_penalty", float(aic_penalty)),
            ("alpha", float(flexible["alpha"])),
            ("delta_ell", float(flexible["delta_ell"])),
        ),
    )
    _copy_to_public(private_files=(out_png, out_pdf, out_json, out_csv), public_dir=out_public)

    print(f"[ok] png : {out_png}")
    print(f"[ok] pdf : {out_pdf}")
    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] public mirror: {out_public}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_cmb_ttee_full_cl_fit",
                "argv": list(sys.argv),
                "inputs": {
                    "tt_data": str(Path(args.tt_data).resolve()),
                    "te_data": str(Path(args.te_data).resolve()),
                    "ee_data": str(Path(args.ee_data).resolve()),
                    "ell_max": float(args.ell_max),
                },
                "outputs": {
                    "png": out_png,
                    "pdf": out_pdf,
                    "metrics_json": out_json,
                    "summary_csv": out_csv,
                },
                "metrics": {
                    "chi2_baseline_lcdm": float(chi2_baseline_total),
                    "chi2_pmodel_flexible": float(flexible["chi2_total"]),
                    "delta_aic": float(delta_aic),
                    "status": status,
                },
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
