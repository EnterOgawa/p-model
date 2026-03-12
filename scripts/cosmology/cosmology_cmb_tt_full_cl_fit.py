#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_cmb_tt_full_cl_fit.py

Step 8.7.34.11（CMB フル C_ell: TT, ell<=2500）

目的：
- Planck 2018 TT binned spectrum を用いて、少なくとも TT（ell<=2500）の
  full-range 適合度を同一I/Fで固定する。
- 既存のピーク監査（ell1-ell6）とは別に、全binを使った chi2/AIC を
  出力し、Pass/Watch/Reject 判定を明示する。

注意：
- 本スクリプトは「TTの運用監査」用であり、TE/EE を含む full likelihood
  置換を主張しない。
- P-model 側は、baseline best-fit テンプレートに対する
  座標 remap（alpha, delta）+ 線形振幅/オフセットで評価する。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

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


# 関数: `_read_planck_tt` の入出力契約と処理意図を定義する。

def _read_planck_tt(path: Path) -> Dict[str, np.ndarray]:
    arr = np.loadtxt(path)
    # 条件分岐: `arr.ndim != 2 or arr.shape[1] < 5` を満たす経路を評価する。
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError(f"unexpected Planck TT format: {path}")

    ell = arr[:, 0].astype(float)
    dl_obs = arr[:, 1].astype(float)
    err_lo = arr[:, 2].astype(float)
    err_hi = arr[:, 3].astype(float)
    dl_baseline = arr[:, 4].astype(float)
    sigma = np.maximum(0.5 * (np.abs(err_lo) + np.abs(err_hi)), 1.0e-12)
    return {
        "ell": ell,
        "dl_obs": dl_obs,
        "sigma": sigma,
        "dl_baseline": dl_baseline,
    }


# 関数: `_weighted_linear_template_fit` の入出力契約と処理意図を定義する。

def _weighted_linear_template_fit(template: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> Tuple[float, float, np.ndarray, float]:
    x = np.column_stack([template, np.ones_like(template, dtype=float)])
    w = 1.0 / np.maximum(sigma, 1.0e-12)
    xw = x * w[:, None]
    yw = y * w
    beta = np.linalg.lstsq(xw, yw, rcond=None)[0]
    gain = float(beta[0])
    offset = float(beta[1])
    pred = gain * template + offset
    chi2 = float(np.sum(((y - pred) / np.maximum(sigma, 1.0e-12)) ** 2))
    return gain, offset, pred, chi2


# 関数: `_iter_grid` の入出力契約と処理意図を定義する。

def _iter_grid(start: float, stop: float, step: float) -> Iterable[float]:
    # 条件分岐: `step <= 0.0` を満たす経路を評価する。
    if step <= 0.0:
        raise ValueError("step must be positive")

    n = int(np.floor((stop - start) / step + 0.5))
    for i in range(n + 1):
        yield float(start + i * step)


# 関数: `_search_template_remap` の入出力契約と処理意図を定義する。

def _search_template_remap(
    *,
    ell: np.ndarray,
    dl_obs: np.ndarray,
    sigma: np.ndarray,
    dl_baseline: np.ndarray,
    alpha_values: Iterable[float],
    delta_values: Iterable[float],
) -> Dict[str, Any]:
    ell_min = float(np.min(ell))
    ell_max = float(np.max(ell))
    best: Dict[str, Any] = {}
    best_chi2 = float("inf")

    for alpha in alpha_values:
        for delta in delta_values:
            ell_map = np.clip(alpha * ell + delta, ell_min, ell_max)
            template = np.interp(ell_map, ell, dl_baseline)
            gain, offset, pred, chi2 = _weighted_linear_template_fit(template, dl_obs, sigma)
            # 条件分岐: `chi2 < best_chi2` を満たす経路を評価する。
            if chi2 < best_chi2:
                best_chi2 = chi2
                best = {
                    "alpha": float(alpha),
                    "delta_ell": float(delta),
                    "gain": gain,
                    "offset": offset,
                    "pred": pred.copy(),
                    "template": template.copy(),
                    "chi2": chi2,
                }

    # 条件分岐: `not best` を満たす経路を評価する。

    if not best:
        raise RuntimeError("grid search failed to produce a candidate")

    return best


# 関数: `_decision_from_delta_aic` の入出力契約と処理意図を定義する。

def _decision_from_delta_aic(delta_aic: float) -> str:
    # 判定規約: ΔAIC = AIC_baseline - AIC_P（正値でP-model優位）
    # 条件分岐: `delta_aic > 2.0` を満たす経路を評価する。
    if delta_aic > 2.0:
        return "pass"

    # 条件分岐: `delta_aic > -2.0` を満たす経路を評価する。

    if delta_aic > -2.0:
        return "watch"

    return "reject"


# 関数: `_plot_fit` の入出力契約と処理意図を定義する。

def _plot_fit(
    *,
    ell: np.ndarray,
    dl_obs: np.ndarray,
    sigma: np.ndarray,
    dl_baseline: np.ndarray,
    dl_pmodel: np.ndarray,
    out_png: Path,
    out_pdf: Path,
    summary_text: str,
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt
    title_fs = 18.8
    label_fs = 16.4
    residual_label_fs = 15.6
    legend_fs = 14.0
    note_fs = 14.0
    tick_fs = 13.6

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(13.2, 9.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.4]},
    )

    ax0.errorbar(
        ell,
        dl_obs,
        yerr=sigma,
        fmt=".",
        color="#7f7f7f",
        alpha=0.55,
        label="Planck 2018 TT（binned）",
    )
    ax0.plot(ell, dl_baseline, color="#1f77b4", linewidth=1.8, label="baseline（Planck best-fit）")
    ax0.plot(ell, dl_pmodel, color="#d62728", linewidth=1.8, label="P-model（template remap）")
    ax0.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]", fontsize=label_fs)
    ax0.set_title(r"CMB TT full-range fit audit ($\ell \leq 2500$)", fontsize=title_fs)
    ax0.grid(True, linestyle="--", alpha=0.45)
    ax0.legend(fontsize=legend_fs, loc="upper right")
    ax0.text(
        0.02,
        0.03,
        summary_text,
        transform=ax0.transAxes,
        fontsize=note_fs,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#999999", "alpha": 0.92},
    )
    ax0.tick_params(labelsize=tick_fs)

    res_baseline = (dl_obs - dl_baseline) / sigma
    res_pmodel = (dl_obs - dl_pmodel) / sigma
    ax1.axhline(0.0, color="#444444", linewidth=1.1, alpha=0.8)
    ax1.plot(ell, res_baseline, color="#1f77b4", linewidth=1.3, label="baseline residual / σ")
    ax1.plot(ell, res_pmodel, color="#d62728", linewidth=1.3, label="P-model residual / σ")
    ax1.set_xlabel(r"Multipole $\ell$", fontsize=label_fs)
    ax1.set_ylabel("residual / σ", fontsize=residual_label_fs)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(fontsize=legend_fs, loc="upper right")
    ax1.tick_params(labelsize=tick_fs)

    fig.tight_layout()
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
    parser = argparse.ArgumentParser(description="CMB TT full C_ell fit audit (ell<=2500).")
    parser.add_argument(
        "--tt-data",
        default=str(ROOT / "data" / "cosmology" / "planck2018_com_power_spect_tt_binned_r3.01.txt"),
        help="Input Planck TT binned spectrum text.",
    )
    parser.add_argument("--ell-max", type=float, default=2500.0, help="Upper multipole bound for full-fit audit.")
    parser.add_argument("--alpha-min", type=float, default=0.97, help="Grid minimum for remap alpha.")
    parser.add_argument("--alpha-max", type=float, default=1.03, help="Grid maximum for remap alpha.")
    parser.add_argument("--alpha-step", type=float, default=0.0005, help="Grid step for remap alpha.")
    parser.add_argument("--delta-min", type=float, default=-80.0, help="Grid minimum for remap delta_ell.")
    parser.add_argument("--delta-max", type=float, default=80.0, help="Grid maximum for remap delta_ell.")
    parser.add_argument("--delta-step", type=float, default=0.5, help="Grid step for remap delta_ell.")
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

    src = _read_planck_tt(Path(args.tt_data).resolve())
    mask = src["ell"] <= float(args.ell_max)
    ell = src["ell"][mask]
    dl_obs = src["dl_obs"][mask]
    sigma = src["sigma"][mask]
    dl_baseline = src["dl_baseline"][mask]

    alpha_values = tuple(_iter_grid(float(args.alpha_min), float(args.alpha_max), float(args.alpha_step)))
    delta_values = tuple(_iter_grid(float(args.delta_min), float(args.delta_max), float(args.delta_step)))
    best = _search_template_remap(
        ell=ell,
        dl_obs=dl_obs,
        sigma=sigma,
        dl_baseline=dl_baseline,
        alpha_values=alpha_values,
        delta_values=delta_values,
    )

    chi2_baseline = float(np.sum(((dl_obs - dl_baseline) / sigma) ** 2))
    chi2_p = float(best["chi2"])
    k_baseline = 0
    k_p = 4  # alpha, delta_ell, gain, offset
    aic_baseline = chi2_baseline + 2.0 * k_baseline
    aic_p = chi2_p + 2.0 * k_p
    delta_aic = aic_baseline - aic_p
    status = _decision_from_delta_aic(delta_aic)

    dof_baseline = max(int(len(ell) - k_baseline), 1)
    dof_p = max(int(len(ell) - k_p), 1)
    chi2_dof_baseline = float(chi2_baseline / dof_baseline)
    chi2_dof_p = float(chi2_p / dof_p)

    out_private = Path(args.out_private_dir).resolve()
    out_public = Path(args.out_public_dir).resolve()
    out_png = out_private / "cosmology_cmb_tt_full_cl_fit.png"
    out_pdf = out_private / "cosmology_cmb_tt_full_cl_fit.pdf"
    out_json = out_private / "cosmology_cmb_tt_full_cl_fit_metrics.json"
    out_csv = out_private / "cosmology_cmb_tt_full_cl_fit_summary.csv"

    summary_text = (
        f"n_bin={len(ell)}  chi2(ΛCDM/P)={chi2_baseline:.3f}/{chi2_p:.3f}\n"
        f"AIC(ΛCDM/P)={aic_baseline:.3f}/{aic_p:.3f}  ΔAIC={delta_aic:.3f}  status={status}\n"
        f"alpha={best['alpha']:.4f}, delta_ell={best['delta_ell']:+.2f}, gain={best['gain']:.4f}, offset={best['offset']:+.3f}"
    )
    _plot_fit(
        ell=ell,
        dl_obs=dl_obs,
        sigma=sigma,
        dl_baseline=dl_baseline,
        dl_pmodel=np.asarray(best["pred"], dtype=float),
        out_png=out_png,
        out_pdf=out_pdf,
        summary_text=summary_text,
    )

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "8.7.34.11 (CMB full C_ell fit; TT, ell<=2500)",
        "dataset": "Planck 2018 TT binned (COM_PowerSpect_CMB-TT-binned_R3.01)",
        "inputs": {
            "tt_data": str(Path(args.tt_data).resolve()).replace("\\", "/"),
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
                "chi2": chi2_baseline,
                "dof": dof_baseline,
                "chi2_dof": chi2_dof_baseline,
                "aic": aic_baseline,
                "k_params": k_baseline,
            },
            "pmodel_template_remap": {
                "chi2": chi2_p,
                "dof": dof_p,
                "chi2_dof": chi2_dof_p,
                "aic": aic_p,
                "k_params": k_p,
                "alpha": float(best["alpha"]),
                "delta_ell": float(best["delta_ell"]),
                "gain": float(best["gain"]),
                "offset": float(best["offset"]),
            },
            "delta_aic_baseline_minus_pmodel": float(delta_aic),
            "winner": "pmodel" if delta_aic > 0.0 else "baseline_lcdm",
            "status": status,
            "decision_rule": "ΔAIC=AIC_baseline-AIC_P; >2:pass, (-2,2]:watch, <=-2:reject",
        },
        "notes": [
            "本監査は TT binned spectrum の full-range 運用評価であり、TE/EE 同時fitは別タスク。",
            "P-model側は baseline best-fit テンプレートへの remap（alpha,delta）+線形振幅/offset で評価し、第一原理 C_ell 生成器の代替ではない。",
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
            ("n_bin", float(len(ell))),
            ("chi2_baseline_lcdm", chi2_baseline),
            ("chi2_pmodel", chi2_p),
            ("aic_baseline_lcdm", aic_baseline),
            ("aic_pmodel", aic_p),
            ("delta_aic_baseline_minus_pmodel", delta_aic),
            ("alpha", float(best["alpha"])),
            ("delta_ell", float(best["delta_ell"])),
            ("gain", float(best["gain"])),
            ("offset", float(best["offset"])),
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
                "event_type": "cosmology_cmb_tt_full_cl_fit",
                "argv": list(sys.argv),
                "inputs": {"tt_data": str(Path(args.tt_data).resolve()), "ell_max": float(args.ell_max)},
                "outputs": {"png": out_png, "pdf": out_pdf, "metrics_json": out_json, "summary_csv": out_csv},
                "metrics": {
                    "chi2_baseline_lcdm": chi2_baseline,
                    "chi2_pmodel": chi2_p,
                    "delta_aic": delta_aic,
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
