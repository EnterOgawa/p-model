#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_distance_indicator_rederivation_global_prior_search.py

Phase 16（宇宙論）/ Step 16.4：
距離指標の再導出候補探索（candidate_search）では、DDR一次ソースごとに
「最も都合の良い」一次ソース拘束（不透明度 α / 標準光源進化 s_L）を選び得るため、
物理的に一貫した“単一の prior（α, s_L）”が存在するかが不明確になり得る。

本スクリプトでは、α と s_L の一次ソースを **1つずつ固定**した上で、
全DDR行について WLS（DDR + BAO(s_R) + α + s_L + p_t + p_e）を回し、
「全DDR行での最悪 max|z|」が最小になる prior の組合せを探索する。

入力（固定）:
  - data/cosmology/distance_duality_constraints.json
  - data/cosmology/cosmic_opacity_constraints.json
  - data/cosmology/sn_standard_candle_evolution_constraints.json
  - data/cosmology/sn_time_dilation_constraints.json
  - data/cosmology/cmb_temperature_scaling_constraints.json
  - output/private/cosmology/cosmology_bao_scaled_distance_fit_metrics.json
  - （任意）output/private/cosmology/cosmology_distance_duality_systematics_envelope_metrics.json（DDR σ_cat）

出力（固定名）:
  - output/private/cosmology/cosmology_distance_indicator_rederivation_global_prior_search.png
  - output/private/cosmology/cosmology_distance_indicator_rederivation_global_prior_search_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402
from scripts.cosmology import (  # noqa: E402
    cosmology_distance_indicator_rederivation_candidate_search as cand,
)


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
        pass


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `_parse_scales` の入出力契約と処理意図を定義する。

def _parse_scales(s: str) -> List[float]:
    out: List[float] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        # 条件分岐: `not tok` を満たす経路を評価する。
        if not tok:
            continue

        try:
            v = float(tok)
        except Exception:
            continue

        # 条件分岐: `not (v > 0.0 and math.isfinite(v))` を満たす経路を評価する。

        if not (v > 0.0 and math.isfinite(v)):
            continue

        out.append(v)

    return sorted(set(out))


# 関数: `_crossing_x_log` の入出力契約と処理意図を定義する。

def _crossing_x_log(xs: Sequence[float], ys: Sequence[float], threshold: float) -> Optional[float]:
    # 条件分岐: `not xs or len(xs) != len(ys)` を満たす経路を評価する。
    if not xs or len(xs) != len(ys):
        return None

    # 条件分岐: `not math.isfinite(float(threshold))` を満たす経路を評価する。

    if not math.isfinite(float(threshold)):
        return None

    xs_f = [float(x) for x in xs]
    ys_f = [float(y) for y in ys]
    # 条件分岐: `any((x <= 0.0 or not math.isfinite(x)) for x in xs_f)` を満たす経路を評価する。
    if any((x <= 0.0 or not math.isfinite(x)) for x in xs_f):
        return None

    # 条件分岐: `math.isfinite(ys_f[0]) and ys_f[0] <= float(threshold)` を満たす経路を評価する。

    if math.isfinite(ys_f[0]) and ys_f[0] <= float(threshold):
        return xs_f[0]

    for i in range(1, len(xs_f)):
        x0, x1 = xs_f[i - 1], xs_f[i]
        y0, y1 = ys_f[i - 1], ys_f[i]
        # 条件分岐: `not (math.isfinite(y0) and math.isfinite(y1))` を満たす経路を評価する。
        if not (math.isfinite(y0) and math.isfinite(y1)):
            continue

        # 条件分岐: `(y0 - threshold) == 0.0` を満たす経路を評価する。

        if (y0 - threshold) == 0.0:
            return x0

        # 条件分岐: `(y0 - threshold) * (y1 - threshold) > 0.0` を満たす経路を評価する。

        if (y0 - threshold) * (y1 - threshold) > 0.0:
            continue

        # 条件分岐: `y1 == y0` を満たす経路を評価する。

        if y1 == y0:
            return x1

        t = (threshold - y0) / (y1 - y0)
        lx0 = math.log10(x0)
        lx1 = math.log10(x1)
        lx = lx0 + float(t) * (lx1 - lx0)
        return float(10.0**lx)

    return None


# 関数: `_classify` の入出力契約と処理意図を定義する。

def _classify(v: float) -> str:
    # 条件分岐: `not math.isfinite(float(v))` を満たす経路を評価する。
    if not math.isfinite(float(v)):
        return "na"

    # 条件分岐: `v <= 1.0` を満たす経路を評価する。

    if v <= 1.0:
        return "ok"

    # 条件分岐: `v <= 3.0` を満たす経路を評価する。

    if v <= 3.0:
        return "warn"

    return "ng"


# 関数: `_plot_heatmap` の入出力契約と処理意図を定義する。

def _plot_heatmap(
    ax: Any,
    *,
    title: str,
    z_grid: np.ndarray,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    best_ij: Tuple[int, int],
    vmax: float,
) -> None:
    import matplotlib.colors as mcolors

    bounds = [0.0, 1.0, 3.0, vmax]
    cmap = mcolors.ListedColormap(["#2ca02c", "#ffbf00", "#d62728"])
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(np.clip(z_grid, 0.0, vmax), cmap=cmap, norm=norm, aspect="auto")
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(list(x_labels), rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(list(y_labels), fontsize=8)

    for iy in range(z_grid.shape[0]):
        for ix in range(z_grid.shape[1]):
            v = float(z_grid[iy, ix])
            ax.text(ix, iy, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#111111")

    by, bx = best_ij
    ax.scatter([bx], [by], marker="*", s=120, c="#000000")

    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=1.0, alpha=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)


# 関数: `_evaluate_global` の入出力契約と処理意図を定義する。

def _evaluate_global(
    *,
    ddr: Sequence[cand.DDRConstraint],
    opacity_list: Sequence[cand.GaussianConstraint],
    candle_list: Sequence[cand.GaussianConstraint],
    p_t: cand.GaussianConstraint,
    p_e: cand.GaussianConstraint,
    sR_bao: float,
    sR_bao_sigma: float,
) -> Dict[str, Any]:
    # Grid over opacity (y) x candle (x)
    z_grid = np.full((len(opacity_list), len(candle_list)), np.nan, dtype=float)
    meta: List[List[Dict[str, Any]]] = [[{} for _ in candle_list] for _ in opacity_list]
    best: Optional[Tuple[float, int, int]] = None

    for iy, op in enumerate(opacity_list):
        for ix, cd in enumerate(candle_list):
            per_ddr: List[Dict[str, Any]] = []
            worst_v = -1.0
            worst_row: Optional[Dict[str, Any]] = None
            ok_1 = 0
            ok_3 = 0
            for d in ddr:
                fit = cand._wls_fit(
                    ddr=d,
                    sR_bao=sR_bao,
                    sR_bao_sigma=sR_bao_sigma,
                    opacity=op,
                    candle=cd,
                    p_t=p_t,
                    p_e=p_e,
                )
                v = float(fit.get("max_abs_z", float("nan")))
                # 条件分岐: `math.isfinite(v)` を満たす経路を評価する。
                if math.isfinite(v):
                    # 条件分岐: `v <= 1.0` を満たす経路を評価する。
                    if v <= 1.0:
                        ok_1 += 1

                    # 条件分岐: `v <= 3.0` を満たす経路を評価する。

                    if v <= 3.0:
                        ok_3 += 1

                    # 条件分岐: `v > worst_v` を満たす経路を評価する。

                    if v > worst_v:
                        worst_v = v
                        worst_row = {
                            "ddr_id": d.id,
                            "ddr_short_label": d.short_label,
                            "max_abs_z": v,
                            "limiting_observation": str(fit.get("limiting_observation") or "na"),
                        }

                per_ddr.append(
                    {
                        "ddr_id": d.id,
                        "ddr_short_label": d.short_label,
                        "max_abs_z": v,
                        "limiting_observation": str(fit.get("limiting_observation") or "na"),
                    }
                )

            z_grid[iy, ix] = worst_v if worst_v >= 0.0 else float("nan")
            meta[iy][ix] = {
                "opacity": {"id": op.id, "short_label": op.short_label, "mean": op.mean, "sigma": op.sigma},
                "candle": {"id": cd.id, "short_label": cd.short_label, "mean": cd.mean, "sigma": cd.sigma},
                "aggregate": {
                    "worst_max_abs_z": float(z_grid[iy, ix]),
                    "worst_row": worst_row,
                    "rows_total": len(ddr),
                    "rows_max_abs_z_le_1": ok_1,
                    "rows_max_abs_z_le_3": ok_3,
                },
                "per_ddr": per_ddr,
            }

            # 条件分岐: `math.isfinite(float(z_grid[iy, ix]))` を満たす経路を評価する。
            if math.isfinite(float(z_grid[iy, ix])):
                key = float(z_grid[iy, ix])
                # 条件分岐: `best is None or key < best[0]` を満たす経路を評価する。
                if best is None or key < best[0]:
                    best = (key, iy, ix)

    # 条件分岐: `best is None` を満たす経路を評価する。

    if best is None:
        best = (float("nan"), 0, 0)

    # Top-k pairs by worst_max_abs_z (then by -rows<=1, then id)

    flat: List[Tuple[float, int, int, int]] = []
    for iy in range(len(opacity_list)):
        for ix in range(len(candle_list)):
            v = float(z_grid[iy, ix])
            # 条件分岐: `not math.isfinite(v)` を満たす経路を評価する。
            if not math.isfinite(v):
                continue

            agg = meta[iy][ix].get("aggregate") if isinstance(meta[iy][ix], dict) else {}
            ok1 = int(agg.get("rows_max_abs_z_le_1", 0)) if isinstance(agg, dict) else 0
            flat.append((v, -ok1, iy, ix))

    flat.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    top_pairs = []
    for v, neg_ok1, iy, ix in flat[:10]:
        m = meta[iy][ix]
        top_pairs.append(
            {
                "worst_max_abs_z": float(v),
                "rows_max_abs_z_le_1": int(-neg_ok1),
                "opacity": m.get("opacity"),
                "candle": m.get("candle"),
                "worst_row": (m.get("aggregate") or {}).get("worst_row"),
            }
        )

    return {"z_grid": z_grid, "meta": meta, "best_ij": (int(best[1]), int(best[2])), "top_pairs": top_pairs}


# 関数: `_evaluate_best_pair` の入出力契約と処理意図を定義する。

def _evaluate_best_pair(
    *,
    ddr: Sequence[cand.DDRConstraint],
    opacity_list: Sequence[cand.GaussianConstraint],
    candle_list: Sequence[cand.GaussianConstraint],
    p_t: cand.GaussianConstraint,
    p_e: cand.GaussianConstraint,
    sR_bao: float,
    sR_bao_sigma: float,
) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None

    for op in opacity_list:
        for cd in candle_list:
            ok_1 = 0
            ok_3 = 0
            worst_v = -1.0
            worst_row: Optional[Dict[str, Any]] = None
            valid = True

            for d in ddr:
                fit = cand._wls_fit(
                    ddr=d,
                    sR_bao=sR_bao,
                    sR_bao_sigma=sR_bao_sigma,
                    opacity=op,
                    candle=cd,
                    p_t=p_t,
                    p_e=p_e,
                )
                v = float(fit.get("max_abs_z", float("nan")))
                # 条件分岐: `not math.isfinite(v)` を満たす経路を評価する。
                if not math.isfinite(v):
                    valid = False
                    break

                # 条件分岐: `v <= 1.0` を満たす経路を評価する。

                if v <= 1.0:
                    ok_1 += 1

                # 条件分岐: `v <= 3.0` を満たす経路を評価する。

                if v <= 3.0:
                    ok_3 += 1

                # 条件分岐: `v > worst_v` を満たす経路を評価する。

                if v > worst_v:
                    worst_v = v
                    worst_row = {
                        "ddr_id": d.id,
                        "ddr_short_label": d.short_label,
                        "max_abs_z": v,
                        "limiting_observation": str(fit.get("limiting_observation") or "na"),
                    }

            # 条件分岐: `not valid or worst_v < 0.0` を満たす経路を評価する。

            if not valid or worst_v < 0.0:
                continue

            candidate = {
                "worst_max_abs_z": float(worst_v),
                "rows_total": int(len(ddr)),
                "rows_max_abs_z_le_1": int(ok_1),
                "rows_max_abs_z_le_3": int(ok_3),
                "worst_row": worst_row,
                "opacity": {"id": op.id, "short_label": op.short_label, "mean": op.mean, "sigma": op.sigma},
                "candle": {"id": cd.id, "short_label": cd.short_label, "mean": cd.mean, "sigma": cd.sigma},
            }
            # 条件分岐: `best is None or float(candidate["worst_max_abs_z"]) < float(best["worst_max_a...` を満たす経路を評価する。
            if best is None or float(candidate["worst_max_abs_z"]) < float(best["worst_max_abs_z"]):
                best = candidate

    return best or {
        "worst_max_abs_z": float("nan"),
        "rows_total": int(len(ddr)),
        "rows_max_abs_z_le_1": 0,
        "rows_max_abs_z_le_3": 0,
        "worst_row": None,
        "opacity": None,
        "candle": None,
    }


# 関数: `_plot_bao_sigma_scan` の入出力契約と処理意図を定義する。

def _plot_bao_sigma_scan(
    *,
    out_png: Path,
    scales: Sequence[float],
    all_best: Sequence[Dict[str, Any]],
    ind_best: Sequence[Dict[str, Any]],
    f_all_1sigma: Optional[float],
    f_ind_1sigma: Optional[float],
    sR_bao: float,
    sR_bao_sigma_base: float,
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    xs = [float(s) for s in scales]
    y_all = [float(x.get("worst_max_abs_z", float("nan"))) for x in all_best]
    y_ind = [float(x.get("worst_max_abs_z", float("nan"))) for x in ind_best]

    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    ax.set_xscale("log")

    ax.plot(xs, y_all, marker="o", linewidth=2.0, color="#1f77b4", label="全候補: best（単一prior）")
    ax.plot(xs, y_ind, marker="o", linewidth=2.0, color="#ff7f0e", label="独立一次ソースのみ: best（単一prior）")

    ax.axhline(1.0, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.7, label="1σ境界")
    ax.axhline(3.0, color="#7f7f7f", linestyle="--", linewidth=1.2, alpha=0.6, label="3σ境界")

    # 条件分岐: `f_all_1sigma is not None and math.isfinite(float(f_all_1sigma)) and float(f_a...` を満たす経路を評価する。
    if f_all_1sigma is not None and math.isfinite(float(f_all_1sigma)) and float(f_all_1sigma) > 0.0:
        ax.axvline(float(f_all_1sigma), color="#1f77b4", linestyle=":", linewidth=1.5, alpha=0.7)
        ax.text(
            float(f_all_1sigma),
            1.02,
            f"全候補: f≈{float(f_all_1sigma):.2g}",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=9,
            color="#1f77b4",
        )

    # 条件分岐: `f_ind_1sigma is not None and math.isfinite(float(f_ind_1sigma)) and float(f_i...` を満たす経路を評価する。

    if f_ind_1sigma is not None and math.isfinite(float(f_ind_1sigma)) and float(f_ind_1sigma) > 0.0:
        ax.axvline(float(f_ind_1sigma), color="#ff7f0e", linestyle=":", linewidth=1.5, alpha=0.7)
        ax.text(
            float(f_ind_1sigma),
            1.02,
            f"独立: f≈{float(f_ind_1sigma):.2g}",
            rotation=90,
            va="bottom",
            ha="left",
            fontsize=9,
            color="#ff7f0e",
        )

    ax.set_xlabel("BAO σスケール f（s_R の σ→fσ）", fontsize=11)
    ax.set_ylabel("全DDRの worst max|z|（単一prior: α と s_L を固定して最適化）", fontsize=11)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, frameon=True)

    fig.suptitle("宇宙論（距離指標）：単一prior（α,s_L）での整合度と BAO(s_R) 緩和の関係", fontsize=14)
    fig.text(
        0.5,
        0.02,
        f"注：各点はその f で『全DDRの worst max|z|』が最小となる prior（α,s_L）を選んだ結果。"
        f" BAO prior は BOSS DR12 fit（s_R={sR_bao:.3f}, σ={sR_bao_sigma_base:.3f}）を基準に f 倍。",
        ha="center",
        fontsize=10,
        color="#333333",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ddr-sigma-policy",
        type=str,
        default="category_sys",
        choices=["raw", "category_sys"],
        help="DDR ε0 の σ を raw で使うか、カテゴリ系統 σ_cat を取り込むか。",
    )
    ap.add_argument(
        "--bao-sigma-scale",
        type=float,
        default=1.0,
        help="BAO s_R の σ を f 倍する（soft constraint）。default=1.0",
    )
    ap.add_argument(
        "--bao-sigma-scale-scan",
        type=str,
        default="1,1.5,2,2.5,3,4,5,6,8,10,12,15",
        help="BAO σスケール f のスキャン（カンマ区切り）。空文字でスキップ。",
    )
    args = ap.parse_args(argv)

    data_dir = _ROOT / "data" / "cosmology"
    out_dir = _ROOT / "output" / "private" / "cosmology"

    in_ddr = data_dir / "distance_duality_constraints.json"
    in_opacity = data_dir / "cosmic_opacity_constraints.json"
    in_candle = data_dir / "sn_standard_candle_evolution_constraints.json"
    in_pt = data_dir / "sn_time_dilation_constraints.json"
    in_pe = data_dir / "cmb_temperature_scaling_constraints.json"
    in_bao_fit = out_dir / "cosmology_bao_scaled_distance_fit_metrics.json"

    for p in (in_ddr, in_opacity, in_candle, in_pt, in_pe, in_bao_fit):
        # 条件分岐: `not p.exists()` を満たす経路を評価する。
        if not p.exists():
            raise FileNotFoundError(f"missing input: {p}")

    # 条件分岐: `not (float(args.bao_sigma_scale) > 0.0 and math.isfinite(float(args.bao_sigma...` を満たす経路を評価する。

    if not (float(args.bao_sigma_scale) > 0.0 and math.isfinite(float(args.bao_sigma_scale))):
        raise ValueError("--bao-sigma-scale must be positive and finite")

    ddr_rows = _read_json(in_ddr).get("constraints") or []
    opacity_rows = _read_json(in_opacity).get("constraints") or []
    candle_rows = _read_json(in_candle).get("constraints") or []
    pt_rows = _read_json(in_pt).get("constraints") or []
    pe_rows = _read_json(in_pe).get("constraints") or []
    bao_fit = _read_json(in_bao_fit)

    # DDR sigma policy
    ddr_sigma_policy = str(args.ddr_sigma_policy)
    ddr_env_path = out_dir / "cosmology_distance_duality_systematics_envelope_metrics.json"
    ddr_env = cand._load_ddr_systematics_envelope(ddr_env_path) if ddr_sigma_policy == "category_sys" else {}
    ddr = [
        cand._apply_ddr_sigma_policy(cand.DDRConstraint.from_json(r), policy=ddr_sigma_policy, envelope=ddr_env)
        for r in ddr_rows
    ]
    # 条件分岐: `not ddr` を満たす経路を評価する。
    if not ddr:
        raise ValueError("no DDR constraints found")

    opacity_all = cand._as_gaussian_list(opacity_rows, mean_key="alpha_opacity", sigma_key="alpha_opacity_sigma")
    candle_all = cand._as_gaussian_list(candle_rows, mean_key="s_L", sigma_key="s_L_sigma")
    # 条件分岐: `not opacity_all or not candle_all` を満たす経路を評価する。
    if not opacity_all or not candle_all:
        raise ValueError("no opacity/candle constraints found")

    opacity_ind = [c for c in opacity_all if c.is_independent()]
    candle_ind = [c for c in candle_all if c.is_independent()]

    pt_all = cand._as_gaussian_list(pt_rows, mean_key="p_t", sigma_key="p_t_sigma")
    pe_all_from_beta = cand._as_pT_constraints(pe_rows)
    # 条件分岐: `not pt_all` を満たす経路を評価する。
    if not pt_all:
        raise ValueError("no SN time dilation constraint found")

    # 条件分岐: `not pe_all_from_beta` を満たす経路を評価する。

    if not pe_all_from_beta:
        raise ValueError("no CMB temperature scaling constraint found")

    p_t = pt_all[0]
    p_e = pe_all_from_beta[0]

    try:
        sR_bao = float(bao_fit["fit"]["best_fit"]["s_R"])
        sR_bao_sigma_base = float(bao_fit["fit"]["best_fit"]["s_R_sigma_1d"])
    except Exception as e:
        raise ValueError("unexpected BAO fit metrics schema") from e

    sR_bao_sigma = float(sR_bao_sigma_base * float(args.bao_sigma_scale))

    all_res = _evaluate_global(
        ddr=ddr,
        opacity_list=opacity_all,
        candle_list=candle_all,
        p_t=p_t,
        p_e=p_e,
        sR_bao=sR_bao,
        sR_bao_sigma=sR_bao_sigma,
    )
    ind_res = _evaluate_global(
        ddr=ddr,
        opacity_list=opacity_ind,
        candle_list=candle_ind,
        p_t=p_t,
        p_e=p_e,
        sR_bao=sR_bao,
        sR_bao_sigma=sR_bao_sigma,
    )

    _set_japanese_font()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    vmax = 6.0
    fig = plt.figure(figsize=(15.8, 7.6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    _plot_heatmap(
        ax1,
        title="全候補（opacity×candle）: 全DDRの worst max|z|（★=最小）",
        z_grid=all_res["z_grid"],
        x_labels=[c.short_label for c in candle_all],
        y_labels=[o.short_label for o in opacity_all],
        best_ij=all_res["best_ij"],
        vmax=vmax,
    )
    _plot_heatmap(
        ax2,
        title="独立一次ソースのみ: 全DDRの worst max|z|（★=最小）",
        z_grid=ind_res["z_grid"],
        x_labels=[c.short_label for c in candle_ind],
        y_labels=[o.short_label for o in opacity_ind],
        best_ij=ind_res["best_ij"],
        vmax=vmax,
    )

    legend_handles = [
        Patch(facecolor="#2ca02c", edgecolor="#333333", label="worst max|z|≤1"),
        Patch(facecolor="#ffbf00", edgecolor="#333333", label="1<worst max|z|≤3"),
        Patch(facecolor="#d62728", edgecolor="#333333", label="worst max|z|>3"),
    ]
    ax2.legend(handles=legend_handles, loc="upper left", fontsize=9, frameon=True, title="色（全DDRの最悪値）")

    fig.suptitle(
        "宇宙論（距離指標の再導出候補探索）：α と s_L を単一priorに固定した場合の全DDR整合（worst max|z|）",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.02,
        f"注：各セルは『全DDR行の最悪 max|z|』。BAO prior は BOSS DR12 fit（s_R={sR_bao:.3f}, σ={sR_bao_sigma:.3f}）を使用。",
        ha="center",
        fontsize=10,
        color="#333333",
    )
    plt.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))

    out_png = out_dir / "cosmology_distance_indicator_rederivation_global_prior_search.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    out_metrics = out_dir / "cosmology_distance_indicator_rederivation_global_prior_search_metrics.json"
    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "objective": "minimize worst(max|z|) across all DDR rows, by choosing a single opacity α prior and a single candle s_L prior",
            "epsilon0_model": "ε0 = (p_e + p_t - s_L)/2 - 2 + s_R + α",
            "notes": [
                "This is a robustness check: candidate_search picks priors per DDR row, but a physical model should use a single prior set.",
                "We keep BAO(s_R), p_t, p_e fixed (same as candidate_search), and only change which (α, s_L) primary-source constraints are used.",
            ],
        },
        "inputs": {
            "ddr": str(in_ddr.relative_to(_ROOT)).replace("\\", "/"),
            "opacity": str(in_opacity.relative_to(_ROOT)).replace("\\", "/"),
            "candle": str(in_candle.relative_to(_ROOT)).replace("\\", "/"),
            "sn_time_dilation": str(in_pt.relative_to(_ROOT)).replace("\\", "/"),
            "cmb_temperature_scaling": str(in_pe.relative_to(_ROOT)).replace("\\", "/"),
            "bao_fit": str(in_bao_fit.relative_to(_ROOT)).replace("\\", "/"),
        },
        "ddr_sigma_policy": {
            "policy": ddr_sigma_policy,
            "envelope_metrics": (
                str(ddr_env_path.relative_to(_ROOT)).replace("\\", "/") if ddr_sigma_policy == "category_sys" else None
            ),
            "applied_count": int(len([x for x in ddr if x.sigma_policy == "category_sys"])),
        },
        "fixed_constraints": {
            "p_t": {"id": p_t.id, "mean": p_t.mean, "sigma": p_t.sigma, "short_label": p_t.short_label},
            "p_e": {"id": p_e.id, "mean": p_e.mean, "sigma": p_e.sigma, "short_label": p_e.short_label},
            "bao_s_R": {
                "mean": sR_bao,
                "sigma_base": sR_bao_sigma_base,
                "sigma_scale": float(args.bao_sigma_scale),
                "sigma_used": sR_bao_sigma,
            },
        },
        "candidates": {
            "opacity_all": [{"id": c.id, "short_label": c.short_label, "mean": c.mean, "sigma": c.sigma} for c in opacity_all],
            "candle_all": [{"id": c.id, "short_label": c.short_label, "mean": c.mean, "sigma": c.sigma} for c in candle_all],
            "opacity_independent": [
                {"id": c.id, "short_label": c.short_label, "mean": c.mean, "sigma": c.sigma} for c in opacity_ind
            ],
            "candle_independent": [
                {"id": c.id, "short_label": c.short_label, "mean": c.mean, "sigma": c.sigma} for c in candle_ind
            ],
        },
        "results": {
            "all": {
                "best_ij": all_res["best_ij"],
                "top_pairs": all_res["top_pairs"],
                "z_grid": all_res["z_grid"].tolist(),
            },
            "independent": {
                "best_ij": ind_res["best_ij"],
                "top_pairs": ind_res["top_pairs"],
                "z_grid": ind_res["z_grid"].tolist(),
            },
        },
        "outputs": {
            "png": str(out_png.relative_to(_ROOT)).replace("\\", "/"),
            "metrics_json": str(out_metrics.relative_to(_ROOT)).replace("\\", "/"),
        },
    }
    _write_json(out_metrics, payload)

    # BAO sigma scan (single prior robustness)
    scan_scales = _parse_scales(str(args.bao_sigma_scale_scan))
    out_scan_png = out_dir / "cosmology_distance_indicator_rederivation_global_prior_search_bao_sigma_scan.png"
    out_scan_metrics = (
        out_dir / "cosmology_distance_indicator_rederivation_global_prior_search_bao_sigma_scan_metrics.json"
    )
    scan_payload: Optional[Dict[str, Any]] = None
    # 条件分岐: `scan_scales` を満たす経路を評価する。
    if scan_scales:
        all_best = []
        ind_best = []
        for f in scan_scales:
            sR_sigma = float(sR_bao_sigma_base * float(f))
            all_best.append(
                _evaluate_best_pair(
                    ddr=ddr,
                    opacity_list=opacity_all,
                    candle_list=candle_all,
                    p_t=p_t,
                    p_e=p_e,
                    sR_bao=sR_bao,
                    sR_bao_sigma=sR_sigma,
                )
            )
            ind_best.append(
                _evaluate_best_pair(
                    ddr=ddr,
                    opacity_list=opacity_ind,
                    candle_list=candle_ind,
                    p_t=p_t,
                    p_e=p_e,
                    sR_bao=sR_bao,
                    sR_bao_sigma=sR_sigma,
                )
            )

        f_all_1sigma = _crossing_x_log(
            scan_scales, [float(x.get("worst_max_abs_z", float("nan"))) for x in all_best], threshold=1.0
        )
        f_ind_1sigma = _crossing_x_log(
            scan_scales, [float(x.get("worst_max_abs_z", float("nan"))) for x in ind_best], threshold=1.0
        )

        _plot_bao_sigma_scan(
            out_png=out_scan_png,
            scales=scan_scales,
            all_best=all_best,
            ind_best=ind_best,
            f_all_1sigma=f_all_1sigma,
            f_ind_1sigma=f_ind_1sigma,
            sR_bao=sR_bao,
            sR_bao_sigma_base=sR_bao_sigma_base,
        )

        scan_payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "definition": {
                "objective": "for each BAO sigma scale f, choose a single (alpha, s_L) prior pair to minimize worst(max|z|) across all DDR rows",
                "notes": [
                    "This scan answers: how much BAO(s_R) must be softened to make a single-prior explanation plausible across DDR rows.",
                    "We do not mix BAO fit covariance modes here (handled in a separate sensitivity script).",
                ],
            },
            "inputs": payload.get("inputs"),
            "ddr_sigma_policy": payload.get("ddr_sigma_policy"),
            "fixed_constraints": payload.get("fixed_constraints"),
            "scan": {
                "bao_sigma_scales": [float(x) for x in scan_scales],
                "all_candidates": all_best,
                "independent_only": ind_best,
                "estimated_f_1sigma": {"all_candidates": f_all_1sigma, "independent_only": f_ind_1sigma},
            },
            "outputs": {
                "png": str(out_scan_png.relative_to(_ROOT)).replace("\\", "/"),
                "metrics_json": str(out_scan_metrics.relative_to(_ROOT)).replace("\\", "/"),
            },
        }
        _write_json(out_scan_metrics, scan_payload)

    outputs_for_log = [out_png, out_metrics]
    # 条件分岐: `scan_scales` を満たす経路を評価する。
    if scan_scales:
        outputs_for_log.extend([out_scan_png, out_scan_metrics])

    worklog.append_event(
        {
            "kind": "cosmology_distance_indicator_rederivation_global_prior_search",
            "script": "scripts/cosmology/cosmology_distance_indicator_rederivation_global_prior_search.py",
            "inputs": [in_ddr, in_opacity, in_candle, in_pt, in_pe, in_bao_fit, ddr_env_path],
            "outputs": outputs_for_log,
            "ddr_sigma_policy": payload.get("ddr_sigma_policy"),
            "bao_sigma_scale": float(args.bao_sigma_scale),
            "bao_sigma_scale_scan": scan_scales,
            "top_all": all_res["top_pairs"][:3],
            "top_independent": ind_res["top_pairs"][:3],
            "scan": (scan_payload or {}).get("scan"),
        }
    )

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_metrics}")
    # 条件分岐: `scan_scales` を満たす経路を評価する。
    if scan_scales:
        print(f"[ok] png : {out_scan_png}")
        print(f"[ok] json: {out_scan_metrics}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
