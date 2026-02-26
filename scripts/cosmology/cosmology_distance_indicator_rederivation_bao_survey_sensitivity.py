#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_distance_indicator_rederivation_bao_survey_sensitivity.py

Phase 16（宇宙論）/ Step 16.4：
BAO(s_R) の prior は「共分散の扱い」だけでなく、どの BAO 一次ソースを用いるか
（BOSS only / eBOSS only / 併合など）でも変わり得る。

本スクリプトでは、BAO の一次ソース（BOSS/eBOSS など）を切り替えた (s_R, σ) を
BAO prior として採用し、単一prior（α,s_L）で全DDR行を 1σ同時整合に入れるために
必要となる BAO σスケール f（σ→fσ）を推定して固定する。

入力（固定）:
  - output/private/cosmology/cosmology_bao_scaled_distance_fit_metrics.json（BOSS DR12: baseline）
  - output/private/cosmology/cosmology_bao_distance_ratio_fit_metrics.json（BOSS / eBOSS / DESI / combined）
  - data/cosmology/distance_duality_constraints.json
  - data/cosmology/cosmic_opacity_constraints.json
  - data/cosmology/sn_standard_candle_evolution_constraints.json
  - data/cosmology/sn_time_dilation_constraints.json
  - data/cosmology/cmb_temperature_scaling_constraints.json
  - （任意）output/private/cosmology/cosmology_distance_duality_systematics_envelope_metrics.json（DDR σ_cat）

出力（固定名）:
  - output/private/cosmology/cosmology_distance_indicator_rederivation_bao_survey_global_prior_sigma_scan.png
  - output/private/cosmology/cosmology_distance_indicator_rederivation_bao_survey_global_prior_sigma_scan_metrics.json
  - output/private/cosmology/cosmology_distance_indicator_rederivation_bao_survey_leave_one_out_global_prior_sigma_scan.png
  - output/private/cosmology/cosmology_distance_indicator_rederivation_bao_survey_leave_one_out_global_prior_sigma_scan_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

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


# 関数: `_plot_f_bars` の入出力契約と処理意図を定義する。

def _plot_f_bars(
    *,
    out_png: Path,
    labels: Sequence[str],
    f_all: Sequence[Optional[float]],
    f_ind: Sequence[Optional[float]],
    max_f: float,
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    # 関数: `_sanitize` の入出力契約と処理意図を定義する。
    def _sanitize(v: Optional[float]) -> float:
        # 条件分岐: `v is None` を満たす経路を評価する。
        if v is None:
            return float(max_f * 1.25)

        # 条件分岐: `not (math.isfinite(float(v)) and float(v) > 0.0)` を満たす経路を評価する。

        if not (math.isfinite(float(v)) and float(v) > 0.0):
            return float(max_f * 1.25)

        return float(v)

    f_all_v = [_sanitize(v) for v in f_all]
    f_ind_v = [_sanitize(v) for v in f_ind]
    ann_all = [(">%.0g" % max_f) if v is None else f"{float(v):.2g}" for v in f_all]
    ann_ind = [(">%.0g" % max_f) if v is None else f"{float(v):.2g}" for v in f_ind]

    x = list(range(len(labels)))
    w = 0.35
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.set_yscale("log")

    bars1 = ax.bar([xi - w / 2 for xi in x], f_all_v, width=w, color="#1f77b4", alpha=0.85, label="全候補")
    bars2 = ax.bar([xi + w / 2 for xi in x], f_ind_v, width=w, color="#ff7f0e", alpha=0.85, label="独立一次ソースのみ")
    ax.axhline(1.0, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.7, label="f=1（緩和なし）")

    ax.set_xticks(x)
    ax.set_xticklabels(list(labels), fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("推定 f（単一priorで1σ同時整合に必要な BAO σスケール）", fontsize=11)
    ax.set_xlabel("BAO一次ソース（s_R 推定に用いたデータ系統）", fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    for bar, txt in zip(bars1, ann_all):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05, txt, ha="center", va="bottom", fontsize=9)

    for bar, txt in zip(bars2, ann_ind):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05, txt, ha="center", va="bottom", fontsize=9)

    fig.suptitle("宇宙論（距離指標）：BAO一次ソースごとの『単一priorで1σ同時整合に必要な緩和量 f』", fontsize=13)
    fig.text(
        0.5,
        0.02,
        "注：各バーは、そのBAO一次ソースの (s_R,σ) を基準に、σ→fσ として単一prior（α,s_L）で全DDRの worst max|z|≤1 となる f を推定（対数x補間）。",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# 関数: `_compute_required_f_scan` の入出力契約と処理意図を定義する。

def _compute_required_f_scan(
    *,
    variants: Sequence[Dict[str, Any]],
    ddr: Sequence[cand.DDRConstraint],
    opacity_all: Sequence[cand.GaussianConstraint],
    candle_all: Sequence[cand.GaussianConstraint],
    opacity_ind: Sequence[cand.GaussianConstraint],
    candle_ind: Sequence[cand.GaussianConstraint],
    p_t: cand.GaussianConstraint,
    p_e: cand.GaussianConstraint,
    scan_scales: Sequence[float],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for v in variants:
        sR = float(v["s_R"])
        sig0 = float(v["sigma"])
        series = []
        y_all = []
        y_ind = []
        for f in scan_scales:
            sig = float(sig0 * float(f))
            best_all = _evaluate_best_pair(
                ddr=ddr,
                opacity_list=opacity_all,
                candle_list=candle_all,
                p_t=p_t,
                p_e=p_e,
                sR_bao=sR,
                sR_bao_sigma=sig,
            )
            best_ind = _evaluate_best_pair(
                ddr=ddr,
                opacity_list=opacity_ind,
                candle_list=candle_ind,
                p_t=p_t,
                p_e=p_e,
                sR_bao=sR,
                sR_bao_sigma=sig,
            )
            series.append(
                {
                    "f": float(f),
                    "all_candidates": best_all,
                    "independent_only": best_ind,
                }
            )
            y_all.append(float(best_all["worst_max_abs_z"]))
            y_ind.append(float(best_ind["worst_max_abs_z"]))

        f_all_1sigma = _crossing_x_log(scan_scales, y_all, threshold=1.0)
        f_ind_1sigma = _crossing_x_log(scan_scales, y_ind, threshold=1.0)
        results.append(
            {
                "variant": v,
                "estimated_f_1sigma_all_candidates": f_all_1sigma,
                "estimated_f_1sigma_independent_only": f_ind_1sigma,
                "series": series,
            }
        )

    return results


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
        "--bao-sigma-scale-scan",
        type=str,
        default="1,1.5,2,2.5,3,4,5,6,8,10,12,15,20,30,40,60,80",
        help="BAO σスケール f のスキャン（カンマ区切り）。",
    )
    args = ap.parse_args(argv)

    data_dir = _ROOT / "data" / "cosmology"
    out_dir = _ROOT / "output" / "private" / "cosmology"

    in_bao_baseline = out_dir / "cosmology_bao_scaled_distance_fit_metrics.json"
    in_bao_ratio = out_dir / "cosmology_bao_distance_ratio_fit_metrics.json"
    in_ddr = data_dir / "distance_duality_constraints.json"
    in_opacity = data_dir / "cosmic_opacity_constraints.json"
    in_candle = data_dir / "sn_standard_candle_evolution_constraints.json"
    in_pt = data_dir / "sn_time_dilation_constraints.json"
    in_pe = data_dir / "cmb_temperature_scaling_constraints.json"

    for p in (in_bao_baseline, in_bao_ratio, in_ddr, in_opacity, in_candle, in_pt, in_pe):
        # 条件分岐: `not p.exists()` を満たす経路を評価する。
        if not p.exists():
            raise FileNotFoundError(f"missing input: {p}")

    scan_scales = _parse_scales(str(args.bao_sigma_scale_scan))
    # 条件分岐: `not scan_scales` を満たす経路を評価する。
    if not scan_scales:
        raise ValueError("empty --bao-sigma-scale-scan")

    # Load constraints

    ddr_rows = _read_json(in_ddr).get("constraints") or []
    opacity_rows = _read_json(in_opacity).get("constraints") or []
    candle_rows = _read_json(in_candle).get("constraints") or []
    pt_rows = _read_json(in_pt).get("constraints") or []
    pe_rows = _read_json(in_pe).get("constraints") or []

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

    # BAO priors (survey/source variants)
    bao_baseline = _read_json(in_bao_baseline)
    bao_ratio = _read_json(in_bao_ratio)

    variants: List[Dict[str, Any]] = []
    try:
        bf = bao_baseline["fit"]["best_fit"]
        variants.append(
            {
                "id": "boss_dr12_dm_h_baseline",
                "label": "BOSS DR12（D_M,H）",
                "s_R": float(bf["s_R"]),
                "sigma": float(bf["s_R_sigma_1d"]),
                "source": str(in_bao_baseline.relative_to(_ROOT)).replace("\\", "/"),
            }
        )
    except Exception as e:
        raise ValueError("unexpected schema in bao_scaled_distance_fit_metrics.json") from e

    # 関数: `_add_ratio_variant` の入出力契約と処理意図を定義する。

    def _add_ratio_variant(key: str, label: str) -> None:
        try:
            bf2 = bao_ratio["results"][key]["best_fit"]
            variants.append(
                {
                    "id": f"bao_ratio_{key}",
                    "label": label,
                    "s_R": float(bf2["s_R"]),
                    "sigma": float(bf2["s_R_sigma_1d"]),
                    "source": str(in_bao_ratio.relative_to(_ROOT)).replace("\\", "/"),
                }
            )
        except Exception:
            return

    _add_ratio_variant("boss_only", "BAO比（BOSSのみ）")
    _add_ratio_variant("eboss_only", "BAO比（eBOSSのみ）")
    _add_ratio_variant("desi_only", "BAO比（DESIのみ）")
    _add_ratio_variant("combined", "BAO比（BOSS+eBOSS+DESI）")

    # Optional: treat the survey-to-survey discrepancy as an additional systematic width in s_R.
    # This is a diagnostic to connect "survey tension" to the effective BAO prior looseness needed in Step 16.4.
    try:
        subset_vs = [v for v in variants if str(v.get("id") or "").startswith("bao_ratio_") and str(v.get("id") or "") != "bao_ratio_combined"]
        subset_vs = [v for v in subset_vs if isinstance(v.get("s_R"), (int, float)) and math.isfinite(float(v["s_R"]))]
        comb = next(v for v in variants if str(v.get("id") or "") == "bao_ratio_combined")
        sRs = [float(v["s_R"]) for v in subset_vs]
        max_pair = 0.0
        for i in range(len(sRs)):
            for j in range(i + 1, len(sRs)):
                max_pair = max(max_pair, abs(sRs[i] - sRs[j]))

        sigma_sys = 0.5 * float(max_pair)
        # 条件分岐: `math.isfinite(sigma_sys) and sigma_sys > 0.0` を満たす経路を評価する。
        if math.isfinite(sigma_sys) and sigma_sys > 0.0:
            sigma_total = float(math.sqrt(float(comb["sigma"]) ** 2 + float(sigma_sys) ** 2))
            variants.append(
                {
                    "id": "bao_ratio_combined_with_survey_sys",
                    "label": "BAO比（多系統差, 系統幅σ_sys）",
                    "s_R": float(comb["s_R"]),
                    "sigma": sigma_total,
                    "sigma_sys": float(sigma_sys),
                    "sigma_stat": float(comb["sigma"]),
                    "source": str(in_bao_ratio.relative_to(_ROOT)).replace("\\", "/"),
                    "note": "σ_total^2 = σ_stat^2 + σ_sys^2,  σ_sys = 0.5*max|s_R(subset_i)-s_R(subset_j)| over available subsets.",
                }
            )
    except Exception:
        pass

    # Optional (diagnostic): treat the maximum leave-one-out shift as an additional systematic width in s_R.
    # This connects "single-point dominance" (especially high-z points) to the effective BAO prior looseness.

    try:
        comb = next(v for v in variants if str(v.get("id") or "") == "bao_ratio_combined")
        sR_full = float(comb["s_R"])
        loo_items = (bao_ratio.get("results") or {}).get("combined_leave_one_out") or []
        shifts: List[float] = []
        for item in loo_items:
            fit = item.get("fit") or {}
            bf = fit.get("best_fit") or {}
            sR = bf.get("s_R")
            # 条件分岐: `isinstance(sR, (int, float)) and math.isfinite(float(sR))` を満たす経路を評価する。
            if isinstance(sR, (int, float)) and math.isfinite(float(sR)):
                shifts.append(float(sR) - sR_full)

        # 条件分岐: `shifts` を満たす経路を評価する。

        if shifts:
            max_shift = float(max(abs(x) for x in shifts))
            sigma_sys = 0.5 * max_shift
            # 条件分岐: `math.isfinite(sigma_sys) and sigma_sys > 0.0` を満たす経路を評価する。
            if math.isfinite(sigma_sys) and sigma_sys > 0.0:
                sigma_total = float(math.sqrt(float(comb["sigma"]) ** 2 + float(sigma_sys) ** 2))
                variants.append(
                    {
                        "id": "bao_ratio_combined_with_loo_sys",
                        "label": "BAO比（combined, LOO最大シフトσ_sys）",
                        "s_R": float(comb["s_R"]),
                        "sigma": sigma_total,
                        "sigma_sys": float(sigma_sys),
                        "sigma_stat": float(comb["sigma"]),
                        "source": str(in_bao_ratio.relative_to(_ROOT)).replace("\\", "/"),
                        "note": "σ_total^2 = σ_stat^2 + σ_sys^2,  σ_sys = 0.5*max|s_R(leave-one-out)-s_R(full)|",
                    }
                )
    except Exception:
        pass

    # 条件分岐: `len(variants) < 2` を満たす経路を評価する。

    if len(variants) < 2:
        raise ValueError("BAO ratio variants not found; expected at least boss_only/eboss_only/combined")

    results = _compute_required_f_scan(
        variants=variants,
        ddr=ddr,
        opacity_all=opacity_all,
        candle_all=candle_all,
        opacity_ind=opacity_ind,
        candle_ind=candle_ind,
        p_t=p_t,
        p_e=p_e,
        scan_scales=scan_scales,
    )

    out_png = out_dir / "cosmology_distance_indicator_rederivation_bao_survey_global_prior_sigma_scan.png"
    out_metrics = (
        out_dir / "cosmology_distance_indicator_rederivation_bao_survey_global_prior_sigma_scan_metrics.json"
    )

    max_f = float(max(scan_scales)) if scan_scales else 1.0
    _plot_f_bars(
        out_png=out_png,
        labels=[str(r["variant"]["label"]) for r in results],
        f_all=[r["estimated_f_1sigma_all_candidates"] for r in results],
        f_ind=[r["estimated_f_1sigma_independent_only"] for r in results],
        max_f=max_f,
    )

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "what": "BAO primary-source sensitivity of the required softening f for a single (alpha, s_L) prior to explain all DDR rows within 1σ",
            "notes": [
                "For each BAO prior variant, we scale its σ as σ→fσ and, at each f, choose a single (alpha, s_L) pair that minimizes worst(max|z|) across all DDR rows.",
                "This is a diagnostic; it does NOT claim BAO uncertainties should be inflated by f.",
            ],
        },
        "inputs": {
            "bao_baseline": str(in_bao_baseline.relative_to(_ROOT)).replace("\\", "/"),
            "bao_ratio_fit": str(in_bao_ratio.relative_to(_ROOT)).replace("\\", "/"),
            "ddr": str(in_ddr.relative_to(_ROOT)).replace("\\", "/"),
            "opacity": str(in_opacity.relative_to(_ROOT)).replace("\\", "/"),
            "candle": str(in_candle.relative_to(_ROOT)).replace("\\", "/"),
            "sn_time_dilation": str(in_pt.relative_to(_ROOT)).replace("\\", "/"),
            "cmb_temperature_scaling": str(in_pe.relative_to(_ROOT)).replace("\\", "/"),
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
        },
        "scan": {"bao_sigma_scales": [float(x) for x in scan_scales], "by_variant": results},
        "outputs": {
            "png": str(out_png.relative_to(_ROOT)).replace("\\", "/"),
            "metrics_json": str(out_metrics.relative_to(_ROOT)).replace("\\", "/"),
        },
    }
    _write_json(out_metrics, payload)

    # --- leave-one-out: does a single BAO point dominate the required f? ---
    def _short_point_label(s: str) -> str:
        txt = str(s or "").strip()
        txt = txt.replace("BOSS DR12 ", "BOSS ")
        txt = txt.replace("eBOSS DR16 ", "")
        txt = txt.replace("eBOSS ", "")
        return txt or "unknown"

    loo_variants: List[Dict[str, Any]] = []
    try:
        bf_comb = bao_ratio["results"]["combined"]["best_fit"]
        loo_variants.append(
            {
                "id": "bao_ratio_combined_baseline",
                "label": "併合（全点）",
                "s_R": float(bf_comb["s_R"]),
                "sigma": float(bf_comb["s_R_sigma_1d"]),
                "source": str(in_bao_ratio.relative_to(_ROOT)).replace("\\", "/"),
                "omitted": None,
            }
        )
    except Exception:
        pass

    for item in (bao_ratio.get("results") or {}).get("combined_leave_one_out") or []:
        omitted = item.get("omitted") or {}
        fit = item.get("fit") or {}
        bf = fit.get("best_fit") or {}
        omitted_id = str(omitted.get("id") or "").strip()
        omitted_label = _short_point_label(str(omitted.get("short_label") or omitted_id or "unknown"))
        # 条件分岐: `not omitted_id` を満たす経路を評価する。
        if not omitted_id:
            continue

        # 条件分岐: `not (isinstance(bf.get("s_R"), (int, float)) and isinstance(bf.get("s_R_sigma...` を満たす経路を評価する。

        if not (isinstance(bf.get("s_R"), (int, float)) and isinstance(bf.get("s_R_sigma_1d"), (int, float))):
            continue

        loo_variants.append(
            {
                "id": f"bao_ratio_combined_drop_{omitted_id}",
                "label": f"併合（{omitted_label}除外）",
                "s_R": float(bf["s_R"]),
                "sigma": float(bf["s_R_sigma_1d"]),
                "source": str(in_bao_ratio.relative_to(_ROOT)).replace("\\", "/"),
                "omitted": {"id": omitted_id, "short_label": omitted_label, "z_eff": omitted.get("z_eff")},
                "fit_chi2_dof": fit.get("chi2_dof"),
            }
        )

    loo_results: List[Dict[str, Any]] = []
    out_png_loo: Optional[Path] = None
    out_metrics_loo: Optional[Path] = None
    # 条件分岐: `len(loo_variants) >= 3` を満たす経路を評価する。
    if len(loo_variants) >= 3:
        loo_results = _compute_required_f_scan(
            variants=loo_variants,
            ddr=ddr,
            opacity_all=opacity_all,
            candle_all=candle_all,
            opacity_ind=opacity_ind,
            candle_ind=candle_ind,
            p_t=p_t,
            p_e=p_e,
            scan_scales=scan_scales,
        )

        out_png_loo = (
            out_dir / "cosmology_distance_indicator_rederivation_bao_survey_leave_one_out_global_prior_sigma_scan.png"
        )
        out_metrics_loo = (
            out_dir
            / "cosmology_distance_indicator_rederivation_bao_survey_leave_one_out_global_prior_sigma_scan_metrics.json"
        )
        _plot_f_bars(
            out_png=out_png_loo,
            labels=[str(r["variant"]["label"]) for r in loo_results],
            f_all=[r["estimated_f_1sigma_all_candidates"] for r in loo_results],
            f_ind=[r["estimated_f_1sigma_independent_only"] for r in loo_results],
            max_f=max_f,
        )

        loo_payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "definition": {
                "what": "Impact of leaving out a single BAO point (combined fit) on the required softening f for a single (alpha, s_L) prior to explain all DDR rows within 1σ",
                "notes": [
                    "Each variant uses the BAO combined-fit leave-one-out (s_R,σ) as the BAO prior, then scans σ→fσ and optimizes a single (alpha, s_L) over all DDR rows.",
                    "This is a diagnostic; it does NOT claim any BAO point should be excluded.",
                ],
            },
            "inputs": {
                "bao_ratio_fit": str(in_bao_ratio.relative_to(_ROOT)).replace("\\", "/"),
                "ddr": str(in_ddr.relative_to(_ROOT)).replace("\\", "/"),
                "opacity": str(in_opacity.relative_to(_ROOT)).replace("\\", "/"),
                "candle": str(in_candle.relative_to(_ROOT)).replace("\\", "/"),
                "sn_time_dilation": str(in_pt.relative_to(_ROOT)).replace("\\", "/"),
                "cmb_temperature_scaling": str(in_pe.relative_to(_ROOT)).replace("\\", "/"),
            },
            "ddr_sigma_policy": payload.get("ddr_sigma_policy"),
            "scan": {"bao_sigma_scales": [float(x) for x in scan_scales], "by_variant": loo_results},
            "outputs": {
                "png": str(out_png_loo.relative_to(_ROOT)).replace("\\", "/"),
                "metrics_json": str(out_metrics_loo.relative_to(_ROOT)).replace("\\", "/"),
            },
        }
        _write_json(out_metrics_loo, loo_payload)

    worklog.append_event(
        {
            "kind": "cosmology_distance_indicator_rederivation_bao_survey_sensitivity",
            "script": "scripts/cosmology/cosmology_distance_indicator_rederivation_bao_survey_sensitivity.py",
            "inputs": [
                in_bao_baseline,
                in_bao_ratio,
                in_ddr,
                in_opacity,
                in_candle,
                in_pt,
                in_pe,
                ddr_env_path,
            ],
            "outputs": (
                [out_png, out_metrics] + ([out_png_loo, out_metrics_loo] if out_png_loo and out_metrics_loo else [])
            ),
            "ddr_sigma_policy": payload.get("ddr_sigma_policy"),
            "scan_summary": [
                {
                    "label": r["variant"]["label"],
                    "s_R": r["variant"]["s_R"],
                    "sigma": r["variant"]["sigma"],
                    "f_1sigma_all": r["estimated_f_1sigma_all_candidates"],
                    "f_1sigma_ind": r["estimated_f_1sigma_independent_only"],
                }
                for r in results
            ],
            "loo_scan_summary": (
                [
                    {
                        "label": r["variant"]["label"],
                        "omitted": r["variant"].get("omitted"),
                        "s_R": r["variant"]["s_R"],
                        "sigma": r["variant"]["sigma"],
                        "f_1sigma_all": r["estimated_f_1sigma_all_candidates"],
                        "f_1sigma_ind": r["estimated_f_1sigma_independent_only"],
                    }
                    for r in loo_results
                ]
                if loo_results
                else None
            ),
        }
    )

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_metrics}")
    # 条件分岐: `out_png_loo and out_metrics_loo` を満たす経路を評価する。
    if out_png_loo and out_metrics_loo:
        print(f"[ok] png : {out_png_loo}")
        print(f"[ok] json: {out_metrics_loo}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
