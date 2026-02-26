#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_distance_indicator_rederivation_bao_mode_sensitivity.py

Phase 16（宇宙論）/ Step 16.4：
BAO(s_R) の一次データ fit の前提（共分散の扱い）によって、
距離指標の再導出候補探索（best_independent）の結論（max|z|, limiting）が
どの程度変わるかを「全DDR一次ソース」で可視化する。

背景：
  - `cosmology_bao_scaled_distance_fit_sensitivity.py` では、
    BOSS DR12 の (D_M, H) の共分散の扱い（block/diag/full など）で
    s_R と σ がどれだけ動き得るかを整理した。
  - 本スクリプトは、その各モードの (s_R, σ) を BAO prior として使い、
    candidate_search（WLS: DDR + BAO(s_R) + α + s_L + p_t + p_e）の
    best_independent を DDR 行ごとに再計算して、影響を固定する。

入力（固定）:
  - output/private/cosmology/cosmology_bao_scaled_distance_fit_sensitivity_metrics.json（BAO prior の候補）
  - data/cosmology/distance_duality_constraints.json
  - data/cosmology/cosmic_opacity_constraints.json
  - data/cosmology/sn_standard_candle_evolution_constraints.json
  - data/cosmology/sn_time_dilation_constraints.json
  - data/cosmology/cmb_temperature_scaling_constraints.json
  - （任意）output/private/cosmology/cosmology_distance_duality_systematics_envelope_metrics.json（DDR σ_cat）

出力（固定名）:
  - output/private/cosmology/cosmology_distance_indicator_rederivation_bao_mode_sensitivity.png
  - output/private/cosmology/cosmology_distance_indicator_rederivation_bao_mode_sensitivity_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _classify_cell(v: Optional[float]) -> int:
    """
    Returns category index for coloring.
      0: ok (<=1)
      1: warn (<=3)
      2: ng (>3)
      3: na
    """
    # 条件分岐: `v is None` を満たす経路を評価する。
    if v is None:
        return 3

    # 条件分岐: `not math.isfinite(float(v))` を満たす経路を評価する。

    if not math.isfinite(float(v)):
        return 3

    v = float(v)
    # 条件分岐: `v <= 1.0` を満たす経路を評価する。
    if v <= 1.0:
        return 0

    # 条件分岐: `v <= 3.0` を満たす経路を評価する。

    if v <= 3.0:
        return 1

    return 2


def _short_mode_label(mode: str) -> str:
    m = str(mode)
    mapping = {
        "block": "block",
        "diag": "diag",
        "full": "full",
        "full_dm_only": "DMのみ",
        "full_h_only": "Hのみ",
    }
    return mapping.get(m, m)


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


def _plot_matrix(
    *,
    out_png: Path,
    ddr_labels: Sequence[str],
    mode_labels: Sequence[str],
    values: np.ndarray,
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Rectangle

    n_rows, n_cols = values.shape
    cats = np.zeros((n_rows, n_cols), dtype=int)
    for i in range(n_rows):
        for j in range(n_cols):
            v = float(values[i, j])
            cats[i, j] = _classify_cell(v if math.isfinite(v) else None)

    cmap = ListedColormap(["#2ca02c", "#ffbf00", "#d62728", "#999999"])

    fig, ax = plt.subplots(figsize=(15.5, 8.5))
    ax.imshow(cats, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest", aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(list(mode_labels), fontsize=10)
    ax.set_xlabel("BAO fit のモード（共分散の扱い）", fontsize=11)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(list(ddr_labels), fontsize=10)
    ax.set_ylabel("DDR一次ソース（best_independent）", fontsize=11)

    # grid lines
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=1.0, alpha=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)

    # annotate numbers
    for i in range(n_rows):
        for j in range(n_cols):
            v = float(values[i, j])
            # 条件分岐: `not math.isfinite(v)` を満たす経路を評価する。
            if not math.isfinite(v):
                continue

            cat = int(cats[i, j])
            txt_color = "#ffffff" if cat == 2 else "#111111"
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=txt_color,
                alpha=0.95,
            )

    # highlight baseline column (block) if present

    try:
        j0 = list(mode_labels).index("block")
    except ValueError:
        j0 = None

    # 条件分岐: `j0 is not None` を満たす経路を評価する。

    if j0 is not None:
        ax.add_patch(
            Rectangle(
                (float(j0) - 0.5, -0.5),
                1.0,
                float(n_rows),
                fill=False,
                edgecolor="#111111",
                linewidth=2.0,
                alpha=0.35,
            )
        )

    legend_handles = [
        Patch(facecolor="#2ca02c", edgecolor="#333333", label="max|z|≤1（1σ）"),
        Patch(facecolor="#ffbf00", edgecolor="#333333", label="1<max|z|≤3"),
        Patch(facecolor="#d62728", edgecolor="#333333", label="max|z|>3"),
        Patch(facecolor="#999999", edgecolor="#333333", label="NA"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10, frameon=True, title="色（整合度）")

    fig.suptitle(
        "宇宙論（距離指標の再導出候補探索）：BAO(s_R) fit 前提（共分散）の感度（best_independent）",
        fontsize=14,
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "各セルの数値は max|z|（DDR + BAO(s_R) + α + s_L + p_t + p_e の同時整合; WLS）。枠は block（既定）。",
        ha="center",
        fontsize=10,
        color="#333333",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_global_prior_sigma_scan_by_mode(
    *,
    out_png: Path,
    modes: Sequence[str],
    scan_scales: Sequence[float],
    scan_results: Sequence[Dict[str, Any]],
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    labels = [str(m) for m in modes]
    max_f = max([float(x) for x in scan_scales], default=1.0)

    # prepare bars (two series per mode)
    f_all = []
    f_ind = []
    ann_all = []
    ann_ind = []
    for r in scan_results:
        a = r.get("estimated_f_1sigma_all_candidates")
        b = r.get("estimated_f_1sigma_independent_only")
        # 条件分岐: `isinstance(a, (int, float)) and math.isfinite(float(a)) and float(a) > 0.0` を満たす経路を評価する。
        if isinstance(a, (int, float)) and math.isfinite(float(a)) and float(a) > 0.0:
            f_all.append(float(a))
            ann_all.append(f"{float(a):.2g}")
        else:
            f_all.append(float(max_f * 1.25))
            ann_all.append(f">{max_f:g}")

        # 条件分岐: `isinstance(b, (int, float)) and math.isfinite(float(b)) and float(b) > 0.0` を満たす経路を評価する。

        if isinstance(b, (int, float)) and math.isfinite(float(b)) and float(b) > 0.0:
            f_ind.append(float(b))
            ann_ind.append(f"{float(b):.2g}")
        else:
            f_ind.append(float(max_f * 1.25))
            ann_ind.append(f">{max_f:g}")

    x = list(range(len(labels)))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    ax.set_yscale("log")

    bars1 = ax.bar([xi - w / 2 for xi in x], f_all, width=w, color="#1f77b4", alpha=0.85, label="全候補（単一prior）")
    bars2 = ax.bar([xi + w / 2 for xi in x], f_ind, width=w, color="#ff7f0e", alpha=0.85, label="独立一次ソースのみ（単一prior）")

    ax.axhline(1.0, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.7, label="1σ境界（f=1）")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("推定 f（1σ同時整合に必要な BAO σスケール）", fontsize=11)
    ax.set_xlabel("BAO fit のモード（共分散の扱い）", fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    # annotate
    for bar, txt in zip(bars1, ann_all):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05, txt, ha="center", va="bottom", fontsize=9)

    for bar, txt in zip(bars2, ann_ind):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05, txt, ha="center", va="bottom", fontsize=9)

    fig.suptitle("宇宙論（距離指標）：BAO fit 前提（共分散）ごとの『単一priorで1σ同時整合に必要な緩和量 f』", fontsize=13)
    fig.text(
        0.5,
        0.02,
        "注：各バーは、そのBAOモードの (s_R,σ) を基準に、σ→fσ として単一prior（α,s_L）で全DDRの worst max|z|≤1 となる f を推定（対数x補間）。",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _extract_bao_modes(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    rows = metrics.get("bao_fit_sensitivity") if isinstance(metrics.get("bao_fit_sensitivity"), list) else []
    for r in rows:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        mode = str(r.get("mode") or "")
        bf = r.get("best_fit") if isinstance(r.get("best_fit"), dict) else {}
        try:
            sR = float(bf["s_R"])
            sig = float(bf["s_R_sigma_1d"])
        except Exception:
            continue

        # 条件分岐: `not (sig > 0.0 and math.isfinite(sig) and math.isfinite(sR))` を満たす経路を評価する。

        if not (sig > 0.0 and math.isfinite(sig) and math.isfinite(sR)):
            continue

        out.append(
            {
                "mode": mode,
                "label": str(r.get("label") or mode),
                "s_R": sR,
                "s_R_sigma": sig,
            }
        )

    return out


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
        default="1,1.5,2,2.5,3,4,5,6,8,10,12,15",
        help="BAO σスケール f のスキャン（カンマ区切り）。空文字でスキップ。",
    )
    args = ap.parse_args(argv)

    data_dir = _ROOT / "data" / "cosmology"
    out_dir = _ROOT / "output" / "private" / "cosmology"

    in_bao_modes = out_dir / "cosmology_bao_scaled_distance_fit_sensitivity_metrics.json"
    in_ddr = data_dir / "distance_duality_constraints.json"
    in_opacity = data_dir / "cosmic_opacity_constraints.json"
    in_candle = data_dir / "sn_standard_candle_evolution_constraints.json"
    in_pt = data_dir / "sn_time_dilation_constraints.json"
    in_pe = data_dir / "cmb_temperature_scaling_constraints.json"

    for p in (in_bao_modes, in_ddr, in_opacity, in_candle, in_pt, in_pe):
        # 条件分岐: `not p.exists()` を満たす経路を評価する。
        if not p.exists():
            raise FileNotFoundError(f"missing input: {p}")

    bao_modes_raw = _extract_bao_modes(_read_json(in_bao_modes))
    # 条件分岐: `not bao_modes_raw` を満たす経路を評価する。
    if not bao_modes_raw:
        raise ValueError("no BAO modes found in bao_scaled_distance_fit_sensitivity metrics")

    # deterministic order (prefer common modes first if present)

    preferred = ["block", "diag", "full", "full_dm_only", "full_h_only"]
    order = {m: i for i, m in enumerate(preferred)}
    bao_modes = sorted(bao_modes_raw, key=lambda r: (order.get(str(r["mode"]), 999), str(r["mode"])))

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

    opacity_all = cand._as_gaussian_list(opacity_rows, mean_key="alpha_opacity", sigma_key="alpha_opacity_sigma")
    candle_all = cand._as_gaussian_list(candle_rows, mean_key="s_L", sigma_key="s_L_sigma")
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

    # Compute per-mode x per-ddr matrix.
    labels_ddr = [d.short_label for d in ddr]
    labels_mode = [_short_mode_label(str(m["mode"])) for m in bao_modes]

    vals = np.full((len(ddr), len(bao_modes)), np.nan, dtype=float)
    rows_out: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for j, m in enumerate(bao_modes):
        sR_bao = float(m["s_R"])
        sR_sig = float(m["s_R_sigma"])
        per_ddr = cand._compute_per_ddr(
            ddr=ddr,
            sR_bao=sR_bao,
            sR_bao_sigma=sR_sig,
            opacity_all=opacity_all,
            candle_all=candle_all,
            p_t=p_t,
            p_e=p_e,
            gw_siren_opacity_observed=None,
            gw_siren_opacity_forecast=None,
        )

        ok_1 = 0
        ok_3 = 0
        max_abs = []
        worst: Optional[Tuple[float, str]] = None
        for i, row in enumerate(per_ddr):
            block = row.get("best_independent")
            # 条件分岐: `not isinstance(block, dict)` を満たす経路を評価する。
            if not isinstance(block, dict):
                continue

            fit = block.get("fit") if isinstance(block.get("fit"), dict) else {}
            v = fit.get("max_abs_z")
            try:
                v_f = float(v)
            except Exception:
                v_f = float("nan")

            vals[i, j] = v_f
            # 条件分岐: `math.isfinite(v_f)` を満たす経路を評価する。
            if math.isfinite(v_f):
                max_abs.append(v_f)
                # 条件分岐: `v_f <= 1.0` を満たす経路を評価する。
                if v_f <= 1.0:
                    ok_1 += 1

                # 条件分岐: `v_f <= 3.0` を満たす経路を評価する。

                if v_f <= 3.0:
                    ok_3 += 1

                # 条件分岐: `worst is None or v_f > worst[0]` を満たす経路を評価する。

                if worst is None or v_f > worst[0]:
                    worst = (v_f, str((row.get("ddr") or {}).get("short_label") or ""))

        # 条件分岐: `max_abs` を満たす経路を評価する。

        if max_abs:
            import statistics

            med = float(statistics.median(max_abs))
        else:
            med = float("nan")

        summaries.append(
            {
                "mode": str(m["mode"]),
                "label": str(m["label"]),
                "s_R_bao": sR_bao,
                "s_R_sigma": sR_sig,
                "rows_total": len(ddr),
                "rows_max_abs_z_le_1": ok_1,
                "rows_max_abs_z_le_3": ok_3,
                "median_max_abs_z": med,
                "worst": (None if worst is None else {"max_abs_z": worst[0], "ddr_short_label": worst[1]}),
            }
        )

        rows_out.append(
            {
                "mode": str(m["mode"]),
                "label": str(m["label"]),
                "s_R_bao": sR_bao,
                "s_R_sigma": sR_sig,
                "per_ddr": per_ddr,
            }
        )

    out_png = out_dir / "cosmology_distance_indicator_rederivation_bao_mode_sensitivity.png"
    _plot_matrix(out_png=out_png, ddr_labels=labels_ddr, mode_labels=labels_mode, values=vals)

    out_metrics = out_dir / "cosmology_distance_indicator_rederivation_bao_mode_sensitivity_metrics.json"
    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "what": "candidate_search(best_independent) recomputed for BAO(s_R) priors from bao_scaled_distance_fit_sensitivity modes",
            "notes": [
                "Each BAO mode provides (s_R, σ) from the same BOSS DR12 BAOFS consensus but with different covariance handling.",
                "This is a sensitivity study of the BAO prior, not a claim that any particular mode is correct.",
                "DDR sigma is controlled by --ddr-sigma-policy and can include category-level systematics (σ_cat).",
            ],
        },
        "inputs": {
            "bao_modes_metrics": str(in_bao_modes.relative_to(_ROOT)).replace("\\", "/"),
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
            "note": "If envelope file is missing, σ_cat inflation is skipped for all rows (falls back to raw).",
        },
        "fixed_constraints": {
            "p_t": {"id": p_t.id, "mean": p_t.mean, "sigma": p_t.sigma, "short_label": p_t.short_label},
            "p_e": {"id": p_e.id, "mean": p_e.mean, "sigma": p_e.sigma, "short_label": p_e.short_label},
        },
        "results": {
            "bao_modes": [{"mode": m["mode"], "label": m["label"], "s_R": m["s_R"], "s_R_sigma": m["s_R_sigma"]} for m in bao_modes],
            "mode_summaries": summaries,
            "by_mode": rows_out,
        },
        "outputs": {
            "png": str(out_png.relative_to(_ROOT)).replace("\\", "/"),
            "metrics_json": str(out_metrics.relative_to(_ROOT)).replace("\\", "/"),
        },
    }
    _write_json(out_metrics, payload)

    # Global-prior BAO sigma scan per BAO covariance mode (robustness of single-prior explanation).
    scan_scales = _parse_scales(str(args.bao_sigma_scale_scan))
    out_scan_png = out_dir / "cosmology_distance_indicator_rederivation_bao_mode_global_prior_sigma_scan.png"
    out_scan_metrics = (
        out_dir / "cosmology_distance_indicator_rederivation_bao_mode_global_prior_sigma_scan_metrics.json"
    )
    scan_payload: Optional[Dict[str, Any]] = None
    scan_results: List[Dict[str, Any]] = []

    # 条件分岐: `scan_scales` を満たす経路を評価する。
    if scan_scales:
        opacity_ind = [c for c in opacity_all if c.is_independent()]
        candle_ind = [c for c in candle_all if c.is_independent()]

        for m in bao_modes:
            mode = str(m["mode"])
            sR_bao = float(m["s_R"])
            sR_sig_base = float(m["s_R_sigma"])

            series_all = []
            series_ind = []
            for f in scan_scales:
                sR_sig = float(sR_sig_base * float(f))
                best_all = _evaluate_best_pair(
                    ddr=ddr,
                    opacity_list=opacity_all,
                    candle_list=candle_all,
                    p_t=p_t,
                    p_e=p_e,
                    sR_bao=sR_bao,
                    sR_bao_sigma=sR_sig,
                )
                best_ind = _evaluate_best_pair(
                    ddr=ddr,
                    opacity_list=opacity_ind,
                    candle_list=candle_ind,
                    p_t=p_t,
                    p_e=p_e,
                    sR_bao=sR_bao,
                    sR_bao_sigma=sR_sig,
                )
                series_all.append({"f": float(f), **best_all})
                series_ind.append({"f": float(f), **best_ind})

            f_all_1sigma = _crossing_x_log(
                scan_scales, [float(x["worst_max_abs_z"]) for x in series_all], threshold=1.0
            )
            f_ind_1sigma = _crossing_x_log(
                scan_scales, [float(x["worst_max_abs_z"]) for x in series_ind], threshold=1.0
            )

            scan_results.append(
                {
                    "mode": mode,
                    "label": str(m["label"]),
                    "s_R_bao": sR_bao,
                    "s_R_sigma_base": sR_sig_base,
                    "estimated_f_1sigma_all_candidates": f_all_1sigma,
                    "estimated_f_1sigma_independent_only": f_ind_1sigma,
                    "series_all_candidates": series_all,
                    "series_independent_only": series_ind,
                }
            )

        _plot_global_prior_sigma_scan_by_mode(
            out_png=out_scan_png,
            modes=[_short_mode_label(str(m["mode"])) for m in bao_modes],
            scan_scales=scan_scales,
            scan_results=scan_results,
        )

        scan_payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "definition": {
                "what": "for each BAO covariance mode, estimate the BAO sigma scale f needed to make a single (alpha, s_L) prior explain all DDR rows within 1σ",
                "notes": [
                    "Each point (f) chooses the best single (alpha, s_L) pair to minimize worst(max|z|) across DDR rows.",
                    "This is a robustness diagnostic; it does NOT claim the BAO uncertainty is actually inflated by f.",
                ],
            },
            "inputs": payload.get("inputs"),
            "ddr_sigma_policy": payload.get("ddr_sigma_policy"),
            "fixed_constraints": payload.get("fixed_constraints"),
            "scan": {"bao_sigma_scales": [float(x) for x in scan_scales], "by_mode": scan_results},
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
            "kind": "cosmology_distance_indicator_rederivation_bao_mode_sensitivity",
            "script": "scripts/cosmology/cosmology_distance_indicator_rederivation_bao_mode_sensitivity.py",
            "inputs": [in_bao_modes, in_ddr, in_opacity, in_candle, in_pt, in_pe, ddr_env_path],
            "outputs": outputs_for_log,
            "ddr_sigma_policy": payload.get("ddr_sigma_policy"),
            "mode_summaries": summaries,
            "global_prior_scan": (scan_payload or {}).get("scan"),
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
