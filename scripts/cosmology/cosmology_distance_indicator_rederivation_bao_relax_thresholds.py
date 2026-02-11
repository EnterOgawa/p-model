#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_distance_indicator_rederivation_bao_relax_thresholds.py

Phase 16（宇宙論）/ Step 16.4：
距離指標の再導出候補探索（candidate_search）において、BAO(s_R) を soft constraint として緩めたとき
（σ→fσ）の「同時整合度 max|z|」が DDR 一次ソースごとにどう変化するかを一覧化する。

目的：
  - BAO(s_R) が支配拘束（limiting）になりやすい点を、代表例ではなく全DDR行で可視化する。
  - 「max|z|<=1（1σ）」達成に必要な BAO σスケール f の目安を固定する。

入力（固定）:
  - data/cosmology/distance_duality_constraints.json
  - data/cosmology/cosmic_opacity_constraints.json
  - data/cosmology/sn_standard_candle_evolution_constraints.json
  - data/cosmology/sn_time_dilation_constraints.json
  - data/cosmology/cmb_temperature_scaling_constraints.json
  - output/private/cosmology/cosmology_bao_scaled_distance_fit_metrics.json
  - （任意）output/private/cosmology/cosmology_distance_duality_systematics_envelope_metrics.json（DDR σ_cat）

出力（固定名）:
  - output/private/cosmology/cosmology_distance_indicator_rederivation_bao_relax_thresholds.png
  - output/private/cosmology/cosmology_distance_indicator_rederivation_bao_relax_thresholds_metrics.json
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
        if not tok:
            continue
        try:
            v = float(tok)
        except Exception:
            continue
        if not (v > 0.0 and math.isfinite(v)):
            continue
        out.append(v)
    # deterministic: unique & sorted
    return sorted(set(out))


def _crossing_x_log(xs: Sequence[float], ys: Sequence[float], threshold: float) -> Optional[float]:
    if not xs or len(xs) != len(ys):
        return None
    if not math.isfinite(float(threshold)):
        return None
    xs_f = [float(x) for x in xs]
    ys_f = [float(y) for y in ys]
    if any((x <= 0.0 or not math.isfinite(x)) for x in xs_f):
        return None
    if any((not math.isfinite(y)) for y in ys_f):
        # still try, but skip invalid segments
        pass

    # already below at smallest f
    if math.isfinite(ys_f[0]) and ys_f[0] <= float(threshold):
        return xs_f[0]

    for i in range(1, len(xs_f)):
        x0, x1 = xs_f[i - 1], xs_f[i]
        y0, y1 = ys_f[i - 1], ys_f[i]
        if not (math.isfinite(y0) and math.isfinite(y1)):
            continue
        if (y0 - threshold) == 0.0:
            return x0
        if (y0 - threshold) * (y1 - threshold) > 0.0:
            continue
        if y1 == y0:
            return x1
        # interpolate in log-x space
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
    if v is None:
        return 3
    if not math.isfinite(float(v)):
        return 3
    v = float(v)
    if v <= 1.0:
        return 0
    if v <= 3.0:
        return 1
    return 2


def _plot_matrix(
    *,
    out_png: Path,
    ddr_labels: Sequence[str],
    scales: Sequence[float],
    values: np.ndarray,
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.colors import ListedColormap

    n_rows, n_cols = values.shape
    cats = np.zeros((n_rows, n_cols), dtype=int)
    for i in range(n_rows):
        for j in range(n_cols):
            v = float(values[i, j]) if math.isfinite(float(values[i, j])) else float("nan")
            cats[i, j] = _classify_cell(v if math.isfinite(v) else None)

    cmap = ListedColormap(["#2ca02c", "#ffbf00", "#d62728", "#999999"])

    fig, ax = plt.subplots(figsize=(15.5, 9.0))
    ax.imshow(cats, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest", aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(s).rstrip("0").rstrip(".") for s in scales], fontsize=10)
    ax.set_xlabel("BAO σスケール f（s_R の σ→fσ）", fontsize=11)

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

    # highlight f=1 column if present
    try:
        j0 = list(scales).index(1.0)
    except ValueError:
        j0 = None
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
        "宇宙論（距離指標の再導出候補探索）：BAO(s_R) を緩めたときの同時整合度（best_independent）",
        fontsize=14,
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "各セルの数値は max|z|（DDR + BAO(s_R) + α + s_L + p_t + p_e の同時整合; WLS）。枠は f=1。",
        ha="center",
        fontsize=10,
        color="#333333",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


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
        "--bao-sigma-scales",
        type=str,
        default="1,1.5,2,2.5,3,4,6,10",
        help="Comma-separated positive floats. BAO σスケール f（σ→fσ）の系列。",
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
        if not p.exists():
            raise FileNotFoundError(f"missing input: {p}")

    scales = _parse_scales(str(args.bao_sigma_scales))
    if not scales:
        raise ValueError("--bao-sigma-scales must contain at least one positive float")

    ddr_rows = _read_json(in_ddr).get("constraints") or []
    opacity_rows = _read_json(in_opacity).get("constraints") or []
    candle_rows = _read_json(in_candle).get("constraints") or []
    pt_rows = _read_json(in_pt).get("constraints") or []
    pe_rows = _read_json(in_pe).get("constraints") or []
    bao_fit = _read_json(in_bao_fit)

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
    if not pt_all:
        raise ValueError("no SN time dilation constraint found")
    if not pe_all_from_beta:
        raise ValueError("no CMB temperature scaling constraint found")

    # Match candidate_search: use single constraints for p_t and p_e (first row).
    p_t = pt_all[0]
    p_e = pe_all_from_beta[0]

    try:
        sR_bao = float(bao_fit["fit"]["best_fit"]["s_R"])
        sR_bao_sigma_base = float(bao_fit["fit"]["best_fit"]["s_R_sigma_1d"])
    except Exception as e:
        raise ValueError("unexpected BAO fit metrics schema") from e

    # Compute series across scales.
    series_by_id: Dict[str, List[Dict[str, Any]]] = {d.id: [] for d in ddr}
    for f in scales:
        sigma_used = float(sR_bao_sigma_base * float(f))
        per_ddr = cand._compute_per_ddr(
            ddr=ddr,
            sR_bao=sR_bao,
            sR_bao_sigma=sigma_used,
            opacity_all=opacity_all,
            candle_all=candle_all,
            p_t=p_t,
            p_e=p_e,
            gw_siren_opacity_observed=None,
            gw_siren_opacity_forecast=None,
        )

        for row in per_ddr:
            ddr_id = str((row.get("ddr") or {}).get("id") or "")
            block = row.get("best_independent")
            if not isinstance(block, dict):
                series_by_id[ddr_id].append(
                    {
                        "bao_sigma_scale": float(f),
                        "max_abs_z": None,
                        "limiting_observation": "na",
                        "selected_constraints": {},
                    }
                )
                continue
            fit = block.get("fit") if isinstance(block.get("fit"), dict) else {}
            series_by_id[ddr_id].append(
                {
                    "bao_sigma_scale": float(f),
                    "max_abs_z": float(fit.get("max_abs_z")) if fit.get("max_abs_z") is not None else None,
                    "limiting_observation": str(fit.get("limiting_observation") or "na"),
                    "z_scores": dict(fit.get("z_scores") or {}),
                    "theta": dict(fit.get("theta") or {}),
                    "selected_constraints": {
                        "opacity": dict(block.get("opacity") or {}),
                        "candle": dict(block.get("candle") or {}),
                    },
                }
            )

    # Build matrix for plot (rows are DDR order; cols are scale order).
    labels: List[str] = []
    vals = np.full((len(ddr), len(scales)), np.nan, dtype=float)
    rows_out: List[Dict[str, Any]] = []
    for i, d in enumerate(ddr):
        labels.append(d.short_label)
        ser = series_by_id.get(d.id, [])
        xs = [float(x.get("bao_sigma_scale")) for x in ser]
        ys = [
            float(x.get("max_abs_z")) if x.get("max_abs_z") is not None and math.isfinite(float(x.get("max_abs_z"))) else float("nan")
            for x in ser
        ]
        for j, y in enumerate(ys):
            vals[i, j] = y

        f_to_1s = _crossing_x_log(xs, ys, 1.0)
        f_to_3s = _crossing_x_log(xs, ys, 3.0)

        rows_out.append(
            {
                "ddr": {
                    "id": d.id,
                    "short_label": d.short_label,
                    "uses_bao": bool(d.uses_bao),
                    "epsilon0_obs": float(d.epsilon0),
                    "epsilon0_sigma": float(d.epsilon0_sigma),
                    "epsilon0_sigma_raw": float(d.epsilon0_sigma_raw),
                    "sigma_sys_category": d.sigma_sys_category,
                    "sigma_policy": d.sigma_policy,
                    "category": d.category,
                },
                "series_best_independent": ser,
                "thresholds": {
                    "f_max_abs_z_le_1": f_to_1s,
                    "f_max_abs_z_le_3": f_to_3s,
                    "scales_scanned_max": float(max(scales)),
                },
            }
        )

    out_png = out_dir / "cosmology_distance_indicator_rederivation_bao_relax_thresholds.png"
    _plot_matrix(out_png=out_png, ddr_labels=labels, scales=scales, values=vals)

    # Summary
    reached_1s = 0
    reached_3s = 0
    f1s: List[Tuple[float, str, str]] = []
    for r in rows_out:
        th = r.get("thresholds") if isinstance(r.get("thresholds"), dict) else {}
        ddr_meta = r.get("ddr") if isinstance(r.get("ddr"), dict) else {}
        ddr_id = str(ddr_meta.get("id") or "")
        ddr_label = str(ddr_meta.get("short_label") or ddr_id)
        if th.get("f_max_abs_z_le_1") is not None:
            reached_1s += 1
            try:
                f1s.append((float(th["f_max_abs_z_le_1"]), ddr_id, ddr_label))
            except Exception:
                pass
        if th.get("f_max_abs_z_le_3") is not None:
            reached_3s += 1

    f1s_sorted = sorted([x for x in f1s if x[0] > 0.0 and math.isfinite(x[0])], key=lambda t: t[0])
    f1_only = [x[0] for x in f1s_sorted]
    f1_stats: Dict[str, Any] = {}
    if f1_only:
        import statistics

        def _pct(p: float) -> float:
            p = float(p)
            if not (0.0 <= p <= 1.0):
                return float("nan")
            idx = int(round(p * (len(f1_only) - 1)))
            return float(f1_only[max(0, min(len(f1_only) - 1, idx))])

        f1_min = float(min(f1_only))
        f1_med = float(statistics.median(f1_only))
        f1_max = float(max(f1_only))
        max_rows = [x for x in f1s_sorted if x[0] == f1_max]
        f1_stats = {
            "f1sigma_min": f1_min,
            "f1sigma_median": f1_med,
            "f1sigma_max": f1_max,
            "f1sigma_p25": _pct(0.25),
            "f1sigma_p75": _pct(0.75),
            "f1sigma_max_rows": [{"ddr_id": rid, "ddr_short_label": lab} for _, rid, lab in max_rows],
        }

    out_metrics = out_dir / "cosmology_distance_indicator_rederivation_bao_relax_thresholds_metrics.json"
    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "scan": "soften BAO(s_R) by scaling sigma: sigma_used = f * sigma_base",
            "objective": "for each DDR row and f, compute best_independent and its max|z| and limiting observation",
            "notes": [
                "This uses the same linear WLS model as candidate_search (θ=[s_R, α, s_L, p_t, p_e]).",
                "Candidate selection (opacity/candle) can change with f and with DDR row.",
                "f is a diagnostic knob (soft constraint), not a claim that BAO errors are actually f times larger.",
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
            "note": "If envelope file is missing, σ_cat inflation is skipped for all rows (falls back to raw).",
        },
        "fixed_constraints": {
            "p_t": {"id": p_t.id, "mean": p_t.mean, "sigma": p_t.sigma, "short_label": p_t.short_label},
            "p_e": {"id": p_e.id, "mean": p_e.mean, "sigma": p_e.sigma, "short_label": p_e.short_label},
            "bao_s_R_base": {"mean": sR_bao, "sigma_base": sR_bao_sigma_base},
        },
        "results": {
            "scales": scales,
            "rows": rows_out,
            "summary": {
                "rows_total": len(rows_out),
                "rows_reaching_max_abs_z_le_1_within_scan": reached_1s,
                "rows_reaching_max_abs_z_le_3_within_scan": reached_3s,
                **f1_stats,
            },
        },
        "outputs": {
            "png": str(out_png.relative_to(_ROOT)).replace("\\", "/"),
            "metrics_json": str(out_metrics.relative_to(_ROOT)).replace("\\", "/"),
        },
    }
    _write_json(out_metrics, payload)

    worklog.append_event(
        {
            "kind": "cosmology_distance_indicator_rederivation_bao_relax_thresholds",
            "script": "scripts/cosmology/cosmology_distance_indicator_rederivation_bao_relax_thresholds.py",
            "inputs": [in_ddr, in_opacity, in_candle, in_pt, in_pe, in_bao_fit, ddr_env_path],
            "outputs": [out_png, out_metrics],
            "ddr_sigma_policy": payload.get("ddr_sigma_policy"),
            "scales": scales,
            "summary": payload["results"]["summary"],
        }
    )

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
