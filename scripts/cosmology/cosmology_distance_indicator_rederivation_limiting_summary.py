#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_distance_indicator_rederivation_limiting_summary.py

Phase 16（宇宙論）/ Step 16.4：
距離指標の再導出候補探索（candidate_search）の結果について、
各 DDR 一次ソースごとに「どの観測が支配拘束（limiting）になっているか」を固定出力する。

入力（既存の固定出力）:
  - output/private/cosmology/cosmology_distance_indicator_rederivation_candidate_search_metrics.json

出力（固定名）:
  - output/private/cosmology/cosmology_distance_indicator_rederivation_limiting_summary.png
  - output/private/cosmology/cosmology_distance_indicator_rederivation_limiting_summary_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


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


def _safe_float(x: Any) -> Optional[float]:
    try:
        # 条件分岐: `x is None` を満たす経路を評価する。
        if x is None:
            return None

        v = float(x)
        # 条件分岐: `not math.isfinite(v)` を満たす経路を評価する。
        if not math.isfinite(v):
            return None

        return v
    except Exception:
        return None


def _classify_sigma(abs_z: Optional[float]) -> str:
    # 条件分岐: `abs_z is None or not math.isfinite(float(abs_z))` を満たす経路を評価する。
    if abs_z is None or not math.isfinite(float(abs_z)):
        return "na"

    # 条件分岐: `abs_z < 3.0` を満たす経路を評価する。

    if abs_z < 3.0:
        return "ok"

    # 条件分岐: `abs_z < 5.0` を満たす経路を評価する。

    if abs_z < 5.0:
        return "mixed"

    return "ng"


def _extract_fit(per_ddr: Dict[str, Any], key: str) -> Tuple[Optional[float], str, Optional[float], Optional[float]]:
    """
    Returns:
      (max_abs_z, limiting_observation, chi2_dof, chi2)
    """
    block = per_ddr.get(key)
    # 条件分岐: `not isinstance(block, dict)` を満たす経路を評価する。
    if not isinstance(block, dict):
        return (None, "na", None, None)

    fit = block.get("fit")
    # 条件分岐: `not isinstance(fit, dict)` を満たす経路を評価する。
    if not isinstance(fit, dict):
        return (None, "na", None, None)

    max_abs_z = _safe_float(fit.get("max_abs_z"))
    limiting = str(fit.get("limiting_observation") or "na")
    chi2_dof = _safe_float(fit.get("chi2_dof"))
    chi2 = _safe_float(fit.get("chi2"))
    return (max_abs_z, limiting, chi2_dof, chi2)


def _extract_choice(per_ddr: Dict[str, Any], key: str) -> Dict[str, Any]:
    block = per_ddr.get(key)
    # 条件分岐: `not isinstance(block, dict)` を満たす経路を評価する。
    if not isinstance(block, dict):
        return {}

    out: Dict[str, Any] = {}
    for k in ("opacity", "candle"):
        v = block.get(k)
        # 条件分岐: `not isinstance(v, dict)` を満たす経路を評価する。
        if not isinstance(v, dict):
            continue

        out[k] = {
            "id": str(v.get("id") or ""),
            "short_label": str(v.get("short_label") or v.get("id") or ""),
        }

    return out


def _limiting_palette() -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    Returns:
      (ordered_labels, label->color, label->short)
    """
    ordered = [
        "BAO s_R",
        "DDR ε0",
        "Opacity α",
        "Candle s_L",
        "SN time dilation p_t",
        "CMB energy p_e",
        "na",
    ]

    colors = {
        "BAO s_R": "#9467bd",  # purple
        "DDR ε0": "#1f77b4",  # blue
        "Opacity α": "#2ca02c",  # green
        "Candle s_L": "#ff7f0e",  # orange
        "SN time dilation p_t": "#17becf",  # teal
        "CMB energy p_e": "#7f7f7f",  # gray
        "na": "#c7c7c7",
    }

    short = {
        "BAO s_R": "BAO",
        "DDR ε0": "DDR",
        "Opacity α": "α",
        "Candle s_L": "s_L",
        "SN time dilation p_t": "p_t",
        "CMB energy p_e": "p_e",
        "na": "na",
    }
    return (ordered, colors, short)


def _fmt(x: Optional[float], *, digits: int = 3) -> str:
    # 条件分岐: `x is None or not math.isfinite(float(x))` を満たす経路を評価する。
    if x is None or not math.isfinite(float(x)):
        return ""

    x = float(x)
    # 条件分岐: `x == 0.0` を満たす経路を評価する。
    if x == 0.0:
        return "0"

    ax = abs(x)
    # 条件分岐: `ax >= 1e4 or ax < 1e-3` を満たす経路を評価する。
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"

    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize limiting observations in distance-indicator rederivation candidate search (Phase 16 / Step 16.4)."
    )
    parser.add_argument(
        "--in-metrics",
        default=str(
            _ROOT
            / "output"
            / "private"
            / "cosmology"
            / "cosmology_distance_indicator_rederivation_candidate_search_metrics.json"
        ),
        help="Input metrics JSON (default: output/private/cosmology/...candidate_search_metrics.json)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_ROOT / "output" / "private" / "cosmology"),
        help="Output directory (default: output/private/cosmology)",
    )
    args = parser.parse_args(argv)

    in_metrics = Path(args.in_metrics)
    # 条件分岐: `not in_metrics.exists()` を満たす経路を評価する。
    if not in_metrics.exists():
        legacy = _ROOT / "output" / "cosmology" / "cosmology_distance_indicator_rederivation_candidate_search_metrics.json"
        # 条件分岐: `legacy.exists()` を満たす経路を評価する。
        if legacy.exists():
            in_metrics = legacy
        else:
            raise FileNotFoundError(
                f"missing required metrics: {in_metrics} (run scripts/summary/run_all.py --offline first)"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_japanese_font()

    src = _read_json(in_metrics)
    per_ddr = ((src.get("results") or {}).get("per_ddr")) if isinstance(src.get("results"), dict) else None
    # 条件分岐: `not isinstance(per_ddr, list) or not per_ddr` を満たす経路を評価する。
    if not isinstance(per_ddr, list) or not per_ddr:
        raise ValueError("invalid candidate_search metrics: results.per_ddr missing or empty")

    scenarios = [
        ("best_any", "best_any", ""),
        ("best_independent", "best_independent", "///"),
    ]

    ordered_labels, colors, short = _limiting_palette()

    rows_out: List[Dict[str, Any]] = []
    counts: Dict[str, Counter[str]] = {k: Counter() for k, _, _ in scenarios}

    chosen_opacity: Dict[str, Counter[str]] = {"best_any": Counter(), "best_independent": Counter()}
    chosen_candle: Dict[str, Counter[str]] = {"best_any": Counter(), "best_independent": Counter()}

    items: List[Dict[str, Any]] = []
    for item in per_ddr:
        # 条件分岐: `not isinstance(item, dict)` を満たす経路を評価する。
        if not isinstance(item, dict):
            continue

        ddr = item.get("ddr")
        # 条件分岐: `not isinstance(ddr, dict)` を満たす経路を評価する。
        if not isinstance(ddr, dict):
            continue

        label = str(ddr.get("short_label") or ddr.get("id") or "").strip()
        # 条件分岐: `not label` を満たす経路を評価する。
        if not label:
            continue

        (z_any, lim_a, chi2_dof_a, chi2_a) = _extract_fit(item, "best_any")
        (z_ind, lim_i, chi2_dof_i, chi2_i) = _extract_fit(item, "best_independent")

        counts["best_any"][lim_a] += 1
        counts["best_independent"][lim_i] += 1

        choice_any = _extract_choice(item, "best_any")
        choice_ind = _extract_choice(item, "best_independent")
        for scenario_key, choice in (("best_any", choice_any), ("best_independent", choice_ind)):
            op = (choice.get("opacity") or {}).get("short_label") or (choice.get("opacity") or {}).get("id") or ""
            ca = (choice.get("candle") or {}).get("short_label") or (choice.get("candle") or {}).get("id") or ""
            # 条件分岐: `op` を満たす経路を評価する。
            if op:
                chosen_opacity[scenario_key][str(op)] += 1

            # 条件分岐: `ca` を満たす経路を評価する。

            if ca:
                chosen_candle[scenario_key][str(ca)] += 1

        row = (
            {
                "ddr": {
                    "id": str(ddr.get("id") or ""),
                    "short_label": label,
                    "uses_bao": bool(ddr.get("uses_bao", False)),
                    "epsilon0_obs": _safe_float(ddr.get("epsilon0_obs")),
                    "epsilon0_sigma": _safe_float(ddr.get("epsilon0_sigma")),
                    "epsilon0_sigma_raw": _safe_float(ddr.get("epsilon0_sigma_raw")),
                    "sigma_sys_category": _safe_float(ddr.get("sigma_sys_category")),
                    "sigma_policy": str(ddr.get("sigma_policy") or "raw"),
                    "category": ddr.get("category"),
                },
                "best_any": {
                    "max_abs_z": z_any,
                    "status": _classify_sigma(z_any),
                    "limiting_observation": lim_a,
                    "chi2": chi2_a,
                    "chi2_dof": chi2_dof_a,
                    **choice_any,
                },
                "best_independent": {
                    "max_abs_z": z_ind,
                    "status": _classify_sigma(z_ind),
                    "limiting_observation": lim_i,
                    "chi2": chi2_i,
                    "chi2_dof": chi2_dof_i,
                    **choice_ind,
                },
            }
        )
        rows_out.append(row)
        items.append(
            {
                "label": label,
                "z_any": z_any,
                "z_ind": z_ind,
                "lim_any": lim_a,
                "lim_ind": lim_i,
            }
        )

    # Sort by best_independent max|z| ascending (then label) for readability.

    def _z_for_sort(v: Optional[float]) -> float:
        # 条件分岐: `v is None or not math.isfinite(float(v))` を満たす経路を評価する。
        if v is None or not math.isfinite(float(v)):
            return float("inf")

        return float(v)

    sort_keys = sorted(
        range(len(items)),
        key=lambda i: (_z_for_sort(items[i]["z_ind"]), _z_for_sort(items[i]["z_any"]), str(items[i]["label"])),
    )
    items = [items[i] for i in sort_keys]
    rows_out = [rows_out[i] for i in sort_keys]

    # --- Figure
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    labels = [it["label"] for it in items]
    vals_any = [it["z_any"] for it in items]
    vals_ind = [it["z_ind"] for it in items]
    lim_any = [it["lim_any"] for it in items]
    lim_ind = [it["lim_ind"] for it in items]

    n = len(labels)
    y = np.arange(n, dtype=float)
    h = 0.36

    max_z = 0.0
    for v in (vals_any + vals_ind):
        # 条件分岐: `v is not None and math.isfinite(float(v))` を満たす経路を評価する。
        if v is not None and math.isfinite(float(v)):
            max_z = max(max_z, float(v))

    x_max = max(1.5, max_z * 1.10 + 0.25)
    # 条件分岐: `x_max < 6.0` を満たす経路を評価する。
    if x_max < 6.0:
        x_max = 6.0

    fig, ax = plt.subplots(figsize=(16, 8))

    # Bars
    def _color_for(lim: str) -> str:
        # 条件分岐: `lim in colors` を満たす経路を評価する。
        if lim in colors:
            return colors[lim]

        return colors["na"]

    bars_any = ax.barh(
        y - h / 2,
        [0.0 if v is None else float(v) for v in vals_any],
        height=h,
        color=[_color_for(l) for l in lim_any],
        edgecolor="#333333",
        linewidth=0.8,
        label="best_any",
    )
    bars_ind = ax.barh(
        y + h / 2,
        [0.0 if v is None else float(v) for v in vals_ind],
        height=h,
        color=[_color_for(l) for l in lim_ind],
        edgecolor="#333333",
        linewidth=0.8,
        hatch="///",
        label="best_independent",
    )

    # Annotate
    for bars, vals, lims in ((bars_any, vals_any, lim_any), (bars_ind, vals_ind, lim_ind)):
        for b, v, lim in zip(bars, vals, lims):
            # 条件分岐: `v is None or not math.isfinite(float(v))` を満たす経路を評価する。
            if v is None or not math.isfinite(float(v)):
                continue

            x = float(v)
            ax.text(
                x + x_max * 0.01,
                b.get_y() + b.get_height() / 2,
                f"{_fmt(x, digits=2)} ({short.get(lim, 'na')})",
                va="center",
                ha="left",
                fontsize=8,
                color="#111111",
            )

    # Reference lines (σ guide)

    for x0, label in ((1.0, "1σ"), (3.0, "3σ"), (5.0, "5σ")):
        ax.axvline(x0, color="#999999", lw=1.0, ls="--")
        ax.text(x0, -0.9, label, ha="center", va="bottom", fontsize=9, color="#666666")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("最大|z|（WLS: DDR + BAO(s_R) + α + s_L + p_t + p_e の同時整合）")
    ax.set_xlim(0.0, x_max)

    pol = src.get("ddr_sigma_policy") if isinstance(src.get("ddr_sigma_policy"), dict) else {}
    pol_label = str(pol.get("policy") or "raw")
    ax.set_title(
        f"宇宙論（再接続候補探索）：支配拘束（limiting）の一覧（DDR σ: {pol_label}）",
        pad=18,
    )

    # Legends: scenario (hatch) and limiting category (colors)
    scenario_handles = [
        Patch(facecolor="#e0e0e0", edgecolor="#333333", label="best_any"),
        Patch(facecolor="#e0e0e0", edgecolor="#333333", hatch="///", label="best_independent"),
    ]
    leg1 = ax.legend(handles=scenario_handles, title="系列", loc="lower right", frameon=True)
    ax.add_artist(leg1)

    limiting_handles = []
    for k in ordered_labels:
        limiting_handles.append(Patch(facecolor=colors[k], edgecolor="#333333", label=k))

    ax.legend(handles=limiting_handles, title="limiting（支配する拘束）", loc="upper right", frameon=True)

    fig.tight_layout()

    out_png = out_dir / "cosmology_distance_indicator_rederivation_limiting_summary.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    out_metrics = out_dir / "cosmology_distance_indicator_rederivation_limiting_summary_metrics.json"
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_metrics": str(in_metrics).replace("\\", "/"),
        "ddr_sigma_policy": pol if isinstance(pol, dict) else {},
        "scenarios": [k for k, _, _ in scenarios],
        "limiting_labels_order": ordered_labels,
        "counts": {
            "best_any": {k: int(v) for k, v in sorted(counts["best_any"].items(), key=lambda kv: (-kv[1], kv[0]))},
            "best_independent": {
                k: int(v) for k, v in sorted(counts["best_independent"].items(), key=lambda kv: (-kv[1], kv[0]))
            },
        },
        "chosen_constraints": {
            "best_any": {
                "opacity": {k: int(v) for k, v in chosen_opacity["best_any"].most_common()},
                "candle": {k: int(v) for k, v in chosen_candle["best_any"].most_common()},
            },
            "best_independent": {
                "opacity": {k: int(v) for k, v in chosen_opacity["best_independent"].most_common()},
                "candle": {k: int(v) for k, v in chosen_candle["best_independent"].most_common()},
            },
        },
        "rows": rows_out,
    }
    _write_json(out_metrics, payload)

    worklog.append_event(
        {
            "kind": "cosmology_distance_indicator_rederivation_limiting_summary",
            "script": "scripts/cosmology/cosmology_distance_indicator_rederivation_limiting_summary.py",
            "input": in_metrics,
            "outputs": [out_png, out_metrics],
            "ddr_sigma_policy": pol,
            "counts_best_any": dict(counts["best_any"]),
            "counts_best_independent": dict(counts["best_independent"]),
        }
    )

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_metrics}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
