#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_distance_indicator_rederivation_selected_sources.py

Phase 16（宇宙論）/ Step 16.4：
距離指標の再導出候補探索（candidate_search）で、各 DDR 一次ソースごとに
「どの一次ソース拘束（不透明度 α / 標準光源進化 s_L）が選ばれやすいか」を頻度として可視化する。

注意：
  - これは “真のメカニズム” の主張ではなく、探索（min max|z|）の性質上
    どの一次ソースが結果を支配しやすいか（選択バイアス）を見える化する目的。

入力（既存の固定出力）:
  - output/private/cosmology/cosmology_distance_indicator_rederivation_candidate_search_metrics.json

出力（固定名）:
  - output/private/cosmology/cosmology_distance_indicator_rederivation_selected_sources.png
  - output/private/cosmology/cosmology_distance_indicator_rederivation_selected_sources_metrics.json
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


def _extract_label(block: Dict[str, Any], key: str) -> str:
    v = block.get(key)
    if not isinstance(v, dict):
        return ""
    label = str(v.get("short_label") or v.get("id") or "").strip()
    return label


def _plot_counts(
    ax: Any,
    *,
    title: str,
    counts_any: Counter[str],
    counts_ind: Counter[str],
    xlabel: str,
) -> None:
    labels = sorted(set(counts_any) | set(counts_ind), key=lambda k: (-counts_ind[k], -counts_any[k], k))
    if not labels:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.axis("off")
        return

    y = np.arange(len(labels), dtype=float)
    h = 0.38

    any_vals = np.array([counts_any.get(k, 0) for k in labels], dtype=float)
    ind_vals = np.array([counts_ind.get(k, 0) for k in labels], dtype=float)

    ax.barh(y - h / 2, any_vals, height=h, color="#e0e0e0", edgecolor="#333333", linewidth=0.8, label="best_any")
    ax.barh(
        y + h / 2,
        ind_vals,
        height=h,
        color="#e0e0e0",
        edgecolor="#333333",
        linewidth=0.8,
        hatch="///",
        label="best_independent",
    )

    x_max = float(max(any_vals.max(), ind_vals.max(), 1.0))
    ax.set_xlim(0.0, x_max * 1.25 + 0.25)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=10)

    for yi, (a, b) in enumerate(zip(any_vals, ind_vals)):
        if a > 0:
            ax.text(a + 0.05, yi - h / 2, f"{int(a)}", va="center", ha="left", fontsize=9)
        if b > 0:
            ax.text(b + 0.05, yi + h / 2, f"{int(b)}", va="center", ha="left", fontsize=9)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize which primary-source constraints are selected in candidate_search (Phase 16 / Step 16.4)."
    )
    parser.add_argument(
        "--in-metrics",
        default=str(
            _ROOT
            / "output"
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
    if not in_metrics.exists():
        raise FileNotFoundError(
            f"missing required metrics: {in_metrics} (run scripts/summary/run_all.py --offline first)"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_japanese_font()

    src = _read_json(in_metrics)
    per_ddr = ((src.get("results") or {}).get("per_ddr")) if isinstance(src.get("results"), dict) else None
    if not isinstance(per_ddr, list) or not per_ddr:
        raise ValueError("invalid candidate_search metrics: results.per_ddr missing or empty")

    opacity_any = Counter()
    opacity_ind = Counter()
    candle_any = Counter()
    candle_ind = Counter()

    for item in per_ddr:
        if not isinstance(item, dict):
            continue
        best_any = item.get("best_any") if isinstance(item.get("best_any"), dict) else {}
        best_ind = item.get("best_independent") if isinstance(item.get("best_independent"), dict) else {}

        op_a = _extract_label(best_any, "opacity")
        op_i = _extract_label(best_ind, "opacity")
        ca_a = _extract_label(best_any, "candle")
        ca_i = _extract_label(best_ind, "candle")

        if op_a:
            opacity_any[op_a] += 1
        if op_i:
            opacity_ind[op_i] += 1
        if ca_a:
            candle_any[ca_a] += 1
        if ca_i:
            candle_ind[ca_i] += 1

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 2, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    _plot_counts(
        ax1,
        title="不透明度 α：選択された一次ソース（頻度）",
        counts_any=opacity_any,
        counts_ind=opacity_ind,
        xlabel="選択回数（DDR一次ソースごとの best_* で集計）",
    )
    _plot_counts(
        ax2,
        title="標準光源進化 s_L：選択された一次ソース（頻度）",
        counts_any=candle_any,
        counts_ind=candle_ind,
        xlabel="選択回数（DDR一次ソースごとの best_* で集計）",
    )

    fig.suptitle("宇宙論（再接続候補探索）：選択された一次ソース拘束の偏り（best_any vs best_independent）", y=0.98)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#e0e0e0", edgecolor="#333333", label="best_any"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#e0e0e0", edgecolor="#333333", hatch="///", label="best_independent"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.14, wspace=0.25)

    out_png = out_dir / "cosmology_distance_indicator_rederivation_selected_sources.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    out_json = out_dir / "cosmology_distance_indicator_rederivation_selected_sources_metrics.json"
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_metrics": str(in_metrics).replace("\\", "/"),
        "ddr_sigma_policy": src.get("ddr_sigma_policy") if isinstance(src.get("ddr_sigma_policy"), dict) else {},
        "counts": {
            "best_any": {
                "opacity": {k: int(v) for k, v in opacity_any.most_common()},
                "candle": {k: int(v) for k, v in candle_any.most_common()},
            },
            "best_independent": {
                "opacity": {k: int(v) for k, v in opacity_ind.most_common()},
                "candle": {k: int(v) for k, v in candle_ind.most_common()},
            },
        },
        "note": "min max|z| 探索のため、σが大きい/系統が大きい一次ソースが選ばれやすい可能性がある（機構の主張ではない）。",
    }
    _write_json(out_json, payload)

    worklog.append_event(
        {
            "kind": "cosmology_distance_indicator_rederivation_selected_sources",
            "script": "scripts/cosmology/cosmology_distance_indicator_rederivation_selected_sources.py",
            "input": in_metrics,
            "outputs": [out_png, out_json],
            "counts_best_any_opacity": dict(opacity_any),
            "counts_best_any_candle": dict(candle_any),
            "counts_best_independent_opacity": dict(opacity_ind),
            "counts_best_independent_candle": dict(candle_ind),
        }
    )

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
