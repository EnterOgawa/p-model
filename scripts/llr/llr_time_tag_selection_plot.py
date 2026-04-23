#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llr_time_tag_selection_plot.py

目的:
- `output/private/llr/batch/llr_time_tag_best_by_station.json` から
  LLR time-tag selection by station 図を軽量に再生成する。
- 重い `llr_batch_eval.py` を回さずに、Part IV 再掲図の言語面と紙面品質だけを差し替える。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402
from scripts.utils.figure_locale_paths import localize_figure_output_path  # noqa: E402
from scripts.utils.plot_style import get_wavep_font_size  # noqa: E402


# 関数: `_read_json` の入出力契約と処理意図を定義する。
def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_resolve_mode_values` の入出力契約と処理意図を定義する。
def _resolve_mode_values(payload: Dict[str, Any], *, station: str, mode: str) -> float:
    per_station = payload.get("rms_by_station_and_mode_ns")
    if not isinstance(per_station, dict):
        return float("nan")

    station_row = per_station.get(station)
    if not isinstance(station_row, dict):
        return float("nan")

    value = station_row.get(mode)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return float("nan")

    return float(value)


# 関数: `main` の入出力契約と処理意図を定義する。
def main(argv: List[str] | None = None) -> int:
    default_in = ROOT / "output" / "private" / "llr" / "batch" / "llr_time_tag_best_by_station.json"
    default_out = ROOT / "output" / "public" / "llr" / "llr_time_tag_selection_by_station.png"

    ap = argparse.ArgumentParser(description="Replot LLR time-tag selection by station from cached selection JSON.")
    ap.add_argument("--in-json", type=str, default=str(default_in))
    ap.add_argument("--out-png", type=str, default=str(default_out))
    args = ap.parse_args(list(argv) if argv is not None else None)

    in_json = Path(str(args.in_json))
    if not in_json.is_absolute():
        in_json = (ROOT / in_json).resolve()

    if not in_json.exists():
        print(f"[err] missing: {in_json}")
        return 2

    out_png = localize_figure_output_path(Path(str(args.out_png)), root=ROOT)
    if not out_png.is_absolute():
        out_png = (ROOT / out_png).resolve()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = out_png.with_suffix(".pdf")

    payload = _read_json(in_json)
    best_by_station = payload.get("best_mode_by_station")
    if not isinstance(best_by_station, dict) or not best_by_station:
        print("[err] best_mode_by_station is missing or empty")
        return 2

    stations = sorted(str(st) for st in best_by_station.keys())
    mode_order = ["tx", "rx", "mid"]
    mode_labels = {"tx": "tx", "rx": "rx", "mid": "mid"}

    figure_lang = str(os.getenv("WAVEP_FIGURE_LANG", "ja")).strip().lower()
    is_en = figure_lang.startswith("en")
    font_scale = 1.0
    tick_scale = 1.18 if is_en else 1.0
    x = np.arange(len(stations), dtype=float)
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.4, 4.5), dpi=180)
    for offset_index, mode in enumerate(mode_order):
        values = [_resolve_mode_values(payload, station=st, mode=mode) for st in stations]
        ax.bar(x + (offset_index - 1) * width, values, width=width, label=mode_labels[mode], alpha=0.88)

    best_markers = [
        _resolve_mode_values(payload, station=st, mode=str(best_by_station.get(st) or "tx"))
        for st in stations
    ]
    ax.scatter(x, best_markers, color="black", s=18, zorder=4, label=("best per station" if is_en else "局ごとの最良"))
    ax.set_xticks(x, stations)
    ax.set_yscale("log")
    ax.set_ylabel(
        "Residual RMS [ns]\n(station-reflector, offset-aligned, station-weighted)"
        if is_en
        else "残差RMS [ns]（観測局→反射器, 定数整列, 局内重み付き）",
        fontsize=get_wavep_font_size("axis") * font_scale,
    )
    ax.set_title(
        "LLR time-tag selection by station (tx / rx / mid)"
        if is_en
        else "LLR：time-tag 最適化（局ごとに tx/rx/mid を選択）",
        fontsize=get_wavep_font_size("title") * font_scale,
    )
    tick_font = get_wavep_font_size("tick") * tick_scale
    ax.tick_params(axis="x", labelsize=tick_font)
    ax.tick_params(axis="y", labelsize=tick_font)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=get_wavep_font_size("legend") * font_scale, ncols=4 if is_en else 2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    worklog.append_event(
        {
            "domain": "llr",
            "action": "llr_time_tag_selection_plot",
            "inputs": [str(in_json).replace("\\", "/")],
            "outputs": [str(out_png).replace("\\", "/"), str(out_pdf).replace("\\", "/")],
        }
    )

    print(f"[ok] png: {out_png}")
    print(f"[ok] pdf: {out_pdf}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
