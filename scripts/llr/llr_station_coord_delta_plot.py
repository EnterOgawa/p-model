#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llr_station_coord_delta_plot.py

Phase 8 / Step 8.4.2（図表品質改善）:
`output/private/llr/batch/llr_station_metadata_used.json` から
`output/private/llr/batch/llr_station_coord_delta_pos_eop.png` を高速に再生成する。

目的：
- `scripts/llr/llr_batch_eval.py` は重いので、図だけを差し替えたいときの軽量再生成経路を用意する。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _set_japanese_font() -> None:
    try:
        import japanize_matplotlib  # type: ignore  # noqa: F401

        return
    except Exception:
        pass

    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        candidates = ["IPAexGothic", "IPAGothic", "Yu Gothic", "Meiryo", "Noto Sans CJK JP"]
        installed = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in candidates if name in installed]
        # 条件分岐: `chosen` を満たす経路を評価する。
        if chosen:
            mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_delta_m(stations: Dict[str, Any]) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for st, meta in sorted(stations.items(), key=lambda kv: str(kv[0])):
        # 条件分岐: `not isinstance(meta, dict)` を満たす経路を評価する。
        if not isinstance(meta, dict):
            continue

        dr = meta.get("delta_vs_slrlog_m")
        # 条件分岐: `not isinstance(dr, (int, float))` を満たす経路を評価する。
        if not isinstance(dr, (int, float)):
            continue

        v = float(dr)
        # 条件分岐: `not math.isfinite(v)` を満たす経路を評価する。
        if not math.isfinite(v):
            continue

        out.append((str(st), v))

    return out


def main(argv: List[str] | None = None) -> int:
    default_in = _ROOT / "output" / "private" / "llr" / "batch" / "llr_station_metadata_used.json"
    default_out = _ROOT / "output" / "private" / "llr" / "batch" / "llr_station_coord_delta_pos_eop.png"

    ap = argparse.ArgumentParser(description="Replot LLR station coordinate delta (pos+eop vs slrlog).")
    ap.add_argument("--in-json", type=str, default=str(default_in))
    ap.add_argument("--out-png", type=str, default=str(default_out))
    ap.add_argument("--title", type=str, default="", help="optional title override")
    args = ap.parse_args(list(argv) if argv is not None else None)

    in_json = Path(str(args.in_json))
    # 条件分岐: `not in_json.is_absolute()` を満たす経路を評価する。
    if not in_json.is_absolute():
        in_json = (_ROOT / in_json).resolve()

    # 条件分岐: `not in_json.exists()` を満たす経路を評価する。

    if not in_json.exists():
        print(f"[err] missing: {in_json}")
        return 2

    out_png = Path(str(args.out_png))
    # 条件分岐: `not out_png.is_absolute()` を満たす経路を評価する。
    if not out_png.is_absolute():
        out_png = (_ROOT / out_png).resolve()

    out_png.parent.mkdir(parents=True, exist_ok=True)

    d = _read_json(in_json)
    stations = d.get("stations") if isinstance(d.get("stations"), dict) else {}
    items = _extract_delta_m(stations)
    # 条件分岐: `not items` を満たす経路を評価する。
    if not items:
        print("[err] no delta_vs_slrlog_m values found in input json")
        return 2

    _set_japanese_font()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[err] matplotlib unavailable: {e}")
        return 2

    xs = [k for k, _ in items]
    ys = [v for _, v in items]
    width_in = max(12.0, 0.6 * len(xs) + 4.0)

    fig, ax = plt.subplots(figsize=(width_in, 6.0), dpi=200)
    ax.bar(xs, ys)
    ax.set_ylabel("||Δr|| [m]（pos+eop - slrlog）")
    title = str(args.title).strip() or "LLR：局座標の差分（pos+eop - slrlog）"
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    worklog.append_event(
        {
            "domain": "llr",
            "action": "llr_station_coord_delta_plot",
            "inputs": [str(in_json).replace("\\", "/")],
            "outputs": [str(out_png).replace("\\", "/")],
            "params": {"title": title},
        }
    )

    print(f"[ok] png: {out_png}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
