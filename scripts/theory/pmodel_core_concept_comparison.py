from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "private" / "theory"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _pick_font_family(candidates: list[str]) -> Optional[str]:
    try:
        from matplotlib import font_manager

        available = {f.name for f in font_manager.fontManager.ttflist}
        for c in candidates:
            if c in available:
                return c
    except Exception:
        return None
    return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "pmodel_core_concept_comparison.png"
    out_json = OUT_DIR / "pmodel_core_concept_comparison_metrics.json"

    # Prefer Japanese-capable fonts when available (Windows), but keep a safe fallback.
    font_candidates = [
        "Yu Gothic",
        "Yu Gothic UI",
        "Meiryo",
        "MS Gothic",
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAexGothic",
        "DejaVu Sans",
    ]
    chosen_font = _pick_font_family(font_candidates)
    if chosen_font:
        plt.rcParams["font.family"] = chosen_font
    plt.rcParams["axes.unicode_minus"] = False

    col_labels = ["観点", "参照枠（GR）", "P-model"]
    cell_text = [
        ["重力の本質", "時空の曲率", "P勾配への応答"],
        ["運動の記述", "測地線に沿う", "P勾配へ滑り落ちる"],
        ["光の伝播", "光は時空に沿う", "光は高P側へ屈折"],
        ["時間の遅れ", "時空の計量", "P比（P0/P）"],
        ["赤方偏移", "空間膨張", "背景Pの時間変化"],
    ]

    fig, ax = plt.subplots(figsize=(12.2, 3.6), dpi=200)
    ax.set_axis_off()

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11.0)
    tbl.scale(1.0, 1.55)

    # Style cells: header + alternating rows
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("0.35")
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor("#f1f1f1")
            cell.set_text_props(weight="bold", color="0.15")
        else:
            cell.set_facecolor("#ffffff" if (r % 2 == 1) else "#fbfbfb")
        # Column widths (approx)
        if c == 0:
            cell.set_width(0.24)
        elif c == 1:
            cell.set_width(0.37)
        elif c == 2:
            cell.set_width(0.37)

    caption = (
        "図2: P-modelと参照枠（GR）の概念比較。P-modelは時空の幾何ではなく、"
        "時間波密度Pの空間変化として重力・光伝播を記述する。"
        "両者は弱場で同等の観測量を与えるが、概念的枠組みは独立である。"
    )
    fig.text(
        0.5,
        0.03,
        textwrap.fill(caption, width=64),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="0.25",
    )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    _write_json(
        out_json,
        {
            "generated_utc": _utc_now(),
            "script": "scripts/theory/pmodel_core_concept_comparison.py",
            "outputs": {"png": str(out_png), "metrics_json": str(out_json)},
            "figure_index_path": "output/private/theory/pmodel_core_concept_comparison.png",
            "diag": {"font_family": chosen_font or "default"},
            "notes": [
                "This is a conceptual comparison diagram (not a numerical result).",
                "GR is used as a reference frame only; the Part I text defines the mapping on the P-model side.",
            ],
        },
    )

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")


if __name__ == "__main__":
    main()
