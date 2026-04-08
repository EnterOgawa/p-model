"""
目的: 理論 topic の pmodel rejection protocol flowchart に対応する公開図・表・監査指標を再生成する。
入力: script 内の既定パラメータと必要な公開データまたは基準値を用いる。
出力: output/public と output/private の canonical artifact を更新する。
前提: 論文本文と README はこの script が出力する公開成果物を正として参照する。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scripts.utils.plot_style import (
    apply_paper_style,
    apply_wavep_figure_layout,
    resolve_wavep_cjk_font_family,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR_CANON = ROOT / "output" / "theory"
OUT_DIR_PUBLIC = ROOT / "output" / "public" / "theory"
OUT_DIR_PRIVATE = ROOT / "output" / "private" / "theory"


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。

def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = resolve_wavep_cjk_font_family(preferred_name="Noto Sans CJK JP")
        if preferred:
            mpl.rcParams["font.family"] = [preferred, "DejaVu Sans"]
            mpl.rcParams["font.sans-serif"] = [preferred, "DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
            return

        available = {font.name for font in fm.fontManager.ttflist}
        fallback = ["Yu Gothic", "Meiryo", "BIZ UDGothic", "MS Gothic", "Yu Mincho", "MS Mincho"]
        selected = [name for name in fallback if name in available]
        if selected:
            mpl.rcParams["font.family"] = selected + ["DejaVu Sans"]
            mpl.rcParams["font.sans-serif"] = selected + ["DejaVu Sans"]

        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        return


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `_save_figure_bundle` の入出力契約と処理意図を定義する。

def _save_figure_bundle(*, fig: Any, stem: str) -> Dict[str, str]:
    outputs = {
        "png_canon": OUT_DIR_CANON / f"{stem}.png",
        "png_public": OUT_DIR_PUBLIC / f"{stem}.png",
        "png_private": OUT_DIR_PRIVATE / f"{stem}.png",
        "pdf_canon": OUT_DIR_CANON / f"{stem}.pdf",
        "pdf_public": OUT_DIR_PUBLIC / f"{stem}.pdf",
        "pdf_private": OUT_DIR_PRIVATE / f"{stem}.pdf",
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        for path in outputs.values():
            if path.suffix.lower() == ".png":
                fig.savefig(path, dpi=220)
            else:
                fig.savefig(path)

    return {key: str(value).replace("\\", "/") for key, value in outputs.items()}


# 関数: `_draw_box` の入出力契約と処理意図を定義する。

def _draw_box(
    *,
    ax: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        transform=ax.transAxes,
        facecolor=color,
        edgecolor="0.25",
        linewidth=1.2,
    )
    ax.add_patch(patch)
    title_text = ax.text(
        x + w * 0.5,
        y + h * 0.67,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="0.10",
    )
    title_text.set_fontsize(10.2)
    title_text.set_fontweight("bold")
    body_text = ax.text(
        x + w * 0.5,
        y + h * 0.33,
        body,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.0,
        color="0.15",
        linespacing=1.22,
    )
    body_text.set_fontsize(8.6)


# 関数: `_draw_arrow` の入出力契約と処理意図を定義する。

def _draw_arrow(
    *,
    ax: Any,
    start: Tuple[float, float],
    end: Tuple[float, float],
    color: str = "0.30",
    linewidth: float = 2.1,
    mutation_scale: float = 22.0,
    zorder: float = 6.0,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        zorder=zorder,
    )
    ax.add_patch(arrow)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    apply_paper_style()
    _set_japanese_font()

    out_json_private = OUT_DIR_PRIVATE / "pmodel_rejection_protocol_flowchart_metrics.json"

    figure, axis = plt.subplots(dpi=220)
    apply_wavep_figure_layout(figure, template="paper_flowchart")
    axis.set_axis_off()

    steps: List[Dict[str, str]] = [
        {"title": "入力", "body": "一次データ\n依存前提\n取得元", "color": "#eef3ff"},
        {"title": "凍結", "body": "凍結パラメータ\n凍結根拠\n固定時点", "color": "#f2edff"},
        {"title": "統計", "body": "RMS, χ², z\nΔAIC, 傾き\n誤差伝播", "color": "#fff7ec"},
        {"title": "判定", "body": "閾値判定\n(3σ, ΔAIC)\n判定分岐へ", "color": "#ffe8e8"},
        {"title": "出力", "body": "固定ファイル名\n再現コマンド\n監査ログ", "color": "#eaf8ea"},
    ]

    box_width = 0.148
    box_height = 0.56
    x0 = 0.035
    y0 = 0.31
    gap = 0.034

    x_positions: List[float] = []

    for index, step in enumerate(steps):
        x = x0 + index * (box_width + gap)
        x_positions.append(x)
        _draw_box(
            ax=axis,
            x=x,
            y=y0,
            w=box_width,
            h=box_height,
            title=step["title"],
            body=step["body"],
            color=step["color"],
        )

    # 条件分岐: `len(x_positions) >= 2` を満たす経路を評価する。

    if len(x_positions) >= 2:
        for left_x, right_x in zip(x_positions[:-1], x_positions[1:]):
            start = (left_x + box_width, y0 + box_height * 0.5)
            end = (right_x, y0 + box_height * 0.5)
            _draw_arrow(ax=axis, start=start, end=end, color="0.30", linewidth=2.7, mutation_scale=25.0, zorder=8.0)

    x_reject = x0 + 3 * (box_width + gap)
    x_output = x0 + 4 * (box_width + gap)
    branch_shift_left = 0.072
    x_branch = x_reject + 0.5 * box_width
    y_branch = y0 - 0.098

    _draw_arrow(
        ax=axis,
        start=(x_branch, y0),
        end=(x_branch, y_branch + 0.022),
        color="#7a1f1f",
        linewidth=2.8,
        mutation_scale=26.0,
        zorder=8.5,
    )

    x_line_start = x_branch + 0.005
    x_line_end = 0.975
    _draw_arrow(
        ax=axis,
        start=(x_line_start, y_branch),
        end=(x_line_end, y_branch),
        color="0.35",
        linewidth=2.0,
        mutation_scale=18.0,
        zorder=7.8,
    )
    branch_labels = [
        ("採用", "#2ca02c", x_output + 0.026 - branch_shift_left),
        ("監視", "#f59e0b", x_output + 0.118 - branch_shift_left),
        ("棄却", "#d62728", x_output + 0.212 - branch_shift_left),
    ]
    for label, color, xb in branch_labels:
        axis.scatter([xb], [y_branch], transform=axis.transAxes, s=56, color=color, zorder=9.0)
        branch_text = axis.text(
            xb,
            y_branch - 0.020,
            label,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8.8,
            color=color,
            fontweight="bold",
            zorder=9.0,
        )
        branch_text.set_fontsize(8.2)
        branch_text.set_fontweight("bold")

    branch_label = axis.text(
        x_branch - branch_shift_left,
        y_branch + 0.033,
        "判定分岐",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="0.30",
    )
    branch_label.set_fontsize(8.0)

    footer_text = axis.text(
        0.5,
        0.07,
        "Part I 基準：入力→凍結→統計→判定→出力 を同一I/Fで固定し、再現可能な棄却手順として運用する。",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=9.8,
        color="0.25",
    )
    footer_text.set_fontsize(8.8)

    outputs = _save_figure_bundle(fig=figure, stem="pmodel_rejection_protocol_flowchart")
    plt.close(figure)

    payload = {
        "generated_utc": _utc_now(),
        "script": "scripts/theory/pmodel_rejection_protocol_flowchart.py",
        "outputs": {
            **outputs,
            "metrics_json": str(out_json_private).replace("\\", "/"),
        },
        "flow_steps": [step["title"] for step in steps],
        "notes": [
            "Part I Method 3.0 の共通棄却手順をフローチャート化した図。",
            "式の追加ではなく、運用I/F（再現と棄却）を可視化する監査図。",
        ],
    }
    _write_json(out_json_private, payload)

    print(f"[ok] png(public) : {outputs['png_public']}")
    print(f"[ok] png(private): {outputs['png_private']}")
    print(f"[ok] pdf(public) : {outputs['pdf_public']}")
    print(f"[ok] json        : {out_json_private}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
