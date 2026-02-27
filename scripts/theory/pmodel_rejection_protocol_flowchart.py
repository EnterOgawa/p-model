from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR_PUBLIC = ROOT / "output" / "theory"
OUT_DIR_PRIVATE = ROOT / "output" / "private" / "theory"


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        available = {font.name for font in fm.fontManager.ttflist}
        selected = [name for name in preferred if name in available]
        # 条件分岐: `not selected` を満たす経路を評価する。
        if not selected:
            return

        mpl.rcParams["font.family"] = selected + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        return


# 関数: `_write_json` の入出力契約と処理意図を定義する。
def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    ax.text(
        x + w * 0.5,
        y + h * 0.67,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=13.0,
        fontweight="bold",
        color="0.10",
    )
    ax.text(
        x + w * 0.5,
        y + h * 0.33,
        body,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11.3,
        color="0.15",
        linespacing=1.22,
    )


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


# 関数: `_save_figure` の入出力契約と処理意図を定義する。
def _save_figure(fig: Any, public_path: Path, private_path: Path) -> None:
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(public_path, dpi=220, bbox_inches="tight")
    fig.savefig(private_path, dpi=220, bbox_inches="tight")


# 関数: `main` の入出力契約と処理意図を定義する。
def main() -> int:
    _set_japanese_font()

    out_png_public = OUT_DIR_PUBLIC / "pmodel_rejection_protocol_flowchart.png"
    out_png_private = OUT_DIR_PRIVATE / "pmodel_rejection_protocol_flowchart.png"
    out_json_private = OUT_DIR_PRIVATE / "pmodel_rejection_protocol_flowchart_metrics.json"

    figure, axis = plt.subplots(figsize=(9.8, 5.2), dpi=220)
    axis.set_axis_off()

    steps: List[Dict[str, str]] = [
        {"title": "Input", "body": "一次データ\n依存前提\n取得元", "color": "#eef3ff"},
        {"title": "Frozen", "body": "凍結パラメータ\n凍結根拠\n固定時点", "color": "#f2edff"},
        {"title": "Statistic", "body": "RMS, χ², z\nΔAIC, 傾き\n誤差伝播", "color": "#fff7ec"},
        {"title": "Reject", "body": "閾値判定\n(3σ, ΔAIC)\n判定分岐へ", "color": "#ffe8e8"},
        {"title": "Output", "body": "固定ファイル名\n再現コマンド\n監査ログ", "color": "#eaf8ea"},
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
    x_line_end = 0.95
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
        ("Pass", "#2ca02c", x_output + 0.04 - branch_shift_left),
        ("Watch", "#f59e0b", x_output + 0.105 - branch_shift_left),
        ("Reject", "#d62728", x_output + 0.165 - branch_shift_left),
    ]
    for label, color, xb in branch_labels:
        axis.scatter([xb], [y_branch], transform=axis.transAxes, s=56, color=color, zorder=9.0)
        axis.text(
            xb + 0.008,
            y_branch + 0.0025,
            label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.6,
            color=color,
            fontweight="bold",
            zorder=9.0,
        )

    axis.text(
        x_branch - branch_shift_left,
        y_branch + 0.033,
        "判定分岐",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.3,
        color="0.30",
    )

    axis.text(
        0.5,
        0.07,
        "Part I 基準：Input→Frozen→Statistic→Reject→Output を同一I/Fで固定し、再現可能な棄却手順として運用する。",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=11.0,
        color="0.25",
    )

    figure.tight_layout()
    _save_figure(figure, public_path=out_png_public, private_path=out_png_private)
    plt.close(figure)

    payload = {
        "generated_utc": _utc_now(),
        "script": "scripts/theory/pmodel_rejection_protocol_flowchart.py",
        "outputs": {
            "png_public": str(out_png_public).replace("\\", "/"),
            "png_private": str(out_png_private).replace("\\", "/"),
            "metrics_json": str(out_json_private).replace("\\", "/"),
        },
        "flow_steps": [step["title"] for step in steps],
        "notes": [
            "Part I Method 3.0 の共通棄却手順をフローチャート化した図。",
            "式の追加ではなく、運用I/F（再現と棄却）を可視化する監査図。",
        ],
    }
    _write_json(out_json_private, payload)

    print(f"[ok] png(public) : {out_png_public}")
    print(f"[ok] png(private): {out_png_private}")
    print(f"[ok] json        : {out_json_private}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。
if __name__ == "__main__":
    raise SystemExit(main())
