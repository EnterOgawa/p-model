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
        fontsize=11.0,
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
        fontsize=9.2,
        color="0.15",
        linespacing=1.15,
    )


# 関数: `_draw_arrow` の入出力契約と処理意図を定義する。
def _draw_arrow(*, ax: Any, start: Tuple[float, float], end: Tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.2,
        color="0.35",
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

    figure, axis = plt.subplots(figsize=(14.0, 4.6), dpi=200)
    axis.set_axis_off()

    steps: List[Dict[str, str]] = [
        {"title": "Input", "body": "一次データ\n依存前提\n取得元", "color": "#f3f7ff"},
        {"title": "Frozen", "body": "凍結パラメータ\n凍結根拠\n固定時点", "color": "#f9f3ff"},
        {"title": "Statistic", "body": "RMS, χ², z\nΔAIC, 傾き\n誤差伝播", "color": "#fff8f3"},
        {"title": "Reject", "body": "閾値判定\n(3σ, ΔAIC)\nPass/Watch/Reject", "color": "#fff3f3"},
        {"title": "Output", "body": "固定ファイル名\n再現コマンド\n監査ログ", "color": "#f3fff3"},
    ]

    box_width = 0.17
    box_height = 0.52
    x0 = 0.03
    y0 = 0.30
    gap = 0.03

    for index, step in enumerate(steps):
        x = x0 + index * (box_width + gap)
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

        # 条件分岐: `index < len(steps) - 1` を満たす経路を評価する。
        if index < len(steps) - 1:
            start = (x + box_width, y0 + box_height * 0.5)
            end = (x + box_width + gap, y0 + box_height * 0.5)
            _draw_arrow(ax=axis, start=start, end=end)

    axis.text(
        0.5,
        0.09,
        "Part I 基準：Input→Frozen→Statistic→Reject→Output を同一I/Fで固定し、再現可能な棄却手順として運用する。",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=9.8,
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
