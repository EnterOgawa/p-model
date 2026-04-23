"""
目的: 理論 topic の pmodel rejection protocol flowchart に対応する公開図・表・監査指標を再生成する。
入力: script 内の既定パラメータと必要な公開データまたは基準値を用いる。
出力: output/public と output/private の canonical artifact を更新する。
前提: 論文本文と README はこの script が出力する公開成果物を正として参照する。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.figure_locale_paths import localize_figure_output_path, resolve_figure_output_locale
from scripts.utils.plot_style import (
    apply_paper_style,
    apply_wavep_figure_layout,
    resolve_wavep_cjk_font_family,
)
OUT_DIR_CANON = ROOT / "output" / "theory"
OUT_DIR_PUBLIC = ROOT / "output" / "public" / "theory"
OUT_DIR_PRIVATE = ROOT / "output" / "private" / "theory"
FIGURE_LOCALE = resolve_figure_output_locale()
IS_EN = FIGURE_LOCALE == "en"


# 関数: `_t` の入出力契約と処理意図を定義する。
def _t(ja: str, en: str) -> str:
    return en if IS_EN else ja


# 関数: `_utc_now` の入出力契約と処理意図を定義する。

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_configure_font` の入出力契約と処理意図を定義する。

def _configure_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        if IS_EN:
            mpl.rcParams["font.family"] = ["DejaVu Sans"]
            mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
            return

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
        "png_canon": OUT_DIR_CANON / f"{stem}.png" if not IS_EN else localize_figure_output_path(OUT_DIR_PRIVATE / f"{stem}.png", root=ROOT, locale=FIGURE_LOCALE),
        "png_public": localize_figure_output_path(OUT_DIR_PUBLIC / f"{stem}.png", root=ROOT, locale=FIGURE_LOCALE),
        "png_private": localize_figure_output_path(OUT_DIR_PRIVATE / f"{stem}.png", root=ROOT, locale=FIGURE_LOCALE),
        "pdf_canon": OUT_DIR_CANON / f"{stem}.pdf" if not IS_EN else localize_figure_output_path(OUT_DIR_PRIVATE / f"{stem}.pdf", root=ROOT, locale=FIGURE_LOCALE),
        "pdf_public": localize_figure_output_path(OUT_DIR_PUBLIC / f"{stem}.pdf", root=ROOT, locale=FIGURE_LOCALE),
        "pdf_private": localize_figure_output_path(OUT_DIR_PRIVATE / f"{stem}.pdf", root=ROOT, locale=FIGURE_LOCALE),
    }
    save_targets = dict(outputs)
    if IS_EN:
        save_targets.pop("png_canon", None)
        save_targets.pop("pdf_canon", None)

    for path in save_targets.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        for path in save_targets.values():
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
    _configure_font()

    out_json_private = localize_figure_output_path(OUT_DIR_PRIVATE / "pmodel_rejection_protocol_flowchart_metrics.json", root=ROOT, locale=FIGURE_LOCALE)

    figure, axis = plt.subplots(dpi=220)
    apply_wavep_figure_layout(figure, template="paper_flowchart")
    figure.subplots_adjust(left=0.035, right=0.985, top=0.965, bottom=0.16)
    axis.set_axis_off()

    steps: List[Dict[str, str]] = [
        {"title": _t("入力", "Inputs"), "body": _t("一次データ\n依存前提\n取得元", "Primary data\nassumed priors\nsource"), "color": "#eef3ff"},
        {"title": _t("凍結", "Freeze"), "body": _t("凍結パラメータ\n凍結根拠\n固定時点", "Frozen parameters\nfreeze rationale\nfreeze point"), "color": "#f2edff"},
        {"title": _t("統計", "Statistics"), "body": _t("RMS, χ², z\nΔAIC, 傾き\n誤差伝播", "RMS, χ², z\nΔAIC, slope\nerror propagation"), "color": "#fff7ec"},
        {"title": _t("判定", "Decision"), "body": _t("閾値判定\n(3σ, ΔAIC)\n判定分岐へ", "Threshold gate\n(3σ, ΔAIC)\nto branch decision"), "color": "#ffe8e8"},
        {"title": _t("出力", "Outputs"), "body": _t("固定ファイル名\n再現コマンド\n監査ログ", "Stable filenames\nrepro command\naudit log"), "color": "#eaf8ea"},
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
        (_t("採用", "Pass"), "#2ca02c", x_output + 0.026 - branch_shift_left),
        (_t("監視", "Watch"), "#f59e0b", x_output + 0.118 - branch_shift_left),
        (_t("棄却", "Reject"), "#d62728", x_output + 0.212 - branch_shift_left),
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
        _t("判定分岐", "Decision branch"),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="0.30",
    )
    branch_label.set_fontsize(8.0)

    footer_text = axis.text(
        0.5,
        0.06,
        _t(
            "Part I 基準：入力→凍結→統計→判定→出力 を同一I/Fで固定し、\n再現可能な棄却手順として運用する。",
            "Part I fixes input → freeze → statistics → decision → output\nas one common I/F and runs it as a reproducible rejection protocol.",
        ),
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=9.8,
        color="0.25",
        linespacing=1.20,
    )
    footer_text.set_fontsize(7.8)

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
            _t("Part I Method 3.0 の共通棄却手順をフローチャート化した図。", "Flowchart of the common rejection protocol in Part I Method 3.0."),
            _t("式の追加ではなく、運用I/F（再現と棄却）を可視化する監査図。", "An audit figure that visualizes the operational I/F for reproducibility and rejection, not an added equation."),
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
