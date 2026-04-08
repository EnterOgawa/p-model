"""
Build the Part V timeline figure with the shared fixed-template paper baseline.

Inputs:
- Frozen timeline rows defined in this script.

Outputs:
- output/private/summary/figures/part5_future_predictions_timeline.{pdf,png}
- output/private/summary/part5_future_predictions_timeline.pdf
- output/private/summary/part5_future_predictions_timeline_metrics.json
- output/public/summary/part5_future_predictions_timeline.{pdf,png}
- output/public/summary/part5_future_predictions_timeline_metrics.json
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.plot_style import apply_wavep_figure_layout, install_wavep_font_profile  # noqa: E402


_STEM = "part5_future_predictions_timeline"
_PRIVATE_SUMMARY_DIR = ROOT / "output" / "private" / "summary"
_PRIVATE_FIGURES_DIR = _PRIVATE_SUMMARY_DIR / "figures"
_PUBLIC_SUMMARY_DIR = ROOT / "output" / "public" / "summary"


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_safe_rel` の入出力契約と処理意図を定義する。

def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


# 関数: `_timeline_rows` の入出力契約と処理意図を定義する。

def _timeline_rows() -> List[Dict[str, Any]]:
    return [
        {
            "label": "銀河回転曲線\n(独立 sample)",
            "start": 2026.0,
            "end": 2032.0,
            "span": "2026-2032",
            "score": "Pass",
        },
        {
            "label": "量子計算デコヒーレンス\n(platform benchmark)",
            "start": 2026.0,
            "end": 2032.0,
            "span": "2026-2032",
            "score": "Reference",
        },
        {
            "label": "重力波偏光\n(LIGO/Virgo/KAGRA)",
            "start": 2027.0,
            "end": 2030.0,
            "span": "2027-2030",
            "score": "Watch",
        },
        {
            "label": "距離指標前提検証\n(標準サイレン/JWST)",
            "start": 2027.0,
            "end": 2030.0,
            "span": "2027-2030",
            "score": "Watch",
        },
        {
            "label": "弱場重力量子\n(原子干渉計/時計)",
            "start": 2026.0,
            "end": 2035.6,
            "span": "2026-2035+",
            "score": "Watch",
        },
        {
            "label": "宇宙論 ln(1+z)\n(DESI/Euclid/Roman)",
            "start": 2027.0,
            "end": 2035.0,
            "span": "2027-2035",
            "score": "Watch",
        },
        {
            "label": "ブラックホール影\n(ngEHT/BHEX)",
            "start": 2030.0,
            "end": 2031.0,
            "span": "2030-2031",
            "score": "Watch",
        },
        {
            "label": "マクロ量子干渉\n(MAQRO/OTIMA)",
            "start": 2035.0,
            "end": 2037.2,
            "span": "2035+",
            "score": "Reference",
        },
    ]


# 関数: `_output_paths` の入出力契約と処理意図を定義する。

def _output_paths() -> Dict[str, Path]:
    return {
        "private_pdf": _PRIVATE_FIGURES_DIR / f"{_STEM}.pdf",
        "private_png": _PRIVATE_FIGURES_DIR / f"{_STEM}.png",
        "private_metrics": _PRIVATE_SUMMARY_DIR / f"{_STEM}_metrics.json",
        "compat_pdf": _PRIVATE_SUMMARY_DIR / f"{_STEM}.pdf",
        "public_pdf": _PUBLIC_SUMMARY_DIR / f"{_STEM}.pdf",
        "public_png": _PUBLIC_SUMMARY_DIR / f"{_STEM}.png",
        "public_metrics": _PUBLIC_SUMMARY_DIR / f"{_STEM}_metrics.json",
    }


# 関数: `_sync_file` の入出力契約と処理意図を定義する。

def _sync_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# 関数: `_render_timeline` の入出力契約と処理意図を定義する。

def _render_timeline(*, rows: List[Dict[str, Any]], out_pdf: Path, out_png: Path) -> Dict[str, Any]:
    install_wavep_font_profile(profile_name="part5_future_predictions")

    fig, ax = plt.subplots()
    apply_wavep_figure_layout(fig, template="part2_single_panel_tall")
    fig.set_size_inches(6.6929, 7.55, forward=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.89, bottom=0.18)

    score_style = {
        "Pass": {"facecolor": "#DCEFD8", "edgecolor": "#5F8F55"},
        "Watch": {"facecolor": "#F7E7BA", "edgecolor": "#C88B1E"},
        "Reference": {"facecolor": "#DDE8F7", "edgecolor": "#5A82B7"},
    }

    y_step = 0.94
    ys = list(reversed([idx * y_step for idx in range(len(rows))]))

    ax.set_xlim(2023.6, 2038.0)
    ax.set_ylim(-0.40, ys[0] + 0.55)

    for y in ys:
        ax.hlines(y=y, xmin=2026.0, xmax=2037.2, color="#E2E7EF", linewidth=1.0, zorder=1)

    for idx, row in enumerate(rows):
        start = float(row["start"])
        end = float(row["end"])
        y = ys[idx]
        label = str(row["label"])
        span = str(row["span"])
        score = str(row["score"])
        style = score_style[score]
        bar_width = end - start

        ax.barh(
            y,
            bar_width,
            left=start,
            height=0.42,
            color=style["facecolor"],
            edgecolor=style["edgecolor"],
            linewidth=1.15,
            zorder=3,
        )

        ax.text(2025.72, y, label, va="center", ha="right")
        ax.text(end + 0.12, y, span, va="center", ha="left", color="#3A3A3A")

        if span.endswith("+"):
            ax.annotate(
                "",
                xy=(end + 0.28, y),
                xytext=(end - 0.04, y),
                arrowprops={"arrowstyle": "->", "color": style["edgecolor"], "lw": 1.1},
                zorder=4,
            )

    ax.set_title("Part V: 将来観測による決着タイムライン", pad=8.0)
    ax.set_xlabel("年（UTC）")
    ax.set_yticks([])
    ax.set_xticks([2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037])
    ax.tick_params(axis="x")
    ax.grid(axis="x", linestyle="--", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=score_style["Pass"]["facecolor"], edgecolor=score_style["Pass"]["edgecolor"]),
        plt.Rectangle((0, 0), 1, 1, facecolor=score_style["Watch"]["facecolor"], edgecolor=score_style["Watch"]["edgecolor"]),
        plt.Rectangle((0, 0), 1, 1, facecolor=score_style["Reference"]["facecolor"], edgecolor=score_style["Reference"]["edgecolor"]),
    ]
    ax.legend(
        legend_handles,
        ["Pass", "Watch", "Reference"],
        loc="upper center",
        frameon=False,
        ncol=3,
        bbox_to_anchor=(0.5, -0.085),
        columnspacing=1.0,
        handlelength=1.2,
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig.savefig(out_pdf)
        fig.savefig(out_png, dpi=220)

    figure_size = tuple(float(v) for v in fig.get_size_inches())
    plt.close(fig)
    return {
        "template": "part2_single_panel_tall",
        "font_profile": "part5_future_predictions",
        "figure_size_inches": {
            "width": round(figure_size[0], 4),
            "height": round(figure_size[1], 4),
        },
    }


# 関数: `_write_payloads` の入出力契約と処理意図を定義する。

def _write_payloads(*, rows: List[Dict[str, Any]], render_meta: Dict[str, Any], paths: Dict[str, Path]) -> None:
    common_payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "figure_profile": render_meta,
        "rows": rows,
        "notes": [
            "Part V の未来予測タイムライン図（PDFベクター正本）。",
            "数値は Part II / Part III-A / Part III-B / Part IV の frozen value を reader-facing roadmap へ再整理した目安。",
            "Part II baseline の fixed-template route で source-level 再生成した。",
        ],
    }
    private_payload = dict(common_payload)
    private_payload["outputs"] = {
        "private_figure_pdf": _safe_rel(paths["private_pdf"]),
        "private_figure_png": _safe_rel(paths["private_png"]),
        "private_compat_pdf": _safe_rel(paths["compat_pdf"]),
        "public_pdf": _safe_rel(paths["public_pdf"]),
        "public_png": _safe_rel(paths["public_png"]),
        "public_metrics": _safe_rel(paths["public_metrics"]),
    }
    public_payload = dict(common_payload)
    public_payload["canonical_public"] = True
    public_payload["outputs"] = {
        "public_pdf": _safe_rel(paths["public_pdf"]),
        "public_png": _safe_rel(paths["public_png"]),
        "public_metrics": _safe_rel(paths["public_metrics"]),
    }
    paths["private_metrics"].parent.mkdir(parents=True, exist_ok=True)
    paths["public_metrics"].parent.mkdir(parents=True, exist_ok=True)
    paths["private_metrics"].write_text(json.dumps(private_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["public_metrics"].write_text(json.dumps(public_payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    rows = _timeline_rows()
    paths = _output_paths()
    render_meta = _render_timeline(rows=rows, out_pdf=paths["private_pdf"], out_png=paths["private_png"])
    _sync_file(paths["private_pdf"], paths["compat_pdf"])
    _sync_file(paths["private_pdf"], paths["public_pdf"])
    _sync_file(paths["private_png"], paths["public_png"])
    _write_payloads(rows=rows, render_meta=render_meta, paths=paths)

    print(f"[ok] private pdf   : {paths['private_pdf']}")
    print(f"[ok] private png   : {paths['private_png']}")
    print(f"[ok] compat pdf    : {paths['compat_pdf']}")
    print(f"[ok] public pdf    : {paths['public_pdf']}")
    print(f"[ok] public png    : {paths['public_png']}")
    print(f"[ok] metrics json  : {paths['private_metrics']}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
