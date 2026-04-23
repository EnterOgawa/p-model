"""
Build the reader-facing Part V conceptual figures.

Inputs:
- Frozen conceptual layouts and explanatory labels defined in this script.

Outputs:
- output/private/summary/figures/part5_conceptual_*.pdf
- output/private/summary/figures/part5_conceptual_*.png
- output/public/summary/part5_conceptual_*.pdf
- output/public/summary/part5_conceptual_*.png
- output/private/summary/part5_conceptual_figures_metrics.json
- output/public/summary/part5_conceptual_figures_metrics.json

Notes:
- `ja` keeps the canonical paths above.
- Non-`ja` locales are written under `.../locales/<locale>/...` so the Japanese
  canonical figures remain available as the comparison baseline.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.figure_locale_paths import localize_figure_output_path, resolve_figure_output_locale  # noqa: E402
from scripts.utils.plot_style import apply_wavep_figure_layout, get_wavep_font_size, install_wavep_font_profile  # noqa: E402


PRIVATE_SUMMARY_DIR = ROOT / "output" / "private" / "summary"
PRIVATE_FIGURES_DIR = PRIVATE_SUMMARY_DIR / "figures"
PUBLIC_SUMMARY_DIR = ROOT / "output" / "public" / "summary"
METRICS_STEM = "part5_conceptual_figures_metrics.json"

PASS_COLOR = "#7CB342"
WATCH_COLOR = "#DDAA33"
REFERENCE_COLOR = "#9E9E9E"
PMODEL_COLOR = "#D95F02"
GR_COLOR = "#4C72B0"
OBS_COLOR = "#2F2F2F"
LIGHT_BG = "#F7F7F7"
FIGURE_LOCALE = resolve_figure_output_locale()


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_t` の入出力契約と処理意図を定義する。

def _t(ja: str, en: str) -> str:
    return en if FIGURE_LOCALE == "en" else ja


# 関数: `_safe_rel` の入出力契約と処理意図を定義する。

def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


# 関数: `_sync_file` の入出力契約と処理意図を定義する。

def _sync_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# 関数: `_output_pair` の入出力契約と処理意図を定義する。

def _output_pair(stem: str) -> dict[str, Path]:
    return {
        "private_pdf": localize_figure_output_path(PRIVATE_FIGURES_DIR / f"{stem}.pdf", root=ROOT),
        "private_png": localize_figure_output_path(PRIVATE_FIGURES_DIR / f"{stem}.png", root=ROOT),
        "public_pdf": localize_figure_output_path(PUBLIC_SUMMARY_DIR / f"{stem}.pdf", root=ROOT),
        "public_png": localize_figure_output_path(PUBLIC_SUMMARY_DIR / f"{stem}.png", root=ROOT),
    }


# 関数: `_base_figure` の入出力契約と処理意図を定義する。

def _base_figure(*, title: str, template: str = "paper_diagram", height_in: float = 4.6) -> tuple[Any, Any, dict[str, float]]:
    fig, ax = plt.subplots()
    apply_wavep_figure_layout(fig, template=template)
    width = fig.get_size_inches()[0]
    fig.set_size_inches(width, height_in, forward=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=get_wavep_font_size("suptitle", name="part5_future_predictions"), y=0.97)
    sizes = {
        "title": get_wavep_font_size("title", name="part5_future_predictions"),
        "axis": get_wavep_font_size("axis", name="part5_future_predictions"),
        "tick": get_wavep_font_size("tick", name="part5_future_predictions"),
        "legend": get_wavep_font_size("legend", name="part5_future_predictions"),
        "note": get_wavep_font_size("note", name="part5_future_predictions"),
    }
    return fig, ax, sizes


# 関数: `_finalize_figure` の入出力契約と処理意図を定義する。

def _finalize_figure(fig: Any, *, paths: dict[str, Path], pad_inches: float = 0.03) -> dict[str, Any]:
    paths["private_pdf"].parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": "tight", "savefig.pad_inches": pad_inches}):
        fig.savefig(paths["private_pdf"])
        fig.savefig(paths["private_png"], dpi=220)

    _sync_file(paths["private_pdf"], paths["public_pdf"])
    _sync_file(paths["private_png"], paths["public_png"])

    size = tuple(float(v) for v in fig.get_size_inches())
    plt.close(fig)
    return {"width_in": round(size[0], 4), "height_in": round(size[1], 4)}


# 関数: `_draw_eht` の入出力契約と処理意図を定義する。

def _draw_eht(paths: dict[str, Path]) -> dict[str, Any]:
    fig, ax, sizes = _base_figure(title=_t("ブラックホール影の差分", "Black-hole shadow differential"), height_in=4.45)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(Circle((0.26, 0.64), 0.12, fill=False, ec=GR_COLOR, lw=2.3))
    ax.add_patch(Circle((0.62, 0.64), 0.126, fill=False, ec=PMODEL_COLOR, lw=2.3))
    ax.text(0.26, 0.80, _t("一般相対論", "General\nrelativity"), ha="center", va="center", fontsize=sizes["title"])
    ax.text(0.26, 0.64, _t("小さい影", "Smaller\nshadow"), ha="center", va="center", fontsize=sizes["note"])
    ax.text(0.62, 0.80, "P-model", ha="center", va="center", fontsize=sizes["title"])
    ax.text(0.62, 0.64, _t("約5%大きい影", "~5% larger\nshadow"), ha="center", va="center", fontsize=sizes["note"])

    ax.add_patch(FancyArrowPatch((0.39, 0.64), (0.49, 0.64), arrowstyle="<->", mutation_scale=14, lw=1.5, color=OBS_COLOR))
    ax.text(0.44, 0.69, _t("4.8084%差", "4.8084% gap"), ha="center", va="bottom", fontsize=sizes["note"])

    ax.text(0.10, 0.33, _t("現在の誤差", "Current uncertainty"), ha="left", va="center", fontsize=sizes["axis"])
    ax.add_patch(Rectangle((0.28, 0.305), 0.28, 0.05, fc=WATCH_COLOR, ec=WATCH_COLOR, alpha=0.6))
    ax.text(0.58, 0.33, _t("差より大きい", "Still larger than\nthe gap"), ha="left", va="center", fontsize=sizes["note"])

    ax.text(0.10, 0.22, _t("ngEHT の誤差", "ngEHT uncertainty"), ha="left", va="center", fontsize=sizes["axis"])
    ax.add_patch(Rectangle((0.28, 0.195), 0.09, 0.05, fc=PASS_COLOR, ec=PASS_COLOR, alpha=0.7))
    ax.text(0.39, 0.22, _t("差を分離できる見込み", "Expected to\nresolve the gap"), ha="left", va="center", fontsize=sizes["note"])

    ax.text(
        0.50,
        0.08,
        _t("影の直径差を直接測る観測が決定打になる。", "A direct shadow-diameter measurement is the decisive test."),
        ha="center",
        va="center",
        fontsize=sizes["note"],
    )
    return _finalize_figure(fig, paths=paths)


# 関数: `_draw_gw` の入出力契約と処理意図を定義する。

def _draw_gw(paths: dict[str, Path]) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 2)
    apply_wavep_figure_layout(fig, template="paper_side_by_side")
    fig.set_size_inches(fig.get_size_inches()[0], 4.55, forward=True)
    fig.subplots_adjust(top=0.79, bottom=0.14, left=0.07, right=0.94, wspace=0.12)
    sizes = {
        "title": get_wavep_font_size("title", name="part5_future_predictions"),
        "axis": get_wavep_font_size("axis", name="part5_future_predictions"),
        "note": get_wavep_font_size("note", name="part5_future_predictions"),
        "legend": get_wavep_font_size("legend", name="part5_future_predictions"),
    }
    fig.suptitle(
        _t("重力波偏光の判別条件", "Gravitational-wave polarization gate"),
        fontsize=get_wavep_font_size("suptitle", name="part5_future_predictions"),
        x=0.515,
        y=0.985,
        ha="center",
    )

    panels = [
        (
            axes[0],
            _t("検出器2台（現在）", "2 detectors\n(today)"),
            [("H1", 0.22, 0.72), ("L1", 0.74, 0.28)],
            _t("方向が足りず\n偏光を分けにくい", "Too few directions\nto separate modes"),
        ),
        (
            axes[1],
            _t("検出器3台（将来）", "3 detectors\n(future)"),
            [("H1", 0.20, 0.72), ("L1", 0.76, 0.30), ("V1", 0.48, 0.82)],
            _t("3方向の情報で\n偏光を判別できる", "Three directions allow\nmode separation"),
        ),
    ]
    for ax, title, nodes, note in panels:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        title_text = ax.set_title(title, fontsize=sizes["title"], pad=2.0, y=0.965)
        title_text.set_linespacing(1.18)
        # 先に接続線をまとめて描き、後から検出器ノードを重ねて丸が手前に見えるようにする。
        for idx, (_, x, y) in enumerate(nodes):
            for _, ox, oy in nodes[idx + 1 :]:
                ax.plot([x, ox], [y, oy], color="#B5BCC8", lw=1.4, zorder=1)

        for label, x, y in nodes:
            ax.add_patch(Circle((x, y), 0.055, fc="#E8EEF8", ec=GR_COLOR, lw=1.7, zorder=3))
            ax.text(x, y, label, ha="center", va="center", fontsize=sizes["axis"], zorder=4)

        ax.text(0.50, 0.08, note, ha="center", va="center", fontsize=sizes["note"])

    return _finalize_figure(fig, paths=paths)


# 関数: `_draw_cosmo` の入出力契約と処理意図を定義する。

def _draw_cosmo(paths: dict[str, Path]) -> dict[str, Any]:
    fig, ax, sizes = _base_figure(
        title=_t("宇宙論の独自項", r"The $\ln(1+z)$ cosmology term"),
        template="paper_single_panel",
        height_in=4.65,
    )
    z = np.linspace(0.0, 3.2, 240)
    lcdm = 0.22 + 0.52 * np.log1p(z)
    pmodel = lcdm + 0.045 * np.clip(z - 1.2, 0.0, None) ** 1.3
    ax.plot(z, lcdm, color=GR_COLOR, lw=2.2, label=_t("標準理論", "Standard model"))
    ax.plot(z, pmodel, color=PMODEL_COLOR, lw=2.2, label="P-model")
    ax.fill_between(z, lcdm, pmodel, where=z >= 1.6, color=WATCH_COLOR, alpha=0.25)
    ax.axvspan(1.6, 3.2, color="#F4F1E1", alpha=0.45)
    ax.text(2.35, 1.05, _t("高赤方偏移で差が開く", "The gap opens at\nhigh redshift"), fontsize=sizes["note"], ha="center")
    ax.set_xlabel(_t("赤方偏移 z", "Redshift z"), fontsize=sizes["axis"])
    ax.set_ylabel(_t("距離指標の模式量", "Schematic distance indicator"), fontsize=sizes["axis"])
    ax.legend(loc="upper left", fontsize=sizes["legend"], frameon=False)
    ax.grid(alpha=0.22, linestyle="--")
    return _finalize_figure(fig, paths=paths)


# 関数: `_draw_sparc` の入出力契約と処理意図を定義する。

def _draw_sparc(paths: dict[str, Path]) -> dict[str, Any]:
    fig, ax, sizes = _base_figure(
        title=_t("銀河回転曲線の差分", "Galaxy rotation-curve differential"),
        template="paper_single_panel",
        height_in=4.65,
    )
    r = np.linspace(0.0, 14.0, 240)
    baryon = 150.0 * (1.0 - np.exp(-r / 2.0)) * np.exp(-r / 8.0)
    observed = 185.0 * (1.0 - np.exp(-r / 2.2))
    pmodel = observed - 5.0 * np.exp(-((r - 8.0) ** 2) / 24.0)
    ax.plot(r, observed, color=OBS_COLOR, lw=2.2, label=_t("観測", "Observed"))
    ax.plot(r, baryon, color=GR_COLOR, lw=2.0, ls="--", label=_t("見える物質だけ", "Baryons only"))
    ax.plot(r, pmodel, color=PASS_COLOR, lw=2.2, label="P-model")
    ax.text(10.2, 135.0, _t("差の説明を\nダークマター以外に置く", "Explain the gap\nwithout dark matter"), fontsize=sizes["note"], ha="left")
    ax.set_xlabel(_t("銀河中心からの距離", "Distance from galactic center"), fontsize=sizes["axis"])
    ax.set_ylabel(_t("回転速度", "Rotation speed"), fontsize=sizes["axis"])
    ax.legend(loc="lower right", fontsize=sizes["legend"], frameon=False)
    ax.grid(alpha=0.22, linestyle="--")
    return _finalize_figure(fig, paths=paths)


# 関数: `_draw_macro` の入出力契約と処理意図を定義する。

def _draw_macro(paths: dict[str, Path]) -> dict[str, Any]:
    fig, axes = plt.subplots(2, 1)
    apply_wavep_figure_layout(fig, template="paper_two_panel")
    fig.set_size_inches(fig.get_size_inches()[0], 6.05, forward=True)
    fig.suptitle(
        _t("マクロ量子干渉の破れ", "Macroscopic interference breakdown"),
        fontsize=get_wavep_font_size("suptitle", name="part5_future_predictions"),
        y=0.985,
    )
    sizes = {
        "title": get_wavep_font_size("title", name="part5_future_predictions"),
        "axis": get_wavep_font_size("axis", name="part5_future_predictions"),
        "tick": get_wavep_font_size("tick", name="part5_future_predictions"),
        "note": get_wavep_font_size("note", name="part5_future_predictions"),
    }
    fig.subplots_adjust(top=0.90, hspace=0.34)

    top = axes[0]
    top.set_xlim(0, 1)
    top.set_ylim(0, 1)
    top.axis("off")
    top.text(
        0.50,
        0.93,
        _t("同じ干渉縞が質量とともに崩れる", "The same fringes fade as mass grows"),
        ha="center",
        va="top",
        fontsize=sizes["title"],
        transform=top.transAxes,
    )
    top.add_patch(
        FancyBboxPatch(
            (0.07, 0.18),
            0.86,
            0.55,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            fc="#FBFBFB",
            ec="#D8D8D8",
            lw=0.8,
        )
    )
    stripe_positions = np.linspace(0.10, 0.88, 24)
    for xpos in stripe_positions:
        progress = (xpos - stripe_positions[0]) / (stripe_positions[-1] - stripe_positions[0])
        contrast = max(0.08, 1.0 - 0.92 * progress)
        height = 0.46 - 0.18 * progress
        y0 = 0.225 + 0.09 * progress
        width = 0.013 + 0.004 * progress
        top.add_patch(Rectangle((xpos, y0), width, height, fc=GR_COLOR, ec="none", alpha=0.90 * contrast))

    top.add_patch(FancyArrowPatch((0.18, 0.76), (0.82, 0.76), arrowstyle="-|>", mutation_scale=14, lw=1.2, color=OBS_COLOR))
    top.text(
        0.68,
        0.79,
        _t("質量が増える", "Higher mass"),
        ha="left",
        va="bottom",
        fontsize=sizes["note"],
        bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.92},
    )
    top.text(0.18, 0.10, _t("質量が小さい：くっきり見える", "Low mass:\nclear fringes"), ha="center", fontsize=sizes["note"])
    top.text(0.80, 0.10, _t("質量が大きい：ぼやけて消える", "High mass:\nfringes wash out"), ha="center", fontsize=sizes["note"])

    bottom = axes[1]
    mass = np.linspace(0.0, 3.0, 280)
    visibility = 1.0 / (1.0 + np.exp((mass - 1.95) / 0.24))
    bottom.plot(mass, visibility, color=PMODEL_COLOR, lw=2.2)
    bottom.axvspan(1.55, 2.20, color=WATCH_COLOR, alpha=0.23)
    bottom.annotate(
        _t("P-model の予測：\nここから壊れ始める\n", "P-model prediction:\nbreakdown starts here\n") + r"($\chi \approx 0.1$–$1$)",
        xy=(1.88, float(np.interp(1.88, mass, visibility))),
        xytext=(2.28, 0.58),
        textcoords="data",
        fontsize=sizes["note"],
        ha="center",
        va="center",
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": OBS_COLOR},
    )
    bottom.set_xlabel(_t("質量", "Mass"), fontsize=sizes["axis"])
    bottom.set_ylabel(_t("干渉縞の見えやすさ", "Interference visibility"), fontsize=sizes["axis"])
    bottom.set_xticks([0.30, 1.45, 2.70])
    bottom.set_xticklabels(
        [_t("原子", "Atoms"), _t("大きな分子", "Large\nmolecules"), _t("巨大分子\n(MAQRO級)", "Huge molecules\n(MAQRO scale)")],
        fontsize=sizes["tick"],
    )
    bottom.grid(alpha=0.22, linestyle="--")
    bottom.set_ylim(-0.02, 1.05)
    return _finalize_figure(fig, paths=paths)


# 関数: `_draw_qc` の入出力契約と処理意図を定義する。

def _draw_qc(paths: dict[str, Path]) -> dict[str, Any]:
    fig, ax, sizes = _base_figure(title=_t("量子計算のノイズ下限", "Quantum-computing noise floor"), template="paper_single_panel", height_in=4.65)
    years = np.array([2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032], dtype=float)
    error = np.array([8.0, 6.5, 5.4, 4.2, 3.4, 2.7, 2.0, 1.55, 1.25], dtype=float)
    floor = 1.0
    ax.plot(years, error, color=GR_COLOR, lw=1.8, marker="o", label=_t("公開ベンチマークの改善", "Public benchmark improvement"))
    ax.axhline(floor, color=PMODEL_COLOR, lw=2.0, ls="--", label=_t("P-model の下限ライン", "P-model lower bound"))
    ax.fill_between(years, floor, error, where=error >= floor, color="#E7EEF8", alpha=0.35)
    ax.text(2029.6, 1.14, _t("ここに届くかが焦点", "The question is\nwhether benchmarks reach it"), fontsize=sizes["note"], ha="left", color=PMODEL_COLOR)
    ax.set_xlabel(_t("年", "Year"), fontsize=sizes["axis"])
    ax.set_ylabel(_t("ゲート誤り率の模式量", "Schematic gate-error rate"), fontsize=sizes["axis"])
    ax.legend(loc="upper right", fontsize=sizes["legend"], frameon=False)
    ax.grid(alpha=0.22, linestyle="--")
    return _finalize_figure(fig, paths=paths)


# 関数: `_draw_gq` の入出力契約と処理意図を定義する。

def _draw_gq(paths: dict[str, Path]) -> dict[str, Any]:
    fig, ax, sizes = _base_figure(title=_t("弱場重力量子の差分", "Weak-field gravity-quantum gap"), template="paper_single_panel", height_in=4.65)
    x = np.linspace(0.0, 1.0, 200)
    baseline = 0.35 + 0.25 * x
    pmodel = baseline + 0.035
    ax.plot(x, baseline, color=GR_COLOR, lw=2.0, label=_t("標準理論", "Standard model"))
    ax.plot(x, pmodel, color=PMODEL_COLOR, lw=2.0, ls="--", label="P-model")
    ax.fill_between(x, baseline - 0.10, baseline + 0.10, color=REFERENCE_COLOR, alpha=0.18, label=_t("現在の精度", "Current precision"))
    ax.fill_between(x, baseline - 0.025, baseline + 0.025, color=PASS_COLOR, alpha=0.20, label=_t("将来の精度", "Future precision"))
    arrow_x = 0.82
    arrow_y_low = float(np.interp(arrow_x, x, baseline))
    arrow_y_high = float(np.interp(arrow_x, x, pmodel))
    arrow_y_mid = 0.5 * (arrow_y_low + arrow_y_high)
    ax.annotate(
        "",
        xy=(arrow_x, arrow_y_high),
        xytext=(arrow_x, arrow_y_low),
        arrowprops={"arrowstyle": "<->", "lw": 1.3, "color": OBS_COLOR},
    )
    ax.text(
        arrow_x - 0.02,
        arrow_y_mid,
        _t("固有のズレ", "Intrinsic gap"),
        fontsize=sizes["note"],
        va="center",
        ha="right",
        bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.92},
    )
    ax.set_xlabel(_t("観測配置の模式座標", "Schematic observing configuration"), fontsize=sizes["axis"])
    ax.set_ylabel(_t("位相・周波数差の模式量", "Schematic phase / frequency gap"), fontsize=sizes["axis"])
    ax.legend(loc="upper left", fontsize=sizes["legend"], frameon=False)
    ax.grid(alpha=0.22, linestyle="--")
    return _finalize_figure(fig, paths=paths)


# 関数: `_draw_ddr` の入出力契約と処理意図を定義する。

def _draw_ddr(paths: dict[str, Path]) -> dict[str, Any]:
    fig, ax, sizes = _base_figure(
        title=_t("距離指標の前提を切り分ける", "Premise-sensitive distance indicators"),
        template="paper_flowchart",
        height_in=5.30,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    current_x = 0.18
    current_w = 0.22
    current_y = 0.22
    block_h = 0.11
    current_layers = [
        (_t("標準光源", "Standard\ncandle"), "#E9B879"),
        (_t("標準定規", "Standard\nruler"), "#E3A85C"),
        (_t("校正進化", "Calibration\ndrift"), "#D68F3A"),
        ("opacity", "#BC7425"),
    ]
    ax.text(current_x + current_w / 2.0, 0.865, _t("現在", "Today"), ha="center", fontsize=sizes["title"])
    current_subtitle = ax.text(
        current_x + current_w / 2.0,
        0.775,
        _t("前提が積み上がる測定", "Measurement with\nstacked assumptions"),
        ha="center",
        fontsize=sizes["note"],
    )
    current_subtitle.set_linespacing(1.16)
    for idx, (label, color) in enumerate(current_layers):
        y = current_y + idx * block_h
        rect = FancyBboxPatch(
            (current_x, y),
            current_w,
            block_h - 0.01,
            boxstyle="round,pad=0.01,rounding_size=0.008",
            fc=color,
            ec="#8C5A1D",
            lw=1.2,
        )
        ax.add_patch(rect)
        ax.text(current_x + current_w / 2.0, y + 0.5 * (block_h - 0.01), label, ha="center", va="center", fontsize=sizes["axis"])

    current_layers_note = ax.text(
        current_x + current_w / 2.0,
        0.156,
        _t("前提が4段積み", "Four layers of\nassumptions"),
        ha="center",
        va="center",
        fontsize=sizes["note"],
    )
    current_layers_note.set_linespacing(1.14)

    future_x = 0.66
    future_w = 0.18
    future_y = 0.22
    future_h = 0.20
    future_rect = FancyBboxPatch(
        (future_x, future_y),
        future_w,
        future_h,
        boxstyle="round,pad=0.01,rounding_size=0.010",
        fc="#8EC07C",
        ec="#4F7D3A",
        lw=1.3,
    )
    ax.add_patch(future_rect)
    ax.text(future_x + future_w / 2.0, 0.865, _t("将来", "Future"), ha="center", fontsize=sizes["title"])
    future_subtitle = ax.text(
        future_x + future_w / 2.0,
        0.775,
        _t("前提を減らした測定", "Lower-premise\nmeasurement"),
        ha="center",
        fontsize=sizes["note"],
    )
    future_subtitle.set_linespacing(1.16)
    ax.text(
        future_x + future_w / 2.0,
        future_y + future_h / 2.0,
        _t("重力波\n（幾何距離）", "Gravitational waves\n(geometric distance)"),
        ha="center",
        va="center",
        fontsize=sizes["axis"],
    )
    future_layers_note = ax.text(
        future_x + future_w / 2.0,
        0.156,
        _t("前提が1段", "One layer only"),
        ha="center",
        va="center",
        fontsize=sizes["note"],
    )
    future_layers_note.set_linespacing(1.14)

    ax.add_patch(
        FancyArrowPatch(
            (0.47, 0.47),
            (0.60, 0.47),
            arrowstyle="simple",
            mutation_scale=24,
            fc=WATCH_COLOR,
            ec=WATCH_COLOR,
            alpha=0.75,
        )
    )
    ax.text(0.535, 0.55, _t("段を減らして再測定", "Re-measure with\nfewer premises"), ha="center", fontsize=sizes["note"])

    question_box = FancyBboxPatch(
        (0.13, 0.000),
        0.30,
        0.092,
        boxstyle="round,pad=0.015,rounding_size=0.015",
        fc="#FFF6E6",
        ec="#C8A25C",
        lw=1.0,
        clip_on=False,
    )
    ax.add_patch(question_box)
    ax.text(
        0.29,
        0.046,
        _t("どの段がズレの原因か\n分かりにくい", "It is hard to see which premise\ncauses the mismatch"),
        ha="center",
        va="center",
        fontsize=sizes["note"],
    )

    answer_box = FancyBboxPatch(
        (0.59, 0.000),
        0.30,
        0.092,
        boxstyle="round,pad=0.015,rounding_size=0.015",
        fc="#EEF7E9",
        ec="#78A35E",
        lw=1.0,
        clip_on=False,
    )
    ax.add_patch(answer_box)
    ax.text(
        0.75,
        0.046,
        _t("段を減らせば原因を\n特定しやすい", "Fewer premises make the source\neasier to isolate"),
        ha="center",
        va="center",
        fontsize=sizes["note"],
    )
    return _finalize_figure(fig, paths=paths, pad_inches=0.11)


# 関数: `_render_specs` の入出力契約と処理意図を定義する。

def _render_specs() -> list[tuple[str, Callable[[dict[str, Path]], dict[str, Any]]]]:
    return [
        ("part5_conceptual_eht", _draw_eht),
        ("part5_conceptual_gw", _draw_gw),
        ("part5_conceptual_cosmo", _draw_cosmo),
        ("part5_conceptual_sparc", _draw_sparc),
        ("part5_conceptual_macro", _draw_macro),
        ("part5_conceptual_qc", _draw_qc),
        ("part5_conceptual_gq", _draw_gq),
        ("part5_conceptual_ddr", _draw_ddr),
    ]


# 関数: `_write_metrics` の入出力契約と処理意図を定義する。

def _write_metrics(rows: list[dict[str, Any]]) -> None:
    payload = {
        "generated_utc": _utc_now(),
        "profile": "part5_future_predictions",
        "outputs": rows,
        "notes": [
            _t("Part V の一般読者向け概念図。", "Reader-facing conceptual figures for Part V."),
            _t("全図は Matplotlib ベクター PDF として生成する。", "All figures are generated as Matplotlib vector PDFs."),
            _t(
                "日本語図は output/public/summary を canonical とし、非 ja は locales/<locale> 配下へ保存する。",
                "Japanese figures remain canonical under output/public/summary, while non-ja locales are saved under locales/<locale>/.",
            ),
        ],
    }
    private_metrics = localize_figure_output_path(PRIVATE_SUMMARY_DIR / METRICS_STEM, root=ROOT)
    public_metrics = localize_figure_output_path(PUBLIC_SUMMARY_DIR / METRICS_STEM, root=ROOT)
    private_metrics.parent.mkdir(parents=True, exist_ok=True)
    private_metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _sync_file(private_metrics, public_metrics)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    install_wavep_font_profile(profile_name="part5_future_predictions")
    PRIVATE_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for stem, renderer in _render_specs():
        paths = _output_pair(stem)
        meta = renderer(paths)
        rows.append(
            {
                "stem": stem,
                "private_pdf": _safe_rel(paths["private_pdf"]),
                "private_png": _safe_rel(paths["private_png"]),
                "public_pdf": _safe_rel(paths["public_pdf"]),
                "public_png": _safe_rel(paths["public_png"]),
                "figure_size_inches": meta,
            }
        )

    _write_metrics(rows)
    print(f"[ok] wrote {len(rows)} Part V conceptual figures")
    return 0


# 条件分岐: `__name__ == \"__main__\"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
