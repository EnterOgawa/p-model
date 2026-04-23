"""
目的: 理論 topic の pmodel core mapping overview に対応する公開図・表・監査指標を再生成する。
入力: script 内の既定パラメータと必要な公開データまたは基準値を用いる。
出力: output/public と output/private の canonical artifact を更新する。
前提: 論文本文と README はこの script が出力する公開成果物を正として参照する。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
OUT_DIR_PRIVATE = ROOT / "output" / "private" / "theory"
OUT_DIR_PUBLIC = ROOT / "output" / "public" / "theory"
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

        available = {f.name for f in fm.fontManager.ttflist}
        fallback = ["Yu Gothic", "Meiryo", "BIZ UDGothic", "MS Gothic", "Yu Mincho", "MS Mincho"]
        chosen = [name for name in fallback if name in available]
        if chosen:
            mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
            mpl.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]

        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        return


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `_save_figure_bundle` の入出力契約と処理意図を定義する。

def _save_figure_bundle(*, fig: Any, stem: str) -> dict[str, str]:
    outputs = {
        "png_canon": OUT_DIR_CANON / f"{stem}.png" if not IS_EN else localize_figure_output_path(OUT_DIR_PRIVATE / f"{stem}.png", root=ROOT, locale=FIGURE_LOCALE),
        "png_private": localize_figure_output_path(OUT_DIR_PRIVATE / f"{stem}.png", root=ROOT, locale=FIGURE_LOCALE),
        "png_public": localize_figure_output_path(OUT_DIR_PUBLIC / f"{stem}.png", root=ROOT, locale=FIGURE_LOCALE),
        "pdf_canon": OUT_DIR_CANON / f"{stem}.pdf" if not IS_EN else localize_figure_output_path(OUT_DIR_PRIVATE / f"{stem}.pdf", root=ROOT, locale=FIGURE_LOCALE),
        "pdf_private": localize_figure_output_path(OUT_DIR_PRIVATE / f"{stem}.pdf", root=ROOT, locale=FIGURE_LOCALE),
        "pdf_public": localize_figure_output_path(OUT_DIR_PUBLIC / f"{stem}.pdf", root=ROOT, locale=FIGURE_LOCALE),
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


# 関数: `_box` の入出力契約と処理意図を定義する。

def _box(
    *,
    ax: Any,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    formula: str,
    desc: str,
    part: str,
    fc: str,
    title_y: float = 0.72,
    formula_y: float = 0.50,
    desc_y: float = 0.25,
    title_fontsize: float = 8.8,
    formula_fontsize: float = 8.2,
    desc_fontsize: float = 7.2,
    part_x: float = 0.02,
    part_y: float = 0.97,
    part_fontsize: float = 6.0,
) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        transform=ax.transAxes,
        facecolor=fc,
        edgecolor="0.25",
        linewidth=1.2,
    )
    ax.add_patch(patch)
    part_text = ax.text(
        x + part_x * w,
        y + part_y * h,
        part,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        color="0.33",
        fontweight="bold",
    )
    part_text.set_fontsize(part_fontsize)
    part_text.set_fontweight("bold")
    title_text = ax.text(
        x + 0.5 * w,
        y + title_y * h,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.8,
        fontweight="bold",
        color="0.10",
        linespacing=1.18,
    )
    title_text.set_fontsize(title_fontsize)
    title_text.set_fontweight("bold")
    formula_text = ax.text(
        x + 0.5 * w,
        y + formula_y * h,
        formula,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.2,
        color="0.10",
        linespacing=1.14,
    )
    formula_text.set_fontsize(formula_fontsize)
    desc_text = ax.text(
        x + 0.5 * w,
        y + desc_y * h,
        desc,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.2,
        color="0.22",
        linespacing=1.15,
    )
    desc_text.set_fontsize(desc_fontsize)


# 関数: `_arrow` の入出力契約と処理意図を定義する。

def _arrow(
    *,
    ax: Any,
    a: tuple[float, float],
    b: tuple[float, float],
    lw: float = 1.9,
    color: str = "0.30",
    linestyle: str = "-",
) -> None:
    patch = FancyArrowPatch(
        a,
        b,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
    )
    ax.add_patch(patch)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    OUT_DIR_CANON.mkdir(parents=True, exist_ok=True)
    OUT_DIR_PRIVATE.mkdir(parents=True, exist_ok=True)
    OUT_DIR_PUBLIC.mkdir(parents=True, exist_ok=True)

    out_json_canon = OUT_DIR_CANON / "pmodel_core_mapping_overview_metrics.json"
    out_json_private = localize_figure_output_path(OUT_DIR_PRIVATE / "pmodel_core_mapping_overview_metrics.json", root=ROOT, locale=FIGURE_LOCALE)
    out_json_public = localize_figure_output_path(OUT_DIR_PUBLIC / "pmodel_core_mapping_overview_metrics.json", root=ROOT, locale=FIGURE_LOCALE)

    apply_paper_style()
    _configure_font()
    fig, ax = plt.subplots(dpi=220)
    apply_wavep_figure_layout(fig, template="paper_diagram")
    fig.set_size_inches(fig.get_size_inches()[0], 4.45, forward=True)
    fig.subplots_adjust(left=0.038, right=0.992, top=0.965, bottom=0.062)
    ax.set_axis_off()

    wh_input = (0.24, 0.20)
    wh_hub = (0.28, 0.24)
    wh_leaf = (0.23, 0.18)

    pos_p = (0.05, 0.67)
    pos_phi = (0.36, 0.60)
    pos_gravity = (0.72, 0.75)
    pos_clock = (0.72, 0.53)
    pos_light = (0.72, 0.31)
    pos_quantum = (0.05, 0.40)
    pos_cosmo = (0.36, 0.21)

    group_labels = [
        (0.845, 0.973, _t("Part I 所管", "Part I scope"), "#2f6b39"),
        (0.140, 0.628, _t("Part III 所管", "Part III scope"), "#9a5c0f"),
        (0.49, 0.165, _t("Part II 所管", "Part II scope"), "#276749"),
    ]
    for gx, gy, text, color in group_labels:
        group_text = ax.text(
            gx,
            gy,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.0,
            color=color,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
        group_text.set_fontsize(7.4)
        group_text.set_fontweight("bold")

    _box(
        ax=ax,
        xy=pos_p,
        wh=wh_input,
        fc="#dfe8f8",
        title=_t("時間波密度（入力）", "Time-wave density\n(input)"),
        formula=r"$P(x)$",
        desc=_t("静止極限の基底量", "Base quantity in\nthe static limit"),
        part="Part I",
        title_y=0.67,
        formula_y=0.40,
        desc_y=0.20,
        title_fontsize=8.2,
        formula_fontsize=7.8,
    )
    _box(
        ax=ax,
        xy=pos_phi,
        wh=wh_hub,
        fc="#e7ddff",
        title=_t("中心ハブ：ポテンシャル写像", "Central hub:\npotential mapping"),
        formula=r"$\phi=-c^2\ln(P/P_{\infty})$",
        desc=_t("重力・時計・光への共通入口", "Common entry to gravity,\nclocks, and light"),
        part="Part I",
    )
    _box(
        ax=ax,
        xy=pos_gravity,
        wh=wh_leaf,
        fc="#e3f1e3",
        title=_t("重力", "Gravity"),
        formula=r"$\mathbf{a}=-\nabla\phi$",
        desc=_t("P勾配へ滑る運動", "Motion sliding down\nthe P gradient"),
        part="Part I",
    )
    _box(
        ax=ax,
        xy=pos_clock,
        wh=wh_leaf,
        fc="#e3f1e3",
        title=_t("時計（束縛モード）", "Clock\n(bound mode)"),
        formula=r"$d\tau/dt=(P_{\infty}/P)(d\tau/dt)_v$",
        desc=_t("重力項と速度項を分離", "Separate gravity and\nvelocity factors"),
        part="Part I",
        title_y=0.66,
        formula_y=0.41,
        desc_y=0.18,
        title_fontsize=8.2,
        formula_fontsize=6.8,
        desc_fontsize=6.6,
    )
    _box(
        ax=ax,
        xy=pos_light,
        wh=wh_leaf,
        fc="#e3f1e3",
        title=_t("光（自由波）", "Light\n(free wave)"),
        formula=r"$n(P)=(P/P_{\infty})^{2\beta}$",
        desc=_t("高P側へ屈折", "Refraction toward\nthe high-P side"),
        part="Part I",
        title_y=0.66,
        formula_y=0.41,
        desc_y=0.18,
        title_fontsize=8.2,
        formula_fontsize=6.9,
        desc_fontsize=6.6,
    )
    _box(
        ax=ax,
        xy=pos_quantum,
        wh=wh_leaf,
        fc="#fdeccf",
        title=_t("量子相関（selection）", "Quantum correlation\n(selection)"),
        formula=_t(r"$P_\mu\leftrightarrow$ 微視結合", r"$P_\mu\leftrightarrow$ microscopic coupling"),
        desc=_t("Bell・核力・V-A（Part III）", "Bell, nuclear force,\nand V-A (Part III)"),
        part="Part III",
        title_y=0.66,
        formula_y=0.35,
        desc_y=0.12,
        title_fontsize=7.8,
        formula_fontsize=6.8,
        desc_fontsize=6.8,
    )
    _box(
        ax=ax,
        xy=pos_cosmo,
        wh=wh_leaf,
        fc="#ddf3e8",
        title=_t("宇宙論背景写像", "Cosmological\nbackground mapping"),
        formula=r"$1+z=P_{\mathrm{em}}/P_{\mathrm{obs}}$",
        desc=_t(
            r"$P_{\mathrm{bg}}(t)$ の時間変化（Part II）",
            "Time evolution of $P_{\\mathrm{bg}}(t)$\n(Part II)",
        ),
        part="Part II",
        title_y=0.66,
        formula_y=0.35,
        desc_y=0.12,
        title_fontsize=7.8,
        formula_fontsize=6.8,
        desc_fontsize=6.8,
    )

    _arrow(
        ax=ax,
        a=(pos_p[0] + wh_input[0], pos_p[1] + 0.52 * wh_input[1]),
        b=(pos_phi[0], pos_phi[1] + 0.65 * wh_hub[1]),
        lw=3.5,
        color="#2f4f6f",
    )
    _arrow(
        ax=ax,
        a=(pos_phi[0] + wh_hub[0], pos_phi[1] + 0.80 * wh_hub[1]),
        b=(pos_gravity[0], pos_gravity[1] + 0.52 * wh_leaf[1]),
        lw=2.4,
        color="#4b5563",
    )
    _arrow(
        ax=ax,
        a=(pos_phi[0] + wh_hub[0], pos_phi[1] + 0.52 * wh_hub[1]),
        b=(pos_clock[0], pos_clock[1] + 0.52 * wh_leaf[1]),
        lw=2.4,
        color="#4b5563",
    )
    _arrow(
        ax=ax,
        a=(pos_phi[0] + wh_hub[0], pos_phi[1] + 0.26 * wh_hub[1]),
        b=(pos_light[0], pos_light[1] + 0.52 * wh_leaf[1]),
        lw=2.4,
        color="#4b5563",
    )
    _arrow(
        ax=ax,
        a=(pos_p[0] + 0.72 * wh_input[0], pos_p[1]),
        b=(pos_quantum[0] + 0.56 * wh_leaf[0], pos_quantum[1] + wh_leaf[1]),
        lw=2.3,
        color="#8b6a2e",
        linestyle="-",
    )
    _arrow(
        ax=ax,
        a=(pos_phi[0] + 0.42 * wh_hub[0], pos_phi[1]),
        b=(pos_cosmo[0] + 0.56 * wh_leaf[0], pos_cosmo[1] + wh_leaf[1]),
        lw=2.3,
        color="#2f7d5b",
        linestyle="-",
    )

    footer_text = ax.text(
        0.5,
        0.044,
        _t(
            r"Part I は写像と $\beta_{\mathrm{frozen}}$ を固定し、Part II/III は固定値のまま反証監査を行う。",
            r"Part I fixes the mapping and $\beta_{\mathrm{frozen}}$; Parts II/III run falsification audits with those frozen values.",
        ),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.8,
        color="0.25",
    )
    footer_text.set_fontsize(7.6)

    outputs = _save_figure_bundle(fig=fig, stem="pmodel_core_mapping_overview")
    plt.close(fig)

    payload = {
        "generated_utc": _utc_now(),
        "script": "scripts/theory/pmodel_core_mapping_overview.py",
        "outputs": {
            **outputs,
            "metrics_json_canon": str(out_json_canon),
            "metrics_json_private": str(out_json_private),
            "metrics_json_public": str(out_json_public),
        },
        "figure_index_path": "output/theory/pmodel_core_mapping_overview.png",
        "notes": [
            "This is a conceptual diagram (not a numerical result).",
            "Text uses the Part I vocabulary: P-field -> φ -> (gravity, clocks, light), plus pointers to Part II/III.",
            "Velocity saturation δ is treated as an extension (not used in the Part I core).",
        ],
    }
    if not IS_EN:
        _write_json(out_json_canon, payload)

    _write_json(out_json_private, payload)
    _write_json(out_json_public, payload)

    print(f"[ok] png(canon)  : {outputs['png_canon']}")
    print(f"[ok] png(private): {outputs['png_private']}")
    print(f"[ok] png(public) : {outputs['png_public']}")
    print(f"[ok] pdf(canon)  : {outputs['pdf_canon']}")
    print(f"[ok] json(canon) : {out_json_canon}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
