from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR_CANON = ROOT / "output" / "theory"
OUT_DIR_PRIVATE = ROOT / "output" / "private" / "theory"
OUT_DIR_PUBLIC = ROOT / "output" / "public" / "theory"


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
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        return


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


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
    ax.text(
        x + 0.03 * w,
        y + 0.93 * h,
        part,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        color="0.33",
        fontweight="bold",
    )
    ax.text(
        x + 0.5 * w,
        y + 0.72 * h,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=13.2,
        fontweight="bold",
        color="0.10",
        linespacing=1.18,
    )
    ax.text(
        x + 0.5 * w,
        y + 0.50 * h,
        formula,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12.6,
        color="0.10",
        linespacing=1.14,
    )
    ax.text(
        x + 0.5 * w,
        y + 0.25 * h,
        desc,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.6,
        color="0.22",
        linespacing=1.15,
    )


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

    out_png_canon = OUT_DIR_CANON / "pmodel_core_mapping_overview.png"
    out_png_private = OUT_DIR_PRIVATE / "pmodel_core_mapping_overview.png"
    out_png_public = OUT_DIR_PUBLIC / "pmodel_core_mapping_overview.png"

    out_json_canon = OUT_DIR_CANON / "pmodel_core_mapping_overview_metrics.json"
    out_json_private = OUT_DIR_PRIVATE / "pmodel_core_mapping_overview_metrics.json"
    out_json_public = OUT_DIR_PUBLIC / "pmodel_core_mapping_overview_metrics.json"

    _set_japanese_font()
    fig, ax = plt.subplots(figsize=(12.2, 7.2), dpi=220)
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
        (0.845, 0.955, "Part I 所管", "#2f6b39"),
        (0.18, 0.615, "Part III 所管", "#9a5c0f"),
        (0.49, 0.165, "Part II 所管", "#276749"),
    ]
    for gx, gy, text, color in group_labels:
        ax.text(
            gx,
            gy,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10.0,
            color=color,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )

    _box(
        ax=ax,
        xy=pos_p,
        wh=wh_input,
        fc="#dfe8f8",
        title="時間波密度（入力）",
        formula=r"$P(x)$",
        desc="静止極限の基底量",
        part="Part I",
    )
    _box(
        ax=ax,
        xy=pos_phi,
        wh=wh_hub,
        fc="#e7ddff",
        title="中心ハブ：ポテンシャル写像",
        formula=r"$\phi=-c^2\ln(P/P_0)$",
        desc="重力・時計・光への共通入口",
        part="Part I",
    )
    _box(
        ax=ax,
        xy=pos_gravity,
        wh=wh_leaf,
        fc="#e3f1e3",
        title="重力",
        formula=r"$\mathbf{a}=-\nabla\phi$",
        desc="P勾配へ滑る運動",
        part="Part I",
    )
    _box(
        ax=ax,
        xy=pos_clock,
        wh=wh_leaf,
        fc="#e3f1e3",
        title="時計（束縛モード）",
        formula=r"$d\tau/dt=(P_0/P)(d\tau/dt)_v$",
        desc="重力項と速度項を分離",
        part="Part I",
    )
    _box(
        ax=ax,
        xy=pos_light,
        wh=wh_leaf,
        fc="#e3f1e3",
        title="光（自由波）",
        formula=r"$n(P)=(P/P_0)^{2\beta}$",
        desc="高P側へ屈折",
        part="Part I",
    )
    _box(
        ax=ax,
        xy=pos_quantum,
        wh=wh_leaf,
        fc="#fdeccf",
        title="量子相関（selection）",
        formula=r"$P_\mu\leftrightarrow$ 微視結合",
        desc="Bell・核力・V-A（Part III）",
        part="Part III",
    )
    _box(
        ax=ax,
        xy=pos_cosmo,
        wh=wh_leaf,
        fc="#ddf3e8",
        title="宇宙論背景写像",
        formula=r"$1+z=P_{\mathrm{em}}/P_{\mathrm{obs}}$",
        desc=r"$P_{\mathrm{bg}}(t)$ の時間変化（Part II）",
        part="Part II",
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

    ax.text(
        0.5,
        0.058,
        r"Part I は写像と $\beta_{\mathrm{frozen}}$ を固定し、Part II/III は固定値のまま反証監査を行う。",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12.2,
        color="0.25",
    )

    fig.tight_layout()
    fig.savefig(out_png_canon, bbox_inches="tight")
    fig.savefig(out_png_private, bbox_inches="tight")
    fig.savefig(out_png_public, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "generated_utc": _utc_now(),
        "script": "scripts/theory/pmodel_core_mapping_overview.py",
        "outputs": {
            "png_canon": str(out_png_canon),
            "png_private": str(out_png_private),
            "png_public": str(out_png_public),
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
    _write_json(out_json_canon, payload)
    _write_json(out_json_private, payload)
    _write_json(out_json_public, payload)

    print(f"[ok] png(canon)  : {out_png_canon}")
    print(f"[ok] png(private): {out_png_private}")
    print(f"[ok] png(public) : {out_png_public}")
    print(f"[ok] json(canon) : {out_json_canon}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
