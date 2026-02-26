from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "private" / "theory"


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

def _box(*, ax: Any, xy: tuple[float, float], wh: tuple[float, float], text: str, fc: str) -> None:
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
        x + 0.5 * w,
        y + 0.5 * h,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        color="0.10",
        linespacing=1.18,
    )


# 関数: `_arrow` の入出力契約と処理意図を定義する。

def _arrow(*, ax: Any, a: tuple[float, float], b: tuple[float, float]) -> None:
    patch = FancyArrowPatch(
        a,
        b,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.2,
        color="0.25",
    )
    ax.add_patch(patch)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "pmodel_core_mapping_overview.png"
    out_json = OUT_DIR / "pmodel_core_mapping_overview_metrics.json"

    _set_japanese_font()
    fig, ax = plt.subplots(figsize=(12.5, 5.0), dpi=200)
    ax.set_axis_off()

    wh_main = (0.25, 0.20)
    wh_small = (0.30, 0.18)

    pos_p = (0.05, 0.67)
    pos_phi = (0.34, 0.67)
    pos_g = (0.66, 0.78)
    pos_clk = (0.66, 0.52)
    pos_light = (0.66, 0.26)
    pos_q = (0.05, 0.26)
    pos_cos = (0.34, 0.26)

    _box(
        ax=ax,
        xy=pos_p,
        wh=wh_main,
        fc="#f3f7ff",
        text="P(x)\n時間波密度 / Time-wave density\n（静止極限スカラー / scalar mode）",
    )
    _box(
        ax=ax,
        xy=pos_phi,
        wh=wh_main,
        fc="#f9f3ff",
        text="φ = −c² ln(P/P0)\nポテンシャル写像 / potential mapping\n（ニュートン形 / Newton form）",
    )
    _box(ax=ax, xy=pos_g, wh=wh_small, fc="#f8fff3", text="重力 / Gravity\n a = −∇φ\n（P勾配へ滑る / slide down P-gradient）")
    _box(
        ax=ax,
        xy=pos_clk,
        wh=wh_small,
        fc="#fff8f3",
        text="時計（束縛モード）/ Clocks (bound)\n dτ/dt = (P0/P)·(dτ/dt)v\n（速度項は標準回収 / SR-recovery）",
    )
    _box(
        ax=ax,
        xy=pos_light,
        wh=wh_small,
        fc="#f3fff9",
        text="光（自由波）/ Light (free wave)\n n(P)=(P/P0)^(2β)\n β: 応答強度 / response strength",
    )
    _box(
        ax=ax,
        xy=pos_q,
        wh=wh_main,
        fc="#fff3f3",
        text="量子（Part III）/ Quantum\n束縛モード・相関\n（selection 感度監査）",
    )
    _box(
        ax=ax,
        xy=pos_cos,
        wh=wh_main,
        fc="#f3fff3",
        text="宇宙論（Part II）/ Cosmology\n背景P: Pbg(t)\n 1+z = Pem/Pobs",
    )

    _arrow(ax=ax, a=(pos_p[0] + wh_main[0], pos_p[1] + 0.5 * wh_main[1]), b=(pos_phi[0], pos_phi[1] + 0.5 * wh_main[1]))
    _arrow(ax=ax, a=(pos_phi[0] + wh_main[0], pos_phi[1] + 0.65 * wh_main[1]), b=(pos_g[0], pos_g[1] + 0.5 * wh_small[1]))
    _arrow(ax=ax, a=(pos_phi[0] + wh_main[0], pos_phi[1] + 0.35 * wh_main[1]), b=(pos_clk[0], pos_clk[1] + 0.5 * wh_small[1]))
    _arrow(ax=ax, a=(pos_p[0] + wh_main[0], pos_p[1] - 0.03), b=(pos_cos[0] + 0.5 * wh_main[0], pos_cos[1] + wh_main[1]))
    _arrow(ax=ax, a=(pos_p[0] + 0.5 * wh_main[0], pos_p[1] - 0.03), b=(pos_q[0] + 0.5 * wh_main[0], pos_q[1] + wh_main[1]))
    _arrow(ax=ax, a=(pos_p[0] + wh_main[0], pos_p[1] + 0.20 * wh_main[1]), b=(pos_light[0], pos_light[1] + 0.5 * wh_small[1]))

    ax.text(
        0.5,
        0.03,
        "Part I: 写像と β を凍結。Part II/III: 凍結値のまま反証監査。",
        transform=ax.transAxes,
        ha="center",
        va="center",
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
            "script": "scripts/theory/pmodel_core_mapping_overview.py",
            "outputs": {"png": str(out_png), "metrics_json": str(out_json)},
            "figure_index_path": "output/private/theory/pmodel_core_mapping_overview.png",
            "notes": [
                "This is a conceptual diagram (not a numerical result).",
                "Text uses the Part I vocabulary: P-field -> φ -> (gravity, clocks, light), plus pointers to Part II/III.",
                "Velocity saturation δ is treated as an extension (not used in the Part I core).",
            ],
        },
    )

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
