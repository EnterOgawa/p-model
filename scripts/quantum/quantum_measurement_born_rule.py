from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


from figure_japanese_localizer import enable_japanese_figure_localization

enable_japanese_figure_localization()

# クラス: `Config` の責務と境界条件を定義する。
@dataclass(frozen=True)
class Config:
    fig_w_in: float = 11.0
    fig_h_in: float = 6.2
    dpi: int = 180


# 関数: `_configure_japanese_font` の入出力契約と処理意図を定義する。
def _configure_japanese_font() -> None:
    import matplotlib as mpl
    from matplotlib import font_manager as fm

    candidates = [
        "Yu Gothic",
        "Meiryo",
        "MS Gothic",
        "MS PGothic",
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAexGothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            mpl.rcParams["font.family"] = name
            mpl.rcParams["font.sans-serif"] = [name] + list(mpl.rcParams.get("font.sans-serif", []))
            break

    mpl.rcParams["axes.unicode_minus"] = False


# 関数: `_add_box` の入出力契約と処理意図を定義する。

def _add_box(ax, x: float, y: float, w: float, h: float, text: str, *, fc: str, ec: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=10.5)


# 関数: `_add_arrow` の入出力契約と処理意図を定義する。

def _add_arrow(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    arrow = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=14, linewidth=1.4, color="#333333")
    ax.add_patch(arrow)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    _configure_japanese_font()
    cfg = Config()

    fig = plt.figure(figsize=(cfg.fig_w_in, cfg.fig_h_in), dpi=cfg.dpi)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    fig.suptitle("量子測定の位置づけ（Born則と状態更新）", fontsize=13)

    # Boxes
    _add_box(
        ax,
        0.05,
        0.74,
        0.38,
        0.18,
        "P場（時間波密度）\n"
        "u = ln(P/P0),  φ = -c²u\n"
        "（Part I の写像・局所記述）",
        fc="#e8f2ff",
        ec="#2b6cb0",
    )
    _add_box(
        ax,
        0.57,
        0.74,
        0.38,
        0.18,
        "束縛モード包絡\n"
        "ψ(x,t)\n"
        "（短波長極限でSchr/KG）",
        fc="#f0fff4",
        ec="#2f855a",
    )
    _add_box(
        ax,
        0.57,
        0.45,
        0.38,
        0.18,
        "検出率 / クリック確率\n"
        "λ(x,t) ∝ |ψ|²\n"
        "（Born則：運用上の採用）",
        fc="#fffaf0",
        ec="#b7791f",
    )
    _add_box(
        ax,
        0.57,
        0.16,
        0.38,
        0.18,
        "測定記録 m\n"
        "ポインタ状態 / 粗視化\n"
        "ρ → ρ_m（条件付き更新）",
        fc="#fff5f5",
        ec="#c53030",
    )
    _add_box(
        ax,
        0.05,
        0.38,
        0.38,
        0.20,
        "選別・解析パイプライン\n"
        "w_ab(λ)（設定依存の受理）\n"
        "Bell解析の系統入口",
        fc="#f7fafc",
        ec="#4a5568",
    )

    # Arrows
    _add_arrow(ax, 0.43, 0.83, 0.57, 0.83)  # P-field -> ψ
    _add_arrow(ax, 0.76, 0.74, 0.76, 0.63)  # ψ -> Born
    _add_arrow(ax, 0.76, 0.45, 0.76, 0.34)  # Born -> update
    _add_arrow(ax, 0.43, 0.48, 0.57, 0.54)  # selection -> Born

    ax.text(
        0.05,
        0.05,
        "初期版では、導出済み要素と採用要素の境界を固定し、\n"
        "「半古典的」という批判に対する検証入口を明示する。Born則と更新則の第一原理導出は今後の課題。",
        fontsize=9.5,
        ha="left",
        va="bottom",
        color="#333333",
    )

    out_png = out_dir / "quantum_measurement_born_rule_flow.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.10",
        "title": "Quantum measurement (Born rule & state update) positioning",
        "sources": {
            "doc": "doc/quantum/15_quantum_measurement_born_rule.md",
            "paper_summary": "doc/paper/12_part3_quantum.md (Sec. 2.5.2)",
            "roadmap": "doc/ROADMAP.md (Step 7.10)",
        },
        "postulates": [
            {
                "id": "QM-P1",
                "kind": "definition",
                "statement": "Introduce ψ as a complex envelope of a bound mode of u=ln(P/P0) (operational; not claimed as first-principles).",
            },
            {
                "id": "QM-P2",
                "kind": "effective_limit",
                "statement": "Rest-phase + proper-time mapping yields the Schr/KG entry point with V=mφ in the weak-field nonrelativistic limit.",
            },
            {
                "id": "QM-P3",
                "kind": "probability_rule",
                "statement": "Born rule is adopted operationally: detection rate λ(x,t) ∝ |ψ(x,t)|².",
            },
            {
                "id": "QM-P4",
                "kind": "state_update",
                "statement": "Measurement update is treated as conditionalization on a macroscopic record (projective / POVM update).",
            },
            {
                "id": "QM-P5",
                "kind": "systematics_positioning",
                "statement": "Selection (acceptance) in Bell time-tag analysis is part of the measurement pipeline; treat as a systematics knob w_ab(λ).",
            },
        ],
        "update_rule": {
            "projective": {
                "p_m": "p(m)=Tr(Π_m ρ)",
                "rho_m": "ρ_m = Π_m ρ Π_m / Tr(Π_m ρ)",
            },
            "povm": {
                "p_m": "p(m)=Tr(E_m ρ), with E_m=M_m^† M_m and ΣE_m=I",
                "rho_m": "ρ_m = M_m ρ M_m^† / Tr(E_m ρ)",
            },
        },
        "open_problems": [
            "First-principles derivation of Born rule from P-field dynamics",
            "First-principles derivation of measurement update (irreversibility/pointer basis) from P-field + macroscopic apparatus",
            "Identification of ψ with a unique P-field degree of freedom (phase, complex structure)",
            "Spin/charge/EM/strong interactions (Step 7.11+)",
        ],
        "outputs": {
            "figure_png": str(out_png.relative_to(root)),
        },
    }

    out_json = out_dir / "quantum_measurement_born_rule_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
