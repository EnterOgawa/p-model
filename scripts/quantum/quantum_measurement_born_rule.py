"""
目的: 量子 topic の quantum measurement born rule に対応する公開図・表・監査指標を再生成する。
入力: script 内の既定パラメータと必要な公開データまたは基準値を用いる。
出力: output/public と output/private の canonical artifact を更新する。
前提: 論文本文と README はこの script が出力する公開成果物を正として参照する。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.plot_style import (  # noqa: E402
    apply_paper_style,
    apply_wavep_figure_layout,
    get_wavep_font_size,
    resolve_wavep_cjk_font_family,
)
from scripts.utils.figure_locale_paths import localize_figure_output_path  # noqa: E402
from scripts.quantum.figure_japanese_localizer import (  # noqa: E402
    enable_japanese_figure_localization,
    get_figure_language,
)

enable_japanese_figure_localization()

# クラス: `Config` の責務と境界条件を定義する。
@dataclass(frozen=True)
class Config:
    dpi: int = 180


# 関数: `_configure_japanese_font` の入出力契約と処理意図を定義する。

def _configure_japanese_font() -> None:
    import matplotlib as mpl

    preferred = resolve_wavep_cjk_font_family(preferred_name="Noto Sans CJK JP")
    if preferred:
        mpl.rcParams["font.family"] = [preferred, "DejaVu Sans"]
        mpl.rcParams["font.sans-serif"] = [preferred, "DejaVu Sans"]

    mpl.rcParams["axes.unicode_minus"] = False


# 関数: `_save_figure_bundle` の入出力契約と処理意図を定義する。

def _save_figure_bundle(*, fig, stem: str) -> dict[str, str]:
    out_public = ROOT / "output" / "public" / "quantum"
    out_private = ROOT / "output" / "private" / "quantum"
    out_canon = ROOT / "output" / "quantum"
    outputs = {
        "png_public": localize_figure_output_path(out_public / f"{stem}.png", root=ROOT),
        "png_private": localize_figure_output_path(out_private / f"{stem}.png", root=ROOT),
        "png_canon": out_canon / f"{stem}.png",
        "pdf_public": localize_figure_output_path(out_public / f"{stem}.pdf", root=ROOT),
        "pdf_private": localize_figure_output_path(out_private / f"{stem}.pdf", root=ROOT),
        "pdf_canon": out_canon / f"{stem}.pdf",
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig.savefig(outputs["png_public"], dpi=220)
        fig.savefig(outputs["pdf_public"])

    for key in ["png_private", "png_canon", "pdf_private", "pdf_canon"]:
        source_key = "png_public" if key.startswith("png_") else "pdf_public"
        shutil.copy2(outputs[source_key], outputs[key])

    return {key: str(value.relative_to(ROOT)).replace("\\", "/") for key, value in outputs.items()}


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
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=get_wavep_font_size("base"),
        linespacing=1.14,
    )


# 関数: `_add_arrow` の入出力契約と処理意図を定義する。

def _add_arrow(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    arrow = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=14, linewidth=1.4, color="#333333")
    ax.add_patch(arrow)


# 関数: `_build_text_map` の入出力契約と処理意図を定義する。

def _build_text_map(*, lang: str) -> dict[str, str]:
    if lang == "en":
        return {
            "suptitle": "Quantum Measurement Positioning (Born Rule and State Update)",
            "p_field": "P field (time-wave density)\n"
            "u = ln(P/P0),  φ = -c²u\n"
            "(Part I mapping and local description)",
            "bound_mode": "Bound-mode envelope\n"
            "ψ(x,t)\n"
            "(Schr/KG in the short-wavelength limit)",
            "born_rate": "Detection rate / click probability\n"
            "λ(x,t) ∝ |ψ|²\n"
            "(Born rule: operational adoption)",
            "record_update": "Measurement record m\n"
            "pointer states / coarse-graining\n"
            "ρ → ρ_m (conditional update)",
            "selection": "Selection and analysis pipeline\n"
            "w_ab(λ) (setting-dependent acceptance)\n"
            "systematics entry for Bell analysis",
            "note": "This first release fixes the boundary between derived elements and adopted elements,\n"
            "and makes the verification entry explicit against the criticism that the model is only semi-classical.\n"
            "A first-principles derivation of the Born rule and update rule remains future work.",
        }

    return {
        "suptitle": "量子測定の位置づけ（Born則と状態更新）",
        "p_field": "P場（時間波密度）\n"
        "u = ln(P/P0),  φ = -c²u\n"
        "（Part I の写像・局所記述）",
        "bound_mode": "束縛モード包絡\n"
        "ψ(x,t)\n"
        "（短波長極限でSchr/KG）",
        "born_rate": "検出率 / クリック確率\n"
        "λ(x,t) ∝ |ψ|²\n"
        "（Born則：運用上の採用）",
        "record_update": "測定記録 m\n"
        "ポインタ状態 / 粗視化\n"
        "ρ → ρ_m（条件付き更新）",
        "selection": "選別・解析パイプライン\n"
        "w_ab(λ)（設定依存の受理）\n"
        "Bell解析の系統入口",
        "note": "初期版では、導出済み要素と採用要素の境界を固定し、\n"
        "「半古典的」という批判に対する検証入口を明示する。Born則と更新則の第一原理導出は今後の課題。",
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    out_dir = ROOT / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_paper_style()
    _configure_japanese_font()
    cfg = Config()
    lang = get_figure_language(default="ja")
    text_map = _build_text_map(lang=lang)
    is_en = lang == "en"

    fig = plt.figure(dpi=cfg.dpi)
    apply_wavep_figure_layout(fig, template="paper_diagram")
    fig.subplots_adjust(top=0.905, bottom=(0.165 if is_en else 0.125))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    suptitle = fig.suptitle(text_map["suptitle"], fontsize=get_wavep_font_size("suptitle"))
    suptitle.set_fontsize(12.4)

    # Boxes
    _add_box(
        ax,
        0.05,
        0.74,
        0.38,
        0.18,
        text_map["p_field"],
        fc="#e8f2ff",
        ec="#2b6cb0",
    )
    _add_box(
        ax,
        0.57,
        0.74,
        0.38,
        0.18,
        text_map["bound_mode"],
        fc="#f0fff4",
        ec="#2f855a",
    )
    _add_box(
        ax,
        0.57,
        0.45,
        0.38,
        0.18,
        text_map["born_rate"],
        fc="#fffaf0",
        ec="#b7791f",
    )
    _add_box(
        ax,
        0.57,
        (0.19 if is_en else 0.16),
        0.38,
        0.18,
        text_map["record_update"],
        fc="#fff5f5",
        ec="#c53030",
    )
    _add_box(
        ax,
        0.05,
        0.38,
        0.38,
        0.20,
        text_map["selection"],
        fc="#f7fafc",
        ec="#4a5568",
    )

    # Arrows
    _add_arrow(ax, 0.43, 0.83, 0.57, 0.83)  # P-field -> ψ
    _add_arrow(ax, 0.76, 0.74, 0.76, 0.63)  # ψ -> Born
    _add_arrow(ax, 0.76, 0.45, 0.76, (0.37 if is_en else 0.34))  # Born -> update
    _add_arrow(ax, 0.43, 0.48, 0.57, 0.54)  # selection -> Born

    fig.text(
        0.10,
        (0.060 if is_en else 0.052),
        text_map["note"],
        fontsize=(get_wavep_font_size("note") - 0.5 if is_en else get_wavep_font_size("note")),
        ha="left",
        va="bottom",
        color="#333333",
    )

    outputs = _save_figure_bundle(fig=fig, stem="quantum_measurement_born_rule_flow")
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
            "figure_png": outputs["png_public"],
            "figure_pdf": outputs["pdf_public"],
        },
    }

    out_json = out_dir / "quantum_measurement_born_rule_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {ROOT / outputs['png_public']}")
    print(f"Wrote: {ROOT / outputs['pdf_public']}")
    print(f"Wrote: {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
