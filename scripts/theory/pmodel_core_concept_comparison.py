"""
目的: 理論 topic の pmodel core concept comparison に対応する公開図と監査指標を再生成する。
入力: script 内の固定テーブル文言と locale 設定を用いる。
出力: output/public と output/private の canonical artifact を更新する。
前提: 論文本文は output/public 側の PDF/PNG を正として参照する。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.figure_locale_paths import localize_figure_output_path, resolve_figure_output_locale
from scripts.utils.plot_style import apply_paper_style, get_wavep_font_size, resolve_wavep_cjk_font_family

OUT_DIR_CANON = ROOT / "output" / "theory"
OUT_DIR_PRIVATE = ROOT / "output" / "private" / "theory"
OUT_DIR_PUBLIC = ROOT / "output" / "public" / "theory"
FIGURE_LOCALE = resolve_figure_output_locale()
IS_EN = FIGURE_LOCALE == "en"
STEM = "pmodel_core_concept_comparison"


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

        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        return


# 関数: `_write_json` の入出力契約と処理意図を定義する。
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `_artifact_paths` の入出力契約と処理意図を定義する。
def _artifact_paths() -> dict[str, Path]:
    paths = {
        "png_private": localize_figure_output_path(OUT_DIR_PRIVATE / f"{STEM}.png", root=ROOT, locale=FIGURE_LOCALE),
        "png_public": localize_figure_output_path(OUT_DIR_PUBLIC / f"{STEM}.png", root=ROOT, locale=FIGURE_LOCALE),
        "pdf_private": localize_figure_output_path(OUT_DIR_PRIVATE / f"{STEM}.pdf", root=ROOT, locale=FIGURE_LOCALE),
        "pdf_public": localize_figure_output_path(OUT_DIR_PUBLIC / f"{STEM}.pdf", root=ROOT, locale=FIGURE_LOCALE),
        "json_private": localize_figure_output_path(OUT_DIR_PRIVATE / f"{STEM}_metrics.json", root=ROOT, locale=FIGURE_LOCALE),
        "json_public": localize_figure_output_path(OUT_DIR_PUBLIC / f"{STEM}_metrics.json", root=ROOT, locale=FIGURE_LOCALE),
    }
    if not IS_EN:
        paths["png_canon"] = OUT_DIR_CANON / f"{STEM}.png"
        paths["pdf_canon"] = OUT_DIR_CANON / f"{STEM}.pdf"
        paths["json_canon"] = OUT_DIR_CANON / f"{STEM}_metrics.json"

    return paths


# 関数: `_save_figure_bundle` の入出力契約と処理意図を定義する。
def _save_figure_bundle(fig: plt.Figure) -> dict[str, str]:
    paths = _artifact_paths()
    save_targets = {key: value for key, value in paths.items() if key.startswith(("png_", "pdf_"))}

    for path in save_targets.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        for path in save_targets.values():
            if path.suffix.lower() == ".png":
                fig.savefig(path, dpi=220)
            else:
                fig.savefig(path)

    return {key: str(value).replace("\\", "/") for key, value in paths.items()}


# 関数: `_build_table_content` の入出力契約と処理意図を定義する。
def _build_table_content() -> tuple[list[str], list[list[str]], str]:
    col_labels = [
        _t("観点", "Viewpoint"),
        _t("参照枠（GR）", "Reference frame (GR)"),
        "P-model",
    ]
    cell_text = [
        [_t("重力の本質", "Nature of gravity"), _t("時空の曲率", "Curvature of spacetime"), _t("P勾配への応答", "Response to the P gradient")],
        [_t("運動の記述", "Equation of motion"), _t("測地線に沿う", "Along geodesics"), _t("P勾配へ滑り落ちる", "Slides down the P gradient")],
        [_t("光の伝播", "Light propagation"), _t("光は時空に沿う", "Light follows spacetime"), _t("光は高P側へ屈折", "Refraction toward high P")],
        [_t("時間の遅れ", "Time delay"), _t("時空の計量", "Metric effect"), _t("P比（P∞/P）", "P ratio (P∞/P)")],
        [_t("赤方偏移", "Redshift"), _t("空間膨張", "Expansion of space"), _t("背景Pの時間変化", "Time variation of background P")],
    ]
    caption = _t(
        "図2: P-modelと参照枠（GR）の概念比較。P-modelは時空の幾何ではなく、"
        "時間波密度Pの空間変化として重力・光伝播を記述する。"
        "両者は弱場で同等の観測量を与えるが、概念的枠組みは独立である。",
        "Figure 2: Conceptual comparison between the P-model and the reference "
        "frame (GR). The P-model does not treat gravity as spacetime geometry; "
        "instead, it describes gravity and light propagation as responses to "
        "spatial variation of the time-wave density P. Both frameworks reproduce "
        "the same weak-field observables, but their conceptual bases are independent.",
    )
    return col_labels, cell_text, caption


# 関数: `_build_figure` の入出力契約と処理意図を定義する。
def _build_figure() -> tuple[plt.Figure, dict[str, Any]]:
    apply_paper_style()
    _configure_font()
    col_labels, cell_text, _caption = _build_table_content()

    fig, ax = plt.subplots(figsize=(10.8, 3.7), dpi=200)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.10)
    ax.set_axis_off()

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table_font = 10.2 if IS_EN else max(get_wavep_font_size("title", name="part4_verification") + 2.6, 13.6)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(table_font)
    tbl.scale(1.0, 1.60 if IS_EN else 2.10)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("0.35")
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor("#f1f1f1")
            cell.set_text_props(weight="bold", color="0.15")
        else:
            cell.set_facecolor("#ffffff" if (r % 2 == 1) else "#fbfbfb")

        if c == 0:
            cell.set_width(0.24 if not IS_EN else 0.25)
        elif c == 1:
            cell.set_width(0.37 if not IS_EN else 0.37)
        elif c == 2:
            cell.set_width(0.37 if not IS_EN else 0.36)

    diag = {
        "locale": FIGURE_LOCALE,
        "columns": col_labels,
        "n_rows": len(cell_text),
    }
    return fig, diag


# 関数: `main` の入出力契約と処理意図を定義する。
def main() -> None:
    fig, diag = _build_figure()
    outputs = _save_figure_bundle(fig)
    plt.close(fig)

    payload = {
        "generated_utc": _utc_now(),
        "locale": FIGURE_LOCALE,
        "script": "scripts/theory/pmodel_core_concept_comparison.py",
        "outputs": outputs,
        "diag": diag,
        "notes": [
            "This is a conceptual comparison diagram rather than a numerical fit.",
            "The English locale is emitted under output/public/theory/locales/en and output/private/theory/locales/en.",
        ],
    }

    json_paths = [Path(outputs["json_private"]), Path(outputs["json_public"])]
    if not IS_EN and "json_canon" in outputs:
        json_paths.append(Path(outputs["json_canon"]))

    for path in json_paths:
        _write_json(path, payload)

    print(f"[ok] locale: {FIGURE_LOCALE}")
    print(f"[ok] pdf   : {outputs['pdf_public']}")
    print(f"[ok] png   : {outputs['png_public']}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。
if __name__ == "__main__":
    main()
