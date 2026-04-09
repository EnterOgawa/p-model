#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_profile_content.py

論文 profile ごとの本文ソース、出力名、見出し表記を一元管理する。

目的:
- `paper_build.py` / `paper_html.py` / `paper_latex.py` / QC 系で、
  profile ごとの条件分岐を散在させない。
- Ver.1.1 で導入する Part III-A / Part III-B の canonical source を
  `doc/paper/12_part3a_quantum_foundations.md` /
  `doc/paper/12_part3b_quantum_verification.md` に固定し、
  legacy な Part III family source は互換用途に残す。
"""

from __future__ import annotations

from pathlib import Path

from scripts.summary import paper_locale_registry as locale_registry


PAPER_PROFILES: tuple[str, ...] = (
    "paper",
    "part2_astrophysics",
    "part3_quantum",
    "part3a_quantum_foundations",
    "part3b_quantum_verification",
    "part4_verification",
    "part5_future_predictions",
)

PART3_COMPAT_PROFILE = "part3_quantum"
PART3A_PROFILE = "part3a_quantum_foundations"
PART3B_PROFILE = "part3b_quantum_verification"


_PROFILE_TO_MANUSCRIPT: dict[str, str] = {
    "paper": "paper",
    "part2_astrophysics": "part2_astrophysics",
    PART3_COMPAT_PROFILE: "part3_quantum",
    PART3A_PROFILE: "part3a_quantum_foundations",
    PART3B_PROFILE: "part3b_quantum_verification",
    "part4_verification": "part4_verification",
    "part5_future_predictions": "part5_future_predictions",
}

_PROFILE_TO_LINT_MANUSCRIPTS: dict[str, tuple[str, ...]] = {
    "paper": ("paper",),
    "part2_astrophysics": ("part2_astrophysics",),
    PART3_COMPAT_PROFILE: ("part3_quantum",),
    PART3A_PROFILE: ("part3a_quantum_foundations",),
    PART3B_PROFILE: ("part3b_quantum_verification",),
    "part4_verification": ("part4_verification",),
    "part5_future_predictions": ("part5_future_predictions",),
}

_PROFILE_TO_HTML: dict[str, str] = {
    "paper": "pmodel_paper.html",
    "part2_astrophysics": "pmodel_paper_part2_astrophysics.html",
    PART3_COMPAT_PROFILE: "pmodel_paper_part3_quantum.html",
    PART3A_PROFILE: "pmodel_paper_part3a_quantum_foundations.html",
    PART3B_PROFILE: "pmodel_paper_part3b_quantum_verification.html",
    "part4_verification": "pmodel_paper_part4_verification.html",
    "part5_future_predictions": "pmodel_paper_part5_future_predictions.html",
}

_PROFILE_TO_DOCX: dict[str, str] = {
    "paper": "pmodel_paper.docx",
    "part2_astrophysics": "pmodel_paper_part2_astrophysics.docx",
    PART3_COMPAT_PROFILE: "pmodel_paper_part3_quantum.docx",
    PART3A_PROFILE: "pmodel_paper_part3a_quantum_foundations.docx",
    PART3B_PROFILE: "pmodel_paper_part3b_quantum_verification.docx",
    "part4_verification": "pmodel_paper_part4_verification.docx",
    "part5_future_predictions": "pmodel_paper_part5_future_predictions.docx",
}

_PROFILE_TO_PDF: dict[str, str] = {
    "paper": "pmodel_paper.pdf",
    "part2_astrophysics": "pmodel_paper_part2_astrophysics.pdf",
    PART3_COMPAT_PROFILE: "pmodel_paper_part3_quantum.pdf",
    PART3A_PROFILE: "pmodel_paper_part3a_quantum_foundations.pdf",
    PART3B_PROFILE: "pmodel_paper_part3b_quantum_verification.pdf",
    "part4_verification": "pmodel_paper_part4_verification.pdf",
    "part5_future_predictions": "pmodel_paper_part5_future_predictions.pdf",
}

_PROFILE_TO_TEX: dict[str, str] = {
    "paper": "pmodel_paper.tex",
    "part2_astrophysics": "pmodel_paper_part2_astrophysics.tex",
    PART3_COMPAT_PROFILE: "pmodel_paper_part3_quantum.tex",
    PART3A_PROFILE: "pmodel_paper_part3a_quantum_foundations.tex",
    PART3B_PROFILE: "pmodel_paper_part3b_quantum_verification.tex",
    "part4_verification": "pmodel_paper_part4_verification.tex",
    "part5_future_predictions": "pmodel_paper_part5_future_predictions.tex",
}

_PROFILE_TO_FONT_PROFILE: dict[str, str] = {
    "paper": "paper",
    "part2_astrophysics": "part2_astrophysics",
    PART3_COMPAT_PROFILE: "part3_quantum",
    PART3A_PROFILE: "part3_quantum",
    PART3B_PROFILE: "part3b_quantum_verification",
    "part4_verification": "part4_verification",
    "part5_future_predictions": "part5_future_predictions",
}

_PROFILE_TO_FONT_FLOOR: dict[str, tuple[str, str]] = {
    "paper": ("7.8", "7.8"),
    "part2_astrophysics": ("7.8", "7.8"),
    PART3_COMPAT_PROFILE: ("7.8", "7.8"),
    PART3A_PROFILE: ("7.8", "7.8"),
    PART3B_PROFILE: ("7.8", "7.8"),
    "part4_verification": ("7.8", "7.8"),
    "part5_future_predictions": ("7.8", "7.8"),
}

_PROFILE_TO_HTML_TITLE: dict[str, str] = {
    "paper": "P-model Part I（コア理論）",
    "part2_astrophysics": "P-model Part II（宇宙物理編）",
    PART3_COMPAT_PROFILE: "P-model Part III（量子物理編）",
    PART3A_PROFILE: "P-model Part III-A（量子基盤理論）",
    PART3B_PROFILE: "P-model Part III-B（量子検証応用）",
    "part4_verification": "P-model Part IV（検証資料）",
    "part5_future_predictions": "P-model Part V（未来予測編）",
}

_PROFILE_TO_HTML_SUBTITLE: dict[str, str] = {
    "paper": "記号規約・最小仮定・写像（公開体裁）",
    "part2_astrophysics": "応用検証：宇宙物理（公開体裁）",
    PART3_COMPAT_PROFILE: "応用検証：量子物理（公開体裁）",
    PART3A_PROFILE: "理論基盤：Pψ 接続・測定・Bell 選別入口（公開体裁）",
    PART3B_PROFILE: "応用検証：量子データ再解析と差分予測（公開体裁）",
    "part4_verification": "検証方法と公開成果物への参照先（GitHub）",
    "part5_future_predictions": "差分予測と将来観測の決着条件（公開体裁）",
}

_PROFILE_TO_HTML_BADGE: dict[str, str] = {
    "paper": "Part I",
    "part2_astrophysics": "Part II",
    PART3_COMPAT_PROFILE: "Part III",
    PART3A_PROFILE: "Part III-A",
    PART3B_PROFILE: "Part III-B",
    "part4_verification": "Part IV",
    "part5_future_predictions": "Part V",
}

_PROFILE_TO_TEX_TITLE: dict[str, str] = {
    "paper": r"時間波ダイナミクスに基づく統一理論 (The P-model)\\[0.5em]Part I: 理論的基礎と写像原理",
    "part2_astrophysics": r"時間波ダイナミクスに基づく統一理論 (The P-model)\\[0.5em]Part II: 宇宙物理学および宇宙論的検証\\[1em]\large",
    PART3_COMPAT_PROFILE: r"時間波ダイナミクスに基づく統一理論 (The P-model)\\[0.5em]Part III: 微視的および量子的現象の再評価",
    PART3A_PROFILE: r"時間波ダイナミクスに基づく統一理論 (The P-model)\\[0.5em]Part III-A: 量子基盤理論",
    PART3B_PROFILE: r"時間波ダイナミクスに基づく統一理論 (The P-model)\\[0.5em]Part III-B: 量子検証応用",
    "part4_verification": r"時間波ダイナミクスに基づく統一理論 (The P-model)\\[0.5em]Part IV: 再現性監査・公開成果物レジストリ・更新運用\\[1em]\large",
    "part5_future_predictions": r"時間波ダイナミクスに基づく統一理論 (The P-model)\\[0.5em]Part V: 将来観測予測\\[1em]\large",
}

_DEFAULT_ROOT = Path(__file__).resolve().parents[2]
_FIGURES_INDEX_KEY = "figures_index"
_DEFINITIONS_KEY = "definitions"
_UNCERTAINTY_KEY = "uncertainty"
_LLR_APPENDIX_KEY = "llr_appendix"
_QUANTUM_APPENDIX_A_KEY = "quantum_appendix_a"
_DATA_SOURCES_KEY = "data_sources"
_REFERENCES_KEY = "references"

_PROFILE_TO_POST_BIBLIOGRAPHY_TEX: dict[str, str] = {
    "part5_future_predictions": r"""
\clearpage

\section*{最後に}

私は哲学という学問について深く考えることがあった

物事を深く理解しようとすると、最終的に科学に行きつく

しかし、今までの科学には哲学に似た部分があったと感じている

それが $\alpha$ であり、特異点である

宇宙も量子も同じ物理的事象であるにもかかわらず、異なる考え方をする

ここが哲学のようなものと感じるゆえんである

人は不完全な存在なので不完全なものを好む

これ、そのものが哲学といえる

この世界には物体の位置を表す3次元と時間しかない

このシンプルな問いからすべては始まった

完全でなくてもいい。科学もまた、人が扱うものだから

大切なことはただひとつ

\textbf{「人は人のために生き、人に生かされている」}

このことを深く理解することが、哲学や科学の本質といえる

この理解こそが、平和へつながる唯一の道徳といえる

この世には神も仏も存在しない

地位も階級も存在しない

存在するのは、時間という波が形を変えて在るだけに過ぎない

何かにすがるのではなく

今生きていること

周囲の人々により生かされていることに感謝することが

幸せにつながる唯一の道である

\vspace{2em}
\hfill Shunji Ogawa
""".strip(),
}


_PART3A_H1 = "# 時間波ダイナミクスに基づく統一理論 (The P-model): Part III-A: 量子基盤理論"
_PART3B_H1 = "# 時間波ダイナミクスに基づく統一理論 (The P-model): Part III-B: 量子検証応用"

_PART3B_PREAMBLE = """# 時間波ダイナミクスに基づく統一理論 (The P-model): Part III-B: 量子検証応用

本稿は P-model 量子編のうち、公開一次データを用いた検証結果と差分予測を扱う。
理論定義、Pψ 接続、電磁気の位置づけ、Born則の採用範囲、測定の有効記述、
Bell 選別の入口は Part III-A を正本とする。

---

## 要旨（Abstract）

本稿（Part III-B）は、Part III-A で固定した最小仮定を保持したまま、
Bell・干渉・量子時計・核物理・物性/熱・弱い相互作用・BBN を
同一I/Fで再解析し、Pass/Watch/Reject の横断サマリと差分予測を固定する。
目的は量子力学の置換宣言ではなく、どの仮定がどの一次データで破れ得るかを
再現可能な検証パックとして提示することにある。

---

## 1. 序論（Introduction）

Part III-B は、量子編の「理論を述べる章」と「検証で採否を決める章」を分離するために、
Ver.1.1 で Part III-A から切り出した検証側の文書である。
本稿の読者は、理論的前提と適用範囲を Part III-A で確認したうえで、
本稿では 3章以降のデータ・結果・差分予測・議論を追えばよい。

節番号は移行期の参照安定性を優先して、Part III-A / III-B 間で連番を維持する。
したがって本稿は 3章から始まるが、これは Part III-A の 2章に続く構成である。

---
"""

_PART3A_POSTSCRIPT = """

---

Part III-A は量子編の理論基盤を固定するための文書であり、
公開一次データに基づく検証結果、横断スコアボード、差分予測、
棄却条件パックは Part III-B を正本とする。
"""


# 関数: `_resolve_root` の入出力契約と処理意図を定義する。
def _resolve_root(root: Path | None = None) -> Path:
    return root if root is not None else _DEFAULT_ROOT


# 関数: `_to_rel_from_root` の入出力契約と処理意図を定義する。
def _to_rel_from_root(path: Path, *, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


# 関数: `resolve_manuscript_path` の入出力契約と処理意図を定義する。
def resolve_manuscript_path(root: Path, profile: str, locale: str | None = None) -> Path:
    manifest_key = _PROFILE_TO_MANUSCRIPT[profile]
    return locale_registry.resolve_source_path(root, manifest_key, locale=locale)


# 関数: `resolve_lint_manuscripts` の入出力契約と処理意図を定義する。
def resolve_lint_manuscripts(profile: str, *, root: Path | None = None, locale: str | None = None) -> tuple[str, ...]:
    resolved_root = _resolve_root(root)
    keys = _PROFILE_TO_LINT_MANUSCRIPTS[profile]
    return tuple(
        _to_rel_from_root(locale_registry.resolve_source_path(resolved_root, key, locale=locale), root=resolved_root)
        for key in keys
    )


# 関数: `resolve_html_name` の入出力契約と処理意図を定義する。
def resolve_html_name(profile: str, locale: str | None = None) -> str:
    return locale_registry.localized_output_name(_PROFILE_TO_HTML[profile], locale=locale)


# 関数: `resolve_docx_name` の入出力契約と処理意図を定義する。
def resolve_docx_name(profile: str, locale: str | None = None) -> str:
    return locale_registry.localized_output_name(_PROFILE_TO_DOCX[profile], locale=locale)


# 関数: `resolve_pdf_name` の入出力契約と処理意図を定義する。
def resolve_pdf_name(profile: str, locale: str | None = None) -> str:
    return locale_registry.localized_output_name(_PROFILE_TO_PDF[profile], locale=locale)


# 関数: `resolve_tex_name` の入出力契約と処理意図を定義する。
def resolve_tex_name(profile: str, locale: str | None = None) -> str:
    return locale_registry.localized_output_name(_PROFILE_TO_TEX[profile], locale=locale)


# 関数: `resolve_font_profile` の入出力契約と処理意図を定義する。
def resolve_font_profile(profile: str) -> str:
    return _PROFILE_TO_FONT_PROFILE[profile]


# 関数: `resolve_font_floors` の入出力契約と処理意図を定義する。
def resolve_font_floors(profile: str) -> tuple[str, str]:
    return _PROFILE_TO_FONT_FLOOR[profile]


# 関数: `resolve_html_title` の入出力契約と処理意図を定義する。
def resolve_html_title(profile: str) -> str:
    return _PROFILE_TO_HTML_TITLE[profile]


# 関数: `resolve_html_subtitle` の入出力契約と処理意図を定義する。
def resolve_html_subtitle(profile: str) -> str:
    return _PROFILE_TO_HTML_SUBTITLE[profile]


# 関数: `resolve_html_badge` の入出力契約と処理意図を定義する。
def resolve_html_badge(profile: str) -> str:
    return _PROFILE_TO_HTML_BADGE[profile]


# 関数: `resolve_tex_title` の入出力契約と処理意図を定義する。
def resolve_tex_title(profile: str) -> str:
    return _PROFILE_TO_TEX_TITLE[profile]


# 関数: `resolve_figures_index_path` の入出力契約と処理意図を定義する。
def resolve_figures_index_path(root: Path, locale: str | None = None) -> Path:
    return locale_registry.resolve_source_path(root, _FIGURES_INDEX_KEY, locale=locale)


# 関数: `resolve_definitions_path` の入出力契約と処理意図を定義する。
def resolve_definitions_path(root: Path, locale: str | None = None) -> Path:
    return locale_registry.resolve_source_path(root, _DEFINITIONS_KEY, locale=locale)


# 関数: `resolve_uncertainty_path` の入出力契約と処理意図を定義する。
def resolve_uncertainty_path(root: Path, locale: str | None = None) -> Path:
    return locale_registry.resolve_source_path(root, _UNCERTAINTY_KEY, locale=locale)


# 関数: `resolve_llr_appendix_path` の入出力契約と処理意図を定義する。
def resolve_llr_appendix_path(root: Path, locale: str | None = None) -> Path:
    return locale_registry.resolve_source_path(root, _LLR_APPENDIX_KEY, locale=locale)


# 関数: `resolve_quantum_appendix_a_path` の入出力契約と処理意図を定義する。
def resolve_quantum_appendix_a_path(root: Path, locale: str | None = None) -> Path:
    return locale_registry.resolve_source_path(root, _QUANTUM_APPENDIX_A_KEY, locale=locale)


# 関数: `resolve_data_sources_path` の入出力契約と処理意図を定義する。
def resolve_data_sources_path(root: Path, locale: str | None = None) -> Path:
    return locale_registry.resolve_source_path(root, _DATA_SOURCES_KEY, locale=locale)


# 関数: `resolve_references_path` の入出力契約と処理意図を定義する。
def resolve_references_path(root: Path, locale: str | None = None) -> Path:
    return locale_registry.resolve_source_path(root, _REFERENCES_KEY, locale=locale)


# 関数: `resolve_post_bibliography_tex` の入出力契約と処理意図を定義する。
def resolve_post_bibliography_tex(profile: str) -> str:
    return _PROFILE_TO_POST_BIBLIOGRAPHY_TEX.get(profile, "")


# 関数: `uses_quantum_table1` の入出力契約と処理意図を定義する。
def uses_quantum_table1(profile: str) -> bool:
    return profile in {PART3_COMPAT_PROFILE, PART3B_PROFILE}


# 関数: `should_run_quantum_presteps` の入出力契約と処理意図を定義する。
def should_run_quantum_presteps(profile: str) -> bool:
    return profile in {PART3_COMPAT_PROFILE, PART3B_PROFILE}


# 関数: `is_quantum_profile` の入出力契約と処理意図を定義する。
def is_quantum_profile(profile: str) -> bool:
    return profile in {PART3_COMPAT_PROFILE, PART3A_PROFILE, PART3B_PROFILE}


# 関数: `_split_part3_sections` の入出力契約と処理意図を定義する。
def _split_part3_sections(md_text: str) -> tuple[str, list[str]]:
    lines = md_text.splitlines()
    if not lines:
        return "", []

    title_line = lines[0]
    sections: list[list[str]] = []
    current: list[str] = []
    for line in lines[1:]:
        if line.startswith("## "):
            if current:
                sections.append(current)

            current = [line]
        else:
            if current:
                current.append(line)

    if current:
        sections.append(current)

    return title_line, ["\n".join(section).strip() for section in sections if section]


# 関数: `_select_part3a_sections` の入出力契約と処理意図を定義する。
def _select_part3a_sections(sections: list[str]) -> list[str]:
    selected: list[str] = []
    for section in sections:
        heading = section.splitlines()[0].strip()
        if heading.startswith("## 要旨"):
            selected.append(section)
            continue

        if heading.startswith("## 1."):
            selected.append(section)
            continue

        if heading.startswith("## 2."):
            selected.append(section)

    return selected


# 関数: `_select_part3b_sections` の入出力契約と処理意図を定義する。
def _select_part3b_sections(sections: list[str]) -> list[str]:
    selected: list[str] = []
    for section in sections:
        heading = section.splitlines()[0].strip()
        if heading.startswith("## 3.") or heading.startswith("## 4.") or heading.startswith("## 5.") or heading.startswith("## 6.") or heading.startswith("## 7.") or heading.startswith("## 8.") or heading.startswith("## 9."):
            selected.append(section)

    return selected


# 関数: `_retitle_part3a` の入出力契約と処理意図を定義する。
def _retitle_part3a(text: str) -> str:
    out = text
    out = out.replace(
        "# 時間波ダイナミクスに基づく統一理論 (The P-model): Part III: 微視的および量子的現象の再評価",
        _PART3A_H1,
        1,
    )
    out = out.replace("本稿（Part III）は", "本稿（Part III-A）は")
    out = out.replace("本稿（Part III）", "本稿（Part III-A）")
    out = out.replace("最後に本稿（Part III）で量子領域を同型のI/Fで扱う", "最後に本稿（Part III-A）で量子基盤理論を固定し、Part III-B で量子領域を同型のI/Fで扱う")
    return out.rstrip() + _PART3A_POSTSCRIPT + "\n"


# 関数: `load_profile_markdown` の入出力契約と処理意図を定義する。
def load_profile_markdown(root: Path, profile: str, locale: str | None = None) -> str:
    md_path = resolve_manuscript_path(root, profile, locale=locale)
    return md_path.read_text(encoding="utf-8", errors="replace")
