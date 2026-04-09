"""Part III figure text localization helper.

目的:
  Part III（量子）の図テキストを、保存直前に `ja/en` で切り替える。

前提:
  - 変換対象は Matplotlib figure 上の Text artist。
  - 数式 (`$...$`) の内部は触らない。
  - source 文字列は英語寄りのまま保持し、保存時に locale を適用する。

切替:
- `WAVEP_FIGURE_LANG=ja|en`
- `WAVEP_FIGURE_LOCALE=ja|en|...`
- 未指定時は `ja`

補足:
- `WAVEP_FIGURE_LANG` は図中テキストの表示言語を切り替える。
- `WAVEP_FIGURE_LOCALE` は artifact の保存先を切り替える。
- `ja` は現行の canonical path を維持し、非 `ja` は `locales/<locale>/...`
  へ保存して日本語図を比較基準として残す。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable


_MATH_TOKEN_RE = re.compile(r"(\$[^$]*\$)")
_JAPANESE_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uff00-\uffef]")
_LOCALE_ENV_NAMES = ("WAVEP_FIGURE_LANG", "WAVEP_MPL_FIGURE_LANG")
_TRUE_VALUES = {"1", "true", "yes", "on"}

_BASE_GLOSSARY: list[tuple[str, str]] = [
    (r"\bcoincidence window half-width\b", "コインシデンス窓半幅"),
    (r"\bwindow half-width\b", "窓半幅"),
    (r"\bstart offset\b", "開始オフセット"),
    (r"\bevent-ready\b", "event-ready"),
    (r"\bpulse separation\b", "パルス間隔"),
    (r"\btilt angle\b", "傾斜角"),
    (r"\bphase shift\b", "位相差"),
    (r"\bphase scaling\b", "位相スケーリング"),
    (r"\bphase\b", "位相"),
    (r"\bgravimeter\b", "重力計"),
    (r"\batom interferometer\b", "原子干渉計"),
    (r"\boptical clock\b", "光格子時計"),
    (r"\bchronometric leveling\b", "クロノメトリック測地"),
    (r"\bgravity-induced\b", "重力誘起"),
    (r"\bdephasing\b", "位相緩和"),
    (r"\bdecoherence\b", "デコヒーレンス"),
    (r"\binterrogation time\b", "観測時間"),
    (r"\brequired\b", "必要"),
    (r"\bvisibility\b", "可視度"),
    (r"\bcontrast\b", "コントラスト"),
    (r"\bnoise budget\b", "ノイズ予算"),
    (r"\bnoise\b", "ノイズ"),
    (r"\brun-to-run\b", "run-to-run"),
    (r"\bsingle-photon\b", "単一光子"),
    (r"\bphoton gas\b", "光子気体"),
    (r"\bphoton density\b", "光子数密度"),
    (r"\bphoton flux\b", "光子フラックス"),
    (r"\bmatter wave\b", "物質波"),
    (r"\binterference\b", "干渉"),
    (r"\bprecision\b", "精度"),
    (r"\baudit\b", "監査"),
    (r"\bunified\b", "統合"),
    (r"\bconsistency\b", "整合性"),
    (r"\bcross-check\b", "クロスチェック"),
    (r"\bbaseline\b", "基準"),
    (r"\bsummary\b", "要約"),
    (r"\bsensitivity\b", "感度"),
    (r"\bselection\b", "選別"),
    (r"\bsweep\b", "掃引"),
    (r"\bwindow\b", "窓幅"),
    (r"\bpairs\b", "ペア数"),
    (r"\bpair\b", "ペア"),
    (r"\bcount\b", "件数"),
    (r"\bcounts\b", "件数"),
    (r"\bdelay\b", "遅延"),
    (r"\boffset\b", "オフセット"),
    (r"\bcovariance\b", "共分散"),
    (r"\bcorrelation\b", "相関"),
    (r"\beigenvalue\b", "固有値"),
    (r"\beigenvalues\b", "固有値"),
    (r"\bsystematics\b", "系統誤差"),
    (r"\bbudget\b", "予算"),
    (r"\bdecomposition\b", "分解"),
    (r"\blongterm\b", "長期"),
    (r"\bfalsification pack\b", "反証パック"),
    (r"\boperational I/F\b", "運用 I/F"),
    (r"\boperational\b", "運用"),
    (r"\bproxy\b", "proxy"),
    (r"\bdirect\b", "直接"),
    (r"\bglobal\b", "全体"),
    (r"\blocal\b", "局所"),
    (r"\bdifferential\b", "差分"),
    (r"\bquantification\b", "定量"),
    (r"\bpredictions\b", "予測"),
    (r"\bprediction\b", "予測"),
    (r"\bdistribution\b", "分布"),
    (r"\btargets\b", "目標"),
    (r"\btarget\b", "目標"),
    (r"\bobserved\b", "観測"),
    (r"\bpredicted\b", "予測"),
    (r"\bobservation\b", "観測"),
    (r"\bresidual\b", "残差"),
    (r"\bresiduals\b", "残差"),
    (r"\bmedian\b", "中央値"),
    (r"\btop-10 nuclei\b", "上位10核種"),
    (r"\brepresentative nuclei\b", "代表核種"),
    (r"\blight nuclei\b", "軽核"),
    (r"\ball nuclei\b", "全核種"),
    (r"\bnuclear force\b", "核力"),
    (r"\bnear-field\b", "近接場"),
    (r"\btwo-mode\b", "二モード"),
    (r"\bbinding energy\b", "束縛エネルギー"),
    (r"\bbinding scale\b", "束縛スケール"),
    (r"\bbound-state\b", "束縛状態"),
    (r"\bbound state\b", "束縛状態"),
    (r"\bbound condition\b", "束縛条件"),
    (r"\bfrequency mapping\b", "周波数写像"),
    (r"\bverification\b", "検証"),
    (r"\bminimal additional physics\b", "最小追加物理"),
    (r"\btheory-difference\b", "理論差分"),
    (r"\btheory diff\b", "理論差分"),
    (r"\bprecision requirement\b", "精度要求"),
    (r"\bHe-4 binding\b", "He-4 束縛"),
    (r"\bDeuteron binding\b", "重水素束縛"),
    (r"\bdeuteron\b", "重水素"),
    (r"\bsinglet\b", "一重項"),
    (r"\btriplet\b", "三重項"),
    (r"\bscattering length\b", "散乱長"),
    (r"\beffective range\b", "有効レンジ"),
    (r"\blow-energy parameters\b", "低エネルギー・パラメータ"),
    (r"\bsquare-well\b", "角型井戸"),
    (r"\bsquare well\b", "角型井戸"),
    (r"\bstanding-wave\b", "定常波"),
    (r"\bwell range\b", "井戸レンジ"),
    (r"\brequired depth\b", "必要深さ"),
    (r"\billustration\b", "模式図"),
    (r"\bSilicon\b", "シリコン"),
    (r"\bCopper\b", "銅"),
    (r"\bthermal expansion\b", "熱膨張"),
    (r"\bheat capacity\b", "熱容量"),
    (r"\bbulk modulus\b", "体積弾性率"),
    (r"\bresistivity\b", "抵抗率"),
    (r"\btemperature coefficient\b", "温度係数"),
    (r"\blattice\b", "格子"),
    (r"\bholdout\b", "ホールドアウト"),
    (r"\btemperature-split\b", "温度分割"),
    (r"\bphonon\b", "フォノン"),
    (r"\banchor\b", "アンカー"),
    (r"\bblackbody radiation\b", "黒体放射"),
    (r"\bblackbody\b", "黒体"),
    (r"\bentropy density\b", "エントロピー密度"),
    (r"\benergy density\b", "エネルギー密度"),
    (r"\bentropy per photon\b", "光子あたりエントロピー"),
    (r"\bentropy\b", "エントロピー"),
    (r"\btemperature\b", "温度"),
    (r"\bvacuum wavelength\b", "真空波長"),
    (r"\bmeasured / predicted\b", "観測 / 予測"),
    (r"\bmeasured\b", "観測"),
    (r"\bpred\b", "予測"),
    (r"\bHydrogen\b", "水素"),
    (r"\bHelium\b", "ヘリウム"),
    (r"\bisotopic reduced-mass scaling\b", "同位体換算質量スケーリング"),
    (r"\bdissociation\b", "解離"),
    (r"\bspectroscopic\b", "分光"),
    (r"\bQED vacuum\b", "QED 真空"),
    (r"\bVacuum \+ QED precision observables\b", "真空 + QED 精密観測量"),
    (r"\bCasimir\b", "Casimir"),
    (r"\bLamb shift\b", "Lamb シフト"),
    (r"\bscaling\b", "スケーリング"),
    (r"\bforce scale\b", "力スケール"),
    (r"\brelative scale\b", "相対スケール"),
    (r"\bnuclear-size term\b", "核サイズ項"),
    (r"\bsource coverage by record type\b", "記録種別ごとの出典カバー率"),
    (r"\bobserved vs predicted\b", "観測と予測"),
    (r"\brepresentative\b", "代表"),
    (r"\bavailable rows\b", "利用可能行"),
    (r"\bheight-equivalent\b", "高さ換算"),
    (r"\bgeopotential difference\b", "重力ポテンシャル差"),
    (r"\bcurrent\b", "現在"),
]

_STEM_SPECIFIC_GLOSSARY: dict[str, list[tuple[str, str]]] = {
    "bell_selection_sensitivity_summary": [
        (r"Weihs 1998: \|S\| vs window \([^)]+\)", "Weihs 1998: |S| と窓幅"),
        (r"\|S\| vs window", "|S| と窓幅"),
        (r"NIST: CH J_prob vs window\s*\|\s*KS\(A\)=[^|]+\|\s*KS\(B\)=[^\n]+", "NIST: CH J_prob と窓幅"),
        (r"NIST: CH J_prob vs window", "NIST: CH J_prob と窓幅"),
        (r"Delft \(event-ready\): CHSH S vs start offset", "Delft: CHSH S と開始オフセット"),
        (r"CHSH S vs start offset", "CHSH S と開始オフセット"),
        (r"Bell selection sensitivity summary", "Bell 選別感度要約"),
    ],
    "cow_phase_shift": [
        (r"COW \(gravity-induced quantum interference\): phase scaling", "COW（重力誘起量子干渉）: 位相差スケーリング"),
        (r"COW phase scaling vs tilt angle", "COW 位相スケーリングと傾斜角"),
        (r"H-v sweep:", "H-v 掃引:"),
    ],
    "optical_clock_chronometric_leveling": [
        (r"Optical clock chronometric leveling", "光格子時計クロノメトリック測地"),
    ],
    "gravity_quantum_interference_delta_predictions": [
        (r"atom-interferometer unified audit", "原子干渉計統合監査"),
        (r"P-model vs GR; Earth field", "P-model と GR; 地球重力場"),
    ],
    "matter_wave_interference_precision_audit": [
        (r"matter-wave interference precision audit", "物質波干渉精度監査"),
    ],
    "gravity_induced_decoherence": [
        (r"gravity-induced dephasing", "重力誘起位相緩和"),
        (r"time-structure", "時間構造"),
    ],
    "hom_squeezed_light_unified_audit": [
        (r"HOM squeezed-light unified audit", "HOM スクイーズド光統合監査"),
    ],
    "qed_vacuum_precision": [
        (r"Vacuum \+ QED precision observables(?:\s*\([^)]+\))?", "真空 + QED 精密観測量"),
        (r"Casimir: sphere–plate force scale", "Casimir: 球-平板スケール"),
        (r"Lamb shift: scaling\s*\(why Z>1 helps\)", "Lamb シフト: Z 依存スケール"),
        (r"nuclear-size term\s*\(example; Table 4\)", "核サイズ項（例; Table 4）"),
        (r"alpha precision cross-check", "α 精度クロスチェック"),
    ],
    "atomic_hydrogen_baseline": [
        (r"Atomic baseline \(Hydrogen, NIST ASD\)", "原子基準（水素, NIST ASD）"),
    ],
    "atomic_helium_baseline": [
        (r"Atomic baseline \(Helium, NIST ASD\)", "原子基準（ヘリウム, NIST ASD）"),
    ],
    "molecular_isotopic_scaling": [
        (r"isotopic reduced-mass scaling", "同位体換算質量スケーリング"),
    ],
    "thermo_blackbody_radiation_baseline": [
        (r"Thermo baseline", "熱力学基準"),
    ],
    "thermo_blackbody_entropy_baseline": [
        (r"Thermo baseline", "熱力学基準"),
        (r"second law", "第二法則"),
    ],
    "nuclear_binding_energy_frequency_mapping_differential_quantification": [
        (r"quantitative differential predictions and precision targets", "定量的差分予測と精度目標"),
    ],
    "nuclear_binding_energy_frequency_mapping_minimal_additional_physics": [
        (r"frozen falsification threshold", "固定反証閾値"),
    ],
    "nuclear_binding_energy_frequency_mapping_theory_diff": [
        (r"theory-difference extraction", "理論差分抽出"),
    ],
    "nuclear_near_field_interference_two_mode_model": [
        (r"nuclear force as near-field interference", "核力の近接場干渉"),
    ],
    "nuclear_binding_energy_frequency_mapping_deuteron_two_body": [
        (r"deuteron \(pn\) two-body: bound-state scales \(frozen\)", "重水素（pn）二体: 束縛状態スケール（固定）"),
        (r"Square-well example \(s-wave\):", "角型井戸の例（s波）:"),
        (
            r"This is an operational I/F for the standing-wave \(bound\) condition;",
            "これは定常波（束縛）条件の運用 I/F を示すものであり、",
        ),
        (
            r"it does not claim the nuclear force is literally a square well\.",
            "核力そのものが文字通りの角型井戸だと主張するものではない。",
        ),
        (r"Square-well depth required to support B \(illustration\)", "B を支えるために必要な角型井戸深さ（模式図）"),
        (r"well range R \(fm\)", "井戸レンジ R (fm)"),
        (r"required depth V0 \(MeV\)", "必要深さ V0 (MeV)"),
        (r"deuteron Δω mapping via 2-body boundary condition", "二体境界条件による重水素 Δω 写像"),
    ],
}


# 関数: `_ordered_patterns` の入出力契約と処理意図を定義する。
def _ordered_patterns(
    patterns: Iterable[tuple[str, str]],
    *,
    ignore_case: bool,
) -> list[tuple[re.Pattern[str], str]]:
    ordered = sorted(patterns, key=lambda item: len(item[0]), reverse=True)
    flags = re.IGNORECASE if ignore_case else 0
    return [(re.compile(pattern, flags), repl) for pattern, repl in ordered]


# 関数: `_pattern_to_display` の入出力契約と処理意図を定義する。

def _pattern_to_display(pattern: str) -> str:
    text = pattern
    replacements = (
        (r"\b", ""),
        (r"\(", "("),
        (r"\)", ")"),
        (r"\[", "["),
        (r"\]", "]"),
        (r"\+", "+"),
        (r"\-", "-"),
        (r"\|", "|"),
    )
    for src, dst in replacements:
        text = text.replace(src, dst)

    text = re.sub(r"\\([.^$*+?{}[\]|()])", r"\1", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


# 関数: `_reverse_patterns` の入出力契約と処理意図を定義する。

def _reverse_patterns(patterns: Iterable[tuple[str, str]]) -> list[tuple[re.Pattern[str], str]]:
    reverse_pairs = [(re.escape(localized), _pattern_to_display(pattern)) for pattern, localized in patterns]
    return _ordered_patterns(reverse_pairs, ignore_case=False)


_BASE_PATTERNS_JA = _ordered_patterns(_BASE_GLOSSARY, ignore_case=True)
_BASE_PATTERNS_EN = _reverse_patterns(_BASE_GLOSSARY)
_STEM_PATTERNS_JA = {
    stem: _ordered_patterns(patterns, ignore_case=True) for stem, patterns in _STEM_SPECIFIC_GLOSSARY.items()
}
_STEM_PATTERNS_EN = {stem: _reverse_patterns(patterns) for stem, patterns in _STEM_SPECIFIC_GLOSSARY.items()}
_FONT_CACHE: str | None = None


# 関数: `_reserve_suptitle_gap_if_needed` の入出力契約と処理意図を定義する。
def _reserve_suptitle_gap_if_needed(fig, *, profile_name: str) -> None:
    normalized = str(profile_name or "").strip().lower()
    if normalized not in {
        "part3_quantum",
        "part3a_quantum_foundations",
        "part3b_quantum_verification",
    }:
        return

    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is None:
        return

    try:
        title_text = str(suptitle.get_text() or "").strip()
    except Exception:
        return

    if not title_text:
        return

    axes = [ax for ax in getattr(fig, "axes", []) if getattr(ax, "get_visible", lambda: True)()]
    if not axes:
        return

    # 関数: `_union_bbox` の入出力契約と処理意図を定義する。

    def _union_bbox(items):
        valid = [bbox for bbox in items if bbox is not None]
        if not valid:
            return None

        x0 = min(float(bbox.x0) for bbox in valid)
        y0 = min(float(bbox.y0) for bbox in valid)
        x1 = max(float(bbox.x1) for bbox in valid)
        y1 = max(float(bbox.y1) for bbox in valid)
        return (x0, y0, x1, y1)

    # 関数: `_group_axes_by_row` の入出力契約と処理意図を定義する。

    def _group_axes_by_row(visible_axes):
        ordered = sorted(visible_axes, key=lambda ax: (-float(ax.get_position().y0), float(ax.get_position().x0)))
        rows: list[list[object]] = []
        tolerance = 0.03
        for ax in ordered:
            y0 = float(ax.get_position().y0)
            if rows and abs(float(rows[-1][0].get_position().y0) - y0) <= tolerance:
                rows[-1].append(ax)
            else:
                rows.append([ax])

        return rows

    if normalized == "part3b_quantum_verification":
        min_suptitle_gap = 0.075
        min_interrow_gap = 0.055
        min_top = 0.18
        max_iter = 10
    else:
        min_suptitle_gap = 0.020
        min_interrow_gap = 0.014
        min_top = 0.44
        max_iter = 4

    try:
        for _ in range(max_iter):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            suptitle_bbox = suptitle.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
            rows = _group_axes_by_row(axes)
            if not rows:
                return

            top_row_boxes = [
                ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
                for ax in rows[0]
            ]
            top_row_union = _union_bbox(top_row_boxes)
            if top_row_union is None:
                return

            top_gap = float(suptitle_bbox.y0) - float(top_row_union[3])
            need_top_delta = 0.0
            need_hspace_delta = 0.0
            if top_gap < min_suptitle_gap:
                need_top_delta = max(need_top_delta, min_suptitle_gap - top_gap + 0.018)

            for upper_row, lower_row in zip(rows, rows[1:]):
                upper_boxes = [
                    ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
                    for ax in upper_row
                ]
                upper_union = _union_bbox(upper_boxes)
                if upper_union is None:
                    continue

                lower_title_boxes = []
                for ax in lower_row:
                    try:
                        title_text = str(ax.get_title() or "").strip()
                    except Exception:
                        title_text = ""

                    if not title_text:
                        continue

                    try:
                        title_box = ax.title.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
                    except Exception:
                        continue

                    lower_title_boxes.append(title_box)

                lower_union = _union_bbox(lower_title_boxes)
                if lower_union is None:
                    continue

                inter_gap = float(upper_union[1]) - float(lower_union[3])
                if inter_gap < min_interrow_gap:
                    need_hspace_delta = max(need_hspace_delta, min_interrow_gap - inter_gap + 0.020)

            if need_top_delta <= 0.0 and need_hspace_delta <= 0.0:
                return

            subplotpars = fig.subplotpars
            avg_axes_height = sum(float(ax.get_position().height) for ax in axes) / max(len(axes), 1)
            hspace_units = 0.0
            if need_hspace_delta > 0.0 and avg_axes_height > 1e-6:
                hspace_units = need_hspace_delta / avg_axes_height

            new_top = max(min_top, float(subplotpars.top) - need_top_delta)
            new_hspace = float(subplotpars.hspace) + hspace_units
            fig.subplots_adjust(top=new_top, hspace=new_hspace)
    except Exception:
        return


# 関数: `_normalize_locale` の入出力契約と処理意図を定義する。

def _normalize_locale(raw: str | None, *, default: str = "ja") -> str:
    if raw is None or not raw.strip():
        return default

    normalized = raw.strip().lower()
    if normalized in {"ja", "jp", "japanese", "ja-jp", "ja_jp"}:
        return "ja"

    if normalized in {"en", "english", "en-us", "en_us", "en-gb", "en_gb"}:
        return "en"

    if normalized in {"source", "off", "none"}:
        return "source"

    return default


# 関数: `get_figure_language` の入出力契約と処理意図を定義する。

def get_figure_language(*, default: str = "ja") -> str:
    for env_name in _LOCALE_ENV_NAMES:
        raw = os.getenv(env_name, "").strip()
        if raw:
            return _normalize_locale(raw, default=default)

    if os.getenv("WAVEP_MPL_FORCE_JA_TEXT", "").strip().lower() in _TRUE_VALUES:
        return "ja"

    return default


# 関数: `_normalize_spacing` の入出力契約と処理意図を定義する。

def _normalize_spacing(text: str, *, target: str) -> str:
    text = re.sub(r"[ ]{2,}", " ", text)
    if target == "ja":
        text = re.sub(r"： ", "：", text)
        text = re.sub(r"\( ", "(", text)
        text = re.sub(r" \)", ")", text)
    elif target == "en":
        text = text.replace("（", "(").replace("）", ")")
        text = text.replace("：", ": ")
        text = text.replace("，", ", ").replace("、", ", ")
        text = text.replace("。", ". ")
        text = re.sub(r"\s+\.", ".", text)
        text = re.sub(r"\s+,", ",", text)

    return text.strip()


# 関数: `_target_patterns` の入出力契約と処理意図を定義する。

def _target_patterns(stem: str | None, *, target: str) -> tuple[list[tuple[re.Pattern[str], str]], list[tuple[re.Pattern[str], str]]]:
    if target == "en":
        base_patterns = _BASE_PATTERNS_EN
        stem_patterns = _STEM_PATTERNS_EN.get(stem or "", [])
        return stem_patterns, base_patterns

    if target == "ja":
        base_patterns = _BASE_PATTERNS_JA
        stem_patterns = _STEM_PATTERNS_JA.get(stem or "", [])
        return stem_patterns, base_patterns

    return [], []


# 関数: `_translate_segment` の入出力契約と処理意図を定義する。

def _translate_segment(text: str, *, stem: str | None, target: str) -> str:
    if target == "source":
        return text

    result = text
    stem_patterns, base_patterns = _target_patterns(stem, target=target)
    for pattern, repl in stem_patterns:
        result = pattern.sub(repl, result)

    for pattern, repl in base_patterns:
        result = pattern.sub(repl, result)

    return _normalize_spacing(result, target=target)


# 関数: `translate_plot_text` の入出力契約と処理意図を定義する。

def translate_plot_text(text: str, *, stem: str | None = None, target: str | None = None) -> str:
    if not text:
        return text

    active_target = _normalize_locale(target, default=get_figure_language())
    parts = _MATH_TOKEN_RE.split(text)
    translated: list[str] = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            translated.append(part)
            continue

        translated.append(_translate_segment(part, stem=stem, target=active_target))

    return "".join(translated)


# 関数: `_pick_japanese_font` の入出力契約と処理意図を定義する。

def _pick_japanese_font() -> str | None:
    global _FONT_CACHE
    if _FONT_CACHE is not None:
        return _FONT_CACHE or None

    try:
        from scripts.utils.plot_style import resolve_wavep_cjk_font_family

        preferred_name = str(os.getenv("WAVEP_MPL_CJK_FONT", "")).strip() or None
        resolved = resolve_wavep_cjk_font_family(preferred_name=preferred_name)
        if resolved:
            _FONT_CACHE = resolved
            return resolved
    except Exception:
        pass

    _FONT_CACHE = ""
    return None


# 関数: `_restore_original_font_if_needed` の入出力契約と処理意図を定義する。

def _restore_original_font_if_needed(artist: object) -> None:
    original_family = getattr(artist, "_wavep_original_fontfamily", None)
    if not original_family:
        return

    try:
        artist.set_fontfamily(original_family)
    except Exception:
        return


# 関数: `_apply_japanese_font_if_needed` の入出力契約と処理意図を定義する。

def _apply_japanese_font_if_needed(artist: object, text: str) -> None:
    if not _JAPANESE_CHAR_RE.search(text):
        _restore_original_font_if_needed(artist)
        return

    font_name = _pick_japanese_font()
    if not font_name:
        return

    try:
        artist.set_fontfamily(font_name)
    except Exception:
        return


# 関数: `_localize_figure_texts` の入出力契約と処理意図を定義する。

def _localize_figure_texts(fig: object, *, stem: str | None, target: str) -> None:
    try:
        from matplotlib.text import Text
    except Exception:
        return

    for artist in fig.findobj(lambda obj: isinstance(obj, Text)):
        try:
            current_text = artist.get_text()
        except Exception:
            continue

        if not isinstance(current_text, str) or not current_text.strip():
            continue

        original_text = getattr(artist, "_wavep_original_text", None)
        if not isinstance(original_text, str):
            original_text = current_text
            try:
                artist._wavep_original_text = current_text
            except Exception:
                pass

        original_family = getattr(artist, "_wavep_original_fontfamily", None)
        if original_family is None:
            try:
                artist._wavep_original_fontfamily = artist.get_fontfamily()
            except Exception:
                pass

        localized = translate_plot_text(original_text, stem=stem, target=target)
        if localized != current_text:
            try:
                artist.set_text(localized)
            except Exception:
                continue

        _apply_japanese_font_if_needed(artist, localized)


# 関数: `_vector_pdf_sidecar_enabled` の入出力契約と処理意図を定義する。

def _vector_pdf_sidecar_enabled() -> bool:
    raw = os.getenv("WAVEP_MPL_AUTOSAVE_VECTOR_PDF", "").strip().lower()
    return raw in _TRUE_VALUES


# 関数: `enable_figure_text_localization` の入出力契約と処理意図を定義する。

def enable_figure_text_localization(*, default_lang: str = "ja") -> None:
    try:
        from scripts.utils.plot_style import install_wavep_cjk_font_override

        preferred_name = str(os.getenv("WAVEP_MPL_CJK_FONT", "")).strip() or None
        install_wavep_cjk_font_override(preferred_name=preferred_name)
    except Exception:
        pass

    try:
        from matplotlib.figure import Figure
    except Exception:
        return

    if getattr(Figure, "_wavep_figure_text_localizer_enabled", False):
        return

    original_savefig = Figure.savefig

    # 関数: `_savefig_with_localization` の入出力契約と処理意図を定義する。
    def _savefig_with_localization(self: Figure, fname, *args, **kwargs):  # type: ignore[override]
        stem = None
        target_path = None
        try:
            target_path = Path(str(fname))
            stem = target_path.stem
        except Exception:
            stem = None

        target_lang = get_figure_language(default=default_lang)
        _localize_figure_texts(self, stem=stem, target=target_lang)
        localized_target = None
        if target_path is not None:
            try:
                from scripts.utils.figure_locale_paths import localize_figure_output_path

                localized_target = localize_figure_output_path(target_path)
                localized_target.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                localized_target = target_path

        save_kwargs = dict(kwargs)
        profile_name = str(os.getenv("WAVEP_MPL_FONT_PROFILE", "")).strip()
        try:
            from scripts.utils.plot_style import (
                normalize_wavep_export_canvas_to_textwidth,
                should_wavep_export_normalize_to_textwidth,
            )
        except Exception:
            normalize_wavep_export_canvas_to_textwidth = None
            should_wavep_export_normalize_to_textwidth = None

        disable_canvas_normalize = str(os.getenv("WAVEP_MPL_DISABLE_CANVAS_NORMALIZE", "")).strip().lower()
        if (
            target_path is not None
            and disable_canvas_normalize not in {"1", "true", "yes", "on"}
            and should_wavep_export_normalize_to_textwidth is not None
            and should_wavep_export_normalize_to_textwidth(profile_name=profile_name)
        ):
            _reserve_suptitle_gap_if_needed(self, profile_name=profile_name)
            if str(save_kwargs.get("bbox_inches", "")).strip().lower() == "tight":
                save_kwargs.pop("bbox_inches", None)

            if normalize_wavep_export_canvas_to_textwidth is not None:
                normalize_wavep_export_canvas_to_textwidth(self, profile_name=profile_name)

        actual_target = localized_target if localized_target is not None else fname
        result = original_savefig(self, actual_target, *args, **save_kwargs)

        if localized_target is not None and localized_target.suffix.lower() == ".png" and not _vector_pdf_sidecar_enabled():
            pdf_target = localized_target.with_suffix(".pdf")
            pdf_kwargs = dict(save_kwargs)
            pdf_kwargs["format"] = "pdf"
            pdf_kwargs.pop("dpi", None)
            original_savefig(self, pdf_target, *args, **pdf_kwargs)

        return result

    Figure.savefig = _savefig_with_localization  # type: ignore[assignment]
    Figure._wavep_figure_text_localizer_enabled = True


# 関数: `enable_japanese_figure_localization` の入出力契約と処理意図を定義する。

def enable_japanese_figure_localization() -> None:
    enable_figure_text_localization(default_lang="ja")
