"""
sitecustomize.py

環境変数で指定された Matplotlib 共通フックを、build 時に自動適用する。

- `WAVEP_MPL_FONT_PROFILE`: 役割別 font profile（title / axis / tick / legend / note / suptitle）
- `WAVEP_MPL_FONT_SCALE`: profile 全体の倍率
- `WAVEP_MPL_CJK_FONT`: 図用の日本語 sans-serif font target
- `WAVEP_MPL_CJK_FONT_PATH`: 図用 font file path の明示 override
- `WAVEP_MPL_LEGEND_NOTE_MIN_FONT`: 凡例・注記の後方互換 floor
- `WAVEP_MPL_TEXT_MIN_FONT`: 文字全体の後方互換 floor
- `WAVEP_FIGURE_LANG`: 図中テキストの表示言語（ja/en）
"""

from __future__ import annotations

import os
import sys
import re
from pathlib import Path
from typing import Any

_PMODEL_TEXTWIDTH_MM = 170.0
_PMODEL_MM_PER_INCH = 25.4
_PMODEL_TEXTWIDTH_IN = _PMODEL_TEXTWIDTH_MM / _PMODEL_MM_PER_INCH
_WAVEP_PART2_CANVAS_HEIGHT_SENTINEL_IN = 10_000.0
_JAPANESE_CHAR_RE = re.compile(r"[ぁ-ゖァ-ヺ一-龯々〆〤]")


# 関数: `_apply_wavep_font_profile_if_enabled` の入出力契約と処理意図を定義する。
def _apply_wavep_font_profile_if_enabled() -> None:
    raw = os.getenv("WAVEP_MPL_FONT_PROFILE", "").strip()
    if not raw:
        return

    scale_raw = os.getenv("WAVEP_MPL_FONT_SCALE", "1.0").strip()
    try:
        scale = float(scale_raw)
    except Exception:
        scale = 1.0

    try:
        from scripts.utils.plot_style import install_wavep_font_profile

        install_wavep_font_profile(profile_name=raw, scale=scale)
    except Exception:
        # sitecustomize では描画不能より build 継続を優先する。
        return


_apply_wavep_font_profile_if_enabled()


# 関数: `_apply_wavep_cjk_font_override_if_enabled` の入出力契約と処理意図を定義する。
def _apply_wavep_cjk_font_override_if_enabled() -> None:
    raw = os.getenv("WAVEP_MPL_CJK_FONT", "").strip()
    if not raw:
        return

    try:
        import matplotlib as mpl
        from scripts.utils.plot_style import install_wavep_cjk_font_override
    except Exception:
        return

    target_family = install_wavep_cjk_font_override(preferred_name=raw)
    if not target_family:
        return

    mpl.rcParams["font.family"] = [target_family, "DejaVu Sans"]
    mpl.rcParams["font.sans-serif"] = [target_family, "DejaVu Sans"]


_apply_wavep_cjk_font_override_if_enabled()


# 関数: rcParams への古い日本語フォント代入を Noto 系へ正規化する。
def _apply_wavep_rcparams_font_rewrite_if_enabled() -> None:
    raw = os.getenv("WAVEP_MPL_CJK_FONT", "").strip()
    if not raw:
        return

    try:
        import matplotlib as mpl
        from scripts.utils.plot_style import resolve_wavep_cjk_font_family
    except Exception:
        return

    target_family = resolve_wavep_cjk_font_family(preferred_name=raw)
    if not target_family:
        return

    alias_names = {
        "sans-serif",
        "noto sans cjk jp",
        "noto sans jp",
        "source han sans",
        "source han sans jp",
        "yu gothic",
        "yu gothic ui",
        "yu mincho",
        "meiryo",
        "biz udgothic",
        "ms gothic",
        "ms mincho",
        "ipaexgothic",
    }

    # 関数: `_normalize_family_value` の入出力契約と処理意図を定義する。
    def _normalize_family_value(value: Any) -> list[str]:
        if isinstance(value, str):
            items = [value]
        else:
            try:
                items = [str(item) for item in value]
            except Exception:
                items = [str(value)]

        lowered = {str(item).strip().lower() for item in items if str(item).strip()}
        if lowered & alias_names:
            return [target_family, "DejaVu Sans"]

        return [str(item) for item in items if str(item).strip()] or [target_family, "DejaVu Sans"]

    rcparams_type = type(mpl.rcParams)
    if getattr(rcparams_type, "_wavep_font_rewrite_patched", False):
        return

    original_setitem = rcparams_type.__setitem__

    # 関数: font.family / font.sans-serif への legacy 代入を Noto 優先へ寄せる。
    def patched_setitem(self, key, value):
        if key in {"font.family", "font.sans-serif"}:
            value = _normalize_family_value(value)

        return original_setitem(self, key, value)

    rcparams_type.__setitem__ = patched_setitem
    rcparams_type._wavep_font_rewrite_patched = True
    mpl.rcParams["font.family"] = [target_family, "DejaVu Sans"]
    mpl.rcParams["font.sans-serif"] = [target_family, "DejaVu Sans"]


_apply_wavep_rcparams_font_rewrite_if_enabled()


# 関数: `_apply_wavep_font_floor_if_enabled` の入出力契約と処理意図を定義する。
def _apply_wavep_font_floor_if_enabled() -> None:
    raw = os.getenv("WAVEP_MPL_LEGEND_NOTE_MIN_FONT", "").strip()
    if not raw:
        return

    try:
        floor = float(raw)
    except Exception:
        return

    try:
        from scripts.utils.plot_style import install_legend_note_font_floor

        install_legend_note_font_floor(min_fontsize=floor)
    except Exception:
        # 環境依存で Matplotlib 未導入の処理を壊さないため、sitecustomize では黙って無効化する。
        return


_apply_wavep_font_floor_if_enabled()


_GLOBAL_TEXT_FLOOR_PATCHED = False
_WAVEP_COPY2_PATCHED = False


# 関数: `_coerce_fontsize_number` の入出力契約と処理意図を定義する。
def _coerce_fontsize_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)

    try:
        return float(str(value).strip())
    except Exception:
        return None


# 関数: `_apply_wavep_global_text_floor_if_enabled` の入出力契約と処理意図を定義する。

def _apply_wavep_global_text_floor_if_enabled() -> None:
    """
    WAVEP_MPL_TEXT_MIN_FONT が設定されている場合、
    図全体のテキスト（軸ラベル・目盛・凡例・注記など）に最小フォントを適用する。
    """
    raw = os.getenv("WAVEP_MPL_TEXT_MIN_FONT", "").strip()
    if not raw:
        return

    try:
        floor = float(raw)
    except Exception:
        return

    try:
        import matplotlib as mpl
        from matplotlib.text import Text
    except Exception:
        return

    # rcParams 側の既定値を先に引き上げる（明示未指定の文字へ効かせる）。

    for key in (
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
        "figure.titlesize",
    ):
        current = _coerce_fontsize_number(mpl.rcParams.get(key))
        if current is not None and current < floor:
            mpl.rcParams[key] = floor

    global _GLOBAL_TEXT_FLOOR_PATCHED
    if _GLOBAL_TEXT_FLOOR_PATCHED:
        return

    original_set_fontsize = Text.set_fontsize

    # 関数: `patched_set_fontsize` の入出力契約と処理意図を定義する。
    def patched_set_fontsize(self, fontsize):
        numeric = _coerce_fontsize_number(fontsize)
        if numeric is not None and numeric < floor:
            fontsize = floor

        return original_set_fontsize(self, fontsize)

    Text.set_fontsize = patched_set_fontsize
    _GLOBAL_TEXT_FLOOR_PATCHED = True


_apply_wavep_global_text_floor_if_enabled()


_VECTOR_PDF_AUTOSAVE_PATCHED = False
_VECTOR_PDF_AUTOSAVE_IN_PROGRESS = False


# 関数: `_translate_wavep_text_to_english` の入出力契約と処理意図を定義する。
def _translate_wavep_text_to_english(text: str, *, stem: str | None = None) -> str:
    """
    非 ja locale の図で、日本語ラベルを英語中心へ戻す。

    方針:
    - 数式は触らない。
    - Part II の reader-facing 図で頻出する phrase を優先して変換する。
    - 完全翻訳ではなく、論文図として読める英語へ寄せる。
    """
    if not text:
        return text

    stripped = text.strip()
    if stripped.startswith("$") and stripped.endswith("$"):
        return text

    if not _JAPANESE_CHAR_RE.search(text):
        return text

    out = text
    phrase_map: list[tuple[str, str]] = [
        ("検証スコアボード", "Validation scoreboard"),
        ("検証サマリ表", "verification summary table"),
        ("月レーザー測距（LLR: Lunar Laser Ranging）", "Lunar Laser Ranging (LLR)"),
        ("月レーザー測距", "Lunar Laser Ranging"),
        ("観測 TOF（平均除去）", "Observed TOF (mean removed)"),
        ("月中心モデル", "Moon-center model"),
        ("反射器モデル", "Reflector model"),
        ("定数オフセット整列後の残差", "Residual after constant-offset alignment"),
        ("定数オフセット整列", "constant-offset aligned"),
        ("平均除去", "mean removed"),
        ("太陽Shapiro含む", "including solar Shapiro"),
        ("往復TOF偏差 [ns]", "Round-trip TOF deviation [ns]"),
        ("UTC時刻", "UTC time"),
        ("観測 vs モデル", "Observed vs model"),
        ("残差（観測 - モデル）[ns]", "Residual (observed - model) [ns]"),
        ("残差 [ns]（定数オフセット整列後）", "Residual [ns] (after constant-offset alignment)"),
        ("モデル改善（地球中心 → 観測局 → 反射器）", "Model improvement (geocenter → station → reflector)"),
        ("地球中心→月中心", "Geocenter → moon center"),
        ("観測局→月中心", "Station → moon center"),
        ("観測局→反射器", "Station → reflector"),
        ("重み付き RMSE [ps]", "Weighted RMSE [ps]"),
        ("重み付き RMSE", "Weighted RMSE"),
        ("観測（PDS TDF 処理後）", "Observed (processed PDS TDF)"),
        ("観測（リング直径 θ_ring）", "Observed (ring diameter θ_ring)"),
        ("参考（推定影直径 d_sh；リングからの推定）", "Reference (estimated shadow diameter d_sh; inferred from ring)"),
        ("角直径 [µas]", "Angular diameter [µas]"),
        ("差（P-model − GR）[μas]", "Difference (P-model − GR) [μas]"),
        ("必要な観測誤差（1σ）[μas]", "Required observational error (1σ) [μas]"),
        ("3σで判別するための必要精度", "Required precision for 3σ discrimination"),
        ("係数の比較（βとスピン依存）", "Coefficient comparison (β and spin dependence)"),
        ("リング≒シャドウ近似の系統誤差（κ）", "Systematic uncertainty of the ring≈shadow approximation (κ)"),
        ("リングの幅（fractional width）", "Ring width (fractional width)"),
        ("リングの非対称性（brightness asymmetry）", "Ring asymmetry (brightness asymmetry)"),
        ("直径スケール：観測リングと散乱blur（参考）", "Diameter scale: observed ring and scattering blur (reference)"),
        ("形状スケール：リング幅（W/d→µas）と散乱blur（参考）", "Shape scale: ring width (W/d→µas) and scattering blur (reference)"),
        ("EHT：屈折散乱のゆらぎスケール（κ系統の一要因）と必要精度（参考）", "EHT: refractive-scattering fluctuation scales and required precision (reference)"),
        ("EHT：観測とモデルのずれ（zスコア, κ=1）", "EHT: observed-model offsets (z score, κ=1)"),
        ("EHT：κ（リング/シャドウ）— κ_fit（モデル）と κ_ref(d_sh)", "EHT: κ (ring/shadow) — κ_fit (model) and κ_ref(d_sh)"),
        ("EHT：κ（リング/シャドウ変換）を何%まで詰める必要があるか（3σ判別の入口）", "EHT: required κ precision (ring/shadow conversion) for 3σ discrimination"),
        ("EHT：δ（Schwarzschild shadow deviation）の必要精度（参考; δはモデル依存）", "EHT: required precision for δ (Schwarzschild shadow deviation; reference)"),
        ("GPS 時計残差: G01（観測 IGS に対する比較）", "GPS clock residuals: G01 (comparison against observed IGS)"),
        ("残差 [ns]（バイアス＋ドリフト除去後）", "Residual [ns] (after bias+drift removal)"),
        ("GPS: 観測 IGS に対する残差 RMS（全衛星）", "GPS: residual RMS against observed IGS (all satellites)"),
        ("RMS [ns]（バイアス＋ドリフト除去後）", "RMS [ns] (after bias+drift removal)"),
        ("衛星PRN", "Satellite PRN"),
        ("GPS: 相対補正（近日点効果）", "GPS: relativistic correction (perigee effect)"),
        ("時間補正 [ns]（バイアス＋ドリフト除去後）", "Time correction [ns] (after bias+drift removal)"),
        ("放送暦（BRDC）- IGS", "Broadcast ephemeris (BRDC) - IGS"),
        ("P-model（dt_rel除去）- IGS", "P-model (with dt_rel removed) - IGS"),
        ("標準式 δt_rel（-2 r·v / c^2）", "Standard formula δt_rel (-2 r·v / c^2)"),
        ("太陽光偏向", "Solar light deflection"),
        ("重力赤方偏移", "Gravitational redshift"),
        ("連星パルサー", "Binary pulsar"),
        ("軌道減衰", "Orbital decay"),
        ("重力波（GW150914 等）", "Gravitational waves (GW150914 etc.)"),
        ("重力波偏光モード（H1/L1/V1）", "GW polarization modes (H1/L1/V1)"),
        ("ブラックホール影", "Black-hole shadow"),
        ("銀河回転曲線（SPARC）", "Galaxy rotation curves (SPARC)"),
        ("XRISM（公開一次データ）", "XRISM (public primary data)"),
        ("背景計量（caseB: 有効計量）", "Background metric (case B: effective metric)"),
        ("背景計量（caseA: 平坦背景）", "Background metric (case A: flat background)"),
        ("純スカラー極限（回転なし）", "Pure-scalar limit (no rotation)"),
        ("速度飽和 δ（理論）", "Velocity saturation δ (theory)"),
        ("回転（フレームドラッグ）", "Rotation (frame dragging)"),
        ("観測", "Observed"),
        ("予測", "Predicted"),
        ("参照", "Reference"),
        ("理論", "Theory"),
        ("比較", "Comparison"),
        ("残差分布", "Residual distribution"),
        ("残差", "Residual"),
        ("分布", "Distribution"),
        ("要約", "Summary"),
        ("監査", "Audit"),
        ("感度", "Sensitivity"),
        ("精度", "Precision"),
        ("必要", "Required"),
        ("誤差", "Error"),
        ("相対不確かさ", "Relative uncertainty"),
        ("角スケール", "Angular scale"),
        ("ターゲット", "Target"),
        ("相対補正", "Relativistic correction"),
        ("観測リング直径の統計誤差 σ_obs", "Statistical error σ_obs of the observed ring diameter"),
        ("3σ判別に必要なσ_obs（理想）", "Required σ_obs for 3σ discrimination (ideal)"),
        ("散乱カーネル係数の不確かさ（1σ, 長軸/短軸）", "Scattering-kernel coefficient uncertainty (1σ, major/minor axes)"),
        ("屈折散乱 wander（min–max, Zhu 2018; 230 GHz）", "Refractive-scattering wander (min–max, Zhu 2018; 230 GHz)"),
        ("屈折散乱 distortion（min–max, Zhu 2018; 230 GHz）", "Refractive-scattering distortion (min–max, Zhu 2018; 230 GHz)"),
        ("屈折散乱 asymmetry（min–max, Zhu 2018; 230 GHz）", "Refractive-scattering asymmetry (min–max, Zhu 2018; 230 GHz)"),
        ("相対不確かさ（%）", "Relative uncertainty (%)"),
        ("観測リングと散乱blur", "observed ring and scattering blur"),
        ("散乱blur", "scattering blur"),
        ("リング直径の相対誤差（1σ, %）", "Relative ring-diameter error (1σ, %)"),
        ("許容 κ の相対誤差（1σ, %）", "Allowed relative κ error (1σ, %)"),
        ("現状 ring σ/diameter", "Current ring σ/diameter"),
        ("参考: Kerr κ系統（constrained）", "Reference: Kerr κ systematics (constrained)"),
        ("参考: Kerr κ系統（full）", "Reference: Kerr κ systematics (full)"),
        ("光偏向（太陽）", "Light deflection (Sun)"),
        ("GPS（衛星時計）", "GPS (satellite clocks)"),
        ("Cassini（太陽会合）", "Cassini (solar conjunction)"),
        ("Viking（太陽会合）", "Viking (solar conjunction)"),
        ("Mercury（近日点移動）", "Mercury (perihelion precession)"),
        ("GP-A / Galileo（重力赤方偏移）", "GP-A / Galileo (gravitational redshift)"),
    ]

    phrase_map.extend(
        [
            ("正規化スコア（0=理想, 1=OK境界, 2=要改善境界）", "Normalized score (0=ideal, 1=pass boundary, 2=improvement boundary)"),
            ("宇宙論（距離二重性）", "Cosmology (distance duality)"),
            ("宇宙論（銀河団衝突オフセット）", "Cosmology (cluster-collision offset)"),
            ("宇宙論（Pantheon+", "Cosmology (Pantheon+"),
            ("Hubble図）", "Hubble diagram)"),
            ("強場（高次項同時拘束）", "Strong field (simultaneous higher-order constraints)"),
            ("太陽会合", "solar conjunction"),
            ("ブラックホール影", "black-hole shadow"),
            ("連星パルサー", "binary pulsar"),
            ("時刻", "Time"),
            ("往復TOF偏差", "Round-trip TOF deviation"),
            ("観測 IGS に対する比較", "comparison against observed IGS"),
            ("時計Residual", "Clock residual"),
            ("時計残差", "Clock residual"),
            ("Observedのずれ", "observed deviation"),
            ("一次ソース", "primary source"),
            ("公表値", "published value"),
            ("インパクトパラメータ", "Impact parameter"),
            ("偏向角", "Deflection angle"),
            ("太陽重力による光の偏向(弱場)", "Light deflection by solar gravity (weak field)"),
            ("代表: VLBI", "Representative: VLBI"),
            ("光偏向パラメータ", "Light-deflection parameter"),
            ("値", "value"),
            ("シミュレーション", "simulation"),
            ("文献代表値", "literature reference"),
            ("マイクロ秒", "microseconds"),
            ("バイキング 太陽合", "Viking solar conjunction"),
            ("リングと, 影直径", "ring and shadow diameter"),
            ("モデル；", "model; "),
            ("リング", "ring"),
            ("影直径", "shadow diameter"),
            ("標準Theory", "Standard theory"),
            ("多時期", "multi-epoch"),
            ("整合", "consistency"),
            ("リング直径", "ring diameter"),
            ("シャドウ直径", "shadow diameter"),
            ("目安", "guide"),
            ("系統誤差", "systematic error"),
            ("参考レンジ", "reference range"),
            ("係数レンジ", "coefficient range"),
            ("目標", "target"),
            ("線重心からの", "from line centroids"),
            ("銀河団", "cluster"),
            ("X線", "X-ray"),
            ("内縁半径拘束", "inner-edge radius constraint"),
            ("全点", "All points"),
            ("適合度", "Goodness of fit"),
            ("単一", "single"),
            ("周波数トラック", "Frequency track"),
            ("除外点", "Excluded points"),
            ("抽出", "Extracted"),
            ("直線化", "Linearized"),
            ("採用点", "Retained points"),
            ("一致度", "Agreement"),
            ("複数イベント", "multi-event"),
            ("外れ値ID", "Outlier ID"),
            ("外れ値の", "Outlier"),
            ("閾値", "threshold"),
            ("混入判定", "mixing criterion"),
            ("現在Target", "Current target"),
            ("推定Target", "Estimated target"),
            ("感度", "sensitivity"),
            ("複数イベント):", "multi-event):"),
            ("複数イベント", "multiple events"),
            ("速度飽和", "velocity saturation"),
            ("反証条件パック", "falsification pack"),
            ("判別", "discrimination"),
            ("必要精度", "required precision"),
            ("現在の精度", "current precision"),
            ("将来の精度", "future precision"),
            ("銀河回転曲線", "galaxy rotation curves"),
            ("観測配置", "observational configuration"),
            ("後付け調整", "post-hoc fitting"),
            ("比較基準", "comparison baseline"),
            ("比較", "comparison"),
            ("図", "diagram"),
        ]
    )

    for src, dst in phrase_map:
        out = out.replace(src, dst)

    generic_word_map: list[tuple[str, str]] = [
        ("局あり", "with station"),
        ("局+月回転", "station + lunar rotation"),
        ("月回転", "lunar rotation"),
        ("モデル", "model"),
        ("公開一次データ", "public primary data"),
        ("有効計量", "effective metric"),
        ("平坦背景", "flat background"),
        ("背景計量", "background metric"),
        ("純スカラー極限", "pure-scalar limit"),
        ("回転なし", "no rotation"),
        ("横断-チャネル", "cross-channel"),
        ("初期熱史", "early thermal history"),
        ("構造形成", "structure formation"),
        ("音響ピーク", "acoustic peaks"),
        ("衛星時計", "satellite clocks"),
        ("フレームドラッグ", "frame dragging"),
        ("太陽会合", "solar conjunction"),
        ("銀河回転曲線", "galaxy rotation curves"),
        ("銀河団衝突オフセット", "cluster-collision offset"),
        ("距離二重性", "distance duality"),
        ("高次項同時拘束", "simultaneous higher-order constraints"),
        ("回転", "rotation"),
        ("公開", "public"),
        ("現在", "current"),
        ("推定", "estimated"),
        ("由来", "induced"),
        ("観測局", "station"),
        ("反射器", "reflector"),
        ("月中心", "moon center"),
        ("光学", "optical"),
        ("published value", "published value"),
        ("時刻", "Time"),
        ("往復", "Round-trip"),
        ("偏差", "deviation"),
        ("周波数比", "frequency ratio"),
        ("ドップラー", "Doppler"),
        ("全区間", "full interval"),
        ("最適", "best-fit"),
        ("日付", "date"),
        ("残差", "Residual"),
        ("累積割合", "Cumulative fraction"),
        ("件数", "Count"),
        ("差", "difference"),
        ("小さいほど良い", "smaller is better"),
        ("後期", "late-time"),
        ("処理後", "processed"),
        ("一次データ", "primary data"),
        ("論文図", "paper figure"),
        ("デジタイズ", "digitized"),
        ("太陽重力による光の偏向", "Light deflection by solar gravity"),
        ("太陽縁", "solar limb"),
        ("角秒", "arcsec"),
        ("偏向角", "deflection angle"),
        ("近日点効果", "perigee effect"),
        ("衛星", "satellite"),
        ("時計", "clock"),
        ("比較", "comparison"),
        ("リング直径", "ring diameter"),
        ("リング", "ring"),
        ("シャドウ", "shadow"),
        ("影直径", "shadow diameter"),
        ("係数", "coefficient"),
        ("系統", "systematic"),
        ("誤差", "error"),
        ("許容域", "allowed region"),
        ("下側", "lower side"),
        ("判別", "discrimination"),
        ("現状", "current"),
        ("将来", "future"),
        ("最良ケース", "best case"),
        ("散乱", "scattering"),
        ("屈折", "refractive"),
        ("幅", "width"),
        ("非対称性", "asymmetry"),
        ("多時期", "multi-epoch"),
        ("整合", "consistency"),
        ("線重心", "line centroid"),
        ("銀河団", "cluster"),
        ("内縁半径拘束", "inner-edge radius constraint"),
        ("全点", "All points"),
        ("適合度", "Goodness of fit"),
        ("周波数トラック", "Frequency track"),
        ("除外点", "Excluded points"),
        ("抽出", "Extracted"),
        ("直線化", "Linearized"),
        ("採用点", "Retained points"),
        ("一致度", "Agreement"),
        ("複数イベント", "multi-event"),
        ("外れ値", "Outlier"),
        ("閾値", "threshold"),
        ("混入", "mixing"),
        ("現在Target", "Current target"),
        ("推定Target", "Estimated target"),
        ("反証条件パック", "falsification pack"),
        ("速度飽和", "velocity saturation"),
        ("必要精度", "required precision"),
        ("観測配置", "observational configuration"),
        ("ための", "for"),
        ("での", "at"),
        ("による", "from"),
        ("のみ", "only"),
        ("からの", "from"),
        ("を", " "),
        ("が", " "),
        ("と", " and "),
        ("の", " "),
        ("図", "diagram"),
    ]
    for src, dst in generic_word_map:
        out = out.replace(src, dst)

    char_map = {
        "（": "(",
        "）": ")",
        "：": ": ",
        "、": ", ",
        "　": " ",
        "〜": "~",
    }
    for src, dst in char_map.items():
        out = out.replace(src, dst)

    out = re.sub(r"\s+", " ", out).strip()
    return out


# 関数: `_localize_wavep_figure_texts_for_locale` の入出力契約と処理意図を定義する。
def _localize_wavep_figure_texts_for_locale(figure: Any, *, stem: str | None, target_lang: str) -> None:
    """
    保存直前の Matplotlib figure を locale 別に正規化する。
    現在は en だけを追加サポートし、ja は source/canonical を維持する。
    """
    if target_lang != "en":
        return

    try:
        texts = list(figure.findobj(lambda artist: hasattr(artist, "get_text") and hasattr(artist, "set_text")))
    except Exception:
        return

    for artist in texts:
        try:
            raw = artist.get_text()
        except Exception:
            continue

        if not isinstance(raw, str) or not raw.strip():
            continue

        translated = _translate_wavep_text_to_english(raw, stem=stem)
        if translated == raw:
            continue

        try:
            artist.set_text(translated)
        except Exception:
            continue


# 関数: `_patch_wavep_copy2_for_localized_figures` の入出力契約と処理意図を定義する。
def _patch_wavep_copy2_for_localized_figures() -> None:
    global _WAVEP_COPY2_PATCHED
    if _WAVEP_COPY2_PATCHED:
        return

    try:
        import shutil
        from scripts.utils.figure_locale_paths import localize_figure_output_path, is_default_figure_locale
    except Exception:
        return

    if is_default_figure_locale():
        return

    original_copy2 = shutil.copy2
    figure_suffixes = {".pdf", ".png", ".jpg", ".jpeg"}

    # 関数: `patched_copy2` の入出力契約と処理意図を定義する。
    def patched_copy2(src, dst, *args, **kwargs):
        try:
            src_path = Path(src)
            dst_path = Path(dst)
        except Exception:
            return original_copy2(src, dst, *args, **kwargs)

        if src_path.suffix.lower() not in figure_suffixes and dst_path.suffix.lower() not in figure_suffixes:
            return original_copy2(src, dst, *args, **kwargs)

        localized_src = localize_figure_output_path(src_path)
        localized_dst = localize_figure_output_path(dst_path)
        actual_src = localized_src if localized_src.exists() else src_path
        try:
            localized_dst.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        return original_copy2(actual_src, localized_dst, *args, **kwargs)

    shutil.copy2 = patched_copy2
    _WAVEP_COPY2_PATCHED = True


# 関数: `_resolve_wavep_canonical_canvas_box` の入出力契約と処理意図を定義する。
def _resolve_wavep_canonical_canvas_box(width_in: float, height_in: float) -> tuple[float, float] | None:
    """
    build profile ごとの figure canvas 正規化 box を返す。

    Part II は TeX 側を width 基準で組むため、source 側も 170 mm 幅へ
    そろえてから保存する。高さは aspect ratio を維持するため sentinel を使う。
    """
    disable_normalize = str(os.getenv("WAVEP_MPL_DISABLE_CANVAS_NORMALIZE", "")).strip().lower()
    if disable_normalize in {"1", "true", "yes", "on"}:
        return None

    profile = str(os.getenv("WAVEP_MPL_FONT_PROFILE", "")).strip().lower()
    if profile == "part2_astrophysics":
        return (_PMODEL_TEXTWIDTH_IN, _WAVEP_PART2_CANVAS_HEIGHT_SENTINEL_IN)

    return None


# 関数: `_normalize_wavep_figure_canvas_for_profile` の入出力契約と処理意図を定義する。

def _normalize_wavep_figure_canvas_for_profile(figure: Any) -> tuple[float, float] | None:
    """
    現在 profile に応じて figure canvas を等倍縮尺で正規化する。

    戻り値は復元用の元サイズ（inch）。正規化不要なら None を返す。
    """
    try:
        width_in, height_in = figure.get_size_inches()
    except Exception:
        return None

    target_box = _resolve_wavep_canonical_canvas_box(float(width_in), float(height_in))
    if target_box is None:
        return None

    target_width, target_height = target_box
    scale = min(target_width / float(width_in), target_height / float(height_in))
    if scale <= 0.0 or abs(scale - 1.0) < 0.03:
        return None

    original_size = (float(width_in), float(height_in))
    try:
        figure.set_size_inches(width_in * scale, height_in * scale, forward=True)
    except Exception:
        return None

    try:
        figure.tight_layout()
    except Exception:
        pass

    return original_size


# 関数: `_enable_vector_pdf_sidecar_if_enabled` の入出力契約と処理意図を定義する。

def _enable_vector_pdf_sidecar_if_enabled() -> None:
    """
    `Figure.savefig` に共通 hook を入れる。

    役割:
    - 非 `ja` locale の図 artifact を `locales/<locale>/...` へ退避する
    - WAVEP_MPL_AUTOSAVE_VECTOR_PDF=1 のときだけ `*.png/*.jpg` 保存時に
      同名 `*.pdf` を追加で保存する
    """
    raw = os.getenv("WAVEP_MPL_AUTOSAVE_VECTOR_PDF", "").strip().lower()
    autosave_pdf_enabled = raw in {"1", "true", "yes", "on"}

    global _VECTOR_PDF_AUTOSAVE_PATCHED
    if _VECTOR_PDF_AUTOSAVE_PATCHED:
        return

    try:
        from matplotlib.figure import Figure
        from scripts.utils.figure_locale_paths import localize_figure_output_path
    except Exception:
        return

    original_savefig = Figure.savefig
    debug_enabled = os.getenv("WAVEP_MPL_AUTOSAVE_VECTOR_PDF_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}

    # 関数: `_extract_fname` の入出力契約と処理意図を定義する。
    def _extract_fname(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
        if args:
            candidate = args[0]
        else:
            candidate = kwargs.get("fname")

        if candidate is None:
            return None

        # BytesIO などの file-like object は path ではないので、locale rewrite 対象から外す。
        if hasattr(candidate, "write") or hasattr(candidate, "read"):
            return None

        if not isinstance(candidate, (str, os.PathLike, Path)):
            return None

        try:
            return str(candidate)
        except Exception:
            return None

    # 関数: `_rewrite_save_target` の入出力契約と処理意図を定義する。
    def _rewrite_save_target(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any], str | None]:
        fname = _extract_fname(args, kwargs)
        if not fname:
            return args, kwargs, None

        localized_target = localize_figure_output_path(Path(fname))
        try:
            localized_target.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        if args:
            new_args = (localized_target, *args[1:])
            return new_args, kwargs, str(localized_target)

        new_kwargs = dict(kwargs)
        new_kwargs["fname"] = localized_target
        return args, new_kwargs, str(localized_target)

    # 関数: `patched_savefig` の入出力契約と処理意図を定義する。

    def patched_savefig(self, *args, **kwargs):
        global _VECTOR_PDF_AUTOSAVE_IN_PROGRESS
        save_args, save_kwargs, resolved_fname = _rewrite_save_target(args, kwargs)
        target_lang = str(os.getenv("WAVEP_FIGURE_LANG", "")).strip().lower()
        stem = None
        if resolved_fname:
            try:
                stem = Path(resolved_fname).stem
            except Exception:
                stem = None

        _localize_wavep_figure_texts_for_locale(self, stem=stem, target_lang=target_lang)
        original_size = _normalize_wavep_figure_canvas_for_profile(self)
        try:
            result = original_savefig(self, *save_args, **save_kwargs)
        finally:
            if original_size is not None:
                try:
                    self.set_size_inches(*original_size, forward=True)
                except Exception:
                    pass

        if _VECTOR_PDF_AUTOSAVE_IN_PROGRESS:
            return result

        fname = resolved_fname
        if not fname:
            return result

        if not autosave_pdf_enabled:
            return result

        suffix = Path(fname).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg"}:
            return result

        pdf_path = str(Path(fname).with_suffix(".pdf"))
        pdf_kwargs = dict(save_kwargs)
        pdf_kwargs["format"] = "pdf"
        # PDFでは dpi 指定は不要なので削除する。
        pdf_kwargs.pop("dpi", None)

        _VECTOR_PDF_AUTOSAVE_IN_PROGRESS = True
        try:
            normalized_pdf_size = _normalize_wavep_figure_canvas_for_profile(self)
            original_savefig(self, pdf_path, **pdf_kwargs)
        except Exception:
            if debug_enabled:
                print(f"[autosave-pdf] failed: src={fname} dst={pdf_path}", file=sys.stderr)
                print(f"[autosave-pdf] kwargs={pdf_kwargs}", file=sys.stderr)
                print(f"[autosave-pdf] error={repr(sys.exc_info()[1])}", file=sys.stderr)
        finally:
            if "normalized_pdf_size" in locals() and normalized_pdf_size is not None:
                try:
                    self.set_size_inches(*normalized_pdf_size, forward=True)
                except Exception:
                    pass

            _VECTOR_PDF_AUTOSAVE_IN_PROGRESS = False

        return result

    Figure.savefig = patched_savefig
    _VECTOR_PDF_AUTOSAVE_PATCHED = True


_enable_vector_pdf_sidecar_if_enabled()
_patch_wavep_copy2_for_localized_figures()


_JA_TEXT_PATCHED = False


# 関数: `_translate_wavep_text_to_japanese` の入出力契約と処理意図を定義する。
def _translate_wavep_text_to_japanese(text: str) -> str:
    """
    図中テキストの英語ラベルを日本語中心へ寄せる。
    完全翻訳ではなく、論文図で頻出する英語語彙を優先変換する。
    """
    if not text:
        return text

    # 数式だけの文字列は変換しない。

    stripped = text.strip()
    if stripped.startswith("$") and stripped.endswith("$"):
        return text

    out = text
    out = re.sub(r"\bPhase\s*\d+(?:\.\d+)*\s*(?:/\s*Step\s*\d+(?:\.\d+)*)?\s*:?\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\bStep\s*\d+(?:\.\d+)*\s*:?\s*", "", out, flags=re.IGNORECASE)

    exact_map = {
        "Action-principle EL derivation audit": "作用原理 EL 導出監査",
        "Observable route uniqueness lock": "観測経路の一意性ロック",
        "Operational checks": "運用チェック",
        "Frame-dragging gate: static vs holdout prediction": "フレームドラッグ判定: 静的モデルとホールドアウト予測の比較",
        "L_rot coupling freeze": "L_rot 結合の凍結",
        "P-model CMB acoustic-peak audit": "P-model CMB 音響ピーク監査",
        "CMB polarization transfer audit": "CMB 偏光伝達監査",
        "phase-offset audit": "位相ずれ監査",
        "strong-field higher-order closure audit": "強場高次閉包監査",
        "explicit N0^(2) terms and strong-field solution": "N0^(2) 明示項と強場解",
        "Direct ring diameter audit": "リング直径の直接監査",
        "Asymmetry range check (when available)": "非対称性レンジ確認（利用可能時）",
        "Bell systematics decomposition (15 items; operational)": "ベル系統分解（15項目・運用）",
        "Selection sensitivity summary": "選択感度サマリー",
        "normalized score (<=1 means threshold satisfied)": "正規化スコア（<=1 で閾値を満足）",
        "test max abs(z)": "検証 max abs(z)",
        "Holdout severity": "ホールドアウト厳しさ",
        "Blackbody ratio holdout": "黒体比ホールドアウト",
        "Blackbody enthalpy flux holdout": "黒体エンタルピー流束ホールドアウト",
        "GW polarization coverage expansion audit": "GW偏光カバレッジ拡張監査",
        "GW polarization event-update watch": "GW偏光イベント更新ウォッチ",
        "Ablation chain": "除去検証連鎖",
        "Station bottleneck": "局ボトルネック",
        "Gain-share split": "改善寄与率分解",
        "Molecular transition baseline (primary line lists; representative transitions)": "分子遷移の基準（一次線リスト；代表遷移）",
        "Selection: top-N by Einstein A.": "選択：Einstein A の上位 N。",
        "Target fix only (not P-model derivation).": "ターゲット固定のみ（P-model 導出ではない）。",
        "Atomic baseline (Helium, NIST ASD)": "原子基準（ヘリウム、NIST ASD）",
        "Atomic baseline (Hydrogen, NIST ASD)": "原子基準（水素、NIST ASD）",
        "Vacuum wavelength λ [nm] (from NIST ASD)": "真空波長 λ [nm]（NIST ASD）",
        "This figure fixes baseline targets for Part III; it is not a derivation.": "本図は Part III の基準ターゲットを固定するものであり、導出ではない。",
        "light nuclei baselines (A=2,3,4) incl. A=3 charge radii": "軽核基準（A=2,3,4、A=3 電荷半径を含む）",
        "Mass defect baseline (CODATA via NIST)": "質量欠損の基準（CODATA/NIST）",
        "Binding energy per nucleon": "核子当たり結合エネルギー",
        "Charge radii (CODATA + IAEA compilation)": "電荷半径（CODATA + IAEA 集約）",
        "Definitions:": "定義：",
        "Notes:": "注記：",
        "representative nuclei (A=2,4,12,16) — A-dependence preview": "代表核種（A=2,4,12,16）— A 依存の概観",
        "Observed vs predicted (baseline) B/A": "観測値と予測値（基準）B/A",
        "Multi-body reduction sensitivity (C choices)": "多体系縮約感度（C の選択）",
        "Bell selection sensitivity summary": "ベル選択感度の要約",
        "Single-photon interference: V → σ L": "単一光子干渉: V → σL",
        "HOM: visibility vs separation (reported)": "HOM: 可視度と分離の比較（報告値）",
        "Low-frequency noise PSD (for indistinguishability)": "低周波ノイズ PSD（不可識別性のため）",
        "Unified summary": "統合要約",
        "Vacuum + QED precision observables (Casimir, Lamb, H 1S-2S, α)": "真空 + QED 精密観測量（カシミール、ラム、H 1S-2S、α）",
        "Casimir: sphere-plate force scale": "カシミール: 球-平板力スケール",
        "Lamb shift: scaling (why Z>1 helps)": "ラムシフト: スケーリング（なぜ Z>1 が有効か）",
        "Nuclear-size term (example; Table 4)": "核サイズ項（例: 表4）",
        "Used as an example of non-QED δR(radius)": "非QED δR（半径）の例として使用",
        "α precision cross-check (recoil vs g-2)": "α 精密度クロスチェック（反跳 対 g-2）",
        "Used as an example of non-QED": "非QED の例として使用",
        "see Methods in arXiv:2106.03871v2": "手法は arXiv:2106.03871v2 を参照",
        "closest at": "最近接",
        "local bound |S|=2": "局所境界 |S|=2",
        "natural window": "自然窓幅",
        "accidental-subtracted": "偶発差し引き",
        "trial-based": "試行ベース",
        "coincidence-based": "同時計数ベース",
        "binding energy B (MeV)": "結合エネルギー B (MeV)",
        "B/A (MeV per nucleon)": "B/A（MeV/核子）",
        "rms radius (fm)": "実効半径 (fm)",
        "Independent σ propagation (no covariance).": "独立な σ 伝播（共分散なし）。",
        "No covariance.": "共分散なし。",
        "Selection summary": "選別要約",
        "window half-width (ns)": "窓幅 半値幅 (ns)",
        "start offset (ns)": "開始オフセット (ns)",
        "IS vs window": "IS 対 窓幅",
        "CH J_prob vs window": "CH J_prob 対 窓幅",
        "CHSH S vs start offset": "CHSH S 対 開始オフセット",
        "Minimal additional physics under frozen falsification thresholds": "凍結反証閾値下での最小追加物理",
        "Operational z-scores (pass if abs(z)<=3)": "運用 zスコア（|z|<=3で通過）",
        "pass if abs(z)<=3": "|z|<=3で通過",
        "if abs(z)<=3": "|z|<=3",
    }
    exact_map.update(
        {
            "validation scoreboard": "検証スコアボード",
            "quantum scoreboard": "量子スコアボード",
            "Table 1 vs Part IV label parity audit": "検証サマリ表 Part IV ラベル整合監査",
            "table1 rows": "検証サマリ表 行数",
            "SPARC audit: Vobs vs Vbar and P-model-corrected VP (single fit)": "SPARC監査: Vobs と Vbar、および P-model補正 VP の比較（単一フィット）",
            "SPARC rotation curves (all points)": "SPARC回転曲線（全点）",
            "baryon-only": "バリオンのみ",
            "P-model corrected": "P-model補正",
            "Vmodel [km/s]": "モデル速度 Vmodel [km/s]",
            "Vobs [km/s]": "観測速度 Vobs [km/s]",
            "Normalized residual distribution": "正規化残差分布",
            "(Vobs - Vmodel) / sigma": "(Vobs - Vmodel) / σ",
            "Fit quality (single M/L parameter)": "フィット品質（単一 M/L パラメータ）",
            "global chi2/dof": "全体 chi2/dof",
            "Beta Cross-Channel Terminal Decision": "β横断チャネル終端判定",
            "VLBI channel": "VLBIチャネル",
            "LLR channel": "LLRチャネル",
            "MESSENGER channel": "MESSENGERチャネル",
            "Cross channel": "横断チャネル",
            "Beta terminal gate": "β終端ゲート",
            "status score (pass=0.5, watch=1.5, reject=2.8)": "状態スコア（通過=0.5、要監視=1.5、棄却=2.8）",
            "VLBI beta direct-fit nuisance sensitivity": "VLBI β直接フィットの擾乱感度",
            "beta estimate (+/-1sigma)": "β推定値（±1σ）",
            "weighted RMSE [ps]": "重み付き RMSE [ps]",
            "beta estimate": "β推定値",
            "VLBI 17MAY01XA direct beta fit (vgosDb primary data)": "VLBI 17MAY01XA の β直接フィット（vgosDb 一次データ）",
            "residual vs template": "残差とテンプレートの比較",
            "weighted fit": "重み付きフィット",
            "Obs(IF) - Base [ps]": "観測(IF) - 基準 [ps]",
            "Cal-BendSun template [ps]": "Cal-BendSun テンプレート [ps]",
            "Residual - fit [ps]": "残差 - フィット [ps]",
            "VLBI beta source-filter sensitivity (all vs selected)": "VLBI βソースフィルタ感度（全件 対 選択）",
            "selected sources": "選択ソース",
            "all sources": "全ソース",
            "Quantum connection block covariance": "量子接続ブロック共分散",
            "Quantum connection block covariance ()": "量子接続ブロック共分散",
            "Quantum connection block covariance (Step 7.21.4)": "量子接続ブロック共分散",
            "Channel covariance": "チャネル共分散",
            "Channel correlation": "チャネル相関",
            "log1p-normalized scale": "log1p正規化スケール",
            "mode index": "モード番号",
            "eigenvalue": "固有値",
            "Spherical average of N0^(2)": "N0^(2) の球面平均",
            "source term": "ソース項",
            "Perturbative solution of P0": "P0 の摂動解",
            "P0 linear": "P0 線形項",
            "Gap budget": "ギャップ内訳",
            "Sr clock (V=0.9)": "Sr時計 (V=0.9)",
            "Cs clock (V=0.9)": "Cs時計 (V=0.9)",
            "Sr clock (V=0.5)": "Sr時計 (V=0.5)",
            "Cs clock (V=0.5)": "Cs時計 (V=0.5)",
        }
    )
    for src, dst in exact_map.items():
        out = re.sub(re.escape(src), dst, out, flags=re.IGNORECASE)

    # 英語語彙を単語単位で置換する。

    word_map = {
        "audit": "監査",
        "summary": "要約",
        "prediction": "予測",
        "predicted": "予測",
        "observed": "観測",
        "baseline": "基準",
        "holdout": "ホールドアウト",
        "gate": "ゲート",
        "score": "スコア",
        "scores": "スコア",
        "check": "チェック",
        "checks": "チェック",
        "closure": "閉包",
        "derivation": "導出",
        "residual": "残差",
        "residuals": "残差",
        "ratio": "比",
        "ratios": "比",
        "distribution": "分布",
        "comparison": "比較",
        "consistency": "整合性",
        "operational": "運用",
        "dynamic": "動的",
        "static": "静的",
        "model": "モデル",
        "models": "モデル",
        "phase": "位相",
        "offset": "オフセット",
        "template": "テンプレート",
        "transfer": "伝達",
        "cluster": "銀河団",
        "collision": "衝突",
        "route": "経路",
        "uniqueness": "一意性",
        "falsification": "反証",
        "pack": "パック",
        "bell": "ベル",
        "cross": "横断",
        "dataset": "データセット",
        "datasets": "データセット",
        "connection": "接続",
        "connections": "接続",
        "matter": "物質",
        "wave": "波",
        "interference": "干渉",
        "thermal": "熱",
        "kpi": "KPI",
        "normalized": "正規化",
        "satisfied": "満足",
        "severity": "厳しさ",
        "test": "検証",
        "tests": "検証",
        "power": "冪",
        "law": "則",
        "fixed": "固定",
        "exponent": "指数",
        "scaling": "スケーリング",
        "universal": "普遍",
        "timescale": "時間スケール",
        "event": "イベント",
        "events": "イベント",
        "atomic": "原子",
        "molecular": "分子",
        "transition": "遷移",
        "transitions": "遷移",
        "primary": "一次",
        "line": "線",
        "lines": "線",
        "representative": "代表",
        "top": "上位",
        "fix": "固定",
        "target": "ターゲット",
        "targets": "ターゲット",
        "only": "のみ",
        "not": "非",
        "derivation": "導出",
        "sensitivity": "感度",
        "choice": "選択",
        "choices": "選択",
        "definition": "定義",
        "definitions": "定義",
        "note": "注記",
        "notes": "注記",
        "nucleon": "核子",
        "charge": "電荷",
        "radii": "半径",
        "mass": "質量",
        "defect": "欠損",
        "multi": "多",
        "body": "体",
        "additional": "追加",
        "physics": "物理",
        "frozen": "凍結",
        "coherent": "コヒーレント",
        "bonds": "結合",
        "preview": "概観",
        "boundary": "境界",
        "coverage": "カバレッジ",
        "expansion": "拡張",
        "subset": "部分集合",
        "sweep": "掃引",
        "count": "件数",
        "flag": "フラグ",
        "station": "局",
        "bottleneck": "ボトルネック",
        "share": "寄与率",
        "gain": "改善",
        "ablation": "除去検証",
        "overall": "全体",
        "proxy": "代替指標",
        "dimensionless": "無次元",
        "scale": "スケール",
        "clock": "時計",
        "neutron": "中性子",
        "atom": "原子",
        "optical": "光学",
        "blackbody": "黒体",
        "radiation": "放射",
        "photon": "光子",
        "energy": "エネルギー",
        "pressure": "圧力",
        "entropy": "エントロピー",
        "enthalpy": "エンタルピー",
        "helmholtz": "ヘルムホルツ",
        "frequency": "周波数",
        "wavelength": "波長",
        "heat": "熱",
        "capacity": "容量",
        "density": "密度",
        "flux": "流束",
        "momentum": "運動量",
        "peak": "ピーク",
        "product": "積",
        "free": "自由",
        "derived": "導出",
        "single": "単一",
        "reported": "報告値",
        "temporal": "時間",
        "separation": "分離",
        "indistinguishability": "不可識別性",
        "unified": "統合",
        "observables": "観測量",
        "observable": "観測量",
        "vacuum": "真空",
        "precision": "精密",
        "visibility": "可視度",
        "squeezing": "スクイージング",
        "noise": "ノイズ",
        "delay": "遅延",
        "equiv": "等価",
        "relative": "相対",
        "methods": "手法",
        "method": "手法",
        "casimir": "カシミール",
        "sphere": "球",
        "plate": "平板",
        "force": "力",
        "ideal": "理想",
        "conductor": "導体",
        "lamb": "ラム",
        "nuclear": "核",
        "example": "例",
        "used": "使用",
        "unknown": "未知",
        "loop": "ループ",
        "cross-check": "クロスチェック",
        "crosscheck": "クロスチェック",
        "half-width": "半値幅",
        "width": "幅",
        "corrected": "補正",
        "accidental": "偶発",
        "subtracted": "差し引き",
        "trial": "試行",
        "coincidence": "同時計数",
        "raw": "生データ",
        "by": "による",
        "for": "用",
        "from": "由来",
        "as": "として",
        "and": "と",
        "of": "の",
        "closest": "最近接",
        "at": "で",
        "kernel": "カーネル",
        "decomposition": "分解",
        "systematics": "系統",
        "selection": "選択",
        "window": "窓幅",
        "threshold": "閾値",
        "error": "誤差",
        "fit": "フィット",
        "fitted": "フィット",
        "trace": "軌跡",
        "chain": "連鎖",
        "watch": "要監視",
        "reject": "棄却",
        "pass": "通過",
        "required": "必要",
        "channel": "チャネル",
        "channels": "チャネル",
        "source": "ソース",
        "input": "入力",
        "output": "出力",
        "global": "全体",
        "local": "局所",
        "range": "範囲",
        "value": "値",
        "values": "値",
        "minimal": "最小",
        "under": "下で",
        "threshold": "閾値",
        "thresholds": "閾値",
        "units": "単位",
        "saturation": "飽和",
        "spacing": "間隔",
        "if": "なら",
        "mean": "平均",
        "median": "中央値",
        "max": "最大",
        "min": "最小",
        "binding": "結合",
        "radius": "半径",
        "radii": "半径",
        "rms": "実効",
        "independent": "独立",
        "propagation": "伝播",
        "covariance": "共分散",
        "compilation": "集約",
        "via": "経由",
        "combined": "統合",
        "best": "最良",
        "bound": "境界",
        "natural": "自然",
        "start": "開始",
        "half": "半",
        "width": "幅",
        "clipped": "クリップ済み",
        "variant": "変種",
        "per": "当たり",
        "no": "なし",
    }
    word_map.update(
        {
            "validation": "検証",
            "quantum": "量子",
            "pmodel": "P-model",
            "ddr": "DDR",
            "sparc": "SPARC",
            "vlbi": "VLBI",
            "beta": "β",
            "scoreboard": "スコアボード",
            "metric": "指標",
            "metrics": "指標",
            "table": "表",
            "label": "ラベル",
            "labels": "ラベル",
            "parity": "整合",
            "row": "行",
            "rows": "行",
            "matched": "一致",
            "missing": "欠落",
            "extra": "余剰",
            "quality": "品質",
            "parameter": "パラメータ",
            "parameters": "パラメータ",
            "network": "ネットワーク",
            "polarization": "偏光",
            "update": "更新",
            "manual": "手計算",
            "terminal": "終端",
            "decision": "判定",
            "covariance": "共分散",
            "correlation": "相関",
            "count": "件数",
            "selected": "選択",
            "direct": "直接",
            "data": "データ",
            "nuisance": "擾乱",
            "filter": "フィルタ",
            "registry": "台帳",
            "spherical": "球面",
            "average": "平均",
            "solution": "解",
            "block": "ブロック",
            "root": "根本",
            "cause": "原因",
            "intercept": "切片",
            "linear": "線形",
            "condensed": "凝縮系",
            "baryon": "バリオン",
            "rotation": "回転",
            "curve": "曲線",
            "rotating": "回転",
            "ring": "リング",
            "lpath": "Lpath",
            "lagrangian": "ラグランジアン",
            "noether": "ネーター",
            "drift": "ドリフト",
            "bh": "BH",
        }
    )
    for eng, ja in word_map.items():
        out = re.sub(rf"(?<![A-Za-z]){re.escape(eng)}(?![A-Za-z])", ja, out, flags=re.IGNORECASE)

    out = out.replace("->", "→")
    out = re.sub(r"\bvs\b", "対", out, flags=re.IGNORECASE)
    out = out.replace("P-モデル", "P-model")
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r"[ ]{2,}", " ", out).strip()
    return out


# 関数: `_apply_wavep_force_japanese_text_if_enabled` の入出力契約と処理意図を定義する。

def _apply_wavep_force_japanese_text_if_enabled() -> None:
    raw = os.getenv("WAVEP_MPL_FORCE_JA_TEXT", "").strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return

    try:
        import matplotlib as mpl
        from matplotlib.text import Text
    except Exception:
        return

    # 日本語グリフを持つフォントを優先して豆腐化を回避する。

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "Yu Gothic",
        "Yu Gothic UI",
        "Meiryo",
        "MS Gothic",
        "IPAexGothic",
        "IPAGothic",
        "TakaoGothic",
        "DejaVu Sans",
    ]

    global _JA_TEXT_PATCHED
    if _JA_TEXT_PATCHED:
        return

    original_set_text = Text.set_text

    # 関数: `patched_set_text` の入出力契約と処理意図を定義する。
    def patched_set_text(self, s):
        if isinstance(s, str):
            s = _translate_wavep_text_to_japanese(s)

        return original_set_text(self, s)

    Text.set_text = patched_set_text
    _JA_TEXT_PATCHED = True


_apply_wavep_force_japanese_text_if_enabled()
