"""
P-model 論文用 Matplotlib プロットスタイル定義 (PDF出力最適化版)
scripts/utils/plot_style.py

学術論文の印刷品質（ベクターデータ、見切れ防止、フォント埋め込み）を
すべての検証スクリプトで一貫して保証するためのモジュールです。

加えて、`paper_build.py` から `WAVEP_MPL_FONT_PROFILE` を渡された場合は、
役割別（title / axis / tick / legend / note / suptitle）の共通サイズを
build 時に一括適用できるようにする。
"""

import os
import re
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from typing import Any

_WAVEP_REPO_ROOT = Path(__file__).resolve().parents[2]
_WAVEP_STATIC_CJK_FONT = _WAVEP_REPO_ROOT / "output" / "private" / "summary" / "fonts" / "NotoSansJP-Regular-static.ttf"
_PMODEL_TEXTWIDTH_MM = 170.0
_PMODEL_TEXTHEIGHT_MM = 257.0
_PMODEL_MM_PER_INCH = 25.4
_PMODEL_TEXTWIDTH_IN = _PMODEL_TEXTWIDTH_MM / _PMODEL_MM_PER_INCH
_PMODEL_TEXTHEIGHT_IN = _PMODEL_TEXTHEIGHT_MM / _PMODEL_MM_PER_INCH
_PMODEL_FULL_TALL_HEIGHT_IN = _PMODEL_TEXTHEIGHT_IN * 0.78


_WAVEP_COMMON_FIGURE_PROFILE: dict[str, float] = {
    "base": 8.2,
    "title": 10.8,
    "axis": 8.8,
    "tick": 7.8,
    "legend": 7.8,
    "note": 7.8,
    "suptitle": 11.0,
}

_WAVEP_PART2_FIGURE_PROFILE: dict[str, float] = {
    "base": 8.2,
    "title": 10.8,
    "axis": 8.8,
    "tick": 7.8,
    "legend": 7.8,
    "note": 7.8,
    "suptitle": 11.0,
}

_WAVEP_FONT_PROFILES: dict[str, dict[str, float]] = {
    "paper": dict(_WAVEP_COMMON_FIGURE_PROFILE),
    "part2_astrophysics": dict(_WAVEP_PART2_FIGURE_PROFILE),
    "part3_quantum": dict(_WAVEP_COMMON_FIGURE_PROFILE),
    "part3a_quantum_foundations": dict(_WAVEP_COMMON_FIGURE_PROFILE),
    "part3b_quantum_verification": dict(_WAVEP_COMMON_FIGURE_PROFILE),
    "part4_verification": dict(_WAVEP_COMMON_FIGURE_PROFILE),
    "part5_future_predictions": dict(_WAVEP_COMMON_FIGURE_PROFILE),
}

_WAVEP_COMMON_LINE_PROFILE: dict[str, float] = {
    "default_linewidth": 1.2,
    "max_linewidth": 1.35,
    "reference_scale": 0.72,
    "reference_max": 0.85,
    "errorbar_scale": 0.75,
    "errorbar_max": 1.0,
    "default_markersize": 4.0,
    "marker_scale": 0.85,
    "max_markersize": 4.2,
}

_WAVEP_PART2_LINE_PROFILE: dict[str, float] = {
    "default_linewidth": 1.2,
    "max_linewidth": 1.35,
    "reference_scale": 0.72,
    "reference_max": 0.85,
    "errorbar_scale": 0.75,
    "errorbar_max": 1.0,
    "default_markersize": 4.0,
    "marker_scale": 0.85,
    "max_markersize": 4.2,
}

_WAVEP_LINE_PROFILES: dict[str, dict[str, float]] = {
    "paper": dict(_WAVEP_COMMON_LINE_PROFILE),
    "part2_astrophysics": dict(_WAVEP_PART2_LINE_PROFILE),
    "part3_quantum": dict(_WAVEP_COMMON_LINE_PROFILE),
    "part3a_quantum_foundations": dict(_WAVEP_COMMON_LINE_PROFILE),
    "part3b_quantum_verification": dict(_WAVEP_COMMON_LINE_PROFILE),
    "part4_verification": dict(_WAVEP_COMMON_LINE_PROFILE),
    "part5_future_predictions": dict(_WAVEP_COMMON_LINE_PROFILE),
}

_WAVEP_LAYOUT_TEMPLATES: dict[str, dict[str, Any]] = {
    "paper_diagram": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 5.10),
        "subplots_adjust": {
            "left": 0.040,
            "right": 0.992,
            "top": 0.955,
            "bottom": 0.090,
        },
    },
    "paper_flowchart": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 4.40),
        "subplots_adjust": {
            "left": 0.035,
            "right": 0.995,
            "top": 0.955,
            "bottom": 0.110,
        },
    },
    "paper_single_panel": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 5.00),
        "subplots_adjust": {
            "left": 0.115,
            "right": 0.980,
            "top": 0.900,
            "bottom": 0.185,
        },
    },
    "paper_side_by_side": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 5.15),
        "subplots_adjust": {
            "left": 0.090,
            "right": 0.988,
            "top": 0.900,
            "bottom": 0.185,
            "wspace": 0.28,
        },
    },
    "paper_two_panel": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 6.95),
        "subplots_adjust": {
            "left": 0.112,
            "right": 0.985,
            "top": 0.940,
            "bottom": 0.105,
            "hspace": 0.36,
        },
    },
    "paper_grid_tall": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 6.90),
        "subplots_adjust": {
            "left": 0.095,
            "right": 0.985,
            "top": 0.945,
            "bottom": 0.095,
            "wspace": 0.25,
            "hspace": 0.34,
        },
    },
    "paper_three_panel": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 8.10),
        "subplots_adjust": {
            "left": 0.105,
            "right": 0.985,
            "top": 0.955,
            "bottom": 0.085,
            "hspace": 0.34,
        },
    },
    "part2_single_panel": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 4.60),
        "subplots_adjust": {
            "left": 0.120,
            "right": 0.960,
            "top": 0.875,
            "bottom": 0.185,
        },
    },
    "part2_single_panel_sparse": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 4.90),
        "subplots_adjust": {
            "left": 0.105,
            "right": 0.980,
            "top": 0.885,
            "bottom": 0.205,
        },
    },
    "part2_single_panel_tall": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 5.75),
        "subplots_adjust": {
            "left": 0.115,
            "right": 0.980,
            "top": 0.895,
            "bottom": 0.180,
        },
    },
    "part2_side_by_side": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 4.95),
        "subplots_adjust": {
            "left": 0.090,
            "right": 0.988,
            "top": 0.875,
            "bottom": 0.190,
            "wspace": 0.30,
        },
    },
    "part2_single_panel_legend_top": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 5.40),
        "subplots_adjust": {
            "left": 0.120,
            "right": 0.960,
            "top": 0.805,
            "bottom": 0.185,
        },
    },
    "part2_single_panel_legend_bottom": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 5.40),
        "subplots_adjust": {
            "left": 0.120,
            "right": 0.960,
            "top": 0.885,
            "bottom": 0.275,
        },
    },
    "part2_two_panel": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 6.75),
        "subplots_adjust": {
            "left": 0.115,
            "right": 0.985,
            "top": 0.935,
            "bottom": 0.090,
            "hspace": 0.38,
        },
    },
    "part2_two_panel_dense_x": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 7.35),
        "subplots_adjust": {
            "left": 0.108,
            "right": 0.985,
            "top": 0.930,
            "bottom": 0.165,
            "hspace": 0.33,
        },
    },
    "part2_two_panel_tall": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 7.80),
        "subplots_adjust": {
            "left": 0.108,
            "right": 0.985,
            "top": 0.932,
            "bottom": 0.145,
            "hspace": 0.34,
        },
    },
    "part2_two_panel_spacious": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 8.20),
        "subplots_adjust": {
            "left": 0.110,
            "right": 0.985,
            "top": 0.885,
            "bottom": 0.105,
            "hspace": 0.56,
        },
    },
    "part2_two_panel_quantum_spacious": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 10.20),
        "subplots_adjust": {
            "left": 0.110,
            "right": 0.985,
            "top": 0.765,
            "bottom": 0.100,
            "hspace": 1.28,
        },
    },
    "part2_quad_panel": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 5.95),
        "subplots_adjust": {
            "left": 0.090,
            "right": 0.988,
            "top": 0.900,
            "bottom": 0.150,
            "wspace": 0.28,
            "hspace": 0.34,
        },
    },
    "part2_quad_panel_spacious": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 6.60),
        "subplots_adjust": {
            "left": 0.090,
            "right": 0.988,
            "top": 0.865,
            "bottom": 0.145,
            "wspace": 0.30,
            "hspace": 0.58,
        },
    },
    "part2_quad_panel_quantum_spacious": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 9.20),
        "subplots_adjust": {
            "left": 0.095,
            "right": 0.988,
            "top": 0.765,
            "bottom": 0.120,
            "wspace": 0.34,
            "hspace": 1.26,
        },
    },
    "part2_three_panel_tall": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 7.80),
        "subplots_adjust": {
            "left": 0.095,
            "right": 0.985,
            "top": 0.945,
            "bottom": 0.090,
            "hspace": 0.33,
        },
    },
    "part2_three_panel_spacious": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 8.70),
        "subplots_adjust": {
            "left": 0.110,
            "right": 0.985,
            "top": 0.875,
            "bottom": 0.090,
            "hspace": 0.62,
        },
    },
    "part2_three_panel_quantum_spacious": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 12.30),
        "subplots_adjust": {
            "left": 0.110,
            "right": 0.985,
            "top": 0.765,
            "bottom": 0.085,
            "hspace": 1.36,
        },
    },
    "part2_four_panel_quantum_spacious": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 15.20),
        "subplots_adjust": {
            "left": 0.110,
            "right": 0.985,
            "top": 0.765,
            "bottom": 0.070,
            "hspace": 1.36,
        },
    },
    "part2_four_panel_spacious": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 11.80),
        "subplots_adjust": {
            "left": 0.110,
            "right": 0.985,
            "top": 0.850,
            "bottom": 0.070,
            "hspace": 0.88,
        },
    },
    "part2_full_tall": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, _PMODEL_FULL_TALL_HEIGHT_IN),
        "subplots_adjust": {
            "left": 0.105,
            "right": 0.985,
            "top": 0.955,
            "bottom": 0.085,
        },
    },
    "part2_full_tall_spacious": {
        "figsize": (_PMODEL_TEXTWIDTH_IN, 9.20),
        "subplots_adjust": {
            "left": 0.110,
            "right": 0.985,
            "top": 0.885,
            "bottom": 0.080,
            "hspace": 0.62,
        },
    },
}

_FONT_PROFILE_STATE: dict[str, Any] = {
    "profile_name": "paper",
    "scale": 1.0,
    "sizes": dict(_WAVEP_FONT_PROFILES["paper"]),
}

_LINE_PROFILE_STATE: dict[str, Any] = {
    "profile_name": "paper",
    "scale": 1.0,
    "sizes": dict(_WAVEP_LINE_PROFILES["paper"]),
}

_WAVEP_CJK_FONT_FILES: tuple[str, ...] = (
    str(_WAVEP_STATIC_CJK_FONT),
    r"C:\Windows\Fonts\NotoSansJP-Regular.ttf",
    r"C:\Windows\Fonts\NotoSansCJKjp-Regular.otf",
    r"C:\Windows\Fonts\NotoSansCJK-Regular.ttc",
    r"C:\Windows\Fonts\NotoSansJP-VF.ttf",
)

_WAVEP_CJK_FONT_NAMES: tuple[str, ...] = (
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "Source Han Sans",
    "Source Han Sans JP",
    "Yu Gothic",
    "Meiryo",
    "BIZ UDGothic",
    "MS Gothic",
)

_ROLE_FONT_PATCHED = False
_CJK_FONT_OVERRIDE_PATCHED = False
_CJK_TEXT_FONTFILE_PATCHED = False
_CJK_TEXT_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uff00-\uffef]")
_FIGURE_TEXT_ROLE_STACK: list[str] = []


# 関数: Noto Sans JP variable font から regular instance を生成し、小文字視認性を安定させる。
def _ensure_wavep_static_cjk_font() -> Path | None:
    if _WAVEP_STATIC_CJK_FONT.exists():
        return _WAVEP_STATIC_CJK_FONT

    source_vf = Path(r"C:\Windows\Fonts\NotoSansJP-VF.ttf")
    if not source_vf.exists():
        return None

    try:
        from fontTools.ttLib import TTFont
        from fontTools.varLib.instancer import instantiateVariableFont
    except Exception:
        return None

    try:
        _WAVEP_STATIC_CJK_FONT.parent.mkdir(parents=True, exist_ok=True)
        vf_font = TTFont(str(source_vf))
        regular_font = instantiateVariableFont(vf_font, {"wght": 400}, inplace=False)
        regular_font.save(str(_WAVEP_STATIC_CJK_FONT))
    except Exception:
        return None

    return _WAVEP_STATIC_CJK_FONT


# 関数: `resolve_wavep_cjk_font_family` の入出力契約と処理意図を定義する。

def resolve_wavep_cjk_font_family(*, preferred_name: str | None = None) -> str | None:
    """
    build / figure 共通で使う日本語 sans-serif font family を返す。

    要求名は `Noto Sans CJK JP` でも、Windows の実体フォント
    `NotoSansJP-VF.ttf` から登録された `Noto Sans JP` へ正規化され得る。
    """
    try:
        from matplotlib import font_manager as fm
    except Exception:
        return None

    requested = str(preferred_name or os.getenv("WAVEP_MPL_CJK_FONT", "")).strip()
    path_candidates: list[Path] = []
    env_path = str(os.getenv("WAVEP_MPL_CJK_FONT_PATH", "")).strip()
    if env_path:
        path_candidates.append(Path(env_path))

    static_font = _ensure_wavep_static_cjk_font()
    if static_font is not None:
        path_candidates.append(static_font)

    for raw in _WAVEP_CJK_FONT_FILES:
        path_candidates.append(Path(raw))

    for path in path_candidates:
        try:
            if not path.exists():
                continue

            fm.fontManager.addfont(str(path))
            resolved_name = fm.FontProperties(fname=str(path)).get_name()
            if resolved_name:
                return str(resolved_name)
        except Exception:
            continue

    available = {font.name for font in fm.fontManager.ttflist}
    preferred_candidates: list[str] = []
    if requested:
        preferred_candidates.append(requested)

    preferred_candidates.extend(_WAVEP_CJK_FONT_NAMES)
    for name in preferred_candidates:
        if name in available:
            return str(name)

    return None


# 関数: `install_wavep_cjk_font_override` の入出力契約と処理意図を定義する。

def install_wavep_cjk_font_override(*, preferred_name: str | None = None) -> str | None:
    """
    明示的な日本語フォント指定も Noto 系へ寄せるため、Matplotlib の
    `findfont` を alias rewrite 付きで patch する。
    """
    target_family = resolve_wavep_cjk_font_family(preferred_name=preferred_name)
    if not target_family:
        return None

    import matplotlib as mpl
    from matplotlib import font_manager
    from matplotlib.font_manager import FontProperties

    mpl.rcParams["font.family"] = [target_family, "DejaVu Sans"]
    mpl.rcParams["font.sans-serif"] = [target_family, "DejaVu Sans"]

    global _CJK_FONT_OVERRIDE_PATCHED
    if _CJK_FONT_OVERRIDE_PATCHED:
        return target_family

    alias_names = {
        "sans-serif",
        "noto sans cjk jp",
        "noto sans jp",
        "source han sans",
        "source han sans jp",
        "yu gothic",
        "yu gothic ui",
        "meiryo",
        "biz udgothic",
        "ms gothic",
        "ipaexgothic",
    }
    original_manager_findfont = font_manager.FontManager.findfont
    target_font_path = original_manager_findfont(
        font_manager.fontManager,
        FontProperties(family=[target_family, "DejaVu Sans"]),
    )

    # 関数: `_coerce_font_properties` の入出力契約と処理意図を定義する。
    def _coerce_font_properties(prop: Any) -> FontProperties:
        if isinstance(prop, FontProperties):
            return prop

        try:
            return FontProperties(prop)
        except Exception:
            return FontProperties()

    # 関数: `_rewrite_font_property_if_needed` の入出力契約と処理意図を定義する。

    def _rewrite_font_property_if_needed(prop: Any) -> Any:
        base = _coerce_font_properties(prop)
        families = [str(name).strip().lower() for name in (base.get_family() or []) if str(name).strip()]
        if not any(name in alias_names for name in families):
            return prop

        rewritten = FontProperties()
        rewritten.set_family([target_family, "DejaVu Sans"])
        rewritten.set_style(base.get_style())
        rewritten.set_variant(base.get_variant())
        rewritten.set_weight(base.get_weight())
        rewritten.set_stretch(base.get_stretch())
        rewritten.set_size(base.get_size())
        return rewritten

    # 関数: `patched_manager_findfont` の入出力契約と処理意図を定義する。

    def patched_manager_findfont(self, prop, *args, **kwargs):
        return original_manager_findfont(self, _rewrite_font_property_if_needed(prop), *args, **kwargs)

    font_manager.FontManager.findfont = patched_manager_findfont

    global _CJK_TEXT_FONTFILE_PATCHED
    if not _CJK_TEXT_FONTFILE_PATCHED:
        from matplotlib.text import Text

        original_set_text = Text.set_text
        original_draw = Text.draw

        # 関数: `_apply_exact_cjk_fontfile_if_needed` の入出力契約と処理意図を定義する。
        def _apply_exact_cjk_fontfile_if_needed(text_artist: Text) -> None:
            current_text = text_artist.get_text()
            if not isinstance(current_text, str) or not _CJK_TEXT_CHAR_RE.search(current_text):
                return

            current_prop = text_artist.get_fontproperties()
            exact_prop = FontProperties(fname=target_font_path)
            exact_prop.set_size(current_prop.get_size_in_points())
            exact_prop.set_style(current_prop.get_style())
            exact_prop.set_variant(current_prop.get_variant())
            exact_prop.set_weight(current_prop.get_weight())
            exact_prop.set_stretch(current_prop.get_stretch())
            text_artist.set_fontproperties(exact_prop)

        # 関数: `patched_set_text` の入出力契約と処理意図を定義する。

        def patched_set_text(self, s):
            result = original_set_text(self, s)
            try:
                _apply_exact_cjk_fontfile_if_needed(self)
            except Exception:
                pass

            return result

        # 関数: `patched_draw` の入出力契約と処理意図を定義する。

        def patched_draw(self, renderer):
            try:
                _apply_exact_cjk_fontfile_if_needed(self)
            except Exception:
                pass

            return original_draw(self, renderer)

        Text.set_text = patched_set_text
        Text.draw = patched_draw
        _CJK_TEXT_FONTFILE_PATCHED = True

    _CJK_FONT_OVERRIDE_PATCHED = True
    return target_family

# 関数: `apply_paper_style` の入出力契約と処理意図を定義する。

def apply_paper_style() -> None:
    """
    P-model論文用の学術的なMatplotlibグローバル設定を適用する。
    各スクリプトの冒頭で一度呼び出すだけで、以降のプロットすべてに適用される。
    """
    sizes = get_wavep_font_profile()
    line_sizes = _current_line_profile()
    cjk_font = install_wavep_cjk_font_override()
    font_family = [cjk_font, "DejaVu Sans"] if cjk_font else ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family": font_family,
        "font.sans-serif": font_family,
        "text.color": "#111111",
        "font.size": sizes["base"],          # 全体の基本フォントサイズ
        "axes.titlesize": sizes["title"],    # グラフタイトルのサイズ
        "axes.labelsize": sizes["axis"],     # X軸・Y軸のラベルサイズ
        "axes.titlecolor": "#111111",
        "axes.labelcolor": "#111111",
        "axes.edgecolor": "#222222",
        "xtick.labelsize": sizes["tick"],    # X軸の目盛り文字サイズ
        "ytick.labelsize": sizes["tick"],    # Y軸の目盛り文字サイズ
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "legend.fontsize": sizes["legend"],  # 凡例の文字サイズ
        "legend.labelcolor": "#111111",
        "figure.titlesize": sizes["suptitle"],   # Figure全体のタイトルサイズ
        "lines.linewidth": line_sizes["default_linewidth"],   # 線の太さ
        "lines.markersize": line_sizes["default_markersize"],  # マーカーのサイズ
        
        # --- PDF出力向けの最適化設定 ---
        "pdf.fonttype": 42,       # フォントをアウトライン化せず埋め込む（論文PDFでの文字検索を可能にする鉄則）
        "ps.fonttype": 42,
        "savefig.format": "pdf",  # デフォルトの保存形式をPDFに固定
        "savefig.bbox": "tight",  # 見切れ防止
        "savefig.pad_inches": 0.1 
    })


# 関数: `_coerce_float_env` の入出力契約と処理意図を定義する。

def _coerce_float_env(raw: str, *, default: float) -> float:
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


# 関数: `_resolve_wavep_font_profile_name` の入出力契約と処理意図を定義する。

def _resolve_wavep_font_profile_name(name: str | None = None) -> str:
    requested = str(name or os.getenv("WAVEP_MPL_FONT_PROFILE", "paper")).strip().lower()
    if not requested:
        return "paper"

    aliases = {
        "part1_core_theory": "paper",
        "part1": "paper",
        "core": "paper",
        "part2": "part2_astrophysics",
        "astro": "part2_astrophysics",
        "astrophysics": "part2_astrophysics",
        "part3": "part3_quantum",
        "quantum": "part3_quantum",
        "part3a": "part3a_quantum_foundations",
        "part3b": "part3b_quantum_verification",
        "quantum_foundations": "part3a_quantum_foundations",
        "quantum_verification": "part3b_quantum_verification",
        "part4": "part4_verification",
        "verification": "part4_verification",
        "part5": "part5_future_predictions",
        "future": "part5_future_predictions",
    }
    resolved = aliases.get(requested, requested)
    if resolved not in _WAVEP_FONT_PROFILES:
        return "paper"

    return resolved


# 関数: `get_wavep_font_profile` の入出力契約と処理意図を定義する。

def get_wavep_font_profile(*, name: str | None = None, scale: float | None = None) -> dict[str, float]:
    """
    役割別 font profile を返す。

    戻り値のキー:
      - base
      - title
      - axis
      - tick
      - legend
      - note
      - suptitle
    """
    profile_name = _resolve_wavep_font_profile_name(name)
    if scale is None:
        scale_value = _coerce_float_env(os.getenv("WAVEP_MPL_FONT_SCALE", "1.0"), default=1.0)
    else:
        scale_value = float(scale)

    base = _WAVEP_FONT_PROFILES[profile_name]
    return {key: float(value) * scale_value for key, value in base.items()}


# 関数: `get_wavep_font_size` の入出力契約と処理意図を定義する。

def get_wavep_font_size(role: str, *, name: str | None = None, scale: float | None = None) -> float:
    """
    単一 role の推奨 font size を返す。
    将来的に各 figure script が role token を直接使うための入口として使う。
    """
    sizes = get_wavep_font_profile(name=name, scale=scale)
    normalized_role = str(role).strip().lower()
    if normalized_role not in sizes:
        raise KeyError(f"unknown font role: {role}")

    return float(sizes[normalized_role])


# 関数: `get_wavep_figure_layout_template` の入出力契約と処理意図を定義する。

def get_wavep_figure_layout_template(name: str) -> dict[str, Any]:
    normalized_name = str(name).strip().lower()
    if normalized_name not in _WAVEP_LAYOUT_TEMPLATES:
        raise KeyError(f"unknown figure layout template: {name}")

    template = _WAVEP_LAYOUT_TEMPLATES[normalized_name]
    return {
        "figsize": tuple(template["figsize"]),
        "subplots_adjust": dict(template["subplots_adjust"]),
    }


# 関数: `apply_wavep_figure_layout` の入出力契約と処理意図を定義する。

def apply_wavep_figure_layout(fig: Figure, *, template: str) -> dict[str, Any]:
    config = get_wavep_figure_layout_template(template)
    fig.set_size_inches(*config["figsize"], forward=True)
    fig.subplots_adjust(**config["subplots_adjust"])
    return config


# 関数: `should_wavep_export_normalize_to_textwidth` の入出力契約と処理意図を定義する。

def should_wavep_export_normalize_to_textwidth(*, profile_name: str | None = None) -> bool:
    """
    保存直前に textwidth 基準の fixed-width canvas へ正規化すべき profile を返す。

    Part III 系は source script 側の manual figsize がまだ混在しているため、
    savefig 直前に統一幅へ寄せる。
    """
    resolved = _resolve_wavep_font_profile_name(profile_name)
    return resolved in {
        "part3_quantum",
        "part3a_quantum_foundations",
        "part3b_quantum_verification",
    }


# 関数: `normalize_wavep_export_canvas_to_textwidth` の入出力契約と処理意図を定義する。

def normalize_wavep_export_canvas_to_textwidth(
    fig: Figure,
    *,
    profile_name: str | None = None,
) -> dict[str, float] | None:
    """
    figure を textwidth（170 mm）基準の fixed-width canvas へ寄せる。

    目的は、旧 script に残る manual figsize を savefig 直前に吸収し、
    PDF 取り込み時の実スケール差で font が小さく見える問題を抑えることにある。
    高さは元の aspect ratio を保持したまま追従させる。
    """
    if not should_wavep_export_normalize_to_textwidth(profile_name=profile_name):
        return None

    try:
        width_in, height_in = fig.get_size_inches()
    except Exception:
        return None

    width_in = float(width_in)
    height_in = float(height_in)
    if width_in <= 0.0 or height_in <= 0.0:
        return None

    scale = _PMODEL_TEXTWIDTH_IN / width_in
    target_width_in = _PMODEL_TEXTWIDTH_IN
    target_height_in = height_in * scale
    if abs(target_width_in - width_in) < 0.03 and abs(target_height_in - height_in) < 0.03:
        return None

    fig.set_size_inches(target_width_in, target_height_in, forward=True)
    return {
        "original_width_in": width_in,
        "original_height_in": height_in,
        "target_width_in": target_width_in,
        "target_height_in": target_height_in,
        "scale": scale,
    }


# 関数: `apply_wavep_compact_legend` の入出力契約と処理意図を定義する。

def apply_wavep_compact_legend(
    axes: Any,
    *,
    ncol: int = 2,
    anchor_y: float = 1.02,
    loc: str = "lower center",
    fontsize_role: str = "legend",
    framealpha: float = 0.95,
) -> Any:
    sizes = _current_role_sizes()
    return axes.legend(
        loc=loc,
        bbox_to_anchor=(0.5, anchor_y),
        ncol=int(max(1, ncol)),
        framealpha=float(framealpha),
        borderaxespad=0.0,
        columnspacing=1.0,
        handletextpad=0.6,
        fontsize=float(sizes[fontsize_role]),
    )


# 関数: `_current_role_sizes` の入出力契約と処理意図を定義する。

def _current_role_sizes() -> dict[str, float]:
    return dict(_FONT_PROFILE_STATE["sizes"])


# 関数: `_role_scale_factor` の入出力契約と処理意図を定義する。

def _role_scale_factor(role: str) -> float:
    normalized_role = str(role).strip().upper()
    specific = _coerce_float_env(os.getenv(f"WAVEP_MPL_ROLE_SCALE_{normalized_role}", "1.0"), default=1.0)
    global_scale = _coerce_float_env(os.getenv("WAVEP_MPL_ROLE_SCALE_ALL", "1.0"), default=1.0)
    return float(specific) * float(global_scale)


# 関数: `_role_font_size` の入出力契約と処理意図を定義する。

def _role_font_size(role: str) -> float:
    sizes = _current_role_sizes()
    return float(sizes[role]) * float(_role_scale_factor(role))


# 関数: `_current_line_profile` の入出力契約と処理意図を定義する。

def _current_line_profile() -> dict[str, float]:
    return dict(_LINE_PROFILE_STATE["sizes"])


# 関数: `_update_font_profile_state` の入出力契約と処理意図を定義する。

def _update_font_profile_state(*, profile_name: str, scale: float) -> dict[str, float]:
    sizes = get_wavep_font_profile(name=profile_name, scale=scale)
    _FONT_PROFILE_STATE["profile_name"] = profile_name
    _FONT_PROFILE_STATE["scale"] = float(scale)
    _FONT_PROFILE_STATE["sizes"] = dict(sizes)
    return sizes


# 関数: `_update_line_profile_state` の入出力契約と処理意図を定義する。

def _update_line_profile_state(*, profile_name: str, scale: float) -> dict[str, float]:
    base = _WAVEP_LINE_PROFILES.get(profile_name, _WAVEP_LINE_PROFILES["paper"])
    sizes = {key: float(value) * float(scale) for key, value in base.items()}
    _LINE_PROFILE_STATE["profile_name"] = profile_name
    _LINE_PROFILE_STATE["scale"] = float(scale)
    _LINE_PROFILE_STATE["sizes"] = dict(sizes)
    return sizes


# 関数: `_apply_linewidth_profile` の入出力契約と処理意図を定義する。

def _apply_linewidth_profile(value: Any, *, kind: str) -> float:
    profile = _current_line_profile()
    default_value = float(profile["default_linewidth"])
    current = _coerce_numeric_fontsize(value)
    if current is None:
        current = default_value

    if kind == "reference":
        return min(float(current) * float(profile["reference_scale"]), float(profile["reference_max"]))

    if kind == "errorbar":
        return min(float(current) * float(profile["errorbar_scale"]), float(profile["errorbar_max"]))

    return min(float(current), float(profile["max_linewidth"]))


# 関数: `_apply_markersize_profile` の入出力契約と処理意図を定義する。

def _apply_markersize_profile(value: Any) -> float:
    profile = _current_line_profile()
    current = _coerce_numeric_fontsize(value)
    if current is None:
        current = float(profile["default_markersize"])

    scaled = float(current) * float(profile["marker_scale"])
    return min(float(scaled), float(profile["max_markersize"]))


# 関数: `_apply_plot_line_kwargs` の入出力契約と処理意図を定義する。

def _apply_plot_line_kwargs(kwargs: dict[str, Any], *, kind: str) -> dict[str, Any]:
    patched = dict(kwargs)
    if "linewidth" in patched or "lw" in patched:
        key = "linewidth" if "linewidth" in patched else "lw"
        patched[key] = _apply_linewidth_profile(patched.get(key), kind=kind)
    else:
        patched["linewidth"] = _apply_linewidth_profile(None, kind=kind)

    if "markersize" in patched or "ms" in patched:
        key = "markersize" if "markersize" in patched else "ms"
        patched[key] = _apply_markersize_profile(patched.get(key))

    return patched


# 関数: `_apply_errorbar_line_kwargs` の入出力契約と処理意図を定義する。

def _apply_errorbar_line_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    patched = _apply_plot_line_kwargs(kwargs, kind="line")
    if "elinewidth" in patched:
        patched["elinewidth"] = _apply_linewidth_profile(patched.get("elinewidth"), kind="errorbar")

    return patched


# 関数: `_apply_role_font_kwargs` の入出力契約と処理意図を定義する。

def _apply_role_font_kwargs(kwargs: dict[str, Any], *, role: str) -> dict[str, Any]:
    patched = dict(kwargs)
    force_role_fonts = str(os.getenv("WAVEP_MPL_FORCE_ROLE_FONTS", "")).strip().lower() in {"1", "true", "yes", "on"}
    if force_role_fonts or "fontsize" not in patched:
        patched["fontsize"] = _role_font_size(role)

    return patched


# 関数: `_apply_role_fontdict` の入出力契約と処理意図を定義する。

def _apply_role_fontdict(fontdict: Any, *, role: str) -> Any:
    if fontdict is None:
        return None

    try:
        patched = dict(fontdict)
    except Exception:
        return fontdict

    force_role_fonts = str(os.getenv("WAVEP_MPL_FORCE_ROLE_FONTS", "")).strip().lower() in {"1", "true", "yes", "on"}
    if force_role_fonts or "fontsize" not in patched:
        patched["fontsize"] = _role_font_size(role)

    return patched


# 関数: `_apply_role_font_value` の入出力契約と処理意図を定義する。

def _apply_role_font_value(value: Any, *, role: str) -> float:
    return _role_font_size(role)


# 関数: `_set_texts_role_fontsize` の入出力契約と処理意図を定義する。

def _set_texts_role_fontsize(texts: Any, *, role: str) -> None:
    target_size = _role_font_size(role)
    try:
        iterable = list(texts)
    except Exception:
        iterable = []

    for text in iterable:
        try:
            text.set_fontsize(target_size)
        except Exception:
            continue


# 関数: `_push_figure_text_role` の入出力契約と処理意図を定義する。

def _push_figure_text_role(role: str) -> None:
    _FIGURE_TEXT_ROLE_STACK.append(str(role).strip().lower())


# 関数: `_pop_figure_text_role` の入出力契約と処理意図を定義する。

def _pop_figure_text_role() -> None:
    if _FIGURE_TEXT_ROLE_STACK:
        _FIGURE_TEXT_ROLE_STACK.pop()


# 関数: `_current_figure_text_role` の入出力契約と処理意図を定義する。

def _current_figure_text_role(default: str = "note") -> str:
    if _FIGURE_TEXT_ROLE_STACK:
        return _FIGURE_TEXT_ROLE_STACK[-1]

    return default


# 関数: `install_wavep_font_profile` の入出力契約と処理意図を定義する。

def install_wavep_font_profile(*, profile_name: str | None = None, scale: float | None = None) -> dict[str, float]:
    """
    build 時の共通 font profile をインストールする。

    役割別の font 下限を Matplotlib API に差し込み、
    各スクリプトで `fontsize=` を細かく書かなくても見た目を一定へ寄せる。
    既に patch 済みでも、profile と scale の状態は更新できる。
    """
    resolved_name = _resolve_wavep_font_profile_name(profile_name)
    resolved_scale = float(scale if scale is not None else _coerce_float_env(os.getenv("WAVEP_MPL_FONT_SCALE", "1.0"), default=1.0))
    sizes = _update_font_profile_state(profile_name=resolved_name, scale=resolved_scale)
    _update_line_profile_state(profile_name=resolved_name, scale=resolved_scale)
    apply_paper_style()

    global _ROLE_FONT_PATCHED
    if _ROLE_FONT_PATCHED:
        return sizes

    from matplotlib.axes import Axes

    original_set_title = Axes.set_title
    original_set_xlabel = Axes.set_xlabel
    original_set_ylabel = Axes.set_ylabel
    original_tick_params = Axes.tick_params
    original_legend = Axes.legend
    original_text = Axes.text
    original_annotate = Axes.annotate
    original_plot = Axes.plot
    original_errorbar = Axes.errorbar
    original_axhline = Axes.axhline
    original_axvline = Axes.axvline
    original_fig_text = Figure.text
    original_suptitle = Figure.suptitle

    # 関数: `patched_set_title` の入出力契約と処理意図を定義する。
    def patched_set_title(self, label, fontdict=None, loc=None, pad=None, *, y=None, **kwargs):
        return original_set_title(
            self,
            label,
            fontdict=_apply_role_fontdict(fontdict, role="title"),
            loc=loc,
            pad=pad,
            y=y,
            **_apply_role_font_kwargs(kwargs, role="title"),
        )

    # 関数: `patched_set_xlabel` の入出力契約と処理意図を定義する。

    def patched_set_xlabel(self, xlabel, fontdict=None, labelpad=None, *, loc=None, **kwargs):
        return original_set_xlabel(
            self,
            xlabel,
            fontdict=_apply_role_fontdict(fontdict, role="axis"),
            labelpad=labelpad,
            loc=loc,
            **_apply_role_font_kwargs(kwargs, role="axis"),
        )

    # 関数: `patched_set_ylabel` の入出力契約と処理意図を定義する。

    def patched_set_ylabel(self, ylabel, fontdict=None, labelpad=None, *, loc=None, **kwargs):
        return original_set_ylabel(
            self,
            ylabel,
            fontdict=_apply_role_fontdict(fontdict, role="axis"),
            labelpad=labelpad,
            loc=loc,
            **_apply_role_font_kwargs(kwargs, role="axis"),
        )

    # 関数: `patched_tick_params` の入出力契約と処理意図を定義する。

    def patched_tick_params(self, axis="both", **kwargs):
        patched = dict(kwargs)
        patched["labelsize"] = _apply_role_font_value(patched.get("labelsize"), role="tick")
        return original_tick_params(self, axis=axis, **patched)

    # 関数: `patched_legend` の入出力契約と処理意図を定義する。

    def patched_legend(self, *args, **kwargs):
        patched_kwargs = _apply_role_font_kwargs(kwargs, role="legend")
        force_role_fonts = str(os.getenv("WAVEP_MPL_FORCE_ROLE_FONTS", "")).strip().lower() in {"1", "true", "yes", "on"}
        if force_role_fonts and isinstance(patched_kwargs.get("prop"), dict):
            patched_prop = dict(patched_kwargs["prop"])
            patched_prop["size"] = float(_current_role_sizes()["legend"])
            patched_kwargs["prop"] = patched_prop

        return original_legend(self, *args, **patched_kwargs)

    # 関数: `patched_text` の入出力契約と処理意図を定義する。

    def patched_text(self, *args, **kwargs):
        return original_text(self, *args, **_apply_role_font_kwargs(kwargs, role="note"))

    # 関数: `patched_annotate` の入出力契約と処理意図を定義する。

    def patched_annotate(self, *args, **kwargs):
        return original_annotate(self, *args, **_apply_role_font_kwargs(kwargs, role="note"))

    # 関数: `patched_plot` の入出力契約と処理意図を定義する。

    def patched_plot(self, *args, **kwargs):
        return original_plot(self, *args, **_apply_plot_line_kwargs(kwargs, kind="line"))

    # 関数: `patched_errorbar` の入出力契約と処理意図を定義する。

    def patched_errorbar(self, *args, **kwargs):
        return original_errorbar(self, *args, **_apply_errorbar_line_kwargs(kwargs))

    # 関数: `patched_axhline` の入出力契約と処理意図を定義する。

    def patched_axhline(self, y=0, xmin=0, xmax=1, **kwargs):
        return original_axhline(self, y=y, xmin=xmin, xmax=xmax, **_apply_plot_line_kwargs(kwargs, kind="reference"))

    # 関数: `patched_axvline` の入出力契約と処理意図を定義する。

    def patched_axvline(self, x=0, ymin=0, ymax=1, **kwargs):
        return original_axvline(self, x=x, ymin=ymin, ymax=ymax, **_apply_plot_line_kwargs(kwargs, kind="reference"))

    # 関数: `patched_fig_text` の入出力契約と処理意図を定義する。

    def patched_fig_text(self, *args, **kwargs):
        role = _current_figure_text_role("note")
        return original_fig_text(self, *args, **_apply_role_font_kwargs(kwargs, role=role))

    # 関数: `patched_suptitle` の入出力契約と処理意図を定義する。

    def patched_suptitle(self, t, **kwargs):
        _push_figure_text_role("suptitle")
        try:
            return original_suptitle(self, t, **_apply_role_font_kwargs(kwargs, role="suptitle"))
        finally:
            _pop_figure_text_role()

    original_set_xticklabels = Axes.set_xticklabels
    original_set_yticklabels = Axes.set_yticklabels

    # 関数: `patched_set_xticklabels` の入出力契約と処理意図を定義する。
    def patched_set_xticklabels(self, labels, *args, **kwargs):
        texts = original_set_xticklabels(
            self,
            labels,
            *args,
            **_apply_role_font_kwargs(kwargs, role="tick"),
        )
        _set_texts_role_fontsize(texts, role="tick")
        return texts

    # 関数: `patched_set_yticklabels` の入出力契約と処理意図を定義する。

    def patched_set_yticklabels(self, labels, *args, **kwargs):
        texts = original_set_yticklabels(
            self,
            labels,
            *args,
            **_apply_role_font_kwargs(kwargs, role="tick"),
        )
        _set_texts_role_fontsize(texts, role="tick")
        return texts

    Axes.set_title = patched_set_title
    Axes.set_xlabel = patched_set_xlabel
    Axes.set_ylabel = patched_set_ylabel
    Axes.tick_params = patched_tick_params
    Axes.set_xticklabels = patched_set_xticklabels
    Axes.set_yticklabels = patched_set_yticklabels
    Axes.legend = patched_legend
    Axes.text = patched_text
    Axes.annotate = patched_annotate
    Axes.plot = patched_plot
    Axes.errorbar = patched_errorbar
    Axes.axhline = patched_axhline
    Axes.axvline = patched_axvline
    Figure.text = patched_fig_text
    Figure.suptitle = patched_suptitle
    _ROLE_FONT_PATCHED = True
    return sizes

# 関数: `save_paper_figure` の入出力契約と処理意図を定義する。

def save_paper_figure(fig: Figure, filepath: str) -> str:
    """
    見切れを完全に防止し、PDF形式（ベクター）で図を保存する統合ラッパー。
    
    引数:
        fig: matplotlibのFigureオブジェクト
        filepath: 保存先のパス（拡張子が.png等の場合は自動で.pdfに置換します）
    """
    # グラフ本体とラベルの間隔を自動最適化
    fig.tight_layout()
    
    # 既存コードとの互換性のため、拡張子を強制的に .pdf に変更
    base_name, ext = os.path.splitext(filepath)
    if ext.lower() != '.pdf':
        filepath = base_name + '.pdf'

    # 出力先ディレクトリが無い場合でも保存できるように事前作成する。

    out_dir = os.path.dirname(filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # PDF（ベクター形式）で保存
    # ※dpi指定はベクターには本来不要ですが、図の中に複雑な散布図等があり
    # 一部がラスタライズ（画像化）されるケースに備えて300dpiを残しています。

    fig.savefig(filepath, dpi=300, bbox_inches="tight", format="pdf")
    
    # メモリリーク防止のため明示的に閉じる
    plt.close(fig)
    return filepath


_FONT_FLOOR_PATCHED = False


# 関数: `_coerce_numeric_fontsize` の入出力契約と処理意図を定義する。
def _coerce_numeric_fontsize(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    try:
        return float(str(value).strip())
    except Exception:
        return None


# 関数: `_apply_fontsize_floor_kwargs` の入出力契約と処理意図を定義する。

def _apply_fontsize_floor_kwargs(kwargs: dict[str, Any], *, floor: float) -> dict[str, Any]:
    patched = dict(kwargs)
    for key in ("fontsize", "size"):
        if key not in patched:
            continue

        current = _coerce_numeric_fontsize(patched.get(key))
        if current is not None and current < floor:
            patched[key] = floor

    return patched


# 関数: `install_legend_note_font_floor` の入出力契約と処理意図を定義する。

def install_legend_note_font_floor(*, min_fontsize: float = 11.0) -> None:
    """
    凡例(`legend`)と注記(`text`/`annotate`)のフォントサイズだけに下限を設ける。
    既存グラフのサイズ・軸スケールは変更しない。

    本関数は monkey patch を一度だけ適用する（冪等）。
    """
    global _FONT_FLOOR_PATCHED
    # 条件分岐: `_FONT_FLOOR_PATCHED` を満たす経路を評価する。
    if _FONT_FLOOR_PATCHED:
        return

    from matplotlib.axes import Axes

    floor = float(min_fontsize)
    original_legend = Axes.legend
    original_text = Axes.text
    original_annotate = Axes.annotate
    original_fig_text = Figure.text

    # 関数: `patched_legend` の入出力契約と処理意図を定義する。
    def patched_legend(self, *args, **kwargs):
        return original_legend(self, *args, **_apply_fontsize_floor_kwargs(kwargs, floor=floor))

    # 関数: `patched_text` の入出力契約と処理意図を定義する。

    def patched_text(self, *args, **kwargs):
        return original_text(self, *args, **_apply_fontsize_floor_kwargs(kwargs, floor=floor))

    # 関数: `patched_annotate` の入出力契約と処理意図を定義する。

    def patched_annotate(self, *args, **kwargs):
        return original_annotate(self, *args, **_apply_fontsize_floor_kwargs(kwargs, floor=floor))

    # 関数: `patched_fig_text` の入出力契約と処理意図を定義する。

    def patched_fig_text(self, *args, **kwargs):
        return original_fig_text(self, *args, **_apply_fontsize_floor_kwargs(kwargs, floor=floor))

    Axes.legend = patched_legend
    Axes.text = patched_text
    Axes.annotate = patched_annotate
    Figure.text = patched_fig_text

    _FONT_FLOOR_PATCHED = True
