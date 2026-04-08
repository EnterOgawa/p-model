from __future__ import annotations

import re
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from scripts.utils.plot_style import (
    apply_paper_style,
    get_wavep_font_size,
    resolve_wavep_cjk_font_family,
)


_BLACKBODY_HOLDOUT_FONT_SCALE = 1.22

_TITLE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("Blackbody Helmholtz free-energy density holdout", "黒体ヘルムホルツ自由エネルギー密度の温度帯分割監査"),
    ("Blackbody Helmholtz free-energy flux holdout", "黒体ヘルムホルツ自由エネルギー流束の温度帯分割監査"),
    ("Blackbody heat-capacity flux holdout", "黒体熱容量流束の温度帯分割監査"),
    ("Blackbody heat capacity density holdout", "黒体熱容量密度の温度帯分割監査"),
    ("Blackbody enthalpy density holdout", "黒体エンタルピー密度の温度帯分割監査"),
    ("Blackbody enthalpy flux holdout", "黒体エンタルピー流束の温度帯分割監査"),
    ("Blackbody entropy flux holdout", "黒体エントロピー流束の温度帯分割監査"),
    ("Blackbody photon density holdout", "黒体光子数密度の温度帯分割監査"),
    ("Blackbody photon flux holdout", "黒体光子流束の温度帯分割監査"),
    ("Blackbody momentum density holdout", "黒体運動量密度の温度帯分割監査"),
    ("Blackbody radiation pressure holdout", "黒体放射圧の温度帯分割監査"),
    ("Blackbody peak spectral radiance holdout", "黒体ピーク放射輝度の温度帯分割監査"),
    ("Blackbody peak frequency holdout", "黒体ピーク周波数の温度帯分割監査"),
    ("Blackbody peak wavelength holdout", "黒体ピーク波長の温度帯分割監査"),
    ("Blackbody peak ratio holdout", "黒体ピーク比の温度帯分割監査"),
    ("Blackbody radiation holdout", "黒体放射の温度帯分割監査"),
    ("Blackbody entropy holdout", "黒体エントロピーの温度帯分割監査"),
    ("Blackbody flux holdout", "黒体流束の温度帯分割監査"),
    ("Blackbody ratio holdout", "黒体比の温度帯分割監査"),
    ("Blackbody peak", "黒体ピーク"),
    ("Blackbody", "黒体"),
    ("holdout", "温度帯分割監査"),
    ("scaling across temperature bands", "温度帯をまたぐスケーリング"),
    ("across temperature bands", "温度帯をまたぐ"),
    ("spectral radiance", "放射輝度"),
    ("peak frequency", "ピーク周波数"),
    ("peak wavelength", "ピーク波長"),
    ("peak radiance", "ピーク放射輝度"),
    ("mean photon energy", "平均光子エネルギー"),
    ("photon density", "光子数密度"),
    ("photon flux", "光子流束"),
    ("free-energy", "自由エネルギー"),
    ("Helmholtz", "ヘルムホルツ"),
    ("momentum density", "運動量密度"),
    ("energy density", "エネルギー密度"),
    ("enthalpy density", "エンタルピー密度"),
    ("radiation pressure", "放射圧"),
    ("radiation", "放射"),
    ("density", "密度"),
    ("pressure", "圧力"),
    ("entropy", "エントロピー"),
    ("enthalpy", "エンタルピー"),
    ("energy", "エネルギー"),
    ("momentum", "運動量"),
    ("helmholtz free energy", "ヘルムホルツ自由エネルギー"),
    ("free energy", "自由エネルギー"),
    ("heat-capacity", "熱容量"),
    ("heat capacity", "熱容量"),
    ("per", "当たり"),
    ("ratio", "比"),
    ("flux", "流束"),
    ("const", "一定"),
    ("negative sign", "負符号"),
    ("Wien", "ウィーン"),
)


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。
def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl

        preferred = resolve_wavep_cjk_font_family(preferred_name="Noto Sans CJK JP")
        if preferred:
            mpl.rcParams["font.family"] = [preferred, "DejaVu Sans"]
            mpl.rcParams["font.sans-serif"] = [preferred, "DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


# 関数: `get_blackbody_holdout_font` の入出力契約と処理意図を定義する。

def get_blackbody_holdout_font(role: str) -> float:
    return get_wavep_font_size(role, name="part4_verification", scale=_BLACKBODY_HOLDOUT_FONT_SCALE)


# 関数: `create_blackbody_holdout_figure` の入出力契約と処理意図を定義する。

def create_blackbody_holdout_figure():
    apply_paper_style()
    _set_japanese_font()
    fig = plt.figure(figsize=(10.6, 4.2), dpi=170)
    fig.subplots_adjust(top=0.860, bottom=0.205, left=0.085, right=0.985)
    ax = fig.add_subplot(1, 1, 1)
    return fig, ax


# 関数: `translate_blackbody_split_name` の入出力契約と処理意図を定義する。

def translate_blackbody_split_name(name: str) -> str:
    mapping = {
        "A_low_to_high": "A: 低温→高温",
        "B_high_to_low": "B: 高温→低温",
    }
    return mapping.get(str(name), str(name))


# 関数: `_apply_title_replacements` の入出力契約と処理意図を定義する。

def _apply_title_replacements(text: str) -> str:
    translated = str(text)
    for before, after in _TITLE_REPLACEMENTS:
        translated = translated.replace(before, after)

    translated = re.sub(r"\s+", " ", translated).strip()
    return translated


# 関数: `translate_blackbody_holdout_title` の入出力契約と処理意図を定義する。

def translate_blackbody_holdout_title(text: str) -> str:
    translated = _apply_title_replacements(str(text))
    translated = translated.replace("(負符号)", "（負符号）")
    translated = translated.replace("(ウィーン)", "（ウィーン）")
    translated = translated.replace(" : ", ": ")
    return translated


# 関数: `translate_blackbody_holdout_ylabel` の入出力契約と処理意図を定義する。

def translate_blackbody_holdout_ylabel(text: str) -> str:
    normalized = str(text).strip().lower()
    if normalized == "test max abs(z)":
        return "テスト側の最大 |z|"

    return str(text)


# 関数: `translate_blackbody_holdout_legend_label` の入出力契約と処理意図を定義する。

def translate_blackbody_holdout_legend_label(label: str) -> str:
    raw = str(label).strip()
    fixed_abs_match = re.fullmatch(r"fixed exponent ([^ ]+) abs fit \((\d+)p\)", raw)
    if fixed_abs_match:
        exponent, params = fixed_abs_match.groups()
        return f"指数{exponent}固定の絶対値近似（{params}変数）"

    fixed_match = re.fullmatch(r"fixed exponent ([^ ]+) \((\d+)p\)", raw)
    if fixed_match:
        exponent, params = fixed_match.groups()
        return f"指数{exponent}固定（{params}変数）"

    fixed_fit_match = re.fullmatch(r"fixed exponent ([^ ]+) fit \((\d+)p\)", raw)
    if fixed_fit_match:
        exponent, params = fixed_fit_match.groups()
        return f"指数{exponent}固定（{params}変数）"

    power_abs_match = re.fullmatch(r"power-law abs fit \((\d+)p\)", raw)
    if power_abs_match:
        params = power_abs_match.group(1)
        return f"絶対値の冪乗近似（{params}変数）"

    power_match = re.fullmatch(r"power-law fit \((\d+)p\)", raw)
    if power_match:
        params = power_match.group(1)
        return f"冪乗近似（{params}変数）"

    return raw


# 関数: `apply_blackbody_holdout_axes_text` の入出力契約と処理意図を定義する。

def apply_blackbody_holdout_axes_text(ax, *, categories: Iterable[str], ylabel: str, title: str) -> None:
    tick_font = get_blackbody_holdout_font("tick")
    axis_font = get_blackbody_holdout_font("axis")
    title_font = get_blackbody_holdout_font("title")
    labels = [translate_blackbody_split_name(name) for name in categories]
    ax.set_xticklabels(labels, fontsize=tick_font)
    ax.set_ylabel(translate_blackbody_holdout_ylabel(ylabel), fontsize=axis_font)
    ax.set_title(translate_blackbody_holdout_title(title), fontsize=title_font, pad=7.0)
    ax.tick_params(axis="both", labelsize=tick_font)


# 関数: `apply_blackbody_holdout_legend` の入出力契約と処理意図を定義する。

def apply_blackbody_holdout_legend(ax, *, loc: str = "upper left") -> None:
    handles, labels = ax.get_legend_handles_labels()
    translated = [translate_blackbody_holdout_legend_label(label) for label in labels]
    ax.legend(handles, translated, loc=loc, fontsize=get_blackbody_holdout_font("legend"), frameon=True)


# 関数: `add_blackbody_display_floor_note` の入出力契約と処理意図を定義する。

def add_blackbody_display_floor_note(ax, visible_vals: np.ndarray) -> None:
    display_values = np.asarray(visible_vals, dtype=float)
    display_values = np.abs(display_values[np.isfinite(display_values)])
    if display_values.size == 0 or float(np.max(display_values)) < 0.12:
        max_text = "n/a" if display_values.size == 0 else f"{float(np.max(display_values)):.3g}"
        ax.text(
            0.5,
            0.82,
            f"全分割が表示床未満\n最大 |z| = {max_text}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=get_blackbody_holdout_font("note"),
            color="#444444",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#999999",
                "alpha": 0.94,
            },
        )
