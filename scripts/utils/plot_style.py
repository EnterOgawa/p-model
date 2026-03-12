"""
P-model 論文用 Matplotlib プロットスタイル定義 (PDF出力最適化版)
scripts/utils/plot_style.py

学術論文の印刷品質（ベクターデータ、見切れ防止、フォント埋め込み）を
すべての検証スクリプトで一貫して保証するためのモジュールです。

加えて、`paper_build.py` から `WAVEP_MPL_FONT_PROFILE` を渡された場合は、
役割別（title / axis / tick / legend / note / suptitle）の共通サイズを
build 時に一括適用できるようにする。
"""

import matplotlib.pyplot as plt
import os
from matplotlib.figure import Figure
from typing import Any


_WAVEP_FONT_PROFILES: dict[str, dict[str, float]] = {
    "paper": {
        "base": 12.0,
        "title": 15.2,
        "axis": 13.8,
        "tick": 12.2,
        "legend": 12.2,
        "note": 12.2,
        "suptitle": 16.2,
    },
    "part2_astrophysics": {
        "base": 12.8,
        "title": 16.4,
        "axis": 14.6,
        "tick": 13.2,
        "legend": 13.2,
        "note": 13.2,
        "suptitle": 17.4,
    },
    "part3_quantum": {
        "base": 12.0,
        "title": 15.4,
        "axis": 13.8,
        "tick": 12.2,
        "legend": 12.2,
        "note": 12.2,
        "suptitle": 16.2,
    },
    "part4_verification": {
        "base": 14.0,
        "title": 17.0,
        "axis": 15.4,
        "tick": 14.0,
        "legend": 14.0,
        "note": 14.0,
        "suptitle": 18.0,
    },
    "part5_future_predictions": {
        "base": 12.6,
        "title": 16.0,
        "axis": 14.4,
        "tick": 12.8,
        "legend": 12.8,
        "note": 12.8,
        "suptitle": 17.0,
    },
}

_FONT_PROFILE_STATE: dict[str, Any] = {
    "profile_name": "paper",
    "scale": 1.0,
    "sizes": dict(_WAVEP_FONT_PROFILES["paper"]),
}

_ROLE_FONT_PATCHED = False

# 関数: `apply_paper_style` の入出力契約と処理意図を定義する。
def apply_paper_style() -> None:
    """
    P-model論文用の学術的なMatplotlibグローバル設定を適用する。
    各スクリプトの冒頭で一度呼び出すだけで、以降のプロットすべてに適用される。
    """
    sizes = get_wavep_font_profile()
    plt.rcParams.update({
        "font.family": "sans-serif", # 論文向けにスッキリしたフォント
        "font.size": sizes["base"],          # 全体の基本フォントサイズ
        "axes.titlesize": sizes["title"],    # グラフタイトルのサイズ
        "axes.labelsize": sizes["axis"],     # X軸・Y軸のラベルサイズ
        "xtick.labelsize": sizes["tick"],    # X軸の目盛り文字サイズ
        "ytick.labelsize": sizes["tick"],    # Y軸の目盛り文字サイズ
        "legend.fontsize": sizes["legend"],  # 凡例の文字サイズ
        "figure.titlesize": sizes["suptitle"],   # Figure全体のタイトルサイズ
        "lines.linewidth": 2.0,   # 線の太さ
        "lines.markersize": 6.0,  # マーカーのサイズ
        
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


# 関数: `_current_role_sizes` の入出力契約と処理意図を定義する。
def _current_role_sizes() -> dict[str, float]:
    return dict(_FONT_PROFILE_STATE["sizes"])


# 関数: `_update_font_profile_state` の入出力契約と処理意図を定義する。
def _update_font_profile_state(*, profile_name: str, scale: float) -> dict[str, float]:
    sizes = get_wavep_font_profile(name=profile_name, scale=scale)
    _FONT_PROFILE_STATE["profile_name"] = profile_name
    _FONT_PROFILE_STATE["scale"] = float(scale)
    _FONT_PROFILE_STATE["sizes"] = dict(sizes)
    return sizes


# 関数: `_apply_role_floor_kwargs` の入出力契約と処理意図を定義する。
def _apply_role_floor_kwargs(kwargs: dict[str, Any], *, role: str) -> dict[str, Any]:
    patched = dict(kwargs)
    sizes = _current_role_sizes()
    floor = float(sizes[role])
    current = _coerce_numeric_fontsize(patched.get("fontsize"))
    if current is None:
        patched["fontsize"] = floor
    elif current < floor:
        patched["fontsize"] = floor
    return patched


# 関数: `_apply_role_floor_value` の入出力契約と処理意図を定義する。
def _apply_role_floor_value(value: Any, *, role: str) -> float:
    sizes = _current_role_sizes()
    floor = float(sizes[role])
    current = _coerce_numeric_fontsize(value)
    if current is None:
        return floor
    if current < floor:
        return floor
    return float(current)


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
    original_fig_text = Figure.text
    original_suptitle = Figure.suptitle

    # 関数: `patched_set_title` の入出力契約と処理意図を定義する。
    def patched_set_title(self, label, fontdict=None, loc=None, pad=None, *, y=None, **kwargs):
        return original_set_title(
            self,
            label,
            fontdict=fontdict,
            loc=loc,
            pad=pad,
            y=y,
            **_apply_role_floor_kwargs(kwargs, role="title"),
        )

    # 関数: `patched_set_xlabel` の入出力契約と処理意図を定義する。
    def patched_set_xlabel(self, xlabel, fontdict=None, labelpad=None, *, loc=None, **kwargs):
        return original_set_xlabel(
            self,
            xlabel,
            fontdict=fontdict,
            labelpad=labelpad,
            loc=loc,
            **_apply_role_floor_kwargs(kwargs, role="axis"),
        )

    # 関数: `patched_set_ylabel` の入出力契約と処理意図を定義する。
    def patched_set_ylabel(self, ylabel, fontdict=None, labelpad=None, *, loc=None, **kwargs):
        return original_set_ylabel(
            self,
            ylabel,
            fontdict=fontdict,
            labelpad=labelpad,
            loc=loc,
            **_apply_role_floor_kwargs(kwargs, role="axis"),
        )

    # 関数: `patched_tick_params` の入出力契約と処理意図を定義する。
    def patched_tick_params(self, axis="both", **kwargs):
        patched = dict(kwargs)
        patched["labelsize"] = _apply_role_floor_value(patched.get("labelsize"), role="tick")
        return original_tick_params(self, axis=axis, **patched)

    # 関数: `patched_legend` の入出力契約と処理意図を定義する。
    def patched_legend(self, *args, **kwargs):
        return original_legend(self, *args, **_apply_role_floor_kwargs(kwargs, role="legend"))

    # 関数: `patched_text` の入出力契約と処理意図を定義する。
    def patched_text(self, *args, **kwargs):
        return original_text(self, *args, **_apply_role_floor_kwargs(kwargs, role="note"))

    # 関数: `patched_annotate` の入出力契約と処理意図を定義する。
    def patched_annotate(self, *args, **kwargs):
        return original_annotate(self, *args, **_apply_role_floor_kwargs(kwargs, role="note"))

    # 関数: `patched_fig_text` の入出力契約と処理意図を定義する。
    def patched_fig_text(self, *args, **kwargs):
        return original_fig_text(self, *args, **_apply_role_floor_kwargs(kwargs, role="note"))

    # 関数: `patched_suptitle` の入出力契約と処理意図を定義する。
    def patched_suptitle(self, t, **kwargs):
        return original_suptitle(self, t, **_apply_role_floor_kwargs(kwargs, role="suptitle"))

    Axes.set_title = patched_set_title
    Axes.set_xlabel = patched_set_xlabel
    Axes.set_ylabel = patched_set_ylabel
    Axes.tick_params = patched_tick_params
    Axes.legend = patched_legend
    Axes.text = patched_text
    Axes.annotate = patched_annotate
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
