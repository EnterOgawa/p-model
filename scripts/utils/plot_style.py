"""
P-model 論文用 Matplotlib プロットスタイル定義 (PDF出力最適化版)
scripts/utils/plot_style.py

学術論文の印刷品質（ベクターデータ、見切れ防止、フォント埋め込み）を
すべての検証スクリプトで一貫して保証するためのモジュールです。
"""

import matplotlib.pyplot as plt
import os
from matplotlib.figure import Figure
from typing import Any

# 関数: `apply_paper_style` の入出力契約と処理意図を定義する。
def apply_paper_style() -> None:
    """
    P-model論文用の学術的なMatplotlibグローバル設定を適用する。
    各スクリプトの冒頭で一度呼び出すだけで、以降のプロットすべてに適用される。
    """
    plt.rcParams.update({
        "font.family": "sans-serif", # 論文向けにスッキリしたフォント
        "font.size": 14,          # 全体の基本フォントサイズ
        "axes.titlesize": 16,     # グラフタイトルのサイズ
        "axes.labelsize": 14,     # X軸・Y軸のラベルサイズ
        "xtick.labelsize": 12,    # X軸の目盛り文字サイズ
        "ytick.labelsize": 12,    # Y軸の目盛り文字サイズ
        "legend.fontsize": 12,    # 凡例の文字サイズ
        "figure.titlesize": 16,   # Figure全体のタイトルサイズ
        "lines.linewidth": 2.0,   # 線の太さ
        "lines.markersize": 6.0,  # マーカーのサイズ
        
        # --- PDF出力向けの最適化設定 ---
        "pdf.fonttype": 42,       # フォントをアウトライン化せず埋め込む（論文PDFでの文字検索を可能にする鉄則）
        "ps.fonttype": 42,
        "savefig.format": "pdf",  # デフォルトの保存形式をPDFに固定
        "savefig.bbox": "tight",  # 見切れ防止
        "savefig.pad_inches": 0.1 
    })

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
