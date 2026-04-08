"""
目的: 理論 topic の pmodel beta freeze rationale に対応する公開図・表・監査指標を再生成する。
入力: script 内の既定パラメータと必要な公開データまたは基準値を用いる。
出力: output/public と output/private の canonical artifact を更新する。
前提: 論文本文と README はこの script が出力する公開成果物を正として参照する。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scripts.utils.plot_style import (
    apply_paper_style,
    apply_wavep_figure_layout,
    resolve_wavep_cjk_font_family,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR_CANON = ROOT / "output" / "theory"
OUT_DIR_PUBLIC = ROOT / "output" / "public" / "theory"
OUT_DIR_PRIVATE = ROOT / "output" / "private" / "theory"
FROZEN_JSON = OUT_DIR_PRIVATE / "frozen_parameters.json"
SOLAR_JSON = OUT_DIR_PRIVATE / "solar_light_deflection_metrics.json"


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。

def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = resolve_wavep_cjk_font_family(preferred_name="Noto Sans CJK JP")
        if preferred:
            mpl.rcParams["font.family"] = [preferred, "DejaVu Sans"]
            mpl.rcParams["font.sans-serif"] = [preferred, "DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
            return

        available = {font.name for font in fm.fontManager.ttflist}
        fallback = ["Yu Gothic", "Meiryo", "BIZ UDGothic", "MS Gothic", "Yu Mincho", "MS Mincho"]
        selected = [name for name in fallback if name in available]
        if selected:
            mpl.rcParams["font.family"] = selected + ["DejaVu Sans"]
            mpl.rcParams["font.sans-serif"] = selected + ["DejaVu Sans"]

        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        return


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `_beta_from_gamma` の入出力契約と処理意図を定義する。

def _beta_from_gamma(gamma: float, sigma_gamma: float) -> Tuple[float, float]:
    beta = 0.5 * (1.0 + float(gamma))
    sigma_beta = 0.5 * abs(float(sigma_gamma))
    return beta, sigma_beta


# 関数: `_load_freeze_payload` の入出力契約と処理意図を定義する。

def _load_freeze_payload() -> Tuple[float, float, str]:
    # 条件分岐: `FROZEN_JSON.exists()` を満たす経路を評価する。
    if FROZEN_JSON.exists():
        payload = _read_json(FROZEN_JSON)
        beta = float(payload.get("beta", 1.0000105))
        sigma = float(payload.get("beta_sigma", 1.15e-5))
        source = str(((payload.get("policy") or {}).get("beta_source")) or "cassini2003")
        return beta, sigma, source

    return 1.0000105, 1.15e-5, "fallback_cassini2003"


# 関数: `_load_vlbi_best` の入出力契約と処理意図を定義する。

def _load_vlbi_best() -> Tuple[float, float, str]:
    # 条件分岐: `SOLAR_JSON.exists()` を満たす経路を評価する。
    if SOLAR_JSON.exists():
        payload = _read_json(SOLAR_JSON)
        metrics = dict(payload.get("metrics") or {})
        gamma = float(metrics.get("observed_gamma_best", 0.99983))
        sigma = float(metrics.get("observed_gamma_best_sigma", 0.00026))
        label = str(metrics.get("observed_best_label") or "VLBI（best）")
        beta, beta_sigma = _beta_from_gamma(gamma, sigma)
        return beta, beta_sigma, label

    beta, beta_sigma = _beta_from_gamma(0.99983, 0.00026)
    return beta, beta_sigma, "VLBI（fallback）"


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows() -> List[Dict[str, Any]]:
    cassini_beta, cassini_sigma = _beta_from_gamma(1.000021, 2.3e-5)
    vlbi_beta, vlbi_sigma, vlbi_label = _load_vlbi_best()
    frozen_beta, frozen_sigma, frozen_source = _load_freeze_payload()
    return [
        {
            "id": "cassini_constraint",
            "label": "Cassini β拘束（一次）",
            "beta": cassini_beta,
            "sigma": cassini_sigma,
            "note": "γ=1.000021±0.000023 から β=(1+γ)/2",
        },
        {
            "id": "vlbi_cross_check",
            "label": f"VLBI 独立チェック（{vlbi_label}）",
            "beta": vlbi_beta,
            "sigma": vlbi_sigma,
            "note": "太陽光偏向の独立測定",
        },
        {
            "id": "frozen_beta",
            "label": "Part I 凍結 β（以後固定）",
            "beta": frozen_beta,
            "sigma": frozen_sigma,
            "note": f"frozen source: {frozen_source}",
        },
    ]


# 関数: `_save_figure` の入出力契約と処理意図を定義する。

def _save_figure(fig: Any, stem: str) -> Dict[str, str]:
    outputs = {
        "png_canon": OUT_DIR_CANON / f"{stem}.png",
        "png_public": OUT_DIR_PUBLIC / f"{stem}.png",
        "png_private": OUT_DIR_PRIVATE / f"{stem}.png",
        "pdf_canon": OUT_DIR_CANON / f"{stem}.pdf",
        "pdf_public": OUT_DIR_PUBLIC / f"{stem}.pdf",
        "pdf_private": OUT_DIR_PRIVATE / f"{stem}.pdf",
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        for path in outputs.values():
            if path.suffix.lower() == ".png":
                fig.savefig(path, dpi=220)
            else:
                fig.savefig(path)

    return {key: str(value).replace("\\", "/") for key, value in outputs.items()}


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    apply_paper_style()
    _set_japanese_font()
    out_json_private = OUT_DIR_PRIVATE / "pmodel_beta_freeze_rationale_metrics.json"

    rows = _build_rows()
    labels = [row["label"] for row in rows]
    display_labels = ["Cassini\n(一次)", "VLBI\n(Shapiro 2004)", "凍結値"]
    values = np.array([float(row["beta"]) for row in rows], dtype=float)
    errors = np.array([float(row["sigma"]) for row in rows], dtype=float)
    values_shift = values - 1.0

    freeze_value = float(rows[2]["beta"])
    freeze_shift = freeze_value - 1.0
    z_scores = np.abs(values - freeze_value) / np.where(errors > 0, errors, 1.0)

    figure, axes = plt.subplots(2, 1, dpi=220, gridspec_kw={"height_ratios": [1.28, 1.00]})
    apply_wavep_figure_layout(figure, template="paper_two_panel")

    ax0 = axes[0]
    positions = np.arange(len(rows))
    marker_styles = ["s", "D", "o"]
    marker_colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    for index, row in enumerate(rows):
        ax0.errorbar(
            [values_shift[index]],
            [positions[index]],
            xerr=[errors[index]],
            fmt=marker_styles[index],
            markersize=9.8,
            capsize=5,
            linewidth=1.45,
            color=marker_colors[index],
            markerfacecolor=marker_colors[index],
        )

    ax0.axvline(
        freeze_shift,
        color="tab:orange",
        linestyle="--",
        linewidth=1.4,
        label=f"β_frozen={freeze_value:.7f}",
    )
    ax0.set_yticks(positions)
    ax0.set_yticklabels(display_labels)
    ax0.invert_yaxis()
    ax0.set_xlabel("β - 1 （unity offset）", fontsize=17.0)
    title0 = ax0.set_title("β凍結の根拠（Cassini拘束 + VLBI独立チェック）", fontsize=21.0, pad=12.0)
    title0.set_fontsize(11.6)
    ax0.grid(alpha=0.25, axis="x")
    ax0.tick_params(axis="both", labelsize=15.0)
    ax0.legend(loc="upper left", fontsize=14.5, frameon=True)
    ax0.text(
        0.02,
        0.08,
        f"β_frozen={freeze_value:.7f}",
        transform=ax0.transAxes,
        ha="left",
        va="bottom",
        fontsize=14.0,
        color="tab:orange",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.4},
    )
    span = float(np.max(np.abs(values_shift) + errors))
    ax0.set_xlim(min(freeze_shift - 1.2 * span, np.min(values_shift - errors) - 0.05 * span), np.max(values_shift + errors) + 0.05 * span)

    for index, row in enumerate(rows):
        ax0.text(
            values_shift[index] + max(errors[index], 3.0e-5) * 1.15,
            positions[index] - 0.05,
            row["note"],
            fontsize=13.8,
            color="0.28",
            va="center",
        )

    ax1 = axes[1]
    colors = ["#4c78a8", "#59a14f", "#f28e2b"]
    ax1.barh(positions, z_scores, color=colors, alpha=0.9)
    ax1.axvline(1.0, color="0.35", linestyle=":", linewidth=1.0, label="1σ")
    ax1.axvline(3.0, color="0.20", linestyle="--", linewidth=1.0, label="3σ")
    ax1.set_yticks(positions)
    ax1.set_yticklabels([])
    ax1.invert_yaxis()
    ax1.set_xlabel(r"$|\beta-\beta_{\mathrm{frozen}}|/\sigma$", fontsize=17.0)
    title1 = ax1.set_title("凍結値との一貫性", fontsize=21.0, pad=12.0)
    title1.set_fontsize(11.4)
    ax1.grid(alpha=0.25, axis="x")
    ax1.tick_params(axis="both", labelsize=15.0)
    ax1.legend(loc="lower right", fontsize=14.5)
    x_max = max(3.4, float(np.nanmax(z_scores)) + 0.4)
    ax1.set_xlim(0.0, x_max)

    for index, score in enumerate(z_scores):
        ax1.text(score + 0.05, positions[index], f"{score:.2f}σ", va="center", fontsize=14.2, color="0.22")

    figure.subplots_adjust(left=0.19, right=0.985, top=0.915, bottom=0.105, hspace=0.42)
    outputs = _save_figure(figure, "pmodel_beta_freeze_rationale")
    plt.close(figure)

    payload = {
        "generated_utc": _utc_now(),
        "script": "scripts/theory/pmodel_beta_freeze_rationale.py",
        "outputs": {
            **outputs,
            "metrics_json": str(out_json_private).replace("\\", "/"),
        },
        "rows": rows,
        "derived": {
            "freeze_value": freeze_value,
            "z_scores_vs_frozen": [float(score) for score in z_scores.tolist()],
            "mapping": "beta=(1+gamma)/2",
        },
        "inputs": {
            "frozen_parameters_json": str(FROZEN_JSON).replace("\\", "/"),
            "solar_light_deflection_metrics_json": str(SOLAR_JSON).replace("\\", "/"),
        },
    }
    _write_json(out_json_private, payload)

    print(f"[ok] png(public) : {outputs['png_public']}")
    print(f"[ok] png(private): {outputs['png_private']}")
    print(f"[ok] pdf(public) : {outputs['pdf_public']}")
    print(f"[ok] pdf(private): {outputs['pdf_private']}")
    print(f"[ok] json        : {out_json_private}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
