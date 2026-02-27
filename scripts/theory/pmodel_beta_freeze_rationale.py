from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR_PUBLIC = ROOT / "output" / "theory"
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

        preferred = [
            "Yu Gothic",
            "Meiryo",
            "BIZ UDGothic",
            "MS Gothic",
            "Yu Mincho",
            "MS Mincho",
        ]
        available = {font.name for font in fm.fontManager.ttflist}
        selected = [name for name in preferred if name in available]
        # 条件分岐: `selected` を満たす経路を評価する。
        if selected:
            mpl.rcParams["font.family"] = selected + ["DejaVu Sans"]

        mpl.rcParams["axes.unicode_minus"] = False
        mpl.rcParams["font.size"] = 13.0
        mpl.rcParams["axes.titlesize"] = 18.0
        mpl.rcParams["axes.labelsize"] = 14.0
        mpl.rcParams["xtick.labelsize"] = 12.0
        mpl.rcParams["ytick.labelsize"] = 12.0
        mpl.rcParams["legend.fontsize"] = 12.0
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
def _save_figure(fig: Any, public_path: Path, private_path: Path) -> None:
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(public_path, dpi=220, bbox_inches="tight")
    fig.savefig(private_path, dpi=220, bbox_inches="tight")


# 関数: `main` の入出力契約と処理意図を定義する。
def main() -> int:
    _set_japanese_font()

    out_png_public = OUT_DIR_PUBLIC / "pmodel_beta_freeze_rationale.png"
    out_png_private = OUT_DIR_PRIVATE / "pmodel_beta_freeze_rationale.png"
    out_json_private = OUT_DIR_PRIVATE / "pmodel_beta_freeze_rationale_metrics.json"

    rows = _build_rows()
    labels = [row["label"] for row in rows]
    values = np.array([float(row["beta"]) for row in rows], dtype=float)
    errors = np.array([float(row["sigma"]) for row in rows], dtype=float)
    values_shift = values - 1.0

    freeze_value = float(rows[2]["beta"])
    freeze_shift = freeze_value - 1.0
    z_scores = np.abs(values - freeze_value) / np.where(errors > 0, errors, 1.0)

    figure, axes = plt.subplots(1, 2, figsize=(16.4, 8.6), dpi=220, gridspec_kw={"width_ratios": [2.45, 1.10]})

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
    ax0.set_yticklabels(labels)
    ax0.invert_yaxis()
    ax0.set_xlabel("β - 1 （unity offset）", fontsize=15.0)
    ax0.set_title("β凍結の根拠（Cassini拘束 + VLBI独立チェック）", fontsize=19.0, pad=12.0)
    ax0.grid(alpha=0.25, axis="x")
    ax0.tick_params(axis="both", labelsize=13.0)
    ax0.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02), fontsize=12.0, frameon=True)
    ax0.text(
        0.02,
        0.08,
        f"β_frozen={freeze_value:.7f}",
        transform=ax0.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.0,
        color="tab:orange",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.4},
    )
    span = float(np.max(np.abs(values_shift) + errors))
    ax0.set_xlim(min(freeze_shift - 1.2 * span, np.min(values_shift - errors) - 0.05 * span), np.max(values_shift + errors) + 0.05 * span)

    for index, row in enumerate(rows):
        ax0.text(
            values_shift[index] + max(errors[index], 3.0e-5) * 1.25,
            positions[index] - 0.05,
            row["note"],
            fontsize=11.3,
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
    ax1.set_xlabel(r"$|\beta-\beta_{\mathrm{frozen}}|/\sigma$", fontsize=15.0)
    ax1.set_title("凍結値との一貫性", fontsize=19.0, pad=12.0)
    ax1.grid(alpha=0.25, axis="x")
    ax1.tick_params(axis="both", labelsize=13.0)
    ax1.legend(loc="lower right", fontsize=12.0)
    x_max = max(3.4, float(np.nanmax(z_scores)) + 0.4)
    ax1.set_xlim(0.0, x_max)

    for index, score in enumerate(z_scores):
        ax1.text(score + 0.05, positions[index], f"{score:.2f}σ", va="center", fontsize=12.0, color="0.22")

    figure.tight_layout(rect=(0.03, 0.01, 1.0, 1.0))
    _save_figure(figure, public_path=out_png_public, private_path=out_png_private)
    plt.close(figure)

    payload = {
        "generated_utc": _utc_now(),
        "script": "scripts/theory/pmodel_beta_freeze_rationale.py",
        "outputs": {
            "png_public": str(out_png_public).replace("\\", "/"),
            "png_private": str(out_png_private).replace("\\", "/"),
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

    print(f"[ok] png(public) : {out_png_public}")
    print(f"[ok] png(private): {out_png_private}")
    print(f"[ok] json        : {out_json_private}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。
if __name__ == "__main__":
    raise SystemExit(main())
