"""
目的: 理論 topic の solar light deflection に対応する公開図・表・監査指標を再生成する。
入力: script 内の既定パラメータと必要な公開データまたは基準値を用いる。
出力: output/public と output/private の canonical artifact を更新する。
前提: 論文本文と README はこの script が出力する公開成果物を正として参照する。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.utils.plot_style import apply_wavep_figure_layout
from scripts.quantum.figure_japanese_localizer import get_figure_language
from scripts.utils.figure_locale_paths import localize_figure_output_path


# Physical constants
C = 299_792_458.0  # m/s
GM_SUN = 1.32712440018e20  # m^3/s^2
R_SUN_M = 695_700_000.0  # m
RAD_TO_ARCSEC = (180.0 / math.pi) * 3600.0


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
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


# 関数: `_plot_text` の入出力契約と処理意図を定義する。

def _plot_text(ja: str, en: str, *, lang: str) -> str:
    return ja if lang == "ja" else en


# クラス: `GammaMeasurement` の責務と境界条件を定義する。

@dataclass(frozen=True)
class GammaMeasurement:
    id: str
    short_label: str
    year: int
    gamma: float
    sigma: float
    method: str
    sigma_note: str
    source: Dict[str, Any]

    # 関数: `from_json` の入出力契約と処理意図を定義する。
    @staticmethod
    def from_json(j: Dict[str, Any]) -> "GammaMeasurement":
        return GammaMeasurement(
            id=str(j["id"]),
            short_label=str(j.get("short_label") or j["id"]),
            year=int(j["year"]),
            gamma=float(j["gamma"]),
            sigma=float(j["sigma"]),
            method=str(j.get("method") or ""),
            sigma_note=str(j.get("sigma_note") or ""),
            source=dict(j.get("source") or {}),
        )


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `_try_load_frozen_beta` の入出力契約と処理意図を定義する。

def _try_load_frozen_beta(root: Path) -> Tuple[Optional[float], str]:
    path = root / "output" / "private" / "theory" / "frozen_parameters.json"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return None, "output/private/theory/frozen_parameters.json (missing)"

    try:
        j = _read_json(path)
        beta = float(j["beta"])
        return beta, "output/private/theory/frozen_parameters.json:beta"
    except Exception:
        return None, "output/private/theory/frozen_parameters.json:beta (read failed)"


# 関数: `_write_measurements_csv` の入出力契約と処理意図を定義する。

def _write_measurements_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "short_label",
        "year",
        "gamma",
        "sigma",
        "z_score",
        "alpha_arcsec_limb",
        "alpha_sigma_arcsec_limb",
        "method",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


# 関数: `_load_measurements` の入出力契約と処理意図を定義する。

def _load_measurements(path: Optional[Path]) -> List[GammaMeasurement]:
    # 条件分岐: `not path` を満たす経路を評価する。
    if not path:
        return []

    try:
        j = _read_json(path)
        raw = j.get("measurements")
        # 条件分岐: `not isinstance(raw, list)` を満たす経路を評価する。
        if not isinstance(raw, list):
            return []

        return [GammaMeasurement.from_json(x) for x in raw if isinstance(x, dict)]
    except Exception:
        return []


# 関数: `deflection_arcsec` の入出力契約と処理意図を定義する。

def deflection_arcsec(beta: float, impact_parameter_m: float) -> float:
    # Weak-field ray deflection (from Phase 1 derivation):
    # alpha ≈ 4*beta*GM/(c^2 b)
    alpha_rad = (4.0 * beta * GM_SUN) / (C * C * impact_parameter_m)
    return alpha_rad * RAD_TO_ARCSEC


# 関数: `compute` の入出力契約と処理意図を定義する。

def compute(beta: float, measurements: List[GammaMeasurement]) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    b_over = np.linspace(1.0, 10.0, 400)
    alpha_arcsec = np.array([deflection_arcsec(beta, bo * R_SUN_M) for bo in b_over], dtype=float)

    alpha_limb = float(deflection_arcsec(beta, R_SUN_M))
    # GR weak-field prediction at the solar limb (gamma=1, beta=1): ~1.75 arcsec (textbook value).
    alpha_gr_limb = float(deflection_arcsec(1.0, R_SUN_M))

    gamma_pmodel = float(2.0 * beta - 1.0)

    obs_rows: List[Dict[str, Any]] = []
    for m in measurements:
        z_score = None
        # 条件分岐: `m.sigma > 0` を満たす経路を評価する。
        if m.sigma > 0:
            z_score = (m.gamma - gamma_pmodel) / m.sigma

        # alpha = alpha_GR * (1+gamma)/2, so sigma_alpha = alpha_GR * sigma/2

        alpha_obs_limb = alpha_gr_limb * (1.0 + float(m.gamma)) / 2.0
        alpha_sigma = abs(alpha_gr_limb * float(m.sigma) / 2.0)

        obs_rows.append(
            {
                "id": m.id,
                "short_label": m.short_label,
                "year": m.year,
                "gamma": float(m.gamma),
                "sigma": float(m.sigma),
                "z_score": float(z_score) if z_score is not None else None,
                "alpha_arcsec_limb": float(alpha_obs_limb),
                "alpha_sigma_arcsec_limb": float(alpha_sigma),
                "method": m.method,
                "sigma_note": m.sigma_note,
                "source": m.source,
            }
        )

    best: Optional[Dict[str, Any]] = None
    # 条件分岐: `obs_rows` を満たす経路を評価する。
    if obs_rows:
        best = min(obs_rows, key=lambda r: float(r["sigma"]) if r.get("sigma") else float("inf"))

    metrics: Dict[str, Any] = {
        "beta": float(beta),
        "gamma_pmodel": gamma_pmodel,
        "gamma_gr": 1.0,
        "impact_parameter_Rsun": 1.0,
        "alpha_pmodel_arcsec_limb": alpha_limb,
        "reference_arcsec_limb": alpha_gr_limb,
        "abs_error_arcsec": abs(alpha_limb - alpha_gr_limb),
        "rel_error": abs(alpha_limb - alpha_gr_limb) / alpha_gr_limb,
        "measurement_count": int(len(obs_rows)),
    }
    # 条件分岐: `best` を満たす経路を評価する。
    if best:
        metrics.update(
            {
                "observed_best_id": best.get("id"),
                "observed_best_label": best.get("short_label"),
                "observed_gamma_best": best.get("gamma"),
                "observed_gamma_best_sigma": best.get("sigma"),
                "observed_alpha_arcsec_limb_best": best.get("alpha_arcsec_limb"),
                "observed_alpha_sigma_arcsec_limb_best": best.get("alpha_sigma_arcsec_limb"),
                "observed_z_score_best": best.get("z_score"),
            }
        )

    payload: Dict[str, Any] = {
        "metrics": metrics,
        "observations": obs_rows,
    }
    return payload, b_over, alpha_arcsec


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    root = _ROOT
    default_outdir = root / "output" / "private" / "theory"
    default_measurements = root / "data" / "theory" / "solar_light_deflection_measurements.json"

    ap = argparse.ArgumentParser(description="Solar light deflection check (P-model vs GR + observed gamma).")
    ap.add_argument(
        "--beta",
        type=float,
        default=None,
        help="P-model beta. If omitted, read output/private/theory/frozen_parameters.json (fallback: 1.0).",
    )
    ap.add_argument(
        "--measurements",
        type=str,
        default=str(default_measurements),
        help="JSON of observed PPN gamma measurements (default: data/theory/solar_light_deflection_measurements.json)",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(default_outdir),
        help="Output directory (default: output/private/theory)",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 条件分岐: `args.beta is not None` を満たす経路を評価する。
    if args.beta is not None:
        beta = float(args.beta)
        beta_source = "cli"
    else:
        beta, beta_source = _try_load_frozen_beta(root)
        # 条件分岐: `beta is None` を満たす経路を評価する。
        if beta is None:
            beta = 1.0
            beta_source = "default_beta_1"

    measurements = _load_measurements(Path(args.measurements) if args.measurements else None)
    meta, b_over, alpha_arcsec = compute(beta=float(beta), measurements=measurements)
    metrics = dict(meta.get("metrics") or {})
    obs_rows = list(meta.get("observations") or [])
    metrics["beta_source"] = beta_source

    figure_lang = get_figure_language(default="ja")
    if figure_lang == "ja":
        _set_japanese_font()

    if figure_lang == "en":
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(7.2, 6.45),
            gridspec_kw={"height_ratios": [1.0, 1.0]},
        )
    else:
        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(11.9, 5.1),
            gridspec_kw={"width_ratios": [1.08, 1.0]},
        )

    # Left: alpha(b)

    ax1.plot(
        b_over,
        alpha_arcsec,
        linewidth=2.2,
        label=_plot_text(
            f"P-model（β={metrics.get('beta', 1.0):g}）",
            f"P-model (β={metrics.get('beta', 1.0):g})",
            lang=figure_lang,
        ),
    )
    ax1.scatter([1.0], [float(metrics.get("alpha_pmodel_arcsec_limb", float("nan")))], color="black", s=32, zorder=3)
    ax1.axhline(
        float(metrics.get("reference_arcsec_limb", 1.75)),
        color="gray",
        linestyle="--",
            linewidth=1.25,
            label=_plot_text(
                f"標準理論（GR, γ=1）: {float(metrics.get('reference_arcsec_limb', 1.75)):.3f} 角秒（太陽縁）",
                f"Standard GR: {float(metrics.get('reference_arcsec_limb', 1.75)):.3f} arcsec at the solar limb",
                lang=figure_lang,
            ),
        )
    # 条件分岐: `metrics.get("observed_alpha_arcsec_limb_best") is not None and metrics.get("o...` を満たす経路を評価する。
    if metrics.get("observed_alpha_arcsec_limb_best") is not None and metrics.get("observed_alpha_sigma_arcsec_limb_best") is not None:
        label = str(metrics.get("observed_best_label") or metrics.get("observed_best_id") or _plot_text("観測", "Observed", lang=figure_lang))
        if figure_lang == "en":
            label = label.replace("（", "(").replace("）", ")")

        ax1.errorbar(
            [1.0],
            [float(metrics["observed_alpha_arcsec_limb_best"])],
            yerr=[float(metrics["observed_alpha_sigma_arcsec_limb_best"])],
            fmt="o",
            color="tab:red",
            capsize=3,
            markersize=5.2,
            label=_plot_text(f"観測（代表: {label}）", f"Observed (best: {label})", lang=figure_lang),
        )

    ax1.set_xlim(1.0, 10.0)
    ax1.set_xlabel(_plot_text("インパクトパラメータ b [太陽半径 R_sun]", "Impact parameter b [R_sun]", lang=figure_lang), fontsize=12.2)
    ax1.set_ylabel(_plot_text("偏向角 α [角秒]", "Deflection angle α [arcsec]", lang=figure_lang), fontsize=15.0)
    ax1.set_title(_plot_text("太陽重力による光の偏向（弱場）", "Deflection curve", lang=figure_lang), fontsize=13.8, pad=6.0)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="both", labelsize=11.0)
    ax1.legend(fontsize=8.6, loc="upper right", frameon=True)

    # Right: observed gamma
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.25, label=_plot_text("標準理論（GR）: γ=1", "Standard theory (GR): γ=1", lang=figure_lang))
    # 条件分岐: `metrics.get("gamma_pmodel") is not None` を満たす経路を評価する。
    if metrics.get("gamma_pmodel") is not None:
        ax2.axhline(
            float(metrics["gamma_pmodel"]),
            color="tab:blue",
            linestyle="-",
            linewidth=1.25,
            label=_plot_text(
                f"P-model 予測: γ=2β-1 = {float(metrics['gamma_pmodel']):.6f}",
                f"P-model prediction: γ = 2β - 1 = {float(metrics['gamma_pmodel']):.6f}",
                lang=figure_lang,
            ),
        )

    # 条件分岐: `obs_rows` を満たす経路を評価する。

    if obs_rows:
        for r in obs_rows:
            # 条件分岐: `r.get("year") is None or r.get("gamma") is None or r.get("sigma") is None` を満たす経路を評価する。
            if r.get("year") is None or r.get("gamma") is None or r.get("sigma") is None:
                continue

            ax2.errorbar(
                [int(r["year"])],
                [float(r["gamma"])],
                yerr=[float(r["sigma"])],
                fmt="o",
                capsize=3,
                markersize=5.2,
                label=str(r.get("short_label") or r.get("id") or "obs").replace("（", "(").replace("）", ")") if figure_lang == "en" else str(r.get("short_label") or r.get("id") or "obs"),
            )
    else:
        ax2.text(0.5, 0.5, _plot_text("観測データなし", "No observed data", lang=figure_lang), ha="center", va="center", transform=ax2.transAxes, fontsize=14.0)

    ax2.set_xlabel(_plot_text("年", "Year", lang=figure_lang), fontsize=12.2)
    ax2.set_ylabel(_plot_text("PPN γ（光偏向の強さ）", "Observed PPN γ", lang=figure_lang), fontsize=13.8)
    ax2.set_title(_plot_text("光偏向パラメータ γ（観測）", "Observed γ", lang=figure_lang), fontsize=13.8, pad=6.0)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="both", labelsize=11.0)
    ax2.legend(fontsize=8.3, loc="lower right", frameon=True)

    apply_wavep_figure_layout(fig, template="part2_two_panel")
    if figure_lang == "en":
        fig.subplots_adjust(hspace=0.36, top=0.93, bottom=0.12, left=0.14, right=0.97)
    else:
        fig.subplots_adjust(wspace=0.36, top=0.90, bottom=0.16)

    png_path = localize_figure_output_path(outdir / "solar_light_deflection.png", root=root, locale=figure_lang)
    pdf_path = localize_figure_output_path(outdir / "solar_light_deflection.pdf", root=root, locale=figure_lang)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)

    out_rows_csv = localize_figure_output_path(outdir / "solar_light_deflection_measurements.csv", root=root, locale=figure_lang)
    _write_measurements_csv(out_rows_csv, obs_rows)

    public_dir = root / "output" / "public" / "theory"
    public_dir.mkdir(parents=True, exist_ok=True)
    public_png = localize_figure_output_path(public_dir / png_path.name, root=root, locale=figure_lang)
    public_pdf = localize_figure_output_path(public_dir / pdf_path.name, root=root, locale=figure_lang)
    public_csv = localize_figure_output_path(public_dir / out_rows_csv.name, root=root, locale=figure_lang)
    shutil.copy2(png_path, public_png)
    shutil.copy2(pdf_path, public_pdf)
    shutil.copy2(out_rows_csv, public_csv)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": "P-model",
        "formula": {
            "alpha_arcsec": "alpha_arcsec = 4*beta*GM_sun/(c^2*b) * RAD_TO_ARCSEC",
            "ppn_deflection": "alpha = 2*(1+gamma)*GM_sun/(c^2*b) (PPN), 1+gamma=2*beta",
        },
        "constants": {"C_m_s": C, "GM_sun_m3_s2": GM_SUN, "R_sun_m": R_SUN_M},
        "input": {"measurements_json": str(Path(args.measurements)) if args.measurements else None},
        "metrics": metrics,
        "observations": obs_rows,
        "outputs": {
            "plot_png": str(png_path),
            "plot_pdf": str(pdf_path),
            "measurements_csv": str(out_rows_csv),
            "public_plot_png": str(public_png),
            "public_plot_pdf": str(public_pdf),
            "public_measurements_csv": str(public_csv),
        },
    }
    json_path = localize_figure_output_path(outdir / "solar_light_deflection_metrics.json", root=root, locale=figure_lang)
    _write_json(json_path, payload)

    print(f"[ok] plot : {png_path}")
    print(f"[ok] plot : {pdf_path}")
    print(f"[ok] metrics: {json_path}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
