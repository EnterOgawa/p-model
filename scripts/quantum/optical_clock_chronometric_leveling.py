from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure_japanese_localizer import enable_japanese_figure_localization
from scripts.utils.figure_locale_paths import localize_figure_output_path

enable_japanese_figure_localization()

# クラス: `Config` の責務と境界条件を定義する。
@dataclass(frozen=True)
class Config:
    # From the arXiv abstract (2309.14953v3): geopotential difference in m^2 s^-2.
    delta_u_clock_m2_s2: float = 3918.1
    sigma_clock_m2_s2: float = 2.6

    delta_u_geodetic_m2_s2: float = 3915.88
    sigma_geodetic_m2_s2: float = 0.30

    # Constants
    c_m_per_s: float = 299_792_458.0
    g_m_per_s2: float = 9.80665


# 関数: `fractional_frequency_shift` の入出力契約と処理意図を定義する。

def fractional_frequency_shift(delta_u_m2_s2: float, *, c_m_per_s: float) -> float:
    # Gravitational redshift for stationary clocks: z ≡ Δf/f ≈ ΔU / c^2.
    return float(delta_u_m2_s2 / (c_m_per_s**2))


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()

    delta_u_clock = cfg.delta_u_clock_m2_s2
    sigma_clock = cfg.sigma_clock_m2_s2
    delta_u_geo = cfg.delta_u_geodetic_m2_s2
    sigma_geo = cfg.sigma_geodetic_m2_s2

    delta_u_diff = float(delta_u_clock - delta_u_geo)
    sigma_diff = float(math.sqrt(sigma_clock**2 + sigma_geo**2))
    z_score = float(delta_u_diff / sigma_diff) if sigma_diff > 0 else float("nan")

    # Convert the agreement into an "epsilon" style deviation: z_obs = (1+epsilon) ΔU_geo/c^2.
    # Here z_obs is inferred from the clock comparison, so epsilon ≈ (ΔU_clock/ΔU_geo) - 1.
    epsilon = float(delta_u_clock / delta_u_geo - 1.0)
    sigma_epsilon = float(
        math.sqrt(
            (sigma_clock / delta_u_geo) ** 2 + ((delta_u_clock * sigma_geo) / (delta_u_geo**2)) ** 2
        )
    )

    z_clock = fractional_frequency_shift(delta_u_clock, c_m_per_s=cfg.c_m_per_s)
    z_geo = fractional_frequency_shift(delta_u_geo, c_m_per_s=cfg.c_m_per_s)

    # Height-equivalent uncertainty.
    height_sigma_m = float(sigma_clock / cfg.g_m_per_s2)

    # Plot
    import matplotlib.pyplot as plt

    labels = ["chronometric (clock)", "geodetic"]
    x = np.arange(2, dtype=float)
    y = np.array([delta_u_clock, delta_u_geo], dtype=float)
    yerr = np.array([sigma_clock, sigma_geo], dtype=float)

    fig, ax = plt.subplots(figsize=(10.8, 5.4), dpi=150)
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        capsize=4,
        elinewidth=1.8,
        color="#1f77b4",
        ecolor="#1f77b4",
        label="ΔU (m²/s²)",
    )
    figure_lang = str(os.getenv("WAVEP_FIGURE_LANG", "ja")).strip().lower()
    is_en = figure_lang.startswith("en")
    tick_font = 12.4 * (0.70 if is_en else 1.0)
    axis_label_font = 13.6 * (0.68 if is_en else 1.0)
    panel_title_font = 14.8 * (0.72 if is_en else 1.0)
    legend_font = 11.8 * (0.72 if is_en else 1.0)
    note_font = 11.2 * (0.66 if is_en else 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=tick_font)
    ax.set_ylabel("geopotential difference ΔU (m²/s²)", fontsize=axis_label_font)
    ax.set_title("Optical clock chronometric leveling (arXiv:2309.14953v3; 87Sr)", fontsize=panel_title_font)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.tick_params(axis="y", labelsize=tick_font)

    ax.text(
        0.02,
        0.02,
        (
            f"ΔU_clock - ΔU_geo = {delta_u_diff:+.2f} ± {sigma_diff:.2f} m²/s² "
            f"(z={z_score:+.2f})\n"
            f"epsilon ≈ {epsilon*1e6:+.2f} ± {sigma_epsilon*1e6:.2f} ppm\n"
            f"z_clock≈{z_clock:.3e}, z_geo≈{z_geo:.3e} (Δf/f)\n"
            f"σ_clock ≈ {height_sigma_m*100:.0f} cm (height-equivalent)"
        ),
        transform=ax.transAxes,
        fontsize=note_font,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    ax.legend(loc="upper right", fontsize=legend_font, frameon=True)
    if is_en:
        fig.subplots_adjust(left=0.13, right=0.965, top=0.85, bottom=0.20)
    else:
        fig.tight_layout()

    out_png = localize_figure_output_path(out_dir / "optical_clock_chronometric_leveling.png", root=root)
    out_pdf = localize_figure_output_path(out_dir / "optical_clock_chronometric_leveling.pdf", root=root)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if is_en:
        fig.savefig(out_png)
        fig.savefig(out_pdf)
    else:
        fig.savefig(out_png, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    local_pdf = root / "data" / "quantum" / "sources" / "arxiv_2309.14953v3.pdf"
    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "source": {
            "reference": "arXiv:2309.14953v3, 'Long-distance chronometric leveling with a portable optical clock'",
            "url": "https://arxiv.org/abs/2309.14953",
            "local_pdf": str(local_pdf),
            "local_pdf_sha256": "ACB4A20342662C6993F761FEC1AAE05730E26B692D0DCF099471748B0B6B8196",
            "abstract_values": {
                "delta_u_clock_m2_s2": delta_u_clock,
                "sigma_clock_m2_s2": sigma_clock,
                "delta_u_geodetic_m2_s2": delta_u_geo,
                "sigma_geodetic_m2_s2": sigma_geo,
            },
        },
        "config": {
            "c_m_per_s": cfg.c_m_per_s,
            "g_m_per_s2": cfg.g_m_per_s2,
        },
        "derived": {
            "z_clock_delta_f_over_f": z_clock,
            "z_geodetic_delta_f_over_f": z_geo,
            "delta_u_diff_m2_s2": delta_u_diff,
            "sigma_diff_m2_s2": sigma_diff,
            "z_score": z_score,
            "epsilon": epsilon,
            "sigma_epsilon": sigma_epsilon,
            "height_sigma_m": height_sigma_m,
        },
        "outputs": {"png": str(out_png), "pdf": str(out_pdf)},
        "notes": [
            "This script treats the geodetic determination of ΔU as the reference and infers epsilon from the clock-derived ΔU.",
            "P-model (stationary clocks) predicts the same leading redshift relation Δf/f ≈ ΔU/c^2 in the weak field.",
        ],
    }
    out_json = localize_figure_output_path(out_dir / "optical_clock_chronometric_leveling_metrics.json", root=root)
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] pdf : {out_pdf}")
    print(f"[ok] json: {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
