"""
目的: 量子 topic の de broglie precision alpha consistency に対応する公開図・表・監査指標を再生成する。
入力: script 内の既定パラメータと必要な公開データまたは基準値を用いる。
出力: output/public と output/private の canonical artifact を更新する。
前提: 論文本文と README はこの script が出力する公開成果物を正として参照する。
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


from figure_japanese_localizer import enable_japanese_figure_localization
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.figure_locale_paths import localize_figure_output_path
from scripts.utils.plot_style import apply_wavep_figure_layout, get_wavep_font_size, install_wavep_font_profile

enable_japanese_figure_localization()

_PROFILE_NAME = "part3b_quantum_verification"

# クラス: `Measurement` の責務と境界条件を定義する。
@dataclass(frozen=True)
class Measurement:
    label: str
    alpha_inv: float
    sigma_alpha_inv: float
    reference: str
    url: str
    local_pdf: str
    local_pdf_sha256: str


# 関数: `epsilon_from_alpha_inv` の入出力契約と処理意図を定義する。

def epsilon_from_alpha_inv(*, alpha_inv_ref: float, alpha_inv_meas: float) -> float:
    """
    Map a discrepancy between alpha determinations into an effective epsilon for the recoil (de Broglie) measurement.

    If alpha is inferred from a recoil measurement of h/m, then alpha ∝ sqrt(h/m).
    A fractional scaling (1+epsilon) in the recoil-based h/m would imply:
      alpha_meas = sqrt(1+epsilon) * alpha_ref
      => alpha_inv_meas = alpha_inv_ref / sqrt(1+epsilon)
      => epsilon = (alpha_inv_ref/alpha_inv_meas)^2 - 1
    """
    return float((alpha_inv_ref / alpha_inv_meas) ** 2 - 1.0)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = ROOT
    install_wavep_font_profile(profile_name=_PROFILE_NAME)
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_lang = str(os.getenv("WAVEP_FIGURE_LANG", "ja")).strip().lower()
    is_en = figure_lang.startswith("en")

    recoil = Measurement(
        label="Recoil (Rb; Bloch+AI)",
        alpha_inv=137.03599945,
        sigma_alpha_inv=0.00000062,
        reference="Bouchendira et al., 'Determination of the fine structure constant with atom interferometry and Bloch oscillations' (arXiv:0812.3139v1)",
        url="https://arxiv.org/abs/0812.3139",
        local_pdf=str(root / "data" / "quantum" / "sources" / "arxiv_0812.3139v1.pdf"),
        local_pdf_sha256="F763334508B9D7F06A390BCF32E38E246CD4468FEEBD5D9EB6FB57AC93782B55",
    )
    g2 = Measurement(
        label="g-2 (electron; QED)",
        alpha_inv=137.035999084,
        sigma_alpha_inv=0.000000051,
        reference="Gabrielse et al., 'New Measurement of the Electron Magnetic Moment and the Fine Structure Constant' (arXiv:0801.1134v2)",
        url="https://arxiv.org/abs/0801.1134",
        local_pdf=str(root / "data" / "quantum" / "sources" / "arxiv_0801.1134v2.pdf"),
        local_pdf_sha256="562D23333D57C1C8D415F357C761508FDC4A5AEF512B28639D3AC0079A7C69F5",
    )

    delta = float(recoil.alpha_inv - g2.alpha_inv)
    sigma = float(math.sqrt(recoil.sigma_alpha_inv**2 + g2.sigma_alpha_inv**2))
    z = float(delta / sigma) if sigma > 0 else float("nan")

    epsilon_required = epsilon_from_alpha_inv(alpha_inv_ref=g2.alpha_inv, alpha_inv_meas=recoil.alpha_inv)

    # Propagate epsilon uncertainty with a deterministic Monte Carlo.
    rng = np.random.default_rng(20260124)
    n_mc = 200_000
    alpha_inv_recoil_mc = rng.normal(recoil.alpha_inv, recoil.sigma_alpha_inv, size=n_mc)
    alpha_inv_g2_mc = rng.normal(g2.alpha_inv, g2.sigma_alpha_inv, size=n_mc)
    eps_mc = (alpha_inv_g2_mc / alpha_inv_recoil_mc) ** 2 - 1.0
    eps_mu = float(np.mean(eps_mc))
    eps_sigma = float(np.std(eps_mc, ddof=1))

    # Plot
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    labels = [recoil.label, g2.label]
    x = np.arange(2, dtype=float)
    y = np.array([recoil.alpha_inv, g2.alpha_inv], dtype=float)
    yerr = np.array([recoil.sigma_alpha_inv, g2.sigma_alpha_inv], dtype=float)

    fig, ax = plt.subplots(dpi=150)
    apply_wavep_figure_layout(fig, template="part2_single_panel_sparse")
    axis_label_font = get_wavep_font_size("axis") * (0.84 if is_en else 1.0)
    panel_title_font = get_wavep_font_size("title") * (0.80 if is_en else 1.0)
    tick_font = get_wavep_font_size("tick")
    note_font = get_wavep_font_size("note")
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4, elinewidth=1.8, color="#1f77b4", ecolor="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=tick_font)
    ax.set_ylabel("alpha^{-1}", fontsize=axis_label_font)
    ax.set_title(
        "de Broglie precision cross-check via alpha (recoil vs electron g-2)",
        fontsize=panel_title_font,
        pad=10.0,
    )
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.tick_params(axis="y", labelsize=tick_font)
    y_formatter = ScalarFormatter(useMathText=False)
    y_formatter.set_scientific(False)
    y_formatter.set_useOffset(False)
    ax.yaxis.set_major_formatter(y_formatter)

    ax.text(
        0.02,
        0.02,
        (
            f"Δ(alpha^-1) = {delta:+.3e} ± {sigma:.3e}  (z={z:+.2f})\n"
            f"epsilon_needed ≈ {epsilon_required*1e9:+.2f} ppb\n"
            f"MC: epsilon = {eps_mu*1e9:+.2f} ± {eps_sigma*1e9:.2f} ppb (1σ)"
        ),
        transform=ax.transAxes,
        fontsize=note_font,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    out_png = localize_figure_output_path(out_dir / "de_broglie_precision_alpha_consistency.png", root=root)
    out_pdf = localize_figure_output_path(out_dir / "de_broglie_precision_alpha_consistency.pdf", root=root)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "measurements": [
            {
                "label": recoil.label,
                "alpha_inv": recoil.alpha_inv,
                "sigma_alpha_inv": recoil.sigma_alpha_inv,
                "reference": recoil.reference,
                "url": recoil.url,
                "local_pdf": recoil.local_pdf,
                "local_pdf_sha256": recoil.local_pdf_sha256,
            },
            {
                "label": g2.label,
                "alpha_inv": g2.alpha_inv,
                "sigma_alpha_inv": g2.sigma_alpha_inv,
                "reference": g2.reference,
                "url": g2.url,
                "local_pdf": g2.local_pdf,
                "local_pdf_sha256": g2.local_pdf_sha256,
            },
        ],
        "derived": {
            "delta_alpha_inv": delta,
            "sigma_delta_alpha_inv": sigma,
            "z_score": z,
            "epsilon_required": epsilon_required,
            "epsilon_mc_mean": eps_mu,
            "epsilon_mc_sigma": eps_sigma,
            "epsilon_units": "dimensionless (fractional scaling in recoil-based h/m)",
        },
        "outputs": {"png": str(out_png), "pdf": str(out_pdf)},
        "notes": [
            "This treats the electron g-2 based alpha as a reference and maps the discrepancy into an effective recoil epsilon via alpha ∝ sqrt(h/m).",
            "In reality, discrepancies could also come from systematics or theory inputs; epsilon here is an interpretive parameterization.",
        ],
    }
    out_json = localize_figure_output_path(out_dir / "de_broglie_precision_alpha_consistency_metrics.json", root=root)
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] pdf : {out_pdf}")
    print(f"[ok] json: {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
