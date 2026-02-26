from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Measurement:
    label: str
    alpha_inv: float
    sigma_alpha_inv: float
    reference: str
    url: str
    local_pdf: str
    local_pdf_sha256: str


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


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    labels = [recoil.label, g2.label]
    x = np.arange(2, dtype=float)
    y = np.array([recoil.alpha_inv, g2.alpha_inv], dtype=float)
    yerr = np.array([recoil.sigma_alpha_inv, g2.sigma_alpha_inv], dtype=float)

    fig, ax = plt.subplots(figsize=(10.8, 5.4), dpi=150)
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4, elinewidth=1.8, color="#1f77b4", ecolor="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("alpha^{-1}")
    ax.set_title("de Broglie precision cross-check via alpha (recoil vs electron g-2)")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)

    ax.text(
        0.02,
        0.02,
        (
            f"Δ(alpha^-1) = {delta:+.3e} ± {sigma:.3e}  (z={z:+.2f})\n"
            f"epsilon_needed ≈ {epsilon_required*1e9:+.2f} ppb\n"
            f"MC: epsilon = {eps_mu*1e9:+.2f} ± {eps_sigma*1e9:.2f} ppb (1σ)"
        ),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    fig.tight_layout()
    out_png = out_dir / "de_broglie_precision_alpha_consistency.png"
    fig.savefig(out_png, bbox_inches="tight")
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
        "outputs": {"png": str(out_png)},
        "notes": [
            "This treats the electron g-2 based alpha as a reference and maps the discrepancy into an effective recoil epsilon via alpha ∝ sqrt(h/m).",
            "In reality, discrepancies could also come from systematics or theory inputs; epsilon here is an interpretive parameterization.",
        ],
    }
    out_json = out_dir / "de_broglie_precision_alpha_consistency_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
