from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    # Constants
    c_m_per_s: float = 299_792_458.0
    g_m_per_s2: float = 9.80665

    # Example clock frequencies
    # 87Sr 1S0-3P0 clock transition is ~429 THz (698 nm).
    f_sr_hz: float = 429_228_004_229_873.0
    # Cs hyperfine (definition of SI second).
    f_cs_hz: float = 9_192_631_770.0

    # 7.6.1: ensemble dephasing by gravity (optical lattice clock)
    # Model: atoms distributed in height with Gaussian σ_z.
    sigma_z_m_list: tuple[float, ...] = (0.1e-3, 1e-3, 1e-2)  # 0.1 mm, 1 mm, 1 cm
    t_min_s: float = 0.1
    t_max_s: float = 1e5
    n_t: int = 600

    # 7.6.2: effective run-to-run fractional rate noise σ_y needed to reduce visibility.
    vis_targets: tuple[float, ...] = (0.9, 0.5)
    t_noise_min_s: float = 0.1
    t_noise_max_s: float = 1e4
    n_t_noise: int = 500


def visibility_gaussian_phase_noise(phase_sigma_rad: np.ndarray) -> np.ndarray:
    # If phase is Gaussian with std σ, visibility V = exp(-σ^2/2).
    return np.exp(-0.5 * (phase_sigma_rad**2))


def ensemble_gravity_phase_sigma_rad(
    *, omega0_rad_s: float, g_m_per_s2: float, sigma_z_m: float, t_s: np.ndarray, c_m_per_s: float
) -> np.ndarray:
    # σ_φ = ω0 * (g σ_z / c^2) * t
    return omega0_rad_s * (g_m_per_s2 * sigma_z_m / (c_m_per_s**2)) * t_s


def required_sigma_y_for_visibility(*, vis: float, omega0_rad_s: float, t_s: np.ndarray) -> np.ndarray:
    # V = exp(-(omega σ_y t)^2/2) -> σ_y = sqrt(-2 ln V) / (omega t)
    if not (0.0 < vis < 1.0):
        raise ValueError(f"vis must be in (0,1): {vis}")
    k = math.sqrt(-2.0 * math.log(vis))
    return (k / (omega0_rad_s * t_s)).astype(float)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()

    omega_sr = 2.0 * math.pi * cfg.f_sr_hz
    omega_cs = 2.0 * math.pi * cfg.f_cs_hz

    # Panel A: ensemble dephasing (gravity-induced inhomogeneous redshift)
    t = np.logspace(math.log10(cfg.t_min_s), math.log10(cfg.t_max_s), cfg.n_t)
    curves = []
    for sigma_z_m in cfg.sigma_z_m_list:
        sigma_phi = ensemble_gravity_phase_sigma_rad(
            omega0_rad_s=omega_sr,
            g_m_per_s2=cfg.g_m_per_s2,
            sigma_z_m=sigma_z_m,
            t_s=t,
            c_m_per_s=cfg.c_m_per_s,
        )
        v = visibility_gaussian_phase_noise(sigma_phi)
        sigma_y = float(cfg.g_m_per_s2 * sigma_z_m / (cfg.c_m_per_s**2))
        t_half = float(math.sqrt(2.0 * math.log(2.0)) / (omega_sr * sigma_y))
        curves.append(
            {
                "sigma_z_m": float(sigma_z_m),
                "sigma_y": sigma_y,
                "t_half_s": t_half,
                "t_grid_s": t,
                "visibility": v,
            }
        )

    # Panel B: σ_y needed to see contrast loss at given interrogation time
    t_noise = np.logspace(math.log10(cfg.t_noise_min_s), math.log10(cfg.t_noise_max_s), cfg.n_t_noise)
    sigma_y_req = []
    for vis in cfg.vis_targets:
        sigma_y_req.append(
            {
                "vis": float(vis),
                "sr": required_sigma_y_for_visibility(vis=vis, omega0_rad_s=omega_sr, t_s=t_noise),
                "cs": required_sigma_y_for_visibility(vis=vis, omega0_rad_s=omega_cs, t_s=t_noise),
            }
        )

    # Plot
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), dpi=150)

    ax = axes[0]
    for c in curves:
        sigma_z_mm = c["sigma_z_m"] * 1e3
        sigma_y = c["sigma_y"]
        t_half = c["t_half_s"]
        ax.plot(
            c["t_grid_s"],
            c["visibility"],
            lw=2.0,
            label=f"σz={sigma_z_mm:.1f} mm  (gσz/c²≈{sigma_y:.1e},  V=0.5 at {t_half:.0f}s)",
        )
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("interrogation time T (s)")
    ax.set_ylabel("visibility V (Ramsey contrast; model)")
    ax.set_title("Gravity-induced dephasing (optical clock ensemble; Gaussian height spread)")
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.7)
    ax.legend(fontsize=8, frameon=True, loc="lower left")

    ax = axes[1]
    colors = {0.9: "#1f77b4", 0.5: "#d62728"}
    for row in sigma_y_req:
        vis = float(row["vis"])
        col = colors.get(vis, None) or "#333333"
        ax.plot(t_noise, row["sr"], lw=2.0, color=col, label=f"Sr clock (V={vis:.1f})")
        ax.plot(t_noise, row["cs"], lw=2.0, color=col, ls="--", label=f"Cs clock (V={vis:.1f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("interrogation time T (s)")
    ax.set_ylabel("required σ_y (RMS fractional rate noise)")
    ax.set_title("P-model time-structure: σ_y needed to mimic decoherence (run-to-run)")
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.7)
    ax.legend(fontsize=8, frameon=True, loc="upper right")

    fig.suptitle("Gravity-induced decoherence: observables and noise budget (Phase 7 / Step 7.6)", y=1.02)
    fig.tight_layout()

    out_png = out_dir / "gravity_induced_decoherence.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "sources": [
            {
                "reference": "Pikovski et al., 'Universal decoherence due to gravitational time dilation' (arXiv:1311.1095v2)",
                "url": "https://arxiv.org/abs/1311.1095",
                "local_pdf": str(root / "data" / "quantum" / "sources" / "arxiv_1311.1095v2.pdf"),
                "sha256": "5507FEF706C4C460485D611506DD3843D9F49E83DC7AAA7E22701FB9EA17B0E7",
            },
            {
                "reference": "Hasegawa et al., 'Decoherence of Atomic Ensembles in Optical Lattice Clocks by Gravity' (arXiv:2107.02405v2)",
                "url": "https://arxiv.org/abs/2107.02405",
                "local_pdf": str(root / "data" / "quantum" / "sources" / "arxiv_2107.02405v2.pdf"),
                "sha256": "7E171DC5C3406C05028261C07471FE6687BC4DC5E8B078B9286D60BEA1617149",
            },
            {
                "reference": "Bonder et al., 'Can gravity account for the emergence of classicality?' (arXiv:1509.04363v3)",
                "url": "https://arxiv.org/abs/1509.04363",
                "local_pdf": str(root / "data" / "quantum" / "sources" / "arxiv_1509.04363v3.pdf"),
                "sha256": "2096AE35E84FC9850DF17CF11AA081AA1749EFB3669ECD22B32D7E72A2FF400C",
            },
            {
                "reference": "Anastopoulos & Hu, 'Centre of mass decoherence due to time dilation' (arXiv:1507.05828v5)",
                "url": "https://arxiv.org/abs/1507.05828",
                "local_pdf": str(root / "data" / "quantum" / "sources" / "arxiv_1507.05828v5.pdf"),
                "sha256": "25FB3DE318478837E4F9CEA0161CD4A965485553F89AC1390EEAFE4127F5E4C1",
            },
            {
                "reference": "Pikovski et al., reply (arXiv:1509.07767v1)",
                "url": "https://arxiv.org/abs/1509.07767",
                "local_pdf": str(root / "data" / "quantum" / "sources" / "arxiv_1509.07767v1.pdf"),
                "sha256": "AE51D0F59C648796D1C3025522D3F58D763A22860CACC118FC11F41887AEB3BF",
            },
        ],
        "model": {
            "ensemble_dephasing": "V=exp(-(ω0*(gσz/c^2)*T)^2/2) for Gaussian height distribution",
            "pmodel_time_noise": "V=exp(-(ω0*σ_y*T)^2/2) for run-to-run fractional rate noise σ_y",
        },
        "config": {
            "c_m_per_s": cfg.c_m_per_s,
            "g_m_per_s2": cfg.g_m_per_s2,
            "f_sr_hz": cfg.f_sr_hz,
            "f_cs_hz": cfg.f_cs_hz,
            "sigma_z_m_list": list(cfg.sigma_z_m_list),
            "t_min_s": cfg.t_min_s,
            "t_max_s": cfg.t_max_s,
            "n_t": cfg.n_t,
            "vis_targets": list(cfg.vis_targets),
            "t_noise_min_s": cfg.t_noise_min_s,
            "t_noise_max_s": cfg.t_noise_max_s,
            "n_t_noise": cfg.n_t_noise,
        },
        "derived": {
            "ensemble": [{"sigma_z_m": c["sigma_z_m"], "sigma_y": c["sigma_y"], "t_half_s": c["t_half_s"]} for c in curves],
            "required_sigma_y": {
                "t_grid_s": t_noise.tolist(),
                "targets": [{"vis": row["vis"], "sr": row["sr"].tolist(), "cs": row["cs"].tolist()} for row in sigma_y_req],
            },
        },
        "outputs": {"png": str(out_png)},
        "notes": [
            "The ensemble model is inhomogeneous dephasing (not dynamical environment-induced decoherence).",
            "The σ_y model is a minimal parametrization of extra time-structure noise (P-model-specific); real experiments may have correlated noise between arms.",
        ],
    }
    out_json = out_dir / "gravity_induced_decoherence_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")


if __name__ == "__main__":
    main()
