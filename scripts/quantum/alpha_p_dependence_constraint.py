from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Config:
    # --- Primary constraints (as-used; see sources in metrics) ---
    # Rosenband et al. 2008 (Science 319, 1808; NIST publication page):
    #   alpha_dot/alpha = (1.4 ± 1.7)×10^-17 / year (preliminary; repeated measurements over ~1 year)
    rosenband_alpha_dot_over_alpha_per_year: float = 1.4e-17
    rosenband_sigma_alpha_dot_over_alpha_per_year: float = 1.7e-17

    # Webb et al. 2011 (Phys. Rev. Lett. 107, 191101): spatial dipole (order ~10 ppm).
    # This is not a pure "bound"; we use it as an amplitude scale for cross-checking.
    webb_dipole_amp_ppm: float = 10.2
    webb_dipole_sigma_ppm: float = 2.1

    # --- Mapping for u=ln(P/P0) in weak field ---
    c_m_per_s: float = 299_792_458.0

    # Earth's orbit (Sun potential modulation); used to turn d(alpha)/dt constraints into a κ bound.
    # Use IAU/standard values (order matters only at ~1% here).
    gm_sun_m3_s2: float = 1.327_124_400_18e20
    au_m: float = 149_597_870_700.0  # exact definition of AU
    earth_orbit_eccentricity: float = 0.0167086

    # H 1S–2S scaling: in the simplest non-relativistic picture f ∝ α^2.
    spectral_alpha_exponent: float = 2.0

    # Plot settings
    fig_w_in: float = 11.8
    fig_h_in: float = 6.0
    dpi: int = 170


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relpath(root: Path, p: Path) -> str:
    try:
        return str(p.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _try_load_h_1s2s_fractional_sigma(root: Path) -> Optional[float]:
    p = root / "output" / "quantum" / "qed_vacuum_precision_metrics.json"
    if not p.exists():
        return None
    try:
        data = _read_json(p)
    except Exception:
        return None
    frac = ((data.get("hydrogen_1s2s") or {}) if isinstance(data.get("hydrogen_1s2s"), dict) else {}).get(
        "fractional_sigma"
    )
    if isinstance(frac, (int, float)) and math.isfinite(float(frac)) and float(frac) > 0:
        return float(frac)
    return None


def _sun_potential_delta_u_earth_orbit(cfg: Config) -> Dict[str, float]:
    e = float(cfg.earth_orbit_eccentricity)
    a = float(cfg.au_m)
    c2 = float(cfg.c_m_per_s) ** 2

    r_peri = a * (1.0 - e)
    r_aphe = a * (1.0 + e)
    u_peri = float(cfg.gm_sun_m3_s2) / r_peri
    u_aphe = float(cfg.gm_sun_m3_s2) / r_aphe
    delta_u_m2_s2 = u_peri - u_aphe
    delta_u_over_c2 = float(delta_u_m2_s2 / c2)

    return {
        "r_peri_m": float(r_peri),
        "r_aphe_m": float(r_aphe),
        "U_peri_m2_s2": float(u_peri),
        "U_aphe_m2_s2": float(u_aphe),
        "delta_U_m2_s2": float(delta_u_m2_s2),
        "delta_u_lnP_over_1": float(delta_u_over_c2),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()

    alpha_src_manifest = root / "data" / "quantum" / "sources" / "alpha_variation_manifest.json"
    alpha_sources: Dict[str, Any] = {"manifest": _relpath(root, alpha_src_manifest)}
    if alpha_src_manifest.exists():
        try:
            alpha_sources["manifest_data"] = _read_json(alpha_src_manifest)
        except Exception:
            alpha_sources["manifest_parse_error"] = True

    orbit = _sun_potential_delta_u_earth_orbit(cfg)
    delta_u_orbit = float(orbit["delta_u_lnP_over_1"])
    if not (delta_u_orbit > 0):
        raise RuntimeError("unexpected delta_u_orbit <= 0")

    # --- Constraint 1: Rosenband (clock drift) interpreted against annual solar potential modulation ---
    a_dot = float(cfg.rosenband_alpha_dot_over_alpha_per_year)
    sigma_a_dot = float(cfg.rosenband_sigma_alpha_dot_over_alpha_per_year)

    # Convert to a 3σ absolute "per-year" bound and treat it as an allowed |Δα/α| over a 1-year scale.
    # This is an order-of-magnitude mapping (drift vs periodic modulation); we record it explicitly.
    alpha_frac_3sigma_per_year = abs(a_dot) + 3.0 * sigma_a_dot
    kappa_max_clock = float(alpha_frac_3sigma_per_year / delta_u_orbit)

    # --- Constraint 2: Spectroscopic baseline (H 1S–2S fractional sigma) under hypothetical α(P) ---
    frac_sigma_1s2s = _try_load_h_1s2s_fractional_sigma(root) or 4.055_049_053_736_426e-15
    alpha_exp = float(cfg.spectral_alpha_exponent)
    # If f ∝ α^p, then |Δf/f| ≈ p |Δα/α| = p |κ| |Δu|.
    kappa_max_h_1s2s = float(frac_sigma_1s2s / (alpha_exp * delta_u_orbit))

    # --- Cross-check: Webb dipole amplitude scale (not a pure bound) ---
    webb_amp = float(cfg.webb_dipole_amp_ppm) * 1e-6
    webb_sigma = float(cfg.webb_dipole_sigma_ppm) * 1e-6
    webb_amp_3sigma = abs(webb_amp) + 3.0 * webb_sigma
    # Translate to implied κ for a few assumed |Δu| scales (heuristic).
    assumed_delta_u = {
        "galaxy_phi_over_c2_1e-6": 1e-6,
        "cluster_phi_over_c2_1e-5": 1e-5,
        "unity_delta_u_1": 1.0,
    }
    webb_implied_kappa = {k: float(webb_amp_3sigma / v) for k, v in assumed_delta_u.items()}

    # --- Plot ---
    labels = [
        "clock drift → κ (Rosenband; 3σ; orbit ΔU⊙)",
        "H 1S–2S σ → κ (1σ; orbit ΔU⊙)",
    ]
    vals = [kappa_max_clock, kappa_max_h_1s2s]

    fig, ax = plt.subplots(figsize=(cfg.fig_w_in, cfg.fig_h_in), dpi=cfg.dpi)
    x = np.arange(len(vals), dtype=float)
    ax.bar(x, vals, color=["#2b6cb0", "#805ad5"], alpha=0.9)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("|κ| upper bound (dimensionless)")
    ax.set_title("Phase 7 / Step 7.11: Constraint on α(P) via α(P/P0)=α0(1+κ ln(P/P0)+...)")
    ax.grid(True, axis="y", ls=":", lw=0.7, alpha=0.6)
    for xi, v in zip(x, vals, strict=True):
        ax.text(
            float(xi),
            float(v) * 1.10,
            f"{v:.2e}",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.9},
        )

    ax.text(
        0.02,
        0.02,
        (
            "Mapping used (weak field): u=ln(P/P0)=−φ/c².\n"
            f"Earth orbit solar potential modulation: Δu≈ΔU⊙/c²≈{delta_u_orbit:.3e}.\n"
            f"Webb (2011) dipole amplitude scale: ~{webb_amp:.1e} (fraction)."
        ),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.95},
    )

    fig.tight_layout()
    out_png = out_dir / "alpha_p_dependence_constraint.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "generated_utc": _iso_utc_now(),
        "phase": 7,
        "step": "7.11",
        "title": "Constraint on alpha(P) dependence (kappa)",
        "model": {
            "alpha_of_p": "α(P/P0) = α0 (1 + κ ln(P/P0) + ...)",
            "u_definition": "u(x,t) ≡ ln(P/P0)",
            "phi_definition": "φ ≡ -c^2 ln(P/P0) = -c^2 u",
            "weak_field_relation": "u ≈ -φ/c^2",
            "note": "If κ=0 is adopted, α is P-independent by construction (this is the current stance).",
        },
        "environment": {
            "constants": {
                "c_m_per_s": cfg.c_m_per_s,
                "gm_sun_m3_s2": cfg.gm_sun_m3_s2,
                "au_m": cfg.au_m,
                "earth_orbit_eccentricity": cfg.earth_orbit_eccentricity,
            },
            "earth_orbit_solar_potential": orbit,
        },
        "constraints": {
            "rosenband_2008_clock_drift": {
                "alpha_dot_over_alpha_per_year": a_dot,
                "sigma_per_year": sigma_a_dot,
                "abs_3sigma_per_year_used_as_delta_alpha_over_alpha_over_1yr": alpha_frac_3sigma_per_year,
                "kappa_max_from_orbit_delta_u": kappa_max_clock,
                "mapping_note": "This treats the annual solar-potential modulation as a 1-year scale for alpha changes (order-of-magnitude; drift vs periodic not strictly identical).",
            },
            "hydrogen_1s2s_spectroscopic_baseline": {
                "fractional_sigma_1s2s": frac_sigma_1s2s,
                "assumed_scaling": f"|Δf/f| ≈ p|Δα/α| with p={alpha_exp:g} (non-relativistic alpha scaling)",
                "kappa_max_from_orbit_delta_u": kappa_max_h_1s2s,
                "note": "This is a conservative back-of-envelope constraint using the fixed 1S–2S baseline uncertainty and the same orbit Δu.",
                "source_metrics": _relpath(root, root / "output" / "quantum" / "qed_vacuum_precision_metrics.json"),
            },
            "webb_2011_spatial_dipole_scale": {
                "dipole_amp_ppm": cfg.webb_dipole_amp_ppm,
                "dipole_sigma_ppm": cfg.webb_dipole_sigma_ppm,
                "amp_abs_3sigma_fraction_used": webb_amp_3sigma,
                "assumed_delta_u_for_translation": assumed_delta_u,
                "implied_kappa_from_assumed_delta_u": webb_implied_kappa,
                "note": "Webb is recorded as an amplitude scale. Translating to κ requires a separate cosmological mapping between absorber environments and u=ln(P/P0).",
            },
        },
        "recommendation": {
            "adopt_kappa": 0.0,
            "rationale": "Given tight clock-based constraints, treat α as P-independent at the current stage and keep κ as a rejected/limited degree of freedom unless a full derivation is provided.",
        },
        "outputs": {
            "png": _relpath(root, out_png),
        },
        "sources": {
            "alpha_variation_manifest": alpha_sources,
            "rosenband_2008": {
                "reference": "Rosenband et al., 'Frequency Ratio of Al+ and Hg+ Single-Ion Optical Clocks; Metrology at the 17th Decimal Place' (Science 319, 1808; 2008).",
                "url": "https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=50655",
                "local_pdf": "data/quantum/sources/nist_pub_50655_rosenband2008_al_hg_clocks.pdf",
                "value_origin": "As-coded value (see primary PDF + manifest; Accessed 2026-01-28).",
            },
            "webb_2011": {
                "reference": "Webb et al., 'Indications of a Spatial Variation of the Fine Structure Constant' (Phys. Rev. Lett. 107, 191101; 2011).",
                "url": "https://arxiv.org/abs/1008.3907",
                "local_pdf": "data/quantum/sources/arxiv_1008.3907v2.pdf",
                "value_origin": "As-coded ppm-scale dipole amplitude (see primary PDF + manifest; Accessed 2026-01-28).",
            },
        },
    }

    out_json = out_dir / "alpha_p_dependence_constraint.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")


if __name__ == "__main__":
    main()
