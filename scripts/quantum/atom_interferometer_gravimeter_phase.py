from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# クラス: `Config` の責務と境界条件を定義する。
@dataclass(frozen=True)
class Config:
    # From Mueller et al. (arXiv:0710.3768), example: pulse separation time T = 400 ms.
    T_s: float = 0.400

    # Cs D2 wavelength used for Raman beams (paper uses 852 nm system).
    lambda_m: float = 852e-9

    # Gravity
    g_m_per_s2: float = 9.80665

    # Sweep around T to show scaling (optional, still deterministic).
    T_min_s: float = 0.05
    T_max_s: float = 0.50
    n_T: int = 200


# 関数: `keff_counterprop` の入出力契約と処理意図を定義する。

def keff_counterprop(lambda_m: float) -> float:
    # k = 2π/λ ; for counterpropagating Raman beams, k_eff ≈ 2k = 4π/λ.
    return float(4.0 * math.pi / lambda_m)


# 関数: `phase_rad` の入出力契約と処理意図を定義する。

def phase_rad(*, keff: float, g_m_per_s2: float, T_s: float) -> float:
    # Mueller et al. Eq.(1): φ = k_eff g T^2 - φ_L (laser phase term used for readout).
    # Here we report the gravity-dependent magnitude k_eff g T^2.
    return float(keff * g_m_per_s2 * (T_s**2))


# 関数: `_as_float` の入出力契約と処理意図を定義する。

def _as_float(v: object) -> float | None:
    # 条件分岐: `isinstance(v, (int, float)) and math.isfinite(float(v))` を満たす経路を評価する。
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)

    return None


# 関数: `_try_load_beta_frozen` の入出力契約と処理意図を定義する。

def _try_load_beta_frozen(root: Path) -> float | None:
    p = root / "output" / "private" / "theory" / "frozen_parameters.json"
    # 条件分岐: `not p.exists()` を満たす経路を評価する。
    if not p.exists():
        return None

    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    beta = _as_float(j.get("beta"))
    return beta


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    keff = keff_counterprop(cfg.lambda_m)
    c_m_per_s = 299_792_458.0

    T_grid = np.linspace(cfg.T_min_s, cfg.T_max_s, cfg.n_T)
    phi_grid = keff * cfg.g_m_per_s2 * (T_grid**2)

    phi_ref = phase_rad(keff=keff, g_m_per_s2=cfg.g_m_per_s2, T_s=cfg.T_s)

    # Plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.8, 5.4), dpi=150)
    ax.plot(T_grid, phi_grid / (2.0 * math.pi), lw=2.0, label="k_eff g T² / 2π (cycles)")
    ax.axvline(cfg.T_s, color="#ff7f0e", ls="-.", lw=1.2, label=f"T={cfg.T_s*1e3:.0f} ms (example)")
    ax.set_xlabel("pulse separation T (s)")
    ax.set_ylabel("phase (cycles)")
    ax.set_title("Atom interferometer gravimeter: phase scaling (φ ≈ k_eff g T²)")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.legend(frameon=True, fontsize=9, loc="upper left")

    note = (
        f"Model: φ_g = k_eff g T².  "
        f"λ={cfg.lambda_m*1e9:.0f} nm ⇒ k_eff≈{keff:.3e} 1/m.  "
        f"At T={cfg.T_s:.3f} s: φ_g≈{phi_ref/(2*math.pi):.3e} cycles."
    )
    fig.text(0.01, -0.02, note, fontsize=9)
    fig.tight_layout()

    out_png = out_dir / "atom_interferometer_gravimeter_phase.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    beta_frozen = _try_load_beta_frozen(root)
    delta_beta_vs_gr = None if beta_frozen is None else float(beta_frozen - 1.0)

    # β enters the atom interferometer phase only through light propagation (dispersion → k_eff).
    # Mueller et al. (0710.3768) derive k_eff from the photon dispersion relation and use:
    #   φ = k_eff g T^2 - φ_L  (their Eq.(1)/(5) in the vertical-beam case).
    #
    # In P-model, light propagation is modeled by n(P)=(P/P0)^(2β)=exp(-2β φ/c^2).
    # For a *local* lab interferometer, a constant offset in φ cancels / is absorbed; therefore the
    # leading β-sensitive term is suppressed by *potential differences inside the interferometer*.
    #
    # Fixed (deterministic) illustrative geometry for that suppression:
    #   - 3-pulse light-pulse Mach–Zehnder at t={0,T,2T}
    #   - assume an atom fountain tuned such that v0=gT (apex at t=T)
    #   - then Δz (apex height above the first/third pulse height) is H = (1/2) g T^2
    #   - to leading order, the β-induced phase difference scales as:
    #       Δφ_beta ≈ (β-1) * k_eff * g^3 * T^4 / c^2
    #
    # We also keep the earlier “absolute φ (Earth-to-infinity) scaling” as an explicit upper-bound style
    # order estimate (not used for Reject), to make the cancellation point visible.
    gm_earth_m3_s2 = 3.986_004_418e14  # WGS84
    r_earth_m = 6_378_137.0  # WGS84 equatorial radius
    phi_earth_m2_s2 = -gm_earth_m3_s2 / r_earth_m
    earth_x = float(-phi_earth_m2_s2 / (c_m_per_s**2))

    beta_models: dict = {"status": "missing_beta_frozen"} if delta_beta_vs_gr is None else {"status": "ok"}
    # 条件分岐: `delta_beta_vs_gr is not None` を満たす経路を評価する。
    if delta_beta_vs_gr is not None:
        d_beta = float(delta_beta_vs_gr)
        H_m = 0.5 * cfg.g_m_per_s2 * (cfg.T_s**2)
        rel_delta_phase_diff = float(d_beta * (cfg.g_m_per_s2**2) * (cfg.T_s**2) / (c_m_per_s**2))
        delta_phase_diff_rad = float(phi_ref * rel_delta_phase_diff)
        sigma_phase_required_diff_1sigma = abs(delta_phase_diff_rad) / 3.0

        rel_delta_phase_abs = float(2.0 * d_beta * earth_x)
        delta_phase_abs_rad = float(phi_ref * rel_delta_phase_abs)

        beta_models = {
            "status": "ok",
            "beta_frozen": float(beta_frozen),
            "delta_beta_vs_gr": float(d_beta),
            "pmodel_light": {"nP": "n(P)=(P/P0)^(2β)=exp(-2β φ/c^2)"},
            "mueller_reference": {
                "paper": "Mueller et al., arXiv:0710.3768",
                "key_line": "Eq.(1)/(5): φ = k_eff g T^2 - φ_L; k_eff is determined from photon dispersion and uses k_eff=k1-k2.",
            },
            "models": {
                "A_absolute_potential_upper_bound": {
                    "description": "Upper-bound style scaling using Earth's potential relative to infinity; shown for comparison (not used for Reject).",
                    "relative_delta_phase": float(rel_delta_phase_abs),
                    "delta_phase_rad": float(delta_phase_abs_rad),
                    "constants": {
                        "gm_earth_m3_s2": float(gm_earth_m3_s2),
                        "r_earth_m": float(r_earth_m),
                        "phi_earth_m2_s2": float(phi_earth_m2_s2),
                        "x_earth": float(earth_x),
                    },
                },
                "B_differential_within_interferometer": {
                    "description": "Differential (local) estimate: constant φ offsets cancel/absorb; keep only suppression by potential differences within the interferometer.",
                    "assumptions": {
                        "pulse_times_s": [0.0, float(cfg.T_s), float(2.0 * cfg.T_s)],
                        "trajectory": "atom_fountain_apex_at_t_equals_T (v0=gT)",
                        "apex_height_m": float(H_m),
                    },
                    "formula": "Δφ_beta ≈ (β-1) * k_eff * g^3 * T^4 / c^2  (equivalently: Δφ/φ ≈ (β-1) g^2 T^2 / c^2)",
                    "relative_delta_phase": float(rel_delta_phase_diff),
                    "delta_phase_rad": float(delta_phase_diff_rad),
                    "sigma_phase_required_1sigma_for_3sigma_detection_rad": float(sigma_phase_required_diff_1sigma),
                    "relative_sigma_required_1sigma_for_3sigma_detection": float(abs(rel_delta_phase_diff) / 3.0),
                },
            },
            "chosen_model_for_falsification": "B_differential_within_interferometer",
        }

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "source": {
            "reference": "Mueller et al., 'Atom Interferometry Tests of the Isotropy of Post-Newtonian Gravity' (arXiv:0710.3768)",
            "local_pdf": str(root / "data" / "quantum" / "sources" / "arxiv_0710.3768.pdf"),
            "formula": "φ ≈ k_eff g T^2 (gravity-dependent term; Eq.(1) in the paper)",
            "example_T_s": 0.400,
        },
        "config": {
            "T_s": cfg.T_s,
            "lambda_m": cfg.lambda_m,
            "g_m_per_s2": cfg.g_m_per_s2,
            "T_min_s": cfg.T_min_s,
            "T_max_s": cfg.T_max_s,
            "n_T": cfg.n_T,
        },
        "derived": {"k_eff_1_per_m": keff},
        "results": {
            "phi_ref_rad": phi_ref,
            "phi_ref_cycles": phi_ref / (2.0 * math.pi),
            "sensitivity_dphi_dg_rad_per_mps2": float(keff * (cfg.T_s**2)),
            "beta_phase_dependence": beta_models,
        },
        "outputs": {"png": str(out_png)},
        "notes": [
            "The full readout includes a laser phase term (φ_L); gravimeters scan/lock φ_L to infer g.",
            "This script focuses on the magnitude/scaling of the gravity-dependent phase term.",
        ],
    }
    out_json = out_dir / "atom_interferometer_gravimeter_phase_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
