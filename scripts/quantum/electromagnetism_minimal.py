from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


@dataclass(frozen=True)
class Config:
    fig_w_in: float = 12.5
    fig_h_in: float = 6.2
    dpi: int = 180

    r_min_m: float = 1e-15
    r_max_m: float = 1e-9
    n_r: int = 320


def _add_box(ax, x: float, y: float, w: float, h: float, text: str, *, fc: str, ec: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=9.8)


def _add_arrow(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    arrow = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=14, linewidth=1.4, color="#333333")
    ax.add_patch(arrow)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()

    # Constants (SI; CODATA exact where applicable)
    c = 299_792_458.0
    G = 6.674_30e-11
    e_charge = 1.602_176_634e-19  # exact
    eps0 = 8.854_187_8128e-12  # 2018 CODATA
    k_e = 1.0 / (4.0 * math.pi * eps0)
    m_p = 1.672_621_923_69e-27  # kg
    m_e = 9.109_383_701_5e-31  # kg
    a0 = 5.291_772_109_03e-11  # m (Bohr radius)
    j_per_ev = 1.602_176_634e-19  # exact

    # Potential energies at r=a0 (magnitudes)
    u_c_j = k_e * (e_charge**2) / a0
    u_g_j = G * m_p * m_e / a0
    ratio = u_c_j / u_g_j if u_g_j > 0 else float("inf")

    u_c_ev = u_c_j / j_per_ev
    u_g_ev = u_g_j / j_per_ev

    # Curves vs r
    r = np.geomspace(cfg.r_min_m, cfg.r_max_m, cfg.n_r)
    u_c_ev_r = (k_e * (e_charge**2) / r) / j_per_ev
    u_g_ev_r = (G * m_p * m_e / r) / j_per_ev

    # ---- Figure ----
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(cfg.fig_w_in, cfg.fig_h_in), dpi=cfg.dpi)
    fig.suptitle("Phase 7 / Step 7.11: Electromagnetism (minimal positioning) in P-model", fontsize=13)

    # Left: energy scale comparison
    ax0.loglog(r, u_c_ev_r, label="|U_C| = k e² / r", color="#2b6cb0", linewidth=2.0)
    ax0.loglog(r, u_g_ev_r, label="|U_G| = G m_p m_e / r", color="#718096", linewidth=2.0)
    ax0.axvline(a0, color="#dd6b20", linestyle="--", linewidth=1.6, label="Bohr radius a0")
    ax0.set_xlabel("r [m]")
    ax0.set_ylabel("Potential energy magnitude [eV]")
    ax0.set_title("Scale check: Coulomb vs gravity (p–e)")
    ax0.grid(True, which="both", alpha=0.25)
    ax0.legend(loc="lower left", fontsize=9)
    ax0.text(
        0.03,
        0.05,
        f"at r=a0:\n|U_C|≈{u_c_ev:.3g} eV\n|U_G|≈{u_g_ev:.3g} eV\nratio≈{ratio:.3g}",
        transform=ax0.transAxes,
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )

    # Right: flow diagram
    ax1.set_axis_off()
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)

    _add_box(
        ax1,
        0.05,
        0.72,
        0.42,
        0.22,
        "P-field (scalar)\nφ = -c² ln(P/P0)\nlight: n(P)=(P/P0)^(2β)\n(Part I mapping)",
        fc="#e8f2ff",
        ec="#2b6cb0",
    )
    _add_box(
        ax1,
        0.53,
        0.72,
        0.42,
        0.22,
        "U(1) gauge field\nAμ, E, B with sources ρ,J\n(Maxwell; adopted)",
        fc="#f0fff4",
        ec="#2f855a",
    )
    _add_box(
        ax1,
        0.53,
        0.44,
        0.42,
        0.20,
        "Electrostatics\n∇·E = ρ/ε0 ⇒ Φ ∝ 1/r\n(Coulomb law; fixed)",
        fc="#fffaf0",
        ec="#b7791f",
    )
    _add_box(
        ax1,
        0.53,
        0.16,
        0.42,
        0.22,
        "Photons (EM waves)\n2 transverse polarizations\nenergy flow: S=(1/μ0)E×B\n(light propagation uses n(P) in coordinate time)",
        fc="#fff5f5",
        ec="#c53030",
    )
    _add_box(
        ax1,
        0.05,
        0.16,
        0.42,
        0.44,
        "Adopt + check (this paper)\n- adopt Maxwell/QED (independent)\n- adopt α P-independent (κ=0)\n- constrain κ from data\nFuture: derive charge/gauge/EM coupling from P",
        fc="#f7fafc",
        ec="#4a5568",
    )

    _add_arrow(ax1, 0.47, 0.83, 0.53, 0.83)  # P-field -> U(1) context
    _add_arrow(ax1, 0.74, 0.72, 0.74, 0.64)  # U(1) -> electrostatics
    _add_arrow(ax1, 0.74, 0.44, 0.74, 0.38)  # electrostatics -> photons
    _add_arrow(ax1, 0.47, 0.33, 0.53, 0.33)  # safety -> photons

    fig.tight_layout()
    out_png = out_dir / "electromagnetism_minimal.png"
    fig.savefig(out_png)
    plt.close(fig)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.11",
        "title": "Electromagnetism minimal positioning",
        "constants": {
            "c_m_s": c,
            "G": G,
            "e_C": e_charge,
            "eps0_F_m": eps0,
            "k_e_Nm2_C2": k_e,
            "m_p_kg": m_p,
            "m_e_kg": m_e,
            "a0_m": a0,
        },
        "scale_check": {
            "at_r_m": a0,
            "U_coulomb_eV": float(u_c_ev),
            "U_gravity_eV": float(u_g_ev),
            "ratio_Uc_over_Ug": float(ratio),
            "note": "p–e interaction energy magnitudes at r=a0 (Bohr radius).",
        },
        "postulates": [
            {
                "id": "EM-P1",
                "kind": "degrees_of_freedom",
                "statement": "Introduce a U(1) gauge field (Aμ, E, B) with sources ρ,J as the minimal EM dof.",
            },
            {
                "id": "EM-P2",
                "kind": "static_limit",
                "statement": "Coulomb 1/r is fixed via Gauss law in the electrostatic limit.",
            },
            {
                "id": "EM-P3",
                "kind": "photon",
                "statement": "Photons are transverse EM waves (two polarizations); energy flow is given by the Poynting vector.",
            },
            {
                "id": "EM-P4",
                "kind": "coupling_stance",
                "statement": "Local Maxwell/QED is kept unchanged at this stage. α is treated as P-independent (κ=0), and κ≠0 is constrained by alpha_p_dependence_constraint.json. P-model affects photon paths via n(P) (coordinate-time description).",
            },
        ],
        "open_problems": [
            "Derive U(1) charge/gauge symmetry from P-model wave structure (phase/topology)",
            "Couple EM stress-energy to P consistently (beyond phenomenology)",
            "Unify photon propagation n(P) with Maxwell-in-background derivation (beyond geometric optics)",
        ],
        "outputs": {
            "figure_png": str(out_png.relative_to(root)),
        },
        "sources": {
            "doc": "doc/quantum/16_electromagnetism_charge_maxwell_photon.md",
            "roadmap": "doc/ROADMAP.md (Step 7.11)",
            "alpha_constraint": "output/quantum/alpha_p_dependence_constraint.json",
        },
    }

    out_json = out_dir / "electromagnetism_minimal_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
