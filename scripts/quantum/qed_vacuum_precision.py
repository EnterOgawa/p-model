from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    # Casimir (Roy, Lin, Mohideen; quant-ph/9906062v3):
    # - Sphere diameter: 201.7 µm (main text)
    # - "average statistical precision ... 1% of the forces measured at the closest separation" (abstract)
    sphere_diameter_um: float = 201.7
    casimir_force_rel_precision: float = 0.01

    # Lamb shift scaling (Ivanov & Karshenboim; physics/0009069v1):
    # - Lamb shift scales as Z^4
    # - Unknown higher-order 2-loop terms scale as Z^6
    # Representative nuclear-size contributions to ΔE_nucl(2s) (Table 4; MHz):
    lamb_nucl_2s_mhz: tuple[tuple[str, int, float, float], ...] = (
        ("He (Z=2)", 2, 8.80, 0.12),
        ("N-14 (Z=7)", 7, 3145.0, 33.0),
        ("N-15 (Z=7)", 7, 3274.0, 30.0),
    )

    # Plot settings
    casimir_a_nm_min: float = 50.0
    casimir_a_nm_max: float = 2000.0


def casimir_pressure_parallel_plates_pa(a_m: float, *, h_j_s: float, c_m_s: float) -> float:
    # Ideal perfectly conducting parallel plates:
    # P = -π^2 ħ c / (240 a^4)
    if a_m <= 0:
        return float("nan")
    hbar = h_j_s / (2.0 * math.pi)
    return float(-(math.pi**2) * hbar * c_m_s / (240.0 * (a_m**4)))


def casimir_force_sphere_plate_n(a_m: float, *, radius_m: float, h_j_s: float, c_m_s: float) -> float:
    # Proximity force approximation (PFA) for ideal conductors:
    # F ≈ 2π R (E/A), with E/A = -π^2 ħ c / (720 a^3)
    # => F = -π^3 ħ c R / (360 a^3)
    if a_m <= 0 or radius_m <= 0:
        return float("nan")
    hbar = h_j_s / (2.0 * math.pi)
    return float(-(math.pi**3) * hbar * c_m_s * radius_m / (360.0 * (a_m**3)))


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest().upper()


def _extract_hydrogen_1s2s_frequency(*, pdf_path: Path) -> tuple[int, int, str]:
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    text = ""
    for page in reader.pages:
        text += "\n" + (page.extract_text() or "")
    text = text.replace("\u00a0", " ")

    # Example (Parthey et al. 2011; arXiv:1107.3101v1):
    #   f1S 2S= 2466061413187035(10)Hz
    pat = re.compile(
        r"f\s*1\s*S\s*(?:[-\u2013\u2212]\s*)?\s*2\s*S\s*=\s*([0-9][0-9\s]*)\(\s*(\d+)\s*\)\s*Hz",
        re.I,
    )
    m = pat.search(text)
    if not m:
        raise RuntimeError(f"[fail] could not extract 1S–2S frequency from PDF text: {pdf_path}")

    raw_val = m.group(1)
    val_digits = re.sub(r"\s+", "", raw_val).strip()
    if not val_digits.isdigit():
        raise RuntimeError(f"[fail] invalid extracted digits: {val_digits!r}")

    f_hz = int(val_digits)
    sigma_hz = int(m.group(2))
    return f_hz, sigma_hz, m.group(0)


def _extract_alpha_inverse(*, pdf_path: Path) -> tuple[float, float, str]:
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    text = ""
    for page in reader.pages:
        text += "\n" + (page.extract_text() or "")
    text = text.replace("\u00a0", " ")

    # Example:
    #   α−1= 137.035 999 084 (51)
    #   α−1= 137.03599945(62)
    pat = re.compile(
        r"(?:α|alpha)\s*[-\u2013\u2212]\s*1\s*=\s*([0-9][0-9.\s]*)\(\s*(\d+)\s*\)",
        re.I,
    )
    m = pat.search(text)
    if not m:
        raise RuntimeError(f"[fail] could not extract alpha^{-1} from PDF text: {pdf_path}")

    raw_val = m.group(1)
    val_s = re.sub(r"\s+", "", raw_val).strip()
    try:
        val = float(val_s)
    except Exception as e:
        raise RuntimeError(f"[fail] invalid extracted alpha^{-1}: {val_s!r}") from e

    decimals = 0
    if "." in val_s:
        decimals = len(val_s.split(".", 1)[1])
    unc_digits = m.group(2)
    try:
        unc_int = int(unc_digits)
    except Exception as e:
        raise RuntimeError(f"[fail] invalid extracted alpha^{-1} uncertainty digits: {unc_digits!r}") from e

    sigma = float(unc_int) * (10.0 ** (-decimals))
    return val, sigma, m.group(0)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()

    # Constants (SI)
    c = 299_792_458.0
    h = 6.626_070_15e-34  # exact
    G = 6.674_30e-11
    m_p = 1.672_621_923_69e-27
    m_e = 9.109_383_701_5e-31
    a0 = 5.291_772_109_03e-11  # Bohr radius
    e_charge = 1.602_176_634e-19  # exact

    local_casimir_pdf = root / "data" / "quantum" / "sources" / "arxiv_quant-ph_9906062v3.pdf"
    local_lamb_pdf = root / "data" / "quantum" / "sources" / "arxiv_physics_0009069v1.pdf"
    local_1s2s_pdf = root / "data" / "quantum" / "sources" / "arxiv_1107.3101v1.pdf"
    local_recoil_pdf = root / "data" / "quantum" / "sources" / "arxiv_0812.3139v1.pdf"
    local_g2_pdf = root / "data" / "quantum" / "sources" / "arxiv_0801.1134v2.pdf"

    if not local_1s2s_pdf.exists():
        raise SystemExit(
            f"[fail] missing primary source PDF for H 1S–2S: {local_1s2s_pdf}\n"
            "Run: python -B scripts/quantum/fetch_qed_vacuum_precision_sources.py"
        )

    f_1s2s_hz, sigma_1s2s_hz, f_1s2s_match = _extract_hydrogen_1s2s_frequency(pdf_path=local_1s2s_pdf)
    frac_1s2s = float(sigma_1s2s_hz) / float(f_1s2s_hz)
    sigma_1s2s_e_ev = float(h * float(sigma_1s2s_hz) / e_charge)

    alpha_inv_recoil, alpha_inv_recoil_sigma, alpha_inv_recoil_match = _extract_alpha_inverse(pdf_path=local_recoil_pdf)
    alpha_inv_g2, alpha_inv_g2_sigma, alpha_inv_g2_match = _extract_alpha_inverse(pdf_path=local_g2_pdf)
    alpha_inv_delta = float(alpha_inv_recoil - alpha_inv_g2)
    alpha_inv_delta_sigma = float(math.sqrt(alpha_inv_recoil_sigma**2 + alpha_inv_g2_sigma**2))
    alpha_inv_z = float(alpha_inv_delta / alpha_inv_delta_sigma) if alpha_inv_delta_sigma > 0 else float("nan")

    # Interpretive parameter: effective fractional scaling needed in the recoil-based alpha to match the g-2 based alpha.
    # alpha ∝ sqrt(h/m) for recoil inference ⇒ alpha_inv ∝ 1/sqrt(1+epsilon).
    epsilon_required = float((alpha_inv_g2 / alpha_inv_recoil) ** 2 - 1.0)

    # --- Casimir (sphere-plane) ---
    radius_m = (cfg.sphere_diameter_um * 1e-6) / 2.0
    a_nm = np.geomspace(cfg.casimir_a_nm_min, cfg.casimir_a_nm_max, 240)
    a_m = a_nm * 1e-9
    force_n = np.asarray([abs(casimir_force_sphere_plate_n(a, radius_m=radius_m, h_j_s=h, c_m_s=c)) for a in a_m])
    pressure_pa = np.asarray(
        [abs(casimir_pressure_parallel_plates_pa(a, h_j_s=h, c_m_s=c)) for a in a_m], dtype=float
    )

    sample_a_nm = np.array([100.0, 200.0, 500.0, 1000.0], dtype=float)
    sample_force_n = np.asarray(
        [abs(casimir_force_sphere_plate_n(a * 1e-9, radius_m=radius_m, h_j_s=h, c_m_s=c)) for a in sample_a_nm],
        dtype=float,
    )
    sample_force_unc_n = cfg.casimir_force_rel_precision * sample_force_n

    # Equivalent vacuum energy scale (parallel plates; magnitude).
    # u ~ |P| (same units), rho = u/c^2.
    sample_pressure_pa = np.asarray(
        [abs(casimir_pressure_parallel_plates_pa(a * 1e-9, h_j_s=h, c_m_s=c)) for a in sample_a_nm],
        dtype=float,
    )
    sample_rho_kg_m3 = sample_pressure_pa / (c**2)

    # --- Lamb shift scaling (Z^4 vs Z^6) ---
    z = np.arange(1, 11, dtype=float)
    z4 = z**4
    z6 = z**6
    z4_rel = z4 / z4[0]
    z6_rel = z6 / z6[0]

    nucl_labels = [t[0] for t in cfg.lamb_nucl_2s_mhz]
    nucl_z = np.array([t[1] for t in cfg.lamb_nucl_2s_mhz], dtype=float)
    nucl_mhz = np.array([t[2] for t in cfg.lamb_nucl_2s_mhz], dtype=float)
    nucl_sigma_mhz = np.array([t[3] for t in cfg.lamb_nucl_2s_mhz], dtype=float)

    # --- Safety check: atomic-scale gravitational potential (order-of-magnitude) ---
    def _phi_nuc_over_c2(*, Z: int, A: int) -> float:
        # Hydrogen-like 1s characteristic scale: r ~ a0/Z.
        r = a0 / float(max(Z, 1))
        m_nuc = float(A) * m_p
        phi = G * m_nuc / r
        return float(phi / (c**2))

    def _deltaE_eV(*, phi_over_c2: float) -> float:
        # Potential energy scale for electron: ΔE ~ m_e * |φ|.
        phi = abs(float(phi_over_c2)) * (c**2)
        dE_j = m_e * phi
        return float(dE_j / e_charge)

    safety_cases = [
        {"label": "H (Z=1, A=1)", "Z": 1, "A": 1},
        {"label": "U-238 (Z=92, A=238)", "Z": 92, "A": 238},
    ]
    for sc in safety_cases:
        poc2 = _phi_nuc_over_c2(Z=int(sc["Z"]), A=int(sc["A"]))
        sc["phi_over_c2_abs"] = abs(float(poc2))
        sc["deltaE_eV_abs"] = _deltaE_eV(phi_over_c2=poc2)

    h_case = next((sc for sc in safety_cases if sc.get("Z") == 1), None)
    u_case = next((sc for sc in safety_cases if sc.get("Z") == 92), None)
    deltaE_grav_h_eV = None if not h_case else float(h_case.get("deltaE_eV_abs") or 0.0)
    ratio_sigma_to_grav_h = None
    if deltaE_grav_h_eV and deltaE_grav_h_eV > 0:
        ratio_sigma_to_grav_h = sigma_1s2s_e_ev / deltaE_grav_h_eV

    # Plot
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14.8, 8.0), dpi=160)
    gs = fig.add_gridspec(2, 2, wspace=0.26, hspace=0.32)

    # Panel A: Casimir force magnitude vs separation
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(a_nm, force_n, lw=2.0, label="ideal conductor (PFA)")
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel("separation a (nm)")
    ax0.set_ylabel("|F| (N)")
    ax0.set_title("Casimir: sphere–plate force scale")
    ax0.grid(True, which="both", ls=":", lw=0.6, alpha=0.6)
    ax0.legend(frameon=True, fontsize=9, loc="lower left")
    ax0.text(
        0.02,
        0.98,
        (
            f"R≈{radius_m*1e6:.1f} μm (Roy+2000)\n"
            "F≈π^3 ħ c R / (360 a^3)\n"
            f"reported precision ~{cfg.casimir_force_rel_precision*100:.0f}% (closest a)"
        ),
        transform=ax0.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
    )

    # Panel B: Lamb shift scaling factors
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(z, z4_rel, marker="o", lw=1.8, label="Lamb shift ~ Z^4")
    ax1.plot(z, z6_rel, marker="o", lw=1.8, label="unknown 2-loop ~ Z^6")
    ax1.set_yscale("log")
    ax1.set_xlabel("Z")
    ax1.set_ylabel("relative scale (Z=1 → 1)")
    ax1.set_title("Lamb shift: scaling (why Z>1 helps)")
    ax1.grid(True, which="both", ls=":", lw=0.6, alpha=0.6)
    ax1.legend(frameon=True, fontsize=9, loc="upper left")
    ax1.scatter([2, 7], [2**4, 7**4], s=45, zorder=5, color="#1f77b4")
    ax1.scatter([2, 7], [2**6, 7**6], s=45, zorder=5, color="#ff7f0e")
    ax1.text(
        0.02,
        0.02,
        "Source: physics/0009069v1 (scaling statements)",
        transform=ax1.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
    )

    # Panel C: nuclear-size contributions (example numbers) + atomic gravity order check
    ax2 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(nucl_mhz), dtype=float)
    ax2.errorbar(x, nucl_mhz, yerr=nucl_sigma_mhz, fmt="o", capsize=4, lw=1.6)
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(nucl_labels, rotation=15, ha="right")
    ax2.set_ylabel("ΔE_nucl(2s) (MHz)")
    ax2.set_title("Nuclear-size term (example; Table 4)")
    ax2.grid(True, which="both", ls=":", lw=0.6, alpha=0.6)
    ax2.text(
        0.02,
        0.98,
        "Used as an example of\nnon-QED systematics (radius)",
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
    )
    try:
        if h_case and u_case:
            extra = ""
            if deltaE_grav_h_eV is not None and deltaE_grav_h_eV > 0:
                extra = (
                    "\nH 1S–2S precision (Parthey+2011):\n"
                    f"σf={sigma_1s2s_hz} Hz ⇒ hσ≈{sigma_1s2s_e_ev:.2e} eV\n"
                    f"ratio (hσ / ΔE_grav(H))≈{ratio_sigma_to_grav_h:.1e}"
                )
            ax2.text(
                0.02,
                0.02,
                (
                    "Atomic gravity (order):\n"
                    f"|φ|/c^2 ~ {h_case['phi_over_c2_abs']:.1e} (H), {u_case['phi_over_c2_abs']:.1e} (U)\n"
                    f"ΔE ~ {h_case['deltaE_eV_abs']:.1e}–{u_case['deltaE_eV_abs']:.1e} eV"
                    + extra
                ),
                transform=ax2.transAxes,
                va="bottom",
                ha="left",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
            )
    except Exception:
        pass

    # Panel D: alpha^{-1} (recoil vs electron g-2)
    ax3 = fig.add_subplot(gs[1, 1])
    labels = ["Recoil (Rb; 0812.3139)", "g-2 (e−; 0801.1134)"]
    y = np.array([alpha_inv_recoil, alpha_inv_g2], dtype=float)
    yerr = np.array([alpha_inv_recoil_sigma, alpha_inv_g2_sigma], dtype=float)
    xx = np.arange(2, dtype=float)
    ax3.errorbar(xx, y, yerr=yerr, fmt="o", capsize=4, elinewidth=1.8, color="#1f77b4", ecolor="#1f77b4")
    ax3.set_xticks(xx)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel("alpha^{-1}")
    ax3.set_title("α precision cross-check (recoil vs g-2)")
    ax3.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax3.text(
        0.02,
        0.02,
        (
            f"Δ(alpha^-1) = {alpha_inv_delta:+.3e} ± {alpha_inv_delta_sigma:.3e} (z={alpha_inv_z:+.2f})\n"
            f"epsilon_needed (recoil) ≈ {epsilon_required*1e9:+.2f} ppb"
        ),
        transform=ax3.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
    )

    fig.suptitle(
        "Phase 7 / Step 7.8: vacuum + QED precision observables (Casimir, Lamb, H 1S–2S, α)", y=0.995
    )
    fig.tight_layout()

    out_png = out_dir / "qed_vacuum_precision.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.8",
        "sources": [
            {
                "topic": "casimir_force_measurement",
                "reference": "Roy, Lin, Mohideen (quant-ph/9906062v3): improved precision measurement of the Casimir force (AFM; Al sphere vs plate)",
                "url": "https://arxiv.org/abs/quant-ph/9906062",
                "local_pdf": str(local_casimir_pdf),
                "local_pdf_sha256": _sha256(local_casimir_pdf),
                "abstract_value": {"relative_precision_at_closest_separation": cfg.casimir_force_rel_precision},
                "main_text_value": {"sphere_diameter_um": cfg.sphere_diameter_um},
            },
            {
                "topic": "lamb_shift_scaling_and_definitions",
                "reference": "Ivanov & Karshenboim (physics/0009069v1): Lamb shift in light hydrogen-like atoms (definitions + scaling)",
                "url": "https://arxiv.org/abs/physics/0009069",
                "local_pdf": str(local_lamb_pdf),
                "local_pdf_sha256": _sha256(local_lamb_pdf),
                "key_points": [
                    "Lamb shift scales as Z^4",
                    "unknown higher-order two-loop terms scale as Z^6",
                ],
                "table4_nuclear_term_mhz": [
                    {"label": lab, "Z": int(zv), "deltaE_mhz": float(v), "sigma_mhz": float(s)}
                    for lab, zv, v, s in cfg.lamb_nucl_2s_mhz
                ],
            },
            {
                "topic": "hydrogen_1s2s_frequency",
                "reference": "Parthey et al. 2011 (1107.3101v1): Improved Measurement of the Hydrogen 1S–2S Transition Frequency",
                "url": "https://arxiv.org/abs/1107.3101",
                "local_pdf": str(local_1s2s_pdf),
                "local_pdf_sha256": _sha256(local_1s2s_pdf),
                "extracted_value": {
                    "f_1s2s_hz": int(f_1s2s_hz),
                    "sigma_hz": int(sigma_1s2s_hz),
                    "fractional_sigma": frac_1s2s,
                    "extract_match": f_1s2s_match,
                },
            },
            {
                "topic": "alpha_inverse_recoil_rb",
                "reference": "Bouchendira et al. 2008 (0812.3139v1): Determination of the fine structure constant with atom interferometry and Bloch oscillations",
                "url": "https://arxiv.org/abs/0812.3139",
                "local_pdf": str(local_recoil_pdf),
                "local_pdf_sha256": _sha256(local_recoil_pdf),
                "extracted_value": {
                    "alpha_inv": float(alpha_inv_recoil),
                    "sigma_alpha_inv": float(alpha_inv_recoil_sigma),
                    "fractional_sigma": float(alpha_inv_recoil_sigma / alpha_inv_recoil),
                    "extract_match": alpha_inv_recoil_match,
                },
            },
            {
                "topic": "alpha_inverse_electron_g2_qed",
                "reference": "Gabrielse et al. 2008 (0801.1134v2): New Measurement of the Electron Magnetic Moment and the Fine Structure Constant",
                "url": "https://arxiv.org/abs/0801.1134",
                "local_pdf": str(local_g2_pdf),
                "local_pdf_sha256": _sha256(local_g2_pdf),
                "extracted_value": {
                    "alpha_inv": float(alpha_inv_g2),
                    "sigma_alpha_inv": float(alpha_inv_g2_sigma),
                    "fractional_sigma": float(alpha_inv_g2_sigma / alpha_inv_g2),
                    "extract_match": alpha_inv_g2_match,
                },
            },
        ],
        "constants_si": {
            "c_m_per_s": c,
            "h_j_s": h,
            "hbar_j_s": h / (2.0 * math.pi),
            "G_m3_kg_s2": G,
            "m_p_kg": m_p,
            "m_e_kg": m_e,
            "a0_m": a0,
            "e_charge_c": e_charge,
        },
        "casimir": {
            "sphere_radius_m": radius_m,
            "formula_notes": [
                "Ideal plates: P = -π^2 ħ c / (240 a^4)",
                "Sphere-plate (PFA): F = -π^3 ħ c R / (360 a^3)",
            ],
            "sample_points": [
                {
                    "a_nm": float(a),
                    "force_n": float(fn),
                    "force_sigma_n_assuming_rel_precision": float(sfn),
                    "pressure_pa_plates": float(p),
                    "rho_equiv_kg_m3": float(rho),
                }
                for a, fn, sfn, p, rho in zip(
                    sample_a_nm.tolist(),
                    sample_force_n.tolist(),
                    sample_force_unc_n.tolist(),
                    sample_pressure_pa.tolist(),
                    sample_rho_kg_m3.tolist(),
                    strict=True,
                )
            ],
        },
        "lamb_shift": {
            "z_grid": list(map(int, z.tolist())),
            "z4_rel": list(map(float, z4_rel.tolist())),
            "z6_rel": list(map(float, z6_rel.tolist())),
            "nuclear_table4_mhz": [
                {"label": lab, "Z": int(zv), "deltaE_mhz": float(v), "sigma_mhz": float(s)}
                for lab, zv, v, s in cfg.lamb_nucl_2s_mhz
            ],
        },
        "atomic_gravity_safety": {
            "assumption_note": "Order-of-magnitude check under minimal coupling: use nuclear gravity only with r~a0/Z; ΔE~m_e|φ| (no new short-range coupling).",
            "cases": safety_cases,
        },
        "hydrogen_1s2s": {
            "f_hz": int(f_1s2s_hz),
            "sigma_hz": int(sigma_1s2s_hz),
            "fractional_sigma": frac_1s2s,
            "sigma_energy_eV": sigma_1s2s_e_ev,
            "ratio_sigma_energy_to_deltaE_grav_H": ratio_sigma_to_grav_h,
        },
        "alpha_precision": {
            "recoil": {
                "alpha_inv": float(alpha_inv_recoil),
                "sigma_alpha_inv": float(alpha_inv_recoil_sigma),
                "fractional_sigma": float(alpha_inv_recoil_sigma / alpha_inv_recoil),
                "extract_match": alpha_inv_recoil_match,
            },
            "g2": {
                "alpha_inv": float(alpha_inv_g2),
                "sigma_alpha_inv": float(alpha_inv_g2_sigma),
                "fractional_sigma": float(alpha_inv_g2_sigma / alpha_inv_g2),
                "extract_match": alpha_inv_g2_match,
            },
            "derived": {
                "delta_alpha_inv": float(alpha_inv_delta),
                "sigma_delta_alpha_inv": float(alpha_inv_delta_sigma),
                "z_score": float(alpha_inv_z),
                "epsilon_required": float(epsilon_required),
                "epsilon_units": "dimensionless (fractional scaling in recoil-based h/m)",
            },
        },
        "outputs": {"png": str(out_png)},
        "notes": [
            "This step fixes observables and primary sources; it does not claim a derivation of QED from P-model.",
            "A P-model quantum interpretation must reproduce Casimir/Lamb phenomenology (or be falsified by precision comparisons).",
        ],
    }
    out_json = out_dir / "qed_vacuum_precision_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")


if __name__ == "__main__":
    main()
