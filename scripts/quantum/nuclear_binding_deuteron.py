from __future__ import annotations

import json
import math
from pathlib import Path


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_nist_codata_constants(*, root: Path) -> dict[str, dict[str, object]]:
    src_dir = root / "data" / "quantum" / "sources" / "nist_codata_2022_nuclear_baseline"
    extracted = src_dir / "extracted_values.json"
    if not extracted.exists():
        raise SystemExit(
            "[fail] missing extracted CODATA constants.\n"
            "Run:\n"
            "  python -B scripts/quantum/fetch_nuclear_binding_sources.py\n"
            f"Expected: {extracted}"
        )
    payload = json.loads(extracted.read_text(encoding="utf-8"))
    consts = payload.get("constants")
    if not isinstance(consts, dict):
        raise SystemExit(f"[fail] invalid extracted_values.json: constants is not a dict: {extracted}")
    return {k: v for k, v in consts.items() if isinstance(v, dict)}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    consts = _load_nist_codata_constants(root=root)
    need = ["mp", "mn", "md", "rd"]
    for k in need:
        if k not in consts:
            raise SystemExit(f"[fail] missing constant {k!r} in extracted_values.json")

    mp = float(consts["mp"]["value_si"])
    mn = float(consts["mn"]["value_si"])
    md = float(consts["md"]["value_si"])
    sigma_mp = float(consts["mp"]["sigma_si"])
    sigma_mn = float(consts["mn"]["sigma_si"])
    sigma_md = float(consts["md"]["sigma_si"])
    rd_m = float(consts["rd"]["value_si"])
    sigma_rd_m = float(consts["rd"]["sigma_si"])

    # Exact SI constants:
    c = 299_792_458.0
    e_charge = 1.602_176_634e-19
    h = 6.626_070_15e-34
    hbar = h / (2.0 * math.pi)

    # Deuteron binding energy from mass defect: B = (m_p + m_n - m_d) c^2
    dm = (mp + mn - md)
    sigma_dm = math.sqrt(sigma_mp**2 + sigma_mn**2 + sigma_md**2)
    b_j = dm * (c**2)
    sigma_b_j = sigma_dm * (c**2)
    b_mev = b_j / (1e6 * e_charge)
    sigma_b_mev = sigma_b_j / (1e6 * e_charge)

    # Reduced mass μ and bound-state tail length scale 1/κ where B = ħ^2 κ^2 / (2μ)
    mu = (mp * mn) / (mp + mn)
    kappa = math.sqrt(2.0 * mu * abs(b_j)) / hbar if b_j > 0 else float("nan")
    inv_kappa_m = (1.0 / kappa) if (kappa and math.isfinite(kappa) and kappa > 0) else float("nan")
    inv_kappa_fm = inv_kappa_m * 1e15

    rd_fm = rd_m * 1e15
    sigma_rd_fm = sigma_rd_m * 1e15

    # Effective potential scale if one writes (semi-classically) V = m φ:
    # |φ|/c^2 ~ B / (m c^2). This is only a bookkeeping number here.
    phi_over_c2 = b_j / (mu * c**2) if mu > 0 else float("nan")

    # Plot
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.8, 4.0), dpi=160)
    gs = fig.add_gridspec(1, 2, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.errorbar([0.0], [b_mev], yerr=[sigma_b_mev], fmt="o", capsize=5, lw=1.8)
    ax0.set_xticks([0.0])
    ax0.set_xticklabels(["deuteron"])
    ax0.set_ylabel("binding energy B (MeV)")
    ax0.set_title("Mass defect baseline (CODATA via NIST)")
    ax0.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax0.text(
        0.02,
        0.98,
        (
            "B = (m_p + m_n − m_d)c²\n"
            f"B ≈ {b_mev:.6f} ± {sigma_b_mev:.6f} MeV\n"
            f"|φ|/c² (bookkeeping) ≈ {abs(phi_over_c2):.3e}"
        ),
        transform=ax0.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
    )

    ax1 = fig.add_subplot(gs[0, 1])
    x = [0.0, 1.0]
    y = [rd_fm, inv_kappa_fm]
    yerr = [sigma_rd_fm, 0.0]
    ax1.errorbar([x[0]], [y[0]], yerr=[yerr[0]], fmt="o", capsize=5, lw=1.8, label="r_d (charge rms)")
    ax1.plot([x[1]], [y[1]], marker="s", lw=0.0, label="1/κ from B (tail scale)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["r_d", "1/κ"])
    ax1.set_ylabel("length scale (fm)")
    ax1.set_title("Size constraints (radius vs binding tail)")
    ax1.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax1.legend(frameon=True, fontsize=9, loc="upper right")
    ax1.text(
        0.02,
        0.02,
        "κ = sqrt(2 μ B) / ħ",
        transform=ax1.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
    )

    fig.suptitle("Phase 7 / Step 7.9: deuteron nuclear baseline (observables fixed)", y=1.03)
    fig.tight_layout()

    out_png = out_dir / "nuclear_binding_deuteron.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # Sources and traceability
    src_dir = root / "data" / "quantum" / "sources" / "nist_codata_2022_nuclear_baseline"
    manifest = src_dir / "manifest.json"
    extracted = src_dir / "extracted_values.json"

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.9",
        "sources": [
            {
                "dataset": "NIST Cuu CODATA constants (mp,mn,md,rd)",
                "local_manifest": str(manifest),
                "local_manifest_sha256": _sha256(manifest) if manifest.exists() else None,
                "local_extracted": str(extracted),
                "local_extracted_sha256": _sha256(extracted) if extracted.exists() else None,
            }
        ],
        "assumptions": [
            "Independent uncertainties for mp,mn,md when propagating σ(B) (no covariance provided).",
            "The length scale 1/κ is the bound-state tail scale from a single-channel nonrelativistic estimate.",
            "The 'phi_over_c2' value is a bookkeeping ratio B/(μc^2) under the minimal coupling V=mφ; not a derived nuclear field yet.",
        ],
        "constants_from_nist_codata": {
            "mp_kg": {"value": mp, "sigma": sigma_mp},
            "mn_kg": {"value": mn, "sigma": sigma_mn},
            "md_kg": {"value": md, "sigma": sigma_md},
            "rd_m": {"value": rd_m, "sigma": sigma_rd_m},
        },
        "derived": {
            "mass_defect_kg": {"value": dm, "sigma": sigma_dm},
            "binding_energy": {
                "B_J": {"value": b_j, "sigma": sigma_b_j},
                "B_MeV": {"value": b_mev, "sigma": sigma_b_mev},
            },
            "reduced_mass_kg": mu,
            "kappa_1_per_m": kappa,
            "inv_kappa_m": inv_kappa_m,
            "inv_kappa_fm": inv_kappa_fm,
            "deuteron_charge_rms_radius_fm": {"value": rd_fm, "sigma": sigma_rd_fm},
            "phi_over_c2_bookkeeping": phi_over_c2,
        },
        "falsification": {
            "acceptance_criteria": [
                "Any proposed P-model nuclear effective equation/potential must support a pn bound state (deuteron) with B within 5σ of this baseline value.",
                "Any proposed P-model prediction for the deuteron size scale must be compatible with r_d and the tail scale 1/κ (within stated model assumptions).",
            ]
        },
        "outputs": {"png": str(out_png)},
        "notes": [
            "This step fixes nuclear baseline observables and traceable primary sources; it does not claim a first-principles derivation of nuclear forces yet.",
            "Next: extend primary data to np scattering (a_t, r_t, phase shifts) and derive the effective nuclear-scale constraint from the P-field model (to avoid 'just insert φ' criticism).",
        ],
    }

    out_json = out_dir / "nuclear_binding_deuteron_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")


if __name__ == "__main__":
    main()

