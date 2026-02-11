from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    payload = _load_json(extracted)
    consts = payload.get("constants")
    if not isinstance(consts, dict):
        raise SystemExit(f"[fail] invalid extracted_values.json: constants is not a dict: {extracted}")
    return {k: v for k, v in consts.items() if isinstance(v, dict)}


def _get_const_si(consts: dict[str, dict[str, object]], key: str) -> tuple[float, float]:
    obj = consts.get(key)
    if not isinstance(obj, dict):
        raise KeyError(key)
    return float(obj["value_si"]), float(obj["sigma_si"])


def _load_lambda_pi_fm(*, root: Path) -> float:
    """
    Prefer the already-fixed PDG baseline output (Step 7.13.1).
    Fallback to the standard π± Compton length if missing.
    """
    metrics = root / "output" / "public" / "quantum" / "qcd_hadron_masses_baseline_metrics.json"
    if metrics.exists():
        j = _load_json(metrics)
        rows = j.get("rows")
        if isinstance(rows, list):
            for r in rows:
                if not isinstance(r, dict):
                    continue
                if r.get("label") == "π±":
                    lam = r.get("compton_lambda_fm")
                    if isinstance(lam, (int, float)) and math.isfinite(float(lam)) and float(lam) > 0:
                        return float(lam)
    # π±: m≈139.57039 MeV, ħc≈197.32698 MeV·fm → λ≈1.4138 fm
    return 1.413816930654131


def _exp_coupling_mev(*, r_fm: float, r0_fm: float, j_at_r0_mev: float, L_fm: float) -> float:
    if not (math.isfinite(r_fm) and r_fm >= 0):
        return float("nan")
    if not (math.isfinite(r0_fm) and r0_fm > 0 and math.isfinite(j_at_r0_mev) and j_at_r0_mev > 0 and math.isfinite(L_fm) and L_fm > 0):
        return float("nan")
    # J(R)=J0 exp(-R/L) with J(R0)=j_at_r0 -> J0=j_at_r0 exp(R0/L)
    # => J(R)=j_at_r0 exp((R0-R)/L)
    return float(j_at_r0_mev * math.exp((r0_fm - r_fm) / L_fm))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r0-fm", type=float, default=2.0, help="reference separation R0 (fm) where E_B=ħΔω is anchored")
    ap.add_argument("--rmax-fm", type=float, default=10.0, help="plot max radius (fm)")
    ap.add_argument(
        "--L-fm",
        type=float,
        default=float("nan"),
        help="decay length L (fm). If omitted/NaN, use λ_pi from the fixed PDG baseline output.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    consts = _load_nist_codata_constants(root=root)
    mp_kg, sigma_mp = _get_const_si(consts, "mp")
    mn_kg, sigma_mn = _get_const_si(consts, "mn")
    md_kg, sigma_md = _get_const_si(consts, "md")

    # Exact SI constants:
    c = 299_792_458.0
    e_charge = 1.602_176_634e-19
    h = 6.626_070_15e-34
    hbar_si = h / (2.0 * math.pi)

    # Deuteron binding energy (CODATA via NIST extraction, same as scripts/quantum/nuclear_binding_deuteron.py)
    dm = (mp_kg + mn_kg - md_kg)
    sigma_dm = math.sqrt(sigma_mp**2 + sigma_mn**2 + sigma_md**2)
    b_j = dm * (c**2)
    sigma_b_j = sigma_dm * (c**2)
    b_mev = b_j / (1e6 * e_charge)
    sigma_b_mev = sigma_b_j / (1e6 * e_charge)

    # Reduced mass in energy units (MeV)
    mu_kg = (mp_kg * mn_kg) / (mp_kg + mn_kg)
    mu_mev = (mu_kg * (c**2)) / (1e6 * e_charge)

    # Two-mode convention (fixed for this roadmap pack):
    #   Δω = 2 J(R0),  E_B = ħ Δω  ->  J_E(R0) = E_B / 2.
    j_at_r0_mev = 0.5 * b_mev

    # Frequency scales (optional bookkeeping)
    hbar_mev_s = hbar_si / (1e6 * e_charge)
    delta_omega_r0 = b_mev / hbar_mev_s  # E_B = ħ Δω
    j_freq_r0 = 0.5 * delta_omega_r0

    # Range proxy
    L_fm = float(args.L_fm)
    if not (math.isfinite(L_fm) and L_fm > 0):
        L_fm = _load_lambda_pi_fm(root=root)

    r0_fm = float(args.r0_fm)
    rmax_fm = float(args.rmax_fm)
    if not (math.isfinite(r0_fm) and r0_fm > 0):
        raise SystemExit("[fail] --r0-fm must be positive")
    if not (math.isfinite(rmax_fm) and rmax_fm > r0_fm):
        raise SystemExit("[fail] --rmax-fm must be > r0")

    # Sample
    rs = [i * 0.02 for i in range(0, int(rmax_fm / 0.02) + 1)]
    js = [_exp_coupling_mev(r_fm=r, r0_fm=r0_fm, j_at_r0_mev=j_at_r0_mev, L_fm=L_fm) for r in rs]
    splits = [2.0 * j for j in js]

    # Simple scale ratios (order-of-magnitude check)
    ratio_b_to_mu = (b_mev / mu_mev) if (mu_mev and math.isfinite(mu_mev) and mu_mev > 0) else float("nan")

    # Plot
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.8, 5.2), dpi=160, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, wspace=0.22)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(rs, js, lw=2.2, color="tab:blue", label="J_E(R) (MeV)")
    ax0.axvline(r0_fm, color="0.35", lw=1.2, ls=":", alpha=0.85, label="R0")
    ax0.axhline(j_at_r0_mev, color="tab:blue", lw=1.2, ls="--", alpha=0.65, label="J_E(R0)=E_B/2")
    ax0.set_xlabel("R (fm)")
    ax0.set_ylabel("coupling energy J_E (MeV)")
    ax0.set_title("Near-field interference proxy: coupling envelope")
    ax0.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax0.legend(frameon=True, fontsize=9, loc="upper right")
    ax0.text(
        0.02,
        0.98,
        (
            "Two-mode convention (fixed):\n"
            "  Δω = 2 J(R0)\n"
            "  E_B = ħ Δω  =>  J_E(R0)=E_B/2\n\n"
            f"E_B ≈ {b_mev:.6f} ± {sigma_b_mev:.6f} MeV\n"
            f"R0 = {r0_fm:.2f} fm\n"
            f"L = {L_fm:.3f} fm (λ_π proxy)"
        ),
        transform=ax0.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.85"},
    )

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(rs, splits, lw=2.2, color="tab:orange", label="E_split(R)=2J_E(R)")
    ax1.axvline(r0_fm, color="0.35", lw=1.2, ls=":", alpha=0.85)
    ax1.axhline(b_mev, color="tab:orange", lw=1.2, ls="--", alpha=0.65, label="E_B at R0")
    ax1.set_xlabel("R (fm)")
    ax1.set_ylabel("splitting energy E_split (MeV)")
    ax1.set_title("Binding scale from mode splitting")
    ax1.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax1.legend(frameon=True, fontsize=9, loc="upper right")
    ax1.text(
        0.02,
        0.02,
        (
            f"Δω(R0) = E_B/ħ ≈ {delta_omega_r0:.3e} s⁻¹\n"
            f"J(R0) = Δω/2 ≈ {j_freq_r0:.3e} s⁻¹\n"
            f"E_B/(μc²) ≈ {ratio_b_to_mu:.3e}"
        ),
        transform=ax1.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.85"},
    )

    fig.suptitle("Phase 7 / Step 7.13.16: nuclear force as near-field interference (two-mode proxy)", y=1.02)

    out_png = out_dir / "nuclear_near_field_interference_two_mode_model.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    out_json = out_dir / "nuclear_near_field_interference_two_mode_model_metrics.json"
    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.13.16",
        "inputs": {
            "codata_nist_extracted": "data/quantum/sources/nist_codata_2022_nuclear_baseline/extracted_values.json",
            "pdg_baseline_metrics": "output/public/quantum/qcd_hadron_masses_baseline_metrics.json",
        },
        "model": {
            "two_mode_convention": "Delta_omega = 2 J(R0); E_B = hbar * Delta_omega; J_E(R0) = E_B/2",
            "J_energy_envelope": "J_E(R) = J_E(R0) * exp((R0-R)/L)",
            "range_proxy": "L ≈ lambda_pi (from PDG baseline output if present)",
        },
        "parameters": {"R0_fm": r0_fm, "L_fm": L_fm, "Rmax_fm": rmax_fm},
        "derived": {
            "binding_energy": {"B_MeV": {"value": b_mev, "sigma": sigma_b_mev}},
            "reduced_mass_MeV": mu_mev,
            "J_energy_at_R0_MeV": j_at_r0_mev,
            "Delta_omega_at_R0_per_s": delta_omega_r0,
            "J_frequency_at_R0_per_s": j_freq_r0,
            "ratio_B_over_mu_c2": ratio_b_to_mu,
        },
        "outputs": {"png": str(out_png)},
        "notes": [
            "This is a scale-indicator/proxy plot: it freezes a minimal I/F for interpreting the coupling J(R) as a near-field interference overlap envelope.",
            "The pn vs nn/pp selection (symmetry) and full falsification conditions are specified separately (ROADMAP: 7.13.16.6–7.13.16.7).",
        ],
    }
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")


if __name__ == "__main__":
    main()

