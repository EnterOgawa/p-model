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


def _solve_bound_x(*, kappa_fm1: float, r_fm: float) -> float:
    """
    Solve x in (pi/2, pi) for the s-wave square-well bound-state condition:

      k cot(kR) = -kappa, with k = x/R

    i.e.

      x cot x + kappa R = 0.
    """
    if not (math.isfinite(kappa_fm1) and kappa_fm1 > 0 and math.isfinite(r_fm) and r_fm > 0):
        raise ValueError("invalid kappa or R")

    lo = (math.pi / 2.0) + 1e-7
    hi = math.pi - 1e-7

    def f(x: float) -> float:
        return (x / math.tan(x)) + (kappa_fm1 * r_fm)

    flo = f(lo)
    fhi = f(hi)
    if not (flo > 0 and fhi < 0):
        raise ValueError(f"no bracket for bound x: f(lo)={flo}, f(hi)={fhi}, kappaR={kappa_fm1*r_fm}")

    for _ in range(96):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if fmid == 0 or (hi - lo) < 1e-15:
            return mid
        if fmid > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _square_well_from_r(*, mu_mev: float, b_mev: float, r_fm: float, hbarc_mev_fm: float) -> dict[str, float]:
    """
    Given B (fixed) and R, solve the well depth V0 by the s-wave bound-state condition.

    Returns: V0 (MeV), x (dimensionless), k (fm^-1), kappa (fm^-1).
    """
    if not (mu_mev > 0 and b_mev > 0 and r_fm > 0 and hbarc_mev_fm > 0):
        raise ValueError("invalid inputs")

    kappa = math.sqrt(2.0 * mu_mev * b_mev) / hbarc_mev_fm
    x = _solve_bound_x(kappa_fm1=kappa, r_fm=r_fm)
    k = x / r_fm
    v0 = b_mev + (hbarc_mev_fm**2) * (k**2) / (2.0 * mu_mev)
    return {"V0_mev": float(v0), "x": float(x), "k_fm1": float(k), "kappa_fm1": float(kappa)}


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

    # Exact SI constants:
    c = 299_792_458.0
    e_charge = 1.602_176_634e-19
    h = 6.626_070_15e-34
    hbar = h / (2.0 * math.pi)

    # Binding energy (CODATA baseline)
    dm = (mp + mn - md)
    sigma_dm = math.sqrt(sigma_mp**2 + sigma_mn**2 + sigma_md**2)
    b_j = dm * (c**2)
    sigma_b_j = sigma_dm * (c**2)
    b_mev = b_j / (1e6 * e_charge)
    sigma_b_mev = sigma_b_j / (1e6 * e_charge)

    # Reduced mass
    mu_kg = (mp * mn) / (mp + mn)
    mu_mev = (mu_kg * (c**2)) / (1e6 * e_charge)

    # Tail scale (kappa) and frequency mapping
    kappa_si = math.sqrt(2.0 * mu_kg * abs(b_j)) / hbar if b_j > 0 else float("nan")
    inv_kappa_fm = (1.0 / kappa_si) * 1e15 if (math.isfinite(kappa_si) and kappa_si > 0) else float("nan")
    delta_omega_per_s = (b_j / hbar) if (b_j > 0) else float("nan")
    j_freq_per_s = 0.5 * delta_omega_per_s if math.isfinite(delta_omega_per_s) else float("nan")

    # Scale proxies used in nuclear steps
    qcd_metrics_path = root / "output" / "public" / "quantum" / "qcd_hadron_masses_baseline_metrics.json"
    qcd_metrics = _load_json(qcd_metrics_path) if qcd_metrics_path.exists() else {}
    hbarc_mev_fm = float(qcd_metrics.get("constants", {}).get("hbar_c_mev_fm", 197.3269804))

    lambda_pi_fm: float | None = None
    if isinstance(qcd_metrics.get("rows"), list):
        for row in qcd_metrics["rows"]:
            if isinstance(row, dict) and row.get("label") == "π±":
                try:
                    lambda_pi_fm = float(row.get("compton_lambda_fm"))
                except Exception:
                    lambda_pi_fm = None
                break

    rd_fm = rd_m * 1e15

    ranges: list[dict[str, object]] = [
        {"label": "R = λπ (π± Compton)", "R_fm": lambda_pi_fm},
        {"label": "R = r_d (charge rms)", "R_fm": rd_fm},
        {"label": "R = 2.0 fm (proxy)", "R_fm": 2.0},
        {"label": "R = 1/κ (tail scale)", "R_fm": inv_kappa_fm},
    ]

    fits: list[dict[str, object]] = []
    for r in ranges:
        r_fm = r.get("R_fm")
        if r_fm is None:
            continue
        r_fm = float(r_fm)
        if not (math.isfinite(r_fm) and r_fm > 0):
            continue
        sw = _square_well_from_r(mu_mev=mu_mev, b_mev=b_mev, r_fm=r_fm, hbarc_mev_fm=hbarc_mev_fm)
        fits.append(
            {
                "label": str(r.get("label")),
                "R_fm": r_fm,
                "kappaR": sw["kappa_fm1"] * r_fm,
                "x": sw["x"],
                "k_fm1": sw["k_fm1"],
                "V0_mev": sw["V0_mev"],
            }
        )

    # Plot
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.8, 4.4), dpi=160)
    gs = fig.add_gridspec(1, 2, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    ax0.text(
        0.0,
        1.0,
        "deuteron (pn) two-body: bound-state scales (frozen)",
        ha="left",
        va="top",
        fontsize=12,
        weight="bold",
        transform=ax0.transAxes,
    )
    ax0.text(
        0.0,
        0.86,
        (
            f"B = {b_mev:.6f} ± {sigma_b_mev:.6f} MeV (CODATA mass defect)\n"
            f"Δω = B/ħ ≈ {delta_omega_per_s:.3e} 1/s\n"
            f"J (2-mode I/F) = Δω/2 ≈ {j_freq_per_s:.3e} 1/s\n"
            f"1/κ (tail) ≈ {inv_kappa_fm:.3f} fm,  r_d ≈ {rd_fm:.3f} fm"
        ),
        ha="left",
        va="top",
        fontsize=10,
        transform=ax0.transAxes,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.85"},
    )
    ax0.text(
        0.0,
        0.46,
        (
            "Square-well example (s-wave):\n"
            "  V(r)=−V0 (r<R), 0 (r≥R)\n"
            "  k cot(kR) = −κ,  κ = sqrt(2μB)/ħ\n"
            "This is an operational I/F for the standing-wave (bound) condition;\n"
            "it does not claim the nuclear force is literally a square well."
        ),
        ha="left",
        va="top",
        fontsize=10,
        transform=ax0.transAxes,
    )

    ax1 = fig.add_subplot(gs[0, 1])
    xs = [float(f["R_fm"]) for f in fits]
    ys = [float(f["V0_mev"]) for f in fits]
    labels = [str(f["label"]) for f in fits]

    ax1.plot(xs, ys, marker="o", lw=1.8)
    for x, y, lab in zip(xs, ys, labels):
        ax1.text(x, y, lab.replace("R = ", ""), fontsize=8, ha="left", va="bottom", rotation=15)
    ax1.set_xlabel("well range R (fm)")
    ax1.set_ylabel("required depth V0 (MeV)")
    ax1.set_title("Square-well depth required to support B (illustration)")
    ax1.grid(True, ls=":", lw=0.6, alpha=0.6)

    fig.suptitle("Phase 7 / Step 7.13.17.2: deuteron Δω mapping via 2-body boundary condition", y=1.02)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.14, wspace=0.28)

    out_png = out_dir / "nuclear_binding_energy_frequency_mapping_deuteron_two_body.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # Sources / traceability
    codata_dir = root / "data" / "quantum" / "sources" / "nist_codata_2022_nuclear_baseline"
    codata_manifest = codata_dir / "manifest.json"
    codata_extracted = codata_dir / "extracted_values.json"

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.13.17.2",
        "sources": [
            {
                "dataset": "NIST Cuu CODATA constants (mp,mn,md,rd)",
                "local_manifest": str(codata_manifest),
                "local_manifest_sha256": _sha256(codata_manifest) if codata_manifest.exists() else None,
                "local_extracted": str(codata_extracted),
                "local_extracted_sha256": _sha256(codata_extracted) if codata_extracted.exists() else None,
            },
            {
                "dataset": "PDG RPP 2024 mass baseline (for λπ proxy and ħc constant)",
                "local_metrics": str(qcd_metrics_path) if qcd_metrics_path.exists() else None,
                "local_metrics_sha256": _sha256(qcd_metrics_path) if qcd_metrics_path.exists() else None,
            },
        ],
        "constants": {
            "c_m_per_s": c,
            "h_J_s": h,
            "hbar_J_s": hbar,
            "e_C": e_charge,
            "hbarc_MeV_fm": hbarc_mev_fm,
        },
        "derived": {
            "binding_energy": {
                "B_J": {"value": b_j, "sigma": sigma_b_j},
                "B_MeV": {"value": b_mev, "sigma": sigma_b_mev},
            },
            "reduced_mass_mu_c2_MeV": mu_mev,
            "kappa_1_per_m": kappa_si,
            "inv_kappa_fm": inv_kappa_fm,
            "deuteron_charge_rms_radius_fm": rd_fm,
            "delta_omega_per_s": delta_omega_per_s,
            "two_mode_J_per_s": j_freq_per_s,
            "lambda_pi_pm_fm": lambda_pi_fm,
        },
        "square_well_example": {
            "condition": "k cot(kR) = -kappa (s-wave; bound state)",
            "fits_from_R": fits,
            "notes": [
                "This is an illustrative operational I/F for the boundary condition; it does not claim a square-well force model.",
                "R is treated as a knob (e.g., λπ, r_d, proxy R0, tail scale) to visualize how the required depth scales.",
            ],
        },
        "outputs": {"png": str(out_png)},
    }

    out_json = out_dir / "nuclear_binding_energy_frequency_mapping_deuteron_two_body_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")


if __name__ == "__main__":
    main()
