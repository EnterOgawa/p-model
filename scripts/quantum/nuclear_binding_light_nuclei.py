from __future__ import annotations

import csv
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


def _load_nist_codata_constants(*, root: Path, src_dirname: str) -> dict[str, dict[str, object]]:
    src_dir = root / "data" / "quantum" / "sources" / src_dirname
    extracted = src_dir / "extracted_values.json"
    if not extracted.exists():
        raise SystemExit(
            "[fail] missing extracted CODATA constants.\n"
            "Run:\n"
            f"  python -B scripts/quantum/fetch_nuclear_binding_sources.py --out-dirname {src_dirname} --include-light-nuclei\n"
            f"Expected: {extracted}"
        )
    payload = json.loads(extracted.read_text(encoding="utf-8"))
    consts = payload.get("constants")
    if not isinstance(consts, dict):
        raise SystemExit(f"[fail] invalid extracted_values.json: constants is not a dict: {extracted}")
    return {k: v for k, v in consts.items() if isinstance(v, dict)}


def _load_iaea_charge_radii_a3(*, root: Path, src_dirname: str) -> dict[str, dict[str, float]]:
    src_dir = root / "data" / "quantum" / "sources" / src_dirname
    extracted = src_dir / "extracted_values.json"
    if not extracted.exists():
        raise SystemExit(
            "[fail] missing extracted charge radii (A=3).\n"
            "Run:\n"
            f"  python -B scripts/quantum/fetch_nuclear_charge_radii_sources.py --out-dirname {src_dirname}\n"
            f"Expected: {extracted}"
        )
    payload = json.loads(extracted.read_text(encoding="utf-8"))
    radii = payload.get("radii")
    if not isinstance(radii, dict):
        raise SystemExit(f"[fail] invalid extracted_values.json: radii is not a dict: {extracted}")

    def unpack(key: str) -> dict[str, float]:
        item = radii.get(key)
        if not isinstance(item, dict):
            raise SystemExit(f"[fail] missing radii entry {key!r}: {extracted}")
        return {"value_fm": float(item["radius_fm"]), "sigma_fm": float(item["radius_sigma_fm"])}

    return {"t": unpack("t"), "h": unpack("h")}


def _binding_energy_from_masses_mev(
    *,
    mass_defect_kg: float,
    sigma_mass_defect_kg: float,
    c_m_per_s: float,
    e_charge_c: float,
) -> tuple[float, float]:
    b_j = float(mass_defect_kg) * float(c_m_per_s**2)
    sigma_b_j = float(sigma_mass_defect_kg) * float(c_m_per_s**2)
    b_mev = b_j / (1e6 * e_charge_c)
    sigma_b_mev = sigma_b_j / (1e6 * e_charge_c)
    return float(b_mev), float(sigma_b_mev)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dirname = "nist_codata_2022_nuclear_light_nuclei"
    consts = _load_nist_codata_constants(root=root, src_dirname=src_dirname)

    iaea_src_dirname = "iaea_charge_radii"
    a3_radii = _load_iaea_charge_radii_a3(root=root, src_dirname=iaea_src_dirname)

    need = ["mp", "mn", "md", "mt", "mh", "mal", "rp", "rd", "ral"]
    for k in need:
        if k not in consts:
            raise SystemExit(f"[fail] missing constant {k!r} in extracted_values.json ({src_dirname})")

    mp = float(consts["mp"]["value_si"])
    mn = float(consts["mn"]["value_si"])
    md = float(consts["md"]["value_si"])
    mt = float(consts["mt"]["value_si"])
    mh = float(consts["mh"]["value_si"])
    mal = float(consts["mal"]["value_si"])

    sigma_mp = float(consts["mp"]["sigma_si"])
    sigma_mn = float(consts["mn"]["sigma_si"])
    sigma_md = float(consts["md"]["sigma_si"])
    sigma_mt = float(consts["mt"]["sigma_si"])
    sigma_mh = float(consts["mh"]["sigma_si"])
    sigma_mal = float(consts["mal"]["sigma_si"])

    rp_fm = float(consts["rp"]["value_si"]) * 1e15
    sigma_rp_fm = float(consts["rp"]["sigma_si"]) * 1e15
    rd_fm = float(consts["rd"]["value_si"]) * 1e15
    sigma_rd_fm = float(consts["rd"]["sigma_si"]) * 1e15
    ral_fm = float(consts["ral"]["value_si"]) * 1e15
    sigma_ral_fm = float(consts["ral"]["sigma_si"]) * 1e15

    rt_fm = float(a3_radii["t"]["value_fm"])
    sigma_rt_fm = float(a3_radii["t"]["sigma_fm"])
    rh_fm = float(a3_radii["h"]["value_fm"])
    sigma_rh_fm = float(a3_radii["h"]["sigma_fm"])

    # Exact SI constants:
    c = 299_792_458.0
    e_charge = 1.602_176_634e-19

    nuclei = [
        {"key": "d", "label": "deuteron (A=2)", "A": 2, "Z": 1, "N": 1, "mass_kg": md},
        {"key": "t", "label": "triton (A=3)", "A": 3, "Z": 1, "N": 2, "mass_kg": mt},
        {"key": "h", "label": "helion (A=3)", "A": 3, "Z": 2, "N": 1, "mass_kg": mh},
        {"key": "alpha", "label": "alpha (A=4)", "A": 4, "Z": 2, "N": 2, "mass_kg": mal},
    ]

    derived: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for nu in nuclei:
        a = int(nu["A"])
        z = int(nu["Z"])
        n = int(nu["N"])
        m_nuc = float(nu["mass_kg"])

        dm = z * mp + n * mn - m_nuc
        # Add the nucleus mass uncertainty term explicitly.
        sigma_m_nuc = (
            sigma_md
            if nu["key"] == "d"
            else (sigma_mt if nu["key"] == "t" else (sigma_mh if nu["key"] == "h" else sigma_mal))
        )
        sigma_dm = math.sqrt((z * sigma_mp) ** 2 + (n * sigma_mn) ** 2 + sigma_m_nuc**2)

        b_mev, sigma_b_mev = _binding_energy_from_masses_mev(
            mass_defect_kg=dm, sigma_mass_defect_kg=sigma_dm, c_m_per_s=c, e_charge_c=e_charge
        )

        rows.append(
            {
                "key": str(nu["key"]),
                "label": str(nu["label"]),
                "A": a,
                "Z": z,
                "N": n,
                "binding_energy_MeV": float(b_mev),
                "binding_energy_sigma_MeV": float(sigma_b_mev),
                "binding_energy_per_nucleon_MeV": float(b_mev / a),
                "binding_energy_per_nucleon_sigma_MeV": float(sigma_b_mev / a),
            }
        )
        derived[str(nu["key"])] = {
            "A": a,
            "Z": z,
            "N": n,
            "mass_defect_kg": {"value": float(dm), "sigma": float(sigma_dm)},
            "binding_energy": {"B_MeV": {"value": float(b_mev), "sigma": float(sigma_b_mev)}},
        }

    # Plot
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.8, 7.2), dpi=160)
    gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.32)

    labels = [r["key"] for r in rows]
    x = list(range(len(labels)))
    b_vals = [float(r["binding_energy_MeV"]) for r in rows]
    b_sig = [float(r["binding_energy_sigma_MeV"]) for r in rows]
    ba_vals = [float(r["binding_energy_per_nucleon_MeV"]) for r in rows]
    ba_sig = [float(r["binding_energy_per_nucleon_sigma_MeV"]) for r in rows]

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.errorbar(x, b_vals, yerr=b_sig, fmt="o", capsize=4, lw=1.6)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylabel("binding energy B (MeV)")
    ax0.set_title("Mass defect baseline (CODATA via NIST)")
    ax0.grid(True, ls=":", lw=0.6, alpha=0.6)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.errorbar(x, ba_vals, yerr=ba_sig, fmt="o", capsize=4, lw=1.6, color="tab:green")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("B/A (MeV per nucleon)")
    ax1.set_title("Binding energy per nucleon")
    ax1.grid(True, ls=":", lw=0.6, alpha=0.6)

    ax2 = fig.add_subplot(gs[1, 0])
    r_labels = ["p", "d", "t", "h", "alpha"]
    rx = list(range(len(r_labels)))
    r_vals = [rp_fm, rd_fm, rt_fm, rh_fm, ral_fm]
    r_sig = [sigma_rp_fm, sigma_rd_fm, sigma_rt_fm, sigma_rh_fm, sigma_ral_fm]
    ax2.errorbar(rx, r_vals, yerr=r_sig, fmt="o", capsize=4, lw=1.6, color="tab:purple")
    ax2.set_xticks(rx)
    ax2.set_xticklabels(r_labels)
    ax2.set_ylabel("charge rms radius (fm)")
    ax2.set_title("Charge radii (CODATA + IAEA compilation)")
    ax2.grid(True, ls=":", lw=0.6, alpha=0.6)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    ax3.text(
        0.0,
        1.0,
        (
            "Definitions:\n"
            "  B = (Z m_p + N m_n − m_nucleus) c²\n"
            "Notes:\n"
            "  - Independent σ propagation (no covariance).\n"
            "  - p/d/alpha radii: CODATA via NIST Cuu.\n"
            "  - t/h radii (A=3): IAEA charge_radii.csv compilation (doi: 10.1016/j.adt.2011.12.006).\n"
        ),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    fig.suptitle("Phase 7 / Step 7.13.10: light nuclei baselines (A=2,3,4) incl. A=3 charge radii", y=1.02)

    out_png = out_dir / "nuclear_binding_light_nuclei.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    out_csv = out_dir / "nuclear_binding_light_nuclei.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "key",
                "label",
                "A",
                "Z",
                "N",
                "binding_energy_MeV",
                "binding_energy_sigma_MeV",
                "binding_energy_per_nucleon_MeV",
                "binding_energy_per_nucleon_sigma_MeV",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["key"],
                    r["label"],
                    int(r["A"]),
                    int(r["Z"]),
                    int(r["N"]),
                    f"{float(r['binding_energy_MeV']):.12g}",
                    f"{float(r['binding_energy_sigma_MeV']):.12g}",
                    f"{float(r['binding_energy_per_nucleon_MeV']):.12g}",
                    f"{float(r['binding_energy_per_nucleon_sigma_MeV']):.12g}",
                ]
            )

    # Sources and traceability
    src_dir = root / "data" / "quantum" / "sources" / src_dirname
    manifest = src_dir / "manifest.json"
    extracted = src_dir / "extracted_values.json"

    iaea_dir = root / "data" / "quantum" / "sources" / iaea_src_dirname
    iaea_manifest = iaea_dir / "manifest.json"
    iaea_extracted = iaea_dir / "extracted_values.json"

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.13.10",
        "sources": [
            {
                "dataset": "NIST Cuu CODATA constants (mp,mn,md,mt,mh,mal,rp,rd,ral)",
                "local_manifest": str(manifest),
                "local_manifest_sha256": _sha256(manifest) if manifest.exists() else None,
                "local_extracted": str(extracted),
                "local_extracted_sha256": _sha256(extracted) if extracted.exists() else None,
            }
            ,
            {
                "dataset": "IAEA nuclear charge radii compilation (charge_radii.csv)",
                "local_manifest": str(iaea_manifest),
                "local_manifest_sha256": _sha256(iaea_manifest) if iaea_manifest.exists() else None,
                "local_extracted": str(iaea_extracted),
                "local_extracted_sha256": _sha256(iaea_extracted) if iaea_extracted.exists() else None,
            },
        ],
        "assumptions": [
            "Independent uncertainties for CODATA constants (no covariance provided).",
            "Binding energy defined via nuclear mass defect using CODATA nuclear masses (not atomic masses).",
            "A=3 charge radii are taken from a dedicated compilation (IAEA radii database CSV).",
        ],
        "constants_from_nist_codata": {
            "mp_kg": {"value": mp, "sigma": sigma_mp},
            "mn_kg": {"value": mn, "sigma": sigma_mn},
            "md_kg": {"value": md, "sigma": sigma_md},
            "mt_kg": {"value": mt, "sigma": sigma_mt},
            "mh_kg": {"value": mh, "sigma": sigma_mh},
            "mal_kg": {"value": mal, "sigma": sigma_mal},
            "rp_fm": {"value": rp_fm, "sigma": sigma_rp_fm},
            "rd_fm": {"value": rd_fm, "sigma": sigma_rd_fm},
            "ral_fm": {"value": ral_fm, "sigma": sigma_ral_fm},
        },
        "charge_radii_a3_from_iaea": {
            "rt_fm": {"value": rt_fm, "sigma": sigma_rt_fm},
            "rh_fm": {"value": rh_fm, "sigma": sigma_rh_fm},
            "source_dirname": iaea_src_dirname,
        },
        "derived": derived,
        "falsification": {
            "acceptance_criteria": [
                "Any proposed P-model nuclear effective equation/potential must reproduce the fixed binding energies (A=2,3,4) within stated uncertainties when evaluated under its declared approximations.",
                "If a model claims a single universal nuclear-scale u-profile (once fixed by np scattering), it must explain the A-dependence trend (B and B/A) without per-nucleus tuning beyond declared higher-order corrections.",
            ],
            "scope_limits": [
                "This step fixes primary targets and does not solve the A>2 few-/many-body problem yet.",
                "Charge radii for A=3 are fixed from a dedicated compilation; extension to broader A (nuclear chart) remains a next step.",
            ],
        },
        "outputs": {"png": str(out_png), "csv": str(out_csv)},
    }

    out_json = out_dir / "nuclear_binding_light_nuclei_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ok] wrote:")
    print(f"  {out_png}")
    print(f"  {out_csv}")
    print(f"  {out_json}")


if __name__ == "__main__":
    main()
