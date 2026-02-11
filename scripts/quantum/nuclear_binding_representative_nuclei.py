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


def _load_ame2020_mass_table(*, root: Path, src_dirname: str) -> list[dict[str, object]]:
    src_dir = root / "data" / "quantum" / "sources" / src_dirname
    extracted = src_dir / "extracted_values.json"
    if not extracted.exists():
        raise SystemExit(
            "[fail] missing extracted AME2020 table.\n"
            "Run:\n"
            f"  python -B scripts/quantum/fetch_ame2020_mass_table_sources.py --out-dirname {src_dirname}\n"
            f"Expected: {extracted}"
        )
    payload = json.loads(extracted.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise SystemExit(f"[fail] invalid extracted_values.json: rows is not a list: {extracted}")
    out: list[dict[str, object]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if not all(k in r for k in ("Z", "A", "binding_keV_per_A")):
            continue
        out.append(r)
    if not out:
        raise SystemExit(f"[fail] parsed 0 usable rows from: {extracted}")
    return out


def _load_iaea_charge_radii_csv(*, root: Path, src_dirname: str) -> dict[tuple[int, int], dict[str, float]]:
    import csv as csv_lib

    src_dir = root / "data" / "quantum" / "sources" / src_dirname
    csv_path = src_dir / "charge_radii.csv"
    if not csv_path.exists():
        raise SystemExit(
            "[fail] missing cached IAEA charge_radii.csv.\n"
            "Run:\n"
            f"  python -B scripts/quantum/fetch_nuclear_charge_radii_sources.py --out-dirname {src_dirname}\n"
            f"Expected: {csv_path}"
        )

    out: dict[tuple[int, int], dict[str, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv_lib.DictReader(f)
        for row in reader:
            try:
                z = int(row["z"])
                a = int(row["a"])
            except Exception:
                continue
            rv = str(row.get("radius_val", "")).strip()
            ru = str(row.get("radius_unc", "")).strip()
            if not rv or not ru:
                continue
            try:
                out[(z, a)] = {"radius_fm": float(rv), "radius_sigma_fm": float(ru)}
            except Exception:
                continue
    if not out:
        raise SystemExit(f"[fail] parsed 0 charge radii rows from: {csv_path}")
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    ame_src_dirname = "iaea_amdc_ame2020_mass_1_mas20"
    ame_rows = _load_ame2020_mass_table(root=root, src_dirname=ame_src_dirname)
    ame_map: dict[tuple[int, int], dict[str, object]] = {}
    for r in ame_rows:
        try:
            z = int(r["Z"])
            a = int(r["A"])
        except Exception:
            continue
        ame_map[(z, a)] = r

    iaea_src_dirname = "iaea_charge_radii"
    radii_map = _load_iaea_charge_radii_csv(root=root, src_dirname=iaea_src_dirname)

    # Representative nuclei for A-dependence baselines (keep stable/benchmark nuclei).
    nuclei = [
        {"key": "d", "label": "d (H-2)", "Z": 1, "A": 2},
        {"key": "t", "label": "t (H-3)", "Z": 1, "A": 3},
        {"key": "h", "label": "h (He-3)", "Z": 2, "A": 3},
        {"key": "alpha", "label": "alpha (He-4)", "Z": 2, "A": 4},
        {"key": "li6", "label": "Li-6", "Z": 3, "A": 6},
        {"key": "c12", "label": "C-12", "Z": 6, "A": 12},
        {"key": "o16", "label": "O-16", "Z": 8, "A": 16},
        {"key": "ca40", "label": "Ca-40", "Z": 20, "A": 40},
        {"key": "fe56", "label": "Fe-56", "Z": 26, "A": 56},
        {"key": "ni62", "label": "Ni-62", "Z": 28, "A": 62},
        {"key": "pb208", "label": "Pb-208", "Z": 82, "A": 208},
    ]

    rows: list[dict[str, object]] = []
    for nuc in nuclei:
        z = int(nuc["Z"])
        a = int(nuc["A"])
        n = a - z
        ame = ame_map.get((z, a))
        if not isinstance(ame, dict):
            raise SystemExit(f"[fail] AME2020 row not found: Z={z} A={a}")
        bea_kev = ame.get("binding_keV_per_A")
        bea_sigma_kev = ame.get("binding_sigma_keV_per_A")
        if not isinstance(bea_kev, (int, float)) or not math.isfinite(float(bea_kev)):
            raise SystemExit(f"[fail] missing/invalid binding_keV_per_A for Z={z} A={a}")
        bea_mev = float(bea_kev) / 1000.0
        bea_sigma_mev = float(bea_sigma_kev) / 1000.0 if isinstance(bea_sigma_kev, (int, float)) else None
        b_mev = bea_mev * a
        b_sigma_mev = (bea_sigma_mev * a) if (bea_sigma_mev is not None) else None

        rr = radii_map.get((z, a))
        if rr is None:
            raise SystemExit(f"[fail] charge radius not found in IAEA CSV: Z={z} A={a}")

        rows.append(
            {
                "key": str(nuc["key"]),
                "label": str(nuc["label"]),
                "Z": z,
                "N": n,
                "A": a,
                "binding_energy_MeV": b_mev,
                "binding_energy_sigma_MeV": b_sigma_mev,
                "binding_energy_per_nucleon_MeV": bea_mev,
                "binding_energy_per_nucleon_sigma_MeV": bea_sigma_mev,
                "charge_radius_fm": float(rr["radius_fm"]),
                "charge_radius_sigma_fm": float(rr["radius_sigma_fm"]),
            }
        )

    # Plot
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"[fail] matplotlib is required for plotting: {e}") from e

    a_vals = [int(r["A"]) for r in rows]
    bea_vals = [float(r["binding_energy_per_nucleon_MeV"]) for r in rows]
    b_vals = [float(r["binding_energy_MeV"]) for r in rows]
    r_vals = [float(r["charge_radius_fm"]) for r in rows]
    labels = [str(r["label"]) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), constrained_layout=True)

    ax = axes[0][0]
    ax.scatter(a_vals, b_vals, s=40)
    ax.set_title("Total binding energy B (AME2020)")
    ax.set_xlabel("A")
    ax.set_ylabel("B (MeV)")
    ax.grid(True, alpha=0.3)

    ax = axes[0][1]
    ax.scatter(a_vals, bea_vals, s=40, color="tab:green")
    ax.set_title("Binding energy per nucleon (AME2020)")
    ax.set_xlabel("A")
    ax.set_ylabel("B/A (MeV per nucleon)")
    ax.grid(True, alpha=0.3)

    ax = axes[1][0]
    a13 = [a ** (1.0 / 3.0) for a in a_vals]
    ax.scatter(a13, r_vals, s=40, color="tab:purple")
    ax.set_title("Charge rms radius (IAEA radii compilation)")
    ax.set_xlabel("A^(1/3)")
    ax.set_ylabel("r_charge (fm)")
    ax.grid(True, alpha=0.3)

    ax = axes[1][1]
    ax.axis("off")
    txt = (
        "Definitions:\n"
        "  B/A: AME2020 binding energy per nucleon\n"
        "  r_charge: IAEA charge radii compilation (charge_radii.csv)\n\n"
        "Notes:\n"
        "  - This baseline fixes representative nuclei targets for A-dependence.\n"
        "  - It does not solve the A>2 few-/many-body problem yet.\n"
    )
    ax.text(0.0, 1.0, txt, va="top", family="monospace", fontsize=9)

    for i, (a, y, lab) in enumerate(zip(a_vals, bea_vals, labels, strict=True)):
        if i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
            axes[0][1].annotate(lab, (a, y), textcoords="offset points", xytext=(5, 4), fontsize=8)

    out_png = out_dir / "nuclear_binding_representative_nuclei.png"
    fig.suptitle("Phase 7 / Step 7.13.11: representative nuclei baselines (A-dependence)", fontsize=12)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    out_csv = out_dir / "nuclear_binding_representative_nuclei.csv"
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
                "charge_radius_fm",
                "charge_radius_sigma_fm",
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
                    "" if r["binding_energy_sigma_MeV"] is None else f"{float(r['binding_energy_sigma_MeV']):.12g}",
                    f"{float(r['binding_energy_per_nucleon_MeV']):.12g}",
                    ""
                    if r["binding_energy_per_nucleon_sigma_MeV"] is None
                    else f"{float(r['binding_energy_per_nucleon_sigma_MeV']):.12g}",
                    f"{float(r['charge_radius_fm']):.12g}",
                    f"{float(r['charge_radius_sigma_fm']):.12g}",
                ]
            )

    # Traceability
    ame_dir = root / "data" / "quantum" / "sources" / ame_src_dirname
    ame_manifest = ame_dir / "manifest.json"
    ame_extracted = ame_dir / "extracted_values.json"

    iaea_dir = root / "data" / "quantum" / "sources" / iaea_src_dirname
    iaea_manifest = iaea_dir / "manifest.json"
    iaea_csv = iaea_dir / "charge_radii.csv"

    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.13.11",
        "sources": [
            {
                "dataset": "AME2020 mass table (binding energy per nucleon)",
                "local_manifest": str(ame_manifest),
                "local_manifest_sha256": _sha256(ame_manifest) if ame_manifest.exists() else None,
                "local_extracted": str(ame_extracted),
                "local_extracted_sha256": _sha256(ame_extracted) if ame_extracted.exists() else None,
            },
            {
                "dataset": "IAEA nuclear charge radii compilation (charge_radii.csv)",
                "local_manifest": str(iaea_manifest),
                "local_manifest_sha256": _sha256(iaea_manifest) if iaea_manifest.exists() else None,
                "local_csv": str(iaea_csv),
                "local_csv_sha256": _sha256(iaea_csv) if iaea_csv.exists() else None,
            },
        ],
        "assumptions": [
            "Binding energies are taken directly from AME2020 'BINDING ENERGY/A' (keV per nucleon) and converted to MeV.",
            "Charge radii are taken from the IAEA radii compilation CSV (charge_radii.csv).",
            "Independent σ are used as given; covariance is not propagated.",
        ],
        "nuclei": rows,
        "falsification": {
            "acceptance_criteria": [
                "Any proposed P-model nuclear-scale effective equation (once fixed by declared inputs) must reproduce these baseline targets within stated uncertainties under its approximation class.",
                "If a model claims a universal nuclear-scale u-profile (shared ansatz class), it must explain the A-dependence trend without per-nucleus tuning beyond declared higher-order corrections.",
            ],
            "scope_limits": [
                "This step fixes representative A-dependence targets; it does not solve the many-body problem.",
                "Nucleon-nucleus scattering constraints are not yet included (planned for Step 7.13.12+).",
            ],
        },
        "outputs": {"png": str(out_png), "csv": str(out_csv)},
    }

    out_json = out_dir / "nuclear_binding_representative_nuclei_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ok] wrote:")
    print(f"  {out_png}")
    print(f"  {out_csv}")
    print(f"  {out_json}")


if __name__ == "__main__":
    main()
