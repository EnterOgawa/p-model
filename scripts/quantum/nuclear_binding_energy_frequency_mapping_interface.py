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


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    light_metrics_path = root / "output" / "quantum" / "nuclear_binding_light_nuclei_metrics.json"
    if not light_metrics_path.exists():
        raise SystemExit(
            "[fail] missing light-nuclei binding baseline metrics.\n"
            "Run:\n"
            "  python -B scripts/quantum/nuclear_binding_light_nuclei.py\n"
            f"Expected: {light_metrics_path}"
        )
    light = _load_json(light_metrics_path)

    derived = light.get("derived")
    if not isinstance(derived, dict) or not derived:
        raise SystemExit(f"[fail] invalid light nuclei metrics: derived missing/empty: {light_metrics_path}")

    # Exact SI constants:
    c = 299_792_458.0
    e_charge = 1.602_176_634e-19
    h = 6.626_070_15e-34
    hbar = h / (2.0 * math.pi)

    codata = light.get("constants_from_nist_codata", {})
    mp = float(codata.get("mp_kg", {}).get("value"))
    mn = float(codata.get("mn_kg", {}).get("value"))
    if not (math.isfinite(mp) and mp > 0 and math.isfinite(mn) and mn > 0):
        raise SystemExit("[fail] missing mp/mn in nuclear_binding_light_nuclei_metrics.json constants_from_nist_codata")

    label_map = {
        "d": "deuteron",
        "t": "triton",
        "h": "helion",
        "alpha": "alpha",
    }

    rows: list[dict[str, object]] = []
    for key, item in derived.items():
        if not isinstance(item, dict):
            continue
        A = int(item.get("A", -1))
        Z = int(item.get("Z", -1))
        N = int(item.get("N", -1))
        if A < 1 or Z < 0 or N < 0:
            continue

        md = item.get("mass_defect_kg", {})
        md_val = float(md.get("value"))
        md_sigma = float(md.get("sigma"))

        be = item.get("binding_energy", {}).get("B_MeV", {})
        b_mev = float(be.get("value"))
        sigma_b_mev = float(be.get("sigma"))

        b_j = b_mev * 1e6 * e_charge
        sigma_b_j = sigma_b_mev * 1e6 * e_charge

        # Δω and Δm mappings (frozen I/F):
        delta_omega = b_j / hbar
        sigma_delta_omega = sigma_b_j / hbar

        delta_m_calc = b_j / (c**2)
        sigma_delta_m_calc = sigma_b_j / (c**2)

        # Sanity check: Δm should match mass defect by construction.
        dm_diff = delta_m_calc - md_val
        dm_diff_over_sigma = dm_diff / math.sqrt((md_sigma**2) + (sigma_delta_m_calc**2)) if md_sigma > 0 else float("nan")

        b_per_a = b_mev / float(A)
        sigma_b_per_a = sigma_b_mev / float(A)
        delta_omega_per_a = delta_omega / float(A)
        sigma_delta_omega_per_a = sigma_delta_omega / float(A)

        # Rest energy per nucleon (use Z,N mix; nuclear masses are close enough at this stage)
        m_eff = (float(Z) * mp + float(N) * mn) / float(A)
        e0_eff_mev = (m_eff * (c**2)) / (1e6 * e_charge)
        omega0_eff = (m_eff * (c**2)) / hbar

        frac_b_over_rest = b_per_a / e0_eff_mev if e0_eff_mev > 0 else float("nan")
        frac_domega_over_omega0 = (delta_omega_per_a / omega0_eff) if omega0_eff > 0 else float("nan")

        rows.append(
            {
                "key": str(key),
                "label": label_map.get(str(key), str(key)),
                "A": int(A),
                "Z": int(Z),
                "N": int(N),
                "B_MeV": b_mev,
                "sigma_B_MeV": sigma_b_mev,
                "B_over_A_MeV": b_per_a,
                "sigma_B_over_A_MeV": sigma_b_per_a,
                "Delta_omega_per_s": delta_omega,
                "sigma_Delta_omega_per_s": sigma_delta_omega,
                "Delta_omega_over_A_per_s": delta_omega_per_a,
                "sigma_Delta_omega_over_A_per_s": sigma_delta_omega_per_a,
                "Delta_m_kg_from_B_over_c2": delta_m_calc,
                "sigma_Delta_m_kg_from_B_over_c2": sigma_delta_m_calc,
                "Delta_m_kg_mass_defect_source": md_val,
                "sigma_Delta_m_kg_mass_defect_source": md_sigma,
                "Delta_m_diff_kg": dm_diff,
                "Delta_m_diff_over_sigma": dm_diff_over_sigma,
                "rest_energy_per_nucleon_MeV": e0_eff_mev,
                "omega0_eff_per_s": omega0_eff,
                "B_over_rest_per_nucleon": frac_b_over_rest,
                "Delta_omega_over_omega0_per_nucleon": frac_domega_over_omega0,
            }
        )

    rows.sort(key=lambda r: (int(r["A"]), int(r["Z"]), str(r["key"])))

    # CSV
    out_csv = out_dir / "nuclear_binding_energy_frequency_mapping_interface.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "key",
                "label",
                "A",
                "Z",
                "N",
                "B_MeV",
                "sigma_B_MeV",
                "B_over_A_MeV",
                "sigma_B_over_A_MeV",
                "Delta_omega_per_s",
                "sigma_Delta_omega_per_s",
                "Delta_omega_over_A_per_s",
                "sigma_Delta_omega_over_A_per_s",
                "Delta_m_kg_from_B_over_c2",
                "sigma_Delta_m_kg_from_B_over_c2",
                "Delta_m_kg_mass_defect_source",
                "sigma_Delta_m_kg_mass_defect_source",
                "Delta_m_diff_kg",
                "Delta_m_diff_over_sigma",
                "rest_energy_per_nucleon_MeV",
                "B_over_rest_per_nucleon",
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
                    f"{float(r['B_MeV']):.12g}",
                    f"{float(r['sigma_B_MeV']):.12g}",
                    f"{float(r['B_over_A_MeV']):.12g}",
                    f"{float(r['sigma_B_over_A_MeV']):.12g}",
                    f"{float(r['Delta_omega_per_s']):.12g}",
                    f"{float(r['sigma_Delta_omega_per_s']):.12g}",
                    f"{float(r['Delta_omega_over_A_per_s']):.12g}",
                    f"{float(r['sigma_Delta_omega_over_A_per_s']):.12g}",
                    f"{float(r['Delta_m_kg_from_B_over_c2']):.12g}",
                    f"{float(r['sigma_Delta_m_kg_from_B_over_c2']):.12g}",
                    f"{float(r['Delta_m_kg_mass_defect_source']):.12g}",
                    f"{float(r['sigma_Delta_m_kg_mass_defect_source']):.12g}",
                    f"{float(r['Delta_m_diff_kg']):.12g}",
                    f"{float(r['Delta_m_diff_over_sigma']):.12g}",
                    f"{float(r['rest_energy_per_nucleon_MeV']):.12g}",
                    f"{float(r['B_over_rest_per_nucleon']):.12g}",
                ]
            )

    # Plot
    import matplotlib.pyplot as plt

    labels = [f"{r['label']} (A={int(r['A'])})" for r in rows]
    bpa = [float(r["B_over_A_MeV"]) for r in rows]
    sigma_bpa = [float(r["sigma_B_over_A_MeV"]) for r in rows]
    frac_permille = [1e3 * float(r["B_over_rest_per_nucleon"]) for r in rows]

    x = list(range(len(rows)))
    fig = plt.figure(figsize=(12.8, 4.2), dpi=160)
    gs = fig.add_gridspec(1, 2, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(x, bpa, color="tab:blue", alpha=0.85)
    ax0.errorbar(x, bpa, yerr=sigma_bpa, fmt="none", ecolor="0.25", capsize=4, lw=1.2)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=20, ha="right")
    ax0.set_ylabel("B/A (MeV)")
    ax0.set_title("Binding energy per nucleon (fixed targets)")
    ax0.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(x, frac_permille, color="tab:orange", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("B/(A m_eff c²) (×10⁻³)")
    ax1.set_title("Frequency-defect fraction per nucleon (mapping)")
    ax1.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)

    fig.suptitle("Phase 7 / Step 7.13.17.3: Δω→Δm→B.E. mapping interface (light nuclei sanity check)", y=1.02)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.28, wspace=0.28)

    out_png = out_dir / "nuclear_binding_energy_frequency_mapping_interface.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # Metrics
    out_json = out_dir / "nuclear_binding_energy_frequency_mapping_interface_metrics.json"
    out_json.write_text(
        json.dumps(
            {
                "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.13.17.3",
                "inputs": {
                    "light_nuclei_metrics": {"path": str(light_metrics_path), "sha256": _sha256(light_metrics_path)},
                },
                "constants": {
                    "c_m_per_s": c,
                    "h_J_s": h,
                    "hbar_J_s": hbar,
                    "e_C": e_charge,
                },
                "mapping_frozen": {
                    "B_equals_hbar_Delta_omega": "B = ħ Δω",
                    "Delta_m_equals_B_over_c2": "Δm = B / c^2 = ħ Δω / c^2",
                    "m_equals_hbar_omega0_over_c2": "m = ħ ω0 / c^2",
                    "Delta_omega_obs_equals_B_obs_over_hbar": "Δω_obs = B_obs / ħ",
                },
                "rows": rows,
                "outputs": {"png": str(out_png), "csv": str(out_csv)},
                "notes": [
                    "This step is a sanity check that the frozen mapping between (B, Δω, Δm) is internally consistent with the primary light-nuclei baselines.",
                    "It does not yet choose a multi-body reduction (pair-wise sum vs collective); that is deferred to ROADMAP 7.13.17.5.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[ok] wrote:")
    print(f"  {out_png}")
    print(f"  {out_csv}")
    print(f"  {out_json}")


if __name__ == "__main__":
    main()

