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
            # 条件分岐: `not b` を満たす経路を評価する。
            if not b:
                break

            h.update(b)

    return h.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_float(obj: object, *, path: Path, key_path: str) -> float:
    try:
        v = float(obj)
    except Exception as e:
        raise SystemExit(f"[fail] invalid float at {key_path} in {path}: {e}") from e

    # 条件分岐: `not math.isfinite(v)` を満たす経路を評価する。

    if not math.isfinite(v):
        raise SystemExit(f"[fail] non-finite float at {key_path} in {path}")

    return v


def _load_ame2020_mass_table(*, root: Path, src_dirname: str) -> dict[tuple[int, int], dict[str, object]]:
    src_dir = root / "data" / "quantum" / "sources" / src_dirname
    extracted = src_dir / "extracted_values.json"
    # 条件分岐: `not extracted.exists()` を満たす経路を評価する。
    if not extracted.exists():
        raise SystemExit(
            "[fail] missing extracted AME2020 table.\n"
            "Run:\n"
            f"  python -B scripts/quantum/fetch_ame2020_mass_table_sources.py --out-dirname {src_dirname}\n"
            f"Expected: {extracted}"
        )

    payload = json.loads(extracted.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    # 条件分岐: `not isinstance(rows, list) or not rows` を満たす経路を評価する。
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"[fail] invalid extracted_values.json: rows missing/empty: {extracted}")

    out: dict[tuple[int, int], dict[str, object]] = {}
    for r in rows:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        try:
            z = int(r["Z"])
            a = int(r["A"])
        except Exception:
            continue

        out[(z, a)] = r

    # 条件分岐: `not out` を満たす経路を評価する。

    if not out:
        raise SystemExit(f"[fail] parsed 0 usable rows from: {extracted}")

    return out


def _load_iaea_charge_radii_csv(*, root: Path, src_dirname: str) -> dict[tuple[int, int], dict[str, float]]:
    import csv as csv_lib

    src_dir = root / "data" / "quantum" / "sources" / src_dirname
    csv_path = src_dir / "charge_radii.csv"
    # 条件分岐: `not csv_path.exists()` を満たす経路を評価する。
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
            # 条件分岐: `not rv or not ru` を満たす経路を評価する。
            if not rv or not ru:
                continue

            try:
                out[(z, a)] = {"r_rms_fm": float(rv), "sigma_r_rms_fm": float(ru)}
            except Exception:
                continue

    # 条件分岐: `not out` を満たす経路を評価する。

    if not out:
        raise SystemExit(f"[fail] parsed 0 charge radii rows from: {csv_path}")

    return out


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    deut_path = root / "output" / "public" / "quantum" / "nuclear_binding_deuteron_metrics.json"
    # 条件分岐: `not deut_path.exists()` を満たす経路を評価する。
    if not deut_path.exists():
        raise SystemExit(
            "[fail] missing deuteron binding baseline metrics.\n"
            "Run:\n"
            "  python -B scripts/quantum/nuclear_binding_deuteron.py\n"
            f"Expected: {deut_path}"
        )

    deut = _load_json(deut_path)

    two_body_path = root / "output" / "public" / "quantum" / "nuclear_binding_energy_frequency_mapping_deuteron_two_body_metrics.json"
    # 条件分岐: `not two_body_path.exists()` を満たす経路を評価する。
    if not two_body_path.exists():
        raise SystemExit(
            "[fail] missing deuteron two-body mapping metrics.\n"
            "Run:\n"
            "  python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_deuteron_two_body.py\n"
            f"Expected: {two_body_path}"
        )

    two_body = _load_json(two_body_path)

    ame_src_dirname = "iaea_amdc_ame2020_mass_1_mas20"
    ame_map = _load_ame2020_mass_table(root=root, src_dirname=ame_src_dirname)

    radii_src_dirname = "iaea_charge_radii"
    radii_map = _load_iaea_charge_radii_csv(root=root, src_dirname=radii_src_dirname)

    # Frozen I/F anchor (deuteron):
    b_d = _require_float(
        deut.get("derived", {}).get("binding_energy", {}).get("B_MeV", {}).get("value"),
        path=deut_path,
        key_path="derived.binding_energy.B_MeV.value",
    )
    r_ref = _require_float(deut.get("derived", {}).get("inv_kappa_fm"), path=deut_path, key_path="derived.inv_kappa_fm")
    # 条件分岐: `not (b_d > 0 and r_ref > 0)` を満たす経路を評価する。
    if not (b_d > 0 and r_ref > 0):
        raise SystemExit("[fail] invalid deuteron anchor (B_d or R_ref non-positive)")

    j_ref_mev = 0.5 * b_d

    l_range = _require_float(two_body.get("derived", {}).get("lambda_pi_pm_fm"), path=two_body_path, key_path="derived.lambda_pi_pm_fm")
    # 条件分岐: `not (l_range > 0)` を満たす経路を評価する。
    if not (l_range > 0):
        raise SystemExit("[fail] invalid range scale L (non-positive)")

    nuclei = [
        {"key": "d", "label": "d (H-2)", "Z": 1, "A": 2, "note": "anchor via tail scale R_ref=1/κ_d"},
        {"key": "alpha", "label": "alpha (He-4)", "Z": 2, "A": 4, "note": "closed-shell / strong binding"},
        {"key": "c12", "label": "C-12", "Z": 6, "A": 12, "note": "light nucleus (cluster-like)"},
        {"key": "o16", "label": "O-16", "Z": 8, "A": 16, "note": "doubly-magic benchmark"},
    ]

    methods = [
        {"method": "collective", "description": "C=A-1 (coherent in-phase mode)", "c_kind": "A_minus_1"},
        {"method": "pn_only", "description": "C=Z*N (pn pairs only)", "c_kind": "Z_times_N"},
        {"method": "pairwise_all", "description": "C=A(A-1)/2 (all pairs; likely overcount)", "c_kind": "A_choose_2"},
    ]

    rows: list[dict[str, object]] = []
    nucleus_summary: list[dict[str, object]] = []
    for nuc in nuclei:
        z = int(nuc["Z"])
        a = int(nuc["A"])
        n = a - z

        ame = ame_map.get((z, a))
        # 条件分岐: `not isinstance(ame, dict)` を満たす経路を評価する。
        if not isinstance(ame, dict):
            raise SystemExit(f"[fail] AME2020 row not found: Z={z} A={a}")

        bea_kev = float(ame["binding_keV_per_A"])
        bea_sigma_kev = float(ame.get("binding_sigma_keV_per_A", 0.0))
        b_obs = (bea_kev / 1000.0) * float(a)
        sigma_b_obs = (bea_sigma_kev / 1000.0) * float(a)

        rr = radii_map.get((z, a))
        # 条件分岐: `rr is None` を満たす経路を評価する。
        if rr is None:
            raise SystemExit(f"[fail] charge radius not found in IAEA CSV: Z={z} A={a}")

        r_rms = float(rr["r_rms_fm"])
        sigma_r_rms = float(rr["sigma_r_rms_fm"])

        # Geometry proxy (uniform-sphere equivalent radius):
        r_uniform = math.sqrt(5.0 / 3.0) * r_rms
        sigma_r_uniform = math.sqrt(5.0 / 3.0) * sigma_r_rms

        # For the deuteron, keep the anchor distance as tail scale (1/κ).
        if str(nuc["key"]) == "d":
            r_model = r_ref
            sigma_r_model = 0.0
            r_model_kind = "tail_scale"
        else:
            r_model = r_uniform
            sigma_r_model = sigma_r_uniform
            r_model_kind = "uniform_sphere_radius"

        j_mev = j_ref_mev * math.exp((r_ref - r_model) / l_range)
        sigma_j_mev = abs(j_mev) * (sigma_r_model / l_range) if (sigma_r_model > 0 and l_range > 0) else 0.0

        c_required = b_obs / (2.0 * j_mev) if (j_mev > 0) else float("nan")
        nucleus_summary.append(
            {
                "key": str(nuc["key"]),
                "label": str(nuc["label"]),
                "Z": z,
                "N": n,
                "A": a,
                "note": str(nuc.get("note", "")),
                "B_obs_MeV": b_obs,
                "sigma_B_obs_MeV": sigma_b_obs,
                "B_over_A_obs_MeV": b_obs / float(a),
                "r_rms_fm": r_rms,
                "sigma_r_rms_fm": sigma_r_rms,
                "R_uniform_fm": r_uniform,
                "sigma_R_uniform_fm": sigma_r_uniform,
                "R_model_fm": r_model,
                "sigma_R_model_fm": sigma_r_model,
                "R_model_kind": r_model_kind,
                "J_E_MeV": j_mev,
                "sigma_J_E_from_R_model_MeV": sigma_j_mev,
                "C_required": c_required,
            }
        )

        for m in methods:
            c_kind = str(m["c_kind"])
            # 条件分岐: `c_kind == "A_minus_1"` を満たす経路を評価する。
            if c_kind == "A_minus_1":
                c_factor = a - 1
            # 条件分岐: 前段条件が不成立で、`c_kind == "Z_times_N"` を追加評価する。
            elif c_kind == "Z_times_N":
                c_factor = z * n
            # 条件分岐: 前段条件が不成立で、`c_kind == "A_choose_2"` を追加評価する。
            elif c_kind == "A_choose_2":
                c_factor = a * (a - 1) // 2
            else:
                raise SystemExit(f"[fail] unknown c_kind: {c_kind}")

            b_pred = 2.0 * float(c_factor) * float(j_mev)
            sigma_b_pred = 2.0 * float(c_factor) * float(sigma_j_mev)
            ratio = b_pred / b_obs if b_obs > 0 else float("nan")
            delta = b_pred - b_obs
            z_from_r_only = delta / sigma_b_pred if sigma_b_pred > 0 else float("nan")

            rows.append(
                {
                    "key": str(nuc["key"]),
                    "label": str(nuc["label"]),
                    "Z": z,
                    "N": n,
                    "A": a,
                    "method": str(m["method"]),
                    "method_description": str(m["description"]),
                    "C": int(c_factor),
                    "B_obs_MeV": b_obs,
                    "sigma_B_obs_MeV": sigma_b_obs,
                    "r_rms_fm": r_rms,
                    "sigma_r_rms_fm": sigma_r_rms,
                    "R_model_fm": r_model,
                    "sigma_R_model_fm": sigma_r_model,
                    "R_model_kind": r_model_kind,
                    "L_fm": l_range,
                    "J_E_MeV": j_mev,
                    "sigma_J_E_from_R_model_MeV": sigma_j_mev,
                    "B_pred_MeV": b_pred,
                    "sigma_B_pred_from_R_model_MeV": sigma_b_pred,
                    "ratio_B_pred_over_obs": ratio,
                    "Delta_B_pred_minus_obs_MeV": delta,
                    "z_from_R_model_only": z_from_r_only,
                }
            )

    # CSV (freeze)

    out_csv = out_dir / "nuclear_binding_energy_frequency_mapping_representative_nuclei.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "key",
                "label",
                "Z",
                "N",
                "A",
                "method",
                "C",
                "B_obs_MeV",
                "sigma_B_obs_MeV",
                "r_rms_fm",
                "sigma_r_rms_fm",
                "R_model_fm",
                "sigma_R_model_fm",
                "R_model_kind",
                "L_fm",
                "J_E_MeV",
                "sigma_J_E_from_R_model_MeV",
                "B_pred_MeV",
                "sigma_B_pred_from_R_model_MeV",
                "ratio_B_pred_over_obs",
                "Delta_B_pred_minus_obs_MeV",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["key"],
                    r["label"],
                    int(r["Z"]),
                    int(r["N"]),
                    int(r["A"]),
                    r["method"],
                    int(r["C"]),
                    f"{float(r['B_obs_MeV']):.12g}",
                    f"{float(r['sigma_B_obs_MeV']):.12g}",
                    f"{float(r['r_rms_fm']):.12g}",
                    f"{float(r['sigma_r_rms_fm']):.12g}",
                    f"{float(r['R_model_fm']):.12g}",
                    f"{float(r['sigma_R_model_fm']):.12g}",
                    r["R_model_kind"],
                    f"{float(r['L_fm']):.12g}",
                    f"{float(r['J_E_MeV']):.12g}",
                    f"{float(r['sigma_J_E_from_R_model_MeV']):.12g}",
                    f"{float(r['B_pred_MeV']):.12g}",
                    f"{float(r['sigma_B_pred_from_R_model_MeV']):.12g}",
                    f"{float(r['ratio_B_pred_over_obs']):.12g}",
                    f"{float(r['Delta_B_pred_minus_obs_MeV']):.12g}",
                ]
            )

    # Plot

    import matplotlib.pyplot as plt

    # Baseline method for B/A comparison
    baseline_method = "collective"
    base_rows = {(r["key"], r["method"]): r for r in rows}

    keys = [str(n["key"]) for n in nuclei]
    labels = [str(n["label"]) for n in nuclei]
    a_vals = [int(n["A"]) for n in nuclei]

    b_over_a_obs = []
    b_over_a_pred = []
    for k, a in zip(keys, a_vals):
        r_obs = base_rows[(k, baseline_method)]
        b_over_a_obs.append(float(r_obs["B_obs_MeV"]) / float(a))
        b_over_a_pred.append(float(r_obs["B_pred_MeV"]) / float(a))

    fig = plt.figure(figsize=(12.0, 4.8), dpi=160)
    gs = fig.add_gridspec(1, 2, wspace=0.30)

    ax0 = fig.add_subplot(gs[0, 0])
    x = list(range(len(labels)))
    w = 0.38
    ax0.bar([v - w / 2 for v in x], b_over_a_obs, width=w, color="0.7", label="obs (AME2020)")
    ax0.bar([v + w / 2 for v in x], b_over_a_pred, width=w, color="tab:blue", alpha=0.85, label=f"pred ({baseline_method})")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylabel("B/A (MeV per nucleon)")
    ax0.set_title("Observed vs predicted (baseline) B/A")
    ax0.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax0.legend(loc="upper right", fontsize=9)

    # Ratio plot (all methods)
    ax1 = fig.add_subplot(gs[0, 1])
    method_names = [m["method"] for m in methods]
    colors = {"collective": "tab:blue", "pn_only": "tab:orange", "pairwise_all": "tab:green"}
    offsets = {"collective": -0.24, "pn_only": 0.0, "pairwise_all": 0.24}
    bar_w = 0.22
    for m in method_names:
        ratios = [float(base_rows[(k, m)]["ratio_B_pred_over_obs"]) for k in keys]
        ax1.bar([v + offsets[m] for v in x], ratios, width=bar_w, color=colors.get(m, "tab:blue"), alpha=0.85, label=m)

    ax1.axhline(1.0, color="0.2", lw=1.2, ls="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("B_pred / B_obs")
    ax1.set_title("Multi-body reduction sensitivity (C choices)")
    ax1.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax1.legend(loc="upper left", fontsize=9)

    ax1.text(
        0.02,
        0.02,
        f"anchor: R_ref=1/κ_d={r_ref:.2f} fm,  J_ref=B_d/2={j_ref_mev:.3f} MeV\n"
        f"range: L=λπ={l_range:.2f} fm,  R(model): d uses tail; others use √(5/3) r_rms",
        transform=ax1.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
    )

    fig.suptitle("Phase 7 / Step 7.13.17.6: representative nuclei (A=2,4,12,16) — A-dependence preview", y=1.02)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.18, wspace=0.30)

    out_png = out_dir / "nuclear_binding_energy_frequency_mapping_representative_nuclei.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # Metrics JSON
    out_json = out_dir / "nuclear_binding_energy_frequency_mapping_representative_nuclei_metrics.json"
    out_json.write_text(
        json.dumps(
            {
                "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.13.17.6",
                "inputs": {
                    "deuteron_binding_metrics": {"path": str(deut_path), "sha256": _sha256(deut_path)},
                    "deuteron_two_body_metrics": {"path": str(two_body_path), "sha256": _sha256(two_body_path)},
                    "ame2020_mass_table": {
                        "src_dirname": ame_src_dirname,
                        "path": str(root / "data" / "quantum" / "sources" / ame_src_dirname / "extracted_values.json"),
                        "sha256": _sha256(root / "data" / "quantum" / "sources" / ame_src_dirname / "extracted_values.json"),
                    },
                    "iaea_charge_radii_csv": {
                        "src_dirname": radii_src_dirname,
                        "path": str(root / "data" / "quantum" / "sources" / radii_src_dirname / "charge_radii.csv"),
                        "sha256": _sha256(root / "data" / "quantum" / "sources" / radii_src_dirname / "charge_radii.csv"),
                    },
                },
                "frozen_if": {
                    "anchor": "R_ref=1/κ_d, J_ref=B_d/2 (deuteron fixed)",
                    "range": "L=λπ (frozen)",
                    "geometry_proxy": "R_uniform = √(5/3) r_rms",
                    "scaling": "J_E(R)=J_ref exp((R_ref - R)/L); B_pred=2 C J_E(R)",
                    "C_methods": methods,
                },
                "anchor_values": {"B_d_MeV": b_d, "R_ref_fm": r_ref, "J_ref_MeV": j_ref_mev, "L_fm": l_range},
                "nuclei": nucleus_summary,
                "rows": rows,
                "outputs": {"png": str(out_png), "csv": str(out_csv)},
                "notes": [
                    "This step is a pre-diagnostic: it freezes a small representative dataset and compares the simplest A-scaling choices before committing to an AME2020 all-nuclei run.",
                    "The main output is the sensitivity to the multi-body reduction factor C and the implied C_required trend with A.",
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


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()

