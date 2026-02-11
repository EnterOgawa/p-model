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


def _require_float(obj: object, *, path: Path, key_path: str) -> float:
    try:
        v = float(obj)
    except Exception as e:
        raise SystemExit(f"[fail] invalid float at {key_path} in {path}: {e}") from e
    if not math.isfinite(v):
        raise SystemExit(f"[fail] non-finite float at {key_path} in {path}")
    return v


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    deut_path = root / "output" / "public" / "quantum" / "nuclear_binding_deuteron_metrics.json"
    if not deut_path.exists():
        raise SystemExit(
            "[fail] missing deuteron binding baseline metrics.\n"
            "Run:\n"
            "  python -B scripts/quantum/nuclear_binding_deuteron.py\n"
            f"Expected: {deut_path}"
        )
    deut = _load_json(deut_path)

    light_path = root / "output" / "public" / "quantum" / "nuclear_binding_light_nuclei_metrics.json"
    if not light_path.exists():
        raise SystemExit(
            "[fail] missing light-nuclei binding baseline metrics.\n"
            "Run:\n"
            "  python -B scripts/quantum/nuclear_binding_light_nuclei.py\n"
            f"Expected: {light_path}"
        )
    light = _load_json(light_path)

    two_body_path = root / "output" / "public" / "quantum" / "nuclear_binding_energy_frequency_mapping_deuteron_two_body_metrics.json"
    if not two_body_path.exists():
        raise SystemExit(
            "[fail] missing deuteron two-body mapping metrics.\n"
            "Run:\n"
            "  python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_deuteron_two_body.py\n"
            f"Expected: {two_body_path}"
        )
    two_body = _load_json(two_body_path)

    # Fixed observables / constants
    b_d = _require_float(deut.get("derived", {}).get("binding_energy", {}).get("B_MeV", {}).get("value"), path=deut_path, key_path="derived.binding_energy.B_MeV.value")
    inv_kappa_d = _require_float(deut.get("derived", {}).get("inv_kappa_fm"), path=deut_path, key_path="derived.inv_kappa_fm")
    if not (b_d > 0 and inv_kappa_d > 0):
        raise SystemExit("[fail] invalid deuteron B or inv_kappa (non-positive)")

    b_alpha_obs = _require_float(
        light.get("derived", {}).get("alpha", {}).get("binding_energy", {}).get("B_MeV", {}).get("value"),
        path=light_path,
        key_path="derived.alpha.binding_energy.B_MeV.value",
    )
    sigma_b_alpha_obs = _require_float(
        light.get("derived", {}).get("alpha", {}).get("binding_energy", {}).get("B_MeV", {}).get("sigma"),
        path=light_path,
        key_path="derived.alpha.binding_energy.B_MeV.sigma",
    )
    if not (b_alpha_obs > 0 and sigma_b_alpha_obs >= 0):
        raise SystemExit("[fail] invalid alpha B_obs or sigma (non-positive/negative)")

    r_alpha_rms = _require_float(
        light.get("constants_from_nist_codata", {}).get("ral_fm", {}).get("value"),
        path=light_path,
        key_path="constants_from_nist_codata.ral_fm.value",
    )
    sigma_r_alpha_rms = _require_float(
        light.get("constants_from_nist_codata", {}).get("ral_fm", {}).get("sigma"),
        path=light_path,
        key_path="constants_from_nist_codata.ral_fm.sigma",
    )
    if not (r_alpha_rms > 0 and sigma_r_alpha_rms >= 0):
        raise SystemExit("[fail] invalid alpha charge radius (non-positive/negative sigma)")

    l_range = _require_float(
        two_body.get("derived", {}).get("lambda_pi_pm_fm"),
        path=two_body_path,
        key_path="derived.lambda_pi_pm_fm",
    )
    if not (l_range > 0):
        raise SystemExit("[fail] invalid range scale L (non-positive)")

    # Geometry proxy: convert rms charge radius to an equivalent uniform-sphere radius.
    # For a uniform sphere: r_rms = sqrt(3/5) R  =>  R = sqrt(5/3) r_rms
    r_alpha_uniform = math.sqrt(5.0 / 3.0) * r_alpha_rms
    sigma_r_alpha_uniform = math.sqrt(5.0 / 3.0) * sigma_r_alpha_rms

    # Coupling-energy scaling (same I/F as Step 7.13.16.4):
    #   J_E(R) = J_E(R_ref) * exp((R_ref - R)/L)
    # Fix R_ref using deuteron tail scale (1/κ) because it is defined directly by the bound-state tail.
    r_ref = inv_kappa_d
    j_ref_mev = 0.5 * b_d  # deuteron: B = 2 J_E(R_ref)
    j_alpha_mev = j_ref_mev * math.exp((r_ref - r_alpha_uniform) / l_range)
    sigma_j_alpha_mev = abs(j_alpha_mev) * (sigma_r_alpha_uniform / l_range) if l_range > 0 else float("nan")

    # Multi-body reduction candidates (choose one baseline; keep others for sys envelope).
    a = 4
    z = 2
    n = 2
    methods = [
        {"method": "collective", "C": a - 1, "description": "collective in-phase coherence (C=A-1)"},
        {"method": "pn_only", "C": z * n, "description": "pairwise pn only (C=Z*N)"},
        {"method": "pairwise_all", "C": a * (a - 1) // 2, "description": "pairwise all pairs (C=A(A-1)/2)"},
    ]

    rows: list[dict[str, object]] = []
    for m in methods:
        c_factor = int(m["C"])
        b_pred = 2.0 * float(j_alpha_mev) * float(c_factor)
        sigma_b_pred = 2.0 * float(sigma_j_alpha_mev) * float(c_factor) if math.isfinite(sigma_j_alpha_mev) else float("nan")
        delta_b = b_pred - b_alpha_obs
        rel = delta_b / b_alpha_obs if b_alpha_obs > 0 else float("nan")
        z_from_r = delta_b / sigma_b_pred if (sigma_b_pred > 0 and math.isfinite(sigma_b_pred)) else float("nan")

        rows.append(
            {
                "method": str(m["method"]),
                "C": c_factor,
                "description": str(m["description"]),
                "B_pred_MeV": b_pred,
                "sigma_B_pred_from_r_alpha_MeV": sigma_b_pred,
                "Delta_B_pred_minus_obs_MeV": delta_b,
                "Delta_B_pred_minus_obs_percent": 100.0 * rel,
                "z_from_r_alpha_only": z_from_r,
            }
        )

    # Required coherence factor to match observation under this geometry/range I/F.
    c_required = b_alpha_obs / (2.0 * j_alpha_mev) if j_alpha_mev > 0 else float("nan")

    # Choose baseline method (minimal overcounting; closest to required C in this frozen I/F).
    baseline_method = "collective"
    baseline_row = next((r for r in rows if r["method"] == baseline_method), rows[0])

    # CSV (freeze)
    out_csv = out_dir / "nuclear_binding_energy_frequency_mapping_alpha_verification.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "C",
                "B_pred_MeV",
                "sigma_B_pred_from_r_alpha_MeV",
                "Delta_B_pred_minus_obs_MeV",
                "Delta_B_pred_minus_obs_percent",
                "B_obs_MeV",
                "sigma_B_obs_MeV",
                "R_ref_fm",
                "R_alpha_uniform_fm",
                "L_fm",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["method"],
                    int(r["C"]),
                    f"{float(r['B_pred_MeV']):.12g}",
                    f"{float(r['sigma_B_pred_from_r_alpha_MeV']):.12g}",
                    f"{float(r['Delta_B_pred_minus_obs_MeV']):.12g}",
                    f"{float(r['Delta_B_pred_minus_obs_percent']):.12g}",
                    f"{b_alpha_obs:.12g}",
                    f"{sigma_b_alpha_obs:.12g}",
                    f"{r_ref:.12g}",
                    f"{r_alpha_uniform:.12g}",
                    f"{l_range:.12g}",
                ]
            )

    # Plot
    import matplotlib.pyplot as plt

    x = list(range(len(rows)))
    labels = [f"{r['method']}\n(C={int(r['C'])})" for r in rows]
    b_preds = [float(r["B_pred_MeV"]) for r in rows]
    b_sig = [float(r["sigma_B_pred_from_r_alpha_MeV"]) for r in rows]
    d_b = [float(r["Delta_B_pred_minus_obs_MeV"]) for r in rows]

    fig = plt.figure(figsize=(10.8, 4.2), dpi=160)
    gs = fig.add_gridspec(1, 2, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.errorbar(x, b_preds, yerr=b_sig, fmt="o", color="tab:blue", capsize=4, lw=1.2)
    ax0.axhline(b_alpha_obs, color="0.15", lw=1.2, ls="-", label="B_obs (alpha; CODATA via NIST)")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylabel("B (MeV)")
    ax0.set_title("He-4 binding: multi-body reduction candidates")
    ax0.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax0.legend(loc="lower right", fontsize=9)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(x, d_b, color=["tab:orange" if r["method"] == baseline_method else "tab:blue" for r in rows], alpha=0.85)
    ax1.axhline(0.0, color="0.15", lw=1.2, ls="-")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("ΔB = B_pred − B_obs (MeV)")
    ax1.set_title("Residuals vs observation")
    ax1.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)

    ax1.text(
        0.02,
        0.96,
        f"C_required≈{c_required:.3f}\n"
        f"baseline={baseline_method}\n"
        f"R_ref=1/κ_d={r_ref:.2f} fm\n"
        f"R_alpha=√(5/3) r_rms={r_alpha_uniform:.2f} fm\n"
        f"L=λπ={l_range:.2f} fm",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    fig.suptitle("Phase 7 / Step 7.13.17.5: He-4 numeric extension (multi-body reduction I/F)", y=1.02)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.22, wspace=0.28)

    out_png = out_dir / "nuclear_binding_energy_frequency_mapping_alpha_verification.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # Metrics (machine-readable freeze)
    out_json = out_dir / "nuclear_binding_energy_frequency_mapping_alpha_verification_metrics.json"
    out_json.write_text(
        json.dumps(
            {
                "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.13.17.5",
                "inputs": {
                    "deuteron_binding_metrics": {"path": str(deut_path), "sha256": _sha256(deut_path)},
                    "light_nuclei_metrics": {"path": str(light_path), "sha256": _sha256(light_path)},
                    "deuteron_two_body_metrics": {"path": str(two_body_path), "sha256": _sha256(two_body_path)},
                },
                "frozen_if": {
                    "reference_scale": "R_ref = 1/κ_d (deuteron tail scale from CODATA B)",
                    "alpha_scale": "R_alpha = √(5/3) r_rms(alpha) (uniform-sphere equivalent radius)",
                    "range_scale": "L = λπ (PDG mass proxy; frozen in Step 7.13.17.2)",
                    "coupling_scaling": "J_E(R) = J_E(R_ref) exp((R_ref - R)/L), with J_E(R_ref)=B_d/2",
                    "multi_body_binding": "B_pred(alpha) = 2 C J_E(R_alpha), with C chosen by a reduction rule",
                },
                "observed": {
                    "deuteron": {"B_d_MeV": b_d, "R_ref_fm": r_ref},
                    "alpha": {
                        "B_obs_MeV": {"value": b_alpha_obs, "sigma": sigma_b_alpha_obs},
                        "r_rms_fm": {"value": r_alpha_rms, "sigma": sigma_r_alpha_rms},
                        "R_uniform_fm": {"value": r_alpha_uniform, "sigma": sigma_r_alpha_uniform},
                    },
                    "range": {"L_fm": l_range},
                },
                "derived": {
                    "J_ref_MeV": j_ref_mev,
                    "J_alpha_MeV": {"value": j_alpha_mev, "sigma_from_r_alpha_only": sigma_j_alpha_mev},
                    "C_required": c_required,
                },
                "methods": rows,
                "baseline": {
                    "method": baseline_method,
                    "row": baseline_row,
                    "notes": [
                        "Baseline chosen as the minimal non-overcounting reduction consistent with a coherent in-phase mode (C=A-1).",
                        "Other methods are retained as a systematics envelope/probe and are expected to overcount binding in this frozen I/F.",
                    ],
                },
                "outputs": {"png": str(out_png), "csv": str(out_csv)},
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

