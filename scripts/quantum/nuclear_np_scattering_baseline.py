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


def _get_value(d: dict, key: str) -> float:
    obj = d.get(key)
    if isinstance(obj, dict) and "value" in obj:
        return float(obj["value"])
    raise KeyError(key)


def _get_sigma(d: dict, key: str) -> float | None:
    obj = d.get(key)
    if not isinstance(obj, dict):
        return None
    sigma = obj.get("sigma")
    return float(sigma) if isinstance(sigma, (int, float)) else None


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load np scattering parameter sets
    src_dir = root / "data" / "quantum" / "sources"
    extracted_path = src_dir / "np_scattering_low_energy_arxiv_0704_1024v1_extracted.json"
    if not extracted_path.exists():
        raise SystemExit(
            "[fail] missing extracted np scattering values.\n"
            "Run:\n"
            "  python -B scripts/quantum/fetch_nuclear_np_scattering_sources.py\n"
            f"Expected: {extracted_path}"
        )
    extracted = _load_json(extracted_path)
    sets = extracted.get("parameter_sets", [])
    if not isinstance(sets, list) or not sets:
        raise SystemExit(f"[fail] invalid extracted file: parameter_sets missing/empty: {extracted_path}")

    # Load CODATA masses for deuteron binding scale (Step 7.9.1 baseline).
    codata_dir = root / "data" / "quantum" / "sources" / "nist_codata_2022_nuclear_baseline"
    codata_path = codata_dir / "extracted_values.json"
    if not codata_path.exists():
        raise SystemExit(
            "[fail] missing CODATA baseline. Run:\n"
            "  python -B scripts/quantum/fetch_nuclear_binding_sources.py\n"
            f"Expected: {codata_path}"
        )
    codata = _load_json(codata_path).get("constants", {})
    mp = float(codata["mp"]["value_si"])
    mn = float(codata["mn"]["value_si"])
    md = float(codata["md"]["value_si"])

    # Exact SI constants:
    c = 299_792_458.0
    e_charge = 1.602_176_634e-19
    h = 6.626_070_15e-34
    hbar = h / (2.0 * math.pi)

    # Deuteron binding energy from mass defect:
    dm = (mp + mn - md)
    b_j = dm * (c**2)
    b_mev = b_j / (1e6 * e_charge)

    # Reduced mass μ and deuteron wave number α (E = ħ^2 α^2 / (2μ))
    mu = (mp * mn) / (mp + mn)
    alpha = math.sqrt(2.0 * mu * abs(b_j)) / hbar if b_j > 0 else float("nan")
    r_deuteron_m = (1.0 / alpha) if (alpha and math.isfinite(alpha) and alpha > 0) else float("nan")
    r_deuteron_fm = r_deuteron_m * 1e15

    def r_t_from_binding_and_a_t(*, a_t_fm: float) -> float:
        # Eq.(10–11) style relation in the paper:
        #   r_t = 2 R (1 - R / a_t), with R = 1/alpha.
        if not (math.isfinite(r_deuteron_fm) and r_deuteron_fm > 0 and math.isfinite(a_t_fm) and a_t_fm != 0):
            return float("nan")
        r = float(r_deuteron_fm)
        return float(2.0 * r * (1.0 - (r / float(a_t_fm))))

    # Flatten sets
    rows: list[dict[str, object]] = []
    for s in sets:
        if not isinstance(s, dict):
            continue
        params = s.get("params")
        if not isinstance(params, dict):
            continue
        row = {
            "eq_label": int(s.get("eq_label", -1)),
            "kind": str(s.get("kind", "")),
            "a_t_fm": _get_value(params, "a_t_fm"),
            "r_t_fm": _get_value(params, "r_t_fm"),
            "a_s_fm": _get_value(params, "a_s_fm"),
            "r_s_fm": _get_value(params, "r_s_fm"),
            "v2t_fm3": _get_value(params, "v2t_fm3") if isinstance(params.get("v2t_fm3"), dict) else None,
            "v2s_fm3": _get_value(params, "v2s_fm3") if isinstance(params.get("v2s_fm3"), dict) else None,
            "a_t_sigma_fm": _get_sigma(params, "a_t_fm"),
            "r_t_sigma_fm": _get_sigma(params, "r_t_fm"),
            "a_s_sigma_fm": _get_sigma(params, "a_s_fm"),
            "r_s_sigma_fm": _get_sigma(params, "r_s_fm"),
        }
        row["r_t_from_B_and_a_t_fm"] = r_t_from_binding_and_a_t(a_t_fm=float(row["a_t_fm"]))
        row["delta_r_t_fm"] = float(row["r_t_fm"]) - float(row["r_t_from_B_and_a_t_fm"])
        rows.append(row)

    # Systematics proxy: GWU vs Nijmegen deltas (eq18 vs eq19).
    gwu = next((r for r in rows if r.get("eq_label") == 18), None)
    nij = next((r for r in rows if r.get("eq_label") == 19), None)
    sys_delta = None
    if gwu and nij:
        sys_delta = {
            "a_t_fm": float(gwu["a_t_fm"]) - float(nij["a_t_fm"]),
            "r_t_fm": float(gwu["r_t_fm"]) - float(nij["r_t_fm"]),
            "a_s_fm": float(gwu["a_s_fm"]) - float(nij["a_s_fm"]),
            "r_s_fm": float(gwu["r_s_fm"]) - float(nij["r_s_fm"]),
        }

    # Plot
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.6, 4.2), dpi=160)
    gs = fig.add_gridspec(1, 2, wspace=0.28)

    # Panel A: triplet channel (a_t vs r_t) with mixed-radius curve from B
    ax0 = fig.add_subplot(gs[0, 0])
    a_min = min(r["a_t_fm"] for r in rows) - 0.01
    a_max = max(r["a_t_fm"] for r in rows) + 0.01
    a_grid = [a_min + (a_max - a_min) * i / 200.0 for i in range(201)]
    r_curve = [r_t_from_binding_and_a_t(a_t_fm=a) for a in a_grid]
    ax0.plot(a_grid, r_curve, lw=2.0, color="0.35", label="r_t = 2R(1−R/a_t), R=1/α from deuteron B")
    for r in rows:
        label = f"eq{r['eq_label']}"
        x = float(r["a_t_fm"])
        y = float(r["r_t_fm"])
        xerr = r.get("a_t_sigma_fm")
        yerr = r.get("r_t_sigma_fm")
        if isinstance(xerr, (int, float)) and xerr:
            ax0.errorbar([x], [y], xerr=[xerr], yerr=[yerr] if yerr else None, fmt="o", capsize=4, label=label)
        else:
            ax0.plot([x], [y], marker="o", lw=0, label=label)
    ax0.set_xlabel("triplet scattering length a_t (fm)")
    ax0.set_ylabel("triplet effective range r_t (fm)")
    ax0.set_title("np triplet low-energy parameters")
    ax0.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax0.legend(frameon=True, fontsize=8, loc="upper right")
    ax0.text(
        0.02,
        0.02,
        f"deuteron B≈{b_mev:.6f} MeV  →  R≈{r_deuteron_fm:.3f} fm",
        transform=ax0.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
    )

    # Panel B: singlet channel summary (a_s vs r_s)
    ax1 = fig.add_subplot(gs[0, 1])
    for r in rows:
        label = f"eq{r['eq_label']}"
        x = float(r["a_s_fm"])
        y = float(r["r_s_fm"])
        xerr = r.get("a_s_sigma_fm")
        yerr = r.get("r_s_sigma_fm")
        if isinstance(xerr, (int, float)) and xerr:
            ax1.errorbar([x], [y], xerr=[xerr], yerr=[yerr] if yerr else None, fmt="o", capsize=4, label=label)
        else:
            ax1.plot([x], [y], marker="o", lw=0, label=label)
    ax1.set_xlabel("singlet scattering length a_s (fm)")
    ax1.set_ylabel("singlet effective range r_s (fm)")
    ax1.set_title("np singlet low-energy parameters")
    ax1.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax1.legend(frameon=True, fontsize=8, loc="upper right")

    fig.suptitle("Phase 7 / Step 7.9.2: np scattering baseline (observables fixed)", y=1.03)
    fig.tight_layout()

    out_png = out_dir / "nuclear_np_scattering_baseline.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    manifest = src_dir / "np_scattering_low_energy_arxiv_0704_1024v1_manifest.json"
    metrics = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.9.2",
        "sources": [
            {
                "dataset": "np scattering low-energy parameters (arXiv:0704.1024v1)",
                "local_manifest": str(manifest),
                "local_manifest_sha256": _sha256(manifest) if manifest.exists() else None,
                "local_extracted": str(extracted_path),
                "local_extracted_sha256": _sha256(extracted_path),
            },
            {
                "dataset": "CODATA masses for deuteron B baseline (NIST Cuu)",
                "local_extracted": str(codata_path),
                "local_extracted_sha256": _sha256(codata_path),
            },
        ],
        "units": {"length": "fm", "energy": "MeV"},
        "deuteron_binding_from_codata": {
            "B_MeV": float(b_mev),
            "R_fm": float(r_deuteron_fm),
            "notes": [
                "B computed from CODATA mp,mn,md (mass defect).",
                "R=1/alpha with alpha from E=ħ^2 alpha^2/(2μ).",
            ],
        },
        "np_low_energy_parameter_sets": rows,
        "systematics_proxy": {
            "gwu_minus_nijmegen": sys_delta,
            "notes": [
                "Eq.(18) and Eq.(19) are computed from different phase-shift analyses in the same primary source.",
                "Their difference is stored as a proxy for analysis-dependent systematics (not a statistical uncertainty).",
            ],
        },
        "falsification": {
            "acceptance_criteria": [
                "Any proposed P-model nuclear effective equation/potential must reproduce (a_t,r_t,a_s,r_s) within the envelope spanned by Eq.(18) and Eq.(19), or explicitly justify why a specific phase-shift analysis is preferred.",
                "Triplet channel must also be consistent with deuteron binding scale: r_t should not deviate wildly from r_t=2R(1−R/a_t) with R set by deuteron B (stored here as a cross-check baseline).",
            ]
        },
        "outputs": {"png": str(out_png)},
        "notes": [
            "This step fixes low-energy np scattering observables as constraints. It does not yet derive nuclear forces from P-model (that is Step 7.9.3).",
        ],
    }
    out_json = out_dir / "nuclear_np_scattering_baseline_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")


if __name__ == "__main__":
    main()
