from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(x: object) -> float | None:
    try:
        if x is None:
            return None
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return None


def _reduced_mass(m1: float, m2: float) -> float:
    return (m1 * m2) / (m1 + m2)


def _load_nist_h_isotope_masses_u(root: Path) -> tuple[dict[str, float], dict[str, Any]] | None:
    path = root / "data" / "quantum" / "sources" / "nist_isotopic_compositions_h" / "extracted_values.json"
    if not path.exists():
        return None
    j = _read_json(path)
    isotopes = j.get("isotopes")
    if not isinstance(isotopes, list):
        return None

    masses: dict[str, float] = {}
    for iso in isotopes:
        if not isinstance(iso, dict):
            continue
        sym = iso.get("symbol")
        a = iso.get("mass_number")
        m_u = iso.get("relative_atomic_mass_u")
        if not isinstance(sym, str) or not isinstance(a, int):
            continue
        m = _as_float(m_u)
        if m is None:
            continue
        masses[f"{sym}{a}"] = m

    meta = {
        "source": "NIST Atomic Weights and Isotopic Compositions (stand_alone.pl)",
        "path": str(path),
        "query_url": j.get("query_url"),
        "raw_sha256": j.get("raw_sha256"),
    }
    return masses, meta


def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Minimal isotopologues for Step 7.12.
    # Prefer primary-source-backed isotope masses (NIST). Fallback to mass-number approximation.
    mass_model: dict[str, Any] = {"kind": "mass_number_approx", "note": "Fallback: H=1, D=2 (dimensionless)."}
    m = _load_nist_h_isotope_masses_u(root)
    if m is not None:
        masses_u, meta = m
        if "H1" in masses_u and "D2" in masses_u:
            mass_model = {
                "kind": "nist_relative_atomic_mass_u",
                "note": "Uses relative atomic masses (u) as a primary-source-backed reduced-mass model.",
                "meta": meta,
                "masses_u": {"H1": masses_u["H1"], "D2": masses_u["D2"]},
            }

    def _get_mass(symbol_a: str) -> float:
        if mass_model["kind"] == "nist_relative_atomic_mass_u":
            return float(mass_model["masses_u"][symbol_a])
        # mass-number approximation
        return {"H1": 1.0, "D2": 2.0}[symbol_a]

    species = [
        {"slug": "h2", "label": "H2", "m1": _get_mass("H1"), "m2": _get_mass("H1"), "isotopes": ("H1", "H1")},
        {"slug": "hd", "label": "HD", "m1": _get_mass("H1"), "m2": _get_mass("D2"), "isotopes": ("H1", "D2")},
        {"slug": "d2", "label": "D2", "m1": _get_mass("D2"), "m2": _get_mass("D2"), "isotopes": ("D2", "D2")},
    ]

    rows: list[dict[str, Any]] = []
    for s in species:
        path = out_dir / f"molecular_{s['slug']}_baseline_metrics.json"
        if not path.exists():
            raise SystemExit(f"[fail] missing baseline metrics: {path}\nRun molecular_h2_baseline.py first.")
        j = _read_json(path)
        consts = j.get("constants")
        if not isinstance(consts, dict):
            raise SystemExit(f"[fail] constants missing in: {path}")
        omega_e = _as_float(consts.get("omega_e_cm^-1"))
        be = _as_float(consts.get("B_e_cm^-1"))
        if omega_e is None or be is None:
            raise SystemExit(f"[fail] missing ωe or Be in: {path}")

        mu = _reduced_mass(float(s["m1"]), float(s["m2"]))
        rows.append(
            {
                "slug": s["slug"],
                "label": s["label"],
                "m1": s["m1"],
                "m2": s["m2"],
                "isotopes": s.get("isotopes"),
                "mu": mu,
                "omega_e_cm^-1": omega_e,
                "B_e_cm^-1": be,
                "source_metrics": str(path),
            }
        )

    # Use H2 as reference for reduced-mass scaling.
    ref = next(r for r in rows if r["slug"] == "h2")
    mu_ref = float(ref["mu"])
    omega_ref = float(ref["omega_e_cm^-1"])
    be_ref = float(ref["B_e_cm^-1"])

    for r in rows:
        mu = float(r["mu"])
        omega_pred = omega_ref * math.sqrt(mu_ref / mu)
        be_pred = be_ref * (mu_ref / mu)
        r["omega_e_pred_cm^-1"] = omega_pred
        r["B_e_pred_cm^-1"] = be_pred
        r["omega_e_ratio_meas_over_pred"] = float(r["omega_e_cm^-1"]) / omega_pred
        r["B_e_ratio_meas_over_pred"] = float(r["B_e_cm^-1"]) / be_pred

    # ---- Figure ----
    labels = [r["label"] for r in rows]
    x = list(range(len(labels)))
    omega_ratios = [r["omega_e_ratio_meas_over_pred"] for r in rows]
    be_ratios = [r["B_e_ratio_meas_over_pred"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=180)
    fig.suptitle("Phase 7 / Step 7.12: Isotopic reduced-mass scaling (WebBook diatomic constants)", fontsize=13)

    ax = axes[0]
    ax.set_title("ωe scaling: ωe ∝ μ^{-1/2} (ratio vs prediction)", fontsize=11)
    ax.axhline(1.0, color="#666666", lw=1.2, alpha=0.8)
    ax.plot(x, omega_ratios, "o-", color="#2b6cb0", lw=2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("measured / predicted")
    ax.set_ylim(0.98, 1.02)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1]
    ax.set_title("Be scaling: Be ∝ μ^{-1} (ratio vs prediction)", fontsize=11)
    ax.axhline(1.0, color="#666666", lw=1.2, alpha=0.8)
    ax.plot(x, be_ratios, "o-", color="#c53030", lw=2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("measured / predicted")
    ax.set_ylim(0.98, 1.02)
    ax.grid(True, axis="y", alpha=0.25)

    fig.text(
        0.01,
        0.02,
        (
            ("NIST isotope masses used for μ (relative atomic masses in u). " if mass_model["kind"] != "mass_number_approx" else "")
            + ("Mass-number approximation used for μ (H=1, D=2). " if mass_model["kind"] == "mass_number_approx" else "")
            + "Reference: H2.\n"
            + "This is a consistency check for Step 7.12 baselines; not a P-model derivation."
        ),
        fontsize=9.3,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    out_png = out_dir / "molecular_isotopic_scaling.png"
    fig.savefig(out_png)
    plt.close(fig)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.12",
        "title": "Isotopic reduced-mass scaling check (WebBook diatomic constants)",
        "mass_model": mass_model,
        "note": (
            "Checks leading reduced-mass scaling (ωe∝μ^{-1/2}, B_e∝μ^{-1}). "
            "Prefer NIST primary-source-backed isotope masses when available; otherwise uses mass-number approximation."
        ),
        "rows": rows,
        "reference": {"slug": "h2", "mu_ref": mu_ref, "omega_e_ref_cm^-1": omega_ref, "B_e_ref_cm^-1": be_ref},
        "outputs": {"png": str(out_png)},
    }
    out_json = out_dir / "molecular_isotopic_scaling_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_json}")


if __name__ == "__main__":
    main()
