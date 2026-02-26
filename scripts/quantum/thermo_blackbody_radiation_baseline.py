from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_sha256` の入出力契約と処理意図を定義する。

def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            # 条件分岐: `not b` を満たす経路を評価する。
            if not b:
                break

            h.update(b)

    return h.hexdigest()


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_as_float` の入出力契約と処理意図を定義する。

def _as_float(v: Any) -> float:
    # 条件分岐: `isinstance(v, (int, float))` を満たす経路を評価する。
    if isinstance(v, (int, float)):
        return float(v)

    raise TypeError(f"expected number, got: {type(v)}")


# 関数: `_get_constant` の入出力契約と処理意図を定義する。

def _get_constant(extracted: Dict[str, Any], code: str) -> Dict[str, Any]:
    c = extracted.get("constants")
    # 条件分岐: `not isinstance(c, dict) or code not in c or not isinstance(c.get(code), dict)` を満たす経路を評価する。
    if not isinstance(c, dict) or code not in c or not isinstance(c.get(code), dict):
        raise KeyError(f"missing constant: {code}")

    return c[code]


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = root / "data" / "quantum" / "sources" / "nist_codata_2022_blackbody_constants"
    extracted_path = src_dir / "extracted_values.json"
    # 条件分岐: `not extracted_path.exists()` を満たす経路を評価する。
    if not extracted_path.exists():
        raise SystemExit(
            f"[fail] missing: {extracted_path}\n"
            "Run: python -B scripts/quantum/fetch_blackbody_constants_sources.py"
        )

    extracted = _read_json(extracted_path)
    c0 = _get_constant(extracted, "c")
    h0 = _get_constant(extracted, "h")
    hb0 = _get_constant(extracted, "hbar")
    k0 = _get_constant(extracted, "k")
    sig0 = _get_constant(extracted, "sigma")

    c = _as_float(c0.get("value_si"))
    h = _as_float(h0.get("value_si"))
    hbar_ref = _as_float(hb0.get("value_si"))
    k = _as_float(k0.get("value_si"))
    sigma_ref = _as_float(sig0.get("value_si"))

    # Derived: Stefan-Boltzmann constant from (k_B, h, c).
    #
    # Use h-form to avoid relying on the truncated hbar decimal shown on NIST Cuu for SI-exact constants.
    # σ = 2π^5 k^4 / (15 h^3 c^2) = π^2 k^4 / (60 ħ^3 c^2)
    sigma_calc = (2.0 * (math.pi**5) * (k**4)) / (15.0 * (h**3) * (c**2))
    sigma_rel_err = (sigma_calc - sigma_ref) / sigma_ref if sigma_ref != 0 else float("nan")

    # Reduced Planck constant (derived; exact in SI but has an infinite decimal expansion).
    hbar = h / (2.0 * math.pi)

    # Radiation constant a = 4σ/c so that u=aT^4.
    a_rad = 4.0 * sigma_calc / c

    # Photon number density: n = (2 ζ(3) / π^2) (kT/ħc)^3
    zeta3 = 1.2020569031595943
    n_coeff = (2.0 * zeta3 / (math.pi**2)) * ((k / (hbar * c)) ** 3)

    # Mean photon energy: <E> = (π^4/(30 ζ(3))) kT
    mean_e_coeff = (math.pi**4) / (30.0 * zeta3)

    # Example temperatures (purely illustrative; not treated as data).
    temps_k: List[float] = [2.725, 77.0, 300.0, 5772.0]

    rows: List[Dict[str, float]] = []
    for t in temps_k:
        flux = sigma_calc * (t**4)
        u = a_rad * (t**4)
        n = n_coeff * (t**3)
        e_mean = mean_e_coeff * k * t
        e_mean_ev = e_mean / 1.602176634e-19
        rows.append(
            {
                "T_K": float(t),
                "flux_W_per_m2": float(flux),
                "energy_density_J_per_m3": float(u),
                "photon_number_density_per_m3": float(n),
                "mean_photon_energy_eV": float(e_mean_ev),
            }
        )

    out_csv = out_dir / "thermo_blackbody_radiation_baseline.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "T_K",
                "flux_W_per_m2",
                "energy_density_J_per_m3",
                "photon_number_density_per_m3",
                "mean_photon_energy_eV",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Plot: log-log scaling (u~T^4, n~T^3).

    xs = [10 ** (i / 20.0) for i in range(0, 81)]  # 1..1e4 K
    u_curve = [a_rad * (t**4) for t in xs]
    n_curve = [n_coeff * (t**3) for t in xs]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    ax.plot(xs, u_curve, color="#1f77b4")
    ax.scatter([r["T_K"] for r in rows], [r["energy_density_J_per_m3"] for r in rows], color="#000000", s=18)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Temperature T (K)")
    ax.set_ylabel("Energy density u (J/m^3)")
    ax.set_title("Blackbody radiation: u(T)=a T^4")
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1]
    ax.plot(xs, n_curve, color="#d62728")
    ax.scatter([r["T_K"] for r in rows], [r["photon_number_density_per_m3"] for r in rows], color="#000000", s=18)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Temperature T (K)")
    ax.set_ylabel("Photon density n (1/m^3)")
    ax.set_title("Blackbody radiation: n(T) ∝ T^3")
    ax.grid(True, which="both", alpha=0.25)

    fig.suptitle("Thermo baseline: blackbody scalings (SI constants; CODATA via NIST)", y=1.02)
    fig.tight_layout()
    out_png = out_dir / "thermo_blackbody_radiation_baseline.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    out_metrics = out_dir / "thermo_blackbody_radiation_baseline_metrics.json"
    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.15.1",
                "inputs": {"blackbody_constants_extracted_values": {"path": str(extracted_path), "sha256": _sha256(extracted_path)}},
                "constants_si": {
                    "c_m_per_s": c,
                    "h_J_s": h,
                    "hbar_J_s": hbar,
                    "hbar_ref_truncated_J_s": hbar_ref,
                    "kB_J_per_K": k,
                    "sigma_ref_W_per_m2K4": sigma_ref,
                },
                "derived": {
                    "sigma_calc_W_per_m2K4": sigma_calc,
                    "sigma_rel_error": sigma_rel_err,
                    "radiation_constant_a_J_per_m3K4": a_rad,
                    "zeta3": zeta3,
                    "photon_number_density_coeff_per_m3K3": n_coeff,
                    "mean_photon_energy_coeff_kT": mean_e_coeff,
                },
                "examples": rows,
                "outputs": {"csv": str(out_csv), "png": str(out_png)},
                "notes": [
                    "This is a baseline/consistency anchor for Step 7.15 (stat mech / thermo).",
                    "The formulas are standard; the purpose is to fix the numerical scales and provide a reproducible reference for later P-model extensions.",
                    "Example temperatures are illustrative only and are not treated as observational inputs.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_metrics}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
