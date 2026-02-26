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
    k0 = _get_constant(extracted, "k")

    c = _as_float(c0.get("value_si"))
    h = _as_float(h0.get("value_si"))
    k = _as_float(k0.get("value_si"))
    hbar = h / (2.0 * math.pi)

    # Stefan-Boltzmann constant from exact (h,c,k_B).
    sigma = (2.0 * (math.pi**5) * (k**4)) / (15.0 * (h**3) * (c**2))
    a_rad = 4.0 * sigma / c

    # Photon number density coefficient and constants.
    zeta3 = 1.2020569031595943
    n_coeff = (2.0 * zeta3 / (math.pi**2)) * ((k / (hbar * c)) ** 3)

    # Radiation thermodynamics.
    # u=aT^4, p=u/3, s=(4/3)aT^3.
    s_coeff = (4.0 / 3.0) * a_rad

    # Entropy per photon (dimensionless in units of k_B): s/(n k_B).
    s_over_n_kb = (2.0 * (math.pi**4)) / (45.0 * zeta3)

    temps_k: List[float] = [2.725, 77.0, 300.0, 5772.0]
    rows: List[Dict[str, float]] = []
    for t in temps_k:
        u = a_rad * (t**4)
        p = u / 3.0
        s = s_coeff * (t**3)
        n = n_coeff * (t**3)
        s_over_n = s / (n * k) if (n > 0) else float("nan")
        rows.append(
            {
                "T_K": float(t),
                "energy_density_J_per_m3": float(u),
                "pressure_Pa": float(p),
                "entropy_density_J_per_m3K": float(s),
                "entropy_per_photon_kB": float(s_over_n),
            }
        )

    out_csv = out_dir / "thermo_blackbody_entropy_baseline.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "T_K",
                "energy_density_J_per_m3",
                "pressure_Pa",
                "entropy_density_J_per_m3K",
                "entropy_per_photon_kB",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Plot: s(T) scaling and s/(n k_B).

    xs = [10 ** (i / 20.0) for i in range(0, 81)]  # 1..1e4 K
    s_curve = [s_coeff * (t**3) for t in xs]
    ratio_curve = [s_over_n_kb for _ in xs]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    ax.plot(xs, s_curve, color="#1f77b4")
    ax.scatter([r["T_K"] for r in rows], [r["entropy_density_J_per_m3K"] for r in rows], color="#000000", s=18)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Temperature T (K)")
    ax.set_ylabel("Entropy density s (J/(m^3·K))")
    ax.set_title("Blackbody radiation: s(T) = (4/3) a T^3")
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1]
    ax.plot(xs, ratio_curve, color="#d62728")
    ax.scatter([r["T_K"] for r in rows], [r["entropy_per_photon_kB"] for r in rows], color="#000000", s=18)
    ax.set_xscale("log")
    ax.set_xlabel("Temperature T (K)")
    ax.set_ylabel("Entropy per photon s/(n k_B)")
    ax.set_title("Photon gas: s/(n k_B) = const.")
    ax.grid(True, which="both", alpha=0.25)

    fig.suptitle("Thermo baseline: blackbody entropy and second-law consistency (SI constants)", y=1.02)
    fig.tight_layout()
    out_png = out_dir / "thermo_blackbody_entropy_baseline.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    out_metrics = out_dir / "thermo_blackbody_entropy_baseline_metrics.json"
    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.15.2",
                "inputs": {
                    "blackbody_constants_extracted_values": {"path": str(extracted_path), "sha256": _sha256(extracted_path)}
                },
                "derived": {
                    "sigma_W_per_m2K4": sigma,
                    "radiation_constant_a_J_per_m3K4": a_rad,
                    "entropy_density_coeff_J_per_m3K4": s_coeff,
                    "zeta3": zeta3,
                    "photon_number_density_coeff_per_m3K3": n_coeff,
                    "entropy_per_photon_kB_theory": s_over_n_kb,
                },
                "examples": rows,
                "falsification": {
                    "notes": [
                        "For a photon gas in equilibrium, the relations u=aT^4, p=u/3, s=(4/3)aT^3 are standard and tightly constrained.",
                        "Any P-model extension that changes these equilibrium scalings without an explicit, testable mechanism should be treated as falsified.",
                    ]
                },
                "outputs": {"csv": str(out_csv), "png": str(out_png)},
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

