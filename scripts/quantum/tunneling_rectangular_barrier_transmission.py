"""Rectangular-barrier tunneling benchmarks for Part III-B.

Purpose:
- Fix the minimal tunneling implementation promised in roadmap step 8.7.49.4.
- Reproduce transmission probabilities for a 1D static rectangular barrier
  under the same nonrelativistic envelope equation adopted in Part III-A.

Inputs:
- Fixed benchmark cases (particle mass, incident energy, barrier height, width).

Outputs:
- `output/public/quantum/tunneling_rectangular_barrier_benchmarks.csv`
- `output/public/quantum/tunneling_rectangular_barrier_benchmarks_metrics.json`

Assumptions:
- One-dimensional static barrier.
- Nonrelativistic Schrödinger envelope equation.
- Benchmark cases satisfy `E < V0`.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HBAR_J_S = 1.054571817e-34
EV_TO_J = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837015e-31


# クラス: `BarrierCase` の責務と境界条件を定義する。
@dataclass(frozen=True)
class BarrierCase:
    label: str
    particle: str
    mass_kg: float
    energy_eV: float
    barrier_eV: float
    width_nm: float


CASES: tuple[BarrierCase, ...] = (
    BarrierCase(
        label="electron_stm_like",
        particle="electron",
        mass_kg=ELECTRON_MASS_KG,
        energy_eV=1.0,
        barrier_eV=4.0,
        width_nm=0.50,
    ),
    BarrierCase(
        label="electron_thin_barrier",
        particle="electron",
        mass_kg=ELECTRON_MASS_KG,
        energy_eV=0.8,
        barrier_eV=2.0,
        width_nm=0.30,
    ),
)


# 関数: `_to_si` の入出力契約と処理意図を定義する。
def _to_si(*, energy_eV: float, barrier_eV: float, width_nm: float) -> tuple[float, float, float]:
    energy_J = float(energy_eV) * EV_TO_J
    barrier_J = float(barrier_eV) * EV_TO_J
    width_m = float(width_nm) * 1.0e-9
    return energy_J, barrier_J, width_m


# 関数: `wave_numbers` の入出力契約と処理意図を定義する。

def wave_numbers(*, mass_kg: float, energy_eV: float, barrier_eV: float) -> tuple[float, float]:
    """Return the propagating and evanescent wave numbers `k` and `κ` in SI."""
    energy_J, barrier_J, _ = _to_si(energy_eV=energy_eV, barrier_eV=barrier_eV, width_nm=1.0)
    gap_J = barrier_J - energy_J

    # 条件分岐: `not gap_J > 0.0` を満たす経路を評価する。
    if not gap_J > 0.0:
        raise ValueError("benchmark cases must satisfy E < V0")

    k = math.sqrt(2.0 * float(mass_kg) * energy_J) / HBAR_J_S
    kappa = math.sqrt(2.0 * float(mass_kg) * gap_J) / HBAR_J_S
    return float(k), float(kappa)


# 関数: `closed_form_transmission` の入出力契約と処理意図を定義する。

def closed_form_transmission(*, mass_kg: float, energy_eV: float, barrier_eV: float, width_nm: float) -> float:
    """Return the exact rectangular-barrier transmission coefficient for `E < V0`."""
    energy_J, barrier_J, width_m = _to_si(
        energy_eV=energy_eV,
        barrier_eV=barrier_eV,
        width_nm=width_nm,
    )
    _, kappa = wave_numbers(mass_kg=mass_kg, energy_eV=energy_eV, barrier_eV=barrier_eV)
    denom = 1.0 + (barrier_J * barrier_J * (math.sinh(kappa * width_m) ** 2)) / (
        4.0 * energy_J * (barrier_J - energy_J)
    )
    return float(1.0 / denom)


# 関数: `matrix_transmission` の入出力契約と処理意図を定義する。

def matrix_transmission(*, mass_kg: float, energy_eV: float, barrier_eV: float, width_nm: float) -> float:
    """Solve the matching conditions directly and return the transmission coefficient."""
    k, kappa = wave_numbers(mass_kg=mass_kg, energy_eV=energy_eV, barrier_eV=barrier_eV)
    _, _, width_m = _to_si(energy_eV=energy_eV, barrier_eV=barrier_eV, width_nm=width_nm)

    exp_plus = math.exp(kappa * width_m)
    exp_minus = math.exp(-kappa * width_m)
    exp_ik = complex(math.cos(k * width_m), math.sin(k * width_m))

    system = np.array(
        [
            [1.0 + 0.0j, -1.0 + 0.0j, -1.0 + 0.0j, 0.0 + 0.0j],
            [-1j * k, -kappa + 0.0j, kappa + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, exp_plus + 0.0j, exp_minus + 0.0j, -exp_ik],
            [0.0 + 0.0j, kappa * exp_plus + 0.0j, -kappa * exp_minus + 0.0j, -1j * k * exp_ik],
        ],
        dtype=complex,
    )
    rhs = np.array(
        [
            -1.0 + 0.0j,
            -1j * k,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        dtype=complex,
    )
    solution = np.linalg.solve(system, rhs)
    t_amp = solution[3]
    return float(abs(t_amp) ** 2)


# 関数: `wkb_transmission` の入出力契約と処理意図を定義する。

def wkb_transmission(*, mass_kg: float, energy_eV: float, barrier_eV: float, width_nm: float) -> float:
    """Return the leading WKB suppression factor `exp(-2 κ a)` as a diagnostic."""
    _, kappa = wave_numbers(mass_kg=mass_kg, energy_eV=energy_eV, barrier_eV=barrier_eV)
    _, _, width_m = _to_si(energy_eV=energy_eV, barrier_eV=barrier_eV, width_nm=width_nm)
    return float(math.exp(-2.0 * kappa * width_m))


# 関数: `build_case_row` の入出力契約と処理意図を定義する。

def build_case_row(case: BarrierCase) -> dict[str, float | str | bool]:
    """Assemble the benchmark metrics for one fixed barrier case."""
    k, kappa = wave_numbers(
        mass_kg=case.mass_kg,
        energy_eV=case.energy_eV,
        barrier_eV=case.barrier_eV,
    )
    _, _, width_m = _to_si(
        energy_eV=case.energy_eV,
        barrier_eV=case.barrier_eV,
        width_nm=case.width_nm,
    )
    exact = closed_form_transmission(
        mass_kg=case.mass_kg,
        energy_eV=case.energy_eV,
        barrier_eV=case.barrier_eV,
        width_nm=case.width_nm,
    )
    matrix = matrix_transmission(
        mass_kg=case.mass_kg,
        energy_eV=case.energy_eV,
        barrier_eV=case.barrier_eV,
        width_nm=case.width_nm,
    )
    wkb = wkb_transmission(
        mass_kg=case.mass_kg,
        energy_eV=case.energy_eV,
        barrier_eV=case.barrier_eV,
        width_nm=case.width_nm,
    )
    exact_vs_matrix_abs_diff = abs(exact - matrix)
    wkb_ratio = exact / wkb if wkb > 0.0 else float("inf")

    return {
        "label": case.label,
        "particle": case.particle,
        "energy_eV": float(case.energy_eV),
        "barrier_eV": float(case.barrier_eV),
        "width_nm": float(case.width_nm),
        "k_inv_m": float(k),
        "kappa_inv_m": float(kappa),
        "kappa_a": float(kappa * width_m),
        "transmission_exact": float(exact),
        "transmission_matrix": float(matrix),
        "transmission_wkb": float(wkb),
        "exact_vs_matrix_abs_diff": float(exact_vs_matrix_abs_diff),
        "exact_vs_matrix_pass": bool(exact_vs_matrix_abs_diff <= 1.0e-12),
        "exact_over_wkb": float(wkb_ratio),
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [build_case_row(case) for case in CASES]
    all_cases_pass = all(bool(row["exact_vs_matrix_pass"]) for row in rows)

    out_csv = out_dir / "tunneling_rectangular_barrier_benchmarks.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "label",
                "particle",
                "energy_eV",
                "barrier_eV",
                "width_nm",
                "k_inv_m",
                "kappa_inv_m",
                "kappa_a",
                "transmission_exact",
                "transmission_matrix",
                "transmission_wkb",
                "exact_vs_matrix_abs_diff",
                "exact_vs_matrix_pass",
                "exact_over_wkb",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if all_cases_pass else "mismatch",
        "model": {
            "equation": "(-(ħ^2)/(2m_*)) ψ'' + V(x) ψ = E ψ",
            "barrier": "V(x)=V0 for 0<x<a, else 0",
            "regime": "nonrelativistic, static barrier, E<V0",
            "pmodel_readout": "V(x)=m_* φ(x) under the Part III-A envelope mapping",
        },
        "checks": {
            "closed_form_vs_matrix_threshold": 1.0e-12,
            "all_cases_pass": bool(all_cases_pass),
            "wkb_role": "diagnostic_only",
        },
        "rows": rows,
        "notes": [
            "Exact transmission is fixed by the closed-form rectangular-barrier solution.",
            "A direct continuity-matrix solve is recorded to show that the implementation reproduces the same result numerically.",
            "Because Part III-A fixes the same Schrödinger envelope equation in this regime, tunneling is not a separate axiom in v1.1.",
        ],
    }

    out_json = out_dir / "tunneling_rectangular_barrier_benchmarks_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_json}")


if __name__ == "__main__":
    main()
