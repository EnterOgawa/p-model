from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _kj_per_mol_to_ev(kj_per_mol: float) -> float:
    # 1 eV per molecule = 96.4853321233 kJ/mol
    return kj_per_mol / 96.4853321233


def _load_dhf_kj_per_mol(root: Path, slug: str) -> tuple[float, Optional[float], Path]:
    src = root / "data" / "quantum" / "sources" / f"nist_webbook_thermo_{slug}" / "extracted_values.json"
    # 条件分岐: `not src.exists()` を満たす経路を評価する。
    if not src.exists():
        raise SystemExit(
            f"[fail] missing thermochemistry cache: {src}\n"
            "Run fetch_nist_webbook_thermochemistry.py to populate the cache."
        )

    j = _read_json(src)
    sel = j.get("selected")
    # 条件分岐: `not isinstance(sel, dict)` を満たす経路を評価する。
    if not isinstance(sel, dict):
        raise SystemExit(f"[fail] selected missing in: {src}")

    val = sel.get("dhf_kj_per_mol")
    # 条件分岐: `not isinstance(val, (int, float))` を満たす経路を評価する。
    if not isinstance(val, (int, float)):
        raise SystemExit(f"[fail] dhf_kj_per_mol missing in selected: {src}")

    unc = sel.get("dhf_unc_kj_per_mol")
    unc_f = None if unc is None else float(unc)
    return float(val), unc_f, src


def _prop_unc_sum(terms: list[tuple[float, Optional[float]]]) -> Optional[float]:
    """
    Uncertainty propagation for linear combination Σ a_i x_i:
      σ = sqrt( Σ (a_i σ_i)^2 )
    where each term is (a_i, σ_i).
    Returns None if all σ_i are None.
    """
    ss = 0.0
    any_unc = False
    for a, sig in terms:
        # 条件分岐: `sig is None` を満たす経路を評価する。
        if sig is None:
            continue

        any_unc = True
        ss += (a * sig) ** 2

    return math.sqrt(ss) if any_unc else None


def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Primary-source-backed thermochemistry baselines (NIST WebBook).
    dhf_h, dhf_h_unc, src_h = _load_dhf_kj_per_mol(root, "h_atom")
    dhf_d, dhf_d_unc, src_d = _load_dhf_kj_per_mol(root, "d_atom")
    dhf_h2, dhf_h2_unc, src_h2 = _load_dhf_kj_per_mol(root, "h2")
    dhf_hd, dhf_hd_unc, src_hd = _load_dhf_kj_per_mol(root, "hd")
    dhf_d2, dhf_d2_unc, src_d2 = _load_dhf_kj_per_mol(root, "d2")

    # Dissociation enthalpy at 298 K (approx; from ΔfH°gas values).
    # H2 -> 2H
    diss_h2 = 2.0 * dhf_h - dhf_h2
    diss_h2_unc = _prop_unc_sum([(2.0, dhf_h_unc), (-1.0, dhf_h2_unc)])

    # HD -> H + D
    diss_hd = dhf_h + dhf_d - dhf_hd
    diss_hd_unc = _prop_unc_sum([(1.0, dhf_h_unc), (1.0, dhf_d_unc), (-1.0, dhf_hd_unc)])

    # D2 -> 2D
    diss_d2 = 2.0 * dhf_d - dhf_d2
    diss_d2_unc = _prop_unc_sum([(2.0, dhf_d_unc), (-1.0, dhf_d2_unc)])

    rows = [
        {
            "species": "H2",
            "reaction": "H2 → 2H",
            "dissociation_kj_per_mol_298K": diss_h2,
            "unc_kj_per_mol": diss_h2_unc,
            "dissociation_eV_298K": _kj_per_mol_to_ev(diss_h2),
            "unc_eV": (None if diss_h2_unc is None else _kj_per_mol_to_ev(diss_h2_unc)),
        },
        {
            "species": "HD",
            "reaction": "HD → H + D",
            "dissociation_kj_per_mol_298K": diss_hd,
            "unc_kj_per_mol": diss_hd_unc,
            "dissociation_eV_298K": _kj_per_mol_to_ev(diss_hd),
            "unc_eV": (None if diss_hd_unc is None else _kj_per_mol_to_ev(diss_hd_unc)),
        },
        {
            "species": "D2",
            "reaction": "D2 → 2D",
            "dissociation_kj_per_mol_298K": diss_d2,
            "unc_kj_per_mol": diss_d2_unc,
            "dissociation_eV_298K": _kj_per_mol_to_ev(diss_d2),
            "unc_eV": (None if diss_d2_unc is None else _kj_per_mol_to_ev(diss_d2_unc)),
        },
    ]

    # ---- Figure ----
    labels = [r["species"] for r in rows]
    y = [float(r["dissociation_eV_298K"]) for r in rows]
    yerr = [
        (0.0 if r["unc_eV"] is None else float(r["unc_eV"]))  # type: ignore[arg-type]
        for r in rows
    ]

    fig, ax = plt.subplots(1, 1, figsize=(11.5, 4.6), dpi=180)
    ax.set_title("Phase 7 / Step 7.12: Molecular dissociation enthalpy (298 K; NIST WebBook thermochemistry)", fontsize=13)
    ax.bar(labels, y, color=["#2b6cb0", "#805ad5", "#c53030"], alpha=0.92)
    ax.errorbar(labels, y, yerr=yerr, fmt="none", ecolor="#222222", elinewidth=1.2, capsize=4)
    ax.set_ylabel("Dissociation enthalpy at 298 K [eV per molecule]")
    ax.grid(True, axis="y", alpha=0.25)

    for i, r in enumerate(rows):
        ax.text(
            i,
            y[i] + 0.01,
            f"{r['dissociation_kj_per_mol_298K']:.2f} kJ/mol",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

    ax.text(
        0.01,
        0.02,
        "Computed from NIST WebBook ΔfH°(gas) values at 298 K (thermochemistry section).\n"
        "This is not spectroscopic D0; use as an independent baseline for binding-energy scale.",
        transform=ax.transAxes,
        fontsize=9.0,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )
    fig.tight_layout()

    out_png = out_dir / "molecular_dissociation_thermochemistry.png"
    fig.savefig(out_png)
    plt.close(fig)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.12",
        "title": "Molecular dissociation enthalpy (298 K; NIST WebBook thermochemistry)",
        "note": (
            "Dissociation enthalpy at 298 K computed from NIST WebBook gas-phase thermochemistry ΔfH° values. "
            "This is an independent baseline and should not be conflated with spectroscopic D0."
        ),
        "inputs": {
            "H_atom": {"dhf_kj_per_mol": dhf_h, "unc_kj_per_mol": dhf_h_unc, "source": str(src_h)},
            "D_atom": {"dhf_kj_per_mol": dhf_d, "unc_kj_per_mol": dhf_d_unc, "source": str(src_d)},
            "H2": {"dhf_kj_per_mol": dhf_h2, "unc_kj_per_mol": dhf_h2_unc, "source": str(src_h2)},
            "HD": {"dhf_kj_per_mol": dhf_hd, "unc_kj_per_mol": dhf_hd_unc, "source": str(src_hd)},
            "D2": {"dhf_kj_per_mol": dhf_d2, "unc_kj_per_mol": dhf_d2_unc, "source": str(src_d2)},
        },
        "rows": rows,
        "outputs": {"png": str(out_png)},
    }
    out_json = out_dir / "molecular_dissociation_thermochemistry_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()

