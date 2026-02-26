from __future__ import annotations

import argparse
import json
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
        # 条件分岐: `x is None` を満たす経路を評価する。
        if x is None:
            return None

        return float(x)  # type: ignore[arg-type]
    except Exception:
        return None


def _cm_inv_to_hz(cm_inv: float) -> float:
    c = 299_792_458.0
    return c * (cm_inv * 100.0)


def _cm_inv_to_ev(cm_inv: float) -> float:
    h = 6.626_070_15e-34  # exact (SI)
    e_charge = 1.602_176_634e-19  # exact (J/eV)
    return (h * _cm_inv_to_hz(cm_inv)) / e_charge


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build an offline-stable molecular baseline from NIST WebBook diatomic constants cache."
    )
    ap.add_argument(
        "--slug",
        default="h2",
        help='Cache slug (data/quantum/sources/nist_webbook_diatomic_<slug>/). Default: "h2".',
    )
    args = ap.parse_args()

    root = _repo_root()
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    slug = str(args.slug).strip().lower()
    src_dir = root / "data" / "quantum" / "sources" / f"nist_webbook_diatomic_{slug}"
    extracted_path = src_dir / "extracted_values.json"
    # 条件分岐: `not extracted_path.exists()` を満たす経路を評価する。
    if not extracted_path.exists():
        raise SystemExit(
            f"[fail] missing extracted values: {extracted_path}\n"
            "Run: python -B scripts/quantum/fetch_nist_webbook_diatomic_constants.py --id <ID> --slug <slug>"
        )

    extracted = _read_json(extracted_path)
    sel = extracted.get("selected")
    # 条件分岐: `not isinstance(sel, dict)` を満たす経路を評価する。
    if not isinstance(sel, dict):
        raise SystemExit(f"[fail] selected missing in: {extracted_path}")

    state = str(sel.get("state") or "").strip()
    te = _as_float(sel.get("Te_cm^-1"))
    omega_e = _as_float(sel.get("omega_e_cm^-1"))
    omega_exe = _as_float(sel.get("omega_e_x_e_cm^-1"))
    be = _as_float(sel.get("B_e_cm^-1"))
    alpha_e = _as_float(sel.get("alpha_e_cm^-1"))
    de = _as_float(sel.get("D_e_cm^-1"))
    re_a = _as_float(sel.get("r_e_A"))

    # 条件分岐: `omega_e is None or omega_exe is None or be is None or alpha_e is None or de i...` を満たす経路を評価する。
    if omega_e is None or omega_exe is None or be is None or alpha_e is None or de is None or re_a is None:
        raise SystemExit("[fail] missing required constants in selected block (expected ωe, ωexe, Be, αe, De, re)")

    # Morse potential (phenomenological) derived estimates.

    d_e_morse_cm_inv = None
    d0_morse_cm_inv = None
    # 条件分岐: `omega_exe != 0.0` を満たす経路を評価する。
    if omega_exe != 0.0:
        d_e_morse_cm_inv = (omega_e * omega_e) / (4.0 * omega_exe)
        # Zero-point energy (Morse; ignoring higher terms):
        # E0 ≈ ωe/2 − ωexe/4  => D0 ≈ De − E0
        d0_morse_cm_inv = d_e_morse_cm_inv - (omega_e / 2.0) + (omega_exe / 4.0)

    derived = {
        "omega_e": {
            "frequency_THz": _cm_inv_to_hz(omega_e) / 1e12,
            "energy_eV": _cm_inv_to_ev(omega_e),
        },
        "omega_e_x_e": {
            "frequency_THz": _cm_inv_to_hz(omega_exe) / 1e12,
            "energy_eV": _cm_inv_to_ev(omega_exe),
        },
        "B_e": {
            "frequency_THz": _cm_inv_to_hz(be) / 1e12,
            "energy_eV": _cm_inv_to_ev(be),
        },
        "alpha_e": {
            "frequency_THz": _cm_inv_to_hz(alpha_e) / 1e12,
            "energy_eV": _cm_inv_to_ev(alpha_e),
        },
        "D_e": {
            "frequency_THz": _cm_inv_to_hz(de) / 1e12,
            "energy_eV": _cm_inv_to_ev(de),
        },
        "r_e": {
            "meters": re_a * 1e-10,
        },
        "morse": {
            "D_e_cm^-1": d_e_morse_cm_inv,
            "D0_cm^-1": d0_morse_cm_inv,
            "D_e_eV": (None if d_e_morse_cm_inv is None else _cm_inv_to_ev(d_e_morse_cm_inv)),
            "D0_eV": (None if d0_morse_cm_inv is None else _cm_inv_to_ev(d0_morse_cm_inv)),
        },
    }

    # ---- Figure ----
    fig_w, fig_h, dpi = 11.5, 4.6, 180
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_axis_off()

    species_label = slug.upper()
    # 条件分岐: `slug == "h2"` を満たす経路を評価する。
    if slug == "h2":
        species_label = "H2"
    # 条件分岐: 前段条件が不成立で、`slug == "d2"` を追加評価する。
    elif slug == "d2":
        species_label = "D2"

    title = f"Phase 7 / Step 7.12: Molecular baseline ({species_label}, NIST WebBook diatomic constants)"
    ax.text(0.01, 0.95, title, transform=ax.transAxes, fontsize=13, va="top")

    lines = [
        ("State (ground)", state or "(unknown)"),
        ("T_e", f"{te:.6g} cm⁻¹" if te is not None else "(missing)"),
        ("ω_e", f"{omega_e:.6f} cm⁻¹  (≈{derived['omega_e']['energy_eV']:.6f} eV)"),
        ("ω_e x_e", f"{omega_exe:.6f} cm⁻¹  (≈{derived['omega_e_x_e']['energy_eV']:.6f} eV)"),
        ("B_e", f"{be:.7f} cm⁻¹  (≈{derived['B_e']['energy_eV']:.8f} eV)"),
        ("α_e", f"{alpha_e:.7f} cm⁻¹  (≈{derived['alpha_e']['energy_eV']:.8f} eV)"),
        ("D_e (distortion)", f"{de:.7f} cm⁻¹  (≈{derived['D_e']['energy_eV']:.10f} eV)"),
        ("r_e", f"{re_a:.6f} Å  (={derived['r_e']['meters']:.6e} m)"),
    ]
    morse_d0 = derived["morse"]["D0_eV"]
    morse_de = derived["morse"]["D_e_eV"]
    # 条件分岐: `morse_de is not None and morse_d0 is not None` を満たす経路を評価する。
    if morse_de is not None and morse_d0 is not None:
        lines.append(("Morse (derived)", f"D_e≈{morse_de:.4f} eV, D0≈{morse_d0:.4f} eV (from ωe, ωexe)"))

    x0 = 0.03
    y = 0.84
    dy = 0.09
    for k, v in lines:
        ax.text(x0, y, k, transform=ax.transAxes, fontsize=11.0, ha="left", va="center")
        ax.text(x0 + 0.26, y, v, transform=ax.transAxes, fontsize=11.0, ha="left", va="center")
        y -= dy

    ax.text(
        0.01,
        0.05,
        "Data: NIST Chemistry WebBook (Huber & Herzberg compilation)\n"
        "This figure fixes baseline targets for Part III; it is not a P-model derivation.",
        transform=ax.transAxes,
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )

    fig.tight_layout()
    out_png = out_dir / f"molecular_{slug}_baseline.png"
    fig.savefig(out_png)
    plt.close(fig)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.12",
        "title": f"Molecular baseline ({species_label}; NIST WebBook diatomic constants)",
        "source_cache": {"extracted_values": str(extracted_path)},
        "species": species_label,
        "webbook_id": str(extracted.get("webbook_id") or ""),
        "state": state,
        "constants": {
            "Te_cm^-1": te,
            "omega_e_cm^-1": omega_e,
            "omega_e_x_e_cm^-1": omega_exe,
            "B_e_cm^-1": be,
            "alpha_e_cm^-1": alpha_e,
            "D_e_cm^-1": de,
            "r_e_A": re_a,
        },
        "derived": derived,
        "note": (
            "This output fixes a small set of diatomic constants as a reproducible baseline. "
            "The Morse D0/De values are phenomenological estimates derived from ωe and ωexe, not independent data. "
            "P-model derivation of molecular binding is tracked in Roadmap Step 7.12+."
        ),
    }
    out_json = out_dir / f"molecular_{slug}_baseline_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
