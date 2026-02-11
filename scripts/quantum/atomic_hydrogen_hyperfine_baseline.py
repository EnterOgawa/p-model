from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = root / "data" / "quantum" / "sources" / "nist_atspec_handbook"
    extracted_path = src_dir / "extracted_values.json"
    if not extracted_path.exists():
        raise SystemExit(
            f"[fail] missing extracted values: {extracted_path}\n"
            "Run: python -B scripts/quantum/fetch_nist_atspec_handbook.py"
        )

    extracted = _read_json(extracted_path)
    rec = extracted.get("hydrogen_hyperfine_21cm")
    if not isinstance(rec, dict):
        raise SystemExit(f"[fail] hydrogen_hyperfine_21cm missing in: {extracted_path}")

    def _get_float(key: str) -> float:
        v = rec.get(key)
        try:
            return float(v)
        except Exception:
            raise SystemExit(f"[fail] invalid {key} in {extracted_path}: {v!r}")

    token = str(rec.get("token") or "").strip()
    f_mhz = _get_float("f_mhz")
    sigma_mhz = _get_float("sigma_mhz")
    f_hz = _get_float("f_hz")
    sigma_hz = _get_float("sigma_hz")

    if f_hz <= 0 or sigma_hz <= 0:
        raise SystemExit(f"[fail] invalid frequency/uncertainty: f_hz={f_hz}, sigma_hz={sigma_hz}")

    c = 299_792_458.0
    h = 6.626_070_15e-34  # exact
    e_charge = 1.602_176_634e-19  # exact

    wavelength_m = c / f_hz
    wavelength_cm = wavelength_m * 100.0
    energy_eV = (h * f_hz) / e_charge
    frac = sigma_hz / f_hz

    # ---- Figure ----
    fig, ax = plt.subplots(1, 1, figsize=(10.6, 3.8), dpi=180)
    ax.set_title("Phase 7 / Step 7.12: H I hyperfine baseline (21 cm; NIST AtSpec)", fontsize=13)
    ax.set_xlabel("Vacuum wavelength λ [cm]")
    ax.set_yticks([])
    ax.set_ylim(0, 1)

    x = float(wavelength_cm)
    ax.set_xlim(max(0.0, x - 2.0), x + 2.0)
    ax.grid(True, axis="x", alpha=0.25)
    ax.axvline(x, color="#2b6cb0", lw=3.0)
    ax.text(
        x + 0.05,
        0.68,
        f"λ={x:.6f} cm\nf={f_mhz:.10f} MHz\nσ={sigma_hz:.3g} Hz (frac={frac:.2e})",
        fontsize=10.5,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )
    ax.text(
        0.01,
        0.05,
        "Data: NIST AtSpec handbook (AtSpec.PDF)\n"
        "This fixes the 21 cm hyperfine benchmark as a reproducible target for Part III; it is not a derivation.",
        transform=ax.transAxes,
        fontsize=9.5,
        ha="left",
        va="bottom",
    )

    fig.tight_layout()
    out_png = out_dir / "atomic_hydrogen_hyperfine_baseline.png"
    fig.savefig(out_png)
    plt.close(fig)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.12",
        "title": "Atomic baseline (Hydrogen hyperfine; NIST AtSpec)",
        "source_cache": {
            "extracted_values": str(extracted_path),
            "raw_pdf": str(Path(str(extracted.get("raw_file") or ""))),
        },
        "hyperfine": {
            "transition": "H I 1s ground-state hyperfine (21 cm)",
            "token": (token or None),
            "frequency_mhz": float(f_mhz),
            "sigma_mhz": float(sigma_mhz),
            "frequency_hz": float(f_hz),
            "sigma_hz": float(sigma_hz),
            "fractional_sigma": float(frac),
            "wavelength_m": float(wavelength_m),
            "wavelength_cm": float(wavelength_cm),
            "photon_energy_eV": float(energy_eV),
        },
        "note": (
            "This output fixes the hydrogen 21 cm hyperfine transition frequency as a baseline target. "
            "A P-model derivation is tracked in Roadmap Step 7.12+."
        ),
    }
    out_json = out_dir / "atomic_hydrogen_hyperfine_baseline_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_json}")


if __name__ == "__main__":
    main()
