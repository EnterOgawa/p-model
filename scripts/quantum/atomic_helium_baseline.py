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


def _as_float(x: object) -> float | None:
    try:
        if x is None:
            return None
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return None


def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Source: NIST ASD cached output (Phase 7 / Step 7.12)
    src_dir = root / "data" / "quantum" / "sources" / "nist_asd_he_i_lines"
    extracted_path = src_dir / "extracted_values.json"
    if not extracted_path.exists():
        raise SystemExit(
            f"[fail] missing extracted values: {extracted_path}\n"
            "Run: python -B scripts/quantum/fetch_nist_asd_lines.py --spectra \"He I\""
        )

    extracted = _read_json(extracted_path)
    selected = extracted.get("selected_lines")
    if not isinstance(selected, list) or not selected:
        raise SystemExit(f"[fail] selected_lines missing/empty in: {extracted_path}")

    c = 299_792_458.0
    h = 6.626_070_15e-34  # exact (SI)
    e_charge = 1.602_176_634e-19  # exact (J/eV)

    lines_out: list[dict[str, Any]] = []
    for rec in selected:
        if not isinstance(rec, dict):
            continue
        sel = rec.get("selected")
        if not isinstance(sel, dict):
            continue
        lam_nm = _as_float(sel.get("lambda_vac_nm"))
        lam_unc_A = _as_float(sel.get("lambda_vac_unc_A"))
        aki = _as_float(sel.get("Aki_s^-1"))
        if lam_nm is None or lam_nm <= 0:
            continue

        lam_m = lam_nm * 1e-9
        freq_hz = c / lam_m
        energy_ev = (h * freq_hz) / e_charge

        lines_out.append(
            {
                "id": str(rec.get("id") or ""),
                "lambda_vac_nm": float(lam_nm),
                "lambda_vac_unc_nm": (None if lam_unc_A is None else float(lam_unc_A / 10.0)),
                "frequency_THz": float(freq_hz / 1e12),
                "photon_energy_eV": float(energy_ev),
                "Aki_s^-1": (None if aki is None else float(aki)),
                "Acc": sel.get("Acc"),
                "Type": sel.get("Type"),
            }
        )

    if not lines_out:
        raise SystemExit("[fail] no usable lines parsed from extracted_values.json")

    lines_out.sort(key=lambda r: float(r["lambda_vac_nm"]))

    # ---- Figure ----
    fig_w, fig_h, dpi = 11.5, 4.6, 180
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_title("Phase 7 / Step 7.12: Atomic baseline (Helium, NIST ASD)", fontsize=13)
    ax.set_xlabel("Vacuum wavelength λ [nm] (from NIST ASD)")
    ax.set_yticks([])
    ax.set_xlim(350, 720)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="x", alpha=0.25)

    for i, r in enumerate(lines_out):
        lam = float(r["lambda_vac_nm"])
        label = str(r.get("id") or "")
        ax.axvline(lam, color="#2f855a", lw=2.2, alpha=0.95)
        y = 0.82 if (i % 2 == 0) else 0.58
        ax.text(
            lam + 2.0,
            y,
            f"{label}\n{lam:.3f} nm",
            fontsize=9.6,
            ha="left",
            va="center",
            color="#2f855a",
        )

    ax.text(
        0.01,
        0.05,
        "Data: NIST ASD (He I)\nThis figure fixes baseline targets for Part III; it is not a derivation.",
        transform=ax.transAxes,
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )

    fig.tight_layout()
    out_png = out_dir / "atomic_helium_baseline.png"
    fig.savefig(out_png)
    plt.close(fig)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.12",
        "title": "Atomic baseline (Helium; NIST ASD)",
        "source_cache": {
            "extracted_values": str(extracted_path),
        },
        "lines": lines_out,
        "note": (
            "This output fixes a small set of observed vacuum wavelengths as a reproducible baseline (multi-electron). "
            "P-model derivation of atomic/molecular binding is tracked in Roadmap Step 7.12+."
        ),
    }
    out_json = out_dir / "atomic_helium_baseline_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_json}")


if __name__ == "__main__":
    main()

