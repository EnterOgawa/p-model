from __future__ import annotations

import argparse
import bz2
import csv
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


def _read_bz2_lines(path: Path) -> list[str]:
    raw = path.read_bytes()
    text = bz2.decompress(raw).decode("utf-8", errors="replace")
    return [ln for ln in text.splitlines() if ln.strip()]


def _read_text_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [ln for ln in text.splitlines() if ln.strip()]


def _cm_inv_to_hz(cm_inv: float) -> float:
    c = 299_792_458.0
    return c * (cm_inv * 100.0)


def _cm_inv_to_ev(cm_inv: float) -> float:
    h = 6.626_070_15e-34  # exact (SI)
    e_charge = 1.602_176_634e-19  # exact (J/eV)
    return (h * _cm_inv_to_hz(cm_inv)) / e_charge


def _cm_inv_to_um(cm_inv: float) -> float:
    if cm_inv == 0.0:
        return math.inf
    return 1e4 / cm_inv


def _safe_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return None


def _parse_state_row(parts: list[str]) -> dict[str, Any] | None:
    """
    ExoMol states format varies by dataset. For our two datasets:
      H2/RACPPK:  i, E, g, J, v
      HD/ADJSAAM: i, E, g, J, lifetime, v, parity
    """
    if len(parts) < 5:
        return None
    try:
        state_id = int(parts[0])
        energy_cm = float(parts[1])
        g = int(float(parts[2]))
        j = int(float(parts[3]))
    except Exception:
        return None

    if len(parts) == 5:
        v = int(float(parts[4]))
        return {"id": state_id, "E_cm^-1": energy_cm, "g": g, "J": j, "v": v, "parity": None, "lifetime_s": None}

    # >= 7 columns (HD dataset)
    v = None
    parity = None
    lifetime_s = None
    if len(parts) >= 6:
        try:
            v = int(float(parts[5]))
        except Exception:
            v = None
    if len(parts) >= 7:
        parity = parts[6].strip() or None
    if len(parts) >= 5:
        lt = parts[4].strip()
        if lt.lower() != "inf" and lt != "":
            try:
                lifetime_s = float(lt)
            except Exception:
                lifetime_s = None

    return {"id": state_id, "E_cm^-1": energy_cm, "g": g, "J": j, "v": v, "parity": parity, "lifetime_s": lifetime_s}


def _load_states_map(states_bz2: Path) -> dict[int, dict[str, Any]]:
    states: dict[int, dict[str, Any]] = {}
    for line in _read_bz2_lines(states_bz2):
        parts = line.split()
        rec = _parse_state_row(parts)
        if rec is None:
            continue
        states[int(rec["id"])] = rec
    if not states:
        raise SystemExit(f"[fail] no states parsed: {states_bz2}")
    return states


def _load_transitions(trans_bz2: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in _read_bz2_lines(trans_bz2):
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            upper = int(parts[0])
            lower = int(parts[1])
            A = float(parts[2])
            nu_cm = float(parts[3])
        except Exception:
            continue
        rows.append({"upper": upper, "lower": lower, "A_s^-1": A, "wavenumber_cm^-1": nu_cm})
    if not rows:
        raise SystemExit(f"[fail] no transitions parsed: {trans_bz2}")
    return rows


def _transition_label(*, upper: dict[str, Any], lower: dict[str, Any]) -> str:
    def fmt_state(s: dict[str, Any]) -> str:
        v = s.get("v")
        j = s.get("J")
        p = s.get("parity")
        if v is None:
            return f"J={j}"
        if p is None:
            return f"v={v},J={j}"
        return f"v={v},J={j},{p}"

    return f"{fmt_state(upper)} → {fmt_state(lower)}"


def _molat_transition_label(*, upper_state: str, vu: int, ju: int, vl: int, jl: int) -> str:
    return f"{upper_state}(v={vu},J={ju}) → X(v={vl},J={jl})"


def _load_molat_d2_transitions(*, source_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    extracted_path = source_dir / "extracted_values.json"
    extracted = _read_json(extracted_path)
    files = extracted.get("files") if isinstance(extracted.get("files"), list) else []
    rows: list[dict[str, Any]] = []
    for f in files:
        if not isinstance(f, dict):
            continue
        upper_state = str(f.get("upper_state_label") or "").strip() or "?"
        pre_file = f.get("pre_text_file")
        if not pre_file:
            continue
        pre_path = source_dir / str(pre_file)
        for line in _read_text_lines(pre_path):
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                vu = int(float(parts[0]))
                ju = int(float(parts[1]))
                vl = int(float(parts[2]))
                jl = int(float(parts[3]))
                a = float(parts[4].replace("D", "E"))
                nu_cm = float(parts[5])
            except Exception:
                continue
            rows.append(
                {
                    "upper_state": upper_state,
                    "upper_v": vu,
                    "upper_J": ju,
                    "lower_v": vl,
                    "lower_J": jl,
                    "A_s^-1": a,
                    "wavenumber_cm^-1": nu_cm,
                }
            )
    if not rows:
        raise SystemExit(f"[fail] no MOLAT D2 transitions parsed: {source_dir}")
    return rows, extracted_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an offline-stable molecular transition baseline from ExoMol line lists.")
    ap.add_argument("--top-n", type=int, default=10, help="Number of representative transitions to select per molecule.")
    args = ap.parse_args()

    top_n = int(args.top_n)
    if top_n <= 0:
        raise SystemExit("[fail] --top-n must be >= 1")

    root = _repo_root()
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: list[dict[str, Any]] = [
        {
            "kind": "exomol",
            "label": "H2",
            "slug": "exomol_h2_1h2_racppk",
            "file_prefix": "1H2__RACPPK",
            "dataset_tag": "RACPPK",
            "note": "ExoMol recommended ground-state line list (quadrupole + dipole contributions summed).",
        },
        {
            "kind": "exomol",
            "label": "HD",
            "slug": "exomol_h2_1h_2h_adjsaam",
            "file_prefix": "1H-2H__ADJSAAM",
            "dataset_tag": "ADJSAAM",
            "note": "ExoMol recommended ab initio line list (HD isotopologue).",
        },
    ]

    molat_dir = root / "data" / "quantum" / "sources" / "molat_d2_fuv_emission_arlsj1999"
    if (molat_dir / "extracted_values.json").exists():
        datasets.append(
            {
                "kind": "molat_d2_fuv",
                "label": "D2",
                "slug": "molat_d2_fuv_emission_arlsj1999",
                "dataset_tag": "ARLSJ1999 (FUV; B/C/B′/D→X)",
                "note": "MOLAT (OBSPM) D2 FUV emission line lists (transition energies + A; Abgrall et al. 1999).",
            }
        )
    else:
        print(f"[warn] MOLAT D2 cache not found; skipping D2: {molat_dir}")

    rows_all: list[dict[str, Any]] = []
    per_dataset: list[dict[str, Any]] = []

    for ds in datasets:
        kind = str(ds.get("kind") or "")
        src_dir = root / "data" / "quantum" / "sources" / str(ds["slug"])

        if kind == "molat_d2_fuv":
            molat_rows, extracted_path = _load_molat_d2_transitions(source_dir=src_dir)
            molat_sorted = sorted(molat_rows, key=lambda r: float(r["A_s^-1"]), reverse=True)
            selected = molat_sorted[:top_n]
            if not selected:
                raise SystemExit(f"[fail] empty MOLAT selection for: {src_dir}")

            a_max = float(selected[0]["A_s^-1"])
            nu_vals = [float(r["wavenumber_cm^-1"]) for r in selected]
            nu_min = float(min(nu_vals))
            nu_max = float(max(nu_vals))

            for rank, r in enumerate(selected, start=1):
                nu_cm = float(r["wavenumber_cm^-1"])
                A = float(r["A_s^-1"])
                upper_state = str(r.get("upper_state") or "?")
                vu = int(r["upper_v"])
                ju = int(r["upper_J"])
                vl = int(r["lower_v"])
                jl = int(r["lower_J"])
                rows_all.append(
                    {
                        "dataset": "MOLAT",
                        "molecule": ds["label"],
                        "dataset_tag": ds["dataset_tag"],
                        "rank_by_A_desc": rank,
                        "upper_v": vu,
                        "upper_J": ju,
                        "upper_parity": None,
                        "lower_v": vl,
                        "lower_J": jl,
                        "lower_parity": None,
                        "wavenumber_cm^-1": nu_cm,
                        "wavelength_um": _cm_inv_to_um(nu_cm),
                        "frequency_THz": _cm_inv_to_hz(nu_cm) / 1e12,
                        "photon_energy_eV": _cm_inv_to_ev(nu_cm),
                        "A_s^-1": A,
                        "transition_label": _molat_transition_label(upper_state=upper_state, vu=vu, ju=ju, vl=vl, jl=jl),
                        "source_cache": str(extracted_path),
                    }
                )

            per_dataset.append(
                {
                    "molecule": ds["label"],
                    "dataset_tag": ds["dataset_tag"],
                    "source": "MOLAT",
                    "top_n": top_n,
                    "A_max_s^-1": a_max,
                    "wavenumber_range_cm^-1": [nu_min, nu_max],
                    "source_cache": str(extracted_path),
                    "notes": [ds["note"]],
                }
            )
            continue

        extracted_path = src_dir / "extracted_values.json"
        if not extracted_path.exists():
            raise SystemExit(
                f"[fail] missing ExoMol cache: {extracted_path}\n"
                "Run: python -B scripts/quantum/fetch_exomol_diatomic_line_lists.py"
            )

        prefix = str(ds["file_prefix"])
        states_bz2 = src_dir / f"{prefix}.states.bz2"
        trans_bz2 = src_dir / f"{prefix}.trans.bz2"
        pf_path = src_dir / f"{prefix}.pf"
        if not (states_bz2.exists() and trans_bz2.exists() and pf_path.exists()):
            raise SystemExit(f"[fail] missing raw ExoMol files under: {src_dir}")

        states = _load_states_map(states_bz2)
        trans = _load_transitions(trans_bz2)

        # Select top-N transitions by Einstein A coefficient (objective & reproducible).
        trans_sorted = sorted(trans, key=lambda r: float(r["A_s^-1"]), reverse=True)
        selected = trans_sorted[:top_n]

        # Derived summary metrics.
        a_max = float(selected[0]["A_s^-1"])
        nu_vals = [float(r["wavenumber_cm^-1"]) for r in selected]
        nu_min = float(min(nu_vals))
        nu_max = float(max(nu_vals))

        for rank, r in enumerate(selected, start=1):
            upper = states.get(int(r["upper"]))
            lower = states.get(int(r["lower"]))
            if upper is None or lower is None:
                raise SystemExit(f"[fail] state id missing in map (upper={r['upper']} lower={r['lower']}) for {ds['label']}")
            nu_cm = float(r["wavenumber_cm^-1"])
            A = float(r["A_s^-1"])
            rows_all.append(
                {
                    "dataset": "ExoMol",
                    "molecule": ds["label"],
                    "dataset_tag": ds["dataset_tag"],
                    "rank_by_A_desc": rank,
                    "upper_id": int(r["upper"]),
                    "lower_id": int(r["lower"]),
                    "upper_v": upper.get("v"),
                    "upper_J": upper.get("J"),
                    "upper_parity": upper.get("parity"),
                    "lower_v": lower.get("v"),
                    "lower_J": lower.get("J"),
                    "lower_parity": lower.get("parity"),
                    "wavenumber_cm^-1": nu_cm,
                    "wavelength_um": _cm_inv_to_um(nu_cm),
                    "frequency_THz": _cm_inv_to_hz(nu_cm) / 1e12,
                    "photon_energy_eV": _cm_inv_to_ev(nu_cm),
                    "A_s^-1": A,
                    "transition_label": _transition_label(upper=upper, lower=lower),
                    "source_cache": str(extracted_path),
                }
            )

        per_dataset.append(
            {
                "molecule": ds["label"],
                "dataset_tag": ds["dataset_tag"],
                "source": "ExoMol",
                "top_n": top_n,
                "A_max_s^-1": a_max,
                "wavenumber_range_cm^-1": [nu_min, nu_max],
                "source_cache": str(extracted_path),
                "notes": [ds["note"]],
            }
        )

    # ---- Figure (text table, stable & readable in paper) ----
    n_panels = max(1, len(per_dataset))
    ncols = 2 if n_panels > 1 else 1
    nrows = int(math.ceil(n_panels / ncols))
    fig_w = 11.5
    fig_h = 5.6 if nrows == 1 else 8.2

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=180)
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]  # type: ignore[truthy-bool]
    for ax in axes_flat:
        ax.set_axis_off()

    fig.suptitle(
        "Phase 7 / Step 7.12: Molecular transition baseline (primary line lists; representative transitions)",
        fontsize=13,
        y=0.98,
    )

    for ax, ds in zip(axes_flat, per_dataset, strict=False):
        mol = str(ds["molecule"])
        dataset_tag = str(ds["dataset_tag"])
        src = str(ds.get("source") or "").strip()
        src_prefix = f"{src} " if src else ""
        title = f"{mol} ({src_prefix}{dataset_tag}; top {top_n} by A)"
        ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=12.0, va="top")

        # Rows for this molecule.
        sub = [r for r in rows_all if str(r["molecule"]) == mol]
        sub = sorted(sub, key=lambda r: int(r["rank_by_A_desc"]))

        y = 0.86
        dy = 0.072
        for r in sub:
            rank = int(r["rank_by_A_desc"])
            nu = float(r["wavenumber_cm^-1"])
            lam = float(r["wavelength_um"])
            A = float(r["A_s^-1"])
            label = str(r["transition_label"])
            ax.text(
                0.03,
                y,
                f"{rank:>2d}) {label}",
                transform=ax.transAxes,
                fontsize=9.3,
                ha="left",
                va="center",
            )
            ax.text(
                0.03,
                y - 0.032,
                f"    ν̃={nu:.6f} cm⁻¹  λ={lam:.3f} μm  A={A:.3e} s⁻¹",
                transform=ax.transAxes,
                fontsize=9.1,
                ha="left",
                va="center",
                color="#222222",
            )
            y -= dy
            if y < 0.18:
                break

        ax.text(
            0.02,
            0.06,
            "Selection rule: pick top-N transitions by Einstein A.\n"
            "This fixes targets; it is not a P-model derivation.",
            transform=ax.transAxes,
            fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
        )

    for ax in axes_flat[len(per_dataset) :]:
        ax.set_axis_off()

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png = out_dir / "molecular_transitions_exomol_baseline.png"
    fig.savefig(out_png)
    plt.close(fig)

    # ---- CSV ----
    out_csv = out_dir / "molecular_transitions_exomol_baseline_selected.csv"
    csv_cols = [
        "dataset",
        "molecule",
        "dataset_tag",
        "rank_by_A_desc",
        "upper_v",
        "upper_J",
        "upper_parity",
        "lower_v",
        "lower_J",
        "lower_parity",
        "wavenumber_cm^-1",
        "wavelength_um",
        "frequency_THz",
        "photon_energy_eV",
        "A_s^-1",
        "transition_label",
        "source_cache",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in rows_all:
            w.writerow({k: r.get(k) for k in csv_cols})

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.12",
        "title": "Molecular transition baseline (primary line lists; representative transitions)",
        "selection": {"method": "top_n_by_Einstein_A_desc", "top_n_per_molecule": top_n},
        "note": (
            "Representative transitions are selected objectively by descending Einstein A from primary line lists "
            "(ExoMol for H2/HD; MOLAT for D2). This output fixes a small baseline target set for future "
            "molecular-binding derivations."
        ),
        "datasets": per_dataset,
        "rows": rows_all,
        "outputs": {"png": str(out_png), "csv": str(out_csv)},
    }
    out_json = out_dir / "molecular_transitions_exomol_baseline_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_json}")


if __name__ == "__main__":
    main()
