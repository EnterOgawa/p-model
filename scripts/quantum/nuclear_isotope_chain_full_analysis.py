from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


REPRESENTATIVE_Z = [8, 20, 50, 82]


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


def _rms(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return math.sqrt(sum(v * v for v in finite) / float(len(finite)))


def _safe_median(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return float(median(finite))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        if not rows:
            f.write("")
            return
        headers = list(rows[0].keys())
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row.get(h) for h in headers])


def _edge_class(*, d_proton: int, d_neutron: int, edge_width: int = 2) -> str:
    proton_edge = d_proton <= edge_width
    neutron_edge = d_neutron <= edge_width
    if proton_edge and neutron_edge:
        return "both_edges"
    if proton_edge:
        return "proton_rich_edge"
    if neutron_edge:
        return "neutron_rich_edge"
    return "interior"


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_nuclei_csv = out_dir / "nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei.csv"
    pairing_csv = out_dir / "nuclear_pairing_effect_systematics_per_nucleus.csv"

    if not all_nuclei_csv.exists():
        raise SystemExit(f"[fail] missing required input: {all_nuclei_csv}")
    if not pairing_csv.exists():
        raise SystemExit(
            f"[fail] missing required input: {pairing_csv}\n"
            "Run Step 7.16.3 first:\n"
            "  python -B scripts/quantum/nuclear_pairing_effect_systematics_analysis.py"
        )

    # Base map from Step 7.16.1.
    by_zn: dict[tuple[int, int], dict[str, Any]] = {}
    with all_nuclei_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            z = int(row["Z"])
            n = int(row["N"])
            a = int(row["A"])
            b_obs = float(row["B_obs_MeV"])
            b_pred_before = float(row["B_pred_collective_MeV"])
            if not (math.isfinite(b_obs) and math.isfinite(b_pred_before)):
                continue
            by_zn[(z, n)] = {
                "Z": z,
                "N": n,
                "A": a,
                "symbol": str(row.get("symbol", "")),
                "B_obs_MeV": b_obs,
                "B_pred_before_MeV": b_pred_before,
            }

    if not by_zn:
        raise SystemExit(f"[fail] no usable nuclei rows: {all_nuclei_csv}")

    # Pairing-corrected predictions from Step 7.16.3.
    b_pred_after_map: dict[tuple[int, int], float] = {}
    with pairing_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            z = int(row["Z"])
            n = int(row["N"])
            val = float(row["B_pred_after_MeV"])
            if math.isfinite(val):
                b_pred_after_map[(z, n)] = val

    by_z_nlist: dict[int, list[int]] = defaultdict(list)
    for z, n in by_zn.keys():
        by_z_nlist[z].append(n)
    for z in list(by_z_nlist.keys()):
        by_z_nlist[z] = sorted(set(by_z_nlist[z]))

    rows: list[dict[str, Any]] = []
    for z in sorted(by_z_nlist.keys()):
        n_list = by_z_nlist[z]
        n_min = int(min(n_list))
        n_max = int(max(n_list))
        for n in n_list:
            parent = by_zn.get((z, n))
            prev = by_zn.get((z, n - 1))
            if parent is None or prev is None:
                continue
            sn_obs = float(parent["B_obs_MeV"] - prev["B_obs_MeV"])
            sn_pred_before = float(parent["B_pred_before_MeV"] - prev["B_pred_before_MeV"])
            b_after_parent = b_pred_after_map.get((z, n), float("nan"))
            b_after_prev = b_pred_after_map.get((z, n - 1), float("nan"))
            if math.isfinite(b_after_parent) and math.isfinite(b_after_prev):
                sn_pred_after = float(b_after_parent - b_after_prev)
                resid_after = float(sn_pred_after - sn_obs)
            else:
                sn_pred_after = float("nan")
                resid_after = float("nan")
            resid_before = float(sn_pred_before - sn_obs)

            d_proton = int(n - n_min)
            d_neutron = int(n_max - n)
            row = {
                "Z": z,
                "N_parent": n,
                "A_parent": int(parent["A"]),
                "symbol": str(parent["symbol"]),
                "N_chain_min": n_min,
                "N_chain_max": n_max,
                "n_chain_isotopes": len(n_list),
                "distance_proton_edge": d_proton,
                "distance_neutron_edge": d_neutron,
                "edge_class": _edge_class(d_proton=d_proton, d_neutron=d_neutron),
                "S_n_obs_MeV": sn_obs,
                "S_n_pred_before_MeV": sn_pred_before,
                "S_n_pred_after_MeV": sn_pred_after,
                "resid_before_MeV": resid_before,
                "resid_after_MeV": resid_after,
                "abs_resid_before_MeV": abs(resid_before),
                "abs_resid_after_MeV": abs(resid_after) if math.isfinite(resid_after) else float("nan"),
            }
            rows.append(row)

    if not rows:
        raise SystemExit("[fail] no S_n rows produced")

    rows_by_z: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_z[int(row["Z"])].append(row)

    chain_summary_rows: list[dict[str, Any]] = []
    for z in sorted(rows_by_z.keys()):
        z_rows = rows_by_z[z]
        before = [float(r["resid_before_MeV"]) for r in z_rows]
        after = [float(r["resid_after_MeV"]) for r in z_rows if math.isfinite(float(r["resid_after_MeV"]))]
        before_abs = [abs(v) for v in before]
        after_abs = [abs(v) for v in after]
        n_min = min(int(r["N_chain_min"]) for r in z_rows)
        n_max = max(int(r["N_chain_max"]) for r in z_rows)
        symbol = str(z_rows[0]["symbol"])

        edge_neutron_before = [
            float(r["resid_before_MeV"])
            for r in z_rows
            if int(r["distance_neutron_edge"]) <= 2 and int(r["distance_proton_edge"]) > 2
        ]
        edge_neutron_after = [
            float(r["resid_after_MeV"])
            for r in z_rows
            if int(r["distance_neutron_edge"]) <= 2 and int(r["distance_proton_edge"]) > 2 and math.isfinite(float(r["resid_after_MeV"]))
        ]
        edge_proton_before = [
            float(r["resid_before_MeV"])
            for r in z_rows
            if int(r["distance_proton_edge"]) <= 2 and int(r["distance_neutron_edge"]) > 2
        ]
        edge_proton_after = [
            float(r["resid_after_MeV"])
            for r in z_rows
            if int(r["distance_proton_edge"]) <= 2 and int(r["distance_neutron_edge"]) > 2 and math.isfinite(float(r["resid_after_MeV"]))
        ]

        chain_summary_rows.append(
            {
                "Z": z,
                "symbol": symbol,
                "n_sn_points": len(z_rows),
                "N_chain_min": n_min,
                "N_chain_max": n_max,
                "rms_resid_before_MeV": _rms(before),
                "rms_resid_after_MeV": _rms(after),
                "median_abs_resid_before_MeV": _safe_median(before_abs),
                "median_abs_resid_after_MeV": _safe_median(after_abs),
                "rms_resid_neutron_edge_before_MeV": _rms(edge_neutron_before),
                "rms_resid_neutron_edge_after_MeV": _rms(edge_neutron_after),
                "rms_resid_proton_edge_before_MeV": _rms(edge_proton_before),
                "rms_resid_proton_edge_after_MeV": _rms(edge_proton_after),
            }
        )

    representative_rows = [row for row in rows if int(row["Z"]) in REPRESENTATIVE_Z]

    # Global and edge-class metrics.
    all_before = [float(r["resid_before_MeV"]) for r in rows]
    all_after = [float(r["resid_after_MeV"]) for r in rows if math.isfinite(float(r["resid_after_MeV"]))]
    edge_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        edge_by_class[str(row["edge_class"])].append(row)

    edge_stats: dict[str, Any] = {}
    for edge_name in ["proton_rich_edge", "interior", "neutron_rich_edge", "both_edges"]:
        sub = edge_by_class.get(edge_name, [])
        before = [float(r["resid_before_MeV"]) for r in sub]
        after = [float(r["resid_after_MeV"]) for r in sub if math.isfinite(float(r["resid_after_MeV"]))]
        edge_stats[edge_name] = {
            "n": len(sub),
            "rms_before_MeV": _rms(before),
            "rms_after_MeV": _rms(after),
            "median_abs_before_MeV": _safe_median([abs(v) for v in before]),
            "median_abs_after_MeV": _safe_median([abs(v) for v in after]),
        }

    # Figures.
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), dpi=160)
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]

    zs = [int(r["Z"]) for r in chain_summary_rows]
    n_points = [int(r["n_sn_points"]) for r in chain_summary_rows]
    ax00.bar(zs, n_points, color="#4c78a8", alpha=0.85, width=0.9)
    ax00.set_xlabel("Z")
    ax00.set_ylabel("S_n points per isotopic chain")
    ax00.set_title("Isotopic-chain coverage (all Z)")
    ax00.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)

    rms_before_by_z = [float(r["rms_resid_before_MeV"]) for r in chain_summary_rows]
    rms_after_by_z = [float(r["rms_resid_after_MeV"]) for r in chain_summary_rows]
    ax01.plot(zs, rms_before_by_z, marker="o", ms=2.0, lw=1.2, label="before pairing corr.")
    ax01.plot(zs, rms_after_by_z, marker="s", ms=2.0, lw=1.2, label="after pairing corr.")
    ax01.set_xlabel("Z")
    ax01.set_ylabel("RMS residual of S_n [MeV]")
    ax01.set_title("Per-chain S_n residual RMS")
    ax01.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax01.legend(loc="best", fontsize=8)

    neutron_edge_before = [
        abs(float(r["resid_before_MeV"]))
        for r in rows
        if str(r["edge_class"]) == "neutron_rich_edge"
    ]
    neutron_edge_after = [
        abs(float(r["resid_after_MeV"]))
        for r in rows
        if str(r["edge_class"]) == "neutron_rich_edge" and math.isfinite(float(r["resid_after_MeV"]))
    ]
    proton_edge_before = [
        abs(float(r["resid_before_MeV"]))
        for r in rows
        if str(r["edge_class"]) == "proton_rich_edge"
    ]
    proton_edge_after = [
        abs(float(r["resid_after_MeV"]))
        for r in rows
        if str(r["edge_class"]) == "proton_rich_edge" and math.isfinite(float(r["resid_after_MeV"]))
    ]
    ax10.hist(neutron_edge_before, bins=32, alpha=0.45, label="neutron-edge before")
    ax10.hist(neutron_edge_after, bins=32, alpha=0.45, label="neutron-edge after")
    ax10.hist(proton_edge_before, bins=32, alpha=0.30, label="proton-edge before")
    ax10.hist(proton_edge_after, bins=32, alpha=0.30, label="proton-edge after")
    ax10.set_xlabel("abs residual of S_n [MeV]")
    ax10.set_ylabel("count")
    ax10.set_title("Dripline-proxy edge residual distributions")
    ax10.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax10.legend(loc="best", fontsize=8)

    bar_labels = ["proton_rich_edge", "interior", "neutron_rich_edge"]
    y_before = [float(edge_stats[k]["median_abs_before_MeV"]) for k in bar_labels]
    y_after = [float(edge_stats[k]["median_abs_after_MeV"]) for k in bar_labels]
    x = list(range(len(bar_labels)))
    ax11.bar([i - 0.18 for i in x], y_before, width=0.36, label="before pairing corr.", alpha=0.85)
    ax11.bar([i + 0.18 for i in x], y_after, width=0.36, label="after pairing corr.", alpha=0.85)
    ax11.set_xticks(x)
    ax11.set_xticklabels(bar_labels, rotation=10)
    ax11.set_ylabel("median abs residual of S_n [MeV]")
    ax11.set_title("Residual medians by edge class")
    ax11.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax11.legend(loc="best", fontsize=8)

    fig.suptitle("Phase 7 / Step 7.16.4: isotope-chain full analysis (S_n + dripline-proxy precision)", y=0.98)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))

    out_png = out_dir / "nuclear_isotope_chain_full_analysis.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # Representative element detailed plot.
    fig2, axes2 = plt.subplots(2, 2, figsize=(13.0, 8.0), dpi=160, sharey=False)
    axes_flat = [axes2[0, 0], axes2[0, 1], axes2[1, 0], axes2[1, 1]]
    for idx, z in enumerate(REPRESENTATIVE_Z):
        ax = axes_flat[idx]
        sub = [r for r in rows if int(r["Z"]) == z]
        if not sub:
            ax.text(0.5, 0.5, f"Z={z}: no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        sub = sorted(sub, key=lambda r: int(r["N_parent"]))
        xs = [int(r["N_parent"]) for r in sub]
        y_obs = [float(r["S_n_obs_MeV"]) for r in sub]
        y_before = [float(r["S_n_pred_before_MeV"]) for r in sub]
        y_after = [float(r["S_n_pred_after_MeV"]) if math.isfinite(float(r["S_n_pred_after_MeV"])) else float("nan") for r in sub]
        symbol = str(sub[0]["symbol"])
        ax.plot(xs, y_obs, marker="o", ms=2.5, lw=1.1, label="obs")
        ax.plot(xs, y_before, marker="s", ms=2.2, lw=1.0, label="pred before")
        ax.plot(xs, y_after, marker="^", ms=2.2, lw=1.0, label="pred after")
        ax.set_title(f"{symbol} (Z={z})")
        ax.set_xlabel("N")
        ax.set_ylabel("S_n [MeV]")
        ax.grid(True, ls=":", lw=0.6, alpha=0.6)
        if idx == 0:
            ax.legend(loc="best", fontsize=7)

    fig2.suptitle("Step 7.16.4 representative isotopic chains: S_n(obs/pred)", y=0.98)
    fig2.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))
    out_rep_png = out_dir / "nuclear_isotope_chain_representative_elements.png"
    fig2.savefig(out_rep_png, bbox_inches="tight")
    plt.close(fig2)

    out_full_csv = out_dir / "nuclear_isotope_chain_full_analysis.csv"
    out_summary_csv = out_dir / "nuclear_isotope_chain_summary_by_z.csv"
    out_rep_csv = out_dir / "nuclear_isotope_chain_representative_elements.csv"
    _write_csv(out_full_csv, rows)
    _write_csv(out_summary_csv, chain_summary_rows)
    _write_csv(out_rep_csv, representative_rows)

    out_json = out_dir / "nuclear_isotope_chain_full_analysis_metrics.json"
    out_json.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.16.4",
                "inputs": {
                    "all_nuclei_csv": {"path": str(all_nuclei_csv), "sha256": _sha256(all_nuclei_csv)},
                    "pairing_csv": {"path": str(pairing_csv), "sha256": _sha256(pairing_csv)},
                },
                "definitions": {
                    "S_n": "S_n(Z,N)=B(Z,N)-B(Z,N-1)",
                    "dripline_proxy_edge": "edge classes use AME chain envelope at fixed Z (distance<=2 from N_min/N_max)",
                    "residual": "S_n_pred - S_n_obs",
                },
                "counts": {
                    "n_total_sn_rows": len(rows),
                    "n_chains_z": len(chain_summary_rows),
                    "n_representative_rows": len(representative_rows),
                    "n_edge_rows_neutron": int(edge_stats.get("neutron_rich_edge", {}).get("n", 0)),
                    "n_edge_rows_proton": int(edge_stats.get("proton_rich_edge", {}).get("n", 0)),
                },
                "global_stats": {
                    "rms_resid_before_MeV": _rms(all_before),
                    "rms_resid_after_MeV": _rms(all_after),
                    "median_abs_resid_before_MeV": _safe_median([abs(v) for v in all_before]),
                    "median_abs_resid_after_MeV": _safe_median([abs(v) for v in all_after]),
                },
                "edge_stats": edge_stats,
                "representative_Z": REPRESENTATIVE_Z,
                "outputs": {
                    "full_csv": str(out_full_csv),
                    "summary_by_z_csv": str(out_summary_csv),
                    "representative_csv": str(out_rep_csv),
                    "figure_png": str(out_png),
                    "representative_figure_png": str(out_rep_png),
                },
                "notes": [
                    "Step 7.16.4 freezes isotope-chain Sn diagnostics across all available Z chains.",
                    "Dripline-near precision is evaluated with AME envelope proxy edges (distance<=2 in N within each fixed-Z chain).",
                    "Pred-after uses Step 7.16.3 pairing-corrected per-nucleus predictions.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[ok] wrote:")
    print(f"  {out_full_csv}")
    print(f"  {out_summary_csv}")
    print(f"  {out_rep_csv}")
    print(f"  {out_png}")
    print(f"  {out_rep_png}")
    print(f"  {out_json}")


if __name__ == "__main__":
    main()
