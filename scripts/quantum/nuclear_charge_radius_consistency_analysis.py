from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


def _parse_float(text: str) -> float:
    s = str(text).strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


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


def _median_abs(values: list[float]) -> float:
    finite = [abs(float(v)) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return float(median(finite))


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
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow([row.get(h) for h in headers])


def _weighted_mean(values: list[float], sigmas: list[float]) -> tuple[float, float]:
    weighted: list[tuple[float, float]] = []
    for value, sigma in zip(values, sigmas):
        if math.isfinite(value) and math.isfinite(sigma) and sigma > 0.0:
            weighted.append((value, sigma))
    if weighted:
        ws = [1.0 / (sigma * sigma) for _, sigma in weighted]
        numerator = sum(w * value for w, (value, _) in zip(ws, weighted))
        denominator = sum(ws)
        if denominator > 0.0:
            return float(numerator / denominator), float(math.sqrt(1.0 / denominator))

    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("nan"), float("nan")
    center = float(sum(finite_values) / float(len(finite_values)))
    if len(finite_values) <= 1:
        return center, float("nan")
    variance = sum((x - center) ** 2 for x in finite_values) / float(len(finite_values) - 1)
    return center, float(math.sqrt(max(variance, 0.0)))


def _fit_radius_a13(*, a13: list[float], radii: list[float], weights: list[float]) -> float:
    s_xy = 0.0
    s_xx = 0.0
    for x, y, w in zip(a13, radii, weights):
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(w) and w > 0.0):
            continue
        s_xy += w * x * y
        s_xx += w * x * x
    if s_xx <= 0.0:
        raise SystemExit("[fail] radius fit (A^(1/3)) is singular")
    return float(s_xy / s_xx)


def _fit_radius_a13_i(
    *,
    a13: list[float],
    i_asym: list[float],
    radii: list[float],
    weights: list[float],
    ridge: float = 1.0e-12,
) -> tuple[float, float]:
    s11 = 0.0
    s22 = 0.0
    s12 = 0.0
    t1 = 0.0
    t2 = 0.0
    for x1, x2, y, w in zip(a13, i_asym, radii, weights):
        if not (math.isfinite(x1) and math.isfinite(x2) and math.isfinite(y) and math.isfinite(w) and w > 0.0):
            continue
        s11 += w * x1 * x1
        s22 += w * x2 * x2
        s12 += w * x1 * x2
        t1 += w * x1 * y
        t2 += w * x2 * y
    a11 = s11 + ridge
    a22 = s22 + ridge
    det = a11 * a22 - s12 * s12
    if det == 0.0:
        raise SystemExit("[fail] radius fit (A^(1/3)+I) is singular")
    r0 = (t1 * a22 - t2 * s12) / det
    r_i = (a11 * t2 - s12 * t1) / det
    return float(r0), float(r_i)


def _j_from_radius(*, r_rms_fm: float, r_ref_fm: float, l_fm: float, j_ref_mev: float) -> float:
    if not (math.isfinite(r_rms_fm) and math.isfinite(r_ref_fm) and math.isfinite(l_fm) and math.isfinite(j_ref_mev) and l_fm > 0.0):
        return float("nan")
    r_sharp = math.sqrt(5.0 / 3.0) * r_rms_fm
    return float(j_ref_mev * math.exp((r_ref_fm - r_sharp) / l_fm))


def _a_band(a: int) -> str:
    if a <= 16:
        return "A_2_16"
    if a <= 40:
        return "A_17_40"
    if a <= 100:
        return "A_41_100"
    return "A_101_plus"


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    ame_csv = out_dir / "nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei.csv"
    iaea_csv = root / "data" / "quantum" / "sources" / "iaea_charge_radii" / "charge_radii.csv"
    if not ame_csv.exists():
        raise SystemExit(f"[fail] missing required input: {ame_csv}")
    if not iaea_csv.exists():
        raise SystemExit(f"[fail] missing required input: {iaea_csv}")

    ame_by_za: dict[tuple[int, int], dict[str, Any]] = {}
    constants: dict[str, float] = {}
    all_rows_for_j_consistency: list[dict[str, Any]] = []
    with ame_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            z = int(row["Z"])
            n = int(row["N"])
            a = int(row["A"])
            ame_by_za[(z, a)] = row
            all_rows_for_j_consistency.append(row)
            if not constants:
                constants["L_fm"] = float(row["L_fm"])
                constants["R_ref_fm"] = float(row["R_ref_fm"])
                constants["J_ref_MeV"] = float(row["J_ref_MeV"])

    if not ame_by_za:
        raise SystemExit("[fail] AME join map is empty")

    # Merge IAEA duplicates per (Z,A).
    iaea_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    n_iaea_raw = 0
    with iaea_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            z = int(float(row["z"]))
            a = int(float(row["a"]))
            if z < 1 or a < 2:
                continue
            radius = _parse_float(row.get("radius_val", ""))
            sigma = _parse_float(row.get("radius_unc", ""))
            source_flag = "main"
            if not math.isfinite(radius):
                radius = _parse_float(row.get("radius_preliminary_val", ""))
                sigma = _parse_float(row.get("radius_preliminary_unc", ""))
                source_flag = "preliminary"
            if not math.isfinite(radius):
                continue
            n_iaea_raw += 1
            iaea_groups[(z, a)].append(
                {
                    "z": z,
                    "a": a,
                    "n": int(float(row["n"])),
                    "symbol": str(row["symbol"]).strip(),
                    "radius_rms_fm": radius,
                    "sigma_radius_rms_fm": sigma if (math.isfinite(sigma) and sigma > 0.0) else float("nan"),
                    "source_flag": source_flag,
                }
            )

    merged_iaea: dict[tuple[int, int], dict[str, Any]] = {}
    for key, items in iaea_groups.items():
        values = [float(item["radius_rms_fm"]) for item in items]
        sigmas = [float(item["sigma_radius_rms_fm"]) for item in items]
        radius, sigma = _weighted_mean(values, sigmas)
        merged_iaea[key] = {
            "z": key[0],
            "a": key[1],
            "n_measurements": len(items),
            "radius_rms_fm": radius,
            "sigma_radius_rms_fm": sigma,
            "source_flags": ",".join(sorted(set(str(item["source_flag"]) for item in items))),
        }

    joined_rows: list[dict[str, Any]] = []
    for (z, a), iaea in sorted(merged_iaea.items()):
        ame = ame_by_za.get((z, a))
        if ame is None:
            continue
        n = int(ame["N"])
        a13 = float(a ** (1.0 / 3.0))
        i_asym = float((n - z) / a)
        sigma = float(iaea["sigma_radius_rms_fm"])
        weight = float(1.0 / (sigma * sigma)) if (math.isfinite(sigma) and sigma > 0.0) else 1.0
        j_from_r = _j_from_radius(
            r_rms_fm=float(iaea["radius_rms_fm"]),
            r_ref_fm=float(constants["R_ref_fm"]),
            l_fm=float(constants["L_fm"]),
            j_ref_mev=float(constants["J_ref_MeV"]),
        )
        j_stored = float(ame["J_E_MeV"])
        j_rel = float((j_from_r - j_stored) / j_stored) if (math.isfinite(j_from_r) and math.isfinite(j_stored) and j_stored != 0.0) else float("nan")
        joined_rows.append(
            {
                "Z": z,
                "N": n,
                "A": a,
                "symbol": str(ame["symbol"]),
                "parity": str(ame["parity"]),
                "is_magic_any": str(ame["is_magic_any"]).lower() == "true",
                "A13": a13,
                "I_asym": i_asym,
                "radius_rms_obs_fm": float(iaea["radius_rms_fm"]),
                "sigma_radius_rms_obs_fm": sigma,
                "weight": weight,
                "radius_source_merge": str(iaea["source_flags"]),
                "n_iaea_measurements": int(iaea["n_measurements"]),
                "J_from_obs_radius_MeV": j_from_r,
                "J_stored_MeV": j_stored,
                "J_rel_diff": j_rel,
                "B_obs_MeV": float(ame["B_obs_MeV"]),
                "C_required": float(ame["C_required"]),
            }
        )

    if len(joined_rows) < 50:
        raise SystemExit(f"[fail] too few joined charge-radius rows: {len(joined_rows)}")

    a13_vals = [float(r["A13"]) for r in joined_rows]
    i_vals = [float(r["I_asym"]) for r in joined_rows]
    r_vals = [float(r["radius_rms_obs_fm"]) for r in joined_rows]
    w_vals = [float(r["weight"]) for r in joined_rows]

    r0_a13 = _fit_radius_a13(a13=a13_vals, radii=r_vals, weights=w_vals)
    r0_a13_i, r_i = _fit_radius_a13_i(a13=a13_vals, i_asym=i_vals, radii=r_vals, weights=w_vals)

    skin_proxy_rows: list[dict[str, Any]] = []
    for row in joined_rows:
        a13 = float(row["A13"])
        i_asym = float(row["I_asym"])
        r_obs = float(row["radius_rms_obs_fm"])
        pred_a13 = float(r0_a13 * a13)
        pred_a13_i = float(r0_a13_i * a13 + r_i * i_asym)
        res_a13 = float(r_obs - pred_a13)
        res_a13_i = float(r_obs - pred_a13_i)
        row["radius_pred_a13_fm"] = pred_a13
        row["radius_pred_a13_i_fm"] = pred_a13_i
        row["resid_radius_a13_fm"] = res_a13
        row["resid_radius_a13_i_fm"] = res_a13_i
        row["A_band"] = _a_band(int(row["A"]))
        # Proxy interpretation: proton radius contraction from neutron excess maps to a positive skin proxy.
        row["neutron_skin_proxy_fm"] = float(max(0.0, -r_i * i_asym))
        c_req = float(row["C_required"])
        j_pred_a13 = _j_from_radius(
            r_rms_fm=pred_a13,
            r_ref_fm=float(constants["R_ref_fm"]),
            l_fm=float(constants["L_fm"]),
            j_ref_mev=float(constants["J_ref_MeV"]),
        )
        j_pred_a13_i = _j_from_radius(
            r_rms_fm=pred_a13_i,
            r_ref_fm=float(constants["R_ref_fm"]),
            l_fm=float(constants["L_fm"]),
            j_ref_mev=float(constants["J_ref_MeV"]),
        )
        b_pred_a13 = float(2.0 * c_req * j_pred_a13) if math.isfinite(j_pred_a13) else float("nan")
        b_pred_a13_i = float(2.0 * c_req * j_pred_a13_i) if math.isfinite(j_pred_a13_i) else float("nan")
        b_obs = float(row["B_obs_MeV"])
        row["B_pred_from_radius_a13_MeV"] = b_pred_a13
        row["B_pred_from_radius_a13_i_MeV"] = b_pred_a13_i
        row["resid_B_from_radius_a13_MeV"] = float(b_pred_a13 - b_obs) if math.isfinite(b_pred_a13) else float("nan")
        row["resid_B_from_radius_a13_i_MeV"] = float(b_pred_a13_i - b_obs) if math.isfinite(b_pred_a13_i) else float("nan")
        skin_proxy_rows.append(
            {
                "Z": int(row["Z"]),
                "N": int(row["N"]),
                "A": int(row["A"]),
                "symbol": str(row["symbol"]),
                "I_asym": i_asym,
                "neutron_skin_proxy_fm": float(row["neutron_skin_proxy_fm"]),
                "radius_resid_a13_i_fm": res_a13_i,
            }
        )

    # A-band summary.
    band_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        band_map[str(row["A_band"])].append(row)

    band_rows: list[dict[str, Any]] = []
    for band in ["A_2_16", "A_17_40", "A_41_100", "A_101_plus"]:
        sub = band_map.get(band, [])
        band_rows.append(
            {
                "A_band": band,
                "n": len(sub),
                "rms_radius_resid_a13_fm": _rms([float(r["resid_radius_a13_fm"]) for r in sub]),
                "rms_radius_resid_a13_i_fm": _rms([float(r["resid_radius_a13_i_fm"]) for r in sub]),
                "median_abs_radius_resid_a13_fm": _median_abs([float(r["resid_radius_a13_fm"]) for r in sub]),
                "median_abs_radius_resid_a13_i_fm": _median_abs([float(r["resid_radius_a13_i_fm"]) for r in sub]),
                "median_neutron_skin_proxy_fm": _safe_median([float(r["neutron_skin_proxy_fm"]) for r in sub]),
            }
        )

    # J_E(R_model) internal consistency over all nuclei.
    j_internal_rel_diffs: list[float] = []
    for row in all_rows_for_j_consistency:
        j_stored = float(row["J_E_MeV"])
        r_model = float(row["R_model_fm"])
        l_fm = float(constants["L_fm"])
        r_ref_fm = float(constants["R_ref_fm"])
        j_ref_mev = float(constants["J_ref_MeV"])
        j_calc = float(j_ref_mev * math.exp((r_ref_fm - r_model) / l_fm))
        if j_stored != 0.0 and math.isfinite(j_stored):
            j_internal_rel_diffs.append((j_calc - j_stored) / j_stored)

    # Output CSVs.
    full_rows_sorted = sorted(joined_rows, key=lambda r: (int(r["A"]), int(r["Z"]), int(r["N"])))
    skin_proxy_top = sorted(skin_proxy_rows, key=lambda r: float(r["neutron_skin_proxy_fm"]), reverse=True)[:50]

    out_full_csv = out_dir / "nuclear_charge_radius_consistency_full.csv"
    out_band_csv = out_dir / "nuclear_charge_radius_consistency_a_band_summary.csv"
    out_skin_csv = out_dir / "nuclear_charge_radius_consistency_neutron_skin_proxy.csv"
    _write_csv(out_full_csv, full_rows_sorted)
    _write_csv(out_band_csv, band_rows)
    _write_csv(out_skin_csv, skin_proxy_top)

    # Plot.
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), dpi=160)
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]

    x_a13 = [float(r["A13"]) for r in full_rows_sorted]
    y_r = [float(r["radius_rms_obs_fm"]) for r in full_rows_sorted]
    i_vals_sorted = [float(r["I_asym"]) for r in full_rows_sorted]
    ax00.scatter(x_a13, y_r, c=i_vals_sorted, s=9.0, alpha=0.35, cmap="coolwarm")
    x_min = min(x_a13)
    x_max = max(x_a13)
    x_line = [x_min + (x_max - x_min) * i / 300.0 for i in range(301)]
    ax00.plot(x_line, [r0_a13 * x for x in x_line], lw=2.0, label=f"A^(1/3) fit: r0={r0_a13:.4f} fm", color="#1f77b4")
    ax00.plot(
        x_line,
        [r0_a13_i * x for x in x_line],
        lw=2.0,
        ls="--",
        label=f"A^(1/3)+I fit (I=0): r0={r0_a13_i:.4f} fm",
        color="#ff7f0e",
    )
    ax00.set_xlabel("A^(1/3)")
    ax00.set_ylabel("charge radius r_rms [fm]")
    ax00.set_title("r_ch vs A^(1/3) with frozen fits")
    ax00.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax00.legend(loc="best", fontsize=8)

    res_a13 = [float(r["resid_radius_a13_fm"]) for r in full_rows_sorted]
    res_a13_i = [float(r["resid_radius_a13_i_fm"]) for r in full_rows_sorted]
    ax01.hist(res_a13, bins=45, alpha=0.55, label="resid: A^(1/3)", color="#1f77b4")
    ax01.hist(res_a13_i, bins=45, alpha=0.55, label="resid: A^(1/3)+I", color="#ff7f0e")
    ax01.axvline(0.0, color="k", lw=1.0, ls="--")
    ax01.set_xlabel("radius residual [fm]")
    ax01.set_ylabel("count")
    ax01.set_title("Residual distribution (charge radius)")
    ax01.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax01.legend(loc="best", fontsize=8)

    j_obs = [float(r["J_stored_MeV"]) for r in full_rows_sorted]
    j_from_r = [float(r["J_from_obs_radius_MeV"]) for r in full_rows_sorted]
    ax10.scatter(j_obs, j_from_r, c=x_a13, s=10.0, alpha=0.35, cmap="viridis")
    j_lo = min(min(j_obs), min(j_from_r))
    j_hi = max(max(j_obs), max(j_from_r))
    ax10.plot([j_lo, j_hi], [j_lo, j_hi], color="k", ls="--", lw=1.1, label="y=x")
    ax10.set_xlabel("J_E stored [MeV]")
    ax10.set_ylabel("J_E from observed r_ch [MeV]")
    ax10.set_title("J_E(R) self-consistency on measured-radii subset")
    ax10.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax10.legend(loc="best", fontsize=8)

    skin_x = [float(r["I_asym"]) for r in full_rows_sorted]
    skin_y = [float(r["neutron_skin_proxy_fm"]) for r in full_rows_sorted]
    ax11.scatter(skin_x, skin_y, c=[int(r["A"]) for r in full_rows_sorted], s=10.0, alpha=0.35, cmap="plasma")
    ax11.set_xlabel("I=(N-Z)/A")
    ax11.set_ylabel("neutron-skin proxy [fm]")
    ax11.set_title("Neutron-skin proxy from asymmetry term")
    ax11.grid(True, ls=":", lw=0.6, alpha=0.6)

    fig.suptitle("Phase 7 / Step 7.16.5: charge-radius consistency (r_ch + J_E(R) + skin proxy)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_png = out_dir / "nuclear_charge_radius_consistency.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.16.5",
        "inputs": {
            "ame_all_nuclei_csv": {"path": str(ame_csv), "sha256": _sha256(ame_csv)},
            "iaea_charge_radii_csv": {"path": str(iaea_csv), "sha256": _sha256(iaea_csv)},
        },
        "counts": {
            "n_iaea_raw_rows": n_iaea_raw,
            "n_iaea_unique_za": len(merged_iaea),
            "n_joined_rows": len(full_rows_sorted),
            "n_ame_all_rows": len(all_rows_for_j_consistency),
        },
        "radius_fit": {
            "model_a13": {"formula": "r_ch = r0*A^(1/3)", "r0_fm": r0_a13},
            "model_a13_i": {"formula": "r_ch = r0*A^(1/3) + rI*I", "r0_fm": r0_a13_i, "rI_fm": r_i},
        },
        "residual_stats_radius_fm": {
            "rms_a13": _rms(res_a13),
            "rms_a13_i": _rms(res_a13_i),
            "median_abs_a13": _median_abs(res_a13),
            "median_abs_a13_i": _median_abs(res_a13_i),
        },
        "j_consistency": {
            "measured_subset_median_abs_rel_diff": _median_abs([float(r["J_rel_diff"]) for r in full_rows_sorted]),
            "measured_subset_rms_rel_diff": _rms([float(r["J_rel_diff"]) for r in full_rows_sorted]),
            "internal_all_nuclei_median_abs_rel_diff": _median_abs(j_internal_rel_diffs),
            "internal_all_nuclei_rms_rel_diff": _rms(j_internal_rel_diffs),
        },
        "a_band_summary": band_rows,
        "neutron_skin_proxy": {
            "definition": "max(0, -rI*I) from r_ch=A^(1/3)+I fit (proxy, not direct neutron-radius measurement)",
            "top50_csv": str(out_skin_csv),
            "median_proxy_fm": _safe_median([float(r["neutron_skin_proxy_fm"]) for r in full_rows_sorted]),
            "max_proxy_fm": max(float(r["neutron_skin_proxy_fm"]) for r in full_rows_sorted),
        },
        "outputs": {
            "full_csv": str(out_full_csv),
            "a_band_summary_csv": str(out_band_csv),
            "skin_proxy_csv": str(out_skin_csv),
            "figure_png": str(out_png),
        },
        "notes": [
            "Step 7.16.5 integrates IAEA charge radii with AME2020 all-nuclei table via (Z,A).",
            "J_E(R) self-consistency is checked on measured-radii joins and on internal all-nuclei reconstruction from R_model.",
            "Neutron-skin values are reported as asymmetry-driven proxies under the frozen radius fit; they are not direct r_n-r_p measurements.",
        ],
    }
    out_json = out_dir / "nuclear_charge_radius_consistency_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] step=7.16.5 rows={len(full_rows_sorted)} metrics={out_json}")


if __name__ == "__main__":
    main()
