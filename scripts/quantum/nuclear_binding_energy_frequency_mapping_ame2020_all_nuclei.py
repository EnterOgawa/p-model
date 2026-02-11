from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


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


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(sorted_vals: list[float], p: float) -> float:
    """
    Inclusive percentile with linear interpolation.
    p in [0,100].
    """
    if not sorted_vals:
        raise ValueError("empty")
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    x = (len(sorted_vals) - 1) * (p / 100.0)
    i0 = int(math.floor(x))
    i1 = int(math.ceil(x))
    if i0 == i1:
        return float(sorted_vals[i0])
    w = x - i0
    return float((1.0 - w) * sorted_vals[i0] + w * sorted_vals[i1])


def _require_float(obj: object, *, path: Path, key_path: str) -> float:
    try:
        v = float(obj)
    except Exception as e:
        raise SystemExit(f"[fail] invalid float at {key_path} in {path}: {e}") from e
    if not math.isfinite(v):
        raise SystemExit(f"[fail] non-finite float at {key_path} in {path}")
    return v


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("nan")
    return numerator / denominator


def _load_ame2020_rows(*, root: Path, src_dirname: str) -> list[dict[str, object]]:
    src_dir = root / "data" / "quantum" / "sources" / src_dirname
    extracted = src_dir / "extracted_values.json"
    if not extracted.exists():
        raise SystemExit(
            "[fail] missing extracted AME2020 table.\n"
            "Run:\n"
            f"  python -B scripts/quantum/fetch_ame2020_mass_table_sources.py --out-dirname {src_dirname}\n"
            f"Expected: {extracted}"
        )
    payload = json.loads(extracted.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"[fail] invalid extracted_values.json: rows missing/empty: {extracted}")
    out: list[dict[str, object]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if not all(k in r for k in ("Z", "A", "binding_keV_per_A")):
            continue
        out.append(r)
    if not out:
        raise SystemExit(f"[fail] parsed 0 usable AME rows from: {extracted}")
    return out


def _load_iaea_charge_radii_csv(*, root: Path, src_dirname: str) -> dict[tuple[int, int], dict[str, float]]:
    import csv as csv_lib

    src_dir = root / "data" / "quantum" / "sources" / src_dirname
    csv_path = src_dir / "charge_radii.csv"
    if not csv_path.exists():
        raise SystemExit(
            "[fail] missing cached IAEA charge_radii.csv.\n"
            "Run:\n"
            f"  python -B scripts/quantum/fetch_nuclear_charge_radii_sources.py --out-dirname {src_dirname}\n"
            f"Expected: {csv_path}"
        )

    out: dict[tuple[int, int], dict[str, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv_lib.DictReader(f)
        for row in reader:
            try:
                z = int(row["z"])
                a = int(row["a"])
            except Exception:
                continue
            rv = str(row.get("radius_val", "")).strip()
            ru = str(row.get("radius_unc", "")).strip()
            if not rv or not ru:
                continue
            try:
                out[(z, a)] = {"r_rms_fm": float(rv), "sigma_r_rms_fm": float(ru)}
            except Exception:
                continue
    if not out:
        raise SystemExit(f"[fail] parsed 0 charge radii rows from: {csv_path}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "AME2020 all-nuclei residual surface for the minimal Δω→B.E. mapping "
            "(with an optional subset gate for fast end-to-end checks)."
        )
    )
    ap.add_argument(
        "--subset",
        choices=["all", "measured_radii", "measured_radii_neighborhood"],
        default="all",
        help=(
            "Subset selection for a fast gate run. "
            "'all' keeps the canonical full outputs. "
            "'measured_radii' selects nuclei with measured r_rms in IAEA charge_radii.csv. "
            "'measured_radii_neighborhood' expands that set by (ΔZ,ΔN)<=1."
        ),
    )
    args = ap.parse_args(list(argv) if argv is not None else None)
    subset = str(args.subset)

    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_stem = (
        "nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei"
        if subset == "all"
        else f"nuclear_binding_energy_frequency_mapping_ame2020_subset_{subset}"
    )
    subset_desc: str | None = None
    metrics_step = "7.13.17.7" if subset == "all" else "7.18.5"
    metrics_extended_step = "7.16.1" if subset == "all" else "7.18.5"

    # Frozen anchors from earlier steps.
    deut_path = root / "output" / "quantum" / "nuclear_binding_deuteron_metrics.json"
    if not deut_path.exists():
        raise SystemExit(
            "[fail] missing deuteron binding baseline metrics.\n"
            "Run:\n"
            "  python -B scripts/quantum/nuclear_binding_deuteron.py\n"
            f"Expected: {deut_path}"
        )
    deut = _load_json(deut_path)

    two_body_path = root / "output" / "quantum" / "nuclear_binding_energy_frequency_mapping_deuteron_two_body_metrics.json"
    if not two_body_path.exists():
        raise SystemExit(
            "[fail] missing deuteron two-body mapping metrics.\n"
            "Run:\n"
            "  python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_deuteron_two_body.py\n"
            f"Expected: {two_body_path}"
        )
    two_body = _load_json(two_body_path)

    b_d = _require_float(
        deut.get("derived", {}).get("binding_energy", {}).get("B_MeV", {}).get("value"),
        path=deut_path,
        key_path="derived.binding_energy.B_MeV.value",
    )
    r_ref = _require_float(deut.get("derived", {}).get("inv_kappa_fm"), path=deut_path, key_path="derived.inv_kappa_fm")
    if not (b_d > 0 and r_ref > 0):
        raise SystemExit("[fail] invalid deuteron anchor (B_d or R_ref non-positive)")
    j_ref_mev = 0.5 * b_d

    l_range = _require_float(two_body.get("derived", {}).get("lambda_pi_pm_fm"), path=two_body_path, key_path="derived.lambda_pi_pm_fm")
    if not (l_range > 0):
        raise SystemExit("[fail] invalid range scale L (non-positive)")

    # Primary datasets for all-nuclei run.
    ame_src_dirname = "iaea_amdc_ame2020_mass_1_mas20"
    ame_rows = _load_ame2020_rows(root=root, src_dirname=ame_src_dirname)

    radii_src_dirname = "iaea_charge_radii"
    radii_map = _load_iaea_charge_radii_csv(root=root, src_dirname=radii_src_dirname)

    # Freeze a minimal radius-law fallback for nuclei without measured r_rms:
    #   R_uniform = r0 * A^(1/3)  with r0 fixed as median over measured radii (A>=4).
    r0_list: list[float] = []
    for (z, a), rr in radii_map.items():
        if a < 4:
            continue
        r_rms = float(rr["r_rms_fm"])
        r_uniform = math.sqrt(5.0 / 3.0) * r_rms
        r0_list.append(r_uniform / (a ** (1.0 / 3.0)))
    if not r0_list:
        raise SystemExit("[fail] cannot compute r0 median from radii_map (no A>=4 entries)")
    r0_list.sort()
    r0_med = _percentile(r0_list, 50.0)
    r0_p10 = _percentile(r0_list, 10.0)
    r0_p90 = _percentile(r0_list, 90.0)
    r0_sigma_proxy = 0.5 * (r0_p90 - r0_p10)

    # Flags (systematics diagnostics)
    magic_z = {2, 8, 20, 28, 50, 82}
    magic_n = {2, 8, 20, 28, 50, 82, 126}

    def parity_class(z: int, n: int) -> str:
        return ("e" if (z % 2 == 0) else "o") + ("e" if (n % 2 == 0) else "o")

    def c_collective(a: int) -> int:
        return a - 1

    def c_pn_only(z: int, n: int) -> int:
        return z * n

    def c_pairwise_all(a: int) -> int:
        return a * (a - 1) // 2

    subset_keep: set[tuple[int, int]] | None = None
    if subset != "all":
        radii_keys = set(radii_map.keys())
        if subset == "measured_radii":
            subset_keep = radii_keys
            subset_desc = "Nuclei with measured charge radii (r_rms) in IAEA charge_radii.csv."
        else:
            # 'measured_radii_neighborhood': expand measured radii set by (ΔZ,ΔN)<=1.
            subset_keep = set(radii_keys)
            for z0, a0 in radii_keys:
                n0 = a0 - z0
                for dz in (-1, 0, 1):
                    for dn in (-1, 0, 1):
                        z1 = int(z0 + dz)
                        n1 = int(n0 + dn)
                        if z1 <= 0 or n1 < 0:
                            continue
                        a1 = int(z1 + n1)
                        if a1 < 2:
                            continue
                        subset_keep.add((z1, a1))
            subset_desc = "Measured-radii nuclei plus (ΔZ,ΔN)<=1 neighborhood (stable-like gate)."

        print(f"[info] subset={subset} keep_keys={len(subset_keep)} (pre-filter); ame_rows={len(ame_rows)}")

    out_rows: list[dict[str, object]] = []
    # For stats
    ratios_collective: list[float] = []
    ratios_collective_measured: list[float] = []
    ratios_collective_model: list[float] = []
    ratios_by_parity: dict[str, list[float]] = {"ee": [], "eo": [], "oe": [], "oo": []}
    ratios_magic_any: list[float] = []
    ratios_nonmagic: list[float] = []

    for r in ame_rows:
        try:
            z = int(r["Z"])
            a = int(r["A"])
            n = int(r.get("N", a - z))
        except Exception:
            continue
        if a < 2:
            continue
        if subset_keep is not None and (z, a) not in subset_keep:
            continue
        sym = str(r.get("symbol", "")).strip()

        bea_kev = float(r["binding_keV_per_A"])
        bea_sigma_kev = float(r.get("binding_sigma_keV_per_A", 0.0))
        b_obs = (bea_kev / 1000.0) * float(a)
        sigma_b_obs = (bea_sigma_kev / 1000.0) * float(a)

        rr = radii_map.get((z, a))
        if (z, a) == (1, 2):
            # anchor definition
            r_model = r_ref
            sigma_r_model = 0.0
            radius_source = "tail_scale_anchor"
            r_rms = float("nan")
            sigma_r_rms = float("nan")
        elif rr is not None:
            r_rms = float(rr["r_rms_fm"])
            sigma_r_rms = float(rr["sigma_r_rms_fm"])
            r_model = math.sqrt(5.0 / 3.0) * r_rms
            sigma_r_model = math.sqrt(5.0 / 3.0) * sigma_r_rms
            radius_source = "measured_r_rms"
        else:
            r_rms = float("nan")
            sigma_r_rms = float("nan")
            r_model = float(r0_med) * (a ** (1.0 / 3.0))
            sigma_r_model = float(r0_sigma_proxy) * (a ** (1.0 / 3.0))
            radius_source = "radius_law_r0_median"

        j_mev = j_ref_mev * math.exp((r_ref - r_model) / l_range)
        sigma_j_mev = abs(j_mev) * (sigma_r_model / l_range) if (sigma_r_model > 0) else 0.0

        c_req = (b_obs / (2.0 * j_mev)) if (j_mev > 0) else float("nan")

        c0 = c_collective(a)
        c1 = c_pn_only(z, n)
        c2 = c_pairwise_all(a)

        def pred(c_factor: int) -> tuple[float, float, float, float]:
            b_pred = 2.0 * float(c_factor) * float(j_mev)
            sigma_b_pred = 2.0 * float(c_factor) * float(sigma_j_mev)
            ratio = b_pred / b_obs if b_obs > 0 else float("nan")
            delta = b_pred - b_obs
            return b_pred, sigma_b_pred, ratio, delta

        b0, sb0, ratio0, d0 = pred(c0)
        b1, sb1, ratio1, d1 = pred(c1)
        b2, sb2, ratio2, d2 = pred(c2)

        pclass = parity_class(z, n)
        is_magic_any = (z in magic_z) or (n in magic_n)

        # Stats (baseline collective)
        if math.isfinite(ratio0) and ratio0 > 0:
            ratios_collective.append(ratio0)
            if radius_source == "measured_r_rms":
                ratios_collective_measured.append(ratio0)
            else:
                ratios_collective_model.append(ratio0)
            if pclass in ratios_by_parity:
                ratios_by_parity[pclass].append(ratio0)
            if is_magic_any:
                ratios_magic_any.append(ratio0)
            else:
                ratios_nonmagic.append(ratio0)

        out_rows.append(
            {
                "Z": z,
                "N": n,
                "A": a,
                "symbol": sym,
                "parity": pclass,
                "is_magic_Z": (z in magic_z),
                "is_magic_N": (n in magic_n),
                "is_magic_any": is_magic_any,
                "B_obs_MeV": b_obs,
                "sigma_B_obs_MeV": sigma_b_obs,
                "B_over_A_obs_MeV": b_obs / float(a),
                "r_rms_fm": r_rms,
                "sigma_r_rms_fm": sigma_r_rms,
                "R_model_fm": r_model,
                "sigma_R_model_fm": sigma_r_model,
                "radius_source": radius_source,
                "L_fm": l_range,
                "R_ref_fm": r_ref,
                "J_ref_MeV": j_ref_mev,
                "J_E_MeV": j_mev,
                "sigma_J_E_from_R_model_MeV": sigma_j_mev,
                "C_required": c_req,
                "C_collective": c0,
                "B_pred_collective_MeV": b0,
                "sigma_B_pred_collective_from_R_MeV": sb0,
                "ratio_collective": ratio0,
                "Delta_B_collective_MeV": d0,
                "C_pn_only": c1,
                "B_pred_pn_only_MeV": b1,
                "sigma_B_pred_pn_only_from_R_MeV": sb1,
                "ratio_pn_only": ratio1,
                "Delta_B_pn_only_MeV": d1,
                "C_pairwise_all": c2,
                "B_pred_pairwise_all_MeV": b2,
                "sigma_B_pred_pairwise_all_from_R_MeV": sb2,
                "ratio_pairwise_all": ratio2,
                "Delta_B_pairwise_all_MeV": d2,
            }
        )

    # Stable sort (A then Z)
    out_rows.sort(key=lambda x: (int(x["A"]), int(x["Z"]), int(x["N"])))

    # Robust residual diagnostics for all nuclei.
    log_ratio_vals = [
        math.log10(float(r["ratio_collective"]))
        for r in out_rows
        if math.isfinite(float(r["ratio_collective"])) and float(r["ratio_collective"]) > 0.0
    ]
    if not log_ratio_vals:
        raise SystemExit(f"[fail] no usable nuclei selected (subset={subset}).")
    sorted_log_ratio_vals = sorted(log_ratio_vals)
    global_log_ratio_median = _percentile(sorted_log_ratio_vals, 50.0)
    abs_dev = sorted(abs(v - global_log_ratio_median) for v in sorted_log_ratio_vals)
    mad = _percentile(abs_dev, 50.0)
    robust_sigma_log_ratio = max(1.4826 * mad, 1.0e-12)

    for row in out_rows:
        ratio0 = float(row["ratio_collective"])
        if math.isfinite(ratio0) and ratio0 > 0.0:
            log_ratio = math.log10(ratio0)
            z_robust = _safe_div(log_ratio - global_log_ratio_median, robust_sigma_log_ratio)
            row["log10_ratio_collective"] = log_ratio
            row["z_robust_log10_ratio_collective"] = z_robust
            row["is_outlier_robust_abs_z_gt3"] = bool(abs(z_robust) > 3.0)
        else:
            row["log10_ratio_collective"] = float("nan")
            row["z_robust_log10_ratio_collective"] = float("nan")
            row["is_outlier_robust_abs_z_gt3"] = False

    def robust_stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"n": 0.0, "median": float("nan"), "p16": float("nan"), "p84": float("nan")}
        s = sorted(vals)
        return {
            "n": float(len(s)),
            "median": _percentile(s, 50.0),
            "p16": _percentile(s, 16.0),
            "p84": _percentile(s, 84.0),
        }

    stats = {
        "collective_ratio_all": robust_stats(ratios_collective),
        "collective_ratio_measured_radii": robust_stats(ratios_collective_measured),
        "collective_ratio_radius_law": robust_stats(ratios_collective_model),
        "collective_ratio_by_parity": {k: robust_stats(v) for k, v in ratios_by_parity.items()},
        "collective_ratio_magic_any": robust_stats(ratios_magic_any),
        "collective_ratio_nonmagic": robust_stats(ratios_nonmagic),
    }

    # A-band statistics and outliers.
    # Fixed 10 A-bands for roadmap Step 7.17.1 (full all-nuclei summary).
    a_bands = [
        (2, 29),
        (30, 59),
        (60, 89),
        (90, 119),
        (120, 149),
        (150, 179),
        (180, 209),
        (210, 239),
        (240, 269),
        (270, 299),
    ]
    a_band_stats: list[dict[str, float | int | str]] = []
    for a_min, a_max in a_bands:
        rows_band = [r for r in out_rows if a_min <= int(r["A"]) <= a_max and math.isfinite(float(r["ratio_collective"])) and float(r["ratio_collective"]) > 0.0]
        ratios_band = [float(r["ratio_collective"]) for r in rows_band]
        log_band = [float(r["log10_ratio_collective"]) for r in rows_band if math.isfinite(float(r["log10_ratio_collective"]))]
        robust = robust_stats(ratios_band)
        sigma_log = float("nan")
        if len(log_band) >= 2:
            mean_log = sum(log_band) / float(len(log_band))
            sigma_log = math.sqrt(sum((v - mean_log) ** 2 for v in log_band) / float(len(log_band) - 1))
        outlier_count = sum(1 for r in rows_band if bool(r["is_outlier_robust_abs_z_gt3"]))
        a_band_stats.append(
            {
                "A_band": f"{a_min}-{a_max}",
                "n": int(len(rows_band)),
                "median_ratio_collective": float(robust["median"]),
                "p16_ratio_collective": float(robust["p16"]),
                "p84_ratio_collective": float(robust["p84"]),
                "sigma_log10_ratio_collective": sigma_log,
                "outlier_abs_z_gt3_n": int(outlier_count),
                "outlier_abs_z_gt3_frac": _safe_div(float(outlier_count), float(len(rows_band))) if rows_band else float("nan"),
            }
        )

    # CSV
    out_csv = out_dir / f"{out_stem}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(out_rows[0].keys()) if out_rows else [])
        for row in out_rows:
            w.writerow([row[k] for k in out_rows[0].keys()])

    out_a_band_csv = out_dir / f"{out_stem}_a_band_stats.csv"
    with out_a_band_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if a_band_stats:
            headers = list(a_band_stats[0].keys())
            w.writerow(headers)
            for row in a_band_stats:
                w.writerow([row[h] for h in headers])

    # Plot
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    ratios = [float(r["ratio_collective"]) for r in out_rows if math.isfinite(float(r["ratio_collective"])) and float(r["ratio_collective"]) > 0]
    log_ratios = [math.log10(x) for x in ratios]
    log_ratios_meas = [
        math.log10(float(r["ratio_collective"]))
        for r in out_rows
        if r["radius_source"] == "measured_r_rms"
        and math.isfinite(float(r["ratio_collective"]))
        and float(r["ratio_collective"]) > 0
    ]
    log_ratios_model = [
        math.log10(float(r["ratio_collective"]))
        for r in out_rows
        if r["radius_source"] != "measured_r_rms"
        and math.isfinite(float(r["ratio_collective"]))
        and float(r["ratio_collective"]) > 0
    ]

    fig = plt.figure(figsize=(13.6, 8.2), dpi=160)
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.30)

    # (0,0) ratio vs A
    ax0 = fig.add_subplot(gs[0, 0])
    color_map = {"ee": "tab:blue", "eo": "tab:orange", "oe": "tab:green", "oo": "tab:red"}
    for p in ("ee", "eo", "oe", "oo"):
        xs = [int(r["A"]) for r in out_rows if r["parity"] == p and math.isfinite(float(r["ratio_collective"])) and float(r["ratio_collective"]) > 0]
        ys = [float(r["ratio_collective"]) for r in out_rows if r["parity"] == p and math.isfinite(float(r["ratio_collective"])) and float(r["ratio_collective"]) > 0]
        ax0.scatter(xs, ys, s=8, alpha=0.35, color=color_map[p], label=p)
    ax0.axhline(1.0, color="0.2", lw=1.2, ls="--")
    ax0.set_yscale("log")
    ax0.set_xlabel("A")
    ax0.set_ylabel("B_pred/B_obs (baseline; log)")
    ax0.set_title("Baseline residuals vs A (color=parity; ee/eo/oe/oo)")
    ax0.grid(True, which="both", axis="y", ls=":", lw=0.6, alpha=0.6)
    ax0.legend(loc="upper right", fontsize=9)

    # (0,1) C_required / (A-1) vs A
    ax1 = fig.add_subplot(gs[0, 1])
    xs = [int(r["A"]) for r in out_rows if math.isfinite(float(r["C_required"])) and float(r["C_required"]) > 0]
    ys = [
        float(r["C_required"]) / float(max(int(r["A"]) - 1, 1))
        for r in out_rows
        if math.isfinite(float(r["C_required"])) and float(r["C_required"]) > 0
    ]
    ax1.scatter(xs, ys, s=8, alpha=0.35, color="tab:purple")
    ax1.axhline(1.0, color="0.2", lw=1.2, ls="--")
    ax1.set_yscale("log")
    ax1.set_xlabel("A")
    ax1.set_ylabel("C_required/(A-1) (log)")
    ax1.set_title("Implied coherence factor vs A (needs >1 for extra binding)")
    ax1.grid(True, which="both", axis="y", ls=":", lw=0.6, alpha=0.6)

    # (1,0) histogram of log10 ratios (measured vs model radii)
    ax2 = fig.add_subplot(gs[1, 0])
    bins = 50
    ax2.hist(log_ratios_model, bins=bins, alpha=0.55, color="0.6", label="radius law (no measured radii)")
    ax2.hist(log_ratios_meas, bins=bins, alpha=0.70, color="tab:blue", label="measured radii subset")
    ax2.axvline(0.0, color="0.2", lw=1.2, ls="--")
    ax2.set_xlabel("log10(B_pred/B_obs)  (baseline)")
    ax2.set_ylabel("count")
    ax2.set_title("Residual distribution (baseline)")
    ax2.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax2.legend(loc="upper left", fontsize=9)

    # (1,1) group medians (parity + magic)
    ax3 = fig.add_subplot(gs[1, 1])
    groups = ["ee", "eo", "oe", "oo", "magic_any", "nonmagic"]
    medians = [
        stats["collective_ratio_by_parity"]["ee"]["median"],
        stats["collective_ratio_by_parity"]["eo"]["median"],
        stats["collective_ratio_by_parity"]["oe"]["median"],
        stats["collective_ratio_by_parity"]["oo"]["median"],
        stats["collective_ratio_magic_any"]["median"],
        stats["collective_ratio_nonmagic"]["median"],
    ]
    ax3.bar(range(len(groups)), medians, color=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "0.6"], alpha=0.85)
    ax3.axhline(1.0, color="0.2", lw=1.2, ls="--")
    ax3.set_yscale("log")
    ax3.set_xticks(range(len(groups)))
    ax3.set_xticklabels(groups, rotation=20, ha="right")
    ax3.set_ylabel("median(B_pred/B_obs) (log)")
    ax3.set_title("Group medians (baseline): parity & magic flags")
    ax3.grid(True, which="both", axis="y", ls=":", lw=0.6, alpha=0.6)

    suptitle = "Phase 7 / Step 7.13.17.7: AME2020 all-nuclei residuals (Δω→B.E. mapping I/F preview)"
    if subset != "all":
        suptitle = f"Phase 7 / Step 7.18.5 (gate): AME2020 subset residuals ({subset}; Δω→B.E. mapping)"
    fig.suptitle(suptitle, y=1.02)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.12, wspace=0.28, hspace=0.30)

    out_png = out_dir / f"{out_stem}.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # Z-N residual map with overlays (magic numbers / stability / AME envelope proxy).
    zn_rows = [r for r in out_rows if math.isfinite(float(r["log10_ratio_collective"]))]
    z_vals = [int(r["Z"]) for r in zn_rows]
    n_vals = [int(r["N"]) for r in zn_rows]
    c_vals = [float(r["log10_ratio_collective"]) for r in zn_rows]
    c_abs_max = max(abs(v) for v in c_vals) if c_vals else 1.0
    if c_abs_max <= 0.0:
        c_abs_max = 1.0

    fig_zn, ax_zn = plt.subplots(figsize=(11.5, 8.4), dpi=160)
    scatter = ax_zn.scatter(
        z_vals,
        n_vals,
        c=c_vals,
        s=11,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-c_abs_max, vcenter=0.0, vmax=c_abs_max),
        alpha=0.80,
        linewidths=0.0,
    )
    cbar = fig_zn.colorbar(scatter, ax=ax_zn, fraction=0.04, pad=0.02)
    cbar.set_label("log10(B_pred/B_obs) [collective baseline]")

    # Magic overlays.
    for mz in sorted(magic_z):
        ax_zn.axvline(mz, color="0.25", lw=0.8, ls=":", alpha=0.4)
    for mn in sorted(magic_n):
        ax_zn.axhline(mn, color="0.25", lw=0.8, ls=":", alpha=0.35)

    # Beta-stability proxy line.
    max_a = max(int(r["A"]) for r in out_rows)
    a_curve = list(range(2, max_a + 1))
    z_beta = [a / (1.98 + 0.0155 * (a ** (2.0 / 3.0))) for a in a_curve]
    n_beta = [a - z for a, z in zip(a_curve, z_beta)]
    ax_zn.plot(z_beta, n_beta, color="k", lw=1.5, ls="--", alpha=0.9, label="stability line proxy")

    # Dripline proxies from AME envelope.
    z_to_n: dict[int, list[int]] = {}
    for r in out_rows:
        z_to_n.setdefault(int(r["Z"]), []).append(int(r["N"]))
    z_sorted = sorted(z_to_n.keys())
    n_min = [min(z_to_n[z]) for z in z_sorted]
    n_max = [max(z_to_n[z]) for z in z_sorted]
    ax_zn.plot(z_sorted, n_min, color="tab:green", lw=1.2, alpha=0.8, label="AME proton-rich edge")
    ax_zn.plot(z_sorted, n_max, color="tab:purple", lw=1.2, alpha=0.8, label="AME neutron-rich edge")

    # Magic nuclei markers.
    z_magic_pts = [int(r["Z"]) for r in zn_rows if bool(r["is_magic_any"])]
    n_magic_pts = [int(r["N"]) for r in zn_rows if bool(r["is_magic_any"])]
    ax_zn.scatter(
        z_magic_pts,
        n_magic_pts,
        s=18,
        facecolors="none",
        edgecolors="black",
        linewidths=0.45,
        alpha=0.55,
        label="magic N/Z involved",
    )

    ax_zn.set_xlabel("Z")
    ax_zn.set_ylabel("N")
    ax_zn.set_title("AME2020 residual map on Z-N plane (baseline collective mapping)")
    ax_zn.grid(True, ls=":", lw=0.5, alpha=0.5)
    ax_zn.legend(loc="upper left", fontsize=8)

    out_zn_png = out_dir / f"{out_stem}_zn_residual_map.png"
    fig_zn.savefig(out_zn_png, bbox_inches="tight")
    plt.close(fig_zn)

    # Metrics JSON (freeze)
    ame_path = root / "data" / "quantum" / "sources" / ame_src_dirname / "extracted_values.json"
    radii_path = root / "data" / "quantum" / "sources" / radii_src_dirname / "charge_radii.csv"
    out_json = out_dir / f"{out_stem}_metrics.json"
    notes = [
        "This step freezes an all-nuclei residual distribution under the minimal Δω mapping I/F. It is expected to fail in detail; the goal is to identify systematic failure regions (A-scaling, magic/pairing signals) before introducing additional physics.",
        "AME2020 σ(B/A) is extremely small; error bars here are dominated by model systematics and radius proxies, so the main diagnostics are ratio distributions and grouped medians.",
        "The Z-N residual map overlays magic numbers, a beta-stability proxy line, and AME envelope edges (proton-rich/neutron-rich) as operational dripline proxies.",
    ]
    if subset != "all":
        notes[0] = (
            "This is a subset gate run (Step 7.18.5) to validate end-to-end regeneration with fixed outputs "
            "under the frozen Δω→B.E. mapping I/F (not the canonical all-nuclei surface)."
        )

    metrics_payload: dict[str, object] = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "phase": 7,
        "step": metrics_step,
        "extended_step": metrics_extended_step,
        "inputs": {
            "deuteron_binding_metrics": {"path": str(deut_path), "sha256": _sha256(deut_path)},
            "deuteron_two_body_metrics": {"path": str(two_body_path), "sha256": _sha256(two_body_path)},
            "ame2020": {"path": str(ame_path), "sha256": _sha256(ame_path), "rows": len(ame_rows)},
            "iaea_charge_radii_csv": {"path": str(radii_path), "sha256": _sha256(radii_path), "rows": len(radii_map)},
        },
        "frozen_if": {
            "anchor": "R_ref=1/κ_d, J_ref=B_d/2 (deuteron fixed)",
            "range": "L=λπ (frozen)",
            "geometry_proxy": "R_uniform = √(5/3) r_rms (when measured)",
            "radius_fallback": "R_uniform = r0_med A^(1/3) when r_rms is missing; r0_med fixed from measured radii (A>=4)",
            "scaling": "J_E(R)=J_ref exp((R_ref - R)/L); B_pred=2 C J_E(R)",
            "baseline_C": "collective: C=A-1",
            "diagnostic_C": ["pn_only: C=Z*N", "pairwise_all: C=A(A-1)/2"],
            "magic_numbers": {"Z": sorted(list(magic_z)), "N": sorted(list(magic_n))},
        },
        "radius_law": {
            "r0_median_fm": r0_med,
            "r0_p10_fm": r0_p10,
            "r0_p90_fm": r0_p90,
            "r0_sigma_proxy_fm": r0_sigma_proxy,
            "notes": [
                "r0 is computed from measured charge radii using uniform-sphere equivalent radius R=√(5/3) r_rms and r0=R/A^(1/3).",
                "The (p10,p90) spread is used as a systematics proxy for nuclei without measured radii.",
            ],
        },
        "anchor_values": {"B_d_MeV": b_d, "R_ref_fm": r_ref, "J_ref_MeV": j_ref_mev, "L_fm": l_range},
        "robust_residual_baseline": {
            "log10_ratio_median": global_log_ratio_median,
            "robust_sigma_log10_ratio": robust_sigma_log_ratio,
            "outlier_abs_z_gt3_n": int(sum(1 for r in out_rows if bool(r["is_outlier_robust_abs_z_gt3"]))),
            "outlier_abs_z_gt3_frac": _safe_div(
                float(sum(1 for r in out_rows if bool(r["is_outlier_robust_abs_z_gt3"]))),
                float(len(out_rows)),
            ),
        },
        "stats": stats,
        "a_band_stats": a_band_stats,
        "outputs": {
            "png": str(out_png),
            "zn_residual_map_png": str(out_zn_png),
            "csv": str(out_csv),
            "a_band_stats_csv": str(out_a_band_csv),
        },
        "notes": notes,
    }
    if subset != "all":
        metrics_payload["base_step"] = {"step": "7.13.17.7", "extended_step": "7.16.1"}
        metrics_payload["subset"] = {
            "name": subset,
            "description": subset_desc,
            "n_selected": int(len(out_rows)),
            "n_total": int(len(ame_rows)),
        }

    out_json.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ok] wrote:")
    print(f"  {out_png}")
    print(f"  {out_zn_png}")
    print(f"  {out_csv}")
    print(f"  {out_a_band_csv}")
    print(f"  {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
