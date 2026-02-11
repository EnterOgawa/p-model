from __future__ import annotations

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


def _stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0.0, "median": float("nan"), "p16": float("nan"), "p84": float("nan")}
    vs = sorted(vals)
    return {
        "n": float(len(vs)),
        "median": _percentile(vs, 50),
        "p16": _percentile(vs, 16),
        "p84": _percentile(vs, 84),
    }


def _robust_sigma_from_p16_p84(*, p16: float, p84: float) -> float:
    if not (math.isfinite(p16) and math.isfinite(p84)):
        return float("nan")
    return 0.5 * (p84 - p16)


def _median(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    vs = sorted(vals)
    return _percentile(vs, 50)


def _read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_ratios(*, csv_path: Path) -> dict[str, list[tuple[int, float]]]:
    """
    Returns A-indexed ratios:
      - global: ratio_collective
      - local: ratio_local_spacing (from Step 7.13.17.9)
    """
    out: dict[str, list[tuple[int, float]]] = {"global": [], "local": []}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                a = int(row["A"])
                b_obs = float(row["B_obs_MeV"])
                if not math.isfinite(b_obs) or b_obs <= 0:
                    continue
                rg = float(row["ratio_collective"])
                rl = float(row["ratio_local_spacing"])
            except Exception:
                continue
            if a <= 1:
                continue
            if math.isfinite(rg) and rg > 0:
                out["global"].append((a, rg))
            if math.isfinite(rl) and rl > 0:
                out["local"].append((a, rl))
    return out


def _load_differential_channels(*, out_dir: Path) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    theory_path = out_dir / "nuclear_binding_energy_frequency_mapping_theory_diff_metrics.json"
    quant_path = out_dir / "nuclear_binding_energy_frequency_mapping_differential_quantification_metrics.json"
    if not theory_path.exists() or not quant_path.exists():
        raise SystemExit(
            "[fail] missing Step 7.13.17.11/.12 metrics.\n"
            "Run:\n"
            "  python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_theory_diff.py\n"
            "  python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_differential_quantification.py\n"
            f"Expected: {theory_path} and {quant_path}"
        )

    j_theory = _read_json(theory_path)
    j_quant = _read_json(quant_path)
    gs = j_quant.get("global_stats") if isinstance(j_quant.get("global_stats"), dict) else {}

    channel_specs = [
        ("P-SEMF", "abs_delta_semf_mev", "required_relative_sigma_semf"),
        ("P-Yukawa proxy", "abs_delta_yukawa_mev", "required_relative_sigma_yukawa"),
    ]
    channels: list[dict[str, object]] = []
    for label, key_abs, key_rel in channel_specs:
        abs_stats = gs.get(key_abs) if isinstance(gs.get(key_abs), dict) else {}
        rel_stats = gs.get(key_rel) if isinstance(gs.get(key_rel), dict) else {}
        channels.append(
            {
                "channel_id": label,
                "abs_delta_mev_stats": {
                    "median": _safe_float(abs_stats.get("median")),
                    "p84": _safe_float(abs_stats.get("p84")),
                    "max": _safe_float(abs_stats.get("max")),
                },
                "required_relative_sigma_3sigma": {
                    "median": _safe_float(rel_stats.get("median")),
                    "p84": _safe_float(rel_stats.get("p84")),
                    "max": _safe_float(rel_stats.get("max")),
                },
            }
        )

    return j_theory, j_quant, channels


def _safe_float(value: object) -> float | None:
    try:
        f = float(value)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


def _median_or_none(vals: list[float]) -> float | None:
    if not vals:
        return None
    return _median(vals)


def _load_separation_crosscheck(*, out_dir: Path) -> dict[str, object]:
    metrics_path = out_dir / "nuclear_a_dependence_hf_three_body_separation_energies_metrics.json"
    if not metrics_path.exists():
        raise SystemExit(
            "[fail] missing separation-energy cross-check metrics.\n"
            "Run:\n"
            "  python -B scripts/quantum/nuclear_a_dependence_mean_field.py --step 7.13.15.9\n"
            f"Expected: {metrics_path}"
        )
    j = _read_json(metrics_path)
    diag = j.get("diag") if isinstance(j.get("diag"), dict) else {}
    sn = diag.get("Sn") if isinstance(diag.get("Sn"), dict) else {}
    s2n = diag.get("S2n") if isinstance(diag.get("S2n"), dict) else {}
    gap_sn = diag.get("gap_Sn") if isinstance(diag.get("gap_Sn"), dict) else {}
    gap_s2n = diag.get("gap_S2n") if isinstance(diag.get("gap_S2n"), dict) else {}

    by_magic_sn = gap_sn.get("by_magic_N") if isinstance(gap_sn.get("by_magic_N"), dict) else {}
    by_magic_s2n = gap_s2n.get("by_magic_N") if isinstance(gap_s2n.get("by_magic_N"), dict) else {}

    gap_sn_rms = []
    for item in by_magic_sn.values():
        if not isinstance(item, dict):
            continue
        v = _safe_float(item.get("rms_gap_n_residual_MeV"))
        if v is not None:
            gap_sn_rms.append(v)

    gap_s2n_rms = []
    for item in by_magic_s2n.values():
        if not isinstance(item, dict):
            continue
        v = _safe_float(item.get("rms_gap_2n_residual_MeV"))
        if v is not None:
            gap_s2n_rms.append(v)

    return {
        "step": str(j.get("step", "")),
        "metrics_json": {"path": str(metrics_path), "sha256": _sha256(metrics_path)},
        "sn_rms_total_mev": _safe_float(sn.get("rms_total_MeV")),
        "s2n_rms_total_mev": _safe_float(s2n.get("rms_total_MeV")),
        "gap_sn_rms_median_mev": _median_or_none(gap_sn_rms),
        "gap_s2n_rms_median_mev": _median_or_none(gap_s2n_rms),
        "gap_sn_points": _safe_float(gap_sn.get("n_total")),
        "gap_s2n_points": _safe_float(gap_s2n.get("n_total")),
        "note": "Diagnostic cross-check from separation energies / shell gaps (Step 7.13.15.9).",
    }


def _extract_domain_result(metrics: dict[str, object], *, domain_min_a: int) -> dict[str, object]:
    results = metrics.get("results")
    if not isinstance(results, list):
        return {}
    for item in results:
        if not isinstance(item, dict):
            continue
        if int(item.get("domain_min_A", -1)) == domain_min_a:
            return item
    return {}


def _extract_radii_block(domain_result: dict[str, object], *, block_key: str) -> dict[str, object]:
    block = domain_result.get(block_key) if isinstance(domain_result.get(block_key), dict) else {}
    metrics = block.get("metrics") if isinstance(block.get("metrics"), dict) else block
    fit = block.get("fit") if isinstance(block.get("fit"), dict) else {}
    return {
        "pass": bool(metrics.get("pass", False)),
        "max_abs_resid_sigma_sn": _safe_float(metrics.get("max_abs_resid_sigma_Sn")),
        "max_abs_resid_sigma_sp": _safe_float(metrics.get("max_abs_resid_sigma_Sp")),
        "k_n_fm_per_mev": _safe_float(fit.get("k_n_fm_per_MeV")),
        "k_p_fm_per_mev": _safe_float(fit.get("k_p_fm_per_MeV")),
    }


def _load_radii_crosscheck(*, out_dir: Path) -> dict[str, object]:
    metrics_path = out_dir / "nuclear_a_dependence_hf_three_body_radii_kink_delta2r_radius_magic_offset_even_even_center_magic_only_pairing_deformation_minimal_metrics.json"
    if not metrics_path.exists():
        raise SystemExit(
            "[fail] missing charge-radius kink cross-check metrics.\n"
            "Run:\n"
            "  python -B scripts/quantum/nuclear_a_dependence_mean_field.py --step 7.13.15.49\n"
            f"Expected: {metrics_path}"
        )

    j = _read_json(metrics_path)
    domain_100 = _extract_domain_result(j, domain_min_a=100)
    baseline = _extract_radii_block(domain_100, block_key="def0_baseline")
    pairing = _extract_radii_block(domain_100, block_key="def0_pair_fit")

    return {
        "step": str(j.get("step", "")),
        "metrics_json": {"path": str(metrics_path), "sha256": _sha256(metrics_path)},
        "domain_min_A": 100,
        "strict_threshold_sigma": _safe_float(
            ((j.get("rule_delta2r") if isinstance(j.get("rule_delta2r"), dict) else {}).get("residual_threshold"))
        ),
        "baseline_def0": baseline,
        "pairing_def0_fit": pairing,
        "note": "Charge-radius kink strict check at shell-closure centers (Step 7.13.15.49, A_min=100).",
    }


def _load_selected_nuclei_rows(*, csv_path: Path, keys: list[tuple[int, int]]) -> list[dict[str, object]]:
    keyset = {(z, a) for z, a in keys}
    out: list[dict[str, object]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                z = int(row["Z"])
                a = int(row["A"])
            except Exception:
                continue
            if (z, a) not in keyset:
                continue
            out.append(row)
    # Keep same order as keys if present.
    out_sorted: list[dict[str, object]] = []
    for z, a in keys:
        for r in out:
            if int(r["Z"]) == z and int(r["A"]) == a:
                out_sorted.append(r)
                break
    return out_sorted


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    diff_csv = out_dir / "nuclear_binding_energy_frequency_mapping_differential_predictions.csv"
    if not diff_csv.exists():
        raise SystemExit(
            "[fail] missing differential predictions CSV.\n"
            "Run:\n"
            "  python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_differential_predictions.py\n"
            f"Expected: {diff_csv}"
        )

    ratios = _load_ratios(csv_path=diff_csv)
    if not ratios["global"] or not ratios["local"]:
        raise SystemExit(f"[fail] parsed 0 usable ratios from: {diff_csv}")
    theory_metrics, quant_metrics, channels = _load_differential_channels(out_dir=out_dir)
    sep_crosscheck = _load_separation_crosscheck(out_dir=out_dir)
    radii_crosscheck = _load_radii_crosscheck(out_dir=out_dir)

    thresholds = {
        "z_median_abs_max": 3.0,
        "z_delta_median_abs_max": 3.0,
        "a_low_bin": [40, 60],
        "a_high_bin": [250, 300],
        "channel_decision_sigma_level": 3.0,
        "note": "Operational thresholds for this repository (not universal physics thresholds).",
    }

    def model_summary(model_id: str, pairs: list[tuple[int, float]]) -> dict[str, object]:
        a = [float(x) for x, _ in pairs]
        log10r = [math.log10(r) for _, r in pairs]
        s = _stats(log10r)
        sigma = _robust_sigma_from_p16_p84(p16=s["p16"], p84=s["p84"])
        z_median = s["median"] / sigma if sigma > 0 and math.isfinite(sigma) else float("nan")
        # A single global normalization (scale) that sets the median residual to 0 in log10 space.
        # This is reported for transparency; applying it is a "shape-only" diagnostic, not a
        # claim of predictive power.
        scale_to_median_1 = 10 ** (-s["median"]) if math.isfinite(s["median"]) else float("nan")

        a_lo0, a_lo1 = thresholds["a_low_bin"]
        a_hi0, a_hi1 = thresholds["a_high_bin"]
        lo_vals = [math.log10(r) for A, r in pairs if a_lo0 <= A < a_lo1]
        hi_vals = [math.log10(r) for A, r in pairs if a_hi0 <= A < a_hi1]
        med_lo = _median(lo_vals)
        med_hi = _median(hi_vals)
        delta = med_hi - med_lo if math.isfinite(med_hi) and math.isfinite(med_lo) else float("nan")
        z_delta = delta / sigma if sigma > 0 and math.isfinite(delta) and math.isfinite(sigma) else float("nan")

        return {
            "model_id": model_id,
            "n": float(len(pairs)),
            "log10_ratio_stats": s,
            "sigma_proxy_log10": sigma,
            "z_median": z_median,
            "scale_to_median_1": scale_to_median_1,
            "a_bin_medians_log10_ratio": {
                "a_low_bin": {"range": [a_lo0, a_lo1], "n": float(len(lo_vals)), "median": med_lo},
                "a_high_bin": {"range": [a_hi0, a_hi1], "n": float(len(hi_vals)), "median": med_hi},
            },
            "delta_median_log10_ratio_high_minus_low": delta,
            "z_delta_median": z_delta,
            "passes": {
                "median_within_3sigma": bool(math.isfinite(z_median) and abs(z_median) <= thresholds["z_median_abs_max"]),
                "a_trend_within_3sigma": bool(math.isfinite(z_delta) and abs(z_delta) <= thresholds["z_delta_median_abs_max"]),
            },
        }

    models = [
        model_summary("global_R_in_exp (Step 7.13.17.7)", ratios["global"]),
        model_summary("local_spacing_d_in_exp (Step 7.13.17.9)", ratios["local"]),
    ]

    # Plot: core falsification metrics + independent cross-check panel.
    import matplotlib.pyplot as plt

    labels = [m["model_id"] for m in models]
    z_vals = [float(m["z_median"]) for m in models]
    z_delta_vals = [float(m["z_delta_median"]) for m in models]

    fig, axs = plt.subplots(2, 2, figsize=(15.6, 8.2))
    ax0, ax1, ax2, ax3 = axs.flatten()

    ax0.bar(range(len(labels)), z_vals, color=["tab:purple", "tab:blue"], alpha=0.85)
    ax0.axhline(thresholds["z_median_abs_max"], color="0.2", lw=1.2, ls="--")
    ax0.axhline(-thresholds["z_median_abs_max"], color="0.2", lw=1.2, ls="--")
    ax0.axhline(0.0, color="0.4", lw=0.9)
    ax0.set_xticks(range(len(labels)))
    ax0.set_xticklabels(labels, rotation=15, ha="right")
    ax0.set_ylabel("z_median (log10 ratio)")
    ax0.set_title("Median residual consistency (|z|<=3)")
    ax0.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)

    ax1.bar(range(len(labels)), z_delta_vals, color=["tab:purple", "tab:blue"], alpha=0.85)
    ax1.axhline(thresholds["z_delta_median_abs_max"], color="0.2", lw=1.2, ls="--")
    ax1.axhline(-thresholds["z_delta_median_abs_max"], color="0.2", lw=1.2, ls="--")
    ax1.axhline(0.0, color="0.4", lw=0.9)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("z_Δmedian (log10 ratio)")
    ax1.set_title("A-trend consistency (|z_Δmedian|<=3)")
    ax1.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)

    ch_labels = [str(c.get("channel_id", "")) for c in channels]
    ch_vals = []
    for c in channels:
        rel = c.get("required_relative_sigma_3sigma") if isinstance(c.get("required_relative_sigma_3sigma"), dict) else {}
        med = _safe_float(rel.get("median"))
        ch_vals.append(100.0 * med if med is not None else float("nan"))
    ax2.bar(range(len(ch_labels)), ch_vals, color=["tab:red", "tab:blue"], alpha=0.85)
    ax2.set_xticks(range(len(ch_labels)))
    ax2.set_xticklabels(ch_labels, rotation=15, ha="right")
    ax2.set_ylabel("median σ_req,rel [%]")
    ax2.set_title("Channel precision demand (3σ)")
    ax2.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)

    radius_threshold = _safe_float(radii_crosscheck.get("strict_threshold_sigma"))
    base = radii_crosscheck.get("baseline_def0") if isinstance(radii_crosscheck.get("baseline_def0"), dict) else {}
    pair = radii_crosscheck.get("pairing_def0_fit") if isinstance(radii_crosscheck.get("pairing_def0_fit"), dict) else {}
    radius_labels = ["base Sn", "base Sp", "pair Sn", "pair Sp"]
    radius_vals = [
        _safe_float(base.get("max_abs_resid_sigma_sn")),
        _safe_float(base.get("max_abs_resid_sigma_sp")),
        _safe_float(pair.get("max_abs_resid_sigma_sn")),
        _safe_float(pair.get("max_abs_resid_sigma_sp")),
    ]
    radius_plot_vals = [v if v is not None else float("nan") for v in radius_vals]
    ax3.bar(range(len(radius_labels)), radius_plot_vals, color=["0.7", "0.6", "tab:green", "tab:green"], alpha=0.9)
    if radius_threshold is not None:
        ax3.axhline(radius_threshold, color="0.2", lw=1.2, ls="--")
    ax3.set_xticks(range(len(radius_labels)))
    ax3.set_xticklabels(radius_labels, rotation=15, ha="right")
    ax3.set_ylabel("max abs residual [σ]")
    ax3.set_title("Independent cross-check (radii kink @ A_min=100)")
    ax3.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)

    sn_rms = _safe_float(sep_crosscheck.get("sn_rms_total_mev"))
    s2n_rms = _safe_float(sep_crosscheck.get("s2n_rms_total_mev"))
    gap_sn_med = _safe_float(sep_crosscheck.get("gap_sn_rms_median_mev"))
    gap_s2n_med = _safe_float(sep_crosscheck.get("gap_s2n_rms_median_mev"))
    text_lines = [
        f"Sn RMS: {sn_rms:.2f} MeV" if sn_rms is not None else "Sn RMS: n/a",
        f"S2n RMS: {s2n_rms:.2f} MeV" if s2n_rms is not None else "S2n RMS: n/a",
        f"gap Sn median RMS: {gap_sn_med:.2f} MeV" if gap_sn_med is not None else "gap Sn median RMS: n/a",
        f"gap S2n median RMS: {gap_s2n_med:.2f} MeV" if gap_s2n_med is not None else "gap S2n median RMS: n/a",
    ]
    ax3.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        transform=ax3.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )

    fig.suptitle("Phase 7 / Step 7.13.17.14: falsification pack + independent cross-checks", y=0.98)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.20, wspace=0.30, hspace=0.34)

    out_png = out_dir / "nuclear_binding_energy_frequency_mapping_falsification_pack.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # Representative nuclei table (operational, for reviewer-facing clarity).
    selected_keys = [
        (1, 2),   # d
        (2, 4),   # He-4
        (6, 12),  # C-12
        (8, 16),  # O-16
        (20, 40), # Ca-40
        (26, 56), # Fe-56
        (28, 56), # Ni-56
        (50, 120),# Sn-120
        (82, 208),# Pb-208
        (92, 238),# U-238
    ]
    selected_rows = _load_selected_nuclei_rows(csv_path=diff_csv, keys=selected_keys)

    # Build per-nucleus pass/fail using the shape-only normalization.
    model_by_id = {m["model_id"]: m for m in models}
    m_global = model_by_id["global_R_in_exp (Step 7.13.17.7)"]
    m_local = model_by_id["local_spacing_d_in_exp (Step 7.13.17.9)"]
    sigma_g = float(m_global["sigma_proxy_log10"])
    sigma_l = float(m_local["sigma_proxy_log10"])
    scale_g = float(m_global["scale_to_median_1"])
    scale_l = float(m_local["scale_to_median_1"])

    table_rows: list[dict[str, object]] = []
    for r in selected_rows:
        try:
            z = int(r["Z"])
            n = int(r["N"])
            a = int(r["A"])
            sym = str(r.get("symbol", "")).strip()
            parity = str(r.get("parity", "")).strip()
            b_obs = float(r["B_obs_MeV"])
            rg = float(r["ratio_collective"])
            rl = float(r["ratio_local_spacing"])
        except Exception:
            continue
        if not (math.isfinite(b_obs) and b_obs > 0 and math.isfinite(rg) and rg > 0 and math.isfinite(rl) and rl > 0):
            continue

        logg = math.log10(rg)
        logl = math.log10(rl)
        logg_norm = math.log10(rg * scale_g) if scale_g > 0 else float("nan")
        logl_norm = math.log10(rl * scale_l) if scale_l > 0 else float("nan")
        zg_norm = logg_norm / sigma_g if sigma_g > 0 and math.isfinite(logg_norm) else float("nan")
        zl_norm = logl_norm / sigma_l if sigma_l > 0 and math.isfinite(logl_norm) else float("nan")

        table_rows.append(
            {
                "Z": z,
                "N": n,
                "A": a,
                "symbol": sym,
                "parity": parity,
                "B_obs_MeV": f"{b_obs:.6f}",
                "ratio_global_R": f"{rg:.6f}",
                "ratio_local_d": f"{rl:.6f}",
                "log10_ratio_global_R": f"{logg:.6f}",
                "log10_ratio_local_d": f"{logl:.6f}",
                "ratio_global_R_norm": f"{(rg * scale_g):.6f}" if scale_g > 0 else "",
                "ratio_local_d_norm": f"{(rl * scale_l):.6f}" if scale_l > 0 else "",
                "z_norm_global_R": f"{zg_norm:.3f}" if math.isfinite(zg_norm) else "",
                "z_norm_local_d": f"{zl_norm:.3f}" if math.isfinite(zl_norm) else "",
                "pass_norm_global_R_abs_z_le_3": "1" if math.isfinite(zg_norm) and abs(zg_norm) <= thresholds["z_median_abs_max"] else "0",
                "pass_norm_local_d_abs_z_le_3": "1" if math.isfinite(zl_norm) and abs(zl_norm) <= thresholds["z_median_abs_max"] else "0",
            }
        )

    out_table = out_dir / "nuclear_binding_energy_frequency_mapping_falsification_table.csv"
    if table_rows:
        with out_table.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
            writer.writeheader()
            writer.writerows(table_rows)

    out_json = out_dir / "nuclear_binding_energy_frequency_mapping_falsification_pack.json"
    out_json.write_text(
        json.dumps(
            {
                "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "thresholds": thresholds,
                "inputs": {
                    "differential_predictions_csv": {"path": str(diff_csv), "sha256": _sha256(diff_csv)},
                    "theory_diff_metrics_json": {
                        "path": str(out_dir / "nuclear_binding_energy_frequency_mapping_theory_diff_metrics.json"),
                        "sha256": _sha256(out_dir / "nuclear_binding_energy_frequency_mapping_theory_diff_metrics.json"),
                    },
                    "differential_quantification_metrics_json": {
                        "path": str(out_dir / "nuclear_binding_energy_frequency_mapping_differential_quantification_metrics.json"),
                        "sha256": _sha256(out_dir / "nuclear_binding_energy_frequency_mapping_differential_quantification_metrics.json"),
                    },
                    "separation_energies_metrics_json": sep_crosscheck.get("metrics_json"),
                    "charge_radius_kink_metrics_json": radii_crosscheck.get("metrics_json"),
                },
                "models": models,
                "differential_channels": channels,
                "independent_cross_checks": {
                    "separation_energies": sep_crosscheck,
                    "charge_radius_kink": radii_crosscheck,
                    "gate_status": {
                        "radius_strict_pass_A100_pairing": bool(
                            (
                                radii_crosscheck.get("pairing_def0_fit")
                                if isinstance(radii_crosscheck.get("pairing_def0_fit"), dict)
                                else {}
                            ).get("pass", False)
                        ),
                        "separation_gap_available": bool(
                            (_safe_float(sep_crosscheck.get("gap_sn_points")) or 0) > 0
                            and (_safe_float(sep_crosscheck.get("gap_s2n_points")) or 0) > 0
                        ),
                    },
                },
                "representative_table": {
                    "note": "Representative nuclei table is evaluated in a 'shape-only' mode (median-normalized).",
                    "selected_keys": [{"Z": z, "A": a} for z, a in selected_keys],
                    "csv": str(out_table) if table_rows else None,
                },
                "conditions": [
                    {
                        "id": "median_residual_falsified_if_abs_z_median_gt_3",
                        "statement": "log10(B_pred/B_obs) の中央値が 0 から 3σ_proxy を超えてずれるなら、このI/Fは棄却（このσ_proxy定義では）",
                    },
                    {
                        "id": "a_trend_falsified_if_abs_z_delta_median_gt_3",
                        "statement": "A-bin（固定）での log10(B_pred/B_obs) の中央値差（high−low）が 3σ_proxy を超えるなら、A-trend を含むI/Fは棄却（手続き依存の入口）",
                    },
                    {
                        "id": "representative_nuclei_shape_falsified_if_many_fail",
                        "statement": "shape-only（median-normalized）でも代表核種の abs(z) が多数で 3 を超えるなら、距離I/F（R vs d）だけでは説明不十分（追加物理が必要）",
                    },
                    {
                        "id": "channel_precision_gate_for_3sigma_decision",
                        "statement": "差分チャネル（P-SEMF / P-Yukawa）の3σ決着は、対象核種で総相対不確かさ σ_rel(total) <= σ_req,rel を満たす場合のみ採用する",
                    },
                    {
                        "id": "channel_gate_requires_independent_crosschecks",
                        "statement": "差分チャネルの3σ判定は、独立cross-check（separation energies と charge-radius kink）の固定指標を併記し、半径kink strict（A_min=100, pairing凍結）と整合する場合に限って採用する",
                    },
                ],
                "outputs": {"png": str(out_png)},
                "notes": [
                    "This pack is intended to prevent 'fit-escape': thresholds are frozen before adding extra physics parameters.",
                    "σ_proxy is a robust internal proxy derived from (p16,p84) of the log10 ratio distribution; it is not an experimental uncertainty.",
                    "P-SEMF / P-Yukawa differential channels are loaded from Step 7.13.17.11/.12 metrics and tracked in the same pack.",
                    "Independent cross-check bundle is linked from Step 7.13.15.9 (separation) and Step 7.13.15.49 (charge-radius kink).",
                ],
                "upstream": {
                    "theory_diff": theory_metrics.get("step"),
                    "differential_quantification": quant_metrics.get("step"),
                    "separation_energies": sep_crosscheck.get("step"),
                    "charge_radius_kink": radii_crosscheck.get("step"),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[ok] wrote:")
    print(f"  {out_png}")
    if table_rows:
        print(f"  {out_table}")
    print(f"  {out_json}")


if __name__ == "__main__":
    main()
