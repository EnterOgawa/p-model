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


def _load_thresholds(*, json_path: Path) -> dict[str, object]:
    if not json_path.exists():
        return {
            "z_median_abs_max": 3.0,
            "z_delta_median_abs_max": 3.0,
            "a_low_bin": [40, 60],
            "a_high_bin": [250, 300],
            "note": "Default thresholds when falsification pack is missing.",
        }
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        raise SystemExit(f"[fail] invalid thresholds in: {json_path}")
    return thresholds


def _pairing_term(*, a: int, z: int, n: int, a_p: float) -> float:
    if a <= 1:
        return 0.0
    if (z % 2 == 0) and (n % 2 == 0):
        return +a_p / math.sqrt(float(a))
    if (z % 2 == 1) and (n % 2 == 1):
        return -a_p / math.sqrt(float(a))
    return 0.0


def _binding_energy_semf_ref(*, a: int, z: int, n: int) -> float:
    """
    Fixed-coefficient SEMF reference (no fitting in this repository step).
    Coefficients are textbook-scale constants used as a stable comparator.
    """
    if a <= 1:
        return float("nan")
    a_f = float(a)
    a13 = a_f ** (1.0 / 3.0)
    a23 = a13 * a13

    # Reference coefficients (MeV).
    a_v = 15.75
    a_s = 17.80
    a_c = 0.711
    a_a = 23.70
    a_p = 11.18

    pairing = _pairing_term(a=a, z=z, n=n, a_p=a_p)
    return (
        (a_v * a_f)
        - (a_s * a23)
        - (a_c * z * (z - 1) / a13)
        - (a_a * ((a - 2 * z) ** 2) / a_f)
        + pairing
    )


def _model_summary(*, model_id: str, pairs: list[tuple[int, float]], thresholds: dict[str, object], abs_residuals: list[float]) -> dict[str, object]:
    if not pairs:
        return {
            "model_id": model_id,
            "n": 0.0,
            "log10_ratio_stats": _stats([]),
            "sigma_proxy_log10": float("nan"),
            "z_median": float("nan"),
            "delta_median_log10_ratio_high_minus_low": float("nan"),
            "z_delta_median": float("nan"),
            "abs_residual_mev_stats": _stats([]),
            "passes": {"median_within_3sigma": False, "a_trend_within_3sigma": False},
        }

    a_low = thresholds.get("a_low_bin", [40, 60])
    a_high = thresholds.get("a_high_bin", [250, 300])
    low_lo, low_hi = int(a_low[0]), int(a_low[1])
    high_lo, high_hi = int(a_high[0]), int(a_high[1])
    z_median_max = float(thresholds.get("z_median_abs_max", 3.0))
    z_delta_max = float(thresholds.get("z_delta_median_abs_max", 3.0))

    log10r_all: list[float] = []
    log10r_low: list[float] = []
    log10r_high: list[float] = []

    for a, ratio in pairs:
        if not (math.isfinite(ratio) and ratio > 0):
            continue
        val = math.log10(ratio)
        log10r_all.append(val)
        if low_lo <= a < low_hi:
            log10r_low.append(val)
        if high_lo <= a < high_hi:
            log10r_high.append(val)

    s = _stats(log10r_all)
    sigma = _robust_sigma_from_p16_p84(p16=s["p16"], p84=s["p84"])
    z_median = s["median"] / sigma if sigma > 0 and math.isfinite(sigma) else float("nan")

    low_med = _percentile(sorted(log10r_low), 50) if log10r_low else float("nan")
    high_med = _percentile(sorted(log10r_high), 50) if log10r_high else float("nan")
    delta = high_med - low_med if (math.isfinite(low_med) and math.isfinite(high_med)) else float("nan")
    z_delta = delta / sigma if sigma > 0 and math.isfinite(delta) else float("nan")

    return {
        "model_id": model_id,
        "n": float(len(log10r_all)),
        "log10_ratio_stats": s,
        "sigma_proxy_log10": sigma,
        "z_median": z_median,
        "delta_median_log10_ratio_high_minus_low": delta,
        "z_delta_median": z_delta,
        "abs_residual_mev_stats": _stats([v for v in abs_residuals if math.isfinite(v)]),
        "passes": {
            "median_within_3sigma": bool(math.isfinite(z_median) and abs(z_median) <= z_median_max),
            "a_trend_within_3sigma": bool(math.isfinite(z_delta) and abs(z_delta) <= z_delta_max),
        },
    }


def _top_abs_delta(rows: list[dict[str, object]], *, key: str, top_n: int = 12) -> list[dict[str, object]]:
    usable: list[dict[str, object]] = []
    for row in rows:
        try:
            delta = float(row[key])
        except Exception:
            continue
        if math.isfinite(delta):
            usable.append({**row, "_abs_delta": abs(delta)})
    usable.sort(key=lambda r: float(r["_abs_delta"]), reverse=True)
    out: list[dict[str, object]] = []
    for row in usable[:top_n]:
        item = dict(row)
        item.pop("_abs_delta", None)
        out.append(item)
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_csv = out_dir / "nuclear_binding_energy_frequency_mapping_minimal_additional_physics.csv"
    if not in_csv.exists():
        raise SystemExit(
            "[fail] missing minimal additional physics CSV.\n"
            "Run:\n"
            "  python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_minimal_additional_physics.py\n"
            f"Expected: {in_csv}"
        )

    frozen_pack = out_dir / "nuclear_binding_energy_frequency_mapping_falsification_pack.json"
    thresholds = _load_thresholds(json_path=frozen_pack)

    rows_out: list[dict[str, object]] = []
    pairs_pmodel: list[tuple[int, float]] = []
    pairs_yukawa_proxy: list[tuple[int, float]] = []
    pairs_semf_ref: list[tuple[int, float]] = []
    abs_resid_pmodel: list[float] = []
    abs_resid_yukawa: list[float] = []
    abs_resid_semf: list[float] = []

    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                z = int(row["Z"])
                n = int(row["N"])
                a = int(row["A"])
                b_obs = float(row["B_obs_MeV"])
                ratio_pmodel = float(row["ratio_local_spacing_sat"])
                ratio_yukawa = float(row["ratio_collective"])
            except Exception:
                continue
            if a <= 1 or not (math.isfinite(b_obs) and b_obs > 0):
                continue

            b_pred_p = ratio_pmodel * b_obs if math.isfinite(ratio_pmodel) else float("nan")
            b_pred_y = ratio_yukawa * b_obs if math.isfinite(ratio_yukawa) else float("nan")
            b_pred_semf = _binding_energy_semf_ref(a=a, z=z, n=n)
            ratio_semf = (b_pred_semf / b_obs) if (math.isfinite(b_pred_semf) and b_pred_semf > 0 and b_obs > 0) else float("nan")

            delta_p_minus_y = b_pred_p - b_pred_y if (math.isfinite(b_pred_p) and math.isfinite(b_pred_y)) else float("nan")
            delta_p_minus_semf = b_pred_p - b_pred_semf if (math.isfinite(b_pred_p) and math.isfinite(b_pred_semf)) else float("nan")

            if math.isfinite(ratio_pmodel) and ratio_pmodel > 0:
                pairs_pmodel.append((a, ratio_pmodel))
                abs_resid_pmodel.append(abs(b_pred_p - b_obs))
            if math.isfinite(ratio_yukawa) and ratio_yukawa > 0:
                pairs_yukawa_proxy.append((a, ratio_yukawa))
                abs_resid_yukawa.append(abs(b_pred_y - b_obs))
            if math.isfinite(ratio_semf) and ratio_semf > 0:
                pairs_semf_ref.append((a, ratio_semf))
                abs_resid_semf.append(abs(b_pred_semf - b_obs))

            rows_out.append(
                {
                    **row,
                    "B_pred_pmodel_local_sat_MeV": f"{b_pred_p:.6f}" if math.isfinite(b_pred_p) else "",
                    "B_pred_yukawa_proxy_MeV": f"{b_pred_y:.6f}" if math.isfinite(b_pred_y) else "",
                    "B_pred_semf_ref_MeV": f"{b_pred_semf:.6f}" if math.isfinite(b_pred_semf) else "",
                    "ratio_pmodel_local_sat": f"{ratio_pmodel:.6f}" if math.isfinite(ratio_pmodel) else "",
                    "ratio_yukawa_proxy": f"{ratio_yukawa:.6f}" if math.isfinite(ratio_yukawa) else "",
                    "ratio_semf_ref": f"{ratio_semf:.6f}" if math.isfinite(ratio_semf) else "",
                    "Delta_B_P_minus_Yukawa_MeV": f"{delta_p_minus_y:.6f}" if math.isfinite(delta_p_minus_y) else "",
                    "Delta_B_P_minus_SEMF_MeV": f"{delta_p_minus_semf:.6f}" if math.isfinite(delta_p_minus_semf) else "",
                    "Delta_Bfrac_P_minus_Yukawa": f"{(delta_p_minus_y / b_obs):.6f}" if math.isfinite(delta_p_minus_y) else "",
                    "Delta_Bfrac_P_minus_SEMF": f"{(delta_p_minus_semf / b_obs):.6f}" if math.isfinite(delta_p_minus_semf) else "",
                }
            )

    if not rows_out:
        raise SystemExit(f"[fail] parsed 0 usable rows from: {in_csv}")

    summary_p = _model_summary(
        model_id="pmodel_local_spacing_plus_nu_sat (Step 7.13.18)",
        pairs=pairs_pmodel,
        thresholds=thresholds,
        abs_residuals=abs_resid_pmodel,
    )
    summary_y = _model_summary(
        model_id="yukawa_range_proxy_global_R (baseline)",
        pairs=pairs_yukawa_proxy,
        thresholds=thresholds,
        abs_residuals=abs_resid_yukawa,
    )
    summary_s = _model_summary(
        model_id="semi_empirical_mass_formula_fixed_coeff",
        pairs=pairs_semf_ref,
        thresholds=thresholds,
        abs_residuals=abs_resid_semf,
    )
    model_summaries = [summary_p, summary_y, summary_s]

    top_delta_semf = _top_abs_delta(rows_out, key="Delta_B_P_minus_SEMF_MeV", top_n=12)
    top_delta_yukawa = _top_abs_delta(rows_out, key="Delta_B_P_minus_Yukawa_MeV", top_n=12)

    out_csv = out_dir / "nuclear_binding_energy_frequency_mapping_theory_diff.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    # Plot
    import matplotlib.pyplot as plt

    a_vals_p = [a for a, _ in pairs_pmodel]
    r_vals_p = [r for _, r in pairs_pmodel]
    a_vals_y = [a for a, _ in pairs_yukawa_proxy]
    r_vals_y = [r for _, r in pairs_yukawa_proxy]
    a_vals_s = [a for a, _ in pairs_semf_ref]
    r_vals_s = [r for _, r in pairs_semf_ref]

    fig = plt.figure(figsize=(13.2, 8.2))
    gs = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(a_vals_p, r_vals_p, s=8, alpha=0.18, color="tab:green", label="P-model (local+d, nu_sat)")
    ax0.scatter(a_vals_y, r_vals_y, s=8, alpha=0.16, color="tab:purple", label="Yukawa proxy (global R)")
    ax0.scatter(a_vals_s, r_vals_s, s=8, alpha=0.16, color="tab:orange", label="SEMF fixed-coeff")
    ax0.axhline(1.0, color="0.2", lw=1.2, ls="--")
    ax0.set_yscale("log")
    ax0.set_xlabel("A")
    ax0.set_ylabel("B_pred/B_obs")
    ax0.set_title("All-nuclei residual scale by model class")
    ax0.grid(True, which="both", axis="y", ls=":", lw=0.6, alpha=0.6)
    ax0.legend(loc="upper right", fontsize=8)

    ax1 = fig.add_subplot(gs[0, 1])
    labels = ["P-model", "Yukawa\nproxy", "SEMF\nfixed"]
    z_med = [summary_p["z_median"], summary_y["z_median"], summary_s["z_median"]]
    z_del = [summary_p["z_delta_median"], summary_y["z_delta_median"], summary_s["z_delta_median"]]
    x = list(range(len(labels)))
    w = 0.38
    ax1.bar([i - w / 2 for i in x], z_med, width=w, color="tab:blue", alpha=0.85, label="z_median")
    ax1.bar([i + w / 2 for i in x], z_del, width=w, color="tab:orange", alpha=0.85, label="z_Δmedian")
    z_thr = float(thresholds.get("z_median_abs_max", 3.0))
    ax1.axhline(+z_thr, color="0.2", lw=1.2, ls="--")
    ax1.axhline(-z_thr, color="0.2", lw=1.2, ls="--")
    ax1.axhline(0.0, color="0.4", lw=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("z (sigma_proxy units)")
    ax1.set_title("Operational consistency under frozen 3sigma thresholds")
    ax1.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist([math.log10(v) for v in r_vals_p if v > 0], bins=70, alpha=0.52, color="tab:green", label="P-model")
    ax2.hist([math.log10(v) for v in r_vals_y if v > 0], bins=70, alpha=0.45, color="tab:purple", label="Yukawa proxy")
    ax2.hist([math.log10(v) for v in r_vals_s if v > 0], bins=70, alpha=0.45, color="tab:orange", label="SEMF fixed")
    ax2.axvline(0.0, color="0.2", lw=1.2, ls="--")
    ax2.set_xlabel("log10(B_pred/B_obs)")
    ax2.set_ylabel("count")
    ax2.set_title("Residual distributions in common log10 scale")
    ax2.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax2.legend(loc="upper left", fontsize=8)

    ax3 = fig.add_subplot(gs[1, 1])
    a_plot: list[int] = []
    d_semf: list[float] = []
    d_y: list[float] = []
    for row in rows_out:
        try:
            a = int(row["A"])
            dv_semf = float(row["Delta_B_P_minus_SEMF_MeV"])
            dv_y = float(row["Delta_B_P_minus_Yukawa_MeV"])
        except Exception:
            continue
        if math.isfinite(dv_semf) and math.isfinite(dv_y):
            a_plot.append(a)
            d_semf.append(dv_semf)
            d_y.append(dv_y)
    ax3.scatter(a_plot, d_semf, s=8, alpha=0.2, color="tab:red", label="P-model - SEMF")
    ax3.scatter(a_plot, d_y, s=8, alpha=0.2, color="tab:cyan", label="P-model - Yukawa proxy")
    ax3.axhline(0.0, color="0.2", lw=1.2)
    ax3.set_xlabel("A")
    ax3.set_ylabel("Delta B [MeV]")
    ax3.set_title("Differential prediction channel (same AME2020 targets)")
    ax3.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax3.legend(loc="upper left", fontsize=8)

    fig.suptitle("Phase 7 / Step 7.13.17.11: theory-difference extraction (P-model vs standard proxies)", y=1.01)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.08, wspace=0.23, hspace=0.30)

    out_png = out_dir / "nuclear_binding_energy_frequency_mapping_theory_diff.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    out_json = out_dir / "nuclear_binding_energy_frequency_mapping_theory_diff_metrics.json"
    out_json.write_text(
        json.dumps(
            {
                "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.13.17.11",
                "inputs": {
                    "minimal_additional_physics_csv": {"path": str(in_csv), "sha256": _sha256(in_csv), "rows": len(rows_out)},
                    "frozen_falsification_pack": {"path": str(frozen_pack), "sha256": _sha256(frozen_pack)} if frozen_pack.exists() else None,
                },
                "thresholds": thresholds,
                "model_definitions": {
                    "pmodel_local_spacing_plus_nu_sat": "Step 7.13.18 frozen I/F",
                    "yukawa_range_proxy_global_R": "baseline exponential range mapping using global nuclear radius inside exp",
                    "semi_empirical_mass_formula_fixed_coeff": {
                        "a_v_MeV": 15.75,
                        "a_s_MeV": 17.80,
                        "a_c_MeV": 0.711,
                        "a_a_MeV": 23.70,
                        "a_p_MeV": 11.18,
                        "pairing_rule": "delta=+/-a_p/sqrt(A) for even-even/odd-odd; 0 otherwise",
                    },
                },
                "model_summaries": model_summaries,
                "pairwise_differential": {
                    "top_abs_delta_p_minus_semf": top_delta_semf,
                    "top_abs_delta_p_minus_yukawa": top_delta_yukawa,
                },
                "outputs": {"png": str(out_png), "csv": str(out_csv)},
                "notes": [
                    "Yukawa and EFT are represented here by stable operational proxies for cross-dataset auditing.",
                    "This step extracts where differential signal is concentrated before precision-target budgeting in Step 7.13.17.12.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[ok] wrote:")
    print(f"  {out_png}")
    print(f"  {out_csv}")
    print(f"  {out_json}")


if __name__ == "__main__":
    main()
