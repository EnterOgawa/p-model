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
            # 条件分岐: `not b` を満たす経路を評価する。
            if not b:
                break

            h.update(b)

    return h.hexdigest()


def _percentile(sorted_vals: list[float], p: float) -> float:
    """
    Inclusive percentile with linear interpolation.
    p in [0,100].
    """
    # 条件分岐: `not sorted_vals` を満たす経路を評価する。
    if not sorted_vals:
        raise ValueError("empty")

    # 条件分岐: `p <= 0` を満たす経路を評価する。

    if p <= 0:
        return float(sorted_vals[0])

    # 条件分岐: `p >= 100` を満たす経路を評価する。

    if p >= 100:
        return float(sorted_vals[-1])

    x = (len(sorted_vals) - 1) * (p / 100.0)
    i0 = int(math.floor(x))
    i1 = int(math.ceil(x))
    # 条件分岐: `i0 == i1` を満たす経路を評価する。
    if i0 == i1:
        return float(sorted_vals[i0])

    w = x - i0
    return float((1.0 - w) * sorted_vals[i0] + w * sorted_vals[i1])


def _stats(vals: list[float]) -> dict[str, float]:
    # 条件分岐: `not vals` を満たす経路を評価する。
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
    # 条件分岐: `not (math.isfinite(p16) and math.isfinite(p84))` を満たす経路を評価する。
    if not (math.isfinite(p16) and math.isfinite(p84)):
        return float("nan")

    return 0.5 * (p84 - p16)


def _load_thresholds(*, json_path: Path) -> dict[str, object]:
    # 条件分岐: `not json_path.exists()` を満たす経路を評価する。
    if not json_path.exists():
        return {
            "z_median_abs_max": 3.0,
            "z_delta_median_abs_max": 3.0,
            "a_low_bin": [40, 60],
            "a_high_bin": [250, 300],
            "note": "Default operational thresholds (frozen in Step 7.13.17.10).",
        }

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    thresholds = payload.get("thresholds")
    # 条件分岐: `not isinstance(thresholds, dict)` を満たす経路を評価する。
    if not isinstance(thresholds, dict):
        raise SystemExit(f"[fail] invalid thresholds in: {json_path}")

    return thresholds


def _model_summary(*, model_id: str, pairs: list[tuple[int, float]], thresholds: dict[str, object]) -> dict[str, object]:
    # 条件分岐: `not pairs` を満たす経路を評価する。
    if not pairs:
        return {
            "model_id": model_id,
            "n": 0.0,
            "log10_ratio_stats": _stats([]),
            "sigma_proxy_log10": float("nan"),
            "z_median": float("nan"),
            "a_bin_medians_log10_ratio": {},
            "delta_median_log10_ratio_high_minus_low": float("nan"),
            "z_delta_median": float("nan"),
            "passes": {"median_within_3sigma": False, "a_trend_within_3sigma": False},
        }

    a_low, a_high = thresholds["a_low_bin"], thresholds["a_high_bin"]
    # 条件分岐: `not (isinstance(a_low, list) and len(a_low) == 2 and isinstance(a_high, list)...` を満たす経路を評価する。
    if not (isinstance(a_low, list) and len(a_low) == 2 and isinstance(a_high, list) and len(a_high) == 2):
        raise SystemExit("[fail] invalid a_low_bin/a_high_bin thresholds")

    low_lo, low_hi = int(a_low[0]), int(a_low[1])
    high_lo, high_hi = int(a_high[0]), int(a_high[1])

    log10r_all: list[float] = []
    log10r_low: list[float] = []
    log10r_high: list[float] = []
    for a, r in pairs:
        # 条件分岐: `not (math.isfinite(r) and r > 0)` を満たす経路を評価する。
        if not (math.isfinite(r) and r > 0):
            continue

        v = math.log10(r)
        log10r_all.append(v)
        # 条件分岐: `low_lo <= a < low_hi` を満たす経路を評価する。
        if low_lo <= a < low_hi:
            log10r_low.append(v)

        # 条件分岐: `high_lo <= a < high_hi` を満たす経路を評価する。

        if high_lo <= a < high_hi:
            log10r_high.append(v)

    s = _stats(log10r_all)
    sigma = _robust_sigma_from_p16_p84(p16=s["p16"], p84=s["p84"])
    z_median = s["median"] / sigma if sigma > 0 and math.isfinite(sigma) else float("nan")

    low_med = _percentile(sorted(log10r_low), 50) if log10r_low else float("nan")
    high_med = _percentile(sorted(log10r_high), 50) if log10r_high else float("nan")
    delta = high_med - low_med if (math.isfinite(low_med) and math.isfinite(high_med)) else float("nan")
    z_delta = delta / sigma if sigma > 0 and math.isfinite(delta) else float("nan")

    z_median_max = float(thresholds["z_median_abs_max"])
    z_delta_max = float(thresholds["z_delta_median_abs_max"])
    passes = {
        "median_within_3sigma": math.isfinite(z_median) and abs(z_median) <= z_median_max,
        "a_trend_within_3sigma": math.isfinite(z_delta) and abs(z_delta) <= z_delta_max,
    }

    return {
        "model_id": model_id,
        "n": float(len(log10r_all)),
        "log10_ratio_stats": s,
        "sigma_proxy_log10": sigma,
        "z_median": z_median,
        "a_bin_medians_log10_ratio": {
            "a_low_bin": {"range": [low_lo, low_hi], "n": float(len(log10r_low)), "median": low_med},
            "a_high_bin": {"range": [high_lo, high_hi], "n": float(len(log10r_high)), "median": high_med},
        },
        "delta_median_log10_ratio_high_minus_low": delta,
        "z_delta_median": z_delta,
        "passes": passes,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    diff_csv = out_dir / "nuclear_binding_energy_frequency_mapping_differential_predictions.csv"
    # 条件分岐: `not diff_csv.exists()` を満たす経路を評価する。
    if not diff_csv.exists():
        raise SystemExit(
            "[fail] missing differential predictions CSV.\n"
            "Run:\n"
            "  python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_differential_predictions.py\n"
            f"Expected: {diff_csv}"
        )

    frozen_pack = out_dir / "nuclear_binding_energy_frequency_mapping_falsification_pack.json"
    thresholds = _load_thresholds(json_path=frozen_pack)

    # Minimal additional physics (frozen):
    # - Keep the distance proxy that removes the A-trend (local spacing d inside exp).
    # - Replace the naive coherent-bond count per nucleon ν_base=2(A−1)/A (→2 for large A)
    #   by a saturation cap ν_eff=min(ν_base, ν_sat), with ν_sat=1.5.
    #   This preserves A<=4 exactly (d, 3He, 4He) and suppresses the heavy-A overbinding.
    nu_sat = 1.5

    rows_out: list[dict[str, object]] = []
    pairs_local: list[tuple[int, float]] = []
    pairs_local_sat: list[tuple[int, float]] = []

    with diff_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                a = int(row["A"])
                b_obs = float(row["B_obs_MeV"])
                ratio_local = float(row["ratio_local_spacing"])
                c_base = float(row["C_collective"])
                b_pred_local = float(row["B_pred_local_spacing_MeV"])
            except Exception:
                continue

            # 条件分岐: `a <= 1` を満たす経路を評価する。

            if a <= 1:
                continue

            # 条件分岐: `not (math.isfinite(b_obs) and b_obs > 0)` を満たす経路を評価する。

            if not (math.isfinite(b_obs) and b_obs > 0):
                continue

            # 条件分岐: `not (math.isfinite(ratio_local) and ratio_local > 0)` を満たす経路を評価する。

            if not (math.isfinite(ratio_local) and ratio_local > 0):
                continue

            # 条件分岐: `not (math.isfinite(c_base) and c_base > 0)` を満たす経路を評価する。

            if not (math.isfinite(c_base) and c_base > 0):
                continue

            # 条件分岐: `not (math.isfinite(b_pred_local) and b_pred_local > 0)` を満たす経路を評価する。

            if not (math.isfinite(b_pred_local) and b_pred_local > 0):
                continue

            nu_base = 2.0 * c_base / float(a)
            nu_eff = min(nu_base, nu_sat)
            c_eff = 0.5 * nu_eff * float(a)

            ratio_local_sat = ratio_local * (c_eff / c_base)
            b_pred_local_sat = b_pred_local * (c_eff / c_base)
            # 条件分岐: `not (math.isfinite(ratio_local_sat) and ratio_local_sat > 0)` を満たす経路を評価する。
            if not (math.isfinite(ratio_local_sat) and ratio_local_sat > 0):
                continue

            pairs_local.append((a, ratio_local))
            pairs_local_sat.append((a, ratio_local_sat))

            rows_out.append(
                {
                    **row,
                    "nu_base_from_C_collective": f"{nu_base:.6f}",
                    "nu_sat_frozen": f"{nu_sat:.6f}",
                    "nu_eff_saturated": f"{nu_eff:.6f}",
                    "C_eff_saturated": f"{c_eff:.6f}",
                    "B_pred_local_spacing_sat_MeV": f"{b_pred_local_sat:.6f}",
                    "ratio_local_spacing_sat": f"{ratio_local_sat:.6f}",
                    "Delta_B_local_spacing_sat_MeV": f"{(b_pred_local_sat - b_obs):.6f}",
                }
            )

    # 条件分岐: `not rows_out` を満たす経路を評価する。

    if not rows_out:
        raise SystemExit(f"[fail] parsed 0 usable rows from: {diff_csv}")

    # Summaries (evaluate under frozen operational thresholds).

    m_local = _model_summary(model_id="local_spacing_d_in_exp (Step 7.13.17.9)", pairs=pairs_local, thresholds=thresholds)
    m_local_sat = _model_summary(
        model_id="local_spacing_d_in_exp + nu_saturation (Step 7.13.18)", pairs=pairs_local_sat, thresholds=thresholds
    )

    out_csv = out_dir / "nuclear_binding_energy_frequency_mapping_minimal_additional_physics.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    # Plot (readability-first; do not overfit the narrative).

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"[fail] matplotlib is required to plot: {e}") from e

    a_vals = [a for a, _ in pairs_local]
    r_local = [r for _, r in pairs_local]
    r_sat = [r for _, r in pairs_local_sat]

    fig = plt.figure(figsize=(12.8, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(a_vals, r_local, s=8, alpha=0.20, color="tab:blue", label="baseline: local spacing d")
    ax0.scatter(a_vals, r_sat, s=8, alpha=0.20, color="tab:green", label=f"+ ν saturation (ν_sat={nu_sat:g})")
    ax0.axhline(1.0, color="0.2", lw=1.2, ls="--")
    ax0.set_xlim(0, 305)
    ax0.set_ylim(0.0, max(2.5, _percentile(sorted(r_sat), 99.5)))
    ax0.set_xlabel("A")
    ax0.set_ylabel("B_pred/B_obs")
    ax0.set_title("Minimal additional physics under frozen falsification thresholds")
    ax0.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax0.legend(loc="upper right", fontsize=9)

    ax1 = fig.add_subplot(gs[0, 1])
    labels = ["baseline\n(z_median)", "baseline\n(z_Δmedian)", "sat\n(z_median)", "sat\n(z_Δmedian)"]
    vals = [m_local["z_median"], m_local["z_delta_median"], m_local_sat["z_median"], m_local_sat["z_delta_median"]]
    colors = ["tab:blue", "tab:blue", "tab:green", "tab:green"]
    ax1.bar(range(len(vals)), vals, color=colors, alpha=0.85)
    ax1.axhline(0.0, color="0.2", lw=1.2)
    ax1.axhline(+float(thresholds["z_median_abs_max"]), color="0.2", lw=1.0, ls="--")
    ax1.axhline(-float(thresholds["z_median_abs_max"]), color="0.2", lw=1.0, ls="--")
    ax1.set_xticks(range(len(vals)))
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("z (σ_proxy units)")
    ax1.set_title("Operational z-scores (pass if abs(z)≤3)")
    ax1.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)

    fig.suptitle("Phase 7 / Step 7.13.18: saturation of coherent bonds per nucleon (ν)", y=1.02)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.18, wspace=0.25)

    out_png = out_dir / "nuclear_binding_energy_frequency_mapping_minimal_additional_physics.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    out_json = out_dir / "nuclear_binding_energy_frequency_mapping_minimal_additional_physics_metrics.json"
    out_json.write_text(
        json.dumps(
            {
                "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.13.18",
                "inputs": {
                    "differential_predictions_csv": {"path": str(diff_csv), "sha256": _sha256(diff_csv), "rows": len(rows_out)},
                    "frozen_falsification_pack": {"path": str(frozen_pack), "sha256": _sha256(frozen_pack)} if frozen_pack.exists() else None,
                },
                "thresholds": thresholds,
                "frozen_if": {
                    "distance_proxy": "local spacing d (same as Step 7.13.17.9)",
                    "bond_count_base": "C_collective=A−1 → ν_base=2(A−1)/A",
                    "bond_count_extra_physics": f"ν_eff=min(ν_base, ν_sat) with ν_sat={nu_sat:g} (frozen); C_eff=(ν_eff·A)/2",
                    "note": "This is a minimal additional physics step intended to remove heavy-A overbinding while preserving A<=4 exactly under the same distance proxy.",
                },
                "models": [m_local, m_local_sat],
                "outputs": {"png": str(out_png), "csv": str(out_csv)},
                "notes": [
                    "Thresholds are loaded from the frozen falsification pack (Step 7.13.17.10).",
                    "This step does not modify the thresholds; it tests whether a minimal saturation rule can bring the model within the frozen operational acceptance region.",
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


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()

