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


def _is_truthy(v: str) -> bool:
    s = v.strip().lower()
    return s in ("1", "true", "yes", "y")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_csv = out_dir / "nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei.csv"
    # 条件分岐: `not in_csv.exists()` を満たす経路を評価する。
    if not in_csv.exists():
        raise SystemExit(
            "[fail] missing frozen all-nuclei residual CSV.\n"
            "Run:\n"
            "  python -B scripts/quantum/nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei.py\n"
            f"Expected: {in_csv}"
        )

    # Local spacing proxy derived from uniform-sphere density:
    #   V/A = (4π/3) r0^3  ⇒  ρ = 3/(4π r0^3)
    #   spacing scale d ≡ ρ^{-1/3} = (4π/3)^{1/3} r0

    spacing_factor = (4.0 * math.pi / 3.0) ** (1.0 / 3.0)

    rows_out: list[dict[str, object]] = []
    ratio_global_all: list[float] = []
    ratio_local_all: list[float] = []
    ratio_local_measured: list[float] = []
    ratio_local_radius_law: list[float] = []
    ratio_local_by_parity: dict[str, list[float]] = {k: [] for k in ("ee", "eo", "oe", "oo")}
    ratio_local_magic_any: list[float] = []
    ratio_local_nonmagic: list[float] = []

    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                a = int(row["A"])
                b_obs = float(row["B_obs_MeV"])
                # 条件分岐: `not math.isfinite(b_obs) or b_obs <= 0` を満たす経路を評価する。
                if not math.isfinite(b_obs) or b_obs <= 0:
                    continue

                ratio_global = float(row["ratio_collective"])
                # 条件分岐: `not math.isfinite(ratio_global) or ratio_global <= 0` を満たす経路を評価する。
                if not math.isfinite(ratio_global) or ratio_global <= 0:
                    continue

                r_model = float(row["R_model_fm"])
                l_range = float(row["L_fm"])
                r_ref = float(row["R_ref_fm"])
                j_ref = float(row["J_ref_MeV"])
                c_collective = float(row["C_collective"])
                # 条件分岐: `not all(math.isfinite(x) for x in (r_model, l_range, r_ref, j_ref, c_collecti...` を満たす経路を評価する。
                if not all(math.isfinite(x) for x in (r_model, l_range, r_ref, j_ref, c_collective)):
                    continue

                # 条件分岐: `a <= 1 or l_range <= 0 or j_ref <= 0 or c_collective <= 0` を満たす経路を評価する。

                if a <= 1 or l_range <= 0 or j_ref <= 0 or c_collective <= 0:
                    continue

                a13 = a ** (1.0 / 3.0)
                r0 = r_model / a13
                d = spacing_factor * r0
                j_e_local = j_ref * math.exp((r_ref - d) / l_range)
                b_pred_local = 2.0 * c_collective * j_e_local
                ratio_local = b_pred_local / b_obs
                # 条件分岐: `not math.isfinite(ratio_local) or ratio_local <= 0` を満たす経路を評価する。
                if not math.isfinite(ratio_local) or ratio_local <= 0:
                    continue
            except Exception:
                continue

            radius_source = str(row.get("radius_source", "")).strip()
            is_magic_any = _is_truthy(str(row.get("is_magic_any", "")))
            parity = str(row.get("parity", "")).strip()

            ratio_global_all.append(ratio_global)
            ratio_local_all.append(ratio_local)
            # 条件分岐: `radius_source == "measured_r_rms" or radius_source == "tail_scale_anchor"` を満たす経路を評価する。
            if radius_source == "measured_r_rms" or radius_source == "tail_scale_anchor":
                ratio_local_measured.append(ratio_local)
            else:
                ratio_local_radius_law.append(ratio_local)

            # 条件分岐: `parity in ratio_local_by_parity` を満たす経路を評価する。

            if parity in ratio_local_by_parity:
                ratio_local_by_parity[parity].append(ratio_local)

            # 条件分岐: `is_magic_any` を満たす経路を評価する。

            if is_magic_any:
                ratio_local_magic_any.append(ratio_local)
            else:
                ratio_local_nonmagic.append(ratio_local)

            rows_out.append(
                {
                    **row,
                    "r0_from_R_model_fm": f"{r0:.6f}",
                    "d_spacing_fm": f"{d:.6f}",
                    "J_E_local_spacing_MeV": f"{j_e_local:.6f}",
                    "B_pred_local_spacing_MeV": f"{b_pred_local:.6f}",
                    "ratio_local_spacing": f"{ratio_local:.6f}",
                    "Delta_B_local_spacing_MeV": f"{(b_pred_local - b_obs):.6f}",
                }
            )

    # 条件分岐: `not ratio_local_all` を満たす経路を評価する。

    if not ratio_local_all:
        raise SystemExit(f"[fail] parsed 0 usable rows from: {in_csv}")

    out_csv = out_dir / "nuclear_binding_energy_frequency_mapping_differential_predictions.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows_out[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    stats = {
        "ratio_global_collective": _stats(ratio_global_all),
        "ratio_local_spacing_collective": _stats(ratio_local_all),
        "ratio_local_spacing_measured_radii": _stats(ratio_local_measured),
        "ratio_local_spacing_radius_law": _stats(ratio_local_radius_law),
        "ratio_local_spacing_by_parity": {k: _stats(v) for k, v in ratio_local_by_parity.items()},
        "ratio_local_spacing_magic_any": _stats(ratio_local_magic_any),
        "ratio_local_spacing_nonmagic": _stats(ratio_local_nonmagic),
    }

    # Plot
    import matplotlib.pyplot as plt

    a_vals: list[int] = []
    rg_vals: list[float] = []
    rl_vals: list[float] = []
    for r in rows_out:
        try:
            a_vals.append(int(r["A"]))
            rg_vals.append(float(r["ratio_collective"]))
            rl_vals.append(float(r["ratio_local_spacing"]))
        except Exception:
            continue

    fig = plt.figure(figsize=(12.8, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])

    # (0,0) global ratio vs A (log scale; baseline underpredicts)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(a_vals, rg_vals, s=8, alpha=0.18, color="tab:purple")
    ax0.axhline(1.0, color="0.2", lw=1.2, ls="--")
    ax0.set_yscale("log")
    ax0.set_xlabel("A")
    ax0.set_ylabel("B_pred/B_obs (global R; baseline)")
    ax0.set_title("Global distance proxy (R in exp): large A-trend mismatch")
    ax0.grid(True, which="both", axis="y", ls=":", lw=0.6, alpha=0.6)

    # (0,1) local spacing ratio vs A (linear-ish around O(1))
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.scatter(a_vals, rl_vals, s=8, alpha=0.18, color="tab:blue")
    ax1.axhline(1.0, color="0.2", lw=1.2, ls="--")
    ax1.set_xlim(0, 305)
    ax1.set_ylim(0.0, max(3.0, _percentile(sorted(rl_vals), 99.5)))
    ax1.set_xlabel("A")
    ax1.set_ylabel("B_pred/B_obs (local spacing d)")
    ax1.set_title("Local spacing proxy (d=ρ^{-1/3}): A-trend largely removed")
    ax1.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)

    # (1,0) distributions (log10)
    ax2 = fig.add_subplot(gs[1, 0])
    lg = [math.log10(x) for x in rg_vals if x > 0]
    ll = [math.log10(x) for x in rl_vals if x > 0]
    ax2.hist(lg, bins=60, alpha=0.55, color="tab:purple", label="global R (log10 ratio)")
    ax2.hist(ll, bins=60, alpha=0.55, color="tab:blue", label="local spacing d (log10 ratio)")
    ax2.axvline(0.0, color="0.2", lw=1.2, ls="--")
    ax2.set_xlabel("log10(B_pred/B_obs)")
    ax2.set_ylabel("count")
    ax2.set_title("Residual distributions (same frozen anchor/range; only distance proxy differs)")
    ax2.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax2.legend(loc="upper left", fontsize=9)

    # (1,1) group medians (parity + magic) for local mapping
    ax3 = fig.add_subplot(gs[1, 1])
    groups = ["ee", "eo", "oe", "oo", "magic_any", "nonmagic"]
    medians = [
        stats["ratio_local_spacing_by_parity"]["ee"]["median"],
        stats["ratio_local_spacing_by_parity"]["eo"]["median"],
        stats["ratio_local_spacing_by_parity"]["oe"]["median"],
        stats["ratio_local_spacing_by_parity"]["oo"]["median"],
        stats["ratio_local_spacing_magic_any"]["median"],
        stats["ratio_local_spacing_nonmagic"]["median"],
    ]
    ax3.bar(range(len(groups)), medians, color=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "0.6"], alpha=0.85)
    ax3.axhline(1.0, color="0.2", lw=1.2, ls="--")
    ax3.set_xticks(range(len(groups)))
    ax3.set_xticklabels(groups, rotation=20, ha="right")
    ax3.set_ylabel("median(B_pred/B_obs)")
    ax3.set_title("Local mapping group medians (parity & magic flags)")
    ax3.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)

    fig.suptitle("Phase 7 / Step 7.13.17.9: differential predictions (distance proxy audit)", y=1.02)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.12, wspace=0.25, hspace=0.35)

    out_png = out_dir / "nuclear_binding_energy_frequency_mapping_differential_predictions.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    out_json = out_dir / "nuclear_binding_energy_frequency_mapping_differential_predictions_metrics.json"
    out_json.write_text(
        json.dumps(
            {
                "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.13.17.9",
                "inputs": {"all_nuclei_csv": {"path": str(in_csv), "sha256": _sha256(in_csv), "rows": len(rows_out)}},
                "frozen_if": {
                    "anchor_and_range": "same as Step 7.13.17.7 (R_ref=1/κ_d, J_ref=B_d/2, L=λπ, C=A−1)",
                    "global_distance_proxy": "R_model (≈ nuclear radius) is used inside exp((R_ref−R)/L)",
                    "local_distance_proxy": "d = ρ^{-1/3} = (4π/3)^{1/3} · r0, with r0=R_model/A^(1/3)",
                },
                "stats": stats,
                "outputs": {"png": str(out_png), "csv": str(out_csv)},
                "notes": [
                    "This step does not add fit freedom; it audits which 'distance' interpretation dominates the all-nuclei mismatch.",
                    "If a local spacing proxy largely removes the A-trend, the remaining structure (pairing/magic) can be treated as secondary corrections rather than the dominant failure mode.",
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

