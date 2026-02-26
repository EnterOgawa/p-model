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


def _load_required_nu(*, csv_path: Path) -> list[dict[str, object]]:
    """
    Reads the frozen AME2020 residual CSV and derives a more physical
    diagnostic:

      ν_eff_required ≡ 2 C_required / A = B_obs / (J_E * A)

    Under a saturation-like picture, ν_eff should remain O(1–10) rather than
    growing strongly with A. Here it is used only as a *diagnostic* of which
    assumptions dominate the mismatch.
    """
    out: list[dict[str, object]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                a = int(str(row.get("A", "")).strip())
                b_obs = float(str(row.get("B_obs_MeV", "")).strip())
                j_e = float(str(row.get("J_E_MeV", "")).strip())
                c_req = float(str(row.get("C_required", "")).strip())
                c_col = float(str(row.get("C_collective", "")).strip())
            except Exception:
                continue

            # Exclude unbound / negative B entries and non-physical rows.

            if a <= 1 or not math.isfinite(b_obs) or b_obs <= 0:
                continue

            # 条件分岐: `not math.isfinite(j_e) or j_e <= 0` を満たす経路を評価する。

            if not math.isfinite(j_e) or j_e <= 0:
                continue

            # 条件分岐: `not math.isfinite(c_req) or c_req <= 0` を満たす経路を評価する。

            if not math.isfinite(c_req) or c_req <= 0:
                continue

            # 条件分岐: `not math.isfinite(c_col) or c_col <= 0` を満たす経路を評価する。

            if not math.isfinite(c_col) or c_col <= 0:
                continue

            nu_eff = b_obs / (j_e * a)
            # 条件分岐: `not math.isfinite(nu_eff)` を満たす経路を評価する。
            if not math.isfinite(nu_eff):
                continue

            radius_source = str(row.get("radius_source", "")).strip()
            is_magic_any = str(row.get("is_magic_any", "")).strip().lower() in ("1", "true", "yes")
            parity = str(row.get("parity", "")).strip()

            out.append(
                {
                    "A": a,
                    "nu_eff_required": float(nu_eff),
                    "c_required_over_collective": float(c_req / c_col),
                    "radius_source": radius_source,
                    "is_magic_any": is_magic_any,
                    "parity": parity,
                }
            )

    return out


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

    rows = _load_required_nu(csv_path=in_csv)
    # 条件分岐: `not rows` を満たす経路を評価する。
    if not rows:
        raise SystemExit(f"[fail] parsed 0 usable rows from: {in_csv}")

    nu_all = [float(r["nu_eff_required"]) for r in rows]
    nu_magic = [float(r["nu_eff_required"]) for r in rows if bool(r["is_magic_any"])]
    nu_nonmagic = [float(r["nu_eff_required"]) for r in rows if not bool(r["is_magic_any"])]

    # Representative A points (median over isotopes at that A).
    a_points = [10, 50, 100, 200]
    per_a: dict[int, dict[str, float]] = {}
    for a0 in a_points:
        vals = [float(r["nu_eff_required"]) for r in rows if int(r["A"]) == a0]
        # 条件分岐: `vals` を満たす経路を評価する。
        if vals:
            per_a[a0] = _stats(vals)

    # Simple bin medians for plotting.

    bins = [2, 6, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300]
    bin_centers: list[float] = []
    bin_medians: list[float] = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        vals = [float(r["nu_eff_required"]) for r in rows if lo <= int(r["A"]) < hi]
        # 条件分岐: `not vals` を満たす経路を評価する。
        if not vals:
            continue

        bin_centers.append(0.5 * (lo + hi))
        bin_medians.append(_stats(vals)["median"])

    # Plot

    import matplotlib.pyplot as plt

    a_measured: list[int] = []
    nu_measured: list[float] = []
    a_fallback: list[int] = []
    nu_fallback: list[float] = []
    for r in rows:
        a = int(r["A"])
        nu = float(r["nu_eff_required"])
        src = str(r["radius_source"])
        # 条件分岐: `src == "measured_r_rms" or src == "tail_scale_anchor"` を満たす経路を評価する。
        if src == "measured_r_rms" or src == "tail_scale_anchor":
            a_measured.append(a)
            nu_measured.append(nu)
        else:
            a_fallback.append(a)
            nu_fallback.append(nu)

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.scatter(a_fallback, nu_fallback, s=10, alpha=0.18, color="0.55", label="radius law subset (r0·A^(1/3))")
    ax.scatter(a_measured, nu_measured, s=14, alpha=0.35, color="tab:blue", label="measured radii subset")

    # 条件分岐: `bin_centers and bin_medians` を満たす経路を評価する。
    if bin_centers and bin_medians:
        ax.plot(bin_centers, bin_medians, color="tab:orange", lw=2.0, label="bin median")

    # Reference guides (diagnostic only; not a claim about nuclear microstructure).

    ax.axhline(2.0, color="0.2", lw=1.2, ls="--", label="baseline (collective C=A−1 ⇒ ν_eff≈2)")
    ax.fill_between([0, 320], [6, 6], [20, 20], color="tab:green", alpha=0.08, label="O(10) saturation-like scale (guide)")

    ax.set_xlim(0, 305)
    ax.set_ylim(0, 105)
    ax.set_xlabel("A")
    ax.set_ylabel("ν_eff_required = B_obs / (J_E·A)")
    ax.set_title("Phase 7 / Step 7.13.17.8: implied coherent bonds per nucleon under frozen J_E(R)")
    ax.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax.legend(loc="upper left", fontsize=9)

    out_png = out_dir / "nuclear_binding_energy_frequency_mapping_physical_interpretation.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    out_json = out_dir / "nuclear_binding_energy_frequency_mapping_physical_interpretation_metrics.json"
    out_json.write_text(
        json.dumps(
            {
                "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.13.17.8",
                "inputs": {"csv": {"path": str(in_csv), "sha256": _sha256(in_csv), "rows": len(rows)}},
                "derived": {
                    "nu_eff_required_all": _stats(nu_all),
                    "nu_eff_required_magic_any": _stats(nu_magic),
                    "nu_eff_required_nonmagic": _stats(nu_nonmagic),
                    "nu_eff_required_by_A": {str(k): v for k, v in sorted(per_a.items())},
                },
                "outputs": {"png": str(out_png)},
                "notes": [
                    "ν_eff_required is a diagnostic derived from the frozen Δω→B.E. mapping I/F: it is the effective number of coherent 'bonds' per nucleon required to match B_obs under J_E(R).",
                    "If ν_eff_required grows strongly with A, the dominant mismatch is not pairing/magic fine-structure but the global geometry/range proxy (using nuclear radius R inside an exponential J_E).",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[ok] wrote:")
    print(f"  {out_png}")
    print(f"  {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()

