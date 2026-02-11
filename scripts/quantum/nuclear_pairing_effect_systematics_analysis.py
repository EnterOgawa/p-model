from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


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
    if not values:
        return float("nan")
    return math.sqrt(sum(v * v for v in values) / float(len(values)))


def _safe_median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(median(values))


def _solve_2x2(
    *,
    s11: float,
    s22: float,
    s12: float,
    t1: float,
    t2: float,
    ridge: float = 1.0e-8,
) -> tuple[float, float]:
    a11 = s11 + ridge
    a22 = s22 + ridge
    det = a11 * a22 - s12 * s12
    if det == 0.0:
        raise SystemExit("[fail] singular 2x2 system in pairing fit")
    k_n = (t1 * a22 - t2 * s12) / det
    k_p = (a11 * t2 - s12 * t1) / det
    return float(k_n), float(k_p)


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


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_csv = out_dir / "nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei.csv"
    if not in_csv.exists():
        raise SystemExit(f"[fail] missing required input: {in_csv}")

    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            z = int(row["Z"])
            n = int(row["N"])
            a = int(row["A"])
            b_obs = float(row["B_obs_MeV"])
            b_pred = float(row["B_pred_collective_MeV"])
            if not (math.isfinite(b_obs) and math.isfinite(b_pred)):
                continue
            by_key[(z, n)] = {
                "Z": z,
                "N": n,
                "A": a,
                "symbol": str(row.get("symbol", "")),
                "parity": str(row.get("parity", "")),
                "is_magic_any": str(row.get("is_magic_any", "False")).lower() == "true",
                "B_obs_MeV": b_obs,
                "B_pred_collective_MeV": b_pred,
            }

    if not by_key:
        raise SystemExit(f"[fail] no usable rows loaded: {in_csv}")

    delta_n_map: dict[tuple[int, int], float] = {}
    delta_p_map: dict[tuple[int, int], float] = {}

    for (z, n), center in by_key.items():
        up = by_key.get((z, n + 1))
        dn = by_key.get((z, n - 1))
        if up is not None and dn is not None:
            b_plus = float(up["B_obs_MeV"])
            b0 = float(center["B_obs_MeV"])
            b_minus = float(dn["B_obs_MeV"])
            val = abs(((-1) ** n) * (b_plus - 2.0 * b0 + b_minus) / 2.0)
            if math.isfinite(val):
                delta_n_map[(z, n)] = float(val)

        zp = by_key.get((z + 1, n))
        zm = by_key.get((z - 1, n))
        if zp is not None and zm is not None:
            b_plus = float(zp["B_obs_MeV"])
            b0 = float(center["B_obs_MeV"])
            b_minus = float(zm["B_obs_MeV"])
            val = abs(((-1) ** z) * (b_plus - 2.0 * b0 + b_minus) / 2.0)
            if math.isfinite(val):
                delta_p_map[(z, n)] = float(val)

    rows: list[dict[str, Any]] = []
    fit_rows: list[tuple[float, float, float, int]] = []  # (res_before, delta_n, delta_p, A)
    for (z, n), row in sorted(by_key.items(), key=lambda kv: (int(kv[1]["A"]), int(kv[0][0]), int(kv[0][1]))):
        b_obs = float(row["B_obs_MeV"])
        b_pred = float(row["B_pred_collective_MeV"])
        delta_n = delta_n_map.get((z, n), float("nan"))
        delta_p = delta_p_map.get((z, n), float("nan"))
        resid_before = float(b_pred - b_obs)
        if math.isfinite(delta_n) and math.isfinite(delta_p):
            fit_rows.append((resid_before, float(delta_n), float(delta_p), int(row["A"])))
        rows.append(
            {
                "Z": z,
                "N": n,
                "A": int(row["A"]),
                "symbol": str(row["symbol"]),
                "parity": str(row["parity"]),
                "is_magic_any": bool(row["is_magic_any"]),
                "B_obs_MeV": b_obs,
                "B_pred_before_MeV": b_pred,
                "resid_before_MeV": resid_before,
                "delta_n_3pt_MeV": float(delta_n) if math.isfinite(delta_n) else float("nan"),
                "delta_p_3pt_MeV": float(delta_p) if math.isfinite(delta_p) else float("nan"),
            }
        )

    if len(fit_rows) < 10:
        raise SystemExit(f"[fail] insufficient fit rows for pairing correction: n={len(fit_rows)}")

    resid_by_a: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        resid_by_a[int(row["A"])].append(float(row["resid_before_MeV"]))
    resid_median_by_a = {a: _safe_median(vs) for a, vs in resid_by_a.items()}

    s11 = 0.0
    s22 = 0.0
    s12 = 0.0
    t1 = 0.0
    t2 = 0.0
    for resid_before, delta_n, delta_p, a in fit_rows:
        y = float(resid_before - resid_median_by_a.get(a, 0.0))
        s11 += delta_n * delta_n
        s22 += delta_p * delta_p
        s12 += delta_n * delta_p
        t1 += delta_n * y
        t2 += delta_p * y
    k_n, k_p = _solve_2x2(s11=s11, s22=s22, s12=s12, t1=t1, t2=t2)

    for row in rows:
        dn = float(row["delta_n_3pt_MeV"])
        dp = float(row["delta_p_3pt_MeV"])
        dn_eff = dn if math.isfinite(dn) else 0.0
        dp_eff = dp if math.isfinite(dp) else 0.0
        correction = float(k_n * dn_eff + k_p * dp_eff)
        b_pred_before = float(row["B_pred_before_MeV"])
        b_pred_after = float(b_pred_before - correction)
        b_obs = float(row["B_obs_MeV"])
        resid_after = float(b_pred_after - b_obs)
        row["pairing_correction_MeV"] = correction
        row["B_pred_after_MeV"] = b_pred_after
        row["resid_after_MeV"] = resid_after
        row["abs_resid_before_MeV"] = abs(float(row["resid_before_MeV"]))
        row["abs_resid_after_MeV"] = abs(resid_after)

    def summarize_group(rows_in: list[dict[str, Any]], key_name: str, key_val: str) -> dict[str, Any]:
        rb = [float(r["resid_before_MeV"]) for r in rows_in if math.isfinite(float(r["resid_before_MeV"]))]
        ra = [float(r["resid_after_MeV"]) for r in rows_in if math.isfinite(float(r["resid_after_MeV"]))]
        ab = [abs(v) for v in rb]
        aa = [abs(v) for v in ra]
        return {
            "group_type": key_name,
            "group": key_val,
            "n": len(rows_in),
            "rms_resid_before_MeV": _rms(rb),
            "rms_resid_after_MeV": _rms(ra),
            "median_abs_resid_before_MeV": _safe_median(ab),
            "median_abs_resid_after_MeV": _safe_median(aa),
            "delta_rms_after_minus_before_MeV": _rms(ra) - _rms(rb),
        }

    grouped_parity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_magic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_parity[str(row["parity"])].append(row)
        grouped_magic["magic_any" if bool(row["is_magic_any"]) else "nonmagic"].append(row)

    summary_rows: list[dict[str, Any]] = []
    summary_rows.append(summarize_group(rows, "all", "all"))
    for key in ["ee", "eo", "oe", "oo"]:
        summary_rows.append(summarize_group(grouped_parity.get(key, []), "parity", key))
    summary_rows.append(summarize_group(grouped_magic.get("magic_any", []), "magic", "magic_any"))
    summary_rows.append(summarize_group(grouped_magic.get("nonmagic", []), "magic", "nonmagic"))

    # Plot.
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), dpi=160)
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]

    dn_x = [int(r["N"]) for r in rows if math.isfinite(float(r["delta_n_3pt_MeV"]))]
    dn_y = [float(r["delta_n_3pt_MeV"]) for r in rows if math.isfinite(float(r["delta_n_3pt_MeV"]))]
    dp_x = [int(r["Z"]) for r in rows if math.isfinite(float(r["delta_p_3pt_MeV"]))]
    dp_y = [float(r["delta_p_3pt_MeV"]) for r in rows if math.isfinite(float(r["delta_p_3pt_MeV"]))]
    ax00.scatter(dn_x, dn_y, s=8.0, alpha=0.28, color="#1f77b4")
    ax00.set_xlabel("N")
    ax00.set_ylabel("Δ_n (3-point) [MeV]")
    ax00.set_title("Neutron pairing indicator vs N")
    ax00.grid(True, ls=":", lw=0.6, alpha=0.6)

    ax01.scatter(dp_x, dp_y, s=8.0, alpha=0.28, color="#ff7f0e")
    ax01.set_xlabel("Z")
    ax01.set_ylabel("Δ_p (3-point) [MeV]")
    ax01.set_title("Proton pairing indicator vs Z")
    ax01.grid(True, ls=":", lw=0.6, alpha=0.6)

    parity_labels = ["ee", "eo", "oe", "oo"]
    pb = []
    pa = []
    for label in parity_labels:
        grp = summarize_group(grouped_parity.get(label, []), "parity", label)
        pb.append(float(grp["rms_resid_before_MeV"]))
        pa.append(float(grp["rms_resid_after_MeV"]))
    x = list(range(len(parity_labels)))
    ax10.bar([i - 0.18 for i in x], pb, width=0.36, label="before pairing corr.", alpha=0.85)
    ax10.bar([i + 0.18 for i in x], pa, width=0.36, label="after pairing corr.", alpha=0.85)
    ax10.set_xticks(x)
    ax10.set_xticklabels(parity_labels)
    ax10.set_xlabel("parity group")
    ax10.set_ylabel("RMS residual [MeV]")
    ax10.set_title("Residual comparison by parity")
    ax10.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax10.legend(loc="best", fontsize=8)

    magic_labels = ["magic_any", "nonmagic"]
    mb = []
    ma = []
    for label in magic_labels:
        grp = summarize_group(grouped_magic.get(label, []), "magic", label)
        mb.append(float(grp["rms_resid_before_MeV"]))
        ma.append(float(grp["rms_resid_after_MeV"]))
    x2 = list(range(len(magic_labels)))
    ax11.bar([i - 0.18 for i in x2], mb, width=0.36, label="before pairing corr.", alpha=0.85)
    ax11.bar([i + 0.18 for i in x2], ma, width=0.36, label="after pairing corr.", alpha=0.85)
    ax11.set_xticks(x2)
    ax11.set_xticklabels(magic_labels)
    ax11.set_xlabel("magic flag")
    ax11.set_ylabel("RMS residual [MeV]")
    ax11.set_title("Residual comparison by magic/nonmagic")
    ax11.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax11.legend(loc="best", fontsize=8)

    fig.suptitle("Phase 7 / Step 7.16.3: pairing OES (3-point) and residual before/after correction", y=0.98)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))

    out_detail_csv = out_dir / "nuclear_pairing_effect_systematics_per_nucleus.csv"
    out_summary_csv = out_dir / "nuclear_pairing_effect_systematics_summary.csv"
    out_png = out_dir / "nuclear_pairing_effect_systematics_quantification.png"
    out_json = out_dir / "nuclear_pairing_effect_systematics_metrics.json"

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    _write_csv(out_detail_csv, rows)
    _write_csv(out_summary_csv, summary_rows)

    out_json.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.16.3",
                "inputs": {
                    "all_nuclei_csv": {"path": str(in_csv), "sha256": _sha256(in_csv)},
                },
                "definitions": {
                    "delta_n_3pt": "abs(((-1)^N)*(B(Z,N+1)-2B(Z,N)+B(Z,N-1))/2)",
                    "delta_p_3pt": "abs(((-1)^Z)*(B(Z+1,N)-2B(Z,N)+B(Z-1,N))/2)",
                    "fit_model": "resid_centered_by_A_median = k_n*delta_n + k_p*delta_p",
                    "residual": "B_pred - B_obs",
                },
                "fit": {
                    "n_fit_rows": len(fit_rows),
                    "k_n_per_MeV": float(k_n),
                    "k_p_per_MeV": float(k_p),
                    "fit_target": "centered residual by A median (no intercept)",
                },
                "counts": {
                    "n_nuclei": len(rows),
                    "n_with_delta_n": len(delta_n_map),
                    "n_with_delta_p": len(delta_p_map),
                },
                "summary": summary_rows,
                "outputs": {
                    "per_nucleus_csv": str(out_detail_csv),
                    "summary_csv": str(out_summary_csv),
                    "figure_png": str(out_png),
                },
                "notes": [
                    "Step 7.16.3 freezes pairing-systematics diagnostics from 3-point OES indicators (Delta_n, Delta_p).",
                    "Residual comparison is reported before/after a global linear correction fit on A-centered residuals (no intercept).",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[ok] wrote:")
    print(f"  {out_detail_csv}")
    print(f"  {out_summary_csv}")
    print(f"  {out_png}")
    print(f"  {out_json}")


if __name__ == "__main__":
    main()
