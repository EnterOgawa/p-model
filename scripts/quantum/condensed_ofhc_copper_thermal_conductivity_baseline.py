from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_sha256` の入出力契約と処理意図を定義する。

def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            # 条件分岐: `not b` を満たす経路を評価する。
            if not b:
                break

            h.update(b)

    return h.hexdigest()


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_log10k_from_coeffs` の入出力契約と処理意図を定義する。

def _log10k_from_coeffs(t_k: np.ndarray, coeffs: dict[str, float]) -> np.ndarray:
    """
    NIST TRC cryogenics OFHC Copper fit:
      log10 k = (a + c*T^0.5 + e*T + g*T^1.5 + i*T^2) / (1 + b*T^0.5 + d*T + f*T^1.5 + h*T^2)
    """
    t = np.asarray(t_k, dtype=float)
    t12 = np.sqrt(t)
    t32 = t * t12
    t2 = t * t

    a = float(coeffs["a"])
    b = float(coeffs["b"])
    c = float(coeffs["c"])
    d = float(coeffs["d"])
    e = float(coeffs["e"])
    f = float(coeffs["f"])
    g = float(coeffs["g"])
    h = float(coeffs["h"])
    i = float(coeffs["i"])

    num = a + c * t12 + e * t + g * t32 + i * t2
    den = 1.0 + b * t12 + d * t + f * t32 + h * t2
    return num / den


# 関数: `_k_w_mk_from_coeffs` の入出力契約と処理意図を定義する。

def _k_w_mk_from_coeffs(t_k: np.ndarray, coeffs: dict[str, float]) -> np.ndarray:
    return np.power(10.0, _log10k_from_coeffs(t_k, coeffs))


# 関数: `_k_at_t` の入出力契約と処理意図を定義する。

def _k_at_t(*, t_k: float, coeffs: dict[str, float], t_min_k: float, t_max_k: float) -> Optional[float]:
    # 条件分岐: `not (math.isfinite(t_k) and math.isfinite(t_min_k) and math.isfinite(t_max_k))` を満たす経路を評価する。
    if not (math.isfinite(t_k) and math.isfinite(t_min_k) and math.isfinite(t_max_k)):
        return None

    # 条件分岐: `t_k < t_min_k - 1e-9 or t_k > t_max_k + 1e-9` を満たす経路を評価する。

    if t_k < t_min_k - 1e-9 or t_k > t_max_k + 1e-9:
        return None

    v = float(_k_w_mk_from_coeffs(np.array([float(t_k)], dtype=float), coeffs)[0])
    return v if math.isfinite(v) else None


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = root / "data" / "quantum" / "sources" / "nist_trc_ofhc_copper_thermal_conductivity"
    extracted_path = src_dir / "extracted_values.json"
    # 条件分岐: `not extracted_path.exists()` を満たす経路を評価する。
    if not extracted_path.exists():
        raise SystemExit(
            f"[fail] missing: {extracted_path}\n"
            "Run: python -B scripts/quantum/fetch_ofhc_copper_thermal_conductivity_sources.py"
        )

    extracted = _read_json(extracted_path)
    # 条件分岐: `extracted.get("property") != "thermal_conductivity"` を満たす経路を評価する。
    if extracted.get("property") != "thermal_conductivity":
        raise SystemExit(f"[fail] unexpected property in {extracted_path}: {extracted.get('property')!r}")

    units = str(extracted.get("units") or "")
    rrr_obj = extracted.get("rrr")
    # 条件分岐: `not isinstance(rrr_obj, dict) or not rrr_obj` を満たす経路を評価する。
    if not isinstance(rrr_obj, dict) or not rrr_obj:
        raise SystemExit(f"[fail] rrr missing/empty: {extracted_path}")

    rrr_values: list[int] = []
    for k in rrr_obj.keys():
        try:
            rrr_values.append(int(str(k)))
        except Exception:
            continue

    rrr_values = sorted(set(rrr_values))
    # 条件分岐: `not rrr_values` を満たす経路を評価する。
    if not rrr_values:
        raise SystemExit(f"[fail] could not parse RRR keys: {list(rrr_obj.keys())[:10]}")

    # Use the intersection of ranges across RRR sets for plotting/CSV.

    t_min = max(float(rrr_obj[str(r)]["data_range"]["t_min_k"]) for r in rrr_values)
    t_max = min(float(rrr_obj[str(r)]["data_range"]["t_max_k"]) for r in rrr_values)
    # 条件分岐: `not (math.isfinite(t_min) and math.isfinite(t_max) and 0 < t_min < t_max)` を満たす経路を評価する。
    if not (math.isfinite(t_min) and math.isfinite(t_max) and 0 < t_min < t_max):
        raise SystemExit(f"[fail] invalid common range: [{t_min}, {t_max}]")

    n_grid = 600
    t_grid = np.geomspace(t_min, t_max, num=n_grid)

    curves: dict[int, np.ndarray] = {}
    for r in rrr_values:
        coeffs = rrr_obj[str(r)].get("coefficients")
        # 条件分岐: `not isinstance(coeffs, dict)` を満たす経路を評価する。
        if not isinstance(coeffs, dict):
            raise SystemExit(f"[fail] missing coefficients for RRR={r}")

        curves[int(r)] = _k_w_mk_from_coeffs(t_grid, coeffs=coeffs)  # W/(m-K)

    # CSV

    out_csv = out_dir / "condensed_ofhc_copper_thermal_conductivity_baseline.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["T_K"] + [f"k_RRR{r}_W_mK" for r in rrr_values]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(len(t_grid)):
            row: dict[str, Any] = {"T_K": float(t_grid[i])}
            for r in rrr_values:
                row[f"k_RRR{r}_W_mK"] = float(curves[r][i])

            w.writerow(row)

    # Plot (log-log)

    plt.figure(figsize=(8.5, 4.8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, r in enumerate(rrr_values):
        c = colors[idx % len(colors)]
        plt.plot(t_grid, curves[r], label=f"RRR={r}", color=c, linewidth=2.0, alpha=0.95)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Temperature T (K)")
    plt.ylabel(f"Thermal conductivity κ(T) ({units})")
    plt.title("OFHC Copper thermal conductivity κ(T) (NIST TRC cryogenics fit)")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_png = out_dir / "condensed_ofhc_copper_thermal_conductivity_baseline.png"
    plt.savefig(out_png, dpi=180)
    plt.close()

    # Summary metrics
    t_ref = [4.0, 10.0, 20.0, 50.0, 77.0, 100.0, 300.0]
    k_ref: dict[str, Any] = {}
    maxima: dict[str, Any] = {}
    for r in rrr_values:
        rr = rrr_obj[str(r)]
        coeffs = rr["coefficients"]
        t_min_r = float(rr["data_range"]["t_min_k"])
        t_max_r = float(rr["data_range"]["t_max_k"])
        k_ref[str(r)] = {str(t): _k_at_t(t_k=float(t), coeffs=coeffs, t_min_k=t_min_r, t_max_k=t_max_r) for t in t_ref}

        k_arr = curves[r]
        imax = int(np.nanargmax(k_arr))
        maxima[str(r)] = {
            "k_max_W_mK": float(k_arr[imax]),
            "t_at_k_max_K": float(t_grid[imax]),
            "k_min_W_mK": float(np.nanmin(k_arr)),
            "t_min_K": float(t_min),
            "t_max_K": float(t_max),
        }

    fit_err_pct = {str(r): float(rrr_obj[str(r)]["curve_fit_error_percent_relative_to_data"]) for r in rrr_values}

    # Falsification targets: treat NIST curve-fit relative error as a 1σ proxy for κ(T).
    sigma_multiplier = 3.0
    falsification_targets_by_rrr: dict[str, Any] = {}
    for r in rrr_values:
        err_rel = float(fit_err_pct[str(r)]) / 100.0
        tt = []
        for t_s, k in k_ref[str(r)].items():
            # 条件分岐: `k is None or not math.isfinite(float(k))` を満たす経路を評価する。
            if k is None or not math.isfinite(float(k)):
                continue

            kf = float(k)
            sigma_k = err_rel * kf
            tt.append(
                {
                    "T_K": float(t_s),
                    "k_target_W_mK": kf,
                    "sigma_fit_W_mK": float(sigma_k),
                    "reject_if_abs_k_minus_target_gt_W_mK": float(sigma_multiplier * sigma_k),
                }
            )

        kmax = float(maxima[str(r)]["k_max_W_mK"])
        tmax = float(maxima[str(r)]["t_at_k_max_K"])
        sigma_kmax = err_rel * kmax
        falsification_targets_by_rrr[str(r)] = {
            "fit_error_percent_relative_to_data": float(fit_err_pct[str(r)]),
            "targets_at_selected_T": tt,
            "target_peak": {
                "t_at_k_max_K": tmax,
                "k_max_W_mK": kmax,
                "sigma_fit_W_mK": float(sigma_kmax),
                "reject_if_abs_k_max_minus_target_gt_W_mK": float(sigma_multiplier * sigma_kmax),
            },
        }

    kmax_by_rrr = [float(maxima[str(r)]["k_max_W_mK"]) for r in rrr_values]
    tpeak_by_rrr = [float(maxima[str(r)]["t_at_k_max_K"]) for r in rrr_values]
    monotonic_kmax_increasing = all(a <= b for a, b in zip(kmax_by_rrr, kmax_by_rrr[1:]))
    monotonic_tpeak_decreasing = all(a >= b for a, b in zip(tpeak_by_rrr, tpeak_by_rrr[1:]))

    out_metrics = out_dir / "condensed_ofhc_copper_thermal_conductivity_baseline_metrics.json"
    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.14.6",
                "inputs": {"extracted_values": {"path": str(extracted_path), "sha256": _sha256(extracted_path)}},
                "results": {
                    "material": str(extracted.get("material") or "OFHC Copper"),
                    "units": units,
                    "rrr_values": rrr_values,
                    "fit_error_percent_relative_to_data": fit_err_pct,
                    "common_grid": {"t_min_k": float(t_min), "t_max_k": float(t_max), "n": int(n_grid), "kind": "geomspace"},
                    "k_at_selected_T_K": k_ref,
                    "k_extrema_on_common_grid": maxima,
                    "monotonic_checks": {
                        "k_max_increases_with_rrr": monotonic_kmax_increasing,
                        "t_at_k_max_decreases_with_rrr": monotonic_tpeak_decreasing,
                    },
                },
                "falsification": {
                    "sigma_multiplier": sigma_multiplier,
                    "targets_by_rrr": falsification_targets_by_rrr,
                    "notes": [
                        "NIST TRC provides a curve-fit relative error vs data (typically 1–2% depending on RRR).",
                        "We treat that fit error as a 1σ proxy and define a strict ±3σ envelope for κ(T) at representative temperatures.",
                    ],
                },
                "outputs": {"csv": str(out_csv), "png": str(out_png)},
                "notes": [
                    "This is a fixed baseline derived from NIST TRC Cryogenics 'Material Properties: OFHC Copper'.",
                    "The NIST page provides a rational fit for log10(κ) as a function of T (K), with separate coefficients for each RRR value.",
                    "The baseline freezes κ(T) on a common temperature grid (intersection of ranges across RRR sets).",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_metrics}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
