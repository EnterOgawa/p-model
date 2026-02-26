from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。

def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


# 関数: `_alpha_1e8_per_k` の入出力契約と処理意図を定義する。

def _alpha_1e8_per_k(*, t_k: float, coeffs: dict[str, float]) -> float:
    """
    Thermal expansion coefficient alpha(T) for silicon in units of 1e-8 / K.

    NIST TRC page provides the blended fit using erf gates and coefficients a–l.
    """
    x = float(t_k)

    a = float(coeffs["a"])
    b = float(coeffs["b"])
    c = float(coeffs["c"])
    d = float(coeffs["d"])
    e = float(coeffs["e"])
    f = float(coeffs["f"])
    g = float(coeffs["g"])
    h = float(coeffs["h"])
    i = float(coeffs["i"])
    j = float(coeffs["j"])
    k = float(coeffs["k"])
    l = float(coeffs["l"])

    w15 = 0.5 * (1.0 + math.erf(x - 15.0))
    w52m = 0.5 * (1.0 - math.erf(0.2 * (x - 52.0)))
    w52p = 0.5 * (1.0 + math.erf(0.2 * (x - 52.0)))
    w200m = 0.5 * (1.0 - math.erf(0.1 * (x - 200.0)))
    w200p = 0.5 * (1.0 + math.erf(0.1 * (x - 200.0)))

    # Region 1 (low T): 4.8e-5 x^3 + polynomial(x) blended in after ~15 K, then gated off after ~52 K.
    poly_low = a * (x**5) + b * (x**5.5) + c * (x**6) + d * (x**6.5) + e * (x**7)
    term1 = (4.8e-5 * (x**3) + poly_low * w15) * w52m

    # Region 2 (mid T): polynomial around (x-76) and gated between ~52 and ~200 K.
    y = x - 76.0
    term2 = (-47.6 + f * (y**2) + g * (y**3) + h * (y**9)) * w52p * w200m

    # Region 3 (high T): rational form, gated on after ~200 K.
    term3 = (i + j / x + k / (x**2) + l / (x**3)) * w200p

    return float(term1 + term2 + term3)


# 関数: `_sigma_fit_1e8_per_k` の入出力契約と処理意図を定義する。

def _sigma_fit_1e8_per_k(*, t_k: float, fit_error: dict[str, Any]) -> float | None:
    """
    Return the curve-fit standard error in units of 1e-8 / K, if parseable.
    """
    # 条件分岐: `not fit_error` を満たす経路を評価する。
    if not fit_error:
        return None

    lt = fit_error.get("lt") if isinstance(fit_error.get("lt"), dict) else None
    ge = fit_error.get("ge") if isinstance(fit_error.get("ge"), dict) else None
    # 条件分岐: `not (isinstance(lt, dict) and isinstance(ge, dict))` を満たす経路を評価する。
    if not (isinstance(lt, dict) and isinstance(ge, dict)):
        return None

    try:
        t0 = float(lt.get("t_k"))
        s0 = float(lt.get("sigma_1e_8_per_k"))
        t1 = float(ge.get("t_k"))
        s1 = float(ge.get("sigma_1e_8_per_k"))
    except Exception:
        return None

    # 条件分岐: `not (math.isfinite(t0) and math.isfinite(s0) and math.isfinite(t1) and math.i...` を満たす経路を評価する。

    if not (math.isfinite(t0) and math.isfinite(s0) and math.isfinite(t1) and math.isfinite(s1)):
        return None

    # 条件分岐: `abs(t0 - t1) > 1e-9` を満たす経路を評価する。

    if abs(t0 - t1) > 1e-9:
        return None

    return s0 if float(t_k) < t0 else s1


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = root / "data" / "quantum" / "sources" / "nist_trc_silicon_thermal_expansion"
    extracted_path = src_dir / "extracted_values.json"
    # 条件分岐: `not extracted_path.exists()` を満たす経路を評価する。
    if not extracted_path.exists():
        raise SystemExit(
            f"[fail] missing: {extracted_path}\n"
            "Run: python -B scripts/quantum/fetch_silicon_thermal_expansion_sources.py"
        )

    extracted = _read_json(extracted_path)
    coeffs = extracted.get("coefficients")
    # 条件分岐: `not isinstance(coeffs, dict)` を満たす経路を評価する。
    if not isinstance(coeffs, dict):
        raise SystemExit(f"[fail] coefficients missing: {extracted_path}")

    coeffs_f = {str(k).lower(): float(v) for k, v in coeffs.items()}
    missing = [k for k in "abcdefghijkl" if k not in coeffs_f]
    # 条件分岐: `missing` を満たす経路を評価する。
    if missing:
        raise SystemExit(f"[fail] missing coefficients: {missing}")

    dr = extracted.get("data_range")
    # 条件分岐: `not isinstance(dr, dict)` を満たす経路を評価する。
    if not isinstance(dr, dict):
        raise SystemExit(f"[fail] data_range missing: {extracted_path}")

    t_min = float(dr.get("t_min_k"))
    t_max = float(dr.get("t_max_k"))
    # 条件分岐: `not (math.isfinite(t_min) and math.isfinite(t_max) and t_min < t_max)` を満たす経路を評価する。
    if not (math.isfinite(t_min) and math.isfinite(t_max) and t_min < t_max):
        raise SystemExit(f"[fail] invalid data_range: {dr}")

    fit_error = extracted.get("fit_error_relative_to_data") if isinstance(extracted.get("fit_error_relative_to_data"), dict) else {}

    # Use an integer-K grid within the declared data range.
    t0 = int(math.ceil(t_min))
    t1 = int(math.floor(t_max))
    temps = [float(t) for t in range(t0, t1 + 1)]

    rows: list[dict[str, Any]] = []
    for t in temps:
        a1 = _alpha_1e8_per_k(t_k=float(t), coeffs=coeffs_f)
        s1 = _sigma_fit_1e8_per_k(t_k=float(t), fit_error=fit_error)
        rows.append(
            {
                "T_K": float(t),
                "alpha_1e-8_per_K": float(a1),
                "alpha_per_K": float(a1) * 1e-8,
                "sigma_fit_1e-8_per_K": float(s1) if (s1 is not None and math.isfinite(float(s1))) else "",
                "sigma_fit_per_K": float(s1) * 1e-8 if (s1 is not None and math.isfinite(float(s1))) else "",
            }
        )

    out_csv = out_dir / "condensed_silicon_thermal_expansion_baseline.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "T_K",
                "alpha_1e-8_per_K",
                "alpha_per_K",
                "sigma_fit_1e-8_per_K",
                "sigma_fit_per_K",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Plot

    xs = [float(r["T_K"]) for r in rows]
    ys = [float(r["alpha_1e-8_per_K"]) for r in rows]
    plt.figure(figsize=(8.5, 4.6))
    plt.plot(xs, ys, color="#1f77b4", lw=2.0)
    plt.axvline(52.0, color="#999999", lw=1.0, ls="--")
    plt.axvline(200.0, color="#999999", lw=1.0, ls="--")
    plt.title("Si thermal expansion coefficient (NIST TRC fit)")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Thermal expansion coefficient (1e-8 / K)")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()

    out_png = out_dir / "condensed_silicon_thermal_expansion_baseline.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    # 関数: `a_at` の入出力契約と処理意図を定義する。
    def a_at(t: float) -> float:
        return _alpha_1e8_per_k(t_k=float(t), coeffs=coeffs_f)

    sample_t = [20.0, 50.0, 100.0, 200.0, 293.15, 298.15, 300.0, 600.0]
    sample = {f"{t:g}K": float(a_at(t)) for t in sample_t}

    sigma_multiplier = 3.0
    falsification_targets = []
    for t in sample_t:
        alpha = float(a_at(t))
        sigma = _sigma_fit_1e8_per_k(t_k=float(t), fit_error=fit_error)
        # 条件分岐: `sigma is None` を満たす経路を評価する。
        if sigma is None:
            continue

        sigma_f = float(sigma)
        falsification_targets.append(
            {
                "T_K": float(t),
                "alpha_target_1e-8_per_K": alpha,
                "sigma_fit_1e-8_per_K": sigma_f,
                "reject_if_abs_alpha_minus_target_gt_1e-8_per_K": float(sigma_multiplier * sigma_f),
            }
        )

    out_metrics = out_dir / "condensed_silicon_thermal_expansion_baseline_metrics.json"
    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": _iso_utc_now(),
                "dataset": "Phase 7 / Step 7.14.5 silicon thermal expansion coefficient baseline (NIST TRC fit)",
                "inputs": {
                    "extracted_values_json": str(extracted_path),
                    "extracted_values_sha256": _sha256(extracted_path),
                },
                "source": extracted.get("source", {}),
                "units": {"alpha": "1e-8 / K", "note": "alpha_per_K = alpha_1e-8_per_K * 1e-8"},
                "data_range_k": {"t_min_k": t_min, "t_max_k": t_max, "grid": {"t0": t0, "t1": t1, "n": len(rows)}},
                "fit_error_relative_to_data_1e_8_per_k": fit_error,
                "falsification": {
                    "sigma_multiplier": sigma_multiplier,
                    "targets": falsification_targets,
                    "notes": [
                        "NIST TRC provides a curve-fit standard error relative to data (given separately for T<50 K and T≥50 K).",
                        "Reject thresholds use a strict ±3σ envelope based on that reported fit error.",
                    ],
                },
                "coefficients": {k: float(coeffs_f[k]) for k in "abcdefghijkl"},
                "formula_constants": {
                    "poly_low_prefactor_x3": 4.8e-5,
                    "gate_low_poly_erf_center_k": 15.0,
                    "gate_52_erf_slope": 0.2,
                    "gate_52_erf_center_k": 52.0,
                    "gate_200_erf_slope": 0.1,
                    "gate_200_erf_center_k": 200.0,
                    "mid_poly_shift_k": 76.0,
                    "mid_poly_const": -47.6,
                },
                "sample_alpha_1e-8_per_K": sample,
                "outputs": {
                    "csv": str(out_csv),
                    "png": str(out_png),
                    "metrics_json": str(out_metrics),
                },
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
