from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt


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


# 関数: `_alpha_1e8_per_k` の入出力契約と処理意図を定義する。

def _alpha_1e8_per_k(*, t_k: float, coeffs: dict[str, float]) -> float:
    """
    Silicon thermal expansion coefficient alpha(T) from NIST TRC Cryogenics fit.
    Units: 1e-8 / K.
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

    poly_low = a * (x**5) + b * (x**5.5) + c * (x**6) + d * (x**6.5) + e * (x**7)
    term1 = (4.8e-5 * (x**3) + poly_low * w15) * w52m

    y = x - 76.0
    term2 = (-47.6 + f * (y**2) + g * (y**3) + h * (y**9)) * w52p * w200m

    term3 = (i + j / x + k / (x**2) + l / (x**3)) * w200p
    return float(term1 + term2 + term3)


# 関数: `_debye_integrand` の入出力契約と処理意図を定義する。

def _debye_integrand(x: float) -> float:
    # 条件分岐: `x <= 0.0` を満たす経路を評価する。
    if x <= 0.0:
        return 0.0

    em1 = math.expm1(x)
    # 条件分岐: `em1 == 0.0` を満たす経路を評価する。
    if em1 == 0.0:
        return 0.0

    inv = 1.0 / em1
    return (x**4) * (inv + inv * inv)


# 関数: `_simpson_integrate` の入出力契約と処理意図を定義する。

def _simpson_integrate(f, a: float, b: float, n: int) -> float:
    # 条件分岐: `n < 2` を満たす経路を評価する。
    if n < 2:
        n = 2

    # 条件分岐: `n % 2 == 1` を満たす経路を評価する。

    if n % 2 == 1:
        n += 1

    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        s += (4.0 if (i % 2 == 1) else 2.0) * f(x)

    return s * (h / 3.0)


# 関数: `_debye_cv_molar` の入出力契約と処理意図を定義する。

def _debye_cv_molar(*, t_k: float, theta_d_k: float) -> float:
    """
    Debye heat capacity Cv for a monatomic solid, per mole.
    """
    # 条件分岐: `t_k <= 0.0 or theta_d_k <= 0.0` を満たす経路を評価する。
    if t_k <= 0.0 or theta_d_k <= 0.0:
        return 0.0

    r = 8.314462618
    y = theta_d_k / t_k

    # For x > ~50, integrand ~ x^4 e^{-x} is negligible.
    y_eff = min(float(y), 50.0)
    n = int(max(800, 40 * y_eff))
    integral = _simpson_integrate(_debye_integrand, 0.0, y_eff, n)
    return 9.0 * r * ((t_k / theta_d_k) ** 3) * integral


# 関数: `_theta_d_from_existing_metrics` の入出力契約と処理意図を定義する。

def _theta_d_from_existing_metrics(root: Path) -> Optional[float]:
    m = root / "output" / "public" / "quantum" / "condensed_silicon_heat_capacity_debye_baseline_metrics.json"
    # 条件分岐: `not m.exists()` を満たす経路を評価する。
    if not m.exists():
        return None

    try:
        obj = _read_json(m)
    except Exception:
        return None

    fit = obj.get("fit") if isinstance(obj.get("fit"), dict) else None
    # 条件分岐: `not isinstance(fit, dict)` を満たす経路を評価する。
    if not isinstance(fit, dict):
        return None

    v = fit.get("theta_D_K")
    # 条件分岐: `isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0` を満たす経路を評価する。
    if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0:
        return float(v)

    return None


# クラス: `FitResult` の責務と境界条件を定義する。

@dataclass(frozen=True)
class FitResult:
    degree: int
    coeffs: list[float]
    max_abs_z: float
    reduced_chi2: float
    exceed_3sigma_n: int
    sign_mismatch_n: int
    n_fit: int


# 関数: `_solve_linear` の入出力契約と処理意図を定義する。

def _solve_linear(a: list[list[float]], b: list[float]) -> Optional[list[float]]:
    n = len(a)
    # 条件分岐: `n == 0 or any(len(row) != n for row in a) or len(b) != n` を満たす経路を評価する。
    if n == 0 or any(len(row) != n for row in a) or len(b) != n:
        return None

    # Augmented matrix.

    m = [row[:] + [float(bb)] for row, bb in zip(a, b)]

    # Gaussian elimination with partial pivoting.
    for col in range(n):
        pivot = col
        pivot_abs = abs(float(m[col][col]))
        for r in range(col + 1, n):
            v = abs(float(m[r][col]))
            # 条件分岐: `v > pivot_abs` を満たす経路を評価する。
            if v > pivot_abs:
                pivot = r
                pivot_abs = v

        # 条件分岐: `pivot_abs <= 1e-30 or not math.isfinite(pivot_abs)` を満たす経路を評価する。

        if pivot_abs <= 1e-30 or not math.isfinite(pivot_abs):
            return None

        # 条件分岐: `pivot != col` を満たす経路を評価する。

        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]

        piv = float(m[col][col])
        inv = 1.0 / piv
        for c in range(col, n + 1):
            m[col][c] = float(m[col][c]) * inv

        for r in range(n):
            # 条件分岐: `r == col` を満たす経路を評価する。
            if r == col:
                continue

            factor = float(m[r][col])
            # 条件分岐: `factor == 0.0` を満たす経路を評価する。
            if factor == 0.0:
                continue

            for c in range(col, n + 1):
                m[r][c] = float(m[r][c]) - factor * float(m[col][c])

    x = [float(m[i][n]) for i in range(n)]
    # 条件分岐: `not all(math.isfinite(v) for v in x)` を満たす経路を評価する。
    if not all(math.isfinite(v) for v in x):
        return None

    return x


# 関数: `_fit_poly_a_eff` の入出力契約と処理意図を定義する。

def _fit_poly_a_eff(
    *,
    degree: int,
    temps: list[float],
    cv: list[float],
    alpha_obs: list[float],
    sigma: list[float],
    fit_idx: list[int],
) -> Optional[list[float]]:
    """
    Fit A_eff(T)=sum_{k=0..degree} c_k T^k by weighted LS on alpha:
      alpha(T) ≈ A_eff(T) * Cv(T).
    """
    p = degree + 1
    xtwx = [[0.0 for _ in range(p)] for _ in range(p)]
    xtwy = [0.0 for _ in range(p)]

    for i in fit_idx:
        sig = float(sigma[i])
        # 条件分岐: `sig <= 0.0` を満たす経路を評価する。
        if sig <= 0.0:
            continue

        w = 1.0 / (sig * sig)
        t = float(temps[i])
        cvi = float(cv[i])
        y = float(alpha_obs[i])
        # Basis: (T^k * Cv)
        basis = [((t**k) * cvi) for k in range(p)]
        for r in range(p):
            xtwy[r] += w * float(basis[r]) * y
            for c in range(p):
                xtwx[r][c] += w * float(basis[r]) * float(basis[c])

    return _solve_linear(xtwx, xtwy)


# 関数: `_poly_eval` の入出力契約と処理意図を定義する。

def _poly_eval(coeffs: list[float], t: float) -> float:
    s = 0.0
    tp = 1.0
    for c in coeffs:
        s += float(c) * tp
        tp *= float(t)

    return float(s)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_src = root / "data" / "quantum" / "sources" / "nist_trc_silicon_thermal_expansion" / "extracted_values.json"
    # 条件分岐: `not alpha_src.exists()` を満たす経路を評価する。
    if not alpha_src.exists():
        raise SystemExit(
            f"[fail] missing: {alpha_src}\n"
            "Run: python -B scripts/quantum/fetch_silicon_thermal_expansion_sources.py"
        )

    alpha_extracted = _read_json(alpha_src)
    coeffs_obj = alpha_extracted.get("coefficients")
    # 条件分岐: `not isinstance(coeffs_obj, dict)` を満たす経路を評価する。
    if not isinstance(coeffs_obj, dict):
        raise SystemExit(f"[fail] coefficients missing: {alpha_src}")

    coeffs = {str(k).lower(): float(v) for k, v in coeffs_obj.items()}
    missing = [k for k in "abcdefghijkl" if k not in coeffs]
    # 条件分岐: `missing` を満たす経路を評価する。
    if missing:
        raise SystemExit(f"[fail] missing coefficients: {missing}")

    dr = alpha_extracted.get("data_range")
    # 条件分岐: `not isinstance(dr, dict)` を満たす経路を評価する。
    if not isinstance(dr, dict):
        raise SystemExit(f"[fail] data_range missing: {alpha_src}")

    t_min = int(math.ceil(float(dr.get("t_min_k"))))
    t_max = int(math.floor(float(dr.get("t_max_k"))))
    # 条件分岐: `not (0 < t_min < t_max)` を満たす経路を評価する。
    if not (0 < t_min < t_max):
        raise SystemExit(f"[fail] invalid data_range: {dr}")

    fe = alpha_extracted.get("fit_error_relative_to_data")
    # 条件分岐: `not isinstance(fe, dict) or not isinstance(fe.get("lt"), dict) or not isinsta...` を満たす経路を評価する。
    if not isinstance(fe, dict) or not isinstance(fe.get("lt"), dict) or not isinstance(fe.get("ge"), dict):
        raise SystemExit(f"[fail] fit_error_relative_to_data missing: {alpha_src}")

    t_sigma_split = float(fe["lt"].get("t_k", 50.0))
    sigma_lt_1e8 = float(fe["lt"].get("sigma_1e_8_per_k", 0.03))
    sigma_ge_1e8 = float(fe["ge"].get("sigma_1e_8_per_k", 0.5))

    theta_from_metrics = _theta_d_from_existing_metrics(root)
    # 条件分岐: `theta_from_metrics is None` を満たす経路を評価する。
    if theta_from_metrics is None:
        raise SystemExit("[fail] missing frozen θ_D. Expected output/public/quantum/condensed_silicon_heat_capacity_debye_baseline_metrics.json")

    theta_d = float(theta_from_metrics)

    temps = [float(t) for t in range(t_min, t_max + 1)]
    alpha_obs_1e8 = [_alpha_1e8_per_k(t_k=t, coeffs=coeffs) for t in temps]
    alpha_obs = [float(a) * 1e-8 for a in alpha_obs_1e8]
    sigma_1e8 = [sigma_lt_1e8 if float(t) < t_sigma_split else sigma_ge_1e8 for t in temps]
    sigma = [float(s) * 1e-8 for s in sigma_1e8]
    cv = [_debye_cv_molar(t_k=t, theta_d_k=theta_d) for t in temps]

    fit_min_k = 50.0
    fit_max_k = float(t_max)
    fit_idx = [i for i, t in enumerate(temps) if fit_min_k <= float(t) <= fit_max_k]
    # 条件分岐: `len(fit_idx) < 100` を満たす経路を評価する。
    if len(fit_idx) < 100:
        raise SystemExit(f"[fail] not enough fit points: n={len(fit_idx)} in [{fit_min_k},{fit_max_k}] K")

    # Scan polynomial degrees (keep minimal DOF).

    degrees = [0, 1, 2, 3]
    results: list[FitResult] = []
    coeffs_by_deg: dict[int, list[float]] = {}

    for deg in degrees:
        coeffs_fit = _fit_poly_a_eff(
            degree=deg,
            temps=temps,
            cv=cv,
            alpha_obs=alpha_obs,
            sigma=sigma,
            fit_idx=fit_idx,
        )
        # 条件分岐: `coeffs_fit is None` を満たす経路を評価する。
        if coeffs_fit is None:
            continue

        coeffs_by_deg[int(deg)] = [float(x) for x in coeffs_fit]

        alpha_pred = [_poly_eval(coeffs_fit, float(t)) * float(c) for t, c in zip(temps, cv)]

        sign_mismatch = 0
        exceed_3sigma = 0
        max_abs_z = 0.0
        chi2 = 0.0
        n_fit = 0
        for i in fit_idx:
            sig = float(sigma[i])
            # 条件分岐: `sig <= 0.0` を満たす経路を評価する。
            if sig <= 0.0:
                continue

            r = float(alpha_pred[i]) - float(alpha_obs[i])
            z = r / sig
            # 条件分岐: `math.isfinite(z)` を満たす経路を評価する。
            if math.isfinite(z):
                n_fit += 1
                chi2 += z * z
                max_abs_z = max(max_abs_z, abs(z))
                # 条件分岐: `abs(z) > 3.0` を満たす経路を評価する。
                if abs(z) > 3.0:
                    exceed_3sigma += 1

            ao = float(alpha_obs[i])
            ap = float(alpha_pred[i])
            # 条件分岐: `ao != 0.0 and ap != 0.0 and ((ao > 0.0) != (ap > 0.0))` を満たす経路を評価する。
            if ao != 0.0 and ap != 0.0 and ((ao > 0.0) != (ap > 0.0)):
                sign_mismatch += 1

        dof = max(1, n_fit - (deg + 1))
        red_chi2 = float(chi2 / dof) if dof > 0 else float("nan")
        results.append(
            FitResult(
                degree=int(deg),
                coeffs=[float(x) for x in coeffs_fit],
                max_abs_z=float(max_abs_z),
                reduced_chi2=float(red_chi2),
                exceed_3sigma_n=int(exceed_3sigma),
                sign_mismatch_n=int(sign_mismatch),
                n_fit=int(n_fit),
            )
        )

    # 条件分岐: `not results` を満たす経路を評価する。

    if not results:
        raise SystemExit("[fail] no successful polynomial fits")

    # Choose minimal degree that passes strict criteria; otherwise choose the smallest reduced chi2.

    def _passes(r: FitResult) -> bool:
        return (r.sign_mismatch_n == 0) and (r.max_abs_z < 3.0) and (r.reduced_chi2 < 2.0)

    passed = [r for r in results if _passes(r)]
    # 条件分岐: `passed` を満たす経路を評価する。
    if passed:
        best = sorted(passed, key=lambda r: r.degree)[0]
        adopted = True
        selection = "min_degree_pass"
    else:
        best = sorted(results, key=lambda r: (r.reduced_chi2, r.max_abs_z))[0]
        adopted = False
        selection = "min_reduced_chi2"

    best_coeffs = coeffs_by_deg[int(best.degree)]
    a_eff_pred = [_poly_eval(best_coeffs, float(t)) for t in temps]
    alpha_pred = [float(a_eff) * float(c) for a_eff, c in zip(a_eff_pred, cv)]
    alpha_pred_1e8 = [float(a) / 1e-8 for a in alpha_pred]
    residual = [float(ap - ao) for ap, ao in zip(alpha_pred, alpha_obs)]
    residual_1e8 = [float(r) / 1e-8 for r in residual]
    z_score = [float("nan") if float(s) <= 0.0 else float(r) / float(s) for r, s in zip(residual, sigma)]

    # CSV
    out_csv = out_dir / "condensed_silicon_thermal_expansion_gruneisen_polyA_model.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "T_K",
                "alpha_obs_1e-8_per_K",
                "alpha_pred_1e-8_per_K",
                "sigma_fit_1e-8_per_K",
                "Cv_debye_J_per_molK",
                "A_eff_pred_mol_per_J",
                "residual_1e-8_per_K",
                "z",
            ],
        )
        w.writeheader()
        for t, ao, ap, s1e8, c, aeff, r1e8, z in zip(
            temps, alpha_obs_1e8, alpha_pred_1e8, sigma_1e8, cv, a_eff_pred, residual_1e8, z_score
        ):
            w.writerow(
                {
                    "T_K": float(t),
                    "alpha_obs_1e-8_per_K": float(ao),
                    "alpha_pred_1e-8_per_K": float(ap),
                    "sigma_fit_1e-8_per_K": float(s1e8),
                    "Cv_debye_J_per_molK": float(c),
                    "A_eff_pred_mol_per_J": float(aeff),
                    "residual_1e-8_per_K": float(r1e8),
                    "z": (float(z) if math.isfinite(float(z)) else ""),
                }
            )

    # Figure

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(temps, alpha_obs_1e8, color="#d62728", lw=2.0, label="NIST TRC fit (obs) α(T)")
    ax1.plot(
        temps,
        alpha_pred_1e8,
        color="#1f77b4",
        lw=2.0,
        label=f"Poly A_eff(T) deg={best.degree}: α≈A_eff(T)·Cv_D(θ_D)",
    )
    ax1.axhline(0.0, color="#666666", lw=1.0, alpha=0.6)
    ax1.set_ylabel("α(T) (10^-8 / K)")
    ax1.set_title("Silicon thermal expansion: polynomial A_eff(T) ansatz (diagnostic)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best", fontsize=8)

    ax2.plot(temps, z_score, color="#000000", lw=1.2, alpha=0.9, label="z = (α_pred−α_obs)/σ_fit")
    ax2.axhline(0.0, color="#666666", lw=1.0, alpha=0.6)
    ax2.axhline(3.0, color="#999999", lw=1.0, ls="--", alpha=0.7)
    ax2.axhline(-3.0, color="#999999", lw=1.0, ls="--", alpha=0.7)
    ax2.set_xlabel("Temperature T (K)")
    ax2.set_ylabel("z")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out_png = out_dir / "condensed_silicon_thermal_expansion_gruneisen_polyA_model.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    out_metrics = out_dir / "condensed_silicon_thermal_expansion_gruneisen_polyA_model_metrics.json"
    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.14.12",
                "inputs": {
                    "silicon_thermal_expansion_extracted_values": {"path": str(alpha_src), "sha256": _sha256(alpha_src)},
                    "theta_d_source": {"kind": "frozen_metrics", "path": "output/public/quantum/condensed_silicon_heat_capacity_debye_baseline_metrics.json"},
                },
                "model_family": {
                    "name": "A_eff(T) polynomial ansatz (diagnostic)",
                    "alpha_form": "alpha(T) = A_eff(T) * Cv_Debye(T; theta_D), with A_eff(T)=sum_{k=0..n} c_k T^k",
                    "theta_D_K": float(theta_d),
                    "fit_range_T_K": {"min": float(fit_min_k), "max": float(fit_max_k)},
                    "degrees_scanned": degrees,
                },
                "fit_results_by_degree": [
                    {
                        "degree": r.degree,
                        "coeffs_mol_per_J": [float(c) for c in r.coeffs],
                        "n_fit": int(r.n_fit),
                        "max_abs_z": float(r.max_abs_z),
                        "reduced_chi2": float(r.reduced_chi2),
                        "exceed_3sigma_n": int(r.exceed_3sigma_n),
                        "sign_mismatch_n": int(r.sign_mismatch_n),
                    }
                    for r in sorted(results, key=lambda x: x.degree)
                ],
                "selection": {
                    "criterion": selection,
                    "best_degree": int(best.degree),
                    "best_coeffs_mol_per_J": [float(c) for c in best_coeffs],
                    "adopted": bool(adopted),
                },
                "diagnostics_best": {
                    "best_degree": int(best.degree),
                    "max_abs_z_fit_range": float(best.max_abs_z),
                    "reduced_chi2_fit_range": float(best.reduced_chi2),
                    "exceed_3sigma_fit_range_n": int(best.exceed_3sigma_n),
                    "sign_mismatch_fit_range_n": int(best.sign_mismatch_n),
                    "sigma_fit_1e-8_per_K": {"lt_T_K": float(t_sigma_split), "lt": float(sigma_lt_1e8), "ge": float(sigma_ge_1e8)},
                },
                "falsification": {
                    "reject_if_sign_mismatch_fit_range_n_gt": 0,
                    "reject_if_max_abs_z_ge": 3.0,
                    "reject_if_reduced_chi2_ge": 2.0,
                    "rejected": (not adopted),
                    "notes": [
                        "This is a diagnostic ansatz to estimate how much temperature-dependence is required in A_eff(T)=γ/(B V_m).",
                        "Passing does not constitute a derivation; it only suggests that a low-order smooth A_eff(T) could reconcile α(T) with Debye Cv at the σ_fit scale.",
                    ],
                },
                "outputs": {"csv": str(out_csv), "png": str(out_png)},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[ok] best_degree={best.degree} adopted={adopted} max_abs_z={best.max_abs_z:.3g} red_chi2={best.reduced_chi2:.3g}")
    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_metrics}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()

