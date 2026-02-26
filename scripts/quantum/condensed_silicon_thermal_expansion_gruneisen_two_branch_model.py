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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _golden_section_minimize(f, lo: float, hi: float, *, tol: float = 1e-6, max_iter: int = 120) -> tuple[float, float]:
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a = float(lo)
    b = float(hi)
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        # 条件分岐: `abs(b - a) <= tol * (abs(c) + abs(d) + 1.0)` を満たす経路を評価する。
        if abs(b - a) <= tol * (abs(c) + abs(d) + 1.0):
            break

        # 条件分岐: `fc < fd` を満たす経路を評価する。

        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)

    x = (a + b) / 2.0
    return x, f(x)


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


@dataclass(frozen=True)
class DebyeFitPoint:
    t_k: float
    cp_obs: float


def _fit_theta_d_from_janaf(root: Path) -> tuple[float, list[DebyeFitPoint]]:
    src = root / "data" / "quantum" / "sources" / "nist_janaf_silicon_si" / "extracted_values.json"
    # 条件分岐: `not src.exists()` を満たす経路を評価する。
    if not src.exists():
        raise SystemExit(
            f"[fail] missing: {src}\n"
            "Run: python -B scripts/quantum/fetch_silicon_janaf_sources.py"
        )

    obj = _read_json(src)
    points = obj.get("points")
    # 条件分岐: `not isinstance(points, list) or not points` を満たす経路を評価する。
    if not isinstance(points, list) or not points:
        raise SystemExit(f"[fail] points missing/empty: {src}")

    solid = [
        p
        for p in points
        if isinstance(p, dict)
        and str(p.get("phase")) == "solid"
        and isinstance(p.get("T_K"), (int, float))
        and isinstance(p.get("Cp_J_per_molK"), (int, float))
    ]
    # 条件分岐: `not solid` を満たす経路を評価する。
    if not solid:
        raise SystemExit(f"[fail] no solid-phase points found in: {src}")

    fit_points = [
        DebyeFitPoint(t_k=float(p["T_K"]), cp_obs=float(p["Cp_J_per_molK"]))
        for p in solid
        if 100.0 <= float(p["T_K"]) <= 300.0
    ]
    # 条件分岐: `len(fit_points) < 4` を満たす経路を評価する。
    if len(fit_points) < 4:
        raise SystemExit(f"[fail] not enough fit points in 100–300 K range: n={len(fit_points)}")

    fit_points = sorted(fit_points, key=lambda x: x.t_k)

    def sse(theta: float) -> float:
        return float(sum((p.cp_obs - _debye_cv_molar(t_k=p.t_k, theta_d_k=theta)) ** 2 for p in fit_points))

    theta0, _ = _golden_section_minimize(sse, 300.0, 900.0, tol=1e-5)
    return float(theta0), fit_points


def _infer_zero_crossing(
    xs: list[float],
    ys: list[float],
    *,
    prefer_neg_to_pos: bool = True,
    min_x: Optional[float] = None,
) -> Optional[dict[str, float]]:
    # 条件分岐: `len(xs) != len(ys) or len(xs) < 2` を満たす経路を評価する。
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    candidates: list[dict[str, float]] = []
    for i in range(len(xs) - 1):
        x0 = float(xs[i])
        x1 = float(xs[i + 1])
        # 条件分岐: `min_x is not None and max(x0, x1) < float(min_x)` を満たす経路を評価する。
        if min_x is not None and max(x0, x1) < float(min_x):
            continue

        y0 = float(ys[i])
        y1 = float(ys[i + 1])
        # 条件分岐: `y0 == 0.0` を満たす経路を評価する。
        if y0 == 0.0:
            candidates.append({"x0": x0, "x1": x0, "x_cross": x0})
            continue

        # 条件分岐: `y1 == 0.0` を満たす経路を評価する。

        if y1 == 0.0:
            candidates.append({"x0": x1, "x1": x1, "x_cross": x1})
            continue

        # Detect sign changes.

        if (y0 < 0.0 and y1 > 0.0) or (y0 > 0.0 and y1 < 0.0):
            x_cross = x0 + (x1 - x0) * (-y0) / (y1 - y0)
            candidates.append({"x0": x0, "x1": x1, "x_cross": float(x_cross), "y0": y0, "y1": y1})

    # 条件分岐: `not candidates` を満たす経路を評価する。

    if not candidates:
        return None

    # 条件分岐: `prefer_neg_to_pos` を満たす経路を評価する。

    if prefer_neg_to_pos:
        neg_to_pos = [c for c in candidates if float(c.get("y0", 0.0)) < 0.0 and float(c.get("y1", 0.0)) > 0.0]
        # 条件分岐: `neg_to_pos` を満たす経路を評価する。
        if neg_to_pos:
            # Prefer the physically relevant negative→positive crossing (ignore the low-T wiggle).
            return max(neg_to_pos, key=lambda c: float(c["x_cross"]))

    return max(candidates, key=lambda c: float(c["x_cross"]))


def _logspace(*, lo: float, hi: float, n: int) -> list[float]:
    # 条件分岐: `n < 2` を満たす経路を評価する。
    if n < 2:
        return [float(lo)]

    # 条件分岐: `lo <= 0.0 or hi <= 0.0` を満たす経路を評価する。

    if lo <= 0.0 or hi <= 0.0:
        raise ValueError("logspace requires positive lo/hi")

    l0 = math.log10(float(lo))
    l1 = math.log10(float(hi))
    return [10 ** (l0 + (l1 - l0) * i / (n - 1)) for i in range(n)]


def _solve_2x2(*, a11: float, a12: float, a22: float, b1: float, b2: float) -> Optional[tuple[float, float]]:
    det = float(a11) * float(a22) - float(a12) * float(a12)
    # 条件分岐: `not math.isfinite(det) or abs(det) <= 1e-30` を満たす経路を評価する。
    if not math.isfinite(det) or abs(det) <= 1e-30:
        return None

    x1 = (float(b1) * float(a22) - float(b2) * float(a12)) / det
    x2 = (float(b2) * float(a11) - float(b1) * float(a12)) / det
    # 条件分岐: `not (math.isfinite(x1) and math.isfinite(x2))` を満たす経路を評価する。
    if not (math.isfinite(x1) and math.isfinite(x2)):
        return None

    return float(x1), float(x2)


def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Alpha(T) source (NIST TRC Cryogenics).
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

    # Theta_D for branch-1: prefer the frozen baseline if present; otherwise refit from JANAF (reproducible fallback).
    theta_from_metrics = _theta_d_from_existing_metrics(root)
    theta_fit_points: list[DebyeFitPoint] = []
    # 条件分岐: `theta_from_metrics is not None` を満たす経路を評価する。
    if theta_from_metrics is not None:
        theta1 = float(theta_from_metrics)
    else:
        theta1, theta_fit_points = _fit_theta_d_from_janaf(root)

    # Prepare grid.

    temps = [float(t) for t in range(t_min, t_max + 1)]
    alpha_obs_1e8 = [_alpha_1e8_per_k(t_k=t, coeffs=coeffs) for t in temps]
    alpha_obs = [float(a) * 1e-8 for a in alpha_obs_1e8]  # 1/K
    sigma_fit_1e8 = [sigma_lt_1e8 if float(t) < t_sigma_split else sigma_ge_1e8 for t in temps]
    sigma_fit = [float(s) * 1e-8 for s in sigma_fit_1e8]

    cv1 = [_debye_cv_molar(t_k=t, theta_d_k=theta1) for t in temps]  # J/molK

    # Fit range: avoid the very-low-T wiggle (and huge weight from tiny sigma).
    fit_min_k = 50.0
    fit_max_k = float(t_max)
    fit_idx = [i for i, t in enumerate(temps) if fit_min_k <= float(t) <= fit_max_k]
    # 条件分岐: `len(fit_idx) < 100` を満たす経路を評価する。
    if len(fit_idx) < 100:
        raise SystemExit(f"[fail] not enough fit points: n={len(fit_idx)} in [{fit_min_k},{fit_max_k}] K")

    # Two-branch Debye–Grüneisen model:
    #   alpha(T) ≈ A1 * Cv(T; theta1) + A2 * Cv(T; theta2)
    # where theta1 is frozen from the Cp Debye baseline, and theta2 is scanned.

    theta2_grid = _logspace(lo=10.0, hi=2000.0, n=700)
    best: dict[str, float] | None = None

    for theta2 in theta2_grid:
        cv2 = [_debye_cv_molar(t_k=t, theta_d_k=float(theta2)) for t in temps]

        s11 = 0.0
        s22 = 0.0
        s12 = 0.0
        b1 = 0.0
        b2 = 0.0
        for i in fit_idx:
            sig = float(sigma_fit[i])
            # 条件分岐: `sig <= 0.0` を満たす経路を評価する。
            if sig <= 0.0:
                continue

            w = 1.0 / (sig * sig)
            x1 = float(cv1[i])
            x2 = float(cv2[i])
            y = float(alpha_obs[i])
            s11 += w * x1 * x1
            s22 += w * x2 * x2
            s12 += w * x1 * x2
            b1 += w * x1 * y
            b2 += w * x2 * y

        sol = _solve_2x2(a11=s11, a12=s12, a22=s22, b1=b1, b2=b2)
        # 条件分岐: `sol is None` を満たす経路を評価する。
        if sol is None:
            continue

        a1, a2 = sol

        sse = 0.0
        for i in fit_idx:
            sig = float(sigma_fit[i])
            # 条件分岐: `sig <= 0.0` を満たす経路を評価する。
            if sig <= 0.0:
                continue

            w = 1.0 / (sig * sig)
            pred = float(a1) * float(cv1[i]) + float(a2) * float(cv2[i])
            r = float(alpha_obs[i]) - pred
            sse += w * r * r

        # 条件分岐: `best is None or float(sse) < float(best["sse"])` を満たす経路を評価する。

        if best is None or float(sse) < float(best["sse"]):
            best = {"theta2_k": float(theta2), "a1": float(a1), "a2": float(a2), "sse": float(sse)}

    # 条件分岐: `best is None` を満たす経路を評価する。

    if best is None:
        raise SystemExit("[fail] theta2 scan failed")

    theta2_best = float(best["theta2_k"])
    a1_best = float(best["a1"])
    a2_best = float(best["a2"])
    cv2_best = [_debye_cv_molar(t_k=t, theta_d_k=theta2_best) for t in temps]

    alpha_pred = [a1_best * float(c1) + a2_best * float(c2) for c1, c2 in zip(cv1, cv2_best)]
    alpha_pred_1e8 = [float(a) / 1e-8 for a in alpha_pred]

    contrib1 = [a1_best * float(c) for c in cv1]
    contrib2 = [a2_best * float(c) for c in cv2_best]
    contrib1_1e8 = [float(a) / 1e-8 for a in contrib1]
    contrib2_1e8 = [float(a) / 1e-8 for a in contrib2]

    residual = [float(ap - ao) for ap, ao in zip(alpha_pred, alpha_obs)]
    residual_1e8 = [float(r) / 1e-8 for r in residual]
    z_score = [float("nan") if float(s) <= 0.0 else float(r) / float(s) for r, s in zip(residual, sigma_fit)]

    # Diagnostics.
    sign_mismatch_fit = 0
    for i in fit_idx:
        ao = float(alpha_obs[i])
        ap = float(alpha_pred[i])
        # 条件分岐: `ao == 0.0 or ap == 0.0` を満たす経路を評価する。
        if ao == 0.0 or ap == 0.0:
            continue

        # 条件分岐: `(ao > 0.0) != (ap > 0.0)` を満たす経路を評価する。

        if (ao > 0.0) != (ap > 0.0):
            sign_mismatch_fit += 1

    exceed_3sigma_fit = 0
    max_abs_z_fit = 0.0
    sum_z2_fit = 0.0
    n_z_fit = 0
    for i in fit_idx:
        z = float(z_score[i])
        # 条件分岐: `not math.isfinite(z)` を満たす経路を評価する。
        if not math.isfinite(z):
            continue

        n_z_fit += 1
        sum_z2_fit += z * z
        max_abs_z_fit = max(max_abs_z_fit, abs(z))
        # 条件分岐: `abs(z) > 3.0` を満たす経路を評価する。
        if abs(z) > 3.0:
            exceed_3sigma_fit += 1

    rms_z_fit = float("nan") if n_z_fit <= 0 else math.sqrt(sum_z2_fit / n_z_fit)
    dof = max(1, n_z_fit - 3)  # (theta2, a1, a2)
    red_chi2 = float(sum_z2_fit / dof) if dof > 0 else float("nan")

    zero_obs = _infer_zero_crossing(temps, alpha_obs, prefer_neg_to_pos=True, min_x=50.0)
    zero_pred = _infer_zero_crossing(temps, alpha_pred, prefer_neg_to_pos=True, min_x=50.0)

    rejected = (sign_mismatch_fit > 0) or (max_abs_z_fit >= 3.0) or (red_chi2 >= 2.0)

    # CSV
    out_csv = out_dir / "condensed_silicon_thermal_expansion_gruneisen_two_branch_model.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "T_K",
                "alpha_obs_1e-8_per_K",
                "alpha_pred_1e-8_per_K",
                "sigma_fit_1e-8_per_K",
                "Cv1_J_per_molK",
                "Cv2_J_per_molK",
                "alpha1_1e-8_per_K",
                "alpha2_1e-8_per_K",
                "residual_1e-8_per_K",
                "z",
            ],
        )
        w.writeheader()
        for t, ao, ap, s1e8, c1, c2, a1c, a2c, r1e8, z in zip(
            temps,
            alpha_obs_1e8,
            alpha_pred_1e8,
            sigma_fit_1e8,
            cv1,
            cv2_best,
            contrib1_1e8,
            contrib2_1e8,
            residual_1e8,
            z_score,
        ):
            w.writerow(
                {
                    "T_K": float(t),
                    "alpha_obs_1e-8_per_K": float(ao),
                    "alpha_pred_1e-8_per_K": float(ap),
                    "sigma_fit_1e-8_per_K": float(s1e8),
                    "Cv1_J_per_molK": float(c1),
                    "Cv2_J_per_molK": float(c2),
                    "alpha1_1e-8_per_K": float(a1c),
                    "alpha2_1e-8_per_K": float(a2c),
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
        label="Two-branch Debye–Grüneisen: α≈A1·Cv(θ1)+A2·Cv(θ2)",
    )
    ax1.plot(temps, contrib1_1e8, color="#1f77b4", lw=1.2, ls="--", alpha=0.7, label="branch-1 contrib")
    ax1.plot(temps, contrib2_1e8, color="#ff7f0e", lw=1.2, ls="--", alpha=0.7, label="branch-2 contrib")
    ax1.axhline(0.0, color="#666666", lw=1.0, alpha=0.6)

    # 条件分岐: `zero_obs is not None` を満たす経路を評価する。
    if zero_obs is not None:
        ax1.axvline(float(zero_obs["x_cross"]), color="#999999", lw=1.0, ls=":", alpha=0.7)

    # 条件分岐: `zero_pred is not None` を満たす経路を評価する。

    if zero_pred is not None:
        ax1.axvline(float(zero_pred["x_cross"]), color="#999999", lw=1.0, ls="--", alpha=0.7)

    ax1.set_ylabel("α(T) (10^-8 / K)")
    ax1.set_title("Silicon thermal expansion: two-branch Debye–Grüneisen ansatz check")
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
    out_png = out_dir / "condensed_silicon_thermal_expansion_gruneisen_two_branch_model.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    out_metrics = out_dir / "condensed_silicon_thermal_expansion_gruneisen_two_branch_model_metrics.json"
    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.14.10",
                "inputs": {
                    "silicon_thermal_expansion_extracted_values": {"path": str(alpha_src), "sha256": _sha256(alpha_src)},
                    "silicon_janaf_extracted_values": (
                        {"path": str(root / "data/quantum/sources/nist_janaf_silicon_si/extracted_values.json")}
                        if theta_from_metrics is None
                        else None
                    ),
                    "theta1_source": (
                        {"kind": "frozen_metrics", "path": "output/public/quantum/condensed_silicon_heat_capacity_debye_baseline_metrics.json"}
                        if theta_from_metrics is not None
                        else {"kind": "refit_from_janaf", "fit_range_K": [100.0, 300.0]}
                    ),
                },
                "model": {
                    "name": "Two-branch Debye–Grüneisen (theta1 frozen; theta2 scanned; A1/A2 weighted LS)",
                    "theta1_K": float(theta1),
                    "theta2_K": float(theta2_best),
                    "A1_mol_per_J": float(a1_best),
                    "A2_mol_per_J": float(a2_best),
                    "fit_range_T_K": {"min": float(fit_min_k), "max": float(fit_max_k)},
                    "theta2_scan": {"lo_K": 10.0, "hi_K": 2000.0, "n": int(len(theta2_grid))},
                },
                "diagnostics": {
                    "alpha_sign_change_T_K": {
                        "obs": (None if zero_obs is None else float(zero_obs["x_cross"])),
                        "pred": (None if zero_pred is None else float(zero_pred["x_cross"])),
                    },
                    "sign_mismatch_fit_range_n": int(sign_mismatch_fit),
                    "fit_range": {
                        "n": int(n_z_fit),
                        "max_abs_z": float(max_abs_z_fit),
                        "rms_z": float(rms_z_fit),
                        "reduced_chi2": float(red_chi2),
                        "exceed_3sigma_n": int(exceed_3sigma_fit),
                    },
                    "sigma_fit_1e-8_per_K": {"lt_T_K": float(t_sigma_split), "lt": float(sigma_lt_1e8), "ge": float(sigma_ge_1e8)},
                },
                "falsification": {
                    "reject_if_sign_mismatch_fit_range_n_gt": 0,
                    "reject_if_max_abs_z_ge": 3.0,
                    "reject_if_reduced_chi2_ge": 2.0,
                    "rejected": bool(rejected),
                    "notes": [
                        "A two-branch model represents mode-dependent Grüneisen parameters (some modes may have negative γ at low T).",
                        "The second Debye temperature θ2 controls when the second branch contributes appreciably.",
                        "Acceptance is evaluated against the NIST TRC curve-fit standard-error scale σ_fit (proxy for data-level accuracy).",
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

    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_metrics}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()

