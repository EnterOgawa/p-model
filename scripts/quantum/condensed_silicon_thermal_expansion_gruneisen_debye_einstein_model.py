from __future__ import annotations

import argparse
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
    if x <= 0.0:
        return 0.0
    em1 = math.expm1(x)
    if em1 == 0.0:
        return 0.0
    inv = 1.0 / em1
    return (x**4) * (inv + inv * inv)


def _simpson_integrate(f, a: float, b: float, n: int) -> float:
    if n < 2:
        n = 2
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
    if t_k <= 0.0 or theta_d_k <= 0.0:
        return 0.0

    r = 8.314462618
    y = theta_d_k / t_k

    # For x > ~50, integrand ~ x^4 e^{-x} is negligible.
    y_eff = min(float(y), 50.0)
    n = int(max(800, 40 * y_eff))
    integral = _simpson_integrate(_debye_integrand, 0.0, y_eff, n)
    return 9.0 * r * ((t_k / theta_d_k) ** 3) * integral


def _einstein_cv_molar(*, t_k: float, theta_e_k: float, dof: float = 3.0) -> float:
    """
    Einstein heat capacity Cv for an oscillator branch, per mole.
    dof=3 corresponds to 3 modes per atom (normalization).
    """
    if t_k <= 0.0 or theta_e_k <= 0.0:
        return 0.0

    r = 8.314462618
    x = float(theta_e_k) / float(t_k)
    # Avoid overflow in exp for x >> 1.
    if x > 700.0:
        return 0.0
    ex = math.exp(x)
    denom = ex - 1.0
    if denom == 0.0:
        return 0.0
    return float(dof) * r * (x * x) * ex / (denom * denom)


def _theta_d_from_existing_metrics(root: Path) -> Optional[float]:
    m = root / "output" / "public" / "quantum" / "condensed_silicon_heat_capacity_debye_baseline_metrics.json"
    if not m.exists():
        return None
    try:
        obj = _read_json(m)
    except Exception:
        return None
    fit = obj.get("fit") if isinstance(obj.get("fit"), dict) else None
    if not isinstance(fit, dict):
        return None
    v = fit.get("theta_D_K")
    if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0:
        return float(v)
    return None


@dataclass(frozen=True)
class DebyeFitPoint:
    t_k: float
    cp_obs: float


def _golden_section_minimize(f, lo: float, hi: float, *, tol: float = 1e-6, max_iter: int = 120) -> tuple[float, float]:
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a = float(lo)
    b = float(hi)
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if abs(b - a) <= tol * (abs(c) + abs(d) + 1.0):
            break
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


def _fit_theta_d_from_janaf(root: Path) -> tuple[float, list[DebyeFitPoint]]:
    src = root / "data" / "quantum" / "sources" / "nist_janaf_silicon_si" / "extracted_values.json"
    if not src.exists():
        raise SystemExit(
            f"[fail] missing: {src}\n"
            "Run: python -B scripts/quantum/fetch_silicon_janaf_sources.py"
        )
    obj = _read_json(src)
    points = obj.get("points")
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
    if not solid:
        raise SystemExit(f"[fail] no solid-phase points found in: {src}")

    fit_points = [
        DebyeFitPoint(t_k=float(p["T_K"]), cp_obs=float(p["Cp_J_per_molK"]))
        for p in solid
        if 100.0 <= float(p["T_K"]) <= 300.0
    ]
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
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    candidates: list[dict[str, float]] = []
    for i in range(len(xs) - 1):
        x0 = float(xs[i])
        x1 = float(xs[i + 1])
        if min_x is not None and max(x0, x1) < float(min_x):
            continue

        y0 = float(ys[i])
        y1 = float(ys[i + 1])
        if y0 == 0.0:
            candidates.append({"x0": x0, "x1": x0, "x_cross": x0})
            continue
        if y1 == 0.0:
            candidates.append({"x0": x1, "x1": x1, "x_cross": x1})
            continue

        # Detect sign changes.
        if (y0 < 0.0 and y1 > 0.0) or (y0 > 0.0 and y1 < 0.0):
            x_cross = x0 + (x1 - x0) * (-y0) / (y1 - y0)
            candidates.append({"x0": x0, "x1": x1, "x_cross": float(x_cross), "y0": y0, "y1": y1})

    if not candidates:
        return None

    if prefer_neg_to_pos:
        neg_to_pos = [c for c in candidates if float(c.get("y0", 0.0)) < 0.0 and float(c.get("y1", 0.0)) > 0.0]
        if neg_to_pos:
            return max(neg_to_pos, key=lambda c: float(c["x_cross"]))

    return max(candidates, key=lambda c: float(c["x_cross"]))


def _logspace(*, lo: float, hi: float, n: int) -> list[float]:
    if n < 2:
        return [float(lo)]
    if lo <= 0.0 or hi <= 0.0:
        raise ValueError("logspace requires positive lo/hi")
    l0 = math.log10(float(lo))
    l1 = math.log10(float(hi))
    return [10 ** (l0 + (l1 - l0) * i / (n - 1)) for i in range(n)]


def _solve_2x2(*, a11: float, a12: float, a22: float, b1: float, b2: float) -> Optional[tuple[float, float]]:
    det = float(a11) * float(a22) - float(a12) * float(a12)
    if not math.isfinite(det) or abs(det) <= 1e-30:
        return None
    x1 = (float(b1) * float(a22) - float(b2) * float(a12)) / det
    x2 = (float(b2) * float(a11) - float(b1) * float(a12)) / det
    if not (math.isfinite(x1) and math.isfinite(x2)):
        return None
    return float(x1), float(x2)


def _solve_3x3(
    *,
    a11: float,
    a12: float,
    a13: float,
    a22: float,
    a23: float,
    a33: float,
    b1: float,
    b2: float,
    b3: float,
) -> Optional[tuple[float, float, float]]:
    # Symmetric 3x3 solve via Gauss-Jordan elimination (with partial pivoting).
    m = [
        [float(a11), float(a12), float(a13), float(b1)],
        [float(a12), float(a22), float(a23), float(b2)],
        [float(a13), float(a23), float(a33), float(b3)],
    ]

    for col in range(3):
        pivot = col
        best = abs(m[col][col])
        for r in range(col + 1, 3):
            v = abs(m[r][col])
            if v > best:
                best = v
                pivot = r
        if not math.isfinite(best) or best <= 1e-30:
            return None
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]

        pv = m[col][col]
        if not math.isfinite(pv) or abs(pv) <= 1e-30:
            return None
        inv = 1.0 / pv
        for j in range(col, 4):
            m[col][j] *= inv

        for r in range(3):
            if r == col:
                continue
            factor = m[r][col]
            if factor == 0.0:
                continue
            for j in range(col, 4):
                m[r][j] -= factor * m[col][j]

    x1 = float(m[0][3])
    x2 = float(m[1][3])
    x3 = float(m[2][3])
    if not (math.isfinite(x1) and math.isfinite(x2) and math.isfinite(x3)):
        return None
    return x1, x2, x3


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Si thermal expansion: Debye+Einstein (Debye–Grüneisen) ansatz checks")
    p.add_argument(
        "--einstein-branches",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of Einstein (optical) branches (1=Step 7.14.11; 2=Step 7.14.14).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    root = _repo_root()
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Alpha(T) source (NIST TRC Cryogenics).
    alpha_src = root / "data" / "quantum" / "sources" / "nist_trc_silicon_thermal_expansion" / "extracted_values.json"
    if not alpha_src.exists():
        raise SystemExit(
            f"[fail] missing: {alpha_src}\n"
            "Run: python -B scripts/quantum/fetch_silicon_thermal_expansion_sources.py"
        )
    alpha_extracted = _read_json(alpha_src)
    coeffs_obj = alpha_extracted.get("coefficients")
    if not isinstance(coeffs_obj, dict):
        raise SystemExit(f"[fail] coefficients missing: {alpha_src}")
    coeffs = {str(k).lower(): float(v) for k, v in coeffs_obj.items()}
    missing = [k for k in "abcdefghijkl" if k not in coeffs]
    if missing:
        raise SystemExit(f"[fail] missing coefficients: {missing}")

    dr = alpha_extracted.get("data_range")
    if not isinstance(dr, dict):
        raise SystemExit(f"[fail] data_range missing: {alpha_src}")
    t_min = int(math.ceil(float(dr.get("t_min_k"))))
    t_max = int(math.floor(float(dr.get("t_max_k"))))
    if not (0 < t_min < t_max):
        raise SystemExit(f"[fail] invalid data_range: {dr}")

    fe = alpha_extracted.get("fit_error_relative_to_data")
    if not isinstance(fe, dict) or not isinstance(fe.get("lt"), dict) or not isinstance(fe.get("ge"), dict):
        raise SystemExit(f"[fail] fit_error_relative_to_data missing: {alpha_src}")
    t_sigma_split = float(fe["lt"].get("t_k", 50.0))
    sigma_lt_1e8 = float(fe["lt"].get("sigma_1e_8_per_k", 0.03))
    sigma_ge_1e8 = float(fe["ge"].get("sigma_1e_8_per_k", 0.5))

    # Theta_D: prefer the frozen baseline if present; otherwise refit from JANAF (reproducible fallback).
    theta_from_metrics = _theta_d_from_existing_metrics(root)
    theta_fit_points: list[DebyeFitPoint] = []
    if theta_from_metrics is not None:
        theta_d = float(theta_from_metrics)
    else:
        theta_d, theta_fit_points = _fit_theta_d_from_janaf(root)

    # Prepare grid.
    temps = [float(t) for t in range(t_min, t_max + 1)]
    alpha_obs_1e8 = [_alpha_1e8_per_k(t_k=t, coeffs=coeffs) for t in temps]
    alpha_obs = [float(a) * 1e-8 for a in alpha_obs_1e8]  # 1/K
    sigma_fit_1e8 = [sigma_lt_1e8 if float(t) < t_sigma_split else sigma_ge_1e8 for t in temps]
    sigma_fit = [float(s) * 1e-8 for s in sigma_fit_1e8]

    cv_d = [_debye_cv_molar(t_k=t, theta_d_k=theta_d) for t in temps]

    # Fit range: avoid the very-low-T wiggle (and huge weight from tiny sigma).
    fit_min_k = 50.0
    fit_max_k = float(t_max)
    fit_idx = [i for i, t in enumerate(temps) if fit_min_k <= float(t) <= fit_max_k]
    if len(fit_idx) < 100:
        raise SystemExit(f"[fail] not enough fit points: n={len(fit_idx)} in [{fit_min_k},{fit_max_k}] K")

    step_id = "7.14.11" if int(args.einstein_branches) == 1 else "7.14.14"
    out_tag = (
        "condensed_silicon_thermal_expansion_gruneisen_debye_einstein_model"
        if int(args.einstein_branches) == 1
        else "condensed_silicon_thermal_expansion_gruneisen_debye_einstein_two_branch_model"
    )

    theta_scan_meta: dict[str, object]
    model_name: str

    if int(args.einstein_branches) == 1:
        # Debye + Einstein (optical) mixture:
        #   alpha(T) ≈ A_D * Cv_D(T; theta_D) + A_E * Cv_E(T; theta_E)
        # theta_D is frozen, theta_E is scanned.
        theta_e_grid = _logspace(lo=10.0, hi=4000.0, n=900)
        best: dict[str, float] | None = None

        for theta_e in theta_e_grid:
            # Precompute Cv_E on the shared grid.
            cv_e = [_einstein_cv_molar(t_k=t, theta_e_k=float(theta_e), dof=3.0) for t in temps]

            s11 = 0.0
            s22 = 0.0
            s12 = 0.0
            b1 = 0.0
            b2 = 0.0
            for i in fit_idx:
                sig = float(sigma_fit[i])
                if sig <= 0.0:
                    continue
                w = 1.0 / (sig * sig)
                x1 = float(cv_d[i])
                x2 = float(cv_e[i])
                y = float(alpha_obs[i])
                s11 += w * x1 * x1
                s22 += w * x2 * x2
                s12 += w * x1 * x2
                b1 += w * x1 * y
                b2 += w * x2 * y

            sol = _solve_2x2(a11=s11, a12=s12, a22=s22, b1=b1, b2=b2)
            if sol is None:
                continue
            a_d, a_e = sol

            sse = 0.0
            for i in fit_idx:
                sig = float(sigma_fit[i])
                if sig <= 0.0:
                    continue
                w = 1.0 / (sig * sig)
                pred = float(a_d) * float(cv_d[i]) + float(a_e) * float(cv_e[i])
                r = float(alpha_obs[i]) - pred
                sse += w * r * r

            if best is None or float(sse) < float(best["sse"]):
                best = {"theta_e_k": float(theta_e), "a_d": float(a_d), "a_e": float(a_e), "sse": float(sse)}

        if best is None:
            raise SystemExit("[fail] theta_E scan failed")

        theta_e_best = float(best["theta_e_k"])
        theta_e2_best: float | None = None
        a_d_best = float(best["a_d"])
        a_e_best = float(best["a_e"])
        a_e2_best: float | None = None
        cv_e_best = [_einstein_cv_molar(t_k=t, theta_e_k=theta_e_best, dof=3.0) for t in temps]
        cv_e2_best: list[float] | None = None

        alpha_pred = [a_d_best * float(cd) + a_e_best * float(ce) for cd, ce in zip(cv_d, cv_e_best)]
        alpha_pred_1e8 = [float(a) / 1e-8 for a in alpha_pred]

        contrib_d = [a_d_best * float(c) for c in cv_d]
        contrib_e = [a_e_best * float(c) for c in cv_e_best]
        contrib_d_1e8 = [float(a) / 1e-8 for a in contrib_d]
        contrib_e_1e8 = [float(a) / 1e-8 for a in contrib_e]
        contrib_e2_1e8: list[float] | None = None

        theta_scan_meta = {"lo_K": 10.0, "hi_K": 4000.0, "n": int(len(theta_e_grid))}
        model_name = "Debye+Einstein Debye–Grüneisen mixture (theta_D frozen; theta_E scanned; A_D/A_E weighted LS)"
    else:
        # Debye + Einstein×2 mixture:
        #   alpha(T) ≈ A_D * Cv_D(T; theta_D)
        #          + A_E1 * Cv_E(T; theta_E1) + A_E2 * Cv_E(T; theta_E2)
        # theta_D is frozen; theta_E1/theta_E2 are scanned with ordering theta_E1 < theta_E2.
        theta_e_grid = _logspace(lo=10.0, hi=4000.0, n=160)
        cv_e_grid = [[_einstein_cv_molar(t_k=t, theta_e_k=float(theta_e), dof=3.0) for t in temps] for theta_e in theta_e_grid]
        best3: dict[str, float] | None = None
        tested = 0

        for j1, theta_e1 in enumerate(theta_e_grid):
            cv_e1 = cv_e_grid[j1]
            for j2 in range(j1 + 1, len(theta_e_grid)):
                cv_e2 = cv_e_grid[j2]
                tested += 1

                s11 = 0.0
                s22 = 0.0
                s33 = 0.0
                s12 = 0.0
                s13 = 0.0
                s23 = 0.0
                b1 = 0.0
                b2 = 0.0
                b3 = 0.0
                for i in fit_idx:
                    sig = float(sigma_fit[i])
                    if sig <= 0.0:
                        continue
                    w = 1.0 / (sig * sig)
                    x1 = float(cv_d[i])
                    x2 = float(cv_e1[i])
                    x3 = float(cv_e2[i])
                    y = float(alpha_obs[i])
                    s11 += w * x1 * x1
                    s22 += w * x2 * x2
                    s33 += w * x3 * x3
                    s12 += w * x1 * x2
                    s13 += w * x1 * x3
                    s23 += w * x2 * x3
                    b1 += w * x1 * y
                    b2 += w * x2 * y
                    b3 += w * x3 * y

                sol3 = _solve_3x3(a11=s11, a12=s12, a13=s13, a22=s22, a23=s23, a33=s33, b1=b1, b2=b2, b3=b3)
                if sol3 is None:
                    continue
                a_d, a_e1, a_e2 = sol3

                sse = 0.0
                for i in fit_idx:
                    sig = float(sigma_fit[i])
                    if sig <= 0.0:
                        continue
                    w = 1.0 / (sig * sig)
                    pred = float(a_d) * float(cv_d[i]) + float(a_e1) * float(cv_e1[i]) + float(a_e2) * float(cv_e2[i])
                    r = float(alpha_obs[i]) - pred
                    sse += w * r * r

                if best3 is None or float(sse) < float(best3["sse"]):
                    best3 = {
                        "theta_e1_k": float(theta_e1),
                        "theta_e2_k": float(theta_e_grid[j2]),
                        "a_d": float(a_d),
                        "a_e1": float(a_e1),
                        "a_e2": float(a_e2),
                        "sse": float(sse),
                        "j1": float(j1),
                        "j2": float(j2),
                    }

        if best3 is None:
            raise SystemExit("[fail] theta_E1/theta_E2 scan failed")

        theta_e_best = float(best3["theta_e1_k"])
        theta_e2_best = float(best3["theta_e2_k"])
        a_d_best = float(best3["a_d"])
        a_e_best = float(best3["a_e1"])
        a_e2_best = float(best3["a_e2"])

        j1_best = int(best3["j1"])
        j2_best = int(best3["j2"])
        cv_e_best = cv_e_grid[j1_best]
        cv_e2_best = cv_e_grid[j2_best]

        alpha_pred = [
            a_d_best * float(cd) + a_e_best * float(ce1) + float(a_e2_best) * float(ce2)
            for cd, ce1, ce2 in zip(cv_d, cv_e_best, cv_e2_best)
        ]
        alpha_pred_1e8 = [float(a) / 1e-8 for a in alpha_pred]

        contrib_d = [a_d_best * float(c) for c in cv_d]
        contrib_e = [a_e_best * float(c) for c in cv_e_best]
        contrib_e2 = [a_e2_best * float(c) for c in cv_e2_best]
        contrib_d_1e8 = [float(a) / 1e-8 for a in contrib_d]
        contrib_e_1e8 = [float(a) / 1e-8 for a in contrib_e]
        contrib_e2_1e8 = [float(a) / 1e-8 for a in contrib_e2]

        theta_scan_meta = {
            "lo_K": 10.0,
            "hi_K": 4000.0,
            "n": int(len(theta_e_grid)),
            "pairs_tested": int(tested),
            "ordering": "theta_E1 < theta_E2",
        }
        model_name = "Debye+Einstein×2 Debye–Grüneisen mixture (theta_D frozen; theta_E1/theta_E2 scanned; A_D/A_E1/A_E2 weighted LS)"

    residual = [float(ap - ao) for ap, ao in zip(alpha_pred, alpha_obs)]
    residual_1e8 = [float(r) / 1e-8 for r in residual]
    z_score = [float("nan") if float(s) <= 0.0 else float(r) / float(s) for r, s in zip(residual, sigma_fit)]

    # Diagnostics.
    sign_mismatch_fit = 0
    for i in fit_idx:
        ao = float(alpha_obs[i])
        ap = float(alpha_pred[i])
        if ao == 0.0 or ap == 0.0:
            continue
        if (ao > 0.0) != (ap > 0.0):
            sign_mismatch_fit += 1

    exceed_3sigma_fit = 0
    max_abs_z_fit = 0.0
    sum_z2_fit = 0.0
    n_z_fit = 0
    for i in fit_idx:
        z = float(z_score[i])
        if not math.isfinite(z):
            continue
        n_z_fit += 1
        sum_z2_fit += z * z
        max_abs_z_fit = max(max_abs_z_fit, abs(z))
        if abs(z) > 3.0:
            exceed_3sigma_fit += 1

    rms_z_fit = float("nan") if n_z_fit <= 0 else math.sqrt(sum_z2_fit / n_z_fit)
    dof = max(1, n_z_fit - (5 if int(args.einstein_branches) == 2 else 3))  # (theta_E*, a_d, a_e*)
    red_chi2 = float(sum_z2_fit / dof) if dof > 0 else float("nan")

    zero_obs = _infer_zero_crossing(temps, alpha_obs, prefer_neg_to_pos=True, min_x=50.0)
    zero_pred = _infer_zero_crossing(temps, alpha_pred, prefer_neg_to_pos=True, min_x=50.0)

    rejected = (sign_mismatch_fit > 0) or (max_abs_z_fit >= 3.0) or (red_chi2 >= 2.0)

    # CSV
    out_csv = out_dir / f"{out_tag}.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        if int(args.einstein_branches) == 1:
            fieldnames = [
                "T_K",
                "alpha_obs_1e-8_per_K",
                "alpha_pred_1e-8_per_K",
                "sigma_fit_1e-8_per_K",
                "Cv_debye_J_per_molK",
                "Cv_einstein_J_per_molK",
                "alpha_debye_1e-8_per_K",
                "alpha_einstein_1e-8_per_K",
                "residual_1e-8_per_K",
                "z",
            ]
        else:
            fieldnames = [
                "T_K",
                "alpha_obs_1e-8_per_K",
                "alpha_pred_1e-8_per_K",
                "sigma_fit_1e-8_per_K",
                "Cv_debye_J_per_molK",
                "Cv_einstein1_J_per_molK",
                "Cv_einstein2_J_per_molK",
                "alpha_debye_1e-8_per_K",
                "alpha_einstein1_1e-8_per_K",
                "alpha_einstein2_1e-8_per_K",
                "residual_1e-8_per_K",
                "z",
            ]

        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if int(args.einstein_branches) == 1:
            for t, ao, ap, s1e8, cd, ce, ad, ae, r1e8, z in zip(
                temps,
                alpha_obs_1e8,
                alpha_pred_1e8,
                sigma_fit_1e8,
                cv_d,
                cv_e_best,
                contrib_d_1e8,
                contrib_e_1e8,
                residual_1e8,
                z_score,
            ):
                w.writerow(
                    {
                        "T_K": float(t),
                        "alpha_obs_1e-8_per_K": float(ao),
                        "alpha_pred_1e-8_per_K": float(ap),
                        "sigma_fit_1e-8_per_K": float(s1e8),
                        "Cv_debye_J_per_molK": float(cd),
                        "Cv_einstein_J_per_molK": float(ce),
                        "alpha_debye_1e-8_per_K": float(ad),
                        "alpha_einstein_1e-8_per_K": float(ae),
                        "residual_1e-8_per_K": float(r1e8),
                        "z": (float(z) if math.isfinite(float(z)) else ""),
                    }
                )
        else:
            if cv_e2_best is None or contrib_e2_1e8 is None:
                raise SystemExit("[fail] internal: missing Einstein2 arrays")
            for t, ao, ap, s1e8, cd, ce1, ce2, ad, ae1, ae2, r1e8, z in zip(
                temps,
                alpha_obs_1e8,
                alpha_pred_1e8,
                sigma_fit_1e8,
                cv_d,
                cv_e_best,
                cv_e2_best,
                contrib_d_1e8,
                contrib_e_1e8,
                contrib_e2_1e8,
                residual_1e8,
                z_score,
            ):
                w.writerow(
                    {
                        "T_K": float(t),
                        "alpha_obs_1e-8_per_K": float(ao),
                        "alpha_pred_1e-8_per_K": float(ap),
                        "sigma_fit_1e-8_per_K": float(s1e8),
                        "Cv_debye_J_per_molK": float(cd),
                        "Cv_einstein1_J_per_molK": float(ce1),
                        "Cv_einstein2_J_per_molK": float(ce2),
                        "alpha_debye_1e-8_per_K": float(ad),
                        "alpha_einstein1_1e-8_per_K": float(ae1),
                        "alpha_einstein2_1e-8_per_K": float(ae2),
                        "residual_1e-8_per_K": float(r1e8),
                        "z": (float(z) if math.isfinite(float(z)) else ""),
                    }
                )

    # Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(temps, alpha_obs_1e8, color="#d62728", lw=2.0, label="NIST TRC fit (obs) α(T)")
    model_label = (
        "Debye+Einstein: α≈A_D·Cv_D(θ_D)+A_E·Cv_E(θ_E)"
        if int(args.einstein_branches) == 1
        else "Debye+Einstein×2: α≈A_D·Cv_D(θ_D)+A_E1·Cv_E(θ_E1)+A_E2·Cv_E(θ_E2)"
    )
    ax1.plot(
        temps,
        alpha_pred_1e8,
        color="#1f77b4",
        lw=2.0,
        label=model_label,
    )
    ax1.plot(temps, contrib_d_1e8, color="#1f77b4", lw=1.2, ls="--", alpha=0.7, label="Debye contrib")
    ax1.plot(
        temps,
        contrib_e_1e8,
        color="#ff7f0e",
        lw=1.2,
        ls="--",
        alpha=0.7,
        label=("Einstein contrib" if int(args.einstein_branches) == 1 else "Einstein1 contrib"),
    )
    if int(args.einstein_branches) == 2 and contrib_e2_1e8 is not None:
        ax1.plot(temps, contrib_e2_1e8, color="#2ca02c", lw=1.2, ls="--", alpha=0.7, label="Einstein2 contrib")
    ax1.axhline(0.0, color="#666666", lw=1.0, alpha=0.6)

    if zero_obs is not None:
        ax1.axvline(float(zero_obs["x_cross"]), color="#999999", lw=1.0, ls=":", alpha=0.7)
    if zero_pred is not None:
        ax1.axvline(float(zero_pred["x_cross"]), color="#999999", lw=1.0, ls="--", alpha=0.7)

    ax1.set_ylabel("α(T) (10^-8 / K)")
    ax1.set_title(
        "Silicon thermal expansion: Debye+Einstein (optical) ansatz check"
        if int(args.einstein_branches) == 1
        else "Silicon thermal expansion: Debye+Einstein×2 (two optical branches) ansatz check"
    )
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
    out_png = out_dir / f"{out_tag}.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    out_metrics = out_dir / f"{out_tag}_metrics.json"
    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": step_id,
                "inputs": {
                    "silicon_thermal_expansion_extracted_values": {"path": str(alpha_src), "sha256": _sha256(alpha_src)},
                    "silicon_janaf_extracted_values": (
                        {"path": str(root / "data/quantum/sources/nist_janaf_silicon_si/extracted_values.json")}
                        if theta_from_metrics is None
                        else None
                    ),
                    "theta_d_source": (
                        {"kind": "frozen_metrics", "path": "output/public/quantum/condensed_silicon_heat_capacity_debye_baseline_metrics.json"}
                        if theta_from_metrics is not None
                        else {"kind": "refit_from_janaf", "fit_range_K": [100.0, 300.0]}
                    ),
                },
                "model": (
                    {
                        "name": model_name,
                        "einstein_branches": int(args.einstein_branches),
                        "theta_D_K": float(theta_d),
                        "theta_E_K": float(theta_e_best),
                        "theta_E2_K": theta_e2_best,
                        "A_D_mol_per_J": float(a_d_best),
                        "A_E_mol_per_J": float(a_e_best),
                        "A_E2_mol_per_J": a_e2_best,
                        "fit_range_T_K": {"min": float(fit_min_k), "max": float(fit_max_k)},
                        "theta_E_scan": theta_scan_meta,
                    }
                    if int(args.einstein_branches) == 1
                    else {
                        "name": model_name,
                        "einstein_branches": int(args.einstein_branches),
                        "theta_D_K": float(theta_d),
                        "theta_E1_K": float(theta_e_best),
                        "theta_E2_K": float(theta_e2_best or 0.0),
                        "A_D_mol_per_J": float(a_d_best),
                        "A_E1_mol_per_J": float(a_e_best),
                        "A_E2_mol_per_J": float(a_e2_best or 0.0),
                        "fit_range_T_K": {"min": float(fit_min_k), "max": float(fit_max_k)},
                        "theta_E_scan": theta_scan_meta,
                    }
                ),
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
                        "Debye+Einstein provides a minimal proxy for adding an optical phonon branch on top of an acoustic Debye baseline.",
                        "Mode-dependent Grüneisen signs can be represented by allowing A_E to be negative while A_D is positive (or vice versa).",
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


if __name__ == "__main__":
    main()
