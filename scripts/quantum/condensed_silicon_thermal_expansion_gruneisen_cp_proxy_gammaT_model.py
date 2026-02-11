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
class CpPoint:
    t_k: float
    cp: float


def _load_janaf_cp_points(*, root: Path) -> list[CpPoint]:
    src = root / "data" / "quantum" / "sources" / "nist_janaf_silicon_si" / "extracted_values.json"
    if not src.exists():
        raise SystemExit(
            f"[fail] missing: {src}\n"
            "Run: python -B scripts/quantum/fetch_silicon_janaf_sources.py"
        )
    obj = _read_json(src)
    pts = obj.get("points")
    if not isinstance(pts, list) or not pts:
        raise SystemExit(f"[fail] missing points list: {src}")

    out: list[CpPoint] = []
    for p in pts:
        if not isinstance(p, dict):
            continue
        if str(p.get("phase")) != "solid":
            continue
        t = p.get("T_K")
        cp = p.get("Cp_J_per_molK")
        if not isinstance(t, (int, float)) or not isinstance(cp, (int, float)):
            continue
        out.append(CpPoint(t_k=float(t), cp=float(cp)))

    out = sorted(out, key=lambda x: x.t_k)
    if len(out) < 8:
        raise SystemExit(f"[fail] too few solid Cp points: n={len(out)}")
    return out


def _interp_linear(points: list[CpPoint], t: float) -> float:
    if not points:
        return float("nan")
    x = float(t)
    if x <= float(points[0].t_k):
        return float(points[0].cp)
    if x >= float(points[-1].t_k):
        return float(points[-1].cp)

    lo = 0
    hi = len(points) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if float(points[mid].t_k) <= x:
            lo = mid
        else:
            hi = mid

    p0 = points[lo]
    p1 = points[hi]
    x0 = float(p0.t_k)
    x1 = float(p1.t_k)
    y0 = float(p0.cp)
    y1 = float(p1.cp)
    if x1 == x0:
        return y0
    w = (x - x0) / (x1 - x0)
    return float(y0 + (y1 - y0) * w)


def _load_ioffe_bulk_modulus_model(*, root: Path) -> dict[str, Any]:
    src = root / "data" / "quantum" / "sources" / "ioffe_silicon_mechanical_properties" / "extracted_values.json"
    if not src.exists():
        raise SystemExit(
            f"[fail] missing: {src}\n"
            "Run: python -B scripts/quantum/fetch_silicon_elastic_constants_sources.py"
        )
    obj = _read_json(src)

    vals = obj.get("values")
    if not isinstance(vals, dict):
        raise SystemExit(f"[fail] invalid ioffe extracted_values.json: missing values dict: {src}")
    b_ref_1e11 = vals.get("bulk_modulus_from_C11_C12_1e11_dyn_cm2")
    if not isinstance(b_ref_1e11, (int, float)):
        raise SystemExit(f"[fail] invalid ioffe extracted_values.json: missing bulk modulus value: {src}")

    temp_dep = obj.get("temperature_dependence_linear")
    if not isinstance(temp_dep, dict):
        raise SystemExit(f"[fail] invalid ioffe extracted_values.json: missing temperature_dependence_linear dict: {src}")

    tr = temp_dep.get("T_range_K")
    if not isinstance(tr, dict):
        raise SystemExit(f"[fail] invalid ioffe extracted_values.json: missing T_range_K dict: {src}")
    t_lin_min = tr.get("min")
    t_lin_max = tr.get("max")
    if not isinstance(t_lin_min, (int, float)) or not isinstance(t_lin_max, (int, float)):
        raise SystemExit(f"[fail] invalid ioffe extracted_values.json: invalid T_range_K values: {src}")

    c11 = temp_dep.get("C11")
    c12 = temp_dep.get("C12")
    if not isinstance(c11, dict) or not isinstance(c12, dict):
        raise SystemExit(f"[fail] invalid ioffe extracted_values.json: missing C11/C12 linear dicts: {src}")

    c11_a = c11.get("intercept_1e11_dyn_cm2")
    c11_b = c11.get("slope_1e11_dyn_cm2_per_K")
    c12_a = c12.get("intercept_1e11_dyn_cm2")
    c12_b = c12.get("slope_1e11_dyn_cm2_per_K")
    for name, v in [("c11_a", c11_a), ("c11_b", c11_b), ("c12_a", c12_a), ("c12_b", c12_b)]:
        if not isinstance(v, (int, float)):
            raise SystemExit(f"[fail] invalid ioffe extracted_values.json: {name} is missing: {src}")

    return {
        "path": str(src),
        "sha256": _sha256(src),
        "b_ref_1e11_dyn_cm2": float(b_ref_1e11),
        "t_lin_min_K": float(t_lin_min),
        "t_lin_max_K": float(t_lin_max),
        "c11_intercept_1e11_dyn_cm2": float(c11_a),
        "c11_slope_1e11_dyn_cm2_per_K": float(c11_b),
        "c12_intercept_1e11_dyn_cm2": float(c12_a),
        "c12_slope_1e11_dyn_cm2_per_K": float(c12_b),
    }


def _bulk_modulus_pa(*, t_k: float, model: dict[str, Any]) -> float:
    """
    Piecewise bulk modulus B(T) in Pa, derived from Ioffe elastic constants.

    - For T < t_lin_min: use constant B_ref (room-temperature value).
    - For t_lin_min <= T <= t_lin_max: use linear C11(T), C12(T) and B=(C11+2*C12)/3,
      shifted to match B_ref at t_lin_min (continuity).
    """
    # 1 dyn/cm^2 = 0.1 Pa => 1e11 dyn/cm^2 = 1e10 Pa.
    t = float(t_k)
    b_ref_1e11 = float(model["b_ref_1e11_dyn_cm2"])
    t_lin_min = float(model["t_lin_min_K"])
    t_lin_max = float(model["t_lin_max_K"])

    if t < t_lin_min:
        b_1e11 = b_ref_1e11
    else:
        tt = min(max(t, t_lin_min), t_lin_max)
        c11_a = float(model["c11_intercept_1e11_dyn_cm2"])
        c11_b = float(model["c11_slope_1e11_dyn_cm2_per_K"])
        c12_a = float(model["c12_intercept_1e11_dyn_cm2"])
        c12_b = float(model["c12_slope_1e11_dyn_cm2_per_K"])

        def b_lin(t_use: float) -> float:
            c11 = float(c11_a + c11_b * t_use)
            c12 = float(c12_a + c12_b * t_use)
            return float((c11 + 2.0 * c12) / 3.0)

        b_at_min = b_lin(float(t_lin_min))
        offset = float(b_ref_1e11 - b_at_min)
        b_1e11 = float(b_lin(float(tt)) + offset)
    return float(b_1e11) * 1e10


def _load_silicon_molar_volume_m3_per_mol(*, root: Path) -> dict[str, Any]:
    src = root / "data" / "quantum" / "sources" / "nist_codata_2022_silicon_lattice" / "extracted_values.json"
    if not src.exists():
        raise SystemExit(
            f"[fail] missing: {src}\n"
            "Run: python -B scripts/quantum/fetch_silicon_lattice_sources.py"
        )
    obj = _read_json(src)
    constants = obj.get("constants")
    if not isinstance(constants, dict):
        raise SystemExit(f"[fail] invalid CODATA silicon lattice extracted_values.json: missing constants dict: {src}")
    asil = constants.get("asil")
    if not isinstance(asil, dict):
        raise SystemExit(f"[fail] invalid CODATA silicon lattice extracted_values.json: missing asil dict: {src}")
    a_m = asil.get("value_si")
    if not isinstance(a_m, (int, float)):
        raise SystemExit(f"[fail] invalid CODATA silicon lattice extracted_values.json: missing asil.value_si: {src}")

    # Diamond-cubic conventional cell contains 8 atoms.
    n_a = 6.02214076e23  # exact (SI definition)
    v_m = n_a * (float(a_m) ** 3) / 8.0
    return {"path": str(src), "sha256": _sha256(src), "a_m": float(a_m), "V_m3_per_mol": float(v_m)}


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Si thermal expansion ansatz check (Cp-proxy Grüneisen).\n"
            "Default: alpha≈A_eff(T)*Cp_proxy, A_eff(T)=A_inf*tanh((T−T0)/ΔT).\n"
            "Optional: include bulk modulus B(T) and molar volume V_m, and fit gamma(T)=gamma_inf*tanh((T−T0)/ΔT) "
            "via alpha≈gamma(T)*Cp_proxy/(B(T)*V_m)."
        )
    )
    ap.add_argument(
        "--use-bulk-modulus",
        action="store_true",
        help="Use B(T) from Ioffe elastic constants and fit gamma_inf (dimensionless) instead of A_inf (mol/J).",
    )
    args = ap.parse_args()

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

    theta_d = _theta_d_from_existing_metrics(root)
    if theta_d is None:
        raise SystemExit("[fail] missing frozen theta_D. Expected condensed_silicon_heat_capacity_debye_baseline_metrics.json")

    # Cp proxy from JANAF (solid) with Debye extrapolation below 100 K.
    janaf_cp_points = _load_janaf_cp_points(root=root)
    cp_at_100 = None
    for p in janaf_cp_points:
        if abs(float(p.t_k) - 100.0) < 1e-9:
            cp_at_100 = float(p.cp)
            break
    if cp_at_100 is None:
        raise SystemExit("[fail] JANAF Cp(100 K) missing; expected a point at 100 K")

    cv_debye_100 = _debye_cv_molar(t_k=100.0, theta_d_k=float(theta_d))
    if cv_debye_100 <= 0.0:
        raise SystemExit("[fail] invalid Debye Cv at 100 K")
    scale_lowT = float(cp_at_100 / cv_debye_100)

    def cp_proxy(t: float) -> tuple[float, str]:
        x = float(t)
        if x < 100.0:
            return float(scale_lowT * _debye_cv_molar(t_k=x, theta_d_k=float(theta_d))), "debye_scaled"
        return float(_interp_linear(janaf_cp_points, x)), "janaf_interp"

    # Build grid.
    temps = [float(t) for t in range(t_min, t_max + 1)]
    alpha_obs_1e8 = [_alpha_1e8_per_k(t_k=t, coeffs=coeffs) for t in temps]
    alpha_obs = [float(a) * 1e-8 for a in alpha_obs_1e8]
    sigma_fit_1e8 = [sigma_lt_1e8 if float(t) < t_sigma_split else sigma_ge_1e8 for t in temps]
    sigma_fit = [float(s) * 1e-8 for s in sigma_fit_1e8]

    cp_vals: list[float] = []
    cp_kind: list[str] = []
    for t in temps:
        v, k = cp_proxy(t)
        cp_vals.append(float(v))
        cp_kind.append(str(k))

    # Freeze the observed negative→positive sign-crossing T0 from alpha(T).
    zero = _infer_zero_crossing(temps, alpha_obs, prefer_neg_to_pos=True, min_x=50.0)
    if zero is None:
        raise SystemExit("[fail] could not infer negative→positive sign change temperature for alpha(T)")
    t0 = float(zero["x_cross"])

    use_bulk_modulus = bool(args.use_bulk_modulus)
    b_model = _load_ioffe_bulk_modulus_model(root=root) if use_bulk_modulus else None
    v_m = _load_silicon_molar_volume_m3_per_mol(root=root) if use_bulk_modulus else None

    # Model:
    #   (A) default: alpha(T) ≈ A_inf * tanh((T-T0)/ΔT) * Cp_proxy(T)
    #   (B) with bulk modulus: alpha(T) ≈ gamma_inf * tanh((T-T0)/ΔT) * Cp_proxy(T) / (B(T)*V_m)
    fit_min_k = 50.0
    fit_max_k = float(t_max)
    fit_idx = [i for i, t in enumerate(temps) if fit_min_k <= float(t) <= fit_max_k]
    if len(fit_idx) < 100:
        raise SystemExit(f"[fail] not enough fit points: n={len(fit_idx)} in [{fit_min_k},{fit_max_k}] K")

    delta_grid = _logspace(lo=5.0, hi=1200.0, n=500)
    best: dict[str, float] | None = None

    for delta in delta_grid:
        # Weighted LS for (A_inf or gamma_inf) with weights 1/sigma^2.
        num = 0.0
        den = 0.0
        for i in fit_idx:
            sig = float(sigma_fit[i])
            if sig <= 0.0:
                continue
            w = 1.0 / (sig * sig)
            t = float(temps[i])
            base = math.tanh((t - t0) / float(delta)) * float(cp_vals[i])
            if use_bulk_modulus:
                assert b_model is not None
                assert v_m is not None
                inv_bv = 1.0 / (float(_bulk_modulus_pa(t_k=t, model=b_model)) * float(v_m["V_m3_per_mol"]))
                x = float(base) * float(inv_bv)
            else:
                x = float(base)
            y = float(alpha_obs[i])
            num += w * x * y
            den += w * x * x
        if den <= 0.0:
            continue
        amp = float(num / den)
        if not math.isfinite(amp):
            continue

        sse = 0.0
        for i in fit_idx:
            sig = float(sigma_fit[i])
            if sig <= 0.0:
                continue
            w = 1.0 / (sig * sig)
            t = float(temps[i])
            base = math.tanh((t - t0) / float(delta)) * float(cp_vals[i])
            if use_bulk_modulus:
                assert b_model is not None
                assert v_m is not None
                inv_bv = 1.0 / (float(_bulk_modulus_pa(t_k=t, model=b_model)) * float(v_m["V_m3_per_mol"]))
                x = float(base) * float(inv_bv)
            else:
                x = float(base)
            r = float(alpha_obs[i]) - amp * x
            sse += w * r * r
        if best is None or sse < float(best["sse"]):
            best = {"delta_k": float(delta), "amp": float(amp), "sse": float(sse)}

    if best is None:
        raise SystemExit("[fail] ΔT scan failed")

    delta_best = float(best["delta_k"])
    amp_best = float(best["amp"])

    out_tag = (
        "condensed_silicon_thermal_expansion_gruneisen_cp_proxy_gammaT_bulkmodulus_model"
        if use_bulk_modulus
        else "condensed_silicon_thermal_expansion_gruneisen_cp_proxy_gammaT_model"
    )

    if use_bulk_modulus:
        assert b_model is not None
        assert v_m is not None
        gamma_model = [amp_best * math.tanh((float(t) - t0) / delta_best) for t in temps]
        inv_bv = [
            1.0 / (float(_bulk_modulus_pa(t_k=float(t), model=b_model)) * float(v_m["V_m3_per_mol"])) for t in temps
        ]
        alpha_pred = [float(g) * float(cp) * float(f) for g, cp, f in zip(gamma_model, cp_vals, inv_bv)]
    else:
        a_eff_model = [amp_best * math.tanh((float(t) - t0) / delta_best) for t in temps]
        alpha_pred = [float(a) * float(cp) for a, cp in zip(a_eff_model, cp_vals)]
    alpha_pred_1e8 = [float(a) / 1e-8 for a in alpha_pred]

    residual = [float(ap - ao) for ap, ao in zip(alpha_pred, alpha_obs)]
    residual_1e8 = [float(r) / 1e-8 for r in residual]
    z = [float("nan") if float(s) <= 0.0 else float(r) / float(s) for r, s in zip(residual, sigma_fit)]

    sign_mismatch_fit = 0
    exceed_3sigma_fit = 0
    max_abs_z_fit = 0.0
    sum_z2_fit = 0.0
    n_z_fit = 0
    for i in fit_idx:
        zi = float(z[i])
        if math.isfinite(zi):
            n_z_fit += 1
            sum_z2_fit += zi * zi
            max_abs_z_fit = max(max_abs_z_fit, abs(zi))
            if abs(zi) > 3.0:
                exceed_3sigma_fit += 1
        ao = float(alpha_obs[i])
        ap = float(alpha_pred[i])
        if ao != 0.0 and ap != 0.0 and ((ao > 0.0) != (ap > 0.0)):
            sign_mismatch_fit += 1

    dof = max(1, n_z_fit - 2)  # (delta, A_inf)
    red_chi2 = float(sum_z2_fit / dof) if dof > 0 else float("nan")

    rejected = (sign_mismatch_fit > 0) or (max_abs_z_fit >= 3.0) or (red_chi2 >= 2.0)

    # CSV
    out_csv = out_dir / f"{out_tag}.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "T_K",
            "alpha_obs_1e-8_per_K",
            "alpha_pred_1e-8_per_K",
            "sigma_fit_1e-8_per_K",
            "Cp_proxy_J_per_molK",
            "Cp_proxy_kind",
            "residual_1e-8_per_K",
            "z",
        ]
        if use_bulk_modulus:
            fieldnames.insert(6, "B_GPa")
            fieldnames.insert(7, "gamma_model_dimensionless")
        else:
            fieldnames.insert(6, "A_eff_model_mol_per_J")

        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if use_bulk_modulus:
            assert b_model is not None
            assert v_m is not None
            for t, ao, ap, s1e8, cp, kind, gmod, r1e8, zi in zip(
                temps, alpha_obs_1e8, alpha_pred_1e8, sigma_fit_1e8, cp_vals, cp_kind, gamma_model, residual_1e8, z
            ):
                b_gpa = float(_bulk_modulus_pa(t_k=float(t), model=b_model)) / 1e9
                w.writerow(
                    {
                        "T_K": float(t),
                        "alpha_obs_1e-8_per_K": float(ao),
                        "alpha_pred_1e-8_per_K": float(ap),
                        "sigma_fit_1e-8_per_K": float(s1e8),
                        "Cp_proxy_J_per_molK": float(cp),
                        "Cp_proxy_kind": str(kind),
                        "B_GPa": float(b_gpa),
                        "gamma_model_dimensionless": float(gmod),
                        "residual_1e-8_per_K": float(r1e8),
                        "z": (float(zi) if math.isfinite(float(zi)) else ""),
                    }
                )
        else:
            for t, ao, ap, s1e8, cp, kind, aeff, r1e8, zi in zip(
                temps, alpha_obs_1e8, alpha_pred_1e8, sigma_fit_1e8, cp_vals, cp_kind, a_eff_model, residual_1e8, z
            ):
                w.writerow(
                    {
                        "T_K": float(t),
                        "alpha_obs_1e-8_per_K": float(ao),
                        "alpha_pred_1e-8_per_K": float(ap),
                        "sigma_fit_1e-8_per_K": float(s1e8),
                        "Cp_proxy_J_per_molK": float(cp),
                        "Cp_proxy_kind": str(kind),
                        "A_eff_model_mol_per_J": float(aeff),
                        "residual_1e-8_per_K": float(r1e8),
                        "z": (float(zi) if math.isfinite(float(zi)) else ""),
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
        label=(
            "Cp-proxy + B(T): α≈γ_inf·tanh((T−T0)/ΔT)·Cp/(B·V)"
            if use_bulk_modulus
            else "Cp-proxy Grüneisen: α≈A_inf·tanh((T−T0)/ΔT)·Cp"
        ),
    )
    ax1.axhline(0.0, color="#666666", lw=1.0, alpha=0.6)
    ax1.axvline(t0, color="#999999", lw=1.0, ls="--", alpha=0.8)
    ax1.set_ylabel("α(T) (10^-8 / K)")
    ax1.set_title("Silicon thermal expansion: Cp-proxy Grüneisen (tanh γ(T)) ansatz check")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best", fontsize=8)

    ax2.plot(temps, z, color="#000000", lw=1.2, alpha=0.9, label="z = (α_pred−α_obs)/σ_fit")
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
                "step": ("7.14.17" if use_bulk_modulus else "7.14.13"),
                "inputs": {
                    "silicon_thermal_expansion_extracted_values": {"path": str(alpha_src), "sha256": _sha256(alpha_src)},
                    "silicon_janaf_extracted_values": {
                        "path": str(root / "data/quantum/sources/nist_janaf_silicon_si/extracted_values.json"),
                        "sha256": _sha256(root / "data/quantum/sources/nist_janaf_silicon_si/extracted_values.json"),
                    },
                    "theta_d_source": {"kind": "frozen_metrics", "path": "output/public/quantum/condensed_silicon_heat_capacity_debye_baseline_metrics.json"},
                    "ioffe_bulk_modulus_model": (b_model if use_bulk_modulus else None),
                    "silicon_molar_volume": (v_m if use_bulk_modulus else None),
                },
                "cp_proxy": {
                    "definition": "Cp_proxy(T)=Cp_JANAF_interp(T) for T>=100K; Cp_proxy(T)=scale*Cv_Debye(T;theta_D) for T<100K; scale=Cp_JANAF(100)/Cv_Debye(100)",
                    "scale_lowT": float(scale_lowT),
                    "join_T_K": 100.0,
                },
                "model": {
                    "name": (
                        "Cp-proxy + bulk modulus: alpha≈gamma(T)*Cp_proxy/(B(T)*V_m); gamma(T)=gamma_inf*tanh((T−T0)/ΔT)"
                        if use_bulk_modulus
                        else "Cp-proxy Grüneisen with tanh γ(T) (A_eff(T)=A_inf·tanh((T−T0)/ΔT))"
                    ),
                    "theta_D_K": float(theta_d),
                    "A_inf_mol_per_J": (None if use_bulk_modulus else float(amp_best)),
                    "gamma_inf_dimensionless": (float(amp_best) if use_bulk_modulus else None),
                    "T0_sign_change_frozen_K": float(t0),
                    "delta_T_K": float(delta_best),
                    "fit_range_T_K": {"min": float(fit_min_k), "max": float(fit_max_k)},
                    "delta_scan": {"lo_K": 5.0, "hi_K": 1200.0, "n": int(len(delta_grid))},
                },
                "diagnostics": {
                    "alpha_sign_change_T0_K": float(t0),
                    "fit_range": {
                        "n": int(n_z_fit),
                        "max_abs_z": float(max_abs_z_fit),
                        "reduced_chi2": float(red_chi2),
                        "exceed_3sigma_n": int(exceed_3sigma_fit),
                        "sign_mismatch_n": int(sign_mismatch_fit),
                    },
                    "sigma_fit_1e-8_per_K": {"lt_T_K": float(t_sigma_split), "lt": float(sigma_lt_1e8), "ge": float(sigma_ge_1e8)},
                },
                "falsification": {
                    "reject_if_sign_mismatch_fit_range_n_gt": 0,
                    "reject_if_max_abs_z_ge": 3.0,
                    "reject_if_reduced_chi2_ge": 2.0,
                    "rejected": bool(rejected),
                    "notes": [
                        "This ansatz keeps the Grüneisen form alpha≈A_eff(T)*C(T) but replaces Debye Cv by a Cp proxy built from JANAF Cp points.",
                        "Passing would not be a derivation; it would only indicate that the remaining mismatch is largely in C(T) rather than in gamma(T).",
                        "Acceptance is evaluated against the NIST TRC curve-fit standard-error scale σ_fit (proxy).",
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

    if use_bulk_modulus:
        print(
            f"[ok] delta_T_K={delta_best:.3g} gamma_inf={amp_best:.3g} max_abs_z={max_abs_z_fit:.3g} red_chi2={red_chi2:.3g} rejected={rejected}"
        )
    else:
        print(
            f"[ok] delta_T_K={delta_best:.3g} A_inf={amp_best:.3g} max_abs_z={max_abs_z_fit:.3g} red_chi2={red_chi2:.3g} rejected={rejected}"
        )
    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_metrics}")


if __name__ == "__main__":
    main()
