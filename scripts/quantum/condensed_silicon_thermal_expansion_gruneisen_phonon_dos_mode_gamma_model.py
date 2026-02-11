from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_ROOT = _repo_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse the same observed α(T) implementation (NIST TRC fit) and linear-algebra helpers
# as the Debye+Einstein steps, to keep comparisons consistent.
from scripts.quantum.condensed_silicon_thermal_expansion_gruneisen_debye_einstein_model import (  # noqa: E402
    _alpha_1e8_per_k,
    _infer_zero_crossing,
    _read_json,
    _solve_2x2,
    _solve_3x3,
)


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    n_a = 6.022_140_76e23  # exact (SI definition)
    v_m = n_a * (float(a_m) ** 3) / 8.0
    return {"path": str(src), "sha256": _sha256(src), "a_m": float(a_m), "V_m3_per_mol": float(v_m)}


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Si thermal expansion: phonon DOS constrained two-group Gruneisen check "
            "(mode-dependent gamma proxy via DOS split at half-integral)."
        )
    )
    p.add_argument(
        "--dos-mode",
        choices=["static_omega", "kim2015_fig1_energy"],
        default="static_omega",
        help=(
            "How to build Cv(T) basis from a phonon DOS. "
            "'static_omega' uses a single omega-D(omega) proxy and optional omega(T) scaling; "
            "'kim2015_fig1_energy' uses temperature-dependent g_T(eps) digitized from Kim et al. (2015) Fig.1 "
            "and interpolates in T with a fixed energy grid."
        ),
    )
    p.add_argument(
        "--dos-source-dir",
        default="hadley_si_phonon_dos",
        help="Directory name under data/quantum/sources/ containing extracted_values.json.",
    )
    p.add_argument(
        "--kim2015-fig1-source-dir",
        default="caltechauthors_kim2015_prb91_014307_si_phonon_anharmonicity",
        help=(
            "Directory name under data/quantum/sources/ containing fig1_digitized_dos.json "
            "(generated by scripts/quantum/extract_silicon_phonon_dos_kim2015_fig1_digitize.py). "
            "Used when --dos-mode=kim2015_fig1_energy."
        ),
    )
    p.add_argument(
        "--kim2015-fig1-json-name",
        default="fig1_digitized_dos.json",
        help="Digitized Fig.1 DOS JSON filename under --kim2015-fig1-source-dir.",
    )
    p.add_argument(
        "--kim2015-fig1-egrid-step-mev",
        type=float,
        default=0.5,
        help="Energy grid step (meV) used to resample/interpolate g_T(eps) when --dos-mode=kim2015_fig1_energy.",
    )
    p.add_argument(
        "--dos-softening",
        choices=["none", "kim2015_linear_proxy"],
        default="none",
        help=(
            "Optional global phonon softening constraint applied to the full DOS (all modes). "
            "'kim2015_linear_proxy' freezes a linear omega scaling inferred from the accepted manuscript "
            "Kim et al., Phys. Rev. B 91, 014307 (2015) (INS-based phonon DOS vs temperature). "
            "This is treated as a fixed (no-fit) constraint candidate for Step 7.14.20."
        ),
    )
    p.add_argument(
        "--mode-softening",
        choices=["none", "kim2015_fig2_features", "kim2015_fig2_features_eq8_quasi"],
        default="none",
        help=(
            "Optional mode-dependent phonon softening constraint derived from Kim2015 Fig.2 (digitized). "
            "This provides fixed omega(T) scales for TA/LA/TO/LO proxies and is applied per DOS group "
            "(no fit parameters). "
            "'kim2015_fig2_features_eq8_quasi' applies an Eq.(8) quasiharmonic-only exponent to the digitized isobaric omega_scale, "
            "to reduce intrinsic (T|V) contamination using the Table I bar_gamma_P reference at 300 K. "
            "Incompatible with --dos-mode=kim2015_fig1_energy, --dos-softening, and --optical-softening."
        ),
    )
    p.add_argument(
        "--cv-omega-dependence",
        choices=["harmonic", "dU_numeric"],
        default="harmonic",
        help=(
            "How to compute Cv(T) from the DOS. "
            "'harmonic' uses the standard harmonic Cv factor (assumes omega is T-independent at fixed V). "
            "'dU_numeric' computes U(T) using the (possibly T-dependent) omega scale and takes a numerical dU/dT. "
            "This can include an anharmonicity proxy (omega(T) dependence) without adding fit parameters. "
            "Note: currently supported only for --dos-mode=static_omega and --optical-softening=none."
        ),
    )
    p.add_argument(
        "--kim2015-source-dir",
        default="osti_kim2015_prb91_014307_si_phonon_anharmonicity",
        help=(
            "Directory name under data/quantum/sources/ containing extracted_values.json with "
            "parsed_from_pdf.softening_proxy (generated by "
            "scripts/quantum/extract_silicon_phonon_anharmonicity_kim2015_softening_proxy.py). "
            "Used when --dos-softening=kim2015_linear_proxy."
        ),
    )
    p.add_argument(
        "--kim2015-fig2-source-dir",
        default="caltechauthors_kim2015_prb91_014307_si_phonon_anharmonicity",
        help=(
            "Directory name under data/quantum/sources/ containing fig2_digitized_softening.json "
            "(generated by scripts/quantum/extract_silicon_phonon_anharmonicity_kim2015_fig2_digitize.py). "
            "Used when --mode-softening=kim2015_fig2_features."
        ),
    )
    p.add_argument(
        "--kim2015-fig2-json-name",
        default="fig2_digitized_softening.json",
        help="Digitized Fig.2 softening JSON filename under --kim2015-fig2-source-dir.",
    )
    p.add_argument(
        "--groups",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help=(
            "Number of DOS groups ("
            "2=acoustic/optical split; "
            "3=TA/LA/optical by mode-count split; "
            "4=TA/LA/TO/LO by mode-count split)."
        ),
    )
    p.add_argument(
        "--enforce-signs",
        action="store_true",
        help="Enforce low<=0 and high>=0 coefficient signs during weighted LS (A or gamma depending on --use-bulk-modulus).",
    )
    p.add_argument(
        "--ridge-factor",
        type=float,
        default=0.0,
        help=(
            "Optional ridge regularization strength (>=0). "
            "When >0, solves weighted LS with (X^T W X + lambda I), where lambda = ridge_factor * median(diag(X^T W X)). "
            "This stabilizes ill-conditioned high-T fits (notably in holdout splits) without adding model degrees of freedom."
        ),
    )
    p.add_argument(
        "--delta-ridge-factor",
        type=float,
        default=0.0,
        help=(
            "Additional ridge regularization applied only to Delta_gamma in --gamma-trend fits (>=0). "
            "Uses lambda_delta = delta_ridge_factor * median(diag(X^T W X)). "
            "This can prevent Delta_gamma from blowing up when the low-T train range weakly constrains the trend."
        ),
    )
    p.add_argument(
        "--use-bulk-modulus",
        action="store_true",
        help=(
            "Use B(T) from Ioffe elastic constants and V_m from CODATA silicon lattice to fit gamma (dimensionless) "
            "in alpha~=sum gamma_i*Cv_i/(B(T)*V_m). By default this script fits effective A_i (mol/J) in alpha~=sum A_i*Cv_i."
        ),
    )
    p.add_argument(
        "--vm-thermal-expansion",
        action="store_true",
        help=(
            "When used with --use-bulk-modulus, apply a temperature-dependent molar-volume correction "
            "V_m(T)=V_ref*exp(3*integral alpha(T)dT) using the observed alpha(T) fit (t_ref=300 K). "
            "This changes the 1/(B(T)*V_m) scaling without adding fit parameters."
        ),
    )
    p.add_argument(
        "--gamma-trend",
        choices=["constant", "kim2015_fig2_softening_common", "kim2015_fig2_softening_common_centered300", "linear_T"],
        default="constant",
        help=(
            "Optional gamma(T) trend model (adds 1 extra fitted parameter). Requires --use-bulk-modulus. "
            "Modes: "
            "'kim2015_fig2_softening_common' requires --mode-softening=kim2015_fig2_features and uses g_i(T)=1-omega_scale_i(T) "
            "frozen from Kim2015 Fig.2 digitization per DOS group; "
            "'kim2015_fig2_softening_common_centered300' is the same but uses g_i(T) centered at 300 K, "
            "to reduce intercept/trend collinearity in holdout splits; "
            "'linear_T' uses a fixed g(T)=(T-300K)/T_max on the alpha(T) integer grid."
        ),
    )
    p.add_argument(
        "--gamma-omega-model",
        choices=["none", "linear_endpoints", "pwlinear_split", "pwlinear_split_leaky", "bernstein2"],
        default="none",
        help=(
            "Optional gamma(omega) model using fixed frequency-basis weights (reduces degrees of freedom). "
            "Requires --use-bulk-modulus and is currently supported only for --dos-mode=static_omega. "
            "Modes: 'linear_endpoints' fits 2 coefficients (low/high) with weights (1-w) and w; "
            "'pwlinear_split' fits 3 coefficients (low/mid/high) with a fixed knot at the acoustic/optical split; "
            "'pwlinear_split_leaky' is like 'pwlinear_split' but adds a small fixed overlap (see --gamma-omega-pwlinear-leak); "
            "'bernstein2' fits 3 coefficients (low/mid/high) using quadratic Bernstein weights (1-w)^2, 2w(1-w), w^2."
        ),
    )
    p.add_argument(
        "--gamma-omega-pwlinear-leak",
        type=float,
        default=0.05,
        help=(
            "Leak strength epsilon for --gamma-omega-model=pwlinear_split_leaky (0<=epsilon<1). "
            "This introduces a small overlap between low/mid/high weights while keeping a partition of unity."
        ),
    )
    p.add_argument(
        "--gamma-omega-pwlinear-warp-power",
        type=float,
        default=1.0,
        help=(
            "Optional warp for --gamma-omega-model=pwlinear_split_leaky: use w->w^p (p>0) "
            "before computing the piecewise-linear weights and leak profile. "
            "p=1 means no warp; p>1 localizes overlap to higher frequencies."
        ),
    )
    p.add_argument(
        "--gamma-omega-high-softening-delta",
        type=float,
        default=0.0,
        help=(
            "Optional fixed correction for gamma(omega) models: "
            "gamma_high(T)=gamma_high0 + delta*(1-scale_optical(T)), where scale_optical(T) is frozen from --mode-softening. "
            "This adds no fitted parameters (delta is fixed). Requires --gamma-omega-model!=none and --mode-softening=kim2015_fig2_features."
        ),
    )
    p.add_argument(
        "--gamma-omega-softening-delta",
        type=float,
        default=0.0,
        help=(
            "Optional fixed common correction for gamma(omega) models: "
            "gamma_i(T)=gamma_i0 + delta*g(T) for all omega-basis coefficients, where g(T)=(1-scale_optical(T)) centered at 300 K "
            "and scale_optical(T) is frozen from --mode-softening. "
            "This adds no fitted parameters (delta is fixed). Requires --gamma-omega-model!=none and --mode-softening=kim2015_fig2_features."
        ),
    )
    p.add_argument(
        "--gamma-omega-softening-fit",
        choices=["none", "common_centered300", "high_centered300", "by_label_centered300"],
        default="none",
        help=(
            "Fit (not fix) a single delta parameter for gamma(omega) models using a frozen g(T) shape from --mode-softening. "
            "Modes: 'common_centered300' fits gamma_i(T)=gamma_i(300K) + delta*g(T) for all omega-basis coefficients; "
            "'high_centered300' fits gamma_high(T)=gamma_high(300K) + delta*g(T) for the 'high' omega-basis coefficient only. "
            "'by_label_centered300' uses g_low(T) from acoustic scale, g_high(T) from optical scale, and g_mid(T) as their average. "
            "Here g(T)=(1-scale_optical(T)) centered at 300 K and scale_optical(T) is frozen from Kim2015 Fig.2. "
            "Mutually exclusive with --gamma-omega-*-softening-delta."
        ),
    )
    p.add_argument(
        "--optical-softening",
        choices=["none", "linear_fit", "raman_shape_fit"],
        default="none",
        help=(
            "Optional anharmonicity proxy: scale optical-group phonon frequencies by "
            "a clamped s(T). Modes: "
            "'linear_fit' uses s(T)=1-f*(T/T_max) with f fit by grid search; "
            "'raman_shape_fit' uses a fixed-shape curve g(T) digitized from a primary Raman omega(T) plot "
            "and fits only the amplitude f via grid search. "
            "These are diagnostics until a fully fixed (no-fit) omega(T) constraint is established."
        ),
    )
    p.add_argument(
        "--raman-source-dir",
        default="arxiv_2001_08458_si_raman_phonon_shift",
        help=(
            "Directory name under data/quantum/sources/ containing extracted_values.json with "
            "softening_shape_0to1(T). Used when --optical-softening=raman_shape_fit."
        ),
    )
    p.add_argument(
        "--softening-max-frac-at-tmax",
        type=float,
        default=0.06,
        help="Grid-search upper bound for f in s(T)=1-f*(T/T_max). Interpreted as fractional softening at T_max.",
    )
    p.add_argument(
        "--softening-step-frac",
        type=float,
        default=0.002,
        help="Grid step for f in s(T)=1-f*(T/T_max).",
    )
    p.add_argument(
        "--softening-min-scale",
        type=float,
        default=0.85,
        help="Clamp lower bound for the softening scale s(T) to avoid unphysical negative/too-small frequencies.",
    )
    return p.parse_args(argv)


def _median_positive(values: list[float]) -> float:
    xs = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0.0]
    if not xs:
        return 0.0
    xs.sort()
    n = int(len(xs))
    mid = n // 2
    if (n % 2) == 1:
        return float(xs[mid])
    return float(0.5 * (float(xs[mid - 1]) + float(xs[mid])))


def _trapz_xy(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("invalid arrays for trapz")
    s = 0.0
    for i in range(1, len(xs)):
        dx = float(xs[i]) - float(xs[i - 1])
        if dx <= 0.0:
            raise ValueError("xs must be strictly increasing")
        s += 0.5 * (float(ys[i - 1]) + float(ys[i])) * dx
    return float(s)


def _integrate_split_trapz(xs: list[float], ys: list[float], *, x_split: float) -> tuple[float, float]:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("invalid arrays for split trapz")

    s_lo = 0.0
    s_hi = 0.0
    for i in range(1, len(xs)):
        x0 = float(xs[i - 1])
        x1 = float(xs[i])
        y0 = float(ys[i - 1])
        y1 = float(ys[i])
        if x1 <= x_split:
            s_lo += 0.5 * (y0 + y1) * (x1 - x0)
            continue
        if x0 >= x_split:
            s_hi += 0.5 * (y0 + y1) * (x1 - x0)
            continue

        # Interval crosses x_split: linearly interpolate y at split.
        if x1 == x0:
            continue
        t = (float(x_split) - x0) / (x1 - x0)
        t = min(1.0, max(0.0, float(t)))
        y_split = y0 + t * (y1 - y0)

        s_lo += 0.5 * (y0 + y_split) * (float(x_split) - x0)
        s_hi += 0.5 * (y_split + y1) * (x1 - float(x_split))

    return float(s_lo), float(s_hi)


def _integrate_range_trapz(xs: list[float], ys: list[float], *, x_min: float, x_max: float) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("invalid arrays for range trapz")
    if float(x_max) <= float(x_min):
        return 0.0

    s = 0.0
    for i in range(1, len(xs)):
        x0 = float(xs[i - 1])
        x1 = float(xs[i])
        y0 = float(ys[i - 1])
        y1 = float(ys[i])
        if x1 <= x_min or x0 >= x_max:
            continue
        xa = max(x0, float(x_min))
        xb = min(x1, float(x_max))
        if xb <= xa:
            continue
        if x1 == x0:
            continue
        # Linear interpolation at boundaries (if clipped).
        t_a = (xa - x0) / (x1 - x0)
        t_b = (xb - x0) / (x1 - x0)
        ya = y0 + t_a * (y1 - y0)
        yb = y0 + t_b * (y1 - y0)
        s += 0.5 * (ya + yb) * (xb - xa)
    return float(s)


def _molar_volume_from_alpha_fit(
    *,
    temps_k: list[float],
    alpha_linear_per_k: list[float],
    v_ref_m3_per_mol: float,
    t_ref_k: float = 300.0,
) -> list[float]:
    """
    Construct a temperature-dependent molar volume V_m(T) from a (linear) thermal-expansion fit α(T).

    Uses the isotropic approximation:
      d ln V / dT = 3 α(T)
    so that, relative to a reference temperature T_ref:
      V(T) = V_ref * exp( 3 * ∫_{T_ref}^{T} α(T') dT' ).
    """
    if len(temps_k) != len(alpha_linear_per_k) or not temps_k:
        raise ValueError("invalid arrays for molar-volume integration")
    if not math.isfinite(v_ref_m3_per_mol) or v_ref_m3_per_mol <= 0.0:
        raise ValueError("invalid v_ref")
    try:
        idx_ref = temps_k.index(float(t_ref_k))
    except ValueError as e:
        raise ValueError(f"t_ref not in temps grid: {t_ref_k}") from e

    delta_ln_v = [0.0 for _ in temps_k]
    # Forward (T >= T_ref).
    for i in range(idx_ref, len(temps_k) - 1):
        t0 = float(temps_k[i])
        t1 = float(temps_k[i + 1])
        a0 = float(alpha_linear_per_k[i])
        a1 = float(alpha_linear_per_k[i + 1])
        dt = float(t1 - t0)
        if dt <= 0.0:
            continue
        delta_ln_v[i + 1] = float(delta_ln_v[i] + (3.0 * 0.5 * (a0 + a1) * dt))
    # Backward (T <= T_ref).
    for i in range(idx_ref, 0, -1):
        t0 = float(temps_k[i - 1])
        t1 = float(temps_k[i])
        a0 = float(alpha_linear_per_k[i - 1])
        a1 = float(alpha_linear_per_k[i])
        dt = float(t1 - t0)
        if dt <= 0.0:
            continue
        delta_ln_v[i - 1] = float(delta_ln_v[i] - (3.0 * 0.5 * (a0 + a1) * dt))

    return [float(v_ref_m3_per_mol * math.exp(float(d))) for d in delta_ln_v]


def _find_x_at_cum_fraction(xs: list[float], ys: list[float], *, frac: float) -> float:
    if not (0.0 < float(frac) < 1.0):
        raise ValueError("frac must be in (0,1)")
    total = _trapz_xy(xs, ys)
    target = float(frac) * float(total)
    cum = 0.0
    for i in range(1, len(xs)):
        x0 = float(xs[i - 1])
        x1 = float(xs[i])
        y0 = float(ys[i - 1])
        y1 = float(ys[i])
        dx = x1 - x0
        if dx <= 0.0:
            continue
        area = 0.5 * (y0 + y1) * dx
        if cum + area >= target and area > 0.0:
            # Interpolate within the trapezoid using a quadratic in t (same as fetch script).
            remaining = target - cum
            a = 0.5 * dx * (y1 - y0)
            b = dx * y0
            c = -remaining
            if abs(a) < 1e-30:
                t = 0.0 if abs(b) < 1e-30 else float(-c / b)
            else:
                disc = b * b - 4.0 * a * c
                disc = max(0.0, float(disc))
                t = float((-b + math.sqrt(disc)) / (2.0 * a))
            t = min(1.0, max(0.0, float(t)))
            return float(x0 + t * dx)
        cum += area
    return float(xs[-1])


def _interp_piecewise_linear(xs: list[float], ys: list[float], xq: list[float]) -> list[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("invalid arrays for interp")
    pairs = sorted(zip((float(x) for x in xs), (float(y) for y in ys)))
    x_sorted = np.asarray([p[0] for p in pairs], dtype=float)
    y_sorted = np.asarray([p[1] for p in pairs], dtype=float)
    xq_np = np.asarray([float(x) for x in xq], dtype=float)
    yq = np.interp(xq_np, x_sorted, y_sorted, left=float(y_sorted[0]), right=float(y_sorted[-1]))
    return [float(v) for v in yq.tolist()]


def _cv_factor(x: float) -> float:
    """
    Dimensionless Einstein mode heat capacity factor:
      f(x) = x^2 e^x / (e^x - 1)^2,  x = θ/T = ħω/(k_B T)
    """
    x = float(x)
    if x <= 0.0:
        return 0.0
    if x < 1e-6:
        return 1.0
    if x > 80.0:
        # For large x, f(x) ~ x^2 e^{-x}.
        return float(x * x * math.exp(-x))
    em1 = math.expm1(x)
    if em1 == 0.0:
        return 1.0
    ex = float(em1 + 1.0)
    return float((x * x * ex) / (em1 * em1))


def _cv_factor_np(x: np.ndarray) -> np.ndarray:
    """
    Vectorized version of _cv_factor for numpy arrays.

    f(x) = x^2 e^x / (e^x - 1)^2, with asymptotic handling for tiny/huge x.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)

    pos = x > 0.0
    if not np.any(pos):
        return out

    xp = x[pos]

    small = xp < 1e-6
    if np.any(small):
        tmp = out[pos]
        tmp[small] = 1.0
        out[pos] = tmp

    large = xp > 80.0
    if np.any(large):
        tmp = out[pos]
        tmp[large] = xp[large] * xp[large] * np.exp(-xp[large])
        out[pos] = tmp

    mid = ~(small | large)
    if np.any(mid):
        xm = xp[mid]
        em1 = np.expm1(xm)
        ex = em1 + 1.0
        out_mid = (xm * xm * ex) / (em1 * em1)
        tmp = out[pos]
        tmp[mid] = out_mid
        out[pos] = tmp

    return out


def _u_th_factor_np(x: np.ndarray) -> np.ndarray:
    """
    Thermal energy factor for a harmonic oscillator, excluding the 1/2 zero-point term.

    For x = ħω/(k_B T), the thermal energy per mode is:
      U_th = k_B T * (x / (exp(x) - 1)).
    This helper returns the dimensionless factor x/(exp(x)-1), evaluated stably via expm1.
    """
    x = np.asarray(x, dtype=float)
    den = np.expm1(x)
    out = np.ones_like(x, dtype=float)
    mask = np.abs(den) > 1e-18
    out[mask] = x[mask] / den[mask]
    return out


def _numeric_dydx(xs: list[float], ys: list[float]) -> list[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("invalid arrays for numeric derivative")
    out: list[float] = []
    n = int(len(xs))
    for i in range(n):
        if i == 0:
            dx = float(xs[1]) - float(xs[0])
            out.append(float("nan") if dx == 0.0 else float((float(ys[1]) - float(ys[0])) / dx))
        elif i == (n - 1):
            dx = float(xs[-1]) - float(xs[-2])
            out.append(float("nan") if dx == 0.0 else float((float(ys[-1]) - float(ys[-2])) / dx))
        else:
            dx = float(xs[i + 1]) - float(xs[i - 1])
            out.append(float("nan") if dx == 0.0 else float((float(ys[i + 1]) - float(ys[i - 1])) / dx))
    return out


def _interp_linear(x0: float, x1: float, y0: float, y1: float, x: float) -> float:
    if x1 == x0:
        return float(y0)
    t = (float(x) - float(x0)) / (float(x1) - float(x0))
    t = min(1.0, max(0.0, float(t)))
    return float(float(y0) + t * (float(y1) - float(y0)))


def _append_boundary_point(
    *, omega: np.ndarray, g_per_atom: np.ndarray, theta_k: np.ndarray, omega_boundary: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Build a tail array starting exactly at omega_boundary by inserting an interpolated
    point at the boundary and then including the existing grid from the first point >= boundary.

    Returns (omega_tail, g_tail, theta_tail, i0) where i0 is the original index of the first omega>=boundary.
    """
    w = np.asarray(omega, dtype=float)
    g = np.asarray(g_per_atom, dtype=float)
    th = np.asarray(theta_k, dtype=float)
    wb = float(omega_boundary)

    i0 = int(bisect.bisect_left(w.tolist(), wb))
    if i0 <= 0:
        return w, g, th, 0
    if i0 >= len(w):
        # Boundary at/above max; keep just a single point to avoid empty arrays.
        return np.array([wb], dtype=float), np.array([g[-1]], dtype=float), np.array([th[-1]], dtype=float), len(w) - 1

    g_b = _interp_linear(w[i0 - 1], w[i0], g[i0 - 1], g[i0], wb)
    # θ(ω)=ħω/kB is linear in ω; use the same scaling even if theta_k has small numeric noise.
    th_b = float(th[0] * (wb / w[0])) if w[0] > 0.0 else _interp_linear(w[i0 - 1], w[i0], th[i0 - 1], th[i0], wb)

    w_tail = np.concatenate((np.array([wb], dtype=float), w[i0:]), axis=0)
    g_tail = np.concatenate((np.array([g_b], dtype=float), g[i0:]), axis=0)
    th_tail = np.concatenate((np.array([th_b], dtype=float), th[i0:]), axis=0)
    return w_tail, g_tail, th_tail, i0


def _kim2015_linear_global_softening_scale(
    *,
    root: Path,
    source_dirname: str,
    temps_k: list[float],
) -> dict[str, object]:
    """
    Build a frozen global ω scaling function s(T) from the Kim+2015 accepted-manuscript cache.

    The proxy assumes a linear fractional energy shift between T_ref and T_max:
      scale(T) = 1 + s*(T - T_ref)/(T_max - T_ref)
    where s is the isobaric mean fractional energy shift parsed from the abstract.
    """
    src = root / "data" / "quantum" / "sources" / str(source_dirname) / "extracted_values.json"
    if not src.exists():
        raise SystemExit(
            f"[fail] missing: {src}\n"
            "Run:\n"
            "  python -B scripts/quantum/fetch_silicon_phonon_dos_sources.py --source osti_kim2015_prb91_014307\n"
            "  python -B scripts/quantum/extract_silicon_phonon_anharmonicity_kim2015_softening_proxy.py"
        )
    obj = _read_json(src)
    parsed = obj.get("parsed_from_pdf", {})
    if not isinstance(parsed, dict):
        raise SystemExit(f"[fail] parsed_from_pdf missing: {src}")
    proxy = parsed.get("softening_proxy", {})
    if not isinstance(proxy, dict):
        raise SystemExit(f"[fail] softening_proxy missing: {src}")

    t_ref = float(proxy.get("t_ref_K", float("nan")))
    t_max = float(proxy.get("t_max_K", float("nan")))
    s_at_tmax = float(proxy.get("fractional_energy_shift_at_t_max_isobaric", float("nan")))
    if not (math.isfinite(t_ref) and math.isfinite(t_max) and math.isfinite(s_at_tmax) and t_max > t_ref):
        raise SystemExit(f"[fail] invalid kim2015 proxy values: t_ref={t_ref}, t_max={t_max}, s={s_at_tmax}")

    scales: list[float] = []
    for t in temps_k:
        tt = float(t)
        frac = (tt - t_ref) / (t_max - t_ref)
        scale = 1.0 + float(s_at_tmax) * float(frac)
        scales.append(float(scale))

    return {
        "source": {"path": str(src), "sha256": _sha256(src)},
        "proxy": {
            "kind": "kim2015_linear_proxy",
            "t_ref_K": float(t_ref),
            "t_max_K": float(t_max),
            "fractional_energy_shift_at_t_max_isobaric": float(s_at_tmax),
        },
        "scales": scales,
        "notes": [
            "Global softening is applied as ω_eff(T)=scale(T)*ω to the Bose heat-capacity factor (all modes).",
            "Mode-weighting (DOS integral splits) remains fixed; only the ω scale changes.",
        ],
    }


def _kim2015_fig2_mode_softening_scales(
    *,
    root: Path,
    source_dirname: str,
    json_name: str,
    temps_k: list[float],
    groups: int,
    eq8_quasiharmonic: bool = False,
    alpha_300k_per_k: float | None = None,
) -> dict[str, object]:
    """
    Build fixed mode-dependent ω(T) scales from digitized Kim2015 Fig.2.

    The digitizer stores the feature softening as s(T)=-Δε/ε and omega_scale=1-s(T).
    Here we construct per-group scales compatible with the DOS split groups:
      - groups=2: acoustic vs optical (mode-count weighted average)
      - groups=3: TA, LA, optical (optical = TO/LO weighted average)
      - groups=4: TA, LA, TO, LO (TO from TO/LO feature; LO from LA/LO feature)

    This constraint is treated as fixed input (no fit parameters).
    """
    src = root / "data" / "quantum" / "sources" / str(source_dirname) / str(json_name)
    if not src.exists():
        raise SystemExit(
            f"[fail] missing: {src}\n"
            "Run:\n"
            "  python -B scripts/quantum/fetch_silicon_phonon_dos_sources.py --source caltechauthors_kim2015_prb91_014307\n"
            "  python -B scripts/quantum/extract_silicon_phonon_anharmonicity_kim2015_fig2_digitize.py"
        )
    obj = _read_json(src)

    series_obj = obj.get("series")
    derived_obj = obj.get("derived")
    if not isinstance(series_obj, dict) or not isinstance(derived_obj, dict):
        raise SystemExit(f"[fail] invalid fig2 digitized structure: {src}")

    def _rows_for(key: str) -> list[dict[str, float]]:
        s = series_obj.get(key)
        if not isinstance(s, dict) or not isinstance(s.get("rows"), list):
            raise SystemExit(f"[fail] missing series.{key}.rows: {src}")
        rows = []
        for r in s["rows"]:
            if not isinstance(r, dict):
                continue
            rows.append({"t_K": float(r["t_K"]), "omega_scale": float(r["omega_scale"])})
        if len(rows) < 2:
            raise SystemExit(f"[fail] too few rows for series {key}: n={len(rows)}")
        rows.sort(key=lambda rr: float(rr["t_K"]))
        return rows

    def _rows_for_derived(key: str) -> list[dict[str, float]]:
        d = derived_obj.get(key)
        if not isinstance(d, dict) or not isinstance(d.get("rows"), list):
            raise SystemExit(f"[fail] missing derived.{key}.rows: {src}")
        rows = []
        for r in d["rows"]:
            if not isinstance(r, dict):
                continue
            rows.append({"t_K": float(r["t_K"]), "omega_scale": float(r["omega_scale"])})
        if len(rows) < 2:
            raise SystemExit(f"[fail] too few rows for derived {key}: n={len(rows)}")
        rows.sort(key=lambda rr: float(rr["t_K"]))
        return rows

    ta_sq = _rows_for("TA_sq")
    ta_circ = _rows_for("TA_circ")
    la = _rows_for("LA_pent")
    lo_proxy = _rows_for("LA_LO_hex")
    to_proxy = _rows_for_derived("TO_LO_triangles")

    eq8_info: dict[str, object] | None = None
    if bool(eq8_quasiharmonic):
        alpha_300 = float(alpha_300k_per_k) if alpha_300k_per_k is not None else float("nan")
        if not (math.isfinite(alpha_300) and alpha_300 > 0.0):
            raise SystemExit("[fail] --mode-softening=kim2015_fig2_features_eq8_quasi requires a valid alpha_300K input")

        def _interp_y(rows: list[dict[str, float]], t0: float) -> float:
            ts = [float(r["t_K"]) for r in rows]
            ys = [float(r["omega_scale"]) for r in rows]
            return float(_interp_piecewise_linear(ts, ys, [float(t0)])[0])

        def _slope_y(rows: list[dict[str, float]], t0: float) -> float:
            # Piecewise-linear slope dy/dT at t0, clamped to end segments.
            ts = [float(r["t_K"]) for r in rows]
            ys = [float(r["omega_scale"]) for r in rows]
            if t0 <= ts[0]:
                i0 = 0
            elif t0 >= ts[-1]:
                i0 = max(0, len(ts) - 2)
            else:
                i0 = bisect.bisect_right(ts, float(t0)) - 1
                i0 = min(max(0, int(i0)), len(ts) - 2)
            t_a = float(ts[i0])
            t_b = float(ts[i0 + 1])
            y_a = float(ys[i0])
            y_b = float(ys[i0 + 1])
            dt = float(t_b - t_a)
            if dt <= 0.0:
                return 0.0
            return float((y_b - y_a) / dt)

        feat_rows = {
            "TA_sq": ta_sq,
            "TA_circ": ta_circ,
            "LA_pent": la,
            "LA_LO_hex": lo_proxy,
            "TO_LO_tri": to_proxy,
        }
        dlnw_dT_by_feat: dict[str, float] = {}
        for k, rows in feat_rows.items():
            y300 = _interp_y(rows, 300.0)
            dy_dT_300 = _slope_y(rows, 300.0)
            if not (math.isfinite(y300) and y300 > 0.0):
                raise SystemExit("[fail] invalid omega_scale at 300 K in Kim2015 Fig.2 digitized data")
            dlnw_dT_by_feat[k] = float(dy_dT_300 / y300)
        dlnw_dT_mean = float(sum(dlnw_dT_by_feat.values()) / float(len(dlnw_dT_by_feat)))
        if not (math.isfinite(dlnw_dT_mean) and dlnw_dT_mean != 0.0):
            raise SystemExit("[fail] invalid dln(omega)/dT from Kim2015 Fig.2 digitized data")

        # Eq.(8) at 300 K: (d ln omega / dT)_P = -bar_gamma_P * 3 alpha + (d ln omega / dT)_V.
        # Use Table I reference bar_gamma_P ≈ 0.98 to estimate the quasiharmonic fraction of the isobaric shift.
        bar_gamma_P_ref = 0.98
        quasiharmonic_dlnw_dT = float(-bar_gamma_P_ref * 3.0 * alpha_300)
        exponent = float(quasiharmonic_dlnw_dT / dlnw_dT_mean)
        if not (math.isfinite(exponent) and 0.0 < exponent < 1.0):
            raise SystemExit(
                f"[fail] invalid Eq8 quasiharmonic exponent: {exponent} (expected in (0,1)); "
                f"alpha_300K={alpha_300}, dlnw_dT_mean={dlnw_dT_mean}"
            )

        # Apply: ln omega_quasi = exponent * ln omega_isobaric => omega_scale_quasi = omega_scale^exponent.
        for rows in feat_rows.values():
            for r in rows:
                s0 = float(r["omega_scale"])
                if not (math.isfinite(s0) and s0 > 0.0):
                    continue
                r["omega_scale"] = float(s0**exponent)

        eq8_info = {
            "t_eval_K": 300.0,
            "alpha_300K_per_K": float(alpha_300),
            "bar_gamma_P_ref": float(bar_gamma_P_ref),
            "dlnomega_dT_isobaric_300K_mean_per_K": float(dlnw_dT_mean),
            "dlnomega_dT_quasiharmonic_300K_per_K": float(quasiharmonic_dlnw_dT),
            "exponent_quasiharmonic_fraction": float(exponent),
            "note": "omega_scale is adjusted as omega_scale^exponent to keep only the Eq.(8) quasiharmonic component at 300 K.",
        }

    def _interp_scale(rows: list[dict[str, float]]) -> list[float]:
        ts = [float(r["t_K"]) for r in rows]
        ys = [float(r["omega_scale"]) for r in rows]
        return _interp_piecewise_linear(ts, ys, [float(t) for t in temps_k])

    scale_ta_sq = _interp_scale(ta_sq)
    scale_ta_circ = _interp_scale(ta_circ)
    scale_ta = [0.5 * (float(a) + float(b)) for a, b in zip(scale_ta_sq, scale_ta_circ)]
    scale_la = _interp_scale(la)
    scale_lo = _interp_scale(lo_proxy)
    scale_to = _interp_scale(to_proxy)

    # Mode-count weighted composites (diamond cubic: 2 TA + 1 LA, 2 TO + 1 LO).
    scale_acoustic = [float((2.0 * float(sta) + float(sla)) / 3.0) for sta, sla in zip(scale_ta, scale_la)]
    scale_optical = [float((2.0 * float(sto) + float(slo)) / 3.0) for sto, slo in zip(scale_to, scale_lo)]

    scales: dict[str, list[float]] = {}
    if int(groups) == 2:
        scales = {"acoustic": scale_acoustic, "optical": scale_optical}
    elif int(groups) == 3:
        scales = {"ta": scale_ta, "la": scale_la, "optical": scale_optical}
    elif int(groups) == 4:
        scales = {"ta": scale_ta, "la": scale_la, "to": scale_to, "lo": scale_lo}
    else:
        raise SystemExit(f"[fail] unsupported groups for fig2 mode softening: {groups}")

    out = {
        "source": {"path": str(src), "sha256": _sha256(src)},
        "eq8_quasiharmonic": eq8_info if eq8_info is not None else "",
        "scales": scales,
        "notes": [
            "Mode-dependent softening is applied as ω_eff(T)=scale_group(T)*ω within each DOS group.",
            "TA scale is the average of the two TA features (red squares and purple circles).",
            "Composite scales use mode-count weights: acoustic=(2 TA + 1 LA)/3, optical=(2 TO + 1 LO)/3.",
            "When enabled, Eq.(8) quasiharmonic-only correction applies omega_scale^exponent (exponent fixed at 300 K using bar_gamma_P reference).",
            "This constraint is frozen from Kim2015 Fig.2 digitization (no fitted parameters).",
        ],
    }
    return out


def _metrics_for_range(
    *,
    idx: list[int],
    alpha_obs: list[float],
    alpha_pred: list[float],
    sigma_fit: list[float],
    param_count: int,
    is_train: bool,
) -> dict[str, float | int]:
    sign_mismatch = 0
    max_abs_z = 0.0
    sum_z2 = 0.0
    n = 0
    exceed_3sigma = 0
    for i in idx:
        ao = float(alpha_obs[i])
        ap = float(alpha_pred[i])
        sig = float(sigma_fit[i])
        if sig <= 0.0:
            continue

        if ao != 0.0 and ap != 0.0 and (ao > 0.0) != (ap > 0.0):
            sign_mismatch += 1

        z = (ap - ao) / sig
        if not math.isfinite(z):
            continue
        n += 1
        sum_z2 += z * z
        max_abs_z = max(max_abs_z, abs(z))
        if abs(z) > 3.0:
            exceed_3sigma += 1

    dof = max(1, n - int(param_count)) if is_train else max(1, n)
    red_chi2 = float(sum_z2 / dof) if dof > 0 else float("nan")
    rms_z = float("nan") if n <= 0 else math.sqrt(float(sum_z2 / n))
    return {
        "n": int(n),
        "sign_mismatch_n": int(sign_mismatch),
        "max_abs_z": float(max_abs_z),
        "rms_z": float(rms_z),
        "reduced_chi2": float(red_chi2),
        "exceed_3sigma_n": int(exceed_3sigma),
    }


def _fit_two_basis_weighted_ls(
    *,
    x1: list[float],
    x2: list[float],
    y: list[float],
    sigma: list[float],
    idx: list[int],
    enforce_signs: bool,
    ridge_factor: float,
) -> dict[str, float]:
    """
    Fit y ≈ a1*x1 + a2*x2 with weights 1/sigma^2 over idx.
    If enforce_signs is True, enforce a1<=0 and a2>=0 by enumerating boundary cases.
    """
    if not idx:
        raise ValueError("empty fit idx")

    def sse_for(a1: float, a2: float) -> float:
        sse = 0.0
        for i in idx:
            sig = float(sigma[i])
            if sig <= 0.0:
                continue
            w = 1.0 / (sig * sig)
            pred = float(a1) * float(x1[i]) + float(a2) * float(x2[i])
            r = float(y[i]) - pred
            sse += w * r * r
        return float(sse)

    # Unconstrained normal equations.
    s11 = 0.0
    s22 = 0.0
    s12 = 0.0
    b1 = 0.0
    b2 = 0.0
    for i in idx:
        sig = float(sigma[i])
        if sig <= 0.0:
            continue
        w = 1.0 / (sig * sig)
        u1 = float(x1[i])
        u2 = float(x2[i])
        yy = float(y[i])
        s11 += w * u1 * u1
        s22 += w * u2 * u2
        s12 += w * u1 * u2
        b1 += w * u1 * yy
        b2 += w * u2 * yy

    ridge_lambda = 0.0
    if float(ridge_factor) > 0.0:
        ridge_lambda = float(float(ridge_factor) * _median_positive([s11, s22]))

    sol = _solve_2x2(a11=s11 + float(ridge_lambda), a12=s12, a22=s22 + float(ridge_lambda), b1=b1, b2=b2)
    if sol is None:
        raise ValueError("singular normal equations in 2x2 fit")
    a1_u, a2_u = float(sol[0]), float(sol[1])

    if not enforce_signs:
        return {
            "A_low_mol_per_J": float(a1_u),
            "A_high_mol_per_J": float(a2_u),
            "sse": float(sse_for(a1_u, a2_u)),
            "ridge_lambda": float(ridge_lambda),
        }

    # Candidate enumeration for sign constraints.
    def ok(a1: float, a2: float) -> bool:
        return (float(a1) <= 0.0) and (float(a2) >= 0.0)

    candidates: list[dict[str, float]] = []
    if ok(a1_u, a2_u):
        candidates.append({"a1": float(a1_u), "a2": float(a2_u), "sse": float(sse_for(a1_u, a2_u)), "kind": "unconstrained"})

    # Boundary: a1=0, fit a2 only (and require a2>=0).
    den2 = 0.0
    num2 = 0.0
    for i in idx:
        sig = float(sigma[i])
        if sig <= 0.0:
            continue
        w = 1.0 / (sig * sig)
        u2 = float(x2[i])
        yy = float(y[i])
        den2 += w * u2 * u2
        num2 += w * u2 * yy
    if den2 > 0.0:
        a2 = float(num2 / (den2 + float(ridge_lambda)))
        if ok(0.0, a2):
            candidates.append({"a1": 0.0, "a2": float(a2), "sse": float(sse_for(0.0, a2)), "kind": "a1=0"})

    # Boundary: a2=0, fit a1 only (and require a1<=0).
    den1 = 0.0
    num1 = 0.0
    for i in idx:
        sig = float(sigma[i])
        if sig <= 0.0:
            continue
        w = 1.0 / (sig * sig)
        u1 = float(x1[i])
        yy = float(y[i])
        den1 += w * u1 * u1
        num1 += w * u1 * yy
    if den1 > 0.0:
        a1 = float(num1 / (den1 + float(ridge_lambda)))
        if ok(a1, 0.0):
            candidates.append({"a1": float(a1), "a2": 0.0, "sse": float(sse_for(a1, 0.0)), "kind": "a2=0"})

    # If no constrained candidate is feasible, fall back to unconstrained (but record).
    if not candidates:
        return {
            "A_low_mol_per_J": float(a1_u),
            "A_high_mol_per_J": float(a2_u),
            "sse": float(sse_for(a1_u, a2_u)),
            "ridge_lambda": float(ridge_lambda),
            "constraint_fallback": 1.0,
        }

    best = min(candidates, key=lambda d: float(d["sse"]))
    return {
        "A_low_mol_per_J": float(best["a1"]),
        "A_high_mol_per_J": float(best["a2"]),
        "sse": float(best["sse"]),
        "ridge_lambda": float(ridge_lambda),
        "constraint_solution_kind": float(0.0),  # kept numeric for consistent JSON typing
        "constraint_solution_kind_str": str(best.get("kind", "")),
    }


def _fit_three_basis_weighted_ls(
    *,
    x1: list[float],
    x2: list[float],
    x3: list[float],
    y: list[float],
    sigma: list[float],
    idx: list[int],
    enforce_signs: bool,
    ridge_factor: float,
) -> dict[str, float]:
    """
    Fit y ≈ a1*x1 + a2*x2 + a3*x3 with weights 1/sigma^2.

    If enforce_signs is True, enforce a1<=0, a2>=0, a3>=0 by enumerating boundary cases
    (setting some coefficients to 0).
    """
    if not idx:
        raise ValueError("empty fit idx")

    ridge_lambda = 0.0

    def sse_for(a1: float, a2: float, a3: float) -> float:
        sse = 0.0
        for i in idx:
            sig = float(sigma[i])
            if sig <= 0.0:
                continue
            w = 1.0 / (sig * sig)
            pred = float(a1) * float(x1[i]) + float(a2) * float(x2[i]) + float(a3) * float(x3[i])
            r = float(y[i]) - pred
            sse += w * r * r
        return float(sse)

    def ok(a1: float, a2: float, a3: float) -> bool:
        if not enforce_signs:
            return True
        return (float(a1) <= 0.0) and (float(a2) >= 0.0) and (float(a3) >= 0.0)

    # Helper: solve 1D fit for a single basis.
    def fit_1d(xx: list[float]) -> float | None:
        den = 0.0
        num = 0.0
        for i in idx:
            sig = float(sigma[i])
            if sig <= 0.0:
                continue
            w = 1.0 / (sig * sig)
            u = float(xx[i])
            den += w * u * u
            num += w * u * float(y[i])
        if den <= 0.0:
            return None
        return float(num / (den + float(ridge_lambda)))

    # Helper: solve 2D fit for two bases.
    def fit_2d(xx1: list[float], xx2: list[float]) -> tuple[float, float] | None:
        s11 = 0.0
        s22 = 0.0
        s12 = 0.0
        b1 = 0.0
        b2 = 0.0
        for i in idx:
            sig = float(sigma[i])
            if sig <= 0.0:
                continue
            w = 1.0 / (sig * sig)
            u1 = float(xx1[i])
            u2 = float(xx2[i])
            yy = float(y[i])
            s11 += w * u1 * u1
            s22 += w * u2 * u2
            s12 += w * u1 * u2
            b1 += w * u1 * yy
            b2 += w * u2 * yy
        sol = _solve_2x2(a11=s11 + float(ridge_lambda), a12=s12, a22=s22 + float(ridge_lambda), b1=b1, b2=b2)
        return None if sol is None else (float(sol[0]), float(sol[1]))

    # Unconstrained 3D normal equations.
    s11 = 0.0
    s22 = 0.0
    s33 = 0.0
    s12 = 0.0
    s13 = 0.0
    s23 = 0.0
    b1 = 0.0
    b2 = 0.0
    b3 = 0.0
    for i in idx:
        sig = float(sigma[i])
        if sig <= 0.0:
            continue
        w = 1.0 / (sig * sig)
        u1 = float(x1[i])
        u2 = float(x2[i])
        u3 = float(x3[i])
        yy = float(y[i])
        s11 += w * u1 * u1
        s22 += w * u2 * u2
        s33 += w * u3 * u3
        s12 += w * u1 * u2
        s13 += w * u1 * u3
        s23 += w * u2 * u3
        b1 += w * u1 * yy
        b2 += w * u2 * yy
        b3 += w * u3 * yy

    if float(ridge_factor) > 0.0:
        ridge_lambda = float(float(ridge_factor) * _median_positive([s11, s22, s33]))

    sol3 = _solve_3x3(
        a11=s11 + float(ridge_lambda),
        a12=s12,
        a13=s13,
        a22=s22 + float(ridge_lambda),
        a23=s23,
        a33=s33 + float(ridge_lambda),
        b1=b1,
        b2=b2,
        b3=b3,
    )
    if sol3 is None:
        raise ValueError("singular normal equations in 3x3 fit")
    a1_u, a2_u, a3_u = float(sol3[0]), float(sol3[1]), float(sol3[2])

    if not enforce_signs:
        return {
            "A_1_mol_per_J": float(a1_u),
            "A_2_mol_per_J": float(a2_u),
            "A_3_mol_per_J": float(a3_u),
            "sse": float(sse_for(a1_u, a2_u, a3_u)),
            "ridge_lambda": float(ridge_lambda),
        }

    candidates: list[dict[str, float]] = []
    if ok(a1_u, a2_u, a3_u):
        candidates.append(
            {"a1": float(a1_u), "a2": float(a2_u), "a3": float(a3_u), "sse": float(sse_for(a1_u, a2_u, a3_u))}
        )

    # Enumerate boundary subsets by zeroing some coefficients.
    # Subsets: keep 2 of 3.
    for keep in [(1, 2), (1, 3), (2, 3)]:
        if keep == (1, 2):
            sol = fit_2d(x1, x2)
            if sol is None:
                continue
            a1, a2 = sol
            a3 = 0.0
        elif keep == (1, 3):
            sol = fit_2d(x1, x3)
            if sol is None:
                continue
            a1, a3 = sol
            a2 = 0.0
        else:
            sol = fit_2d(x2, x3)
            if sol is None:
                continue
            a2, a3 = sol
            a1 = 0.0
        if ok(a1, a2, a3):
            candidates.append({"a1": float(a1), "a2": float(a2), "a3": float(a3), "sse": float(sse_for(a1, a2, a3))})

    # Subsets: keep 1 of 3.
    for j in [1, 2, 3]:
        if j == 1:
            a1 = fit_1d(x1)
            if a1 is None:
                continue
            a2, a3 = 0.0, 0.0
        elif j == 2:
            a2 = fit_1d(x2)
            if a2 is None:
                continue
            a1, a3 = 0.0, 0.0
        else:
            a3 = fit_1d(x3)
            if a3 is None:
                continue
            a1, a2 = 0.0, 0.0
        if ok(a1, a2, a3):
            candidates.append({"a1": float(a1), "a2": float(a2), "a3": float(a3), "sse": float(sse_for(a1, a2, a3))})

    if not candidates:
        return {
            "A_1_mol_per_J": float(a1_u),
            "A_2_mol_per_J": float(a2_u),
            "A_3_mol_per_J": float(a3_u),
            "sse": float(sse_for(a1_u, a2_u, a3_u)),
            "ridge_lambda": float(ridge_lambda),
            "constraint_fallback": 1.0,
        }

    best = min(candidates, key=lambda d: float(d["sse"]))
    return {
        "A_1_mol_per_J": float(best["a1"]),
        "A_2_mol_per_J": float(best["a2"]),
        "A_3_mol_per_J": float(best["a3"]),
        "sse": float(best["sse"]),
        "ridge_lambda": float(ridge_lambda),
    }


def _solve_4x4(
    *,
    a11: float,
    a12: float,
    a13: float,
    a14: float,
    a22: float,
    a23: float,
    a24: float,
    a33: float,
    a34: float,
    a44: float,
    b1: float,
    b2: float,
    b3: float,
    b4: float,
) -> Optional[tuple[float, float, float, float]]:
    # Symmetric 4x4 solve via Gauss-Jordan elimination (with partial pivoting).
    m = [
        [float(a11), float(a12), float(a13), float(a14), float(b1)],
        [float(a12), float(a22), float(a23), float(a24), float(b2)],
        [float(a13), float(a23), float(a33), float(a34), float(b3)],
        [float(a14), float(a24), float(a34), float(a44), float(b4)],
    ]

    for col in range(4):
        pivot = col
        best = abs(m[col][col])
        for r in range(col + 1, 4):
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
        for j in range(col, 5):
            m[col][j] *= inv

        for r in range(4):
            if r == col:
                continue
            factor = m[r][col]
            if factor == 0.0:
                continue
            for j in range(col, 5):
                m[r][j] -= factor * m[col][j]

    x1 = float(m[0][4])
    x2 = float(m[1][4])
    x3 = float(m[2][4])
    x4 = float(m[3][4])
    if not (math.isfinite(x1) and math.isfinite(x2) and math.isfinite(x3) and math.isfinite(x4)):
        return None
    return x1, x2, x3, x4


def _fit_four_basis_weighted_ls(
    *,
    x1: list[float],
    x2: list[float],
    x3: list[float],
    x4: list[float],
    y: list[float],
    sigma: list[float],
    idx: list[int],
    enforce_signs: bool,
    ridge_factor: float,
) -> dict[str, float]:
    """
    Fit y ≈ a1*x1 + a2*x2 + a3*x3 + a4*x4 with weights 1/sigma^2.

    If enforce_signs is True, enforce a1<=0 and a2,a3,a4>=0 by enumerating boundary
    cases (setting some coefficients to 0).
    """
    if not idx:
        raise ValueError("empty fit idx")

    def sse_for(a1: float, a2: float, a3: float, a4: float) -> float:
        sse = 0.0
        for i in idx:
            sig = float(sigma[i])
            if sig <= 0.0:
                continue
            w = 1.0 / (sig * sig)
            pred = float(a1) * float(x1[i]) + float(a2) * float(x2[i]) + float(a3) * float(x3[i]) + float(a4) * float(x4[i])
            r = float(y[i]) - pred
            sse += w * r * r
        return float(sse)

    def ok(a1: float, a2: float, a3: float, a4: float) -> bool:
        if not enforce_signs:
            return True
        return (float(a1) <= 0.0) and (float(a2) >= 0.0) and (float(a3) >= 0.0) and (float(a4) >= 0.0)

    # Full normal equations.
    s11 = 0.0
    s22 = 0.0
    s33 = 0.0
    s44 = 0.0
    s12 = 0.0
    s13 = 0.0
    s14 = 0.0
    s23 = 0.0
    s24 = 0.0
    s34 = 0.0
    b1 = 0.0
    b2 = 0.0
    b3 = 0.0
    b4 = 0.0
    for i in idx:
        sig = float(sigma[i])
        if sig <= 0.0:
            continue
        w = 1.0 / (sig * sig)
        u1 = float(x1[i])
        u2 = float(x2[i])
        u3 = float(x3[i])
        u4 = float(x4[i])
        yy = float(y[i])
        s11 += w * u1 * u1
        s22 += w * u2 * u2
        s33 += w * u3 * u3
        s44 += w * u4 * u4
        s12 += w * u1 * u2
        s13 += w * u1 * u3
        s14 += w * u1 * u4
        s23 += w * u2 * u3
        s24 += w * u2 * u4
        s34 += w * u3 * u4
        b1 += w * u1 * yy
        b2 += w * u2 * yy
        b3 += w * u3 * yy
        b4 += w * u4 * yy

    ridge_lambda = 0.0
    if float(ridge_factor) > 0.0:
        ridge_lambda = float(float(ridge_factor) * _median_positive([s11, s22, s33, s44]))
        if float(ridge_lambda) > 0.0:
            s11 += float(ridge_lambda)
            s22 += float(ridge_lambda)
            s33 += float(ridge_lambda)
            s44 += float(ridge_lambda)

    sol4 = _solve_4x4(
        a11=s11,
        a12=s12,
        a13=s13,
        a14=s14,
        a22=s22,
        a23=s23,
        a24=s24,
        a33=s33,
        a34=s34,
        a44=s44,
        b1=b1,
        b2=b2,
        b3=b3,
        b4=b4,
    )
    if sol4 is None:
        raise ValueError("singular normal equations in 4x4 fit")
    a1_u, a2_u, a3_u, a4_u = float(sol4[0]), float(sol4[1]), float(sol4[2]), float(sol4[3])

    if not enforce_signs:
        return {
            "A_1_mol_per_J": float(a1_u),
            "A_2_mol_per_J": float(a2_u),
            "A_3_mol_per_J": float(a3_u),
            "A_4_mol_per_J": float(a4_u),
            "sse": float(sse_for(a1_u, a2_u, a3_u, a4_u)),
            "ridge_lambda": float(ridge_lambda),
        }

    candidates: list[dict[str, float]] = []

    def add_candidate(a1: float, a2: float, a3: float, a4: float) -> None:
        if ok(a1, a2, a3, a4):
            candidates.append({"a1": float(a1), "a2": float(a2), "a3": float(a3), "a4": float(a4), "sse": float(sse_for(a1, a2, a3, a4))})

    add_candidate(a1_u, a2_u, a3_u, a4_u)

    # 3D subsets
    sol_123 = _solve_3x3(a11=s11, a12=s12, a13=s13, a22=s22, a23=s23, a33=s33, b1=b1, b2=b2, b3=b3)
    if sol_123 is not None:
        add_candidate(sol_123[0], sol_123[1], sol_123[2], 0.0)
    sol_124 = _solve_3x3(a11=s11, a12=s12, a13=s14, a22=s22, a23=s24, a33=s44, b1=b1, b2=b2, b3=b4)
    if sol_124 is not None:
        add_candidate(sol_124[0], sol_124[1], 0.0, sol_124[2])
    sol_134 = _solve_3x3(a11=s11, a12=s13, a13=s14, a22=s33, a23=s34, a33=s44, b1=b1, b2=b3, b3=b4)
    if sol_134 is not None:
        add_candidate(sol_134[0], 0.0, sol_134[1], sol_134[2])
    sol_234 = _solve_3x3(a11=s22, a12=s23, a13=s24, a22=s33, a23=s34, a33=s44, b1=b2, b2=b3, b3=b4)
    if sol_234 is not None:
        add_candidate(0.0, sol_234[0], sol_234[1], sol_234[2])

    # 2D subsets
    sol_12 = _solve_2x2(a11=s11, a12=s12, a22=s22, b1=b1, b2=b2)
    if sol_12 is not None:
        add_candidate(sol_12[0], sol_12[1], 0.0, 0.0)
    sol_13 = _solve_2x2(a11=s11, a12=s13, a22=s33, b1=b1, b2=b3)
    if sol_13 is not None:
        add_candidate(sol_13[0], 0.0, sol_13[1], 0.0)
    sol_14 = _solve_2x2(a11=s11, a12=s14, a22=s44, b1=b1, b2=b4)
    if sol_14 is not None:
        add_candidate(sol_14[0], 0.0, 0.0, sol_14[1])
    sol_23 = _solve_2x2(a11=s22, a12=s23, a22=s33, b1=b2, b2=b3)
    if sol_23 is not None:
        add_candidate(0.0, sol_23[0], sol_23[1], 0.0)
    sol_24 = _solve_2x2(a11=s22, a12=s24, a22=s44, b1=b2, b2=b4)
    if sol_24 is not None:
        add_candidate(0.0, sol_24[0], 0.0, sol_24[1])
    sol_34 = _solve_2x2(a11=s33, a12=s34, a22=s44, b1=b3, b2=b4)
    if sol_34 is not None:
        add_candidate(0.0, 0.0, sol_34[0], sol_34[1])

    # 1D subsets
    if s11 > 0.0:
        add_candidate(b1 / s11, 0.0, 0.0, 0.0)
    if s22 > 0.0:
        add_candidate(0.0, b2 / s22, 0.0, 0.0)
    if s33 > 0.0:
        add_candidate(0.0, 0.0, b3 / s33, 0.0)
    if s44 > 0.0:
        add_candidate(0.0, 0.0, 0.0, b4 / s44)

    if not candidates:
        return {
            "A_1_mol_per_J": float(a1_u),
            "A_2_mol_per_J": float(a2_u),
            "A_3_mol_per_J": float(a3_u),
            "A_4_mol_per_J": float(a4_u),
            "sse": float(sse_for(a1_u, a2_u, a3_u, a4_u)),
            "ridge_lambda": float(ridge_lambda),
            "constraint_fallback": 1.0,
        }

    best = min(candidates, key=lambda d: float(d["sse"]))
    return {
        "A_1_mol_per_J": float(best["a1"]),
        "A_2_mol_per_J": float(best["a2"]),
        "A_3_mol_per_J": float(best["a3"]),
        "A_4_mol_per_J": float(best["a4"]),
        "sse": float(best["sse"]),
        "ridge_lambda": float(ridge_lambda),
    }


def _solve_weighted_normal_equations(
    *,
    cols: list[list[float]],
    y: list[float],
    sigma: list[float],
    idx: list[int],
    ridge_factor: float,
    delta_ridge_factor: float,
) -> Optional[np.ndarray]:
    """
    Solve weighted normal equations for y ≈ Σ a_j * cols[j] over idx with weights 1/sigma^2.

    Returns a vector of coefficients (len=cols) or None if the system is singular/ill-conditioned.
    """
    m = int(len(cols))
    if m < 1:
        raise ValueError("no columns")
    n = int(len(y))
    if any(len(c) != n for c in cols):
        raise ValueError("column length mismatch")
    if len(sigma) != n:
        raise ValueError("sigma length mismatch")
    if not idx:
        raise ValueError("empty fit idx")

    a = np.zeros((m, m), dtype=float)
    b = np.zeros((m,), dtype=float)
    for i in idx:
        sig = float(sigma[i])
        if sig <= 0.0:
            continue
        w = 1.0 / (sig * sig)
        xs = np.asarray([float(c[i]) for c in cols], dtype=float)
        a += w * np.outer(xs, xs)
        b += w * xs * float(y[i])
    base = float(_median_positive([float(a[j, j]) for j in range(m)]))
    if base > 0.0 and math.isfinite(base):
        lam = float(float(ridge_factor) * base) if float(ridge_factor) > 0.0 else 0.0
        lam_delta = float(float(delta_ridge_factor) * base) if float(delta_ridge_factor) > 0.0 else 0.0
        if lam > 0.0 and math.isfinite(lam):
            for j in range(m):
                a[j, j] += lam
        if lam_delta > 0.0 and math.isfinite(lam_delta):
            a[m - 1, m - 1] += lam_delta
    try:
        sol = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(sol)):
        return None
    return sol


def _fit_basis_plus_delta_weighted_ls(
    *,
    x_cols: list[list[float]],
    g_cols: list[list[float]],
    y: list[float],
    sigma: list[float],
    idx: list[int],
    enforce_signs: bool,
    sign_constraints: list[int],
    ridge_factor: float,
    delta_ridge_factor: float,
) -> dict[str, object]:
    """
    Fit y ≈ Σ a_j * x_j + delta * (Σ g_j*x_j) with weights 1/sigma^2 over idx.

    This corresponds to γ_j(T)=a_j + delta*g_j(T) (per group) with a single common delta.

    sign_constraints[j] is -1 (require a_j<=0), +1 (require a_j>=0), or 0 (unconstrained).
    The delta parameter is always unconstrained.
    """
    k = int(len(x_cols))
    if k < 1 or len(g_cols) != k or len(sign_constraints) != k:
        raise ValueError("invalid basis sizes")
    n = int(len(y))
    if any(len(c) != n for c in x_cols) or any(len(g) != n for g in g_cols) or len(sigma) != n:
        raise ValueError("length mismatch")

    x_delta = [sum(float(g_cols[j][i]) * float(x_cols[j][i]) for j in range(k)) for i in range(n)]

    def sse_for(a_vals: list[float], delta: float) -> float:
        sse = 0.0
        for i in idx:
            sig = float(sigma[i])
            if sig <= 0.0:
                continue
            w = 1.0 / (sig * sig)
            pred = float(delta) * float(x_delta[i])
            for j in range(k):
                pred += float(a_vals[j]) * float(x_cols[j][i])
            r = float(y[i]) - pred
            sse += w * r * r
        return float(sse)

    def ok(a_vals: list[float]) -> bool:
        if not enforce_signs:
            return True
        for j, a in enumerate(a_vals):
            c = int(sign_constraints[j])
            if c == -1 and float(a) > 0.0:
                return False
            if c == 1 and float(a) < 0.0:
                return False
        return True

    if not enforce_signs:
        cols = [list(col) for col in x_cols] + [x_delta]
        sol = _solve_weighted_normal_equations(
            cols=cols,
            y=y,
            sigma=sigma,
            idx=idx,
            ridge_factor=float(ridge_factor),
            delta_ridge_factor=float(delta_ridge_factor),
        )
        if sol is None:
            raise ValueError("singular normal equations in basis+delta fit")
        a_vals = [float(sol[j]) for j in range(k)]
        delta = float(sol[-1])
        return {"a": a_vals, "delta_gamma": delta, "sse": float(sse_for(a_vals, delta)), "active_mask": int((1 << k) - 1)}

    # Enumerate boundary cases by activating a subset of the group coefficients; delta is always active.
    best: dict[str, object] | None = None
    best_sse = float("inf")
    for mask in range(1 << k):
        active_idx = [j for j in range(k) if (mask & (1 << j)) != 0]
        cols = [x_cols[j] for j in active_idx] + [x_delta]
        sol = _solve_weighted_normal_equations(
            cols=cols,
            y=y,
            sigma=sigma,
            idx=idx,
            ridge_factor=float(ridge_factor),
            delta_ridge_factor=float(delta_ridge_factor),
        )
        if sol is None:
            continue
        a_vals = [0.0 for _ in range(k)]
        for jj, j in enumerate(active_idx):
            a_vals[j] = float(sol[jj])
        delta = float(sol[-1])
        if not ok(a_vals):
            continue
        sse = float(sse_for(a_vals, delta))
        if math.isfinite(sse) and sse < best_sse:
            best_sse = float(sse)
            best = {"a": list(a_vals), "delta_gamma": float(delta), "sse": float(sse), "active_mask": int(mask)}

    if best is None:
        # Fall back to unconstrained full fit (record it).
        cols = [list(col) for col in x_cols] + [x_delta]
        sol = _solve_weighted_normal_equations(
            cols=cols,
            y=y,
            sigma=sigma,
            idx=idx,
            ridge_factor=float(ridge_factor),
            delta_ridge_factor=float(delta_ridge_factor),
        )
        if sol is None:
            raise ValueError("singular normal equations in basis+delta fit (fallback)")
        a_vals = [float(sol[j]) for j in range(k)]
        delta = float(sol[-1])
        return {
            "a": a_vals,
            "delta_gamma": delta,
            "sse": float(sse_for(a_vals, delta)),
            "active_mask": int((1 << k) - 1),
            "constraint_fallback": 1.0,
        }
    return best


def _kim2015_table1_gruneisen_diagnostics(
    *,
    root: Path,
    fig2_source_dirname: str,
    fig2_json_name: str,
    temps_k: list[float],
    alpha_obs: list[float],
    cv_total_j_per_mol_k: list[float],
    bulk_modulus_model: dict[str, Any] | None,
    silicon_molar_volume: dict[str, Any] | None,
) -> dict[str, object] | None:
    """
    Diagnostic connection to Kim et al., PRB 91, 014307 (2015).

    Computes room-temperature (T≈300 K) Gruneisen-parameter proxies based on the digitized Fig.2
    ω scales and the definitions in Eqs. (4), (6), and (9) of the paper.

    Notes:
      - Eq.(6) uses α(T); here α(T) is taken from the NIST TRC α(T) fit already used in this script.
      - Eq.(4) requires d ln ε / d ln V; we approximate it from finite differences between 100 K and 300 K,
        using Δ ln V inferred by integrating α(T) (Δ ln V ≈ 3∫α dT). This is a proxy for procedure checks.
      - Eq.(9) uses a bulk modulus and Cv; we compute γ_thermo at 300 K using B(T) from Ioffe and Cv_total
        from the current Cv basis.
    """
    if len(temps_k) != len(alpha_obs) or len(temps_k) != len(cv_total_j_per_mol_k):
        return None

    # Require T=300 K to be present in the integer grid.
    try:
        idx_300 = temps_k.index(300.0)
    except ValueError:
        return None
    try:
        idx_100 = temps_k.index(100.0)
    except ValueError:
        # Fig.2 references 100 K; if the α grid does not include it (should), skip this proxy.
        return None

    # Load digitized Fig.2 series.
    src = root / "data" / "quantum" / "sources" / str(fig2_source_dirname) / str(fig2_json_name)
    if not src.exists():
        return None
    obj = _read_json(src)
    series_obj = obj.get("series")
    derived_obj = obj.get("derived")
    if not isinstance(series_obj, dict) or not isinstance(derived_obj, dict):
        return None

    def _rows_for(key: str, *, derived: bool = False) -> list[dict[str, float]]:
        container = derived_obj if derived else series_obj
        s = container.get(key)
        if not isinstance(s, dict) or not isinstance(s.get("rows"), list):
            return []
        rows = []
        for r in s["rows"]:
            if not isinstance(r, dict):
                continue
            try:
                rows.append({"t_K": float(r["t_K"]), "omega_scale": float(r["omega_scale"])})
            except Exception:
                continue
        rows.sort(key=lambda rr: float(rr["t_K"]))
        return rows

    # Five features as described around Fig.2: TA (two markers), LA, LA/LO, TO/LO.
    feat_rows = {
        "TA_sq": _rows_for("TA_sq"),
        "TA_circ": _rows_for("TA_circ"),
        "LA_pent": _rows_for("LA_pent"),
        "LA_LO_hex": _rows_for("LA_LO_hex"),
        "TO_LO_tri": _rows_for("TO_LO_triangles", derived=True),
    }
    if any(len(v) < 2 for v in feat_rows.values()):
        return None

    def _interp_y(rows: list[dict[str, float]], t0: float) -> float:
        ts = [float(r["t_K"]) for r in rows]
        ys = [float(r["omega_scale"]) for r in rows]
        return float(_interp_piecewise_linear(ts, ys, [float(t0)])[0])

    def _slope_y(rows: list[dict[str, float]], t0: float) -> float:
        # Piecewise-linear slope dy/dT at t0, clamped to end segments.
        ts = [float(r["t_K"]) for r in rows]
        ys = [float(r["omega_scale"]) for r in rows]
        if t0 <= ts[0]:
            i0 = 0
        elif t0 >= ts[-1]:
            i0 = max(0, len(ts) - 2)
        else:
            i0 = bisect.bisect_right(ts, float(t0)) - 1
            i0 = min(max(0, int(i0)), len(ts) - 2)
        t_a = float(ts[i0])
        t_b = float(ts[i0 + 1])
        y_a = float(ys[i0])
        y_b = float(ys[i0 + 1])
        dt = float(t_b - t_a)
        if dt <= 0.0:
            return 0.0
        return float((y_b - y_a) / dt)

    # Eq.(6): gamma_T = - <∂ ln ε_i / ∂T>_P / (3 α(T)).
    alpha_300 = float(alpha_obs[idx_300])
    if not (math.isfinite(alpha_300) and alpha_300 != 0.0):
        return None

    dlnw_dT_by_feat: dict[str, float] = {}
    lnw_100_300_by_feat: dict[str, float] = {}
    for k, rows in feat_rows.items():
        y300 = _interp_y(rows, 300.0)
        y100 = _interp_y(rows, 100.0)
        dy_dT_300 = _slope_y(rows, 300.0)
        if not (y300 > 0.0 and y100 > 0.0):
            return None
        dlnw_dT_by_feat[k] = float(dy_dT_300 / y300)
        lnw_100_300_by_feat[k] = float(math.log(y300 / y100))

    dlnw_dT_mean = float(sum(dlnw_dT_by_feat.values()) / float(len(dlnw_dT_by_feat)))
    gammaT_eq6_mean = float(-dlnw_dT_mean / (3.0 * alpha_300))

    # Eq.(4) proxy: bar_gamma ≈ -Δ ln ω / Δ ln V between 100 K and 300 K.
    delta_lnV_100_300 = 0.0
    for i in range(idx_100, idx_300):
        t0 = float(temps_k[i])
        t1 = float(temps_k[i + 1])
        a0 = float(alpha_obs[i])
        a1 = float(alpha_obs[i + 1])
        dt = float(t1 - t0)
        if dt <= 0.0:
            continue
        delta_lnV_100_300 += 3.0 * 0.5 * (a0 + a1) * dt
    lnw_mean_100_300 = float(sum(lnw_100_300_by_feat.values()) / float(len(lnw_100_300_by_feat)))
    bar_gamma_eq4_proxy = float("nan") if delta_lnV_100_300 == 0.0 else float(-lnw_mean_100_300 / delta_lnV_100_300)

    # Eq.(9): thermodynamic gamma = 3 α V0 B / Cv.
    gamma_thermo_eq9 = float("nan")
    b300 = float("nan")
    v_m = float("nan")
    cv300 = float(cv_total_j_per_mol_k[idx_300])
    if bulk_modulus_model is not None and silicon_molar_volume is not None and cv300 > 0.0 and math.isfinite(cv300):
        b300 = float(_bulk_modulus_pa(t_k=300.0, model=bulk_modulus_model))
        v_m = float(silicon_molar_volume.get("V_m3_per_mol", float("nan")))
        if math.isfinite(b300) and math.isfinite(v_m) and b300 > 0.0 and v_m > 0.0:
            gamma_thermo_eq9 = float(3.0 * alpha_300 * v_m * b300 / cv300)

    # Eq.(8): intrinsic temperature term at constant volume (diagnostic).
    #
    #   bar_gamma_T = bar_gamma_P - (1 / (3 α)) * <∂ ln ε / ∂T>_V
    # => <∂ ln ε / ∂T>_V = 3 α (bar_gamma_P - bar_gamma_T)
    #
    # We report two versions:
    #   - using Table I reference bar_gamma_T (7.00±0.67)
    #   - using the digitized Fig.2-derived bar_gamma_T (gammaT_eq6_mean)
    bar_gamma_p_ref = 0.98
    bar_gamma_t_ref = 7.00
    intrinsic_dlnomega_dT_v_300_table = float(3.0 * alpha_300 * (bar_gamma_p_ref - bar_gamma_t_ref))
    intrinsic_dlnomega_dT_v_300_digitized = float(3.0 * alpha_300 * (bar_gamma_p_ref - gammaT_eq6_mean))
    quasiharmonic_dlnomega_dT_via_volume_300 = float(-3.0 * alpha_300 * bar_gamma_p_ref)
    isobaric_dlnomega_dT_300 = float(dlnw_dT_mean)
    frac_quasi = (
        float("nan")
        if isobaric_dlnomega_dT_300 == 0.0
        else float(quasiharmonic_dlnomega_dT_via_volume_300 / isobaric_dlnomega_dT_300)
    )
    frac_intr_table = (
        float("nan")
        if isobaric_dlnomega_dT_300 == 0.0
        else float(intrinsic_dlnomega_dT_v_300_table / isobaric_dlnomega_dT_300)
    )
    frac_intr_digitized = (
        float("nan")
        if isobaric_dlnomega_dT_300 == 0.0
        else float(intrinsic_dlnomega_dT_v_300_digitized / isobaric_dlnomega_dT_300)
    )

    return {
        "source": {"path": str(src), "sha256": _sha256(src)},
        "t_eval_K": 300.0,
        "t_base_for_delta_K": 100.0,
        "computed": {
            "alpha_300K_per_K": float(alpha_300),
            "dlnomega_dT_300K_mean_per_K": float(dlnw_dT_mean),
            "gammaT_eq6_mean": float(gammaT_eq6_mean),
            "delta_lnV_100_to_300": float(delta_lnV_100_300),
            "delta_lnomega_100_to_300_mean": float(lnw_mean_100_300),
            "bar_gamma_eq4_proxy": float(bar_gamma_eq4_proxy),
            "Cv_total_300K_J_per_molK": float(cv300),
            "B_300K_Pa": float(b300),
            "V_m_m3_per_mol": float(v_m),
            "gamma_thermo_eq9": float(gamma_thermo_eq9),
            "Eq8_table_ref": {
                "bar_gamma_P_ref": float(bar_gamma_p_ref),
                "bar_gamma_T_ref": float(bar_gamma_t_ref),
            },
            "Eq8_intrinsic_dlnomega_dT_300K_at_V_from_table_ref_per_K": float(intrinsic_dlnomega_dT_v_300_table),
            "Eq8_intrinsic_dlnomega_dT_300K_at_V_from_digitized_gammaT_per_K": float(
                intrinsic_dlnomega_dT_v_300_digitized
            ),
            "Eq8_quasiharmonic_dlnomega_dT_300K_from_bar_gamma_P_ref_per_K": float(
                quasiharmonic_dlnomega_dT_via_volume_300
            ),
            "Eq8_fraction_quasiharmonic_vs_isobaric_300K": float(frac_quasi),
            "Eq8_fraction_intrinsic_vs_isobaric_300K_from_table_ref": float(frac_intr_table),
            "Eq8_fraction_intrinsic_vs_isobaric_300K_from_digitized_gammaT": float(frac_intr_digitized),
        },
        "by_feature": {
            "dlnomega_dT_300K_per_K": dlnw_dT_by_feat,
            "delta_lnomega_100_to_300": lnw_100_300_by_feat,
        },
        "kim2015_table1_reference": {
            "note": "Values transcribed from the extracted text of Kim et al., Phys. Rev. B 91, 014307 (2015), Table I.",
            "bar_gamma": {"value": 1.00, "sigma": 0.60},
            "bar_gamma_T": {"value": 7.00, "sigma": 0.67},
            "bar_gamma_P": {"value": 0.98},
            "gamma_thermo": {"value": 0.367},
        },
        "notes": [
            "Eq.(6) uses α(T); here α(T) is taken from the NIST TRC fit used for the main α(T) evaluation.",
            "Eq.(4) proxy uses Δ ln V inferred from integrating α(T) (Δ ln V≈3∫α dT) between 100 K and 300 K.",
            "Eq.(9) uses B(T) and Cv(T); here B(T) is from Ioffe and Cv_total is from the current DOS-based basis.",
            "Eq.(8) decomposes the isobaric phonon shift into a quasiharmonic (volume) term and an intrinsic (T|_V) term.",
        ],
    }


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    ridge_factor = float(getattr(args, "ridge_factor", 0.0))
    if not math.isfinite(ridge_factor) or ridge_factor < 0.0:
        raise SystemExit(f"[fail] --ridge-factor must be a finite value >= 0: got {getattr(args, 'ridge_factor', None)}")
    delta_ridge_factor = float(getattr(args, "delta_ridge_factor", 0.0))
    if not math.isfinite(delta_ridge_factor) or delta_ridge_factor < 0.0:
        raise SystemExit(
            f"[fail] --delta-ridge-factor must be a finite value >= 0: got {getattr(args, 'delta_ridge_factor', None)}"
        )

    root = _repo_root()
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    cv_omega_dependence = str(getattr(args, "cv_omega_dependence", "harmonic"))
    if cv_omega_dependence not in ("harmonic", "dU_numeric"):
        raise SystemExit(f"[fail] invalid --cv-omega-dependence: {cv_omega_dependence!r}")

    # α(T) (observed): NIST TRC fit.
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

    temps = [float(t) for t in range(t_min, t_max + 1)]
    alpha_obs_1e8 = [_alpha_1e8_per_k(t_k=t, coeffs=coeffs) for t in temps]
    alpha_obs = [float(a) * 1e-8 for a in alpha_obs_1e8]
    sigma_fit_1e8 = [sigma_lt_1e8 if float(t) < t_sigma_split else sigma_ge_1e8 for t in temps]
    sigma_fit = [float(s) * 1e-8 for s in sigma_fit_1e8]

    use_bulk_modulus = bool(args.use_bulk_modulus)
    use_vm_thermal_expansion = bool(getattr(args, "vm_thermal_expansion", False))
    if use_vm_thermal_expansion and not use_bulk_modulus:
        raise SystemExit("[fail] --vm-thermal-expansion requires --use-bulk-modulus")
    gamma_trend = str(getattr(args, "gamma_trend", "constant"))
    gamma_omega_model = str(getattr(args, "gamma_omega_model", "none"))
    gamma_omega_pwlinear_leak = float(getattr(args, "gamma_omega_pwlinear_leak", 0.05))
    gamma_omega_pwlinear_warp_power = float(getattr(args, "gamma_omega_pwlinear_warp_power", 1.0))
    gamma_omega_high_softening_delta = float(getattr(args, "gamma_omega_high_softening_delta", 0.0))
    gamma_omega_softening_delta = float(getattr(args, "gamma_omega_softening_delta", 0.0))
    gamma_omega_softening_fit = str(getattr(args, "gamma_omega_softening_fit", "none"))
    if not math.isfinite(gamma_omega_pwlinear_leak) or gamma_omega_pwlinear_leak < 0.0 or gamma_omega_pwlinear_leak >= 1.0:
        raise SystemExit(
            f"[fail] --gamma-omega-pwlinear-leak must be a finite value in [0,1): got {getattr(args, 'gamma_omega_pwlinear_leak', None)}"
        )
    if not math.isfinite(gamma_omega_pwlinear_warp_power) or gamma_omega_pwlinear_warp_power <= 0.0:
        raise SystemExit(
            f"[fail] --gamma-omega-pwlinear-warp-power must be a finite value > 0: got {getattr(args, 'gamma_omega_pwlinear_warp_power', None)}"
        )
    if gamma_omega_model != "pwlinear_split_leaky" and abs(gamma_omega_pwlinear_warp_power - 1.0) > 1e-12:
        raise SystemExit("[fail] --gamma-omega-pwlinear-warp-power is supported only for --gamma-omega-model=pwlinear_split_leaky")
    if not math.isfinite(gamma_omega_high_softening_delta):
        raise SystemExit(
            f"[fail] --gamma-omega-high-softening-delta must be a finite value: got {getattr(args, 'gamma_omega_high_softening_delta', None)}"
        )
    if not math.isfinite(gamma_omega_softening_delta):
        raise SystemExit(
            f"[fail] --gamma-omega-softening-delta must be a finite value: got {getattr(args, 'gamma_omega_softening_delta', None)}"
        )
    if gamma_omega_high_softening_delta != 0.0 and gamma_omega_model == "none":
        raise SystemExit("[fail] --gamma-omega-high-softening-delta requires --gamma-omega-model!=none")
    if gamma_omega_softening_delta != 0.0 and gamma_omega_model == "none":
        raise SystemExit("[fail] --gamma-omega-softening-delta requires --gamma-omega-model!=none")
    if gamma_omega_high_softening_delta != 0.0 and gamma_omega_softening_delta != 0.0:
        raise SystemExit("[fail] --gamma-omega-high-softening-delta and --gamma-omega-softening-delta are mutually exclusive")
    if gamma_omega_softening_fit != "none" and gamma_omega_model == "none":
        raise SystemExit("[fail] --gamma-omega-softening-fit requires --gamma-omega-model!=none")
    if gamma_omega_softening_fit != "none" and (gamma_omega_high_softening_delta != 0.0 or gamma_omega_softening_delta != 0.0):
        raise SystemExit("[fail] --gamma-omega-softening-fit is mutually exclusive with fixed --gamma-omega-*-softening-delta")
    if gamma_omega_model != "none":
        if not use_bulk_modulus:
            raise SystemExit("[fail] --gamma-omega-model requires --use-bulk-modulus (gamma units)")
        if gamma_trend != "constant":
            raise SystemExit("[fail] --gamma-omega-model is currently incompatible with --gamma-trend (use constant)")
        if str(getattr(args, "optical_softening", "none")) != "none":
            raise SystemExit("[fail] --gamma-omega-model is currently incompatible with --optical-softening")
    b_model = _load_ioffe_bulk_modulus_model(root=root) if use_bulk_modulus else None
    v_m = _load_silicon_molar_volume_m3_per_mol(root=root) if use_bulk_modulus else None
    v_m_t: list[float] | None = None
    if use_bulk_modulus:
        if b_model is None or v_m is None:
            raise SystemExit("[fail] internal: bulk modulus enabled but models are missing")
        v_ref = float(v_m["V_m3_per_mol"])
        if use_vm_thermal_expansion:
            v_m_t = _molar_volume_from_alpha_fit(
                temps_k=temps,
                alpha_linear_per_k=alpha_obs,
                v_ref_m3_per_mol=v_ref,
                t_ref_k=300.0,
            )
        inv_bv = [
            1.0
            / (
                float(_bulk_modulus_pa(t_k=float(t), model=b_model))
                * float(v_m_t[i] if v_m_t is not None else v_ref)
            )
            for i, t in enumerate(temps)
        ]
    else:
        inv_bv = [1.0 for _ in temps]
    inv_bv_np = np.asarray(inv_bv, dtype=float)

    dos_mode = str(args.dos_mode)
    if gamma_omega_model != "none" and dos_mode != "static_omega":
        raise SystemExit("[fail] --gamma-omega-model is currently supported only for --dos-mode=static_omega")
    if dos_mode == "kim2015_fig1_energy":
        if str(args.dos_softening) != "none":
            raise SystemExit("[fail] --dos-softening must be 'none' when --dos-mode=kim2015_fig1_energy")
        if str(args.mode_softening) != "none":
            raise SystemExit("[fail] --mode-softening must be 'none' when --dos-mode=kim2015_fig1_energy")
        if str(args.optical_softening) != "none":
            raise SystemExit("[fail] --optical-softening must be 'none' when --dos-mode=kim2015_fig1_energy")
    if str(args.mode_softening) != "none":
        if dos_mode != "static_omega":
            raise SystemExit("[fail] --mode-softening requires --dos-mode=static_omega")
        if str(args.dos_softening) != "none":
            raise SystemExit("[fail] --dos-softening must be 'none' when --mode-softening is enabled")
        if str(args.optical_softening) != "none":
            raise SystemExit("[fail] --optical-softening must be 'none' when --mode-softening is enabled")
    if cv_omega_dependence != "harmonic":
        if dos_mode != "static_omega":
            raise SystemExit("[fail] --cv-omega-dependence=dU_numeric requires --dos-mode=static_omega")
        if str(args.optical_softening) != "none":
            raise SystemExit("[fail] --cv-omega-dependence=dU_numeric requires --optical-softening=none")

    # Phonon DOS (ω–D(ω)).
    dos_src = root / "data" / "quantum" / "sources" / str(args.dos_source_dir) / "extracted_values.json"
    if not dos_src.exists():
        raise SystemExit(
            f"[fail] missing: {dos_src}\n"
            "Run: python -B scripts/quantum/fetch_silicon_phonon_dos_sources.py"
        )
    dos_extracted = _read_json(dos_src)
    rows = dos_extracted.get("rows")
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"[fail] rows missing: {dos_src}")

    omega = [float(r["omega_rad_s"]) for r in rows]
    dos = [float(r["dos_per_m3_per_rad_s"]) for r in rows]

    # Normalize to per-atom DOS so that ∫ g(ω) dω ≈ 3.
    integral = _trapz_xy(omega, dos)
    if integral <= 0.0:
        raise SystemExit("[fail] non-positive integral for DOS")
    n_atoms_m3 = float(integral / 3.0)
    g_per_atom = [float(d) / float(n_atoms_m3) for d in dos]  # 1/(rad/s)
    integral_per_atom = _trapz_xy(omega, g_per_atom)

    # Split frequency: half integral (acoustic/optical proxy).
    split = dos_extracted.get("derived", {}).get("split_half_integral", {}) if isinstance(dos_extracted.get("derived"), dict) else {}
    omega_split = float(split.get("omega_rad_s")) if isinstance(split, dict) and split.get("omega_rad_s") else float("nan")
    if not math.isfinite(omega_split):
        # Fallback: choose ω where cumulative integral reaches half.
        target = 0.5 * float(integral)
        cum = 0.0
        omega_split = float(omega[-1])
        for i in range(1, len(omega)):
            area = 0.5 * (dos[i - 1] + dos[i]) * (omega[i] - omega[i - 1])
            if cum + area >= target:
                omega_split = float(omega[i])
                break
            cum += area

    omega_split_ta: float | None = None
    omega_split_to: float | None = None
    if int(args.groups) in (3, 4):
        # For diamond Si (2 atoms/cell): total modes per cell=6 (=3 acoustic + 3 optical).
        # Within acoustic, TA:LA mode count is 2:1, so the TA proxy boundary corresponds to
        # 2/6 of the total mode count (i.e., 1/3 of the total DOS integral).
        omega_split_ta = _find_x_at_cum_fraction(omega, dos, frac=1.0 / 3.0)
    if int(args.groups) == 4:
        # Within optical, TO:LO mode count is 2:1, so the TO proxy boundary corresponds to
        # 5/6 of the total mode count (i.e., 0.5 + 2/6).
        omega_split_to = _find_x_at_cum_fraction(omega, dos, frac=5.0 / 6.0)

    nu_thz = [float(w / (2.0 * math.pi) / 1e12) for w in omega]
    nu_split_thz = float(omega_split / (2.0 * math.pi) / 1e12)
    nu_split_ta_thz = float(omega_split_ta / (2.0 * math.pi) / 1e12) if omega_split_ta is not None else float("nan")
    nu_split_to_thz = float(omega_split_to / (2.0 * math.pi) / 1e12) if omega_split_to is not None else float("nan")

    gamma_omega_basis_labels: list[str] = []
    gamma_omega_basis_weights: list[list[float]] = []
    gamma_omega_w_split = float("nan")
    if gamma_omega_model != "none":
        omega_max = float(omega[-1]) if omega else float("nan")
        if not (math.isfinite(omega_max) and omega_max > 0.0):
            raise SystemExit("[fail] invalid omega_max for --gamma-omega-model")
        gamma_omega_w_split = float(float(omega_split) / float(omega_max))
        if not (0.0 < gamma_omega_w_split < 1.0):
            raise SystemExit("[fail] invalid split w for --gamma-omega-model (omega_split/omega_max not in (0,1))")
        w_norm = [float(w) / float(omega_max) for w in omega]

        if gamma_omega_model == "linear_endpoints":
            w_low = [float(1.0 - wi) for wi in w_norm]
            w_high = [float(wi) for wi in w_norm]
            gamma_omega_basis_labels = ["low", "high"]
            gamma_omega_basis_weights = [w_low, w_high]
        elif gamma_omega_model == "bernstein2":
            w0 = [float((1.0 - wi) ** 2) for wi in w_norm]
            w1 = [float(2.0 * wi * (1.0 - wi)) for wi in w_norm]
            w2 = [float(wi**2) for wi in w_norm]
            gamma_omega_basis_labels = ["low", "mid", "high"]
            gamma_omega_basis_weights = [w0, w1, w2]
        elif gamma_omega_model == "pwlinear_split_leaky":
            warp_p = float(gamma_omega_pwlinear_warp_power)
            if not (math.isfinite(warp_p) and warp_p > 0.0):
                raise SystemExit("[fail] invalid --gamma-omega-pwlinear-warp-power (expected finite >0)")
            # Optional warp w -> w^p (monotone for p>0) to localize overlap to higher frequencies.
            ws_raw = float(gamma_omega_w_split)
            ws = float(ws_raw**warp_p)
            w_norm_warp = [float(wi**warp_p) for wi in w_norm]
            eps = float(gamma_omega_pwlinear_leak)
            if not (0.0 <= eps < 1.0):
                raise SystemExit("[fail] invalid --gamma-omega-pwlinear-leak (expected in [0,1))")
            w_low = []
            w_mid = []
            w_high = []
            for wi in w_norm_warp:
                wv = float(wi)
                if wv <= ws:
                    # Start from pwlinear_split (low/mid) and leak some weight into 'high'.
                    t = 0.0 if ws <= 0.0 else float(wv / ws)
                    leak = float(eps * t * t)
                    low0 = float((ws - wv) / ws) if ws > 0.0 else 1.0
                    mid0 = float(wv / ws) if ws > 0.0 else 0.0
                    w_low.append(float((1.0 - leak) * low0))
                    w_mid.append(float((1.0 - leak) * mid0))
                    w_high.append(float(leak))
                else:
                    # Start from pwlinear_split (mid/high) and leak some weight into 'low'.
                    t = 0.0 if (1.0 - ws) <= 0.0 else float((1.0 - wv) / (1.0 - ws))
                    leak = float(eps * t * t)
                    mid0 = float((1.0 - wv) / (1.0 - ws)) if (1.0 - ws) > 0.0 else 0.0
                    high0 = float((wv - ws) / (1.0 - ws)) if (1.0 - ws) > 0.0 else 1.0
                    w_low.append(float(leak))
                    w_mid.append(float((1.0 - leak) * mid0))
                    w_high.append(float((1.0 - leak) * high0))
            gamma_omega_basis_labels = ["low", "mid", "high"]
            gamma_omega_basis_weights = [w_low, w_mid, w_high]
        elif gamma_omega_model == "pwlinear_split":
            ws = float(gamma_omega_w_split)
            w_low: list[float] = []
            w_mid: list[float] = []
            w_high: list[float] = []
            for wi in w_norm:
                if float(wi) <= ws:
                    w_low.append(float((ws - float(wi)) / ws))
                    w_mid.append(float(float(wi) / ws))
                    w_high.append(0.0)
                else:
                    w_low.append(0.0)
                    w_mid.append(float((1.0 - float(wi)) / (1.0 - ws)))
                    w_high.append(float((float(wi) - ws) / (1.0 - ws)))
            gamma_omega_basis_labels = ["low", "mid", "high"]
            gamma_omega_basis_weights = [w_low, w_mid, w_high]
        else:
            raise SystemExit(f"[fail] unsupported --gamma-omega-model: {gamma_omega_model!r}")

        # Preallocate omega-basis Cv containers (computed inside the Cv loop).
        cv_gamma_omega = {lbl: [] for lbl in gamma_omega_basis_labels}
        u_gamma_omega = {lbl: [] for lbl in gamma_omega_basis_labels}
    else:
        cv_gamma_omega = {}
        u_gamma_omega = {}

    # θ(ω) = ħω/k_B using exact SI constants.
    h_J_s = 6.626_070_15e-34
    k_B_J_K = 1.380_649e-23
    n_A = 6.022_140_76e23
    R_J_molK = float(n_A * k_B_J_K)
    hbar_over_kb_K_s = float((h_J_s / (2.0 * math.pi)) / k_B_J_K)
    theta_k = [float(hbar_over_kb_K_s * w) for w in omega]

    # Cv basis functions from DOS split.
    cv_low: list[float] = []
    cv_high: list[float] = []
    cv_ta: list[float] = []
    cv_la: list[float] = []
    cv_opt: list[float] = []
    cv_to: list[float] = []
    cv_lo: list[float] = []

    global_softening: dict[str, object] | None = None
    global_scales: list[float] = [1.0 for _ in temps]
    if str(args.dos_softening) == "kim2015_linear_proxy":
        global_softening = _kim2015_linear_global_softening_scale(
            root=root,
            source_dirname=str(args.kim2015_source_dir),
            temps_k=temps,
        )
        scales_obj = global_softening.get("scales") if isinstance(global_softening, dict) else None
        if not isinstance(scales_obj, list) or len(scales_obj) != len(temps):
            raise SystemExit("[fail] invalid kim2015 global scales")
        global_scales = [float(x) for x in scales_obj]

    mode_softening: dict[str, object] | None = None
    mode_scales: dict[str, list[float]] | None = None
    mode_softening_mode = str(args.mode_softening)
    if mode_softening_mode in ("kim2015_fig2_features", "kim2015_fig2_features_eq8_quasi"):
        try:
            idx_300 = temps.index(300.0)
        except ValueError:
            raise SystemExit("[fail] internal: alpha(T) grid does not include 300 K (expected integer grid)")
        alpha_300k = float(alpha_obs[idx_300])
        mode_softening = _kim2015_fig2_mode_softening_scales(
            root=root,
            source_dirname=str(args.kim2015_fig2_source_dir),
            json_name=str(args.kim2015_fig2_json_name),
            temps_k=temps,
            groups=int(args.groups),
            eq8_quasiharmonic=(mode_softening_mode == "kim2015_fig2_features_eq8_quasi"),
            alpha_300k_per_k=float(alpha_300k),
        )
        scales_obj = mode_softening.get("scales") if isinstance(mode_softening, dict) else None
        if not isinstance(scales_obj, dict) or not scales_obj:
            raise SystemExit("[fail] invalid kim2015 fig2 mode scales")
        mode_scales = {str(k): [float(v) for v in vals] for k, vals in scales_obj.items() if isinstance(vals, list)}
        if not mode_scales or any(len(v) != len(temps) for v in mode_scales.values()):
            raise SystemExit("[fail] invalid kim2015 fig2 mode scales length")

    if gamma_omega_high_softening_delta != 0.0 or gamma_omega_softening_delta != 0.0:
        if mode_softening_mode != "kim2015_fig2_features":
            raise SystemExit(
                "[fail] --gamma-omega-(high-)softening-delta requires --mode-softening=kim2015_fig2_features"
            )
        if mode_scales is None:
            raise SystemExit("[fail] --gamma-omega-(high-)softening-delta requires valid mode softening scales")

    if gamma_omega_softening_fit != "none":
        if mode_softening_mode != "kim2015_fig2_features":
            raise SystemExit("[fail] --gamma-omega-softening-fit requires --mode-softening=kim2015_fig2_features")
        if mode_scales is None:
            raise SystemExit("[fail] --gamma-omega-softening-fit requires valid mode softening scales")

    gamma_trend_weights: dict[str, list[float]] | None = None
    if gamma_trend != "constant":
        if not use_bulk_modulus:
            raise SystemExit("[fail] --gamma-trend requires --use-bulk-modulus (γ units)")
        if gamma_trend in ("kim2015_fig2_softening_common", "kim2015_fig2_softening_common_centered300"):
            if mode_scales is None:
                raise SystemExit(
                    f"[fail] --gamma-trend={gamma_trend} requires --mode-softening=kim2015_fig2_features (or *_eq8_quasi)"
                )
            # g_i(T) := 1 - omega_scale_i(T) from Kim2015 Fig.2 (digitized), interpolated on the same grid as temps.
            try:
                idx_300 = temps.index(300.0)
            except ValueError:
                raise SystemExit("[fail] internal: alpha(T) grid does not include 300 K (expected integer grid)")

            def _centered(vals: list[float]) -> list[float]:
                g_raw = [1.0 - float(x) for x in vals]
                if gamma_trend == "kim2015_fig2_softening_common_centered300":
                    g0 = float(g_raw[idx_300])
                    g_raw = [float(x) - g0 for x in g_raw]
                    span = max(1e-12, max(abs(float(x)) for x in g_raw))
                    return [float(x) / span for x in g_raw]
                return g_raw

            if int(args.groups) == 2:
                gamma_trend_weights = {
                    "acoustic": _centered(mode_scales["acoustic"]),
                    "optical": _centered(mode_scales["optical"]),
                }
            elif int(args.groups) == 3:
                gamma_trend_weights = {
                    "ta": _centered(mode_scales["ta"]),
                    "la": _centered(mode_scales["la"]),
                    "optical": _centered(mode_scales["optical"]),
                }
            elif int(args.groups) == 4:
                gamma_trend_weights = {
                    "ta": _centered(mode_scales["ta"]),
                    "la": _centered(mode_scales["la"]),
                    "to": _centered(mode_scales["to"]),
                    "lo": _centered(mode_scales["lo"]),
                }
            else:
                raise SystemExit(f"[fail] unsupported groups for --gamma-trend: {args.groups}")
        elif gamma_trend == "linear_T":
            # g(T) := (T-300K)/T_max (clamped), shared across groups.
            t_max_ref = float(t_max)
            if not (math.isfinite(t_max_ref) and t_max_ref > 0.0):
                raise SystemExit(f"[fail] invalid alpha(T) t_max_k for --gamma-trend=linear_T: {t_max_ref}")
            t0 = 300.0
            g_shared = [min(1.0, max(-1.0, (float(t) - t0) / t_max_ref)) for t in temps]
            if int(args.groups) == 2:
                gamma_trend_weights = {"acoustic": g_shared, "optical": g_shared}
            elif int(args.groups) == 3:
                gamma_trend_weights = {"ta": g_shared, "la": g_shared, "optical": g_shared}
            elif int(args.groups) == 4:
                gamma_trend_weights = {"ta": g_shared, "la": g_shared, "to": g_shared, "lo": g_shared}
            else:
                raise SystemExit(f"[fail] unsupported groups for --gamma-trend: {args.groups}")
        else:
            raise SystemExit(f"[fail] unsupported --gamma-trend: {gamma_trend!r}")

    omega_np = np.asarray(omega, dtype=float)
    g_np = np.asarray(g_per_atom, dtype=float)
    theta_np = np.asarray(theta_k, dtype=float)

    omega_hi, g_hi, theta_hi, _ = _append_boundary_point(
        omega=omega_np, g_per_atom=g_np, theta_k=theta_np, omega_boundary=float(omega_split)
    )

    softening_enabled = str(args.optical_softening) != "none"
    softening_mode = str(args.optical_softening)
    f_grid: np.ndarray | None = None
    cv_hi_candidates: list[np.ndarray] | None = None
    cv_to_candidates: list[np.ndarray] | None = None
    cv_lo_candidates: list[np.ndarray] | None = None
    softening_scan: list[dict[str, float]] = []
    softening_best: dict[str, float] | None = None
    raman_shape_g: list[float] | None = None
    raman_source_path: Path | None = None

    # Harmonic reference (no softening): compute all groups once.
    u_low: list[float] = []
    u_high: list[float] = []
    u_ta: list[float] = []
    u_la: list[float] = []
    u_opt: list[float] = []
    u_to: list[float] = []
    u_lo: list[float] = []
    for ti, t in enumerate(temps):
        t_f = float(t)
        scale_all = float(global_scales[ti]) if global_scales is not None else 1.0
        if mode_scales is None:
            if cv_omega_dependence == "harmonic":
                factors = _cv_factor_np((theta_np * scale_all) / t_f)
                integrand = (g_np * factors).tolist()
            else:
                u_factor = _u_th_factor_np((theta_np * scale_all) / t_f)
                integrand = (g_np * u_factor).tolist()

            if gamma_omega_model != "none":
                for lbl, w_basis in zip(gamma_omega_basis_labels, gamma_omega_basis_weights):
                    y_w = [float(integrand[i]) * float(w_basis[i]) for i in range(len(integrand))]
                    i_w = _trapz_xy(omega, y_w)
                    if cv_omega_dependence == "harmonic":
                        cv_gamma_omega[lbl].append(float(R_J_molK * i_w))
                    else:
                        u_gamma_omega[lbl].append(float(R_J_molK * t_f * i_w))
            if int(args.groups) == 2:
                i_lo, i_hi0 = _integrate_split_trapz(omega, integrand, x_split=omega_split)
                if cv_omega_dependence == "harmonic":
                    cv_low.append(float(R_J_molK * i_lo))
                    cv_high.append(float(R_J_molK * i_hi0))
                else:
                    u_low.append(float(R_J_molK * t_f * i_lo))
                    u_high.append(float(R_J_molK * t_f * i_hi0))
            elif int(args.groups) == 3:
                assert omega_split_ta is not None
                i_ta0 = _integrate_range_trapz(omega, integrand, x_min=0.0, x_max=float(omega_split_ta))
                i_la0 = _integrate_range_trapz(omega, integrand, x_min=float(omega_split_ta), x_max=float(omega_split))
                i_op0 = _integrate_range_trapz(omega, integrand, x_min=float(omega_split), x_max=float(omega[-1]))
                if cv_omega_dependence == "harmonic":
                    cv_ta.append(float(R_J_molK * i_ta0))
                    cv_la.append(float(R_J_molK * i_la0))
                    cv_opt.append(float(R_J_molK * i_op0))
                else:
                    u_ta.append(float(R_J_molK * t_f * i_ta0))
                    u_la.append(float(R_J_molK * t_f * i_la0))
                    u_opt.append(float(R_J_molK * t_f * i_op0))
            else:
                assert omega_split_ta is not None
                assert omega_split_to is not None
                i_ta0 = _integrate_range_trapz(omega, integrand, x_min=0.0, x_max=float(omega_split_ta))
                i_la0 = _integrate_range_trapz(omega, integrand, x_min=float(omega_split_ta), x_max=float(omega_split))
                i_to0 = _integrate_range_trapz(omega, integrand, x_min=float(omega_split), x_max=float(omega_split_to))
                i_lo0 = _integrate_range_trapz(omega, integrand, x_min=float(omega_split_to), x_max=float(omega[-1]))
                if cv_omega_dependence == "harmonic":
                    cv_ta.append(float(R_J_molK * i_ta0))
                    cv_la.append(float(R_J_molK * i_la0))
                    cv_to.append(float(R_J_molK * i_to0))
                    cv_lo.append(float(R_J_molK * i_lo0))
                else:
                    u_ta.append(float(R_J_molK * t_f * i_ta0))
                    u_la.append(float(R_J_molK * t_f * i_la0))
                    u_to.append(float(R_J_molK * t_f * i_to0))
                    u_lo.append(float(R_J_molK * t_f * i_lo0))
        else:
            # Mode-dependent ω scaling per group (fixed primary constraint).
            if int(args.groups) == 2:
                s_ac = float(scale_all) * float(mode_scales["acoustic"][ti])
                s_op = float(scale_all) * float(mode_scales["optical"][ti])
                if cv_omega_dependence == "harmonic":
                    integrand_ac = (g_np * _cv_factor_np((theta_np * s_ac) / t_f)).tolist()
                    integrand_op = (g_np * _cv_factor_np((theta_np * s_op) / t_f)).tolist()
                else:
                    integrand_ac = (g_np * _u_th_factor_np((theta_np * s_ac) / t_f)).tolist()
                    integrand_op = (g_np * _u_th_factor_np((theta_np * s_op) / t_f)).tolist()

                if gamma_omega_model != "none":
                    for lbl, w_basis in zip(gamma_omega_basis_labels, gamma_omega_basis_weights):
                        y_ac = [float(integrand_ac[i]) * float(w_basis[i]) for i in range(len(integrand_ac))]
                        y_op = [float(integrand_op[i]) * float(w_basis[i]) for i in range(len(integrand_op))]
                        i_w = _integrate_range_trapz(omega, y_ac, x_min=0.0, x_max=float(omega_split)) + _integrate_range_trapz(
                            omega, y_op, x_min=float(omega_split), x_max=float(omega[-1])
                        )
                        if cv_omega_dependence == "harmonic":
                            cv_gamma_omega[lbl].append(float(R_J_molK * i_w))
                        else:
                            u_gamma_omega[lbl].append(float(R_J_molK * t_f * i_w))
                i_lo = _integrate_range_trapz(omega, integrand_ac, x_min=0.0, x_max=float(omega_split))
                i_hi = _integrate_range_trapz(omega, integrand_op, x_min=float(omega_split), x_max=float(omega[-1]))
                if cv_omega_dependence == "harmonic":
                    cv_low.append(float(R_J_molK * i_lo))
                    cv_high.append(float(R_J_molK * i_hi))
                else:
                    u_low.append(float(R_J_molK * t_f * i_lo))
                    u_high.append(float(R_J_molK * t_f * i_hi))
            elif int(args.groups) == 3:
                assert omega_split_ta is not None
                s_ta = float(scale_all) * float(mode_scales["ta"][ti])
                s_la = float(scale_all) * float(mode_scales["la"][ti])
                s_op = float(scale_all) * float(mode_scales["optical"][ti])
                if cv_omega_dependence == "harmonic":
                    integrand_ta = (g_np * _cv_factor_np((theta_np * s_ta) / t_f)).tolist()
                    integrand_la = (g_np * _cv_factor_np((theta_np * s_la) / t_f)).tolist()
                    integrand_op = (g_np * _cv_factor_np((theta_np * s_op) / t_f)).tolist()
                else:
                    integrand_ta = (g_np * _u_th_factor_np((theta_np * s_ta) / t_f)).tolist()
                    integrand_la = (g_np * _u_th_factor_np((theta_np * s_la) / t_f)).tolist()
                    integrand_op = (g_np * _u_th_factor_np((theta_np * s_op) / t_f)).tolist()

                if gamma_omega_model != "none":
                    for lbl, w_basis in zip(gamma_omega_basis_labels, gamma_omega_basis_weights):
                        y_ta = [float(integrand_ta[i]) * float(w_basis[i]) for i in range(len(integrand_ta))]
                        y_la = [float(integrand_la[i]) * float(w_basis[i]) for i in range(len(integrand_la))]
                        y_op = [float(integrand_op[i]) * float(w_basis[i]) for i in range(len(integrand_op))]
                        i_w = (
                            _integrate_range_trapz(omega, y_ta, x_min=0.0, x_max=float(omega_split_ta))
                            + _integrate_range_trapz(omega, y_la, x_min=float(omega_split_ta), x_max=float(omega_split))
                            + _integrate_range_trapz(omega, y_op, x_min=float(omega_split), x_max=float(omega[-1]))
                        )
                        if cv_omega_dependence == "harmonic":
                            cv_gamma_omega[lbl].append(float(R_J_molK * i_w))
                        else:
                            u_gamma_omega[lbl].append(float(R_J_molK * t_f * i_w))
                i_ta0 = _integrate_range_trapz(omega, integrand_ta, x_min=0.0, x_max=float(omega_split_ta))
                i_la0 = _integrate_range_trapz(
                    omega, integrand_la, x_min=float(omega_split_ta), x_max=float(omega_split)
                )
                i_op0 = _integrate_range_trapz(omega, integrand_op, x_min=float(omega_split), x_max=float(omega[-1]))
                if cv_omega_dependence == "harmonic":
                    cv_ta.append(float(R_J_molK * i_ta0))
                    cv_la.append(float(R_J_molK * i_la0))
                    cv_opt.append(float(R_J_molK * i_op0))
                else:
                    u_ta.append(float(R_J_molK * t_f * i_ta0))
                    u_la.append(float(R_J_molK * t_f * i_la0))
                    u_opt.append(float(R_J_molK * t_f * i_op0))
            else:
                assert omega_split_ta is not None
                assert omega_split_to is not None
                s_ta = float(scale_all) * float(mode_scales["ta"][ti])
                s_la = float(scale_all) * float(mode_scales["la"][ti])
                s_to = float(scale_all) * float(mode_scales["to"][ti])
                s_lo = float(scale_all) * float(mode_scales["lo"][ti])
                if cv_omega_dependence == "harmonic":
                    integrand_ta = (g_np * _cv_factor_np((theta_np * s_ta) / t_f)).tolist()
                    integrand_la = (g_np * _cv_factor_np((theta_np * s_la) / t_f)).tolist()
                    integrand_to = (g_np * _cv_factor_np((theta_np * s_to) / t_f)).tolist()
                    integrand_lo = (g_np * _cv_factor_np((theta_np * s_lo) / t_f)).tolist()
                else:
                    integrand_ta = (g_np * _u_th_factor_np((theta_np * s_ta) / t_f)).tolist()
                    integrand_la = (g_np * _u_th_factor_np((theta_np * s_la) / t_f)).tolist()
                    integrand_to = (g_np * _u_th_factor_np((theta_np * s_to) / t_f)).tolist()
                    integrand_lo = (g_np * _u_th_factor_np((theta_np * s_lo) / t_f)).tolist()

                if gamma_omega_model != "none":
                    for lbl, w_basis in zip(gamma_omega_basis_labels, gamma_omega_basis_weights):
                        y_ta = [float(integrand_ta[i]) * float(w_basis[i]) for i in range(len(integrand_ta))]
                        y_la = [float(integrand_la[i]) * float(w_basis[i]) for i in range(len(integrand_la))]
                        y_to = [float(integrand_to[i]) * float(w_basis[i]) for i in range(len(integrand_to))]
                        y_lo = [float(integrand_lo[i]) * float(w_basis[i]) for i in range(len(integrand_lo))]
                        i_w = (
                            _integrate_range_trapz(omega, y_ta, x_min=0.0, x_max=float(omega_split_ta))
                            + _integrate_range_trapz(omega, y_la, x_min=float(omega_split_ta), x_max=float(omega_split))
                            + _integrate_range_trapz(omega, y_to, x_min=float(omega_split), x_max=float(omega_split_to))
                            + _integrate_range_trapz(omega, y_lo, x_min=float(omega_split_to), x_max=float(omega[-1]))
                        )
                        if cv_omega_dependence == "harmonic":
                            cv_gamma_omega[lbl].append(float(R_J_molK * i_w))
                        else:
                            u_gamma_omega[lbl].append(float(R_J_molK * t_f * i_w))
                i_ta0 = _integrate_range_trapz(omega, integrand_ta, x_min=0.0, x_max=float(omega_split_ta))
                i_la0 = _integrate_range_trapz(
                    omega, integrand_la, x_min=float(omega_split_ta), x_max=float(omega_split)
                )
                i_to0 = _integrate_range_trapz(
                    omega, integrand_to, x_min=float(omega_split), x_max=float(omega_split_to)
                )
                i_lo0 = _integrate_range_trapz(
                    omega, integrand_lo, x_min=float(omega_split_to), x_max=float(omega[-1])
                )
                if cv_omega_dependence == "harmonic":
                    cv_ta.append(float(R_J_molK * i_ta0))
                    cv_la.append(float(R_J_molK * i_la0))
                    cv_to.append(float(R_J_molK * i_to0))
                    cv_lo.append(float(R_J_molK * i_lo0))
                else:
                    u_ta.append(float(R_J_molK * t_f * i_ta0))
                    u_la.append(float(R_J_molK * t_f * i_la0))
                    u_to.append(float(R_J_molK * t_f * i_to0))
                    u_lo.append(float(R_J_molK * t_f * i_lo0))

    if cv_omega_dependence != "harmonic":
        if int(args.groups) == 2:
            cv_low = _numeric_dydx(temps, u_low)
            cv_high = _numeric_dydx(temps, u_high)
        elif int(args.groups) == 3:
            cv_ta = _numeric_dydx(temps, u_ta)
            cv_la = _numeric_dydx(temps, u_la)
            cv_opt = _numeric_dydx(temps, u_opt)
        else:
            cv_ta = _numeric_dydx(temps, u_ta)
            cv_la = _numeric_dydx(temps, u_la)
            cv_to = _numeric_dydx(temps, u_to)
            cv_lo = _numeric_dydx(temps, u_lo)
        if gamma_omega_model != "none":
            for lbl in gamma_omega_basis_labels:
                cv_gamma_omega[lbl] = _numeric_dydx(temps, u_gamma_omega[lbl])

    # Optional optical softening scan (linear in T/T_max; fit f by train SSE).
    if softening_enabled:
        if softening_mode not in ("linear_fit", "raman_shape_fit"):
            raise SystemExit(f"[fail] unsupported optical_softening mode: {softening_mode!r}")
        if float(args.softening_step_frac) <= 0.0 or float(args.softening_max_frac_at_tmax) < 0.0:
            raise SystemExit("[fail] invalid softening grid params")

        f_max = float(args.softening_max_frac_at_tmax)
        f_step = float(args.softening_step_frac)
        f_grid = np.arange(0.0, f_max + 0.5 * f_step, f_step, dtype=float)
        if len(f_grid) < 1:
            f_grid = np.array([0.0], dtype=float)

        if softening_mode == "linear_fit":
            t_max_ref = float(max(temps))
        else:
            raman_source_path = _ROOT / "data" / "quantum" / "sources" / str(args.raman_source_dir) / "extracted_values.json"
            raman_obj = _read_json(raman_source_path)
            rows = raman_obj.get("rows", [])
            if not isinstance(rows, list) or len(rows) < 5:
                raise SystemExit(f"[fail] invalid Raman extracted rows: {raman_source_path}")
            t_src = [float(r["t_k"]) for r in rows]
            g_src = [float(r["softening_shape_0to1"]) for r in rows]
            raman_shape_g = _interp_piecewise_linear(t_src, g_src, [float(t) for t in temps])

        f_grid_col = f_grid.reshape((-1, 1))  # (n_f,1)
        cv_hi_mat = np.zeros((len(f_grid), len(temps)), dtype=float)
        cv_to_mat = np.zeros((len(f_grid), len(temps)), dtype=float) if int(args.groups) == 4 else None
        cv_lo_mat = np.zeros((len(f_grid), len(temps)), dtype=float) if int(args.groups) == 4 else None
        omega_hi_list = omega_hi.tolist()
        for ti, t in enumerate(temps):
            t_f = float(t)
            scale_all = float(global_scales[ti]) if global_scales is not None else 1.0
            if softening_mode == "linear_fit":
                scale = 1.0 - f_grid_col * (t_f / t_max_ref)
            else:
                assert raman_shape_g is not None
                g_t = float(raman_shape_g[ti])
                scale = 1.0 - f_grid_col * g_t
            scale = np.clip(scale, float(args.softening_min_scale), 1.0)
            x = (theta_hi.reshape((1, -1)) * scale_all * scale) / t_f
            factors_hi = _cv_factor_np(x)
            integrand_hi = g_hi.reshape((1, -1)) * factors_hi
            integral_hi = np.trapezoid(integrand_hi, omega_hi, axis=1)
            cv_hi_mat[:, ti] = float(R_J_molK) * integral_hi

            if int(args.groups) == 4:
                assert cv_to_mat is not None and cv_lo_mat is not None
                assert omega_split_to is not None
                for fi in range(int(len(f_grid))):
                    y_row = integrand_hi[fi, :].tolist()
                    i_to = _integrate_range_trapz(
                        omega_hi_list,
                        y_row,
                        x_min=float(omega_split),
                        x_max=float(omega_split_to),
                    )
                    i_lo = _integrate_range_trapz(
                        omega_hi_list,
                        y_row,
                        x_min=float(omega_split_to),
                        x_max=float(omega_hi_list[-1]),
                    )
                    cv_to_mat[fi, ti] = float(R_J_molK) * float(i_to)
                    cv_lo_mat[fi, ti] = float(R_J_molK) * float(i_lo)
        cv_hi_candidates = [cv_hi_mat[i, :] for i in range(cv_hi_mat.shape[0])]
        if int(args.groups) == 4:
            assert cv_to_mat is not None and cv_lo_mat is not None
            cv_to_candidates = [cv_to_mat[i, :] for i in range(cv_to_mat.shape[0])]
            cv_lo_candidates = [cv_lo_mat[i, :] for i in range(cv_lo_mat.shape[0])]

    if dos_mode == "kim2015_fig1_energy":
        fig1_src = (
            root
            / "data"
            / "quantum"
            / "sources"
            / str(args.kim2015_fig1_source_dir)
            / str(args.kim2015_fig1_json_name)
        )
        if not fig1_src.exists():
            raise SystemExit(
                f"[fail] missing: {fig1_src}\n"
                "Run: python -B scripts/quantum/extract_silicon_phonon_dos_kim2015_fig1_digitize.py"
            )

        fig1_obj = _read_json(fig1_src)
        curves_obj = fig1_obj.get("curves")
        if not isinstance(curves_obj, list) or len(curves_obj) < 6:
            raise SystemExit(f"[fail] invalid curves in fig1 json: {fig1_src}")

        src_info = fig1_obj.get("source") if isinstance(fig1_obj.get("source"), dict) else {}
        energy_max_mev = float(src_info.get("energy_max_meV", 80.0))
        if not (energy_max_mev > 0.0):
            raise SystemExit("[fail] invalid energy_max_meV in fig1 source")

        e_step = float(args.kim2015_fig1_egrid_step_mev)
        if not (e_step > 0.0):
            raise SystemExit("[fail] invalid --kim2015-fig1-egrid-step-mev")

        e_grid = np.arange(0.0, energy_max_mev + 0.5 * e_step, e_step, dtype=float)
        e_grid_list = [float(x) for x in e_grid.tolist()]

        temps_src: list[float] = []
        g_src_rows: list[np.ndarray] = []
        for c in curves_obj:
            if not isinstance(c, dict):
                continue
            t_k = c.get("temperature_K")
            rows = c.get("rows")
            if not isinstance(t_k, int) or not isinstance(rows, list) or len(rows) < 10:
                continue
            es = [float(r["E_meV"]) for r in rows if isinstance(r, dict) and "E_meV" in r]
            gs = [float(r["g_per_meV"]) for r in rows if isinstance(r, dict) and "g_per_meV" in r]
            if len(es) != len(gs) or len(es) < 10:
                continue
            pairs = sorted(zip(es, gs))
            e_sorted = np.asarray([p[0] for p in pairs], dtype=float)
            g_sorted = np.asarray([max(0.0, float(p[1])) for p in pairs], dtype=float)
            g_interp = np.interp(e_grid, e_sorted, g_sorted, left=0.0, right=0.0)
            g_interp = np.clip(g_interp, 0.0, float("inf"))
            area = float(np.trapezoid(g_interp, e_grid))
            if area <= 0.0:
                raise SystemExit(f"[fail] non-positive area after resample: T={t_k} K")
            g_interp /= area
            temps_src.append(float(t_k))
            g_src_rows.append(g_interp)

        if len(temps_src) < 6:
            raise SystemExit(f"[fail] too few usable curves in fig1 json: n={len(temps_src)}")

        order = sorted(range(len(temps_src)), key=lambda i: temps_src[i])
        temps_src = [temps_src[i] for i in order]
        g_src_mat = np.stack([g_src_rows[i] for i in order], axis=0)  # (n_T, n_E)

        def _dos_at_t(t_k: float) -> np.ndarray:
            t_k = float(t_k)
            if t_k <= float(temps_src[0]):
                g = np.array(g_src_mat[0, :], dtype=float)
            elif t_k >= float(temps_src[-1]):
                g = np.array(g_src_mat[-1, :], dtype=float)
            else:
                j = int(bisect.bisect_left(temps_src, t_k))
                j = min(max(1, j), len(temps_src) - 1)
                t0 = float(temps_src[j - 1])
                t1 = float(temps_src[j])
                w = 0.0 if t1 == t0 else float((t_k - t0) / (t1 - t0))
                g = (1.0 - w) * g_src_mat[j - 1, :] + w * g_src_mat[j, :]
            g = np.clip(g, 0.0, float("inf"))
            area = float(np.trapezoid(g, e_grid))
            if area <= 0.0:
                return np.array(g_src_mat[0, :], dtype=float)
            return g / area

        # Recompute Cv(T) basis using g_T(ε).
        cv_low.clear()
        cv_high.clear()
        cv_ta.clear()
        cv_la.clear()
        cv_opt.clear()
        cv_to.clear()
        cv_lo.clear()

        # Physical constants (exact SI).
        eV_J = 1.602_176_634e-19
        h_J_s = 6.626_070_15e-34
        hbar_J_s = float(h_J_s / (2.0 * math.pi))
        k_B_J_K = 1.380_649e-23
        n_A = 6.022_140_76e23
        R_J_molK = float(n_A * k_B_J_K)
        meV_J = float(1e-3 * eV_J)
        e_grid_J = e_grid * meV_J

        for t in temps:
            t_f = float(t)
            g_t = _dos_at_t(t_f)
            x = e_grid_J / (k_B_J_K * t_f)
            factors = _cv_factor_np(x)
            integrand = (g_t * factors).tolist()
            g_list = g_t.tolist()

            e_split = float(_find_x_at_cum_fraction(e_grid_list, g_list, frac=0.5))
            e_split_ta = float("nan")
            e_split_to = float("nan")
            if int(args.groups) in (3, 4):
                e_split_ta = float(_find_x_at_cum_fraction(e_grid_list, g_list, frac=1.0 / 3.0))
            if int(args.groups) == 4:
                e_split_to = float(_find_x_at_cum_fraction(e_grid_list, g_list, frac=5.0 / 6.0))

            if int(args.groups) == 2:
                i_lo, i_hi = _integrate_split_trapz(e_grid_list, integrand, x_split=e_split)
                cv_low.append(float(3.0 * R_J_molK * i_lo))
                cv_high.append(float(3.0 * R_J_molK * i_hi))
            elif int(args.groups) == 3:
                i_ta = _integrate_range_trapz(e_grid_list, integrand, x_min=0.0, x_max=float(e_split_ta))
                i_la = _integrate_range_trapz(e_grid_list, integrand, x_min=float(e_split_ta), x_max=float(e_split))
                i_op = _integrate_range_trapz(e_grid_list, integrand, x_min=float(e_split), x_max=float(e_grid_list[-1]))
                cv_ta.append(float(3.0 * R_J_molK * i_ta))
                cv_la.append(float(3.0 * R_J_molK * i_la))
                cv_opt.append(float(3.0 * R_J_molK * i_op))
            else:
                i_ta = _integrate_range_trapz(e_grid_list, integrand, x_min=0.0, x_max=float(e_split_ta))
                i_la = _integrate_range_trapz(e_grid_list, integrand, x_min=float(e_split_ta), x_max=float(e_split))
                i_to = _integrate_range_trapz(e_grid_list, integrand, x_min=float(e_split), x_max=float(e_split_to))
                i_lo = _integrate_range_trapz(e_grid_list, integrand, x_min=float(e_split_to), x_max=float(e_grid_list[-1]))
                cv_ta.append(float(3.0 * R_J_molK * i_ta))
                cv_la.append(float(3.0 * R_J_molK * i_la))
                cv_to.append(float(3.0 * R_J_molK * i_to))
                cv_lo.append(float(3.0 * R_J_molK * i_lo))

        # Build a reference DOS (t_ref=300 K) in ω units for the top-panel plot.
        t_ref = 300.0
        g_ref = _dos_at_t(t_ref)
        omega = (e_grid_J / hbar_J_s).tolist()
        g_per_atom = (3.0 * hbar_J_s * (g_ref / meV_J)).tolist()
        dos = [float(x) for x in g_per_atom]  # dummy per m^3 (n_atoms_m3≈1) to reuse the existing plot code
        integral = float(_trapz_xy(omega, dos))
        n_atoms_m3 = float(integral / 3.0) if integral > 0.0 else float("nan")
        integral_per_atom = float(_trapz_xy(omega, g_per_atom)) if omega else float("nan")

        omega_split = float(_find_x_at_cum_fraction(omega, dos, frac=0.5))
        omega_split_ta = (
            float(_find_x_at_cum_fraction(omega, dos, frac=1.0 / 3.0)) if int(args.groups) in (3, 4) else None
        )
        omega_split_to = (
            float(_find_x_at_cum_fraction(omega, dos, frac=5.0 / 6.0)) if int(args.groups) == 4 else None
        )

        nu_thz = [float(w / (2.0 * math.pi) / 1e12) for w in omega]
        nu_split_thz = float(omega_split / (2.0 * math.pi) / 1e12)
        nu_split_ta_thz = (
            float(omega_split_ta / (2.0 * math.pi) / 1e12) if omega_split_ta is not None else float("nan")
        )
        nu_split_to_thz = (
            float(omega_split_to / (2.0 * math.pi) / 1e12) if omega_split_to is not None else float("nan")
        )

        # Disable additional softening knobs in this mode (temperature dependence is already in the DOS data).
        global_softening = None
        softening_enabled = False
        softening_mode = "none"
        f_grid = None
        cv_hi_candidates = None
        cv_to_candidates = None
        cv_lo_candidates = None
        softening_scan = []
        softening_best = None
        raman_shape_g = None
        raman_source_path = None

        dos_src = fig1_src
        fig1_ref = {
            "source_path": str(fig1_src),
            "source_sha256": _sha256(fig1_src),
            "t_ref_K": float(t_ref),
            "energy_grid_step_meV": float(e_step),
            "energy_max_meV": float(energy_max_mev),
        }

    # Fit range (global): same as 7.14.11/7.14.18.
    fit_min_k = 50.0
    fit_max_k = float(t_max)
    fit_idx = [i for i, t in enumerate(temps) if fit_min_k <= float(t) <= fit_max_k]
    if len(fit_idx) < 100:
        raise SystemExit(f"[fail] not enough fit points: n={len(fit_idx)} in [{fit_min_k},{fit_max_k}] K")

    def _basis_list(values: list[float]) -> list[float]:
        if not use_bulk_modulus:
            return values
        if len(values) != len(inv_bv):
            raise ValueError("basis length mismatch")
        return [float(values[i]) * float(inv_bv[i]) for i in range(len(values))]

    def _basis_np(values: np.ndarray) -> np.ndarray:
        if not use_bulk_modulus:
            return values
        if values.shape[0] != inv_bv_np.shape[0]:
            raise ValueError("basis length mismatch")
        return values * inv_bv_np

    softening_global_frac: float | None = None
    delta_gamma = 0.0
    gamma_omega_softening_fit_delta = 0.0

    gamma_omega_coeffs: dict[str, float] | None = None
    gamma_omega_fit: dict[str, float] | None = None

    if gamma_omega_model != "none":
        x_cols = [_basis_list(cv_gamma_omega[lbl]) for lbl in gamma_omega_basis_labels]
        g_opt_centered = [0.0 for _ in temps]
        high_idx = None
        if gamma_omega_high_softening_delta != 0.0 or gamma_omega_softening_delta != 0.0 or gamma_omega_softening_fit != "none":
            if mode_scales is None:
                raise SystemExit("[fail] internal: gamma_omega_softening_delta set but mode_scales is missing")
            scales_ac: list[float]
            scales_opt: list[float]
            if isinstance(mode_scales.get("optical"), list):
                scales_opt = [float(v) for v in mode_scales["optical"]]
            elif isinstance(mode_scales.get("to"), list) and isinstance(mode_scales.get("lo"), list):
                to = [float(v) for v in mode_scales["to"]]
                lo = [float(v) for v in mode_scales["lo"]]
                if len(to) != len(lo):
                    raise SystemExit("[fail] invalid mode scales for optical composite (to/lo length mismatch)")
                scales_opt = [float((2.0 * float(sto) + float(slo)) / 3.0) for sto, slo in zip(to, lo)]
            else:
                raise SystemExit(
                    "[fail] --gamma-omega-(high-)softening-delta requires optical scale (groups=2/3) or to/lo scales (groups=4)"
                )

            if isinstance(mode_scales.get("acoustic"), list):
                scales_ac = [float(v) for v in mode_scales["acoustic"]]
            elif isinstance(mode_scales.get("ta"), list) and isinstance(mode_scales.get("la"), list):
                ta = [float(v) for v in mode_scales["ta"]]
                la = [float(v) for v in mode_scales["la"]]
                if len(ta) != len(la):
                    raise SystemExit("[fail] invalid mode scales for acoustic composite (ta/la length mismatch)")
                scales_ac = [float((2.0 * float(sta) + float(sla)) / 3.0) for sta, sla in zip(ta, la)]
            else:
                raise SystemExit(
                    "[fail] --gamma-omega-(fit)softening requires acoustic scale (groups=2) or ta/la scales (groups=3/4)"
                )

            if len(scales_opt) != len(temps):
                raise SystemExit("[fail] invalid optical scale length for --gamma-omega-(high-)softening-delta")
            if len(scales_ac) != len(temps):
                raise SystemExit("[fail] invalid acoustic scale length for --gamma-omega-(fit)softening")

            g_raw_opt = [float(1.0 - float(s)) for s in scales_opt]
            g_raw_ac = [float(1.0 - float(s)) for s in scales_ac]
            try:
                idx_300 = temps.index(300.0)
            except ValueError:
                raise SystemExit("[fail] internal: alpha(T) grid does not include 300 K (expected integer grid)")
            g0_opt = float(g_raw_opt[idx_300])
            g0_ac = float(g_raw_ac[idx_300])
            g_opt_centered = [float(g_raw_opt[i] - g0_opt) for i in range(len(g_raw_opt))]
            g_ac_centered = [float(g_raw_ac[i] - g0_ac) for i in range(len(g_raw_ac))]

            if gamma_omega_high_softening_delta != 0.0 or gamma_omega_softening_fit == "high_centered300":
                if "high" not in gamma_omega_basis_labels:
                    raise SystemExit("[fail] --gamma-omega-high-softening-delta requires a 'high' basis label")
                high_idx = int(gamma_omega_basis_labels.index("high"))

        alpha_corr = [0.0 for _ in temps]
        x_delta = [0.0 for _ in temps]
        coeffs: list[float] = []
        fit: dict[str, object] = {}
        y_fit = alpha_obs

        if gamma_omega_softening_fit != "none":
            if gamma_omega_softening_fit == "common_centered300":
                g_cols = [list(g_opt_centered) for _ in x_cols]
            elif gamma_omega_softening_fit == "high_centered300":
                if high_idx is None:
                    raise SystemExit("[fail] internal: high_idx missing for --gamma-omega-softening-fit=high_centered300")
                g_cols = [[0.0 for _ in temps] for _ in x_cols]
                g_cols[high_idx] = list(g_opt_centered)
            elif gamma_omega_softening_fit == "by_label_centered300":
                g_mid_centered = [0.5 * (float(g_ac_centered[i]) + float(g_opt_centered[i])) for i in range(len(temps))]
                g_cols = []
                for lbl in gamma_omega_basis_labels:
                    if str(lbl) == "low":
                        g_cols.append(list(g_ac_centered))
                    elif str(lbl) == "high":
                        g_cols.append(list(g_opt_centered))
                    elif str(lbl) == "mid":
                        g_cols.append(list(g_mid_centered))
                    else:
                        g_cols.append(list(g_opt_centered))
            else:
                raise SystemExit(f"[fail] unsupported --gamma-omega-softening-fit: {gamma_omega_softening_fit!r}")

            sign_constraints = ([-1] + [1] * (len(x_cols) - 1)) if bool(args.enforce_signs) else [0] * len(x_cols)
            fit = _fit_basis_plus_delta_weighted_ls(
                x_cols=[list(c) for c in x_cols],
                g_cols=[list(g) for g in g_cols],
                y=alpha_obs,
                sigma=sigma_fit,
                idx=fit_idx,
                enforce_signs=bool(args.enforce_signs),
                sign_constraints=sign_constraints,
                ridge_factor=float(ridge_factor),
                delta_ridge_factor=float(delta_ridge_factor),
            )
            coeffs = [float(v) for v in fit.get("a", [])]
            gamma_omega_softening_fit_delta = float(fit.get("delta_gamma", 0.0))
            if len(coeffs) != len(x_cols):
                raise SystemExit("[fail] internal: invalid fit size for gamma_omega_softening_fit")
            x_delta = [sum(float(g_cols[j][i]) * float(x_cols[j][i]) for j in range(len(x_cols))) for i in range(len(temps))]
            for i in range(len(temps)):
                alpha_corr[i] = float(gamma_omega_softening_fit_delta) * float(x_delta[i])
        else:
            # Fixed delta correction (optional), treated as a frozen physics term (no fit parameter).
            if gamma_omega_softening_delta != 0.0:
                g_cols = [list(g_opt_centered) for _ in x_cols]
                x_delta = [sum(float(g_cols[j][i]) * float(x_cols[j][i]) for j in range(len(x_cols))) for i in range(len(temps))]
                for i in range(len(temps)):
                    alpha_corr[i] = float(gamma_omega_softening_delta) * float(x_delta[i])
            elif gamma_omega_high_softening_delta != 0.0:
                if high_idx is None:
                    raise SystemExit("[fail] internal: high_idx missing for gamma omega high delta correction")
                g_cols = [[0.0 for _ in temps] for _ in x_cols]
                g_cols[high_idx] = list(g_opt_centered)
                x_delta = [sum(float(g_cols[j][i]) * float(x_cols[j][i]) for j in range(len(x_cols))) for i in range(len(temps))]
                for i in range(len(temps)):
                    alpha_corr[i] = float(gamma_omega_high_softening_delta) * float(x_delta[i])

            y_fit = [float(alpha_obs[i]) - float(alpha_corr[i]) for i in range(len(temps))] if any(alpha_corr) else alpha_obs

            if len(x_cols) == 2:
                fit = _fit_two_basis_weighted_ls(
                    x1=x_cols[0],
                    x2=x_cols[1],
                    y=y_fit,
                    sigma=sigma_fit,
                    idx=fit_idx,
                    enforce_signs=bool(args.enforce_signs),
                    ridge_factor=float(ridge_factor),
                )
                coeffs = [float(fit["A_low_mol_per_J"]), float(fit["A_high_mol_per_J"])]
            else:
                fit = _fit_three_basis_weighted_ls(
                    x1=x_cols[0],
                    x2=x_cols[1],
                    x3=x_cols[2],
                    y=y_fit,
                    sigma=sigma_fit,
                    idx=fit_idx,
                    enforce_signs=bool(args.enforce_signs),
                    ridge_factor=float(ridge_factor),
                )
                coeffs = [float(fit["A_1_mol_per_J"]), float(fit["A_2_mol_per_J"]), float(fit["A_3_mol_per_J"])]

        gamma_omega_coeffs = {str(lbl): float(coeffs[i]) for i, lbl in enumerate(gamma_omega_basis_labels)}
        gamma_omega_fit = {str(k): float(v) for k, v in fit.items() if isinstance(v, (int, float)) and math.isfinite(float(v))}

        alpha_pred = [
            sum(float(coeffs[j]) * float(x_cols[j][i]) for j in range(len(coeffs))) + float(alpha_corr[i])
            for i in range(len(temps))
        ]

        a_low = float("nan")
        a_high = float("nan")
        a_ta = float("nan")
        a_la = float("nan")
        a_opt = float("nan")
        a_to = float("nan")
        a_lo = float("nan")

        out_tag = f"condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_gamma_omega_{gamma_omega_model}_model"
        model_name = f"DOS-constrained Gruneisen gamma(omega) basis ({gamma_omega_model}; weighted LS)"
    elif int(args.groups) == 2:
        x_low = _basis_list(cv_low)
        x_high: list[float]
        if softening_enabled:
            if f_grid is None or cv_hi_candidates is None:
                raise SystemExit("[fail] softening candidates missing")
            best_fit: dict[str, float] | None = None
            best_i = 0
            scan_rows: list[dict[str, float]] = []
            for i, cv_hi_row in enumerate(cv_hi_candidates):
                x_hi_row = _basis_np(cv_hi_row)
                fit_i = _fit_two_basis_weighted_ls(
                    x1=x_low,
                    x2=x_hi_row,
                    y=alpha_obs,
                    sigma=sigma_fit,
                    idx=fit_idx,
                    enforce_signs=bool(args.enforce_signs),
                    ridge_factor=float(ridge_factor),
                )
                sse = float(fit_i.get("sse", float("nan")))
                scan_rows.append({"frac_at_Tmax": float(f_grid[i]), "sse": float(sse)})
                if best_fit is None or (math.isfinite(sse) and sse < float(best_fit.get("sse", float("inf")))):
                    best_fit = fit_i
                    best_i = int(i)
            if best_fit is None:
                raise SystemExit("[fail] softening scan failed to produce a fit (2-group)")
            cv_high = cv_hi_candidates[best_i].tolist()
            x_high = _basis_np(cv_hi_candidates[best_i]).tolist()
            softening_global_frac = float(f_grid[best_i])
            softening_best = {"frac_at_Tmax": float(softening_global_frac), "sse": float(best_fit.get("sse", float("nan")))}
            softening_scan = scan_rows
            fit = best_fit
        else:
            x_high = _basis_list(cv_high)
            if gamma_trend == "constant":
                fit = _fit_two_basis_weighted_ls(
                    x1=x_low,
                    x2=x_high,
                    y=alpha_obs,
                    sigma=sigma_fit,
                    idx=fit_idx,
                    enforce_signs=bool(args.enforce_signs),
                    ridge_factor=float(ridge_factor),
                )
            else:
                if gamma_trend_weights is None:
                    raise SystemExit("[fail] gamma_trend_weights missing")
                fit = _fit_basis_plus_delta_weighted_ls(
                    x_cols=[x_low, x_high],
                    g_cols=[gamma_trend_weights["acoustic"], gamma_trend_weights["optical"]],
                    y=alpha_obs,
                    sigma=sigma_fit,
                    idx=fit_idx,
                    enforce_signs=bool(args.enforce_signs),
                    sign_constraints=[-1, 1],
                    ridge_factor=float(ridge_factor),
                    delta_ridge_factor=float(delta_ridge_factor),
                )
        if gamma_trend == "constant":
            a_low = float(fit["A_low_mol_per_J"])
            a_high = float(fit["A_high_mol_per_J"])
            delta_gamma = 0.0
        else:
            if gamma_trend_weights is None:
                raise SystemExit("[fail] gamma_trend_weights missing")
            a_low = float(fit["a"][0])
            a_high = float(fit["a"][1])
            delta_gamma = float(fit["delta_gamma"])
        a_ta = float("nan")
        a_la = float("nan")
        a_opt = float("nan")
        a_to = float("nan")
        a_lo = float("nan")
        if gamma_trend == "constant":
            alpha_pred = [a_low * float(c1) + a_high * float(c2) for c1, c2 in zip(x_low, x_high)]
        else:
            g_low = gamma_trend_weights["acoustic"]
            g_high = gamma_trend_weights["optical"]
            alpha_pred = [
                (a_low + delta_gamma * float(g_low[i])) * float(x_low[i])
                + (a_high + delta_gamma * float(g_high[i])) * float(x_high[i])
                for i in range(len(temps))
            ]
        if softening_enabled:
            if softening_mode == "linear_fit":
                out_tag = "condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_optical_softening_linear_model"
                model_name = "DOS-split Gruneisen 2-group + optical softening (linear; f fit by grid scan)"
            else:
                out_tag = "condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_optical_softening_raman_shape_model"
                model_name = "DOS-split Gruneisen 2-group + optical softening (Raman shape; f fit by grid scan)"
        else:
            out_tag = "condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_model"
            model_name = "DOS-split Gruneisen 2-group (A_low/A_high weighted LS; split at half-integral)"

        if str(args.dos_softening) == "kim2015_linear_proxy" and out_tag.endswith("_model"):
            out_tag = out_tag[: -len("_model")] + "_dos_softening_kim2015_linear_model"
    elif int(args.groups) == 3:
        x_ta = _basis_list(cv_ta)
        x_la = _basis_list(cv_la)
        x_opt: list[float]
        if softening_enabled:
            if f_grid is None or cv_hi_candidates is None:
                raise SystemExit("[fail] softening candidates missing")
            best_fit3: dict[str, float] | None = None
            best_i = 0
            scan_rows: list[dict[str, float]] = []
            for i, cv_opt_row in enumerate(cv_hi_candidates):
                x_opt_row = _basis_np(cv_opt_row)
                fit_i = _fit_three_basis_weighted_ls(
                    x1=x_ta,
                    x2=x_la,
                    x3=x_opt_row,
                    y=alpha_obs,
                    sigma=sigma_fit,
                    idx=fit_idx,
                    enforce_signs=bool(args.enforce_signs),
                    ridge_factor=float(ridge_factor),
                )
                sse = float(fit_i.get("sse", float("nan")))
                scan_rows.append({"frac_at_Tmax": float(f_grid[i]), "sse": float(sse)})
                if best_fit3 is None or (math.isfinite(sse) and sse < float(best_fit3.get("sse", float("inf")))):
                    best_fit3 = fit_i
                    best_i = int(i)
            if best_fit3 is None:
                raise SystemExit("[fail] softening scan failed to produce a fit (3-group)")
            cv_opt = cv_hi_candidates[best_i].tolist()
            x_opt = _basis_np(cv_hi_candidates[best_i]).tolist()
            softening_global_frac = float(f_grid[best_i])
            softening_best = {"frac_at_Tmax": float(softening_global_frac), "sse": float(best_fit3.get("sse", float("nan")))}
            softening_scan = scan_rows
            fit3 = best_fit3
        else:
            x_opt = _basis_list(cv_opt)
            if gamma_trend == "constant":
                fit3 = _fit_three_basis_weighted_ls(
                    x1=x_ta,
                    x2=x_la,
                    x3=x_opt,
                    y=alpha_obs,
                    sigma=sigma_fit,
                    idx=fit_idx,
                    enforce_signs=bool(args.enforce_signs),
                    ridge_factor=float(ridge_factor),
                )
            else:
                if gamma_trend_weights is None:
                    raise SystemExit("[fail] gamma_trend_weights missing")
                fit3 = _fit_basis_plus_delta_weighted_ls(
                    x_cols=[x_ta, x_la, x_opt],
                    g_cols=[gamma_trend_weights["ta"], gamma_trend_weights["la"], gamma_trend_weights["optical"]],
                    y=alpha_obs,
                    sigma=sigma_fit,
                    idx=fit_idx,
                    enforce_signs=bool(args.enforce_signs),
                    sign_constraints=[-1, 1, 1],
                    ridge_factor=float(ridge_factor),
                    delta_ridge_factor=float(delta_ridge_factor),
                )
        if gamma_trend == "constant":
            a_ta = float(fit3["A_1_mol_per_J"])
            a_la = float(fit3["A_2_mol_per_J"])
            a_opt = float(fit3["A_3_mol_per_J"])
            delta_gamma = 0.0
        else:
            if gamma_trend_weights is None:
                raise SystemExit("[fail] gamma_trend_weights missing")
            a_ta = float(fit3["a"][0])
            a_la = float(fit3["a"][1])
            a_opt = float(fit3["a"][2])
            delta_gamma = float(fit3["delta_gamma"])
        a_low = float("nan")
        a_high = float("nan")
        a_to = float("nan")
        a_lo = float("nan")
        if gamma_trend == "constant":
            alpha_pred = [
                a_ta * float(c1) + a_la * float(c2) + a_opt * float(c3) for c1, c2, c3 in zip(x_ta, x_la, x_opt)
            ]
        else:
            g_ta = gamma_trend_weights["ta"]
            g_la = gamma_trend_weights["la"]
            g_opt = gamma_trend_weights["optical"]
            alpha_pred = [
                (a_ta + delta_gamma * float(g_ta[i])) * float(x_ta[i])
                + (a_la + delta_gamma * float(g_la[i])) * float(x_la[i])
                + (a_opt + delta_gamma * float(g_opt[i])) * float(x_opt[i])
                for i in range(len(temps))
            ]
        if softening_enabled:
            if softening_mode == "linear_fit":
                out_tag = "condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_optical_softening_linear_model"
                model_name = "DOS-split Gruneisen 3-group + optical softening (linear; f fit by grid scan)"
            else:
                out_tag = "condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_optical_softening_raman_shape_model"
                model_name = "DOS-split Gruneisen 3-group + optical softening (Raman shape; f fit by grid scan)"
        else:
            out_tag = "condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_model"
            model_name = "DOS-split Gruneisen 3-group (TA/LA/optical; mode-count split; weighted LS)"

        if str(args.dos_softening) == "kim2015_linear_proxy" and out_tag.endswith("_model"):
            out_tag = out_tag[: -len("_model")] + "_dos_softening_kim2015_linear_model"
    else:
        x_ta = _basis_list(cv_ta)
        x_la = _basis_list(cv_la)
        x_to: list[float]
        x_lo: list[float]
        if softening_enabled:
            if f_grid is None or cv_to_candidates is None or cv_lo_candidates is None:
                raise SystemExit("[fail] softening candidates missing")
            best_fit4: dict[str, float] | None = None
            best_i = 0
            scan_rows: list[dict[str, float]] = []
            for i, (cv_to_row, cv_lo_row) in enumerate(zip(cv_to_candidates, cv_lo_candidates)):
                x_to_row = _basis_np(cv_to_row)
                x_lo_row = _basis_np(cv_lo_row)
                fit_i = _fit_four_basis_weighted_ls(
                    x1=x_ta,
                    x2=x_la,
                    x3=x_to_row,
                    x4=x_lo_row,
                    y=alpha_obs,
                    sigma=sigma_fit,
                    idx=fit_idx,
                    enforce_signs=bool(args.enforce_signs),
                    ridge_factor=float(ridge_factor),
                )
                sse = float(fit_i.get("sse", float("nan")))
                scan_rows.append({"frac_at_Tmax": float(f_grid[i]), "sse": float(sse)})
                if best_fit4 is None or (math.isfinite(sse) and sse < float(best_fit4.get("sse", float("inf")))):
                    best_fit4 = fit_i
                    best_i = int(i)
            if best_fit4 is None:
                raise SystemExit("[fail] softening scan failed to produce a fit (4-group)")
            cv_to = cv_to_candidates[best_i].tolist()
            cv_lo = cv_lo_candidates[best_i].tolist()
            x_to = _basis_np(cv_to_candidates[best_i]).tolist()
            x_lo = _basis_np(cv_lo_candidates[best_i]).tolist()
            softening_global_frac = float(f_grid[best_i])
            softening_best = {"frac_at_Tmax": float(softening_global_frac), "sse": float(best_fit4.get("sse", float("nan")))}
            softening_scan = scan_rows
            fit4 = best_fit4
        else:
            x_to = _basis_list(cv_to)
            x_lo = _basis_list(cv_lo)
            if gamma_trend == "constant":
                fit4 = _fit_four_basis_weighted_ls(
                    x1=x_ta,
                    x2=x_la,
                    x3=x_to,
                    x4=x_lo,
                    y=alpha_obs,
                    sigma=sigma_fit,
                    idx=fit_idx,
                    enforce_signs=bool(args.enforce_signs),
                    ridge_factor=float(ridge_factor),
                )
            else:
                if gamma_trend_weights is None:
                    raise SystemExit("[fail] gamma_trend_weights missing")
                fit4 = _fit_basis_plus_delta_weighted_ls(
                    x_cols=[x_ta, x_la, x_to, x_lo],
                    g_cols=[
                        gamma_trend_weights["ta"],
                        gamma_trend_weights["la"],
                        gamma_trend_weights["to"],
                        gamma_trend_weights["lo"],
                    ],
                    y=alpha_obs,
                    sigma=sigma_fit,
                    idx=fit_idx,
                    enforce_signs=bool(args.enforce_signs),
                    sign_constraints=[-1, 1, 1, 1],
                    ridge_factor=float(ridge_factor),
                    delta_ridge_factor=float(delta_ridge_factor),
                )
        if gamma_trend == "constant":
            a_ta = float(fit4["A_1_mol_per_J"])
            a_la = float(fit4["A_2_mol_per_J"])
            a_to = float(fit4["A_3_mol_per_J"])
            a_lo = float(fit4["A_4_mol_per_J"])
            delta_gamma = 0.0
        else:
            if gamma_trend_weights is None:
                raise SystemExit("[fail] gamma_trend_weights missing")
            a_ta = float(fit4["a"][0])
            a_la = float(fit4["a"][1])
            a_to = float(fit4["a"][2])
            a_lo = float(fit4["a"][3])
            delta_gamma = float(fit4["delta_gamma"])
        a_low = float("nan")
        a_high = float("nan")
        a_opt = float("nan")
        if gamma_trend == "constant":
            alpha_pred = [
                a_ta * float(c1) + a_la * float(c2) + a_to * float(c3) + a_lo * float(c4)
                for c1, c2, c3, c4 in zip(x_ta, x_la, x_to, x_lo)
            ]
        else:
            g_ta = gamma_trend_weights["ta"]
            g_la = gamma_trend_weights["la"]
            g_to = gamma_trend_weights["to"]
            g_lo = gamma_trend_weights["lo"]
            alpha_pred = [
                (a_ta + delta_gamma * float(g_ta[i])) * float(x_ta[i])
                + (a_la + delta_gamma * float(g_la[i])) * float(x_la[i])
                + (a_to + delta_gamma * float(g_to[i])) * float(x_to[i])
                + (a_lo + delta_gamma * float(g_lo[i])) * float(x_lo[i])
                for i in range(len(temps))
            ]
        if softening_enabled:
            if softening_mode == "linear_fit":
                out_tag = "condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_four_group_optical_softening_linear_model"
                model_name = "DOS-split Gruneisen 4-group + optical softening (linear; f fit by grid scan)"
            else:
                out_tag = "condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_four_group_optical_softening_raman_shape_model"
                model_name = "DOS-split Gruneisen 4-group + optical softening (Raman shape; f fit by grid scan)"
        else:
            out_tag = "condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_four_group_model"
            model_name = "DOS-split Gruneisen 4-group (TA/LA/TO/LO; mode-count split; weighted LS)"

        if str(args.dos_softening) == "kim2015_linear_proxy" and out_tag.endswith("_model"):
            out_tag = out_tag[: -len("_model")] + "_dos_softening_kim2015_linear_model"

    if mode_softening_mode in ("kim2015_fig2_features", "kim2015_fig2_features_eq8_quasi"):
        suffix = "mode_softening_kim2015_fig2"
        if mode_softening_mode == "kim2015_fig2_features_eq8_quasi":
            suffix = suffix + "_eq8_quasi"
        if out_tag.endswith("_model"):
            out_tag = out_tag[: -len("_model")] + f"_{suffix}_model"
        else:
            out_tag = out_tag + f"_{suffix}"
        name_extra = "Kim2015 Fig.2 digitized"
        if mode_softening_mode == "kim2015_fig2_features_eq8_quasi":
            name_extra = name_extra + "; Eq8 quasiharmonic-only"
        model_name = f"{model_name} + mode softening ({name_extra})"

    if gamma_trend != "constant":
        if out_tag.endswith("_model"):
            suffix = "gamma_trend_kim2015_fig2" if gamma_trend == "kim2015_fig2_softening_common" else f"gamma_trend_{str(gamma_trend)}"
            out_tag = out_tag[: -len("_model")] + f"_{suffix}_model"
        else:
            suffix = "gamma_trend_kim2015_fig2" if gamma_trend == "kim2015_fig2_softening_common" else f"gamma_trend_{str(gamma_trend)}"
            out_tag = out_tag + f"_{suffix}"
        model_name = (
            f"{model_name} + gamma(T) trend (gamma_i(T)=gamma_i0 + Delta_gamma*(1-omega_scale_i(T)))"
        )

    if dos_mode == "kim2015_fig1_energy":
        if out_tag.endswith("_model"):
            out_tag = out_tag[: -len("_model")] + "_kim2015_fig1_energy_model"
        else:
            out_tag = out_tag + "_kim2015_fig1_energy"
        model_name = f"{model_name} + T-dependent DOS (Kim2015 Fig.1 digitized)"

    if use_bulk_modulus:
        if out_tag.endswith("_model"):
            out_tag = out_tag[: -len("_model")] + "_bulkmodulus_model"
        else:
            out_tag = out_tag + "_bulkmodulus"
        model_name = f"{model_name} + B(T) scaling (fit γ)"

    if use_vm_thermal_expansion:
        if out_tag.endswith("_model"):
            out_tag = out_tag[: -len("_model")] + "_vmT_model"
        else:
            out_tag = out_tag + "_vmT"
        model_name = f"{model_name} + V_m(T) from alpha(T) integral"

    if cv_omega_dependence != "harmonic":
        if out_tag.endswith("_model"):
            out_tag = out_tag[: -len("_model")] + "_cv_dU_model"
        else:
            out_tag = out_tag + "_cv_dU"
        model_name = f"{model_name} + Cv via numeric dU/dT (omega(T) included)"

    def _fmt_tag_sci(v: float) -> str:
        s = f"{float(v):.0e}"
        s = s.replace("+", "")
        return s

    def _fmt_tag_sci_precise(v: float) -> str:
        s = f"{float(v):.2e}"
        s = s.replace("+", "")
        s = s.replace(".", "p")
        return s

    if float(ridge_factor) > 0.0:
        suffix = f"ridge{_fmt_tag_sci(float(ridge_factor))}"
        if out_tag.endswith("_model"):
            out_tag = out_tag[: -len("_model")] + f"_{suffix}_model"
        else:
            out_tag = out_tag + f"_{suffix}"
        model_name = f"{model_name} + ridge (factor={float(ridge_factor):.2g})"

    if float(delta_ridge_factor) > 0.0:
        suffix = f"dridge{_fmt_tag_sci(float(delta_ridge_factor))}"
        if out_tag.endswith("_model"):
            out_tag = out_tag[: -len("_model")] + f"_{suffix}_model"
        else:
            out_tag = out_tag + f"_{suffix}"
        model_name = f"{model_name} + delta ridge (factor={float(delta_ridge_factor):.2g})"

    if gamma_omega_model == "pwlinear_split_leaky":
        suffix = f"leak{_fmt_tag_sci_precise(float(gamma_omega_pwlinear_leak))}"
        if out_tag.endswith("_model"):
            out_tag = out_tag[: -len("_model")] + f"_{suffix}_model"
        else:
            out_tag = out_tag + f"_{suffix}"
        model_name = (
            f"{model_name} + pwlinear leak (epsilon={float(gamma_omega_pwlinear_leak):.2g}, "
            f"warp_p={float(gamma_omega_pwlinear_warp_power):.3g})"
        )
        if abs(float(gamma_omega_pwlinear_warp_power) - 1.0) > 1e-12:
            suffix = f"warp{float(gamma_omega_pwlinear_warp_power):.3g}".replace(".", "p")
            if out_tag.endswith("_model"):
                out_tag = out_tag[: -len("_model")] + f"_{suffix}_model"
            else:
                out_tag = out_tag + f"_{suffix}"

    if gamma_omega_softening_fit != "none":
        if gamma_omega_softening_fit == "common_centered300":
            suffix = "gsoftfitc300"
        elif gamma_omega_softening_fit == "high_centered300":
            suffix = "gsofthighfitc300"
        elif gamma_omega_softening_fit == "by_label_centered300":
            suffix = "gsoftfitlabelc300"
        else:
            raise SystemExit(f"[fail] unsupported --gamma-omega-softening-fit: {gamma_omega_softening_fit!r}")
        if out_tag.endswith("_model"):
            out_tag = out_tag[: -len("_model")] + f"_{suffix}_model"
        else:
            out_tag = out_tag + f"_{suffix}"
        model_name = (
            f"{model_name} + gamma softening delta fit "
            f"(delta_fit={float(gamma_omega_softening_fit_delta):.2g}; mode={gamma_omega_softening_fit}; g centered at 300K)"
        )

    if float(gamma_omega_softening_delta) != 0.0:
        suffix = f"gsoftc300{_fmt_tag_sci_precise(float(gamma_omega_softening_delta))}"
        if out_tag.endswith("_model"):
            out_tag = out_tag[: -len("_model")] + f"_{suffix}_model"
        else:
            out_tag = out_tag + f"_{suffix}"
        model_name = (
            f"{model_name} + gamma softening delta "
            f"(delta={float(gamma_omega_softening_delta):.2g}; common; g centered at 300K)"
        )

    if float(gamma_omega_high_softening_delta) != 0.0:
        suffix = f"gsofthighc300{_fmt_tag_sci_precise(float(gamma_omega_high_softening_delta))}"
        if out_tag.endswith("_model"):
            out_tag = out_tag[: -len("_model")] + f"_{suffix}_model"
        else:
            out_tag = out_tag + f"_{suffix}"
        model_name = (
            f"{model_name} + gamma_high softening delta "
            f"(delta={float(gamma_omega_high_softening_delta):.2g}; g centered at 300K)"
        )

    alpha_pred_1e8 = [float(a) / 1e-8 for a in alpha_pred]

    # Diagnostics.
    residual = [float(ap - ao) for ap, ao in zip(alpha_pred, alpha_obs)]
    residual_1e8 = [float(r) / 1e-8 for r in residual]
    z = [float("nan") if float(sigma_fit[i]) <= 0.0 else float(residual[i] / float(sigma_fit[i])) for i in range(len(temps))]
    zero = _infer_zero_crossing(temps, alpha_obs, prefer_neg_to_pos=True, min_x=50.0)

    # Holdout splits (same definitions as 7.14.15).
    splits = [
        {"name": "A", "train": [50.0, 300.0], "test": [300.0, 600.0]},
        {"name": "B", "train": [200.0, 600.0], "test": [50.0, 200.0]},
    ]
    split_results: list[dict[str, object]] = []
    for s in splits:
        train_min, train_max = float(s["train"][0]), float(s["train"][1])
        test_min, test_max = float(s["test"][0]), float(s["test"][1])
        train_idx = [i for i, t in enumerate(temps) if train_min <= float(t) <= train_max]
        test_idx = [i for i, t in enumerate(temps) if test_min <= float(t) <= test_max]
        if len(train_idx) < 20 or len(test_idx) < 20:
            raise SystemExit(f"[fail] split {s['name']} has too few points: train={len(train_idx)}, test={len(test_idx)}")

        if gamma_omega_model != "none":
            x_cols = [_basis_list(cv_gamma_omega[lbl]) for lbl in gamma_omega_basis_labels]
            delta_fit_s = 0.0
            if gamma_omega_softening_fit != "none":
                if mode_scales is None:
                    raise SystemExit("[fail] internal: gamma_omega_softening_fit set but mode_scales is missing")
                scales_ac: list[float]
                scales_opt: list[float]
                if isinstance(mode_scales.get("optical"), list):
                    scales_opt = [float(v) for v in mode_scales["optical"]]
                elif isinstance(mode_scales.get("to"), list) and isinstance(mode_scales.get("lo"), list):
                    to = [float(v) for v in mode_scales["to"]]
                    lo = [float(v) for v in mode_scales["lo"]]
                    if len(to) != len(lo):
                        raise SystemExit("[fail] invalid mode scales for optical composite (to/lo length mismatch)")
                    scales_opt = [float((2.0 * float(sto) + float(slo)) / 3.0) for sto, slo in zip(to, lo)]
                else:
                    raise SystemExit(
                        "[fail] --gamma-omega-softening-fit requires optical scale (groups=2/3) or to/lo scales (groups=4)"
                    )

                if isinstance(mode_scales.get("acoustic"), list):
                    scales_ac = [float(v) for v in mode_scales["acoustic"]]
                elif isinstance(mode_scales.get("ta"), list) and isinstance(mode_scales.get("la"), list):
                    ta = [float(v) for v in mode_scales["ta"]]
                    la = [float(v) for v in mode_scales["la"]]
                    if len(ta) != len(la):
                        raise SystemExit("[fail] invalid mode scales for acoustic composite (ta/la length mismatch)")
                    scales_ac = [float((2.0 * float(sta) + float(sla)) / 3.0) for sta, sla in zip(ta, la)]
                else:
                    raise SystemExit(
                        "[fail] --gamma-omega-softening-fit requires acoustic scale (groups=2) or ta/la scales (groups=3/4)"
                    )
                if len(scales_opt) != len(temps):
                    raise SystemExit("[fail] invalid optical scale length for --gamma-omega-softening-fit")
                if len(scales_ac) != len(temps):
                    raise SystemExit("[fail] invalid acoustic scale length for --gamma-omega-softening-fit")
                g_raw_opt = [float(1.0 - float(s)) for s in scales_opt]
                g_raw_ac = [float(1.0 - float(s)) for s in scales_ac]
                try:
                    idx_300 = temps.index(300.0)
                except ValueError:
                    raise SystemExit("[fail] internal: alpha(T) grid does not include 300 K (expected integer grid)")
                g0_opt = float(g_raw_opt[idx_300])
                g0_ac = float(g_raw_ac[idx_300])
                g_opt_centered = [float(g_raw_opt[i] - g0_opt) for i in range(len(g_raw_opt))]
                g_ac_centered = [float(g_raw_ac[i] - g0_ac) for i in range(len(g_raw_ac))]

                if gamma_omega_softening_fit == "common_centered300":
                    g_cols = [list(g_opt_centered) for _ in x_cols]
                elif gamma_omega_softening_fit == "high_centered300":
                    if "high" not in gamma_omega_basis_labels:
                        raise SystemExit("[fail] --gamma-omega-softening-fit=high_centered300 requires a 'high' basis label")
                    high_idx = int(gamma_omega_basis_labels.index("high"))
                    g_cols = [[0.0 for _ in temps] for _ in x_cols]
                    g_cols[high_idx] = list(g_opt_centered)
                elif gamma_omega_softening_fit == "by_label_centered300":
                    g_mid_centered = [0.5 * (float(g_ac_centered[i]) + float(g_opt_centered[i])) for i in range(len(temps))]
                    g_cols = []
                    for lbl in gamma_omega_basis_labels:
                        if str(lbl) == "low":
                            g_cols.append(list(g_ac_centered))
                        elif str(lbl) == "high":
                            g_cols.append(list(g_opt_centered))
                        elif str(lbl) == "mid":
                            g_cols.append(list(g_mid_centered))
                        else:
                            g_cols.append(list(g_opt_centered))
                else:
                    raise SystemExit(f"[fail] unsupported --gamma-omega-softening-fit: {gamma_omega_softening_fit!r}")

                sign_constraints = ([-1] + [1] * (len(x_cols) - 1)) if bool(args.enforce_signs) else [0] * len(x_cols)
                fit_s = _fit_basis_plus_delta_weighted_ls(
                    x_cols=[list(c) for c in x_cols],
                    g_cols=[list(g) for g in g_cols],
                    y=alpha_obs,
                    sigma=sigma_fit,
                    idx=train_idx,
                    enforce_signs=bool(args.enforce_signs),
                    sign_constraints=sign_constraints,
                    ridge_factor=float(ridge_factor),
                    delta_ridge_factor=float(delta_ridge_factor),
                )
                coeffs_s = [float(v) for v in fit_s.get("a", [])]
                delta_fit_s = float(fit_s.get("delta_gamma", 0.0))
                if len(coeffs_s) != len(x_cols):
                    raise SystemExit("[fail] internal: invalid holdout fit size for gamma_omega_softening_fit")
                x_delta_s = [sum(float(g_cols[j][i]) * float(x_cols[j][i]) for j in range(len(x_cols))) for i in range(len(temps))]
                pred_s = [
                    sum(float(coeffs_s[j]) * float(x_cols[j][i]) for j in range(len(coeffs_s)))
                    + float(delta_fit_s) * float(x_delta_s[i])
                    for i in range(len(temps))
                ]
            else:
                if len(x_cols) == 2:
                    fit_s = _fit_two_basis_weighted_ls(
                        x1=x_cols[0],
                        x2=x_cols[1],
                        y=y_fit,
                        sigma=sigma_fit,
                        idx=train_idx,
                        enforce_signs=bool(args.enforce_signs),
                        ridge_factor=float(ridge_factor),
                    )
                    coeffs_s = [float(fit_s["A_low_mol_per_J"]), float(fit_s["A_high_mol_per_J"])]
                else:
                    fit_s = _fit_three_basis_weighted_ls(
                        x1=x_cols[0],
                        x2=x_cols[1],
                        x3=x_cols[2],
                        y=y_fit,
                        sigma=sigma_fit,
                        idx=train_idx,
                        enforce_signs=bool(args.enforce_signs),
                        ridge_factor=float(ridge_factor),
                    )
                    coeffs_s = [
                        float(fit_s["A_1_mol_per_J"]),
                        float(fit_s["A_2_mol_per_J"]),
                        float(fit_s["A_3_mol_per_J"]),
                    ]

                pred_s = [
                    sum(float(coeffs_s[j]) * float(x_cols[j][i]) for j in range(len(coeffs_s))) + float(alpha_corr[i])
                    for i in range(len(temps))
                ]
            m_train = _metrics_for_range(
                idx=train_idx,
                alpha_obs=alpha_obs,
                alpha_pred=pred_s,
                sigma_fit=sigma_fit,
                param_count=(len(coeffs_s) + (1 if gamma_omega_softening_fit != "none" else 0)),
                is_train=True,
            )
            m_test = _metrics_for_range(
                idx=test_idx,
                alpha_obs=alpha_obs,
                alpha_pred=pred_s,
                sigma_fit=sigma_fit,
                param_count=(len(coeffs_s) + (1 if gamma_omega_softening_fit != "none" else 0)),
                is_train=False,
            )
            split_results.append(
                {
                    "name": str(s["name"]),
                    "train_T_K": [train_min, train_max],
                    "test_T_K": [test_min, test_max],
                    "params": {
                        "gamma_omega_model": str(gamma_omega_model),
                        "coefficients": {str(lbl): float(coeffs_s[i]) for i, lbl in enumerate(gamma_omega_basis_labels)},
                        "pwlinear_leak_epsilon": (
                            float(gamma_omega_pwlinear_leak) if gamma_omega_model == "pwlinear_split_leaky" else ""
                        ),
                        "pwlinear_leak_warp_power": (
                            float(gamma_omega_pwlinear_warp_power) if gamma_omega_model == "pwlinear_split_leaky" else ""
                        ),
                        "gamma_omega_softening_fit": (str(gamma_omega_softening_fit) if gamma_omega_softening_fit != "none" else ""),
                        "gamma_omega_softening_fit_delta": (float(delta_fit_s) if gamma_omega_softening_fit != "none" else ""),
                        "gamma_omega_softening_delta": (
                            float(gamma_omega_softening_delta) if float(gamma_omega_softening_delta) != 0.0 else ""
                        ),
                        "gamma_omega_high_softening_delta": (
                            float(gamma_omega_high_softening_delta) if float(gamma_omega_high_softening_delta) != 0.0 else ""
                        ),
                        "enforce_signs": bool(args.enforce_signs),
                        "ridge_factor": float(ridge_factor),
                        "ridge_lambda": (
                            float(fit_s.get("ridge_lambda", float("nan")))
                            if isinstance(fit_s.get("ridge_lambda"), (int, float))
                            else ""
                        ),
                        "coeff_unit": ("gamma_dimensionless" if use_bulk_modulus else "A_mol_per_J"),
                        "fit_sse": float(fit_s.get("sse", float("nan"))),
                    },
                    "train": m_train,
                    "test": m_test,
                }
            )
        elif int(args.groups) == 2:
            x_low = _basis_list(cv_low)
            x_hi_use: list[float]
            if softening_enabled:
                if f_grid is None or cv_hi_candidates is None:
                    raise SystemExit("[fail] softening candidates missing")
                best_fit_s: dict[str, float] | None = None
                best_i = 0
                for i, cv_hi_row in enumerate(cv_hi_candidates):
                    x_hi_row = _basis_np(cv_hi_row)
                    fit_i = _fit_two_basis_weighted_ls(
                        x1=x_low,
                        x2=x_hi_row,
                        y=alpha_obs,
                        sigma=sigma_fit,
                        idx=train_idx,
                        enforce_signs=bool(args.enforce_signs),
                        ridge_factor=float(ridge_factor),
                    )
                    sse = float(fit_i.get("sse", float("nan")))
                    if best_fit_s is None or (math.isfinite(sse) and sse < float(best_fit_s.get("sse", float("inf")))):
                        best_fit_s = fit_i
                        best_i = int(i)
                if best_fit_s is None:
                    raise SystemExit("[fail] softening scan failed in holdout (2-group)")
                fit_s = best_fit_s
                cv_hi_use = cv_hi_candidates[best_i].tolist()
                x_hi_use = _basis_np(cv_hi_candidates[best_i]).tolist()
                f_use = float(f_grid[best_i])
            else:
                x_hi_use = _basis_list(cv_high)
                if gamma_trend == "constant":
                    fit_s = _fit_two_basis_weighted_ls(
                        x1=x_low,
                        x2=x_hi_use,
                        y=alpha_obs,
                        sigma=sigma_fit,
                        idx=train_idx,
                        enforce_signs=bool(args.enforce_signs),
                        ridge_factor=float(ridge_factor),
                    )
                else:
                    if gamma_trend_weights is None:
                        raise SystemExit("[fail] gamma_trend_weights missing")
                    fit_s = _fit_basis_plus_delta_weighted_ls(
                        x_cols=[x_low, x_hi_use],
                        g_cols=[gamma_trend_weights["acoustic"], gamma_trend_weights["optical"]],
                        y=alpha_obs,
                        sigma=sigma_fit,
                        idx=train_idx,
                        enforce_signs=bool(args.enforce_signs),
                        sign_constraints=[-1, 1],
                        ridge_factor=float(ridge_factor),
                        delta_ridge_factor=float(delta_ridge_factor),
                    )
                cv_hi_use = cv_high
                f_use = float("nan")

            if gamma_trend == "constant":
                a1 = float(fit_s["A_low_mol_per_J"])
                a2 = float(fit_s["A_high_mol_per_J"])
                delta_s = 0.0
                pred_s = [a1 * float(c1) + a2 * float(c2) for c1, c2 in zip(x_low, x_hi_use)]
            else:
                if gamma_trend_weights is None:
                    raise SystemExit("[fail] gamma_trend_weights missing")
                a1 = float(fit_s["a"][0])
                a2 = float(fit_s["a"][1])
                delta_s = float(fit_s["delta_gamma"])
                g_low = gamma_trend_weights["acoustic"]
                g_high = gamma_trend_weights["optical"]
                pred_s = [
                    (a1 + delta_s * float(g_low[i])) * float(x_low[i]) + (a2 + delta_s * float(g_high[i])) * float(x_hi_use[i])
                    for i in range(len(temps))
                ]
            m_train = _metrics_for_range(
                idx=train_idx,
                alpha_obs=alpha_obs,
                alpha_pred=pred_s,
                sigma_fit=sigma_fit,
                param_count=((3 if softening_enabled else 2) + (1 if gamma_trend != "constant" else 0)),
                is_train=True,
            )
            m_test = _metrics_for_range(
                idx=test_idx,
                alpha_obs=alpha_obs,
                alpha_pred=pred_s,
                sigma_fit=sigma_fit,
                param_count=((3 if softening_enabled else 2) + (1 if gamma_trend != "constant" else 0)),
                is_train=False,
            )
            split_results.append(
                {
                    "name": str(s["name"]),
                    "train_T_K": [train_min, train_max],
                    "test_T_K": [test_min, test_max],
                    "params": {
                        "A_low_mol_per_J": float(a1),
                        "A_high_mol_per_J": float(a2),
                        "delta_gamma_dimensionless": float(delta_s) if gamma_trend != "constant" else "",
                        "softening_frac_at_Tmax": float(f_use) if softening_enabled else "",
                        "enforce_signs": bool(args.enforce_signs),
                        "ridge_factor": float(ridge_factor),
                        "ridge_lambda": (
                            float(fit_s.get("ridge_lambda", float("nan"))) if isinstance(fit_s.get("ridge_lambda"), (int, float)) else ""
                        ),
                        "coeff_unit": ("gamma_dimensionless" if use_bulk_modulus else "A_mol_per_J"),
                        "fit_sse": float(fit_s.get("sse", float("nan"))),
                        "constraint_solution_kind": (
                            fit_s.get("constraint_solution_kind_str") if isinstance(fit_s.get("constraint_solution_kind_str"), str) else ""
                        ),
                    },
                    "train": m_train,
                    "test": m_test,
                }
            )
        elif int(args.groups) == 3:
            x_ta = _basis_list(cv_ta)
            x_la = _basis_list(cv_la)
            x_opt_use: list[float]
            if softening_enabled:
                if f_grid is None or cv_hi_candidates is None:
                    raise SystemExit("[fail] softening candidates missing")
                best_fit_s: dict[str, float] | None = None
                best_i = 0
                for i, cv_opt_row in enumerate(cv_hi_candidates):
                    x_opt_row = _basis_np(cv_opt_row)
                    fit_i = _fit_three_basis_weighted_ls(
                        x1=x_ta,
                        x2=x_la,
                        x3=x_opt_row,
                        y=alpha_obs,
                        sigma=sigma_fit,
                        idx=train_idx,
                        enforce_signs=bool(args.enforce_signs),
                        ridge_factor=float(ridge_factor),
                    )
                    sse = float(fit_i.get("sse", float("nan")))
                    if best_fit_s is None or (math.isfinite(sse) and sse < float(best_fit_s.get("sse", float("inf")))):
                        best_fit_s = fit_i
                        best_i = int(i)
                if best_fit_s is None:
                    raise SystemExit("[fail] softening scan failed in holdout (3-group)")
                fit_s = best_fit_s
                cv_opt_use = cv_hi_candidates[best_i].tolist()
                x_opt_use = _basis_np(cv_hi_candidates[best_i]).tolist()
                f_use = float(f_grid[best_i])
            else:
                x_opt_use = _basis_list(cv_opt)
                if gamma_trend == "constant":
                    fit_s = _fit_three_basis_weighted_ls(
                        x1=x_ta,
                        x2=x_la,
                        x3=x_opt_use,
                        y=alpha_obs,
                        sigma=sigma_fit,
                        idx=train_idx,
                        enforce_signs=bool(args.enforce_signs),
                        ridge_factor=float(ridge_factor),
                    )
                else:
                    if gamma_trend_weights is None:
                        raise SystemExit("[fail] gamma_trend_weights missing")
                    fit_s = _fit_basis_plus_delta_weighted_ls(
                        x_cols=[x_ta, x_la, x_opt_use],
                        g_cols=[gamma_trend_weights["ta"], gamma_trend_weights["la"], gamma_trend_weights["optical"]],
                        y=alpha_obs,
                        sigma=sigma_fit,
                        idx=train_idx,
                        enforce_signs=bool(args.enforce_signs),
                        sign_constraints=[-1, 1, 1],
                        ridge_factor=float(ridge_factor),
                        delta_ridge_factor=float(delta_ridge_factor),
                    )
                cv_opt_use = cv_opt
                f_use = float("nan")

            if gamma_trend == "constant":
                a1 = float(fit_s["A_1_mol_per_J"])
                a2 = float(fit_s["A_2_mol_per_J"])
                a3 = float(fit_s["A_3_mol_per_J"])
                delta_s = 0.0
                pred_s = [
                    a1 * float(c1) + a2 * float(c2) + a3 * float(c3) for c1, c2, c3 in zip(x_ta, x_la, x_opt_use)
                ]
            else:
                if gamma_trend_weights is None:
                    raise SystemExit("[fail] gamma_trend_weights missing")
                a1 = float(fit_s["a"][0])
                a2 = float(fit_s["a"][1])
                a3 = float(fit_s["a"][2])
                delta_s = float(fit_s["delta_gamma"])
                g_ta = gamma_trend_weights["ta"]
                g_la = gamma_trend_weights["la"]
                g_opt = gamma_trend_weights["optical"]
                pred_s = [
                    (a1 + delta_s * float(g_ta[i])) * float(x_ta[i])
                    + (a2 + delta_s * float(g_la[i])) * float(x_la[i])
                    + (a3 + delta_s * float(g_opt[i])) * float(x_opt_use[i])
                    for i in range(len(temps))
                ]
            m_train = _metrics_for_range(
                idx=train_idx,
                alpha_obs=alpha_obs,
                alpha_pred=pred_s,
                sigma_fit=sigma_fit,
                param_count=((4 if softening_enabled else 3) + (1 if gamma_trend != "constant" else 0)),
                is_train=True,
            )
            m_test = _metrics_for_range(
                idx=test_idx,
                alpha_obs=alpha_obs,
                alpha_pred=pred_s,
                sigma_fit=sigma_fit,
                param_count=((4 if softening_enabled else 3) + (1 if gamma_trend != "constant" else 0)),
                is_train=False,
            )
            split_results.append(
                {
                    "name": str(s["name"]),
                    "train_T_K": [train_min, train_max],
                    "test_T_K": [test_min, test_max],
                    "params": {
                        "A_TA_mol_per_J": float(a1),
                        "A_LA_mol_per_J": float(a2),
                        "A_opt_mol_per_J": float(a3),
                        "delta_gamma_dimensionless": float(delta_s) if gamma_trend != "constant" else "",
                        "softening_frac_at_Tmax": float(f_use) if softening_enabled else "",
                        "enforce_signs": bool(args.enforce_signs),
                        "ridge_factor": float(ridge_factor),
                        "ridge_lambda": (
                            float(fit_s.get("ridge_lambda", float("nan"))) if isinstance(fit_s.get("ridge_lambda"), (int, float)) else ""
                        ),
                        "coeff_unit": ("gamma_dimensionless" if use_bulk_modulus else "A_mol_per_J"),
                        "fit_sse": float(fit_s.get("sse", float("nan"))),
                    },
                    "train": m_train,
                    "test": m_test,
                }
            )
        else:
            x_ta = _basis_list(cv_ta)
            x_la = _basis_list(cv_la)
            x_to_use: list[float]
            x_lo_use: list[float]
            if softening_enabled:
                if f_grid is None or cv_to_candidates is None or cv_lo_candidates is None:
                    raise SystemExit("[fail] softening candidates missing")
                best_fit_s: dict[str, float] | None = None
                best_i = 0
                for i, (cv_to_row, cv_lo_row) in enumerate(zip(cv_to_candidates, cv_lo_candidates)):
                    x_to_row = _basis_np(cv_to_row)
                    x_lo_row = _basis_np(cv_lo_row)
                    fit_i = _fit_four_basis_weighted_ls(
                        x1=x_ta,
                        x2=x_la,
                        x3=x_to_row,
                        x4=x_lo_row,
                        y=alpha_obs,
                        sigma=sigma_fit,
                        idx=train_idx,
                        enforce_signs=bool(args.enforce_signs),
                        ridge_factor=float(ridge_factor),
                    )
                    sse = float(fit_i.get("sse", float("nan")))
                    if best_fit_s is None or (math.isfinite(sse) and sse < float(best_fit_s.get("sse", float("inf")))):
                        best_fit_s = fit_i
                        best_i = int(i)
                if best_fit_s is None:
                    raise SystemExit("[fail] softening scan failed in holdout (4-group)")
                fit_s = best_fit_s
                cv_to_use = cv_to_candidates[best_i].tolist()
                cv_lo_use = cv_lo_candidates[best_i].tolist()
                x_to_use = _basis_np(cv_to_candidates[best_i]).tolist()
                x_lo_use = _basis_np(cv_lo_candidates[best_i]).tolist()
                f_use = float(f_grid[best_i])
            else:
                x_to_use = _basis_list(cv_to)
                x_lo_use = _basis_list(cv_lo)
                if gamma_trend == "constant":
                    fit_s = _fit_four_basis_weighted_ls(
                        x1=x_ta,
                        x2=x_la,
                        x3=x_to_use,
                        x4=x_lo_use,
                        y=alpha_obs,
                        sigma=sigma_fit,
                        idx=train_idx,
                        enforce_signs=bool(args.enforce_signs),
                        ridge_factor=float(ridge_factor),
                    )
                else:
                    if gamma_trend_weights is None:
                        raise SystemExit("[fail] gamma_trend_weights missing")
                    fit_s = _fit_basis_plus_delta_weighted_ls(
                        x_cols=[x_ta, x_la, x_to_use, x_lo_use],
                        g_cols=[
                            gamma_trend_weights["ta"],
                            gamma_trend_weights["la"],
                            gamma_trend_weights["to"],
                            gamma_trend_weights["lo"],
                        ],
                        y=alpha_obs,
                        sigma=sigma_fit,
                        idx=train_idx,
                        enforce_signs=bool(args.enforce_signs),
                        sign_constraints=[-1, 1, 1, 1],
                        ridge_factor=float(ridge_factor),
                        delta_ridge_factor=float(delta_ridge_factor),
                    )
                cv_to_use = cv_to
                cv_lo_use = cv_lo
                f_use = float("nan")

            if gamma_trend == "constant":
                a1 = float(fit_s["A_1_mol_per_J"])
                a2 = float(fit_s["A_2_mol_per_J"])
                a3 = float(fit_s["A_3_mol_per_J"])
                a4 = float(fit_s["A_4_mol_per_J"])
                delta_s = 0.0
                pred_s = [
                    a1 * float(c1) + a2 * float(c2) + a3 * float(c3) + a4 * float(c4)
                    for c1, c2, c3, c4 in zip(x_ta, x_la, x_to_use, x_lo_use)
                ]
            else:
                if gamma_trend_weights is None:
                    raise SystemExit("[fail] gamma_trend_weights missing")
                a1 = float(fit_s["a"][0])
                a2 = float(fit_s["a"][1])
                a3 = float(fit_s["a"][2])
                a4 = float(fit_s["a"][3])
                delta_s = float(fit_s["delta_gamma"])
                g_ta = gamma_trend_weights["ta"]
                g_la = gamma_trend_weights["la"]
                g_to = gamma_trend_weights["to"]
                g_lo = gamma_trend_weights["lo"]
                pred_s = [
                    (a1 + delta_s * float(g_ta[i])) * float(x_ta[i])
                    + (a2 + delta_s * float(g_la[i])) * float(x_la[i])
                    + (a3 + delta_s * float(g_to[i])) * float(x_to_use[i])
                    + (a4 + delta_s * float(g_lo[i])) * float(x_lo_use[i])
                    for i in range(len(temps))
                ]
            m_train = _metrics_for_range(
                idx=train_idx,
                alpha_obs=alpha_obs,
                alpha_pred=pred_s,
                sigma_fit=sigma_fit,
                param_count=((5 if softening_enabled else 4) + (1 if gamma_trend != "constant" else 0)),
                is_train=True,
            )
            m_test = _metrics_for_range(
                idx=test_idx,
                alpha_obs=alpha_obs,
                alpha_pred=pred_s,
                sigma_fit=sigma_fit,
                param_count=((5 if softening_enabled else 4) + (1 if gamma_trend != "constant" else 0)),
                is_train=False,
            )
            split_results.append(
                {
                    "name": str(s["name"]),
                    "train_T_K": [train_min, train_max],
                    "test_T_K": [test_min, test_max],
                    "params": {
                        "A_TA_mol_per_J": float(a1),
                        "A_LA_mol_per_J": float(a2),
                        "A_TO_mol_per_J": float(a3),
                        "A_LO_mol_per_J": float(a4),
                        "delta_gamma_dimensionless": float(delta_s) if gamma_trend != "constant" else "",
                        "softening_frac_at_Tmax": float(f_use) if softening_enabled else "",
                        "enforce_signs": bool(args.enforce_signs),
                        "ridge_factor": float(ridge_factor),
                        "ridge_lambda": (
                            float(fit_s.get("ridge_lambda", float("nan"))) if isinstance(fit_s.get("ridge_lambda"), (int, float)) else ""
                        ),
                        "coeff_unit": ("gamma_dimensionless" if use_bulk_modulus else "A_mol_per_J"),
                        "fit_sse": float(fit_s.get("sse", float("nan"))),
                    },
                    "train": m_train,
                    "test": m_test,
                }
            )

    # CSV
    out_csv = out_dir / f"{out_tag}.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "T_K",
            "alpha_obs_1e-8_per_K",
            "alpha_pred_1e-8_per_K",
            "sigma_fit_1e-8_per_K",
            "residual_1e-8_per_K",
            "z",
        ]
        if gamma_omega_model != "none":
            for lbl in gamma_omega_basis_labels:
                cols += [
                    f"Cv_basis_{lbl}_J_per_molK",
                    f"contrib_basis_{lbl}_1e-8_per_K",
                ]
            if (
                float(gamma_omega_high_softening_delta) != 0.0
                or float(gamma_omega_softening_delta) != 0.0
                or gamma_omega_softening_fit != "none"
            ):
                cols += ["contrib_gamma_omega_softening_delta_1e-8_per_K"]
        elif int(args.groups) == 2:
            cols += [
                "Cv_low_J_per_molK",
                "Cv_high_J_per_molK",
                "contrib_low_1e-8_per_K",
                "contrib_high_1e-8_per_K",
            ]
        elif int(args.groups) == 3:
            cols += [
                "Cv_TA_J_per_molK",
                "Cv_LA_J_per_molK",
                "Cv_opt_J_per_molK",
                "contrib_TA_1e-8_per_K",
                "contrib_LA_1e-8_per_K",
                "contrib_opt_1e-8_per_K",
            ]
        else:
            cols += [
                "Cv_TA_J_per_molK",
                "Cv_LA_J_per_molK",
                "Cv_TO_J_per_molK",
                "Cv_LO_J_per_molK",
                "contrib_TA_1e-8_per_K",
                "contrib_LA_1e-8_per_K",
                "contrib_TO_1e-8_per_K",
                "contrib_LO_1e-8_per_K",
            ]

        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, (t, ao, ap, sig, r1, zz) in enumerate(
            zip(temps, alpha_obs_1e8, alpha_pred_1e8, sigma_fit_1e8, residual_1e8, z)
        ):
            row: dict[str, object] = {
                "T_K": float(t),
                "alpha_obs_1e-8_per_K": float(ao),
                "alpha_pred_1e-8_per_K": float(ap),
                "sigma_fit_1e-8_per_K": float(sig),
                "residual_1e-8_per_K": float(r1),
                "z": float(zz) if math.isfinite(float(zz)) else "",
            }
            if gamma_omega_model != "none":
                if gamma_omega_coeffs is None:
                    raise SystemExit("[fail] internal: gamma_omega_coeffs missing")
                inv = float(inv_bv[i]) if use_bulk_modulus else 1.0
                for lbl in gamma_omega_basis_labels:
                    c = float(cv_gamma_omega[lbl][i])
                    a = float(gamma_omega_coeffs.get(lbl, float("nan")))
                    row.update(
                        {
                            f"Cv_basis_{lbl}_J_per_molK": c,
                            f"contrib_basis_{lbl}_1e-8_per_K": float(a * c * inv) / 1e-8,
                        }
                    )
                if (
                    float(gamma_omega_high_softening_delta) != 0.0
                    or float(gamma_omega_softening_delta) != 0.0
                    or gamma_omega_softening_fit != "none"
                ):
                    row["contrib_gamma_omega_softening_delta_1e-8_per_K"] = float(alpha_corr[i]) / 1e-8
            elif int(args.groups) == 2:
                c1 = float(cv_low[i])
                c2 = float(cv_high[i])
                inv = float(inv_bv[i]) if use_bulk_modulus else 1.0
                a_low_eff = float(a_low)
                a_high_eff = float(a_high)
                if gamma_trend != "constant":
                    if gamma_trend_weights is None:
                        raise SystemExit("[fail] gamma_trend_weights missing")
                    a_low_eff = float(a_low) + float(delta_gamma) * float(gamma_trend_weights["acoustic"][i])
                    a_high_eff = float(a_high) + float(delta_gamma) * float(gamma_trend_weights["optical"][i])
                row.update(
                    {
                        "Cv_low_J_per_molK": c1,
                        "Cv_high_J_per_molK": c2,
                        "contrib_low_1e-8_per_K": float(a_low_eff * c1 * inv) / 1e-8,
                        "contrib_high_1e-8_per_K": float(a_high_eff * c2 * inv) / 1e-8,
                    }
                )
            elif int(args.groups) == 3:
                c1 = float(cv_ta[i])
                c2 = float(cv_la[i])
                c3 = float(cv_opt[i])
                inv = float(inv_bv[i]) if use_bulk_modulus else 1.0
                a_ta_eff = float(a_ta)
                a_la_eff = float(a_la)
                a_opt_eff = float(a_opt)
                if gamma_trend != "constant":
                    if gamma_trend_weights is None:
                        raise SystemExit("[fail] gamma_trend_weights missing")
                    a_ta_eff = float(a_ta) + float(delta_gamma) * float(gamma_trend_weights["ta"][i])
                    a_la_eff = float(a_la) + float(delta_gamma) * float(gamma_trend_weights["la"][i])
                    a_opt_eff = float(a_opt) + float(delta_gamma) * float(gamma_trend_weights["optical"][i])
                row.update(
                    {
                        "Cv_TA_J_per_molK": c1,
                        "Cv_LA_J_per_molK": c2,
                        "Cv_opt_J_per_molK": c3,
                        "contrib_TA_1e-8_per_K": float(a_ta_eff * c1 * inv) / 1e-8,
                        "contrib_LA_1e-8_per_K": float(a_la_eff * c2 * inv) / 1e-8,
                        "contrib_opt_1e-8_per_K": float(a_opt_eff * c3 * inv) / 1e-8,
                    }
                )
            else:
                c1 = float(cv_ta[i])
                c2 = float(cv_la[i])
                c3 = float(cv_to[i])
                c4 = float(cv_lo[i])
                inv = float(inv_bv[i]) if use_bulk_modulus else 1.0
                a_ta_eff = float(a_ta)
                a_la_eff = float(a_la)
                a_to_eff = float(a_to)
                a_lo_eff = float(a_lo)
                if gamma_trend != "constant":
                    if gamma_trend_weights is None:
                        raise SystemExit("[fail] gamma_trend_weights missing")
                    a_ta_eff = float(a_ta) + float(delta_gamma) * float(gamma_trend_weights["ta"][i])
                    a_la_eff = float(a_la) + float(delta_gamma) * float(gamma_trend_weights["la"][i])
                    a_to_eff = float(a_to) + float(delta_gamma) * float(gamma_trend_weights["to"][i])
                    a_lo_eff = float(a_lo) + float(delta_gamma) * float(gamma_trend_weights["lo"][i])
                row.update(
                    {
                        "Cv_TA_J_per_molK": c1,
                        "Cv_LA_J_per_molK": c2,
                        "Cv_TO_J_per_molK": c3,
                        "Cv_LO_J_per_molK": c4,
                        "contrib_TA_1e-8_per_K": float(a_ta_eff * c1 * inv) / 1e-8,
                        "contrib_LA_1e-8_per_K": float(a_la_eff * c2 * inv) / 1e-8,
                        "contrib_TO_1e-8_per_K": float(a_to_eff * c3 * inv) / 1e-8,
                        "contrib_LO_1e-8_per_K": float(a_lo_eff * c4 * inv) / 1e-8,
                    }
                )
            w.writerow(row)

    # Plot
    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(11.0, 9.2), sharex=False, gridspec_kw={"height_ratios": [1, 2, 1]}
    )

    # DOS (per atom), converted to per-THz for readability.
    # g_omega dω = g_nu dν, with dω/dν = 2π.
    dos_per_thz = [float(g) * (2.0 * math.pi) * 1e12 for g in g_per_atom]  # 1/THz per atom
    ax0.plot(nu_thz, dos_per_thz, color="#444444", lw=1.6)
    ax0.axvline(nu_split_thz, color="#999999", lw=1.2, ls="--", alpha=0.9, label="acoustic/optical split")

    dos_title_prefix = "Si phonon DOS proxy"
    if dos_mode == "kim2015_fig1_energy":
        dos_title_prefix = "Si phonon DOS (Kim2015 Fig.1; T=300 K reference)"
    if int(args.groups) == 3:
        ax0.axvline(nu_split_ta_thz, color="#bbbbbb", lw=1.0, ls=":", alpha=0.9, label="TA/LA split (mode-count)")
        ax0.set_title(f"{dos_title_prefix} (per atom; mode-count splits: TA/LA/optical)")
    elif int(args.groups) == 4:
        ax0.axvline(nu_split_ta_thz, color="#bbbbbb", lw=1.0, ls=":", alpha=0.9, label="TA/LA split (mode-count)")
        ax0.axvline(nu_split_to_thz, color="#bbbbbb", lw=1.0, ls="-.", alpha=0.9, label="TO/LO split (mode-count)")
        ax0.set_title(f"{dos_title_prefix} (per atom; mode-count splits: TA/LA/TO/LO)")
    else:
        ax0.set_title(f"{dos_title_prefix} (per atom; split at half-integral ≈ acoustic/optical mode count)")
    ax0.set_ylabel("DOS (1/THz)")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="best", fontsize=9)

    # Alpha(T)
    ax1.plot(temps, alpha_obs_1e8, color="#d62728", lw=2.0, label="obs: NIST TRC fit α(T)")
    ax1.plot(temps, alpha_pred_1e8, color="#1f77b4", lw=2.0, label=f"pred: {model_name}")
    if gamma_omega_model != "none":
        if gamma_omega_coeffs is None:
            raise SystemExit("[fail] internal: gamma_omega_coeffs missing")
        styles = ["--", ":", "-."]
        for j, lbl in enumerate(gamma_omega_basis_labels):
            ls = styles[j] if j < len(styles) else (0, (1, 1))
            a = float(gamma_omega_coeffs.get(lbl, float("nan")))
            ax1.plot(
                temps,
                [
                    float(a) * float(cv_gamma_omega[lbl][i]) * (float(inv_bv[i]) if use_bulk_modulus else 1.0) / 1e-8
                    for i in range(len(temps))
                ],
                color="#1f77b4",
                lw=1.1,
                ls=ls,
                alpha=0.9,
                label=f"contrib: omega-basis {lbl}",
            )
        if float(gamma_omega_high_softening_delta) != 0.0 or float(gamma_omega_softening_delta) != 0.0:
            ax1.plot(
                temps,
                [float(alpha_corr[i]) / 1e-8 for i in range(len(temps))],
                color="#9467bd",
                lw=1.1,
                ls=(0, (3, 1)),
                alpha=0.9,
                label="contrib: gamma softening delta (fixed)",
            )
    elif int(args.groups) == 2:
        g_low_series = (
            gamma_trend_weights["acoustic"]
            if (gamma_trend != "constant" and gamma_trend_weights is not None)
            else [0.0 for _ in temps]
        )
        g_high_series = (
            gamma_trend_weights["optical"]
            if (gamma_trend != "constant" and gamma_trend_weights is not None)
            else [0.0 for _ in temps]
        )
        ax1.plot(
            temps,
            [
                float((a_low + delta_gamma * float(g_low_series[i])) * float(c) * (float(inv_bv[i]) if use_bulk_modulus else 1.0))
                / 1e-8
                for i, c in enumerate(cv_low)
            ],
            color="#1f77b4",
            lw=1.2,
            ls="--",
            alpha=0.9,
            label="contrib: low-ω group",
        )
        ax1.plot(
            temps,
            [
                float((a_high + delta_gamma * float(g_high_series[i])) * float(c) * (float(inv_bv[i]) if use_bulk_modulus else 1.0))
                / 1e-8
                for i, c in enumerate(cv_high)
            ],
            color="#1f77b4",
            lw=1.2,
            ls=":",
            alpha=0.9,
            label="contrib: high-ω group",
        )
    elif int(args.groups) == 3:
        g_ta_series = (
            gamma_trend_weights["ta"]
            if (gamma_trend != "constant" and gamma_trend_weights is not None)
            else [0.0 for _ in temps]
        )
        g_la_series = (
            gamma_trend_weights["la"]
            if (gamma_trend != "constant" and gamma_trend_weights is not None)
            else [0.0 for _ in temps]
        )
        g_opt_series = (
            gamma_trend_weights["optical"]
            if (gamma_trend != "constant" and gamma_trend_weights is not None)
            else [0.0 for _ in temps]
        )
        ax1.plot(
            temps,
            [
                float((a_ta + delta_gamma * float(g_ta_series[i])) * float(c) * (float(inv_bv[i]) if use_bulk_modulus else 1.0))
                / 1e-8
                for i, c in enumerate(cv_ta)
            ],
            color="#1f77b4",
            lw=1.0,
            ls="--",
            alpha=0.9,
            label="contrib: TA group",
        )
        ax1.plot(
            temps,
            [
                float((a_la + delta_gamma * float(g_la_series[i])) * float(c) * (float(inv_bv[i]) if use_bulk_modulus else 1.0))
                / 1e-8
                for i, c in enumerate(cv_la)
            ],
            color="#1f77b4",
            lw=1.0,
            ls=":",
            alpha=0.9,
            label="contrib: LA group",
        )
        ax1.plot(
            temps,
            [
                float((a_opt + delta_gamma * float(g_opt_series[i])) * float(c) * (float(inv_bv[i]) if use_bulk_modulus else 1.0))
                / 1e-8
                for i, c in enumerate(cv_opt)
            ],
            color="#1f77b4",
            lw=1.0,
            ls="-.",
            alpha=0.9,
            label="contrib: optical group",
        )
    else:
        g_ta_series = (
            gamma_trend_weights["ta"]
            if (gamma_trend != "constant" and gamma_trend_weights is not None)
            else [0.0 for _ in temps]
        )
        g_la_series = (
            gamma_trend_weights["la"]
            if (gamma_trend != "constant" and gamma_trend_weights is not None)
            else [0.0 for _ in temps]
        )
        g_to_series = (
            gamma_trend_weights["to"]
            if (gamma_trend != "constant" and gamma_trend_weights is not None)
            else [0.0 for _ in temps]
        )
        g_lo_series = (
            gamma_trend_weights["lo"]
            if (gamma_trend != "constant" and gamma_trend_weights is not None)
            else [0.0 for _ in temps]
        )
        ax1.plot(
            temps,
            [
                float((a_ta + delta_gamma * float(g_ta_series[i])) * float(c) * (float(inv_bv[i]) if use_bulk_modulus else 1.0))
                / 1e-8
                for i, c in enumerate(cv_ta)
            ],
            color="#1f77b4",
            lw=1.0,
            ls="--",
            alpha=0.9,
            label="contrib: TA group",
        )
        ax1.plot(
            temps,
            [
                float((a_la + delta_gamma * float(g_la_series[i])) * float(c) * (float(inv_bv[i]) if use_bulk_modulus else 1.0))
                / 1e-8
                for i, c in enumerate(cv_la)
            ],
            color="#1f77b4",
            lw=1.0,
            ls=":",
            alpha=0.9,
            label="contrib: LA group",
        )
        ax1.plot(
            temps,
            [
                float((a_to + delta_gamma * float(g_to_series[i])) * float(c) * (float(inv_bv[i]) if use_bulk_modulus else 1.0))
                / 1e-8
                for i, c in enumerate(cv_to)
            ],
            color="#1f77b4",
            lw=1.0,
            ls="-.",
            alpha=0.9,
            label="contrib: TO group",
        )
        ax1.plot(
            temps,
            [
                float((a_lo + delta_gamma * float(g_lo_series[i])) * float(c) * (float(inv_bv[i]) if use_bulk_modulus else 1.0))
                / 1e-8
                for i, c in enumerate(cv_lo)
            ],
            color="#1f77b4",
            lw=1.0,
            ls=(0, (1, 1)),
            alpha=0.9,
            label="contrib: LO group",
        )
    ax1.axhline(0.0, color="#666666", lw=1.0, alpha=0.6)
    if zero is not None:
        ax1.axvline(float(zero["x_cross"]), color="#999999", lw=1.0, ls="--", alpha=0.8)
        ax1.text(
            float(zero["x_cross"]) + 5,
            ax1.get_ylim()[0] * 0.75,
            f"sign change ~{zero['x_cross']:.0f} K",
            fontsize=9,
            alpha=0.85,
        )
    ax1.axvspan(50.0, float(t_max), color="#dddddd", alpha=0.12, label="fit range (global)")
    ax1.set_ylabel("α(T) (10^-8 / K)")
    ax1.set_title("Si thermal expansion: phonon DOS constrained Gruneisen check")
    if softening_enabled and softening_global_frac is not None:
        ax1.text(
            0.01,
            0.98,
            f"optical softening: f(Tmax)={softening_global_frac:.3f}",
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            alpha=0.9,
        )
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best", fontsize=9)

    # Residual z
    ax2.plot(temps, z, color="#000000", lw=1.2)
    ax2.axhline(0.0, color="#666666", lw=1.0, alpha=0.6)
    ax2.axhline(3.0, color="#d62728", lw=1.0, ls="--", alpha=0.7)
    ax2.axhline(-3.0, color="#d62728", lw=1.0, ls="--", alpha=0.7)
    ax2.set_xlabel("Temperature (K)")
    ax2.set_ylabel("z = (pred-obs)/σ_fit")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    out_png = out_dir / f"{out_tag}.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    # Metrics
    fit_metrics = _metrics_for_range(
        idx=fit_idx,
        alpha_obs=alpha_obs,
        alpha_pred=alpha_pred,
        sigma_fit=sigma_fit,
        param_count=(
            (len(gamma_omega_basis_labels) + (1 if gamma_omega_softening_fit != "none" else 0))
            if gamma_omega_model != "none"
            else (int(args.groups) + (1 if softening_enabled else 0) + (1 if gamma_trend != "constant" else 0))
        ),
        is_train=True,
    )
    strict_ok = bool(fit_metrics["max_abs_z"] <= 3.0 and fit_metrics["reduced_chi2"] <= 2.0 and fit_metrics["sign_mismatch_n"] == 0)

    holdout_ok = True
    for s in split_results:
        m_test = s["test"]
        if not isinstance(m_test, dict):
            holdout_ok = False
            continue
        if (
            float(m_test.get("max_abs_z", 1e9)) > 3.0
            or float(m_test.get("reduced_chi2", 1e9)) > 2.0
            or int(m_test.get("sign_mismatch_n", 1)) != 0
        ):
            holdout_ok = False

    if dos_mode == "kim2015_fig1_energy":
        dos_info = {
            "mode": "kim2015_fig1_energy",
            "fig1_digitized_ref": fig1_ref,
            "note": (
                "Cv(T) is built from temperature-dependent g_T(ε) digitized from Kim2015 Fig.1. "
                "The DOS curve shown in the plot is a reference conversion of g_T(ε) at T=300 K into ω space "
                "with normalization ∫g dω=3 per atom; the implied number density is a dummy and not physical."
            ),
            "splits_at_tref": {
                "acoustic_optical": {"omega_rad_s": float(omega_split), "nu_THz": float(nu_split_thz)},
            },
        }
        if int(args.groups) in (3, 4):
            dos_info["splits_at_tref"]["ta_la"] = {"omega_rad_s": float(omega_split_ta), "nu_THz": float(nu_split_ta_thz)}
        if int(args.groups) == 4:
            dos_info["splits_at_tref"]["to_lo"] = {"omega_rad_s": float(omega_split_to), "nu_THz": float(nu_split_to_thz)}
    else:
        dos_info = {
            "mode": "static_omega",
            "integral_dos_domega_per_m3": float(integral),
            "implied_atom_number_density_per_m3": float(n_atoms_m3),
            "integral_per_atom": float(integral_per_atom),
            "splits": {
                "acoustic_optical": {"omega_rad_s": float(omega_split), "nu_THz": float(nu_split_thz)},
            },
        }
        if int(args.groups) in (3, 4):
            dos_info["splits"]["ta_la"] = {"omega_rad_s": float(omega_split_ta), "nu_THz": float(nu_split_ta_thz)}
        if int(args.groups) == 4:
            dos_info["splits"]["to_lo"] = {"omega_rad_s": float(omega_split_to), "nu_THz": float(nu_split_to_thz)}

    model_info: dict[str, object] = {
        "name": model_name,
        "groups": int(args.groups),
        "fit_range_T_K": {"min": float(fit_min_k), "max": float(fit_max_k)},
        "enforce_signs": bool(args.enforce_signs),
        "coeff_unit": ("gamma_dimensionless" if use_bulk_modulus else "A_mol_per_J"),
        "ridge_factor": float(ridge_factor),
        "delta_ridge_factor": float(delta_ridge_factor),
        "cv_omega_dependence": str(cv_omega_dependence),
    }
    if use_bulk_modulus:
        model_info["formula"] = "alpha≈Σ γ_i·Cv_i/(B(T)·V_m)"
        if b_model is not None:
            model_info["bulk_modulus_model"] = dict(b_model)
        if v_m is not None:
            model_info["silicon_molar_volume"] = dict(v_m)
        if use_vm_thermal_expansion and v_m is not None and v_m_t is not None and v_m_t:
            model_info["molar_volume_T_correction"] = {
                "mode": "alpha_integral",
                "t_ref_K": 300.0,
                "v_ref_m3_per_mol": float(v_m.get("V_m3_per_mol", float("nan"))),
                "v_min_m3_per_mol": float(min(float(x) for x in v_m_t)),
                "v_max_m3_per_mol": float(max(float(x) for x in v_m_t)),
                "note": "V_m(T) is derived from the observed alpha(T) fit via d ln V / dT = 3 alpha(T).",
            }
    if str(args.dos_softening) == "kim2015_linear_proxy" and global_softening is not None:
        model_info["dos_softening"] = {
            "mode": "kim2015_linear_proxy",
            "source": dict(global_softening.get("source", {})) if isinstance(global_softening.get("source"), dict) else {},
            "proxy": dict(global_softening.get("proxy", {})) if isinstance(global_softening.get("proxy"), dict) else {},
        }
    if mode_softening_mode in ("kim2015_fig2_features", "kim2015_fig2_features_eq8_quasi") and mode_softening is not None:
        model_info["mode_softening"] = {
            "mode": str(mode_softening_mode),
            "source": dict(mode_softening.get("source", {})) if isinstance(mode_softening.get("source"), dict) else {},
            "eq8_quasiharmonic": (
                dict(mode_softening.get("eq8_quasiharmonic", {}))
                if isinstance(mode_softening.get("eq8_quasiharmonic"), dict)
                else (mode_softening.get("eq8_quasiharmonic", "") if mode_softening_mode.endswith("_eq8_quasi") else "")
            ),
            "note": "Mode-dependent ω(T) scales are frozen from Kim2015 Fig.2 digitization (no fit parameters).",
        }
    if softening_enabled:
        soft_obj: dict[str, object] = {
            "mode": str(args.optical_softening),
            "min_scale": float(args.softening_min_scale),
            "frac_at_Tmax_fit": float(softening_global_frac) if softening_global_frac is not None else "",
            "scan": softening_scan,
            "best": softening_best or {},
            "note": "f is fit by grid scan on the train range (global or per-holdout split) as an anharmonicity proxy.",
        }
        if softening_mode == "linear_fit":
            soft_obj.update({"kind": "linear_scale", "scale_Tmax_ref_K": float(max(temps))})
        else:
            soft_obj.update({"kind": "raman_shape"})
            if raman_source_path is not None:
                soft_obj["raman_shape_source"] = {"path": str(raman_source_path), "sha256": _sha256(raman_source_path)}
        model_info["optical_softening"] = soft_obj
        if use_bulk_modulus:
            if gamma_omega_model != "none":
                gamma_omega_note = (
                    "gamma(omega) is represented by fixed basis weights in normalized frequency w=omega/omega_max. "
                )
                if gamma_omega_model == "linear_endpoints":
                    gamma_omega_note += "Weights are (1-w), w."
                elif gamma_omega_model == "pwlinear_split":
                    gamma_omega_note += "Piecewise-linear weights with a knot at w_split (acoustic/optical split; half-integral of the DOS)."
                elif gamma_omega_model == "pwlinear_split_leaky":
                    gamma_omega_note += (
                        "Piecewise-linear weights with a knot at w_split plus a small overlap controlled by epsilon "
                        "(--gamma-omega-pwlinear-leak). Optional warp w->w^p is controlled by "
                        "--gamma-omega-pwlinear-warp-power (p=1 means no warp)."
                    )
                elif gamma_omega_model == "bernstein2":
                    gamma_omega_note += (
                        "Quadratic Bernstein weights: (1-w)^2, 2w(1-w), w^2. "
                        "w_split (acoustic/optical split) is reported for reference."
                    )
                else:
                    gamma_omega_note += f"Mode={gamma_omega_model}."
                model_info["gamma_omega_model"] = {
                    "mode": str(gamma_omega_model),
                    "w_split_acoustic_optical": (float(gamma_omega_w_split) if math.isfinite(float(gamma_omega_w_split)) else ""),
                    "basis_labels": list(gamma_omega_basis_labels),
                    "pwlinear_leak_epsilon": (float(gamma_omega_pwlinear_leak) if gamma_omega_model == "pwlinear_split_leaky" else ""),
                    "pwlinear_leak_warp_power": (
                        float(gamma_omega_pwlinear_warp_power) if gamma_omega_model == "pwlinear_split_leaky" else ""
                    ),
                    "gamma_softening_fit": (str(gamma_omega_softening_fit) if gamma_omega_softening_fit != "none" else ""),
                    "gamma_softening_fit_delta": (float(gamma_omega_softening_fit_delta) if gamma_omega_softening_fit != "none" else ""),
                    "gamma_softening_delta": (
                        float(gamma_omega_softening_delta) if float(gamma_omega_softening_delta) != 0.0 else ""
                    ),
                    "gamma_high_softening_delta": (
                        float(gamma_omega_high_softening_delta) if float(gamma_omega_high_softening_delta) != 0.0 else ""
                    ),
                    "gamma_softening_g_centered_at_K": (
                        300.0
                        if (
                            gamma_omega_softening_fit != "none"
                            or float(gamma_omega_softening_delta) != 0.0
                            or float(gamma_omega_high_softening_delta) != 0.0
                        )
                        else ""
                    ),
                    "note": gamma_omega_note,
                }
            if gamma_omega_coeffs is not None:
                model_info["gamma_omega_coefficients_dimensionless"] = dict(gamma_omega_coeffs)
            if gamma_omega_fit is not None:
                model_info["gamma_omega_fit"] = dict(gamma_omega_fit)
        elif gamma_trend != "constant":
            if gamma_trend == "kim2015_fig2_softening_common":
                gamma_trend_def = "gamma_i(T)=gamma_i0 + Delta_gamma*(1-omega_scale_i(T))"
                gamma_trend_note = (
                    "omega_scale_i(T) is frozen from Kim2015 Fig.2 digitization per DOS group (see mode_softening)."
                )
            elif gamma_trend == "kim2015_fig2_softening_common_centered300":
                gamma_trend_def = "gamma_i(T)=gamma_i(300K) + Delta_gamma*g_i(T)"
                gamma_trend_note = (
                    "g_i(T) is (1-omega_scale_i(T)) centered at 300 K and normalized to max abs=1 over the alpha(T) grid; "
                    "omega_scale_i(T) is frozen from Kim2015 Fig.2 digitization per DOS group (see mode_softening)."
                )
            elif gamma_trend == "linear_T":
                gamma_trend_def = "gamma_i(T)=gamma_i0 + Delta_gamma*((T-300K)/T_max)"
                gamma_trend_note = "T_max is the alpha(T) max grid temperature (data_range.t_max_k); 300K is the centering point."
            else:
                gamma_trend_def = "gamma_i(T)=gamma_i0 + Delta_gamma*g_i(T)"
                gamma_trend_note = "g_i(T) is defined by the selected --gamma-trend mode."
            model_info["gamma_trend"] = {
                "mode": str(gamma_trend),
                "delta_gamma_dimensionless": float(delta_gamma),
                "definition": gamma_trend_def,
                "note": gamma_trend_note,
            }
            if int(args.groups) == 2:
                model_info.update({"gamma_low0_dimensionless": float(a_low), "gamma_high0_dimensionless": float(a_high)})
            elif int(args.groups) == 3:
                model_info.update(
                    {"gamma_TA0_dimensionless": float(a_ta), "gamma_LA0_dimensionless": float(a_la), "gamma_opt0_dimensionless": float(a_opt)}
                )
            else:
                model_info.update(
                    {
                        "gamma_TA0_dimensionless": float(a_ta),
                        "gamma_LA0_dimensionless": float(a_la),
                        "gamma_TO0_dimensionless": float(a_to),
                        "gamma_LO0_dimensionless": float(a_lo),
                    }
                )
        else:
            if int(args.groups) == 2:
                model_info.update({"gamma_low_dimensionless": float(a_low), "gamma_high_dimensionless": float(a_high)})
            elif int(args.groups) == 3:
                model_info.update(
                    {"gamma_TA_dimensionless": float(a_ta), "gamma_LA_dimensionless": float(a_la), "gamma_opt_dimensionless": float(a_opt)}
                )
            else:
                model_info.update(
                    {
                        "gamma_TA_dimensionless": float(a_ta),
                        "gamma_LA_dimensionless": float(a_la),
                        "gamma_TO_dimensionless": float(a_to),
                        "gamma_LO_dimensionless": float(a_lo),
                    }
                )
    else:
        if int(args.groups) == 2:
            model_info.update({"A_low_mol_per_J": float(a_low), "A_high_mol_per_J": float(a_high)})
        elif int(args.groups) == 3:
            model_info.update({"A_TA_mol_per_J": float(a_ta), "A_LA_mol_per_J": float(a_la), "A_opt_mol_per_J": float(a_opt)})
        else:
            model_info.update(
                {
                    "A_TA_mol_per_J": float(a_ta),
                    "A_LA_mol_per_J": float(a_la),
                    "A_TO_mol_per_J": float(a_to),
                    "A_LO_mol_per_J": float(a_lo),
                }
            )

    falsification_notes = [
        (
            "This test freezes the mode-weighting via a phonon DOS proxy, and fits only a small number of effective coefficients (omega-basis γ)."
            if (use_bulk_modulus and gamma_omega_model != "none")
            else (
                "This test freezes the mode-weighting via a phonon DOS proxy, and fits only a small number of effective coefficients (per-group γ)."
                if use_bulk_modulus
                else "This test freezes the mode-weighting via a phonon DOS proxy, and fits only a small number of effective coefficients (per-group A)."
            )
        ),
        "holdout_ok requires the test ranges (A/B) to remain within the same strict criteria, to reject range-selection artifacts.",
    ]
    if dos_mode == "kim2015_fig1_energy":
        falsification_notes.append(
            "Temperature-dependent g_T(ε) is frozen from a primary INS-based reference (Kim et al. PRB 91, 014307; Fig.1 digitized); it is not fit and is not counted as a free parameter."
        )
    if str(args.dos_softening) == "kim2015_linear_proxy":
        falsification_notes.append(
            "A global ω(T) softening scale is frozen from a primary INS-based reference (Kim et al. PRB 91, 014307); it is not fit and is not counted as a free parameter."
        )
    if mode_softening_mode in ("kim2015_fig2_features", "kim2015_fig2_features_eq8_quasi"):
        note = (
            "Mode-dependent ω(T) scales are frozen from a primary INS-based reference (Kim et al. PRB 91, 014307; Fig.2 digitized); "
            "they are not fit and are not counted as free parameters."
        )
        if mode_softening_mode == "kim2015_fig2_features_eq8_quasi":
            note = note + " An Eq.(8) quasiharmonic-only exponent is applied to reduce intrinsic (T|V) contamination (fixed at 300 K)."
        falsification_notes.append(note)
    if gamma_trend != "constant":
        if gamma_trend in ("kim2015_fig2_softening_common", "kim2015_fig2_softening_common_centered300"):
            falsification_notes.append(
                "An additional 1D parameter Delta_gamma (gamma(T) trend tied to the frozen omega_scale_i(T)) is fit on train; it is counted as a free parameter in reduced χ²."
            )
        else:
            falsification_notes.append(
                "An additional 1D parameter Delta_gamma (gamma(T) trend) is fit on train; it is counted as a free parameter in reduced χ²."
            )
    if softening_enabled:
        falsification_notes.append(
            "An additional 1D softening parameter f (optical ω scaling) is fit on train via grid scan; it is counted as a free parameter in reduced χ²."
        )

    inputs_obj: dict[str, object] = {
        "silicon_thermal_expansion_extracted_values": {"path": str(alpha_src), "sha256": _sha256(alpha_src)},
        "silicon_phonon_dos_extracted_values": {"path": str(dos_src), "sha256": _sha256(dos_src)},
    }
    if use_bulk_modulus and b_model is not None:
        inputs_obj["ioffe_silicon_mechanical_properties_extracted_values"] = {"path": str(b_model["path"]), "sha256": str(b_model["sha256"])}
    if use_bulk_modulus and v_m is not None:
        inputs_obj["silicon_lattice_codata_extracted_values"] = {"path": str(v_m["path"]), "sha256": str(v_m["sha256"])}
    if str(args.dos_softening) == "kim2015_linear_proxy" and global_softening is not None:
        src_obj = global_softening.get("source") if isinstance(global_softening, dict) else None
        if isinstance(src_obj, dict) and isinstance(src_obj.get("path"), str):
            p = Path(str(src_obj["path"]))
            if p.exists():
                inputs_obj["silicon_phonon_anharmonicity_extracted_values"] = {"path": str(p), "sha256": _sha256(p)}
    if mode_softening_mode in ("kim2015_fig2_features", "kim2015_fig2_features_eq8_quasi") and mode_softening is not None:
        src_obj = mode_softening.get("source") if isinstance(mode_softening, dict) else None
        if isinstance(src_obj, dict) and isinstance(src_obj.get("path"), str):
            p = Path(str(src_obj["path"]))
            if p.exists():
                inputs_obj["silicon_phonon_feature_softening_extracted_values"] = {"path": str(p), "sha256": _sha256(p)}
    if softening_enabled and softening_mode == "raman_shape_fit" and raman_source_path is not None:
        inputs_obj["silicon_raman_phonon_shift_extracted_values"] = {
            "path": str(raman_source_path),
            "sha256": _sha256(raman_source_path),
        }

    out_metrics = out_dir / f"{out_tag}_metrics.json"

    cv_total = [float(sum(float(v[i]) for v in [cv_ta, cv_la, cv_to, cv_lo])) for i in range(len(temps))] if int(args.groups) == 4 else []
    kim2015_diag: dict[str, object] | None = None
    if mode_softening_mode in ("kim2015_fig2_features", "kim2015_fig2_features_eq8_quasi"):
        kim2015_diag = _kim2015_table1_gruneisen_diagnostics(
            root=root,
            fig2_source_dirname=str(args.kim2015_fig2_source_dir),
            fig2_json_name=str(args.kim2015_fig2_json_name),
            temps_k=temps,
            alpha_obs=alpha_obs,
            cv_total_j_per_mol_k=cv_total if cv_total else [float("nan") for _ in temps],
            bulk_modulus_model=b_model if use_bulk_modulus else None,
            silicon_molar_volume=v_m if use_bulk_modulus else None,
        )

    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": _iso_utc_now(),
                "phase": 7,
                "step": "7.14.20",
                "inputs": inputs_obj,
                "dos": dos_info,
                "model": model_info,
                "kim2015_gruneisen_diagnostics": kim2015_diag if kim2015_diag is not None else "",
                "fit_metrics": fit_metrics,
                "holdout_splits": split_results,
                "falsification": {
                    "strict_criteria": {"max_abs_z_le": 3.0, "reduced_chi2_le": 2.0, "sign_mismatch_n_eq": 0},
                    "strict_ok": bool(strict_ok),
                    "holdout_ok": bool(holdout_ok),
                    "notes": falsification_notes,
                },
                "outputs": {"csv": str(out_csv), "png": str(out_png)},
                "notes": [
                    "ω-split is defined at the half-integral of D(ω) to proxy the 3+3 mode count split (acoustic vs optical) in diamond Si.",
                    "For groups=3, TA/LA split uses the 2:1 mode-count ratio within acoustic branches (2 TA, 1 LA).",
                    (
                        "With --use-bulk-modulus, coefficients are γ_i (dimensionless) and α is computed as Σ γ_i·Cv_i/(B(T)·V_m)."
                        if use_bulk_modulus
                        else "A parameters correspond to effective (γ/(B V_m)) per group up to a convention factor; this step focuses on procedure sensitivity (holdout) rather than absolute γ values."
                    ),
                ],
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
