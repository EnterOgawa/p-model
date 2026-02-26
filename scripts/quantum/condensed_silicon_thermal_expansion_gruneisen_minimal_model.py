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


def _rmse(xs: list[float]) -> float:
    # 条件分岐: `not xs` を満たす経路を評価する。
    if not xs:
        return float("nan")

    return math.sqrt(sum(float(x) ** 2 for x in xs) / len(xs))


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
            # Prefer the physically relevant negative→positive crossing (e.g., Si has a low-T positive→negative wiggle).
            return max(neg_to_pos, key=lambda c: float(c["x_cross"]))

    return max(candidates, key=lambda c: float(c["x_cross"]))


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

    # Theta_D: prefer the frozen baseline if present; otherwise refit from JANAF (reproducible fallback).

    theta_from_metrics = _theta_d_from_existing_metrics(root)
    theta_fit_points: list[DebyeFitPoint] = []
    # 条件分岐: `theta_from_metrics is not None` を満たす経路を評価する。
    if theta_from_metrics is not None:
        theta_d = float(theta_from_metrics)
    else:
        theta_d, theta_fit_points = _fit_theta_d_from_janaf(root)

    # Prepare grid.

    temps = [float(t) for t in range(t_min, t_max + 1)]
    alpha_obs_1e8 = [_alpha_1e8_per_k(t_k=t, coeffs=coeffs) for t in temps]
    alpha_obs = [float(a) * 1e-8 for a in alpha_obs_1e8]  # 1/K
    cv = [_debye_cv_molar(t_k=t, theta_d_k=theta_d) for t in temps]  # J/molK

    # Minimal Debye–Grüneisen model: alpha(T) ≈ A_eff * Cv(T; theta_D).
    # Fit A_eff on a temperature range where alpha is positive.
    fit_min_k = 200.0
    fit_max_k = float(t_max)
    fit_idx = [i for i, t in enumerate(temps) if fit_min_k <= float(t) <= fit_max_k]
    # 条件分岐: `len(fit_idx) < 10` を満たす経路を評価する。
    if len(fit_idx) < 10:
        raise SystemExit(f"[fail] not enough fit points for A_eff fit: n={len(fit_idx)} in [{fit_min_k},{fit_max_k}] K")

    num = sum(float(cv[i]) * float(alpha_obs[i]) for i in fit_idx)
    den = sum(float(cv[i]) ** 2 for i in fit_idx)
    # 条件分岐: `den <= 0.0` を満たす経路を評価する。
    if den <= 0.0:
        raise SystemExit("[fail] invalid denominator for A_eff fit")

    a_eff = float(num / den)  # mol/J

    alpha_pred = [a_eff * float(c) for c in cv]
    alpha_pred_1e8 = [float(a) / 1e-8 for a in alpha_pred]

    residual = [float(ap - ao) for ap, ao in zip(alpha_pred, alpha_obs)]
    residual_1e8 = [float(r) / 1e-8 for r in residual]

    rel_residual = []
    for ap, ao in zip(alpha_pred, alpha_obs):
        # 条件分岐: `ao == 0.0` を満たす経路を評価する。
        if ao == 0.0:
            rel_residual.append(float("nan"))
        else:
            rel_residual.append(float((ap - ao) / ao))

    # Diagnostics.

    ratio_alpha_over_cv = []
    for ao, c in zip(alpha_obs, cv):
        ratio_alpha_over_cv.append(float("nan") if c == 0.0 else float(ao / c))

    sign_mismatch = 0
    for ao, ap in zip(alpha_obs, alpha_pred):
        # 条件分岐: `ao == 0.0 or ap == 0.0` を満たす経路を評価する。
        if ao == 0.0 or ap == 0.0:
            continue

        # 条件分岐: `(ao > 0.0) != (ap > 0.0)` を満たす経路を評価する。

        if (ao > 0.0) != (ap > 0.0):
            sign_mismatch += 1

    fit_rel = [rel_residual[i] for i in fit_idx if math.isfinite(float(rel_residual[i]))]
    fit_rel_rmse = _rmse([float(x) for x in fit_rel])
    fit_rel_min = float(min(fit_rel)) if fit_rel else float("nan")
    fit_rel_max = float(max(fit_rel)) if fit_rel else float("nan")

    zero = _infer_zero_crossing(temps, alpha_obs, prefer_neg_to_pos=True, min_x=50.0)

    # CSV
    out_csv = out_dir / "condensed_silicon_thermal_expansion_gruneisen_minimal_model.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "T_K",
                "alpha_obs_1e-8_per_K",
                "alpha_pred_1e-8_per_K",
                "Cv_debye_J_per_molK",
                "A_eff_alpha_over_Cv_mol_per_J",
                "residual_1e-8_per_K",
                "rel_residual",
            ],
        )
        w.writeheader()
        for t, ao, ap, c, a_over_c, r1e8, rr in zip(
            temps, alpha_obs_1e8, alpha_pred_1e8, cv, ratio_alpha_over_cv, residual_1e8, rel_residual
        ):
            w.writerow(
                {
                    "T_K": float(t),
                    "alpha_obs_1e-8_per_K": float(ao),
                    "alpha_pred_1e-8_per_K": float(ap),
                    "Cv_debye_J_per_molK": float(c),
                    "A_eff_alpha_over_Cv_mol_per_J": float(a_over_c) if math.isfinite(float(a_over_c)) else "",
                    "residual_1e-8_per_K": float(r1e8),
                    "rel_residual": float(rr) if math.isfinite(float(rr)) else "",
                }
            )

    # Plot

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.2, 7.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(temps, alpha_obs_1e8, color="#d62728", lw=2.0, label="NIST TRC fit (obs) α(T)")
    ax1.plot(temps, alpha_pred_1e8, color="#1f77b4", lw=2.0, label="Debye–Grüneisen minimal: α≈A·Cv(θ_D)")
    ax1.axhline(0.0, color="#666666", lw=1.0, alpha=0.6)
    # 条件分岐: `zero is not None` を満たす経路を評価する。
    if zero is not None:
        ax1.axvline(float(zero["x_cross"]), color="#999999", lw=1.0, ls="--", alpha=0.8)
        ax1.text(
            float(zero["x_cross"]) + 5,
            ax1.get_ylim()[0] * 0.8,
            f"sign change ~{zero['x_cross']:.0f} K",
            fontsize=9,
            alpha=0.85,
        )

    ax1.set_ylabel("α(T) (10^-8 / K)")
    ax1.set_title("Silicon thermal expansion: Debye–Grüneisen minimal model check")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    ax2.plot(temps, ratio_alpha_over_cv, color="#000000", lw=1.5, label="α/Cv (obs)")
    ax2.axhline(a_eff, color="#1f77b4", lw=2.0, alpha=0.85, label=f"A_fit (T≥{fit_min_k:.0f} K)")
    ax2.set_xlabel("Temperature T (K)")
    ax2.set_ylabel("α/Cv (mol/J)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    fig.tight_layout()
    out_png = out_dir / "condensed_silicon_thermal_expansion_gruneisen_minimal_model.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    # Falsification: this minimal model is rejected if it cannot reproduce the sign change and/or
    # if proportionality fails badly even on the positive-alpha range.
    reject_rel_rmse_threshold = 0.05
    rejected = (sign_mismatch > 0) or (fit_rel_rmse >= reject_rel_rmse_threshold)

    out_metrics = out_dir / "condensed_silicon_thermal_expansion_gruneisen_minimal_model_metrics.json"
    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.14.8",
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
                "model": {
                    "name": "Debye–Grüneisen minimal (constant A_eff)",
                    "theta_D_K": float(theta_d),
                    "fit_range_T_K": {"min": float(fit_min_k), "max": float(fit_max_k)},
                    "A_eff_mol_per_J": float(a_eff),
                },
                "diagnostics": {
                    "alpha_sign_change_T_K": (None if zero is None else float(zero["x_cross"])),
                    "sign_mismatch_n": int(sign_mismatch),
                    "rel_error_fit_range": {
                        "rmse": float(fit_rel_rmse),
                        "min": float(fit_rel_min),
                        "max": float(fit_rel_max),
                    },
                },
                "falsification": {
                    "reject_if_sign_mismatch_n_gt": 0,
                    "reject_if_rel_error_rmse_ge": float(reject_rel_rmse_threshold),
                    "rejected": bool(rejected),
                    "notes": [
                        "In a strict Debye–Grüneisen picture with constant A_eff=γ/(B V_m), α(T) is proportional to Cv(T) and cannot change sign.",
                        "The NIST TRC fit shows negative α at low T, so the constant-A_eff minimal model is rejected over the full 13–600 K range.",
                        "Even restricting to the positive-α range (T≥200 K), α is not proportional to Debye Cv with a constant factor; additional physics (e.g., mode-dependent γ or anharmonicity) is required.",
                    ],
                },
                "outputs": {"csv": str(out_csv), "png": str(out_png)},
                "notes": [
                    "This step is a minimal-model check to define what must be reproduced by any future derivation.",
                    "Cv is computed from a 1-parameter Debye model using θ_D fixed (or refit) from JANAF Cp° points.",
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


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
