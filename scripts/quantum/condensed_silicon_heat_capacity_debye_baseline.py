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


def _cp_shomate(*, coeffs: dict[str, float], t_k: float) -> float:
    t = t_k / 1000.0
    a = float(coeffs["A"])
    b = float(coeffs["B"])
    c = float(coeffs["C"])
    d = float(coeffs["D"])
    e = float(coeffs["E"])
    return a + b * t + c * (t**2) + d * (t**3) + (e / (t**2))


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
    # 条件分岐: `t_k <= 0.0 or theta_d_k <= 0.0` を満たす経路を評価する。
    if t_k <= 0.0 or theta_d_k <= 0.0:
        return 0.0

    # Molar gas constant.

    r = 8.314462618
    y = theta_d_k / t_k

    # Truncate the upper integration limit: for x>~50 the integrand ~ x^4 e^{-x} is negligible.
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


@dataclass(frozen=True)
class FitRow:
    t_k: float
    cp_obs: float
    cp_debye: float
    resid: float


def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = root / "data" / "quantum" / "sources" / "nist_janaf_silicon_si"
    extracted_path = src_dir / "extracted_values.json"
    # 条件分岐: `not extracted_path.exists()` を満たす経路を評価する。
    if not extracted_path.exists():
        raise SystemExit(
            f"[fail] missing: {extracted_path}\n"
            "Run: python -B scripts/quantum/fetch_silicon_janaf_sources.py"
        )

    extracted = _read_json(extracted_path)
    points = extracted.get("points")
    # 条件分岐: `not isinstance(points, list) or not points` を満たす経路を評価する。
    if not isinstance(points, list) or not points:
        raise SystemExit(f"[fail] points missing/empty: {extracted_path}")

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
        raise SystemExit(f"[fail] no solid-phase points found in: {extracted_path}")

    # Fit range: JANAF provides discrete points; keep a "low-to-mid" range where Debye is meaningful.

    fit_points = [
        (float(p["T_K"]), float(p["Cp_J_per_molK"]))
        for p in solid
        if 100.0 <= float(p["T_K"]) <= 300.0
    ]
    # 条件分岐: `len(fit_points) < 4` を満たす経路を評価する。
    if len(fit_points) < 4:
        raise SystemExit(f"[fail] not enough fit points in 100–300 K range: n={len(fit_points)}")

    fit_points = sorted(fit_points, key=lambda x: x[0])

    def sse(theta: float) -> float:
        return sum((cp - _debye_cv_molar(t_k=t, theta_d_k=theta)) ** 2 for (t, cp) in fit_points)

    theta0, _ = _golden_section_minimize(sse, 300.0, 900.0, tol=1e-5)

    fit_rows: list[FitRow] = []
    for t, cp in fit_points:
        cp_fit = _debye_cv_molar(t_k=t, theta_d_k=theta0)
        fit_rows.append(FitRow(t_k=t, cp_obs=cp, cp_debye=cp_fit, resid=(cp - cp_fit)))

    n = len(fit_rows)
    dof = max(1, n - 1)
    sse_val = sum(r.resid**2 for r in fit_rows)
    rmse = math.sqrt(sse_val / dof)

    # Parameter uncertainty (proxy): linearized least-squares variance using residual RMS.
    dtheta = max(1e-3, 1e-4 * theta0)
    derivs = []
    for r in fit_rows:
        cp_p = _debye_cv_molar(t_k=r.t_k, theta_d_k=theta0 + dtheta)
        cp_m = _debye_cv_molar(t_k=r.t_k, theta_d_k=theta0 - dtheta)
        dcp_dtheta = (cp_p - cp_m) / (2.0 * dtheta)
        derivs.append(dcp_dtheta)

    denom = sum(d * d for d in derivs)
    sigma_theta = math.sqrt((rmse**2) / denom) if denom > 0 else None

    # Cross-check against WebBook Shomate (solid) at overlap points.
    shomate_src = root / "data" / "quantum" / "sources" / "nist_webbook_condensed_silicon_si" / "extracted_values.json"
    shomate_cross: list[dict[str, Any]] = []
    shomate_blocks = None
    # 条件分岐: `shomate_src.exists()` を満たす経路を評価する。
    if shomate_src.exists():
        sh = _read_json(shomate_src)
        sh_list = sh.get("shomate")
        # 条件分岐: `isinstance(sh_list, list)` を満たす経路を評価する。
        if isinstance(sh_list, list):
            shomate_blocks = sh_list

    shomate_solid_coeffs: Optional[dict[str, float]] = None
    # 条件分岐: `isinstance(shomate_blocks, list)` を満たす経路を評価する。
    if isinstance(shomate_blocks, list):
        for b in shomate_blocks:
            # 条件分岐: `not isinstance(b, dict)` を満たす経路を評価する。
            if not isinstance(b, dict):
                continue

            # 条件分岐: `str(b.get("phase")) != "solid"` を満たす経路を評価する。

            if str(b.get("phase")) != "solid":
                continue

            coeffs = b.get("coeffs")
            # 条件分岐: `isinstance(coeffs, dict) and all(k in coeffs for k in ["A", "B", "C", "D", "E"])` を満たす経路を評価する。
            if isinstance(coeffs, dict) and all(k in coeffs for k in ["A", "B", "C", "D", "E"]):
                shomate_solid_coeffs = {k: float(coeffs[k]) for k in coeffs.keys()}
                break

    for t in [298.15, 300.0]:
        cp_j = None
        for tt, cp in fit_points:
            # 条件分岐: `abs(tt - t) < 1e-6` を満たす経路を評価する。
            if abs(tt - t) < 1e-6:
                cp_j = cp

        # 条件分岐: `cp_j is None` を満たす経路を評価する。

        if cp_j is None:
            continue

        # 条件分岐: `shomate_solid_coeffs is None` を満たす経路を評価する。

        if shomate_solid_coeffs is None:
            continue

        cp_s = _cp_shomate(coeffs=shomate_solid_coeffs, t_k=t)
        shomate_cross.append(
            {
                "T_K": float(t),
                "Cp_JANAF_J_per_molK": float(cp_j),
                "Cp_Shomate_J_per_molK": float(cp_s),
                "delta_J_per_molK": float(cp_s - cp_j),
            }
        )

    # CSV (fit range points).

    out_csv = out_dir / "condensed_silicon_heat_capacity_debye_baseline.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["T_K", "Cp_obs_J_per_molK", "Cp_debye_J_per_molK", "residual_J_per_molK"])
        w.writeheader()
        for r in fit_rows:
            w.writerow(
                {
                    "T_K": r.t_k,
                    "Cp_obs_J_per_molK": r.cp_obs,
                    "Cp_debye_J_per_molK": r.cp_debye,
                    "residual_J_per_molK": r.resid,
                }
            )

    # Plot: low temperature domain + overlap with Shomate near 300 K.

    xs_curve = [float(t) for t in range(0, 351, 2)]
    ys_curve = [_debye_cv_molar(t_k=t, theta_d_k=theta0) for t in xs_curve]

    plt.figure(figsize=(8.5, 4.6))
    plt.plot(xs_curve, ys_curve, color="#1f77b4", label=f"Debye fit (θ_D≈{theta0:.1f} K)")
    plt.scatter([r.t_k for r in fit_rows], [r.cp_obs for r in fit_rows], color="#000000", s=24, label="JANAF Cp° (fit points)")

    # 条件分岐: `shomate_solid_coeffs is not None` を満たす経路を評価する。
    if shomate_solid_coeffs is not None:
        xs_s = [298 + i for i in range(0, 53, 2)]
        ys_s = [_cp_shomate(coeffs=shomate_solid_coeffs, t_k=float(t)) for t in xs_s]
        plt.plot(xs_s, ys_s, color="#d62728", linestyle="--", label="WebBook Shomate (solid; 298–350 K)")

    plt.xlabel("Temperature T (K)")
    plt.ylabel("Heat capacity (J/mol·K)")
    plt.title("Silicon heat capacity: JANAF baseline + Debye-fit target (low T)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_png = out_dir / "condensed_silicon_heat_capacity_debye_baseline.png"
    plt.savefig(out_png, dpi=180)
    plt.close()

    reject_abs_k = None
    # 条件分岐: `sigma_theta is not None and math.isfinite(float(sigma_theta))` を満たす経路を評価する。
    if sigma_theta is not None and math.isfinite(float(sigma_theta)):
        reject_abs_k = float(max(3.0 * sigma_theta, 5.0))

    out_metrics = out_dir / "condensed_silicon_heat_capacity_debye_baseline_metrics.json"
    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.14.3",
                "inputs": {
                    "janaf_extracted_values": {"path": str(extracted_path), "sha256": _sha256(extracted_path)},
                    "webbook_extracted_values": (
                        {"path": str(shomate_src), "sha256": _sha256(shomate_src)} if shomate_src.exists() else None
                    ),
                },
                "fit": {
                    "model": "Debye heat capacity (Cv) used as low-T Cp proxy (monatomic; 3R limit)",
                    "fit_range_T_K": {"min": float(fit_points[0][0]), "max": float(fit_points[-1][0])},
                    "n_points": int(n),
                    "theta_D_K": float(theta0),
                    "sse": float(sse_val),
                    "rmse_proxy_J_per_molK": float(rmse),
                    "theta_D_sigma_proxy_K": None if sigma_theta is None else float(sigma_theta),
                    "theta_D_covariance_proxy_K2": None if sigma_theta is None else float(sigma_theta**2),
                },
                "fit_points": [
                    {"T_K": r.t_k, "Cp_obs_J_per_molK": r.cp_obs, "Cp_debye_J_per_molK": r.cp_debye, "residual_J_per_molK": r.resid}
                    for r in fit_rows
                ],
                "cross_checks": {"janaf_vs_webbook_shomate_solid": shomate_cross},
                "falsification": {
                    "target": "theta_D_K",
                    "reject_if_abs_theta_D_minus_target_gt_K": reject_abs_k,
                    "notes": [
                        "JANAF table provides no explicit uncertainties; sigma is a proxy estimated from residual scatter and local sensitivity.",
                        "Reject threshold uses max(3*sigma_proxy, 5 K) as a conservative floor to avoid false precision.",
                    ],
                },
                "outputs": {"csv": str(out_csv), "png": str(out_png)},
                "notes": [
                    "This is a baseline/target for Step 7.14: low-temperature Cp and a 1-parameter Debye fit (θ_D) as a compact summary.",
                    "Debye model yields Cv; Cp≈Cv at low T. Near 300 K, Cp−Cv and anharmonicity can matter; this is treated as a proxy.",
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
