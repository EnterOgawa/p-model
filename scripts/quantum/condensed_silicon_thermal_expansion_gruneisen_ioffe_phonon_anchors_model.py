from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_ROOT = _repo_root()
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse the same implementation as Step 7.14.11/7.14.14 so that this test
# does not fork the underlying physics/utility code.

from scripts.quantum.condensed_silicon_thermal_expansion_gruneisen_debye_einstein_model import (  # noqa: E402
    _alpha_1e8_per_k,
    _debye_cv_molar,
    _einstein_cv_molar,
    _fit_theta_d_from_janaf,
    _infer_zero_crossing,
    _read_json,
    _solve_3x3,
    _theta_d_from_existing_metrics,
)


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


def _pick_ioffe_phonon_anchors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Choose two anchor modes from Ioffe's small phonon-frequency table.

    Heuristic (candidate set):
      - low branch candidates: TA modes (often associated with negative contributions to α(T))
      - high branch candidates: modes containing 'O' (optical-like)

    Selection:
      - choose the pair that minimizes the weighted SSE on the target fit range (the weights
        are based on the NIST σ_fit scale, consistent with earlier Step 7.14 tests).
    """
    # 条件分岐: `not rows` を満たす経路を評価する。
    if not rows:
        raise ValueError("empty phonon rows")

    low_candidates = [r for r in rows if str(r.get("mode") or "").upper() == "TA"]
    high_candidates = [r for r in rows if "O" in str(r.get("mode") or "").upper()]

    # 条件分岐: `not low_candidates` を満たす経路を評価する。
    if not low_candidates:
        low_candidates = list(rows)

    # 条件分岐: `not high_candidates` を満たす経路を評価する。

    if not high_candidates:
        high_candidates = list(rows)

    return {
        "low_candidates": low_candidates,
        "high_candidates": high_candidates,
        "notes": [
            "Anchors are selected from the cached Ioffe table and evaluated as a small discrete set.",
            "Candidate sets: low=TA modes; high=optical-like (mode contains 'O') modes; fall back to all rows if empty.",
        ],
    }


def _weighted_sse(
    *, idx: list[int], alpha_obs: list[float], alpha_pred: list[float], sigma_fit: list[float]
) -> float:
    sse = 0.0
    for i in idx:
        sig = float(sigma_fit[i])
        # 条件分岐: `sig <= 0.0` を満たす経路を評価する。
        if sig <= 0.0:
            continue

        w = 1.0 / (sig * sig)
        r = float(alpha_obs[i]) - float(alpha_pred[i])
        sse += w * r * r

    return float(sse)


def _weighted_fit_3branch(
    *,
    idx: list[int],
    alpha_obs: list[float],
    sigma_fit: list[float],
    cv_d: list[float],
    cv_e1: list[float],
    cv_e2: list[float],
) -> tuple[float, float, float]:
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
        sig = float(sigma_fit[i])
        # 条件分岐: `sig <= 0.0` を満たす経路を評価する。
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

    sol = _solve_3x3(a11=s11, a12=s12, a13=s13, a22=s22, a23=s23, a33=s33, b1=b1, b2=b2, b3=b3)
    # 条件分岐: `sol is None` を満たす経路を評価する。
    if sol is None:
        raise SystemExit("[fail] singular 3x3 system (fit)")

    a_d, a_e1, a_e2 = sol
    return float(a_d), float(a_e1), float(a_e2)


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
    # Symmetric 4x4 with Gaussian elimination on the augmented matrix.
    m = [
        [float(a11), float(a12), float(a13), float(a14), float(b1)],
        [float(a12), float(a22), float(a23), float(a24), float(b2)],
        [float(a13), float(a23), float(a33), float(a34), float(b3)],
        [float(a14), float(a24), float(a34), float(a44), float(b4)],
    ]

    # Partial pivoting.
    for col in range(4):
        pivot = col
        max_abs = abs(float(m[col][col]))
        for r in range(col + 1, 4):
            v = abs(float(m[r][col]))
            # 条件分岐: `v > max_abs` を満たす経路を評価する。
            if v > max_abs:
                max_abs = v
                pivot = r

        # 条件分岐: `max_abs <= 1e-30 or not math.isfinite(max_abs)` を満たす経路を評価する。

        if max_abs <= 1e-30 or not math.isfinite(max_abs):
            return None

        # 条件分岐: `pivot != col` を満たす経路を評価する。

        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]

        piv = float(m[col][col])
        inv = 1.0 / piv
        for j in range(col, 5):
            m[col][j] *= inv

        # Eliminate other rows.

        for r in range(4):
            # 条件分岐: `r == col` を満たす経路を評価する。
            if r == col:
                continue

            factor = float(m[r][col])
            # 条件分岐: `factor == 0.0` を満たす経路を評価する。
            if factor == 0.0:
                continue

            for j in range(col, 5):
                m[r][j] -= factor * float(m[col][j])

    x1 = float(m[0][4])
    x2 = float(m[1][4])
    x3 = float(m[2][4])
    x4 = float(m[3][4])
    # 条件分岐: `not all(math.isfinite(x) for x in (x1, x2, x3, x4))` を満たす経路を評価する。
    if not all(math.isfinite(x) for x in (x1, x2, x3, x4)):
        return None

    return x1, x2, x3, x4


def _weighted_fit_4branch(
    *,
    idx: list[int],
    alpha_obs: list[float],
    sigma_fit: list[float],
    cv_d: list[float],
    cv_e1: list[float],
    cv_e2: list[float],
    cv_e3: list[float],
) -> tuple[float, float, float, float]:
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
        sig = float(sigma_fit[i])
        # 条件分岐: `sig <= 0.0` を満たす経路を評価する。
        if sig <= 0.0:
            continue

        w = 1.0 / (sig * sig)
        x1 = float(cv_d[i])
        x2 = float(cv_e1[i])
        x3 = float(cv_e2[i])
        x4 = float(cv_e3[i])
        y = float(alpha_obs[i])
        s11 += w * x1 * x1
        s22 += w * x2 * x2
        s33 += w * x3 * x3
        s44 += w * x4 * x4
        s12 += w * x1 * x2
        s13 += w * x1 * x3
        s14 += w * x1 * x4
        s23 += w * x2 * x3
        s24 += w * x2 * x4
        s34 += w * x3 * x4
        b1 += w * x1 * y
        b2 += w * x2 * y
        b3 += w * x3 * y
        b4 += w * x4 * y

    sol = _solve_4x4(
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
    # 条件分岐: `sol is None` を満たす経路を評価する。
    if sol is None:
        raise SystemExit("[fail] singular 4x4 system (fit)")

    a_d, a_e1, a_e2, a_e3 = sol
    return float(a_d), float(a_e1), float(a_e2), float(a_e3)


def _metrics_fit_range(
    *,
    idx: list[int],
    alpha_obs: list[float],
    alpha_pred: list[float],
    sigma_fit: list[float],
    param_count: int,
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
        # 条件分岐: `sig <= 0.0` を満たす経路を評価する。
        if sig <= 0.0:
            continue

        # 条件分岐: `ao != 0.0 and ap != 0.0 and (ao > 0.0) != (ap > 0.0)` を満たす経路を評価する。

        if ao != 0.0 and ap != 0.0 and (ao > 0.0) != (ap > 0.0):
            sign_mismatch += 1

        z = (ap - ao) / sig
        # 条件分岐: `not math.isfinite(z)` を満たす経路を評価する。
        if not math.isfinite(z):
            continue

        n += 1
        sum_z2 += z * z
        max_abs_z = max(max_abs_z, abs(z))
        # 条件分岐: `abs(z) > 3.0` を満たす経路を評価する。
        if abs(z) > 3.0:
            exceed_3sigma += 1

    dof = max(1, n - int(param_count))
    rms_z = float("nan") if n <= 0 else math.sqrt(float(sum_z2 / n))
    red_chi2 = float(sum_z2 / dof)
    return {
        "n": int(n),
        "sign_mismatch_n": int(sign_mismatch),
        "max_abs_z": float(max_abs_z),
        "rms_z": float(rms_z),
        "reduced_chi2": float(red_chi2),
        "exceed_3sigma_n": int(exceed_3sigma),
    }


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Si thermal expansion: Debye+Einstein with θE anchors from Ioffe phonon-frequency table "
            "(discrete sweep; fit A's; optional holdout)."
        )
    )
    p.add_argument(
        "--einstein-anchors",
        type=int,
        default=2,
        choices=[2, 3],
        help="Number of Einstein anchors (2=Step 7.14.18; 3=Step 7.14.19).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
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

    temps = [float(t) for t in range(t_min, t_max + 1)]
    alpha_obs_1e8 = [_alpha_1e8_per_k(t_k=t, coeffs=coeffs) for t in temps]
    alpha_obs = [float(a) * 1e-8 for a in alpha_obs_1e8]  # 1/K
    sigma_fit_1e8 = [sigma_lt_1e8 if float(t) < t_sigma_split else sigma_ge_1e8 for t in temps]
    sigma_fit = [float(s) * 1e-8 for s in sigma_fit_1e8]

    # Theta_D baseline.
    theta_from_metrics = _theta_d_from_existing_metrics(root)
    # 条件分岐: `theta_from_metrics is not None` を満たす経路を評価する。
    if theta_from_metrics is not None:
        theta_d_k = float(theta_from_metrics)
        theta_d_source = {"kind": "frozen_metrics", "path": "output/public/quantum/condensed_silicon_heat_capacity_debye_baseline_metrics.json"}
    else:
        theta_d_k, _ = _fit_theta_d_from_janaf(root)
        theta_d_source = {"kind": "refit_from_janaf", "fit_range_K": [100.0, 300.0]}

    # Ioffe phonon anchors.

    ioffe_src = root / "data" / "quantum" / "sources" / "ioffe_silicon_mechanical_properties" / "extracted_values.json"
    # 条件分岐: `not ioffe_src.exists()` を満たす経路を評価する。
    if not ioffe_src.exists():
        raise SystemExit(
            f"[fail] missing: {ioffe_src}\n"
            "Run: python -B scripts/quantum/fetch_silicon_elastic_constants_sources.py"
        )

    ioffe_obj = _read_json(ioffe_src)
    pf = ioffe_obj.get("phonon_frequencies")
    # 条件分岐: `not isinstance(pf, dict) or not isinstance(pf.get("rows"), list)` を満たす経路を評価する。
    if not isinstance(pf, dict) or not isinstance(pf.get("rows"), list):
        raise SystemExit(f"[fail] missing ioffe phonon_frequencies rows: {ioffe_src}")

    cv_d = [_debye_cv_molar(t_k=t, theta_d_k=theta_d_k) for t in temps]

    # Fit range: match Step 7.14.11/7.14.14 definition.
    fit_min_k = 50.0
    fit_max_k = float(t_max)
    fit_idx = [i for i, t in enumerate(temps) if fit_min_k <= float(t) <= fit_max_k]
    # 条件分岐: `len(fit_idx) < 100` を満たす経路を評価する。
    if len(fit_idx) < 100:
        raise SystemExit(f"[fail] not enough fit points: n={len(fit_idx)} in [{fit_min_k},{fit_max_k}] K")

    # Anchor selection: evaluate a small discrete set from Ioffe and pick the best by weighted SSE.

    anchors_meta = _pick_ioffe_phonon_anchors([dict(r) for r in pf["rows"]])
    low_candidates = list(anchors_meta["low_candidates"])
    optical_candidates = list(anchors_meta["high_candidates"])

    pair_sweep: list[dict[str, Any]] = []
    triple_sweep: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    einstein_anchors = int(args.einstein_anchors)
    # 条件分岐: `einstein_anchors == 2` を満たす経路を評価する。
    if einstein_anchors == 2:
        for low in low_candidates:
            theta_e1_k = float(low["theta_K"])
            for high in optical_candidates:
                theta_e2_k = float(high["theta_K"])
                # 条件分岐: `not (0.0 < theta_e1_k < theta_e2_k)` を満たす経路を評価する。
                if not (0.0 < theta_e1_k < theta_e2_k):
                    continue

                cv_e1 = [_einstein_cv_molar(t_k=t, theta_e_k=theta_e1_k, dof=3.0) for t in temps]
                cv_e2 = [_einstein_cv_molar(t_k=t, theta_e_k=theta_e2_k, dof=3.0) for t in temps]
                a_d, a_e1, a_e2 = _weighted_fit_3branch(
                    idx=fit_idx,
                    alpha_obs=alpha_obs,
                    sigma_fit=sigma_fit,
                    cv_d=cv_d,
                    cv_e1=cv_e1,
                    cv_e2=cv_e2,
                )
                alpha_pred = [
                    a_d * float(cd) + a_e1 * float(c1) + a_e2 * float(c2)
                    for cd, c1, c2 in zip(cv_d, cv_e1, cv_e2)
                ]
                sse = _weighted_sse(idx=fit_idx, alpha_obs=alpha_obs, alpha_pred=alpha_pred, sigma_fit=sigma_fit)
                fit_metrics = _metrics_fit_range(
                    idx=fit_idx, alpha_obs=alpha_obs, alpha_pred=alpha_pred, sigma_fit=sigma_fit, param_count=3
                )
                rec = {
                    "low": {"label": low.get("label"), "mode": low.get("mode"), "point": low.get("point"), "theta_K": theta_e1_k},
                    "high": {"label": high.get("label"), "mode": high.get("mode"), "point": high.get("point"), "theta_K": theta_e2_k},
                    "fit": {
                        "A_D_mol_per_J": float(a_d),
                        "A_E1_mol_per_J": float(a_e1),
                        "A_E2_mol_per_J": float(a_e2),
                        "sse": float(sse),
                        "metrics": fit_metrics,
                    },
                }
                pair_sweep.append(rec)
                # 条件分岐: `best is None or float(sse) < float(best["fit"]["sse"])` を満たす経路を評価する。
                if best is None or float(sse) < float(best["fit"]["sse"]):
                    best = dict(rec)

        # 条件分岐: `best is None` を満たす経路を評価する。

        if best is None:
            raise SystemExit("[fail] no valid ioffe anchor pairs found")

        low = best["low"]
        high = best["high"]
        theta_e1_k = float(low["theta_K"])
        theta_e2_k = float(high["theta_K"])
        theta_e3_k = None

        cv_e1 = [_einstein_cv_molar(t_k=t, theta_e_k=theta_e1_k, dof=3.0) for t in temps]
        cv_e2 = [_einstein_cv_molar(t_k=t, theta_e_k=theta_e2_k, dof=3.0) for t in temps]
        cv_e3 = None

        a_d = float(best["fit"]["A_D_mol_per_J"])
        a_e1 = float(best["fit"]["A_E1_mol_per_J"])
        a_e2 = float(best["fit"]["A_E2_mol_per_J"])
        a_e3 = None
        alpha_pred = [a_d * float(cd) + a_e1 * float(c1) + a_e2 * float(c2) for cd, c1, c2 in zip(cv_d, cv_e1, cv_e2)]
        step_id = "7.14.18"
        out_tag = "condensed_silicon_thermal_expansion_gruneisen_ioffe_phonon_anchors_model"
        model_name = "Debye+Einstein×2 (θ_D frozen; θ_E1/θ_E2 anchored from Ioffe phonon frequencies; fit A's)"
    else:
        # Three Einstein anchors: E1 from TA candidates, E2/E3 from optical candidates (ordered).
        for low in low_candidates:
            theta_e1_k = float(low["theta_K"])
            for j2 in range(len(optical_candidates)):
                mid = optical_candidates[j2]
                theta_e2_k = float(mid["theta_K"])
                # 条件分岐: `not (0.0 < theta_e1_k < theta_e2_k)` を満たす経路を評価する。
                if not (0.0 < theta_e1_k < theta_e2_k):
                    continue

                for j3 in range(j2 + 1, len(optical_candidates)):
                    high = optical_candidates[j3]
                    theta_e3_k = float(high["theta_K"])
                    # 条件分岐: `not (0.0 < theta_e2_k < theta_e3_k)` を満たす経路を評価する。
                    if not (0.0 < theta_e2_k < theta_e3_k):
                        continue

                    cv_e1 = [_einstein_cv_molar(t_k=t, theta_e_k=theta_e1_k, dof=3.0) for t in temps]
                    cv_e2 = [_einstein_cv_molar(t_k=t, theta_e_k=theta_e2_k, dof=3.0) for t in temps]
                    cv_e3 = [_einstein_cv_molar(t_k=t, theta_e_k=theta_e3_k, dof=3.0) for t in temps]
                    a_d, a_e1, a_e2, a_e3 = _weighted_fit_4branch(
                        idx=fit_idx,
                        alpha_obs=alpha_obs,
                        sigma_fit=sigma_fit,
                        cv_d=cv_d,
                        cv_e1=cv_e1,
                        cv_e2=cv_e2,
                        cv_e3=cv_e3,
                    )
                    alpha_pred = [
                        a_d * float(cd) + a_e1 * float(c1) + a_e2 * float(c2) + a_e3 * float(c3)
                        for cd, c1, c2, c3 in zip(cv_d, cv_e1, cv_e2, cv_e3)
                    ]
                    sse = _weighted_sse(idx=fit_idx, alpha_obs=alpha_obs, alpha_pred=alpha_pred, sigma_fit=sigma_fit)
                    fit_metrics = _metrics_fit_range(
                        idx=fit_idx, alpha_obs=alpha_obs, alpha_pred=alpha_pred, sigma_fit=sigma_fit, param_count=4
                    )
                    rec = {
                        "low": {
                            "label": low.get("label"),
                            "mode": low.get("mode"),
                            "point": low.get("point"),
                            "theta_K": theta_e1_k,
                        },
                        "mid": {
                            "label": mid.get("label"),
                            "mode": mid.get("mode"),
                            "point": mid.get("point"),
                            "theta_K": theta_e2_k,
                        },
                        "high": {
                            "label": high.get("label"),
                            "mode": high.get("mode"),
                            "point": high.get("point"),
                            "theta_K": theta_e3_k,
                        },
                        "fit": {
                            "A_D_mol_per_J": float(a_d),
                            "A_E1_mol_per_J": float(a_e1),
                            "A_E2_mol_per_J": float(a_e2),
                            "A_E3_mol_per_J": float(a_e3),
                            "sse": float(sse),
                            "metrics": fit_metrics,
                        },
                    }
                    triple_sweep.append(rec)
                    # 条件分岐: `best is None or float(sse) < float(best["fit"]["sse"])` を満たす経路を評価する。
                    if best is None or float(sse) < float(best["fit"]["sse"]):
                        best = dict(rec)

        # 条件分岐: `best is None` を満たす経路を評価する。

        if best is None:
            raise SystemExit("[fail] no valid ioffe anchor triples found")

        low = best["low"]
        mid = best["mid"]
        high = best["high"]
        theta_e1_k = float(low["theta_K"])
        theta_e2_k = float(mid["theta_K"])
        theta_e3_k = float(high["theta_K"])

        cv_e1 = [_einstein_cv_molar(t_k=t, theta_e_k=theta_e1_k, dof=3.0) for t in temps]
        cv_e2 = [_einstein_cv_molar(t_k=t, theta_e_k=theta_e2_k, dof=3.0) for t in temps]
        cv_e3 = [_einstein_cv_molar(t_k=t, theta_e_k=theta_e3_k, dof=3.0) for t in temps]

        a_d = float(best["fit"]["A_D_mol_per_J"])
        a_e1 = float(best["fit"]["A_E1_mol_per_J"])
        a_e2 = float(best["fit"]["A_E2_mol_per_J"])
        a_e3 = float(best["fit"]["A_E3_mol_per_J"])
        alpha_pred = [
            a_d * float(cd) + a_e1 * float(c1) + a_e2 * float(c2) + a_e3 * float(c3)
            for cd, c1, c2, c3 in zip(cv_d, cv_e1, cv_e2, cv_e3)
        ]
        step_id = "7.14.19"
        out_tag = "condensed_silicon_thermal_expansion_gruneisen_ioffe_phonon_anchors_three_einstein_model"
        model_name = "Debye+Einstein×3 (θ_D frozen; θ_E anchored from Ioffe phonon frequencies; fit A's)"

    # Holdout (fixed θE1/θE2): fit only A's on train, evaluate test.

    holdout = []
    for name, train, test in [
        ("A", (50.0, 300.0), (300.0, 600.0)),
        ("B", (200.0, 600.0), (50.0, 200.0)),
    ]:
        train_idx = [i for i, t in enumerate(temps) if float(train[0]) <= float(t) <= float(train[1])]
        test_idx = [i for i, t in enumerate(temps) if float(test[0]) <= float(t) <= float(test[1])]
        # 条件分岐: `einstein_anchors == 2` を満たす経路を評価する。
        if einstein_anchors == 2:
            a_d_h, a_e1_h, a_e2_h = _weighted_fit_3branch(
                idx=train_idx,
                alpha_obs=alpha_obs,
                sigma_fit=sigma_fit,
                cv_d=cv_d,
                cv_e1=cv_e1,
                cv_e2=cv_e2,
            )
            alpha_pred_h = [
                a_d_h * float(cd) + a_e1_h * float(c1) + a_e2_h * float(c2) for cd, c1, c2 in zip(cv_d, cv_e1, cv_e2)
            ]
            params = {"A_D_mol_per_J": float(a_d_h), "A_E1_mol_per_J": float(a_e1_h), "A_E2_mol_per_J": float(a_e2_h)}
            param_count_train = 3
        else:
            assert cv_e3 is not None
            a_d_h, a_e1_h, a_e2_h, a_e3_h = _weighted_fit_4branch(
                idx=train_idx,
                alpha_obs=alpha_obs,
                sigma_fit=sigma_fit,
                cv_d=cv_d,
                cv_e1=cv_e1,
                cv_e2=cv_e2,
                cv_e3=cv_e3,
            )
            alpha_pred_h = [
                a_d_h * float(cd) + a_e1_h * float(c1) + a_e2_h * float(c2) + a_e3_h * float(c3)
                for cd, c1, c2, c3 in zip(cv_d, cv_e1, cv_e2, cv_e3)
            ]
            params = {
                "A_D_mol_per_J": float(a_d_h),
                "A_E1_mol_per_J": float(a_e1_h),
                "A_E2_mol_per_J": float(a_e2_h),
                "A_E3_mol_per_J": float(a_e3_h),
            }
            param_count_train = 4

        holdout.append(
            {
                "name": name,
                "train_T_K": [float(train[0]), float(train[1])],
                "test_T_K": [float(test[0]), float(test[1])],
                "params": params,
                "train": _metrics_fit_range(idx=train_idx, alpha_obs=alpha_obs, alpha_pred=alpha_pred_h, sigma_fit=sigma_fit, param_count=param_count_train),
                "test": _metrics_fit_range(idx=test_idx, alpha_obs=alpha_obs, alpha_pred=alpha_pred_h, sigma_fit=sigma_fit, param_count=0),
            }
        )

    zero_obs = _infer_zero_crossing(temps, alpha_obs, prefer_neg_to_pos=True, min_x=50.0)
    zero_pred = _infer_zero_crossing(temps, alpha_pred, prefer_neg_to_pos=True, min_x=50.0)

    metrics_fit = _metrics_fit_range(
        idx=fit_idx,
        alpha_obs=alpha_obs,
        alpha_pred=alpha_pred,
        sigma_fit=sigma_fit,
        param_count=(3 if einstein_anchors == 2 else 4),
    )

    rejected = bool(
        (int(metrics_fit["sign_mismatch_n"]) > 0)
        or (float(metrics_fit["max_abs_z"]) >= 3.0)
        or (float(metrics_fit["reduced_chi2"]) >= 2.0)
    )

    # CSV
    out_csv = out_dir / f"{out_tag}.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "T_K",
            "alpha_obs_1e-8_per_K",
            "alpha_pred_1e-8_per_K",
            "sigma_fit_1e-8_per_K",
            "z",
            "alpha_contrib_D_1e-8_per_K",
            "alpha_contrib_E1_1e-8_per_K",
            "alpha_contrib_E2_1e-8_per_K",
        ]
        # 条件分岐: `einstein_anchors == 3` を満たす経路を評価する。
        if einstein_anchors == 3:
            fieldnames.append("alpha_contrib_E3_1e-8_per_K")

        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        w.writeheader()
        # 条件分岐: `einstein_anchors == 2` を満たす経路を評価する。
        if einstein_anchors == 2:
            assert cv_e3 is None
            for t, ao, ap, sf, cd, c1, c2 in zip(temps, alpha_obs, alpha_pred, sigma_fit, cv_d, cv_e1, cv_e2):
                z = float("nan") if float(sf) <= 0.0 else (float(ap) - float(ao)) / float(sf)
                w.writerow(
                    {
                        "T_K": float(t),
                        "alpha_obs_1e-8_per_K": float(ao) / 1e-8,
                        "alpha_pred_1e-8_per_K": float(ap) / 1e-8,
                        "sigma_fit_1e-8_per_K": float(sf) / 1e-8,
                        "z": float(z),
                        "alpha_contrib_D_1e-8_per_K": float(a_d) * float(cd) / 1e-8,
                        "alpha_contrib_E1_1e-8_per_K": float(a_e1) * float(c1) / 1e-8,
                        "alpha_contrib_E2_1e-8_per_K": float(a_e2) * float(c2) / 1e-8,
                    }
                )
        else:
            assert cv_e3 is not None and a_e3 is not None
            for t, ao, ap, sf, cd, c1, c2, c3 in zip(temps, alpha_obs, alpha_pred, sigma_fit, cv_d, cv_e1, cv_e2, cv_e3):
                z = float("nan") if float(sf) <= 0.0 else (float(ap) - float(ao)) / float(sf)
                w.writerow(
                    {
                        "T_K": float(t),
                        "alpha_obs_1e-8_per_K": float(ao) / 1e-8,
                        "alpha_pred_1e-8_per_K": float(ap) / 1e-8,
                        "sigma_fit_1e-8_per_K": float(sf) / 1e-8,
                        "z": float(z),
                        "alpha_contrib_D_1e-8_per_K": float(a_d) * float(cd) / 1e-8,
                        "alpha_contrib_E1_1e-8_per_K": float(a_e1) * float(c1) / 1e-8,
                        "alpha_contrib_E2_1e-8_per_K": float(a_e2) * float(c2) / 1e-8,
                        "alpha_contrib_E3_1e-8_per_K": float(a_e3) * float(c3) / 1e-8,
                    }
                )

    # Figure

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11.8, 7.2), sharex=True)
    ax0.plot(temps, alpha_obs_1e8, color="#000000", lw=2.2, label="obs α(T) (NIST TRC fit)")
    ax0.plot(
        temps,
        [float(a) / 1e-8 for a in alpha_pred],
        color="#1f77b4",
        lw=2.0,
        label=("pred (Debye+Einstein×2)" if einstein_anchors == 2 else "pred (Debye+Einstein×3)"),
    )
    ax0.plot(temps, [float(a_d) * float(cd) / 1e-8 for cd in cv_d], color="#1f77b4", lw=1.2, ls="--", alpha=0.65, label="Debye contrib")
    ax0.plot(temps, [float(a_e1) * float(c1) / 1e-8 for c1 in cv_e1], color="#ff7f0e", lw=1.2, ls="--", alpha=0.65, label="E1 contrib")
    ax0.plot(temps, [float(a_e2) * float(c2) / 1e-8 for c2 in cv_e2], color="#2ca02c", lw=1.2, ls="--", alpha=0.65, label="E2 contrib")
    # 条件分岐: `einstein_anchors == 3 and cv_e3 is not None and a_e3 is not None` を満たす経路を評価する。
    if einstein_anchors == 3 and cv_e3 is not None and a_e3 is not None:
        ax0.plot(temps, [float(a_e3) * float(c3) / 1e-8 for c3 in cv_e3], color="#9467bd", lw=1.2, ls="--", alpha=0.65, label="E3 contrib")

    ax0.axhline(0.0, color="#999999", lw=1.0, alpha=0.6)
    # 条件分岐: `zero_obs is not None` を満たす経路を評価する。
    if zero_obs is not None:
        ax0.axvline(float(zero_obs["x_cross"]), color="#999999", lw=1.0, ls=":", alpha=0.7)

    # 条件分岐: `zero_pred is not None` を満たす経路を評価する。

    if zero_pred is not None:
        ax0.axvline(float(zero_pred["x_cross"]), color="#999999", lw=1.0, ls="--", alpha=0.7)

    ax0.set_ylabel("α (1e-8 / K)")
    ax0.set_title(f"Si thermal expansion: Debye+Einstein×{einstein_anchors} with θE anchors from Ioffe phonon frequencies")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper left", fontsize=9, ncols=2)

    zs = []
    for ao, ap, sf in zip(alpha_obs, alpha_pred, sigma_fit):
        # 条件分岐: `float(sf) <= 0.0` を満たす経路を評価する。
        if float(sf) <= 0.0:
            zs.append(float("nan"))
        else:
            zs.append((float(ap) - float(ao)) / float(sf))

    ax1.plot(temps, zs, color="#d62728", lw=1.4)
    ax1.axhline(0.0, color="#999999", lw=1.0, alpha=0.6)
    ax1.axhline(3.0, color="#999999", lw=1.0, ls="--", alpha=0.6)
    ax1.axhline(-3.0, color="#999999", lw=1.0, ls="--", alpha=0.6)
    ax1.set_xlabel("Temperature T (K)")
    ax1.set_ylabel("z = (pred−obs)/σ_fit")
    ax1.grid(True, alpha=0.25)

    # Small annotation block (anchors + fit metrics).
    if einstein_anchors == 2:
        ann = (
            f"E1(anchor)={low['label']}: θE1={theta_e1_k:.1f} K\n"
            f"E2(anchor)={high['label']}: θE2={theta_e2_k:.1f} K\n"
            f"fit: A_D={a_d:.3e}, A_E1={a_e1:.3e}, A_E2={a_e2:.3e}\n"
            f"fit-range: max|z|={metrics_fit['max_abs_z']:.2f}, χ²ν={metrics_fit['reduced_chi2']:.2f}"
        )
    else:
        assert theta_e3_k is not None and a_e3 is not None
        ann = (
            f"E1(anchor)={low['label']}: θE1={theta_e1_k:.1f} K\n"
            f"E2(anchor)={mid['label']}: θE2={theta_e2_k:.1f} K\n"
            f"E3(anchor)={high['label']}: θE3={theta_e3_k:.1f} K\n"
            f"fit: A_D={a_d:.3e}, A_E1={a_e1:.3e}, A_E2={a_e2:.3e}, A_E3={a_e3:.3e}\n"
            f"fit-range: max|z|={metrics_fit['max_abs_z']:.2f}, χ²ν={metrics_fit['reduced_chi2']:.2f}"
        )

    ax1.text(0.01, 0.02, ann, transform=ax1.transAxes, fontsize=9, va="bottom", ha="left")

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
                    "theta_d_source": theta_d_source,
                    "ioffe_silicon_mechanical_properties_extracted_values": {"path": str(ioffe_src), "sha256": _sha256(ioffe_src)},
                },
                "ioffe_phonon_anchors": {
                    "candidates": anchors_meta,
                    "pair_sweep": pair_sweep,
                    "triple_sweep": triple_sweep,
                    "best": best,
                },
                "model": {
                    "name": model_name,
                    "einstein_branches": int(einstein_anchors),
                    "theta_D_K": float(theta_d_k),
                    "theta_E1_K": float(theta_e1_k),
                    "theta_E2_K": float(theta_e2_k),
                    "theta_E3_K": (None if theta_e3_k is None else float(theta_e3_k)),
                    "A_D_mol_per_J": float(a_d),
                    "A_E1_mol_per_J": float(a_e1),
                    "A_E2_mol_per_J": float(a_e2),
                    "A_E3_mol_per_J": (None if a_e3 is None else float(a_e3)),
                    "fit_range_T_K": {"min": float(fit_min_k), "max": float(fit_max_k)},
                },
                "diagnostics": {
                    "alpha_sign_change_T_K": {
                        "obs": (None if zero_obs is None else float(zero_obs["x_cross"])),
                        "pred": (None if zero_pred is None else float(zero_pred["x_cross"])),
                    },
                    "fit_range": metrics_fit,
                    "sigma_fit_1e-8_per_K": {"lt_T_K": float(t_sigma_split), "lt": float(sigma_lt_1e8), "ge": float(sigma_ge_1e8)},
                    "holdout": holdout,
                },
                "falsification": {
                    "reject_if_sign_mismatch_fit_range_n_gt": 0,
                    "reject_if_max_abs_z_ge": 3.0,
                    "reject_if_reduced_chi2_ge": 2.0,
                    "rejected": bool(rejected),
                    "notes": [
                        "θ_E anchors are fixed from a primary reference table (Ioffe), to reduce fit-range sensitivity.",
                        "Acceptance is evaluated against the NIST TRC curve-fit standard-error scale σ_fit (proxy).",
                        "This check is still a fit (A's are estimated); it is not a derivation.",
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
