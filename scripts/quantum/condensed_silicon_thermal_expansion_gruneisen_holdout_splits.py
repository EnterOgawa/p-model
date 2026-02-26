from __future__ import annotations

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

# Reuse the same implementation as Step 7.14.11/7.14.14 so that
# the holdout test (7.14.15) is comparable and does not fork physics code.

from scripts.quantum.condensed_silicon_thermal_expansion_gruneisen_debye_einstein_model import (  # noqa: E402
    _alpha_1e8_per_k,
    _debye_cv_molar,
    _einstein_cv_molar,
    _fit_theta_d_from_janaf,
    _logspace,
    _read_json,
    _solve_2x2,
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


def _range_idx(temps: list[float], *, t_min: float, t_max: float) -> list[int]:
    return [i for i, t in enumerate(temps) if float(t_min) <= float(t) <= float(t_max)]


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


def _fit_debye_einstein_1(
    *,
    temps: list[float],
    alpha_obs: list[float],
    sigma_fit: list[float],
    cv_d: list[float],
    train_idx: list[int],
) -> dict[str, Any]:
    theta_e_grid = _logspace(lo=10.0, hi=4000.0, n=900)
    best: dict[str, float] | None = None

    for theta_e in theta_e_grid:
        cv_e = [_einstein_cv_molar(t_k=t, theta_e_k=float(theta_e), dof=3.0) for t in temps]
        s11 = 0.0
        s22 = 0.0
        s12 = 0.0
        b1 = 0.0
        b2 = 0.0
        for i in train_idx:
            sig = float(sigma_fit[i])
            # 条件分岐: `sig <= 0.0` を満たす経路を評価する。
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
        # 条件分岐: `sol is None` を満たす経路を評価する。
        if sol is None:
            continue

        a_d, a_e = sol

        sse = 0.0
        for i in train_idx:
            sig = float(sigma_fit[i])
            # 条件分岐: `sig <= 0.0` を満たす経路を評価する。
            if sig <= 0.0:
                continue

            w = 1.0 / (sig * sig)
            pred = float(a_d) * float(cv_d[i]) + float(a_e) * float(cv_e[i])
            r = float(alpha_obs[i]) - pred
            sse += w * r * r

        # 条件分岐: `best is None or float(sse) < float(best["sse"])` を満たす経路を評価する。

        if best is None or float(sse) < float(best["sse"]):
            best = {
                "theta_e_k": float(theta_e),
                "a_d": float(a_d),
                "a_e": float(a_e),
                "sse": float(sse),
            }

    # 条件分岐: `best is None` を満たす経路を評価する。

    if best is None:
        raise SystemExit("[fail] theta_E scan failed (einstein_branches=1)")

    theta_e_best = float(best["theta_e_k"])
    a_d_best = float(best["a_d"])
    a_e_best = float(best["a_e"])
    cv_e_best = [_einstein_cv_molar(t_k=t, theta_e_k=theta_e_best, dof=3.0) for t in temps]
    alpha_pred = [a_d_best * float(cd) + a_e_best * float(ce) for cd, ce in zip(cv_d, cv_e_best)]
    return {
        "params": {
            "theta_E_K": float(theta_e_best),
            "A_D_mol_per_J": float(a_d_best),
            "A_E_mol_per_J": float(a_e_best),
            "theta_E_scan": {"lo_K": 10.0, "hi_K": 4000.0, "n": int(len(theta_e_grid))},
        },
        "alpha_pred": alpha_pred,
    }


def _fit_debye_einstein_2(
    *,
    temps: list[float],
    alpha_obs: list[float],
    sigma_fit: list[float],
    cv_d: list[float],
    train_idx: list[int],
) -> dict[str, Any]:
    theta_e_grid = _logspace(lo=10.0, hi=4000.0, n=160)
    cv_e_grid = [[_einstein_cv_molar(t_k=t, theta_e_k=float(theta_e), dof=3.0) for t in temps] for theta_e in theta_e_grid]
    best: dict[str, float] | None = None
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
            for i in train_idx:
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
                continue

            a_d, a_e1, a_e2 = sol

            sse = 0.0
            for i in train_idx:
                sig = float(sigma_fit[i])
                # 条件分岐: `sig <= 0.0` を満たす経路を評価する。
                if sig <= 0.0:
                    continue

                w = 1.0 / (sig * sig)
                pred = float(a_d) * float(cv_d[i]) + float(a_e1) * float(cv_e1[i]) + float(a_e2) * float(cv_e2[i])
                r = float(alpha_obs[i]) - pred
                sse += w * r * r

            # 条件分岐: `best is None or float(sse) < float(best["sse"])` を満たす経路を評価する。

            if best is None or float(sse) < float(best["sse"]):
                best = {
                    "theta_e1_k": float(theta_e1),
                    "theta_e2_k": float(theta_e_grid[j2]),
                    "a_d": float(a_d),
                    "a_e1": float(a_e1),
                    "a_e2": float(a_e2),
                    "sse": float(sse),
                }

    # 条件分岐: `best is None` を満たす経路を評価する。

    if best is None:
        raise SystemExit("[fail] theta_E1/theta_E2 scan failed (einstein_branches=2)")

    theta_e1_best = float(best["theta_e1_k"])
    theta_e2_best = float(best["theta_e2_k"])
    a_d_best = float(best["a_d"])
    a_e1_best = float(best["a_e1"])
    a_e2_best = float(best["a_e2"])
    cv_e1_best = [_einstein_cv_molar(t_k=t, theta_e_k=theta_e1_best, dof=3.0) for t in temps]
    cv_e2_best = [_einstein_cv_molar(t_k=t, theta_e_k=theta_e2_best, dof=3.0) for t in temps]
    alpha_pred = [
        a_d_best * float(cd) + a_e1_best * float(ce1) + a_e2_best * float(ce2)
        for cd, ce1, ce2 in zip(cv_d, cv_e1_best, cv_e2_best)
    ]
    return {
        "params": {
            "theta_E1_K": float(theta_e1_best),
            "theta_E2_K": float(theta_e2_best),
            "A_D_mol_per_J": float(a_d_best),
            "A_E1_mol_per_J": float(a_e1_best),
            "A_E2_mol_per_J": float(a_e2_best),
            "theta_E_scan": {
                "lo_K": 10.0,
                "hi_K": 4000.0,
                "n": int(len(theta_e_grid)),
                "pairs_tested": int(tested),
                "ordering": "theta_E1 < theta_E2",
            },
        },
        "alpha_pred": alpha_pred,
    }


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
    # 条件分岐: `theta_from_metrics is not None` を満たす経路を評価する。
    if theta_from_metrics is not None:
        theta_d = float(theta_from_metrics)
        theta_d_source: dict[str, object] = {
            "kind": "frozen_metrics",
            "path": "output/public/quantum/condensed_silicon_heat_capacity_debye_baseline_metrics.json",
        }
    else:
        theta_d, _ = _fit_theta_d_from_janaf(root)
        theta_d_source = {"kind": "refit_from_janaf", "fit_range_K": [100.0, 300.0]}

    temps = [float(t) for t in range(t_min, t_max + 1)]
    alpha_obs_1e8 = [_alpha_1e8_per_k(t_k=t, coeffs=coeffs) for t in temps]
    alpha_obs = [float(a) * 1e-8 for a in alpha_obs_1e8]
    sigma_fit_1e8 = [sigma_lt_1e8 if float(t) < t_sigma_split else sigma_ge_1e8 for t in temps]
    sigma_fit = [float(s) * 1e-8 for s in sigma_fit_1e8]
    cv_d = [_debye_cv_molar(t_k=t, theta_d_k=theta_d) for t in temps]

    splits = [
        {"name": "A", "train": [50.0, 300.0], "test": [300.0, 600.0]},
        {"name": "B", "train": [200.0, 600.0], "test": [50.0, 200.0]},
    ]

    rows: list[dict[str, object]] = []
    split_results: list[dict[str, object]] = []

    for s in splits:
        train_min, train_max = float(s["train"][0]), float(s["train"][1])
        test_min, test_max = float(s["test"][0]), float(s["test"][1])
        train_idx = _range_idx(temps, t_min=train_min, t_max=train_max)
        test_idx = _range_idx(temps, t_min=test_min, t_max=test_max)
        # 条件分岐: `len(train_idx) < 20 or len(test_idx) < 20` を満たす経路を評価する。
        if len(train_idx) < 20 or len(test_idx) < 20:
            raise SystemExit(f"[fail] split {s['name']} has too few points: train={len(train_idx)}, test={len(test_idx)}")

        fit1 = _fit_debye_einstein_1(temps=temps, alpha_obs=alpha_obs, sigma_fit=sigma_fit, cv_d=cv_d, train_idx=train_idx)
        alpha_pred_1 = [float(x) for x in fit1["alpha_pred"]]
        m1_train = _metrics_for_range(
            idx=train_idx, alpha_obs=alpha_obs, alpha_pred=alpha_pred_1, sigma_fit=sigma_fit, param_count=3, is_train=True
        )
        m1_test = _metrics_for_range(
            idx=test_idx, alpha_obs=alpha_obs, alpha_pred=alpha_pred_1, sigma_fit=sigma_fit, param_count=3, is_train=False
        )

        fit2 = _fit_debye_einstein_2(temps=temps, alpha_obs=alpha_obs, sigma_fit=sigma_fit, cv_d=cv_d, train_idx=train_idx)
        alpha_pred_2 = [float(x) for x in fit2["alpha_pred"]]
        m2_train = _metrics_for_range(
            idx=train_idx, alpha_obs=alpha_obs, alpha_pred=alpha_pred_2, sigma_fit=sigma_fit, param_count=5, is_train=True
        )
        m2_test = _metrics_for_range(
            idx=test_idx, alpha_obs=alpha_obs, alpha_pred=alpha_pred_2, sigma_fit=sigma_fit, param_count=5, is_train=False
        )

        split_results.append(
            {
                "name": str(s["name"]),
                "train_T_K": [train_min, train_max],
                "test_T_K": [test_min, test_max],
                "models": {
                    "einstein_1": {"params": fit1["params"], "train": m1_train, "test": m1_test},
                    "einstein_2": {"params": fit2["params"], "train": m2_train, "test": m2_test},
                },
            }
        )

        for model_key, params, m_train, m_test in [
            ("einstein_1", fit1["params"], m1_train, m1_test),
            ("einstein_2", fit2["params"], m2_train, m2_test),
        ]:
            row: dict[str, object] = {
                "split": str(s["name"]),
                "model": model_key,
                "train_min_T_K": train_min,
                "train_max_T_K": train_max,
                "test_min_T_K": test_min,
                "test_max_T_K": test_max,
                "train_max_abs_z": float(m_train["max_abs_z"]),
                "train_reduced_chi2": float(m_train["reduced_chi2"]),
                "test_max_abs_z": float(m_test["max_abs_z"]),
                "test_reduced_chi2": float(m_test["reduced_chi2"]),
                "train_sign_mismatch_n": int(m_train["sign_mismatch_n"]),
                "test_sign_mismatch_n": int(m_test["sign_mismatch_n"]),
                "train_exceed_3sigma_n": int(m_train["exceed_3sigma_n"]),
                "test_exceed_3sigma_n": int(m_test["exceed_3sigma_n"]),
            }
            row.update({f"param_{k}": v for k, v in params.items() if k != "theta_E_scan"})
            rows.append(row)

    out_csv = out_dir / "condensed_silicon_thermal_expansion_gruneisen_holdout_splits.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        cols: list[str] = [
            "split",
            "model",
            "train_min_T_K",
            "train_max_T_K",
            "test_min_T_K",
            "test_max_T_K",
            "train_max_abs_z",
            "train_reduced_chi2",
            "test_max_abs_z",
            "test_reduced_chi2",
            "train_sign_mismatch_n",
            "test_sign_mismatch_n",
            "train_exceed_3sigma_n",
            "test_exceed_3sigma_n",
            "param_theta_E_K",
            "param_theta_E1_K",
            "param_theta_E2_K",
            "param_A_D_mol_per_J",
            "param_A_E_mol_per_J",
            "param_A_E1_mol_per_J",
            "param_A_E2_mol_per_J",
        ]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Plot: compare holdout severity (max abs z / reduced chi2)

    categories: list[str] = []
    maxz_train: list[float] = []
    maxz_test: list[float] = []
    chi2_train: list[float] = []
    chi2_test: list[float] = []
    for s in split_results:
        for model_key in ["einstein_1", "einstein_2"]:
            categories.append(f"{s['name']}-{model_key[-1]}")
            m_train = s["models"][model_key]["train"]
            m_test = s["models"][model_key]["test"]
            maxz_train.append(float(m_train["max_abs_z"]))
            maxz_test.append(float(m_test["max_abs_z"]))
            chi2_train.append(float(m_train["reduced_chi2"]))
            chi2_test.append(float(m_test["reduced_chi2"]))

    x = list(range(len(categories)))
    w_bar = 0.38
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.8, 6.6), sharex=True, gridspec_kw={"height_ratios": [1, 1]})

    ax1.bar([xi - w_bar / 2 for xi in x], maxz_train, width=w_bar, color="#1f77b4", alpha=0.85, label="train")
    ax1.bar([xi + w_bar / 2 for xi in x], maxz_test, width=w_bar, color="#ff7f0e", alpha=0.85, label="test")
    ax1.axhline(3.0, color="#666666", lw=1.0, ls="--", alpha=0.7)
    ax1.set_ylabel("max abs(z)")
    ax1.set_title("Si α(T): temperature-split holdout (Debye+Einstein)")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="upper right", fontsize=9)

    ax2.bar([xi - w_bar / 2 for xi in x], chi2_train, width=w_bar, color="#1f77b4", alpha=0.85, label="train")
    ax2.bar([xi + w_bar / 2 for xi in x], chi2_test, width=w_bar, color="#ff7f0e", alpha=0.85, label="test")
    ax2.axhline(2.0, color="#666666", lw=1.0, ls="--", alpha=0.7)
    ax2.set_ylabel("reduced χ² (proxy)")
    ax2.set_xlabel("split-model (A:50–300→300–600, B:200–600→50–200; 1=Einstein1, 2=Einstein2)")
    ax2.set_xticks(x, categories)
    ax2.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    out_png = out_dir / "condensed_silicon_thermal_expansion_gruneisen_holdout_splits.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    out_metrics = out_dir / "condensed_silicon_thermal_expansion_gruneisen_holdout_splits_metrics.json"
    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.14.15",
                "inputs": {
                    "silicon_thermal_expansion_extracted_values": {"path": str(alpha_src), "sha256": _sha256(alpha_src)},
                    "theta_d_source": theta_d_source,
                },
                "assumptions": {
                    "sigma_fit_proxy": {
                        "lt_T_K": float(t_sigma_split),
                        "lt_sigma_1e-8_per_K": float(sigma_lt_1e8),
                        "ge_sigma_1e-8_per_K": float(sigma_ge_1e8),
                        "note": "σ_fit is the NIST TRC curve-fit standard-error scale, treated here as a proxy for data-level accuracy.",
                    },
                    "holdout_reduced_chi2_definition": {
                        "train": "sum(z^2)/max(1,n-params)",
                        "test": "sum(z^2)/max(1,n)  (no parameters are fit on the test range)",
                    },
                },
                "data_range_T_K": [int(t_min), int(t_max)],
                "splits": split_results,
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

