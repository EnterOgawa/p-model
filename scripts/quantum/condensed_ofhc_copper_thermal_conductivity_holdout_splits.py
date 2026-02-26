from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_ROOT = _repo_root()


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(int(chunk_bytes))
            # 条件分岐: `not b` を満たす経路を評価する。
            if not b:
                break

            h.update(b)

    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _log10k_from_coeffs(t_k: np.ndarray, coeffs: dict[str, float]) -> np.ndarray:
    t = np.asarray(t_k, dtype=float)
    t12 = np.sqrt(t)
    t32 = t * t12
    t2 = t * t

    a = float(coeffs["a"])
    b = float(coeffs["b"])
    c = float(coeffs["c"])
    d = float(coeffs["d"])
    e = float(coeffs["e"])
    f = float(coeffs["f"])
    g = float(coeffs["g"])
    h = float(coeffs["h"])
    i = float(coeffs["i"])

    num = a + c * t12 + e * t + g * t32 + i * t2
    den = 1.0 + b * t12 + d * t + f * t32 + h * t2
    return num / den


def _k_from_coeffs(t_k: np.ndarray, coeffs: dict[str, float]) -> np.ndarray:
    return np.power(10.0, _log10k_from_coeffs(t_k, coeffs=coeffs))


def _metrics_for_idx(
    *,
    idx: np.ndarray,
    k_obs: np.ndarray,
    k_pred: np.ndarray,
    sigma_k: np.ndarray,
    param_count: int,
    is_train: bool,
) -> dict[str, float | int]:
    ii = np.asarray(idx, dtype=np.int64).reshape(-1)
    # 条件分岐: `ii.size == 0` を満たす経路を評価する。
    if ii.size == 0:
        return {"n": 0, "max_abs_z": float("nan"), "rms_z": float("nan"), "reduced_chi2": float("nan"), "exceed_3sigma_n": 0}

    z = (k_pred[ii] - k_obs[ii]) / np.maximum(1e-30, sigma_k[ii])
    z = z[np.isfinite(z)]
    # 条件分岐: `z.size == 0` を満たす経路を評価する。
    if z.size == 0:
        return {"n": 0, "max_abs_z": float("nan"), "rms_z": float("nan"), "reduced_chi2": float("nan"), "exceed_3sigma_n": 0}

    n = int(z.size)
    sum_z2 = float(np.sum(z * z))
    max_abs_z = float(np.max(np.abs(z)))
    rms_z = float(math.sqrt(sum_z2 / max(1, n)))
    exceed = int(np.sum(np.abs(z) > 3.0))
    dof = max(1, n - int(param_count)) if is_train else max(1, n)
    red_chi2 = float(sum_z2 / dof)
    return {
        "n": n,
        "max_abs_z": max_abs_z,
        "rms_z": rms_z,
        "reduced_chi2": red_chi2,
        "exceed_3sigma_n": exceed,
    }


def _fit_rational9_linearized(*, t_k: np.ndarray, y_log10k: np.ndarray, train_idx: np.ndarray) -> dict[str, float]:
    t = np.asarray(t_k, dtype=float).reshape(-1)
    y = np.asarray(y_log10k, dtype=float).reshape(-1)
    ii = np.asarray(train_idx, dtype=np.int64).reshape(-1)
    # 条件分岐: `ii.size < 60` を満たす経路を評価する。
    if ii.size < 60:
        raise RuntimeError(f"not enough points for rational9 fit (need >=60, got {int(ii.size)})")

    tt = t[ii]
    yy = y[ii]
    s = np.sqrt(tt)
    ts = tt * s
    t2 = tt * tt

    # Linearized form (unknown vector p = [a,c,e,g,i,b,d,f,h]):
    #   a + c*s + e*T + g*T*s + i*T^2 = y*(1 + b*s + d*T + f*T*s + h*T^2)
    # => [1,s,T,T*s,T^2, -y*s, -y*T, -y*T*s, -y*T^2] · p = y
    x = np.column_stack([np.ones_like(tt), s, tt, ts, t2, -yy * s, -yy * tt, -yy * ts, -yy * t2]).astype(float)
    rhs = yy.astype(float)
    coeff, *_ = np.linalg.lstsq(x, rhs, rcond=None)
    a, c, e, g, i, b, d, f, h = [float(v) for v in coeff.tolist()]
    return {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f, "g": g, "h": h, "i": i}


def _predict_rational9(*, t_k: np.ndarray, coeffs: dict[str, float]) -> np.ndarray:
    return _k_from_coeffs(np.asarray(t_k, dtype=float), coeffs=coeffs)


def _fit_logpoly2(*, t_k: np.ndarray, y_log10k: np.ndarray, train_idx: np.ndarray) -> dict[str, float]:
    t = np.asarray(t_k, dtype=float).reshape(-1)
    y = np.asarray(y_log10k, dtype=float).reshape(-1)
    ii = np.asarray(train_idx, dtype=np.int64).reshape(-1)
    # 条件分岐: `ii.size < 20` を満たす経路を評価する。
    if ii.size < 20:
        raise RuntimeError(f"not enough points for logpoly2 fit (need >=20, got {int(ii.size)})")

    u = np.log10(np.maximum(1e-30, t[ii]))
    x = np.column_stack([np.ones_like(u), u, u * u]).astype(float)
    coeff, *_ = np.linalg.lstsq(x, y[ii], rcond=None)
    a0, a1, a2 = [float(v) for v in coeff.tolist()]
    return {"a0": a0, "a1": a1, "a2": a2}


def _predict_logpoly2(*, t_k: np.ndarray, params: dict[str, float]) -> np.ndarray:
    t = np.asarray(t_k, dtype=float).reshape(-1)
    u = np.log10(np.maximum(1e-30, t))
    y = float(params["a0"]) + float(params["a1"]) * u + float(params["a2"]) * (u * u)
    return np.power(10.0, y)


def main() -> None:
    out_dir = _ROOT / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = _ROOT / "data" / "quantum" / "sources" / "nist_trc_ofhc_copper_thermal_conductivity"
    extracted_path = src_dir / "extracted_values.json"
    # 条件分岐: `not extracted_path.exists()` を満たす経路を評価する。
    if not extracted_path.exists():
        raise SystemExit(
            f"[fail] missing: {extracted_path}\n"
            "Run: python -B scripts/quantum/fetch_ofhc_copper_thermal_conductivity_sources.py"
        )

    extracted = _read_json(extracted_path)
    rrr_obj = extracted.get("rrr")
    # 条件分岐: `not isinstance(rrr_obj, dict) or not rrr_obj` を満たす経路を評価する。
    if not isinstance(rrr_obj, dict) or not rrr_obj:
        raise SystemExit(f"[fail] rrr missing/empty: {extracted_path}")

    want_rrr = [100, 300]
    have_rrr = sorted({int(k) for k in rrr_obj.keys() if str(k).isdigit()})
    rrr_values = [r for r in want_rrr if r in have_rrr]
    # 条件分岐: `not rrr_values` を満たす経路を評価する。
    if not rrr_values:
        rrr_values = have_rrr[:2]

    # Holdout split boundary (K) – chosen to separate the low-T peak region from high-T tail.

    t_split_k = 40.0

    # Common grid per RRR, log-spaced to resolve the low-T structure.
    n_grid = 420

    all_rows: list[dict[str, Any]] = []
    split_results_all: list[dict[str, Any]] = []

    for rrr in rrr_values:
        rr = rrr_obj.get(str(rrr))
        # 条件分岐: `not isinstance(rr, dict)` を満たす経路を評価する。
        if not isinstance(rr, dict):
            continue

        coeffs_true = rr.get("coefficients")
        # 条件分岐: `not isinstance(coeffs_true, dict)` を満たす経路を評価する。
        if not isinstance(coeffs_true, dict):
            raise SystemExit(f"[fail] missing coefficients for RRR={rrr}")

        t_min = float(rr.get("data_range", {}).get("t_min_k", 0.0))
        t_max = float(rr.get("data_range", {}).get("t_max_k", 0.0))
        # 条件分岐: `not (math.isfinite(t_min) and math.isfinite(t_max) and 0 < t_min < t_max)` を満たす経路を評価する。
        if not (math.isfinite(t_min) and math.isfinite(t_max) and 0 < t_min < t_max):
            raise SystemExit(f"[fail] invalid data range for RRR={rrr}: [{t_min}, {t_max}]")

        t_grid = np.geomspace(t_min, t_max, num=n_grid).astype(float)
        k_obs = _k_from_coeffs(t_grid, coeffs=coeffs_true).astype(float)
        y_obs = np.log10(np.maximum(1e-300, k_obs)).astype(float)

        err_pct = float(rr.get("curve_fit_error_percent_relative_to_data", 2.0))
        err_rel = max(0.001, float(err_pct) / 100.0)
        sigma_k = (err_rel * np.abs(k_obs)).astype(float)

        def _idx_range(t0: float, t1: float) -> np.ndarray:
            m = (t_grid >= float(t0)) & (t_grid <= float(t1))
            return np.nonzero(m)[0].astype(np.int64, copy=False)

        splits = [
            {
                "name": f"RRR{rrr}_A_low_to_high",
                "train_T_K": [float(t_min), float(min(t_split_k, t_max))],
                "test_T_K": [float(min(t_split_k, t_max)), float(t_max)],
            },
            {
                "name": f"RRR{rrr}_B_high_to_low",
                "train_T_K": [float(min(t_split_k, t_max)), float(t_max)],
                "test_T_K": [float(t_min), float(min(t_split_k, t_max))],
            },
        ]

        for sp in splits:
            name = str(sp["name"])
            train_r = [float(sp["train_T_K"][0]), float(sp["train_T_K"][1])]
            test_r = [float(sp["test_T_K"][0]), float(sp["test_T_K"][1])]
            idx_train = _idx_range(train_r[0], train_r[1])
            idx_test = _idx_range(test_r[0], test_r[1])

            models: dict[str, Any] = {}

            # Model 1: NIST rational (9 coefficients) refit by linearized LS.
            try:
                coeffs_fit = _fit_rational9_linearized(t_k=t_grid, y_log10k=y_obs, train_idx=idx_train)
                k_pred = _predict_rational9(t_k=t_grid, coeffs=coeffs_fit)
                m_tr = _metrics_for_idx(idx=idx_train, k_obs=k_obs, k_pred=k_pred, sigma_k=sigma_k, param_count=9, is_train=True)
                m_te = _metrics_for_idx(idx=idx_test, k_obs=k_obs, k_pred=k_pred, sigma_k=sigma_k, param_count=9, is_train=False)
                models["rational9_refit"] = {"params": coeffs_fit, "train": m_tr, "test": m_te}
            except Exception as exc:
                models["rational9_refit"] = {"supported": False, "reason": str(exc)}

            # Model 2: simple log-log quadratic (insufficient around the low-T peak).

            try:
                p2 = _fit_logpoly2(t_k=t_grid, y_log10k=y_obs, train_idx=idx_train)
                k2 = _predict_logpoly2(t_k=t_grid, params=p2)
                m2_tr = _metrics_for_idx(idx=idx_train, k_obs=k_obs, k_pred=k2, sigma_k=sigma_k, param_count=3, is_train=True)
                m2_te = _metrics_for_idx(idx=idx_test, k_obs=k_obs, k_pred=k2, sigma_k=sigma_k, param_count=3, is_train=False)
                models["logpoly2"] = {"params": p2, "train": m2_tr, "test": m2_te}
            except Exception as exc:
                models["logpoly2"] = {"supported": False, "reason": str(exc)}

            split_results_all.append(
                {
                    "name": name,
                    "train_T_K": train_r,
                    "test_T_K": test_r,
                    "meta": {"rrr": int(rrr), "fit_error_percent_relative_to_data": float(err_pct)},
                    "models": models,
                }
            )

            # CSV rows
            for mid, m in models.items():
                row: dict[str, Any] = {
                    "split": name,
                    "rrr": int(rrr),
                    "fit_error_percent_relative_to_data": float(err_pct),
                    "model_id": str(mid),
                    "train_min_T_K": float(train_r[0]),
                    "train_max_T_K": float(train_r[1]),
                    "test_min_T_K": float(test_r[0]),
                    "test_max_T_K": float(test_r[1]),
                }
                # 条件分岐: `not isinstance(m, dict) or m.get("supported") is False` を満たす経路を評価する。
                if not isinstance(m, dict) or m.get("supported") is False:
                    row["supported"] = False
                    row["reason"] = str(m.get("reason") or "")
                    all_rows.append(row)
                    continue

                tr = m.get("train") if isinstance(m.get("train"), dict) else {}
                te = m.get("test") if isinstance(m.get("test"), dict) else {}
                row.update(
                    {
                        "supported": True,
                        "train_n": int(tr.get("n", 0) or 0),
                        "train_max_abs_z": tr.get("max_abs_z"),
                        "train_reduced_chi2": tr.get("reduced_chi2"),
                        "test_n": int(te.get("n", 0) or 0),
                        "test_max_abs_z": te.get("max_abs_z"),
                        "test_reduced_chi2": te.get("reduced_chi2"),
                        "train_exceed_3sigma_n": int(tr.get("exceed_3sigma_n", 0) or 0),
                        "test_exceed_3sigma_n": int(te.get("exceed_3sigma_n", 0) or 0),
                    }
                )
                params = m.get("params") if isinstance(m.get("params"), dict) else {}
                row.update({f"param_{k}": v for k, v in params.items()})
                all_rows.append(row)

    out_csv = out_dir / "condensed_ofhc_copper_thermal_conductivity_holdout_splits.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "split",
            "rrr",
            "fit_error_percent_relative_to_data",
            "model_id",
            "train_min_T_K",
            "train_max_T_K",
            "test_min_T_K",
            "test_max_T_K",
            "supported",
            "reason",
            "train_n",
            "train_max_abs_z",
            "train_reduced_chi2",
            "test_n",
            "test_max_abs_z",
            "test_reduced_chi2",
            "train_exceed_3sigma_n",
            "test_exceed_3sigma_n",
            "param_a",
            "param_b",
            "param_c",
            "param_d",
            "param_e",
            "param_f",
            "param_g",
            "param_h",
            "param_i",
            "param_a0",
            "param_a1",
            "param_a2",
        ]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # Plot: per split, show test max abs(z) by model (grouped).

    try:
        splits_unique = [str(s.get("name")) for s in split_results_all if isinstance(s, dict) and s.get("name")]
        models_unique = sorted(
            {
                str(mid)
                for s in split_results_all
                if isinstance(s, dict)
                for mid in (s.get("models") or {}).keys()
                if isinstance(s.get("models"), dict)
            }
        )
        # 条件分岐: `splits_unique and len(models_unique) <= 6 and len(splits_unique) <= 10` を満たす経路を評価する。
        if splits_unique and len(models_unique) <= 6 and len(splits_unique) <= 10:
            x0 = np.arange(len(splits_unique), dtype=float)
            width = 0.8 / max(1, len(models_unique))
            fig, ax = plt.subplots(1, 1, figsize=(12.8, 4.8), dpi=160)
            for j, mid in enumerate(models_unique):
                ys = []
                for sp in split_results_all:
                    # 条件分岐: `not isinstance(sp, dict) or sp.get("name") not in splits_unique` を満たす経路を評価する。
                    if not isinstance(sp, dict) or sp.get("name") not in splits_unique:
                        continue

                    m = (sp.get("models") or {}).get(mid) if isinstance(sp.get("models"), dict) else None
                    v = (m.get("test") or {}).get("max_abs_z") if isinstance(m, dict) else None
                    ys.append(float(v) if isinstance(v, (int, float)) else float("nan"))

                xs = x0 + (j - (len(models_unique) - 1) / 2) * width
                ax.bar(xs, ys, width=width, alpha=0.85, label=mid)

            ax.axhline(3.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
            ax.set_xticks(x0)
            ax.set_xticklabels(splits_unique, rotation=0, ha="center")
            ax.set_ylabel("test max abs(z)")
            ax.set_title("OFHC Copper κ(T): temperature-split holdout (RRR subsets)")
            ax.grid(axis="y", linestyle=":", alpha=0.35)
            ax.legend(fontsize=9, loc="upper left", ncol=2)
            fig.tight_layout()
            out_png = out_dir / "condensed_ofhc_copper_thermal_conductivity_holdout_splits.png"
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
        else:
            out_png = out_dir / "condensed_ofhc_copper_thermal_conductivity_holdout_splits.png"
    except Exception:
        out_png = out_dir / "condensed_ofhc_copper_thermal_conductivity_holdout_splits.png"

    out_metrics = out_dir / "condensed_ofhc_copper_thermal_conductivity_holdout_splits_metrics.json"
    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.20.6",
                "inputs": {"extracted_values": {"path": str(extracted_path), "sha256": _sha256(extracted_path)}},
                "assumptions": {
                    "sigma_proxy": {"kappa": "NIST curve-fit relative error percent used as 1σ proxy (per RRR)"},
                    "holdout_reduced_chi2_definition": {
                        "train": "sum(z^2)/max(1,n-params)",
                        "test": "sum(z^2)/max(1,n)  (no parameters are fit on the test range)",
                    },
                    "models": {
                        "rational9_refit": "re-fit NIST rational form via linearized least squares (cross-multiplication)",
                        "logpoly2": "quadratic in log10(T) for log10(kappa); expected to degrade near low-T peak",
                    },
                },
                "splits": split_results_all,
                "outputs": {"csv": str(out_csv), "png": str(out_png) if out_png.exists() else None},
                "notes": [
                    "Holdout uses the NIST rational fit curve as the operational reference and tests procedural sensitivity of re-fitting on partial temperature bands.",
                    "This is a methodology audit; it does not claim new condensed-matter physics beyond the NIST baseline definition.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[ok] wrote: {out_csv}")
    # 条件分岐: `out_png.exists()` を満たす経路を評価する。
    if out_png.exists():
        print(f"[ok] wrote: {out_png}")

    print(f"[ok] wrote: {out_metrics}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()

