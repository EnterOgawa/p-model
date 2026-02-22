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


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    hash_obj = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rho_at(table: list[dict[str, Any]], t_c: float) -> Optional[float]:
    for row in table:
        if abs(float(row.get("T_C", -1.0)) - float(t_c)) < 1e-6:
            value = row.get("rho_ohm_cm")
            if isinstance(value, (int, float)) and float(value) > 0:
                return float(value)
    return None


def _coeff_ln_rho_per_k(*, rho20: float, rho30: float) -> float:
    return (math.log(rho30) - math.log(rho20)) / 10.0


def _metrics_for_idx(
    *,
    idx: np.ndarray,
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    sigma: float,
    param_count: int,
    is_train: bool,
) -> dict[str, float | int]:
    indices = np.asarray(idx, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return {"n": 0, "max_abs_z": float("nan"), "rms_z": float("nan"), "reduced_chi2": float("nan"), "exceed_3sigma_n": 0}
    z_scores = (y_pred[indices] - y_obs[indices]) / float(max(1e-12, sigma))
    z_scores = z_scores[np.isfinite(z_scores)]
    if z_scores.size == 0:
        return {"n": 0, "max_abs_z": float("nan"), "rms_z": float("nan"), "reduced_chi2": float("nan"), "exceed_3sigma_n": 0}
    n_points = int(z_scores.size)
    sum_z2 = float(np.sum(z_scores * z_scores))
    max_abs_z = float(np.max(np.abs(z_scores)))
    rms_z = float(math.sqrt(sum_z2 / max(1, n_points)))
    exceed = int(np.sum(np.abs(z_scores) > 3.0))
    dof = max(1, n_points - int(param_count)) if is_train else max(1, n_points)
    red_chi2 = float(sum_z2 / dof)
    return {
        "n": n_points,
        "max_abs_z": max_abs_z,
        "rms_z": rms_z,
        "reduced_chi2": red_chi2,
        "exceed_3sigma_n": exceed,
    }


def _robust_sigma(residuals: np.ndarray) -> float:
    values = np.asarray(residuals, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0:
        return float(1.4826 * mad)
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return float(std if std > 0 else 1.0)


def _global_sigma_floor(*, x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, dict[str, float | str]]:
    indices = np.arange(np.asarray(y_values, dtype=float).reshape(-1).size, dtype=np.int64)
    intercept, slope = _fit_linear(x_values, y_values, indices)
    residuals = np.asarray(y_values, dtype=float).reshape(-1) - (float(intercept) + float(slope) * np.asarray(x_values, dtype=float).reshape(-1))
    sigma_mad = _robust_sigma(residuals)
    sigma_std = float(np.std(residuals, ddof=1)) if residuals.size > 1 else sigma_mad
    sigma_floor = float(max(1e-6, sigma_mad, sigma_std))
    return sigma_floor, {
        "reference_model": "global_linear_logrho_2p",
        "sigma_mad_pct_per_K": float(sigma_mad),
        "sigma_std_pct_per_K": float(sigma_std),
        "sigma_floor_pct_per_K": float(sigma_floor),
    }


def _fit_const(values: np.ndarray, idx: np.ndarray) -> float:
    indices = np.asarray(idx, dtype=np.int64).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)[indices]
    return float(np.mean(values))


def _fit_linear(x_values: np.ndarray, y_values: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    indices = np.asarray(idx, dtype=np.int64).reshape(-1)
    x_vals = np.asarray(x_values, dtype=float).reshape(-1)[indices]
    y_vals = np.asarray(y_values, dtype=float).reshape(-1)[indices]
    design = np.column_stack([np.ones_like(x_vals), x_vals]).astype(float)
    try:
        beta, *_ = np.linalg.lstsq(design, y_vals, rcond=None)
    except Exception:
        beta = np.linalg.pinv(design) @ y_vals
    intercept = float(beta[0])
    slope = float(beta[1])
    return intercept, slope


def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = root / "data" / "quantum" / "sources" / "nist_nbsir74_496_silicon_resistivity"
    extracted_path = src_dir / "extracted_values.json"
    if not extracted_path.exists():
        raise SystemExit(
            f"[fail] missing: {extracted_path}\n"
            "Run: python -B scripts/quantum/fetch_silicon_resistivity_nbsir74_496_sources.py"
        )

    extracted = _read_json(extracted_path)
    samples = extracted.get("samples")
    if not isinstance(samples, list) or not samples:
        raise SystemExit(f"[fail] samples missing/empty: {extracted_path}")

    rows: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("sample_id") or "")
        sample_type = sample.get("type")
        doping = sample.get("doping")
        table = sample.get("rho_range_table")
        if not sample_id or not isinstance(table, list) or not table:
            continue
        if sample_type not in ("p", "n"):
            continue
        rho20 = _rho_at(table, 20.0)
        rho30 = _rho_at(table, 30.0)
        rho23 = _rho_at(table, 23.0)
        if rho20 is None or rho30 is None or rho23 is None:
            continue
        coeff = _coeff_ln_rho_per_k(rho20=rho20, rho30=rho30)
        rows.append(
            {
                "sample_id": sample_id,
                "type": sample_type,
                "doping": "" if doping is None else str(doping),
                "rho_23C_ohm_cm": float(rho23),
                "dlnrho_dT_23C_per_K_proxy": float(coeff),
                "pct_per_K_proxy": float(100.0 * coeff),
            }
        )

    if not rows:
        raise SystemExit("[fail] no usable samples with {20,23,30}°C mean rho extracted")

    log10_rho_23c = np.asarray([math.log10(float(r["rho_23C_ohm_cm"])) for r in rows], dtype=float)
    coeff_pct_per_k = np.asarray([float(r["pct_per_K_proxy"]) for r in rows], dtype=float)
    sigma_floor_pct_per_k, sigma_floor_meta = _global_sigma_floor(x_values=log10_rho_23c, y_values=coeff_pct_per_k)

    log10_median = float(np.median(log10_rho_23c))
    idx_low = np.where(log10_rho_23c <= log10_median)[0].astype(np.int64)
    idx_high = np.where(log10_rho_23c > log10_median)[0].astype(np.int64)
    if idx_low.size == 0 or idx_high.size == 0:
        raise SystemExit("[fail] split by median log10 rho produced empty train/test")

    splits = [
        {"name": "A_low_to_high", "train_idx": idx_low, "test_idx": idx_high},
        {"name": "B_high_to_low", "train_idx": idx_high, "test_idx": idx_low},
    ]

    rows_out: list[dict[str, Any]] = []
    split_results: list[dict[str, Any]] = []

    for split in splits:
        split_name = str(split["name"])
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]

        models: dict[str, Any] = {}

        mean_value = _fit_const(coeff_pct_per_k, train_idx)
        coeff_pred_const = np.full_like(coeff_pct_per_k, float(mean_value), dtype=float)
        sigma_const_raw = _robust_sigma(coeff_pct_per_k[train_idx] - coeff_pred_const[train_idx])
        sigma_const = float(max(float(sigma_const_raw), float(sigma_floor_pct_per_k)))
        m0_train = _metrics_for_idx(
            idx=train_idx,
            y_obs=coeff_pct_per_k,
            y_pred=coeff_pred_const,
            sigma=sigma_const,
            param_count=1,
            is_train=True,
        )
        m0_test = _metrics_for_idx(
            idx=test_idx,
            y_obs=coeff_pct_per_k,
            y_pred=coeff_pred_const,
            sigma=sigma_const,
            param_count=1,
            is_train=False,
        )
        models["const_mean_fit"] = {
            "params": {
                "mean_pct_per_K": float(mean_value),
                "sigma_train_proxy_pct_per_K": float(sigma_const_raw),
                "sigma_floor_pct_per_K": float(sigma_floor_pct_per_k),
                "sigma_proxy_pct_per_K": float(sigma_const),
            },
            "train": m0_train,
            "test": m0_test,
        }

        intercept, slope = _fit_linear(log10_rho_23c, coeff_pct_per_k, train_idx)
        coeff_pred_linear = intercept + slope * log10_rho_23c
        sigma_linear_raw = _robust_sigma(coeff_pct_per_k[train_idx] - coeff_pred_linear[train_idx])
        sigma_linear = float(max(float(sigma_linear_raw), float(sigma_floor_pct_per_k)))
        m1_train = _metrics_for_idx(
            idx=train_idx,
            y_obs=coeff_pct_per_k,
            y_pred=coeff_pred_linear,
            sigma=sigma_linear,
            param_count=2,
            is_train=True,
        )
        m1_test = _metrics_for_idx(
            idx=test_idx,
            y_obs=coeff_pct_per_k,
            y_pred=coeff_pred_linear,
            sigma=sigma_linear,
            param_count=2,
            is_train=False,
        )
        models["linear_logrho_2p"] = {
            "params": {
                "intercept_pct_per_K": float(intercept),
                "slope_pct_per_K_per_log10rho": float(slope),
                "sigma_train_proxy_pct_per_K": float(sigma_linear_raw),
                "sigma_floor_pct_per_K": float(sigma_floor_pct_per_k),
                "sigma_proxy_pct_per_K": float(sigma_linear),
            },
            "train": m1_train,
            "test": m1_test,
        }

        for model_id, model in models.items():
            train = model.get("train") if isinstance(model.get("train"), dict) else {}
            test = model.get("test") if isinstance(model.get("test"), dict) else {}
            params = model.get("params") if isinstance(model.get("params"), dict) else {}
            rows_out.append(
                {
                    "split": split_name,
                    "model_id": model_id,
                    "train_min_log10rho": float(np.min(log10_rho_23c[train_idx])),
                    "train_max_log10rho": float(np.max(log10_rho_23c[train_idx])),
                    "test_min_log10rho": float(np.min(log10_rho_23c[test_idx])),
                    "test_max_log10rho": float(np.max(log10_rho_23c[test_idx])),
                    "supported": True,
                    "reason": "",
                    "train_n": int(train.get("n", 0) or 0),
                    "train_max_abs_z": train.get("max_abs_z"),
                    "train_reduced_chi2": train.get("reduced_chi2"),
                    "test_n": int(test.get("n", 0) or 0),
                    "test_max_abs_z": test.get("max_abs_z"),
                    "test_reduced_chi2": test.get("reduced_chi2"),
                    "train_exceed_3sigma_n": int(train.get("exceed_3sigma_n", 0) or 0),
                    "test_exceed_3sigma_n": int(test.get("exceed_3sigma_n", 0) or 0),
                    **{f"param_{k}": v for k, v in params.items()},
                }
            )

        split_results.append(
            {
                "name": split_name,
                "train_log10rho": [float(np.min(log10_rho_23c[train_idx])), float(np.max(log10_rho_23c[train_idx]))],
                "test_log10rho": [float(np.min(log10_rho_23c[test_idx])), float(np.max(log10_rho_23c[test_idx]))],
                "models": models,
            }
        )

    out_csv = out_dir / "condensed_silicon_resistivity_temperature_coefficient_holdout_splits.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "split",
            "model_id",
            "train_min_log10rho",
            "train_max_log10rho",
            "test_min_log10rho",
            "test_max_log10rho",
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
            "param_mean_pct_per_K",
            "param_sigma_train_proxy_pct_per_K",
            "param_sigma_floor_pct_per_K",
            "param_sigma_proxy_pct_per_K",
            "param_intercept_pct_per_K",
            "param_slope_pct_per_K_per_log10rho",
        ]
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.2), dpi=170, gridspec_kw={"height_ratios": [2.0, 1.0]})
    ax0 = axes[0]

    def pick_style(sample_type: str, doping_label: str) -> tuple[str, str]:
        if sample_type == "p" and doping_label == "Al":
            return ("#d62728", "p-type (Al)")
        if sample_type == "p":
            return ("#ff7f0e", "p-type (B)")
        return ("#1f77b4", "n-type")

    series: dict[str, dict[str, Any]] = {}
    for row in rows:
        color, label = pick_style(str(row["type"]), str(row.get("doping") or ""))
        if label not in series:
            series[label] = {"color": color, "xs": [], "ys": []}
        series[label]["xs"].append(math.log10(float(row["rho_23C_ohm_cm"])))
        series[label]["ys"].append(float(row["pct_per_K_proxy"]))

    for label, series_entry in series.items():
        ax0.scatter(series_entry["xs"], series_entry["ys"], s=24, alpha=0.85, label=label, color=series_entry["color"])

    ax0.axvline(log10_median, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax0.set_xlabel("log10 ρ(23°C) (Ω·cm)")
    ax0.set_ylabel("d ln ρ / dT @ 23°C ( % / K )")
    ax0.set_title("Si resistivity temperature coefficient: split-by-log10ρ holdout (NBS IR 74-496)")
    ax0.grid(True, linestyle=":", alpha=0.35)
    ax0.legend(loc="best", fontsize=9)

    ax1 = axes[1]
    categories = [str(sp["name"]) for sp in split_results]
    y_const = []
    y_linear = []
    for sp in split_results:
        models = sp.get("models") if isinstance(sp.get("models"), dict) else {}
        m_const = models.get("const_mean_fit") if isinstance(models, dict) else None
        m_linear = models.get("linear_logrho_2p") if isinstance(models, dict) else None
        test_max_const = (m_const.get("test") or {}).get("max_abs_z") if isinstance(m_const, dict) else None
        test_max_linear = (m_linear.get("test") or {}).get("max_abs_z") if isinstance(m_linear, dict) else None
        y_const.append(float(test_max_const) if isinstance(test_max_const, (int, float)) else float("nan"))
        y_linear.append(float(test_max_linear) if isinstance(test_max_linear, (int, float)) else float("nan"))

    bar_positions = np.arange(len(categories), dtype=float)
    bar_width = 0.32
    ax1.bar(bar_positions - bar_width / 2, y_const, width=bar_width, color="#1f77b4", alpha=0.85, label="Const fit (1p)")
    ax1.bar(bar_positions + bar_width / 2, y_linear, width=bar_width, color="#ff7f0e", alpha=0.85, label="Linear fit (2p)")
    ax1.axhline(3.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(categories)
    ax1.set_ylabel("test max abs(z)")
    ax1.grid(axis="y", linestyle=":", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    out_png = out_dir / "condensed_silicon_resistivity_temperature_coefficient_holdout_splits.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    out_metrics = out_dir / "condensed_silicon_resistivity_temperature_coefficient_holdout_splits_metrics.json"
    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.20.6",
                "inputs": {"extracted_values": {"path": str(extracted_path), "sha256": _sha256(extracted_path)}},
                "assumptions": {
                    "sigma_proxy": "max(train residual robust sigma, global heterogeneity floor from all-sample linear baseline)",
                    "holdout_reduced_chi2_definition": {
                        "train": "sum(z^2)/max(1,n-params)",
                        "test": "sum(z^2)/max(1,n)  (no parameters are fit on the test split)",
                    },
                    "sigma_floor_reference": sigma_floor_meta,
                },
                "data_range_log10rho": [float(np.min(log10_rho_23c)), float(np.max(log10_rho_23c))],
                "splits": split_results,
                "outputs": {"csv": str(out_csv), "png": str(out_png)},
                "notes": [
                    "Operational holdout: split samples by median log10 ρ(23°C) and test whether the coefficient trend transfers across resistivity regimes.",
                    "The temperature coefficient is a near-room-temperature proxy (20–30°C finite difference) and does not encode a full ρ(T) curve.",
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
