from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"expected number, got: {type(value)}")


def _get_constant(extracted: dict[str, Any], code: str) -> dict[str, Any]:
    constants = extracted.get("constants")
    if not isinstance(constants, dict) or code not in constants or not isinstance(constants.get(code), dict):
        raise KeyError(f"missing constant: {code}")
    return constants[code]


def _metrics_for_idx(
    *,
    idx: np.ndarray,
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    param_count: int,
    is_train: bool,
) -> dict[str, float | int]:
    indices = np.asarray(idx, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return {"n": 0, "max_abs_z": float("nan"), "rms_z": float("nan"), "reduced_chi2": float("nan"), "exceed_3sigma_n": 0}
    z_scores = (y_pred[indices] - y_obs[indices]) / np.maximum(1e-30, sigma[indices])
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


def _fit_log_powerlaw(x_values: np.ndarray, y_values: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    indices = np.asarray(idx, dtype=np.int64).reshape(-1)
    x_vals = np.asarray(x_values, dtype=float).reshape(-1)[indices]
    y_vals = np.asarray(y_values, dtype=float).reshape(-1)[indices]
    log_x = np.log10(np.maximum(1e-300, x_vals))
    log_y = np.log10(np.maximum(1e-300, y_vals))
    design = np.column_stack([np.ones_like(log_x), log_x]).astype(float)
    beta, *_ = np.linalg.lstsq(design, log_y, rcond=None)
    log10_coeff = float(beta[0])
    exponent = float(beta[1])
    return log10_coeff, exponent


def _fit_fixed_exponent(x_values: np.ndarray, y_values: np.ndarray, idx: np.ndarray, exponent: float) -> float:
    indices = np.asarray(idx, dtype=np.int64).reshape(-1)
    x_vals = np.asarray(x_values, dtype=float).reshape(-1)[indices]
    y_vals = np.asarray(y_values, dtype=float).reshape(-1)[indices]
    denom = np.maximum(1e-300, x_vals**float(exponent))
    return float(np.mean(y_vals / denom))


def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = root / "data" / "quantum" / "sources" / "nist_codata_2022_blackbody_constants"
    extracted_path = src_dir / "extracted_values.json"
    if not extracted_path.exists():
        raise SystemExit(
            f"[fail] missing: {extracted_path}\n"
            "Run: python -B scripts/quantum/fetch_blackbody_constants_sources.py"
        )

    extracted = _read_json(extracted_path)
    speed_of_light = _as_float(_get_constant(extracted, "c").get("value_si"))
    planck_h = _as_float(_get_constant(extracted, "h").get("value_si"))
    boltzmann_k = _as_float(_get_constant(extracted, "k").get("value_si"))

    stefan_boltzmann_sigma = (2.0 * (math.pi**5) * (boltzmann_k**4)) / (15.0 * (planck_h**3) * (speed_of_light**2))
    radiation_constant_a = 4.0 * stefan_boltzmann_sigma / speed_of_light
    heat_capacity_coeff = 4.0 * radiation_constant_a

    temperature_grid_k = np.logspace(0.0, 4.0, 80)  # 1..1e4 K
    heat_capacity_density = heat_capacity_coeff * (temperature_grid_k**3)

    heat_capacity_sigma = np.maximum(1e-12, 1e-6 * heat_capacity_density)

    split_index = int(len(temperature_grid_k) / 2)
    idx_low = np.arange(0, split_index, dtype=np.int64)
    idx_high = np.arange(split_index, len(temperature_grid_k), dtype=np.int64)

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

        log10_coeff, exponent = _fit_log_powerlaw(temperature_grid_k, heat_capacity_density, train_idx)
        heat_capacity_pred = (10 ** log10_coeff) * (temperature_grid_k**exponent)
        m1_train = _metrics_for_idx(
            idx=train_idx,
            y_obs=heat_capacity_density,
            y_pred=heat_capacity_pred,
            sigma=heat_capacity_sigma,
            param_count=2,
            is_train=True,
        )
        m1_test = _metrics_for_idx(
            idx=test_idx,
            y_obs=heat_capacity_density,
            y_pred=heat_capacity_pred,
            sigma=heat_capacity_sigma,
            param_count=2,
            is_train=False,
        )
        models["powerlaw_fit_2p"] = {
            "params": {"log10_coeff": float(log10_coeff), "exponent": float(exponent)},
            "train": m1_train,
            "test": m1_test,
        }

        coeff_fixed = _fit_fixed_exponent(temperature_grid_k, heat_capacity_density, train_idx, exponent=3.0)
        heat_capacity_pred_fixed = coeff_fixed * (temperature_grid_k**3.0)
        m2_train = _metrics_for_idx(
            idx=train_idx,
            y_obs=heat_capacity_density,
            y_pred=heat_capacity_pred_fixed,
            sigma=heat_capacity_sigma,
            param_count=1,
            is_train=True,
        )
        m2_test = _metrics_for_idx(
            idx=test_idx,
            y_obs=heat_capacity_density,
            y_pred=heat_capacity_pred_fixed,
            sigma=heat_capacity_sigma,
            param_count=1,
            is_train=False,
        )
        models["fixed_exponent_3_fit"] = {
            "params": {"coeff": float(coeff_fixed), "exponent": 3.0},
            "train": m2_train,
            "test": m2_test,
        }

        for model_id, model in models.items():
            train = model.get("train") if isinstance(model.get("train"), dict) else {}
            test = model.get("test") if isinstance(model.get("test"), dict) else {}
            params = model.get("params") if isinstance(model.get("params"), dict) else {}
            rows_out.append(
                {
                    "split": split_name,
                    "model_id": model_id,
                    "train_min_T_K": float(np.min(temperature_grid_k[train_idx])),
                    "train_max_T_K": float(np.max(temperature_grid_k[train_idx])),
                    "test_min_T_K": float(np.min(temperature_grid_k[test_idx])),
                    "test_max_T_K": float(np.max(temperature_grid_k[test_idx])),
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
                "train_T_K": [float(np.min(temperature_grid_k[train_idx])), float(np.max(temperature_grid_k[train_idx]))],
                "test_T_K": [float(np.min(temperature_grid_k[test_idx])), float(np.max(temperature_grid_k[test_idx]))],
                "models": models,
            }
        )

    out_csv = out_dir / "thermo_blackbody_heat_capacity_density_holdout_splits.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "split",
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
            "param_log10_coeff",
            "param_exponent",
            "param_coeff",
        ]
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    categories = [str(sp["name"]) for sp in split_results]
    y_power = []
    y_fixed = []
    for sp in split_results:
        models = sp.get("models") if isinstance(sp.get("models"), dict) else {}
        model_power = models.get("powerlaw_fit_2p") if isinstance(models, dict) else None
        model_fixed = models.get("fixed_exponent_3_fit") if isinstance(models, dict) else None
        test_power = (model_power.get("test") or {}).get("max_abs_z") if isinstance(model_power, dict) else None
        test_fixed = (model_fixed.get("test") or {}).get("max_abs_z") if isinstance(model_fixed, dict) else None
        y_power.append(float(test_power) if isinstance(test_power, (int, float)) else float("nan"))
        y_fixed.append(float(test_fixed) if isinstance(test_fixed, (int, float)) else float("nan"))

    bar_positions = np.arange(len(categories), dtype=float)
    bar_width = 0.3
    fig, ax = plt.subplots(1, 1, figsize=(10.6, 4.2), dpi=170)
    ax.bar(bar_positions - bar_width / 2, y_power, width=bar_width, color="#1f77b4", alpha=0.85, label="power-law fit (2p)")
    ax.bar(bar_positions + bar_width / 2, y_fixed, width=bar_width, color="#ff7f0e", alpha=0.85, label="fixed exponent 3 (1p)")
    ax.axhline(3.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(categories)
    ax.set_ylabel("test max abs(z)")
    ax.set_title("Blackbody heat capacity density holdout: c_v = 4 a T^3")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_png = out_dir / "thermo_blackbody_heat_capacity_density_holdout_splits.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    out_metrics = out_dir / "thermo_blackbody_heat_capacity_density_holdout_splits_metrics.json"
    out_metrics.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.20.6",
                "inputs": {"blackbody_constants_extracted_values": {"path": str(extracted_path), "sha256": _sha256(extracted_path)}},
                "constants": {
                    "c_m_per_s": speed_of_light,
                    "h_J_s": planck_h,
                    "kB_J_per_K": boltzmann_k,
                    "radiation_constant_a": radiation_constant_a,
                },
                "assumptions": {
                    "sigma_proxy": "1 ppm relative tolerance (operational, analytic baseline)",
                    "holdout_reduced_chi2_definition": {
                        "train": "sum(z^2)/max(1,n-params)",
                        "test": "sum(z^2)/max(1,n)  (no parameters are fit on the test split)",
                    },
                },
                "data_range_T_K": [float(np.min(temperature_grid_k)), float(np.max(temperature_grid_k))],
                "splits": split_results,
                "outputs": {"csv": str(out_csv), "png": str(out_png)},
                "notes": [
                    "Analytic holdout for blackbody heat capacity density c_v = du/dT = 4 a T^3.",
                    "This is a procedural regression check, not an observational inference.",
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
