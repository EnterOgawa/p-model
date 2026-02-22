#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparc_rotation_curve_pmodel_audit.py

Phase 8 / Step 8.7（Part II 強化）:
SPARC（175銀河）の回転曲線について、
V_obs(R) を baryon-only と P-model 補正付きで比較する最小監査を固定する。

目的：
- 観測回転速度 V_obs が baryon-only（V_bar）で説明できるか、
  あるいは P 場由来の有効重力増強を加えた V_P で改善するかを、
  同一I/Fで定量比較する。
- 自由パラメータは M/L（Υ）1つに制限し、χ²で評価する。

入力：
- data/cosmology/sparc/raw/Rotmod_LTG.zip
- output/private/cosmology/cosmology_redshift_pbg_metrics.json
  （無い場合は output/public/cosmology/cosmology_redshift_pbg_metrics.json を参照）

出力（固定）：
- output/public/cosmology/sparc_rotation_curve_pmodel_audit_points.csv
- output/public/cosmology/sparc_rotation_curve_pmodel_audit_galaxy_summary.csv
- output/public/cosmology/sparc_rotation_curve_pmodel_audit.png
- output/public/cosmology/sparc_rotation_curve_pmodel_audit_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from scripts.summary import worklog  # type: ignore # noqa: E402
except Exception:  # pragma: no cover
    worklog = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

KPC_TO_M = 3.0856775814913673e19
KM_TO_M = 1.0e3
C_LIGHT_M_S = 299_792_458.0
MPC_TO_M = 3.0856775814913673e22
DEFAULT_PBG_KAPPA = 1.0 / (2.0 * math.pi)
DISTANCE_RE = re.compile(r"^#\s*Distance\s*=\s*(?P<d>[0-9.+-Ee]+)\s*Mpc\s*$")


@dataclass(frozen=True)
class RotmodPoint:
    galaxy: str
    radius_kpc: float
    velocity_obs_km_s: float
    velocity_obs_sigma_km_s: float
    velocity_gas_km_s: float
    velocity_disk_unit_km_s: float
    velocity_bulge_unit_km_s: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_rotmod_lines(lines: Iterable[str]) -> List[RotmodPoint]:
    points: List[RotmodPoint] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        _ = DISTANCE_RE.match(line)
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            radius_kpc = float(parts[0])
            velocity_obs_km_s = float(parts[1])
            velocity_obs_sigma_km_s = float(parts[2])
            velocity_gas_km_s = float(parts[3])
            velocity_disk_unit_km_s = float(parts[4])
            velocity_bulge_unit_km_s = float(parts[5])
        except Exception:
            continue
        if (
            not np.isfinite(radius_kpc)
            or radius_kpc <= 0.0
            or not np.isfinite(velocity_obs_km_s)
            or velocity_obs_km_s <= 0.0
            or not np.isfinite(velocity_gas_km_s)
            or not np.isfinite(velocity_disk_unit_km_s)
            or not np.isfinite(velocity_bulge_unit_km_s)
        ):
            continue
        points.append(
            RotmodPoint(
                galaxy="",
                radius_kpc=float(radius_kpc),
                velocity_obs_km_s=float(velocity_obs_km_s),
                velocity_obs_sigma_km_s=float(velocity_obs_sigma_km_s),
                velocity_gas_km_s=float(velocity_gas_km_s),
                velocity_disk_unit_km_s=float(velocity_disk_unit_km_s),
                velocity_bulge_unit_km_s=float(velocity_bulge_unit_km_s),
            )
        )
    return points


def _load_rotmod_points(rotmod_zip: Path) -> List[RotmodPoint]:
    if not rotmod_zip.exists():
        raise FileNotFoundError(f"missing Rotmod zip: {rotmod_zip}")

    all_points: List[RotmodPoint] = []
    with zipfile.ZipFile(rotmod_zip, "r") as zip_file:
        names = [name for name in zip_file.namelist() if name.lower().endswith("_rotmod.dat")]
        names.sort()
        for name in names:
            galaxy = Path(name).name.replace("_rotmod.dat", "")
            text_lines = zip_file.read(name).decode("utf-8", errors="replace").splitlines()
            parsed = _parse_rotmod_lines(text_lines)
            for point in parsed:
                all_points.append(
                    RotmodPoint(
                        galaxy=galaxy,
                        radius_kpc=point.radius_kpc,
                        velocity_obs_km_s=point.velocity_obs_km_s,
                        velocity_obs_sigma_km_s=point.velocity_obs_sigma_km_s,
                        velocity_gas_km_s=point.velocity_gas_km_s,
                        velocity_disk_unit_km_s=point.velocity_disk_unit_km_s,
                        velocity_bulge_unit_km_s=point.velocity_bulge_unit_km_s,
                    )
                )
    return all_points


def _h0p_from_metrics(metrics_path: Path) -> float:
    payload = _read_json(metrics_path)
    derived = payload.get("derived") if isinstance(payload.get("derived"), dict) else {}
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    h0_si = derived.get("H0P_SI_s^-1")
    if isinstance(h0_si, (int, float)) and np.isfinite(h0_si) and float(h0_si) > 0.0:
        return float(h0_si)
    h0_km_s_mpc = params.get("H0P_km_s_Mpc")
    if isinstance(h0_km_s_mpc, (int, float)) and np.isfinite(h0_km_s_mpc) and float(h0_km_s_mpc) > 0.0:
        return float(h0_km_s_mpc) * 1.0e3 / MPC_TO_M
    raise RuntimeError(f"failed to read H0^(P) from: {metrics_path}")


def _arrays_from_points(points: Sequence[RotmodPoint]) -> Dict[str, np.ndarray]:
    return {
        "radius_m": np.asarray([point.radius_kpc for point in points], dtype=float) * KPC_TO_M,
        "velocity_obs_km_s": np.asarray([point.velocity_obs_km_s for point in points], dtype=float),
        "velocity_obs_sigma_km_s": np.asarray([point.velocity_obs_sigma_km_s for point in points], dtype=float),
        "velocity_gas_m_s": np.asarray([point.velocity_gas_km_s for point in points], dtype=float) * KM_TO_M,
        "velocity_disk_unit_m_s": np.asarray([point.velocity_disk_unit_km_s for point in points], dtype=float) * KM_TO_M,
        "velocity_bulge_unit_m_s": np.asarray([point.velocity_bulge_unit_km_s for point in points], dtype=float) * KM_TO_M,
    }


def _velocity_model(
    arrays: Dict[str, np.ndarray],
    *,
    upsilon: float,
    bulge_to_disk_ratio: float,
    acceleration_floor_m_s2: float,
    a0_m_s2: float,
    mode: str,
) -> Dict[str, np.ndarray]:
    upsilon_disk = max(0.0, float(upsilon))
    upsilon_bulge = max(0.0, float(upsilon) * float(bulge_to_disk_ratio))

    velocity_gas_m_s = arrays["velocity_gas_m_s"]
    velocity_disk_unit_m_s = arrays["velocity_disk_unit_m_s"]
    velocity_bulge_unit_m_s = arrays["velocity_bulge_unit_m_s"]
    radius_m = arrays["radius_m"]

    velocity_bar_sq_m2_s2 = (
        velocity_gas_m_s * velocity_gas_m_s
        + upsilon_disk * velocity_disk_unit_m_s * velocity_disk_unit_m_s
        + upsilon_bulge * velocity_bulge_unit_m_s * velocity_bulge_unit_m_s
    )
    velocity_bar_sq_m2_s2 = np.maximum(velocity_bar_sq_m2_s2, 0.0)
    acceleration_bar_m_s2 = velocity_bar_sq_m2_s2 / np.maximum(radius_m, acceleration_floor_m_s2)
    acceleration_bar_m_s2 = np.maximum(acceleration_bar_m_s2, acceleration_floor_m_s2)

    if mode == "baryon_only":
        acceleration_pred_m_s2 = acceleration_bar_m_s2
    elif mode == "pmodel_boost":
        x_ratio = acceleration_bar_m_s2 / max(float(a0_m_s2), acceleration_floor_m_s2)
        x_ratio = np.maximum(x_ratio, 1e-30)
        denominator = -np.expm1(-np.sqrt(x_ratio))
        denominator = np.maximum(denominator, 1e-30)
        acceleration_pred_m_s2 = acceleration_bar_m_s2 / denominator
    else:
        raise ValueError(f"unknown mode: {mode}")

    velocity_pred_m_s = np.sqrt(np.maximum(acceleration_pred_m_s2 * radius_m, 0.0))
    velocity_bar_m_s = np.sqrt(velocity_bar_sq_m2_s2)
    return {
        "velocity_bar_km_s": velocity_bar_m_s / KM_TO_M,
        "velocity_pred_km_s": velocity_pred_m_s / KM_TO_M,
        "acceleration_bar_m_s2": acceleration_bar_m_s2,
        "acceleration_pred_m_s2": acceleration_pred_m_s2,
    }


def _chi2_metrics(
    velocity_obs_km_s: np.ndarray,
    velocity_obs_sigma_km_s: np.ndarray,
    velocity_pred_km_s: np.ndarray,
    *,
    sigma_floor_km_s: float,
    fit_params: int,
) -> Dict[str, float]:
    sigma_km_s = np.maximum(np.maximum(velocity_obs_sigma_km_s, 0.0), float(sigma_floor_km_s))
    residual_km_s = velocity_obs_km_s - velocity_pred_km_s
    pull_sigma = residual_km_s / sigma_km_s
    chi2 = float(np.sum(pull_sigma * pull_sigma))
    n_points = int(velocity_obs_km_s.size)
    dof = max(1, n_points - int(fit_params))
    return {
        "chi2": chi2,
        "dof": float(dof),
        "chi2_dof": float(chi2 / float(dof)),
        "rms_residual_km_s": float(np.sqrt(np.mean(residual_km_s * residual_km_s))),
        "median_abs_residual_km_s": float(np.median(np.abs(residual_km_s))),
        "max_abs_pull": float(np.max(np.abs(pull_sigma))),
    }


def _fit_upsilon(
    arrays: Dict[str, np.ndarray],
    *,
    mode: str,
    upsilon_grid: np.ndarray,
    bulge_to_disk_ratio: float,
    a0_m_s2: float,
    sigma_floor_km_s: float,
    acceleration_floor_m_s2: float,
) -> Dict[str, Any]:
    velocity_obs_km_s = arrays["velocity_obs_km_s"]
    velocity_obs_sigma_km_s = arrays["velocity_obs_sigma_km_s"]

    chi2_list: List[float] = []
    metrics_by_grid: List[Dict[str, np.ndarray]] = []
    for upsilon in upsilon_grid.tolist():
        modeled = _velocity_model(
            arrays,
            upsilon=float(upsilon),
            bulge_to_disk_ratio=float(bulge_to_disk_ratio),
            acceleration_floor_m_s2=float(acceleration_floor_m_s2),
            a0_m_s2=float(a0_m_s2),
            mode=mode,
        )
        metrics = _chi2_metrics(
            velocity_obs_km_s,
            velocity_obs_sigma_km_s,
            modeled["velocity_pred_km_s"],
            sigma_floor_km_s=float(sigma_floor_km_s),
            fit_params=1,
        )
        chi2_list.append(float(metrics["chi2"]))
        metrics_by_grid.append(modeled)

    chi2_arr = np.asarray(chi2_list, dtype=float)
    best_idx = int(np.argmin(chi2_arr))
    best_upsilon = float(upsilon_grid[best_idx])
    chi2_min = float(chi2_arr[best_idx])
    delta = chi2_arr - chi2_min
    one_sigma_mask = delta <= 1.0
    if np.any(one_sigma_mask):
        upsilon_lo = float(upsilon_grid[np.where(one_sigma_mask)[0][0]])
        upsilon_hi = float(upsilon_grid[np.where(one_sigma_mask)[0][-1]])
    else:
        upsilon_lo = float(best_upsilon)
        upsilon_hi = float(best_upsilon)

    best_modeled = metrics_by_grid[best_idx]
    best_metrics = _chi2_metrics(
        velocity_obs_km_s,
        velocity_obs_sigma_km_s,
        best_modeled["velocity_pred_km_s"],
        sigma_floor_km_s=float(sigma_floor_km_s),
        fit_params=1,
    )

    return {
        "upsilon_best": best_upsilon,
        "upsilon_1sigma": [upsilon_lo, upsilon_hi],
        "fit_metrics": best_metrics,
        "modeled": best_modeled,
    }


def _write_points_csv(
    out_csv: Path,
    points: Sequence[RotmodPoint],
    *,
    baryon_best: Dict[str, Any],
    pmodel_best: Dict[str, Any],
    sigma_floor_km_s: float,
) -> None:
    velocity_obs_km_s = np.asarray([point.velocity_obs_km_s for point in points], dtype=float)
    sigma_obs_km_s = np.asarray([point.velocity_obs_sigma_km_s for point in points], dtype=float)
    sigma_used_km_s = np.maximum(np.maximum(sigma_obs_km_s, 0.0), float(sigma_floor_km_s))

    velocity_bar_baryon = np.asarray(baryon_best["modeled"]["velocity_bar_km_s"], dtype=float)
    velocity_pred_baryon = np.asarray(baryon_best["modeled"]["velocity_pred_km_s"], dtype=float)
    velocity_bar_pmodel = np.asarray(pmodel_best["modeled"]["velocity_bar_km_s"], dtype=float)
    velocity_pred_pmodel = np.asarray(pmodel_best["modeled"]["velocity_pred_km_s"], dtype=float)
    accel_bar_pmodel = np.asarray(pmodel_best["modeled"]["acceleration_bar_m_s2"], dtype=float)
    accel_pred_pmodel = np.asarray(pmodel_best["modeled"]["acceleration_pred_m_s2"], dtype=float)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "galaxy",
            "radius_kpc",
            "vobs_km_s",
            "evobs_km_s",
            "sigma_used_km_s",
            "vbar_bestfit_baryon_km_s",
            "vpred_baryon_km_s",
            "vbar_bestfit_pmodel_km_s",
            "vpred_pmodel_km_s",
            "resid_baryon_km_s",
            "resid_pmodel_km_s",
            "pull_baryon_sigma",
            "pull_pmodel_sigma",
            "gbar_bestfit_pmodel_m_s2",
            "gpred_pmodel_m_s2",
        ]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for idx, point in enumerate(points):
            resid_baryon = float(velocity_obs_km_s[idx] - velocity_pred_baryon[idx])
            resid_pmodel = float(velocity_obs_km_s[idx] - velocity_pred_pmodel[idx])
            sigma_used = float(sigma_used_km_s[idx])
            writer.writerow(
                {
                    "galaxy": point.galaxy,
                    "radius_kpc": float(point.radius_kpc),
                    "vobs_km_s": float(velocity_obs_km_s[idx]),
                    "evobs_km_s": float(sigma_obs_km_s[idx]),
                    "sigma_used_km_s": sigma_used,
                    "vbar_bestfit_baryon_km_s": float(velocity_bar_baryon[idx]),
                    "vpred_baryon_km_s": float(velocity_pred_baryon[idx]),
                    "vbar_bestfit_pmodel_km_s": float(velocity_bar_pmodel[idx]),
                    "vpred_pmodel_km_s": float(velocity_pred_pmodel[idx]),
                    "resid_baryon_km_s": resid_baryon,
                    "resid_pmodel_km_s": resid_pmodel,
                    "pull_baryon_sigma": float(resid_baryon / sigma_used),
                    "pull_pmodel_sigma": float(resid_pmodel / sigma_used),
                    "gbar_bestfit_pmodel_m_s2": float(accel_bar_pmodel[idx]),
                    "gpred_pmodel_m_s2": float(accel_pred_pmodel[idx]),
                }
            )


def _write_galaxy_summary_csv(
    out_csv: Path,
    points: Sequence[RotmodPoint],
    *,
    baryon_best: Dict[str, Any],
    pmodel_best: Dict[str, Any],
    sigma_floor_km_s: float,
) -> Dict[str, float]:
    velocity_obs_km_s = np.asarray([point.velocity_obs_km_s for point in points], dtype=float)
    sigma_obs_km_s = np.asarray([point.velocity_obs_sigma_km_s for point in points], dtype=float)
    sigma_used_km_s = np.maximum(np.maximum(sigma_obs_km_s, 0.0), float(sigma_floor_km_s))
    velocity_pred_baryon = np.asarray(baryon_best["modeled"]["velocity_pred_km_s"], dtype=float)
    velocity_pred_pmodel = np.asarray(pmodel_best["modeled"]["velocity_pred_km_s"], dtype=float)

    galaxy_to_indices: Dict[str, List[int]] = {}
    for idx, point in enumerate(points):
        galaxy_to_indices.setdefault(point.galaxy, []).append(idx)

    rows: List[Dict[str, Any]] = []
    chi2_dof_baryon_list: List[float] = []
    chi2_dof_pmodel_list: List[float] = []
    for galaxy, indices in sorted(galaxy_to_indices.items()):
        idx_arr = np.asarray(indices, dtype=int)
        residual_baryon = velocity_obs_km_s[idx_arr] - velocity_pred_baryon[idx_arr]
        residual_pmodel = velocity_obs_km_s[idx_arr] - velocity_pred_pmodel[idx_arr]
        sigma_used = sigma_used_km_s[idx_arr]
        pull_baryon = residual_baryon / sigma_used
        pull_pmodel = residual_pmodel / sigma_used
        chi2_baryon = float(np.sum(pull_baryon * pull_baryon))
        chi2_pmodel = float(np.sum(pull_pmodel * pull_pmodel))
        dof_local = max(1, int(idx_arr.size))
        chi2_dof_baryon = float(chi2_baryon / float(dof_local))
        chi2_dof_pmodel = float(chi2_pmodel / float(dof_local))
        chi2_dof_baryon_list.append(chi2_dof_baryon)
        chi2_dof_pmodel_list.append(chi2_dof_pmodel)
        rows.append(
            {
                "galaxy": galaxy,
                "n_points": int(idx_arr.size),
                "chi2_baryon": chi2_baryon,
                "chi2_dof_baryon": chi2_dof_baryon,
                "chi2_pmodel": chi2_pmodel,
                "chi2_dof_pmodel": chi2_dof_pmodel,
                "delta_chi2_baryon_minus_pmodel": float(chi2_baryon - chi2_pmodel),
                "rms_resid_baryon_km_s": float(np.sqrt(np.mean(residual_baryon * residual_baryon))),
                "rms_resid_pmodel_km_s": float(np.sqrt(np.mean(residual_pmodel * residual_pmodel))),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "galaxy",
            "n_points",
            "chi2_baryon",
            "chi2_dof_baryon",
            "chi2_pmodel",
            "chi2_dof_pmodel",
            "delta_chi2_baryon_minus_pmodel",
            "rms_resid_baryon_km_s",
            "rms_resid_pmodel_km_s",
        ]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return {
        "median_chi2_dof_baryon": float(np.median(np.asarray(chi2_dof_baryon_list, dtype=float))) if chi2_dof_baryon_list else float("nan"),
        "median_chi2_dof_pmodel": float(np.median(np.asarray(chi2_dof_pmodel_list, dtype=float))) if chi2_dof_pmodel_list else float("nan"),
    }


def _plot_summary(
    out_png: Path,
    *,
    points: Sequence[RotmodPoint],
    baryon_best: Dict[str, Any],
    pmodel_best: Dict[str, Any],
    sigma_floor_km_s: float,
) -> None:
    if plt is None:
        return

    velocity_obs_km_s = np.asarray([point.velocity_obs_km_s for point in points], dtype=float)
    sigma_obs_km_s = np.asarray([point.velocity_obs_sigma_km_s for point in points], dtype=float)
    sigma_used_km_s = np.maximum(np.maximum(sigma_obs_km_s, 0.0), float(sigma_floor_km_s))
    velocity_pred_baryon = np.asarray(baryon_best["modeled"]["velocity_pred_km_s"], dtype=float)
    velocity_pred_pmodel = np.asarray(pmodel_best["modeled"]["velocity_pred_km_s"], dtype=float)

    residual_pull_baryon = (velocity_obs_km_s - velocity_pred_baryon) / sigma_used_km_s
    residual_pull_pmodel = (velocity_obs_km_s - velocity_pred_pmodel) / sigma_used_km_s

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=170)

    vmax = float(np.nanmax([np.max(velocity_obs_km_s), np.max(velocity_pred_baryon), np.max(velocity_pred_pmodel)]))
    axes[0].scatter(velocity_obs_km_s, velocity_pred_baryon, s=6, alpha=0.35, color="#d62728", label="baryon-only")
    axes[0].scatter(velocity_obs_km_s, velocity_pred_pmodel, s=6, alpha=0.35, color="#1f77b4", label="P-model corrected")
    axes[0].plot([0.0, vmax], [0.0, vmax], "k--", lw=1.0, alpha=0.7)
    axes[0].set_xlabel("Vobs [km/s]")
    axes[0].set_ylabel("Vmodel [km/s]")
    axes[0].set_title("SPARC rotation curves (all points)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper left", fontsize=8)

    bins = np.linspace(-8.0, 8.0, 61)
    axes[1].hist(residual_pull_baryon, bins=bins, alpha=0.55, color="#d62728", label="baryon-only")
    axes[1].hist(residual_pull_pmodel, bins=bins, alpha=0.55, color="#1f77b4", label="P-model corrected")
    axes[1].axvline(0.0, color="k", ls="--", lw=1.0)
    axes[1].set_xlabel("(Vobs - Vmodel) / sigma")
    axes[1].set_ylabel("count")
    axes[1].set_title("Normalized residual distribution")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8)

    abs_pull_p50 = float(np.median(np.abs(residual_pull_pmodel)))
    abs_pull_b50 = float(np.median(np.abs(residual_pull_baryon)))
    chi2_dof_baryon = float(baryon_best["fit_metrics"]["chi2_dof"])
    chi2_dof_pmodel = float(pmodel_best["fit_metrics"]["chi2_dof"])
    model_labels = ["baryon-only", "P-model corrected"]
    model_values = [chi2_dof_baryon, chi2_dof_pmodel]
    model_colors = ["#d62728", "#1f77b4"]
    axes[2].bar(model_labels, model_values, color=model_colors, alpha=0.85)
    axes[2].set_ylabel("global chi2/dof")
    axes[2].set_title("Fit quality (single M/L parameter)")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].text(
        0.02,
        0.96,
        f"median |pull|: baryon={abs_pull_b50:.3f}, P-model={abs_pull_p50:.3f}",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    fig.suptitle("SPARC audit: Vobs vs Vbar and P-model-corrected VP (single Υ fit)", fontsize=13)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SPARC rotation-curve audit with single M/L fit.")
    parser.add_argument(
        "--rotmod-zip",
        default=str(_ROOT / "data" / "cosmology" / "sparc" / "raw" / "Rotmod_LTG.zip"),
        help="Path to SPARC Rotmod_LTG.zip",
    )
    parser.add_argument(
        "--h0p-metrics",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_redshift_pbg_metrics.json"),
        help="Path to H0^(P) metrics JSON (fallback to output/public if missing).",
    )
    parser.add_argument("--pbg-kappa", type=float, default=DEFAULT_PBG_KAPPA, help="a0 = kappa * c * H0^(P)")
    parser.add_argument("--bulge-to-disk-ml-ratio", type=float, default=1.4, help="Fix Υ_bulge = ratio * Υ_disk")
    parser.add_argument("--upsilon-min", type=float, default=0.05, help="Υ grid min")
    parser.add_argument("--upsilon-max", type=float, default=1.20, help="Υ grid max")
    parser.add_argument("--upsilon-step", type=float, default=0.005, help="Υ grid step")
    parser.add_argument("--sigma-floor-km-s", type=float, default=3.0, help="Minimum velocity uncertainty for χ²")
    parser.add_argument("--accel-floor", type=float, default=1e-14, help="Acceleration floor [m/s²]")
    parser.add_argument(
        "--out-dir",
        default=str(_ROOT / "output" / "public" / "cosmology"),
        help="Output directory",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    rotmod_zip = Path(args.rotmod_zip)
    h0p_metrics = Path(args.h0p_metrics)
    if not h0p_metrics.exists():
        fallback = _ROOT / "output" / "public" / "cosmology" / "cosmology_redshift_pbg_metrics.json"
        if fallback.exists():
            h0p_metrics = fallback
        else:
            raise FileNotFoundError(f"missing h0p metrics: {args.h0p_metrics} (fallback also missing)")

    if float(args.upsilon_step) <= 0.0 or float(args.upsilon_max) < float(args.upsilon_min):
        raise ValueError("invalid upsilon grid")

    points = _load_rotmod_points(rotmod_zip)
    if len(points) < 100:
        raise RuntimeError(f"too few SPARC points: {len(points)}")

    arrays = _arrays_from_points(points)
    n_galaxies = int(len({point.galaxy for point in points}))
    n_points = int(len(points))

    h0p_si_s = _h0p_from_metrics(h0p_metrics)
    a0_m_s2 = float(args.pbg_kappa) * C_LIGHT_M_S * h0p_si_s
    if not np.isfinite(a0_m_s2) or a0_m_s2 <= 0.0:
        raise RuntimeError("invalid a0 from kappa*c*H0P")

    upsilon_grid = np.arange(float(args.upsilon_min), float(args.upsilon_max) + 0.5 * float(args.upsilon_step), float(args.upsilon_step))
    upsilon_grid = upsilon_grid[np.isfinite(upsilon_grid) & (upsilon_grid >= 0.0)]
    if upsilon_grid.size < 2:
        raise RuntimeError("upsilon grid too small")

    baryon_best = _fit_upsilon(
        arrays,
        mode="baryon_only",
        upsilon_grid=upsilon_grid,
        bulge_to_disk_ratio=float(args.bulge_to_disk_ml_ratio),
        a0_m_s2=a0_m_s2,
        sigma_floor_km_s=float(args.sigma_floor_km_s),
        acceleration_floor_m_s2=float(args.accel_floor),
    )
    pmodel_best = _fit_upsilon(
        arrays,
        mode="pmodel_boost",
        upsilon_grid=upsilon_grid,
        bulge_to_disk_ratio=float(args.bulge_to_disk_ml_ratio),
        a0_m_s2=a0_m_s2,
        sigma_floor_km_s=float(args.sigma_floor_km_s),
        acceleration_floor_m_s2=float(args.accel_floor),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_points_csv = out_dir / "sparc_rotation_curve_pmodel_audit_points.csv"
    out_galaxy_csv = out_dir / "sparc_rotation_curve_pmodel_audit_galaxy_summary.csv"
    out_png = out_dir / "sparc_rotation_curve_pmodel_audit.png"
    out_metrics = out_dir / "sparc_rotation_curve_pmodel_audit_metrics.json"

    _write_points_csv(
        out_points_csv,
        points,
        baryon_best=baryon_best,
        pmodel_best=pmodel_best,
        sigma_floor_km_s=float(args.sigma_floor_km_s),
    )
    galaxy_summary_stats = _write_galaxy_summary_csv(
        out_galaxy_csv,
        points,
        baryon_best=baryon_best,
        pmodel_best=pmodel_best,
        sigma_floor_km_s=float(args.sigma_floor_km_s),
    )
    _plot_summary(
        out_png,
        points=points,
        baryon_best=baryon_best,
        pmodel_best=pmodel_best,
        sigma_floor_km_s=float(args.sigma_floor_km_s),
    )

    chi2_baryon = float(baryon_best["fit_metrics"]["chi2"])
    chi2_pmodel = float(pmodel_best["fit_metrics"]["chi2"])
    delta_chi2 = float(chi2_baryon - chi2_pmodel)

    payload = {
        "generated_utc": _utc_now(),
        "phase": 8,
        "step": "8.7.23",
        "inputs": {
            "rotmod_zip": _rel(rotmod_zip),
            "h0p_metrics": _rel(h0p_metrics),
            "pbg_kappa": float(args.pbg_kappa),
            "bulge_to_disk_ml_ratio": float(args.bulge_to_disk_ml_ratio),
            "upsilon_grid": {
                "min": float(args.upsilon_min),
                "max": float(args.upsilon_max),
                "step": float(args.upsilon_step),
                "n": int(upsilon_grid.size),
            },
            "sigma_floor_km_s": float(args.sigma_floor_km_s),
            "acceleration_floor_m_s2": float(args.accel_floor),
        },
        "counts": {"n_galaxies": n_galaxies, "n_points": n_points},
        "pmodel_fixed": {
            "a0_m_s2": float(a0_m_s2),
            "a0_log10_m_s2": float(math.log10(a0_m_s2)),
            "h0p_si_s^-1": float(h0p_si_s),
            "formula": "a0 = kappa * c * H0^(P)",
        },
        "fit_results": {
            "baryon_only": {
                "upsilon_best": float(baryon_best["upsilon_best"]),
                "upsilon_1sigma": [float(x) for x in baryon_best["upsilon_1sigma"]],
                **{key: float(value) for key, value in baryon_best["fit_metrics"].items()},
            },
            "pmodel_corrected": {
                "upsilon_best": float(pmodel_best["upsilon_best"]),
                "upsilon_1sigma": [float(x) for x in pmodel_best["upsilon_1sigma"]],
                **{key: float(value) for key, value in pmodel_best["fit_metrics"].items()},
            },
            "comparison": {
                "delta_chi2_baryon_minus_pmodel": delta_chi2,
                "chi2_dof_ratio_pmodel_over_baryon": float(
                    float(pmodel_best["fit_metrics"]["chi2_dof"]) / max(float(baryon_best["fit_metrics"]["chi2_dof"]), 1e-30)
                ),
                "better_model_by_chi2": "pmodel_corrected" if delta_chi2 > 0.0 else "baryon_only",
            },
            "galaxy_level_summary": galaxy_summary_stats,
        },
        "outputs": {
            "points_csv": _rel(out_points_csv),
            "galaxy_summary_csv": _rel(out_galaxy_csv),
            "figure_png": _rel(out_png) if out_png.exists() else None,
            "metrics_json": _rel(out_metrics),
        },
        "notes": [
            "Single free parameter fit: Υ_disk=Υ, Υ_bulge=(bulge_to_disk_ml_ratio)*Υ.",
            "P-model correction uses gP = gbar / (1 - exp(-sqrt(gbar/a0))) with a0 fixed by kappa*c*H0^(P).",
            "This is an operational audit I/F for Part II; it does not by itself establish a fundamental derivation.",
        ],
    }
    _write_json(out_metrics, payload)

    if worklog is not None:
        try:
            worklog.append_event(
                kind="run",
                action="sparc_rotation_curve_pmodel_audit",
                inputs={
                    "rotmod_zip": _rel(rotmod_zip),
                    "h0p_metrics": _rel(h0p_metrics),
                    "pbg_kappa": float(args.pbg_kappa),
                    "bulge_to_disk_ml_ratio": float(args.bulge_to_disk_ml_ratio),
                    "upsilon_grid": {"min": float(args.upsilon_min), "max": float(args.upsilon_max), "step": float(args.upsilon_step)},
                },
                outputs={
                    "metrics_json": _rel(out_metrics),
                    "points_csv": _rel(out_points_csv),
                    "galaxy_summary_csv": _rel(out_galaxy_csv),
                    "figure_png": _rel(out_png) if out_png.exists() else None,
                },
                meta={
                    "n_galaxies": n_galaxies,
                    "n_points": n_points,
                    "delta_chi2_baryon_minus_pmodel": delta_chi2,
                    "chi2_dof_baryon": float(baryon_best["fit_metrics"]["chi2_dof"]),
                    "chi2_dof_pmodel": float(pmodel_best["fit_metrics"]["chi2_dof"]),
                },
            )
        except Exception:
            pass

    print(f"[ok] wrote: {out_points_csv}")
    print(f"[ok] wrote: {out_galaxy_csv}")
    if out_png.exists():
        print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

