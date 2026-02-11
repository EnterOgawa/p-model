#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparc_falsification_pack.py

Phase 6 / Step 6.5（SPARC：RAR/BTFR）:
RAR 再構築（観測側 g_obs/g_bar）を入力として、最小の falsification pack（統計量・閾値・凍結条件）を固定する。

現段階（6.5.1 の既定：P-model = GR弱場）では、baryons-only（g_P=g_bar）を「参照ヌル」として扱い、
RAR がそのヌルからどれだけ系統的にずれているかを数値で固定する。

入力：
- output/private/cosmology/sparc_rar_reconstruction.csv

出力（固定）：
- output/private/cosmology/sparc_falsification_pack.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from scripts.summary import worklog  # type: ignore # noqa: E402
except Exception:  # pragma: no cover
    worklog = None

C_LIGHT_M_S = 299_792_458.0
MPC_TO_M = 3.0856775814913673e22
DEFAULT_PBG_KAPPA = 1.0 / (2.0 * math.pi)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_points(csv_path: Path) -> List[Dict[str, float]]:
    pts: List[Dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                g_bar = float(row.get("g_bar_m_s2") or "nan")
                g_obs = float(row.get("g_obs_m_s2") or "nan")
                sg_obs = float(row.get("g_obs_sigma_m_s2") or "nan")
            except Exception:
                continue
            if not (np.isfinite(g_bar) and np.isfinite(g_obs)) or g_bar <= 0.0 or g_obs <= 0.0:
                continue
            pts.append({"g_bar": g_bar, "g_obs": g_obs, "sg_obs": sg_obs})
    return pts


def _rar_mcgaugh2016_log10_pred(g_bar: np.ndarray, *, log10_a0: float) -> np.ndarray:
    # McGaugh+ (2016) empirical RAR function:
    # g_obs = g_bar / (1 - exp(-sqrt(g_bar/a0)))
    a0 = float(10.0 ** float(log10_a0))
    x = np.asarray(g_bar, dtype=float) / a0
    x = np.where(np.isfinite(x) & (x > 0.0), x, np.nan)
    s = np.sqrt(x)
    # denom = 1 - exp(-s)  (use expm1 for numerical stability at small s)
    denom = -np.expm1(-s)
    denom = np.where(np.isfinite(denom) & (denom > 0.0), denom, np.nan)
    g_pred = np.asarray(g_bar, dtype=float) / denom
    return np.log10(g_pred)


def _fit_log10_a0_grid(
    g_bar: np.ndarray,
    y_log10_gobs: np.ndarray,
    sigma_y: np.ndarray,
    *,
    log10_a0_min: float = -13.0,
    log10_a0_max: float = -9.0,
    n_grid: int = 801,
) -> Dict[str, Any]:
    m = np.isfinite(g_bar) & (g_bar > 0.0) & np.isfinite(y_log10_gobs) & np.isfinite(sigma_y) & (sigma_y > 0.0)
    if int(np.count_nonzero(m)) < 10:
        return {"status": "not_enough_points", "n_used": int(np.count_nonzero(m))}

    gb = g_bar[m]
    y = y_log10_gobs[m]
    sy = sigma_y[m]

    grid = np.linspace(float(log10_a0_min), float(log10_a0_max), int(max(n_grid, 50)))
    chi2 = np.full(grid.shape, np.nan, dtype=float)
    for i, la0 in enumerate(grid):
        y_pred = _rar_mcgaugh2016_log10_pred(gb, log10_a0=float(la0))
        rr = y - y_pred
        mm = np.isfinite(rr)
        if int(np.count_nonzero(mm)) < 10:
            continue
        chi2[i] = float(np.sum((rr[mm] / sy[mm]) ** 2))

    if not np.isfinite(chi2).any():
        return {"status": "failed", "reason": "chi2 all nan"}

    i_best = int(np.nanargmin(chi2))
    la0_best = float(grid[i_best])
    chi2_min = float(chi2[i_best])
    y_pred_best = _rar_mcgaugh2016_log10_pred(gb, log10_a0=la0_best)
    r_best = y - y_pred_best
    mm = np.isfinite(r_best)
    r_best = r_best[mm]
    dof = int(np.count_nonzero(mm)) - 1  # 1 parameter (a0)

    # 1σ interval for 1 parameter: Δχ²=1
    # Find contiguous region around the minimum on the grid.
    thresh = chi2_min + 1.0
    left = i_best
    while left > 0 and np.isfinite(chi2[left - 1]) and chi2[left - 1] <= thresh:
        left -= 1
    right = i_best
    while right + 1 < chi2.size and np.isfinite(chi2[right + 1]) and chi2[right + 1] <= thresh:
        right += 1

    la0_lo = float(grid[left])
    la0_hi = float(grid[right])

    return {
        "status": "ok",
        "model": "g_obs=g_bar/(1-exp(-sqrt(g_bar/a0)))",
        "n_used": int(np.count_nonzero(m)),
        "log10_a0_best_m_s2": la0_best,
        "a0_best_m_s2": float(10.0**la0_best),
        "log10_a0_1sigma_m_s2": [la0_lo, la0_hi],
        "a0_1sigma_m_s2": [float(10.0**la0_lo), float(10.0**la0_hi)],
        "chi2_min": chi2_min,
        "dof": int(dof),
        "chi2_dof": float(chi2_min / float(dof)) if dof > 0 else float("nan"),
        "residual_stats_dex": {
            "median": float(np.median(r_best)),
            "rms": float(np.sqrt(np.mean(r_best * r_best))),
            "q16": float(np.quantile(r_best, 0.16)),
            "q84": float(np.quantile(r_best, 0.84)),
        },
    }

def _h0p_si_from_km_s_mpc(h0_km_s_mpc: float) -> float:
    return float(h0_km_s_mpc) * 1.0e3 / MPC_TO_M


def _get_h0p_si(
    *,
    h0p_metrics: Path,
    h0p_km_s_mpc_override: Optional[float],
) -> Tuple[float, Dict[str, Any]]:
    if h0p_km_s_mpc_override is not None and np.isfinite(h0p_km_s_mpc_override) and float(h0p_km_s_mpc_override) > 0:
        h0_si = _h0p_si_from_km_s_mpc(float(h0p_km_s_mpc_override))
        return h0_si, {"mode": "override", "H0P_km_s_Mpc": float(h0p_km_s_mpc_override), "H0P_SI_s^-1": h0_si}

    d = _read_json(h0p_metrics)
    derived = d.get("derived", {}) if isinstance(d.get("derived", {}), dict) else {}
    params = d.get("params", {}) if isinstance(d.get("params", {}), dict) else {}
    h0_si = derived.get("H0P_SI_s^-1")
    if isinstance(h0_si, (int, float)) and np.isfinite(h0_si) and float(h0_si) > 0:
        return float(h0_si), {"mode": "metrics", "path": _rel(h0p_metrics), "H0P_SI_s^-1": float(h0_si)}

    h0_km_s_mpc = params.get("H0P_km_s_Mpc")
    if isinstance(h0_km_s_mpc, (int, float)) and np.isfinite(h0_km_s_mpc) and float(h0_km_s_mpc) > 0:
        h0_si2 = _h0p_si_from_km_s_mpc(float(h0_km_s_mpc))
        return h0_si2, {"mode": "metrics_params", "path": _rel(h0p_metrics), "H0P_km_s_Mpc": float(h0_km_s_mpc), "H0P_SI_s^-1": h0_si2}

    raise RuntimeError(f"failed to read H0^(P) from metrics: {h0p_metrics}")


def _rar_fixed_a0_stats(
    *,
    g_bar: np.ndarray,
    y_log10_gobs: np.ndarray,
    sigma_y: np.ndarray,
    log10_a0: float,
    low_accel_cut_log10_gbar: float,
) -> Dict[str, Any]:
    m = np.isfinite(g_bar) & (g_bar > 0.0) & np.isfinite(y_log10_gobs) & np.isfinite(sigma_y) & (sigma_y > 0.0)
    if int(np.count_nonzero(m)) < 10:
        return {"status": "not_enough_points", "n_used": int(np.count_nonzero(m))}

    gb = g_bar[m]
    y = y_log10_gobs[m]
    sy = sigma_y[m]
    y_pred = _rar_mcgaugh2016_log10_pred(gb, log10_a0=float(log10_a0))
    resid = y - y_pred
    mm = np.isfinite(resid)
    if int(np.count_nonzero(mm)) < 10:
        return {"status": "failed", "reason": "residual all nan"}

    resid2 = resid[mm]
    sy2 = sy[mm]
    chi2 = float(np.sum((resid2 / sy2) ** 2))
    dof = int(np.count_nonzero(mm))  # 0 free parameters (a0 is fixed)

    # Low-accel z (weighted mean / SEM)
    x = np.log10(gb)
    m_low = np.isfinite(resid) & np.isfinite(sy) & (x < float(low_accel_cut_log10_gbar))
    mean_low = float("nan")
    sem_low = float("nan")
    z_low = float("nan")
    if int(np.count_nonzero(m_low)) >= 10:
        r_low = resid[m_low]
        w = 1.0 / (sy[m_low] ** 2)
        wsum = float(np.sum(w))
        mean_low = float(np.sum(w * r_low) / wsum) if wsum > 0 else float(np.mean(r_low))
        sem_low = float(math.sqrt(1.0 / wsum)) if wsum > 0 else float(np.std(r_low) / math.sqrt(max(len(r_low), 1)))
        z_low = mean_low / sem_low if sem_low > 0 else float("inf")

    return {
        "status": "ok",
        "model": "g_obs=g_bar/(1-exp(-sqrt(g_bar/a0)))",
        "n_used": int(np.count_nonzero(m)),
        "log10_a0_m_s2": float(log10_a0),
        "a0_m_s2": float(10.0 ** float(log10_a0)),
        "chi2": float(chi2),
        "dof": int(dof),
        "chi2_dof": float(chi2 / float(dof)) if dof > 0 else float("nan"),
        "residual_stats_dex": {
            "median": float(np.median(resid2)),
            "rms": float(np.sqrt(np.mean(resid2 * resid2))),
            "q16": float(np.quantile(resid2, 0.16)),
            "q84": float(np.quantile(resid2, 0.84)),
        },
        "low_accel": {
            "cut_log10_gbar": float(low_accel_cut_log10_gbar),
            "weighted_mean_residual_dex": float(mean_low),
            "sem_dex": float(sem_low),
            "z": float(z_low),
        },
    }

def _summarize_freeze_test_metrics(d: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {"status": "invalid"}
    models = d.get("models")
    if not isinstance(models, list) or not models:
        return {"status": "not_found"}
    sweep_summary = d.get("sweep_summary") if isinstance(d.get("sweep_summary"), dict) else {}
    sweep_summary_galaxy = d.get("sweep_summary_galaxy") if isinstance(d.get("sweep_summary_galaxy"), dict) else {}
    runs = d.get("runs") if isinstance(d.get("runs"), list) else []
    out_models: List[Dict[str, Any]] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "")
        test = m.get("test") if isinstance(m.get("test"), dict) else {}
        with_si = test.get("with_sigma_int") if isinstance(test.get("with_sigma_int"), dict) else {}
        low = with_si.get("low_accel") if isinstance(with_si.get("low_accel"), dict) else {}
        low_gal = with_si.get("low_accel_galaxy") if isinstance(with_si.get("low_accel_galaxy"), dict) else {}
        out_models.append(
            {
                "name": name,
                "fit": m.get("fit", {}) if isinstance(m.get("fit"), dict) else {},
                "holdout_with_sigma_int": {
                    "chi2_dof": with_si.get("chi2_dof"),
                    "sigma_int_dex": with_si.get("sigma_int_dex"),
                    "low_accel": {
                        "cut_log10_gbar": low.get("cut_log10_gbar"),
                        "weighted_mean_residual_dex": low.get("weighted_mean_residual_dex"),
                        "sem_dex": low.get("sem_dex"),
                        "z": low.get("z"),
                    },
                    "low_accel_galaxy": {
                        "cut_log10_gbar": low_gal.get("cut_log10_gbar"),
                        "min_points_per_galaxy": low_gal.get("min_points_per_galaxy"),
                        "n_galaxies_low_accel": low_gal.get("n_galaxies_low_accel"),
                        "mean_residual_dex": low_gal.get("mean_residual_dex"),
                        "sem_dex": low_gal.get("sem_dex"),
                        "z": low_gal.get("z"),
                    },
                },
            }
        )
    return {
        "status": "ok",
        "generated_utc": d.get("generated_utc"),
        "inputs": d.get("inputs", {}) if isinstance(d.get("inputs"), dict) else {},
        "counts": d.get("counts", {}) if isinstance(d.get("counts"), dict) else {},
        "n_runs": int(len(runs)),
        "sweep_summary": sweep_summary,
        "sweep_summary_galaxy": sweep_summary_galaxy,
        "models": out_models,
    }


def _summarize_freeze_test_mlr_sweep_metrics(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize output/private/cosmology/sparc_rar_freeze_test_mlr_sweep_metrics.json into a small dict
    suitable for inclusion in sparc_falsification_pack.json (avoid embedding the full variants list).
    """
    if not isinstance(d, dict):
        return {"status": "invalid"}

    inputs = d.get("inputs", {}) if isinstance(d.get("inputs"), dict) else {}
    envelope = d.get("envelope", {}) if isinstance(d.get("envelope"), dict) else {}
    pr_env_gal = envelope.get("pass_rate_abs_lt_threshold_galaxy", {}) if isinstance(envelope.get("pass_rate_abs_lt_threshold_galaxy"), dict) else {}
    marginal = d.get("marginal", {}) if isinstance(d.get("marginal"), dict) else {}
    pr_marg_gal = (
        marginal.get("pass_rate_abs_lt_threshold_galaxy", {})
        if isinstance(marginal.get("pass_rate_abs_lt_threshold_galaxy"), dict)
        else {}
    )
    robustness = d.get("robustness", {}) if isinstance(d.get("robustness"), dict) else {}

    # Identify worst/best (by candidate galaxy-level pass_rate) across the M/L grid.
    cand_name = "candidate_rar_pbg_a0_fixed_kappa"
    worst: Optional[Dict[str, Any]] = None
    best: Optional[Dict[str, Any]] = None
    variants = d.get("variants", []) if isinstance(d.get("variants"), list) else []
    for v in variants:
        if not isinstance(v, dict) or v.get("status") != "ok":
            continue
        ud = v.get("upsilon_disk")
        ub = v.get("upsilon_bulge")
        ss_g = v.get("sweep_summary_galaxy", {}) if isinstance(v.get("sweep_summary_galaxy"), dict) else {}
        cand = ss_g.get(cand_name, {}) if isinstance(ss_g.get(cand_name), dict) else {}
        pr = cand.get("pass_rate_abs_lt_threshold")
        if not (isinstance(pr, (int, float)) and np.isfinite(pr)):
            continue
        row = {"upsilon_disk": ud, "upsilon_bulge": ub, "pass_rate_abs_lt_threshold": float(pr), "median_z": cand.get("median")}
        if worst is None or float(row["pass_rate_abs_lt_threshold"]) < float(worst["pass_rate_abs_lt_threshold"]):
            worst = row
        if best is None or float(row["pass_rate_abs_lt_threshold"]) > float(best["pass_rate_abs_lt_threshold"]):
            best = row

    candidate_env = robustness.get("candidate", {}) if isinstance(robustness.get("candidate"), dict) else {}
    robust_adopted = robustness.get("robust_adopted")
    candidate_marg = pr_marg_gal.get(cand_name, {}) if isinstance(pr_marg_gal.get(cand_name), dict) else {}

    return {
        "status": "ok",
        "generated_utc": d.get("generated_utc"),
        "inputs": {
            "upsilon_disk": inputs.get("upsilon_disk"),
            "upsilon_bulge": inputs.get("upsilon_bulge"),
            "sigma_floor_dex": inputs.get("sigma_floor_dex"),
            "low_accel_cut_log10_gbar": inputs.get("low_accel_cut_log10_gbar"),
            "preferred_metric": inputs.get("preferred_metric"),
            "threshold_abs_z": inputs.get("threshold_abs_z"),
            "ref_upsilon": inputs.get("ref_upsilon"),
        },
        "envelope_pass_rate_abs_lt_threshold_galaxy": pr_env_gal,
        "candidate": {
            "name": cand_name,
            "worst": worst,
            "best": best,
            "envelope": candidate_env,
            "marginal_by_upsilon_disk": candidate_marg.get("by_upsilon_disk"),
            "marginal_by_upsilon_bulge": candidate_marg.get("by_upsilon_bulge"),
            "robust_adopted": robust_adopted,
            "decision_rule": robustness.get("decision_rule"),
            "pass_rate_required": robustness.get("pass_rate_required"),
        },
    }


def _adoption_eval_from_sweep_summary(
    sweep_summary: Dict[str, Any],
    *,
    pass_rate_required: float,
    threshold_abs_z: float,
    min_runs: int,
) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for name, stats in sweep_summary.items():
        if not isinstance(stats, dict):
            continue
        n = int(stats.get("n") or 0)
        pr = stats.get("pass_rate_abs_lt_threshold")
        pr_f = float(pr) if isinstance(pr, (int, float)) and np.isfinite(pr) else float("nan")
        adopted = bool(n >= int(min_runs) and np.isfinite(pr_f) and pr_f >= float(pass_rate_required))
        models[str(name)] = {
            "n": n,
            "pass_rate_abs_lt_threshold": pr_f,
            "min_z": stats.get("min"),
            "max_z": stats.get("max"),
            "median_z": stats.get("median"),
            "adopted": adopted,
        }
    return {
        "criteria": {
            "threshold_abs_z": float(threshold_abs_z),
            "pass_rate_required": float(pass_rate_required),
            "min_runs": int(min_runs),
            "decision_rule": "Adopt if pass_rate(|z|<threshold) >= pass_rate_required over the sweep_summary and n_runs>=min_runs.",
        },
        "models": models,
    }


def _summarize_freeze_test_upsilon_fit_metrics(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize a freeze-test metrics JSON that includes the nuisance model
    `candidate_rar_pbg_a0_fixed_kappa_fit_upsilon_global` (Υ fitted on train).
    """
    if not isinstance(d, dict):
        return {"status": "invalid"}

    model_name = "candidate_rar_pbg_a0_fixed_kappa_fit_upsilon_global"
    inputs = d.get("inputs", {}) if isinstance(d.get("inputs"), dict) else {}
    ssg = d.get("sweep_summary_galaxy", {}) if isinstance(d.get("sweep_summary_galaxy"), dict) else {}
    model_ssg = ssg.get(model_name, {}) if isinstance(ssg.get(model_name), dict) else {}

    # Count fitted Υ choices across runs (keep only compact counts).
    disk_counts: Dict[str, int] = {}
    bulge_counts: Dict[str, int] = {}
    n_ok = 0
    runs = d.get("runs", []) if isinstance(d.get("runs"), list) else []
    for run in runs:
        models = run.get("models", []) if isinstance(run, dict) and isinstance(run.get("models"), list) else []
        for m in models:
            if not isinstance(m, dict) or str(m.get("name") or "") != model_name:
                continue
            ups = (m.get("fit", {}) if isinstance(m.get("fit"), dict) else {}).get("upsilon_fit", {})
            if not isinstance(ups, dict) or ups.get("status") != "ok":
                continue
            ud = ups.get("upsilon_disk_best")
            ub = ups.get("upsilon_bulge_best")
            if isinstance(ud, (int, float)) and np.isfinite(ud):
                k = f"{float(ud):g}"
                disk_counts[k] = int(disk_counts.get(k, 0) + 1)
            if isinstance(ub, (int, float)) and np.isfinite(ub):
                k = f"{float(ub):g}"
                bulge_counts[k] = int(bulge_counts.get(k, 0) + 1)
            n_ok += 1

    return {
        "status": "ok",
        "generated_utc": d.get("generated_utc"),
        "inputs": {
            "fit_upsilon_global": inputs.get("fit_upsilon_global"),
            "upsilon_fit_objective": inputs.get("upsilon_fit_objective"),
            "upsilon_grid": inputs.get("upsilon_grid"),
            "sigma_floor_dex": inputs.get("sigma_floor_dex"),
            "low_accel_cut_log10_gbar": inputs.get("low_accel_cut_log10_gbar"),
        },
        "model": {
            "name": model_name,
            "sweep_summary_galaxy": model_ssg,
            "upsilon_fit_counts": {
                "n_ok": int(n_ok),
                "upsilon_disk_best": {k: disk_counts[k] for k in sorted(disk_counts.keys(), key=float)},
                "upsilon_bulge_best": {k: bulge_counts[k] for k in sorted(bulge_counts.keys(), key=float)},
            },
        },
        "note": "Fits (upsilon_disk, upsilon_bulge) on train only (nuisance), then freezes (upsilon,a0) and evaluates holdout. This adds 2 nuisance DOF per split.",
    }


def _summarize_freeze_test_procedure_sweep_metrics(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize output/private/cosmology/sparc_rar_freeze_test_procedure_sweep_metrics.json into a small dict.
    """
    if not isinstance(d, dict):
        return {"status": "invalid"}

    inputs = d.get("inputs", {}) if isinstance(d.get("inputs"), dict) else {}
    env = d.get("envelope_pass_rate_abs_lt_threshold_galaxy", {}) if isinstance(d.get("envelope_pass_rate_abs_lt_threshold_galaxy"), dict) else {}
    cand_env = env.get("candidate_rar_pbg_a0_fixed_kappa", {}) if isinstance(env.get("candidate_rar_pbg_a0_fixed_kappa"), dict) else {}
    candidate = d.get("candidate", {}) if isinstance(d.get("candidate"), dict) else {}

    robust_adopted = None
    if cand_env.get("status") == "ok":
        try:
            robust_adopted = bool(float(cand_env.get("min")) >= 0.95)
        except Exception:
            robust_adopted = None

    return {
        "status": "ok",
        "generated_utc": d.get("generated_utc"),
        "inputs": {
            "seeds": inputs.get("seeds"),
            "train_fracs": inputs.get("train_fracs"),
            "procedure_grid": inputs.get("procedure_grid"),
            "threshold_abs_z": inputs.get("threshold_abs_z"),
            "models": inputs.get("models"),
        },
        "candidate": {
            "envelope_pass_rate_abs_lt_threshold_galaxy": cand_env,
            "worst": candidate.get("worst"),
            "best": candidate.get("best"),
            "robust_adopted": robust_adopted,
            "decision_rule": "robust_adopted := (min pass_rate(|z|<3) across procedure variants) >= 0.95 (galaxy-level).",
        },
        "note": inputs.get("note"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rar-csv",
        default=str(_ROOT / "output" / "private" / "cosmology" / "sparc_rar_reconstruction.csv"),
        help="RAR reconstruction CSV (default: output/private/cosmology/sparc_rar_reconstruction.csv)",
    )
    p.add_argument(
        "--out",
        default=str(_ROOT / "output" / "private" / "cosmology" / "sparc_falsification_pack.json"),
        help="Output JSON path",
    )
    p.add_argument("--sigma-floor-dex", type=float, default=0.01, help="Floor for sigma(log10 g_obs) in dex (default: 0.01)")
    p.add_argument(
        "--bin-start",
        type=float,
        default=-13.0,
        help="Start of log10(g_bar) bin edges (default: -13.0)",
    )
    p.add_argument("--bin-stop", type=float, default=-8.0, help="Stop of log10(g_bar) bin edges (default: -8.0)")
    p.add_argument("--bin-step", type=float, default=0.25, help="Bin width in dex (default: 0.25)")
    p.add_argument(
        "--low-accel-cut",
        type=float,
        default=-10.5,
        help="Low-acceleration domain cut: log10(g_bar) < cut (default: -10.5)",
    )
    p.add_argument(
        "--h0p-metrics",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_redshift_pbg_metrics.json"),
        help="Path to cosmology_redshift_pbg_metrics.json (default: output/private/cosmology/cosmology_redshift_pbg_metrics.json)",
    )
    p.add_argument("--h0p-km-s-mpc", type=float, default=None, help="Override H0^(P) in km/s/Mpc (optional)")
    p.add_argument("--pbg-kappa", type=float, default=DEFAULT_PBG_KAPPA, help="a0 = kappa * c * H0^(P) (default: 1/(2π))")
    p.add_argument(
        "--freeze-test-metrics",
        default=str(_ROOT / "output" / "private" / "cosmology" / "sparc_rar_freeze_test_metrics.json"),
        help="Path to sparc_rar_freeze_test_metrics.json (default: output/private/cosmology/sparc_rar_freeze_test_metrics.json)",
    )
    p.add_argument(
        "--freeze-test-upsilon-fit-metrics",
        default=str(_ROOT / "output" / "private" / "cosmology" / "sparc_rar_freeze_test_fit_upsilon_global_chi2_high_accel_metrics.json"),
        help="Path to freeze-test metrics for nuisance Υ fit on train (default: output/private/cosmology/sparc_rar_freeze_test_fit_upsilon_global_chi2_high_accel_metrics.json)",
    )
    p.add_argument(
        "--freeze-test-procedure-sweep-metrics",
        default=str(_ROOT / "output" / "private" / "cosmology" / "sparc_rar_freeze_test_procedure_sweep_metrics.json"),
        help="Path to sparc_rar_freeze_test_procedure_sweep_metrics.json (default: output/private/cosmology/sparc_rar_freeze_test_procedure_sweep_metrics.json)",
    )
    p.add_argument(
        "--freeze-test-mlr-sweep-metrics",
        default=str(_ROOT / "output" / "private" / "cosmology" / "sparc_rar_freeze_test_mlr_sweep_metrics.json"),
        help="Path to sparc_rar_freeze_test_mlr_sweep_metrics.json (default: output/private/cosmology/sparc_rar_freeze_test_mlr_sweep_metrics.json)",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    rar_csv = Path(args.rar_csv)
    if not rar_csv.exists():
        raise FileNotFoundError(f"missing rar csv: {rar_csv}")

    pts = _read_points(rar_csv)
    if not pts:
        raise RuntimeError("no valid points found in rar csv")

    g_bar = np.asarray([p["g_bar"] for p in pts], dtype=float)
    g_obs = np.asarray([p["g_obs"] for p in pts], dtype=float)
    sg_obs = np.asarray([p["sg_obs"] for p in pts], dtype=float)

    x = np.log10(g_bar)
    y = np.log10(g_obs)
    # σ_log10(g_obs) ≈ σ_gobs / (g_obs ln 10)
    sigma_y = sg_obs / (g_obs * math.log(10.0))
    sigma_floor = float(max(args.sigma_floor_dex, 1e-6))
    sigma_y = np.where(np.isfinite(sigma_y) & (sigma_y > 0.0), sigma_y, np.nan)
    sigma_y = np.where(np.isfinite(sigma_y) & (sigma_y < sigma_floor), sigma_floor, sigma_y)

    resid = y - x  # baryons-only (g_bar) as null

    edges = np.arange(float(args.bin_start), float(args.bin_stop) + 1e-12, float(args.bin_step))
    bins: List[Dict[str, Any]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi) & np.isfinite(resid)
        if int(np.count_nonzero(m)) == 0:
            continue
        rr = resid[m]
        bins.append(
            {
                "log10_gbar_lo": float(lo),
                "log10_gbar_hi": float(hi),
                "n": int(np.count_nonzero(m)),
                "median_residual_dex": float(np.median(rr)),
                "q16_residual_dex": float(np.quantile(rr, 0.16)),
                "q84_residual_dex": float(np.quantile(rr, 0.84)),
            }
        )

    # Simple null test in low-acceleration domain: mean(resid)/SEM(resid)
    m_low = np.isfinite(resid) & np.isfinite(sigma_y) & (x < float(args.low_accel_cut))
    z_low = float("nan")
    mean_low = float("nan")
    sem_low = float("nan")
    if int(np.count_nonzero(m_low)) >= 10:
        r_low = resid[m_low]
        w = 1.0 / (sigma_y[m_low] ** 2)
        wsum = float(np.sum(w))
        mean_low = float(np.sum(w * r_low) / wsum) if wsum > 0 else float(np.mean(r_low))
        sem_low = float(math.sqrt(1.0 / wsum)) if wsum > 0 else float(np.std(r_low) / math.sqrt(max(len(r_low), 1)))
        z_low = mean_low / sem_low if sem_low > 0 else float("inf")

    # Baseline fit (not a P-model prediction): empirical RAR function with fitted a0.
    baseline_rar_fit = _fit_log10_a0_grid(g_bar, y, sigma_y)
    candidate_pbg: Dict[str, Any] = {"status": "skipped"}
    h0p_src: Dict[str, Any] = {}
    try:
        h0p_si, h0p_src = _get_h0p_si(h0p_metrics=Path(args.h0p_metrics), h0p_km_s_mpc_override=args.h0p_km_s_mpc)
        kappa = float(args.pbg_kappa)
        if not np.isfinite(kappa) or kappa <= 0:
            raise ValueError("--pbg-kappa must be positive")
        a0_pbg = float(kappa) * float(C_LIGHT_M_S) * float(h0p_si)
        if not np.isfinite(a0_pbg) or a0_pbg <= 0:
            raise ValueError("invalid a0 computed from kappa*c*H0^(P)")
        la0_pbg = float(math.log10(a0_pbg))
        candidate_pbg = _rar_fixed_a0_stats(
            g_bar=g_bar,
            y_log10_gobs=y,
            sigma_y=sigma_y,
            log10_a0=la0_pbg,
            low_accel_cut_log10_gbar=float(args.low_accel_cut),
        )
        candidate_pbg["pbg_a0"] = {"H0P_source": h0p_src, "kappa": float(kappa), "a0_m_s2": float(a0_pbg), "log10_a0_m_s2": float(la0_pbg)}
    except Exception as e:
        candidate_pbg = {"status": "failed", "reason": str(e), "h0p_source": {"path": _rel(Path(args.h0p_metrics))}}

    freeze_test_path = Path(args.freeze_test_metrics)
    freeze_test_summary: Dict[str, Any] = {"status": "skipped"}
    if freeze_test_path.exists():
        freeze_test_summary = _summarize_freeze_test_metrics(_read_json(freeze_test_path))
        freeze_test_summary["metrics_path"] = _rel(freeze_test_path)
    else:
        freeze_test_summary = {"status": "missing", "metrics_path": _rel(freeze_test_path)}

    upsilon_fit_path = Path(args.freeze_test_upsilon_fit_metrics)
    upsilon_fit_summary: Dict[str, Any] = {"status": "skipped"}
    if upsilon_fit_path.exists():
        upsilon_fit_summary = _summarize_freeze_test_upsilon_fit_metrics(_read_json(upsilon_fit_path))
        upsilon_fit_summary["metrics_path"] = _rel(upsilon_fit_path)
    else:
        upsilon_fit_summary = {"status": "missing", "metrics_path": _rel(upsilon_fit_path)}

    proc_sweep_path = Path(args.freeze_test_procedure_sweep_metrics)
    proc_sweep_summary: Dict[str, Any] = {"status": "skipped"}
    if proc_sweep_path.exists():
        proc_sweep_summary = _summarize_freeze_test_procedure_sweep_metrics(_read_json(proc_sweep_path))
        proc_sweep_summary["metrics_path"] = _rel(proc_sweep_path)
    else:
        proc_sweep_summary = {"status": "missing", "metrics_path": _rel(proc_sweep_path)}

    mlr_sweep_path = Path(args.freeze_test_mlr_sweep_metrics)
    mlr_sweep_summary: Dict[str, Any] = {"status": "skipped"}
    if mlr_sweep_path.exists():
        mlr_sweep_summary = _summarize_freeze_test_mlr_sweep_metrics(_read_json(mlr_sweep_path))
        mlr_sweep_summary["metrics_path"] = _rel(mlr_sweep_path)
    else:
        mlr_sweep_summary = {"status": "missing", "metrics_path": _rel(mlr_sweep_path)}

    adoption_eval: Dict[str, Any] = {"status": "skipped"}
    adoption_eval_galaxy: Dict[str, Any] = {"status": "skipped"}
    if isinstance(freeze_test_summary, dict) and freeze_test_summary.get("status") == "ok":
        ss = freeze_test_summary.get("sweep_summary", {})
        if isinstance(ss, dict):
            adoption_eval = _adoption_eval_from_sweep_summary(ss, pass_rate_required=0.95, threshold_abs_z=3.0, min_runs=100)
        ss_gal = freeze_test_summary.get("sweep_summary_galaxy", {})
        if isinstance(ss_gal, dict):
            adoption_eval_galaxy = _adoption_eval_from_sweep_summary(ss_gal, pass_rate_required=0.95, threshold_abs_z=3.0, min_runs=100)

    # Combine stability gate (split dependence), M/L robustness gate, and procedure robustness gate
    # into a single "final" adopted flag.
    adopted_fixed_ml = None
    robust_adopted_mlr = None
    try:
        cand_fixed = (adoption_eval_galaxy.get("models", {}) if isinstance(adoption_eval_galaxy.get("models", {}), dict) else {}).get("candidate_rar_pbg_a0_fixed_kappa", {})
        if isinstance(cand_fixed, dict) and "adopted" in cand_fixed:
            adopted_fixed_ml = bool(cand_fixed.get("adopted"))
    except Exception:
        adopted_fixed_ml = None
    try:
        if isinstance(mlr_sweep_summary, dict) and isinstance(mlr_sweep_summary.get("candidate"), dict):
            ra = mlr_sweep_summary["candidate"].get("robust_adopted")
            if isinstance(ra, bool):
                robust_adopted_mlr = ra
    except Exception:
        robust_adopted_mlr = None

    robust_adopted_procedure = None
    try:
        if isinstance(proc_sweep_summary, dict) and isinstance(proc_sweep_summary.get("candidate"), dict):
            ra = proc_sweep_summary["candidate"].get("robust_adopted")
            if isinstance(ra, bool):
                robust_adopted_procedure = ra
    except Exception:
        robust_adopted_procedure = None

    adopted_final = None
    if isinstance(adopted_fixed_ml, bool) and isinstance(robust_adopted_mlr, bool) and isinstance(robust_adopted_procedure, bool):
        adopted_final = bool(adopted_fixed_ml and robust_adopted_mlr and robust_adopted_procedure)

    out = Path(args.out)
    pack: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "inputs": {
            "rar_csv": _rel(rar_csv),
            "bin_edges": [float(v) for v in edges.tolist()],
            "sigma_floor_dex": sigma_floor,
            "low_accel_cut_log10_gbar": float(args.low_accel_cut),
            "null_model": "baryons_only: log10(g_obs) - log10(g_bar) = 0",
            "h0p_source": h0p_src,
            "pbg_kappa": float(args.pbg_kappa),
            "freeze_test_metrics": _rel(freeze_test_path),
            "freeze_test_upsilon_fit_metrics": _rel(upsilon_fit_path),
            "freeze_test_procedure_sweep_metrics": _rel(proc_sweep_path),
            "freeze_test_mlr_sweep_metrics": _rel(mlr_sweep_path),
        },
        "counts": {"n_points": int(len(pts)), "n_low_accel": int(np.count_nonzero(m_low))},
        "summary": {
            "log10_gbar_range": [float(np.min(x)), float(np.max(x))],
            "log10_gobs_range": [float(np.min(y)), float(np.max(y))],
            "residual_dex_median": float(np.median(resid)),
            "residual_dex_rms": float(np.sqrt(np.mean(resid * resid))),
        },
        "binned_residuals": bins,
        "null_test_low_accel": {
            "domain": f"log10(g_bar) < {float(args.low_accel_cut):g}",
            "weighted_mean_residual_dex": mean_low,
            "sem_dex": sem_low,
            "z": z_low,
            "note": "Uses only σ(log10 g_obs) from V_obs errors; g_bar systematics (e.g., M/L) are not included.",
        },
        "reject": {
            "hypothesis": "baryons_only (g_P=g_bar) as a sufficient explanation",
            "threshold": {"z_low_accel": 3.0},
            "decision_rule": "Reject if z_low_accel >= 3 in the low-acceleration domain.",
        },
        "scenario_b": {
            "note": "Scenario (B) candidates are evaluated via fit→freeze→holdout (galaxy split) in sparc_rar_freeze_test_metrics.json. Candidates/baselines here are not treated as established P-model predictions.",
            "sigma_int_rule": "Estimate σ_int (dex) on train by solving chi2/dof≈1 with dof=n_used-1, then carry σ_int to holdout; evaluate low-accel z on holdout with σ_total^2=σ_y^2+σ_int^2.",
            "reject": {
                "threshold": {"abs_z_low_accel_holdout_with_sigma_int": 3.0},
                "decision_rule": "Reject a candidate if |z_low_accel| >= 3 on holdout when parameters (e.g., a0, κ) are frozen.",
            },
            "freeze_test_summary": freeze_test_summary,
            "systematics": {
                "note": "g_bar systematics (e.g., stellar M/L) change the input g_bar itself; this is evaluated separately via M/L sweep using the same freeze-test split grid.",
                "upsilon_fit_on_train": upsilon_fit_summary,
                "procedure_sweep": proc_sweep_summary,
                "mlr_sweep": mlr_sweep_summary,
            },
            "adoption": {
                "note": "This is a stability criterion against split-dependence; it is stricter than the per-split reject rule.",
                "evaluation": adoption_eval,
                "evaluation_galaxy": adoption_eval_galaxy,
                "preferred_metric": "sweep_summary_galaxy",
                "final": {
                    "decision_rule": "adopted_final := adopted_fixed_ml && robust_adopted_mlr && robust_adopted_procedure",
                    "adopted_fixed_ml": adopted_fixed_ml,
                    "robust_adopted_mlr": robust_adopted_mlr,
                    "robust_adopted_procedure": robust_adopted_procedure,
                    "adopted_final": adopted_final,
                    "note": "robust_adopted_mlr is computed from sparc_rar_freeze_test_mlr_sweep_metrics.json over the specified (upsilon_disk, upsilon_bulge) grid; robust_adopted_procedure is computed from sparc_rar_freeze_test_procedure_sweep_metrics.json over the specified procedure grid.",
                },
            },
        },
        "baselines": {
            "note": "Baselines are included for comparison; they are not treated as P-model predictions.",
            "rar_mcgaugh2016_fit_a0": baseline_rar_fit,
            "rar_mcgaugh2016_a0_pbg_fixed_kappa": candidate_pbg,
        },
    }
    _write_json(out, pack)

    if worklog is not None:
        try:
            worklog.append_event(
                "cosmology.sparc_falsification_pack",
                {"pack": _rel(out), "rar_csv": _rel(rar_csv), "n_points": int(len(pts))},
            )
        except Exception:
            pass

    print(json.dumps({"pack": _rel(out), "n_points": int(len(pts))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
