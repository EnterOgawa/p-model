#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparc_rar_mlr_sweep.py

Phase 6 / Step 6.5（SPARC：RAR）:
RAR 再構築（g_obs/g_bar）の系統要因として、恒星 M/L（Υ_disk, Υ_bulge）を sweep し、
baryons-only ヌル（g_P=g_bar）の残差・低加速度領域の統計量がどれだけ動くかを固定出力化する。

入力（一次）:
- data/cosmology/sparc/raw/Rotmod_LTG.zip

出力（固定）:
- output/cosmology/sparc_rar_mlr_sweep_metrics.json

注意:
- g_bar 系統（M/L）だけを動かす最小 sweep。距離・inclination・gas の系統などは次段で扱う。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cosmology.sparc_rar_from_rotmod import _compute_accel_points, _parse_rotmod_text  # noqa: E402
from scripts.cosmology.sparc_falsification_pack import (  # noqa: E402
    C_LIGHT_M_S,
    DEFAULT_PBG_KAPPA,
    _fit_log10_a0_grid,
    _get_h0p_si,
    _rar_mcgaugh2016_log10_pred,
)

try:
    from scripts.summary import worklog  # type: ignore # noqa: E402
except Exception:  # pragma: no cover
    worklog = None


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


def _baryons_only_stats(
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    sg_obs: np.ndarray,
    *,
    sigma_floor_dex: float,
    low_accel_cut_log10_gbar: float,
) -> Dict[str, Any]:
    x = np.log10(g_bar)
    y = np.log10(g_obs)
    resid = y - x
    sigma_y = sg_obs / (g_obs * math.log(10.0))
    sigma_floor = float(max(sigma_floor_dex, 1e-6))
    sigma_y = np.where(np.isfinite(sigma_y) & (sigma_y > 0.0), sigma_y, np.nan)
    sigma_y = np.where(np.isfinite(sigma_y) & (sigma_y < sigma_floor), sigma_floor, sigma_y)

    m_low = np.isfinite(resid) & np.isfinite(sigma_y) & (x < float(low_accel_cut_log10_gbar))
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

    return {
        "n_points": int(g_bar.size),
        "n_low_accel": int(np.count_nonzero(m_low)),
        "residual_dex_median": float(np.median(resid)),
        "residual_dex_rms": float(np.sqrt(np.mean(resid * resid))),
        "low_accel": {"cut_log10_gbar": float(low_accel_cut_log10_gbar), "weighted_mean_dex": mean_low, "sem_dex": sem_low, "z": z_low},
    }


def _rar_fixed_a0_stats(
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    sg_obs: np.ndarray,
    *,
    log10_a0: float,
    sigma_floor_dex: float,
    low_accel_cut_log10_gbar: float,
) -> Dict[str, Any]:
    x = np.log10(g_bar)
    y = np.log10(g_obs)
    y_pred = _rar_mcgaugh2016_log10_pred(g_bar, log10_a0=float(log10_a0))
    resid = y - y_pred
    sigma_y = sg_obs / (g_obs * math.log(10.0))
    sigma_floor = float(max(sigma_floor_dex, 1e-6))
    sigma_y = np.where(np.isfinite(sigma_y) & (sigma_y > 0.0), sigma_y, np.nan)
    sigma_y = np.where(np.isfinite(sigma_y) & (sigma_y < sigma_floor), sigma_floor, sigma_y)

    m_low = np.isfinite(resid) & np.isfinite(sigma_y) & (x < float(low_accel_cut_log10_gbar))
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

    return {
        "status": "ok",
        "n_points": int(g_bar.size),
        "n_low_accel": int(np.count_nonzero(m_low)),
        "log10_a0_m_s2": float(log10_a0),
        "a0_m_s2": float(10.0 ** float(log10_a0)),
        "residual_dex_median": float(np.median(resid)),
        "residual_dex_rms": float(np.sqrt(np.mean(resid * resid))),
        "low_accel": {"cut_log10_gbar": float(low_accel_cut_log10_gbar), "weighted_mean_dex": mean_low, "sem_dex": sem_low, "z": z_low},
    }


def _collect_points_from_zip(
    rotmod_zip: Path,
    *,
    upsilon_disk: float,
    upsilon_bulge: float,
) -> Dict[str, Any]:
    points: List[Dict[str, Any]] = []
    n_gal = 0
    with zipfile.ZipFile(rotmod_zip, "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith("_rotmod.dat")]
        names.sort()
        for n in names:
            galaxy = Path(n).name.replace("_rotmod.dat", "")
            text = zf.read(n).decode("utf-8", errors="replace").splitlines()
            dist_mpc, rows = _parse_rotmod_text(text)
            pts = _compute_accel_points(
                galaxy,
                dist_mpc,
                rows,
                upsilon_disk=float(upsilon_disk),
                upsilon_bulge=float(upsilon_bulge),
            )
            points.extend(pts)
            n_gal += 1

    # Keep only positive accelerations
    g_bar = np.asarray([float(p["g_bar_m_s2"]) for p in points], dtype=float)
    g_obs = np.asarray([float(p["g_obs_m_s2"]) for p in points], dtype=float)
    sg_obs = np.asarray([float(p["g_obs_sigma_m_s2"]) for p in points], dtype=float)
    m = np.isfinite(g_bar) & np.isfinite(g_obs) & (g_bar > 0.0) & (g_obs > 0.0)
    return {"n_galaxies": int(n_gal), "g_bar": g_bar[m], "g_obs": g_obs[m], "sg_obs": sg_obs[m]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rotmod-zip",
        default=str(_ROOT / "data" / "cosmology" / "sparc" / "raw" / "Rotmod_LTG.zip"),
        help="Path to Rotmod_LTG.zip",
    )
    p.add_argument(
        "--upsilon-disk",
        action="append",
        type=float,
        default=[],
        help="Disk M/L values to sweep (repeatable). Default: 0.4,0.5,0.6",
    )
    p.add_argument(
        "--upsilon-bulge",
        action="append",
        type=float,
        default=[],
        help="Bulge M/L values to sweep (repeatable). Default: 0.5,0.7,0.9",
    )
    p.add_argument("--sigma-floor-dex", type=float, default=0.01, help="Floor for sigma(log10 g_obs) in dex (default: 0.01)")
    p.add_argument(
        "--low-accel-cut",
        type=float,
        default=-10.5,
        help="Low-acceleration domain cut: log10(g_bar) < cut (default: -10.5)",
    )
    p.add_argument(
        "--h0p-metrics",
        default=str(_ROOT / "output" / "cosmology" / "cosmology_redshift_pbg_metrics.json"),
        help="Path to cosmology_redshift_pbg_metrics.json (default: output/cosmology/cosmology_redshift_pbg_metrics.json)",
    )
    p.add_argument("--h0p-km-s-mpc", type=float, default=None, help="Override H0^(P) in km/s/Mpc (optional)")
    p.add_argument("--pbg-kappa", type=float, default=DEFAULT_PBG_KAPPA, help="a0 = kappa * c * H0^(P) (default: 1/(2π))")
    p.add_argument(
        "--out",
        default=str(_ROOT / "output" / "cosmology" / "sparc_rar_mlr_sweep_metrics.json"),
        help="Output JSON path",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    rotmod_zip = Path(args.rotmod_zip)
    if not rotmod_zip.exists():
        raise FileNotFoundError(f"missing Rotmod zip: {rotmod_zip}")

    ud_list = [float(x) for x in (args.upsilon_disk or [])] or [0.4, 0.5, 0.6]
    ub_list = [float(x) for x in (args.upsilon_bulge or [])] or [0.5, 0.7, 0.9]

    pbg_a0: Dict[str, Any] = {"status": "skipped"}
    la0_pbg = float("nan")
    try:
        h0p_si, h0p_src = _get_h0p_si(h0p_metrics=Path(args.h0p_metrics), h0p_km_s_mpc_override=args.h0p_km_s_mpc)
        kappa = float(args.pbg_kappa)
        if not np.isfinite(kappa) or kappa <= 0:
            raise ValueError("--pbg-kappa must be positive")
        a0_pbg = float(kappa) * float(C_LIGHT_M_S) * float(h0p_si)
        if not np.isfinite(a0_pbg) or a0_pbg <= 0:
            raise ValueError("invalid a0 computed from kappa*c*H0^(P)")
        la0_pbg = float(math.log10(a0_pbg))
        pbg_a0 = {"status": "ok", "H0P_source": h0p_src, "kappa": float(kappa), "a0_m_s2": float(a0_pbg), "log10_a0_m_s2": float(la0_pbg)}
    except Exception as e:
        pbg_a0 = {"status": "failed", "reason": str(e), "path": _rel(Path(args.h0p_metrics))}

    variants: List[Dict[str, Any]] = []
    env = {
        "baryons_only_low_accel_weighted_mean_dex": [float("nan"), float("nan")],
        "baryons_only_low_accel_z": [float("nan"), float("nan")],
        "rar_fit_a0_best_m_s2": [float("nan"), float("nan")],
        "candidate_pbg_low_accel_weighted_mean_dex": [float("nan"), float("nan")],
        "candidate_pbg_low_accel_z": [float("nan"), float("nan")],
    }

    for ud in ud_list:
        for ub in ub_list:
            pts = _collect_points_from_zip(rotmod_zip, upsilon_disk=ud, upsilon_bulge=ub)
            g_bar = pts["g_bar"]
            g_obs = pts["g_obs"]
            sg_obs = pts["sg_obs"]

            stats = _baryons_only_stats(
                g_bar,
                g_obs,
                sg_obs,
                sigma_floor_dex=float(args.sigma_floor_dex),
                low_accel_cut_log10_gbar=float(args.low_accel_cut),
            )

            # Baseline RAR fit (McGaugh+2016 empirical) for comparison
            x = np.log10(g_bar)
            y = np.log10(g_obs)
            sigma_y = sg_obs / (g_obs * math.log(10.0))
            sigma_floor = float(max(float(args.sigma_floor_dex), 1e-6))
            sigma_y = np.where(np.isfinite(sigma_y) & (sigma_y > 0.0), sigma_y, np.nan)
            sigma_y = np.where(np.isfinite(sigma_y) & (sigma_y < sigma_floor), sigma_floor, sigma_y)
            rar_fit = _fit_log10_a0_grid(g_bar, y, sigma_y)
            cand_stats: Dict[str, Any] = {"status": "skipped"}
            if np.isfinite(la0_pbg):
                cand_stats = _rar_fixed_a0_stats(
                    g_bar,
                    g_obs,
                    sg_obs,
                    log10_a0=float(la0_pbg),
                    sigma_floor_dex=float(args.sigma_floor_dex),
                    low_accel_cut_log10_gbar=float(args.low_accel_cut),
                )

            v = {
                "upsilon_disk": float(ud),
                "upsilon_bulge": float(ub),
                "counts": {"n_galaxies": int(pts["n_galaxies"]), "n_points": int(g_bar.size)},
                "baryons_only": stats,
                "baseline_rar_mcgaugh2016": rar_fit,
                "candidate_rar_pbg_fixed_kappa": cand_stats,
            }
            variants.append(v)

            # envelope update
            mean_low = float(stats["low_accel"]["weighted_mean_dex"])
            z_low = float(stats["low_accel"]["z"])
            if np.isfinite(mean_low):
                lo, hi = env["baryons_only_low_accel_weighted_mean_dex"]
                env["baryons_only_low_accel_weighted_mean_dex"] = [min(lo, mean_low) if np.isfinite(lo) else mean_low, max(hi, mean_low) if np.isfinite(hi) else mean_low]
            if np.isfinite(z_low):
                lo, hi = env["baryons_only_low_accel_z"]
                env["baryons_only_low_accel_z"] = [min(lo, z_low) if np.isfinite(lo) else z_low, max(hi, z_low) if np.isfinite(hi) else z_low]
            a0_best = float(rar_fit.get("a0_best_m_s2") or float("nan"))
            if np.isfinite(a0_best):
                lo, hi = env["rar_fit_a0_best_m_s2"]
                env["rar_fit_a0_best_m_s2"] = [min(lo, a0_best) if np.isfinite(lo) else a0_best, max(hi, a0_best) if np.isfinite(hi) else a0_best]
            if isinstance(cand_stats, dict) and cand_stats.get("status") == "ok":
                c_low = cand_stats.get("low_accel") if isinstance(cand_stats.get("low_accel"), dict) else {}
                c_mean = float(c_low.get("weighted_mean_dex") or float("nan"))
                c_z = float(c_low.get("z") or float("nan"))
                if np.isfinite(c_mean):
                    lo, hi = env["candidate_pbg_low_accel_weighted_mean_dex"]
                    env["candidate_pbg_low_accel_weighted_mean_dex"] = [min(lo, c_mean) if np.isfinite(lo) else c_mean, max(hi, c_mean) if np.isfinite(hi) else c_mean]
                if np.isfinite(c_z):
                    lo, hi = env["candidate_pbg_low_accel_z"]
                    env["candidate_pbg_low_accel_z"] = [min(lo, c_z) if np.isfinite(lo) else c_z, max(hi, c_z) if np.isfinite(hi) else c_z]

    out = Path(args.out)
    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "inputs": {
            "rotmod_zip": _rel(rotmod_zip),
            "upsilon_disk": ud_list,
            "upsilon_bulge": ub_list,
            "sigma_floor_dex": float(args.sigma_floor_dex),
            "low_accel_cut_log10_gbar": float(args.low_accel_cut),
            "pbg_a0": pbg_a0,
            "note": "vdisk and vbul in Rotmod are scaled by sqrt(Υ) (same convention as sparc_rar_from_rotmod.py).",
        },
        "variants": variants,
        "envelope": env,
    }
    _write_json(out, payload)

    if worklog is not None:
        try:
            worklog.append_event("cosmology.sparc_rar_mlr_sweep", {"metrics": _rel(out), "n_variants": int(len(variants))})
        except Exception:
            pass

    print(json.dumps({"metrics": _rel(out), "n_variants": len(variants)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
