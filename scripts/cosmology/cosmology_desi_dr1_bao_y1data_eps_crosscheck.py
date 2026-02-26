# -*- coding: utf-8 -*-
"""
cosmology_desi_dr1_bao_y1data_eps_crosscheck.py

Phase 4 / Step 4.5B（宇宙論：BAO一次統計の確証決定打）補助:
DESI DR1 BAO 論文（VI）の公開距離制約表（Y1data: D_M/r_d, D_H/r_d, corr）から
F_AP=D_M/D_H を組み立て、AP warping ε を計算する。

目的:
  - catalog-based ξℓ→peakfit（cov代替：sky jackknife; dv+cov）で得た ε と、
    公開距離制約（distance constraints; 派生値）から計算した ε_expected を同じ指標で比較する。
  - cov を使う比較（D_M/D_Hの2×2）なので、diag 依存の位置づけを明確にする。
  - 公式公開の mean/cov（`CobayaSampler/bao_data` の `desi_2024_gaussian_bao_*`）が手元にある場合は、
    Y1data（TeX）と数値整合をチェックし、必要なら “参照値” として利用できるようにする。

入出力:
  - 入力:
    - data/cosmology/desi_dr1_bao_y1data.json
    - data/cosmology/desi_dr1_bao_bao_data.json（任意：公式 mean/cov）
    - output/private/cosmology/cosmology_bao_catalog_peakfit_lrg_combined__*.json（metrics）
  - 出力:
    - output/private/cosmology/cosmology_desi_dr1_bao_y1data_eps_crosscheck__{out_tag}.png
    - output/private/cosmology/cosmology_desi_dr1_bao_y1data_eps_crosscheck__{out_tag}_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402

# Reuse the verified F_AP/eps helpers from the peakfit module.
from scripts.cosmology.cosmology_bao_xi_multipole_peakfit import (  # noqa: E402
    _eps_from_f_ap_ratio as _eps_from_f_ap_ratio,
    _f_ap_lcdm_flat as _f_ap_lcdm_flat,
    _f_ap_pbg_exponential as _f_ap_pbg_exponential,
)


@dataclass(frozen=True)
class _EpsPoint:
    tracer: str
    z_eff: float
    eps_mean: float
    eps_sigma: float


@dataclass(frozen=True)
class _DmDh:
    z_eff: float
    dm_mean: float
    dm_sigma: float
    dh_mean: float
    dh_sigma: float
    corr: float


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mc_eps_from_dm_dh(
    *,
    dm_mean: float,
    dm_sigma: float,
    dh_mean: float,
    dh_sigma: float,
    corr: float,
    f_ap_fid: float,
    n_draws: int,
    seed: int,
) -> _EpsPoint:
    mu = np.array([float(dm_mean), float(dh_mean)], dtype=float)
    sig = np.array([float(dm_sigma), float(dh_sigma)], dtype=float)
    corr = float(corr)
    cov = np.array(
        [
            [sig[0] * sig[0], corr * sig[0] * sig[1]],
            [corr * sig[0] * sig[1], sig[1] * sig[1]],
        ],
        dtype=float,
    )
    rng = np.random.default_rng(int(seed))
    draws = rng.multivariate_normal(mu, cov, size=int(n_draws))
    dm = draws[:, 0]
    dh = draws[:, 1]
    mask = (dm > 0.0) & (dh > 0.0) & np.isfinite(dm) & np.isfinite(dh)
    dm = dm[mask]
    dh = dh[mask]
    f_ap_obs = dm / dh
    eps = (float(f_ap_fid) / f_ap_obs) ** (1.0 / 3.0) - 1.0
    return float(np.mean(eps)), float(np.std(eps, ddof=1))


def _sigma_from_ci(ci: List[float]) -> float:
    try:
        lo, hi = float(ci[0]), float(ci[1])
        # 条件分岐: `math.isfinite(lo) and math.isfinite(hi)` を満たす経路を評価する。
        if math.isfinite(lo) and math.isfinite(hi):
            return 0.5 * abs(hi - lo)
    except Exception:
        pass

    return float("nan")


def _extract_peakfit_eps(points: List[Dict[str, Any]], *, dist: str, z_target: float, z_tol: float) -> Optional[_EpsPoint]:
    dist = str(dist)
    best: Optional[Dict[str, Any]] = None
    best_dz = float("inf")
    for p in points:
        # 条件分岐: `str(p.get("dist")) != dist` を満たす経路を評価する。
        if str(p.get("dist")) != dist:
            continue

        z_eff = float(p.get("z_eff"))
        dz = abs(z_eff - float(z_target))
        # 条件分岐: `dz < best_dz` を満たす経路を評価する。
        if dz < best_dz:
            best_dz = dz
            best = p

    # 条件分岐: `best is None or best_dz > float(z_tol)` を満たす経路を評価する。

    if best is None or best_dz > float(z_tol):
        return None

    eps = float(best["fit"]["free"]["eps"])
    ci = best["fit"]["free"].get("eps_ci_1sigma", None)
    sig = _sigma_from_ci(ci) if isinstance(ci, list) else float("nan")
    return _EpsPoint(tracer="(from_peakfit)", z_eff=float(best["z_eff"]), eps_mean=eps, eps_sigma=sig)


def _extract_wedge_eps_proxy(
    wedge_metrics: Dict[str, Any],
    *,
    dist: str,
    z_target: float,
    z_tol: float,
    sample: str,
    caps: str,
) -> Optional[_EpsPoint]:
    groups = wedge_metrics.get("groups", {}) if isinstance(wedge_metrics.get("groups", {}), dict) else {}
    key = f"{str(sample).strip().lower()}/{str(caps).strip().lower()}/{str(dist).strip().lower()}"
    g = groups.get(key)
    # 条件分岐: `not isinstance(g, dict)` を満たす経路を評価する。
    if not isinstance(g, dict):
        return None

    pts = g.get("points", [])
    # 条件分岐: `not isinstance(pts, list)` を満たす経路を評価する。
    if not isinstance(pts, list):
        return None

    best: Optional[Dict[str, Any]] = None
    best_dz = float("inf")
    for p in pts:
        try:
            z_eff = float(p.get("z_eff"))
            dz = abs(z_eff - float(z_target))
            # 条件分岐: `dz < best_dz` を満たす経路を評価する。
            if dz < best_dz:
                best_dz = dz
                best = p
        except Exception:
            continue

    # 条件分岐: `best is None or best_dz > float(z_tol)` を満たす経路を評価する。

    if best is None or best_dz > float(z_tol):
        return None

    try:
        eps = float(best.get("eps_proxy"))
        return _EpsPoint(tracer="(from_wedge)", z_eff=float(best.get("z_eff")), eps_mean=eps, eps_sigma=float("nan"))
    except Exception:
        return None


def _extract_dm_dh_from_y1data_row(r: Dict[str, Any]) -> _DmDh:
    z_eff = float(r["z_eff"])
    dm = r["dm_over_rd"]
    dh = r["dh_over_rd"]
    corr = float(r.get("corr_r_dm_dh", 0.0))
    # 条件分岐: `dm is None or dh is None` を満たす経路を評価する。
    if dm is None or dh is None:
        raise ValueError("row does not contain dm/dh (expected for LRG1/LRG2)")

    return _DmDh(
        z_eff=z_eff,
        dm_mean=float(dm["mean"]),
        dm_sigma=float(dm["sigma"]),
        dh_mean=float(dh["mean"]),
        dh_sigma=float(dh["sigma"]),
        corr=corr,
    )


def _extract_dm_dh_from_bao_data(bao: Dict[str, Any], *, dataset_name: str) -> _DmDh:
    ds = None
    for d in bao.get("datasets", []):
        # 条件分岐: `isinstance(d, dict) and str(d.get("name")) == str(dataset_name)` を満たす経路を評価する。
        if isinstance(d, dict) and str(d.get("name")) == str(dataset_name):
            ds = d
            break

    # 条件分岐: `not isinstance(ds, dict)` を満たす経路を評価する。

    if not isinstance(ds, dict):
        raise ValueError(f"bao_data missing dataset: {dataset_name}")

    z_eff = ds.get("z_eff", None)
    # 条件分岐: `z_eff is None` を満たす経路を評価する。
    if z_eff is None:
        raise ValueError(f"bao_data dataset has no single z_eff: {dataset_name}")

    z_eff = float(z_eff)
    qs = ds.get("quantities", [])
    mean = ds.get("mean", [])
    cov = ds.get("cov", [])
    # 条件分岐: `not (isinstance(qs, list) and isinstance(mean, list) and isinstance(cov, list))` を満たす経路を評価する。
    if not (isinstance(qs, list) and isinstance(mean, list) and isinstance(cov, list)):
        raise ValueError(f"bao_data dataset malformed: {dataset_name}")

    # 条件分岐: `len(qs) != len(mean) or len(qs) != len(cov)` を満たす経路を評価する。

    if len(qs) != len(mean) or len(qs) != len(cov):
        raise ValueError(f"bao_data dataset size mismatch: {dataset_name}")

    q_to_i = {str(q): i for i, q in enumerate(qs)}
    # 条件分岐: `"DM_over_rs" not in q_to_i or "DH_over_rs" not in q_to_i` を満たす経路を評価する。
    if "DM_over_rs" not in q_to_i or "DH_over_rs" not in q_to_i:
        raise ValueError(f"bao_data dataset missing DM_over_rs/DH_over_rs: {dataset_name} qs={qs}")

    i_dm = int(q_to_i["DM_over_rs"])
    i_dh = int(q_to_i["DH_over_rs"])
    dm_mean = float(mean[i_dm])
    dh_mean = float(mean[i_dh])
    dm_var = float(cov[i_dm][i_dm])
    dh_var = float(cov[i_dh][i_dh])
    dm_sig = float(np.sqrt(dm_var))
    dh_sig = float(np.sqrt(dh_var))
    cov12 = float(cov[i_dm][i_dh])
    corr = cov12 / (dm_sig * dh_sig) if (dm_sig > 0.0 and dh_sig > 0.0) else float("nan")
    return _DmDh(z_eff=z_eff, dm_mean=dm_mean, dm_sigma=dm_sig, dh_mean=dh_mean, dh_sigma=dh_sig, corr=corr)


def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = [
            "Yu Gothic",
            "Meiryo",
            "BIZ UDGothic",
            "MS Gothic",
            "Yu Mincho",
            "MS Mincho",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="DESI DR1 BAO Y1data -> eps cross-check vs catalog peakfit.")
    ap.add_argument(
        "--y1data-json",
        default=str(_ROOT / "data" / "cosmology" / "desi_dr1_bao_y1data.json"),
        help="Path to cached Y1data JSON (default: data/cosmology/desi_dr1_bao_y1data.json)",
    )
    ap.add_argument(
        "--bao-data-json",
        default=str(_ROOT / "data" / "cosmology" / "desi_dr1_bao_bao_data.json"),
        help="Optional: path to cached bao_data JSON (default: data/cosmology/desi_dr1_bao_bao_data.json)",
    )
    ap.add_argument(
        "--prefer-bao-data",
        action="store_true",
        help="Use bao_data (DM_over_rs/DH_over_rs + full cov) as primary source instead of Y1data table.",
    )
    ap.add_argument(
        "--peakfit-metrics-json",
        default=str(
            _ROOT
            / "output"
            / "cosmology"
            / "cosmology_bao_catalog_peakfit_lrg_combined__w_desi_default_ms_off_y1bins__jk_cov_both_metrics.json"
        ),
        help="Path to catalog peakfit metrics JSON (default: LRG combined y1bins out_tag)",
    )
    ap.add_argument("--wedge-metrics-json", default="", help="Optional: wedge anisotropy metrics JSON to add ε_proxy points.")
    ap.add_argument("--out-tag", default="w_desi_default_ms_off_y1bins", help="Output tag suffix (default matches pipeline)")
    ap.add_argument(
        "--tracers",
        type=str,
        default="LRG1,LRG2",
        help=(
            "Comma-separated tracer list from Y1data to cross-check (default: LRG1,LRG2). "
            "Note: this script requires DM/DH (anisotropic) constraints, so DV-only tracers (e.g., BGS, QSO) are not supported."
        ),
    )
    ap.add_argument("--lcdm-omega-m", type=float, default=0.315, help="LCDM Omega_m used for fid F_AP (default: 0.315)")
    ap.add_argument("--lcdm-n-grid", type=int, default=4000, help="LCDM z integral grid for fid F_AP (default: 4000)")
    ap.add_argument("--n-draws", type=int, default=120000, help="MC draws for eps sigma (default: 120000)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    ap.add_argument("--z-tol", type=float, default=0.02, help="Match tolerance between Y1data z_eff and peakfit z_eff (default: 0.02)")
    args = ap.parse_args()

    y1_path = Path(args.y1data_json)
    bao_path = Path(args.bao_data_json)
    peakfit_path = Path(args.peakfit_metrics_json)
    wedge_path = Path(args.wedge_metrics_json) if str(args.wedge_metrics_json).strip() else None
    # 条件分岐: `not y1_path.exists()` を満たす経路を評価する。
    if not y1_path.exists():
        raise SystemExit(f"missing y1data json: {y1_path}")

    bao: Optional[Dict[str, Any]] = None
    # 条件分岐: `bao_path.exists()` を満たす経路を評価する。
    if bao_path.exists():
        try:
            bao_any = _load_json(bao_path)
            bao = bao_any if isinstance(bao_any, dict) else None
        except Exception:
            bao = None

    # 条件分岐: `not peakfit_path.exists()` を満たす経路を評価する。

    if not peakfit_path.exists():
        raise SystemExit(f"missing peakfit metrics json: {peakfit_path}")

    # 条件分岐: `wedge_path is not None and not wedge_path.exists()` を満たす経路を評価する。

    if wedge_path is not None and not wedge_path.exists():
        raise SystemExit(f"missing wedge metrics json: {wedge_path}")

    y1 = _load_json(y1_path)
    peakfit = _load_json(peakfit_path)
    wedge_metrics: Optional[Dict[str, Any]] = _load_json(wedge_path) if wedge_path is not None else None
    peakfit_points = peakfit.get("results", [])
    # 条件分岐: `not isinstance(peakfit_points, list)` を満たす経路を評価する。
    if not isinstance(peakfit_points, list):
        raise SystemExit("peakfit metrics has invalid 'results' (expected list)")

    rows = [r for r in y1.get("rows", []) if isinstance(r, dict)]
    y1_by_tracer: Dict[str, Dict[str, Any]] = {str(r.get("tracer")): r for r in rows if str(r.get("tracer", "")).strip()}

    tracers = [t.strip() for t in str(args.tracers).split(",") if t.strip()]
    # 条件分岐: `not tracers` を満たす経路を評価する。
    if not tracers:
        raise SystemExit("--tracers must not be empty")

    missing = [t for t in tracers if t not in y1_by_tracer]
    # 条件分岐: `missing` を満たす経路を評価する。
    if missing:
        raise SystemExit(f"y1data missing tracers: {missing}")

    out_dir = _ROOT / "output" / "private" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tag = str(args.out_tag).strip() or "none"
    out_png = out_dir / f"cosmology_desi_dr1_bao_y1data_eps_crosscheck__{out_tag}.png"
    out_json = out_dir / f"cosmology_desi_dr1_bao_y1data_eps_crosscheck__{out_tag}_metrics.json"

    omega_m = float(args.lcdm_omega_m)
    n_grid = int(args.lcdm_n_grid)

    expected: Dict[str, Dict[str, _EpsPoint]] = {}
    fit: Dict[str, Dict[str, _EpsPoint]] = {}
    deltas: Dict[str, Dict[str, Dict[str, float]]] = {}
    source_consistency: Dict[str, Any] = {}

    bao_dataset_map = {
        "LRG1": "desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6",
        "LRG2": "desi_2024_gaussian_bao_LRG_GCcomb_z0.6-0.8",
        "LRG3+ELG1": "desi_2024_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1",
        "ELG2": "desi_2024_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6",
        "Lya QSO": "desi_2024_gaussian_bao_Lya_GCcomb",
    }

    for tracer in tracers:
        r = y1_by_tracer[tracer]
        # 条件分岐: `r.get("dm_over_rd") in (None, "null") or r.get("dh_over_rd") in (None, "null")` を満たす経路を評価する。
        if r.get("dm_over_rd") in (None, "null") or r.get("dh_over_rd") in (None, "null"):
            raise SystemExit(
                f"tracer '{tracer}' does not provide DM/DH in Y1data (likely DV-only). "
                "This eps cross-check requires anisotropic (DM/DH) constraints."
            )

        dm_dh_y1 = _extract_dm_dh_from_y1data_row(r)
        dm_dh = dm_dh_y1
        source_used = "y1data"
        # 条件分岐: `bool(args.prefer_bao_data) and bao is not None` を満たす経路を評価する。
        if bool(args.prefer_bao_data) and bao is not None:
            # 条件分岐: `tracer not in bao_dataset_map` を満たす経路を評価する。
            if tracer not in bao_dataset_map:
                raise SystemExit(f"bao_data dataset mapping missing for tracer '{tracer}' (update bao_dataset_map).")

            dm_dh = _extract_dm_dh_from_bao_data(bao, dataset_name=str(bao_dataset_map[tracer]))
            source_used = "bao_data"

        # Always compute and record y1data vs bao_data consistency if possible.

        if bao is not None:
            try:
                # 条件分岐: `tracer in bao_dataset_map` を満たす経路を評価する。
                if tracer in bao_dataset_map:
                    dm_dh_bao = _extract_dm_dh_from_bao_data(bao, dataset_name=str(bao_dataset_map[tracer]))
                else:
                    dm_dh_bao = None
                    raise ValueError(f"bao_dataset_map missing for tracer '{tracer}'")

                source_consistency[tracer] = {
                    "y1data": {
                        "z_eff": dm_dh_y1.z_eff,
                        "dm_mean": dm_dh_y1.dm_mean,
                        "dm_sigma": dm_dh_y1.dm_sigma,
                        "dh_mean": dm_dh_y1.dh_mean,
                        "dh_sigma": dm_dh_y1.dh_sigma,
                        "corr": dm_dh_y1.corr,
                    },
                    "bao_data": {
                        "z_eff": dm_dh_bao.z_eff,
                        "dm_mean": dm_dh_bao.dm_mean,
                        "dm_sigma": dm_dh_bao.dm_sigma,
                        "dh_mean": dm_dh_bao.dh_mean,
                        "dh_sigma": dm_dh_bao.dh_sigma,
                        "corr": dm_dh_bao.corr,
                    },
                    "delta": {
                        "z_eff": float(dm_dh_y1.z_eff - dm_dh_bao.z_eff),
                        "dm_mean": float(dm_dh_y1.dm_mean - dm_dh_bao.dm_mean),
                        "dh_mean": float(dm_dh_y1.dh_mean - dm_dh_bao.dh_mean),
                    },
                }
            except Exception:
                pass

        z_eff = float(dm_dh.z_eff)
        dm_mean = float(dm_dh.dm_mean)
        dm_sig = float(dm_dh.dm_sigma)
        dh_mean = float(dm_dh.dh_mean)
        dh_sig = float(dm_dh.dh_sigma)
        corr = float(dm_dh.corr)

        f_obs = dm_mean / dh_mean
        f_fid_lcdm = _f_ap_lcdm_flat(z_eff, omega_m=omega_m, n_grid=n_grid)
        f_fid_pbg = _f_ap_pbg_exponential(z_eff)
        eps_exp_lcdm = _eps_from_f_ap_ratio(f_ap_model=f_obs, f_ap_fid=f_fid_lcdm)
        eps_exp_pbg = _eps_from_f_ap_ratio(f_ap_model=f_obs, f_ap_fid=f_fid_pbg)

        m_lcdm, s_lcdm = _mc_eps_from_dm_dh(
            dm_mean=dm_mean,
            dm_sigma=dm_sig,
            dh_mean=dh_mean,
            dh_sigma=dh_sig,
            corr=corr,
            f_ap_fid=f_fid_lcdm,
            n_draws=int(args.n_draws),
            seed=int(args.seed) + 1000 + (0 if tracer == "LRG1" else 1),
        )
        m_pbg, s_pbg = _mc_eps_from_dm_dh(
            dm_mean=dm_mean,
            dm_sigma=dm_sig,
            dh_mean=dh_mean,
            dh_sigma=dh_sig,
            corr=corr,
            f_ap_fid=f_fid_pbg,
            n_draws=int(args.n_draws),
            seed=int(args.seed) + 2000 + (0 if tracer == "LRG1" else 1),
        )

        expected[tracer] = {
            "lcdm": _EpsPoint(tracer=tracer, z_eff=z_eff, eps_mean=float(eps_exp_lcdm), eps_sigma=float(s_lcdm)),
            "pbg": _EpsPoint(tracer=tracer, z_eff=z_eff, eps_mean=float(eps_exp_pbg), eps_sigma=float(s_pbg)),
        }

        fit_lcdm = _extract_peakfit_eps(peakfit_points, dist="lcdm", z_target=z_eff, z_tol=float(args.z_tol))
        fit_pbg = _extract_peakfit_eps(peakfit_points, dist="pbg", z_target=z_eff, z_tol=float(args.z_tol))
        # 条件分岐: `fit_lcdm is None or fit_pbg is None` を満たす経路を評価する。
        if fit_lcdm is None or fit_pbg is None:
            raise SystemExit(
                f"peakfit metrics does not contain both lcdm/pbg points close to z_eff={z_eff:.3f} for tracer {tracer}"
            )

        fit[tracer] = {"lcdm": fit_lcdm, "pbg": fit_pbg}

        deltas[tracer] = {}
        for dist in ("lcdm", "pbg"):
            d = float(fit[tracer][dist].eps_mean) - float(expected[tracer][dist].eps_mean)
            sig_expected = float(expected[tracer][dist].eps_sigma)
            sig_fit = float(fit[tracer][dist].eps_sigma)
            sig_combined = math.sqrt(max(0.0, sig_expected**2 + sig_fit**2))
            z_score_expected = (
                (d / sig_expected) if (sig_expected > 0.0 and math.isfinite(sig_expected)) else float("nan")
            )
            z_score_combined = (
                (d / sig_combined) if (sig_combined > 0.0 and math.isfinite(sig_combined)) else float("nan")
            )
            deltas[tracer][dist] = {
                "delta_eps": d,
                "z_score_vs_y1data": z_score_expected,
                "z_score_combined": z_score_combined,
                "sigma_expected": sig_expected,
                "sigma_fit_diag": sig_fit,
            }
        # Record source used for this tracer.

        deltas[tracer]["_source_used_for_expected"] = {"name": source_used}

    # Plot

    _set_japanese_font()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, dist, title in zip(axes, ["lcdm", "pbg"], ["lcdm 座標（fid=Ωm=0.315）", "P_bg 座標（静的）"], strict=True):
        xs = [expected[t][dist].z_eff for t in tracers]
        ys_exp = [expected[t][dist].eps_mean for t in tracers]
        es_exp = [expected[t][dist].eps_sigma for t in tracers]
        ys_fit = [fit[t][dist].eps_mean for t in tracers]
        es_fit = [fit[t][dist].eps_sigma for t in tracers]
        ys_wedge = []
        for t in tracers:
            # 条件分岐: `wedge_metrics is None` を満たす経路を評価する。
            if wedge_metrics is None:
                ys_wedge.append(float("nan"))
                continue

            pt_w = _extract_wedge_eps_proxy(
                wedge_metrics,
                dist=dist,
                z_target=float(expected[t][dist].z_eff),
                z_tol=float(args.z_tol),
                sample="lrg",
                caps="combined",
            )
            ys_wedge.append(float(pt_w.eps_mean) if pt_w is not None else float("nan"))

        ax.axhline(0.0, color="0.6", lw=1.0)
        ax.errorbar(xs, ys_exp, yerr=es_exp, fmt="o", ms=7, capsize=3, label="公開距離制約（BAO fit; 派生）→ε_expected")
        ax.errorbar(xs, ys_fit, yerr=es_fit, fmt="s", ms=6, capsize=3, label="catalog ξℓ→peakfit（diag）")
        ok_w = [math.isfinite(float(y)) for y in ys_wedge]
        # 条件分岐: `any(ok_w)` を満たす経路を評価する。
        if any(ok_w):
            ax.scatter(
                [x for x, ok in zip(xs, ok_w, strict=True) if ok],
                [y for y, ok in zip(ys_wedge, ok_w, strict=True) if ok],
                marker="^",
                s=55,
                label="catalog μ-wedge→ε_proxy",
            )

        for x, y, lab in zip(xs, ys_exp, tracers, strict=True):
            ax.annotate(lab, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("z_eff", fontsize=11)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("ε (AP warping)", fontsize=11)
    axes[1].legend(loc="best", fontsize=9)
    fig.suptitle(f"DESI DR1 BAO（{', '.join(tracers)}）: 公開距離制約 ε_expected vs catalog peakfit ε", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    wedge_proxy: Optional[Dict[str, Dict[str, Optional[float]]]] = None
    # 条件分岐: `wedge_metrics is not None` を満たす経路を評価する。
    if wedge_metrics is not None:
        wedge_proxy = {}
        for t in tracers:
            wedge_proxy[t] = {}
            for k in ("lcdm", "pbg"):
                pt = _extract_wedge_eps_proxy(
                    wedge_metrics,
                    dist=k,
                    z_target=float(expected[t][k].z_eff),
                    z_tol=float(args.z_tol),
                    sample="lrg",
                    caps="combined",
                )
                wedge_proxy[t][k] = float(pt.eps_mean) if pt is not None else None

    metrics: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "4.5B.21.4.4 (DESI Y1data eps cross-check)",
        "inputs": {
            "y1data_json": str(y1_path),
            "bao_data_json": (str(bao_path) if bao is not None else None),
            "peakfit_metrics_json": str(peakfit_path),
            "wedge_metrics_json": str(wedge_path) if wedge_path is not None else None,
        },
        "params": {
            "out_tag": out_tag,
            "lcdm_omega_m": omega_m,
            "lcdm_n_grid": n_grid,
            "mc": {"n_draws": int(args.n_draws), "seed": int(args.seed)},
            "z_match_tol": float(args.z_tol),
            "prefer_bao_data": bool(args.prefer_bao_data),
        },
        "outputs": {"png": str(out_png), "metrics_json": str(out_json)},
        "results": {
            "tracers": tracers,
            "source_consistency": (source_consistency if source_consistency else None),
            "expected_from_y1data": {
                t: {k: {"z_eff": expected[t][k].z_eff, "eps": expected[t][k].eps_mean, "eps_sigma": expected[t][k].eps_sigma} for k in ("lcdm", "pbg")}
                for t in tracers
            },
            "fit_from_catalog_peakfit": {
                t: {k: {"z_eff": fit[t][k].z_eff, "eps": fit[t][k].eps_mean, "eps_sigma": fit[t][k].eps_sigma} for k in ("lcdm", "pbg")}
                for t in tracers
            },
            "eps_proxy_from_wedges": wedge_proxy,
            "delta_fit_minus_expected": deltas,
        },
        "notes": [
            "Y1data は D_M/r_d, D_H/r_d と相関係数 r を与える。F_AP=D_M/D_H は r_d に依存せず計算できる。",
            "同じ mean/cov は `CobayaSampler/bao_data` の `desi_2024_gaussian_bao_*` にも公開されており、本スクリプトは可能な範囲で両者の数値整合を記録する（source_consistency）。",
            "ε_expected は fid の F_AP と観測 F_AP の比から ε=(F_fid/F_obs)^(1/3)-1 で算出。",
            "catalog peakfit は dv=[ξ0,ξ2] の最小モデル（smooth+peak）で、誤差は diag（paircount proxy）。",
            "z_score は (a) ε_expected のみの誤差（z_score_vs_y1data）と、(b) ε_expected と peakfit(diag) を二乗和で合成した誤差（z_score_combined）を併記する。",
            "wedge は ξ(s,μ) から μ-wedge のピーク位置比 s∥/s⊥ を作り、ε_proxy≈(s∥/s⊥)^(-1/3)-1 として併記する。",
            "注意：BGS/QSO など DV-only の公開距離制約では ε_expected が定義できないため、本スクリプトでは対象外。",
        ],
    }
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    worklog.append_event(
        {
            "domain": "cosmology",
            "action": "desi_dr1_bao_y1data_eps_crosscheck",
            "inputs": [y1_path, peakfit_path],
            "outputs": [out_png, out_json],
            "params": metrics["params"],
        }
    )
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
