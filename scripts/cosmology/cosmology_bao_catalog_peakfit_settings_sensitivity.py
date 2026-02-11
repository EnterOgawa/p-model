#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_catalog_peakfit_settings_sensitivity.py

Phase 4 / Step 4.5B.21.4.4.5.1:
DESI（等）で catalog-based ξℓ（銀河+random→ξ(s,μ)→ξ0/ξ2）から行う peakfit の
設定感度（r_range / template / α-ε grid / quadrupole weight 等）を系統的にスキャンし、
特に LRG2 の ε（AP warping）が設定に依存して“動く/動かない”を判定する。

重要：
- Corrfunc は使わない（WSL不要）。既に生成された `xi_from_catalogs` と `__jk_cov.npz` を読むだけ。
- 共分散は jackknife（RA quantile）を前提（DESI LRG2 の切り分けの主戦場）。

出力（固定名）:
- output/cosmology/cosmology_bao_catalog_peakfit_settings_sensitivity__{sample}_{caps}__{out_tag}.png
- output/cosmology/cosmology_bao_catalog_peakfit_settings_sensitivity__{sample}_{caps}__{out_tag}_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cosmology import cosmology_bao_catalog_peakfit as _catalog_peakfit  # noqa: E402
from scripts.cosmology import cosmology_bao_xi_multipole_peakfit as _peakfit  # noqa: E402
from scripts.summary import worklog  # noqa: E402


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
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _sanitize_tag(s: str) -> str:
    return _catalog_peakfit._sanitize_out_tag(str(s))


def _zrange_key(z_min: float, z_max: float) -> str:
    def fmt(x: float) -> str:
        t = f"{float(x):.3f}".rstrip("0").rstrip(".")
        return t.replace(".", "p")

    return f"zmin{fmt(z_min)}_zmax{fmt(z_max)}"


def _zrange_label(z_min: float, z_max: float) -> str:
    return f"{float(z_min):.1f}–{float(z_max):.1f}"


def _profile_sigma_from_ci(ci_1sigma: Sequence[float | None]) -> Optional[float]:
    try:
        lo = float(ci_1sigma[0])
        hi = float(ci_1sigma[1])
        if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
            return 0.5 * (hi - lo)
        return None
    except Exception:
        return None


@dataclass(frozen=True)
class LoadedCase:
    case: _catalog_peakfit.CatalogCase
    z_min: float
    z_max: float
    s_all: np.ndarray
    xi0_all: np.ndarray
    xi2_all: np.ndarray
    cov_full: np.ndarray  # shape (2*n, 2*n)

    @property
    def zrange_key(self) -> str:
        return _zrange_key(self.z_min, self.z_max)

    @property
    def zrange_label(self) -> str:
        return _zrange_label(self.z_min, self.z_max)


@dataclass(frozen=True)
class FitScenario:
    id: str
    title: str
    r_min: float
    r_max: float
    mu_n: int
    r0: float
    sigma: float
    smooth_power_max: int
    alpha_min: float
    alpha_max: float
    alpha_step: float
    eps_min: float
    eps_max: float
    eps_step: float
    eps_rescan_factor: float
    eps_rescan_max_expands: int
    quad_weight: float


def _load_case(case: _catalog_peakfit.CatalogCase) -> LoadedCase:
    m = json.loads(case.metrics_path.read_text(encoding="utf-8"))
    z_cut = (m.get("params", {}) or {}).get("z_cut", {}) or {}
    z_min = float(z_cut.get("z_min", float("nan")))
    z_max = float(z_cut.get("z_max", float("nan")))
    if not (math.isfinite(z_min) and math.isfinite(z_max) and z_max > z_min):
        raise ValueError(f"invalid z_cut in metrics: {case.metrics_path}")

    with np.load(case.npz_path) as z:
        s_all = np.asarray(z["s"], dtype=float).reshape(-1)
        xi0_all = np.asarray(z["xi0"], dtype=float).reshape(-1)
        xi2_all = np.asarray(z["xi2"], dtype=float).reshape(-1)

    cov_path = case.npz_path.with_name(f"{case.npz_path.stem}__jk_cov.npz")
    if not cov_path.exists():
        raise FileNotFoundError(f"missing jackknife cov: {cov_path}")
    with np.load(cov_path) as zc:
        s_cov = np.asarray(zc["s"], dtype=float).reshape(-1)
        cov_full = np.asarray(zc["cov"], dtype=float)

    if s_cov.shape != s_all.shape or not np.allclose(s_cov, s_all, rtol=0.0, atol=1e-12):
        raise ValueError(f"jackknife cov uses different s bins: xi={case.npz_path.name}, cov={cov_path.name}")
    n_all = int(s_all.size)
    cov_full = np.asarray(cov_full, dtype=float).reshape(2 * n_all, 2 * n_all)
    return LoadedCase(case=case, z_min=z_min, z_max=z_max, s_all=s_all, xi0_all=xi0_all, xi2_all=xi2_all, cov_full=cov_full)


def _fit_one(
    *,
    lc: LoadedCase,
    scenario: FitScenario,
    ok_max: float,
    mixed_max: float,
) -> Dict[str, Any]:
    s_all = lc.s_all
    xi0_all = lc.xi0_all
    xi2_all = lc.xi2_all

    m_fit = (s_all >= float(scenario.r_min)) & (s_all <= float(scenario.r_max)) & np.isfinite(xi0_all) & np.isfinite(xi2_all)
    idx = np.nonzero(m_fit)[0].astype(int, copy=False)
    if idx.size < 10:
        raise ValueError(f"too few bins after range cut: n={idx.size} (r={scenario.r_min}..{scenario.r_max})")

    s = s_all[m_fit]
    xi0 = xi0_all[m_fit]
    xi2 = xi2_all[m_fit]
    y = np.concatenate([xi0, xi2], axis=0)

    n_all = int(s_all.size)
    sel = np.concatenate([idx, idx + n_all], axis=0)
    cov = lc.cov_full[np.ix_(sel, sel)]
    cov_inv = np.linalg.pinv(cov, rcond=1e-12)
    cov_inv = 0.5 * (cov_inv + cov_inv.T)

    mu_n = int(scenario.mu_n)
    if mu_n < 20:
        raise ValueError("--mu-n must be >= 20")
    mu, w = np.polynomial.legendre.leggauss(mu_n)
    sqrt1mu2 = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
    p2_fid = _peakfit._p2(mu)

    alpha_grid = np.arange(float(scenario.alpha_min), float(scenario.alpha_max) + 0.5 * float(scenario.alpha_step), float(scenario.alpha_step))
    eps_grid = np.arange(float(scenario.eps_min), float(scenario.eps_max) + 0.5 * float(scenario.eps_step), float(scenario.eps_step))
    if alpha_grid.size < 5 or eps_grid.size < 5:
        raise ValueError("grid too small; widen ranges or reduce steps")

    best_eps0 = _peakfit._scan_grid(
        y=y,
        cov_inv=cov_inv,
        s_fid=s,
        alpha_grid=alpha_grid,
        eps_grid=np.asarray([0.0], dtype=float),
        r0_mpc_h=float(scenario.r0),
        sigma_mpc_h=float(scenario.sigma),
        mu=mu,
        w=w,
        p2_fid=p2_fid,
        sqrt1mu2=sqrt1mu2,
        smooth_power_max=int(scenario.smooth_power_max),
        return_eps_profile=False,
    )

    # eps scan with optional rescan (match cosmology_bao_catalog_peakfit).
    eps_grid_use = eps_grid
    expands_done = 0
    eps_scan_initial = {
        "min": float(np.min(eps_grid_use)),
        "max": float(np.max(eps_grid_use)),
        "step": float(eps_grid_use[1] - eps_grid_use[0]) if eps_grid_use.size >= 2 else float("nan"),
    }
    edge_hit = False
    ci_clipped = False
    eps_ci_1 = float("nan")
    eps_ci_2 = float("nan")
    eps_ci_2s = float("nan")
    eps_ci_2s_hi = float("nan")

    while True:
        best_free = _peakfit._scan_grid(
            y=y,
            cov_inv=cov_inv,
            s_fid=s,
            alpha_grid=alpha_grid,
            eps_grid=eps_grid_use,
            r0_mpc_h=float(scenario.r0),
            sigma_mpc_h=float(scenario.sigma),
            mu=mu,
            w=w,
            p2_fid=p2_fid,
            sqrt1mu2=sqrt1mu2,
            smooth_power_max=int(scenario.smooth_power_max),
            return_eps_profile=True,
        )

        eps_profile = list(best_free.get("eps_profile") or [])
        eps_grid_vals = np.array([float(r["eps"]) for r in eps_profile], dtype=float) if eps_profile else np.array([], dtype=float)
        chi2_prof = np.array([float(r["chi2"]) for r in eps_profile], dtype=float) if eps_profile else np.array([], dtype=float)
        eps_ci_1, eps_ci_2 = _peakfit._profile_ci(x=eps_grid_vals, chi2=chi2_prof, delta=1.0)
        eps_ci_2s, eps_ci_2s_hi = _peakfit._profile_ci(x=eps_grid_vals, chi2=chi2_prof, delta=4.0)

        eps_best_tmp = float(best_free["eps"])
        eps_min_tmp = float(np.min(eps_grid_use))
        eps_max_tmp = float(np.max(eps_grid_use))
        edge_hit = (eps_best_tmp <= eps_min_tmp + 1e-12) or (eps_best_tmp >= eps_max_tmp - 1e-12)
        try:
            lo = float(eps_ci_1)
            hi = float(eps_ci_2)
            ci_clipped = (lo <= eps_min_tmp + 1e-12) or (hi >= eps_max_tmp - 1e-12)
        except Exception:
            ci_clipped = True

        if (not edge_hit) and (not ci_clipped):
            break
        if expands_done >= int(scenario.eps_rescan_max_expands):
            break

        width0 = float(eps_max_tmp - eps_min_tmp)
        center0 = 0.5 * float(eps_max_tmp + eps_min_tmp)
        width1 = width0 * float(scenario.eps_rescan_factor)
        eps_min1 = center0 - 0.5 * width1
        eps_max1 = center0 + 0.5 * width1
        eps_step1 = float(eps_grid_use[1] - eps_grid_use[0]) if eps_grid_use.size >= 2 else float(scenario.eps_step)
        eps_grid_use = np.arange(eps_min1, eps_max1 + 0.5 * eps_step1, eps_step1, dtype=float)
        expands_done += 1

    chi2_free = float(best_free["chi2"])
    chi2_eps0 = float(best_eps0["chi2"])
    delta_chi2_eps = chi2_eps0 - chi2_free

    eps_best = float(best_free["eps"])
    abs_eps = abs(eps_best)
    sigma_eps = None
    abs_sigma = None
    abs_sigma_is_lower_bound = False
    try:
        sigma_eps = _profile_sigma_from_ci([eps_ci_1, eps_ci_2])
        if sigma_eps is not None and sigma_eps > 0:
            abs_sigma = abs_eps / float(sigma_eps)
    except Exception:
        sigma_eps = None
        abs_sigma = None

    if edge_hit or ci_clipped:
        abs_sigma = float(mixed_max) + 1.0
        abs_sigma_is_lower_bound = True
        status = "ng"
    else:
        status = _catalog_peakfit._status_from_abs_sigma(abs_sigma, ok_max=float(ok_max), mixed_max=float(mixed_max))

    return {
        "sample": lc.case.sample,
        "caps": lc.case.caps,
        "dist": lc.case.dist,
        "z_range": {"z_min": float(lc.z_min), "z_max": float(lc.z_max), "label": lc.zrange_label, "key": lc.zrange_key},
        "z_eff": float(lc.case.z_eff),
        "fit": {
            "free": {
                "alpha": float(best_free["alpha"]),
                "eps": float(best_free["eps"]),
                "chi2": chi2_free,
                "eps_ci_1sigma": [float(eps_ci_1), float(eps_ci_2)],
                "eps_ci_2sigma": [float(eps_ci_2s), float(eps_ci_2s_hi)],
            },
            "eps_fixed_0": {"alpha": float(best_eps0["alpha"]), "eps": 0.0, "chi2": chi2_eps0},
            "delta_chi2_eps0_vs_free": float(delta_chi2_eps),
        },
        "screening": {
            "abs_eps": float(abs_eps),
            "sigma_eps_1sigma": float(sigma_eps) if sigma_eps is not None else None,
            "abs_sigma": float(abs_sigma) if abs_sigma is not None else None,
            "abs_sigma_is_lower_bound": bool(abs_sigma_is_lower_bound),
            "scan": {
                "eps_grid": {
                    "min": float(np.min(eps_grid_use)),
                    "max": float(np.max(eps_grid_use)),
                    "step": float(eps_grid_use[1] - eps_grid_use[0]) if eps_grid_use.size >= 2 else float("nan"),
                },
                "edge_hit": bool(edge_hit),
                "ci_clipped": bool(ci_clipped),
                "rescan": {
                    "factor": float(scenario.eps_rescan_factor),
                    "max_expands": int(scenario.eps_rescan_max_expands),
                    "expands_done": int(expands_done),
                    "eps_grid_initial": eps_scan_initial,
                },
            },
            "status": str(status),
        },
        "inputs": {
            "npz": str(lc.case.npz_path),
            "metrics_json": str(lc.case.metrics_path),
            "jk_cov_npz": str(lc.case.npz_path.with_name(f"{lc.case.npz_path.stem}__jk_cov.npz")),
        },
        "scenario_id": str(scenario.id),
    }


def _default_scenarios() -> List[FitScenario]:
    # Baseline matches cosmology_bao_catalog_peakfit defaults (for comparability).
    base = dict(
        r_min=50.0,
        r_max=150.0,
        mu_n=80,
        r0=105.0,
        sigma=10.0,
        smooth_power_max=2,
        alpha_min=0.9,
        alpha_max=1.1,
        alpha_step=0.002,
        eps_min=-0.1,
        eps_max=0.1,
        eps_step=0.002,
        eps_rescan_factor=2.0,
        eps_rescan_max_expands=2,
        quad_weight=1.0,
    )

    def sc(sid: str, title: str, **kw: Any) -> FitScenario:
        cfg = dict(base)
        cfg.update(kw)
        return FitScenario(id=sid, title=title, **cfg)

    return [
        sc("base", "baseline"),
        sc("smooth3", "smooth p=3 (broadband)", smooth_power_max=3),
        sc("rmin40", "r_min=40", r_min=40.0),
        sc("rmin60", "r_min=60", r_min=60.0),
        sc("rmax130", "r_max=130", r_max=130.0),
        sc("rmax160", "r_max=160", r_max=160.0),
        sc("r0_100", "r0=100", r0=100.0),
        sc("r0_110", "r0=110", r0=110.0),
        sc("sig8", "sigma=8", sigma=8.0),
        sc("sig12", "sigma=12", sigma=12.0),
        sc("mu40", "mu_n=40", mu_n=40),
        sc("mu120", "mu_n=120", mu_n=120),
        sc("qw2", "quad_weight=2", quad_weight=2.0),
        sc("qw5", "quad_weight=5", quad_weight=5.0),
        sc("epswide", "eps∈[-0.2,0.2]", eps_min=-0.2, eps_max=0.2, eps_step=0.004),
        sc("awide", "alpha∈[0.8,1.2]", alpha_min=0.8, alpha_max=1.2, alpha_step=0.004),
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cosmology: peakfit setting sensitivity (jackknife cov; catalog-based xiℓ).")
    ap.add_argument("--sample", default="lrg", help="sample (default: lrg)")
    ap.add_argument("--caps", default="combined", help="caps (default: combined)")
    ap.add_argument("--dists", default="lcdm,pbg", help="distance models (comma; default: lcdm,pbg)")
    ap.add_argument("--out-tag", default="w_desi_default_ms_off_y1bins", help="filter out_tag for xi inputs")
    ap.add_argument("--ok-max", type=float, default=1.0, help="OK threshold for |eps|/sigma (default: 1)")
    ap.add_argument("--mixed-max", type=float, default=2.0, help="mixed threshold for |eps|/sigma (default: 2)")
    ap.add_argument(
        "--out-png",
        default="",
        help="output png (default: auto under output/cosmology/)",
    )
    ap.add_argument(
        "--out-json",
        default="",
        help="output metrics json (default: auto under output/cosmology/)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    sample = str(args.sample)
    caps = str(args.caps)
    dists = [d.strip() for d in str(args.dists).split(",") if d.strip()]
    out_tag = str(args.out_tag)
    if not dists:
        raise SystemExit("--dists must not be empty")

    # Load the same case definition as the main peakfit script.
    cases = _catalog_peakfit._load_cases(sample=sample, caps=caps, dists=dists, require_zbin=False, out_tag=out_tag)
    if not cases:
        raise SystemExit(f"no matching xi inputs: sample={sample}, caps={caps}, dists={dists}, out_tag={out_tag}")

    # Coordinate-spec guard (same rationale as cosmology_bao_catalog_peakfit).
    z_sources = {str(c.z_source) for c in cases if str(c.z_source)}
    los_defs = {str(c.los) for c in cases if str(c.los)}
    weight_schemes = {str(c.weight_scheme) for c in cases if str(c.weight_scheme)}
    est_hashes = {str(c.estimator_spec_hash) for c in cases if str(c.estimator_spec_hash)}
    recon_modes = {str(c.recon_mode) for c in cases if str(c.recon_mode)}
    if len(z_sources) != 1:
        raise SystemExit(f"mixed z_source across inputs: {', '.join(sorted(z_sources))}")
    if len(los_defs) != 1:
        raise SystemExit(f"mixed los across inputs: {', '.join(sorted(los_defs))}")
    if len(weight_schemes) != 1:
        raise SystemExit(f"mixed weight_scheme across inputs: {', '.join(sorted(weight_schemes))}")
    if len(est_hashes) != 1:
        raise SystemExit(f"mixed estimator_spec across inputs: {', '.join(sorted(est_hashes))}")
    if len(recon_modes) != 1:
        raise SystemExit(f"mixed recon_mode across inputs: {', '.join(sorted(recon_modes))}")

    loaded: List[LoadedCase] = []
    for c in cases:
        loaded.append(_load_case(c))

    scenarios = _default_scenarios()
    results_by_scenario: List[Dict[str, Any]] = []
    for sc in scenarios:
        sc_res: List[Dict[str, Any]] = []
        for lc in loaded:
            sc_res.append(_fit_one(lc=lc, scenario=sc, ok_max=float(args.ok_max), mixed_max=float(args.mixed_max)))
        results_by_scenario.append(
            {
                "scenario": {
                    "id": sc.id,
                    "title": sc.title,
                    "fit_config": {
                        "r_range_mpc_h": [float(sc.r_min), float(sc.r_max)],
                        "template_peak": {"r0_mpc_h": float(sc.r0), "sigma_mpc_h": float(sc.sigma)},
                        "smooth_basis": {
                            "power_max": int(sc.smooth_power_max),
                            "terms": _peakfit._smooth_basis_labels(smooth_power_max=int(sc.smooth_power_max)),
                        },
                        "grid": {
                            "alpha": {"min": float(sc.alpha_min), "max": float(sc.alpha_max), "step": float(sc.alpha_step)},
                            "eps": {"min": float(sc.eps_min), "max": float(sc.eps_max), "step": float(sc.eps_step)},
                        },
                        "grid_rescan": {"eps": {"factor": float(sc.eps_rescan_factor), "max_expands": int(sc.eps_rescan_max_expands)}},
                        "mu_integral": {"mu_n": int(sc.mu_n)},
                        "quad_weight": float(sc.quad_weight),
                    },
                },
                "results": sc_res,
            }
        )

    out_dir = _ROOT / "output" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{_sanitize_tag(sample)}_{_sanitize_tag(caps)}__{_sanitize_tag(out_tag)}"
    out_png = Path(str(args.out_png)).resolve() if str(args.out_png).strip() else out_dir / f"cosmology_bao_catalog_peakfit_settings_sensitivity__{tag}.png"
    out_json = Path(str(args.out_json)).resolve() if str(args.out_json).strip() else out_dir / f"cosmology_bao_catalog_peakfit_settings_sensitivity__{tag}_metrics.json"

    # Plot: ε per scenario, split by z-range.
    _set_japanese_font()
    dists_sorted = sorted({r.case.dist for r in loaded})
    zranges = sorted({lc.zrange_key for lc in loaded})
    zrange_meta = {lc.zrange_key: lc.zrange_label for lc in loaded}

    fig, axes = plt.subplots(nrows=len(zranges), ncols=1, figsize=(12.8, 3.6 * len(zranges)), sharex=True)
    if len(zranges) == 1:
        axes = [axes]
    x = np.arange(len(scenarios), dtype=float)
    colors = {"lcdm": "#1f77b4", "pbg": "#ff7f0e"}
    markers = {"lcdm": "o", "pbg": "s"}

    for ax, zkey in zip(axes, zranges, strict=True):
        ax.axhline(0.0, color="#888888", lw=1.2)
        for di, dist in enumerate(dists_sorted):
            xs: List[float] = []
            ys: List[float] = []
            yerr_lo: List[float] = []
            yerr_hi: List[float] = []
            for pack in results_by_scenario:
                rr = [r for r in pack["results"] if r["dist"] == dist and r["z_range"]["key"] == zkey]
                if not rr:
                    xs.append(float("nan"))
                    ys.append(float("nan"))
                    yerr_lo.append(float("nan"))
                    yerr_hi.append(float("nan"))
                    continue
                r0 = rr[0]
                eps = float(r0["fit"]["free"]["eps"])
                ci = r0["fit"]["free"].get("eps_ci_1sigma") or [float("nan"), float("nan")]
                lo = float(ci[0])
                hi = float(ci[1])
                ys.append(eps)
                yerr_lo.append(max(0.0, eps - lo) if math.isfinite(lo) else float("nan"))
                yerr_hi.append(max(0.0, hi - eps) if math.isfinite(hi) else float("nan"))
                xs.append(0.0)

            offset = (di - 0.5 * (len(dists_sorted) - 1)) * 0.08
            ax.errorbar(
                x + offset,
                np.asarray(ys, dtype=float),
                yerr=np.vstack([np.asarray(yerr_lo, dtype=float), np.asarray(yerr_hi, dtype=float)]),
                fmt=markers.get(dist, "o"),
                color=colors.get(dist, None),
                capsize=3,
                markersize=6,
                linestyle="-",
                lw=1.2,
                label=dist,
            )

        ax.set_ylabel("ε (smooth+peak)")
        ax.set_title(f"z ∈ {zrange_meta.get(zkey, zkey)}  /  out_tag={out_tag}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([sc.id for sc in scenarios], rotation=45, ha="right")
    axes[-1].set_xlabel("scenario")
    fig.suptitle(f"BAO peakfit 設定感度（sample={sample}, caps={caps}, cov=jackknife）", y=0.98)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "4.5B.21.4.4.5.1 (BAO catalog peakfit setting sensitivity)",
        "inputs": {
            "sample": sample,
            "caps": caps,
            "dists": dists,
            "out_tag": out_tag,
            "coordinate_spec": {
                "z_source": next(iter(z_sources)) if z_sources else "",
                "los": next(iter(los_defs)) if los_defs else "",
                "weight_scheme": next(iter(weight_schemes)) if weight_schemes else "",
                "estimator_spec_hash": next(iter(est_hashes)) if est_hashes else "",
                "recon_mode": next(iter(recon_modes)) if recon_modes else "",
            },
            "covariance": {"source": "jackknife_ra_quantile", "rcond": 1e-12},
            "n_cases": int(len(loaded)),
        },
        "policy": {
            "status_metric": "|eps|/sigma_eps_1sigma (profile CI; jackknife cov)",
            "status_thresholds": {"ok_max": float(args.ok_max), "mixed_max": float(args.mixed_max)},
        },
        "outputs": {"png": str(out_png), "metrics_json": str(out_json)},
        "results": results_by_scenario,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_bao_catalog_peakfit_settings_sensitivity",
                "argv": sys.argv,
                "outputs": {"png": out_png, "metrics_json": out_json},
                "metrics": {"sample": sample, "caps": caps, "dists": dists, "out_tag": out_tag},
            }
        )
    except Exception:
        pass

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
