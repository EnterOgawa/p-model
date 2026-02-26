# -*- coding: utf-8 -*-
"""
cosmology_desi_dr1_bao_y1data_alpha_crosscheck.py

Phase 4 / Step 4.5B（宇宙論：BAO一次統計の確証決定打）補助:
DESI DR1 BAO 論文（VI）の DV-only 公開距離制約（Y1data: D_V/r_d）から
α_expected = (D_V/r_d)_obs / (D_V/r_d)_fid を作り、catalog peakfit の α と比較する。

注意：
- DV-only（BGS/QSO）は ε_expected が定義できないため、本スクリプトは α のみを扱う。
- r_d は相殺しない（εと違い距離指標非依存ではない）ため、位置づけは補助。

入出力:
  - 入力:
    - data/cosmology/desi_dr1_bao_y1data.json
    - output/private/cosmology/cosmology_bao_catalog_peakfit_*_metrics.json
  - 出力:
    - output/private/cosmology/cosmology_desi_dr1_bao_y1data_alpha_crosscheck__{out_tag}.png
    - output/private/cosmology/cosmology_desi_dr1_bao_y1data_alpha_crosscheck__{out_tag}_metrics.json
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

from scripts.cosmology.cosmology_bao_xi_multipole_peakfit import (  # noqa: E402
    _f_ap_lcdm_flat as _f_ap_lcdm_flat,
    _f_ap_pbg_exponential as _f_ap_pbg_exponential,
)
from scripts.summary import worklog  # noqa: E402

_C_OVER_100_MPC_OVER_H = 2997.92458  # (km/s) / (100 km/s/Mpc) in Mpc/h


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


@dataclass(frozen=True)
class _AlphaPoint:
    z_eff: float
    alpha_mean: float
    alpha_sigma: float


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sigma_from_ci(ci: Any) -> float:
    try:
        lo, hi = float(ci[0]), float(ci[1])
        # 条件分岐: `math.isfinite(lo) and math.isfinite(hi)` を満たす経路を評価する。
        if math.isfinite(lo) and math.isfinite(hi):
            return 0.5 * abs(hi - lo)
    except Exception:
        pass

    return float("nan")


def _extract_dv_over_rd_from_y1data_row(r: Dict[str, Any]) -> tuple[float, float, float]:
    dv = r.get("dv_over_rd")
    # 条件分岐: `not isinstance(dv, dict)` を満たす経路を評価する。
    if not isinstance(dv, dict):
        raise ValueError("row does not contain dv_over_rd (expected for DV-only tracers)")

    mu = dv.get("mean")
    sig = dv.get("sigma")
    # 条件分岐: `mu is None or sig is None` を満たす経路を評価する。
    if mu is None or sig is None:
        raise ValueError("dv_over_rd missing mean/sigma")

    return float(r["z_eff"]), float(mu), float(sig)


def _dv_over_rd_fid_lcdm(z_eff: float, *, omega_m: float, n_grid: int, rd_mpc_over_h: float) -> float:
    z = float(z_eff)
    # 条件分岐: `not (z > 0.0 and math.isfinite(z))` を満たす経路を評価する。
    if not (z > 0.0 and math.isfinite(z)):
        return float("nan")

    # 条件分岐: `not (0.0 < float(omega_m) < 1.0)` を満たす経路を評価する。

    if not (0.0 < float(omega_m) < 1.0):
        return float("nan")

    # E(z)

    ez = float(math.sqrt(float(omega_m) * (1.0 + z) ** 3 + (1.0 - float(omega_m))))
    dh = float(_C_OVER_100_MPC_OVER_H / ez)  # D_H=c/H in Mpc/h (flat LCDM)

    # D_M=c/H0 ∫ dz/E(z) in Mpc/h
    z_grid = np.linspace(0.0, z, int(n_grid), dtype=float)
    one_p = 1.0 + z_grid
    ez_grid = np.sqrt(float(omega_m) * one_p**3 + (1.0 - float(omega_m)))
    integrand = 1.0 / ez_grid
    try:
        integral = float(np.trapezoid(integrand, z_grid))
    except AttributeError:
        integral = float(np.trapz(integrand, z_grid))

    dm = float(_C_OVER_100_MPC_OVER_H * integral)

    dv = float((z * dm * dm * dh) ** (1.0 / 3.0))
    return float(dv / float(rd_mpc_over_h))


def _dv_over_rd_fid_pbg(z_eff: float, *, rd_mpc_over_h: float) -> float:
    z = float(z_eff)
    # 条件分岐: `not (z > 0.0 and math.isfinite(z))` を満たす経路を評価する。
    if not (z > 0.0 and math.isfinite(z)):
        return float("nan")

    op = 1.0 + z
    dm = float(_C_OVER_100_MPC_OVER_H * math.log(op))  # D_M=(c/H0)ln(1+z)
    dh = float(_C_OVER_100_MPC_OVER_H / op)  # H_eff=H0(1+z) => D_H=c/H

    dv = float((z * dm * dm * dh) ** (1.0 / 3.0))
    return float(dv / float(rd_mpc_over_h))


def _extract_peakfit_alpha(
    points: List[Dict[str, Any]],
    *,
    dist: str,
    z_target: float,
    z_tol: float,
    mode: str,
) -> Optional[_AlphaPoint]:
    dist = str(dist)
    mode = str(mode)
    best: Optional[Dict[str, Any]] = None
    best_dz = float("inf")
    for p in points:
        # 条件分岐: `str(p.get("dist")) != dist` を満たす経路を評価する。
        if str(p.get("dist")) != dist:
            continue

        try:
            z_eff = float(p.get("z_eff", float("nan")))
        except Exception:
            continue

        # 条件分岐: `not math.isfinite(z_eff)` を満たす経路を評価する。

        if not math.isfinite(z_eff):
            continue

        dz = abs(z_eff - float(z_target))
        # 条件分岐: `dz > float(z_tol)` を満たす経路を評価する。
        if dz > float(z_tol):
            continue

        # 条件分岐: `dz < best_dz` を満たす経路を評価する。

        if dz < best_dz:
            best = p
            best_dz = dz

    # 条件分岐: `best is None` を満たす経路を評価する。

    if best is None:
        return None

    fit = (best.get("fit", {}) or {}).get(mode, {}) or {}
    alpha = float(fit.get("alpha", float("nan")))
    ci = fit.get("alpha_ci_1sigma")
    sig = _sigma_from_ci(ci) if ci is not None else float("nan")
    return _AlphaPoint(z_eff=float(best.get("z_eff")), alpha_mean=alpha, alpha_sigma=sig)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DESI DR1 DV-only: alpha cross-check vs Y1data DV/rd.")
    ap.add_argument(
        "--peakfit-metrics-json",
        required=True,
        help="Path to cosmology_bao_catalog_peakfit_*_metrics.json",
    )
    ap.add_argument(
        "--y1data-json",
        default=str(_ROOT / "data" / "cosmology" / "desi_dr1_bao_y1data.json"),
        help="Path to cached Y1data JSON (default: data/cosmology/desi_dr1_bao_y1data.json)",
    )
    ap.add_argument("--tracers", type=str, default="BGS,QSO", help="DV-only tracers (comma; default: BGS,QSO)")
    ap.add_argument("--z-tol", type=float, default=0.08, help="z match tolerance (default: 0.08)")
    ap.add_argument("--lcdm-omega-m", type=float, default=0.315, help="LCDM Omega_m for fid DV (default: 0.315)")
    ap.add_argument("--lcdm-n-grid", type=int, default=4000, help="LCDM z integral grid (default: 4000)")
    ap.add_argument("--fit-mode", choices=["eps_fixed_0", "free"], default="eps_fixed_0", help="Use alpha from which fit (default: eps_fixed_0)")
    ap.add_argument("--out-tag", type=str, required=True, help="Output tag for filenames")
    args = ap.parse_args(list(argv) if argv is not None else None)

    peakfit_path = Path(str(args.peakfit_metrics_json))
    y1_path = Path(str(args.y1data_json))
    # 条件分岐: `not peakfit_path.exists()` を満たす経路を評価する。
    if not peakfit_path.exists():
        raise SystemExit(f"missing peakfit metrics: {peakfit_path}")

    # 条件分岐: `not y1_path.exists()` を満たす経路を評価する。

    if not y1_path.exists():
        raise SystemExit(f"missing y1data json: {y1_path}")

    peakfit = _load_json(peakfit_path)
    points = peakfit.get("results")
    # 条件分岐: `not isinstance(points, list)` を満たす経路を評価する。
    if not isinstance(points, list):
        raise SystemExit("peakfit metrics has invalid 'results' (expected list)")

    # Template BAO scale as rd_fid proxy (Mpc/h).

    r0_vals = []
    for p in points:
        tp = p.get("template_peak") or {}
        # 条件分岐: `isinstance(tp, dict) and ("r0_mpc_h" in tp)` を満たす経路を評価する。
        if isinstance(tp, dict) and ("r0_mpc_h" in tp):
            r0_vals.append(float(tp["r0_mpc_h"]))

    # 条件分岐: `not r0_vals or not all(math.isfinite(x) for x in r0_vals)` を満たす経路を評価する。

    if not r0_vals or not all(math.isfinite(x) for x in r0_vals):
        raise SystemExit("peakfit metrics missing template_peak.r0_mpc_h")

    rd_fid = float(np.median(np.array(r0_vals, dtype=float)))

    tracers = [t.strip() for t in str(args.tracers).split(",") if t.strip()]
    # 条件分岐: `not tracers` を満たす経路を評価する。
    if not tracers:
        raise SystemExit("--tracers must not be empty")

    y1 = _load_json(y1_path)
    rows = y1.get("rows")
    # 条件分岐: `not isinstance(rows, list)` を満たす経路を評価する。
    if not isinstance(rows, list):
        raise SystemExit("y1data json invalid (expected rows=list)")

    y1_by_tracer = {str(r.get("tracer")): r for r in rows if isinstance(r, dict)}

    missing = [t for t in tracers if t not in y1_by_tracer]
    # 条件分岐: `missing` を満たす経路を評価する。
    if missing:
        raise SystemExit(f"y1data missing tracers: {missing}")

    expected: Dict[str, Dict[str, _AlphaPoint]] = {}
    fit: Dict[str, Dict[str, _AlphaPoint]] = {}
    deltas: Dict[str, Dict[str, Any]] = {}

    for tracer in tracers:
        r = y1_by_tracer[tracer]
        z_eff, dv_mean, dv_sig = _extract_dv_over_rd_from_y1data_row(r)

        dv_fid_lcdm = _dv_over_rd_fid_lcdm(
            z_eff,
            omega_m=float(args.lcdm_omega_m),
            n_grid=int(args.lcdm_n_grid),
            rd_mpc_over_h=rd_fid,
        )
        dv_fid_pbg = _dv_over_rd_fid_pbg(z_eff, rd_mpc_over_h=rd_fid)
        # 条件分岐: `not (math.isfinite(dv_fid_lcdm) and dv_fid_lcdm > 0.0 and math.isfinite(dv_fi...` を満たす経路を評価する。
        if not (math.isfinite(dv_fid_lcdm) and dv_fid_lcdm > 0.0 and math.isfinite(dv_fid_pbg) and dv_fid_pbg > 0.0):
            raise SystemExit("invalid fid DV/rd (check fid params)")

        alpha_exp_lcdm = float(dv_mean / dv_fid_lcdm)
        alpha_exp_pbg = float(dv_mean / dv_fid_pbg)
        sig_exp_lcdm = float(dv_sig / dv_fid_lcdm)
        sig_exp_pbg = float(dv_sig / dv_fid_pbg)

        expected[tracer] = {
            "lcdm": _AlphaPoint(z_eff=z_eff, alpha_mean=alpha_exp_lcdm, alpha_sigma=sig_exp_lcdm),
            "pbg": _AlphaPoint(z_eff=z_eff, alpha_mean=alpha_exp_pbg, alpha_sigma=sig_exp_pbg),
        }

        fit_lcdm = _extract_peakfit_alpha(
            points,
            dist="lcdm",
            z_target=z_eff,
            z_tol=float(args.z_tol),
            mode=str(args.fit_mode),
        )
        fit_pbg = _extract_peakfit_alpha(
            points,
            dist="pbg",
            z_target=z_eff,
            z_tol=float(args.z_tol),
            mode=str(args.fit_mode),
        )
        # 条件分岐: `fit_lcdm is None or fit_pbg is None` を満たす経路を評価する。
        if fit_lcdm is None or fit_pbg is None:
            raise SystemExit(f"peakfit metrics missing lcdm/pbg points near z_eff={z_eff:.3f} for tracer {tracer}")

        fit[tracer] = {"lcdm": fit_lcdm, "pbg": fit_pbg}

        deltas[tracer] = {}
        for dist in ("lcdm", "pbg"):
            d = float(fit[tracer][dist].alpha_mean) - float(expected[tracer][dist].alpha_mean)
            sig_expected = float(expected[tracer][dist].alpha_sigma)
            sig_fit = float(fit[tracer][dist].alpha_sigma)
            sig_combined = (
                math.sqrt(max(0.0, sig_expected**2 + sig_fit**2))
                if (math.isfinite(sig_fit) and sig_fit > 0.0)
                else float("nan")
            )
            z_score_expected = (d / sig_expected) if (sig_expected > 0.0 and math.isfinite(sig_expected)) else float("nan")
            z_score_combined = (d / sig_combined) if (sig_combined > 0.0 and math.isfinite(sig_combined)) else float("nan")
            deltas[tracer][dist] = {
                "z_eff": float(expected[tracer][dist].z_eff),
                "alpha_fit": float(fit[tracer][dist].alpha_mean),
                "alpha_fit_sigma": (float(sig_fit) if (math.isfinite(sig_fit) and sig_fit > 0.0) else None),
                "alpha_expected": float(expected[tracer][dist].alpha_mean),
                "alpha_expected_sigma": float(sig_expected),
                "delta_fit_minus_expected": float(d),
                "z_score_vs_y1data": float(z_score_expected),
                "z_score_combined": float(z_score_combined),
            }

    out_dir = _ROOT / "output" / "private" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tag = str(args.out_tag)
    out_png = out_dir / f"cosmology_desi_dr1_bao_y1data_alpha_crosscheck__{out_tag}.png"
    out_json = out_dir / f"cosmology_desi_dr1_bao_y1data_alpha_crosscheck__{out_tag}_metrics.json"

    # Plot.
    try:
        import matplotlib.pyplot as plt

        _set_japanese_font()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
        for ax, dist, title in zip(axes, ["lcdm", "pbg"], ["lcdm 座標（fid=Ωm=0.315）", "P_bg 座標（静的）"], strict=True):
            xs = [expected[t][dist].z_eff for t in tracers]
            ys_exp = [expected[t][dist].alpha_mean for t in tracers]
            es_exp = [expected[t][dist].alpha_sigma for t in tracers]
            ys_fit = [fit[t][dist].alpha_mean for t in tracers]
            es_fit = [fit[t][dist].alpha_sigma for t in tracers]

            ax.errorbar(xs, ys_exp, yerr=es_exp, fmt="o", ms=7, capsize=3, label="公開距離制約（DV/rd）→α_expected")
            ax.errorbar(xs, ys_fit, yerr=es_fit, fmt="s", ms=6, capsize=3, label=f"catalog peakfit → α ({args.fit_mode})")
            ax.axhline(1.0, color="#999999", linewidth=1.0, linestyle="--", alpha=0.8)
            ax.set_xlabel("z_eff")
            ax.set_ylabel("alpha")
            ax.set_title(title, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

        fig.suptitle(f"DESI DR1 BAO（DV-only: {', '.join(tracers)}）: α_expected vs catalog peakfit α", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
    except Exception:
        # plotting is optional
        pass

    metrics: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "4.5B.21.4.4.7 (DESI Y1data alpha cross-check; DV-only)",
        "inputs": {"y1data_json": str(y1_path), "peakfit_metrics_json": str(peakfit_path)},
        "params": {
            "out_tag": out_tag,
            "tracers": tracers,
            "fit_mode": str(args.fit_mode),
            "rd_fid_proxy_mpc_h": float(rd_fid),
            "lcdm_omega_m": float(args.lcdm_omega_m),
            "lcdm_n_grid": int(args.lcdm_n_grid),
        },
        "outputs": {"png": (str(out_png) if out_png.exists() else None), "metrics_json": str(out_json)},
        "results": {
            "expected_from_y1data": {
                t: {k: {"z_eff": expected[t][k].z_eff, "alpha": expected[t][k].alpha_mean, "alpha_sigma": expected[t][k].alpha_sigma} for k in ("lcdm", "pbg")}
                for t in tracers
            },
            "fit_from_catalog_peakfit": {
                t: {
                    k: {
                        "z_eff": fit[t][k].z_eff,
                        "alpha": fit[t][k].alpha_mean,
                        "alpha_sigma": (fit[t][k].alpha_sigma if math.isfinite(fit[t][k].alpha_sigma) else None),
                    }
                    for k in ("lcdm", "pbg")
                }
                for t in tracers
            },
            "delta_fit_minus_expected": deltas,
        },
        "notes": [
            "DV-only（BGS/QSO）の公開距離制約は D_V/r_d のみ（DM/DHなし）のため、ε_expected は定義できない。",
            "α_expected は (D_V/r_d)_obs / (D_V/r_d)_fid として算出（r_d は相殺しない）。",
            "fid の D_V は lcdm では D_M=c/H0∫dz/E(z), D_H=c/H0/E(z), D_V=(z D_M^2 D_H)^(1/3)。",
            "P_bg（静的指数）では D_M=(c/H0)ln(1+z), D_H=(c/H0)/(1+z)。",
            "peakfit の r0_mpc_h を r_d,fid の proxy として用い、D_V/r_d,fid= D_V / r0_mpc_h で規格化した。",
            f"参考（AP）：F_AP_pbg=(1+z)ln(1+z) (z=1 => {_f_ap_pbg_exponential(1.0):.6g}), F_AP_lcdm は _f_ap_lcdm_flat。",
        ],
    }
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    worklog.append_event(
        {
            "domain": "cosmology",
            "action": "desi_dr1_bao_y1data_alpha_crosscheck",
            "inputs": [y1_path, peakfit_path],
            "outputs": [out_png, out_json],
            "params": metrics["params"],
        }
    )
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
