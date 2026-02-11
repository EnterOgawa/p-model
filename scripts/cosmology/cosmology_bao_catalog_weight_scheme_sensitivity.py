#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_catalog_weight_scheme_sensitivity.py

Step 16.4（BAO一次情報：銀河+random）/ Phase A5（系統切り分け）:
catalog-based ξℓ（Corrfunc: galaxy+random）において、重み付け仕様（weight scheme）が
peakfit の ε（AP warping）へ与える影響を可視化する。

狙い：
- 「理論差」ではなく「座標化/重み仕様差」で ε が動く混入を避けるため、
  A0（z_source/LOS/距離積分）に加えて重み仕様も固定・記録する。

出力（固定）:
- output/cosmology/cosmology_bao_catalog_weight_scheme_sensitivity.png
- output/cosmology/cosmology_bao_catalog_weight_scheme_sensitivity_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ci_asym_err(eps: float, ci_1sigma: Any) -> Tuple[float, float]:
    try:
        lo, hi = float(ci_1sigma[0]), float(ci_1sigma[1])
        return max(0.0, eps - lo), max(0.0, hi - eps)
    except Exception:
        return float("nan"), float("nan")


def _sym_sigma_from_ci(eps: float, ci_1sigma: Any) -> float:
    lo, hi = _ci_asym_err(eps, ci_1sigma)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return float("nan")
    return float(0.5 * (lo + hi))


@dataclass(frozen=True)
class EpsPoint:
    zbin: int
    z_eff: float
    eps: float
    err_lo: float
    err_hi: float
    meta: Dict[str, Any]


def _zbin_label_to_int(z_bin: str) -> Optional[int]:
    z = str(z_bin).strip().lower()
    if z == "b1":
        return 1
    if z == "b2":
        return 2
    if z == "b3":
        return 3
    return None


def _extract_catalog_eps_peakfit(metrics: Dict[str, Any], *, dist: str) -> Dict[int, EpsPoint]:
    out: Dict[int, EpsPoint] = {}
    for r in metrics.get("results", []):
        try:
            if str(r.get("dist")) != str(dist):
                continue
            z_int = _zbin_label_to_int(str(r.get("z_bin")))
            if z_int is None:
                continue
            z_eff = float(r.get("z_eff"))
            eps = float(r["fit"]["free"]["eps"])
            ci = r["fit"]["free"].get("eps_ci_1sigma")
            lo, hi = _ci_asym_err(eps, ci)
            out[z_int] = EpsPoint(
                zbin=z_int,
                z_eff=z_eff,
                eps=eps,
                err_lo=lo,
                err_hi=hi,
                meta={
                    "eps_ci_1sigma": ci,
                    "alpha": r["fit"]["free"].get("alpha"),
                    "screening_status": (r.get("screening", {}) or {}).get("status"),
                    "inputs": r.get("inputs", {}),
                },
            )
        except Exception:
            continue
    return out


def _extract_published_eps(metrics: Dict[str, Any]) -> Dict[int, EpsPoint]:
    out: Dict[int, EpsPoint] = {}
    for r in metrics.get("results", []):
        try:
            zbin = int(r.get("zbin"))
            z_eff = float(r.get("z_eff"))
            eps = float(r["fit"]["free"]["eps"])
            ci = r["fit"]["free"].get("eps_ci_1sigma")
            lo, hi = _ci_asym_err(eps, ci)
            out[zbin] = EpsPoint(
                zbin=zbin,
                z_eff=z_eff,
                eps=eps,
                err_lo=lo,
                err_hi=hi,
                meta={"eps_ci_1sigma": ci, "alpha": r["fit"]["free"].get("alpha"), "dataset": metrics.get("dataset", "")},
            )
        except Exception:
            continue
    return out


def _errorbar(
    ax: Any,
    *,
    x: np.ndarray,
    y: np.ndarray,
    err_lo: np.ndarray,
    err_hi: np.ndarray,
    label: str,
    color: str,
    marker: str,
    x_offset: float,
) -> None:
    x2 = np.asarray(x, dtype=float) + float(x_offset)
    yerr = np.vstack([np.asarray(err_lo, dtype=float), np.asarray(err_hi, dtype=float)])
    ax.errorbar(
        x2,
        y,
        yerr=yerr,
        fmt=marker,
        linestyle="none",
        color=color,
        label=label,
        capsize=3,
        markersize=6,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BAO catalog-based ε sensitivity to weight schemes.")
    ap.add_argument(
        "--published-post",
        default="output/cosmology/cosmology_bao_xi_multipole_peakfit_metrics.json",
        help="published post-recon multipoles peakfit (Ross)",
    )
    ap.add_argument(
        "--catalog-boss",
        default="output/cosmology/cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly_metrics.json",
        help="catalog peakfit metrics (boss_default weights; out_tag=none)",
    )
    ap.add_argument(
        "--catalog-fkp-only",
        default="output/cosmology/cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__w_fkp_only_metrics.json",
        help="catalog peakfit metrics (fkp_only weights)",
    )
    ap.add_argument(
        "--catalog-none",
        default="output/cosmology/cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__w_none_metrics.json",
        help="catalog peakfit metrics (no weights)",
    )
    ap.add_argument(
        "--out-png",
        default="output/cosmology/cosmology_bao_catalog_weight_scheme_sensitivity.png",
        help="output png",
    )
    ap.add_argument(
        "--out-json",
        default="output/cosmology/cosmology_bao_catalog_weight_scheme_sensitivity_metrics.json",
        help="output metrics json",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    p_pub = (_ROOT / str(args.published_post)).resolve()
    p_boss = (_ROOT / str(args.catalog_boss)).resolve()
    p_fkp = (_ROOT / str(args.catalog_fkp_only)).resolve()
    p_none = (_ROOT / str(args.catalog_none)).resolve()
    out_png = (_ROOT / str(args.out_png)).resolve()
    out_json = (_ROOT / str(args.out_json)).resolve()

    missing = [p for p in [p_pub, p_boss, p_fkp, p_none] if not p.exists()]
    if missing:
        print("[skip] missing inputs:")
        for p in missing:
            print(f"  - {p}")
        return 0

    pub = _extract_published_eps(_load_json(p_pub))
    boss = _extract_catalog_eps_peakfit(_load_json(p_boss), dist="lcdm")
    fkp = _extract_catalog_eps_peakfit(_load_json(p_fkp), dist="lcdm")
    none = _extract_catalog_eps_peakfit(_load_json(p_none), dist="lcdm")

    z_bins = [1, 2, 3]
    z_ref = np.array([pub[z].z_eff if z in pub else float("nan") for z in z_bins], dtype=float)

    def _arr(points: Dict[int, EpsPoint]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = np.array([points[z].eps if z in points else float("nan") for z in z_bins], dtype=float)
        lo = np.array([points[z].err_lo if z in points else float("nan") for z in z_bins], dtype=float)
        hi = np.array([points[z].err_hi if z in points else float("nan") for z in z_bins], dtype=float)
        return y, lo, hi

    y_pub, lo_pub, hi_pub = _arr(pub)
    y_boss, lo_boss, hi_boss = _arr(boss)
    y_fkp, lo_fkp, hi_fkp = _arr(fkp)
    y_none, lo_none, hi_none = _arr(none)

    _set_japanese_font()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=160, sharey=False)
    ax0, ax1 = axes

    ax0.set_title("ε（peakfit）: 重み仕様の感度（catalog ξℓ; LCDM座標）")
    ax0.axhline(0.0, color="0.6", lw=1, ls="--")
    _errorbar(ax0, x=z_ref, y=y_pub, err_lo=lo_pub, err_hi=hi_pub, label="published（Ross 2016; post-recon）", color="#1f77b4", marker="o", x_offset=-0.006)
    _errorbar(ax0, x=z_ref, y=y_boss, err_lo=lo_boss, err_hi=hi_boss, label="catalog（boss_default）", color="#ff7f0e", marker="^", x_offset=-0.002)
    _errorbar(ax0, x=z_ref, y=y_fkp, err_lo=lo_fkp, err_hi=hi_fkp, label="catalog（fkp_only）", color="#2ca02c", marker="s", x_offset=+0.002)
    _errorbar(ax0, x=z_ref, y=y_none, err_lo=lo_none, err_hi=hi_none, label="catalog（none）", color="#d62728", marker="v", x_offset=+0.006)
    ax0.set_xlabel("z_eff")
    ax0.set_ylabel("ε（AP warping）")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best", fontsize=8)

    ax1.set_title("Δε = ε(catalog) − ε(published)")
    ax1.axhline(0.0, color="0.6", lw=1, ls="--")

    def _delta(points: Dict[int, EpsPoint], name: str, color: str, x_offset: float) -> Dict[str, Any]:
        y = []
        yerr = []
        for z in z_bins:
            if z not in pub or z not in points:
                y.append(float("nan"))
                yerr.append(float("nan"))
                continue
            p = pub[z]
            c = points[z]
            d = float(c.eps - p.eps)
            sigma = float(
                np.sqrt(_sym_sigma_from_ci(p.eps, p.meta.get("eps_ci_1sigma")) ** 2 + _sym_sigma_from_ci(c.eps, c.meta.get("eps_ci_1sigma")) ** 2)
            )
            y.append(d)
            yerr.append(sigma)
        y = np.asarray(y, dtype=float)
        yerr = np.asarray(yerr, dtype=float)
        ok = np.isfinite(z_ref) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
        ax1.errorbar(
            z_ref[ok] + float(x_offset),
            y[ok],
            yerr=yerr[ok],
            fmt="o",
            linestyle="none",
            color=color,
            capsize=3,
            markersize=6,
            label=name,
        )
        return {str(z): {"delta": float(y[i]) if np.isfinite(y[i]) else None, "sigma_sym": float(yerr[i]) if np.isfinite(yerr[i]) else None} for i, z in enumerate(z_bins)}

    deltas = {
        "boss_default": _delta(boss, "boss_default", "#ff7f0e", -0.002),
        "fkp_only": _delta(fkp, "fkp_only", "#2ca02c", +0.002),
        "none": _delta(none, "none", "#d62728", +0.006),
    }

    ax1.set_xlabel("z_eff")
    ax1.set_ylabel("Δε（1σ合成誤差）")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=8)

    fig.suptitle("BOSS DR12（z-bin 0.38/0.51/0.61）：重み付け仕様の感度（Phase A5）", y=1.02)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "16.4 (BAO catalog-based weight scheme sensitivity)",
        "inputs": {
            "published_post": str(p_pub),
            "catalog_boss_default": str(p_boss),
            "catalog_fkp_only": str(p_fkp),
            "catalog_none": str(p_none),
        },
        "results": {
            "z_bins": z_bins,
            "z_eff_ref": [float(x) if np.isfinite(x) else None for x in z_ref],
            "eps": {
                "published_post": {str(z): pub[z].__dict__ for z in z_bins if z in pub},
                "boss_default": {str(z): boss[z].__dict__ for z in z_bins if z in boss},
                "fkp_only": {str(z): fkp[z].__dict__ for z in z_bins if z in fkp},
                "none": {str(z): none[z].__dict__ for z in z_bins if z in none},
            },
            "delta_eps_vs_published": deltas,
        },
        "outputs": {"png": str(out_png), "metrics_json": str(out_json)},
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_bao_catalog_weight_scheme_sensitivity",
                "argv": sys.argv,
                "outputs": {"png": out_png, "metrics_json": out_json},
                "metrics": {"published_post": str(p_pub), "catalog_boss": str(p_boss), "catalog_fkp_only": str(p_fkp), "catalog_none": str(p_none)},
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

