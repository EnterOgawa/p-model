#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_catalog_vs_published_crosscheck.py

Step 16.4（BAO一次情報：銀河+random）/ Phase A4:
catalog-based（Corrfunc: galaxy+random）で再計算した ξℓ（ℓ=0,2）の peakfit と、
公開済み multipoles（Ross post-recon / Satpathy pre-recon）peakfit を
同一指標 ε（AP warping）で突き合わせるクロスチェック図を作る。

目的（スクリーニング）:
- catalog-based の ξℓ が「公開 multipoles と同じ特徴（ピーク位置・異方）」を
  どの程度再現しているかを可視化し、Phase B（reconstruction 自前化）へつなぐ。
- A0（座標化仕様）を固定した上で比較し、ε の変動が「理論差」か「座標化差」かを混ぜない。

出力（固定）:
- output/private/cosmology/cosmology_bao_catalog_vs_published_crosscheck.png
- output/private/cosmology/cosmology_bao_catalog_vs_published_crosscheck_metrics.json
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
    label: str
    z_eff: float
    eps: float
    err_lo: float
    err_hi: float
    meta: Dict[str, Any]


def _extract_published_eps(metrics: Dict[str, Any]) -> Dict[int, EpsPoint]:
    out: Dict[int, EpsPoint] = {}
    for r in metrics.get("results", []):
        try:
            zbin = int(r.get("zbin"))
            z_eff = float(r.get("z_eff"))
            eps = float(r["fit"]["free"]["eps"])
            ci = r["fit"]["free"].get("eps_ci_1sigma")
            e_lo, e_hi = _ci_asym_err(eps, ci)
            out[zbin] = EpsPoint(
                label=str(r.get("label", f"bin{zbin}")),
                z_eff=z_eff,
                eps=eps,
                err_lo=e_lo,
                err_hi=e_hi,
                meta={"eps_ci_1sigma": ci, "dataset": metrics.get("dataset", "")},
            )
        except Exception:
            continue
    return out


def _zbin_label_to_int(z_bin: str) -> Optional[int]:
    z = str(z_bin).strip().lower()
    if z == "b1":
        return 1
    if z == "b2":
        return 2
    if z == "b3":
        return 3
    return None


def _extract_catalog_eps(metrics: Dict[str, Any]) -> Dict[Tuple[str, int], EpsPoint]:
    out: Dict[Tuple[str, int], EpsPoint] = {}
    for r in metrics.get("results", []):
        try:
            dist = str(r.get("dist"))
            z_int = _zbin_label_to_int(str(r.get("z_bin")))
            if z_int is None:
                continue
            z_eff = float(r.get("z_eff"))
            eps = float(r["fit"]["free"]["eps"])
            ci = r["fit"]["free"].get("eps_ci_1sigma")
            e_lo, e_hi = _ci_asym_err(eps, ci)
            status = None
            try:
                status = str(r.get("screening", {}).get("status"))
            except Exception:
                status = None
            out[(dist, z_int)] = EpsPoint(
                label=f"{r.get('sample','')}/{r.get('caps','')}/{dist}/{r.get('z_bin','')}",
                z_eff=z_eff,
                eps=eps,
                err_lo=e_lo,
                err_hi=e_hi,
                meta={
                    "eps_ci_1sigma": ci,
                    "screening_status": status,
                    "inputs": r.get("inputs", {}),
                },
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
    ap = argparse.ArgumentParser(description="Cross-check BAO ε: catalog-based ξℓ vs published multipoles.")
    ap.add_argument(
        "--published-post",
        default="output/private/cosmology/cosmology_bao_xi_multipole_peakfit_metrics.json",
        help="Ross post-recon multipoles peakfit metrics JSON",
    )
    ap.add_argument(
        "--published-pre",
        default="output/private/cosmology/cosmology_bao_xi_multipole_peakfit_pre_recon_metrics.json",
        help="Satpathy pre-recon multipoles peakfit metrics JSON",
    )
    ap.add_argument(
        "--catalog",
        default="output/private/cosmology/cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly_metrics.json",
        help="catalog-based peakfit metrics JSON (cmasslowztot/combined/zbinonly)",
    )
    ap.add_argument(
        "--catalog-recon",
        default="output/private/cosmology/cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__recon_grid_iso_metrics.json",
        help="optional catalog-based recon peakfit metrics JSON (when present, include in plot)",
    )
    ap.add_argument(
        "--out-png",
        default="output/private/cosmology/cosmology_bao_catalog_vs_published_crosscheck.png",
        help="output PNG path (default: fixed)",
    )
    ap.add_argument(
        "--out-json",
        default="output/private/cosmology/cosmology_bao_catalog_vs_published_crosscheck_metrics.json",
        help="output metrics JSON path (default: fixed)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    published_post_path = (_ROOT / str(args.published_post)).resolve()
    published_pre_path = (_ROOT / str(args.published_pre)).resolve()
    catalog_path = (_ROOT / str(args.catalog)).resolve()
    catalog_recon_path = (_ROOT / str(args.catalog_recon)).resolve()
    out_png = (_ROOT / str(args.out_png)).resolve()
    out_json = (_ROOT / str(args.out_json)).resolve()

    missing = [p for p in [published_post_path, published_pre_path, catalog_path] if not p.exists()]
    if missing:
        print("[skip] missing inputs:")
        for p in missing:
            print(f"  - {p}")
        return 0

    pub_post = _load_json(published_post_path)
    pub_pre = _load_json(published_pre_path)
    cat = _load_json(catalog_path)
    cat_recon: Dict[str, Any] | None = None
    if catalog_recon_path.exists():
        try:
            cat_recon = _load_json(catalog_recon_path)
        except Exception:
            cat_recon = None

    pub_post_eps = _extract_published_eps(pub_post)
    pub_pre_eps = _extract_published_eps(pub_pre)
    cat_eps = _extract_catalog_eps(cat)
    cat_recon_eps = _extract_catalog_eps(cat_recon) if isinstance(cat_recon, dict) else {}

    z_bins = [1, 2, 3]
    # Use published z_eff as the reference x-grid for readability.
    z_ref = np.array([pub_post_eps[z].z_eff if z in pub_post_eps else float("nan") for z in z_bins], dtype=float)

    def _arr(points: Dict[int, EpsPoint]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = np.array([points[z].eps if z in points else float("nan") for z in z_bins], dtype=float)
        lo = np.array([points[z].err_lo if z in points else float("nan") for z in z_bins], dtype=float)
        hi = np.array([points[z].err_hi if z in points else float("nan") for z in z_bins], dtype=float)
        return y, lo, hi

    y_post, lo_post, hi_post = _arr(pub_post_eps)
    y_pre, lo_pre, hi_pre = _arr(pub_pre_eps)

    def _arr_cat(points: Dict[Tuple[str, int], EpsPoint], dist: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = np.array([points.get((dist, z), EpsPoint("", float("nan"), float("nan"), float("nan"), float("nan"), {})).eps for z in z_bins], dtype=float)
        lo = np.array([points.get((dist, z), EpsPoint("", float("nan"), float("nan"), float("nan"), float("nan"), {})).err_lo for z in z_bins], dtype=float)
        hi = np.array([points.get((dist, z), EpsPoint("", float("nan"), float("nan"), float("nan"), float("nan"), {})).err_hi for z in z_bins], dtype=float)
        return y, lo, hi

    y_cat_lcdm, lo_cat_lcdm, hi_cat_lcdm = _arr_cat(cat_eps, "lcdm")
    y_cat_pbg, lo_cat_pbg, hi_cat_pbg = _arr_cat(cat_eps, "pbg")
    y_cat_lcdm_rec, lo_cat_lcdm_rec, hi_cat_lcdm_rec = _arr_cat(cat_recon_eps, "lcdm")

    _set_japanese_font()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=160, sharey=True)
    ax0, ax1 = axes

    ax0.set_title("ε（peakfit）：公開 ξℓ vs catalog-based ξℓ（LCDM座標）")
    ax0.axhline(0.0, color="0.6", lw=1, ls="--")
    _errorbar(ax0, x=z_ref, y=y_post, err_lo=lo_post, err_hi=hi_post, label="Ross 2016（post-recon）", color="#1f77b4", marker="o", x_offset=-0.004)
    _errorbar(ax0, x=z_ref, y=y_pre, err_lo=lo_pre, err_hi=hi_pre, label="Satpathy 2016（pre-recon）", color="#9467bd", marker="s", x_offset=0.000)
    _errorbar(ax0, x=z_ref, y=y_cat_lcdm, err_lo=lo_cat_lcdm, err_hi=hi_cat_lcdm, label="catalog再計算（lcdm）", color="#ff7f0e", marker="^", x_offset=+0.004)
    if np.any(np.isfinite(y_cat_lcdm_rec)):
        _errorbar(
            ax0,
            x=z_ref,
            y=y_cat_lcdm_rec,
            err_lo=lo_cat_lcdm_rec,
            err_hi=hi_cat_lcdm_rec,
            label="catalog再計算（lcdm, recon_grid_iso）",
            color="#2ca02c",
            marker="x",
            x_offset=+0.008,
        )

    ax0.set_xlabel("z_eff")
    ax0.set_ylabel("ε（AP warping）")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best", fontsize=8)

    ax1.set_title("ε（peakfit）：catalog距離写像の差（lcdm vs P_bg）")
    ax1.axhline(0.0, color="0.6", lw=1, ls="--")
    _errorbar(ax1, x=z_ref, y=y_cat_lcdm, err_lo=lo_cat_lcdm, err_hi=hi_cat_lcdm, label="catalog（lcdm）", color="#ff7f0e", marker="^", x_offset=-0.003)
    _errorbar(ax1, x=z_ref, y=y_cat_pbg, err_lo=lo_cat_pbg, err_hi=hi_cat_pbg, label="catalog（P_bg）", color="#2ca02c", marker="o", x_offset=+0.003)

    ax1.set_xlabel("z_eff")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=8)

    fig.suptitle("BOSS DR12（z-bin 0.38/0.51/0.61）：εクロスチェック（スクリーニング）", y=1.02)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "16.4 (BAO catalog vs published multipoles cross-check)",
        "inputs": {
            "published_post": str(published_post_path),
            "published_pre": str(published_pre_path),
            "catalog": str(catalog_path),
            "catalog_recon": str(catalog_recon_path) if catalog_recon_path.exists() else None,
        },
        "notes": [
            "Published の ε は公開 multipoles（ξ0, ξ2）に対する smooth+peak peakfit の結果。",
            "Catalog の ε は Corrfunc（galaxy+random）で計算した ξ(s,μ) から ξℓ を積分して同じ peakfit を適用した結果（Phase A: screening）。",
            "Catalog の誤差は diag paircount proxy に基づく粗い CI（Phase B で共分散・reconstruction を自前化して更新予定）。",
            "Catalog recon は、銀河+randomから簡易reconstruction（grid; recon-mode=iso）を適用した ξℓ に同じ peakfit を適用した結果（公開post-reconとの距離感を確認するための入口）。",
        ],
        "results": {
            "z_bins": z_bins,
            "z_eff_ref": [float(x) if np.isfinite(x) else None for x in z_ref],
            "published_post": {str(z): pub_post_eps[z].__dict__ for z in z_bins if z in pub_post_eps},
            "published_pre": {str(z): pub_pre_eps[z].__dict__ for z in z_bins if z in pub_pre_eps},
            "catalog_lcdm": {str(z): cat_eps.get(("lcdm", z)).__dict__ for z in z_bins if ("lcdm", z) in cat_eps},
            "catalog_pbg": {str(z): cat_eps.get(("pbg", z)).__dict__ for z in z_bins if ("pbg", z) in cat_eps},
            "catalog_recon_lcdm": {str(z): cat_recon_eps.get(("lcdm", z)).__dict__ for z in z_bins if ("lcdm", z) in cat_recon_eps},
        },
        "diagnostics": {},
        "outputs": {"png": str(out_png), "metrics_json": str(out_json)},
    }

    # Simple delta diagnostics (symmetrized 1σ).
    deltas: Dict[str, Any] = {}
    for z in z_bins:
        if z not in pub_post_eps:
            continue
        if ("lcdm", z) not in cat_eps:
            continue
        p = pub_post_eps[z]
        c = cat_eps[("lcdm", z)]
        sigma = np.sqrt(_sym_sigma_from_ci(p.eps, p.meta.get("eps_ci_1sigma")) ** 2 + _sym_sigma_from_ci(c.eps, c.meta.get("eps_ci_1sigma")) ** 2)
        deltas[str(z)] = {
            "eps_catalog_minus_published_post": float(c.eps - p.eps),
            "sigma_combined_sym_1sigma": float(sigma) if np.isfinite(sigma) else None,
            "abs_sigma_sym": float(abs(c.eps - p.eps) / sigma) if np.isfinite(sigma) and sigma > 0 else None,
        }
    payload["diagnostics"]["delta_eps_lcdm_minus_published_post"] = deltas

    deltas_recon: Dict[str, Any] = {}
    for z in z_bins:
        if z not in pub_post_eps:
            continue
        if ("lcdm", z) not in cat_recon_eps:
            continue
        p = pub_post_eps[z]
        c = cat_recon_eps[("lcdm", z)]
        sigma = np.sqrt(_sym_sigma_from_ci(p.eps, p.meta.get("eps_ci_1sigma")) ** 2 + _sym_sigma_from_ci(c.eps, c.meta.get("eps_ci_1sigma")) ** 2)
        deltas_recon[str(z)] = {
            "eps_catalog_recon_minus_published_post": float(c.eps - p.eps),
            "sigma_combined_sym_1sigma": float(sigma) if np.isfinite(sigma) else None,
            "abs_sigma_sym": float(abs(c.eps - p.eps) / sigma) if np.isfinite(sigma) and sigma > 0 else None,
        }
    payload["diagnostics"]["delta_eps_lcdm_recon_minus_published_post"] = deltas_recon

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_bao_catalog_vs_published_crosscheck",
                "argv": sys.argv,
                "outputs": {"png": out_png, "metrics_json": out_json},
                "metrics": {
                    "published_post": str(published_post_path),
                    "published_pre": str(published_pre_path),
                    "catalog": str(catalog_path),
                    "catalog_recon": str(catalog_recon_path) if catalog_recon_path.exists() else None,
                },
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
