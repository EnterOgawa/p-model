#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_catalog_random_max_rows_sensitivity.py

Step 16.4（BAO一次情報：銀河+random）/ Phase A6（差分要因の切り分け）:
catalog-based ξℓ（Corrfunc: galaxy+random）において、random カタログの行数（max_rows）
が peakfit の ε（AP warping）へ与える影響を可視化する。

狙い：
- published multipoles（Ross post-recon）と catalog-based の ε 差を議論する前に、
  random のサンプル数（/抽出方法）が ε をどの程度揺らすかを固定する。
- 本スクリプトは Phase A の品質管理（スクリーニング）であり、最終的な共分散や
  reconstruction 自前化（Phase B）で更新される。

出力（固定）:
- output/cosmology/cosmology_bao_catalog_random_max_rows_sensitivity.png
- output/cosmology/cosmology_bao_catalog_random_max_rows_sensitivity_metrics.json
"""

from __future__ import annotations

import argparse
import json
import re
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


_RE_PREFIX = re.compile(r"\.prefix_(\d+)\.npz$", re.IGNORECASE)
_RE_RES = re.compile(r"\.reservoir_(\d+)_seed(\d+)(?:_scan(\d+))?\.npz$", re.IGNORECASE)


def _parse_random_npz_name(name: str) -> Optional[Dict[str, Any]]:
    n = str(name)
    m = _RE_PREFIX.search(n)
    if m:
        return {"method": "prefix_rows", "max_rows": int(m.group(1))}
    m = _RE_RES.search(n)
    if m:
        return {
            "method": "reservoir",
            "sample_rows": int(m.group(1)),
            "seed": int(m.group(2)),
            "scan_max_rows": None if m.group(3) is None else int(m.group(3)),
        }
    return None


def _infer_random_sampling_from_xi_metrics(xi_metrics_path: Path) -> Dict[str, Any]:
    d = _load_json(xi_metrics_path)
    rnd = d.get("inputs", {}).get("random_npz")
    paths = rnd if isinstance(rnd, list) else []
    parsed = []
    for p in paths:
        try:
            name = Path(str(p)).name
            meta = _parse_random_npz_name(name)
            if meta:
                parsed.append(meta)
        except Exception:
            continue
    if not parsed:
        return {"unknown": True, "random_npz": paths}
    # Guard: require consistent sampling across caps.
    keys = {json.dumps(x, sort_keys=True) for x in parsed}
    if len(keys) != 1:
        return {"inconsistent": True, "by_cap": parsed, "random_npz": paths}
    out = parsed[0].copy()
    out["random_npz"] = paths
    return out


def _format_sampling_label(meta: Dict[str, Any]) -> str:
    if meta.get("unknown"):
        return "unknown"
    if meta.get("inconsistent"):
        return "inconsistent"
    if str(meta.get("method")) == "prefix_rows":
        n = int(meta.get("max_rows"))
        if n >= 1_000_000:
            return f"prefix {n/1_000_000:.1f}M"
        return f"prefix {n/1_000:.0f}k"
    if str(meta.get("method")) == "reservoir":
        n = int(meta.get("sample_rows"))
        if n >= 1_000_000:
            return f"reservoir {n/1_000_000:.1f}M"
        return f"reservoir {n/1_000:.0f}k"
    return str(meta.get("method") or "unknown")


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
    ap = argparse.ArgumentParser(description="BAO catalog-based ε sensitivity to random max_rows.")
    ap.add_argument(
        "--published-post",
        default="output/cosmology/cosmology_bao_xi_multipole_peakfit_metrics.json",
        help="published post-recon multipoles peakfit (Ross)",
    )
    ap.add_argument(
        "--catalog-baseline",
        default="output/cosmology/cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly_metrics.json",
        help="catalog peakfit metrics (baseline; out_tag=none)",
    )
    ap.add_argument(
        "--catalog-alt",
        default="output/cosmology/cosmology_bao_catalog_peakfit_cmasslowztot_combined_zbinonly__rnd_prefix_500k_metrics.json",
        help="catalog peakfit metrics (alt; e.g. random prefix 500k)",
    )
    ap.add_argument(
        "--out-png",
        default="output/cosmology/cosmology_bao_catalog_random_max_rows_sensitivity.png",
        help="output png",
    )
    ap.add_argument(
        "--out-json",
        default="output/cosmology/cosmology_bao_catalog_random_max_rows_sensitivity_metrics.json",
        help="output metrics json",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    p_pub = (_ROOT / str(args.published_post)).resolve()
    p_base = (_ROOT / str(args.catalog_baseline)).resolve()
    p_alt = (_ROOT / str(args.catalog_alt)).resolve()
    out_png = (_ROOT / str(args.out_png)).resolve()
    out_json = (_ROOT / str(args.out_json)).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    pub = _extract_published_eps(_load_json(p_pub))
    base_pts = _extract_catalog_eps_peakfit(_load_json(p_base), dist="lcdm")
    alt_pts = _extract_catalog_eps_peakfit(_load_json(p_alt), dist="lcdm") if p_alt.exists() else {}

    # Infer random sampling spec from the xi metrics referenced by peakfit inputs.
    def _sampling_for_case(points: Dict[int, EpsPoint]) -> Dict[str, Any]:
        metas = []
        for zbin, pt in points.items():
            mi = (pt.meta.get("inputs", {}) or {}).get("metrics_json")
            if not mi:
                continue
            try:
                xi_path = Path(str(mi))
                metas.append(_infer_random_sampling_from_xi_metrics(xi_path))
            except Exception:
                continue
        if not metas:
            return {"unknown": True}
        keys = {json.dumps(m, sort_keys=True) for m in metas}
        if len(keys) != 1:
            return {"inconsistent": True, "by_zbin": metas}
        return metas[0]

    sampling_base = _sampling_for_case(base_pts)
    sampling_alt = _sampling_for_case(alt_pts) if alt_pts else {"missing": True}

    z_bins = [1, 2, 3]
    z_eff_ref = [float(pub[z].z_eff) if z in pub else float("nan") for z in z_bins]

    _set_japanese_font()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    ax0, ax1 = axes[0], axes[1]

    # Plot eps
    xs = np.asarray(z_eff_ref, dtype=float)
    _errorbar(
        ax0,
        x=xs,
        y=np.asarray([pub[z].eps if z in pub else np.nan for z in z_bins], dtype=float),
        err_lo=np.asarray([pub[z].err_lo if z in pub else np.nan for z in z_bins], dtype=float),
        err_hi=np.asarray([pub[z].err_hi if z in pub else np.nan for z in z_bins], dtype=float),
        label="published (Ross post)",
        color="black",
        marker="o",
        x_offset=-0.003,
    )

    _errorbar(
        ax0,
        x=np.asarray([base_pts[z].z_eff if z in base_pts else np.nan for z in z_bins], dtype=float),
        y=np.asarray([base_pts[z].eps if z in base_pts else np.nan for z in z_bins], dtype=float),
        err_lo=np.asarray([base_pts[z].err_lo if z in base_pts else np.nan for z in z_bins], dtype=float),
        err_hi=np.asarray([base_pts[z].err_hi if z in base_pts else np.nan for z in z_bins], dtype=float),
        label=f"catalog ({_format_sampling_label(sampling_base)})",
        color="#1f77b4",
        marker="s",
        x_offset=0.0,
    )

    if alt_pts:
        _errorbar(
            ax0,
            x=np.asarray([alt_pts[z].z_eff if z in alt_pts else np.nan for z in z_bins], dtype=float),
            y=np.asarray([alt_pts[z].eps if z in alt_pts else np.nan for z in z_bins], dtype=float),
            err_lo=np.asarray([alt_pts[z].err_lo if z in alt_pts else np.nan for z in z_bins], dtype=float),
            err_hi=np.asarray([alt_pts[z].err_hi if z in alt_pts else np.nan for z in z_bins], dtype=float),
            label=f"catalog ({_format_sampling_label(sampling_alt)})",
            color="#ff7f0e",
            marker="^",
            x_offset=0.003,
        )

    ax0.set_title("ε (AP warping) vs z")
    ax0.set_xlabel("z_eff")
    ax0.set_ylabel("ε")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=9)

    # Plot delta eps vs published
    def _delta(points: Dict[int, EpsPoint]) -> Tuple[np.ndarray, np.ndarray]:
        dy = []
        ds = []
        for z in z_bins:
            if z not in points or z not in pub:
                dy.append(np.nan)
                ds.append(np.nan)
                continue
            dy.append(points[z].eps - pub[z].eps)
            s = np.sqrt(_sym_sigma_from_ci(pub[z].eps, pub[z].meta.get("eps_ci_1sigma")) ** 2 + _sym_sigma_from_ci(points[z].eps, points[z].meta.get("eps_ci_1sigma")) ** 2)
            ds.append(float(s))
        return np.asarray(dy, dtype=float), np.asarray(ds, dtype=float)

    d_base, s_base = _delta(base_pts)
    ax1.plot(xs, d_base, "-s", color="#1f77b4", label=f"Δε ({_format_sampling_label(sampling_base)})")
    ax1.fill_between(xs, d_base - s_base, d_base + s_base, color="#1f77b4", alpha=0.15)

    if alt_pts:
        d_alt, s_alt = _delta(alt_pts)
        ax1.plot(xs, d_alt, "-^", color="#ff7f0e", label=f"Δε ({_format_sampling_label(sampling_alt)})")
        ax1.fill_between(xs, d_alt - s_alt, d_alt + s_alt, color="#ff7f0e", alpha=0.15)

    ax1.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax1.set_title("Δε = ε_catalog − ε_published (±1σ sym)")
    ax1.set_xlabel("z_eff (published)")
    ax1.set_ylabel("Δε")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    fig.suptitle("BAO catalog-based ε sensitivity: random max_rows", fontsize=12)
    fig.savefig(out_png, dpi=160)

    out: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "16.4 (BAO catalog-based random max_rows sensitivity)",
        "inputs": {
            "published_post": str(p_pub),
            "catalog_baseline": str(p_base),
            "catalog_alt": str(p_alt) if p_alt.exists() else None,
        },
        "results": {
            "z_bins": z_bins,
            "z_eff_ref": z_eff_ref,
            "sampling": {"baseline": sampling_base, "alt": sampling_alt},
            "eps": {
                "published_post": {str(k): pub[k].__dict__ for k in pub},
                "baseline": {str(k): base_pts[k].__dict__ for k in base_pts},
                "alt": {str(k): alt_pts[k].__dict__ for k in alt_pts} if alt_pts else {},
            },
            "delta_eps_vs_published": {
                "baseline": {str(z): {"delta": float(d_base[i]), "sigma_sym": float(s_base[i])} for i, z in enumerate(z_bins)},
                "alt": {str(z): {"delta": float(_delta(alt_pts)[0][i]), "sigma_sym": float(_delta(alt_pts)[1][i])} for i, z in enumerate(z_bins)}
                if alt_pts
                else {},
            },
        },
        "outputs": {"png": str(out_png), "metrics_json": str(out_json)},
    }
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_bao_catalog_random_max_rows_sensitivity",
                "generated_utc": out["generated_utc"],
                "inputs": out["inputs"],
                "outputs": out["outputs"],
            }
        )
    except Exception:
        pass

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

