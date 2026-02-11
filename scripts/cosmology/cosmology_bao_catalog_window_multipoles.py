#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_catalog_window_multipoles.py

Phase 4 / Step 4.5（BAOを一次統計から再構築）Stage B（確証決定打）:
catalog-based（Corrfunc; galaxy+random）で得た pair-count grid（RR0/SS）から、
窓関数（survey selection/geometry）の異方性を簡易に可視化する。

狙い：
- Ross post-recon との ξ2 ギャップが「理論差」ではなく、窓関数/選択関数/正規化の差に起因していないかを切り分ける。
- ここでは RR(s,μ) の multipoles 比（RR2/RR0）を window anisotropy 指標として使う。

出力（固定）:
- output/private/cosmology/cosmology_bao_catalog_window_multipoles.png
- output/private/cosmology/cosmology_bao_catalog_window_multipoles_metrics.json

注意：
- 本スクリプト自体は Corrfunc を使わず、既に生成済みの npz を読むだけ（Windowsでも実行可）。
- Corrfunc計算（npz生成）は AGENTS.md の通り WSL（threads=24）で行う。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = ["Yu Gothic", "Meiryo", "BIZ UDGothic", "MS Gothic", "Yu Mincho", "MS Mincho"]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        if not chosen:
            return
        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _l2(mu: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64)
    return 0.5 * (3.0 * mu * mu - 1.0)


def _ratio_r2_over_r0(*, counts: np.ndarray, mu_edges: np.ndarray) -> np.ndarray:
    """
    Return r2/r0 where:
      r0(s) = ∫ RR(s,μ) dμ  (approx: sum over μ bins)
      r2(s) = 5 ∫ RR(s,μ) L2(μ) dμ
    Using μ in [0,1] bins (Corrfunc convention; evenness assumed).
    """
    counts = np.asarray(counts, dtype=np.float64)
    mu_edges = np.asarray(mu_edges, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError("counts must be 2D (nbins,nmu)")
    if mu_edges.ndim != 1 or mu_edges.size < 2:
        raise ValueError("mu_edges must be 1D (nmu+1)")
    nmu = int(mu_edges.size - 1)
    if counts.shape[1] != nmu:
        raise ValueError(f"nmu mismatch: counts={counts.shape[1]} mu_edges={nmu}")

    mu_mid = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    r0 = np.sum(counts, axis=1)
    r2 = 5.0 * np.sum(counts * _l2(mu_mid)[None, :], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = r2 / r0
    return np.where(np.isfinite(out), out, np.nan)


@dataclass(frozen=True)
class WindowSeries:
    s: np.ndarray
    r2_over_r0: np.ndarray


def _load_window_series(path: Path, *, key: str) -> WindowSeries:
    with np.load(path) as z:
        s = np.asarray(z["s"], dtype=np.float64)
        mu_edges = np.asarray(z["mu_edges"], dtype=np.float64)
        if key not in z.files:
            raise KeyError(f"missing {key} in npz: {path}")
        counts = np.asarray(z[key], dtype=np.float64)
    return WindowSeries(s=s, r2_over_r0=_ratio_r2_over_r0(counts=counts, mu_edges=mu_edges))


def _safe_max_abs_in_range(s: np.ndarray, y: np.ndarray, *, s_min: float, s_max: float) -> float:
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = (s >= float(s_min)) & (s <= float(s_max)) & np.isfinite(y)
    if not np.any(m):
        return float("nan")
    return float(np.nanmax(np.abs(y[m])))


def _path_for(
    *,
    sample: str,
    caps: str,
    dist: str,
    zbin: str,
    suffix: str,
) -> Path:
    base = f"cosmology_bao_xi_from_catalogs_{sample}_{caps}_{dist}_{zbin}"
    if suffix:
        base = f"{base}{suffix}"
    return _ROOT / "output" / "private" / "cosmology" / f"{base}.npz"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compute window anisotropy proxy RR2/RR0 from catalog-based BAO NPZ.")
    ap.add_argument("--sample", default="cmasslowztot", help="sample (default: cmasslowztot)")
    ap.add_argument("--caps", default="combined", help="caps (default: combined)")
    ap.add_argument("--dist", default="lcdm", help="distance model (default: lcdm)")
    ap.add_argument(
        "--suffixes",
        default=";__recon_grid_iso;__recon_grid_ani_rsdshift0",
        help="semicolon-separated NPZ suffixes to include (default: ';__recon_grid_iso;__recon_grid_ani_rsdshift0')",
    )
    ap.add_argument("--out-png", default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_bao_catalog_window_multipoles.png"))
    ap.add_argument(
        "--out-json",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_bao_catalog_window_multipoles_metrics.json"),
    )
    ap.add_argument("--s-min", type=float, default=30.0, help="evaluation s_min [Mpc/h] (default: 30)")
    ap.add_argument("--s-max", type=float, default=150.0, help="evaluation s_max [Mpc/h] (default: 150)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    sample = str(args.sample)
    caps = str(args.caps)
    dist = str(args.dist)
    suffixes = [s.strip() for s in str(args.suffixes).split(";")]
    # Keep empty suffix ("") to represent pre-recon.
    suffixes = [s for s in suffixes if s != "" or True]

    out_png = Path(args.out_png)
    out_json = Path(args.out_json)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    zbins = ["b1", "b2", "b3"]

    # Load available series.
    loaded: Dict[str, Dict[str, Dict[str, WindowSeries]]] = {}
    missing: list[str] = []
    for zbin in zbins:
        loaded[zbin] = {}
        for suf in suffixes:
            suf_norm = suf
            if suf_norm and not suf_norm.startswith("__"):
                suf_norm = "__" + suf_norm
            p = _path_for(sample=sample, caps=caps, dist=dist, zbin=zbin, suffix=suf_norm)
            if not p.exists():
                missing.append(str(p))
                continue
            try:
                rr = _load_window_series(p, key="rr_w")
                series: Dict[str, WindowSeries] = {"RR0": rr}
                # Recon npz also has SS; pre-recon won't.
                try:
                    ss = _load_window_series(p, key="ss_w")
                    series["SS"] = ss
                except Exception:
                    pass
                loaded[zbin][suf_norm or ""] = {"path": str(p), **series}  # type: ignore[dict-item]
            except Exception:
                missing.append(str(p))

    # If nothing loaded, skip gracefully (run_all optional).
    any_loaded = any(bool(loaded[z]) for z in zbins)
    if not any_loaded:
        print("[skip] cosmology_bao_catalog_window_multipoles: no input npz found")
        return 0

    # Plot
    _set_japanese_font()
    import matplotlib.pyplot as plt  # noqa: E402

    fig, axes = plt.subplots(len(zbins), 2, figsize=(12, 8), sharex=True, sharey=True)
    if len(zbins) == 1:
        axes = np.asarray([axes])

    colors = {
        "": "#7f7f7f",  # pre-recon
        "__recon_grid_iso": "#1f77b4",
        "__recon_grid_ani_rsdshift0": "#ff7f0e",
    }

    for i, zbin in enumerate(zbins):
        ax_rr = axes[i, 0]
        ax_ss = axes[i, 1]
        ax_rr.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
        ax_ss.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")

        variants = loaded[zbin]
        for suf, payload in variants.items():
            c = colors.get(suf, None) or "#2ca02c"
            label = suf.lstrip("_") if suf else "pre"
            rr: WindowSeries = payload["RR0"]  # type: ignore[assignment]
            ax_rr.plot(rr.s, rr.r2_over_r0, color=c, linewidth=1.5, alpha=0.9, label=label)
            if "SS" in payload:
                ss: WindowSeries = payload["SS"]  # type: ignore[assignment]
                ax_ss.plot(ss.s, ss.r2_over_r0, color=c, linewidth=1.5, alpha=0.9, label=label)

        ax_rr.set_title(f"{zbin}: RR0 の異方（RR2/RR0）")
        ax_ss.set_title(f"{zbin}: SS の異方（SS2/SS0）")
        ax_rr.grid(True, alpha=0.3)
        ax_ss.grid(True, alpha=0.3)

        if i == len(zbins) - 1:
            ax_rr.set_xlabel("s [Mpc/h]")
            ax_ss.set_xlabel("s [Mpc/h]")
        ax_rr.set_ylabel("ratio (ℓ=2 / ℓ=0)")

    axes[0, 0].legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.suptitle(f"BAO 窓関数（RR/SS）異方性の診断：{sample}/{caps}/{dist}", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    # Metrics
    metrics: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "4.5B (window/selection diagnostic)",
        "inputs": {
            "sample": sample,
            "caps": caps,
            "dist": dist,
            "suffixes": suffixes,
            "s_eval_range_mpc_over_h": [float(args.s_min), float(args.s_max)],
            "npz_missing": missing[:50],  # cap
        },
        "results": {},
    }
    for zbin in zbins:
        out_z: Dict[str, Any] = {}
        for suf, payload in loaded[zbin].items():
            rr: WindowSeries = payload["RR0"]  # type: ignore[assignment]
            entry: Dict[str, Any] = {
                "npz": payload.get("path"),
                "rr0": {
                    "max_abs_r2_over_r0_in_eval_range": _safe_max_abs_in_range(
                        rr.s, rr.r2_over_r0, s_min=float(args.s_min), s_max=float(args.s_max)
                    )
                },
            }
            if "SS" in payload:
                ss: WindowSeries = payload["SS"]  # type: ignore[assignment]
                entry["ss"] = {
                    "max_abs_r2_over_r0_in_eval_range": _safe_max_abs_in_range(
                        ss.s, ss.r2_over_r0, s_min=float(args.s_min), s_max=float(args.s_max)
                    )
                }
            out_z[suf or "pre"] = entry
        metrics["results"][zbin] = out_z
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    worklog.append_event(
        {
            "event": "cosmology_bao_catalog_window_multipoles",
            "generated_utc": metrics["generated_utc"],
            "inputs": metrics["inputs"],
            "outputs": {
                "png": str(out_png),
                "json": str(out_json),
            },
        }
    )

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

