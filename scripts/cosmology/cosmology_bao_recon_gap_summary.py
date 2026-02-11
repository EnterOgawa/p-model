#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_recon_gap_summary.py

Phase 4 / Step 4.5（BAOを一次統計から再構築） Stage B（確証決定打）:
Ross 2016 post-recon の ξ0/ξ2（公開データ）と、catalog-based recon（Corrfunc + 簡易recon）
の一致度ギャップを「試した設定の範囲でどこまで改善できたか」として要約する。

目的：
- 既存の overlay metrics（`cosmology_bao_catalog_vs_published_multipoles_overlay*_metrics.json`）を集約し、
  recon 設定の違いが ξ2 ギャップに与える影響を一覧化する。
- “小手先のパラメータ調整で改善しない”状況を可視化し、次工程（公式recon仕様差/選択関数の切り分け）へ繋げる。

出力（固定）:
- output/cosmology/cosmology_bao_recon_gap_summary.png
- output/cosmology/cosmology_bao_recon_gap_summary_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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


def _parse_utc(s: Any) -> Optional[datetime]:
    if s is None:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def _norm_variant_name(s: str) -> str:
    s = str(s).strip()
    if not s:
        return "no_recon"
    while s.startswith("_"):
        s = s[1:]
    return s


def _color_for_rmse(v: float) -> str:
    if not np.isfinite(v):
        return "#cccccc"
    if v <= 20.0:
        return "#2ca02c"  # green
    if v <= 40.0:
        return "#ffcc00"  # yellow
    return "#d62728"  # red


def _as_float_or_nan(v: Any) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Summarize recon gap (RMSE vs Ross post-recon) across existing overlay metrics."
    )
    ap.add_argument(
        "--metrics-dir",
        default=str(_ROOT / "output" / "cosmology"),
        help="directory containing overlay metrics json (default: output/cosmology)",
    )
    ap.add_argument(
        "--out-png",
        default=str(_ROOT / "output" / "cosmology" / "cosmology_bao_recon_gap_summary.png"),
        help="output png path",
    )
    ap.add_argument(
        "--out-json",
        default=str(_ROOT / "output" / "cosmology" / "cosmology_bao_recon_gap_summary_metrics.json"),
        help="output json path",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    metrics_dir = Path(args.metrics_dir)
    out_png = Path(args.out_png)
    out_json = Path(args.out_json)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted(metrics_dir.glob("cosmology_bao_catalog_vs_published_multipoles_overlay*metrics.json"))
    if not paths:
        raise SystemExit(f"no overlay metrics found under: {metrics_dir}")

    by_variant: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        inputs = j.get("inputs", {}) if isinstance(j.get("inputs", {}), dict) else {}
        variant = _norm_variant_name(inputs.get("catalog_recon_suffix", ""))
        recon_estimator = str(inputs.get("catalog_recon_estimator", "stored")).strip() or "stored"
        if recon_estimator != "stored":
            variant = f"{variant}__{recon_estimator}"
        gen = _parse_utc(j.get("generated_utc"))
        prev = by_variant.get(variant)
        prev_gen = _parse_utc(prev.get("generated_utc")) if isinstance(prev, dict) else None
        if (prev is None) or ((gen is not None) and ((prev_gen is None) or (gen > prev_gen))):
            by_variant[variant] = {"path": str(p), **j}

    # Collect arrays: per-variant, per-zbin RMSE(s^2 xi2) vs Ross post-recon for recon output.
    zbins = [1, 2, 3]
    variants = sorted(by_variant.keys())
    rmse = np.full((len(variants), len(zbins)), np.nan, dtype=float)
    rmse0 = np.full((len(variants), len(zbins)), np.nan, dtype=float)
    chi2dof = np.full((len(variants), len(zbins)), np.nan, dtype=float)

    for i, vname in enumerate(variants):
        j = by_variant[vname]
        results = j.get("results", {})
        for k, zbin in enumerate(zbins):
            r = results.get(str(zbin), {}) if isinstance(results, dict) else {}
            if not isinstance(r, dict):
                continue
            rmse[i, k] = _as_float_or_nan(r.get("rmse_s2_xi2_ross_post_recon__catalog_recon"))
            rmse0[i, k] = _as_float_or_nan(r.get("rmse_s2_xi0_ross_post_recon__catalog_recon"))
            chi2dof[i, k] = _as_float_or_nan(r.get("chi2_dof_xi0_xi2_ross_post_recon__catalog_recon"))

    # Sort by median RMSE over available zbins (lower is better).
    score = np.nanmedian(rmse, axis=1)
    order = np.argsort(np.where(np.isfinite(score), score, np.inf))
    variants = [variants[i] for i in order.tolist()]
    rmse = rmse[order]
    rmse0 = rmse0[order]
    chi2dof = chi2dof[order]

    # Plot
    _set_japanese_font()
    import matplotlib.pyplot as plt  # noqa: E402

    fig_h = 2.2 * len(zbins) + 1.0
    fig_w = max(10.0, 0.45 * len(variants))
    fig, axes = plt.subplots(len(zbins), 1, figsize=(fig_w, fig_h), sharex=True)
    if len(zbins) == 1:
        axes = [axes]

    x = np.arange(len(variants))
    for ax, zbin_idx in zip(axes, range(len(zbins))):
        y = rmse[:, zbin_idx]
        colors = [_color_for_rmse(float(v)) for v in y]
        ax.bar(x, np.nan_to_num(y, nan=0.0), color=colors, edgecolor="#333333", linewidth=0.5)
        ax.set_ylabel(f"zbin{zbins[zbin_idx]}\nRMSE(s²ξ₂)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(bottom=0.0)
        # Annotate missing values
        for xi, yi in zip(x, y):
            if not np.isfinite(yi):
                ax.text(xi, 0.5, "NA", ha="center", va="bottom", fontsize=7, rotation=90)
        # Baseline guide line (if present)
        if "recon_grid_iso" in variants:
            b = variants.index("recon_grid_iso")
            base = y[b]
            if np.isfinite(base):
                ax.axhline(base, color="#666666", linestyle="--", linewidth=1.0)
                ax.text(
                    0.99,
                    base,
                    f"baseline iso={base:.1f}",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color="#666666",
                )

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(variants, rotation=60, ha="right", fontsize=8)
    fig.suptitle("BOSS DR12: recon設定スキャン要約（Ross post-recon との ξ₂ ギャップ）", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    metrics_out: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "4.5B (BAO recon gap summary from overlay metrics)",
        "inputs": {"overlay_metrics_files": [str(p) for p in paths]},
        "results": {
            v: {
                "source_metrics_json": by_variant[v].get("path"),
                "rmse_s2_xi2_ross_post_recon__catalog_recon": {str(z): float(rmse[i, j]) for j, z in enumerate(zbins)},
                "rmse_s2_xi0_ross_post_recon__catalog_recon": {str(z): float(rmse0[i, j]) for j, z in enumerate(zbins)},
                "chi2_dof_xi0_xi2_ross_post_recon__catalog_recon": {str(z): float(chi2dof[i, j]) for j, z in enumerate(zbins)},
            }
            for i, v in enumerate(variants)
        },
        "outputs": {"png": str(out_png), "metrics_json": str(out_json)},
    }
    out_json.write_text(json.dumps(metrics_out, ensure_ascii=False, indent=2), encoding="utf-8")

    worklog.append_event(
        {
            "domain": "cosmology",
            "step": "4.5B (bao recon gap summary)",
            "inputs": [Path(p) for p in paths],
            "outputs": [out_png, out_json],
            "notes": {
                "variant_count": int(len(variants)),
                "baseline": "recon_grid_iso" if ("recon_grid_iso" in variants) else None,
            },
        }
    )

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
