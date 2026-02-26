#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_recon_gap_broadband_fit.py

Phase 4 / Step 4.5（BAO一次統計） Stage B（確証決定打）:
Ross 2016 post-recon の ξ2 ギャップを「broadband（A0+A1/r+A2/r^2）」でどこまで吸収できるかを定量化する。

背景：
- catalog-based recon（Corrfunc + recon）で ξ0 は概ね一致する一方、ξ2 は RMSE が大きい状態が残る。
- しかし Ross の公開フィット（baofit_pub2D.py）は、ξ2 に対しても broadband を周辺化している。
  そのため「生の ξ2 カーブが一致しない」ことが、必ずしも BAOピーク（AP/ε）の不一致を意味しない。

本スクリプトは、
- Ross post-recon ξ2 と catalog-based recon ξ2 の差分に対し、
  diff(r)=a0+a1/r+a2/r^2 を重み付き最小二乗（Ross err）で当てはめ、
  RMSE(s^2 ξ2) がどの程度縮むかを zbin1..3 で示す。

出力（固定）:
- output/private/cosmology/cosmology_bao_recon_gap_broadband_fit.png
- output/private/cosmology/cosmology_bao_recon_gap_broadband_fit_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
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
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _parse_ross_xi_file(path: Path) -> dict[float, tuple[float, float]]:
    """
    Return mapping: s -> (xi, err_xi)
    """
    out: dict[float, tuple[float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        # 条件分岐: `not t or t.startswith("#")` を満たす経路を評価する。
        if not t or t.startswith("#"):
            continue

        cols = t.split()
        # 条件分岐: `len(cols) < 3` を満たす経路を評価する。
        if len(cols) < 3:
            continue

        s = float(cols[0])
        xi = float(cols[1])
        err = float(cols[2])
        out[s] = (xi, err)

    # 条件分岐: `not out` を満たす経路を評価する。

    if not out:
        raise ValueError(f"no data rows found: {path}")

    return out


def _fit_broadband_diff(
    *,
    s: np.ndarray,
    diff: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit diff(s) = a0 + a1/s + a2/s^2 with weights 1/sigma^2.
    Return (coef[3], fitted_diff).
    """
    s = np.asarray(s, dtype=np.float64)
    diff = np.asarray(diff, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    # 条件分岐: `s.ndim != 1` を満たす経路を評価する。
    if s.ndim != 1:
        raise ValueError("s must be 1D")

    # 条件分岐: `diff.shape != s.shape or sigma.shape != s.shape` を満たす経路を評価する。

    if diff.shape != s.shape or sigma.shape != s.shape:
        raise ValueError("shape mismatch in fit inputs")

    # 条件分岐: `np.any(~np.isfinite(s)) or np.any(s <= 0.0)` を満たす経路を評価する。

    if np.any(~np.isfinite(s)) or np.any(s <= 0.0):
        raise ValueError("invalid s (requires finite and >0)")

    # Guard against zeros; Ross err should be >0 but be defensive.

    sigma_eff = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, np.nanmedian(sigma[sigma > 0.0]))
    w = 1.0 / (sigma_eff * sigma_eff)
    sw = np.sqrt(w)

    X = np.column_stack([np.ones_like(s), 1.0 / s, 1.0 / (s * s)])
    coef, *_ = np.linalg.lstsq(X * sw[:, None], diff * sw, rcond=None)
    fit = X @ coef
    return coef.astype(np.float64), fit.astype(np.float64)


def _rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Quantify recon ξ2 gap absorbable by broadband (a0+a1/r+a2/r^2).")
    ap.add_argument(
        "--ross-dir",
        default=str(_ROOT / "data" / "cosmology" / "ross_2016_combineddr12_corrfunc"),
        help="Ross 2016 published multipoles directory (default: data/cosmology/ross_2016_combineddr12_corrfunc)",
    )
    ap.add_argument("--bincent", type=int, default=0, help="Ross bincent id (default: 0)")
    ap.add_argument(
        "--sample",
        default="cmasslowztot",
        help="catalog-based sample name used in output NPZ prefix (default: cmasslowztot)",
    )
    ap.add_argument("--caps", default="combined", help="caps used in output NPZ prefix (default: combined)")
    ap.add_argument("--dist", default="lcdm", choices=["lcdm", "pbg"], help="distance mapping label (default: lcdm)")
    ap.add_argument(
        "--catalog-recon-suffix",
        default="__recon_mw_multigrid_ani_mw_rw_boss_recon_rz_zmin0p2_zmax0p75_g512_bias1p85_f0p757",
        help="suffix of catalog recon NPZ (default: mw_multigrid baseline)",
    )
    ap.add_argument("--s-min", type=float, default=30.0, help="s range min (Mpc/h; default: 30)")
    ap.add_argument("--s-max", type=float, default=150.0, help="s range max (Mpc/h; default: 150)")
    ap.add_argument(
        "--out-png",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_bao_recon_gap_broadband_fit.png"),
        help="output png path",
    )
    ap.add_argument(
        "--out-json",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_bao_recon_gap_broadband_fit_metrics.json"),
        help="output metrics json path",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    ross_dir = Path(args.ross_dir)
    out_png = Path(args.out_png)
    out_json = Path(args.out_json)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    s_min = float(args.s_min)
    s_max = float(args.s_max)
    # 条件分岐: `not (0.0 < s_min < s_max)` を満たす経路を評価する。
    if not (0.0 < s_min < s_max):
        raise SystemExit("--s-min must be >0 and < --s-max")

    zbins = [1, 2, 3]
    zbin_to_b = {1: "b1", 2: "b2", 3: "b3"}

    _set_japanese_font()
    import matplotlib.pyplot as plt  # noqa: E402

    fig, axes = plt.subplots(len(zbins), 1, figsize=(11.5, 8.5), sharex=True)
    # 条件分岐: `len(zbins) == 1` を満たす経路を評価する。
    if len(zbins) == 1:
        axes = [axes]

    results: dict[str, Any] = {}
    used_inputs: list[Path] = []
    used_outputs: list[Path] = [out_png, out_json]

    for ax, zbin in zip(axes, zbins):
        b = zbin_to_b[zbin]
        cat_npz = _ROOT / "output" / "private" / "cosmology" / f"cosmology_bao_xi_from_catalogs_{args.sample}_{args.caps}_{args.dist}_{b}{args.catalog_recon_suffix}.npz"
        # 条件分岐: `not cat_npz.exists()` を満たす経路を評価する。
        if not cat_npz.exists():
            raise SystemExit(f"missing catalog recon npz: {cat_npz}")

        used_inputs.append(cat_npz)

        with np.load(cat_npz, allow_pickle=True) as z:
            s = np.asarray(z["s"], dtype=np.float64)
            xi2_cat = np.asarray(z["xi2"], dtype=np.float64)

        m = (s >= s_min) & (s <= s_max)
        s = s[m]
        xi2_cat = xi2_cat[m]

        ross_quad = ross_dir / f"Ross_2016_COMBINEDDR12_zbin{zbin}_correlation_function_quadrupole_post_recon_bincent{int(args.bincent)}.dat"
        # 条件分岐: `not ross_quad.exists()` を満たす経路を評価する。
        if not ross_quad.exists():
            raise SystemExit(f"missing Ross file: {ross_quad}")

        used_inputs.append(ross_quad)

        ross_map = _parse_ross_xi_file(ross_quad)
        xi2_ross = np.empty_like(s)
        sig2 = np.empty_like(s)
        for i, sv in enumerate(s.tolist()):
            # 条件分岐: `sv not in ross_map` を満たす経路を評価する。
            if sv not in ross_map:
                # Ross is on 5 Mpc/h bins; this should match exactly. Be strict to avoid silent shifts.
                raise SystemExit(f"Ross quadrupole missing s={sv:g} (expected exact match)")

            xi2_ross[i] = float(ross_map[sv][0])
            sig2[i] = float(ross_map[sv][1])

        diff = xi2_ross - xi2_cat
        coef, fit = _fit_broadband_diff(s=s, diff=diff, sigma=sig2)
        xi2_corr = xi2_cat + fit

        rmse_raw = _rmse((s * s) * (xi2_cat - xi2_ross))
        rmse_corr = _rmse((s * s) * (xi2_corr - xi2_ross))
        # χ² is computed on ξ2 (not s²ξ2) using Ross σ(ξ2).
        chi2_raw = float(np.sum(((xi2_cat - xi2_ross) / sig2) ** 2))
        dof_raw = int(s.size)
        chi2_dof_raw = float(chi2_raw / max(dof_raw, 1))
        chi2_after = float(np.sum(((xi2_corr - xi2_ross) / sig2) ** 2))
        dof_after = int(s.size - 3)  # a0,a1,a2
        chi2_dof_after = float(chi2_after / max(dof_after, 1))

        # Plot in s^2 * xi2 for readability.
        y_ross = (s * s) * xi2_ross
        y_cat = (s * s) * xi2_cat
        y_corr = (s * s) * xi2_corr
        yerr_ross = (s * s) * sig2

        ax.errorbar(s, y_ross, yerr=yerr_ross, fmt="o", color="black", ecolor="black", elinewidth=1.0, capsize=2, markersize=3, label="published (Ross)")
        ax.plot(s, y_cat, color="#1f77b4", linewidth=2.0, label="catalog recon (raw)")
        ax.plot(s, y_corr, color="#2ca02c", linewidth=2.0, linestyle="--", label="catalog recon + broadband fit")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(f"zbin{zbin}\ns² ξ₂")
        ax.text(
            0.01,
            0.97,
            f"RMSE raw={rmse_raw:.1f} → broadband={rmse_corr:.1f}\n"
            f"χ²/dof raw={chi2_dof_raw:.2f} → broadband={chi2_dof_after:.2f}\n"
            f"diff(r)=a0+a1/r+a2/r²\n"
            f"a0={coef[0]:.3g}, a1={coef[1]:.3g}, a2={coef[2]:.3g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

        results[str(zbin)] = {
            "zbin": int(zbin),
            "s_range_mpc_h": [float(s_min), float(s_max)],
            "rmse_s2_xi2_raw": float(rmse_raw),
            "rmse_s2_xi2_after_broadband_fit": float(rmse_corr),
            "chi2_dof_xi2_raw": float(chi2_dof_raw),
            "chi2_dof_xi2_after_broadband_fit": float(chi2_dof_after),
            "broadband_coef": {"a0": float(coef[0]), "a1": float(coef[1]), "a2": float(coef[2])},
            "inputs": {"catalog_npz_recon": str(cat_npz), "ross_quad": str(ross_quad)},
            "n_points": int(s.size),
        }

    axes[-1].set_xlabel("s [Mpc/h]")
    # One shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("BOSS DR12 post-recon ξ₂ gap: broadband absorption test (Ross vs catalog-based recon)", y=0.98)
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.94])

    fig.savefig(out_png, dpi=160)

    metrics: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "4.5B.17.4 (recon gap broadband fit)",
        "inputs": {
            "ross_dir": str(ross_dir),
            "bincent": int(args.bincent),
            "catalog": {
                "sample": str(args.sample),
                "caps": str(args.caps),
                "dist": str(args.dist),
                "catalog_recon_suffix": str(args.catalog_recon_suffix),
            },
            "s_range_mpc_h": [float(s_min), float(s_max)],
        },
        "results": results,
        "outputs": {"png": str(out_png), "metrics_json": str(out_json)},
    }
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    worklog.append_event(
        {
            "domain": "cosmology",
            "step": "4.5B.17.4 (recon gap broadband fit)",
            "inputs": used_inputs,
            "outputs": used_outputs,
            "notes": {
                "catalog_recon_suffix": str(args.catalog_recon_suffix),
                "bincent": int(args.bincent),
            },
        }
    )

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
