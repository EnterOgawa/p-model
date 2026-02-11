#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_recon_gap_broadband_sensitivity.py

Phase 4 / Step 4.5（BAO一次統計）Stage B（確証決定打）:
Ross 2016 post-recon の ξ2 ギャップについて、
 - bin-centering（bincent=0..4）
 - broadband fit の使用レンジ（例：s_min=30 vs 50）
に対する「broadband吸収（A0+A1/r+A2/r^2）」の結論が頑健かを確認する。

出力（固定）:
- output/private/cosmology/cosmology_bao_recon_gap_broadband_sensitivity.png
- output/private/cosmology/cosmology_bao_recon_gap_broadband_sensitivity_metrics.json

注意：
- Corrfunc は使わない（既存の catalog-based recon NPZ と Ross 公開ファイルのみを使用）。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class FitResult:
    rmse_s2_raw: float
    rmse_s2_after: float
    chi2_dof_raw: float
    chi2_dof_after: float
    n_points: int
    coef_a0_a1_a2: list[float]


def _parse_ross_xi_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (s, xi, err_xi) arrays from Ross file.
    """
    s_list: list[float] = []
    xi_list: list[float] = []
    err_list: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        cols = t.split()
        if len(cols) < 3:
            continue
        s_list.append(float(cols[0]))
        xi_list.append(float(cols[1]))
        err_list.append(float(cols[2]))
    if not s_list:
        raise ValueError(f"no data rows found: {path}")
    s = np.asarray(s_list, dtype=np.float64)
    xi = np.asarray(xi_list, dtype=np.float64)
    err = np.asarray(err_list, dtype=np.float64)
    return s, xi, err


def _fit_broadband_diff(*, s: np.ndarray, diff: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit diff(s) = a0 + a1/s + a2/s^2 with weights 1/sigma^2.
    Return (coef[3], fitted_diff).
    """
    s = np.asarray(s, dtype=np.float64)
    diff = np.asarray(diff, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError("s must be 1D")
    if diff.shape != s.shape or sigma.shape != s.shape:
        raise ValueError("shape mismatch in fit inputs")
    if np.any(~np.isfinite(s)) or np.any(s <= 0.0):
        raise ValueError("invalid s (requires finite and >0)")

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


def _summarize_across_bincent(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Broadband absorption sensitivity vs Ross bincent and s-range.")
    ap.add_argument(
        "--ross-dir",
        default=str(_ROOT / "data" / "cosmology" / "ross_2016_combineddr12_corrfunc"),
        help="Ross 2016 published multipoles directory",
    )
    ap.add_argument("--bincent-min", type=int, default=0, help="min bincent id (default: 0)")
    ap.add_argument("--bincent-max", type=int, default=4, help="max bincent id (default: 4)")
    ap.add_argument("--s-mins", default="30,50", help="comma-separated s_min list (Mpc/h; default: 30,50)")
    ap.add_argument("--s-max", type=float, default=150.0, help="s range max (Mpc/h; default: 150)")
    ap.add_argument("--sample", default="cmasslowztot", help="catalog sample name (default: cmasslowztot)")
    ap.add_argument("--caps", default="combined", help="caps name (default: combined)")
    ap.add_argument("--dist", default="lcdm", choices=["lcdm", "pbg"], help="distance mapping label (default: lcdm)")
    ap.add_argument(
        "--catalog-recon-suffix",
        default="__recon_mw_multigrid_ani_mw_rw_boss_recon_rz_zmin0p2_zmax0p75_g512_bias1p85_f0p757",
        help="suffix of catalog recon NPZ (default: mw_multigrid baseline)",
    )
    ap.add_argument(
        "--out-png",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_bao_recon_gap_broadband_sensitivity.png"),
        help="output png path",
    )
    ap.add_argument(
        "--out-json",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_bao_recon_gap_broadband_sensitivity_metrics.json"),
        help="output metrics json path",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    ross_dir = Path(args.ross_dir)
    out_png = Path(args.out_png)
    out_json = Path(args.out_json)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    bincent_min = int(args.bincent_min)
    bincent_max = int(args.bincent_max)
    if bincent_min < 0 or bincent_max < bincent_min:
        raise SystemExit("--bincent-min/max invalid")
    bincents = list(range(bincent_min, bincent_max + 1))

    try:
        s_mins = [float(x.strip()) for x in str(args.s_mins).split(",") if x.strip()]
    except Exception as e:
        raise SystemExit(f"invalid --s-mins: {e}") from e
    if not s_mins:
        raise SystemExit("--s-mins must not be empty")
    s_max = float(args.s_max)
    if not (s_max > 0.0):
        raise SystemExit("--s-max must be >0")
    for sm in s_mins:
        if not (0.0 < sm < s_max):
            raise SystemExit(f"invalid s range: s_min={sm:g} s_max={s_max:g}")

    zbins = [1, 2, 3]
    zbin_to_b = {1: "b1", 2: "b2", 3: "b3"}

    # Load catalog recon curves (xi2 only) for each zbin once.
    cat_curves: dict[int, dict[str, np.ndarray]] = {}
    used_inputs: list[str] = []
    for zbin in zbins:
        b = zbin_to_b[zbin]
        cat_npz = _ROOT / "output" / "private" / "cosmology" / f"cosmology_bao_xi_from_catalogs_{args.sample}_{args.caps}_{args.dist}_{b}{args.catalog_recon_suffix}.npz"
        if not cat_npz.exists():
            raise SystemExit(f"missing catalog recon npz: {cat_npz}")
        used_inputs.append(str(cat_npz))
        with np.load(cat_npz, allow_pickle=True) as z:
            cat_curves[zbin] = {
                "s": np.asarray(z["s"], dtype=np.float64),
                "xi2": np.asarray(z["xi2"], dtype=np.float64),
            }

    results: dict[str, Any] = {"per": {}, "summary": {}}
    for zbin in zbins:
        results["per"][str(zbin)] = {}
        for s_min in s_mins:
            key = f"s_min={s_min:g}"
            results["per"][str(zbin)][key] = {}
            per_bincent: dict[int, FitResult] = {}
            for binc in bincents:
                ross_quad = ross_dir / (
                    f"Ross_2016_COMBINEDDR12_zbin{zbin}_correlation_function_quadrupole_post_recon_bincent{binc}.dat"
                )
                if not ross_quad.exists():
                    raise SystemExit(f"missing Ross file: {ross_quad}")
                used_inputs.append(str(ross_quad))
                s_r, xi2_r, sig_r = _parse_ross_xi_file(ross_quad)

                m = (s_r >= float(s_min)) & (s_r <= s_max)
                s = s_r[m]
                xi2_ross = xi2_r[m]
                sig2 = sig_r[m]
                if s.size < 8:
                    raise SystemExit(f"too few points after s-range cut: zbin={zbin} bincent={binc} n={s.size}")

                # Interpolate catalog xi2 to Ross bin centers (bincent shifts).
                s_cat = cat_curves[zbin]["s"]
                xi2_cat = np.interp(s, s_cat, cat_curves[zbin]["xi2"])

                diff = xi2_ross - xi2_cat
                coef, fit = _fit_broadband_diff(s=s, diff=diff, sigma=sig2)
                xi2_corr = xi2_cat + fit

                rmse_raw = _rmse((s * s) * (xi2_cat - xi2_ross))
                rmse_corr = _rmse((s * s) * (xi2_corr - xi2_ross))
                chi2_raw = float(np.sum(((xi2_cat - xi2_ross) / sig2) ** 2))
                chi2_after = float(np.sum(((xi2_corr - xi2_ross) / sig2) ** 2))
                dof_raw = int(s.size)
                dof_after = int(max(s.size - 3, 1))

                per_bincent[binc] = FitResult(
                    rmse_s2_raw=float(rmse_raw),
                    rmse_s2_after=float(rmse_corr),
                    chi2_dof_raw=float(chi2_raw / max(dof_raw, 1)),
                    chi2_dof_after=float(chi2_after / float(dof_after)),
                    n_points=int(s.size),
                    coef_a0_a1_a2=[float(x) for x in coef.tolist()],
                )
                results["per"][str(zbin)][key][str(binc)] = asdict(per_bincent[binc])

            # Summaries across bincent (min/median/max)
            results["summary"].setdefault(str(zbin), {})
            results["summary"][str(zbin)][key] = {
                "rmse_s2_raw": _summarize_across_bincent([v.rmse_s2_raw for v in per_bincent.values()]),
                "rmse_s2_after": _summarize_across_bincent([v.rmse_s2_after for v in per_bincent.values()]),
                "chi2_dof_raw": _summarize_across_bincent([v.chi2_dof_raw for v in per_bincent.values()]),
                "chi2_dof_after": _summarize_across_bincent([v.chi2_dof_after for v in per_bincent.values()]),
                "n_points": {str(b): int(per_bincent[b].n_points) for b in bincents},
            }

    # Plot: chi2/dof raw vs after across bincent, for each s_min.
    _set_japanese_font()
    import matplotlib.pyplot as plt  # noqa: E402

    fig, axes = plt.subplots(len(zbins), 1, figsize=(11.5, 8.5), sharex=True)
    if len(zbins) == 1:
        axes = [axes]
    x = np.asarray(bincents, dtype=int)

    for ax, zbin in zip(axes, zbins):
        for i, s_min in enumerate(sorted(s_mins)):
            key = f"s_min={s_min:g}"
            y_raw = np.array([results["per"][str(zbin)][key][str(b)]["chi2_dof_raw"] for b in bincents], dtype=float)
            y_after = np.array([results["per"][str(zbin)][key][str(b)]["chi2_dof_after"] for b in bincents], dtype=float)

            # Stagger markers slightly for readability when multiple s_min exist.
            dx = (i - (len(s_mins) - 1) / 2.0) * 0.05
            ax.plot(x + dx, y_raw, linestyle="--", marker="o", markersize=4, label=f"raw (s≥{s_min:g})" if zbin == 1 else None)
            ax.plot(x + dx, y_after, linestyle="-", marker="s", markersize=4, label=f"broadband (s≥{s_min:g})" if zbin == 1 else None)

        ax.axhline(1.0, color="#999999", linewidth=1.0, linestyle=":")
        ax.set_ylabel(f"χ²/dof（ξ2）\\n(zbin{zbin})")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Ross bincent（0..4）")
    axes[0].set_title("Ross post-recon ξ2 gap: broadband吸収の感度（bincent / s_min）")
    if len(axes) > 0:
        axes[0].legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)

    metrics: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "4.5B.17.4.4 (broadband sensitivity vs bincent/range)",
        "inputs": {
            "ross_dir": str(ross_dir),
            "bincent_range": [bincent_min, bincent_max],
            "s_mins": s_mins,
            "s_max": s_max,
            "sample": str(args.sample),
            "caps": str(args.caps),
            "dist": str(args.dist),
            "catalog_recon_suffix": str(args.catalog_recon_suffix),
            "used_inputs": sorted(set(used_inputs)),
        },
        "results": results,
        "outputs": {"png": str(out_png), "metrics_json": str(out_json)},
    }
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    worklog.append_event(
        {
            "domain": "cosmology",
            "step": "4.5B.17.4.4 (broadband sensitivity vs bincent/range)",
            "inputs": sorted(set(used_inputs)),
            "outputs": [str(out_png), str(out_json)],
            "notes": {
                "sample": str(args.sample),
                "caps": str(args.caps),
                "dist": str(args.dist),
                "catalog_recon_suffix": str(args.catalog_recon_suffix),
                "bincent_range": [bincent_min, bincent_max],
                "s_mins": s_mins,
                "s_max": s_max,
            },
        }
    )
    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
