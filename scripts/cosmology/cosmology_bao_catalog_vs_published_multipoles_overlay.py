#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_catalog_vs_published_multipoles_overlay.py

Step 16.4（BAO一次情報：銀河+random）/ Phase A6（差分要因の切り分け）:
公開済み multipoles（Ross+2016 post-recon; ξ0/ξ2）と、catalog-based（Corrfunc: galaxy+random）
で再計算した ξ0/ξ2 を、曲線（s^2 xi_ell）として直接重ねて比較する。

狙い：
- ε（AP warping）のクロスチェックだけだと、正規化/振幅/形状の差が見えにくい。
  そこで、s^2 ξ0/ξ2 の曲線そのものを重ね、パイプラインの整合（Phase A）を確認する。

出力（固定）:
- output/private/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay.png
- output/private/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay_metrics.json

Notes:
- catalog-based 出力は `cosmology_bao_xi_from_catalogs.py` の out_tag（例: `__om0p274`）で分岐する。
  このスクリプトは `--catalog-suffix` / `--catalog-recon-suffix` で、比較対象のタグ付きファイルを選べる。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

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


def _load_ross_dat(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s: list[float] = []
    y: list[float] = []
    e: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        parts = t.split()
        if len(parts) < 3:
            continue
        s.append(float(parts[0]))
        y.append(float(parts[1]))
        e.append(float(parts[2]))
    return np.asarray(s, dtype=float), np.asarray(y, dtype=float), np.asarray(e, dtype=float)


def _shell_mean_r_from_bin_center(r: np.ndarray, *, bin_size: float) -> np.ndarray:
    """
    Shell-mean radius used in Ross' baofit_pub2D.py as a small bin-centering correction.

    rbc = 0.75 * ((r+bs/2)^4 - (r-bs/2)^4) / ((r+bs/2)^3 - (r-bs/2)^3)
    """
    r = np.asarray(r, dtype=float)
    bs = float(bin_size)
    rp = r + 0.5 * bs
    rm = r - 0.5 * bs
    num = 0.75 * (rp**4 - rm**4)
    den = (rp**3 - rm**3)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(np.isfinite(out), out, r)


def _load_ross_covariance(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        rows.append([float(x) for x in t.split()])
    if not rows:
        raise ValueError(f"empty covariance file: {path}")
    mat = np.asarray(rows, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"invalid covariance shape: {mat.shape} ({path})")
    return mat


def _chi2_from_cov(residual: np.ndarray, cov: np.ndarray) -> float:
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    c = np.asarray(cov, dtype=np.float64)
    if c.shape[0] != c.shape[1] or c.shape[0] != r.size:
        raise ValueError(f"chi2 dim mismatch: residual={r.size} cov={c.shape}")
    # Numerical guard: tiny diagonal jitter.
    diag = np.diag(c)
    jitter = float(np.nanmax(diag)) * 1e-12 if diag.size else 0.0
    if not np.isfinite(jitter) or jitter <= 0.0:
        jitter = 1e-12
    try:
        x = np.linalg.solve(c + jitter * np.eye(c.shape[0], dtype=np.float64), r)
        return float(r @ x)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(c)
        return float(r @ inv @ r)


def _load_two_col_dat(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    s: list[float] = []
    y: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        parts = t.split()
        if len(parts) < 2:
            continue
        s.append(float(parts[0]))
        y.append(float(parts[1]))
    return np.asarray(s, dtype=float), np.asarray(y, dtype=float)


@dataclass(frozen=True)
class Series:
    s: np.ndarray
    s2_xi: np.ndarray
    err_s2_xi: np.ndarray | None


def _load_catalog_npz(path: Path, *, ell: int) -> Series:
    with np.load(path) as z:
        s = np.asarray(z["s"], dtype=float)
        if ell == 0:
            xi = np.asarray(z["xi0"], dtype=float)
        elif ell == 2:
            xi = np.asarray(z["xi2"], dtype=float)
        else:
            raise ValueError("ell must be 0 or 2")
    s2_xi = (s * s) * xi
    return Series(s=s, s2_xi=s2_xi, err_s2_xi=None)


def _load_recon_multipole_from_counts(
    *,
    path: Path,
    ell: int,
    recon_estimator: str,
) -> Series:
    """
    Recompute recon multipoles from stored Corrfunc pair-count grids.

    This is a diagnostic helper for the recon gap investigation:
    - `cosmology_bao_xi_from_catalogs.py` currently stores xi0/xi2 derived from:
        xi = (DDn - 2DSn + SSn) / RR0n   (RR0: unshifted randoms)
    - Some published pipelines use slightly different normalization choices.

    Here we allow a controlled alternative:
    - delta_rr0: (DDn - 2DSn + SSn) / RR0n   (expected to match stored)
    - delta_ss : (DDn - 2DSn + SSn) / SSn    (diagnostic only; changes the denominator)
    - counts_rr0: xi_ell = (DD_ell - 2DS_ell + SS_ell) / RR0_0   (ratio of multipoles; window/normalization diagnostic)
    - counts_ss : xi_ell = (DD_ell - 2DS_ell + SS_ell) / SS_0    (ratio of multipoles; diagnostic only)

    NOTE: This does NOT change the reconstruction algorithm itself—only the estimator normalization
    applied after reconstruction.
    """
    recon_estimator = str(recon_estimator).strip()
    if recon_estimator == "stored":
        return _load_catalog_npz(path, ell=ell)
    if recon_estimator not in ("delta_rr0", "delta_ss", "counts_rr0", "counts_ss"):
        raise ValueError(f"invalid recon_estimator: {recon_estimator}")

    with np.load(path) as z:
        s = np.asarray(z["s"], dtype=float)
        dd_w = np.asarray(z["dd_w"], dtype=float)
        ds_w = np.asarray(z["dr_w"], dtype=float)
        rr0_w = np.asarray(z["rr_w"], dtype=float)
        ss_w = np.asarray(z["ss_w"], dtype=float) if ("ss_w" in z) else None
        dd_tot = float(z["dd_tot"])
        ds_tot = float(z["dr_tot"])
        rr0_tot = float(z["rr_tot"])
        ss_tot = float(z["ss_tot"])
        mu_edges = np.asarray(z["mu_edges"], dtype=float) if ("mu_edges" in z) else None

    if ss_w is None:
        raise ValueError(f"recon estimator requested but ss_w missing in npz: {path}")
    if mu_edges is None or mu_edges.size < 2:
        raise ValueError(f"mu_edges missing/invalid in npz: {path}")

    nmu = int(mu_edges.size - 1)
    nb = int(s.size)
    if dd_w.shape != (nb, nmu) or ds_w.shape != (nb, nmu) or rr0_w.shape != (nb, nmu) or ss_w.shape != (nb, nmu):
        raise ValueError(
            f"pair-count grid shape mismatch in {path}: "
            f"dd={dd_w.shape} ds={ds_w.shape} rr0={rr0_w.shape} ss={ss_w.shape} "
            f"(expected {(nb, nmu)})"
        )

    mu_mid = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    dmu = float(mu_edges[1] - mu_edges[0])

    ddn = dd_w / dd_tot
    dsn = ds_w / ds_tot
    rr0n = rr0_w / rr0_tot
    ssn = ss_w / ss_tot

    if recon_estimator in ("delta_rr0", "delta_ss"):
        num = ddn - 2.0 * dsn + ssn
        denom = rr0n if (recon_estimator == "delta_rr0") else ssn
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = num / denom
        xi = np.where(np.isfinite(xi), xi, 0.0)
        if ell == 0:
            xi_l = np.sum(xi * dmu, axis=1)
        elif ell == 2:
            p2 = 0.5 * (3.0 * mu_mid * mu_mid - 1.0)
            xi_l = 5.0 * np.sum(xi * p2[None, :] * dmu, axis=1)
        else:
            raise ValueError("ell must be 0 or 2")
    else:
        # Ratio of multipoles:
        #   xi_l = (DD_l - 2 DS_l + SS_l) / denom_0
        # where denom_0 is either RR0 monopole or SS monopole.
        if ell == 0:
            pl = np.ones_like(mu_mid, dtype=np.float64)
            norm = 1.0
        elif ell == 2:
            pl = 0.5 * (3.0 * mu_mid * mu_mid - 1.0)
            norm = 5.0
        else:
            raise ValueError("ell must be 0 or 2")

        dd_l = float(norm) * np.sum(ddn * pl[None, :] * dmu, axis=1)
        ds_l = float(norm) * np.sum(dsn * pl[None, :] * dmu, axis=1)
        ss_l = float(norm) * np.sum(ssn * pl[None, :] * dmu, axis=1)
        num_l = dd_l - 2.0 * ds_l + ss_l

        denom0 = np.sum(rr0n * dmu, axis=1) if (recon_estimator == "counts_rr0") else np.sum(ssn * dmu, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi_l = num_l / denom0
        xi_l = np.where(np.isfinite(xi_l), xi_l, 0.0)

    s2_xi = (s * s) * np.asarray(xi_l, dtype=float)
    return Series(s=s, s2_xi=s2_xi, err_s2_xi=None)


def _maybe_recompute_recon_series_with_rr0_denominator(
    *, recon_npz: Path, rr0_npz: Path, ell: int
) -> Series | None:
    """
    Deprecated reconstruction debug helper.

    Older versions of `cosmology_bao_xi_from_catalogs.py` used a naive Landy–Szalay
    estimator on shifted randoms. The current implementation uses the standard
    reconstruction estimator for δ_rec = δ_D - δ_S and already normalizes by RR0
    (unshifted randoms), so this diagnostic is no longer meaningful.

    Keep the function to avoid breaking older notebooks/scripts, but always return None.
    """
    _ = recon_npz
    _ = rr0_npz
    _ = ell
    return None


def _maybe_load_catalog_npz(path: Path, *, ell: int) -> Series | None:
    if not path.exists():
        return None
    return _load_catalog_npz(path, ell=ell)


def _clip_range(series: Series, *, s_min: float, s_max: float) -> Series:
    m = (series.s >= float(s_min)) & (series.s <= float(s_max))
    s = series.s[m]
    y = series.s2_xi[m]
    e = series.err_s2_xi[m] if series.err_s2_xi is not None else None
    return Series(s=s, s2_xi=y, err_s2_xi=e)


def _clip_range_by_center_then_project_x(
    *,
    s_center: np.ndarray,
    s_x: np.ndarray,
    s2_xi: np.ndarray,
    err_s2_xi: np.ndarray | None,
    s_min: float,
    s_max: float,
) -> Series:
    """
    Clip by Ross' published bin centers, then project the x-axis.

    Ross' public files are tabulated at bin centers. For a more faithful overlay with Ross' own
    baofit_pub2D.py, we sometimes want to show the x-axis at the shell-mean radius (rbc), while
    keeping the selection (s_min/s_max) based on the original bin centers.
    """
    s_center = np.asarray(s_center, dtype=float)
    s_x = np.asarray(s_x, dtype=float)
    s2_xi = np.asarray(s2_xi, dtype=float)
    if err_s2_xi is not None:
        err_s2_xi = np.asarray(err_s2_xi, dtype=float)
    m = (s_center >= float(s_min)) & (s_center <= float(s_max))
    return Series(s=s_x[m], s2_xi=s2_xi[m], err_s2_xi=(err_s2_xi[m] if err_s2_xi is not None else None))


def _align_on_s(pub: Series, cat: Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Expect matching s bins (5 Mpc/h), but be robust via interpolation.
    s = np.asarray(pub.s, dtype=float)
    y_pub = np.asarray(pub.s2_xi, dtype=float)
    e_pub = np.asarray(pub.err_s2_xi, dtype=float) if pub.err_s2_xi is not None else np.full_like(y_pub, np.nan)
    y_cat = np.interp(s, np.asarray(cat.s, dtype=float), np.asarray(cat.s2_xi, dtype=float))
    return s, y_pub, e_pub, y_cat


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Overlay published multipoles (Ross 2016 post-recon / Satpathy 2016 pre-recon) "
            "vs catalog-based multipoles (Corrfunc: galaxy+random), as s^2 xi0/xi2 curves."
        )
    )
    ap.add_argument("--ross-dir", default="data/cosmology/ross_2016_combineddr12_corrfunc", help="Ross 2016 data dir")
    ap.add_argument(
        "--satpathy-dir",
        default="data/cosmology/satpathy_2016_combineddr12_fs_corrfunc_multipoles",
        help="Satpathy 2016 pre-recon multipoles dir",
    )
    ap.add_argument("--bincent", type=int, default=0, help="Ross bin center shift index (0..4; default: 0)")
    ap.add_argument(
        "--ross-x-axis",
        choices=["center", "rbc"],
        default="center",
        help=(
            "x-axis for Ross published points: 'center' uses the tabulated bin centers; "
            "'rbc' uses the shell-mean radius correction from Ross' baofit_pub2D.py. "
            "Selection (s_min/s_max) always uses bin centers. (default: center)"
        ),
    )
    ap.add_argument(
        "--ross-bin-size",
        type=float,
        default=5.0,
        help="Ross bin size (Mpc/h) used by the rbc correction (default: 5)",
    )
    ap.add_argument(
        "--catalog-prefix",
        default="output/private/cosmology/cosmology_bao_xi_from_catalogs_cmasslowztot_combined_lcdm_",
        help="catalog npz prefix (default: cmasslowztot/combined/lcdm)",
    )
    ap.add_argument(
        "--catalog-suffix",
        default="",
        help=(
            "optional suffix for base catalog npz files (default: empty). "
            "Example: '__om0p274' to compare against tagged outputs like ..._b1__om0p274.npz"
        ),
    )
    ap.add_argument(
        "--catalog-recon-suffix",
        default="__recon_grid_iso",
        help="optional recon suffix for catalog npz (default: __recon_grid_iso; expects {prefix}b{1,2,3}{suffix}.npz when present)",
    )
    ap.add_argument(
        "--catalog-recon-estimator",
        choices=["stored", "delta_rr0", "delta_ss", "counts_rr0", "counts_ss"],
        default="stored",
        help=(
            "how to interpret recon outputs when computing xi_ell for overlay/metrics (default: stored). "
            "stored uses xi0/xi2 saved in the recon npz (current default corresponds to delta_rr0). "
            "delta_ss recomputes using SS as denominator (diagnostic normalization test). "
            "counts_rr0 uses a ratio-of-multipoles estimator (DD_ell-2DS_ell+SS_ell)/RR0_0 to probe window/normalization differences."
        ),
    )
    ap.add_argument("--s-min", type=float, default=30.0)
    ap.add_argument("--s-max", type=float, default=150.0)
    ap.add_argument(
        "--out-png",
        default="output/private/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay.png",
        help="output png",
    )
    ap.add_argument(
        "--out-json",
        default="output/private/cosmology/cosmology_bao_catalog_vs_published_multipoles_overlay_metrics.json",
        help="output metrics json",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    ross_dir = (_ROOT / str(args.ross_dir)).resolve()
    satpathy_dir = (_ROOT / str(args.satpathy_dir)).resolve()
    bincent = int(args.bincent)
    ross_x_axis = str(args.ross_x_axis)
    ross_bin_size = float(args.ross_bin_size)
    if bincent < 0 or bincent > 4:
        raise SystemExit("--bincent must be in 0..4")

    cat_prefix = str(args.catalog_prefix)
    cat_suffix = str(args.catalog_suffix)
    recon_suffix = str(args.catalog_recon_suffix)
    recon_estimator = str(args.catalog_recon_estimator)
    s_min = float(args.s_min)
    s_max = float(args.s_max)
    out_png = (_ROOT / str(args.out_png)).resolve()
    out_json = (_ROOT / str(args.out_json)).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    z_bins = [1, 2, 3]
    _set_japanese_font()
    fig, axes = plt.subplots(len(z_bins), 2, figsize=(12, 9), sharex=True, constrained_layout=True)

    metrics: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "16.4 (BAO catalog vs published multipoles overlay)",
        "inputs": {
            "ross_dir": str(ross_dir),
            "satpathy_dir": str(satpathy_dir),
            "bincent": bincent,
            "ross_x_axis": ross_x_axis,
            "ross_bin_size_mpc_h": ross_bin_size,
            "catalog_prefix": cat_prefix,
            "catalog_suffix": cat_suffix,
            "catalog_recon_suffix": recon_suffix,
            "catalog_recon_estimator": recon_estimator,
            "s_range_mpc_h": [s_min, s_max],
        },
        "results": {},
        "outputs": {"png": str(out_png), "metrics_json": str(out_json)},
    }

    for i, zb in enumerate(z_bins):
        # Published (Ross): files store (s, xi, err_xi). Convert to s^2 xi for plotting.
        p_mono = ross_dir / f"Ross_2016_COMBINEDDR12_zbin{zb}_correlation_function_monopole_post_recon_bincent{bincent}.dat"
        p_quad = ross_dir / f"Ross_2016_COMBINEDDR12_zbin{zb}_correlation_function_quadrupole_post_recon_bincent{bincent}.dat"
        p_cov = ross_dir / f"Ross_2016_COMBINEDDR12_zbin{zb}_covariance_monoquad_post_recon_bincent{bincent}.dat"
        if not (p_mono.exists() and p_quad.exists()):
            raise SystemExit(f"missing Ross files for zbin{zb} bincent{bincent}: {ross_dir}")

        s0, y0, e0 = _load_ross_dat(p_mono)
        s2, y2, e2 = _load_ross_dat(p_quad)
        # Keep the range selection on the published bin centers, but optionally show the x-axis
        # using Ross' shell-mean radius correction (rbc) for easier cross-referencing with baofit_pub2D.py.
        s0_x = _shell_mean_r_from_bin_center(s0, bin_size=ross_bin_size) if (ross_x_axis == "rbc") else s0
        s2_x = _shell_mean_r_from_bin_center(s2, bin_size=ross_bin_size) if (ross_x_axis == "rbc") else s2
        pub0 = _clip_range_by_center_then_project_x(
            s_center=s0,
            s_x=s0_x,
            s2_xi=(s0 * s0) * y0,
            err_s2_xi=(s0 * s0) * e0,
            s_min=s_min,
            s_max=s_max,
        )
        pub2 = _clip_range_by_center_then_project_x(
            s_center=s2,
            s_x=s2_x,
            s2_xi=(s2 * s2) * y2,
            err_s2_xi=(s2 * s2) * e2,
            s_min=s_min,
            s_max=s_max,
        )

        # Published (Satpathy): files store (s, xi) for pre-recon multipoles. Convert to s^2 xi.
        s_mono, xi_mono = _load_two_col_dat(
            satpathy_dir / f"Satpathy_2016_COMBINEDDR12_Bin{zb}_Monopole_pre_recon.dat"
        )
        s_quad, xi_quad = _load_two_col_dat(
            satpathy_dir / f"Satpathy_2016_COMBINEDDR12_Bin{zb}_Quadrupole_pre_recon.dat"
        )
        sat0 = _clip_range(Series(s=s_mono, s2_xi=(s_mono * s_mono) * xi_mono, err_s2_xi=None), s_min=s_min, s_max=s_max)
        sat2 = _clip_range(Series(s=s_quad, s2_xi=(s_quad * s_quad) * xi_quad, err_s2_xi=None), s_min=s_min, s_max=s_max)

        # Catalog-based (Corrfunc): NPZ stores xi; convert to s^2 xi.
        ztag = f"b{zb}"
        cat_npz = (_ROOT / f"{cat_prefix}{ztag}{cat_suffix}.npz").resolve()
        if not cat_npz.exists():
            raise SystemExit(f"missing catalog npz: {cat_npz}")
        cat0 = _clip_range(_load_catalog_npz(cat_npz, ell=0), s_min=s_min, s_max=s_max)
        cat2 = _clip_range(_load_catalog_npz(cat_npz, ell=2), s_min=s_min, s_max=s_max)

        cat_npz_recon = (_ROOT / f"{cat_prefix}{ztag}{recon_suffix}.npz").resolve()
        cat0_rec = _load_recon_multipole_from_counts(path=cat_npz_recon, ell=0, recon_estimator=recon_estimator) if cat_npz_recon.exists() else None
        cat2_rec = _load_recon_multipole_from_counts(path=cat_npz_recon, ell=2, recon_estimator=recon_estimator) if cat_npz_recon.exists() else None
        cat0_rec = _clip_range(cat0_rec, s_min=s_min, s_max=s_max) if cat0_rec is not None else None
        cat2_rec = _clip_range(cat2_rec, s_min=s_min, s_max=s_max) if cat2_rec is not None else None

        # Align and compute metrics
        s_al, y_pub0, e_pub0, y_cat0 = _align_on_s(pub0, cat0)
        _, y_pub2, e_pub2, y_cat2 = _align_on_s(pub2, cat2)
        rmse0 = _rmse(y_pub0, y_cat0)
        rmse2 = _rmse(y_pub2, y_cat2)

        rmse0_rec = None
        rmse2_rec = None
        if cat0_rec is not None and cat2_rec is not None:
            _, _, _, y_cat0_rec = _align_on_s(pub0, cat0_rec)
            _, _, _, y_cat2_rec = _align_on_s(pub2, cat2_rec)
            rmse0_rec = _rmse(y_pub0, y_cat0_rec)
            rmse2_rec = _rmse(y_pub2, y_cat2_rec)

        # Optional chi2/dof using Ross covariance (xi, not s^2 xi).
        chi2_dof_pre = None
        chi2_dof_rec = None
        try:
            if p_cov.exists():
                cov_full = _load_ross_covariance(p_cov)
                n_full = int(np.asarray(s0, dtype=float).size)
                if cov_full.shape[0] == 2 * n_full:
                    m0 = (np.asarray(s0, dtype=float) >= s_min) & (np.asarray(s0, dtype=float) <= s_max)
                    m2 = (np.asarray(s2, dtype=float) >= s_min) & (np.asarray(s2, dtype=float) <= s_max)
                    idx0 = np.nonzero(m0)[0].astype(int, copy=False)
                    idx2 = np.nonzero(m2)[0].astype(int, copy=False)
                    if (idx0.size > 0) and (idx2.size > 0):
                        idx_all = np.concatenate([idx0, idx2 + n_full], axis=0)
                        cov = cov_full[np.ix_(idx_all, idx_all)]

                        # Load catalog xi (not s^2 xi) for chi2.
                        with np.load(cat_npz) as z:
                            s_cat = np.asarray(z["s"], dtype=float)
                            xi_cat0 = np.asarray(z["xi0"], dtype=float)
                            xi_cat2 = np.asarray(z["xi2"], dtype=float)
                        r0 = np.asarray(y0, dtype=float)[m0] - np.interp(np.asarray(s0, dtype=float)[m0], s_cat, xi_cat0)
                        r2 = np.asarray(y2, dtype=float)[m2] - np.interp(np.asarray(s2, dtype=float)[m2], s_cat, xi_cat2)
                        res = np.concatenate([r0, r2], axis=0)
                        chi2 = _chi2_from_cov(res, cov)
                        chi2_dof_pre = float(chi2 / max(1, int(res.size)))

                        if cat_npz_recon.exists():
                            with np.load(cat_npz_recon) as z:
                                s_cat_r = np.asarray(z["s"], dtype=float)
                                xi_cat0_r = np.asarray(z["xi0"], dtype=float)
                                xi_cat2_r = np.asarray(z["xi2"], dtype=float)
                            r0r = np.asarray(y0, dtype=float)[m0] - np.interp(np.asarray(s0, dtype=float)[m0], s_cat_r, xi_cat0_r)
                            r2r = np.asarray(y2, dtype=float)[m2] - np.interp(np.asarray(s2, dtype=float)[m2], s_cat_r, xi_cat2_r)
                            res_r = np.concatenate([r0r, r2r], axis=0)
                            chi2_r = _chi2_from_cov(res_r, cov)
                            chi2_dof_rec = float(chi2_r / max(1, int(res_r.size)))
        except Exception:
            # Best-effort only: keep RMSE as the primary metric.
            chi2_dof_pre = None
            chi2_dof_rec = None

        # Satpathy vs catalog: evaluate on the catalog grid (Phase A pre-recon validation).
        y_sat0_on_cat = np.interp(np.asarray(cat0.s, dtype=float), np.asarray(sat0.s, dtype=float), np.asarray(sat0.s2_xi, dtype=float))
        y_sat2_on_cat = np.interp(np.asarray(cat2.s, dtype=float), np.asarray(sat2.s, dtype=float), np.asarray(sat2.s2_xi, dtype=float))
        rmse0_sat = _rmse(np.asarray(cat0.s2_xi, dtype=float), y_sat0_on_cat)
        rmse2_sat = _rmse(np.asarray(cat2.s2_xi, dtype=float), y_sat2_on_cat)

        metrics["results"][str(zb)] = {
            "zbin": zb,
            "rmse_s2_xi0_ross_post_recon": rmse0,
            "rmse_s2_xi2_ross_post_recon": rmse2,
            "rmse_s2_xi0_ross_post_recon__catalog_recon": rmse0_rec,
            "rmse_s2_xi2_ross_post_recon__catalog_recon": rmse2_rec,
            "chi2_dof_xi0_xi2_ross_post_recon__catalog_pre": chi2_dof_pre,
            "chi2_dof_xi0_xi2_ross_post_recon__catalog_recon": chi2_dof_rec,
            "rmse_s2_xi0_satpathy_pre_recon_on_cat_grid": rmse0_sat,
            "rmse_s2_xi2_satpathy_pre_recon_on_cat_grid": rmse2_sat,
            "published_files": {"mono": str(p_mono), "quad": str(p_quad)},
            "published_covariance": str(p_cov) if p_cov.exists() else None,
            "satpathy_files": {
                "mono": str(satpathy_dir / f"Satpathy_2016_COMBINEDDR12_Bin{zb}_Monopole_pre_recon.dat"),
                "quad": str(satpathy_dir / f"Satpathy_2016_COMBINEDDR12_Bin{zb}_Quadrupole_pre_recon.dat"),
            },
            "catalog_npz": str(cat_npz),
            "catalog_npz_recon": str(cat_npz_recon) if cat_npz_recon.exists() else None,
            "n_points": int(s_al.size),
        }

        ax_m = axes[i, 0]
        ax_q = axes[i, 1]

        ax_m.errorbar(s_al, y_pub0, yerr=e_pub0, fmt="o", markersize=3, color="black", alpha=0.8, label="published (Ross post-recon)")
        ax_m.plot(sat0.s, sat0.s2_xi, "-", color="#9467bd", linewidth=1.2, alpha=0.8, label="published (Satpathy pre-recon)")
        ax_m.plot(cat0.s, cat0.s2_xi, "-", color="#1f77b4", linewidth=1.6, label="catalog (Corrfunc)")
        if cat0_rec is not None:
            label = "catalog (recon grid)"
            if recon_estimator != "stored":
                label = f"catalog (recon: {recon_estimator})"
            ax_m.plot(cat0_rec.s, cat0_rec.s2_xi, "-", color="#ff7f0e", linewidth=1.6, label=label)
        ax_m.set_ylabel(f"zbin{zb}\n$s^2\\,\\xi_0$")
        ax_m.grid(True, alpha=0.3)

        ax_q.errorbar(s_al, y_pub2, yerr=e_pub2, fmt="o", markersize=3, color="black", alpha=0.8, label="published (Ross post-recon)")
        ax_q.plot(sat2.s, sat2.s2_xi, "-", color="#9467bd", linewidth=1.2, alpha=0.8, label="published (Satpathy pre-recon)")
        ax_q.plot(cat2.s, cat2.s2_xi, "-", color="#1f77b4", linewidth=1.6, label="catalog (Corrfunc)")
        if cat2_rec is not None:
            label = "catalog (recon grid)"
            if recon_estimator != "stored":
                label = f"catalog (recon: {recon_estimator})"
            ax_q.plot(cat2_rec.s, cat2_rec.s2_xi, "-", color="#ff7f0e", linewidth=1.6, label=label)
        ax_q.set_ylabel(f"zbin{zb}\n$s^2\\,\\xi_2$")
        ax_q.grid(True, alpha=0.3)

        rmse_lines0 = [f"RMSE (Ross)={rmse0:.3g}", f"RMSE (Sat)={rmse0_sat:.3g}"]
        if rmse0_rec is not None:
            rmse_lines0.insert(1, f"RMSE (Ross, recon)={rmse0_rec:.3g}")
        if chi2_dof_rec is not None:
            rmse_lines0.append(f"chi2/dof (Ross, recon)={chi2_dof_rec:.3g}")
        ax_m.text(
            0.02,
            0.95,
            "\n".join(rmse_lines0),
            transform=ax_m.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, edgecolor="none"),
        )
        rmse_lines2 = [f"RMSE (Ross)={rmse2:.3g}", f"RMSE (Sat)={rmse2_sat:.3g}"]
        if rmse2_rec is not None:
            rmse_lines2.insert(1, f"RMSE (Ross, recon)={rmse2_rec:.3g}")
        ax_q.text(
            0.02,
            0.95,
            "\n".join(rmse_lines2),
            transform=ax_q.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, edgecolor="none"),
        )

    axes[-1, 0].set_xlabel("s [Mpc/h]")
    axes[-1, 1].set_xlabel("s [Mpc/h]")
    axes[0, 0].legend(fontsize=9, loc="upper right")
    axes[0, 1].legend(fontsize=9, loc="upper right")
    fig.suptitle(
        "BOSS DR12 BAO multipoles: published (Ross post-recon / Satpathy pre-recon) vs catalog-based (Corrfunc)",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=160)

    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_bao_catalog_vs_published_multipoles_overlay",
                "generated_utc": metrics["generated_utc"],
                "inputs": metrics["inputs"],
                "outputs": metrics["outputs"],
            }
        )
    except Exception:
        pass

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
