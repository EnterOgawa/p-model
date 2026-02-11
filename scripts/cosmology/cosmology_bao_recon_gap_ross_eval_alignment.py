#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_recon_gap_ross_eval_alignment.py

Phase 4 / Step 4.5（BAO一次統計）Stage B（確証決定打）:
recon gap（Ross post-recon multipoles vs catalog-based recon multipoles）の「残差評価」を、
Ross の一次コード（.tmp_lssanalysis/baofit_pub2D.py）基準へ揃える。

目的（Roadmap: 4.5B.17.4.4.2）:
- fitレンジ：Ross 既定の r∈(50,150) を採用
- 評価軸：Ross の mono+quad の covariance（mono→quad の順）で χ² を評価
- 基底：Ross が broadband として使う {1, 1/r, 1/r^2} を mono/quad それぞれに付与（合計6係数）
- ξ4（モデル側）：Ross の公開fitは AP変換後の xi(mu) に ξ4 テンプレートを含む点を、
  “残差評価”と“AP/ε議論”で混線しないように補助図で定量化する。

出力（固定）:
- output/private/cosmology/cosmology_bao_recon_gap_ross_eval_alignment.png
- output/private/cosmology/cosmology_bao_recon_gap_ross_eval_alignment_metrics.json

注意：
- Corrfunc は使わない（既存の catalog-based recon NPZ と Ross 公開ファイルのみ）。
"""

from __future__ import annotations

import argparse
import json
import sys
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


def _parse_ross_xi_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (s, xi, err_xi) arrays from Ross file.
    """
    s_list: list[float] = []
    xi_list: list[float] = []
    err_list: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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
    return np.asarray(s_list, dtype=float), np.asarray(xi_list, dtype=float), np.asarray(err_list, dtype=float)


def _read_cov(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        rows.append([float(x) for x in t.split()])
    cov = np.asarray(rows, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"invalid covariance shape: {cov.shape} from {path}")
    cov = 0.5 * (cov + cov.T)
    return cov


def _rbc(r: np.ndarray, *, bs: float) -> np.ndarray:
    """
    Ross baofit_pub2D.py の bin-centering 補正（殻内平均距離）。
    r: bin center（Mpc/h）
    """
    r = np.asarray(r, dtype=float)
    return 0.75 * ((r + bs / 2.0) ** 4 - (r - bs / 2.0) ** 4) / ((r + bs / 2.0) ** 3 - (r - bs / 2.0) ** 3)


def _pinv_sym(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a = 0.5 * (a + a.T)
    return np.linalg.pinv(a)


def _chi2(residual: np.ndarray, cov_inv: np.ndarray) -> float:
    r = np.asarray(residual, dtype=float)
    return float(r.T @ cov_inv @ r)


def _fit_broadband_6param(*, r: np.ndarray, residual: np.ndarray, cov_inv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Ross の broadband に揃え、mono/quad それぞれに
      a0 + a1/r + a2/r^2
    を当てはめる（合計6係数）。

    Fitは GLS: minimize (res - Xb)^T C^{-1} (res - Xb).
    Returns (coef[6], fitted_component=Xb).
    """
    r = np.asarray(r, dtype=float)
    residual = np.asarray(residual, dtype=float)
    if r.ndim != 1:
        raise ValueError("r must be 1D")
    n = int(r.size)
    if residual.shape != (2 * n,):
        raise ValueError("residual must have shape (2*n,)")

    X = np.zeros((2 * n, 6), dtype=float)
    # mono block
    X[:n, 0] = 1.0
    X[:n, 1] = 1.0 / r
    X[:n, 2] = 1.0 / (r * r)
    # quad block
    X[n:, 3] = 1.0
    X[n:, 4] = 1.0 / r
    X[n:, 5] = 1.0 / (r * r)

    xt_cinv = X.T @ cov_inv
    a = xt_cinv @ X
    b = xt_cinv @ residual
    coef = np.linalg.pinv(a) @ b
    fit = X @ coef
    return coef.astype(float), fit.astype(float)


def _p2(mu: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    return 0.5 * (3.0 * mu * mu - 1.0)


def _p4(mu: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    mu2 = mu * mu
    mu4 = mu2 * mu2
    return 0.125 * (35.0 * mu4 - 30.0 * mu2 + 3.0)


def _load_ross_template(*, base_dir: Path, mod: str, scale: float = 2.1) -> dict[str, np.ndarray]:
    """
    Ross baofit_pub2D.py が使う BAOtemplates を読む（xi0/xi2/xi4）。
    """
    p0 = base_dir / f"xi0{mod}"
    p2 = base_dir / f"xi2{mod}"
    p4 = base_dir / f"xi4{mod}"
    if not (p0.exists() and p2.exists() and p4.exists()):
        raise FileNotFoundError(f"missing template files under {base_dir} for mod={mod}")

    def _read_xy(p: Path) -> tuple[np.ndarray, np.ndarray]:
        x: list[float] = []
        y: list[float] = []
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            a = t.split()
            if len(a) < 2:
                continue
            x.append(float(a[0]))
            y.append(float(a[1]))
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    r0, xi0 = _read_xy(p0)
    r2, xi2 = _read_xy(p2)
    r4, xi4 = _read_xy(p4)
    if not (np.allclose(r0, r2) and np.allclose(r0, r4)):
        raise ValueError("template r grids mismatch (xi0/xi2/xi4)")
    r = r0
    return {"r": r, "xi0": scale * xi0, "xi2": scale * xi2, "xi4": scale * xi4}


def _interp1(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    return np.interp(x, xp, fp, left=0.0, right=0.0)


def _ap_multipoles_from_template(
    *,
    r: float,
    alpha: float,
    eps: float,
    tpl: dict[str, np.ndarray],
    include_xi4: bool,
    nmu: int = 400,
) -> tuple[float, float]:
    """
    xi(mu) = xi0(r') + P2(mu') xi2(r') + P4(mu') xi4(r') を
    AP変換 (alpha,eps) の下で積分し、(xi0, xi2) を返す。

    パラメータ化（標準）:
      alpha_parallel = alpha * (1+eps)^2
      alpha_perp     = alpha / (1+eps)
    """
    r = float(r)
    alpha = float(alpha)
    eps = float(eps)
    if nmu <= 0:
        raise ValueError("nmu must be >0")
    onepe = 1.0 + eps
    ar = alpha * (onepe * onepe)
    at = alpha / onepe

    mu = (np.arange(nmu, dtype=float) + 0.5) / float(nmu)
    al = np.sqrt(mu * mu * (ar * ar) + (1.0 - mu * mu) * (at * at))
    mup = (mu * ar) / al
    rp = r * al

    rt = tpl["r"]
    xi0p = _interp1(rp, rt, tpl["xi0"])
    xi2p = _interp1(rp, rt, tpl["xi2"])
    ximu = xi0p + _p2(mup) * xi2p
    if include_xi4:
        xi4p = _interp1(rp, rt, tpl["xi4"])
        ximu = ximu + _p4(mup) * xi4p

    dmu = 1.0 / float(nmu)
    xi0 = float(np.sum(ximu) * dmu)
    xi2 = float(5.0 * np.sum(ximu * _p2(mu)) * dmu)
    return xi0, xi2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Align recon-gap residual evaluation to Ross baofit_pub2D axis.")
    ap.add_argument(
        "--ross-dir",
        default=str(_ROOT / "data" / "cosmology" / "ross_2016_combineddr12_corrfunc"),
        help="Ross 2016 published multipoles directory",
    )
    ap.add_argument("--bincent-min", type=int, default=0, help="min bincent id (default: 0)")
    ap.add_argument("--bincent-max", type=int, default=4, help="max bincent id (default: 4)")
    ap.add_argument("--s-min", type=float, default=50.0, help="Ross fit range min (strict; default: 50)")
    ap.add_argument("--s-max", type=float, default=150.0, help="Ross fit range max (strict; default: 150)")
    ap.add_argument("--bs", type=float, default=5.0, help="bin size for rbc (default: 5)")
    ap.add_argument("--sample", default="cmasslowztot", help="catalog sample name (default: cmasslowztot)")
    ap.add_argument("--caps", default="combined", help="caps name (default: combined)")
    ap.add_argument("--dist", default="lcdm", choices=["lcdm", "pbg"], help="distance mapping label (default: lcdm)")
    ap.add_argument(
        "--catalog-recon-suffix",
        default="__recon_mw_multigrid_ani_mw_rw_boss_recon_rz_zmin0p2_zmax0p75_g512_bias1p85_f0p757",
        help="suffix of catalog recon NPZ (default: mw_multigrid baseline)",
    )
    ap.add_argument(
        "--template-mod",
        default="Challenge_matterpower0.44.02.54.015.01.0.dat",
        help="Ross baofit_pub2D.py template mod (default: Challenge_matterpower0.44.02.54.015.01.0.dat)",
    )
    ap.add_argument(
        "--template-dir",
        default=str(_ROOT / ".tmp_lssanalysis" / "BAOtemplates"),
        help="Ross BAOtemplates directory (default: .tmp_lssanalysis/BAOtemplates)",
    )
    ap.add_argument(
        "--eps-grid",
        default="-0.05,-0.04,-0.03,-0.02,-0.01,0,0.01,0.02,0.03,0.04,0.05",
        help="comma-separated epsilon grid for xi4 mixing check (default: -0.05..0.05 step 0.01)",
    )
    ap.add_argument(
        "--out-png",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_bao_recon_gap_ross_eval_alignment.png"),
        help="output png path",
    )
    ap.add_argument(
        "--out-json",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_bao_recon_gap_ross_eval_alignment_metrics.json"),
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

    s_min = float(args.s_min)
    s_max = float(args.s_max)
    if not (0.0 < s_min < s_max):
        raise SystemExit("--s-min/max invalid (requires 0 < s_min < s_max)")

    bs = float(args.bs)
    if bs <= 0.0:
        raise SystemExit("--bs must be >0")

    # Load catalog recon curves (xi0/xi2) for each zbin once.
    zbins = [1, 2, 3]
    zbin_to_b = {1: "b1", 2: "b2", 3: "b3"}
    cat_curves: dict[int, dict[str, np.ndarray]] = {}
    used_inputs: list[str] = []
    for zbin in zbins:
        b = zbin_to_b[zbin]
        cat_npz = (
            _ROOT
            / "output"
            / "cosmology"
            / f"cosmology_bao_xi_from_catalogs_{args.sample}_{args.caps}_{args.dist}_{b}{args.catalog_recon_suffix}.npz"
        )
        if not cat_npz.exists():
            raise SystemExit(f"missing catalog recon npz: {cat_npz}")
        used_inputs.append(str(cat_npz))
        with np.load(cat_npz, allow_pickle=True) as z:
            cat_curves[zbin] = {
                "s": np.asarray(z["s"], dtype=float),
                "xi0": np.asarray(z["xi0"], dtype=float),
                "xi2": np.asarray(z["xi2"], dtype=float),
            }

    # Evaluate recon gap significance using Ross covariance (monoquad) in the Ross fit range.
    results: dict[str, Any] = {"per": {}, "summary": {}}
    for zbin in zbins:
        results["per"][str(zbin)] = {}
        b = zbin_to_b[zbin]

        s_cat = cat_curves[zbin]["s"]
        xi0_cat = cat_curves[zbin]["xi0"]
        xi2_cat = cat_curves[zbin]["xi2"]

        for binc in bincents:
            ross_mono = ross_dir / f"Ross_2016_COMBINEDDR12_zbin{zbin}_correlation_function_monopole_post_recon_bincent{binc}.dat"
            ross_quad = ross_dir / f"Ross_2016_COMBINEDDR12_zbin{zbin}_correlation_function_quadrupole_post_recon_bincent{binc}.dat"
            ross_cov = ross_dir / f"Ross_2016_COMBINEDDR12_zbin{zbin}_covariance_monoquad_post_recon_bincent{binc}.dat"
            if not (ross_mono.exists() and ross_quad.exists() and ross_cov.exists()):
                raise SystemExit(f"missing Ross files: zbin={zbin} bincent={binc} under {ross_dir}")
            used_inputs.extend([str(ross_mono), str(ross_quad), str(ross_cov)])

            s0, x0, _ = _parse_ross_xi_file(ross_mono)
            s2, x2, _ = _parse_ross_xi_file(ross_quad)
            if not np.allclose(s0, s2):
                raise SystemExit(f"Ross mono/quad s grids mismatch: zbin={zbin} bincent={binc}")

            cov_full = _read_cov(ross_cov)
            n_all = int(s0.size)
            if cov_full.shape != (2 * n_all, 2 * n_all):
                raise SystemExit(
                    f"Ross covariance shape mismatch: zbin={zbin} bincent={binc} cov={cov_full.shape} expected={(2*n_all,2*n_all)}"
                )

            m = (s0 > s_min) & (s0 < s_max)  # match baofit_pub2D.py (strict)
            idx = np.nonzero(m)[0].astype(int)
            if idx.size < 8:
                raise SystemExit(f"too few points after fit-range cut: zbin={zbin} bincent={binc} n={idx.size}")

            s_sel = s0[idx]
            n = int(s_sel.size)
            # Catalog values interpolated to Ross bin centers (bincent shifts).
            x0c = np.interp(s_sel, s_cat, xi0_cat)
            x2c = np.interp(s_sel, s_cat, xi2_cat)

            # Residual vector (catalog - Ross), in Ross data-vector order: [mono..., quad...]
            res = np.concatenate([(x0c - x0[idx]), (x2c - x2[idx])]).astype(float)

            # Sub-covariance in the same order
            idx_dv = np.concatenate([idx, idx + n_all]).astype(int)
            cov = cov_full[np.ix_(idx_dv, idx_dv)]
            cov_inv = _pinv_sym(cov)

            # Ross uses rbc for model evaluation and broadband basis; use rbc here as evaluation axis.
            r_eval = _rbc(s_sel, bs=bs)

            chi2_raw = _chi2(res, cov_inv=cov_inv)
            dof_raw = int(res.size)
            coef6, fit6 = _fit_broadband_6param(r=r_eval, residual=res, cov_inv=cov_inv)
            res_after = res - fit6
            chi2_after = _chi2(res_after, cov_inv=cov_inv)
            dof_after = int(res.size - 6)

            # Diagnostics: mono-only / quad-only (ignore cross-cov; rough indicator only).
            cov00 = cov[:n, :n]
            cov22 = cov[n:, n:]
            chi2_0_raw = _chi2(res[:n], cov_inv=_pinv_sym(cov00))
            chi2_2_raw = _chi2(res[n:], cov_inv=_pinv_sym(cov22))
            chi2_0_after = _chi2(res_after[:n], cov_inv=_pinv_sym(cov00))
            chi2_2_after = _chi2(res_after[n:], cov_inv=_pinv_sym(cov22))

            results["per"][str(zbin)][str(binc)] = {
                "zbin": int(zbin),
                "bincent": int(binc),
                "s_range_strict_mpc_h": [float(s_min), float(s_max)],
                "n_bins": int(n),
                "chi2_raw": float(chi2_raw),
                "dof_raw": int(dof_raw),
                "chi2_dof_raw": float(chi2_raw / max(dof_raw, 1)),
                "chi2_after_broadband": float(chi2_after),
                "dof_after_broadband": int(dof_after),
                "chi2_dof_after_broadband": float(chi2_after / max(dof_after, 1)),
                "broadband_coef_mono": {"a0": float(coef6[0]), "a1": float(coef6[1]), "a2": float(coef6[2])},
                "broadband_coef_quad": {"a0": float(coef6[3]), "a1": float(coef6[4]), "a2": float(coef6[5])},
                "diagnostics_ignore_crosscov": {
                    "chi2_mono_raw": float(chi2_0_raw),
                    "chi2_quad_raw": float(chi2_2_raw),
                    "chi2_mono_after": float(chi2_0_after),
                    "chi2_quad_after": float(chi2_2_after),
                },
                "inputs": {
                    "catalog_npz_recon": str(
                        _ROOT
                        / "output"
                        / "cosmology"
                        / f"cosmology_bao_xi_from_catalogs_{args.sample}_{args.caps}_{args.dist}_{b}{args.catalog_recon_suffix}.npz"
                    ),
                    "ross_mono": str(ross_mono),
                    "ross_quad": str(ross_quad),
                    "ross_cov": str(ross_cov),
                },
            }

        # Summaries across bincent
        chi2d_raw = [float(results["per"][str(zbin)][str(b)]["chi2_dof_raw"]) for b in bincents]
        chi2d_after = [float(results["per"][str(zbin)][str(b)]["chi2_dof_after_broadband"]) for b in bincents]
        results["summary"][str(zbin)] = {
            "chi2_dof_raw": {"min": float(np.min(chi2d_raw)), "median": float(np.median(chi2d_raw)), "max": float(np.max(chi2d_raw))},
            "chi2_dof_after_broadband": {
                "min": float(np.min(chi2d_after)),
                "median": float(np.median(chi2d_after)),
                "max": float(np.max(chi2d_after)),
            },
        }

    # ξ4 mixing check (model-side, AP変換での ξ4→ξ2 の寄与の目安)
    try:
        eps_grid = [float(x.strip()) for x in str(args.eps_grid).split(",") if x.strip()]
    except Exception as e:
        raise SystemExit(f"invalid --eps-grid: {e}") from e
    if not eps_grid:
        raise SystemExit("--eps-grid must not be empty")

    tpl = _load_ross_template(base_dir=Path(args.template_dir), mod=str(args.template_mod))
    # Use the same r-sampling as Ross fit range (bin centers), for an order-of-magnitude check.
    r_probe = np.arange(52.5, 150.0, 5.0, dtype=float)
    xi4_mix: dict[str, Any] = {"template_mod": str(args.template_mod), "template_dir": str(args.template_dir), "r_probe": r_probe.tolist()}
    mix_rows: list[dict[str, float]] = []
    for eps in eps_grid:
        d_s2xi2: list[float] = []
        rel_s2xi2: list[float] = []
        for r in r_probe.tolist():
            _, xi2_full = _ap_multipoles_from_template(r=r, alpha=1.0, eps=float(eps), tpl=tpl, include_xi4=True)
            _, xi2_no4 = _ap_multipoles_from_template(r=r, alpha=1.0, eps=float(eps), tpl=tpl, include_xi4=False)
            d = (r * r) * (xi2_full - xi2_no4)
            denom = max(abs((r * r) * xi2_full), 1e-30)
            d_s2xi2.append(float(abs(d)))
            rel_s2xi2.append(float(abs(d) / denom))
        mix_rows.append(
            {
                "eps": float(eps),
                "max_abs_delta_s2_xi2": float(np.max(d_s2xi2)),
                "median_abs_delta_s2_xi2": float(np.median(d_s2xi2)),
                "max_rel_delta_s2_xi2": float(np.max(rel_s2xi2)),
                "median_rel_delta_s2_xi2": float(np.median(rel_s2xi2)),
            }
        )
    xi4_mix["rows"] = mix_rows

    # Plot
    _set_japanese_font()
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.gridspec import GridSpec  # noqa: E402

    fig = plt.figure(figsize=(12.5, 8.5))
    gs = GridSpec(2, 3, height_ratios=[2.0, 1.0], hspace=0.35, wspace=0.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_mix = fig.add_subplot(gs[1, :])

    x = np.asarray(bincents, dtype=int)
    for ax, zbin in zip(axes, zbins):
        y_raw = np.array([results["per"][str(zbin)][str(b)]["chi2_dof_raw"] for b in bincents], dtype=float)
        y_after = np.array([results["per"][str(zbin)][str(b)]["chi2_dof_after_broadband"] for b in bincents], dtype=float)
        ax.plot(x, y_raw, linestyle="--", marker="o", markersize=4, label="raw (catalog−Ross)")
        ax.plot(x, y_after, linestyle="-", marker="s", markersize=4, label="after broadband (6 params)")
        ax.axhline(1.0, color="#999999", linewidth=1.0, linestyle=":")
        ax.set_title(f"zbin{zbin}: χ²/dof（Ross cov）")
        ax.set_xlabel("bincent")
        ax.set_ylabel("χ²/dof")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    # xi4 mixing (AP)
    eps_vals = np.array([r["eps"] for r in mix_rows], dtype=float)
    y_abs = np.array([r["max_abs_delta_s2_xi2"] for r in mix_rows], dtype=float)
    y_rel = np.array([r["max_rel_delta_s2_xi2"] for r in mix_rows], dtype=float)
    ax_mix.plot(eps_vals, y_abs, marker="o", linewidth=2.0, label="max |Δ(s²ξ2)| (xi4 on/off)")
    ax_mix2 = ax_mix.twinx()
    ax_mix2.plot(eps_vals, y_rel, marker="s", linewidth=2.0, color="#ff7f0e", label="max relative Δ(s²ξ2)")
    ax_mix.axhline(0.0, color="#999999", linewidth=1.0, linestyle=":")
    ax_mix.set_xlabel("ε（AP warping）")
    ax_mix.set_ylabel("max |Δ(s²ξ₂)|")
    ax_mix2.set_ylabel("max relative Δ")
    ax_mix.grid(True, alpha=0.25)
    # combined legend
    h1, l1 = ax_mix.get_legend_handles_labels()
    h2, l2 = ax_mix2.get_legend_handles_labels()
    ax_mix.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax_mix.set_title("ξ4（モデル側）: AP変換での ξ4→ξ2 混合（Ross template; α=1）")

    fig.suptitle("Ross基準の残差評価（cov/broadband）と ξ4（モデル側）の混合チェック", y=0.98)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    fig.savefig(out_png, dpi=150)

    metrics: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "4.5B.17.4.4.2 (Ross-eval alignment: cov+broadband; xi4(AP) mixing)",
        "inputs": {
            "ross_dir": str(ross_dir),
            "bincent_range": [bincent_min, bincent_max],
            "s_range_strict_mpc_h": [float(s_min), float(s_max)],
            "bs_mpc_h": float(bs),
            "sample": str(args.sample),
            "caps": str(args.caps),
            "dist": str(args.dist),
            "catalog_recon_suffix": str(args.catalog_recon_suffix),
            "xi4_template_mod": str(args.template_mod),
            "xi4_template_dir": str(args.template_dir),
            "eps_grid": eps_grid,
            "used_inputs": sorted(set(used_inputs)),
        },
        "results": {"recon_gap_eval": results, "xi4_ap_mixing": xi4_mix},
        "outputs": {"png": str(out_png), "metrics_json": str(out_json)},
    }
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    worklog.append_event(
        {
            "domain": "cosmology",
            "step": "4.5B.17.4.4.2 (Ross-eval alignment: cov+broadband; xi4(AP) mixing)",
            "inputs": sorted(set(used_inputs)),
            "outputs": [str(out_png), str(out_json)],
            "notes": {
                "s_range_strict": [float(s_min), float(s_max)],
                "bincent_range": [bincent_min, bincent_max],
                "catalog": {"sample": str(args.sample), "caps": str(args.caps), "dist": str(args.dist), "suffix": str(args.catalog_recon_suffix)},
                "xi4_template_mod": str(args.template_mod),
                "eps_grid": eps_grid,
            },
        }
    )
    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

