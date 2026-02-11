#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_recon_param_scan.py

Step 16.4（BAO一次情報：銀河+random）/ Phase A（スクリーニング）:
catalog-based ξℓ（Corrfunc; galaxy+random）の reconstruction（recon）パラメータを振り、
公開 post-recon multipoles（Ross+2016）に対する一致度（RMSE）を比較する。

目的：
- recon の前提（grid/smoothing/bias/mode）が ξ0/ξ2（特に ξ2）に与える影響を定量化し、
  Phase B（P-model距離写像へ差し替え、reconも自前化）へ進む前に、
  「LCDM fiducialで公開post-reconをどの程度再現できるか」を固定する。

出力（固定）:
- output/cosmology/cosmology_bao_recon_param_scan.png
- output/cosmology/cosmology_bao_recon_param_scan_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402

_WIN_ABS_RE = re.compile(r"^[a-zA-Z]:[\\/]")
_WSL_ABS_RE = re.compile(r"^/mnt/([a-zA-Z])/(.+)$")


def _resolve_path_like(p: Any) -> Optional[Path]:
    if p is None:
        return None
    s = str(p).strip()
    if not s:
        return None
    if os.name == "nt":
        m = _WSL_ABS_RE.match(s)
        if m:
            drive = m.group(1).upper()
            rest = m.group(2).replace("/", "\\")
            return Path(f"{drive}:\\{rest}")
    else:
        if _WIN_ABS_RE.match(s):
            drive = s[0].lower()
            rest = s[2:].lstrip("\\/").replace("\\", "/")
            return Path(f"/mnt/{drive}/{rest}")
    path = Path(s)
    if path.is_absolute():
        return path
    return _ROOT / path


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


def _cov_inv_with_jitter(cov: np.ndarray) -> np.ndarray:
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError(f"invalid covariance shape: {c.shape}")
    diag = np.diag(c)
    jitter = float(np.nanmax(diag)) * 1e-12 if diag.size else 0.0
    if not np.isfinite(jitter) or jitter <= 0.0:
        jitter = 1e-12
    cj = c + jitter * np.eye(c.shape[0], dtype=np.float64)
    try:
        return np.linalg.inv(cj)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(cj)


def _chi2_from_cov_inv(residual: np.ndarray, cov_inv: np.ndarray) -> float:
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    ci = np.asarray(cov_inv, dtype=np.float64)
    if ci.shape[0] != ci.shape[1] or ci.shape[0] != r.size:
        raise ValueError(f"chi2 dim mismatch: residual={r.size} cov_inv={ci.shape}")
    return float(r @ ci @ r)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def _align_interp(pub_s: np.ndarray, pub_y: np.ndarray, cat_s: np.ndarray, cat_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pub_s = np.asarray(pub_s, dtype=float)
    pub_y = np.asarray(pub_y, dtype=float)
    cat_s = np.asarray(cat_s, dtype=float)
    cat_y = np.asarray(cat_y, dtype=float)
    y_cat = np.interp(pub_s, cat_s, cat_y)
    return pub_s, pub_y, y_cat


@dataclass(frozen=True)
class CatalogCase:
    zbin: int
    out_tag: Optional[str]
    recon: Dict[str, Any]
    npz_path: Path
    metrics_path: Path


def _iter_metrics_files() -> Iterable[Path]:
    out_dir = _ROOT / "output" / "cosmology"
    yield from sorted(out_dir.glob("cosmology_bao_xi_from_catalogs_*_metrics.json"))


def _zbin_label_to_int(z_bin: str) -> Optional[int]:
    z = str(z_bin).strip().lower()
    if z == "b1":
        return 1
    if z == "b2":
        return 2
    if z == "b3":
        return 3
    return None


def _load_catalog_cases(
    *,
    sample: str,
    caps: str,
    dist: str,
    weight_scheme: str,
    random_kind: str,
    require_zbin: bool,
    out_tag_prefix: str,
    include_baseline: bool,
) -> List[CatalogCase]:
    cases: List[CatalogCase] = []
    for p in _iter_metrics_files():
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        params = (m.get("params", {}) or {}) if isinstance(m, dict) else {}
        if str(params.get("sample", "")) != str(sample):
            continue
        if str(params.get("caps", "")) != str(caps):
            continue
        if str(params.get("distance_model", "")) != str(dist):
            continue
        if str(params.get("weight_scheme", "")) != str(weight_scheme):
            continue
        if str(params.get("random_kind", "")) != str(random_kind):
            continue

        z_bin = ((params.get("z_cut", {}) or {}).get("bin", "none")) if isinstance(params.get("z_cut", {}), dict) else "none"
        z_int = _zbin_label_to_int(str(z_bin))
        if require_zbin and z_int is None:
            continue
        if not require_zbin and z_int is not None:
            continue

        tag_in = params.get("out_tag", None)
        out_tag: Optional[str]
        if tag_in is None:
            out_tag = None
        else:
            s = str(tag_in).strip()
            out_tag = s if s and s.lower() != "null" else None

        if out_tag is None:
            if not include_baseline:
                continue
        else:
            if out_tag_prefix and (not str(out_tag).startswith(str(out_tag_prefix))):
                continue

        outputs = m.get("outputs", {}) if isinstance(m.get("outputs", {}), dict) else {}
        npz_path = _resolve_path_like(outputs.get("npz", None))
        if npz_path is None:
            continue
        if not npz_path.is_absolute():
            npz_path = (_ROOT / npz_path).resolve()
        if not npz_path.exists():
            continue

        recon = (params.get("recon", {}) or {}) if isinstance(params.get("recon", {}), dict) else {}
        cases.append(
            CatalogCase(zbin=int(z_int) if z_int is not None else 0, out_tag=out_tag, recon=dict(recon), npz_path=npz_path, metrics_path=p)
        )
    return cases


def _group_by_out_tag(cases: List[CatalogCase]) -> Dict[str, Dict[int, CatalogCase]]:
    out: Dict[str, Dict[int, CatalogCase]] = {}
    for c in cases:
        key = c.out_tag if c.out_tag is not None else "baseline"
        out.setdefault(key, {})[int(c.zbin)] = c
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scan reconstruction params by RMSE vs Ross 2016 post-recon multipoles.")
    ap.add_argument("--sample", default="cmasslowztot", help="BOSS sample (default: cmasslowztot)")
    ap.add_argument("--caps", default="combined", help="caps: combined (default)")
    ap.add_argument("--dist", default="lcdm", help="distance model used for catalog coordinate (default: lcdm)")
    ap.add_argument("--weight-scheme", default="boss_default", help="weight scheme filter (default: boss_default)")
    ap.add_argument("--random-kind", default="random1", help="random kind filter (default: random1)")
    ap.add_argument("--require-zbin", action="store_true", help="only include z-binned cases (b1/b2/b3)")
    ap.add_argument("--out-tag-prefix", default="recon_grid_iso", help="prefix filter for out_tag (default: recon_grid_iso)")
    ap.add_argument("--include-baseline", action="store_true", help="include baseline (out_tag=null) in scan")
    ap.add_argument(
        "--ross-dir",
        default="data/cosmology/ross_2016_combineddr12_corrfunc",
        help="Ross 2016 post-recon multipoles dir",
    )
    ap.add_argument("--bincent", type=int, default=0, help="Ross bin center shift index (0..4; default: 0)")
    ap.add_argument("--s-min", type=float, default=30.0)
    ap.add_argument("--s-max", type=float, default=150.0)
    ap.add_argument("--xi2-weight", type=float, default=2.0, help="weight for xi2 RMSE in overall score (default: 2)")
    ap.add_argument("--rank-by", choices=["rmse", "chi2"], default="chi2", help="ranking metric (default: chi2)")
    ap.add_argument("--top-n", type=int, default=10, help="plot top-N configs by score (default: 10)")
    ap.add_argument(
        "--out-png",
        default="output/cosmology/cosmology_bao_recon_param_scan.png",
        help="output PNG path (default: fixed)",
    )
    ap.add_argument(
        "--out-json",
        default="output/cosmology/cosmology_bao_recon_param_scan_metrics.json",
        help="output metrics JSON path (default: fixed)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    sample = str(args.sample)
    caps = str(args.caps)
    dist = str(args.dist)
    weight_scheme = str(args.weight_scheme)
    random_kind = str(args.random_kind)
    require_zbin = bool(args.require_zbin)
    out_tag_prefix = str(args.out_tag_prefix).strip()
    include_baseline = bool(args.include_baseline)
    ross_dir = (_ROOT / str(args.ross_dir)).resolve()
    if not ross_dir.exists():
        alt = (_ROOT / "data" / "cosmology" / "ross_2016_combineddr12_corrfunc").resolve()
        if alt.exists():
            ross_dir = alt
    bincent = int(args.bincent)
    if bincent < 0 or bincent > 4:
        raise SystemExit("--bincent must be in 0..4")
    s_min = float(args.s_min)
    s_max = float(args.s_max)
    xi2_weight = float(args.xi2_weight)
    if not (np.isfinite(xi2_weight) and xi2_weight >= 0.0):
        raise SystemExit("--xi2-weight must be >= 0")
    rank_by = str(args.rank_by)

    out_png = (_ROOT / str(args.out_png)).resolve()
    out_json = (_ROOT / str(args.out_json)).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    cases = _load_catalog_cases(
        sample=sample,
        caps=caps,
        dist=dist,
        weight_scheme=weight_scheme,
        random_kind=random_kind,
        require_zbin=True if require_zbin else True,  # this scan is for z-bins only
        out_tag_prefix=out_tag_prefix,
        include_baseline=include_baseline,
    )
    groups = _group_by_out_tag(cases)

    z_bins = [1, 2, 3]
    # Load Ross published curves once.
    ross: Dict[int, Dict[str, Any]] = {}
    for zb in z_bins:
        p_mono = ross_dir / f"Ross_2016_COMBINEDDR12_zbin{zb}_correlation_function_monopole_post_recon_bincent{bincent}.dat"
        p_quad = ross_dir / f"Ross_2016_COMBINEDDR12_zbin{zb}_correlation_function_quadrupole_post_recon_bincent{bincent}.dat"
        if not (p_mono.exists() and p_quad.exists()):
            raise SystemExit(f"missing Ross files for zbin{zb} bincent{bincent}: {ross_dir}")
        s0, xi0, e0 = _load_ross_dat(p_mono)
        s2, xi2, e2 = _load_ross_dat(p_quad)
        p_cov = ross_dir / f"Ross_2016_COMBINEDDR12_zbin{zb}_covariance_monoquad_post_recon_bincent{bincent}.dat"
        if not p_cov.exists():
            raise SystemExit(f"missing Ross covariance for zbin{zb} bincent{bincent}: {p_cov}")
        cov_full = _load_ross_covariance(p_cov)
        if cov_full.shape[0] != cov_full.shape[1] or cov_full.shape[0] != 2 * s0.size:
            raise SystemExit(f"unexpected Ross covariance shape for zbin{zb} (expected {2*s0.size}x{2*s0.size}): {cov_full.shape}")

        idx_s = np.where((s0 >= s_min) & (s0 <= s_max))[0]
        if idx_s.size < 6:
            raise SystemExit(f"too few Ross bins in range [{s_min},{s_max}] for zbin{zb} (n={idx_s.size})")
        idx_all = np.concatenate([idx_s, idx_s + s0.size])
        cov_sub = cov_full[np.ix_(idx_all, idx_all)]
        cov_inv = _cov_inv_with_jitter(cov_sub)
        dof = int(idx_all.size)
        # Convert to s^2 xi for the same display as other scripts.
        ross[zb] = {
            "s0": s0,
            "xi0": xi0,
            "xi2": xi2,
            "idx_s": idx_s.astype(np.int64, copy=False),
            "cov_inv": cov_inv,
            "dof": dof,
            "s2_xi0": (s0 * s0) * xi0,
            "s2_xi2": (s2 * s2) * xi2,
            "mask0": (s0 >= s_min) & (s0 <= s_max),
            "mask2": (s2 >= s_min) & (s2 <= s_max),
        }

    results: List[Dict[str, Any]] = []
    for out_tag, by_z in sorted(groups.items(), key=lambda kv: kv[0]):
        # Require all z-bins.
        if any(z not in by_z for z in z_bins):
            continue

        per_bin: Dict[str, Any] = {}
        rmse0_list: list[float] = []
        rmse2_list: list[float] = []
        chi2_dof_list: list[float] = []
        recon_meta: Dict[str, Any] = {}

        for zb in z_bins:
            c = by_z[zb]
            with np.load(str(c.npz_path)) as z:
                s_cat = np.asarray(z["s"], dtype=float)
                xi0_cat = np.asarray(z["xi0"], dtype=float)
                xi2_cat = np.asarray(z["xi2"], dtype=float)
            s2_xi0_cat = (s_cat * s_cat) * xi0_cat
            s2_xi2_cat = (s_cat * s_cat) * xi2_cat

            pub = ross[zb]
            s0 = np.asarray(pub["s0"], dtype=float)
            m0 = np.asarray(pub["mask0"], dtype=bool)
            s2 = np.asarray(pub["s0"], dtype=float)  # same grid for mono/quad in Ross package
            m2 = np.asarray(pub["mask2"], dtype=bool)
            y0_pub = np.asarray(pub["s2_xi0"], dtype=float)
            y2_pub = np.asarray(pub["s2_xi2"], dtype=float)

            s_al0, y0_pub_al, y0_cat_al = _align_interp(s0[m0], y0_pub[m0], s_cat, s2_xi0_cat)
            s_al2, y2_pub_al, y2_cat_al = _align_interp(s2[m2], y2_pub[m2], s_cat, s2_xi2_cat)

            rmse0 = _rmse(y0_pub_al, y0_cat_al)
            rmse2 = _rmse(y2_pub_al, y2_cat_al)
            rmse0_list.append(rmse0)
            rmse2_list.append(rmse2)

            # chi2/dof using Ross covariance (xi, not s^2 xi).
            idx_s = np.asarray(pub["idx_s"], dtype=np.int64)
            s_sub = s0[idx_s]
            xi0_pub_sub = np.asarray(pub["xi0"], dtype=float)[idx_s]
            xi2_pub_sub = np.asarray(pub["xi2"], dtype=float)[idx_s]
            xi0_cat_sub = np.interp(s_sub, s_cat, xi0_cat)
            xi2_cat_sub = np.interp(s_sub, s_cat, xi2_cat)
            res = np.concatenate([xi0_cat_sub - xi0_pub_sub, xi2_cat_sub - xi2_pub_sub])
            chi2 = _chi2_from_cov_inv(res, np.asarray(pub["cov_inv"], dtype=np.float64))
            dof = int(pub.get("dof", res.size))
            chi2_dof = float(chi2 / max(1, dof))
            chi2_dof_list.append(chi2_dof)

            per_bin[str(zb)] = {
                "npz": str(c.npz_path),
                "rmse_s2_xi0": rmse0,
                "rmse_s2_xi2": rmse2,
                "chi2_dof_xi0_xi2_ross_post_recon": chi2_dof,
            }
            if zb == 2:
                recon_meta = dict(c.recon or {})

        rmse0_mean = float(np.mean(rmse0_list)) if rmse0_list else float("nan")
        rmse2_mean = float(np.mean(rmse2_list)) if rmse2_list else float("nan")
        score = float(np.sqrt((rmse0_mean * rmse0_mean + xi2_weight * rmse2_mean * rmse2_mean) / (1.0 + xi2_weight)))
        chi2_dof_mean = float(np.mean(chi2_dof_list)) if chi2_dof_list else float("nan")

        results.append(
            {
                "out_tag": None if out_tag == "baseline" else out_tag,
                "label": out_tag,
                "recon": recon_meta,
                "rmse_mean": {"s2_xi0": rmse0_mean, "s2_xi2": rmse2_mean, "score": score, "xi2_weight": xi2_weight},
                "chi2_dof_mean": {"xi0_xi2": chi2_dof_mean},
                "per_zbin": per_bin,
            }
        )

    # Rank by requested metric.
    if rank_by == "rmse":
        ranked = sorted(results, key=lambda r: float(r.get("rmse_mean", {}).get("score", float("inf"))))
    else:
        ranked = sorted(results, key=lambda r: float((r.get("chi2_dof_mean", {}) or {}).get("xi0_xi2", float("inf"))))

    # Plot: top-N configs.
    top_n = max(1, int(args.top_n))
    show = ranked[:top_n]
    labels = [str(r["label"]) for r in show]
    rmse0 = np.array([float(r["rmse_mean"]["s2_xi0"]) for r in show], dtype=float)
    rmse2 = np.array([float(r["rmse_mean"]["s2_xi2"]) for r in show], dtype=float)
    scores = np.array([float(r["rmse_mean"]["score"]) for r in show], dtype=float)
    chi2_dof = np.array([float((r.get("chi2_dof_mean", {}) or {}).get("xi0_xi2", float("nan"))) for r in show], dtype=float)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), dpi=160, constrained_layout=True)
    ax0, ax1 = axes

    x = np.arange(len(labels), dtype=float)
    ax0.bar(x - 0.15, rmse0, width=0.3, label="RMSE(s²ξ0)", color="#1f77b4")
    ax0.bar(x + 0.15, rmse2, width=0.3, label="RMSE(s²ξ2)", color="#ff7f0e")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax0.set_ylabel("RMSE")
    ax0.set_title("Ross post-recon vs catalog-based (top configs)")
    ax0.grid(True, axis="y", alpha=0.25)
    ax0.legend(fontsize=9)

    ax1.bar(x, chi2_dof, width=0.55, color="#2ca02c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("chi2/dof (Ross cov; xi0+xi2)")
    ax1.set_title(f"rank_by={rank_by}  (lower is better)")
    ax1.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"BAO recon param scan (catalog-based ξℓ): sample={sample}, dist={dist}, out_tag_prefix={out_tag_prefix or 'any'}",
        fontsize=12,
    )
    fig.savefig(out_png)
    plt.close(fig)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "16.4 (BAO recon param scan vs Ross post-recon)",
        "inputs": {
            "sample": sample,
            "caps": caps,
            "dist": dist,
            "weight_scheme": weight_scheme,
            "random_kind": random_kind,
            "out_tag_prefix": out_tag_prefix,
            "include_baseline": include_baseline,
            "ross_dir": str(ross_dir),
            "bincent": bincent,
            "s_range_mpc_h": [s_min, s_max],
            "xi2_weight": xi2_weight,
            "rank_by": rank_by,
        },
        "results": {"ranked": ranked},
        "outputs": {"png": str(out_png), "metrics_json": str(out_json)},
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_bao_recon_param_scan",
                "argv": sys.argv,
                "outputs": {"png": out_png, "metrics_json": out_json},
                "metrics": payload.get("inputs", {}),
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
