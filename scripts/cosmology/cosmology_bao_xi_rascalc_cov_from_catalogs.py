#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_xi_rascalc_cov_from_catalogs.py

Phase 4（宇宙論）/ Step 4.5B.21.4.4.6.3（DESI DR1: RascalC + Jackknife）:
`cosmology_bao_xi_from_catalogs.py` で計算した「銀河+random→ξℓ」の case に対して、
RascalC（Legendre projected）で dv=[xi0, xi2] の共分散を推定し、peakfit を dv+cov に更新する入口を作る。

前提：
- RascalC は C++ バイナリを含むため **WSL（Linux）で実行**する。
  - 依存（GSL/pkg-config）が無い環境では `scripts/cosmology/wsl_install_rascalc.sh` を先に実行する。

入力：
- `output/private/cosmology/cosmology_bao_xi_from_catalogs_*_metrics.json`（1ケース分を指定）

出力（固定名）：
- `output/private/cosmology/cosmology_bao_xi_from_catalogs_{tag}__<output_suffix>.npz`（default: `rascalc_cov`）
  - `s`（xi の s bin center; xi出力と一致）
  - `cov`（shape=(2*n_s, 2*n_s); dv順序は y=[xi0, xi2]）
- `output/private/cosmology/cosmology_bao_xi_from_catalogs_{tag}__<output_suffix>_metrics.json`

備考：
- RascalC の covariance は内部の「RascalC convention（r-major × ℓ）」で出てくるため、本スクリプトで
  peakfit と同じ dv 順序（xi0 block → xi2 block）へ並び替えて保存する。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cosmology import cosmology_bao_xi_from_catalogs as _xi  # noqa: E402
from scripts.summary import worklog  # noqa: E402


# 関数: `_infer_cap_label` の入出力契約と処理意図を定義する。
def _infer_cap_label(path: Path, *, fallback: str) -> str:
    s = path.name.lower()
    # 条件分岐: `"ngc" in s or "north" in s` を満たす経路を評価する。
    if "ngc" in s or "north" in s:
        return "north"

    # 条件分岐: `"sgc" in s or "south" in s` を満たす経路を評価する。

    if "sgc" in s or "south" in s:
        return "south"

    return fallback


# 関数: `_load_cap_inputs` の入出力契約と処理意図を定義する。

def _load_cap_inputs(
    *,
    gal_npz: Path,
    rnd_npz: Path,
    z_source: str,
    weight_scheme: str,
    z_min: float | None,
    z_max: float | None,
    dist_model: str,
    omega_m: float,
    lcdm_n_grid: int,
    lcdm_z_grid_max: float | None,
) -> Dict[str, Any]:
    gal = _xi._load_npz(gal_npz)
    rnd = _xi._load_npz(rnd_npz)

    ra_g0 = np.asarray(gal["RA"], dtype=np.float64)
    dec_g0 = np.asarray(gal["DEC"], dtype=np.float64)
    z_g0, _ = _xi._select_redshift(gal, z_source=str(z_source))
    w_g0, _ = _xi._weights_galaxy(gal, scheme=str(weight_scheme))

    ra_r0 = np.asarray(rnd["RA"], dtype=np.float64)
    dec_r0 = np.asarray(rnd["DEC"], dtype=np.float64)
    z_r0, _ = _xi._select_redshift(rnd, z_source=str(z_source))
    w_r0, _ = _xi._weights_random(rnd, scheme=str(weight_scheme))

    mg = np.isfinite(z_g0) & (z_g0 > 0.0) & np.isfinite(w_g0)
    mr = np.isfinite(z_r0) & (z_r0 > 0.0) & np.isfinite(w_r0)
    # 条件分岐: `z_min is not None` を満たす経路を評価する。
    if z_min is not None:
        mg = mg & (z_g0 >= float(z_min))
        mr = mr & (z_r0 >= float(z_min))

    # 条件分岐: `z_max is not None` を満たす経路を評価する。

    if z_max is not None:
        mg = mg & (z_g0 < float(z_max))
        mr = mr & (z_r0 < float(z_max))

    ra_g = ra_g0[mg]
    dec_g = dec_g0[mg]
    z_g = np.asarray(z_g0[mg], dtype=np.float64)
    w_g = np.asarray(w_g0[mg], dtype=np.float64)

    ra_r = ra_r0[mr]
    dec_r = dec_r0[mr]
    z_r = np.asarray(z_r0[mr], dtype=np.float64)
    w_r = np.asarray(w_r0[mr], dtype=np.float64)

    z_all = np.concatenate([z_g, z_r], axis=0).astype(np.float64, copy=False)
    d_all, dist_meta = _xi._comoving_distance_mpc_over_h(
        z_all,
        model=str(dist_model),
        lcdm_omega_m=float(omega_m),
        lcdm_n_grid=int(lcdm_n_grid),
        lcdm_z_grid_max=lcdm_z_grid_max,
    )
    d_all = np.asarray(d_all, dtype=np.float64)
    d_g = d_all[: int(z_g.size)]
    d_r = d_all[int(z_g.size) :]

    return {
        "gal": {"ra": ra_g, "dec": dec_g, "z": z_g, "d": d_g, "w": w_g},
        "rnd": {"ra": ra_r, "dec": dec_r, "z": z_r, "d": d_r, "w": w_r},
        "dist_meta": dist_meta,
    }


# 関数: `_permute_cov_rascalc_to_peakfit` の入出力契約と処理意図を定義する。

def _permute_cov_rascalc_to_peakfit(cov: np.ndarray, *, n_r: int, max_l: int) -> np.ndarray:
    # 条件分岐: `int(max_l) % 2 != 0` を満たす経路を評価する。
    if int(max_l) % 2 != 0:
        raise ValueError("max_l must be even")

    n_l = int(max_l) // 2 + 1
    cov = np.asarray(cov, dtype=float).reshape(int(n_r) * n_l, int(n_r) * n_l)

    # RascalC convention: [r0:l0,l2,..., r1:l0,l2,...]
    # Peakfit dv convention: [l0:r0.., l2:r0..]
    perm = np.empty(int(n_r) * n_l, dtype=np.int32)
    k = 0
    for l_idx in range(n_l):
        for r in range(int(n_r)):
            perm[k] = int(r) * n_l + int(l_idx)
            k += 1

    cov2 = cov[np.ix_(perm, perm)]
    cov2 = 0.5 * (cov2 + cov2.T)
    return cov2


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cosmology: RascalC covariance for catalog-based xi multipoles (xi0/xi2).")
    ap.add_argument(
        "--xi-metrics-json",
        required=True,
        help="Path to a single cosmology_bao_xi_from_catalogs_*_metrics.json (the case definition).",
    )
    ap.add_argument(
        "--output-suffix",
        type=str,
        default="rascalc_cov",
        help="Output suffix for covariance files (default: rascalc_cov => ...__rascalc_cov.npz and ...__rascalc_cov_metrics.json).",
    )
    ap.add_argument("--threads", type=int, default=24, help="RascalC OpenMP threads (default: 24)")
    ap.add_argument("--max-l", type=int, default=2, help="Max even multipole ell for RascalC (default: 2 => xi0/xi2)")
    ap.add_argument("--N2", type=int, default=5, help="RascalC N2 (default: 5)")
    ap.add_argument("--N3", type=int, default=10, help="RascalC N3 (default: 10)")
    ap.add_argument("--N4", type=int, default=20, help="RascalC N4 (default: 20)")
    ap.add_argument("--n-loops", type=int, default=480, help="RascalC n_loops (default: 480; must be divisible by threads)")
    ap.add_argument("--loops-per-sample", type=int, default=48, help="RascalC loops_per_sample (default: 48; must divide n_loops)")
    ap.add_argument(
        "--randoms-subset",
        type=int,
        default=200_000,
        help="Random subset size for RascalC sampling geometry (default: 200000; <=n_random).",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    ap.add_argument(
        "--xi-refinement-iterations",
        type=int,
        default=0,
        help="RascalC xi_refinement_iterations (cf_loops). Use 0 to disable refinement (default: 0).",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    # 条件分岐: `os.name == "nt"` を満たす経路を評価する。
    if os.name == "nt":
        raise SystemExit("RascalC is not supported on Windows; run this under WSL (Ubuntu-24.04) as per AGENTS.md.")

    metrics_path = Path(str(args.xi_metrics_json)).resolve()
    # 条件分岐: `not metrics_path.exists()` を満たす経路を評価する。
    if not metrics_path.exists():
        raise SystemExit(f"metrics json not found: {metrics_path}")

    output_suffix = str(args.output_suffix).strip()
    # 条件分岐: `not output_suffix` を満たす経路を評価する。
    if not output_suffix:
        raise SystemExit("--output-suffix must be non-empty")

    # 条件分岐: `any(sep in output_suffix for sep in ("/", "\\", ":", "\0"))` を満たす経路を評価する。

    if any(sep in output_suffix for sep in ("/", "\\", ":", "\0")):
        raise SystemExit(f"invalid --output-suffix (contains path separator or invalid char): {output_suffix!r}")

    # 条件分岐: `int(args.max_l) % 2 != 0 or int(args.max_l) < 0` を満たす経路を評価する。

    if int(args.max_l) % 2 != 0 or int(args.max_l) < 0:
        raise SystemExit("--max-l must be an even integer >= 0")

    # 条件分岐: `int(args.threads) <= 0` を満たす経路を評価する。

    if int(args.threads) <= 0:
        raise SystemExit("--threads must be >= 1")

    # 条件分岐: `int(args.n_loops) <= 0 or int(args.n_loops) % int(args.threads) != 0` を満たす経路を評価する。

    if int(args.n_loops) <= 0 or int(args.n_loops) % int(args.threads) != 0:
        raise SystemExit("--n-loops must be divisible by --threads")

    # 条件分岐: `int(args.loops_per_sample) <= 0 or int(args.n_loops) % int(args.loops_per_sam...` を満たす経路を評価する。

    if int(args.loops_per_sample) <= 0 or int(args.n_loops) % int(args.loops_per_sample) != 0:
        raise SystemExit("--loops-per-sample must divide --n-loops")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    xi_npz_path = Path(str(metrics["outputs"]["npz"])).resolve()
    # 条件分岐: `not xi_npz_path.exists()` を満たす経路を評価する。
    if not xi_npz_path.exists():
        raise SystemExit(f"xi npz not found: {xi_npz_path}")

    tag = xi_npz_path.stem

    params = metrics.get("params", {})
    inputs = metrics.get("inputs", {})

    gal_npz_raw = inputs.get("galaxy_npz")
    rnd_npz_raw = inputs.get("random_npz")
    gal_npz_list = list(gal_npz_raw) if isinstance(gal_npz_raw, list) else [str(gal_npz_raw)]
    rnd_npz_list = list(rnd_npz_raw) if isinstance(rnd_npz_raw, list) else [str(rnd_npz_raw)]
    gal_npzs = [Path(str(p)).resolve() for p in gal_npz_list]
    rnd_npzs = [Path(str(p)).resolve() for p in rnd_npz_list]
    # 条件分岐: `len(gal_npzs) != len(rnd_npzs)` を満たす経路を評価する。
    if len(gal_npzs) != len(rnd_npzs):
        raise SystemExit(f"galaxy_npz and random_npz count mismatch: {len(gal_npzs)} vs {len(rnd_npzs)}")

    # 条件分岐: `not gal_npzs` を満たす経路を評価する。

    if not gal_npzs:
        raise SystemExit("inputs.galaxy_npz is empty")

    for p in gal_npzs:
        # 条件分岐: `not p.exists()` を満たす経路を評価する。
        if not p.exists():
            raise SystemExit(f"galaxy npz not found: {p}")

    for p in rnd_npzs:
        # 条件分岐: `not p.exists()` を満たす経路を評価する。
        if not p.exists():
            raise SystemExit(f"random npz not found: {p}")

    z_source = str(params.get("z_source", "obs"))
    weight_scheme = str(params.get("weight_scheme", "desi_default"))
    z_cut = params.get("z_cut", {}) or {}
    z_min = float(z_cut["z_min"]) if ("z_min" in z_cut and z_cut["z_min"] is not None) else None
    z_max = float(z_cut["z_max"]) if ("z_max" in z_cut and z_cut["z_max"] is not None) else None

    dist_model = str(params.get("distance_model", "lcdm"))
    omega_m = float(params.get("lcdm_omega_m", 0.315))
    lcdm_n_grid = int(params.get("lcdm_n_grid", 6000))
    lcdm_z_grid_max = params.get("lcdm_z_grid_max", None)
    lcdm_z_grid_max = None if lcdm_z_grid_max is None else float(lcdm_z_grid_max)

    s_bins = params.get("s_bins", {}) or {}
    s_min = float(s_bins.get("min", 30.0))
    s_max = float(s_bins.get("max", 150.0))
    s_step = float(s_bins.get("step", 5.0))
    # 条件分岐: `not (math.isfinite(s_min) and math.isfinite(s_max) and math.isfinite(s_step)...` を満たす経路を評価する。
    if not (math.isfinite(s_min) and math.isfinite(s_max) and math.isfinite(s_step) and s_step > 0 and s_max > s_min):
        raise SystemExit(f"invalid s_bins in metrics: {s_bins}")

    mu_bins = params.get("mu_bins", {}) or {}
    mu_max = float(mu_bins.get("mu_max", 1.0))
    nmu = int(mu_bins.get("nmu", 120))
    # 条件分岐: `not (math.isfinite(mu_max) and mu_max > 0.0 and nmu >= 1)` を満たす経路を評価する。
    if not (math.isfinite(mu_max) and mu_max > 0.0 and nmu >= 1):
        raise SystemExit(f"invalid mu_bins in metrics: {mu_bins}")

    # Load xi output bins (for output file convention).

    with np.load(xi_npz_path) as z:
        s_xi = np.asarray(z["s"], dtype=float).reshape(-1)

    n_s = int(s_xi.size)

    out_cov_npz = xi_npz_path.with_name(f"{xi_npz_path.stem}__{output_suffix}.npz")
    out_cov_json = xi_npz_path.with_name(f"{xi_npz_path.stem}__{output_suffix}_metrics.json")

    base_dir = _ROOT / "output" / "private" / "cosmology" / "rascalc" / tag
    run_dir = base_dir / output_suffix
    tmp_dir = run_dir / "tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Compute positions / weights from raw inputs (z-filtered) to match the xi case.
    combine_caps = (params.get("estimator_spec", {}) or {}).get("combine_caps", {}) or {}
    rnd_rescale = (combine_caps.get("random_weight_rescale", {}) or {}).get("runtime_by_cap", []) or []
    scale_by_cap = {str(d.get("cap")): float(d.get("scale_random_w")) for d in rnd_rescale if d.get("cap") is not None}

    cap_g: List[Dict[str, np.ndarray]] = []
    cap_r: List[Dict[str, np.ndarray]] = []
    for gal_npz, rnd_npz in zip(gal_npzs, rnd_npzs):
        cap_label = _infer_cap_label(gal_npz, fallback="combined")
        cap_inputs = _load_cap_inputs(
            gal_npz=gal_npz,
            rnd_npz=rnd_npz,
            z_source=z_source,
            weight_scheme=weight_scheme,
            z_min=z_min,
            z_max=z_max,
            dist_model=dist_model,
            omega_m=omega_m,
            lcdm_n_grid=lcdm_n_grid,
            lcdm_z_grid_max=lcdm_z_grid_max,
        )
        g = cap_inputs["gal"]
        r = cap_inputs["rnd"]
        w_r_cap = np.asarray(r["w"], dtype=np.float64)
        # 条件分岐: `cap_label in scale_by_cap` を満たす経路を評価する。
        if cap_label in scale_by_cap:
            w_r_cap = w_r_cap * float(scale_by_cap[cap_label])

        cap_g.append(
            {
                "ra": np.asarray(g["ra"], dtype=np.float64),
                "dec": np.asarray(g["dec"], dtype=np.float64),
                "d": np.asarray(g["d"], dtype=np.float64),
                "w": np.asarray(g["w"], dtype=np.float64),
            }
        )
        cap_r.append(
            {
                "ra": np.asarray(r["ra"], dtype=np.float64),
                "dec": np.asarray(r["dec"], dtype=np.float64),
                "d": np.asarray(r["d"], dtype=np.float64),
                "w": w_r_cap,
            }
        )

    ra_g = np.concatenate([c["ra"] for c in cap_g], axis=0)
    dec_g = np.concatenate([c["dec"] for c in cap_g], axis=0)
    d_g = np.concatenate([c["d"] for c in cap_g], axis=0)
    w_g = np.concatenate([c["w"] for c in cap_g], axis=0)

    ra_r = np.concatenate([c["ra"] for c in cap_r], axis=0)
    dec_r = np.concatenate([c["dec"] for c in cap_r], axis=0)
    d_r = np.concatenate([c["d"] for c in cap_r], axis=0)
    w_r = np.concatenate([c["w"] for c in cap_r], axis=0)

    pos_g = _xi._radec_to_xyz_mpc_over_h(ra_g, dec_g, d_g)
    pos_r = _xi._radec_to_xyz_mpc_over_h(ra_r, dec_r, d_r)

    # pycorr counts cache (small; safe to keep for reproducibility).
    try:
        from pycorr import TwoPointCorrelationFunction
    except Exception as e:  # pragma: no cover
        raise SystemExit("pycorr is required (install it in .venv_wsl).") from e

    # Match the binning convention used by cosmology_bao_xi_from_catalogs.py:
    # s_bins are edges with inclusive max.

    n_bins_exact = (float(s_max) - float(s_min)) / float(s_step)
    n_bins = int(np.rint(n_bins_exact))
    # 条件分岐: `not np.allclose(n_bins_exact, float(n_bins), rtol=0.0, atol=1e-9) or n_bins < 1` を満たす経路を評価する。
    if not np.allclose(n_bins_exact, float(n_bins), rtol=0.0, atol=1e-9) or n_bins < 1:
        raise SystemExit(f"s_bins are not an integer grid: (max-min)/step={n_bins_exact} from {s_bins}")

    s_edges = (float(s_min) + float(s_step) * np.arange(int(n_bins) + 1, dtype=float)).astype(float, copy=False)
    mu_edges = np.linspace(-float(mu_max), float(mu_max), int(2 * int(nmu)) + 1, dtype=float)

    counts_path = base_dir / "pycorr_allcounts_cov.npy"
    # 条件分岐: `counts_path.exists()` を満たす経路を評価する。
    if counts_path.exists():
        cf_cov = TwoPointCorrelationFunction.load(str(counts_path))
    else:
        t0 = time.time()
        cf_unwrapped = TwoPointCorrelationFunction(
            "smu",
            (s_edges, mu_edges),
            data_positions1=np.asarray(pos_g, dtype=np.float64),
            randoms_positions1=np.asarray(pos_r, dtype=np.float64),
            data_weights1=np.asarray(w_g, dtype=np.float64),
            randoms_weights1=np.asarray(w_r, dtype=np.float64),
            position_type="pos",
            engine="corrfunc",
            nthreads=int(args.threads),
            compute_sepsavg=False,
            los="midpoint",
        )
        cf_cov = cf_unwrapped.wrap()
        cf_cov.save(str(counts_path))
        dt = time.time() - t0
        print(f"[info] computed pycorr counts in {dt:.1f} s -> {counts_path}")

    # Sanity: ensure s bins match the xi output bins (within tolerance).

    s_cov = np.asarray(cf_cov.sepavg(axis=0), dtype=float).reshape(-1)
    # 条件分岐: `s_cov.size != s_xi.size or not np.allclose(s_cov, s_xi, rtol=0.0, atol=1e-9)` を満たす経路を評価する。
    if s_cov.size != s_xi.size or not np.allclose(s_cov, s_xi, rtol=0.0, atol=1e-9):
        raise SystemExit(
            "RascalC cov binning mismatch vs xi output. "
            f"s_xi(n={s_xi.size})!=s_cov(n={s_cov.size}). "
            "Fix by re-running xi-from-catalogs with the same s_bins used here."
        )

    # Use the wrapped pycorr estimator directly as the xi(s,mu) table.
    # (Passing a raw grid tuple triggers a known RascalC interface pitfall where bin edges are built by
    # elementwise addition instead of concatenation.)

    xi_table = cf_cov

    try:
        from RascalC import run_cov
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "RascalC is required. If not installed, run:\n"
            "  bash scripts/cosmology/wsl_install_rascalc.sh\n"
            "under WSL."
        ) from e

    # Random subset for geometry sampling (avoid ordering bias).

    n_rnd = int(pos_r.shape[0])
    n_sub = int(min(max(1, int(args.randoms_subset)), n_rnd))
    rng = np.random.default_rng(int(args.seed))
    # 条件分岐: `n_sub < n_rnd` を満たす経路を評価する。
    if n_sub < n_rnd:
        idx = rng.choice(n_rnd, size=n_sub, replace=False)
        pos_r_sub = pos_r[idx]
        w_r_sub = w_r[idx]
    else:
        idx = np.arange(n_rnd, dtype=np.int64)
        pos_r_sub = pos_r
        w_r_sub = w_r

    # Run RascalC.

    res = run_cov(
        mode="legendre_projected",
        max_l=int(args.max_l),
        boxsize=None,
        nthread=int(args.threads),
        N2=int(args.N2),
        N3=int(args.N3),
        N4=int(args.N4),
        n_loops=int(args.n_loops),
        loops_per_sample=int(args.loops_per_sample),
        out_dir=str(run_dir),
        tmp_dir=str(tmp_dir),
        randoms_positions1=np.asarray(pos_r_sub, dtype=np.float64),
        randoms_weights1=np.asarray(w_r_sub, dtype=np.float64),
        pycorr_allcounts_11=cf_cov,
        xi_table_11=xi_table,
        position_type="pos",
        normalize_wcounts=True,
        xi_refinement_iterations=int(args.xi_refinement_iterations),
        seed=int(args.seed),
        verbose=False,
    )

    cov_ras = np.asarray(res["full_theory_covariance"], dtype=float)
    cov = _permute_cov_rascalc_to_peakfit(cov_ras, n_r=n_s, max_l=int(args.max_l))

    np.savez_compressed(
        out_cov_npz,
        s=s_xi,
        cov=cov,
        shot_noise_rescaling=np.asarray(res.get("shot_noise_rescaling", np.nan), dtype=float),
        N_eff=np.asarray(res.get("N_eff", np.nan), dtype=float),
    )

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "4.5B.21.4.4.6.3",
        "inputs": {
            "xi_metrics_json": str(metrics_path),
            "xi_npz": str(xi_npz_path),
            "galaxy_npz": [str(p) for p in gal_npzs] if len(gal_npzs) > 1 else str(gal_npzs[0]),
            "random_npz": [str(p) for p in rnd_npzs] if len(rnd_npzs) > 1 else str(rnd_npzs[0]),
            "pycorr_allcounts_cov_npy": str(counts_path),
        },
        "params": {
            "distance_model": dist_model,
            "z_source": z_source,
            "weight_scheme": weight_scheme,
            "z_min": z_min,
            "z_max": z_max,
            "s_bins": {"min": s_min, "max": s_max, "step": s_step},
            "mu_bins": {"mu_max": float(mu_max), "nmu": int(nmu)},
            "combine_caps_random_weight_rescale": {"runtime_by_cap": rnd_rescale},
            "rascalc": {
                "mode": "legendre_projected",
                "max_l": int(args.max_l),
                "threads": int(args.threads),
                "N2": int(args.N2),
                "N3": int(args.N3),
                "N4": int(args.N4),
                "n_loops": int(args.n_loops),
                "loops_per_sample": int(args.loops_per_sample),
                "randoms_subset": int(n_sub),
                "seed": int(args.seed),
                "xi_refinement_iterations": int(args.xi_refinement_iterations),
                "out_dir": str(run_dir),
                "base_dir": str(base_dir),
            },
        },
        "outputs": {
            "cov_npz": str(out_cov_npz),
            "metrics_json": str(out_cov_json),
            "rascalc_rescaled_cov_npz": str(Path(str(res.get("path", ""))).resolve()) if res.get("path") else "",
        },
        "summary": {
            "dv_dim": int(cov.shape[0]),
            "shot_noise_rescaling": float(np.asarray(res.get("shot_noise_rescaling", np.nan), dtype=float)),
            "N_eff": float(np.asarray(res.get("N_eff", np.nan), dtype=float)),
        },
    }
    out_cov_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_bao_xi_rascalc_cov_from_catalogs",
                "argv": sys.argv,
                "inputs": {"xi_metrics_json": metrics_path, "xi_npz": xi_npz_path},
                "outputs": {"cov_npz": out_cov_npz, "metrics_json": out_cov_json},
                "metrics": {"tag": tag, "threads": int(args.threads), "max_l": int(args.max_l), "n_loops": int(args.n_loops)},
            }
        )
    except Exception:
        pass

    print(f"[ok] cov npz : {out_cov_npz}")
    print(f"[ok] metrics : {out_cov_json}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
