#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_catalog_peakfit.py

Phase 16（宇宙論）/ Step 16.4（BAO一次情報：銀河+random）:
銀河+random から再計算した ξℓ（ℓ=0,2）に対して、最小モデル（smooth+peak）で
AP/warping パラメータ（α, ε）をグリッドサーチし、幾何整合（特に ε）を定量化する。

位置づけ：
- Phase A（スクリーニング）：正確な共分散やreconstruction自前化の前段として、
  「幾何が合う/合わない」の指標を固定するための簡易ピークfit。
- 誤差モデルは選択可能：
  - Phase A: `diag`（paircount由来の対角近似；スクリーニング）
  - Phase B: `ross`（Ross 2016 公開 cov: mono+quad の full covariance で χ² を評価）

入力：
- `scripts/cosmology/cosmology_bao_xi_from_catalogs.py` の出力
  - output/private/cosmology/cosmology_bao_xi_from_catalogs_*_metrics.json
  - 対応する .npz（xi0/xi2 + counts/xi_mu 等を含む）

出力（固定名）：
- output/private/cosmology/cosmology_bao_catalog_peakfit_{sample}_{caps}{_zbinonly}.png
- output/private/cosmology/cosmology_bao_catalog_peakfit_{sample}_{caps}{_zbinonly}_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402

# Reuse the proven smooth+peak + AP warp machinery from the published-multipole fit.
from scripts.cosmology import cosmology_bao_xi_multipole_peakfit as _peakfit  # noqa: E402


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


def _resolve_wsl_windows_path(p: str) -> Path:
    """
    Resolve a path stored by WSL runs (e.g. /mnt/c/...) into a Windows path.
    """
    s = str(p).strip()
    if not s:
        return Path(s)
    # Only rewrite /mnt/<drive>/... when running on Windows.
    # When running inside WSL/Linux, keep the POSIX path as-is.
    if os.name != "nt":
        return Path(s)
    if s.startswith("/mnt/") and len(s) >= 7 and s[5].isalpha() and s[6] == "/":
        drive = s[5].upper()
        rest = s[7:].replace("/", "\\")
        return Path(f"{drive}:\\{rest}")
    return Path(s)


def _iter_metrics_files() -> Iterable[Path]:
    out_dir = _ROOT / "output" / "private" / "cosmology"
    yield from sorted(out_dir.glob("cosmology_bao_xi_from_catalogs_*_metrics.json"))


def _sanitize_out_tag(tag: str) -> str:
    t = str(tag).strip()
    if not t:
        return ""
    out: list[str] = []
    for ch in t:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("._-")
    return s[:80]


def _estimator_spec_hash_from_params(params: Dict[str, Any]) -> str:
    """
    Stable-ish hash to guard against mixing estimator definitions across inputs.

    Newer xi-from-catalogs outputs provide `estimator_spec_hash`. For legacy
    outputs, compute a best-effort hash from common params so that `run_all.py`
    remains usable without immediately re-running Corrfunc.
    """
    h = params.get("estimator_spec_hash", None)
    if isinstance(h, str) and h.strip():
        return h.strip()

    s_bins = params.get("s_bins", {}) or {}
    mu_bins = params.get("mu_bins", {}) or {}
    spec = {
        "data_selection": {
            "random_kind": str(params.get("random_kind") or ""),
            "match_sectors": str(params.get("match_sectors") or ""),
            "sector_key": str(params.get("sector_key") or ""),
        },
        "bins": {
            "s": {
                "min": float(s_bins.get("min", float("nan"))),
                "max": float(s_bins.get("max", float("nan"))),
                "step": float(s_bins.get("step", float("nan"))),
            },
            "mu": {"nmu": int(mu_bins.get("nmu", 0) or 0), "mu_max": float(mu_bins.get("mu_max", float("nan")))},
        },
        "paircounts": {
            "backend": "Corrfunc.DDsmu_mocks",
            "weight_type": "pair_product",
            "autocorr_convention": "ordered_pairs (i!=j)",
            "xi_multipoles_discretization": "riemann_midpoint (uniform mu bins)",
        },
        "combine_caps": {
            "caps": str(params.get("caps") or ""),
            "random_weight_rescale_policy": "rescale random weights per cap to match global (sum_w_gal/sum_w_rnd) before aggregating LS terms",
        },
    }
    import hashlib

    blob = json.dumps(spec, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _parse_zbin_id(z_bin: str) -> Optional[int]:
    zb = str(z_bin).strip().lower()
    if not zb or zb == "none":
        return None
    if zb.startswith("b") and zb[1:].isdigit():
        return int(zb[1:])
    if zb.startswith("zbin") and zb[4:].isdigit():
        return int(zb[4:])
    if zb.isdigit():
        return int(zb)
    return None


def _ross_cov_inv_for_s(
    *,
    ross_dir: Path,
    zbin: int,
    bincent: int,
    s: np.ndarray,
    s_step: float,
    rcond: float = 1e-12,
) -> Tuple[np.ndarray, Path]:
    """
    Build inverse covariance for dv=[xi0(s), xi2(s)] by slicing Ross 2016 full covariance (mono+quad).

    Assumptions:
    - Ross covariance grid is uniform with bs=5 Mpc/h and bin-centering offset `bincent` in {0..4}:
        r = i*bs + bs/2 + bincent
    - Caller supplies `s` already sliced to the fit range and aligned to that grid.
    """
    ross_dir = Path(ross_dir)
    if not (ross_dir.exists() and ross_dir.is_dir()):
        raise FileNotFoundError(f"ross_dir not found: {ross_dir}")
    if int(bincent) < 0 or int(bincent) > 4:
        raise ValueError("--ross-bincent must be in 0..4")
    if int(zbin) not in (1, 2, 3):
        raise ValueError(f"zbin must be 1..3 (got {zbin})")

    cov_path = ross_dir / f"Ross_2016_COMBINEDDR12_zbin{int(zbin)}_covariance_monoquad_post_recon_bincent{int(bincent)}.dat"
    if not cov_path.exists():
        raise FileNotFoundError(f"missing Ross covariance: {cov_path}")

    cov_full = _peakfit._read_cov(cov_path)
    if cov_full.shape[0] != cov_full.shape[1] or (cov_full.shape[0] % 2) != 0:
        raise ValueError(f"invalid Ross covariance shape: {cov_full.shape} from {cov_path}")

    n_all = int(cov_full.shape[0] // 2)
    s = np.asarray(s, dtype=float).reshape(-1)
    if s.size == 0:
        raise ValueError("empty s array for Ross covariance selection")

    bs = float(s_step)
    base = 0.5 * bs + float(bincent)

    idx_mono: List[int] = []
    for sv in s:
        t = (float(sv) - base) / bs
        it = int(round(t))
        if abs(float(t) - float(it)) > 1e-6:
            raise ValueError(f"s value not aligned to Ross bins: s={sv} (bincent={bincent}, step={bs})")
        if not (0 <= it < n_all):
            raise ValueError(f"s index out of bounds for Ross covariance: s={sv} -> i={it} (n_all={n_all})")
        idx_mono.append(it)
    if len(set(idx_mono)) != len(idx_mono):
        raise ValueError("duplicate s bins detected while mapping to Ross covariance indices")

    idx_dv = idx_mono + [n_all + i for i in idx_mono]
    cov = cov_full[np.ix_(idx_dv, idx_dv)]
    cov_inv = np.linalg.pinv(cov, rcond=float(rcond))
    cov_inv = 0.5 * (cov_inv + cov_inv.T)
    return cov_inv, cov_path


def _satpathy_cov_inv_for_s(
    *,
    satpathy_dir: Path,
    zbin: int,
    s: np.ndarray,
    s_step: float,
    rcond: float = 1e-12,
) -> Tuple[np.ndarray, Path]:
    """
    Build inverse covariance for dv=[xi0(s), xi2(s)] using Satpathy et al. (2016) pre-recon full covariance.

    Notes
    -----
    - Satpathy provides a 48x48 (mono+quad) covariance per z-bin for COMBINEDDR12.
    - Our catalog-based ξℓ uses Corrfunc radial bins with edges [30,35,...,150] so bin centers are
      32.5, 37.5, ..., 147.5 (step 5). We assume Satpathy cov uses the same bin centers.
    """
    if not (math.isfinite(float(s_step)) and abs(float(s_step) - 5.0) < 1e-9):
        raise ValueError(f"Satpathy cov requires s_step=5.0 (got {s_step})")

    cov_name = (
        f"Satpathy_2016_COMBINEDDR12_Bin{int(zbin)}_Covariance_pre_recon.txt"
        if int(zbin) in (1, 2)
        else f"Satpathy_2016_COMBINEDDR12_Bin{int(zbin)}_CovarianceMatrix_pre_recon.txt"
    )
    cov_path = satpathy_dir / cov_name
    if not cov_path.exists():
        raise FileNotFoundError(f"missing Satpathy covariance: {cov_path}")

    cov_full = _peakfit._read_satpathy_cov(cov_path)
    if cov_full.shape[0] != cov_full.shape[1] or (cov_full.shape[0] % 2) != 0:
        raise ValueError(f"invalid Satpathy covariance shape: {cov_full.shape} from {cov_path}")
    n_all = int(cov_full.shape[0] // 2)

    s = np.asarray(s, dtype=float).reshape(-1)
    if s.size == 0:
        raise ValueError("empty s array for Satpathy covariance selection")

    # Expected Satpathy bin-centers for 5 Mpc/h bins with edges [30,35,...,150].
    bs = float(s_step)
    s0 = 30.0 + 0.5 * bs  # 32.5
    idx: List[int] = []
    for sv in s:
        it = int(round((float(sv) - s0) / bs))
        expected = s0 + bs * float(it)
        if abs(float(sv) - expected) > 1e-6:
            raise ValueError(f"s not aligned to Satpathy cov grid: s={sv} (expected {expected})")
        if it < 0 or it >= n_all:
            raise ValueError(f"s index out of bounds for Satpathy covariance: s={sv} -> i={it} (n_all={n_all})")
        idx.append(it)
    if len(set(idx)) != len(idx):
        raise ValueError("duplicate s bins detected while mapping to Satpathy covariance indices")

    idx_dv = idx + [it + n_all for it in idx]
    cov = cov_full[np.ix_(idx_dv, idx_dv)]
    cov_inv = np.linalg.pinv(cov, rcond=float(rcond))
    cov_inv = 0.5 * (cov_inv + cov_inv.T)
    return cov_inv, cov_path


@dataclass(frozen=True)
class CatalogCase:
    sample: str
    caps: str
    dist: str
    recon_mode: str
    z_bin: str
    z_eff: float
    z_source: str
    los: str
    weight_scheme: str
    estimator_spec_hash: str
    lcdm_omega_m: float
    lcdm_n_grid: int
    lcdm_z_grid_max: float
    coordinate_spec: Dict[str, Any]
    npz_path: Path
    metrics_path: Path

    @property
    def label(self) -> str:
        ztag = f", {self.z_bin}" if self.z_bin != "none" else ""
        return f"{self.sample}/{self.caps}/{self.dist}{ztag}"


def _load_cases(
    *,
    sample: str,
    caps: str,
    dists: List[str],
    require_zbin: bool,
    out_tag: str,
) -> List[CatalogCase]:
    cases: List[CatalogCase] = []
    for path in _iter_metrics_files():
        try:
            m = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        params = m.get("params", {}) or {}
        if str(params.get("sample", "")) != str(sample):
            continue
        if str(params.get("caps", "")) != str(caps):
            continue
        dist = str(params.get("distance_model", ""))
        if dist not in set(dists):
            continue
        z_source = str(params.get("z_source", ""))
        los = str(params.get("los", ""))
        weight_scheme = str(params.get("weight_scheme", "boss_default") or "boss_default")
        estimator_spec_hash = _estimator_spec_hash_from_params(params)
        lcdm_omega_m = float(params.get("lcdm_omega_m", float("nan")))
        lcdm_n_grid = int(params.get("lcdm_n_grid", 0) or 0)
        lcdm_z_grid_max = float(params.get("lcdm_z_grid_max", float("nan")))
        recon_mode = str(((params.get("recon", {}) or {}).get("mode", "") or "")).strip().lower() or "none"
        coordinate_spec = params.get("coordinate_spec", {}) or {}
        z_cut = (params.get("z_cut", {}) or {}).get("bin", "none")
        z_bin = str(z_cut) if z_cut is not None else "none"
        if require_zbin and (z_bin == "none"):
            continue
        if (not require_zbin) and (z_bin != "none"):
            # Avoid mixing z-binned files when the caller wants a single effective-z case.
            continue

        tag_in = params.get("out_tag", None)
        if out_tag == "none":
            if tag_in not in (None, "", "null"):
                continue
        elif out_tag == "any":
            pass
        else:
            if str(tag_in) != str(out_tag):
                continue

        z_eff = float((m.get("derived", {}) or {}).get("z_eff_gal_weighted", float("nan")))
        outputs = m.get("outputs", {}) or {}
        npz_raw = outputs.get("npz", "")
        npz_path = _resolve_wsl_windows_path(str(npz_raw))
        if not npz_path.is_absolute():
            npz_path = (_ROOT / npz_path).resolve()
        if not npz_path.exists():
            continue
        cases.append(
            CatalogCase(
                sample=str(sample),
                caps=str(caps),
                dist=str(dist),
                recon_mode=recon_mode,
                z_bin=str(z_bin),
                z_eff=z_eff,
                z_source=z_source,
                los=los,
                weight_scheme=weight_scheme,
                estimator_spec_hash=str(estimator_spec_hash),
                lcdm_omega_m=lcdm_omega_m,
                lcdm_n_grid=lcdm_n_grid,
                lcdm_z_grid_max=lcdm_z_grid_max,
                coordinate_spec=dict(coordinate_spec),
                npz_path=npz_path,
                metrics_path=path,
            )
        )

    # Sort within each dist by z_eff, then z_bin.
    cases.sort(key=lambda c: (c.dist, c.z_eff, c.z_bin))
    return cases


def _ensure_2d(x: np.ndarray, *, n0: int, n1: int, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.zeros((n0, n1), dtype=float)
    if x.ndim == 2:
        if x.shape != (n0, n1):
            raise ValueError(f"{name} shape mismatch: {x.shape} vs ({n0},{n1})")
        return x
    if x.ndim == 1:
        if x.size != n0 * n1:
            raise ValueError(f"{name} size mismatch: {x.size} vs {n0*n1}")
        return x.reshape(n0, n1)
    raise ValueError(f"{name} invalid ndim: {x.ndim}")


def _p2(mu: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    return 0.5 * (3.0 * mu * mu - 1.0)


def _approx_var_from_paircounts(
    *,
    s: np.ndarray,
    mu_edges: np.ndarray,
    dd_w: np.ndarray,
    dr_w: np.ndarray,
    rr_w: np.ndarray,
    var_source: str,
    min_pairs: float,
    quad_weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagonal variance estimate for (xi0, xi2) using a crude paircount-based model.

    Assumptions (screening only):
    - ξ(s,μ) bins are independent.
    - Var[ξ(s,μ)] ∝ 1 / N_pairs(s,μ), where N_pairs is approximated by DD or RR.
    - Propagate to multipoles via μ integration.
    """
    mu_edges = np.asarray(mu_edges, dtype=float).reshape(-1)
    if mu_edges.size < 2:
        # Fallback: flat unit variance.
        n = int(np.asarray(s, dtype=float).size)
        return np.full(n, 1.0, dtype=float), np.full(n, 1.0, dtype=float)
    dmu = np.diff(mu_edges)
    if np.any(dmu <= 0):
        raise ValueError("mu_edges must be strictly increasing")
    mu_mid = 0.5 * (mu_edges[:-1] + mu_edges[1:])

    n0 = int(np.asarray(s, dtype=float).size)
    n1 = int(mu_mid.size)
    dd = _ensure_2d(dd_w, n0=n0, n1=n1, name="dd_w")
    dr = _ensure_2d(dr_w, n0=n0, n1=n1, name="dr_w")
    rr = _ensure_2d(rr_w, n0=n0, n1=n1, name="rr_w")

    src = str(var_source)
    if src == "auto":
        src = "dd" if np.any(dd > 0) else "rr"
    if src == "dd":
        pairs = dd
    elif src == "rr":
        pairs = rr
    elif src == "dr":
        pairs = dr
    else:
        raise ValueError("--var-source must be one of: auto, dd, dr, rr")

    pairs = np.maximum(pairs, float(min_pairs))
    var_xi_mu = 1.0 / pairs

    # xi0 = ∫_0^1 ξ dμ  (same as (1/2)∫_{-1}^{1} for even multipoles)
    var_xi0 = np.sum(var_xi_mu * (dmu[None, :] ** 2), axis=1)

    # xi2 = 5 ∫_0^1 ξ P2(μ) dμ
    p2 = _p2(mu_mid)
    var_xi2 = 25.0 * np.sum(var_xi_mu * (p2[None, :] ** 2) * (dmu[None, :] ** 2), axis=1)

    # Emphasize quadrupole if requested (screening policy).
    qw = float(quad_weight)
    if not (qw > 0.0 and math.isfinite(qw)):
        raise ValueError("--quad-weight must be > 0")
    var_xi2 = var_xi2 / (qw * qw)

    # Guard against zeros.
    var_xi0 = np.where(np.isfinite(var_xi0) & (var_xi0 > 0), var_xi0, 1.0)
    var_xi2 = np.where(np.isfinite(var_xi2) & (var_xi2 > 0), var_xi2, 1.0)
    return var_xi0, var_xi2


def _diag_cov_inv(var0: np.ndarray, var2: np.ndarray, *, eps: float = 1e-30) -> np.ndarray:
    var0 = np.asarray(var0, dtype=float).reshape(-1)
    var2 = np.asarray(var2, dtype=float).reshape(-1)
    if var0.size != var2.size:
        raise ValueError("var0/var2 size mismatch")
    inv0 = 1.0 / np.maximum(var0, float(eps))
    inv2 = 1.0 / np.maximum(var2, float(eps))
    d = np.concatenate([inv0, inv2], axis=0)
    return np.diag(d)


def _cov_shrink_to_diag(cov: np.ndarray, *, lam: float) -> np.ndarray:
    """
    Simple shrinkage towards diagonal:
      C' = (1-λ)C + λ diag(C)

    λ=0 keeps C as-is, λ=1 makes it diagonal-only.
    """
    lam = float(lam)
    if not (math.isfinite(lam) and 0.0 <= lam <= 1.0):
        raise ValueError("--cov-shrinkage must be within [0,1]")
    cov = np.asarray(cov, dtype=float)
    if lam <= 0.0:
        return cov
    d = np.diag(np.diag(cov))
    return (1.0 - lam) * cov + lam * d


def _cov_band_by_s_index(
    cov: np.ndarray,
    *,
    n_bins: int,
    bandwidth_bins: int,
    bandwidth_xi02_bins: int | None = None,
    n_components: int = 1,
) -> tuple[np.ndarray, Dict[str, float]]:
    """
    Band the full dv=[xi0(s), xi2(s)] covariance by |Δs_bin| in fit space.

    dv ordering is assumed to be component-major:
      n_components=1:
        [xi0 bin0..bin{n-1}, xi2 bin0..bin{n-1}]
      n_components>1:
        [comp0: xi0 bin0.., xi2 bin0.., comp1: xi0 bin0.., xi2 bin0.., ...]

    bandwidth_bins:
      - < 0 => no banding (keep full cov)
      - 0   => keep only same-bin entries (incl. xi0-xi2 at same s bin)
      - k   => keep entries with |i-j|<=k in s-bin index

    bandwidth_xi02_bins (optional):
      - if provided (>=0), use this bandwidth only for xi0-xi2 cross blocks.
      - if None or <0, reuse bandwidth_bins for all blocks.
    """
    cov = np.asarray(cov, dtype=float)
    n_bins = int(n_bins)
    n_components = int(n_components)
    bw = int(bandwidth_bins)
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")
    if n_components <= 0:
        raise ValueError("n_components must be > 0")
    expected = (2 * n_components * n_bins, 2 * n_components * n_bins)
    if cov.shape != expected:
        raise ValueError(f"invalid cov shape for banding: {cov.shape} (expected {expected})")
    if bw < 0:
        return cov, {"keep_fraction": 1.0, "kept": float(cov.size), "zeroed": 0.0}

    bw_x = bw
    if bandwidth_xi02_bins is not None:
        bw_x_raw = int(bandwidth_xi02_bins)
        if bw_x_raw >= 0:
            bw_x = bw_x_raw

    idx = np.arange(2 * n_components * n_bins, dtype=int)
    s_idx = idx % n_bins
    block = idx // n_bins  # 0..(2*n_components-1): [xi0,xi2] per component
    ell = block % 2  # 0: xi0, 1: xi2 (component ignored)
    ds = np.abs(s_idx[:, None] - s_idx[None, :])
    within = ds <= bw
    cross = ds <= bw_x
    same_ell = ell[:, None] == ell[None, :]
    mask = np.where(same_ell, within, cross)

    cov_banded = np.where(mask, cov, 0.0)
    kept = float(np.count_nonzero(mask))
    zeroed = float(mask.size - int(kept))
    keep_fraction = float(kept / float(mask.size)) if mask.size else 1.0
    return cov_banded, {"keep_fraction": keep_fraction, "kept": kept, "zeroed": zeroed}


def _cov_zero_xi02_cross(
    cov: np.ndarray,
    *,
    n_bins: int,
    n_components: int = 1,
) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    n_bins = int(n_bins)
    n_components = int(n_components)
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")
    if n_components <= 0:
        raise ValueError("n_components must be > 0")
    expected = (2 * n_components * n_bins, 2 * n_components * n_bins)
    if cov.shape != expected:
        raise ValueError(f"invalid cov shape for xi02 zeroing: {cov.shape} (expected {expected})")
    idx = np.arange(2 * n_components * n_bins, dtype=int)
    ell = (idx // n_bins) % 2  # 0: xi0, 1: xi2
    same_ell = ell[:, None] == ell[None, :]
    return np.where(same_ell, cov, 0.0)


def _ledoit_wolf_shrinkage_to_diag(y: np.ndarray) -> float:
    """
    Estimate Ledoit-Wolf shrinkage intensity λ∈[0,1] towards diagonal.

    We use the diagonal target F=diag(S), so shrinkage affects only off-diagonal terms:
      S' = (1-λ)S + λ diag(S)

    Implementation:
      λ = phi / gamma  (clipped to [0,1])
      gamma = Σ_{i≠j} S_ij^2
      phi   = (1/n) Σ_k Σ_{i≠j} (x_ki x_kj - S_ij)^2
    where x_k are centered samples and S is the sample second-moment matrix.

    Notes:
    - This is used as a practical stabilization for jackknife cov inversion.
    - Scaling of y (or jackknife vs sample covariance convention) does not affect λ.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 2:
        raise ValueError("y must be 2D (n_samples, p)")
    n, p = int(y.shape[0]), int(y.shape[1])
    if n < 2 or p < 2:
        return 1.0

    x = y - np.mean(y, axis=0, keepdims=True)
    finite_cols = np.all(np.isfinite(x), axis=0)
    x = x[:, finite_cols]
    n, p = int(x.shape[0]), int(x.shape[1])
    if p < 2:
        return 1.0

    # Second moment (not the unbiased covariance); scaling does not affect λ.
    s = (x.T @ x) / float(n)

    off = ~np.eye(p, dtype=bool)
    gamma = float(np.sum(np.square(s[off])))
    if not (math.isfinite(gamma) and gamma > 0.0):
        return 1.0

    # E[||x x^T - S||_F^2] for off-diagonal only (estimated by sample mean).
    prod = x[:, :, None] * x[:, None, :]
    diff = prod - s[None, :, :]
    diff_flat = diff.reshape(n, p * p)
    off_flat = off.reshape(p * p)
    phi = float(np.mean(np.square(diff_flat[:, off_flat])))
    if not math.isfinite(phi):
        return 1.0

    lam = float(phi / gamma)
    lam = max(0.0, min(1.0, lam))
    return lam


def _status_from_abs_sigma(abs_sigma: float | None, *, ok_max: float, mixed_max: float) -> str:
    """
    Convert an (approximately) z-scored metric to ok/mixed/ng.

    For ε screening:
      abs_sigma := |ε| / σ_ε,  where σ_ε is derived from the 1σ profile CI width.
    """
    if abs_sigma is None:
        return "info"
    a = float(abs_sigma)
    if not math.isfinite(a):
        return "info"
    if a <= float(ok_max):
        return "ok"
    if a <= float(mixed_max):
        return "mixed"
    return "ng"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cosmology: BAO catalog-based smooth+peak peakfit (alpha, eps).")
    ap.add_argument("--sample", type=str, default="cmass", help="BOSS sample (cmass/lowz/cmasslowztot)")
    ap.add_argument("--caps", type=str, default="combined", help="caps: north/south/combined (default: combined)")
    ap.add_argument("--dists", type=str, default="lcdm,pbg", help="distance models to include (comma, default: lcdm,pbg)")
    ap.add_argument("--require-zbin", action="store_true", help="Require z_cut bin (b1/b2/b3) in inputs")
    ap.add_argument(
        "--out-tag",
        type=str,
        default="none",
        help="Filter out_tag in inputs: none (default; only out_tag=null), any, or exact string",
    )
    ap.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Append suffix to output filenames (does not affect input filtering).",
    )
    ap.add_argument(
        "--cov-source",
        choices=["auto", "diag", "ross", "satpathy", "jackknife", "rascalc", "vac"],
        default="auto",
        help=(
            "covariance source: auto/diag/ross/satpathy/jackknife/rascalc/vac (default: auto). "
            "'ross' uses Ross 2016 full cov (post-recon). "
            "'satpathy' uses Satpathy 2016 full cov (pre-recon). "
            "'jackknife' uses a per-case cov file: output/private/cosmology/cosmology_bao_xi_from_catalogs_{tag}__jk_cov.npz. "
            "'rascalc' uses a per-case cov file: output/private/cosmology/cosmology_bao_xi_from_catalogs_{tag}__rascalc_cov.npz. "
            "'vac' uses a per-case projected full cov file: ...__vac_cov.npz (DESI DR1 VAC lya-correlations). "
            "Use --cov-path to override the cov npz path for jackknife/rascalc/vac."
        ),
    )
    ap.add_argument(
        "--cov-path",
        type=str,
        default="",
        help="Override covariance npz path for full cov sources (jackknife/rascalc). Relative paths are resolved from repo root.",
    )
    ap.add_argument(
        "--cov-suffix",
        type=str,
        default="",
        help=(
            "Override per-case covariance suffix for full cov sources (jackknife/rascalc). "
            "Example: --cov-suffix jk_cov_per_cap uses ...__jk_cov_per_cap.npz. "
            "Mutually exclusive with --cov-path."
        ),
    )
    ap.add_argument(
        "--cov-shrinkage",
        type=str,
        default="0.0",
        help="shrink full covariance towards diagonal: C'=(1-λ)C+λdiag(C); λ∈[0,1] or 'auto' (default: 0.0)",
    )
    ap.add_argument(
        "--cov-bandwidth-bins",
        type=int,
        default=-1,
        help=(
            "for full cov sources (jackknife/rascalc): band the dv=[xi0,xi2] covariance by |Δs_bin| in fit space. "
            "-1 disables banding (default). 0 keeps only same-bin entries (incl. xi0-xi2 at same s)."
        ),
    )
    ap.add_argument(
        "--cov-bandwidth-xi02-bins",
        type=int,
        default=-1,
        help=(
            "optional override bandwidth for xi0-xi2 cross-cov blocks only (default: -1 => same as --cov-bandwidth-bins)."
        ),
    )
    ap.add_argument(
        "--cov-zero-xi02",
        action="store_true",
        help="for full cov sources (jackknife/rascalc): zero the xi0-xi2 cross-cov blocks before inversion (default: off)",
    )
    ap.add_argument(
        "--ross-dir",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "ross_2016_combineddr12_corrfunc"),
        help="Ross 2016 COMBINEDDR12 covariance dir (default: data/cosmology/ross_2016_combineddr12_corrfunc)",
    )
    ap.add_argument("--ross-bincent", type=int, default=0, help="Ross covariance bincent (0..4; default: 0)")
    ap.add_argument(
        "--satpathy-dir",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "satpathy_2016_combineddr12_fs_corrfunc_multipoles"),
        help="Satpathy 2016 COMBINEDDR12 pre-recon covariance dir (default: data/cosmology/satpathy_2016_combineddr12_fs_corrfunc_multipoles)",
    )

    # Fit/scan controls (match the published-multipole fit defaults).
    ap.add_argument("--r-min", type=float, default=50.0, help="min separation r [Mpc/h] for fit (default: 50)")
    ap.add_argument("--r-max", type=float, default=150.0, help="max separation r [Mpc/h] for fit (default: 150)")
    ap.add_argument("--mu-n", type=int, default=80, help="Gauss-Legendre points for μ integral (default: 80)")
    ap.add_argument("--r0", type=float, default=105.0, help="BAO peak center (template) [Mpc/h] (default: 105)")
    ap.add_argument("--sigma", type=float, default=10.0, help="BAO peak width [Mpc/h] (default: 10)")
    ap.add_argument(
        "--smooth-power-max",
        type=int,
        default=2,
        help="smooth basis max power p for 1/r^p (default: 2 => [1, 1/r, 1/r^2])",
    )
    ap.add_argument("--alpha-min", type=float, default=0.9, help="alpha grid min (default: 0.9)")
    ap.add_argument("--alpha-max", type=float, default=1.1, help="alpha grid max (default: 1.1)")
    ap.add_argument("--alpha-step", type=float, default=0.002, help="alpha grid step (default: 0.002)")
    ap.add_argument("--eps-min", type=float, default=-0.1, help="eps grid min (default: -0.1)")
    ap.add_argument("--eps-max", type=float, default=0.1, help="eps grid max (default: 0.1)")
    ap.add_argument("--eps-step", type=float, default=0.002, help="eps grid step (default: 0.002)")
    ap.add_argument(
        "--eps-rescan-factor",
        type=float,
        default=2.0,
        help="if eps scan hits edge/CI clipped, widen eps range by this factor and rescan (default: 2.0)",
    )
    ap.add_argument(
        "--eps-rescan-max-expands",
        type=int,
        default=2,
        help="max number of eps-range expansions for rescan (0 disables; default: 2)",
    )

    # Error model (screening).
    ap.add_argument("--var-source", choices=["auto", "dd", "dr", "rr"], default="dd", help="paircount source for var proxy (default: dd)")
    ap.add_argument("--min-pairs", type=float, default=1.0, help="floor for paircount in variance proxy (default: 1)")
    ap.add_argument("--quad-weight", type=float, default=1.0, help="quadrupole weight (>1 emphasizes xi2; default: 1)")

    # Status thresholds (screening policy; tuned later).
    ap.add_argument("--ok-max", type=float, default=1.0, help="|eps|/sigma_eps <= ok-max => ok (default: 1.0)")
    ap.add_argument("--mixed-max", type=float, default=2.0, help="|eps|/sigma_eps <= mixed-max => mixed (default: 2.0)")

    args = ap.parse_args(list(argv) if argv is not None else None)
    cov_path_override = str(getattr(args, "cov_path", "")).strip()
    cov_suffix_override = str(getattr(args, "cov_suffix", "")).strip()
    if cov_path_override and cov_suffix_override:
        raise SystemExit("--cov-path and --cov-suffix are mutually exclusive")
    if (cov_path_override or cov_suffix_override) and str(args.cov_source) not in {"jackknife", "rascalc", "vac"}:
        raise SystemExit("--cov-path/--cov-suffix is only supported with --cov-source jackknife, rascalc, or vac")

    sample = str(args.sample)
    caps = str(args.caps)
    dists = [d.strip() for d in str(args.dists).split(",") if d.strip()]
    if not dists:
        raise SystemExit("--dists must not be empty")

    cases = _load_cases(sample=sample, caps=caps, dists=dists, require_zbin=bool(args.require_zbin), out_tag=str(args.out_tag))
    if not cases:
        raise SystemExit(f"no matching inputs for sample={sample}, caps={caps}, dists={dists}, require_zbin={args.require_zbin}")

    # Guard: coordinate-spec must be fixed across inputs to avoid mixing "theory vs coordinate" effects.
    z_sources = {str(c.z_source) for c in cases if str(c.z_source)}
    los_defs = {str(c.los) for c in cases if str(c.los)}
    weight_schemes = {str(c.weight_scheme) for c in cases if str(c.weight_scheme)}
    est_hashes = {str(c.estimator_spec_hash) for c in cases if str(c.estimator_spec_hash)}
    recon_modes = {str(c.recon_mode) for c in cases if str(c.recon_mode)}
    if len(z_sources) != 1:
        details = ", ".join(sorted(z_sources)) if z_sources else "(missing)"
        raise SystemExit(f"mixed z_source across inputs: {details}")
    if len(los_defs) != 1:
        details = ", ".join(sorted(los_defs)) if los_defs else "(missing)"
        raise SystemExit(f"mixed los across inputs: {details}")
    if len(weight_schemes) != 1:
        details = ", ".join(sorted(weight_schemes)) if weight_schemes else "(missing)"
        raise SystemExit(f"mixed weight_scheme across inputs: {details}")
    if len(est_hashes) != 1:
        details = ", ".join(sorted(est_hashes)) if est_hashes else "(missing)"
        raise SystemExit(f"mixed estimator_spec across inputs (re-run xi-from-catalogs or filter inputs): {details}")
    if len(recon_modes) != 1:
        details = ", ".join(sorted(recon_modes)) if recon_modes else "(missing)"
        raise SystemExit(f"mixed recon_mode across inputs (filter inputs): {details}")

    lcdm_cases = [c for c in cases if str(c.dist) == "lcdm"]
    if lcdm_cases:
        base_omega_m = float(lcdm_cases[0].lcdm_omega_m)
        base_n_grid = int(lcdm_cases[0].lcdm_n_grid)
        base_z_grid_max = float(lcdm_cases[0].lcdm_z_grid_max)
        if not (math.isfinite(base_omega_m) and math.isfinite(base_z_grid_max) and base_n_grid > 0):
            raise SystemExit("lcdm coordinate spec missing/invalid in inputs (run xi-from-catalogs with fixed --lcdm-*)")
        for c in lcdm_cases[1:]:
            if (not math.isfinite(float(c.lcdm_omega_m))) or abs(float(c.lcdm_omega_m) - base_omega_m) > 1e-12:
                raise SystemExit("mixed lcdm_omega_m across inputs (fix by re-running xi-from-catalogs)")
            if int(c.lcdm_n_grid) != base_n_grid:
                raise SystemExit("mixed lcdm_n_grid across inputs (fix by re-running xi-from-catalogs)")
            if (not math.isfinite(float(c.lcdm_z_grid_max))) or abs(float(c.lcdm_z_grid_max) - base_z_grid_max) > 1e-12:
                raise SystemExit("mixed lcdm_z_grid_max across inputs (fix by re-running xi-from-catalogs)")
    else:
        base_omega_m = float("nan")
        base_n_grid = 0
        base_z_grid_max = float("nan")

    coordinate_spec_common: Dict[str, Any] = {
        "z_source": next(iter(z_sources)) if z_sources else "",
        "los": next(iter(los_defs)) if los_defs else "",
        "weight_scheme": next(iter(weight_schemes)) if weight_schemes else "",
        "lcdm": {"Omega_m": base_omega_m, "n_grid": int(base_n_grid), "z_grid_max": base_z_grid_max},
    }
    estimator_spec_common: Dict[str, Any] = {"estimator_spec_hash": next(iter(est_hashes)) if est_hashes else ""}

    mu_n = int(args.mu_n)
    if mu_n < 20:
        raise SystemExit("--mu-n must be >= 20")
    mu, w = np.polynomial.legendre.leggauss(mu_n)
    sqrt1mu2 = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
    p2_fid = _peakfit._p2(mu)

    alpha_grid = np.arange(float(args.alpha_min), float(args.alpha_max) + 0.5 * float(args.alpha_step), float(args.alpha_step))
    eps_grid = np.arange(float(args.eps_min), float(args.eps_max) + 0.5 * float(args.eps_step), float(args.eps_step))
    if alpha_grid.size < 5 or eps_grid.size < 5:
        raise SystemExit("grid too small; widen ranges or reduce steps")

    eps_rescan_factor = float(args.eps_rescan_factor)
    eps_rescan_max_expands = int(args.eps_rescan_max_expands)
    if not (math.isfinite(eps_rescan_factor) and eps_rescan_factor > 1.0):
        raise SystemExit("--eps-rescan-factor must be > 1")
    if eps_rescan_max_expands < 0:
        raise SystemExit("--eps-rescan-max-expands must be >= 0")

    cov_shrinkage_raw = str(getattr(args, "cov_shrinkage", "") or "").strip().lower()
    cov_shrinkage_mode = "fixed"
    cov_shrinkage_fixed: float | None = None
    if cov_shrinkage_raw in ("auto", "lw", "ledoit_wolf"):
        cov_shrinkage_mode = "ledoit_wolf_diag"
        cov_shrinkage_fixed = None
    else:
        try:
            cov_shrinkage_fixed = float(cov_shrinkage_raw)
        except Exception as e:
            raise SystemExit("--cov-shrinkage must be a float in [0,1] or 'auto'") from e
        if not (math.isfinite(cov_shrinkage_fixed) and 0.0 <= cov_shrinkage_fixed <= 1.0):
            raise SystemExit("--cov-shrinkage must be within [0,1]")

    r_min = float(args.r_min)
    r_max = float(args.r_max)
    r0 = float(args.r0)
    sigma = float(args.sigma)
    smooth_power_max = int(args.smooth_power_max)
    if smooth_power_max < 0:
        raise SystemExit("--smooth-power-max must be >= 0")

    cov_bandwidth_bins = int(args.cov_bandwidth_bins)
    cov_bandwidth_xi02_bins = int(args.cov_bandwidth_xi02_bins)
    if cov_bandwidth_bins < -1:
        raise SystemExit("--cov-bandwidth-bins must be >= -1")
    if cov_bandwidth_xi02_bins < -1:
        raise SystemExit("--cov-bandwidth-xi02-bins must be >= -1")

    records: List[Dict[str, Any]] = []
    curves: List[Dict[str, Any]] = []
    ross_cache: Dict[Tuple[int, int, float, float, int], Tuple[np.ndarray, Path]] = {}
    satpathy_cache: Dict[Tuple[int, float, float, int], Tuple[np.ndarray, Path]] = {}

    for case in cases:
        with np.load(case.npz_path) as z:
            s_all = np.asarray(z["s"], dtype=float)
            xi0_all = np.asarray(z["xi0"], dtype=float)
            xi2_all = np.asarray(z["xi2"], dtype=float)
            components = np.asarray(z.get("components", []))

            mu_edges = np.asarray(z.get("mu_edges", []), dtype=float)
            dd_w = np.asarray(z.get("dd_w", []), dtype=float)
            dr_w = np.asarray(z.get("dr_w", []), dtype=float)
            rr_w = np.asarray(z.get("rr_w", []), dtype=float)

        if xi0_all.ndim == 1:
            n_components = 1
            m_fit = (s_all >= r_min) & (s_all <= r_max) & np.isfinite(xi0_all) & np.isfinite(xi2_all)
        elif xi0_all.ndim == 2:
            if xi2_all.ndim != 2 or xi2_all.shape != xi0_all.shape:
                raise SystemExit(f"invalid multi-component xi shapes: xi0={xi0_all.shape}, xi2={xi2_all.shape}")
            if xi0_all.shape[1] != s_all.shape[0]:
                raise SystemExit(f"invalid multi-component xi vs s: xi0={xi0_all.shape}, s={s_all.shape}")
            n_components = int(xi0_all.shape[0])
            finite = np.all(np.isfinite(xi0_all), axis=0) & np.all(np.isfinite(xi2_all), axis=0)
            m_fit = (s_all >= r_min) & (s_all <= r_max) & finite
            if components.size and int(components.size) != n_components:
                raise SystemExit(f"components length mismatch: components={components.size}, n_components={n_components}")
        else:
            raise SystemExit(f"unsupported xi0 ndim: {xi0_all.ndim}")

        if int(np.count_nonzero(m_fit)) < 10:
            # Too few points to stabilize the linear parameters.
            continue
        s = s_all[m_fit]
        xi0 = xi0_all[m_fit] if n_components == 1 else xi0_all[:, m_fit]
        xi2 = xi2_all[m_fit] if n_components == 1 else xi2_all[:, m_fit]

        # Approximate diagonal covariance.
        try:
            var0_all, var2_all = _approx_var_from_paircounts(
                s=s_all,
                mu_edges=mu_edges,
                dd_w=dd_w,
                dr_w=dr_w,
                rr_w=rr_w,
                var_source=str(args.var_source),
                min_pairs=float(args.min_pairs),
                quad_weight=float(args.quad_weight),
            )
            var0 = np.asarray(var0_all, dtype=float)[m_fit]
            var2 = np.asarray(var2_all, dtype=float)[m_fit]
        except Exception:
            var0 = np.full(int(s.size), 1.0, dtype=float)
            var2 = np.full(int(s.size), 1.0, dtype=float)
        cov_source = str(args.cov_source)
        cov_source_actual = "diag_paircount_proxy"
        cov_inv = _diag_cov_inv(var0, var2)
        if int(n_components) > 1:
            # Treat components as independent under the crude diagonal proxy.
            cov_inv = np.kron(np.eye(int(n_components), dtype=float), cov_inv)
        cov_inputs: Dict[str, Any] = {
            "type": cov_source_actual,
            "var_source": str(args.var_source),
            "min_pairs": float(args.min_pairs),
            "quad_weight": float(args.quad_weight),
        }

        # Phase B option: use published full covariance (mono+quad) for z-binned combined cases.
        use_jackknife = False
        use_rascalc = False
        use_ross = False
        use_satpathy = False
        use_vac = False
        if cov_source == "jackknife":
            use_jackknife = True
        elif cov_source == "rascalc":
            use_rascalc = True
        elif cov_source == "ross":
            use_ross = True
        elif cov_source == "satpathy":
            use_satpathy = True
        elif cov_source == "vac":
            use_vac = True
        elif cov_source == "auto":
            if case.sample == "cmasslowztot" and case.caps == "combined" and case.z_bin != "none":
                # Use Satpathy for pre-recon, Ross for post-recon.
                if str(case.recon_mode) == "none":
                    use_satpathy = True
                else:
                    use_ross = True

        if use_vac:
            # Per-case projected full covariance for multi-component inputs (DESI DR1 VAC lya-correlations).
            # Expected file name convention:
            #   cosmology_bao_xi_from_catalogs_{tag}.npz
            #   cosmology_bao_xi_from_catalogs_{tag}__vac_cov.npz
            if cov_path_override:
                cov_path = Path(cov_path_override)
                cov_path = cov_path if cov_path.is_absolute() else (_ROOT / cov_path).resolve()
            elif cov_suffix_override:
                cov_path = case.npz_path.with_name(f"{case.npz_path.stem}__{cov_suffix_override}.npz")
            else:
                cov_path = case.npz_path.with_name(f"{case.npz_path.stem}__vac_cov.npz")
            if not cov_path.exists():
                raise SystemExit(
                    "vac covariance file not found. "
                    f"Expected: {cov_path} "
                    "(run scripts/cosmology/cosmology_bao_xi_from_desi_dr1_vac_lya_correlations.py)"
                )
            with np.load(cov_path) as zc:
                s_cov = np.asarray(zc["s"], dtype=float).reshape(-1)
                cov_full = np.asarray(zc["cov"], dtype=float)

            if s_cov.shape != s_all.shape or not np.allclose(s_cov, s_all, rtol=0.0, atol=1e-12):
                raise SystemExit(
                    "vac cov uses different s bins vs xi output "
                    f"(xi={case.npz_path.name}, cov={cov_path.name})"
                )

            n_all = int(s_all.size)
            idx = np.nonzero(m_fit)[0].astype(int, copy=False)
            p = int(2 * int(n_components) * n_all)
            cov_full = np.asarray(cov_full, dtype=float).reshape(p, p)
            blocks: List[np.ndarray] = []
            for c in range(int(n_components)):
                blocks.append(idx + (2 * c + 0) * n_all)
                blocks.append(idx + (2 * c + 1) * n_all)
            sel = np.concatenate(blocks, axis=0)
            cov = cov_full[np.ix_(sel, sel)]

            if bool(args.cov_zero_xi02):
                cov = _cov_zero_xi02_cross(cov, n_bins=int(idx.size), n_components=int(n_components))

            banding_stats: Dict[str, float] | None = None
            if cov_bandwidth_bins >= 0:
                cov, banding_stats = _cov_band_by_s_index(
                    cov,
                    n_bins=int(idx.size),
                    n_components=int(n_components),
                    bandwidth_bins=int(cov_bandwidth_bins),
                    bandwidth_xi02_bins=int(cov_bandwidth_xi02_bins),
                )

            cov_shrinkage_actual = cov_shrinkage_fixed
            shrinkage_note: str | None = None
            if cov_shrinkage_actual is None:
                cov_shrinkage_actual = 0.0
                shrinkage_note = "auto requested, but vac cov has no y_jk; fallback to λ=0"

            cov = _cov_shrink_to_diag(cov, lam=float(cov_shrinkage_actual))
            cov_inv = np.linalg.pinv(cov, rcond=1e-12)
            cov_inv = 0.5 * (cov_inv + cov_inv.T)

            cov_source_actual = "desi_dr1_vac_fullcov_projected"
            cov_inputs = {
                "type": cov_source_actual,
                "vac_cov": str(cov_path),
                "rcond": 1e-12,
                "shrinkage": float(cov_shrinkage_actual),
                "shrinkage_mode": str(cov_shrinkage_mode),
                "bandwidth_bins": int(cov_bandwidth_bins),
                "bandwidth_xi02_bins": int(cov_bandwidth_xi02_bins),
                **({"banding_keep_fraction": float(banding_stats.get("keep_fraction", 1.0))} if banding_stats else {}),
                "xi02_zeroed": bool(args.cov_zero_xi02),
                **({"shrinkage_note": str(shrinkage_note)} if shrinkage_note else {}),
            }
            if cov_path_override:
                cov_inputs["cov_path_override"] = str(cov_path_override)
            if cov_suffix_override:
                cov_inputs["cov_suffix_override"] = str(cov_suffix_override)

        if use_jackknife:
            # Per-case covariance estimated from the same catalogs via sky jackknife.
            # Expected file name convention (1:1 with the xi output):
            #   cosmology_bao_xi_from_catalogs_{tag}.npz
            #   cosmology_bao_xi_from_catalogs_{tag}__jk_cov.npz
            if cov_path_override:
                cov_path = Path(cov_path_override)
                cov_path = cov_path if cov_path.is_absolute() else (_ROOT / cov_path).resolve()
            elif cov_suffix_override:
                cov_path = case.npz_path.with_name(f"{case.npz_path.stem}__{cov_suffix_override}.npz")
            else:
                cov_path = case.npz_path.with_name(f"{case.npz_path.stem}__jk_cov.npz")
            if not cov_path.exists():
                raise SystemExit(
                    "jackknife covariance file not found. "
                    f"Expected: {cov_path} "
                    "(run scripts/cosmology/cosmology_bao_xi_jackknife_cov_from_catalogs.py under WSL)"
                )
            with np.load(cov_path) as zc:
                s_cov = np.asarray(zc["s"], dtype=float).reshape(-1)
                cov_full = np.asarray(zc["cov"], dtype=float)
                y_jk = np.asarray(zc["y_jk"], dtype=float) if ("y_jk" in zc.files) else np.asarray([], dtype=float)

            if s_cov.shape != s_all.shape or not np.allclose(s_cov, s_all, rtol=0.0, atol=1e-12):
                raise SystemExit(
                    "jackknife cov uses different s bins vs xi output "
                    f"(xi={case.npz_path.name}, cov={cov_path.name})"
                )

            n_all = int(s_all.size)
            idx = np.nonzero(m_fit)[0].astype(int, copy=False)
            p = int(2 * int(n_components) * n_all)
            cov_full = np.asarray(cov_full, dtype=float).reshape(p, p)
            if int(n_components) == 1:
                sel = np.concatenate([idx, idx + n_all], axis=0)
            else:
                blocks: List[np.ndarray] = []
                for c in range(int(n_components)):
                    blocks.append(idx + (2 * c + 0) * n_all)
                    blocks.append(idx + (2 * c + 1) * n_all)
                sel = np.concatenate(blocks, axis=0)
            cov = cov_full[np.ix_(sel, sel)]
            if bool(args.cov_zero_xi02):
                cov = _cov_zero_xi02_cross(cov, n_bins=int(idx.size), n_components=int(n_components))

            banding_stats: Dict[str, float] | None = None
            if cov_bandwidth_bins >= 0:
                cov, banding_stats = _cov_band_by_s_index(
                    cov,
                    n_bins=int(idx.size),
                    n_components=int(n_components),
                    bandwidth_bins=int(cov_bandwidth_bins),
                    bandwidth_xi02_bins=int(cov_bandwidth_xi02_bins),
                )

            cov_shrinkage_actual = cov_shrinkage_fixed
            shrinkage_note: str | None = None
            if cov_shrinkage_actual is None:
                if y_jk.size == 0:
                    cov_shrinkage_actual = 0.0
                    shrinkage_note = "auto requested, but y_jk missing in jk_cov; fallback to λ=0"
                else:
                    y_jk = np.asarray(y_jk, dtype=float)
                    if y_jk.ndim != 2 or int(y_jk.shape[1]) != p:
                        raise SystemExit(f"y_jk shape mismatch in jk_cov: got {y_jk.shape}, expected (*,{p})")
                    y_fit = y_jk[:, sel]
                    cov_shrinkage_actual = _ledoit_wolf_shrinkage_to_diag(y_fit)

            cov = _cov_shrink_to_diag(cov, lam=float(cov_shrinkage_actual))
            cov_inv = np.linalg.pinv(cov, rcond=1e-12)
            cov_inv = 0.5 * (cov_inv + cov_inv.T)

            cov_source_actual = "jackknife"
            cov_metrics_path = cov_path.with_name(f"{cov_path.stem}_metrics.json")
            if cov_metrics_path.exists():
                try:
                    cov_meta = json.loads(cov_metrics_path.read_text(encoding="utf-8"))
                    jk = (cov_meta.get("params", {}) or {}).get("jackknife", {}) or {}
                    jk_mode = str(jk.get("mode", "")).strip()
                    if jk_mode:
                        cov_source_actual = f"jackknife_{jk_mode}"
                except Exception:
                    pass
            cov_inputs = {
                "type": cov_source_actual,
                "jk_cov": str(cov_path),
                "rcond": 1e-12,
                "shrinkage": float(cov_shrinkage_actual),
                "shrinkage_mode": str(cov_shrinkage_mode),
                "bandwidth_bins": int(cov_bandwidth_bins),
                "bandwidth_xi02_bins": int(cov_bandwidth_xi02_bins),
                **({"banding_keep_fraction": float(banding_stats.get("keep_fraction", 1.0))} if banding_stats else {}),
                "xi02_zeroed": bool(args.cov_zero_xi02),
                **({"shrinkage_note": str(shrinkage_note)} if shrinkage_note else {}),
            }
            if cov_path_override:
                cov_inputs["cov_path_override"] = str(cov_path_override)
            if cov_suffix_override:
                cov_inputs["cov_suffix_override"] = str(cov_suffix_override)

        if use_rascalc:
            # Per-case covariance estimated from the same catalogs via RascalC (legendre_projected).
            # Expected file name convention (1:1 with the xi output):
            #   cosmology_bao_xi_from_catalogs_{tag}.npz
            #   cosmology_bao_xi_from_catalogs_{tag}__rascalc_cov.npz
            if cov_path_override:
                cov_path = Path(cov_path_override)
                cov_path = cov_path if cov_path.is_absolute() else (_ROOT / cov_path).resolve()
            elif cov_suffix_override:
                cov_path = case.npz_path.with_name(f"{case.npz_path.stem}__{cov_suffix_override}.npz")
            else:
                cov_path = case.npz_path.with_name(f"{case.npz_path.stem}__rascalc_cov.npz")
            if not cov_path.exists():
                raise SystemExit(
                    "RascalC covariance file not found. "
                    f"Expected: {cov_path} "
                    "(run scripts/cosmology/cosmology_bao_xi_rascalc_cov_from_catalogs.py under WSL)"
                )
            with np.load(cov_path) as zc:
                s_cov = np.asarray(zc["s"], dtype=float).reshape(-1)
                cov_full = np.asarray(zc["cov"], dtype=float)

            if s_cov.shape != s_all.shape or not np.allclose(s_cov, s_all, rtol=0.0, atol=1e-12):
                raise SystemExit(
                    "RascalC cov uses different s bins vs xi output "
                    f"(xi={case.npz_path.name}, cov={cov_path.name})"
                )

            n_all = int(s_all.size)
            idx = np.nonzero(m_fit)[0].astype(int, copy=False)
            p = int(2 * int(n_components) * n_all)
            cov_full = np.asarray(cov_full, dtype=float).reshape(p, p)
            if int(n_components) == 1:
                sel = np.concatenate([idx, idx + n_all], axis=0)
            else:
                blocks: List[np.ndarray] = []
                for c in range(int(n_components)):
                    blocks.append(idx + (2 * c + 0) * n_all)
                    blocks.append(idx + (2 * c + 1) * n_all)
                sel = np.concatenate(blocks, axis=0)
            cov = cov_full[np.ix_(sel, sel)]
            if bool(args.cov_zero_xi02):
                cov = _cov_zero_xi02_cross(cov, n_bins=int(idx.size), n_components=int(n_components))

            banding_stats = None
            if cov_bandwidth_bins >= 0:
                cov, banding_stats = _cov_band_by_s_index(
                    cov,
                    n_bins=int(idx.size),
                    n_components=int(n_components),
                    bandwidth_bins=int(cov_bandwidth_bins),
                    bandwidth_xi02_bins=int(cov_bandwidth_xi02_bins),
                )

            cov_shrinkage_actual = cov_shrinkage_fixed
            shrinkage_note: str | None = None
            if cov_shrinkage_actual is None:
                cov_shrinkage_actual = 0.0
                shrinkage_note = "auto requested, but RascalC cov has no y_jk; fallback to λ=0"

            cov = _cov_shrink_to_diag(cov, lam=float(cov_shrinkage_actual))
            cov_inv = np.linalg.pinv(cov, rcond=1e-12)
            cov_inv = 0.5 * (cov_inv + cov_inv.T)
            cov_source_actual = "rascalc_legendre_projected"
            cov_inputs = {
                "type": cov_source_actual,
                "rascalc_cov": str(cov_path),
                "rcond": 1e-12,
                "shrinkage": float(cov_shrinkage_actual),
                "shrinkage_mode": str(cov_shrinkage_mode),
                "bandwidth_bins": int(cov_bandwidth_bins),
                "bandwidth_xi02_bins": int(cov_bandwidth_xi02_bins),
                **({"banding_keep_fraction": float(banding_stats.get("keep_fraction", 1.0))} if banding_stats else {}),
                "xi02_zeroed": bool(args.cov_zero_xi02),
                **({"shrinkage_note": str(shrinkage_note)} if shrinkage_note else {}),
            }
            if cov_path_override:
                cov_inputs["cov_path_override"] = str(cov_path_override)
            if cov_suffix_override:
                cov_inputs["cov_suffix_override"] = str(cov_suffix_override)

        if use_ross:
            zbin_id = _parse_zbin_id(case.z_bin)
            if zbin_id is None:
                raise SystemExit(f"--cov-source {cov_source} requires z-binned inputs (got z_bin={case.z_bin})")
            if not (case.sample == "cmasslowztot" and case.caps == "combined"):
                raise SystemExit(
                    f"--cov-source {cov_source} currently supports sample=cmasslowztot,caps=combined only (got {case.sample}/{case.caps})"
                )

            s_step = float(np.median(np.diff(s_all))) if s_all.size >= 2 else float("nan")
            if not (math.isfinite(s_step) and abs(s_step - 5.0) < 1e-9):
                raise SystemExit(f"Ross cov requires s_step=5.0 (got {s_step}); re-run xi-from-catalogs with --s-step 5")

            cache_key = (int(zbin_id), int(args.ross_bincent), float(r_min), float(r_max), int(s.size))
            cached = ross_cache.get(cache_key)
            if cached is None:
                cov_inv_ross, cov_path = _ross_cov_inv_for_s(
                    ross_dir=Path(str(args.ross_dir)),
                    zbin=int(zbin_id),
                    bincent=int(args.ross_bincent),
                    s=s,
                    s_step=float(s_step),
                    rcond=1e-12,
                )
                ross_cache[cache_key] = (cov_inv_ross, cov_path)
                cached = (cov_inv_ross, cov_path)
            cov_inv, cov_path = cached
            cov_source_actual = "ross2016_mocks_fullcov"
            cov_inputs = {
                "type": cov_source_actual,
                "ross_cov": str(cov_path),
                "ross_bincent": int(args.ross_bincent),
                "rcond": 1e-12,
            }

        if use_satpathy:
            if str(case.recon_mode) != "none":
                raise SystemExit(f"Satpathy cov is pre-recon only (got recon_mode={case.recon_mode})")
            zbin_id = _parse_zbin_id(case.z_bin)
            if zbin_id is None:
                raise SystemExit(f"--cov-source {cov_source} requires z-binned inputs (got z_bin={case.z_bin})")
            if not (case.sample == "cmasslowztot" and case.caps == "combined"):
                raise SystemExit(
                    f"--cov-source {cov_source} currently supports sample=cmasslowztot,caps=combined only (got {case.sample}/{case.caps})"
                )

            s_step = float(np.median(np.diff(s_all))) if s_all.size >= 2 else float("nan")
            if not (math.isfinite(s_step) and abs(s_step - 5.0) < 1e-9):
                raise SystemExit(f"Satpathy cov requires s_step=5.0 (got {s_step}); re-run xi-from-catalogs with --s-step 5")

            cache_key = (int(zbin_id), float(r_min), float(r_max), int(s.size))
            cached = satpathy_cache.get(cache_key)
            if cached is None:
                cov_inv_sat, cov_path = _satpathy_cov_inv_for_s(
                    satpathy_dir=Path(str(args.satpathy_dir)),
                    zbin=int(zbin_id),
                    s=s,
                    s_step=float(s_step),
                    rcond=1e-12,
                )
                satpathy_cache[cache_key] = (cov_inv_sat, cov_path)
                cached = (cov_inv_sat, cov_path)
            cov_inv, cov_path = cached
            cov_source_actual = "satpathy2016_mocks_fullcov_prerecon"
            cov_inputs = {
                "type": cov_source_actual,
                "satpathy_cov": str(cov_path),
                "rcond": 1e-12,
            }

        if int(n_components) == 1:
            y = np.concatenate([xi0, xi2], axis=0)
        else:
            n_fit = int(s.size)
            y2 = np.empty((2 * int(n_components), n_fit), dtype=float)
            for c in range(int(n_components)):
                y2[2 * c + 0, :] = np.asarray(xi0[c], dtype=float)
                y2[2 * c + 1, :] = np.asarray(xi2[c], dtype=float)
            y = y2.reshape(-1)

        best_eps0 = _peakfit._scan_grid(
            y=y,
            cov_inv=cov_inv,
            s_fid=s,
            alpha_grid=alpha_grid,
            eps_grid=np.asarray([0.0], dtype=float),
            r0_mpc_h=r0,
            sigma_mpc_h=sigma,
            mu=mu,
            w=w,
            p2_fid=p2_fid,
            sqrt1mu2=sqrt1mu2,
            smooth_power_max=smooth_power_max,
            n_components=int(n_components),
            return_eps_profile=False,
            return_alpha_profile=True,
        )

        # eps scan (with optional rescan to avoid over-confident CI when the best-fit is clipped by bounds).
        eps_grid_use = eps_grid
        expands_done = 0
        eps_scan_initial = {
            "min": float(np.min(eps_grid_use)),
            "max": float(np.max(eps_grid_use)),
            "step": float(eps_grid_use[1] - eps_grid_use[0]) if eps_grid_use.size >= 2 else float("nan"),
        }

        n_linear = _peakfit._n_linear_params(smooth_power_max=smooth_power_max, n_components=int(n_components))
        dof_free = int(y.size - (n_linear + 2))
        dof_eps_fixed = int(y.size - (n_linear + 1))
        chi2_free = float("nan")
        chi2_scale = 1.0
        eps_ci_1 = float("nan")
        eps_ci_2 = float("nan")
        eps_ci_2s = float("nan")
        eps_ci_2s_hi = float("nan")
        edge_hit = False
        ci_clipped = False

        while True:
            best_free = _peakfit._scan_grid(
                y=y,
                cov_inv=cov_inv,
                s_fid=s,
                alpha_grid=alpha_grid,
                eps_grid=eps_grid_use,
                r0_mpc_h=r0,
                sigma_mpc_h=sigma,
                mu=mu,
                w=w,
                p2_fid=p2_fid,
                sqrt1mu2=sqrt1mu2,
                smooth_power_max=smooth_power_max,
                n_components=int(n_components),
                return_eps_profile=True,
                return_alpha_profile=True,
            )

            chi2_free = float(best_free["chi2"])
            if cov_source_actual == "diag_paircount_proxy":
                # Inflate the diagonal covariance so that the best-fit reduced chi2 becomes ~1.
                # This is a common screening trick to avoid over-confident CI when the error model is crude.
                chi2_scale = (chi2_free / float(dof_free)) if (dof_free > 0) else 1.0
                if not (math.isfinite(chi2_scale) and chi2_scale > 0.0):
                    chi2_scale = 1.0
                if chi2_scale < 1.0:
                    chi2_scale = 1.0
            else:
                chi2_scale = 1.0

            eps_profile = list(best_free.get("eps_profile") or [])
            eps_grid_vals = np.array([float(r["eps"]) for r in eps_profile], dtype=float) if eps_profile else np.array([], dtype=float)
            chi2_prof = np.array([float(r["chi2"]) for r in eps_profile], dtype=float) if eps_profile else np.array([], dtype=float)
            chi2_prof_scaled = chi2_prof / float(chi2_scale) if chi2_prof.size else chi2_prof
            eps_ci_1, eps_ci_2 = _peakfit._profile_ci(x=eps_grid_vals, chi2=chi2_prof_scaled, delta=1.0)
            eps_ci_2s, eps_ci_2s_hi = _peakfit._profile_ci(x=eps_grid_vals, chi2=chi2_prof_scaled, delta=4.0)

            eps_best_tmp = float(best_free["eps"])
            eps_min_tmp = float(np.min(eps_grid_use))
            eps_max_tmp = float(np.max(eps_grid_use))
            edge_hit = (eps_best_tmp <= eps_min_tmp + 1e-12) or (eps_best_tmp >= eps_max_tmp - 1e-12)
            try:
                lo = float(eps_ci_1)
                hi = float(eps_ci_2)
                ci_clipped = (lo <= eps_min_tmp + 1e-12) or (hi >= eps_max_tmp - 1e-12)
            except Exception:
                ci_clipped = True

            if (not edge_hit) and (not ci_clipped):
                break
            if expands_done >= eps_rescan_max_expands:
                break

            # Widen eps range symmetrically around the current center to probe outside-of-bounds best-fit.
            width0 = float(eps_max_tmp - eps_min_tmp)
            center0 = 0.5 * float(eps_max_tmp + eps_min_tmp)
            width1 = width0 * float(eps_rescan_factor)
            eps_min1 = center0 - 0.5 * width1
            eps_max1 = center0 + 0.5 * width1
            eps_step1 = float(eps_grid_use[1] - eps_grid_use[0]) if eps_grid_use.size >= 2 else float(args.eps_step)
            eps_grid_use = np.arange(eps_min1, eps_max1 + 0.5 * eps_step1, eps_step1, dtype=float)
            expands_done += 1

        chi2_eps0 = float(best_eps0["chi2"])

        n = int(s.size)
        res_free = np.asarray(best_free["residual"], dtype=float)
        if int(n_components) == 1:
            chi2_0 = float(np.sum((res_free[:n] ** 2) * (1.0 / np.maximum(var0, 1e-30))))
            chi2_2 = float(np.sum((res_free[n:] ** 2) * (1.0 / np.maximum(var2, 1e-30))))
        else:
            res2 = res_free.reshape(2 * int(n_components), n)
            chi2_0 = float(np.sum((res2[0::2, :] ** 2) * (1.0 / np.maximum(var0[None, :], 1e-30))))
            chi2_2 = float(np.sum((res2[1::2, :] ** 2) * (1.0 / np.maximum(var2[None, :], 1e-30))))
        chi2_0_scaled = chi2_0 / float(chi2_scale)
        chi2_2_scaled = chi2_2 / float(chi2_scale)

        delta_chi2_eps = (chi2_eps0 - chi2_free) / float(chi2_scale)

        # Alpha profile CI (free; eps marginalized via profile).
        alpha_profile = list(best_free.get("alpha_profile") or [])
        alpha_grid_vals = (
            np.array([float(r["alpha"]) for r in alpha_profile], dtype=float) if alpha_profile else np.array([], dtype=float)
        )
        chi2_alpha_prof = (
            np.array([float(r["chi2"]) for r in alpha_profile], dtype=float) if alpha_profile else np.array([], dtype=float)
        )
        chi2_alpha_prof_scaled = chi2_alpha_prof / float(chi2_scale) if chi2_alpha_prof.size else chi2_alpha_prof
        alpha_ci_1, alpha_ci_2 = _peakfit._profile_ci(x=alpha_grid_vals, chi2=chi2_alpha_prof_scaled, delta=1.0)
        alpha_ci_2s, alpha_ci_2s_hi = _peakfit._profile_ci(x=alpha_grid_vals, chi2=chi2_alpha_prof_scaled, delta=4.0)

        # Alpha profile CI (eps fixed 0).
        chi2_scale_eps0 = 1.0
        if cov_source_actual == "diag_paircount_proxy":
            chi2_scale_eps0 = (chi2_eps0 / float(dof_eps_fixed)) if (dof_eps_fixed > 0) else 1.0
            if not (math.isfinite(chi2_scale_eps0) and chi2_scale_eps0 > 0.0):
                chi2_scale_eps0 = 1.0
            if chi2_scale_eps0 < 1.0:
                chi2_scale_eps0 = 1.0

        alpha0_profile = list(best_eps0.get("alpha_profile") or [])
        alpha0_grid_vals = (
            np.array([float(r["alpha"]) for r in alpha0_profile], dtype=float) if alpha0_profile else np.array([], dtype=float)
        )
        chi2_alpha0_prof = (
            np.array([float(r["chi2"]) for r in alpha0_profile], dtype=float) if alpha0_profile else np.array([], dtype=float)
        )
        chi2_alpha0_prof_scaled = (
            chi2_alpha0_prof / float(chi2_scale_eps0) if chi2_alpha0_prof.size else chi2_alpha0_prof
        )
        alpha0_ci_1, alpha0_ci_2 = _peakfit._profile_ci(x=alpha0_grid_vals, chi2=chi2_alpha0_prof_scaled, delta=1.0)
        alpha0_ci_2s, alpha0_ci_2s_hi = _peakfit._profile_ci(x=alpha0_grid_vals, chi2=chi2_alpha0_prof_scaled, delta=4.0)

        eps_best = float(best_free["eps"])
        abs_eps = abs(eps_best)
        sigma_eps = None
        abs_sigma = None
        abs_sigma_is_lower_bound = False
        eps_min_grid = float(np.min(eps_grid_use))
        eps_max_grid = float(np.max(eps_grid_use))
        eps_step = float(eps_grid_use[1] - eps_grid_use[0]) if eps_grid_use.size >= 2 else float("nan")
        try:
            lo = float(eps_ci_1)
            hi = float(eps_ci_2)
            if (not edge_hit) and (not ci_clipped):
                if math.isfinite(lo) and math.isfinite(hi) and (hi > lo):
                    sigma_eps = 0.5 * (hi - lo)
                    if sigma_eps > 0:
                        abs_sigma = abs_eps / float(sigma_eps)
        except Exception:
            sigma_eps = None
            abs_sigma = None

        if edge_hit or ci_clipped:
            # When the best-fit or its CI is clipped by the scan boundary, do not pretend we have a precise σ.
            # Mark as NG with a conservative lower-bound score (>= mixed_max+1).
            abs_sigma = float(args.mixed_max) + 1.0
            abs_sigma_is_lower_bound = True
            status = "ng"
        else:
            status = _status_from_abs_sigma(abs_sigma, ok_max=float(args.ok_max), mixed_max=float(args.mixed_max))

        rec = {
            "sample": case.sample,
            "caps": case.caps,
            "dist": case.dist,
            "z_bin": case.z_bin,
            "z_eff": case.z_eff,
            "fit_range_mpc_h": [r_min, r_max],
            "template_peak": {"r0_mpc_h": r0, "sigma_mpc_h": sigma},
            "covariance": {"source_requested": cov_source, "source_used": cov_inputs},
            "error_model": {
                "type": "diag_paircount_proxy",
                "var_source": str(args.var_source),
                "min_pairs": float(args.min_pairs),
                "quad_weight": float(args.quad_weight),
            },
            "fit": {
                "free": {
                    "alpha": float(best_free["alpha"]),
                    "eps": eps_best,
                    "chi2": chi2_free,
                    "dof": dof_free,
                    "chi2_dof": (chi2_free / float(dof_free)) if dof_free > 0 else float("nan"),
                    "chi2_scale_to_unit_reduced": float(chi2_scale),
                    "chi2_dof_scaled": (chi2_free / float(chi2_scale) / float(dof_free)) if dof_free > 0 else float("nan"),
                    "eps_ci_1sigma": [eps_ci_1, eps_ci_2],
                    "eps_ci_2sigma": [eps_ci_2s, eps_ci_2s_hi],
                    "alpha_ci_1sigma": [alpha_ci_1, alpha_ci_2],
                    "alpha_ci_2sigma": [alpha_ci_2s, alpha_ci_2s_hi],
                },
                "eps_fixed_0": {
                    "alpha": float(best_eps0["alpha"]),
                    "eps": 0.0,
                    "chi2": chi2_eps0,
                    "dof": dof_eps_fixed,
                    "chi2_dof": (chi2_eps0 / float(dof_eps_fixed)) if dof_eps_fixed > 0 else float("nan"),
                    "chi2_dof_scaled": (chi2_eps0 / float(chi2_scale) / float(dof_eps_fixed)) if dof_eps_fixed > 0 else float("nan"),
                    "alpha_ci_1sigma": [alpha0_ci_1, alpha0_ci_2],
                    "alpha_ci_2sigma": [alpha0_ci_2s, alpha0_ci_2s_hi],
                },
                "delta_chi2_eps0_vs_free": delta_chi2_eps,
            },
            "diagnostics": {
                "chi2_mono_only_free_diag": chi2_0,
                "chi2_quad_only_free_diag": chi2_2,
                "chi2_mono_only_free_diag_scaled": chi2_0_scaled,
                "chi2_quad_only_free_diag_scaled": chi2_2_scaled,
            },
            "screening": {
                "abs_eps": abs_eps,
                "sigma_eps_1sigma": sigma_eps,
                "abs_sigma": abs_sigma,
                "abs_sigma_is_lower_bound": abs_sigma_is_lower_bound,
                "scan": {
                    "eps_grid": {"min": eps_min_grid, "max": eps_max_grid, "step": eps_step},
                    "edge_hit": edge_hit,
                    "ci_clipped": ci_clipped,
                    "rescan": {
                        "factor": float(eps_rescan_factor),
                        "max_expands": int(eps_rescan_max_expands),
                        "expands_done": int(expands_done),
                        "eps_grid_initial": eps_scan_initial,
                    },
                },
                "status": status,
            },
            "inputs": {
                "npz": str(case.npz_path),
                "metrics_json": str(case.metrics_path),
            },
        }
        records.append(rec)

        # Curves for plot (optional, light).
        s_grid = np.linspace(r_min, r_max, 400, dtype=float)
        xi0_free, xi2_free = _peakfit._predict_curve(
            s_grid=s_grid,
            x=np.asarray(best_free["x"], dtype=float),
            alpha=float(best_free["alpha"]),
            eps=float(best_free["eps"]),
            r0_mpc_h=r0,
            sigma_mpc_h=sigma,
            mu=mu,
            w=w,
            p2_fid=p2_fid,
            sqrt1mu2=sqrt1mu2,
            smooth_power_max=smooth_power_max,
        )
        curves.append(
            {
                "case": {"sample": case.sample, "caps": case.caps, "dist": case.dist, "z_bin": case.z_bin, "z_eff": case.z_eff},
                "s_fit": s,
                "xi0": xi0,
                "xi2": xi2,
                "s_grid": s_grid,
                "xi0_fit": xi0_free,
                "xi2_fit": xi2_free,
            }
        )

    if not records:
        raise SystemExit("no cases produced fit results (check r-range / inputs)")

    # Plot: eps(z) per dist.
    import matplotlib.pyplot as plt

    _set_japanese_font()
    out_dir = _ROOT / "output" / "private" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{sample}_{caps}"
    if bool(args.require_zbin):
        tag = f"{tag}_zbinonly"
    out_tag_label = _sanitize_out_tag(str(args.out_tag))
    if str(args.out_tag) != "none":
        tag = f"{tag}__{out_tag_label or 'tag'}"
    output_suffix_label = _sanitize_out_tag(str(args.output_suffix).strip()) if str(args.output_suffix).strip() else ""
    if output_suffix_label:
        tag = f"{tag}__{output_suffix_label}"
    out_png = out_dir / f"cosmology_bao_catalog_peakfit_{tag}.png"
    out_json = out_dir / f"cosmology_bao_catalog_peakfit_{tag}_metrics.json"

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.axhline(0.0, color="#888888", lw=1.2)

    # Group by dist.
    by_dist: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_dist.setdefault(str(r["dist"]), []).append(r)
    colors = {"lcdm": "#1f77b4", "pbg": "#ff7f0e"}
    markers = {"lcdm": "o", "pbg": "s"}

    for dist, xs in sorted(by_dist.items(), key=lambda kv: kv[0]):
        xs_sorted = sorted(xs, key=lambda r: float(r.get("z_eff", float("nan"))))
        z = np.array([float(r.get("z_eff", float("nan"))) for r in xs_sorted], dtype=float)
        eps = np.array([float(((r.get("fit", {}) or {}).get("free", {}) or {}).get("eps", float("nan"))) for r in xs_sorted], dtype=float)
        ci = [((r.get("fit", {}) or {}).get("free", {}) or {}).get("eps_ci_1sigma", [None, None]) for r in xs_sorted]
        lo = np.array([float(c[0]) if (c and c[0] is not None) else float("nan") for c in ci], dtype=float)
        hi = np.array([float(c[1]) if (c and c[1] is not None) else float("nan") for c in ci], dtype=float)
        yerr = np.vstack([np.maximum(0.0, eps - lo), np.maximum(0.0, hi - eps)])
        ok = np.isfinite(z) & np.isfinite(eps)
        if not np.any(ok):
            continue
        ax.errorbar(
            z[ok],
            eps[ok],
            yerr=yerr[:, ok],
            color=colors.get(dist, None),
            marker=markers.get(dist, "o"),
            linestyle="-",
            lw=1.6,
            ms=6,
            capsize=3,
            label=dist,
        )

        # Annotate z_bin (only when present) to disambiguate.
        for rr in xs_sorted:
            zb = str(rr.get("z_bin", "none"))
            if zb == "none":
                continue
            ax.annotate(
                zb,
                (float(rr["z_eff"]), float(rr["fit"]["free"]["eps"])),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=9,
                color=colors.get(dist, "#333333"),
            )

    ax.set_xlabel("z_eff (galaxy-weighted)")
    ax.set_ylabel("ε (best-fit; smooth+peak)")
    ax.set_title(
        f"BAO catalog-based peakfit（{sample}, {caps}）: ε(z)  (smooth_p={smooth_power_max}, cov={str(args.cov_source)}, quad_weight={float(args.quad_weight):g})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "16.4 (BAO catalog-based smooth+peak peakfit)",
        "inputs": {
            "glob": "output/private/cosmology/cosmology_bao_xi_from_catalogs_*_metrics.json",
            "coordinate_spec_common": coordinate_spec_common,
            "estimator_spec_common": estimator_spec_common,
            "sample": sample,
            "caps": caps,
            "dists": dists,
            "require_zbin": bool(args.require_zbin),
            "out_tag": str(args.out_tag),
            "output_suffix": str(args.output_suffix),
            "covariance": {"source": str(args.cov_source), "ross_dir": str(args.ross_dir), "ross_bincent": int(args.ross_bincent)},
            "n_cases": int(len(records)),
        },
        "policy": {
            "note": "smooth+peak peakfit. Phase A uses diag proxy cov; Phase B may use Ross 2016 full cov for dv=[xi0,xi2].",
            "status_metric": "|eps|/sigma_eps_1sigma (profile CI; scaled chi2)",
            "edge_policy": "if best-fit/CI is clipped by eps scan bounds => mark as NG with abs_sigma >= mixed_max+1 (lower bound)",
            "status_thresholds": {"ok_max": float(args.ok_max), "mixed_max": float(args.mixed_max)},
        },
        "fit_config": {
            "r_range_mpc_h": [r_min, r_max],
            "template_peak": {"r0_mpc_h": r0, "sigma_mpc_h": sigma},
            "smooth_basis": {"power_max": int(smooth_power_max), "terms": _peakfit._smooth_basis_labels(smooth_power_max=smooth_power_max)},
            "grid": {
                "alpha": {"min": float(args.alpha_min), "max": float(args.alpha_max), "step": float(args.alpha_step)},
                "eps": {"min": float(args.eps_min), "max": float(args.eps_max), "step": float(args.eps_step)},
            },
            "grid_rescan": {"eps": {"factor": float(eps_rescan_factor), "max_expands": int(eps_rescan_max_expands)}},
            "mu_integral": {"mu_n": mu_n},
            "error_model": {"var_source": str(args.var_source), "min_pairs": float(args.min_pairs), "quad_weight": float(args.quad_weight)},
            "covariance": {"source": str(args.cov_source), "ross_dir": str(args.ross_dir), "ross_bincent": int(args.ross_bincent)},
        },
        "outputs": {"png": str(out_png), "metrics_json": str(out_json)},
        "results": records,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_bao_catalog_peakfit",
                "argv": sys.argv,
                "outputs": {"png": out_png, "metrics_json": out_json},
                "metrics": {
                    "sample": sample,
                    "caps": caps,
                    "dists": dists,
                    "require_zbin": bool(args.require_zbin),
                    "out_tag": str(args.out_tag),
                    "output_suffix": str(args.output_suffix),
                },
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
