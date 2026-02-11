#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_xi_from_desi_dr1_vac_lya_correlations.py

Phase 4 / Step 4.5B.21.4.4.7（DESI DR1 multi-tracer; Lya QSO）:
DESI DR1 VAC "lya-correlations"（Y1 BAO; Lyα forest auto/cross）の 2D 相関（RP,RT,DA）と
公開の full covariance（15000x15000）から、
本リポジトリの peakfit 入口（ξ0/ξ2 + covariance）へ落とす。

方針（最小）:
- 相関関数は (r_parallel=RP, r_transverse=RT) の格子点（row=NP*NT）上の値 DA として与えられる。
- multipoles 推定は 2 通り：
  - project_mode=wls（既定）：
    各 s-bin（[30,35,...,150]）ごとに、xi(s,mu) ≈ xi0(s) + xi2(s) P2(mu) の重み付き最小二乗で
    multipoles (xi0, xi2) を推定する（weight=NB or uniform）。
  - project_mode=mu_bin：
    mu∈[0,1] の等間隔bin（nmu）へ一度圧縮し、Riemann midpoint の Legendre 積分で (xi0,xi2) を計算する
    （gridのmu分布に依存しにくい推定量に寄せる）。
- full covariance（bins=15000=2500+2500+5000+5000）を、上記の線形射影で multipole cov へ伝播する。
- P_bg（pbg）座標は、fid LCDM→P_bg の AP スケール（z_ref で評価）で (RP,RT) を rescale して作る（近似）。

出力（cosmology_bao_xi_from_catalogs と互換の metrics/npz 形式で保存）:
- output/private/cosmology/cosmology_bao_xi_from_catalogs_lya_qso_combined_<dist>_zmin1p77_zmax4p16__<out_tag>.npz
- output/private/cosmology/cosmology_bao_xi_from_catalogs_lya_qso_combined_<dist>_...__<out_tag>_metrics.json
- output/private/cosmology/...__<out_tag>__vac_cov.npz

注意：
- 本スクリプトは Corrfunc/pycorr を使わない（Windows で実行可）。
- pbg への変換は「bin center の AP rescale」近似であり、厳密な再推定量（raw再計算）ではない。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import sparse  # type: ignore
except Exception:  # pragma: no cover
    sparse = None

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cosmology.boss_dr12v5_fits import read_bintable_columns, read_first_bintable_layout  # noqa: E402
from scripts.summary import worklog  # noqa: E402


@dataclass(frozen=True)
class _Component:
    key: str
    file: str
    n_rows: int


_COMPONENTS = [
    _Component(key="lyaxlya", file="cf_lya_x_lya_exp.fits", n_rows=2500),
    _Component(key="lyaxlyb", file="cf_lya_x_lyb_exp.fits", n_rows=2500),
    _Component(key="qsoxlya", file="cf_qso_x_lya_exp.fits", n_rows=5000),
    _Component(key="qsoxlyb", file="cf_qso_x_lyb_exp.fits", n_rows=5000),
]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relpath(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _p2(mu: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    return 0.5 * (3.0 * mu * mu - 1.0)


def _fmt_tag(x: float) -> str:
    s = f"{float(x):.6g}".rstrip("0").rstrip(".")
    if s == "":
        s = "0"
    return s.replace("-", "m").replace(".", "p")


def _dm_lcdm_dimless(z: float, *, omega_m: float, n_grid: int) -> float:
    z = float(z)
    if z < 0:
        return float("nan")
    if not (0.0 < float(omega_m) < 1.0):
        return float("nan")
    z_grid = np.linspace(0.0, z, int(n_grid), dtype=float)
    one_p = 1.0 + z_grid
    ez_grid = np.sqrt(float(omega_m) * one_p**3 + (1.0 - float(omega_m)))
    integrand = 1.0 / ez_grid
    try:
        integral = float(np.trapezoid(integrand, z_grid))
    except AttributeError:
        integral = float(np.trapz(integrand, z_grid))
    return float(integral)


def _dh_lcdm_dimless(z: float, *, omega_m: float) -> float:
    z = float(z)
    if z < 0:
        return float("nan")
    ez = float(math.sqrt(float(omega_m) * (1.0 + z) ** 3 + (1.0 - float(omega_m))))
    return float(1.0 / ez) if ez > 0 else float("nan")


def _dm_pbg_dimless(z: float) -> float:
    op = 1.0 + float(z)
    if not (op > 0.0):
        return float("nan")
    return float(math.log(op))


def _dh_pbg_dimless(z: float) -> float:
    op = 1.0 + float(z)
    if not (op > 0.0):
        return float("nan")
    return float(1.0 / op)


def _read_omegam_from_cf_header(path: Path) -> float:
    # Minimal FITS header scan: primary header + first extension header.
    CARD = 80
    BLOCK = 2880

    def read_header_blocks(f) -> bytes:
        chunks: List[bytes] = []
        while True:
            block = f.read(BLOCK)
            if len(block) != BLOCK:
                raise EOFError("unexpected EOF while reading FITS header")
            chunks.append(block)
            for i in range(0, BLOCK, CARD):
                card = block[i : i + CARD].decode("ascii", errors="ignore")
                if card.startswith("END"):
                    return b"".join(chunks)

    def iter_cards(hdr: bytes) -> Iterable[str]:
        for i in range(0, len(hdr), CARD):
            yield hdr[i : i + CARD].decode("ascii", errors="ignore")

    with path.open("rb") as f:
        _ = read_header_blocks(f)
        h1 = read_header_blocks(f)
    for card in iter_cards(h1):
        if card.startswith("END"):
            break
        if card[:8].strip() != "OMEGAM":
            continue
        if "=" not in card:
            continue
        rhs = card.split("=", 1)[1].split("/", 1)[0].strip()
        try:
            return float(rhs)
        except Exception:
            return float("nan")
    return float("nan")


def _load_components(*, raw_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[int]]:
    """
    Return global arrays (DA, RP, RT, Z, NB) concatenated in the canonical order
    and metadata: component_keys, component_offsets.
    """
    da_all: List[np.ndarray] = []
    rp_all: List[np.ndarray] = []
    rt_all: List[np.ndarray] = []
    z_all: List[np.ndarray] = []
    nb_all: List[np.ndarray] = []
    keys: List[str] = []
    offsets: List[int] = []
    off = 0
    for comp in _COMPONENTS:
        path = raw_dir / comp.file
        if not path.exists():
            raise SystemExit(f"missing VAC cf file: {path}")
        with path.open("rb") as f:
            layout = read_first_bintable_layout(f)
            if int(layout.n_rows) != int(comp.n_rows):
                raise SystemExit(f"unexpected NAXIS2 for {path.name}: {layout.n_rows} (expected {comp.n_rows})")
            cols = read_bintable_columns(f, layout=layout, columns=["DA", "RP", "RT", "Z", "NB"], max_rows=None)
        da = np.asarray(cols["DA"], dtype=float).reshape(-1)
        rp = np.asarray(cols["RP"], dtype=float).reshape(-1)
        rt = np.asarray(cols["RT"], dtype=float).reshape(-1)
        z = np.asarray(cols["Z"], dtype=float).reshape(-1)
        nb = np.asarray(cols["NB"], dtype=float).reshape(-1)
        if not (da.size == rp.size == rt.size == z.size == nb.size == int(comp.n_rows)):
            raise SystemExit(f"size mismatch in {path.name}")
        da_all.append(da)
        rp_all.append(rp)
        rt_all.append(rt)
        z_all.append(z)
        nb_all.append(nb)
        keys.append(comp.key)
        offsets.append(int(off))
        off += int(comp.n_rows)
    return (
        np.concatenate(da_all),
        np.concatenate(rp_all),
        np.concatenate(rt_all),
        np.concatenate(z_all),
        np.concatenate(nb_all),
        keys,
        offsets,
    )


def _covariance_submatrix(
    *,
    cov_path: Path,
    sel_idx: np.ndarray,
    chunk_rows: int,
) -> np.ndarray:
    if sparse is None:  # pragma: no cover
        raise SystemExit("scipy is required for this script (scipy.sparse)")
    sel_idx = np.asarray(sel_idx, dtype=np.int32).reshape(-1)
    if sel_idx.size == 0:
        raise ValueError("empty sel_idx")

    # Build row mapping for fast filtering (full matrix is 15000x15000).
    n_total = int(np.max(sel_idx)) + 1
    sel_pos = np.full(n_total, -1, dtype=np.int32)
    for j, i in enumerate(sel_idx.tolist()):
        sel_pos[int(i)] = int(j)

    k = int(sel_idx.size)
    cov_sel = np.empty((k, k), dtype=np.float64)

    with cov_path.open("rb") as f:
        layout = read_first_bintable_layout(f)
        n_rows = int(layout.n_rows)
        row_bytes = int(layout.row_bytes)
        if n_rows <= 0 or row_bytes <= 0:
            raise SystemExit(f"invalid covariance layout: n_rows={n_rows} row_bytes={row_bytes}")
        if row_bytes % 8 != 0:
            raise SystemExit(f"unexpected covariance row_bytes (not multiple of 8): {row_bytes}")
        n_cols = int(row_bytes // 8)
        if n_rows != n_cols:
            raise SystemExit(f"expected square covariance (n_rows={n_rows}, n_cols={n_cols})")
        if n_cols <= int(np.max(sel_idx)):
            raise SystemExit(f"sel_idx out of bounds for covariance: max(sel_idx)={int(np.max(sel_idx))} >= {n_cols}")

        i0 = 0
        while i0 < n_rows:
            n_chunk = min(int(chunk_rows), n_rows - i0)
            b = f.read(n_chunk * row_bytes)
            if len(b) != n_chunk * row_bytes:
                raise EOFError("unexpected EOF while reading covariance table")
            arr = np.frombuffer(b, dtype=">f8", count=n_chunk * n_cols).reshape(n_chunk, n_cols)
            arr = np.asarray(arr, dtype=np.float64)

            rows = np.arange(i0, i0 + n_chunk, dtype=np.int32)
            # Avoid out-of-bounds indexing: only consult sel_pos for in-range rows.
            in_bounds = rows < int(sel_pos.size)
            m = np.zeros(rows.shape, dtype=bool)
            if bool(np.any(in_bounds)):
                m[in_bounds] = sel_pos[rows[in_bounds]] >= 0
            if bool(np.any(m)):
                src_rows = np.nonzero(m)[0].astype(np.int32, copy=False)
                dst_rows = sel_pos[rows[m]].astype(np.int32, copy=False)
                block = arr[src_rows][:, sel_idx]
                cov_sel[dst_rows, :] = block
            i0 += n_chunk

    cov_sel = 0.5 * (cov_sel + cov_sel.T)
    return cov_sel


def _fit_multipoles_and_T(
    *,
    rp: np.ndarray,
    rt: np.ndarray,
    da: np.ndarray,
    nb: np.ndarray,
    keys: List[str],
    offsets: List[int],
    sel_pos: np.ndarray,
    s_centers: np.ndarray,
    s_edges: np.ndarray,
    weight_mode: str,
    project_mode: str,
    nmu: int,
) -> Tuple[np.ndarray, np.ndarray, "sparse.csr_matrix", np.ndarray]:
    if sparse is None:  # pragma: no cover
        raise SystemExit("scipy is required for this script (scipy.sparse)")
    rp = np.asarray(rp, dtype=float).reshape(-1)
    rt = np.asarray(rt, dtype=float).reshape(-1)
    da = np.asarray(da, dtype=float).reshape(-1)
    nb = np.asarray(nb, dtype=float).reshape(-1)
    if not (rp.size == rt.size == da.size == nb.size):
        raise ValueError("rp/rt/da/nb size mismatch")

    n_total = int(rp.size)
    n_comp = int(len(keys))
    n_s = int(np.asarray(s_centers, dtype=float).size)
    if n_comp <= 0 or n_s <= 0:
        raise ValueError("invalid n_comp/n_s")

    s = np.sqrt(rp * rp + rt * rt)
    mu = np.where(s > 0.0, rp / s, 0.0)
    p2 = _p2(mu)

    mode = str(project_mode).strip().lower()
    if mode not in {"wls", "mu_bin"}:
        raise ValueError("project_mode must be 'wls' or 'mu_bin'")
    nmu = int(nmu)
    if mode == "mu_bin" and nmu < 4:
        raise ValueError("nmu must be >= 4 for project_mode=mu_bin")
    if mode == "mu_bin":
        mu_abs = np.abs(mu)
        mu_abs = np.clip(mu_abs, 0.0, 1.0)
        mu_edges = np.linspace(0.0, 1.0, int(nmu) + 1, dtype=float)
        mu_mid = 0.5 * (mu_edges[:-1] + mu_edges[1:])
        p2_mid = _p2(mu_mid)
        dmu = float(1.0 / float(nmu))

    # Output arrays.
    xi0 = np.full((n_comp, n_s), float("nan"), dtype=float)
    xi2 = np.full((n_comp, n_s), float("nan"), dtype=float)
    weight_sums = np.zeros((n_comp, n_s), dtype=float)

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    for c in range(n_comp):
        off = int(offsets[c])
        n_rows_c = int((_COMPONENTS[c].n_rows))
        g_idx = off + np.arange(n_rows_c, dtype=np.int32)
        # Only bins that are present in sel_pos (union selection) can be used.
        ok = (g_idx < int(sel_pos.size)) & (sel_pos[g_idx] >= 0)
        if not bool(np.any(ok)):
            raise ValueError(f"no selected points for component {keys[c]}")
        g_idx_ok = g_idx[ok]

        # Local views for speed.
        s_c = s[g_idx_ok]
        da_c = da[g_idx_ok]
        nb_c = nb[g_idx_ok]
        p2_c = p2[g_idx_ok]
        mu_abs_c = mu_abs[g_idx_ok] if mode == "mu_bin" else None
        col_c = sel_pos[g_idx_ok].astype(np.int32, copy=False)

        for k in range(n_s):
            lo = float(s_edges[k])
            hi = float(s_edges[k + 1])
            if k < n_s - 1:
                m_bin = (s_c >= lo) & (s_c < hi)
            else:
                m_bin = (s_c >= lo) & (s_c <= hi)
            if int(np.count_nonzero(m_bin)) < 3:
                continue

            y = da_c[m_bin]
            cols_k = col_c[m_bin].astype(np.int32, copy=False)

            if str(weight_mode) == "nb":
                w = np.maximum(nb_c[m_bin], 0.0)
            elif str(weight_mode) == "uniform":
                w = np.ones_like(y)
            else:
                raise ValueError("weight_mode must be 'nb' or 'uniform'")
            weight_sums[c, k] = float(np.sum(w))

            if mode == "wls":
                p2v = p2_c[m_bin]
                # Normalize weights to improve conditioning.
                w = w / max(1e-300, float(np.mean(w)))

                # Design: xi = a0*1 + a2*P2
                m0 = np.ones_like(p2v)
                # Normal equations A x = b
                a00 = float(np.sum(w * m0 * m0))
                a02 = float(np.sum(w * m0 * p2v))
                a22 = float(np.sum(w * p2v * p2v))
                b0 = float(np.sum(w * m0 * y))
                b2 = float(np.sum(w * p2v * y))
                A = np.array([[a00, a02], [a02, a22]], dtype=float)
                b = np.array([b0, b2], dtype=float)
                try:
                    x = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                xi0[c, k] = float(x[0])
                xi2[c, k] = float(x[1])

                # Coefficients: x = inv(A) @ (M^T W y)
                try:
                    invA = np.linalg.inv(A)
                except np.linalg.LinAlgError:
                    invA = np.linalg.pinv(A, rcond=1e-12)
                # M^T W is 2xK: [w*1, w*P2]
                mw0 = w * m0
                mw2 = w * p2v
                coeff0 = invA[0, 0] * mw0 + invA[0, 1] * mw2
                coeff2 = invA[1, 0] * mw0 + invA[1, 1] * mw2
            else:
                # project_mode=mu_bin: compress to uniform mu bins in [0,1], then Legendre integral (Riemann midpoint).
                # Map each point to a mu-bin.
                mu_bin = np.minimum((mu_abs_c[m_bin] * float(nmu)).astype(np.int32), int(nmu) - 1)
                denom = np.bincount(mu_bin, weights=w, minlength=int(nmu)).astype(float, copy=False)
                denom_p = denom[mu_bin]
                denom_safe = np.where(denom_p > 0.0, denom_p, 1.0)
                base = w / denom_safe
                coeff0 = float(dmu) * base
                coeff2 = float(5.0 * dmu) * p2_mid[mu_bin] * base
                xi0[c, k] = float(np.sum(coeff0 * y))
                xi2[c, k] = float(np.sum(coeff2 * y))

            # Output row indices (component-major blocks).
            row0 = int((2 * c + 0) * n_s + k)
            row2 = int((2 * c + 1) * n_s + k)
            rows.extend([row0] * int(cols_k.size))
            cols.extend(cols_k.tolist())
            vals.extend([float(v) for v in np.asarray(coeff0, dtype=float).tolist()])
            rows.extend([row2] * int(cols_k.size))
            cols.extend(cols_k.tolist())
            vals.extend([float(v) for v in np.asarray(coeff2, dtype=float).tolist()])

    m = int(2 * n_comp * n_s)
    k = int(np.max(sel_pos) + 1)
    T = sparse.csr_matrix((np.asarray(vals, dtype=float), (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))), shape=(m, k))
    return xi0, xi2, T, weight_sums


def _combine_components(
    *,
    xi0: np.ndarray,
    xi2: np.ndarray,
    cov_m: np.ndarray,
    weight_sums: np.ndarray,
    keys: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Combine multi-component multipoles into a single effective component per s-bin.

    Definition:
      xiℓ_comb(s_k) = Σ_c w_ck xiℓ_c(s_k) / Σ_c w_ck
    with covariance propagated as cov_comb = B cov_m B^T.
    """
    xi0 = np.asarray(xi0, dtype=float)
    xi2 = np.asarray(xi2, dtype=float)
    cov_m = np.asarray(cov_m, dtype=float)
    weight_sums = np.asarray(weight_sums, dtype=float)
    if xi0.ndim != 2 or xi2.ndim != 2:
        raise ValueError("xi0/xi2 must be 2D [n_comp,n_s]")
    if xi0.shape != xi2.shape or xi0.shape != weight_sums.shape:
        raise ValueError("xi0/xi2/weight_sums shape mismatch")
    n_comp, n_s = xi0.shape
    p = int(2 * n_comp * n_s)
    if cov_m.shape != (p, p):
        raise ValueError(f"cov_m shape mismatch: {cov_m.shape} (expected {(p, p)})")

    w = weight_sums.copy()
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    w_tot = np.sum(w, axis=0)
    missing = np.nonzero(w_tot <= 0.0)[0].astype(int, copy=False)
    if int(missing.size) > 0:
        w[:, missing] = 1.0
        w_tot = np.sum(w, axis=0)
    w_norm = w / w_tot.reshape(1, -1)

    xi0_c = np.sum(w_norm * xi0, axis=0, keepdims=True)
    xi2_c = np.sum(w_norm * xi2, axis=0, keepdims=True)

    # Linear map B: y_comb = B y, where y stacks [xi0_c0(s),xi2_c0(s),xi0_c1(s),...].
    B = np.zeros((2 * n_s, p), dtype=float)
    for k in range(n_s):
        for c in range(n_comp):
            wc = float(w_norm[c, k])
            B[k, (2 * c + 0) * n_s + k] = wc
            B[n_s + k, (2 * c + 1) * n_s + k] = wc
    cov_c = B @ cov_m @ B.T
    cov_c = 0.5 * (cov_c + cov_c.T)

    meta: Dict[str, Any] = {
        "mode": "weighted_average_per_s",
        "components_in": list(keys),
        "components_out": ["combined"],
        "weights": {
            "weight_sums": weight_sums.tolist(),
            "weight_norm": w_norm.tolist(),
        },
    }
    return xi0_c, xi2_c, cov_c, meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="DESI DR1 VAC lya-correlations: project to xi0/xi2 + cov for peakfit.")
    ap.add_argument(
        "--data-dir",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "desi_dr1_vac_lya_correlations_v1p0"),
        help="Input data dir (default: data/cosmology/desi_dr1_vac_lya_correlations_v1p0)",
    )
    ap.add_argument("--out-tag", type=str, default="w_desi_vac_lya_corr_v1p0", help="Output tag (default: w_desi_vac_lya_corr_v1p0)")
    ap.add_argument("--sample", type=str, default="lya_qso", help="Sample name for peakfit pipeline (default: lya_qso)")
    ap.add_argument("--caps", type=str, default="combined", help="Caps label (default: combined)")
    ap.add_argument("--z-min", type=float, default=1.77, help="z_min label for filenames (default: 1.77)")
    ap.add_argument("--z-max", type=float, default=4.16, help="z_max label for filenames (default: 4.16)")
    ap.add_argument("--z-ref", type=float, default=2.33, help="z_ref used for fid LCDM->PBG AP rescale (default: 2.33)")
    ap.add_argument("--lcdm-omega-m", type=float, default=float("nan"), help="Omega_m for fid LCDM (default: read from header; fallback 0.315)")
    ap.add_argument("--lcdm-n-grid", type=int, default=4000, help="LCDM integral grid (default: 4000)")
    ap.add_argument(
        "--dists",
        type=str,
        default="lcdm,pbg",
        help="Comma-separated dist list to generate (default: lcdm,pbg)",
    )
    ap.add_argument("--weight-mode", type=str, default="nb", choices=["nb", "uniform"], help="Multipole fit weight mode (default: nb)")
    ap.add_argument(
        "--combine-components",
        action="store_true",
        help="Combine the 4 VAC correlation components into a single effective component per s-bin (default: off).",
    )
    ap.add_argument(
        "--project-mode",
        type=str,
        default="wls",
        choices=["wls", "mu_bin"],
        help="Multipole projection mode: wls (default) or mu_bin (uniform mu-bin integral).",
    )
    ap.add_argument("--nmu", type=int, default=120, help="mu bins for project_mode=mu_bin (default: 120)")
    ap.add_argument("--cov-chunk-rows", type=int, default=64, help="Covariance read chunk rows (default: 64)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if sparse is None:
        raise SystemExit("scipy is required (scipy.sparse)")

    data_dir = Path(str(args.data_dir)).resolve()
    raw_dir = data_dir / "raw"
    cov_path = raw_dir / "full-covariance-smoothed.fits"
    if not cov_path.exists():
        raise SystemExit(f"missing covariance file: {cov_path} (run fetch_desi_dr1_vac_lya_correlations.py)")

    # Load scalar data from cf_* (concatenated in the official order).
    da, rp_lcdm, rt_lcdm, z, nb, comp_keys, comp_offsets = _load_components(raw_dir=raw_dir)
    if da.size != 15000:
        raise SystemExit(f"unexpected total bins: {da.size} (expected 15000)")

    omega_m = float(args.lcdm_omega_m)
    if not (math.isfinite(omega_m) and 0.0 < omega_m < 1.0):
        omega_m = _read_omegam_from_cf_header(raw_dir / _COMPONENTS[2].file)
    if not (math.isfinite(omega_m) and 0.0 < omega_m < 1.0):
        omega_m = 0.315

    z_ref = float(args.z_ref)
    # Compute LCDM distances on a grid and interpolate per-bin AP rescale vs the fiducial LCDM coordinates.
    z_grid_max = float(max(float(args.z_max), float(np.max(z)), float(z_ref)))
    n_grid = int(args.lcdm_n_grid)
    if n_grid < 16:
        raise SystemExit("--lcdm-n-grid must be >= 16 for stable interpolation")

    z_grid = np.linspace(0.0, z_grid_max, n_grid, dtype=float)
    one_p = 1.0 + z_grid
    ez_grid = np.sqrt(float(omega_m) * one_p**3 + (1.0 - float(omega_m)))
    integrand = 1.0 / ez_grid
    dz = np.diff(z_grid)
    dm_grid = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dz)])

    dm_lcdm_z = np.interp(np.asarray(z, dtype=float), z_grid, dm_grid)
    dh_lcdm_z = 1.0 / np.sqrt(float(omega_m) * (1.0 + np.asarray(z, dtype=float)) ** 3 + (1.0 - float(omega_m)))
    dm_pbg_z = np.log(1.0 + np.asarray(z, dtype=float))
    dh_pbg_z = 1.0 / (1.0 + np.asarray(z, dtype=float))

    alpha_perp_z = np.where((dm_lcdm_z > 0.0) & (dm_pbg_z > 0.0), dm_pbg_z / dm_lcdm_z, float("nan"))
    alpha_par_z = np.where((dh_lcdm_z > 0.0) & (dh_pbg_z > 0.0), dh_pbg_z / dh_lcdm_z, float("nan"))
    if not bool(np.all(np.isfinite(alpha_perp_z) & (alpha_perp_z > 0.0))):
        raise SystemExit("invalid per-bin alpha_perp_z (check z range / omega_m)")
    if not bool(np.all(np.isfinite(alpha_par_z) & (alpha_par_z > 0.0))):
        raise SystemExit("invalid per-bin alpha_par_z (check z range / omega_m)")

    # Reference values (for logging).
    dm_lcdm_ref = float(np.interp(float(z_ref), z_grid, dm_grid))
    dh_lcdm_ref = _dh_lcdm_dimless(z_ref, omega_m=omega_m)
    dm_pbg_ref = _dm_pbg_dimless(z_ref)
    dh_pbg_ref = _dh_pbg_dimless(z_ref)
    alpha_perp_ref = (dm_pbg_ref / dm_lcdm_ref) if (dm_lcdm_ref > 0 and dm_pbg_ref > 0) else float("nan")
    alpha_par_ref = (dh_pbg_ref / dh_lcdm_ref) if (dh_lcdm_ref > 0 and dh_pbg_ref > 0) else float("nan")

    # Define standard s bins (same as Corrfunc bins used elsewhere in this repo).
    s_edges = np.arange(30.0, 150.0 + 5.0, 5.0, dtype=float)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])

    # Union selection (so we only read covariance once).
    s_lcdm = np.sqrt(rp_lcdm * rp_lcdm + rt_lcdm * rt_lcdm)
    rp_pbg = rp_lcdm * alpha_par_z
    rt_pbg = rt_lcdm * alpha_perp_z
    s_pbg = np.sqrt(rp_pbg * rp_pbg + rt_pbg * rt_pbg)
    in_lcdm = (s_lcdm >= float(s_edges[0])) & (s_lcdm <= float(s_edges[-1]))
    in_pbg = (s_pbg >= float(s_edges[0])) & (s_pbg <= float(s_edges[-1]))
    union = in_lcdm | in_pbg
    sel_idx = np.nonzero(union)[0].astype(np.int32, copy=False)
    sel_idx.sort()
    sel_pos = np.full(int(da.size), -1, dtype=np.int32)
    sel_pos[sel_idx] = np.arange(int(sel_idx.size), dtype=np.int32)

    # Extract covariance submatrix for selected points.
    cov_sel = _covariance_submatrix(cov_path=cov_path, sel_idx=sel_idx, chunk_rows=int(args.cov_chunk_rows))

    # Pre-slice vectors for selected points.
    da_sel = da[sel_idx]
    nb_sel = nb[sel_idx]

    out_dir = _ROOT / "output" / "private" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    zmin_tag = _fmt_tag(float(args.z_min))
    zmax_tag = _fmt_tag(float(args.z_max))

    dists = [d.strip() for d in str(args.dists).split(",") if d.strip()]
    for dist in dists:
        if dist not in {"lcdm", "pbg"}:
            raise SystemExit(f"unsupported dist: {dist} (expected lcdm or pbg)")

        if dist == "lcdm":
            rp_use = rp_lcdm
            rt_use = rt_lcdm
            coordinate_spec = {
                "source": "desi_dr1_vac_lya_correlations_v1p0",
                "fid": "lcdm",
                "z_ref_for_ap_rescale": float(z_ref),
                "lcdm": {"omega_m": float(omega_m), "n_grid": int(args.lcdm_n_grid)},
            }
        else:
            rp_use = rp_pbg
            rt_use = rt_pbg
            coordinate_spec = {
                "source": "desi_dr1_vac_lya_correlations_v1p0",
                "fid": "pbg",
                "z_ref_for_ap_rescale": float(z_ref),
                "ap_rescale_vs_lcdm": {
                    "mode": "per_bin_z",
                    "z_ref": float(z_ref),
                    "alpha_perp_at_z_ref": float(alpha_perp_ref),
                    "alpha_par_at_z_ref": float(alpha_par_ref),
                    "alpha_perp_stats": {
                        "min": float(np.min(alpha_perp_z)),
                        "max": float(np.max(alpha_perp_z)),
                        "mean": float(np.mean(alpha_perp_z)),
                    },
                    "alpha_par_stats": {
                        "min": float(np.min(alpha_par_z)),
                        "max": float(np.max(alpha_par_z)),
                        "mean": float(np.mean(alpha_par_z)),
                    },
                },
                "lcdm": {"omega_m": float(omega_m), "n_grid": int(args.lcdm_n_grid)},
            }

        # Multipole projection and coefficient matrix T (for covariance propagation).
        xi0, xi2, T, weight_sums = _fit_multipoles_and_T(
            rp=rp_use,
            rt=rt_use,
            da=da,
            nb=nb,
            keys=comp_keys,
            offsets=comp_offsets,
            sel_pos=sel_pos,
            s_centers=s_centers,
            s_edges=s_edges,
            weight_mode=str(args.weight_mode),
            project_mode=str(args.project_mode),
            nmu=int(args.nmu),
        )
        # Covariance propagation: cov_m = T * cov_sel * T^T
        U = T @ cov_sel
        cov_m = U @ T.T
        cov_m = np.asarray(cov_m, dtype=np.float64)
        cov_m = 0.5 * (cov_m + cov_m.T)

        comp_keys_out = list(comp_keys)
        combine_components_meta: Optional[Dict[str, Any]] = None
        if bool(args.combine_components):
            xi0, xi2, cov_m, combine_components_meta = _combine_components(
                xi0=xi0,
                xi2=xi2,
                cov_m=cov_m,
                weight_sums=weight_sums,
                keys=comp_keys,
            )
            # peakfit expects 1D xi arrays for n_components=1.
            xi0 = np.asarray(xi0, dtype=float).reshape(-1)
            xi2 = np.asarray(xi2, dtype=float).reshape(-1)
            comp_keys_out = ["combined"]

        # File names (xi-from-catalogs compatible).
        stem = f"cosmology_bao_xi_from_catalogs_{str(args.sample)}_{str(args.caps)}_{dist}_zmin{zmin_tag}_zmax{zmax_tag}__{str(args.out_tag)}"
        out_npz = out_dir / f"{stem}.npz"
        out_metrics = out_dir / f"{stem}_metrics.json"
        out_cov = out_dir / f"{stem}__vac_cov.npz"

        # Write xi npz.
        np.savez(
            out_npz,
            s=s_centers.astype(np.float64),
            xi0=xi0.astype(np.float64),
            xi2=xi2.astype(np.float64),
            components=np.asarray(comp_keys_out, dtype="U"),
            mu_edges=np.asarray([], dtype=np.float64),
            dd_w=np.asarray([], dtype=np.float64),
            dr_w=np.asarray([], dtype=np.float64),
            rr_w=np.asarray([], dtype=np.float64),
        )

        # Write covariance npz in the same convention used by peakfit (y=[xi0,xi2] but component-major blocks).
        np.savez(
            out_cov,
            s=s_centers.astype(np.float64),
            cov=cov_m.astype(np.float64),
            components=np.asarray(comp_keys_out, dtype="U"),
            sel_bins=int(sel_idx.size),
            sel_bins_sha256=_sha256_bytes(sel_idx.tobytes()),
        )

        # Minimal metrics to make cosmology_bao_catalog_peakfit.py pick it up.
        # - derived.z_eff_gal_weighted is used to match Y1data z_eff (tolerance is ~0.02).
        z_eff_used = float(z_ref)
        params: Dict[str, Any] = {
            "sample": str(args.sample),
            "caps": str(args.caps),
            "distance_model": str(dist),
            "z_source": "desi_dr1_vac_lya_correlations",
            "los": "rp_rt_grid",
            "weight_scheme": "desi_lya_vac",
            "out_tag": str(args.out_tag),
            "z_cut": {"bin": "none"},
            "z_min": float(args.z_min),
            "z_max": float(args.z_max),
            "lcdm_omega_m": float(omega_m),
            "lcdm_n_grid": int(args.lcdm_n_grid),
            "lcdm_z_grid_max": float(args.z_max),
            "coordinate_spec": coordinate_spec,
            "estimator_spec_hash": _sha256_bytes(
                json.dumps(
                    {
                        "source": "desi_dr1_vac_lya_correlations_v1p0",
                        "components": comp_keys,
                        "combine_components": bool(args.combine_components),
                        "s_edges": s_edges.tolist(),
                        "weight_mode": str(args.weight_mode),
                        "project_mode": str(args.project_mode),
                        "nmu": int(args.nmu),
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ),
        }
        derived = {
            "z_eff_gal_weighted": z_eff_used,
            "z_eff_data_mean": float(np.mean(z)),
            "z_eff_data_nb_weighted_mean": float(np.sum(z * nb) / max(1e-300, float(np.sum(nb)))),
        }
        payload: Dict[str, Any] = {
            "generated_utc": _now_utc(),
            "domain": "cosmology",
            "step": "4.5B.21.4.4.7 (DESI DR1 VAC lya-correlations -> xi multipoles)",
            "inputs": {
                "data_dir": _relpath(data_dir),
                "cov_path": _relpath(cov_path),
                "cf_files": {c.key: _relpath(raw_dir / c.file) for c in _COMPONENTS},
            },
            "params": params,
            "derived": derived,
            "outputs": {"npz": _relpath(out_npz), "metrics_json": _relpath(out_metrics), "cov_npz": _relpath(out_cov)},
            "stats": {"sel_bins": int(sel_idx.size), "cov_sel_shape": [int(x) for x in cov_sel.shape], "cov_m_shape": [int(x) for x in cov_m.shape]},
            "notes": [
                "xi0/xi2 は project_mode（wls / mu_bin）で VAC 2D correlation（RP,RT,DA）から線形射影したもの。",
                "pbg は fid LCDM->PBG の AP rescale を bin の Z 列で評価して (RP,RT) に適用した近似（per-bin z）。",
                "cov は full-covariance-smoothed.fits（15000x15000）を同一射影で伝播したもの。",
            ],
        }
        if combine_components_meta is not None:
            payload["combine_components"] = combine_components_meta
        out_metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        try:
            worklog.append_event(
                {
                    "domain": "cosmology",
                    "action": "desi_dr1_vac_lya_correlations_xi_multipoles",
                    "inputs": [cov_path],
                    "outputs": [out_npz, out_cov, out_metrics],
                    "params": {"dist": dist, "out_tag": str(args.out_tag), "sample": str(args.sample), "caps": str(args.caps)},
                }
            )
        except Exception:
            pass

        print(f"[ok] xi npz : {out_npz}")
        print(f"[ok] cov npz: {out_cov}")
        print(f"[ok] metrics: {out_metrics}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
