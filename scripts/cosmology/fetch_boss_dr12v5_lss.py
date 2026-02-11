#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_boss_dr12v5_lss.py

Phase 16（宇宙論）/ Step 16.4（BAO一次統計の再導出）:
BOSS DR12v5 LSS（銀河 catalog + ランダム catalog）を一次入力として取得し、
RA/DEC/z/weight を軽量フォーマット（npz）へ抽出してキャッシュする。

背景：
- BAO圧縮出力（D_M/r_d 等）は、観測統計→距離変換→テンプレートfit を含む推定量。
- 次工程「(θ,z) → (s,μ) 変換を P-model 側で定義し、ξ(s,μ), ξℓ を再計算」へ向けて、
  まず一次入力（銀河+ランダム）を確定し、再現可能な形で保存する。

データソース（SAS）:
  - 推奨（高速ミラー）: https://dr12.sdss3.org/sas/dr12/boss/lss/
  - 旧（遅い場合あり）: https://data.sdss.org/sas/dr12/boss/lss/

出力（固定）:
  data/cosmology/boss_dr12v5_lss/
    - raw/               (galaxy fits.gz は原則保存)
    - extracted/         (RA/DEC/z/weight を npz に保存)
    - manifest.json      (取得メタ・抽出条件)

注意：
- random catalog は非常に大きい（GB級）。デフォルトでは「先頭N行」だけ抽出する。
  これは厳密な一様サンプルではない可能性があるため、幾何のスクリーニング用途に限定し、
  本番（Phase B）では別途「均一サンプル（全量 or reservoir）」に移行する。
- 先頭N行抽出は、galaxy/random で「含まれるセクタ（IPOLY/ISECT）」がずれると LS 推定が破綻し得る。
  そのため、抽出時点で IPOLY/ISECT も保存し、後段で同一セクタ集合にそろえてから ξ を計算する。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency for offline/local runs
    requests = None

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cosmology.boss_dr12v5_fits import (  # noqa: E402
    iter_bintable_column_chunks,
    read_bintable_columns,
    read_first_bintable_layout,
)
from scripts.summary import worklog  # noqa: E402


_BASE_URL = "https://dr12.sdss3.org/sas/dr12/boss/lss/"
_LEGACY_BASE_URL = "https://data.sdss.org/sas/dr12/boss/lss/"

_SAMPLE_FILES: Dict[str, Dict[str, Dict[str, str]]] = {
    "cmass": {
        "north": {
            "galaxy": "galaxy_DR12v5_CMASS_North.fits.gz",
            "random0": "random0_DR12v5_CMASS_North.fits.gz",
            "random1": "random1_DR12v5_CMASS_North.fits.gz",
        },
        "south": {
            "galaxy": "galaxy_DR12v5_CMASS_South.fits.gz",
            "random0": "random0_DR12v5_CMASS_South.fits.gz",
            "random1": "random1_DR12v5_CMASS_South.fits.gz",
        },
    },
    "lowz": {
        "north": {
            "galaxy": "galaxy_DR12v5_LOWZ_North.fits.gz",
            "random0": "random0_DR12v5_LOWZ_North.fits.gz",
            "random1": "random1_DR12v5_LOWZ_North.fits.gz",
        },
        "south": {
            "galaxy": "galaxy_DR12v5_LOWZ_South.fits.gz",
            "random0": "random0_DR12v5_LOWZ_South.fits.gz",
            "random1": "random1_DR12v5_LOWZ_South.fits.gz",
        },
    },
    # Combined catalogs (CMASS+LOWZ) as provided by BOSS DR12v5 LSS.
    # Note: we use these for redshift-bin studies (z≈0.38/0.51/0.61) by applying
    # z-cuts downstream in `cosmology_bao_xi_from_catalogs.py`, so we can keep NGC+SGC symmetry.
    "cmasslowz": {
        "north": {
            "galaxy": "galaxy_DR12v5_CMASSLOWZ_North.fits.gz",
            "random0": "random0_DR12v5_CMASSLOWZ_North.fits.gz",
            "random1": "random1_DR12v5_CMASSLOWZ_North.fits.gz",
        },
        "south": {
            "galaxy": "galaxy_DR12v5_CMASSLOWZ_South.fits.gz",
            "random0": "random0_DR12v5_CMASSLOWZ_South.fits.gz",
            "random1": "random1_DR12v5_CMASSLOWZ_South.fits.gz",
        },
    },
    "cmasslowztot": {
        "north": {
            "galaxy": "galaxy_DR12v5_CMASSLOWZTOT_North.fits.gz",
            "random0": "random0_DR12v5_CMASSLOWZTOT_North.fits.gz",
            "random1": "random1_DR12v5_CMASSLOWZTOT_North.fits.gz",
        },
        "south": {
            "galaxy": "galaxy_DR12v5_CMASSLOWZTOT_South.fits.gz",
            "random0": "random0_DR12v5_CMASSLOWZTOT_South.fits.gz",
            "random1": "random1_DR12v5_CMASSLOWZTOT_South.fits.gz",
        },
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _relpath(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _download_file(url: str, dst: Path) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is required to download remote files. Install it or place files under data/.../raw/")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
    tmp.replace(dst)
    return {"bytes": dst.stat().st_size, "sha256": _sha256(dst)}


def _extract_local_gz_to_npz(
    gz_path: Path,
    *,
    out_npz: Path,
    want_cols: list[str],
    max_rows: Optional[int],
) -> Dict[str, Any]:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(gz_path, "rb") as f:
        layout = read_first_bintable_layout(f)
        cols = read_bintable_columns(f, layout=layout, columns=want_cols, max_rows=max_rows)
    # Save as float64 arrays.
    np.savez_compressed(out_npz, **cols)
    meta = {
        "rows_total": int(layout.n_rows),
        "rows_saved": int(next(iter(cols.values())).size) if cols else 0,
        "row_bytes": int(layout.row_bytes),
        "columns": want_cols,
    }
    return meta


def _extract_local_gz_to_npz_prefix_rows(
    gz_path: Path,
    *,
    out_npz: Path,
    want_cols: list[str],
    max_rows: int,
    source_url: str,
) -> Dict[str, Any]:
    """
    Read only the first `max_rows` rows from a local .fits.gz and store selected columns into NPZ.
    Keeps extract meta consistent with the remote streaming variant.
    """
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(gz_path, "rb") as f:
        layout = read_first_bintable_layout(f)
        cols = read_bintable_columns(f, layout=layout, columns=want_cols, max_rows=int(max_rows))
    np.savez_compressed(out_npz, **cols)
    meta = {
        "rows_total": int(layout.n_rows),
        "rows_saved": int(next(iter(cols.values())).size) if cols else 0,
        "row_bytes": int(layout.row_bytes),
        "columns": want_cols,
        "sampling": {"method": "prefix_rows", "max_rows": int(max_rows)},
        "source_url": source_url,
        "source_path": _relpath(gz_path),
    }
    return meta


def _extract_local_gz_to_npz_reservoir(
    gz_path: Path,
    *,
    out_npz: Path,
    want_cols: list[str],
    sample_rows: int,
    seed: int,
    scan_max_rows: Optional[int],
    chunk_rows: int,
    source_url: str,
    cover_sectors: Optional[set[int]] = None,
    cover_on_col: str = "ISECT",
    cover_z_min: Optional[float] = None,
    cover_z_max: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Scan a local .fits.gz and reservoir-sample `sample_rows` rows (uniform without replacement) into NPZ.
    Keeps extract meta consistent with the remote streaming variant.
    """
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    cover_meta: Optional[Dict[str, Any]] = None
    with gzip.open(gz_path, "rb") as f:
        layout = read_first_bintable_layout(f)
        chunks = iter_bintable_column_chunks(
            f,
            layout=layout,
            columns=want_cols,
            max_rows=scan_max_rows,
            chunk_rows=int(chunk_rows),
        )
        cover_rows: Dict[int, Dict[str, float]] = {}
        cover_target: Optional[np.ndarray] = None
        cover_remaining: Optional[set[int]] = None
        if cover_sectors:
            cover_sectors_int = {int(s) for s in cover_sectors}
            if len(cover_sectors_int) > int(sample_rows):
                raise ValueError("cover_sectors must be <= sample_rows")
            if cover_on_col == "SECTOR_KEY":
                if "IPOLY" not in want_cols or "ISECT" not in want_cols:
                    raise ValueError("cover_on_col=SECTOR_KEY requires IPOLY and ISECT in want_cols")
            else:
                if cover_on_col not in want_cols:
                    raise ValueError(f"cover_on_col must be included in want_cols: {cover_on_col}")
            if (cover_z_min is not None) or (cover_z_max is not None):
                if "Z" not in want_cols:
                    raise ValueError("cover_z_min/max requires Z in want_cols")
                if (cover_z_min is not None) and (cover_z_max is not None) and not (float(cover_z_min) < float(cover_z_max)):
                    raise ValueError("cover_z_min must be < cover_z_max when both are provided")
            cover_target = np.fromiter(sorted(cover_sectors_int), dtype=np.int64)
            cover_remaining = set(cover_sectors_int)

            def _cover_keys(ch: Dict[str, Any]) -> np.ndarray:
                if cover_on_col == "SECTOR_KEY":
                    ip = np.asarray(ch["IPOLY"], dtype=np.float64)
                    isect = np.asarray(ch["ISECT"], dtype=np.float64)
                    valid = np.isfinite(ip) & np.isfinite(isect)
                    ip_i = ip.astype(np.int64, copy=False)
                    is_i = isect.astype(np.int64, copy=False)
                    out = np.full(int(ip.shape[0]), -1, dtype=np.int64)
                    out[valid] = (ip_i[valid] << 32) + (is_i[valid] & np.int64(0xFFFFFFFF))
                    return out
                v = np.asarray(ch[cover_on_col], dtype=np.float64)
                valid = np.isfinite(v)
                out = np.full(int(v.shape[0]), -1, dtype=np.int64)
                out[valid] = v[valid].astype(np.int64)
                return out

            def _cover_wrap(it: Any) -> Any:
                nonlocal cover_remaining
                for ch in it:
                    if not ch:
                        continue
                    if cover_remaining:
                        try:
                            keys = _cover_keys(ch)
                        except Exception:
                            yield ch
                            continue
                        m = np.isin(keys, cover_target)
                        if (cover_z_min is not None) or (cover_z_max is not None):
                            z = np.asarray(ch["Z"], dtype=np.float64)
                            z_ok = np.isfinite(z)
                            if cover_z_min is not None:
                                z_ok = z_ok & (z >= float(cover_z_min))
                            if cover_z_max is not None:
                                z_ok = z_ok & (z < float(cover_z_max))
                            m = m & z_ok
                        if np.any(m):
                            idxs = np.nonzero(m)[0]
                            for ii in idxs:
                                s = int(keys[ii])
                                if s not in cover_remaining:
                                    continue
                                cover_rows[s] = {c: float(np.asarray(ch[c], dtype=np.float64)[ii]) for c in want_cols}
                                cover_remaining.remove(s)
                                if not cover_remaining:
                                    break
                    yield ch

            chunks = _cover_wrap(chunks)
        cols, scanned = _reservoir_sample_from_chunks(
            chunks,
            want_cols=want_cols,
            sample_rows=int(sample_rows),
            seed=int(seed),
            total_rows=int(scan_max_rows) if scan_max_rows is not None else int(layout.n_rows),
        )
        if cover_target is not None:
            sect_out = _cover_keys(cols)
            present = set(int(s) for s in np.unique(sect_out) if int(s) >= 0)
            cover_set = set(int(s) for s in cover_target.tolist())
            missing_before = sorted(cover_set - present)

            replaced = 0
            missing_not_found: list[int] = []
            if missing_before:
                # Replace only rows from sectors with duplicates to preserve coverage.
                uniq, cnt = np.unique(sect_out, return_counts=True)
                order = np.argsort(-cnt, kind="mergesort")
                replace_idx: list[int] = []
                need = len(missing_before)
                for s, c in zip(uniq[order], cnt[order]):
                    if int(c) <= 1 or len(replace_idx) >= need:
                        break
                    idxs = np.nonzero(sect_out == int(s))[0]
                    take = min(int(c) - 1, need - len(replace_idx))
                    if take > 0:
                        replace_idx.extend([int(x) for x in idxs[-take:]])
                if len(replace_idx) < need:
                    # Fallback (should not happen for multi-million row randoms).
                    extra = [int(i) for i in range(int(sect_out.size)) if int(i) not in set(replace_idx)]
                    replace_idx.extend(extra[: need - len(replace_idx)])

                for idx, sec in zip(replace_idx[:need], missing_before):
                    row = cover_rows.get(int(sec))
                    if not row:
                        missing_not_found.append(int(sec))
                        continue
                    for c in want_cols:
                        cols[c][int(idx)] = float(row[c])
                    replaced += 1

            sect_after = _cover_keys(cols)
            present_after = set(int(s) for s in np.unique(sect_after) if int(s) >= 0)
            missing_after = sorted(cover_set - present_after)
            cover_meta = {
                "enabled": True,
                "cover_on_col": cover_on_col,
                "cover_z_min": cover_z_min,
                "cover_z_max": cover_z_max,
                "target_sectors": int(len(cover_set)),
                "cover_rows_collected": int(len(cover_rows)),
                "missing_before": int(len(missing_before)),
                "missing_after": int(len(missing_after)),
                "replaced_rows": int(replaced),
                "missing_not_found": int(len(missing_not_found)),
                "missing_after_sample": missing_after[:20],
            }
    np.savez_compressed(out_npz, **cols)
    meta = {
        "rows_total": int(layout.n_rows),
        "rows_scanned": int(scanned),
        "rows_saved": int(next(iter(cols.values())).size) if cols else 0,
        "row_bytes": int(layout.row_bytes),
        "columns": want_cols,
        "sampling": {
            "method": "reservoir",
            "sample_rows": int(sample_rows),
            "seed": int(seed),
            "scan_max_rows": int(scan_max_rows) if scan_max_rows is not None else None,
            "chunk_rows": int(chunk_rows),
            "cover": cover_meta,
        },
        "source_url": source_url,
        "source_path": _relpath(gz_path),
    }
    return meta


def _extract_remote_gz_to_npz_prefix_rows(
    url: str,
    *,
    out_npz: Path,
    want_cols: list[str],
    max_rows: int,
) -> Dict[str, Any]:
    """
    Stream a remote .fits.gz, read only the first `max_rows` rows from the first BINTABLE,
    and store selected columns into a compressed npz.
    """
    if requests is None:
        raise RuntimeError("requests is required to stream remote files. Install it or download files to data/.../raw/")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with gzip.GzipFile(fileobj=r.raw, mode="rb") as f:
            layout = read_first_bintable_layout(f)
            cols = read_bintable_columns(f, layout=layout, columns=want_cols, max_rows=int(max_rows))
    np.savez_compressed(out_npz, **cols)
    meta = {
        "rows_total": int(layout.n_rows),
        "rows_saved": int(next(iter(cols.values())).size) if cols else 0,
        "row_bytes": int(layout.row_bytes),
        "columns": want_cols,
        "sampling": {"method": "prefix_rows", "max_rows": int(max_rows)},
        "source_url": url,
    }
    return meta


def _try_build_prefix_from_existing_npz(
    *,
    ext_dir: Path,
    base_name: str,
    out_npz: Path,
    want_cols: list[str],
    max_rows: int,
) -> Optional[Dict[str, Any]]:
    """
    If a larger prefix NPZ already exists locally (e.g. prefix_500000),
    derive a smaller prefix NPZ (e.g. prefix_200000) by slicing it.
    This avoids re-downloading the remote random catalog for convergence runs.
    """
    if int(max_rows) <= 0:
        return None
    # Find the smallest existing prefix >= requested.
    candidates: list[tuple[int, Path]] = []
    for p in ext_dir.glob(f"{base_name}.prefix_*.npz"):
        try:
            n = int(p.name.rsplit(".prefix_", 1)[1].split(".npz", 1)[0])
        except Exception:
            continue
        if n >= int(max_rows):
            candidates.append((n, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    src_n, src_npz = candidates[0]

    with np.load(src_npz) as z:
        cols: Dict[str, np.ndarray] = {}
        for c in want_cols:
            if c not in z.files:
                return None
            arr = np.asarray(z[c], dtype=np.float64)
            cols[c] = arr[: int(max_rows)]
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **cols)

    meta = {
        "rows_total": None,
        "rows_saved": int(next(iter(cols.values())).size) if cols else 0,
        "row_bytes": None,
        "columns": want_cols,
        "sampling": {"method": "prefix_rows", "max_rows": int(max_rows), "derived_from": _relpath(src_npz), "source_rows": int(src_n)},
        "source_url": None,
    }
    return meta


def _reservoir_sample_from_chunks(
    chunks: Any,
    *,
    want_cols: list[str],
    sample_rows: int,
    seed: int,
    total_rows: Optional[int] = None,
) -> tuple[Dict[str, np.ndarray], int]:
    """
    Reservoir sample (uniform without replacement) using "smallest random keys" trick:
      - Assign each row an IID key u ~ U(0,1).
      - Keep the `sample_rows` smallest keys.
    This can be done in streaming fashion, and is robust against row ordering bias.
    """
    if int(sample_rows) <= 0:
        raise ValueError("sample_rows must be > 0")
    rng = np.random.default_rng(int(seed))

    # Fast path for large-N catalogs:
    # - If total_rows is known, pick a slightly-too-large Bernoulli sample (threshold p),
    #   then downselect to exactly `sample_rows` using argpartition once.
    # This avoids repeated argpartition on arrays of size k (~millions) per chunk.
    if total_rows is not None and int(total_rows) > 0:
        k = int(sample_rows)
        # Bias the threshold upward by ~5σ to avoid undersampling with overwhelming probability.
        # For N~O(1e7..1e8), σ~O(1e3) at k~2e6, so this adds only ~0.3% overhead.
        margin = int(max(0.0, 5.0 * float(np.sqrt(float(k)))))
        p = float((k + margin) / float(int(total_rows)))
        p = min(max(p, 0.0), 1.0)

        buf_keys: list[np.ndarray] = []
        buf_cols: Dict[str, list[np.ndarray]] = {c: [] for c in want_cols}
        scanned = 0

        for chunk in chunks:
            if not chunk:
                continue
            n = int(next(iter(chunk.values())).shape[0])
            if n <= 0:
                continue
            scanned += n
            keys = rng.random(n, dtype=np.float64)
            m = keys < p
            if not np.any(m):
                continue
            buf_keys.append(keys[m])
            for c in want_cols:
                buf_cols[c].append(np.asarray(chunk[c], dtype=np.float64)[m])

        if not buf_keys:
            return ({c: np.zeros(0, dtype=np.float64) for c in want_cols}, scanned)

        keys_all = np.concatenate(buf_keys)
        if int(keys_all.size) < k:
            # Extremely unlikely when p is biased upward, but keep best-effort.
            cols_out = {c: np.concatenate(buf_cols[c]) if buf_cols[c] else np.zeros(0, dtype=np.float64) for c in want_cols}
            return (cols_out, scanned)

        idx = np.argpartition(keys_all, k - 1)[:k]
        cols_out = {}
        for c in want_cols:
            vals = np.concatenate(buf_cols[c])
            cols_out[c] = np.asarray(vals[idx], dtype=np.float64)
        return (cols_out, scanned)

    res_keys: np.ndarray | None = None
    res_cols: Dict[str, np.ndarray] = {}
    scanned = 0

    for chunk in chunks:
        if not chunk:
            continue
        n = int(next(iter(chunk.values())).shape[0])
        if n <= 0:
            continue
        scanned += n
        keys = rng.random(n, dtype=np.float64)

        if res_keys is None:
            if n <= int(sample_rows):
                res_keys = keys
                res_cols = {c: np.asarray(chunk[c], dtype=np.float64) for c in want_cols}
                continue

            k = int(sample_rows)
            idx = np.argpartition(keys, k - 1)[:k]
            res_keys = keys[idx]
            res_cols = {c: np.asarray(chunk[c][idx], dtype=np.float64) for c in want_cols}
            continue

        k = int(sample_rows)
        n_res = int(res_keys.shape[0])
        if n_res < k:
            # Still filling: merge then keep min(k, total).
            keys_comb = np.concatenate([res_keys, keys])
            keep_n = min(k, int(keys_comb.size))
            idx_keep = np.argpartition(keys_comb, keep_n - 1)[:keep_n]
            keep_old = idx_keep < n_res
            old_idx = idx_keep[keep_old]
            new_idx = idx_keep[~keep_old] - n_res

            res_keys_new = np.empty(keep_n, dtype=np.float64)
            res_keys_new[: old_idx.size] = res_keys[old_idx]
            res_keys_new[old_idx.size :] = keys[new_idx]
            res_cols_new: Dict[str, np.ndarray] = {}
            for c in want_cols:
                out = np.empty(keep_n, dtype=np.float64)
                out[: old_idx.size] = res_cols[c][old_idx]
                out[old_idx.size :] = chunk[c][new_idx]
                res_cols_new[c] = out

            res_keys = res_keys_new
            res_cols = res_cols_new
            continue

        # Full reservoir: only consider candidates that can enter (key < current max).
        thresh = float(np.max(res_keys))
        cand = keys < thresh
        if not np.any(cand):
            continue

        cand_keys = keys[cand]
        keys_comb = np.concatenate([res_keys, cand_keys])
        idx_keep = np.argpartition(keys_comb, k - 1)[:k]
        keep_old = idx_keep < k
        old_idx = idx_keep[keep_old]
        new_idx = idx_keep[~keep_old] - k

        res_keys_new = np.empty(k, dtype=np.float64)
        res_keys_new[: old_idx.size] = res_keys[old_idx]
        res_keys_new[old_idx.size :] = cand_keys[new_idx]
        res_cols_new = {}
        for c in want_cols:
            cand_vals = chunk[c][cand]
            out = np.empty(k, dtype=np.float64)
            out[: old_idx.size] = res_cols[c][old_idx]
            out[old_idx.size :] = cand_vals[new_idx]
            res_cols_new[c] = out

        res_keys = res_keys_new
        res_cols = res_cols_new

    if res_keys is None:
        return ({c: np.zeros(0, dtype=np.float64) for c in want_cols}, scanned)
    return (res_cols, scanned)


def _extract_remote_gz_to_npz_reservoir(
    url: str,
    *,
    out_npz: Path,
    want_cols: list[str],
    sample_rows: int,
    seed: int,
    scan_max_rows: Optional[int],
    chunk_rows: int,
    cover_sectors: Optional[set[int]] = None,
    cover_on_col: str = "ISECT",
    cover_z_min: Optional[float] = None,
    cover_z_max: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Stream a remote .fits.gz, scan the first BINTABLE, and reservoir-sample `sample_rows` rows
    (uniform without replacement) into a compressed NPZ.
    """
    if requests is None:
        raise RuntimeError("requests is required to stream remote files. Install it or download files to data/.../raw/")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    cover_meta: Optional[Dict[str, Any]] = None

    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with gzip.GzipFile(fileobj=r.raw, mode="rb") as f:
            layout = read_first_bintable_layout(f)
            chunks = iter_bintable_column_chunks(
                f,
                layout=layout,
                columns=want_cols,
                max_rows=scan_max_rows,
                chunk_rows=int(chunk_rows),
            )
            cover_rows: Dict[int, Dict[str, float]] = {}
            cover_target: Optional[np.ndarray] = None
            cover_remaining: Optional[set[int]] = None
            if cover_sectors:
                cover_sectors_int = {int(s) for s in cover_sectors}
                if len(cover_sectors_int) > int(sample_rows):
                    raise ValueError("cover_sectors must be <= sample_rows")
                if cover_on_col == "SECTOR_KEY":
                    if "IPOLY" not in want_cols or "ISECT" not in want_cols:
                        raise ValueError("cover_on_col=SECTOR_KEY requires IPOLY and ISECT in want_cols")
                else:
                    if cover_on_col not in want_cols:
                        raise ValueError(f"cover_on_col must be included in want_cols: {cover_on_col}")
                if (cover_z_min is not None) or (cover_z_max is not None):
                    if "Z" not in want_cols:
                        raise ValueError("cover_z_min/max requires Z in want_cols")
                    if (cover_z_min is not None) and (cover_z_max is not None) and not (float(cover_z_min) < float(cover_z_max)):
                        raise ValueError("cover_z_min must be < cover_z_max when both are provided")
                cover_target = np.fromiter(sorted(cover_sectors_int), dtype=np.int64)
                cover_remaining = set(cover_sectors_int)

                def _cover_keys(ch: Dict[str, Any]) -> np.ndarray:
                    if cover_on_col == "SECTOR_KEY":
                        ip = np.asarray(ch["IPOLY"], dtype=np.float64)
                        isect = np.asarray(ch["ISECT"], dtype=np.float64)
                        valid = np.isfinite(ip) & np.isfinite(isect)
                        ip_i = ip.astype(np.int64, copy=False)
                        is_i = isect.astype(np.int64, copy=False)
                        out = np.full(int(ip.shape[0]), -1, dtype=np.int64)
                        out[valid] = (ip_i[valid] << 32) + (is_i[valid] & np.int64(0xFFFFFFFF))
                        return out
                    v = np.asarray(ch[cover_on_col], dtype=np.float64)
                    valid = np.isfinite(v)
                    out = np.full(int(v.shape[0]), -1, dtype=np.int64)
                    out[valid] = v[valid].astype(np.int64)
                    return out

                def _cover_wrap(it: Any) -> Any:
                    nonlocal cover_remaining
                    for ch in it:
                        if not ch:
                            continue
                        if cover_remaining:
                            try:
                                keys = _cover_keys(ch)
                            except Exception:
                                yield ch
                                continue
                            m = np.isin(keys, cover_target)
                            if (cover_z_min is not None) or (cover_z_max is not None):
                                z = np.asarray(ch["Z"], dtype=np.float64)
                                z_ok = np.isfinite(z)
                                if cover_z_min is not None:
                                    z_ok = z_ok & (z >= float(cover_z_min))
                                if cover_z_max is not None:
                                    z_ok = z_ok & (z < float(cover_z_max))
                                m = m & z_ok
                            if np.any(m):
                                idxs = np.nonzero(m)[0]
                                for ii in idxs:
                                    s = int(keys[ii])
                                    if s not in cover_remaining:
                                        continue
                                    cover_rows[s] = {c: float(np.asarray(ch[c], dtype=np.float64)[ii]) for c in want_cols}
                                    cover_remaining.remove(s)
                                    if not cover_remaining:
                                        break
                        yield ch

                chunks = _cover_wrap(chunks)
            cols, scanned = _reservoir_sample_from_chunks(
                chunks,
                want_cols=want_cols,
                sample_rows=int(sample_rows),
                seed=int(seed),
                total_rows=int(scan_max_rows) if scan_max_rows is not None else int(layout.n_rows),
            )
            if cover_target is not None:
                sect_out = _cover_keys(cols)
                present = set(int(s) for s in np.unique(sect_out) if int(s) >= 0)
                cover_set = set(int(s) for s in cover_target.tolist())
                missing_before = sorted(cover_set - present)

                replaced = 0
                missing_not_found: list[int] = []
                if missing_before:
                    uniq, cnt = np.unique(sect_out, return_counts=True)
                    order = np.argsort(-cnt, kind="mergesort")
                    replace_idx: list[int] = []
                    need = len(missing_before)
                    for s, c in zip(uniq[order], cnt[order]):
                        if int(c) <= 1 or len(replace_idx) >= need:
                            break
                        idxs = np.nonzero(sect_out == int(s))[0]
                        take = min(int(c) - 1, need - len(replace_idx))
                        if take > 0:
                            replace_idx.extend([int(x) for x in idxs[-take:]])
                    if len(replace_idx) < need:
                        extra = [int(i) for i in range(int(sect_out.size)) if int(i) not in set(replace_idx)]
                        replace_idx.extend(extra[: need - len(replace_idx)])

                    for idx, sec in zip(replace_idx[:need], missing_before):
                        row = cover_rows.get(int(sec))
                        if not row:
                            missing_not_found.append(int(sec))
                            continue
                        for c in want_cols:
                            cols[c][int(idx)] = float(row[c])
                        replaced += 1

                sect_after = _cover_keys(cols)
                present_after = set(int(s) for s in np.unique(sect_after) if int(s) >= 0)
                missing_after = sorted(cover_set - present_after)
                cover_meta = {
                    "enabled": True,
                    "cover_on_col": cover_on_col,
                    "cover_z_min": cover_z_min,
                    "cover_z_max": cover_z_max,
                    "target_sectors": int(len(cover_set)),
                    "cover_rows_collected": int(len(cover_rows)),
                    "missing_before": int(len(missing_before)),
                    "missing_after": int(len(missing_after)),
                    "replaced_rows": int(replaced),
                    "missing_not_found": int(len(missing_not_found)),
                    "missing_after_sample": missing_after[:20],
                }

    np.savez_compressed(out_npz, **cols)
    meta = {
        "rows_total": int(layout.n_rows),
        "rows_scanned": int(scanned),
        "rows_saved": int(next(iter(cols.values())).size) if cols else 0,
        "row_bytes": int(layout.row_bytes),
        "columns": want_cols,
        "sampling": {
            "method": "reservoir",
            "sample_rows": int(sample_rows),
            "seed": int(seed),
            "scan_max_rows": int(scan_max_rows) if scan_max_rows is not None else None,
            "chunk_rows": int(chunk_rows),
            "cover": cover_meta,
        },
        "source_url": url,
    }
    return meta


def _existing_npz_meta(path: Path) -> Dict[str, Any]:
    name = path.name
    sampling: Dict[str, Any] = {"method": "cached"}
    try:
        if ".prefix_" in name:
            n = int(name.rsplit(".prefix_", 1)[1].split(".npz", 1)[0])
            sampling = {"method": "prefix_rows", "max_rows": n}
        elif ".reservoir_" in name:
            tail = name.rsplit(".reservoir_", 1)[1].split(".npz", 1)[0]
            # reservoir_<N>_seed<S>[_scan<M>]
            parts = tail.split("_")
            if len(parts) >= 2 and parts[1].startswith("seed"):
                sample_rows = int(parts[0])
                seed = int(parts[1].removeprefix("seed"))
                scan_max_rows = None
                for p in parts[2:]:
                    if p.startswith("scan"):
                        scan_max_rows = int(p.removeprefix("scan"))
                sampling = {"method": "reservoir", "sample_rows": sample_rows, "seed": seed, "scan_max_rows": scan_max_rows}
    except Exception:
        sampling = {"method": "cached"}
    try:
        with np.load(path) as z:
            files = list(z.files)
            n = int(z[files[0]].shape[0]) if files else 0
    except Exception:
        files = []
        n = 0
    return {"rows_total": None, "rows_saved": n, "row_bytes": None, "columns": files, "sampling": sampling}


def _normalize_cached_extract_meta(path: Path, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ensure extract meta has a concrete sampling description.

    Older manifests may have "sampling.method=cached" for prefix/reservoir NPZs, which makes later
    analysis ambiguous. When this happens, infer sampling from the filename and merge in any
    high-fidelity fields (rows_total/row_bytes) from the existing meta.
    """
    meta_in: Dict[str, Any] = dict(meta) if isinstance(meta, dict) else {}
    sampling_method = str(meta_in.get("sampling", {}).get("method", "")).strip()
    if (not sampling_method) or sampling_method == "cached":
        inferred = _existing_npz_meta(path)
        # Preserve high-fidelity fields if present in the existing meta.
        for k in ("rows_total", "row_bytes", "source_url"):
            if k in meta_in and meta_in.get(k) is not None:
                inferred[k] = meta_in.get(k)
        return inferred
    return meta_in


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch BOSS DR12v5 LSS catalogs (galaxy+random) and extract columns to NPZ.")
    ap.add_argument(
        "--base-url",
        default=_BASE_URL,
        help="SAS base URL (default: dr12.sdss3.org mirror). Example: https://dr12.sdss3.org/sas/dr12/boss/lss/",
    )
    ap.add_argument(
        "--samples",
        default="cmass,lowz",
        help="comma-separated: cmass,lowz,cmasslowz,cmasslowztot (default: cmass,lowz)",
    )
    ap.add_argument("--caps", default="north,south", help="comma-separated: north,south (default: north,south)")
    ap.add_argument("--random-kind", default="random0", help="random file key: random0 or random1 (default: random0)")
    ap.add_argument(
        "--random-sampling",
        default="prefix_rows",
        choices=["prefix_rows", "reservoir"],
        help="random sampling method: prefix_rows or reservoir (default: prefix_rows)",
    )
    ap.add_argument("--random-max-rows", type=int, default=2_000_000, help="rows to extract from random (default: 2,000,000)")
    ap.add_argument(
        "--random-scan-max-rows",
        type=int,
        default=0,
        help="if >0, limit scanned rows for random reservoir sampling (default: 0 => all rows)",
    )
    ap.add_argument(
        "--galaxy-max-rows",
        type=int,
        default=0,
        help="if >0, stream and extract only prefix rows from galaxy (faster; default: 0 => download full galaxy file)",
    )
    ap.add_argument(
        "--galaxy-sampling",
        default="auto",
        choices=["auto", "download_full", "prefix_rows", "reservoir"],
        help="galaxy sampling method (default: auto => prefix_rows if --galaxy-max-rows>0 else download_full)",
    )
    ap.add_argument(
        "--galaxy-scan-max-rows",
        type=int,
        default=0,
        help="if >0, limit scanned rows for galaxy reservoir sampling (default: 0 => all rows)",
    )
    ap.add_argument("--sampling-seed", type=int, default=0, help="seed for reservoir sampling (default: 0)")
    ap.add_argument("--chunk-rows", type=int, default=200_000, help="chunk rows for reservoir sampling (default: 200,000)")
    ap.add_argument(
        "--random-cover-galaxy-sectors",
        action="store_true",
        help="when using --random-sampling reservoir, ensure the sampled randoms cover all galaxy ISECT sectors "
        "by swapping a small number of rows (deterministic; avoids LS breakage due to rare missing sectors)",
    )
    ap.add_argument(
        "--random-cover-z-min",
        type=float,
        default=None,
        help="optional z min when selecting cover rows/sectors (applies only with --random-cover-galaxy-sectors)",
    )
    ap.add_argument(
        "--random-cover-z-max",
        type=float,
        default=None,
        help="optional z max when selecting cover rows/sectors (applies only with --random-cover-galaxy-sectors)",
    )
    ap.add_argument("--skip-random", action="store_true", help="skip random extraction")
    ap.add_argument("--skip-galaxy", action="store_true", help="skip galaxy download/extraction")
    ap.add_argument("--force", action="store_true", help="overwrite existing extracted npz (re-extract)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    base_url = str(args.base_url).strip()
    if not base_url:
        raise SystemExit("--base-url must be non-empty")
    if not (base_url.startswith("http://") or base_url.startswith("https://")):
        raise SystemExit("--base-url must start with http:// or https://")
    if not base_url.endswith("/"):
        base_url += "/"

    samples = [s.strip().lower() for s in str(args.samples).split(",") if s.strip()]
    caps = [c.strip().lower() for c in str(args.caps).split(",") if c.strip()]
    random_kind = str(args.random_kind).strip()
    if random_kind not in ("random0", "random1"):
        raise SystemExit("--random-kind must be random0 or random1")
    random_sampling = str(args.random_sampling).strip().lower()
    if random_sampling not in ("prefix_rows", "reservoir"):
        raise SystemExit("--random-sampling must be prefix_rows or reservoir")
    random_cover_galaxy_sectors = bool(args.random_cover_galaxy_sectors)
    if random_cover_galaxy_sectors and random_sampling != "reservoir":
        raise SystemExit("--random-cover-galaxy-sectors requires --random-sampling reservoir")
    random_cover_z_min = args.random_cover_z_min
    random_cover_z_max = args.random_cover_z_max
    if (random_cover_z_min is not None or random_cover_z_max is not None) and not random_cover_galaxy_sectors:
        raise SystemExit("--random-cover-z-min/max requires --random-cover-galaxy-sectors")
    if (random_cover_z_min is not None) and (random_cover_z_max is not None) and not (float(random_cover_z_min) < float(random_cover_z_max)):
        raise SystemExit("--random-cover-z-min must be < --random-cover-z-max when both are provided")
    if int(args.random_max_rows) <= 0:
        raise SystemExit("--random-max-rows must be > 0")
    if int(args.galaxy_max_rows) < 0:
        raise SystemExit("--galaxy-max-rows must be >= 0")
    if int(args.random_scan_max_rows) < 0:
        raise SystemExit("--random-scan-max-rows must be >= 0")
    if int(args.galaxy_scan_max_rows) < 0:
        raise SystemExit("--galaxy-scan-max-rows must be >= 0")
    if int(args.chunk_rows) <= 0:
        raise SystemExit("--chunk-rows must be > 0")

    data_dir = _ROOT / "data" / "cosmology" / "boss_dr12v5_lss"
    raw_dir = data_dir / "raw"
    ext_dir = data_dir / "extracted"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    ext_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = data_dir / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    now = datetime.now(timezone.utc).isoformat()
    manifest.setdefault("generated_utc", now)
    manifest["last_run_utc"] = now
    manifest["source_base_url"] = base_url
    manifest.setdefault("items", {})

    galaxy_sampling_arg = str(args.galaxy_sampling).strip().lower()
    if galaxy_sampling_arg not in ("auto", "download_full", "prefix_rows", "reservoir"):
        raise SystemExit("--galaxy-sampling must be auto, download_full, prefix_rows, or reservoir")
    if galaxy_sampling_arg == "auto":
        galaxy_sampling = "prefix_rows" if int(args.galaxy_max_rows) > 0 else "download_full"
    else:
        galaxy_sampling = galaxy_sampling_arg
    if galaxy_sampling in ("prefix_rows", "reservoir") and int(args.galaxy_max_rows) <= 0:
        raise SystemExit("--galaxy-max-rows must be > 0 when --galaxy-sampling is prefix_rows/reservoir")

    for sample in samples:
        if sample not in _SAMPLE_FILES:
            raise SystemExit(f"unknown sample: {sample} (supported: {sorted(_SAMPLE_FILES)})")
        for cap in caps:
            if cap not in _SAMPLE_FILES[sample]:
                raise SystemExit(f"unknown cap: {cap} (supported: north,south)")

            files = _SAMPLE_FILES[sample][cap]
            item_key = f"{sample}:{cap}"
            manifest["items"].setdefault(item_key, {})
            prev_item = dict(manifest["items"].get(item_key, {}))

            if not args.skip_galaxy:
                gal_name = files["galaxy"]
                gal_url = base_url + gal_name
                # Extract columns
                gal_cols = [
                    "RA",
                    "DEC",
                    "Z",
                    "WEIGHT_FKP",
                    "WEIGHT_CP",
                    "WEIGHT_NOZ",
                    "WEIGHT_SYSTOT",
                    "IPOLY",
                    "ISECT",
                ]
                if galaxy_sampling == "prefix_rows":
                    gal_npz = ext_dir / f"{gal_name}.prefix_{int(args.galaxy_max_rows)}.npz"
                    gal_dst = raw_dir / gal_name
                    raw_path = gal_dst if gal_dst.exists() else None
                    raw_bytes = int(gal_dst.stat().st_size) if gal_dst.exists() else None
                    prev = prev_item.get("galaxy", {})
                    prev_npz = str(prev.get("npz_path", "")).strip()
                    if gal_npz.exists() and not args.force:
                        print(f"[skip] {gal_npz.name} (cached)")
                        emeta_prev = prev.get("extract") if prev_npz == _relpath(gal_npz) else None
                        emeta = _normalize_cached_extract_meta(gal_npz, emeta_prev)
                    else:
                        if gal_dst.exists():
                            print(f"[extract] {gal_name} (prefix {int(args.galaxy_max_rows):,} rows; local) -> {gal_npz.name}")
                            emeta = _extract_local_gz_to_npz_prefix_rows(
                                gal_dst,
                                out_npz=gal_npz,
                                want_cols=gal_cols,
                                max_rows=int(args.galaxy_max_rows),
                                source_url=gal_url,
                            )
                        else:
                            print(f"[extract] {gal_name} (prefix {int(args.galaxy_max_rows):,} rows; remote) -> {gal_npz.name}")
                            emeta = _extract_remote_gz_to_npz_prefix_rows(
                                gal_url,
                                out_npz=gal_npz,
                                want_cols=gal_cols,
                                max_rows=int(args.galaxy_max_rows),
                            )
                    manifest["items"][item_key]["galaxy"] = {
                        "url": gal_url,
                        "raw_path": _relpath(raw_path),
                        "raw_sha256": None,
                        "raw_bytes": raw_bytes,
                        "npz_path": _relpath(gal_npz),
                        "extract": emeta,
                    }
                elif galaxy_sampling == "reservoir":
                    tag = f"reservoir_{int(args.galaxy_max_rows)}_seed{int(args.sampling_seed)}"
                    if int(args.galaxy_scan_max_rows) > 0:
                        tag += f"_scan{int(args.galaxy_scan_max_rows)}"
                    gal_npz = ext_dir / f"{gal_name}.{tag}.npz"
                    gal_dst = raw_dir / gal_name
                    raw_path = gal_dst if gal_dst.exists() else None
                    raw_bytes = int(gal_dst.stat().st_size) if gal_dst.exists() else None
                    prev = prev_item.get("galaxy", {})
                    prev_npz = str(prev.get("npz_path", "")).strip()
                    if gal_npz.exists() and not args.force:
                        print(f"[skip] {gal_npz.name} (cached)")
                        emeta = prev.get("extract") if prev_npz == _relpath(gal_npz) else None
                        if not isinstance(emeta, dict):
                            emeta = _existing_npz_meta(gal_npz)
                    else:
                        scan_max = int(args.galaxy_scan_max_rows) if int(args.galaxy_scan_max_rows) > 0 else None
                        if gal_dst.exists():
                            print(
                                f"[extract] {gal_name} (reservoir {int(args.galaxy_max_rows):,} rows, seed={int(args.sampling_seed)}; local) -> {gal_npz.name}"
                            )
                            emeta = _extract_local_gz_to_npz_reservoir(
                                gal_dst,
                                out_npz=gal_npz,
                                want_cols=gal_cols,
                                sample_rows=int(args.galaxy_max_rows),
                                seed=int(args.sampling_seed),
                                scan_max_rows=scan_max,
                                chunk_rows=int(args.chunk_rows),
                                source_url=gal_url,
                            )
                        else:
                            print(
                                f"[extract] {gal_name} (reservoir {int(args.galaxy_max_rows):,} rows, seed={int(args.sampling_seed)}; remote) -> {gal_npz.name}"
                            )
                            emeta = _extract_remote_gz_to_npz_reservoir(
                                gal_url,
                                out_npz=gal_npz,
                                want_cols=gal_cols,
                                sample_rows=int(args.galaxy_max_rows),
                                seed=int(args.sampling_seed),
                                scan_max_rows=scan_max,
                                chunk_rows=int(args.chunk_rows),
                            )
                    manifest["items"][item_key]["galaxy"] = {
                        "url": gal_url,
                        "raw_path": _relpath(raw_path),
                        "raw_sha256": None,
                        "raw_bytes": raw_bytes,
                        "npz_path": _relpath(gal_npz),
                        "extract": emeta,
                    }
                else:
                    gal_dst = raw_dir / gal_name
                    if not gal_dst.exists():
                        print(f"[download] {gal_name}")
                        dmeta = _download_file(gal_url, gal_dst)
                    else:
                        dmeta = {"bytes": gal_dst.stat().st_size, "sha256": _sha256(gal_dst)}

                    gal_npz = ext_dir / f"{gal_name}.npz"
                    if gal_npz.exists() and not args.force:
                        print(f"[skip] {gal_npz.name} (cached)")
                        emeta = _normalize_cached_extract_meta(gal_npz, _existing_npz_meta(gal_npz))
                    else:
                        print(f"[extract] {gal_name} -> {gal_npz.name}")
                        emeta = _extract_local_gz_to_npz(gal_dst, out_npz=gal_npz, want_cols=gal_cols, max_rows=None)
                    manifest["items"][item_key]["galaxy"] = {
                        "url": gal_url,
                        "raw_path": _relpath(gal_dst),
                        "raw_sha256": dmeta["sha256"],
                        "raw_bytes": int(dmeta["bytes"]),
                        "npz_path": _relpath(gal_npz),
                        "extract": emeta,
                    }

            if not args.skip_random:
                rnd_name = files[random_kind]
                rnd_url = base_url + rnd_name
                rnd_cols = ["RA", "DEC", "Z", "WEIGHT_FKP", "IPOLY", "ISECT"]
                if random_sampling == "prefix_rows":
                    rnd_npz = ext_dir / f"{rnd_name}.prefix_{int(args.random_max_rows)}.npz"
                    prev = prev_item.get("random", {})
                    prev_npz = str(prev.get("npz_path", "")).strip()
                    if rnd_npz.exists() and not args.force:
                        print(f"[skip] {rnd_npz.name} (cached)")
                        rmeta_prev = prev.get("extract") if prev_npz == _relpath(rnd_npz) else None
                        rmeta = _normalize_cached_extract_meta(rnd_npz, rmeta_prev)
                    else:
                        # Try to derive from an existing larger prefix to avoid re-downloading huge random catalogs.
                        rmeta = _try_build_prefix_from_existing_npz(
                            ext_dir=ext_dir,
                            base_name=rnd_name,
                            out_npz=rnd_npz,
                            want_cols=rnd_cols,
                            max_rows=int(args.random_max_rows),
                        )
                        if isinstance(rmeta, dict):
                            print(
                                f"[derive] {rnd_name} (prefix {int(args.random_max_rows):,} rows) from existing NPZ -> {rnd_npz.name}"
                            )
                        else:
                            rnd_dst = raw_dir / rnd_name
                            if rnd_dst.exists():
                                print(f"[extract] {rnd_name} (prefix {int(args.random_max_rows):,} rows; local) -> {rnd_npz.name}")
                                rmeta = _extract_local_gz_to_npz_prefix_rows(
                                    rnd_dst,
                                    out_npz=rnd_npz,
                                    want_cols=rnd_cols,
                                    max_rows=int(args.random_max_rows),
                                    source_url=rnd_url,
                                )
                            else:
                                print(f"[extract] {rnd_name} (prefix {int(args.random_max_rows):,} rows; remote) -> {rnd_npz.name}")
                                rmeta = _extract_remote_gz_to_npz_prefix_rows(
                                    rnd_url,
                                    out_npz=rnd_npz,
                                    want_cols=rnd_cols,
                                    max_rows=int(args.random_max_rows),
                                )
                else:
                    tag = f"reservoir_{int(args.random_max_rows)}_seed{int(args.sampling_seed)}"
                    if int(args.random_scan_max_rows) > 0:
                        tag += f"_scan{int(args.random_scan_max_rows)}"
                    if random_cover_galaxy_sectors:
                        tag += "_covergal"
                    rnd_npz = ext_dir / f"{rnd_name}.{tag}.npz"
                    prev = prev_item.get("random", {})
                    prev_npz = str(prev.get("npz_path", "")).strip()
                    if rnd_npz.exists() and not args.force:
                        print(f"[skip] {rnd_npz.name} (cached)")
                        rmeta_prev = prev.get("extract") if prev_npz == _relpath(rnd_npz) else None
                        rmeta = _normalize_cached_extract_meta(rnd_npz, rmeta_prev)
                    else:
                        scan_max = int(args.random_scan_max_rows) if int(args.random_scan_max_rows) > 0 else None
                        rnd_dst = raw_dir / rnd_name
                        cover_sectors: Optional[set[int]] = None
                        if random_cover_galaxy_sectors:
                            gal_npz_for_cover: Optional[Path] = None
                            try:
                                gal_entry = manifest.get("items", {}).get(item_key, {}).get("galaxy", {})
                                gal_npz_rel = str(gal_entry.get("npz_path", "")).strip()
                                if gal_npz_rel:
                                    gal_npz_for_cover = Path(gal_npz_rel)
                                    if not gal_npz_for_cover.is_absolute():
                                        gal_npz_for_cover = _ROOT / gal_npz_for_cover
                            except Exception:
                                gal_npz_for_cover = None
                            if gal_npz_for_cover is None:
                                cand = ext_dir / f"{files['galaxy']}.npz"
                                if cand.exists():
                                    gal_npz_for_cover = cand
                            if gal_npz_for_cover is None or (not gal_npz_for_cover.exists()):
                                raise SystemExit(
                                    f"--random-cover-galaxy-sectors requires an existing galaxy NPZ for {item_key}. "
                                    "Run this script once without --skip-galaxy to create it."
                                )
                            with np.load(gal_npz_for_cover) as z:
                                if "IPOLY" not in z.files or "ISECT" not in z.files:
                                    raise SystemExit(f"galaxy NPZ missing IPOLY/ISECT: {gal_npz_for_cover}")
                                if "Z" not in z.files:
                                    raise SystemExit(f"galaxy NPZ missing Z: {gal_npz_for_cover}")
                                ip = np.asarray(z["IPOLY"], dtype=np.float64)
                                isect = np.asarray(z["ISECT"], dtype=np.float64)
                                zg = np.asarray(z["Z"], dtype=np.float64)
                                valid = np.isfinite(ip) & np.isfinite(isect)
                                valid = valid & np.isfinite(zg)
                                if random_cover_z_min is not None:
                                    valid = valid & (zg >= float(random_cover_z_min))
                                if random_cover_z_max is not None:
                                    valid = valid & (zg < float(random_cover_z_max))
                                ip_i = ip.astype(np.int64, copy=False)
                                is_i = isect.astype(np.int64, copy=False)
                                keys = (ip_i[valid] << 32) + (is_i[valid] & np.int64(0xFFFFFFFF))
                                cover_sectors = {int(s) for s in np.unique(keys)}
                            ztag = ""
                            if random_cover_z_min is not None or random_cover_z_max is not None:
                                lo = "-inf" if (random_cover_z_min is None) else str(float(random_cover_z_min))
                                hi = "+inf" if (random_cover_z_max is None) else str(float(random_cover_z_max))
                                ztag = f" (Z in [{lo},{hi}))"
                            print(f"[cover] random will cover galaxy sector keys: {len(cover_sectors):,} (IPOLY<<32 + ISECT){ztag}")
                        if rnd_dst.exists():
                            print(
                                f"[extract] {rnd_name} (reservoir {int(args.random_max_rows):,} rows, seed={int(args.sampling_seed)}; local) -> {rnd_npz.name}"
                            )
                            rmeta = _extract_local_gz_to_npz_reservoir(
                                rnd_dst,
                                out_npz=rnd_npz,
                                want_cols=rnd_cols,
                                sample_rows=int(args.random_max_rows),
                                seed=int(args.sampling_seed),
                                scan_max_rows=scan_max,
                                chunk_rows=int(args.chunk_rows),
                                source_url=rnd_url,
                                cover_sectors=cover_sectors,
                                cover_on_col="SECTOR_KEY",
                                cover_z_min=random_cover_z_min,
                                cover_z_max=random_cover_z_max,
                            )
                        else:
                            print(
                                f"[extract] {rnd_name} (reservoir {int(args.random_max_rows):,} rows, seed={int(args.sampling_seed)}; remote) -> {rnd_npz.name}"
                            )
                            rmeta = _extract_remote_gz_to_npz_reservoir(
                                rnd_url,
                                out_npz=rnd_npz,
                                want_cols=rnd_cols,
                                sample_rows=int(args.random_max_rows),
                                seed=int(args.sampling_seed),
                                scan_max_rows=scan_max,
                                chunk_rows=int(args.chunk_rows),
                                cover_sectors=cover_sectors,
                                cover_on_col="SECTOR_KEY",
                                cover_z_min=random_cover_z_min,
                                cover_z_max=random_cover_z_max,
                            )
                rnd_dst = raw_dir / rnd_name
                raw_path = rnd_dst if rnd_dst.exists() else None
                raw_bytes = int(rnd_dst.stat().st_size) if rnd_dst.exists() else None
                manifest["items"][item_key]["random"] = {
                    "kind": random_kind,
                    "url": rnd_url,
                    "raw_path": _relpath(raw_path),
                    "raw_sha256": None,
                    "raw_bytes": raw_bytes,
                    "npz_path": _relpath(rnd_npz),
                    "extract": rmeta,
                }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] manifest: {manifest_path}")

    try:
        worklog.append_event(
            {
                "event_type": "fetch_boss_dr12v5_lss",
                "argv": sys.argv,
                "outputs": {"manifest_json": manifest_path},
                "metrics": {
                    "samples": samples,
                    "caps": caps,
                    "random_kind": random_kind,
                    "random_sampling": random_sampling,
                    "random_cover_galaxy_sectors": bool(random_cover_galaxy_sectors),
                    "random_cover_z_min": random_cover_z_min,
                    "random_cover_z_max": random_cover_z_max,
                    "random_max_rows": int(args.random_max_rows),
                    "random_scan_max_rows": int(args.random_scan_max_rows),
                    "galaxy_max_rows": int(args.galaxy_max_rows),
                    "galaxy_sampling": galaxy_sampling,
                    "galaxy_scan_max_rows": int(args.galaxy_scan_max_rows),
                    "sampling_seed": int(args.sampling_seed),
                    "chunk_rows": int(args.chunk_rows),
                    "force": bool(args.force),
                },
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
