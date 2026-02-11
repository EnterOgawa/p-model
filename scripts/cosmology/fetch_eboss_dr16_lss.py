#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_eboss_dr16_lss.py

Phase 4（宇宙論）/ Step 4.5B.21（eBOSS/DESI拡張）:
eBOSS DR16 LSS（銀河 catalog + ランダム catalog）を一次入力として取得し、
RA/DEC/z/weight を軽量フォーマット（npz）へ抽出してキャッシュする。

背景：
- BAO圧縮出力（D_M/r_d 等）は、観測統計→距離変換→テンプレートfit を含む推定量。
- 本プロジェクトでは「銀河+random の一次統計 ξ(s,μ)→ξℓ」を P-model 側の距離写像で再計算し、
  幾何（ε, AP）を検証する。まず一次入力（銀河+random）を確定する。

データソース（SAS）:
  - 既定: https://data.sdss.org/sas/dr16/eboss/lss/catalogs/DR16/

出力（固定）:
  data/cosmology/eboss_dr16_lss/
    - raw/               (fits は原則ここに配置)
    - extracted/         (RA/DEC/z/weight を npz に保存)
    - manifest.json      (取得メタ・抽出条件)

注意：
- random catalog は GB級。既定では raw/ へ保存しない（事故防止）。
  - galaxy raw だけ保存：`--download-galaxy-raw`
  - random も raw で保存（非推奨・必要時のみ）：`--download-missing`
  - raw を保存せず NPZ 抽出のみ（推奨）：`--stream-missing`
"""

from __future__ import annotations

import argparse
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

# Prefer the dedicated DR16 host (often faster/more stable than data.sdss.org).
_BASE_URL = "https://dr16.sdss.org/sas/dr16/eboss/lss/catalogs/DR16/"
_REQ_TIMEOUT = (30, 600)  # (connect, read) seconds

# Targets for Phase 4.5B.21 (BAO primary-statistics): eBOSS clustering catalogs + randoms.
_SAMPLE_FILES: Dict[str, Dict[str, Dict[str, str]]] = {
    "lrgpcmass_rec": {
        "north": {
            "galaxy": "eBOSS_LRGpCMASS_clustering_data_rec-NGC-vDR16.fits",
            "random": "eBOSS_LRGpCMASS_clustering_random_rec-NGC-vDR16.fits",
        },
        "south": {
            "galaxy": "eBOSS_LRGpCMASS_clustering_data_rec-SGC-vDR16.fits",
            "random": "eBOSS_LRGpCMASS_clustering_random_rec-SGC-vDR16.fits",
        },
    },
    "qso": {
        "north": {
            "galaxy": "eBOSS_QSO_clustering_data-NGC-vDR16.fits",
            "random": "eBOSS_QSO_clustering_random-NGC-vDR16.fits",
        },
        "south": {
            "galaxy": "eBOSS_QSO_clustering_data-SGC-vDR16.fits",
            "random": "eBOSS_QSO_clustering_random-SGC-vDR16.fits",
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
    with requests.get(url, stream=True, timeout=_REQ_TIMEOUT) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
    tmp.replace(dst)
    return {"bytes": dst.stat().st_size, "sha256": _sha256(dst)}

def _extract_remote_fits_to_npz(
    url: str,
    *,
    out_npz: Path,
    want_cols: list[str],
    max_rows: Optional[int],
) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is required to stream remote files. Install it or place files under data/.../raw/")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=_REQ_TIMEOUT) as r:
        r.raise_for_status()
        f = r.raw
        layout = read_first_bintable_layout(f)
        cols = read_bintable_columns(f, layout=layout, columns=want_cols, max_rows=max_rows)
    np.savez_compressed(out_npz, **cols)
    meta = {
        "rows_total": int(layout.n_rows),
        "rows_saved": int(next(iter(cols.values())).size) if cols else 0,
        "row_bytes": int(layout.row_bytes),
        "columns": want_cols,
        "sampling": {"method": "full"} if max_rows is None else {"method": "prefix_rows", "max_rows": int(max_rows)},
        "source_url": url,
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

    (Copied/adapted from `fetch_boss_dr12v5_lss.py` to keep the eBOSS fetch self-contained.)
    """
    if int(sample_rows) <= 0:
        raise ValueError("sample_rows must be > 0")
    rng = np.random.default_rng(int(seed))

    if total_rows is not None and int(total_rows) > 0:
        k = int(sample_rows)
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


def _extract_local_fits_to_npz(
    fits_path: Path,
    *,
    out_npz: Path,
    want_cols: list[str],
    max_rows: Optional[int],
) -> Dict[str, Any]:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    with fits_path.open("rb") as f:
        layout = read_first_bintable_layout(f)
        cols = read_bintable_columns(f, layout=layout, columns=want_cols, max_rows=max_rows)
    np.savez_compressed(out_npz, **cols)
    meta = {
        "rows_total": int(layout.n_rows),
        "rows_saved": int(next(iter(cols.values())).size) if cols else 0,
        "row_bytes": int(layout.row_bytes),
        "columns": want_cols,
        "sampling": {"method": "full"} if max_rows is None else {"method": "prefix_rows", "max_rows": int(max_rows)},
    }
    return meta


def _npz_rows(npz_path: Path) -> int:
    with np.load(npz_path) as z:
        if "RA" in z:
            return int(np.asarray(z["RA"]).size)
        for name in z.files:
            return int(np.asarray(z[name]).size)
    raise ValueError(f"npz has no arrays: {npz_path}")


def _read_layout_local(fits_path: Path) -> Dict[str, int]:
    with fits_path.open("rb") as f:
        layout = read_first_bintable_layout(f)
    return {"rows_total": int(layout.n_rows), "row_bytes": int(layout.row_bytes)}


def _read_layout_remote(url: str) -> Dict[str, int]:
    if requests is None:
        raise RuntimeError("requests is required to read remote FITS layout. Install it or use local raw FITS.")
    with requests.get(url, stream=True, timeout=_REQ_TIMEOUT) as r:
        r.raise_for_status()
        layout = read_first_bintable_layout(r.raw)
    return {"rows_total": int(layout.n_rows), "row_bytes": int(layout.row_bytes)}


def _extract_local_fits_to_npz_reservoir(
    fits_path: Path,
    *,
    out_npz: Path,
    want_cols: list[str],
    sample_rows: int,
    seed: int,
    chunk_rows: int,
    scan_max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    with fits_path.open("rb") as f:
        layout = read_first_bintable_layout(f)
        chunks = iter_bintable_column_chunks(
            f, layout=layout, columns=want_cols, max_rows=scan_max_rows, chunk_rows=int(chunk_rows)
        )
        cols, scanned = _reservoir_sample_from_chunks(
            chunks,
            want_cols=want_cols,
            sample_rows=int(sample_rows),
            seed=int(seed),
            total_rows=int(layout.n_rows) if scan_max_rows is None else int(scan_max_rows),
        )
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
            "scan_max_rows": scan_max_rows,
            "chunk_rows": int(chunk_rows),
        },
    }
    return meta


def _build_manifest_skeleton(*, generated_utc: str, base_url: str) -> Dict[str, Any]:
    return {
        "generated_utc": generated_utc,
        "last_run_utc": generated_utc,
        "source_base_url": base_url,
        "items": {},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch and extract eBOSS DR16 LSS catalogs for BAO primary-statistics re-derivation.")
    ap.add_argument("--sample", choices=sorted(_SAMPLE_FILES), default="lrgpcmass_rec", help="catalog sample (default: lrgpcmass_rec)")
    ap.add_argument("--caps", choices=["combined", "north", "south"], default="combined", help="sky cap selection (default: combined)")
    ap.add_argument("--base-url", default=_BASE_URL, help=f"base URL for SAS (default: {_BASE_URL})")
    ap.add_argument("--download-missing", action="store_true", help="download missing raw fits into data/cosmology/eboss_dr16_lss/raw/")
    ap.add_argument(
        "--download-galaxy-raw",
        action="store_true",
        help="download missing galaxy raw FITS into raw/ (random is unaffected; use --download-missing to download random too)",
    )
    ap.add_argument(
        "--stream-missing",
        action="store_true",
        help="when raw is missing, stream remote FITS and extract NPZ without saving raw (recommended for large randoms)",
    )
    ap.add_argument("--print-needed-files", action="store_true", help="print required filenames/URLs and exit")
    ap.add_argument("--random-sampling", choices=["prefix_rows", "reservoir"], default="prefix_rows", help="random sampling method (default: prefix_rows)")
    ap.add_argument("--random-max-rows", type=int, default=2_000_000, help="random rows for prefix/reservoir (default: 2,000,000)")
    ap.add_argument("--random-scan-max-rows", type=int, default=0, help="if >0, limit scanned rows for reservoir (default: 0 => all rows)")
    ap.add_argument("--sampling-seed", type=int, default=0, help="seed for reservoir sampling (default: 0)")
    ap.add_argument("--chunk-rows", type=int, default=200_000, help="chunk rows for reservoir sampling (default: 200,000)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    sample = str(args.sample)
    base_url = str(args.base_url).rstrip("/") + "/"

    out_dir = _ROOT / "data" / "cosmology" / "eboss_dr16_lss"
    raw_dir = out_dir / "raw"
    ext_dir = out_dir / "extracted"
    manifest_path = out_dir / "manifest.json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ext_dir.mkdir(parents=True, exist_ok=True)

    caps_req = str(args.caps)
    if caps_req == "combined":
        caps = ["north", "south"]
    else:
        caps = [caps_req]

    want_gal_cols = ["RA", "DEC", "Z", "WEIGHT_FKP", "WEIGHT_CP", "WEIGHT_NOZ", "WEIGHT_SYSTOT"]
    want_rnd_cols = ["RA", "DEC", "Z", "WEIGHT_FKP"]

    if bool(args.print_needed_files):
        for cap in caps:
            files = _SAMPLE_FILES[sample][cap]
            for k in ("galaxy", "random"):
                name = files[k]
                print(f"{cap}\t{k}\t{name}\t{base_url}{name}")
        return 0

    now = datetime.now(timezone.utc).isoformat()
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = _build_manifest_skeleton(generated_utc=now, base_url=base_url)
    else:
        manifest = _build_manifest_skeleton(generated_utc=now, base_url=base_url)
    manifest["last_run_utc"] = now
    manifest["source_base_url"] = base_url

    for cap in caps:
        files = _SAMPLE_FILES[sample][cap]
        gal_name = files["galaxy"]
        rnd_name = files["random"]
        gal_url = f"{base_url}{gal_name}"
        rnd_url = f"{base_url}{rnd_name}"

        gal_raw = raw_dir / gal_name
        rnd_raw = raw_dir / rnd_name

        stream_missing = bool(args.stream_missing)
        download_missing = bool(args.download_missing)
        download_galaxy_raw = bool(args.download_galaxy_raw) or download_missing

        # Extract galaxy (full).
        gal_npz = ext_dir / f"{gal_name}.npz"
        gal_raw_meta: Dict[str, Any] = {"raw_path": None, "raw_bytes": None, "raw_sha256": None}
        if gal_raw.exists():
            gal_raw_meta = {"raw_path": _relpath(gal_raw), "raw_bytes": int(gal_raw.stat().st_size), "raw_sha256": _sha256(gal_raw)}
        elif download_galaxy_raw:
            print(f"[download] {gal_url} -> {gal_raw}")
            dmeta = _download_file(gal_url, gal_raw)
            print(f"  done: {dmeta.get('bytes')} bytes")
            gal_raw_meta = {"raw_path": _relpath(gal_raw), "raw_bytes": int(gal_raw.stat().st_size), "raw_sha256": _sha256(gal_raw)}

        if not gal_npz.exists():
            if gal_raw.exists():
                print(f"[extract] {gal_raw.name} -> {gal_npz.name}")
                gal_extract = _extract_local_fits_to_npz(gal_raw, out_npz=gal_npz, want_cols=want_gal_cols, max_rows=None)
            elif stream_missing:
                print(f"[stream] {gal_url} -> {gal_npz.name}")
                gal_extract = _extract_remote_fits_to_npz(gal_url, out_npz=gal_npz, want_cols=want_gal_cols, max_rows=None)
            else:
                raise SystemExit(f"missing galaxy file: {gal_raw} (download it or pass --download-galaxy-raw/--download-missing/--stream-missing)")
        else:
            rows_saved = _npz_rows(gal_npz)
            try:
                layout = _read_layout_local(gal_raw) if gal_raw.exists() else _read_layout_remote(gal_url)
            except Exception:
                layout = {"rows_total": rows_saved, "row_bytes": None}
            gal_extract = {
                "rows_total": int(layout["rows_total"]) if layout.get("rows_total") is not None else None,
                "rows_saved": int(rows_saved),
                "row_bytes": int(layout["row_bytes"]) if layout.get("row_bytes") is not None else None,
                "columns": want_gal_cols,
                "sampling": {"method": "full"},
                "source_url": gal_url,
                "source_path": _relpath(gal_raw) if gal_raw.exists() else None,
            }

        # Extract random (prefix or reservoir).
        rnd_sampling = str(args.random_sampling)
        scan_max_rows = int(args.random_scan_max_rows)
        scan_max_rows_eff: Optional[int] = None if scan_max_rows <= 0 else scan_max_rows
        if rnd_sampling == "prefix_rows":
            rnd_npz = ext_dir / f"{rnd_name}.prefix_{int(args.random_max_rows)}.npz"
        elif rnd_sampling == "reservoir":
            rnd_npz = ext_dir / f"{rnd_name}.reservoir_{int(args.random_max_rows)}_seed{int(args.sampling_seed)}.npz"
        else:
            raise SystemExit(f"invalid --random-sampling: {rnd_sampling}")

        rnd_raw_meta: Dict[str, Any] = {"raw_path": None, "raw_bytes": None, "raw_sha256": None}
        if rnd_raw.exists():
            rnd_raw_meta = {"raw_path": _relpath(rnd_raw), "raw_bytes": int(rnd_raw.stat().st_size), "raw_sha256": _sha256(rnd_raw)}

        if not rnd_npz.exists():
            if rnd_sampling == "prefix_rows":
                if rnd_raw.exists():
                    print(f"[extract] {rnd_raw.name} (prefix {int(args.random_max_rows):,}) -> {rnd_npz.name}")
                    rnd_extract = _extract_local_fits_to_npz(rnd_raw, out_npz=rnd_npz, want_cols=want_rnd_cols, max_rows=int(args.random_max_rows))
                elif download_missing:
                    print(f"[download] {rnd_url} -> {rnd_raw}")
                    dmeta = _download_file(rnd_url, rnd_raw)
                    print(f"  done: {dmeta.get('bytes')} bytes")
                    rnd_raw_meta = {"raw_path": _relpath(rnd_raw), "raw_bytes": int(rnd_raw.stat().st_size), "raw_sha256": _sha256(rnd_raw)}
                    print(f"[extract] {rnd_raw.name} (prefix {int(args.random_max_rows):,}) -> {rnd_npz.name}")
                    rnd_extract = _extract_local_fits_to_npz(rnd_raw, out_npz=rnd_npz, want_cols=want_rnd_cols, max_rows=int(args.random_max_rows))
                elif stream_missing:
                    print(f"[stream] {rnd_url} (prefix {int(args.random_max_rows):,}) -> {rnd_npz.name}")
                    rnd_extract = _extract_remote_fits_to_npz(rnd_url, out_npz=rnd_npz, want_cols=want_rnd_cols, max_rows=int(args.random_max_rows))
                else:
                    raise SystemExit(f"missing random file: {rnd_raw} (download it or pass --download-missing/--stream-missing)")
            else:
                if rnd_raw.exists():
                    msg = f"[extract] {rnd_raw.name} (reservoir {int(args.random_max_rows):,}, seed={int(args.sampling_seed)})"
                    if scan_max_rows_eff is not None:
                        msg += f", scan_max_rows={scan_max_rows_eff:,}"
                    print(f"{msg} -> {rnd_npz.name}")
                    rnd_extract = _extract_local_fits_to_npz_reservoir(
                        rnd_raw,
                        out_npz=rnd_npz,
                        want_cols=want_rnd_cols,
                        sample_rows=int(args.random_max_rows),
                        seed=int(args.sampling_seed),
                        chunk_rows=int(args.chunk_rows),
                        scan_max_rows=scan_max_rows_eff,
                    )
                elif download_missing:
                    print(f"[download] {rnd_url} -> {rnd_raw}")
                    dmeta = _download_file(rnd_url, rnd_raw)
                    print(f"  done: {dmeta.get('bytes')} bytes")
                    rnd_raw_meta = {"raw_path": _relpath(rnd_raw), "raw_bytes": int(rnd_raw.stat().st_size), "raw_sha256": _sha256(rnd_raw)}
                    msg = f"[extract] {rnd_raw.name} (reservoir {int(args.random_max_rows):,}, seed={int(args.sampling_seed)})"
                    if scan_max_rows_eff is not None:
                        msg += f", scan_max_rows={scan_max_rows_eff:,}"
                    print(f"{msg} -> {rnd_npz.name}")
                    rnd_extract = _extract_local_fits_to_npz_reservoir(
                        rnd_raw,
                        out_npz=rnd_npz,
                        want_cols=want_rnd_cols,
                        sample_rows=int(args.random_max_rows),
                        seed=int(args.sampling_seed),
                        chunk_rows=int(args.chunk_rows),
                        scan_max_rows=scan_max_rows_eff,
                    )
                elif stream_missing:
                    print(f"[stream] {rnd_url} (reservoir {int(args.random_max_rows):,}, seed={int(args.sampling_seed)}) -> {rnd_npz.name}")
                    if requests is None:
                        raise RuntimeError("requests is required to stream remote files")
                    with requests.get(rnd_url, stream=True, timeout=_REQ_TIMEOUT) as r:
                        r.raise_for_status()
                        f = r.raw
                        layout = read_first_bintable_layout(f)
                        chunks = iter_bintable_column_chunks(
                            f,
                            layout=layout,
                            columns=want_rnd_cols,
                            max_rows=scan_max_rows_eff,
                            chunk_rows=int(args.chunk_rows),
                        )
                        cols, scanned = _reservoir_sample_from_chunks(
                            chunks,
                            want_cols=want_rnd_cols,
                            sample_rows=int(args.random_max_rows),
                            seed=int(args.sampling_seed),
                            total_rows=int(layout.n_rows) if scan_max_rows_eff is None else int(scan_max_rows_eff),
                        )
                    np.savez_compressed(rnd_npz, **cols)
                    rnd_extract = {
                        "rows_total": int(layout.n_rows),
                        "rows_scanned": int(scanned),
                        "rows_saved": int(next(iter(cols.values())).size) if cols else 0,
                        "row_bytes": int(layout.row_bytes),
                        "columns": want_rnd_cols,
                        "sampling": {
                            "method": "reservoir",
                            "sample_rows": int(args.random_max_rows),
                            "seed": int(args.sampling_seed),
                            "scan_max_rows": scan_max_rows_eff,
                            "chunk_rows": int(args.chunk_rows),
                        },
                        "source_url": rnd_url,
                    }
                else:
                    raise SystemExit(f"missing random file: {rnd_raw} (download it or pass --download-missing/--stream-missing)")
        else:
            rows_saved = _npz_rows(rnd_npz)
            try:
                layout = _read_layout_local(rnd_raw) if rnd_raw.exists() else _read_layout_remote(rnd_url)
            except Exception:
                layout = {"rows_total": None, "row_bytes": None}

            rows_total = layout.get("rows_total")
            row_bytes = layout.get("row_bytes")
            if rnd_sampling == "reservoir":
                if scan_max_rows_eff is None:
                    rows_scanned = int(rows_total) if rows_total is not None else None
                else:
                    rows_scanned = int(min(int(scan_max_rows_eff), int(rows_total))) if rows_total is not None else int(scan_max_rows_eff)
                rnd_extract = {
                    "rows_total": int(rows_total) if rows_total is not None else None,
                    "rows_scanned": rows_scanned,
                    "rows_saved": int(rows_saved),
                    "row_bytes": int(row_bytes) if row_bytes is not None else None,
                    "columns": want_rnd_cols,
                    "sampling": {
                        "method": "reservoir",
                        "sample_rows": int(args.random_max_rows),
                        "seed": int(args.sampling_seed),
                        "scan_max_rows": scan_max_rows_eff,
                        "chunk_rows": int(args.chunk_rows),
                    },
                    "source_url": rnd_url,
                    "source_path": _relpath(rnd_raw) if rnd_raw.exists() else None,
                }
            else:
                rnd_extract = {
                    "rows_total": int(rows_total) if rows_total is not None else None,
                    "rows_saved": int(rows_saved),
                    "row_bytes": int(row_bytes) if row_bytes is not None else None,
                    "columns": want_rnd_cols,
                    "sampling": {"method": "prefix_rows", "max_rows": int(args.random_max_rows)},
                    "source_url": rnd_url,
                    "source_path": _relpath(rnd_raw) if rnd_raw.exists() else None,
                }

        key = f"{sample}:{cap}"
        manifest["items"][key] = {
            "galaxy": {
                "url": gal_url,
                "raw_path": gal_raw_meta["raw_path"],
                "raw_bytes": gal_raw_meta["raw_bytes"],
                "raw_sha256": gal_raw_meta["raw_sha256"],
                "npz_path": _relpath(gal_npz),
                "extract": gal_extract,
            },
            "random": {
                "kind": "random",
                "url": rnd_url,
                "raw_path": rnd_raw_meta["raw_path"],
                "raw_bytes": rnd_raw_meta["raw_bytes"],
                "raw_sha256": rnd_raw_meta["raw_sha256"],
                "npz_path": _relpath(rnd_npz),
                "extract": rnd_extract,
            },
        }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        worklog.append_event(
            {
                "event_type": "fetch_eboss_dr16_lss",
                "argv": sys.argv,
                "params": {
                    "sample": sample,
                    "caps": caps_req,
                    "base_url": base_url,
                    "download_missing": bool(args.download_missing),
                    "download_galaxy_raw": bool(args.download_galaxy_raw),
                    "stream_missing": bool(args.stream_missing),
                    "random_sampling": str(args.random_sampling),
                    "random_max_rows": int(args.random_max_rows),
                    "random_scan_max_rows": int(args.random_scan_max_rows),
                    "sampling_seed": int(args.sampling_seed),
                    "chunk_rows": int(args.chunk_rows),
                },
                "outputs": {"manifest_path": manifest_path},
            }
        )
    except Exception:
        pass
    print(f"[ok] wrote: {manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
