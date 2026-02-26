#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_desi_dr1_lss.py

Phase 4（宇宙論）/ Step 4.5B.21.4.3（DESI拡張）:
DESI DR1 LSS（iron/LSScats/v1.5）の clustering catalogs（galaxy + random）を一次入力として取得し、
RA/DEC/z/weight を軽量フォーマット（npz）へ抽出してキャッシュする。

背景：
- BAO圧縮出力（D_M/r_d 等）は、観測統計→距離変換→テンプレートfit を含む推定量。
- 本プロジェクトでは「銀河+random の一次統計 ξ(s,μ)→ξℓ」を、P-model 側の距離写像で再計算し、
  幾何（ε, AP）の整合性を検証する。
- その入口として、DESI も「銀河+random → ξℓ → peakfit」の一次統計ラインに接続する。

データソース（DESI public DR1）:
  - https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/

出力（固定）:
  data/cosmology/desi_dr1_lss/
    - raw/               (fits は任意。巨大な random は既定で保存しない)
    - extracted/         (RA/DEC/z/weight を npz に保存)
    - manifest.json      (取得メタ・抽出条件)

注意：
- random catalog は非常に大きい。既定では `reservoir` で 2,000,000 行を一様サンプルする。
- 結果の議論がブレないよう、random 抽出法（method/seed/max_rows）も manifest に固定記録する。
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
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cosmology.boss_dr12v5_fits import (  # noqa: E402
    iter_bintable_column_chunks,
    read_bintable_columns,
    read_first_bintable_layout,
)
from scripts.summary import worklog  # noqa: E402


_BASE_URL = "https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/"
_REQ_TIMEOUT = (30, 600)  # (connect, read)

# Phase 4.5B.21.4.3: start with LRG clustering catalogs (NGC/SGC).
_SAMPLE_FILES: Dict[str, Dict[str, Dict[str, str]]] = {
    "lrg": {
        "north": {"galaxy": "LRG_NGC_clustering.dat.fits", "random_tmpl": "LRG_NGC_{idx}_clustering.ran.fits"},
        "south": {"galaxy": "LRG_SGC_clustering.dat.fits", "random_tmpl": "LRG_SGC_{idx}_clustering.ran.fits"},
    },
    # Additional tracers (Phase 4.5B.21.4.4.6):
    "qso": {
        "north": {"galaxy": "QSO_NGC_clustering.dat.fits", "random_tmpl": "QSO_NGC_{idx}_clustering.ran.fits"},
        "south": {"galaxy": "QSO_SGC_clustering.dat.fits", "random_tmpl": "QSO_SGC_{idx}_clustering.ran.fits"},
    },
    "bgs_bright": {
        "north": {"galaxy": "BGS_BRIGHT_NGC_clustering.dat.fits", "random_tmpl": "BGS_BRIGHT_NGC_{idx}_clustering.ran.fits"},
        "south": {"galaxy": "BGS_BRIGHT_SGC_clustering.dat.fits", "random_tmpl": "BGS_BRIGHT_SGC_{idx}_clustering.ran.fits"},
    },
    "elg_lopnotqso": {
        "north": {
            "galaxy": "ELG_LOPnotqso_NGC_clustering.dat.fits",
            "random_tmpl": "ELG_LOPnotqso_NGC_{idx}_clustering.ran.fits",
        },
        "south": {
            "galaxy": "ELG_LOPnotqso_SGC_clustering.dat.fits",
            "random_tmpl": "ELG_LOPnotqso_SGC_{idx}_clustering.ran.fits",
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
    # 条件分岐: `path is None` を満たす経路を評価する。
    if path is None:
        return None

    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _download_file(url: str, dst: Path) -> Dict[str, Any]:
    # 条件分岐: `requests is None` を満たす経路を評価する。
    if requests is None:
        raise RuntimeError("requests is required to download remote files. Install it or place files under data/.../raw/")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=_REQ_TIMEOUT) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                # 条件分岐: `not chunk` を満たす経路を評価する。
                if not chunk:
                    continue

                f.write(chunk)

    tmp.replace(dst)
    return {"bytes": dst.stat().st_size, "sha256": _sha256(dst)}


def _extract_local_fits_to_npz(
    fits_path: Path,
    *,
    out_npz: Path,
    want_cols: list[str],
    max_rows: Optional[int],
    source_url: str,
) -> Dict[str, Any]:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    with fits_path.open("rb") as f:
        layout = read_first_bintable_layout(f)
        cols = read_bintable_columns(f, layout=layout, columns=want_cols, max_rows=max_rows)

    np.savez_compressed(out_npz, **cols)
    meta: Dict[str, Any] = {
        "rows_total": int(layout.n_rows),
        "rows_saved": int(next(iter(cols.values())).size) if cols else 0,
        "row_bytes": int(layout.row_bytes),
        "columns": want_cols,
        "sampling": {"method": "full"} if max_rows is None else {"method": "prefix_rows", "max_rows": int(max_rows)},
        "source_url": source_url,
        "source_path": _relpath(fits_path),
    }
    return meta


def _npz_rows(npz_path: Path) -> int:
    with np.load(npz_path) as z:
        # 条件分岐: `"RA" in z` を満たす経路を評価する。
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
    # 条件分岐: `requests is None` を満たす経路を評価する。
    if requests is None:
        raise RuntimeError("requests is required to read remote FITS layout. Install it or use local raw FITS.")

    with requests.get(url, stream=True, timeout=_REQ_TIMEOUT) as r:
        r.raise_for_status()
        layout = read_first_bintable_layout(r.raw)

    return {"rows_total": int(layout.n_rows), "row_bytes": int(layout.row_bytes)}


def _extract_remote_fits_to_npz(
    url: str,
    *,
    out_npz: Path,
    want_cols: list[str],
    max_rows: Optional[int],
) -> Dict[str, Any]:
    # 条件分岐: `requests is None` を満たす経路を評価する。
    if requests is None:
        raise RuntimeError("requests is required to stream remote files. Install it or place files under data/.../raw/")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=_REQ_TIMEOUT) as r:
        r.raise_for_status()
        f = r.raw
        layout = read_first_bintable_layout(f)
        cols = read_bintable_columns(f, layout=layout, columns=want_cols, max_rows=max_rows)

    np.savez_compressed(out_npz, **cols)
    meta: Dict[str, Any] = {
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
    """
    # 条件分岐: `int(sample_rows) <= 0` を満たす経路を評価する。
    if int(sample_rows) <= 0:
        raise ValueError("sample_rows must be > 0")

    rng = np.random.default_rng(int(seed))

    # 条件分岐: `total_rows is not None and int(total_rows) > 0` を満たす経路を評価する。
    if total_rows is not None and int(total_rows) > 0:
        k = int(sample_rows)
        margin = int(max(0.0, 5.0 * float(np.sqrt(float(k)))))
        p = float((k + margin) / float(int(total_rows)))
        p = min(max(p, 0.0), 1.0)

        buf_keys: list[np.ndarray] = []
        buf_cols: Dict[str, list[np.ndarray]] = {c: [] for c in want_cols}
        scanned = 0

        for chunk in chunks:
            # 条件分岐: `not chunk` を満たす経路を評価する。
            if not chunk:
                continue

            n = int(next(iter(chunk.values())).shape[0])
            # 条件分岐: `n <= 0` を満たす経路を評価する。
            if n <= 0:
                continue

            scanned += n
            keys = rng.random(n, dtype=np.float64)
            m = keys < p
            # 条件分岐: `not np.any(m)` を満たす経路を評価する。
            if not np.any(m):
                continue

            buf_keys.append(keys[m])
            for c in want_cols:
                buf_cols[c].append(np.asarray(chunk[c], dtype=np.float64)[m])

        # 条件分岐: `not buf_keys` を満たす経路を評価する。

        if not buf_keys:
            return ({c: np.zeros(0, dtype=np.float64) for c in want_cols}, scanned)

        keys_all = np.concatenate(buf_keys)
        # 条件分岐: `int(keys_all.size) < k` を満たす経路を評価する。
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
        # 条件分岐: `not chunk` を満たす経路を評価する。
        if not chunk:
            continue

        n = int(next(iter(chunk.values())).shape[0])
        # 条件分岐: `n <= 0` を満たす経路を評価する。
        if n <= 0:
            continue

        scanned += n
        keys = rng.random(n, dtype=np.float64)

        # 条件分岐: `res_keys is None` を満たす経路を評価する。
        if res_keys is None:
            # 条件分岐: `n <= int(sample_rows)` を満たす経路を評価する。
            if n <= int(sample_rows):
                res_keys = keys
                for c in want_cols:
                    res_cols[c] = np.asarray(chunk[c], dtype=np.float64)

                continue

            idx0 = np.argpartition(keys, int(sample_rows) - 1)[: int(sample_rows)]
            res_keys = keys[idx0]
            for c in want_cols:
                res_cols[c] = np.asarray(chunk[c], dtype=np.float64)[idx0]

            continue

        assert res_keys is not None
        cand_keys = np.concatenate([res_keys, keys], axis=0)
        k = int(sample_rows)
        idx = np.argpartition(cand_keys, k - 1)[:k]
        keep_old = idx < int(res_keys.size)
        keep_new = ~keep_old
        # 条件分岐: `not np.any(keep_new)` を満たす経路を評価する。
        if not np.any(keep_new):
            res_keys = cand_keys[idx]
            for c in want_cols:
                res_cols[c] = res_cols[c][idx]

            continue

        new_idx = idx[keep_new] - int(res_keys.size)
        res_keys = cand_keys[idx]
        for c in want_cols:
            res_cols[c] = np.concatenate([res_cols[c], np.asarray(chunk[c], dtype=np.float64)[new_idx]], axis=0)[idx]

    # 条件分岐: `res_keys is None` を満たす経路を評価する。

    if res_keys is None:
        return ({c: np.zeros(0, dtype=np.float64) for c in want_cols}, scanned)

    return (res_cols, scanned)


def _extract_remote_fits_to_npz_reservoir(
    url: str,
    *,
    out_npz: Path,
    want_cols: list[str],
    sample_rows: int,
    seed: int,
    scan_max_rows: Optional[int],
    chunk_rows: int,
) -> Dict[str, Any]:
    # 条件分岐: `requests is None` を満たす経路を評価する。
    if requests is None:
        raise RuntimeError("requests is required to stream remote files. Install it or place files under data/.../raw/")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=_REQ_TIMEOUT) as r:
        r.raise_for_status()
        f = r.raw
        layout = read_first_bintable_layout(f)
        chunks = iter_bintable_column_chunks(
            f,
            layout=layout,
            columns=want_cols,
            chunk_rows=int(chunk_rows),
            max_rows=scan_max_rows,
        )
        cols, scanned = _reservoir_sample_from_chunks(
            chunks,
            want_cols=want_cols,
            sample_rows=int(sample_rows),
            seed=int(seed),
            total_rows=(None if scan_max_rows is not None else int(layout.n_rows)),
        )

    np.savez_compressed(out_npz, **cols)
    meta: Dict[str, Any] = {
        "rows_total": int(layout.n_rows),
        "rows_scanned": int(scanned),
        "rows_saved": int(next(iter(cols.values())).size) if cols else 0,
        "row_bytes": int(layout.row_bytes),
        "columns": want_cols,
        "sampling": {
            "method": "reservoir",
            "sample_rows": int(sample_rows),
            "seed": int(seed),
            "scan_max_rows": None if scan_max_rows is None else int(scan_max_rows),
            "chunk_rows": int(chunk_rows),
        },
        "source_url": url,
    }
    return meta


def _extract_local_fits_to_npz_reservoir(
    fits_path: Path,
    *,
    out_npz: Path,
    want_cols: list[str],
    sample_rows: int,
    seed: int,
    scan_max_rows: Optional[int],
    chunk_rows: int,
    source_url: str,
) -> Dict[str, Any]:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    with fits_path.open("rb") as f:
        layout = read_first_bintable_layout(f)
        chunks = iter_bintable_column_chunks(
            f,
            layout=layout,
            columns=want_cols,
            chunk_rows=int(chunk_rows),
            max_rows=scan_max_rows,
        )
        cols, scanned = _reservoir_sample_from_chunks(
            chunks,
            want_cols=want_cols,
            sample_rows=int(sample_rows),
            seed=int(seed),
            total_rows=(None if scan_max_rows is not None else int(layout.n_rows)),
        )

    np.savez_compressed(out_npz, **cols)
    meta: Dict[str, Any] = {
        "rows_total": int(layout.n_rows),
        "rows_scanned": int(scanned),
        "rows_saved": int(next(iter(cols.values())).size) if cols else 0,
        "row_bytes": int(layout.row_bytes),
        "columns": want_cols,
        "sampling": {
            "method": "reservoir",
            "sample_rows": int(sample_rows),
            "seed": int(seed),
            "scan_max_rows": None if scan_max_rows is None else int(scan_max_rows),
            "chunk_rows": int(chunk_rows),
        },
        "source_url": source_url,
        "source_path": _relpath(fits_path),
    }
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch DESI DR1 LSS clustering catalogs and extract columns to NPZ.")
    ap.add_argument("--data-dir", default="data/cosmology/desi_dr1_lss", help="output data directory (default: data/cosmology/desi_dr1_lss)")
    ap.add_argument(
        "--raw-dir",
        default="",
        help=(
            "Optional: directory containing raw FITS (galaxy+random). "
            "If provided, read/write raw files there instead of data-dir/raw/. "
            "Useful to reuse an existing raw cache without duplicating files."
        ),
    )
    ap.add_argument("--base-url", default=_BASE_URL, help=f"DESI base URL (default: {_BASE_URL})")
    ap.add_argument("--sample", choices=sorted(_SAMPLE_FILES), default="lrg", help="sample id (default: lrg)")
    ap.add_argument("--caps", choices=["combined", "north", "south"], default="combined", help="caps selection (default: combined)")
    ap.add_argument("--random-index", type=int, default=0, help="random catalog index (default: 0)")
    ap.add_argument("--download-missing", action="store_true", help="download missing raw FITS into raw-dir (default: data-dir/raw/)")
    ap.add_argument("--stream-missing", action="store_true", help="stream remote FITS and extract without saving raw")
    ap.add_argument("--random-sampling", choices=["full", "prefix_rows", "reservoir"], default="reservoir", help="random extraction mode (default: reservoir)")
    ap.add_argument("--random-max-rows", type=int, default=2_000_000, help="random max rows (prefix/reservoir; default: 2,000,000)")
    ap.add_argument("--random-scan-max-rows", type=int, default=0, help="max rows to scan for reservoir (0=all; default: 0)")
    ap.add_argument("--sampling-seed", type=int, default=0, help="reservoir sampling seed (default: 0)")
    ap.add_argument("--chunk-rows", type=int, default=200_000, help="rows per chunk for streaming reservoir (default: 200,000)")

    args = ap.parse_args(argv)
    data_dir = (_ROOT / str(args.data_dir)).resolve() if not Path(str(args.data_dir)).is_absolute() else Path(str(args.data_dir))
    # 条件分岐: `str(args.raw_dir).strip()` を満たす経路を評価する。
    if str(args.raw_dir).strip():
        raw_dir = (_ROOT / str(args.raw_dir)).resolve() if not Path(str(args.raw_dir)).is_absolute() else Path(str(args.raw_dir))
    else:
        raw_dir = data_dir / "raw"

    ext_dir = data_dir / "extracted"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    ext_dir.mkdir(parents=True, exist_ok=True)

    base_url = str(args.base_url).rstrip("/") + "/"
    sample = str(args.sample)
    caps_req = str(args.caps)
    random_index = int(args.random_index)

    manifest_path = data_dir / "manifest.json"
    # 条件分岐: `manifest_path.exists()` を満たす経路を評価する。
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(), "source_base_url": base_url, "items": {}}

    manifest["last_run_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["source_base_url"] = base_url

    caps_to_use = ["north", "south"] if caps_req == "combined" else [caps_req]

    want_cols = ["RA", "DEC", "Z", "WEIGHT_FKP", "WEIGHT"]

    scan_max_rows_eff: Optional[int] = None
    # 条件分岐: `int(args.random_scan_max_rows) > 0` を満たす経路を評価する。
    if int(args.random_scan_max_rows) > 0:
        scan_max_rows_eff = int(args.random_scan_max_rows)

    for cap in caps_to_use:
        cap_spec = _SAMPLE_FILES[sample][cap]
        gal_name = cap_spec["galaxy"]
        rnd_name = cap_spec["random_tmpl"].format(idx=random_index)

        gal_url = base_url + gal_name
        rnd_url = base_url + rnd_name

        gal_raw = raw_dir / gal_name
        rnd_raw = raw_dir / rnd_name

        gal_npz = ext_dir / f"{gal_name}.npz"
        rnd_npz = ext_dir / f"{rnd_name}.npz"
        # 条件分岐: `str(args.random_sampling) == "reservoir"` を満たす経路を評価する。
        if str(args.random_sampling) == "reservoir":
            rnd_npz = ext_dir / f"{rnd_name}.reservoir_{int(args.random_max_rows)}_seed{int(args.sampling_seed)}.npz"
        # 条件分岐: 前段条件が不成立で、`str(args.random_sampling) == "prefix_rows"` を追加評価する。
        elif str(args.random_sampling) == "prefix_rows":
            rnd_npz = ext_dir / f"{rnd_name}.prefix_{int(args.random_max_rows)}.npz"

        key = f"{sample}:{cap}"
        old_item = manifest.get("items", {}).get(key) if isinstance(manifest.get("items", {}), dict) else None

        # Galaxy (always full; small enough for DR1 LRG).
        gal_raw_meta = {"raw_path": None, "raw_bytes": None, "raw_sha256": None}
        # 条件分岐: `gal_raw.exists()` を満たす経路を評価する。
        if gal_raw.exists():
            gal_raw_meta = {"raw_path": _relpath(gal_raw), "raw_bytes": int(gal_raw.stat().st_size), "raw_sha256": _sha256(gal_raw)}
        # 条件分岐: 前段条件が不成立で、`bool(args.download_missing)` を追加評価する。
        elif bool(args.download_missing):
            info = _download_file(gal_url, gal_raw)
            gal_raw_meta = {"raw_path": _relpath(gal_raw), "raw_bytes": int(info["bytes"]), "raw_sha256": str(info["sha256"])}

        # 条件分岐: `not gal_npz.exists()` を満たす経路を評価する。

        if not gal_npz.exists():
            # 条件分岐: `gal_raw.exists()` を満たす経路を評価する。
            if gal_raw.exists():
                gal_extract = _extract_local_fits_to_npz(gal_raw, out_npz=gal_npz, want_cols=want_cols, max_rows=None, source_url=gal_url)
            # 条件分岐: 前段条件が不成立で、`bool(args.stream_missing)` を追加評価する。
            elif bool(args.stream_missing):
                gal_extract = _extract_remote_fits_to_npz(gal_url, out_npz=gal_npz, want_cols=want_cols, max_rows=None)
            else:
                raise SystemExit(f"missing galaxy file: {gal_raw} (download it or pass --download-missing/--stream-missing)")
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
                "columns": want_cols,
                "sampling": {"method": "full"},
                "source_url": gal_url,
                "source_path": _relpath(gal_raw) if gal_raw.exists() else None,
            }
            # 条件分岐: `isinstance(old_item, dict)` を満たす経路を評価する。
            if isinstance(old_item, dict):
                old_g = old_item.get("galaxy")
                # 条件分岐: `isinstance(old_g, dict) and isinstance(old_g.get("extract"), dict)` を満たす経路を評価する。
                if isinstance(old_g, dict) and isinstance(old_g.get("extract"), dict):
                    for k2, v2 in old_g["extract"].items():
                        gal_extract.setdefault(k2, v2)

        # Random (default: reservoir)

        rnd_raw_meta = {"raw_path": None, "raw_bytes": None, "raw_sha256": None}
        # 条件分岐: `rnd_raw.exists()` を満たす経路を評価する。
        if rnd_raw.exists():
            rnd_raw_meta = {"raw_path": _relpath(rnd_raw), "raw_bytes": int(rnd_raw.stat().st_size), "raw_sha256": _sha256(rnd_raw)}

        # 条件分岐: `not rnd_npz.exists()` を満たす経路を評価する。

        if not rnd_npz.exists():
            rnd_sampling = str(args.random_sampling)
            # 条件分岐: `rnd_sampling == "full"` を満たす経路を評価する。
            if rnd_sampling == "full":
                # 条件分岐: `rnd_raw.exists()` を満たす経路を評価する。
                if rnd_raw.exists():
                    rnd_extract = _extract_local_fits_to_npz(rnd_raw, out_npz=rnd_npz, want_cols=want_cols, max_rows=None, source_url=rnd_url)
                # 条件分岐: 前段条件が不成立で、`bool(args.download_missing)` を追加評価する。
                elif bool(args.download_missing):
                    info = _download_file(rnd_url, rnd_raw)
                    rnd_raw_meta = {"raw_path": _relpath(rnd_raw), "raw_bytes": int(info["bytes"]), "raw_sha256": str(info["sha256"])}
                    rnd_extract = _extract_local_fits_to_npz(rnd_raw, out_npz=rnd_npz, want_cols=want_cols, max_rows=None, source_url=rnd_url)
                # 条件分岐: 前段条件が不成立で、`bool(args.stream_missing)` を追加評価する。
                elif bool(args.stream_missing):
                    rnd_extract = _extract_remote_fits_to_npz(rnd_url, out_npz=rnd_npz, want_cols=want_cols, max_rows=None)
                else:
                    raise SystemExit(f"missing random file: {rnd_raw} (download it or pass --download-missing/--stream-missing)")
            # 条件分岐: 前段条件が不成立で、`rnd_sampling == "prefix_rows"` を追加評価する。
            elif rnd_sampling == "prefix_rows":
                # 条件分岐: `int(args.random_max_rows) <= 0` を満たす経路を評価する。
                if int(args.random_max_rows) <= 0:
                    raise SystemExit("--random-max-rows must be > 0 for prefix_rows")

                # 条件分岐: `rnd_raw.exists()` を満たす経路を評価する。

                if rnd_raw.exists():
                    rnd_extract = _extract_local_fits_to_npz(
                        rnd_raw, out_npz=rnd_npz, want_cols=want_cols, max_rows=int(args.random_max_rows), source_url=rnd_url
                    )
                # 条件分岐: 前段条件が不成立で、`bool(args.download_missing)` を追加評価する。
                elif bool(args.download_missing):
                    info = _download_file(rnd_url, rnd_raw)
                    rnd_raw_meta = {"raw_path": _relpath(rnd_raw), "raw_bytes": int(info["bytes"]), "raw_sha256": str(info["sha256"])}
                    rnd_extract = _extract_local_fits_to_npz(
                        rnd_raw, out_npz=rnd_npz, want_cols=want_cols, max_rows=int(args.random_max_rows), source_url=rnd_url
                    )
                # 条件分岐: 前段条件が不成立で、`bool(args.stream_missing)` を追加評価する。
                elif bool(args.stream_missing):
                    rnd_extract = _extract_remote_fits_to_npz(rnd_url, out_npz=rnd_npz, want_cols=want_cols, max_rows=int(args.random_max_rows))
                else:
                    raise SystemExit(f"missing random file: {rnd_raw} (download it or pass --download-missing/--stream-missing)")
            # 条件分岐: 前段条件が不成立で、`rnd_sampling == "reservoir"` を追加評価する。
            elif rnd_sampling == "reservoir":
                # 条件分岐: `int(args.random_max_rows) <= 0` を満たす経路を評価する。
                if int(args.random_max_rows) <= 0:
                    raise SystemExit("--random-max-rows must be > 0 for reservoir")

                # 条件分岐: `rnd_raw.exists()` を満たす経路を評価する。

                if rnd_raw.exists():
                    rnd_extract = _extract_local_fits_to_npz_reservoir(
                        rnd_raw,
                        out_npz=rnd_npz,
                        want_cols=want_cols,
                        sample_rows=int(args.random_max_rows),
                        seed=int(args.sampling_seed),
                        scan_max_rows=scan_max_rows_eff,
                        chunk_rows=int(args.chunk_rows),
                        source_url=rnd_url,
                    )
                else:
                    # 条件分岐: `not bool(args.stream_missing)` を満たす経路を評価する。
                    if not bool(args.stream_missing):
                        raise SystemExit(
                            "reservoir sampling needs either a local raw FITS under raw-dir, or --stream-missing "
                            "(to avoid storing multi-GB raw randoms)"
                        )

                    rnd_extract = _extract_remote_fits_to_npz_reservoir(
                        rnd_url,
                        out_npz=rnd_npz,
                        want_cols=want_cols,
                        sample_rows=int(args.random_max_rows),
                        seed=int(args.sampling_seed),
                        scan_max_rows=scan_max_rows_eff,
                        chunk_rows=int(args.chunk_rows),
                    )
            else:
                raise SystemExit(f"invalid --random-sampling: {rnd_sampling}")
        else:
            rows_saved = _npz_rows(rnd_npz)
            rnd_sampling = str(args.random_sampling)
            try:
                layout = _read_layout_local(rnd_raw) if rnd_raw.exists() else _read_layout_remote(rnd_url)
            except Exception:
                layout = {"rows_total": None, "row_bytes": None}

            rows_total = layout.get("rows_total")
            row_bytes = layout.get("row_bytes")
            # 条件分岐: `rnd_sampling == "reservoir"` を満たす経路を評価する。
            if rnd_sampling == "reservoir":
                # 条件分岐: `scan_max_rows_eff is None` を満たす経路を評価する。
                if scan_max_rows_eff is None:
                    rows_scanned = int(rows_total) if rows_total is not None else None
                else:
                    rows_scanned = int(min(int(scan_max_rows_eff), int(rows_total))) if rows_total is not None else int(scan_max_rows_eff)

                rnd_extract = {
                    "rows_total": int(rows_total) if rows_total is not None else None,
                    "rows_scanned": rows_scanned,
                    "rows_saved": int(rows_saved),
                    "row_bytes": int(row_bytes) if row_bytes is not None else None,
                    "columns": want_cols,
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
            # 条件分岐: 前段条件が不成立で、`rnd_sampling == "prefix_rows"` を追加評価する。
            elif rnd_sampling == "prefix_rows":
                rnd_extract = {
                    "rows_total": int(rows_total) if rows_total is not None else None,
                    "rows_saved": int(rows_saved),
                    "row_bytes": int(row_bytes) if row_bytes is not None else None,
                    "columns": want_cols,
                    "sampling": {"method": "prefix_rows", "max_rows": int(args.random_max_rows)},
                    "source_url": rnd_url,
                    "source_path": _relpath(rnd_raw) if rnd_raw.exists() else None,
                }
            else:
                rnd_extract = {
                    "rows_total": int(rows_total) if rows_total is not None else None,
                    "rows_saved": int(rows_saved),
                    "row_bytes": int(row_bytes) if row_bytes is not None else None,
                    "columns": want_cols,
                    "sampling": {"method": "full"},
                    "source_url": rnd_url,
                    "source_path": _relpath(rnd_raw) if rnd_raw.exists() else None,
                }

            # 条件分岐: `isinstance(old_item, dict)` を満たす経路を評価する。

            if isinstance(old_item, dict):
                old_r = old_item.get("random")
                # 条件分岐: `isinstance(old_r, dict) and isinstance(old_r.get("extract"), dict)` を満たす経路を評価する。
                if isinstance(old_r, dict) and isinstance(old_r.get("extract"), dict):
                    for k2, v2 in old_r["extract"].items():
                        rnd_extract.setdefault(k2, v2)

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
                "random_index": int(random_index),
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
                "event_type": "fetch_desi_dr1_lss",
                "argv": sys.argv,
                "params": {
                    "sample": sample,
                    "caps": caps_req,
                    "base_url": base_url,
                    "random_index": int(random_index),
                    "download_missing": bool(args.download_missing),
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


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
