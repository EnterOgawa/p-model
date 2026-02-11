#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_xrism_heasarc.py

Phase 4（宇宙論）/ Step 4.8（XRISM）:
HEASARC（XRISM archive）の公開一次データ（FITS）を obsid 単位で取得・キャッシュし、
offline 再現できる形に manifest を固定する。併せて最小のスペクトルQC図を生成する。

出力（固定）:
- data/xrism/heasarc/manifest.json（取得条件・obsid・ファイル一覧・sha256）
- data/xrism/heasarc/obs/<cat>/<obsid>/...（キャッシュ）
- output/xrism/<obsid>__spectrum_qc.png（最小QC；PHAが取得できた場合）

注意:
- HEASARC の XRISM データは mission archive の設計に従い、obsid 配下に
  auxil/derived/products/raw 等のディレクトリを持つ。
- 本スクリプトは「取得と可視化の土台」を提供し、線同定や物理パラメータ推定は
  Step 4.8.2 以降で別スクリプトとして拡張する。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse

import numpy as np

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency for offline/local runs
    requests = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cosmology.boss_dr12v5_fits import read_bintable_columns, read_first_bintable_layout  # noqa: E402
from scripts.summary import worklog  # noqa: E402

_DEFAULT_BASE_URL = "https://heasarc.gsfc.nasa.gov/FTP/xrism/data/obs"
_REQ_TIMEOUT = (30, 600)  # (connect, read)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not isinstance(r, dict):
                continue
            rows.append({str(k): (v or "").strip() for k, v in r.items() if k is not None})
    return rows


def _ensure_targets_template(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["obsid", "target_name", "role", "z_sys", "instrument_prefer", "comment", "remote_cat_hint"]
    example = {
        "obsid": "000126000",
        "target_name": "N132D",
        "role": "snr_qc",
        "z_sys": "",
        "instrument_prefer": "resolve",
        "comment": "SNR（QC/energy-scaleの基準候補）。まず products の取得でスペクトルQCを確認。",
        "remote_cat_hint": "0",
    }
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerow(example)


def _http_get_text(url: str) -> str:
    if requests is None:
        raise RuntimeError("requests is required for online mode")
    r = requests.get(url, timeout=_REQ_TIMEOUT)
    r.raise_for_status()
    r.encoding = r.encoding or "utf-8"
    return r.text


def _http_exists(url: str) -> bool:
    if requests is None:
        return False
    try:
        r = requests.get(url, timeout=(10, 30), stream=True)
    except Exception:
        return False
    try:
        ok = (200 <= int(getattr(r, "status_code", 0)) < 300) and bool(r.headers.get("Content-Type", ""))
    finally:
        try:
            r.close()
        except Exception:
            pass
    return ok


def _parse_apache_index_links(html: str) -> List[str]:
    # HEASARC FTP over HTTPS typically exposes an auto index page. Keep parsing lightweight.
    hrefs = re.findall(r'href=[\"\\\']([^\"\\\']+)[\"\\\']', html, flags=re.IGNORECASE)
    out: List[str] = []
    for h in hrefs:
        h = (h or "").strip()
        if not h:
            continue
        if h in ("../", "./"):
            continue
        # Apache index sometimes includes absolute paths (e.g. "/FTP/...") for parent links.
        # We must not follow them, otherwise a recursive walk can escape the intended subtree.
        if h.startswith("/"):
            continue
        if h.lower().startswith(("http://", "https://")):
            continue
        if h.startswith("?") or h.startswith("#"):
            continue
        h = h.split("#", 1)[0].split("?", 1)[0]
        out.append(h)
    # Deduplicate while preserving order
    seen = set()
    dedup: List[str] = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        dedup.append(x)
    return dedup


def _list_remote_files_recursive(dir_url: str, *, max_files: Optional[int] = None) -> List[str]:
    if not dir_url.endswith("/"):
        dir_url += "/"
    files: List[str] = []
    stack: List[str] = [dir_url]
    visited: set[str] = set()

    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)

        html = _http_get_text(cur)
        for href in _parse_apache_index_links(html):
            full = urljoin(cur, href)
            if href.endswith("/"):
                stack.append(full)
                continue
            files.append(full)
            if max_files is not None and len(files) >= max_files:
                return files

    return files


def _list_remote_dir_hrefs(dir_url: str) -> List[str]:
    if not dir_url.endswith("/"):
        dir_url += "/"
    html = _http_get_text(dir_url)
    return _parse_apache_index_links(html)


def _infer_remote_obs_root(
    *,
    base_url: str,
    obsid: str,
    cat_hint: Optional[str],
    cat_candidates: Sequence[str] = tuple(str(i) for i in range(10)),
) -> Tuple[Optional[str], Optional[str]]:
    base_url = base_url.rstrip("/")
    candidates: List[str] = []
    if cat_hint:
        candidates.append(cat_hint.strip())
    candidates.extend([c for c in cat_candidates if c != cat_hint])

    for cat in candidates:
        url = f"{base_url}/{cat}/{obsid}/"
        if _http_exists(url):
            return cat, url
    return None, None


def _download_file(url: str, out_path: Path) -> None:
    if requests is None:
        raise RuntimeError("requests is required for online mode")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=_REQ_TIMEOUT, stream=True)
    r.raise_for_status()
    with out_path.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)


def _find_first_file(root: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(root.glob(pat))
        for p in hits:
            if p.is_file():
                return p
    return None


def _load_pha_counts(pha_path: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    if read_first_bintable_layout is None or read_bintable_columns is None:
        raise RuntimeError("FITS reader helpers are unavailable (scripts/cosmology/boss_dr12v5_fits.py import failed)")

    opener = gzip.open if pha_path.name.endswith(".gz") else Path.open
    with opener(pha_path, "rb") as f:  # type: ignore[arg-type]
        layout = read_first_bintable_layout(f)
        col_map = {str(c).upper(): str(c) for c in layout.columns}

        ch_key = col_map.get("CHANNEL")
        cnt_key = col_map.get("COUNTS")
        rate_key = col_map.get("RATE")

        if ch_key is None:
            raise RuntimeError("PHA table has no CHANNEL column (unsupported layout)")

        if cnt_key is None and rate_key is None:
            raise RuntimeError("PHA table has no COUNTS/RATE column")

        want = [ch_key, cnt_key or rate_key]  # type: ignore[list-item]
        cols = read_bintable_columns(f, layout=layout, columns=want)
        x = np.asarray(cols[ch_key], dtype=float)
        if cnt_key is not None:
            y = np.asarray(cols[cnt_key], dtype=float)
            y_label = "counts"
        else:
            y = np.asarray(cols[rate_key], dtype=float)  # type: ignore[index]
            y_label = "rate"

    return x, y, y_label


def _try_channel_to_energy_keV(rmf_path: Optional[Path]) -> Optional[Dict[int, float]]:
    if rmf_path is None:
        return None
    if read_first_bintable_layout is None or read_bintable_columns is None:
        return None
    try:
        opener = gzip.open if rmf_path.name.endswith(".gz") else Path.open
        with opener(rmf_path, "rb") as f:  # type: ignore[arg-type]
            layout = read_first_bintable_layout(f)
            col_map = {str(c).upper(): str(c) for c in layout.columns}
            ch_key = col_map.get("CHANNEL")
            e_min_key = col_map.get("E_MIN")
            e_max_key = col_map.get("E_MAX")
            if ch_key is None or e_min_key is None or e_max_key is None:
                return None
            cols = read_bintable_columns(f, layout=layout, columns=[ch_key, e_min_key, e_max_key])
            ch = np.asarray(cols[ch_key], dtype=int)
            e_min = np.asarray(cols[e_min_key], dtype=float)
            e_max = np.asarray(cols[e_max_key], dtype=float)
            e_mid = 0.5 * (e_min + e_max)
            return {int(c): float(e) for c, e in zip(ch, e_mid)}
    except Exception:
        return None


def _write_spectrum_qc(obsid: str, pha_path: Path, rmf_path: Optional[Path], out_png: Path) -> Optional[str]:
    if plt is None:
        return "matplotlib is not available"
    try:
        ch, y, y_label = _load_pha_counts(pha_path)
    except Exception as e:
        return f"failed to read PHA: {e}"

    ch_to_e = _try_channel_to_energy_keV(rmf_path)
    x = ch
    x_label = "channel"
    if ch_to_e is not None:
        e = np.array([ch_to_e.get(int(c), float("nan")) for c in ch], dtype=float)
        if np.isfinite(e).any():
            x = e
            x_label = "energy (keV)"

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 4.2), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, lw=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"XRISM spectrum QC (obsid={obsid})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=_DEFAULT_BASE_URL, help="HEASARC XRISM base URL (obs root)")
    p.add_argument("--targets", default=str((_ROOT / "output" / "xrism" / "xrism_targets_catalog.csv")), help="targets CSV")
    p.add_argument("--obsid", action="append", default=[], help="override: obsid(s) to fetch (can repeat)")
    p.add_argument(
        "--instrument",
        action="append",
        default=[],
        choices=["resolve", "xtend"],
        help="limit instruments to fetch (repeatable; default=auto)",
    )
    p.add_argument("--download-missing", action="store_true", help="download missing files")
    p.add_argument("--offline", action="store_true", help="do not access network; use existing cache only")
    p.add_argument(
        "--download-scope",
        choices=["products", "derived", "raw", "auxil", "all"],
        default="products",
        help="which subdirectory to download under obsid root",
    )
    p.add_argument(
        "--include-regex",
        action="append",
        default=[],
        help="download only files whose basename matches any regex (repeatable; online mode only)",
    )
    p.add_argument("--max-files", type=int, default=5000, help="safety cap for remote listing per subdir")
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.offline:
        download_missing = False
    else:
        download_missing = bool(args.download_missing)
        if requests is None:
            raise RuntimeError("requests is required for online mode")

    include_patterns: List[str] = [str(x) for x in (args.include_regex or []) if str(x)]
    if download_missing and (args.download_scope == "products") and not include_patterns:
        # Default: enough for minimal spectrum QC.
        include_patterns = [r"(?:\.rmf|\.arf|\.pha|\.pi)(?:\.gz)?$"]
    include_res = [re.compile(pat, flags=re.IGNORECASE) for pat in include_patterns]

    targets_csv = Path(args.targets)
    _ensure_targets_template(targets_csv)
    rows = _read_csv_rows(targets_csv)

    obsids: List[str] = []
    if args.obsid:
        obsids = [str(x).strip() for x in args.obsid if str(x).strip()]
    else:
        for r in rows:
            o = (r.get("obsid") or "").strip()
            if o:
                obsids.append(o)
    obsids = sorted(set(obsids))
    if not obsids:
        print(f"[warn] no obsids found (targets={targets_csv})")
        return 0

    out_manifest = _ROOT / "data" / "xrism" / "heasarc" / "manifest.json"
    out_dir = _ROOT / "output" / "xrism"
    data_root = _ROOT / "data" / "xrism" / "heasarc" / "obs"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_obsid: Dict[str, Any] = {}
    for obsid in obsids:
        row = next((r for r in rows if (r.get("obsid") or "").strip() == obsid), {})
        cat_hint = (row.get("remote_cat_hint") or "").strip() or None

        cat = None
        remote_root = None
        if not args.offline:
            cat, remote_root = _infer_remote_obs_root(base_url=args.base_url, obsid=obsid, cat_hint=cat_hint)
        else:
            # Best-effort: infer from local cache (if any)
            for cand in sorted(data_root.glob(f"*/{obsid}")):
                if cand.is_dir():
                    cat = cand.parent.name
                    remote_root = f"{args.base_url.rstrip('/')}/{cat}/{obsid}/"
                    break

        item: Dict[str, Any] = {
            "obsid": obsid,
            "target_name": (row.get("target_name") or "").strip(),
            "role": (row.get("role") or "").strip(),
            "remote": {"base_url": args.base_url, "cat": cat, "obs_root": remote_root},
            "download_scope": args.download_scope,
            "subpaths": [],
            "status": "not_found" if remote_root is None else "ok",
            "files": [],
            "qc": {},
        }

        if remote_root is None:
            by_obsid[obsid] = item
            continue

        local_obs_root = data_root / str(cat) / obsid
        local_obs_root.mkdir(parents=True, exist_ok=True)

        instrument_dirs: List[str] = []
        remote_paths: List[str] = []
        if not args.offline:
            try:
                hrefs = _list_remote_dir_hrefs(remote_root)
                for h in hrefs:
                    if not h.endswith("/"):
                        continue
                    name = h.rstrip("/")
                    if name in ("resolve", "xtend"):
                        instrument_dirs.append(name)
            except Exception:
                instrument_dirs = []
        else:
            for cand in ("resolve", "xtend"):
                if (local_obs_root / cand).is_dir():
                    instrument_dirs.append(cand)

        if args.instrument:
            allow = {str(x) for x in args.instrument}
            instrument_dirs = [x for x in instrument_dirs if x in allow]

        # Build which subpaths to include under obsroot.
        if args.download_scope == "products":
            for inst in instrument_dirs:
                remote_paths.append(f"{inst}/products")
        elif args.download_scope == "derived":
            for inst in instrument_dirs:
                remote_paths.append(f"{inst}/event_cl")
        elif args.download_scope == "raw":
            for inst in instrument_dirs:
                remote_paths.extend([f"{inst}/event_uf", f"{inst}/hk"])
        elif args.download_scope == "auxil":
            remote_paths.extend(["auxil", "log"])
        elif args.download_scope == "all":
            remote_paths.extend(["auxil", "log"])
            remote_paths.extend(instrument_dirs)
        else:
            remote_paths = []

        # Fallback: if we couldn't infer instrument dirs but products was requested,
        # try listing obsroot and include everything (still capped by --max-files).
        if not remote_paths and args.download_scope == "products":
            remote_paths = instrument_dirs or ["resolve/products", "xtend/products"]

        item["instrument_dirs"] = instrument_dirs
        item["subpaths"] = remote_paths

        files: List[str] = []
        if not args.offline:
            for sub in remote_paths:
                files.extend(_list_remote_files_recursive(urljoin(remote_root, sub.rstrip("/") + "/"), max_files=args.max_files))

        # Offline: enumerate local files only (no remote listing)
        if args.offline:
            for sub in remote_paths:
                for pth in sorted((local_obs_root / sub).glob("**/*")):
                    if pth.is_file():
                        # Reconstruct remote URL deterministically
                        rel = pth.relative_to(local_obs_root).as_posix()
                        files.append(urljoin(remote_root, rel))

        # Download/cache
        for file_url in sorted(set(files)):
            if include_res and not args.offline:
                name = PurePosixPath(urlparse(file_url).path).name
                if not any(rx.search(name) for rx in include_res):
                    continue
            u_path = PurePosixPath(urlparse(file_url).path)
            root_path = PurePosixPath(urlparse(remote_root).path)
            try:
                rel = u_path.relative_to(root_path).as_posix()
            except Exception:
                # fallback: keep only last 2 path segments
                rel = "/".join(str(u_path).strip("/").split("/")[-2:])

            local_path = local_obs_root / Path(rel)
            downloaded = False
            err = None
            need_download = bool(download_missing)
            if need_download and local_path.exists():
                try:
                    # Treat zero-length files as incomplete cache (e.g., interrupted download).
                    need_download = int(local_path.stat().st_size) <= 0
                except Exception:
                    need_download = True
            if need_download:
                try:
                    _download_file(file_url, local_path)
                    downloaded = True
                except Exception as e:
                    err = str(e)

            sha = None
            size = None
            if local_path.exists():
                try:
                    size = int(local_path.stat().st_size)
                    sha = _sha256(local_path)
                except Exception:
                    pass

            item["files"].append(
                {
                    "url": file_url,
                    "rel": str(Path(rel).as_posix()),
                    "local": _relpath(local_path),
                    "size_bytes": size,
                    "sha256": sha,
                    "downloaded": downloaded,
                    "error": err,
                }
            )

        # QC (best-effort): find first PHA and RMF under products
        qc_err: Optional[str] = None
        qc_png = out_dir / f"{obsid}__spectrum_qc.png"
        product_roots = []
        for inst in (instrument_dirs or ["resolve", "xtend"]):
            pdir = local_obs_root / inst / "products"
            if pdir.is_dir():
                product_roots.append(pdir)
        if not product_roots:
            product_roots = [local_obs_root]
        pha = None
        rmf = None
        for pr in product_roots:
            pha = _find_first_file(
                pr,
                patterns=[
                    "**/*.pha",
                    "**/*.pha.gz",
                    "**/*.pi",
                    "**/*.pi.gz",
                    "**/*pha*.fits",
                    "**/*pha*.fits.gz",
                    "**/*pi*.fits",
                    "**/*pi*.fits.gz",
                ],
            )
            rmf = _find_first_file(pr, patterns=["**/*.rmf", "**/*.rmf.gz", "**/*rmf*.fits", "**/*rmf*.fits.gz"])
            if pha is not None:
                break
        if pha is not None:
            qc_err = _write_spectrum_qc(obsid, pha, rmf, qc_png)
        else:
            qc_err = "no PHA found under */products/"

        item["qc"] = {
            "qc_png": _relpath(qc_png) if qc_png.exists() else None,
            "pha": _relpath(pha),
            "rmf": _relpath(rmf),
            "error": qc_err,
        }

        by_obsid[obsid] = item

    manifest: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "mode": "offline" if args.offline else "online",
        "base_url": args.base_url,
        "download_missing": bool(download_missing),
        "download_scope": args.download_scope,
        "targets_csv": _relpath(targets_csv),
        "obsids": obsids,
        "obs": by_obsid,
        "outputs": {
            "manifest_json": _relpath(out_manifest),
            "qc_dir": _relpath(out_dir),
        },
    }
    _write_json(out_manifest, manifest)

    try:
        worklog.append_event(
            {
                "event_type": "xrism_fetch",
                "argv": list(sys.argv),
                "inputs": {"targets_csv": targets_csv},
                "outputs": {"manifest_json": out_manifest, "qc_dir": out_dir},
                "summary": {"obsids": obsids, "download_scope": args.download_scope, "offline": bool(args.offline)},
            }
        )
    except Exception:
        pass

    print(f"[ok] targets : {targets_csv}")
    print(f"[ok] manifest : {out_manifest}")
    print(f"[ok] qc dir   : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
