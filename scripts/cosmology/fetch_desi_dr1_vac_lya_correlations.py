#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_desi_dr1_vac_lya_correlations.py

Phase 4 / Step 4.5B.21.4.4.7（DESI DR1 multi-tracer）:
DESI DR1 VAC "lya-correlations"（Y1 BAO; Lyα forest auto/cross）を取得し、
ローカルへキャッシュして offline 再現可能にする。

データソース（DESI DR1 VAC; WebDAV mirror）:
  - https://webdav-hdfs.pic.es/data/public/DESI/DR1/vac/dr1/lya-correlations/v1.0/

出力（固定）:
  data/cosmology/desi_dr1_vac_lya_correlations_v1p0/
    - raw/*.fits
    - raw/dr1_vac_dr1_lya-correlations_v1.0.sha256sum
    - manifest.json

注意：
- `cf_*.fits` / `full-covariance-smoothed.fits` は巨大（数百MB〜数GB）。
- dmat_* はさらに巨大になり得るため、既定ではダウンロードしない（--include-dmat で有効化）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


_DEFAULT_BASE_URL = "https://webdav-hdfs.pic.es/data/public/DESI/DR1/vac/dr1/lya-correlations/v1.0/"
_REQ_TIMEOUT: Tuple[int, int] = (30, 1800)  # connect, read


# 関数: `_now_utc` の入出力契約と処理意図を定義する。
def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_sha256` の入出力契約と処理意図を定義する。

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


# 関数: `_relpath` の入出力契約と処理意図を定義する。

def _relpath(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


# 関数: `_download` の入出力契約と処理意図を定義する。

def _download(url: str, dst: Path) -> Dict[str, Any]:
    # 条件分岐: `requests is None` を満たす経路を評価する。
    if requests is None:
        raise RuntimeError("requests is required (pip install requests)")

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
    return {"bytes": int(dst.stat().st_size), "sha256": _sha256(dst)}


# 関数: `_parse_sha256sum` の入出力契約と処理意図を定義する。

def _parse_sha256sum(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in str(text).splitlines():
        s = line.strip()
        # 条件分岐: `not s or s.startswith("#")` を満たす経路を評価する。
        if not s or s.startswith("#"):
            continue

        parts = s.split()
        # 条件分岐: `len(parts) < 2` を満たす経路を評価する。
        if len(parts) < 2:
            continue

        sha = parts[0].strip().lower()
        name = parts[-1].strip()
        # 条件分岐: `len(sha) == 64 and name` を満たす経路を評価する。
        if len(sha) == 64 and name:
            out[name] = sha

    return out


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch DESI DR1 VAC: lya-correlations (v1.0) into data/ cache.")
    ap.add_argument(
        "--data-dir",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "desi_dr1_vac_lya_correlations_v1p0"),
        help="Output data dir (default: data/cosmology/desi_dr1_vac_lya_correlations_v1p0)",
    )
    ap.add_argument("--base-url", type=str, default=_DEFAULT_BASE_URL, help="Base URL (default: WebDAV mirror v1.0)")
    ap.add_argument("--download-missing", action="store_true", help="Download missing files (default: off)")
    ap.add_argument("--include-dmat", action="store_true", help="Also fetch dmat_*.fits (default: off)")
    ap.add_argument("--offline", action="store_true", help="Offline mode: do not download (default: off)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    data_dir = Path(str(args.data_dir)).resolve()
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    base_url = str(args.base_url).rstrip("/") + "/"

    want = [
        "dr1_vac_dr1_lya-correlations_v1.0.sha256sum",
        "cf_lya_x_lya_exp.fits",
        "cf_lya_x_lyb_exp.fits",
        "cf_qso_x_lya_exp.fits",
        "cf_qso_x_lyb_exp.fits",
        "full-covariance-smoothed.fits",
    ]
    # 条件分岐: `bool(args.include_dmat)` を満たす経路を評価する。
    if bool(args.include_dmat):
        want.extend(
            [
                "dmat_lya_x_lya.fits",
                "dmat_lya_x_lyb.fits",
                "dmat_qso_x_lya.fits",
                "dmat_qso_x_lyb.fits",
            ]
        )

    # Fetch sha256sum (small) first so we can validate as we go.

    sha_expected: Dict[str, str] = {}
    sha_path = raw_dir / "dr1_vac_dr1_lya-correlations_v1.0.sha256sum"
    # 条件分岐: `sha_path.exists()` を満たす経路を評価する。
    if sha_path.exists():
        sha_expected = _parse_sha256sum(sha_path.read_text(encoding="utf-8", errors="ignore"))
    # 条件分岐: 前段条件が不成立で、`(not bool(args.offline)) and bool(args.download_missing)` を追加評価する。
    elif (not bool(args.offline)) and bool(args.download_missing):
        meta = _download(base_url + sha_path.name, sha_path)
        sha_expected = _parse_sha256sum(sha_path.read_text(encoding="utf-8", errors="ignore"))
        meta  # keep for manifest (collected below)

    files_meta: Dict[str, Any] = {}
    for name in want:
        dst = raw_dir / name
        url = base_url + name
        item: Dict[str, Any] = {"url": url, "path": _relpath(dst)}
        # 条件分岐: `dst.exists()` を満たす経路を評価する。
        if dst.exists():
            item["bytes"] = int(dst.stat().st_size)
            item["sha256"] = _sha256(dst)
        # 条件分岐: 前段条件が不成立で、`(not bool(args.offline)) and bool(args.download_missing)` を追加評価する。
        elif (not bool(args.offline)) and bool(args.download_missing):
            item.update(_download(url, dst))

        exp = sha_expected.get(name)
        # 条件分岐: `exp` を満たす経路を評価する。
        if exp:
            item["sha256_expected"] = exp
            got = str(item.get("sha256") or "").lower()
            item["sha256_ok"] = bool(got and got == exp)

        files_meta[name] = item

    manifest = {
        "generated_utc": _now_utc(),
        "domain": "cosmology",
        "step": "4.5B.21.4.4.7 (DESI DR1 VAC lya-correlations fetch)",
        "params": {"data_dir": _relpath(data_dir), "base_url": base_url, "include_dmat": bool(args.include_dmat)},
        "files": files_meta,
        "notes": [
            "dmat_* は巨大になり得るため既定では取得しない（--include-dmat で有効化）。",
            "cf_* は BINTABLE で、DA/RP/RT/Z/NB は scalar、CO/DM は row-major の巨大行列として格納される。",
            "full-covariance-smoothed.fits は 15000x15000 の共分散（行ごとに 15000D）。",
        ],
    }
    out_manifest = data_dir / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        worklog.append_event(
            {
                "event_type": "fetch_desi_dr1_vac_lya_correlations",
                "argv": sys.argv,
                "params": manifest.get("params", {}),
                "outputs": {"manifest_json": str(out_manifest)},
            }
        )
    except Exception:
        pass

    missing = [k for k, v in files_meta.items() if not Path(str(_ROOT / v["path"])).exists()]
    ok_n = sum(1 for v in files_meta.values() if v.get("sha256_ok") is True)
    exp_n = sum(1 for v in files_meta.values() if v.get("sha256_expected"))
    print(f"[ok] wrote: {out_manifest}")
    # 条件分岐: `exp_n` を満たす経路を評価する。
    if exp_n:
        print(f"[ok] sha256_ok: {ok_n}/{exp_n}")

    # 条件分岐: `missing` を満たす経路を評価する。

    if missing:
        print("[warn] missing files (run with --download-missing):")
        for m in missing:
            print(f"  - {m}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

