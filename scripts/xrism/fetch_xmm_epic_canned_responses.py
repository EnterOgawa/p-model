#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_xmm_epic_canned_responses.py

Phase 4 / Step 4.13.7（Fe-Kα relativistic broad line / ISCO）:
XMM-Newton PPS のスペクトル（SRSPEC）の RESPFILE が参照する
EPIC の canned RMF を HEASARC CALDB から取得し、offline 再現できる形でキャッシュする。

背景：
- XMM PPS の EPIC スペクトルは、ARF（ANCRFILE）は PPS 内に同梱される一方、
  RMF（RESPFILE）は “canned response file name” として header に書かれているだけで
  PPS ディレクトリに存在しない場合がある（例：m1_e7_im_pall_c.rmf）。
- Step 4.13.7 の RMF/ARF 畳み込み fit（reflection / diskline）へ進むため、
  RESPFILE を一次ソースとして取得→sha256 を manifest に固定する。

入手先（一次ソース）：
- HEASARC CALDB: https://heasarc.gsfc.nasa.gov/FTP/caldb/data/xmm/ccf/extras/responses/
  - MOS: .../MOS/
  - PN : .../PN/

出力（固定）：
- raw cache: data/xrism/xmm_epic_responses/{MOS|PN}/<rmf>
- manifest : data/xrism/sources/xmm_epic_responses_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402
from scripts.xrism.fek_relativistic_broadening_isco_constraints import _fits_read_spectrum_header  # noqa: E402

_REQ_TIMEOUT = (30, 600)  # (connect, read)

_BASE = "https://heasarc.gsfc.nasa.gov/FTP/caldb/data/xmm/ccf/extras/responses/"
_BASE_MOS = urljoin(_BASE, "MOS/")
_BASE_PN = urljoin(_BASE, "PN/")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _download(url: str, *, dst: Path, force: bool) -> Dict[str, Any]:
    # 条件分岐: `requests is None` を満たす経路を評価する。
    if requests is None:  # pragma: no cover
        raise RuntimeError("requests is required for online fetch")

    dst.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `dst.exists() and not force` を満たす経路を評価する。
    if dst.exists() and not force:
        return {
            "status": "skipped_exists",
            "path": _rel(dst),
            "bytes": int(dst.stat().st_size),
            "sha256": _sha256_file(dst),
            "url": url,
        }

    r = requests.get(url, timeout=_REQ_TIMEOUT, stream=True)
    r.raise_for_status()
    h = hashlib.sha256()
    n = 0
    with dst.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            # 条件分岐: `not chunk` を満たす経路を評価する。
            if not chunk:
                continue

            f.write(chunk)
            h.update(chunk)
            n += len(chunk)

    return {"status": "downloaded", "path": _rel(dst), "bytes": int(n), "sha256": h.hexdigest(), "url": url}


_SRSPEC_RE = re.compile(r"SRSPEC", flags=re.IGNORECASE)


def _iter_xmm_srspec_files(cache_root: Path, *, obsids: Optional[Sequence[str]] = None) -> Iterable[Path]:
    base = cache_root / "xmm"
    # 条件分岐: `not base.exists()` を満たす経路を評価する。
    if not base.exists():
        return []

    wanted = {str(x).strip() for x in (obsids or []) if str(x).strip()}
    for rev_dir in sorted(base.glob("rev*")):
        # 条件分岐: `not rev_dir.is_dir()` を満たす経路を評価する。
        if not rev_dir.is_dir():
            continue

        for obs_dir in sorted(rev_dir.glob("*")):
            # 条件分岐: `not obs_dir.is_dir()` を満たす経路を評価する。
            if not obs_dir.is_dir():
                continue

            obsid = obs_dir.name
            # 条件分岐: `wanted and obsid not in wanted` を満たす経路を評価する。
            if wanted and obsid not in wanted:
                continue

            pps = obs_dir / "PPS"
            # 条件分岐: `not pps.exists()` を満たす経路を評価する。
            if not pps.exists():
                continue

            for p in sorted(pps.glob("*.FTZ")):
                # 条件分岐: `_SRSPEC_RE.search(p.name)` を満たす経路を評価する。
                if _SRSPEC_RE.search(p.name):
                    yield p


def _resp_base_for_name(respfile: str) -> Optional[Tuple[str, str]]:
    s = str(respfile).strip().strip("'")
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        return None

    low = s.lower()
    # 条件分岐: `low.startswith(("m1_", "m2_", "m11_", "m21_"))` を満たす経路を評価する。
    if low.startswith(("m1_", "m2_", "m11_", "m21_")):
        return "MOS", _BASE_MOS

    # 条件分岐: `low.startswith(("epn_", "pn_", "pnu_", "p11_", "p21_"))` を満たす経路を評価する。

    if low.startswith(("epn_", "pn_", "pnu_", "p11_", "p21_")):
        return "PN", _BASE_PN

    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch XMM EPIC canned RMFs referenced by PPS spectra (RESPFILE).")
    ap.add_argument(
        "--xmm-obsid",
        action="append",
        default=[],
        help="Limit scan to specific XMM obsid (10 digits). Repeatable. Default: scan all cached XMM obsids.",
    )
    ap.add_argument(
        "--respfile",
        action="append",
        default=[],
        help="Additional RESPFILE names to fetch (repeatable). Example: m1_e7_im_pall_c.rmf",
    )
    ap.add_argument("--download-missing", action="store_true", help="Download missing files (online).")
    ap.add_argument("--offline", action="store_true", help="Offline mode: do not access the network.")
    ap.add_argument("--force", action="store_true", help="Force re-download even if local file exists.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    cache_root = _ROOT / "data" / "xrism" / "xmm_nustar"
    out_dir = _ROOT / "data" / "xrism" / "xmm_epic_responses"
    out_manifest = _ROOT / "data" / "xrism" / "sources" / "xmm_epic_responses_manifest.json"

    # 条件分岐: `args.offline and args.download_missing` を満たす経路を評価する。
    if args.offline and args.download_missing:
        ap.error("--offline and --download-missing cannot be used together")

    # 条件分岐: `(not args.offline) and requests is None` を満たす経路を評価する。

    if (not args.offline) and requests is None:
        raise RuntimeError("requests is required for online fetch")

    obsids = [str(x).strip() for x in (args.xmm_obsid or []) if str(x).strip()]
    extra = [str(x).strip() for x in (args.respfile or []) if str(x).strip()]

    respfiles: List[str] = []
    debug_sources: List[str] = []
    for p in _iter_xmm_srspec_files(cache_root, obsids=obsids if obsids else None):
        try:
            hdr = _fits_read_spectrum_header(p)
        except Exception:
            continue

        resp = str(hdr.get("RESPFILE") or "").strip().strip("'")
        # 条件分岐: `resp` を満たす経路を評価する。
        if resp:
            respfiles.append(resp)
            debug_sources.append(_rel(p))

    respfiles.extend(extra)

    uniq = []
    seen = set()
    for r in respfiles:
        # 条件分岐: `r in seen` を満たす経路を評価する。
        if r in seen:
            continue

        seen.add(r)
        uniq.append(r)

    run = {
        "generated_utc": _utc_now(),
        "inputs": {
            "cache_root": _rel(cache_root),
            "xmm_obsids": obsids,
            "respfile_extra": extra,
            "offline": bool(args.offline),
            "download_missing": bool(args.download_missing),
            "force": bool(args.force),
        },
        "sources": {"base": _BASE, "mos": _BASE_MOS, "pn": _BASE_PN},
        "detected": {"n_srspec_scanned": len(debug_sources), "respfiles": uniq[:]},
        "results": {"downloaded": [], "skipped": [], "errors": []},
    }

    for name in uniq:
        info = _resp_base_for_name(name)
        # 条件分岐: `info is None` を満たす経路を評価する。
        if info is None:
            run["results"]["skipped"].append({"status": "skipped_unknown_instrument", "respfile": name})
            continue

        inst, base_url = info
        url = urljoin(base_url, name)
        dst = out_dir / inst / name
        # 条件分岐: `args.offline` を満たす経路を評価する。
        if args.offline:
            # 条件分岐: `dst.exists()` を満たす経路を評価する。
            if dst.exists():
                run["results"]["skipped"].append(
                    {"status": "offline_exists", "respfile": name, "path": _rel(dst), "sha256": _sha256_file(dst)}
                )
            else:
                run["results"]["skipped"].append({"status": "offline_missing", "respfile": name, "path": _rel(dst)})

            continue

        # 条件分岐: `not args.download_missing and not dst.exists()` を満たす経路を評価する。

        if not args.download_missing and not dst.exists():
            run["results"]["skipped"].append({"status": "skipped_missing", "respfile": name, "path": _rel(dst), "url": url})
            continue

        try:
            res = _download(url, dst=dst, force=bool(args.force))
        except Exception as e:
            run["results"]["errors"].append({"status": "error", "respfile": name, "path": _rel(dst), "url": url, "error": str(e)})
            continue

        # 条件分岐: `res.get("status") == "downloaded"` を満たす経路を評価する。

        if res.get("status") == "downloaded":
            run["results"]["downloaded"].append({**res, "respfile": name})
        else:
            run["results"]["skipped"].append({**res, "respfile": name})

    prev = _read_json(out_manifest)
    manifest = dict(prev)
    history = manifest.get("history", [])
    # 条件分岐: `not isinstance(history, list)` を満たす経路を評価する。
    if not isinstance(history, list):
        history = []

    history.append(run)
    manifest["history"] = history
    manifest["latest"] = run
    manifest["generated_utc"] = run["generated_utc"]
    manifest["cache_dir"] = _rel(out_dir)
    _write_json(out_manifest, manifest)

    try:
        worklog.append_event(
            "xrism.fetch_xmm_epic_canned_responses",
            {
                "xmm_obsids": obsids,
                "n_respfiles": len(uniq),
                "downloaded": len(run["results"]["downloaded"]),
                "offline": bool(args.offline),
                "manifest": _rel(out_manifest),
            },
        )
    except Exception:
        pass

    print(json.dumps({"manifest": _rel(out_manifest), "downloaded": len(run["results"]["downloaded"])}, ensure_ascii=False))
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())

