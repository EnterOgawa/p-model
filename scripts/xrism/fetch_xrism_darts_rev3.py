#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_xrism_darts_rev3.py

Phase 4 / Step 4.8（XRISM）:
DARTS rev3 の公開一次データ（FITS）を obsid 単位で取得・キャッシュし、manifest（sha256）を固定する。

目的:
- HEASARC（products中心）に加えて、DARTS rev3（event/auxil/log を含む）を一次ソースとして扱える入口を作る。
- 将来の HEASoft/xrism-pipeline による event→スペクトル抽出（Pixel除外/GTI適用等）へ接続できる形で、
  まずは「データ取得と整合性確認」を再現可能に固定する。

出力（固定）:
- raw cache: data/xrism/raw/<obsid>/**（rev3 の <obsid>/ 以下を mirror）
- manifest: data/xrism/sources/<obsid>/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urljoin, urlparse

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency for offline/local runs
    requests = None

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402

_REQ_TIMEOUT = (30, 600)  # (connect, read)

DEFAULT_BASE_URL = "https://data.darts.isas.jaxa.jp/pub/xrism/data/obs/rev3/"

_MANIFEST_NAME = "manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def _http_get_text(url: str) -> str:
    # 条件分岐: `requests is None` を満たす経路を評価する。
    if requests is None:
        raise RuntimeError("requests is required for online fetch")

    r = requests.get(url, timeout=_REQ_TIMEOUT)
    r.raise_for_status()
    r.encoding = r.encoding or "utf-8"
    return r.text


def _http_get_stream(url: str) -> requests.Response:  # type: ignore[name-defined]
    # 条件分岐: `requests is None` を満たす経路を評価する。
    if requests is None:
        raise RuntimeError("requests is required for online fetch")

    r = requests.get(url, timeout=_REQ_TIMEOUT, stream=True)
    r.raise_for_status()
    return r


_HREF_RE = re.compile(r'href=\"(?P<href>[^\"]+)\"', flags=re.IGNORECASE)


def _list_dir(url: str) -> List[Tuple[str, bool]]:
    """
    Return list of (href, is_dir) under an Apache index directory.
    """
    html = _http_get_text(url)
    out: List[Tuple[str, bool]] = []
    for m in _HREF_RE.finditer(html):
        href = str(m.group("href") or "")
        # 条件分岐: `not href or href.startswith("?")` を満たす経路を評価する。
        if not href or href.startswith("?"):
            continue

        # 条件分岐: `href in {"../", "./"}` を満たす経路を評価する。

        if href in {"../", "./"}:
            continue
        # Skip links that clearly jump out (parent listing absolute path)

        if href.startswith("/pub/") and not href.endswith("/"):
            # still a file; keep
            pass

        is_dir = href.endswith("/")
        out.append((href, is_dir))
    # Dedup while keeping order.

    seen: Set[Tuple[str, bool]] = set()
    uniq: List[Tuple[str, bool]] = []
    for it in out:
        # 条件分岐: `it in seen` を満たす経路を評価する。
        if it in seen:
            continue

        seen.add(it)
        uniq.append(it)

    return uniq


def _obsid_type(obsid: str) -> str:
    s = str(obsid).strip()
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        raise ValueError("empty obsid")

    # 条件分岐: `not re.match(r"^[0-9]{9}$", s)` を満たす経路を評価する。

    if not re.match(r"^[0-9]{9}$", s):
        raise ValueError(f"invalid obsid: {obsid}")

    return s[0]


def _safe_rel_under_obsid(url: str, *, obsid: str) -> Optional[PurePosixPath]:
    """
    Map remote URL to a relative posix path under <obsid>/.
    Example:
      .../rev3/3/300019010/resolve/products/foo.pi.gz  ->  resolve/products/foo.pi.gz
    """
    p = urlparse(url).path
    key = f"/{obsid}/"
    # 条件分岐: `key not in p` を満たす経路を評価する。
    if key not in p:
        return None

    rel = p.split(key, 1)[1]
    rel = rel.lstrip("/")
    # 条件分岐: `not rel` を満たす経路を評価する。
    if not rel:
        return None
    # Normalize to PurePosixPath for stability.

    from pathlib import PurePosixPath

    return PurePosixPath(rel)


def _matches_any(patterns: Sequence[re.Pattern[str]], s: str) -> bool:
    # 条件分岐: `not patterns` を満たす経路を評価する。
    if not patterns:
        return True

    return any(p.search(s) for p in patterns)


def _download_file(url: str, *, dst: Path, force: bool) -> Dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `dst.exists() and not force` を満たす経路を評価する。
    if dst.exists() and not force:
        return {"status": "skipped_exists", "path": _rel(dst), "bytes": int(dst.stat().st_size), "sha256": _sha256_file(dst)}

    r = _http_get_stream(url)
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

    return {"status": "downloaded", "path": _rel(dst), "bytes": int(n), "sha256": h.hexdigest()}


def _scope_prefixes(scopes: Sequence[str]) -> List[str]:
    pref: List[str] = []
    for s in scopes:
        s0 = s.strip().lower()
        # 条件分岐: `s0 == "products"` を満たす経路を評価する。
        if s0 == "products":
            pref.append("resolve/products/")
        # 条件分岐: 前段条件が不成立で、`s0 in {"event", "event_cl"}` を追加評価する。
        elif s0 in {"event", "event_cl"}:
            pref.append("resolve/event_cl/")
        # 条件分岐: 前段条件が不成立で、`s0 == "event_uf"` を追加評価する。
        elif s0 == "event_uf":
            pref.append("resolve/event_uf/")
        # 条件分岐: 前段条件が不成立で、`s0 == "hk"` を追加評価する。
        elif s0 == "hk":
            pref.append("resolve/hk/")
        # 条件分岐: 前段条件が不成立で、`s0 == "auxil"` を追加評価する。
        elif s0 == "auxil":
            pref.append("auxil/")
        # 条件分岐: 前段条件が不成立で、`s0 == "log"` を追加評価する。
        elif s0 == "log":
            pref.append("log/")
        # 条件分岐: 前段条件が不成立で、`s0 == "resolve_all"` を追加評価する。
        elif s0 == "resolve_all":
            pref.append("resolve/")
        # 条件分岐: 前段条件が不成立で、`s0 == "all"` を追加評価する。
        elif s0 == "all":
            pref.extend(["resolve/", "auxil/", "log/"])
        else:
            raise ValueError(f"unknown download scope: {s}")
    # Dedup while keeping order.

    out: List[str] = []
    for p in pref:
        # 条件分岐: `p not in out` を満たす経路を評価する。
        if p not in out:
            out.append(p)

    return out


def _traverse_files(base_url: str, *, obsid: str, allow_prefixes: Sequence[str]) -> List[str]:
    """
    Traverse under <base_url> (<...>/<obsid>/) and return file URLs for allowed prefixes.
    """
    queue: List[str] = [base_url]
    files: List[str] = []
    seen_dirs: Set[str] = set()

    while queue:
        url = queue.pop(0)
        # 条件分岐: `url in seen_dirs` を満たす経路を評価する。
        if url in seen_dirs:
            continue

        seen_dirs.add(url)

        for href, is_dir in _list_dir(url):
            full = urljoin(url, href)
            rel = _safe_rel_under_obsid(full, obsid=obsid)
            # 条件分岐: `rel is None` を満たす経路を評価する。
            if rel is None:
                continue

            rel_s = rel.as_posix()
            # 条件分岐: `is_dir` を満たす経路を評価する。
            if is_dir:
                # Traverse if this directory is an ancestor of any allowed prefix.
                if any(p.startswith(rel_s) for p in allow_prefixes):
                    # 条件分岐: `not full.endswith("/")` を満たす経路を評価する。
                    if not full.endswith("/"):
                        full = full + "/"

                    queue.append(full)
            else:
                # 条件分岐: `any(rel_s.startswith(p) for p in allow_prefixes)` を満たす経路を評価する。
                if any(rel_s.startswith(p) for p in allow_prefixes):
                    files.append(full)

    files = sorted(set(files))
    return files


def _read_target_catalog_obsids(path: Path) -> List[str]:
    try:
        j = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    targets = j.get("targets")
    # 条件分岐: `not isinstance(targets, list)` を満たす経路を評価する。
    if not isinstance(targets, list):
        return []

    obsids: List[str] = []
    for t in targets:
        # 条件分岐: `not isinstance(t, dict)` を満たす経路を評価する。
        if not isinstance(t, dict):
            continue

        o = str(t.get("obsid") or "").strip()
        # 条件分岐: `re.match(r"^[0-9]{9}$", o)` を満たす経路を評価する。
        if re.match(r"^[0-9]{9}$", o):
            obsids.append(o)

    return sorted(set(obsids))


def fetch_one(
    *,
    obsid: str,
    base_url: str,
    scopes: Sequence[str],
    include_regex: Sequence[str],
    exclude_regex: Sequence[str],
    out_raw_root: Path,
    out_manifest_dir: Path,
    force: bool,
) -> Dict[str, Any]:
    type_dir = _obsid_type(obsid)
    obs_base_url = urljoin(base_url.rstrip("/") + "/", f"{type_dir}/{obsid}/")
    allow_prefixes = _scope_prefixes(scopes)
    include_pats = [re.compile(p) for p in include_regex]
    exclude_pats = [re.compile(p) for p in exclude_regex]

    file_urls = _traverse_files(obs_base_url, obsid=obsid, allow_prefixes=allow_prefixes)

    downloaded: List[Dict[str, Any]] = []
    encrypted: List[str] = []
    skipped: List[str] = []

    local_obs_root = out_raw_root / obsid
    local_obs_root.mkdir(parents=True, exist_ok=True)

    for url in file_urls:
        rel = _safe_rel_under_obsid(url, obsid=obsid)
        # 条件分岐: `rel is None` を満たす経路を評価する。
        if rel is None:
            continue

        rel_s = rel.as_posix()
        # 条件分岐: `any(p.search(rel_s) for p in exclude_pats)` を満たす経路を評価する。
        if any(p.search(rel_s) for p in exclude_pats):
            skipped.append(rel_s)
            continue

        # 条件分岐: `not _matches_any(include_pats, rel_s)` を満たす経路を評価する。

        if not _matches_any(include_pats, rel_s):
            skipped.append(rel_s)
            continue

        # 条件分岐: `rel_s.endswith(".gpg")` を満たす経路を評価する。

        if rel_s.endswith(".gpg"):
            encrypted.append(rel_s)
            continue

        dst = local_obs_root / Path(rel_s)
        rec = _download_file(url, dst=dst, force=force)
        rec["fetched_utc"] = _utc_now()
        rec["url"] = url
        rec["rel_under_obsid"] = rel_s
        downloaded.append(rec)

    # Basic integrity checks (presence).

    checks = {
        "has_resolve_event_cl": (local_obs_root / "resolve" / "event_cl").is_dir(),
        "has_resolve_products": (local_obs_root / "resolve" / "products").is_dir(),
        "has_auxil": (local_obs_root / "auxil").is_dir(),
        "has_log": (local_obs_root / "log").is_dir(),
    }

    session = {
        "generated_utc": _utc_now(),
        "download": {
            "scopes": list(scopes),
            "allow_prefixes": list(allow_prefixes),
            "include_regex": list(include_regex),
            "exclude_regex": list(exclude_regex),
            "force": bool(force),
        },
        "result": {
            "n_file_urls": len(file_urls),
            "n_downloaded": len(downloaded),
            "n_encrypted": len(encrypted),
            "n_skipped": len(skipped),
        },
        "new_downloaded_relpaths": sorted({str(d.get("rel_under_obsid") or "") for d in downloaded if str(d.get("rel_under_obsid") or "")}),
    }

    out_path = out_manifest_dir / obsid / _MANIFEST_NAME
    prev = _read_json(out_path) if out_path.exists() else {}

    # Merge downloaded records by rel_under_obsid.
    merged_by_rel: Dict[str, Dict[str, Any]] = {}
    for it in (prev.get("downloaded") or []):
        # 条件分岐: `not isinstance(it, dict)` を満たす経路を評価する。
        if not isinstance(it, dict):
            continue

        k = str(it.get("rel_under_obsid") or "").strip()
        # 条件分岐: `k` を満たす経路を評価する。
        if k:
            merged_by_rel[k] = dict(it)

    for it in downloaded:
        k = str(it.get("rel_under_obsid") or "").strip()
        # 条件分岐: `k` を満たす経路を評価する。
        if k:
            merged_by_rel[k] = dict(it)

    # Merge encrypted relpaths.

    enc: Set[str] = set()
    for x in (prev.get("encrypted") or []):
        # 条件分岐: `isinstance(x, str) and x.strip()` を満たす経路を評価する。
        if isinstance(x, str) and x.strip():
            enc.add(x.strip())

    for x in encrypted:
        # 条件分岐: `x.strip()` を満たす経路を評価する。
        if x.strip():
            enc.add(x.strip())

    sessions: List[Dict[str, Any]] = []
    for s in (prev.get("sessions") or []):
        # 条件分岐: `isinstance(s, dict)` を満たす経路を評価する。
        if isinstance(s, dict):
            sessions.append(s)

    sessions.append(session)

    manifest = {
        "generated_utc": _utc_now(),
        "obsid": obsid,
        "source": {"base_url": base_url, "obs_base_url": obs_base_url},
        "outputs": {
            "raw_root": _rel(local_obs_root),
            "manifest_path": _rel(out_path),
        },
        "checks": checks,
        "summary": {
            "n_files_total": len(merged_by_rel),
            "n_encrypted_total": len(enc),
            "n_sessions": len(sessions),
        },
        "downloaded": [merged_by_rel[k] for k in sorted(merged_by_rel.keys())],
        "encrypted": sorted(enc),
        "sessions": sessions[-50:],  # keep tail only (stability)
        # Keep the last run's skipped list as a diagnostic (avoid huge file).
        "skipped": skipped[:2000],
    }

    _write_json(out_path, manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--catalog", default=str(_ROOT / "data" / "xrism" / "sources" / "xrism_target_catalog.json"))
    p.add_argument("--obsid", action="append", default=[], help="override: obsid(s) to fetch (repeatable)")
    p.add_argument("--scope", action="append", default=[], help="download scope(s): products/event_cl/auxil/log/all")
    p.add_argument("--include-regex", action="append", default=[], help="download only paths matching regex (repeatable)")
    p.add_argument("--exclude-regex", action="append", default=[], help="skip paths matching regex (repeatable)")
    p.add_argument("--raw-root", default=str(_ROOT / "data" / "xrism" / "raw"))
    p.add_argument("--manifest-root", default=str(_ROOT / "data" / "xrism" / "sources"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    raw_root = Path(args.raw_root)
    manifest_root = Path(args.manifest_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    scopes = [str(s).strip() for s in (args.scope or []) if str(s).strip()]
    # 条件分岐: `not scopes` を満たす経路を評価する。
    if not scopes:
        scopes = ["products"]
    # Dedup while preserving order.

    scopes_uniq: List[str] = []
    for s in scopes:
        # 条件分岐: `s not in scopes_uniq` を満たす経路を評価する。
        if s not in scopes_uniq:
            scopes_uniq.append(s)

    obsids = [str(x).strip() for x in (args.obsid or []) if str(x).strip()]
    # 条件分岐: `not obsids` を満たす経路を評価する。
    if not obsids:
        obsids = _read_target_catalog_obsids(Path(args.catalog))

    # 条件分岐: `not obsids` を満たす経路を評価する。

    if not obsids:
        print("[warn] no obsids")
        return 0

    manifests: List[Dict[str, Any]] = []
    for obsid in obsids:
        print(f"[info] fetch obsid={obsid} scopes={scopes_uniq}")
        man = fetch_one(
            obsid=obsid,
            base_url=str(args.base_url),
            scopes=scopes_uniq,
            include_regex=list(args.include_regex),
            exclude_regex=list(args.exclude_regex),
            out_raw_root=raw_root,
            out_manifest_dir=manifest_root,
            force=bool(args.force),
        )
        manifests.append(man)

    worklog.append_event(
        {
            "task": "xrism_darts_rev3_fetch",
            "inputs": {"catalog": Path(args.catalog), "base_url": str(args.base_url), "obsids": obsids},
            "outputs": {
                "raw_root": raw_root,
                "manifest_paths": [Path(m["outputs"]["manifest_path"]) for m in manifests if isinstance(m, dict)],
            },
            "metrics": {
                "n_obsids": len(obsids),
                "n_downloaded_total": int(sum(int(m.get("result", {}).get("n_downloaded", 0)) for m in manifests)),
                "n_encrypted_total": int(sum(int(m.get("result", {}).get("n_encrypted", 0)) for m in manifests)),
            },
        }
    )

    print(f"[ok] fetched obsids={len(obsids)} raw_root={raw_root}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
