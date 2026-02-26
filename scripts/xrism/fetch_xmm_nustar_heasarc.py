#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_xmm_nustar_heasarc.py

Phase 4 / Step 4.8.7（Fe-Kα 相対論的 broad line / ISCO）:
HEASARC FTP over HTTPS から XMM-Newton / NuSTAR の公開一次データ（主に products/event）
を obsid 単位で取得・キャッシュし、sha256 manifest を固定する。

目的（最小）:
- XMM（PPS）/ NuSTAR（event_cl 等）の取得I/Fを閉じ、offline 再現できる土台を固定する。
- スペクトル抽出・反射モデルfit（ISCO制約）は別スクリプトで実装する前提で、
  まず「取得条件・ファイル一覧・sha256」を再現可能に固定する。

出力（固定）:
- raw cache: data/xrism/xmm_nustar/{xmm|nustar}/...（取得したファイルを保存）
- manifest: data/xrism/sources/xmm_nustar_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
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

_XMM_BASE_URL = "https://heasarc.gsfc.nasa.gov/FTP/xmm/data"
_NUSTAR_BASE_URL = "https://heasarc.gsfc.nasa.gov/FTP/nustar/data/obs"


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


def _http_head_content_length(url: str) -> Optional[int]:
    # 条件分岐: `requests is None` を満たす経路を評価する。
    if requests is None:
        return None

    try:
        r = requests.head(url, timeout=_REQ_TIMEOUT, allow_redirects=True)
        r.raise_for_status()
    except Exception:
        return None

    try:
        n = int(r.headers.get("Content-Length") or "0")
    except Exception:
        return None

    return n if n > 0 else None


_HREF_RE = re.compile(r'href=\"(?P<href>[^\"]+)\"', flags=re.IGNORECASE)


def _list_apache_dir(url: str) -> List[Tuple[str, bool]]:
    # 条件分岐: `not url.endswith("/")` を満たす経路を評価する。
    if not url.endswith("/"):
        url += "/"

    html = _http_get_text(url)
    out: List[Tuple[str, bool]] = []
    for m in _HREF_RE.finditer(html):
        href = str(m.group("href") or "").strip()
        # 条件分岐: `not href or href.startswith("?") or href in {"../", "./"}` を満たす経路を評価する。
        if not href or href.startswith("?") or href in {"../", "./"}:
            continue

        # 条件分岐: `href.startswith("/")` を満たす経路を評価する。

        if href.startswith("/"):
            continue

        # 条件分岐: `href.lower().startswith(("http://", "https://"))` を満たす経路を評価する。

        if href.lower().startswith(("http://", "https://")):
            continue

        href = href.split("#", 1)[0].split("?", 1)[0]
        out.append((href, href.endswith("/")))

    seen: set[Tuple[str, bool]] = set()
    uniq: List[Tuple[str, bool]] = []
    for it in out:
        # 条件分岐: `it in seen` を満たす経路を評価する。
        if it in seen:
            continue

        seen.add(it)
        uniq.append(it)

    return uniq


def _download(url: str, *, dst: Path, force: bool, max_file_bytes: int) -> Dict[str, Any]:
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

    # 条件分岐: `max_file_bytes > 0` を満たす経路を評価する。

    if max_file_bytes > 0:
        remote_size = _http_head_content_length(url)
        # 条件分岐: `remote_size is not None and remote_size > max_file_bytes` を満たす経路を評価する。
        if remote_size is not None and remote_size > max_file_bytes:
            return {
                "status": "skipped_too_large",
                "path": _rel(dst),
                "bytes_remote": int(remote_size),
                "max_file_bytes": int(max_file_bytes),
                "url": url,
            }

    r = _http_get_stream(url)
    # 条件分岐: `max_file_bytes > 0` を満たす経路を評価する。
    if max_file_bytes > 0:
        try:
            remote_size = int(r.headers.get("Content-Length") or "0")
        except Exception:
            remote_size = 0

        # 条件分岐: `remote_size and remote_size > max_file_bytes` を満たす経路を評価する。

        if remote_size and remote_size > max_file_bytes:
            try:
                r.close()
            except Exception:
                pass

            return {
                "status": "skipped_too_large",
                "path": _rel(dst),
                "bytes_remote": int(remote_size),
                "max_file_bytes": int(max_file_bytes),
                "url": url,
            }

    h = hashlib.sha256()
    n = 0
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        # 条件分岐: `tmp.exists()` を満たす経路を評価する。
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass

    with tmp.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            # 条件分岐: `not chunk` を満たす経路を評価する。
            if not chunk:
                continue

            f.write(chunk)
            h.update(chunk)
            n += len(chunk)
            # 条件分岐: `max_file_bytes > 0 and n > max_file_bytes` を満たす経路を評価する。
            if max_file_bytes > 0 and n > max_file_bytes:
                try:
                    r.close()
                except Exception:
                    pass

                try:
                    tmp.unlink()
                except Exception:
                    pass

                return {
                    "status": "skipped_too_large",
                    "path": _rel(dst),
                    "bytes_partial": int(n),
                    "max_file_bytes": int(max_file_bytes),
                    "url": url,
                }

    tmp.replace(dst)
    return {"status": "downloaded", "path": _rel(dst), "bytes": int(n), "sha256": h.hexdigest(), "url": url}


def _compile_patterns(patterns: Sequence[str]) -> List[re.Pattern[str]]:
    out: List[re.Pattern[str]] = []
    for s in patterns:
        s0 = (s or "").strip()
        # 条件分岐: `not s0` を満たす経路を評価する。
        if not s0:
            continue

        out.append(re.compile(s0))

    return out


def _matches_any(patterns: Sequence[re.Pattern[str]], s: str) -> bool:
    # 条件分岐: `not patterns` を満たす経路を評価する。
    if not patterns:
        return True

    return any(p.search(s) for p in patterns)


def _matches_any_exclude(patterns: Sequence[re.Pattern[str]], s: str) -> bool:
    return any(p.search(s) for p in patterns)


def _obsid_xmm(s: str) -> str:
    s0 = str(s).strip()
    # 条件分岐: `not re.match(r"^[0-9]{10}$", s0)` を満たす経路を評価する。
    if not re.match(r"^[0-9]{10}$", s0):
        raise ValueError(f"invalid XMM obsid: {s}")

    return s0


def _obsid_nustar(s: str) -> str:
    s0 = str(s).strip()
    # 条件分岐: `not re.match(r"^[0-9]{11}$", s0)` を満たす経路を評価する。
    if not re.match(r"^[0-9]{11}$", s0):
        raise ValueError(f"invalid NuSTAR obsid: {s}")

    return s0


def _nustar_remote_root(obsid: str) -> str:
    s = _obsid_nustar(obsid)
    sub = s[1:3]
    lead = s[0]
    return f"{_NUSTAR_BASE_URL}/{sub}/{lead}/{s}/"


def _safe_rel(url: str, *, anchor: str) -> Optional[str]:
    p = urlparse(url).path
    key = f"/{anchor}/"
    # 条件分岐: `key not in p` を満たす経路を評価する。
    if key not in p:
        return None

    rel = p.split(key, 1)[1].lstrip("/")
    # 条件分岐: `not rel` を満たす経路を評価する。
    if not rel:
        return None

    return rel


@dataclass(frozen=True)
class FetchItem:
    mission: str
    obsid: str
    rev: Optional[str]
    scope: str
    rel: str
    url: str


def _plan_xmm(
    obsid: str,
    *,
    rev: str,
    scopes: Sequence[str],
    include: Sequence[re.Pattern[str]],
    exclude: Sequence[re.Pattern[str]],
) -> List[FetchItem]:
    obsid0 = _obsid_xmm(obsid)
    rev0 = (rev or "rev0").strip()
    root = f"{_XMM_BASE_URL}/{rev0}/{obsid0}/"
    items: List[FetchItem] = []
    for scope in scopes:
        s0 = (scope or "").strip().lower()
        # 条件分岐: `s0 in {"pps", "products"}` を満たす経路を評価する。
        if s0 in {"pps", "products"}:
            dir_url = urljoin(root, "PPS/")
            for href, is_dir in _list_apache_dir(dir_url):
                # 条件分岐: `is_dir` を満たす経路を評価する。
                if is_dir:
                    continue

                rel = f"PPS/{href}"
                # 条件分岐: `not _matches_any(include, rel)` を満たす経路を評価する。
                if not _matches_any(include, rel):
                    continue

                # 条件分岐: `_matches_any_exclude(exclude, rel)` を満たす経路を評価する。

                if _matches_any_exclude(exclude, rel):
                    continue

                items.append(FetchItem("xmm", obsid0, rev0, "PPS", rel, urljoin(dir_url, href)))
        # 条件分岐: 前段条件が不成立で、`s0 == "odf"` を追加評価する。
        elif s0 == "odf":
            dir_url = urljoin(root, "ODF/")
            for href, is_dir in _list_apache_dir(dir_url):
                # 条件分岐: `is_dir` を満たす経路を評価する。
                if is_dir:
                    continue

                rel = f"ODF/{href}"
                # 条件分岐: `not _matches_any(include, rel)` を満たす経路を評価する。
                if not _matches_any(include, rel):
                    continue

                # 条件分岐: `_matches_any_exclude(exclude, rel)` を満たす経路を評価する。

                if _matches_any_exclude(exclude, rel):
                    continue

                items.append(FetchItem("xmm", obsid0, rev0, "ODF", rel, urljoin(dir_url, href)))
        else:
            raise ValueError(f"unknown XMM scope: {scope}")

    return items


def _plan_nustar(
    obsid: str,
    *,
    scopes: Sequence[str],
    include: Sequence[re.Pattern[str]],
    exclude: Sequence[re.Pattern[str]],
) -> List[FetchItem]:
    obsid0 = _obsid_nustar(obsid)
    root = _nustar_remote_root(obsid0)
    items: List[FetchItem] = []
    for scope in scopes:
        s0 = (scope or "").strip().lower()
        # 条件分岐: `s0 == "root"` を満たす経路を評価する。
        if s0 == "root":
            for href, is_dir in _list_apache_dir(root):
                # 条件分岐: `is_dir` を満たす経路を評価する。
                if is_dir:
                    continue

                rel = href
                # 条件分岐: `not _matches_any(include, rel)` を満たす経路を評価する。
                if not _matches_any(include, rel):
                    continue

                # 条件分岐: `_matches_any_exclude(exclude, rel)` を満たす経路を評価する。

                if _matches_any_exclude(exclude, rel):
                    continue

                items.append(FetchItem("nustar", obsid0, None, "root", rel, urljoin(root, href)))
        # 条件分岐: 前段条件が不成立で、`s0 in {"event_cl", "auxil", "hk", "event_uf"}` を追加評価する。
        elif s0 in {"event_cl", "auxil", "hk", "event_uf"}:
            dir_url = urljoin(root, f"{s0}/")
            for href, is_dir in _list_apache_dir(dir_url):
                # 条件分岐: `is_dir` を満たす経路を評価する。
                if is_dir:
                    continue

                rel = f"{s0}/{href}"
                # 条件分岐: `not _matches_any(include, rel)` を満たす経路を評価する。
                if not _matches_any(include, rel):
                    continue

                # 条件分岐: `_matches_any_exclude(exclude, rel)` を満たす経路を評価する。

                if _matches_any_exclude(exclude, rel):
                    continue

                items.append(FetchItem("nustar", obsid0, None, s0, rel, urljoin(dir_url, href)))
        else:
            raise ValueError(f"unknown NuSTAR scope: {scope}")

    return items


def _default_xmm_include_regex(obsid: str) -> List[str]:
    s = _obsid_xmm(obsid)
    return [
        rf"^PPS/PP{s}RSPECT.*\.HTM$",
        rf"^PPS/P{s}PNS.*SRSPEC.*\.FTZ$",
        rf"^PPS/P{s}PNS.*SRCARF.*\.FTZ$",
        rf"^PPS/P{s}PNS.*BGSPEC.*\.FTZ$",
        rf"^PPS/P{s}M1.*SRSPEC.*\.FTZ$",
        rf"^PPS/P{s}M1.*SRCARF.*\.FTZ$",
        rf"^PPS/P{s}M1.*BGSPEC.*\.FTZ$",
        rf"^PPS/P{s}M2.*SRSPEC.*\.FTZ$",
        rf"^PPS/P{s}M2.*SRCARF.*\.FTZ$",
        rf"^PPS/P{s}M2.*BGSPEC.*\.FTZ$",
        # Response matrices referenced by SRSPEC headers (RESPFILE=...RSPMAT....FTZ).
        rf"^PPS/P{s}R1.*RSPMAT.*\.FTZ$",
        rf"^PPS/P{s}R2.*RSPMAT.*\.FTZ$",
    ]


def _default_nustar_include_regex(obsid: str) -> List[str]:
    s = _obsid_nustar(obsid)
    return [
        rf"^nu{s}\.cat\.gz$",
        r"^pipe\.log$",
        rf"^event_cl/nu{s}[AB].*\.(fits|evt|gz)$",
        rf"^event_cl/nu{s}[AB].*\.reg$",
        rf"^auxil/nu{s}_.*\.(fits|gz)$",
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--xmm-obsid", action="append", default=[], help="XMM obsid (10 digits). repeatable.")
    p.add_argument("--nustar-obsid", action="append", default=[], help="NuSTAR obsid (11 digits). repeatable.")
    p.add_argument("--xmm-rev", default="rev0", help="XMM data revision under heasarc FTP (rev0 or rev1).")

    p.add_argument("--xmm-scope", action="append", default=["pps"], help="XMM scope: pps|odf. repeatable.")
    p.add_argument(
        "--nustar-scope",
        action="append",
        default=["root", "auxil", "event_cl"],
        help="NuSTAR scope: root|auxil|event_cl|event_uf|hk. repeatable.",
    )

    p.add_argument("--xmm-include-regex", action="append", default=[], help="Regex filter for XMM relpath (e.g. ^PPS/...)")
    p.add_argument(
        "--nustar-include-regex", action="append", default=[], help="Regex filter for NuSTAR relpath (e.g. ^event_cl/...)"
    )
    p.add_argument("--xmm-exclude-regex", action="append", default=[], help="Regex exclude filter for XMM relpath.")
    p.add_argument("--nustar-exclude-regex", action="append", default=[], help="Regex exclude filter for NuSTAR relpath.")

    p.add_argument("--download-missing", action="store_true", help="Download missing files (online).")
    p.add_argument("--force", action="store_true", help="Force re-download even if local file exists.")
    p.add_argument("--offline", action="store_true", help="Offline mode: do not access the network.")
    p.add_argument("--max-files", type=int, default=0, help="Safety: max number of files to download (0=unlimited).")
    p.add_argument(
        "--max-file-mib",
        type=int,
        default=0,
        help="Safety: skip a single file larger than this size (0=unlimited).",
    )

    args = p.parse_args(list(argv) if argv is not None else None)

    xmm_obsids = [_obsid_xmm(x) for x in args.xmm_obsid]
    nustar_obsids = [_obsid_nustar(x) for x in args.nustar_obsid]
    # 条件分岐: `not xmm_obsids and not nustar_obsids` を満たす経路を評価する。
    if not xmm_obsids and not nustar_obsids:
        p.error("at least one --xmm-obsid or --nustar-obsid is required")

    out_manifest = _ROOT / "data/xrism/sources/xmm_nustar_manifest.json"
    prev = _read_json(out_manifest)

    # 条件分岐: `args.offline and args.download_missing` を満たす経路を評価する。
    if args.offline and args.download_missing:
        p.error("--offline and --download-missing cannot be used together")

    # 条件分岐: `args.offline and requests is None` を満たす経路を評価する。

    if args.offline and requests is None:
        pass

    # 条件分岐: `not args.offline and requests is None` を満たす経路を評価する。

    if not args.offline and requests is None:
        raise RuntimeError("requests is required for online fetch")

    cache_root = _ROOT / "data/xrism/xmm_nustar"
    cache_root.mkdir(parents=True, exist_ok=True)
    max_file_bytes = int(args.max_file_mib) * 1024 * 1024 if int(args.max_file_mib) > 0 else 0

    run: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "inputs": {
            "xmm_obsids": xmm_obsids,
            "nustar_obsids": nustar_obsids,
            "xmm_rev": args.xmm_rev,
            "xmm_scope": args.xmm_scope,
            "nustar_scope": args.nustar_scope,
            "xmm_include_regex": args.xmm_include_regex,
            "nustar_include_regex": args.nustar_include_regex,
            "xmm_exclude_regex": args.xmm_exclude_regex,
            "nustar_exclude_regex": args.nustar_exclude_regex,
            "offline": bool(args.offline),
            "download_missing": bool(args.download_missing),
            "force": bool(args.force),
            "max_files": int(args.max_files),
            "max_file_mib": int(args.max_file_mib),
        },
        "sources": {
            "xmm_base_url": _XMM_BASE_URL,
            "nustar_base_url": _NUSTAR_BASE_URL,
        },
        "results": {"downloaded": [], "skipped": [], "errors": []},
    }

    downloaded = 0

    # 条件分岐: `not args.offline` を満たす経路を評価する。
    if not args.offline:
        for obsid in xmm_obsids:
            include = _compile_patterns(args.xmm_include_regex or _default_xmm_include_regex(obsid))
            exclude = _compile_patterns(args.xmm_exclude_regex or [])
            planned = _plan_xmm(obsid, rev=args.xmm_rev, scopes=args.xmm_scope, include=include, exclude=exclude)
            for it in planned:
                local = cache_root / "xmm" / str(it.rev) / it.obsid / it.rel
                # 条件分岐: `not args.download_missing and not local.exists()` を満たす経路を評価する。
                if not args.download_missing and not local.exists():
                    run["results"]["skipped"].append({"status": "skipped_missing", "path": _rel(local), "url": it.url})
                    continue

                try:
                    res = _download(it.url, dst=local, force=bool(args.force), max_file_bytes=max_file_bytes)
                except Exception as e:
                    run["results"]["errors"].append({"status": "error", "path": _rel(local), "url": it.url, "error": str(e)})
                    continue

                # 条件分岐: `res["status"] == "downloaded"` を満たす経路を評価する。

                if res["status"] == "downloaded":
                    downloaded += 1
                    run["results"]["downloaded"].append(res)
                else:
                    run["results"]["skipped"].append(res)

                # 条件分岐: `args.max_files and downloaded >= int(args.max_files)` を満たす経路を評価する。

                if args.max_files and downloaded >= int(args.max_files):
                    break

            # 条件分岐: `args.max_files and downloaded >= int(args.max_files)` を満たす経路を評価する。

            if args.max_files and downloaded >= int(args.max_files):
                break

        # 条件分岐: `not (args.max_files and downloaded >= int(args.max_files))` を満たす経路を評価する。

        if not (args.max_files and downloaded >= int(args.max_files)):
            for obsid in nustar_obsids:
                include = _compile_patterns(args.nustar_include_regex or _default_nustar_include_regex(obsid))
                exclude = _compile_patterns(args.nustar_exclude_regex or [])
                planned = _plan_nustar(obsid, scopes=args.nustar_scope, include=include, exclude=exclude)
                for it in planned:
                    local = cache_root / "nustar" / it.obsid / it.rel
                    # 条件分岐: `not args.download_missing and not local.exists()` を満たす経路を評価する。
                    if not args.download_missing and not local.exists():
                        run["results"]["skipped"].append({"status": "skipped_missing", "path": _rel(local), "url": it.url})
                        continue

                    try:
                        res = _download(it.url, dst=local, force=bool(args.force), max_file_bytes=max_file_bytes)
                    except Exception as e:
                        run["results"]["errors"].append({"status": "error", "path": _rel(local), "url": it.url, "error": str(e)})
                        continue

                    # 条件分岐: `res["status"] == "downloaded"` を満たす経路を評価する。

                    if res["status"] == "downloaded":
                        downloaded += 1
                        run["results"]["downloaded"].append(res)
                    else:
                        run["results"]["skipped"].append(res)

                    # 条件分岐: `args.max_files and downloaded >= int(args.max_files)` を満たす経路を評価する。

                    if args.max_files and downloaded >= int(args.max_files):
                        break

                # 条件分岐: `args.max_files and downloaded >= int(args.max_files)` を満たす経路を評価する。

                if args.max_files and downloaded >= int(args.max_files):
                    break

    else:
        run["results"]["skipped"].append({"status": "offline", "note": "offline mode: no network access"})

    manifest = dict(prev)
    history = manifest.get("history", [])
    # 条件分岐: `not isinstance(history, list)` を満たす経路を評価する。
    if not isinstance(history, list):
        history = []

    history.append(run)
    manifest["history"] = history
    manifest["latest"] = run
    manifest["cache_dir"] = _rel(cache_root)
    manifest["generated_utc"] = run["generated_utc"]

    _write_json(out_manifest, manifest)

    try:
        worklog.append_event(
            "xrism.fetch_xmm_nustar_heasarc",
            {
                "xmm_obsids": xmm_obsids,
                "nustar_obsids": nustar_obsids,
                "downloaded": len(run["results"]["downloaded"]),
                "skipped": len(run["results"]["skipped"]),
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
