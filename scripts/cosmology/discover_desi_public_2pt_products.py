#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
discover_desi_public_2pt_products.py

Phase 4（宇宙論）/ Step 4.5B.21.4.4.1.2（DESI: 公開 multipoles/cov 探索）の補助:
DESI 公開サーバ上で「2pt測定（ξ multipoles / P(k) multipoles）」や「共分散」を配布している場所を、
ディレクトリ listing をクロールして探索する。

注意:
- このスクリプトは **巨大ファイルをダウンロードしない**（HTML listing の取得のみ）。
- 出力は「候補URLの一覧」を固定し、後続で必要なものだけ fetch する。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree as ET

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


_DEFAULT_ROOTS = [
    # DR1 (catalogs / VACs / etc.)
    "https://data.desi.lbl.gov/public/dr1/",
    # Papers (Y1 key papers; large data may live under /public/papers/)
    "https://data.desi.lbl.gov/public/papers/",
    "https://data.desi.lbl.gov/public/papers/y1/",
    # DR2 (3-year) key paper bundles (public, but not a full DR2 catalog release)
    "https://data.desi.lbl.gov/public/papers/y3/",
    # EDR (just in case)
    "https://data.desi.lbl.gov/public/edr/",
    # DR1 mirror (WebDAV)
    "https://webdav-hdfs.pic.es/data/public/DESI/DR1/",
]

# We only flag candidates by filename. ("xi" is broad but we also require an extension of interest.)
_CANDIDATE_NAME_RE = re.compile(
    # NOTE:
    # - "xi/pk" alone is too narrow (DESI official likelihood packaging often uses "likelihood_*.h5").
    # - We keep this broad because we do not download large files here; we only list candidate URLs.
    r"(cov|covar|covariance|covmat|covmatrix|matrix|datavector|data[_-]?vector|2pt|2pcf|corrfunc|correlation|"
    r"multipole|poles|spectrum|pk|p_?k|power|bao|recon|shapefit|fs|likelihood|xi)",
    re.IGNORECASE,
)

_EXT_ALLOW = {
    ".txt",
    ".dat",
    ".csv",
    ".tsv",
    ".fits",
    ".fit",
    ".fits.gz",
    ".npz",
    ".npy",
    ".hdf5",
    ".h5",
    ".asdf",
    ".pkl",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".zip",
    ".json",
    ".yaml",
    ".yml",
}

# Avoid exploding crawls (random catalogs etc.). We skip these directory-path substrings.
_SKIP_DIR_SUBSTRS = [
    "/random",
    "/rancomb_",
    "/datcomb_",
    "/mocks/",
    "/ezmock/",
    "/abacussummit/",
    "/spectro/",
    "/target/",
    "/fiberassign/",
    "/ops/",
    "/GFA/",
]

_HREF_RE = re.compile(r'href="([^"]+)"', re.IGNORECASE)
_RETRY_STATUSES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Candidate:
    url: str
    filename: str
    source_dir: str


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_url(u: str) -> str:
    # Normalize: keep scheme+netloc+path, drop fragments.
    p = urlparse(u)
    path = p.path
    # 条件分岐: `path.endswith("//")` を満たす経路を評価する。
    if path.endswith("//"):
        path = path.rstrip("/") + "/"

    return f"{p.scheme}://{p.netloc}{path}"


def _is_dir_url(u: str) -> bool:
    return u.endswith("/")


def _should_skip_dir(u: str) -> bool:
    lu = u.lower()
    return any(s in lu for s in _SKIP_DIR_SUBSTRS)


def _is_candidate_file(name: str) -> bool:
    n = str(name).strip()
    # 条件分岐: `not n or n in ("../", "./")` を満たす経路を評価する。
    if not n or n in ("../", "./"):
        return False
    # Extension allow-list (handle ".fits.gz" etc).

    lower = n.lower()
    ok_ext = any(lower.endswith(ext) for ext in _EXT_ALLOW)
    # 条件分岐: `not ok_ext` を満たす経路を評価する。
    if not ok_ext:
        return False

    return bool(_CANDIDATE_NAME_RE.search(n))


def _fetch_text(url: str, *, timeout_sec: int) -> str:
    # 条件分岐: `requests is None` を満たす経路を評価する。
    if requests is None:
        raise RuntimeError("requests is required (pip install requests)")

    r = _request_with_retries("GET", url, timeout_sec=timeout_sec)
    return r.text


def _webdav_propfind(url: str, *, timeout_sec: int) -> str:
    # 条件分岐: `requests is None` を満たす経路を評価する。
    if requests is None:
        raise RuntimeError("requests is required (pip install requests)")
    # Minimal request body (some WebDAV servers require it).

    body = (
        '<?xml version="1.0" encoding="utf-8" ?>'
        '<D:propfind xmlns:D="DAV:">'
        "<D:prop><D:displayname/><D:resourcetype/></D:prop>"
        "</D:propfind>"
    )
    r = _request_with_retries(
        "PROPFIND",
        url,
        timeout_sec=timeout_sec,
        headers={"Depth": "1", "Content-Type": "application/xml"},
        data=body,
    )
    return r.text


def _session() -> "requests.Session":
    # Lazily create one session to reuse TCP/TLS connections (directory crawls are request-heavy).
    # NOTE: requests is Optional; type ignore for mypy isn't used here.
    if requests is None:  # pragma: no cover
        raise RuntimeError("requests is required (pip install requests)")

    s = getattr(_session, "_s", None)
    # 条件分岐: `s is None` を満たす経路を評価する。
    if s is None:
        s = requests.Session()
        setattr(_session, "_s", s)

    return s


def _request_with_retries(
    method: str,
    url: str,
    *,
    timeout_sec: int,
    retries: int = 4,
    retry_backoff_sec: float = 2.0,
    headers: Optional[Dict[str, str]] = None,
    data: Optional[str] = None,
) -> "requests.Response":
    """
    Request wrapper tolerant to transient gateway timeouts (common on data.desi.lbl.gov).
    """
    last_err: Optional[BaseException] = None
    for attempt in range(int(retries) + 1):
        try:
            r = _session().request(
                str(method).upper(),
                url,
                timeout=float(timeout_sec),
                headers=headers,
                data=data,
            )
            # 条件分岐: `r.status_code in _RETRY_STATUSES and attempt < int(retries)` を満たす経路を評価する。
            if r.status_code in _RETRY_STATUSES and attempt < int(retries):
                # Respect Retry-After if present, else exponential backoff.
                ra = r.headers.get("Retry-After")
                # 条件分岐: `ra` を満たす経路を評価する。
                if ra:
                    try:
                        wait_sec = float(ra)
                    except Exception:
                        wait_sec = float(retry_backoff_sec) * (2.0**attempt)
                else:
                    wait_sec = float(retry_backoff_sec) * (2.0**attempt)

                time.sleep(min(wait_sec, 60.0))
                continue

            r.raise_for_status()
            return r
        except BaseException as e:
            last_err = e
            # 条件分岐: `attempt >= int(retries)` を満たす経路を評価する。
            if attempt >= int(retries):
                break

            time.sleep(min(float(retry_backoff_sec) * (2.0**attempt), 60.0))

    assert last_err is not None
    raise last_err


def _iter_hrefs(html: str) -> Iterable[str]:
    # Directory listing is usually <pre><a href="...">; a regex is enough.
    for m in _HREF_RE.finditer(html):
        href = str(m.group(1)).strip()
        # 条件分岐: `not href` を満たす経路を評価する。
        if not href:
            continue

        yield href


def _iter_webdav_hrefs(xml_text: str) -> Iterable[str]:
    # WebDAV PROPFIND depth=1 returns <D:multistatus><D:response>...</D:response></...>.
    # We extract <D:href> and emit file/dir hrefs. Dir URLs must end with "/".
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return

    # Namespaces vary; match by localname.

    def _local(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    for resp in root.iter():
        # 条件分岐: `_local(resp.tag) != "response"` を満たす経路を評価する。
        if _local(resp.tag) != "response":
            continue

        href = None
        is_collection = False
        for child in resp:
            # 条件分岐: `_local(child.tag) == "href" and child.text` を満たす経路を評価する。
            if _local(child.tag) == "href" and child.text:
                href = child.text.strip()

            # 条件分岐: `_local(child.tag) == "propstat"` を満たす経路を評価する。

            if _local(child.tag) == "propstat":
                for pchild in child.iter():
                    # 条件分岐: `_local(pchild.tag) == "collection"` を満たす経路を評価する。
                    if _local(pchild.tag) == "collection":
                        is_collection = True

        # 条件分岐: `not href` を満たす経路を評価する。

        if not href:
            continue

        # 条件分岐: `is_collection and not href.endswith("/")` を満たす経路を評価する。

        if is_collection and not href.endswith("/"):
            href = href + "/"

        yield href


def _iter_listing_hrefs(
    url: str,
    *,
    timeout_sec: int,
    retries: int,
    retry_backoff_sec: float,
) -> Tuple[List[str], str]:
    """
    Return (hrefs, method) for a directory URL.
    - method is one of: "html", "webdav", "none"
    """
    # Try HTML-ish listing first.
    try:
        html = _request_with_retries(
            "GET",
            url,
            timeout_sec=timeout_sec,
            retries=retries,
            retry_backoff_sec=retry_backoff_sec,
        ).text
        hrefs = list(_iter_hrefs(html))
        # 条件分岐: `hrefs` を満たす経路を評価する。
        if hrefs:
            return hrefs, "html"
    except Exception:
        pass

    # Fallback: WebDAV PROPFIND depth=1.

    try:
        xml_text = _request_with_retries(
            "PROPFIND",
            url,
            timeout_sec=timeout_sec,
            retries=retries,
            retry_backoff_sec=retry_backoff_sec,
            headers={"Depth": "1", "Content-Type": "application/xml"},
            data=(
                '<?xml version="1.0" encoding="utf-8" ?>'
                '<D:propfind xmlns:D="DAV:">'
                "<D:prop><D:displayname/><D:resourcetype/></D:prop>"
                "</D:propfind>"
            ),
        ).text
        hrefs = list(_iter_webdav_hrefs(xml_text))
        # 条件分岐: `hrefs` を満たす経路を評価する。
        if hrefs:
            return hrefs, "webdav"
    except Exception:
        pass

    return [], "none"


def discover(
    roots: List[str],
    *,
    max_depth: int,
    timeout_sec: int,
    max_dirs: int,
    max_links_per_dir: int,
    retries: int,
    retry_backoff_sec: float,
) -> Tuple[List[Candidate], Dict[str, Any]]:
    q: deque[Tuple[str, int]] = deque()
    seen: Set[str] = set()
    visited_dirs: List[str] = []
    visited_dir_methods: Dict[str, str] = {}
    candidates: List[Candidate] = []

    for r in roots:
        u = _norm_url(str(r).strip())
        # 条件分岐: `not _is_dir_url(u)` を満たす経路を評価する。
        if not _is_dir_url(u):
            u = u.rstrip("/") + "/"

        q.append((u, 0))

    while q:
        # 条件分岐: `len(visited_dirs) >= int(max_dirs)` を満たす経路を評価する。
        if len(visited_dirs) >= int(max_dirs):
            break

        url, depth = q.popleft()
        url = _norm_url(url)
        # 条件分岐: `url in seen` を満たす経路を評価する。
        if url in seen:
            continue

        seen.add(url)
        # 条件分岐: `_should_skip_dir(url)` を満たす経路を評価する。
        if _should_skip_dir(url):
            continue

        # 条件分岐: `depth > int(max_depth)` を満たす経路を評価する。

        if depth > int(max_depth):
            continue

        visited_dirs.append(url)

        hrefs, method = _iter_listing_hrefs(
            url,
            timeout_sec=int(timeout_sec),
            retries=int(retries),
            retry_backoff_sec=float(retry_backoff_sec),
        )
        visited_dir_methods[url] = method
        # 条件分岐: `not hrefs` を満たす経路を評価する。
        if not hrefs:
            continue

        # Cap file links, but keep all directory links so we don't miss deep structures.

        dir_hrefs: List[str] = []
        file_hrefs: List[str] = []
        for href in hrefs:
            # 条件分岐: `href in ("../", "./")` を満たす経路を評価する。
            if href in ("../", "./"):
                continue

            child = urljoin(url, href)
            child = _norm_url(child)
            # 条件分岐: `_is_dir_url(child)` を満たす経路を評価する。
            if _is_dir_url(child):
                dir_hrefs.append(href)
            else:
                file_hrefs.append(href)

        # 条件分岐: `max_links_per_dir > 0` を満たす経路を評価する。

        if max_links_per_dir > 0:
            file_hrefs = file_hrefs[: int(max_links_per_dir)]

        hrefs = dir_hrefs + file_hrefs

        for href in hrefs:
            # 条件分岐: `href in ("../", "./")` を満たす経路を評価する。
            if href in ("../", "./"):
                continue

            child = urljoin(url, href)
            child = _norm_url(child)
            # 条件分岐: `_is_dir_url(child)` を満たす経路を評価する。
            if _is_dir_url(child):
                q.append((child, depth + 1))
                continue
            # File

            fname = Path(urlparse(child).path).name
            # 条件分岐: `not _is_candidate_file(fname)` を満たす経路を評価する。
            if not _is_candidate_file(fname):
                continue

            candidates.append(Candidate(url=child, filename=fname, source_dir=url))

    # Deduplicate candidates by URL

    uniq: Dict[str, Candidate] = {c.url: c for c in candidates}
    cand_sorted = sorted(uniq.values(), key=lambda c: c.url)
    meta = {
        "generated_utc": _now_utc(),
        "params": {
            "roots": roots,
            "max_depth": int(max_depth),
            "timeout_sec": int(timeout_sec),
            "max_dirs": int(max_dirs),
            "max_links_per_dir": int(max_links_per_dir),
        },
        "stats": {"visited_dirs": int(len(visited_dirs)), "candidates": int(len(cand_sorted))},
        "visited_dirs": visited_dirs,
        "visited_dir_methods": visited_dir_methods,
    }
    return cand_sorted, meta


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Discover DESI public 2pt products (multipoles/cov) by crawling directory listings.")
    ap.add_argument(
        "--roots",
        type=str,
        default=",".join(_DEFAULT_ROOTS),
        help="comma-separated root URLs to crawl (directory URLs recommended)",
    )
    ap.add_argument("--max-depth", type=int, default=6, help="crawl depth (default: 6)")
    ap.add_argument("--timeout-sec", type=int, default=30, help="HTTP timeout seconds (default: 30)")
    ap.add_argument("--retries", type=int, default=4, help="retry count for transient HTTP errors (default: 4)")
    ap.add_argument("--retry-backoff-sec", type=float, default=2.0, help="base backoff seconds for retries (default: 2.0)")
    ap.add_argument("--max-dirs", type=int, default=400, help="hard cap on number of directories to visit (default: 400)")
    ap.add_argument("--max-links-per-dir", type=int, default=300, help="cap links parsed per directory (default: 300)")
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(_ROOT / "output" / "private" / "cosmology" / "desi_public_2pt_discovery.json"),
        help="output JSON path (default: output/private/cosmology/desi_public_2pt_discovery.json)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    # 条件分岐: `requests is None` を満たす経路を評価する。
    if requests is None:
        raise SystemExit("requests is required to crawl remote listings. Install it in the Windows Python env.")

    roots = [r.strip() for r in str(args.roots).split(",") if r.strip()]
    cand, meta = discover(
        roots,
        max_depth=int(args.max_depth),
        timeout_sec=int(args.timeout_sec),
        max_dirs=int(args.max_dirs),
        max_links_per_dir=int(args.max_links_per_dir),
        retries=int(args.retries),
        retry_backoff_sec=float(args.retry_backoff_sec),
    )

    out_json = Path(str(args.out_json)).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        **meta,
        "candidates": [{"url": c.url, "filename": c.filename, "source_dir": c.source_dir} for c in cand],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_json}")
    print(f"[ok] visited_dirs={payload['stats']['visited_dirs']} candidates={payload['stats']['candidates']}")

    try:
        worklog.append_event(
            {
                "event_type": "discover_desi_public_2pt_products",
                "argv": sys.argv,
                "params": payload.get("params", {}),
                "outputs": {"json": str(out_json)},
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
