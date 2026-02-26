from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from html import unescape
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMEOUT_S = 25
DEFAULT_ATTEMPTS = 2


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_stable_sha256_text` の入出力契約と処理意図を定義する。

def _stable_sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# 関数: `_fetch_json` の入出力契約と処理意図を定義する。

def _fetch_json(url: str, *, timeout_s: int = 60) -> Any:
    req = Request(url, headers={"User-Agent": "waveP/quantum-search"})
    with urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8"))


# 関数: `_fetch_text` の入出力契約と処理意図を定義する。

def _fetch_text(url: str, *, timeout_s: int = 60) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (waveP/quantum-search)"})
    with urlopen(req, timeout=timeout_s) as r:
        return r.read().decode("utf-8", errors="replace")


# 関数: `_fetch_json_retry` の入出力契約と処理意図を定義する。

def _fetch_json_retry(url: str, *, timeout_s: int | None = None, attempts: int | None = None, sleep_s: float = 1.0) -> Any:
    timeout_s = int(DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s)
    attempts = int(DEFAULT_ATTEMPTS if attempts is None else attempts)
    last_err: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            return _fetch_json(url, timeout_s=int(timeout_s))
        except (HTTPError, URLError, TimeoutError) as e:
            last_err = e
            # 条件分岐: `i >= attempts` を満たす経路を評価する。
            if i >= attempts:
                break

            time.sleep(sleep_s * i)

    raise RuntimeError(f"fetch failed after {attempts} attempts: {url} ({type(last_err).__name__}: {last_err})")


# 関数: `_fetch_text_retry` の入出力契約と処理意図を定義する。

def _fetch_text_retry(url: str, *, timeout_s: int | None = None, attempts: int | None = None, sleep_s: float = 1.0) -> str:
    timeout_s = int(DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s)
    attempts = int(DEFAULT_ATTEMPTS if attempts is None else attempts)
    last_err: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            return _fetch_text(url, timeout_s=int(timeout_s))
        except (HTTPError, URLError, TimeoutError) as e:
            last_err = e
            # 条件分岐: `i >= attempts` を満たす経路を評価する。
            if i >= attempts:
                break

            time.sleep(sleep_s * i)

    raise RuntimeError(f"fetch failed after {attempts} attempts: {url} ({type(last_err).__name__}: {last_err})")


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `_zenodo_search` の入出力契約と処理意図を定義する。

def _zenodo_search(*, q: str, size: int = 10) -> dict[str, Any]:
    url = f"https://zenodo.org/api/records/?q={quote(q)}&size={int(size)}"
    obj = _fetch_json_retry(url)
    hits = obj.get("hits") if isinstance(obj.get("hits"), dict) else {}
    total = hits.get("total")
    items = hits.get("hits") if isinstance(hits.get("hits"), list) else []
    out_items: list[dict[str, Any]] = []
    for it in items:
        # 条件分岐: `not isinstance(it, dict)` を満たす経路を評価する。
        if not isinstance(it, dict):
            continue

        md = it.get("metadata") if isinstance(it.get("metadata"), dict) else {}
        files = it.get("files") if isinstance(it.get("files"), list) else []
        out_items.append(
            {
                "record_id": it.get("id"),
                "title": md.get("title"),
                "doi": md.get("doi"),
                "created": it.get("created"),
                "updated": it.get("updated"),
                "html": (it.get("links") or {}).get("html") if isinstance(it.get("links"), dict) else None,
                "files_n": int(len(files)),
                "files_names": [f.get("key") for f in files if isinstance(f, dict) and f.get("key")] if files else [],
            }
        )

    return {"platform": "zenodo", "q": q, "url": url, "total": total, "items": out_items}


# 関数: `_osf_search` の入出力契約と処理意図を定義する。

def _osf_search(*, q: str, size: int = 10) -> dict[str, Any]:
    # OSF search is very broad; we record only the top few hits and then filter offline.
    url = f"https://api.osf.io/v2/search/?q={quote(q)}&page[size]={int(size)}"
    obj = _fetch_json_retry(url)
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    total = meta.get("total")
    data = obj.get("data") if isinstance(obj.get("data"), list) else []
    items: list[dict[str, Any]] = []
    for d in data:
        # 条件分岐: `not isinstance(d, dict)` を満たす経路を評価する。
        if not isinstance(d, dict):
            continue

        attrs = d.get("attributes") if isinstance(d.get("attributes"), dict) else {}
        links = d.get("links") if isinstance(d.get("links"), dict) else {}
        items.append(
            {
                "id": d.get("id"),
                "type": d.get("type"),
                "title": attrs.get("title"),
                "date_created": attrs.get("date_created"),
                "url": attrs.get("absolute_url") or links.get("html"),
            }
        )

    return {"platform": "osf", "q": q, "url": url, "total": total, "items": items}


# 関数: `_datacite_search` の入出力契約と処理意図を定義する。

def _datacite_search(*, q: str, size: int = 10) -> dict[str, Any]:
    url = f"https://api.datacite.org/dois?query={quote(q)}&page[size]={int(size)}"
    obj = _fetch_json_retry(url)
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    total = meta.get("total")
    data = obj.get("data") if isinstance(obj.get("data"), list) else []
    items: list[dict[str, Any]] = []
    for d in data:
        # 条件分岐: `not isinstance(d, dict)` を満たす経路を評価する。
        if not isinstance(d, dict):
            continue

        attrs = d.get("attributes") if isinstance(d.get("attributes"), dict) else {}
        items.append(
            {
                "doi": d.get("id"),
                "titles": attrs.get("titles"),
                "publisher": attrs.get("publisher"),
                "publicationYear": attrs.get("publicationYear"),
                "types": attrs.get("types"),
                "url": attrs.get("url"),
            }
        )

    return {"platform": "datacite", "q": q, "url": url, "total": total, "items": items}


# 関数: `_crossref_search` の入出力契約と処理意図を定義する。

def _crossref_search(*, q: str, size: int = 10) -> dict[str, Any]:
    url = f"https://api.crossref.org/works?query={quote(q)}&rows={int(size)}"
    obj = _fetch_json_retry(url)
    msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
    total = msg.get("total-results")
    data = msg.get("items") if isinstance(msg.get("items"), list) else []
    items: list[dict[str, Any]] = []
    for d in data:
        # 条件分岐: `not isinstance(d, dict)` を満たす経路を評価する。
        if not isinstance(d, dict):
            continue

        title = d.get("title")
        title0 = title[0] if isinstance(title, list) and title else title
        items.append(
            {
                "doi": d.get("DOI"),
                "type": d.get("type"),
                "title": title0,
                "publisher": d.get("publisher"),
                "issued": d.get("issued"),
                "url": d.get("URL"),
                "link": d.get("link"),
            }
        )

    return {"platform": "crossref", "q": q, "url": url, "total": total, "items": items}


# 関数: `_openalex_search` の入出力契約と処理意図を定義する。

def _openalex_search(*, q: str, size: int = 10) -> dict[str, Any]:
    # OpenAlex API (no key required). Use per-page; fall back to per_page if needed.
    url = f"https://api.openalex.org/works?search={quote(q)}&per-page={int(size)}"
    try:
        obj = _fetch_json_retry(url)
    except Exception:
        url = f"https://api.openalex.org/works?search={quote(q)}&per_page={int(size)}"
        obj = _fetch_json_retry(url)

    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    total = meta.get("count")
    data = obj.get("results") if isinstance(obj.get("results"), list) else []
    items: list[dict[str, Any]] = []
    for d in data:
        # 条件分岐: `not isinstance(d, dict)` を満たす経路を評価する。
        if not isinstance(d, dict):
            continue

        items.append(
            {
                "id": d.get("id"),
                "doi": d.get("doi"),
                "type": d.get("type"),
                "title": d.get("title"),
                "publication_year": d.get("publication_year"),
                "primary_location": d.get("primary_location"),
                "open_access": d.get("open_access"),
            }
        )

    return {"platform": "openalex", "q": q, "url": url, "total": total, "items": items}


# 関数: `_figshare_search` の入出力契約と処理意図を定義する。

def _figshare_search(*, q: str, size: int = 10) -> dict[str, Any]:
    # Figshare public API: broad; we record the first page of hits.
    url = f"https://api.figshare.com/v2/articles?search_for={quote(q)}&page_size={int(size)}&page=1"
    data = _fetch_json_retry(url)
    items: list[dict[str, Any]] = []
    # 条件分岐: `isinstance(data, list)` を満たす経路を評価する。
    if isinstance(data, list):
        for d in data:
            # 条件分岐: `not isinstance(d, dict)` を満たす経路を評価する。
            if not isinstance(d, dict):
                continue

            items.append(
                {
                    "id": d.get("id"),
                    "title": d.get("title"),
                    "doi": d.get("doi"),
                    "published_date": d.get("published_date"),
                    "defined_type_name": d.get("defined_type_name"),
                    "group_id": d.get("group_id"),
                    "url_public_html": d.get("url_public_html"),
                    "url_public_api": d.get("url_public_api"),
                }
            )

    return {"platform": "figshare", "q": q, "url": url, "total": None, "items": items}


# 関数: `_harvard_dataverse_search` の入出力契約と処理意図を定義する。

def _harvard_dataverse_search(*, q: str, size: int = 10) -> dict[str, Any]:
    # There is no central Dataverse search; Harvard is a reasonable public baseline.
    url = f"https://dataverse.harvard.edu/api/search?q={quote(q)}&type=dataset&per_page={int(size)}"
    obj = _fetch_json_retry(url)
    data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
    total = data.get("total_count")
    items0 = data.get("items") if isinstance(data.get("items"), list) else []
    items: list[dict[str, Any]] = []
    for d in items0:
        # 条件分岐: `not isinstance(d, dict)` を満たす経路を評価する。
        if not isinstance(d, dict):
            continue

        items.append(
            {
                "name": d.get("name"),
                "type": d.get("type"),
                "url": d.get("url"),
                "global_id": d.get("global_id"),
                "published_at": d.get("published_at"),
                "citation": d.get("citation"),
                "authors": d.get("authors"),
            }
        )

    return {"platform": "dataverse_harvard", "q": q, "url": url, "total": total, "items": items}


_PIRACY_HOST_SUBSTRINGS = [
    "sci-hub",
    "scihub",
    "libgen",
    "library.lol",
    "z-lib",
    "zlibrary",
    "annas-archive",
]


# 関数: `_is_piracy_url` の入出力契約と処理意図を定義する。
def _is_piracy_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(s in host for s in _PIRACY_HOST_SUBSTRINGS)


# 関数: `_ddg_unwrap_redirect` の入出力契約と処理意図を定義する。

def _ddg_unwrap_redirect(url: str) -> str:
    # DuckDuckGo result links are usually of the form:
    #   //duckduckgo.com/l/?uddg=<urlencoded>&rut=<...>
    # We prefer to store the underlying target URL (uddg) for reproducibility.
    u = unescape(url.strip())
    # 条件分岐: `u.startswith("//")` を満たす経路を評価する。
    if u.startswith("//"):
        u = "https:" + u

    parsed = urlparse(u)
    # 条件分岐: `"duckduckgo.com" not in parsed.netloc.lower()` を満たす経路を評価する。
    if "duckduckgo.com" not in parsed.netloc.lower():
        return u

    qs = parse_qs(parsed.query)
    # 条件分岐: `"uddg" in qs and qs["uddg"]` を満たす経路を評価する。
    if "uddg" in qs and qs["uddg"]:
        return unquote(qs["uddg"][0])

    return u


# 関数: `_duckduckgo_html_search` の入出力契約と処理意図を定義する。

def _duckduckgo_html_search(*, q: str, size: int = 10) -> dict[str, Any]:
    # DuckDuckGo HTML endpoint (no JS) for best-effort web search.
    url = f"https://html.duckduckgo.com/html/?q={quote(q)}"
    html = _fetch_text_retry(url)

    # 関数: `strip_tags` の入出力契約と処理意図を定義する。
    def strip_tags(s: str) -> str:
        return re.sub(r"<[^>]+>", "", s).strip()

    # Extract result links and titles.

    raw = re.findall(r'<a\s+rel="nofollow"\s+class="result__a"\s+href="([^"]+)">(.*?)</a>', html, flags=re.S)
    filter_keys = ["giustina", "bell", "zeilinger", "physrevlett", "1511.03190", "loophole"]
    items: list[dict[str, Any]] = []
    for href, raw_title in raw[: int(size) * 6]:
        target = _ddg_unwrap_redirect(href)
        # 条件分岐: `not target or _is_piracy_url(target)` を満たす経路を評価する。
        if not target or _is_piracy_url(target):
            continue

        title = strip_tags(unescape(raw_title))
        hay = f"{title} {target}".lower()
        # 条件分岐: `not any(k in hay for k in filter_keys)` を満たす経路を評価する。
        if not any(k in hay for k in filter_keys):
            continue

        items.append({"title": title, "url": target})
        # 条件分岐: `len(items) >= int(size)` を満たす経路を評価する。
        if len(items) >= int(size):
            break

    return {"platform": "duckduckgo_html", "q": q, "url": url, "total": None, "items": items}


# 関数: `_to_text` の入出力契約と処理意図を定義する。

def _to_text(v: Any) -> str:
    # 条件分岐: `v is None` を満たす経路を評価する。
    if v is None:
        return ""

    # 条件分岐: `isinstance(v, (str, int, float))` を満たす経路を評価する。

    if isinstance(v, (str, int, float)):
        return str(v)

    # 条件分岐: `isinstance(v, list)` を満たす経路を評価する。

    if isinstance(v, list):
        return " ".join(_to_text(x) for x in v[:5])

    # 条件分岐: `isinstance(v, dict)` を満たす経路を評価する。

    if isinstance(v, dict):
        # Keep it shallow; avoid dumping huge JSON into the matcher.
        parts: list[str] = []
        for k in ("title", "doi", "url", "URL", "global_id", "id"):
            # 条件分岐: `k in v` を満たす経路を評価する。
            if k in v:
                parts.append(_to_text(v.get(k)))

        return " ".join(p for p in parts if p)

    return str(v)


# 関数: `_is_plausible_item` の入出力契約と処理意図を定義する。

def _is_plausible_item(item: dict[str, Any]) -> bool:
    hay = " ".join(
        [
            _to_text(item.get("title")),
            _to_text(item.get("doi")),
            _to_text(item.get("url")),
            _to_text(item.get("url_public_html")),
            _to_text(item.get("global_id")),
        ]
    ).lower()
    # 250401 is too generic as a bare token (it appears in many unrelated DOIs). Keep the matcher strict.
    if "10.1103/physrevlett.115.250401" in hay or "physrevlett.115.250401" in hay:
        return True

    # 条件分岐: `"1511.03190" in hay` を満たす経路を評価する。

    if "1511.03190" in hay:
        return True
    # Name-based fallback: require author + context words to avoid "vienna" false positives.

    if "giustina" in hay and ("bell" in hay or "loophole" in hay or "zeilinger" in hay):
        return True

    return False


# 関数: `_is_target_work_hit` の入出力契約と処理意図を定義する。

def _is_target_work_hit(item: dict[str, Any]) -> bool:
    hay = " ".join(
        [
            _to_text(item.get("doi")),
            _to_text(item.get("url")),
            _to_text(item.get("url_public_html")),
            _to_text(item.get("id")),
            _to_text(item.get("global_id")),
        ]
    ).lower()
    return ("10.1103/physrevlett.115.250401" in hay) or ("arxiv.1511.03190" in hay) or ("1511.03190" in hay)


# 関数: `_looks_like_dataset_url` の入出力契約と処理意図を定義する。

def _looks_like_dataset_url(url: str) -> bool:
    u = url.lower()
    # File-like direct links
    exts = [".zip", ".tar", ".tar.gz", ".tgz", ".csv", ".json", ".jsonl", ".txt", ".dat", ".h5", ".fits"]
    # 条件分岐: `any(u.endswith(ext) for ext in exts)` を満たす経路を評価する。
    if any(u.endswith(ext) for ext in exts):
        return True

    # 条件分岐: `any(ext in u for ext in exts) and ("download" in u or "files" in u or "attach...` を満たす経路を評価する。

    if any(ext in u for ext in exts) and ("download" in u or "files" in u or "attachment" in u):
        return True
    # Repository-like landing pages

    repos = [
        "zenodo.org/record",
        "zenodo.org/records",
        "osf.io",
        "figshare.com",
        "dataverse.harvard.edu/dataset.xhtml",
        "data.4tu.nl",
        "doi.org/10.4121",
    ]
    return any(r in u for r in repos)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=20, help="Max hits per query per platform (default: 20)")
    ap.add_argument("--timeout-s", type=int, default=25, help="Per-request timeout in seconds (default: 25)")
    ap.add_argument("--attempts", type=int, default=2, help="Retry attempts per request (default: 2)")
    ap.add_argument("--out", default="output/public/quantum/bell/giustina2015_clicklog_public_search.json")
    args = ap.parse_args()

    global DEFAULT_TIMEOUT_S, DEFAULT_ATTEMPTS
    DEFAULT_TIMEOUT_S = int(args.timeout_s)
    DEFAULT_ATTEMPTS = int(args.attempts)

    queries = [
        '"PhysRevLett.115.250401"',
        '"10.1103/PhysRevLett.115.250401"',
        "Giustina loophole-free Bell data",
        "Giustina Zeilinger 2015 Bell experiment data",
        "Vienna loophole-free bell time tag",
        "Bell violation with entangled photons fair sampling data",
    ]

    ddg_queries = [
        "Giustina Bell time tag data",
        "Giustina click log Bell",
        "PhysRevLett.115.250401 data",
        "10.1103/PhysRevLett.115.250401 data",
        "1511.03190 data",
        "site:univie.ac.at Giustina Bell 2015 data",
        "site:coqus.at Giustina Bell 2015",
        # Thesis-related probes (Giustina 2016 uTheses/Phaidra) sometimes cited as “In preparation”.
        "Bell’s inequality and two conscientious experiments pdf",
        "utheses univie 39955 pdf",
        "othes univie 45139 pdf",
        "phaidra o:1331601 39955.pdf",
        "urn:nbn:at:at-ubw:1-25688.17316.644853-9 pdf",
    ]

    started = _utc_now()
    results: list[dict[str, Any]] = []
    for q in queries:
        print(f"[run] q={q}")
        # Keep each platform independent; failures become logged errors rather than aborting the run.
        for fn, name in [
            (_zenodo_search, "zenodo"),
            (_osf_search, "osf"),
            (_datacite_search, "datacite"),
            (_crossref_search, "crossref"),
            (_openalex_search, "openalex"),
            (_figshare_search, "figshare"),
            (_harvard_dataverse_search, "dataverse_harvard"),
        ]:
            try:
                r = fn(q=q, size=int(args.size))
                results.append(r)
                print(f"  - {name}: total={r.get('total')}")
            except Exception as e:
                results.append({"platform": name, "q": q, "error": f"{type(e).__name__}: {e}"})
                print(f"  - {name}: error={type(e).__name__}: {e}")

    # Best-effort web search (DDG HTML).

    for q in ddg_queries:
        print(f"[run] q(ddg)={q}")
        try:
            r = _duckduckgo_html_search(q=q, size=int(args.size))
            results.append(r)
            print(f"  - duckduckgo_html: n_items={len(r.get('items') or [])}")
        except Exception as e:
            results.append({"platform": "duckduckgo_html", "q": q, "error": f"{type(e).__name__}: {e}"})
            print(f"  - duckduckgo_html: error={type(e).__name__}: {e}")

    # Collect a short list of plausible hits (strict heuristic).

    plausible: list[dict[str, Any]] = []
    for r in results:
        items = r.get("items") if isinstance(r.get("items"), list) else []
        for it in items:
            # 条件分岐: `not isinstance(it, dict)` を満たす経路を評価する。
            if not isinstance(it, dict):
                continue

            # 条件分岐: `_is_plausible_item(it)` を満たす経路を評価する。

            if _is_plausible_item(it):
                plausible.append({"platform": r.get("platform"), "q": r.get("q"), **it})

    known_work_hits = [p for p in plausible if _is_target_work_hit(p)]
    candidate_dataset_hits = [p for p in plausible if (not _is_target_work_hit(p)) and _looks_like_dataset_url(_to_text(p))]

    out = {
        "generated_utc": _utc_now(),
        "started_utc": started,
        "goal": "Find a publicly accessible primary source for Giustina 2015 click log (time-tag) suitable for fixed reanalysis.",
        "target_work": {"doi": "10.1103/PhysRevLett.115.250401", "arxiv": "1511.03190"},
        "queries": queries,
        "ddg_queries": ddg_queries,
        "request_limits": {"timeout_s": int(args.timeout_s), "attempts": int(args.attempts)},
        "results": results,
        "plausible_hits": plausible,
        "known_work_hits": known_work_hits,
        "candidate_dataset_hits": candidate_dataset_hits,
        "summary": {
            "n_queries": int(len(queries)),
            "n_platform_results": int(len(results)),
            "n_plausible_hits": int(len(plausible)),
            "n_known_work_hits": int(len(known_work_hits)),
            "n_candidate_dataset_hits": int(len(candidate_dataset_hits)),
            "status": (
                "no_public_clicklog_found_yet"
                if not candidate_dataset_hits
                else "candidate_dataset_hits_found_check_manually"
            ),
            "note": (
                "This is a best-effort automated search over major open repositories; it does not guarantee completeness."
            ),
        },
        "reproduce": {
            "cmd": (
                f"python -B scripts/quantum/{Path(__file__).name} --size {int(args.size)}"
                f" --timeout-s {int(args.timeout_s)} --attempts {int(args.attempts)}"
            )
        },
    }

    out_path = ROOT / str(args.out)
    _write_json(out_path, out)
    print(f"[ok] wrote: {out_path}")
    print(f"[info] out_sha256: {_stable_sha256_text(json.dumps(out, ensure_ascii=False))}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
