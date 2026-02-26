#!/usr/bin/env python3
"""
BepiColombo / MORE (Mercury Orbiter Radio-science Experiment)

PSA `bc_mpo_more/` の data_raw 等が公開されたタイミングで、
各ディレクトリの "collection / label / index"（小さめのメタデータ）を
自動で取得してキャッシュする。

狙い：
- 公開直後に「何が配布されるか」を一次ソースで確定する（collection_*.xml/csv 等）。
- 2回目以降はオフラインで再実行できる（data/ にキャッシュ）。

出力：
- data/bepicolombo/psa_more/fetch_collections_meta.json（キャッシュ/詳細ログ）
- output/private/bepicolombo/more_fetch_collections.json（要約）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            # 条件分岐: `not b` を満たす経路を評価する。
            if not b:
                break

            h.update(b)

    return h.hexdigest()


def _http_status(url: str, *, timeout_sec: float) -> Tuple[Optional[int], Optional[str]]:
    req = Request(url, method="HEAD")
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            return int(getattr(resp, "status", 200)), None
    except HTTPError as e:
        return int(getattr(e, "code", 0) or 0), None
    except Exception as e:
        return None, str(e)


def _download(url: str, dst: Path, *, timeout_sec: float) -> Dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `dst.exists() and dst.stat().st_size > 0` を満たす経路を評価する。
    if dst.exists() and dst.stat().st_size > 0:
        return {
            "url": url,
            "path": str(dst),
            "downloaded": False,
            "bytes": int(dst.stat().st_size),
            "sha256": _sha256(dst),
        }

    req = Request(url, method="GET")
    with urlopen(req, timeout=timeout_sec) as resp:
        data = resp.read()

    dst.write_bytes(data)
    return {
        "url": url,
        "path": str(dst),
        "downloaded": True,
        "bytes": int(len(data)),
        "sha256": _sha256(dst),
    }


def _parse_apache_index_rows(html_text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for ln in html_text.splitlines():
        # 条件分岐: `"<a href=" not in ln` を満たす経路を評価する。
        if "<a href=" not in ln:
            continue

        m = re.search(r'href="([^"]+)"', ln)
        # 条件分岐: `not m` を満たす経路を評価する。
        if not m:
            continue

        name = m.group(1)
        # 条件分岐: `name.startswith("?") or name.startswith("/") or name in ("../",)` を満たす経路を評価する。
        if name.startswith("?") or name.startswith("/") or name in ("../",):
            continue

        # 条件分岐: `name.endswith("/")` を満たす経路を評価する。

        if name.endswith("/"):
            continue

        # 条件分岐: `"Parent Directory" in ln` を満たす経路を評価する。

        if "Parent Directory" in ln:
            continue

        m2 = re.search(r"</a></td><td[^>]*>([^<]+)</td><td[^>]*>([^<]+)</td>", ln)
        last_modified = (m2.group(1).strip() if m2 else "").replace("\xa0", " ")
        size = (m2.group(2).strip() if m2 else "").replace("\xa0", " ")
        rows.append({"name": name, "last_modified": last_modified, "size": size})

    seen: set[str] = set()
    out: List[Dict[str, str]] = []
    for r in rows:
        n = r.get("name") or ""
        # 条件分岐: `not n or n in seen` を満たす経路を評価する。
        if not n or n in seen:
            continue

        seen.add(n)
        out.append(r)

    return out


def _is_metadata_file(name: str) -> bool:
    """
    Default policy: be conservative.

    In `data_raw/` etc there may be大量の製品ラベル（*.lblx）やデータ本体が並ぶため、
    ここでは「コレクション/索引に相当する小さめのメタ」に限定する。

    例（document/ で確認できるパターン）:
      - collection_<dir>.xml
      - collection_<dir>.csv
    """
    n = (name or "").lower()
    # 条件分岐: `not n` を満たす経路を評価する。
    if not n:
        return False

    # 条件分岐: `n.startswith("collection_") and n.endswith((".xml", ".csv", ".txt", ".md"))` を満たす経路を評価する。

    if n.startswith("collection_") and n.endswith((".xml", ".csv", ".txt", ".md")):
        return True
    # PSA側でREADME/INDEXが用意される場合に備える（小さめ想定）

    if n in ("readme.txt", "readme.md", "index.txt", "index.html"):
        return True

    return False


def _parse_size_hint_to_bytes(size_hint: str) -> Optional[int]:
    # Apache listing uses e.g. "21M", "549", "806K". Best-effort.
    s = (size_hint or "").strip()
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        return None

    m = re.match(r"^([0-9.]+)\s*([kKmMgG]?)$", s)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        return None

    val = float(m.group(1))
    suf = m.group(2).lower()
    mult = 1
    # 条件分岐: `suf == "k"` を満たす経路を評価する。
    if suf == "k":
        mult = 1024
    # 条件分岐: 前段条件が不成立で、`suf == "m"` を追加評価する。
    elif suf == "m":
        mult = 1024 * 1024
    # 条件分岐: 前段条件が不成立で、`suf == "g"` を追加評価する。
    elif suf == "g":
        mult = 1024 * 1024 * 1024

    return int(val * mult)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch PSA bc_mpo_more collection/label metadata when available")
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://archives.esac.esa.int/psa/ftp/BepiColombo/bc_mpo_more/",
        help="ESA PSA base URL for bc_mpo_more (trailing slash required).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not access the network; use cache meta if present and still write summary JSON.",
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Also download non-metadata files (can be huge). Default: disabled.",
    )
    parser.add_argument("--max-file-mb", type=float, default=50.0, help="Max single file size to download (MB).")
    parser.add_argument("--timeout-sec", type=float, default=30.0, help="Network timeout seconds. Default: 30")
    args = parser.parse_args()

    root = _ROOT
    data_dir = root / "data" / "bepicolombo" / "psa_more"
    out_dir = root / "output" / "private" / "bepicolombo"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = str(args.base_url).rstrip("/") + "/"
    expected_dirs = [
        "data_raw/",
        "data_calibrated/",
        "browse_raw/",
        "browse_calibrated/",
        "calibration_raw/",
        "miscellaneous/",
    ]

    cache_meta_path = data_dir / "fetch_collections_meta.json"
    meta: Dict[str, Any] = {"generated_utc": _utc_now_iso(), "base_url": base_url, "offline": bool(args.offline), "dirs": {}}

    prev_meta: Dict[str, Any] = {}
    # 条件分岐: `cache_meta_path.exists()` を満たす経路を評価する。
    if cache_meta_path.exists():
        try:
            prev_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        except Exception:
            prev_meta = {}

    # 条件分岐: `args.offline` を満たす経路を評価する。

    if args.offline:
        meta["note"] = "offline mode; network was not used"
        # 条件分岐: `isinstance(prev_meta, dict) and prev_meta.get("dirs")` を満たす経路を評価する。
        if isinstance(prev_meta, dict) and prev_meta.get("dirs"):
            meta["dirs"] = prev_meta.get("dirs") or {}
    else:
        max_file_bytes = int(float(args.max_file_mb) * 1024 * 1024)
        for d in expected_dirs:
            dir_url = urljoin(base_url, d)
            st, err = _http_status(dir_url, timeout_sec=float(args.timeout_sec))
            entry: Dict[str, Any] = {"status": st, "error": err, "downloaded": [], "skipped": []}

            # 条件分岐: `st != 200` を満たす経路を評価する。
            if st != 200:
                meta["dirs"][d] = entry
                continue

            # Fast path: try known collection file names without listing the directory.
            # (Avoid parsing huge Apache index when the directory contains many products.)

            collection_base = f"collection_{d.rstrip('/')}"
            for name in (f"{collection_base}.xml", f"{collection_base}.csv"):
                try:
                    dl = _download(urljoin(dir_url, name), data_dir / d.rstrip("/") / name, timeout_sec=float(args.timeout_sec))
                    entry["downloaded"].append(
                        {
                            "name": name,
                            "size_hint": "",
                            "bytes": dl.get("bytes"),
                            "path": dl.get("path"),
                            "downloaded": dl.get("downloaded"),
                        }
                    )
                except HTTPError as e:
                    # 404 is common when naming differs; fall back to listing below.
                    entry["skipped"].append({"name": name, "reason": f"http_{getattr(e, 'code', '') or 'error'}"})
                except Exception as e:
                    entry["skipped"].append({"name": name, "reason": f"download_failed: {e}"})

            # If we got the main collection files, do not list the directory unless explicitly requested.

            if entry["downloaded"] and not bool(args.download_data):
                meta["dirs"][d] = entry
                continue

            try:
                with urlopen(Request(dir_url, method="GET"), timeout=float(args.timeout_sec)) as resp:
                    html_text = resp.read().decode("utf-8", errors="replace")

                rows = _parse_apache_index_rows(html_text)
            except Exception as e:
                entry["error"] = f"list_failed: {e}"
                meta["dirs"][d] = entry
                continue

            for r in rows:
                name = str(r.get("name") or "")
                # 条件分岐: `not name` を満たす経路を評価する。
                if not name:
                    continue

                size_hint = str(r.get("size") or "")
                size_bytes = _parse_size_hint_to_bytes(size_hint)
                is_meta = _is_metadata_file(name)
                will_dl = is_meta or bool(args.download_data)
                # 条件分岐: `not will_dl` を満たす経路を評価する。
                if not will_dl:
                    entry["skipped"].append({"name": name, "reason": "non-metadata", "size_hint": size_hint})
                    continue

                # 条件分岐: `size_bytes is not None and size_bytes > max_file_bytes` を満たす経路を評価する。

                if size_bytes is not None and size_bytes > max_file_bytes:
                    entry["skipped"].append({"name": name, "reason": "too_large", "size_hint": size_hint, "size_bytes": size_bytes})
                    continue

                try:
                    dl = _download(dir_url + name, data_dir / d.rstrip("/") / name, timeout_sec=float(args.timeout_sec))
                    entry["downloaded"].append(
                        {
                            "name": name,
                            "size_hint": size_hint,
                            "bytes": dl.get("bytes"),
                            "path": dl.get("path"),
                            "downloaded": dl.get("downloaded"),
                        }
                    )
                except Exception as e:
                    entry["skipped"].append({"name": name, "reason": f"download_failed: {e}", "size_hint": size_hint})

            meta["dirs"][d] = entry

    # Persist cache meta for offline re-run.

    data_dir.mkdir(parents=True, exist_ok=True)
    cache_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Summary JSON (dashboard-facing)
    out_json = out_dir / "more_fetch_collections.json"
    summary: Dict[str, Any] = {"generated_utc": meta.get("generated_utc"), "base_url": base_url, "offline": bool(meta.get("offline")), "dirs": {}}
    for d, v in (meta.get("dirs") or {}).items():
        # 条件分岐: `not isinstance(v, dict)` を満たす経路を評価する。
        if not isinstance(v, dict):
            continue

        downloaded = v.get("downloaded") or []
        skipped = v.get("skipped") or []
        summary["dirs"][d] = {
            "status": v.get("status"),
            "downloaded_n": int(len(downloaded)) if isinstance(downloaded, list) else 0,
            "skipped_n": int(len(skipped)) if isinstance(skipped, list) else 0,
            "error": v.get("error"),
        }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        worklog.append_event(
            {
                "event_type": "bepicolombo_more_fetch_collections",
                "argv": sys.argv,
                "inputs": {"base_url": base_url},
                "params": {"offline": bool(args.offline), "download_data": bool(args.download_data), "max_file_mb": float(args.max_file_mb)},
                "outputs": {"cache_meta_json": cache_meta_path, "summary_json": out_json},
            }
        )
    except Exception:
        pass

    print("Wrote:", out_json)


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
