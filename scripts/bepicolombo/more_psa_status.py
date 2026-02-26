#!/usr/bin/env python3
"""
BepiColombo / MORE (Mercury Orbiter Radio-science Experiment)

ESA PSA に公開されている `bc_mpo_more` バンドルの「公開状況」を確認し、
ローカルにキャッシュしつつ、一般向けレポートに載せられる PNG を生成する。

目的：
- Cassini の次ステップとして BepiColombo（MORE）の一次データ検証を準備する。
- まず「公開されているか？」を一次ソース（PSA）で確認し、再現可能に残す。

出力：
- `output/private/bepicolombo/more_psa_status.json`
- `output/private/bepicolombo/more_psa_status.png`

キャッシュ：
- `data/bepicolombo/psa_more/`（bundle xml / document など）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


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

def _fetch_cached_html(url: str, cache_path: Path, *, offline: bool, timeout_sec: float) -> Tuple[Optional[str], Optional[str]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `offline` を満たす経路を評価する。
    if offline:
        # 条件分岐: `not cache_path.exists()` を満たす経路を評価する。
        if not cache_path.exists():
            return None, f"offline mode: cache missing: {cache_path}"

        try:
            return cache_path.read_bytes().decode("utf-8", errors="replace"), None
        except Exception as e:
            return None, f"offline mode: failed to read cache: {e}"

    try:
        _download(url, cache_path, timeout_sec=timeout_sec)
        return cache_path.read_bytes().decode("utf-8", errors="replace"), None
    except Exception as e:
        return None, str(e)


def _parse_apache_index_entries(html_text: str, *, include_dirs: bool, include_files: bool) -> List[Dict[str, str]]:
    """
    Apache-style directory listing row example:
      <tr><td ...><a href="file.pdf">file.pdf</a></td><td align="right">2025-..</td><td align="right"> 21M</td>...
    """
    rows: List[Dict[str, str]] = []
    for ln in html_text.splitlines():
        # 条件分岐: `"<a href=" not in ln` を満たす経路を評価する。
        if "<a href=" not in ln:
            continue

        m = re.search(r'href="([^"]+)"', ln)
        # 条件分岐: `not m` を満たす経路を評価する。
        if not m:
            continue

        href = m.group(1)
        # 条件分岐: `href.startswith("?") or href.startswith("/")` を満たす経路を評価する。
        if href.startswith("?") or href.startswith("/"):
            continue

        # 条件分岐: `href in ("../",)` を満たす経路を評価する。

        if href in ("../",):
            continue

        # 条件分岐: `"Parent Directory" in ln` を満たす経路を評価する。

        if "Parent Directory" in ln:
            continue

        is_dir = href.endswith("/")
        # 条件分岐: `is_dir and not include_dirs` を満たす経路を評価する。
        if is_dir and not include_dirs:
            continue

        # 条件分岐: `(not is_dir) and not include_files` を満たす経路を評価する。

        if (not is_dir) and not include_files:
            continue

        # Try to capture modified/size columns (best-effort).

        m2 = re.search(r"</a></td><td[^>]*>([^<]+)</td><td[^>]*>([^<]+)</td>", ln)
        last_modified = (m2.group(1).strip() if m2 else "").replace("\xa0", " ")
        size = (m2.group(2).strip() if m2 else "").replace("\xa0", " ")
        rows.append({"name": href, "is_dir": "1" if is_dir else "0", "last_modified": last_modified, "size": size})
    # De-dup while preserving order.

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


def _set_japanese_font() -> None:
    # Best-effort: pick a Japanese-capable font if available.
    try:
        import matplotlib

        candidates = [
            "Yu Gothic",
            "Yu Gothic UI",
            "Meiryo",
            "MS Gothic",
            "Noto Sans CJK JP",
            "IPAexGothic",
        ]
        for name in candidates:
            try:
                matplotlib.font_manager.findfont(name, fallback_to_default=False)  # type: ignore[attr-defined]
                matplotlib.rcParams["font.family"] = name
                break
            except Exception:
                continue
    except Exception:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="BepiColombo (MORE) PSA availability check + report artifacts")
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://archives.esac.esa.int/psa/ftp/BepiColombo/bc_mpo_more/",
        help="ESA PSA base URL for bc_mpo_more (trailing slash required).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not access the network; use local cache if present and still generate outputs.",
    )
    parser.add_argument(
        "--download-pdfs",
        action="store_true",
        help="Also download PDF documents (can be large). Default: disabled.",
    )
    parser.add_argument("--timeout-sec", type=float, default=30.0, help="Network timeout seconds. Default: 30")
    args = parser.parse_args()

    root = _repo_root()
    data_dir = root / "data" / "bepicolombo" / "psa_more"
    out_dir = root / "output" / "private" / "bepicolombo"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = str(args.base_url).rstrip("/") + "/"
    bundle_url = urljoin(base_url, "bundle_bc_mpo_more.xml")
    doc_url = urljoin(base_url, "document/")
    parent_url = urljoin(base_url, "../")

    expected_dirs = [
        "data_raw/",
        "data_calibrated/",
        "browse_raw/",
        "browse_calibrated/",
        "calibration_raw/",
        "miscellaneous/",
        "document/",
    ]

    meta: Dict[str, Any] = {
        "generated_utc": _utc_now_iso(),
        "base_url": base_url,
        "offline": bool(args.offline),
        "expected": {},
        "listing": {},
        "downloads": [],
        "documents": [],
    }

    # Load previous meta if present (offline-friendly).
    cache_meta_path = data_dir / "fetch_meta.json"
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
        # 条件分岐: `prev_meta` を満たす経路を評価する。
        if prev_meta:
            meta["expected"] = prev_meta.get("expected") or {}
            meta["listing"] = prev_meta.get("listing") or {}
            meta["downloads"] = prev_meta.get("downloads") or []
            meta["documents"] = prev_meta.get("documents") or []
    else:
        # Check expected directories by HTTP status.
        exp: Dict[str, Any] = {}
        for d in expected_dirs:
            st, err = _http_status(urljoin(base_url, d), timeout_sec=float(args.timeout_sec))
            exp[d] = {"status": st, "error": err}

        meta["expected"] = exp

        # Always fetch the bundle XML (small).
        try:
            meta["downloads"].append(_download(bundle_url, data_dir / "bundle_bc_mpo_more.xml", timeout_sec=float(args.timeout_sec)))
        except Exception as e:
            meta.setdefault("errors", []).append({"stage": "download_bundle", "error": str(e)})

        # Document listing + selective downloads.

        try:
            html_text, html_err = _fetch_cached_html(
                doc_url, data_dir / "document_index.html", offline=False, timeout_sec=float(args.timeout_sec)
            )
            # 条件分岐: `html_err` を満たす経路を評価する。
            if html_err:
                raise RuntimeError(html_err)

            doc_rows = _parse_apache_index_entries(html_text or "", include_dirs=False, include_files=True)

            docs: List[Dict[str, Any]] = []
            for r in doc_rows:
                name = str(r.get("name") or "")
                # 条件分岐: `not name` を満たす経路を評価する。
                if not name:
                    continue

                is_pdf = name.lower().endswith(".pdf")
                will_download = (not is_pdf) or bool(args.download_pdfs)
                doc_entry: Dict[str, Any] = {
                    "name": name,
                    "last_modified": str(r.get("last_modified") or ""),
                    "size": str(r.get("size") or ""),
                    "downloaded": False,
                    "path": None,
                }
                # 条件分岐: `will_download` を満たす経路を評価する。
                if will_download:
                    try:
                        dl = _download(urljoin(doc_url, name), data_dir / "document" / name, timeout_sec=float(args.timeout_sec))
                        doc_entry["downloaded"] = bool(dl.get("downloaded"))
                        doc_entry["path"] = str((data_dir / "document" / name).resolve())
                    except Exception as e:
                        doc_entry["error"] = str(e)

                docs.append(doc_entry)

            meta["documents"] = docs
        except Exception as e:
            meta.setdefault("errors", []).append({"stage": "list_or_download_docs", "error": str(e)})

        # Parent/base listing (what is present on PSA today)

        try:
            parent_html, parent_err = _fetch_cached_html(
                parent_url, data_dir / "parent_index.html", offline=False, timeout_sec=float(args.timeout_sec)
            )
            base_html, base_err = _fetch_cached_html(
                base_url, data_dir / "base_index.html", offline=False, timeout_sec=float(args.timeout_sec)
            )

            listing: Dict[str, Any] = {}
            # 条件分岐: `parent_err` を満たす経路を評価する。
            if parent_err:
                listing["parent_error"] = parent_err
            else:
                parent_entries = _parse_apache_index_entries(parent_html or "", include_dirs=True, include_files=True)
                listing["parent_url"] = parent_url
                listing["parent_entries"] = parent_entries

            # 条件分岐: `base_err` を満たす経路を評価する。

            if base_err:
                listing["base_error"] = base_err
            else:
                base_entries = _parse_apache_index_entries(base_html or "", include_dirs=True, include_files=True)
                listing["base_url"] = base_url
                listing["base_entries"] = base_entries

            meta["listing"] = listing
        except Exception as e:
            meta.setdefault("errors", []).append({"stage": "list_base_parent", "error": str(e)})

    # Persist cache meta (for offline re-run).

    data_dir.mkdir(parents=True, exist_ok=True)
    cache_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Output JSON (smaller, dashboard-facing).
    status_out = out_dir / "more_psa_status.json"
    status_payload: Dict[str, Any] = {
        "generated_utc": meta.get("generated_utc"),
        "base_url": base_url,
        "parent_url": parent_url,
        "offline": bool(meta.get("offline")),
        "expected": meta.get("expected") or {},
        "listing": meta.get("listing") or {},
        "document_total": int(len(meta.get("documents") or [])),
        "document_downloaded": int(sum(1 for d in (meta.get("documents") or []) if d.get("path"))),
        "has_data_dirs": any(
            (isinstance(v, dict) and int(v.get("status") or 0) == 200)
            for k, v in (meta.get("expected") or {}).items()
            if str(k) not in ("document/",)
        ),
        "errors": meta.get("errors") or [],
    }
    status_out.write_text(json.dumps(status_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # PNG: simple "status board" for the public report.
    try:
        import matplotlib.pyplot as plt  # type: ignore

        _set_japanese_font()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("off")

        lines: List[str] = []
        lines.append("BepiColombo（MORE）: ESA PSA 公開状況（一次ソース確認）")
        lines.append(f"確認時刻(UTC): {status_payload.get('generated_utc')}")
        lines.append(f"対象: {base_url}")
        # 条件分岐: `status_payload.get("offline")` を満たす経路を評価する。
        if status_payload.get("offline"):
            lines.append("モード: offline（ネットワーク未使用）")

        exp = status_payload.get("expected") or {}
        # 条件分岐: `exp` を満たす経路を評価する。
        if exp:
            lines.append("")
            lines.append("公開ディレクトリ（HTTPステータス）:")
            for d in expected_dirs:
                v = exp.get(d) or {}
                st = v.get("status")
                # 条件分岐: `st is None` を満たす経路を評価する。
                if st is None:
                    s = "unknown"
                else:
                    s = str(st)

                ok = "OK" if str(st) == "200" else "NG"
                lines.append(f"  - {d:<18} : {ok} ({s})")
        else:
            lines.append("")
            lines.append("公開ディレクトリ: 未確認（キャッシュ無し/未取得）")

        lines.append("")
        lines.append(
            f"ドキュメント: {status_payload.get('document_downloaded')}/{status_payload.get('document_total')} 件をローカルに保持"
        )

        listing = status_payload.get("listing") or {}
        # 条件分岐: `isinstance(listing, dict)` を満たす経路を評価する。
        if isinstance(listing, dict):
            base_entries = listing.get("base_entries") if isinstance(listing.get("base_entries"), list) else []
            parent_entries = listing.get("parent_entries") if isinstance(listing.get("parent_entries"), list) else []
            # 条件分岐: `base_entries` を満たす経路を評価する。
            if base_entries:
                dirs = [e.get("name") for e in base_entries if isinstance(e, dict) and e.get("is_dir") == "1"]
                files = [e.get("name") for e in base_entries if isinstance(e, dict) and e.get("is_dir") == "0"]
                lines.append("")
                lines.append(f"bc_mpo_more/ 直下: dir={len(dirs)}, file={len(files)}")
                # 条件分岐: `files` を満たす経路を評価する。
                if files:
                    show = ", ".join(str(x) for x in files[:3])
                    lines.append(f"  files例: {show}{' …' if len(files) > 3 else ''}")

            # 条件分岐: `parent_entries` を満たす経路を評価する。

            if parent_entries:
                dirs = [e.get("name") for e in parent_entries if isinstance(e, dict) and e.get("is_dir") == "1"]
                # 条件分岐: `dirs` を満たす経路を評価する。
                if dirs:
                    show = ", ".join(str(x) for x in dirs[:6])
                    lines.append("")
                    lines.append(f"PSA BepiColombo/ 直下 dir例: {show}{' …' if len(dirs) > 6 else ''}")

        # 条件分岐: `status_payload.get("errors")` を満たす経路を評価する。

        if status_payload.get("errors"):
            lines.append("")
            lines.append(f"注意: エラー {len(status_payload.get('errors') or [])} 件（details: output/private/bepicolombo/more_psa_status.json）")

        ax.text(
            0.01,
            0.98,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=12,
        )

        fig.tight_layout()
        png_out = out_dir / "more_psa_status.png"
        fig.savefig(png_out, dpi=150)
        plt.close(fig)
    except Exception as e:
        meta.setdefault("errors", []).append({"stage": "plot", "error": str(e)})

    # Worklog

    try:
        worklog.append_event(
            {
                "event_type": "bepicolombo_more_psa_status",
                "argv": sys.argv,
                "inputs": {"base_url": base_url},
                "params": {"offline": bool(args.offline), "download_pdfs": bool(args.download_pdfs)},
                "outputs": {
                    "cache_meta_json": cache_meta_path,
                    "status_json": status_out,
                    "status_png": out_dir / "more_psa_status.png",
                },
            }
        )
    except Exception:
        pass

    print("Wrote:", status_out)
    print("Wrote:", out_dir / "more_psa_status.png")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
