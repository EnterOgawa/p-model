#!/usr/bin/env python3
"""
BepiColombo / SPICE kernels (ESA PSA: bc_spice)

目的：
- BepiColombo（MPO/MORE）検証で必要になる幾何（軌道/座標系）を、一次ソース（ESA PSA）で確定する。
- PSAの `bc_spice/` 公開状況と、SPICE inventory（collection_spice_kernels_inventory_*.csv）を取得して
  オフライン再現できる形でキャッシュする。

出力：
- `output/bepicolombo/spice_psa_status.json`
- `output/bepicolombo/spice_psa_status.png`

キャッシュ：
- `data/bepicolombo/psa_spice/`（index html / bundle xml / inventory csv）
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
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _download(url: str, dst: Path, *, timeout_sec: float) -> Dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
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


def _http_status(url: str, *, timeout_sec: float) -> Tuple[Optional[int], Optional[str]]:
    req = Request(url, method="HEAD")
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            return int(getattr(resp, "status", 200)), None
    except HTTPError as e:
        return int(getattr(e, "code", 0) or 0), None
    except Exception as e:
        return None, str(e)


def _fetch_cached_html(url: str, cache_path: Path, *, offline: bool, timeout_sec: float) -> Tuple[Optional[str], Optional[str]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if offline:
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
    rows: List[Dict[str, str]] = []
    for ln in html_text.splitlines():
        if "<a href=" not in ln:
            continue
        m = re.search(r'href="([^"]+)"', ln)
        if not m:
            continue
        href = m.group(1)
        if href.startswith("?") or href.startswith("/") or href in ("../",):
            continue
        if "Parent Directory" in ln:
            continue

        is_dir = href.endswith("/")
        if is_dir and not include_dirs:
            continue
        if (not is_dir) and not include_files:
            continue

        m2 = re.search(r"</a></td><td[^>]*>([^<]+)</td><td[^>]*>([^<]+)</td>", ln)
        last_modified = (m2.group(1).strip() if m2 else "").replace("\xa0", " ")
        size = (m2.group(2).strip() if m2 else "").replace("\xa0", " ")
        rows.append({"name": href, "is_dir": "1" if is_dir else "0", "last_modified": last_modified, "size": size})

    seen: set[str] = set()
    out: List[Dict[str, str]] = []
    for r in rows:
        n = r.get("name") or ""
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(r)
    return out


def _pick_latest_versioned_name(names: List[str], *, prefix: str, suffix: str) -> Tuple[Optional[str], Optional[int]]:
    best_name = None
    best_v = None
    pat = re.compile(rf"^{re.escape(prefix)}v(\d+){re.escape(suffix)}$")
    for n in names:
        m = pat.match(n)
        if not m:
            continue
        v = int(m.group(1))
        if best_v is None or v > best_v:
            best_v = v
            best_name = n
    return best_name, best_v


def _read_lines(path: Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except Exception:
        try:
            return path.read_text(encoding="latin-1", errors="replace").splitlines()
        except Exception:
            return []


def _inventory_summary(lines: List[str]) -> Dict[str, Any]:
    type_counts: Dict[str, int] = {}
    craft_counts: Dict[str, int] = {"mpo": 0, "mmo": 0, "mtm": 0, "other": 0}
    examples: Dict[str, List[str]] = {"mpo": [], "mmo": [], "mtm": []}

    n_total = 0
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        parts = s.split(",", 1)
        if len(parts) != 2:
            continue
        lidvid = parts[1].strip()
        if not lidvid:
            continue

        n_total += 1
        base = lidvid.split("::", 1)[0]
        prod = base.rsplit(":", 1)[-1]
        prefix = prod.split("_", 1)[0]
        type_counts[prefix] = int(type_counts.get(prefix, 0) + 1)

        lower = f"_{prod.lower()}_"
        if "_mpo_" in lower:
            craft_counts["mpo"] += 1
            if len(examples["mpo"]) < 3:
                examples["mpo"].append(prod)
        elif "_mmo_" in lower:
            craft_counts["mmo"] += 1
            if len(examples["mmo"]) < 3:
                examples["mmo"].append(prod)
        elif "_mtm_" in lower:
            craft_counts["mtm"] += 1
            if len(examples["mtm"]) < 3:
                examples["mtm"].append(prod)
        else:
            craft_counts["other"] += 1

    # Sort for stable output
    type_counts_sorted = dict(sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return {
        "n_total": n_total,
        "type_counts": type_counts_sorted,
        "craft_counts": craft_counts,
        "examples": examples,
    }


def _set_japanese_font() -> None:
    try:
        import matplotlib

        candidates = [
            "Yu Gothic",
            "Yu Gothic UI",
            "Meiryo",
            "BIZ UDGothic",
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
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="BepiColombo SPICE (PSA bc_spice) status + inventory cache")
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://archives.esac.esa.int/psa/ftp/BepiColombo/bc_spice/",
        help="ESA PSA base URL for bc_spice (trailing slash required).",
    )
    parser.add_argument("--offline", action="store_true", help="Do not access the network; use local cache if present.")
    parser.add_argument("--timeout-sec", type=float, default=30.0, help="Network timeout seconds. Default: 30")
    args = parser.parse_args()

    root = _ROOT
    data_dir = root / "data" / "bepicolombo" / "psa_spice"
    out_dir = root / "output" / "private" / "bepicolombo"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = str(args.base_url).rstrip("/") + "/"
    spice_kernels_url = urljoin(base_url, "spice_kernels/")

    cache_meta_path = data_dir / "fetch_meta.json"
    prev_meta: Dict[str, Any] = {}
    if cache_meta_path.exists():
        try:
            prev_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        except Exception:
            prev_meta = {}

    meta: Dict[str, Any] = {
        "generated_utc": _utc_now_iso(),
        "base_url": base_url,
        "spice_kernels_url": spice_kernels_url,
        "offline": bool(args.offline),
        "expected": {},
        "listing": {},
        "downloads": [],
        "inventory": {},
    }

    if args.offline:
        meta["note"] = "offline mode; network was not used"
        if prev_meta:
            for k in ("expected", "listing", "downloads", "inventory"):
                if k in prev_meta:
                    meta[k] = prev_meta.get(k)
    else:
        # Expected directories
        exp: Dict[str, Any] = {}
        for d in ("document/", "miscellaneous/", "spice_kernels/"):
            st, err = _http_status(urljoin(base_url, d), timeout_sec=float(args.timeout_sec))
            exp[d] = {"status": st, "error": err}
        meta["expected"] = exp

        # Listings (cache html for offline)
        listing: Dict[str, Any] = {}
        base_html, base_err = _fetch_cached_html(
            base_url, data_dir / "base_index.html", offline=False, timeout_sec=float(args.timeout_sec)
        )
        if base_err:
            listing["base_error"] = base_err
            base_entries: List[Dict[str, str]] = []
        else:
            base_entries = _parse_apache_index_entries(base_html or "", include_dirs=True, include_files=True)
            listing["base_entries"] = base_entries

        sk_html, sk_err = _fetch_cached_html(
            spice_kernels_url, data_dir / "spice_kernels_index.html", offline=False, timeout_sec=float(args.timeout_sec)
        )
        if sk_err:
            listing["spice_kernels_error"] = sk_err
            sk_entries: List[Dict[str, str]] = []
        else:
            sk_entries = _parse_apache_index_entries(sk_html or "", include_dirs=True, include_files=True)
            listing["spice_kernels_entries"] = sk_entries

        meta["listing"] = listing

        # Pick latest bundle xml + latest inventory csv
        base_names = [str(e.get("name") or "") for e in base_entries if isinstance(e, dict)]
        bundle_name, bundle_v = _pick_latest_versioned_name(base_names, prefix="bundle_bc_spice_", suffix=".xml")
        if bundle_name and bundle_v is not None:
            try:
                meta["downloads"].append(
                    _download(urljoin(base_url, bundle_name), data_dir / bundle_name, timeout_sec=float(args.timeout_sec))
                )
                meta["bundle_latest"] = {"name": bundle_name, "version": bundle_v, "path": str((data_dir / bundle_name).resolve())}
            except Exception as e:
                meta.setdefault("errors", []).append({"stage": "download_bundle_xml", "error": str(e)})

        sk_names = [str(e.get("name") or "") for e in sk_entries if isinstance(e, dict)]
        inv_name, inv_v = _pick_latest_versioned_name(sk_names, prefix="collection_spice_kernels_inventory_", suffix=".csv")
        if inv_name and inv_v is not None:
            inv_path = data_dir / inv_name
            try:
                meta["downloads"].append(_download(urljoin(spice_kernels_url, inv_name), inv_path, timeout_sec=float(args.timeout_sec)))
                inv_lines = _read_lines(inv_path)
                meta["inventory"] = {
                    "name": inv_name,
                    "version": inv_v,
                    "path": str(inv_path.resolve()),
                    "sha256": _sha256(inv_path) if inv_path.exists() else None,
                    "summary": _inventory_summary(inv_lines),
                }
            except Exception as e:
                meta.setdefault("errors", []).append({"stage": "download_or_parse_inventory", "error": str(e)})

    # Persist cache meta for offline
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Output json (dashboard-friendly)
    status_out = out_dir / "spice_psa_status.json"
    inv = meta.get("inventory") if isinstance(meta.get("inventory"), dict) else {}
    inv_sum = inv.get("summary") if isinstance(inv.get("summary"), dict) else {}
    status_payload: Dict[str, Any] = {
        "generated_utc": meta.get("generated_utc"),
        "base_url": base_url,
        "spice_kernels_url": spice_kernels_url,
        "offline": bool(meta.get("offline")),
        "expected": meta.get("expected") or {},
        "listing": meta.get("listing") or {},
        "bundle_latest": meta.get("bundle_latest") or None,
        "inventory_latest": {
            "name": inv.get("name"),
            "version": inv.get("version"),
            "n_total": inv_sum.get("n_total"),
            "type_counts": inv_sum.get("type_counts"),
            "craft_counts": inv_sum.get("craft_counts"),
            "examples": inv_sum.get("examples"),
        }
        if inv
        else None,
        "errors": meta.get("errors") or [],
    }
    status_out.write_text(json.dumps(status_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # PNG status board
    try:
        import matplotlib.pyplot as plt  # type: ignore

        _set_japanese_font()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("off")

        lines: List[str] = []
        lines.append("BepiColombo（SPICE）：ESA PSA 公開状況（一次ソース確認）")
        lines.append(f"確認時刻(UTC): {status_payload.get('generated_utc')}")
        lines.append(f"対象: {base_url}")
        if status_payload.get("offline"):
            lines.append("モード: offline（ネットワーク未使用）")

        exp = status_payload.get("expected") or {}
        if exp:
            lines.append("")
            lines.append("公開ディレクトリ（HTTPステータス）:")
            for d in ("document/", "miscellaneous/", "spice_kernels/"):
                v = exp.get(d) or {}
                st = v.get("status")
                s = "unknown" if st is None else str(st)
                ok = "OK" if str(st) == "200" else "NG"
                lines.append(f"  - {d:<14} : {ok} ({s})")

        bl = status_payload.get("bundle_latest") or {}
        if isinstance(bl, dict) and bl.get("name"):
            lines.append("")
            lines.append(f"最新bundle: {bl.get('name')}")

        inv_latest = status_payload.get("inventory_latest") or {}
        if isinstance(inv_latest, dict) and inv_latest.get("name"):
            lines.append("")
            lines.append(f"最新inventory: {inv_latest.get('name')}（件数 {inv_latest.get('n_total')}）")

            cc = inv_latest.get("craft_counts") or {}
            if isinstance(cc, dict):
                lines.append(
                    f"  spacecraft内訳: MPO={cc.get('mpo')}, MMO={cc.get('mmo')}, MTM={cc.get('mtm')}, other={cc.get('other')}"
                )

            tc = inv_latest.get("type_counts") or {}
            if isinstance(tc, dict) and tc:
                top = list(tc.items())[:6]
                top_s = ", ".join([f"{k}={v}" for k, v in top])
                lines.append(f"  type上位: {top_s}")

        if status_payload.get("errors"):
            lines.append("")
            lines.append(f"注意: エラー {len(status_payload.get('errors') or [])} 件（details: output/bepicolombo/spice_psa_status.json）")

        ax.text(0.01, 0.98, "\n".join(lines), va="top", ha="left", fontsize=12)
        fig.tight_layout()

        png_out = out_dir / "spice_psa_status.png"
        fig.savefig(png_out, dpi=150)
        plt.close(fig)
    except Exception as e:
        # Keep non-fatal
        status_payload.setdefault("errors", []).append({"stage": "plot", "error": str(e)})

    try:
        worklog.append_event(
            {
                "event_type": "bepicolombo_spice_psa_status",
                "argv": sys.argv,
                "inputs": {"base_url": base_url},
                "params": {"offline": bool(args.offline)},
                "outputs": {"status_json": status_out, "status_png": out_dir / "spice_psa_status.png"},
            }
        )
    except Exception:
        pass

    print("Wrote:", status_out)
    print("Wrote:", out_dir / "spice_psa_status.png")


if __name__ == "__main__":
    main()
