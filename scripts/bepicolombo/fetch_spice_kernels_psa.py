#!/usr/bin/env python3
"""
fetch_spice_kernels_psa.py

BepiColombo の幾何計算（太陽会合/Shapiro 等）に必要な SPICE カーネルを
ESA PSA（bc_spice）から取得して `data/bepicolombo/kernels/psa/` にキャッシュする。

目的:
  - BepiColombo（MPO/MORE）の検証で、探査機・地球・太陽の幾何を一次ソースで再現できるようにする
  - 2回目以降は data/ 配下のカーネルでオフライン再現する

既定（kernel-set=minimal）で取得するもの:
  - mk: 最新の meta-kernel（bc_v###.tm）
  - lsk: meta-kernel が参照する *.tls（例: naif0012.tls）
  - spk: meta-kernel が参照する惑星暦（例: de432s.bsp）
  - spk: meta-kernel が参照する MPO 軌道（例: bc_mpo_fcp_*.bsp）

出力（固定）:
  - data/bepicolombo/kernels/psa/kernels_meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


@dataclass(frozen=True)
class KernelRef:
    rel_path: str  # relative to spice_kernels/
    url: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return _ROOT


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def _download(url: str, dst: Path, *, force: bool, timeout_sec: float) -> Dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `dst.exists() and dst.stat().st_size > 0 and not force` を満たす経路を評価する。
    if dst.exists() and dst.stat().st_size > 0 and not force:
        return {"url": url, "path": str(dst), "downloaded": False, "bytes": int(dst.stat().st_size), "sha256": _sha256(dst)}

    tmp = dst.with_suffix(dst.suffix + ".part")
    req = Request(url, method="GET")
    with urlopen(req, timeout=timeout_sec) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)

    tmp.replace(dst)
    return {"url": url, "path": str(dst), "downloaded": True, "bytes": int(dst.stat().st_size), "sha256": _sha256(dst)}


def _fetch_text(url: str, *, timeout_sec: float) -> str:
    req = Request(url, method="GET")
    with urlopen(req, timeout=timeout_sec) as r:
        b = r.read()

    return b.decode("utf-8", errors="replace")


def _parse_apache_index_filenames(html_text: str) -> List[str]:
    out: List[str] = []
    for ln in html_text.splitlines():
        # 条件分岐: `"<a href=" not in ln` を満たす経路を評価する。
        if "<a href=" not in ln:
            continue

        m = re.search(r'href="([^"]+)"', ln)
        # 条件分岐: `not m` を満たす経路を評価する。
        if not m:
            continue

        href = m.group(1)
        # 条件分岐: `href.startswith("?") or href.startswith("/") or href in ("../",)` を満たす経路を評価する。
        if href.startswith("?") or href.startswith("/") or href in ("../",):
            continue

        # 条件分岐: `href.endswith("/")` を満たす経路を評価する。

        if href.endswith("/"):
            continue

        # 条件分岐: `"Parent Directory" in ln` を満たす経路を評価する。

        if "Parent Directory" in ln:
            continue

        out.append(href)

    seen: set[str] = set()
    uniq: List[str] = []
    for n in out:
        # 条件分岐: `n in seen` を満たす経路を評価する。
        if n in seen:
            continue

        seen.add(n)
        uniq.append(n)

    return uniq


def _pick_latest_meta_kernel(names: Iterable[str]) -> Tuple[Optional[str], Optional[int]]:
    best_name = None
    best_v = None
    pat = re.compile(r"^bc_v(\d+)\.tm$")
    for n in names:
        m = pat.match(n)
        # 条件分岐: `not m` を満たす経路を評価する。
        if not m:
            continue

        v = int(m.group(1))
        # 条件分岐: `best_v is None or v > best_v` を満たす経路を評価する。
        if best_v is None or v > best_v:
            best_v = v
            best_name = n

    return best_name, best_v


def _extract_kernel_paths_from_meta(meta_text: str) -> List[str]:
    """
    Extract '$KERNELS/<dir>/<file>' entries from bc_v###.tm.
    Return list of '<dir>/<file>' (relative to spice_kernels/).
    """
    paths: List[str] = []
    # Meta-kernel lists kernels like: '$KERNELS/lsk/naif0012.tls'
    # In regex, `$` is special, so escape it as `\$` (do NOT match a literal backslash).
    for m in re.finditer(r"'\$KERNELS/([^']+)'", meta_text):
        p = m.group(1).strip()
        # 条件分岐: `not p or p.startswith("../") or p.startswith("/") or p.endswith("/")` を満たす経路を評価する。
        if not p or p.startswith("../") or p.startswith("/") or p.endswith("/"):
            continue

        paths.append(p)
    # stable unique

    seen: set[str] = set()
    out: List[str] = []
    for p in paths:
        # 条件分岐: `p in seen` を満たす経路を評価する。
        if p in seen:
            continue

        seen.add(p)
        out.append(p)

    return out


def _select_minimal_kernel_paths(meta_paths: List[str]) -> List[str]:
    """
    Keep only kernels needed to compute Earth/Sun/MPO geometry and Shapiro y(t).
    Derived from meta-kernel (so version stays consistent).
    """
    lsk = [p for p in meta_paths if p.startswith("lsk/") and p.endswith(".tls")]
    de_spk = [p for p in meta_paths if p.startswith("spk/") and re.search(r"^spk/de\d+.*\.bsp$", p)]
    mpo_spk = [p for p in meta_paths if p.startswith("spk/") and "bc_mpo_fcp_" in p and p.endswith(".bsp")]

    # If multiple, keep the last one (meta-kernel order: most recent coverage last).
    out: List[str] = []
    # 条件分岐: `lsk` を満たす経路を評価する。
    if lsk:
        out.append(lsk[-1])

    # 条件分岐: `de_spk` を満たす経路を評価する。

    if de_spk:
        out.append(de_spk[-1])

    # 条件分岐: `mpo_spk` を満たす経路を評価する。

    if mpo_spk:
        out.append(mpo_spk[-1])

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch minimal BepiColombo SPICE kernels from ESA PSA (bc_spice).")
    ap.add_argument(
        "--base-url",
        type=str,
        default="https://archives.esac.esa.int/psa/ftp/BepiColombo/bc_spice/spice_kernels/",
        help="Base URL to bc_spice spice_kernels/ (trailing slash required).",
    )
    ap.add_argument(
        "--kernel-set",
        choices=["minimal"],
        default="minimal",
        help="Which kernels to fetch. Default: minimal",
    )
    ap.add_argument("--offline", action="store_true", help="Do not use network; only succeed if cache is complete.")
    ap.add_argument("--force", action="store_true", help="Re-download and overwrite cached kernels.")
    ap.add_argument("--timeout-sec", type=float, default=180.0, help="Network timeout seconds. Default: 180")
    args = ap.parse_args()

    root = _repo_root()
    out_dir = root / "data" / "bepicolombo" / "kernels" / "psa"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "kernels_meta.json"

    base_url = str(args.base_url).rstrip("/") + "/"
    mk_url = urljoin(base_url, "mk/")

    # 条件分岐: `args.offline` を満たす経路を評価する。
    if args.offline:
        # 条件分岐: `not meta_path.exists()` を満たす経路を評価する。
        if not meta_path.exists():
            print(f"[err] offline mode: missing meta: {meta_path}")
            return 2

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = meta.get("expected_files") if isinstance(meta, dict) else None
        # 条件分岐: `not isinstance(expected, list) or not expected` を満たす経路を評価する。
        if not isinstance(expected, list) or not expected:
            print(f"[err] offline mode: meta missing expected_files: {meta_path}")
            return 2

        missing = [p for p in expected if not (out_dir / p).exists()]
        # 条件分岐: `missing` を満たす経路を評価する。
        if missing:
            print("[err] offline and missing kernels:")
            for m in missing:
                print("  -", m)

            return 2

        print(f"[ok] offline: {out_dir} ({len(expected)} files)")
        return 0

    # Pick latest meta-kernel

    mk_index = _fetch_text(mk_url, timeout_sec=float(args.timeout_sec))
    mk_names = _parse_apache_index_filenames(mk_index)
    mk_name, mk_v = _pick_latest_meta_kernel(mk_names)
    # 条件分岐: `not mk_name or mk_v is None` を満たす経路を評価する。
    if not mk_name or mk_v is None:
        raise RuntimeError(f"Failed to find bc_v###.tm in: {mk_url}")

    mk_rel = f"mk/{mk_name}"
    mk_local = out_dir / mk_rel
    mk_dl = _download(urljoin(mk_url, mk_name), mk_local, force=bool(args.force), timeout_sec=float(args.timeout_sec))
    mk_text = mk_local.read_text(encoding="utf-8", errors="replace")

    meta_kernel_paths = _extract_kernel_paths_from_meta(mk_text)
    # 条件分岐: `args.kernel_set == "minimal"` を満たす経路を評価する。
    if args.kernel_set == "minimal":
        selected_paths = _select_minimal_kernel_paths(meta_kernel_paths)
    else:
        selected_paths = []

    kernel_refs: List[KernelRef] = [KernelRef(rel_path=mk_rel, url=urljoin(mk_url, mk_name))]
    for rel in selected_paths:
        kernel_refs.append(KernelRef(rel_path=rel, url=urljoin(base_url, rel)))

    downloaded: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for k in kernel_refs:
        try:
            res = _download(k.url, out_dir / k.rel_path, force=bool(args.force), timeout_sec=float(args.timeout_sec))
            downloaded.append(res)
        except HTTPError as e:
            errors.append({"rel_path": k.rel_path, "url": k.url, "error": f"HTTP {getattr(e, 'code', '')}"})
        except Exception as e:
            errors.append({"rel_path": k.rel_path, "url": k.url, "error": str(e)})

    meta: Dict[str, Any] = {
        "generated_utc": _utc_now_iso(),
        "source": "ESA PSA bc_spice",
        "base_url": base_url,
        "kernel_set": args.kernel_set,
        "meta_kernel": {"name": mk_name, "version": mk_v, "download": mk_dl},
        "selected_paths": selected_paths,
        "expected_files": [k.rel_path for k in kernel_refs],
        "downloads": downloaded,
        "errors": errors,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] meta: {meta_path}")
    for k in kernel_refs:
        p = out_dir / k.rel_path
        ok = p.exists() and p.stat().st_size > 0
        print(f"[{'ok' if ok else '??'}] {k.rel_path} ({p.stat().st_size if p.exists() else 'missing'})")

    try:
        worklog.append_event(
            {
                "event_type": "bepicolombo_fetch_spice_kernels_psa",
                "argv": sys.argv,
                "inputs": {"base_url": base_url},
                "params": {"kernel_set": args.kernel_set, "offline": bool(args.offline), "force": bool(args.force)},
                "outputs": {"kernels_meta_json": meta_path},
            }
        )
    except Exception:
        pass

    return 0 if not errors else 1


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
