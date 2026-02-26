#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_sparc.py

Phase 6 / Step 6.5（SPARC：RAR/BTFR）:
SPARC database（Lelli, McGaugh, Schombert）から一次データを取得し、
offline 再現できる形で raw cache + sha256 manifest を固定する。

注意:
- http は環境によって接続拒否されることがあるため、既定は https を用いる。
- 本スクリプトは「取得I/Fの固定」が目的で、RAR/BTFR の再構築は別スクリプトで行う。

出力（固定）:
- raw cache: data/cosmology/sparc/raw/<filename>
- manifest:  data/cosmology/sparc/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def _download(url: str, *, dst: Path, force: bool) -> Dict[str, Any]:
    # 条件分岐: `requests is None` を満たす経路を評価する。
    if requests is None:
        raise RuntimeError("requests is required for online fetch")

    dst.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `dst.exists() and not force` を満たす経路を評価する。
    if dst.exists() and not force:
        return {"status": "skipped_exists", "path": _rel(dst), "bytes": int(dst.stat().st_size), "sha256": _sha256_file(dst), "url": url}

    r = requests.get(url, timeout=_REQ_TIMEOUT, stream=True)
    r.raise_for_status()
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

    tmp.replace(dst)
    return {"status": "downloaded", "path": _rel(dst), "bytes": int(n), "sha256": h.hexdigest(), "url": url}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="https://astroweb.case.edu/SPARC/", help="SPARC base URL (default: https://astroweb.case.edu/SPARC/)")
    p.add_argument("--download-missing", action="store_true", help="Download missing files (online).")
    p.add_argument("--force", action="store_true", help="Force re-download even if local file exists.")
    p.add_argument("--offline", action="store_true", help="Offline mode: do not access the network.")
    p.add_argument(
        "--file",
        action="append",
        default=[],
        help="File to fetch (repeatable). Default is a minimal set for RAR/BTFR reconstruction.",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    base_url = str(args.base_url).strip()
    # 条件分岐: `not base_url.endswith("/")` を満たす経路を評価する。
    if not base_url.endswith("/"):
        base_url += "/"

    default_files = [
        "SPARC_Lelli2016c.mrt",
        "MassModels_Lelli2016c.mrt",
        "Rotmod_LTG.zip",
        "RAR.mrt",
        "RARbins.mrt",
        "BTFR_Lelli2019.mrt",
    ]
    files = [str(x).strip() for x in (args.file or []) if str(x).strip()] or default_files

    # 条件分岐: `args.offline and args.download_missing` を満たす経路を評価する。
    if args.offline and args.download_missing:
        p.error("--offline and --download-missing cannot be used together")

    # 条件分岐: `not args.offline and requests is None` を満たす経路を評価する。

    if not args.offline and requests is None:
        raise RuntimeError("requests is required for online fetch")

    out_root = _ROOT / "data" / "cosmology" / "sparc"
    raw_dir = out_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = out_root / "manifest.json"

    prev = _read_json(out_manifest)
    history: List[Dict[str, Any]] = list(prev.get("history", [])) if isinstance(prev.get("history", []), list) else []

    run: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "inputs": {
            "base_url": base_url,
            "files": files,
            "offline": bool(args.offline),
            "download_missing": bool(args.download_missing),
            "force": bool(args.force),
        },
        "results": {"downloaded": [], "skipped": [], "errors": []},
    }

    file_index: Dict[str, Any] = dict(prev.get("files", {})) if isinstance(prev.get("files", {}), dict) else {}

    for name in files:
        url = base_url + name
        dst = raw_dir / name
        # 条件分岐: `args.offline` を満たす経路を評価する。
        if args.offline:
            # 条件分岐: `not dst.exists()` を満たす経路を評価する。
            if not dst.exists():
                run["results"]["skipped"].append({"status": "skipped_missing", "path": _rel(dst), "url": url})
                continue

            res = {"status": "skipped_exists", "path": _rel(dst), "bytes": int(dst.stat().st_size), "sha256": _sha256_file(dst), "url": url}
        else:
            # 条件分岐: `not args.download_missing and not dst.exists()` を満たす経路を評価する。
            if not args.download_missing and not dst.exists():
                run["results"]["skipped"].append({"status": "skipped_missing", "path": _rel(dst), "url": url})
                continue

            try:
                res = _download(url, dst=dst, force=bool(args.force))
            except Exception as e:
                run["results"]["errors"].append({"status": "error", "path": _rel(dst), "url": url, "error": str(e)})
                continue

        # 条件分岐: `res["status"] == "downloaded"` を満たす経路を評価する。

        if res["status"] == "downloaded":
            run["results"]["downloaded"].append(res)
        else:
            run["results"]["skipped"].append(res)

        file_index[name] = {
            "url": url,
            "raw_path": _rel(dst),
            "raw_bytes": int(res.get("bytes") or 0),
            "raw_sha256": str(res.get("sha256") or ""),
        }

    history.append(run)
    manifest = {
        "generated_utc": str(prev.get("generated_utc") or run["generated_utc"]),
        "last_run_utc": run["generated_utc"],
        "source_base_url": base_url,
        "files": file_index,
        "history": history[-20:],  # keep recent runs
    }
    _write_json(out_manifest, manifest)

    try:
        worklog.append_event(
            "cosmology.sparc_fetch",
            {
                "manifest": _rel(out_manifest),
                "raw_dir": _rel(raw_dir),
                "downloaded": len(run["results"]["downloaded"]),
                "skipped": len(run["results"]["skipped"]),
                "errors": len(run["results"]["errors"]),
                "base_url": base_url,
            },
        )
    except Exception:
        pass

    print(json.dumps({"manifest": _rel(out_manifest), "raw_dir": _rel(raw_dir)}, ensure_ascii=False))
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())

