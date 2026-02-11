#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
release_bundle.py

Phase 8 / Step 8.3（データ・コード公開）向けの “配布バンドル（zip）” を生成する。

目的：
- 第三者へ渡す最小パッケージを固定し、同じ成果物（HTML）を確認できる状態を作る。
- 大きすぎる raw データは同梱せず、一次ソース（doc/PRIMARY_SOURCES.md）と取得スクリプトで再現する。

出力（既定）：
- paper: `output/summary/pmodel_release_bundle_paper.zip`（読み物＋最小ドキュメント）
- repro: `output/summary/pmodel_release_bundle_repro.zip`（コード＋docs＋入口＋成果物（summary））
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import release_manifest, worklog  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(_ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


@dataclass(frozen=True)
class _BundleSpec:
    name: str
    include_docs: bool
    include_entrypoints: bool
    include_publish_outputs: bool
    include_scripts_tree: bool


_BUNDLE_SPECS: Dict[str, _BundleSpec] = {
    "paper": _BundleSpec(
        name="paper",
        include_docs=True,
        include_entrypoints=False,
        include_publish_outputs=True,
        include_scripts_tree=False,
    ),
    "repro": _BundleSpec(
        name="repro",
        include_docs=True,
        include_entrypoints=True,
        include_publish_outputs=True,
        include_scripts_tree=True,
    ),
}


def _load_or_build_manifest(*, compute_hash: bool) -> Dict[str, Any]:
    manifest_path = _ROOT / "output" / "summary" / "release_manifest.json"
    if (not manifest_path.exists()) or (manifest_path.stat().st_size <= 10):
        argv = [] if compute_hash else ["--no-hash"]
        rc = release_manifest.main(argv)
        if rc != 0:
            raise RuntimeError("release_manifest failed")
    payload = _read_json(manifest_path)
    if not payload.get("ok", False):
        missing = payload.get("missing_required") or []
        raise RuntimeError(f"release_manifest ok=false (missing_required={missing})")
    return payload


def _manifest_group_paths(payload: Dict[str, Any], group: str) -> List[Path]:
    files = payload.get("files") or {}
    items = files.get(group) or []
    out: List[Path] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        rel = str(it.get("path") or "")
        if not rel:
            continue
        if it.get("exists") is False:
            continue
        out.append((_ROOT / rel).resolve())
    return out


def _iter_repo_files(root: Path, *, base: Path, patterns: Sequence[str]) -> Iterable[Path]:
    for pat in patterns:
        for p in base.rglob(pat):
            if p.is_file():
                yield p


def _collect_files(spec: _BundleSpec, payload: Dict[str, Any]) -> List[Path]:
    files: List[Path] = []
    if spec.include_docs:
        files += _manifest_group_paths(payload, "docs")
    if spec.include_entrypoints:
        files += _manifest_group_paths(payload, "entrypoints")
    if spec.include_publish_outputs:
        files += _manifest_group_paths(payload, "publish_outputs")

    if spec.include_scripts_tree:
        scripts_dir = _ROOT / "scripts"
        # Include only small, text-like sources (no venv/data/output).
        files += list(
            _iter_repo_files(
                _ROOT,
                base=scripts_dir,
                patterns=["*.py", "*.md", "*.txt", "*.json", "*.yml", "*.yaml", "*.sh", "*.bat", "*.ps1", "*.cpp", "*.h"],
            )
        )

    # De-dup while keeping stable order.
    seen: Set[str] = set()
    uniq: List[Path] = []
    for p in files:
        key = str(p.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _write_bundle_zip(*, out_zip: Path, spec: _BundleSpec, files: Sequence[Path], payload: Dict[str, Any]) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_zip.with_suffix(out_zip.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    manifest_path = (_ROOT / "output" / "summary" / "release_manifest.json").resolve()
    file_set = {str(p.resolve()).lower() for p in files}
    if manifest_path.exists() and (str(manifest_path).lower() not in file_set):
        files = list(files) + [manifest_path]

    sizes: List[Dict[str, Any]] = []
    total_size = 0
    for p in files:
        try:
            st = p.stat()
            sz = int(st.st_size)
        except Exception:
            sz = 0
        total_size += sz
        sizes.append({"path": _rel(p), "size_bytes": sz})
    sizes_sorted = sorted(sizes, key=lambda x: int(x.get("size_bytes") or 0), reverse=True)

    meta = {
        "generated_utc": _utc_now(),
        "domain": "summary",
        "step": "8.3 (release bundle)",
        "mode": spec.name,
        "manifest": "output/summary/release_manifest.json",
        "file_count": int(len(files)),
        "files_total_size_bytes": int(total_size),
        "largest_files": sizes_sorted[:20],
        "notes": [
            "This bundle intentionally excludes raw data under data/ (fetch via PRIMARY_SOURCES + scripts).",
            "Paper HTML embeds PNG as base64; rebuilding requires running build_materials (quick-nodocx).",
        ],
    }

    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("BUNDLE_INFO.json", json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        for p in files:
            arc = _rel(p)
            if not arc:
                continue
            zf.write(p, arcname=arc)

    if out_zip.exists():
        out_zip.unlink()
    tmp.replace(out_zip)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 8 / Step 8.3: build release zip bundles.")
    ap.add_argument("--mode", choices=sorted(_BUNDLE_SPECS.keys()), default="paper", help="bundle mode (default: paper)")
    ap.add_argument(
        "--out-zip",
        type=str,
        default=None,
        help="output zip path (default: output/summary/pmodel_release_bundle_<mode>.zip)",
    )
    ap.add_argument("--no-hash", action="store_true", help="skip sha256 in release_manifest (faster)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    spec = _BUNDLE_SPECS[str(args.mode)]
    out_zip = Path(args.out_zip) if args.out_zip else (_ROOT / "output" / "summary" / f"pmodel_release_bundle_{spec.name}.zip")
    if not out_zip.is_absolute():
        out_zip = (_ROOT / out_zip).resolve()

    payload = _load_or_build_manifest(compute_hash=(not bool(args.no_hash)))
    files = _collect_files(spec, payload)
    _write_bundle_zip(out_zip=out_zip, spec=spec, files=files, payload=payload)
    zip_size = int(out_zip.stat().st_size) if out_zip.exists() else None

    try:
        worklog.append_event(
            {
                "domain": "summary",
                "event_type": "release_bundle",
                "mode": spec.name,
                "params": {"no_hash": bool(args.no_hash)},
                "outputs": {"zip": out_zip},
                "result": {"file_count": int(len(files))},
            }
        )
    except Exception:
        pass

    print("release_bundle:")
    print(f"- mode: {spec.name}")
    print(f"- out : {out_zip}")
    print(f"- files: {len(files)}")
    if zip_size is not None:
        print(f"- zip_size_bytes: {zip_size}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
