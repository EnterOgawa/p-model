#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
release_manifest.py

Phase 8 / Step 8.3（データ・コード公開）向けの “公開マニフェスト” を生成する。

目的：
- 第三者がどのファイルを見ればよいか（成果物/入口/QC）を機械可読で固定する。
- 「公開前に何を確認したか」を `output/private/summary/work_history.jsonl` に残す。

出力：
- `output/private/summary/release_manifest.json`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402
from scripts.summary import env_fingerprint  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(_ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _sha256(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass(frozen=True)
class _FileInfo:
    path: str
    exists: bool
    size_bytes: Optional[int]
    mtime_utc: Optional[str]
    sha256: Optional[str]


def _file_info(path: Path, *, compute_hash: bool) -> _FileInfo:
    if not path.exists():
        return _FileInfo(path=_rel(path), exists=False, size_bytes=None, mtime_utc=None, sha256=None)
    st = path.stat()
    mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    digest = _sha256(path) if compute_hash else None
    return _FileInfo(path=_rel(path), exists=True, size_bytes=int(st.st_size), mtime_utc=mtime, sha256=digest)


def _default_manifest_paths() -> Dict[str, List[Path]]:
    """
    Keep this list small and stable: publish artifacts + QC + pointers.
    """
    return {
        "local_docs": [
            _ROOT / "doc" / "STATUS.md",
            _ROOT / "doc" / "ROADMAP.md",
            _ROOT / "doc" / "PRIMARY_SOURCES.md",
            _ROOT / "doc" / "WORK_HISTORY_RECENT.md",
            _ROOT / "doc" / "AI_CONTEXT_MIN.json",
            _ROOT / "doc" / "PUBLISHING.md",
            _ROOT / "doc" / "P_model_handoff.md",
            _ROOT / "doc" / "paper" / "README.md",
            _ROOT / "doc" / "paper" / "00_outline.md",
            _ROOT / "doc" / "paper" / "05_definitions.md",
            _ROOT / "doc" / "paper" / "06_uncertainty.md",
            _ROOT / "doc" / "paper" / "07_llr_appendix.md",
            _ROOT / "doc" / "paper" / "10_manuscript.md",
            _ROOT / "doc" / "paper" / "13_part4_verification.md",
            _ROOT / "doc" / "paper" / "20_data_sources.md",
            _ROOT / "doc" / "paper" / "01_figures_index.md",
            _ROOT / "doc" / "paper" / "30_references.md",
            _ROOT / "doc" / "paper" / "40_publication_plan.md",
        ],
        "entrypoints": [
            _ROOT / "scripts" / "summary" / "build_materials.bat",
            _ROOT / "scripts" / "summary" / "paper_qc.py",
            _ROOT / "scripts" / "summary" / "paper_build.py",
            _ROOT / "scripts" / "summary" / "public_dashboard.py",
            _ROOT / "scripts" / "summary" / "env_fingerprint.py",
            _ROOT / "scripts" / "summary" / "release_manifest.py",
            _ROOT / "scripts" / "summary" / "release_bundle.py",
            _ROOT / "scripts" / "summary" / "run_all.py",
        ],
        "publish_outputs": [
            _ROOT / "output" / "private" / "summary" / "pmodel_paper.html",
            _ROOT / "output" / "private" / "summary" / "pmodel_paper_part2_astrophysics.html",
            _ROOT / "output" / "private" / "summary" / "pmodel_paper_part3_quantum.html",
            _ROOT / "output" / "private" / "summary" / "pmodel_paper_part4_verification.html",
            _ROOT / "output" / "private" / "summary" / "pmodel_public_report.html",
            _ROOT / "output" / "private" / "summary" / "paper_qc.json",
            _ROOT / "output" / "private" / "summary" / "env_fingerprint.json",
            _ROOT / "output" / "private" / "summary" / "paper_table1_results.json",
            _ROOT / "output" / "private" / "summary" / "paper_table1_results.csv",
            _ROOT / "output" / "private" / "summary" / "paper_table1_results.md",
            _ROOT / "output" / "private" / "summary" / "paper_table1_quantum_results.json",
            _ROOT / "output" / "private" / "summary" / "paper_table1_quantum_results.csv",
            _ROOT / "output" / "private" / "summary" / "paper_table1_quantum_results.md",
        ],
        # Optional: generated bundles for sharing (not required for manifest ok=true).
        "release_bundles": [
            _ROOT / "output" / "private" / "summary" / "pmodel_release_bundle_paper.zip",
            _ROOT / "output" / "private" / "summary" / "pmodel_release_bundle_repro.zip",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 8 / Step 8.3: generate release manifest JSON.")
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "release_manifest.json"),
        help="output path (default: output/private/summary/release_manifest.json)",
    )
    ap.add_argument("--no-hash", action="store_true", help="skip sha256 (faster)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    out_json = Path(str(args.out_json))
    if not out_json.is_absolute():
        out_json = (_ROOT / out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    groups = _default_manifest_paths()
    compute_hash = not bool(args.no_hash)

    # Always generate a small environment fingerprint for reproducibility.
    try:
        env_fingerprint.main(["--out-json", str(_ROOT / "output" / "private" / "summary" / "env_fingerprint.json")])
    except Exception:
        # Best-effort only; manifest will report missing files if generation failed.
        pass

    files: Dict[str, List[Dict[str, Any]]] = {}
    missing_required: List[str] = []

    for group_name, paths in groups.items():
        items: List[Dict[str, Any]] = []
        for p in paths:
            info = _file_info(p, compute_hash=compute_hash)
            items.append(info.__dict__)
            if (group_name in {"entrypoints", "publish_outputs"}) and (not info.exists):
                missing_required.append(info.path)
        files[group_name] = items

    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "domain": "summary",
        "step": "8.3 (release manifest)",
        "env": {"os_name": os.name, "platform": sys.platform, "python": sys.version.split()[0]},
        "params": {"compute_sha256": bool(compute_hash)},
        "ok": (len(missing_required) == 0),
        "missing_required": sorted(set(missing_required)),
        "files": files,
        "repro_commands": {
            "build_quick": r"cmd /c scripts\summary\build_materials.bat quick",
            "build_quick_nodocx": r"cmd /c scripts\summary\build_materials.bat quick-nodocx",
            "build_part4_verification_html": "python -B scripts/summary/paper_build.py --profile part4_verification --mode publish --outdir output/private/summary --skip-docx --skip-tables",
            "paper_qc": "python -B scripts/summary/paper_qc.py",
            "release_manifest_fast": "python -B scripts/summary/release_manifest.py --no-hash",
            "release_bundle_paper": "python -B scripts/summary/release_bundle.py --mode paper --no-hash",
            "release_bundle_repro": "python -B scripts/summary/release_bundle.py --mode repro --no-hash",
            "run_all_offline": "python -B scripts/summary/run_all.py --offline --jobs 2",
        },
        "outputs": {"json": _rel(out_json)},
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    worklog.append_event(
        {
            "domain": "summary",
            "action": "release_manifest",
            "outputs": [out_json],
            "result": {"ok": bool(payload["ok"]), "missing_required": payload["missing_required"]},
            "params": payload["params"],
        }
    )

    print("release_manifest:")
    print(f"- ok: {payload['ok']}")
    print(f"- out: {out_json}")
    if payload["missing_required"]:
        print("[warn] missing required:")
        for p in payload["missing_required"][:40]:
            print(f"  - {p}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
