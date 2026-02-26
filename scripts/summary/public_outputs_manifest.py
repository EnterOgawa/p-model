#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
public_outputs_manifest.py

Phase 8 / Step 8.7.16 向けの「公開成果物 manifest/hash 台帳」を生成する。

目的：
- `output/public/` 配下の成果物を、再現監査に使える hash 付き台帳として固定する。
- 主要コア成果物（CMB/SPARC/GW偏光/量子接続）の存在確認を同時に記録する。

出力（既定）：
- `output/public/summary/public_outputs_manifest.json`
- `output/public/summary/public_outputs_manifest_topics.csv`
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


_CORE_REQUIRED_REL: List[str] = [
    "output/public/cosmology/cosmology_cmb_acoustic_peak_reconstruction_falsification_pack.json",
    "output/public/cosmology/sparc_rotation_curve_pmodel_audit_metrics.json",
    "output/public/gw/gw_polarization_h1_l1_v1_network_audit.json",
    "output/public/quantum/quantum_connection_shared_kpi.json",
    "output/public/quantum/quantum_connection_bridge_table.json",
    "output/public/quantum/quantum_connection_born_ab_gate.json",
    "output/public/quantum/derivation_parameter_falsification_pack.json",
    "output/public/quantum/nuclear_condensed_cross_check_matrix.json",
    "output/public/quantum/weak_interaction_beta_decay_route_ab_audit.json",
]


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(_ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


# 関数: `_sha256` の入出力契約と処理意図を定義する。

def _sha256(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            # 条件分岐: `not b` を満たす経路を評価する。
            if not b:
                break

            h.update(b)

    return h.hexdigest()


# 関数: `_collect_public_files` の入出力契約と処理意図を定義する。

def _collect_public_files(*, public_root: Path, exclude: List[Path]) -> List[Path]:
    ex = {str(p.resolve()).lower() for p in exclude}
    files: List[Path] = []
    for p in public_root.rglob("*"):
        # 条件分岐: `not p.is_file()` を満たす経路を評価する。
        if not p.is_file():
            continue

        # 条件分岐: `str(p.resolve()).lower() in ex` を満たす経路を評価する。

        if str(p.resolve()).lower() in ex:
            continue

        files.append(p.resolve())

    files.sort(key=lambda x: _rel(x).lower())
    return files


# 関数: `_topic_from_rel` の入出力契約と処理意図を定義する。

def _topic_from_rel(rel_path: str) -> str:
    rel_norm = rel_path.replace("\\", "/")
    prefix = "output/public/"
    # 条件分岐: `not rel_norm.startswith(prefix)` を満たす経路を評価する。
    if not rel_norm.startswith(prefix):
        return "(unknown)"

    rem = rel_norm[len(prefix) :]
    # 条件分岐: `"/" not in rem` を満たす経路を評価する。
    if "/" not in rem:
        return rem or "(root)"

    return rem.split("/", 1)[0]


# 関数: `_write_topics_csv` の入出力契約と処理意図を定義する。

def _write_topics_csv(*, out_csv: Path, rows: List[Dict[str, Any]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topic", "file_count", "total_size_bytes"])
        for row in rows:
            w.writerow([row.get("topic", ""), int(row.get("file_count", 0)), int(row.get("total_size_bytes", 0))])


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 8 / Step 8.7.16: build hash manifest for output/public.")
    ap.add_argument(
        "--public-root",
        type=str,
        default=str(_ROOT / "output" / "public"),
        help="public output root (default: output/public)",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(_ROOT / "output" / "public" / "summary" / "public_outputs_manifest.json"),
        help="output JSON path (default: output/public/summary/public_outputs_manifest.json)",
    )
    ap.add_argument(
        "--out-topics-csv",
        type=str,
        default=str(_ROOT / "output" / "public" / "summary" / "public_outputs_manifest_topics.csv"),
        help="output CSV path (default: output/public/summary/public_outputs_manifest_topics.csv)",
    )
    ap.add_argument("--no-hash", action="store_true", help="skip per-file sha256 (faster)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    public_root = Path(str(args.public_root))
    # 条件分岐: `not public_root.is_absolute()` を満たす経路を評価する。
    if not public_root.is_absolute():
        public_root = (_ROOT / public_root).resolve()

    out_json = Path(str(args.out_json))
    # 条件分岐: `not out_json.is_absolute()` を満たす経路を評価する。
    if not out_json.is_absolute():
        out_json = (_ROOT / out_json).resolve()

    out_csv = Path(str(args.out_topics_csv))
    # 条件分岐: `not out_csv.is_absolute()` を満たす経路を評価する。
    if not out_csv.is_absolute():
        out_csv = (_ROOT / out_csv).resolve()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    compute_hash = not bool(args.no_hash)
    public_root_exists = public_root.exists()
    public_root_is_dir = public_root.is_dir()
    files = _collect_public_files(public_root=public_root, exclude=[out_json, out_csv]) if public_root_is_dir else []

    tree_digest = hashlib.sha256()
    rows: List[Dict[str, Any]] = []
    topic_map: Dict[str, Dict[str, Any]] = {}
    total_size = 0

    for p in files:
        st = p.stat()
        rel = _rel(p)
        size = int(st.st_size)
        mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        digest = _sha256(p) if compute_hash else None
        rows.append(
            {
                "path": rel,
                "size_bytes": size,
                "mtime_utc": mtime,
                "sha256": digest,
            }
        )
        tree_digest.update(f"{rel}\t{size}\t{digest or ''}\n".encode("utf-8"))
        total_size += size

        topic = _topic_from_rel(rel)
        s = topic_map.setdefault(topic, {"topic": topic, "file_count": 0, "total_size_bytes": 0})
        s["file_count"] = int(s["file_count"]) + 1
        s["total_size_bytes"] = int(s["total_size_bytes"]) + size

    topics = sorted(topic_map.values(), key=lambda x: str(x.get("topic") or ""))
    _write_topics_csv(out_csv=out_csv, rows=topics)

    required_core: List[Dict[str, Any]] = []
    missing_required_core: List[str] = []
    for rel in _CORE_REQUIRED_REL:
        p = (_ROOT / rel).resolve()
        exists = p.exists()
        item: Dict[str, Any] = {
            "path": rel,
            "exists": bool(exists),
            "size_bytes": int(p.stat().st_size) if exists else None,
            "sha256": _sha256(p) if (exists and compute_hash) else None,
        }
        required_core.append(item)
        # 条件分岐: `not exists` を満たす経路を評価する。
        if not exists:
            missing_required_core.append(rel)

    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "domain": "summary",
        "step": "8.7.16 (public outputs manifest)",
        "params": {
            "public_root": _rel(public_root),
            "compute_sha256": bool(compute_hash),
        },
        "public_root_status": {
            "exists": bool(public_root_exists),
            "is_directory": bool(public_root_is_dir),
        },
        "summary": {
            "file_count": int(len(rows)),
            "total_size_bytes": int(total_size),
            "topics_count": int(len(topics)),
            "public_tree_sha256": tree_digest.hexdigest(),
        },
        "topics": topics,
        "required_core_files": required_core,
        "ok": bool(public_root_is_dir and len(rows) > 0 and len(missing_required_core) == 0),
        "missing_required_core": sorted(missing_required_core),
        "repro_commands": {
            "public_outputs_manifest": "python -B scripts/summary/public_outputs_manifest.py",
            "public_outputs_manifest_fast": "python -B scripts/summary/public_outputs_manifest.py --no-hash",
            "release_manifest": "python -B scripts/summary/release_manifest.py",
        },
        "files": rows,
        "outputs": {
            "json": _rel(out_json),
            "topics_csv": _rel(out_csv),
        },
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    worklog.append_event(
        {
            "domain": "summary",
            "action": "public_outputs_manifest",
            "outputs": [out_json, out_csv],
            "params": {"compute_sha256": bool(compute_hash)},
            "result": {
                "ok": bool(payload["ok"]),
                "file_count": int(payload["summary"]["file_count"]),
                "missing_required_core": payload["missing_required_core"],
            },
        }
    )

    print("public_outputs_manifest:")
    print(f"- ok: {payload['ok']}")
    print(f"- out_json: {out_json}")
    print(f"- out_topics_csv: {out_csv}")
    print(f"- file_count: {payload['summary']['file_count']}")
    # 条件分岐: `payload["missing_required_core"]` を満たす経路を評価する。
    if payload["missing_required_core"]:
        print("[warn] missing required core files:")
        for rel in payload["missing_required_core"]:
            print(f"  - {rel}")

    return 0 if payload["ok"] else 1


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
