#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
public_outputs_manifest_continuity.py

Phase 8 / Step 8.7.16 follow-up（release snapshot 同期 + hash continuity）向けの
運用監査を固定する。

目的：
- `output/public/summary/public_outputs_manifest.json` を基準に、直前スナップショットとの差分を機械判定する。
- release/tag 参照（latest tag 既定）に紐づく compact snapshot を保存し、連続性監査を残す。

出力（既定）：
- `output/private/summary/public_outputs_manifest_continuity.json`
- `output/private/summary/public_outputs_manifest_continuity_topics.csv`
- `output/private/summary/public_manifest_snapshots/public_outputs_manifest_snapshot_<label>.json`
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import public_outputs_manifest  # noqa: E402
from scripts.summary import worklog  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(_ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _git(*args: str) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception:
        return (1, "")

    return (int(proc.returncode), (proc.stdout or "").strip())


def _latest_tag() -> str:
    rc, out = _git("tag", "--sort=-creatordate")
    # 条件分岐: `rc != 0` を満たす経路を評価する。
    if rc != 0:
        return ""

    tags = [line.strip() for line in out.splitlines() if line.strip()]
    return tags[0] if tags else ""


def _head_short() -> str:
    rc, out = _git("rev-parse", "--short", "HEAD")
    # 条件分岐: `rc != 0 or not out` を満たす経路を評価する。
    if rc != 0 or not out:
        return "no-git"

    return out


def _tag_exists(tag: str) -> bool:
    # 条件分岐: `not tag` を満たす経路を評価する。
    if not tag:
        return False

    rc, _ = _git("rev-parse", "-q", "--verify", f"refs/tags/{tag}")
    return rc == 0


def _sanitize_label(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    s = s.strip("._-")
    return s or "snapshot"


def _topic_map(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        topic = str(item.get("topic") or "")
        # 条件分岐: `not topic` を満たす経路を評価する。
        if not topic:
            continue

        out[topic] = {
            "topic": topic,
            "file_count": int(item.get("file_count") or 0),
            "total_size_bytes": int(item.get("total_size_bytes") or 0),
        }

    return out


def _required_core_map(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        path = str(item.get("path") or "")
        # 条件分岐: `not path` を満たす経路を評価する。
        if not path:
            continue

        out[path] = {
            "path": path,
            "exists": bool(item.get("exists")),
            "size_bytes": int(item.get("size_bytes") or 0) if item.get("size_bytes") is not None else None,
            "sha256": item.get("sha256"),
        }

    return out


def _build_topic_deltas(
    prev_topics: List[Dict[str, Any]],
    curr_topics: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pmap = _topic_map(prev_topics)
    cmap = _topic_map(curr_topics)
    all_topics = sorted(set(pmap.keys()) | set(cmap.keys()))
    rows: List[Dict[str, Any]] = []
    for topic in all_topics:
        p = pmap.get(topic)
        c = cmap.get(topic)
        prev_count = int(p["file_count"]) if p else 0
        curr_count = int(c["file_count"]) if c else 0
        prev_size = int(p["total_size_bytes"]) if p else 0
        curr_size = int(c["total_size_bytes"]) if c else 0
        # 条件分岐: `p is None` を満たす経路を評価する。
        if p is None:
            status = "added"
        # 条件分岐: 前段条件が不成立で、`c is None` を追加評価する。
        elif c is None:
            status = "removed"
        # 条件分岐: 前段条件が不成立で、`(prev_count != curr_count) or (prev_size != curr_size)` を追加評価する。
        elif (prev_count != curr_count) or (prev_size != curr_size):
            status = "changed"
        else:
            status = "unchanged"

        rows.append(
            {
                "topic": topic,
                "status": status,
                "previous_file_count": prev_count,
                "current_file_count": curr_count,
                "delta_file_count": curr_count - prev_count,
                "previous_total_size_bytes": prev_size,
                "current_total_size_bytes": curr_size,
                "delta_total_size_bytes": curr_size - prev_size,
            }
        )

    return rows


def _build_required_core_deltas(
    prev_items: List[Dict[str, Any]],
    curr_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pmap = _required_core_map(prev_items)
    cmap = _required_core_map(curr_items)
    all_paths = sorted(set(pmap.keys()) | set(cmap.keys()))
    rows: List[Dict[str, Any]] = []
    for path in all_paths:
        p = pmap.get(path)
        c = cmap.get(path)
        prev_exists = bool(p["exists"]) if p else False
        curr_exists = bool(c["exists"]) if c else False
        prev_sha = str(p.get("sha256") or "") if p else ""
        curr_sha = str(c.get("sha256") or "") if c else ""
        prev_size = p.get("size_bytes") if p else None
        curr_size = c.get("size_bytes") if c else None
        changed = (prev_exists != curr_exists) or (prev_sha != curr_sha) or (prev_size != curr_size)
        # 条件分岐: `p is None` を満たす経路を評価する。
        if p is None:
            status = "added"
        # 条件分岐: 前段条件が不成立で、`c is None` を追加評価する。
        elif c is None:
            status = "removed"
        # 条件分岐: 前段条件が不成立で、`changed` を追加評価する。
        elif changed:
            status = "changed"
        else:
            status = "unchanged"

        rows.append(
            {
                "path": path,
                "status": status,
                "previous_exists": prev_exists,
                "current_exists": curr_exists,
                "previous_size_bytes": prev_size,
                "current_size_bytes": curr_size,
                "previous_sha256": prev_sha or None,
                "current_sha256": curr_sha or None,
                "changed": bool(changed),
            }
        )

    return rows


def _write_topic_delta_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "topic",
                "status",
                "previous_file_count",
                "current_file_count",
                "delta_file_count",
                "previous_total_size_bytes",
                "current_total_size_bytes",
                "delta_total_size_bytes",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    row["topic"],
                    row["status"],
                    row["previous_file_count"],
                    row["current_file_count"],
                    row["delta_file_count"],
                    row["previous_total_size_bytes"],
                    row["current_total_size_bytes"],
                    row["delta_total_size_bytes"],
                ]
            )


def _find_latest_snapshot(snapshots_dir: Path, *, exclude: Path) -> Optional[Path]:
    files = [p for p in snapshots_dir.glob("public_outputs_manifest_snapshot_*.json") if p.is_file()]
    # 条件分岐: `not files` を満たす経路を評価する。
    if not files:
        return None

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files:
        # 条件分岐: `p.resolve() != exclude.resolve()` を満たす経路を評価する。
        if p.resolve() != exclude.resolve():
            return p

    return None


def _compact_snapshot(
    *,
    current_manifest: Dict[str, Any],
    source_manifest_path: Path,
    release_ref: str,
    release_ref_exists: bool,
    head_short: str,
    snapshot_label: str,
) -> Dict[str, Any]:
    summary = current_manifest.get("summary") or {}
    required_core = current_manifest.get("required_core_files") or []
    topics = current_manifest.get("topics") or []
    return {
        "generated_utc": _utc_now(),
        "domain": "summary",
        "step": "8.7.16.2 (public manifest continuity snapshot)",
        "snapshot_label": snapshot_label,
        "release_ref": release_ref,
        "release_ref_exists": bool(release_ref_exists),
        "git_head_short": head_short,
        "source_manifest": {
            "path": _rel(source_manifest_path),
            "generated_utc": current_manifest.get("generated_utc"),
        },
        "summary": {
            "file_count": int(summary.get("file_count") or 0),
            "total_size_bytes": int(summary.get("total_size_bytes") or 0),
            "topics_count": int(summary.get("topics_count") or 0),
            "public_tree_sha256": str(summary.get("public_tree_sha256") or ""),
            "ok": bool(current_manifest.get("ok", False)),
        },
        "required_core_files": required_core,
        "topics": topics,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 8 / Step 8.7.16.2: public outputs manifest continuity check."
    )
    ap.add_argument(
        "--manifest-json",
        type=str,
        default=str(_ROOT / "output" / "public" / "summary" / "public_outputs_manifest.json"),
        help="source public manifest JSON (default: output/public/summary/public_outputs_manifest.json)",
    )
    ap.add_argument(
        "--manifest-topics-csv",
        type=str,
        default=str(_ROOT / "output" / "public" / "summary" / "public_outputs_manifest_topics.csv"),
        help="source public manifest topic CSV (default: output/public/summary/public_outputs_manifest_topics.csv)",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "public_outputs_manifest_continuity.json"),
        help="continuity JSON output (default: output/private/summary/public_outputs_manifest_continuity.json)",
    )
    ap.add_argument(
        "--out-topics-csv",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "public_outputs_manifest_continuity_topics.csv"),
        help="topic delta CSV output (default: output/private/summary/public_outputs_manifest_continuity_topics.csv)",
    )
    ap.add_argument(
        "--snapshots-dir",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "public_manifest_snapshots"),
        help="snapshot directory (default: output/private/summary/public_manifest_snapshots)",
    )
    ap.add_argument(
        "--release-ref",
        type=str,
        default="",
        help="release/tag reference (default: latest local tag)",
    )
    ap.add_argument(
        "--snapshot-label",
        type=str,
        default="",
        help="manual snapshot label (default: <release_ref>_head-<sha>)",
    )
    ap.add_argument(
        "--skip-refresh-manifest",
        action="store_true",
        help="do not re-run public_outputs_manifest before continuity check",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    manifest_json = Path(str(args.manifest_json))
    # 条件分岐: `not manifest_json.is_absolute()` を満たす経路を評価する。
    if not manifest_json.is_absolute():
        manifest_json = (_ROOT / manifest_json).resolve()

    manifest_topics_csv = Path(str(args.manifest_topics_csv))
    # 条件分岐: `not manifest_topics_csv.is_absolute()` を満たす経路を評価する。
    if not manifest_topics_csv.is_absolute():
        manifest_topics_csv = (_ROOT / manifest_topics_csv).resolve()

    out_json = Path(str(args.out_json))
    # 条件分岐: `not out_json.is_absolute()` を満たす経路を評価する。
    if not out_json.is_absolute():
        out_json = (_ROOT / out_json).resolve()

    out_topics_csv = Path(str(args.out_topics_csv))
    # 条件分岐: `not out_topics_csv.is_absolute()` を満たす経路を評価する。
    if not out_topics_csv.is_absolute():
        out_topics_csv = (_ROOT / out_topics_csv).resolve()

    snapshots_dir = Path(str(args.snapshots_dir))
    # 条件分岐: `not snapshots_dir.is_absolute()` を満たす経路を評価する。
    if not snapshots_dir.is_absolute():
        snapshots_dir = (_ROOT / snapshots_dir).resolve()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_topics_csv.parent.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # 条件分岐: `not bool(args.skip_refresh_manifest)` を満たす経路を評価する。
    if not bool(args.skip_refresh_manifest):
        rc = public_outputs_manifest.main(
            [
                "--out-json",
                str(manifest_json),
                "--out-topics-csv",
                str(manifest_topics_csv),
            ]
        )
        # 条件分岐: `rc != 0` を満たす経路を評価する。
        if rc != 0:
            raise SystemExit(rc)

    # 条件分岐: `not manifest_json.exists()` を満たす経路を評価する。

    if not manifest_json.exists():
        raise SystemExit(f"[fail] missing manifest: {manifest_json}")

    current_manifest = _read_json(manifest_json)

    release_ref = str(args.release_ref or "").strip() or _latest_tag() or "(none)"
    release_ref_exists = _tag_exists(release_ref) if release_ref != "(none)" else False
    head_short = _head_short()
    snapshot_label = str(args.snapshot_label or "").strip()
    # 条件分岐: `not snapshot_label` を満たす経路を評価する。
    if not snapshot_label:
        snapshot_label = f"{release_ref}_head-{head_short}"

    snapshot_label = _sanitize_label(snapshot_label)

    snapshot_path = snapshots_dir / f"public_outputs_manifest_snapshot_{snapshot_label}.json"

    baseline_path: Optional[Path] = snapshot_path if snapshot_path.exists() else None
    # 条件分岐: `baseline_path is None` を満たす経路を評価する。
    if baseline_path is None:
        baseline_path = _find_latest_snapshot(snapshots_dir, exclude=snapshot_path)

    previous_snapshot = _read_json(baseline_path) if baseline_path is not None else None

    current_snapshot = _compact_snapshot(
        current_manifest=current_manifest,
        source_manifest_path=manifest_json,
        release_ref=release_ref,
        release_ref_exists=release_ref_exists,
        head_short=head_short,
        snapshot_label=snapshot_label,
    )

    prev_summary = (previous_snapshot or {}).get("summary") or {}
    curr_summary = current_snapshot.get("summary") or {}
    prev_topics = (previous_snapshot or {}).get("topics") or []
    curr_topics = current_snapshot.get("topics") or []
    prev_core = (previous_snapshot or {}).get("required_core_files") or []
    curr_core = current_snapshot.get("required_core_files") or []

    topic_deltas = _build_topic_deltas(prev_topics, curr_topics)
    _write_topic_delta_csv(out_topics_csv, topic_deltas)

    core_deltas = _build_required_core_deltas(prev_core, curr_core)
    changed_core = [row for row in core_deltas if row.get("status") == "changed"]
    core_missing_regression = [
        row["path"]
        for row in core_deltas
        if bool(row.get("previous_exists")) and (not bool(row.get("current_exists")))
    ]
    prev_tree = str(prev_summary.get("public_tree_sha256") or "")
    curr_tree = str(curr_summary.get("public_tree_sha256") or "")
    tree_changed = bool(previous_snapshot is not None and prev_tree != curr_tree)

    # 条件分岐: `not bool(current_manifest.get("ok", False))` を満たす経路を評価する。
    if not bool(current_manifest.get("ok", False)):
        continuity_status = "reject_manifest_not_ok"
    # 条件分岐: 前段条件が不成立で、`previous_snapshot is None` を追加評価する。
    elif previous_snapshot is None:
        continuity_status = "baseline_initialized"
    # 条件分岐: 前段条件が不成立で、`core_missing_regression` を追加評価する。
    elif core_missing_regression:
        continuity_status = "reject_core_missing_regression"
    # 条件分岐: 前段条件が不成立で、`tree_changed` を追加評価する。
    elif tree_changed:
        continuity_status = "watch_tree_hash_changed"
    # 条件分岐: 前段条件が不成立で、`changed_core` を追加評価する。
    elif changed_core:
        continuity_status = "watch_required_core_changed"
    else:
        continuity_status = "pass_tree_stable"

    current_ok = bool(current_manifest.get("ok", False))
    continuity_ok = bool(current_ok and len(core_missing_regression) == 0)

    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "domain": "summary",
        "step": "8.7.16.2 (public manifest continuity)",
        "params": {
            "release_ref": release_ref,
            "release_ref_exists": bool(release_ref_exists),
            "snapshot_label": snapshot_label,
        },
        "status": continuity_status,
        "ok": continuity_ok,
        "baseline": {
            "path": _rel(baseline_path) if baseline_path is not None else None,
            "available": bool(previous_snapshot is not None),
            "generated_utc": (previous_snapshot or {}).get("generated_utc"),
            "release_ref": (previous_snapshot or {}).get("release_ref"),
        },
        "current": {
            "snapshot_path": _rel(snapshot_path),
            "manifest_path": _rel(manifest_json),
            "manifest_generated_utc": current_manifest.get("generated_utc"),
            "summary": curr_summary,
        },
        "deltas": {
            "tree_hash_changed": bool(tree_changed),
            "previous_tree_sha256": prev_tree or None,
            "current_tree_sha256": curr_tree or None,
            "file_count_delta": (
                int(curr_summary.get("file_count") or 0) - int(prev_summary.get("file_count") or 0)
                if previous_snapshot is not None
                else None
            ),
            "total_size_bytes_delta": (
                int(curr_summary.get("total_size_bytes") or 0) - int(prev_summary.get("total_size_bytes") or 0)
                if previous_snapshot is not None
                else None
            ),
            "topics_count_delta": (
                int(curr_summary.get("topics_count") or 0) - int(prev_summary.get("topics_count") or 0)
                if previous_snapshot is not None
                else None
            ),
            "required_core_changed_count": int(len(changed_core)),
            "required_core_missing_regression_count": int(len(core_missing_regression)),
            "required_core_missing_regression": sorted(core_missing_regression),
            "required_core_changed": [row for row in changed_core],
            "topic_status_counts": {
                "added": sum(1 for row in topic_deltas if row["status"] == "added"),
                "removed": sum(1 for row in topic_deltas if row["status"] == "removed"),
                "changed": sum(1 for row in topic_deltas if row["status"] == "changed"),
                "unchanged": sum(1 for row in topic_deltas if row["status"] == "unchanged"),
            },
        },
        "outputs": {
            "json": _rel(out_json),
            "topic_delta_csv": _rel(out_topics_csv),
            "snapshot_json": _rel(snapshot_path),
        },
        "repro_commands": {
            "public_outputs_manifest": "python -B scripts/summary/public_outputs_manifest.py",
            "public_outputs_manifest_continuity": "python -B scripts/summary/public_outputs_manifest_continuity.py",
            "release_manifest": "python -B scripts/summary/release_manifest.py",
        },
    }

    snapshot_path.write_text(json.dumps(current_snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    worklog.append_event(
        {
            "domain": "summary",
            "action": "public_outputs_manifest_continuity",
            "outputs": [out_json, out_topics_csv, snapshot_path],
            "params": {"release_ref": release_ref, "snapshot_label": snapshot_label},
            "result": {
                "ok": bool(continuity_ok),
                "status": continuity_status,
                "tree_hash_changed": bool(tree_changed),
                "required_core_missing_regression_count": int(len(core_missing_regression)),
            },
        }
    )

    print("public_outputs_manifest_continuity:")
    print(f"- ok: {continuity_ok}")
    print(f"- status: {continuity_status}")
    print(f"- out_json: {out_json}")
    print(f"- out_topics_csv: {out_topics_csv}")
    print(f"- snapshot: {snapshot_path}")
    # 条件分岐: `core_missing_regression` を満たす経路を評価する。
    if core_missing_regression:
        print("[warn] required-core regressions:")
        for path in core_missing_regression:
            print(f"  - {path}")

    return 0 if continuity_ok else 1


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
