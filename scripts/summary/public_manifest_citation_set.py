#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
public_manifest_citation_set.py

Phase 8 / Step 8.7.16.3 向けの citation set を固定する。

出力（既定）：
- output/private/summary/public_manifest_citation_set.json
- output/private/summary/public_manifest_citation_set.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import public_outputs_manifest_continuity  # noqa: E402
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


def _sha256(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


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


def _head_short() -> str:
    rc, out = _git("rev-parse", "--short", "HEAD")
    if rc != 0 or not out:
        return "no-git"
    return out


def _tag_exists(tag: str) -> bool:
    if not tag or tag == "(none)":
        return False
    rc, _ = _git("rev-parse", "-q", "--verify", f"refs/tags/{tag}")
    return rc == 0


def _latest_tag() -> str:
    rc, out = _git("tag", "--sort=-creatordate")
    if rc != 0:
        return ""
    tags = [line.strip() for line in out.splitlines() if line.strip()]
    return tags[0] if tags else ""


def _origin_url() -> str:
    rc, out = _git("remote", "get-url", "origin")
    if rc != 0:
        return ""
    return out.strip()


def _origin_web_base(origin: str) -> str:
    s = origin.strip()
    if not s:
        return ""
    if s.startswith("git@github.com:"):
        s = "https://github.com/" + s.split(":", 1)[1]
    if s.startswith("ssh://git@github.com/"):
        s = "https://github.com/" + s.split("ssh://git@github.com/", 1)[1]
    if s.endswith(".git"):
        s = s[:-4]
    return s.rstrip("/")


def _artifact(path: Path, *, kind: str, generated_utc: Optional[str] = None) -> Dict[str, Any]:
    exists = path.exists()
    return {
        "kind": kind,
        "path": _rel(path),
        "exists": bool(exists),
        "size_bytes": int(path.stat().st_size) if exists else None,
        "sha256": _sha256(path) if exists else None,
        "generated_utc": generated_utc,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "path", "exists", "size_bytes", "sha256", "generated_utc"])
        for row in rows:
            w.writerow(
                [
                    row.get("kind"),
                    row.get("path"),
                    bool(row.get("exists")),
                    row.get("size_bytes"),
                    row.get("sha256"),
                    row.get("generated_utc"),
                ]
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 8 / Step 8.7.16.3: lock public-manifest citation set.")
    ap.add_argument(
        "--continuity-json",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "public_outputs_manifest_continuity.json"),
        help="continuity JSON path",
    )
    ap.add_argument(
        "--manifest-json",
        type=str,
        default=str(_ROOT / "output" / "public" / "summary" / "public_outputs_manifest.json"),
        help="public manifest JSON path",
    )
    ap.add_argument(
        "--topics-csv",
        type=str,
        default=str(_ROOT / "output" / "public" / "summary" / "public_outputs_manifest_topics.csv"),
        help="public manifest topics CSV path",
    )
    ap.add_argument(
        "--release-manifest-json",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "release_manifest.json"),
        help="release manifest JSON path",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "public_manifest_citation_set.json"),
        help="output JSON path",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "public_manifest_citation_set.csv"),
        help="output CSV path",
    )
    ap.add_argument(
        "--release-ref",
        type=str,
        default="",
        help="release/tag reference (default: continuity param, else latest tag)",
    )
    ap.add_argument(
        "--skip-refresh-continuity",
        action="store_true",
        help="skip running public_outputs_manifest_continuity.py before locking",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    continuity_json = Path(str(args.continuity_json))
    if not continuity_json.is_absolute():
        continuity_json = (_ROOT / continuity_json).resolve()
    manifest_json = Path(str(args.manifest_json))
    if not manifest_json.is_absolute():
        manifest_json = (_ROOT / manifest_json).resolve()
    topics_csv = Path(str(args.topics_csv))
    if not topics_csv.is_absolute():
        topics_csv = (_ROOT / topics_csv).resolve()
    release_manifest_json = Path(str(args.release_manifest_json))
    if not release_manifest_json.is_absolute():
        release_manifest_json = (_ROOT / release_manifest_json).resolve()
    out_json = Path(str(args.out_json))
    if not out_json.is_absolute():
        out_json = (_ROOT / out_json).resolve()
    out_csv = Path(str(args.out_csv))
    if not out_csv.is_absolute():
        out_csv = (_ROOT / out_csv).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not bool(args.skip_refresh_continuity):
        rc = public_outputs_manifest_continuity.main([])
        if rc != 0:
            raise SystemExit(rc)

    if not continuity_json.exists():
        raise SystemExit(f"[fail] missing continuity JSON: {continuity_json}")
    continuity = _read_json(continuity_json)
    manifest = _read_json(manifest_json) if manifest_json.exists() else {}
    release_manifest = _read_json(release_manifest_json) if release_manifest_json.exists() else {}

    continuity_release_ref = str((continuity.get("params") or {}).get("release_ref") or "").strip()
    release_ref = str(args.release_ref or "").strip() or continuity_release_ref or _latest_tag() or "(none)"
    release_ref_exists = _tag_exists(release_ref)
    head_short = _head_short()
    origin = _origin_url()
    web_base = _origin_web_base(origin)

    snapshot_rel = str(((continuity.get("outputs") or {}).get("snapshot_json")) or "")
    snapshot_path = (_ROOT / snapshot_rel).resolve() if snapshot_rel else Path("")

    artifacts: List[Dict[str, Any]] = [
        _artifact(manifest_json, kind="public_outputs_manifest_json", generated_utc=manifest.get("generated_utc")),
        _artifact(topics_csv, kind="public_outputs_manifest_topics_csv"),
        _artifact(
            continuity_json,
            kind="public_outputs_manifest_continuity_json",
            generated_utc=continuity.get("generated_utc"),
        ),
        _artifact(out_csv, kind="public_manifest_citation_set_csv"),
        _artifact(release_manifest_json, kind="release_manifest_json", generated_utc=release_manifest.get("generated_utc")),
    ]
    if snapshot_rel:
        artifacts.append(_artifact(snapshot_path, kind="public_manifest_snapshot_json"))

    summary = (manifest.get("summary") or {}) if isinstance(manifest, dict) else {}
    continuity_deltas = (continuity.get("deltas") or {}) if isinstance(continuity, dict) else {}
    continuity_status = str(continuity.get("status") or "")
    continuity_ok = bool(continuity.get("ok", False))
    manifest_ok = bool(manifest.get("ok", False))
    required_core_regressions = int(continuity_deltas.get("required_core_missing_regression_count") or 0)

    if manifest_ok and continuity_ok and continuity_status == "pass_tree_stable" and required_core_regressions == 0:
        decision = "ready_for_release_citation_lock"
    elif manifest_ok and continuity_ok and required_core_regressions == 0:
        decision = "watch_before_release_tree_changed"
    else:
        decision = "reject_not_ready_for_release"

    release_tag_url = f"{web_base}/releases/tag/{release_ref}" if (web_base and release_ref_exists and release_ref != "(none)") else None
    manifest_blob_url = (
        f"{web_base}/blob/{release_ref}/output/public/summary/public_outputs_manifest.json"
        if (web_base and release_ref_exists and release_ref != "(none)")
        else None
    )
    topics_blob_url = (
        f"{web_base}/blob/{release_ref}/output/public/summary/public_outputs_manifest_topics.csv"
        if (web_base and release_ref_exists and release_ref != "(none)")
        else None
    )

    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "domain": "summary",
        "step": "8.7.16.3 (public manifest citation lock set)",
        "release_ref": release_ref,
        "release_ref_exists": bool(release_ref_exists),
        "git_head_short": head_short,
        "origin_remote": origin or None,
        "origin_web_base": web_base or None,
        "decision": decision,
        "gates": {
            "manifest_ok": manifest_ok,
            "continuity_ok": continuity_ok,
            "continuity_status": continuity_status,
            "required_core_missing_regression_count": required_core_regressions,
        },
        "manifest_summary": {
            "file_count": int(summary.get("file_count") or 0),
            "topics_count": int(summary.get("topics_count") or 0),
            "total_size_bytes": int(summary.get("total_size_bytes") or 0),
            "public_tree_sha256": str(summary.get("public_tree_sha256") or ""),
        },
        "continuity_snapshot": {
            "path": snapshot_rel or None,
            "label": str((continuity.get("params") or {}).get("snapshot_label") or ""),
            "status": continuity_status,
            "tree_hash_changed": bool(continuity_deltas.get("tree_hash_changed", False)),
            "topic_status_counts": continuity_deltas.get("topic_status_counts") or {},
        },
        "artifacts": artifacts,
        "citation_links": {
            "release_tag": release_tag_url,
            "public_outputs_manifest_blob": manifest_blob_url,
            "public_outputs_manifest_topics_blob": topics_blob_url,
        },
        "outputs": {
            "json": _rel(out_json),
            "csv": _rel(out_csv),
        },
        "repro_commands": {
            "public_outputs_manifest_continuity": "python -B scripts/summary/public_outputs_manifest_continuity.py",
            "public_manifest_citation_set": "python -B scripts/summary/public_manifest_citation_set.py",
            "release_manifest": "python -B scripts/summary/release_manifest.py",
        },
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_rows: List[Dict[str, Any]] = []
    for row in artifacts:
        if row.get("kind") == "public_manifest_citation_set_csv":
            row = dict(row)
            row["exists"] = True
            row["size_bytes"] = None
            row["sha256"] = None
        csv_rows.append(row)
    _write_csv(out_csv, csv_rows)

    csv_art = _artifact(out_csv, kind="public_manifest_citation_set_csv")
    for row in artifacts:
        if row.get("kind") == "public_manifest_citation_set_csv":
            row["exists"] = csv_art["exists"]
            row["size_bytes"] = csv_art["size_bytes"]
            row["sha256"] = csv_art["sha256"]

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    worklog.append_event(
        {
            "domain": "summary",
            "action": "public_manifest_citation_set",
            "outputs": [out_json, out_csv],
            "params": {"release_ref": release_ref},
            "result": {"decision": decision, "release_ref_exists": bool(release_ref_exists)},
        }
    )

    print("public_manifest_citation_set:")
    print(f"- decision: {decision}")
    print(f"- release_ref: {release_ref} (exists={release_ref_exists})")
    print(f"- out_json: {out_json}")
    print(f"- out_csv: {out_csv}")
    return 0 if decision != "reject_not_ready_for_release" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
