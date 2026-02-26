#!/usr/bin/env python3
"""
worklog.py

スクリプト実行の作業履歴を JSONL で `output/private/summary/work_history.jsonl` に追記する。
（人間向けの要約は `doc/WORK_HISTORY.md` を更新する）

目的:
  - 何をいつ実行し、どの出力を生成したかを機械可読で残す（重複作業の防止）。
  - 主要スクリプトが自動で記録する前提。
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_jsonl_path(root: Path) -> Path:
    return root / "output" / "private" / "summary" / "work_history.jsonl"

def _lock_path_for(jsonl_path: Path) -> Path:
    # Cross-platform append lock (prevents interleaved JSON when multiple scripts run concurrently).
    return jsonl_path.with_suffix(jsonl_path.suffix + ".lock")


def _ensure_lock_file(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `lock_path.exists() and lock_path.stat().st_size >= 1` を満たす経路を評価する。
    if lock_path.exists() and lock_path.stat().st_size >= 1:
        return
    # Ensure the file is at least 1 byte so region locking is well-defined on Windows.

    with open(lock_path, "ab") as f:
        # 条件分岐: `f.tell() == 0` を満たす経路を評価する。
        if f.tell() == 0:
            f.write(b"0")
            f.flush()


def _lock_file(f) -> None:  # type: ignore[no-untyped-def]
    # 条件分岐: `os.name == "nt"` を満たす経路を評価する。
    if os.name == "nt":
        import msvcrt

        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        return

    import fcntl

    fcntl.flock(f.fileno(), fcntl.LOCK_EX)


def _unlock_file(f) -> None:  # type: ignore[no-untyped-def]
    # 条件分岐: `os.name == "nt"` を満たす経路を評価する。
    if os.name == "nt":
        import msvcrt

        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _rel(root: Path, p: Path) -> str:
    try:
        return str(p.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def append_event(event: Dict[str, Any], *, jsonl_path: Optional[Path] = None) -> Path:
    """
    Append one event to JSONL history and return the path written.
    The caller should pass only small, stable fields (paths, key metrics, args).
    """
    root = _repo_root()
    path = jsonl_path or _default_jsonl_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)

    ev: Dict[str, Any] = dict(event)
    ev.setdefault("generated_utc", datetime.now(timezone.utc).isoformat())

    def _norm(v: Any) -> Any:
        # 条件分岐: `isinstance(v, Path)` を満たす経路を評価する。
        if isinstance(v, Path):
            return _rel(root, v)

        # 条件分岐: `isinstance(v, dict)` を満たす経路を評価する。

        if isinstance(v, dict):
            return {kk: _norm(vv) for kk, vv in v.items()}

        # 条件分岐: `isinstance(v, (list, tuple))` を満たす経路を評価する。

        if isinstance(v, (list, tuple)):
            return [_norm(vv) for vv in v]

        return v

    # Normalize common path-like fields if present

    for k in ("output", "outputs", "summary_path", "status_path", "log", "input", "inputs"):
        # 条件分岐: `k not in ev` を満たす経路を評価する。
        if k not in ev:
            continue

        ev[k] = _norm(ev.get(k))

    line = json.dumps(ev, ensure_ascii=False)
    lock_path = _lock_path_for(path)
    _ensure_lock_file(lock_path)
    with open(lock_path, "r+b") as lf:
        _lock_file(lf)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        finally:
            _unlock_file(lf)

    return path
