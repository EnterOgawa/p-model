#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
env_fingerprint.py

Phase 8 / Step 8.3（データ・コード公開）向けの “実行環境フィンガープリント” を出力する。

目的：
- どの Python / OS / パッケージ環境で生成された成果物かを機械可読で固定する（再現性の補助）。

出力（既定）：
- `output/private/summary/env_fingerprint.json`
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(_ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


# 関数: `_run_text` の入出力契約と処理意図を定義する。

def _run_text(cmd: List[str]) -> Dict[str, Any]:
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "ok": (cp.returncode == 0),
            "returncode": int(cp.returncode),
            "stdout": cp.stdout,
            "stderr": cp.stderr,
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# 関数: `_pip_version` の入出力契約と処理意図を定義する。

def _pip_version() -> Optional[str]:
    res = _run_text([sys.executable, "-m", "pip", "--version"])
    # 条件分岐: `not res.get("ok")` を満たす経路を評価する。
    if not res.get("ok"):
        return None

    s = str(res.get("stdout") or "").strip()
    return s or None


# 関数: `_pip_freeze` の入出力契約と処理意図を定義する。

def _pip_freeze() -> Dict[str, Any]:
    res = _run_text([sys.executable, "-m", "pip", "freeze"])
    # 条件分岐: `not res.get("ok")` を満たす経路を評価する。
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error") or (res.get("stderr") or "").strip() or "pip freeze failed"}

    lines = [ln.strip() for ln in str(res.get("stdout") or "").splitlines() if ln.strip()]
    return {"ok": True, "packages": lines}


# 関数: `build_fingerprint` の入出力契約と処理意図を定義する。

def build_fingerprint(*, include_freeze: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "env": {
            "os_name": os.name,
            "platform": sys.platform,
            "platform_full": platform.platform(),
            "python": sys.version.split()[0],
            "python_full": sys.version,
            "python_executable": sys.executable,
            "cwd": str(Path.cwd()),
        },
        "pip": {"version": _pip_version()},
    }
    # 条件分岐: `include_freeze` を満たす経路を評価する。
    if include_freeze:
        payload["pip"]["freeze"] = _pip_freeze()

    return payload


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 8 / Step 8.3: write environment fingerprint JSON.")
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "env_fingerprint.json"),
        help="output path (default: output/private/summary/env_fingerprint.json)",
    )
    ap.add_argument("--no-freeze", action="store_true", help="skip pip freeze (faster)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    out_json = Path(str(args.out_json))
    # 条件分岐: `not out_json.is_absolute()` を満たす経路を評価する。
    if not out_json.is_absolute():
        out_json = (_ROOT / out_json).resolve()

    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload = build_fingerprint(include_freeze=(not bool(args.no_freeze)))
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    worklog.append_event(
        {
            "domain": "summary",
            "action": "env_fingerprint",
            "outputs": [out_json],
            "params": {"include_freeze": (not bool(args.no_freeze))},
            "result": {
                "pip_freeze_ok": bool((payload.get("pip") or {}).get("freeze", {}).get("ok", True)),
                "pip_freeze_n": len(((payload.get("pip") or {}).get("freeze") or {}).get("packages") or []),
            },
        }
    )

    print("env_fingerprint:")
    print(f"- out: {out_json}")
    # 条件分岐: `"freeze" in (payload.get("pip") or {})` を満たす経路を評価する。
    if "freeze" in (payload.get("pip") or {}):
        freeze = (payload.get("pip") or {}).get("freeze") or {}
        print(f"- pip_freeze_ok: {freeze.get('ok', False)}")
        # 条件分岐: `freeze.get("ok", False)` を満たす経路を評価する。
        if freeze.get("ok", False):
            print(f"- packages: {len(freeze.get('packages') or [])}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
