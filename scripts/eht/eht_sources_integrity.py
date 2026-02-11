#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_repo_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return root / p


def _extract_local_files(source: Dict[str, Any]) -> List[Tuple[str, str, str | None]]:
    out: List[Tuple[str, str, str | None]] = []
    for k, v in source.items():
        if not (isinstance(k, str) and k.startswith("local_")):
            continue
        if k.endswith("_sha256"):
            continue
        if not isinstance(v, str) or not v.strip():
            continue
        sha_key = f"{k}_sha256"
        sha_expected = source.get(sha_key)
        sha_expected_s = str(sha_expected).strip().upper() if isinstance(sha_expected, str) else None
        out.append((k, v.strip(), sha_expected_s or None))
    return out


def main() -> int:
    root = _repo_root()
    inp = root / "data" / "eht" / "eht_black_holes.json"
    out_dir = root / "output" / "eht"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "eht_sources_integrity.json"

    obj = _read_json(inp)
    sources = obj.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("data/eht/eht_black_holes.json: sources is missing or not a dict")

    rows: List[Dict[str, Any]] = []
    for source_key, source in sources.items():
        if not isinstance(source_key, str):
            continue
        if not isinstance(source, dict):
            continue

        label = str(source.get("label") or "").strip() or None
        url = str(source.get("url") or "").strip() or None

        local_files = _extract_local_files(source)
        if not local_files:
            rows.append(
                {
                    "source_key": source_key,
                    "label": label,
                    "url": url,
                    "files": [],
                    "ok": None,
                    "notes": ["no local_* files declared in data/eht/eht_black_holes.json:sources"],
                }
            )
            continue

        file_rows: List[Dict[str, Any]] = []
        ok_all = True
        for field, rel, sha_expected in local_files:
            path = _to_repo_path(root, rel)
            exists = path.exists()
            actual = _sha256(path) if exists else None
            match = None
            if sha_expected and actual:
                match = sha_expected == actual
            if sha_expected and actual and not match:
                ok_all = False
            if not exists:
                ok_all = False
            file_rows.append(
                {
                    "field": field,
                    "path": str(path.relative_to(root)).replace("\\", "/") if exists and not path.is_absolute() else str(path),
                    "exists": exists,
                    "bytes": int(path.stat().st_size) if exists else None,
                    "sha256_expected": sha_expected,
                    "sha256_actual": actual,
                    "sha256_match": match,
                }
            )
        rows.append(
            {
                "source_key": source_key,
                "label": label,
                "url": url,
                "files": file_rows,
                "ok": ok_all,
            }
        )

    totals = {
        "sources_total": len(rows),
        "sources_with_local_files": sum(1 for r in rows if isinstance(r.get("files"), list) and len(r.get("files")) > 0),
        "sources_ok": sum(1 for r in rows if r.get("ok") is True),
        "sources_not_ok": sum(1 for r in rows if r.get("ok") is False),
        "sources_no_local_files": sum(1 for r in rows if r.get("ok") is None),
    }

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(inp.relative_to(root)).replace("\\", "/"),
        "output": str(out_json.relative_to(root)).replace("\\", "/"),
        "totals": totals,
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "eht_sources_integrity",
                "outputs": [str(out_json.relative_to(root)).replace("\\", "/")],
                "metrics": totals,
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    print(f"[ok] totals: {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

