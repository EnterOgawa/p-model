#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
desi_dr1_lss_raw_inventory.py

Phase 4（宇宙論）/ Step 4.5B.21.4.4.7（DESI DR1 multi-tracer）:
ローカルに置いた DESI DR1 LSS raw（clustering catalogs; fits）を棚卸しし、
どの tracer/cap/random_index が存在するかを再現可能な形で固定する。

目的：
- 「raw 内の対象セットを特定」を機械的に行い、議論のブレ（Lya QSO の有無など）を防ぐ。

入力：
- `data/cosmology/desi_dr1_lss/raw/`（既定）

出力（固定）：
- `output/private/cosmology/desi_dr1_lss_raw_inventory.csv`
- `output/private/cosmology/desi_dr1_lss_raw_inventory_summary.json`
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


@dataclass(frozen=True)
class _Entry:
    file_name: str
    file_relpath: str
    size_bytes: int
    sample: str
    cap: str
    kind: str  # "data" | "random"
    random_index: Optional[int]


_RE_RAN = re.compile(r"^(?P<sample>.+)_(?P<cap>NGC|SGC)_(?P<idx>\d+)_clustering\.ran\.fits$", re.IGNORECASE)
_RE_DAT = re.compile(r"^(?P<sample>.+)_(?P<cap>NGC|SGC)_clustering\.dat\.fits$", re.IGNORECASE)


def _parse_entry(path: Path) -> Optional[_Entry]:
    name = path.name
    m_ran = _RE_RAN.match(name)
    if m_ran:
        sample = str(m_ran.group("sample"))
        cap = str(m_ran.group("cap")).upper()
        idx = int(m_ran.group("idx"))
        return _Entry(
            file_name=name,
            file_relpath=str(path.relative_to(_ROOT)).replace("\\", "/"),
            size_bytes=int(path.stat().st_size),
            sample=sample,
            cap=cap,
            kind="random",
            random_index=idx,
        )
    m_dat = _RE_DAT.match(name)
    if m_dat:
        sample = str(m_dat.group("sample"))
        cap = str(m_dat.group("cap")).upper()
        return _Entry(
            file_name=name,
            file_relpath=str(path.relative_to(_ROOT)).replace("\\", "/"),
            size_bytes=int(path.stat().st_size),
            sample=sample,
            cap=cap,
            kind="data",
            random_index=None,
        )
    return None


def _write_csv(path: Path, entries: List[_Entry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "file_relpath", "size_bytes", "sample", "cap", "kind", "random_index"])
        for e in entries:
            w.writerow(
                [
                    e.file_name,
                    e.file_relpath,
                    int(e.size_bytes),
                    e.sample,
                    e.cap,
                    e.kind,
                    "" if e.random_index is None else int(e.random_index),
                ]
            )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="DESI DR1 LSS raw inventory (local fits listing).")
    ap.add_argument(
        "--raw-dir",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "desi_dr1_lss" / "raw"),
        help="Raw directory containing *_clustering.{dat,ran}.fits (default: data/cosmology/desi_dr1_lss/raw).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(_ROOT / "output" / "private" / "cosmology"),
        help="Output dir (default: output/private/cosmology).",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    raw_dir = Path(str(args.raw_dir)).resolve()
    if not raw_dir.exists():
        raise SystemExit(f"raw dir not found: {raw_dir}")
    if not raw_dir.is_dir():
        raise SystemExit(f"raw dir is not a directory: {raw_dir}")

    out_dir = Path(str(args.out_dir)).resolve()
    if out_dir.is_absolute():
        pass
    else:
        out_dir = (_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "desi_dr1_lss_raw_inventory.csv"
    out_json = out_dir / "desi_dr1_lss_raw_inventory_summary.json"

    entries: List[_Entry] = []
    ignored: List[str] = []
    for p in sorted(raw_dir.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file():
            continue
        ent = _parse_entry(p)
        if ent is None:
            ignored.append(p.name)
            continue
        entries.append(ent)

    entries = sorted(entries, key=lambda e: (e.sample.lower(), e.cap, 0 if e.kind == "data" else 1, e.random_index or -1, e.file_name.lower()))
    _write_csv(out_csv, entries)

    by_sample: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        s = by_sample.setdefault(
            e.sample,
            {"sample": e.sample, "caps": {}, "n_files": 0, "n_data": 0, "n_random": 0, "random_indices": []},
        )
        s["n_files"] += 1
        caps = s["caps"]
        cap_block = caps.setdefault(e.cap, {"cap": e.cap, "has_data": False, "random_indices": []})
        if e.kind == "data":
            s["n_data"] += 1
            cap_block["has_data"] = True
        else:
            s["n_random"] += 1
            if e.random_index is not None:
                cap_block["random_indices"].append(int(e.random_index))
                s["random_indices"].append(int(e.random_index))

    # normalize lists
    for s in by_sample.values():
        s["random_indices"] = sorted(set(int(x) for x in s["random_indices"]))
        for cap_block in s["caps"].values():
            cap_block["random_indices"] = sorted(set(int(x) for x in cap_block["random_indices"]))
        s["caps"] = {k: s["caps"][k] for k in sorted(s["caps"].keys())}

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "4.5B.21.4.4.7 (DESI DR1 raw inventory)",
        "inputs": {"raw_dir": str(raw_dir)},
        "outputs": {"csv": str(out_csv), "summary_json": str(out_json)},
        "summary": {
            "n_files_parsed": int(len(entries)),
            "n_files_ignored": int(len(ignored)),
            "samples": sorted(by_sample.keys()),
        },
        "samples": {k: by_sample[k] for k in sorted(by_sample.keys(), key=str.lower)},
        "ignored_files": ignored,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        worklog.append_event(
            {
                "event_type": "desi_dr1_lss_raw_inventory",
                "argv": sys.argv,
                "inputs": {"raw_dir": raw_dir},
                "outputs": {"csv": out_csv, "summary_json": out_json},
                "metrics": {"n_files_parsed": len(entries), "samples": sorted(by_sample.keys())},
            }
        )
    except Exception:
        pass

    print(f"[ok] csv : {out_csv}")
    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

