#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _find_block(text: str, needle: str, *, window: int = 900) -> Optional[str]:
    i = text.find(needle)
    # 条件分岐: `i < 0` を満たす経路を評価する。
    if i < 0:
        return None

    a = max(0, i - 120)
    b = min(len(text), i + window)
    return text[a:b]


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    x = np.array(list(values), dtype=float)
    x = x[np.isfinite(x)]
    # 条件分岐: `x.size == 0` を満たす経路を評価する。
    if x.size == 0:
        return {"n": 0}

    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size >= 2 else 0.0,
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


@dataclass(frozen=True)
class MringFitRow:
    scan: int
    t_utc_hours: float
    d_uas: float
    w_uas: float
    beta1: float
    source_anchor: Dict[str, Any]


_RE_ROW = re.compile(
    r"^\s*(?P<scan>\d+)\s*&\s*(?P<t>\d+(?:\.\d+)?)\s*&\s*(?P<d>\d+(?:\.\d+)?)\s*&\s*(?P<w>\d+(?:\.\d+)?)\s*&\s*(?P<b>\d+(?:\.\d+)?)\s*\\\\"
)


def _parse_table(tex: str, *, source_path: Path) -> List[MringFitRow]:
    label = "\\label{tab:mringfits}"
    lines = tex.splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        # 条件分岐: `label in line` を満たす経路を評価する。
        if label in line:
            start_idx = i
            break

    # 条件分岐: `start_idx is None` を満たす経路を評価する。

    if start_idx is None:
        return []

    # Find \startdata after label.

    startdata_idx = None
    for j in range(start_idx, len(lines)):
        # 条件分岐: `"\\startdata" in lines[j]` を満たす経路を評価する。
        if "\\startdata" in lines[j]:
            startdata_idx = j
            break

    # 条件分岐: `startdata_idx is None` を満たす経路を評価する。

    if startdata_idx is None:
        return []

    enddata_idx = None
    for j in range(startdata_idx, len(lines)):
        # 条件分岐: `"\\enddata" in lines[j]` を満たす経路を評価する。
        if "\\enddata" in lines[j]:
            enddata_idx = j
            break

    # 条件分岐: `enddata_idx is None` を満たす経路を評価する。

    if enddata_idx is None:
        enddata_idx = len(lines)

    rows: List[MringFitRow] = []
    for off, raw in enumerate(lines[startdata_idx:enddata_idx], start=0):
        lineno = (startdata_idx + 1) + off
        m = _RE_ROW.match(raw)
        # 条件分岐: `not m` を満たす経路を評価する。
        if not m:
            continue

        rows.append(
            MringFitRow(
                scan=int(m.group("scan")),
                t_utc_hours=float(m.group("t")),
                d_uas=float(m.group("d")),
                w_uas=float(m.group("w")),
                beta1=float(m.group("b")),
                source_anchor={"path": str(source_path), "line": int(lineno), "label": "tab:mringfits"},
            )
        )

    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()
    default_tex = root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "observations.tex"
    default_outdir = root / "output" / "private" / "eht"

    ap = argparse.ArgumentParser(description="Parse Sgr A* Paper V m-ring fits table (tab:mringfits) from observations.tex.")
    ap.add_argument("--tex", type=str, default=str(default_tex), help="Input TeX (default: Paper V observations.tex)")
    ap.add_argument("--outdir", type=str, default=str(default_outdir), help="Output dir (default: output/private/eht)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    tex_path = Path(args.tex)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_json = outdir / "eht_sgra_mringfits_table_metrics.json"
    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"tex": str(tex_path)},
        "ok": True,
        "extracted": {},
        "derived": {},
        "outputs": {"json": str(out_json)},
    }

    # 条件分岐: `not tex_path.exists()` を満たす経路を評価する。
    if not tex_path.exists():
        payload["ok"] = False
        payload["reason"] = "missing_input_tex"
        _write_json(out_json, payload)
        print(f"[warn] missing input; wrote: {out_json}")
        return 0

    tex = _read_text(tex_path)
    payload["extracted"]["mringfits_anchor_snippet"] = _find_block(tex, "\\label{tab:mringfits}")

    rows = _parse_table(tex, source_path=tex_path)
    payload["extracted"]["rows_n"] = int(len(rows))
    payload["extracted"]["rows"] = [
        {
            "scan": r.scan,
            "t_utc_hours": r.t_utc_hours,
            "d_uas": r.d_uas,
            "w_uas": r.w_uas,
            "beta1": r.beta1,
            "source_anchor": r.source_anchor,
        }
        for r in rows
    ]

    # 条件分岐: `not rows` を満たす経路を評価する。
    if not rows:
        payload["ok"] = False
        payload["reason"] = "no_rows_parsed"
        _write_json(out_json, payload)
        print(f"[warn] no rows; wrote: {out_json}")
        return 0

    payload["derived"] = {
        "d_uas_summary": _summary([r.d_uas for r in rows]),
        "w_uas_summary": _summary([r.w_uas for r in rows]),
        "beta1_summary": _summary([r.beta1 for r in rows]),
        "notes": [
            "This table contains maximum-likelihood m-ring fit parameters for selected 120s scans (Paper V).",
            "It is not the published static-ring diameter used in eht_shadow_compare; treat as a variability/snapshot-fit scale indicator.",
        ],
    }

    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "eht_sgra_mringfits_table_metrics",
                "outputs": [str(out_json.relative_to(root)).replace("\\", "/")],
                "metrics": {"ok": bool(payload.get("ok")), "rows_n": int(len(rows))},
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
