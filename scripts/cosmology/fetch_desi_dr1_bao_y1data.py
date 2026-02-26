#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_desi_dr1_bao_y1data.py

Phase 4（宇宙論）/ Step 4.5B.21.4.4（DESI: 公式 cov へ寄せる）の補助:
DESI 2024 VI（BAO cosmology results）論文の Table "Y1data" から、
DESI DR1 の BAO 測定（D_M/r_d, D_H/r_d, r あるいは D_V/r_d）を抽出して
`data/cosmology/desi_dr1_bao_y1data.json` にキャッシュする。

狙い:
- catalog-based ξℓ→peakfit（ε）と、公式の BAO 測定（距離指標）を同一zで突き合わせる
  ための一次ソース（数値＋誤差＋相関）を固定する。

入力（一次ソース）:
- arXiv:2404.03002v3 のソース（TeX）を既に `data/cosmology/sources/` に展開している想定:
  - `data/cosmology/sources/arxiv_2404.03002v3_src/JournalReview_adds_v2_kp7_paper1.tex`

出力（固定）:
- `data/cosmology/desi_dr1_bao_y1data.json`
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BaoVal:
    mean: float
    sigma: float


def _parse_pm(cell: str) -> Optional[BaoVal]:
    t = str(cell).strip()
    # 条件分岐: `t in ("---", "—", "")` を満たす経路を評価する。
    if t in ("---", "—", ""):
        return None
    # Remove TeX $...$ wrappers if present.

    t = t.strip("$").strip()
    # e.g. "13.62 \\pm 0.25"
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*\\pm\s*([+-]?\d+(?:\.\d+)?)\s*$", t)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        raise ValueError(f"unexpected value format: {cell!r}")

    return BaoVal(mean=float(m.group(1)), sigma=float(m.group(2)))


def _strip_tex_math(cell: str) -> str:
    t = str(cell).strip()
    # 条件分岐: `t.startswith("$") and t.endswith("$")` を満たす経路を評価する。
    if t.startswith("$") and t.endswith("$"):
        t = t[1:-1]

    return t.strip()


def _parse_float_or_none(cell: str) -> Optional[float]:
    t = _strip_tex_math(cell)
    # 条件分岐: `t in ("---", "—", "")` を満たす経路を評価する。
    if t in ("---", "—", ""):
        return None

    return float(t)


def _parse_int_commas(cell: str) -> int:
    t = str(cell).strip().replace(",", "")
    return int(t)


def _find_table_y1data(tex_lines: List[str]) -> Tuple[int, int]:
    """
    Return (start_idx, end_idx) inclusive slice for the tabular rows (data lines).
    """
    start = None
    end = None
    for i, line in enumerate(tex_lines):
        # 条件分岐: `"\\label{tab:Y1data}" in line` を満たす経路を評価する。
        if "\\label{tab:Y1data}" in line:
            # Search backwards for begin{tabular} and forward for end{tabular}
            # so we don't depend on exact formatting.
            start = i
            break

    # 条件分岐: `start is None` を満たす経路を評価する。

    if start is None:
        raise ValueError("could not find \\\\label{tab:Y1data} in TeX")

    tab_begin = None
    for j in range(start, -1, -1):
        # 条件分岐: `"\\begin{tabular" in tex_lines[j]` を満たす経路を評価する。
        if "\\begin{tabular" in tex_lines[j]:
            tab_begin = j
            break

    # 条件分岐: `tab_begin is None` を満たす経路を評価する。

    if tab_begin is None:
        raise ValueError("could not find \\\\begin{tabular} for tab:Y1data")

    tab_end = None
    for j in range(start, len(tex_lines)):
        # 条件分岐: `"\\end{tabular" in tex_lines[j]` を満たす経路を評価する。
        if "\\end{tabular" in tex_lines[j]:
            tab_end = j
            break

    # 条件分岐: `tab_end is None` を満たす経路を評価する。

    if tab_end is None:
        raise ValueError("could not find \\\\end{tabular} for tab:Y1data")

    # Data rows: after header hline block until the closing hline/end.
    # We'll parse any row with 8 '&' fields and ending with '\\'.

    return tab_begin, tab_end


def _parse_rows(tex_lines: List[str], i0: int, i1: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(i0, i1 + 1):
        raw = tex_lines[i].strip()
        # 条件分岐: `not raw` を満たす経路を評価する。
        if not raw:
            continue

        # 条件分岐: `raw.startswith("%")` を満たす経路を評価する。

        if raw.startswith("%"):
            continue

        # 条件分岐: `"\\hline" in raw` を満たす経路を評価する。

        if "\\hline" in raw:
            continue

        # 条件分岐: `"&" not in raw or "\\\\" not in raw` を満たす経路を評価する。

        if "&" not in raw or "\\\\" not in raw:
            continue
        # Strip comments

        raw = raw.split("%", 1)[0].strip()
        # Remove trailing '\\' and surrounding whitespace
        raw = re.sub(r"\\\\\s*$", "", raw).strip()
        parts = [p.strip() for p in raw.split("&")]
        # 条件分岐: `len(parts) != 8` を満たす経路を評価する。
        if len(parts) != 8:
            continue
        # Skip header / formatting rows (e.g., multirow lines).

        if not re.match(r"^[0-9][0-9,]*$", parts[2].strip()):
            continue

        tracer = parts[0]
        z_range = _strip_tex_math(parts[1])
        n_tracer = _parse_int_commas(parts[2])
        z_eff = float(_strip_tex_math(parts[3]))
        dmrd = _parse_pm(parts[4])
        dhrd = _parse_pm(parts[5])
        r_or_dvrd = parts[6].strip()
        v_eff = _parse_float_or_none(parts[7])

        r: Optional[float] = None
        dvrd: Optional[BaoVal] = None
        # 条件分岐: `dmrd is None and dhrd is None` を満たす経路を評価する。
        if dmrd is None and dhrd is None:
            dvrd = _parse_pm(r_or_dvrd)
            # 条件分岐: `dvrd is None` を満たす経路を評価する。
            if dvrd is None:
                raise ValueError(f"expected DV/rd in row but got {r_or_dvrd!r} ({tracer})")
        else:
            r = float(_strip_tex_math(r_or_dvrd))

        record: Dict[str, Any] = {
            "tracer": tracer,
            "z_range": z_range,
            "n_tracer": int(n_tracer),
            "z_eff": float(z_eff),
            "v_eff_gpc3": v_eff,
            "dm_over_rd": (None if dmrd is None else {"mean": dmrd.mean, "sigma": dmrd.sigma}),
            "dh_over_rd": (None if dhrd is None else {"mean": dhrd.mean, "sigma": dhrd.sigma}),
            "corr_r_dm_dh": r,
            "dv_over_rd": (None if dvrd is None else {"mean": dvrd.mean, "sigma": dvrd.sigma}),
            "source": {"tex_line_1based": int(i + 1)},
        }
        out.append(record)

    # 条件分岐: `not out` を満たす経路を評価する。

    if not out:
        raise ValueError("no rows parsed for tab:Y1data (format changed?)")

    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract DESI DR1 BAO Y1data table values from arXiv TeX source.")
    ap.add_argument(
        "--tex",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "sources" / "arxiv_2404.03002v3_src" / "JournalReview_adds_v2_kp7_paper1.tex"),
        help="path to TeX source containing tab:Y1data",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "desi_dr1_bao_y1data.json"),
        help="output JSON path",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    tex_path = Path(str(args.tex)).resolve()
    out_path = Path(str(args.out)).resolve()

    # 条件分岐: `not tex_path.exists()` を満たす経路を評価する。
    if not tex_path.exists():
        raise SystemExit(
            "missing TeX source. Expected:\n"
            f"  - {tex_path}\n"
            "Hint: download+extract arXiv source for 2404.03002v3 into data/cosmology/sources/"
        )

    tex_lines = tex_path.read_text(encoding="utf-8", errors="replace").splitlines()
    tab0, tab1 = _find_table_y1data(tex_lines)
    rows = _parse_rows(tex_lines, tab0, tab1)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "arxiv": "2404.03002v3",
            "tex_path": str(tex_path),
            "table_label": "tab:Y1data",
        },
        "rows": rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_path}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
