#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _find_first(lines: Sequence[str], pattern: re.Pattern[str]) -> Optional[Tuple[int, re.Match[str]]]:
    for i, line in enumerate(lines, start=1):
        m = pattern.search(line)
        if m:
            return i, m
    return None


def _anchor(path: Path, lineno: int, label: str, line: str, match: Optional[re.Match[str]] = None) -> Dict[str, Any]:
    s = line.rstrip("\n")
    if match is None:
        snippet = s.strip()[:240]
    else:
        center = (match.start() + match.end()) // 2
        half = 120
        start = max(0, center - half)
        end = min(len(s), start + 240)
        start = max(0, end - 240)
        snippet = s[start:end].strip()
    return {"path": str(path), "line": int(lineno), "label": label, "snippet": snippet}


def _maybe_float(x: str) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _strip_tex_math(s: str) -> str:
    # Remove surrounding $...$ and common wrappers.
    s = s.strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    s = s.replace("\\,", "").replace("\\mathrm{mJy}", "").strip()
    return s


def _split_tex_cells(line: str) -> List[str]:
    # Split a TeX table row by '&' while keeping '\&' as a literal ampersand.
    placeholder = "__AMP__PLACEHOLDER__"
    safe = line.replace("\\&", placeholder)
    parts = [p.replace(placeholder, "\\&") for p in safe.split("&")]
    return parts


def _parse_value_cell(cell: str) -> Dict[str, Any]:
    raw = cell.strip()
    struck = False
    inner = raw
    m = re.search(r"\\sout\{(.+?)\}", raw)
    if m:
        struck = True
        inner = m.group(1)

    inner = _strip_tex_math(inner)
    # Patterns like "0.8\\pm0.3" or "0.8 \\pm 0.3" or just "1.0"
    m2 = re.match(r"^\s*([0-9.]+)\s*(?:\\pm\s*([0-9.]+))?\s*$", inner)
    val = None
    err = None
    if m2:
        val = _maybe_float(m2.group(1))
        if m2.group(2) is not None:
            err = _maybe_float(m2.group(2))

    return {"raw": raw, "value_mJy": val, "sigma_mJy": err, "struck_out": struck}


def _parse_percentiles_table(lines: Sequence[str], tex_path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": True}

    label_pat = re.compile(r"\\label\{table1:percentiles\}")
    label_hit = _find_first(lines, label_pat)
    if label_hit is None:
        out["ok"] = False
        out["reason"] = "table_label_not_found"
        return out
    label_line, _ = label_hit

    # Parse from label line forward until end of table.
    rows: List[Dict[str, Any]] = []
    current_year: Optional[str] = None
    current_percentiles: Optional[List[str]] = None

    # Example year header line:
    # "Percentiles $[\\mathrm{mJy}]$: 2017 & 5 \\% & 14 \\% & 50\\% & 86\\% & 95\\% \\\\"
    year_line_pat = re.compile(r"Percentiles.*?:\s*([0-9]{4}[^&]*)\s*&\s*(.+?)\\\\")

    for idx in range(label_line, min(label_line + 200, len(lines))):
        line = lines[idx - 1]
        if "\\end{table" in line:
            break

        if "Percentiles" in line and ":" in line and "&" in line:
            m = year_line_pat.search(line)
            if m:
                current_year = _strip_tex_math(m.group(1)).replace("\\&", "&").strip()
                cols = [c.strip() for c in m.group(2).split("&")]
                current_percentiles = cols
            continue

        if current_year is None or current_percentiles is None:
            continue

        # Interested in the polarization-averaged row only.
        if re.search(r"\bAverage\b", line):
            parts = [p.strip() for p in _split_tex_cells(line)]
            if len(parts) < 2:
                continue
            name = parts[0]
            cells = parts[1:]
            # remove trailing '\\' on last cell
            if cells:
                cells[-1] = cells[-1].replace("\\\\", "").strip()

            # Expect 5 percentile columns
            if len(cells) < 5:
                continue

            row = {
                "year_label": current_year,
                "row_label": _strip_tex_math(name),
                "percentiles": {},
                "source_anchor": _anchor(tex_path, idx, f"percentiles_{current_year}_average_row", line),
            }
            for p_label, cell in zip(current_percentiles[:5], cells[:5], strict=False):
                # Normalize labels like "5 \\%" -> "5"
                pnum = re.sub(r"[^0-9]", "", p_label)
                key = f"p{pnum}" if pnum else p_label
                row["percentiles"][key] = _parse_value_cell(cell)
            rows.append(row)
            continue

        # Final combined row: "2017, 2018 \\& 2019 average & ..."
        if "average" in line and "2017" in line and "2018" in line and "2019" in line and "&" in line:
            parts = [p.strip() for p in _split_tex_cells(line)]
            name = parts[0]
            cells = parts[1:]
            if cells:
                cells[-1] = cells[-1].replace("\\\\", "").strip()
            if len(cells) < 5:
                continue
            row = {
                "year_label": "2017-2019",
                "row_label": _strip_tex_math(name).replace("\\&", "&"),
                "percentiles": {},
                "source_anchor": _anchor(tex_path, idx, "percentiles_2017_2019_average_row", line),
            }
            for p_label, cell in zip(current_percentiles[:5], cells[:5], strict=False):
                pnum = re.sub(r"[^0-9]", "", p_label)
                key = f"p{pnum}" if pnum else p_label
                row["percentiles"][key] = _parse_value_cell(cell)
            rows.append(row)

    if not rows:
        out["ok"] = False
        out["reason"] = "no_rows_parsed"
        out["anchor"] = _anchor(tex_path, label_line, "table1_percentiles_label", lines[label_line - 1])
        return out

    out["rows"] = rows
    out["anchor"] = _anchor(tex_path, label_line, "table1_percentiles_label", lines[label_line - 1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract GRAVITY (2020) Sgr A* NIR flux distribution percentiles (Table 1).")
    ap.add_argument(
        "--tex",
        type=str,
        default=str(_repo_root() / "data" / "eht" / "sources" / "arxiv_2004.07185" / "37717corr.tex"),
        help="Path to the GRAVITY arXiv TeX file.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(_repo_root() / "output" / "eht" / "gravity_sgra_flux_distribution_metrics.json"),
        help="Output JSON path.",
    )
    args = ap.parse_args()

    root = _repo_root()
    tex_path = Path(args.tex)
    out_json = Path(args.out)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"tex": str(tex_path)},
        "ok": True,
        "extracted": {},
        "derived": {},
        "outputs": {"json": str(out_json)},
    }

    if not tex_path.exists():
        payload["ok"] = False
        payload["reason"] = "missing_input_tex"
        _write_json(out_json, payload)
        print(f"[warn] missing input; wrote: {out_json}")
        return 0

    lines = _read_lines(tex_path)

    # Abstract-style statement about median turnover.
    med_pat = re.compile(
        r"turns over at a median flux density of\s*\$\(\s*([0-9.]+)\s*\\pm\s*([0-9.]+)\s*\)~\\mathrm\{mJy\}\$"
    )
    med_hit = _find_first(lines, med_pat)
    if med_hit is not None:
        lineno, m = med_hit
        payload["extracted"]["median_turnover_mJy"] = _maybe_float(m.group(1))
        payload["extracted"]["median_turnover_sigma_mJy"] = _maybe_float(m.group(2))
        payload["extracted"]["median_turnover_anchor"] = _anchor(
            tex_path, lineno, "median_turnover_abstract", lines[lineno - 1], m
        )
    else:
        payload["extracted"]["median_turnover_mJy"] = None
        payload["extracted"]["median_turnover_sigma_mJy"] = None
        payload["extracted"]["median_turnover_missing"] = True

    table = _parse_percentiles_table(lines, tex_path)
    payload["extracted"]["percentiles_table"] = table

    # Convenience derived dicts: pull polarization-averaged percentiles for 2017/2018/2019/2017-2019.
    by_year: Dict[str, Any] = {}
    if table.get("ok") and isinstance(table.get("rows"), list):
        for r in table["rows"]:
            y = r.get("year_label")
            if not isinstance(y, str):
                continue
            per = r.get("percentiles") or {}
            by_year[y] = {
                "p5": per.get("p5"),
                "p14": per.get("p14"),
                "p50": per.get("p50"),
                "p86": per.get("p86"),
                "p95": per.get("p95"),
            }
    payload["derived"]["percentiles_avg_by_year"] = by_year

    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "gravity_sgra_flux_distribution_metrics",
                "outputs": [str(out_json.relative_to(root)).replace("\\", "/")],
                "metrics": {
                    "table_ok": bool(table.get("ok")),
                    "rows_n": len(table.get("rows") or []) if isinstance(table.get("rows"), list) else 0,
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
