#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
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


def _find_snippet(text: str, needle: str, *, window: int = 700) -> Optional[str]:
    i = text.find(needle)
    if i < 0:
        return None
    a = max(0, i - 120)
    b = min(len(text), i + window)
    return text[a:b]


def _mid(lo: float, hi: float) -> float:
    return 0.5 * (float(lo) + float(hi))


def _sigma_var(u_gly: float, *, a: float, b: float, c: float, u0_gly: float) -> float:
    if u_gly <= 0 or u0_gly <= 0:
        return float("nan")
    p = b + c
    num = 1.0 + (4.0 / u0_gly) ** p
    den = 1.0 + (u_gly / u0_gly) ** p
    s2 = (a * a) * ((u_gly / 4.0) ** c) * (num / den)
    return math.sqrt(s2) if s2 >= 0 and math.isfinite(s2) else float("nan")


def _max_sigma_var_in_range(
    *, u_min: float, u_max: float, step: float, a: float, b: float, c: float, u0_gly: float
) -> Dict[str, Any]:
    if step <= 0:
        raise ValueError("step must be positive")
    n = int(round((u_max - u_min) / step)) + 1
    best_u = None
    best = None
    for i in range(max(0, n)):
        u = u_min + i * step
        if u > u_max + 1e-12:
            break
        v = _sigma_var(u, a=a, b=b, c=c, u0_gly=u0_gly)
        if not math.isfinite(v):
            continue
        if best is None or v > best:
            best = v
            best_u = u
    return {"u_min_gly": u_min, "u_max_gly": u_max, "step_gly": step, "max": best, "u_at_max_gly": best_u}


@dataclass(frozen=True)
class PremodelingRow:
    source: str
    a_pct_range: Tuple[float, float]
    b_range: Tuple[float, float]
    u0_gly_range: Tuple[float, float]
    source_anchor: Dict[str, Any]


_RE_RANGE = re.compile(r"\$\[\s*(?P<lo>-?\d+(?:\.\d+)?)\s*,\s*(?P<hi>-?\d+(?:\.\d+)?)\s*\]\$")


def _parse_range_cell(cell: str) -> Optional[Tuple[float, float]]:
    m = _RE_RANGE.search(cell.strip())
    if not m:
        return None
    lo = float(m.group("lo"))
    hi = float(m.group("hi"))
    return (lo, hi) if lo <= hi else (hi, lo)


def _parse_premodeling_table(tex: str, *, source_path: Path) -> List[PremodelingRow]:
    label = "\\label{tab:premodeling}"
    lines = tex.splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if label in line:
            start_idx = i
            break
    if start_idx is None:
        return []

    end_idx = None
    for j in range(start_idx, len(lines)):
        if lines[j].strip().startswith("\\end{deluxetable}"):
            end_idx = j
            break

    block_lines = lines[start_idx : (end_idx if end_idx is not None else len(lines))]
    block_start_lineno = start_idx + 1

    rows: List[PremodelingRow] = []
    for off, raw in enumerate(block_lines):
        lineno = block_start_lineno + off
        line = raw.strip()
        if not line or "&" not in line:
            continue
        if line.startswith("\\") and ("\\startdata" in line or "\\enddata" in line or "\\hline" in line):
            continue
        if line.startswith("\\colhead") or line.startswith("\\tablehead") or line.startswith("\\tablecaption"):
            continue
        if line.startswith("\\tablenotetext") or line.startswith("\\begin{deluxetable}") or line.startswith("\\end{deluxetable}"):
            continue

        # Normalize.
        parts = [p.strip().rstrip("\\").strip() for p in line.split("&")]
        if len(parts) < 4:
            continue

        src = parts[0]
        if src == "\\hline":
            continue

        a_rng = _parse_range_cell(parts[1])
        b_rng = _parse_range_cell(parts[2])
        u0_rng = _parse_range_cell(parts[3])
        if a_rng is None or b_rng is None or u0_rng is None:
            continue

        rows.append(
            PremodelingRow(
                source=src,
                a_pct_range=a_rng,
                b_range=b_rng,
                u0_gly_range=u0_rng,
                source_anchor={"path": str(source_path), "line": int(lineno), "label": "tab:premodeling"},
            )
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()
    default_tex = root / "data" / "eht" / "sources" / "arxiv_2311.09479" / "pre-imaging.tex"
    default_outdir = root / "output" / "private" / "eht"

    ap = argparse.ArgumentParser(description="Extract variability noise model parameters from Sgr A* Paper III (pre-imaging).")
    ap.add_argument("--tex", type=str, default=str(default_tex), help="Input TeX (default: Paper III pre-imaging.tex)")
    ap.add_argument("--outdir", type=str, default=str(default_outdir), help="Output dir (default: output/private/eht)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    tex_path = Path(args.tex)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_json = outdir / "eht_sgra_variability_noise_model_metrics.json"
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

    tex = _read_text(tex_path)

    payload["extracted"]["eq_psd_noise_anchor_snippet"] = _find_snippet(tex, "\\label{eq:PSD_noise}")
    payload["extracted"]["tab_premodeling_anchor_snippet"] = _find_snippet(tex, "\\label{tab:premodeling}")

    rows = _parse_premodeling_table(tex, source_path=tex_path)
    payload["extracted"]["tab_premodeling_rows_n"] = int(len(rows))
    payload["extracted"]["tab_premodeling_rows"] = [
        {
            "source": r.source,
            "a_pct_range": list(r.a_pct_range),
            "b_range": list(r.b_range),
            "u0_gly_range": list(r.u0_gly_range),
            "source_anchor": r.source_anchor,
        }
        for r in rows
    ]

    # Find Sgr A* row.
    sgra_row = None
    for r in rows:
        if r.source.strip() in {"\\sgra", "SgrA", "Sgr A*", "SgrA*"}:
            sgra_row = r
            break
    if sgra_row is None:
        payload["ok"] = False
        payload["reason"] = "sgra_row_not_found"
        payload["available_sources"] = sorted({r.source for r in rows})
        _write_json(out_json, payload)
        print(f"[warn] missing sgra row; wrote: {out_json}")
        return 0

    a_pct_lo, a_pct_hi = sgra_row.a_pct_range
    b_lo, b_hi = sgra_row.b_range
    u0_lo, u0_hi = sgra_row.u0_gly_range

    # Representative choice: use midpoints of reported ranges; set c=2 (used as a representative value in Paper III examples).
    c_rep = 2.0
    a_rep = _mid(a_pct_lo, a_pct_hi) / 100.0
    b_rep = _mid(b_lo, b_hi)
    u0_rep = _mid(u0_lo, u0_hi)

    sigma_u = {}
    for u in (2.0, 3.0, 4.0, 5.0, 6.0):
        sigma_u[f"u{u:g}_gly"] = _sigma_var(u, a=a_rep, b=b_rep, c=c_rep, u0_gly=u0_rep)

    max_mid = _max_sigma_var_in_range(u_min=2.0, u_max=6.0, step=0.01, a=a_rep, b=b_rep, c=c_rep, u0_gly=u0_rep)

    # Range envelope using corner combinations (a,b,u0).
    corner_maxes = []
    for a_pct in (a_pct_lo, a_pct_hi):
        for b in (b_lo, b_hi):
            for u0 in (u0_lo, u0_hi):
                a = a_pct / 100.0
                corner_maxes.append(
                    {
                        "a_pct": a_pct,
                        "b": b,
                        "u0_gly": u0,
                        "max_2to6": _max_sigma_var_in_range(u_min=2.0, u_max=6.0, step=0.01, a=a, b=b, c=c_rep, u0_gly=u0),
                    }
                )

    corner_values = [((c.get("max_2to6") or {}).get("max")) for c in corner_maxes]
    corner_values = [float(v) for v in corner_values if v is not None and math.isfinite(float(v))]

    payload["derived"]["sgra_table_ranges"] = {
        "a_pct_range": [a_pct_lo, a_pct_hi],
        "b_range": [b_lo, b_hi],
        "u0_gly_range": [u0_lo, u0_hi],
        "source_anchor": sgra_row.source_anchor,
    }
    payload["derived"]["representative"] = {
        "c": c_rep,
        "a_fraction_at_4gly": a_rep,
        "b": b_rep,
        "u0_gly": u0_rep,
        "sigma_var": sigma_u,
        "sigma_var_max_2to6_midpoint": max_mid,
        "sigma_var_max_2to6_corners": {
            "n": int(len(corner_values)),
            "min": min(corner_values) if corner_values else None,
            "max": max(corner_values) if corner_values else None,
        },
        "corners": corner_maxes,
    }

    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "eht_sgra_variability_noise_model_metrics",
                "outputs": [str(out_json.relative_to(root)).replace("\\", "/")],
                "metrics": {"ok": bool(payload.get("ok")), "rows_n": int(len(rows))},
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
