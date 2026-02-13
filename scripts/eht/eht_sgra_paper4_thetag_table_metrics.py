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

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _find_block(text: str, needle: str, *, window: int = 1700) -> Optional[str]:
    i = text.find(needle)
    if i < 0:
        return None
    a = max(0, i - 220)
    b = min(len(text), i + window)
    return text[a:b]


def _unwrap_multirow_cell(s: str) -> str:
    s = str(s).strip()
    m = re.match(r"^\\multirow\{[^}]+\}\{[^}]+\}\{(.+)\}$", s)
    if m:
        return m.group(1).strip()
    return s


def _tex_to_plain(s: str) -> str:
    s = str(s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\\texttt\s*", "", s)
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _parse_subsup_pm(s: str) -> Optional[Tuple[float, float, float]]:
    """
    Parse "x_{-a}^{+b}" into (x, a, b). Returns None if not parseable.
    """
    raw = str(s).strip()
    if not raw:
        return None
    if "\\ldots" in raw or raw == "..." or raw == r"\dots":
        return None
    m = re.search(
        r"(?P<mid>-?\d+(?:\.\d+)?)_\{-(?P<minus>\d+(?:\.\d+)?)\}\^\{\+(?P<plus>\d+(?:\.\d+)?)\}",
        raw,
    )
    if not m:
        return None
    try:
        mid = float(m.group("mid"))
        minus = float(m.group("minus"))
        plus = float(m.group("plus"))
    except Exception:
        return None
    if not (math.isfinite(mid) and math.isfinite(minus) and math.isfinite(plus)):
        return None
    return (mid, abs(minus), abs(plus))


def _sym_sigma(minus: float, plus: float) -> float:
    return 0.5 * (float(minus) + float(plus))


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    x = np.array(list(values), dtype=float)
    x = x[np.isfinite(x)]
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
class ThetaGRow:
    analysis_class: str
    software_plain: str
    software_tex: str
    day: str
    hops_lo: Optional[Tuple[float, float, float]]
    hops_hi: Optional[Tuple[float, float, float]]
    hops_combined: Optional[Tuple[float, float, float]]
    casa_lo: Optional[Tuple[float, float, float]]
    casa_hi: Optional[Tuple[float, float, float]]
    casa_combined: Optional[Tuple[float, float, float]]
    source_anchor: Dict[str, Any]


def _parse_thetag_table(tex: str, *, source_path: Path) -> List[ThetaGRow]:
    label = "\\label{tab:thetag}"
    lines = tex.splitlines()

    label_idx = None
    for i, line in enumerate(lines):
        if label in line:
            label_idx = i
            break
    if label_idx is None:
        return []

    startdata_idx = None
    for j in range(label_idx, len(lines)):
        if "\\startdata" in lines[j]:
            startdata_idx = j
            break
    if startdata_idx is None:
        return []

    enddata_idx = None
    for j in range(startdata_idx, len(lines)):
        if "\\enddata" in lines[j]:
            enddata_idx = j
            break
    if enddata_idx is None:
        enddata_idx = len(lines)

    rows: List[ThetaGRow] = []
    cur_class: Optional[str] = None

    buf = ""
    buf_start_lineno: Optional[int] = None

    def _flush_row(row_text: str, *, lineno: int) -> None:
        nonlocal cur_class, rows
        t = str(row_text).strip()
        if not t:
            return
        if t.startswith("\\cline") or t.startswith("\\hline"):
            return

        # Keep only the row portion before the first "\\" (may have \cline/\hline afterwards).
        if "\\\\" in t:
            t = t.split("\\\\", 1)[0].strip()
        if not t:
            return

        parts = [p.strip() for p in t.split("&")]
        if len(parts) < 4:
            return

        cell0 = _tex_to_plain(_unwrap_multirow_cell(parts[0]))
        if cell0:
            cur_class = cell0
        if not cur_class:
            return

        software_tex = _unwrap_multirow_cell(parts[1]) if len(parts) >= 2 else ""
        software_plain = _tex_to_plain(software_tex)
        day = _tex_to_plain(parts[2]) if len(parts) >= 3 else ""

        # Two main shapes in Paper IV results.tex:
        # - Separate bands: ... & HOPS LO & HOPS HI & (sep) & CASA LO & CASA HI \\
        # - Band-combined (multicolumn): ... & \multicolumn{2}{c}{...} & (sep) & \multicolumn{2}{c}{...} \\
        hops_lo = hops_hi = casa_lo = casa_hi = None
        hops_comb = casa_comb = None

        if len(parts) >= 8:
            hops_lo = _parse_subsup_pm(parts[3])
            hops_hi = _parse_subsup_pm(parts[4])
            casa_lo = _parse_subsup_pm(parts[6])
            casa_hi = _parse_subsup_pm(parts[7])
        elif len(parts) == 6:
            hops_comb = _parse_subsup_pm(parts[3])
            casa_comb = _parse_subsup_pm(parts[5])
        else:
            # Conservative fallback: treat as unparsed/partial row (keep metadata).
            pass

        rows.append(
            ThetaGRow(
                analysis_class=str(cur_class),
                software_plain=str(software_plain),
                software_tex=str(software_tex).strip(),
                day=str(day),
                hops_lo=hops_lo,
                hops_hi=hops_hi,
                hops_combined=hops_comb,
                casa_lo=casa_lo,
                casa_hi=casa_hi,
                casa_combined=casa_comb,
                source_anchor={"path": str(source_path), "line": int(lineno), "label": "tab:thetag"},
            )
        )

    for off, raw in enumerate(lines[startdata_idx + 1 : enddata_idx], start=0):
        lineno = (startdata_idx + 2) + off  # 1-based
        s = raw.strip()
        if not s:
            continue

        if buf_start_lineno is None:
            buf_start_lineno = lineno
        buf = (buf + " " + s).strip()

        if "\\\\" in s:
            _flush_row(buf, lineno=buf_start_lineno)
            buf = ""
            buf_start_lineno = None

    return rows


def _row_mid_for_scatter(row: ThetaGRow, *, pipeline: str) -> Optional[float]:
    if pipeline == "hops":
        if row.hops_combined is not None:
            return float(row.hops_combined[0])
        mids = [row.hops_lo[0]] if row.hops_lo is not None else []
        if row.hops_hi is not None:
            mids.append(row.hops_hi[0])
    else:
        if row.casa_combined is not None:
            return float(row.casa_combined[0])
        mids = [row.casa_lo[0]] if row.casa_lo is not None else []
        if row.casa_hi is not None:
            mids.append(row.casa_hi[0])

    if not mids:
        return None
    return float(np.mean([float(x) for x in mids]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()
    default_tex = root / "data" / "eht" / "sources" / "arxiv_2311.08697" / "results.tex"
    default_shadow = root / "output" / "private" / "eht" / "eht_shadow_compare.json"
    default_outdir = root / "output" / "private" / "eht"

    ap = argparse.ArgumentParser(description="Parse Sgr A* Paper IV theta_g table (tab:thetag).")
    ap.add_argument("--tex", type=str, default=str(default_tex), help="Input TeX (default: Paper IV results.tex)")
    ap.add_argument(
        "--shadow-compare-json",
        type=str,
        default=str(default_shadow),
        help="eht_shadow_compare.json for theta_unit reference (default: output/private/eht/eht_shadow_compare.json)",
    )
    ap.add_argument("--outdir", type=str, default=str(default_outdir), help="Output dir (default: output/private/eht)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    tex_path = Path(args.tex)
    shadow_path = Path(args.shadow_compare_json)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_json = outdir / "eht_sgra_paper4_thetag_table_metrics.json"
    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"tex": str(tex_path), "shadow_compare_json": str(shadow_path)},
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
    payload["extracted"]["thetag_anchor_snippet"] = _find_block(tex, "\\label{tab:thetag}")

    rows = _parse_thetag_table(tex, source_path=tex_path)
    payload["extracted"]["rows_n"] = int(len(rows))

    def _pack_pm(x: Optional[Tuple[float, float, float]]) -> Optional[Dict[str, float]]:
        if x is None:
            return None
        return {"mid": x[0], "minus": x[1], "plus": x[2], "sigma_sym": _sym_sigma(x[1], x[2])}

    payload["extracted"]["rows"] = [
        {
            "analysis_class": r.analysis_class,
            "software": r.software_plain,
            "software_tex": r.software_tex,
            "day": r.day,
            "hops": {"lo": _pack_pm(r.hops_lo), "hi": _pack_pm(r.hops_hi), "combined": _pack_pm(r.hops_combined)},
            "casa": {"lo": _pack_pm(r.casa_lo), "hi": _pack_pm(r.casa_hi), "combined": _pack_pm(r.casa_combined)},
            "source_anchor": r.source_anchor,
        }
        for r in rows
    ]

    if not rows:
        payload["ok"] = False
        payload["reason"] = "no_rows_parsed"
        _write_json(out_json, payload)
        print(f"[warn] no rows; wrote: {out_json}")
        return 0

    theta_unit_uas: Optional[float] = None
    if shadow_path.exists():
        try:
            shadow = _read_json(shadow_path)
            for r in shadow.get("rows") or []:
                if isinstance(r, dict) and r.get("key") == "sgra":
                    theta_unit_uas = float(r.get("theta_unit_uas"))
                    break
        except Exception:
            theta_unit_uas = None

    hops_mids = [_row_mid_for_scatter(r, pipeline="hops") for r in rows]
    casa_mids = [_row_mid_for_scatter(r, pipeline="casa") for r in rows]
    hops_mids_f = [float(x) for x in hops_mids if x is not None and math.isfinite(float(x))]
    casa_mids_f = [float(x) for x in casa_mids if x is not None and math.isfinite(float(x))]

    derived: Dict[str, Any] = {
        "theta_unit_uas_reference": theta_unit_uas,
        "hops_thetag_uas_method_medians_summary": _summary(hops_mids_f),
        "casa_thetag_uas_method_medians_summary": _summary(casa_mids_f),
        "notes": [
            "Paper IV tab:thetag lists theta_g medians and 68% credible intervals across analyses.",
            "We treat the scatter of theta_g medians across methods as a scale indicator of ring→theta_g (and thus κ) systematics (not a direct measurement uncertainty of the published ring diameter).",
            "For entries with separate LO/HI columns, a method-median is computed as the mean of available LO/HI medians.",
        ],
    }

    # Convenience proxy: interpret sigma(theta_g)/theta_g as a κ-scale indicator (κ≈1).
    if theta_unit_uas is not None and math.isfinite(theta_unit_uas) and theta_unit_uas > 0:
        s = derived["hops_thetag_uas_method_medians_summary"].get("std")
        if isinstance(s, (int, float)) and math.isfinite(float(s)):
            derived["kappa_sigma_proxy_paper4_thetag_hops_method_std_over_theta_unit"] = float(s) / float(theta_unit_uas)

    payload["derived"] = derived
    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "eht_sgra_paper4_thetag_table_metrics",
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
