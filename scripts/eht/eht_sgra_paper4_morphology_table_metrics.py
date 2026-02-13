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


def _find_block(text: str, needle: str, *, window: int = 1600) -> Optional[str]:
    i = text.find(needle)
    if i < 0:
        return None
    a = max(0, i - 200)
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
    m = re.search(r"(?P<mid>-?\d+(?:\.\d+)?)_\{-(?P<minus>\d+(?:\.\d+)?)\}\^\{\+(?P<plus>\d+(?:\.\d+)?)\}", raw)
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
class MorphRow:
    analysis_class: str
    software: str
    day: str
    band: str
    casa_d: Optional[Tuple[float, float, float]]
    hops_d: Optional[Tuple[float, float, float]]
    source_anchor: Dict[str, Any]


def _parse_table(tex: str, *, source_path: Path) -> List[MorphRow]:
    label = "\\label{tab:SgrAMorphology}"
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

    cur_class: Optional[str] = None
    cur_software: Optional[str] = None
    rows: List[MorphRow] = []

    for off, raw in enumerate(lines[startdata_idx + 1 : enddata_idx], start=0):
        lineno = (startdata_idx + 2) + off  # 1-based
        s = raw.strip()
        if not s:
            continue
        if s.startswith("\\cline") or s.startswith("\\hline"):
            continue
        if "\\\\" not in s:
            continue
        s = s.replace("\\\\", "").strip()
        parts = [p.strip() for p in s.split("&")]
        if len(parts) < 15:
            continue

        a0 = _tex_to_plain(_unwrap_multirow_cell(parts[0]))
        if a0:
            cur_class = a0
        s0 = _tex_to_plain(_unwrap_multirow_cell(parts[1]))
        if s0:
            cur_software = s0
        if not cur_class or not cur_software:
            continue

        day = _tex_to_plain(parts[2])
        band = _tex_to_plain(parts[3])

        casa_d = _parse_subsup_pm(parts[4])
        hops_d = _parse_subsup_pm(parts[10])

        rows.append(
            MorphRow(
                analysis_class=str(cur_class),
                software=str(cur_software),
                day=str(day),
                band=str(band),
                casa_d=casa_d,
                hops_d=hops_d,
                source_anchor={"path": str(source_path), "line": int(lineno), "label": "tab:SgrAMorphology"},
            )
        )

    return rows


def _select_rows(rows: List[MorphRow], *, analysis_class: str) -> List[MorphRow]:
    return [r for r in rows if r.analysis_class == analysis_class]


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()
    default_tex = root / "data" / "eht" / "sources" / "arxiv_2311.08697" / "results.tex"
    default_shadow = root / "output" / "private" / "eht" / "eht_shadow_compare.json"
    default_outdir = root / "output" / "private" / "eht"

    ap = argparse.ArgumentParser(description="Parse Sgr A* Paper IV morphology table (tab:SgrAMorphology) for diameter scatter proxies.")
    ap.add_argument("--tex", type=str, default=str(default_tex), help="Input TeX (default: Paper IV results.tex)")
    ap.add_argument(
        "--shadow-compare-json",
        type=str,
        default=str(default_shadow),
        help="eht_shadow_compare.json for reference ring diameter (default: output/private/eht/eht_shadow_compare.json)",
    )
    ap.add_argument("--outdir", type=str, default=str(default_outdir), help="Output dir (default: output/private/eht)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    tex_path = Path(args.tex)
    shadow_path = Path(args.shadow_compare_json)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_json = outdir / "eht_sgra_paper4_morphology_table_metrics.json"

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
    payload["extracted"]["morphology_anchor_snippet"] = _find_block(tex, "\\label{tab:SgrAMorphology}")

    rows = _parse_table(tex, source_path=tex_path)
    payload["extracted"]["rows_n"] = int(len(rows))
    payload["extracted"]["rows"] = [
        {
            "analysis_class": r.analysis_class,
            "software": r.software,
            "day": r.day,
            "band": r.band,
            "casa_d": None
            if r.casa_d is None
            else {"mid": r.casa_d[0], "minus": r.casa_d[1], "plus": r.casa_d[2], "sigma_sym": _sym_sigma(r.casa_d[1], r.casa_d[2])},
            "hops_d": None
            if r.hops_d is None
            else {"mid": r.hops_d[0], "minus": r.hops_d[1], "plus": r.hops_d[2], "sigma_sym": _sym_sigma(r.hops_d[1], r.hops_d[2])},
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

    ring_ref_uas: Optional[float] = None
    if shadow_path.exists():
        try:
            shadow = _read_json(shadow_path)
            for r in shadow.get("rows") or []:
                if isinstance(r, dict) and r.get("key") == "sgra":
                    ring_ref_uas = float(r.get("ring_diameter_obs_uas"))
                    break
        except Exception:
            ring_ref_uas = None

    def _gather(values: List[MorphRow], *, pipeline: str) -> List[float]:
        out: List[float] = []
        for r in values:
            x = r.hops_d if pipeline == "hops" else r.casa_d
            if x is None:
                continue
            out.append(float(x[0]))
        return out

    imaging = _select_rows(rows, analysis_class="Imaging")
    snapshot = _select_rows(rows, analysis_class="Snapshot")
    fulltrack = _select_rows(rows, analysis_class="Full-track")

    derived: Dict[str, Any] = {
        "ring_reference_uas": ring_ref_uas,
        "imaging": {
            "hops_dhat_uas_summary": _summary(_gather(imaging, pipeline="hops")),
            "casa_dhat_uas_summary": _summary(_gather(imaging, pipeline="casa")),
        },
        "snapshot": {
            "hops_dhat_uas_summary": _summary(_gather(snapshot, pipeline="hops")),
            "casa_dhat_uas_summary": _summary(_gather(snapshot, pipeline="casa")),
        },
        "full_track": {
            "hops_dhat_uas_summary": _summary(_gather(fulltrack, pipeline="hops")),
            "casa_dhat_uas_summary": _summary(_gather(fulltrack, pipeline="casa")),
        },
        "notes": [
            "Paper IV tab:SgrAMorphology reports debiased ring diameter d_hat and other morphology parameters for CASA/HOPS pipelines.",
            "We treat the scatter of d_hat medians across methods as a scale indicator for ring-diameter systematics (not a direct measurement uncertainty of the published ring diameter).",
        ],
    }

    # Convenience proxies (relative to the ring diameter used in eht_shadow_compare, if available).
    if ring_ref_uas is not None and math.isfinite(ring_ref_uas) and ring_ref_uas > 0:
        s = derived["imaging"]["hops_dhat_uas_summary"].get("std")
        if isinstance(s, (int, float)) and math.isfinite(float(s)):
            derived["kappa_sigma_proxy_paper4_morphology_hops_imaging_dhat_std"] = float(s) / float(ring_ref_uas)

    payload["derived"] = derived
    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "eht_sgra_paper4_morphology_table_metrics",
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
