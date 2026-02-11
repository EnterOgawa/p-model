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


def _anchor(path: Path, line: int, *, label: str, snippet: str) -> Dict[str, Any]:
    return {"path": str(path), "line": int(line), "label": label, "snippet": snippet.strip()[:240]}


_RE_MULTICOL = re.compile(r"\\multicolumn\{\d+\}\{[^}]*\}\{(?P<content>[^}]*)\}")
_RE_FLOAT = re.compile(r"(?P<num>-?\d+(?:\.\d+)?)")


def _parse_float_list(cell: str) -> List[float]:
    s = cell.strip()
    m = _RE_MULTICOL.search(s)
    if m:
        s = m.group("content")
    s = s.replace("\\,", " ").replace("\\ ", " ").replace(",", " ")
    out: List[float] = []
    for mm in _RE_FLOAT.finditer(s):
        try:
            out.append(float(mm.group("num")))
        except Exception:
            continue
    return out


def _extract_sigma_mu_3h_table(*, s4_path: Path, lines: Sequence[str]) -> Dict[str, Any]:
    label = "\\label{tab:sigma_mu_3h}"
    label_idx = None
    for i, line in enumerate(lines):
        if label in line:
            label_idx = i
            break
    if label_idx is None:
        return {"ok": False, "reason": "label_not_found", "label": label}

    begin_idx = None
    for j in range(label_idx, -1, -1):
        if "\\begin{table}" in lines[j]:
            begin_idx = j
            break
    if begin_idx is None:
        begin_idx = max(0, label_idx - 200)

    end_idx = None
    for j in range(label_idx, min(len(lines), label_idx + 200)):
        if "\\end{table}" in lines[j]:
            end_idx = j
            break
    if end_idx is None:
        end_idx = min(len(lines) - 1, label_idx + 200)

    table_lines = list(lines[begin_idx : end_idx + 1])
    header_anchor = _anchor(
        s4_path,
        line=label_idx + 1,
        label="wielgus2022_tab_sigma_mu_3h",
        snippet=lines[label_idx],
    )

    current_instrument: Optional[str] = None
    current_pipeline: Optional[str] = None
    rows: List[Dict[str, Any]] = []

    section_pat = re.compile(r"\\multicolumn\{6\}\{c\}\{(?P<name>[^}]*)\}")

    for k, raw in enumerate(table_lines, start=begin_idx + 1):
        line = raw.strip()
        if not line:
            continue

        msec = section_pat.search(line)
        if msec:
            name = msec.group("name").strip()
            if name.startswith("ALMA "):
                current_instrument = "ALMA"
                current_pipeline = name.replace("ALMA", "").strip()
            elif name == "SMA":
                current_instrument = "SMA"
                current_pipeline = None
            else:
                current_instrument = name
                current_pipeline = None
            continue

        if "&" not in line:
            continue
        if "\\hline" in line or line.startswith("Band"):
            continue

        parts = [p.strip().rstrip("\\").strip() for p in line.split("&")]
        if not parts:
            continue

        band = parts[0].split()[0] if parts[0] else ""
        if not band or current_instrument is None:
            continue

        out_row: Dict[str, Any] = {
            "instrument": current_instrument,
            "pipeline": current_pipeline,
            "band": band,
            "source": {"path": str(s4_path), "line": int(k), "raw": raw.rstrip("\n")},
        }

        if current_instrument == "ALMA":
            # Columns: Band, Apr6, Apr7 (multicolumn list), Apr11
            if len(parts) < 4:
                continue
            out_row["apr6"] = _parse_float_list(parts[1])
            out_row["apr7"] = _parse_float_list(parts[2])
            out_row["apr11"] = _parse_float_list(parts[3])
        elif current_instrument == "SMA":
            # Columns: Band, Apr5, Apr6, Apr7, Apr10, Apr11
            if len(parts) < 6:
                continue
            out_row["apr5"] = _parse_float_list(parts[1])
            out_row["apr6"] = _parse_float_list(parts[2])
            out_row["apr7"] = _parse_float_list(parts[3])
            out_row["apr10"] = _parse_float_list(parts[4])
            out_row["apr11"] = _parse_float_list(parts[5])
        else:
            continue

        rows.append(out_row)

    return {
        "ok": True,
        "source_anchor": header_anchor,
        "rows_n": len(rows),
        "rows": rows,
    }


_RE_D3H = re.compile(
    r"\\left\s*\(\s*\\sigma/\\mu\s*\\right\)\s*_\{\\rm\s*3h\}\s*=\s*"
    r"(?P<v>\d+(?:\.\d+)?)\s*\^\{\+?(?P<plus>\d+(?:\.\d+)?)\}_\{-(?P<minus>\d+(?:\.\d+)?)\}",
)


def _extract_drw_predicted_sigma_mu_3h(*, s4_path: Path, lines: Sequence[str]) -> Dict[str, Any]:
    for i, raw in enumerate(lines, start=1):
        m = _RE_D3H.search(raw)
        if not m:
            continue
        return {
            "ok": True,
            "value": float(m.group("v")),
            "plus": float(m.group("plus")),
            "minus": float(m.group("minus")),
            "source_anchor": _anchor(s4_path, line=i, label="wielgus2022_drw_pred_sigma_mu_3h", snippet=raw),
        }
    return {"ok": False, "reason": "pattern_not_found"}


_RE_PM3 = re.compile(r"\$\s*(?P<v>\d+(?:\.\d+)?)\^\{\+?(?P<plus>\d+(?:\.\d+)?)\}_\{-(?P<minus>\d+(?:\.\d+)?)\}\s*\$")


def _parse_pm3(cell: str) -> Optional[Tuple[float, float, float]]:
    m = _RE_PM3.search(cell.strip())
    if not m:
        return None
    return float(m.group("v")), float(m.group("plus")), float(m.group("minus"))


def _extract_gpresults_tau(*, s5_path: Path, lines: Sequence[str]) -> Dict[str, Any]:
    # Parse tau (hours) from a subset of rows in Table tab:GPresults.
    wanted = {
        "A1 all HI": "A1_all_HI",
        "FULL HI": "FULL_HI",
        "2005-2019": "2005-2019",
    }
    out_rows: Dict[str, Any] = {}

    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        if "&" not in line or "\\hline" in line:
            continue
        # Row name sits before first '&'
        name_cell = line.split("&", 1)[0].strip()
        name_clean = name_cell.replace("$^{a}$", "").replace("$^{b}$", "").strip()
        key = None
        for prefix, k in wanted.items():
            if name_clean.startswith(prefix):
                key = k
                break
        if key is None:
            continue

        parts = [p.strip().rstrip("\\").strip() for p in line.split("&")]
        if len(parts) < 4:
            continue
        tau_cell = parts[3]
        pm = _parse_pm3(tau_cell)
        if pm is None:
            continue
        tau, plus, minus = pm
        out_rows[key] = {
            "tau_h": float(tau),
            "tau_plus_h": float(plus),
            "tau_minus_h": float(minus),
            "row_name_raw": name_cell,
            "source": {"path": str(s5_path), "line": int(i), "raw": raw.rstrip("\n")},
        }

    ok = bool(out_rows)
    return {"ok": ok, "rows": out_rows, "rows_n": len(out_rows), "label": "tab:GPresults"}


def _summary_stats(values: Sequence[float]) -> Dict[str, Any]:
    xs = [float(x) for x in values if isinstance(x, (int, float))]
    xs = [x for x in xs if x == x]  # drop NaN
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    if n == 0:
        return {"n": 0}
    med = xs_sorted[n // 2] if (n % 2 == 1) else 0.5 * (xs_sorted[n // 2 - 1] + xs_sorted[n // 2])
    mean = sum(xs_sorted) / n
    return {"n": n, "min": xs_sorted[0], "max": xs_sorted[-1], "mean": mean, "median": med}


_MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def _parse_ymd(date_raw: str) -> Optional[str]:
    # Expected formats: "2005 June 4", "2017 October 11a", etc.
    tokens = date_raw.strip().split()
    if len(tokens) < 3:
        return None
    if not tokens[0].isdigit():
        return None
    year = int(tokens[0])
    month = _MONTHS.get(tokens[1])
    if month is None:
        return None
    m = re.match(r"(?P<day>\d+)", tokens[2])
    if m is None:
        return None
    day = int(m.group("day"))
    if not (1 <= day <= 31):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def _clean_cell(cell: str) -> str:
    return cell.strip().rstrip("\\").strip()


_RE_CITET = re.compile(r"\\citet\{(?P<key>[^}]+)\}")


def _normalize_reference(reference_raw: str) -> Dict[str, Any]:
    m = _RE_CITET.search(reference_raw)
    if not m:
        return {"raw": reference_raw, "key": None}
    return {"raw": reference_raw, "key": m.group("key").strip()}


def _normalize_array(array_raw: str) -> str:
    # e.g., "CARMA$^a$" -> "CARMA"
    s = array_raw.strip()
    s = re.sub(r"\$\^.*?\$", "", s)
    s = s.replace("{", "").replace("}", "").strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()[0] if s else array_raw.strip()


def _extract_detections_other_papers_table(*, s4_path: Path, lines: Sequence[str]) -> Dict[str, Any]:
    label = "\\label{tab:detections_other_papers}"
    label_idx = None
    for i, line in enumerate(lines):
        if label in line:
            label_idx = i
            break
    if label_idx is None:
        return {"ok": False, "reason": "label_not_found", "label": label}

    # Look for startdata/enddata around the label (deluxetable).
    start_idx = None
    for j in range(label_idx, min(len(lines), label_idx + 120)):
        if "\\startdata" in lines[j]:
            start_idx = j
            break
    if start_idx is None:
        return {"ok": False, "reason": "startdata_not_found", "label": label}

    end_idx = None
    for j in range(start_idx, min(len(lines), start_idx + 800)):
        if "\\enddata" in lines[j]:
            end_idx = j
            break
    if end_idx is None:
        return {"ok": False, "reason": "enddata_not_found", "label": label}

    header_anchor = _anchor(s4_path, line=label_idx + 1, label="wielgus2022_tab_detections_other_papers", snippet=lines[label_idx])

    rows: List[Dict[str, Any]] = []
    cur_ref_raw: Optional[str] = None
    cur_array_raw: Optional[str] = None

    for lineno, raw in enumerate(lines[start_idx + 1 : end_idx], start=start_idx + 2):
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if line.startswith("\\hline") or line.startswith("\\cline"):
            continue
        if "&" not in line:
            continue
        parts = [_clean_cell(p) for p in raw.split("&")]
        if len(parts) < 8:
            continue

        ref_raw = parts[0] or cur_ref_raw
        array_raw = parts[1] or cur_array_raw
        date_raw = parts[2]
        dur_raw = parts[3]
        sigma_over_mu_raw = parts[6]

        if ref_raw is None or array_raw is None:
            continue
        cur_ref_raw = ref_raw
        cur_array_raw = array_raw

        ymd = _parse_ymd(date_raw)
        try:
            dur_h = float(dur_raw)
        except Exception:
            continue
        try:
            sigma_over_mu = float(sigma_over_mu_raw)
        except Exception:
            continue

        ref_norm = _normalize_reference(ref_raw)
        rows.append(
            {
                "reference": ref_norm,
                "array_raw": array_raw,
                "array": _normalize_array(array_raw),
                "date_raw": date_raw,
                "date_ymd": ymd,
                "duration_h": float(dur_h),
                "sigma_over_mu": float(sigma_over_mu),
                "source": {"path": str(s4_path), "line": int(lineno), "raw": raw.rstrip("\n")},
            }
        )

    return {"ok": True, "source_anchor": header_anchor, "rows_n": len(rows), "rows": rows}


def _paper5_historical_distribution_candidate(
    detections_other_papers: Dict[str, Any],
    *,
    deltaT_hours: float,
    date_cutoff_ymd: str,
    duration_h_min: float,
) -> Dict[str, Any]:
    rows = detections_other_papers.get("rows") if isinstance(detections_other_papers.get("rows"), list) else []

    selected: List[Dict[str, Any]] = []
    for r in rows:
        ymd = r.get("date_ymd")
        dur = r.get("duration_h")
        if not isinstance(ymd, str) or not isinstance(dur, (int, float)):
            continue
        if ymd > date_cutoff_ymd:
            continue
        if float(dur) < float(duration_h_min):
            continue
        selected.append(r)

    curves: List[Dict[str, Any]] = []
    segments: List[Dict[str, Any]] = []
    proxy_values: List[float] = []

    segments_by_array: Dict[str, int] = {}
    curves_by_array: Dict[str, int] = {}

    for r in selected:
        dur = float(r["duration_h"])
        sigma_over_mu = float(r["sigma_over_mu"])
        k = int(math.floor(dur / float(deltaT_hours)))
        if k <= 0:
            continue
        arr = str(r.get("array") or "")
        if arr:
            segments_by_array[arr] = int(segments_by_array.get(arr, 0) + k)
            curves_by_array[arr] = int(curves_by_array.get(arr, 0) + 1)

        curve_item = {
            "date_ymd": r.get("date_ymd"),
            "date_raw": r.get("date_raw"),
            "array": r.get("array"),
            "reference_key": (r.get("reference") or {}).get("key") if isinstance(r.get("reference"), dict) else None,
            "duration_h": dur,
            "segments_n": k,
            "sigma_over_mu_proxy_full_duration": sigma_over_mu,
            "source": r.get("source"),
        }
        curves.append(curve_item)
        for j in range(1, k + 1):
            segments.append(
                {
                    "segment_index": int(j),
                    "date_ymd": r.get("date_ymd"),
                    "array": r.get("array"),
                    "reference_key": (r.get("reference") or {}).get("key") if isinstance(r.get("reference"), dict) else None,
                    "sigma_over_mu_proxy_full_duration": sigma_over_mu,
                    "duration_h": dur,
                    "source": r.get("source"),
                }
            )
            proxy_values.append(sigma_over_mu)

    multi = [c for c in curves if int(c.get("segments_n") or 0) >= 2]
    multi = sorted(multi, key=lambda x: float(x.get("duration_h") or 0.0), reverse=True)

    return {
        "ok": True,
        "note": (
            "This reconstructs the *composition* and sample count (n) of the Paper V 'historical distribution' "
            "candidate using Wielgus+2022 Table tab:detections_other_papers: select duration>=3h and date<=cutoff, "
            "then take floor(duration/ΔT) non-overlapping 3h segments. The per-segment modulation index values are NOT "
            "available from this table; sigma/mu over the full duration is stored as a proxy only."
        ),
        "deltaT_hours": float(deltaT_hours),
        "duration_h_min": float(duration_h_min),
        "date_cutoff_ymd": str(date_cutoff_ymd),
        "curves_n": int(len(curves)),
        "segments_n": int(len(segments)),
        "segments_by_array": segments_by_array,
        "curves_by_array": curves_by_array,
        "curves_with_multiple_segments_n": int(len(multi)),
        "curves_with_multiple_segments_top": multi[:10],
        "proxy_sigma_over_mu_values_n": int(len(proxy_values)),
        "proxy_sigma_over_mu_summary": _summary_stats(proxy_values),
        "curves": curves,
        "segments": segments,
    }


def _ks_two_sample_d(sample_a: Sequence[float], sample_b: Sequence[float]) -> Optional[float]:
    a = sorted(float(x) for x in sample_a if isinstance(x, (int, float)))
    b = sorted(float(x) for x in sample_b if isinstance(x, (int, float)))
    n = len(a)
    m = len(b)
    if n == 0 or m == 0:
        return None

    i = 0
    j = 0
    d = 0.0
    while i < n or j < m:
        if j >= m or (i < n and a[i] <= b[j]):
            t = a[i]
        else:
            t = b[j]

        while i < n and a[i] <= t:
            i += 1
        while j < m and b[j] <= t:
            j += 1

        fa = i / n
        fb = j / m
        d = max(d, abs(fa - fb))
    return float(d)


def _ks_qks(lam: float, *, max_terms: int = 200) -> float:
    # Q_KS(λ) = 2 Σ_{k=1..∞} (-1)^{k-1} exp(-2 k^2 λ^2)
    if lam <= 0.0:
        return 1.0
    s = 0.0
    for k in range(1, max_terms + 1):
        term = math.exp(-2.0 * (k * k) * (lam * lam))
        s += (term if (k % 2 == 1) else -term)
        if term < 1e-12:
            break
    q = 2.0 * s
    if q < 0.0:
        return 0.0
    if q > 1.0:
        return 1.0
    return float(q)


def _ks_pvalue_2samp_asymptotic(sample_a: Sequence[float], sample_b: Sequence[float]) -> Optional[float]:
    d = _ks_two_sample_d(sample_a, sample_b)
    if d is None:
        return None
    n = len([x for x in sample_a if isinstance(x, (int, float))])
    m = len([x for x in sample_b if isinstance(x, (int, float))])
    if n <= 0 or m <= 0:
        return None
    n_eff = (n * m) / (n + m)
    if n_eff <= 0:
        return None
    sq = math.sqrt(n_eff)
    lam = (sq + 0.12 + 0.11 / sq) * float(d)
    return _ks_qks(lam)


def _ks_two_sample_dcrit(alpha: float, n: int, m: int) -> Optional[float]:
    if n <= 0 or m <= 0:
        return None
    c = None
    if abs(alpha - 0.10) < 1e-12:
        c = 1.22
    elif abs(alpha - 0.05) < 1e-12:
        c = 1.36
    elif abs(alpha - 0.01) < 1e-12:
        c = 1.63
    elif abs(alpha - 0.001) < 1e-12:
        c = 1.95
    if c is None:
        return None
    return float(c * math.sqrt((n + m) / (n * m)))


def _select_paper5_7samples_candidate(
    sigma_mu_table: Dict[str, Any],
    *,
    s4_path: Path,
) -> Dict[str, Any]:
    rows = sigma_mu_table.get("rows") if isinstance(sigma_mu_table.get("rows"), list) else []
    idx: Dict[Tuple[str, Optional[str], str], Dict[str, Any]] = {}
    for r in rows:
        inst = r.get("instrument")
        band = r.get("band")
        if not isinstance(inst, str) or not isinstance(band, str):
            continue
        pipe = r.get("pipeline") if (r.get("pipeline") is None or isinstance(r.get("pipeline"), str)) else None
        idx[(inst, pipe, band)] = r

    alma_a1_hi = idx.get(("ALMA", "A1", "HI"))
    sma_hi = idx.get(("SMA", None, "HI"))
    if not isinstance(alma_a1_hi, dict) or not isinstance(sma_hi, dict):
        return {"ok": False, "reason": "required_rows_not_found"}

    alma_apr6 = alma_a1_hi.get("apr6")
    alma_apr7 = alma_a1_hi.get("apr7")
    alma_apr11 = alma_a1_hi.get("apr11")
    sma_apr5 = sma_hi.get("apr5")
    sma_apr10 = sma_hi.get("apr10")

    if not all(isinstance(v, list) for v in (alma_apr6, alma_apr7, alma_apr11, sma_apr5, sma_apr10)):
        return {"ok": False, "reason": "invalid_row_cells"}

    samples: List[Dict[str, Any]] = []

    def _add(inst: str, pipe: Optional[str], band: str, day: int, vals: List[float], *, source: Dict[str, Any]) -> None:
        for j, x in enumerate(vals, start=1):
            samples.append(
                {
                    "instrument": inst,
                    "pipeline": pipe,
                    "band": band,
                    "date_utc": f"2017-04-{day:02d}",
                    "sigma_over_mu_3h": float(x),
                    "segment_index": int(j),
                    "source": source,
                }
            )

    _add("ALMA", "A1", "HI", 6, alma_apr6, source=alma_a1_hi.get("source") or {})
    _add("ALMA", "A1", "HI", 7, alma_apr7, source=alma_a1_hi.get("source") or {})
    _add("ALMA", "A1", "HI", 11, alma_apr11, source=alma_a1_hi.get("source") or {})
    _add("SMA", None, "HI", 5, sma_apr5, source=sma_hi.get("source") or {})
    _add("SMA", None, "HI", 10, sma_apr10, source=sma_hi.get("source") or {})

    values = [s["sigma_over_mu_3h"] for s in samples]
    flare_values = [s["sigma_over_mu_3h"] for s in samples if s.get("date_utc") == "2017-04-11"]
    nonflare_values = [s["sigma_over_mu_3h"] for s in samples if s.get("date_utc") != "2017-04-11"]

    return {
        "ok": True,
        "note": (
            "Inferred 7-sample composition to match Paper V statement (2017 provides 7 samples): "
            "ALMA A1 HI Apr6 (1) + Apr7 (3) + Apr11 (1) + SMA HI Apr5 (1) + Apr10 (1) = 7."
        ),
        "samples_n": len(samples),
        "samples": samples,
        "summary_all": _summary_stats(values),
        "summary_flare_day_2017_04_11": _summary_stats(flare_values),
        "summary_nonflare_days": _summary_stats(nonflare_values),
        "selection_anchor": _anchor(
            s4_path,
            line=(sigma_mu_table.get("source_anchor") or {}).get("line", 0),
            label="wielgus2022_sigma_mu_3h_selection_basis",
            snippet="See Table tab:sigma_mu_3h; selection inferred for Paper V 7 samples (Apr 5--11).",
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()

    default_s4 = root / "data" / "eht" / "sources" / "arxiv_2207.06829" / "S4_Variability.tex"
    default_s5 = root / "data" / "eht" / "sources" / "arxiv_2207.06829" / "S5_Modeling.tex"
    default_out = root / "output" / "eht" / "wielgus2022_m3_observed_metrics.json"

    ap = argparse.ArgumentParser(description="Extract Wielgus+2022 (arXiv:2207.06829) M3-related observed metrics for EHT Sgr A* Paper V context.")
    ap.add_argument("--s4-variability-tex", type=str, default=str(default_s4))
    ap.add_argument("--s5-modeling-tex", type=str, default=str(default_s5))
    ap.add_argument("--out", type=str, default=str(default_out))
    args = ap.parse_args(list(argv) if argv is not None else None)

    s4_path = Path(args.s4_variability_tex)
    s5_path = Path(args.s5_modeling_tex)
    out_path = Path(args.out)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "inputs": {
            "s4_variability_tex": str(s4_path),
            "s5_modeling_tex": str(s5_path),
        },
        "extracted": {},
        "derived": {},
        "outputs": {"json": str(out_path)},
    }

    missing = [str(p) for p in (s4_path, s5_path) if not p.exists()]
    if missing:
        payload["ok"] = False
        payload["reason"] = "missing_inputs"
        payload["missing"] = missing
        _write_json(out_path, payload)
        print(f"[warn] missing inputs: {missing}")
        return 2

    s4_lines = _read_lines(s4_path)
    s5_lines = _read_lines(s5_path)

    sigma_mu = _extract_sigma_mu_3h_table(s4_path=s4_path, lines=s4_lines)
    drw_pred = _extract_drw_predicted_sigma_mu_3h(s4_path=s4_path, lines=s4_lines)
    detections_other = _extract_detections_other_papers_table(s4_path=s4_path, lines=s4_lines)
    gp_tau = _extract_gpresults_tau(s5_path=s5_path, lines=s5_lines)

    payload["extracted"] = {
        "sigma_mu_3h_table": sigma_mu,
        "drw_predicted_sigma_over_mu_3h": drw_pred,
        "detections_other_papers_table": detections_other,
        "gpresults_tau_hours": gp_tau,
    }

    if bool(sigma_mu.get("ok")):
        payload["derived"]["paper5_m3_2017_7sample_candidate"] = _select_paper5_7samples_candidate(sigma_mu, s4_path=s4_path)
    else:
        payload["derived"]["paper5_m3_2017_7sample_candidate"] = {"ok": False, "reason": "sigma_mu_table_not_ok"}

    # deltaT proxy (Paper V uses ΔT=3h); connect to tau to indicate scale.
    deltaT = 3.0
    tau_rows = (gp_tau.get("rows") or {}) if isinstance(gp_tau.get("rows"), dict) else {}
    ratios = {}
    for key, rr in tau_rows.items():
        tau = rr.get("tau_h")
        if isinstance(tau, (int, float)) and tau > 0:
            ratios[key] = float(deltaT / float(tau))
    payload["derived"]["deltaT_hours"] = deltaT
    payload["derived"]["deltaT_over_tau_by_row"] = ratios

    if bool(detections_other.get("ok")):
        # Candidate: pre-EHT historical record (cut at end of EHT 2017 campaign).
        payload["derived"]["paper5_m3_historical_distribution_candidate_pre_eht_2017_apr11"] = _paper5_historical_distribution_candidate(
            detections_other,
            deltaT_hours=deltaT,
            date_cutoff_ymd="2017-04-11",
            duration_h_min=3.0,
        )
        # Alternative: include all of year 2017 (captures July 2017 SMA long tracks).
        payload["derived"]["paper5_m3_historical_distribution_candidate_2017_inclusive"] = _paper5_historical_distribution_candidate(
            detections_other,
            deltaT_hours=deltaT,
            date_cutoff_ymd="2017-12-31",
            duration_h_min=3.0,
        )
    else:
        payload["derived"]["paper5_m3_historical_distribution_candidate_pre_eht_2017_apr11"] = {
            "ok": False,
            "reason": "detections_other_papers_table_not_ok",
        }
        payload["derived"]["paper5_m3_historical_distribution_candidate_2017_inclusive"] = {
            "ok": False,
            "reason": "detections_other_papers_table_not_ok",
        }

    hist = payload["derived"]["paper5_m3_historical_distribution_candidate_pre_eht_2017_apr11"] or {}
    hist_ok = bool(hist.get("ok")) and int(hist.get("segments_n") or 0) == 42

    # KS sanity check between (i) 2017 7-sample candidate and (ii) reconstructed historical distribution.
    # NOTE: historical per-segment mi3 values are not available in tab:detections_other_papers; we therefore use
    # sigma/mu over the full duration as a proxy repeated per 3h segment. This is only a stress-test.
    ks_block: Dict[str, Any] = {"ok": False}
    try:
        sel = payload["derived"]["paper5_m3_2017_7sample_candidate"] or {}
        hist_seg = hist.get("segments") if isinstance(hist, dict) else None
        if isinstance(sel, dict) and bool(sel.get("ok")) and isinstance(hist_seg, list):
            x7 = [float(s.get("sigma_over_mu_3h")) for s in (sel.get("samples") or []) if isinstance(s, dict) and isinstance(s.get("sigma_over_mu_3h"), (int, float))]
            yh = [float(s.get("sigma_over_mu_proxy_full_duration")) for s in hist_seg if isinstance(s, dict) and isinstance(s.get("sigma_over_mu_proxy_full_duration"), (int, float))]
            d = _ks_two_sample_d(x7, yh)
            p = _ks_pvalue_2samp_asymptotic(x7, yh)
            dcrit = _ks_two_sample_dcrit(0.01, len(x7), len(yh))
            ks_block = {
                "ok": (d is not None and p is not None),
                "note": "Historical sample uses full-duration sigma/mu repeated per 3h segment (proxy). Do NOT interpret this as the true Paper V mi3 distribution; use only as an upper-bound stress test.",
                "n_2017": len(x7),
                "n_historical_proxy": len(yh),
                "d": d,
                "p_asymptotic": p,
                "dcrit_alpha_0p01": dcrit,
                "reject_if_p_lt": 0.01,
            }
    except Exception:
        ks_block = {"ok": False, "reason": "exception"}
    payload["derived"]["paper5_m3_ks_sanity_2017_vs_historical_proxy"] = ks_block

    payload["ok"] = (
        bool(sigma_mu.get("ok"))
        and bool(gp_tau.get("ok"))
        and bool(payload["derived"]["paper5_m3_2017_7sample_candidate"].get("ok"))
        and bool(detections_other.get("ok"))
        and hist_ok
    )

    _write_json(out_path, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "wielgus2022_m3_observed_metrics",
                "outputs": [str(out_path.relative_to(root)).replace("\\", "/")],
                "metrics": {
                    "ok": bool(payload.get("ok")),
                    "sigma_mu_rows_n": int((sigma_mu.get("rows_n") or 0) if isinstance(sigma_mu, dict) else 0),
                    "paper5_7samples_n": int(
                        ((payload.get("derived") or {}).get("paper5_m3_2017_7sample_candidate") or {}).get("samples_n") or 0
                    ),
                    "paper5_hist_segments_n": int(
                        ((payload.get("derived") or {}).get("paper5_m3_historical_distribution_candidate_pre_eht_2017_apr11") or {}).get("segments_n")
                        or 0
                    ),
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_path}")
    return 0 if bool(payload.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
