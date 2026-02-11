#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summary(xs: Sequence[float]) -> Dict[str, Any]:
    ys = [float(x) for x in xs if isinstance(x, (int, float)) and x == x]
    if not ys:
        return {"n": 0}
    ys = sorted(ys)
    n = len(ys)
    med = ys[n // 2] if (n % 2 == 1) else 0.5 * (ys[n // 2 - 1] + ys[n // 2])
    return {"n": n, "min": ys[0], "max": ys[-1], "mean": sum(ys) / n, "median": med}


def _std_pop(xs: Sequence[float]) -> float:
    ys = [float(x) for x in xs if isinstance(x, (int, float)) and x == x]
    if not ys:
        return float("nan")
    mu = sum(ys) / len(ys)
    return math.sqrt(sum((v - mu) ** 2 for v in ys) / len(ys))


def _ks_two_sample_d(sample_a: Sequence[float], sample_b: Sequence[float]) -> Optional[float]:
    a = sorted(float(x) for x in sample_a if isinstance(x, (int, float)) and x == x)
    b = sorted(float(x) for x in sample_b if isinstance(x, (int, float)) and x == x)
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
        d = max(d, abs((i / n) - (j / m)))
    return float(d)


def _ks_qks(lam: float, *, max_terms: int = 200) -> float:
    if lam <= 0.0:
        return 1.0
    s = 0.0
    for k in range(1, max_terms + 1):
        term = math.exp(-2.0 * (k * k) * (lam * lam))
        s += (term if (k % 2 == 1) else -term)
        if term < 1e-12:
            break
    return max(0.0, min(1.0, 2.0 * s))


def _ks_p_value_asymptotic(d: float, n: int, m: int) -> Optional[float]:
    if not isinstance(d, (int, float)) or not (d == d):
        return None
    if n <= 0 or m <= 0:
        return None
    ne = n * m / (n + m)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * float(d)
    return float(_ks_qks(lam))


def _mjd_to_date(mjd: float) -> date:
    # MJD 51544.0 = 2000-01-01 00:00:00 UTC.
    base = datetime(2000, 1, 1, tzinfo=timezone.utc)
    dt = base + timedelta(days=float(mjd - 51544.0))
    return dt.date()


def _ymd_from_mjd_floor(mjd: float) -> str:
    return _mjd_to_date(math.floor(float(mjd))).isoformat()


def _parse_n_mu_sigma_from_wielgus_raw(raw: str) -> Optional[Tuple[int, float, float]]:
    # Parse "... & <duration_h> & <N> & $<mu>\\pm<sigma>$ & <sigma_over_mu> & ..." from Wielgus+2022 TeX table row.
    if not isinstance(raw, str):
        return None
    m = re.search(
        r"&\s*(?P<n>\d+)\s*&\s*\$\s*(?P<mu>\d+(?:\.\d+)?)\s*\\pm\s*(?P<sig>\d+(?:\.\d+)?)\s*\$",
        raw,
    )
    if not m:
        return None
    try:
        n = int(m.group("n"))
        mu = float(m.group("mu"))
        sig = float(m.group("sig"))
    except Exception:
        return None
    if n <= 0:
        return None
    if not (mu > 0.0 and sig >= 0.0):
        return None
    return n, mu, sig


def _cluster_by_y_gaps(points: Sequence[Tuple[float, float]], k: int) -> List[List[Tuple[float, float]]]:
    if k <= 0:
        return []
    pts = [(float(x), float(y)) for x, y in points if isinstance(x, (int, float)) and isinstance(y, (int, float)) and x == x and y == y]
    if not pts:
        return [[] for _ in range(k)]
    ys = sorted(y for _, y in pts)
    if len(ys) < k:
        return [[p] for p in pts] + [[] for _ in range(k - len(pts))]

    gaps: List[Tuple[float, int]] = []
    for i in range(len(ys) - 1):
        gaps.append((ys[i + 1] - ys[i], i))
    gaps = sorted(gaps, reverse=True)[: max(0, k - 1)]
    idxs = sorted(i for _, i in gaps)
    thresholds: List[float] = []
    for i in idxs:
        thresholds.append(0.5 * (ys[i] + ys[i + 1]))

    clusters: List[List[Tuple[float, float]]] = [[] for _ in range(k)]
    for x, y in pts:
        j = 0
        while j < len(thresholds) and y > thresholds[j]:
            j += 1
        clusters[j].append((x, y))
    return clusters


def _best_assignment_by_count(clusters: Sequence[Sequence[Any]], expected_counts_by_key: Dict[str, int]) -> Optional[Dict[int, str]]:
    keys = list(expected_counts_by_key.keys())
    if not keys:
        return None
    if len(clusters) != len(keys):
        return None
    import itertools

    best_cost = None
    best_perm = None
    for perm in itertools.permutations(keys):
        cost = 0.0
        for i, k in enumerate(perm):
            cost += float(len(clusters[i]) - int(expected_counts_by_key[k])) ** 2
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm
    if best_perm is None:
        return None
    return {i: str(best_perm[i]) for i in range(len(best_perm))}


def _calibrate_affine_from_mean_std(y_vals: Sequence[float], mu_target: float, sig_target: float) -> Optional[Tuple[float, float]]:
    ys = [float(y) for y in y_vals if isinstance(y, (int, float)) and y == y]
    if not ys:
        return None
    my = sum(ys) / len(ys)
    sy = _std_pop(ys)
    if not (sy > 0.0 and mu_target > 0.0 and sig_target > 0.0):
        return None
    a = float(sig_target) / float(sy)
    b = float(mu_target) - a * float(my)
    return a, b


def _extract_largest_rgb_image_from_pdf(pdf_path: Path) -> Optional["Any"]:
    try:
        import numpy as np
        from PIL import Image
        from pypdf import PdfReader
    except Exception:
        return None
    try:
        reader = PdfReader(str(pdf_path))
        page = reader.pages[0]
        res = page["/Resources"].get_object()
        xobj = res.get("/XObject")
        if xobj is None:
            return None
        xobj = xobj.get_object()
        best = None
        best_area = None
        for name, obj in xobj.items():
            o = obj.get_object()
            if str(o.get("/Subtype") or "") != "/Image":
                continue
            w = int(o.get("/Width") or 0)
            h = int(o.get("/Height") or 0)
            if w <= 0 or h <= 0:
                continue
            data = o.get_data()
            # assume 8-bit RGB if it matches 3*W*H
            if len(data) != (w * h * 3):
                continue
            area = w * h
            if best is None or (best_area is not None and area > best_area):
                best = (w, h, data)
                best_area = area
        if best is None:
            return None
        w, h, data = best
        img = Image.frombytes("RGB", (w, h), data)
        return np.array(img)
    except Exception:
        return None


def _detect_strong_border_lines_1d(counts: "Any", *, frac: float) -> List[int]:
    # counts: 1D array-like of nonnegative ints
    try:
        import numpy as np
    except Exception:
        return []
    arr = np.asarray(counts)
    if arr.size == 0:
        return []
    mx = int(arr.max())
    if mx <= 0:
        return []
    thr = int(frac * mx)
    idx = np.where(arr >= thr)[0]
    if idx.size == 0:
        return []
    lines: List[int] = []
    s = int(idx[0])
    prev = int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v == prev + 1:
            prev = v
        else:
            lines.append(int((s + prev) // 2))
            s = v
            prev = v
    lines.append(int((s + prev) // 2))
    return sorted(lines)


def _digitize_red_curve_from_bower2018_timeseries_pdf(
    pdf_path: Path,
    *,
    duration_h: float,
    n_points: int,
    mu_target: float,
    sig_target: float,
) -> Tuple[Optional[List[Tuple[float, float]]], List[str]]:
    notes: List[str] = []
    if not pdf_path.exists():
        return None, ["missing_pdf"]
    if not (duration_h > 0.0 and n_points > 0 and mu_target > 0.0 and sig_target > 0.0):
        return None, ["bad_meta"]

    arr = _extract_largest_rgb_image_from_pdf(pdf_path)
    if arr is None:
        # Fall back to vector extraction (some PDFs contain no XObject images).
        rows_vec, notes_vec = _digitize_red_markers_from_bower2018_timeseries_pdf_vector(
            pdf_path, duration_h=duration_h, n_points=n_points, mu_target=mu_target, sig_target=sig_target
        )
        return rows_vec, ["vector_fallback"] + list(notes_vec)
    try:
        import numpy as np
    except Exception:
        return None, ["numpy_unavailable"]

    black = (arr[:, :, 0] < 40) & (arr[:, :, 1] < 40) & (arr[:, :, 2] < 40)
    row_sum = black.sum(axis=1)
    col_sum = black.sum(axis=0)
    y_lines = _detect_strong_border_lines_1d(row_sum, frac=0.8)
    x_lines = _detect_strong_border_lines_1d(col_sum, frac=0.8)
    if len(y_lines) < 2 or len(x_lines) < 2:
        return None, ["border_detection_failed"]

    # Top panel is between first two strong horizontal border lines.
    y0, y1 = int(y_lines[0]), int(y_lines[1])
    x0, x1 = int(min(x_lines)), int(max(x_lines))
    if not (y1 > y0 and x1 > x0):
        return None, ["bad_border_extents"]

    margin = 3
    crop = arr[(y0 + margin) : (y1 - margin), (x0 + margin) : (x1 - margin)]
    if crop.size == 0:
        return None, ["empty_crop"]

    R = crop[:, :, 0].astype(np.int16)
    G = crop[:, :, 1].astype(np.int16)
    B = crop[:, :, 2].astype(np.int16)
    red = (R > 140) & (R > G + 40) & (R > B + 40) & (G < 180) & (B < 180)
    if int(red.sum()) == 0:
        return None, ["no_red_pixels"]

    ys, xs = np.where(red)
    if xs.size == 0:
        return None, ["no_red_coords"]

    # Robustly focus on the main trace by trimming x extremes (legend fragments).
    x_lo = int(np.quantile(xs, 0.01))
    x_hi = int(np.quantile(xs, 0.99))
    keep = (xs >= x_lo) & (xs <= x_hi)
    xs = xs[keep]
    ys = ys[keep]
    if xs.size == 0:
        return None, ["x_trim_removed_all"]

    # Median y per x (pixel coords; y increases downward). Use -y as "upwards" coordinate for calibration.
    y_med_by_x: Dict[int, float] = {}
    for x in sorted(set(int(v) for v in xs.tolist())):
        yy = ys[xs == x]
        if yy.size == 0:
            continue
        y_med_by_x[int(x)] = float(np.median(yy))
    if len(y_med_by_x) < 5:
        return None, ["insufficient_trace_support"]

    x_keys = sorted(y_med_by_x.keys())
    x_min = float(x_keys[0])
    x_max = float(x_keys[-1])
    if not (x_max > x_min):
        return None, ["bad_x_span"]

    # Sample N points uniformly in x across the detected trace span.
    sample_xs = np.linspace(x_min, x_max, int(n_points))
    x_keys_arr = np.array(x_keys, dtype=float)
    y_vals_up: List[float] = []
    rows_xy: List[Tuple[float, float]] = []
    for sx in sample_xs:
        j = int(np.searchsorted(x_keys_arr, sx))
        if j <= 0:
            x_sel = x_keys[0]
        elif j >= len(x_keys):
            x_sel = x_keys[-1]
        else:
            left = x_keys_arr[j - 1]
            right = x_keys_arr[j]
            x_sel = x_keys[j - 1] if abs(sx - left) <= abs(sx - right) else x_keys[j]
        y_pix = float(y_med_by_x[int(x_sel)])
        t_min = ((float(x_sel) - x_min) / (x_max - x_min)) * (duration_h * 60.0)
        y_up = -y_pix
        rows_xy.append((t_min, y_up))
        y_vals_up.append(y_up)

    ab = _calibrate_affine_from_mean_std(y_vals_up, mu_target, sig_target)
    if ab is None:
        return None, ["affine_calibration_failed"]
    a, b = ab
    out_rows = [(t, a * y_up + b) for (t, y_up) in rows_xy]
    out_rows = sorted(out_rows, key=lambda t: t[0])
    notes.append(f"digitized_from_raster_pdf:red_curve:n={len(out_rows)}")
    return out_rows, notes


def _digitize_red_markers_from_bower2018_timeseries_pdf_vector(
    pdf_path: Path,
    *,
    duration_h: float,
    n_points: int,
    mu_target: float,
    sig_target: float,
) -> Tuple[Optional[List[Tuple[float, float]]], List[str]]:
    notes: List[str] = []
    try:
        from pypdf import PdfReader
        from pypdf.generic import ContentStream
    except Exception as e:
        return None, [f"pypdf_unavailable:{e}"]
    try:
        import numpy as np
    except Exception as e:
        return None, [f"numpy_unavailable:{e}"]

    def _mat_mul(m1: Tuple[float, float, float, float, float, float], m2: Tuple[float, float, float, float, float, float]) -> Tuple[float, float, float, float, float, float]:
        a, b, c, d, e, f = m1
        a2, b2, c2, d2, e2, f2 = m2
        return (
            a * a2 + c * b2,
            b * a2 + d * b2,
            a * c2 + c * d2,
            b * c2 + d * d2,
            a * e2 + c * f2 + e,
            b * e2 + d * f2 + f,
        )

    def _tf(m: Tuple[float, float, float, float, float, float], x: float, y: float) -> Tuple[float, float]:
        a, b, c, d, e, f = m
        return (a * x + c * y + e, b * x + d * y + f)

    try:
        reader = PdfReader(str(pdf_path))
        page = reader.pages[0]
        cs = ContentStream(page["/Contents"].get_object(), reader)
    except Exception as e:
        return None, [f"contentstream_failed:{e}"]

    ctm: Tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    stroke: Optional[Tuple[float, float, float]] = None
    stack: List[Tuple[Tuple[float, float, float, float, float, float], Optional[Tuple[float, float, float]]]] = []
    cur: List[Tuple[float, float]] = []
    centers: List[Tuple[float, float]] = []

    for operands, op in cs.operations:
        if op == b"q":
            stack.append((ctm, stroke))
            continue
        if op == b"Q":
            if stack:
                ctm, stroke = stack.pop()
            cur = []
            continue
        if op == b"cm" and len(operands) == 6:
            m = tuple(float(v) for v in operands)  # type: ignore[assignment]
            ctm = _mat_mul(m, ctm)
            continue
        if op == b"RG" and len(operands) == 3:
            stroke = (float(operands[0]), float(operands[1]), float(operands[2]))
            continue
        if op == b"m" and len(operands) == 2:
            cur = [_tf(ctm, float(operands[0]), float(operands[1]))]
            continue
        if op == b"l" and len(operands) == 2 and cur is not None:
            cur.append(_tf(ctm, float(operands[0]), float(operands[1])))
            continue
        if op == b"c" and len(operands) == 6 and cur is not None:
            cur.append(_tf(ctm, float(operands[4]), float(operands[5])))
            continue
        if op in (b"S", b"s", b"B", b"b", b"B*", b"b*"):
            if stroke == (1.0, 0.0, 0.0) and cur and len(cur) >= 2:
                xs = [x for x, _ in cur]
                ys = [y for _, y in cur]
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                # Marker-sized paths: treat their centers as data points.
                if (x1 - x0) <= 20.0 and (y1 - y0) <= 20.0:
                    centers.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
            cur = []
            continue
        if op == b"n":
            cur = []
            continue

    if len(centers) < int(n_points):
        return None, [f"insufficient_red_markers:{len(centers)}<{int(n_points)}"]

    # Take the top-most cluster (highest y) by selecting the highest-y N points.
    centers = sorted(centers, key=lambda t: t[1], reverse=True)
    picked = centers[: int(n_points)]
    # Sort by x for time ordering.
    picked = sorted(picked, key=lambda t: t[0])

    xs = [x for x, _ in picked]
    ys = [y for _, y in picked]
    x0 = min(xs)
    x1 = max(xs)
    if not (x1 > x0):
        return None, ["bad_x_span"]
    scale_t = (duration_h * 60.0) / (x1 - x0)
    ts = [(x - x0) * scale_t for x in xs]

    ab = _calibrate_affine_from_mean_std(ys, mu_target, sig_target)
    if ab is None:
        return None, ["affine_calibration_failed"]
    a, b = ab
    out_rows = [(t, a * y + b) for t, y in zip(ts, ys)]
    out_rows = sorted(out_rows, key=lambda t: t[0])
    notes.append(f"digitized_from_vector_pdf:red_markers:n={len(out_rows)}")
    return out_rows, notes


def _month_day_to_ymd_2005(label: str) -> Optional[str]:
    # Expected labels: "Jun 4", "Jul 30" (from thesis figure legends).
    if not isinstance(label, str):
        return None
    m = re.match(r"^\s*(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(?P<day>\d{1,2})\s*$", label)
    if not m:
        return None
    mon = m.group("mon")
    day = int(m.group("day"))
    mon_map = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    mm = mon_map.get(mon)
    if mm is None:
        return None
    if not (1 <= day <= 31):
        return None
    return f"2005-{mm:02d}-{day:02d}"


def _extract_eps_block_from_thesis_ps_gz(ps_gz_path: Path, required_substrings: Sequence[str]) -> Optional[List[str]]:
    try:
        import gzip
    except Exception:
        return None
    if not ps_gz_path.exists():
        return None
    want = [str(s) for s in required_substrings if isinstance(s, str) and s]
    if not want:
        return None

    in_block = False
    buf: List[str] = []
    try:
        with gzip.open(ps_gz_path, "rt", encoding="latin1", errors="ignore") as f:
            for line in f:
                if not in_block:
                    if line.startswith("%%BeginDocument:"):
                        in_block = True
                        buf = [line]
                    continue
                buf.append(line)
                if line.startswith("%%EndDocument"):
                    block = buf
                    in_block = False
                    buf = []
                    text = "".join(block)
                    if all(s in text for s in want):
                        return block
    except Exception:
        return None
    return None


def _parse_marrone2006_sma_from_thesis_ps_gz(
    ps_gz_path: Path, expected_by_date: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, List[Tuple[float, float]]], List[str]]:
    # Extract SMA 230 GHz total intensity time-series points for 2005 Jun/Jul nights from the thesis PS.
    # We locate the embedded EPS block whose legend contains the required (Jun/Jul) labels, then parse BoxF markers.
    notes: List[str] = []
    required = ["Jun 4", "Jun 9", "Jun 16", "Jul 20", "Jul 22", "Jul 30"]
    block_lines = _extract_eps_block_from_thesis_ps_gz(ps_gz_path, required)
    if block_lines is None:
        return {}, ["embedded_eps_block_not_found"]

    lt_to_label: Dict[str, str] = {}
    pts_by_lt: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    current_lt: Optional[str] = None
    skip_next_boxf = False

    re_lt = re.compile(r"\bLT(?P<n>\d+)\b")
    re_boxf = re.compile(r"^\s*(?P<x>-?\d+(?:\.\d+)?)\s+(?P<y>-?\d+(?:\.\d+)?)\s+BoxF\s*$")
    re_label = re.compile(r"\((?P<label>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2})\)")

    for raw in block_lines:
        mlt = re_lt.search(raw)
        if mlt:
            current_lt = f"LT{mlt.group('n')}"

        mlab = re_label.search(raw)
        if mlab and current_lt is not None:
            lt_to_label[current_lt] = str(mlab.group("label"))
            skip_next_boxf = True

        mpt = re_boxf.match(raw)
        if mpt and current_lt is not None:
            if skip_next_boxf:
                # legend marker
                skip_next_boxf = False
                continue
            try:
                x = float(mpt.group("x"))
                y = float(mpt.group("y"))
            except Exception:
                continue
            pts_by_lt[current_lt].append((x, y))

    # Map LT groups to requested dates (2005).
    by_date_xy: Dict[str, List[Tuple[float, float]]] = {}
    for lt, pts in pts_by_lt.items():
        lab = lt_to_label.get(lt)
        if not lab:
            continue
        ymd = _month_day_to_ymd_2005(lab)
        if not ymd:
            continue
        by_date_xy[ymd] = pts

    out: Dict[str, List[Tuple[float, float]]] = {}
    for ymd, meta in expected_by_date.items():
        pts = by_date_xy.get(ymd)
        if not pts:
            notes.append(f"date_not_found:{ymd}")
            continue
        dur_h = float(meta.get("duration_h") or 0.0)
        mu = float(meta.get("mu_Jy") or 0.0)
        sig = float(meta.get("sigma_Jy") or 0.0)
        if not (dur_h > 0.0 and mu > 0.0 and sig > 0.0):
            notes.append(f"bad_meta:{ymd}")
            continue
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        x0 = min(xs)
        x1 = max(xs)
        if not (x1 > x0):
            notes.append(f"bad_x_span:{ymd}")
            continue
        scale_t = (dur_h * 60.0) / (x1 - x0)
        ab = _calibrate_affine_from_mean_std(ys, mu, sig)
        if ab is None:
            notes.append(f"flux_calibration_failed:{ymd}")
            continue
        a, b = ab
        rows = [((x - x0) * scale_t, a * y + b) for x, y in pts]
        out[ymd] = sorted(rows, key=lambda t: t[0])
        exp_n = int(meta.get("n_points") or 0)
        if exp_n and exp_n != len(pts):
            notes.append(f"n_points_mismatch:{ymd}:expected={exp_n} got={len(pts)}")
    return out, notes


def _parse_yusef2009_sma_fig11_ps(
    ps_path: Path, expected_by_date: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, List[Tuple[float, float]]], List[str]]:
    notes: List[str] = []
    if not ps_path.exists():
        return {}, ["missing_file"]
    if not expected_by_date:
        return {}, ["no_expected_by_date"]
    text = ps_path.read_text(encoding="utf-8", errors="replace")
    pts: List[Tuple[float, float]] = []
    for m in re.finditer(r"\b(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+M20\b", text):
        try:
            x = float(m.group(1))
            y = float(m.group(2))
        except Exception:
            continue
        pts.append((x, y))
    if not pts:
        return {}, ["no_M20_points_found"]

    k = len(expected_by_date)
    clusters = _cluster_by_y_gaps(pts, k)
    # clusters are ordered from low-y to high-y by construction (threshold compare). Keep this order.

    expected_counts = {d: int(meta.get("n_points") or 0) for d, meta in expected_by_date.items()}
    assignment = _best_assignment_by_count(clusters, expected_counts)
    if assignment is None:
        notes.append("cluster_assignment_failed")
        return {}, notes

    out: Dict[str, List[Tuple[float, float]]] = {}
    for i, cluster in enumerate(clusters):
        ymd = assignment.get(i)
        if not ymd:
            continue
        meta = expected_by_date.get(ymd) or {}
        dur_h = float(meta.get("duration_h") or 0.0)
        mu = float(meta.get("mu_Jy") or 0.0)
        sig = float(meta.get("sigma_Jy") or 0.0)
        if not (dur_h > 0.0 and mu > 0.0 and sig > 0.0):
            notes.append(f"bad_meta:{ymd}")
            continue
        xs = [x for x, _ in cluster]
        ys = [y for _, y in cluster]
        x0 = min(xs)
        x1 = max(xs)
        if not (x1 > x0):
            notes.append(f"bad_x_span:{ymd}")
            continue
        scale_t = (dur_h * 60.0) / (x1 - x0)
        ab = _calibrate_affine_from_mean_std(ys, mu, sig)
        if ab is None:
            notes.append(f"flux_calibration_failed:{ymd}")
            continue
        a, b = ab
        rows = [((x - x0) * scale_t, a * y + b) for x, y in cluster]
        rows = sorted(rows, key=lambda t: t[0])
        out[str(ymd)] = rows
        if int(meta.get("n_points") or 0) != len(cluster):
            notes.append(f"n_points_mismatch:{ymd}:expected={int(meta.get('n_points') or 0)} got={len(cluster)}")
    return out, notes


def _parse_witzel2021_dbf1(path: Path) -> Dict[Tuple[str, float], List[Tuple[float, float]]]:
    # key: (OBS, freq_GHz) -> [(t_min, flux_Jy)]
    out: Dict[Tuple[str, float], List[Tuple[float, float]]] = defaultdict(list)
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not (raw.startswith("SMA") or raw.startswith("ALMA")):
            continue
        toks = raw.split()
        if len(toks) < 5:
            continue
        obs = toks[0]
        try:
            time_min = float(toks[2])
            flux = float(toks[3])
            freq = float(toks[-1])
        except Exception:
            continue
        key = (obs, float(freq))
        out[key].append((time_min, flux))
    for k in list(out.keys()):
        out[k] = sorted(out[k], key=lambda t: t[0])
    return dict(out)


def _parse_fazio2018_dbf3_sma(path: Path) -> Dict[str, List[Tuple[float, float]]]:
    # key: date_ymd -> [(t_min, flux_Jy)] for SMA rows
    out: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.startswith("SMA"):
            continue
        toks = raw.split()
        if len(toks) < 3:
            continue
        try:
            mjd = float(toks[1])
            flux = float(toks[2])
        except Exception:
            continue
        d = _mjd_to_date(mjd).isoformat()
        # relative time within the day is enough
        t_min = (mjd - math.floor(mjd)) * 24.0 * 60.0
        out[d].append((t_min, flux))
    for k in list(out.keys()):
        out[k] = sorted(out[k], key=lambda t: t[0])
    return dict(out)


@dataclass(frozen=True)
class _PanelMap:
    x0: float
    y0: float
    w: float
    h: float
    a: float
    b: float
    c: float
    d: float

    def contains(self, x: float, y: float) -> bool:
        return (self.x0 <= x <= self.x0 + self.w) and (self.y0 <= y <= self.y0 + self.h)

    def xy_to_data(self, x: float, y: float) -> Tuple[float, float]:
        # x_data is in "days since 2000-01-01" per plot label
        xday = self.a * x + self.b
        flux = self.c * y + self.d
        return float(xday), float(flux)


def _extract_idl_panels(text: str) -> List[Tuple[float, float, float, float]]:
    re_h = re.compile(
        r"(?P<x0>-?\d+(?:\.\d+)?)\s+(?P<y0>-?\d+(?:\.\d+)?)\s+M\s+(?P<w>-?\d+(?:\.\d+)?)\s+0\s+R\s+D"
    )
    re_v = re.compile(
        r"(?P<x0>-?\d+(?:\.\d+)?)\s+(?P<y0>-?\d+(?:\.\d+)?)\s+M\s+0\s+(?P<h>-?\d+(?:\.\d+)?)\s+R\s+D"
    )
    hv: Dict[Tuple[float, float], Dict[str, Optional[float]]] = defaultdict(lambda: {"w": None, "h": None})
    for m in re_h.finditer(text):
        w = float(m.group("w"))
        if abs(w) <= 2000:
            continue
        key = (round(float(m.group("x0")), 3), round(float(m.group("y0")), 3))
        hv[key]["w"] = float(w)
    for m in re_v.finditer(text):
        h = float(m.group("h"))
        if abs(h) <= 2000:
            continue
        key = (round(float(m.group("x0")), 3), round(float(m.group("y0")), 3))
        hv[key]["h"] = float(h)

    panels: List[Tuple[float, float, float, float]] = []
    for (x0, y0), wh in hv.items():
        if wh.get("w") is None or wh.get("h") is None:
            continue
        panels.append((float(x0), float(y0), float(wh["w"]), float(wh["h"])))
    return sorted(panels, key=lambda t: (t[1], t[0]))


def _extract_idl_ticks(lines: Sequence[str]) -> List[Tuple[float, float, float]]:
    # IDL EPS uses both:
    #   gsave x y translate ... (123.4) show grestore
    # and multiline:
    #   gsave x y translate ...
    #   (123.4) show grestore ...
    re_gsave = re.compile(r"gsave\s+(?P<x>-?\d+(?:\.\d+)?)\s+(?P<y>-?\d+(?:\.\d+)?)\s+translate\b")
    re_show = re.compile(r"\((?P<label>[^)]*)\)\s+show\b")
    cur_xy: Optional[Tuple[float, float]] = None
    out: List[Tuple[float, float, float]] = []
    for raw in lines:
        m = re_gsave.search(raw)
        if m:
            try:
                cur_xy = (float(m.group("x")), float(m.group("y")))
            except Exception:
                cur_xy = None
        m2 = re_show.search(raw)
        if m2 and cur_xy is not None:
            lab = m2.group("label").strip()
            if lab:
                try:
                    val = float(lab)
                except Exception:
                    val = None
                if val is not None:
                    out.append((cur_xy[0], cur_xy[1], float(val)))
            cur_xy = None
    return out


def _fit_panel_maps(lines: Sequence[str], panels: Sequence[Tuple[float, float, float, float]]) -> List[_PanelMap]:
    ticks = _extract_idl_ticks(lines)
    maps: List[_PanelMap] = []
    try:
        import numpy as np
    except Exception:
        return []

    # Some IDL multi-panel plots omit y-axis numeric labels in non-leftmost panels.
    # The y mapping (Flux) is shared per row (same y0, h), so we infer it from any panel in the row that has y ticks.
    y_map_by_row: Dict[float, Tuple[float, float]] = {}
    for x0, y0, w, h in panels:
        yt = [(y, val) for x, y, val in ticks if ((x0 - 600) <= x <= (x0 + 600) and y0 <= y <= y0 + h)]
        if len(yt) < 2:
            continue
        X = np.array([y for y, _ in yt])
        Y = np.array([v for _, v in yt])
        A = np.vstack([X, np.ones(len(X))]).T
        c, d = np.linalg.lstsq(A, Y, rcond=None)[0]
        y_map_by_row[float(y0)] = (float(c), float(d))

    for x0, y0, w, h in panels:
        xt = [(x, val) for x, y, val in ticks if (x0 <= x <= x0 + w and (y0 - 900) <= y <= (y0 + 200))]
        if len(xt) < 2:
            continue
        y_map = y_map_by_row.get(float(y0))
        if y_map is None:
            continue
        c, d = y_map

        X = np.array([x for x, _ in xt])
        Y = np.array([v for _, v in xt])
        A = np.vstack([X, np.ones(len(X))]).T
        a, b = np.linalg.lstsq(A, Y, rcond=None)[0]

        maps.append(
            _PanelMap(
                x0=float(x0),
                y0=float(y0),
                w=float(w),
                h=float(h),
                a=float(a),
                b=float(b),
                c=float(c),
                d=float(d),
            )
        )
    return maps


def _extract_idl_mz_points(text: str) -> List[Tuple[float, float]]:
    re_mz = re.compile(r"(?P<x>-?\d+(?:\.\d+)?)\s+(?P<y>-?\d+(?:\.\d+)?)\s+M\s+Z\b")
    return [(float(m.group("x")), float(m.group("y"))) for m in re_mz.finditer(text)]


def _parse_dexter2014_idl_eps(path: Path) -> Dict[str, List[Tuple[float, float]]]:
    # Returns date_ymd -> [(t_min, flux_Jy)] using "days since 2000-01-01" axis.
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    panels = _extract_idl_panels(text)
    maps = _fit_panel_maps(lines, panels)
    pts = _extract_idl_mz_points(text)

    base = datetime(2000, 1, 1, tzinfo=timezone.utc)
    by_date: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for x, y in pts:
        pm = next((m for m in maps if m.contains(x, y)), None)
        if pm is None:
            continue
        xday, flux = pm.xy_to_data(x, y)
        d = (base + timedelta(days=float(xday))).date().isoformat()
        t_min = (xday - math.floor(xday)) * 24.0 * 60.0
        by_date[d].append((t_min, flux))

    for k in list(by_date.keys()):
        by_date[k] = sorted(by_date[k], key=lambda t: t[0])
    return dict(by_date)


def _parse_gnuplot_eps_ticks(lines: Sequence[str]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    # Return (x_ticks, y_ticks) as (coord, value), supporting multi-line label emission.
    # We detect the (x,y) anchor from a "<x> <y> M" line, then associate the last "(...)" label seen
    # before an "MCshow"/"MRshow" line.
    re_xy = re.compile(r"^\s*(?P<x>-?\d+(?:\.\d+)?)\s+(?P<y>-?\d+(?:\.\d+)?)\s+M\s*$")
    re_paren = re.compile(r"\((?P<label>[^)]*)\)")
    cur_xy: Optional[Tuple[float, float]] = None
    cur_val: Optional[float] = None

    xt: List[Tuple[float, float]] = []
    yt: List[Tuple[float, float]] = []

    for raw in lines:
        m = re_xy.match(raw)
        if m:
            cur_xy = (float(m.group("x")), float(m.group("y")))
            cur_val = None
            continue

        labels = re_paren.findall(raw)
        if labels:
            lab = labels[-1].strip()
            try:
                cur_val = float(lab)
            except Exception:
                pass

        if "MCshow" in raw and cur_xy is not None and cur_val is not None:
            xt.append((cur_xy[0], float(cur_val)))
            cur_xy = None
            cur_val = None
            continue
        if "MRshow" in raw and cur_xy is not None and cur_val is not None:
            yt.append((cur_xy[1], float(cur_val)))
            cur_xy = None
            cur_val = None
            continue
    return xt, yt


def _fit_linear_map(pairs: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    # value = a*coord + b
    try:
        import numpy as np
    except Exception:
        return None
    if len(pairs) < 2:
        return None
    X = np.array([c for c, _ in pairs])
    Y = np.array([v for _, v in pairs])
    A = np.vstack([X, np.ones(len(X))]).T
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    return float(a), float(b)


def _parse_marrone2008_f2b_sgra(path: Path) -> List[Tuple[float, float]]:
    # Extract Sgr A* flux time series from gnuplot EPS (filled circles = CircleF).
    # Returns [(t_min, flux_Jy)] where t_min is relative to the first point.
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    xt, yt = _parse_gnuplot_eps_ticks(lines)
    if xt:
        xt = sorted(xt, key=lambda t: t[0])
    if yt:
        yt = sorted(yt, key=lambda t: t[0])

    x_map = _fit_linear_map([(x, v) for x, v in xt])
    y_map = _fit_linear_map([(y, v) for y, v in yt])
    if x_map is None or y_map is None:
        return []
    ax, bx = x_map
    ay, by = y_map

    # Plot box extents: infer from tick coordinates (use the min/max tick coords).
    x0 = min(x for x, _ in xt) if xt else None
    x1 = max(x for x, _ in xt) if xt else None
    y0 = min(y for y, _ in yt) if yt else None
    y1 = max(y for y, _ in yt) if yt else None

    out: List[Tuple[float, float]] = []
    re_pt = re.compile(r"^\s*(?P<x>-?\d+(?:\.\d+)?)\s+(?P<y>-?\d+(?:\.\d+)?)\s+CircleF\s*$")
    for raw in lines:
        m = re_pt.match(raw)
        if not m:
            continue
        x = float(m.group("x"))
        y = float(m.group("y"))
        if x0 is not None and x1 is not None and not (x0 <= x <= x1):
            continue
        if y0 is not None and y1 is not None and not (y0 <= y <= y1):
            continue
        # Exclude the legend marker (it is exactly on the legend baseline, not part of the light curve).
        if abs(y - 4637.0) < 1e-6:
            continue
        hour = ax * x + bx
        flux = ay * y + by
        out.append((hour * 60.0, flux))
    out = sorted(out, key=lambda t: t[0])
    if out:
        t0 = out[0][0]
        out = [(t - t0, f) for t, f in out]
    return out


def _segment_mi3(rows: Sequence[Tuple[float, float]], segments_n: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    notes: List[str] = []
    if not rows:
        return [], ["no_rows"]
    if segments_n <= 0:
        return [], ["segments_n<=0"]

    t0 = min(t for t, _ in rows)
    out: List[Dict[str, Any]] = []
    for j in range(1, segments_n + 1):
        start = (j - 1) * 180.0
        end = j * 180.0
        vals = [flux for t, flux in rows if start <= (t - t0) <= end]
        if not vals:
            out.append({"segment_index": j, "ok": False, "reason": "no_points_in_window", "window_min": [start, end]})
            continue
        mu = sum(vals) / len(vals)
        sig = _std_pop(vals)
        out.append(
            {
                "segment_index": j,
                "ok": True,
                "window_min": [start, end],
                "n_points": len(vals),
                "mean_Jy": mu,
                "std_Jy": sig,
                "mi3": (sig / mu) if mu != 0.0 else None,
            }
        )
    return out, notes


def _plot_ecdf(samples: Dict[str, Sequence[float]], out_png: Path, *, title: str) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return f"matplotlib_unavailable: {e}"

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=200)
    for name, xs in samples.items():
        ys = sorted(float(x) for x in xs if isinstance(x, (int, float)) and x == x)
        if not ys:
            continue
        n = len(ys)
        cdf = [(i + 1) / n for i in range(n)]
        ax.step(ys, cdf, where="post", label=f"{name} (n={n})")
    ax.set_xlabel("m3 = σ/μ (3h)")
    ax.set_ylabel("ECDF")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Reconstruct Paper V 'historical distribution' M3 (3h σ/μ) values from primary time-series data where available."
    )
    ap.add_argument(
        "--mode",
        choices=["pre_eht_2017_apr11"],
        default="pre_eht_2017_apr11",
        help="Which Wielgus+2022-derived candidate set to target (default: pre-EHT cutoff 2017-04-11).",
    )
    args = ap.parse_args()

    root = _repo_root()
    in_w = root / "output" / "eht" / "wielgus2022_m3_observed_metrics.json"
    if not in_w.exists():
        print(f"[err] missing: {in_w}")
        return 2

    w = _read_json(in_w)
    w_der = w.get("derived") or {}
    if args.mode == "pre_eht_2017_apr11":
        cand = w_der.get("paper5_m3_historical_distribution_candidate_pre_eht_2017_apr11") or {}
    else:
        cand = {}

    if not (isinstance(cand, dict) and bool(cand.get("ok")) and isinstance(cand.get("curves"), list)):
        print("[err] candidate curves not found / not ok in wielgus2022_m3_observed_metrics.json")
        return 2

    curves = list(cand["curves"])
    expected_segments_total = int(cand.get("segments_n") or 0)

    sample_2017 = w_der.get("paper5_m3_2017_7sample_candidate") or {}
    sample_2017_vals = []
    if isinstance(sample_2017, dict) and isinstance(sample_2017.get("samples"), list):
        for r in sample_2017["samples"]:
            v = (r or {}).get("sigma_over_mu_3h")
            if isinstance(v, (int, float)):
                sample_2017_vals.append(float(v))

    # Load primary time-series caches (where available).
    sources_used: List[Dict[str, Any]] = []
    time_series_by_curve: Dict[Tuple[str, str, str], Tuple[str, List[Tuple[float, float]], List[str]]] = {}
    missing_curves: List[Dict[str, Any]] = []

    # Dexter2014 (IDL EPS): provides most of the 230 GHz historical light curves used in the historical distribution.
    dex_eps = root / "data" / "eht" / "sources" / "arxiv_1308.5968" / "sgra_mm_lcurve_230.eps"
    dex_by_date = _parse_dexter2014_idl_eps(dex_eps) if dex_eps.exists() else {}
    if dex_eps.exists():
        sources_used.append({"key": "dexter2014_idl_eps", "path": str(dex_eps.relative_to(root)).replace("\\", "/")})
    else:
        missing_curves.append({"reason": "missing_file", "path": str(dex_eps.relative_to(root)).replace("\\", "/")})

    # Witzel2021 (IOP "data behind figure"): dbf1 contains SMA/ALMA light curves (SMA 2015/2016 and ALMA 2016).
    w_dbf1 = root / "data" / "eht" / "lightcurves" / "witzel2021" / "dbf1.txt"
    witzel = _parse_witzel2021_dbf1(w_dbf1) if w_dbf1.exists() else {}
    if w_dbf1.exists():
        sources_used.append({"key": "witzel2021_dbf1", "path": str(w_dbf1.relative_to(root)).replace("\\", "/")})
    else:
        missing_curves.append({"reason": "missing_file", "path": str(w_dbf1.relative_to(root)).replace("\\", "/")})

    # Fazio2018 (IOP "data behind figure"): dbf3 contains SMA 2015-05-14 mm light curve.
    f_dbf3 = root / "data" / "eht" / "lightcurves" / "fazio2018" / "dbf3.txt"
    fazio_sma = _parse_fazio2018_dbf3_sma(f_dbf3) if f_dbf3.exists() else {}
    if f_dbf3.exists():
        sources_used.append({"key": "fazio2018_dbf3", "path": str(f_dbf3.relative_to(root)).replace("\\", "/")})
    else:
        missing_curves.append({"reason": "missing_file", "path": str(f_dbf3.relative_to(root)).replace("\\", "/")})

    # Yusef-Zadeh et al. 2009 (PGPLOT PS): Figure 11 contains SMA 230 GHz light curves for 2007 Apr 1 and 3-5.
    # We extract plotted points (M20 markers) and calibrate (time, flux) using duration_h and (mu, sigma) from
    # Wielgus+2022 tab:detections_other_papers (same table that defines the historical distribution composition).
    yusef_by_date: Dict[str, List[Tuple[float, float]]] = {}
    yusef_notes: List[str] = []
    yusef_ps = root / "data" / "eht" / "sources" / "arxiv_0907.3786" / "f11_plot_all_sma.ps"
    yusef_expected: Dict[str, Dict[str, Any]] = {}
    for c in curves:
        if not (isinstance(c, dict) and str(c.get("reference_key") or "") == "Yusef2009"):
            continue
        ymd = str(c.get("date_ymd") or "")
        if not ymd:
            continue
        dur_h = float(c.get("duration_h") or 0.0)
        src_raw = ((c.get("source") or {}) if isinstance(c.get("source"), dict) else {}).get("raw")
        parsed = _parse_n_mu_sigma_from_wielgus_raw(str(src_raw) if src_raw is not None else "")
        if parsed is None:
            continue
        n_points, mu, sig = parsed
        yusef_expected[ymd] = {"duration_h": dur_h, "n_points": n_points, "mu_Jy": mu, "sigma_Jy": sig}
    if yusef_ps.exists() and yusef_expected:
        parsed_by_date, yusef_notes = _parse_yusef2009_sma_fig11_ps(yusef_ps, yusef_expected)
        if parsed_by_date:
            yusef_by_date = dict(parsed_by_date)
            sources_used.append({"key": "yusef2009_fig11_sma_ps", "path": str(yusef_ps.relative_to(root)).replace("\\", "/")})
        else:
            missing_curves.append({"reason": "parse_failed", "path": str(yusef_ps.relative_to(root)).replace("\\", "/"), "notes": yusef_notes})
    elif yusef_expected and not yusef_ps.exists():
        missing_curves.append({"reason": "missing_file", "path": str(yusef_ps.relative_to(root)).replace("\\", "/")})

    # Marrone (2006) thesis: SMA 2005 light curves used in the historical distribution.
    marrone_by_date: Dict[str, List[Tuple[float, float]]] = {}
    marrone_notes: List[str] = []
    marrone_ps_gz = root / "data" / "eht" / "sources" / "marrone2006_thesis_dpm_thesis.ps.gz"
    marrone_expected: Dict[str, Dict[str, Any]] = {}
    for c in curves:
        if not (isinstance(c, dict) and str(c.get("reference_key") or "") == "Marrone2006"):
            continue
        ymd = str(c.get("date_ymd") or "")
        if not ymd:
            continue
        dur_h = float(c.get("duration_h") or 0.0)
        src_raw = ((c.get("source") or {}) if isinstance(c.get("source"), dict) else {}).get("raw")
        parsed = _parse_n_mu_sigma_from_wielgus_raw(str(src_raw) if src_raw is not None else "")
        if parsed is None:
            continue
        n_points, mu, sig = parsed
        marrone_expected[ymd] = {"duration_h": dur_h, "n_points": n_points, "mu_Jy": mu, "sigma_Jy": sig}
    if marrone_ps_gz.exists() and marrone_expected:
        parsed_by_date, marrone_notes = _parse_marrone2006_sma_from_thesis_ps_gz(marrone_ps_gz, marrone_expected)
        if parsed_by_date:
            marrone_by_date = dict(parsed_by_date)
            sources_used.append({"key": "marrone2006_thesis_ps_gz", "path": str(marrone_ps_gz.relative_to(root)).replace("\\", "/")})
        else:
            missing_curves.append(
                {"reason": "parse_failed", "reference_key": "Marrone2006", "path": str(marrone_ps_gz.relative_to(root)).replace("\\", "/"), "notes": marrone_notes}
            )
    elif marrone_expected and not marrone_ps_gz.exists():
        missing_curves.append({"reason": "missing_file", "path": str(marrone_ps_gz.relative_to(root)).replace("\\", "/")})

    # Bower et al. 2018 (ALMA): arXiv source includes per-epoch time series PDFs; in the arXiv package these are raster images.
    # We digitize the red Stokes I trace in the top panel and calibrate (time, flux) using (duration_h, N, mu, sigma) from Wielgus+2022 table.
    bower_by_date: Dict[str, List[Tuple[float, float]]] = {}
    bower_notes: Dict[str, List[str]] = {}
    bower_pdf_by_date = {
        "2016-03-03": root / "data" / "eht" / "sources" / "arxiv_1810.07317" / "SgrAstar_alltimeplots_Epoch1.pdf",
        "2016-08-13": root / "data" / "eht" / "sources" / "arxiv_1810.07317" / "SgrAstar_alltimeplots_Epoch3.pdf",
    }
    for c in curves:
        if not (isinstance(c, dict) and str(c.get("reference_key") or "") == "Bower2018"):
            continue
        ymd = str(c.get("date_ymd") or "")
        if ymd not in bower_pdf_by_date:
            continue
        dur_h = float(c.get("duration_h") or 0.0)
        src_raw = ((c.get("source") or {}) if isinstance(c.get("source"), dict) else {}).get("raw")
        parsed = _parse_n_mu_sigma_from_wielgus_raw(str(src_raw) if src_raw is not None else "")
        if parsed is None:
            continue
        n_points, mu, sig = parsed
        pdf_path = bower_pdf_by_date[ymd]
        rows, notes = _digitize_red_curve_from_bower2018_timeseries_pdf(
            pdf_path, duration_h=dur_h, n_points=n_points, mu_target=mu, sig_target=sig
        )
        if rows:
            bower_by_date[ymd] = rows
            bower_notes[ymd] = notes
            sources_used.append({"key": f"bower2018_alma_timeseries_pdf_{ymd}", "path": str(pdf_path.relative_to(root)).replace("\\", "/")})
        else:
            missing_curves.append(
                {"reason": "parse_failed", "reference_key": "Bower2018", "date_ymd": ymd, "path": str(pdf_path.relative_to(root)).replace("\\", "/"), "notes": notes}
            )

    # Build per-curve time series using prioritized sources.
    for c in curves:
        if not isinstance(c, dict):
            continue
        ref = str(c.get("reference_key") or "")
        arr = str(c.get("array") or "")
        ymd = str(c.get("date_ymd") or "")
        seg_n = int(c.get("segments_n") or 0)
        key = (ref, arr, ymd)

        rows: Optional[List[Tuple[float, float]]] = None
        notes: List[str] = []
        method = None

        if ref == "Dexter2014":
            rows = dex_by_date.get(ymd)
            method = "Dexter2014_IDL_EPS"
            if rows is None:
                notes.append("date_not_found_in_eps")

        elif ref == "Witzel2021":
            # Use frequency heuristics to pick the correct epoch.
            if arr == "ALMA":
                # Prefer ~232 GHz (ALMA) curves.
                candidates = [(freq, r) for (obs, freq), r in witzel.items() if obs == "ALMA" and abs(freq - 232.0) < 2.0]
                candidates = sorted(candidates, key=lambda t: abs(t[0] - 232.0))
                rows = candidates[0][1] if candidates else None
                method = "Witzel2021_IOP_dbf1"
            elif arr == "SMA":
                # Prefer ~236 GHz for the 2016 SMA epoch used in the historical distribution.
                candidates = [(freq, r) for (obs, freq), r in witzel.items() if obs == "SMA" and abs(freq - 236.1) < 2.0]
                candidates = sorted(candidates, key=lambda t: abs(t[0] - 236.1))
                rows = candidates[0][1] if candidates else None
                method = "Witzel2021_IOP_dbf1"
            else:
                rows = None
            if rows is None:
                notes.append("no_matching_epoch_in_dbf1")

            # Normalize time origin (the dbf1 file uses an arbitrary time offset per epoch for plotting).
            if rows:
                t0 = min(t for t, _ in rows)
                rows = [(t - t0, flux) for t, flux in rows]

        elif ref == "fazio:2018":
            if arr == "SMA":
                rows = fazio_sma.get(ymd)
                method = "Fazio2018_IOP_dbf3"
            if rows is None:
                notes.append("date_not_found_in_dbf3")

        elif ref == "Yusef2009":
            if arr == "SMA":
                rows = yusef_by_date.get(ymd)
                method = "Yusef2009_PGPLOT_PS_fig11"
                if yusef_notes:
                    notes.extend(yusef_notes)
            if rows is None:
                notes.append("date_not_found_in_fig11_ps")

        elif ref == "Marrone2006":
            if arr == "SMA":
                rows = marrone_by_date.get(ymd)
                method = "Marrone2006_thesis_PS_gz_embedded_EPS"
                if marrone_notes:
                    notes.extend(marrone_notes)
            if rows is None:
                notes.append("date_not_found_in_thesis_ps_gz")

        elif ref == "Bower2018":
            if arr == "ALMA":
                rows = bower_by_date.get(ymd)
                method = "Bower2018_digitized_raster_PDF_timeseries"
                if ymd in bower_notes:
                    notes.extend(bower_notes[ymd])
            if rows is None:
                notes.append("date_not_found_in_timeseries_pdf")

        else:
            rows = None
            notes.append("no_extractor_for_reference_key")

        if ref == "Marrone2008" and rows is None:
            # gnuplot EPS from arXiv source (SMA light curve; Sgr A* is CircleF markers).
            m_eps = root / "data" / "eht" / "sources" / "arxiv_0712.2877" / "f2b.eps"
            if m_eps.exists():
                rows = _parse_marrone2008_f2b_sgra(m_eps)
                method = "Marrone2008_gnuplot_EPS_f2b"
                if rows:
                    sources_used.append({"key": "marrone2008_f2b_eps", "path": str(m_eps.relative_to(root)).replace("\\", "/")})
            else:
                notes.append("missing_marrone2008_f2b_eps")
            if rows is None:
                notes.append("no_matching_epoch_in_eps")

        if rows is None:
            missing_curves.append({"reference_key": ref, "array": arr, "date_ymd": ymd, "segments_n": seg_n, "reason": ";".join(notes) if notes else "no_data"})
            continue

        if isinstance(rows, list) and len(rows) == 0:
            notes.append("empty_time_series")
            missing_curves.append({"reference_key": ref, "array": arr, "date_ymd": ymd, "segments_n": seg_n, "reason": ";".join(notes)})
            continue

        time_series_by_curve[key] = (str(method or "unknown"), list(rows), notes)

    # Compute per-segment mi3 values.
    segments_out: List[Dict[str, Any]] = []
    computed_values: List[float] = []
    per_curve_summary: List[Dict[str, Any]] = []

    for (ref, arr, ymd), (method, rows, notes) in sorted(time_series_by_curve.items()):
        seg_n = 0
        for c in curves:
            if isinstance(c, dict) and str(c.get("reference_key") or "") == ref and str(c.get("array") or "") == arr and str(c.get("date_ymd") or "") == ymd:
                seg_n = int(c.get("segments_n") or 0)
                break
        segs, seg_notes = _segment_mi3(rows, seg_n)
        per_curve_summary.append(
            {
                "reference_key": ref,
                "array": arr,
                "date_ymd": ymd,
                "segments_expected": seg_n,
                "rows_n": len(rows),
                "rows_span_min": float((max(t for t, _ in rows) - min(t for t, _ in rows))) if rows else None,
                "method": method,
                "notes": list(dict.fromkeys(list(notes) + list(seg_notes))),
            }
        )
        for s in segs:
            row = {"reference_key": ref, "array": arr, "date_ymd": ymd, "method": method}
            row.update(s)
            segments_out.append(row)
            if row.get("ok") and isinstance(row.get("mi3"), (int, float)):
                computed_values.append(float(row["mi3"]))

    ks_d = _ks_two_sample_d(sample_2017_vals, computed_values)
    ks_p = _ks_p_value_asymptotic(float(ks_d) if ks_d is not None else float("nan"), len(sample_2017_vals), len(computed_values)) if ks_d is not None else None

    is_full = expected_segments_total > 0 and len(computed_values) == expected_segments_total
    hist_label = "historical_reconstructed" if is_full else "historical_reconstructed_partial"

    out_dir = root / "output" / "eht"
    out_json = out_dir / "eht_sgra_paper5_m3_historical_distribution_values.json"
    out_png = out_dir / "eht_sgra_paper5_m3_historical_distribution_values_ecdf.png"

    plot_err = _plot_ecdf(
        {"2017_7sample": sample_2017_vals, hist_label: computed_values},
        out_png,
        title=f"Sgr A* M3 (3h σ/μ): 2017 vs historical ({'full' if is_full else 'partial'} reconstruction)",
    )

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "inputs": {
            "wielgus2022_m3_observed_metrics": str(in_w.relative_to(root)).replace("\\", "/"),
            "candidate_mode": args.mode,
            "expected_segments_total": expected_segments_total,
        },
        "sources_used": sources_used,
        "coverage": {
            "segments_expected_total": expected_segments_total,
            "segments_computed_total": len(computed_values),
            "segments_computed_fraction": (len(computed_values) / expected_segments_total) if expected_segments_total > 0 else None,
            "segments_missing_total": (expected_segments_total - len(computed_values)) if expected_segments_total > 0 else None,
            "is_full_reconstruction": is_full,
            "curves_expected_total": len(curves),
            "curves_with_any_data_total": len(time_series_by_curve),
        },
        "derived": {
            "historical_mi3_values_summary": _summary(computed_values),
            "ks_2017_7sample_vs_historical": {
                "ok": True if ks_d is not None else False,
                "d": ks_d,
                "p_asymptotic": ks_p,
                "n_2017": len(sample_2017_vals),
                "n_historical": len(computed_values),
                "note": "Full reconstruction from cached primary sources." if is_full else "Partial reconstruction from cached primary sources.",
            },
        },
        "data": {
            "sample_2017_7": sample_2017_vals,
            "historical_mi3_values": computed_values,
            "segments": segments_out,
            "per_curve_summary": per_curve_summary,
            "missing_curves": missing_curves,
        },
        "outputs": {
            "json": str(out_json.relative_to(root)).replace("\\", "/"),
            "ecdf_png": str(out_png.relative_to(root)).replace("\\", "/") if plot_err is None else None,
            "plot_error": plot_err,
        },
    }
    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "eht_sgra_paper5_m3_historical_distribution_values",
                "outputs": [str(out_json.relative_to(root)).replace("\\", "/")] + ([str(out_png.relative_to(root)).replace("\\", "/")] if plot_err is None else []),
                "metrics": {
                    "segments_expected": expected_segments_total,
                    "segments_computed": len(computed_values),
                    "ks_d": ks_d,
                    "ks_p_asymptotic": ks_p,
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    if plot_err is None:
        print(f"[ok] png:  {out_png}")
    else:
        print(f"[warn] plot skipped: {plot_err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
