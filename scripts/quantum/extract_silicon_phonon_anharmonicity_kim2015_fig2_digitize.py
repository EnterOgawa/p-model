from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from pypdf import PdfReader
from pypdf.generic import ContentStream


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。

def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_sha256` の入出力契約と処理意図を定義する。

def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            # 条件分岐: `not b` を満たす経路を評価する。
            if not b:
                break

            h.update(b)

    return h.hexdigest()


# 関数: `_mat_mul` の入出力契約と処理意図を定義する。

def _mat_mul(
    m1: tuple[float, float, float, float, float, float],
    m2: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


# 関数: `_mat_apply` の入出力契約と処理意図を定義する。

def _mat_apply(m: tuple[float, float, float, float, float, float], x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = m
    return (a * float(x) + c * float(y) + e, b * float(x) + d * float(y) + f)


# 関数: `_round_rgb` の入出力契約と処理意図を定義する。

def _round_rgb(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(round(float(c), 2) for c in rgb)  # type: ignore[return-value]


# クラス: `_PathRec` の責務と境界条件を定義する。

@dataclass(frozen=True)
class _PathRec:
    paint: str
    n_points: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    stroke_rgb: tuple[float, float, float]
    fill_rgb: tuple[float, float, float]

    # 関数: `cx` の入出力契約と処理意図を定義する。
    @property
    def cx(self) -> float:
        return 0.5 * (float(self.xmin) + float(self.xmax))

    # 関数: `cy` の入出力契約と処理意図を定義する。

    @property
    def cy(self) -> float:
        return 0.5 * (float(self.ymin) + float(self.ymax))

    # 関数: `w` の入出力契約と処理意図を定義する。

    @property
    def w(self) -> float:
        return float(self.xmax) - float(self.xmin)

    # 関数: `h` の入出力契約と処理意図を定義する。

    @property
    def h(self) -> float:
        return float(self.ymax) - float(self.ymin)


# 関数: `_extract_paths` の入出力契約と処理意図を定義する。

def _extract_paths(page, reader: PdfReader) -> list[_PathRec]:
    """
    Extract painted paths (stroke/fill) as bounding boxes with current RGB colors.

    This is intentionally lightweight: we approximate Bezier segments by storing control
    points + endpoints so that marker bounding boxes are stable.
    """
    cs = ContentStream(page.get_contents(), reader)
    ops = cs.operations

    ctm: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    stack: list[tuple[tuple[float, float, float, float, float, float], tuple[float, float, float], tuple[float, float, float]]] = []

    stroke_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fill_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)

    cur: list[tuple[float, float]] = []
    cur_start: tuple[float, float] | None = None
    out: list[_PathRec] = []

    for operands, op in ops:
        # 条件分岐: `op == b"q"` を満たす経路を評価する。
        if op == b"q":
            stack.append((ctm, stroke_rgb, fill_rgb))
            continue

        # 条件分岐: `op == b"Q"` を満たす経路を評価する。

        if op == b"Q":
            # 条件分岐: `stack` を満たす経路を評価する。
            if stack:
                ctm, stroke_rgb, fill_rgb = stack.pop()

            continue

        # 条件分岐: `op == b"cm"` を満たす経路を評価する。

        if op == b"cm":
            a, b, c, d, e, f = map(float, operands)
            ctm = _mat_mul(ctm, (a, b, c, d, e, f))
            continue

        # RGB color operators (this PDF uses RGB in the target figure).

        if op == b"RG":
            stroke_rgb = _round_rgb(tuple(map(float, operands)))  # type: ignore[arg-type]
            continue

        # 条件分岐: `op == b"rg"` を満たす経路を評価する。

        if op == b"rg":
            fill_rgb = _round_rgb(tuple(map(float, operands)))  # type: ignore[arg-type]
            continue

        # 条件分岐: `op == b"m"` を満たす経路を評価する。

        if op == b"m":
            x, y = map(float, operands)
            pt = _mat_apply(ctm, x, y)
            cur = [pt]
            cur_start = pt
            continue

        # 条件分岐: `op == b"l"` を満たす経路を評価する。

        if op == b"l":
            # 条件分岐: `not cur` を満たす経路を評価する。
            if not cur:
                continue

            x, y = map(float, operands)
            cur.append(_mat_apply(ctm, x, y))
            continue

        # 条件分岐: `op == b"c"` を満たす経路を評価する。

        if op == b"c":
            # 条件分岐: `not cur` を満たす経路を評価する。
            if not cur:
                continue

            x1, y1, x2, y2, x3, y3 = map(float, operands)
            cur.append(_mat_apply(ctm, x1, y1))
            cur.append(_mat_apply(ctm, x2, y2))
            cur.append(_mat_apply(ctm, x3, y3))
            continue

        # 条件分岐: `op == b"h"` を満たす経路を評価する。

        if op == b"h":
            # 条件分岐: `cur and cur_start is not None` を満たす経路を評価する。
            if cur and cur_start is not None:
                cur.append(cur_start)

            continue

        # 条件分岐: `op == b"re"` を満たす経路を評価する。

        if op == b"re":
            # Rectangle: x y w h
            x, y, w, h = map(float, operands)
            p1 = _mat_apply(ctm, x, y)
            p2 = _mat_apply(ctm, x + w, y)
            p3 = _mat_apply(ctm, x + w, y + h)
            p4 = _mat_apply(ctm, x, y + h)
            cur = [p1, p2, p3, p4, p1]
            cur_start = p1
            continue

        # 条件分岐: `op in (b"S", b"s", b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*")` を満たす経路を評価する。

        if op in (b"S", b"s", b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*"):
            # 条件分岐: `cur and len(cur) >= 2` を満たす経路を評価する。
            if cur and len(cur) >= 2:
                xs = [p[0] for p in cur]
                ys = [p[1] for p in cur]
                out.append(
                    _PathRec(
                        paint=op.decode("ascii", errors="ignore"),
                        n_points=int(len(cur)),
                        xmin=float(min(xs)),
                        xmax=float(max(xs)),
                        ymin=float(min(ys)),
                        ymax=float(max(ys)),
                        stroke_rgb=stroke_rgb,
                        fill_rgb=fill_rgb,
                    )
                )

            cur = []
            cur_start = None
            continue

    return out


# 関数: `_dedup_sorted` の入出力契約と処理意図を定義する。

def _dedup_sorted(vals: Iterable[float], *, tol: float) -> list[float]:
    out: list[float] = []
    for v in sorted(float(x) for x in vals):
        # 条件分岐: `not out or abs(v - out[-1]) > float(tol)` を満たす経路を評価する。
        if not out or abs(v - out[-1]) > float(tol):
            out.append(float(v))

    return out


# 関数: `_infer_fig2_bbox` の入出力契約と処理意図を定義する。

def _infer_fig2_bbox(paths: list[_PathRec]) -> tuple[float, float, float, float]:
    """
    Infer the Fig.2 plot frame bbox from long border lines:
    - Two vertical segments (dx≈0) with dy>100
    - With matching y-extent
    """
    vlines: list[tuple[float, float, float]] = []
    for p in paths:
        # 条件分岐: `p.paint not in ("S", "s")` を満たす経路を評価する。
        if p.paint not in ("S", "s"):
            continue

        # 条件分岐: `p.n_points != 2` を満たす経路を評価する。

        if p.n_points != 2:
            continue

        dx = float(p.xmax - p.xmin)
        dy = float(p.ymax - p.ymin)
        # 条件分岐: `dx > 0.2 or dy < 120.0` を満たす経路を評価する。
        if dx > 0.2 or dy < 120.0:
            continue

        # 条件分岐: `p.ymin < 450.0` を満たす経路を評価する。

        if p.ymin < 450.0:
            continue

        vlines.append((p.cx, p.ymin, p.ymax))

    # 条件分岐: `len(vlines) < 2` を満たす経路を評価する。

    if len(vlines) < 2:
        raise ValueError("cannot infer bbox: too few vertical frame lines")

    # Group by y-range (rounded) and pick the best group.

    buckets: dict[tuple[float, float], list[float]] = {}
    for x, y0, y1 in vlines:
        key = (round(float(y0), 1), round(float(y1), 1))
        buckets.setdefault(key, []).append(float(x))

    best_key: tuple[float, float] | None = None
    best_score = -1.0
    for key, xs in buckets.items():
        # 条件分岐: `len(xs) < 2` を満たす経路を評価する。
        if len(xs) < 2:
            continue

        y0, y1 = key
        score = float(y1 - y0)
        # 条件分岐: `score > best_score` を満たす経路を評価する。
        if score > best_score:
            best_score = score
            best_key = key

    # 条件分岐: `best_key is None` を満たす経路を評価する。

    if best_key is None:
        raise ValueError("cannot infer bbox: no suitable frame group")

    y0, y1 = best_key
    xs = buckets[best_key]
    x0 = float(min(xs))
    x1 = float(max(xs))
    # 条件分岐: `not (x1 > x0 and y1 > y0)` を満たす経路を評価する。
    if not (x1 > x0 and y1 > y0):
        raise ValueError("invalid inferred bbox")

    return (x0, x1, float(y0), float(y1))


# 関数: `_infer_zero_line_y` の入出力契約と処理意図を定義する。

def _infer_zero_line_y(paths: list[_PathRec], *, bbox: tuple[float, float, float, float]) -> float:
    x0, x1, y0, _y1 = bbox
    width = float(x1 - x0)
    best: _PathRec | None = None
    for p in paths:
        # 条件分岐: `p.paint not in ("S", "s")` を満たす経路を評価する。
        if p.paint not in ("S", "s"):
            continue

        # 条件分岐: `p.n_points < 5` を満たす経路を評価する。

        if p.n_points < 5:
            continue

        # 条件分岐: `p.w < 0.9 * width or p.h > 0.2` を満たす経路を評価する。

        if p.w < 0.9 * width or p.h > 0.2:
            continue

        # 条件分岐: `not (x0 - 1.0 <= p.xmin <= x1 + 1.0 and x0 - 1.0 <= p.xmax <= x1 + 1.0)` を満たす経路を評価する。

        if not (x0 - 1.0 <= p.xmin <= x1 + 1.0 and x0 - 1.0 <= p.xmax <= x1 + 1.0):
            continue
        # Zero line is close to the bottom (but inside the frame).

        if not (y0 + 5.0 <= p.ymin <= y0 + 45.0):
            continue

        # 条件分岐: `best is None or p.w > best.w` を満たす経路を評価する。

        if best is None or p.w > best.w:
            best = p

    # 条件分岐: `best is None` を満たす経路を評価する。

    if best is None:
        raise ValueError("cannot infer zero line")

    return float(best.cy)


# 関数: `_find_nearest` の入出力契約と処理意図を定義する。

def _find_nearest(points: list[tuple[float, float]], *, x_target: float, tol_x: float) -> tuple[float, float]:
    best: tuple[float, float] | None = None
    for x, y in points:
        # 条件分岐: `abs(float(x) - float(x_target)) > float(tol_x)` を満たす経路を評価する。
        if abs(float(x) - float(x_target)) > float(tol_x):
            continue

        # 条件分岐: `best is None or abs(float(x) - float(x_target)) < abs(float(best[0]) - float(...` を満たす経路を評価する。

        if best is None or abs(float(x) - float(x_target)) < abs(float(best[0]) - float(x_target)):
            best = (float(x), float(y))

    # 条件分岐: `best is None` を満たす経路を評価する。

    if best is None:
        raise ValueError(f"missing marker for x={x_target:.3f}")

    return best


# 関数: `_try_read_mean_softening_at_tmax` の入出力契約と処理意図を定義する。

def _try_read_mean_softening_at_tmax(root: Path, *, default_value: float) -> float:
    """
    Attempt to read |<Δε/ε>| at T_max from the accepted manuscript cache
    (generated by extract_silicon_phonon_anharmonicity_kim2015_softening_proxy.py).
    """
    src = root / "data" / "quantum" / "sources" / "osti_kim2015_prb91_014307_si_phonon_anharmonicity" / "extracted_values.json"
    # 条件分岐: `not src.exists()` を満たす経路を評価する。
    if not src.exists():
        return float(default_value)

    try:
        obj = json.loads(src.read_text(encoding="utf-8"))
    except Exception:
        return float(default_value)

    parsed = obj.get("parsed_from_pdf", {})
    # 条件分岐: `not isinstance(parsed, dict)` を満たす経路を評価する。
    if not isinstance(parsed, dict):
        return float(default_value)

    mean_frac = parsed.get("mean_fractional_energy_shift", {})
    # 条件分岐: `isinstance(mean_frac, dict) and isinstance(mean_frac.get("isobaric"), (int, f...` を満たす経路を評価する。
    if isinstance(mean_frac, dict) and isinstance(mean_frac.get("isobaric"), (int, float)):
        v = float(mean_frac["isobaric"])
        # 条件分岐: `math.isfinite(v) and abs(v) > 0` を満たす経路を評価する。
        if math.isfinite(v) and abs(v) > 0:
            return float(abs(v))

    proxy = parsed.get("softening_proxy", {})
    # 条件分岐: `isinstance(proxy, dict) and isinstance(proxy.get("fractional_energy_shift_at_...` を満たす経路を評価する。
    if isinstance(proxy, dict) and isinstance(proxy.get("fractional_energy_shift_at_t_max_isobaric"), (int, float)):
        v = float(proxy["fractional_energy_shift_at_t_max_isobaric"])
        # 条件分岐: `math.isfinite(v) and abs(v) > 0` を満たす経路を評価する。
        if math.isfinite(v) and abs(v) > 0:
            return float(abs(v))

    return float(default_value)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Digitize Kim et al., Phys. Rev. B 91, 014307 (2015) Fig.2 from the published PDF "
            "(CaltechAUTHORS mirror) as a primary constraint candidate for Phase 7 / Step 7.14.20.\n"
            "This extracts marker positions (TA/LA/LA-LO/TO-LO features) and the dotted average, then "
            "calibrates the y-axis using the mean isobaric softening at T_max (~0.07 at 1500 K)."
        )
    )
    ap.add_argument(
        "--source-dirname",
        default="caltechauthors_kim2015_prb91_014307_si_phonon_anharmonicity",
        help="Source directory name under data/quantum/sources/.",
    )
    ap.add_argument(
        "--pdf-name",
        default="PhysRevB.91.014307.pdf",
        help="Cached PDF filename under the source directory.",
    )
    ap.add_argument(
        "--page-index",
        type=int,
        default=2,
        help="0-based PDF page index containing Fig.2 (default: 2).",
    )
    ap.add_argument(
        "--out-json-name",
        default="fig2_digitized_softening.json",
        help="Output JSON filename under the source directory.",
    )
    ap.add_argument(
        "--temps-k",
        default="100,200,300,600,900,1000,1100,1200,1300,1400,1500",
        help="Comma-separated temperature list (K) expected for Fig.2 data points (sorted by x).",
    )
    ap.add_argument(
        "--mean-softening-at-tmax",
        type=float,
        default=float("nan"),
        help=(
            "Mean isobaric softening < -Δε/ε > at T_max used to calibrate y-axis. "
            "If omitted/NaN, tries to read from the accepted-manuscript cache; fallback is 0.07."
        ),
    )
    args = ap.parse_args()

    root = _repo_root()
    src_dir = root / "data" / "quantum" / "sources" / str(args.source_dirname)
    extracted_json = src_dir / "extracted_values.json"
    pdf_path = src_dir / str(args.pdf_name)

    # 条件分岐: `not extracted_json.exists()` を満たす経路を評価する。
    if not extracted_json.exists():
        raise SystemExit(f"[fail] missing: {extracted_json}")

    # 条件分岐: `not (pdf_path.exists() and pdf_path.stat().st_size > 0)` を満たす経路を評価する。

    if not (pdf_path.exists() and pdf_path.stat().st_size > 0):
        raise SystemExit(f"[fail] missing: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    # 条件分岐: `not (0 <= int(args.page_index) < len(reader.pages))` を満たす経路を評価する。
    if not (0 <= int(args.page_index) < len(reader.pages)):
        raise SystemExit(f"[fail] invalid page index: {args.page_index} (pages={len(reader.pages)})")

    page = reader.pages[int(args.page_index)]

    paths = _extract_paths(page, reader)
    # 条件分岐: `not paths` を満たす経路を評価する。
    if not paths:
        raise SystemExit("[fail] no paths extracted")

    bbox = _infer_fig2_bbox(paths)
    y_zero = _infer_zero_line_y(paths, bbox=bbox)
    x0, x1, y0, y1 = bbox

    # Marker series definition (rounded RGB from the PDF).
    series_spec = {
        "TA_sq": {"fill_rgb": (0.93, 0.13, 0.14), "n_points": 5, "label": "TA (red squares)"},
        "TA_circ": {"fill_rgb": (0.73, 0.32, 0.62), "n_points": 25, "label": "TA (purple circles)"},
        "LA_pent": {"fill_rgb": (0.22, 0.33, 0.64), "n_points": 6, "label": "LA (blue pentagons)"},
        "LA_LO_hex": {"fill_rgb": (0.04, 0.51, 0.25), "n_points": 7, "label": "LA/LO (green hexagons)"},
        # Average line markers (dotted) appear as small 3-vertex closed paths (n=4 with closure)
        # with neutral fill color state; we identify them by n_points and stroke/fill equality.
        "avg": {"fill_rgb": (0.14, 0.12, 0.13), "n_points": 4, "label": "mean over 5 features (dotted)"},
    }

    # 関数: `_is_marker` の入出力契約と処理意図を定義する。
    def _is_marker(p: _PathRec) -> bool:
        # 条件分岐: `p.paint not in ("S", "s")` を満たす経路を評価する。
        if p.paint not in ("S", "s"):
            return False

        # 条件分岐: `not (x0 <= p.xmin <= x1 and x0 <= p.xmax <= x1 and y0 <= p.ymin <= y1 and y0...` を満たす経路を評価する。

        if not (x0 <= p.xmin <= x1 and x0 <= p.xmax <= x1 and y0 <= p.ymin <= y1 and y0 <= p.ymax <= y1):
            return False

        # 条件分岐: `not (3.0 <= p.w <= 9.0 and 3.0 <= p.h <= 9.0)` を満たす経路を評価する。

        if not (3.0 <= p.w <= 9.0 and 3.0 <= p.h <= 9.0):
            return False

        return True

    series_pts: dict[str, list[tuple[float, float]]] = {k: [] for k in series_spec}
    for p in paths:
        # 条件分岐: `not _is_marker(p)` を満たす経路を評価する。
        if not _is_marker(p):
            continue

        fill = _round_rgb(p.fill_rgb)
        for key, spec in series_spec.items():
            # 条件分岐: `fill != spec["fill_rgb"]` を満たす経路を評価する。
            if fill != spec["fill_rgb"]:
                continue

            # 条件分岐: `int(p.n_points) != int(spec["n_points"])` を満たす経路を評価する。

            if int(p.n_points) != int(spec["n_points"]):
                continue

            series_pts[key].append((p.cx, p.cy))

    missing = [k for k, pts in series_pts.items() if not pts]
    # 条件分岐: `missing` を満たす経路を評価する。
    if missing:
        raise SystemExit(f"[fail] missing marker series: {missing}")

    # Determine x positions of data points as the intersection across all series.

    tol_x = 0.6
    x_candidates = _dedup_sorted([x for x, _ in series_pts["LA_pent"]], tol=tol_x)
    x_data: list[float] = []
    for x in x_candidates:
        ok = True
        for key in ("TA_sq", "TA_circ", "LA_LO_hex", "avg"):
            xs = _dedup_sorted([xx for xx, _ in series_pts[key]], tol=tol_x)
            # 条件分岐: `min(abs(x - v) for v in xs) > tol_x` を満たす経路を評価する。
            if min(abs(x - v) for v in xs) > tol_x:
                ok = False
                break

        # 条件分岐: `ok` を満たす経路を評価する。

        if ok:
            x_data.append(float(x))

    x_data = sorted(x_data)

    temps_k = [int(s) for s in str(args.temps_k).split(",") if s.strip()]
    # 条件分岐: `len(x_data) != len(temps_k)` を満たす経路を評価する。
    if len(x_data) != len(temps_k):
        raise SystemExit(
            f"[fail] unexpected data point count: n_x={len(x_data)} vs n_T={len(temps_k)}\n"
            f"x_data={ [round(x,1) for x in x_data] }\n"
            f"temps_k={temps_k}"
        )

    # Collect y per series and build the derived 5th feature (TO/LO) via the mean line.

    rows_by_series: dict[str, list[dict[str, float]]] = {}
    for key, pts in series_pts.items():
        pts_u = sorted(pts, key=lambda t: t[0])
        rows: list[dict[str, float]] = []
        for x, t in zip(x_data, temps_k, strict=True):
            _x, y = _find_nearest(pts_u, x_target=x, tol_x=tol_x)
            rows.append({"t_K": float(t), "x_pdf": float(x), "y_pdf": float(y)})

        rows_by_series[key] = rows

    # Calibration: scale y-differences so that mean softening at T_max matches the manuscript mean (~0.07).

    mean_soft_tmax = float(args.mean_softening_at_tmax)
    # 条件分岐: `not math.isfinite(mean_soft_tmax)` を満たす経路を評価する。
    if not math.isfinite(mean_soft_tmax):
        mean_soft_tmax = _try_read_mean_softening_at_tmax(root, default_value=0.07)

    mean_soft_tmax = float(abs(mean_soft_tmax))

    t_max = float(max(temps_k))
    avg_at_tmax = next((r for r in rows_by_series["avg"] if float(r["t_K"]) == t_max), None)
    # 条件分岐: `avg_at_tmax is None` を満たす経路を評価する。
    if avg_at_tmax is None:
        raise SystemExit("[fail] missing avg point at T_max")

    y_avg_tmax = float(avg_at_tmax["y_pdf"])
    dy_avg = float(y_avg_tmax - y_zero)
    # 条件分岐: `dy_avg <= 0.0` を満たす経路を評価する。
    if dy_avg <= 0.0:
        raise SystemExit("[fail] invalid y calibration: avg(T_max) is not above y_zero")

    soft_per_y = float(mean_soft_tmax / dy_avg)

    # 関数: `to_softening_frac` の入出力契約と処理意図を定義する。
    def to_softening_frac(y_pdf: float) -> float:
        return float((float(y_pdf) - float(y_zero)) * soft_per_y)

    # Series values: s(T) = -Δε/ε (positive for softening).

    out_series: dict[str, Any] = {}
    for key, spec in series_spec.items():
        # 条件分岐: `key == "avg"` を満たす経路を評価する。
        if key == "avg":
            continue

        out_series[key] = {
            "label": str(spec["label"]),
            "rows": [
                {
                    "t_K": float(r["t_K"]),
                    "softening_frac_neg_dE_over_E": float(to_softening_frac(float(r["y_pdf"]))),
                    "omega_scale": float(1.0 - to_softening_frac(float(r["y_pdf"]))),
                }
                for r in rows_by_series[key]
            ],
        }

    # Derived 5th feature (TO/LO) from the average line:
    # y_avg = (y1 + y2 + y3 + y4 + y5)/5  => y5 = 5*y_avg - sum(y1..y4)

    derived_rows: list[dict[str, float]] = []
    for i, t in enumerate(temps_k):
        y_avg = float(rows_by_series["avg"][i]["y_pdf"])
        y_sum = float(rows_by_series["TA_sq"][i]["y_pdf"]) + float(rows_by_series["TA_circ"][i]["y_pdf"])
        y_sum += float(rows_by_series["LA_pent"][i]["y_pdf"]) + float(rows_by_series["LA_LO_hex"][i]["y_pdf"])
        y_tolo = float(5.0 * y_avg - y_sum)
        s = float(to_softening_frac(y_tolo))
        derived_rows.append({"t_K": float(t), "softening_frac_neg_dE_over_E": float(s), "omega_scale": float(1.0 - s)})

    out_obj: dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "dataset": "Kim2015 Fig.2: silicon phonon feature softening (digitized; -Δε/ε vs T)",
        "source": {
            "pdf_path": str(pdf_path),
            "pdf_sha256": _sha256(pdf_path),
            "page_index": int(args.page_index),
            "fig2_bbox_pdf": {"x_min": float(x0), "x_max": float(x1), "y_min": float(y0), "y_max": float(y1)},
            "y_zero_line_pdf": float(y_zero),
            "notes": [
                "Markers are digitized from vector paths (stroke outlines) with RGB fill colors in the published PDF.",
                "Temperature mapping uses the known experimental set used in Fig.2 (see --temps-k).",
                "Y-axis is calibrated so that the mean softening at T_max matches the manuscript mean (~0.07 at 1500 K).",
            ],
        },
        "calibration": {
            "t_max_K": float(t_max),
            "mean_softening_at_t_max": float(mean_soft_tmax),
            "avg_y_pdf_at_t_max": float(y_avg_tmax),
            "softening_per_pdf_y": float(soft_per_y),
        },
        "temps_K": [int(t) for t in temps_k],
        "series": out_series,
        "average": {
            "label": str(series_spec["avg"]["label"]),
            "rows": [
                {
                    "t_K": float(r["t_K"]),
                    "softening_frac_neg_dE_over_E": float(to_softening_frac(float(r["y_pdf"]))),
                    "omega_scale": float(1.0 - to_softening_frac(float(r["y_pdf"]))),
                }
                for r in rows_by_series["avg"]
            ],
        },
        "derived": {"TO_LO_triangles": {"label": "TO/LO (derived from mean-of-5)", "rows": derived_rows}},
    }

    out_json = src_dir / str(args.out_json_name)
    out_json.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_json}")

    # Diagnostic plot.
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "silicon_phonon_softening_kim2015_fig2_digitized.png"

    plt.figure(figsize=(9.6, 5.6), dpi=150)
    ts = [float(t) for t in temps_k]

    # 関数: `_plot_series` の入出力契約と処理意図を定義する。
    def _plot_series(label: str, rows: list[dict[str, float]], *, color: str) -> None:
        ys = [float(r["softening_frac_neg_dE_over_E"]) for r in rows]
        plt.plot(ts, ys, marker="o", ms=3, lw=1.2, color=color, label=label)

    _plot_series(out_series["TA_sq"]["label"], out_series["TA_sq"]["rows"], color="#e41a1c")
    _plot_series(out_series["TA_circ"]["label"], out_series["TA_circ"]["rows"], color="#984ea3")
    _plot_series(out_series["LA_pent"]["label"], out_series["LA_pent"]["rows"], color="#377eb8")
    _plot_series(out_series["LA_LO_hex"]["label"], out_series["LA_LO_hex"]["rows"], color="#4daf4a")
    _plot_series(out_obj["derived"]["TO_LO_triangles"]["label"], derived_rows, color="#000000")
    _plot_series(out_obj["average"]["label"], out_obj["average"]["rows"], color="#ff7f00")

    plt.xlabel("temperature T (K)")
    plt.ylabel("softening s(T) = -Δε/ε (dimensionless)")
    plt.title("Kim et al. PRB 91, 014307 (2015) Fig.2: digitized phonon feature softening")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"[ok] wrote: {out_png}")

    # Attach a reference to the digitized output in extracted_values.json.
    meta: dict[str, Any] = json.loads(extracted_json.read_text(encoding="utf-8"))
    meta.setdefault("digitized", {})
    meta["digitized"]["kim2015_fig2_phonon_feature_softening"] = {
        "updated_utc": _iso_utc_now(),
        "fig2_digitized_softening_json": str(out_json),
        "fig2_digitized_softening_json_sha256": _sha256(out_json),
        "diagnostic_png": str(out_png),
        "diagnostic_png_sha256": _sha256(out_png),
    }
    extracted_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] updated: {extracted_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
