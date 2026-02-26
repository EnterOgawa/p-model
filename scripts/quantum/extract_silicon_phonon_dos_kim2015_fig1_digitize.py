from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


# 関数: `_trapz_xy` の入出力契約と処理意図を定義する。

def _trapz_xy(xs: list[float], ys: list[float]) -> float:
    # 条件分岐: `len(xs) != len(ys) or len(xs) < 2` を満たす経路を評価する。
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("invalid arrays for trapz")

    s = 0.0
    for i in range(1, len(xs)):
        dx = float(xs[i]) - float(xs[i - 1])
        # 条件分岐: `dx <= 0.0` を満たす経路を評価する。
        if dx <= 0.0:
            raise ValueError("xs must be strictly increasing")

        s += 0.5 * (float(ys[i - 1]) + float(ys[i])) * dx

    return float(s)


# 関数: `_mat_mul` の入出力契約と処理意図を定義する。

def _mat_mul(m1: tuple[float, float, float, float, float, float], m2: tuple[float, float, float, float, float, float]) -> tuple[float, float, float, float, float, float]:
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


# 関数: `_extract_stroke_paths` の入出力契約と処理意図を定義する。

def _extract_stroke_paths(page, reader: PdfReader) -> list[dict[str, object]]:
    cs = ContentStream(page.get_contents(), reader)
    ops = cs.operations

    ctm: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    stack: list[tuple[float, float, float, float, float, float]] = []

    cur: list[tuple[float, float]] = []
    cur_start: tuple[float, float] | None = None
    out: list[dict[str, object]] = []

    for operands, op in ops:
        # 条件分岐: `op == b"q"` を満たす経路を評価する。
        if op == b"q":
            stack.append(ctm)
            continue

        # 条件分岐: `op == b"Q"` を満たす経路を評価する。

        if op == b"Q":
            # 条件分岐: `stack` を満たす経路を評価する。
            if stack:
                ctm = stack.pop()

            continue

        # 条件分岐: `op == b"cm"` を満たす経路を評価する。

        if op == b"cm":
            a, b, c, d, e, f = map(float, operands)
            ctm = _mat_mul(ctm, (a, b, c, d, e, f))
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

        # 条件分岐: `op == b"h"` を満たす経路を評価する。

        if op == b"h":
            # 条件分岐: `cur and cur_start is not None` を満たす経路を評価する。
            if cur and cur_start is not None:
                cur.append(cur_start)

            continue

        # 条件分岐: `op == b"S"` を満たす経路を評価する。

        if op == b"S":
            # 条件分岐: `cur and len(cur) >= 2` を満たす経路を評価する。
            if cur and len(cur) >= 2:
                xs = [p[0] for p in cur]
                ys = [p[1] for p in cur]
                out.append(
                    {
                        "n_points": int(len(cur)),
                        "xmin": float(min(xs)),
                        "xmax": float(max(xs)),
                        "ymin": float(min(ys)),
                        "ymax": float(max(ys)),
                        "pts": [{"x": float(x), "y": float(y)} for x, y in cur],
                    }
                )

            cur = []
            cur_start = None
            continue

    return out


# 関数: `_extract_axis_ticks_from_strokes` の入出力契約と処理意図を定義する。

def _extract_axis_ticks_from_strokes(strokes: list[dict[str, object]]) -> dict[str, list[float]]:
    """
    Infer plot axis ticks from stroke segments.

    - X major ticks: vertical segments at the bottom axis with dy≈2 and dx≈0.
    - Y major ticks: horizontal segments at the left axis with dx≈2 and dy≈0.
    """
    x_ticks: list[float] = []
    y_ticks: list[float] = []

    for s in strokes:
        n = int(s["n_points"])
        # 条件分岐: `n != 2` を満たす経路を評価する。
        if n != 2:
            continue

        xmin = float(s["xmin"])
        xmax = float(s["xmax"])
        ymin = float(s["ymin"])
        ymax = float(s["ymax"])
        dx = float(xmax - xmin)
        dy = float(ymax - ymin)

        # X major ticks: near-vertical, small dy (~2), at plot bottom.
        if dx < 0.2 and 1.6 <= dy <= 2.6 and 250.0 <= ymin <= 400.0:
            x_ticks.append(0.5 * (xmin + xmax))

        # Y major ticks: near-horizontal, small dx (~2), at left axis.

        if dy < 0.2 and 1.6 <= dx <= 2.6 and xmin <= 110.0 and 250.0 <= ymin <= 800.0:
            y_ticks.append(0.5 * (ymin + ymax))

    # 関数: `unique_sorted` の入出力契約と処理意図を定義する。

    def unique_sorted(vals: list[float], *, tol: float) -> list[float]:
        out: list[float] = []
        for v in sorted(float(x) for x in vals):
            # 条件分岐: `not out or abs(v - out[-1]) > tol` を満たす経路を評価する。
            if not out or abs(v - out[-1]) > tol:
                out.append(v)

        return out

    x_ticks_u = unique_sorted(x_ticks, tol=0.2)
    y_ticks_u = unique_sorted(y_ticks, tol=0.5)
    return {"x_ticks": x_ticks_u, "y_ticks": y_ticks_u}


# 関数: `_find_best_major_y_ticks` の入出力契約と処理意図を定義する。

def _find_best_major_y_ticks(y_ticks: list[float]) -> list[float]:
    """
    Pick the most plausible 8 major ticks (0..0.35 at 0.05 steps) from candidates.
    """
    ys = sorted(float(y) for y in y_ticks)
    # 条件分岐: `len(ys) < 8` を満たす経路を評価する。
    if len(ys) < 8:
        raise ValueError(f"too few y ticks: n={len(ys)}")

    best: list[float] | None = None
    best_score = float("inf")
    for i in range(0, len(ys) - 7):
        chunk = ys[i : i + 8]
        steps = [chunk[j + 1] - chunk[j] for j in range(7)]
        med = sorted(steps)[len(steps) // 2]
        # 条件分岐: `med <= 0` を満たす経路を評価する。
        if med <= 0:
            continue
        # Score = variance of step sizes.

        score = sum((s - med) * (s - med) for s in steps) / float(len(steps))
        # 条件分岐: `score < best_score` を満たす経路を評価する。
        if score < best_score:
            best_score = score
            best = chunk

    # 条件分岐: `best is None` を満たす経路を評価する。

    if best is None:
        raise ValueError("cannot find major y tick sequence")

    return [float(y) for y in best]


# 関数: `_filter_curves` の入出力契約と処理意図を定義する。

def _filter_curves(
    strokes: list[dict[str, object]],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> list[dict[str, object]]:
    curves: list[dict[str, object]] = []
    for s in strokes:
        n = int(s["n_points"])
        # 条件分岐: `n < 30` を満たす経路を評価する。
        if n < 30:
            continue

        xmin = float(s["xmin"])
        xmax = float(s["xmax"])
        ymin = float(s["ymin"])
        ymax = float(s["ymax"])
        # 条件分岐: `not (x_min - 1.0 <= xmin <= x_max + 1.0 and x_min - 1.0 <= xmax <= x_max + 1.0)` を満たす経路を評価する。
        if not (x_min - 1.0 <= xmin <= x_max + 1.0 and x_min - 1.0 <= xmax <= x_max + 1.0):
            continue

        # 条件分岐: `not (y_min - 1.0 <= ymin <= y_max + 1.0 and y_min - 1.0 <= ymax <= y_max + 1.0)` を満たす経路を評価する。

        if not (y_min - 1.0 <= ymin <= y_max + 1.0 and y_min - 1.0 <= ymax <= y_max + 1.0):
            continue

        # 条件分岐: `(xmax - xmin) < 80.0` を満たす経路を評価する。

        if (xmax - xmin) < 80.0:
            continue

        # 条件分岐: `(ymax - ymin) < 5.0` を満たす経路を評価する。

        if (ymax - ymin) < 5.0:
            # Exclude horizontal grid lines.
            continue

        curves.append(s)

    # Sort by baseline y (ymin) ascending.

    curves.sort(key=lambda c: float(c["ymin"]))
    return curves


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Digitize temperature-dependent phonon DOS curves (Fig.1) from the published PDF\n"
            "Kim et al., Phys. Rev. B 91, 014307 (2015) cached via CaltechAUTHORS.\n"
            "This extraction targets the vector plot strokes and reconstructs g_T(ε) (normalized to unity)\n"
            "as a fixed primary constraint candidate for Phase 7 / Step 7.14.20."
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
        default=1,
        help="0-based PDF page index containing Fig.1 (default: 1).",
    )
    ap.add_argument(
        "--energy-max-mev",
        type=float,
        default=80.0,
        help="Right-edge energy label of Fig.1 x-axis (meV). Expected: 80.",
    )
    ap.add_argument(
        "--out-json-name",
        default="fig1_digitized_dos.json",
        help="Output JSON filename under the source directory.",
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
    strokes = _extract_stroke_paths(page, reader)
    # 条件分岐: `not strokes` を満たす経路を評価する。
    if not strokes:
        raise SystemExit("[fail] no stroke paths extracted")

    ticks = _extract_axis_ticks_from_strokes(strokes)
    x_ticks = ticks["x_ticks"]
    y_ticks = ticks["y_ticks"]
    # 条件分岐: `len(x_ticks) != 9` を満たす経路を評価する。
    if len(x_ticks) != 9:
        raise SystemExit(f"[fail] unexpected x major ticks: n={len(x_ticks)} (expected 9 for 0..80 by 10)")

    y_major = _find_best_major_y_ticks(y_ticks)
    # 条件分岐: `len(y_major) != 8` を満たす経路を評価する。
    if len(y_major) != 8:
        raise SystemExit(f"[fail] unexpected y major ticks: n={len(y_major)} (expected 8 for 0..0.35 by 0.05)")

    # Axis mapping (from tick marks).

    x0 = float(min(x_ticks))
    x1 = float(max(x_ticks))
    y0 = float(min(y_major))
    y1_major = float(max(y_major))
    # 条件分岐: `x1 <= x0 or y1_major <= y0` を満たす経路を評価する。
    if x1 <= x0 or y1_major <= y0:
        raise SystemExit("[fail] invalid tick-derived axis bounds")

    # Plot y-extent is larger than the last labeled major tick because curves are offset
    # for clarity. Infer the plot top from stroke extents within the x range.

    y_plot_max = y1_major
    for s in strokes:
        xmin_s = float(s["xmin"])
        xmax_s = float(s["xmax"])
        ymin_s = float(s["ymin"])
        ymax_s = float(s["ymax"])
        # 条件分岐: `not (x0 - 1.0 <= xmin_s <= x1 + 1.0 and x0 - 1.0 <= xmax_s <= x1 + 1.0)` を満たす経路を評価する。
        if not (x0 - 1.0 <= xmin_s <= x1 + 1.0 and x0 - 1.0 <= xmax_s <= x1 + 1.0):
            continue

        # 条件分岐: `ymin_s < y0 - 5.0` を満たす経路を評価する。

        if ymin_s < y0 - 5.0:
            continue

        y_plot_max = max(y_plot_max, ymax_s)

    # 条件分岐: `y_plot_max <= y0` を満たす経路を評価する。

    if y_plot_max <= y0:
        raise SystemExit("[fail] invalid plot y range")

    energy_max = float(args.energy_max_mev)
    # 条件分岐: `energy_max <= 0` を満たす経路を評価する。
    if energy_max <= 0:
        raise SystemExit("[fail] invalid --energy-max-mev")

    # y major ticks correspond to g in [0,0.35] at 0.05 steps.

    g_major = [0.05 * i for i in range(8)]
    dy = float(y_major[1] - y_major[0])
    # 条件分岐: `dy <= 0` を満たす経路を評価する。
    if dy <= 0:
        raise SystemExit("[fail] invalid y tick spacing")

    slope_g_per_y = float(0.05 / dy)

    # Curve candidates inside the plot bounding box.
    curves = _filter_curves(strokes, x_min=x0, x_max=x1, y_min=y0, y_max=y_plot_max)
    # 条件分岐: `len(curves) != 12` を満たす経路を評価する。
    if len(curves) != 12:
        raise SystemExit(f"[fail] unexpected curve count: n={len(curves)} (expected 12 temperatures)")

    # Temperature order (as described in the paper for Fig.1).

    temps_k = [100, 200, 300, 301, 600, 900, 1000, 1100, 1200, 1300, 1400, 1500]

    out_curves: list[dict[str, object]] = []
    for curve, t_k in zip(curves, temps_k, strict=True):
        pts = curve.get("pts")
        # 条件分岐: `not isinstance(pts, list) or not pts` を満たす経路を評価する。
        if not isinstance(pts, list) or not pts:
            raise SystemExit("[fail] curve points missing")

        y_base = float(curve["ymin"])
        g_base = float(slope_g_per_y * (y_base - y0))
        rows: list[dict[str, float]] = []
        for p in pts:
            # 条件分岐: `not isinstance(p, dict)` を満たす経路を評価する。
            if not isinstance(p, dict):
                continue

            x = float(p["x"])
            y = float(p["y"])
            e_mev = float((x - x0) * energy_max / (x1 - x0))
            g = float(slope_g_per_y * (y - y0) - g_base)
            # 条件分岐: `e_mev < -1e-6 or e_mev > energy_max + 1e-3` を満たす経路を評価する。
            if e_mev < -1e-6 or e_mev > energy_max + 1e-3:
                continue

            rows.append({"E_meV": float(e_mev), "g_per_meV_raw": float(g)})

        rows.sort(key=lambda r: float(r["E_meV"]))
        # Deduplicate by E (keep the last).
        dedup: dict[float, dict[str, float]] = {}
        for r in rows:
            key = round(float(r["E_meV"]), 6)
            dedup[key] = r

        rows_u = [dedup[k] for k in sorted(dedup.keys())]

        es = [float(r["E_meV"]) for r in rows_u]
        gs_raw = [max(0.0, float(r["g_per_meV_raw"])) for r in rows_u]
        area = _trapz_xy(es, gs_raw)
        # 条件分岐: `area <= 0.0` を満たす経路を評価する。
        if area <= 0.0:
            raise SystemExit(f"[fail] non-positive area for T={t_k}K")

        gs = [float(g / area) for g in gs_raw]

        out_rows = [{"E_meV": float(e), "g_per_meV": float(g)} for e, g in zip(es, gs, strict=True)]
        out_curves.append(
            {
                "temperature_K": int(t_k),
                "baseline_offset_g": float(g_base),
                "n_points": int(len(out_rows)),
                "area_before_norm": float(area),
                "rows": out_rows,
            }
        )

    # Write digitized dataset.

    out_obj: dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "dataset": "Kim2015 Fig.1: silicon phonon DOS g_T(ε) (digitized; normalized to unity)",
        "source": {
            "pdf_path": str(pdf_path),
            "pdf_sha256": _sha256(pdf_path),
            "page_index": int(args.page_index),
            "energy_max_meV": float(energy_max),
            "notes": [
                "Curves are digitized from vector stroke paths in the published PDF (CaltechAUTHORS mirror).",
                "g_T(ε) is normalized to unity per the figure caption; vertical offsets are removed by baseline subtraction.",
            ],
        },
        "axis_map": {
            "x": {"tick_x_pdf": [float(x) for x in x_ticks], "tick_E_meV": [10.0 * i for i in range(9)]},
            "y": {"tick_y_pdf": [float(y) for y in y_major], "tick_g_per_meV": [float(g) for g in g_major], "slope_g_per_y": float(slope_g_per_y)},
        },
        "curves": out_curves,
        "notes": [
            "Temperature assignment follows the ordering of vertically-offset curves (bottom→top) in Fig.1.",
            "This extraction is intended as a fixed primary constraint candidate for Step 7.14.20 (no fit parameters).",
        ],
    }

    out_json = src_dir / str(args.out_json_name)
    out_json.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_json}")

    # Diagnostic plot (no offsets; all curves overlaid).
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "silicon_phonon_dos_kim2015_fig1_digitized.png"

    plt.figure(figsize=(9.6, 6.0), dpi=150)
    for c in out_curves:
        t_k = int(c["temperature_K"])
        rows = c["rows"]
        es = [float(r["E_meV"]) for r in rows]
        gs = [float(r["g_per_meV"]) for r in rows]
        plt.plot(es, gs, lw=1.2, label=f"{t_k} K")

    plt.xlabel("energy ε (meV)")
    plt.ylabel("g_T(ε) (1/meV; normalized)")
    plt.title("Kim et al. PRB 91, 014307 (2015) Fig.1: digitized phonon DOS (offset removed)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=3, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"[ok] wrote: {out_png}")

    # Attach a reference to the digitized output in extracted_values.json (keep the original keys).
    meta: dict[str, Any] = json.loads(extracted_json.read_text(encoding="utf-8"))
    meta.setdefault("digitized", {})
    meta["digitized"]["kim2015_fig1_phonon_dos"] = {
        "updated_utc": _iso_utc_now(),
        "fig1_digitized_dos_json": str(out_json),
        "fig1_digitized_dos_json_sha256": _sha256(out_json),
        "diagnostic_png": str(out_png),
        "diagnostic_png_sha256": _sha256(out_png),
    }
    extracted_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] updated: {extracted_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
