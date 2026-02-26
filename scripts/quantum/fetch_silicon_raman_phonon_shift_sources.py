from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from pypdf import PdfReader


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `out_path.exists() and out_path.stat().st_size > 0` を満たす経路を評価する。
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] exists: {out_path}")
        return

    req = Request(url, headers={"User-Agent": "waveP/quantum-fetch"})
    with urlopen(req, timeout=60) as resp, out_path.open("wb") as f:
        f.write(resp.read())

    # 条件分岐: `out_path.stat().st_size == 0` を満たす経路を評価する。

    if out_path.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty file: {out_path}")

    print(f"[ok] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


def _mat_mul(m1: tuple[float, float, float, float, float, float], m2: tuple[float, float, float, float, float, float]) -> tuple[float, float, float, float, float, float]:
    # PDF affine matrices: (a,b,c,d,e,f)
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


def _mat_apply(m: tuple[float, float, float, float, float, float], x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = m
    return (a * x + c * y + e, b * x + d * y + f)


def _is_blue_rgb(fill: tuple[float, float, float]) -> bool:
    r, g, b = fill
    return abs(b - 1.0) < 1e-6 and abs(r - g) < 1e-6 and (0.0 <= r <= 1.0)


def _largest_gap_split(values: list[float]) -> float | None:
    # 条件分岐: `len(values) < 2` を満たす経路を評価する。
    if len(values) < 2:
        return None

    vs = sorted(float(v) for v in values)
    best_gap = 0.0
    best_mid: float | None = None
    for a, b in zip(vs, vs[1:]):
        gap = float(b) - float(a)
        # 条件分岐: `gap > best_gap` を満たす経路を評価する。
        if gap > best_gap:
            best_gap = gap
            best_mid = 0.5 * (float(a) + float(b))

    return best_mid


def _extract_blue_markers_from_form(
    *,
    form_stream_text: str,
    bbox_min: float = 0.4,
    bbox_max: float = 2.5,
) -> list[dict[str, float]]:
    """
    Digitize small filled markers from a vector PDF plot.

    The target plot in arXiv:2001.08458 is embedded as a Form XObject with all
    text converted to outlines (no BT/ET). Data markers are small filled shapes
    using blue-ish RGB fills (b=1, r=g in {0,0.42,0.698,0.976}).
    """
    ops = {
        "m",
        "l",
        "c",
        "h",
        "re",
        "rg",
        "g",
        "cm",
        "q",
        "Q",
        "f",
        "f*",
        "S",
        "s",
        "B",
        "b",
        "b*",
        "B*",
        "n",
    }

    tokens = re.split(r"\s+", form_stream_text.strip())
    state_stack: list[tuple[tuple[float, float, float, float, float, float], tuple[float, float, float]]] = []
    ctm = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    fill = (0.0, 0.0, 0.0)

    path_pts: list[tuple[float, float]] = []
    operand_stack: list[str] = []
    out: list[dict[str, float]] = []

    def flush(op: str) -> None:
        nonlocal path_pts
        # 条件分岐: `not path_pts` を満たす経路を評価する。
        if not path_pts:
            return

        # 条件分岐: `op in ("f", "f*") and _is_blue_rgb(fill)` を満たす経路を評価する。

        if op in ("f", "f*") and _is_blue_rgb(fill):
            xs = [p[0] for p in path_pts]
            ys = [p[1] for p in path_pts]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            w = float(xmax - xmin)
            h = float(ymax - ymin)
            # 条件分岐: `bbox_min <= w <= bbox_max and bbox_min <= h <= bbox_max` を満たす経路を評価する。
            if bbox_min <= w <= bbox_max and bbox_min <= h <= bbox_max:
                out.append(
                    {
                        "x": float(0.5 * (xmin + xmax)),
                        "y": float(0.5 * (ymin + ymax)),
                        "w": float(w),
                        "h": float(h),
                    }
                )

        path_pts = []

    for tok in tokens:
        # 条件分岐: `not tok` を満たす経路を評価する。
        if not tok:
            continue

        # 条件分岐: `tok in ops` を満たす経路を評価する。

        if tok in ops:
            op = tok
            # 条件分岐: `op == "q"` を満たす経路を評価する。
            if op == "q":
                state_stack.append((ctm, fill))
                continue

            # 条件分岐: `op == "Q"` を満たす経路を評価する。

            if op == "Q":
                # 条件分岐: `not state_stack` を満たす経路を評価する。
                if not state_stack:
                    continue

                ctm, fill = state_stack.pop()
                continue

            # 条件分岐: `op == "cm"` を満たす経路を評価する。

            if op == "cm":
                # 条件分岐: `len(operand_stack) >= 6` を満たす経路を評価する。
                if len(operand_stack) >= 6:
                    a, b, c, d, e, f_ = map(float, operand_stack[-6:])
                    operand_stack = operand_stack[:-6]
                    ctm = _mat_mul(ctm, (a, b, c, d, e, f_))

                continue

            # 条件分岐: `op == "rg"` を満たす経路を評価する。

            if op == "rg":
                # 条件分岐: `len(operand_stack) >= 3` を満たす経路を評価する。
                if len(operand_stack) >= 3:
                    r, g, b = map(float, operand_stack[-3:])
                    operand_stack = operand_stack[:-3]
                    fill = (float(r), float(g), float(b))

                continue

            # 条件分岐: `op == "g"` を満たす経路を評価する。

            if op == "g":
                # 条件分岐: `len(operand_stack) >= 1` を満たす経路を評価する。
                if len(operand_stack) >= 1:
                    gg = float(operand_stack[-1])
                    operand_stack = operand_stack[:-1]
                    fill = (gg, gg, gg)

                continue

            # 条件分岐: `op == "m"` を満たす経路を評価する。

            if op == "m":
                # 条件分岐: `len(operand_stack) >= 2` を満たす経路を評価する。
                if len(operand_stack) >= 2:
                    x, y = map(float, operand_stack[-2:])
                    operand_stack = operand_stack[:-2]
                    path_pts.append(_mat_apply(ctm, x, y))

                continue

            # 条件分岐: `op == "l"` を満たす経路を評価する。

            if op == "l":
                # 条件分岐: `len(operand_stack) >= 2` を満たす経路を評価する。
                if len(operand_stack) >= 2:
                    x, y = map(float, operand_stack[-2:])
                    operand_stack = operand_stack[:-2]
                    path_pts.append(_mat_apply(ctm, x, y))

                continue

            # 条件分岐: `op == "c"` を満たす経路を評価する。

            if op == "c":
                # 条件分岐: `len(operand_stack) >= 6` を満たす経路を評価する。
                if len(operand_stack) >= 6:
                    x1, y1, x2, y2, x3, y3 = map(float, operand_stack[-6:])
                    operand_stack = operand_stack[:-6]
                    path_pts.extend(
                        [
                            _mat_apply(ctm, x1, y1),
                            _mat_apply(ctm, x2, y2),
                            _mat_apply(ctm, x3, y3),
                        ]
                    )

                continue

            # 条件分岐: `op == "re"` を満たす経路を評価する。

            if op == "re":
                # 条件分岐: `len(operand_stack) >= 4` を満たす経路を評価する。
                if len(operand_stack) >= 4:
                    x, y, w, h = map(float, operand_stack[-4:])
                    operand_stack = operand_stack[:-4]
                    pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                    path_pts.extend([_mat_apply(ctm, px, py) for px, py in pts])

                continue

            # 条件分岐: `op in ("f", "f*", "S", "s", "B", "b", "b*", "B*")` を満たす経路を評価する。

            if op in ("f", "f*", "S", "s", "B", "b", "b*", "B*"):
                flush(op)
                continue

            # 条件分岐: `op == "n"` を満たす経路を評価する。

            if op == "n":
                path_pts = []
                continue
            # ignore h and others

            continue

        # operand

        try:
            float(tok)
            operand_stack.append(tok)
        except ValueError:
            # ignore non-numeric names
            continue

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch a primary Raman ω(T) dataset for silicon (optical phonon near 520 cm^-1) "
            "and digitize the frequency-vs-temperature markers from the vector PDF plot. "
            "Intended for Phase 7 / Step 7.14.20 (anharmonic phonon softening constraint)."
        )
    )
    ap.add_argument("--offline", action="store_true", help="Do not download; only verify expected files exist.")
    ap.add_argument(
        "--out-dirname",
        default="arxiv_2001_08458_si_raman_phonon_shift",
        help="Output directory name under data/quantum/sources/.",
    )
    ap.add_argument(
        "--t-min-k",
        type=float,
        default=4.0,
        help="Minimum temperature shown in the plot (used to map x coordinates to T).",
    )
    ap.add_argument(
        "--t-max-k",
        type=float,
        default=623.0,
        help="Maximum temperature shown in the plot (used to map x coordinates to T).",
    )
    ap.add_argument(
        "--page-index",
        type=int,
        default=1,
        help="0-based page index containing the frequency-vs-temperature plot Form XObject.",
    )
    ap.add_argument(
        "--xobject-name",
        default="Meta23",
        help="Form XObject name containing the multi-panel plot (without leading slash).",
    )
    args = ap.parse_args()

    root = _repo_root()
    src_dir = root / "data" / "quantum" / "sources" / str(args.out_dirname)
    src_dir.mkdir(parents=True, exist_ok=True)

    url_pdf = "https://arxiv.org/pdf/2001.08458.pdf"
    pdf_path = src_dir / "arxiv_2001_08458v1.pdf"
    # 条件分岐: `not args.offline` を満たす経路を評価する。
    if not args.offline:
        _download(url_pdf, pdf_path)

    # 条件分岐: `not pdf_path.exists() or pdf_path.stat().st_size == 0` を満たす経路を評価する。

    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise SystemExit(f"[fail] missing: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    # 条件分岐: `not (0 <= int(args.page_index) < len(reader.pages))` を満たす経路を評価する。
    if not (0 <= int(args.page_index) < len(reader.pages)):
        raise SystemExit(f"[fail] page_index out of range: {args.page_index} (pages={len(reader.pages)})")

    page = reader.pages[int(args.page_index)]
    xobj = page.get("/Resources", {}).get("/XObject", {})
    xname = "/" + str(args.xobject_name).lstrip("/")
    # 条件分岐: `xname not in xobj` を満たす経路を評価する。
    if xname not in xobj:
        raise SystemExit(f"[fail] missing XObject {xname} on page {args.page_index}")

    form = xobj[xname].get_object()
    form_text = form.get_data().decode("latin1", errors="replace")
    markers = _extract_blue_markers_from_form(form_stream_text=form_text)
    # 条件分岐: `len(markers) < 20` を満たす経路を評価する。
    if len(markers) < 20:
        raise SystemExit(f"[fail] too few markers extracted: n={len(markers)}")

    xs = [float(m["x"]) for m in markers]
    x_split = _largest_gap_split(xs)
    # 条件分岐: `x_split is None` を満たす経路を評価する。
    if x_split is None:
        raise SystemExit("[fail] cannot find x split (largest gap) for panel separation")

    left = [m for m in markers if float(m["x"]) < float(x_split)]
    right = [m for m in markers if float(m["x"]) >= float(x_split)]
    # 条件分岐: `not left or not right` を満たす経路を評価する。
    if not left or not right:
        raise SystemExit("[fail] panel split produced empty side")

    # Choose the left-most panel as the ω(T) panel (lower median x).

    left_med = sorted(float(m["x"]) for m in left)[len(left) // 2]
    right_med = sorted(float(m["x"]) for m in right)[len(right) // 2]
    panel = left if left_med <= right_med else right

    # Split main vs inset by y largest gap.
    ys = [float(m["y"]) for m in panel]
    y_split = _largest_gap_split(ys)
    # 条件分岐: `y_split is None` を満たす経路を評価する。
    if y_split is None:
        raise SystemExit("[fail] cannot find y split (largest gap) for inset separation")

    main = [m for m in panel if float(m["y"]) >= float(y_split)]
    # 条件分岐: `len(main) < 20` を満たす経路を評価する。
    if len(main) < 20:
        # Fallback: treat all panel points as main.
        main = panel

    x_min = min(float(m["x"]) for m in main)
    x_max = max(float(m["x"]) for m in main)
    # 条件分岐: `x_max <= x_min` を満たす経路を評価する。
    if x_max <= x_min:
        raise SystemExit("[fail] invalid x range for main panel")

    # Map x (PDF plot units) to T (assume linear axis).

    t_min = float(args.t_min_k)
    t_max = float(args.t_max_k)
    # 条件分岐: `t_max <= t_min` を満たす経路を評価する。
    if t_max <= t_min:
        raise SystemExit("[fail] invalid t range")

    y_min = min(float(m["y"]) for m in main)
    y_max = max(float(m["y"]) for m in main)
    # 条件分岐: `y_max <= y_min` を満たす経路を評価する。
    if y_max <= y_min:
        raise SystemExit("[fail] invalid y range for main panel")

    rows: list[dict[str, float]] = []
    for m in main:
        x = float(m["x"])
        y = float(m["y"])
        t_k = t_min + (x - x_min) * (t_max - t_min) / (x_max - x_min)
        # Dimensionless shape: 0 at low T (y=y_max), 1 at high T (y=y_min).
        soft_shape = (y_max - y) / (y_max - y_min)
        rows.append(
            {
                "t_k": float(t_k),
                "x_pdf": float(x),
                "y_pdf": float(y),
                "softening_shape_0to1": float(soft_shape),
            }
        )

    rows.sort(key=lambda r: float(r["t_k"]))

    extracted_obj: dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "dataset": "Phase 7 / Step 7.14 silicon Raman optical-phonon ω(T) (digitized from arXiv:2001.08458 PDF)",
        "source": {"url": url_pdf, "local_path": str(pdf_path), "local_sha256": _sha256(pdf_path)},
        "digitize": {
            "pdf": {"page_index": int(args.page_index), "xobject_name": str(args.xobject_name)},
            "marker_filter": {"bbox_min": 0.4, "bbox_max": 2.5, "fill_rule": "blue-ish rgb with b=1 and r=g"},
            "panel_split": {"x_split": float(x_split), "chosen_panel": "left" if left_med <= right_med else "right"},
            "inset_split": {"y_split": float(y_split), "main_points": int(len(main)), "total_panel_points": int(len(panel))},
            "axis_map": {
                "t_min_k": float(t_min),
                "t_max_k": float(t_max),
                "x_min_pdf": float(x_min),
                "x_max_pdf": float(x_max),
                "y_min_pdf": float(y_min),
                "y_max_pdf": float(y_max),
                "notes": [
                    "T mapping assumes a linear temperature axis spanning the measurement range reported in the abstract (4–623 K).",
                    "softening_shape_0to1 is derived from y_pdf only; it is intended as a fixed-shape proxy for optical phonon softening.",
                    "Absolute ω(T) in cm^-1 is not reconstructed here because all plot text is outlined; use this dataset as a shape constraint.",
                ],
            },
        },
        "rows": rows,
        "notes": [
            "This extraction digitizes the plotted ω(T) markers from a vector PDF by bounding-box filtering of small filled shapes.",
            "The source paper discusses anomalous low-T phonon softening and negative thermal expansion in silicon.",
        ],
    }

    out_extracted = src_dir / "extracted_values.json"
    out_extracted.write_text(json.dumps(extracted_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "dataset": extracted_obj["dataset"],
        "notes": [
            "Cache the primary PDF for offline reproducibility.",
            "extracted_values.json is derived from the cached PDF via digitization of vector plot markers.",
        ],
        "files": [
            {
                "name": pdf_path.name,
                "url": url_pdf,
                "path": str(pdf_path),
                "bytes": int(pdf_path.stat().st_size),
                "sha256": _sha256(pdf_path).upper(),
            }
        ],
    }
    out_manifest = src_dir / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[ok] wrote: {out_extracted}")
    print(f"[ok] wrote: {out_manifest}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()

