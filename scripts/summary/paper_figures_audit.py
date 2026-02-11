#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_figures_audit.py

Phase 8 の「図が読める」ことを機械的にチェックするための簡易監査。

目的：
- doc/paper/10_manuscript.md（結果章）で参照される代表図について、
  (1) ファイルの存在、(2) 解像度（px）, (3) はみ出しの兆候（端まで非背景が張り付く）を点検し、
  修正対象（生成スクリプト側）をリストアップする。

出力（固定）：
- output/private/summary/paper_figures_audit.json
- output/private/summary/paper_figures_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.image as mpimg
except Exception as e:  # pragma: no cover
    raise SystemExit(f"[err] matplotlib is required to read PNGs: {e}")


ROOT = Path(__file__).resolve().parents[2]


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_result_figure_paths(manuscript_text: str) -> List[str]:
    """
    Extract `output/...png` like paths referenced in the Results chapter (4.2..4.12).

    We treat these as the "Table 1 referenced figures" (入口となる代表図) because Table 1
    summarizes 4.2..4.12.
    """

    start = re.search(r"^###\s+4\.2\s", manuscript_text, flags=re.M)
    end = re.search(r"^###\s+4\.13\s", manuscript_text, flags=re.M)
    sub = manuscript_text[start.start() : end.start()] if (start and end) else manuscript_text

    # NOTE: Manuscript uses backticks for fixed paths: `output/...png`
    pat = re.compile(r"`(output[\\/][^`]+?\.(?:png|jpg|jpeg))`", flags=re.I)
    paths = pat.findall(sub)
    # Normalize slashes to keep output stable across environments.
    norm = sorted({p.replace("\\", "/") for p in paths})
    return norm


def _as_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype.kind == "f":
        x = img.astype(np.float32, copy=False)
        if x.max() > 1.5:  # defensive: some readers may return 0..255 floats
            x = x / 255.0
        return np.clip(x, 0.0, 1.0)
    if img.dtype.kind in ("u", "i"):
        x = img.astype(np.float32, copy=False)
        if x.max() > 1.5:
            x = x / 255.0
        return np.clip(x, 0.0, 1.0)
    return img.astype(np.float32, copy=False)


def _median_bg_rgb(img01: np.ndarray) -> np.ndarray:
    if img01.ndim == 2:
        v = float(np.median([img01[0, 0], img01[0, -1], img01[-1, 0], img01[-1, -1]]))
        return np.array([v, v, v], dtype=np.float32)

    rgb = img01[:, :, :3]
    corners = np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]], axis=0)
    return np.median(corners, axis=0).astype(np.float32)


def _nonbg_mask(img01: np.ndarray, *, tol: float) -> np.ndarray:
    if img01.ndim == 2:
        bg = float(np.median([img01[0, 0], img01[0, -1], img01[-1, 0], img01[-1, -1]]))
        return np.abs(img01 - bg) > tol

    rgb = img01[:, :, :3]
    bg = _median_bg_rgb(img01)
    return np.any(np.abs(rgb - bg[None, None, :]) > tol, axis=2)


def _edge_nonbg_fraction(mask: np.ndarray, *, border_px: int) -> float:
    h, w = mask.shape
    b = max(1, int(border_px))
    border = np.zeros((h, w), dtype=bool)
    border[:b, :] = True
    border[-b:, :] = True
    border[:, :b] = True
    border[:, -b:] = True
    denom = int(border.sum())
    if denom <= 0:
        return 0.0
    return float((mask & border).sum()) / float(denom)


def _bbox_margins(mask: np.ndarray) -> Optional[Dict[str, int]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    h, w = mask.shape
    return {
        "left": x0,
        "right": (w - 1 - x1),
        "top": y0,
        "bottom": (h - 1 - y1),
    }


@dataclass(frozen=True)
class FigureAudit:
    path: str
    exists: bool
    width_px: Optional[int] = None
    height_px: Optional[int] = None
    min_size_ok: Optional[bool] = None
    edge_nonbg_frac: Optional[float] = None
    bbox_margins_px: Optional[Dict[str, int]] = None
    edge_touch: Optional[bool] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "exists": self.exists,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "min_size_ok": self.min_size_ok,
            "edge_nonbg_frac": self.edge_nonbg_frac,
            "bbox_margins_px": self.bbox_margins_px,
            "edge_touch": self.edge_touch,
            "notes": self.notes,
        }


def audit_figures(
    root: Path,
    *,
    min_width_px: int,
    min_height_px: int,
    border_px: int,
    edge_tol: float,
    edge_touch_margin_px: int,
) -> Tuple[List[str], List[FigureAudit]]:
    manuscript = root / "doc" / "paper" / "10_manuscript.md"
    paths = _extract_result_figure_paths(_read_text(manuscript))

    audits: List[FigureAudit] = []
    missing: List[str] = []
    for rel in paths:
        full = root / Path(rel)
        if not full.exists():
            missing.append(rel)
            audits.append(FigureAudit(path=rel, exists=False, notes="missing"))
            continue

        img = mpimg.imread(full)
        h, w = (int(img.shape[0]), int(img.shape[1])) if img.ndim >= 2 else (None, None)
        img01 = _as_float01(np.asarray(img))
        mask = _nonbg_mask(img01, tol=edge_tol)

        frac = _edge_nonbg_fraction(mask, border_px=border_px)
        margins = _bbox_margins(mask)
        touch = None
        if margins is not None:
            touch = any(int(v) <= int(edge_touch_margin_px) for v in margins.values())

        min_ok = (w is not None and h is not None and (w >= min_width_px and h >= min_height_px))
        note_parts: List[str] = []
        if not min_ok:
            note_parts.append(f"small(<{min_width_px}x{min_height_px})")
        if touch:
            note_parts.append("edge_touch")

        audits.append(
            FigureAudit(
                path=rel,
                exists=True,
                width_px=w,
                height_px=h,
                min_size_ok=bool(min_ok),
                edge_nonbg_frac=frac,
                bbox_margins_px=margins,
                edge_touch=touch,
                notes=";".join(note_parts),
            )
        )

    return missing, audits


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, audits: Sequence[FigureAudit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "path",
                "exists",
                "width_px",
                "height_px",
                "min_size_ok",
                "edge_nonbg_frac",
                "bbox_left",
                "bbox_right",
                "bbox_top",
                "bbox_bottom",
                "edge_touch",
                "notes",
            ]
        )
        for a in audits:
            margins = a.bbox_margins_px or {}
            w.writerow(
                [
                    a.path,
                    int(bool(a.exists)),
                    a.width_px if a.width_px is not None else "",
                    a.height_px if a.height_px is not None else "",
                    "" if a.min_size_ok is None else int(bool(a.min_size_ok)),
                    "" if a.edge_nonbg_frac is None else f"{a.edge_nonbg_frac:.6f}",
                    margins.get("left", ""),
                    margins.get("right", ""),
                    margins.get("top", ""),
                    margins.get("bottom", ""),
                    "" if a.edge_touch is None else int(bool(a.edge_touch)),
                    a.notes,
                ]
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit readability-related properties of key paper figures.")
    parser.add_argument("--out-dir", default=None, help="Override output directory (default: output/private/summary).")
    parser.add_argument("--min-width-px", type=int, default=1600)
    parser.add_argument("--min-height-px", type=int, default=700)
    parser.add_argument("--border-px", type=int, default=2)
    parser.add_argument("--edge-tol", type=float, default=0.06, help="Per-channel tolerance in 0..1 for non-bg.")
    parser.add_argument(
        "--edge-touch-margin-px",
        type=int,
        default=0,
        help="Treat bbox margin <= this as 'edge_touch' (0 is the strictest).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "output" / "private" / "summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    missing, audits = audit_figures(
        ROOT,
        min_width_px=int(args.min_width_px),
        min_height_px=int(args.min_height_px),
        border_px=int(args.border_px),
        edge_tol=float(args.edge_tol),
        edge_touch_margin_px=int(args.edge_touch_margin_px),
    )

    payload = {
        "generated_utc": _iso_utc_now(),
        "domain": "paper",
        "scope": "results_figures (manuscript 4.2..4.12)",
        "params": {
            "min_width_px": int(args.min_width_px),
            "min_height_px": int(args.min_height_px),
            "border_px": int(args.border_px),
            "edge_tol": float(args.edge_tol),
            "edge_touch_margin_px": int(args.edge_touch_margin_px),
        },
        "summary": {
            "figures_total": len(audits),
            "missing_total": len(missing),
            "small_total": sum(1 for a in audits if a.exists and a.min_size_ok is False),
            "edge_touch_total": sum(1 for a in audits if a.exists and a.edge_touch),
        },
        "missing": missing,
        "figures": [a.to_dict() for a in audits],
    }

    json_path = out_dir / "paper_figures_audit.json"
    csv_path = out_dir / "paper_figures_audit.csv"
    _write_json(json_path, payload)
    _write_csv(csv_path, audits)
    print(f"[ok] json: {json_path}")
    print(f"[ok] csv : {csv_path}")

    if missing:
        print(f"[warn] missing figures: {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
