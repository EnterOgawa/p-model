#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
part2_figure_canvas_normalize.py

Part II manuscript が参照する公開図 PDF を、LaTeX の textwidth
（170 mm）基準の fixed-width page box へ正規化する。

目的は、`\\includegraphics[width=\\linewidth]{...}` で組版したときの
最終縮尺を図ごとに一致させ、タイトル / 軸ラベル / 凡例の見え方を
そろえることにある。

入力:
- output/private/summary/pmodel_paper_part2_astrophysics.tex
- output/public/**/<stem>.pdf

出力:
- output/public/**/<stem>.pdf （fixed-width 正規化上書き）
- output/public/**/<stem>.png （PDF から再レンダリング）
- output/private/**/<stem>.pdf / .png （存在時に同期）
- output/private/summary/part2_figure_canvas_normalization.json
- output/private/summary/part2_figure_canvas_normalization.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter, Transformation


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEX = ROOT / "output" / "private" / "summary" / "pmodel_paper_part2_astrophysics.tex"
DEFAULT_PUBLIC_ROOT = ROOT / "output" / "public"
DEFAULT_PRIVATE_ROOT = ROOT / "output" / "private"
DEFAULT_SUMMARY_FIGURES_ROOT = ROOT / "output" / "private" / "summary" / "figures"
DEFAULT_OUT_JSON = ROOT / "output" / "private" / "summary" / "part2_figure_canvas_normalization.json"
DEFAULT_OUT_CSV = ROOT / "output" / "private" / "summary" / "part2_figure_canvas_normalization.csv"
PDFTOCAIRO = Path(r"C:\texlive\2024\bin\windows\pdftocairo.exe")
TEXTWIDTH_MM = 170.0
TEXTHEIGHT_MM = 257.0
MM_PER_INCH = 25.4
TEXTWIDTH_IN = TEXTWIDTH_MM / MM_PER_INCH
FULL_TALL_HEIGHT_IN = (TEXTHEIGHT_MM / MM_PER_INCH) * 0.78
FIXED_CANVAS_SOURCE_STEMS = {
    "cassini_fig2_overlay_full",
    "cassini_fig2_overlay_zoom10d",
    "cassini_fig2_overlay_bestbeta_zoom10d",
    "cassini_fig2_residuals",
    "cassini_beta_sweep_rmse",
    "cassini_pds_vs_digitized",
}


# 関数: UTC 現在時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: Part II TeX から includegraphics の stem 一覧を抽出する。

def _extract_stems_from_tex(tex_path: Path) -> list[str]:
    stems: list[str] = []
    for line in tex_path.read_text(encoding="utf-8").splitlines():
        marker = r"\includegraphics["
        if marker not in line:
            continue

        try:
            path_token = line.rsplit("{", 1)[1].split("}", 1)[0]
        except Exception:
            continue

        stems.append(Path(path_token).stem)

    return stems


# 関数: public root 配下から stem に一致する PDF を一意に解決する。

def _resolve_public_pdf(public_root: Path, stem: str) -> Path:
    matches = sorted(public_root.glob(f"**/{stem}.pdf"))
    if not matches:
        raise FileNotFoundError(f"public pdf not found for stem={stem}")

    if len(matches) > 1:
        # 関数: `_score` の入出力契約と処理意図を定義する。
        def _score(path: Path) -> tuple[int, int]:
            normalized = str(path).replace("\\", "/")
            nested_bonus = 1 if ("/batch/" in normalized or "/out_llr/" in normalized) else 0
            return (nested_bonus, len(path.parts))

        matches = sorted(matches, key=_score, reverse=True)

    return matches[0]


# 関数: source aspect から target preset と page box を決める。

def _resolve_target_box(*, src_width_in: float, src_height_in: float) -> tuple[str, float, float, float]:
    if src_width_in <= 0.0 or src_height_in <= 0.0:
        raise ValueError("source figure size must be positive")

    scale = TEXTWIDTH_IN / src_width_in
    target_height_in = src_height_in * scale
    preset_name = "full"
    if target_height_in >= 5.4:
        preset_name = "full-tall"

    if target_height_in > (FULL_TALL_HEIGHT_IN + 0.05):
        raise ValueError(
            "target height exceeds full-tall limit: "
            f"src={src_width_in:.3f}x{src_height_in:.3f} in "
            f"-> dst={TEXTWIDTH_IN:.3f}x{target_height_in:.3f} in"
        )

    return (preset_name, TEXTWIDTH_IN, target_height_in, scale)


# 関数: 既存 PDF の page box から metrics 行を作る。

def _measure_existing_pdf(pdf_path: Path, *, stem: str, preset_name: str) -> dict[str, float | str]:
    reader = PdfReader(str(pdf_path))
    page = reader.pages[0]
    width_in = float(page.mediabox.width) / 72.0
    height_in = float(page.mediabox.height) / 72.0
    return {
        "stem": stem,
        "src_width_in": width_in,
        "src_height_in": height_in,
        "target_width_in": width_in,
        "target_height_in": height_in,
        "scale": 1.0,
        "crop_factor_applied": 1.0,
        "target_preset": preset_name,
    }


# 関数: PDF content を fixed textwidth page box へ詰め直した新 PDF を書き出す。

def _write_normalized_pdf(src_pdf: Path, dst_pdf: Path, *, stem: str) -> dict[str, float | str]:
    reader = PdfReader(str(src_pdf))
    src_page = reader.pages[0]
    src_box = src_page.mediabox
    src_width_pt = float(src_box.width)
    src_height_pt = float(src_box.height)
    src_width_in = src_width_pt / 72.0
    src_height_in = src_height_pt / 72.0

    target_preset, target_width_in, target_height_in, scale = _resolve_target_box(
        src_width_in=src_width_in,
        src_height_in=src_height_in,
    )
    target_width_pt = target_width_in * 72.0
    target_height_pt = target_height_in * 72.0

    writer = PdfWriter()
    dst_page = writer.add_blank_page(width=target_width_pt, height=target_height_pt)
    dst_page.merge_transformed_page(src_page, Transformation().scale(scale, scale))

    dst_pdf.parent.mkdir(parents=True, exist_ok=True)
    with dst_pdf.open("wb") as handle:
        writer.write(handle)

    return {
        "stem": stem,
        "src_width_in": src_width_in,
        "src_height_in": src_height_in,
        "target_width_in": target_width_in,
        "target_height_in": target_height_in,
        "scale": scale,
        "crop_factor_applied": 1.0,
        "target_preset": target_preset,
    }


# 関数: PDF から単一 PNG を再レンダリングする。

def _render_png_from_pdf(pdf_path: Path, png_path: Path, *, dpi: int = 220) -> None:
    if not PDFTOCAIRO.exists():
        raise FileNotFoundError(f"missing pdftocairo: {PDFTOCAIRO}")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    out_base = png_path.with_suffix("")
    subprocess.run(
        [
            str(PDFTOCAIRO),
            "-png",
            "-singlefile",
            "-r",
            str(int(dpi)),
            str(pdf_path),
            str(out_base),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


# 関数: public 正本の正規化結果を private 側へ同期する。

def _sync_private_copy(public_pdf: Path, private_root: Path) -> tuple[Path | None, Path | None]:
    try:
        topic = public_pdf.parent.name
    except Exception:
        return (None, None)

    private_pdf = private_root / topic / public_pdf.name
    private_png = private_pdf.with_suffix(".png")
    if not private_pdf.parent.exists():
        return (None, None)

    shutil.copy2(public_pdf, private_pdf)
    public_png = public_pdf.with_suffix(".png")
    if public_png.exists():
        shutil.copy2(public_png, private_png)

    return (private_pdf, private_png if private_png.exists() else None)


# 関数: summary/figures 側の実参照コピーを public 正本へ同期する。

def _sync_summary_figure_copy(public_pdf: Path, summary_root: Path) -> tuple[Path | None, Path | None]:
    summary_pdf = summary_root / public_pdf.name
    summary_png = summary_pdf.with_suffix(".png")
    if not summary_root.exists():
        return (None, None)

    shutil.copy2(public_pdf, summary_pdf)
    public_png = public_pdf.with_suffix(".png")
    if public_png.exists():
        shutil.copy2(public_png, summary_png)

    return (summary_pdf, summary_png if summary_png.exists() else None)


# 関数: fixed-canvas source を public 正本へ優先反映すべきか判定する。

def _resolve_fixed_canvas_source(public_pdf: Path, *, stem: str) -> Path | None:
    if stem not in FIXED_CANVAS_SOURCE_STEMS:
        return None

    topic = public_pdf.parent.name
    direct_pdf = ROOT / "output" / topic / public_pdf.name
    if direct_pdf.exists():
        return direct_pdf

    return None


# 関数: metrics 行を JSON / CSV の両方へ書き出す。

def _write_metrics(rows: list[dict[str, Any]], out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_utc": _utc_now_iso(),
        "textwidth_in": TEXTWIDTH_IN,
        "full_tall_height_in": FULL_TALL_HEIGHT_IN,
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "stem",
        "public_pdf",
        "public_png",
        "private_pdf",
        "private_png",
        "summary_pdf",
        "summary_png",
        "src_width_in",
        "src_height_in",
        "target_width_in",
        "target_height_in",
        "scale",
        "crop_factor_applied",
        "target_preset",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


# 関数: CLI 引数を解釈して Part II figure canvas 正規化を実行する。

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize Part II figure PDFs to fixed textwidth page boxes.")
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX, help="Part II generated TeX path.")
    parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT, help="Public output root.")
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT, help="Private output root.")
    parser.add_argument("--summary-figures-root", type=Path, default=DEFAULT_SUMMARY_FIGURES_ROOT, help="Summary figures mirror root.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON, help="Metrics JSON path.")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV, help="Metrics CSV path.")
    args = parser.parse_args(argv)

    stems = _extract_stems_from_tex(args.tex)
    rows: list[dict[str, Any]] = []
    for stem in stems:
        public_pdf = _resolve_public_pdf(args.public_root, stem)
        fixed_canvas_source = _resolve_fixed_canvas_source(public_pdf, stem=stem)
        if fixed_canvas_source is not None:
            shutil.copy2(fixed_canvas_source, public_pdf)
            metrics = _measure_existing_pdf(public_pdf, stem=stem, preset_name="fixed-canvas-source")
            print(f"[ok] fixed-canvas source kept: {public_pdf}")
        else:
            temp_pdf = public_pdf.with_name(public_pdf.stem + ".__normalized__.pdf")
            metrics = _write_normalized_pdf(public_pdf, temp_pdf, stem=stem)
            temp_pdf.replace(public_pdf)
            print(f"[ok] normalized: {public_pdf}")

        public_png = public_pdf.with_suffix(".png")
        _render_png_from_pdf(public_pdf, public_png)
        private_pdf, private_png = _sync_private_copy(public_pdf, args.private_root)
        summary_pdf, summary_png = _sync_summary_figure_copy(public_pdf, args.summary_figures_root)
        rows.append(
            {
                "public_pdf": str(public_pdf).replace("\\", "/"),
                "public_png": str(public_png).replace("\\", "/"),
                "private_pdf": str(private_pdf).replace("\\", "/") if private_pdf else "",
                "private_png": str(private_png).replace("\\", "/") if private_png else "",
                "summary_pdf": str(summary_pdf).replace("\\", "/") if summary_pdf else "",
                "summary_png": str(summary_png).replace("\\", "/") if summary_png else "",
                **metrics,
            }
        )

    _write_metrics(rows, args.out_json, args.out_csv)
    print(f"[ok] fixed-width normalization applied: textwidth={TEXTWIDTH_IN:.3f} in")
    print(f"[ok] metrics: {args.out_json}")
    print(f"[ok] metrics: {args.out_csv}")
    return 0


# 条件分岐: スクリプト直接実行時だけ main を呼ぶ。

if __name__ == "__main__":
    raise SystemExit(main())
