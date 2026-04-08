#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
part3b_figure_canvas_normalize.py

Part III-B manuscript が参照する公開図 PDF を、LaTeX の textwidth
（170 mm）基準の fixed-width page box へ正規化する。

目的は、`\\includegraphics[width=\\linewidth]{...}` で組版したときの
最終縮尺を図ごとに一致させ、タイトル / 軸ラベル / 凡例 / 注記の
見え方を Part III-B 全体でそろえることにある。

入力:
- output/private/summary/pmodel_paper_part3b_quantum_verification.tex
- output/public/**/<stem>.pdf

出力:
- output/public/**/<stem>.pdf （fixed-width 正規化上書き）
- output/public/**/<stem>.png （PDF から再レンダリング）
- output/private/**/<stem>.pdf / .png （存在時に同期）
- output/private/summary/figures/<stem>.pdf / .png （flat mirror）
- output/private/summary/part3b_figure_canvas_normalization.json
- output/private/summary/part3b_figure_canvas_normalization.csv
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
DEFAULT_TEX = ROOT / "output" / "private" / "summary" / "pmodel_paper_part3b_quantum_verification.tex"
DEFAULT_PUBLIC_ROOT = ROOT / "output" / "public"
DEFAULT_PRIVATE_ROOT = ROOT / "output" / "private"
DEFAULT_SUMMARY_FIGURES_ROOT = ROOT / "output" / "private" / "summary" / "figures"
DEFAULT_OUT_JSON = ROOT / "output" / "private" / "summary" / "part3b_figure_canvas_normalization.json"
DEFAULT_OUT_CSV = ROOT / "output" / "private" / "summary" / "part3b_figure_canvas_normalization.csv"
PDFTOCAIRO = Path(r"C:\texlive\2024\bin\windows\pdftocairo.exe")
TEXTWIDTH_MM = 170.0
MM_PER_INCH = 25.4
TEXTWIDTH_IN = TEXTWIDTH_MM / MM_PER_INCH
SKIP_WIDTH_NORMALIZE_STEMS = {
    "bell_selection_sensitivity_summary",
    "falsification_pack",
    "nist_belltest_time_tag_bias",
    "nist_belltest_trial_based",
    "matter_wave_interference_precision_audit",
    "gravity_induced_decoherence",
    "photon_quantum_interference",
    "qed_vacuum_precision",
    "nuclear_binding_deuteron",
    "nuclear_np_scattering_baseline",
    "nuclear_effective_potential_two_range_fit_as_rs",
    "nuclear_effective_potential_pion_constrained_barrier_tail_kq_scan",
    "nuclear_effective_potential_pion_constrained_barrier_tail_channel_split_kq_scan_triplet_barrier_fraction_scan",
    "nuclear_binding_energy_frequency_mapping_minimal_additional_physics",
    "nuclear_binding_energy_frequency_mapping_theory_diff",
    "nuclear_binding_energy_frequency_mapping_differential_quantification",
    "nuclear_binding_energy_frequency_mapping_deuteron_verification",
    "nuclear_binding_energy_frequency_mapping_deuteron_two_body",
    "nuclear_binding_energy_frequency_mapping_alpha_verification",
    "nuclear_binding_light_nuclei",
    "nuclear_binding_energy_frequency_mapping_representative_nuclei",
    "nuclear_near_field_interference_two_mode_model",
    "nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei",
    "nuclear_binding_energy_frequency_mapping_differential_predictions",
    "cross_dataset_covariance",
    "systematics_decomposition_15items",
    "condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_model",
    "condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_optical_softening_linear_model",
    "condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_optical_softening_raman_shape_model",
    "condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_three_group_dos_softening_kim2015_linear_model",
    "molecular_isotopic_scaling",
    "thermo_blackbody_radiation_baseline",
    "thermo_blackbody_entropy_baseline",
    "nuclear_binding_energy_frequency_mapping_falsification_pack",
}


# 関数: fixed-canvas 正本を優先すべき stem かどうかを判定する。
def _should_skip_width_normalize(stem: str) -> bool:
    return any(stem == prefix or stem.startswith(prefix + "__") for prefix in SKIP_WIDTH_NORMALIZE_STEMS)


# 関数: UTC 現在時刻を ISO 8601 文字列で返す。

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: Part III-B TeX から includegraphics の stem 一覧を抽出する。

def _extract_stems_from_tex(tex_path: Path) -> list[str]:
    stems: list[str] = []
    seen: set[str] = set()
    for line in tex_path.read_text(encoding="utf-8").splitlines():
        marker = r"\includegraphics["
        if marker not in line:
            continue

        try:
            path_token = line.rsplit("{", 1)[1].split("}", 1)[0]
        except Exception:
            continue

        stem = Path(path_token).stem
        if stem in seen:
            continue

        seen.add(stem)
        stems.append(stem)

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
            nested_bonus = 1 if "/bell/" in normalized else 0
            return (nested_bonus, len(path.parts))

        matches = sorted(matches, key=_score, reverse=True)

    return matches[0]


# 関数: PDF content を fixed textwidth page box へ詰め直した新 PDF を書き出す。

def _write_normalized_pdf(src_pdf: Path, dst_pdf: Path, *, stem: str) -> dict[str, float | str]:
    reader = PdfReader(str(src_pdf))
    src_page = reader.pages[0]
    src_box = src_page.mediabox
    src_width_pt = float(src_box.width)
    src_height_pt = float(src_box.height)
    src_width_in = src_width_pt / 72.0
    src_height_in = src_height_pt / 72.0

    if src_width_in <= 0.0 or src_height_in <= 0.0:
        raise ValueError(f"source figure size must be positive: stem={stem}")

    scale = TEXTWIDTH_IN / src_width_in
    target_width_in = TEXTWIDTH_IN
    target_height_in = src_height_in * scale
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

def _sync_private_copy(public_pdf: Path, public_root: Path, private_root: Path) -> tuple[Path | None, Path | None]:
    try:
        relative = public_pdf.relative_to(public_root)
    except Exception:
        return (None, None)

    private_pdf = private_root / relative
    private_png = private_pdf.with_suffix(".png")
    private_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(public_pdf, private_pdf)
    public_png = public_pdf.with_suffix(".png")
    if public_png.exists():
        shutil.copy2(public_png, private_png)

    return (private_pdf, private_png if private_png.exists() else None)


# 関数: summary/figures 側の実参照コピーを public 正本へ同期する。

def _sync_summary_figure_copy(public_pdf: Path, summary_root: Path) -> tuple[Path | None, Path | None]:
    summary_pdf = summary_root / public_pdf.name
    summary_png = summary_pdf.with_suffix(".png")
    summary_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(public_pdf, summary_pdf)
    public_png = public_pdf.with_suffix(".png")
    if public_png.exists():
        shutil.copy2(public_png, summary_png)

    return (summary_pdf, summary_png if summary_png.exists() else None)


# 関数: metrics 行を JSON / CSV の両方へ書き出す。

def _write_metrics(rows: list[dict[str, Any]], out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_utc": _utc_now_iso(),
        "textwidth_in": TEXTWIDTH_IN,
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
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


# 関数: CLI 引数を解釈して Part III-B figure canvas 正規化を実行する。

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize Part III-B figure PDFs to fixed textwidth page boxes.")
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX, help="Part III-B generated TeX path.")
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
        if _should_skip_width_normalize(stem):
            metrics = {
                "stem": stem,
                "src_width_in": None,
                "src_height_in": None,
                "target_width_in": None,
                "target_height_in": None,
                "scale": None,
                "normalize_policy": "preserve_source_canvas",
            }
            print(f"[ok] preserved source canvas: {public_pdf}")
        else:
            temp_pdf = public_pdf.with_name(public_pdf.stem + ".__normalized__.pdf")
            metrics = _write_normalized_pdf(public_pdf, temp_pdf, stem=stem)
            temp_pdf.replace(public_pdf)
            print(f"[ok] normalized: {public_pdf}")

        public_png = public_pdf.with_suffix(".png")
        _render_png_from_pdf(public_pdf, public_png)
        private_pdf, private_png = _sync_private_copy(public_pdf, args.public_root, args.private_root)
        summary_pdf, summary_png = _sync_summary_figure_copy(public_pdf, args.summary_figures_root)
        rows.append(
            {
                "public_pdf": str(public_pdf).replace("\\", "/"),
                "public_png": str(public_png).replace("\\", "/"),
                "private_pdf": (str(private_pdf).replace("\\", "/") if private_pdf else None),
                "private_png": (str(private_png).replace("\\", "/") if private_png else None),
                "summary_pdf": (str(summary_pdf).replace("\\", "/") if summary_pdf else None),
                "summary_png": (str(summary_png).replace("\\", "/") if summary_png else None),
                **metrics,
            }
        )

    _write_metrics(rows, args.out_json, args.out_csv)
    print(f"[done] normalized {len(rows)} Part III-B figure(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
