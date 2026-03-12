#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_build.py

Phase 8（論文化・公開）向けの「ビルド入口」。

実行内容（既定）:
  1) 検証サマリ表 を生成（paper_tables.py）
  2) 論文HTMLを生成（paper_html.py）
  3) 整合チェック（paper_lint.py）
  4) TeX 生成・PDF 生成・TeX監査

出力（既定）:
  - output/private/summary/paper_table1_results.md（ほか .json/.csv）
  - profile=paper: output/private/summary/pmodel_paper.html（Part I）
  - profile=part2_astrophysics: output/private/summary/pmodel_paper_part2_astrophysics.html
  - profile=part3_quantum: output/private/summary/pmodel_paper_part3_quantum.html
  - profile=part4_verification: output/private/summary/pmodel_paper_part4_verification.html
  - profile=part5_future_predictions: output/private/summary/pmodel_paper_part5_future_predictions.html
  - profileごとの PDF（.tex から生成）
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cosmology import jwst_spectra_integration, jwst_spectra_release_waitlist
from scripts.gw import gw_multi_event_summary
from scripts.xrism import fek_relativistic_broadening_isco_constraints, xrism_integration
from scripts.summary import html_to_docx, paper_html, paper_latex, paper_lint, paper_pdf, paper_tables, paper_tex_audit, worklog


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return _ROOT


# 関数: `_run_best_effort` の入出力契約と処理意図を定義する。

def _run_best_effort(argv: list[str], *, cwd: Path, env_overrides: Optional[dict[str, str]] = None) -> None:
    try:
        env = os.environ.copy()
        if env_overrides:
            for k, v in env_overrides.items():
                env[str(k)] = str(v)

        subprocess.run(argv, cwd=str(cwd), check=True, env=env)
    except Exception as e:
        cmd = " ".join(str(x) for x in argv)
        print(f"[warn] pre-step failed (continuing): {cmd}\n  {e}")


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build paper artifacts (検証サマリ表 + HTML + lint).")
    ap.add_argument(
        "--profile",
        choices=["paper", "part2_astrophysics", "part3_quantum", "part4_verification", "part5_future_predictions"],
        default="paper",
        help="build profile: paper (Part I) / part2_astrophysics / part3_quantum / part4_verification / part5_future_predictions",
    )
    ap.add_argument(
        "--mode",
        choices=["publish", "internal"],
        default="publish",
        help="paper_html render mode (default: publish).",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory for paper artifacts (default: output/private/summary).",
    )
    ap.add_argument(
        "--figure-lang",
        choices=["auto", "ja", "en"],
        default="auto",
        help="Locale for helper-managed figure text (default: auto; Part III defaults to ja).",
    )
    ap.add_argument(
        "--figure-font-scale",
        type=float,
        default=1.0,
        help="Global scale applied to the shared figure font profile (default: 1.0).",
    )
    ap.add_argument(
        "--no-embed-images",
        action="store_true",
        help="Do not embed images in publish HTML (pass-through to paper_html).",
    )
    ap.add_argument("--skip-tables", action="store_true", help="Skip 検証サマリ表 generation.")
    ap.add_argument("--skip-lint", action="store_true", help="Skip paper_lint check.")
    ap.add_argument(
        "--with-docx",
        action="store_true",
        help="Enable HTML→DOCX export explicitly (default: disabled).",
    )
    ap.add_argument(
        "--skip-docx",
        dest="skip_docx",
        action="store_true",
        help="Skip HTML→DOCX export (kept for backward compatibility).",
    )
    ap.add_argument(
        "--docx-orientation",
        choices=["portrait", "landscape"],
        default="landscape",
        help="Page orientation for DOCX export (default: landscape).",
    )
    ap.add_argument(
        "--docx-margin-mm",
        type=float,
        default=7.0,
        help="Page margins in mm for DOCX export (default: 7; slightly safer than 5 for Word UI/print).",
    )
    ap.add_argument(
        "--skip-tex-audit",
        action="store_true",
        help="Skip strict post-build TeX audit (default: enabled).",
    )
    ap.add_argument(
        "--tex-audit-engine",
        choices=["auto", "lualatex", "xelatex", "pdflatex", "none"],
        default="auto",
        help="TeX compiler selection for audit (default: auto).",
    )
    ap.add_argument(
        "--tex-audit-require-engine",
        action="store_true",
        help="Fail audit when TeX compiler is unavailable.",
    )
    ap.add_argument(
        "--tex-audit-fail-on-overfull",
        action="store_true",
        help="Treat Overfull \\\\hbox warnings as errors in TeX audit.",
    )
    ap.add_argument("--skip-pdf", action="store_true", help="Skip TeX→PDF export step.")
    ap.add_argument(
        "--pdf-engine",
        choices=["auto", "lualatex", "xelatex", "pdflatex"],
        default="auto",
        help="TeX compiler selection for PDF build (default: auto).",
    )
    ap.add_argument(
        "--pdf-require-engine",
        action="store_true",
        help="Fail PDF build when no TeX compiler is available.",
    )
    ap.add_argument(
        "--pdf-fail-on-overfull",
        action="store_true",
        help="Treat Overfull \\\\hbox warnings as blocking in PDF build.",
    )
    ap.add_argument(
        "--sync-papers",
        action="store_true",
        help="(互換) papers同期を明示。現在は常時有効。",
    )
    ap.add_argument(
        "--papers-dir",
        default=str(_ROOT / "papers"),
        help="Destination directory used by --sync-papers (default: papers).",
    )
    args = ap.parse_args(argv)
    enable_docx = bool(args.with_docx) and (not bool(args.skip_docx))
    if bool(args.with_docx) and bool(args.skip_docx):
        print("[warn] --with-docx と --skip-docx が同時指定されたため DOCX はスキップします。")

    # 条件分岐: `args.skip_pdf` を満たす経路を評価する。

    if args.skip_pdf:
        print("[err] --skip-pdf は運用ルールで無効です（TeX更新時は必ずPDFを生成）。")
        print("[hint] --skip-pdf を外して再実行してください。")
        return 2

    # 運用固定: TeX更新時のPDFは常に papers/ へ同期する。

    sync_papers = True
    if not bool(args.sync_papers):
        print("[info] --sync-papers 未指定でも運用ルールにより papers 同期を強制します。")

    root = _repo_root()
    root_str = str(root)
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        parts = existing_pythonpath.split(os.pathsep)
        if root_str not in parts:
            os.environ["PYTHONPATH"] = root_str + os.pathsep + existing_pythonpath
    else:
        os.environ["PYTHONPATH"] = root_str

    out_dir = Path(args.outdir) if args.outdir else (root / "output" / "private" / "summary")
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = str(args.profile)
    requested_figure_lang = str(args.figure_lang)
    figure_font_scale = float(args.figure_font_scale)
    if figure_font_scale <= 0:
        print("[err] --figure-font-scale は 0 より大きい値が必要です。")
        return 2

    env_figure_lang = os.environ.get("WAVEP_FIGURE_LANG", "").strip().lower()
    auto_figure_lang = env_figure_lang if env_figure_lang in {"ja", "en"} else "ja"
    auto_lang_profiles = {"part3_quantum", "part4_verification", "part5_future_predictions"}
    figure_lang = auto_figure_lang if (requested_figure_lang == "auto" and profile in auto_lang_profiles) else requested_figure_lang
    font_profile_by_build_profile = {
        "paper": "paper",
        "part2_astrophysics": "part2_astrophysics",
        "part3_quantum": "part3_quantum",
        "part4_verification": "part4_verification",
        "part5_future_predictions": "part5_future_predictions",
    }
    font_floor_by_build_profile = {
        "paper": ("12.2", "12.2"),
        "part2_astrophysics": ("13.2", "13.2"),
        "part3_quantum": ("12.2", "12.2"),
        "part4_verification": ("14.0", "14.0"),
        "part5_future_predictions": ("12.8", "12.8"),
    }

    os.environ.setdefault("WAVEP_MPL_AUTOSAVE_VECTOR_PDF", "1")
    os.environ.setdefault("WAVEP_MPL_FONT_PROFILE", font_profile_by_build_profile.get(profile, "paper"))
    os.environ["WAVEP_MPL_FONT_SCALE"] = str(figure_font_scale)
    text_floor, legend_note_floor = font_floor_by_build_profile.get(profile, ("12.2", "12.2"))
    os.environ.setdefault("WAVEP_MPL_TEXT_MIN_FONT", text_floor)
    os.environ.setdefault("WAVEP_MPL_LEGEND_NOTE_MIN_FONT", legend_note_floor)
    if profile == "part3_quantum":
        if requested_figure_lang == "auto":
            os.environ.setdefault("WAVEP_FIGURE_LANG", figure_lang)
            if figure_lang == "ja":
                os.environ.setdefault("WAVEP_MPL_FORCE_JA_TEXT", "1")
            else:
                os.environ.pop("WAVEP_MPL_FORCE_JA_TEXT", None)
        else:
            os.environ["WAVEP_FIGURE_LANG"] = figure_lang
            if figure_lang == "ja":
                os.environ.setdefault("WAVEP_MPL_FORCE_JA_TEXT", "1")
            else:
                os.environ.pop("WAVEP_MPL_FORCE_JA_TEXT", None)
    elif profile == "part4_verification":
        # Part4 は図5/6基準で可読性を確保しつつ、過大な文字重なりを避ける。
        if requested_figure_lang == "auto":
            os.environ.setdefault("WAVEP_FIGURE_LANG", figure_lang)
        else:
            os.environ["WAVEP_FIGURE_LANG"] = figure_lang

        if figure_lang == "ja":
            os.environ.setdefault("WAVEP_MPL_FORCE_JA_TEXT", "1")
        else:
            os.environ.pop("WAVEP_MPL_FORCE_JA_TEXT", None)
    elif profile == "part5_future_predictions":
        if requested_figure_lang == "auto":
            os.environ.setdefault("WAVEP_FIGURE_LANG", figure_lang)
        else:
            os.environ["WAVEP_FIGURE_LANG"] = figure_lang

        if figure_lang == "ja":
            os.environ.setdefault("WAVEP_MPL_FORCE_JA_TEXT", "1")
        else:
            os.environ.pop("WAVEP_MPL_FORCE_JA_TEXT", None)

    # sitecustomize（フォント床上げ/PNG保存時のPDF sidecar）を現プロセスでも有効化する。
    # 既に読み込まれている可能性があるため、環境変数設定後に再読込して反映する。

    try:
        import importlib
        import sitecustomize

        importlib.reload(sitecustomize)
    except Exception:
        pass

    py = sys.executable or "python"

    # Best-effort refresh of Part III (quantum) figures/metrics so a single
    # `paper_build --profile part3_quantum` yields a consistent publish artifact.
    if profile == "part3_quantum":
        _run_best_effort(
            [py, "-B", str(root / "scripts" / "quantum" / "molecular_h2_baseline.py"), "--slug", "h2"],
            cwd=root,
        )
        _run_best_effort(
            [py, "-B", str(root / "scripts" / "quantum" / "molecular_h2_baseline.py"), "--slug", "hd"],
            cwd=root,
        )
        _run_best_effort(
            [py, "-B", str(root / "scripts" / "quantum" / "molecular_h2_baseline.py"), "--slug", "d2"],
            cwd=root,
        )
        _run_best_effort([py, "-B", str(root / "scripts" / "quantum" / "molecular_isotopic_scaling.py")], cwd=root)
        _run_best_effort(
            [py, "-B", str(root / "scripts" / "quantum" / "de_broglie_precision_alpha_consistency.py")], cwd=root
        )
        _run_best_effort(
            [py, "-B", str(root / "scripts" / "quantum" / "gravity_quantum_interference_delta_predictions.py")],
            cwd=root,
        )
        _run_best_effort([py, "-B", str(root / "scripts" / "quantum" / "electron_double_slit_interference.py")], cwd=root)
    elif profile == "part4_verification":
        # Part4 の図1/図2（scoreboard）は同一の控えめフォントで統一する。
        scoreboard_env = {
            "WAVEP_MPL_TEXT_MIN_FONT": "11",
            "WAVEP_MPL_LEGEND_NOTE_MIN_FONT": "11",
            "WAVEP_MPL_AUTOSAVE_VECTOR_PDF": "1",
        }
        public_summary_dir = root / "output" / "public" / "summary"
        public_summary_dir.mkdir(parents=True, exist_ok=True)
        _run_best_effort(
            [
                py,
                "-B",
                str(root / "scripts" / "summary" / "validation_scoreboard.py"),
                "--target-fig-h-in",
                "15.6",
            ],
            cwd=root,
            env_overrides=scoreboard_env,
        )
        _run_best_effort([py, "-B", str(root / "scripts" / "summary" / "quantum_scoreboard.py")], cwd=root, env_overrides=scoreboard_env)
        # 公開本文参照用に scoreboard を public/summary へ明示出力する。
        _run_best_effort(
            [
                py,
                "-B",
                str(root / "scripts" / "summary" / "validation_scoreboard.py"),
                "--target-fig-h-in",
                "15.6",
                "--out-json",
                str(public_summary_dir / "validation_scoreboard.json"),
                "--out-png",
                str(public_summary_dir / "validation_scoreboard.png"),
            ],
            cwd=root,
            env_overrides=scoreboard_env,
        )
        _run_best_effort(
            [
                py,
                "-B",
                str(root / "scripts" / "summary" / "quantum_scoreboard.py"),
                "--out-json",
                str(public_summary_dir / "quantum_scoreboard.json"),
                "--out-png",
                str(public_summary_dir / "quantum_scoreboard.png"),
            ],
            cwd=root,
            env_overrides=scoreboard_env,
        )
        _run_best_effort(
            [py, "-B", str(root / "scripts" / "summary" / "table1_part4_label_parity_audit.py")],
            cwd=root,
            env_overrides=scoreboard_env,
        )
        # 図4/7/8（同系列監査パック）を毎回再生成してフォント調整を確実に反映する。
        _run_best_effort([py, "-B", str(root / "scripts" / "quantum" / "born_route_a_proxy_constraints.py")], cwd=root)
        action_env = {
            "WAVEP_MPL_TEXT_MIN_FONT": "11",
            "WAVEP_MPL_LEGEND_NOTE_MIN_FONT": "11",
            "WAVEP_MPL_AUTOSAVE_VECTOR_PDF": "1",
        }
        # 図5のみは過大化を避けるため、低めフォント下限で再生成する。
        _run_best_effort(
            [py, "-B", str(root / "scripts" / "quantum" / "action_principle_el_derivation_audit.py")],
            cwd=root,
            env_overrides=action_env,
        )
        _run_best_effort([py, "-B", str(root / "scripts" / "quantum" / "derivation_parameter_falsification_pack.py")], cwd=root)
        _run_best_effort([py, "-B", str(root / "scripts" / "quantum" / "derivation_observable_chain_lock_audit.py")], cwd=root)
        # 図11近傍（Noether系）を毎回再生成して、フォント設定の反映漏れを防ぐ。
        _run_best_effort([py, "-B", str(root / "scripts" / "quantum" / "lagrangian_noether_observable_closure_audit.py")], cwd=root)
        _run_best_effort([py, "-B", str(root / "scripts" / "quantum" / "lagrangian_noether_observable_closure_drift_audit.py")], cwd=root)
        _run_best_effort([py, "-B", str(root / "scripts" / "quantum" / "lagrangian_noether_rotational_closure_audit.py")], cwd=root)
        # 図13, 93-100 は凡例/注記を優先してフォント下限を上げて再生成する。
        part4_legend_env = {
            "WAVEP_MPL_TEXT_MIN_FONT": "14",
            "WAVEP_MPL_LEGEND_NOTE_MIN_FONT": "14",
            "WAVEP_MPL_AUTOSAVE_VECTOR_PDF": "1",
        }
        _run_best_effort(
            [py, "-B", str(root / "scripts" / "quantum" / "quantum_measurement_dynamic_collapse_simulation.py")],
            cwd=root,
            env_overrides=part4_legend_env,
        )
        _run_best_effort(
            [py, "-B", str(root / "scripts" / "quantum" / "thermo_blackbody_peak_frequency_wavelength_product_holdout_splits.py")],
            cwd=root,
            env_overrides=part4_legend_env,
        )
        _run_best_effort(
            [py, "-B", str(root / "scripts" / "quantum" / "thermo_blackbody_peak_frequency_per_wavelength_holdout_splits.py")],
            cwd=root,
            env_overrides=part4_legend_env,
        )
        _run_best_effort([py, "-B", str(root / "scripts" / "llr" / "llr_operational_metrics_audit.py")], cwd=root, env_overrides=part4_legend_env)
        _run_best_effort([py, "-B", str(root / "scripts" / "llr" / "llr_precision_reaudit.py")], cwd=root, env_overrides=part4_legend_env)
        _run_best_effort(
            [py, "-B", str(root / "scripts" / "cosmology" / "cosmology_ddr_1pz_manual_trace_audit.py")],
            cwd=root,
            env_overrides=part4_legend_env,
        )
        _run_best_effort(
            [py, "-B", str(root / "scripts" / "cosmology" / "sparc_rotation_curve_pmodel_audit.py")],
            cwd=root,
            env_overrides=part4_legend_env,
        )
        _run_best_effort(
            [py, "-B", str(root / "scripts" / "cosmology" / "cosmology_cluster_collision_p_peak_offset_audit.py")],
            cwd=root,
            env_overrides=part4_legend_env,
        )
    elif profile == "part5_future_predictions":
        _run_best_effort([py, "-B", str(root / "scripts" / "summary" / "part5_future_predictions_timeline.py")], cwd=root)

    # 条件分岐: `not args.skip_tables` を満たす経路を評価する。

    if not args.skip_tables:
        # Best-effort refresh of lightweight inputs used by 検証サマリ表 so that
        # build_materials.bat (quick/full) yields an up-to-date 検証サマリ表 without manual pre-steps.
        try:
            gw_multi_event_summary.main([])
        except Exception:
            pass

        try:
            xrism_integration.main([])
        except Exception:
            pass

        try:
            jwst_spectra_release_waitlist.main([])
        except Exception:
            pass

        try:
            jwst_spectra_integration.main([])
        except Exception:
            pass

        table_argv: list[str] = []
        # 条件分岐: `args.outdir` を満たす経路を評価する。
        if args.outdir:
            table_argv += ["--out-dir", str(args.outdir)]

        rc = paper_tables.main(table_argv)
        # 条件分岐: `rc != 0` を満たす経路を評価する。
        if rc != 0:
            return rc

        # 条件分岐: `profile == "part3_quantum"` を満たす経路を評価する。

        if profile == "part3_quantum":
            _run_best_effort([py, "-B", str(root / "scripts" / "summary" / "quantum_scoreboard.py")], cwd=root)

    # Ensure summary figures referenced by the manuscript exist (best effort).

    try:
        gw_multi_event_summary.main([])
    except Exception:
        pass

    # 条件分岐: `profile == "part2_astrophysics"` を満たす経路を評価する。

    if profile == "part2_astrophysics":
        try:
            xrism_integration.main([])
        except Exception:
            pass

        try:
            # Build Fig (Fe-K ISCO proxy) robustly: --plot-only emits a placeholder when the CSV is missing.
            out_csv = root / "output" / "private" / "xrism" / "fek_relativistic_broadening_isco_constraints.csv"
            has_rows = False
            try:
                # 条件分岐: `out_csv.exists()` を満たす経路を評価する。
                if out_csv.exists():
                    with out_csv.open("r", encoding="utf-8") as f:
                        # Skip header; count any non-empty data line.
                        next(f, "")
                        for line in f:
                            # 条件分岐: `line.strip()` を満たす経路を評価する。
                            if line.strip():
                                has_rows = True
                                break
            except Exception:
                has_rows = False

            # 条件分岐: `has_rows` を満たす経路を評価する。

            if has_rows:
                fek_relativistic_broadening_isco_constraints.main(["--plot-only"])
            else:
                fek_relativistic_broadening_isco_constraints.main([])
        except Exception:
            pass

    html_argv: list[str] = ["--profile", profile, "--mode", str(args.mode)]
    # 条件分岐: `args.outdir` を満たす経路を評価する。
    if args.outdir:
        html_argv += ["--outdir", str(args.outdir)]

    # 条件分岐: `args.no_embed_images` を満たす経路を評価する。

    if args.no_embed_images:
        html_argv.append("--no-embed-images")

    rc = paper_html.main(html_argv)
    # 条件分岐: `rc != 0` を満たす経路を評価する。
    if rc != 0:
        return rc

    # 条件分岐: `not args.skip_lint` を満たす経路を評価する。

    if not args.skip_lint:
        lint_argv: list[str] = []
        # 条件分岐: `profile == "paper"` を満たす経路を評価する。
        if profile == "paper":
            lint_argv += ["--manuscript", "doc/paper/10_part1_core_theory.md"]
        # 条件分岐: 前段条件が不成立で、`profile == "part2_astrophysics"` を追加評価する。
        elif profile == "part2_astrophysics":
            lint_argv += ["--manuscript", "doc/paper/11_part2_astrophysics.md"]
        # 条件分岐: 前段条件が不成立で、`profile == "part3_quantum"` を追加評価する。
        elif profile == "part3_quantum":
            lint_argv += ["--manuscript", "doc/paper/12_part3_quantum.md"]
        # 条件分岐: 前段条件が不成立で、`profile == "part4_verification"` を追加評価する。
        elif profile == "part4_verification":
            lint_argv += ["--manuscript", "doc/paper/13_part4_verification.md"]
        elif profile == "part5_future_predictions":
            lint_argv += ["--manuscript", "doc/paper/14_part5_future_predictions.md"]

        rc = paper_lint.main(lint_argv)
        # 条件分岐: `rc != 0` を満たす経路を評価する。
        if rc != 0:
            return rc

    # 条件分岐: `profile == "paper"` を満たす経路を評価する。

    if profile == "paper":
        html_name = "pmodel_paper.html"
        docx_name = "pmodel_paper.docx"
        pdf_name = "pmodel_paper.pdf"
    # 条件分岐: 前段条件が不成立で、`profile == "part2_astrophysics"` を追加評価する。
    elif profile == "part2_astrophysics":
        html_name = "pmodel_paper_part2_astrophysics.html"
        docx_name = "pmodel_paper_part2_astrophysics.docx"
        pdf_name = "pmodel_paper_part2_astrophysics.pdf"
    # 条件分岐: 前段条件が不成立で、`profile == "part3_quantum"` を追加評価する。
    elif profile == "part3_quantum":
        html_name = "pmodel_paper_part3_quantum.html"
        docx_name = "pmodel_paper_part3_quantum.docx"
        pdf_name = "pmodel_paper_part3_quantum.pdf"
    # 条件分岐: 前段条件が不成立で、`profile == "part4_verification"` を追加評価する。
    elif profile == "part4_verification":
        html_name = "pmodel_paper_part4_verification.html"
        docx_name = "pmodel_paper_part4_verification.docx"
        pdf_name = "pmodel_paper_part4_verification.pdf"
    elif profile == "part5_future_predictions":
        html_name = "pmodel_paper_part5_future_predictions.html"
        docx_name = "pmodel_paper_part5_future_predictions.docx"
        pdf_name = "pmodel_paper_part5_future_predictions.pdf"
    else:  # pragma: no cover (guarded by argparse choices)
        raise ValueError(f"unknown profile: {profile}")

    paper_html_path = out_dir / html_name
    paper_docx_path = out_dir / docx_name
    # 条件分岐: `enable_docx` を満たす経路を評価する。
    if enable_docx:
        rc = html_to_docx.main(
            [
                "--in",
                str(paper_html_path),
                "--out",
                str(paper_docx_path),
                "--paper-equations",
                "--orientation",
                str(args.docx_orientation),
                "--margin-mm",
                str(float(args.docx_margin_mm)),
            ]
        )
        # 条件分岐: `rc == 3` を満たす経路を評価する。
        if rc == 3:
            # No supported Word backend found; treat as a non-fatal skip.
            print("[warn] DOCX export skipped (Microsoft Word not available).")
        # 条件分岐: 前段条件が不成立で、`rc != 0` を追加評価する。
        elif rc != 0:
            return rc
    else:
        print("[info] DOCX export disabled by default (enable with --with-docx).")

    # strict post-build TeX/PDF phase (Part I-IV common gate)

    tex_argv: list[str] = ["--profile", profile, "--outdir", str(out_dir)]
    rc = paper_latex.main(tex_argv)
    # 条件分岐: `rc != 0` を満たす経路を評価する。
    if rc != 0:
        return rc

    paper_pdf_path = out_dir / pdf_name
    papers_pdf_path = Path(str(args.papers_dir)) / pdf_name
    # 条件分岐: `not args.skip_pdf` を満たす経路を評価する。
    if not args.skip_pdf:
        pdf_argv: list[str] = [
            "--profile",
            profile,
            "--outdir",
            str(out_dir),
            "--engine",
            str(args.pdf_engine),
        ]
        # 条件分岐: `bool(args.pdf_require_engine)` を満たす経路を評価する。
        if bool(args.pdf_require_engine):
            pdf_argv.append("--require-engine")

        # 条件分岐: `bool(args.pdf_fail_on_overfull)` を満たす経路を評価する。

        if bool(args.pdf_fail_on_overfull):
            pdf_argv.append("--fail-on-overfull")

        # 条件分岐: `bool(args.sync_papers)` を満たす経路を評価する。

        if sync_papers:
            pdf_argv += ["--sync-papers", "--papers-dir", str(args.papers_dir)]

        rc = paper_pdf.main(pdf_argv)
        # 条件分岐: `rc != 0` を満たす経路を評価する。
        if rc != 0:
            return rc

        # 運用固定: papers 側PDFが存在しなければ失敗扱いにする。

        if not papers_pdf_path.exists():
            print(f"[err] papers PDF missing: {papers_pdf_path}")
            return 2

    # 条件分岐: `not args.skip_tex_audit` を満たす経路を評価する。

    if not args.skip_tex_audit:
        audit_argv: list[str] = [
            "--profile",
            profile,
            "--outdir",
            str(out_dir),
            "--engine",
            str(args.tex_audit_engine),
        ]
        # 条件分岐: `bool(args.tex_audit_require_engine)` を満たす経路を評価する。
        if bool(args.tex_audit_require_engine):
            audit_argv.append("--require-engine")

        # 条件分岐: `bool(args.tex_audit_fail_on_overfull)` を満たす経路を評価する。

        if bool(args.tex_audit_fail_on_overfull):
            audit_argv.append("--fail-on-overfull")

        rc = paper_tex_audit.main(audit_argv)
        # 条件分岐: `rc != 0` を満たす経路を評価する。
        if rc != 0:
            return rc

    try:
        worklog.append_event(
            {
                "event_type": "paper_build",
                "argv": list(argv) if argv is not None else None,
                "profile": profile,
                "mode": str(args.mode),
                "no_embed_images": bool(args.no_embed_images),
                "outputs": {
                    "paper_html": paper_html_path,
                    "paper_docx": paper_docx_path if (enable_docx and paper_docx_path.exists()) else None,
                    "paper_pdf": paper_pdf_path if ((not args.skip_pdf) and paper_pdf_path.exists()) else None,
                    "papers_pdf": (
                        papers_pdf_path
                        if (sync_papers and (not args.skip_pdf) and papers_pdf_path.exists())
                        else None
                    ),
                    "table1_md": out_dir / "paper_table1_results.md",
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] build: {paper_html_path}")
    # 条件分岐: `enable_docx and paper_docx_path.exists()` を満たす経路を評価する。
    if enable_docx and paper_docx_path.exists():
        print(f"[ok] docx : {paper_docx_path}")

    # 条件分岐: `(not args.skip_pdf) and paper_pdf_path.exists()` を満たす経路を評価する。

    if (not args.skip_pdf) and paper_pdf_path.exists():
        print(f"[ok] pdf  : {paper_pdf_path}")

    # 条件分岐: `bool(args.sync_papers) and (not args.skip_pdf) and papers_pdf_path.exists()` を満たす経路を評価する。

    if sync_papers and (not args.skip_pdf) and papers_pdf_path.exists():
        print(f"[ok] papers: {papers_pdf_path}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
