#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_build.py

Phase 8（論文化・公開）向けの「ビルド入口」。

実行内容（既定）:
  1) Table 1 を生成（paper_tables.py）
  2) 論文HTMLを生成（paper_html.py）
  3) 整合チェック（paper_lint.py）

出力（既定）:
  - output/private/summary/paper_table1_results.md（ほか .json/.csv）
  - profile=paper: output/private/summary/pmodel_paper.html（Part I; + .docx）
  - profile=part2_astrophysics: output/private/summary/pmodel_paper_part2_astrophysics.html（+ .docx）
  - profile=part3_quantum: output/private/summary/pmodel_paper_part3_quantum.html（+ .docx）
  - profile=part4_verification: output/private/summary/pmodel_paper_part4_verification.html（+ .docx）
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cosmology import jwst_spectra_integration, jwst_spectra_release_waitlist
from scripts.gw import gw_multi_event_summary
from scripts.xrism import fek_relativistic_broadening_isco_constraints, xrism_integration
from scripts.summary import html_to_docx, paper_html, paper_lint, paper_tables, worklog


def _repo_root() -> Path:
    return _ROOT


def _run_best_effort(argv: list[str], *, cwd: Path) -> None:
    try:
        subprocess.run(argv, cwd=str(cwd), check=True)
    except Exception as e:
        cmd = " ".join(str(x) for x in argv)
        print(f"[warn] pre-step failed (continuing): {cmd}\n  {e}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build paper artifacts (Table 1 + HTML + lint).")
    ap.add_argument(
        "--profile",
        choices=["paper", "part2_astrophysics", "part3_quantum", "part4_verification"],
        default="paper",
        help="build profile: paper (Part I) / part2_astrophysics / part3_quantum / part4_verification",
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
        "--no-embed-images",
        action="store_true",
        help="Do not embed images in publish HTML (pass-through to paper_html).",
    )
    ap.add_argument("--skip-tables", action="store_true", help="Skip Table 1 generation.")
    ap.add_argument("--skip-lint", action="store_true", help="Skip paper_lint check.")
    ap.add_argument(
        "--skip-docx",
        dest="skip_docx",
        action="store_true",
        help="Skip HTML→DOCX export (default: enabled when Microsoft Word is available).",
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
    # Backward-compatible alias (historical name). Keep it, but generate DOCX instead of PDF.
    ap.add_argument("--skip-pdf", dest="skip_docx", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    root = _repo_root()
    out_dir = Path(args.outdir) if args.outdir else (root / "output" / "private" / "summary")
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = str(args.profile)
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

    if not args.skip_tables:
        # Best-effort refresh of lightweight inputs used by Table 1 so that
        # build_materials.bat (quick/full) yields an up-to-date Table 1 without manual pre-steps.
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
        if args.outdir:
            table_argv += ["--out-dir", str(args.outdir)]
        rc = paper_tables.main(table_argv)
        if rc != 0:
            return rc

        if profile == "part3_quantum":
            _run_best_effort([py, "-B", str(root / "scripts" / "summary" / "quantum_scoreboard.py")], cwd=root)

    # Ensure summary figures referenced by the manuscript exist (best effort).
    try:
        gw_multi_event_summary.main([])
    except Exception:
        pass
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
                if out_csv.exists():
                    with out_csv.open("r", encoding="utf-8") as f:
                        # Skip header; count any non-empty data line.
                        next(f, "")
                        for line in f:
                            if line.strip():
                                has_rows = True
                                break
            except Exception:
                has_rows = False
            if has_rows:
                fek_relativistic_broadening_isco_constraints.main(["--plot-only"])
            else:
                fek_relativistic_broadening_isco_constraints.main([])
        except Exception:
            pass

    html_argv: list[str] = ["--profile", profile, "--mode", str(args.mode)]
    if args.outdir:
        html_argv += ["--outdir", str(args.outdir)]
    if args.no_embed_images:
        html_argv.append("--no-embed-images")
    rc = paper_html.main(html_argv)
    if rc != 0:
        return rc

    if not args.skip_lint:
        lint_argv: list[str] = []
        if profile == "paper":
            lint_argv += ["--manuscript", "doc/paper/10_part1_core_theory.md"]
        elif profile == "part2_astrophysics":
            lint_argv += ["--manuscript", "doc/paper/11_part2_astrophysics.md"]
        elif profile == "part3_quantum":
            lint_argv += ["--manuscript", "doc/paper/12_part3_quantum.md"]
        elif profile == "part4_verification":
            lint_argv += ["--manuscript", "doc/paper/13_part4_verification.md"]
        rc = paper_lint.main(lint_argv)
        if rc != 0:
            return rc

    if profile == "paper":
        html_name = "pmodel_paper.html"
        docx_name = "pmodel_paper.docx"
    elif profile == "part2_astrophysics":
        html_name = "pmodel_paper_part2_astrophysics.html"
        docx_name = "pmodel_paper_part2_astrophysics.docx"
    elif profile == "part3_quantum":
        html_name = "pmodel_paper_part3_quantum.html"
        docx_name = "pmodel_paper_part3_quantum.docx"
    elif profile == "part4_verification":
        html_name = "pmodel_paper_part4_verification.html"
        docx_name = "pmodel_paper_part4_verification.docx"
    else:  # pragma: no cover (guarded by argparse choices)
        raise ValueError(f"unknown profile: {profile}")
    paper_html_path = out_dir / html_name
    paper_docx_path = out_dir / docx_name
    if not args.skip_docx:
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
        if rc == 3:
            # No supported Word backend found; treat as a non-fatal skip.
            print("[warn] DOCX export skipped (Microsoft Word not available).")
        elif rc != 0:
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
                    "paper_docx": (
                        paper_docx_path if ((not args.skip_docx) and paper_docx_path.exists()) else None
                    ),
                    "table1_md": out_dir / "paper_table1_results.md",
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] build: {paper_html_path}")
    if (not args.skip_docx) and paper_docx_path.exists():
        print(f"[ok] docx : {paper_docx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
