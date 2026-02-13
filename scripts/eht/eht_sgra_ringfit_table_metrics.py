#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = [
            "Yu Gothic",
            "Meiryo",
            "BIZ UDGothic",
            "MS Gothic",
            "Yu Mincho",
            "MS Mincho",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


_RE_PM = re.compile(r"\$(?P<mean>-?\d+(?:\.\d+)?)\s*\\pm\s*(?P<sigma>\d+(?:\.\d+)?)\s*\$")


def _parse_pm(cell: str) -> Optional[Tuple[float, float]]:
    m = _RE_PM.search(cell)
    if not m:
        return None
    return float(m.group("mean")), float(m.group("sigma"))


@dataclass(frozen=True)
class RingFitRow:
    table: str  # "descattered" or "on_sky"
    pipeline: str
    cluster: str
    fit_method: str
    d_mean_uas: float
    d_sigma_uas: float
    w_mean_uas: Optional[float]
    w_sigma_uas: Optional[float]
    eta_mean_deg: Optional[float]
    eta_sigma_deg: Optional[float]
    a_mean: Optional[float]
    a_sigma: Optional[float]
    fc_mean: Optional[float]
    fc_sigma: Optional[float]
    source: Dict[str, Any]


@dataclass(frozen=True)
class RingFitSummaryRow:
    table: str  # "descattered" or "on_sky"
    pipeline: str
    day: str  # "april_6" or "april_7"
    method: str  # "rex" or "vida"
    d_mean_uas: float
    d_sigma_uas: float
    w_mean_uas: Optional[float]
    w_sigma_uas: Optional[float]
    source: Dict[str, Any]


def _clean_cell(s: str) -> str:
    return s.strip().rstrip("\\").strip()


def _parse_table(tex: str, *, source_path: Path) -> List[RingFitRow]:
    # Identify which table we are currently in.
    table = None
    pipeline = None
    cluster = None

    pipeline_map = {
        "\\difmap": "difmap",
        "\\ehtim": "ehtim",
        "\\smili": "smili",
        "\\themis": "themis",
    }

    rows: List[RingFitRow] = []
    for lineno, raw in enumerate(tex.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue

        if "\\label{tab:ring_fulfits_descattered}" in line:
            table = "descattered"
            pipeline = None
            cluster = None
            continue
        if "\\label{tab:ring_fulfits_on-sky}" in line:
            table = "on_sky"
            pipeline = None
            cluster = None
            continue

        for token, name in pipeline_map.items():
            if line.startswith(token):
                pipeline = name
                continue

        if table is None or pipeline is None:
            continue

        if "&" not in line:
            continue

        # Skip header lines with units.
        if "\\rm{(\\mu as)}" in line or "\\hline" in line:
            continue

        parts = [_clean_cell(p) for p in line.split("&")]
        if len(parts) < 7:
            continue

        p0, p1, p2, p3, p4, p5, p6 = parts[:7]
        if p0:
            cluster = p0
        if not cluster:
            continue

        fit_method = p1
        d = _parse_pm(p2)
        if d is None:
            continue
        d_mean, d_sigma = d

        w = _parse_pm(p3)
        eta = _parse_pm(p4)
        a = _parse_pm(p5)
        fc = _parse_pm(p6)

        row = RingFitRow(
            table=table,
            pipeline=pipeline,
            cluster=cluster,
            fit_method=fit_method,
            d_mean_uas=float(d_mean),
            d_sigma_uas=float(d_sigma),
            w_mean_uas=None if w is None else float(w[0]),
            w_sigma_uas=None if w is None else float(w[1]),
            eta_mean_deg=None if eta is None else float(eta[0]),
            eta_sigma_deg=None if eta is None else float(eta[1]),
            a_mean=None if a is None else float(a[0]),
            a_sigma=None if a is None else float(a[1]),
            fc_mean=None if fc is None else float(fc[0]),
            fc_sigma=None if fc is None else float(fc[1]),
            source={"path": str(source_path), "line": int(lineno), "note": "appendix_ringfits.tex"},
        )
        rows.append(row)
    return rows


def _parse_image_analysis_ringfit_summary_table(tex: str, *, source_path: Path) -> List[RingFitSummaryRow]:
    # Parse Table "tab:SgrA_ringfit" in Paper III (image_analysis.tex).
    label = "\\label{tab:SgrA_ringfit}"
    lines = tex.splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if label in line:
            start_idx = i
            break
    if start_idx is None:
        return []

    end_idx = None
    for j in range(start_idx, len(lines)):
        if lines[j].strip().startswith("\\end{table*}"):
            end_idx = j
            break

    block_lines = lines[start_idx : (end_idx if end_idx is not None else len(lines))]
    block_start_lineno = start_idx + 1

    pipeline_map = {
        "\\difmap": "difmap",
        "\\ehtim": "ehtim",
        "\\smili": "smili",
        "\\themis": "themis",
    }

    table: Optional[str] = None
    pipeline: Optional[str] = None
    day: Optional[str] = None
    method_index = 0

    rows: List[RingFitSummaryRow] = []
    for off, raw in enumerate(block_lines):
        lineno = block_start_lineno + off
        line = raw.strip()
        if not line:
            continue

        if line.startswith("Descattered"):
            table = "descattered"
            pipeline = None
            day = None
            method_index = 0
            continue
        if line.startswith("On-sky"):
            table = "on_sky"
            pipeline = None
            day = None
            method_index = 0
            continue

        for token, name in pipeline_map.items():
            if line.startswith(token):
                pipeline = name
                day = None
                method_index = 0
                continue

        if table is None or pipeline is None:
            continue
        if "&" not in line:
            continue
        if "\\rm{(\\mu as)}" in line or "\\hline" in line:
            continue

        parts = [_clean_cell(p) for p in line.split("&")]
        if len(parts) < 4:
            continue
        p0, p1, p2, p3 = parts[:4]

        if p0:
            if "April 6" in p0:
                day = "april_6"
            elif "April 7" in p0:
                day = "april_7"
            else:
                day = None
            method_index = 0

        if day is None:
            continue

        d = _parse_pm(p2)
        if d is None:
            continue
        d_mean, d_sigma = d
        w = _parse_pm(p3)

        if "REx" in p1:
            method = "rex"
        elif "\\vida" in p1 or "vida" in p1:
            method = "vida"
        else:
            method = "rex" if method_index == 0 else "vida" if method_index == 1 else f"m{method_index + 1}"
        method_index += 1

        rows.append(
            RingFitSummaryRow(
                table=table,
                pipeline=pipeline,
                day=day,
                method=method,
                d_mean_uas=float(d_mean),
                d_sigma_uas=float(d_sigma),
                w_mean_uas=None if w is None else float(w[0]),
                w_sigma_uas=None if w is None else float(w[1]),
                source={"path": str(source_path), "line": int(lineno), "note": "image_analysis.tex:tab:SgrA_ringfit"},
            )
        )
    return rows


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    x = np.array(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size >= 2 else 0.0,
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _make_plot(*, metrics: Dict[str, Any], out_png: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"matplotlib not available: {e}") from e

    _set_japanese_font()
    pipelines = [p for p in ("difmap", "ehtim", "smili", "themis") if p in (metrics.get("by_pipeline") or {})]
    if not pipelines:
        raise RuntimeError("no pipelines to plot")

    means = []
    stds = []
    counts = []
    for p in pipelines:
        s = (metrics["by_pipeline"][p] or {}).get("d_mean_uas_summary") or {}
        means.append(float(s.get("mean")))
        stds.append(float(s.get("std")))
        counts.append(int(s.get("n") or 0))

    x = np.arange(len(pipelines), dtype=float)
    fig = plt.figure(figsize=(9.2, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x, means, color="#1f77b4", alpha=0.75)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", elinewidth=1.2, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n(n={n})" for p, n in zip(pipelines, counts, strict=False)])
    ax.set_ylabel("d mean (μas)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()
    default_tex = root / "data" / "eht" / "sources" / "arxiv_2311.09479" / "appendix_ringfits.tex"
    default_image_analysis_tex = root / "data" / "eht" / "sources" / "arxiv_2311.09479" / "image_analysis.tex"
    default_outdir = root / "output" / "private" / "eht"

    ap = argparse.ArgumentParser(description="Parse Sgr A* Paper III ring fitting table and summarize diameter scatter.")
    ap.add_argument("--tex", type=str, default=str(default_tex), help="Input TeX (default: Paper III appendix_ringfits.tex).")
    ap.add_argument(
        "--image-analysis-tex",
        type=str,
        default=str(default_image_analysis_tex),
        help="Input TeX (default: Paper III image_analysis.tex; Table tab:SgrA_ringfit).",
    )
    ap.add_argument(
        "--outdir", type=str, default=str(default_outdir), help="Output directory (default: output/private/eht)."
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    tex_path = Path(args.tex)
    image_analysis_path = Path(args.image_analysis_tex)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_json = outdir / "eht_sgra_ringfit_table_metrics.json"
    out_png_desc = outdir / "eht_sgra_ringfit_table_diameter_by_pipeline_descattered.png"
    out_png_onsky = outdir / "eht_sgra_ringfit_table_diameter_by_pipeline_on_sky.png"

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"tex": str(tex_path), "image_analysis_tex": str(image_analysis_path)},
        "ok": True,
        "metrics": {},
        "metrics_image_analysis_ringfit": {},
        "outputs": {
            "json": str(out_json),
            "png_descattered": str(out_png_desc),
            "png_on_sky": str(out_png_onsky),
        },
    }

    if not tex_path.exists():
        payload["ok"] = False
        payload["reason"] = "missing_input_tex"
        _write_json(out_json, payload)
        print(f"[warn] missing input; wrote: {out_json}")
        return 0

    tex = _read_text(tex_path)
    rows = _parse_table(tex, source_path=tex_path)
    payload["rows_n"] = int(len(rows))

    by_table: Dict[str, List[RingFitRow]] = {"descattered": [], "on_sky": []}
    for r in rows:
        by_table.setdefault(r.table, []).append(r)

    metrics: Dict[str, Any] = {}
    for tab, tab_rows in by_table.items():
        if not tab_rows:
            continue

        by_pipeline: Dict[str, Any] = {}
        for p in sorted({r.pipeline for r in tab_rows}):
            rp = [r for r in tab_rows if r.pipeline == p]
            d_means = [r.d_mean_uas for r in rp]
            d_sigmas = [r.d_sigma_uas for r in rp]
            by_pipeline[p] = {
                "d_mean_uas_summary": _summary(d_means),
                "d_sigma_uas_summary": _summary(d_sigmas),
            }

        d_means_all = [r.d_mean_uas for r in tab_rows]
        d_sigmas_all = [r.d_sigma_uas for r in tab_rows]
        all_sum = _summary(d_means_all)
        rel_std = float(all_sum.get("std") or 0.0) / float(all_sum.get("mean") or 1.0) if all_sum.get("n") else None
        metrics[tab] = {
            "table": tab,
            "rows_n": int(len(tab_rows)),
            "d_mean_uas_summary_all": all_sum,
            "d_sigma_uas_summary_all": _summary(d_sigmas_all),
            "d_mean_rel_std_all": rel_std,
            "by_pipeline": by_pipeline,
        }

        try:
            if tab == "descattered":
                _make_plot(metrics=metrics[tab], out_png=out_png_desc, title="Sgr A* ring fits (descattered): diameter by pipeline")
            elif tab == "on_sky":
                _make_plot(metrics=metrics[tab], out_png=out_png_onsky, title="Sgr A* ring fits (on-sky): diameter by pipeline")
        except Exception as e:
            metrics[tab]["plot_error"] = str(e)

    payload["metrics"] = metrics

    if image_analysis_path.exists():
        ia_tex = _read_text(image_analysis_path)
        ia_rows = _parse_image_analysis_ringfit_summary_table(ia_tex, source_path=image_analysis_path)
        payload["image_analysis_rows_n"] = int(len(ia_rows))

        ia_by_table: Dict[str, List[RingFitSummaryRow]] = {"descattered": [], "on_sky": []}
        for r in ia_rows:
            ia_by_table.setdefault(r.table, []).append(r)

        ia_metrics: Dict[str, Any] = {}
        for tab, tab_rows in ia_by_table.items():
            if not tab_rows:
                continue

            by_pipeline: Dict[str, Any] = {}
            for p in sorted({r.pipeline for r in tab_rows}):
                rp = [r for r in tab_rows if r.pipeline == p]
                d_means = [r.d_mean_uas for r in rp]
                d_sigmas = [r.d_sigma_uas for r in rp]
                by_pipeline[p] = {
                    "d_mean_uas_summary": _summary(d_means),
                    "d_sigma_uas_summary": _summary(d_sigmas),
                }

            d_means_all = [r.d_mean_uas for r in tab_rows]
            d_sigmas_all = [r.d_sigma_uas for r in tab_rows]
            all_sum = _summary(d_means_all)
            rel_std = float(all_sum.get("std") or 0.0) / float(all_sum.get("mean") or 1.0) if all_sum.get("n") else None
            ia_metrics[tab] = {
                "table": tab,
                "rows_n": int(len(tab_rows)),
                "d_mean_uas_summary_all": all_sum,
                "d_sigma_uas_summary_all": _summary(d_sigmas_all),
                "d_mean_rel_std_all": rel_std,
                "by_pipeline": by_pipeline,
                "source": {"path": str(image_analysis_path), "label": "tab:SgrA_ringfit"},
            }

        payload["metrics_image_analysis_ringfit"] = ia_metrics
    else:
        payload["metrics_image_analysis_ringfit"] = {"ok": False, "reason": "missing_image_analysis_tex"}

    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "eht_sgra_ringfit_table_metrics",
                "outputs": [
                    str(out_json.relative_to(root)).replace("\\", "/"),
                    str(out_png_desc.relative_to(root)).replace("\\", "/"),
                    str(out_png_onsky.relative_to(root)).replace("\\", "/"),
                ],
                "metrics": {"rows_n": int(len(rows)), "image_analysis_rows_n": int(payload.get("image_analysis_rows_n") or 0)},
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
