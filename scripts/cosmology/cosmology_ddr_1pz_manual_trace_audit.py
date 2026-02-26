#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_ddr_1pz_manual_trace_audit.py

Step 5.3.18（DDR監査の(1+z)手計算トレース）:
SNe Ia（SALT2）と BAO の距離推定プロセスを行単位で追跡し、
どこで暗黙の膨張前提（D_A = D_M/(1+z)）が入り込むかを固定出力する。

出力（固定名）:
  - output/public/cosmology/cosmology_ddr_1pz_manual_trace_audit.json
  - output/public/cosmology/cosmology_ddr_1pz_manual_trace_audit.csv
  - output/public/cosmology/cosmology_ddr_1pz_manual_trace_audit.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402


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
        available = {font.name for font in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _build_trace_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add_row(
        *,
        pipeline_id: str,
        pipeline_label: str,
        line_no: int,
        stage: str,
        equation: str,
        assumption: str,
        has_one_plus_z: bool,
        one_plus_z_role: str,
        introduces_geometry_factor: bool,
        note: str,
    ) -> None:
        rows.append(
            {
                "pipeline_id": pipeline_id,
                "pipeline_label": pipeline_label,
                "line_no": int(line_no),
                "stage": stage,
                "equation": equation,
                "assumption": assumption,
                "has_one_plus_z": bool(has_one_plus_z),
                "one_plus_z_role": one_plus_z_role,
                "introduces_geometry_factor": bool(introduces_geometry_factor),
                "note": note,
            }
        )

    # SNe Ia (SALT2) pipeline
    add_row(
        pipeline_id="snia_salt2",
        pipeline_label="SNe Ia (SALT2): photometry -> D_L",
        line_no=1,
        stage="fit observables",
        equation="(m_B^*, x_1, c) <- SALT2 light-curve fit",
        assumption="Empirical template fit for apparent magnitude, stretch, color.",
        has_one_plus_z=False,
        one_plus_z_role="none",
        introduces_geometry_factor=False,
        note="Distance not computed yet.",
    )
    add_row(
        pipeline_id="snia_salt2",
        pipeline_label="SNe Ia (SALT2): photometry -> D_L",
        line_no=2,
        stage="standardization",
        equation="mu_SN = m_B^* - M_B + alpha*x_1 - beta*c + Delta_M + Delta_B",
        assumption="Standardizable-candle relation with nuisance calibration terms.",
        has_one_plus_z=False,
        one_plus_z_role="none",
        introduces_geometry_factor=False,
        note="No D_A / D_M appears explicitly.",
    )
    add_row(
        pipeline_id="snia_salt2",
        pipeline_label="SNe Ia (SALT2): photometry -> D_L",
        line_no=3,
        stage="distance conversion",
        equation="D_L[Mpc] = 10^((mu_SN - 25)/5)",
        assumption="Definition of distance modulus.",
        has_one_plus_z=False,
        one_plus_z_role="none",
        introduces_geometry_factor=False,
        note="Pure luminosity-distance conversion.",
    )
    add_row(
        pipeline_id="snia_salt2",
        pipeline_label="SNe Ia (SALT2): photometry -> D_L",
        line_no=4,
        stage="rest-frame remapping",
        equation="t_rest = t_obs/(1+z)",
        assumption="Rest-frame light-curve phase uses redshift time-dilation mapping.",
        has_one_plus_z=True,
        one_plus_z_role="timing_restframe",
        introduces_geometry_factor=False,
        note="(1+z) enters as timing/flux-side mapping, not D_A geometry conversion.",
    )

    # BAO pipeline
    add_row(
        pipeline_id="bao_anisotropic",
        pipeline_label="BAO anisotropic: angular/redshift scales -> D_M, D_H",
        line_no=1,
        stage="anisotropic fit",
        equation="alpha_perp = (D_M/r_d)/(D_M_fid/r_d_fid), alpha_par = (D_H/r_d)/(D_H_fid/r_d_fid)",
        assumption="Compressed BAO constraints are defined relative to a fiducial cosmology.",
        has_one_plus_z=False,
        one_plus_z_role="none",
        introduces_geometry_factor=False,
        note="Primary compressed observables can be written in D_M/r_d and D_H/r_d.",
    )
    add_row(
        pipeline_id="bao_anisotropic",
        pipeline_label="BAO anisotropic: angular/redshift scales -> D_M, D_H",
        line_no=2,
        stage="distance publication",
        equation="D_A = D_M/(1+z)",
        assumption="FRW geometric conversion from transverse comoving distance to angular-diameter distance.",
        has_one_plus_z=True,
        one_plus_z_role="geometry_DA_from_DM",
        introduces_geometry_factor=True,
        note="This is the explicit insertion point of the expansion-side (1+z) factor.",
    )
    add_row(
        pipeline_id="bao_anisotropic",
        pipeline_label="BAO anisotropic: angular/redshift scales -> D_M, D_H",
        line_no=3,
        stage="isotropic compression",
        equation="D_V = [(1+z)^2 * D_A^2 * c*z/H(z)]^(1/3) = [D_M^2 * c*z/H(z)]^(1/3)",
        assumption="Volume distance expression inherits the same D_A <-> D_M conversion.",
        has_one_plus_z=True,
        one_plus_z_role="geometry_DA_from_DM",
        introduces_geometry_factor=True,
        note="Even if D_A is eliminated algebraically, the geometric conversion is assumed.",
    )

    # DDR combined estimator path
    add_row(
        pipeline_id="ddr_combined",
        pipeline_label="DDR combined (SNe Ia + BAO): eta construction",
        line_no=1,
        stage="definition",
        equation="eta = D_L / ((1+z)^2 * D_A)",
        assumption="Standard DDR definition used in many published constraints.",
        has_one_plus_z=True,
        one_plus_z_role="definition",
        introduces_geometry_factor=False,
        note="Contains two redshift factors before pipeline substitution.",
    )
    add_row(
        pipeline_id="ddr_combined",
        pipeline_label="DDR combined (SNe Ia + BAO): eta construction",
        line_no=2,
        stage="BAO substitution",
        equation="if D_A = D_M/(1+z): eta = D_L / ((1+z)*D_M)",
        assumption="BAO-side conversion introduces one geometric (1+z) directly into eta.",
        has_one_plus_z=True,
        one_plus_z_role="geometry_DA_from_DM",
        introduces_geometry_factor=True,
        note="This is the exact place where the hidden geometry-side factor appears in combined DDR tests.",
    )
    add_row(
        pipeline_id="ddr_combined",
        pipeline_label="DDR combined (SNe Ia + BAO): eta construction",
        line_no=3,
        stage="P-model audit split",
        equation="eta^(P) = D_L / ((1+z)*D_A) (flux-side only divided out)",
        assumption="Separate flux-side redshift factor from geometry-side conversion in the audit layer.",
        has_one_plus_z=True,
        one_plus_z_role="flux_or_timing_side",
        introduces_geometry_factor=False,
        note="Used to avoid mixing physical redshift effects with D_A convention.",
    )

    return rows


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_pipeline: Dict[str, Dict[str, Any]] = {}
    geometry_points: List[Dict[str, Any]] = []

    total_rows = len(rows)
    total_with_1pz = 0
    total_geometry = 0

    for row in rows:
        pipeline_id = str(row["pipeline_id"])
        pipeline_label = str(row["pipeline_label"])
        has_one_plus_z = bool(row["has_one_plus_z"])
        geometry = bool(row["introduces_geometry_factor"])
        role = str(row["one_plus_z_role"])

        pipeline_stats = by_pipeline.setdefault(
            pipeline_id,
            {
                "pipeline_label": pipeline_label,
                "rows_total": 0,
                "rows_with_1pz": 0,
                "rows_geometry_1pz": 0,
                "role_counts": {},
            },
        )

        pipeline_stats["rows_total"] = int(pipeline_stats["rows_total"]) + 1
        if has_one_plus_z:
            total_with_1pz += 1
            pipeline_stats["rows_with_1pz"] = int(pipeline_stats["rows_with_1pz"]) + 1
            role_counts: Dict[str, int] = dict(pipeline_stats["role_counts"])
            role_counts[role] = int(role_counts.get(role, 0)) + 1
            pipeline_stats["role_counts"] = role_counts

        if geometry:
            total_geometry += 1
            pipeline_stats["rows_geometry_1pz"] = int(pipeline_stats["rows_geometry_1pz"]) + 1
            geometry_points.append(
                {
                    "pipeline_id": pipeline_id,
                    "pipeline_label": pipeline_label,
                    "line_no": int(row["line_no"]),
                    "equation": str(row["equation"]),
                }
            )

    return {
        "rows_total": int(total_rows),
        "rows_with_1pz": int(total_with_1pz),
        "rows_geometry_1pz": int(total_geometry),
        "geometry_injection_detected": bool(total_geometry > 0),
        "by_pipeline": by_pipeline,
        "geometry_injection_points": geometry_points,
        "ddr_symbolic_split": {
            "before_substitution": "eta = D_L / ((1+z)^2 * D_A)",
            "after_DA_from_DM_substitution": "eta = D_L / ((1+z) * D_M)",
            "interpretation": "One redshift factor is explicitly tied to the D_A = D_M/(1+z) conversion layer.",
        },
        "hard_gate": {
            "name": "geometry_side_1pz_injection_trace_complete",
            "condition": "at least one line with introduces_geometry_factor=true must be identified",
            "pass": bool(total_geometry > 0),
        },
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pipeline_id",
        "pipeline_label",
        "line_no",
        "stage",
        "equation",
        "assumption",
        "has_one_plus_z",
        "one_plus_z_role",
        "introduces_geometry_factor",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _plot(path: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    pipeline_ids = list(summary["by_pipeline"].keys())
    pipeline_labels = [str(summary["by_pipeline"][pipeline_id]["pipeline_label"]) for pipeline_id in pipeline_ids]

    role_order = [
        "geometry_DA_from_DM",
        "timing_restframe",
        "flux_or_timing_side",
        "definition",
    ]
    role_colors = {
        "geometry_DA_from_DM": "#dc2626",
        "timing_restframe": "#f59e0b",
        "flux_or_timing_side": "#3b82f6",
        "definition": "#6b7280",
    }

    role_matrix = np.zeros((len(role_order), len(pipeline_ids)), dtype=float)
    for pipeline_index, pipeline_id in enumerate(pipeline_ids):
        role_counts = dict(summary["by_pipeline"][pipeline_id]["role_counts"])
        for role_index, role_name in enumerate(role_order):
            role_matrix[role_index, pipeline_index] = float(role_counts.get(role_name, 0))

    figure = plt.figure(figsize=(14.5, 8.5))
    grid = figure.add_gridspec(2, 1, height_ratios=(0.58, 0.42))
    axis_top = figure.add_subplot(grid[0, 0])
    axis_bottom = figure.add_subplot(grid[1, 0])

    x_positions = np.arange(len(pipeline_ids))
    cumulative = np.zeros(len(pipeline_ids), dtype=float)
    for role_index, role_name in enumerate(role_order):
        counts = role_matrix[role_index]
        axis_top.bar(
            x_positions,
            counts,
            bottom=cumulative,
            color=role_colors[role_name],
            edgecolor="#222222",
            linewidth=0.8,
            label=role_name,
        )
        cumulative += counts

    axis_top.set_xticks(x_positions)
    axis_top.set_xticklabels(pipeline_labels, rotation=8, ha="right")
    axis_top.set_ylabel("rows with (1+z)")
    axis_top.set_title("DDR audit: where (1+z) enters in SNe Ia/BAO pipelines")
    axis_top.grid(axis="y", linestyle="--", alpha=0.35)
    axis_top.legend(loc="upper right", frameon=True)

    geometry_points = summary["geometry_injection_points"]
    if geometry_points:
        y_positions = np.arange(len(geometry_points))
        labels = [
            f"{item['pipeline_label']} / line {item['line_no']}"
            for item in geometry_points
        ]
        axis_bottom.barh(y_positions, np.ones(len(geometry_points)), color="#dc2626", edgecolor="#222222", linewidth=0.8)
        axis_bottom.set_yticks(y_positions)
        axis_bottom.set_yticklabels(labels)
        axis_bottom.set_xlim(0.0, 1.4)
        axis_bottom.set_xlabel("geometry-side injection marker (1 = detected)")
        axis_bottom.set_title("Detected insertion points for D_A = D_M/(1+z)")
        axis_bottom.grid(axis="x", linestyle="--", alpha=0.35)
        for point_index, item in enumerate(geometry_points):
            axis_bottom.text(
                1.02,
                point_index,
                str(item["equation"]),
                va="center",
                ha="left",
                fontsize=9,
            )
    else:
        axis_bottom.axis("off")
        axis_bottom.text(0.5, 0.5, "No geometry-side (1+z) insertion detected.", ha="center", va="center")

    figure.suptitle(
        "Step 5.3.18: manual line-by-line trace for implicit expansion assumption in DDR pipelines",
        y=0.98,
    )
    figure.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08, hspace=0.38)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a line-by-line DDR (1+z) trace for SNe Ia(SALT2)/BAO pipelines and detect D_A=D_M/(1+z) insertion points."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "public" / "cosmology",
        help="Output directory (default: output/public/cosmology).",
    )
    parser.add_argument(
        "--step-tag",
        type=str,
        default="5.3.18",
        help="Roadmap step tag recorded in JSON payload.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "cosmology_ddr_1pz_manual_trace_audit.json"
    out_csv = out_dir / "cosmology_ddr_1pz_manual_trace_audit.csv"
    out_png = out_dir / "cosmology_ddr_1pz_manual_trace_audit.png"

    rows = _build_trace_rows()
    summary = _summarize_rows(rows)

    payload: Dict[str, Any] = {
        "schema": "wavep.cosmology.ddr_1pz_manual_trace_audit.v1",
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 5, "step": str(args.step_tag), "name": "DDR (1+z) manual trace audit"},
        "intent": (
            "Trace SNe Ia (SALT2) and BAO distance-estimation formulas line by line and identify where "
            "the expansion-side relation D_A=D_M/(1+z) is inserted."
        ),
        "summary": summary,
        "trace_rows": rows,
    }

    _write_json(out_json, payload)
    _write_csv(out_csv, rows)
    _plot(out_png, rows, summary)

    worklog.append_event(
        {
            "kind": "cosmology_ddr_1pz_manual_trace_audit",
            "script": "scripts/cosmology/cosmology_ddr_1pz_manual_trace_audit.py",
            "step_tag": str(args.step_tag),
            "outputs": [_rel(out_json), _rel(out_csv), _rel(out_png)],
            "geometry_injection_detected": bool(summary["geometry_injection_detected"]),
            "rows_geometry_1pz": int(summary["rows_geometry_1pz"]),
        }
    )

    print(f"[ok] json: {_rel(out_json)}")
    print(f"[ok] csv : {_rel(out_csv)}")
    print(f"[ok] png : {_rel(out_png)}")
    print(
        f"[ok] geometry-side injection points: {int(summary['rows_geometry_1pz'])}"
        f" (pass={bool(summary['hard_gate']['pass'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
