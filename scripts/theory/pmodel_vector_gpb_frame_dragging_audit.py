from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog


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
        return


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _fmt_float(value: float, digits: int = 6) -> str:
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1e4 or abs_value < 1e-3:
        return f"{value:.{digits}g}"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _extract_channels(gpb_payload: Dict[str, Any], frame_payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    geodetic_row: Dict[str, Any] = {}
    frame_rows: List[Dict[str, Any]] = []

    for channel in gpb_payload.get("channels") or []:
        channel_id = str(channel.get("id") or "")
        if channel_id == "geodetic_precession":
            geodetic_row = {
                "id": "gpb_geodetic_control",
                "label": "GP-B geodetic (control)",
                "experiment": "GP-B",
                "observable": "geodetic_precession",
                "unit": str(channel.get("unit") or "arcsec/yr"),
                "observed": _to_float(channel.get("observed")),
                "observed_sigma": _to_float(channel.get("observed_sigma")),
                "reference_prediction": _to_float(channel.get("reference_prediction")),
                "scalar_prediction": _to_float(channel.get("pmodel_scalar_prediction")),
                "note": str(channel.get("note") or ""),
            }
        elif channel_id == "frame_dragging":
            reference_prediction = _to_float(channel.get("reference_prediction"))
            if reference_prediction is None:
                reference_prediction = _to_float(channel.get("observed"))
            frame_rows.append(
                {
                    "id": "gpb_frame_dragging",
                    "label": "GP-B frame-dragging",
                    "experiment": "GP-B",
                    "observable": "frame_dragging",
                    "unit": str(channel.get("unit") or "arcsec/yr"),
                    "value_domain": "absolute",
                    "observed": _to_float(channel.get("observed")),
                    "observed_sigma": _to_float(channel.get("observed_sigma")),
                    "reference_prediction": reference_prediction,
                    "note": str(channel.get("note") or ""),
                }
            )

    for experiment in frame_payload.get("experiments") or []:
        experiment_id = str(experiment.get("id") or "").lower()
        if "lageos" not in experiment_id:
            continue
        observed_mu = _to_float(experiment.get("mu"))
        observed_mu_sigma = _to_float(experiment.get("mu_sigma"))
        if observed_mu is None:
            observed_rate = _to_float(experiment.get("omega_obs_mas_per_yr"))
            predicted_rate = _to_float(experiment.get("omega_pred_mas_per_yr"))
            if observed_rate is not None and predicted_rate is not None and abs(predicted_rate) > 0.0:
                observed_mu = abs(observed_rate) / abs(predicted_rate)
                observed_rate_sigma = _to_float(experiment.get("omega_obs_sigma_mas_per_yr"))
                if observed_rate_sigma is not None:
                    observed_mu_sigma = abs(observed_rate_sigma) / abs(predicted_rate)
        frame_rows.append(
            {
                "id": "lageos_frame_dragging",
                "label": "LAGEOS frame-dragging",
                "experiment": "LAGEOS",
                "observable": "frame_dragging",
                "unit": "mu_ratio",
                "value_domain": "ratio_to_gr",
                "observed": observed_mu,
                "observed_sigma": observed_mu_sigma,
                "reference_prediction": 1.0,
                "note": str(experiment.get("sigma_note") or ""),
            }
        )
        break

    if not frame_rows:
        raise SystemExit("no frame-dragging channels available")
    return geodetic_row, frame_rows


def _fit_kappa_rot(frame_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    numerator = 0.0
    denominator = 0.0
    used_channels = 0
    for row in frame_rows:
        observed = _to_float(row.get("observed"))
        sigma = _to_float(row.get("observed_sigma"))
        reference_prediction = _to_float(row.get("reference_prediction"))
        if observed is None or sigma is None or sigma <= 0.0 or reference_prediction is None:
            continue
        weight = 1.0 / (sigma * sigma)
        numerator += observed * reference_prediction * weight
        denominator += reference_prediction * reference_prediction * weight
        used_channels += 1

    if used_channels == 0 or denominator <= 0.0:
        return {
            "status": "inconclusive",
            "fit_method": "weighted_least_squares",
            "fit_channels_n": used_channels,
            "kappa_rot": None,
            "kappa_rot_sigma": None,
        }

    kappa_rot = numerator / denominator
    kappa_rot_sigma = math.sqrt(1.0 / denominator)
    return {
        "status": "ok",
        "fit_method": "weighted_least_squares",
        "fit_channels_n": used_channels,
        "kappa_rot": float(kappa_rot),
        "kappa_rot_sigma": float(kappa_rot_sigma),
    }


def _score_status(z_score: Optional[float], z_reject: float) -> str:
    if z_score is None:
        return "inconclusive"
    return "reject" if abs(z_score) > z_reject else "pass"


def _build_rows(
    *,
    branch: str,
    geodetic_row: Dict[str, Any],
    frame_rows: Sequence[Dict[str, Any]],
    kappa_rot: Optional[float],
    z_reject: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if geodetic_row:
        observed = _to_float(geodetic_row.get("observed"))
        observed_sigma = _to_float(geodetic_row.get("observed_sigma"))
        scalar_prediction = _to_float(geodetic_row.get("scalar_prediction"))
        residual = None if observed is None or scalar_prediction is None else observed - scalar_prediction
        z_score = None if residual is None or observed_sigma is None or observed_sigma <= 0.0 else residual / observed_sigma
        rows.append(
            {
                "branch": branch,
                "id": str(geodetic_row.get("id") or "gpb_geodetic_control"),
                "label": str(geodetic_row.get("label") or "GP-B geodetic (control)"),
                "kind": "control",
                "experiment": str(geodetic_row.get("experiment") or "GP-B"),
                "observable": "geodetic_precession",
                "unit": str(geodetic_row.get("unit") or "arcsec/yr"),
                "value_domain": "absolute",
                "observed": observed,
                "observed_sigma": observed_sigma,
                "reference_prediction": _to_float(geodetic_row.get("reference_prediction")),
                "pmodel_prediction": scalar_prediction,
                "residual": residual,
                "z_score": z_score,
                "status": _score_status(z_score, z_reject=z_reject),
                "note": str(geodetic_row.get("note") or ""),
            }
        )

    for row in frame_rows:
        observed = _to_float(row.get("observed"))
        observed_sigma = _to_float(row.get("observed_sigma"))
        reference_prediction = _to_float(row.get("reference_prediction"))
        if branch == "scalar_static":
            prediction = 0.0
        else:
            prediction = None if kappa_rot is None or reference_prediction is None else kappa_rot * reference_prediction
        residual = None if observed is None or prediction is None else observed - prediction
        z_score = None if residual is None or observed_sigma is None or observed_sigma <= 0.0 else residual / observed_sigma
        rows.append(
            {
                "branch": branch,
                "id": str(row.get("id") or ""),
                "label": str(row.get("label") or ""),
                "kind": "frame_dragging",
                "experiment": str(row.get("experiment") or ""),
                "observable": "frame_dragging",
                "unit": str(row.get("unit") or ""),
                "value_domain": str(row.get("value_domain") or ""),
                "observed": observed,
                "observed_sigma": observed_sigma,
                "reference_prediction": reference_prediction,
                "pmodel_prediction": prediction,
                "residual": residual,
                "z_score": z_score,
                "status": _score_status(z_score, z_reject=z_reject),
                "note": str(row.get("note") or ""),
            }
        )

    return rows


def _branch_summary(rows: Sequence[Dict[str, Any]], z_reject: float) -> Dict[str, Any]:
    frame_rows = [row for row in rows if str(row.get("kind") or "") == "frame_dragging"]
    pass_count = sum(1 for row in frame_rows if row.get("status") == "pass")
    reject_count = sum(1 for row in frame_rows if row.get("status") == "reject")
    inconclusive_count = sum(1 for row in frame_rows if row.get("status") == "inconclusive")
    max_abs_z = max((abs(float(row["z_score"])) for row in frame_rows if row.get("z_score") is not None), default=None)

    branch_status = "inconclusive"
    if reject_count > 0:
        branch_status = "reject"
    elif frame_rows and pass_count == len(frame_rows):
        branch_status = "pass"
    elif pass_count > 0:
        branch_status = "watch"

    return {
        "status": branch_status,
        "z_reject": z_reject,
        "frame_channels_n": len(frame_rows),
        "pass_n": pass_count,
        "reject_n": reject_count,
        "inconclusive_n": inconclusive_count,
        "max_abs_z": max_abs_z,
    }


def _build_gates(
    *,
    scalar_summary: Dict[str, Any],
    vector_summary: Dict[str, Any],
    vector_rows: Sequence[Dict[str, Any]],
    kappa_fit: Dict[str, Any],
    z_reject: float,
) -> Dict[str, Any]:
    geodetic_row = next((row for row in vector_rows if str(row.get("id") or "") == "gpb_geodetic_control"), None)
    geodetic_abs_z = abs(float(geodetic_row["z_score"])) if geodetic_row and geodetic_row.get("z_score") is not None else None

    gate_static_reject = {
        "id": "vector_gpb::scalar_branch_rejected",
        "metric": "scalar_frame_reject_n",
        "value": int(scalar_summary.get("reject_n") or 0),
        "threshold": 1,
        "comparator": ">=",
        "status": "pass" if int(scalar_summary.get("reject_n") or 0) >= 1 else "watch",
        "hardness": "watch",
        "note": "拡張必要性の監査。静的スカラー枝が frame-dragging を再現できず reject になること。",
    }
    gate_vector_pass = {
        "id": "vector_gpb::vector_branch_frame_gate",
        "metric": "vector_max_abs_z_frame",
        "value": float(vector_summary.get("max_abs_z")) if vector_summary.get("max_abs_z") is not None else None,
        "threshold": float(z_reject),
        "comparator": "<=",
        "status": "pass"
        if vector_summary.get("max_abs_z") is not None and float(vector_summary.get("max_abs_z")) <= float(z_reject)
        else "reject",
        "hardness": "hard",
        "note": "回転カレント起因の P_i 渦（kappa_rot）導入後、GP-B/LAGEOS の frame-dragging が 3σ 内に入ること。",
    }
    gate_geodetic = {
        "id": "vector_gpb::geodetic_control_gate",
        "metric": "abs(z_geodetic_control)",
        "value": geodetic_abs_z,
        "threshold": float(z_reject),
        "comparator": "<=",
        "status": "pass" if geodetic_abs_z is not None and geodetic_abs_z <= float(z_reject) else "reject",
        "hardness": "hard",
        "note": "拡張で geodetic control が崩れていないこと。",
    }
    kappa_sigma = _to_float(kappa_fit.get("kappa_rot_sigma"))
    gate_kappa_detect = {
        "id": "vector_gpb::kappa_rot_nonzero_watch",
        "metric": "abs(kappa_rot)/sigma",
        "value": None,
        "threshold": 1.0,
        "comparator": ">=",
        "status": "watch",
        "hardness": "watch",
        "note": "最小拡張係数 kappa_rot の識別性を監視する（補助）。",
    }
    kappa_value = _to_float(kappa_fit.get("kappa_rot"))
    if kappa_value is not None and kappa_sigma is not None and kappa_sigma > 0.0:
        kappa_significance = abs(kappa_value) / kappa_sigma
        gate_kappa_detect["value"] = float(kappa_significance)
        gate_kappa_detect["status"] = "pass" if kappa_significance >= 1.0 else "watch"

    gates = [gate_static_reject, gate_vector_pass, gate_geodetic, gate_kappa_detect]
    hard_reject_count = sum(1 for gate in gates if gate["hardness"] == "hard" and gate["status"] == "reject")
    watch_count = sum(1 for gate in gates if gate["status"] == "watch")
    overall_status = "pass" if hard_reject_count == 0 else "reject"
    decision = "vector_branch_required_and_consistent" if overall_status == "pass" else "vector_branch_not_validated"

    return {
        "overall_status": overall_status,
        "decision": decision,
        "hard_reject_n": hard_reject_count,
        "watch_n": watch_count,
        "gates": gates,
    }


def _plot(
    scalar_rows: Sequence[Dict[str, Any]],
    vector_rows: Sequence[Dict[str, Any]],
    kappa_fit: Dict[str, Any],
    z_reject: float,
    out_png: Path,
) -> None:
    _set_japanese_font()

    frame_scalar = [row for row in scalar_rows if str(row.get("kind") or "") == "frame_dragging"]
    frame_vector = [row for row in vector_rows if str(row.get("kind") or "") == "frame_dragging"]
    labels = [str(row.get("label") or row.get("id") or "") for row in frame_scalar]
    index = np.arange(len(labels), dtype=float)

    observed_ratios: List[float] = []
    scalar_ratios: List[float] = []
    vector_ratios: List[float] = []
    scalar_z: List[float] = []
    vector_z: List[float] = []

    for scalar_row, vector_row in zip(frame_scalar, frame_vector):
        observed = _to_float(scalar_row.get("observed"))
        reference_prediction = _to_float(scalar_row.get("reference_prediction"))
        scalar_prediction = _to_float(scalar_row.get("pmodel_prediction"))
        vector_prediction = _to_float(vector_row.get("pmodel_prediction"))
        observed_ratios.append(
            float("nan") if observed is None or reference_prediction is None or abs(reference_prediction) == 0.0 else observed / reference_prediction
        )
        scalar_ratios.append(
            float("nan")
            if scalar_prediction is None or reference_prediction is None or abs(reference_prediction) == 0.0
            else scalar_prediction / reference_prediction
        )
        vector_ratios.append(
            float("nan")
            if vector_prediction is None or reference_prediction is None or abs(reference_prediction) == 0.0
            else vector_prediction / reference_prediction
        )
        scalar_z.append(float(scalar_row.get("z_score")) if scalar_row.get("z_score") is not None else float("nan"))
        vector_z.append(float(vector_row.get("z_score")) if vector_row.get("z_score") is not None else float("nan"))

    figure, (axis0, axis1, axis2) = plt.subplots(1, 3, figsize=(15.0, 5.1))
    figure.suptitle("P_μ minimal extension: GP-B/LAGEOS frame-dragging audit")

    bar_width = 0.25
    axis0.bar(index - bar_width, observed_ratios, width=bar_width, color="#1f77b4", label="obs/ref")
    axis0.bar(index, scalar_ratios, width=bar_width, color="#d62728", alpha=0.9, label="scalar")
    axis0.bar(index + bar_width, vector_ratios, width=bar_width, color="#2ca02c", alpha=0.9, label="vector")
    axis0.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0)
    axis0.set_xticks(index)
    axis0.set_xticklabels(labels, rotation=16, ha="right")
    axis0.set_ylabel("ratio to reference")
    axis0.set_title("Frame-dragging ratio")
    axis0.grid(True, axis="y", alpha=0.25)
    axis0.legend(loc="best")

    axis1.bar(index - 0.18, scalar_z, width=0.36, color="#d62728", alpha=0.9, label="scalar z")
    axis1.bar(index + 0.18, vector_z, width=0.36, color="#2ca02c", alpha=0.9, label="vector z")
    axis1.axhline(z_reject, color="#333333", linestyle="--", linewidth=1.0)
    axis1.axhline(-z_reject, color="#333333", linestyle="--", linewidth=1.0)
    axis1.axhline(0.0, color="#666666", linestyle="-", linewidth=0.9)
    axis1.set_xticks(index)
    axis1.set_xticklabels(labels, rotation=16, ha="right")
    axis1.set_ylabel("z = (obs - pred)/sigma")
    axis1.set_title("z-score gate")
    axis1.grid(True, axis="y", alpha=0.25)
    axis1.legend(loc="best")

    kappa_value = _to_float(kappa_fit.get("kappa_rot"))
    kappa_sigma = _to_float(kappa_fit.get("kappa_rot_sigma"))
    axis2.axhline(0.0, color="#666666", linewidth=0.9)
    if kappa_value is not None:
        axis2.bar(["kappa_rot"], [kappa_value], color="#9467bd", alpha=0.9)
        if kappa_sigma is not None and kappa_sigma > 0.0:
            axis2.errorbar(["kappa_rot"], [kappa_value], yerr=[kappa_sigma], fmt="none", ecolor="#222222", capsize=6, linewidth=1.2)
    axis2.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0, label="GR-like scale")
    axis2.set_title("Fitted rotational coupling")
    axis2.set_ylabel("kappa_rot")
    axis2.grid(True, axis="y", alpha=0.25)
    axis2.legend(loc="best")

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_png, dpi=200)
    plt.close(figure)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "branch",
                "id",
                "label",
                "kind",
                "experiment",
                "observable",
                "unit",
                "value_domain",
                "observed",
                "observed_sigma",
                "reference_prediction",
                "pmodel_prediction",
                "residual",
                "z_score",
                "status",
                "note",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.get("branch", ""),
                    row.get("id", ""),
                    row.get("label", ""),
                    row.get("kind", ""),
                    row.get("experiment", ""),
                    row.get("observable", ""),
                    row.get("unit", ""),
                    row.get("value_domain", ""),
                    "" if row.get("observed") is None else _fmt_float(float(row["observed"]), 8),
                    "" if row.get("observed_sigma") is None else _fmt_float(float(row["observed_sigma"]), 8),
                    "" if row.get("reference_prediction") is None else _fmt_float(float(row["reference_prediction"]), 8),
                    "" if row.get("pmodel_prediction") is None else _fmt_float(float(row["pmodel_prediction"]), 8),
                    "" if row.get("residual") is None else _fmt_float(float(row["residual"]), 8),
                    "" if row.get("z_score") is None else _fmt_float(float(row["z_score"]), 8),
                    row.get("status", ""),
                    row.get("note", ""),
                ]
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    default_gpb_data = _ROOT / "data" / "theory" / "gpb_scalar_limit_audit.json"
    default_frame_data = _ROOT / "data" / "theory" / "frame_dragging_experiments.json"
    default_outdir = _ROOT / "output" / "private" / "theory"
    default_public_outdir = _ROOT / "output" / "public" / "theory"

    parser = argparse.ArgumentParser(description="P_mu vector GP-B frame-dragging audit.")
    parser.add_argument("--gpb-data", type=str, default=str(default_gpb_data), help="Input JSON path for GP-B channels.")
    parser.add_argument("--frame-data", type=str, default=str(default_frame_data), help="Input JSON path for frame-dragging experiments.")
    parser.add_argument("--outdir", type=str, default=str(default_outdir), help="Output directory.")
    parser.add_argument("--public-outdir", type=str, default=str(default_public_outdir), help="Public output directory.")
    parser.add_argument("--z-reject", type=float, default=3.0, help="Reject gate for |z|.")
    parser.add_argument("--no-public-copy", action="store_true", help="Do not copy outputs to public directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    gpb_payload = _read_json(Path(args.gpb_data))
    frame_payload = _read_json(Path(args.frame_data))
    geodetic_row, frame_rows = _extract_channels(gpb_payload, frame_payload)
    kappa_fit = _fit_kappa_rot(frame_rows)
    kappa_rot = _to_float(kappa_fit.get("kappa_rot"))

    scalar_rows = _build_rows(
        branch="scalar_static",
        geodetic_row=geodetic_row,
        frame_rows=frame_rows,
        kappa_rot=0.0,
        z_reject=float(args.z_reject),
    )
    vector_rows = _build_rows(
        branch="vector_vorticity",
        geodetic_row=geodetic_row,
        frame_rows=frame_rows,
        kappa_rot=kappa_rot,
        z_reject=float(args.z_reject),
    )
    scalar_summary = _branch_summary(scalar_rows, z_reject=float(args.z_reject))
    vector_summary = _branch_summary(vector_rows, z_reject=float(args.z_reject))
    gate_summary = _build_gates(
        scalar_summary=scalar_summary,
        vector_summary=vector_summary,
        vector_rows=vector_rows,
        kappa_fit=kappa_fit,
        z_reject=float(args.z_reject),
    )

    outdir = Path(args.outdir)
    public_outdir = Path(args.public_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    public_outdir.mkdir(parents=True, exist_ok=True)

    out_json = outdir / "pmodel_vector_gpb_frame_dragging_audit.json"
    out_csv = outdir / "pmodel_vector_gpb_frame_dragging_audit.csv"
    out_png = outdir / "pmodel_vector_gpb_frame_dragging_audit.png"

    all_rows = scalar_rows + vector_rows
    _plot(scalar_rows, vector_rows, kappa_fit, z_reject=float(args.z_reject), out_png=out_png)
    _write_csv(out_csv, all_rows)

    payload_out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "schema": "wavep.theory.pmodel_vector_gpb_frame_dragging_audit.v1",
        "title": "P_mu minimal extension: GP-B / LAGEOS frame-dragging audit",
        "intent": (
            "Inject minimal rotational current coupling into P_mu and test whether frame-dragging observables "
            "are recovered without breaking geodetic control."
        ),
        "equations": {
            "interaction": "L_int = g * P_mu * J^mu",
            "rotational_ansatz": "P_i ~ kappa_rot * epsilon_ijk * J^j * x^k / r^3",
            "precession_mapping": "Omega_P = kappa_rot * Omega_LT_ref",
            "scalar_limit": "J^i=0 => P_i=0 (reduction to scalar branch)",
        },
        "gate": {"z_reject": float(args.z_reject)},
        "inputs": {
            "gpb_data": str(Path(args.gpb_data)).replace("\\", "/"),
            "frame_data": str(Path(args.frame_data)).replace("\\", "/"),
        },
        "fit": kappa_fit,
        "branches": {
            "scalar_static": {"summary": scalar_summary, "rows": scalar_rows},
            "vector_vorticity": {"summary": vector_summary, "rows": vector_rows},
        },
        "summary": gate_summary,
        "outputs": {
            "rows_json": str(out_json).replace("\\", "/"),
            "rows_csv": str(out_csv).replace("\\", "/"),
            "plot_png": str(out_png).replace("\\", "/"),
        },
    }
    _write_json(out_json, payload_out)

    public_copies: List[Path] = []
    if not args.no_public_copy:
        for src in (out_json, out_csv, out_png):
            dst = public_outdir / src.name
            shutil.copy2(src, dst)
            public_copies.append(dst)

    try:
        worklog.append_event(
            {
                "event_type": "theory_pmodel_vector_gpb_frame_dragging_audit",
                "argv": sys.argv,
                "inputs": {"gpb_data": Path(args.gpb_data), "frame_data": Path(args.frame_data)},
                "outputs": {
                    "rows_json": out_json,
                    "rows_csv": out_csv,
                    "plot_png": out_png,
                    "public_copies": public_copies,
                },
                "metrics": {
                    "overall_status": gate_summary.get("overall_status"),
                    "decision": gate_summary.get("decision"),
                    "kappa_rot": kappa_fit.get("kappa_rot"),
                    "kappa_rot_sigma": kappa_fit.get("kappa_rot_sigma"),
                    "scalar_frame_status": scalar_summary.get("status"),
                    "vector_frame_status": vector_summary.get("status"),
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] json : {out_json}")
    print(f"[ok] csv  : {out_csv}")
    print(f"[ok] png  : {out_png}")
    if public_copies:
        print(f"[ok] public copies: {len(public_copies)} files -> {public_outdir}")
    print(f"[ok] overall_status={gate_summary.get('overall_status')} decision={gate_summary.get('decision')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
