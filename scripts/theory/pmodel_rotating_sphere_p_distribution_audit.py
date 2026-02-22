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
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        if not chosen:
            return
        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        return


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _read_json(path)
    except Exception:
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fmt_float(x: float, digits: int = 6) -> str:
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _to_float(v: Any) -> Optional[float]:
    try:
        val = float(v)
    except Exception:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _extract_inputs(gpb_payload: Dict[str, Any], frame_payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    geodetic_row: Dict[str, Any] = {}
    frame_rows: List[Dict[str, Any]] = []

    for ch in gpb_payload.get("channels") or []:
        cid = str(ch.get("id") or "")
        if cid == "geodetic_precession":
            geodetic_row = {
                "id": "gpb_geodetic_control",
                "label": "GP-B geodetic (control)",
                "experiment": "GP-B",
                "observable": "geodetic_precession",
                "unit": str(ch.get("unit") or "arcsec/yr"),
                "observed": _to_float(ch.get("observed")),
                "observed_sigma": _to_float(ch.get("observed_sigma")),
                "reference_prediction": _to_float(ch.get("reference_prediction")),
                "pmodel_scalar_prediction": _to_float(ch.get("pmodel_scalar_prediction")),
                "note": str(ch.get("note") or ""),
            }
        elif cid == "frame_dragging":
            observed = _to_float(ch.get("observed"))
            sigma = _to_float(ch.get("observed_sigma"))
            ref = _to_float(ch.get("reference_prediction"))
            if ref is None:
                ref = _to_float(ch.get("observed"))
            frame_rows.append(
                {
                    "id": "gpb_frame_dragging",
                    "label": "GP-B frame-dragging",
                    "experiment": "GP-B",
                    "observable": "frame_dragging",
                    "unit": str(ch.get("unit") or "arcsec/yr"),
                    "observed": observed,
                    "observed_sigma": sigma,
                    "reference_prediction": ref,
                    "value_domain": "absolute",
                    "note": str(ch.get("note") or ""),
                }
            )

    for exp in frame_payload.get("experiments") or []:
        exp_id = str(exp.get("id") or "").lower()
        if "lageos" not in exp_id:
            continue
        observed = _to_float(exp.get("mu"))
        sigma = _to_float(exp.get("mu_sigma"))
        if observed is None:
            obs_rate = _to_float(exp.get("omega_obs_mas_per_yr"))
            pred_rate = _to_float(exp.get("omega_pred_mas_per_yr"))
            if obs_rate is not None and pred_rate is not None and abs(pred_rate) > 0:
                observed = abs(obs_rate) / abs(pred_rate)
                sigma_rate = _to_float(exp.get("omega_obs_sigma_mas_per_yr"))
                if sigma_rate is not None:
                    sigma = abs(sigma_rate) / abs(pred_rate)
        frame_rows.append(
            {
                "id": "lageos_frame_dragging",
                "label": "LAGEOS frame-dragging",
                "experiment": "LAGEOS",
                "observable": "frame_dragging",
                "unit": "mu_ratio",
                "observed": observed,
                "observed_sigma": sigma,
                "reference_prediction": 1.0,
                "value_domain": "ratio_to_gr",
                "note": str(exp.get("sigma_note") or ""),
            }
        )
        break

    if not frame_rows:
        raise SystemExit("no frame-dragging channels extracted from inputs")
    return geodetic_row, frame_rows


def _fit_kappa(frame_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    num = 0.0
    den = 0.0
    used = 0
    for row in frame_rows:
        observed = _to_float(row.get("observed"))
        sigma = _to_float(row.get("observed_sigma"))
        ref = _to_float(row.get("reference_prediction"))
        if observed is None or sigma is None or sigma <= 0.0 or ref is None:
            continue
        weight = 1.0 / (sigma * sigma)
        num += observed * ref * weight
        den += (ref * ref) * weight
        used += 1

    if used == 0 or den <= 0.0:
        return {
            "kappa_rot": None,
            "kappa_sigma": None,
            "fit_channels_n": used,
            "fit_method": "weighted_least_squares",
            "status": "inconclusive",
        }

    kappa = num / den
    kappa_sigma = math.sqrt(1.0 / den)
    return {
        "kappa_rot": float(kappa),
        "kappa_sigma": float(kappa_sigma),
        "fit_channels_n": used,
        "fit_method": "weighted_least_squares",
        "status": "ok",
    }


def _score_status(z: Optional[float], z_reject: float) -> str:
    if z is None:
        return "inconclusive"
    return "reject" if abs(z) > z_reject else "pass"


def _build_branch_rows(
    branch: str,
    frame_rows: Sequence[Dict[str, Any]],
    geodetic_row: Dict[str, Any],
    kappa_rot: Optional[float],
    z_reject: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if geodetic_row:
        observed = _to_float(geodetic_row.get("observed"))
        sigma = _to_float(geodetic_row.get("observed_sigma"))
        pred = _to_float(geodetic_row.get("pmodel_scalar_prediction"))
        residual = None if observed is None or pred is None else observed - pred
        z = None if residual is None or sigma is None or sigma <= 0 else residual / sigma
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
                "observed_sigma": sigma,
                "reference_prediction": _to_float(geodetic_row.get("reference_prediction")),
                "pmodel_prediction": pred,
                "residual": residual,
                "z_score": z,
                "status": _score_status(z, z_reject),
                "note": str(geodetic_row.get("note") or ""),
            }
        )

    for row in frame_rows:
        observed = _to_float(row.get("observed"))
        sigma = _to_float(row.get("observed_sigma"))
        ref = _to_float(row.get("reference_prediction"))
        if branch == "static_iso":
            pred = 0.0
        else:
            pred = None if kappa_rot is None or ref is None else kappa_rot * ref

        residual = None if observed is None or pred is None else observed - pred
        z = None if residual is None or sigma is None or sigma <= 0 else residual / sigma
        rows.append(
            {
                "branch": branch,
                "id": str(row.get("id") or ""),
                "label": str(row.get("label") or ""),
                "kind": "frame_dragging",
                "experiment": str(row.get("experiment") or ""),
                "observable": str(row.get("observable") or "frame_dragging"),
                "unit": str(row.get("unit") or ""),
                "value_domain": str(row.get("value_domain") or ""),
                "observed": observed,
                "observed_sigma": sigma,
                "reference_prediction": ref,
                "pmodel_prediction": pred,
                "residual": residual,
                "z_score": z,
                "status": _score_status(z, z_reject),
                "note": str(row.get("note") or ""),
            }
        )
    return rows


def _branch_summary(rows: Sequence[Dict[str, Any]], z_reject: float) -> Dict[str, Any]:
    frame = [r for r in rows if str(r.get("kind") or "") == "frame_dragging"]
    pass_n = sum(1 for r in frame if r.get("status") == "pass")
    reject_n = sum(1 for r in frame if r.get("status") == "reject")
    inconclusive_n = sum(1 for r in frame if r.get("status") == "inconclusive")
    zvals = [abs(float(r["z_score"])) for r in frame if r.get("z_score") is not None]
    max_abs_z = max(zvals) if zvals else None

    status = "inconclusive"
    if reject_n > 0:
        status = "reject"
    elif pass_n == len(frame) and len(frame) > 0:
        status = "pass"
    elif pass_n > 0 and inconclusive_n > 0:
        status = "watch"

    return {
        "frame_channels_n": len(frame),
        "pass_n": pass_n,
        "reject_n": reject_n,
        "inconclusive_n": inconclusive_n,
        "max_abs_z": max_abs_z,
        "z_reject": z_reject,
        "status": status,
    }


def _extract_metric_bridge(metric_payload: Dict[str, Any]) -> Dict[str, Any]:
    case_result = metric_payload.get("case_result") or {}
    summary = case_result.get("summary") or {}
    case_comparison = metric_payload.get("case_comparison") or {}

    metric_choice = str(
        metric_payload.get("metric_choice_decision")
        or summary.get("metric_choice_decision")
        or ""
    ).strip()
    case_status = str(summary.get("overall_status") or metric_payload.get("overall_status") or "").strip()
    case_a_status = str(case_comparison.get("case_a_status") or "").strip()
    case_b_status = str(case_comparison.get("case_b_status") or "").strip()

    gate_pass = metric_choice.lower() == "effective" and case_status.lower() == "pass"
    status = "pass" if gate_pass else "watch"
    if not metric_choice:
        status = "inconclusive"

    return {
        "metric_choice_decision": metric_choice or None,
        "case_status": case_status or None,
        "case_a_status": case_a_status or None,
        "case_b_status": case_b_status or None,
        "gate_pass": gate_pass,
        "status": status,
    }


def _extract_boundary_bridge(strong_payload: Dict[str, Any], flux_closure_threshold: float) -> Dict[str, Any]:
    axisymmetric = strong_payload.get("axisymmetric_pde_block") or {}
    diagnostics = axisymmetric.get("boundary_diagnostics") or {}

    formulation_complete = bool(axisymmetric.get("formulation_complete"))
    boundary_closure_pass = bool(
        axisymmetric.get("boundary_closure_pass")
        or diagnostics.get("boundary_closure_pass")
    )
    max_flux_rel_std = _to_float(diagnostics.get("max_flux_rel_std"))
    flux_closure_pass = max_flux_rel_std is not None and max_flux_rel_std <= flux_closure_threshold
    gate_pass = formulation_complete and boundary_closure_pass and flux_closure_pass

    status = "pass" if gate_pass else "watch"
    if not axisymmetric:
        status = "inconclusive"

    return {
        "formulation_complete": formulation_complete,
        "boundary_closure_pass": boundary_closure_pass,
        "max_flux_rel_std": max_flux_rel_std,
        "flux_closure_threshold": float(flux_closure_threshold),
        "flux_closure_pass": flux_closure_pass,
        "gate_pass": gate_pass,
        "status": status,
    }


def _plot(
    static_rows: Sequence[Dict[str, Any]],
    rot_rows: Sequence[Dict[str, Any]],
    kappa_rot: Optional[float],
    z_reject: float,
    out_png: Path,
) -> None:
    _set_japanese_font()

    fig = plt.figure(figsize=(13.0, 4.8))

    ax0 = fig.add_subplot(1, 3, 1)
    theta = np.linspace(0.0, math.pi, 200)
    r_ratio = 1.5
    static_profile = np.ones_like(theta)
    if kappa_rot is None:
        rot_profile = np.ones_like(theta)
    else:
        rot_profile = 1.0 + kappa_rot * (r_ratio ** -3) * (np.sin(theta) ** 2)
    ax0.plot(theta * 180.0 / math.pi, static_profile, color="#444444", linestyle="--", label="δP_rot=0")
    ax0.plot(theta * 180.0 / math.pi, rot_profile, color="#1f77b4", linewidth=2.0, label="δP_rot≠0")
    ax0.set_xlabel("polar angle θ [deg]")
    ax0.set_ylabel("P/P_static (r=1.5R)")
    ax0.set_title("Rotational P-profile (minimal ansatz)")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="best")

    frame_static = [r for r in static_rows if str(r.get("kind") or "") == "frame_dragging"]
    frame_rot = [r for r in rot_rows if str(r.get("kind") or "") == "frame_dragging"]
    labels = [str(r.get("label") or r.get("id") or "") for r in frame_static]

    obs_ratio: List[float] = []
    static_ratio: List[float] = []
    rot_ratio: List[float] = []
    for rs, rr in zip(frame_static, frame_rot):
        obs = _to_float(rs.get("observed"))
        ref = _to_float(rs.get("reference_prediction"))
        pred_static = _to_float(rs.get("pmodel_prediction"))
        pred_rot = _to_float(rr.get("pmodel_prediction"))
        obs_ratio.append(float("nan") if obs is None or ref is None or abs(ref) == 0 else obs / ref)
        static_ratio.append(float("nan") if pred_static is None or ref is None or abs(ref) == 0 else pred_static / ref)
        rot_ratio.append(float("nan") if pred_rot is None or ref is None or abs(ref) == 0 else pred_rot / ref)

    ax1 = fig.add_subplot(1, 3, 2)
    x = np.arange(len(labels), dtype=float)
    width = 0.26
    ax1.bar(x - width, obs_ratio, width=width, color="#1f77b4", label="observed/reference")
    ax1.bar(x, static_ratio, width=width, color="#d62728", alpha=0.9, label="δP_rot=0 prediction")
    ax1.bar(x + width, rot_ratio, width=width, color="#2ca02c", alpha=0.9, label="δP_rot≠0 prediction")
    ax1.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=12, ha="right")
    ax1.set_ylabel("dimensionless ratio")
    ax1.set_title("Frame-dragging ratio audit")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="best", fontsize=8.8)

    ax2 = fig.add_subplot(1, 3, 3)
    z_static = [float(r["z_score"]) if r.get("z_score") is not None else 0.0 for r in frame_static]
    z_rot = [float(r["z_score"]) if r.get("z_score") is not None else 0.0 for r in frame_rot]
    ax2.bar(x - width / 2.0, z_static, width=width, color="#d62728", alpha=0.92, label="δP_rot=0")
    ax2.bar(x + width / 2.0, z_rot, width=width, color="#2ca02c", alpha=0.92, label="δP_rot≠0")
    ax2.axhline(z_reject, color="#333333", linestyle="--", linewidth=1.0)
    ax2.axhline(-z_reject, color="#333333", linestyle="--", linewidth=1.0)
    ax2.axhline(0.0, color="#666666", linestyle="-", linewidth=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=12, ha="right")
    ax2.set_ylabel("z = (obs - pred) / sigma")
    ax2.set_title("Precession gate comparison")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(loc="best")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
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
        for r in rows:
            w.writerow(
                [
                    r.get("branch", ""),
                    r.get("id", ""),
                    r.get("label", ""),
                    r.get("kind", ""),
                    r.get("experiment", ""),
                    r.get("observable", ""),
                    r.get("unit", ""),
                    r.get("value_domain", ""),
                    "" if r.get("observed") is None else _fmt_float(float(r["observed"]), 8),
                    "" if r.get("observed_sigma") is None else _fmt_float(float(r["observed_sigma"]), 8),
                    ""
                    if r.get("reference_prediction") is None
                    else _fmt_float(float(r["reference_prediction"]), 8),
                    "" if r.get("pmodel_prediction") is None else _fmt_float(float(r["pmodel_prediction"]), 8),
                    "" if r.get("residual") is None else _fmt_float(float(r["residual"]), 8),
                    "" if r.get("z_score") is None else _fmt_float(float(r["z_score"]), 8),
                    r.get("status", ""),
                    r.get("note", ""),
                ]
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _ROOT
    default_gpb_data = root / "data" / "theory" / "gpb_scalar_limit_audit.json"
    default_frame_data = root / "data" / "theory" / "frame_dragging_experiments.json"
    default_metric_audit_json = root / "output" / "public" / "theory" / "pmodel_vector_metric_choice_audit_caseB_effective.json"
    default_strong_field_audit_json = root / "output" / "public" / "theory" / "pmodel_rotating_bh_photon_ring_direct_audit.json"
    default_outdir = root / "output" / "private" / "theory"
    default_public_outdir = root / "output" / "public" / "theory"

    ap = argparse.ArgumentParser(
        description="Rotating-sphere P-distribution and precession audit (deltaP_rot=0 vs deltaP_rot!=0)."
    )
    ap.add_argument("--gpb-data", type=str, default=str(default_gpb_data), help="Input JSON for GP-B channels.")
    ap.add_argument(
        "--frame-data",
        type=str,
        default=str(default_frame_data),
        help="Input JSON for frame-dragging experiments (LAGEOS included).",
    )
    ap.add_argument(
        "--metric-audit-json",
        type=str,
        default=str(default_metric_audit_json),
        help="Step 8.7.32.8 metric-choice audit JSON (caseB effective).",
    )
    ap.add_argument(
        "--strong-field-audit-json",
        type=str,
        default=str(default_strong_field_audit_json),
        help="Strong-field axisymmetric PDE audit JSON (boundary closure source).",
    )
    ap.add_argument("--outdir", type=str, default=str(default_outdir), help="Output directory.")
    ap.add_argument("--public-outdir", type=str, default=str(default_public_outdir), help="Public output directory.")
    ap.add_argument("--z-reject", type=float, default=3.0, help="Reject gate on |z|.")
    ap.add_argument(
        "--flux-closure-threshold",
        type=float,
        default=1.0e-3,
        help="Threshold for boundary flux closure gate (max_flux_rel_std).",
    )
    ap.add_argument("--no-public-copy", action="store_true", help="Do not copy outputs to public directory.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    gpb_data = Path(args.gpb_data)
    frame_data = Path(args.frame_data)
    metric_audit_json = Path(args.metric_audit_json)
    strong_field_audit_json = Path(args.strong_field_audit_json)
    outdir = Path(args.outdir)
    public_outdir = Path(args.public_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    public_outdir.mkdir(parents=True, exist_ok=True)

    gpb_payload = _read_json(gpb_data)
    frame_payload = _read_json(frame_data)
    metric_payload = _read_json_if_exists(metric_audit_json)
    strong_payload = _read_json_if_exists(strong_field_audit_json)
    geodetic_row, frame_rows = _extract_inputs(gpb_payload, frame_payload)

    fit = _fit_kappa(frame_rows)
    kappa_rot = _to_float(fit.get("kappa_rot"))

    static_rows = _build_branch_rows(
        branch="static_iso",
        frame_rows=frame_rows,
        geodetic_row=geodetic_row,
        kappa_rot=0.0,
        z_reject=float(args.z_reject),
    )
    rot_rows = _build_branch_rows(
        branch="vortex_gradient",
        frame_rows=frame_rows,
        geodetic_row=geodetic_row,
        kappa_rot=kappa_rot,
        z_reject=float(args.z_reject),
    )

    static_summary = _branch_summary(static_rows, z_reject=float(args.z_reject))
    rot_summary = _branch_summary(rot_rows, z_reject=float(args.z_reject))

    overall_decision = "inconclusive"
    if static_summary.get("status") == "reject" and rot_summary.get("status") in {"pass", "watch"}:
        overall_decision = "deltaP_rot_required_by_precession_data"
    elif static_summary.get("status") == "reject" and rot_summary.get("status") == "reject":
        overall_decision = "current_rotational_extension_rejected"
    elif static_summary.get("status") == "pass" and rot_summary.get("status") == "pass":
        overall_decision = "both_branches_allowed_currently"

    metric_bridge = _extract_metric_bridge(metric_payload)
    boundary_bridge = _extract_boundary_bridge(strong_payload, flux_closure_threshold=float(args.flux_closure_threshold))

    overall_status = "watch"
    if overall_decision == "current_rotational_extension_rejected":
        overall_status = "reject"
    elif (
        overall_decision == "deltaP_rot_required_by_precession_data"
        and metric_bridge.get("gate_pass")
        and boundary_bridge.get("gate_pass")
    ):
        overall_status = "pass"
        overall_decision = "deltaP_rot_required_under_effective_metric"
    elif overall_decision == "both_branches_allowed_currently":
        overall_status = "watch"
    elif overall_decision == "inconclusive":
        overall_status = "inconclusive"

    out_json = outdir / "pmodel_rotating_sphere_p_distribution_audit.json"
    out_csv = outdir / "pmodel_rotating_sphere_p_distribution_audit.csv"
    out_png = outdir / "pmodel_rotating_sphere_p_distribution_audit.png"

    all_rows = static_rows + rot_rows
    payload_out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "schema": "wavep.theory.pmodel_rotating_sphere_p_distribution_audit.v2",
        "title": "Rotating-sphere P distribution audit (deltaP_rot branches; effective-metric bridge)",
        "equations": {
            "static": "P_static(r) = P0 * exp(-Phi(r)/c^2)",
            "rotating_ansatz": "P_rot(r,theta) = P_static(r) * (1 + kappa_rot * (R/r)^3 * sin(theta)^2)",
            "precession_mapping": "Omega_LT^(P) = kappa_rot * Omega_LT^(ref)",
            "effective_metric_vacuum": "nabla^(g(P))_mu F_(P)^(mu nu) = 0",
        },
        "gate": {
            "z_reject": float(args.z_reject),
            "flux_closure_threshold": float(args.flux_closure_threshold),
        },
        "inputs": {
            "gpb_data": str(gpb_data).replace("\\", "/"),
            "frame_data": str(frame_data).replace("\\", "/"),
            "metric_audit_json": str(metric_audit_json).replace("\\", "/"),
            "strong_field_audit_json": str(strong_field_audit_json).replace("\\", "/"),
        },
        "calibration": fit,
        "effective_metric_bridge": metric_bridge,
        "boundary_bridge": boundary_bridge,
        "branches": {
            "static_iso": {
                "assumption": "deltaP_rot = 0",
                "summary": static_summary,
                "rows": static_rows,
            },
            "vortex_gradient": {
                "assumption": "deltaP_rot != 0",
                "summary": rot_summary,
                "rows": rot_rows,
            },
        },
        "summary": {
            "overall_status": overall_status,
            "decision": overall_decision,
            "static_status": static_summary.get("status"),
            "vortex_status": rot_summary.get("status"),
            "frame_dragging_reject_static_n": static_summary.get("reject_n"),
            "frame_dragging_reject_vortex_n": rot_summary.get("reject_n"),
            "metric_bridge_status": metric_bridge.get("status"),
            "boundary_bridge_status": boundary_bridge.get("status"),
        },
        "outputs": {
            "rows_json": str(out_json).replace("\\", "/"),
            "rows_csv": str(out_csv).replace("\\", "/"),
            "plot_png": str(out_png).replace("\\", "/"),
        },
    }

    _plot(static_rows, rot_rows, kappa_rot, float(args.z_reject), out_png)
    _write_json(out_json, payload_out)
    _write_csv(out_csv, all_rows)

    copied: List[Path] = []
    if not args.no_public_copy:
        for src in (out_json, out_csv, out_png):
            dst = public_outdir / src.name
            shutil.copy2(src, dst)
            copied.append(dst)

    try:
        worklog.append_event(
            {
                "event_type": "theory_rotating_sphere_p_distribution_audit",
                "argv": sys.argv,
                "inputs": {"gpb_data": gpb_data, "frame_data": frame_data},
                "outputs": {
                    "rows_json": out_json,
                    "rows_csv": out_csv,
                    "plot_png": out_png,
                    "public_copies": copied,
                },
                "metrics": {
                    "kappa_rot": fit.get("kappa_rot"),
                    "kappa_sigma": fit.get("kappa_sigma"),
                    "fit_channels_n": fit.get("fit_channels_n"),
                    "decision": overall_decision,
                    "overall_status": overall_status,
                    "static_status": static_summary.get("status"),
                    "vortex_status": rot_summary.get("status"),
                    "metric_bridge_pass": metric_bridge.get("gate_pass"),
                    "boundary_bridge_pass": boundary_bridge.get("gate_pass"),
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] json : {out_json}")
    print(f"[ok] csv  : {out_csv}")
    print(f"[ok] png  : {out_png}")
    if copied:
        print(f"[ok] public copies: {len(copied)} files -> {public_outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
