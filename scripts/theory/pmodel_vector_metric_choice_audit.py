from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
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


def _status_bool(value: bool) -> str:
    return "pass" if value else "reject"


def _load_common_inputs(mercury_payload: Dict[str, Any], solar_payload: Dict[str, Any]) -> Dict[str, float]:
    mercury_obs = _to_float(mercury_payload.get("reference_arcsec_century"))
    mercury_gr = _to_float(((mercury_payload.get("einstein_approx") or {}).get("arcsec_per_century")))
    if mercury_obs is None:
        raise SystemExit("mercury reference_arcsec_century missing")
    if mercury_gr is None or mercury_gr <= 0.0:
        raise SystemExit("mercury einstein_approx.arcsec_per_century missing")

    solar_metrics = solar_payload.get("metrics") or {}
    gamma_obs_best = _to_float(solar_metrics.get("observed_gamma_best"))
    gamma_obs_sigma = _to_float(solar_metrics.get("observed_gamma_best_sigma"))
    if gamma_obs_best is None:
        raise SystemExit("solar observed_gamma_best missing")
    if gamma_obs_sigma is None or gamma_obs_sigma <= 0.0:
        raise SystemExit("solar observed_gamma_best_sigma missing")

    return {
        "mercury_observed_arcsec_century": float(mercury_obs),
        "mercury_gr_like_arcsec_century": float(mercury_gr),
        "mercury_coeff_observed": float(mercury_obs / mercury_gr),
        "solar_gamma_observed_best": float(gamma_obs_best),
        "solar_gamma_observed_sigma_best": float(gamma_obs_sigma),
    }


def _load_beta_from_sources(
    *, beta_override: Optional[float], frozen_path: Path, solar_payload: Dict[str, Any], fallback: float
) -> float:
    beta_value = _to_float(beta_override)
    if beta_value is not None:
        return float(beta_value)

    try:
        frozen_payload = _read_json(frozen_path)
        beta_value = _to_float(frozen_payload.get("beta"))
    except Exception:
        beta_value = None
    if beta_value is not None:
        return float(beta_value)

    beta_value = _to_float(((solar_payload.get("metrics") or {}).get("beta")))
    if beta_value is not None:
        return float(beta_value)
    return float(fallback)


def _row_summary(rows: Sequence[Dict[str, Any]]) -> Tuple[int, int, int]:
    hard_reject = 0
    watch_n = 0
    pass_n = 0
    for row in rows:
        status = str(row.get("status") or "")
        if status == "reject":
            hard_reject += 1
        elif status == "watch":
            watch_n += 1
        elif status == "pass":
            pass_n += 1
    return hard_reject, watch_n, pass_n


def _build_case_a(
    *,
    common: Dict[str, float],
    beta_flat: float,
    gamma_flat: float,
    mercury_coeff_threshold: float,
    z_reject: float,
) -> Dict[str, Any]:
    mercury_coeff_model = (2.0 - beta_flat + 2.0 * gamma_flat) / 3.0
    mercury_coeff_obs = common["mercury_coeff_observed"]
    mercury_pred_model = mercury_coeff_model * common["mercury_gr_like_arcsec_century"]
    mercury_residual = mercury_pred_model - common["mercury_observed_arcsec_century"]
    mercury_rel_error = (
        mercury_residual / common["mercury_observed_arcsec_century"] if common["mercury_observed_arcsec_century"] != 0.0 else None
    )
    mercury_coeff_diff = mercury_coeff_model - mercury_coeff_obs

    gamma_obs = common["solar_gamma_observed_best"]
    gamma_sigma = common["solar_gamma_observed_sigma_best"]
    z_gamma = (gamma_obs - gamma_flat) / gamma_sigma
    light_coeff_model = 0.5 * (1.0 + gamma_flat)
    light_coeff_obs = 0.5 * (1.0 + gamma_obs)

    rows: List[Dict[str, Any]] = [
        {
            "gate_id": "metric_choice::caseA_flat_mercury_coeff",
            "observable": "mercury_precession_coeff",
            "case": "A_flat_eta",
            "value": mercury_coeff_model,
            "reference": mercury_coeff_obs,
            "residual": mercury_coeff_diff,
            "threshold": mercury_coeff_threshold,
            "comparator": "|value-reference|<=threshold",
            "status": _status_bool(abs(mercury_coeff_diff) <= mercury_coeff_threshold),
            "note": "kappa_flat=(2-beta+2gamma)/3 with gamma_flat=0",
        },
        {
            "gate_id": "metric_choice::caseA_flat_light_deflection_gamma",
            "observable": "ppn_gamma",
            "case": "A_flat_eta",
            "value": gamma_flat,
            "reference": gamma_obs,
            "residual": gamma_flat - gamma_obs,
            "threshold": z_reject,
            "comparator": "|z|<=threshold",
            "status": _status_bool(abs(z_gamma) <= z_reject),
            "z_score": z_gamma,
            "note": "gamma from best VLBI constraint row",
        },
        {
            "gate_id": "metric_choice::caseA_flat_pde_closure",
            "observable": "flat_linear_pde_closure",
            "case": "A_flat_eta",
            "value": 1.0,
            "reference": 1.0,
            "residual": 0.0,
            "threshold": 0.0,
            "comparator": "linear closure expected",
            "status": "pass",
            "note": "partial_mu F^{mu nu}=0 on eta_{mu nu}",
        },
    ]

    hard_reject, watch_n, _ = _row_summary(rows)
    overall_status = "pass" if hard_reject == 0 else "reject"
    decision = "flat_metric_caseA_baseline_fixed" if overall_status == "pass" else "flat_metric_caseA_reject_baseline_fixed"

    return {
        "case_id": "8.7.32.8.caseA",
        "case_name": "flat_metric_eta",
        "inputs": {
            "beta_model": beta_flat,
            "gamma_model": gamma_flat,
            **common,
        },
        "derived": {
            "mercury_coeff_model": mercury_coeff_model,
            "mercury_coeff_observed": mercury_coeff_obs,
            "mercury_pred_model_arcsec_century": mercury_pred_model,
            "mercury_residual_arcsec_century": mercury_residual,
            "mercury_rel_error": mercury_rel_error,
            "light_gamma_model": gamma_flat,
            "light_gamma_observed": gamma_obs,
            "light_coeff_model": light_coeff_model,
            "light_coeff_observed": light_coeff_obs,
            "z_gamma": z_gamma,
        },
        "summary": {
            "overall_status": overall_status,
            "decision": decision,
            "hard_reject_n": hard_reject,
            "watch_n": watch_n,
            "metric_choice_decision": "defer_case_b_required",
            "rows": rows,
        },
    }


def _build_case_b(
    *,
    common: Dict[str, float],
    strong_field_payload: Dict[str, Any],
    beta_effective: float,
    gamma_effective: float,
    mercury_coeff_threshold: float,
    z_reject: float,
    flux_closure_threshold: float,
) -> Dict[str, Any]:
    mercury_coeff_model = (2.0 - beta_effective + 2.0 * gamma_effective) / 3.0
    mercury_coeff_obs = common["mercury_coeff_observed"]
    mercury_pred_model = mercury_coeff_model * common["mercury_gr_like_arcsec_century"]
    mercury_residual = mercury_pred_model - common["mercury_observed_arcsec_century"]
    mercury_rel_error = (
        mercury_residual / common["mercury_observed_arcsec_century"] if common["mercury_observed_arcsec_century"] != 0.0 else None
    )
    mercury_coeff_diff = mercury_coeff_model - mercury_coeff_obs

    gamma_obs = common["solar_gamma_observed_best"]
    gamma_sigma = common["solar_gamma_observed_sigma_best"]
    z_gamma = (gamma_obs - gamma_effective) / gamma_sigma
    light_coeff_model = 0.5 * (1.0 + gamma_effective)
    light_coeff_obs = 0.5 * (1.0 + gamma_obs)

    axis_block = strong_field_payload.get("axisymmetric_pde_block") or {}
    boundary_diagnostics = axis_block.get("boundary_diagnostics") or {}
    pde_formulation_complete = bool(axis_block.get("formulation_complete"))
    boundary_closure_pass = bool(axis_block.get("boundary_closure_pass"))
    flux_metric = _to_float(boundary_diagnostics.get("max_flux_rel_std"))
    flux_closure_pass = flux_metric is not None and flux_metric <= flux_closure_threshold
    nonlinear_pde_closure_pass = pde_formulation_complete and boundary_closure_pass and flux_closure_pass

    rows: List[Dict[str, Any]] = [
        {
            "gate_id": "metric_choice::caseB_effective_mercury_coeff",
            "observable": "mercury_precession_coeff",
            "case": "B_effective_metric",
            "value": mercury_coeff_model,
            "reference": mercury_coeff_obs,
            "residual": mercury_coeff_diff,
            "threshold": mercury_coeff_threshold,
            "comparator": "|value-reference|<=threshold",
            "status": _status_bool(abs(mercury_coeff_diff) <= mercury_coeff_threshold),
            "note": "kappa_effective=(2-beta+2gamma_effective)/3",
        },
        {
            "gate_id": "metric_choice::caseB_effective_light_deflection_gamma",
            "observable": "ppn_gamma",
            "case": "B_effective_metric",
            "value": gamma_effective,
            "reference": gamma_obs,
            "residual": gamma_effective - gamma_obs,
            "threshold": z_reject,
            "comparator": "|z|<=threshold",
            "status": _status_bool(abs(z_gamma) <= z_reject),
            "z_score": z_gamma,
            "note": "gamma_effective from g_{mu nu}(P) weak-field mapping",
        },
        {
            "gate_id": "metric_choice::caseB_effective_pde_closure",
            "observable": "nonlinear_pde_closure",
            "case": "B_effective_metric",
            "value": 1.0 if nonlinear_pde_closure_pass else 0.0,
            "reference": 1.0,
            "residual": 0.0 if nonlinear_pde_closure_pass else -1.0,
            "threshold": flux_closure_threshold,
            "comparator": "formulation&&boundary&&max_flux<=threshold",
            "status": _status_bool(nonlinear_pde_closure_pass),
            "note": "from pmodel_rotating_bh_photon_ring_direct_audit.axisymmetric_pde_block",
        },
    ]

    hard_reject, watch_n, _ = _row_summary(rows)
    overall_status = "pass" if hard_reject == 0 else "reject"
    decision = (
        "effective_metric_caseB_baseline_fixed" if overall_status == "pass" else "effective_metric_caseB_reject_baseline_fixed"
    )

    return {
        "case_id": "8.7.32.8.caseB",
        "case_name": "effective_metric_gmu_nu_P",
        "inputs": {
            "beta_model": beta_effective,
            "gamma_model": gamma_effective,
            **common,
            "strong_field_json_loaded": True,
            "flux_closure_threshold": flux_closure_threshold,
        },
        "derived": {
            "mercury_coeff_model": mercury_coeff_model,
            "mercury_coeff_observed": mercury_coeff_obs,
            "mercury_pred_model_arcsec_century": mercury_pred_model,
            "mercury_residual_arcsec_century": mercury_residual,
            "mercury_rel_error": mercury_rel_error,
            "light_gamma_model": gamma_effective,
            "light_gamma_observed": gamma_obs,
            "light_coeff_model": light_coeff_model,
            "light_coeff_observed": light_coeff_obs,
            "z_gamma": z_gamma,
            "pde_formulation_complete": pde_formulation_complete,
            "boundary_closure_pass": boundary_closure_pass,
            "max_flux_rel_std": flux_metric,
            "flux_closure_pass": flux_closure_pass,
            "nonlinear_pde_closure_pass": nonlinear_pde_closure_pass,
        },
        "summary": {
            "overall_status": overall_status,
            "decision": decision,
            "hard_reject_n": hard_reject,
            "watch_n": watch_n,
            "metric_choice_decision": "defer_case_comparison",
            "rows": rows,
        },
    }


def _load_case_a_summary(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = _read_json(path)
    except Exception:
        return None
    case_result = payload.get("case_result") or {}
    summary = case_result.get("summary")
    if isinstance(summary, dict):
        return summary
    return None


def _resolve_metric_choice_decision(
    *,
    case_mode: str,
    case_summary: Dict[str, Any],
    case_a_summary: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    if case_mode == "flat":
        return "defer_case_b_required", {
            "case_mode": "flat",
            "case_a_status": case_summary.get("overall_status"),
            "case_b_status": None,
            "reason": "caseA baseline fixed; caseB evaluation required for final choice",
        }

    case_b_status = str(case_summary.get("overall_status") or "reject")
    case_a_status = str((case_a_summary or {}).get("overall_status") or "missing")
    if case_a_status == "missing":
        return "defer_case_a_missing", {
            "case_mode": "effective",
            "case_a_status": case_a_status,
            "case_b_status": case_b_status,
            "reason": "caseA baseline file missing",
        }

    if case_b_status == "pass" and case_a_status == "reject":
        return "effective", {
            "case_mode": "effective",
            "case_a_status": case_a_status,
            "case_b_status": case_b_status,
            "reason": "caseB passes while caseA rejected",
        }
    if case_b_status == "pass" and case_a_status == "pass":
        return "defer_both_pass_need_tighter_gate", {
            "case_mode": "effective",
            "case_a_status": case_a_status,
            "case_b_status": case_b_status,
            "reason": "both cases pass",
        }
    if case_b_status == "reject" and case_a_status == "pass":
        return "flat", {
            "case_mode": "effective",
            "case_a_status": case_a_status,
            "case_b_status": case_b_status,
            "reason": "caseA passes while caseB rejected",
        }
    if case_b_status == "reject" and case_a_status == "reject":
        return "reject_both_cases", {
            "case_mode": "effective",
            "case_a_status": case_a_status,
            "case_b_status": case_b_status,
            "reason": "both cases rejected",
        }
    return "defer", {
        "case_mode": "effective",
        "case_a_status": case_a_status,
        "case_b_status": case_b_status,
        "reason": "indeterminate",
    }


def _plot(case_payload: Dict[str, Any], out_png: Path, title: str, row_prefix: str) -> None:
    _set_japanese_font()

    derived = case_payload.get("derived") or {}
    inputs = case_payload.get("inputs") or {}
    summary = case_payload.get("summary") or {}
    rows = summary.get("rows") or []

    mercury_coeff_model = float(derived.get("mercury_coeff_model"))
    mercury_coeff_obs = float(derived.get("mercury_coeff_observed"))
    light_coeff_model = float(derived.get("light_coeff_model"))
    light_coeff_obs = float(derived.get("light_coeff_observed"))
    gamma_model = float(derived.get("light_gamma_model"))
    gamma_obs = float(derived.get("light_gamma_observed"))
    gamma_sigma = float(inputs.get("solar_gamma_observed_sigma_best"))
    z_gamma = float(derived.get("z_gamma"))

    figure, (axis0, axis1, axis2) = plt.subplots(1, 3, figsize=(15.0, 5.0))
    figure.suptitle(title)

    labels = ["Mercury coeff", "Light coeff"]
    obs_values = [mercury_coeff_obs, light_coeff_obs]
    model_values = [mercury_coeff_model, light_coeff_model]
    x_values = np.arange(len(labels))
    width = 0.36
    axis0.bar(x_values - width / 2.0, obs_values, width=width, label="observed-derived", color="#1f77b4")
    axis0.bar(x_values + width / 2.0, model_values, width=width, label="model-case", color="#d62728")
    axis0.axhline(1.0, linestyle="--", linewidth=1.0, color="#555555")
    axis0.set_xticks(x_values)
    axis0.set_xticklabels(labels, rotation=8)
    axis0.set_ylabel("coefficient (normalized)")
    axis0.set_title("Weak-field coefficient comparison")
    axis0.grid(True, axis="y", alpha=0.25)
    axis0.legend(loc="best")

    axis1.errorbar(["obs"], [gamma_obs], yerr=[gamma_sigma], fmt="o", color="#1f77b4", capsize=4, label="observed")
    axis1.scatter(["model"], [gamma_model], color="#d62728", label="model-case")
    axis1.axhline(1.0, linestyle="--", linewidth=1.0, color="#555555")
    axis1.set_ylabel("PPN gamma")
    axis1.set_title(f"Light-deflection gamma (z={z_gamma:.2f})")
    axis1.grid(True, axis="y", alpha=0.25)
    axis1.legend(loc="best")

    gate_ids = [str(row.get("gate_id") or "").replace(row_prefix, "") for row in rows]
    gate_values = [1.0 if row.get("status") == "pass" else 0.0 for row in rows]
    colors = ["#2ca02c" if value > 0.5 else "#d62728" for value in gate_values]
    idx = np.arange(len(gate_ids))
    axis2.barh(idx, gate_values, color=colors)
    axis2.set_yticks(idx)
    axis2.set_yticklabels(gate_ids)
    axis2.set_xlim(0.0, 1.0)
    axis2.set_xlabel("pass=1 / reject=0")
    axis2.set_title(f"Gate summary ({summary.get('overall_status')})")
    axis2.grid(True, axis="x", alpha=0.25)

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_png, dpi=200)
    plt.close(figure)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "gate_id",
                "observable",
                "case",
                "value",
                "reference",
                "residual",
                "threshold",
                "comparator",
                "status",
                "z_score",
                "note",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.get("gate_id", ""),
                    row.get("observable", ""),
                    row.get("case", ""),
                    "" if row.get("value") is None else _fmt_float(float(row["value"]), 10),
                    "" if row.get("reference") is None else _fmt_float(float(row["reference"]), 10),
                    "" if row.get("residual") is None else _fmt_float(float(row["residual"]), 10),
                    "" if row.get("threshold") is None else _fmt_float(float(row["threshold"]), 10),
                    row.get("comparator", ""),
                    row.get("status", ""),
                    "" if row.get("z_score") is None else _fmt_float(float(row["z_score"]), 10),
                    row.get("note", ""),
                ]
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    default_mercury = _ROOT / "output" / "private" / "mercury" / "mercury_precession_metrics.json"
    default_solar = _ROOT / "output" / "private" / "theory" / "solar_light_deflection_metrics.json"
    default_frozen = _ROOT / "output" / "private" / "theory" / "frozen_parameters.json"
    default_strong_field = _ROOT / "output" / "public" / "theory" / "pmodel_rotating_bh_photon_ring_direct_audit.json"
    default_outdir = _ROOT / "output" / "private" / "theory"
    default_public_outdir = _ROOT / "output" / "public" / "theory"

    parser = argparse.ArgumentParser(description="Metric-choice audit for Step 8.7.32.8 (case A/B).")
    parser.add_argument("--case", choices=["flat", "effective"], default="flat", help="Audit case mode.")
    parser.add_argument("--mercury-metrics", type=str, default=str(default_mercury), help="Path to Mercury metrics JSON.")
    parser.add_argument("--solar-metrics", type=str, default=str(default_solar), help="Path to solar light-deflection metrics JSON.")
    parser.add_argument("--frozen-parameters", type=str, default=str(default_frozen), help="Path to frozen parameters JSON.")
    parser.add_argument("--strong-field-metrics", type=str, default=str(default_strong_field), help="Path to strong-field axisymmetric PDE JSON.")
    parser.add_argument("--beta-flat", type=float, default=None, help="Override beta for flat case.")
    parser.add_argument("--gamma-flat", type=float, default=0.0, help="Flat background PPN gamma baseline.")
    parser.add_argument("--beta-effective", type=float, default=None, help="Override beta for effective-metric case.")
    parser.add_argument("--gamma-effective", type=float, default=None, help="Override gamma for effective-metric case.")
    parser.add_argument("--mercury-coeff-threshold", type=float, default=0.05, help="Gate threshold for Mercury coefficient residual.")
    parser.add_argument("--z-reject", type=float, default=3.0, help="Reject threshold for |z|.")
    parser.add_argument("--flux-closure-threshold", type=float, default=1.0e-3, help="Max flux std threshold for nonlinear PDE closure.")
    parser.add_argument("--case-a-baseline-json", type=str, default=None, help="Optional caseA baseline JSON path for case comparison.")
    parser.add_argument("--outdir", type=str, default=str(default_outdir), help="Output directory.")
    parser.add_argument("--public-outdir", type=str, default=str(default_public_outdir), help="Public output directory.")
    parser.add_argument("--no-public-copy", action="store_true", help="Do not copy fixed outputs to public dir.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    outdir = Path(args.outdir)
    public_outdir = Path(args.public_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    public_outdir.mkdir(parents=True, exist_ok=True)

    mercury_payload = _read_json(Path(args.mercury_metrics))
    solar_payload = _read_json(Path(args.solar_metrics))
    common = _load_common_inputs(mercury_payload, solar_payload)

    if args.case == "flat":
        beta_model = _load_beta_from_sources(
            beta_override=args.beta_flat,
            frozen_path=Path(args.frozen_parameters),
            solar_payload=solar_payload,
            fallback=1.0,
        )
        case_payload = _build_case_a(
            common=common,
            beta_flat=beta_model,
            gamma_flat=float(args.gamma_flat),
            mercury_coeff_threshold=float(args.mercury_coeff_threshold),
            z_reject=float(args.z_reject),
        )
        schema = "wavep.theory.pmodel_vector_metric_choice_audit.caseA.v2"
        case_step = "8.7.32.8.caseA"
        title = "Step 8.7.32.8 case A: flat metric eta_{mu nu} baseline"
        file_stem = "pmodel_vector_metric_choice_audit_caseA_flat"
        row_prefix = "metric_choice::caseA_flat_"
        event_type = "theory_pmodel_vector_metric_choice_audit_caseA_flat"
        equations = {
            "precession_coeff": "kappa_flat=(2-beta+2*gamma)/3",
            "light_deflection_coeff": "kappa_light=(1+gamma)/2",
            "flat_pde": "partial_mu F^{mu nu}=0 on eta_{mu nu}",
        }
        intent = "Fix flat-metric baseline using identical weak-field interfaces."
        case_a_summary = None
    else:
        beta_model = _load_beta_from_sources(
            beta_override=args.beta_effective,
            frozen_path=Path(args.frozen_parameters),
            solar_payload=solar_payload,
            fallback=1.0,
        )
        gamma_model = _to_float(args.gamma_effective)
        if gamma_model is None:
            gamma_model = float(2.0 * beta_model - 1.0)
        strong_field_payload = _read_json(Path(args.strong_field_metrics))
        case_payload = _build_case_b(
            common=common,
            strong_field_payload=strong_field_payload,
            beta_effective=beta_model,
            gamma_effective=float(gamma_model),
            mercury_coeff_threshold=float(args.mercury_coeff_threshold),
            z_reject=float(args.z_reject),
            flux_closure_threshold=float(args.flux_closure_threshold),
        )
        schema = "wavep.theory.pmodel_vector_metric_choice_audit.caseB.v2"
        case_step = "8.7.32.8.caseB"
        title = "Step 8.7.32.8 case B: effective metric g_{mu nu}(P) baseline"
        file_stem = "pmodel_vector_metric_choice_audit_caseB_effective"
        row_prefix = "metric_choice::caseB_effective_"
        event_type = "theory_pmodel_vector_metric_choice_audit_caseB_effective"
        equations = {
            "precession_coeff": "kappa_effective=(2-beta+2*gamma_effective)/3",
            "light_deflection_coeff": "kappa_light=(1+gamma_effective)/2",
            "nonlinear_pde_gate": "formulation_complete && boundary_closure_pass && max_flux_rel_std<=threshold",
        }
        intent = "Fix effective-metric baseline using weak-field coefficients and nonlinear PDE closure gate."
        case_a_path = Path(args.case_a_baseline_json) if args.case_a_baseline_json else (outdir / "pmodel_vector_metric_choice_audit_caseA_flat.json")
        case_a_summary = _load_case_a_summary(case_a_path)

    metric_choice_decision, comparison = _resolve_metric_choice_decision(
        case_mode=args.case,
        case_summary=case_payload["summary"],
        case_a_summary=case_a_summary,
    )
    case_payload["summary"]["metric_choice_decision"] = metric_choice_decision

    out_json = outdir / f"{file_stem}.json"
    out_csv = outdir / f"{file_stem}.csv"
    out_png = outdir / f"{file_stem}.png"

    _plot(case_payload, out_png, title=title, row_prefix=row_prefix)
    _write_csv(out_csv, case_payload["summary"]["rows"])

    payload_out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "schema": schema,
        "title": title,
        "step": case_step,
        "intent": intent,
        "equations": equations,
        "inputs": {
            "case_mode": args.case,
            "mercury_metrics_json": str(Path(args.mercury_metrics)).replace("\\", "/"),
            "solar_metrics_json": str(Path(args.solar_metrics)).replace("\\", "/"),
            "frozen_parameters_json": str(Path(args.frozen_parameters)).replace("\\", "/"),
            "strong_field_metrics_json": str(Path(args.strong_field_metrics)).replace("\\", "/"),
            "mercury_coeff_threshold": float(args.mercury_coeff_threshold),
            "z_reject": float(args.z_reject),
            "flux_closure_threshold": float(args.flux_closure_threshold),
        },
        "case_result": case_payload,
        "case_comparison": comparison,
        "metric_choice_decision": metric_choice_decision,
        "outputs": {
            "rows_json": str(out_json).replace("\\", "/"),
            "rows_csv": str(out_csv).replace("\\", "/"),
            "plot_png": str(out_png).replace("\\", "/"),
        },
    }
    _write_json(out_json, payload_out)

    public_copies: List[Path] = []
    if not args.no_public_copy:
        for source in (out_json, out_csv, out_png):
            destination = public_outdir / source.name
            shutil.copy2(source, destination)
            public_copies.append(destination)

    summary = case_payload.get("summary") or {}
    derived = case_payload.get("derived") or {}
    try:
        worklog.append_event(
            {
                "event_type": event_type,
                "argv": sys.argv,
                "inputs": {
                    "case_mode": args.case,
                    "mercury_metrics": Path(args.mercury_metrics),
                    "solar_metrics": Path(args.solar_metrics),
                    "frozen_parameters": Path(args.frozen_parameters),
                    "strong_field_metrics": Path(args.strong_field_metrics),
                    "mercury_coeff_threshold": float(args.mercury_coeff_threshold),
                    "z_reject": float(args.z_reject),
                    "flux_closure_threshold": float(args.flux_closure_threshold),
                },
                "outputs": {
                    "rows_json": out_json,
                    "rows_csv": out_csv,
                    "plot_png": out_png,
                    "public_copies": public_copies,
                },
                "metrics": {
                    "overall_status": summary.get("overall_status"),
                    "decision": summary.get("decision"),
                    "metric_choice_decision": summary.get("metric_choice_decision"),
                    "mercury_coeff_model": derived.get("mercury_coeff_model"),
                    "z_gamma": derived.get("z_gamma"),
                    "hard_reject_n": summary.get("hard_reject_n"),
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
    print(
        "[ok] overall_status="
        f"{summary.get('overall_status')} decision={summary.get('decision')} metric_choice={summary.get('metric_choice_decision')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
