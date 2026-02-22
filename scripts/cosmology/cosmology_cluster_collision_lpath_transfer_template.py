#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_cluster_collision_lpath_transfer_template.py

Step 8.7.25.22:
Bullet 系で固定した L_path 比スケーリング則を用いて、
Bullet 再評価（同一I/F）と非Bullet系（例: El Gordo）への
転用テンプレートを同時に固定する。

固定出力:
  - output/public/cosmology/cosmology_cluster_collision_lpath_transfer_template.json
  - output/public/cosmology/cosmology_cluster_collision_lpath_transfer_template.csv
  - output/public/cosmology/cosmology_cluster_collision_lpath_transfer_template.png
  - output/public/cosmology/cluster_collision_lpath_transfer_input_template.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.summary import worklog  # type: ignore  # noqa: E402
except Exception:
    worklog = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None


@dataclass(frozen=True)
class TransferCase:
    case_id: str
    label: str
    cluster_family: str
    branch_anchor: str
    rho_ratio: Optional[float]
    v_ratio: Optional[float]
    temp_ratio: Optional[float]
    offset_obs_kpc: Optional[float]
    offset_sigma_kpc: Optional[float]
    source_mode: str
    note: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _parse_opt_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return _safe_float(value)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _load_cases(path: Path) -> List[TransferCase]:
    if not path.exists():
        raise FileNotFoundError(f"missing transfer cases CSV: {path}")
    out: List[TransferCase] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = str(row.get("case_id", "")).strip()
            if not case_id:
                continue
            label = str(row.get("label", case_id)).strip() or case_id
            cluster_family = str(row.get("cluster_family", "")).strip() or "unknown"
            branch_anchor = str(row.get("branch_anchor", "")).strip() or "bullet_main"
            source_mode = str(row.get("source_mode", "")).strip() or "template_pending"
            note = str(row.get("note", "")).strip()
            out.append(
                TransferCase(
                    case_id=case_id,
                    label=label,
                    cluster_family=cluster_family,
                    branch_anchor=branch_anchor,
                    rho_ratio=_parse_opt_float(row.get("rho_ratio")),
                    v_ratio=_parse_opt_float(row.get("v_ratio")),
                    temp_ratio=_parse_opt_float(row.get("temp_ratio")),
                    offset_obs_kpc=_parse_opt_float(row.get("offset_obs_kpc")),
                    offset_sigma_kpc=_parse_opt_float(row.get("offset_sigma_kpc")),
                    source_mode=source_mode,
                    note=note,
                )
            )
    if not out:
        raise RuntimeError(f"no valid rows in transfer cases CSV: {path}")
    return out


def _load_bullet_reference(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing Bullet derivation JSON: {path}")
    src = json.loads(path.read_text(encoding="utf-8"))
    cluster_rows = src.get("cluster_rows", [])
    if not isinstance(cluster_rows, list):
        cluster_rows = []
    pred_map: Dict[str, float] = {}
    for row in cluster_rows:
        cid = str(row.get("cluster_id", "")).strip()
        pred = _safe_float(row.get("pred_offset_kpc"))
        if cid and pred is not None and pred > 0.0:
            pred_map[cid] = float(abs(pred))
    if "bullet_main" not in pred_map or "bullet_sub" not in pred_map:
        raise RuntimeError("missing bullet_main/bullet_sub predicted offsets in derivation JSON")
    return {
        "source_json": src,
        "pred_offset_by_anchor": pred_map,
    }


def _load_lpath_reference(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing L_path scaling JSON: {path}")
    src = json.loads(path.read_text(encoding="utf-8"))
    ref = src.get("reference_state", {})
    decision = src.get("decision", {})
    inputs = src.get("inputs", {})
    pi0 = _safe_float(ref.get("pi0_median"))
    tau_free0 = _safe_float(ref.get("tau_free_gyr"))
    lpath0 = _safe_float(ref.get("lpath_kpc"))
    temp_power = _safe_float(inputs.get("temp_power_collisional_branch"))
    if pi0 is None or tau_free0 is None or lpath0 is None or temp_power is None:
        raise RuntimeError("missing reference_state/input fields in lpath scaling JSON")
    return {
        "source_json": src,
        "pi0": float(pi0),
        "tau_free0_gyr": float(tau_free0),
        "lpath0_kpc": float(lpath0),
        "temp_power": float(temp_power),
        "lpath_source_status": str(decision.get("overall_status", "unknown")),
        "lpath_source_decision": str(decision.get("decision", "unknown")),
    }


def _ratio_lpath(pi0: float, rho_ratio: float, v_ratio: float, temp_ratio: float, *, temp_power: float) -> float:
    rr = max(float(rho_ratio), 1.0e-12)
    rv = max(float(v_ratio), 1.0e-12)
    rt = max(float(temp_ratio), 1.0e-12)
    denom = rv + pi0 * rr * (rt**temp_power)
    if (not math.isfinite(denom)) or denom <= 0.0:
        return float("nan")
    return float((1.0 + pi0) / denom)


def _render_png(path: Path, rows: Sequence[Dict[str, Any]], *, pi0: float, temp_power: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if plt is None:
        path.write_bytes(b"")
        return

    labels = [str(r["case_id"]) for r in rows]
    ratios = [float(r["lpath_ratio"]) if _safe_float(r.get("lpath_ratio")) is not None else float("nan") for r in rows]
    x = np.arange(len(labels), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=160)

    valid_mask = np.isfinite(ratios)
    axes[0].bar(x[valid_mask], np.asarray(ratios)[valid_mask], color="#4c78a8")
    axes[0].axhline(1.0, color="#666666", linestyle="--", linewidth=1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_ylabel("L_path / L_path,0")
    axes[0].set_title("L_path transfer ratios")
    axes[0].grid(True, axis="y", alpha=0.25)

    obs_rows = [r for r in rows if _safe_float(r.get("offset_obs_kpc")) is not None and _safe_float(r.get("offset_sigma_kpc")) is not None]
    if obs_rows:
        x2 = np.arange(len(obs_rows), dtype=float)
        obs = np.array([float(r["offset_obs_kpc"]) for r in obs_rows], dtype=float)
        sig = np.array([float(r["offset_sigma_kpc"]) for r in obs_rows], dtype=float)
        pred = np.array([float(r["offset_pred_kpc"]) for r in obs_rows], dtype=float)
        labels2 = [str(r["case_id"]) for r in obs_rows]
        axes[1].errorbar(x2, obs, yerr=sig, fmt="o", color="#2ca02c", capsize=4, label="observed offset")
        axes[1].scatter(x2, pred, color="#d62728", marker="s", label="predicted offset")
        axes[1].set_xticks(x2)
        axes[1].set_xticklabels(labels2, rotation=25, ha="right")
        axes[1].set_ylabel("offset [kpc]")
        axes[1].set_title("Observed vs predicted (rows with observations)")
        axes[1].grid(True, axis="y", alpha=0.25)
        axes[1].legend(loc="best")
    else:
        axes[1].text(0.5, 0.5, "no observed offsets in transfer table", ha="center", va="center")
        axes[1].set_axis_off()

    fig.suptitle("Cluster-collision L_path transfer template audit")
    fig.text(
        0.01,
        -0.02,
        f"Pi0={pi0:.6f}, temp_power={temp_power:.3f}, formula: (1+Pi0)/(r_v + Pi0*r_rho*r_T^temp_power)",
        fontsize=8,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recheck Bullet offsets with fixed L_path ratio law and build transfer template for non-Bullet collisions."
    )
    parser.add_argument(
        "--cases-csv",
        type=Path,
        default=ROOT / "data" / "cosmology" / "bullet_cluster" / "collision_lpath_transfer_cases.csv",
        help="Input transfer case table (editable template).",
    )
    parser.add_argument(
        "--bullet-derivation-json",
        type=Path,
        default=ROOT / "output" / "public" / "cosmology" / "cosmology_cluster_collision_pmu_jmu_separation_derivation.json",
    )
    parser.add_argument(
        "--lpath-scaling-json",
        type=Path,
        default=ROOT / "output" / "public" / "cosmology" / "cosmology_lpath_scaling_law_prediction.json",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "output" / "private" / "cosmology",
    )
    parser.add_argument(
        "--public-outdir",
        type=Path,
        default=ROOT / "output" / "public" / "cosmology",
    )
    parser.add_argument("--step-tag", default="8.7.25.22")
    args = parser.parse_args()

    cases = _load_cases(args.cases_csv)
    bullet_ref = _load_bullet_reference(args.bullet_derivation_json)
    lpath_ref = _load_lpath_reference(args.lpath_scaling_json)

    pi0 = float(lpath_ref["pi0"])
    tau0 = float(lpath_ref["tau_free0_gyr"])
    lpath0 = float(lpath_ref["lpath0_kpc"])
    temp_power = float(lpath_ref["temp_power"])
    anchor_pred_map: Dict[str, float] = dict(bullet_ref["pred_offset_by_anchor"])

    rows_out: List[Dict[str, Any]] = []
    for case in cases:
        rho_ratio = case.rho_ratio
        v_ratio = case.v_ratio
        temp_ratio = case.temp_ratio
        has_ratio = (
            rho_ratio is not None
            and v_ratio is not None
            and temp_ratio is not None
            and rho_ratio > 0.0
            and v_ratio > 0.0
            and temp_ratio > 0.0
        )
        lpath_ratio = (
            _ratio_lpath(pi0, float(rho_ratio), float(v_ratio), float(temp_ratio), temp_power=temp_power)
            if has_ratio
            else None
        )
        tau_pred = float(tau0 * lpath_ratio) if lpath_ratio is not None and math.isfinite(lpath_ratio) else None
        lpath_pred = float(lpath0 * lpath_ratio) if lpath_ratio is not None and math.isfinite(lpath_ratio) else None

        offset_ref = anchor_pred_map.get(case.branch_anchor)
        offset_pred = (
            float(offset_ref * lpath_ratio)
            if (offset_ref is not None and lpath_ratio is not None and math.isfinite(lpath_ratio))
            else None
        )

        obs = case.offset_obs_kpc
        sig = case.offset_sigma_kpc
        residual = (float(offset_pred - obs) if (offset_pred is not None and obs is not None) else None)
        z = (
            float(residual / sig)
            if (residual is not None and sig is not None and sig > 0.0)
            else None
        )

        if not has_ratio:
            row_status = "input_missing_ratio"
        elif offset_ref is None:
            row_status = "unknown_branch_anchor"
        elif obs is None or sig is None or sig <= 0.0:
            row_status = "template_no_observation"
        elif z is None or (not math.isfinite(z)):
            row_status = "invalid_observation"
        elif abs(z) <= 3.0:
            row_status = "pass"
        else:
            row_status = "reject"

        rows_out.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "cluster_family": case.cluster_family,
                "branch_anchor": case.branch_anchor,
                "rho_ratio": (None if rho_ratio is None else float(rho_ratio)),
                "v_ratio": (None if v_ratio is None else float(v_ratio)),
                "temp_ratio": (None if temp_ratio is None else float(temp_ratio)),
                "lpath_ratio": lpath_ratio,
                "tau_free_pred_gyr": tau_pred,
                "lpath_pred_kpc": lpath_pred,
                "offset_ref_kpc": offset_ref,
                "offset_pred_kpc": offset_pred,
                "offset_obs_kpc": obs,
                "offset_sigma_kpc": sig,
                "residual_kpc": residual,
                "z_offset": z,
                "row_status": row_status,
                "source_mode": case.source_mode,
                "note": case.note,
            }
        )

    bullet_eval_rows = [
        row
        for row in rows_out
        if str(row["cluster_family"]).lower() == "bullet" and _safe_float(row.get("z_offset")) is not None
    ]
    bullet_count = len(bullet_eval_rows)
    bullet_chi2 = float(sum(float(row["z_offset"]) ** 2 for row in bullet_eval_rows))
    bullet_chi2_dof = float(bullet_chi2 / max(bullet_count, 1))
    bullet_max_abs_z = (
        float(max(abs(float(row["z_offset"])) for row in bullet_eval_rows))
        if bullet_eval_rows
        else float("nan")
    )

    non_bullet_missing_obs_n = int(
        sum(
            1
            for row in rows_out
            if str(row["cluster_family"]).lower() != "bullet"
            and (_safe_float(row.get("offset_obs_kpc")) is None or _safe_float(row.get("offset_sigma_kpc")) is None)
        )
    )

    checks = [
        {
            "check_id": "lpath_transfer::pi0_positive",
            "metric": "Pi0_median",
            "value": pi0,
            "expected": ">0",
            "gate_level": "hard",
            "hard_fail": (not math.isfinite(pi0)) or pi0 <= 0.0,
        },
        {
            "check_id": "lpath_transfer::source_lpath_status",
            "metric": "lpath_source_overall_status",
            "value": str(lpath_ref["lpath_source_status"]),
            "expected": "pass",
            "gate_level": "hard",
            "hard_fail": str(lpath_ref["lpath_source_status"]) != "pass",
        },
        {
            "check_id": "lpath_transfer::bullet_rows_available",
            "metric": "bullet_recheck_rows",
            "value": bullet_count,
            "expected": ">=2",
            "gate_level": "hard",
            "hard_fail": bullet_count < 2,
        },
        {
            "check_id": "lpath_transfer::bullet_max_abs_z",
            "metric": "bullet_recheck_max_abs_z",
            "value": bullet_max_abs_z,
            "expected": "<=3",
            "gate_level": "hard",
            "hard_fail": (not math.isfinite(bullet_max_abs_z)) or bullet_max_abs_z > 3.0,
        },
        {
            "check_id": "lpath_transfer::bullet_chi2_dof",
            "metric": "bullet_recheck_chi2_dof",
            "value": bullet_chi2_dof,
            "expected": "<=4",
            "gate_level": "hard",
            "hard_fail": (not math.isfinite(bullet_chi2_dof)) or bullet_chi2_dof > 4.0,
        },
        {
            "check_id": "lpath_transfer::nonbullet_observation_ready",
            "metric": "non_bullet_rows_missing_observed_offset",
            "value": non_bullet_missing_obs_n,
            "expected": "0",
            "gate_level": "watch",
            "hard_fail": False,
            "watch_fail": non_bullet_missing_obs_n > 0,
        },
    ]

    hard_reject_n = int(sum(1 for check in checks if bool(check.get("hard_fail"))))
    watch_fail_n = int(sum(1 for check in checks if bool(check.get("watch_fail"))))
    if hard_reject_n > 0:
        overall_status = "reject"
        decision = "cluster_collision_lpath_transfer_reject"
    elif watch_fail_n > 0:
        overall_status = "watch"
        decision = "cluster_collision_lpath_transfer_watch_missing_nonbullet_observed_offsets"
    else:
        overall_status = "pass"
        decision = "cluster_collision_lpath_transfer_pass"

    out_private_json = args.outdir / "cosmology_cluster_collision_lpath_transfer_template.json"
    out_private_csv = args.outdir / "cosmology_cluster_collision_lpath_transfer_template.csv"
    out_private_png = args.outdir / "cosmology_cluster_collision_lpath_transfer_template.png"
    out_private_template = args.outdir / "cluster_collision_lpath_transfer_input_template.csv"

    out_public_json = args.public_outdir / "cosmology_cluster_collision_lpath_transfer_template.json"
    out_public_csv = args.public_outdir / "cosmology_cluster_collision_lpath_transfer_template.csv"
    out_public_png = args.public_outdir / "cosmology_cluster_collision_lpath_transfer_template.png"
    out_public_template = args.public_outdir / "cluster_collision_lpath_transfer_input_template.csv"

    fieldnames = [
        "case_id",
        "label",
        "cluster_family",
        "branch_anchor",
        "rho_ratio",
        "v_ratio",
        "temp_ratio",
        "lpath_ratio",
        "tau_free_pred_gyr",
        "lpath_pred_kpc",
        "offset_ref_kpc",
        "offset_pred_kpc",
        "offset_obs_kpc",
        "offset_sigma_kpc",
        "residual_kpc",
        "z_offset",
        "row_status",
        "source_mode",
        "note",
    ]
    _write_csv(out_private_csv, rows_out, fieldnames)
    _write_csv(out_public_csv, rows_out, fieldnames)
    _render_png(out_private_png, rows_out, pi0=pi0, temp_power=temp_power)
    shutil.copy2(out_private_png, out_public_png)

    template_rows: List[Dict[str, Any]] = []
    for case in cases:
        template_rows.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "cluster_family": case.cluster_family,
                "branch_anchor": case.branch_anchor,
                "rho_ratio": case.rho_ratio,
                "v_ratio": case.v_ratio,
                "temp_ratio": case.temp_ratio,
                "offset_obs_kpc": case.offset_obs_kpc,
                "offset_sigma_kpc": case.offset_sigma_kpc,
                "source_mode": case.source_mode,
                "note": case.note,
            }
        )
    template_fields = [
        "case_id",
        "label",
        "cluster_family",
        "branch_anchor",
        "rho_ratio",
        "v_ratio",
        "temp_ratio",
        "offset_obs_kpc",
        "offset_sigma_kpc",
        "source_mode",
        "note",
    ]
    _write_csv(out_private_template, template_rows, template_fields)
    _write_csv(out_public_template, template_rows, template_fields)

    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "phase": {
            "phase": 8,
            "step": str(args.step_tag),
            "name": "cluster collision lpath transfer template",
        },
        "intent": (
            "Re-evaluate Bullet offsets with fixed L_path ratio law and lock reusable transfer template for "
            "non-Bullet collision clusters."
        ),
        "inputs": {
            "cases_csv": _rel(args.cases_csv),
            "bullet_derivation_json": _rel(args.bullet_derivation_json),
            "lpath_scaling_json": _rel(args.lpath_scaling_json),
        },
        "reference_state": {
            "pi0_median": pi0,
            "tau_free0_gyr": tau0,
            "lpath0_kpc": lpath0,
            "temp_power_collisional_branch": temp_power,
            "anchor_offset_reference_kpc": anchor_pred_map,
            "lpath_source_status": str(lpath_ref["lpath_source_status"]),
            "lpath_source_decision": str(lpath_ref["lpath_source_decision"]),
        },
        "equations": {
            "lpath_balance": "L_path=c_w/(v/L_corr + A_col rho T^{temp_power})",
            "lpath_ratio": "L_path/L_path,0=(1+Pi0)/(r_v + Pi0 r_rho r_T^{temp_power})",
            "offset_transfer": "Delta_x_pred(case)=Delta_x_ref(anchor) * (L_path/L_path,0)",
        },
        "cluster_rows": rows_out,
        "summary": {
            "bullet_recheck_rows": bullet_count,
            "bullet_recheck_chi2_dof": bullet_chi2_dof,
            "bullet_recheck_max_abs_z": bullet_max_abs_z,
            "non_bullet_rows_missing_observed_offset": non_bullet_missing_obs_n,
            "template_case_rows": len(rows_out),
        },
        "checks": checks,
        "decision": {
            "overall_status": overall_status,
            "decision": decision,
            "hard_reject_n": hard_reject_n,
            "watch_fail_n": watch_fail_n,
        },
        "falsification_gate": {
            "reject_if": [
                "Pi0 <= 0 or non-finite.",
                "L_path source status is not pass.",
                "Bullet recheck rows < 2.",
                "Bullet max abs z > 3.",
                "Bullet chi2/dof > 4.",
            ],
            "watch_if": [
                "Non-Bullet transfer rows do not yet have observed offsets/sigma.",
            ],
        },
        "outputs": {
            "private_json": _rel(out_private_json),
            "private_csv": _rel(out_private_csv),
            "private_png": _rel(out_private_png),
            "private_template_csv": _rel(out_private_template),
            "public_json": _rel(out_public_json),
            "public_csv": _rel(out_public_csv),
            "public_png": _rel(out_public_png),
            "public_template_csv": _rel(out_public_template),
        },
    }

    _write_json(out_private_json, payload)
    _write_json(out_public_json, payload)

    if worklog is not None:
        try:
            worklog.append_event(
                {
                    "topic": "cosmology",
                    "action": "cluster_collision_lpath_transfer_template",
                    "step": str(args.step_tag),
                    "status": overall_status,
                    "decision": decision,
                    "metrics": {
                        "bullet_recheck_chi2_dof": bullet_chi2_dof,
                        "bullet_recheck_max_abs_z": bullet_max_abs_z,
                        "non_bullet_rows_missing_observed_offset": non_bullet_missing_obs_n,
                    },
                    "outputs": {
                        "json": _rel(out_public_json),
                        "csv": _rel(out_public_csv),
                        "png": _rel(out_public_png),
                        "template_csv": _rel(out_public_template),
                    },
                }
            )
        except Exception:
            pass

    print(
        "[summary] status={0} decision={1} bullet_chi2/dof={2:.4f} bullet_max|z|={3:.4f} "
        "nonbullet_missing_obs={4}".format(
            overall_status,
            decision,
            bullet_chi2_dof,
            bullet_max_abs_z if math.isfinite(bullet_max_abs_z) else float("nan"),
            non_bullet_missing_obs_n,
        )
    )
    print(f"[out] {out_public_json}")
    print(f"[out] {out_public_csv}")
    print(f"[out] {out_public_png}")
    print(f"[out] {out_public_template}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
