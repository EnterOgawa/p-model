#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_cluster_collision_lpath_transfer_template.py

Step 8.7.25.25:
Bullet 系で固定した L_path 比スケーリング則を用いて、
Bullet 再評価（同一I/F）と非Bullet系（例: El Gordo）への
転用テンプレートを同時に固定する。

固定出力:
  - output/public/cosmology/cosmology_cluster_collision_lpath_transfer_template.json
  - output/public/cosmology/cosmology_cluster_collision_lpath_transfer_template.csv
  - output/public/cosmology/cosmology_cluster_collision_lpath_transfer_template.png
  - output/public/cosmology/cluster_collision_lpath_transfer_input_template.csv
  - output/public/cosmology/cluster_collision_nonbullet_primary_registration_template.csv
  - output/public/cosmology/cluster_collision_nonbullet_primary_registration_apply_report.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _file_signature(path: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"path": _rel(path), "exists": bool(path.exists())}
    if not path.exists():
        return payload
    try:
        stat = path.stat()
        payload["size_bytes"] = int(stat.st_size)
        payload["mtime_utc"] = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    except Exception:
        pass
    digest = _sha256_file(path)
    if digest is not None:
        payload["sha256"] = digest
    return payload


def _load_previous_watchpack(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    watchpack = diagnostics.get("primary_dataset_update_watchpack")
    if not isinstance(watchpack, dict):
        return {}
    return watchpack


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


def _load_primary_registration(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = str(row.get("case_id", "")).strip()
            if not case_id:
                continue
            out[case_id] = {
                "offset_obs_kpc": _parse_opt_float(row.get("offset_obs_kpc")),
                "offset_sigma_kpc": _parse_opt_float(row.get("offset_sigma_kpc")),
                "source_mode": str(row.get("source_mode", "primary_map_registered")).strip() or "primary_map_registered",
                "note": str(row.get("note", "")).strip(),
            }
    return out


def _bootstrap_primary_registration_csv(path: Path, cases: Sequence[TransferCase]) -> int:
    rows: List[Dict[str, Any]] = []
    for case in cases:
        if str(case.cluster_family).lower() == "bullet":
            continue
        rows.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "cluster_family": case.cluster_family,
                "offset_obs_kpc": "",
                "offset_sigma_kpc": "",
                "source_mode": "primary_map_pending_observation",
                "note": "fill offset_obs_kpc and offset_sigma_kpc from primary lensing/offset map",
            }
        )
    if not rows:
        return 0
    _write_csv(
        path,
        rows,
        [
            "case_id",
            "label",
            "cluster_family",
            "offset_obs_kpc",
            "offset_sigma_kpc",
            "source_mode",
            "note",
        ],
    )
    return len(rows)


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
    parser.add_argument(
        "--primary-registration-csv",
        type=Path,
        default=ROOT / "data" / "cosmology" / "bullet_cluster" / "collision_lpath_nonbullet_primary_registration.csv",
        help="Optional non-Bullet primary registration table (case_id, offset_obs_kpc, offset_sigma_kpc, source_mode, note).",
    )
    parser.add_argument(
        "--bootstrap-primary-registration",
        action="store_true",
        help="If --primary-registration-csv is missing, create a template file for non-Bullet rows and continue.",
    )
    parser.add_argument(
        "--apply-primary-registration",
        action="store_true",
        help="Apply values from --primary-registration-csv to non-Bullet rows before evaluation.",
    )
    parser.add_argument("--step-tag", default="8.7.25.25")
    args = parser.parse_args()

    out_private_json = args.outdir / "cosmology_cluster_collision_lpath_transfer_template.json"
    out_private_csv = args.outdir / "cosmology_cluster_collision_lpath_transfer_template.csv"
    out_private_png = args.outdir / "cosmology_cluster_collision_lpath_transfer_template.png"
    out_private_template = args.outdir / "cluster_collision_lpath_transfer_input_template.csv"
    out_private_checklist = args.outdir / "cluster_collision_nonbullet_primary_registration_checklist.csv"
    out_private_primary_template = args.outdir / "cluster_collision_nonbullet_primary_registration_template.csv"
    out_private_apply_report = args.outdir / "cluster_collision_nonbullet_primary_registration_apply_report.csv"

    out_public_json = args.public_outdir / "cosmology_cluster_collision_lpath_transfer_template.json"
    out_public_csv = args.public_outdir / "cosmology_cluster_collision_lpath_transfer_template.csv"
    out_public_png = args.public_outdir / "cosmology_cluster_collision_lpath_transfer_template.png"
    out_public_template = args.public_outdir / "cluster_collision_lpath_transfer_input_template.csv"
    out_public_checklist = args.public_outdir / "cluster_collision_nonbullet_primary_registration_checklist.csv"
    out_public_primary_template = args.public_outdir / "cluster_collision_nonbullet_primary_registration_template.csv"
    out_public_apply_report = args.public_outdir / "cluster_collision_nonbullet_primary_registration_apply_report.csv"

    previous_watchpack = _load_previous_watchpack(out_private_json)
    input_file_signature = _file_signature(args.cases_csv)

    cases_loaded = _load_cases(args.cases_csv)
    primary_registration_bootstrap_rows = 0
    primary_registration_bootstrapped = False
    if args.bootstrap_primary_registration and (not args.primary_registration_csv.exists()):
        primary_registration_bootstrap_rows = _bootstrap_primary_registration_csv(args.primary_registration_csv, cases_loaded)
        primary_registration_bootstrapped = primary_registration_bootstrap_rows > 0

    primary_registration_file_signature = _file_signature(args.primary_registration_csv)
    primary_registration_map = _load_primary_registration(args.primary_registration_csv)
    bullet_ref = _load_bullet_reference(args.bullet_derivation_json)
    lpath_ref = _load_lpath_reference(args.lpath_scaling_json)

    primary_registration_rows_loaded = int(len(primary_registration_map))
    primary_registration_rows_applied = 0
    primary_registration_rows_missing = 0
    primary_registration_rows_pending_values = 0
    primary_registration_rows_invalid = 0
    primary_registration_status_by_case: Dict[str, str] = {}
    primary_apply_report_rows: List[Dict[str, Any]] = []
    cases: List[TransferCase] = []

    for case in cases_loaded:
        is_non_bullet = str(case.cluster_family).lower() != "bullet"
        source_mode_before = str(case.source_mode).strip() or "template_pending"
        source_mode_after = source_mode_before
        before_obs = case.offset_obs_kpc
        before_sig = case.offset_sigma_kpc
        updated_obs = before_obs
        updated_sig = before_sig
        updated_note = case.note
        primary_status = "skipped_bullet"
        next_action = "none"

        if is_non_bullet:
            reg = primary_registration_map.get(case.case_id)
            if not args.apply_primary_registration:
                primary_status = "apply_not_requested"
            elif reg is None:
                primary_status = "missing_registration_row"
                primary_registration_rows_missing += 1
            else:
                reg_obs_raw = reg.get("offset_obs_kpc")
                reg_sig_raw = reg.get("offset_sigma_kpc")
                reg_obs_raw_txt = "" if reg_obs_raw is None else str(reg_obs_raw).strip()
                reg_sig_raw_txt = "" if reg_sig_raw is None else str(reg_sig_raw).strip()
                reg_obs = _parse_opt_float(reg.get("offset_obs_kpc"))
                reg_sig = _parse_opt_float(reg.get("offset_sigma_kpc"))
                reg_source_mode = str(reg.get("source_mode", "primary_map_registered")).strip() or "primary_map_registered"
                reg_note = str(reg.get("note", "")).strip()
                if reg_obs is not None and reg_sig is not None and reg_sig > 0.0:
                    updated_obs = float(reg_obs)
                    updated_sig = float(reg_sig)
                    source_mode_after = reg_source_mode
                    if reg_note:
                        updated_note = f"{updated_note} | registration_note:{reg_note}" if updated_note else f"registration_note:{reg_note}"
                    primary_status = "applied"
                    primary_registration_rows_applied += 1
                elif reg_obs_raw_txt == "" and reg_sig_raw_txt == "":
                    primary_status = "pending_registration_values"
                    primary_registration_rows_pending_values += 1
                else:
                    primary_status = "invalid_registration_row"
                    primary_registration_rows_invalid += 1

            if updated_obs is None or updated_sig is None or updated_sig <= 0.0:
                next_action = "register_offset_obs_and_sigma_from_primary_dataset"
            elif source_mode_after != "primary_map_registered":
                next_action = "set_source_mode_primary_map_registered_after_primary_map_link"

        cases.append(
            TransferCase(
                case_id=case.case_id,
                label=case.label,
                cluster_family=case.cluster_family,
                branch_anchor=case.branch_anchor,
                rho_ratio=case.rho_ratio,
                v_ratio=case.v_ratio,
                temp_ratio=case.temp_ratio,
                offset_obs_kpc=updated_obs,
                offset_sigma_kpc=updated_sig,
                source_mode=source_mode_after,
                note=updated_note,
            )
        )
        primary_registration_status_by_case[case.case_id] = primary_status
        if is_non_bullet:
            primary_apply_report_rows.append(
                {
                    "case_id": case.case_id,
                    "label": case.label,
                    "cluster_family": case.cluster_family,
                    "apply_primary_registration": bool(args.apply_primary_registration),
                    "primary_registration_csv_exists": bool(args.primary_registration_csv.exists()),
                    "source_mode_before": source_mode_before,
                    "source_mode_after": source_mode_after,
                    "offset_obs_before_kpc": before_obs,
                    "offset_sigma_before_kpc": before_sig,
                    "offset_obs_after_kpc": updated_obs,
                    "offset_sigma_after_kpc": updated_sig,
                    "primary_registration_status": primary_status,
                    "required_action": next_action if next_action != "none" else "none",
                }
            )

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

        is_non_bullet = str(case.cluster_family).lower() != "bullet"
        has_observed_offset = obs is not None and sig is not None and sig > 0.0
        source_mode_primary = str(case.source_mode).strip() == "primary_map_registered"
        primary_registration_status = str(primary_registration_status_by_case.get(case.case_id, "not_evaluated"))
        if is_non_bullet and not has_observed_offset:
            readiness_state = "missing_observed_offset"
            priority_class = "low_blocked_missing_primary_data"
            required_action = "register_offset_obs_and_sigma_from_primary_dataset"
        elif is_non_bullet and not source_mode_primary:
            readiness_state = "pending_primary_registration"
            priority_class = "low_blocked_missing_primary_data"
            required_action = "set_source_mode_primary_map_registered_after_primary_map_link"
        elif row_status == "pass":
            readiness_state = "ready"
            priority_class = "active"
            required_action = "none"
        else:
            readiness_state = row_status
            priority_class = "active"
            required_action = "review_row_status_and_input"

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
                "readiness_state": readiness_state,
                "priority_class": priority_class,
                "required_action": required_action,
                "primary_registration_status": primary_registration_status,
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
    non_bullet_missing_obs_case_ids = sorted(
        str(row["case_id"])
        for row in rows_out
        if str(row["cluster_family"]).lower() != "bullet"
        and (_safe_float(row.get("offset_obs_kpc")) is None or _safe_float(row.get("offset_sigma_kpc")) is None)
    )
    non_bullet_pending_source_mode_n = int(
        sum(
            1
            for row in rows_out
            if str(row["cluster_family"]).lower() != "bullet"
            and str(row.get("source_mode", "")).strip() != "primary_map_registered"
        )
    )
    non_bullet_pending_source_mode_case_ids = sorted(
        str(row["case_id"])
        for row in rows_out
        if str(row["cluster_family"]).lower() != "bullet"
        and str(row.get("source_mode", "")).strip() != "primary_map_registered"
    )
    non_bullet_blocked_case_ids = sorted(
        set(non_bullet_missing_obs_case_ids) | set(non_bullet_pending_source_mode_case_ids)
    )
    blocked_priority_class = "low_blocked_missing_primary_data" if non_bullet_blocked_case_ids else "active"
    blocked_reason = "missing_nonbullet_primary_offsets" if non_bullet_blocked_case_ids else "none"

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
            "expected": "pass or watch",
            "gate_level": "hard",
            "hard_fail": str(lpath_ref["lpath_source_status"]) not in {"pass", "watch"},
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
        {
            "check_id": "lpath_transfer::nonbullet_primary_source_mode_ready",
            "metric": "non_bullet_rows_not_primary_registered",
            "value": non_bullet_pending_source_mode_n,
            "expected": "0",
            "gate_level": "watch",
            "hard_fail": False,
            "watch_fail": non_bullet_pending_source_mode_n > 0,
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

    prev_signature = previous_watchpack.get("input_file_signature", {})
    prev_sha = (
        str(prev_signature.get("sha256"))
        if isinstance(prev_signature, dict) and prev_signature.get("sha256") is not None
        else None
    )
    prev_primary_signature = previous_watchpack.get("primary_registration_file_signature", {})
    prev_primary_sha = (
        str(prev_primary_signature.get("sha256"))
        if isinstance(prev_primary_signature, dict) and prev_primary_signature.get("sha256") is not None
        else None
    )
    current_sha = input_file_signature.get("sha256")
    current_primary_sha = primary_registration_file_signature.get("sha256")
    prev_blocked = previous_watchpack.get("nonbullet_blocked_case_ids", [])
    prev_blocked_set = {str(x) for x in prev_blocked} if isinstance(prev_blocked, list) else set()
    blocked_set = set(non_bullet_blocked_case_ids)
    prev_missing_obs = int(previous_watchpack.get("nonbullet_missing_obs_count", 0))
    prev_pending_source_mode = int(previous_watchpack.get("nonbullet_pending_source_mode_count", 0))
    prev_applied_rows = int(previous_watchpack.get("primary_registration_rows_applied", 0))
    prev_apply_requested = bool(previous_watchpack.get("apply_primary_registration", False))
    prev_event_counter = int(previous_watchpack.get("event_counter", 0))
    if not previous_watchpack:
        update_event_type = "bootstrap"
    elif prev_sha != current_sha:
        update_event_type = "input_hash_changed"
    elif prev_apply_requested != bool(args.apply_primary_registration):
        update_event_type = "apply_mode_changed"
    elif bool(args.apply_primary_registration) and prev_primary_sha != current_primary_sha:
        update_event_type = "primary_registration_hash_changed"
    elif prev_blocked_set != blocked_set:
        update_event_type = "blocked_case_set_changed"
    elif prev_missing_obs != non_bullet_missing_obs_n:
        update_event_type = "missing_obs_count_changed"
    elif prev_pending_source_mode != non_bullet_pending_source_mode_n:
        update_event_type = "source_mode_pending_count_changed"
    elif prev_applied_rows != primary_registration_rows_applied:
        update_event_type = "applied_rows_changed"
    else:
        update_event_type = "no_change"
    update_event_detected = update_event_type != "no_change"
    event_counter = prev_event_counter + (1 if update_event_detected else 0)
    next_action = (
        "register_nonbullet_primary_offsets_then_rerun_step_8_7_25"
        if non_bullet_blocked_case_ids
        else "wait_for_input_hash_changed_then_rerun_step_8_7_25"
    )
    primary_dataset_update_watchpack = {
        "update_event_type": update_event_type,
        "update_event_detected": bool(update_event_detected),
        "event_counter": event_counter,
        "input_file_signature": input_file_signature,
        "primary_registration_file_signature": primary_registration_file_signature,
        "apply_primary_registration": bool(args.apply_primary_registration),
        "primary_registration_rows_loaded": primary_registration_rows_loaded,
        "primary_registration_rows_applied": primary_registration_rows_applied,
        "primary_registration_rows_missing": primary_registration_rows_missing,
        "primary_registration_rows_pending_values": primary_registration_rows_pending_values,
        "primary_registration_rows_invalid": primary_registration_rows_invalid,
        "primary_registration_bootstrapped": bool(primary_registration_bootstrapped),
        "primary_registration_bootstrap_rows": int(primary_registration_bootstrap_rows),
        "nonbullet_missing_obs_count": non_bullet_missing_obs_n,
        "nonbullet_pending_source_mode_count": non_bullet_pending_source_mode_n,
        "nonbullet_missing_obs_case_ids": non_bullet_missing_obs_case_ids,
        "nonbullet_pending_source_mode_case_ids": non_bullet_pending_source_mode_case_ids,
        "nonbullet_blocked_case_ids": non_bullet_blocked_case_ids,
        "blocked_priority_class": blocked_priority_class,
        "blocked_reason": blocked_reason,
        "next_action": next_action,
    }
    non_bullet_checklist_rows: List[Dict[str, Any]] = [
        {
            "case_id": str(row["case_id"]),
            "label": str(row["label"]),
            "cluster_family": str(row["cluster_family"]),
            "source_mode": str(row.get("source_mode", "")),
            "offset_obs_kpc": row.get("offset_obs_kpc"),
            "offset_sigma_kpc": row.get("offset_sigma_kpc"),
            "readiness_state": str(row.get("readiness_state", "")),
            "priority_class": str(row.get("priority_class", "")),
            "required_action": str(row.get("required_action", "")),
            "next_action": next_action if str(row.get("case_id", "")) in non_bullet_blocked_case_ids else "none",
        }
        for row in rows_out
        if str(row["cluster_family"]).lower() != "bullet"
    ]

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
        "readiness_state",
        "priority_class",
        "required_action",
        "primary_registration_status",
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
                "readiness_state": (
                    "ready" if str(case.source_mode).strip() == "primary_map_registered" else "pending_primary_registration"
                ),
                "priority_class": (
                    "active" if str(case.source_mode).strip() == "primary_map_registered" else "low_blocked_missing_primary_data"
                ),
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
        "readiness_state",
        "priority_class",
        "source_mode",
        "note",
    ]
    _write_csv(out_private_template, template_rows, template_fields)
    _write_csv(out_public_template, template_rows, template_fields)

    primary_registration_template_rows: List[Dict[str, Any]] = []
    for row in rows_out:
        if str(row["cluster_family"]).lower() == "bullet":
            continue
        required_action = str(row.get("required_action", "none"))
        primary_registration_template_rows.append(
            {
                "case_id": row.get("case_id"),
                "label": row.get("label"),
                "cluster_family": row.get("cluster_family"),
                "offset_obs_kpc": row.get("offset_obs_kpc"),
                "offset_sigma_kpc": row.get("offset_sigma_kpc"),
                "source_mode": row.get("source_mode"),
                "target_source_mode": "primary_map_registered",
                "required_action": required_action,
                "note": row.get("note"),
            }
        )
    primary_registration_template_fields = [
        "case_id",
        "label",
        "cluster_family",
        "offset_obs_kpc",
        "offset_sigma_kpc",
        "source_mode",
        "target_source_mode",
        "required_action",
        "note",
    ]
    _write_csv(out_private_primary_template, primary_registration_template_rows, primary_registration_template_fields)
    _write_csv(out_public_primary_template, primary_registration_template_rows, primary_registration_template_fields)

    checklist_fields = [
        "case_id",
        "label",
        "cluster_family",
        "source_mode",
        "offset_obs_kpc",
        "offset_sigma_kpc",
        "readiness_state",
        "priority_class",
        "required_action",
        "next_action",
    ]
    _write_csv(out_private_checklist, non_bullet_checklist_rows, checklist_fields)
    _write_csv(out_public_checklist, non_bullet_checklist_rows, checklist_fields)

    apply_report_fields = [
        "case_id",
        "label",
        "cluster_family",
        "apply_primary_registration",
        "primary_registration_csv_exists",
        "source_mode_before",
        "source_mode_after",
        "offset_obs_before_kpc",
        "offset_sigma_before_kpc",
        "offset_obs_after_kpc",
        "offset_sigma_after_kpc",
        "primary_registration_status",
        "required_action",
    ]
    _write_csv(out_private_apply_report, primary_apply_report_rows, apply_report_fields)
    _write_csv(out_public_apply_report, primary_apply_report_rows, apply_report_fields)

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
            "primary_registration_csv": _rel(args.primary_registration_csv),
            "bootstrap_primary_registration": bool(args.bootstrap_primary_registration),
            "apply_primary_registration": bool(args.apply_primary_registration),
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
            "non_bullet_rows_not_primary_registered": non_bullet_pending_source_mode_n,
            "non_bullet_blocked_case_ids": non_bullet_blocked_case_ids,
            "blocked_priority_class": blocked_priority_class,
            "non_bullet_checklist_rows": len(non_bullet_checklist_rows),
            "primary_registration_template_rows": len(primary_registration_template_rows),
            "primary_registration_rows_loaded": primary_registration_rows_loaded,
            "primary_registration_rows_applied": primary_registration_rows_applied,
            "primary_registration_rows_missing": primary_registration_rows_missing,
            "primary_registration_rows_pending_values": primary_registration_rows_pending_values,
            "primary_registration_rows_invalid": primary_registration_rows_invalid,
            "primary_registration_bootstrapped": bool(primary_registration_bootstrapped),
            "primary_registration_bootstrap_rows": int(primary_registration_bootstrap_rows),
            "template_case_rows": len(rows_out),
        },
        "diagnostics": {
            "primary_dataset_update_watchpack": primary_dataset_update_watchpack,
            "non_bullet_primary_registration_checklist": {
                "n_rows": len(non_bullet_checklist_rows),
                "n_blocked_rows": len(non_bullet_blocked_case_ids),
                "blocked_case_ids": non_bullet_blocked_case_ids,
            },
            "non_bullet_primary_registration_apply_report": {
                "n_rows": len(primary_apply_report_rows),
                "n_applied": primary_registration_rows_applied,
                "n_missing_registration_rows": primary_registration_rows_missing,
                "n_pending_registration_rows": primary_registration_rows_pending_values,
                "n_invalid_registration_rows": primary_registration_rows_invalid,
                "bootstrapped": bool(primary_registration_bootstrapped),
                "bootstrap_rows": int(primary_registration_bootstrap_rows),
            },
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
            "private_nonbullet_checklist_csv": _rel(out_private_checklist),
            "private_nonbullet_primary_registration_template_csv": _rel(out_private_primary_template),
            "private_nonbullet_primary_registration_apply_report_csv": _rel(out_private_apply_report),
            "public_json": _rel(out_public_json),
            "public_csv": _rel(out_public_csv),
            "public_png": _rel(out_public_png),
            "public_template_csv": _rel(out_public_template),
            "public_nonbullet_checklist_csv": _rel(out_public_checklist),
            "public_nonbullet_primary_registration_template_csv": _rel(out_public_primary_template),
            "public_nonbullet_primary_registration_apply_report_csv": _rel(out_public_apply_report),
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
                        "non_bullet_rows_not_primary_registered": non_bullet_pending_source_mode_n,
                        "non_bullet_blocked_rows": len(non_bullet_blocked_case_ids),
                        "primary_registration_rows_loaded": primary_registration_rows_loaded,
                        "primary_registration_rows_applied": primary_registration_rows_applied,
                        "primary_registration_rows_missing": primary_registration_rows_missing,
                        "primary_registration_rows_pending_values": primary_registration_rows_pending_values,
                        "primary_registration_rows_invalid": primary_registration_rows_invalid,
                        "primary_registration_bootstrapped": bool(primary_registration_bootstrapped),
                        "primary_registration_bootstrap_rows": int(primary_registration_bootstrap_rows),
                        "update_event_type": update_event_type,
                        "update_event_detected": bool(update_event_detected),
                        "event_counter": event_counter,
                    },
                    "outputs": {
                        "json": _rel(out_public_json),
                        "csv": _rel(out_public_csv),
                        "png": _rel(out_public_png),
                        "template_csv": _rel(out_public_template),
                        "nonbullet_checklist_csv": _rel(out_public_checklist),
                        "nonbullet_primary_registration_template_csv": _rel(out_public_primary_template),
                        "nonbullet_primary_registration_apply_report_csv": _rel(out_public_apply_report),
                    },
                }
            )
        except Exception:
            pass

    print(
        "[summary] status={0} decision={1} bullet_chi2/dof={2:.4f} bullet_max|z|={3:.4f} "
        "nonbullet_missing_obs={4} nonbullet_pending_source_mode={5} nonbullet_blocked_rows={6} "
        "reg_loaded={7} reg_applied={8} reg_missing={9} reg_pending={10} reg_invalid={11} "
        "reg_bootstrapped={12} update_event={13}".format(
            overall_status,
            decision,
            bullet_chi2_dof,
            bullet_max_abs_z if math.isfinite(bullet_max_abs_z) else float("nan"),
            non_bullet_missing_obs_n,
            non_bullet_pending_source_mode_n,
            len(non_bullet_blocked_case_ids),
            primary_registration_rows_loaded,
            primary_registration_rows_applied,
            primary_registration_rows_missing,
            primary_registration_rows_pending_values,
            primary_registration_rows_invalid,
            bool(primary_registration_bootstrapped),
            update_event_type,
        )
    )
    print(f"[out] {out_public_json}")
    print(f"[out] {out_public_csv}")
    print(f"[out] {out_public_png}")
    print(f"[out] {out_public_template}")
    print(f"[out] {out_public_checklist}")
    print(f"[out] {out_public_primary_template}")
    print(f"[out] {out_public_apply_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
