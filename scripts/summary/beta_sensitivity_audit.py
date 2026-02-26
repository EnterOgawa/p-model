#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
beta_sensitivity_audit.py

Phase 8 / Step 8.7.24:
β感度解析（Sensitivity Audit）の初版を固定出力する。

目的:
- β_low=1.000000 / β_high=1.000022 と凍結β（ref）で、
  Part II/III の主要指標がどの程度変動するかを同一I/Fで比較する。
- 既存固定出力（EHT/核/量子接続）に加え、CMB / SPARC / Bell bridge の
  cross-check 指標も heavy-rerun で横断監査し、差分の pass/watch/reject を機械判定する。

出力:
- output/public/summary/beta_sensitivity_audit.json
- output/public/summary/beta_sensitivity_audit.csv
- output/public/summary/beta_sensitivity_audit.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。
def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `_as_float` の入出力契約と処理意図を定義する。

def _as_float(value: Any) -> Optional[float]:
    # 条件分岐: `isinstance(value, (int, float))` を満たす経路を評価する。
    if isinstance(value, (int, float)):
        number = float(value)
        # 条件分岐: `math.isfinite(number)` を満たす経路を評価する。
        if math.isfinite(number):
            return number

    return None


# 関数: `_classify_numeric` の入出力契約と処理意図を定義する。

def _classify_numeric(*, max_abs_delta: float, pass_abs_threshold: float, watch_abs_threshold: float) -> str:
    # 条件分岐: `max_abs_delta <= pass_abs_threshold` を満たす経路を評価する。
    if max_abs_delta <= pass_abs_threshold:
        return "pass"

    # 条件分岐: `max_abs_delta <= watch_abs_threshold` を満たす経路を評価する。

    if max_abs_delta <= watch_abs_threshold:
        return "watch"

    return "reject"


# 関数: `_status_severity` の入出力契約と処理意図を定義する。

def _status_severity(status: str) -> int:
    # 条件分岐: `status == "pass"` を満たす経路を評価する。
    if status == "pass":
        return 0

    # 条件分岐: `status == "watch"` を満たす経路を評価する。

    if status == "watch":
        return 1

    return 2


# 関数: `_categorical_triplet_status` の入出力契約と処理意図を定義する。

def _categorical_triplet_status(*, case_low: str, case_ref: str, case_high: str) -> str:
    # 条件分岐: `case_low == case_ref == case_high` を満たす経路を評価する。
    if case_low == case_ref == case_high:
        return "pass"

    # 条件分岐: `case_ref in (case_low, case_high)` を満たす経路を評価する。

    if case_ref in (case_low, case_high):
        return "watch"

    return "reject"


# 関数: `_invariance_thresholds` の入出力契約と処理意図を定義する。

def _invariance_thresholds(ref_value: float) -> Dict[str, float]:
    scale = max(1.0, abs(float(ref_value)))
    return {
        "pass_abs_threshold": float(1e-6 * scale),
        "watch_abs_threshold": float(1e-4 * scale),
    }


# 関数: `_load_frozen_beta` の入出力契約と処理意図を定義する。

def _load_frozen_beta() -> Dict[str, Any]:
    candidates = [
        ROOT / "output" / "private" / "theory" / "frozen_parameters.json",
        ROOT / "output" / "public" / "theory" / "frozen_parameters.json",
    ]
    for path in candidates:
        # 条件分岐: `not path.exists()` を満たす経路を評価する。
        if not path.exists():
            continue

        payload = _read_json(path)
        beta = _as_float(payload.get("beta"))
        # 条件分岐: `beta is None` を満たす経路を評価する。
        if beta is None:
            continue

        gamma_sigma = _as_float(payload.get("gamma_pmodel_sigma"))
        beta_sigma = _as_float(payload.get("beta_sigma"))
        # 条件分岐: `gamma_sigma is None and beta_sigma is not None` を満たす経路を評価する。
        if gamma_sigma is None and beta_sigma is not None:
            gamma_sigma = float(2.0 * beta_sigma)

        return {
            "path": path,
            "beta_ref": float(beta),
            "beta_sigma": beta_sigma,
            "gamma_sigma": gamma_sigma,
        }

    raise SystemExit("[fail] frozen beta not found in output/private/theory or output/public/theory.")


# 関数: `_run_checked` の入出力契約と処理意図を定義する。

def _run_checked(command: List[str]) -> None:
    proc = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
    # 条件分岐: `proc.returncode != 0` を満たす経路を評価する。
    if proc.returncode != 0:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(command)
            + "\n--- stdout ---\n"
            + (proc.stdout or "")
            + "\n--- stderr ---\n"
            + (proc.stderr or "")
        )


# 関数: `_extract_eht_delta_from_output` の入出力契約と処理意図を定義する。

def _extract_eht_delta_from_output() -> float:
    candidates = [
        ROOT / "output" / "private" / "eht" / "eht_shadow_compare.json",
        ROOT / "output" / "public" / "eht" / "eht_shadow_compare.json",
    ]
    for path in candidates:
        # 条件分岐: `not path.exists()` を満たす経路を評価する。
        if not path.exists():
            continue

        payload = _read_json(path)
        delta_ref = _as_float((payload.get("delta_reference") or {}).get("delta_coeff_p_minus_gr_schwarzschild"))
        # 条件分岐: `delta_ref is None` を満たす経路を評価する。
        if delta_ref is None:
            ratio = _as_float((payload.get("phase4") or {}).get("shadow_diameter_coeff_ratio_p_over_gr"))
            # 条件分岐: `ratio is not None` を満たす経路を評価する。
            if ratio is not None:
                delta_ref = float(ratio - 1.0)

        # 条件分岐: `delta_ref is not None` を満たす経路を評価する。

        if delta_ref is not None:
            return float(delta_ref)

    raise RuntimeError("failed to extract EHT delta from eht_shadow_compare outputs")


# 関数: `_extract_deuteron_b_from_output` の入出力契約と処理意図を定義する。

def _extract_deuteron_b_from_output() -> float:
    path = ROOT / "output" / "public" / "quantum" / "nuclear_binding_deuteron_metrics.json"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        raise RuntimeError(f"missing deuteron metrics: {path}")

    payload = _read_json(path)
    b = _as_float((((payload.get("derived") or {}).get("binding_energy") or {}).get("B_MeV") or {}).get("value"))
    # 条件分岐: `b is None` を満たす経路を評価する。
    if b is None:
        raise RuntimeError("failed to extract deuteron B_MeV value")

    return float(b)


# 関数: `_extract_cmb_metrics_from_output` の入出力契約と処理意図を定義する。

def _extract_cmb_metrics_from_output() -> Dict[str, Any]:
    candidates = [
        ROOT / "output" / "public" / "cosmology" / "cosmology_cmb_acoustic_peak_reconstruction_metrics.json",
        ROOT / "output" / "private" / "cosmology" / "cosmology_cmb_acoustic_peak_reconstruction_metrics.json",
    ]
    for path in candidates:
        # 条件分岐: `not path.exists()` を満たす経路を評価する。
        if not path.exists():
            continue

        payload = _read_json(path)
        gate = payload.get("gate") if isinstance(payload.get("gate"), dict) else {}
        extended = gate.get("extended_4to6") if isinstance(gate.get("extended_4to6"), dict) else {}
        overall_ext = gate.get("overall_extended") if isinstance(gate.get("overall_extended"), dict) else {}
        max_abs_delta_ell = _as_float(extended.get("max_abs_delta_ell"))
        max_abs_delta_amp_rel = _as_float(extended.get("max_abs_delta_amp_rel"))
        overall_status = str(overall_ext.get("status") or "unknown")
        # 条件分岐: `max_abs_delta_ell is None or max_abs_delta_amp_rel is None` を満たす経路を評価する。
        if max_abs_delta_ell is None or max_abs_delta_amp_rel is None:
            continue

        return {
            "path": path,
            "holdout46_max_abs_delta_ell": float(max_abs_delta_ell),
            "holdout46_max_abs_delta_amp_rel": float(max_abs_delta_amp_rel),
            "overall_extended_status": overall_status,
        }

    raise RuntimeError("failed to extract CMB acoustic holdout metrics from output")


# 関数: `_extract_sparc_metrics_from_output` の入出力契約と処理意図を定義する。

def _extract_sparc_metrics_from_output() -> Dict[str, Any]:
    path = ROOT / "output" / "public" / "cosmology" / "sparc_rotation_curve_pmodel_audit_metrics.json"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        raise RuntimeError(f"missing SPARC metrics: {path}")

    payload = _read_json(path)
    fit_results = payload.get("fit_results") if isinstance(payload.get("fit_results"), dict) else {}
    pmodel = fit_results.get("pmodel_corrected") if isinstance(fit_results.get("pmodel_corrected"), dict) else {}
    comparison = fit_results.get("comparison") if isinstance(fit_results.get("comparison"), dict) else {}
    chi2_dof_pmodel = _as_float(pmodel.get("chi2_dof"))
    delta_chi2 = _as_float(comparison.get("delta_chi2_baryon_minus_pmodel"))
    better_model = str(comparison.get("better_model_by_chi2") or "unknown")
    # 条件分岐: `chi2_dof_pmodel is None or delta_chi2 is None` を満たす経路を評価する。
    if chi2_dof_pmodel is None or delta_chi2 is None:
        raise RuntimeError("failed to extract SPARC chi2 metrics")

    return {
        "path": path,
        "chi2_dof_pmodel": float(chi2_dof_pmodel),
        "delta_chi2_baryon_minus_pmodel": float(delta_chi2),
        "better_model_by_chi2": better_model,
    }


# 関数: `_extract_bell_bridge_metrics_from_output` の入出力契約と処理意図を定義する。

def _extract_bell_bridge_metrics_from_output() -> Dict[str, Any]:
    path = ROOT / "output" / "public" / "quantum" / "quantum_connection_bridge_table.json"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        raise RuntimeError(f"missing bell bridge metrics: {path}")

    payload = _read_json(path)
    overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
    selection = overall.get("selection_summary") if isinstance(overall.get("selection_summary"), dict) else {}
    physics = overall.get("physics_summary") if isinstance(overall.get("physics_summary"), dict) else {}
    selection_max = _as_float(selection.get("normalized_score_max"))
    physics_max = _as_float(physics.get("normalized_score_max"))
    overall_status = str(overall.get("status") or "unknown")
    # 条件分岐: `selection_max is None or physics_max is None` を満たす経路を評価する。
    if selection_max is None or physics_max is None:
        raise RuntimeError("failed to extract bell bridge normalized-score metrics")

    return {
        "path": path,
        "overall_status": overall_status,
        "selection_normalized_score_max": float(selection_max),
        "physics_normalized_score_max": float(physics_max),
    }


# 関数: `_set_beta_in_frozen_payload` の入出力契約と処理意図を定義する。

def _set_beta_in_frozen_payload(payload: Dict[str, Any], beta: float) -> Dict[str, Any]:
    gamma = float(2.0 * beta - 1.0)
    out = json.loads(json.dumps(payload, ensure_ascii=False))
    out["generated_utc"] = _iso_utc_now()
    out["beta"] = float(beta)
    out["gamma_pmodel"] = gamma

    constraints = out.get("constraints")
    # 条件分岐: `isinstance(constraints, list)` を満たす経路を評価する。
    if isinstance(constraints, list):
        for row in constraints:
            # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
            if not isinstance(row, dict):
                continue

            # 条件分岐: `"beta" in row` を満たす経路を評価する。

            if "beta" in row:
                row["beta"] = float(beta)

            # 条件分岐: `"gamma" in row` を満たす経路を評価する。

            if "gamma" in row:
                row["gamma"] = gamma

    return out


# 関数: `_heavy_rerun_cases` の入出力契約と処理意図を定義する。

def _heavy_rerun_cases(
    *,
    frozen_path: Path,
    beta_cases: Dict[str, float],
) -> Dict[str, Any]:
    original_text = frozen_path.read_text(encoding="utf-8")
    original_payload = json.loads(original_text)

    out: Dict[str, Any] = {
        "mode": "eht+nuclear+cmb+sparc+bell_bridge_rerun_under_beta_override",
        "frozen_parameters_json": _rel(frozen_path),
        "scripts": {
            "eht_shadow_compare": "python -B scripts/eht/eht_shadow_compare.py",
            "nuclear_binding_deuteron": "python -B scripts/quantum/nuclear_binding_deuteron.py",
            "cmb_acoustic_peak_reconstruction": "python -B scripts/cosmology/cosmology_cmb_acoustic_peak_reconstruction.py",
            "sparc_rotation_curve_pmodel_audit": "python -B scripts/cosmology/sparc_rotation_curve_pmodel_audit.py",
            "quantum_connection_bridge_table": "python -B scripts/quantum/quantum_connection_bridge_table.py",
        },
        "cases": {},
        "restore": {"restored": False, "rerun_ref_done": False},
    }

    try:
        for case_name, beta in beta_cases.items():
            payload_case = _set_beta_in_frozen_payload(original_payload, float(beta))
            _write_json(frozen_path, payload_case)

            _run_checked([sys.executable, "-B", str(ROOT / "scripts" / "eht" / "eht_shadow_compare.py")])
            _run_checked([sys.executable, "-B", str(ROOT / "scripts" / "quantum" / "nuclear_binding_deuteron.py")])
            _run_checked(
                [sys.executable, "-B", str(ROOT / "scripts" / "cosmology" / "cosmology_cmb_acoustic_peak_reconstruction.py")]
            )
            _run_checked(
                [sys.executable, "-B", str(ROOT / "scripts" / "cosmology" / "sparc_rotation_curve_pmodel_audit.py")]
            )
            _run_checked([sys.executable, "-B", str(ROOT / "scripts" / "quantum" / "quantum_connection_bridge_table.py")])

            eht_delta = _extract_eht_delta_from_output()
            deuteron_b = _extract_deuteron_b_from_output()
            cmb_metrics = _extract_cmb_metrics_from_output()
            sparc_metrics = _extract_sparc_metrics_from_output()
            bridge_metrics = _extract_bell_bridge_metrics_from_output()
            out["cases"][case_name] = {
                "beta": float(beta),
                "eht_delta_coeff_p_minus_gr": float(eht_delta),
                "deuteron_binding_B_MeV": float(deuteron_b),
                "cmb_holdout46_max_abs_delta_ell": float(cmb_metrics["holdout46_max_abs_delta_ell"]),
                "cmb_holdout46_max_abs_delta_amp_rel": float(cmb_metrics["holdout46_max_abs_delta_amp_rel"]),
                "cmb_overall_extended_status": str(cmb_metrics["overall_extended_status"]),
                "sparc_chi2_dof_pmodel": float(sparc_metrics["chi2_dof_pmodel"]),
                "sparc_delta_chi2_baryon_minus_pmodel": float(sparc_metrics["delta_chi2_baryon_minus_pmodel"]),
                "sparc_better_model_by_chi2": str(sparc_metrics["better_model_by_chi2"]),
                "bell_bridge_overall_status": str(bridge_metrics["overall_status"]),
                "bell_bridge_selection_normalized_score_max": float(bridge_metrics["selection_normalized_score_max"]),
                "bell_bridge_physics_normalized_score_max": float(bridge_metrics["physics_normalized_score_max"]),
            }
    finally:
        frozen_path.write_text(original_text, encoding="utf-8")
        out["restore"]["restored"] = True
        try:
            _run_checked([sys.executable, "-B", str(ROOT / "scripts" / "eht" / "eht_shadow_compare.py")])
            _run_checked([sys.executable, "-B", str(ROOT / "scripts" / "quantum" / "nuclear_binding_deuteron.py")])
            _run_checked(
                [sys.executable, "-B", str(ROOT / "scripts" / "cosmology" / "cosmology_cmb_acoustic_peak_reconstruction.py")]
            )
            _run_checked(
                [sys.executable, "-B", str(ROOT / "scripts" / "cosmology" / "sparc_rotation_curve_pmodel_audit.py")]
            )
            _run_checked([sys.executable, "-B", str(ROOT / "scripts" / "quantum" / "quantum_connection_bridge_table.py")])
            out["restore"]["rerun_ref_done"] = True
        except Exception:
            out["restore"]["rerun_ref_done"] = False

    return out


# 関数: `_full_rerun_followup` の入出力契約と処理意図を定義する。

def _full_rerun_followup(
    *,
    frozen_path: Path,
    beta_cases: Dict[str, float],
) -> Dict[str, Any]:
    run1 = _heavy_rerun_cases(frozen_path=frozen_path, beta_cases=beta_cases)
    run2 = _heavy_rerun_cases(frozen_path=frozen_path, beta_cases=beta_cases)

    numeric_keys = [
        "eht_delta_coeff_p_minus_gr",
        "deuteron_binding_B_MeV",
        "cmb_holdout46_max_abs_delta_ell",
        "cmb_holdout46_max_abs_delta_amp_rel",
        "sparc_chi2_dof_pmodel",
        "sparc_delta_chi2_baryon_minus_pmodel",
        "bell_bridge_selection_normalized_score_max",
        "bell_bridge_physics_normalized_score_max",
    ]
    categorical_keys = [
        "cmb_overall_extended_status",
        "sparc_better_model_by_chi2",
        "bell_bridge_overall_status",
    ]

    run1_cases = run1.get("cases") if isinstance(run1.get("cases"), dict) else {}
    run2_cases = run2.get("cases") if isinstance(run2.get("cases"), dict) else {}
    case_names = sorted(set(run1_cases.keys()) & set(run2_cases.keys()))

    numeric_max_abs_delta = 0.0
    numeric_per_metric_max: Dict[str, float] = {}
    numeric_mismatch_rows: List[Dict[str, Any]] = []
    categorical_mismatch_rows: List[Dict[str, Any]] = []

    for case_name in case_names:
        row1 = run1_cases.get(case_name)
        row2 = run2_cases.get(case_name)
        # 条件分岐: `not isinstance(row1, dict) or not isinstance(row2, dict)` を満たす経路を評価する。
        if not isinstance(row1, dict) or not isinstance(row2, dict):
            continue

        for key in numeric_keys:
            value1 = _as_float(row1.get(key))
            value2 = _as_float(row2.get(key))
            # 条件分岐: `value1 is None or value2 is None` を満たす経路を評価する。
            if value1 is None or value2 is None:
                continue

            delta = abs(float(value2) - float(value1))
            metric_key = f"{case_name}::{key}"
            numeric_per_metric_max[metric_key] = float(delta)
            # 条件分岐: `delta > numeric_max_abs_delta` を満たす経路を評価する。
            if delta > numeric_max_abs_delta:
                numeric_max_abs_delta = float(delta)

            # 条件分岐: `delta > 0.0` を満たす経路を評価する。

            if delta > 0.0:
                numeric_mismatch_rows.append(
                    {
                        "case": case_name,
                        "metric": key,
                        "run1": float(value1),
                        "run2": float(value2),
                        "abs_delta": float(delta),
                    }
                )

        for key in categorical_keys:
            value1 = str(row1.get(key) or "")
            value2 = str(row2.get(key) or "")
            # 条件分岐: `value1 != value2` を満たす経路を評価する。
            if value1 != value2:
                categorical_mismatch_rows.append(
                    {
                        "case": case_name,
                        "metric": key,
                        "run1": value1,
                        "run2": value2,
                    }
                )

    pass_abs_threshold = 1.0e-12
    watch_abs_threshold = 1.0e-9
    numeric_status = _classify_numeric(
        max_abs_delta=numeric_max_abs_delta,
        pass_abs_threshold=pass_abs_threshold,
        watch_abs_threshold=watch_abs_threshold,
    )
    categorical_status = "pass" if not categorical_mismatch_rows else "reject"
    restore_ok = bool(
        (run1.get("restore") or {}).get("restored")
        and (run1.get("restore") or {}).get("rerun_ref_done")
        and (run2.get("restore") or {}).get("restored")
        and (run2.get("restore") or {}).get("rerun_ref_done")
    )
    overall_severity = max(
        _status_severity(numeric_status),
        _status_severity(categorical_status),
        0 if restore_ok else 2,
    )
    overall_status = "pass" if overall_severity == 0 else ("watch" if overall_severity == 1 else "reject")

    return {
        "mode": "full_rerun_followup",
        "runs": {
            "run1": run1,
            "run2": run2,
        },
        "comparison": {
            "numeric": {
                "keys": numeric_keys,
                "case_count": len(case_names),
                "max_abs_delta": float(numeric_max_abs_delta),
                "pass_abs_threshold": pass_abs_threshold,
                "watch_abs_threshold": watch_abs_threshold,
                "status": numeric_status,
                "nonzero_mismatch_rows": numeric_mismatch_rows,
                "per_metric_max_abs_delta": numeric_per_metric_max,
            },
            "categorical": {
                "keys": categorical_keys,
                "status": categorical_status,
                "mismatch_rows": categorical_mismatch_rows,
            },
            "restore": {
                "status": "pass" if restore_ok else "reject",
                "run1": run1.get("restore"),
                "run2": run2.get("restore"),
            },
        },
        "overall": {
            "status": overall_status,
            "severity": overall_severity,
            "rule": "pass when numeric/categorical/restore checks all pass; reject on restore failure or categorical mismatch.",
        },
    }


# 関数: `_load_eht_metrics` の入出力契約と処理意図を定義する。

def _load_eht_metrics() -> Dict[str, Any]:
    candidates = [
        ROOT / "output" / "public" / "eht" / "eht_shadow_compare.json",
        ROOT / "output" / "private" / "eht" / "eht_shadow_compare.json",
    ]
    for path in candidates:
        # 条件分岐: `path.exists()` を満たす経路を評価する。
        if path.exists():
            payload = _read_json(path)
            coeff_gr = _as_float((payload.get("reference_gr") or {}).get("shadow_diameter_coeff_rg"))
            delta_sigma_required = _as_float((payload.get("delta_reference") or {}).get("delta_sigma_required_3sigma"))
            delta_ref = _as_float((payload.get("delta_reference") or {}).get("delta_coeff_p_minus_gr_schwarzschild"))
            # 条件分岐: `coeff_gr is None or delta_sigma_required is None` を満たす経路を評価する。
            if coeff_gr is None or delta_sigma_required is None:
                continue

            # 条件分岐: `delta_ref is None` を満たす経路を評価する。

            if delta_ref is None:
                delta_ref = _as_float((payload.get("phase4") or {}).get("shadow_diameter_coeff_ratio_p_over_gr"))
                # 条件分岐: `delta_ref is not None` を満たす経路を評価する。
                if delta_ref is not None:
                    delta_ref = float(delta_ref - 1.0)

            return {
                "path": path,
                "coeff_gr": float(coeff_gr),
                "delta_sigma_required_3sigma": float(delta_sigma_required),
                "delta_ref": float(delta_ref) if delta_ref is not None else None,
            }

    raise SystemExit("[fail] eht_shadow_compare.json not found (public/private).")


# 関数: `_load_deuteron_metrics` の入出力契約と処理意図を定義する。

def _load_deuteron_metrics() -> Dict[str, Any]:
    path = ROOT / "output" / "public" / "quantum" / "nuclear_binding_deuteron_metrics.json"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        raise SystemExit(f"[fail] missing: {path}")

    payload = _read_json(path)
    b_mev = _as_float((((payload.get("derived") or {}).get("binding_energy") or {}).get("B_MeV") or {}).get("value"))
    b_sigma = _as_float((((payload.get("derived") or {}).get("binding_energy") or {}).get("B_MeV") or {}).get("sigma"))
    # 条件分岐: `b_mev is None or b_sigma is None` を満たす経路を評価する。
    if b_mev is None or b_sigma is None:
        raise SystemExit("[fail] invalid deuteron metrics JSON (B_MeV value/sigma missing).")

    return {"path": path, "b_mev": float(b_mev), "b_sigma": float(b_sigma)}


# 関数: `_load_cmb_metrics` の入出力契約と処理意図を定義する。

def _load_cmb_metrics() -> Dict[str, Any]:
    try:
        return _extract_cmb_metrics_from_output()
    except Exception as exc:
        raise SystemExit(f"[fail] {exc}") from exc


# 関数: `_load_sparc_metrics` の入出力契約と処理意図を定義する。

def _load_sparc_metrics() -> Dict[str, Any]:
    try:
        return _extract_sparc_metrics_from_output()
    except Exception as exc:
        raise SystemExit(f"[fail] {exc}") from exc


# 関数: `_load_bell_bridge_metrics` の入出力契約と処理意図を定義する。

def _load_bell_bridge_metrics() -> Dict[str, Any]:
    try:
        return _extract_bell_bridge_metrics_from_output()
    except Exception as exc:
        raise SystemExit(f"[fail] {exc}") from exc


# 関数: `_load_quantum_shared_kpi` の入出力契約と処理意図を定義する。

def _load_quantum_shared_kpi() -> Dict[str, Any]:
    path = ROOT / "output" / "public" / "quantum" / "quantum_connection_shared_kpi.json"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        raise SystemExit(f"[fail] missing: {path}")

    payload = _read_json(path)
    overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
    channels = payload.get("channels") if isinstance(payload.get("channels"), list) else []
    channel_status = {}
    for row in channels:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        key = str(row.get("channel") or "")
        # 条件分岐: `key` を満たす経路を評価する。
        if key:
            channel_status[key] = str(row.get("status") or "")

    return {
        "path": path,
        "overall_status": str(overall.get("status") or "unknown"),
        "overall_severity": int(overall.get("severity") or 99),
        "channel_status": channel_status,
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `not rows` を満たす経路を評価する。
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# 関数: `_plot_summary` の入出力契約と処理意図を定義する。

def _plot_summary(
    *,
    rows_numeric: List[Dict[str, Any]],
    rows_categorical: List[Dict[str, Any]],
    out_png: Path,
    beta_low: float,
    beta_ref: float,
    beta_high: float,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(row["indicator"]) for row in rows_numeric]
    ratios = []
    colors = []
    for row in rows_numeric:
        max_abs_delta = float(row["max_abs_delta_vs_ref"])
        pass_thr = float(row["pass_abs_threshold"])
        ratio = max_abs_delta / pass_thr if pass_thr > 0 else float("nan")
        ratios.append(ratio)
        status = str(row["status"])
        colors.append("#2ca02c" if status == "pass" else ("#ffbf00" if status == "watch" else "#d62728"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=150)
    ax0, ax1 = axes

    y = np.arange(len(labels), dtype=float)
    ax0.barh(y, ratios, color=colors, alpha=0.9)
    ax0.axvline(1.0, color="#555555", ls="--", lw=1.2, label="pass threshold")
    ax0.set_yticks(y, labels)
    ax0.set_xlabel("max |Δ(metric)| / pass threshold")
    ax0.set_title("β sensitivity (numeric indicators)")
    ax0.grid(True, axis="x", ls=":", alpha=0.5)
    ax0.legend(loc="lower right", frameon=True, fontsize=8)

    ax1.axis("off")
    lines = [
        "β sensitivity audit (Step 8.7.24)",
        f"β_low={beta_low:.9f}",
        f"β_ref={beta_ref:.9f}",
        f"β_high={beta_high:.9f}",
        "",
        "[categorical indicators]",
    ]
    for row in rows_categorical:
        lines.append(f"- {row['indicator']}: {row['status']} ({row['case_low']}/{row['case_ref']}/{row['case_high']})")

    ax1.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    fig.suptitle("Phase 8 / Step 8.7.24: beta sensitivity audit (Part II/III key indicators)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    ap = argparse.ArgumentParser(description="Beta sensitivity audit for Part II/III key indicators.")
    ap.add_argument("--beta-low", type=float, default=1.000000)
    ap.add_argument("--beta-high", type=float, default=1.000022)
    ap.add_argument(
        "--heavy-rerun",
        action="store_true",
        help=(
            "Re-run EHT/deuteron plus CMB/SPARC/Bell bridge scripts under temporary beta overrides "
            "(restores frozen parameters after run)."
        ),
    )
    ap.add_argument(
        "--full-rerun-followup",
        action="store_true",
        help=(
            "Run heavy rerun twice and add reproducibility comparison "
            "(run2 vs run1, including restore-state checks)."
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "output" / "public" / "summary"),
        help="Output directory (default: output/public/summary)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen = _load_frozen_beta()
    eht = _load_eht_metrics()
    deuteron = _load_deuteron_metrics()
    cmb = _load_cmb_metrics()
    sparc = _load_sparc_metrics()
    bell_bridge = _load_bell_bridge_metrics()
    qkpi = _load_quantum_shared_kpi()

    beta_ref = float(frozen["beta_ref"])
    beta_low = float(args.beta_low)
    beta_high = float(args.beta_high)
    cases = {"beta_low": beta_low, "beta_ref": beta_ref, "beta_high": beta_high}
    heavy: Optional[Dict[str, Any]] = None
    heavy_eht_values: Optional[Dict[str, float]] = None
    heavy_deuteron_values: Optional[Dict[str, float]] = None
    heavy_cmb_dell_values: Optional[Dict[str, float]] = None
    heavy_cmb_damp_values: Optional[Dict[str, float]] = None
    heavy_cmb_status_values: Optional[Dict[str, str]] = None
    heavy_sparc_chi2_values: Optional[Dict[str, float]] = None
    heavy_sparc_delta_chi2_values: Optional[Dict[str, float]] = None
    heavy_sparc_better_model_values: Optional[Dict[str, str]] = None
    heavy_bell_bridge_selection_values: Optional[Dict[str, float]] = None
    heavy_bell_bridge_physics_values: Optional[Dict[str, float]] = None
    heavy_bell_bridge_status_values: Optional[Dict[str, str]] = None
    full_followup: Optional[Dict[str, Any]] = None
    heavy_enabled = bool(args.heavy_rerun or args.full_rerun_followup)

    # 条件分岐: `args.full_rerun_followup` を満たす経路を評価する。
    if args.full_rerun_followup:
        frozen_path = Path(frozen["path"])
        full_followup = _full_rerun_followup(frozen_path=frozen_path, beta_cases=cases)
        runs_block = full_followup.get("runs") if isinstance(full_followup.get("runs"), dict) else {}
        run1_block = runs_block.get("run1") if isinstance(runs_block.get("run1"), dict) else {}
        heavy = run1_block
    # 条件分岐: 前段条件が不成立で、`heavy_enabled` を追加評価する。
    elif heavy_enabled:
        frozen_path = Path(frozen["path"])
        heavy = _heavy_rerun_cases(frozen_path=frozen_path, beta_cases=cases)
        case_rows = heavy.get("cases") if isinstance(heavy.get("cases"), dict) else {}

    case_rows = heavy.get("cases") if isinstance((heavy or {}).get("cases"), dict) else {}
    # 条件分岐: `case_rows` を満たす経路を評価する。
    if case_rows:
        heavy_eht_values = {}
        heavy_deuteron_values = {}
        heavy_cmb_dell_values = {}
        heavy_cmb_damp_values = {}
        heavy_cmb_status_values = {}
        heavy_sparc_chi2_values = {}
        heavy_sparc_delta_chi2_values = {}
        heavy_sparc_better_model_values = {}
        heavy_bell_bridge_selection_values = {}
        heavy_bell_bridge_physics_values = {}
        heavy_bell_bridge_status_values = {}
        for case_name, row in case_rows.items():
            # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
            if not isinstance(row, dict):
                continue

            delta = _as_float(row.get("eht_delta_coeff_p_minus_gr"))
            # 条件分岐: `delta is not None` を満たす経路を評価する。
            if delta is not None:
                heavy_eht_values[case_name] = float(delta)

            b_val = _as_float(row.get("deuteron_binding_B_MeV"))
            # 条件分岐: `b_val is not None` を満たす経路を評価する。
            if b_val is not None:
                heavy_deuteron_values[case_name] = float(b_val)

            cmb_dell = _as_float(row.get("cmb_holdout46_max_abs_delta_ell"))
            # 条件分岐: `cmb_dell is not None` を満たす経路を評価する。
            if cmb_dell is not None:
                heavy_cmb_dell_values[case_name] = float(cmb_dell)

            cmb_damp = _as_float(row.get("cmb_holdout46_max_abs_delta_amp_rel"))
            # 条件分岐: `cmb_damp is not None` を満たす経路を評価する。
            if cmb_damp is not None:
                heavy_cmb_damp_values[case_name] = float(cmb_damp)

            cmb_status = str(row.get("cmb_overall_extended_status") or "")
            # 条件分岐: `cmb_status` を満たす経路を評価する。
            if cmb_status:
                heavy_cmb_status_values[case_name] = cmb_status

            sparc_chi2 = _as_float(row.get("sparc_chi2_dof_pmodel"))
            # 条件分岐: `sparc_chi2 is not None` を満たす経路を評価する。
            if sparc_chi2 is not None:
                heavy_sparc_chi2_values[case_name] = float(sparc_chi2)

            sparc_delta = _as_float(row.get("sparc_delta_chi2_baryon_minus_pmodel"))
            # 条件分岐: `sparc_delta is not None` を満たす経路を評価する。
            if sparc_delta is not None:
                heavy_sparc_delta_chi2_values[case_name] = float(sparc_delta)

            sparc_model = str(row.get("sparc_better_model_by_chi2") or "")
            # 条件分岐: `sparc_model` を満たす経路を評価する。
            if sparc_model:
                heavy_sparc_better_model_values[case_name] = sparc_model

            bridge_selection = _as_float(row.get("bell_bridge_selection_normalized_score_max"))
            # 条件分岐: `bridge_selection is not None` を満たす経路を評価する。
            if bridge_selection is not None:
                heavy_bell_bridge_selection_values[case_name] = float(bridge_selection)

            bridge_physics = _as_float(row.get("bell_bridge_physics_normalized_score_max"))
            # 条件分岐: `bridge_physics is not None` を満たす経路を評価する。
            if bridge_physics is not None:
                heavy_bell_bridge_physics_values[case_name] = float(bridge_physics)

            bridge_status = str(row.get("bell_bridge_overall_status") or "")
            # 条件分岐: `bridge_status` を満たす経路を評価する。
            if bridge_status:
                heavy_bell_bridge_status_values[case_name] = bridge_status

    # Indicator 1: EHT coefficient delta (dimensionless; tied to Part II strong-field argument)

    coeff_gr = float(eht["coeff_gr"])

    # 関数: `eht_delta` の入出力契約と処理意図を定義する。
    def eht_delta(beta: float) -> float:
        return float((4.0 * math.e * beta / coeff_gr) - 1.0)

    eht_values = (
        heavy_eht_values
        if (heavy_eht_values is not None and len(heavy_eht_values) == 3)
        else {name: eht_delta(beta) for name, beta in cases.items()}
    )
    eht_abs_deltas = [abs(eht_values["beta_low"] - eht_values["beta_ref"]), abs(eht_values["beta_high"] - eht_values["beta_ref"])]
    eht_max_abs_delta = max(eht_abs_deltas)
    eht_pass_thr = min(0.001, 0.1 * float(eht["delta_sigma_required_3sigma"]))
    eht_watch_thr = float(eht["delta_sigma_required_3sigma"])
    eht_status = _classify_numeric(
        max_abs_delta=eht_max_abs_delta,
        pass_abs_threshold=eht_pass_thr,
        watch_abs_threshold=eht_watch_thr,
    )

    # Indicator 2: PPN gamma(β)=2β-1
    gamma_sigma = _as_float(frozen.get("gamma_sigma")) or 2.3e-5

    # 関数: `gamma_pred` の入出力契約と処理意図を定義する。
    def gamma_pred(beta: float) -> float:
        return float(2.0 * beta - 1.0)

    gamma_values = {name: gamma_pred(beta) for name, beta in cases.items()}
    gamma_abs_deltas = [abs(gamma_values["beta_low"] - gamma_values["beta_ref"]), abs(gamma_values["beta_high"] - gamma_values["beta_ref"])]
    gamma_max_abs_delta = max(gamma_abs_deltas)
    gamma_pass_thr = float(gamma_sigma)
    gamma_watch_thr = float(3.0 * gamma_sigma)
    gamma_status = _classify_numeric(
        max_abs_delta=gamma_max_abs_delta,
        pass_abs_threshold=gamma_pass_thr,
        watch_abs_threshold=gamma_watch_thr,
    )

    # Indicator 3: Deuteron binding baseline (Part III nuclear proxy; beta independent in current I/F)
    b_mev = float(deuteron["b_mev"])
    b_sigma = float(deuteron["b_sigma"])
    deuteron_values = (
        heavy_deuteron_values
        if (heavy_deuteron_values is not None and len(heavy_deuteron_values) == 3)
        else {name: b_mev for name in cases.keys()}
    )
    deuteron_abs_deltas = [
        abs(deuteron_values["beta_low"] - deuteron_values["beta_ref"]),
        abs(deuteron_values["beta_high"] - deuteron_values["beta_ref"]),
    ]
    deuteron_max_abs_delta = max(deuteron_abs_deltas)
    deuteron_pass_thr = float(b_sigma)
    deuteron_watch_thr = float(5.0 * b_sigma)
    deuteron_status = _classify_numeric(
        max_abs_delta=deuteron_max_abs_delta,
        pass_abs_threshold=deuteron_pass_thr,
        watch_abs_threshold=deuteron_watch_thr,
    )

    # Indicator 4: CMB acoustic holdout residual (ℓ4-ℓ6 max |Δℓ|)
    cmb_dell_ref = float(cmb["holdout46_max_abs_delta_ell"])
    cmb_dell_values = (
        heavy_cmb_dell_values
        if (heavy_cmb_dell_values is not None and len(heavy_cmb_dell_values) == 3)
        else {name: cmb_dell_ref for name in cases.keys()}
    )
    cmb_dell_abs_deltas = [
        abs(cmb_dell_values["beta_low"] - cmb_dell_values["beta_ref"]),
        abs(cmb_dell_values["beta_high"] - cmb_dell_values["beta_ref"]),
    ]
    cmb_dell_max_abs_delta = max(cmb_dell_abs_deltas)
    cmb_dell_thr = _invariance_thresholds(cmb_dell_values["beta_ref"])
    cmb_dell_pass_thr = float(cmb_dell_thr["pass_abs_threshold"])
    cmb_dell_watch_thr = float(cmb_dell_thr["watch_abs_threshold"])
    cmb_dell_status = _classify_numeric(
        max_abs_delta=cmb_dell_max_abs_delta,
        pass_abs_threshold=cmb_dell_pass_thr,
        watch_abs_threshold=cmb_dell_watch_thr,
    )

    # Indicator 5: CMB acoustic holdout residual (ℓ4-ℓ6 max |ΔA/A|)
    cmb_damp_ref = float(cmb["holdout46_max_abs_delta_amp_rel"])
    cmb_damp_values = (
        heavy_cmb_damp_values
        if (heavy_cmb_damp_values is not None and len(heavy_cmb_damp_values) == 3)
        else {name: cmb_damp_ref for name in cases.keys()}
    )
    cmb_damp_abs_deltas = [
        abs(cmb_damp_values["beta_low"] - cmb_damp_values["beta_ref"]),
        abs(cmb_damp_values["beta_high"] - cmb_damp_values["beta_ref"]),
    ]
    cmb_damp_max_abs_delta = max(cmb_damp_abs_deltas)
    cmb_damp_thr = _invariance_thresholds(cmb_damp_values["beta_ref"])
    cmb_damp_pass_thr = float(cmb_damp_thr["pass_abs_threshold"])
    cmb_damp_watch_thr = float(cmb_damp_thr["watch_abs_threshold"])
    cmb_damp_status = _classify_numeric(
        max_abs_delta=cmb_damp_max_abs_delta,
        pass_abs_threshold=cmb_damp_pass_thr,
        watch_abs_threshold=cmb_damp_watch_thr,
    )

    # Indicator 6: SPARC global chi2/dof (P-model corrected)
    sparc_chi2_ref = float(sparc["chi2_dof_pmodel"])
    sparc_chi2_values = (
        heavy_sparc_chi2_values
        if (heavy_sparc_chi2_values is not None and len(heavy_sparc_chi2_values) == 3)
        else {name: sparc_chi2_ref for name in cases.keys()}
    )
    sparc_chi2_abs_deltas = [
        abs(sparc_chi2_values["beta_low"] - sparc_chi2_values["beta_ref"]),
        abs(sparc_chi2_values["beta_high"] - sparc_chi2_values["beta_ref"]),
    ]
    sparc_chi2_max_abs_delta = max(sparc_chi2_abs_deltas)
    sparc_chi2_thr = _invariance_thresholds(sparc_chi2_values["beta_ref"])
    sparc_chi2_pass_thr = float(sparc_chi2_thr["pass_abs_threshold"])
    sparc_chi2_watch_thr = float(sparc_chi2_thr["watch_abs_threshold"])
    sparc_chi2_status = _classify_numeric(
        max_abs_delta=sparc_chi2_max_abs_delta,
        pass_abs_threshold=sparc_chi2_pass_thr,
        watch_abs_threshold=sparc_chi2_watch_thr,
    )

    # Indicator 7: SPARC delta chi2 (baryon - pmodel)
    sparc_delta_chi2_ref = float(sparc["delta_chi2_baryon_minus_pmodel"])
    sparc_delta_chi2_values = (
        heavy_sparc_delta_chi2_values
        if (heavy_sparc_delta_chi2_values is not None and len(heavy_sparc_delta_chi2_values) == 3)
        else {name: sparc_delta_chi2_ref for name in cases.keys()}
    )
    sparc_delta_chi2_abs_deltas = [
        abs(sparc_delta_chi2_values["beta_low"] - sparc_delta_chi2_values["beta_ref"]),
        abs(sparc_delta_chi2_values["beta_high"] - sparc_delta_chi2_values["beta_ref"]),
    ]
    sparc_delta_chi2_max_abs_delta = max(sparc_delta_chi2_abs_deltas)
    sparc_delta_chi2_thr = _invariance_thresholds(sparc_delta_chi2_values["beta_ref"])
    sparc_delta_chi2_pass_thr = float(sparc_delta_chi2_thr["pass_abs_threshold"])
    sparc_delta_chi2_watch_thr = float(sparc_delta_chi2_thr["watch_abs_threshold"])
    sparc_delta_chi2_status = _classify_numeric(
        max_abs_delta=sparc_delta_chi2_max_abs_delta,
        pass_abs_threshold=sparc_delta_chi2_pass_thr,
        watch_abs_threshold=sparc_delta_chi2_watch_thr,
    )

    # Indicator 8: Bell bridge selection summary max(normalized_score)
    bridge_selection_ref = float(bell_bridge["selection_normalized_score_max"])
    bridge_selection_values = (
        heavy_bell_bridge_selection_values
        if (heavy_bell_bridge_selection_values is not None and len(heavy_bell_bridge_selection_values) == 3)
        else {name: bridge_selection_ref for name in cases.keys()}
    )
    bridge_selection_abs_deltas = [
        abs(bridge_selection_values["beta_low"] - bridge_selection_values["beta_ref"]),
        abs(bridge_selection_values["beta_high"] - bridge_selection_values["beta_ref"]),
    ]
    bridge_selection_max_abs_delta = max(bridge_selection_abs_deltas)
    bridge_selection_thr = _invariance_thresholds(bridge_selection_values["beta_ref"])
    bridge_selection_pass_thr = float(bridge_selection_thr["pass_abs_threshold"])
    bridge_selection_watch_thr = float(bridge_selection_thr["watch_abs_threshold"])
    bridge_selection_status = _classify_numeric(
        max_abs_delta=bridge_selection_max_abs_delta,
        pass_abs_threshold=bridge_selection_pass_thr,
        watch_abs_threshold=bridge_selection_watch_thr,
    )

    # Indicator 9: Bell bridge physics summary max(normalized_score)
    bridge_physics_ref = float(bell_bridge["physics_normalized_score_max"])
    bridge_physics_values = (
        heavy_bell_bridge_physics_values
        if (heavy_bell_bridge_physics_values is not None and len(heavy_bell_bridge_physics_values) == 3)
        else {name: bridge_physics_ref for name in cases.keys()}
    )
    bridge_physics_abs_deltas = [
        abs(bridge_physics_values["beta_low"] - bridge_physics_values["beta_ref"]),
        abs(bridge_physics_values["beta_high"] - bridge_physics_values["beta_ref"]),
    ]
    bridge_physics_max_abs_delta = max(bridge_physics_abs_deltas)
    bridge_physics_thr = _invariance_thresholds(bridge_physics_values["beta_ref"])
    bridge_physics_pass_thr = float(bridge_physics_thr["pass_abs_threshold"])
    bridge_physics_watch_thr = float(bridge_physics_thr["watch_abs_threshold"])
    bridge_physics_status = _classify_numeric(
        max_abs_delta=bridge_physics_max_abs_delta,
        pass_abs_threshold=bridge_physics_pass_thr,
        watch_abs_threshold=bridge_physics_watch_thr,
    )

    # 関数: `_safe_rel_pct` の入出力契約と処理意図を定義する。
    def _safe_rel_pct(delta: float, ref: float) -> Optional[float]:
        # 条件分岐: `ref == 0.0` を満たす経路を評価する。
        if ref == 0.0:
            return None

        return float(100.0 * delta / ref)

    rows_numeric: List[Dict[str, Any]] = [
        {
            "indicator": "EHT delta_coeff_p_minus_GR",
            "section": "Part II",
            "unit": "dimensionless",
            "case_low": eht_values["beta_low"],
            "case_ref": eht_values["beta_ref"],
            "case_high": eht_values["beta_high"],
            "max_abs_delta_vs_ref": eht_max_abs_delta,
            "max_rel_delta_percent_vs_ref": max(
                abs(_safe_rel_pct(eht_abs_deltas[0], eht_values["beta_ref"]) or float("nan")),
                abs(_safe_rel_pct(eht_abs_deltas[1], eht_values["beta_ref"]) or float("nan")),
            ),
            "pass_abs_threshold": eht_pass_thr,
            "watch_abs_threshold": eht_watch_thr,
            "status": eht_status,
            "note": "formula re-evaluation: delta=(4eβ/(2sqrt27))-1",
        },
        {
            "indicator": "PPN gamma_pmodel(β)",
            "section": "Part II",
            "unit": "dimensionless",
            "case_low": gamma_values["beta_low"],
            "case_ref": gamma_values["beta_ref"],
            "case_high": gamma_values["beta_high"],
            "max_abs_delta_vs_ref": gamma_max_abs_delta,
            "max_rel_delta_percent_vs_ref": max(
                abs(_safe_rel_pct(gamma_abs_deltas[0], gamma_values["beta_ref"]) or float("nan")),
                abs(_safe_rel_pct(gamma_abs_deltas[1], gamma_values["beta_ref"]) or float("nan")),
            ),
            "pass_abs_threshold": gamma_pass_thr,
            "watch_abs_threshold": gamma_watch_thr,
            "status": gamma_status,
            "note": "derived from frozen beta: gamma=2β-1",
        },
        {
            "indicator": "Deuteron binding B",
            "section": "Part III",
            "unit": "MeV",
            "case_low": deuteron_values["beta_low"],
            "case_ref": deuteron_values["beta_ref"],
            "case_high": deuteron_values["beta_high"],
            "max_abs_delta_vs_ref": deuteron_max_abs_delta,
            "max_rel_delta_percent_vs_ref": max(
                abs(_safe_rel_pct(deuteron_abs_deltas[0], deuteron_values["beta_ref"]) or float("nan")),
                abs(_safe_rel_pct(deuteron_abs_deltas[1], deuteron_values["beta_ref"]) or float("nan")),
            ),
            "pass_abs_threshold": deuteron_pass_thr,
            "watch_abs_threshold": deuteron_watch_thr,
            "status": deuteron_status,
            "note": (
                "heavy rerun under beta overrides"
                if (heavy_deuteron_values is not None and len(heavy_deuteron_values) == 3)
                else "current nuclear baseline is beta-independent in this I/F"
            ),
        },
        {
            "indicator": "CMB holdout max|Δℓ| (ℓ4-ℓ6)",
            "section": "Part II",
            "unit": "multipole",
            "case_low": cmb_dell_values["beta_low"],
            "case_ref": cmb_dell_values["beta_ref"],
            "case_high": cmb_dell_values["beta_high"],
            "max_abs_delta_vs_ref": cmb_dell_max_abs_delta,
            "max_rel_delta_percent_vs_ref": max(
                abs(_safe_rel_pct(cmb_dell_abs_deltas[0], cmb_dell_values["beta_ref"]) or float("nan")),
                abs(_safe_rel_pct(cmb_dell_abs_deltas[1], cmb_dell_values["beta_ref"]) or float("nan")),
            ),
            "pass_abs_threshold": cmb_dell_pass_thr,
            "watch_abs_threshold": cmb_dell_watch_thr,
            "status": cmb_dell_status,
            "note": "heavy rerun invariance check (CMB acoustic holdout gate metric).",
        },
        {
            "indicator": "CMB holdout max|ΔA/A| (ℓ4-ℓ6)",
            "section": "Part II",
            "unit": "dimensionless",
            "case_low": cmb_damp_values["beta_low"],
            "case_ref": cmb_damp_values["beta_ref"],
            "case_high": cmb_damp_values["beta_high"],
            "max_abs_delta_vs_ref": cmb_damp_max_abs_delta,
            "max_rel_delta_percent_vs_ref": max(
                abs(_safe_rel_pct(cmb_damp_abs_deltas[0], cmb_damp_values["beta_ref"]) or float("nan")),
                abs(_safe_rel_pct(cmb_damp_abs_deltas[1], cmb_damp_values["beta_ref"]) or float("nan")),
            ),
            "pass_abs_threshold": cmb_damp_pass_thr,
            "watch_abs_threshold": cmb_damp_watch_thr,
            "status": cmb_damp_status,
            "note": "heavy rerun invariance check (CMB amplitude holdout gate metric).",
        },
        {
            "indicator": "SPARC chi2/dof (P-model)",
            "section": "Part II",
            "unit": "dimensionless",
            "case_low": sparc_chi2_values["beta_low"],
            "case_ref": sparc_chi2_values["beta_ref"],
            "case_high": sparc_chi2_values["beta_high"],
            "max_abs_delta_vs_ref": sparc_chi2_max_abs_delta,
            "max_rel_delta_percent_vs_ref": max(
                abs(_safe_rel_pct(sparc_chi2_abs_deltas[0], sparc_chi2_values["beta_ref"]) or float("nan")),
                abs(_safe_rel_pct(sparc_chi2_abs_deltas[1], sparc_chi2_values["beta_ref"]) or float("nan")),
            ),
            "pass_abs_threshold": sparc_chi2_pass_thr,
            "watch_abs_threshold": sparc_chi2_watch_thr,
            "status": sparc_chi2_status,
            "note": "heavy rerun invariance check (SPARC global fit quality).",
        },
        {
            "indicator": "SPARC Δchi2 (baryon-pmodel)",
            "section": "Part II",
            "unit": "dimensionless",
            "case_low": sparc_delta_chi2_values["beta_low"],
            "case_ref": sparc_delta_chi2_values["beta_ref"],
            "case_high": sparc_delta_chi2_values["beta_high"],
            "max_abs_delta_vs_ref": sparc_delta_chi2_max_abs_delta,
            "max_rel_delta_percent_vs_ref": max(
                abs(_safe_rel_pct(sparc_delta_chi2_abs_deltas[0], sparc_delta_chi2_values["beta_ref"]) or float("nan")),
                abs(_safe_rel_pct(sparc_delta_chi2_abs_deltas[1], sparc_delta_chi2_values["beta_ref"]) or float("nan")),
            ),
            "pass_abs_threshold": sparc_delta_chi2_pass_thr,
            "watch_abs_threshold": sparc_delta_chi2_watch_thr,
            "status": sparc_delta_chi2_status,
            "note": "heavy rerun invariance check (SPARC model-comparison metric).",
        },
        {
            "indicator": "Bell bridge selection max score",
            "section": "Part III",
            "unit": "dimensionless",
            "case_low": bridge_selection_values["beta_low"],
            "case_ref": bridge_selection_values["beta_ref"],
            "case_high": bridge_selection_values["beta_high"],
            "max_abs_delta_vs_ref": bridge_selection_max_abs_delta,
            "max_rel_delta_percent_vs_ref": max(
                abs(_safe_rel_pct(bridge_selection_abs_deltas[0], bridge_selection_values["beta_ref"]) or float("nan")),
                abs(_safe_rel_pct(bridge_selection_abs_deltas[1], bridge_selection_values["beta_ref"]) or float("nan")),
            ),
            "pass_abs_threshold": bridge_selection_pass_thr,
            "watch_abs_threshold": bridge_selection_watch_thr,
            "status": bridge_selection_status,
            "note": "heavy rerun invariance check (Bell bridge selection summary).",
        },
        {
            "indicator": "Bell bridge physics max score",
            "section": "Part III",
            "unit": "dimensionless",
            "case_low": bridge_physics_values["beta_low"],
            "case_ref": bridge_physics_values["beta_ref"],
            "case_high": bridge_physics_values["beta_high"],
            "max_abs_delta_vs_ref": bridge_physics_max_abs_delta,
            "max_rel_delta_percent_vs_ref": max(
                abs(_safe_rel_pct(bridge_physics_abs_deltas[0], bridge_physics_values["beta_ref"]) or float("nan")),
                abs(_safe_rel_pct(bridge_physics_abs_deltas[1], bridge_physics_values["beta_ref"]) or float("nan")),
            ),
            "pass_abs_threshold": bridge_physics_pass_thr,
            "watch_abs_threshold": bridge_physics_watch_thr,
            "status": bridge_physics_status,
            "note": "heavy rerun invariance check (Bell bridge physics summary).",
        },
    ]

    followup_numeric_status: Optional[str] = None
    followup_categorical_status: Optional[str] = None
    followup_restore_status: Optional[str] = None
    # 条件分岐: `full_followup is not None` を満たす経路を評価する。
    if full_followup is not None:
        comparison = full_followup.get("comparison") if isinstance(full_followup.get("comparison"), dict) else {}
        numeric_comp = comparison.get("numeric") if isinstance(comparison.get("numeric"), dict) else {}
        categorical_comp = comparison.get("categorical") if isinstance(comparison.get("categorical"), dict) else {}
        restore_comp = comparison.get("restore") if isinstance(comparison.get("restore"), dict) else {}

        followup_max_abs_delta = float(_as_float(numeric_comp.get("max_abs_delta")) or 0.0)
        followup_pass_thr = float(_as_float(numeric_comp.get("pass_abs_threshold")) or 1.0e-12)
        followup_watch_thr = float(_as_float(numeric_comp.get("watch_abs_threshold")) or 1.0e-9)
        followup_numeric_status = str(numeric_comp.get("status") or _classify_numeric(
            max_abs_delta=followup_max_abs_delta,
            pass_abs_threshold=followup_pass_thr,
            watch_abs_threshold=followup_watch_thr,
        ))
        rows_numeric.append(
            {
                "indicator": "Full-rerun reproducibility max|Δ(run2-run1)|",
                "section": "Part II/III",
                "unit": "dimensionless",
                "case_low": followup_max_abs_delta,
                "case_ref": 0.0,
                "case_high": followup_max_abs_delta,
                "max_abs_delta_vs_ref": followup_max_abs_delta,
                "max_rel_delta_percent_vs_ref": None,
                "pass_abs_threshold": followup_pass_thr,
                "watch_abs_threshold": followup_watch_thr,
                "status": followup_numeric_status,
                "note": "full-rerun follow-up: heavy rerun run2 vs run1 numeric reproducibility.",
            }
        )

        followup_categorical_status = str(categorical_comp.get("status") or "unknown")
        followup_restore_status = str(restore_comp.get("status") or "unknown")

    # Categorical indicators

    q_status_ref = str(qkpi["overall_status"])
    q_status_low = q_status_ref
    q_status_high = q_status_ref
    q_status = _categorical_triplet_status(case_low=q_status_low, case_ref=q_status_ref, case_high=q_status_high)

    cmb_status_ref = str(cmb["overall_extended_status"])
    cmb_status_values = (
        heavy_cmb_status_values
        if (heavy_cmb_status_values is not None and len(heavy_cmb_status_values) == 3)
        else {name: cmb_status_ref for name in cases.keys()}
    )
    cmb_status = _categorical_triplet_status(
        case_low=str(cmb_status_values["beta_low"]),
        case_ref=str(cmb_status_values["beta_ref"]),
        case_high=str(cmb_status_values["beta_high"]),
    )

    sparc_model_ref = str(sparc["better_model_by_chi2"])
    sparc_model_values = (
        heavy_sparc_better_model_values
        if (heavy_sparc_better_model_values is not None and len(heavy_sparc_better_model_values) == 3)
        else {name: sparc_model_ref for name in cases.keys()}
    )
    sparc_model_status = _categorical_triplet_status(
        case_low=str(sparc_model_values["beta_low"]),
        case_ref=str(sparc_model_values["beta_ref"]),
        case_high=str(sparc_model_values["beta_high"]),
    )

    bridge_status_ref = str(bell_bridge["overall_status"])
    bridge_status_values = (
        heavy_bell_bridge_status_values
        if (heavy_bell_bridge_status_values is not None and len(heavy_bell_bridge_status_values) == 3)
        else {name: bridge_status_ref for name in cases.keys()}
    )
    bridge_status = _categorical_triplet_status(
        case_low=str(bridge_status_values["beta_low"]),
        case_ref=str(bridge_status_values["beta_ref"]),
        case_high=str(bridge_status_values["beta_high"]),
    )

    rows_categorical: List[Dict[str, Any]] = [
        {
            "indicator": "Quantum shared KPI overall status",
            "section": "Part III",
            "case_low": q_status_low,
            "case_ref": q_status_ref,
            "case_high": q_status_high,
            "status": q_status,
            "note": "beta-independent aggregation in current pipeline (invariance by construction).",
        },
        {
            "indicator": "CMB holdout overall status (extended)",
            "section": "Part II",
            "case_low": str(cmb_status_values["beta_low"]),
            "case_ref": str(cmb_status_values["beta_ref"]),
            "case_high": str(cmb_status_values["beta_high"]),
            "status": cmb_status,
            "note": "heavy rerun invariance check (CMB core/extended gate status).",
        },
        {
            "indicator": "SPARC better model by chi2",
            "section": "Part II",
            "case_low": str(sparc_model_values["beta_low"]),
            "case_ref": str(sparc_model_values["beta_ref"]),
            "case_high": str(sparc_model_values["beta_high"]),
            "status": sparc_model_status,
            "note": "heavy rerun invariance check (model-selection label).",
        },
        {
            "indicator": "Bell bridge overall status",
            "section": "Part III",
            "case_low": str(bridge_status_values["beta_low"]),
            "case_ref": str(bridge_status_values["beta_ref"]),
            "case_high": str(bridge_status_values["beta_high"]),
            "status": bridge_status,
            "note": "heavy rerun invariance check (selection/physics bridge gate status).",
        },
    ]
    # 条件分岐: `full_followup is not None` を満たす経路を評価する。
    if full_followup is not None:
        rows_categorical.append(
            {
                "indicator": "Full-rerun categorical reproducibility",
                "section": "Part II/III",
                "case_low": str(followup_categorical_status or "unknown"),
                "case_ref": "pass",
                "case_high": str(followup_categorical_status or "unknown"),
                "status": (
                    "pass"
                    if str(followup_categorical_status) == "pass"
                    else ("watch" if str(followup_categorical_status) == "watch" else "reject")
                ),
                "note": "full-rerun follow-up: status labels (CMB/SPARC/Bell bridge) run2 vs run1 consistency.",
            }
        )
        rows_categorical.append(
            {
                "indicator": "Full-rerun restore-state reproducibility",
                "section": "Part II/III",
                "case_low": str(followup_restore_status or "unknown"),
                "case_ref": "pass",
                "case_high": str(followup_restore_status or "unknown"),
                "status": (
                    "pass"
                    if str(followup_restore_status) == "pass"
                    else ("watch" if str(followup_restore_status) == "watch" else "reject")
                ),
                "note": "full-rerun follow-up: frozen-parameter restore + ref-rerun consistency for run1/run2.",
            }
        )

    severity = max(_status_severity(str(r["status"])) for r in rows_numeric + rows_categorical)
    overall_status = "pass" if severity == 0 else ("watch" if severity == 1 else "reject")

    out_json = out_dir / "beta_sensitivity_audit.json"
    out_csv = out_dir / "beta_sensitivity_audit.csv"
    out_png = out_dir / "beta_sensitivity_audit.png"

    payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 8, "step": "8.7.24", "name": "Beta sensitivity audit"},
        "method": {
            "type": (
                "formula_re_evaluation_plus_full_rerun_followup_repro_check"
                if args.full_rerun_followup
                else (
                    "formula_re_evaluation_plus_heavy_rerun_extended_crosschecks"
                    if heavy_enabled
                    else "formula_re_evaluation_plus_frozen_output_probe"
                )
            ),
            "note": (
                "Evaluate beta sensitivity from fixed formulas/frozen outputs and, when heavy-rerun is enabled, "
                "recompute EHT/deuteron plus CMB/SPARC/Bell bridge cross-checks under beta overrides."
            ),
        },
        "inputs": {
            "frozen_parameters_json": _rel(Path(frozen["path"])),
            "eht_shadow_compare_json": _rel(Path(eht["path"])),
            "deuteron_metrics_json": _rel(Path(deuteron["path"])),
            "cmb_metrics_json": _rel(Path(cmb["path"])),
            "sparc_metrics_json": _rel(Path(sparc["path"])),
            "bell_bridge_json": _rel(Path(bell_bridge["path"])),
            "quantum_connection_shared_kpi_json": _rel(Path(qkpi["path"])),
        },
        "beta_cases": {"beta_low": beta_low, "beta_ref": beta_ref, "beta_high": beta_high},
        "threshold_policy": {
            "eht": {
                "pass_abs_threshold": eht_pass_thr,
                "watch_abs_threshold": eht_watch_thr,
                "source": "delta_sigma_required_3sigma from eht_shadow_compare.json",
            },
            "gamma": {
                "pass_abs_threshold": gamma_pass_thr,
                "watch_abs_threshold": gamma_watch_thr,
                "source": "frozen gamma sigma (or 2*beta_sigma fallback)",
            },
            "deuteron_B": {
                "pass_abs_threshold": deuteron_pass_thr,
                "watch_abs_threshold": deuteron_watch_thr,
                "source": "nuclear_binding_deuteron_metrics.json B_MeV sigma",
            },
            "cmb_holdout_metrics": {
                "pass_abs_threshold": max(cmb_dell_pass_thr, cmb_damp_pass_thr),
                "watch_abs_threshold": max(cmb_dell_watch_thr, cmb_damp_watch_thr),
                "source": "invariance threshold for heavy-rerun (CMB holdout gate metrics)",
            },
            "sparc_metrics": {
                "pass_abs_threshold": max(sparc_chi2_pass_thr, sparc_delta_chi2_pass_thr),
                "watch_abs_threshold": max(sparc_chi2_watch_thr, sparc_delta_chi2_watch_thr),
                "source": "invariance threshold for heavy-rerun (SPARC fit/comparison metrics)",
            },
            "bell_bridge_metrics": {
                "pass_abs_threshold": max(bridge_selection_pass_thr, bridge_physics_pass_thr),
                "watch_abs_threshold": max(bridge_selection_watch_thr, bridge_physics_watch_thr),
                "source": "invariance threshold for heavy-rerun (bridge selection/physics summaries)",
            },
        },
        "indicators": {"numeric": rows_numeric, "categorical": rows_categorical},
        "overall": {
            "status": overall_status,
            "severity": severity,
            "counts": {
                "pass": sum(1 for r in rows_numeric + rows_categorical if str(r["status"]) == "pass"),
                "watch": sum(1 for r in rows_numeric + rows_categorical if str(r["status"]) == "watch"),
                "reject": sum(1 for r in rows_numeric + rows_categorical if str(r["status"]) == "reject"),
            },
        },
        "heavy_rerun": heavy if heavy_enabled else None,
        "full_rerun_followup": full_followup if args.full_rerun_followup else None,
        "outputs": {
            "json": _rel(out_json),
            "csv": _rel(out_csv),
            "png": _rel(out_png),
        },
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out_csv, rows_numeric + rows_categorical)
    _plot_summary(
        rows_numeric=rows_numeric,
        rows_categorical=rows_categorical,
        out_png=out_png,
        beta_low=beta_low,
        beta_ref=beta_ref,
        beta_high=beta_high,
    )

    try:
        worklog.append_event(
            {
                "event_type": "beta_sensitivity_audit",
                "phase": "Phase 8",
                "step": "8.7.24",
                "outputs": {"json": out_json, "csv": out_csv, "png": out_png},
                "metrics": {
                    "overall_status": overall_status,
                    "overall_severity": severity,
                    "eht_max_abs_delta": eht_max_abs_delta,
                    "gamma_max_abs_delta": gamma_max_abs_delta,
                    "deuteron_max_abs_delta": deuteron_max_abs_delta,
                    "cmb_holdout_max_abs_delta_ell": cmb_dell_max_abs_delta,
                    "cmb_holdout_max_abs_delta_amp_rel": cmb_damp_max_abs_delta,
                    "sparc_chi2_max_abs_delta": sparc_chi2_max_abs_delta,
                    "sparc_delta_chi2_max_abs_delta": sparc_delta_chi2_max_abs_delta,
                    "bell_bridge_selection_max_abs_delta": bridge_selection_max_abs_delta,
                    "bell_bridge_physics_max_abs_delta": bridge_physics_max_abs_delta,
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] png : {out_png}")
    print(f"[ok] overall_status={overall_status}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
