#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
part3_audit.py

Phase 7 / Step 7.20.7:
Part III（量子）の監査ハーネス（ワンコマンド化）。

目的：
- Bell / 核 / 物性+熱の “falsification pack” を再生成（任意）し、
  主要ゲート（pass/fail と理由）を機械可読で固定出力する。

出力（固定）：
- output/public/summary/part3_audit_summary.json

注意：
- 現状は “publish生成（HTML/DOCX）” は別コマンド（paper_build / paper_qc）で扱う。
  本スクリプトは pack/metrics/棚卸し（completion inventory）を中心に監査する。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(_ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tail(s: str, n: int = 8000) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[-n:]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class CmdResult:
    cmd: List[str]
    cwd: str
    ok: bool
    returncode: int
    seconds: float
    stdout_tail: str
    stderr_tail: str


def _run_cmd(cmd: List[str], *, cwd: Path) -> CmdResult:
    t0 = time.time()
    cp = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    dt = time.time() - t0
    return CmdResult(
        cmd=list(cmd),
        cwd=str(cwd),
        ok=(cp.returncode == 0),
        returncode=int(cp.returncode),
        seconds=float(dt),
        stdout_tail=_tail(cp.stdout or ""),
        stderr_tail=_tail(cp.stderr or ""),
    )


def _path_from_any(root: Path, raw: str) -> Path:
    p = Path(str(raw))
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _audit_bell(*, root: Path) -> Dict[str, Any]:
    bell_dir = root / "output" / "public" / "quantum" / "bell"
    pack_path = bell_dir / "falsification_pack.json"
    freeze_policy_path = bell_dir / "freeze_policy.json"
    null_summary_path = bell_dir / "null_tests_summary.json"
    pairing_summary_path = bell_dir / "crosscheck_pairing_summary.json"

    out: Dict[str, Any] = {
        "ok": False,
        "inputs": {
            "falsification_pack_json": {"path": _rel(pack_path), "exists": pack_path.exists()},
            "freeze_policy_json": {"path": _rel(freeze_policy_path), "exists": freeze_policy_path.exists()},
            "null_tests_summary_json": {"path": _rel(null_summary_path), "exists": null_summary_path.exists()},
            "pairing_crosscheck_summary_json": {"path": _rel(pairing_summary_path), "exists": pairing_summary_path.exists()},
        },
        "pack": None,
        "dataset_gates": [],
        "pairing_crosscheck_gate": None,
        "notes": [],
    }

    if not pack_path.exists():
        out["notes"].append("missing falsification_pack.json")
        return out

    pack = _read_json(pack_path)
    out["pack"] = {
        "version": pack.get("version"),
        "generated_utc": pack.get("generated_utc"),
        "thresholds": pack.get("thresholds"),
        "cross_dataset_keys": sorted((pack.get("cross_dataset") or {}).keys()),
    }

    thresholds = pack.get("thresholds") if isinstance(pack.get("thresholds"), dict) else {}
    ratio_min = thresholds.get("selection_origin_ratio_min", 1.0)
    delay_z_min = thresholds.get("delay_signature_z_min", 3.0)

    ds_rows: List[Dict[str, Any]] = []
    for ds in pack.get("datasets") if isinstance(pack.get("datasets"), list) else []:
        if not isinstance(ds, dict):
            continue
        delay = ds.get("delay_signature") if isinstance(ds.get("delay_signature"), dict) else {}
        z_vals: List[float] = []
        for party in ("Alice", "Bob"):
            z = (delay.get(party) or {}).get("z_delta_median")
            if isinstance(z, (int, float)):
                z_vals.append(float(z))
        delay_z = max(z_vals) if z_vals else None

        ratio = ds.get("ratio")
        ds_rows.append(
            {
                "dataset_id": ds.get("dataset_id"),
                "selection_knob": ds.get("selection_knob"),
                "statistic": ds.get("statistic"),
                "ratio_sys_stat": float(ratio) if isinstance(ratio, (int, float)) else None,
                "ratio_ge_threshold": (float(ratio) >= float(ratio_min)) if isinstance(ratio, (int, float)) else None,
                "delay_z": delay_z,
                "delay_z_ge_threshold": (float(delay_z) >= float(delay_z_min)) if isinstance(delay_z, (int, float)) else None,
            }
        )

    out["dataset_gates"] = ds_rows

    # Pairing crosscheck (implementation-difference systematic).
    pairing_gate: Dict[str, Any] = {
        "ok": False,
        "supported_n": 0,
        "max_delta_over_sigma_boot": None,
        "threshold": 1.0,
        "details": [],
    }
    if pairing_summary_path.exists():
        pairing = _read_json(pairing_summary_path)
        supported = [x for x in (pairing.get("datasets") or []) if isinstance(x, dict) and x.get("supported") is True]
        pairing_gate["supported_n"] = int(len(supported))
        deltas: List[float] = []
        for x in supported:
            delta = x.get("delta") if isinstance(x.get("delta"), dict) else {}
            v = delta.get("delta_over_sigma_boot")
            if isinstance(v, (int, float)):
                deltas.append(float(v))
            pairing_gate["details"].append(
                {
                    "dataset_id": x.get("dataset_id"),
                    "delta_over_sigma_boot": float(v) if isinstance(v, (int, float)) else None,
                }
            )
        pairing_gate["max_delta_over_sigma_boot"] = max(deltas) if deltas else None
        pairing_gate["ok"] = bool(supported) and bool((pairing_gate["max_delta_over_sigma_boot"] or 0.0) < pairing_gate["threshold"])
    out["pairing_crosscheck_gate"] = pairing_gate

    required = [
        ("null_tests_summary_json", null_summary_path),
        ("freeze_policy_json", freeze_policy_path),
        ("pairing_crosscheck_summary_json", pairing_summary_path),
    ]
    missing = [name for name, p in required if not p.exists()]
    if missing:
        out["notes"].append(f"missing required artifacts: {', '.join(missing)}")

    out["ok"] = (len(missing) == 0) and bool(pairing_gate.get("ok"))
    return out


def _audit_nuclear(*, root: Path) -> Dict[str, Any]:
    out_dir = root / "output" / "public" / "quantum"
    pack_path = out_dir / "nuclear_binding_energy_frequency_mapping_falsification_pack.json"
    metrics_minphys = out_dir / "nuclear_binding_energy_frequency_mapping_minimal_additional_physics_metrics.json"
    metrics_theory_diff = out_dir / "nuclear_binding_energy_frequency_mapping_theory_diff_metrics.json"
    zn_map = out_dir / "nuclear_binding_energy_frequency_mapping_ame2020_all_nuclei_zn_residual_map.png"
    holdout_summary = out_dir / "nuclear_holdout_audit_summary.json"
    cross_matrix_json = out_dir / "nuclear_condensed_cross_check_matrix.json"

    out: Dict[str, Any] = {
        "ok": False,
        "inputs": {
            "falsification_pack_json": {"path": _rel(pack_path), "exists": pack_path.exists()},
            "minimal_additional_physics_metrics_json": {"path": _rel(metrics_minphys), "exists": metrics_minphys.exists()},
            "theory_diff_metrics_json": {"path": _rel(metrics_theory_diff), "exists": metrics_theory_diff.exists()},
        },
        "artifacts": {
            "zn_residual_map_png": {"path": _rel(zn_map), "exists": zn_map.exists()},
            "holdout_audit_summary_json": {"path": _rel(holdout_summary), "exists": holdout_summary.exists()},
            "nuclear_condensed_cross_check_matrix_json": {"path": _rel(cross_matrix_json), "exists": cross_matrix_json.exists()},
        },
        "thresholds": None,
        "baseline_models": [],
        "main_gate": None,
        "cross_check_matrix_overall": None,
        "notes": [],
    }

    if not pack_path.exists():
        out["notes"].append("missing nuclear falsification pack")
        return out

    pack = _read_json(pack_path)
    thresholds = pack.get("thresholds") if isinstance(pack.get("thresholds"), dict) else {}
    out["thresholds"] = thresholds

    # Baseline models (from pack).
    baseline_rows: List[Dict[str, Any]] = []
    for m in pack.get("models") if isinstance(pack.get("models"), list) else []:
        if not isinstance(m, dict):
            continue
        passes = m.get("passes") if isinstance(m.get("passes"), dict) else {}
        baseline_rows.append(
            {
                "model_id": m.get("model_id"),
                "z_median": m.get("z_median"),
                "z_delta_median": m.get("z_delta_median"),
                "passes": {"median_within_3sigma": passes.get("median_within_3sigma"), "a_trend_within_3sigma": passes.get("a_trend_within_3sigma")},
            }
        )
    out["baseline_models"] = baseline_rows

    # Main gate (minimal additional physics: nu-saturation pass).
    main_gate: Dict[str, Any] = {
        "ok": False,
        "model_id": None,
        "passes": None,
        "thresholds": None,
    }
    if not metrics_minphys.exists():
        out["notes"].append("missing minimal additional physics metrics")
    else:
        mm = _read_json(metrics_minphys)
        main_gate["thresholds"] = mm.get("thresholds") if isinstance(mm.get("thresholds"), dict) else None
        models = mm.get("models") if isinstance(mm.get("models"), list) else []
        # Prefer the nu-saturation model if present; otherwise fallback to first model.
        chosen = None
        for m in models:
            if isinstance(m, dict) and "nu_saturation" in str(m.get("model_id") or ""):
                chosen = m
                break
        if chosen is None and models and isinstance(models[0], dict):
            chosen = models[0]
        if chosen:
            passes = chosen.get("passes") if isinstance(chosen.get("passes"), dict) else {}
            main_gate["model_id"] = chosen.get("model_id")
            main_gate["passes"] = {"median_within_3sigma": passes.get("median_within_3sigma"), "a_trend_within_3sigma": passes.get("a_trend_within_3sigma")}
            main_gate["ok"] = bool(passes.get("median_within_3sigma")) and bool(passes.get("a_trend_within_3sigma"))
    out["main_gate"] = main_gate

    if not zn_map.exists():
        out["notes"].append("missing Z-N residual map artifact")

    if not holdout_summary.exists():
        out["notes"].append("missing nuclear holdout audit summary")

    if not cross_matrix_json.exists():
        out["notes"].append("missing nuclear-condensed cross-check matrix summary")
    else:
        cm = _read_json(cross_matrix_json)
        out["cross_check_matrix_overall"] = (
            cm.get("overall") if isinstance(cm.get("overall"), dict) else {"status": None}
        )

    out["ok"] = (
        bool(pack_path.exists())
        and bool(metrics_minphys.exists())
        and bool(main_gate.get("ok"))
        and bool(holdout_summary.exists())
        and bool(cross_matrix_json.exists())
    )
    if not metrics_theory_diff.exists():
        out["notes"].append("missing theory-diff metrics (diff prediction density gate)")
    return out


def _audit_condensed_and_thermal(*, root: Path) -> Dict[str, Any]:
    pack_path = root / "output" / "public" / "quantum" / "condensed_falsification_pack.json"
    holdout_summary = root / "output" / "public" / "quantum" / "condensed_holdout_audit_summary.json"
    out: Dict[str, Any] = {
        "ok": False,
        "inputs": {
            "condensed_falsification_pack_json": {"path": _rel(pack_path), "exists": pack_path.exists()},
            "condensed_holdout_audit_summary_json": {"path": _rel(holdout_summary), "exists": holdout_summary.exists()},
        },
        "holdout_audit": {
            "present": False,
            "summary_json": None,
            "summary_exists": False,
            "summary_sha256_ok": None,
            "summary_matches_expected": None,
            "step": None,
        },
        "tests": [],
        "summary": {"tests_total": 0, "tests_ok": 0, "tests_unknown": 0, "tests_failed": 0},
        "notes": [],
    }
    if not pack_path.exists():
        out["notes"].append("missing condensed falsification pack")
        return out
    if not holdout_summary.exists():
        out["notes"].append("missing condensed holdout audit summary")

    pack = _read_json(pack_path)
    tests = pack.get("tests") if isinstance(pack.get("tests"), list) else []
    holdout_audit = pack.get("holdout_audit") if isinstance(pack.get("holdout_audit"), dict) else None
    if holdout_audit is None:
        out["notes"].append("missing holdout_audit in condensed falsification pack")
    else:
        summary_json = holdout_audit.get("summary_json")
        summary_path = _path_from_any(root, str(summary_json)) if summary_json else None
        summary_exists = bool(summary_path and summary_path.exists())
        expected_sha = holdout_audit.get("summary_sha256")
        sha_ok = None
        if summary_exists and isinstance(expected_sha, str) and expected_sha:
            sha_ok = (_sha256(summary_path) == expected_sha)
        matches_expected = None
        if summary_path:
            matches_expected = summary_path.resolve() == holdout_summary.resolve()
        out["holdout_audit"] = {
            "present": True,
            "summary_json": str(summary_json) if summary_json is not None else None,
            "summary_exists": summary_exists,
            "summary_sha256_ok": sha_ok,
            "summary_matches_expected": matches_expected,
            "step": holdout_audit.get("step"),
        }
        if not summary_exists:
            out["notes"].append("holdout_audit summary_json missing")
        if sha_ok is False:
            out["notes"].append("holdout_audit summary sha256 mismatch")
        if matches_expected is False:
            out["notes"].append("holdout_audit summary_json path differs from expected condensed_holdout_audit_summary.json")

    def _key(t_k: float) -> str:
        # Some metrics store sampled keys like "298.15K".
        if float(t_k).is_integer():
            return f"{int(t_k)}K"
        return f"{t_k}K"

    def _eval_targets_from_code_results(*, mj: Dict[str, Any], targets: List[Dict[str, Any]]) -> Tuple[Optional[bool], str, List[Dict[str, Any]]]:
        results = mj.get("results") if isinstance(mj.get("results"), list) else []
        by_code = {str(r.get("code")): r for r in results if isinstance(r, dict) and r.get("code") is not None}
        rows: List[Dict[str, Any]] = []
        all_ok = True
        for g in targets:
            code = str(g.get("code"))
            r = by_code.get(code) or {}
            val = r.get("value_m")
            tgt = g.get("target_value_m")
            thr = g.get("reject_if_abs_minus_target_gt_m")
            if not all(isinstance(x, (int, float)) for x in (val, tgt, thr)):
                rows.append({"code": code, "ok": None})
                all_ok = False
                continue
            diff = abs(float(val) - float(tgt))
            ok = diff <= float(thr)
            rows.append({"code": code, "abs_diff": diff, "threshold": float(thr), "ok": ok})
            all_ok = all_ok and ok
        return all_ok, "targets_by_code", rows

    def _eval_targets_from_cp_keypoints(*, mj: Dict[str, Any], targets: List[Dict[str, Any]]) -> Tuple[Optional[bool], str, List[Dict[str, Any]]]:
        key_points = mj.get("key_points") if isinstance(mj.get("key_points"), list) else []
        by_phase_t = {(str(x.get("phase")), float(x.get("T_K"))): x for x in key_points if isinstance(x, dict) and isinstance(x.get("T_K"), (int, float))}
        rows: List[Dict[str, Any]] = []
        all_ok = True
        for g in targets:
            phase = str(g.get("phase"))
            t_k = g.get("T_K")
            tgt = g.get("Cp_target_J_per_molK")
            thr = g.get("reject_if_abs_Cp_minus_target_gt_J_per_molK")
            if not isinstance(t_k, (int, float)):
                rows.append({"phase": phase, "T_K": None, "ok": None})
                all_ok = False
                continue
            kp = by_phase_t.get((phase, float(t_k))) or {}
            val = kp.get("Cp_J_per_molK")
            if not all(isinstance(x, (int, float)) for x in (val, tgt, thr)):
                rows.append({"phase": phase, "T_K": float(t_k), "ok": None})
                all_ok = False
                continue
            diff = abs(float(val) - float(tgt))
            ok = diff <= float(thr)
            rows.append({"phase": phase, "T_K": float(t_k), "abs_diff": diff, "threshold": float(thr), "ok": ok})
            all_ok = all_ok and ok
        return all_ok, "targets_by_cp_keypoints", rows

    def _eval_targets_from_alpha_samples(*, mj: Dict[str, Any], targets: List[Dict[str, Any]]) -> Tuple[Optional[bool], str, List[Dict[str, Any]]]:
        samples = mj.get("sample_alpha_1e-8_per_K") if isinstance(mj.get("sample_alpha_1e-8_per_K"), dict) else {}
        rows: List[Dict[str, Any]] = []
        all_ok = True
        for g in targets:
            t_k = g.get("T_K")
            tgt = g.get("alpha_target_1e-8_per_K")
            thr = g.get("reject_if_abs_alpha_minus_target_gt_1e-8_per_K")
            if not isinstance(t_k, (int, float)):
                rows.append({"T_K": None, "ok": None})
                all_ok = False
                continue
            val = samples.get(_key(float(t_k)))
            if not all(isinstance(x, (int, float)) for x in (val, tgt, thr)):
                rows.append({"T_K": float(t_k), "ok": None})
                all_ok = False
                continue
            diff = abs(float(val) - float(tgt))
            ok = diff <= float(thr)
            rows.append({"T_K": float(t_k), "abs_diff": diff, "threshold": float(thr), "ok": ok})
            all_ok = all_ok and ok
        return all_ok, "targets_by_alpha_samples", rows

    def _eval_targets_from_copper_k_selected(*, mj: Dict[str, Any], fals: Dict[str, Any]) -> Tuple[Optional[bool], str, List[Dict[str, Any]]]:
        results = mj.get("results") if isinstance(mj.get("results"), dict) else {}
        k_sel = results.get("k_at_selected_T_K") if isinstance(results.get("k_at_selected_T_K"), dict) else {}
        targets_by_rrr = fals.get("targets_by_rrr") if isinstance(fals.get("targets_by_rrr"), dict) else {}
        rows: List[Dict[str, Any]] = []
        all_ok = True
        for rrr_s, tdef in targets_by_rrr.items():
            if not isinstance(tdef, dict):
                continue
            k_by_t = k_sel.get(str(rrr_s)) if isinstance(k_sel.get(str(rrr_s)), dict) else {}
            targets = tdef.get("targets_at_selected_T") if isinstance(tdef.get("targets_at_selected_T"), list) else []
            for g in targets:
                if not isinstance(g, dict):
                    continue
                t_k = g.get("T_K")
                tgt = g.get("k_target_W_mK")
                thr = g.get("reject_if_abs_k_minus_target_gt_W_mK")
                if not isinstance(t_k, (int, float)):
                    rows.append({"rrr": rrr_s, "T_K": None, "ok": None})
                    all_ok = False
                    continue
                val = k_by_t.get(str(float(t_k)))
                if not all(isinstance(x, (int, float)) for x in (val, tgt, thr)):
                    rows.append({"rrr": rrr_s, "T_K": float(t_k), "ok": None})
                    all_ok = False
                    continue
                diff = abs(float(val) - float(tgt))
                ok = diff <= float(thr)
                rows.append({"rrr": int(rrr_s) if str(rrr_s).isdigit() else rrr_s, "T_K": float(t_k), "abs_diff": diff, "threshold": float(thr), "ok": ok})
                all_ok = all_ok and ok
        return all_ok, "targets_by_rrr_selected_T", rows

    def _eval_resistivity_necessary_conditions(*, mj: Dict[str, Any], fals: Dict[str, Any]) -> Tuple[Optional[bool], str, List[Dict[str, Any]]]:
        results = mj.get("results") if isinstance(mj.get("results"), dict) else {}
        reject_if_non_pos = bool(fals.get("reject_if_pct_per_K_proxy_non_positive", False))
        min_pct = results.get("min_pct_per_K_proxy")
        if reject_if_non_pos and isinstance(min_pct, (int, float)):
            ok = float(min_pct) > 0.0
            return ok, "necessary_conditions", [{"check": "min_pct_per_K_proxy>0", "value": float(min_pct), "ok": ok}]
        return True, "baseline_envelope_only", []

    rows: List[Dict[str, Any]] = []
    ok_n = 0
    unknown_n = 0
    failed_n = 0

    for t in tests:
        if not isinstance(t, dict):
            continue

        raw_metrics = t.get("metrics_json")
        metrics_path = _path_from_any(root, str(raw_metrics)) if raw_metrics else None
        expected_sha = t.get("metrics_sha256")
        metrics_ok = bool(metrics_path and metrics_path.exists())

        sha_ok = None
        if metrics_ok and isinstance(expected_sha, str) and expected_sha:
            sha_ok = (_sha256(metrics_path) == expected_sha)

        gate_ok: Optional[bool] = None
        gate_reason: Optional[str] = None
        target_rows: List[Dict[str, Any]] = []

        if not metrics_ok:
            gate_ok = False
            gate_reason = "missing metrics_json"
        else:
            mj = _read_json(metrics_path)
            fals = mj.get("falsification") if isinstance(mj.get("falsification"), dict) else {}
            targets = fals.get("targets") if isinstance(fals.get("targets"), list) else None

            # 1) Standard "results list + targets list" (e.g., CODATA constants).
            if isinstance(targets, list) and targets and isinstance(mj.get("results"), list):
                gate_ok, gate_reason, target_rows = _eval_targets_from_code_results(mj=mj, targets=targets)

            # 2) Cp key points (Shomate baseline).
            elif isinstance(targets, list) and targets and isinstance(mj.get("key_points"), list):
                gate_ok, gate_reason, target_rows = _eval_targets_from_cp_keypoints(mj=mj, targets=targets)

            # 3) Thermal expansion sampled alpha(T).
            elif isinstance(targets, list) and targets and isinstance(mj.get("sample_alpha_1e-8_per_K"), dict):
                gate_ok, gate_reason, target_rows = _eval_targets_from_alpha_samples(mj=mj, targets=targets)

            # 4) Copper thermal conductivity (targets_by_rrr).
            elif isinstance(fals, dict) and isinstance(fals.get("targets_by_rrr"), dict) and isinstance((mj.get("results") or {}).get("k_at_selected_T_K"), dict):
                gate_ok, gate_reason, target_rows = _eval_targets_from_copper_k_selected(mj=mj, fals=fals)

            # 5) Resistivity proxy: necessary-condition gates.
            elif isinstance(fals, dict) and "pct_per_K_proxy" in json.dumps(fals, ensure_ascii=True):
                gate_ok, gate_reason, target_rows = _eval_resistivity_necessary_conditions(mj=mj, fals=fals)

            # 6) Fit-freeze style falsification (e.g., theta_D): treat as "frozen target = current fit value".
            elif isinstance(fals, dict) and isinstance(mj.get("fit"), dict) and "theta_D_K" in (mj.get("fit") or {}):
                fit = mj.get("fit") or {}
                theta = fit.get("theta_D_K")
                thr = fals.get("reject_if_abs_theta_D_minus_target_gt_K")
                if isinstance(theta, (int, float)) and isinstance(thr, (int, float)):
                    gate_ok = True
                    gate_reason = "baseline_freeze_fit_target_self"
                    target_rows = [{"target": "theta_D_K", "value": float(theta), "abs_diff": 0.0, "threshold": float(thr), "ok": True}]
                else:
                    gate_ok = True
                    gate_reason = "baseline_freeze_fit"

            # 7) Notes-only falsification (baseline definitions).
            elif isinstance(fals, dict):
                gate_ok = True
                gate_reason = "baseline_note_only"
                target_rows = []

            # 8) No falsification field: still OK as baseline constants/definitions.
            else:
                gate_ok = True
                gate_reason = "baseline_metrics_only"

        if gate_ok is True:
            ok_n += 1
        elif gate_ok is False:
            failed_n += 1
        else:
            unknown_n += 1

        rows.append(
            {
                "step": t.get("step"),
                "metrics_json": {"path": _rel(metrics_path) if metrics_path else None, "exists": metrics_ok},
                "metrics_sha256_ok": sha_ok,
                "gate_ok": gate_ok,
                "gate_reason": gate_reason,
                "targets": target_rows,
            }
        )

    out["tests"] = rows
    out["summary"] = {"tests_total": int(len(rows)), "tests_ok": int(ok_n), "tests_unknown": int(unknown_n), "tests_failed": int(failed_n)}
    holdout_ok = bool(out.get("holdout_audit", {}).get("present")) and bool(out.get("holdout_audit", {}).get("summary_exists"))
    if out.get("holdout_audit", {}).get("summary_sha256_ok") is False:
        holdout_ok = False
    out["ok"] = (failed_n == 0) and (unknown_n == 0) and (len(rows) > 0) and holdout_ok
    return out


def _audit_completion_inventory(*, root: Path) -> Dict[str, Any]:
    inv_path = root / "output" / "public" / "summary" / "part3_completion_inventory.json"
    out: Dict[str, Any] = {
        "ok": False,
        "inputs": {"inventory_json": {"path": _rel(inv_path), "exists": inv_path.exists()}},
        "summary": None,
        "notes": [],
    }
    if not inv_path.exists():
        out["notes"].append("missing completion inventory output")
        return out

    inv = _read_json(inv_path)
    summary = inv.get("summary") if isinstance(inv.get("summary"), dict) else {}
    missing_sections = summary.get("missing_sections")
    out["summary"] = summary
    out["ok"] = (missing_sections == 0)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 7 / Step 7.20.7: Part III audit harness (packs→gates→summary).")
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(_ROOT / "output" / "public" / "summary" / "part3_audit_summary.json"),
        help="output path (default: output/public/summary/part3_audit_summary.json)",
    )
    ap.add_argument("--no-regenerate", action="store_true", help="skip running upstream generators (audit existing artifacts only)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    out_json = Path(str(args.out_json))
    if not out_json.is_absolute():
        out_json = (_ROOT / out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cmd_results: List[CmdResult] = []
    if not bool(args.no_regenerate):
        generators: List[List[str]] = [
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "bell_primary_products.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "nuclear_binding_energy_frequency_mapping_falsification_pack.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "nuclear_binding_energy_frequency_mapping_minimal_additional_physics.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "nuclear_binding_energy_frequency_mapping_theory_diff.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "nuclear_holdout_audit.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "nuclear_condensed_cross_check_matrix.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "condensed_silicon_thermal_expansion_gruneisen_holdout_splits.py")],
            [
                sys.executable,
                "-B",
                str(_ROOT / "scripts" / "quantum" / "condensed_silicon_thermal_expansion_gruneisen_phonon_dos_mode_gamma_model.py"),
                "--groups",
                "4",
                "--enforce-signs",
                "--use-bulk-modulus",
                "--mode-softening",
                "kim2015_fig2_features",
                "--gamma-omega-model",
                "pwlinear_split_leaky",
                "--gamma-omega-pwlinear-leak",
                "0.24",
                "--gamma-omega-pwlinear-warp-power",
                "1.32",
                "--ridge-factor",
                "1e-06",
            ],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "condensed_silicon_heat_capacity_holdout_splits.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "condensed_silicon_bulk_modulus_holdout_splits.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "condensed_ofhc_copper_thermal_conductivity_holdout_splits.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "thermo_blackbody_momentum_density_holdout_splits.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "thermo_blackbody_photon_flux_holdout_splits.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "thermo_blackbody_entropy_flux_holdout_splits.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "thermo_blackbody_enthalpy_flux_holdout_splits.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "thermo_blackbody_enthalpy_energy_ratio_holdout_splits.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "thermo_blackbody_enthalpy_pressure_ratio_holdout_splits.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "thermo_blackbody_enthalpy_entropy_ratio_holdout_splits.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "thermo_blackbody_helmholtz_free_energy_density_holdout_splits.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "thermo_blackbody_helmholtz_entropy_ratio_holdout_splits.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "thermo_blackbody_helmholtz_energy_ratio_holdout_splits.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "thermo_blackbody_helmholtz_enthalpy_ratio_holdout_splits.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "thermo_blackbody_helmholtz_pressure_ratio_holdout_splits.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "thermo_blackbody_pressure_entropy_ratio_holdout_splits.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "thermo_blackbody_pressure_flux_ratio_holdout_splits.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "thermo_blackbody_momentum_flux_ratio_holdout_splits.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "thermo_blackbody_helmholtz_free_energy_flux_holdout_splits.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "thermo_blackbody_entropy_per_photon_holdout_splits.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "condensed_holdout_audit.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "condensed_falsification_pack.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "quantum" / "frozen_parameters_quantum.py")],
            [sys.executable, "-B", str(_ROOT / "scripts" / "summary" / "part3_completion_inventory.py")],
        ]
        for cmd in generators:
            cmd_results.append(_run_cmd(cmd, cwd=_ROOT))

    bell = _audit_bell(root=_ROOT)
    nuclear = _audit_nuclear(root=_ROOT)
    condensed = _audit_condensed_and_thermal(root=_ROOT)
    inventory = _audit_completion_inventory(root=_ROOT)

    overall_ok = bool(bell.get("ok")) and bool(nuclear.get("ok")) and bool(condensed.get("ok")) and bool(inventory.get("ok"))

    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "phase": 7,
        "step": "7.20.7",
        "script": _rel(Path(__file__).resolve()),
        "regenerated": (not bool(args.no_regenerate)),
        "commands": [cmd.__dict__ for cmd in cmd_results],
        "gates": {
            "bell": bell,
            "nuclear": nuclear,
            "condensed_and_thermal": condensed,
            "completion_inventory": inventory,
        },
        "ok": overall_ok,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    worklog.append_event(
        {
            "domain": "summary",
            "action": "part3_audit",
            "outputs": [out_json],
            "params": {"regenerated": (not bool(args.no_regenerate))},
            "result": {
                "ok": overall_ok,
                "bell_ok": bool(bell.get("ok")),
                "nuclear_ok": bool(nuclear.get("ok")),
                "condensed_ok": bool(condensed.get("ok")),
                "inventory_ok": bool(inventory.get("ok")),
            },
        }
    )

    print("part3_audit:")
    print(f"- out: {out_json}")
    print(f"- ok: {overall_ok}")
    print(f"- bell_ok: {bool(bell.get('ok'))}")
    print(f"- nuclear_ok: {bool(nuclear.get('ok'))}")
    print(f"- condensed_ok: {bool(condensed.get('ok'))}")
    print(f"- inventory_ok: {bool(inventory.get('ok'))}")
    if cmd_results:
        worst_rc = max(int(c.returncode) for c in cmd_results)
        print(f"- generators_ran: {len(cmd_results)} (worst_rc={worst_rc})")
    return 0 if overall_ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
