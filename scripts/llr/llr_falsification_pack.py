#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llr_falsification_pack.py

Phase 2 / Step 2.1（LLR）:
LLR（月レーザー測距）のバッチ監査結果（time-tag/外れ値/補正の寄与）を、
公開用の “falsification pack” として `output/public/llr/` に固定する。

このスクリプトは計算の再実行を行わない（既存の private 出力を public に昇格する）。

入力（既存の固定出力）:
- output/private/llr/batch/llr_batch_summary.json
- output/private/llr/batch/llr_time_tag_best_by_station.json
- output/private/llr/batch/llr_outliers_diagnosis_summary.json

出力（固定; Git tracked）:
- output/public/llr/llr_falsification_pack.json
- output/public/llr/（主要PNG/JSON/CSVをコピー）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from scripts.summary import worklog  # type: ignore # noqa: E402
except Exception:  # pragma: no cover
    worklog = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def _copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    # 条件分岐: `not src.exists()` を満たす経路を評価する。
    if not src.exists():
        raise FileNotFoundError(str(src))

    dst.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `dst.exists() and not overwrite` を満たす経路を評価する。
    if dst.exists() and not overwrite:
        return

    shutil.copy2(src, dst)


def _artifact_record(path: Path) -> Dict[str, Any]:
    return {
        "path": _rel(path),
        "sha256": _sha256(path) if path.exists() else None,
        "bytes": int(path.stat().st_size) if path.exists() else None,
    }


def _as_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None

    return v if v == v else None


def _build_pack(
    *,
    batch_summary: Dict[str, Any],
    time_tag_best: Dict[str, Any],
    outliers_diag: Dict[str, Any],
    operational_metrics: Optional[Dict[str, Any]],
    precision_reaudit: Optional[Dict[str, Any]],
    apol_pos_eop_feasibility: Optional[Dict[str, Any]],
    apol_primary_coord_route: Optional[Dict[str, Any]],
    time_tag_mode_diff_audit: Optional[Dict[str, Any]],
    out_dir: Path,
    copied: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    med = batch_summary.get("median_rms_ns") if isinstance(batch_summary.get("median_rms_ns"), dict) else {}
    rms_tide = _as_float(med.get("station_reflector_tropo_tide") if isinstance(med, dict) else None)
    rms_no_shapiro = _as_float(med.get("station_reflector_tropo_no_shapiro") if isinstance(med, dict) else None)
    ratio = None
    # 条件分岐: `rms_tide is not None and rms_no_shapiro not in (None, 0.0)` を満たす経路を評価する。
    if rms_tide is not None and rms_no_shapiro not in (None, 0.0):
        ratio = rms_tide / float(rms_no_shapiro)

    # Legacy support gate: Shapiro term should improve the median RMS.

    shapiro_threshold = 0.95
    shapiro_pass = None if ratio is None else (ratio <= shapiro_threshold)

    op_summary = []
    # 条件分岐: `isinstance(operational_metrics, dict)` を満たす経路を評価する。
    if isinstance(operational_metrics, dict):
        v = operational_metrics.get("summary")
        # 条件分岐: `isinstance(v, list)` を満たす経路を評価する。
        if isinstance(v, list):
            op_summary = v

    def _op_metric(subset: str, key: str) -> Optional[float]:
        for row in op_summary:
            # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
            if not isinstance(row, dict):
                continue

            # 条件分岐: `str(row.get("subset", "")) != subset` を満たす経路を評価する。

            if str(row.get("subset", "")) != subset:
                continue

            return _as_float(row.get(key))

        return None

    op_all_wrms = _op_metric("all", "weighted_rms_ns_bin_rms")
    op_ex_nglr1_wrms = _op_metric("exclude_nglr1", "weighted_rms_ns_bin_rms")
    op_modern_wrms = _op_metric("modern_2023_apol_exclude_nglr1", "weighted_rms_ns_bin_rms")
    op_modern_floor = _op_metric("modern_2023_apol_exclude_nglr1", "model_floor_ns_for_chi2eq1")

    nglr1_delta = None
    # 条件分岐: `op_all_wrms is not None and op_ex_nglr1_wrms is not None` を満たす経路を評価する。
    if op_all_wrms is not None and op_ex_nglr1_wrms is not None:
        nglr1_delta = float(op_all_wrms - op_ex_nglr1_wrms)

    nglr1_pass = None if nglr1_delta is None else (nglr1_delta >= 0.0)

    modern_floor_pass = None
    # 条件分岐: `op_modern_wrms is not None and op_modern_floor is not None` を満たす経路を評価する。
    if op_modern_wrms is not None and op_modern_floor is not None:
        modern_floor_pass = bool(op_modern_wrms <= op_modern_floor)

    precision_status = "unknown"
    precision_decision = None
    precision_watch_checks: List[str] = []
    precision_watch_reasons: List[str] = []
    # 条件分岐: `isinstance(precision_reaudit, dict)` を満たす経路を評価する。
    if isinstance(precision_reaudit, dict):
        precision_status = str(precision_reaudit.get("overall_status") or "unknown")
        precision_decision = precision_reaudit.get("decision")
        checks = precision_reaudit.get("checks")
        # 条件分岐: `isinstance(checks, list)` を満たす経路を評価する。
        if isinstance(checks, list):
            for c in checks:
                # 条件分岐: `not isinstance(c, dict)` を満たす経路を評価する。
                if not isinstance(c, dict):
                    continue

                # 条件分岐: `str(c.get("status") or "").lower() == "watch"` を満たす経路を評価する。

                if str(c.get("status") or "").lower() == "watch":
                    cid = str(c.get("id") or "").strip()
                    # 条件分岐: `cid` を満たす経路を評価する。
                    if cid:
                        precision_watch_checks.append(cid)

        reasons = precision_reaudit.get("likely_missing_or_next_items")
        # 条件分岐: `isinstance(reasons, list)` を満たす経路を評価する。
        if isinstance(reasons, list):
            precision_watch_reasons = [str(v) for v in reasons if str(v).strip()]

    apol_feasibility_decision = "unknown"
    apol_code_present = None
    # 条件分岐: `isinstance(apol_pos_eop_feasibility, dict)` を満たす経路を評価する。
    if isinstance(apol_pos_eop_feasibility, dict):
        apol_feasibility_decision = str(apol_pos_eop_feasibility.get("decision") or "unknown")
        metrics = apol_pos_eop_feasibility.get("metrics")
        # 条件分岐: `isinstance(metrics, dict)` を満たす経路を評価する。
        if isinstance(metrics, dict):
            v = metrics.get("apol_code_present_in_cache")
            apol_code_present = bool(v) if v is not None else None

    apol_primary_route_decision = "unknown"
    apol_primary_xyz_available = None
    apol_primary_xyz_source_groups: Optional[Dict[str, Any]] = None
    apol_primary_merge_route_status: Optional[str] = None
    apol_primary_merge_route_resolved_source_group: Optional[str] = None
    apol_primary_earthdata_auth_status: Optional[str] = None
    # 条件分岐: `isinstance(apol_primary_coord_route, dict)` を満たす経路を評価する。
    if isinstance(apol_primary_coord_route, dict):
        apol_primary_route_decision = str(apol_primary_coord_route.get("decision") or "unknown")
        metrics = apol_primary_coord_route.get("metrics")
        # 条件分岐: `isinstance(metrics, dict)` を満たす経路を評価する。
        if isinstance(metrics, dict):
            xyz_any = metrics.get("apol_xyz_available_any_source")
            # 条件分岐: `xyz_any is None` を満たす経路を評価する。
            if xyz_any is None:
                n_xyz = _as_float(metrics.get("n_logs_with_xyz"))
                apol_primary_xyz_available = bool(n_xyz is not None and n_xyz > 0.0)
            else:
                apol_primary_xyz_available = bool(xyz_any)

            group_counts = metrics.get("source_group_xyz_counts")
            # 条件分岐: `isinstance(group_counts, dict)` を満たす経路を評価する。
            if isinstance(group_counts, dict):
                apol_primary_xyz_source_groups = group_counts

            apol_primary_merge_route_status = (
                str(metrics.get("apol_deterministic_merge_route_status"))
                if metrics.get("apol_deterministic_merge_route_status") is not None
                else None
            )
            apol_primary_merge_route_resolved_source_group = (
                str(metrics.get("apol_deterministic_merge_route_resolved_source_group"))
                if metrics.get("apol_deterministic_merge_route_resolved_source_group") is not None
                else None
            )

        route = apol_primary_coord_route.get("deterministic_merge_route")
        # 条件分岐: `isinstance(route, dict)` を満たす経路を評価する。
        if isinstance(route, dict):
            # 条件分岐: `apol_primary_merge_route_status is None and route.get("status") is not None` を満たす経路を評価する。
            if apol_primary_merge_route_status is None and route.get("status") is not None:
                apol_primary_merge_route_status = str(route.get("status"))

            # 条件分岐: `apol_primary_merge_route_resolved_source_group is None and route.get("resolve...` を満たす経路を評価する。

            if apol_primary_merge_route_resolved_source_group is None and route.get("resolved_source_group") is not None:
                apol_primary_merge_route_resolved_source_group = str(route.get("resolved_source_group"))

            auth = route.get("earthdata_auth")
            # 条件分岐: `isinstance(auth, dict) and auth.get("status") is not None` を満たす経路を評価する。
            if isinstance(auth, dict) and auth.get("status") is not None:
                apol_primary_earthdata_auth_status = str(auth.get("status"))

        # 条件分岐: `apol_primary_earthdata_auth_status is None` を満たす経路を評価する。

        if apol_primary_earthdata_auth_status is None:
            inputs = apol_primary_coord_route.get("inputs")
            # 条件分岐: `isinstance(inputs, dict)` を満たす経路を評価する。
            if isinstance(inputs, dict):
                auth = inputs.get("earthdata_auth")
                # 条件分岐: `isinstance(auth, dict) and auth.get("status") is not None` を満たす経路を評価する。
                if isinstance(auth, dict) and auth.get("status") is not None:
                    apol_primary_earthdata_auth_status = str(auth.get("status"))

    time_tag_mode_diff_status: Optional[str] = None
    time_tag_mode_diff_decision: Optional[str] = None
    time_tag_mode_diff_max_abs: Optional[float] = None
    time_tag_mode_diff_changed_rows: Optional[int] = None
    # 条件分岐: `isinstance(time_tag_mode_diff_audit, dict)` を満たす経路を評価する。
    if isinstance(time_tag_mode_diff_audit, dict):
        # 条件分岐: `time_tag_mode_diff_audit.get("status") is not None` を満たす経路を評価する。
        if time_tag_mode_diff_audit.get("status") is not None:
            time_tag_mode_diff_status = str(time_tag_mode_diff_audit.get("status"))

        # 条件分岐: `time_tag_mode_diff_audit.get("decision") is not None` を満たす経路を評価する。

        if time_tag_mode_diff_audit.get("decision") is not None:
            time_tag_mode_diff_decision = str(time_tag_mode_diff_audit.get("decision"))

        metrics_diff = time_tag_mode_diff_audit.get("metrics_diff")
        # 条件分岐: `isinstance(metrics_diff, dict)` を満たす経路を評価する。
        if isinstance(metrics_diff, dict):
            time_tag_mode_diff_max_abs = _as_float(metrics_diff.get("max_abs_delta_rms_ns"))
            try:
                # 条件分岐: `metrics_diff.get("changed_rows_over_tol") is not None` を満たす経路を評価する。
                if metrics_diff.get("changed_rows_over_tol") is not None:
                    time_tag_mode_diff_changed_rows = int(metrics_diff.get("changed_rows_over_tol"))
            except Exception:
                time_tag_mode_diff_changed_rows = None

    criteria: List[Dict[str, Any]] = [
        {
            "id": "llr_operational_modern_within_inferred_floor",
            "title": "現代APOL（nglr1除外）の weighted RMS が推定model floor以内であること",
            "value": op_modern_wrms,
            "op": "<=",
            "threshold": op_modern_floor,
            "pass": modern_floor_pass,
            "gate": True,
            "unit": "ns",
            "rationale": "中央値RMSではなく、NP bin RMS重みの運用指標で採否を判定する。",
        },
        {
            "id": "llr_operational_nglr1_not_artificially_improving",
            "title": "nglr1混在が weighted RMS を人工的に下げていないこと",
            "value": nglr1_delta,
            "op": ">=",
            "threshold": 0.0,
            "pass": nglr1_pass,
            "gate": True,
            "unit": "ns (all - exclude_nglr1)",
            "rationale": "ターゲット混在で見かけ上の改善が出る場合を排除する。",
        },
        {
            "id": "llr_shapiro_improves",
            "title": "Shapiro項を除去するとRMSが悪化すること（中央値）",
            "value": ratio,
            "op": "<=",
            "threshold": shapiro_threshold,
            "pass": shapiro_pass,
            "gate": False,
            "unit": "(tropo+tide)/(tropo+no_shapiro)",
            "rationale": "Shapiro項の符号/係数監査として保持（補助指標）。",
        },
        {
            "id": "llr_precision_watch_freeze",
            "title": "LLR精度再監査のwatch要因をpack内で凍結し、解除条件を明示すること",
            "value": {
                "overall_status": precision_status,
                "watch_checks": precision_watch_checks,
            },
            "op": "informational",
            "threshold": "watch要因と解除条件が decision.precision_watch_release_conditions に記録されること",
            "pass": bool(isinstance(precision_reaudit, dict)),
            "gate": False,
            "unit": None,
            "rationale": "coverage不足や未導入補正を運用上の監視理由として固定し、主ゲート（operational）と分離する。",
        },
        {
            "id": "llr_time_tag_auto_tx_equivalence",
            "title": "time-tag auto と tx の full-batch 差分が許容内であること",
            "value": {
                "decision": time_tag_mode_diff_decision,
                "max_abs_delta_rms_ns": time_tag_mode_diff_max_abs,
                "changed_rows_over_tol": time_tag_mode_diff_changed_rows,
            },
            "op": "==",
            "threshold": {"decision": "equivalent", "changed_rows_over_tol": 0},
            "pass": bool(
                isinstance(time_tag_mode_diff_audit, dict)
                and time_tag_mode_diff_decision == "equivalent"
                and (time_tag_mode_diff_changed_rows == 0)
            ),
            "gate": False,
            "unit": None,
            "rationale": "運用モード由来の系統（auto選択）が tx 固定と差分ゼロであることを補助監査として固定する。",
        },
    ]

    primary_gate_ids = [
        "llr_operational_modern_within_inferred_floor",
        "llr_operational_nglr1_not_artificially_improving",
    ]
    primary_states = [
        c.get("pass")
        for c in criteria
        if c.get("id") in primary_gate_ids and bool(c.get("gate"))
    ]
    primary_gate_pass = bool(primary_states) and all(v is True for v in primary_states)
    support_gate_pass = bool(shapiro_pass is True)
    overall_status = "pass" if primary_gate_pass else "watch"

    pack: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "domain": "llr",
        "version": 4,
        "intent": "Promote LLR batch audit outputs, keep operational weighted metrics as primary gate, freeze precision re-audit watch reasons, and include auto-vs-tx time-tag mode audit.",
        "inputs": {
            "batch_summary": {
                "generated_utc": batch_summary.get("generated_utc"),
                "n_files": batch_summary.get("n_files"),
                "n_points_total": batch_summary.get("n_points_total"),
                "stations": batch_summary.get("stations"),
                "targets": batch_summary.get("targets"),
                "beta": batch_summary.get("beta"),
                "time_tag_mode": batch_summary.get("time_tag_mode"),
                "station_coords_mode": batch_summary.get("station_coords_mode"),
                "outlier_gate": {
                    "clip_sigma": batch_summary.get("outlier_clip_sigma"),
                    "clip_min_ns": batch_summary.get("outlier_clip_ns"),
                },
            },
            "time_tag_best_by_station": {
                "selection_metric": time_tag_best.get("selection_metric"),
                "best_mode_by_station": time_tag_best.get("best_mode_by_station"),
            },
            "outliers_diagnosis_summary": {
                "n_outliers": outliers_diag.get("n_outliers"),
                "by_cause_hint": outliers_diag.get("by_cause_hint"),
                "time_tag_sensitivity": outliers_diag.get("time_tag_sensitivity"),
                "target_mixing_sensitivity": outliers_diag.get("target_mixing_sensitivity"),
            },
            "operational_metrics_audit": {
                "generated_utc": (operational_metrics or {}).get("generated_utc")
                if isinstance(operational_metrics, dict)
                else None,
                "exclude_target": ((operational_metrics or {}).get("inputs") or {}).get("exclude_target")
                if isinstance(operational_metrics, dict)
                else None,
                "modern_start_year": ((operational_metrics or {}).get("inputs") or {}).get("modern_start_year")
                if isinstance(operational_metrics, dict)
                else None,
            },
            "precision_reaudit": {
                "generated_utc": (precision_reaudit or {}).get("generated_utc")
                if isinstance(precision_reaudit, dict)
                else None,
                "overall_status": precision_status if isinstance(precision_reaudit, dict) else None,
                "decision": precision_decision if isinstance(precision_reaudit, dict) else None,
            },
            "apol_pos_eop_feasibility": {
                "generated_utc": (apol_pos_eop_feasibility or {}).get("generated_utc")
                if isinstance(apol_pos_eop_feasibility, dict)
                else None,
                "decision": apol_feasibility_decision if isinstance(apol_pos_eop_feasibility, dict) else None,
                "apol_code_present_in_cache": apol_code_present,
            },
            "apol_primary_coord_route": {
                "generated_utc": (apol_primary_coord_route or {}).get("generated_utc")
                if isinstance(apol_primary_coord_route, dict)
                else None,
                "decision": apol_primary_route_decision if isinstance(apol_primary_coord_route, dict) else None,
                "apol_primary_xyz_available": apol_primary_xyz_available,
                "apol_primary_xyz_source_groups": apol_primary_xyz_source_groups,
                "apol_primary_merge_route_status": apol_primary_merge_route_status,
                "apol_primary_merge_route_resolved_source_group": apol_primary_merge_route_resolved_source_group,
                "apol_primary_earthdata_auth_status": apol_primary_earthdata_auth_status,
            },
            "time_tag_mode_diff_audit": {
                "generated_utc": (time_tag_mode_diff_audit or {}).get("generated_utc")
                if isinstance(time_tag_mode_diff_audit, dict)
                else None,
                "status": time_tag_mode_diff_status if isinstance(time_tag_mode_diff_audit, dict) else None,
                "decision": time_tag_mode_diff_decision if isinstance(time_tag_mode_diff_audit, dict) else None,
                "max_abs_delta_rms_ns": time_tag_mode_diff_max_abs,
                "changed_rows_over_tol": time_tag_mode_diff_changed_rows,
            },
        },
        "metrics": {
            "median_rms_ns": {
                "station_reflector": _as_float(med.get("station_reflector") if isinstance(med, dict) else None),
                "station_reflector_tropo": _as_float(med.get("station_reflector_tropo") if isinstance(med, dict) else None),
                "station_reflector_tropo_tide": rms_tide,
                "station_reflector_tropo_no_shapiro": rms_no_shapiro,
            },
            "point_weighted_rms_ns": (
                batch_summary.get("point_weighted_rms_ns")
                if isinstance(batch_summary.get("point_weighted_rms_ns"), dict)
                else None
            ),
            "modern_subset_rms_ns": (
                batch_summary.get("modern_subset_rms_ns")
                if isinstance(batch_summary.get("modern_subset_rms_ns"), dict)
                else None
            ),
            "shapiro_rms_ratio": {
                "value": ratio,
                "unit": "(tropo+tide)/(tropo+no_shapiro)",
            },
            "operational_metrics_ns": {
                "all_weighted_rms": op_all_wrms,
                "exclude_nglr1_weighted_rms": op_ex_nglr1_wrms,
                "modern_apol_ex_nglr1_weighted_rms": op_modern_wrms,
                "modern_apol_ex_nglr1_model_floor": op_modern_floor,
                "nglr1_mixing_delta_all_minus_excluded": nglr1_delta,
            },
        },
        "criteria": criteria,
        "decision": {
            "primary_gate_basis": "operational_weighted_metrics",
            "primary_gate_ids": primary_gate_ids,
            "primary_gate_pass": primary_gate_pass,
            "support_gate_ids": ["llr_shapiro_improves"],
            "support_gate_pass": support_gate_pass,
            "precision_reaudit_status": precision_status,
            "precision_reaudit_watch_checks": precision_watch_checks,
            "precision_watch_release_conditions": {
                "nglr1_coverage_gate": "nglr1 点数が nglr1_min_points 以上",
                "apol_modern_operational_gate": "modern APOL ex-nglr1 weighted RMS が inferred floor 以内",
                "station_data_balance": "局偏在が dominance 閾値以内、または bias補正 gain > 0 を維持",
                "summary_metric_semantics": "NP不確かさ復元 coverage が閾値以上",
                "iers_unified_modeling": "全局で局座標/EOP/遅延補正のIERS統一を適用した再監査を記録",
            },
            "precision_watch_reasons": precision_watch_reasons,
            "apol_pos_eop_feasibility_status": apol_feasibility_decision,
            "apol_pos_eop_code_present_in_cache": apol_code_present,
            "apol_primary_coord_route_status": apol_primary_route_decision,
            "apol_primary_coord_xyz_available": apol_primary_xyz_available,
            "apol_primary_coord_xyz_source_groups": apol_primary_xyz_source_groups,
            "apol_primary_coord_merge_route_status": apol_primary_merge_route_status,
            "apol_primary_coord_merge_route_resolved_source_group": apol_primary_merge_route_resolved_source_group,
            "apol_primary_coord_earthdata_auth_status": apol_primary_earthdata_auth_status,
            "time_tag_mode_diff_status": time_tag_mode_diff_status,
            "time_tag_mode_diff_decision": time_tag_mode_diff_decision,
            "time_tag_mode_diff_max_abs_delta_rms_ns": time_tag_mode_diff_max_abs,
            "time_tag_mode_diff_changed_rows_over_tol": time_tag_mode_diff_changed_rows,
            "overall_status": overall_status,
        },
        "artifacts": {
            "dir": _rel(out_dir),
            "copied": copied,
        },
    }
    return pack


def run(
    *,
    in_dir: Path,
    operational_dir: Path,
    precision_dir: Path,
    out_dir: Path,
    overwrite: bool,
) -> Tuple[Path, List[str]]:
    warnings: List[str] = []

    required = {
        "llr_batch_summary.json": in_dir / "llr_batch_summary.json",
        "llr_time_tag_best_by_station.json": in_dir / "llr_time_tag_best_by_station.json",
        "llr_outliers_diagnosis_summary.json": in_dir / "llr_outliers_diagnosis_summary.json",
    }
    for name, path in required.items():
        # 条件分岐: `not path.exists()` を満たす経路を評価する。
        if not path.exists():
            raise FileNotFoundError(f"missing required input: {name} ({_rel(path)})")

    batch_summary = _read_json(required["llr_batch_summary.json"])
    time_tag_best = _read_json(required["llr_time_tag_best_by_station.json"])
    outliers_diag = _read_json(required["llr_outliers_diagnosis_summary.json"])
    operational_metrics_path = operational_dir / "llr_operational_metrics_audit.json"
    operational_metrics: Optional[Dict[str, Any]] = None
    # 条件分岐: `operational_metrics_path.exists()` を満たす経路を評価する。
    if operational_metrics_path.exists():
        operational_metrics = _read_json(operational_metrics_path)
    else:
        warnings.append(f"missing optional input: {_rel(operational_metrics_path)}")

    precision_reaudit_path = precision_dir / "llr_precision_reaudit.json"
    precision_reaudit: Optional[Dict[str, Any]] = None
    # 条件分岐: `precision_reaudit_path.exists()` を満たす経路を評価する。
    if precision_reaudit_path.exists():
        precision_reaudit = _read_json(precision_reaudit_path)
    else:
        warnings.append(f"missing optional input: {_rel(precision_reaudit_path)}")

    apol_pos_eop_feasibility_path = operational_dir / "llr_apol_pos_eop_feasibility_audit.json"
    apol_pos_eop_feasibility: Optional[Dict[str, Any]] = None
    # 条件分岐: `apol_pos_eop_feasibility_path.exists()` を満たす経路を評価する。
    if apol_pos_eop_feasibility_path.exists():
        apol_pos_eop_feasibility = _read_json(apol_pos_eop_feasibility_path)
    else:
        warnings.append(f"missing optional input: {_rel(apol_pos_eop_feasibility_path)}")

    apol_primary_coord_route_path = operational_dir / "llr_apol_primary_coord_route_audit.json"
    apol_primary_coord_route: Optional[Dict[str, Any]] = None
    # 条件分岐: `apol_primary_coord_route_path.exists()` を満たす経路を評価する。
    if apol_primary_coord_route_path.exists():
        apol_primary_coord_route = _read_json(apol_primary_coord_route_path)
    else:
        warnings.append(f"missing optional input: {_rel(apol_primary_coord_route_path)}")

    time_tag_mode_diff_path = precision_dir / "llr_time_tag_mode_auto_vs_tx_audit.json"
    time_tag_mode_diff_audit: Optional[Dict[str, Any]] = None
    # 条件分岐: `time_tag_mode_diff_path.exists()` を満たす経路を評価する。
    if time_tag_mode_diff_path.exists():
        time_tag_mode_diff_audit = _read_json(time_tag_mode_diff_path)
    else:
        warnings.append(f"missing optional input: {_rel(time_tag_mode_diff_path)}")

    # Copy a minimal set of artifacts used as audit evidence.

    out_dir.mkdir(parents=True, exist_ok=True)
    copy_specs: List[Tuple[Path, Path]] = []

    # Small tables (optional but useful)
    for fname in (
        "llr_batch_summary.json",
        "llr_batch_metrics.csv",
        "llr_time_tag_best_by_station.json",
        "llr_outliers_summary.json",
        "llr_outliers_diagnosis_summary.json",
        "llr_station_diagnostics.json",
    ):
        src = in_dir / fname
        # 条件分岐: `src.exists()` を満たす経路を評価する。
        if src.exists():
            copy_specs.append((src, out_dir / fname))
        else:
            warnings.append(f"missing optional input: {_rel(src)}")

    for fname in (
        "llr_operational_metrics_audit.json",
        "llr_operational_metrics_audit.csv",
        "llr_operational_metrics_by_station.csv",
        "llr_operational_metrics_by_target.csv",
        "llr_operational_metrics_audit.png",
    ):
        src = operational_dir / fname
        # 条件分岐: `src.exists()` を満たす経路を評価する。
        if src.exists():
            copy_specs.append((src, out_dir / fname))
        else:
            warnings.append(f"missing optional operational artifact: {_rel(src)}")

    for fname in (
        "llr_precision_reaudit.json",
        "llr_precision_reaudit.csv",
        "llr_precision_reaudit.png",
        "llr_time_tag_mode_auto_vs_tx_audit.json",
        "llr_time_tag_mode_auto_vs_tx_diff.csv",
        "llr_time_tag_mode_auto_vs_tx_audit.png",
    ):
        src = precision_dir / fname
        # 条件分岐: `src.exists()` を満たす経路を評価する。
        if src.exists():
            copy_specs.append((src, out_dir / fname))
        else:
            warnings.append(f"missing optional precision-reaudit artifact: {_rel(src)}")

    for fname in (
        "llr_apol_pos_eop_feasibility_audit.json",
        "llr_apol_pos_eop_feasibility_audit.csv",
        "llr_apol_pos_eop_feasibility_audit.png",
        "llr_apol_primary_coord_route_audit.json",
        "llr_apol_primary_coord_route_audit.csv",
        "llr_apol_primary_coord_route_audit.png",
        "llr_apol_primary_coord_merge_route.json",
    ):
        src = operational_dir / fname
        # 条件分岐: `src.exists()` を満たす経路を評価する。
        if src.exists():
            copy_specs.append((src, out_dir / fname))
        else:
            warnings.append(f"missing optional APOL feasibility artifact: {_rel(src)}")

    # Key figures (keep small; omit the huge per-point CSV)

    for fname in (
        "llr_residual_distribution.png",
        "llr_rms_improvement_overall.png",
        "llr_rms_ablations_overall.png",
        "llr_shapiro_ablations_overall.png",
        "llr_tide_ablations_overall.png",
        "llr_time_tag_selection_by_station.png",
        "llr_outliers_overview.png",
        "llr_outliers_time_tag_sensitivity.png",
        "llr_station_coord_delta_pos_eop.png",
    ):
        src = in_dir / fname
        # 条件分岐: `src.exists()` を満たす経路を評価する。
        if src.exists():
            copy_specs.append((src, out_dir / fname))
        else:
            warnings.append(f"missing optional figure: {_rel(src)}")

    copied: Dict[str, Dict[str, Any]] = {}
    for src, dst in copy_specs:
        _copy_file(src, dst, overwrite=overwrite)
        copied[str(dst.name)] = _artifact_record(dst)

    pack = _build_pack(
        batch_summary=batch_summary,
        time_tag_best=time_tag_best,
        outliers_diag=outliers_diag,
        operational_metrics=operational_metrics,
        precision_reaudit=precision_reaudit,
        apol_pos_eop_feasibility=apol_pos_eop_feasibility,
        apol_primary_coord_route=apol_primary_coord_route,
        time_tag_mode_diff_audit=time_tag_mode_diff_audit,
        out_dir=out_dir,
        copied=copied,
    )
    pack_path = out_dir / "llr_falsification_pack.json"
    pack_path.write_text(
        json.dumps(pack, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # 条件分岐: `worklog is not None` を満たす経路を評価する。
    if worklog is not None:
        try:
            worklog.append_event(
                {
                    "event_type": "llr_falsification_pack_public",
                    "inputs": {
                        **{k: _rel(v) for k, v in required.items()},
                        "llr_operational_metrics_audit.json": _rel(operational_metrics_path),
                        "llr_precision_reaudit.json": _rel(precision_reaudit_path),
                        "llr_apol_pos_eop_feasibility_audit.json": _rel(apol_pos_eop_feasibility_path),
                        "llr_apol_primary_coord_route_audit.json": _rel(apol_primary_coord_route_path),
                        "llr_time_tag_mode_auto_vs_tx_audit.json": _rel(time_tag_mode_diff_path),
                    },
                    "outputs": {"llr_falsification_pack_json": _rel(pack_path), "public_llr_dir": _rel(out_dir)},
                    "warnings": warnings,
                }
            )
        except Exception:
            pass

    return pack_path, warnings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote LLR batch audit outputs into output/public/llr.")
    parser.add_argument(
        "--in-dir",
        default="output/private/llr/batch",
        help="Input directory (default: output/private/llr/batch).",
    )
    parser.add_argument(
        "--out-dir",
        default="output/public/llr",
        help="Output directory (default: output/public/llr).",
    )
    parser.add_argument(
        "--operational-dir",
        default="output/private/llr",
        help="Operational metrics directory (default: output/private/llr).",
    )
    parser.add_argument(
        "--precision-dir",
        default="output/private/llr",
        help="Precision re-audit directory (default: output/private/llr).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    in_dir = Path(str(args.in_dir))
    # 条件分岐: `not in_dir.is_absolute()` を満たす経路を評価する。
    if not in_dir.is_absolute():
        in_dir = _ROOT / in_dir

    out_dir = Path(str(args.out_dir))
    # 条件分岐: `not out_dir.is_absolute()` を満たす経路を評価する。
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir

    operational_dir = Path(str(args.operational_dir))
    # 条件分岐: `not operational_dir.is_absolute()` を満たす経路を評価する。
    if not operational_dir.is_absolute():
        operational_dir = _ROOT / operational_dir

    precision_dir = Path(str(args.precision_dir))
    # 条件分岐: `not precision_dir.is_absolute()` を満たす経路を評価する。
    if not precision_dir.is_absolute():
        precision_dir = _ROOT / precision_dir

    try:
        pack_path, warnings = run(
            in_dir=in_dir,
            operational_dir=operational_dir,
            precision_dir=precision_dir,
            out_dir=out_dir,
            overwrite=bool(args.overwrite),
        )
    except FileNotFoundError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    for w in warnings:
        print(f"[warn] {w}", file=sys.stderr)

    print(f"[ok] wrote: {_rel(pack_path)}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
