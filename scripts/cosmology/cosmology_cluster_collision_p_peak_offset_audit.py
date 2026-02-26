#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_cluster_collision_p_peak_offset_audit.py

Step 8.7.25（優先度S2）:
銀河団衝突（Bullet系）での Pピーク-ガス分離監査を、
公開可能な固定I/F（JSON/CSV/PNG）として初版実装する。

目的:
- 観測されたレンズ重心-ガス重心オフセットに対して、
  baryon-only（Pピーク=ガス）と P補正モデルを同一I/Fで比較する。
- Δx_P-gas / Δx_P-lens を明示し、pass/watch/reject 判定を固定する。

固定出力:
- output/public/cosmology/cosmology_cluster_collision_p_peak_offset_audit.json
- output/public/cosmology/cosmology_cluster_collision_p_peak_offset_audit.csv
- output/public/cosmology/cosmology_cluster_collision_p_peak_offset_audit.png
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.summary import worklog  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    worklog = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


PRIMARY_REFERENCE_COLUMNS: Tuple[str, str] = ("lens_map_ref", "xray_map_ref")
PRIMARY_MODE_COLUMN = "source_mode"
PRIMARY_MODE_VALUE = "primary_map_registered"
PRIMARY_TEMPLATE_COLUMNS: Tuple[str, ...] = (
    "cluster_id",
    "label",
    "lens_gas_offset_kpc_obs",
    "lens_gas_offset_kpc_sigma",
    "lens_map_ref",
    "xray_map_ref",
    "source_mode",
    "note",
)


@dataclass(frozen=True)
class CollisionObs:
    cluster_id: str
    label: str
    lens_gas_offset_kpc_obs: float
    lens_gas_offset_kpc_sigma: float


def _write_dict_rows_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            # 条件分岐: `not chunk` を満たす経路を評価する。
            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest().upper()


def _file_signature(path: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"exists": bool(path.exists()), "path": _rel(path)}
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return payload

    try:
        stat = path.stat()
        payload["size_bytes"] = int(stat.st_size)
        payload["mtime_utc"] = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    except Exception:
        pass

    try:
        payload["sha256"] = _sha256_file(path)
    except Exception:
        pass

    return payload


def _load_previous_watchpack(path: Path) -> Dict[str, Any]:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    diagnostics = payload.get("diagnostics")
    # 条件分岐: `not isinstance(diagnostics, dict)` を満たす経路を評価する。
    if not isinstance(diagnostics, dict):
        return {}

    watchpack = diagnostics.get("primary_map_update_watchpack")
    # 条件分岐: `not isinstance(watchpack, dict)` を満たす経路を評価する。
    if not isinstance(watchpack, dict):
        return {}

    return watchpack


def _set_japanese_font() -> None:
    # 条件分岐: `plt is None` を満たす経路を評価する。
    if plt is None:
        return

    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = ["Yu Gothic", "Meiryo", "BIZ UDGothic", "MS Gothic", "Yu Mincho", "MS Mincho"]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        return


def _load_collision_observations(csv_path: Path) -> Tuple[List[CollisionObs], str, List[str], Dict[str, Any]]:
    notes: List[str] = []
    source_diag: Dict[str, Any] = {
        "input_csv_exists": bool(csv_path.exists()),
        "fieldnames": [],
        "required_primary_reference_columns": list(PRIMARY_REFERENCE_COLUMNS),
        "required_primary_mode_column": PRIMARY_MODE_COLUMN,
        "n_valid_rows": 0,
        "n_rows_with_primary_refs": 0,
        "n_rows_source_mode_primary_flag": 0,
        "primary_reference_columns_present": False,
        "primary_reference_columns_missing": list(PRIMARY_REFERENCE_COLUMNS),
        "primary_mode_column_present": False,
        "rows_missing_primary_refs": [],
        "rows_source_mode_not_primary": [],
        "mode_decision_reason": "fallback_embedded_proxy",
    }
    # 条件分岐: `csv_path.exists()` を満たす経路を評価する。
    if csv_path.exists():
        rows: List[CollisionObs] = []
        n_rows_with_primary_refs = 0
        n_rows_mode_primary = 0
        rows_missing_primary_refs: List[str] = []
        rows_source_mode_not_primary: List[str] = []
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = [str(name).strip() for name in (reader.fieldnames or [])]
            fieldnames_set = set(fieldnames)
            source_diag["fieldnames"] = fieldnames
            source_diag["primary_reference_columns_missing"] = [
                col for col in PRIMARY_REFERENCE_COLUMNS if col not in fieldnames_set
            ]
            source_diag["primary_reference_columns_present"] = len(source_diag["primary_reference_columns_missing"]) == 0
            source_diag["primary_mode_column_present"] = PRIMARY_MODE_COLUMN in fieldnames_set
            for row in reader:
                try:
                    cluster_id = str(row.get("cluster_id", "")).strip()
                    label = str(row.get("label", cluster_id)).strip() or cluster_id
                    offset_kpc = float(row.get("lens_gas_offset_kpc_obs", "nan"))
                    sigma_kpc = float(row.get("lens_gas_offset_kpc_sigma", "nan"))
                except Exception:
                    continue

                if (
                    not cluster_id
                    or not np.isfinite(offset_kpc)
                    or not np.isfinite(sigma_kpc)
                    or sigma_kpc <= 0.0
                    or offset_kpc == 0.0
                ):
                    continue

                rows.append(
                    CollisionObs(
                        cluster_id=cluster_id,
                        label=label,
                        lens_gas_offset_kpc_obs=float(offset_kpc),
                        lens_gas_offset_kpc_sigma=float(sigma_kpc),
                    )
                )
                lens_ref = str(row.get(PRIMARY_REFERENCE_COLUMNS[0], "")).strip()
                xray_ref = str(row.get(PRIMARY_REFERENCE_COLUMNS[1], "")).strip()
                source_mode_flag = str(row.get(PRIMARY_MODE_COLUMN, "")).strip().lower()
                # 条件分岐: `lens_ref and xray_ref` を満たす経路を評価する。
                if lens_ref and xray_ref:
                    n_rows_with_primary_refs += 1
                else:
                    rows_missing_primary_refs.append(cluster_id)

                # 条件分岐: `source_mode_flag == PRIMARY_MODE_VALUE` を満たす経路を評価する。

                if source_mode_flag == PRIMARY_MODE_VALUE:
                    n_rows_mode_primary += 1
                else:
                    rows_source_mode_not_primary.append(cluster_id)

        # 条件分岐: `rows` を満たす経路を評価する。

        if rows:
            n_valid = len(rows)
            source_diag["n_valid_rows"] = int(n_valid)
            source_diag["n_rows_with_primary_refs"] = int(n_rows_with_primary_refs)
            source_diag["n_rows_source_mode_primary_flag"] = int(n_rows_mode_primary)
            source_diag["rows_missing_primary_refs"] = rows_missing_primary_refs
            source_diag["rows_source_mode_not_primary"] = rows_source_mode_not_primary
            has_primary_cols = bool(source_diag["primary_reference_columns_present"])
            has_mode_col = bool(source_diag["primary_mode_column_present"])
            all_rows_primary = (
                has_primary_cols
                and has_mode_col
                and n_rows_with_primary_refs == n_valid
                and n_rows_mode_primary == n_valid
            )
            # 条件分岐: `all_rows_primary` を満たす経路を評価する。
            if all_rows_primary:
                mode = "primary_map_registered"
                source_diag["mode_decision_reason"] = "all_rows_have_primary_registration_fields"
            # 条件分岐: 前段条件が不成立で、`not has_primary_cols` を追加評価する。
            elif not has_primary_cols:
                mode = "proxy_table_csv"
                source_diag["mode_decision_reason"] = "missing_primary_reference_columns"
            # 条件分岐: 前段条件が不成立で、`not has_mode_col` を追加評価する。
            elif not has_mode_col:
                mode = "proxy_table_csv"
                source_diag["mode_decision_reason"] = "missing_primary_mode_column"
            # 条件分岐: 前段条件が不成立で、`n_rows_with_primary_refs < n_valid` を追加評価する。
            elif n_rows_with_primary_refs < n_valid:
                mode = "proxy_table_csv"
                source_diag["mode_decision_reason"] = "rows_missing_primary_map_refs"
            # 条件分岐: 前段条件が不成立で、`n_rows_mode_primary < n_valid` を追加評価する。
            elif n_rows_mode_primary < n_valid:
                mode = "proxy_table_csv"
                source_diag["mode_decision_reason"] = "rows_not_marked_primary_map_registered"
            else:
                mode = "proxy_table_csv"
                source_diag["mode_decision_reason"] = "proxy_rows_or_partial_primary_fields"

            return rows, mode, notes, source_diag

        notes.append("Input CSV exists but no valid rows were found; fallback proxy table is used.")

    fallback = [
        CollisionObs("bullet_main", "Bullet main halo", 190.0, 25.0),
        CollisionObs("bullet_sub", "Bullet sub-halo", 140.0, 20.0),
    ]
    notes.append("Fallback proxy offsets were used (public summary scale).")
    source_diag["n_valid_rows"] = int(len(fallback))
    source_diag["mode_decision_reason"] = "fallback_embedded_proxy"
    return fallback, "embedded_proxy", notes, source_diag


def _derive_primary_registration_readiness(source_diag: Dict[str, Any]) -> Dict[str, Any]:
    n_valid_rows = int(source_diag.get("n_valid_rows", 0) or 0)
    missing_columns_raw = source_diag.get("primary_reference_columns_missing")
    missing_columns: List[str] = list(missing_columns_raw) if isinstance(missing_columns_raw, list) else []
    # 条件分岐: `not bool(source_diag.get("primary_mode_column_present")) and PRIMARY_MODE_COL...` を満たす経路を評価する。
    if not bool(source_diag.get("primary_mode_column_present")) and PRIMARY_MODE_COLUMN not in missing_columns:
        missing_columns.append(PRIMARY_MODE_COLUMN)

    rows_missing_refs_raw = source_diag.get("rows_missing_primary_refs")
    rows_missing_refs = list(rows_missing_refs_raw) if isinstance(rows_missing_refs_raw, list) else []
    rows_mode_not_primary_raw = source_diag.get("rows_source_mode_not_primary")
    rows_mode_not_primary = list(rows_mode_not_primary_raw) if isinstance(rows_mode_not_primary_raw, list) else []

    # 条件分岐: `n_valid_rows <= 0` を満たす経路を評価する。
    if n_valid_rows <= 0:
        decision_reason = "no_valid_rows"
    # 条件分岐: 前段条件が不成立で、`len(missing_columns) > 0` を追加評価する。
    elif len(missing_columns) > 0:
        decision_reason = "missing_required_columns"
    # 条件分岐: 前段条件が不成立で、`len(rows_missing_refs) > 0` を追加評価する。
    elif len(rows_missing_refs) > 0:
        decision_reason = "rows_missing_primary_refs"
    # 条件分岐: 前段条件が不成立で、`len(rows_mode_not_primary) > 0` を追加評価する。
    elif len(rows_mode_not_primary) > 0:
        decision_reason = "rows_not_marked_primary_map_registered"
    else:
        decision_reason = "ready"

    ready = decision_reason == "ready"
    # 条件分岐: `decision_reason == "missing_required_columns"` を満たす経路を評価する。
    if decision_reason == "missing_required_columns":
        next_action = "add_required_primary_columns_then_rerun_step_8_7_25"
    # 条件分岐: 前段条件が不成立で、`decision_reason == "rows_missing_primary_refs"` を追加評価する。
    elif decision_reason == "rows_missing_primary_refs":
        next_action = "fill_lens_map_ref_and_xray_map_ref_for_all_rows_then_rerun_step_8_7_25"
    # 条件分岐: 前段条件が不成立で、`decision_reason == "rows_not_marked_primary_map_registered"` を追加評価する。
    elif decision_reason == "rows_not_marked_primary_map_registered":
        next_action = "set_source_mode_primary_map_registered_for_all_rows_then_rerun_step_8_7_25"
    # 条件分岐: 前段条件が不成立で、`decision_reason == "no_valid_rows"` を追加評価する。
    elif decision_reason == "no_valid_rows":
        next_action = "populate_valid_cluster_rows_then_rerun_step_8_7_25"
    else:
        next_action = "run_step_8_7_25_primary_map_recheck_now"

    return {
        "ready": ready,
        "decision_reason": decision_reason,
        "required_columns": [*PRIMARY_REFERENCE_COLUMNS, PRIMARY_MODE_COLUMN],
        "missing_required_columns": missing_columns,
        "n_valid_rows": n_valid_rows,
        "rows_missing_primary_refs": rows_missing_refs,
        "rows_source_mode_not_primary": rows_mode_not_primary,
        "next_action": next_action,
    }


def _build_primary_registration_checklist(
    *,
    input_csv: Path,
    observations: Sequence[CollisionObs],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows_by_cluster: Dict[str, Dict[str, str]] = {}
    # 条件分岐: `input_csv.exists()` を満たす経路を評価する。
    if input_csv.exists():
        try:
            with input_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cluster_id = str(row.get("cluster_id", "")).strip()
                    # 条件分岐: `not cluster_id` を満たす経路を評価する。
                    if not cluster_id:
                        continue

                    rows_by_cluster[cluster_id] = {str(k): str(v) for k, v in row.items()}
        except Exception:
            rows_by_cluster = {}

    checklist: List[Dict[str, Any]] = []
    for obs in observations:
        cluster_id = obs.cluster_id
        row = rows_by_cluster.get(cluster_id, {})
        lens_ref = str(row.get(PRIMARY_REFERENCE_COLUMNS[0], "")).strip()
        xray_ref = str(row.get(PRIMARY_REFERENCE_COLUMNS[1], "")).strip()
        source_mode = str(row.get(PRIMARY_MODE_COLUMN, "")).strip()

        missing_fields: List[str] = []
        # 条件分岐: `not lens_ref` を満たす経路を評価する。
        if not lens_ref:
            missing_fields.append(PRIMARY_REFERENCE_COLUMNS[0])

        # 条件分岐: `not xray_ref` を満たす経路を評価する。

        if not xray_ref:
            missing_fields.append(PRIMARY_REFERENCE_COLUMNS[1])

        mode_ok = source_mode.lower() == PRIMARY_MODE_VALUE
        row_present = bool(row)
        ready = row_present and len(missing_fields) == 0 and mode_ok

        # 条件分岐: `not row_present` を満たす経路を評価する。
        if not row_present:
            action = "add_cluster_row_with_primary_refs_and_source_mode"
            status = "missing_row"
        # 条件分岐: 前段条件が不成立で、`len(missing_fields) > 0` を追加評価する。
        elif len(missing_fields) > 0:
            action = "fill_missing_primary_refs"
            status = "pending"
        # 条件分岐: 前段条件が不成立で、`not mode_ok` を追加評価する。
        elif not mode_ok:
            action = "set_source_mode_primary_map_registered"
            status = "pending"
        else:
            action = "ready"
            status = "ready"

        checklist.append(
            {
                "cluster_id": cluster_id,
                "label": obs.label,
                "row_present": row_present,
                "lens_map_ref": lens_ref,
                "xray_map_ref": xray_ref,
                "source_mode": source_mode,
                "missing_fields": ";".join(missing_fields),
                "mode_primary_ok": mode_ok,
                "ready": ready,
                "status": status,
                "required_action": action,
            }
        )

    n_total = len(checklist)
    n_ready = sum(1 for row in checklist if bool(row.get("ready")))
    n_missing_row = sum(1 for row in checklist if str(row.get("status")) == "missing_row")
    n_missing_ref = sum(1 for row in checklist if PRIMARY_REFERENCE_COLUMNS[0] in str(row.get("missing_fields")) or PRIMARY_REFERENCE_COLUMNS[1] in str(row.get("missing_fields")))
    n_mode_pending = sum(
        1
        for row in checklist
        if bool(row.get("row_present"))
        and str(row.get("missing_fields", "")).strip() == ""
        and not bool(row.get("mode_primary_ok"))
    )
    summary = {
        "n_clusters": n_total,
        "n_ready": n_ready,
        "n_pending": max(n_total - n_ready, 0),
        "n_missing_row": n_missing_row,
        "n_missing_primary_refs": n_missing_ref,
        "n_source_mode_pending": n_mode_pending,
        "overall_ready": n_total > 0 and n_ready == n_total,
        "next_action": (
            "run_step_8_7_25_primary_map_recheck_now"
            if n_total > 0 and n_ready == n_total
            else "fill_primary_registration_checklist_then_rerun_step_8_7_25"
        ),
    }
    return checklist, summary


def _derive_primary_map_update_watchpack(
    *,
    current_signature: Dict[str, Any],
    source_mode: str,
    source_diag: Dict[str, Any],
    primary_readiness: Dict[str, Any],
    previous_watchpack: Dict[str, Any],
) -> Dict[str, Any]:
    previous_signature = (
        previous_watchpack.get("input_signature")
        if isinstance(previous_watchpack.get("input_signature"), dict)
        else {}
    )
    current_sha = str(current_signature.get("sha256", "")).strip().upper()
    previous_sha = str(previous_signature.get("sha256", "")).strip().upper()
    current_exists = bool(current_signature.get("exists"))
    previous_exists = bool(previous_signature.get("exists"))
    hash_changed = current_exists and previous_exists and bool(current_sha) and bool(previous_sha) and current_sha != previous_sha

    current_mtime = str(current_signature.get("mtime_utc", "")).strip()
    previous_mtime = str(previous_signature.get("mtime_utc", "")).strip()
    current_size = current_signature.get("size_bytes")
    previous_size = previous_signature.get("size_bytes")
    metadata_changed_without_hash = False
    # 条件分岐: `current_exists and previous_exists and not hash_changed` を満たす経路を評価する。
    if current_exists and previous_exists and not hash_changed:
        # 条件分岐: `(current_mtime and previous_mtime and current_mtime != previous_mtime) or (cu...` を満たす経路を評価する。
        if (current_mtime and previous_mtime and current_mtime != previous_mtime) or (current_size != previous_size):
            metadata_changed_without_hash = True

    previous_source_mode = str(previous_watchpack.get("source_mode", "")).strip()
    source_mode_changed = bool(previous_source_mode) and (previous_source_mode != source_mode)
    gate_ready_now = bool(primary_readiness.get("ready"))
    previous_gate_ready = bool(previous_watchpack.get("gate_ready_now")) if previous_watchpack else False
    gate_state_changed = previous_gate_ready != gate_ready_now
    transitioned_to_primary = (not previous_gate_ready) and gate_ready_now
    baseline_initialized_now = current_exists and not previous_exists

    # 条件分岐: `transitioned_to_primary` を満たす経路を評価する。
    if transitioned_to_primary:
        update_event_type = "source_mode_transition_to_primary"
    # 条件分岐: 前段条件が不成立で、`hash_changed` を追加評価する。
    elif hash_changed:
        update_event_type = "input_hash_changed"
    # 条件分岐: 前段条件が不成立で、`baseline_initialized_now` を追加評価する。
    elif baseline_initialized_now:
        update_event_type = "baseline_initialized"
    # 条件分岐: 前段条件が不成立で、`metadata_changed_without_hash` を追加評価する。
    elif metadata_changed_without_hash:
        update_event_type = "metadata_changed_hash_same"
    else:
        update_event_type = "no_change"

    update_event_detected = bool(hash_changed or transitioned_to_primary)
    event_counter_prev = int(previous_watchpack.get("event_counter", 0)) if previous_watchpack else 0
    event_counter = event_counter_prev + 1 if update_event_detected else event_counter_prev

    next_action = (
        str(primary_readiness.get("next_action", "")).strip()
        if isinstance(primary_readiness.get("next_action"), str)
        else ""
    )
    # 条件分岐: `not next_action` を満たす経路を評価する。
    if not next_action:
        next_action = (
            "run_step_8_7_25_primary_map_recheck_now"
            if gate_ready_now
            else "wait_for_primary_map_registration_then_rerun_step_8_7_25"
        )

    return {
        "source_mode": source_mode,
        "source_mode_previous": previous_source_mode if previous_source_mode else None,
        "source_mode_changed": source_mode_changed,
        "gate_ready_previous": previous_gate_ready,
        "gate_ready_state_changed": gate_state_changed,
        "transitioned_to_primary_map_registered": transitioned_to_primary,
        "gate_ready_now": gate_ready_now,
        "primary_readiness_reason": primary_readiness.get("decision_reason"),
        "missing_required_columns": primary_readiness.get("missing_required_columns"),
        "rows_missing_primary_refs": primary_readiness.get("rows_missing_primary_refs"),
        "rows_source_mode_not_primary": primary_readiness.get("rows_source_mode_not_primary"),
        "input_signature": current_signature,
        "previous_input_signature": previous_signature,
        "update_event_detected": update_event_detected,
        "update_event_type": update_event_type,
        "input_hash_changed": hash_changed,
        "input_metadata_changed_without_hash_change": metadata_changed_without_hash,
        "baseline_initialized_now": baseline_initialized_now,
        "event_counter": event_counter,
        "n_valid_rows": source_diag.get("n_valid_rows"),
        "n_rows_with_primary_refs": source_diag.get("n_rows_with_primary_refs"),
        "n_rows_source_mode_primary_flag": source_diag.get("n_rows_source_mode_primary_flag"),
        "next_action": next_action,
        "note": (
            "Step 8.7.25 rerun trigger is fixed to input hash change or source-mode transition to primary map. "
            "Proxy rows keep watch status until primary-map registration is complete."
        ),
    }


def _write_primary_registration_template(path: Path, observations: Sequence[CollisionObs], source_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PRIMARY_TEMPLATE_COLUMNS))
        writer.writeheader()
        for obs in observations:
            writer.writerow(
                {
                    "cluster_id": obs.cluster_id,
                    "label": obs.label,
                    "lens_gas_offset_kpc_obs": f"{obs.lens_gas_offset_kpc_obs:.6g}",
                    "lens_gas_offset_kpc_sigma": f"{obs.lens_gas_offset_kpc_sigma:.6g}",
                    "lens_map_ref": "",
                    "xray_map_ref": "",
                    "source_mode": source_mode if source_mode else "proxy_table_csv",
                    "note": "fill lens_map_ref/xray_map_ref and set source_mode=primary_map_registered after map registration",
                }
            )


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_sparc_anchor(paths: Sequence[Path]) -> Tuple[Dict[str, Any], str]:
    for path in paths:
        # 条件分岐: `not path.exists()` を満たす経路を評価する。
        if not path.exists():
            continue

        payload = _read_json(path)
        pmodel_fixed = payload.get("pmodel_fixed") if isinstance(payload.get("pmodel_fixed"), dict) else {}
        fit_results = payload.get("fit_results") if isinstance(payload.get("fit_results"), dict) else {}
        pmodel_row = fit_results.get("pmodel_corrected") if isinstance(fit_results.get("pmodel_corrected"), dict) else {}
        baryon_row = fit_results.get("baryon_only") if isinstance(fit_results.get("baryon_only"), dict) else {}
        comparison = fit_results.get("comparison") if isinstance(fit_results.get("comparison"), dict) else {}
        a0 = float(pmodel_fixed.get("a0_m_s2", 1.0e-10))
        upsilon_p = float(pmodel_row.get("upsilon_best", 0.5))
        upsilon_b = float(baryon_row.get("upsilon_best", 0.9))
        chi2_ratio = float(comparison.get("chi2_dof_ratio_pmodel_over_baryon", 1.0))
        return {
            "a0_m_s2": a0,
            "upsilon_pmodel_best": upsilon_p,
            "upsilon_baryon_best": upsilon_b,
            "chi2_dof_ratio_pmodel_over_baryon": chi2_ratio,
            "source": _rel(path),
        }, "sparc_anchor"

    return {
        "a0_m_s2": 1.0e-10,
        "upsilon_pmodel_best": 0.55,
        "upsilon_baryon_best": 0.95,
        "chi2_dof_ratio_pmodel_over_baryon": 0.25,
        "source": "built_in_default",
    }, "built_in_default"


def _alpha_from_sparc(anchor: Dict[str, Any]) -> Dict[str, float]:
    a0 = max(float(anchor.get("a0_m_s2", 1.0e-10)), 1.0e-16)
    ups_p = float(anchor.get("upsilon_pmodel_best", 0.55))
    ups_b = max(float(anchor.get("upsilon_baryon_best", 0.95)), 1.0e-12)
    chi2_ratio = float(anchor.get("chi2_dof_ratio_pmodel_over_baryon", 0.25))

    chi_term = 0.25 * (1.0 - max(min(chi2_ratio, 1.0), 0.0))
    ml_term = 0.20 * (ups_p / ups_b)
    a0_term = 0.05 * math.tanh(math.log10(a0 / 1.0e-10))
    alpha = 0.55 + chi_term + ml_term + a0_term
    alpha = max(0.0, min(1.5, alpha))
    return {
        "alpha_from_sparc": float(alpha),
        "components": {
            "base": 0.55,
            "chi_term": float(chi_term),
            "ml_term": float(ml_term),
            "a0_term": float(a0_term),
        },
    }


def _evaluate_model(
    obs_rows: Sequence[CollisionObs],
    *,
    alpha: float,
    model_name: str,
    lens_sigma_scale: float,
    lens_sigma_floor_kpc: float,
) -> Dict[str, Any]:
    row_payloads: List[Dict[str, Any]] = []
    z_pgas: List[float] = []
    z_plens: List[float] = []
    sign_ok = 0
    for obs in obs_rows:
        obs_offset = float(obs.lens_gas_offset_kpc_obs)
        sigma_pgas = max(float(obs.lens_gas_offset_kpc_sigma), 1.0e-6)
        sigma_plens = max(float(obs.lens_gas_offset_kpc_sigma) * float(lens_sigma_scale), float(lens_sigma_floor_kpc))

        pred_delta_x_p_gas = float(alpha) * obs_offset
        pred_delta_x_p_lens = pred_delta_x_p_gas - obs_offset

        residual_pgas = pred_delta_x_p_gas - obs_offset
        residual_plens = pred_delta_x_p_lens

        z_gas = residual_pgas / sigma_pgas
        z_lens = residual_plens / sigma_plens

        z_pgas.append(float(z_gas))
        z_plens.append(float(z_lens))
        # 条件分岐: `np.sign(pred_delta_x_p_gas) == np.sign(obs_offset)` を満たす経路を評価する。
        if np.sign(pred_delta_x_p_gas) == np.sign(obs_offset):
            sign_ok += 1

        row_payloads.append(
            {
                "cluster_id": obs.cluster_id,
                "label": obs.label,
                "model": model_name,
                "obs_lens_gas_offset_kpc": obs_offset,
                "obs_lens_gas_sigma_kpc": sigma_pgas,
                "sigma_p_lens_kpc": sigma_plens,
                "pred_delta_x_p_gas_kpc": pred_delta_x_p_gas,
                "pred_delta_x_p_lens_kpc": pred_delta_x_p_lens,
                "residual_delta_x_p_gas_kpc": residual_pgas,
                "residual_delta_x_p_lens_kpc": residual_plens,
                "z_delta_x_p_gas": float(z_gas),
                "z_delta_x_p_lens": float(z_lens),
                "alpha": float(alpha),
            }
        )

    chi2 = float(np.sum(np.square(np.asarray(z_pgas + z_plens, dtype=float))))
    dof = max(1, 2 * len(obs_rows))
    return {
        "rows": row_payloads,
        "summary": {
            "chi2": chi2,
            "dof": int(dof),
            "chi2_dof": float(chi2 / float(dof)),
            "max_abs_z_p_gas": float(np.max(np.abs(np.asarray(z_pgas, dtype=float)))) if z_pgas else float("nan"),
            "max_abs_z_p_lens": float(np.max(np.abs(np.asarray(z_plens, dtype=float)))) if z_plens else float("nan"),
            "mean_abs_delta_x_p_lens_kpc": float(
                np.mean(np.abs([float(r["pred_delta_x_p_lens_kpc"]) for r in row_payloads]))
            )
            if row_payloads
            else float("nan"),
            "direction_consistency_fraction": float(sign_ok / max(len(obs_rows), 1)),
            "n_clusters": int(len(obs_rows)),
        },
    }


def _status_from_gate(passed: bool, gate_level: str) -> str:
    # 条件分岐: `passed` を満たす経路を評価する。
    if passed:
        return "pass"

    return "reject" if gate_level == "hard" else "watch"


def _score_from_status(status: str) -> float:
    # 条件分岐: `status == "pass"` を満たす経路を評価する。
    if status == "pass":
        return 1.0

    # 条件分岐: `status == "watch"` を満たす経路を評価する。

    if status == "watch":
        return 0.5

    return 0.0


def _build_checks(
    *,
    source_mode: str,
    baryon_summary: Dict[str, Any],
    pmodel_summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    def add_check(check_id: str, metric: str, value: Any, expected: str, passed: bool, gate_level: str, note: str) -> None:
        status = _status_from_gate(bool(passed), gate_level)
        checks.append(
            {
                "id": check_id,
                "metric": metric,
                "value": value,
                "expected": expected,
                "pass": bool(passed),
                "gate_level": gate_level,
                "status": status,
                "score": _score_from_status(status),
                "note": note,
            }
        )

    baryon_chi2_dof = float(baryon_summary.get("chi2_dof", float("nan")))
    pmodel_chi2_dof = float(pmodel_summary.get("chi2_dof", float("nan")))
    improvement_ratio = pmodel_chi2_dof / max(baryon_chi2_dof, 1.0e-12)
    max_abs_z_lens_p = float(pmodel_summary.get("max_abs_z_p_lens", float("nan")))

    add_check(
        "bullet::input_clusters",
        "n_clusters",
        int(pmodel_summary.get("n_clusters", 0)),
        ">=2",
        int(pmodel_summary.get("n_clusters", 0)) >= 2,
        "hard",
        "少なくとも2成分（main/sub）のオフセット監査が必要。",
    )
    add_check(
        "bullet::baryon_reject",
        "chi2_dof(baryon_only)",
        baryon_chi2_dof,
        ">=4.0",
        np.isfinite(baryon_chi2_dof) and baryon_chi2_dof >= 4.0,
        "hard",
        "baryon-only 基線がオフセットを説明できないことを確認する。",
    )
    add_check(
        "bullet::pmodel_improves",
        "chi2_dof_ratio(pmodel/baryon)",
        improvement_ratio,
        "<=0.35",
        np.isfinite(improvement_ratio) and improvement_ratio <= 0.35,
        "hard",
        "P補正で baryon-only より有意に改善すること。",
    )
    add_check(
        "bullet::direction_consistency",
        "direction_consistency_fraction(pmodel)",
        float(pmodel_summary.get("direction_consistency_fraction", 0.0)),
        "=1.0",
        abs(float(pmodel_summary.get("direction_consistency_fraction", 0.0)) - 1.0) < 1.0e-12,
        "hard",
        "Pピークと観測オフセットの向き（先行/遅延）が一致すること。",
    )
    add_check(
        "bullet::lens_anchor_gate",
        "max_abs_z(delta_x_p_lens)",
        max_abs_z_lens_p,
        "<=3.0",
        np.isfinite(max_abs_z_lens_p) and max_abs_z_lens_p <= 3.0,
        "hard",
        "観測レンズ重心に対する Pピークのズレが3σ以内。",
    )
    add_check(
        "bullet::proxy_input_watch",
        "source_mode",
        source_mode,
        "primary_map_registered",
        source_mode == "primary_map_registered",
        "watch",
        "一次マップ（κ/Σ + X線）へ切替時に pass へ昇格する運用監視。",
    )
    return checks


def _decision_from_checks(checks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    hard_fail_ids = [str(row["id"]) for row in checks if row.get("gate_level") == "hard" and row.get("pass") is not True]
    watch_ids = [str(row["id"]) for row in checks if row.get("gate_level") == "watch" and row.get("pass") is not True]
    # 条件分岐: `hard_fail_ids` を満たす経路を評価する。
    if hard_fail_ids:
        status = "reject"
        decision = "cluster_collision_p_peak_rejected"
    # 条件分岐: 前段条件が不成立で、`watch_ids` を追加評価する。
    elif watch_ids:
        status = "watch"
        decision = "cluster_collision_p_peak_watch"
    else:
        status = "pass"
        decision = "cluster_collision_p_peak_pass"

    return {
        "overall_status": status,
        "decision": decision,
        "hard_fail_ids": hard_fail_ids,
        "watch_ids": watch_ids,
        "rule": "Reject if any hard gate fails; watch if only watch gates fail; otherwise pass.",
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "cluster_id",
        "label",
        "model",
        "obs_lens_gas_offset_kpc",
        "obs_lens_gas_sigma_kpc",
        "sigma_p_lens_kpc",
        "pred_delta_x_p_gas_kpc",
        "pred_delta_x_p_lens_kpc",
        "residual_delta_x_p_gas_kpc",
        "residual_delta_x_p_lens_kpc",
        "z_delta_x_p_gas",
        "z_delta_x_p_lens",
        "alpha",
    ]
    _write_dict_rows_csv(path, fieldnames, rows)


def _plot(path: Path, *, observations: Sequence[CollisionObs], rows_baryon: Sequence[Dict[str, Any]], rows_pmodel: Sequence[Dict[str, Any]], alpha_p: float) -> None:
    # 条件分岐: `plt is None` を満たす経路を評価する。
    if plt is None:
        return

    _set_japanese_font()
    path.parent.mkdir(parents=True, exist_ok=True)

    by_cluster_b = {str(r["cluster_id"]): r for r in rows_baryon}
    by_cluster_p = {str(r["cluster_id"]): r for r in rows_pmodel}
    labels = [obs.label for obs in observations]
    y = np.arange(len(observations), dtype=float)
    obs_vals = np.asarray([obs.lens_gas_offset_kpc_obs for obs in observations], dtype=float)
    obs_err = np.asarray([obs.lens_gas_offset_kpc_sigma for obs in observations], dtype=float)
    pred_b = np.asarray([float(by_cluster_b[obs.cluster_id]["pred_delta_x_p_gas_kpc"]) for obs in observations], dtype=float)
    pred_p = np.asarray([float(by_cluster_p[obs.cluster_id]["pred_delta_x_p_gas_kpc"]) for obs in observations], dtype=float)

    z_lens_b = np.asarray([abs(float(by_cluster_b[obs.cluster_id]["z_delta_x_p_lens"])) for obs in observations], dtype=float)
    z_lens_p = np.asarray([abs(float(by_cluster_p[obs.cluster_id]["z_delta_x_p_lens"])) for obs in observations], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.6), constrained_layout=True)
    ax0, ax1 = axes

    ax0.errorbar(obs_vals, y, xerr=obs_err, fmt="o", color="black", capsize=4, label="observed lens-gas offset")
    ax0.scatter(pred_b, y + 0.08, marker="s", color="#1f77b4", label="baryon-only: Δx_P-gas")
    ax0.scatter(pred_p, y - 0.08, marker="o", color="#ff7f0e", label=f"P-model: Δx_P-gas (α={alpha_p:.3f})")
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels)
    ax0.set_xlabel("offset along collision axis [kpc]")
    ax0.set_title("Bullet-cluster offset audit: observed lens-gas vs model Δx_P-gas")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best")

    bar_w = 0.35
    ax1.bar(y - bar_w / 2.0, z_lens_b, width=bar_w, color="#1f77b4", label="baryon-only: |z(Δx_P-lens)|")
    ax1.bar(y + bar_w / 2.0, z_lens_p, width=bar_w, color="#ff7f0e", label="P-model: |z(Δx_P-lens)|")
    ax1.axhline(3.0, color="k", linestyle="--", linewidth=1.2, label="hard gate |z|=3")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("|z|")
    ax1.set_title("Lens-anchor residual gate")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")

    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 8.7.25: cluster collision P-peak offset audit")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=ROOT / "data" / "cosmology" / "bullet_cluster" / "collision_offset_observables.csv",
        help="Input offset table (kpc). If missing, embedded proxy values are used.",
    )
    parser.add_argument(
        "--sparc-metrics",
        type=Path,
        default=ROOT / "output" / "public" / "cosmology" / "sparc_rotation_curve_pmodel_audit_metrics.json",
        help="SPARC anchor metrics JSON.",
    )
    parser.add_argument(
        "--sparc-metrics-fallback",
        type=Path,
        default=ROOT / "output" / "private" / "cosmology" / "sparc_rotation_curve_pmodel_audit_metrics.json",
        help="Fallback SPARC anchor metrics JSON.",
    )
    parser.add_argument(
        "--lens-sigma-scale",
        type=float,
        default=0.6,
        help="Sigma scale for Δx_P-lens gate relative to observed lens-gas sigma.",
    )
    parser.add_argument(
        "--lens-sigma-floor-kpc",
        type=float,
        default=10.0,
        help="Minimum sigma for Δx_P-lens gate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "public" / "cosmology",
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "cosmology_cluster_collision_p_peak_offset_audit.json"
    csv_path = output_dir / "cosmology_cluster_collision_p_peak_offset_audit.csv"
    png_path = output_dir / "cosmology_cluster_collision_p_peak_offset_audit.png"
    template_csv_path = output_dir / "collision_offset_observables_primary_template.csv"
    registration_checklist_csv_path = output_dir / "collision_offset_primary_registration_checklist.csv"
    previous_watchpack = _load_previous_watchpack(json_path)
    input_signature = _file_signature(args.input_csv)

    observations, source_mode, source_notes, source_diag = _load_collision_observations(args.input_csv)
    primary_readiness = _derive_primary_registration_readiness(source_diag)
    registration_checklist_rows, registration_checklist_summary = _build_primary_registration_checklist(
        input_csv=args.input_csv,
        observations=observations,
    )
    anchor, anchor_mode = _load_sparc_anchor([args.sparc_metrics, args.sparc_metrics_fallback])
    alpha_payload = _alpha_from_sparc(anchor)
    primary_map_watchpack = _derive_primary_map_update_watchpack(
        current_signature=input_signature,
        source_mode=source_mode,
        source_diag=source_diag,
        primary_readiness=primary_readiness,
        previous_watchpack=previous_watchpack,
    )

    evaluated_b = _evaluate_model(
        observations,
        alpha=0.0,
        model_name="baryon_only",
        lens_sigma_scale=float(args.lens_sigma_scale),
        lens_sigma_floor_kpc=float(args.lens_sigma_floor_kpc),
    )
    evaluated_p = _evaluate_model(
        observations,
        alpha=float(alpha_payload["alpha_from_sparc"]),
        model_name="pmodel_corrected",
        lens_sigma_scale=float(args.lens_sigma_scale),
        lens_sigma_floor_kpc=float(args.lens_sigma_floor_kpc),
    )

    checks = _build_checks(
        source_mode=source_mode,
        baryon_summary=evaluated_b["summary"],
        pmodel_summary=evaluated_p["summary"],
    )
    decision = _decision_from_checks(checks)

    rows_out = [*evaluated_b["rows"], *evaluated_p["rows"]]
    _write_csv(csv_path, rows_out)
    _write_primary_registration_template(template_csv_path, observations, source_mode)
    _write_dict_rows_csv(
        registration_checklist_csv_path,
        (
            "cluster_id",
            "label",
            "row_present",
            "lens_map_ref",
            "xray_map_ref",
            "source_mode",
            "missing_fields",
            "mode_primary_ok",
            "ready",
            "status",
            "required_action",
        ),
        registration_checklist_rows,
    )
    _plot(
        png_path,
        observations=observations,
        rows_baryon=evaluated_b["rows"],
        rows_pmodel=evaluated_p["rows"],
        alpha_p=float(alpha_payload["alpha_from_sparc"]),
    )

    baryon_chi2_dof = float(evaluated_b["summary"]["chi2_dof"])
    pmodel_chi2_dof = float(evaluated_p["summary"]["chi2_dof"])
    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "phase": {"phase": 8, "step": "8.7.25.9", "name": "cluster collision P-peak offset audit"},
        "intent": (
            "Quantify whether P-peak offsets can improve lens-gas separation consistency "
            "relative to baryon-only baseline using a fixed audit interface."
        ),
        "inputs": {
            "offset_observations_csv": _rel(args.input_csv),
            "offset_source_mode": source_mode,
            "offset_source_notes": source_notes,
            "sparc_anchor_mode": anchor_mode,
            "sparc_anchor_source": anchor.get("source"),
        },
        "anchors": {
            "sparc": anchor,
            "alpha_mapping": alpha_payload,
            "lens_sigma_scale": float(args.lens_sigma_scale),
            "lens_sigma_floor_kpc": float(args.lens_sigma_floor_kpc),
        },
        "models": {
            "baryon_only": evaluated_b["summary"],
            "pmodel_corrected": evaluated_p["summary"],
            "comparison": {
                "delta_chi2_baryon_minus_pmodel": float(evaluated_b["summary"]["chi2"] - evaluated_p["summary"]["chi2"]),
                "chi2_dof_ratio_pmodel_over_baryon": float(pmodel_chi2_dof / max(baryon_chi2_dof, 1.0e-12)),
            },
        },
        "diagnostics": {
            "input_mode_details": source_diag,
            "primary_registration_readiness": primary_readiness,
            "primary_registration_checklist_summary": registration_checklist_summary,
            "primary_map_update_watchpack": primary_map_watchpack,
        },
        "cluster_rows": rows_out,
        "checks": checks,
        "decision": decision,
        "outputs": {
            "audit_json": _rel(json_path),
            "audit_csv": _rel(csv_path),
            "audit_png": _rel(png_path),
            "primary_registration_template_csv": _rel(template_csv_path),
            "primary_registration_checklist_csv": _rel(registration_checklist_csv_path),
        },
        "falsification_gate": {
            "reject_if": [
                "Any hard check fails.",
                "baryon-only baseline is not rejected by offset gate (chi2/dof < 4).",
                "P-model corrected branch fails lens anchor gate (max |z(delta_x_p_lens)| > 3).",
            ],
            "watch_if": [
                "Only watch-level checks fail (current proxy-input stage).",
            ],
        },
    }
    _write_json(json_path, payload)

    # 条件分岐: `worklog is not None` を満たす経路を評価する。
    if worklog is not None:
        try:
            worklog.append_event(
                "step_8_7_25_9_cluster_collision_offset_audit",
                {
                    "decision": decision.get("decision"),
                    "overall_status": decision.get("overall_status"),
                    "hard_fail_ids": decision.get("hard_fail_ids"),
                    "watch_ids": decision.get("watch_ids"),
                    "chi2_dof_baryon": baryon_chi2_dof,
                    "chi2_dof_pmodel": pmodel_chi2_dof,
                    "alpha_from_sparc": float(alpha_payload["alpha_from_sparc"]),
                    "offset_source_mode": source_mode,
                    "primary_readiness_reason": primary_readiness.get("decision_reason"),
                    "primary_ready": primary_readiness.get("ready"),
                    "registration_checklist_next_action": registration_checklist_summary.get("next_action"),
                    "registration_checklist_pending": registration_checklist_summary.get("n_pending"),
                    "primary_map_update_event_detected": primary_map_watchpack.get("update_event_detected"),
                    "primary_map_update_event_type": primary_map_watchpack.get("update_event_type"),
                    "primary_map_update_event_counter": primary_map_watchpack.get("event_counter"),
                    "output_json": _rel(json_path),
                },
            )
        except Exception:
            pass

    print(f"[ok] wrote {json_path}")
    print(f"[ok] wrote {csv_path}")
    print(f"[ok] wrote {png_path}")
    print(f"[ok] wrote {template_csv_path}")
    print(f"[ok] wrote {registration_checklist_csv_path}")
    print(
        "[summary] decision={0} status={1} source_mode={2} readiness={3} chi2/dof(baryon)={4:.3f} chi2/dof(P)={5:.3f} alpha={6:.4f}".format(
            decision.get("decision"),
            decision.get("overall_status"),
            source_mode,
            primary_readiness.get("decision_reason"),
            baryon_chi2_dof,
            pmodel_chi2_dof,
            float(alpha_payload["alpha_from_sparc"]),
        )
    )
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
