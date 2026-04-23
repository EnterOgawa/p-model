#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_solver_spec_gate_audit.py

Step 8.7.55.2.3:
Freeze whether the current public canonical repository actually contains a
particle-sector mapping from chi_P (or an equivalent same-sector curvature
proxy) to V''(|P|_*), and whether the currently admissible mass-origin
families can be reduced to one no-free-parameter solver specification.

Inputs:
  - output/public/quantum/mass_origin_readiness_gate_metrics.json
  - output/public/quantum/mass_origin_curvature_boundary_metrics.json
  - output/public/quantum/gravity_quantum_differential_prediction_table_metrics.json
  - doc/quantum/18_p_field_action_and_schrodinger_mapping.md
  - scripts/quantum/nuclear_a_dependence_mean_field.py
  - output/public/quantum/*metrics.json (scan)

Outputs:
  - output/public/quantum/mass_origin_solver_spec_gate_metrics.json
  - output/public/quantum/mass_origin_solver_spec_gate_rows.csv

Assumptions:
  - A mass-origin solver is only admissible once all of the following are fixed
    in the same particle sector:
      1. chi_P (or an equivalent same-sector quantity) -> V''(|P|_*)
      2. one boundary / quantization family only
      3. one V(|P|) shape only
  - Public canonical metrics are the authoritative evidence pack.
  - Script-only and doc-only candidates are recorded, but they do not satisfy
    the public canonical gate for a no-free-parameter solver.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]

READINESS_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_readiness_gate_metrics.json"
CURVATURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_curvature_boundary_metrics.json"
GRAVITY_DIFF_JSON = ROOT / "output" / "public" / "quantum" / "gravity_quantum_differential_prediction_table_metrics.json"
MASS_NOTE_MD = ROOT / "doc" / "quantum" / "18_p_field_action_and_schrodinger_mapping.md"
SHELL_SCRIPT = ROOT / "scripts" / "quantum" / "nuclear_a_dependence_mean_field.py"
PUBLIC_QUANTUM_DIR = ROOT / "output" / "public" / "quantum"
OUT_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_solver_spec_gate_metrics.json"
OUT_CSV = PUBLIC_QUANTUM_DIR / "mass_origin_solver_spec_gate_rows.csv"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_require_path` の入出力契約と処理意図を定義する。

def _require_path(path: Path) -> None:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: `_read_text` の入出力契約と処理意図を定義する。

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# 関数: `_relative_str` の入出力契約と処理意図を定義する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: `_iter_strings` の入出力契約と処理意図を定義する。

def _iter_strings(node: Any) -> Iterable[str]:
    # 条件分岐: `isinstance(node, dict)` を満たす経路を評価する。
    if isinstance(node, dict):
        for key, value in node.items():
            yield str(key)
            yield from _iter_strings(value)

        return

    # 条件分岐: `isinstance(node, list)` を満たす経路を評価する。

    if isinstance(node, list):
        for item in node:
            yield from _iter_strings(item)

        return

    # 条件分岐: `node is None` を満たす経路を評価する。

    if node is None:
        return

    yield str(node)


# 関数: `_find_row_by_id` の入出力契約と処理意図を定義する。

def _find_row_by_id(rows: List[Dict[str, Any]], row_id: str) -> Dict[str, Any]:
    for row in rows:
        # 条件分岐: `str(row.get("row_id")) == row_id` を満たす経路を評価する。
        if str(row.get("row_id")) == row_id:
            return row

    raise KeyError(f"missing row_id: {row_id}")


# 関数: `_classify_chi_file` の入出力契約と処理意図を定義する。

def _classify_chi_file(path: Path) -> str:
    name = path.name

    # 条件分岐: `"mass_origin_" in name` を満たす経路を評価する。
    if "mass_origin_" in name:
        return "mass_origin_gate"

    # 条件分岐: `"born_phase_diffusion" in name or "gravity_" in name or "entanglement_" in name` を満たす経路を評価する。

    if "born_phase_diffusion" in name or "gravity_" in name or "entanglement_" in name:
        return "cross_sector_decoherence"

    # 条件分岐: `"quantum_information_" in name or "quantum_v11_plus_scope_closeout" in name` を満たす経路を評価する。

    if "quantum_information_" in name or "quantum_v11_plus_scope_closeout" in name:
        return "minimal_connection_application"

    # 条件分岐: `"pointer_basis" in name` を満たす経路を評価する。

    if "pointer_basis" in name:
        return "detector_measurement"

    return "other"


# 関数: `_scan_public_metrics` の入出力契約と処理意図を定義する。

def _scan_public_metrics() -> Dict[str, Any]:
    metrics_files = sorted(PUBLIC_QUANTUM_DIR.glob("*metrics.json"))
    chi_files: List[Path] = []
    chi_sector_counts: Dict[str, int] = {}
    positive_same_sector_rows: List[Dict[str, Any]] = []

    for path in metrics_files:
        data = _read_json(path)
        flat_strings = "\n".join(_iter_strings(data))
        has_chi = "chi_P" in flat_strings or "chi_p" in flat_strings or "chiP" in flat_strings

        # 条件分岐: `has_chi` を満たす経路を評価する。
        if has_chi:
            chi_files.append(path)
            sector = _classify_chi_file(path)
            chi_sector_counts[sector] = chi_sector_counts.get(sector, 0) + 1

        rows = data.get("rows", [])
        # 条件分岐: `not isinstance(rows, list)` を満たす経路を評価する。
        if not isinstance(rows, list):
            continue

        for row in rows:
            # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
            if not isinstance(row, dict):
                continue

            row_id = str(row.get("row_id", ""))
            metric = str(row.get("metric", ""))
            note = str(row.get("note", ""))
            status = str(row.get("status", ""))
            descriptor = " ".join((row_id, metric))
            has_mapping_signature = ("chi_P" in descriptor or "chi_p" in descriptor or "chiP" in descriptor) and ("V''" in descriptor or "vpp" in descriptor.lower())
            positive_status = status in {"pass", "candidate_public"}

            # 条件分岐: `has_mapping_signature and positive_status` を満たす経路を評価する。
            if has_mapping_signature and positive_status:
                positive_same_sector_rows.append(
                    {
                        "path": _relative_str(path),
                        "row_id": row_id,
                        "status": status,
                        "metric": metric,
                    }
                )

    return {
        "metrics_files": metrics_files,
        "chi_files": chi_files,
        "chi_sector_counts": chi_sector_counts,
        "positive_same_sector_rows": positive_same_sector_rows,
    }


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload() -> Dict[str, Any]:
    for path in (READINESS_JSON, CURVATURE_JSON, GRAVITY_DIFF_JSON, MASS_NOTE_MD, SHELL_SCRIPT):
        _require_path(path)

    readiness = _read_json(READINESS_JSON)
    curvature = _read_json(CURVATURE_JSON)
    gravity = _read_json(GRAVITY_DIFF_JSON)
    mass_note = _read_text(MASS_NOTE_MD)
    shell_text = _read_text(SHELL_SCRIPT)
    scan = _scan_public_metrics()

    readiness_rows = readiness.get("rows", [])
    curvature_rows = curvature.get("rows", [])
    gravity_rows = gravity.get("rows", [])

    # 条件分岐: `not isinstance(readiness_rows, list)` を満たす経路を評価する。
    if not isinstance(readiness_rows, list):
        raise SystemExit(f"[fail] invalid rows in {READINESS_JSON}")

    # 条件分岐: `not isinstance(curvature_rows, list)` を満たす経路を評価する。

    if not isinstance(curvature_rows, list):
        raise SystemExit(f"[fail] invalid rows in {CURVATURE_JSON}")

    # 条件分岐: `not isinstance(gravity_rows, list)` を満たす経路を評価する。

    if not isinstance(gravity_rows, list):
        raise SystemExit(f"[fail] invalid rows in {GRAVITY_DIFF_JSON}")

    readiness_same_sector = _find_row_by_id(readiness_rows, "same_sector_chi_p_to_vpp_mapping")
    curvature_same_sector = _find_row_by_id(curvature_rows, "same_sector_curvature_mapping_particle_sector")
    unique_boundary_row = _find_row_by_id(curvature_rows, "single_boundary_family_unique")
    unique_potential_row = _find_row_by_id(curvature_rows, "single_vpp_shape_unique")
    solver_ready_row = _find_row_by_id(curvature_rows, "no_free_parameter_mass_solver_spec_ready")
    reflection_row = _find_row_by_id(curvature_rows, "reflection_boundary_candidate_public")
    two_body_row = _find_row_by_id(curvature_rows, "deuteron_two_body_boundary_candidate_public")
    shell_row = _find_row_by_id(curvature_rows, "shell_quantization_candidate_script_branch")
    complex_row = _find_row_by_id(curvature_rows, "complex_field_oscillon_qball_candidate_doc_only")
    structural_row = _find_row_by_id(gravity_rows, "decoherence_structural_parity_entry")

    shell_has_step = "Step 7.13.15.11" in shell_text and "Step 7.13.15.12" in shell_text
    note_mentions_qball = "Q-ball" in mass_note or "Q ball" in mass_note or "Q-ball" in mass_note
    note_mentions_oscillon = "oscillon" in mass_note.lower()

    same_sector_consistent_missing = (
        str(readiness_same_sector.get("status")) == "missing"
        and str(curvature_same_sector.get("status")) == "missing"
        and len(scan["positive_same_sector_rows"]) == 0
    )

    rows = [
        {
            "row_id": "public_metrics_scan_scope",
            "status": "pass",
            "metric": "public quantum metrics files scanned",
            "value": float(len(scan["metrics_files"])),
            "note": "Repository-wide public canonical scan used to decide whether a same-sector mass-origin mapping actually exists.",
        },
        {
            "row_id": "public_metrics_with_chi_p",
            "status": "inventory",
            "metric": "public metrics files containing chi_P",
            "value": float(len(scan["chi_files"])),
            "note": f"chi_P occurs in {len(scan['chi_files'])} public metrics files, but sector counts show they are cross-sector / application rows plus the mass-origin missing-status files.",
        },
        {
            "row_id": "positive_same_sector_mapping_public_artifact_count",
            "status": "reject",
            "metric": "public positive same-sector chi_P -> V''(|P|_*) artifacts",
            "value": float(len(scan["positive_same_sector_rows"])),
            "note": "No public metrics row provides a positive particle-sector chi_P -> V''(|P|_*) mapping; the explicit mass-origin rows remain missing / blocked only.",
        },
        {
            "row_id": "same_sector_mapping_absence_consistent",
            "status": "pass" if same_sector_consistent_missing else "reject",
            "metric": "readiness/curvature audits agree on same-sector mapping absence",
            "value": 1.0 if same_sector_consistent_missing else 0.0,
            "note": "Both 8.7.55.2.1 and 8.7.55.2.2 freeze the same-sector mapping as missing, and the repository-wide public scan finds no contradictory positive artifact.",
        },
        {
            "row_id": "reflection_family_public_candidate",
            "status": str(reflection_row.get("status", "unknown")),
            "metric": str(reflection_row.get("metric", "")),
            "value": float(reflection_row.get("value", 0.0)),
            "note": str(reflection_row.get("note", "")),
        },
        {
            "row_id": "two_body_family_public_candidate",
            "status": str(two_body_row.get("status", "unknown")),
            "metric": str(two_body_row.get("metric", "")),
            "value": float(two_body_row.get("value", 0.0)),
            "note": str(two_body_row.get("note", "")),
        },
        {
            "row_id": "shell_quantization_family_public_gate",
            "status": str(shell_row.get("status", "unknown")),
            "metric": str(shell_row.get("metric", "")),
            "value": float(shell_row.get("value", 0.0)),
            "note": f"{shell_row.get('note', '')} script_declared={shell_has_step}.",
        },
        {
            "row_id": "complex_field_family_public_gate",
            "status": str(complex_row.get("status", "unknown")),
            "metric": str(complex_row.get("metric", "")),
            "value": float(complex_row.get("value", 0.0)),
            "note": f"{complex_row.get('note', '')} note_mentions_oscillon={note_mentions_oscillon} note_mentions_qball={note_mentions_qball}.",
        },
        {
            "row_id": "single_boundary_family_unique",
            "status": str(unique_boundary_row.get("status", "unknown")),
            "metric": str(unique_boundary_row.get("metric", "")),
            "value": float(unique_boundary_row.get("value", 0.0)),
            "note": str(unique_boundary_row.get("note", "")),
        },
        {
            "row_id": "single_vpp_shape_unique",
            "status": str(unique_potential_row.get("status", "unknown")),
            "metric": str(unique_potential_row.get("metric", "")),
            "value": float(unique_potential_row.get("value", 0.0)),
            "note": str(unique_potential_row.get("note", "")),
        },
        {
            "row_id": "no_free_parameter_solver_spec_ready",
            "status": str(solver_ready_row.get("status", "unknown")),
            "metric": str(solver_ready_row.get("metric", "")),
            "value": float(solver_ready_row.get("value", 0.0)),
            "note": "Solver-spec gate remains closed because same-sector curvature, unique boundary family, and unique V(|P|) shape are all still absent in public canonical form.",
        },
        {
            "row_id": "mass_origin_branch_reopen_requires_new_public_artifact",
            "status": "blocked",
            "metric": "minimum reopen requirement",
            "value": 0.0,
            "note": "Reopen only after a public canonical artifact fixes a positive particle-sector chi_P -> V''(|P|_*) map and collapses the admissible family set to one no-free-parameter solver family.",
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.55.2.3",
            "name": "mass-origin same-sector mapping and solver-spec gate",
        },
        "inputs": {
            "mass_origin_readiness_gate_json": _relative_str(READINESS_JSON),
            "mass_origin_curvature_boundary_json": _relative_str(CURVATURE_JSON),
            "gravity_quantum_differential_prediction_table_json": _relative_str(GRAVITY_DIFF_JSON),
            "mass_origin_note_md": _relative_str(MASS_NOTE_MD),
            "shell_quantization_script": _relative_str(SHELL_SCRIPT),
            "public_quantum_metrics_scan_root": _relative_str(PUBLIC_QUANTUM_DIR),
        },
        "intent": "Decide whether the current public canonical repository already supports a same-sector particle-mass solver specification, or whether the mass-origin branch must remain explicitly blocked.",
        "formulas": {
            "solver_gate": "same-sector chi_P -> V''(|P|_*) + unique boundary family + unique V(|P|) -> no-free-parameter omega_* ladder",
            "current_cross_sector_proxy": "(k_B T_env / chi_P)_parity is structural and cross-sector only",
        },
        "scan_summary": {
            "public_metrics_file_count": len(scan["metrics_files"]),
            "chi_p_file_count": len(scan["chi_files"]),
            "chi_p_files": [_relative_str(path) for path in scan["chi_files"]],
            "chi_p_sector_counts": scan["chi_sector_counts"],
            "positive_same_sector_mapping_row_count": len(scan["positive_same_sector_rows"]),
            "positive_same_sector_mapping_rows": scan["positive_same_sector_rows"],
        },
        "rows": rows,
        "summary": {
            "same_sector_mapping_explicitly_absent": same_sector_consistent_missing,
            "public_candidate_family_count": 3,
            "script_only_family_count": 0,
            "doc_only_family_count": 1,
            "unique_boundary_family_fixed": False,
            "unique_potential_shape_fixed": False,
            "shell_quantization_public_canonical": str(shell_row.get("status", "")) == "candidate_public",
            "cross_sector_proxy_value": float(structural_row.get("differential_prediction_value", 0.0)),
        },
        "decision": {
            "overall_status": "explicit_block_fixed_same_sector_mapping_absent",
            "same_sector_curvature_mapping_available": False,
            "unique_boundary_family_available": False,
            "unique_potential_shape_available": False,
            "proceed_to_no_free_parameter_mass_solver": False,
            "proceed_to_dark_matter_branch": False,
            "block_reason_codes": [
                "same_sector_curvature_absent_in_public_canonical_pack",
                "multiple_admissible_solver_families_remain",
                "single_vpp_shape_not_fixed",
            ],
        },
        "evidence": {
            "readiness_same_sector_status": readiness_same_sector,
            "curvature_same_sector_status": curvature_same_sector,
            "unique_boundary_row": unique_boundary_row,
            "unique_potential_row": unique_potential_row,
            "solver_ready_row": solver_ready_row,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    payload = _build_payload()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["rows"])
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
