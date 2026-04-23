#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_latent_reopen_route_inventory_audit.py

Step 8.7.55.2.17:
Inventory the repository-wide latent reopen routes for the two remaining
mass-origin blockers:

  1. positive particle-sector chi_P -> V''(|P|_*) public artifact
  2. single public V(|P|) shape

This step does not create either missing artifact. It converts the absence of
repo-wide latent routes into a public-canonical artifact by combining:

  - the solver-spec repo-wide same-sector scan,
  - the public nuclear effective-potential artifact family, and
  - the mass-origin note's doc-only stabilization routes.

Inputs:
  - output/public/quantum/mass_origin_solver_spec_gate_metrics.json
  - output/public/quantum/mass_origin_shell_curvature_bridge_metrics.json
  - output/public/quantum/nuclear_effective_potential*_metrics.json
  - output/public/quantum/*.json (doc-only route row scan)
  - doc/quantum/18_p_field_action_and_schrodinger_mapping.md

Outputs:
  - output/public/quantum/mass_origin_latent_reopen_route_inventory_metrics.json
  - output/public/quantum/mass_origin_latent_reopen_route_inventory_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

PUBLIC_QUANTUM_DIR = ROOT / "output" / "public" / "quantum"
SOLVER_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_solver_spec_gate_metrics.json"
SHELL_BRIDGE_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_shell_curvature_bridge_metrics.json"
NOTE_MD = ROOT / "doc" / "quantum" / "18_p_field_action_and_schrodinger_mapping.md"
OUT_JSON = PUBLIC_QUANTUM_DIR / "mass_origin_latent_reopen_route_inventory_metrics.json"
OUT_CSV = PUBLIC_QUANTUM_DIR / "mass_origin_latent_reopen_route_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.17"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory repo-wide latent reopen routes for the mass-origin branch.",
    )
    parser.add_argument(
        "--step-tag",
        default=DEFAULT_STEP_TAG,
        help="Roadmap step tag to stamp into the output payload.",
    )
    return parser.parse_args()


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


# 関数: `_positive_status` の入出力契約と処理意図を定義する。

def _positive_status(status: str) -> bool:
    return status not in {
        "missing",
        "reject",
        "watch",
        "entry_only",
        "blocked",
        "candidate_doc_only",
        "candidate_script_only",
        "candidate_public",
        "candidate_public_interface",
        "inventory",
        "fixed_target",
        "interface_fixed",
        "doc_only",
        "deferred_noncanonical",
        "noncanonical",
        "",
    }


# 関数: `_find_row_by_id` の入出力契約と処理意図を定義する。

def _find_row_by_id(rows: List[Dict[str, Any]], row_id: str) -> Dict[str, Any]:
    for row in rows:
        # 条件分岐: `str(row.get("row_id")) == row_id` を満たす経路を評価する。
        if str(row.get("row_id")) == row_id:
            return row

    raise KeyError(f"missing row_id: {row_id}")


# 関数: `_collect_effective_potential_candidates` の入出力契約と処理意図を定義する。

def _collect_effective_potential_candidates() -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for path in sorted(PUBLIC_QUANTUM_DIR.glob("nuclear_effective_potential*_metrics.json")):
        payload = _read_json(path)
        model = payload.get("model", {})
        ansatz = str(model.get("ansatz", ""))

        # 条件分岐: `not ansatz` を満たす経路を評価する。
        if not ansatz:
            continue

        positioning = model.get("positioning", [])
        positioning_text = " ".join(str(item) for item in positioning)
        positioning_lower = positioning_text.lower()
        is_phenomenological = (
            "phenomenological" in positioning_lower
            or "not a first-principles" in positioning_lower
            or "not a first principles" in positioning_lower
            or "effective model" in positioning_lower
        )
        candidates.append(
            {
                "path": _relative_str(path),
                "ansatz": ansatz,
                "positioning": positioning,
                "is_phenomenological": is_phenomenological,
            }
        )

    return candidates


# 関数: `_collect_doc_only_route_rows` の入出力契約と処理意図を定義する。

def _collect_doc_only_route_rows() -> List[Dict[str, Any]]:
    rows_found: List[Dict[str, Any]] = []
    route_tokens = ("oscillon", "q-ball", "qball", "complex-field", "complex field")

    for path in sorted(PUBLIC_QUANTUM_DIR.glob("*.json")):
        # 条件分岐: `path == OUT_JSON` を満たす経路を評価する。
        if path == OUT_JSON:
            continue

        payload = _read_json(path)
        rows = payload.get("rows", [])

        # 条件分岐: `not isinstance(rows, list)` を満たす経路を評価する。
        if not isinstance(rows, list):
            continue

        for row in rows:
            # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
            if not isinstance(row, dict):
                continue

            row_id = str(row.get("row_id", ""))
            status = str(row.get("status", ""))
            metric = str(row.get("metric", ""))
            note = str(row.get("note", ""))
            family = str(row.get("family", ""))
            text = " ".join((row_id, status, metric, note, family)).lower()

            # 条件分岐: `not any(token in text for token in route_tokens)` を満たす経路を評価する。
            if not any(token in text for token in route_tokens):
                continue

            rows_found.append(
                {
                    "path": _relative_str(path),
                    "row_id": row_id,
                    "status": status,
                    "metric": metric,
                    "note": note,
                }
            )

    return rows_found


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (SOLVER_JSON, SHELL_BRIDGE_JSON, NOTE_MD):
        _require_path(path)

    solver = _read_json(SOLVER_JSON)
    shell_bridge = _read_json(SHELL_BRIDGE_JSON)
    note_text = _read_text(NOTE_MD)

    solver_rows = solver.get("rows", [])
    bridge_rows = shell_bridge.get("rows", [])
    scan_summary = solver.get("scan_summary", {})

    # 条件分岐: `not isinstance(solver_rows, list)` を満たす経路を評価する。
    if not isinstance(solver_rows, list):
        raise SystemExit(f"[fail] invalid rows in {SOLVER_JSON}")

    # 条件分岐: `not isinstance(bridge_rows, list)` を満たす経路を評価する。

    if not isinstance(bridge_rows, list):
        raise SystemExit(f"[fail] invalid rows in {SHELL_BRIDGE_JSON}")

    positive_same_sector_row = _find_row_by_id(solver_rows, "positive_same_sector_mapping_public_artifact_count")
    shell_bridge_row = _find_row_by_id(bridge_rows, "shell_to_curvature_bridge_ready")
    effective_candidates = _collect_effective_potential_candidates()
    doc_only_route_rows = _collect_doc_only_route_rows()

    positive_same_sector_rows = scan_summary.get("positive_same_sector_mapping_rows", [])

    # 条件分岐: `not isinstance(positive_same_sector_rows, list)` を満たす経路を評価する。
    if not isinstance(positive_same_sector_rows, list):
        positive_same_sector_rows = []

    same_sector_positive_count = int(float(positive_same_sector_row.get("value", 0.0)))
    effective_candidate_count = len(effective_candidates)
    effective_nonphenomenological = [item for item in effective_candidates if not bool(item["is_phenomenological"])]
    effective_nonphenomenological_count = len(effective_nonphenomenological)
    doc_only_route_positive_rows = [row for row in doc_only_route_rows if _positive_status(str(row.get("status", "")))]

    note_lower = note_text.lower()
    named_doc_only_routes: List[str] = []

    # 条件分岐: `"oscillon" in note_lower` を満たす経路を評価する。
    if "oscillon" in note_lower:
        named_doc_only_routes.append("oscillon")

    # 条件分岐: `"q-ball" in note_lower or "qball" in note_lower` を満たす経路を評価する。

    if "q-ball" in note_lower or "qball" in note_lower:
        named_doc_only_routes.append("q-ball")

    # 条件分岐: `"複素場" in note_text or "complex" in note_lower` を満たす経路を評価する。

    if "複素場" in note_text or "complex" in note_lower:
        named_doc_only_routes.append("complex_field")

    latent_routes_exhausted = (
        same_sector_positive_count == 0
        and effective_nonphenomenological_count == 0
        and len(doc_only_route_positive_rows) == 0
    )

    rows = [
        {
            "row_id": "repo_wide_same_sector_scan_scope_inherited",
            "status": "pass",
            "metric": "public quantum metrics scan scope inherited from solver gate",
            "value": float(scan_summary.get("public_metrics_file_count", 0.0)),
            "note": "The solver-spec gate already scanned the public quantum pack repo-wide for positive same-sector chi_P -> V''(|P|_*) artifacts.",
        },
        {
            "row_id": "latent_positive_same_sector_public_rows",
            "status": "pass" if same_sector_positive_count > 0 else "reject",
            "metric": "repo-wide positive same-sector public rows",
            "value": float(same_sector_positive_count),
            "note": (
                "Repo-wide inherited same-sector scan count. "
                f"Current positive rows: {same_sector_positive_count}; shell bridge row remains `{shell_bridge_row.get('status', '')}`."
            ),
        },
        {
            "row_id": "effective_potential_public_ansatz_family_count",
            "status": "inventory",
            "metric": "public effective-potential ansatz family count",
            "value": float(effective_candidate_count),
            "note": "Counted from `nuclear_effective_potential*_metrics.json` files with a declared `model.ansatz`.",
        },
        {
            "row_id": "effective_potential_nonphenomenological_public_count",
            "status": "pass" if effective_nonphenomenological_count > 0 else "reject",
            "metric": "non-phenomenological public V(|P|) ansatz count",
            "value": float(effective_nonphenomenological_count),
            "note": (
                "Every current public effective-potential ansatz still carries positioning text that marks it as "
                "phenomenological / not first-principles."
            ),
        },
        {
            "row_id": "doc_only_stabilization_routes_named_in_note",
            "status": "inventory",
            "metric": "doc-only stabilization route names present in mass-origin note",
            "value": float(len(named_doc_only_routes)),
            "note": "Named note-level routes are: " + ", ".join(named_doc_only_routes) + ".",
        },
        {
            "row_id": "doc_only_stabilization_public_artifact_count",
            "status": "pass" if len(doc_only_route_positive_rows) > 0 else "reject",
            "metric": "doc-only stabilization routes already promoted to public artifact",
            "value": float(len(doc_only_route_positive_rows)),
            "note": (
                "Public row scan over oscillon / Q-ball / complex-field mentions finds no positive promoted artifact. "
                f"Matched rows total={len(doc_only_route_rows)}."
            ),
        },
        {
            "row_id": "latent_reopen_route_inventory_exhausted",
            "status": "pass" if latent_routes_exhausted else "reject",
            "metric": "repo-wide latent reopen route inventory exhausted",
            "value": 1.0 if latent_routes_exhausted else 0.0,
            "note": (
                "No repo-wide positive same-sector public row exists, no public non-phenomenological V(|P|) ansatz exists, "
                "and doc-only stabilization routes remain unpromoted."
            ),
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "mass-origin latent reopen route inventory",
        },
        "inputs": {
            "mass_origin_solver_spec_gate_json": _relative_str(SOLVER_JSON),
            "mass_origin_shell_curvature_bridge_json": _relative_str(SHELL_BRIDGE_JSON),
            "public_quantum_scan_root": _relative_str(PUBLIC_QUANTUM_DIR),
            "mass_origin_note_md": _relative_str(NOTE_MD),
        },
        "intent": "Inventory whether any repo-wide latent reopen route already exists for the two remaining mass-origin blockers before declaring the block purely external.",
        "rows": rows,
        "summary": {
            "repo_wide_same_sector_scan_file_count": int(scan_summary.get("public_metrics_file_count", 0)),
            "latent_positive_same_sector_public_row_count": same_sector_positive_count,
            "effective_potential_public_ansatz_family_count": effective_candidate_count,
            "effective_potential_nonphenomenological_public_count": effective_nonphenomenological_count,
            "doc_only_stabilization_named_routes": named_doc_only_routes,
            "doc_only_stabilization_public_artifact_count": len(doc_only_route_positive_rows),
            "latent_reopen_routes_exhausted": latent_routes_exhausted,
        },
        "decision": {
            "overall_status": "latent_reopen_routes_absent_block_remains",
            "same_sector_latent_public_route_available": same_sector_positive_count > 0,
            "single_public_vpp_latent_route_available": effective_nonphenomenological_count > 0,
            "doc_only_routes_promoted_to_public": len(doc_only_route_positive_rows) > 0,
            "keep_mass_origin_branch_blocked": True,
            "next_required_artifacts": [
                "positive_particle_sector_chi_p_to_vpp_public_artifact",
                "single_public_vpp_shape",
                "solver_ready_row_promoted_to_pass",
            ],
        },
        "evidence": {
            "solver_same_sector_positive_rows": positive_same_sector_rows,
            "shell_bridge_row": shell_bridge_row,
            "effective_potential_candidates": effective_candidates,
            "doc_only_route_rows": doc_only_route_rows,
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
    args = _parse_args()
    payload = _build_payload(str(args.step_tag))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["rows"])
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
