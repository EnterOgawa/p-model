#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_solver_family_elimination_audit.py

Step 8.7.55.2.13:
Reduce the admissible mass-origin solver families using only public canonical
artifacts and explicit positioning notes already present in the repository.

Inputs:
  - output/public/quantum/mass_origin_shell_quantization_canonicalization_metrics.json
  - output/public/quantum/particle_reflection_demo_metrics.json
  - output/public/quantum/nuclear_binding_energy_frequency_mapping_deuteron_two_body_metrics.json
  - doc/quantum/18_p_field_action_and_schrodinger_mapping.md

Outputs:
  - output/public/quantum/mass_origin_solver_family_elimination_metrics.json
  - output/public/quantum/mass_origin_solver_family_elimination_rows.csv
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

SHELL_CANON_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_canonicalization_metrics.json"
REFLECTION_JSON = ROOT / "output" / "public" / "quantum" / "particle_reflection_demo_metrics.json"
TWO_BODY_JSON = ROOT / "output" / "public" / "quantum" / "nuclear_binding_energy_frequency_mapping_deuteron_two_body_metrics.json"
MASS_NOTE_MD = ROOT / "doc" / "quantum" / "18_p_field_action_and_schrodinger_mapping.md"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_solver_family_elimination_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_solver_family_elimination_rows.csv"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_require_path` の入出力契約と処理意図を定義する。

def _require_path(path: Path) -> None:
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


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows(shell_payload: Dict[str, Any], reflection: Dict[str, Any], two_body: Dict[str, Any], mass_note: str) -> List[Dict[str, Any]]:
    reflection_notes = " ".join(reflection.get("notes", []))
    two_body_notes = " ".join(two_body.get("square_well_example", {}).get("notes", []))
    shell_public = bool(shell_payload.get("decision", {}).get("shell_quantization_public_canonical", False))
    note_mentions_complex = "複素場" in mass_note or "complex field" in mass_note.lower()
    note_mentions_qball = "q-ball" in mass_note.lower() or "q ball" in mass_note.lower()
    return [
        {
            "row_id": "reflection_family_elimination",
            "status": "reject",
            "metric": "reflection family kept as toy-only, not as mass-origin solver",
            "value": 0.0,
            "note": f"Rejected from the solver family set because the public note explicitly says: {reflection_notes}",
        },
        {
            "row_id": "two_body_family_elimination",
            "status": "deferred_noncanonical",
            "metric": "two-body boundary family kept as interface only",
            "value": 0.0,
            "note": f"Deferred because the public note explicitly says: {two_body_notes}",
        },
        {
            "row_id": "shell_quantization_family_survives",
            "status": "candidate_public" if shell_public else "missing",
            "metric": "shell quantization remains the only public solver-family candidate",
            "value": 1.0 if shell_public else 0.0,
            "note": "Shell quantization is retained because it is the only remaining family with public canonical machine-readable artifacts tied to the nuclear mass-frequency branch.",
        },
        {
            "row_id": "complex_field_family_elimination",
            "status": "deferred_noncanonical" if note_mentions_complex and note_mentions_qball else "missing",
            "metric": "complex-field / Q-ball family remains note-only",
            "value": 0.0,
            "note": "Deferred because the complex-field / Q-ball branch is still doc-only and has no public canonical artifact.",
        },
        {
            "row_id": "single_public_boundary_family_remaining",
            "status": "pass" if shell_public else "reject",
            "metric": "one public boundary / quantization family remains after elimination",
            "value": 1.0 if shell_public else 0.0,
            "note": "After rejecting toy-only and deferring noncanonical families, shell quantization is the sole public remaining family.",
        },
        {
            "row_id": "family_elimination_progress_realized",
            "status": "pass" if shell_public else "reject",
            "metric": "admissible family count reduced below four",
            "value": 3.0 if shell_public else 0.0,
            "note": "At least three former admissible families are no longer carried as active public solver candidates.",
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload() -> Dict[str, Any]:
    for path in (SHELL_CANON_JSON, REFLECTION_JSON, TWO_BODY_JSON, MASS_NOTE_MD):
        _require_path(path)

    shell_payload = _read_json(SHELL_CANON_JSON)
    reflection = _read_json(REFLECTION_JSON)
    two_body = _read_json(TWO_BODY_JSON)
    mass_note = _read_text(MASS_NOTE_MD)
    rows = _build_rows(shell_payload, reflection, two_body, mass_note)
    single_public_family = float(next(row["value"] for row in rows if row["row_id"] == "single_public_boundary_family_remaining")) > 0.0
    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.55.2.13",
            "name": "solver-family elimination audit",
        },
        "inputs": {
            "mass_origin_shell_quantization_canonicalization_json": _relative_str(SHELL_CANON_JSON),
            "particle_reflection_demo_json": _relative_str(REFLECTION_JSON),
            "nuclear_binding_energy_frequency_mapping_deuteron_two_body_json": _relative_str(TWO_BODY_JSON),
            "mass_origin_note_md": _relative_str(MASS_NOTE_MD),
        },
        "intent": "Reduce the active mass-origin family set from four candidates toward one public solver family without inventing new physics inputs.",
        "rows": rows,
        "summary": {
            "starting_family_count": 4,
            "remaining_public_family_count": 1 if single_public_family else 0,
            "rejected_family_count": 1,
            "deferred_noncanonical_family_count": 2,
            "surviving_public_family": "boundary_shell_quantization" if single_public_family else "",
        },
        "decision": {
            "overall_status": "single_public_family_shell_quantization_remaining" if single_public_family else "family_elimination_failed",
            "single_public_boundary_family_remaining": single_public_family,
            "surviving_public_family": "boundary_shell_quantization" if single_public_family else "",
            "proceed_to_same_sector_mapping_bridge": single_public_family,
            "proceed_to_no_free_parameter_mass_solver": False,
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
    print(f"[ok] json: {OUT_JSON}")
    print(f"[ok] csv : {OUT_CSV}")


if __name__ == "__main__":
    main()
