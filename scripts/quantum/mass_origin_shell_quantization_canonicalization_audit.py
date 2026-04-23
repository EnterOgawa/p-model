#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_quantization_canonicalization_audit.py

Step 8.7.55.2.12:
Promote the shell-quantization branch from source-only existence to a public
canonical artifact, using the already generated Step 7.13.15.11 / .12 outputs.

Inputs:
  - output/public/quantum/nuclear_a_dependence_hf_three_body_shell_quantization_metrics.json
  - output/public/quantum/nuclear_a_dependence_hf_three_body_shell_quantization_asym_metrics.json

Outputs:
  - output/public/quantum/mass_origin_shell_quantization_canonicalization_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_canonicalization_rows.csv
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

SHELL_JSON = ROOT / "output" / "public" / "quantum" / "nuclear_a_dependence_hf_three_body_shell_quantization_metrics.json"
SHELL_ASYM_JSON = ROOT / "output" / "public" / "quantum" / "nuclear_a_dependence_hf_three_body_shell_quantization_asym_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_canonicalization_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_canonicalization_rows.csv"


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


# 関数: `_relative_str` の入出力契約と処理意図を定義する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows(shell: Dict[str, Any], shell_asym: Dict[str, Any]) -> List[Dict[str, Any]]:
    shell_diag = shell.get("diag", {})
    asym_diag = shell_asym.get("diag", {})
    return [
        {
            "row_id": "shell_quantization_step11_public_metrics",
            "status": "pass",
            "metric": "Step 7.13.15.11 public metrics present",
            "value": 1.0,
            "note": "Symmetric shell-quantization correction is now machine-readable in public canonical form.",
        },
        {
            "row_id": "shell_quantization_step12_public_metrics",
            "status": "pass",
            "metric": "Step 7.13.15.12 public metrics present",
            "value": 1.0,
            "note": "Asymmetric shell-quantization correction is now machine-readable in public canonical form.",
        },
        {
            "row_id": "shell_quantization_family_public_candidate",
            "status": "candidate_public",
            "metric": "shell quantization family promoted to public canonical candidate",
            "value": 1.0,
            "note": "Step 7.13.15.11/.12 outputs exist as public canonical artifacts, so the family is no longer script-only.",
        },
        {
            "row_id": "shell_quantization_fit_kappa",
            "status": "inventory",
            "metric": "symmetric shell quantization kappa",
            "value": float(shell_diag.get("fit", {}).get("kappa", 0.0)),
            "note": "Least-squares shell quantization coefficient fitted on magic-N gap_n rows.",
        },
        {
            "row_id": "shell_quantization_fit_kz_over_kn",
            "status": "inventory",
            "metric": "asymmetric shell quantization kZ / kN",
            "value": float(asym_diag.get("fit", {}).get("kZ_over_kN", 0.0)),
            "note": "Asymmetric proton / neutron shell correction ratio from Step 7.13.15.12.",
        },
        {
            "row_id": "shell_quantization_gap_s2n_corrected_rms",
            "status": "inventory",
            "metric": "gap_S2n corrected RMS residual",
            "value": float(shell_diag.get("gap_S2n", {}).get("rms_resid_corrected_MeV", 0.0)),
            "note": "Representative corrected shell-gap residual kept as canonical diagnostic.",
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload() -> Dict[str, Any]:
    _require_path(SHELL_JSON)
    _require_path(SHELL_ASYM_JSON)
    shell = _read_json(SHELL_JSON)
    shell_asym = _read_json(SHELL_ASYM_JSON)
    rows = _build_rows(shell, shell_asym)
    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.55.2.12",
            "name": "shell-quantization public canonicalization",
        },
        "inputs": {
            "shell_quantization_metrics_json": _relative_str(SHELL_JSON),
            "shell_quantization_asym_metrics_json": _relative_str(SHELL_ASYM_JSON),
        },
        "intent": "Freeze that shell quantization is now a public canonical family candidate rather than a script-only branch.",
        "rows": rows,
        "summary": {
            "shell_quantization_public_canonical": True,
            "public_artifact_count": 2,
            "family_status": "candidate_public",
        },
        "decision": {
            "overall_status": "shell_quantization_public_canonical_fixed",
            "shell_quantization_public_canonical": True,
            "shell_quantization_family_status": "candidate_public",
            "rerun_curvature_and_solver_equivalent_audits": True,
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
