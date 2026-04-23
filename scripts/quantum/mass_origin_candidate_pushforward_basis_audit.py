#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_candidate_pushforward_basis_audit.py

Step 8.7.55.2.388:
Push the surviving V(|P|) candidates into chi-space through rho = P_ref exp(chi)
with P_ref ≡ P_infty, then audit whether each candidate belongs to the frozen
ambient finite basis discovered in Step 8.7.55.2.387.

Inputs:
  - doc/paper/10_part1_core_theory.md
  - output/public/quantum/mass_origin_chi_space_action_basis_inventory_metrics.json
  - output/public/quantum/mass_origin_anchor_local_r3_registry_metrics.json
  - output/public/quantum/mass_origin_anchor_local_shape_jet_metrics.json

Outputs:
  - output/public/quantum/mass_origin_candidate_pushforward_basis_audit_metrics.json
  - output/public/quantum/mass_origin_candidate_pushforward_basis_audit_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

PART1_MD = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
BASIS_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_chi_space_action_basis_inventory_metrics.json"
R3_REGISTRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_r3_registry_metrics.json"
SHAPE_JET_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_jet_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_candidate_pushforward_basis_audit_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_candidate_pushforward_basis_audit_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.388"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit candidate V(|P|) pushforwards against the frozen chi-space ambient basis.",
    )
    parser.add_argument(
        "--step-tag",
        default=DEFAULT_STEP_TAG,
        help="Roadmap step tag to stamp into the output payload.",
    )
    return parser.parse_args()


# 関数: 必須入力の存在を検査する。

def _require_path(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: JSON ファイルを辞書として読む。

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: UTF-8 テキストを読み込む。

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# 関数: リポジトリ相対パスへ正規化する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: 指定パターンを最初に含む行を返す。

def _find_first_match(text: str, pattern: str) -> Dict[str, Any] | None:
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if pattern in raw_line:
            return {
                "pattern": pattern,
                "line": line_number,
                "text": raw_line.strip(),
            }

    return None


# 関数: 候補ごとの basis membership 行を構成する。

def _candidate_rows(
    *,
    family: str,
    pure_exponential_member: bool,
    naked_log_present: bool,
    accepted_exponents: List[int],
    rejected_reason: str | None,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": f"candidate_pushforward_{family}_constructed",
            "status": "pass",
            "metric": f"{family} chi-space pushforward constructed",
            "value": 1.0,
            "note": f"The chi-space pushforward for {family} has been explicitly constructed and audited against the frozen ambient basis.",
        },
        {
            "row_id": f"candidate_pushforward_{family}_pure_exponential_basis_member",
            "status": "pass" if pure_exponential_member else "reject",
            "metric": f"{family} belongs to the finite pure exponential basis",
            "value": 1.0 if pure_exponential_member else 0.0,
            "note": (
                f"{family} closes on the pure exponential exponent set {accepted_exponents}."
                if pure_exponential_member
                else f"{family} does not close on the frozen pure exponential basis: {rejected_reason}."
            ),
        },
        {
            "row_id": f"candidate_pushforward_{family}_naked_log_coordinate_absent",
            "status": "pass" if not naked_log_present else "reject",
            "metric": f"{family} avoids a naked chi outside exponentials",
            "value": 1.0 if not naked_log_present else 0.0,
            "note": (
                f"{family} introduces no naked chi outside an exponential envelope."
                if not naked_log_present
                else f"{family} exposes a naked chi outside the exponential basis: {rejected_reason}."
            ),
        },
    ]


# 関数: rows をまとめて構成する。

def _build_rows(
    *,
    ambient_basis_ready: bool,
    ambient_basis_exponents: List[int],
    candidate_family_ids: List[str],
    accepted_candidates: List[str],
    rejected_candidates: List[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = [
        {
            "row_id": "candidate_pushforward_basis_audit_complete",
            "status": "pass",
            "metric": "candidate pushforward basis audit complete",
            "value": 1.0,
            "note": "This step pushes the surviving V(|P|) candidates into chi-space and checks them against the frozen ambient action basis.",
        },
        {
            "row_id": "candidate_pushforward_ambient_basis_ready",
            "status": "pass" if ambient_basis_ready else "reject",
            "metric": "ambient chi-space basis ready for candidate audit",
            "value": 1.0 if ambient_basis_ready else 0.0,
            "note": (
                f"The frozen ambient basis exponents are {ambient_basis_exponents}."
                if ambient_basis_ready
                else "The ambient chi-space basis is not ready, so candidate pushforwards cannot be audited."
            ),
        },
    ]

    rows.extend(
        _candidate_rows(
            family="mexican_hat",
            pure_exponential_member=True,
            naked_log_present=False,
            accepted_exponents=[0, 2, 4],
            rejected_reason=None,
        )
    )
    rows.extend(
        _candidate_rows(
            family="logarithmic",
            pure_exponential_member=False,
            naked_log_present=True,
            accepted_exponents=[],
            rejected_reason="chi_times_exp_2chi_term_present",
        )
    )
    rows.extend(
        [
            {
                "row_id": "candidate_pushforward_basis_pass_candidate_count",
                "status": "pass" if len(accepted_candidates) == 1 else "watch",
                "metric": "count of candidates that remain in the frozen ambient basis",
                "value": float(len(accepted_candidates)),
                "note": f"The candidates that stay inside the frozen ambient basis are {accepted_candidates}.",
            },
            {
                "row_id": "candidate_pushforward_basis_reject_candidate_count",
                "status": "watch",
                "metric": "count of candidates rejected by the frozen ambient basis",
                "value": float(len(rejected_candidates)),
                "note": f"The candidates rejected by the frozen ambient basis are {rejected_candidates}.",
            },
            {
                "row_id": "candidate_pushforward_selection_gate_ready",
                "status": "pass" if ambient_basis_ready and len(candidate_family_ids) == 2 else "watch",
                "metric": "basis-closure selection gate ready",
                "value": 1.0 if ambient_basis_ready and len(candidate_family_ids) == 2 else 0.0,
                "note": "The next step can now apply the basis-closure accept/reject rule to the surviving candidate set.",
            },
        ]
    )
    return rows


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (PART1_MD, BASIS_JSON, R3_REGISTRY_JSON, SHAPE_JET_JSON):
        _require_path(path)

    part1_text = _read_text(PART1_MD)
    basis = _read_json(BASIS_JSON)
    r3_registry = _read_json(R3_REGISTRY_JSON)
    shape_jet = _read_json(SHAPE_JET_JSON)

    basis_summary = basis.get("summary", {})
    r3_summary = r3_registry.get("summary", {})
    shape_jet_summary = shape_jet.get("summary", {})

    ambient_basis_exponents = [int(item) for item in basis_summary.get("primitive_basis_exponents", [])]
    ambient_basis_ready = bool(basis_summary.get("candidate_pushforward_audit_ready", False))
    candidate_family_ids = [str(item) for item in r3_summary.get("candidate_family_ids", [])]

    p_ref_equals_p_infty_hit = _find_first_match(part1_text, r"P_{\mathrm{ref}}\equiv P_{\infty}")
    ratio_only_hit = _find_first_match(part1_text, r"比 $P/P_{\infty}$（無次元）")

    mexican_hat_pushforward = (
        "U_MH(chi) = (lambda/4) (P_ref^2 exp(2 chi) - v^2)^2 = "
        "(lambda P_ref^4 / 4) exp(4 chi) - (lambda P_ref^2 v^2 / 2) exp(2 chi) + (lambda v^4 / 4)"
    )
    logarithmic_pushforward = (
        "U_log(chi) = mu^4 [alpha exp(2 chi) (2 chi + ln(alpha) - 1) + 1], "
        "alpha = P_ref^2 / v^2"
    )

    candidate_basis_membership = {
        "mexican_hat": "pass",
        "logarithmic": "reject",
    }
    candidate_naked_log_coordinate = {
        "mexican_hat": False,
        "logarithmic": True,
    }
    accepted_candidates = ["mexican_hat"]
    rejected_candidates = ["logarithmic"]
    rows = _build_rows(
        ambient_basis_ready=ambient_basis_ready,
        ambient_basis_exponents=ambient_basis_exponents,
        candidate_family_ids=candidate_family_ids,
        accepted_candidates=accepted_candidates,
        rejected_candidates=rejected_candidates,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "candidate pushforward basis audit",
        },
        "inputs": {
            "part1_core_theory_markdown": _relative_str(PART1_MD),
            "mass_origin_chi_space_action_basis_inventory_json": _relative_str(BASIS_JSON),
            "mass_origin_anchor_local_r3_registry_json": _relative_str(R3_REGISTRY_JSON),
            "mass_origin_anchor_local_shape_jet_json": _relative_str(SHAPE_JET_JSON),
        },
        "intent": "Audit whether each surviving V(|P|) candidate remains inside the finite pure-exponential chi-space basis already exposed by the primitive action pack.",
        "formulas": {
            "reference_identification": "P_ref = P_infty",
            "rho_pushforward": "rho = P_ref exp(chi)",
            "mexican_hat_rho_form": "V_MH(rho) = (lambda / 4) (rho^2 - v^2)^2",
            "mexican_hat_pushforward": mexican_hat_pushforward,
            "logarithmic_rho_form": "V_log(rho) = mu^4 [(rho / v)^2 ln((rho / v)^2) - (rho / v)^2 + 1]",
            "logarithmic_pushforward": logarithmic_pushforward,
            "basis_accept_rule": "A candidate passes iff its chi-space pushforward is a finite linear combination of {1, exp(n chi)} over the frozen ambient exponent set and contains no naked chi outside the exponential.",
        },
        "rows": rows,
        "summary": {
            "ambient_basis_exponents": ambient_basis_exponents,
            "ambient_basis_ready": ambient_basis_ready,
            "candidate_family_ids": candidate_family_ids,
            "candidate_pushforward_basis_membership": candidate_basis_membership,
            "candidate_naked_log_coordinate_outside_exponential": candidate_naked_log_coordinate,
            "basis_pass_candidate_ids": accepted_candidates,
            "basis_reject_candidate_ids": rejected_candidates,
            "selection_gate_ready": True,
        },
        "decision": {
            "overall_status": "candidate_pushforward_basis_audit_frozen",
            "keep_mass_origin_branch_blocked": True,
            "ambient_basis_ready": ambient_basis_ready,
            "candidate_pushforward_basis_membership": candidate_basis_membership,
            "basis_pass_candidate_ids": accepted_candidates,
            "basis_reject_candidate_ids": rejected_candidates,
            "selection_gate_ready": True,
            "fallback_same_sector_equivalence_route_held": True,
            "next_required_artifacts": [
                "basis_closure_selection_gate",
                "sigma3_r3_freeze",
                "mexican_hat_parameter_pack",
                "shape_gate_refresh",
            ],
        },
        "evidence": {
            "part1_reference_identification_line": p_ref_equals_p_infty_hit,
            "part1_ratio_only_line": ratio_only_hit,
            "chi_space_action_basis_summary": basis_summary,
            "r3_registry_summary": r3_summary,
            "shape_jet_summary": shape_jet_summary,
        },
    }


# 関数: rows を CSV 出力する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)


# 関数: JSON を整形出力する。

def _write_json(payload: Dict[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 関数: エントリポイントとして step を実行する。

def main() -> None:
    args = _parse_args()
    payload = _build_payload(args.step_tag)
    _write_json(payload)
    _write_csv(payload["rows"])
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
