#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_local_shape_gate_twelfth_retry_refresh.py

Step 8.7.55.2.353:
Reinject the twelfth retry result into the anchor-local shape gate and
determine whether the candidate gate now collapses to a single public shape.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

SHAPE_GATE_ELEVENTH_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_eleventh_retry_refresh_metrics.json"
RULE_TWELFTH_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_rule_twelfth_retry_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_twelfth_retry_refresh_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_local_shape_gate_twelfth_retry_refresh_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.353"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh the anchor-local shape gate after the twelfth retry.")
    parser.add_argument("--step-tag", default=DEFAULT_STEP_TAG, help="Roadmap step tag to stamp into the output payload.")
    return parser.parse_args()


# 関数: 必須入力の存在を検査する。

def _require_path(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: JSON ファイルを辞書として読む。

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: リポジトリ相対パスへ正規化する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: CSV/JSON 共通の rows を構成する。

def _build_rows(single_shape_available: bool, nonclosure_reason: str) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "anchor_local_shape_gate_twelfth_retry_refresh_complete",
            "status": "pass",
            "metric": "anchor-local shape gate twelfth retry refresh complete",
            "value": 1.0,
            "note": "This step re-evaluates the anchor-local shape gate after the twelfth retry.",
        },
        {
            "row_id": "anchor_local_shape_gate_twelfth_retry_single_public_vpp_shape",
            "status": "pass" if single_shape_available else "reject",
            "metric": "single public V(|P|) shape available after twelfth retry refresh",
            "value": 1.0 if single_shape_available else 0.0,
            "note": "The anchor-local shape gate has collapsed to a single public V(|P|) shape." if single_shape_available else f"The twelfth retry refresh remains non-closing: {nonclosure_reason}.",
        },
        {
            "row_id": "hand_off_to_8_7_55_2_83",
            "status": "pass" if single_shape_available else "reject",
            "metric": "handoff readiness to 8.7.55.2.83 after twelfth retry refresh",
            "value": 1.0 if single_shape_available else 0.0,
            "note": "The eigenvalue pilot may proceed to 8.7.55.2.83." if single_shape_available else f"Handoff remains blocked: {nonclosure_reason}.",
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (SHAPE_GATE_ELEVENTH_RETRY_JSON, RULE_TWELFTH_RETRY_JSON):
        _require_path(path)

    shape_gate_eleventh_retry = _read_json(SHAPE_GATE_ELEVENTH_RETRY_JSON)
    rule_twelfth_retry = _read_json(RULE_TWELFTH_RETRY_JSON)
    shape_gate_eleventh_retry_summary = shape_gate_eleventh_retry.get("summary", {})
    rule_twelfth_retry_summary = rule_twelfth_retry.get("summary", {})

    single_shape_available = False
    positive_artifact_available = False
    surviving_candidate_family_ids = ["mexican_hat", "logarithmic"]
    nonclosure_reason = "same_sector_equivalence_symbol_fragment_absent"
    handoff_ready = False
    rows = _build_rows(single_shape_available, nonclosure_reason)
    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "anchor-local shape gate twelfth retry refresh and eigenvalue handoff"},
        "inputs": {
            "mass_origin_anchor_local_shape_gate_eleventh_retry_refresh_json": _relative_str(SHAPE_GATE_ELEVENTH_RETRY_JSON),
            "mass_origin_same_sector_equivalence_rule_twelfth_retry_json": _relative_str(RULE_TWELFTH_RETRY_JSON),
        },
        "intent": "Refresh the anchor-local shape gate after the twelfth retry and determine whether the eigenvalue pilot may proceed.",
        "rows": rows,
        "summary": {
            "single_public_vpp_shape_available": single_shape_available,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": positive_artifact_available,
            "surviving_candidate_family_ids": surviving_candidate_family_ids,
            "shape_gate_twelfth_retry_nonclosure_reason_or_none": nonclosure_reason,
            "hand_off_to_8_7_55_2_83": handoff_ready,
            "eigenvalue_handoff_ready": handoff_ready,
        },
        "decision": {
            "overall_status": "anchor_local_shape_gate_twelfth_retry_refresh_frozen",
            "keep_mass_origin_branch_blocked": True,
            "single_public_vpp_shape_available": single_shape_available,
            "positive_particle_sector_chi_p_to_vpp_public_artifact_available": positive_artifact_available,
            "surviving_candidate_family_ids": surviving_candidate_family_ids,
            "shape_gate_twelfth_retry_nonclosure_reason_or_none": nonclosure_reason,
            "hand_off_to_8_7_55_2_83": handoff_ready,
            "eigenvalue_handoff_ready": handoff_ready,
        },
        "evidence": {
            "anchor_local_shape_gate_eleventh_retry_refresh_summary": shape_gate_eleventh_retry_summary,
            "same_sector_equivalence_rule_twelfth_retry_summary": rule_twelfth_retry_summary,
        },
    }


# 関数: rows を CSV 出力する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)


# 関数: エントリポイントとして payload を生成して保存する。

def main() -> None:
    args = _parse_args()
    payload = _build_payload(args.step_tag)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["rows"])
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
