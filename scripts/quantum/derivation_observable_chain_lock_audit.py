#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
derivation_observable_chain_lock_audit.py

Step 8.7.13:
作用原理→観測式の導出チェーン（EL / nonrel / Born A/B / bridge / shared / derivation pack）
の整合を単一ゲートとして監査し、運用ロック判定を固定する。

出力:
  - output/public/quantum/derivation_observable_chain_lock_audit.json
  - output/public/quantum/derivation_observable_chain_lock_audit.csv
  - output/public/quantum/derivation_observable_chain_lock_audit.png
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_list_of_str(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(x) for x in value if isinstance(x, str)]


def _status_from_pass(passed: Optional[bool], gate_level: str) -> str:
    if passed is True:
        return "pass"
    if passed is None:
        return "unknown"
    if gate_level == "hard":
        return "reject"
    return "watch"


def _row(
    *,
    check_id: str,
    metric: str,
    value: Any,
    expected: Any,
    passed: Optional[bool],
    gate_level: str,
    source: str,
    note: str,
) -> Dict[str, Any]:
    return {
        "id": check_id,
        "metric": metric,
        "value": value,
        "expected": expected,
        "pass": passed,
        "gate_level": gate_level,
        "status": _status_from_pass(passed, gate_level),
        "score": 1.0 if passed is True else (0.5 if passed is None else 0.0),
        "source": source,
        "note": note,
    }


def _all_true(rows: Any) -> Optional[bool]:
    if not isinstance(rows, list) or not rows:
        return None
    flags: List[bool] = []
    for row in rows:
        if isinstance(row, dict):
            item = row.get("pass")
            if isinstance(item, bool):
                flags.append(item)
    if not flags:
        return None
    return all(flags)


def _bridge_watch_ids(bridge: Dict[str, Any]) -> List[str]:
    rows = bridge.get("rows") if isinstance(bridge.get("rows"), list) else []
    out: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "") == "watch":
            row_id = str(row.get("id") or "")
            if row_id:
                out.append(row_id)
    return sorted(set(out))


def build_payload(
    *,
    action_json: Path,
    nonrel_json: Path,
    born_pack_json: Path,
    born_ab_json: Path,
    bridge_json: Path,
    shared_json: Path,
    deriv_pack_json: Path,
    kwiat_watch_json: Optional[Path] = None,
    hom_watch_json: Optional[Path] = None,
) -> Dict[str, Any]:
    action = _read_json(action_json)
    nonrel = _read_json(nonrel_json)
    born_pack = _read_json(born_pack_json)
    born_ab = _read_json(born_ab_json)
    bridge = _read_json(bridge_json)
    shared = _read_json(shared_json)
    deriv_pack = _read_json(deriv_pack_json)
    kwiat_watch = _read_json(kwiat_watch_json) if kwiat_watch_json else {}
    hom_watch = _read_json(hom_watch_json) if hom_watch_json else {}

    action_audit = action.get("numerical_audit") if isinstance(action.get("numerical_audit"), dict) else {}
    action_status = str(action_audit.get("status") or "")
    action_fail_ids = _as_list_of_str(action_audit.get("fail_ids"))
    action_all_pass = _all_true(action_audit.get("criteria"))

    nonrel_decision = nonrel.get("decision") if isinstance(nonrel.get("decision"), dict) else {}
    nonrel_status = str(nonrel_decision.get("nonrelativistic_reduction_gate") or "")
    nonrel_fail_channels = _as_list_of_str(nonrel_decision.get("fail_channels"))
    nonrel_all_pass = _all_true(nonrel.get("criteria"))

    born_decision = born_pack.get("decision") if isinstance(born_pack.get("decision"), dict) else {}
    born_route_a = str(born_decision.get("route_a_gate") or "")

    born_ab_decision = born_ab.get("decision") if isinstance(born_ab.get("decision"), dict) else {}
    born_ab_route_a = str(born_ab_decision.get("route_a_gate") or "")
    born_ab_transition = str(born_ab_decision.get("transition") or "")
    born_ab_hard_fail = _as_list_of_str(born_ab_decision.get("hard_fail_ids"))
    born_ab_hard_unknown = _as_list_of_str(born_ab_decision.get("hard_unknown_ids"))
    born_ab_watchlist = _as_list_of_str(born_ab_decision.get("watchlist"))

    bridge_overall = bridge.get("overall") if isinstance(bridge.get("overall"), dict) else {}
    bridge_status = str(bridge_overall.get("status") or "")
    bridge_watchlist = _bridge_watch_ids(bridge)

    shared_overall = shared.get("overall") if isinstance(shared.get("overall"), dict) else {}
    shared_status = str(shared_overall.get("status") or "")

    deriv_decision = deriv_pack.get("decision") if isinstance(deriv_pack.get("decision"), dict) else {}
    deriv_route_a = str(deriv_decision.get("route_a_gate") or "")
    deriv_transition = str(deriv_decision.get("transition") or "")
    deriv_hard_fail = _as_list_of_str(deriv_decision.get("hard_fail_ids"))
    deriv_hard_unknown = _as_list_of_str(deriv_decision.get("hard_unknown_ids"))
    deriv_watchlist = _as_list_of_str(deriv_decision.get("watchlist"))

    kwiat_summary = kwiat_watch.get("summary") if isinstance(kwiat_watch.get("summary"), dict) else {}
    kwiat_watch_decision = str(kwiat_summary.get("decision") or "")
    kwiat_max_abs_z_any = kwiat_summary.get("max_abs_z_any")
    kwiat_hard_gate_applicable = bool(kwiat_summary.get("hard_gate_applicable")) if "hard_gate_applicable" in kwiat_summary else None
    kwiat_margin_to_hard = None
    if isinstance(kwiat_max_abs_z_any, (int, float)):
        kwiat_margin_to_hard = 3.0 - float(kwiat_max_abs_z_any)
    kwiat_nonhard_stable = (
        kwiat_watch_decision == "keep_watch_nonhard_gate"
        and isinstance(kwiat_max_abs_z_any, (int, float))
        and float(kwiat_max_abs_z_any) < 3.0
    )

    hom_summary = hom_watch.get("summary") if isinstance(hom_watch.get("summary"), dict) else {}
    hom_watch_decision = str(hom_summary.get("decision") or "")
    hom_ratio_detrended = hom_summary.get("detrended_ratio_median")
    hom_hard_gate_applicable = bool(hom_summary.get("hard_gate_applicable")) if "hard_gate_applicable" in hom_summary else None
    hom_margin_to_threshold = None
    if isinstance(hom_ratio_detrended, (int, float)):
        hom_margin_to_threshold = 1.0 - float(hom_ratio_detrended)
    hom_nonhard_stable = (
        hom_watch_decision == "keep_watch_nonhard_gate"
        and isinstance(hom_ratio_detrended, (int, float))
        and float(hom_ratio_detrended) < 1.0
    )

    checks: List[Dict[str, Any]] = [
        _row(
            check_id="action::status",
            metric="route_a_el_derivation_gate",
            value=action_status,
            expected="pass",
            passed=action_status == "pass",
            gate_level="hard",
            source="action_principle_el_derivation_audit",
            note="作用原理→EL 導出の数値監査 status。",
        ),
        _row(
            check_id="action::criteria",
            metric="all_action_criteria_pass",
            value=action_all_pass,
            expected=True,
            passed=action_all_pass,
            gate_level="hard",
            source="action_principle_el_derivation_audit",
            note="作用原理監査の criteria が全て pass。",
        ),
        _row(
            check_id="action::fail_ids",
            metric="fail_ids_n",
            value=len(action_fail_ids),
            expected=0,
            passed=len(action_fail_ids) == 0,
            gate_level="hard",
            source="action_principle_el_derivation_audit",
            note="作用原理監査の fail_ids が空。",
        ),
        _row(
            check_id="nonrel::status",
            metric="nonrelativistic_reduction_gate",
            value=nonrel_status,
            expected="pass",
            passed=nonrel_status == "pass",
            gate_level="hard",
            source="nonrelativistic_reduction_schrodinger_mapping_audit",
            note="2.6→2.5 非相対論極限の運用ゲート。",
        ),
        _row(
            check_id="nonrel::criteria",
            metric="all_nonrel_criteria_pass",
            value=nonrel_all_pass,
            expected=True,
            passed=nonrel_all_pass,
            gate_level="hard",
            source="nonrelativistic_reduction_schrodinger_mapping_audit",
            note="非相対論監査の criteria が全て pass。",
        ),
        _row(
            check_id="nonrel::fail_channels",
            metric="fail_channels_n",
            value=len(nonrel_fail_channels),
            expected=0,
            passed=len(nonrel_fail_channels) == 0,
            gate_level="hard",
            source="nonrelativistic_reduction_schrodinger_mapping_audit",
            note="非相対論監査の fail_channels が空。",
        ),
        _row(
            check_id="born_pack::route_a_gate",
            metric="route_a_gate",
            value=born_route_a,
            expected="A_continue",
            passed=born_route_a == "A_continue",
            gate_level="hard",
            source="born_route_a_proxy_constraints_pack",
            note="Born route-A proxy の gate 判定。",
        ),
        _row(
            check_id="born_ab::route_transition",
            metric="route_a_gate/transition",
            value=f"{born_ab_route_a}/{born_ab_transition}",
            expected="A_continue/A_stay",
            passed=(born_ab_route_a == "A_continue" and born_ab_transition == "A_stay"),
            gate_level="hard",
            source="quantum_connection_born_ab_gate",
            note="横断A/B 判定の route と transition。",
        ),
        _row(
            check_id="born_ab::hard_lists",
            metric="hard_fail_or_unknown_n",
            value=len(born_ab_hard_fail) + len(born_ab_hard_unknown),
            expected=0,
            passed=(len(born_ab_hard_fail) == 0 and len(born_ab_hard_unknown) == 0),
            gate_level="hard",
            source="quantum_connection_born_ab_gate",
            note="横断A/B 判定の hard fail/unknown が空。",
        ),
        _row(
            check_id="deriv_pack::route_transition",
            metric="route_a_gate/transition",
            value=f"{deriv_route_a}/{deriv_transition}",
            expected="A_continue/A_stay",
            passed=(deriv_route_a == "A_continue" and deriv_transition == "A_stay"),
            gate_level="hard",
            source="derivation_parameter_falsification_pack",
            note="導出由来パラメータ pack の最終判定。",
        ),
        _row(
            check_id="deriv_pack::hard_lists",
            metric="hard_fail_or_unknown_n",
            value=len(deriv_hard_fail) + len(deriv_hard_unknown),
            expected=0,
            passed=(len(deriv_hard_fail) == 0 and len(deriv_hard_unknown) == 0),
            gate_level="hard",
            source="derivation_parameter_falsification_pack",
            note="導出由来パラメータ pack の hard fail/unknown が空。",
        ),
        _row(
            check_id="cross::route_consistency",
            metric="born_ab_vs_deriv_route_transition",
            value=f"{born_ab_route_a}/{born_ab_transition} == {deriv_route_a}/{deriv_transition}",
            expected=True,
            passed=(born_ab_route_a == deriv_route_a and born_ab_transition == deriv_transition),
            gate_level="hard",
            source="cross",
            note="A/B横断判定と導出packの route/transition が一致。",
        ),
        _row(
            check_id="shared::overall_status",
            metric="shared_kpi_overall",
            value=shared_status,
            expected="pass",
            passed=shared_status == "pass",
            gate_level="hard",
            source="quantum_connection_shared_kpi",
            note="量子接続共通KPIの overall.status。",
        ),
        _row(
            check_id="bridge::overall_not_reject",
            metric="bridge_overall_status",
            value=bridge_status,
            expected="pass or watch",
            passed=(bridge_status in {"pass", "watch"}),
            gate_level="hard",
            source="quantum_connection_bridge_table",
            note="bridge table は reject でないこと（watch は許容）。",
        ),
        _row(
            check_id="watch::kwiat_nonhard_stability",
            metric="kwiat_max_abs_z_any_margin_to_3sigma",
            value=kwiat_margin_to_hard,
            expected="> 0 (non-hard watch stability)",
            passed=kwiat_nonhard_stable if kwiat_watch_decision else None,
            gate_level="soft",
            source="bell_kwiat_delay_signature_watch_audit",
            note="Kwiat watch項目の非hard監視安定性（max_abs_z_any<3）。",
        ),
        _row(
            check_id="watch::hom_nonhard_stability",
            metric="hom_detrended_ratio_margin_to_1",
            value=hom_margin_to_threshold,
            expected="> 0 (non-hard watch stability)",
            passed=hom_nonhard_stable if hom_watch_decision else None,
            gate_level="soft",
            source="hom_noise_psd_watch_audit",
            note="HOM watch項目の非hard監視安定性（detrended_ratio_median<1）。",
        ),
    ]

    hard_fail_ids = [str(row.get("id") or "") for row in checks if str(row.get("gate_level") or "") == "hard" and row.get("pass") is not True]
    watchlist = sorted(set(born_ab_watchlist + deriv_watchlist + bridge_watchlist))
    route_a_gate = "A_reject" if hard_fail_ids else "A_continue"
    transition = "A_to_B" if route_a_gate == "A_reject" else "A_stay"
    overall_status = "reject" if hard_fail_ids else ("watch" if watchlist else "pass")

    input_paths = {
        "action_principle_el_derivation_audit_json": action_json,
        "nonrelativistic_reduction_schrodinger_mapping_audit_json": nonrel_json,
        "born_route_a_proxy_constraints_pack_json": born_pack_json,
        "quantum_connection_born_ab_gate_json": born_ab_json,
        "quantum_connection_bridge_table_json": bridge_json,
        "quantum_connection_shared_kpi_json": shared_json,
        "derivation_parameter_falsification_pack_json": deriv_pack_json,
    }
    if kwiat_watch_json is not None:
        input_paths["bell_kwiat_delay_signature_watch_audit_json"] = kwiat_watch_json
    if hom_watch_json is not None:
        input_paths["hom_noise_psd_watch_audit_json"] = hom_watch_json

    nonhard_watch_stable: List[str] = []
    if "kwiat2013_prl111_130406_05082013_15:delay_signature" in watchlist and kwiat_nonhard_stable:
        nonhard_watch_stable.append("kwiat2013_prl111_130406_05082013_15:delay_signature")
    if "hom_noise_psd_shape" in watchlist and hom_nonhard_stable:
        nonhard_watch_stable.append("hom_noise_psd_shape")

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 8, "step": "8.7.13", "name": "Derivation-to-observable chain lock audit"},
        "intent": (
            "Re-freeze one machine-readable decision for the derivation chain "
            "(EL/nonrel/Born A-B/bridge/shared/derivation-pack) with hard consistency gates."
        ),
        "inputs": {key: _rel(path) for key, path in input_paths.items()},
        "input_hashes_sha256": {key: _sha256(path) for key, path in input_paths.items()},
        "checks": checks,
        "decision": {
            "route_a_gate": route_a_gate,
            "transition": transition,
            "overall_status": overall_status,
            "hard_fail_ids": hard_fail_ids,
            "watchlist": watchlist,
            "watch_convergence": {
                "watch_n": len(watchlist),
                "nonhard_stable_n": len(nonhard_watch_stable),
                "nonhard_stable_watchlist": sorted(nonhard_watch_stable),
                "remaining_watch_n": max(0, len(watchlist) - len(nonhard_watch_stable)),
                "all_nonhard_watch_stable": (len(watchlist) > 0 and len(nonhard_watch_stable) == len(watchlist)),
            },
            "rule": "Reject if any hard consistency gate fails; otherwise A_stay and track watchlist.",
        },
        "diagnostics": {
            "source_route_a_gate": {
                "born_pack": born_route_a,
                "born_ab": born_ab_route_a,
                "derivation_pack": deriv_route_a,
            },
            "source_transition": {
                "born_ab": born_ab_transition,
                "derivation_pack": deriv_transition,
            },
            "source_watchlist": {
                "born_ab": born_ab_watchlist,
                "derivation_pack": deriv_watchlist,
                "bridge_watch_rows": bridge_watchlist,
            },
            "bridge_status": bridge_status,
            "shared_status": shared_status,
            "watch_stability": {
                "kwiat_delay_signature": {
                    "decision": kwiat_watch_decision,
                    "hard_gate_applicable": kwiat_hard_gate_applicable,
                    "max_abs_z_any": kwiat_max_abs_z_any,
                    "margin_to_hard_z3": kwiat_margin_to_hard,
                    "stable_nonhard_watch": kwiat_nonhard_stable if kwiat_watch_decision else None,
                },
                "hom_noise_psd_shape": {
                    "decision": hom_watch_decision,
                    "hard_gate_applicable": hom_hard_gate_applicable,
                    "detrended_ratio_median": hom_ratio_detrended,
                    "margin_to_ratio_threshold_1": hom_margin_to_threshold,
                    "stable_nonhard_watch": hom_nonhard_stable if hom_watch_decision else None,
                },
            },
        },
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "metric",
                "value",
                "expected",
                "pass",
                "gate_level",
                "status",
                "score",
                "source",
                "note",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot(path: Path, payload: Dict[str, Any]) -> None:
    rows = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    labels: List[str] = []
    scores: List[float] = []
    colors: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        labels.append(str(row.get("id") or ""))
        score = row.get("score")
        score_f = float(score) if isinstance(score, (int, float)) else np.nan
        scores.append(score_f)
        status = str(row.get("status") or "")
        if status == "pass":
            colors.append("#2f9e44")
        elif status == "reject":
            colors.append("#e03131")
        elif status == "watch":
            colors.append("#f2c94c")
        else:
            colors.append("#9ca3af")

    fig_h = max(4.8, 0.34 * len(labels) + 1.6)
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12.4, fig_h), dpi=180)
    ax.barh(y, scores, color=colors)
    ax.axvline(1.0, linestyle="--", color="#6b7280", linewidth=1.2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("consistency score (1=pass, 0=reject)")
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    title_status = str(decision.get("overall_status") or "unknown")
    title_route = str(decision.get("route_a_gate") or "unknown")
    title_trans = str(decision.get("transition") or "unknown")
    ax.set_title(f"Derivation-observable chain lock audit ({title_status}; {title_route}/{title_trans})")
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate derivation-observable chain lock audit (Step 8.7.13).")
    parser.add_argument(
        "--action",
        default=str(ROOT / "output" / "public" / "quantum" / "action_principle_el_derivation_audit.json"),
        help="Input action-principle audit JSON.",
    )
    parser.add_argument(
        "--nonrel",
        default=str(ROOT / "output" / "public" / "quantum" / "nonrelativistic_reduction_schrodinger_mapping_audit.json"),
        help="Input nonrelativistic reduction audit JSON.",
    )
    parser.add_argument(
        "--born-pack",
        default=str(ROOT / "output" / "public" / "quantum" / "born_route_a_proxy_constraints_pack.json"),
        help="Input Born route-A proxy pack JSON.",
    )
    parser.add_argument(
        "--born-ab",
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_connection_born_ab_gate.json"),
        help="Input cross-channel Born A/B gate JSON.",
    )
    parser.add_argument(
        "--bridge",
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_connection_bridge_table.json"),
        help="Input bridge table JSON.",
    )
    parser.add_argument(
        "--shared",
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_connection_shared_kpi.json"),
        help="Input shared KPI JSON.",
    )
    parser.add_argument(
        "--deriv-pack",
        default=str(ROOT / "output" / "public" / "quantum" / "derivation_parameter_falsification_pack.json"),
        help="Input derivation-parameter falsification pack JSON.",
    )
    parser.add_argument(
        "--kwiat-watch",
        default=str(ROOT / "output" / "public" / "quantum" / "bell_kwiat_delay_signature_watch_audit.json"),
        help="Input Kwiat watch audit JSON (optional stability diagnostics).",
    )
    parser.add_argument(
        "--hom-watch",
        default=str(ROOT / "output" / "public" / "quantum" / "hom_noise_psd_watch_audit.json"),
        help="Input HOM watch audit JSON (optional stability diagnostics).",
    )
    parser.add_argument(
        "--out-json",
        default=str(ROOT / "output" / "public" / "quantum" / "derivation_observable_chain_lock_audit.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-csv",
        default=str(ROOT / "output" / "public" / "quantum" / "derivation_observable_chain_lock_audit.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-png",
        default=str(ROOT / "output" / "public" / "quantum" / "derivation_observable_chain_lock_audit.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args(argv)

    def _resolve(path_text: str) -> Path:
        p = Path(path_text)
        if p.is_absolute():
            return p.resolve()
        return (ROOT / p).resolve()

    action = _resolve(args.action)
    nonrel = _resolve(args.nonrel)
    born_pack = _resolve(args.born_pack)
    born_ab = _resolve(args.born_ab)
    bridge = _resolve(args.bridge)
    shared = _resolve(args.shared)
    deriv_pack = _resolve(args.deriv_pack)
    kwiat_watch = _resolve(args.kwiat_watch) if str(args.kwiat_watch).strip() else None
    hom_watch = _resolve(args.hom_watch) if str(args.hom_watch).strip() else None
    out_json = _resolve(args.out_json)
    out_csv = _resolve(args.out_csv)
    out_png = _resolve(args.out_png)

    for p in [action, nonrel, born_pack, born_ab, bridge, shared, deriv_pack]:
        if not p.exists():
            raise FileNotFoundError(f"required input not found: {_rel(p)}")

    payload = build_payload(
        action_json=action,
        nonrel_json=nonrel,
        born_pack_json=born_pack,
        born_ab_json=born_ab,
        bridge_json=bridge,
        shared_json=shared,
        deriv_pack_json=deriv_pack,
        kwiat_watch_json=kwiat_watch if (kwiat_watch and kwiat_watch.exists()) else None,
        hom_watch_json=hom_watch if (hom_watch and hom_watch.exists()) else None,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    _write_csv(out_csv, checks)
    _plot(out_png, payload)

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")
    print(f"[ok] wrote: {_rel(out_png)}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_derivation_observable_chain_lock_audit",
                "phase": "8.7.13",
                "inputs": payload.get("inputs"),
                "outputs": {
                    "derivation_observable_chain_lock_audit_json": _rel(out_json),
                    "derivation_observable_chain_lock_audit_csv": _rel(out_csv),
                    "derivation_observable_chain_lock_audit_png": _rel(out_png),
                },
                "decision": payload.get("decision"),
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
