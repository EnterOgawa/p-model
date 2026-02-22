#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lagrangian_noether_observable_closure_audit.py

Step 8.7.21.1:
L_total -> EL -> observables の閉包を、Noether 条件と非相対論写像を含む
単一監査パックとして固定出力する。

出力:
  - output/public/quantum/lagrangian_noether_observable_closure_audit.json
  - output/public/quantum/lagrangian_noether_observable_closure_audit.csv
  - output/public/quantum/lagrangian_noether_observable_closure_audit.png
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
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _all_true(rows: Any) -> Optional[bool]:
    if not isinstance(rows, list) or not rows:
        return None
    flags: List[bool] = []
    for row in rows:
        if isinstance(row, dict):
            p = row.get("pass")
            if isinstance(p, bool):
                flags.append(p)
    if not flags:
        return None
    return all(flags)


def _status_from_pass(passed: Optional[bool], gate_level: str) -> str:
    if passed is True:
        return "pass"
    if passed is None:
        return "unknown"
    if gate_level == "hard":
        return "reject"
    return "watch"


@dataclass
class CheckRow:
    cid: str
    metric: str
    value: Any
    expected: Any
    passed: Optional[bool]
    gate_level: str
    source: str
    note: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.cid,
            "metric": self.metric,
            "value": self.value,
            "expected": self.expected,
            "pass": self.passed,
            "gate_level": self.gate_level,
            "status": _status_from_pass(self.passed, self.gate_level),
            "score": 1.0 if self.passed is True else (0.5 if self.passed is None else 0.0),
            "source": self.source,
            "note": self.note,
        }


def _get_action_noether(criteria: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(criteria, list):
        return out
    for row in criteria:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id") or "")
        if rid:
            out[rid] = row
    return out


def build_payload(
    *,
    action_json: Path,
    nonrel_json: Path,
    deriv_pack_json: Path,
    chain_lock_json: Path,
) -> Dict[str, Any]:
    action = _read_json(action_json)
    nonrel = _read_json(nonrel_json)
    deriv_pack = _read_json(deriv_pack_json)
    chain = _read_json(chain_lock_json)

    action_eq = action.get("equations") if isinstance(action.get("equations"), dict) else {}
    action_audit = action.get("numerical_audit") if isinstance(action.get("numerical_audit"), dict) else {}
    action_decision = action.get("decision") if isinstance(action.get("decision"), dict) else {}
    action_criteria = action_audit.get("criteria") if isinstance(action_audit.get("criteria"), list) else []
    action_noether = _get_action_noether(action_criteria)
    action_fail_ids = action_audit.get("fail_ids") if isinstance(action_audit.get("fail_ids"), list) else []

    nonrel_decision = nonrel.get("decision") if isinstance(nonrel.get("decision"), dict) else {}
    nonrel_channels = nonrel.get("channels") if isinstance(nonrel.get("channels"), list) else []
    nonrel_criteria = nonrel.get("criteria") if isinstance(nonrel.get("criteria"), list) else []
    nonrel_channel_names = {
        str(row.get("channel"))
        for row in nonrel_channels
        if isinstance(row, dict) and isinstance(row.get("channel"), str)
    }

    deriv_decision = deriv_pack.get("decision") if isinstance(deriv_pack.get("decision"), dict) else {}
    deriv_summary = deriv_pack.get("channel_summary") if isinstance(deriv_pack.get("channel_summary"), dict) else {}
    deriv_channel = deriv_summary.get("derivation") if isinstance(deriv_summary.get("derivation"), dict) else {}
    deriv_status_counts = deriv_channel.get("status_counts") if isinstance(deriv_channel.get("status_counts"), dict) else {}
    deriv_rows_n = int(deriv_channel.get("rows_n", 0)) if isinstance(deriv_channel.get("rows_n"), (int, float)) else 0
    deriv_pass_n = int(deriv_status_counts.get("pass", 0)) if isinstance(deriv_status_counts.get("pass"), (int, float)) else 0

    chain_decision = chain.get("decision") if isinstance(chain.get("decision"), dict) else {}
    chain_watch_conv = chain_decision.get("watch_convergence") if isinstance(chain_decision.get("watch_convergence"), dict) else {}

    required_equations = [
        "lagrangian_density",
        "el_for_P_conjugate",
        "el_for_A_nu",
        "continuity",
    ]
    eq_missing = [key for key in required_equations if not str(action_eq.get(key) or "").strip()]

    required_nonrel_channels = {
        "cow_neutron",
        "atom_gravimeter",
        "optical_clock_leveling_proxy",
    }
    missing_nonrel_channels = sorted(required_nonrel_channels - nonrel_channel_names)

    noether_gauge = action_noether.get("noether_current_gauge_invariance", {})
    noether_real = action_noether.get("noether_current_realness", {})
    noether_value = noether_gauge.get("value")
    noether_th = noether_gauge.get("threshold")
    noether_real_value = noether_real.get("value")
    noether_real_th = noether_real.get("threshold")
    noether_margin = (
        float(noether_th) - float(noether_value)
        if isinstance(noether_value, (int, float)) and isinstance(noether_th, (int, float))
        else None
    )
    noether_real_margin = (
        float(noether_real_th) - float(noether_real_value)
        if isinstance(noether_real_value, (int, float)) and isinstance(noether_real_th, (int, float))
        else None
    )

    closure_matrix = [
        {
            "observable": "gravity_acceleration",
            "route_id": "EL(P) + nonrel(cow_neutron)",
            "required_keys": ["el_for_P_conjugate", "cow_neutron"],
            "present": bool("el_for_P_conjugate" in action_eq and "cow_neutron" in nonrel_channel_names),
            "route_count": 1,
            "unique_route": True,
        },
        {
            "observable": "clock_rate_mapping",
            "route_id": "EL(P) + nonrel(optical_clock_leveling_proxy)",
            "required_keys": ["el_for_P_conjugate", "optical_clock_leveling_proxy"],
            "present": bool("el_for_P_conjugate" in action_eq and "optical_clock_leveling_proxy" in nonrel_channel_names),
            "route_count": 1,
            "unique_route": True,
        },
        {
            "observable": "optical_refractive_mapping",
            "route_id": "EL(A_mu) + noether + nonrel(atom_gravimeter)",
            "required_keys": ["el_for_A_nu", "noether_current_gauge_invariance", "atom_gravimeter"],
            "present": bool(
                "el_for_A_nu" in action_eq
                and "noether_current_gauge_invariance" in action_noether
                and "atom_gravimeter" in nonrel_channel_names
            ),
            "route_count": 1,
            "unique_route": True,
        },
    ]
    closure_present_n = sum(1 for row in closure_matrix if bool(row.get("present")))
    closure_unique_n = sum(1 for row in closure_matrix if bool(row.get("unique_route")))

    checks: List[CheckRow] = [
        CheckRow(
            cid="action::route_a_gate",
            metric="route_a_el_derivation_gate",
            value=str(action_decision.get("route_a_el_derivation_gate") or ""),
            expected="pass",
            passed=str(action_decision.get("route_a_el_derivation_gate") or "") == "pass",
            gate_level="hard",
            source="action_principle_el_derivation_audit",
            note="EL 導出ゲートが pass。",
        ),
        CheckRow(
            cid="action::criteria_all_pass",
            metric="all_action_criteria_pass",
            value=_all_true(action_criteria),
            expected=True,
            passed=_all_true(action_criteria),
            gate_level="hard",
            source="action_principle_el_derivation_audit",
            note="作用原理監査 criteria が全て pass。",
        ),
        CheckRow(
            cid="action::required_equations",
            metric="missing_equations_n",
            value=len(eq_missing),
            expected=0,
            passed=(len(eq_missing) == 0),
            gate_level="hard",
            source="action_principle_el_derivation_audit",
            note="L_total→EL→Noether に必要な式キーが欠けていない。",
        ),
        CheckRow(
            cid="action::noether_gauge",
            metric="noether_current_gauge_margin",
            value=noether_margin,
            expected="> 0",
            passed=(noether_margin is not None and noether_margin > 0.0),
            gate_level="hard",
            source="action_principle_el_derivation_audit",
            note="Noether 電流のゲージ不変性マージン。",
        ),
        CheckRow(
            cid="action::noether_realness",
            metric="noether_current_realness_margin",
            value=noether_real_margin,
            expected="> 0",
            passed=(noether_real_margin is not None and noether_real_margin > 0.0),
            gate_level="hard",
            source="action_principle_el_derivation_audit",
            note="Noether 電流の実数性マージン。",
        ),
        CheckRow(
            cid="nonrel::route_gate",
            metric="nonrelativistic_reduction_gate",
            value=str(nonrel_decision.get("nonrelativistic_reduction_gate") or ""),
            expected="pass",
            passed=str(nonrel_decision.get("nonrelativistic_reduction_gate") or "") == "pass",
            gate_level="hard",
            source="nonrelativistic_reduction_schrodinger_mapping_audit",
            note="非相対論写像ゲートが pass。",
        ),
        CheckRow(
            cid="nonrel::criteria_all_pass",
            metric="all_nonrel_criteria_pass",
            value=_all_true(nonrel_criteria),
            expected=True,
            passed=_all_true(nonrel_criteria),
            gate_level="hard",
            source="nonrelativistic_reduction_schrodinger_mapping_audit",
            note="非相対論写像 criteria が全て pass。",
        ),
        CheckRow(
            cid="nonrel::required_channels",
            metric="missing_nonrel_channels_n",
            value=len(missing_nonrel_channels),
            expected=0,
            passed=(len(missing_nonrel_channels) == 0),
            gate_level="hard",
            source="nonrelativistic_reduction_schrodinger_mapping_audit",
            note="観測写像に必要な channel が揃っている。",
        ),
        CheckRow(
            cid="deriv_pack::hard_fail_unknown",
            metric="hard_fail_or_unknown_n",
            value=int(
                len(deriv_decision.get("hard_fail_ids") or [])
                + len(deriv_decision.get("hard_unknown_ids") or [])
            ),
            expected=0,
            passed=(
                len(deriv_decision.get("hard_fail_ids") or []) == 0
                and len(deriv_decision.get("hard_unknown_ids") or []) == 0
            ),
            gate_level="hard",
            source="derivation_parameter_falsification_pack",
            note="導出packの hard fail/unknown が空。",
        ),
        CheckRow(
            cid="deriv_pack::derivation_channel_pass",
            metric="derivation_channel_pass_n/rows_n",
            value=f"{deriv_pass_n}/{deriv_rows_n}",
            expected="all pass",
            passed=(deriv_rows_n > 0 and deriv_pass_n == deriv_rows_n),
            gate_level="hard",
            source="derivation_parameter_falsification_pack",
            note="導出channel行が全件 pass。",
        ),
        CheckRow(
            cid="closure::observables_present",
            metric="present_routes_n/required_n",
            value=f"{closure_present_n}/{len(closure_matrix)}",
            expected="all present",
            passed=(closure_present_n == len(closure_matrix)),
            gate_level="hard",
            source="lagrangian_noether_observable_closure_audit",
            note="3観測量の L_total→EL→観測 route が揃っている。",
        ),
        CheckRow(
            cid="closure::observables_unique",
            metric="unique_routes_n/required_n",
            value=f"{closure_unique_n}/{len(closure_matrix)}",
            expected="all unique",
            passed=(closure_unique_n == len(closure_matrix)),
            gate_level="hard",
            source="lagrangian_noether_observable_closure_audit",
            note="3観測量の route を一意（route_count=1）で固定。",
        ),
        CheckRow(
            cid="chain::hard_fail",
            metric="hard_fail_ids_n",
            value=len(chain_decision.get("hard_fail_ids") or []),
            expected=0,
            passed=(len(chain_decision.get("hard_fail_ids") or []) == 0),
            gate_level="hard",
            source="derivation_observable_chain_lock_audit",
            note="導出チェーン全体で hard fail が無い。",
        ),
        CheckRow(
            cid="chain::watch_convergence",
            metric="remaining_watch_n",
            value=chain_watch_conv.get("remaining_watch_n"),
            expected=0,
            passed=(chain_watch_conv.get("remaining_watch_n") == 0),
            gate_level="watch",
            source="derivation_observable_chain_lock_audit",
            note="non-hard watch が安定収束している。",
        ),
    ]

    rows = [row.as_dict() for row in checks]
    hard_fail_ids = [str(r["id"]) for r in rows if r.get("gate_level") == "hard" and r.get("pass") is not True]
    watch_ids = [str(r["id"]) for r in rows if r.get("gate_level") == "watch" and r.get("pass") is not True]
    overall_status = "reject" if hard_fail_ids else ("watch" if watch_ids else "pass")

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 8, "step": "8.7.21.1", "name": "Lagrangian-Noether observable closure audit"},
        "intent": (
            "Freeze one machine-readable closure decision for L_total -> EL -> observables "
            "with Noether constraints and nonrelativistic mapping."
        ),
        "inputs": {
            "action_principle_el_derivation_audit_json": _rel(action_json),
            "nonrelativistic_reduction_schrodinger_mapping_audit_json": _rel(nonrel_json),
            "derivation_parameter_falsification_pack_json": _rel(deriv_pack_json),
            "derivation_observable_chain_lock_audit_json": _rel(chain_lock_json),
        },
        "input_hashes_sha256": {
            "action_principle_el_derivation_audit_json": _sha256(action_json),
            "nonrelativistic_reduction_schrodinger_mapping_audit_json": _sha256(nonrel_json),
            "derivation_parameter_falsification_pack_json": _sha256(deriv_pack_json),
            "derivation_observable_chain_lock_audit_json": _sha256(chain_lock_json),
        },
        "assumptions": [
            "Boundary variation is fixed to zero at the audit boundary.",
            "No higher-derivative correction terms are allowed in the minimal route-A closure.",
            "Observable mapping uses one fixed route per observable (no branch re-fit).",
        ],
        "checks": rows,
        "closure_matrix": closure_matrix,
        "diagnostics": {
            "missing_equations": eq_missing,
            "missing_nonrel_channels": missing_nonrel_channels,
            "action_fail_ids": action_fail_ids,
            "deriv_watchlist": list(deriv_decision.get("watchlist") or []),
            "chain_watchlist": list(chain_decision.get("watchlist") or []),
        },
        "decision": {
            "overall_status": overall_status,
            "hard_fail_ids": hard_fail_ids,
            "watch_ids": watch_ids,
            "route_a_gate": str(deriv_decision.get("route_a_gate") or ""),
            "transition": str(deriv_decision.get("transition") or ""),
            "rule": (
                "Reject if any hard closure check fails; "
                "watch if only non-hard watch-convergence check fails; otherwise pass."
            ),
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
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    matrix = payload.get("closure_matrix") if isinstance(payload.get("closure_matrix"), list) else []

    labels: List[str] = []
    scores: List[float] = []
    colors: List[str] = []
    for row in checks:
        if not isinstance(row, dict):
            continue
        labels.append(str(row.get("id") or ""))
        score = row.get("score")
        scores.append(float(score) if isinstance(score, (int, float)) else math.nan)
        status = str(row.get("status") or "")
        if status == "pass":
            colors.append("#2f9e44")
        elif status == "watch":
            colors.append("#eab308")
        elif status == "reject":
            colors.append("#dc2626")
        else:
            colors.append("#94a3b8")

    obs_labels: List[str] = []
    obs_values: List[float] = []
    for row in matrix:
        if not isinstance(row, dict):
            continue
        obs_labels.append(str(row.get("observable") or ""))
        present = bool(row.get("present"))
        unique = bool(row.get("unique_route"))
        obs_values.append(1.0 if present and unique else (0.5 if present else 0.0))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12.4, 8.2), dpi=180, gridspec_kw={"height_ratios": [3.0, 1.4]})

    y = np.arange(len(labels))
    ax0.barh(y, scores, color=colors)
    ax0.set_yticks(y, labels)
    ax0.set_xlim(0.0, 1.05)
    ax0.axvline(1.0, linestyle="--", color="#6b7280", linewidth=1.2)
    ax0.set_xlabel("check score (1=pass, 0.5=watch, 0=reject)")
    ax0.set_title("L_total -> EL -> observables closure audit (Noether + nonrel mapping)")
    ax0.grid(axis="x", alpha=0.25, linestyle=":")

    x = np.arange(len(obs_labels))
    ax1.bar(x, obs_values, color="#2563eb")
    ax1.set_xticks(x, obs_labels, rotation=10, ha="right")
    ax1.set_ylim(0.0, 1.05)
    ax1.axhline(1.0, linestyle="--", color="#6b7280", linewidth=1.2)
    ax1.set_ylabel("route lock score")
    ax1.set_title("Observable route uniqueness lock")
    ax1.grid(axis="y", alpha=0.25, linestyle=":")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Lagrangian-Noether observable closure audit pack (Step 8.7.21.1).")
    parser.add_argument(
        "--action-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "action_principle_el_derivation_audit.json"),
        help="Input action-principle EL derivation audit JSON.",
    )
    parser.add_argument(
        "--nonrel-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "nonrelativistic_reduction_schrodinger_mapping_audit.json"),
        help="Input nonrelativistic reduction audit JSON.",
    )
    parser.add_argument(
        "--deriv-pack-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "derivation_parameter_falsification_pack.json"),
        help="Input derivation-parameter falsification pack JSON.",
    )
    parser.add_argument(
        "--chain-lock-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "derivation_observable_chain_lock_audit.json"),
        help="Input derivation-observable chain lock audit JSON.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "lagrangian_noether_observable_closure_audit.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "lagrangian_noether_observable_closure_audit.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "lagrangian_noether_observable_closure_audit.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args(argv)

    action_json = Path(args.action_json)
    nonrel_json = Path(args.nonrel_json)
    deriv_pack_json = Path(args.deriv_pack_json)
    chain_lock_json = Path(args.chain_lock_json)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)

    for name, path in [
        ("action-json", action_json),
        ("nonrel-json", nonrel_json),
        ("deriv-pack-json", deriv_pack_json),
        ("chain-lock-json", chain_lock_json),
        ("out-json", out_json),
        ("out-csv", out_csv),
        ("out-png", out_png),
    ]:
        if not path.is_absolute():
            resolved = (ROOT / path).resolve()
            if name == "action-json":
                action_json = resolved
            elif name == "nonrel-json":
                nonrel_json = resolved
            elif name == "deriv-pack-json":
                deriv_pack_json = resolved
            elif name == "chain-lock-json":
                chain_lock_json = resolved
            elif name == "out-json":
                out_json = resolved
            elif name == "out-csv":
                out_csv = resolved
            elif name == "out-png":
                out_png = resolved

    for p in [action_json, nonrel_json, deriv_pack_json, chain_lock_json]:
        if not p.exists():
            print(f"[error] missing input: {_rel(p)}")
            return 2

    payload = build_payload(
        action_json=action_json,
        nonrel_json=nonrel_json,
        deriv_pack_json=deriv_pack_json,
        chain_lock_json=chain_lock_json,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    _write_csv(out_csv, rows if isinstance(rows, list) else [])
    _plot(out_png, payload)

    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")
    print(f"[ok] wrote: {_rel(out_png)}")
    print(f"[summary] overall_status={decision.get('overall_status')}, hard_fail_ids={len(decision.get('hard_fail_ids') or [])}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_lagrangian_noether_observable_closure_audit",
                "phase": "8",
                "step": "8.7.21.1",
                "outputs": {
                    "lagrangian_noether_observable_closure_audit_json": _rel(out_json),
                    "lagrangian_noether_observable_closure_audit_csv": _rel(out_csv),
                    "lagrangian_noether_observable_closure_audit_png": _rel(out_png),
                },
                "metrics": {
                    "overall_status": decision.get("overall_status"),
                    "hard_fail_ids_n": len(decision.get("hard_fail_ids") or []),
                    "watch_ids_n": len(decision.get("watch_ids") or []),
                },
            }
        )
    except Exception as exc:
        print(f"[warn] worklog append skipped: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

