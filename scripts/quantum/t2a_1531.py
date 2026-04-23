#!/usr/bin/env python3
"""Generate 8.7.56.1531-.1534 computation-mainline-reactivation artifacts.

The previous branch `.1527-.1530` correctly froze the no-new-input text-search
lane, but it also exposed that the real blocker is no longer external advice
inventory. The strongest retained scalar candidate is already fixed, while the
remaining scientific blocker is the missing exact charge-current / Noether-
current closure for the restored exact vector branch.

This branch therefore does not search for more text. It converts the retry-gate
judgment into an explicit computation-first route reset:

- external-input wait becomes a side lane rather than the mainline stopper,
- exact charge-current / Noether-current closure returns to the primary lane,
- effective source theorem stays downstream of that closure,
- observable dictionary remains downstream of both.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

PRIOR_DORMANT_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1527_1530_future_external_input_dormant_checkpoint_declaration_gate_metrics.json"
)
PRIOR_DORMANT_ROUTE = (
    PUBLIC_OUT
    / "q_8_7_56_1527_1530_future_external_input_dormant_checkpoint_route_sync_metrics.json"
)
CHARGE_CLOSURE_GATE = (
    PUBLIC_OUT / "q_8_7_56_1491_1494_charge_current_closure_declaration_gate_metrics.json"
)
SOURCE_THEOREM_GATE = (
    PUBLIC_OUT / "q_8_7_56_1487_1490_effective_source_theorem_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1531-1534"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor computation mainline reactivation"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "computation_mainline_reactivation",
    prefix="q",
)

PRIOR_CLASS = "vector_qball_form_factor_future_external_input_dormant_checkpoint_completed"
BRANCH_CLASS = (
    "vector_qball_form_factor_computation_mainline_reactivated_exact_charge_current_closure_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exact_charge_current_noether_closure_derivation"
)
NEXT_ROUTE = "8.7.56.1535"
SIDE_LANE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_future_external_input_reactivation"
)
PRIMARY_TRIGGER = "exact_charge_current_noether_closure_reopen"
SECONDARY_TRIGGER = "effective_source_theorem_reopen"
RESERVE_TRIGGER = "observable_dictionary_exact_charge_current_bridge"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Fail when one required path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON を読み込む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: 表示用の相対パス文字列へ変換する。

def display_path(path: Path) -> str:
    """Convert one absolute path into repo-relative display form when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 標準 row を生成する。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準 payload を生成する。

def payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Build one standard metrics payload."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# 関数: compact stem で JSON / CSV を出力する。

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and its rows CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "status", "metric", "value", "note"],
        )
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": display_path(paths["json"]), "csv": display_path(paths["csv"])}


# 関数: 真偽値を 0/1 に変換する。

def truth(value: bool) -> float:
    """Convert one boolean into 0/1 float form."""
    return 1.0 if value else 0.0


# 関数: `.1531-.1534` を実行する。

def main() -> None:
    """Execute the computation-mainline-reactivation branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PART5,
        PRIOR_DORMANT_GATE,
        PRIOR_DORMANT_ROUTE,
        CHARGE_CLOSURE_GATE,
        SOURCE_THEOREM_GATE,
    ):
        require(path)

    prior_dormant = read_json(PRIOR_DORMANT_GATE)["summary"]
    prior_route = read_json(PRIOR_DORMANT_ROUTE)["summary"]
    closure_summary = read_json(CHARGE_CLOSURE_GATE)["summary"]
    source_summary = read_json(SOURCE_THEOREM_GATE)["summary"]

    prior_dormant_checkpoint_completed = bool(
        prior_dormant.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and prior_dormant.get("dormant_checkpoint_completed", False)
        and prior_dormant.get("retry_gate_triggered", False)
    )
    exact_charge_current_noether_closure_missing = not closure_summary.get(
        "exact_charge_current_noether_closure_available",
        True,
    )
    exact_source_theorem_missing = not source_summary.get(
        "exact_source_theorem_derived",
        True,
    )
    observable_dictionary_still_blocked = not closure_summary.get(
        "observable_dictionary_gate_admissible_now",
        True,
    )
    scalar_strong_candidate_retained = bool(
        prior_route.get("retained_scalar_strong_candidate_retained", True)
    )
    blind_vector_no_go_retained = bool(
        abs(prior_route.get("vector_F_at_q_theory", -0.083735013520183) + 0.083735013520183)
        < 1e-15
    )
    retry_gate_active = bool(prior_dormant.get("retry_gate_triggered", False))
    text_search_lane_exhausted = bool(
        prior_dormant.get("internal_loop_terminated_until_new_input", False)
    )
    computation_mainline_reactivated = all(
        [
            prior_dormant_checkpoint_completed,
            exact_charge_current_noether_closure_missing,
            exact_source_theorem_missing,
            observable_dictionary_still_blocked,
            scalar_strong_candidate_retained,
            retry_gate_active,
            text_search_lane_exhausted,
        ]
    )
    external_input_wait_demoted_to_side_lane = computation_mainline_reactivated
    next_route_ready = computation_mainline_reactivated

    rows = [
        row(
            "prior_dormant_checkpoint_completed",
            "pass" if prior_dormant_checkpoint_completed else "reject",
            "prior dormant checkpoint completed",
            truth(prior_dormant_checkpoint_completed),
            "The route reset should start only after the no-new-input text-search lane was closed honestly.",
        ),
        row(
            "exact_charge_current_noether_closure_missing",
            "pass" if exact_charge_current_noether_closure_missing else "reject",
            "exact charge-current / Noether-current closure still missing",
            truth(exact_charge_current_noether_closure_missing),
            "This is now the primary scientific blocker for the restored exact vector branch.",
        ),
        row(
            "exact_source_theorem_missing",
            "pass" if exact_source_theorem_missing else "reject",
            "exact effective source theorem still missing",
            truth(exact_source_theorem_missing),
            "Source-theorem work remains downstream of the exact current closure rather than a parallel mainline.",
        ),
        row(
            "observable_dictionary_still_blocked",
            "pass" if observable_dictionary_still_blocked else "reject",
            "observable dictionary still blocked",
            truth(observable_dictionary_still_blocked),
            "Observable mapping remains downstream of both current closure and source theorem.",
        ),
        row(
            "scalar_strong_candidate_retained",
            "pass" if scalar_strong_candidate_retained else "reject",
            "retained scalar strong candidate preserved",
            truth(scalar_strong_candidate_retained),
            "The route reset is computation-first, not a physical reject of the scalar-side candidate.",
        ),
        row(
            "blind_vector_no_go_retained",
            "pass" if blind_vector_no_go_retained else "reject",
            "blind vector fixed-q no-go retained",
            truth(blind_vector_no_go_retained),
            "The vector-side failure stays visible while the closure derivation is reprioritized.",
        ),
        row(
            "retry_gate_active",
            "pass" if retry_gate_active else "reject",
            "retry gate active",
            truth(retry_gate_active),
            "The same no-new-input pattern already exhausted the text-search branch.",
        ),
        row(
            "text_search_lane_exhausted",
            "pass" if text_search_lane_exhausted else "reject",
            "text-search lane exhausted",
            truth(text_search_lane_exhausted),
            "External-input wait remains valid as a side lane, but no longer as the main scientific route.",
        ),
        row(
            "computation_mainline_reactivated",
            "pass" if computation_mainline_reactivated else "reject",
            "computation mainline reactivated",
            truth(computation_mainline_reactivated),
            "The next official branch returns to exact current derivation rather than waiting for more notes.",
        ),
        row(
            "external_input_wait_demoted_to_side_lane",
            "pass" if external_input_wait_demoted_to_side_lane else "reject",
            "external-input wait demoted to side lane",
            truth(external_input_wait_demoted_to_side_lane),
            "Future expert input remains useful, but it is no longer the mainline blocker after the retry-gate judgment.",
        ),
        row(
            "next_route_ready",
            "pass" if next_route_ready else "reject",
            "next exact current-closure route ready",
            truth(next_route_ready),
            "The immediate next branch is exact charge-current / Noether-current closure derivation.",
        ),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem": display_path(CURRENT_PROBLEM),
            "current_status": display_path(CURRENT_STATUS),
            "unified_roadmap": display_path(UNIFIED_ROADMAP),
            "part5": display_path(PART5),
        },
        "prior_metrics": {
            "prior_dormant_gate": display_path(PRIOR_DORMANT_GATE),
            "prior_dormant_route": display_path(PRIOR_DORMANT_ROUTE),
            "charge_closure_gate": display_path(CHARGE_CLOSURE_GATE),
            "source_theorem_gate": display_path(SOURCE_THEOREM_GATE),
        },
        "constants": {
            "primary_trigger": PRIMARY_TRIGGER,
            "secondary_trigger": SECONDARY_TRIGGER,
            "reserve_trigger": RESERVE_TRIGGER,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "side_lane_name": SIDE_LANE_NAME,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "prior_dormant_checkpoint_completed": prior_dormant_checkpoint_completed,
        "exact_charge_current_noether_closure_missing": exact_charge_current_noether_closure_missing,
        "exact_source_theorem_missing": exact_source_theorem_missing,
        "observable_dictionary_still_blocked": observable_dictionary_still_blocked,
        "scalar_strong_candidate_retained": scalar_strong_candidate_retained,
        "blind_vector_no_go_retained": blind_vector_no_go_retained,
        "retry_gate_active": retry_gate_active,
        "text_search_lane_exhausted": text_search_lane_exhausted,
        "computation_mainline_reactivated": computation_mainline_reactivated,
        "external_input_wait_demoted_to_side_lane": external_input_wait_demoted_to_side_lane,
        "physical_reject_required": False,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "side_lane_future_route_or_none": SIDE_LANE_NAME,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": computation_mainline_reactivated,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "prior_dormant_summary": prior_dormant,
        "prior_route_summary": prior_route,
        "charge_closure_summary": closure_summary,
        "source_theorem_summary": source_summary,
        "retained_numeric_state": {
            "scalar_F_exact_at_q_theory": 0.2998913524347805,
            "scalar_alpha_exact_at_q_theory": 0.00715678583937324,
            "vector_F_at_q_theory": -0.083735013520183,
            "vector_alpha_at_q_theory": 0.0005579616187042394,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
        "route_reset_text": (
            "The retry gate proved that the text-search lane is exhausted. The mainline therefore returns to "
            "exact charge-current / Noether-current closure derivation, while future external input remains a side lane."
        ),
    }

    for kind in ("inventory", "audit", "declaration_gate", "route_sync"):
        write_artifact(kind, payload(STEP_TAG, STEP_NAME, inputs, rows, summary, decision, evidence))

    print(f"[ok] wrote compact artifacts for {STEP_TAG}: {STEM}")


if __name__ == "__main__":
    main()
