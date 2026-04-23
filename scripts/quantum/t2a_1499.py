#!/usr/bin/env python3
"""Generate 8.7.56.1499-.1502 future reopen ordering registry artifacts.

This branch turns the negative current-pack closeout into an explicit reopen
ordering. The closeout result is already fixed:

- scalar-side strong candidate stays retained,
- restored exact vector branch stays nontrivial but fails the blind vector gate,
- exact charge-current / Noether-current closure failed under the current pack,
- physical reject is not selected.

The honest next task is therefore to freeze the future reopen ordering in a
machine-readable form before refreshing the expert-facing advice pack.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
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
REOPEN_ADVICE = ROOT / "doc" / "quantum" / "40_trial2_numeric_alpha_vector_qball_reopen_advice_request.md"
CASE_GAMMA_ADVICE = ROOT / "doc" / "quantum" / "42_trial2_numeric_alpha_vector_qball_case_gamma_advice_request.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

GAP_CLOSEOUT_GATE = PUBLIC_OUT / "q_8_7_56_1495_1498_charge_current_gap_closeout_declaration_gate_metrics.json"
GAP_CLOSEOUT_ROUTE = PUBLIC_OUT / "q_8_7_56_1495_1498_charge_current_gap_closeout_route_sync_metrics.json"
CHARGE_CLOSURE_GATE = PUBLIC_OUT / "q_8_7_56_1491_1494_charge_current_closure_declaration_gate_metrics.json"
SOURCE_THEOREM_GATE = PUBLIC_OUT / "q_8_7_56_1487_1490_effective_source_theorem_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.1499-1502"
STEM = build_compact_artifact_stem(STEP_TAG, "future_reopen_registry", prefix="q")
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor future reopen ordering registry"

PRIOR_CLASS = "vector_qball_form_factor_exact_charge_current_noether_gap_closeout_sync_completed"
BRANCH_CLASS = "vector_qball_form_factor_future_reopen_ordering_registry_completed"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_future_reopen_advice_pack_refresh"
NEXT_ROUTE = "8.7.56.1503"

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


# 関数: UTF-8テキストを読み込む。

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# 関数: UTF-8 JSONを読み込む。

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


# 関数: 部分文字列に一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first matching line for one substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準形式の metrics row を生成する。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準形式の payload を生成する。

def payload(step: str, name: str, inputs: dict, rows: list[dict], summary: dict, decision: dict, evidence: dict) -> dict:
    """Build one standard payload."""
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
    paths["json"].write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": display_path(paths["json"]), "csv": display_path(paths["csv"])}


# 関数: 真偽値を 0 / 1 に変換する。

def truth(value: bool) -> float:
    """Convert one boolean into 0/1 float form."""
    return 1.0 if value else 0.0


# 関数: `.1499-.1502` を実行する。

def main() -> None:
    """Execute the future reopen ordering registry branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        REOPEN_ADVICE,
        CASE_GAMMA_ADVICE,
        PART1,
        PART3A,
        PART5,
        GAP_CLOSEOUT_GATE,
        GAP_CLOSEOUT_ROUTE,
        CHARGE_CLOSURE_GATE,
        SOURCE_THEOREM_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    reopen_advice_text = read_text(REOPEN_ADVICE)
    case_gamma_text = read_text(CASE_GAMMA_ADVICE)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)

    gap_closeout_gate = read_json(GAP_CLOSEOUT_GATE)
    gap_closeout_route = read_json(GAP_CLOSEOUT_ROUTE)
    charge_closure_gate = read_json(CHARGE_CLOSURE_GATE)
    source_theorem_gate = read_json(SOURCE_THEOREM_GATE)

    closeout_summary = gap_closeout_gate["summary"]
    route_sync_summary = gap_closeout_route["summary"]
    closure_summary = charge_closure_gate["summary"]
    source_summary = source_theorem_gate["summary"]

    status_registry = hit(status_text, "future reopen ordering registry")
    status_advice_pack = hit(status_text, "future reopen advice-pack refresh")
    roadmap_registry = hit(roadmap_text, "`8.7.56.1499-.1502`")
    roadmap_advice_pack = hit(roadmap_text, "`8.7.56.1503-.1506`")
    problem_registry = hit(current_problem_text, "future reopen ordering registry -> future reopen advice-pack refresh")
    current_status_registry = hit(current_status_text, "future reopen ordering")
    unified_registry = hit(unified_roadmap_text, "future reopen ordering registry")
    unified_advice_pack = hit(unified_roadmap_text, "future reopen advice-pack refresh")
    advice_case_c = hit(reopen_advice_text, "Case C honest partial")
    advice_scalar = hit(reopen_advice_text, "0.2998913524347805")
    case_gamma_hit = hit(case_gamma_text, "Case γ")
    part1_noether = hit(part1_text, "Noether保存則")
    part3a_identity = hit(part3a_text, "Q-ball Noether charge = adopted U(1) charge")
    part5_registry = hit(part5_text, "future reopen ordering registry")
    part5_reject_false = hit(part5_text, "physical_reject_required = false")

    inventory_ready = all(
        item is not None
        for item in (
            status_registry,
            status_advice_pack,
            roadmap_registry,
            roadmap_advice_pack,
            problem_registry,
            current_status_registry,
            unified_registry,
            unified_advice_pack,
            advice_case_c,
            advice_scalar,
            case_gamma_hit,
            part1_noether,
            part3a_identity,
            part5_registry,
            part5_reject_false,
        )
    )

    closeout_sync_available = bool(
        closeout_summary.get("route_sync_ready", False)
        and closeout_summary.get("future_reopen_ordering_registry_required", False)
        and route_sync_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
    )
    primary_trigger_honest = bool(
        not closure_summary.get("exact_charge_current_noether_closure_available", True)
        and closure_summary.get("primary_residual_lane") == "vector_qball_form_factor_exact_charge_current_noether_closure_missing"
    )
    secondary_trigger_honest = bool(
        not source_summary.get("exact_source_theorem_derived", True)
        and source_summary.get("secondary_residual_lane") == "vector_qball_form_factor_effective_source_formula_absent"
    )
    reserve_trigger_honest = bool(
        closeout_summary.get("observable_dictionary_deferred", False)
        and not closure_summary.get("observable_dictionary_gate_admissible_now", True)
    )
    retained_scalar_strong_candidate_retained = bool(
        closeout_summary.get("retained_scalar_strong_candidate_retained", False)
        and advice_scalar is not None
    )
    physical_reject_not_selected = bool(
        not closeout_summary.get("physical_reject_required", True)
        and part5_reject_false is not None
    )
    reopen_ordering_honest = all(
        [
            inventory_ready,
            closeout_sync_available,
            primary_trigger_honest,
            secondary_trigger_honest,
            reserve_trigger_honest,
            retained_scalar_strong_candidate_retained,
            physical_reject_not_selected,
        ]
    )
    reopen_ordering_registry_ready = bool(reopen_ordering_honest)
    expert_facing_followup_required = bool(reopen_ordering_registry_ready)

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "future reopen ordering registry inventory ready",
            truth(inventory_ready),
            "The registry is only honest after closeout sync outputs, current notes, Part I / Part III-A / Part V surfaces, and advice notes coexist in one pack.",
        ),
        row(
            "closeout_sync_available",
            "pass" if closeout_sync_available else "reject",
            "noether-current gap closeout sync available",
            truth(closeout_sync_available),
            "The reopen ordering should be frozen only after the prior closeout sync has already stabilized the negative current-pack result.",
        ),
        row(
            "primary_trigger_honest",
            "pass" if primary_trigger_honest else "reject",
            "primary future reopen trigger honest",
            truth(primary_trigger_honest),
            "The first reopen lane is exact charge-current / Noether-current closure because that is now the most localized missing theorem surface.",
        ),
        row(
            "secondary_trigger_honest",
            "pass" if secondary_trigger_honest else "reject",
            "secondary future reopen trigger honest",
            truth(secondary_trigger_honest),
            "Effective source theorem stays secondary because it now depends on a future exact charge-current / Noether-current closure reopen.",
        ),
        row(
            "reserve_trigger_honest",
            "pass" if reserve_trigger_honest else "reject",
            "reserve future reopen trigger honest",
            truth(reserve_trigger_honest),
            "Observable dictionary remains reserve because the exact charge-current bridge is still absent and explicitly deferred downstream.",
        ),
        row(
            "retained_scalar_strong_candidate_retained",
            "pass" if retained_scalar_strong_candidate_retained else "reject",
            "retained scalar strong candidate kept visible in reopen registry",
            truth(retained_scalar_strong_candidate_retained),
            "The registry remains route-local only if the strong scalar-side candidate stays visible during the reopen freeze.",
        ),
        row(
            "physical_reject_not_selected",
            "pass" if physical_reject_not_selected else "reject",
            "physical reject not selected after reopen ordering registry",
            truth(physical_reject_not_selected),
            "Reopen ordering is meaningful only if the route still stays below a full physical reject.",
        ),
        row(
            "reopen_ordering_honest",
            "pass" if reopen_ordering_honest else "reject",
            "future reopen ordering honest",
            truth(reopen_ordering_honest),
            "The ordering is honest only if the current-pack closeout, theorem gaps, retained scalar candidate, and non-reject state stay explicit together.",
        ),
        row(
            "expert_facing_followup_required",
            "pass" if expert_facing_followup_required else "reject",
            "expert-facing followup required",
            truth(expert_facing_followup_required),
            "Once the internal ordering is frozen, the next action should turn expert-facing rather than extending the internal wording loop.",
        ),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem_note": display_path(CURRENT_PROBLEM),
            "current_status_note": display_path(CURRENT_STATUS),
            "unified_roadmap_note": display_path(UNIFIED_ROADMAP),
            "reopen_advice_note": display_path(REOPEN_ADVICE),
            "case_gamma_advice_note": display_path(CASE_GAMMA_ADVICE),
            "part1": display_path(PART1),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
        },
        "prior_metrics": {
            "gap_closeout_gate": display_path(GAP_CLOSEOUT_GATE),
            "gap_closeout_route": display_path(GAP_CLOSEOUT_ROUTE),
            "charge_closure_gate": display_path(CHARGE_CLOSURE_GATE),
            "source_theorem_gate": display_path(SOURCE_THEOREM_GATE),
        },
        "constants": {
            "primary_future_reopen_trigger": PRIMARY_TRIGGER,
            "secondary_future_reopen_trigger": SECONDARY_TRIGGER,
            "reserve_future_reopen_trigger": RESERVE_TRIGGER,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }
    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "reopen_ordering_registry_ready": reopen_ordering_registry_ready,
        "reopen_ordering_honest": reopen_ordering_honest,
        "primary_future_reopen_trigger": PRIMARY_TRIGGER,
        "secondary_future_reopen_trigger": SECONDARY_TRIGGER,
        "reserve_future_reopen_trigger": RESERVE_TRIGGER,
        "retained_scalar_strong_candidate_retained": retained_scalar_strong_candidate_retained,
        "physical_reject_required": False,
        "expert_facing_followup_required": expert_facing_followup_required,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
    }
    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }
    evidence = {
        "part_hits": {
            "part1_noether": part1_noether,
            "part3a_identity": part3a_identity,
            "part5_registry": part5_registry,
            "part5_reject_false": part5_reject_false,
        },
        "doc_hits": {
            "status_registry": status_registry,
            "status_advice_pack": status_advice_pack,
            "roadmap_registry": roadmap_registry,
            "roadmap_advice_pack": roadmap_advice_pack,
            "problem_registry": problem_registry,
            "current_status_registry": current_status_registry,
            "unified_registry": unified_registry,
            "unified_advice_pack": unified_advice_pack,
            "advice_case_c": advice_case_c,
            "advice_scalar": advice_scalar,
            "case_gamma": case_gamma_hit,
        },
        "carry_over": {
            "gap_closeout_summary": closeout_summary,
            "gap_closeout_route_sync_summary": route_sync_summary,
            "charge_closure_summary": closure_summary,
            "source_theorem_summary": source_summary,
        },
        "retained_numeric_state": {
            "retained_scalar_F_exact_at_q_theory": 0.2998913524347805,
            "retained_scalar_alpha_exact_at_q_theory": 0.00715678583937324,
            "phase1_equivalent_F_at_q_theory": -0.083735013520183,
            "phase1_equivalent_alpha_at_q_theory": 0.0005579616187042394,
            "phase1_equivalent_max_abs_ratio": 0.11918404084753811,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    for kind in ("inventory", "audit", "declaration_gate", "route_sync"):
        write_artifact(kind, payload(STEP_TAG, STEP_NAME, inputs, rows, summary, decision, evidence))

    print(f"[ok] wrote compact artifacts for {STEP_TAG}: {STEM}")


if __name__ == "__main__":
    main()
