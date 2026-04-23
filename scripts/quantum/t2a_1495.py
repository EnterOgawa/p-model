#!/usr/bin/env python3
"""Generate 8.7.56.1495-.1498 exact charge-current / Noether-current gap closeout sync artifacts.

This branch does not try to reopen the theorem. The preceding closure audit
already fixed the honest current-pack limit:

- generic continuity exists,
- adopted-U(1) / Q-ball identity exists,
- proxy signed density exists,
- but exact charge-current / Noether-current closure does not.

The present task is therefore administrative but still important:

1. freeze the negative closure result as an honest closeout,
2. keep the retained scalar strong candidate visible,
3. keep `physical_reject_required = false`,
4. hand off the route to a future reopen-ordering registry.
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
NEXT_STEPS_NOTE = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")

CLOSURE_GATE = PUBLIC_OUT / "q_8_7_56_1491_1494_charge_current_closure_declaration_gate_metrics.json"
CLOSURE_EVAL = PUBLIC_OUT / "q_8_7_56_1491_1494_charge_current_closure_numeric_evaluation_metrics.json"
SOURCE_GATE = PUBLIC_OUT / "q_8_7_56_1487_1490_effective_source_theorem_declaration_gate_metrics.json"
SOURCE_EVAL = PUBLIC_OUT / "q_8_7_56_1487_1490_effective_source_theorem_numeric_evaluation_metrics.json"
ANCHOR_EVAL = PUBLIC_OUT / "q_8_7_56_1483_1486_ell0_anchor_continuation_numeric_evaluation_metrics.json"
QBALL_CHARGE_MAPPING = PUBLIC_OUT / "mass_origin_qball_charge_mapping_statement_freeze_metrics.json"
QBALL_CHARGE_NORMALIZATION = PUBLIC_OUT / "mass_origin_qball_charge_operator_normalization_audit_metrics.json"

STEP_TAG = "8.7.56.1495-1498"
STEM = build_compact_artifact_stem(STEP_TAG, "charge_current_gap_closeout", prefix="q")
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor exact charge-current / Noether-current gap closeout sync"

PRIOR_CLASS = "vector_qball_form_factor_exact_charge_current_noether_closure_failed_proxy_signed_density_only_retained"
BRANCH_CLASS = "vector_qball_form_factor_exact_charge_current_noether_gap_closeout_sync_completed"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_future_reopen_ordering_registry"
NEXT_ROUTE = "8.7.56.1499"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_future_reopen_advice_pack_refresh"
FOLLOWUP_ROUTE = "8.7.56.1503"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Fail when one required input path is missing."""
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


# 関数: `.1495-.1498` を実行する。

def main() -> None:
    """Execute the exact charge-current / Noether-current gap closeout sync."""
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
        NEXT_STEPS_NOTE,
        CLOSURE_GATE,
        CLOSURE_EVAL,
        SOURCE_GATE,
        SOURCE_EVAL,
        ANCHOR_EVAL,
        QBALL_CHARGE_MAPPING,
        QBALL_CHARGE_NORMALIZATION,
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
    next_steps_text = read_text(NEXT_STEPS_NOTE)

    closure_gate = read_json(CLOSURE_GATE)
    closure_eval = read_json(CLOSURE_EVAL)
    source_gate = read_json(SOURCE_GATE)
    source_eval = read_json(SOURCE_EVAL)
    anchor_eval = read_json(ANCHOR_EVAL)
    qball_charge_mapping = read_json(QBALL_CHARGE_MAPPING)
    qball_charge_normalization = read_json(QBALL_CHARGE_NORMALIZATION)

    closure_summary = closure_gate["summary"]
    closure_numeric = closure_eval["evidence"]["retained_numeric_state"]
    source_summary = source_gate["summary"]
    source_numeric = source_eval["evidence"]["retained_numeric_state"]
    anchor_numeric = anchor_eval["summary"]

    status_gap = hit(status_text, "exact charge-current / Noether-current gap closeout sync")
    status_proxy = hit(status_text, "proxy signed density")
    status_future_reopen = hit(status_text, "future reopen ordering")
    roadmap_gap = hit(roadmap_text, "`8.7.56.1495-.1498`")
    roadmap_registry = hit(roadmap_text, "`8.7.56.1499-.1502`")
    problem_gap = hit(current_problem_text, "exact charge-current / Noether-current gap closeout sync")
    problem_future_reopen = hit(current_problem_text, "future reopen ordering")
    current_status_gap = hit(current_status_text, "exact charge-current / Noether-current gap closeout sync")
    current_status_dictionary = hit(current_status_text, "observable dictionary")
    unified_gap = hit(unified_roadmap_text, "exact charge-current / Noether-current gap closeout sync")
    unified_registry = hit(unified_roadmap_text, "future reopen ordering")
    reopen_ordering_hit = hit(reopen_advice_text, "exact_action_level_ell0_operator_reopen")
    case_gamma_hit = hit(case_gamma_text, "Case C honest partial")
    part1_noether = hit(part1_text, "Noether保存則")
    part3a_identity = hit(part3a_text, "Q-ball Noether charge = adopted U(1) charge")
    part5_gap = hit(part5_text, "exact charge-current / Noether-current gap closeout sync")
    step_c_hit = hit(next_steps_text, "### Step C.")

    inventory_ready = all(
        item is not None
        for item in (
            status_gap,
            status_proxy,
            status_future_reopen,
            roadmap_gap,
            roadmap_registry,
            problem_gap,
            problem_future_reopen,
            current_status_gap,
            current_status_dictionary,
            unified_gap,
            unified_registry,
            reopen_ordering_hit,
            case_gamma_hit,
            part1_noether,
            part3a_identity,
            part5_gap,
            step_c_hit,
        )
    )

    closure_fail_retained = bool(
        not closure_summary.get("exact_charge_current_noether_closure_available", False)
        and closure_summary.get("proxy_signed_density_only", False)
        and not closure_summary.get("observable_dictionary_gate_admissible_now", True)
    )
    retained_scalar_strong_candidate_retained = bool(
        hit(current_problem_text, "0.2998913524347805")
        and hit(current_status_text, "0.2998913524347805")
    )
    proxy_signed_density_only_retained = bool(
        closure_summary.get("proxy_signed_density_only", False)
        and status_proxy
        and hit(current_problem_text, "proxy_signed_density_only = true")
        and hit(current_status_text, "proxy_signed_density_only = true")
    )
    physical_reject_not_selected = bool(
        not closure_summary.get("physical_reject_required", True)
        and not source_summary.get("physical_reject_required", True)
        and hit(status_text, "physical_reject_required = false")
    )
    observable_dictionary_deferred = bool(
        not closure_summary.get("observable_dictionary_gate_admissible_now", True)
        and current_status_dictionary
        and hit(status_text, "observable dictionary は immediate next ではなく")
    )
    closeout_wording_honest = all(
        [
            closure_fail_retained,
            retained_scalar_strong_candidate_retained,
            proxy_signed_density_only_retained,
            physical_reject_not_selected,
            observable_dictionary_deferred,
        ]
    )
    route_sync_ready = bool(inventory_ready and closeout_wording_honest)
    future_reopen_ordering_registry_required = bool(route_sync_ready)
    post_registry_expert_advice_pack_planned = bool(route_sync_ready)

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "noether-current gap closeout inventory ready",
            truth(inventory_ready),
            "The closeout sync is only honest after closure-audit outputs, source-theorem outputs, charge mapping artifacts, and current wording all coexist in one pack.",
        ),
        row(
            "closure_fail_retained",
            "pass" if closure_fail_retained else "reject",
            "closure fail retained honestly",
            truth(closure_fail_retained),
            "The current-pack limit must stay phrased as an exact closure failure, not as a generic continuity failure.",
        ),
        row(
            "retained_scalar_strong_candidate_retained",
            "pass" if retained_scalar_strong_candidate_retained else "reject",
            "retained scalar strong candidate kept visible",
            truth(retained_scalar_strong_candidate_retained),
            "Closeout wording must keep the scalar exact-profile candidate visible so the branch does not collapse into a false full no-go.",
        ),
        row(
            "proxy_signed_density_only_retained",
            "pass" if proxy_signed_density_only_retained else "reject",
            "proxy signed density retained as proxy-only",
            truth(proxy_signed_density_only_retained),
            "The signed-density readout remains only a proxy hint and must not be promoted during closeout sync.",
        ),
        row(
            "physical_reject_not_selected",
            "pass" if physical_reject_not_selected else "reject",
            "physical reject not selected",
            truth(physical_reject_not_selected),
            "Gap closeout remains route-local and therefore must keep physical_reject_required = false.",
        ),
        row(
            "observable_dictionary_deferred",
            "pass" if observable_dictionary_deferred else "reject",
            "observable dictionary deferred downstream",
            truth(observable_dictionary_deferred),
            "Observable-dictionary work stays downstream until a future exact charge-current / Noether-current closure reopen succeeds.",
        ),
        row(
            "closeout_wording_honest",
            "pass" if closeout_wording_honest else "reject",
            "noether-current gap closeout wording honest",
            truth(closeout_wording_honest),
            "The wording is honest only if it simultaneously retains closure fail, scalar strong candidate, proxy-only status, and physical-reject false.",
        ),
        row(
            "route_sync_ready",
            "pass" if route_sync_ready else "reject",
            "noether-current gap closeout route sync ready",
            truth(route_sync_ready),
            "Route sync becomes ready only after the inventory and wording audit pass together.",
        ),
        row(
            "future_reopen_ordering_registry_required",
            "pass" if future_reopen_ordering_registry_required else "reject",
            "future reopen ordering registry required",
            truth(future_reopen_ordering_registry_required),
            "Once the closeout wording is fixed, the honest next internal task is to freeze the future reopen ordering.",
        ),
        row(
            "post_registry_expert_advice_pack_planned",
            "pass" if post_registry_expert_advice_pack_planned else "reject",
            "post-registry expert advice pack planned",
            truth(post_registry_expert_advice_pack_planned),
            "To avoid another wording-only loop, the roadmap should turn expert-facing immediately after the reopen registry is frozen.",
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
            "next_steps_note": display_path(NEXT_STEPS_NOTE),
        },
        "prior_metrics": {
            "closure_gate": display_path(CLOSURE_GATE),
            "closure_eval": display_path(CLOSURE_EVAL),
            "source_gate": display_path(SOURCE_GATE),
            "source_eval": display_path(SOURCE_EVAL),
            "anchor_eval": display_path(ANCHOR_EVAL),
            "qball_charge_mapping": display_path(QBALL_CHARGE_MAPPING),
            "qball_charge_normalization": display_path(QBALL_CHARGE_NORMALIZATION),
        },
        "constants": {
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "post_registry_route_name": FOLLOWUP_ROUTE_NAME,
            "post_registry_route": FOLLOWUP_ROUTE,
        },
    }
    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "closure_fail_retained": closure_fail_retained,
        "retained_scalar_strong_candidate_retained": retained_scalar_strong_candidate_retained,
        "proxy_signed_density_only_retained": proxy_signed_density_only_retained,
        "physical_reject_required": False,
        "observable_dictionary_deferred": observable_dictionary_deferred,
        "closeout_wording_honest": closeout_wording_honest,
        "route_sync_ready": route_sync_ready,
        "future_reopen_ordering_registry_required": future_reopen_ordering_registry_required,
        "post_registry_expert_advice_pack_planned": post_registry_expert_advice_pack_planned,
        "primary_residual_lane": closure_summary.get("primary_residual_lane"),
        "secondary_residual_lane": closure_summary.get("secondary_residual_lane"),
        "reserve_residual_lane": closure_summary.get("reserve_residual_lane"),
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "post_registry_followup_route_or_none": FOLLOWUP_ROUTE,
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
            "part5_gap": part5_gap,
            "step_c": step_c_hit,
        },
        "doc_hits": {
            "status_gap": status_gap,
            "status_proxy": status_proxy,
            "status_future_reopen": status_future_reopen,
            "problem_gap": problem_gap,
            "problem_future_reopen": problem_future_reopen,
            "current_status_gap": current_status_gap,
            "current_status_dictionary": current_status_dictionary,
            "unified_gap": unified_gap,
            "unified_registry": unified_registry,
            "reopen_ordering": reopen_ordering_hit,
            "case_gamma": case_gamma_hit,
        },
        "carry_over": {
            "closure_summary": closure_summary,
            "source_summary": source_summary,
            "charge_mapping_summary": qball_charge_mapping["summary"],
            "charge_normalization_summary": qball_charge_normalization["summary"],
        },
        "retained_numeric_state": {
            "phase1_equivalent_max_abs_ratio": float(source_numeric["phase1_equivalent_max_abs_ratio"]),
            "phase1_equivalent_F_at_q_theory": float(source_numeric["phase1_equivalent_F_at_q_theory"]),
            "phase1_equivalent_alpha_at_q_theory": float(source_numeric["phase1_equivalent_alpha_at_q_theory"]),
            "retained_scalar_F_exact_at_q_theory": 0.2998913524347805,
            "retained_scalar_alpha_exact_at_q_theory": 0.00715678583937324,
            "anchor_preserving_continuation_restored": bool(anchor_numeric.get("anchor_preserving_continuation_restored", False)),
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    for kind in ("inventory", "audit", "declaration_gate", "route_sync"):
        write_artifact(kind, payload(STEP_TAG, STEP_NAME, inputs, rows, summary, decision, evidence))

    print(f"[ok] wrote compact artifacts for {STEP_TAG}: {STEM}")


if __name__ == "__main__":
    main()
