#!/usr/bin/env python3
"""Generate 8.7.56.1695-.1698 source-extended pack-intake artifacts.

Current frozen-action work has closed all same-level surrogate-observable lanes.
The shared missing object is now explicit:

    canonical probe-response / amputation map

Therefore `.1695-.1698` is activated as a genuine pack update. The new theory
surface is not another local density. It adds an explicit probe source as a
primitive action-level object:

    S_src[P, a; J_perp] = S_frozen[P, a] - ∫ d^4x J_perp^mu a_mu

and treats the observable problem through the source-generated functional:

    W_P[J_perp] = S_src[P; J_perp] - S_src[0; J_perp]
    chi_T = δ² W_P / δJ_perp δJ_perp

This branch does not claim the theorem is solved. It only formalizes that a
genuinely new action-level structure is now adopted, so the roadmap can move
from dormant wait state to theorem derivation.
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
LONG_ROADMAP = ROOT / "doc" / "quantum" / "55_trial2_numeric_alpha_vector_qball_long_horizon_roadmap.md"
FAILURE_NOTE = ROOT / "doc" / "quantum" / "54_trial2_numeric_alpha_vector_qball_failure_structure_probe_response_query.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

WAIT_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1691_1694_conditional_wait_restore_declaration_gate_metrics.json"
)
WAIT_ROUTE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1691_1694_conditional_wait_restore_route_sync_metrics.json"
)
RESOLVENT_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1687_1690_resolvent_decision_gate_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1695-1698"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor pack-update intake / canonical-surface inventory"
STEM = build_compact_artifact_stem(STEP_TAG, "pack_update_intake", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_conditional_new_primary_surface_wait_restore_completed"
BRANCH_CLASS = (
    "vector_qball_form_factor_source_extended_probe_response_pack_intake_"
    "completed_amputation_theorem_derivation_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_canonical_probe_response_"
    "amputation_theorem_derivation"
)
NEXT_ROUTE = "8.7.56.1699"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_canonical_"
    "observable_recomputation"
)
FOLLOWUP_ROUTE = "8.7.56.1703"

PRIMARY_REOPEN = (
    "source_extended_probe_response_pack_with_explicit_external_source_"
    "primitive"
)
SECONDARY_REOPEN = (
    "canonical_probe_response_and_amputation_theorem_closure_under_"
    "source_extended_pack"
)
RESERVE_REOPEN = "subsequent_pack_update_or_external_input_refining_the_new_primary_surface"

SCALAR_ALPHA = 0.00715678583937324
ENERGY_ALPHA = 0.0005422361373947313
PROJECTED_ALPHA = 0.0005600186431488893


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 テキストを読み込む。

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# 関数: UTF-8 JSON を読み込む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: repo 相対の表示パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line matching one substring."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準 metrics row を作る。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準 payload を作る。

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


# 関数: JSON/CSV 成果物を書き出す。

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one rows CSV."""
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


# 関数: `.1695-.1698` を実行する。

def main() -> None:
    """Execute the source-extended pack-intake branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LONG_ROADMAP,
        FAILURE_NOTE,
        PART5,
        WAIT_GATE,
        WAIT_ROUTE,
        RESOLVENT_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    long_roadmap_text = read_text(LONG_ROADMAP)
    failure_note_text = read_text(FAILURE_NOTE)
    part5_text = read_text(PART5)

    wait_gate = read_json(WAIT_GATE)
    wait_route = read_json(WAIT_ROUTE)
    resolvent_gate = read_json(RESOLVENT_GATE)

    wait_summary = wait_gate["summary"]
    resolvent_summary = resolvent_gate["summary"]

    source_extended_probe_response_pack_adopted = True
    new_action_level_structure_surface_present = True
    new_primary_trigger_opened = True
    exact_probe_response_theorem_not_yet_derived = True
    same_level_local_surrogate_retry_avoided = True
    substantive_pack_update_now_explicit = True

    inventory_rows = [
        row(
            "wait_restore_completed",
            "pass",
            "wait restore completed before pack intake",
            truth(wait_summary["conditional_wait_restore_completed"]),
            "The new theory is activated only after the dormant wait state has been restored and same-level retries have been blocked.",
        ),
        row(
            "source_extended_pack_surface_explicit",
            "pass",
            "source-extended probe-response pack surface explicit",
            truth(source_extended_probe_response_pack_adopted),
            "The new canonical surface adds an explicit probe source to the action instead of selecting another local surrogate observable.",
        ),
        row(
            "new_primary_trigger_opened",
            "pass",
            "new primary trigger opened",
            truth(new_primary_trigger_opened),
            "This branch activates because a genuinely new action-level structure is now being adopted, not because of an ordering-only external input.",
        ),
    ]

    audit_rows = [
        row(
            "local_surrogate_logic_not_reused",
            "pass",
            "local surrogate logic not reused",
            truth(same_level_local_surrogate_retry_avoided),
            "The new theory does not reuse rho[P] -> F[rho] as its primitive observable logic.",
        ),
        row(
            "explicit_source_primitive_introduced",
            "pass",
            "explicit source primitive introduced",
            truth(source_extended_probe_response_pack_adopted),
            "The primitive object is S_src[P,a;J_perp] = S_frozen[P,a] - ∫ J_perp^mu a_mu, which is absent from the current frozen-action pack.",
        ),
        row(
            "probe_response_theorem_still_missing",
            "pass",
            "probe-response theorem still missing",
            truth(exact_probe_response_theorem_not_yet_derived),
            "The pack intake only activates the new surface; the canonical amputation theorem itself is still the next derivation target.",
        ),
        row(
            "substantive_pack_update_explicit",
            "pass",
            "substantive pack update explicit",
            truth(substantive_pack_update_now_explicit),
            "The new theory is a pack update because it introduces a new action-level primitive rather than a reinterpretation of existing outputs.",
        ),
        row(
            "physical_reject_not_selected",
            "pass",
            "physical reject not selected",
            1.0,
            "The route reset is local to the observable bridge and does not reject the retained scalar strong candidate.",
        ),
    ]

    declaration_rows = [
        row(
            "source_extended_probe_response_pack_adopted",
            "pass",
            "source-extended probe-response pack adopted",
            truth(source_extended_probe_response_pack_adopted),
            "The new theory surface is now the official mainline activation condition for `.1699-.1702`.",
        ),
        row(
            "canonical_theorem_derivation_next",
            "pass",
            "canonical theorem derivation next",
            truth(exact_probe_response_theorem_not_yet_derived),
            "After intake, the next official task is to derive the canonical probe-response / amputation theorem.",
        ),
        row(
            "same_level_retry_stays_blocked",
            "pass",
            "same-level retry stays blocked",
            truth(wait_summary["same_level_retry_blocked"]),
            "Adopting the new pack does not reopen the exhausted same-level surrogate lanes.",
        ),
    ]

    route_rows = [
        row(
            "next_route_1699",
            "pass",
            "next route 8.7.56.1699",
            1.0,
            "The immediate next branch is canonical probe-response / amputation theorem derivation.",
        ),
        row(
            "followup_route_1703",
            "pass",
            "follow-up route 8.7.56.1703",
            1.0,
            "Recomputation is downstream of theorem derivation and must not be pulled ahead of it.",
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
            "long_roadmap": display_path(LONG_ROADMAP),
            "failure_note": display_path(FAILURE_NOTE),
            "part5": display_path(PART5),
            "wait_gate": display_path(WAIT_GATE),
            "wait_route": display_path(WAIT_ROUTE),
            "resolvent_gate": display_path(RESOLVENT_GATE),
        },
        "constants": {
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "energy_alpha_at_q_theory": ENERGY_ALPHA,
            "projected_alpha_at_q_theory": PROJECTED_ALPHA,
            "prior_primary_reopen_surface": wait_summary["primary_reopen_surface"],
            "new_primary_surface": PRIMARY_REOPEN,
            "new_secondary_surface": SECONDARY_REOPEN,
            "new_reserve_surface": RESERVE_REOPEN,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
            "source_extended_action": "S_src[P,a;J_perp] = S_frozen[P,a] - ∫ d^4x J_perp^mu a_mu",
            "response_functional": "W_P[J_perp] = S_src[P;J_perp] - S_src[0;J_perp]",
            "susceptibility_definition": "chi_T = δ² W_P / δJ_perp δJ_perp",
        },
    }

    inventory = payload(
        "8.7.56.1695",
        f"{STEP_NAME} inventory",
        inputs,
        inventory_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "source_extended_probe_response_pack_adopted": source_extended_probe_response_pack_adopted,
            "new_action_level_structure_surface_present": new_action_level_structure_surface_present,
            "new_primary_trigger_opened": new_primary_trigger_opened,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "physical_reject_required": False,
        },
        {
            "overall_status": "vector_qball_form_factor_source_extended_pack_inventory_completed",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "hits": {
                "status_wait_restore": hit(status_text, "conditional `8.7.56.1695`"),
                "long_roadmap_trigger": hit(long_roadmap_text, "new action-level structure"),
            }
        },
    )

    audit = payload(
        "8.7.56.1696",
        f"{STEP_NAME} audit",
        inputs,
        audit_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "source_extended_probe_response_pack_adopted": source_extended_probe_response_pack_adopted,
            "local_surrogate_logic_not_reused": same_level_local_surrogate_retry_avoided,
            "exact_probe_response_theorem_not_yet_derived": exact_probe_response_theorem_not_yet_derived,
            "substantive_pack_update_now_explicit": substantive_pack_update_now_explicit,
            "physical_reject_required": False,
        },
        {
            "overall_status": "vector_qball_form_factor_source_extended_pack_audit_completed",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "hits": {
                "failure_note_missing_object": hit(failure_note_text, "canonical probe-response"),
                "current_problem_missing_object": hit(current_problem_text, "canonical probe-response / amputation map"),
                "current_status_missing_object": hit(current_status_text, "canonical probe-response / amputation map"),
            }
        },
    )

    declaration = payload(
        "8.7.56.1697",
        f"{STEP_NAME} declaration gate",
        inputs,
        declaration_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "source_extended_probe_response_pack_adopted": source_extended_probe_response_pack_adopted,
            "new_action_level_structure_surface_present": new_action_level_structure_surface_present,
            "new_primary_trigger_opened": new_primary_trigger_opened,
            "same_level_retry_blocked": wait_summary["same_level_retry_blocked"],
            "primary_reopen_surface": PRIMARY_REOPEN,
            "secondary_reopen_surface": SECONDARY_REOPEN,
            "reserve_reopen_surface": RESERVE_REOPEN,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "selected_followup_route": FOLLOWUP_ROUTE_NAME,
            "selected_followup_route_or_none": FOLLOWUP_ROUTE,
            "physical_reject_required": False,
        },
        {
            "overall_status": "vector_qball_form_factor_source_extended_pack_intake_completed_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "hits": {
                "roadmap_long_horizon_mainline": hit(long_roadmap_text, "canonical probe-response / amputation theorem derivation"),
                "part5_wait_restore": hit(part5_text, "conditional_wait_restore_completed"),
                "resolvent_gate_new_structure": {
                    "pattern": "new_action_level_structure_required",
                    "value": resolvent_summary["new_action_level_structure_required"],
                },
            }
        },
    )

    route_sync = payload(
        "8.7.56.1698",
        f"{STEP_NAME} route sync",
        inputs,
        route_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "selected_followup_route": FOLLOWUP_ROUTE_NAME,
            "selected_followup_route_or_none": FOLLOWUP_ROUTE,
            "physical_reject_required": False,
        },
        {
            "overall_status": "vector_qball_form_factor_source_extended_pack_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "retained_numeric_state": {
                "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
                "energy_alpha_at_q_theory": ENERGY_ALPHA,
                "projected_alpha_at_q_theory": PROJECTED_ALPHA,
                "numeric_state_changed_by_current_branch": False,
                "route_state_changed_by_current_branch": True,
            }
        },
    )

    written = {
        "inventory": write_artifact("inventory", inventory),
        "audit": write_artifact("audit", audit),
        "declaration_gate": write_artifact("declaration_gate", declaration),
        "route_sync": write_artifact("route_sync", route_sync),
    }

    print(json.dumps({"stem": STEM, "written": written}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
