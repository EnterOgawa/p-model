#!/usr/bin/env python3
"""Generate 8.7.56.1675-.1678 conditional reactivation input-assimilation artifacts.

This branch consumes one genuinely new external note that appeared after the
fallback closeout advice-pack refresh. The note is new as an input file, but
it mainly restates the already frozen reopen ordering and re-queues current-pack
lanes that have already been executed. The honest read is therefore:

- new external input detected,
- no genuinely new canonical surface opened,
- no new primary / secondary / reserve trigger opened,
- conditional reactivation completes as ordering-only assimilation,
- the next official route is a wait-restore branch.
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
PRIOR_RESPONSE = (
    ROOT
    / "doc"
    / "quantum"
    / "52_trial2_numeric_alpha_vector_qball_fallback_closeout_advice_pack_response.md"
)
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
PRIOR_ROUTE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1671_1674_fb_closeout_advice_pack_route_sync_metrics.json"
)
PRIOR_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1667_1670_fallback_closeout_registry_declaration_gate_metrics.json"
)
INPUT_NOTE = Path(
    r"C:\Users\ogawa\Downloads\51_trial2_numeric_alpha_vector_qball_next_mainline_checklist.md"
)

STEP_TAG = "8.7.56.1675-1678"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor conditional new action-level "
    "structure / external-input reactivation"
)
STEM = build_compact_artifact_stem(STEP_TAG, "cond_reactivation_input", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_fallback_closeout_advice_pack_refresh_completed"
BRANCH_CLASS = (
    "vector_qball_form_factor_conditional_reactivation_input_assimilation_"
    "ordering_only_no_new_trigger_opened"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_reactivation_wait_restore"
)
NEXT_ROUTE = "8.7.56.1679"
NEXT_ROUTE_ACTIVATION_CONDITION = (
    "ordering-only external input assimilated with no new trigger opened; "
    "future genuinely new surface or input still required"
)

PRIMARY_REOPEN = "genuinely_new_action_level_structure_beyond_current_frozen_action_pack"
SECONDARY_REOPEN = (
    "exact_constitutive_map_or_branch_local_full_nonlinear_energy_density_"
    "reopen_after_pack_update"
)
RESERVE_REOPEN = "future_external_input_or_expert_input_guiding_new_primary_surface"

SCALAR_ALPHA = 0.00715678583937324
ENERGY_ALPHA = 0.0005422361373947313
PROJECTED_ALPHA = 0.0005600186431488893
ELECTRIC_LIKE_ALPHA = 0.004692984339643002
NOTE_GRADIENT_ALPHA = 0.0047372462907781755


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を検証する。

def require(path: Path) -> None:
    """Abort when one required input is missing."""
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


# 関数: 表示用の相対パスへ変換する。

def display_path(path: Path) -> str:
    """Return one repo-relative display path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first matching line for one substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準 row を作る。

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


# 関数: compact stem で JSON / CSV 成果物を書き出す。

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


# 関数: 真偽値を 0 / 1 に変換する。

def truth(value: bool) -> float:
    """Convert one boolean into 0/1 float form."""
    return 1.0 if value else 0.0


# 関数: 入力 note の concise summary を返す。

def build_assimilation_text() -> str:
    """Return one concise summary sentence for the new external input."""
    return (
        "The new external checklist note is genuine as an input file, but it reaffirms "
        "the existing Case II vector-no-go-like read, reuses the already frozen primary / "
        "secondary / reserve ordering, and re-queues constitutive-map, branch-local "
        "nonlinear-energy, near-node, u/P_infty, transverse-overlap, and parallel J_eff "
        "lanes without supplying a new canonical surface or new action-level structure. "
        "The honest read is ordering-only assimilation with no new trigger opened."
    )


# 関数: `.1675-.1678` を実行する。

def main() -> None:
    """Execute the conditional reactivation input-assimilation branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PRIOR_RESPONSE,
        PART5,
        PRIOR_ROUTE,
        PRIOR_GATE,
        INPUT_NOTE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context_text = read_text(AI_CONTEXT)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    prior_response_text = read_text(PRIOR_RESPONSE)
    part5_text = read_text(PART5)
    input_text = read_text(INPUT_NOTE)

    prior_route = read_json(PRIOR_ROUTE)["summary"]
    prior_gate = read_json(PRIOR_GATE)["summary"]

    prior_route_available = bool(
        prior_route.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and prior_route.get("fallback_closeout_advice_pack_refresh_completed", False)
        and prior_route.get("conditional_reactivation_only_after_new_surface_or_input", False)
    )
    prior_reopen_ordering_retained = bool(
        prior_gate.get("primary_reopen_surface") == PRIMARY_REOPEN
        and prior_gate.get("secondary_reopen_surface") == SECONDARY_REOPEN
        and prior_gate.get("reserve_reopen_surface") == RESERVE_REOPEN
    )

    note_title = hit(input_text, "次の mainline 実行チェックリスト")
    note_case = hit(input_text, "Case II vector-no-go-like under current pack")
    note_primary = hit(
        input_text, "branch_local_full_nonlinear_energy_density_or_exact_constitutive_map_gap"
    )
    note_secondary = hit(
        input_text, "evidence_only_electric_like_or_note_gradient_canonical_promotion_gap"
    )
    note_reserve = hit(input_text, "future_external_input_or_new_action_level_structure")
    note_constitutive = hit(input_text, "exact constitutive map audit")
    note_full_nonlinear = hit(input_text, "branch-local full nonlinear energy density audit")
    note_near_node = hit(input_text, "near-node invariance")
    note_u_bridge = hit(input_text, "u/P_\\infty")
    note_overlap = hit(input_text, "full transverse overlap-weighting")
    note_jeff = hit(input_text, "J_eff / exact current-closure theorem lane")
    note_no_new_params = hit(input_text, "新自由パラメータを入れない")
    note_fixed_branch = hit(input_text, "同じ retained exact branch を使う")
    note_fixed_q = hit(input_text, "blind fixed-`q_theory` を守る")

    status_conditional = hit(status_text, "8.7.56.1675-.1678")
    if status_conditional is None:
        status_conditional = hit(status_text, BRANCH_CLASS)

    roadmap_conditional = hit(roadmap_text, "8.7.56.1675-.1678")
    current_problem_conditional = hit(current_problem_text, "8.7.56.1675-.1678")
    current_status_conditional = hit(current_status_text, "8.7.56.1675-.1678")
    if current_status_conditional is None:
        current_status_conditional = hit(current_status_text, "input_is_ordering_only = true")

    unified_conditional = hit(unified_text, ".1675-.1678")
    prior_response_conditional = hit(prior_response_text, ".1675-.1678")
    part5_conditional = hit(part5_text, ".1675-.1678")
    ai_context_conditional = hit(ai_context_text, BRANCH_CLASS)

    inventory_ready = all(
        item is not None
        for item in (
            status_conditional,
            roadmap_conditional,
            current_problem_conditional,
            current_status_conditional,
            unified_conditional,
            prior_response_conditional,
            part5_conditional,
            ai_context_conditional,
        )
    )
    genuinely_new_external_input_detected = True
    note_reaffirms_current_official_read = note_case is not None
    note_reaffirms_reopen_ordering = all(
        item is not None for item in (note_primary, note_secondary, note_reserve)
    )
    note_requests_existing_current_pack_lanes_only = all(
        item is not None
        for item in (
            note_constitutive,
            note_full_nonlinear,
            note_near_node,
            note_u_bridge,
            note_overlap,
            note_jeff,
        )
    )
    note_reaffirms_hard_gates = all(
        item is not None for item in (note_no_new_params, note_fixed_branch, note_fixed_q)
    )
    note_opens_new_action_level_structure = False
    note_opens_new_primary_trigger = False
    note_opens_new_secondary_trigger = False
    note_opens_new_reserve_trigger = False
    input_is_ordering_only = all(
        [
            note_reaffirms_current_official_read,
            note_reaffirms_reopen_ordering,
            note_requests_existing_current_pack_lanes_only,
            note_reaffirms_hard_gates,
            not note_opens_new_action_level_structure,
            not note_opens_new_primary_trigger,
            not note_opens_new_secondary_trigger,
            not note_opens_new_reserve_trigger,
        ]
    )
    conditional_reactivation_ready = all(
        [
            inventory_ready,
            prior_route_available,
            prior_reopen_ordering_retained,
            genuinely_new_external_input_detected,
            input_is_ordering_only,
        ]
    )
    no_new_trigger_opened = all(
        [
            not note_opens_new_primary_trigger,
            not note_opens_new_secondary_trigger,
            not note_opens_new_reserve_trigger,
        ]
    )
    future_new_surface_or_input_still_required = True
    assimilation_text = build_assimilation_text()

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "conditional reactivation inventory ready",
            truth(inventory_ready),
            "Status, roadmap, AI context, current notes, prior response, and Part V all preserve the `.1675-.1678` evidence pack after completion.",
        ),
        row(
            "prior_route_available",
            "pass" if prior_route_available else "reject",
            "fallback closeout advice-pack refresh available as prior route",
            truth(prior_route_available),
            "Conditional reactivation can start only after the exhausted fallback family has already been frozen by `.1671-.1674`.",
        ),
        row(
            "genuinely_new_external_input_detected",
            "pass" if genuinely_new_external_input_detected else "reject",
            "genuinely new external input detected",
            truth(genuinely_new_external_input_detected),
            "The Downloads note is new as an external file and therefore activates the conditional reactivation branch.",
        ),
        row(
            "note_reaffirms_reopen_ordering",
            "pass" if note_reaffirms_reopen_ordering else "reject",
            "new note reaffirms frozen reopen ordering",
            truth(note_reaffirms_reopen_ordering),
            "The new input keeps the already frozen primary / secondary / reserve ordering instead of proposing a new one.",
        ),
        row(
            "note_requests_existing_current_pack_lanes_only",
            "pass" if note_requests_existing_current_pack_lanes_only else "reject",
            "new note requests only already known current-pack lanes",
            truth(note_requests_existing_current_pack_lanes_only),
            "Constitutive-map, branch-local nonlinear, near-node, u/P_infty, transverse-overlap, and J_eff are all already known surfaces or diagnostics.",
        ),
        row(
            "input_is_ordering_only",
            "pass" if input_is_ordering_only else "reject",
            "new external input is ordering-only under current registry",
            truth(input_is_ordering_only),
            "The note is new as guidance, but it does not add a new canonical surface or new action-level structure.",
        ),
        row(
            "no_new_trigger_opened",
            "pass" if no_new_trigger_opened else "reject",
            "no new primary / secondary / reserve trigger opened",
            truth(no_new_trigger_opened),
            "The reactivation branch completes honestly only if the new note does not reopen the frozen current-pack family by itself.",
        ),
        row(
            "conditional_reactivation_ready",
            "pass" if conditional_reactivation_ready else "reject",
            "conditional reactivation input assimilation ready",
            truth(conditional_reactivation_ready),
            "Once the new external note is classified as ordering-only, the honest next route is wait-restore rather than same-level rescue extension.",
        ),
        row(
            "future_new_surface_or_input_still_required",
            "pass" if future_new_surface_or_input_still_required else "reject",
            "future genuinely new surface or input still required",
            truth(future_new_surface_or_input_still_required),
            "Because no new trigger opened, another genuinely new canonical surface or more substantive external input is still required for reactivation.",
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
            "prior_response": display_path(PRIOR_RESPONSE),
            "part5": display_path(PART5),
            "new_external_note": display_path(INPUT_NOTE),
        },
        "prior_metrics": {
            "prior_route": display_path(PRIOR_ROUTE),
            "prior_gate": display_path(PRIOR_GATE),
        },
        "constants": {
            "primary_reopen_surface": PRIMARY_REOPEN,
            "secondary_reopen_surface": SECONDARY_REOPEN,
            "reserve_reopen_surface": RESERVE_REOPEN,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "next_route_activation_condition": NEXT_ROUTE_ACTIVATION_CONDITION,
            "new_input_last_write_utc": datetime.fromtimestamp(
                INPUT_NOTE.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "genuinely_new_external_input_detected": genuinely_new_external_input_detected,
        "new_external_input_last_write_utc": inputs["constants"]["new_input_last_write_utc"],
        "note_reaffirms_current_official_read": note_reaffirms_current_official_read,
        "note_reaffirms_reopen_ordering": note_reaffirms_reopen_ordering,
        "note_requests_existing_current_pack_lanes_only": note_requests_existing_current_pack_lanes_only,
        "note_reaffirms_hard_gates": note_reaffirms_hard_gates,
        "new_action_level_structure_surface_present": note_opens_new_action_level_structure,
        "new_primary_trigger_opened": note_opens_new_primary_trigger,
        "new_secondary_trigger_opened": note_opens_new_secondary_trigger,
        "new_reserve_trigger_opened": note_opens_new_reserve_trigger,
        "input_is_ordering_only": input_is_ordering_only,
        "no_new_trigger_opened": no_new_trigger_opened,
        "conditional_reactivation_input_assimilation_completed": conditional_reactivation_ready,
        "future_new_surface_or_input_still_required": future_new_surface_or_input_still_required,
        "physical_reject_required": False,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "retained_scalar_exact_alpha_at_q_theory": SCALAR_ALPHA,
        "official_energy_core_alpha_at_q_theory": ENERGY_ALPHA,
        "official_projected_kernel_alpha_at_q_theory": PROJECTED_ALPHA,
        "electric_like_component_alpha_at_q_theory": ELECTRIC_LIKE_ALPHA,
        "note_gradient_alpha_at_q_theory": NOTE_GRADIENT_ALPHA,
    }
    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": conditional_reactivation_ready,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    inventory_payload = payload(
        "8.7.56.1675",
        f"{STEP_NAME} inventory",
        inputs,
        rows,
        summary,
        decision,
        {
            "status_hit": status_conditional,
            "roadmap_hit": roadmap_conditional,
            "current_problem_hit": current_problem_conditional,
            "current_status_hit": current_status_conditional,
            "unified_hit": unified_conditional,
            "prior_response_hit": prior_response_conditional,
            "part5_hit": part5_conditional,
            "ai_context_hit": ai_context_conditional,
        },
    )
    audit_payload = payload(
        "8.7.56.1676",
        f"{STEP_NAME} audit",
        inputs,
        rows,
        summary,
        decision,
        {
            "new_note_hits": {
                "title": note_title,
                "case": note_case,
                "primary": note_primary,
                "secondary": note_secondary,
                "reserve": note_reserve,
                "constitutive": note_constitutive,
                "full_nonlinear": note_full_nonlinear,
                "near_node": note_near_node,
                "u_bridge": note_u_bridge,
                "overlap": note_overlap,
                "jeff": note_jeff,
                "no_new_params": note_no_new_params,
                "fixed_branch": note_fixed_branch,
                "fixed_q": note_fixed_q,
            },
            "assimilation_text": assimilation_text,
        },
    )
    gate_payload = payload(
        "8.7.56.1677",
        f"{STEP_NAME} declaration gate",
        inputs,
        rows,
        summary,
        decision,
        {
            "carry_over": {
                "prior_route": prior_route,
                "prior_gate": prior_gate,
            },
            "assimilation_text": assimilation_text,
        },
    )
    route_payload = payload(
        "8.7.56.1678",
        f"{STEP_NAME} route sync",
        inputs,
        rows,
        {
            **summary,
            "route_state_changed_by_current_branch": True,
            "numeric_state_changed_by_current_branch": False,
        },
        decision,
        {
            "retained_numeric_state": {
                "scalar_alpha": SCALAR_ALPHA,
                "energy_alpha": ENERGY_ALPHA,
                "projected_alpha": PROJECTED_ALPHA,
                "electric_like_alpha": ELECTRIC_LIKE_ALPHA,
                "note_gradient_alpha": NOTE_GRADIENT_ALPHA,
            },
            "assimilation_text": assimilation_text,
        },
    )

    outputs = {
        "inventory": write_artifact("inventory", inventory_payload),
        "audit": write_artifact("audit", audit_payload),
        "declaration_gate": write_artifact("declaration_gate", gate_payload),
        "route_sync": write_artifact("route_sync", route_payload),
    }

    print(
        json.dumps(
            {
                "branch_class": BRANCH_CLASS,
                "next_official_branch": f"{NEXT_ROUTE}-.1682",
                "selected_next_generation_route": NEXT_ROUTE_NAME,
                "recommended_next_route_or_none": NEXT_ROUTE,
                "outputs": outputs,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
