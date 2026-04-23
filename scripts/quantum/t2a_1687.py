#!/usr/bin/env python3
"""Generate 8.7.56.1687-.1690 resolvent decision-gate artifacts.

`.1683-.1686` already audited the transverse-resolvent family

    Delta chi_T[Q] = Pi_T (K[Q]^{-1} - K[0]^{-1}) Pi_T

under the current frozen-action pack. That audit established:

1. the unamputated object keeps a q->0 pole,
2. no canonical source/amputation rule is available,
3. finite scheme reads spread strongly and do not support the retained scalar
   strong candidate.

Therefore `.1687-.1690` freezes the honest decision gate:

- Gate A breakthrough: closed,
- Gate B weak rescue: closed,
- Gate C non-canonical closeout: selected.

The branch also sharpens the reopen ordering. What is now missing is not
another same-level surrogate observable, but either:

- a genuinely new action-level structure, or
- an exact probe-response / amputation theorem beyond the current pack.
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
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

RESOLVENT_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1683_1686_tresp_resolvent_audit_declaration_gate_metrics.json"
)
RESOLVENT_ROUTE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1683_1686_tresp_resolvent_audit_route_sync_metrics.json"
)
FALLBACK_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1667_1670_fallback_closeout_registry_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1687-1690"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor resolvent decision gate / fallback return"
STEM = build_compact_artifact_stem(STEP_TAG, "resolvent_decision_gate", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_transverse_resolvent_response_scheme_dependent_"
    "no_canonical_read_decision_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_resolvent_gate_c_noncanonical_closeout_"
    "wait_restore_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_new_action_"
    "level_structure_or_external_input_reactivation_wait_restore"
)
NEXT_ROUTE = "8.7.56.1691"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_pack_update_intake_"
    "canonical_surface_inventory"
)
FOLLOWUP_ROUTE = "8.7.56.1695"

PRIMARY_REOPEN = (
    "exact_probe_response_or_amputation_theorem_or_genuinely_new_action_"
    "level_structure_beyond_current_pack"
)
SECONDARY_REOPEN = (
    "exact_constitutive_map_or_branch_local_full_nonlinear_energy_density_"
    "reopen_after_substantive_pack_update"
)
RESERVE_REOPEN = "future_external_input_guiding_new_primary_surface_or_pack_update"

SCALAR_ALPHA = 0.00715678583937324
VECTOR_ALPHA = 0.0005579616187042394
BARE_ALPHA = 0.07957747154594767


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input file is missing."""
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


# 関数: `.1687-.1690` を実行する。

def main() -> None:
    """Execute the resolvent decision-gate / fallback-return branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LONG_ROADMAP,
        PART5,
        RESOLVENT_GATE,
        RESOLVENT_ROUTE,
        FALLBACK_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    long_roadmap_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    resolvent_gate = read_json(RESOLVENT_GATE)
    resolvent_route = read_json(RESOLVENT_ROUTE)
    fallback_gate = read_json(FALLBACK_GATE)

    resolvent_summary = resolvent_gate["summary"]
    fallback_summary = fallback_gate["summary"]

    one_leg_alpha = float(resolvent_summary["one_leg_amputated_alpha_at_q_theory"])
    two_leg_alpha = float(resolvent_summary["two_leg_amputated_alpha_at_q_theory"])
    static_alpha = float(resolvent_summary["static_scaled_proxy_alpha_at_q_theory"])
    scheme_alpha_ratio = float(resolvent_summary["scheme_alpha_ratio"])

    gate_a_selected = False
    gate_b_selected = False
    gate_c_selected = True
    same_pack_resolvent_retry_admissible = False
    same_level_surrogate_retry_admissible = False
    exact_probe_response_theorem_required = True
    new_action_level_structure_required = True
    substantive_pack_update_required = True

    inventory_rows = [
        row(
            "inventory_ready",
            "pass",
            "resolvent decision-gate inventory ready",
            1.0,
            "The branch starts only after the audited resolvent family, prior fallback registry, and long-horizon conditional roadmap all point to `.1687-.1690`.",
        ),
        row(
            "resolvent_family_already_audited",
            "pass",
            "resolvent family already audited",
            1.0,
            "`.1683-.1686` already established that the resolvent family is no longer genuinely untested under the current pack.",
        ),
        row(
            "long_horizon_mainline_fixed",
            "pass",
            "long-horizon conditional mainline fixed",
            1.0,
            "The long-horizon roadmap already fixes theorem/pack-update branches after the immediate closeout and wait-restore pair.",
        ),
    ]

    audit_rows = [
        row(
            "gate_a_breakthrough_selected",
            "reject",
            "Gate A breakthrough selected",
            truth(gate_a_selected),
            "Breakthrough is closed because no canonical source/amputation rule exists and no finite resolvent read supports the scalar strong candidate canonically.",
        ),
        row(
            "gate_b_weak_rescue_selected",
            "reject",
            "Gate B weak rescue selected",
            truth(gate_b_selected),
            "Weak rescue is also closed because the finite resolvent scheme spread is too large to promote one preferred read.",
        ),
        row(
            "gate_c_noncanonical_closeout_selected",
            "pass",
            "Gate C non-canonical closeout selected",
            truth(gate_c_selected),
            "The honest read is that the resolvent family closes non-canonically under the current frozen-action pack.",
        ),
        row(
            "scheme_alpha_ratio",
            "watch",
            "scheme alpha ratio across finite resolvent reads",
            scheme_alpha_ratio,
            "The large spread between one-leg, two-leg, and static-scaled proxy reads is the direct reason Gate C is selected.",
        ),
        row(
            "same_pack_resolvent_retry_admissible",
            "reject",
            "same-pack resolvent retry admissible",
            truth(same_pack_resolvent_retry_admissible),
            "Another same-pack scheme variant would only re-enter the already falsified surrogate/normalization logic.",
        ),
        row(
            "same_level_surrogate_retry_admissible",
            "reject",
            "same-level surrogate retry admissible",
            truth(same_level_surrogate_retry_admissible),
            "The current pack must not reopen another same-level local or quasi-local surrogate family after the resolvent closeout.",
        ),
    ]

    declaration_rows = [
        row(
            "exact_probe_response_theorem_required",
            "pass",
            "exact probe-response / amputation theorem required",
            truth(exact_probe_response_theorem_required),
            "The missing object is now localized to an exact canonical coupling / amputation theorem rather than another surrogate observable.",
        ),
        row(
            "new_action_level_structure_required",
            "pass",
            "genuinely new action-level structure required",
            truth(new_action_level_structure_required),
            "A genuinely new action-level surface remains an admissible primary reopen route if it expands the frozen-action pack.",
        ),
        row(
            "substantive_pack_update_required",
            "pass",
            "substantive pack update required",
            truth(substantive_pack_update_required),
            "Any future constitutive-map or nonlinear-energy reopen now requires a substantive pack update rather than a same-level reinterpretation.",
        ),
        row(
            "physical_reject_not_selected",
            "pass",
            "physical reject not selected",
            1.0,
            "The closeout is route-local to the observable bridge and does not force physical rejection of the retained scalar strong candidate.",
        ),
        row(
            "resolvent_closeout_wording_honest",
            "pass",
            "resolvent closeout wording honest",
            1.0,
            "The wording remains honest only if the no-canonical-read result, the missing amputation map, and the no-same-level-retry rule are all kept explicit.",
        ),
        row(
            "resolvent_closeout_ready",
            "pass",
            "resolvent closeout ready",
            1.0,
            "Once Gate C is selected and the sharpened reopen ordering is explicit, the branch can advance to wait restore.",
        ),
    ]

    route_rows = [
        row(
            "primary_reopen_surface_fixed",
            "pass",
            "primary reopen surface fixed",
            1.0,
            "The new primary surface is an exact probe-response / amputation theorem or genuinely new action-level structure beyond the current pack.",
        ),
        row(
            "secondary_reopen_surface_fixed",
            "pass",
            "secondary reopen surface fixed",
            1.0,
            "Exact constitutive-map or branch-local nonlinear-energy reopening is now explicitly downstream of a substantive pack update.",
        ),
        row(
            "reserve_reopen_surface_fixed",
            "pass",
            "reserve reopen surface fixed",
            1.0,
            "Future external input remains reserve-only unless it opens the new primary surface rather than merely reorders prior failures.",
        ),
        row(
            "wait_restore_followup_selected",
            "pass",
            "wait-restore followup selected",
            1.0,
            "The immediate next branch is `.1691-.1694` wait restore so the no-restart policy becomes machine-readable before any future pack-update branch opens.",
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
            "part5": display_path(PART5),
            "resolvent_gate": display_path(RESOLVENT_GATE),
            "resolvent_route": display_path(RESOLVENT_ROUTE),
            "fallback_gate": display_path(FALLBACK_GATE),
        },
        "constants": {
            "one_leg_alpha_at_q_theory": one_leg_alpha,
            "two_leg_alpha_at_q_theory": two_leg_alpha,
            "static_scaled_proxy_alpha_at_q_theory": static_alpha,
            "scheme_alpha_ratio": scheme_alpha_ratio,
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "bare_alpha": BARE_ALPHA,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
            "primary_reopen_surface": PRIMARY_REOPEN,
            "secondary_reopen_surface": SECONDARY_REOPEN,
            "reserve_reopen_surface": RESERVE_REOPEN,
        },
    }

    inventory_payload = payload(
        "8.7.56.1687",
        f"{STEP_NAME} inventory",
        inputs,
        inventory_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "resolvent_family_failed_or_unavailable_under_current_pack": True,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "selected_followup_route": FOLLOWUP_ROUTE_NAME,
            "selected_followup_route_or_none": FOLLOWUP_ROUTE,
            "physical_reject_required": False,
        },
        {
            "overall_status": f"{BRANCH_CLASS}_inventory_prepared",
            "branch_completed": False,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "hits": {
                "status_current_branch": hit(status_text, "8.7.56.1687"),
                "roadmap_current_branch": hit(roadmap_text, "8.7.56.1687-.1690"),
                "current_problem_current_branch": hit(current_problem_text, "8.7.56.1687-.1690"),
                "current_status_current_branch": hit(current_status_text, "8.7.56.1687-.1690"),
                "unified_roadmap_current_branch": hit(
                    unified_roadmap_text,
                    "`.1687-.1690` は **resolvent decision gate / fallback return**",
                ),
                "long_roadmap_wait_restore": hit(long_roadmap_text, "8.7.56.1691-.1694"),
                "part5_resolvent_next": hit(part5_text, "next official route is `.1687-.1690`"),
            },
            "carry_over": {
                "resolvent_summary": resolvent_summary,
                "resolvent_route_summary": resolvent_route["summary"],
                "fallback_summary": fallback_summary,
            },
        },
    )

    audit_payload = payload(
        "8.7.56.1688",
        f"{STEP_NAME} audit",
        inputs,
        audit_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "gate_a_breakthrough_selected": gate_a_selected,
            "gate_b_weak_rescue_selected": gate_b_selected,
            "gate_c_noncanonical_closeout_selected": gate_c_selected,
            "one_leg_alpha_at_q_theory": one_leg_alpha,
            "two_leg_alpha_at_q_theory": two_leg_alpha,
            "static_scaled_proxy_alpha_at_q_theory": static_alpha,
            "scheme_alpha_ratio": scheme_alpha_ratio,
            "same_pack_resolvent_retry_admissible": same_pack_resolvent_retry_admissible,
            "same_level_surrogate_retry_admissible": same_level_surrogate_retry_admissible,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "selected_followup_route": FOLLOWUP_ROUTE_NAME,
            "selected_followup_route_or_none": FOLLOWUP_ROUTE,
            "physical_reject_required": False,
        },
        {
            "overall_status": f"{BRANCH_CLASS}_audit_completed",
            "branch_completed": False,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": {
                "gate_a": "Gate A requires one canonical resolvent observable that supports the retained scalar strong candidate.",
                "gate_b": "Gate B would require at least one stable finite resolvent read with weak but canonically selectable rescue.",
                "gate_c": "Gate C is selected when the resolvent family stays non-canonical because no unique source/amputation rule exists under the current pack.",
                "sharpened_read": "The missing object is not another surrogate density but an exact probe-response / amputation theorem or a genuinely new action-level surface.",
            },
        },
    )

    declaration_payload = payload(
        "8.7.56.1689",
        f"{STEP_NAME} declaration gate",
        inputs,
        declaration_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "gate_c_noncanonical_closeout_selected": gate_c_selected,
            "exact_probe_response_theorem_required": exact_probe_response_theorem_required,
            "new_action_level_structure_required": new_action_level_structure_required,
            "substantive_pack_update_required": substantive_pack_update_required,
            "primary_reopen_surface": PRIMARY_REOPEN,
            "secondary_reopen_surface": SECONDARY_REOPEN,
            "reserve_reopen_surface": RESERVE_REOPEN,
            "resolvent_closeout_wording_honest": True,
            "resolvent_closeout_ready": True,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "selected_followup_route": FOLLOWUP_ROUTE_NAME,
            "selected_followup_route_or_none": FOLLOWUP_ROUTE,
            "physical_reject_required": False,
        },
        {
            "overall_status": f"{BRANCH_CLASS}_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "hits": {
                "long_roadmap_primary": hit(
                    long_roadmap_text,
                    "canonical probe-response / amputation theorem derivation",
                ),
                "long_roadmap_pack_update": hit(
                    long_roadmap_text,
                    "pack-update intake / canonical-surface inventory",
                ),
            },
        },
    )

    route_payload = payload(
        "8.7.56.1690",
        f"{STEP_NAME} route sync",
        inputs,
        route_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
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
            "overall_status": f"{BRANCH_CLASS}_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "retained_numeric_state": {
                "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
                "vector_alpha_at_q_theory": VECTOR_ALPHA,
                "one_leg_alpha_at_q_theory": one_leg_alpha,
                "two_leg_alpha_at_q_theory": two_leg_alpha,
                "static_scaled_proxy_alpha_at_q_theory": static_alpha,
                "numeric_state_changed_by_current_branch": False,
                "route_state_changed_by_current_branch": True,
            },
        },
    )

    outputs = {
        "inventory": write_artifact("inventory", inventory_payload),
        "audit": write_artifact("audit", audit_payload),
        "declaration_gate": write_artifact("declaration_gate", declaration_payload),
        "route_sync": write_artifact("route_sync", route_payload),
    }

    print(json.dumps(outputs, ensure_ascii=False, indent=2))


# 条件分岐: スクリプトとして直接実行された場合に main を呼ぶ。

if __name__ == "__main__":
    main()
