#!/usr/bin/env python3
"""Generate 8.7.56.1691-.1694 conditional reactivation wait-restore artifacts.

After the resolvent decision gate closes as non-canonical under the current
frozen-action pack, the roadmap must return to an explicit conditional wait
state. This branch does not open new computation. It freezes three rules:

1. same-level retries stay blocked,
2. `.1695-.1730` remain conditional future branches only,
3. reactivation requires a genuinely new primary surface:
   - new action-level structure,
   - exact probe-response / amputation theorem,
   - or a substantive pack update that opens one of those surfaces.
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

DECISION_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1687_1690_resolvent_decision_gate_declaration_gate_metrics.json"
)
DECISION_ROUTE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1687_1690_resolvent_decision_gate_route_sync_metrics.json"
)

STEP_TAG = "8.7.56.1691-1694"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor conditional reactivation wait restore"
STEM = build_compact_artifact_stem(STEP_TAG, "conditional_wait_restore", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_resolvent_gate_c_noncanonical_closeout_wait_restore_next"
BRANCH_CLASS = "vector_qball_form_factor_conditional_new_primary_surface_wait_restore_completed"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_pack_update_intake_canonical_surface_inventory"
NEXT_ROUTE = "8.7.56.1695"
NEXT_ROUTE_ACTIVATION_CONDITION = (
    "genuinely new action-level structure, exact probe-response/amputation theorem, "
    "or substantive pack update opens a new canonical primary surface"
)

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


# 関数: `.1691-.1694` を実行する。

def main() -> None:
    """Execute the conditional reactivation wait-restore branch."""
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
        DECISION_GATE,
        DECISION_ROUTE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    long_roadmap_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    decision_gate = read_json(DECISION_GATE)
    decision_route = read_json(DECISION_ROUTE)
    decision_summary = decision_gate["summary"]

    same_level_retry_blocked = True
    conditional_future_branches_dormant = True
    next_route_conditional = True
    substantive_new_primary_surface_required = True
    ordering_only_input_inadmissible = True

    inventory_rows = [
        row(
            "inventory_ready",
            "pass",
            "conditional wait-restore inventory ready",
            1.0,
            "The branch starts only after `.1687-.1690` has frozen the resolvent family as non-canonical and the long-horizon note has fixed `.1695-.1730` as conditional branches.",
        ),
        row(
            "resolvent_closeout_carried_forward",
            "pass",
            "resolvent closeout carried forward",
            1.0,
            "The wait state is meaningful only because the current pack now treats the resolvent family as closed rather than still exploratory.",
        ),
    ]

    audit_rows = [
        row(
            "same_level_retry_blocked",
            "pass",
            "same-level retry blocked",
            truth(same_level_retry_blocked),
            "Current-pack surrogate retries remain blocked after the resolvent Gate C closeout.",
        ),
        row(
            "conditional_future_branches_dormant",
            "pass",
            "conditional future branches dormant",
            truth(conditional_future_branches_dormant),
            "`.1695-.1730` are dormant until a genuinely new primary surface exists.",
        ),
        row(
            "ordering_only_input_inadmissible",
            "pass",
            "ordering-only input inadmissible for reactivation",
            truth(ordering_only_input_inadmissible),
            "An external note that only restates ordering cannot reopen the pack after the resolvent closeout.",
        ),
        row(
            "substantive_new_primary_surface_required",
            "pass",
            "substantive new primary surface required",
            truth(substantive_new_primary_surface_required),
            "Reactivation now requires a new theorem surface, new action-level structure, or a substantive pack update that opens one of them.",
        ),
    ]

    declaration_rows = [
        row(
            "wait_restore_completed",
            "pass",
            "conditional wait restore completed",
            1.0,
            "The current pack is now restored to an explicit dormant state rather than an implicit 'maybe retry' state.",
        ),
        row(
            "next_route_is_conditional",
            "pass",
            "next route is conditional",
            truth(next_route_conditional),
            "The next official route `.1695-.1698` is conditional and must not start automatically.",
        ),
        row(
            "primary_reopen_surface_fixed",
            "pass",
            "primary reopen surface fixed",
            1.0,
            "The primary reopen surface is an exact probe-response/amputation theorem or genuinely new action-level structure beyond the current pack.",
        ),
        row(
            "secondary_reopen_surface_fixed",
            "pass",
            "secondary reopen surface fixed",
            1.0,
            "Constitutive-map or nonlinear-energy reopening remains downstream of a substantive pack update.",
        ),
        row(
            "reserve_reopen_surface_fixed",
            "pass",
            "reserve reopen surface fixed",
            1.0,
            "Future external input is reserve-only unless it opens the new primary surface.",
        ),
        row(
            "physical_reject_not_selected",
            "pass",
            "physical reject not selected",
            1.0,
            "The wait restore freezes the roadmap locally and does not change the broader physical-disposition read.",
        ),
    ]

    route_rows = [
        row(
            "conditional_activation_condition_fixed",
            "pass",
            "conditional activation condition fixed",
            1.0,
            "`.1695-.1698` activates only after genuinely new action-level structure, exact probe-response/amputation theorem, or substantive pack update appears.",
        ),
        row(
            "long_horizon_pack_update_mainline_retained",
            "pass",
            "long-horizon pack-update mainline retained",
            1.0,
            "The next live computation mainline remains theorem/pack-update driven rather than another same-level surrogate retry.",
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
            "decision_gate": display_path(DECISION_GATE),
            "decision_route": display_path(DECISION_ROUTE),
        },
        "constants": {
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "energy_alpha_at_q_theory": ENERGY_ALPHA,
            "projected_alpha_at_q_theory": PROJECTED_ALPHA,
            "primary_reopen_surface": PRIMARY_REOPEN,
            "secondary_reopen_surface": SECONDARY_REOPEN,
            "reserve_reopen_surface": RESERVE_REOPEN,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "next_route_activation_condition": NEXT_ROUTE_ACTIVATION_CONDITION,
        },
    }

    inventory_payload = payload(
        "8.7.56.1691",
        f"{STEP_NAME} inventory",
        inputs,
        inventory_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "physical_reject_required": False,
        },
        {
            "overall_status": f"{BRANCH_CLASS}_inventory_prepared",
            "branch_completed": False,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "hits": {
                "status_current_branch": hit(status_text, "8.7.56.1691-.1694"),
                "roadmap_current_branch": hit(roadmap_text, "8.7.56.1691-.1694"),
                "unified_roadmap_current_branch": hit(
                    unified_roadmap_text,
                    "`.1691-.1694` は **conditional new action-level structure / external-input reactivation wait restore**",
                ),
                "long_roadmap_pack_update": hit(long_roadmap_text, "8.7.56.1695-.1698"),
            },
            "carry_over": {
                "decision_summary": decision_summary,
                "decision_route_summary": decision_route["summary"],
            },
        },
    )

    audit_payload = payload(
        "8.7.56.1692",
        f"{STEP_NAME} audit",
        inputs,
        audit_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "same_level_retry_blocked": same_level_retry_blocked,
            "conditional_future_branches_dormant": conditional_future_branches_dormant,
            "ordering_only_input_inadmissible": ordering_only_input_inadmissible,
            "substantive_new_primary_surface_required": substantive_new_primary_surface_required,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "physical_reject_required": False,
        },
        {
            "overall_status": f"{BRANCH_CLASS}_audit_completed",
            "branch_completed": False,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": {
                "wait_rule": "No same-level retry may restart under the current frozen-action pack after the resolvent Gate C closeout.",
                "activation_rule": "Reactivation requires a genuinely new primary surface rather than another ordering-only or scheme-only refinement.",
            },
        },
    )

    declaration_payload = payload(
        "8.7.56.1693",
        f"{STEP_NAME} declaration gate",
        inputs,
        declaration_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "conditional_wait_restore_completed": True,
            "same_level_retry_blocked": same_level_retry_blocked,
            "conditional_future_branches_dormant": conditional_future_branches_dormant,
            "primary_reopen_surface": PRIMARY_REOPEN,
            "secondary_reopen_surface": SECONDARY_REOPEN,
            "reserve_reopen_surface": RESERVE_REOPEN,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "next_route_activation_condition": NEXT_ROUTE_ACTIVATION_CONDITION,
            "physical_reject_required": False,
        },
        {
            "overall_status": f"{BRANCH_CLASS}_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "hits": {
                "current_problem_missing_object": hit(
                    current_problem_text,
                    "canonical probe-response / amputation map",
                ),
                "current_status_missing_object": hit(
                    current_status_text,
                    "canonical probe-response / amputation map",
                ),
                "part5_resolvent_gate": hit(part5_text, "`.1687-.1690` **resolvent decision gate / fallback return**"),
            },
        },
    )

    route_payload = payload(
        "8.7.56.1694",
        f"{STEP_NAME} route sync",
        inputs,
        route_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "next_route_activation_condition": NEXT_ROUTE_ACTIVATION_CONDITION,
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
                "energy_alpha_at_q_theory": ENERGY_ALPHA,
                "projected_alpha_at_q_theory": PROJECTED_ALPHA,
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
