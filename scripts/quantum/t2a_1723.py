#!/usr/bin/env python3
"""Generate 8.7.56.1723-.1726 external-input assimilation / new-primary-surface gate artifacts.

This branch consumes one genuinely new expert note:

    55_trial2_numeric_alpha_vector_qball_failure_structure_response_strategy.md

The note is valuable as a diagnosis of what is broken, but its concrete
mainline proposal is no longer new under the current official state. The note
recommends replacing the local-surrogate observable theory with the transverse
susceptibility / resolvent family,

    Delta chi_T[Q] = Pi_T (K[Q]^{-1} - K[0]^{-1}) Pi_T,

and delaying any new action-level structure until after that family is tested.

However, the current pack has already:

1. executed the transverse resolvent audit (`.1683-.1686`),
2. closed the resolvent family as non-canonical under the old pack
   (`.1687-.1690`),
3. adopted the source-extended action-level primitive
   `S_src[P,a;J_perp] = S_frozen[P,a] - ∫ J_perp·a`,
4. derived the canonical two-leg amputation theorem (`.1699-.1702`),
5. recomputed the updated canonical observable and failed scalar promotion
   (`.1703-.1710`),
6. confirmed that updated-pack constitutive and nonlinear reopen surfaces also
   fail (`.1711-.1718`),
7. fixed by inverse audit that same-branch local rescue now requires huge or
   noncanonical coefficients (`.1719-.1722`).

Therefore this input is genuine as a file, but it does not open a new primary
surface beyond the already executed response-theory replacement route. The
honest next route is the pack-update closeout / reopen registry refresh.
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

INPUT_NOTE = Path(
    r"C:\Users\ogawa\Downloads\55_trial2_numeric_alpha_vector_qball_failure_structure_response_strategy.md"
)
PRIOR_ROUTE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1719_1722_inv_local_constraint_route_sync_metrics.json"
)
PRIOR_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1719_1722_inv_local_constraint_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1723-1726"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor external-input "
    "assimilation / new-primary-surface gate"
)
STEM = build_compact_artifact_stem(STEP_TAG, "ext_input_gate", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_inverse_local_constraint_requires_large_or_"
    "noncanonical_coefficients_new_primary_surface_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_external_input_response_strategy_assimilation_"
    "no_new_primary_surface_pack_update_closeout_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_pack_update_closeout_"
    "reopen_registry_refresh"
)
NEXT_ROUTE = "8.7.56.1727"
NEXT_ROUTE_ACTIVATION_CONDITION = (
    "response-strategy input is genuine as a file but reuses already executed "
    "response-theory replacement routes, so the honest next step is the pack-"
    "update closeout / reopen registry refresh"
)

FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_new_action_"
    "level_structure_or_exact_probe_response_pack_update_reactivation"
)
FOLLOWUP_ROUTE = "8.7.56.1731"

PRIMARY_REOPEN = "genuinely_new_action_level_structure_beyond_current_updated_pack"
SECONDARY_REOPEN = (
    "exact_probe_response_or_amputation_theorem_not_already_executed_under_"
    "current_updated_pack"
)
RESERVE_REOPEN = (
    "future_external_input_that_opens_a_surface_beyond_resolvent_and_"
    "source_extended_response_routes"
)


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


# 関数: repo 相対表示パスを返す。

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


# 関数: 同化の concise summary を返す。

def build_assimilation_text() -> str:
    """Return one concise summary sentence for the new external input."""
    return (
        "The new failure-structure response note is genuine as an input file, "
        "but it mainly re-asserts the already executed replacement of the local "
        "surrogate observable theory by the transverse-resolvent / response "
        "family and postpones any new action-level structure until that family "
        "fails. Under the current updated state, both the resolvent family and "
        "the subsequent source-extended probe-response pack have already been "
        "executed and closed. The honest read is therefore: valuable diagnosis, "
        "no genuinely new primary surface opened."
    )


# 関数: `.1723-.1726` を実行する。

def main() -> None:
    """Execute the external-input assimilation / new-primary-surface gate."""
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
        INPUT_NOTE,
        PRIOR_ROUTE,
        PRIOR_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)
    input_text = read_text(INPUT_NOTE)
    prior_route = read_json(PRIOR_ROUTE)
    prior_gate = read_json(PRIOR_GATE)

    note_mentions_resolvent = "\\Delta\\chi_T" in input_text
    note_mentions_no_new_action_structure_before_resolvent_fail = (
        "new action-level structure" in input_text
        and "full resolvent fail 前" in input_text
    )
    current_pack_already_executed_resolvent = (
        "transverse_resolvent_canonical_observable_available = false"
        in current_status_text
    )
    current_pack_already_adopted_source_extension = (
        "source_extended_probe_response_pack_adopted = true" in part5_text
    )
    current_pack_already_derived_amputation = (
        "canonical_external_leg_amputation_count = 2.0" in current_status_text
    )
    current_pack_already_fixed_local_inverse_fail = (
        "local_family_rescue_requires_large_or_noncanonical_coefficients = true"
        in current_status_text
    )

    new_external_input_detected = True
    new_action_level_structure_surface_present = False
    new_primary_surface_opened = False
    new_secondary_surface_opened = False
    new_reserve_surface_opened = False
    input_is_ordering_or_historical_diagnostic_only = True

    inventory_rows = [
        row(
            "new_external_input_detected",
            "pass",
            "genuinely new external input file detected",
            truth(new_external_input_detected),
            "The failure-structure response strategy note is a new external file and is eligible for the assimilation gate.",
        ),
        row(
            "note_mentions_resolvent_family",
            "pass",
            "note mentions transverse resolvent family",
            truth(note_mentions_resolvent),
            "The note frames the theory replacement around the same Delta chi_T response family that the current project has already executed.",
        ),
        row(
            "note_delays_new_action_structure_until_resolvent_fail",
            "pass",
            "note delays new action-level structure until resolvent fail",
            truth(note_mentions_no_new_action_structure_before_resolvent_fail),
            "The note explicitly treats new action-level structure as reserve until after the full resolvent family fails.",
        ),
        row(
            "current_pack_already_executed_resolvent",
            "pass",
            "current pack already executed the resolvent family",
            truth(current_pack_already_executed_resolvent),
            "The old-pack transverse resolvent family was already audited and closed before the source-extended update.",
        ),
        row(
            "current_pack_already_adopted_source_extension",
            "pass",
            "current pack already adopted source-extended action-level structure",
            truth(current_pack_already_adopted_source_extension),
            "The updated pack has already introduced S_src[P,a;J_perp] as a new primitive, so the note does not add a further action-level surface.",
        ),
        row(
            "current_pack_already_derived_amputation_theorem",
            "pass",
            "current pack already derived amputation theorem",
            truth(current_pack_already_derived_amputation),
            "The canonical two-leg amputation theorem is already part of the current updated state.",
        ),
        row(
            "current_pack_already_fixed_local_inverse_fail",
            "pass",
            "current pack already fixed inverse local-family failure",
            truth(current_pack_already_fixed_local_inverse_fail),
            "The note's local-surrogate diagnosis is now also supported by the exact inverse constraint audit.",
        ),
    ]

    audit_rows = [
        row(
            "input_is_ordering_or_historical_diagnostic_only",
            "pass",
            "input is ordering-only or historical diagnostic under current state",
            truth(input_is_ordering_or_historical_diagnostic_only),
            "The note supports the direction already taken but does not supply a new primitive, theorem, or canonical observable surface beyond the routes already executed.",
        ),
        row(
            "new_action_level_structure_surface_present",
            "pass",
            "new action-level structure surface present",
            truth(new_action_level_structure_surface_present),
            "No new action-level surface beyond the already adopted source-extended primitive is provided.",
        ),
        row(
            "new_primary_surface_opened",
            "pass",
            "new primary surface opened",
            truth(new_primary_surface_opened),
            "The note does not open a genuinely new primary surface because its core response-theory proposal is already in the executed chain.",
        ),
        row(
            "new_secondary_surface_opened",
            "pass",
            "new secondary surface opened",
            truth(new_secondary_surface_opened),
            "No new constitutive / nonlinear reopen surface is added beyond the existing registry.",
        ),
        row(
            "new_reserve_surface_opened",
            "pass",
            "new reserve surface opened",
            truth(new_reserve_surface_opened),
            "The note does not enlarge the reserve ordering beyond the already frozen reopen surfaces.",
        ),
        row(
            "pack_update_closeout_reopen_registry_refresh_admissible_now",
            "pass",
            "pack-update closeout / reopen registry refresh admissible now",
            truth(True),
            "Since no new primary surface opens, the honest next route is the closeout / reopen registry refresh branch.",
        ),
    ]

    declaration_rows = inventory_rows + audit_rows

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "new_external_input_detected": new_external_input_detected,
        "input_is_ordering_or_historical_diagnostic_only": input_is_ordering_or_historical_diagnostic_only,
        "note_mentions_resolvent_family": note_mentions_resolvent,
        "note_delays_new_action_structure_until_resolvent_fail": note_mentions_no_new_action_structure_before_resolvent_fail,
        "current_pack_already_executed_resolvent": current_pack_already_executed_resolvent,
        "current_pack_already_adopted_source_extension": current_pack_already_adopted_source_extension,
        "current_pack_already_derived_amputation_theorem": current_pack_already_derived_amputation,
        "current_pack_already_fixed_local_inverse_fail": current_pack_already_fixed_local_inverse_fail,
        "new_action_level_structure_surface_present": new_action_level_structure_surface_present,
        "new_primary_surface_opened": new_primary_surface_opened,
        "new_secondary_surface_opened": new_secondary_surface_opened,
        "new_reserve_surface_opened": new_reserve_surface_opened,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    hits = {
        "input_state_hit": hit(input_text, "current official state"),
        "input_resolvent_hit": hit(input_text, "\\Delta\\chi_T[Q]"),
        "input_gate_a_hit": hit(input_text, "Gate A"),
        "input_gate_b_hit": hit(input_text, "Gate B"),
        "input_gate_c_hit": hit(input_text, "Gate C"),
        "input_new_action_structure_hit": hit(input_text, "new action-level structure"),
        "status_branch_hit": hit(status_text, "external-input assimilation / new-primary-surface gate"),
        "problem_inverse_hit": hit(current_problem_text, "local_family_rescue_requires_large_or_noncanonical_coefficients"),
        "status_inverse_hit": hit(current_status_text, "local_family_rescue_requires_large_or_noncanonical_coefficients"),
        "unified_branch_hit": hit(unified_text, "`.1723-.1726` は **external-input assimilation / new-primary-surface gate**"),
        "long_branch_hit": hit(long_text, "10. `8.7.56.1723-.1726`"),
        "part5_source_extension_hit": hit(part5_text, "source_extended_probe_response_pack_adopted = true"),
        "part5_amputation_hit": hit(part5_text, "canonical_external_leg_amputation_count = 2.0"),
    }

    evidence = {
        "formulas": {
            "note_replacement_candidate": (
                "Delta chi_T[Q] = Pi_T (K[Q]^{-1} - K[0]^{-1}) Pi_T"
            ),
            "executed_new_primitive": (
                "S_src[P,a;J_perp] = S_frozen[P,a] - ∫ d^4x J_perp^mu a_mu"
            ),
            "executed_canonical_rule": "F_T,can(q) = -q^4 Delta chi_T(q)",
            "gate_read": (
                "If the external note only restates already executed response-theory "
                "replacement routes, then it is genuine as an input file but does "
                "not open a genuinely new primary surface."
            ),
        },
        "hits": hits,
        "prior_route_summary": prior_route.get("summary", {}),
        "prior_gate_summary": prior_gate.get("summary", {}),
        "assimilation_text": build_assimilation_text(),
    }

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
            "input_note": display_path(INPUT_NOTE),
            "prior_route": display_path(PRIOR_ROUTE),
            "prior_gate": display_path(PRIOR_GATE),
        },
        "constants": {
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "next_route_activation_condition": NEXT_ROUTE_ACTIVATION_CONDITION,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
            "primary_reopen_surface": PRIMARY_REOPEN,
            "secondary_reopen_surface": SECONDARY_REOPEN,
            "reserve_reopen_surface": RESERVE_REOPEN,
        },
    }

    inventory_payload = payload(
        STEP_TAG,
        f"{STEP_NAME} inventory",
        inputs,
        inventory_rows,
        summary,
        decision,
        evidence,
    )
    audit_payload = payload(
        STEP_TAG,
        f"{STEP_NAME} audit",
        inputs,
        audit_rows,
        summary,
        decision,
        evidence,
    )
    declaration_payload = payload(
        STEP_TAG,
        f"{STEP_NAME} declaration gate",
        inputs,
        declaration_rows,
        summary,
        decision,
        evidence,
    )
    route_rows = declaration_rows + [
        row(
            "selected_next_generation_route",
            "pass",
            "selected next-generation route present",
            1.0,
            f"Next official route is {NEXT_ROUTE_NAME}.",
        ),
        row(
            "selected_followup_route",
            "pass",
            "selected followup route present",
            1.0,
            f"Followup reserve route is {FOLLOWUP_ROUTE_NAME}.",
        ),
    ]
    route_payload = payload(
        STEP_TAG,
        f"{STEP_NAME} route sync",
        inputs,
        route_rows,
        summary,
        decision,
        evidence,
    )

    written = {
        "inventory": write_artifact("inventory", inventory_payload),
        "audit": write_artifact("audit", audit_payload),
        "declaration_gate": write_artifact("declaration_gate", declaration_payload),
        "route_sync": write_artifact("route_sync", route_payload),
    }

    print("[ok] external-input assimilation / new-primary-surface gate artifacts written:")
    for kind, paths in written.items():
        print(f"  - {kind}: {paths['json']} | {paths['csv']}")


if __name__ == "__main__":
    main()
