#!/usr/bin/env python3
"""Generate 8.7.56.1571-.1574 quadratic-expansion mainline-reset artifacts.

The prior `.1563-.1570` branches completed the frozen-action direct-current
lane:

1. derive the explicit five-piece split of `J_eff^mu`,
2. classify `J_eff^0` honestly under the current pack,
3. fix the result as Case IV-like zero under the current pack.

An expert directive then sharpened the retry-gate judgment again: once the
linear tadpole/current lane has been exhausted honestly, the next computation
should not loop back into wording or microscopic-functional searches. The next
object to compute is the quadratic fluctuation operator obtained from the
frozen action under

    P_mu = P_mu^Qball + a_mu

and the collection of the `a_mu^2` terms,

    L_total^vec |_{a^2} = (1/2) a_mu K^{mu nu}[Q] a_nu .

This branch therefore closes the zero-current-pack disposition briefly and
resets the scientific mainline toward the quadratic `K^{mu nu}[Q]` lane.
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
PRIOR_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1567_1570_jeff_q0_class_declaration_gate_metrics.json"
)
DIRECTIVE_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_quadratic_expansion_20260328.md"
)

STEP_TAG = "8.7.56.1571-1574"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor quadratic expansion mainline reset"
)
STEM = build_compact_artifact_stem(STEP_TAG, "quad_mainline_reset", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_jeff_charge_density_zero_current_pack_disposition_sync_next"
BRANCH_CLASS = "vector_qball_form_factor_quadratic_expansion_mainline_reset_completed"
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_frozen_action_quadratic_k_operator_derivation"
)
NEXT_ROUTE = "8.7.56.1575"
NEXT_SECONDARY_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_quadratic_operator_structure_classification"
)
NEXT_SECONDARY_ROUTE = "8.7.56.1579"
NEXT_DISPOSITION_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_quadratic_operator_disposition_sync"
)
NEXT_DISPOSITION_ROUTE = "8.7.56.1583"
DOWNSTREAM_SOURCE_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_effective_source_theorem_revisit_after_quadratic_operator"
)
DOWNSTREAM_DICTIONARY_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_observable_dictionary_revisit_after_quadratic_operator"
)


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

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


# 関数: 表示用の相対パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分文字列に一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line matching one substring."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準 metrics row を構成する。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準 payload を構成する。

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


# 関数: quadratic branch の基礎式を返す。

def build_formulae() -> dict[str, str]:
    """Return the quadratic branch formulas."""
    return {
        "fluctuation_split": "P_mu = P_mu^Qball + a_mu",
        "quadratic_collect": "L_total^vec|_(a^2) = (1/2) a_mu K^{mu nu}[Q] a_nu",
        "free_kinetic_piece": "-(Z_P/4) f_{mu nu} f^{mu nu} with f_{mu nu}=partial_mu a_nu-partial_nu a_mu",
        "nonlinear_core": "U(Q^2 + 2 Q·a + a^2) -> U''(Q^2)(Q·a)^2 + U'(Q^2)a^2 at quadratic order",
        "classification_goal": "Classify Delta K[Q] as scalar-foundation / shifted-structure / transparent-zero before any renewed source theorem or dictionary work",
    }


# 関数: `.1571-.1574` を実行する。

def main() -> None:
    """Execute the quadratic-expansion mainline-reset branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PART5,
        PRIOR_GATE,
        DIRECTIVE_NOTE,
    ):
        require(path)

    prior_gate = read_json(PRIOR_GATE)
    prior_summary = prior_gate["summary"]
    directive_text = read_text(DIRECTIVE_NOTE)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part5_text = read_text(PART5)

    prior_zero_ready = bool(
        prior_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and prior_summary.get("classification_case_iv_zero_under_current_pack", False)
        and prior_summary.get("zero_structure_selected_under_current_pack", False)
    )
    directive_requires_quadratic_expansion = (
        hit(directive_text, "a_μ 二次項") is not None
    )
    directive_replaces_linear_lane = (
        hit(directive_text, "一次が消えた。二次を見る。") is not None
    )
    directive_bans_new_parameters = (
        hit(directive_text, "新しい自由パラメータなし") is not None
    )
    directive_bans_su2_hopf_defect_micro = (
        hit(directive_text, "SU(2) / Hopf / defect / microscopic functional は導入しない")
        is not None
    )
    directive_focuses_on_step4 = hit(directive_text, "Step 4 が核心") is not None

    zero_current_pack_sync_completed = prior_zero_ready
    disposition_case_selected = prior_zero_ready
    quadratic_operator_mainline_promoted = bool(
        prior_zero_ready and directive_requires_quadratic_expansion
    )
    microscopic_functional_reopen_demoted_from_mainline = bool(
        directive_bans_su2_hopf_defect_micro and quadratic_operator_mainline_promoted
    )
    quadratic_structure_classification_scheduled = quadratic_operator_mainline_promoted
    quadratic_disposition_sync_scheduled = quadratic_operator_mainline_promoted
    source_theorem_now_downstream_of_quadratic = quadratic_operator_mainline_promoted
    observable_dictionary_now_downstream_of_quadratic_and_source = (
        quadratic_operator_mainline_promoted
    )
    physical_reject_required = False

    rows_inventory = [
        row(
            "prior_zero_case",
            "pass" if prior_zero_ready else "fail",
            "prior_zero_ready",
            truth(prior_zero_ready),
            "The direct J_eff lane already closed at zero under the current pack.",
        ),
        row(
            "directive_quadratic",
            "pass" if directive_requires_quadratic_expansion else "fail",
            "directive_requires_quadratic_expansion",
            truth(directive_requires_quadratic_expansion),
            "The new directive explicitly instructs the a_mu quadratic expansion.",
        ),
        row(
            "directive_linear_stop",
            "pass" if directive_replaces_linear_lane else "fail",
            "directive_replaces_linear_lane",
            truth(directive_replaces_linear_lane),
            "The directive treats the linear lane as exhausted and points to quadratic terms.",
        ),
        row(
            "directive_no_new_params",
            "pass" if directive_bans_new_parameters else "fail",
            "directive_bans_new_parameters",
            truth(directive_bans_new_parameters),
            "The directive keeps the working action fixed and parameter-free.",
        ),
        row(
            "directive_no_micro_lane",
            "pass" if directive_bans_su2_hopf_defect_micro else "fail",
            "directive_bans_su2_hopf_defect_micro",
            truth(directive_bans_su2_hopf_defect_micro),
            "The directive bans SU(2)/Hopf/defect/microscopic-functional additions in this branch.",
        ),
    ]

    inventory = payload(
        STEP_TAG,
        STEP_NAME,
        inputs={
            "required_paths": [
                display_path(PRIOR_GATE),
                display_path(CURRENT_PROBLEM),
                display_path(CURRENT_STATUS),
                display_path(UNIFIED_ROADMAP),
                display_path(PART5),
                display_path(DIRECTIVE_NOTE),
            ],
            "prior_problem_classification": prior_summary.get(
                "trial2_numeric_alpha_problem_classification"
            ),
            "current_step_context": "direct_jeff_zero_current_pack_case_iv",
            "candidate_mainline_shift": "quadratic_a_mu_expansion",
        },
        rows=rows_inventory,
        summary={
            "prior_zero_ready": prior_zero_ready,
            "directive_requires_quadratic_expansion": directive_requires_quadratic_expansion,
            "directive_replaces_linear_lane": directive_replaces_linear_lane,
            "directive_bans_new_parameters": directive_bans_new_parameters,
            "directive_bans_su2_hopf_defect_micro": directive_bans_su2_hopf_defect_micro,
        },
        decision={
            "inventory_ready": prior_zero_ready and directive_requires_quadratic_expansion,
        },
        evidence={
            "current_problem_hits": [
                hit(current_problem_text, "classification_case_iv_zero_under_current_pack = true"),
                hit(current_problem_text, "microscopic matter-current / rotational-source functional reopen"),
            ],
            "current_status_hits": [
                hit(current_status_text, "classification_case_iv_zero_under_current_pack = true"),
                hit(current_status_text, "microscopic matter-current / rotational-source functional reopen"),
            ],
            "directive_hits": [
                hit(directive_text, "一次が消えた。二次を見る。"),
                hit(directive_text, "L_total^vec|_{a^2}"),
                hit(directive_text, "Step 4 が核心"),
            ],
        },
    )

    rows_audit = [
        row(
            "zero_sync",
            "pass" if zero_current_pack_sync_completed else "fail",
            "zero_current_pack_sync_completed",
            truth(zero_current_pack_sync_completed),
            "The zero-current-pack result is honest and can be synced without reopening the linear lane.",
        ),
        row(
            "quad_mainline",
            "pass" if quadratic_operator_mainline_promoted else "fail",
            "quadratic_operator_mainline_promoted",
            truth(quadratic_operator_mainline_promoted),
            "Quadratic K[Q] derivation is promoted to the next scientific mainline.",
        ),
        row(
            "micro_lane_demoted",
            "pass" if microscopic_functional_reopen_demoted_from_mainline else "fail",
            "microscopic_functional_reopen_demoted_from_mainline",
            truth(microscopic_functional_reopen_demoted_from_mainline),
            "Microscopic-functional reopen is demoted from the immediate mainline by the new directive.",
        ),
        row(
            "quad_case_schedule",
            "pass" if quadratic_structure_classification_scheduled else "fail",
            "quadratic_structure_classification_scheduled",
            truth(quadratic_structure_classification_scheduled),
            "The next downstream checkpoint is quadratic-operator structure classification.",
        ),
        row(
            "source_downstream",
            "pass" if source_theorem_now_downstream_of_quadratic else "fail",
            "source_theorem_now_downstream_of_quadratic",
            truth(source_theorem_now_downstream_of_quadratic),
            "Effective source theorem is downstream of the quadratic operator branch.",
        ),
        row(
            "dictionary_downstream",
            "pass" if observable_dictionary_now_downstream_of_quadratic_and_source else "fail",
            "observable_dictionary_now_downstream_of_quadratic_and_source",
            truth(observable_dictionary_now_downstream_of_quadratic_and_source),
            "Observable dictionary stays downstream of both quadratic structure and any later source theorem.",
        ),
        row(
            "directive_step4",
            "pass" if directive_focuses_on_step4 else "fail",
            "directive_focuses_on_step4",
            truth(directive_focuses_on_step4),
            "The directive explicitly localizes the branch core in the nonlinear quadratic expansion.",
        ),
    ]

    audit = payload(
        STEP_TAG,
        STEP_NAME,
        inputs=inventory["summary"],
        rows=rows_audit,
        summary={
            "zero_current_pack_sync_completed": zero_current_pack_sync_completed,
            "disposition_case_selected": disposition_case_selected,
            "quadratic_operator_mainline_promoted": quadratic_operator_mainline_promoted,
            "microscopic_functional_reopen_demoted_from_mainline": microscopic_functional_reopen_demoted_from_mainline,
            "quadratic_structure_classification_scheduled": quadratic_structure_classification_scheduled,
            "quadratic_disposition_sync_scheduled": quadratic_disposition_sync_scheduled,
            "source_theorem_now_downstream_of_quadratic": source_theorem_now_downstream_of_quadratic,
            "observable_dictionary_now_downstream_of_quadratic_and_source": observable_dictionary_now_downstream_of_quadratic_and_source,
            "physical_reject_required": physical_reject_required,
        },
        decision={
            "audit_passed": (
                zero_current_pack_sync_completed
                and quadratic_operator_mainline_promoted
                and microscopic_functional_reopen_demoted_from_mainline
            ),
        },
        evidence={
            "formulas": build_formulae(),
            "unified_roadmap_hits": [
                hit(unified_roadmap_text, "`.1567-.1570` では **`J_{\\mathrm{eff}}^0` structure classification**"),
                hit(unified_roadmap_text, "`.1571-.1574` は **`J_{\\mathrm{eff}}` Case IV / zero-current-pack disposition sync**"),
            ],
            "part5_hits": [
                hit(part5_text, "next mainline は **`8.7.56.1571-.1574`"),
                hit(part5_text, "microscopic matter-current / rotational-source functional reopen"),
            ],
        },
    )

    declaration_summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "zero_current_pack_sync_completed": zero_current_pack_sync_completed,
        "disposition_case_selected": disposition_case_selected,
        "selected_disposition_case": "case_iv_zero_under_current_pack",
        "quadratic_operator_mainline_promoted": quadratic_operator_mainline_promoted,
        "microscopic_functional_reopen_demoted_from_mainline": microscopic_functional_reopen_demoted_from_mainline,
        "quadratic_structure_classification_scheduled": quadratic_structure_classification_scheduled,
        "quadratic_disposition_sync_scheduled": quadratic_disposition_sync_scheduled,
        "source_theorem_now_downstream_of_quadratic": source_theorem_now_downstream_of_quadratic,
        "observable_dictionary_now_downstream_of_quadratic_and_source": observable_dictionary_now_downstream_of_quadratic_and_source,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": NEXT_SECONDARY_ROUTE_NAME,
        "selected_followup_route_or_none": NEXT_SECONDARY_ROUTE,
        "selected_disposition_route": NEXT_DISPOSITION_ROUTE_NAME,
        "selected_disposition_route_or_none": NEXT_DISPOSITION_ROUTE,
        "downstream_source_route_name": DOWNSTREAM_SOURCE_ROUTE_NAME,
        "downstream_dictionary_route_name": DOWNSTREAM_DICTIONARY_ROUTE_NAME,
        "physical_reject_required": physical_reject_required,
        "new_free_parameters_introduced": False,
    }
    declaration = payload(
        STEP_TAG,
        STEP_NAME,
        inputs=audit["summary"],
        rows=rows_audit,
        summary=declaration_summary,
        decision={
            "declaration_gate_passed": (
                zero_current_pack_sync_completed
                and quadratic_operator_mainline_promoted
                and not physical_reject_required
            ),
        },
        evidence={"formulas": build_formulae()},
    )

    route_summary = {
        "route_state_changed_by_current_branch": True,
        "numeric_state_changed_by_current_branch": False,
        "current_official_step_after_branch": "8.7.56.1575",
        "current_official_branch_after_branch": "8.7.56.1575-.1578",
        "current_official_next_route": NEXT_ROUTE_NAME,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "selected_followup_route": NEXT_SECONDARY_ROUTE_NAME,
        "selected_disposition_route": NEXT_DISPOSITION_ROUTE_NAME,
        "future_external_input_side_lane_retained": True,
        "physical_reject_required": physical_reject_required,
    }
    route_sync = payload(
        STEP_TAG,
        STEP_NAME,
        inputs=declaration_summary,
        rows=[
            row(
                "route_change",
                "pass",
                "route_state_changed_by_current_branch",
                1.0,
                "The branch changes the scientific route from linear disposition sync to quadratic mainline reset.",
            ),
            row(
                "numeric_hold",
                "pass",
                "numeric_state_changed_by_current_branch",
                0.0,
                "No numeric candidate is changed by this route reset.",
            ),
        ],
        summary=route_summary,
        decision={
            "route_sync_passed": True,
        },
        evidence={
            "selected_routes": {
                "next": NEXT_ROUTE_NAME,
                "classification": NEXT_SECONDARY_ROUTE_NAME,
                "disposition": NEXT_DISPOSITION_ROUTE_NAME,
                "downstream_source": DOWNSTREAM_SOURCE_ROUTE_NAME,
                "downstream_dictionary": DOWNSTREAM_DICTIONARY_ROUTE_NAME,
            }
        },
    )

    manifest = {
        "inventory": write_artifact("inventory", inventory),
        "audit": write_artifact("audit", audit),
        "declaration_gate": write_artifact("declaration_gate", declaration),
        "route_sync": write_artifact("route_sync", route_sync),
    }

    print(json.dumps({"step": STEP_TAG, "stem": STEM, "artifacts": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
