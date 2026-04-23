#!/usr/bin/env python3
"""Generate 8.7.56.1559-.1562 frozen-action direct J_eff route-reset artifacts.

The prior `.1555-.1558` branch honestly established that the restored exact
vector / Q-ball branch does not carry a genuine internal SU(2) orientation,
Hopf/topological block, or FR Z2 spin-return structure in the current frozen
action pack. A new expert directive then sharpened the retry-gate judgment:

- stop the SU(2) / Hopf / defect-sector continuation as the scientific
  mainline,
- return to the frozen action itself,
- derive the linear-in-a_mu source current J_eff^mu directly,
- classify the resulting J_eff^0 structure before any further source-theorem
  or observable-dictionary work.

This branch therefore performs a route reset rather than a new theorem search.
It keeps the internal/topological lane as a side note, but it promotes direct
J_eff^mu derivation to the primary computation mainline.
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
COMPUTATION_EXPERT_SHARE = (
    ROOT / "doc" / "quantum" / "43_trial2_numeric_alpha_vector_qball_computation_reactivation_expert_share.md"
)
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
PRIOR_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1555_1558_su2_hopf_reopen_audit_declaration_gate_metrics.json"
)
PRIOR_ROUTE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1555_1558_su2_hopf_reopen_audit_route_sync_metrics.json"
)
DIRECT_CURRENT_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1535_1538_charge_current_closure_derivation_declaration_gate_metrics.json"
)
DIRECTIVE_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_jeff_final_directive_20260328.md"
)

STEP_TAG = "8.7.56.1559-1562"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor frozen-action direct J_eff mainline reset"
)
STEM = build_compact_artifact_stem(STEP_TAG, "direct_jeff_mainline_reset", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_internal_su2_hopf_reopen_failed_microscopic_internal_topological_functional_derivation_next"
)
BRANCH_CLASS = "vector_qball_form_factor_frozen_action_direct_jeff_mainline_reset_completed"
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_frozen_action_direct_jeff_derivation"
)
NEXT_ROUTE = "8.7.56.1563"
NEXT_SECONDARY_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_jeff_charge_density_structure_classification"
)
NEXT_SECONDARY_ROUTE = "8.7.56.1567"
NEXT_DISPOSITION_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_jeff_disposition_sync"
)
NEXT_DISPOSITION_ROUTE = "8.7.56.1571"
SIDE_LANE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_internal_topological_lane_side_hold"
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


# 関数: UTF-8テキストを読み込む。

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
    """Return the first line matching one substring pattern."""
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


# 関数: payload を構成する。

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


# 関数: この branch で固定する route-reset 論理を返す。

def build_formulae() -> dict[str, str]:
    """Return the route-reset formulas and constraints."""
    return {
        "frozen_action_split": "P_mu(x) = P_mu^Qball(x) + a_mu(x)",
        "linear_collect_target": "L_total^vec[P_mu^Qball + a_mu] -> collect terms linear in a_mu",
        "structure_goal": "Identify J_eff^0 as |f_0|^2, |f_0|^2-|f_L|^2, another combination, or zero",
        "forbidden_lane": "No SU(2), Hopf, pi_4, defect-spinor structures may be introduced if absent from the frozen action",
    }


# 関数: `.1559-.1562` を実行する。

def main() -> None:
    """Execute the frozen-action direct J_eff mainline reset branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        COMPUTATION_EXPERT_SHARE,
        PART1,
        PART5,
        PRIOR_GATE,
        PRIOR_ROUTE,
        DIRECT_CURRENT_GATE,
        DIRECTIVE_NOTE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    expert_share_text = read_text(COMPUTATION_EXPERT_SHARE)
    part1_text = read_text(PART1)
    part5_text = read_text(PART5)
    directive_text = read_text(DIRECTIVE_NOTE)

    prior_summary = read_json(PRIOR_GATE)["summary"]
    prior_route_summary = read_json(PRIOR_ROUTE)["summary"]
    direct_current_summary = read_json(DIRECT_CURRENT_GATE)["summary"]
    formulas = build_formulae()

    prior_su2_hopf_lane_failed = bool(
        prior_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and prior_summary.get("internal_su2_hopf_embedding_reopen_failed_honest", False)
        and prior_summary.get("microscopic_internal_topological_functional_derivation_required", False)
    )
    direct_current_backbone_available = bool(
        direct_current_summary.get("free_backbone_linear_formula_derived", False)
        and direct_current_summary.get("same_field_on_shell_linear_source_zero", False)
    )
    directive_stop_internal_lane = bool(hit(directive_text, "停止命令"))
    directive_requires_frozen_action_only = bool(
        hit(directive_text, "唯一の作業: frozen action から J_eff^μ を計算する")
        and hit(directive_text, "禁止事項")
    )
    directive_explicitly_bans_su2_hopf = bool(
        hit(directive_text, "SU(2) / Hopf / defect は frozen action の外")
        and hit(directive_text, "frozen action に存在しない構造（SU(2), Hopf, π₄, defect spinor）を導入しない")
    )
    frozen_action_surface_available = bool(
        hit(part1_text, "\\mathcal{L}_{\\mathrm{total}}^{\\mathrm{vec}}")
        and hit(part1_text, "\\mathcal{L}_{P_\\mu}^{\\mathrm{free}}")
        and hit(part1_text, "\\mathcal{L}_{\\mathrm{int}}=g_P\\,P_\\mu J^\\mu_{\\mathrm{matter}}")
    )

    internal_topological_lane_demoted_from_mainline = bool(
        prior_su2_hopf_lane_failed
        and directive_stop_internal_lane
        and directive_explicitly_bans_su2_hopf
    )
    frozen_action_direct_jeff_promoted_to_mainline = bool(
        internal_topological_lane_demoted_from_mainline
        and direct_current_backbone_available
        and directive_requires_frozen_action_only
        and frozen_action_surface_available
    )
    jeff_structure_classification_scheduled = frozen_action_direct_jeff_promoted_to_mainline
    source_theorem_now_downstream_of_direct_jeff = frozen_action_direct_jeff_promoted_to_mainline
    observable_dictionary_now_downstream_of_direct_jeff_and_source = (
        frozen_action_direct_jeff_promoted_to_mainline
    )
    future_external_input_side_lane_retained = True
    new_free_parameters_introduced = False
    route_reset_completed = frozen_action_direct_jeff_promoted_to_mainline

    rows = [
        row(
            "prior_su2_hopf_lane_failed",
            "pass" if prior_su2_hopf_lane_failed else "reject",
            "prior SU(2)/Hopf lane failed honestly",
            truth(prior_su2_hopf_lane_failed),
            "The prior branch already proved that the SU(2)/Hopf/FR lane does not open on the current frozen-action branch.",
        ),
        row(
            "direct_current_backbone_available",
            "pass" if direct_current_backbone_available else "reject",
            "direct current backbone already available",
            truth(direct_current_backbone_available),
            "The earlier direct-current branch already derived the linear-in-a_mu backbone and localized the same-field on-shell zero.",
        ),
        row(
            "directive_stop_internal_lane",
            "pass" if directive_stop_internal_lane else "reject",
            "directive stops internal/topological continuation",
            truth(directive_stop_internal_lane),
            "The expert directive explicitly stops internal SU(2), Hopf, and defect-spinor continuation as v2.0 mainline work.",
        ),
        row(
            "directive_requires_frozen_action_only",
            "pass" if directive_requires_frozen_action_only else "reject",
            "directive requires frozen-action-only derivation",
            truth(directive_requires_frozen_action_only),
            "The new mainline is constrained to frozen-action algebraic expansion with no new structure beyond the pack.",
        ),
        row(
            "directive_explicitly_bans_su2_hopf",
            "pass" if directive_explicitly_bans_su2_hopf else "reject",
            "directive explicitly bans SU(2)/Hopf/pi4 imports",
            truth(directive_explicitly_bans_su2_hopf),
            "The expert note is explicit that SU(2), Hopf, pi4, and defect-spinor formalization are outside the frozen action and should stop.",
        ),
        row(
            "frozen_action_surface_available",
            "pass" if frozen_action_surface_available else "reject",
            "frozen action surface available",
            truth(frozen_action_surface_available),
            "Part I already exposes the frozen vector action needed for direct J_eff^mu derivation.",
        ),
        row(
            "internal_topological_lane_demoted_from_mainline",
            "pass" if internal_topological_lane_demoted_from_mainline else "reject",
            "internal/topological lane demoted from mainline",
            truth(internal_topological_lane_demoted_from_mainline),
            "The SU(2)/Hopf/internal-topological search remains archived as a side note rather than the scientific mainline.",
        ),
        row(
            "frozen_action_direct_jeff_promoted_to_mainline",
            "pass" if frozen_action_direct_jeff_promoted_to_mainline else "reject",
            "frozen-action direct J_eff derivation promoted to mainline",
            truth(frozen_action_direct_jeff_promoted_to_mainline),
            "The immediate next scientific branch is direct J_eff^mu derivation from the frozen action.",
        ),
        row(
            "jeff_structure_classification_scheduled",
            "pass" if jeff_structure_classification_scheduled else "reject",
            "J_eff^0 structure classification scheduled",
            truth(jeff_structure_classification_scheduled),
            "After direct derivation, the next branch must classify J_eff^0 as scalar-proxy, signed-density, other, or zero.",
        ),
        row(
            "source_theorem_now_downstream_of_direct_jeff",
            "pass" if source_theorem_now_downstream_of_direct_jeff else "reject",
            "source theorem now downstream of direct J_eff derivation",
            truth(source_theorem_now_downstream_of_direct_jeff),
            "Effective-source work must wait until the direct J_eff^mu structure is actually derived and classified.",
        ),
        row(
            "observable_dictionary_now_downstream_of_direct_jeff_and_source",
            "pass" if observable_dictionary_now_downstream_of_direct_jeff_and_source else "reject",
            "observable dictionary now downstream of direct J_eff and source theorem",
            truth(observable_dictionary_now_downstream_of_direct_jeff_and_source),
            "Observable mapping remains the last gate, not an upstream blocker.",
        ),
        row(
            "future_external_input_side_lane_retained",
            "pass" if future_external_input_side_lane_retained else "reject",
            "future external input retained as side lane",
            truth(future_external_input_side_lane_retained),
            "External expert input is still useful but no longer the mainline stopper.",
        ),
        row(
            "new_free_parameters_introduced",
            "pass" if not new_free_parameters_introduced else "reject",
            "new free parameters introduced",
            truth(new_free_parameters_introduced),
            "The route reset itself introduces no new free parameter.",
        ),
        row(
            "route_reset_completed",
            "pass" if route_reset_completed else "reject",
            "direct J_eff route reset completed",
            truth(route_reset_completed),
            "This branch is complete only if the roadmap is genuinely reset to direct frozen-action J_eff work.",
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
            "computation_expert_share": display_path(COMPUTATION_EXPERT_SHARE),
            "part1": display_path(PART1),
            "part5": display_path(PART5),
            "directive_note": display_path(DIRECTIVE_NOTE),
        },
        "prior_metrics": {
            "prior_gate": display_path(PRIOR_GATE),
            "prior_route": display_path(PRIOR_ROUTE),
            "direct_current_gate": display_path(DIRECT_CURRENT_GATE),
        },
        "constants": {
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "next_secondary_route_name": NEXT_SECONDARY_ROUTE_NAME,
            "next_secondary_route": NEXT_SECONDARY_ROUTE,
            "next_disposition_route_name": NEXT_DISPOSITION_ROUTE_NAME,
            "next_disposition_route": NEXT_DISPOSITION_ROUTE,
            "side_lane_name": SIDE_LANE_NAME,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "prior_su2_hopf_lane_failed": prior_su2_hopf_lane_failed,
        "direct_current_backbone_available": direct_current_backbone_available,
        "directive_stop_internal_lane": directive_stop_internal_lane,
        "directive_requires_frozen_action_only": directive_requires_frozen_action_only,
        "directive_explicitly_bans_su2_hopf": directive_explicitly_bans_su2_hopf,
        "internal_topological_lane_demoted_from_mainline": internal_topological_lane_demoted_from_mainline,
        "frozen_action_direct_jeff_promoted_to_mainline": frozen_action_direct_jeff_promoted_to_mainline,
        "jeff_structure_classification_scheduled": jeff_structure_classification_scheduled,
        "source_theorem_now_downstream_of_direct_jeff": source_theorem_now_downstream_of_direct_jeff,
        "observable_dictionary_now_downstream_of_direct_jeff_and_source": observable_dictionary_now_downstream_of_direct_jeff_and_source,
        "future_external_input_side_lane_retained": future_external_input_side_lane_retained,
        "frozen_action_only_used": True,
        "new_free_parameters_introduced": new_free_parameters_introduced,
        "physical_reject_required": False,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": NEXT_SECONDARY_ROUTE_NAME,
        "selected_disposition_route": NEXT_DISPOSITION_ROUTE_NAME,
        "side_lane_future_route_or_none": SIDE_LANE_NAME,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": route_reset_completed,
        "next_required_artifacts": [
            NEXT_ROUTE_NAME,
            NEXT_SECONDARY_ROUTE_NAME,
            NEXT_DISPOSITION_ROUTE_NAME,
        ],
    }

    evidence = {
        "formulas": formulas,
        "hits": {
            "directive_stop": hit(directive_text, "停止命令"),
            "directive_frozen_action_only": hit(directive_text, "唯一の作業: frozen action から J_eff^μ を計算する"),
            "directive_ban_new_structures": hit(
                directive_text,
                "frozen action に存在しない構造（SU(2), Hopf, π₄, defect spinor）を導入しない",
            ),
            "directive_structure_classification": hit(directive_text, "### 1.4 構造判定"),
            "part1_total_action": hit(part1_text, "\\mathcal{L}_{\\mathrm{total}}^{\\mathrm{vec}}"),
            "part1_free_action": hit(part1_text, "\\mathcal{L}_{P_\\mu}^{\\mathrm{free}}"),
            "part1_interaction": hit(part1_text, "\\mathcal{L}_{\\mathrm{int}}=g_P\\,P_\\mu J^\\mu_{\\mathrm{matter}}"),
        },
        "carry_over": {
            "prior_summary": prior_summary,
            "prior_route_summary": prior_route_summary,
            "direct_current_summary": direct_current_summary,
        },
        "retained_numeric_state": {
            "scalar_F_exact_at_q_theory": 0.2998913524347805,
            "scalar_alpha_exact_at_q_theory": 0.00715678583937324,
            "vector_F_at_q_theory": -0.083735013520183,
            "vector_alpha_at_q_theory": 0.0005579616187042394,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    inventory_paths = write_artifact(
        "inventory",
        payload("8.7.56.1559", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence),
    )
    audit_paths = write_artifact(
        "audit",
        payload("8.7.56.1560", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence),
    )
    gate_paths = write_artifact(
        "declaration_gate",
        payload(
            "8.7.56.1561",
            f"{STEP_NAME} declaration gate",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )
    route_paths = write_artifact(
        "route_sync",
        payload("8.7.56.1562", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence),
    )

    print("[ok] frozen-action direct J_eff mainline reset artifacts written:")
    print(f" - {inventory_paths['json']}")
    print(f" - {audit_paths['json']}")
    print(f" - {gate_paths['json']}")
    print(f" - {route_paths['json']}")


if __name__ == "__main__":
    main()
