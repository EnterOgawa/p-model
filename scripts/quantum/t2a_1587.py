#!/usr/bin/env python3
"""Generate 8.7.56.1587-.1590 effective-source-theorem revisit artifacts.

The prior quadratic branches changed the scientific situation materially:

1. The frozen-action linear source lane is already classified as
   zero under the current pack.
2. The frozen-action quadratic operator has now been derived explicitly and
   synced honestly as a nonzero shifted-structure.

This branch therefore revisits the old source-theorem question after the
quadratic disposition. The key question is no longer "can the linear theorem be
reopened by rhetoric?" but rather:

- does the nonzero quadratic operator reopen an exact linear
  `a_mu J_eff^mu[Q]` theorem, or
- does it instead demote the linear-source lane and force the next mainline to
  audit the quadratic alpha-foundation directly?

Under the current pack, the honest answer is the second one.
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
DIRECTIVE_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_quadratic_expansion_20260328.md"
)

PRIOR_LINEAR_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1567_1570_jeff_q0_class_declaration_gate_metrics.json"
)
PRIOR_DERIV_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1575_1578_quadratic_k_deriv_declaration_gate_metrics.json"
)
PRIOR_CLASS_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1579_1582_quadratic_k_class_declaration_gate_metrics.json"
)
PRIOR_DISP_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1583_1586_quadratic_k_disp_declaration_gate_metrics.json"
)
OLD_SOURCE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1487_1490_effective_source_theorem_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1587-1590"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor effective source theorem "
    "revisit after quadratic disposition"
)
STEM = build_compact_artifact_stem(STEP_TAG, "eff_src_revisit", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_quadratic_shifted_structure_disposition_sync_completed"
BRANCH_CLASS = (
    "vector_qball_form_factor_effective_source_theorem_revisit_closed_quadratic_alpha_foundation_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_quadratic_alpha_foundation_audit"
)
NEXT_ROUTE = "8.7.56.1591"
DOWNSTREAM_CLASS_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_quadratic_alpha_case_classification"
)
DOWNSTREAM_CLASS_ROUTE = "8.7.56.1595"
DOWNSTREAM_DISP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_quadratic_alpha_disposition_sync"
)
DOWNSTREAM_DISP_ROUTE = "8.7.56.1599"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required path is missing."""
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


# 関数: metrics row を構成する。

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


# 関数: 今回の branch で使う formula summary を返す。

def build_formulae() -> dict[str, str]:
    """Return the linear/quadratic formulas used by the revisit."""
    return {
        "linear_collect": "L_total^vec|_(a^1) = a_mu J_eff^mu[Q]",
        "same_field_on_shell": "J_eff^mu[Q]_same-field,on-shell = 0",
        "quadratic_collect": "L_total^vec|_(a^2) = (1/2) a_mu K^{mu nu}[Q] a_nu",
        "quadratic_core": (
            "Delta K_core^{mu nu}[Q] = lambda[(Q^2-v^2) eta^{mu nu} + 2 Q^mu Q^nu]"
        ),
        "revisit_rule": (
            "Once the linear tadpole is fixed to zero while the quadratic background kernel "
            "is nonzero shifted-structure, the honest next foundation question is quadratic "
            "operator readout rather than exact linear source-theorem reopen."
        ),
    }


# 関数: `.1587-.1590` を実行する。

def main() -> None:
    """Execute the effective-source-theorem revisit after quadratic disposition."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PART5,
        DIRECTIVE_NOTE,
        PRIOR_LINEAR_GATE,
        PRIOR_DERIV_GATE,
        PRIOR_CLASS_GATE,
        PRIOR_DISP_GATE,
        OLD_SOURCE_GATE,
    ):
        require(path)

    linear_summary = read_json(PRIOR_LINEAR_GATE)["summary"]
    deriv_summary = read_json(PRIOR_DERIV_GATE)["summary"]
    class_summary = read_json(PRIOR_CLASS_GATE)["summary"]
    disp_summary = read_json(PRIOR_DISP_GATE)["summary"]
    old_source_summary = read_json(OLD_SOURCE_GATE)["summary"]

    directive_text = read_text(DIRECTIVE_NOTE)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part5_text = read_text(PART5)

    prior_ready = bool(
        disp_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and disp_summary.get("shifted_structure_disposition_sync_honest", False)
        and disp_summary.get("effective_source_theorem_revisit_admissible_now", False)
    )
    linear_zero_retained = bool(
        linear_summary.get("zero_structure_selected_under_current_pack", False)
        and linear_summary.get("classification_case_iv_zero_under_current_pack", False)
    )
    quadratic_shifted_structure_retained = bool(
        deriv_summary.get("quadratic_operator_core_derived", False)
        and class_summary.get("shifted_structure_selected_under_current_pack", False)
        and disp_summary.get("nonzero_shifted_structure_retained", False)
    )
    directive_quadratic_primary = (
        hit(directive_text, "一次が消えた。次は二次。") is not None
    )
    directive_vacuum_polarization_read = (
        hit(
            directive_text,
            "QED で α が出るのは tree-level tadpole からではなく、one-loop vacuum polarization から。",
        )
        is not None
    )
    directive_final_test_is_delta_k = (
        hit(
            directive_text,
            "ΔK が scalar proxy α = 0.00716 を reproduce するかどうかが最終判定。",
        )
        is not None
    )

    effective_source_theorem_revisit_ready = bool(
        prior_ready and linear_zero_retained and quadratic_shifted_structure_retained
    )
    exact_linear_source_theorem_reopened = False
    linear_source_tadpole_lane_exhausted_under_current_pack = bool(
        effective_source_theorem_revisit_ready and linear_zero_retained
    )
    quadratic_operator_is_not_linear_source_term = bool(
        quadratic_shifted_structure_retained and linear_zero_retained
    )
    effective_source_theorem_revisit_honest = bool(
        effective_source_theorem_revisit_ready
        and not exact_linear_source_theorem_reopened
        and quadratic_operator_is_not_linear_source_term
    )
    effective_source_theorem_demoted_from_mainline = bool(
        effective_source_theorem_revisit_honest
        and directive_quadratic_primary
        and directive_vacuum_polarization_read
        and directive_final_test_is_delta_k
    )
    quadratic_alpha_foundation_audit_admissible_now = bool(
        effective_source_theorem_demoted_from_mainline
    )
    observable_dictionary_revisit_admissible_now = False
    prior_exact_source_theorem_derived = bool(
        old_source_summary.get("exact_source_theorem_derived", False)
    )
    scalar_proxy_leading_approximation_retained = bool(
        disp_summary.get("scalar_proxy_leading_approximation_retained", False)
    )
    direct_vector_fixed_q_no_go_retained = bool(
        disp_summary.get("direct_vector_fixed_q_no_go_retained", False)
    )
    physical_reject_required = False

    rows = [
        row(
            "prior_quadratic_disposition_ready",
            "pass" if prior_ready else "reject",
            "prior quadratic disposition ready",
            truth(prior_ready),
            "The revisit only starts after shifted-structure disposition has been synced honestly.",
        ),
        row(
            "linear_zero_retained",
            "pass" if linear_zero_retained else "reject",
            "linear zero-current-pack retained",
            truth(linear_zero_retained),
            "The direct linear J_eff lane remains fixed to zero under the current pack.",
        ),
        row(
            "quadratic_shifted_structure_retained",
            "pass" if quadratic_shifted_structure_retained else "reject",
            "quadratic shifted-structure retained",
            truth(quadratic_shifted_structure_retained),
            "The frozen-action quadratic kernel remains explicitly nonzero and officially synced as shifted-structure.",
        ),
        row(
            "effective_source_theorem_revisit_ready",
            "pass" if effective_source_theorem_revisit_ready else "reject",
            "effective source theorem revisit ready",
            truth(effective_source_theorem_revisit_ready),
            "The revisit is honest only after both the linear zero and the quadratic shifted structure have been fixed.",
        ),
        row(
            "exact_linear_source_theorem_reopened",
            "pass" if exact_linear_source_theorem_reopened else "reject",
            "exact linear source theorem reopened",
            truth(exact_linear_source_theorem_reopened),
            "Quadratic shifted structure does not by itself produce a new nonzero linear a_mu J_eff^mu theorem.",
        ),
        row(
            "quadratic_operator_is_not_linear_source_term",
            "pass" if quadratic_operator_is_not_linear_source_term else "reject",
            "quadratic operator is not linear source term",
            truth(quadratic_operator_is_not_linear_source_term),
            "The retained quadratic object is (1/2) a_mu K^{mu nu}[Q] a_nu, not a reopened linear source functional.",
        ),
        row(
            "linear_source_tadpole_lane_exhausted_under_current_pack",
            "pass" if linear_source_tadpole_lane_exhausted_under_current_pack else "reject",
            "linear source/tadpole lane exhausted under current pack",
            truth(linear_source_tadpole_lane_exhausted_under_current_pack),
            "Once the linear lane is zero and the next surviving object is quadratic, repeating the old source-theorem ask becomes a dead loop.",
        ),
        row(
            "effective_source_theorem_revisit_honest",
            "pass" if effective_source_theorem_revisit_honest else "reject",
            "effective source theorem revisit honest",
            truth(effective_source_theorem_revisit_honest),
            "The honest revisit result is to retain the old exact-fail and redirect the mainline to the quadratic foundation question.",
        ),
        row(
            "effective_source_theorem_demoted_from_mainline",
            "pass" if effective_source_theorem_demoted_from_mainline else "reject",
            "effective source theorem demoted from mainline",
            truth(effective_source_theorem_demoted_from_mainline),
            "The quadratic directive makes Delta K, not linear source recovery, the next primary computation target.",
        ),
        row(
            "quadratic_alpha_foundation_audit_admissible_now",
            "pass" if quadratic_alpha_foundation_audit_admissible_now else "reject",
            "quadratic alpha foundation audit admissible now",
            truth(quadratic_alpha_foundation_audit_admissible_now),
            "The next honest branch is to ask how the shifted quadratic operator could underwrite the retained scalar alpha candidate.",
        ),
        row(
            "observable_dictionary_revisit_admissible_now",
            "pass" if observable_dictionary_revisit_admissible_now else "reject",
            "observable dictionary revisit admissible now",
            truth(observable_dictionary_revisit_admissible_now),
            "Observable dictionary remains downstream and is not reopened by the present revisit result.",
        ),
        row(
            "scalar_proxy_leading_approximation_retained",
            "pass" if scalar_proxy_leading_approximation_retained else "reject",
            "scalar proxy leading approximation retained",
            truth(scalar_proxy_leading_approximation_retained),
            "The retained scalar strong candidate still survives, but only as a leading approximation rather than exact source theorem output.",
        ),
        row(
            "direct_vector_fixed_q_no_go_retained",
            "pass" if direct_vector_fixed_q_no_go_retained else "reject",
            "direct vector fixed-q no-go retained",
            truth(direct_vector_fixed_q_no_go_retained),
            "The blind fixed-q vector no-go remains a hard carry-over constraint during the route reset.",
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
            "directive_note": display_path(DIRECTIVE_NOTE),
            "prior_linear_gate": display_path(PRIOR_LINEAR_GATE),
            "prior_derivation_gate": display_path(PRIOR_DERIV_GATE),
            "prior_classification_gate": display_path(PRIOR_CLASS_GATE),
            "prior_disposition_gate": display_path(PRIOR_DISP_GATE),
            "old_source_gate": display_path(OLD_SOURCE_GATE),
        },
        "constants": {
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "downstream_classification_route_name": DOWNSTREAM_CLASS_ROUTE_NAME,
            "downstream_classification_route": DOWNSTREAM_CLASS_ROUTE,
            "downstream_disposition_route_name": DOWNSTREAM_DISP_ROUTE_NAME,
            "downstream_disposition_route": DOWNSTREAM_DISP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "linear_zero_retained": linear_zero_retained,
        "quadratic_shifted_structure_retained": quadratic_shifted_structure_retained,
        "effective_source_theorem_revisit_ready": effective_source_theorem_revisit_ready,
        "exact_linear_source_theorem_reopened": exact_linear_source_theorem_reopened,
        "linear_source_tadpole_lane_exhausted_under_current_pack": (
            linear_source_tadpole_lane_exhausted_under_current_pack
        ),
        "quadratic_operator_is_not_linear_source_term": (
            quadratic_operator_is_not_linear_source_term
        ),
        "effective_source_theorem_revisit_honest": effective_source_theorem_revisit_honest,
        "effective_source_theorem_demoted_from_mainline": (
            effective_source_theorem_demoted_from_mainline
        ),
        "prior_exact_source_theorem_derived": prior_exact_source_theorem_derived,
        "quadratic_alpha_foundation_audit_admissible_now": (
            quadratic_alpha_foundation_audit_admissible_now
        ),
        "observable_dictionary_revisit_admissible_now": (
            observable_dictionary_revisit_admissible_now
        ),
        "scalar_proxy_leading_approximation_retained": (
            scalar_proxy_leading_approximation_retained
        ),
        "direct_vector_fixed_q_no_go_retained": direct_vector_fixed_q_no_go_retained,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "downstream_classification_route_name": DOWNSTREAM_CLASS_ROUTE_NAME,
        "downstream_classification_route_or_none": DOWNSTREAM_CLASS_ROUTE,
        "downstream_disposition_route_name": DOWNSTREAM_DISP_ROUTE_NAME,
        "downstream_disposition_route_or_none": DOWNSTREAM_DISP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": effective_source_theorem_revisit_honest,
        "next_required_artifacts": [
            NEXT_ROUTE_NAME,
            DOWNSTREAM_CLASS_ROUTE_NAME,
            DOWNSTREAM_DISP_ROUTE_NAME,
        ],
    }

    evidence = {
        "formulas": build_formulae(),
        "hits": {
            "directive_quadratic": hit(directive_text, "一次が消えた。次は二次。"),
            "directive_qed_vacuum_pol": hit(
                directive_text,
                "QED で α が出るのは tree-level tadpole からではなく、one-loop vacuum polarization から。",
            ),
            "directive_final_test": hit(
                directive_text,
                "ΔK が scalar proxy α = 0.00716 を reproduce するかどうかが最終判定。",
            ),
            "current_problem_shifted_structure": hit(
                current_problem_text, "shifted-structure"
            ),
            "current_problem_next_branch": hit(
                current_problem_text,
                "effective source theorem revisit after quadratic disposition",
            ),
            "current_status_next_branch": hit(
                current_status_text,
                "effective source theorem revisit after quadratic disposition",
            ),
            "unified_roadmap_next_branch": hit(
                unified_roadmap_text,
                "effective source theorem revisit after quadratic disposition",
            ),
            "part5_next_branch": hit(
                part5_text,
                "effective source theorem revisit after quadratic disposition",
            ),
        },
        "support_counts": {
            "linear_zero_support_count": 1.0,
            "quadratic_shift_component_count": 2.0,
            "exact_linear_source_reopen_support_count": 0.0,
            "quadratic_foundation_followup_count": 3.0,
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
        payload(
            "8.7.56.1587",
            f"{STEP_NAME} inventory",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )
    audit_paths = write_artifact(
        "audit",
        payload(
            "8.7.56.1588",
            f"{STEP_NAME} audit",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )
    gate_paths = write_artifact(
        "declaration_gate",
        payload(
            "8.7.56.1589",
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
        payload(
            "8.7.56.1590",
            f"{STEP_NAME} route sync",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )

    print("[ok] effective-source-theorem revisit artifacts written:")
    print(f" - {inventory_paths['json']}")
    print(f" - {audit_paths['json']}")
    print(f" - {gate_paths['json']}")
    print(f" - {route_paths['json']}")


if __name__ == "__main__":
    main()
