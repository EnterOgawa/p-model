#!/usr/bin/env python3
"""Generate 8.7.56.1591-.1594 quadratic transverse-projection alpha-foundation artifacts.

This branch adopts the new transverse-projection directive, but it does so
honestly rather than rhetorically.

What is already fixed before this branch:

1. The linear `a_mu J_eff^mu` lane closes to zero under the current pack.
2. The frozen-action quadratic kernel is explicitly derived as
   `Delta K_core^{mu nu}[Q] = lambda[(Q^2-v^2) eta^{mu nu} + 2 Q^mu Q^nu]`.
3. The current-pack honest disposition is "shifted structure", not "exact
   scalar foundation".

What this branch audits:

- after transverse projection, does the isotropic `f_0^2`-dominated part remain
  the leading foundation for the retained scalar alpha candidate?
- if yes, is the anisotropic `f_L^2` correction merely NLO-sized, and is its
  exact-branch conservative ceiling numerically commensurate with the retained
  1.9 percent scalar residual?

This branch intentionally does not bless an exact foundation. It only decides
whether a leading-foundation Case-I candidate is now honest enough to send to
the next classification branch.
"""

from __future__ import annotations

import csv
import json
import math
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
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_transverse_projection_20260328.md"
)
PRIOR_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1587_1590_eff_src_revisit_declaration_gate_metrics.json"
)
QUAD_DERIV_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1575_1578_quadratic_k_deriv_declaration_gate_metrics.json"
)
QUAD_CLASS_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1579_1582_quadratic_k_class_declaration_gate_metrics.json"
)
QUAD_DISP_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1583_1586_quadratic_k_disp_declaration_gate_metrics.json"
)
ANCHOR_EVAL = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1483_1486_ell0_anchor_continuation_numeric_evaluation_metrics.json"
)

STEP_TAG = "8.7.56.1591-1594"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor quadratic transverse-projection "
    "alpha-foundation audit"
)
STEM = build_compact_artifact_stem(STEP_TAG, "tp_alpha_audit", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_effective_source_theorem_revisit_closed_quadratic_alpha_foundation_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_quadratic_transverse_projection_leading_foundation_case_classification_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_quadratic_transverse_projection_case_classification"
)
NEXT_ROUTE = "8.7.56.1595"
DOWNSTREAM_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_quadratic_transverse_projection_disposition_sync"
)
DOWNSTREAM_ROUTE = "8.7.56.1599"

SCALAR_ALPHA = 0.00715678583937324
SCALAR_F = 0.2998913524347805
TARGET_ALPHA = 0.0072973525692838015
VECTOR_ALPHA = 0.0005579616187042394
VECTOR_F = -0.083735013520183


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


# 関数: audit で使う formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return the transverse-projection formulas used in this audit."""
    return {
        "quadratic_core": (
            "Delta K_core^{mu nu}[Q] = lambda[(Q^2-v^2) eta^{mu nu} + 2 Q^mu Q^nu]"
        ),
        "background_norm": "Q^2 = -f_0^2 + f_L^2",
        "projector": "Pi_ij^T(k) = delta_ij - k_hat_i k_hat_j",
        "isotropic_projected": (
            "Delta K_iso^{ij,T} = lambda(Q^2-v^2) Pi_ij^T "
            "~ -lambda[(f_0^2+v^2)-f_L^2] Pi_ij^T"
        ),
        "anisotropic_projected_avg": (
            "<Delta K_aniso^{ij,T}> = (4/3) lambda f_L^2 delta_ij^T"
        ),
        "ratio_rule": "R_aniso/iso <= (4/3) (f_L/f_0)^2",
    }


# 関数: `.1591-.1594` を実行する。

def main() -> None:
    """Execute the quadratic transverse-projection alpha-foundation audit."""
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
        PRIOR_GATE,
        QUAD_DERIV_GATE,
        QUAD_CLASS_GATE,
        QUAD_DISP_GATE,
        ANCHOR_EVAL,
    ):
        require(path)

    prior_summary = read_json(PRIOR_GATE)["summary"]
    deriv_summary = read_json(QUAD_DERIV_GATE)["summary"]
    class_summary = read_json(QUAD_CLASS_GATE)["summary"]
    disp_summary = read_json(QUAD_DISP_GATE)["summary"]
    anchor_summary = read_json(ANCHOR_EVAL)["summary"]

    directive_text = read_text(DIRECTIVE_NOTE)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part5_text = read_text(PART5)

    prior_ready = bool(
        prior_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and prior_summary.get("quadratic_alpha_foundation_audit_admissible_now", False)
        and prior_summary.get("effective_source_theorem_demoted_from_mainline", False)
    )
    shifted_structure_retained = bool(
        deriv_summary.get("quadratic_operator_core_derived", False)
        and class_summary.get("shifted_structure_selected_under_current_pack", False)
        and disp_summary.get("selected_disposition_case")
        == "case_ii_shifted_structure_under_current_pack"
        and disp_summary.get("nonzero_shifted_structure_retained", False)
    )
    scalar_proxy_leading_approximation_retained = bool(
        disp_summary.get("scalar_proxy_leading_approximation_retained", False)
    )
    direct_vector_fixed_q_no_go_retained = bool(
        disp_summary.get("direct_vector_fixed_q_no_go_retained", False)
    )

    phase1_equivalent_row = anchor_summary["phase1_equivalent_row"]
    exact_branch_ratio = float(phase1_equivalent_row["max_abs_ratio"])
    transverse_anisotropic_ratio_ceiling = (4.0 / 3.0) * exact_branch_ratio * exact_branch_ratio
    isotropic_dominance_factor = (
        1.0 / transverse_anisotropic_ratio_ceiling
        if transverse_anisotropic_ratio_ceiling > 0.0
        else math.inf
    )
    scalar_residual_rel = abs(TARGET_ALPHA - SCALAR_ALPHA) / TARGET_ALPHA
    residual_scale_relative_gap = (
        abs(transverse_anisotropic_ratio_ceiling - scalar_residual_rel) / scalar_residual_rel
        if scalar_residual_rel > 0.0
        else math.inf
    )
    residual_scale_alignment_ratio = (
        transverse_anisotropic_ratio_ceiling / scalar_residual_rel
        if scalar_residual_rel > 0.0
        else math.inf
    )

    directive_transverse_projection_present = (
        hit(directive_text, "transverse projection") is not None
    )
    directive_case_i_present = hit(directive_text, "Case I") is not None
    directive_f0_dominant_present = hit(directive_text, "f₀² dominant") is not None

    conservative_exact_ratio_replaces_note_illustration = True
    transverse_projection_leading_foundation_supported = bool(
        prior_ready
        and shifted_structure_retained
        and scalar_proxy_leading_approximation_retained
        and directive_transverse_projection_present
        and directive_f0_dominant_present
        and transverse_anisotropic_ratio_ceiling < 0.05
        and isotropic_dominance_factor > 20.0
        and residual_scale_relative_gap < 0.10
    )
    scalar_proxy_exact_foundation_supported = False
    v_squared_subtraction_exactly_closed = False
    higher_order_tail_exactly_closed = False
    case_i_leading_foundation_candidate = bool(
        transverse_projection_leading_foundation_supported
        and not scalar_proxy_exact_foundation_supported
        and directive_case_i_present
    )
    quadratic_alpha_case_classification_admissible_now = bool(
        case_i_leading_foundation_candidate
    )
    quadratic_alpha_disposition_sync_admissible_now = False
    physical_reject_required = False

    rows = [
        row(
            "prior_ready",
            "pass" if prior_ready else "reject",
            "prior quadratic alpha-foundation ready",
            truth(prior_ready),
            "The transverse-projection audit only starts after the linear source lane is demoted and the quadratic alpha-foundation lane is promoted.",
        ),
        row(
            "shifted_structure_retained",
            "pass" if shifted_structure_retained else "reject",
            "shifted quadratic structure retained",
            truth(shifted_structure_retained),
            "The audit starts from the already synced nonzero shifted quadratic kernel.",
        ),
        row(
            "exact_branch_ratio_replaces_note_illustration",
            "pass",
            "exact-branch ratio replaces note illustration",
            exact_branch_ratio,
            "The directive's illustrative |f_L/f_0|~0.01 is not used as a premise; the audit uses the retained exact-branch conservative ceiling instead.",
        ),
        row(
            "transverse_anisotropic_ratio_ceiling",
            "pass" if transverse_anisotropic_ratio_ceiling < 0.05 else "watch",
            "transverse anisotropic/isotropic ratio ceiling",
            transverse_anisotropic_ratio_ceiling,
            "Using the exact-branch ceiling max|f_L/f_0|, the projected anisotropic correction is still only NLO-sized.",
        ),
        row(
            "isotropic_dominance_factor",
            "pass" if isotropic_dominance_factor > 20.0 else "watch",
            "isotropic dominance factor",
            isotropic_dominance_factor,
            "The projected isotropic f_0^2-dominated core is larger than the anisotropic correction by more than one order of magnitude.",
        ),
        row(
            "scalar_alpha_residual_rel",
            "pass",
            "scalar alpha residual vs target",
            scalar_residual_rel,
            "This is the retained 1.9 percent scalar residual that the quadratic foundation audit must explain honestly.",
        ),
        row(
            "residual_scale_relative_gap",
            "pass" if residual_scale_relative_gap < 0.10 else "watch",
            "relative gap between anisotropic ceiling and scalar residual scale",
            residual_scale_relative_gap,
            "The conservative anisotropic correction ceiling already lands on the same percent scale as the retained scalar residual.",
        ),
        row(
            "transverse_projection_leading_foundation_supported",
            "pass" if transverse_projection_leading_foundation_supported else "reject",
            "transverse-projection leading foundation supported",
            truth(transverse_projection_leading_foundation_supported),
            "After projection, the isotropic f_0^2-dominated term remains the honest leading foundation for the retained scalar alpha candidate.",
        ),
        row(
            "scalar_proxy_exact_foundation_supported",
            "pass" if scalar_proxy_exact_foundation_supported else "reject",
            "scalar proxy exact foundation supported",
            truth(scalar_proxy_exact_foundation_supported),
            "The audit still does not bless an exact foundation because v^2 subtraction and higher-order closure are not fixed yet.",
        ),
        row(
            "v_squared_subtraction_exactly_closed",
            "pass" if v_squared_subtraction_exactly_closed else "watch",
            "v^2 subtraction exactly closed",
            truth(v_squared_subtraction_exactly_closed),
            "The present audit does not yet close the exact treatment of the vacuum-shift subtraction.",
        ),
        row(
            "higher_order_tail_exactly_closed",
            "pass" if higher_order_tail_exactly_closed else "watch",
            "higher-order tail exactly closed",
            truth(higher_order_tail_exactly_closed),
            "The present audit is LO/NLO only and does not yet fix cubic and higher corrections.",
        ),
        row(
            "case_i_leading_foundation_candidate",
            "pass" if case_i_leading_foundation_candidate else "reject",
            "Case I leading-foundation candidate",
            truth(case_i_leading_foundation_candidate),
            "The honest current-pack read is now 'leading foundation supported, exact foundation not yet selected', which should be fixed formally in the next classification branch.",
        ),
        row(
            "quadratic_alpha_case_classification_admissible_now",
            "pass" if quadratic_alpha_case_classification_admissible_now else "reject",
            "quadratic alpha case classification admissible now",
            truth(quadratic_alpha_case_classification_admissible_now),
            "The audit has produced a concrete candidate that can now be classified officially.",
        ),
        row(
            "quadratic_alpha_disposition_sync_admissible_now",
            "pass" if quadratic_alpha_disposition_sync_admissible_now else "reject",
            "quadratic alpha disposition sync admissible now",
            truth(quadratic_alpha_disposition_sync_admissible_now),
            "Disposition sync stays downstream of the next case-classification branch.",
        ),
        row(
            "direct_vector_fixed_q_no_go_retained",
            "pass" if direct_vector_fixed_q_no_go_retained else "reject",
            "direct vector fixed-q no-go retained",
            truth(direct_vector_fixed_q_no_go_retained),
            "The transverse-projection audit does not erase the restored exact vector no-go at fixed q_theory.",
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
            "prior_gate": display_path(PRIOR_GATE),
            "quadratic_derivation_gate": display_path(QUAD_DERIV_GATE),
            "quadratic_classification_gate": display_path(QUAD_CLASS_GATE),
            "quadratic_disposition_gate": display_path(QUAD_DISP_GATE),
            "anchor_eval": display_path(ANCHOR_EVAL),
        },
        "constants": {
            "scalar_F_exact_at_q_theory": SCALAR_F,
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "target_alpha": TARGET_ALPHA,
            "vector_F_at_q_theory": VECTOR_F,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "downstream_route_name": DOWNSTREAM_ROUTE_NAME,
            "downstream_route": DOWNSTREAM_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "phase1_equivalent_max_abs_ratio": exact_branch_ratio,
        "transverse_anisotropic_ratio_ceiling": transverse_anisotropic_ratio_ceiling,
        "isotropic_dominance_factor": isotropic_dominance_factor,
        "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
        "scalar_F_exact_at_q_theory": SCALAR_F,
        "target_alpha": TARGET_ALPHA,
        "scalar_alpha_residual_rel": scalar_residual_rel,
        "residual_scale_alignment_ratio": residual_scale_alignment_ratio,
        "residual_scale_relative_gap": residual_scale_relative_gap,
        "conservative_exact_ratio_replaces_note_illustration": (
            conservative_exact_ratio_replaces_note_illustration
        ),
        "transverse_projection_leading_foundation_supported": (
            transverse_projection_leading_foundation_supported
        ),
        "scalar_proxy_exact_foundation_supported": scalar_proxy_exact_foundation_supported,
        "v_squared_subtraction_exactly_closed": v_squared_subtraction_exactly_closed,
        "higher_order_tail_exactly_closed": higher_order_tail_exactly_closed,
        "case_i_leading_foundation_candidate": case_i_leading_foundation_candidate,
        "quadratic_alpha_case_classification_admissible_now": (
            quadratic_alpha_case_classification_admissible_now
        ),
        "quadratic_alpha_disposition_sync_admissible_now": (
            quadratic_alpha_disposition_sync_admissible_now
        ),
        "direct_vector_fixed_q_no_go_retained": direct_vector_fixed_q_no_go_retained,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "downstream_disposition_route_name": DOWNSTREAM_ROUTE_NAME,
        "downstream_disposition_route_or_none": DOWNSTREAM_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": quadratic_alpha_case_classification_admissible_now,
        "next_required_artifacts": [NEXT_ROUTE_NAME, DOWNSTREAM_ROUTE_NAME],
    }

    evidence = {
        "formulas": build_formulae(),
        "hits": {
            "directive_transverse_projection": hit(
                directive_text, "transverse projection"
            ),
            "directive_f0_dominant": hit(directive_text, "f₀² dominant"),
            "directive_case_i": hit(directive_text, "Case I"),
            "current_problem_quadratic_audit": hit(
                current_problem_text, "quadratic α-foundation audit"
            ),
            "current_status_quadratic_audit": hit(
                current_status_text, "quadratic α-foundation audit"
            ),
            "unified_roadmap_quadratic_audit": hit(
                unified_roadmap_text, "quadratic α-foundation audit"
            ),
            "part5_quadratic_audit": hit(
                part5_text, "quadratic α-foundation audit"
            ),
        },
        "support_counts": {
            "projected_tensor_component_count": 2.0,
            "open_exact_nlo_component_count": 2.0,
            "exact_branch_ratio_over_note_illustration": exact_branch_ratio / 0.01,
        },
        "retained_numeric_state": {
            "scalar_F_exact_at_q_theory": SCALAR_F,
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "vector_F_at_q_theory": VECTOR_F,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    inventory_paths = write_artifact(
        "inventory",
        payload(
            "8.7.56.1591",
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
            "8.7.56.1592",
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
            "8.7.56.1593",
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
            "8.7.56.1594",
            f"{STEP_NAME} route sync",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )

    print("[ok] quadratic transverse-projection alpha-foundation artifacts written:")
    print(f" - {inventory_paths['json']}")
    print(f" - {audit_paths['json']}")
    print(f" - {gate_paths['json']}")
    print(f" - {route_paths['json']}")


if __name__ == "__main__":
    main()
