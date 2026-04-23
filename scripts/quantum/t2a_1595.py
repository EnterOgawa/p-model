#!/usr/bin/env python3
"""Generate 8.7.56.1595-.1598 transverse-projection Case I classification artifacts.

This branch converts the completed transverse-projection audit into an official
classification result.

What is already fixed before this branch:

1. The linear `J_eff` lane closes to zero under the current pack.
2. The frozen-action quadratic kernel survives as a nonzero shifted structure.
3. The transverse-projection audit supports the scalar proxy only as a leading
   foundation candidate, not as an exact foundation.

What this branch must decide:

- is the honest official read still Case II "shifted structure retain", or can
  the current pack now fix Case I "leading foundation candidate"?
- if Case I candidate is fixed, should the next mainline move immediately to
  the exact `v^2` subtraction treatment rather than generic disposition wording?

This branch does not bless an exact foundation. It only fixes the candidate
classification and promotes the same-level `v^2` subtraction computation.
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

CLASSIFICATION_DIRECTIVE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_case_i_v2_subtraction_20260328.md"
)
PRIOR_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1591_1594_tp_alpha_audit_declaration_gate_metrics.json"
)
ZERO_CLASS_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1567_1570_jeff_q0_class_declaration_gate_metrics.json"
)
SHIFTED_DISP_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1583_1586_quadratic_k_disp_declaration_gate_metrics.json"
)
MAINLINE_RESET_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1559_1562_direct_jeff_mainline_reset_declaration_gate_metrics.json"
)
PERTURBATIVE_GAMMA_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_perturbative_fl_driven_ode_diagnostic_reopen_review_declaration_gate_metrics.json"
)
PHASE1_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase1_exact_coupled_l0_solver_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1595-1598"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor quadratic transverse-projection "
    "alpha case classification"
)
STEM = build_compact_artifact_stem(STEP_TAG, "tp_alpha_case_class", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_quadratic_transverse_projection_leading_foundation_case_classification_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_case_i_leading_foundation_candidate_v2_subtraction_exact_treatment_next"
)
SELECTED_CASE = "case_i_leading_foundation_candidate_under_current_pack"
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_v2_subtraction_exact_treatment_audit"
)
NEXT_ROUTE = "8.7.56.1599"
DOWNSTREAM_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_quadratic_transverse_projection_disposition_sync"
)
DOWNSTREAM_ROUTE = "8.7.56.1603"


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


# 関数: `.1595-.1598` を実行する。

def main() -> None:
    """Execute the transverse-projection Case I classification branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PART5,
        CLASSIFICATION_DIRECTIVE,
        PRIOR_GATE,
        ZERO_CLASS_GATE,
        SHIFTED_DISP_GATE,
        MAINLINE_RESET_GATE,
        PERTURBATIVE_GAMMA_GATE,
        PHASE1_GATE,
    ):
        require(path)

    prior_summary = read_json(PRIOR_GATE)["summary"]
    zero_summary = read_json(ZERO_CLASS_GATE)["summary"]
    shifted_summary = read_json(SHIFTED_DISP_GATE)["summary"]
    reset_summary = read_json(MAINLINE_RESET_GATE)["summary"]
    gamma_summary = read_json(PERTURBATIVE_GAMMA_GATE)["summary"]
    phase1_summary = read_json(PHASE1_GATE)["summary"]

    directive_text = read_text(CLASSIFICATION_DIRECTIVE)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part5_text = read_text(PART5)

    isotropic_dominance_factor = float(prior_summary["isotropic_dominance_factor"])
    anisotropic_ceiling = float(prior_summary["transverse_anisotropic_ratio_ceiling"])
    scalar_residual_rel = float(prior_summary["scalar_alpha_residual_rel"])
    residual_scale_relative_gap = float(prior_summary["residual_scale_relative_gap"])
    absolute_gap = abs(anisotropic_ceiling - scalar_residual_rel)
    gap_percent_points = absolute_gap * 100.0

    prior_case_i_candidate = bool(prior_summary["case_i_leading_foundation_candidate"])
    prior_leading_foundation = bool(
        prior_summary["transverse_projection_leading_foundation_supported"]
    )
    prior_exact_foundation = bool(prior_summary["scalar_proxy_exact_foundation_supported"])

    dominance_supports_case_i = isotropic_dominance_factor > 20.0
    residual_scale_matching_supports_case_i = gap_percent_points < 0.05
    linear_source_zero_closed = bool(zero_summary["classification_case_iv_zero_under_current_pack"])
    naive_signed_density_wrong_path_closed = not bool(
        zero_summary["signed_density_exact_supported"]
    )
    perturbative_case_gamma_closed = bool(gamma_summary["case_gamma_selected"]) and bool(
        gamma_summary["perturbative_breakdown_detected"]
    )
    phase1_wrong_regime_closed = not bool(phase1_summary["phase1_close_within_one_percent"])
    internal_topological_wrong_path_closed = bool(
        reset_summary["internal_topological_lane_demoted_from_mainline"]
    )
    shifted_structure_demoted_to_nlo_retain = bool(
        shifted_summary["selected_disposition_case"]
        == "case_ii_shifted_structure_under_current_pack"
        and shifted_summary["nonzero_shifted_structure_retained"]
    )

    wrong_path_closure_support_count = float(
        sum(
            (
                linear_source_zero_closed,
                naive_signed_density_wrong_path_closed,
                perturbative_case_gamma_closed,
                phase1_wrong_regime_closed,
                internal_topological_wrong_path_closed,
            )
        )
    )

    case_i_leading_foundation_candidate_selected = bool(
        prior_case_i_candidate
        and prior_leading_foundation
        and not prior_exact_foundation
        and dominance_supports_case_i
        and residual_scale_matching_supports_case_i
        and wrong_path_closure_support_count >= 5.0
    )
    case_ii_shifted_structure_selected = False
    case_iii_transparent_zero_selected = False
    v_squared_subtraction_exact_treatment_admissible_now = bool(
        case_i_leading_foundation_candidate_selected
    )
    quadratic_tp_disposition_sync_admissible_now = False
    physical_reject_required = False

    rows = [
        row(
            "prior_case_i_candidate",
            "pass" if prior_case_i_candidate else "reject",
            "prior Case I leading-foundation candidate",
            truth(prior_case_i_candidate),
            "The classification branch starts from the already completed transverse-projection audit.",
        ),
        row(
            "isotropic_dominance_factor",
            "pass" if dominance_supports_case_i else "watch",
            "isotropic dominance factor",
            isotropic_dominance_factor,
            "The leading isotropic term remains larger than the projected anisotropic NLO correction by more than one order of magnitude.",
        ),
        row(
            "anisotropic_nlo_ceiling",
            "pass",
            "anisotropic NLO ceiling",
            anisotropic_ceiling,
            "The exact-branch ceiling for the projected anisotropic correction stays on the percent scale.",
        ),
        row(
            "scalar_residual_rel",
            "pass",
            "scalar residual vs target",
            scalar_residual_rel,
            "The retained scalar residual to be explained by NLO and subtraction effects.",
        ),
        row(
            "nlo_residual_gap_percent_points",
            "pass" if residual_scale_matching_supports_case_i else "watch",
            "NLO-residual gap in percentage points",
            gap_percent_points,
            "The NLO ceiling and the retained scalar residual already match within a few hundredths of one percentage point.",
        ),
        row(
            "linear_source_zero_closed",
            "pass" if linear_source_zero_closed else "reject",
            "linear source wrong path closed",
            truth(linear_source_zero_closed),
            "The linear J_eff lane is already closed to zero under the current pack.",
        ),
        row(
            "naive_signed_density_wrong_path_closed",
            "pass" if naive_signed_density_wrong_path_closed else "reject",
            "naive signed-density wrong path closed",
            truth(naive_signed_density_wrong_path_closed),
            "The naive signed-density exact read is already rejected by the direct current classification.",
        ),
        row(
            "perturbative_case_gamma_closed",
            "pass" if perturbative_case_gamma_closed else "reject",
            "perturbative f_L rescue closed by Case gamma",
            truth(perturbative_case_gamma_closed),
            "The perturbative rescue lane already closed through Case gamma breakdown.",
        ),
        row(
            "phase1_wrong_regime_closed",
            "pass" if phase1_wrong_regime_closed else "reject",
            "phase-1 exact solver wrong-regime closure",
            truth(phase1_wrong_regime_closed),
            "The original phase-1 exact solver did not close within one percent and therefore does not block the current Case-I candidate read.",
        ),
        row(
            "internal_topological_wrong_path_closed",
            "pass" if internal_topological_wrong_path_closed else "reject",
            "internal SU(2)/Hopf wrong path closed",
            truth(internal_topological_wrong_path_closed),
            "The internal topological lane is already demoted from the frozen-action mainline.",
        ),
        row(
            "wrong_path_closure_support_count",
            "pass" if wrong_path_closure_support_count >= 5.0 else "watch",
            "closed wrong-path support count",
            wrong_path_closure_support_count,
            "All major alternative rescue lanes are already closed honestly before fixing the current classification.",
        ),
        row(
            "case_i_leading_foundation_candidate_selected",
            "pass" if case_i_leading_foundation_candidate_selected else "reject",
            "Case I leading-foundation candidate selected",
            truth(case_i_leading_foundation_candidate_selected),
            "The current pack supports Case I at the leading-foundation level, but not exact foundation.",
        ),
        row(
            "case_ii_shifted_structure_demoted_to_nlo_retain",
            "pass" if shifted_structure_demoted_to_nlo_retain else "reject",
            "shifted structure demoted to NLO retain",
            truth(shifted_structure_demoted_to_nlo_retain),
            "The shifted structure is retained as the NLO correction rather than as the primary classification.",
        ),
        row(
            "v_squared_subtraction_exact_treatment_admissible_now",
            "pass" if v_squared_subtraction_exact_treatment_admissible_now else "reject",
            "v^2 subtraction exact treatment admissible now",
            truth(v_squared_subtraction_exact_treatment_admissible_now),
            "Once Case I candidate is fixed, the same-level quadratic next task is the exact v^2 subtraction treatment.",
        ),
        row(
            "quadratic_tp_disposition_sync_admissible_now",
            "pass" if quadratic_tp_disposition_sync_admissible_now else "reject",
            "quadratic transverse-projection disposition sync admissible now",
            truth(quadratic_tp_disposition_sync_admissible_now),
            "Disposition sync stays downstream of the exact v^2 subtraction treatment.",
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
            "classification_directive": display_path(CLASSIFICATION_DIRECTIVE),
            "prior_gate": display_path(PRIOR_GATE),
            "zero_class_gate": display_path(ZERO_CLASS_GATE),
            "shifted_disp_gate": display_path(SHIFTED_DISP_GATE),
            "mainline_reset_gate": display_path(MAINLINE_RESET_GATE),
            "perturbative_gamma_gate": display_path(PERTURBATIVE_GAMMA_GATE),
            "phase1_gate": display_path(PHASE1_GATE),
        }
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_classification_case": SELECTED_CASE,
        "transverse_projection_leading_foundation_supported": prior_leading_foundation,
        "scalar_proxy_exact_foundation_supported": prior_exact_foundation,
        "case_i_leading_foundation_candidate_selected": case_i_leading_foundation_candidate_selected,
        "case_ii_shifted_structure_selected": case_ii_shifted_structure_selected,
        "case_iii_transparent_zero_selected": case_iii_transparent_zero_selected,
        "shifted_structure_demoted_to_nlo_retain": shifted_structure_demoted_to_nlo_retain,
        "isotropic_dominance_factor": isotropic_dominance_factor,
        "anisotropic_nlo_ceiling": anisotropic_ceiling,
        "scalar_alpha_residual_rel": scalar_residual_rel,
        "nlo_residual_gap_percent_points": gap_percent_points,
        "nlo_residual_scale_match_supported": residual_scale_matching_supports_case_i,
        "closed_wrong_path_support_count": wrong_path_closure_support_count,
        "v_squared_subtraction_exact_treatment_admissible_now": (
            v_squared_subtraction_exact_treatment_admissible_now
        ),
        "quadratic_tp_disposition_sync_admissible_now": (
            quadratic_tp_disposition_sync_admissible_now
        ),
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "downstream_disposition_route_name": DOWNSTREAM_ROUTE_NAME,
        "downstream_disposition_route_or_none": DOWNSTREAM_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": case_i_leading_foundation_candidate_selected,
        "next_required_artifacts": [NEXT_ROUTE_NAME, DOWNSTREAM_ROUTE_NAME],
    }

    evidence = {
        "hits": {
            "directive_case_i": hit(CLASSIFICATION_DIRECTIVE.read_text(encoding="utf-8"), "Case I"),
            "directive_v2_subtraction": hit(
                CLASSIFICATION_DIRECTIVE.read_text(encoding="utf-8"), "v² subtraction"
            ),
            "directive_leading_term": hit(
                CLASSIFICATION_DIRECTIVE.read_text(encoding="utf-8"), "leading term は −λf₀²"
            ),
            "current_problem_leading_foundation": hit(
                current_problem_text, "leading foundation candidate"
            ),
            "current_status_leading_foundation": hit(
                current_status_text, "leading foundation candidate"
            ),
            "unified_roadmap_case_classification": hit(
                unified_roadmap_text, "quadratic transverse-projection α case classification"
            ),
            "part5_case_classification": hit(
                part5_text, "quadratic transverse-projection α case classification"
            ),
        },
        "support_counts": {
            "wrong_path_closure_support_count": wrong_path_closure_support_count,
            "same_level_open_component_count": 1.0,
            "future_v3_component_count": 2.0,
        },
    }

    inventory_paths = write_artifact(
        "inventory",
        payload(
            "8.7.56.1595",
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
            "8.7.56.1596",
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
            "8.7.56.1597",
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
            "8.7.56.1598",
            f"{STEP_NAME} route sync",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )

    print("[ok] quadratic transverse-projection case-classification artifacts written:")
    print(f" - {inventory_paths['json']}")
    print(f" - {audit_paths['json']}")
    print(f" - {gate_paths['json']}")
    print(f" - {route_paths['json']}")


if __name__ == "__main__":
    main()
