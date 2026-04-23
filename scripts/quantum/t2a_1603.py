#!/usr/bin/env python3
"""Generate 8.7.56.1603-.1606 effective-metric quadratic mainline-reset artifacts.

The prior `.1599-.1602` branch closed the exact `v^2` subtraction treatment
honestly: under the current Minkowski-contracted quadratic lane, the exact
subtraction profile collapses onto the previously rejected signed-density
kernel and therefore worsens the retained scalar candidate.

An expert directive then sharpened the retry-gate judgment again. The recent
computational lanes shared one common contraction choice:

- the quadratic core was built with `eta^{mu nu}`,
- the transverse-projection audit classified that `eta`-contracted kernel,
- the exact `v^2` subtraction treatment evaluated that same `eta`-contracted
  kernel.

But Part I already froze the metric choice itself:

    caseA: eta_{mu nu}  -> reject
    caseB: g_{mu nu}(P) -> pass

So this branch does not try to reinterpret the worsen result away. Instead it
retains the Minkowski-lane worsen result honestly, closes that current lane as
one rejected current-pack reading, and resets the scientific mainline toward a
caseB/effective-metric self-consistent recomputation of the quadratic kernel.
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
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

DIRECTIVE_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_effective_metric_contraction_20260328.md"
)
PRIOR_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1599_1602_v2_sub_exact_treat_declaration_gate_metrics.json"
)
TP_AUDIT_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1591_1594_tp_alpha_audit_declaration_gate_metrics.json"
)
QUAD_DERIV_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1575_1578_quadratic_k_deriv_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1603-1606"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor effective-metric quadratic mainline reset"
)
STEM = build_compact_artifact_stem(STEP_TAG, "eff_metric_mainline_reset", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_v2_subtraction_exact_treatment_signed_kernel_worsen_disposition_sync_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_effective_metric_quadratic_mainline_reset_completed"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_effective_metric_quadratic_contraction_derivation"
)
NEXT_ROUTE = "8.7.56.1607"
NEXT_AUDIT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_effective_metric_transverse_projection_alpha_audit"
)
NEXT_AUDIT_ROUTE = "8.7.56.1611"
NEXT_SUBTRACTION_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_effective_metric_v2_subtraction_exact_treatment"
)
NEXT_SUBTRACTION_ROUTE = "8.7.56.1615"
NEXT_DISPOSITION_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_effective_metric_disposition_sync"
)
NEXT_DISPOSITION_ROUTE = "8.7.56.1619"


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


# 関数: reset branch で固定する主要式を返す。

def build_formulae() -> dict[str, str]:
    """Return the caseA/caseB reset formulas."""
    return {
        "metric_choice_gate": "caseA: eta_{mu nu} -> reject, caseB: g_{mu nu}(P) -> pass",
        "effective_metric_static_limit": "g^{00}(P) = -e^{2u}, g^{ij}(P) = e^{-2u} delta^{ij}",
        "effective_metric_log_profile": "u(r) = ln(|f_0(r)| / P_ref)",
        "minkowski_norm": "Q^2|_eta = -f_0^2 + f_L^2",
        "effective_metric_norm": "Q^2|_g = -e^{2u} f_0^2 + e^{-2u} f_L^2",
        "suppression_ratio": "R_g = e^{-4u}(f_L/f_0)^2",
    }


# 関数: `.1603-.1606` を実行する。

def main() -> None:
    """Execute the effective-metric quadratic mainline-reset branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PART1,
        PART5,
        DIRECTIVE_NOTE,
        PRIOR_GATE,
        TP_AUDIT_GATE,
        QUAD_DERIV_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part1_text = read_text(PART1)
    part5_text = read_text(PART5)
    directive_text = read_text(DIRECTIVE_NOTE)

    prior_summary = read_json(PRIOR_GATE)["summary"]
    tp_summary = read_json(TP_AUDIT_GATE)["summary"]
    quad_summary = read_json(QUAD_DERIV_GATE)["summary"]

    prior_worsen_ready = bool(
        prior_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and prior_summary.get("selected_subtraction_result_case") == "worsen"
        and not prior_summary.get("case_i_exact_foundation_supported", True)
    )
    casea_reject_caseb_pass_available = bool(
        hit(part1_text, "\\mathrm{caseA}:\\eta_{\\mu\\nu}\\Rightarrow \\mathrm{reject},")
        and hit(part1_text, "\\mathrm{caseB}:g_{\\mu\\nu}(P)\\Rightarrow \\mathrm{pass}")
    )
    effective_metric_surface_available = bool(
        hit(part1_text, "g_{\\mu\\nu}^{(P)}")
        and hit(part1_text, "g_{tt}^{(P)}=-e^{-2u}")
        and hit(part1_text, "\\nabla^{(g(P))}_\\mu F_{(P)}^{\\mu\\nu}=0")
    )
    current_quadratic_lane_uses_eta = bool(
        quad_summary.get("quadratic_operator_core_derived", False)
        and "eta^{mu nu}" in read_json(QUAD_DERIV_GATE)["evidence"]["formulas"]["hessian_form"]
        and tp_summary.get("transverse_projection_leading_foundation_supported", False)
    )
    directive_requires_effective_metric_recompute = bool(
        hit(directive_text, "quadratic kernel を effective metric で contract し直す")
        and hit(directive_text, "caseA（Minkowski、rejected）")
        and hit(directive_text, "caseB（effective metric）")
    )
    directive_bans_new_parameters = bool(
        hit(directive_text, "新パラメータなし")
        and hit(directive_text, "frozen action + caseB")
    )

    minkowski_worsen_retained_as_casea_result = bool(
        prior_worsen_ready and current_quadratic_lane_uses_eta
    )
    quadratic_tp_disposition_sync_demoted_from_mainline = bool(
        minkowski_worsen_retained_as_casea_result
        and casea_reject_caseb_pass_available
        and directive_requires_effective_metric_recompute
    )
    caseb_effective_metric_promoted_to_mainline = bool(
        quadratic_tp_disposition_sync_demoted_from_mainline
        and effective_metric_surface_available
        and directive_bans_new_parameters
    )
    effective_metric_quadratic_contraction_derivation_scheduled = bool(
        caseb_effective_metric_promoted_to_mainline
    )
    effective_metric_tp_audit_scheduled = bool(caseb_effective_metric_promoted_to_mainline)
    effective_metric_v2_subtraction_scheduled = bool(caseb_effective_metric_promoted_to_mainline)
    effective_metric_disposition_sync_scheduled = bool(caseb_effective_metric_promoted_to_mainline)
    future_external_input_side_lane_retained = True
    new_free_parameters_introduced = False
    physical_reject_required = False

    rows = [
        row(
            "prior_worsen_ready",
            "pass" if prior_worsen_ready else "reject",
            "prior v2 subtraction worsen retained",
            truth(prior_worsen_ready),
            "The prior branch already fixed that the exact v^2 subtraction treatment worsens the retained scalar candidate.",
        ),
        row(
            "casea_reject_caseb_pass_available",
            "pass" if casea_reject_caseb_pass_available else "reject",
            "Part I caseA reject / caseB pass gate available",
            truth(casea_reject_caseb_pass_available),
            "Part I already freezes the metric-choice gate itself: eta is reject and g(P) is pass.",
        ),
        row(
            "effective_metric_surface_available",
            "pass" if effective_metric_surface_available else "reject",
            "effective metric surface available",
            truth(effective_metric_surface_available),
            "The current pack already exposes g_{mu nu}(P), its static limit, and the g(P)-covariant vacuum equation.",
        ),
        row(
            "current_quadratic_lane_uses_eta",
            "pass" if current_quadratic_lane_uses_eta else "reject",
            "current quadratic lane uses eta contraction",
            truth(current_quadratic_lane_uses_eta),
            "The current quadratic core and transverse-projection lane were built from the eta-contracted Hessian core.",
        ),
        row(
            "directive_requires_effective_metric_recompute",
            "pass" if directive_requires_effective_metric_recompute else "reject",
            "directive requires effective-metric recomputation",
            truth(directive_requires_effective_metric_recompute),
            "The new directive identifies Minkowski contraction as the shared failure mode and asks for a caseB self-consistent recomputation.",
        ),
        row(
            "directive_bans_new_parameters",
            "pass" if directive_bans_new_parameters else "reject",
            "directive bans new parameters",
            truth(directive_bans_new_parameters),
            "The proposed reset keeps the frozen action and introduces no new parameter.",
        ),
        row(
            "minkowski_worsen_retained_as_casea_result",
            "pass" if minkowski_worsen_retained_as_casea_result else "reject",
            "Minkowski worsen retained as caseA result",
            truth(minkowski_worsen_retained_as_casea_result),
            "The prior worsen result is retained honestly as the outcome of the eta-contracted current-pack lane rather than erased.",
        ),
        row(
            "quadratic_tp_disposition_sync_demoted_from_mainline",
            "pass" if quadratic_tp_disposition_sync_demoted_from_mainline else "reject",
            "quadratic TP disposition sync demoted from mainline",
            truth(quadratic_tp_disposition_sync_demoted_from_mainline),
            "Once the metric-choice mismatch is recognized, disposition wording is no longer the immediate mainline step.",
        ),
        row(
            "caseb_effective_metric_promoted_to_mainline",
            "pass" if caseb_effective_metric_promoted_to_mainline else "reject",
            "caseB effective-metric quadratic recomputation promoted to mainline",
            truth(caseb_effective_metric_promoted_to_mainline),
            "The next scientific branch is the caseB/effective-metric quadratic contraction derivation.",
        ),
        row(
            "effective_metric_quadratic_contraction_derivation_scheduled",
            "pass" if effective_metric_quadratic_contraction_derivation_scheduled else "reject",
            "effective-metric quadratic contraction derivation scheduled",
            truth(effective_metric_quadratic_contraction_derivation_scheduled),
            "The immediate next branch derives the caseB-contracted quadratic kernel explicitly.",
        ),
        row(
            "effective_metric_tp_audit_scheduled",
            "pass" if effective_metric_tp_audit_scheduled else "reject",
            "effective-metric transverse-projection audit scheduled",
            truth(effective_metric_tp_audit_scheduled),
            "After derivation, the same transverse-projection audit must be rerun on the caseB kernel.",
        ),
        row(
            "effective_metric_v2_subtraction_scheduled",
            "pass" if effective_metric_v2_subtraction_scheduled else "reject",
            "effective-metric v2 subtraction exact treatment scheduled",
            truth(effective_metric_v2_subtraction_scheduled),
            "The v^2 subtraction exact treatment is downstream of the caseB-contracted kernel rather than closed by the caseA result.",
        ),
        row(
            "new_free_parameters_introduced",
            "pass" if not new_free_parameters_introduced else "reject",
            "new free parameters introduced",
            truth(new_free_parameters_introduced),
            "This mainline reset adds no new free parameter.",
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
            "part1": display_path(PART1),
            "part5": display_path(PART5),
            "directive_note": display_path(DIRECTIVE_NOTE),
            "prior_gate": display_path(PRIOR_GATE),
            "tp_audit_gate": display_path(TP_AUDIT_GATE),
            "quadratic_derivation_gate": display_path(QUAD_DERIV_GATE),
        },
        "constants": {
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "next_audit_route_name": NEXT_AUDIT_ROUTE_NAME,
            "next_audit_route": NEXT_AUDIT_ROUTE,
            "next_subtraction_route_name": NEXT_SUBTRACTION_ROUTE_NAME,
            "next_subtraction_route": NEXT_SUBTRACTION_ROUTE,
            "next_disposition_route_name": NEXT_DISPOSITION_ROUTE_NAME,
            "next_disposition_route": NEXT_DISPOSITION_ROUTE,
            "scalar_F_exact_at_q_theory": 0.2998913524347805,
            "scalar_alpha_exact_at_q_theory": 0.00715678583937324,
            "vector_F_at_q_theory": -0.083735013520183,
            "vector_alpha_at_q_theory": 0.0005579616187042394,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "prior_v2_subtraction_worsen_retained": prior_worsen_ready,
        "casea_reject_caseb_pass_available": casea_reject_caseb_pass_available,
        "effective_metric_surface_available": effective_metric_surface_available,
        "current_quadratic_lane_uses_eta": current_quadratic_lane_uses_eta,
        "directive_requires_effective_metric_recompute": (
            directive_requires_effective_metric_recompute
        ),
        "directive_bans_new_parameters": directive_bans_new_parameters,
        "minkowski_worsen_retained_as_casea_result": (
            minkowski_worsen_retained_as_casea_result
        ),
        "quadratic_tp_disposition_sync_demoted_from_mainline": (
            quadratic_tp_disposition_sync_demoted_from_mainline
        ),
        "caseb_effective_metric_promoted_to_mainline": (
            caseb_effective_metric_promoted_to_mainline
        ),
        "effective_metric_quadratic_contraction_derivation_scheduled": (
            effective_metric_quadratic_contraction_derivation_scheduled
        ),
        "effective_metric_tp_audit_scheduled": effective_metric_tp_audit_scheduled,
        "effective_metric_v2_subtraction_scheduled": effective_metric_v2_subtraction_scheduled,
        "effective_metric_disposition_sync_scheduled": (
            effective_metric_disposition_sync_scheduled
        ),
        "future_external_input_side_lane_retained": future_external_input_side_lane_retained,
        "new_free_parameters_introduced": new_free_parameters_introduced,
        "physical_reject_required": physical_reject_required,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": NEXT_AUDIT_ROUTE_NAME,
        "selected_followup_route_or_none": NEXT_AUDIT_ROUTE,
        "selected_subtraction_route": NEXT_SUBTRACTION_ROUTE_NAME,
        "selected_subtraction_route_or_none": NEXT_SUBTRACTION_ROUTE,
        "selected_disposition_route": NEXT_DISPOSITION_ROUTE_NAME,
        "selected_disposition_route_or_none": NEXT_DISPOSITION_ROUTE,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": caseb_effective_metric_promoted_to_mainline,
        "next_required_artifacts": [
            NEXT_ROUTE_NAME,
            NEXT_AUDIT_ROUTE_NAME,
            NEXT_SUBTRACTION_ROUTE_NAME,
            NEXT_DISPOSITION_ROUTE_NAME,
        ],
    }

    evidence = {
        "formulas": build_formulae(),
        "hits": {
            "part1_caseA_caseB": hit(part1_text, "\\mathrm{caseA}:\\eta_{\\mu\\nu}\\Rightarrow \\mathrm{reject},"),
            "part1_caseB_pass": hit(part1_text, "\\mathrm{caseB}:g_{\\mu\\nu}(P)\\Rightarrow \\mathrm{pass}"),
            "part1_effective_metric": hit(part1_text, "g_{\\mu\\nu}^{(P)}"),
            "part1_covariant_vacuum_eq": hit(part1_text, "\\nabla^{(g(P))}_\\mu F_{(P)}^{\\mu\\nu}=0"),
            "directive_minkowski_failure": hit(directive_text, "全て Minkowski metric で P_μ の contraction"),
            "directive_caseB_apply": hit(directive_text, "caseB を self-consistently に適用する"),
            "current_problem_v2_subtraction": hit(current_problem_text, "v^2 subtraction"),
            "current_status_v2_subtraction": hit(current_status_text, "v^2 subtraction"),
            "unified_roadmap_v2_subtraction": hit(unified_roadmap_text, "`.1599-.1602` は **`v^2` subtraction exact treatment audit**"),
            "part5_v2_subtraction": hit(part5_text, "v^2 subtraction"),
        },
        "carry_over": {
            "prior_summary": prior_summary,
            "tp_summary": tp_summary,
            "quad_summary": quad_summary,
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

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload("8.7.56.1603", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence),
        ),
        "audit": write_artifact(
            "audit",
            payload("8.7.56.1604", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload(
                "8.7.56.1605",
                f"{STEP_NAME} declaration gate",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "route_sync": write_artifact(
            "route_sync",
            payload("8.7.56.1606", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence),
        ),
    }

    print(json.dumps({"step": STEP_TAG, "stem": STEM, "artifacts": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
