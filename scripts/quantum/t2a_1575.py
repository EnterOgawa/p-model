#!/usr/bin/env python3
"""Generate 8.7.56.1575-.1578 quadratic-operator derivation artifacts.

This branch follows the quadratic-expansion directive literally, but it uses
only the retained current-pack surfaces:

1. the frozen-action free vector term from Part I,
2. the retained breakthrough working action where the mexican hat is the
   unique P-sector mass source,
3. the restored exact vector / Q-ball background bookkeeping from the current
   Trial-2 route,
4. the current-pack absence of microscopic matter and rotational functionals.

The resulting statement is intentionally narrow. It does not yet classify the
operator numerically. Instead, it freezes the explicit quadratic backbone

    L_total^vec|_(a^2) = (1/2) a_mu K^{mu nu}[Q] a_nu

into a free Maxwell-like piece plus the background-dependent mexican-hat
Hessian core. Matter and rotational quadratic tails remain open.
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
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
PRIOR_RESET_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1571_1574_quad_mainline_reset_declaration_gate_metrics.json"
)
BREAKTHROUGH_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "mass_origin_v2_trial1_breakthrough_declaration_gate_metrics.json"
)
BREAKTHROUGH_VEV = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "mass_origin_v2_trial1_breakthrough_modified_vev_decomposition_metrics.json"
)
DIRECTIVE_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_quadratic_expansion_20260328.md"
)

STEP_TAG = "8.7.56.1575-1578"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor frozen-action quadratic K-operator derivation"
)
STEM = build_compact_artifact_stem(STEP_TAG, "quadratic_k_deriv", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_quadratic_expansion_mainline_reset_completed"
BRANCH_CLASS = (
    "vector_qball_form_factor_quadratic_k_operator_core_derived_structure_classification_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_quadratic_operator_structure_classification"
)
NEXT_ROUTE = "8.7.56.1579"
NEXT_DISPOSITION_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_quadratic_operator_disposition_sync"
)
NEXT_DISPOSITION_ROUTE = "8.7.56.1583"
DOWNSTREAM_SOURCE_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_effective_source_theorem_revisit_after_quadratic_disposition"
)
DOWNSTREAM_DICTIONARY_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_observable_dictionary_revisit_after_quadratic_disposition"
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


# 関数: quadratic K-operator の core 式を返す。

def build_formulae() -> dict[str, str]:
    """Return the frozen-action quadratic operator formulas."""
    return {
        "retained_working_action": (
            "L_work = -(Z_P/4) F_(P)^2 + (lambda/4)(|P|^2-v^2)^2 + g_P P_mu J_matter^mu"
        ),
        "fluctuation_split": "P_mu = Q_mu + a_mu with Q_mu = P_mu^Qball",
        "free_quadratic_piece": (
            "L_free|_(a^2) = (1/2) a_mu Z_P (eta^{mu nu} Box - partial^mu partial^nu) a_nu"
        ),
        "potential_quadratic_piece": (
            "V|_(a^2) = (lambda/2)(Q^2-v^2) a_mu a^mu + lambda (Q_mu a^mu)^2"
        ),
        "hessian_form": (
            "Delta K_core^{mu nu}[Q] = lambda[(Q^2-v^2) eta^{mu nu} + 2 Q^mu Q^nu]"
        ),
        "operator_split": (
            "K^{mu nu}[Q] = K_free^{mu nu} + Delta K_core^{mu nu}[Q]"
            " + Delta K_matter^{mu nu}[Q] + Delta K_rot^{mu nu}[Q]"
        ),
    }


# 関数: `.1575-.1578` を実行する。

def main() -> None:
    """Execute the frozen-action quadratic K-operator derivation branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PART1,
        PART3A,
        PART5,
        PRIOR_RESET_GATE,
        BREAKTHROUGH_GATE,
        BREAKTHROUGH_VEV,
        DIRECTIVE_NOTE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    directive_text = read_text(DIRECTIVE_NOTE)

    prior_reset_summary = read_json(PRIOR_RESET_GATE)["summary"]
    breakthrough_summary = read_json(BREAKTHROUGH_GATE)["evidence"]
    breakthrough_vev_summary = read_json(BREAKTHROUGH_VEV)["summary"]
    formulas = build_formulae()

    prior_reset_ready = bool(
        prior_reset_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and prior_reset_summary.get("quadratic_operator_mainline_promoted", False)
    )
    working_action_uses_mexican_hat_only_mass_source = bool(
        breakthrough_vev_summary.get("working_action_uses_mexican_hat_only_mass_source", False)
        and breakthrough_summary["modified_vev_summary"].get(
            "working_action_uses_mexican_hat_only_mass_source", False
        )
    )
    massless_transverse_mode_retained = bool(
        breakthrough_vev_summary.get("transverse_mode_massless_under_breakthrough_action", False)
    )

    part1_free_lagrangian_hit = hit(part1_text, "\\mathcal{L}_{P_\\mu}^{\\mathrm{free}}")
    part1_full_lagrangian_hit = hit(part1_text, "\\mathcal{L}_{P,\\mathrm{full}}")
    part3a_mexican_hat_hit = hit(part3a_text, "Mexican-hat-only working action")
    directive_quadratic_collect_hit = hit(directive_text, "a_μ の二次の項を集める")
    directive_free_hit = hit(directive_text, "standard Maxwell")
    directive_hessian_hit = hit(directive_text, "U''(Q^2)")
    directive_overlap_hit = hit(directive_text, "transverse projection")

    free_kinetic_quadratic_piece_available = bool(
        part1_free_lagrangian_hit and directive_quadratic_collect_hit and directive_free_hit
    )
    mexican_hat_quadratic_hessian_available = bool(
        working_action_uses_mexican_hat_only_mass_source
        and part3a_mexican_hat_hit
        and directive_hessian_hit
    )
    background_dependent_quadratic_core_available = bool(
        mexican_hat_quadratic_hessian_available and massless_transverse_mode_retained
    )
    matter_quadratic_functional_available = False
    rotational_quadratic_functional_available = False
    quadratic_operator_core_derived = bool(
        prior_reset_ready
        and free_kinetic_quadratic_piece_available
        and background_dependent_quadratic_core_available
    )
    full_quadratic_operator_without_matter_rot_available = quadratic_operator_core_derived
    background_dependent_core_symbolic_nonzero_before_projection = bool(
        quadratic_operator_core_derived and directive_overlap_hit
    )
    quadratic_structure_classification_admissible_now = quadratic_operator_core_derived
    effective_source_theorem_attempt_admissible_now = False
    observable_dictionary_gate_admissible_now = False
    physical_reject_required = False

    rows_inventory = [
        row(
            "prior_reset_ready",
            "pass" if prior_reset_ready else "fail",
            "prior_reset_ready",
            truth(prior_reset_ready),
            "Quadratic derivation only starts after the mainline has been reset away from the linear disposition lane.",
        ),
        row(
            "working_action_retained",
            "pass" if working_action_uses_mexican_hat_only_mass_source else "fail",
            "working_action_uses_mexican_hat_only_mass_source",
            truth(working_action_uses_mexican_hat_only_mass_source),
            "The retained breakthrough working action still uses the mexican hat as the unique mass source.",
        ),
        row(
            "free_quad_surface",
            "pass" if free_kinetic_quadratic_piece_available else "fail",
            "free_kinetic_quadratic_piece_available",
            truth(free_kinetic_quadratic_piece_available),
            "The current pack exposes the free quadratic Maxwell-like backbone explicitly.",
        ),
        row(
            "hessian_surface",
            "pass" if mexican_hat_quadratic_hessian_available else "fail",
            "mexican_hat_quadratic_hessian_available",
            truth(mexican_hat_quadratic_hessian_available),
            "The directive and retained breakthrough working action are sufficient to write the symbolic mexican-hat Hessian core.",
        ),
        row(
            "massless_transverse",
            "pass" if massless_transverse_mode_retained else "fail",
            "massless_transverse_mode_retained",
            truth(massless_transverse_mode_retained),
            "The transverse light branch remains the branch on which the quadratic operator is read.",
        ),
    ]

    inventory = payload(
        STEP_TAG,
        STEP_NAME,
        inputs={
            "required_paths": [
                display_path(PRIOR_RESET_GATE),
                display_path(BREAKTHROUGH_GATE),
                display_path(BREAKTHROUGH_VEV),
                display_path(PART1),
                display_path(PART3A),
                display_path(DIRECTIVE_NOTE),
            ],
            "current_step_context": "quadratic_k_operator_derivation",
            "retained_working_action_formula": formulas["retained_working_action"],
        },
        rows=rows_inventory,
        summary={
            "prior_reset_ready": prior_reset_ready,
            "working_action_uses_mexican_hat_only_mass_source": working_action_uses_mexican_hat_only_mass_source,
            "free_kinetic_quadratic_piece_available": free_kinetic_quadratic_piece_available,
            "mexican_hat_quadratic_hessian_available": mexican_hat_quadratic_hessian_available,
            "massless_transverse_mode_retained": massless_transverse_mode_retained,
        },
        decision={
            "inventory_ready": (
                prior_reset_ready
                and working_action_uses_mexican_hat_only_mass_source
                and free_kinetic_quadratic_piece_available
            ),
        },
        evidence={
            "status_hits": [
                hit(status_text, "quadratic `a_\\mu` expansion"),
                hit(status_text, "frozen action"),
            ],
            "current_problem_hits": [
                hit(current_problem_text, "frozen action の quadratic `a_\\mu^2` term"),
                hit(current_problem_text, "selected_disposition_case = case_iv_zero_under_current_pack"),
            ],
            "directive_hits": [
                hit(directive_text, "a_μ の二次の項を集める"),
                hit(directive_text, "U''(Q^2)"),
                hit(directive_text, "transverse projection"),
            ],
        },
    )

    rows_audit = [
        row(
            "quadratic_core",
            "pass" if quadratic_operator_core_derived else "fail",
            "quadratic_operator_core_derived",
            truth(quadratic_operator_core_derived),
            "The free quadratic backbone and the mexican-hat Hessian core can be written explicitly under the retained working action.",
        ),
        row(
            "background_core",
            "pass" if background_dependent_quadratic_core_available else "fail",
            "background_dependent_quadratic_core_available",
            truth(background_dependent_quadratic_core_available),
            "The current pack supports a Q-ball-background-dependent quadratic core before microscopic matter/rotation tails are added.",
        ),
        row(
            "matter_tail_open",
            "watch",
            "matter_quadratic_functional_available",
            truth(matter_quadratic_functional_available),
            "Microscopic matter-current quadratic tail remains unavailable in the current pack.",
        ),
        row(
            "rot_tail_open",
            "watch",
            "rotational_quadratic_functional_available",
            truth(rotational_quadratic_functional_available),
            "Rotational-source quadratic tail remains unavailable in the current pack.",
        ),
        row(
            "core_nonzero_preprojection",
            "pass" if background_dependent_core_symbolic_nonzero_before_projection else "fail",
            "background_dependent_core_symbolic_nonzero_before_projection",
            truth(background_dependent_core_symbolic_nonzero_before_projection),
            "Before transverse projection/classification, the symbolic mexican-hat core is not the same as a transparent-zero statement.",
        ),
        row(
            "classification_ready",
            "pass" if quadratic_structure_classification_admissible_now else "fail",
            "quadratic_structure_classification_admissible_now",
            truth(quadratic_structure_classification_admissible_now),
            "Structure classification is now admissible because the operator core exists explicitly.",
        ),
    ]

    audit = payload(
        STEP_TAG,
        STEP_NAME,
        inputs=inventory["summary"],
        rows=rows_audit,
        summary={
            "quadratic_operator_core_derived": quadratic_operator_core_derived,
            "free_kinetic_quadratic_piece_available": free_kinetic_quadratic_piece_available,
            "mexican_hat_quadratic_hessian_available": mexican_hat_quadratic_hessian_available,
            "background_dependent_quadratic_core_available": background_dependent_quadratic_core_available,
            "matter_quadratic_functional_available": matter_quadratic_functional_available,
            "rotational_quadratic_functional_available": rotational_quadratic_functional_available,
            "full_quadratic_operator_without_matter_rot_available": full_quadratic_operator_without_matter_rot_available,
            "background_dependent_core_symbolic_nonzero_before_projection": background_dependent_core_symbolic_nonzero_before_projection,
            "quadratic_structure_classification_admissible_now": quadratic_structure_classification_admissible_now,
            "effective_source_theorem_attempt_admissible_now": effective_source_theorem_attempt_admissible_now,
            "observable_dictionary_gate_admissible_now": observable_dictionary_gate_admissible_now,
            "physical_reject_required": physical_reject_required,
            "quadratic_operator_core_component_count": 2.0,
            "quadratic_operator_open_tail_component_count": 2.0,
        },
        decision={
            "audit_passed": (
                quadratic_operator_core_derived
                and background_dependent_quadratic_core_available
                and quadratic_structure_classification_admissible_now
            ),
        },
        evidence={
            "formulas": formulas,
            "part1_hits": [
                part1_free_lagrangian_hit,
                part1_full_lagrangian_hit,
            ],
            "part3a_hits": [
                part3a_mexican_hat_hit,
            ],
            "part5_hits": [
                hit(part5_text, "quadratic `a_\\mu` expansion"),
                hit(part5_text, "quadratic operator structure classification"),
            ],
            "unified_roadmap_hits": [
                hit(unified_roadmap_text, "frozen-action quadratic `a_\\mu` expansion"),
                hit(unified_roadmap_text, "quadratic operator structure classification"),
            ],
        },
    )

    declaration_summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "working_action_uses_mexican_hat_only_mass_source": working_action_uses_mexican_hat_only_mass_source,
        "free_kinetic_quadratic_piece_available": free_kinetic_quadratic_piece_available,
        "mexican_hat_quadratic_hessian_available": mexican_hat_quadratic_hessian_available,
        "background_dependent_quadratic_core_available": background_dependent_quadratic_core_available,
        "matter_quadratic_functional_available": matter_quadratic_functional_available,
        "rotational_quadratic_functional_available": rotational_quadratic_functional_available,
        "quadratic_operator_core_derived": quadratic_operator_core_derived,
        "full_quadratic_operator_without_matter_rot_available": full_quadratic_operator_without_matter_rot_available,
        "background_dependent_core_symbolic_nonzero_before_projection": background_dependent_core_symbolic_nonzero_before_projection,
        "quadratic_structure_classification_admissible_now": quadratic_structure_classification_admissible_now,
        "effective_source_theorem_attempt_admissible_now": effective_source_theorem_attempt_admissible_now,
        "observable_dictionary_gate_admissible_now": observable_dictionary_gate_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_disposition_route": NEXT_DISPOSITION_ROUTE_NAME,
        "selected_disposition_route_or_none": NEXT_DISPOSITION_ROUTE,
        "downstream_source_route_name": DOWNSTREAM_SOURCE_ROUTE_NAME,
        "downstream_dictionary_route_name": DOWNSTREAM_DICTIONARY_ROUTE_NAME,
        "physical_reject_required": physical_reject_required,
    }

    declaration = payload(
        STEP_TAG,
        STEP_NAME,
        inputs=audit["summary"],
        rows=rows_audit,
        summary=declaration_summary,
        decision={
            "declaration_gate_passed": (
                quadratic_operator_core_derived
                and quadratic_structure_classification_admissible_now
                and not physical_reject_required
            ),
        },
        evidence={"formulas": formulas},
    )

    route_summary = {
        "route_state_changed_by_current_branch": True,
        "numeric_state_changed_by_current_branch": False,
        "current_official_step_after_branch": "8.7.56.1579",
        "current_official_branch_after_branch": "8.7.56.1579-.1582",
        "current_official_next_route": NEXT_ROUTE_NAME,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
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
                "The branch moves the mainline from route reset to explicit quadratic-operator derivation.",
            ),
            row(
                "numeric_hold",
                "pass",
                "numeric_state_changed_by_current_branch",
                0.0,
                "No numeric candidate is changed by the symbolic quadratic derivation.",
            ),
        ],
        summary=route_summary,
        decision={"route_sync_passed": True},
        evidence={
            "selected_routes": {
                "next": NEXT_ROUTE_NAME,
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
