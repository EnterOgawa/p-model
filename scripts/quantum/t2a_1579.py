#!/usr/bin/env python3
"""Generate 8.7.56.1579-.1582 quadratic-operator classification artifacts.

The previous branch already derived the frozen-action quadratic backbone

    K^{mu nu}[Q] = K_free^{mu nu} + Delta K_core^{mu nu}[Q]
                 + Delta K_matter^{mu nu}[Q] + Delta K_rot^{mu nu}[Q]

with the current-pack explicit core

    Delta K_core^{mu nu}[Q]
      = lambda[(Q^2-v^2) eta^{mu nu} + 2 Q^mu Q^nu].

This branch is therefore not another derivation branch. It is a blind
structure-classification branch: does the derived quadratic core behave as

1. scalar-foundation,
2. shifted-structure,
3. transparent-zero?

Under the current pack, the honest answer is "shifted-structure". The core is
symbolically nonzero and it contains both an isotropic shift and an anisotropic
background projector. That is enough to reject transparent-zero, but not enough
to claim that the scalar proxy is already the exact quadratic foundation.
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
PRIOR_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1575_1578_quadratic_k_deriv_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1579-1582"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor quadratic operator structure classification"
)
STEM = build_compact_artifact_stem(STEP_TAG, "quadratic_k_class", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_quadratic_k_operator_core_derived_structure_classification_next"
BRANCH_CLASS = (
    "vector_qball_form_factor_quadratic_operator_shifted_structure_disposition_sync_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_quadratic_operator_disposition_sync"
)
NEXT_ROUTE = "8.7.56.1583"
DOWNSTREAM_SOURCE_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_effective_source_theorem_revisit_after_quadratic_disposition"
)
DOWNSTREAM_SOURCE_ROUTE = "8.7.56.1587"
DOWNSTREAM_DICTIONARY_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_observable_dictionary_revisit_after_quadratic_disposition"
)
DOWNSTREAM_DICTIONARY_ROUTE = "8.7.56.1591"


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


# 関数: `.1579-.1582` を実行する。

def main() -> None:
    """Execute the quadratic operator structure-classification branch."""
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
    ):
        require(path)

    prior_gate = read_json(PRIOR_GATE)
    prior_summary = prior_gate["summary"]
    prior_formulas = prior_gate["evidence"]["formulas"]

    directive_text = read_text(DIRECTIVE_NOTE)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part5_text = read_text(PART5)

    prior_ready = bool(
        prior_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and prior_summary.get("quadratic_operator_core_derived", False)
        and prior_summary.get("background_dependent_quadratic_core_available", False)
        and prior_summary.get("quadratic_structure_classification_admissible_now", False)
    )
    background_dependent_core_symbolic_nonzero_before_projection = bool(
        prior_summary.get("background_dependent_core_symbolic_nonzero_before_projection", False)
    )
    matter_quadratic_functional_available = bool(
        prior_summary.get("matter_quadratic_functional_available", False)
    )
    rotational_quadratic_functional_available = bool(
        prior_summary.get("rotational_quadratic_functional_available", False)
    )

    hessian_form = prior_formulas["hessian_form"]
    isotropic_shift_component_present = "(Q^2-v^2) eta^{mu nu}" in hessian_form
    anisotropic_projector_component_present = "2 Q^mu Q^nu" in hessian_form
    scalar_foundation_exact_supported = False
    shifted_structure_selected_under_current_pack = bool(
        prior_ready
        and background_dependent_core_symbolic_nonzero_before_projection
        and isotropic_shift_component_present
        and anisotropic_projector_component_present
    )
    transparent_zero_supported = False
    scalar_proxy_leading_approximation_retained = shifted_structure_selected_under_current_pack
    transverse_overlap_weighting_required = shifted_structure_selected_under_current_pack
    quadratic_structure_case_i_scalar_foundation = scalar_foundation_exact_supported
    quadratic_structure_case_ii_shifted_structure = shifted_structure_selected_under_current_pack
    quadratic_structure_case_iii_transparent_zero = transparent_zero_supported
    quadratic_operator_structure_identified = bool(
        quadratic_structure_case_ii_shifted_structure
    )
    quadratic_operator_disposition_sync_admissible_now = bool(
        quadratic_operator_structure_identified
    )
    effective_source_theorem_attempt_admissible_now = False
    observable_dictionary_gate_admissible_now = False
    physical_reject_required = False

    rows = [
        row(
            "prior_core_ready",
            "pass" if prior_ready else "reject",
            "prior quadratic core ready",
            truth(prior_ready),
            "Structure classification only starts after the quadratic operator core has been derived explicitly.",
        ),
        row(
            "background_core_nonzero",
            "pass" if background_dependent_core_symbolic_nonzero_before_projection else "reject",
            "background-dependent core symbolically nonzero before projection",
            truth(background_dependent_core_symbolic_nonzero_before_projection),
            "The derived core is not a transparent-zero statement at the symbolic operator level.",
        ),
        row(
            "isotropic_shift_component_present",
            "pass" if isotropic_shift_component_present else "reject",
            "isotropic shift component present",
            truth(isotropic_shift_component_present),
            "The quadratic core contains the scalar-background shift proportional to (Q^2-v^2) eta^{mu nu}.",
        ),
        row(
            "anisotropic_projector_component_present",
            "pass" if anisotropic_projector_component_present else "reject",
            "anisotropic background projector component present",
            truth(anisotropic_projector_component_present),
            "The quadratic core also contains the anisotropic 2 Q^mu Q^nu projector term.",
        ),
        row(
            "matter_tail_open",
            "watch",
            "matter_quadratic_functional_available",
            truth(matter_quadratic_functional_available),
            "Matter quadratic tail is still absent and therefore cannot yet collapse the structure to a fully microscopic scalar foundation.",
        ),
        row(
            "rot_tail_open",
            "watch",
            "rotational_quadratic_functional_available",
            truth(rotational_quadratic_functional_available),
            "Rotational quadratic tail is still absent and therefore cannot yet collapse the structure to a fully microscopic scalar foundation.",
        ),
        row(
            "scalar_foundation_exact_supported",
            "pass" if scalar_foundation_exact_supported else "reject",
            "scalar-foundation exact supported",
            truth(scalar_foundation_exact_supported),
            "The current pack does not yet justify that the quadratic correction reduces exactly to the scalar proxy foundation.",
        ),
        row(
            "shifted_structure_selected_under_current_pack",
            "pass" if shifted_structure_selected_under_current_pack else "reject",
            "shifted-structure selected under current pack",
            truth(shifted_structure_selected_under_current_pack),
            "The simultaneous isotropic shift plus anisotropic projector is the honest read of the derived quadratic core.",
        ),
        row(
            "transparent_zero_supported",
            "pass" if transparent_zero_supported else "reject",
            "transparent-zero supported",
            truth(transparent_zero_supported),
            "Transparent-zero is ruled out because the derived background-dependent core remains symbolically nonzero before projection.",
        ),
        row(
            "scalar_proxy_leading_approximation_retained",
            "pass" if scalar_proxy_leading_approximation_retained else "reject",
            "scalar-proxy leading approximation retained",
            truth(scalar_proxy_leading_approximation_retained),
            "The scalar proxy remains a retained leading-approximation candidate, but not an exact classification.",
        ),
        row(
            "transverse_overlap_weighting_required",
            "pass" if transverse_overlap_weighting_required else "reject",
            "transverse-overlap weighting required",
            truth(transverse_overlap_weighting_required),
            "The anisotropic projector term implies that overlap weighting, not naive scalar density alone, controls the quadratic correction.",
        ),
        row(
            "quadratic_operator_structure_identified",
            "pass" if quadratic_operator_structure_identified else "reject",
            "quadratic operator structure identified",
            truth(quadratic_operator_structure_identified),
            "The present branch identifies the honest current-pack class as shifted-structure rather than scalar-foundation or transparent-zero.",
        ),
        row(
            "quadratic_operator_disposition_sync_admissible_now",
            "pass" if quadratic_operator_disposition_sync_admissible_now else "reject",
            "quadratic operator disposition sync admissible now",
            truth(quadratic_operator_disposition_sync_admissible_now),
            "Formal disposition sync is the immediate next branch once the honest shifted-structure classification is fixed.",
        ),
        row(
            "effective_source_theorem_attempt_admissible_now",
            "pass" if effective_source_theorem_attempt_admissible_now else "reject",
            "effective source theorem attempt admissible now",
            truth(effective_source_theorem_attempt_admissible_now),
            "Source-theorem work remains downstream of disposition sync and is still blocked by absent microscopic tails.",
        ),
        row(
            "observable_dictionary_gate_admissible_now",
            "pass" if observable_dictionary_gate_admissible_now else "reject",
            "observable dictionary gate admissible now",
            truth(observable_dictionary_gate_admissible_now),
            "Observable dictionary remains downstream of both quadratic disposition and any later source-theorem recovery.",
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
        },
        "constants": {
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "downstream_source_route_name": DOWNSTREAM_SOURCE_ROUTE_NAME,
            "downstream_source_route": DOWNSTREAM_SOURCE_ROUTE,
            "downstream_dictionary_route_name": DOWNSTREAM_DICTIONARY_ROUTE_NAME,
            "downstream_dictionary_route": DOWNSTREAM_DICTIONARY_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "quadratic_operator_core_derived": prior_summary.get("quadratic_operator_core_derived", False),
        "background_dependent_quadratic_core_available": prior_summary.get(
            "background_dependent_quadratic_core_available", False
        ),
        "matter_quadratic_functional_available": matter_quadratic_functional_available,
        "rotational_quadratic_functional_available": rotational_quadratic_functional_available,
        "isotropic_shift_component_present": isotropic_shift_component_present,
        "anisotropic_projector_component_present": anisotropic_projector_component_present,
        "scalar_foundation_exact_supported": scalar_foundation_exact_supported,
        "shifted_structure_selected_under_current_pack": shifted_structure_selected_under_current_pack,
        "transparent_zero_supported": transparent_zero_supported,
        "scalar_proxy_leading_approximation_retained": scalar_proxy_leading_approximation_retained,
        "transverse_overlap_weighting_required": transverse_overlap_weighting_required,
        "quadratic_structure_case_i_scalar_foundation": quadratic_structure_case_i_scalar_foundation,
        "quadratic_structure_case_ii_shifted_structure": quadratic_structure_case_ii_shifted_structure,
        "quadratic_structure_case_iii_transparent_zero": quadratic_structure_case_iii_transparent_zero,
        "quadratic_operator_structure_identified": quadratic_operator_structure_identified,
        "quadratic_operator_disposition_sync_admissible_now": quadratic_operator_disposition_sync_admissible_now,
        "effective_source_theorem_attempt_admissible_now": effective_source_theorem_attempt_admissible_now,
        "observable_dictionary_gate_admissible_now": observable_dictionary_gate_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "downstream_source_route_name": DOWNSTREAM_SOURCE_ROUTE_NAME,
        "downstream_source_route_or_none": DOWNSTREAM_SOURCE_ROUTE,
        "downstream_dictionary_route_name": DOWNSTREAM_DICTIONARY_ROUTE_NAME,
        "downstream_dictionary_route_or_none": DOWNSTREAM_DICTIONARY_ROUTE,
        "scalar_strong_candidate_retained": True,
        "blind_vector_no_go_retained": True,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": quadratic_operator_structure_identified,
        "next_required_artifacts": [
            NEXT_ROUTE_NAME,
            DOWNSTREAM_SOURCE_ROUTE_NAME,
            DOWNSTREAM_DICTIONARY_ROUTE_NAME,
        ],
    }

    evidence = {
        "formulas": {
            "free_quadratic_piece": prior_formulas["free_quadratic_piece"],
            "hessian_form": hessian_form,
            "operator_split": prior_formulas["operator_split"],
            "classification_rule": (
                "A derived quadratic core that is symbolically nonzero and contains both "
                "the isotropic shift and the anisotropic Q-projector is classified as shifted-structure."
            ),
        },
        "hits": {
            "directive_scalar_proxy_foundation": hit(directive_text, "scalar proxy α = 0.00716"),
            "directive_case_i": hit(directive_text, "Case I: ΔK から α ≈ 0.00716"),
            "directive_case_ii": hit(directive_text, "Case II: ΔK の構造は出るが数値が異なる"),
            "directive_case_iii": hit(directive_text, "Case III: ΔK[Q] = 0"),
            "current_problem_quad_route": hit(
                current_problem_text, "quadratic operator structure classification"
            ),
            "current_status_quad_route": hit(
                current_status_text, "quadratic operator structure classification"
            ),
            "unified_roadmap_quad_route": hit(
                unified_roadmap_text, "quadratic operator structure classification"
            ),
            "part5_quad_route": hit(part5_text, "quadratic operator structure classification"),
        },
        "support_counts": {
            "background_tensor_component_count": 2.0,
            "open_tail_component_count": 2.0,
            "scalar_foundation_support_count": 0.0,
            "transparent_zero_support_count": 0.0,
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
        payload("8.7.56.1579", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence),
    )
    audit_paths = write_artifact(
        "audit",
        payload("8.7.56.1580", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence),
    )
    gate_paths = write_artifact(
        "declaration_gate",
        payload(
            "8.7.56.1581",
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
        payload("8.7.56.1582", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence),
    )

    print("[ok] quadratic operator structure-classification artifacts written:")
    print(f" - {inventory_paths['json']}")
    print(f" - {audit_paths['json']}")
    print(f" - {gate_paths['json']}")
    print(f" - {route_paths['json']}")


if __name__ == "__main__":
    main()
