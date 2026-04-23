#!/usr/bin/env python3
"""Generate 8.7.56.1583-.1586 quadratic-operator disposition-sync artifacts.

The prior two branches already fixed the essential computation:

1. `.1575-.1578` derived the frozen-action quadratic operator

       L_total^vec|_(a^2) = (1/2) a_mu K^{mu nu}[Q] a_nu

   with the current-pack explicit core

       Delta K_core^{mu nu}[Q]
         = lambda[(Q^2-v^2) eta^{mu nu} + 2 Q^mu Q^nu].

2. `.1579-.1582` then classified that derived operator honestly:
   it is neither exact scalar-foundation nor transparent-zero, but
   shifted-structure under the current pack.

This branch does not derive a new operator. It synchronizes that result as the
official disposition:

- nonzero shifted-structure is retained honestly,
- the scalar proxy remains only a leading approximation,
- the restored direct-vector fixed-q no-go is retained,
- the next admissible computation becomes the source-theorem revisit after the
  quadratic disposition.
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

STEP_TAG = "8.7.56.1583-1586"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor quadratic operator disposition sync"
)
STEM = build_compact_artifact_stem(STEP_TAG, "quadratic_k_disp", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_quadratic_operator_shifted_structure_disposition_sync_next"
BRANCH_CLASS = (
    "vector_qball_form_factor_quadratic_shifted_structure_disposition_sync_completed"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_effective_source_theorem_revisit_after_quadratic_disposition"
)
NEXT_ROUTE = "8.7.56.1587"
DOWNSTREAM_DICTIONARY_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_observable_dictionary_revisit_after_quadratic_disposition"
)
DOWNSTREAM_DICTIONARY_ROUTE = "8.7.56.1591"
SIDE_LANE_NAME = "future_external_input_side_lane_retained"


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


# 関数: `.1583-.1586` を実行する。

def main() -> None:
    """Execute the quadratic operator disposition-sync branch."""
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
        PRIOR_DERIV_GATE,
        PRIOR_CLASS_GATE,
    ):
        require(path)

    deriv_gate = read_json(PRIOR_DERIV_GATE)
    class_gate = read_json(PRIOR_CLASS_GATE)
    deriv_summary = deriv_gate["summary"]
    deriv_formulas = deriv_gate["evidence"]["formulas"]
    class_summary = class_gate["summary"]
    class_evidence = class_gate["evidence"]

    directive_text = read_text(DIRECTIVE_NOTE)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part5_text = read_text(PART5)

    prior_derivation_ready = bool(
        deriv_summary.get("quadratic_operator_core_derived", False)
        and deriv_summary.get("background_dependent_quadratic_core_available", False)
    )
    prior_classification_ready = bool(
        class_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and class_summary.get("quadratic_operator_structure_identified", False)
        and class_summary.get("shifted_structure_selected_under_current_pack", False)
        and not class_summary.get("scalar_foundation_exact_supported", True)
        and not class_summary.get("transparent_zero_supported", True)
    )

    isotropic_shift_component_present = bool(
        class_summary.get("isotropic_shift_component_present", False)
    )
    anisotropic_projector_component_present = bool(
        class_summary.get("anisotropic_projector_component_present", False)
    )
    scalar_foundation_exact_supported = bool(
        class_summary.get("scalar_foundation_exact_supported", False)
    )
    shifted_structure_selected_under_current_pack = bool(
        class_summary.get("shifted_structure_selected_under_current_pack", False)
    )
    transparent_zero_supported = bool(
        class_summary.get("transparent_zero_supported", False)
    )
    scalar_proxy_leading_approximation_retained = bool(
        class_summary.get("scalar_proxy_leading_approximation_retained", False)
    )
    transverse_overlap_weighting_required = bool(
        class_summary.get("transverse_overlap_weighting_required", False)
    )

    directive_explicitly_promotes_quadratic = (
        hit(directive_text, "一次が消えた。次は二次。") is not None
        or hit(directive_text, "Step 4 が核心") is not None
        or hit(directive_text, "K^{\\mu\\nu}[Q]") is not None
    )

    shifted_structure_disposition_sync_ready = bool(
        prior_derivation_ready and prior_classification_ready
    )
    shifted_structure_disposition_sync_honest = bool(
        shifted_structure_disposition_sync_ready
        and isotropic_shift_component_present
        and anisotropic_projector_component_present
        and shifted_structure_selected_under_current_pack
        and not scalar_foundation_exact_supported
        and not transparent_zero_supported
        and scalar_proxy_leading_approximation_retained
        and transverse_overlap_weighting_required
    )
    selected_disposition_case = "case_ii_shifted_structure_under_current_pack"
    nonzero_shifted_structure_retained = bool(
        shifted_structure_disposition_sync_honest
    )
    scalar_proxy_exact_foundation_selected = False
    transparent_zero_selected = False
    retained_scalar_strong_candidate_retained = True
    direct_vector_fixed_q_no_go_retained = True
    effective_source_theorem_revisit_admissible_now = bool(
        shifted_structure_disposition_sync_honest
    )
    observable_dictionary_revisit_admissible_now = False
    future_external_input_side_lane_retained = True
    physical_reject_required = False

    rows = [
        row(
            "prior_derivation_ready",
            "pass" if prior_derivation_ready else "reject",
            "prior quadratic derivation ready",
            truth(prior_derivation_ready),
            "Disposition sync only starts after the frozen-action quadratic core has been derived explicitly.",
        ),
        row(
            "prior_classification_ready",
            "pass" if prior_classification_ready else "reject",
            "prior shifted-structure classification ready",
            truth(prior_classification_ready),
            "Disposition sync only starts after shifted-structure has already been selected honestly.",
        ),
        row(
            "shifted_structure_disposition_sync_ready",
            "pass" if shifted_structure_disposition_sync_ready else "reject",
            "shifted-structure disposition sync ready",
            truth(shifted_structure_disposition_sync_ready),
            "The derivation and blind classification are both complete, so the official disposition can now be synced.",
        ),
        row(
            "shifted_structure_disposition_sync_honest",
            "pass" if shifted_structure_disposition_sync_honest else "reject",
            "shifted-structure disposition sync honest",
            truth(shifted_structure_disposition_sync_honest),
            "The honest current-pack read retains a nonzero shifted operator, not an exact scalar-foundation and not a transparent-zero.",
        ),
        row(
            "nonzero_shifted_structure_retained",
            "pass" if nonzero_shifted_structure_retained else "reject",
            "nonzero shifted-structure retained",
            truth(nonzero_shifted_structure_retained),
            "Both the isotropic shift and the anisotropic projector are retained as a nonzero quadratic lane under the current pack.",
        ),
        row(
            "scalar_proxy_exact_foundation_selected",
            "pass" if scalar_proxy_exact_foundation_selected else "reject",
            "scalar proxy exact foundation selected",
            truth(scalar_proxy_exact_foundation_selected),
            "The current pack still does not justify upgrading the scalar proxy from leading approximation to exact quadratic foundation.",
        ),
        row(
            "scalar_proxy_leading_approximation_retained",
            "pass" if scalar_proxy_leading_approximation_retained else "reject",
            "scalar proxy leading approximation retained",
            truth(scalar_proxy_leading_approximation_retained),
            "The retained scalar strong candidate survives only as a leading-approximation read under the shifted-structure disposition.",
        ),
        row(
            "direct_vector_fixed_q_no_go_retained",
            "pass" if direct_vector_fixed_q_no_go_retained else "reject",
            "direct vector fixed-q no-go retained",
            truth(direct_vector_fixed_q_no_go_retained),
            "The restored exact-vector blind fixed-q no-go is retained and is not overwritten by the quadratic disposition sync.",
        ),
        row(
            "effective_source_theorem_revisit_admissible_now",
            "pass" if effective_source_theorem_revisit_admissible_now else "reject",
            "effective source theorem revisit admissible now",
            truth(effective_source_theorem_revisit_admissible_now),
            "Once shifted-structure has been synced honestly, the next admissible computation is the source-theorem revisit after quadratic disposition.",
        ),
        row(
            "observable_dictionary_revisit_admissible_now",
            "pass" if observable_dictionary_revisit_admissible_now else "reject",
            "observable dictionary revisit admissible now",
            truth(observable_dictionary_revisit_admissible_now),
            "Observable dictionary remains downstream of the source-theorem revisit and therefore is still not active now.",
        ),
        row(
            "future_external_input_side_lane_retained",
            "pass" if future_external_input_side_lane_retained else "reject",
            "future external-input side lane retained",
            truth(future_external_input_side_lane_retained),
            "External expert input remains useful but no longer blocks the computation mainline.",
        ),
        row(
            "directive_explicitly_promotes_quadratic",
            "pass" if directive_explicitly_promotes_quadratic else "watch",
            "directive explicitly promotes quadratic route",
            truth(directive_explicitly_promotes_quadratic),
            "The quadratic directive remains consistent with retaining shifted-structure rather than reopening text-search loops.",
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
            "prior_derivation_gate": display_path(PRIOR_DERIV_GATE),
            "prior_classification_gate": display_path(PRIOR_CLASS_GATE),
        },
        "constants": {
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "downstream_dictionary_route_name": DOWNSTREAM_DICTIONARY_ROUTE_NAME,
            "downstream_dictionary_route": DOWNSTREAM_DICTIONARY_ROUTE,
            "side_lane_name": SIDE_LANE_NAME,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "quadratic_operator_core_derived": deriv_summary.get(
            "quadratic_operator_core_derived", False
        ),
        "background_dependent_quadratic_core_available": deriv_summary.get(
            "background_dependent_quadratic_core_available", False
        ),
        "isotropic_shift_component_present": isotropic_shift_component_present,
        "anisotropic_projector_component_present": anisotropic_projector_component_present,
        "scalar_foundation_exact_supported": scalar_foundation_exact_supported,
        "shifted_structure_selected_under_current_pack": shifted_structure_selected_under_current_pack,
        "transparent_zero_supported": transparent_zero_supported,
        "shifted_structure_disposition_sync_ready": shifted_structure_disposition_sync_ready,
        "shifted_structure_disposition_sync_honest": shifted_structure_disposition_sync_honest,
        "selected_disposition_case": selected_disposition_case,
        "nonzero_shifted_structure_retained": nonzero_shifted_structure_retained,
        "scalar_proxy_exact_foundation_selected": scalar_proxy_exact_foundation_selected,
        "transparent_zero_selected": transparent_zero_selected,
        "scalar_proxy_leading_approximation_retained": scalar_proxy_leading_approximation_retained,
        "transverse_overlap_weighting_required": transverse_overlap_weighting_required,
        "retained_scalar_strong_candidate_retained": retained_scalar_strong_candidate_retained,
        "direct_vector_fixed_q_no_go_retained": direct_vector_fixed_q_no_go_retained,
        "effective_source_theorem_revisit_admissible_now": effective_source_theorem_revisit_admissible_now,
        "observable_dictionary_revisit_admissible_now": observable_dictionary_revisit_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "downstream_dictionary_route_name": DOWNSTREAM_DICTIONARY_ROUTE_NAME,
        "downstream_dictionary_route_or_none": DOWNSTREAM_DICTIONARY_ROUTE,
        "future_external_input_side_lane_retained": future_external_input_side_lane_retained,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": shifted_structure_disposition_sync_honest,
        "next_required_artifacts": [
            NEXT_ROUTE_NAME,
            DOWNSTREAM_DICTIONARY_ROUTE_NAME,
        ],
    }

    evidence = {
        "formulas": {
            "retained_working_action": deriv_formulas["retained_working_action"],
            "quadratic_collect": "(1/2) a_mu K^{mu nu}[Q] a_nu",
            "hessian_form": deriv_formulas["hessian_form"],
            "operator_split": deriv_formulas["operator_split"],
            "disposition_rule": (
                "When the explicit quadratic core is symbolically nonzero, contains both the "
                "isotropic shift and anisotropic projector, and still does not justify exact scalar "
                "foundation, the honest official disposition is shifted-structure under the current pack."
            ),
        },
        "hits": {
            "directive_quadratic": hit(directive_text, "一次が消えた。次は二次。"),
            "directive_case_ii": hit(directive_text, "Case II: ΔK の構造は出るが数値が異なる"),
            "current_problem_disposition": hit(
                current_problem_text, "quadratic operator disposition sync"
            ),
            "current_status_disposition": hit(
                current_status_text, "quadratic operator disposition sync"
            ),
            "unified_roadmap_disposition": hit(
                unified_roadmap_text, "quadratic operator disposition sync"
            ),
            "part5_disposition": hit(part5_text, "quadratic operator disposition sync"),
        },
        "support_counts": {
            "nonzero_shift_component_count": 2.0,
            "exact_scalar_foundation_support_count": 0.0,
            "transparent_zero_support_count": 0.0,
            "downstream_reopen_lane_count": 2.0,
        },
        "retained_numeric_state": {
            "scalar_F_exact_at_q_theory": 0.2998913524347805,
            "scalar_alpha_exact_at_q_theory": 0.00715678583937324,
            "vector_F_at_q_theory": -0.083735013520183,
            "vector_alpha_at_q_theory": 0.0005579616187042394,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
        "classification_summary": class_evidence.get("support_counts", {}),
    }

    inventory_paths = write_artifact(
        "inventory",
        payload("8.7.56.1583", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence),
    )
    audit_paths = write_artifact(
        "audit",
        payload("8.7.56.1584", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence),
    )
    gate_paths = write_artifact(
        "declaration_gate",
        payload(
            "8.7.56.1585",
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
        payload("8.7.56.1586", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence),
    )

    print("[ok] quadratic operator disposition-sync artifacts written:")
    print(f" - {inventory_paths['json']}")
    print(f" - {audit_paths['json']}")
    print(f" - {gate_paths['json']}")
    print(f" - {route_paths['json']}")


if __name__ == "__main__":
    main()
