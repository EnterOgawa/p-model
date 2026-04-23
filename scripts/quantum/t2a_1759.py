#!/usr/bin/env python3
"""Generate 8.7.56.1759-.1762 positive internal-impedance decision-gate artifacts.

This branch closes the positive/passive internal-Hamiltonian constitutive family
after `.1755-.1758` proves the hard theorem

    0 < alpha_F,can^(Z)(q) < q^2 / (4 pi)

under the retained one-leg field-strength theorem. Since the retained scalar
strong candidate exceeds that bound, the honest next move is not another
same-level constitutive retry but a route reset toward a genuinely new mixed
source / internal-Hamiltonian surface.
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

THEOREM_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1755_1758_int_ham_impedance_theorem_declaration_gate_metrics.json"
)
FIELD_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1739_1742_field_strength_recompute_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1759-1762"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor positive internal-"
    "Hamiltonian impedance decision gate / route reset"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "int_ham_pos_impedance_gate",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_positive_internal_impedance_bound_blocks_exact_"
    "scalar_promotion_decision_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_positive_internal_impedance_family_closed_mixed_"
    "source_internal_surface_reactivation_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_mixed_source_internal_"
    "hamiltonian_surface_reactivation"
)
NEXT_ROUTE = "8.7.56.1763"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_mixed_source_internal_"
    "hamiltonian_eigenchannel_theorem_derivation"
)
FOLLOWUP_ROUTE = "8.7.56.1767"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input file is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON を読み込む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: repo 相対の表示パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


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


# 関数: closeout の式を返す。

def build_formulae() -> dict[str, str]:
    """Return the decision-gate formulas."""
    return {
        "positive_family_bound": "0 < alpha_F,can^(Z)(q) < q^2 / (4 pi)",
        "saturation_point": "alpha_F,can(q_theory) = 0.004696068876801584, q_theory^2/(4 pi) = 0.0046980922402105815",
        "route_reset_rule": "If alpha_scalar(q_theory) > q_theory^2/(4 pi), no positive/passive constitutive retry can close the gap.",
        "next_surface": "Introduce a mixed source / internal-Hamiltonian channel rather than another multiplicative Z_T rescaling.",
    }


# 関数: `.1759-.1762` を実行する。

def main() -> None:
    """Execute the positive internal-impedance decision gate branch."""
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
        THEOREM_GATE,
        FIELD_GATE,
    ):
        require(path)

    theorem_summary = read_json(THEOREM_GATE)["summary"]
    field_gate = read_json(FIELD_GATE)
    field_summary = field_gate["summary"]
    field_constants = field_gate["inputs"]["constants"]

    family_closed = bool(
        theorem_summary["positive_transverse_impedance_family_defined"]
        and theorem_summary["monotone_positive_family_bound_derived"]
        and theorem_summary["scalar_exceeds_positive_alpha_bound"]
        and theorem_summary["scalar_exceeds_positive_amplitude_bound"]
        and not theorem_summary["exact_scalar_promotion_available_under_positive_family"]
        and not theorem_summary["required_positive_impedance_for_scalar_candidate_exists"]
    )
    same_level_positive_retry_admissible = False
    route_reset_required = bool(family_closed)
    mixed_source_internal_surface_required = bool(route_reset_required)
    mixed_source_internal_surface_admissible_now = bool(mixed_source_internal_surface_required)
    gate_c_reject_selected = False
    gate_b_partial_retain_selected = True
    physical_reject_not_selected = True
    route_sync_honest = all(
        (
            family_closed,
            route_reset_required,
            mixed_source_internal_surface_required,
            mixed_source_internal_surface_admissible_now,
            not same_level_positive_retry_admissible,
            physical_reject_not_selected,
        )
    )

    rows = [
        row(
            "positive_internal_impedance_family_closed",
            "pass" if family_closed else "reject",
            "positive internal-Hamiltonian impedance family closed",
            truth(family_closed),
            "The theorem-level bound already closes the entire positive/passive constitutive family under the retained one-leg field-strength theorem.",
        ),
        row(
            "same_level_positive_retry_admissible",
            "reject",
            "same-level positive constitutive retry admissible",
            truth(same_level_positive_retry_admissible),
            "No same-level positive/passive Z_T retry remains after the bound has been proved and saturated numerically.",
        ),
        row(
            "route_reset_required",
            "pass" if route_reset_required else "reject",
            "route reset required",
            truth(route_reset_required),
            "Because the scalar target lies outside the positive-family saturation bound, the next honest move is a route reset rather than a recomputation.",
        ),
        row(
            "mixed_source_internal_surface_required",
            "pass" if mixed_source_internal_surface_required else "reject",
            "mixed source / internal-Hamiltonian surface required",
            truth(mixed_source_internal_surface_required),
            "The missing bridge must change the source/channel structure itself, not just rescale the old internal constitutive family.",
        ),
        row(
            "mixed_source_internal_surface_admissible_now",
            "pass" if mixed_source_internal_surface_admissible_now else "reject",
            "mixed source / internal-Hamiltonian surface admissible now",
            truth(mixed_source_internal_surface_admissible_now),
            "The next route is now to formalize a new mixed channel that can bypass the pure |q| saturation bound.",
        ),
        row(
            "gate_b_partial_retain_selected",
            "pass" if gate_b_partial_retain_selected else "reject",
            "gate b partial retain selected",
            truth(gate_b_partial_retain_selected),
            "The field-strength canonical read remains valuable partial evidence even though exact scalar promotion fails.",
        ),
        row(
            "gate_c_reject_selected",
            "reject",
            "gate c reject selected",
            truth(gate_c_reject_selected),
            "The scalar strong candidate is not rejected; only the positive/passive internal constitutive family is closed.",
        ),
        row(
            "physical_reject_not_selected",
            "pass" if physical_reject_not_selected else "reject",
            "physical reject not selected",
            truth(physical_reject_not_selected),
            "The route reset localizes the missing surface without rejecting the retained scalar-side signal.",
        ),
        row(
            "route_sync_honest",
            "pass" if route_sync_honest else "reject",
            "positive internal-impedance route reset honest",
            truth(route_sync_honest),
            "The closeout is honest only if it freezes the pure positive/passive family and points the roadmap to a genuinely new mixed channel.",
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
            "long_roadmap": display_path(LONG_ROADMAP),
            "part5": display_path(PART5),
            "theorem_gate": display_path(THEOREM_GATE),
            "field_gate": display_path(FIELD_GATE),
        },
        "constants": {
            "field_strength_alpha_at_q_theory": field_summary["updated_field_strength_alpha_at_q_theory"],
            "scalar_alpha_exact_at_q_theory": field_constants["scalar_alpha_exact_at_q_theory"],
            "positive_family_alpha_upper_bound_at_q_theory": theorem_summary["positive_family_alpha_upper_bound_at_q_theory"],
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "positive_internal_impedance_family_closed": family_closed,
        "same_level_positive_retry_admissible": same_level_positive_retry_admissible,
        "route_reset_required": route_reset_required,
        "mixed_source_internal_surface_required": mixed_source_internal_surface_required,
        "mixed_source_internal_surface_admissible_now": mixed_source_internal_surface_admissible_now,
        "gate_b_partial_retain_selected": gate_b_partial_retain_selected,
        "gate_c_reject_selected": gate_c_reject_selected,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": route_sync_honest,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": build_formulae(),
        "carry_over": {
            "theorem_summary": theorem_summary,
            "field_summary": field_summary,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload("8.7.56.1759", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence),
        ),
        "audit": write_artifact(
            "audit",
            payload("8.7.56.1760", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload("8.7.56.1761", f"{STEP_NAME} declaration gate", inputs, rows, summary, decision, evidence),
        ),
        "route_sync": write_artifact(
            "route_sync",
            payload("8.7.56.1762", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence),
        ),
    }

    print(
        json.dumps(
            {"step": STEP_TAG, "stem": STEM, "manifest": manifest, "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
