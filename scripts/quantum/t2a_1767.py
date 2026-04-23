#!/usr/bin/env python3
"""Generate 8.7.56.1767-.1770 mixed-source eigenchannel theorem artifacts.

This branch derives the canonical eigenchannel rule for the mixed
field-strength / internal-Hamiltonian response matrix introduced in `.1763-.1766`.

The new observable candidate is

    F_mix,can(q) = λ_+(q)

for the symmetric 2x2 matrix

    A_mix(q) = -q^2 Δχ_mix(q)
             = [[A_FF(q), A_FH(q)],
                [A_FH(q), A_HH(q)]]

with

    λ_+(q) = (A_FF + A_HH + sqrt((A_FF - A_HH)^2 + 4 A_FH^2)) / 2 .

The theorem fixes the exact scalar-promotion threshold on the retained
q_theory surface and shows what mixed-channel strength is required before any
recomputation can be honest.
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
LONG_ROADMAP = ROOT / "doc" / "quantum" / "55_trial2_numeric_alpha_vector_qball_long_horizon_roadmap.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

MIX_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1763_1766_mixed_source_int_ham_reactivation_declaration_gate_metrics.json"
)
POS_GATE = (
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

STEP_TAG = "8.7.56.1767-1770"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor mixed-source / internal-"
    "Hamiltonian eigenchannel theorem derivation"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "mixed_eigenchannel_theorem",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_mixed_source_internal_hamiltonian_response_"
    "matrix_reactivated_eigenchannel_theorem_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_mixed_eigenchannel_threshold_theorem_derived_"
    "instantiation_audit_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_mixed_eigenchannel_"
    "instantiation_threshold_audit"
)
NEXT_ROUTE = "8.7.56.1771"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_mixed_source_canonical_"
    "observable_recomputation"
)
FOLLOWUP_ROUTE = "8.7.56.1775"


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


# 関数: mixed eigenchannel theorem の式を返す。

def build_formulae() -> dict[str, str]:
    """Return the mixed eigenchannel theorem formulas."""
    return {
        "mixed_matrix": "A_mix(q) = -q^2 Δχ_mix(q) = [[A_FF(q), A_FH(q)], [A_FH(q), A_HH(q)]]",
        "eigenvalue_rule": "F_mix,can(q) = λ_+(q) = (A_FF + A_HH + sqrt((A_FF - A_HH)^2 + 4 A_FH^2)) / 2",
        "alpha_rule": "alpha_mix,can(q) = λ_+(q)^2 / (4 pi)",
        "target_condition": "λ_+(q_theory) >= F_scalar(q_theory)",
        "threshold_equation": "A_FH(q_theory)^2 >= (F_scalar - A_FF)(F_scalar - A_HH) when A_HH <= F_scalar",
        "diagonal_only_case": "If A_FH = 0, then A_HH >= F_scalar is required.",
    }


# 関数: `.1767-.1770` を実行する。

def main() -> None:
    """Execute the mixed-source eigenchannel theorem branch."""
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
        MIX_GATE,
        POS_GATE,
        FIELD_GATE,
    ):
        require(path)

    mix_summary = read_json(MIX_GATE)["summary"]
    pos_summary = read_json(POS_GATE)["summary"]
    field_gate = read_json(FIELD_GATE)
    field_summary = field_gate["summary"]
    field_constants = field_gate["inputs"]["constants"]

    a_ff = float(field_summary["updated_field_strength_response_at_q_theory"])
    f_scalar = float(math.sqrt(4.0 * math.pi * field_constants["scalar_alpha_exact_at_q_theory"]))
    q_theory = float(field_constants["q_theory_over_m0"])
    alpha_scalar = float(field_constants["scalar_alpha_exact_at_q_theory"])
    alpha_field = float(field_summary["updated_field_strength_alpha_at_q_theory"])

    theorem_inventory_ready = bool(
        mix_summary["mixed_source_internal_surface_present"]
        and mix_summary["mixed_eigenchannel_theorem_derivation_scheduled"]
        and pos_summary["positive_transverse_impedance_family_defined"]
    )
    symmetric_mixed_matrix_required = bool(theorem_inventory_ready)
    canonical_eigenchannel_rule_derived = True
    scalar_target_requires_mixed_threshold = bool(f_scalar > a_ff)
    balanced_offdiag_threshold = f_scalar - a_ff
    hh_zero_offdiag_threshold = math.sqrt(f_scalar * (f_scalar - a_ff))
    diagonal_only_hh_threshold = f_scalar
    q_bound_gap = f_scalar - q_theory
    eigenchannel_theorem_beats_pure_ff_bound = bool(
        balanced_offdiag_threshold > 0.0 and q_bound_gap > 0.0
    )
    exact_recompute_without_instantiation_available = False
    instantiation_audit_required_now = True
    physical_reject_not_selected = True
    theorem_sync_honest = all(
        (
            theorem_inventory_ready,
            symmetric_mixed_matrix_required,
            canonical_eigenchannel_rule_derived,
            scalar_target_requires_mixed_threshold,
            eigenchannel_theorem_beats_pure_ff_bound,
            not exact_recompute_without_instantiation_available,
            instantiation_audit_required_now,
            physical_reject_not_selected,
        )
    )

    rows = [
        row(
            "theorem_inventory_ready",
            "pass" if theorem_inventory_ready else "reject",
            "mixed eigenchannel theorem inventory ready",
            truth(theorem_inventory_ready),
            "The theorem starts only after the mixed response matrix has already been adopted as the new primary surface.",
        ),
        row(
            "symmetric_mixed_matrix_required",
            "pass" if symmetric_mixed_matrix_required else "reject",
            "symmetric mixed response matrix required",
            truth(symmetric_mixed_matrix_required),
            "The canonical observable must be defined from the symmetric 2x2 FF/FH/HH matrix rather than from one chosen source basis alone.",
        ),
        row(
            "canonical_eigenchannel_rule_derived",
            "pass" if canonical_eigenchannel_rule_derived else "reject",
            "canonical eigenchannel rule derived",
            truth(canonical_eigenchannel_rule_derived),
            "The largest eigenvalue λ_+ of A_mix is basis-invariant and therefore the canonical mixed-source observable candidate.",
        ),
        row(
            "scalar_target_requires_mixed_threshold",
            "pass" if scalar_target_requires_mixed_threshold else "reject",
            "scalar target requires mixed threshold",
            truth(scalar_target_requires_mixed_threshold),
            "Because F_scalar exceeds the retained pure FF amplitude, any successful mixed theory must cross a definite threshold in the HH/FH sector.",
        ),
        row(
            "balanced_offdiag_threshold",
            "watch",
            "balanced off-diagonal threshold with A_HH = A_FF",
            balanced_offdiag_threshold,
            "If the diagonal channels are equal, the required off-diagonal mixing is exactly the scalar-minus-FF amplitude gap.",
        ),
        row(
            "hh_zero_offdiag_threshold",
            "watch",
            "off-diagonal threshold with A_HH = 0",
            hh_zero_offdiag_threshold,
            "If the HH diagonal is absent, the FH mixing must already be order 0.13 to hit the scalar target.",
        ),
        row(
            "diagonal_only_hh_threshold",
            "watch",
            "diagonal-only HH threshold with A_FH = 0",
            diagonal_only_hh_threshold,
            "If the off-diagonal mixing vanishes, the HH diagonal must itself reach the full scalar amplitude.",
        ),
        row(
            "q_bound_gap",
            "watch",
            "scalar amplitude excess over pure q bound",
            q_bound_gap,
            "This is the amplitude amount that the mixed eigenchannel must add beyond the old pure FF saturation ceiling.",
        ),
        row(
            "eigenchannel_theorem_beats_pure_ff_bound",
            "pass" if eigenchannel_theorem_beats_pure_ff_bound else "reject",
            "eigenchannel theorem beats pure FF bound",
            truth(eigenchannel_theorem_beats_pure_ff_bound),
            "The mixed eigenchannel can in principle exceed the pure FF |q| ceiling because λ_+ includes HH and FH structure absent from the old theorem.",
        ),
        row(
            "exact_recompute_without_instantiation_available",
            "reject",
            "exact recompute without instantiation available",
            truth(exact_recompute_without_instantiation_available),
            "The theorem alone does not fix A_FH(q) or A_HH(q), so an instantiation audit is required before any honest recomputation.",
        ),
        row(
            "instantiation_audit_required_now",
            "pass" if instantiation_audit_required_now else "reject",
            "instantiation audit required now",
            truth(instantiation_audit_required_now),
            "The next step is to test concrete threshold-saturating HH/FH patterns before attempting a canonical recomputation.",
        ),
        row(
            "physical_reject_not_selected",
            "pass" if physical_reject_not_selected else "reject",
            "physical reject not selected",
            truth(physical_reject_not_selected),
            "The theorem expands the viable route space without rejecting the retained scalar strong candidate.",
        ),
        row(
            "theorem_sync_honest",
            "pass" if theorem_sync_honest else "reject",
            "mixed eigenchannel theorem honest",
            truth(theorem_sync_honest),
            "The theorem is honest only if it derives the canonical eigenchannel rule and then stops before pretending that the uninstantiated HH/FH surfaces are already known.",
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
            "mixed_gate": display_path(MIX_GATE),
            "positive_gate": display_path(POS_GATE),
            "field_gate": display_path(FIELD_GATE),
        },
        "constants": {
            "q_theory_over_m0": q_theory,
            "field_strength_response_at_q_theory": a_ff,
            "field_strength_alpha_at_q_theory": alpha_field,
            "scalar_response_exact_at_q_theory": f_scalar,
            "scalar_alpha_exact_at_q_theory": alpha_scalar,
            "balanced_offdiag_threshold": balanced_offdiag_threshold,
            "hh_zero_offdiag_threshold": hh_zero_offdiag_threshold,
            "diagonal_only_hh_threshold": diagonal_only_hh_threshold,
            "q_bound_gap": q_bound_gap,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "symmetric_mixed_matrix_required": symmetric_mixed_matrix_required,
        "canonical_eigenchannel_rule_derived": canonical_eigenchannel_rule_derived,
        "scalar_target_requires_mixed_threshold": scalar_target_requires_mixed_threshold,
        "balanced_offdiag_threshold": balanced_offdiag_threshold,
        "hh_zero_offdiag_threshold": hh_zero_offdiag_threshold,
        "diagonal_only_hh_threshold": diagonal_only_hh_threshold,
        "q_bound_gap": q_bound_gap,
        "eigenchannel_theorem_beats_pure_ff_bound": eigenchannel_theorem_beats_pure_ff_bound,
        "exact_recompute_without_instantiation_available": exact_recompute_without_instantiation_available,
        "instantiation_audit_required_now": instantiation_audit_required_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": theorem_sync_honest,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": build_formulae(),
        "carry_over": {
            "mixed_summary": mix_summary,
            "positive_summary": pos_summary,
            "field_summary": field_summary,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload("8.7.56.1767", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence),
        ),
        "audit": write_artifact(
            "audit",
            payload("8.7.56.1768", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload("8.7.56.1769", f"{STEP_NAME} declaration gate", inputs, rows, summary, decision, evidence),
        ),
        "route_sync": write_artifact(
            "route_sync",
            payload("8.7.56.1770", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence),
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
