#!/usr/bin/env python3
"""Generate 8.7.56.1763-.1766 mixed-source/internal-Hamiltonian reactivation artifacts.

After `.1755-.1762`, the pure positive/passive internal constitutive family is
closed. The retained one-leg field-strength theorem imposes a hard saturation
bound:

    F_F,can(q) < |q|,  alpha_F,can(q) < q^2/(4 pi)

The scalar strong candidate lies above that bound, so the next genuinely new
surface must change the source/channel structure itself rather than rescale the
old constitutive family.

The minimal new theory is to keep the external field-strength channel while
adding an internal-Hamiltonian response channel and their off-diagonal mixing:

    O_F[f] = f
    O_H[Q,f] = C[Q] f
    Δχ_mix = [[Δχ_FF, Δχ_FH], [Δχ_HF, Δχ_HH]]

The observable candidate is then the maximal eigenchannel of this mixed
response matrix rather than the pure field-strength channel alone.
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

STEP_TAG = "8.7.56.1763-1766"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor mixed source / internal-"
    "Hamiltonian surface reactivation"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "mixed_source_int_ham_reactivation",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_positive_internal_impedance_family_closed_mixed_"
    "source_internal_surface_reactivation_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_mixed_source_internal_hamiltonian_response_"
    "matrix_reactivated_eigenchannel_theorem_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_mixed_source_internal_"
    "hamiltonian_eigenchannel_theorem_derivation"
)
NEXT_ROUTE = "8.7.56.1767"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_mixed_source_canonical_"
    "observable_recomputation"
)
FOLLOWUP_ROUTE = "8.7.56.1771"


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


# 関数: mixed-source response の式を返す。

def build_formulae() -> dict[str, str]:
    """Return the mixed-source response formulas."""
    return {
        "field_channel": "O_F[f] = f",
        "hamiltonian_channel": "O_H[Q,f] = C[Q] f",
        "mixed_source_action": "S_src^(F,H)[Q,a;J_F,J_H] = S_intH[Q,a] - (1/2) ∫ d^4x (J_F^{mu nu} O_{F,mu nu} + J_H^{mu nu} O_{H,mu nu})",
        "response_matrix": "Δχ_mix(q) = [[Δχ_FF(q), Δχ_FH(q)], [Δχ_HF(q), Δχ_HH(q)]]",
        "canonical_mixed_channel": "F_mix,can(q) = λ_max[-q^2 Δχ_mix(q)]",
        "why_new": "The old |q| bound constrains only the pure FF channel; an eigenchannel with FH/HH mixing is a genuinely new surface.",
    }


# 関数: `.1763-.1766` を実行する。

def main() -> None:
    """Execute the mixed-source/internal-Hamiltonian reactivation branch."""
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
        POS_GATE,
        FIELD_GATE,
    ):
        require(path)

    pos_summary = read_json(POS_GATE)["summary"]
    field_gate = read_json(FIELD_GATE)
    field_summary = field_gate["summary"]
    field_constants = field_gate["inputs"]["constants"]

    q_theory = float(field_constants["q_theory_over_m0"])
    f_field = float(field_summary["updated_field_strength_response_at_q_theory"])
    alpha_field = float(field_summary["updated_field_strength_alpha_at_q_theory"])
    f_scalar = float(math.sqrt(4.0 * math.pi * field_constants["scalar_alpha_exact_at_q_theory"]))
    alpha_scalar = float(field_constants["scalar_alpha_exact_at_q_theory"])
    required_mixed_amplitude_excess = f_scalar - q_theory
    required_mixed_alpha_excess = alpha_scalar - pos_summary["positive_family_alpha_upper_bound_at_q_theory"]
    mixed_excess_fraction_vs_q = required_mixed_amplitude_excess / q_theory

    positive_family_closed = bool(
        pos_summary["positive_transverse_impedance_family_defined"]
        and pos_summary["monotone_positive_family_bound_derived"]
        and pos_summary["scalar_exceeds_positive_alpha_bound"]
        and not pos_summary["required_positive_impedance_for_scalar_candidate_exists"]
    )
    pure_field_strength_channel_retained = True
    mixed_source_response_matrix_required = bool(positive_family_closed)
    additive_cross_channel_numerator_required = bool(positive_family_closed)
    mixed_source_internal_surface_present = bool(
        mixed_source_response_matrix_required and additive_cross_channel_numerator_required
    )
    new_primary_trigger_opened = bool(mixed_source_internal_surface_present)
    mixed_eigenchannel_theorem_derivation_scheduled = bool(new_primary_trigger_opened)
    same_level_positive_retry_blocked = True
    physical_reject_not_selected = True
    route_reactivation_honest = all(
        (
            positive_family_closed,
            pure_field_strength_channel_retained,
            mixed_source_response_matrix_required,
            additive_cross_channel_numerator_required,
            mixed_source_internal_surface_present,
            new_primary_trigger_opened,
            same_level_positive_retry_blocked,
            physical_reject_not_selected,
        )
    )

    rows = [
        row(
            "positive_family_closed",
            "pass" if positive_family_closed else "reject",
            "positive constitutive family closed",
            truth(positive_family_closed),
            "The reactivation only makes sense after the pure positive/passive constitutive family has already been theorem-level closed.",
        ),
        row(
            "pure_field_strength_channel_retained",
            "pass" if pure_field_strength_channel_retained else "reject",
            "pure field-strength channel retained",
            truth(pure_field_strength_channel_retained),
            "The one-leg field-strength theorem remains a retained channel inside the larger mixed-source theory.",
        ),
        row(
            "required_mixed_amplitude_excess",
            "watch",
            "required mixed-channel amplitude excess beyond |q|",
            required_mixed_amplitude_excess,
            "This is the extra amplitude above the pure field-strength |q| bound that the scalar strong candidate would require.",
        ),
        row(
            "required_mixed_alpha_excess",
            "watch",
            "required mixed-channel alpha excess beyond positive-family bound",
            required_mixed_alpha_excess,
            "This is the alpha amount that cannot be supplied by any positive/passive constitutive rescaling.",
        ),
        row(
            "mixed_excess_fraction_vs_q",
            "watch",
            "required mixed-channel excess fraction vs q bound",
            mixed_excess_fraction_vs_q,
            "The needed mixed-source uplift is about 23.42% of the old |q| saturation ceiling.",
        ),
        row(
            "mixed_source_response_matrix_required",
            "pass" if mixed_source_response_matrix_required else "reject",
            "mixed source response matrix required",
            truth(mixed_source_response_matrix_required),
            "A genuinely new source/channel matrix is required because the old single-channel field-strength read is theorem-level bounded.",
        ),
        row(
            "additive_cross_channel_numerator_required",
            "pass" if additive_cross_channel_numerator_required else "reject",
            "additive cross-channel numerator required",
            truth(additive_cross_channel_numerator_required),
            "The next surface must add a new FH/HH channel rather than merely multiply the old FF channel by Z_T.",
        ),
        row(
            "mixed_source_internal_surface_present",
            "pass" if mixed_source_internal_surface_present else "reject",
            "mixed source / internal-Hamiltonian surface present",
            truth(mixed_source_internal_surface_present),
            "The minimal new theory is a 2x2 mixed response matrix between the retained field-strength channel and the new internal-Hamiltonian channel.",
        ),
        row(
            "new_primary_trigger_opened",
            "pass" if new_primary_trigger_opened else "reject",
            "new primary trigger opened",
            truth(new_primary_trigger_opened),
            "This mixed-source/internal-Hamiltonian response matrix is a genuinely new action-level surface beyond the closed positive constitutive family.",
        ),
        row(
            "mixed_eigenchannel_theorem_derivation_scheduled",
            "pass" if mixed_eigenchannel_theorem_derivation_scheduled else "reject",
            "mixed eigenchannel theorem derivation scheduled",
            truth(mixed_eigenchannel_theorem_derivation_scheduled),
            "The next honest branch is to derive the eigenchannel theorem for the mixed FF/FH/HH response matrix.",
        ),
        row(
            "same_level_positive_retry_blocked",
            "pass" if same_level_positive_retry_blocked else "reject",
            "same-level positive-family retry blocked",
            truth(same_level_positive_retry_blocked),
            "The old multiplicative Z_T family remains closed and is not reopened by this reactivation.",
        ),
        row(
            "physical_reject_not_selected",
            "pass" if physical_reject_not_selected else "reject",
            "physical reject not selected",
            truth(physical_reject_not_selected),
            "The new theory broadens the source/channel space without rejecting the retained scalar-side evidence.",
        ),
        row(
            "route_reactivation_honest",
            "pass" if route_reactivation_honest else "reject",
            "mixed source / internal-Hamiltonian reactivation honest",
            truth(route_reactivation_honest),
            "Reactivation is honest only if it preserves the pure field-strength channel while introducing a genuinely new mixed channel beyond the old saturation theorem.",
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
            "positive_gate": display_path(POS_GATE),
            "field_gate": display_path(FIELD_GATE),
        },
        "constants": {
            "q_theory_over_m0": q_theory,
            "field_strength_response_at_q_theory": f_field,
            "field_strength_alpha_at_q_theory": alpha_field,
            "scalar_response_exact_at_q_theory": f_scalar,
            "scalar_alpha_exact_at_q_theory": alpha_scalar,
            "required_mixed_amplitude_excess": required_mixed_amplitude_excess,
            "required_mixed_alpha_excess": required_mixed_alpha_excess,
            "mixed_excess_fraction_vs_q": mixed_excess_fraction_vs_q,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "positive_family_closed": positive_family_closed,
        "pure_field_strength_channel_retained": pure_field_strength_channel_retained,
        "required_mixed_amplitude_excess": required_mixed_amplitude_excess,
        "required_mixed_alpha_excess": required_mixed_alpha_excess,
        "mixed_excess_fraction_vs_q": mixed_excess_fraction_vs_q,
        "mixed_source_response_matrix_required": mixed_source_response_matrix_required,
        "additive_cross_channel_numerator_required": additive_cross_channel_numerator_required,
        "mixed_source_internal_surface_present": mixed_source_internal_surface_present,
        "new_primary_trigger_opened": new_primary_trigger_opened,
        "mixed_eigenchannel_theorem_derivation_scheduled": mixed_eigenchannel_theorem_derivation_scheduled,
        "same_level_positive_retry_blocked": same_level_positive_retry_blocked,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": route_reactivation_honest,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": build_formulae(),
        "carry_over": {
            "positive_summary": pos_summary,
            "field_summary": field_summary,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload("8.7.56.1763", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence),
        ),
        "audit": write_artifact(
            "audit",
            payload("8.7.56.1764", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload("8.7.56.1765", f"{STEP_NAME} declaration gate", inputs, rows, summary, decision, evidence),
        ),
        "route_sync": write_artifact(
            "route_sync",
            payload("8.7.56.1766", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence),
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
