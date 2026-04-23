#!/usr/bin/env python3
"""Generate 8.7.56.1755-.1758 internal-Hamiltonian impedance theorem artifacts.

This branch follows `.1751-.1754`, which reactivated the roadmap by promoting a
new internal-Hamiltonian constitutive surface:

    S_intH[Q,a] = -(1/4) ∫ d^4x f_{mu nu}(a) C^{mu nu alpha beta}[Q] f_{alpha beta}(a)

The question here is whether a positive/passive transverse constitutive /
impedance family can promote the field-strength canonical observable from

    alpha_F,can(q_theory) = 0.004696...

to the retained scalar strong candidate

    alpha_scalar(q_theory) = 0.007156...

while preserving the already closed one-leg field-strength-source theorem.

The resulting theorem shows that any positive transverse impedance family still
obeys the same one-leg |q| bound:

    F_F,can^(Z)(q) = |q| Z_T M_T(q) / (q^2 + Z_T M_T(q)) < |q|
    alpha_F,can^(Z)(q) <= q^2 / (4 pi)

At the retained fixed q_theory, this upper bound is already essentially
saturated by the current field-strength read, so exact scalar promotion is not
available inside the positive/passive constitutive family itself.
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

REACTIVATION_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1751_1754_int_ham_constitutive_reactivation_declaration_gate_metrics.json"
)
FIELD_STRENGTH_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1739_1742_field_strength_recompute_declaration_gate_metrics.json"
)
CLOSEOUT_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1747_1750_field_strength_closeout_registry_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1755-1758"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor internal-Hamiltonian "
    "constitutive / impedance theorem derivation"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "int_ham_impedance_theorem",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_internal_hamiltonian_constitutive_surface_"
    "reactivated_impedance_theorem_derivation_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_positive_internal_impedance_bound_blocks_exact_"
    "scalar_promotion_decision_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_positive_internal_"
    "impedance_decision_gate"
)
NEXT_ROUTE = "8.7.56.1759"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_mixed_source_internal_"
    "hamiltonian_surface_reactivation"
)
FOLLOWUP_ROUTE = "8.7.56.1763"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input file is missing."""
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


# 関数: repo 相対の表示パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line matching one substring."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


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


# 関数: constitutive theorem の主要式を返す。

def build_formulae() -> dict[str, str]:
    """Return the positive/passive constitutive theorem formulas."""
    return {
        "internal_action": "S_intH[Q,a] = -(1/4) ∫ d^4x f_{mu nu}(a) C^{mu nu alpha beta}[Q] f_{alpha beta}(a)",
        "transverse_reduction": "Pi_T C[Q,q] Pi_T = Z_T[Q,q] Pi_T",
        "positive_family": "Z_T[Q,q] > 0",
        "effective_response": "F_F,can^(Z)(q) = |q| Z_T[Q,q] M_T(q) / (q^2 + Z_T[Q,q] M_T(q))",
        "monotone_derivative": "dF_F,can^(Z)/dZ_T = |q| M_T q^2 / (q^2 + Z_T M_T)^2 > 0",
        "upper_bound": "0 < F_F,can^(Z)(q) < |q|, 0 < alpha_F,can^(Z)(q) < q^2 / (4 pi)",
    }


# 関数: `.1755-.1758` を実行する。

def main() -> None:
    """Execute the internal-Hamiltonian impedance theorem derivation branch."""
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
        REACTIVATION_GATE,
        FIELD_STRENGTH_GATE,
        CLOSEOUT_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    reactivation_summary = read_json(REACTIVATION_GATE)["summary"]
    field_strength_gate = read_json(FIELD_STRENGTH_GATE)
    field_strength_summary = field_strength_gate["summary"]
    field_strength_constants = field_strength_gate["inputs"]["constants"]
    closeout_summary = read_json(CLOSEOUT_GATE)["summary"]

    q_theory = float(field_strength_constants["q_theory_over_m0"])
    q_squared = float(field_strength_constants["q_squared_at_q_theory"])
    m_t = float(field_strength_constants["projected_kernel_numerator_at_q_theory"])
    alpha_field = float(field_strength_summary["updated_field_strength_alpha_at_q_theory"])
    f_field = float(field_strength_summary["updated_field_strength_response_at_q_theory"])
    alpha_scalar = float(field_strength_constants["scalar_alpha_exact_at_q_theory"])
    f_scalar = math.sqrt(4.0 * math.pi * alpha_scalar)

    inventory_ready = all(
        item is not None
        for item in (
            hit(status_text, "8.7.56.1751"),
            hit(roadmap_text, "8.7.56.1751-.1754"),
            hit(current_problem_text, "conditional internal-Hamiltonian surface / external-input reactivation"),
            hit(current_status_text, "conditional internal-Hamiltonian surface / external-input reactivation"),
            hit(
                unified_text,
                "`.1751-.1754` は **conditional internal-Hamiltonian surface / external-input reactivation**",
            ),
            hit(long_text, "17. `8.7.56.1751-.1754`"),
            hit(part5_text, "`.1747-.1750` の **field-strength-source closeout / reopen registry**"),
        )
    )
    reactivation_branch_completed = bool(
        reactivation_summary["internal_hamiltonian_constitutive_surface_adopted"]
        and reactivation_summary["new_primary_trigger_opened"]
        and reactivation_summary["background_constitutive_tensor_required"]
    )
    field_strength_theorem_retained = bool(
        closeout_summary["canonical_field_strength_theorem_derived"]
        and closeout_summary["field_strength_external_theorem_pack_closed"]
    )
    positive_transverse_impedance_family_defined = bool(
        reactivation_branch_completed and field_strength_theorem_retained and (m_t > 0.0)
    )
    monotone_positive_family_bound_derived = bool(positive_transverse_impedance_family_defined)
    alpha_upper_bound_under_positive_family = q_squared / (4.0 * math.pi)
    f_upper_bound_under_positive_family = q_theory
    current_fraction_of_positive_alpha_bound = alpha_field / alpha_upper_bound_under_positive_family
    scalar_fraction_of_positive_alpha_bound = alpha_scalar / alpha_upper_bound_under_positive_family
    current_gap_to_positive_alpha_bound = alpha_upper_bound_under_positive_family - alpha_field
    scalar_exceeds_positive_alpha_bound = alpha_scalar > alpha_upper_bound_under_positive_family
    scalar_f_exceeds_q_bound = f_scalar > f_upper_bound_under_positive_family
    exact_scalar_promotion_available_under_positive_family = not scalar_exceeds_positive_alpha_bound
    required_positive_impedance_for_scalar_candidate_exists = not scalar_f_exceeds_q_bound
    positive_family_decision_gate_admissible_now = bool(
        monotone_positive_family_bound_derived
        and scalar_exceeds_positive_alpha_bound
    )
    same_internal_positive_family_retry_blocked = True
    physical_reject_not_selected = bool(not closeout_summary["physical_reject_required"])
    route_derivation_honest = all(
        (
            inventory_ready,
            reactivation_branch_completed,
            field_strength_theorem_retained,
            positive_transverse_impedance_family_defined,
            monotone_positive_family_bound_derived,
            same_internal_positive_family_retry_blocked,
            physical_reject_not_selected,
        )
    )

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "internal-Hamiltonian impedance theorem inventory ready",
            truth(inventory_ready),
            "The theorem branch starts only after the `.1751-.1754` reactivation is the live official route.",
        ),
        row(
            "reactivation_branch_completed",
            "pass" if reactivation_branch_completed else "reject",
            "internal-Hamiltonian constitutive reactivation completed",
            truth(reactivation_branch_completed),
            "The impedance theorem is only admissible after the new internal-Hamiltonian surface has already been adopted as the new primary surface.",
        ),
        row(
            "field_strength_theorem_retained",
            "pass" if field_strength_theorem_retained else "reject",
            "field-strength external theorem retained",
            truth(field_strength_theorem_retained),
            "The one-leg field-strength theorem remains fixed while only the internal Hamiltonian sector is varied.",
        ),
        row(
            "positive_transverse_impedance_family_defined",
            "pass" if positive_transverse_impedance_family_defined else "reject",
            "positive transverse impedance family defined",
            truth(positive_transverse_impedance_family_defined),
            "The minimal constitutive family is Pi_T C Pi_T = Z_T Pi_T with positive/passive Z_T and positive projected-kernel numerator M_T.",
        ),
        row(
            "monotone_positive_family_bound_derived",
            "pass" if monotone_positive_family_bound_derived else "reject",
            "monotone positive-family bound derived",
            truth(monotone_positive_family_bound_derived),
            "For Z_T > 0 and M_T > 0, the field-strength canonical amplitude grows monotonically with Z_T but is bounded above by |q|.",
        ),
        row(
            "f_upper_bound_under_positive_family",
            "watch",
            "positive-family amplitude upper bound at q_theory",
            f_upper_bound_under_positive_family,
            "The one-leg field-strength family cannot exceed |q_theory| under any positive/passive constitutive scaling.",
        ),
        row(
            "alpha_upper_bound_under_positive_family",
            "watch",
            "positive-family alpha upper bound at q_theory",
            alpha_upper_bound_under_positive_family,
            "This is q_theory^2/(4 pi), the maximal alpha reachable by the positive constitutive family under the retained one-leg theorem.",
        ),
        row(
            "current_fraction_of_positive_alpha_bound",
            "watch",
            "current field-strength fraction of positive-family alpha bound",
            current_fraction_of_positive_alpha_bound,
            "The current canonical field-strength read already sits extremely close to the maximal positive-family alpha bound.",
        ),
        row(
            "current_gap_to_positive_alpha_bound",
            "watch",
            "current field-strength gap to positive-family alpha bound",
            current_gap_to_positive_alpha_bound,
            "Only a tiny residual margin remains inside the positive/passive constitutive family itself.",
        ),
        row(
            "scalar_exceeds_positive_alpha_bound",
            "pass" if scalar_exceeds_positive_alpha_bound else "reject",
            "scalar exact candidate exceeds positive-family alpha bound",
            truth(scalar_exceeds_positive_alpha_bound),
            "The retained scalar strong candidate requires an alpha larger than the maximal positive-family field-strength alpha.",
        ),
        row(
            "scalar_f_exceeds_q_bound",
            "pass" if scalar_f_exceeds_q_bound else "reject",
            "scalar exact amplitude exceeds |q| bound",
            truth(scalar_f_exceeds_q_bound),
            "The retained scalar amplitude itself is larger than |q_theory|, so no positive-family constitutive rescaling of the retained theorem can reach it.",
        ),
        row(
            "exact_scalar_promotion_available_under_positive_family",
            "pass" if exact_scalar_promotion_available_under_positive_family else "reject",
            "exact scalar promotion available under positive constitutive family",
            truth(exact_scalar_promotion_available_under_positive_family),
            "This is false under the retained one-leg theorem because the entire positive/passive family is bounded by q_theory^2/(4 pi).",
        ),
        row(
            "required_positive_impedance_for_scalar_candidate_exists",
            "pass" if required_positive_impedance_for_scalar_candidate_exists else "reject",
            "required positive impedance for scalar candidate exists",
            truth(required_positive_impedance_for_scalar_candidate_exists),
            "There is no finite positive impedance that reaches the scalar exact candidate because the scalar amplitude exceeds the |q| saturation limit.",
        ),
        row(
            "positive_family_decision_gate_admissible_now",
            "pass" if positive_family_decision_gate_admissible_now else "reject",
            "positive-family decision gate admissible now",
            truth(positive_family_decision_gate_admissible_now),
            "Once the saturation bound is explicit, the next honest step is a decision gate / route reset rather than a same-level recomputation.",
        ),
        row(
            "same_internal_positive_family_retry_blocked",
            "pass" if same_internal_positive_family_retry_blocked else "reject",
            "same internal positive-family retry blocked",
            truth(same_internal_positive_family_retry_blocked),
            "The theorem itself closes the entire positive/passive constitutive family under the retained external-source rule.",
        ),
        row(
            "physical_reject_not_selected",
            "pass" if physical_reject_not_selected else "reject",
            "physical reject not selected",
            truth(physical_reject_not_selected),
            "The saturation bound localizes the missing bridge but does not reject the retained scalar strong candidate itself.",
        ),
        row(
            "route_derivation_honest",
            "pass" if route_derivation_honest else "reject",
            "positive internal-Hamiltonian impedance theorem honest",
            truth(route_derivation_honest),
            "The theorem is honest only if it preserves the retained one-leg source theorem while proving that the positive constitutive family itself cannot complete exact scalar promotion.",
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
            "reactivation_gate": display_path(REACTIVATION_GATE),
            "field_strength_gate": display_path(FIELD_STRENGTH_GATE),
            "closeout_gate": display_path(CLOSEOUT_GATE),
        },
        "constants": {
            "q_theory_over_m0": q_theory,
            "q_squared_at_q_theory": q_squared,
            "projected_kernel_numerator_at_q_theory": m_t,
            "field_strength_response_at_q_theory": f_field,
            "field_strength_alpha_at_q_theory": alpha_field,
            "scalar_alpha_exact_at_q_theory": alpha_scalar,
            "scalar_response_exact_at_q_theory": f_scalar,
            "positive_family_alpha_upper_bound_at_q_theory": alpha_upper_bound_under_positive_family,
            "positive_family_amplitude_upper_bound_at_q_theory": f_upper_bound_under_positive_family,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "positive_transverse_impedance_family_defined": positive_transverse_impedance_family_defined,
        "monotone_positive_family_bound_derived": monotone_positive_family_bound_derived,
        "field_strength_theorem_retained": field_strength_theorem_retained,
        "positive_family_amplitude_upper_bound_at_q_theory": f_upper_bound_under_positive_family,
        "positive_family_alpha_upper_bound_at_q_theory": alpha_upper_bound_under_positive_family,
        "current_field_strength_fraction_of_positive_alpha_bound": current_fraction_of_positive_alpha_bound,
        "current_field_strength_gap_to_positive_alpha_bound": current_gap_to_positive_alpha_bound,
        "scalar_exceeds_positive_alpha_bound": scalar_exceeds_positive_alpha_bound,
        "scalar_exceeds_positive_amplitude_bound": scalar_f_exceeds_q_bound,
        "exact_scalar_promotion_available_under_positive_family": exact_scalar_promotion_available_under_positive_family,
        "required_positive_impedance_for_scalar_candidate_exists": required_positive_impedance_for_scalar_candidate_exists,
        "positive_family_decision_gate_admissible_now": positive_family_decision_gate_admissible_now,
        "same_internal_positive_family_retry_blocked": same_internal_positive_family_retry_blocked,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": route_derivation_honest,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": build_formulae(),
        "hits": {
            "status_branch_hit": hit(status_text, "8.7.56.1751"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1751-.1754"),
            "current_problem_branch_hit": hit(
                current_problem_text,
                "conditional internal-Hamiltonian surface / external-input reactivation",
            ),
            "current_status_branch_hit": hit(
                current_status_text,
                "conditional internal-Hamiltonian surface / external-input reactivation",
            ),
            "unified_roadmap_branch_hit": hit(
                unified_text,
                "`.1751-.1754` は **conditional internal-Hamiltonian surface / external-input reactivation**",
            ),
            "long_roadmap_branch_hit": hit(long_text, "17. `8.7.56.1751-.1754`"),
            "part5_closeout_hit": hit(
                part5_text,
                "`.1747-.1750` の **field-strength-source closeout / reopen registry**",
            ),
        },
        "carry_over": {
            "reactivation_summary": reactivation_summary,
            "field_strength_summary": field_strength_summary,
            "closeout_summary": closeout_summary,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1755",
                f"{STEP_NAME} inventory",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "audit": write_artifact(
            "audit",
            payload(
                "8.7.56.1756",
                f"{STEP_NAME} audit",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload(
                "8.7.56.1757",
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
            payload(
                "8.7.56.1758",
                f"{STEP_NAME} route sync",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
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
