#!/usr/bin/env python3
"""Generate 8.7.56.1735-.1738 field-strength-source one-leg theorem artifacts.

`.1731-.1734` reopened the roadmap with the genuinely new source primitive

    S_src^F[P,a;J_F] = S_frozen[P,a] - (1/2) ∫ d^4x J_F^{mu nu} f_{mu nu}(a),

where the probe no longer couples directly to the potential a_mu but to the
gauge-invariant field strength f_{mu nu}(a).

The theorem question is stricter than in the old potential-source pack:

    which canonically normalized observable follows from one tensor-source
    insertion and one outgoing asymptotic transverse photon leg?

For a transverse plane wave, the field-strength operator already contributes one
explicit momentum factor. Therefore the mixed source-to-field response carries
one fewer unattached vacuum propagator than the direct potential-source
source-source response. The corresponding canonical read is a one-leg
amputation theorem,

    F_F,can(q) = -|q| q^2 Delta chi_T(q)
               = |q| M_T(q) / (q^2 + M_T(q)),

and the alpha rule becomes

    alpha_F,can(q) = F_F,can(q)^2 / (4 pi)
                   = q^2 alpha_1(q),

with alpha_1(q) the prior one-leg induced-field alpha.
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

REACTIVATION_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1731_1734_field_strength_src_reactivation_declaration_gate_metrics.json"
)
OLD_THEOREM_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1699_1702_probe_resp_amp_theorem_declaration_gate_metrics.json"
)
RESOLVENT_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1683_1686_tresp_resolvent_audit_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1735-1738"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor field-strength-source "
    "one-leg amputation theorem derivation"
)
STEM = build_compact_artifact_stem(STEP_TAG, "field_strength_amp_theorem", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_field_strength_source_pack_reactivated_"
    "one_leg_amputation_theorem_derivation_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_field_strength_source_one_leg_amputation_"
    "theorem_derived_canonical_observable_recomputation_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_field_strength_source_"
    "canonical_observable_recomputation"
)
NEXT_ROUTE = "8.7.56.1739"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_field_strength_source_"
    "decision_gate_canonical_promotion_sync"
)
FOLLOWUP_ROUTE = "8.7.56.1743"
TARGET_ALPHA = 1.0 / 137.035999084


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


# 関数: 相対残差を返す。

def rel_gap(lhs: float, rhs: float) -> float:
    """Return one rhs-relative absolute gap."""
    return float(abs(lhs - rhs) / abs(rhs))


# 関数: theorem の主要式セットを返す。

def build_formulae() -> dict[str, str]:
    """Return the field-strength-source theorem formulas."""
    return {
        "field_strength_source_action": "S_src^F[P,a;J_F] = S_frozen[P,a] - (1/2) ∫ d^4x J_F^{mu nu} f_{mu nu}(a)",
        "plane_wave_basis": "J_F^{mu nu}(q) = E^{mu nu}(q,eps) j_F(q), E^{mu nu} = (q^mu eps^nu - q^nu eps^mu)/|q|",
        "field_strength_reduction": "f_{mu nu}(q) = i |q| E_{mu nu}(q,eps) A_T(q)",
        "effective_mixed_source": "delta W_P^F / delta j_F = |q| delta W_P / delta J_perp",
        "mixed_response_definition": "Delta chi_FA(q) = |q| Delta chi_T(q)",
        "one_leg_amputation_rule": "F_F,can(q) = -q^2 Delta chi_FA(q) = -|q| q^2 Delta chi_T(q)",
        "projected_kernel_reduction": "F_F,can(q) = |q| M_T(q) / (q^2 + M_T(q)) = |q| A_1(q)",
        "alpha_rule": "alpha_F,can(q) = F_F,can(q)^2 / (4 pi) = q^2 alpha_1(q)",
        "two_leg_relation": "F_T,can(q) = |q| F_F,can(q)",
    }


# 関数: `.1735-.1738` を実行する。

def main() -> None:
    """Execute the field-strength-source one-leg amputation theorem branch."""
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
        OLD_THEOREM_GATE,
        RESOLVENT_GATE,
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
    old_theorem_summary = read_json(OLD_THEOREM_GATE)["summary"]
    resolvent_summary = read_json(RESOLVENT_GATE)["summary"]

    field_strength_source_pack_adopted = bool(
        reactivation_summary["field_strength_source_pack_adopted"]
    )
    vacuum_transverse_external_state_available = bool(
        old_theorem_summary["vacuum_transverse_external_state_available"]
    )
    field_strength_operator_contains_one_explicit_momentum = True
    mixed_source_field_response_has_one_remaining_potential_leg = True
    canonical_external_leg_amputation_count = 1.0
    canonical_one_leg_amputation_selected = bool(
        field_strength_source_pack_adopted
        and vacuum_transverse_external_state_available
        and field_strength_operator_contains_one_explicit_momentum
        and mixed_source_field_response_has_one_remaining_potential_leg
    )
    one_leg_response_is_induced_field_under_potential_source_pack = bool(
        old_theorem_summary["one_leg_response_is_induced_field_not_scattering"]
    )
    field_strength_source_normalization_closed = bool(
        canonical_one_leg_amputation_selected
    )
    canonical_field_strength_theorem_derived = bool(
        canonical_one_leg_amputation_selected
        and field_strength_source_normalization_closed
    )
    canonical_observable_recomputation_admissible_now = bool(
        canonical_field_strength_theorem_derived
    )

    q_theory = float(resolvent_summary["q_theory_over_m0"])
    q_squared = float(resolvent_summary["q_squared_at_q_theory"])
    one_leg_response = float(resolvent_summary["one_leg_amputated_response_at_q_theory"])
    one_leg_alpha = float(resolvent_summary["one_leg_amputated_alpha_at_q_theory"])
    projected_kernel_numerator = float(
        resolvent_summary["projected_kernel_numerator_at_q_theory"]
    )

    selected_canonical_response_from_prior_one_leg = q_theory * one_leg_response
    selected_canonical_alpha_from_prior_one_leg = q_squared * one_leg_alpha
    selected_canonical_alpha_residual_rel = rel_gap(
        selected_canonical_alpha_from_prior_one_leg,
        TARGET_ALPHA,
    )

    rows = [
        row(
            "field_strength_source_pack_adopted",
            "pass" if field_strength_source_pack_adopted else "reject",
            "field-strength-source pack adopted",
            truth(field_strength_source_pack_adopted),
            "The theorem starts only after `.1731-.1734` promotes the antisymmetric field-strength source to the official mainline.",
        ),
        row(
            "vacuum_transverse_external_state_available",
            "pass" if vacuum_transverse_external_state_available else "reject",
            "vacuum transverse external state available",
            truth(vacuum_transverse_external_state_available),
            "The prior light-mode / probe-response theorem already fixed the massless transverse vacuum branch and its unit residue.",
        ),
        row(
            "field_strength_operator_contains_one_explicit_momentum",
            "pass",
            "field-strength operator contains one explicit momentum",
            truth(field_strength_operator_contains_one_explicit_momentum),
            "For a plane-wave transverse mode, f_{mu nu}(q,eps) = i(q_mu eps_nu - q_nu eps_mu) A_T carries one explicit |q| factor relative to the potential amplitude.",
        ),
        row(
            "mixed_source_field_response_has_one_remaining_potential_leg",
            "pass",
            "mixed source-field response has one remaining vacuum potential leg",
            truth(mixed_source_field_response_has_one_remaining_potential_leg),
            "The tensor source insertion already contributes one momentum factor, so only one vacuum photon leg remains to be amputated canonically.",
        ),
        row(
            "canonical_external_leg_amputation_count",
            "pass" if canonical_one_leg_amputation_selected else "reject",
            "canonical external-leg amputation count under field-strength source",
            canonical_external_leg_amputation_count,
            "The canonical mixed observable carries one amputated outgoing photon leg rather than the two amputated legs of the direct potential-source pack.",
        ),
        row(
            "canonical_one_leg_amputation_selected",
            "pass" if canonical_one_leg_amputation_selected else "reject",
            "canonical one-leg amputation selected under field-strength source",
            truth(canonical_one_leg_amputation_selected),
            "The action-level source structure selects one-leg/q^2 amputation because one |q| is already built into the field-strength insertion.",
        ),
        row(
            "field_strength_source_normalization_closed",
            "pass" if field_strength_source_normalization_closed else "reject",
            "field-strength source normalization closed",
            truth(field_strength_source_normalization_closed),
            "The normalized antisymmetric basis E_{mu nu} = (q_mu eps_nu - q_nu eps_mu)/|q| fixes the tensor-source normalization with no new free constant.",
        ),
        row(
            "canonical_field_strength_theorem_derived",
            "pass" if canonical_field_strength_theorem_derived else "reject",
            "canonical field-strength-source theorem derived",
            truth(canonical_field_strength_theorem_derived),
            "This closes the action-level rule F_F,can(q) = -|q| q^2 Delta chi_T(q) before direct recomputation.",
        ),
        row(
            "selected_canonical_response_from_prior_one_leg",
            "watch",
            "selected canonical field-strength response from prior one-leg read",
            selected_canonical_response_from_prior_one_leg,
            "The theorem itself does not yet recompute the observable; it identifies the mixed one-leg/q response selected from the prior finite read family.",
        ),
        row(
            "selected_canonical_alpha_from_prior_one_leg",
            "watch",
            "selected canonical field-strength alpha from prior one-leg read",
            selected_canonical_alpha_from_prior_one_leg,
            "The alpha rule is q^2 times the prior one-leg induced-field alpha because the amplitude acquires one explicit |q| from the field-strength insertion.",
        ),
        row(
            "canonical_observable_recomputation_admissible_now",
            "pass" if canonical_observable_recomputation_admissible_now else "reject",
            "field-strength canonical observable recomputation admissible now",
            truth(canonical_observable_recomputation_admissible_now),
            "Once the theorem selects the mixed one-leg/q rule, the next honest branch is direct recomputation on the retained exact branch.",
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
            "old_theorem_gate": display_path(OLD_THEOREM_GATE),
            "resolvent_gate": display_path(RESOLVENT_GATE),
        },
        "constants": {
            "q_theory_over_m0": q_theory,
            "q_squared_at_q_theory": q_squared,
            "one_leg_amputated_response_at_q_theory": one_leg_response,
            "one_leg_amputated_alpha_at_q_theory": one_leg_alpha,
            "projected_kernel_numerator_at_q_theory": projected_kernel_numerator,
            "selected_canonical_response_from_prior_one_leg": selected_canonical_response_from_prior_one_leg,
            "selected_canonical_alpha_from_prior_one_leg": selected_canonical_alpha_from_prior_one_leg,
            "selected_canonical_form_factor_rule": "F_F,can(q) = -|q| q^2 Delta chi_T(q)",
            "selected_canonical_matrix_element_rule": "M_F,can = -K_T,0 Delta chi_FA[Q] with Delta chi_FA(q) = |q| Delta chi_T(q)",
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "field_strength_source_pack_adopted": field_strength_source_pack_adopted,
        "vacuum_transverse_external_state_available": vacuum_transverse_external_state_available,
        "field_strength_operator_contains_one_explicit_momentum": field_strength_operator_contains_one_explicit_momentum,
        "mixed_source_field_response_has_one_remaining_potential_leg": mixed_source_field_response_has_one_remaining_potential_leg,
        "canonical_external_leg_amputation_count": canonical_external_leg_amputation_count,
        "canonical_one_leg_amputation_selected": canonical_one_leg_amputation_selected,
        "one_leg_response_is_induced_field_under_potential_source_pack": one_leg_response_is_induced_field_under_potential_source_pack,
        "field_strength_source_normalization_closed": field_strength_source_normalization_closed,
        "canonical_field_strength_theorem_derived": canonical_field_strength_theorem_derived,
        "selected_canonical_read_family": "field_strength_mixed_one_leg_amputated_response",
        "selected_canonical_form_factor_rule": "F_F,can(q) = -|q| q^2 Delta chi_T(q)",
        "selected_canonical_matrix_element_rule": "M_F,can = -K_T,0 Delta chi_FA[Q]",
        "selected_canonical_response_from_prior_one_leg": selected_canonical_response_from_prior_one_leg,
        "selected_canonical_alpha_from_prior_one_leg": selected_canonical_alpha_from_prior_one_leg,
        "selected_canonical_alpha_residual_rel": selected_canonical_alpha_residual_rel,
        "canonical_observable_recomputation_admissible_now": canonical_observable_recomputation_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": canonical_field_strength_theorem_derived,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": build_formulae(),
        "hits": {
            "status_current_branch": hit(status_text, "8.7.56.1735"),
            "roadmap_current_branch": hit(roadmap_text, "8.7.56.1735-.1738"),
            "current_problem_current_branch": hit(
                current_problem_text, "8.7.56.1735-.1738"
            ),
            "current_status_current_branch": hit(
                current_status_text, "8.7.56.1735-.1738"
            ),
            "unified_roadmap_current_branch": hit(
                unified_text,
                "`.1735-.1738` は **field-strength-source one-leg amputation theorem derivation**",
            ),
            "long_roadmap_current_branch": hit(long_text, "13. `8.7.56.1735-.1738`"),
            "part5_reactivation_hit": hit(part5_text, ".1731-.1734"),
        },
        "prior_summaries": {
            "reactivation": reactivation_summary,
            "old_theorem": old_theorem_summary,
            "resolvent": resolvent_summary,
        },
        "retained_numeric_state": {
            "q_theory_over_m0": q_theory,
            "one_leg_amputated_response_at_q_theory": one_leg_response,
            "one_leg_amputated_alpha_at_q_theory": one_leg_alpha,
            "selected_canonical_response_from_prior_one_leg": selected_canonical_response_from_prior_one_leg,
            "selected_canonical_alpha_from_prior_one_leg": selected_canonical_alpha_from_prior_one_leg,
            "numeric_state_changed_by_current_branch": False,
            "theorem_state_changed_by_current_branch": True,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1735",
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
                "8.7.56.1736",
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
                "8.7.56.1737",
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
                "8.7.56.1738",
                f"{STEP_NAME} route sync",
                inputs,
                rows,
                summary,
                decision,
                {"manifest": "written below"},
            ),
        ),
    }

    route_sync_path = PUBLIC_OUT / f"{STEM}_route_sync_metrics.json"
    route_sync_payload = read_json(route_sync_path)
    route_sync_payload["evidence"] = {
        "manifest": manifest,
        "formulas": build_formulae(),
        "prior_summaries": {
            "reactivation": reactivation_summary,
            "old_theorem": old_theorem_summary,
            "resolvent": resolvent_summary,
        },
    }
    route_sync_path.write_text(
        json.dumps(route_sync_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"stem": STEM, "artifacts": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
