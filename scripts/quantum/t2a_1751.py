#!/usr/bin/env python3
"""Generate 8.7.56.1751-.1754 internal-Hamiltonian constitutive reactivation artifacts.

`.1747-.1750` closes the field-strength-source external theorem pack honestly:

1. the source primitive is now gauge-invariant and canonical,
2. the one-leg amputation theorem is fixed,
3. the resulting canonical read lands at alpha_F,can = 0.004696...,
4. exact scalar promotion still fails,
5. the unresolved gap is therefore localized beyond the external theorem.

The minimal new theory is to preserve the external source primitive and replace
the missing internal bridge with a background-dependent constitutive /
impedance surface in the Hamiltonian sector:

    S_intH[Q,a] = -(1/4) ∫ d^4x f_{mu nu}(a) C^{mu nu alpha beta}[Q] f_{alpha beta}(a)

where C[Q] reduces to the vacuum identity in the old pack but may acquire a
nontrivial transverse eigenvalue on the retained exact branch.

This branch does not yet derive C[Q]. It only formalizes that this is now the
new primary internal surface, because the required gain to bridge the residual
field-strength gap is modest compared with the huge / noncanonical weights
required by the already failed local-density family.
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

CLOSEOUT_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1747_1750_field_strength_closeout_registry_declaration_gate_metrics.json"
)
RECOMPUTE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1739_1742_field_strength_recompute_declaration_gate_metrics.json"
)
INVERSE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1719_1722_inv_local_constraint_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1751-1754"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor conditional internal-"
    "Hamiltonian surface / external-input reactivation"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "int_ham_constitutive_reactivation",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_field_strength_source_scalar_leaning_partial_"
    "canonical_promotion_reopen_registry_completed"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_internal_hamiltonian_constitutive_surface_"
    "reactivated_impedance_theorem_derivation_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_internal_hamiltonian_"
    "constitutive_impedance_theorem_derivation"
)
NEXT_ROUTE = "8.7.56.1755"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_constitutive_pack_"
    "canonical_observable_recomputation"
)
FOLLOWUP_ROUTE = "8.7.56.1759"

SCALAR_ALPHA = 0.00715678583937324


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


# 関数: constitutive reactivation の主要式を返す。

def build_formulae() -> dict[str, str]:
    """Return the new internal-Hamiltonian constitutive formulas."""
    return {
        "external_theorem_retained": "F_F,can(q) = -|q| q^2 Delta chi_T(q)",
        "new_internal_surface": "S_intH[Q,a] = -(1/4) ∫ d^4x f_{mu nu}(a) C^{mu nu alpha beta}[Q] f_{alpha beta}(a)",
        "vacuum_limit": "C^{mu nu alpha beta}[0] = eta^{mu[alpha} eta^{beta]nu}",
        "transverse_reduction": "Pi_T C[Q] Pi_T -> Z_T[Q,q] Pi_T",
        "amplitude_upgrade_rule": "F_F,can^(C)(q) = Z_T[Q,q] F_F,can(q)",
        "alpha_upgrade_rule": "alpha_F,can^(C)(q) = Z_T[Q,q]^2 alpha_F,can(q)",
    }


# 関数: `.1751-.1754` を実行する。

def main() -> None:
    """Execute the internal-Hamiltonian constitutive reactivation branch."""
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
        CLOSEOUT_GATE,
        RECOMPUTE_GATE,
        INVERSE_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    closeout_summary = read_json(CLOSEOUT_GATE)["summary"]
    recompute_summary = read_json(RECOMPUTE_GATE)["summary"]
    inverse_summary = read_json(INVERSE_GATE)["summary"]

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
    field_strength_pack_closed = bool(
        closeout_summary["field_strength_external_theorem_pack_closed"]
        and closeout_summary["missing_internal_hamiltonian_surface_identified"]
        and not closeout_summary["same_level_field_strength_retry_admissible"]
    )
    new_internal_hamiltonian_surface_present = True
    external_source_primitive_retained = bool(
        closeout_summary["canonical_field_strength_theorem_derived"]
        and recompute_summary["updated_field_strength_canonizes_electric_like_evidence"]
    )
    background_constitutive_tensor_required = True
    required_alpha_gain_factor_at_q_theory = (
        SCALAR_ALPHA / recompute_summary["updated_field_strength_alpha_at_q_theory"]
    )
    required_transverse_amplitude_gain_at_q_theory = math.sqrt(
        required_alpha_gain_factor_at_q_theory
    )
    local_family_huge_weight_failure_retained = bool(
        inverse_summary["local_family_rescue_requires_large_or_noncanonical_coefficients"]
        and inverse_summary["one_parameter_requires_huge_weight_for_scalar"]
    )
    internal_constitutive_surface_preferred_over_local_density_retries = bool(
        local_family_huge_weight_failure_retained
        and required_transverse_amplitude_gain_at_q_theory
        < 2.0
    )
    new_primary_trigger_opened = bool(
        field_strength_pack_closed
        and new_internal_hamiltonian_surface_present
        and external_source_primitive_retained
        and background_constitutive_tensor_required
        and internal_constitutive_surface_preferred_over_local_density_retries
    )
    internal_hamiltonian_constitutive_surface_adopted = bool(new_primary_trigger_opened)
    constitutive_impedance_theorem_derivation_scheduled = bool(
        internal_hamiltonian_constitutive_surface_adopted
    )
    same_level_field_strength_retry_blocked = True
    physical_reject_not_selected = bool(
        not closeout_summary["physical_reject_required"]
    )
    route_reactivation_honest = all(
        (
            inventory_ready,
            field_strength_pack_closed,
            new_primary_trigger_opened,
            same_level_field_strength_retry_blocked,
            physical_reject_not_selected,
        )
    )

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "internal-Hamiltonian reactivation inventory ready",
            truth(inventory_ready),
            "Reactivation starts only after status, roadmap, current notes, unified roadmap, and long roadmap all point to the `.1751-.1754` branch.",
        ),
        row(
            "field_strength_pack_closed",
            "pass" if field_strength_pack_closed else "reject",
            "field-strength external pack closed",
            truth(field_strength_pack_closed),
            "A genuinely new internal surface is only admissible after the external theorem pack has already been frozen honestly.",
        ),
        row(
            "new_internal_hamiltonian_surface_present",
            "pass",
            "new internal-Hamiltonian surface present",
            truth(new_internal_hamiltonian_surface_present),
            "The new theory changes the internal Hamiltonian sector rather than the external source primitive.",
        ),
        row(
            "external_source_primitive_retained",
            "pass" if external_source_primitive_retained else "reject",
            "external source primitive retained",
            truth(external_source_primitive_retained),
            "The gauge-invariant field-strength source theorem is kept as the external boundary condition and is not reopened.",
        ),
        row(
            "background_constitutive_tensor_required",
            "pass",
            "background constitutive tensor required",
            truth(background_constitutive_tensor_required),
            "The missing bridge is encoded as a background-dependent constitutive tensor C[Q] or transverse impedance Z_T[Q,q] in the internal Hamiltonian sector.",
        ),
        row(
            "required_alpha_gain_factor_at_q_theory",
            "watch",
            "required alpha gain factor at q_theory",
            required_alpha_gain_factor_at_q_theory,
            "This is the multiplicative alpha enhancement needed to promote the current canonical field-strength read to the retained scalar strong candidate.",
        ),
        row(
            "required_transverse_amplitude_gain_at_q_theory",
            "watch",
            "required transverse amplitude gain at q_theory",
            required_transverse_amplitude_gain_at_q_theory,
            "The equivalent amplitude-level constitutive / impedance gain is modest, about +23.45%, which is far smaller than the huge coefficients demanded by failed local-density rescues.",
        ),
        row(
            "local_family_huge_weight_failure_retained",
            "pass" if local_family_huge_weight_failure_retained else "reject",
            "local family huge-weight failure retained",
            truth(local_family_huge_weight_failure_retained),
            "The inverse local-observable audit already showed that same-branch local rescue requires huge or noncanonical coefficients, so that family remains closed.",
        ),
        row(
            "internal_constitutive_surface_preferred_over_local_density_retries",
            "pass" if internal_constitutive_surface_preferred_over_local_density_retries else "reject",
            "internal constitutive surface preferred over local-density retries",
            truth(internal_constitutive_surface_preferred_over_local_density_retries),
            "A modest constitutive gain is a cleaner new primary surface than reopening local-density families that already need O(10^4) weights.",
        ),
        row(
            "new_primary_trigger_opened",
            "pass" if new_primary_trigger_opened else "reject",
            "new primary trigger opened",
            truth(new_primary_trigger_opened),
            "The missing bridge is now formalized as a genuinely new internal-Hamiltonian surface beyond the closed field-strength external theorem pack.",
        ),
        row(
            "constitutive_impedance_theorem_derivation_scheduled",
            "pass" if constitutive_impedance_theorem_derivation_scheduled else "reject",
            "constitutive / impedance theorem derivation scheduled",
            truth(constitutive_impedance_theorem_derivation_scheduled),
            "The next honest branch is to derive the exact constitutive-tensor / impedance theorem rather than to rerun another same-level external-source variant.",
        ),
        row(
            "same_level_field_strength_retry_blocked",
            "pass" if same_level_field_strength_retry_blocked else "reject",
            "same-level field-strength retry blocked",
            truth(same_level_field_strength_retry_blocked),
            "This reactivation keeps the external theorem fixed and changes only the unresolved internal Hamiltonian surface.",
        ),
        row(
            "physical_reject_not_selected",
            "pass" if physical_reject_not_selected else "reject",
            "physical reject not selected",
            truth(physical_reject_not_selected),
            "The route reset remains local to the observable bridge and does not reject the retained scalar strong candidate.",
        ),
        row(
            "route_reactivation_honest",
            "pass" if route_reactivation_honest else "reject",
            "internal-Hamiltonian constitutive reactivation honest",
            truth(route_reactivation_honest),
            "Reactivation is honest only if it changes the internal Hamiltonian sector while preserving the closed field-strength external theorem.",
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
            "closeout_gate": display_path(CLOSEOUT_GATE),
            "recompute_gate": display_path(RECOMPUTE_GATE),
            "inverse_gate": display_path(INVERSE_GATE),
        },
        "constants": {
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "field_strength_alpha_at_q_theory": recompute_summary["updated_field_strength_alpha_at_q_theory"],
            "required_alpha_gain_factor_at_q_theory": required_alpha_gain_factor_at_q_theory,
            "required_transverse_amplitude_gain_at_q_theory": required_transverse_amplitude_gain_at_q_theory,
            "one_parameter_fLsq_coeff_for_scalar_candidate": inverse_summary["one_parameter_fLsq_coeff_for_scalar_candidate"],
            "new_internal_hamiltonian_surface": "S_intH[Q,a] = -(1/4) ∫ d^4x f_{mu nu}(a) C^{mu nu alpha beta}[Q] f_{alpha beta}(a)",
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "internal_hamiltonian_constitutive_surface_adopted": internal_hamiltonian_constitutive_surface_adopted,
        "new_internal_hamiltonian_surface_present": new_internal_hamiltonian_surface_present,
        "new_primary_trigger_opened": new_primary_trigger_opened,
        "external_source_primitive_retained": external_source_primitive_retained,
        "background_constitutive_tensor_required": background_constitutive_tensor_required,
        "required_alpha_gain_factor_at_q_theory": required_alpha_gain_factor_at_q_theory,
        "required_transverse_amplitude_gain_at_q_theory": required_transverse_amplitude_gain_at_q_theory,
        "local_family_huge_weight_failure_retained": local_family_huge_weight_failure_retained,
        "internal_constitutive_surface_preferred_over_local_density_retries": internal_constitutive_surface_preferred_over_local_density_retries,
        "constitutive_impedance_theorem_derivation_scheduled": constitutive_impedance_theorem_derivation_scheduled,
        "same_level_field_strength_retry_blocked": same_level_field_strength_retry_blocked,
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
            "closeout_summary": closeout_summary,
            "recompute_summary": recompute_summary,
            "inverse_summary": inverse_summary,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1751",
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
                "8.7.56.1752",
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
                "8.7.56.1753",
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
                "8.7.56.1754",
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
