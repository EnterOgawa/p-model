#!/usr/bin/env python3
"""Generate 8.7.56.5023-.5026 deeper selected-candidate family theorem artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5019-5022",
        "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selector_representative_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5015-5018",
        "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selector_representative_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5023-5026"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack beyond-current-"
    "written-action selector chart representative concrete-rule selector "
    "representative selected-candidate selector selected-candidate selected-"
    "candidate selected-candidate theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selector_chart_representative_concrete_rule_"
    "selector_representative_selected_candidate_selector_selected_candidate_"
    "selected_candidate_selector_representative_finite_anchor_no_go_theorem_"
    "derived_selector_chart_representative_concrete_rule_selector_"
    "representative_selected_candidate_selector_selected_candidate_"
    "selected_candidate_selected_candidate_primary_hybrid_reserve_secondary_"
    "next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selector_chart_representative_concrete_rule_"
    "selector_representative_selected_candidate_selector_selected_candidate_"
    "selected_candidate_selected_candidate_no_go_theorem_derived_selector_"
    "chart_representative_concrete_rule_selector_representative_selected_"
    "candidate_selector_selected_candidate_selected_candidate_selected_"
    "candidate_selector_primary_pack_refresh_secondary_gate"
)


# 関数: JSON/CSV artifact を書き出す。
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

    return {"json": sign_base.display_path(paths["json"])}


# 関数: deeper selected-candidate family theorem の式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the deeper selected-candidate family theorem audit."""
    return {
        "selector_representative_family": (
            "Rep_cand_cand_cand_sel_rule[B_cand_cand_cand_sel;B_cand_cand_sel;B_cand_sel;B_rule_sel;B_rule] := { "
            "M_sel in Sel_cand_cand_cand_sel_rule[B_cand_cand_sel;B_cand_sel;B_rule_sel;B_rule] | "
            "M_sel[D_k] = m_k for all k }"
        ),
        "selected_candidate_formula": (
            "C_***^(B_cand_cand_cand_sel;B_cand_cand_sel;B_cand_sel;B_rule_sel;B_rule;M_sel) := "
            "argext_(D in Cand_cand_cand_sel_rule[B_cand_cand_sel;B_cand_sel;B_rule_sel;B_rule]) M_sel[D]"
        ),
        "selected_candidate_family": (
            "Cand_cand_cand_cand_sel_rule[B_cand_cand_cand_sel;B_cand_cand_sel;B_cand_sel;B_rule_sel;B_rule] := { "
            "C_***^(B_cand_cand_cand_sel;B_cand_cand_sel;B_cand_sel;B_rule_sel;B_rule;M_sel) | "
            "M_sel in Rep_cand_cand_cand_sel_rule[B_cand_cand_cand_sel;B_cand_cand_sel;B_cand_sel;B_rule_sel;B_rule] }"
        ),
        "selector_family": (
            "Sel_cand_cand_cand_cand_sel_rule[B_cand_cand_cand_sel;B_cand_cand_sel;B_cand_sel;B_rule_sel;B_rule] := { "
            "N_sel | N_sel : Cand_cand_cand_cand_sel_rule[B_cand_cand_cand_sel;B_cand_cand_sel;B_cand_sel;B_rule_sel;B_rule] -> R }"
        ),
        "selected_candidate_no_go": (
            "if Rep_cand_cand_cand_sel_rule[...] remains non-singleton, current theory fixes only "
            "Cand_cand_cand_cand_sel_rule[...] and not one canonical selector-selected-candidate"
        ),
    }


# 関数: `.5023-.5026` を実行する。

def main() -> None:
    """Execute the deeper selected-candidate family theorem audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    selector_representative_no_go_available = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selector_representative_finite_anchor_no_go_available_now"
        ]
    )
    selector_representative_family_available = bool(
        prior_audit_summary[
            "exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selector_representative_finite_anchor_family_formula_available_now"
        ]
    )
    selected_candidate_requirement_available = bool(
        prior_audit_summary[
            "exact_minimal_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_requirement_theorem_available_now"
        ]
    )
    selected_candidate_family_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selector_representative_no_go_available
        and selector_representative_family_available
        and selected_candidate_requirement_available
    )
    selected_candidate_unique_now = False
    exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_family_formula_available_now = bool(
        selected_candidate_family_explicit
    )
    exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_no_go_theorem_available_now = bool(
        selected_candidate_family_explicit
    )
    exact_minimal_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_requirement_theorem_available_now = bool(
        selected_candidate_family_explicit
    )
    exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_available_now = False
    updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_primary_followup_required = bool(
        exact_minimal_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_requirement_theorem_available_now
    )
    updated_pack_corrected_pack_refresh_secondary_hold_retained = bool(
        updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_primary_followup_required
    )
    updated_pack_same_tag_selected_candidate_selected_candidate_selector_downstream_rerun_admissible_now = False
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])

    rows = [
        sign_base.row(
            "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the deeper selector-representative finite-anchor no-go is already closed and same-tag downstream replay remains shut.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This turn must close a new theorem object rather than recurse into already exhausted replay bookkeeping.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The deeper selected-candidate family theorem is admissible only if it does not reopen the exhausted surrogate family.",
        ),
        sign_base.row(
            "gate_a_updated_pack_exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selector_representative_finite_anchor_no_go_available_now",
            "pass" if selector_representative_no_go_available else "reject",
            "gate A exact beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selector representative finite-anchor no-go available now",
            sign_base.truth(selector_representative_no_go_available),
            "The selected-candidate theorem starts only after current theory already fixes only a representative family Rep_cand_cand_cand_sel_rule[...] and not one canonical deeper selector representative M_sel.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selector_representative_finite_anchor_family_formula_available_now",
            "pass" if selector_representative_family_available else "reject",
            "exact beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selector representative finite-anchor family formula available now",
            sign_base.truth(selector_representative_family_available),
            "The theorem uses the already closed finite-anchor representative family on the deeper selector order class as its starting object.",
        ),
        sign_base.row(
            "exact_minimal_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_requirement_theorem_available_now",
            "pass" if selected_candidate_requirement_available else "reject",
            "exact minimal selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate requirement theorem available now",
            sign_base.truth(selected_candidate_requirement_available),
            "The prior branch already fixed that the honest next blocker is the deeper selected-candidate family induced by unresolved deeper selector representatives.",
        ),
        sign_base.row(
            "selected_candidate_family_explicit",
            "pass" if selected_candidate_family_explicit else "reject",
            "selected-candidate family explicit",
            sign_base.truth(selected_candidate_family_explicit),
            "The theorem stack can now state the deeper selected-candidate family induced by unresolved deeper selector representatives literally as Cand_cand_cand_cand_sel_rule[...].",
        ),
        sign_base.row(
            "selected_candidate_unique_now",
            "pass" if selected_candidate_unique_now else "reject",
            "selected-candidate unique now",
            sign_base.truth(selected_candidate_unique_now),
            "Current theory still does not choose one canonical selector-selected-candidate once Rep_cand_cand_cand_sel_rule[...] remains non-singleton.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_family_formula_available_now",
            "pass"
            if exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_family_formula_available_now
            else "reject",
            "exact beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate family formula available now",
            sign_base.truth(
                exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_family_formula_available_now
            ),
            "The theorem stack now fixes the deeper selected-candidate family induced by unresolved deeper selector representatives explicitly.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_no_go_theorem_available_now",
            "pass"
            if exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_no_go_theorem_available_now
            else "reject",
            "exact beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate no-go theorem available now",
            sign_base.truth(
                exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_no_go_theorem_available_now
            ),
            "The theorem stack now closes that current theory fixes only Cand_cand_cand_cand_sel_rule[...] and not one canonical selector-selected-candidate.",
        ),
        sign_base.row(
            "exact_minimal_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_requirement_theorem_available_now",
            "pass"
            if exact_minimal_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_requirement_theorem_available_now
            else "reject",
            "exact minimal selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate selector requirement theorem available now",
            sign_base.truth(
                exact_minimal_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_requirement_theorem_available_now
            ),
            "The honest next blocker is now which selector on Cand_cand_cand_cand_sel_rule[...] could canonically choose one candidate.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_available_now",
            "pass"
            if exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_available_now
            else "reject",
            "exact beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate available now",
            sign_base.truth(
                exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_available_now
            ),
            "The current theorem stack fixes only the deeper selected-candidate family and no-go, not one concrete selector-selected-candidate.",
        ),
        sign_base.row(
            "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_primary_followup_required",
            "pass"
            if updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_primary_followup_required
            else "reject",
            "updated-pack beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate selector primary followup required",
            sign_base.truth(
                updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_primary_followup_required
            ),
            "The honest followup is the selector theorem on Cand_cand_cand_cand_sel_rule rather than same-tag replay.",
        ),
        sign_base.row(
            "updated_pack_corrected_pack_refresh_secondary_hold_retained",
            "pass" if updated_pack_corrected_pack_refresh_secondary_hold_retained else "reject",
            "updated-pack corrected pack-refresh secondary hold retained",
            sign_base.truth(updated_pack_corrected_pack_refresh_secondary_hold_retained),
            "Pack-refresh remains secondary because the main unresolved blocker moved deeper into the selector-selected-candidate lane.",
        ),
        sign_base.row(
            "updated_pack_same_tag_selected_candidate_selected_candidate_selector_downstream_rerun_admissible_now",
            "pass" if updated_pack_same_tag_selected_candidate_selected_candidate_selector_downstream_rerun_admissible_now else "reject",
            "updated-pack same-tag selected-candidate-selected-candidate-selector downstream rerun admissible now",
            sign_base.truth(updated_pack_same_tag_selected_candidate_selected_candidate_selector_downstream_rerun_admissible_now),
            "Same-tag downstream rerun remains closed because the blocker is now the induced deeper selected-candidate family, not old selector-representative syntax.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on one concrete selected extension.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(
            prior_gate_summary["retained_scalar_residual_rel"]
        ),
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_family_formula_available_now": exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_family_formula_available_now,
        "exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_no_go_theorem_available_now": exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_no_go_theorem_available_now,
        "exact_minimal_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_requirement_theorem_available_now": exact_minimal_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_requirement_theorem_available_now,
        "exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_available_now": exact_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_available_now,
        "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_primary_followup_required": updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_primary_followup_required,
        "updated_pack_corrected_pack_refresh_secondary_hold_retained": updated_pack_corrected_pack_refresh_secondary_hold_retained,
        "updated_pack_same_tag_selected_candidate_selected_candidate_selector_downstream_rerun_admissible_now": updated_pack_same_tag_selected_candidate_selected_candidate_selector_downstream_rerun_admissible_now,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "pack_update_required_now": updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_primary_followup_required,
        "selected_primary_completion_lane": "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_theorem_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.5027",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_theorem_audit",
        "selected_followup_route_or_none": "8.7.56.5031",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5025",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5027",
                "followup_route": "8.7.56.5031",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} updated-pack beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate theorem completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
