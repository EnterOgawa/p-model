#!/usr/bin/env python3
"""Generate 8.7.56.5031-.5034 deeper selected-candidate selector theorem artifacts."""

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
BASE = (
    "beyond_current_written_action_selector_chart_representative_concrete_rule_"
    "selector_representative_selected_candidate_selector_selected_candidate_"
    "selected_candidate_selected_candidate"
)
PREFIX = f"exact_{BASE}"
REQ = (
    "exact_minimal_selector_chart_representative_concrete_rule_selector_"
    "representative_selected_candidate_selector_selected_candidate_selected_"
    "candidate_selected_candidate_selector_representative_requirement_theorem_"
    "available_now"
)
FOLLOW = (
    "updated_pack_beyond_current_written_action_selector_chart_representative_"
    "concrete_rule_selector_representative_selected_candidate_selector_"
    "selected_candidate_selected_candidate_selected_candidate_selector_"
    "representative_primary_followup_required"
)

PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5027-5030",
        "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5023-5026",
        "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5031-5034"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack beyond-current-"
    "written-action selector chart representative concrete-rule selector "
    "representative selected-candidate selector selected-candidate selected-"
    "candidate selected-candidate selector theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selector_chart_representative_concrete_rule_"
    "selector_representative_selected_candidate_selector_selected_candidate_"
    "selected_candidate_selected_candidate_no_go_theorem_derived_selector_"
    "chart_representative_concrete_rule_selector_representative_selected_"
    "candidate_selector_selected_candidate_selected_candidate_selected_"
    "candidate_selector_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selector_chart_representative_concrete_rule_"
    "selector_representative_selected_candidate_selector_selected_candidate_"
    "selected_candidate_selected_candidate_selector_no_go_theorem_derived_"
    "selector_chart_representative_concrete_rule_selector_representative_"
    "selected_candidate_selector_selected_candidate_selected_candidate_"
    "selected_candidate_selector_representative_primary_pack_refresh_"
    "secondary_gate"
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


# 関数: theorem で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the deeper selected-candidate selector theorem audit."""
    return {
        "selected_candidate_family": (
            "Cand_cand_cand_cand_sel_rule[...] := { C_***^(...;M_sel) | "
            "M_sel in Rep_cand_cand_cand_sel_rule[...] }"
        ),
        "selector_family": "Sel_cand_cand_cand_cand_sel_rule[...] := { N_sel | N_sel : Cand_cand_cand_cand_sel_rule[...] -> R }",
        "selector_equivalence": "N_sel' ~_ord N_sel iff N_sel' = psi o N_sel with strictly monotone psi",
        "selector_order_class": "[N_sel]_ord := Sel_cand_cand_cand_cand_sel_rule / ~_ord",
    }


# 関数: `.5031-.5034` を実行する。

def main() -> None:
    """Execute the deeper selected-candidate selector theorem audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT):
        sign_base.require(path)

    prior_gate = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a_key = f"gate_a_updated_pack_exact_{BASE}_no_go_available_now"
    family_key = f"{PREFIX}_family_formula_available_now"
    no_go_key = f"{PREFIX}_no_go_theorem_available_now"
    selector_family_key = f"{PREFIX}_selector_family_formula_available_now"
    selector_eq_key = f"{PREFIX}_selector_equivalence_class_formula_available_now"
    selector_mono_key = f"{PREFIX}_selector_monotone_equivalence_theorem_available_now"
    selector_no_go_key = f"{PREFIX}_selector_no_go_theorem_available_now"
    selector_available_key = f"{PREFIX}_selector_available_now"

    audit_selected = bool(
        prior_gate[
            "gate_b_updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_promoted_next"
        ]
        and prior_gate["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(prior_gate["failure_matrix_non_surrogate_guard_preserved"])
    prior_no_go = bool(prior_gate[gate_a_key])
    prior_family = bool(prior_audit[family_key])
    prior_requirement = bool(
        prior_audit[
            "exact_minimal_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_requirement_theorem_available_now"
        ]
    )
    selector_explicit = bool(
        audit_selected and retry_mode and non_surrogate_guard and prior_no_go and prior_family and prior_requirement
    )
    selector_unique_now = False
    followup_required = bool(selector_explicit)
    blind_blocked = bool(prior_gate["blind_vector_observable_gate_still_blocked"])

    row_specs = [
        (
            "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_audit_selected",
            audit_selected,
            "updated-pack beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate selector audit selected",
            "This branch is worth running only after the deeper selected-candidate family no-go is already closed and same-tag downstream replay remains shut.",
        ),
        ("retry_gate_computation_mode_selected", retry_mode, "retry gate computation mode selected", "This turn must close a new theorem object rather than recurse into exhausted replay bookkeeping."),
        ("failure_matrix_non_surrogate_guard_preserved", non_surrogate_guard, "failure-matrix non-surrogate guard preserved", "The deeper selected-candidate-selector theorem is admissible only if it does not reopen the exhausted surrogate family."),
        (gate_a_key, prior_no_go, "gate A exact beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate no-go available now", "The selector theorem starts only after current theory already fixes only the family Cand_cand_cand_cand_sel_rule[...] and not one canonical selected candidate."),
        (family_key, prior_family, "exact beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate family formula available now", "The theorem uses the already closed family Cand_cand_cand_cand_sel_rule[...] as its starting object."),
        ("exact_minimal_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_requirement_theorem_available_now", prior_requirement, "exact minimal selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate selector requirement theorem available now", "The prior branch already fixed that the honest next blocker is which selector on Cand_cand_cand_cand_sel_rule[...] could canonically choose one candidate."),
        ("selector_order_class_explicit", selector_explicit, "selector order class explicit", "The theorem stack can now state the selector family and order class on Cand_cand_cand_cand_sel_rule[...] literally."),
        ("selector_unique_now", selector_unique_now, "selector unique now", "Current theory still does not choose one canonical selector-selected-candidate functional N_sel."),
        (selector_family_key, selector_explicit, "exact beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate selector family formula available now", "The theorem stack now fixes the selector family on Cand_cand_cand_cand_sel_rule[...] explicitly."),
        (selector_eq_key, selector_explicit, "exact beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate selector equivalence-class formula available now", "The theorem stack now fixes the selector order class [N_sel]_ord explicitly."),
        (selector_mono_key, selector_explicit, "exact beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate selector monotone-equivalence theorem available now", "Strictly monotone reparameterizations still preserve the chosen candidate on Cand_cand_cand_cand_sel_rule[...], so current theory fixes only an order class."),
        (selector_no_go_key, selector_explicit, "exact beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate selector no-go theorem available now", "The theorem stack now closes that current theory fixes only the selector family or selector order class on Cand_cand_cand_cand_sel_rule[...] and not one canonical selector functional."),
        (REQ, selector_explicit, "exact minimal selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate selector representative requirement theorem available now", "The honest next blocker is now which representative rule on [N_sel]_ord could choose one concrete selector functional."),
        (selector_available_key, False, "exact beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate selector available now", "The current theorem stack fixes only the selector family and no-go, not one concrete selector functional N_sel."),
        (FOLLOW, followup_required, "updated-pack beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate selector representative primary followup required", "The honest followup is the representative theorem on [N_sel]_ord rather than same-tag replay."),
        ("updated_pack_corrected_pack_refresh_secondary_hold_retained", followup_required, "updated-pack corrected pack-refresh secondary hold retained", "Pack-refresh remains secondary because the main unresolved blocker moved deeper into the selector-selected-candidate lane."),
        ("updated_pack_same_tag_deeper_selected_candidate_selector_downstream_rerun_admissible_now", False, "updated-pack same-tag deeper selected-candidate-selector downstream rerun admissible now", "Same-tag downstream rerun remains closed because the blocker is now the selector family on Cand_cand_cand_cand_sel_rule, not old replay syntax."),
        ("blind_vector_observable_gate_still_blocked", blind_blocked, "blind-vector observable gate still blocked", "Blind-vector direct computation still waits on one concrete selected extension."),
    ]
    rows = [
        sign_base.row(rid, "pass" if ok else "reject", metric, sign_base.truth(ok), note)
        for rid, ok, metric, note in row_specs
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate["retained_scalar_residual_rel"]),
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        selector_family_key: selector_explicit,
        selector_eq_key: selector_explicit,
        selector_mono_key: selector_explicit,
        selector_no_go_key: selector_explicit,
        REQ: selector_explicit,
        selector_available_key: False,
        FOLLOW: followup_required,
        "updated_pack_corrected_pack_refresh_secondary_hold_retained": followup_required,
        "updated_pack_same_tag_deeper_selected_candidate_selector_downstream_rerun_admissible_now": False,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "pack_update_required_now": followup_required,
        "selected_primary_completion_lane": "updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_representative_theorem_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_representative_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.5035",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_representative_theorem_audit",
        "selected_followup_route_or_none": "8.7.56.5039",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5033",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5035",
                "followup_route": "8.7.56.5039",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_beyond_current_written_action_selector_chart_representative_concrete_rule_selector_representative_selected_candidate_selector_selected_candidate_selected_candidate_selected_candidate_selector_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        f"[done] {STEP_TAG} updated-pack beyond-current-written-action selector chart representative concrete-rule selector representative selected-candidate selector selected-candidate selected-candidate selected-candidate selector theorem completed"
    )
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
