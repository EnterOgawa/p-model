#!/usr/bin/env python3
"""Generate 8.7.56.4807-.4810 selected-extension-convention-selector-selected-extension-family artifacts."""

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
        "8.7.56.4803-4806",
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.4799-4802",
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_SUPPORT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.4791-4794",
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.4807-4810"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack beyond-current-"
    "written-action selected extension convention selector selected extension family theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selected_extension_convention_selector_"
    "selected_extension_no_go_theorem_derived_selected_extension_convention_"
    "selector_selected_extension_family_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "beyond_current_written_action_selected_extension_convention_selector_"
    "selected_extension_family_no_go_theorem_derived_selected_extension_"
    "convention_selector_selected_extension_selector_primary_pack_refresh_"
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


# 関数: selected-extension-family theorem で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the selector selected-extension-family theorem audit."""
    return {
        "selector_representative_family": (
            "Rep_sel_conv_ext[B_sel_conv_ext;W,K] := { A'_conv_ext in "
            "[A_conv_ext]_conv_ext | A'_conv_ext[chi_i] = a_i for all i }"
        ),
        "selected_extension_formula": (
            "Sigma_sel_conv_ext,*^(B_sel_conv_ext;B_conv_ext;W,K,A_conv_ext) := "
            "Sigma_*^(W;K,chi_*^(B_conv_ext;W,K,A_conv_ext))"
        ),
        "selected_extension_family": (
            "Ext_sel_conv_ext[B_sel_conv_ext;B_conv_ext;W,K] := { "
            "Sigma_sel_conv_ext,*^(B_sel_conv_ext;B_conv_ext;W,K,A_conv_ext) | "
            "A_conv_ext in Rep_sel_conv_ext[B_sel_conv_ext;W,K] }"
        ),
        "selected_extension_family_no_go": (
            "without one concrete selector representative A_conv_ext, the current "
            "theorem stack still fixes only the admissible selected-extension "
            "family Ext_sel_conv_ext[...] and not one canonical selected extension"
        ),
    }


# 関数: `.4807-.4810` を実行する。

def main() -> None:
    """Execute the selector selected-extension-family theorem audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT, PRIOR_SUPPORT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    prior_support_summary = sign_base.read_json(PRIOR_SUPPORT)["summary"]

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    selected_extension_no_go_available = bool(
        prior_gate_summary[
            "gate_a_updated_pack_exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_no_go_available_now"
        ]
    )
    selected_extension_formula_available = bool(
        prior_audit_summary[
            "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_formula_available_now"
        ]
    )
    selector_representative_family_available = bool(
        prior_support_summary[
            "exact_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_family_formula_available_now"
        ]
    )
    selector_selected_extension_family_requirement_available = bool(
        prior_audit_summary[
            "exact_minimal_selected_extension_convention_selector_selected_extension_family_requirement_theorem_available_now"
        ]
    )
    selected_extension_family_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selected_extension_no_go_available
        and selected_extension_formula_available
        and selector_representative_family_available
        and selector_selected_extension_family_requirement_available
    )
    exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_formula_available_now = bool(
        selected_extension_family_explicit
    )
    exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_no_go_theorem_available_now = bool(
        selected_extension_family_explicit
    )
    exact_minimal_selected_extension_convention_selector_selected_extension_selector_requirement_theorem_available_now = bool(
        selected_extension_family_explicit
    )
    exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_available_now = False
    updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_selector_primary_followup_required = bool(
        exact_minimal_selected_extension_convention_selector_selected_extension_selector_requirement_theorem_available_now
    )
    updated_pack_corrected_pack_refresh_secondary_hold_retained = bool(
        updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_selector_primary_followup_required
    )
    updated_pack_same_tag_pack_refresh_reentry_admissible_now = False
    updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_breakthrough_passed_now = False
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])

    rows = [
        sign_base.row(
            "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack beyond-current-written-action selected extension convention selector selected extension family audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after selector selected-extension underdetermination is already closed and same-tag reentry remains shut.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This turn must close a new theorem object rather than restate selected-extension underdetermination in new words.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The selector selected-extension-family theorem is admissible only if it does not reopen the exhausted surrogate family.",
        ),
        sign_base.row(
            "gate_a_updated_pack_exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_no_go_available_now",
            "pass" if selected_extension_no_go_available else "reject",
            "gate A exact beyond-current-written-action selected extension convention selector selected extension no-go available now",
            sign_base.truth(selected_extension_no_go_available),
            "The family theorem starts only after the current theory already closes that it cannot choose one canonical selected extension under unresolved selector representatives.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_formula_available_now",
            "pass" if selected_extension_formula_available else "reject",
            "exact beyond-current-written-action selected extension convention selector selected extension formula available now",
            sign_base.truth(selected_extension_formula_available),
            "The family theorem uses the already closed selected-extension member formula Sigma_sel_conv_ext,*.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_family_formula_available_now",
            "pass" if selector_representative_family_available else "reject",
            "exact beyond-current-written-action selected extension convention selector selected candidate family formula available now",
            sign_base.truth(selector_representative_family_available),
            "The family theorem uses the already closed selector-representative family of admissible A_conv_ext choices.",
        ),
        sign_base.row(
            "exact_minimal_selected_extension_convention_selector_selected_extension_family_requirement_theorem_available_now",
            "pass" if selector_selected_extension_family_requirement_available else "reject",
            "exact minimal selected extension convention selector selected extension family requirement theorem available now",
            sign_base.truth(selector_selected_extension_family_requirement_available),
            "The prior branch already fixed that the honest next blocker is the family of selected extensions induced by unresolved selector representatives.",
        ),
        sign_base.row(
            "selected_extension_family_explicit",
            "pass" if selected_extension_family_explicit else "reject",
            "selected-extension convention selector selected-extension family explicit",
            sign_base.truth(selected_extension_family_explicit),
            "Once both Sigma_sel_conv_ext,* and Rep_sel_conv_ext are explicit, the honest next object is the induced selected-extension family Ext_sel_conv_ext.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_formula_available_now",
            "pass"
            if exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_formula_available_now
            else "reject",
            "exact beyond-current-written-action selected extension convention selector selected extension family formula available now",
            sign_base.truth(
                exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_formula_available_now
            ),
            "The theorem stack now fixes the admissible selected-extension family induced by unresolved selector representatives A_conv_ext.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_no_go_theorem_available_now",
            "pass"
            if exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_no_go_theorem_available_now
            else "reject",
            "exact beyond-current-written-action selected extension convention selector selected extension family no-go theorem available now",
            sign_base.truth(
                exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_no_go_theorem_available_now
            ),
            "Because current theory still does not choose one concrete selector representative A_conv_ext, it still fixes only the admissible selected-extension family and not one canonical selected extension.",
        ),
        sign_base.row(
            "exact_minimal_selected_extension_convention_selector_selected_extension_selector_requirement_theorem_available_now",
            "pass"
            if exact_minimal_selected_extension_convention_selector_selected_extension_selector_requirement_theorem_available_now
            else "reject",
            "exact minimal selected extension convention selector selected extension selector requirement theorem available now",
            sign_base.truth(
                exact_minimal_selected_extension_convention_selector_selected_extension_selector_requirement_theorem_available_now
            ),
            "The honest next blocker is now whether any selector on Ext_sel_conv_ext adds information beyond already unresolved selector-representative choice.",
        ),
        sign_base.row(
            "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_available_now",
            "pass"
            if exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_available_now
            else "reject",
            "exact beyond-current-written-action selected extension convention selector selected extension available now",
            sign_base.truth(
                exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_available_now
            ),
            "The theorem stack fixes only the selected-extension family and its no-go, not one concrete selected extension.",
        ),
        sign_base.row(
            "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_selector_primary_followup_required",
            "pass"
            if updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_selector_primary_followup_required
            else "reject",
            "updated-pack beyond-current-written-action selected extension convention selector selected extension selector primary followup required",
            sign_base.truth(
                updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_selector_primary_followup_required
            ),
            "The honest next blocker is the selector family on Ext_sel_conv_ext, not another same-tag route-sync.",
        ),
        sign_base.row(
            "updated_pack_corrected_pack_refresh_secondary_hold_retained",
            "pass" if updated_pack_corrected_pack_refresh_secondary_hold_retained else "reject",
            "updated-pack corrected pack-refresh secondary hold retained",
            sign_base.truth(updated_pack_corrected_pack_refresh_secondary_hold_retained),
            "Pack-refresh remains secondary because the blocker is still theorem-side selected-extension-selector completion.",
        ),
        sign_base.row(
            "updated_pack_same_tag_pack_refresh_reentry_admissible_now",
            "pass" if updated_pack_same_tag_pack_refresh_reentry_admissible_now else "reject",
            "updated-pack same-tag pack-refresh reentry admissible now",
            sign_base.truth(updated_pack_same_tag_pack_refresh_reentry_admissible_now),
            "Same-tag reentry remains closed because the blocker is theorem-side selected-extension-selector completion, not bookkeeping syntax.",
        ),
        sign_base.row(
            "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_breakthrough_passed_now",
            "pass"
            if updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_breakthrough_passed_now
            else "reject",
            "updated-pack beyond-current-written-action selected extension convention selector selected extension family breakthrough passed now",
            sign_base.truth(
                updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_breakthrough_passed_now
            ),
            "This branch sharpens selected-extension-family underdetermination but still does not choose one concrete selected extension.",
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
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_audit_selected": audit_selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "gate_a_updated_pack_exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_no_go_available_now": selected_extension_no_go_available,
        "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_formula_available_now": selected_extension_formula_available,
        "exact_beyond_current_written_action_selected_extension_convention_selector_selected_candidate_family_formula_available_now": selector_representative_family_available,
        "exact_minimal_selected_extension_convention_selector_selected_extension_family_requirement_theorem_available_now": selector_selected_extension_family_requirement_available,
        "selected_extension_family_explicit": selected_extension_family_explicit,
        "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_formula_available_now": exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_formula_available_now,
        "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_no_go_theorem_available_now": exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_no_go_theorem_available_now,
        "exact_minimal_selected_extension_convention_selector_selected_extension_selector_requirement_theorem_available_now": exact_minimal_selected_extension_convention_selector_selected_extension_selector_requirement_theorem_available_now,
        "exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_available_now": exact_beyond_current_written_action_selected_extension_convention_selector_selected_extension_available_now,
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_selector_primary_followup_required": updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_selector_primary_followup_required,
        "updated_pack_corrected_pack_refresh_secondary_hold_retained": updated_pack_corrected_pack_refresh_secondary_hold_retained,
        "updated_pack_same_tag_pack_refresh_reentry_admissible_now": updated_pack_same_tag_pack_refresh_reentry_admissible_now,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_breakthrough_passed_now": updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_breakthrough_passed_now,
        "selected_primary_completion_lane": "updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_selector_theorem_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_selector_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.4815",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_gate",
        "selected_followup_route_or_none": "8.7.56.4811",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.4809",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_selected_extension_audit": sign_base.display_path(PRIOR_AUDIT),
                "prior_selected_candidate_audit": sign_base.display_path(PRIOR_SUPPORT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.4815",
                "followup_route": "8.7.56.4811",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_beyond_current_written_action_selected_extension_convention_selector_selected_extension_family_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(
        json.dumps(
            {
                "json": declaration_paths["json"],
                "classification": BRANCH_CLASS,
                "breakthrough_passed_now": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
