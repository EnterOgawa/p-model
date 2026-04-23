#!/usr/bin/env python3
"""Generate 8.7.56.4519-.4522 corrected probe-split additive no-go artifacts."""

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

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
LONG_ROADMAP = ROOT / "doc" / "quantum" / "55_trial2_numeric_alpha_vector_qball_long_horizon_roadmap.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.4515-4518",
        "updated_pack_corrected_pack_refresh_return_gate_probe_split_reset",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_REPEAT_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.4511-4514",
        "updated_pack_corrected_pack_refresh_return_repeat_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PURE_DERIVATION_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_pure_derivation_20260330.md"
)

STEP_TAG = "8.7.56.4519-4522"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack corrected "
    "probe-split additive one-point-current no-go theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_corrected_probe_split_additive_no_go_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_pack_refresh_return_cycle_repeat_detected_probe_split_primary_"
    "mixed_kernel_secondary_hybrid_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_probe_split_additive_one_point_no_go_theorem_audited_"
    "mixed_kernel_primary_vacuum_subtraction_secondary_gate"
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


# 関数: additive corrected probe-split no-go theorem で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the corrected probe-split additive no-go audit."""
    return {
        "additive_probe_split": "P_mu(x) = Q_mu(x) + xi_mu(x) + A_mu(x)",
        "linear_additive_expansion": (
            "S[Q + xi + A] = S[Q] + int d^4x (xi_mu + A_mu) "
            "(delta S / delta P_mu)|_(P=Q) + O((xi,A)^2)"
        ),
        "background_stationarity": (
            "Q on shell => (delta S / delta P_mu)|_(P=Q) = 0"
        ),
        "additive_one_point_current": (
            "J_add^mu[Q](x) := delta S[Q+xi+A] / delta A_mu(x) |_(xi=0,A=0) "
            "= (delta S / delta P_mu)|_(P=Q) = 0"
        ),
        "mixed_kernel_fallback": (
            "If J_add^mu[Q] vanishes exactly under the additive split, the first "
            "honest observable object must appear in V^{mu nu}[Q] or "
            "Pi^{mu nu}[Q]."
        ),
        "mixed_probe_kernel": (
            "V^{mu nu}[Q](x,y) := delta^2 S / (delta xi_mu(x) delta A_nu(y))|_(Q)"
        ),
        "pure_probe_kernel": (
            "Pi^{mu nu}[Q](x,y) := delta^2 S / (delta A_mu(x) delta A_nu(y))|_(Q)"
        ),
    }


# 関数: `.4519-.4522` を実行する。

def main() -> None:
    """Execute the corrected probe-split additive no-go theorem audit."""
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
        PRIOR_GATE,
        PRIOR_REPEAT_AUDIT,
        PURE_DERIVATION_NOTE,
    ):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_repeat_summary = sign_base.read_json(PRIOR_REPEAT_AUDIT)["summary"]
    note_text = sign_base.read_text(PURE_DERIVATION_NOTE)

    audit_selected = bool(
        prior_gate_summary["gate_b_updated_pack_corrected_probe_split_return_promoted_next"]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    repeat_hard_stop_explicit = bool(
        prior_repeat_summary[
            "updated_pack_corrected_pack_refresh_return_cycle_exhaustion_machine_readable_now"
        ]
        and prior_repeat_summary[
            "updated_pack_corrected_pack_refresh_return_cycle_repeat_detected"
        ]
        and prior_repeat_summary[
            "updated_pack_corrected_pack_refresh_return_no_new_public_canonical_surface_now"
        ]
    )
    pure_derivation_single_split_only_explicit = bool(
        sign_base.hit(note_text, "P_\\mu(x) = P_\\mu^{\\rm Q}(x) + a_\\mu(x)")
        is not None
    )
    pure_derivation_on_shell_zero_explicit = bool(
        sign_base.hit(note_text, "S^(1) = 0") is not None
        or sign_base.hit(note_text, "S^{(1)} = 0") is not None
    )
    corrected_additive_probe_split_formula_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and repeat_hard_stop_explicit
        and pure_derivation_single_split_only_explicit
        and pure_derivation_on_shell_zero_explicit
    )
    corrected_additive_linear_term_zero_theorem_explicit = bool(
        corrected_additive_probe_split_formula_explicit
    )
    exact_corrected_probe_split_formula_available_now = bool(
        corrected_additive_probe_split_formula_explicit
    )
    exact_external_probe_current_vertex_formula_available_now = bool(
        corrected_additive_linear_term_zero_theorem_explicit
    )
    exact_external_probe_current_vertex_zero_under_additive_split = bool(
        corrected_additive_linear_term_zero_theorem_explicit
    )
    exact_external_probe_current_one_point_no_go_theorem_available_now = bool(
        exact_external_probe_current_vertex_zero_under_additive_split
    )
    updated_pack_corrected_mixed_kernel_primary_followup_required = bool(
        exact_external_probe_current_one_point_no_go_theorem_available_now
    )
    updated_pack_corrected_vacuum_subtraction_secondary_hold_retained = bool(
        updated_pack_corrected_mixed_kernel_primary_followup_required
    )
    updated_pack_corrected_probe_split_breakthrough_passed_now = False
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])
    farther_hybrid = False

    rows = [
        sign_base.row(
            "updated_pack_corrected_probe_split_additive_no_go_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack corrected probe-split additive no-go audit selected",
            sign_base.truth(audit_selected),
            "The same-tag reserve loop was already marked exhausted, so the honest next move is to test whether the corrected probe split closes any exact theorem object at all.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This turn stays on the derivation side and does not treat another loop re-sync as progress.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The additive split theorem is only admissible if it does not silently reopen the exhausted density/proxy/eigenvalue family.",
        ),
        sign_base.row(
            "updated_pack_repeat_hard_stop_explicit",
            "pass" if repeat_hard_stop_explicit else "reject",
            "updated-pack repeat hard-stop explicit",
            sign_base.truth(repeat_hard_stop_explicit),
            "The current branch starts only after the corrected reserve-registry / corrected pack-refresh loop has already been certified as same-tag exhaustion.",
        ),
        sign_base.row(
            "pure_derivation_single_split_only_explicit",
            "pass" if pure_derivation_single_split_only_explicit else "reject",
            "pure-derivation single split only explicit",
            sign_base.truth(pure_derivation_single_split_only_explicit),
            "The source note still uses one additive fluctuation symbol, which is exactly what makes the one-point external current test well-posed here.",
        ),
        sign_base.row(
            "pure_derivation_on_shell_zero_explicit",
            "pass" if pure_derivation_on_shell_zero_explicit else "reject",
            "pure-derivation on-shell zero explicit",
            sign_base.truth(pure_derivation_on_shell_zero_explicit),
            "The on-shell background already kills the linear first variation, so any additive one-point probe term must inherit that zero.",
        ),
        sign_base.row(
            "corrected_additive_probe_split_formula_explicit",
            "pass" if corrected_additive_probe_split_formula_explicit else "reject",
            "corrected additive probe-split formula explicit",
            sign_base.truth(corrected_additive_probe_split_formula_explicit),
            "The corrected additive split P = Q + xi + A can now be written literally on the current theorem lane instead of being left as a placeholder target.",
        ),
        sign_base.row(
            "corrected_additive_linear_term_zero_theorem_explicit",
            "pass" if corrected_additive_linear_term_zero_theorem_explicit else "reject",
            "corrected additive linear term zero theorem explicit",
            sign_base.truth(corrected_additive_linear_term_zero_theorem_explicit),
            "Under the additive same-action split, the linear A_mu term collapses to the same on-shell zero as xi_mu, so the one-point source vanishes exactly.",
        ),
        sign_base.row(
            "exact_corrected_probe_split_formula_available_now",
            "pass" if exact_corrected_probe_split_formula_available_now else "reject",
            "exact corrected probe-split formula available now",
            sign_base.truth(exact_corrected_probe_split_formula_available_now),
            "The current branch now closes the additive corrected split formula itself, rather than merely restating it as a target surface.",
        ),
        sign_base.row(
            "exact_external_probe_current_vertex_formula_available_now",
            "pass" if exact_external_probe_current_vertex_formula_available_now else "reject",
            "exact external-probe current-vertex formula available now",
            sign_base.truth(exact_external_probe_current_vertex_formula_available_now),
            "What is available is the additive one-point current formula, and it closes as an exact zero rather than as a nontrivial source term.",
        ),
        sign_base.row(
            "exact_external_probe_current_vertex_zero_under_additive_split",
            "pass" if exact_external_probe_current_vertex_zero_under_additive_split else "reject",
            "exact external-probe current vertex zero under additive split",
            sign_base.truth(exact_external_probe_current_vertex_zero_under_additive_split),
            "This exact zero is the new theorem object: the additive same-field corrected split cannot generate a distinct one-point probe current on shell.",
        ),
        sign_base.row(
            "exact_external_probe_current_one_point_no_go_theorem_available_now",
            "pass" if exact_external_probe_current_one_point_no_go_theorem_available_now else "reject",
            "exact external-probe current one-point no-go theorem available now",
            sign_base.truth(exact_external_probe_current_one_point_no_go_theorem_available_now),
            "The one-point probe lane is now closed theorem-side as a no-go under the additive corrected split.",
        ),
        sign_base.row(
            "updated_pack_corrected_mixed_kernel_primary_followup_required",
            "pass" if updated_pack_corrected_mixed_kernel_primary_followup_required else "reject",
            "updated-pack corrected mixed-kernel primary followup required",
            sign_base.truth(updated_pack_corrected_mixed_kernel_primary_followup_required),
            "Because the additive one-point current closes as exact zero, the first honest observable candidate is now the mixed/pure probe-response kernel itself.",
        ),
        sign_base.row(
            "updated_pack_corrected_vacuum_subtraction_secondary_hold_retained",
            "pass" if updated_pack_corrected_vacuum_subtraction_secondary_hold_retained else "reject",
            "updated-pack corrected vacuum-subtraction secondary hold retained",
            sign_base.truth(updated_pack_corrected_vacuum_subtraction_secondary_hold_retained),
            "Vacuum subtraction remains downstream of the mixed-kernel selection and still cannot be promoted ahead of the rank-matched kernel object.",
        ),
        sign_base.row(
            "updated_pack_corrected_probe_split_breakthrough_passed_now",
            "pass" if updated_pack_corrected_probe_split_breakthrough_passed_now else "reject",
            "updated-pack corrected probe-split breakthrough passed now",
            sign_base.truth(updated_pack_corrected_probe_split_breakthrough_passed_now),
            "This branch closes a meaningful no-go theorem but does not yet explain the residual or reopen blind-vector computation.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on the corrected mixed-kernel / subtraction / reserve-verdict stack.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid),
            "Extra q-range evidence remains reserve-only because the blocker is theorem-side, not range-side.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_corrected_probe_split_additive_no_go_audit_selected": audit_selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_repeat_hard_stop_explicit": repeat_hard_stop_explicit,
        "pure_derivation_single_split_only_explicit": pure_derivation_single_split_only_explicit,
        "pure_derivation_on_shell_zero_explicit": pure_derivation_on_shell_zero_explicit,
        "corrected_additive_probe_split_formula_explicit": corrected_additive_probe_split_formula_explicit,
        "corrected_additive_linear_term_zero_theorem_explicit": corrected_additive_linear_term_zero_theorem_explicit,
        "exact_corrected_probe_split_formula_available_now": exact_corrected_probe_split_formula_available_now,
        "exact_external_probe_current_vertex_formula_available_now": exact_external_probe_current_vertex_formula_available_now,
        "exact_external_probe_current_vertex_zero_under_additive_split": exact_external_probe_current_vertex_zero_under_additive_split,
        "exact_external_probe_current_one_point_no_go_theorem_available_now": exact_external_probe_current_one_point_no_go_theorem_available_now,
        "updated_pack_corrected_mixed_kernel_primary_followup_required": updated_pack_corrected_mixed_kernel_primary_followup_required,
        "updated_pack_corrected_vacuum_subtraction_secondary_hold_retained": updated_pack_corrected_vacuum_subtraction_secondary_hold_retained,
        "updated_pack_corrected_probe_split_breakthrough_passed_now": updated_pack_corrected_probe_split_breakthrough_passed_now,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "selected_primary_pack_update_surface": "updated_pack_corrected_mixed_kernel_return_refresh_audit",
        "selected_secondary_pack_update_surface": "updated_pack_corrected_vacuum_subtraction_refresh",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_corrected_probe_split_additive_no_go_gate",
        "recommended_next_route_or_none": "8.7.56.4523",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_corrected_mixed_kernel_return_refresh_audit",
        "selected_followup_route_or_none": "8.7.56.4527",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.4521",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "ai_context": sign_base.display_path(AI_CONTEXT),
                "work_history_recent": sign_base.display_path(WORK_HISTORY_RECENT),
                "current_problem": sign_base.display_path(CURRENT_PROBLEM),
                "current_status": sign_base.display_path(CURRENT_STATUS),
                "unified_roadmap": sign_base.display_path(UNIFIED_ROADMAP),
                "long_roadmap": sign_base.display_path(LONG_ROADMAP),
                "part5": sign_base.display_path(PART5),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_repeat_audit": sign_base.display_path(PRIOR_REPEAT_AUDIT),
                "pure_derivation_note": sign_base.display_path(PURE_DERIVATION_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.4523",
                "followup_route": "8.7.56.4527",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_corrected_probe_split_additive_no_go_theorem_derived",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack corrected probe-split additive no-go audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
