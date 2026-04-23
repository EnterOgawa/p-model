#!/usr/bin/env python3
"""Generate 8.7.56.2775-.2778 exact external-probe current-vertex audit artifacts."""

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
        "8.7.56.2771-2774",
        "updated_pack_pure_derivation_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2767-2770",
        "updated_pack_pure_derivation_probe_split_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PURE_DERIVATION_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_pure_derivation_20260330.md"
)

STEP_TAG = "8.7.56.2775-2778"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack exact "
    "external-probe current-vertex audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_exact_external_probe_current_vertex_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_pure_"
    "derivation_probe_split_audited_external_probe_current_vertex_primary_"
    "mixed_probe_response_secondary_hybrid_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_probe_current_vertex_audited_mixed_probe_response_primary_"
    "vacuum_subtraction_tertiary_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_external_probe_"
    "gate_mixed_response_refresh"
)
NEXT_ROUTE = "8.7.56.2779"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_mixed_"
    "probe_response_kernel_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2783"


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

    return {
        "json": sign_base.display_path(paths["json"]),
        "csv": sign_base.display_path(paths["csv"]),
    }


# 関数: current-vertex audit で使う式を返す。

def build_formulae(prior_formulas: dict[str, str]) -> dict[str, str]:
    """Return formulas used in the external-probe current-vertex audit."""
    return {
        "background_stationarity": prior_formulas["background_stationarity"],
        "probe_split_definition": (
            "P_mu(x) = Q_mu(x) + xi_mu(x),   A_mu(x): external probe"
        ),
        "external_probe_current": (
            "J_ext^mu[Q](x) := delta S_frozen[Q;A] / delta A_mu(x) |_(A=0)"
        ),
        "kinetic_linear_probe_term": (
            "delta L_kin^(1) ~ -(Z_P/2) Re[F_Q^{*mu nu} f_mu_nu[A]]"
        ),
        "potential_linear_probe_term": (
            "delta Phi_1[A] = Q_mu^* A^mu + A_mu^* Q^mu,   "
            "S_pot^(1)[Q;A] = int d^4x U'(Phi_Q) delta Phi_1[A]"
        ),
        "mixed_probe_kernel": (
            "V^{mu nu}[Q](x,y) := delta^2 S_frozen / (delta xi_mu(x) delta A_nu(y))"
            " |_(Q,A=0)"
        ),
        "vacuum_subtraction_hold_rule": (
            "Vacuum subtraction is downstream of the rank-matched current/kernel "
            "selection and cannot be treated as the last piece beforehand."
        ),
    }


# 関数: `.2775-.2778` を実行する。

def main() -> None:
    """Execute the updated-pack exact external-probe current-vertex audit."""
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
        PRIOR_AUDIT,
        PURE_DERIVATION_NOTE,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)
    note_text = sign_base.read_text(PURE_DERIVATION_NOTE)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_payload = sign_base.read_json(PRIOR_AUDIT)
    prior_audit_summary = prior_audit_payload["summary"]
    prior_formulas = prior_audit_payload["evidence"]["formulae"]

    updated_pack_exact_external_probe_current_vertex_audit_selected = bool(
        prior_gate_summary["gate_b_updated_pack_exact_external_probe_current_vertex_promoted_next"]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_gate_computation_mode_selected = bool(
        prior_gate_summary["retry_gate_computation_mode_selected"]
    )
    current_pack_same_field_no_go_fixed = bool(
        prior_audit_summary["current_pack_same_field_no_go_fixed"]
    )
    failure_matrix_non_surrogate_guard_preserved = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    pure_derivation_single_split_only_explicit = bool(
        sign_base.hit(note_text, "P_\\mu(x) = P_\\mu^{\\rm Q}(x) + a_\\mu(x)") is not None
    )
    pure_derivation_on_shell_zero_explicit = bool(
        sign_base.hit(note_text, "S^{(1)} = 0") is not None
        or sign_base.hit(note_text, "S^(1) = 0") is not None
    )
    pure_derivation_kinetic_linear_cross_term_explicit = bool(
        sign_base.hit(note_text, "2F^Q_{\\mu\\nu}f^{\\mu\\nu}") is not None
    )
    pure_derivation_potential_linear_delta_phi_explicit = bool(
        sign_base.hit(note_text, "\\delta\\Phi_1 = Q_\\mu^* a^\\mu + a_\\mu^* Q^\\mu")
        is not None
    )
    pure_derivation_hessian_surface_explicit = bool(
        prior_audit_summary["pure_derivation_note_hessian_surface_explicit"]
    )
    corrected_probe_split_symbol_available_now = False
    exact_external_probe_current_vertex_target_surface_explicit = bool(
        updated_pack_exact_external_probe_current_vertex_audit_selected
        and retry_gate_computation_mode_selected
        and current_pack_same_field_no_go_fixed
        and failure_matrix_non_surrogate_guard_preserved
        and pure_derivation_single_split_only_explicit
        and pure_derivation_on_shell_zero_explicit
        and pure_derivation_kinetic_linear_cross_term_explicit
        and pure_derivation_potential_linear_delta_phi_explicit
    )
    updated_pack_external_probe_current_vertex_machine_readable_now = bool(
        exact_external_probe_current_vertex_target_surface_explicit
        and pure_derivation_hessian_surface_explicit
    )
    exact_external_probe_current_vertex_formula_available_now = False
    exact_external_probe_current_support_or_no_go_verdict_available_now = False
    updated_pack_external_probe_current_vertex_fully_localized_now = bool(
        updated_pack_external_probe_current_vertex_machine_readable_now
        and (not corrected_probe_split_symbol_available_now)
    )
    updated_pack_mixed_probe_response_primary_followup_required = bool(
        updated_pack_external_probe_current_vertex_fully_localized_now
        and (not exact_external_probe_current_vertex_formula_available_now)
    )
    updated_pack_vacuum_subtraction_tertiary_hold_retained = bool(
        updated_pack_mixed_probe_response_primary_followup_required
        and prior_audit_summary["pure_derivation_vacuum_subtraction_not_last_piece_now"]
    )
    updated_pack_external_probe_current_vertex_breakthrough_passed_now = False
    blind_vector_observable_gate_still_blocked = bool(
        prior_gate_summary["blind_vector_observable_gate_still_blocked"]
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_exact_external_probe_current_vertex_audit_selected",
            "pass" if updated_pack_exact_external_probe_current_vertex_audit_selected else "reject",
            "updated-pack exact external-probe current-vertex audit selected",
            sign_base.truth(updated_pack_exact_external_probe_current_vertex_audit_selected),
            "The pure-derivation gate already promoted exact external-probe current-vertex completion as the next honest derivation lane.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_gate_computation_mode_selected else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_gate_computation_mode_selected),
            "The lane remains on the derivation side rather than falling back into another registry-only cycle.",
        ),
        sign_base.row(
            "current_pack_same_field_no_go_fixed",
            "pass" if current_pack_same_field_no_go_fixed else "reject",
            "current-pack same-field no-go fixed",
            sign_base.truth(current_pack_same_field_no_go_fixed),
            "The external-probe lane starts only after the same-field one-point source route is already closed on the exact zero / no-go branch.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if failure_matrix_non_surrogate_guard_preserved else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(failure_matrix_non_surrogate_guard_preserved),
            "The external-probe lane stays admissible only if it does not silently reopen the exhausted density/proxy/eigenvalue family.",
        ),
        sign_base.row(
            "pure_derivation_single_split_only_explicit",
            "pass" if pure_derivation_single_split_only_explicit else "reject",
            "pure-derivation single split only explicit",
            sign_base.truth(pure_derivation_single_split_only_explicit),
            "The note still uses one fluctuation symbol a_mu for every role, so the self/probe split is not yet implemented directly in the source note.",
        ),
        sign_base.row(
            "pure_derivation_on_shell_zero_explicit",
            "pass" if pure_derivation_on_shell_zero_explicit else "reject",
            "pure-derivation on-shell zero explicit",
            sign_base.truth(pure_derivation_on_shell_zero_explicit),
            "The note correctly exposes S^(1)=0, but only for the on-shell background fluctuation expansion.",
        ),
        sign_base.row(
            "pure_derivation_kinetic_linear_cross_term_explicit",
            "pass" if pure_derivation_kinetic_linear_cross_term_explicit else "reject",
            "pure-derivation kinetic linear cross term explicit",
            sign_base.truth(pure_derivation_kinetic_linear_cross_term_explicit),
            "The note already contains the 2F^Q f cross term that must be kept before collapsing the rank-matched external probe current.",
        ),
        sign_base.row(
            "pure_derivation_potential_linear_delta_phi_explicit",
            "pass" if pure_derivation_potential_linear_delta_phi_explicit else "reject",
            "pure-derivation potential linear delta-phi explicit",
            sign_base.truth(pure_derivation_potential_linear_delta_phi_explicit),
            "The potential-side linear variation is also explicit before any quadratic Hessian fallback is chosen.",
        ),
        sign_base.row(
            "corrected_probe_split_symbol_available_now",
            "pass" if corrected_probe_split_symbol_available_now else "reject",
            "corrected distinct probe symbol available now",
            sign_base.truth(corrected_probe_split_symbol_available_now),
            "The note does not yet separate self fluctuation xi_mu from external probe A_mu, so the exact one-point current object is still not literal there.",
        ),
        sign_base.row(
            "exact_external_probe_current_vertex_target_surface_explicit",
            "pass" if exact_external_probe_current_vertex_target_surface_explicit else "reject",
            "exact external-probe current-vertex target surface explicit",
            sign_base.truth(exact_external_probe_current_vertex_target_surface_explicit),
            "What survives the audit is the rank-matched target itself: derive J_ext^mu[Q] from the frozen action, keeping the linear probe terms before any Hessian collapse.",
        ),
        sign_base.row(
            "updated_pack_external_probe_current_vertex_machine_readable_now",
            "pass" if updated_pack_external_probe_current_vertex_machine_readable_now else "reject",
            "updated-pack external-probe current-vertex stack machine-readable now",
            sign_base.truth(updated_pack_external_probe_current_vertex_machine_readable_now),
            "The one-point external-probe lane is now explicit as a derivation stack rather than as a vague vacuum-subtraction intuition.",
        ),
        sign_base.row(
            "exact_external_probe_current_vertex_formula_available_now",
            "pass" if exact_external_probe_current_vertex_formula_available_now else "reject",
            "exact external-probe current-vertex formula available now",
            sign_base.truth(exact_external_probe_current_vertex_formula_available_now),
            "The note does not yet supply the exact frozen-action one-point current formula after the self/probe split is corrected.",
        ),
        sign_base.row(
            "exact_external_probe_current_support_or_no_go_verdict_available_now",
            "pass" if exact_external_probe_current_support_or_no_go_verdict_available_now else "reject",
            "exact external-probe current support/no-go verdict available now",
            sign_base.truth(exact_external_probe_current_support_or_no_go_verdict_available_now),
            "Without the exact one-point current formula, neither a support verdict nor a no-go verdict can yet be fixed theorem-level here.",
        ),
        sign_base.row(
            "updated_pack_external_probe_current_vertex_fully_localized_now",
            "pass" if updated_pack_external_probe_current_vertex_fully_localized_now else "reject",
            "updated-pack external-probe current-vertex fully localized now",
            sign_base.truth(updated_pack_external_probe_current_vertex_fully_localized_now),
            "The blocker is no longer vacuum subtraction or reserve bookkeeping; it is the missing literal self/probe split and the exact one-point current formula.",
        ),
        sign_base.row(
            "updated_pack_mixed_probe_response_primary_followup_required",
            "pass" if updated_pack_mixed_probe_response_primary_followup_required else "reject",
            "updated-pack mixed probe-response primary followup required",
            sign_base.truth(updated_pack_mixed_probe_response_primary_followup_required),
            "If the one-point external current is still unavailable, the next honest fallback is the mixed probe-response kernel rather than another same-level wording retry.",
        ),
        sign_base.row(
            "updated_pack_vacuum_subtraction_tertiary_hold_retained",
            "pass" if updated_pack_vacuum_subtraction_tertiary_hold_retained else "reject",
            "updated-pack vacuum subtraction tertiary hold retained",
            sign_base.truth(updated_pack_vacuum_subtraction_tertiary_hold_retained),
            "Vacuum subtraction remains downstream of the rank-matched current or kernel selection and cannot honestly be promoted ahead of them.",
        ),
        sign_base.row(
            "updated_pack_external_probe_current_vertex_breakthrough_passed_now",
            "pass" if updated_pack_external_probe_current_vertex_breakthrough_passed_now else "reject",
            "updated-pack external-probe current-vertex breakthrough passed now",
            sign_base.truth(updated_pack_external_probe_current_vertex_breakthrough_passed_now),
            "The branch sharpens the blocker but does not yet close the missing action-level term.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Blind-vector direct computation still waits on the exact probe/current object and its downstream theorem stack.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains reserve-only because the blocker is still the missing rank-matched probe object.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_exact_external_probe_current_vertex_audit_selected": updated_pack_exact_external_probe_current_vertex_audit_selected,
        "retry_gate_computation_mode_selected": retry_gate_computation_mode_selected,
        "current_pack_same_field_no_go_fixed": current_pack_same_field_no_go_fixed,
        "failure_matrix_non_surrogate_guard_preserved": failure_matrix_non_surrogate_guard_preserved,
        "pure_derivation_single_split_only_explicit": pure_derivation_single_split_only_explicit,
        "pure_derivation_on_shell_zero_explicit": pure_derivation_on_shell_zero_explicit,
        "pure_derivation_kinetic_linear_cross_term_explicit": pure_derivation_kinetic_linear_cross_term_explicit,
        "pure_derivation_potential_linear_delta_phi_explicit": pure_derivation_potential_linear_delta_phi_explicit,
        "corrected_probe_split_symbol_available_now": corrected_probe_split_symbol_available_now,
        "exact_external_probe_current_vertex_target_surface_explicit": exact_external_probe_current_vertex_target_surface_explicit,
        "updated_pack_external_probe_current_vertex_machine_readable_now": updated_pack_external_probe_current_vertex_machine_readable_now,
        "exact_external_probe_current_vertex_formula_available_now": exact_external_probe_current_vertex_formula_available_now,
        "exact_external_probe_current_support_or_no_go_verdict_available_now": exact_external_probe_current_support_or_no_go_verdict_available_now,
        "updated_pack_external_probe_current_vertex_fully_localized_now": updated_pack_external_probe_current_vertex_fully_localized_now,
        "updated_pack_mixed_probe_response_primary_followup_required": updated_pack_mixed_probe_response_primary_followup_required,
        "updated_pack_vacuum_subtraction_tertiary_hold_retained": updated_pack_vacuum_subtraction_tertiary_hold_retained,
        "updated_pack_external_probe_current_vertex_breakthrough_passed_now": updated_pack_external_probe_current_vertex_breakthrough_passed_now,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_exact_external_probe_current_vertex",
        "selected_secondary_pack_update_surface": "updated_pack_exact_mixed_probe_response_kernel",
        "selected_tertiary_pack_update_surface": "updated_pack_vacuum_subtraction_hold",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2777",
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
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "pure_derivation_note": sign_base.display_path(PURE_DERIVATION_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_external_probe_current_vertex_declared",
            "branch_completed": True,
            "breakthrough_passed_now": updated_pack_external_probe_current_vertex_breakthrough_passed_now,
            "physical_reject_required": False,
        },
        {
            "formulas": build_formulae(prior_formulas),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2775"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2771-.2774"),
                "current_problem_hit": sign_base.hit(
                    current_problem_text,
                    "exact external-probe current-vertex audit",
                ),
                "current_status_hit": sign_base.hit(
                    current_status_text,
                    "exact external-probe current-vertex audit",
                ),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2771-.2774"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2767-.2774"),
                "part5_hit": sign_base.hit(
                    part5_text,
                    "external-probe current vertex",
                ),
                "note_split_hit": sign_base.hit(
                    note_text,
                    "P_\\mu(x) = P_\\mu^{\\rm Q}(x) + a_\\mu(x)",
                ),
                "note_on_shell_hit": sign_base.hit(note_text, "S^(1) = 0"),
                "note_kinetic_hit": sign_base.hit(note_text, "2F^Q_{\\mu\\nu}f^{\\mu\\nu}"),
                "note_potential_hit": sign_base.hit(
                    note_text,
                    "\\delta\\Phi_1 = Q_\\mu^* a^\\mu + a_\\mu^* Q^\\mu",
                ),
            },
            "inference": {
                "why_no_breakthrough_now": (
                    "The note exposes the linear kinetic and potential ingredients but "
                    "still keeps self fluctuation and external probe under one symbol. "
                    "That leaves the exact one-point external current formula absent."
                ),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = sign_base.payload(
        "8.7.56.2778",
        STEP_NAME + " route sync",
        {
            "source_files": declaration_payload["inputs"]["source_files"],
            "declaration": {},
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_external_probe_current_vertex_route_synced",
            "branch_completed": True,
            "breakthrough_passed_now": updated_pack_external_probe_current_vertex_breakthrough_passed_now,
            "physical_reject_required": False,
        },
        {
            "formulas": build_formulae(prior_formulas),
            "notes": {
                "primary_transition": (
                    "The current one-point external-probe lane is now explicit, but "
                    "the honest fallback remains the mixed probe-response kernel."
                ),
                "tertiary_hold": (
                    "Vacuum subtraction stays tertiary until the rank-matched current "
                    "or kernel object is canonical."
                ),
            },
        },
    )
    write_artifact("route_sync", route_payload)

    print(f"[done] {STEP_TAG} updated-pack exact external-probe current-vertex audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
