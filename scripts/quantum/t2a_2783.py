#!/usr/bin/env python3
"""Generate 8.7.56.2783-.2786 exact mixed probe-response kernel audit artifacts."""

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
        "8.7.56.2779-2782",
        "updated_pack_external_probe_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2775-2778",
        "updated_pack_exact_external_probe_current_vertex_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PURE_DERIVATION_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_pure_derivation_20260330.md"
)

STEP_TAG = "8.7.56.2783-2786"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack exact mixed "
    "probe-response kernel audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_exact_mixed_probe_response_kernel_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "external_probe_current_vertex_audited_mixed_probe_response_primary_"
    "vacuum_subtraction_secondary_hybrid_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "mixed_probe_response_audited_kernel_completion_primary_vacuum_subtraction_"
    "secondary_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_mixed_probe_"
    "response_gate_pack_refresh"
)
NEXT_ROUTE = "8.7.56.2787"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_mixed_"
    "probe_response_kernel_completion_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2791"


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


# 関数: mixed-response audit で使う式を返す。

def build_formulae(prior_formulas: dict[str, str]) -> dict[str, str]:
    """Return formulas used in the mixed probe-response kernel audit."""
    return {
        "external_probe_current": prior_formulas["external_probe_current"],
        "self_fluctuation_hessian": (
            "H_{mu nu}[Q] = delta^2 S / (delta P_mu delta P_nu) |_(P=Q)"
        ),
        "mixed_probe_kernel": prior_formulas["mixed_probe_kernel"],
        "pure_probe_kernel": (
            "Pi^{mu nu}[Q](x,y) := delta^2 S_frozen / (delta A_mu(x) delta A_nu(y))"
            " |_(Q,A=0)"
        ),
        "kernel_selection_rule": (
            "If the one-point external current stays unavailable, the honest "
            "fallback is V^{mu nu}[Q] and then Pi^{mu nu}[Q], not H_{mu nu}[Q] "
            "or vacuum subtraction alone."
        ),
        "vacuum_subtraction_hold_rule": prior_formulas["vacuum_subtraction_hold_rule"],
    }


# 関数: `.2783-.2786` を実行する。

def main() -> None:
    """Execute the updated-pack exact mixed probe-response kernel audit."""
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
    prior_formulas = prior_audit_payload["evidence"].get(
        "formulas",
        prior_audit_payload["evidence"].get("formulae", {}),
    )

    updated_pack_exact_mixed_probe_response_kernel_audit_selected = bool(
        prior_gate_summary["gate_b_updated_pack_exact_mixed_probe_response_kernel_promoted_next"]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_gate_computation_mode_selected = bool(
        prior_gate_summary["retry_gate_computation_mode_selected"]
    )
    failure_matrix_non_surrogate_guard_preserved = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    updated_pack_external_probe_current_vertex_machine_readable_now = bool(
        prior_audit_summary["updated_pack_external_probe_current_vertex_machine_readable_now"]
    )
    pure_derivation_hessian_surface_explicit = bool(
        sign_base.hit(note_text, "delta^2 S") is not None
        or sign_base.hit(note_text, "S^{(2)}[Q;a]") is not None
    )
    pure_derivation_scattering_from_hessian_explicit = bool(
        sign_base.hit(note_text, "photon scattering amplitude") is not None
        and sign_base.hit(note_text, "\\tilde{\\mathcal{K}}_{\\mu\\nu}") is not None
    )
    pure_derivation_vacuum_subtraction_claim_explicit = bool(
        sign_base.hit(note_text, "vacuum subtraction") is not None
        and sign_base.hit(note_text, "唯一の不確定") is not None
    )
    note_scattering_kernel_explicit = bool(
        sign_base.hit(note_text, "\\tilde{\\mathcal{K}}_{\\mu\\nu}") is not None
    )
    updated_pack_mixed_probe_response_kernel_target_surface_explicit = bool(
        updated_pack_exact_mixed_probe_response_kernel_audit_selected
        and retry_gate_computation_mode_selected
        and failure_matrix_non_surrogate_guard_preserved
        and updated_pack_external_probe_current_vertex_machine_readable_now
        and pure_derivation_hessian_surface_explicit
        and pure_derivation_scattering_from_hessian_explicit
        and note_scattering_kernel_explicit
    )
    updated_pack_pure_probe_response_kernel_target_surface_explicit = bool(
        updated_pack_mixed_probe_response_kernel_target_surface_explicit
        and pure_derivation_vacuum_subtraction_claim_explicit
    )
    updated_pack_mixed_probe_response_kernel_machine_readable_now = bool(
        updated_pack_mixed_probe_response_kernel_target_surface_explicit
    )
    exact_mixed_probe_response_kernel_formula_available_now = False
    exact_pure_probe_response_kernel_formula_available_now = False
    exact_vacuum_subtraction_rule_available_now = False
    updated_pack_mixed_probe_response_kernel_fully_localized_now = bool(
        updated_pack_mixed_probe_response_kernel_machine_readable_now
    )
    updated_pack_exact_mixed_probe_response_completion_primary_followup_required = bool(
        updated_pack_mixed_probe_response_kernel_fully_localized_now
        and (not exact_mixed_probe_response_kernel_formula_available_now)
    )
    updated_pack_vacuum_subtraction_secondary_followup_required = bool(
        updated_pack_exact_mixed_probe_response_completion_primary_followup_required
        and (not exact_vacuum_subtraction_rule_available_now)
    )
    updated_pack_mixed_probe_response_breakthrough_passed_now = False
    blind_vector_observable_gate_still_blocked = bool(
        prior_gate_summary["blind_vector_observable_gate_still_blocked"]
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_exact_mixed_probe_response_kernel_audit_selected",
            "pass" if updated_pack_exact_mixed_probe_response_kernel_audit_selected else "reject",
            "updated-pack exact mixed probe-response kernel audit selected",
            sign_base.truth(updated_pack_exact_mixed_probe_response_kernel_audit_selected),
            "The external-probe gate already promoted the mixed probe-response object as the next honest fallback lane.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_gate_computation_mode_selected else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_gate_computation_mode_selected),
            "The mixed-response lane continues the derivation reset rather than reopening bookkeeping-only cycles.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if failure_matrix_non_surrogate_guard_preserved else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(failure_matrix_non_surrogate_guard_preserved),
            "The fallback kernel lane remains admissible only if it does not silently return to the exhausted scalar surrogate family.",
        ),
        sign_base.row(
            "updated_pack_external_probe_current_vertex_machine_readable_now",
            "pass" if updated_pack_external_probe_current_vertex_machine_readable_now else "reject",
            "updated-pack external-probe current-vertex stack machine-readable now",
            sign_base.truth(updated_pack_external_probe_current_vertex_machine_readable_now),
            "The mixed-response lane starts only after the one-point current target has already been made explicit as the missing primary object.",
        ),
        sign_base.row(
            "pure_derivation_hessian_surface_explicit",
            "pass" if pure_derivation_hessian_surface_explicit else "reject",
            "pure-derivation Hessian surface explicit",
            sign_base.truth(pure_derivation_hessian_surface_explicit),
            "The note does expose the quadratic Hessian, which becomes relevant only after the one-point current route fails to close.",
        ),
        sign_base.row(
            "pure_derivation_scattering_from_hessian_explicit",
            "pass" if pure_derivation_scattering_from_hessian_explicit else "reject",
            "pure-derivation scattering from Hessian explicit",
            sign_base.truth(pure_derivation_scattering_from_hessian_explicit),
            "The note explicitly tries to read a scattering object directly from the Hessian-side kernel.",
        ),
        sign_base.row(
            "note_scattering_kernel_explicit",
            "pass" if note_scattering_kernel_explicit else "reject",
            "note scattering kernel explicit",
            sign_base.truth(note_scattering_kernel_explicit),
            "The kernel notation itself is explicit enough to promote a rank-corrected mixed-response audit.",
        ),
        sign_base.row(
            "updated_pack_mixed_probe_response_kernel_target_surface_explicit",
            "pass" if updated_pack_mixed_probe_response_kernel_target_surface_explicit else "reject",
            "updated-pack mixed probe-response kernel target surface explicit",
            sign_base.truth(updated_pack_mixed_probe_response_kernel_target_surface_explicit),
            "The honest fallback object is now explicit: derive the mixed probe-response kernel V^{mu nu}[Q] before treating the Hessian as canonical scattering data.",
        ),
        sign_base.row(
            "updated_pack_pure_probe_response_kernel_target_surface_explicit",
            "pass" if updated_pack_pure_probe_response_kernel_target_surface_explicit else "reject",
            "updated-pack pure probe-response kernel target surface explicit",
            sign_base.truth(updated_pack_pure_probe_response_kernel_target_surface_explicit),
            "If the mixed kernel is still insufficient, the pure probe-response Pi^{mu nu}[Q] is the honest downstream fallback before subtraction.",
        ),
        sign_base.row(
            "updated_pack_mixed_probe_response_kernel_machine_readable_now",
            "pass" if updated_pack_mixed_probe_response_kernel_machine_readable_now else "reject",
            "updated-pack mixed probe-response kernel stack machine-readable now",
            sign_base.truth(updated_pack_mixed_probe_response_kernel_machine_readable_now),
            "The kernel lane is now explicit as a rank-corrected fallback rather than as a loose scattering-from-Hessian intuition.",
        ),
        sign_base.row(
            "exact_mixed_probe_response_kernel_formula_available_now",
            "pass" if exact_mixed_probe_response_kernel_formula_available_now else "reject",
            "exact mixed probe-response kernel formula available now",
            sign_base.truth(exact_mixed_probe_response_kernel_formula_available_now),
            "The note does not yet derive the exact mixed probe-response kernel after the self/probe split is corrected.",
        ),
        sign_base.row(
            "exact_pure_probe_response_kernel_formula_available_now",
            "pass" if exact_pure_probe_response_kernel_formula_available_now else "reject",
            "exact pure probe-response kernel formula available now",
            sign_base.truth(exact_pure_probe_response_kernel_formula_available_now),
            "The exact pure probe-response kernel is also still absent at this branch.",
        ),
        sign_base.row(
            "exact_vacuum_subtraction_rule_available_now",
            "pass" if exact_vacuum_subtraction_rule_available_now else "reject",
            "exact vacuum subtraction rule available now",
            sign_base.truth(exact_vacuum_subtraction_rule_available_now),
            "Because the canonical kernel object is still unresolved, vacuum subtraction still cannot honestly be finalized here.",
        ),
        sign_base.row(
            "updated_pack_mixed_probe_response_kernel_fully_localized_now",
            "pass" if updated_pack_mixed_probe_response_kernel_fully_localized_now else "reject",
            "updated-pack mixed probe-response kernel fully localized now",
            sign_base.truth(updated_pack_mixed_probe_response_kernel_fully_localized_now),
            "The fallback blocker is now localized to the exact mixed/pure probe kernel formulas rather than to reserve-policy bookkeeping.",
        ),
        sign_base.row(
            "updated_pack_exact_mixed_probe_response_completion_primary_followup_required",
            "pass" if updated_pack_exact_mixed_probe_response_completion_primary_followup_required else "reject",
            "updated-pack exact mixed probe-response completion primary followup required",
            sign_base.truth(updated_pack_exact_mixed_probe_response_completion_primary_followup_required),
            "The honest next derivation step is to complete the exact mixed probe-response kernel itself.",
        ),
        sign_base.row(
            "updated_pack_vacuum_subtraction_secondary_followup_required",
            "pass" if updated_pack_vacuum_subtraction_secondary_followup_required else "reject",
            "updated-pack vacuum subtraction secondary followup required",
            sign_base.truth(updated_pack_vacuum_subtraction_secondary_followup_required),
            "Vacuum subtraction survives only as a downstream secondary issue after the kernel object is fixed canonically.",
        ),
        sign_base.row(
            "updated_pack_mixed_probe_response_breakthrough_passed_now",
            "pass" if updated_pack_mixed_probe_response_breakthrough_passed_now else "reject",
            "updated-pack mixed probe-response breakthrough passed now",
            sign_base.truth(updated_pack_mixed_probe_response_breakthrough_passed_now),
            "The branch sharpens the fallback object but still does not close the missing action-level term.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "Blind-vector direct computation remains downstream of the unresolved probe kernel theorem stack.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains reserve-only because the blocker is still the mixed probe-response kernel completion.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_exact_mixed_probe_response_kernel_audit_selected": updated_pack_exact_mixed_probe_response_kernel_audit_selected,
        "retry_gate_computation_mode_selected": retry_gate_computation_mode_selected,
        "failure_matrix_non_surrogate_guard_preserved": failure_matrix_non_surrogate_guard_preserved,
        "updated_pack_external_probe_current_vertex_machine_readable_now": updated_pack_external_probe_current_vertex_machine_readable_now,
        "pure_derivation_hessian_surface_explicit": pure_derivation_hessian_surface_explicit,
        "pure_derivation_scattering_from_hessian_explicit": pure_derivation_scattering_from_hessian_explicit,
        "note_scattering_kernel_explicit": note_scattering_kernel_explicit,
        "updated_pack_mixed_probe_response_kernel_target_surface_explicit": updated_pack_mixed_probe_response_kernel_target_surface_explicit,
        "updated_pack_pure_probe_response_kernel_target_surface_explicit": updated_pack_pure_probe_response_kernel_target_surface_explicit,
        "updated_pack_mixed_probe_response_kernel_machine_readable_now": updated_pack_mixed_probe_response_kernel_machine_readable_now,
        "exact_mixed_probe_response_kernel_formula_available_now": exact_mixed_probe_response_kernel_formula_available_now,
        "exact_pure_probe_response_kernel_formula_available_now": exact_pure_probe_response_kernel_formula_available_now,
        "exact_vacuum_subtraction_rule_available_now": exact_vacuum_subtraction_rule_available_now,
        "updated_pack_mixed_probe_response_kernel_fully_localized_now": updated_pack_mixed_probe_response_kernel_fully_localized_now,
        "updated_pack_exact_mixed_probe_response_completion_primary_followup_required": updated_pack_exact_mixed_probe_response_completion_primary_followup_required,
        "updated_pack_vacuum_subtraction_secondary_followup_required": updated_pack_vacuum_subtraction_secondary_followup_required,
        "updated_pack_mixed_probe_response_breakthrough_passed_now": updated_pack_mixed_probe_response_breakthrough_passed_now,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_exact_mixed_probe_response_kernel_completion",
        "selected_secondary_pack_update_surface": "updated_pack_vacuum_subtraction_hold",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2785",
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
            "overall_status": "vector_qball_form_factor_updated_pack_mixed_probe_response_declared",
            "branch_completed": True,
            "breakthrough_passed_now": updated_pack_mixed_probe_response_breakthrough_passed_now,
            "physical_reject_required": False,
        },
        {
            "formulas": build_formulae(prior_formulas),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2783"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2779-.2782"),
                "current_problem_hit": sign_base.hit(
                    current_problem_text,
                    "exact mixed probe-response kernel audit",
                ),
                "current_status_hit": sign_base.hit(
                    current_status_text,
                    "exact mixed probe-response kernel audit",
                ),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2779-.2782"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2767-.2782"),
                "part5_hit": sign_base.hit(
                    part5_text,
                    "mixed probe-response kernel",
                ),
                "note_kernel_hit": sign_base.hit(note_text, "\\tilde{\\mathcal{K}}_{\\mu\\nu}"),
                "note_vacuum_subtraction_hit": sign_base.hit(note_text, "vacuum subtraction"),
            },
            "inference": {
                "why_mixed_response_now": (
                    "Once the one-point external current stays underived, the note's "
                    "Hessian-side scattering object must be re-read as a mixed/pure "
                    "probe-response kernel rather than as canonical one-photon data."
                ),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = sign_base.payload(
        "8.7.56.2786",
        STEP_NAME + " route sync",
        {
            "source_files": declaration_payload["inputs"]["source_files"],
            "declaration": {},
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_mixed_probe_response_route_synced",
            "branch_completed": True,
            "breakthrough_passed_now": updated_pack_mixed_probe_response_breakthrough_passed_now,
            "physical_reject_required": False,
        },
        {
            "formulas": build_formulae(prior_formulas),
            "notes": {
                "primary_transition": (
                    "The official next bottleneck is the exact mixed probe-response "
                    "kernel completion rather than same-level external-current restatement."
                ),
                "secondary_hold": (
                    "Vacuum subtraction remains secondary and downstream of the kernel "
                    "selection/completion step."
                ),
            },
        },
    )
    write_artifact("route_sync", route_payload)

    print(f"[done] {STEP_TAG} updated-pack exact mixed probe-response kernel audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
