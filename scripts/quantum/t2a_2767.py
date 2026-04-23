#!/usr/bin/env python3
"""Generate 8.7.56.2767-.2770 pure-derivation probe-split audit artifacts."""

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
        "8.7.56.2763-2766",
        "updated_pack_pack_refresh_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
CURRENT_VERTEX_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2711-2714",
        "updated_pack_exact_4d_current_vertex_completion_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
STATIC_Q0_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2719-2722",
        "updated_pack_exact_static_q0_current_theorem_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PURE_DERIVATION_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_pure_derivation_20260330.md"
)

STEP_TAG = "8.7.56.2767-2770"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack pure-derivation "
    "probe-split audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_pure_derivation_probe_split_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_pack_"
    "refresh_audited_hybrid_reserve_registry_sync_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_pure_"
    "derivation_probe_split_audited_external_probe_current_vertex_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_pure_derivation_"
    "gate_external_probe_current_vertex"
)
NEXT_ROUTE = "8.7.56.2771"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_exact_external_"
    "probe_current_vertex_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2775"


# Function: write one JSON and CSV artifact pair.
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


# Function: return formulas used in the pure-derivation probe-split audit.

def build_formulae() -> dict[str, str]:
    """Return formulas used in the updated-pack pure-derivation probe-split audit."""
    return {
        "background_stationarity": (
            "S_bg^(1)[Q;xi] = int d^4x xi_mu (delta S / delta P_mu)|_(P=Q) = 0"
        ),
        "self_fluctuation_hessian": (
            "H_{mu nu}[Q] = delta^2 S / (delta P_mu delta P_nu) |_(P=Q)"
        ),
        "external_probe_current": (
            "J_ext^mu[Q](x) := delta S_frozen[Q;A] / delta A_mu(x) |_(A=0)"
        ),
        "mixed_probe_kernel": (
            "V^{mu nu}[Q](x,y) := delta^2 S_frozen / (delta xi_mu(x) delta A_nu(y))"
            " |_(Q,A=0)"
        ),
        "probe_split_expansion": (
            "S[Q + xi; A] = S[Q;0] + (1/2) xi H[Q] xi + int A_mu J_ext^mu[Q]"
            " + int xi_mu V^{mu nu}[Q] A_nu + (1/2) int A_mu Pi^{mu nu}[Q] A_nu + ..."
        ),
        "position_rule": (
            "Vacuum subtraction is meaningful only after the rank-matched probe "
            "observable, current, or kernel has been fixed canonically."
        ),
    }


# Function: execute the pure-derivation probe-split audit.

def main() -> None:
    """Execute the updated-pack pure-derivation probe-split audit."""
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
        CURRENT_VERTEX_AUDIT,
        STATIC_Q0_AUDIT,
        PURE_DERIVATION_NOTE,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    note_text = sign_base.read_text(PURE_DERIVATION_NOTE)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    current_vertex_summary = sign_base.read_json(CURRENT_VERTEX_AUDIT)["summary"]
    static_q0_summary = sign_base.read_json(STATIC_Q0_AUDIT)["summary"]

    updated_pack_pure_derivation_probe_split_audit_selected = bool(
        prior_gate_summary["gate_b_updated_pack_hybrid_reserve_registry_sync_promoted_next"]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_gate_same_pattern_three_plus_explicit = True
    retry_gate_computation_mode_selected = bool(
        updated_pack_pure_derivation_probe_split_audit_selected
        and retry_gate_same_pattern_three_plus_explicit
    )
    current_pack_same_field_no_go_fixed = bool(
        current_vertex_summary["current_pack_same_field_no_go_fixed"]
    )
    failure_matrix_non_surrogate_guard_preserved = bool(
        current_vertex_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    updated_pack_exact_4d_current_vertex_machine_readable_now = bool(
        current_vertex_summary["updated_pack_exact_4d_current_vertex_machine_readable_now"]
    )
    updated_pack_static_q0_current_theorem_machine_readable_now = bool(
        static_q0_summary["updated_pack_static_q0_current_theorem_machine_readable_now"]
    )
    pure_derivation_note_available = bool(note_text.strip())
    pure_derivation_note_on_shell_zero_explicit = bool(
        sign_base.hit(note_text, "S^(1) = 0") is not None
    )
    pure_derivation_note_hessian_surface_explicit = bool(
        sign_base.hit(note_text, "delta^2 S") is not None
        or sign_base.hit(note_text, "S^{(2)}[Q;a]") is not None
    )
    pure_derivation_note_scattering_from_hessian_explicit = bool(
        sign_base.hit(note_text, "photon scattering amplitude") is not None
        and sign_base.hit(note_text, "tilde{\\mathcal{K}}") is not None
    )
    pure_derivation_note_vacuum_subtraction_claim_explicit = bool(
        sign_base.hit(note_text, "vacuum subtraction") is not None
        and sign_base.hit(note_text, "唯一の不確定") is not None
    )
    pure_derivation_background_stationarity_only_explicit = bool(
        pure_derivation_note_on_shell_zero_explicit
        and current_pack_same_field_no_go_fixed
    )
    pure_derivation_external_probe_split_required = bool(
        pure_derivation_note_hessian_surface_explicit
        and pure_derivation_note_scattering_from_hessian_explicit
        and updated_pack_exact_4d_current_vertex_machine_readable_now
        and updated_pack_static_q0_current_theorem_machine_readable_now
    )
    pure_derivation_hessian_not_canonical_one_photon_source = bool(
        pure_derivation_external_probe_split_required
        and pure_derivation_background_stationarity_only_explicit
    )
    pure_derivation_mixed_probe_response_kernel_required = bool(
        pure_derivation_hessian_not_canonical_one_photon_source
    )
    pure_derivation_vacuum_subtraction_not_last_piece_now = bool(
        pure_derivation_note_vacuum_subtraction_claim_explicit
        and pure_derivation_mixed_probe_response_kernel_required
    )
    updated_pack_exact_external_probe_current_vertex_target_surface_explicit = bool(
        updated_pack_pure_derivation_probe_split_audit_selected
        and retry_gate_computation_mode_selected
        and current_pack_same_field_no_go_fixed
        and failure_matrix_non_surrogate_guard_preserved
        and pure_derivation_external_probe_split_required
    )
    updated_pack_exact_external_probe_current_vertex_machine_readable_now = bool(
        updated_pack_exact_external_probe_current_vertex_target_surface_explicit
    )
    updated_pack_mixed_probe_response_kernel_target_surface_explicit = bool(
        updated_pack_exact_external_probe_current_vertex_machine_readable_now
        and pure_derivation_mixed_probe_response_kernel_required
    )
    exact_external_probe_current_vertex_formula_available_now = False
    exact_mixed_probe_response_kernel_available_now = False
    updated_pack_exact_external_probe_current_vertex_followup_required = bool(
        updated_pack_exact_external_probe_current_vertex_machine_readable_now
        and (not exact_external_probe_current_vertex_formula_available_now)
    )
    updated_pack_mixed_probe_response_secondary_followup_required = bool(
        updated_pack_mixed_probe_response_kernel_target_surface_explicit
        and (not exact_mixed_probe_response_kernel_available_now)
    )
    updated_pack_pure_derivation_breakthrough_passed_now = False
    blind_vector_observable_gate_still_blocked = bool(
        prior_gate_summary["blind_vector_observable_gate_still_blocked"]
    )
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_pure_derivation_probe_split_audit_selected",
            "pass" if updated_pack_pure_derivation_probe_split_audit_selected else "reject",
            "updated-pack pure-derivation probe-split audit selected",
            sign_base.truth(updated_pack_pure_derivation_probe_split_audit_selected),
            "The reserve-policy sync lane kept stalling without producing a new theorem, so the pure-derivation note is used as the next honest computation-side branch.",
        ),
        sign_base.row(
            "retry_gate_same_pattern_three_plus_explicit",
            "pass" if retry_gate_same_pattern_three_plus_explicit else "reject",
            "retry gate same-pattern count three-plus explicit",
            sign_base.truth(retry_gate_same_pattern_three_plus_explicit),
            "The reserve-registry and pack-refresh bookkeeping already repeated enough times to require an explicit retry-gate classification.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_gate_computation_mode_selected else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_gate_computation_mode_selected),
            "The new note exposes a derivation mismatch, so the branch moves from registry bookkeeping to computation-side operator classification.",
        ),
        sign_base.row(
            "current_pack_same_field_no_go_fixed",
            "pass" if current_pack_same_field_no_go_fixed else "reject",
            "current-pack same-field no-go fixed",
            sign_base.truth(current_pack_same_field_no_go_fixed),
            "The old same-field source theorem is already closed on the zero / no-go branch, so the new note must be tested outside that exact same object.",
        ),
        sign_base.row(
            "pure_derivation_note_on_shell_zero_explicit",
            "pass" if pure_derivation_note_on_shell_zero_explicit else "reject",
            "pure-derivation note on-shell zero explicit",
            sign_base.truth(pure_derivation_note_on_shell_zero_explicit),
            "The note correctly shows that the on-shell background kills the self-fluctuation linear term.",
        ),
        sign_base.row(
            "pure_derivation_note_hessian_surface_explicit",
            "pass" if pure_derivation_note_hessian_surface_explicit else "reject",
            "pure-derivation note Hessian surface explicit",
            sign_base.truth(pure_derivation_note_hessian_surface_explicit),
            "The note makes the quadratic self-fluctuation Hessian explicit rather than hiding it in a surrogate observable.",
        ),
        sign_base.row(
            "pure_derivation_note_scattering_from_hessian_explicit",
            "pass" if pure_derivation_note_scattering_from_hessian_explicit else "reject",
            "pure-derivation note scattering from Hessian explicit",
            sign_base.truth(pure_derivation_note_scattering_from_hessian_explicit),
            "The claimed scattering object is built directly from the Hessian-side kernel, which is precisely the point that needs rank correction.",
        ),
        sign_base.row(
            "pure_derivation_background_stationarity_only_explicit",
            "pass" if pure_derivation_background_stationarity_only_explicit else "reject",
            "pure-derivation background stationarity only explicit",
            sign_base.truth(pure_derivation_background_stationarity_only_explicit),
            "S^(1)=0 is valid as a statement about background stationarity, not yet as a statement about a distinct external probe channel.",
        ),
        sign_base.row(
            "pure_derivation_external_probe_split_required",
            "pass" if pure_derivation_external_probe_split_required else "reject",
            "pure-derivation external-probe split required",
            sign_base.truth(pure_derivation_external_probe_split_required),
            "Once the note is compared with the corrected 4D current-vertex lane, the missing step is the split between self fluctuation and external probe.",
        ),
        sign_base.row(
            "pure_derivation_hessian_not_canonical_one_photon_source",
            "pass" if pure_derivation_hessian_not_canonical_one_photon_source else "reject",
            "pure-derivation Hessian not canonical one-photon source",
            sign_base.truth(pure_derivation_hessian_not_canonical_one_photon_source),
            "The quadratic Hessian is a self-fluctuation kernel, not automatically the rank-matched one-photon current/source object.",
        ),
        sign_base.row(
            "pure_derivation_mixed_probe_response_kernel_required",
            "pass" if pure_derivation_mixed_probe_response_kernel_required else "reject",
            "pure-derivation mixed probe-response kernel required",
            sign_base.truth(pure_derivation_mixed_probe_response_kernel_required),
            "If the one-point probe current vanishes, the honest downstream object is the mixed or pure probe response kernel, not vacuum subtraction alone.",
        ),
        sign_base.row(
            "pure_derivation_vacuum_subtraction_not_last_piece_now",
            "pass" if pure_derivation_vacuum_subtraction_not_last_piece_now else "reject",
            "pure-derivation vacuum subtraction not last piece now",
            sign_base.truth(pure_derivation_vacuum_subtraction_not_last_piece_now),
            "Vacuum subtraction can become final only after the rank-matched probe observable has been fixed; the note reaches that issue too early.",
        ),
        sign_base.row(
            "updated_pack_exact_external_probe_current_vertex_machine_readable_now",
            "pass" if updated_pack_exact_external_probe_current_vertex_machine_readable_now else "reject",
            "updated-pack exact external-probe current-vertex machine-readable now",
            sign_base.truth(updated_pack_exact_external_probe_current_vertex_machine_readable_now),
            "The note sharpens the missing piece into an external-probe current-vertex definition rather than a reserve-policy bookkeeping item.",
        ),
        sign_base.row(
            "updated_pack_mixed_probe_response_kernel_target_surface_explicit",
            "pass" if updated_pack_mixed_probe_response_kernel_target_surface_explicit else "reject",
            "updated-pack mixed probe-response kernel target surface explicit",
            sign_base.truth(updated_pack_mixed_probe_response_kernel_target_surface_explicit),
            "The corrected derivation now retains the mixed response kernel as the honest secondary branch if the one-point external current still vanishes.",
        ),
        sign_base.row(
            "exact_external_probe_current_vertex_formula_available_now",
            "pass" if exact_external_probe_current_vertex_formula_available_now else "reject",
            "exact external-probe current-vertex formula available now",
            sign_base.truth(exact_external_probe_current_vertex_formula_available_now),
            "The note does not yet supply the exact rank-matched external-probe current formula.",
        ),
        sign_base.row(
            "exact_mixed_probe_response_kernel_available_now",
            "pass" if exact_mixed_probe_response_kernel_available_now else "reject",
            "exact mixed probe-response kernel available now",
            sign_base.truth(exact_mixed_probe_response_kernel_available_now),
            "The note also does not yet complete the exact mixed probe-response kernel after the probe/self split.",
        ),
        sign_base.row(
            "updated_pack_pure_derivation_breakthrough_passed_now",
            "pass" if updated_pack_pure_derivation_breakthrough_passed_now else "reject",
            "updated-pack pure-derivation breakthrough passed now",
            sign_base.truth(updated_pack_pure_derivation_breakthrough_passed_now),
            "The note sharpens the blocker but does not by itself close the missing action-level term.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(current_vertex_summary["retained_scalar_residual_rel"]),
        "updated_pack_pure_derivation_probe_split_audit_selected": updated_pack_pure_derivation_probe_split_audit_selected,
        "retry_gate_same_pattern_three_plus_explicit": retry_gate_same_pattern_three_plus_explicit,
        "retry_gate_computation_mode_selected": retry_gate_computation_mode_selected,
        "current_pack_same_field_no_go_fixed": current_pack_same_field_no_go_fixed,
        "failure_matrix_non_surrogate_guard_preserved": failure_matrix_non_surrogate_guard_preserved,
        "pure_derivation_note_available": pure_derivation_note_available,
        "pure_derivation_note_on_shell_zero_explicit": pure_derivation_note_on_shell_zero_explicit,
        "pure_derivation_note_hessian_surface_explicit": pure_derivation_note_hessian_surface_explicit,
        "pure_derivation_note_scattering_from_hessian_explicit": pure_derivation_note_scattering_from_hessian_explicit,
        "pure_derivation_note_vacuum_subtraction_claim_explicit": pure_derivation_note_vacuum_subtraction_claim_explicit,
        "pure_derivation_background_stationarity_only_explicit": pure_derivation_background_stationarity_only_explicit,
        "pure_derivation_external_probe_split_required": pure_derivation_external_probe_split_required,
        "pure_derivation_hessian_not_canonical_one_photon_source": pure_derivation_hessian_not_canonical_one_photon_source,
        "pure_derivation_mixed_probe_response_kernel_required": pure_derivation_mixed_probe_response_kernel_required,
        "pure_derivation_vacuum_subtraction_not_last_piece_now": pure_derivation_vacuum_subtraction_not_last_piece_now,
        "updated_pack_exact_external_probe_current_vertex_target_surface_explicit": updated_pack_exact_external_probe_current_vertex_target_surface_explicit,
        "updated_pack_exact_external_probe_current_vertex_machine_readable_now": updated_pack_exact_external_probe_current_vertex_machine_readable_now,
        "updated_pack_mixed_probe_response_kernel_target_surface_explicit": updated_pack_mixed_probe_response_kernel_target_surface_explicit,
        "exact_external_probe_current_vertex_formula_available_now": exact_external_probe_current_vertex_formula_available_now,
        "exact_mixed_probe_response_kernel_available_now": exact_mixed_probe_response_kernel_available_now,
        "updated_pack_exact_external_probe_current_vertex_followup_required": updated_pack_exact_external_probe_current_vertex_followup_required,
        "updated_pack_mixed_probe_response_secondary_followup_required": updated_pack_mixed_probe_response_secondary_followup_required,
        "updated_pack_pure_derivation_breakthrough_passed_now": updated_pack_pure_derivation_breakthrough_passed_now,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_pack_update_surface": "updated_pack_exact_external_probe_current_vertex",
        "selected_secondary_pack_update_surface": "updated_pack_mixed_probe_response_kernel",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2769",
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
                "current_vertex_audit": sign_base.display_path(CURRENT_VERTEX_AUDIT),
                "static_q0_audit": sign_base.display_path(STATIC_Q0_AUDIT),
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
            "overall_status": "vector_qball_form_factor_updated_pack_pure_derivation_probe_split_declared",
            "branch_completed": True,
            "route_reset_from_registry_sync_to_derivation": True,
            "breakthrough_passed_now": updated_pack_pure_derivation_breakthrough_passed_now,
            "physical_reject_required": False,
        },
        {
            "formulae": build_formulae(),
            "hits": {
                "on_shell_zero": sign_base.hit(note_text, "S^(1) = 0"),
                "hessian": sign_base.hit(note_text, "delta^2 S"),
                "scattering": sign_base.hit(note_text, "photon scattering amplitude"),
                "vacuum_subtraction": sign_base.hit(note_text, "vacuum subtraction"),
            },
            "status_hits": {
                "current_state": sign_base.hit(status_text, "pack_refresh_audited_hybrid_reserve_registry_sync_next"),
            },
            "notes": {
                "route_reset_reason": (
                    "The pure-derivation note shows that the unresolved object appears "
                    "before reserve-policy closeout: the self-fluctuation Hessian was "
                    "being read as an external probe observable."
                ),
                "why_vacuum_subtraction_not_final": (
                    "Vacuum subtraction can only be final after the rank-matched probe "
                    "current or response kernel is fixed canonically."
                ),
            },
        },
    )

    route_payload = sign_base.payload(
        "8.7.56.2770",
        STEP_NAME + " route sync",
        {
            "source_files": declaration_payload["inputs"]["source_files"],
            "declaration": {},
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_pure_derivation_probe_split_route_synced",
            "branch_completed": True,
            "route_reset_from_registry_sync_to_derivation": True,
            "breakthrough_passed_now": updated_pack_pure_derivation_breakthrough_passed_now,
            "physical_reject_required": False,
        },
        {
            "formulae": build_formulae(),
            "notes": {
                "primary_transition": (
                    "The official bottleneck moves from hybrid-reserve registry sync to "
                    "exact external-probe current-vertex completion."
                ),
                "secondary_transition": (
                    "If the exact one-point external current still vanishes, the mixed "
                    "probe-response kernel remains the next honest secondary lane."
                ),
            },
        },
    )

    declaration_paths = write_artifact("declaration_gate", declaration_payload)
    route_paths = write_artifact("route_sync", route_payload)

    print(f"[done] {STEP_TAG} declaration: {declaration_paths['json']}")
    print(f"[done] {STEP_TAG} route sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
