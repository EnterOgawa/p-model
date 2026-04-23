#!/usr/bin/env python3
"""Generate 8.7.56.2799-.2802 vacuum-subtraction refresh audit artifacts."""

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
        "8.7.56.2795-2798",
        "updated_pack_mixed_probe_response_vacuum_subtraction_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2791-2794",
        "updated_pack_exact_mixed_probe_response_kernel_completion_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
LEGACY_V2_SUBTRACTION = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.1599-1602",
        "v2_sub_exact_treat",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
LEGACY_RESET = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.1603-1606",
        "eff_metric_mainline_reset",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PURE_DERIVATION_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_pure_derivation_20260330.md"
)
STEP_TAG = "8.7.56.2799-2802"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack vacuum-"
    "subtraction refresh audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_vacuum_subtraction_refresh_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "exact_kernel_completion_audited_vacuum_subtraction_primary_hybrid_"
    "reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "vacuum_subtraction_audited_pack_refresh_sync_gate"
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


# 関数: vacuum-subtraction refresh で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the vacuum-subtraction refresh audit."""
    return {
        "kernel_read": (
            "\\tilde{\\mathcal{K}}_{\\mu\\nu}(\\mathbf{k},\\mathbf{k}') = "
            "\\int d^3r\\,e^{-i\\mathbf{k}\\cdot\\mathbf{r}}"
            "\\left[2\\lambda(\\Phi_Q-v^2)\\eta_{\\mu\\nu}+4\\lambda q_\\mu q_\\nu\\right]"
            "e^{i\\mathbf{k}'\\cdot\\mathbf{r}}"
        ),
        "vacuum_subtraction": "\\Delta\\mathcal{M}=\\mathcal{M}[Q]-\\mathcal{M}[\\mathrm{vacuum}]",
        "note_vacuum_state": "\\Phi_{\\rm vac}=-v^2",
        "legacy_casea_result": "caseA: eta_{\\mu\\nu} -> reject, caseB: g_{\\mu\\nu}(P) -> pass",
    }


# 関数: `.2799-.2802` を実行する。

def main() -> None:
    """Execute the updated-pack vacuum-subtraction refresh audit."""
    for path in (
        PRIOR_GATE,
        PRIOR_AUDIT,
        LEGACY_V2_SUBTRACTION,
        LEGACY_RESET,
        PURE_DERIVATION_NOTE,
    ):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    legacy_v2_summary = sign_base.read_json(LEGACY_V2_SUBTRACTION)["summary"]
    legacy_reset_summary = sign_base.read_json(LEGACY_RESET)["summary"]
    note_text = sign_base.read_text(PURE_DERIVATION_NOTE)

    selected = bool(
        prior_gate_summary["gate_b_updated_pack_vacuum_subtraction_refresh_promoted_next"]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    note_vacuum_subtraction_explicit = bool(
        sign_base.hit(note_text, "vacuum subtraction") is not None
        and sign_base.hit(note_text, "\\Delta\\mathcal{M}") is not None
    )
    note_vacuum_state_surface_explicit = bool(
        sign_base.hit(note_text, "\\Phi_{\\rm vac}") is not None
        and sign_base.hit(note_text, "-v^2") is not None
    )
    note_box_divergence_explicit = bool(sign_base.hit(note_text, "V_{\\rm box}") is not None)
    note_divergent_vacuum_amplitude_explicit = bool(
        sign_base.hit(note_text, "\\mathcal{M}[\\text{vacuum}]") is not None
        and sign_base.hit(note_text, "divergent") is not None
    )
    note_subtraction_surface_explicit = bool(
        note_vacuum_subtraction_explicit
        and note_vacuum_state_surface_explicit
        and note_box_divergence_explicit
        and note_divergent_vacuum_amplitude_explicit
    )
    legacy_casea_worsen_retained = bool(
        legacy_v2_summary["worsen_selected"]
        and legacy_reset_summary["minkowski_worsen_retained_as_casea_result"]
    )
    legacy_casea_noncanonical_retained = bool(
        legacy_reset_summary["current_quadratic_lane_uses_eta"]
        and legacy_reset_summary["caseb_effective_metric_promoted_to_mainline"]
    )
    subtraction_surface = bool(
        selected
        and retry_mode
        and non_surrogate_guard
        and prior_audit_summary["updated_pack_exact_mixed_probe_response_completion_fully_localized_now"]
        and note_subtraction_surface_explicit
        and legacy_casea_worsen_retained
        and legacy_casea_noncanonical_retained
    )
    corrected_probe_split_symbol_available_now = bool(
        prior_audit_summary["corrected_probe_split_symbol_available_now"]
    )
    exact_mixed_probe_response_kernel_formula_available_now = bool(
        prior_audit_summary["exact_mixed_probe_response_kernel_formula_available_now"]
    )
    exact_vacuum_state_definition_available_now = False
    exact_vacuum_subtraction_rule_available_now = False
    exact_subtracted_observable_rank_match_available_now = False
    fully_localized = bool(
        subtraction_surface
        and (not exact_vacuum_state_definition_available_now)
        and (not exact_vacuum_subtraction_rule_available_now)
    )
    pack_refresh_sync_followup_required = bool(
        fully_localized and (not exact_subtracted_observable_rank_match_available_now)
    )
    breakthrough = False
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])
    farther_hybrid = False

    rows = [
        sign_base.row(
            "updated_pack_vacuum_subtraction_refresh_audit_selected",
            "pass" if selected else "reject",
            "updated-pack vacuum-subtraction refresh audit selected",
            sign_base.truth(selected),
            "The mixed probe-response gate already promoted vacuum-subtraction refresh as the next honest derivation lane.",
        ),
        sign_base.row(
            "updated_pack_vacuum_subtraction_machine_readable_now",
            "pass" if subtraction_surface else "reject",
            "updated-pack vacuum subtraction machine-readable now",
            sign_base.truth(subtraction_surface),
            "The note now exposes subtraction, vacuum-state, and divergence surfaces explicitly, while the old Minkowski worsen result is retained as a noncanonical caseA reference.",
        ),
        sign_base.row(
            "exact_vacuum_state_definition_available_now",
            "pass" if exact_vacuum_state_definition_available_now else "reject",
            "exact vacuum-state definition available now",
            sign_base.truth(exact_vacuum_state_definition_available_now),
            "The note proposes Phi_vac = -v^2, but the corrected probe/current lane still lacks a canonical frozen-action vacuum-state theorem.",
        ),
        sign_base.row(
            "exact_vacuum_subtraction_rule_available_now",
            "pass" if exact_vacuum_subtraction_rule_available_now else "reject",
            "exact vacuum-subtraction rule available now",
            sign_base.truth(exact_vacuum_subtraction_rule_available_now),
            "Without the corrected split, literal mixed/pure kernel formula, and exact vacuum-state theorem, subtraction cannot honestly close here.",
        ),
        sign_base.row(
            "exact_subtracted_observable_rank_match_available_now",
            "pass" if exact_subtracted_observable_rank_match_available_now else "reject",
            "exact subtracted observable rank match available now",
            sign_base.truth(exact_subtracted_observable_rank_match_available_now),
            "The subtraction target still sits on a rank-mismatched amplitude skeleton rather than on a closed corrected probe-response observable.",
        ),
        sign_base.row(
            "legacy_casea_v2_subtraction_worsen_retained",
            "pass" if legacy_casea_worsen_retained else "reject",
            "legacy caseA v^2 subtraction worsen retained",
            sign_base.truth(legacy_casea_worsen_retained),
            "The old Minkowski-contracted v^2 subtraction already worsened the retained scalar candidate and remains only as a noncanonical warning surface.",
        ),
        sign_base.row(
            "updated_pack_pack_refresh_sync_followup_required",
            "pass" if pack_refresh_sync_followup_required else "reject",
            "updated-pack pack-refresh sync followup required",
            sign_base.truth(pack_refresh_sync_followup_required),
            "Once subtraction is localized but still unresolved, the honest next move is to sync that unresolved subtraction state into the pack-refresh lane.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation remains downstream of the unresolved corrected probe/kernel/subtraction theorem stack.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid),
            "Extra q-range evidence remains reserve-only because the blocker is still theorem-side subtraction closure.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_vacuum_subtraction_refresh_audit_selected": selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "pure_derivation_vacuum_subtraction_surface_explicit": note_vacuum_subtraction_explicit,
        "pure_derivation_vacuum_state_surface_explicit": note_vacuum_state_surface_explicit,
        "pure_derivation_box_divergence_surface_explicit": note_box_divergence_explicit,
        "pure_derivation_divergent_vacuum_amplitude_explicit": note_divergent_vacuum_amplitude_explicit,
        "legacy_casea_v2_subtraction_worsen_retained": legacy_casea_worsen_retained,
        "legacy_casea_v2_subtraction_noncanonical_for_probe_lane": legacy_casea_noncanonical_retained,
        "updated_pack_vacuum_subtraction_target_surface_explicit": subtraction_surface,
        "updated_pack_vacuum_subtraction_machine_readable_now": subtraction_surface,
        "corrected_probe_split_symbol_available_now": corrected_probe_split_symbol_available_now,
        "exact_mixed_probe_response_kernel_formula_available_now": exact_mixed_probe_response_kernel_formula_available_now,
        "exact_vacuum_state_definition_available_now": exact_vacuum_state_definition_available_now,
        "exact_vacuum_subtraction_rule_available_now": exact_vacuum_subtraction_rule_available_now,
        "exact_subtracted_observable_rank_match_available_now": exact_subtracted_observable_rank_match_available_now,
        "updated_pack_vacuum_subtraction_fully_localized_now": fully_localized,
        "updated_pack_pack_refresh_sync_followup_required": pack_refresh_sync_followup_required,
        "updated_pack_vacuum_subtraction_breakthrough_passed_now": breakthrough,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid,
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_vacuum_subtraction_gate_pack_refresh_sync",
        "recommended_next_route_or_none": "8.7.56.2803",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_pack_refresh_sync_audit",
        "selected_followup_route_or_none": "8.7.56.2807",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.2801",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "legacy_v2_subtraction": sign_base.display_path(LEGACY_V2_SUBTRACTION),
                "legacy_reset": sign_base.display_path(LEGACY_RESET),
                "pure_derivation_note": sign_base.display_path(PURE_DERIVATION_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.2803",
                "followup_route": "8.7.56.2807",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_vacuum_subtraction_refresh_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack vacuum-subtraction refresh audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
