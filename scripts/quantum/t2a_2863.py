#!/usr/bin/env python3
"""Generate 8.7.56.2863-.2866 corrected pack-refresh sync audit artifacts."""

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
        "8.7.56.2859-2862",
        "updated_pack_corrected_vacuum_subtraction_gate_pack_refresh_sync",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2855-2858",
        "updated_pack_corrected_vacuum_subtraction_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
STEP_TAG = "8.7.56.2863-2866"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack corrected "
    "pack-refresh sync audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_corrected_pack_refresh_sync_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_vacuum_subtraction_audited_pack_refresh_sync_primary_"
    "hybrid_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_pack_refresh_sync_audited_hybrid_reserve_refresh_secondary_"
    "gate"
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


# 関数: corrected pack-refresh sync audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the corrected pack-refresh sync audit."""
    return {
        "pack_refresh_sync_role": (
            "corrected pack-refresh sync := sync the unresolved corrected "
            "probe split / corrected mixed kernel / corrected subtraction "
            "stack back into one updated-pack verdict surface"
        ),
        "unresolved_stack": (
            "exact corrected probe split + exact corrected mixed/pure kernel "
            "formula + exact corrected vacuum-state theorem + exact corrected "
            "subtraction rule + exact corrected rank match"
        ),
        "reserve_order": (
            "corrected pack-refresh sync -> hybrid reserve refresh -> reserve "
            "registry refresh -> only then extra q-range reopen"
        ),
    }


# 関数: `.2863-.2866` を実行する。

def main() -> None:
    """Execute the updated-pack corrected pack-refresh sync audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    selected = bool(
        prior_gate_summary["gate_b_updated_pack_pack_refresh_sync_promoted_next"]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    corrected_vacuum_subtraction_surface_retained = bool(
        prior_audit_summary["updated_pack_corrected_vacuum_subtraction_machine_readable_now"]
    )
    unresolved_corrected_subtraction_state_retained = bool(
        (not prior_audit_summary["exact_corrected_vacuum_state_definition_available_now"])
        and (not prior_audit_summary["exact_corrected_vacuum_subtraction_rule_available_now"])
        and (
            not prior_audit_summary[
                "exact_corrected_subtracted_observable_rank_match_available_now"
            ]
        )
    )
    corrected_mixed_kernel_gap_retained = bool(
        (not prior_audit_summary["exact_corrected_vacuum_subtraction_rule_available_now"])
        and (not prior_audit_summary["exact_corrected_subtracted_observable_rank_match_available_now"])
    )
    pack_refresh_sync_target_surface_explicit = bool(
        selected
        and retry_mode
        and non_surrogate_guard
        and corrected_vacuum_subtraction_surface_retained
        and unresolved_corrected_subtraction_state_retained
        and corrected_mixed_kernel_gap_retained
    )
    pack_refresh_sync_machine_readable_now = bool(pack_refresh_sync_target_surface_explicit)
    exact_pack_refresh_sync_verdict_available_now = False
    exact_hybrid_reserve_refresh_available_now = False
    exact_reserve_registry_refresh_available_now = False
    hybrid_reserve_refresh_primary_followup_required = bool(
        pack_refresh_sync_machine_readable_now
        and (not exact_pack_refresh_sync_verdict_available_now)
    )
    breakthrough = False
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])
    farther_hybrid = False

    rows = [
        sign_base.row(
            "updated_pack_corrected_pack_refresh_sync_audit_selected",
            "pass" if selected else "reject",
            "updated-pack corrected pack-refresh sync audit selected",
            sign_base.truth(selected),
            "Once corrected subtraction is localized but unresolved, the honest next move is to sync that corrected unresolved state back into one updated-pack verdict surface.",
        ),
        sign_base.row(
            "updated_pack_corrected_vacuum_subtraction_surface_retained",
            "pass" if corrected_vacuum_subtraction_surface_retained else "reject",
            "updated-pack corrected vacuum-subtraction surface retained",
            sign_base.truth(corrected_vacuum_subtraction_surface_retained),
            "The pack-refresh sync audit inherits the corrected subtraction surface instead of reopening the old surrogate families.",
        ),
        sign_base.row(
            "updated_pack_unresolved_corrected_subtraction_state_retained",
            "pass" if unresolved_corrected_subtraction_state_retained else "reject",
            "updated-pack unresolved corrected subtraction state retained",
            sign_base.truth(unresolved_corrected_subtraction_state_retained),
            "The absent corrected vacuum-state theorem, corrected subtraction rule, and corrected rank-matched observable remain explicit while entering the sync lane.",
        ),
        sign_base.row(
            "updated_pack_corrected_mixed_kernel_gap_retained",
            "pass" if corrected_mixed_kernel_gap_retained else "reject",
            "updated-pack corrected mixed-kernel gap retained",
            sign_base.truth(corrected_mixed_kernel_gap_retained),
            "Corrected pack-refresh sync still sits downstream of the absent corrected mixed/pure kernel formulas and the unresolved corrected subtraction theorem.",
        ),
        sign_base.row(
            "updated_pack_pack_refresh_sync_target_surface_explicit",
            "pass" if pack_refresh_sync_target_surface_explicit else "reject",
            "updated-pack pack-refresh sync target surface explicit",
            sign_base.truth(pack_refresh_sync_target_surface_explicit),
            "The unresolved corrected subtraction state now sits on one explicit corrected pack-refresh verdict surface.",
        ),
        sign_base.row(
            "updated_pack_pack_refresh_sync_machine_readable_now",
            "pass" if pack_refresh_sync_machine_readable_now else "reject",
            "updated-pack pack-refresh sync machine-readable now",
            sign_base.truth(pack_refresh_sync_machine_readable_now),
            "Canonical corrected pack-refresh synchronization is now explicit as a machine-readable audit object.",
        ),
        sign_base.row(
            "exact_pack_refresh_sync_verdict_available_now",
            "pass" if exact_pack_refresh_sync_verdict_available_now else "reject",
            "exact pack-refresh sync verdict available now",
            sign_base.truth(exact_pack_refresh_sync_verdict_available_now),
            "The corrected unresolved state is now synchronized, but the corrected split, corrected kernel formulas, and corrected subtraction theorem still do not yield a canonical verdict.",
        ),
        sign_base.row(
            "updated_pack_hybrid_reserve_refresh_primary_followup_required",
            "pass" if hybrid_reserve_refresh_primary_followup_required else "reject",
            "updated-pack hybrid reserve refresh primary followup required",
            sign_base.truth(hybrid_reserve_refresh_primary_followup_required),
            "Once the corrected pack-refresh sync surface is explicit but unresolved, the honest next move is to refresh the hybrid reserve lane rather than reopen farther q-range continuation.",
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
            "Extra q-range evidence remains reserve-only because the blocker is still theorem-side corrected pack-refresh closure.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_audit_summary["retained_scalar_residual_rel"]),
        "updated_pack_corrected_pack_refresh_sync_audit_selected": selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_corrected_vacuum_subtraction_surface_retained": corrected_vacuum_subtraction_surface_retained,
        "updated_pack_unresolved_corrected_subtraction_state_retained": unresolved_corrected_subtraction_state_retained,
        "updated_pack_corrected_mixed_kernel_gap_retained": corrected_mixed_kernel_gap_retained,
        "updated_pack_pack_refresh_sync_target_surface_explicit": pack_refresh_sync_target_surface_explicit,
        "updated_pack_pack_refresh_sync_machine_readable_now": pack_refresh_sync_machine_readable_now,
        "exact_pack_refresh_sync_verdict_available_now": exact_pack_refresh_sync_verdict_available_now,
        "exact_hybrid_reserve_refresh_available_now": exact_hybrid_reserve_refresh_available_now,
        "exact_reserve_registry_refresh_available_now": exact_reserve_registry_refresh_available_now,
        "updated_pack_hybrid_reserve_refresh_primary_followup_required": hybrid_reserve_refresh_primary_followup_required,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid,
        "selected_primary_completion_lane": "updated_pack_hybrid_reserve_refresh_audit",
        "selected_secondary_completion_lane": "updated_pack_hybrid_reserve_gate_reserve_registry_refresh",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_hybrid_reserve_refresh_audit",
        "recommended_next_route_or_none": "8.7.56.2871",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_hybrid_reserve_gate_reserve_registry_refresh",
        "selected_followup_route_or_none": "8.7.56.2875",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.2865",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.2867",
                "followup_route": "8.7.56.2871",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_corrected_pack_refresh_sync_declared",
            "branch_completed": True,
            "breakthrough_passed_now": breakthrough,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack corrected pack-refresh sync audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
