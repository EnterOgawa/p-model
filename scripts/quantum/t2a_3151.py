#!/usr/bin/env python3
"""Generate 8.7.56.3151-.3154 corrected hybrid-reserve return audit artifacts."""

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
        "8.7.56.3147-3150",
        "updated_pack_corrected_pack_refresh_gate_hybrid_reserve_return",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.3143-3146",
        "updated_pack_corrected_pack_refresh_return_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
STEP_TAG = "8.7.56.3151-3154"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack corrected "
    "hybrid-reserve return audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_corrected_hybrid_reserve_return_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_pack_refresh_return_audited_hybrid_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_hybrid_reserve_return_audited_reserve_registry_secondary_gate"
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


# 関数: corrected hybrid-reserve return で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the corrected hybrid-reserve return audit."""
    return {
        "hybrid_reserve_order": (
            "corrected pack-refresh return -> corrected hybrid reserve return -> "
            "corrected reserve registry return -> only then extra q-range reopen"
        ),
        "unresolved_stack": (
            "exact corrected probe split + exact corrected mixed/pure kernel formula + "
            "exact corrected vacuum-state theorem + exact corrected subtraction rule + "
            "exact corrected rank match"
        ),
        "reopen_rule": (
            "Reopen farther hybrid continuation only if exact residual-origin "
            "discrimination still requires extra q-range after corrected reserve judgement."
        ),
    }


# 関数: `.3151-.3154` を実行する。

def main() -> None:
    """Execute the updated-pack corrected hybrid-reserve return audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_corrected_hybrid_reserve_return_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    corrected_pack_refresh_return_surface_retained = bool(
        prior_audit_summary["updated_pack_corrected_pack_refresh_return_machine_readable_now"]
    )
    unresolved_corrected_vacuum_subtraction_state_retained = bool(
        prior_audit_summary["updated_pack_unresolved_corrected_vacuum_subtraction_state_retained"]
    )
    corrected_mixed_kernel_return_gap_retained = bool(
        prior_audit_summary["updated_pack_corrected_mixed_kernel_return_gap_retained"]
    )
    farther_hybrid_hold_explicit = bool(
        prior_gate_summary["selected_reserve_completion_lane"]
        == "farther_hybrid_extra_q_range_only"
    )
    extra_q_range_reopen_condition_explicit = bool(
        farther_hybrid_hold_explicit
        and (not prior_gate_summary["gate_c_farther_hybrid_continuation_reopen_required_now"])
    )
    reserve_registry_followup_explicit = bool(
        prior_gate_summary["selected_secondary_completion_lane"]
        == "updated_pack_corrected_hybrid_reserve_gate_reserve_registry_refresh"
    )
    target_surface_explicit = bool(
        selected
        and retry_mode
        and non_surrogate_guard
        and corrected_pack_refresh_return_surface_retained
        and unresolved_corrected_vacuum_subtraction_state_retained
        and corrected_mixed_kernel_return_gap_retained
        and farther_hybrid_hold_explicit
        and extra_q_range_reopen_condition_explicit
        and reserve_registry_followup_explicit
    )
    machine_readable_now = bool(target_surface_explicit)
    exact_corrected_pack_refresh_return_verdict_available_now = False
    exact_corrected_hybrid_reserve_return_judgement_available_now = False
    exact_corrected_reserve_registry_return_available_now = False
    corrected_reserve_registry_return_primary_followup_required = bool(
        machine_readable_now and (not exact_corrected_hybrid_reserve_return_judgement_available_now)
    )
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])
    farther_hybrid = False

    rows = [
        sign_base.row(
            "updated_pack_corrected_hybrid_reserve_return_audit_selected",
            "pass" if selected else "reject",
            "updated-pack corrected hybrid-reserve return audit selected",
            sign_base.truth(selected),
            "Once corrected pack-refresh return is explicit but unresolved, the honest next move is to reproject that corrected unresolved stack onto one hybrid-reserve return surface.",
        ),
        sign_base.row(
            "updated_pack_corrected_pack_refresh_return_surface_retained",
            "pass" if corrected_pack_refresh_return_surface_retained else "reject",
            "updated-pack corrected pack-refresh return surface retained",
            sign_base.truth(corrected_pack_refresh_return_surface_retained),
            "The corrected hybrid-reserve return inherits the synchronized corrected probe/kernel/subtraction stack instead of reopening older surrogate families.",
        ),
        sign_base.row(
            "updated_pack_unresolved_corrected_vacuum_subtraction_state_retained",
            "pass" if unresolved_corrected_vacuum_subtraction_state_retained else "reject",
            "updated-pack unresolved corrected vacuum-subtraction state retained",
            sign_base.truth(unresolved_corrected_vacuum_subtraction_state_retained),
            "The absent corrected vacuum-state theorem, corrected subtraction rule, and corrected rank match remain explicit and are carried into reserve judgement unchanged.",
        ),
        sign_base.row(
            "updated_pack_corrected_mixed_kernel_return_gap_retained",
            "pass" if corrected_mixed_kernel_return_gap_retained else "reject",
            "updated-pack corrected mixed-kernel return gap retained",
            sign_base.truth(corrected_mixed_kernel_return_gap_retained),
            "Reserve refresh remains downstream of the absent corrected mixed/pure kernel formulas.",
        ),
        sign_base.row(
            "updated_pack_farther_hybrid_extra_q_range_hold_explicit",
            "pass" if farther_hybrid_hold_explicit else "reject",
            "updated-pack farther hybrid extra q-range hold explicit",
            sign_base.truth(farther_hybrid_hold_explicit),
            "Farther hybrid continuation remains reserve-only and is tracked explicitly as an extra q-range hold item.",
        ),
        sign_base.row(
            "updated_pack_extra_q_range_reopen_condition_explicit",
            "pass" if extra_q_range_reopen_condition_explicit else "reject",
            "updated-pack extra q-range reopen condition explicit",
            sign_base.truth(extra_q_range_reopen_condition_explicit),
            "The branch states explicitly that extra q-range may reopen only after exact residual-origin discrimination still needs it.",
        ),
        sign_base.row(
            "updated_pack_hybrid_reserve_return_registry_followup_explicit",
            "pass" if reserve_registry_followup_explicit else "reject",
            "updated-pack hybrid-reserve return registry followup explicit",
            sign_base.truth(reserve_registry_followup_explicit),
            "The branch already knows that reserve bookkeeping must be refreshed in one dedicated registry lane after this audit.",
        ),
        sign_base.row(
            "updated_pack_corrected_hybrid_reserve_return_target_surface_explicit",
            "pass" if target_surface_explicit else "reject",
            "updated-pack corrected hybrid-reserve return target surface explicit",
            sign_base.truth(target_surface_explicit),
            "The corrected reserve-return target is explicit: unresolved corrected pack-refresh return stack plus reserve-only hybrid evidence and a controlled reopen condition.",
        ),
        sign_base.row(
            "updated_pack_corrected_hybrid_reserve_return_machine_readable_now",
            "pass" if machine_readable_now else "reject",
            "updated-pack corrected hybrid-reserve return machine-readable now",
            sign_base.truth(machine_readable_now),
            "The corrected hybrid-reserve return lane is now explicit as a machine-readable audit object.",
        ),
        sign_base.row(
            "exact_corrected_pack_refresh_return_verdict_available_now",
            "pass" if exact_corrected_pack_refresh_return_verdict_available_now else "reject",
            "exact corrected pack-refresh return verdict available now",
            sign_base.truth(exact_corrected_pack_refresh_return_verdict_available_now),
            "The corrected synchronized return stack still does not yield a canonical pack-refresh verdict.",
        ),
        sign_base.row(
            "exact_corrected_hybrid_reserve_return_judgement_available_now",
            "pass" if exact_corrected_hybrid_reserve_return_judgement_available_now else "reject",
            "exact corrected hybrid-reserve return judgement available now",
            sign_base.truth(exact_corrected_hybrid_reserve_return_judgement_available_now),
            "The branch still lacks a canonical corrected hybrid-reserve return judgement after projecting the unresolved stack onto the reserve surface.",
        ),
        sign_base.row(
            "exact_corrected_reserve_registry_return_available_now",
            "pass" if exact_corrected_reserve_registry_return_available_now else "reject",
            "exact corrected reserve-registry return available now",
            sign_base.truth(exact_corrected_reserve_registry_return_available_now),
            "Reserve-registry return closeout still depends on the missing corrected pack-refresh return and corrected reserve-return verdicts.",
        ),
        sign_base.row(
            "updated_pack_corrected_reserve_registry_return_primary_followup_required",
            "pass" if corrected_reserve_registry_return_primary_followup_required else "reject",
            "updated-pack corrected reserve-registry return primary followup required",
            sign_base.truth(corrected_reserve_registry_return_primary_followup_required),
            "Once the corrected reserve-return surface is explicit but unresolved, the honest next move is to refresh the reserve registry return rather than reopen farther continuation.",
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
            "Extra q-range evidence remains reserve-only because the blocker is still theorem-side corrected reserve-return judgement.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_corrected_hybrid_reserve_return_audit_selected": selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_corrected_pack_refresh_return_surface_retained": corrected_pack_refresh_return_surface_retained,
        "updated_pack_unresolved_corrected_vacuum_subtraction_state_retained": unresolved_corrected_vacuum_subtraction_state_retained,
        "updated_pack_corrected_mixed_kernel_return_gap_retained": corrected_mixed_kernel_return_gap_retained,
        "updated_pack_farther_hybrid_extra_q_range_hold_explicit": farther_hybrid_hold_explicit,
        "updated_pack_extra_q_range_reopen_condition_explicit": extra_q_range_reopen_condition_explicit,
        "updated_pack_hybrid_reserve_return_registry_followup_explicit": reserve_registry_followup_explicit,
        "updated_pack_corrected_hybrid_reserve_return_target_surface_explicit": target_surface_explicit,
        "updated_pack_corrected_hybrid_reserve_return_machine_readable_now": machine_readable_now,
        "exact_corrected_pack_refresh_return_verdict_available_now": exact_corrected_pack_refresh_return_verdict_available_now,
        "exact_corrected_hybrid_reserve_return_judgement_available_now": exact_corrected_hybrid_reserve_return_judgement_available_now,
        "exact_corrected_reserve_registry_return_available_now": exact_corrected_reserve_registry_return_available_now,
        "updated_pack_corrected_reserve_registry_return_primary_followup_required": corrected_reserve_registry_return_primary_followup_required,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid,
    }

    payload = sign_base.payload(
        "8.7.56.3153",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.3155",
                "followup_route": "8.7.56.3159",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_corrected_hybrid_reserve_return_audit_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack corrected hybrid-reserve return audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
