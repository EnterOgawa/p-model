#!/usr/bin/env python3
"""Generate 8.7.56.2815-.2818 hybrid-reserve refresh audit artifacts."""

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
        "8.7.56.2811-2814",
        "updated_pack_pack_refresh_gate_hybrid_reserve_refresh",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2807-2810",
        "updated_pack_pack_refresh_sync_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
STEP_TAG = "8.7.56.2815-2818"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack hybrid-reserve "
    "refresh audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_hybrid_reserve_refresh_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "pack_refresh_sync_audited_hybrid_reserve_refresh_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "hybrid_reserve_refresh_audited_reserve_registry_secondary_gate"
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


# 関数: hybrid-reserve refresh audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the hybrid-reserve refresh audit."""
    return {
        "hybrid_reserve_order": (
            "pack-refresh sync -> hybrid reserve refresh -> reserve-registry refresh "
            "-> only then extra q-range reopen"
        ),
        "unresolved_stack": (
            "corrected probe split + exact mixed/pure kernel formula + exact "
            "vacuum-state theorem + exact subtraction rule + exact rank match"
        ),
        "reopen_rule": (
            "Reopen farther hybrid continuation only if exact residual-origin "
            "discrimination still requires extra q-range after reserve judgement."
        ),
    }


# 関数: `.2815-.2818` を実行する。

def main() -> None:
    """Execute the updated-pack hybrid-reserve refresh audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    selected = bool(
        prior_gate_summary["gate_b_updated_pack_hybrid_reserve_refresh_promoted_next"]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    pack_refresh_sync_surface_retained = bool(
        prior_audit_summary["updated_pack_pack_refresh_sync_machine_readable_now"]
    )
    unresolved_subtraction_state_retained = bool(
        prior_audit_summary["updated_pack_unresolved_subtraction_state_retained"]
    )
    corrected_probe_kernel_gap_retained = bool(
        prior_audit_summary["updated_pack_corrected_probe_kernel_gap_retained"]
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
        == "updated_pack_hybrid_reserve_gate_reserve_registry_refresh"
    )
    target_surface_explicit = bool(
        selected
        and retry_mode
        and non_surrogate_guard
        and pack_refresh_sync_surface_retained
        and unresolved_subtraction_state_retained
        and corrected_probe_kernel_gap_retained
        and farther_hybrid_hold_explicit
        and extra_q_range_reopen_condition_explicit
        and reserve_registry_followup_explicit
    )
    machine_readable_now = bool(target_surface_explicit)
    exact_pack_refresh_sync_verdict_available_now = False
    exact_hybrid_reserve_judgement_available_now = False
    exact_reserve_registry_refresh_available_now = False
    reserve_registry_refresh_primary_followup_required = bool(
        machine_readable_now and (not exact_hybrid_reserve_judgement_available_now)
    )
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])
    farther_hybrid = False

    rows = [
        sign_base.row(
            "updated_pack_hybrid_reserve_refresh_audit_selected",
            "pass" if selected else "reject",
            "updated-pack hybrid-reserve refresh audit selected",
            sign_base.truth(selected),
            "Once pack-refresh sync is explicit but unresolved, the honest next move is to reproject that unresolved stack onto one hybrid-reserve surface.",
        ),
        sign_base.row(
            "updated_pack_pack_refresh_sync_surface_retained",
            "pass" if pack_refresh_sync_surface_retained else "reject",
            "updated-pack pack-refresh sync surface retained",
            sign_base.truth(pack_refresh_sync_surface_retained),
            "The hybrid-reserve audit inherits the synchronized corrected probe/kernel/subtraction stack instead of reopening older surrogate families.",
        ),
        sign_base.row(
            "updated_pack_unresolved_subtraction_state_retained",
            "pass" if unresolved_subtraction_state_retained else "reject",
            "updated-pack unresolved subtraction state retained",
            sign_base.truth(unresolved_subtraction_state_retained),
            "The absent vacuum-state theorem, subtraction rule, and rank match remain explicit and are carried into reserve judgement unchanged.",
        ),
        sign_base.row(
            "updated_pack_corrected_probe_kernel_gap_retained",
            "pass" if corrected_probe_kernel_gap_retained else "reject",
            "updated-pack corrected probe/kernel gap retained",
            sign_base.truth(corrected_probe_kernel_gap_retained),
            "Reserve refresh remains downstream of the absent corrected split and literal mixed/pure kernel formulas.",
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
            "updated_pack_hybrid_reserve_registry_followup_explicit",
            "pass" if reserve_registry_followup_explicit else "reject",
            "updated-pack hybrid-reserve registry followup explicit",
            sign_base.truth(reserve_registry_followup_explicit),
            "The branch already knows that reserve bookkeeping must be refreshed in one dedicated registry lane after this audit.",
        ),
        sign_base.row(
            "updated_pack_hybrid_reserve_refresh_target_surface_explicit",
            "pass" if target_surface_explicit else "reject",
            "updated-pack hybrid-reserve refresh target surface explicit",
            sign_base.truth(target_surface_explicit),
            "The reserve-refresh target is explicit: unresolved corrected pack-refresh stack plus reserve-only hybrid evidence and a controlled reopen condition.",
        ),
        sign_base.row(
            "updated_pack_hybrid_reserve_refresh_machine_readable_now",
            "pass" if machine_readable_now else "reject",
            "updated-pack hybrid-reserve refresh machine-readable now",
            sign_base.truth(machine_readable_now),
            "Hybrid reserve policy now lives on one explicit machine-readable surface under the corrected pack-refresh ordering.",
        ),
        sign_base.row(
            "exact_pack_refresh_sync_verdict_available_now",
            "pass" if exact_pack_refresh_sync_verdict_available_now else "reject",
            "exact pack-refresh sync verdict available now",
            sign_base.truth(exact_pack_refresh_sync_verdict_available_now),
            "Reserve refresh does not itself derive the missing corrected pack-refresh verdict.",
        ),
        sign_base.row(
            "exact_hybrid_reserve_judgement_available_now",
            "pass" if exact_hybrid_reserve_judgement_available_now else "reject",
            "exact hybrid-reserve judgement available now",
            sign_base.truth(exact_hybrid_reserve_judgement_available_now),
            "The retained hybrid evidence still cannot be adjudicated canonically under the corrected theorem stack.",
        ),
        sign_base.row(
            "exact_reserve_registry_refresh_available_now",
            "pass" if exact_reserve_registry_refresh_available_now else "reject",
            "exact reserve-registry refresh available now",
            sign_base.truth(exact_reserve_registry_refresh_available_now),
            "This audit localizes the reserve state but does not yet refresh the registry that tracks it.",
        ),
        sign_base.row(
            "updated_pack_reserve_registry_refresh_primary_followup_required",
            "pass" if reserve_registry_refresh_primary_followup_required else "reject",
            "updated-pack reserve-registry refresh primary followup required",
            sign_base.truth(reserve_registry_refresh_primary_followup_required),
            "Once reserve policy is localized but unresolved, the next honest move is reserve-registry refresh rather than reopening farther q-range evidence.",
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
            "Extra q-range evidence remains reserve-only because the blocker is still theorem-side reserve judgement.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_audit_summary["retained_scalar_residual_rel"]),
        "updated_pack_hybrid_reserve_refresh_audit_selected": selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_pack_refresh_sync_surface_retained": pack_refresh_sync_surface_retained,
        "updated_pack_unresolved_subtraction_state_retained": unresolved_subtraction_state_retained,
        "updated_pack_corrected_probe_kernel_gap_retained": corrected_probe_kernel_gap_retained,
        "updated_pack_farther_hybrid_extra_q_range_hold_explicit": farther_hybrid_hold_explicit,
        "updated_pack_extra_q_range_reopen_condition_explicit": extra_q_range_reopen_condition_explicit,
        "updated_pack_hybrid_reserve_registry_followup_explicit": reserve_registry_followup_explicit,
        "updated_pack_hybrid_reserve_refresh_target_surface_explicit": target_surface_explicit,
        "updated_pack_hybrid_reserve_refresh_machine_readable_now": machine_readable_now,
        "exact_pack_refresh_sync_verdict_available_now": exact_pack_refresh_sync_verdict_available_now,
        "exact_hybrid_reserve_judgement_available_now": exact_hybrid_reserve_judgement_available_now,
        "exact_reserve_registry_refresh_available_now": exact_reserve_registry_refresh_available_now,
        "updated_pack_reserve_registry_refresh_primary_followup_required": reserve_registry_refresh_primary_followup_required,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid,
        "recommended_next_route_or_none": "8.7.56.2819",
        "selected_followup_route_or_none": "8.7.56.2823",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.2817",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.2819",
                "followup_route": "8.7.56.2823",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_hybrid_reserve_refresh_audit_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack hybrid-reserve refresh audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
