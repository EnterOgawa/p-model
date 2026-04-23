#!/usr/bin/env python3
"""Generate 8.7.56.2895-.2898 corrected probe-split return audit artifacts."""

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
        "8.7.56.2891-2894",
        "updated_pack_corrected_pack_refresh_gate_probe_split_reset",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_REPEAT_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2887-2890",
        "updated_pack_corrected_pack_refresh_sync_repeat_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
OLDER_SPLIT_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2839-2842",
        "updated_pack_corrected_probe_split_rederivation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
STEP_TAG = "8.7.56.2895-2898"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack corrected "
    "probe-split return audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_corrected_probe_split_return_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_pack_refresh_cycle_repeat_detected_probe_split_primary_"
    "mixed_kernel_secondary_hybrid_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_probe_split_return_audited_mixed_kernel_primary_"
    "vacuum_subtraction_secondary_gate"
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


# 関数: corrected probe-split return audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the corrected probe-split return audit."""
    return {
        "repeat_reset_rule": (
            "If corrected pack-refresh bookkeeping repeats with the same absent "
            "exact verdicts, return to the corrected dual-field probe split "
            "before reopening any reserve cycle."
        ),
        "dual_field_split": "P_mu(x) = Q_mu(x) + xi_mu(x),   A_mu(x): external probe",
        "background_stationarity": (
            "S_bg^(1)[Q;xi] = int d^4x xi_mu (delta S / delta P_mu)|_(P=Q) = 0"
        ),
        "ordering": (
            "corrected probe split return -> corrected mixed probe-response "
            "kernel return -> corrected vacuum subtraction"
        ),
    }


# 関数: `.2895-.2898` を実行する。

def main() -> None:
    """Execute the updated-pack corrected probe-split return audit."""
    for path in (PRIOR_GATE, PRIOR_REPEAT_AUDIT, OLDER_SPLIT_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_repeat_summary = sign_base.read_json(PRIOR_REPEAT_AUDIT)["summary"]
    older_split_summary = sign_base.read_json(OLDER_SPLIT_AUDIT)["summary"]

    selected = bool(
        prior_gate_summary["gate_b_updated_pack_corrected_probe_split_rederivation_promoted_next"]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    repeat_reset_from_corrected_cycle_explicit = bool(
        prior_repeat_summary["updated_pack_corrected_pack_refresh_cycle_exhaustion_machine_readable_now"]
        and prior_repeat_summary["updated_pack_corrected_pack_refresh_sync_cycle_repeat_detected"]
        and prior_repeat_summary["updated_pack_corrected_pack_refresh_sync_no_new_public_canonical_surface_now"]
    )
    prior_corrected_probe_split_surface_retained = bool(
        older_split_summary["updated_pack_corrected_probe_split_rederivation_machine_readable_now"]
        and older_split_summary["rank_matched_dual_field_probe_split_definition_explicit"]
        and older_split_summary["background_stationarity_only_for_self_fluctuation_explicit"]
    )
    target_surface_explicit = bool(
        selected
        and retry_mode
        and non_surrogate_guard
        and repeat_reset_from_corrected_cycle_explicit
        and prior_corrected_probe_split_surface_retained
    )
    machine_readable_now = bool(target_surface_explicit)
    exact_corrected_probe_split_formula_available_now = False
    exact_external_probe_current_vertex_formula_available_now = False
    corrected_mixed_kernel_primary_followup_required = bool(
        machine_readable_now and (not exact_corrected_probe_split_formula_available_now)
    )
    corrected_vacuum_subtraction_secondary_hold_retained = bool(
        corrected_mixed_kernel_primary_followup_required
    )
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])
    farther_hybrid = False

    rows = [
        sign_base.row(
            "updated_pack_corrected_probe_split_return_audit_selected",
            "pass" if selected else "reject",
            "updated-pack corrected probe-split return audit selected",
            sign_base.truth(selected),
            "After repeat-detected corrected bookkeeping, the honest move is to return to the corrected dual-field split rather than extend the cycle.",
        ),
        sign_base.row(
            "updated_pack_repeat_reset_from_corrected_cycle_explicit",
            "pass" if repeat_reset_from_corrected_cycle_explicit else "reject",
            "updated-pack repeat reset from corrected cycle explicit",
            sign_base.truth(repeat_reset_from_corrected_cycle_explicit),
            "The corrected pack-refresh cycle is now explicitly marked exhausted, so the route reset itself is machine-readable.",
        ),
        sign_base.row(
            "updated_pack_prior_corrected_probe_split_surface_retained",
            "pass" if prior_corrected_probe_split_surface_retained else "reject",
            "updated-pack prior corrected probe-split surface retained",
            sign_base.truth(prior_corrected_probe_split_surface_retained),
            "The previous corrected probe-split target survives the repeat reset and can be reused as the canonical return target.",
        ),
        sign_base.row(
            "updated_pack_corrected_probe_split_return_target_surface_explicit",
            "pass" if target_surface_explicit else "reject",
            "updated-pack corrected probe-split return target surface explicit",
            sign_base.truth(target_surface_explicit),
            "The repeat-reset branch now places the corrected dual-field split back on one explicit computation-side target surface.",
        ),
        sign_base.row(
            "updated_pack_corrected_probe_split_return_machine_readable_now",
            "pass" if machine_readable_now else "reject",
            "updated-pack corrected probe-split return machine-readable now",
            sign_base.truth(machine_readable_now),
            "The corrected split return is now explicit and machine-readable under the repeat-detected state.",
        ),
        sign_base.row(
            "exact_corrected_probe_split_formula_available_now",
            "pass" if exact_corrected_probe_split_formula_available_now else "reject",
            "exact corrected probe-split formula available now",
            sign_base.truth(exact_corrected_probe_split_formula_available_now),
            "The return branch restates the corrected split target but still does not derive the literal canonical split formula.",
        ),
        sign_base.row(
            "exact_external_probe_current_vertex_formula_available_now",
            "pass" if exact_external_probe_current_vertex_formula_available_now else "reject",
            "exact external-probe current-vertex formula available now",
            sign_base.truth(exact_external_probe_current_vertex_formula_available_now),
            "Current-vertex completion remains downstream of the corrected split return itself.",
        ),
        sign_base.row(
            "updated_pack_corrected_mixed_kernel_primary_followup_required",
            "pass" if corrected_mixed_kernel_primary_followup_required else "reject",
            "updated-pack corrected mixed-kernel primary followup required",
            sign_base.truth(corrected_mixed_kernel_primary_followup_required),
            "Once the corrected split return is restated, the next honest move is to restate the corrected mixed-kernel blocker under that return.",
        ),
        sign_base.row(
            "updated_pack_corrected_vacuum_subtraction_secondary_hold_retained",
            "pass" if corrected_vacuum_subtraction_secondary_hold_retained else "reject",
            "updated-pack corrected vacuum-subtraction secondary hold retained",
            sign_base.truth(corrected_vacuum_subtraction_secondary_hold_retained),
            "Vacuum subtraction stays downstream of the corrected split return and corrected mixed-kernel completion.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation remains downstream of the unresolved corrected split and corrected kernel theorem stack.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid),
            "Repeat-reset does not justify reopening extra q-range continuation.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_repeat_summary["retained_scalar_residual_rel"]),
        "updated_pack_corrected_probe_split_return_audit_selected": selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_repeat_reset_from_corrected_cycle_explicit": repeat_reset_from_corrected_cycle_explicit,
        "updated_pack_prior_corrected_probe_split_surface_retained": prior_corrected_probe_split_surface_retained,
        "updated_pack_corrected_probe_split_return_target_surface_explicit": target_surface_explicit,
        "updated_pack_corrected_probe_split_return_machine_readable_now": machine_readable_now,
        "exact_corrected_probe_split_formula_available_now": exact_corrected_probe_split_formula_available_now,
        "exact_external_probe_current_vertex_formula_available_now": exact_external_probe_current_vertex_formula_available_now,
        "updated_pack_corrected_mixed_kernel_primary_followup_required": corrected_mixed_kernel_primary_followup_required,
        "updated_pack_corrected_vacuum_subtraction_secondary_hold_retained": corrected_vacuum_subtraction_secondary_hold_retained,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid,
        "updated_pack_corrected_probe_split_return_breakthrough_passed_now": False,
        "recommended_next_route_or_none": "8.7.56.2899",
        "selected_followup_route_or_none": "8.7.56.2903",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.2897",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_repeat_audit": sign_base.display_path(PRIOR_REPEAT_AUDIT),
                "older_split_audit": sign_base.display_path(OLDER_SPLIT_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.2899",
                "followup_route": "8.7.56.2903",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_corrected_probe_split_return_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack corrected probe-split return audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
