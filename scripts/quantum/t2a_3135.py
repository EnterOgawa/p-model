#!/usr/bin/env python3
"""Generate 8.7.56.3135-.3138 corrected vacuum-subtraction return audit artifacts."""

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
        "8.7.56.3131-3134",
        "updated_pack_corrected_mixed_kernel_gate_vacuum_subtraction_refresh",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_SPLIT_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.3119-3122",
        "updated_pack_corrected_probe_split_return_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_MIXED_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.3127-3130",
        "updated_pack_corrected_mixed_kernel_return_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
OLDER_VACUUM_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.3079-3082",
        "updated_pack_corrected_vacuum_subtraction_return_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
STEP_TAG = "8.7.56.3135-3138"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack corrected "
    "vacuum-subtraction return refresh audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_corrected_vacuum_subtraction_return_refresh_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_mixed_kernel_return_audited_vacuum_subtraction_primary_"
    "pack_refresh_secondary_hybrid_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_vacuum_subtraction_return_audited_pack_refresh_primary_"
    "hybrid_reserve_gate"
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


# 関数: corrected subtraction return で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the corrected vacuum-subtraction return audit."""
    return {
        "corrected_split": "P_mu(x) = Q_mu(x) + xi_mu(x),   A_mu(x): external probe",
        "corrected_vacuum_subtraction": (
            "Delta O_corrected = O_corrected[Q;xi,A] - O_corrected[vac;xi,A]"
        ),
        "ordering": (
            "corrected probe split return -> corrected mixed kernel return -> "
            "corrected vacuum subtraction return -> corrected pack-refresh sync"
        ),
        "legacy_warning": (
            "legacy caseA Minkowski v^2 subtraction worsen retained only as "
            "noncanonical warning surface"
        ),
    }


# 関数: `.3135-.3138` を実行する。

def main() -> None:
    """Execute the updated-pack corrected vacuum-subtraction return refresh audit."""
    for path in (PRIOR_GATE, PRIOR_SPLIT_AUDIT, PRIOR_MIXED_AUDIT, OLDER_VACUUM_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_split_summary = sign_base.read_json(PRIOR_SPLIT_AUDIT)["summary"]
    prior_mixed_summary = sign_base.read_json(PRIOR_MIXED_AUDIT)["summary"]
    older_vacuum_summary = sign_base.read_json(OLDER_VACUUM_AUDIT)["summary"]

    selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_corrected_vacuum_subtraction_refresh_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    corrected_probe_split_return_machine_readable = bool(
        prior_split_summary["updated_pack_corrected_probe_split_return_machine_readable_now"]
    )
    corrected_mixed_kernel_return_machine_readable = bool(
        prior_mixed_summary["updated_pack_corrected_mixed_kernel_return_machine_readable_now"]
    )
    prior_vacuum_surface_explicit = bool(
        older_vacuum_summary[
            "updated_pack_corrected_vacuum_subtraction_return_machine_readable_now"
        ]
        and older_vacuum_summary[
            "updated_pack_corrected_vacuum_subtraction_return_target_surface_explicit"
        ]
    )
    legacy_warning_retained = bool(
        older_vacuum_summary["legacy_casea_v2_subtraction_warning_retained"]
    )
    target_surface_explicit = bool(
        selected
        and retry_mode
        and non_surrogate_guard
        and corrected_probe_split_return_machine_readable
        and corrected_mixed_kernel_return_machine_readable
        and prior_vacuum_surface_explicit
        and legacy_warning_retained
    )
    machine_readable_now = bool(target_surface_explicit)
    exact_corrected_vacuum_state_definition_available_now = False
    exact_corrected_vacuum_subtraction_rule_available_now = False
    exact_corrected_subtracted_observable_rank_match_available_now = False
    corrected_pack_refresh_primary_followup_required = bool(
        machine_readable_now and (not exact_corrected_vacuum_subtraction_rule_available_now)
    )
    corrected_hybrid_reserve_secondary_hold_retained = bool(
        corrected_pack_refresh_primary_followup_required
        and (not exact_corrected_subtracted_observable_rank_match_available_now)
    )
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])
    farther_hybrid = False

    rows = [
        sign_base.row(
            "updated_pack_corrected_vacuum_subtraction_return_audit_selected",
            "pass" if selected else "reject",
            "updated-pack corrected vacuum-subtraction return audit selected",
            sign_base.truth(selected),
            "Once corrected mixed-kernel return is restated, the honest next move is to restate corrected subtraction under the same return ordering.",
        ),
        sign_base.row(
            "updated_pack_corrected_probe_split_return_machine_readable_now",
            "pass" if corrected_probe_split_return_machine_readable else "reject",
            "updated-pack corrected probe-split return machine-readable now",
            sign_base.truth(corrected_probe_split_return_machine_readable),
            "The corrected subtraction return inherits the already explicit corrected split return surface.",
        ),
        sign_base.row(
            "updated_pack_corrected_mixed_kernel_return_machine_readable_now",
            "pass" if corrected_mixed_kernel_return_machine_readable else "reject",
            "updated-pack corrected mixed-kernel return machine-readable now",
            sign_base.truth(corrected_mixed_kernel_return_machine_readable),
            "The corrected subtraction return also inherits the already explicit corrected mixed-kernel return surface.",
        ),
        sign_base.row(
            "updated_pack_prior_vacuum_subtraction_surface_retained",
            "pass" if prior_vacuum_surface_explicit else "reject",
            "updated-pack prior vacuum-subtraction surface retained",
            sign_base.truth(prior_vacuum_surface_explicit),
            "The earlier corrected subtraction return surface remains reusable inside the current return lane.",
        ),
        sign_base.row(
            "legacy_casea_v2_subtraction_warning_retained",
            "pass" if legacy_warning_retained else "reject",
            "legacy caseA v^2 subtraction warning retained",
            sign_base.truth(legacy_warning_retained),
            "The old Minkowski worsen result remains only as a noncanonical warning and not as the corrected subtraction theorem.",
        ),
        sign_base.row(
            "updated_pack_corrected_vacuum_subtraction_return_target_surface_explicit",
            "pass" if target_surface_explicit else "reject",
            "updated-pack corrected vacuum-subtraction return target surface explicit",
            sign_base.truth(target_surface_explicit),
            "The corrected split return, corrected mixed-kernel return, and subtraction surfaces now sit on one explicit target surface.",
        ),
        sign_base.row(
            "updated_pack_corrected_vacuum_subtraction_return_machine_readable_now",
            "pass" if machine_readable_now else "reject",
            "updated-pack corrected vacuum subtraction return machine-readable now",
            sign_base.truth(machine_readable_now),
            "The corrected subtraction blocker is now explicit and machine-readable under the post-reset return ordering.",
        ),
        sign_base.row(
            "exact_corrected_vacuum_state_definition_available_now",
            "pass" if exact_corrected_vacuum_state_definition_available_now else "reject",
            "exact corrected vacuum-state definition available now",
            sign_base.truth(exact_corrected_vacuum_state_definition_available_now),
            "The corrected probe/kernel lane still lacks a canonical corrected vacuum-state theorem.",
        ),
        sign_base.row(
            "exact_corrected_vacuum_subtraction_rule_available_now",
            "pass" if exact_corrected_vacuum_subtraction_rule_available_now else "reject",
            "exact corrected vacuum-subtraction rule available now",
            sign_base.truth(exact_corrected_vacuum_subtraction_rule_available_now),
            "Without a literal corrected mixed kernel and a closed corrected vacuum-state theorem, subtraction cannot honestly close here.",
        ),
        sign_base.row(
            "exact_corrected_subtracted_observable_rank_match_available_now",
            "pass" if exact_corrected_subtracted_observable_rank_match_available_now else "reject",
            "exact corrected subtracted observable rank match available now",
            sign_base.truth(exact_corrected_subtracted_observable_rank_match_available_now),
            "The subtraction target still lacks a closed rank-matched corrected observable theorem.",
        ),
        sign_base.row(
            "updated_pack_corrected_pack_refresh_primary_followup_required",
            "pass" if corrected_pack_refresh_primary_followup_required else "reject",
            "updated-pack corrected pack-refresh primary followup required",
            sign_base.truth(corrected_pack_refresh_primary_followup_required),
            "Once corrected subtraction return is localized but unresolved, the honest next move is corrected pack-refresh sync return.",
        ),
        sign_base.row(
            "updated_pack_corrected_hybrid_reserve_secondary_hold_retained",
            "pass" if corrected_hybrid_reserve_secondary_hold_retained else "reject",
            "updated-pack corrected hybrid-reserve secondary hold retained",
            sign_base.truth(corrected_hybrid_reserve_secondary_hold_retained),
            "Hybrid-reserve bookkeeping stays secondary until corrected pack-refresh sync return is restated.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation remains downstream of the unresolved corrected split, corrected kernel, and corrected subtraction theorem stack.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid),
            "Corrected subtraction return does not justify reopening extra q-range continuation.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_corrected_vacuum_subtraction_return_audit_selected": selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_corrected_probe_split_return_machine_readable_now": corrected_probe_split_return_machine_readable,
        "updated_pack_corrected_mixed_kernel_return_machine_readable_now": corrected_mixed_kernel_return_machine_readable,
        "updated_pack_prior_vacuum_subtraction_surface_retained": prior_vacuum_surface_explicit,
        "legacy_casea_v2_subtraction_warning_retained": legacy_warning_retained,
        "updated_pack_corrected_vacuum_subtraction_return_target_surface_explicit": target_surface_explicit,
        "updated_pack_corrected_vacuum_subtraction_return_machine_readable_now": machine_readable_now,
        "exact_corrected_vacuum_state_definition_available_now": exact_corrected_vacuum_state_definition_available_now,
        "exact_corrected_vacuum_subtraction_rule_available_now": exact_corrected_vacuum_subtraction_rule_available_now,
        "exact_corrected_subtracted_observable_rank_match_available_now": exact_corrected_subtracted_observable_rank_match_available_now,
        "updated_pack_corrected_pack_refresh_primary_followup_required": corrected_pack_refresh_primary_followup_required,
        "updated_pack_corrected_hybrid_reserve_secondary_hold_retained": corrected_hybrid_reserve_secondary_hold_retained,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid,
        "selected_primary_completion_lane": "updated_pack_corrected_pack_refresh_return_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_gate_hybrid_reserve_return",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_corrected_pack_refresh_return_audit",
        "recommended_next_route_or_none": "8.7.56.3139",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_corrected_pack_refresh_gate_hybrid_reserve_return",
        "selected_followup_route_or_none": "8.7.56.3143",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.3137",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_split_audit": sign_base.display_path(PRIOR_SPLIT_AUDIT),
                "prior_mixed_audit": sign_base.display_path(PRIOR_MIXED_AUDIT),
                "older_vacuum_audit": sign_base.display_path(OLDER_VACUUM_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.3139",
                "followup_route": "8.7.56.3143",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_corrected_vacuum_subtraction_return_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack corrected vacuum-subtraction return audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
