#!/usr/bin/env python3
"""Generate 8.7.56.2855-.2858 corrected vacuum-subtraction refresh audit artifacts."""

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
        "8.7.56.2851-2854",
        "updated_pack_mixed_kernel_gate_vacuum_subtraction_refresh",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_SPLIT_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2839-2842",
        "updated_pack_corrected_probe_split_rederivation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_MIXED_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2847-2850",
        "updated_pack_mixed_probe_response_kernel_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_VACUUM_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2799-2802",
        "updated_pack_vacuum_subtraction_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
STEP_TAG = "8.7.56.2855-2858"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack corrected "
    "vacuum-subtraction refresh audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_corrected_vacuum_subtraction_refresh_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "mixed_kernel_refresh_audited_vacuum_subtraction_primary_pack_refresh_"
    "secondary_hybrid_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_vacuum_subtraction_audited_pack_refresh_sync_primary_"
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


# 関数: corrected subtraction refresh で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the corrected vacuum-subtraction audit."""
    return {
        "corrected_split": "P_mu(x) = Q_mu(x) + xi_mu(x),   A_mu(x): external probe",
        "corrected_vacuum_subtraction": (
            "Delta O = O_corrected[Q;xi,A] - O_corrected[vac;xi,A]"
        ),
        "ordering": (
            "corrected probe split -> corrected mixed kernel -> corrected vacuum subtraction -> pack refresh"
        ),
        "legacy_warning": "legacy caseA v^2 subtraction worsen retained as noncanonical warning only",
    }


# 関数: `.2855-.2858` を実行する。

def main() -> None:
    """Execute the updated-pack corrected vacuum-subtraction refresh audit."""
    for path in (PRIOR_GATE, PRIOR_SPLIT_AUDIT, PRIOR_MIXED_AUDIT, PRIOR_VACUUM_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_split_summary = sign_base.read_json(PRIOR_SPLIT_AUDIT)["summary"]
    prior_mixed_summary = sign_base.read_json(PRIOR_MIXED_AUDIT)["summary"]
    prior_vacuum_summary = sign_base.read_json(PRIOR_VACUUM_AUDIT)["summary"]

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
    corrected_split_machine_readable = bool(
        prior_split_summary["updated_pack_corrected_probe_split_rederivation_machine_readable_now"]
    )
    mixed_kernel_refresh_machine_readable = bool(
        prior_mixed_summary["updated_pack_mixed_probe_response_kernel_refresh_machine_readable_now"]
    )
    vacuum_subtraction_surface_explicit = bool(
        prior_vacuum_summary["pure_derivation_vacuum_subtraction_surface_explicit"]
        and prior_vacuum_summary["pure_derivation_vacuum_state_surface_explicit"]
        and prior_vacuum_summary["pure_derivation_box_divergence_surface_explicit"]
        and prior_vacuum_summary["pure_derivation_divergent_vacuum_amplitude_explicit"]
    )
    legacy_casea_warning_retained = bool(
        prior_vacuum_summary["legacy_casea_v2_subtraction_worsen_retained"]
        and prior_vacuum_summary["legacy_casea_v2_subtraction_noncanonical_for_probe_lane"]
    )
    corrected_subtraction_target_surface = bool(
        selected
        and retry_mode
        and non_surrogate_guard
        and corrected_split_machine_readable
        and mixed_kernel_refresh_machine_readable
        and vacuum_subtraction_surface_explicit
        and legacy_casea_warning_retained
    )
    corrected_subtraction_machine_readable = bool(corrected_subtraction_target_surface)
    exact_corrected_vacuum_state_definition_available_now = False
    exact_corrected_vacuum_subtraction_rule_available_now = False
    exact_corrected_subtracted_observable_rank_match_available_now = False
    pack_refresh_sync_followup_required = bool(
        corrected_subtraction_machine_readable
        and (not exact_corrected_vacuum_subtraction_rule_available_now)
    )
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])
    farther_hybrid = False

    rows = [
        sign_base.row(
            "updated_pack_corrected_vacuum_subtraction_refresh_audit_selected",
            "pass" if selected else "reject",
            "updated-pack corrected vacuum-subtraction refresh audit selected",
            sign_base.truth(selected),
            "The corrected mixed-kernel gate already promoted subtraction refresh under the corrected probe ordering.",
        ),
        sign_base.row(
            "updated_pack_corrected_probe_split_rederivation_machine_readable_now",
            "pass" if corrected_split_machine_readable else "reject",
            "updated-pack corrected probe-split rederivation machine-readable now",
            sign_base.truth(corrected_split_machine_readable),
            "The subtraction lane now explicitly depends on the corrected dual-field split surface.",
        ),
        sign_base.row(
            "updated_pack_mixed_probe_response_kernel_refresh_machine_readable_now",
            "pass" if mixed_kernel_refresh_machine_readable else "reject",
            "updated-pack mixed probe-response kernel refresh machine-readable now",
            sign_base.truth(mixed_kernel_refresh_machine_readable),
            "The subtraction lane also depends on the corrected mixed-kernel refresh surface rather than on the old single-symbol amplitude read.",
        ),
        sign_base.row(
            "pure_derivation_vacuum_subtraction_surface_explicit",
            "pass" if vacuum_subtraction_surface_explicit else "reject",
            "pure-derivation vacuum-subtraction surface explicit",
            sign_base.truth(vacuum_subtraction_surface_explicit),
            "The note still exposes subtraction, vacuum-state, and divergence surfaces, which remain reusable after the corrected split reset.",
        ),
        sign_base.row(
            "legacy_casea_v2_subtraction_warning_retained",
            "pass" if legacy_casea_warning_retained else "reject",
            "legacy caseA v^2 subtraction warning retained",
            sign_base.truth(legacy_casea_warning_retained),
            "The old Minkowski worsen result remains only as a noncanonical warning and not as the corrected subtraction theorem.",
        ),
        sign_base.row(
            "updated_pack_corrected_vacuum_subtraction_target_surface_explicit",
            "pass" if corrected_subtraction_target_surface else "reject",
            "updated-pack corrected vacuum-subtraction target surface explicit",
            sign_base.truth(corrected_subtraction_target_surface),
            "The corrected split, corrected mixed kernel, and subtraction surfaces now sit on one explicit target surface.",
        ),
        sign_base.row(
            "updated_pack_corrected_vacuum_subtraction_machine_readable_now",
            "pass" if corrected_subtraction_machine_readable else "reject",
            "updated-pack corrected vacuum subtraction machine-readable now",
            sign_base.truth(corrected_subtraction_machine_readable),
            "The corrected subtraction blocker is now explicit and machine-readable under the post-reset ordering.",
        ),
        sign_base.row(
            "exact_corrected_vacuum_state_definition_available_now",
            "pass" if exact_corrected_vacuum_state_definition_available_now else "reject",
            "exact corrected vacuum-state definition available now",
            sign_base.truth(exact_corrected_vacuum_state_definition_available_now),
            "The corrected probe/kernel lane still lacks a canonical vacuum-state theorem.",
        ),
        sign_base.row(
            "exact_corrected_vacuum_subtraction_rule_available_now",
            "pass" if exact_corrected_vacuum_subtraction_rule_available_now else "reject",
            "exact corrected vacuum-subtraction rule available now",
            sign_base.truth(exact_corrected_vacuum_subtraction_rule_available_now),
            "Without a literal corrected mixed kernel and a closed vacuum-state theorem, subtraction cannot honestly close here.",
        ),
        sign_base.row(
            "exact_corrected_subtracted_observable_rank_match_available_now",
            "pass" if exact_corrected_subtracted_observable_rank_match_available_now else "reject",
            "exact corrected subtracted observable rank match available now",
            sign_base.truth(exact_corrected_subtracted_observable_rank_match_available_now),
            "The subtraction target still lacks a closed rank-matched corrected observable theorem.",
        ),
        sign_base.row(
            "updated_pack_pack_refresh_sync_followup_required",
            "pass" if pack_refresh_sync_followup_required else "reject",
            "updated-pack pack-refresh sync followup required",
            sign_base.truth(pack_refresh_sync_followup_required),
            "Once corrected subtraction is localized but unresolved, the honest next move is to sync that unresolved state into pack refresh again.",
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
            "Extra q-range evidence remains reserve-only because the blocker is still corrected subtraction closure.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_corrected_vacuum_subtraction_refresh_audit_selected": selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_corrected_probe_split_rederivation_machine_readable_now": corrected_split_machine_readable,
        "updated_pack_mixed_probe_response_kernel_refresh_machine_readable_now": mixed_kernel_refresh_machine_readable,
        "pure_derivation_vacuum_subtraction_surface_explicit": vacuum_subtraction_surface_explicit,
        "legacy_casea_v2_subtraction_warning_retained": legacy_casea_warning_retained,
        "updated_pack_corrected_vacuum_subtraction_target_surface_explicit": corrected_subtraction_target_surface,
        "updated_pack_corrected_vacuum_subtraction_machine_readable_now": corrected_subtraction_machine_readable,
        "exact_corrected_vacuum_state_definition_available_now": exact_corrected_vacuum_state_definition_available_now,
        "exact_corrected_vacuum_subtraction_rule_available_now": exact_corrected_vacuum_subtraction_rule_available_now,
        "exact_corrected_subtracted_observable_rank_match_available_now": exact_corrected_subtracted_observable_rank_match_available_now,
        "updated_pack_pack_refresh_sync_followup_required": pack_refresh_sync_followup_required,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid,
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_corrected_vacuum_subtraction_gate_pack_refresh_sync",
        "recommended_next_route_or_none": "8.7.56.2859",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_pack_refresh_sync_audit",
        "selected_followup_route_or_none": "8.7.56.2863",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.2857",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_split_audit": sign_base.display_path(PRIOR_SPLIT_AUDIT),
                "prior_mixed_audit": sign_base.display_path(PRIOR_MIXED_AUDIT),
                "prior_vacuum_audit": sign_base.display_path(PRIOR_VACUUM_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.2859",
                "followup_route": "8.7.56.2863",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_corrected_vacuum_subtraction_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack corrected vacuum-subtraction refresh completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
