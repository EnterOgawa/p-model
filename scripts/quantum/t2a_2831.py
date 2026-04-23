#!/usr/bin/env python3
"""Generate 8.7.56.2831-.2834 pack-refresh sync repeat-detection artifacts."""

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
        "8.7.56.2827-2830",
        "updated_pack_reserve_registry_gate_pack_refresh_sync",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2823-2826",
        "updated_pack_reserve_registry_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
OLDER_SYNC_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2807-2810",
        "updated_pack_pack_refresh_sync_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
STEP_TAG = "8.7.56.2831-2834"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack pack-refresh "
    "sync audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_pack_refresh_sync_repeat_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "reserve_registry_refresh_audited_pack_refresh_sync_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "pack_refresh_cycle_repeat_detected_probe_split_primary_kernel_secondary_gate"
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


# 関数: repeat-detection audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the repeat-detection pack-refresh audit."""
    return {
        "cycle_surface": (
            "pack-refresh sync -> hybrid reserve refresh -> reserve-registry "
            "refresh -> pack-refresh sync"
        ),
        "repeat_rule": (
            "If the same cycle returns with the same absent exact verdicts and "
            "no new canonical surface, the honest next move is a computation-side "
            "route reset rather than one more bookkeeping-only loop."
        ),
        "primary_followup": (
            "corrected probe split rederivation -> mixed probe-response kernel "
            "refresh -> only then vacuum-subtraction revisit"
        ),
    }


# 関数: `.2831-.2834` を実行する。

def main() -> None:
    """Execute the updated-pack pack-refresh sync repeat-detection audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT, OLDER_SYNC_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    older_sync_summary = sign_base.read_json(OLDER_SYNC_AUDIT)["summary"]

    selected = bool(
        prior_gate_summary["gate_b_updated_pack_pack_refresh_sync_promoted_next"]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    reserve_registry_refresh_surface_retained = bool(
        prior_audit_summary["updated_pack_reserve_registry_refresh_machine_readable_now"]
    )
    hybrid_reserve_refresh_surface_retained = bool(
        prior_audit_summary["updated_pack_hybrid_reserve_refresh_surface_retained"]
    )
    unresolved_pack_refresh_state_registry_explicit = bool(
        prior_audit_summary["updated_pack_unresolved_pack_refresh_state_registry_explicit"]
    )
    older_sync_surface_explicit = bool(
        older_sync_summary["updated_pack_pack_refresh_sync_machine_readable_now"]
    )
    exact_pack_refresh_sync_verdict_available_now = False
    exact_hybrid_reserve_judgement_available_now = False
    exact_reserve_registry_refresh_available_now = False
    cycle_repeat_detected = bool(
        older_sync_surface_explicit
        and reserve_registry_refresh_surface_retained
        and (not older_sync_summary["exact_pack_refresh_sync_verdict_available_now"])
        and (not exact_pack_refresh_sync_verdict_available_now)
        and (not exact_hybrid_reserve_judgement_available_now)
        and (not exact_reserve_registry_refresh_available_now)
    )
    no_new_public_canonical_surface_now = bool(
        cycle_repeat_detected
        and older_sync_summary["updated_pack_unresolved_subtraction_state_retained"]
        and unresolved_pack_refresh_state_registry_explicit
    )
    cycle_exhaustion_machine_readable_now = bool(
        selected
        and retry_mode
        and non_surrogate_guard
        and reserve_registry_refresh_surface_retained
        and hybrid_reserve_refresh_surface_retained
        and unresolved_pack_refresh_state_registry_explicit
        and cycle_repeat_detected
        and no_new_public_canonical_surface_now
    )
    corrected_probe_split_rederivation_primary_followup_required = bool(
        cycle_exhaustion_machine_readable_now
        and (not exact_pack_refresh_sync_verdict_available_now)
    )
    mixed_probe_response_kernel_secondary_followup_required = bool(
        corrected_probe_split_rederivation_primary_followup_required
    )
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])
    farther_hybrid = False

    rows = [
        sign_base.row(
            "updated_pack_pack_refresh_sync_audit_selected",
            "pass" if selected else "reject",
            "updated-pack pack-refresh sync audit selected",
            sign_base.truth(selected),
            "Reserve-registry refresh returned the corrected unresolved stack to pack-refresh sync, so the sync lane remains the honest immediate object.",
        ),
        sign_base.row(
            "updated_pack_reserve_registry_refresh_surface_retained",
            "pass" if reserve_registry_refresh_surface_retained else "reject",
            "updated-pack reserve-registry refresh surface retained",
            sign_base.truth(reserve_registry_refresh_surface_retained),
            "The current sync audit inherits the explicit reserve-registry surface instead of reopening stale surrogate families.",
        ),
        sign_base.row(
            "updated_pack_hybrid_reserve_refresh_surface_retained",
            "pass" if hybrid_reserve_refresh_surface_retained else "reject",
            "updated-pack hybrid-reserve refresh surface retained",
            sign_base.truth(hybrid_reserve_refresh_surface_retained),
            "Hybrid-reserve bookkeeping stays explicit inside the current pack-refresh sync return path.",
        ),
        sign_base.row(
            "updated_pack_unresolved_pack_refresh_state_registry_explicit",
            "pass" if unresolved_pack_refresh_state_registry_explicit else "reject",
            "updated-pack unresolved pack-refresh state registry explicit",
            sign_base.truth(unresolved_pack_refresh_state_registry_explicit),
            "The corrected unresolved stack is still being carried as one explicit registry object.",
        ),
        sign_base.row(
            "updated_pack_pack_refresh_sync_cycle_repeat_detected",
            "pass" if cycle_repeat_detected else "reject",
            "updated-pack pack-refresh sync cycle repeat detected",
            sign_base.truth(cycle_repeat_detected),
            "The current sync return reproduces the same absent exact verdicts seen in the earlier pack-refresh sync audit.",
        ),
        sign_base.row(
            "updated_pack_pack_refresh_sync_no_new_public_canonical_surface_now",
            "pass" if no_new_public_canonical_surface_now else "reject",
            "updated-pack pack-refresh sync no new public-canonical surface now",
            sign_base.truth(no_new_public_canonical_surface_now),
            "The cycle reappears without adding a new canonical theorem or verdict surface beyond bookkeeping explicitness.",
        ),
        sign_base.row(
            "updated_pack_pack_refresh_cycle_exhaustion_machine_readable_now",
            "pass" if cycle_exhaustion_machine_readable_now else "reject",
            "updated-pack pack-refresh cycle exhaustion machine-readable now",
            sign_base.truth(cycle_exhaustion_machine_readable_now),
            "Cycle exhaustion is now explicit as a machine-readable result rather than an informal impression.",
        ),
        sign_base.row(
            "exact_pack_refresh_sync_verdict_available_now",
            "pass" if exact_pack_refresh_sync_verdict_available_now else "reject",
            "exact pack-refresh sync verdict available now",
            sign_base.truth(exact_pack_refresh_sync_verdict_available_now),
            "The repeated cycle still does not produce a canonical corrected pack-refresh verdict.",
        ),
        sign_base.row(
            "updated_pack_corrected_probe_split_rederivation_primary_followup_required",
            "pass" if corrected_probe_split_rederivation_primary_followup_required else "reject",
            "updated-pack corrected probe-split rederivation primary followup required",
            sign_base.truth(corrected_probe_split_rederivation_primary_followup_required),
            "Because the pack-refresh cycle repeats without a new verdict, the honest next move is to return to corrected probe-split derivation.",
        ),
        sign_base.row(
            "updated_pack_mixed_probe_response_kernel_secondary_followup_required",
            "pass" if mixed_probe_response_kernel_secondary_followup_required else "reject",
            "updated-pack mixed probe-response kernel secondary followup required",
            sign_base.truth(mixed_probe_response_kernel_secondary_followup_required),
            "Kernel completion remains the next fallback once the corrected probe split is restated under the repeat-detection reset.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation remains downstream of the unresolved corrected split, kernel, and subtraction theorem stack.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid),
            "The repeat cycle still does not justify reopening extra q-range continuation.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_audit_summary["retained_scalar_residual_rel"]),
        "updated_pack_pack_refresh_sync_audit_selected": selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_reserve_registry_refresh_surface_retained": reserve_registry_refresh_surface_retained,
        "updated_pack_hybrid_reserve_refresh_surface_retained": hybrid_reserve_refresh_surface_retained,
        "updated_pack_unresolved_pack_refresh_state_registry_explicit": unresolved_pack_refresh_state_registry_explicit,
        "updated_pack_pack_refresh_sync_cycle_repeat_detected": cycle_repeat_detected,
        "updated_pack_pack_refresh_sync_no_new_public_canonical_surface_now": no_new_public_canonical_surface_now,
        "updated_pack_pack_refresh_cycle_exhaustion_machine_readable_now": cycle_exhaustion_machine_readable_now,
        "exact_pack_refresh_sync_verdict_available_now": exact_pack_refresh_sync_verdict_available_now,
        "exact_hybrid_reserve_judgement_available_now": exact_hybrid_reserve_judgement_available_now,
        "exact_reserve_registry_refresh_available_now": exact_reserve_registry_refresh_available_now,
        "updated_pack_corrected_probe_split_rederivation_primary_followup_required": corrected_probe_split_rederivation_primary_followup_required,
        "updated_pack_mixed_probe_response_kernel_secondary_followup_required": mixed_probe_response_kernel_secondary_followup_required,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid,
        "updated_pack_pack_refresh_sync_breakthrough_passed_now": False,
        "recommended_next_route_or_none": "8.7.56.2835",
        "selected_followup_route_or_none": "8.7.56.2839",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.2833",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "older_sync_audit": sign_base.display_path(OLDER_SYNC_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.2835",
                "followup_route": "8.7.56.2839",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_pack_refresh_repeat_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack pack-refresh repeat audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
