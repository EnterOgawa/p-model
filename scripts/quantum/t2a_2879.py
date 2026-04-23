#!/usr/bin/env python3
"""Generate 8.7.56.2879-.2882 corrected reserve-registry refresh audit artifacts."""

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
        "8.7.56.2875-2878",
        "updated_pack_corrected_hybrid_reserve_gate_reserve_registry_refresh",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2871-2874",
        "updated_pack_corrected_hybrid_reserve_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
STEP_TAG = "8.7.56.2879-2882"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack corrected "
    "reserve-registry refresh audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_corrected_reserve_registry_refresh_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_hybrid_reserve_refresh_audited_reserve_registry_refresh_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_reserve_registry_refresh_audited_pack_refresh_sync_"
    "secondary_gate"
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


# 関数: corrected reserve-registry refresh audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the corrected reserve-registry refresh audit."""
    return {
        "reserve_registry_role": (
            "corrected reserve-registry refresh := keep the unresolved corrected "
            "pack-refresh stack and reserve-only hybrid evidence explicit without "
            "pretending that the canonical corrected verdict already exists"
        ),
        "registry_order": (
            "corrected pack-refresh sync -> corrected hybrid reserve refresh -> "
            "corrected reserve-registry refresh -> corrected pack-refresh sync -> "
            "only then extra q-range reopen"
        ),
        "reopen_rule": (
            "Farther hybrid continuation reopens only if exact residual-origin "
            "discrimination still requires extra q-range after the corrected "
            "reserve-registry state closes honestly."
        ),
    }


# 関数: `.2879-.2882` を実行する。

def main() -> None:
    """Execute the updated-pack corrected reserve-registry refresh audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    selected = bool(
        prior_gate_summary["gate_b_updated_pack_reserve_registry_refresh_promoted_next"]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    hybrid_reserve_refresh_surface_retained = bool(
        prior_audit_summary["updated_pack_hybrid_reserve_refresh_machine_readable_now"]
    )
    unresolved_corrected_pack_refresh_state_registry_explicit = bool(
        prior_audit_summary["updated_pack_unresolved_corrected_subtraction_state_retained"]
        and prior_audit_summary["updated_pack_corrected_mixed_kernel_gap_retained"]
        and (not prior_audit_summary["exact_pack_refresh_sync_verdict_available_now"])
        and (not prior_audit_summary["exact_hybrid_reserve_judgement_available_now"])
    )
    farther_hybrid_hold_registry_explicit = bool(
        prior_audit_summary["updated_pack_farther_hybrid_extra_q_range_hold_explicit"]
        and prior_audit_summary["updated_pack_extra_q_range_reopen_condition_explicit"]
        and (not prior_gate_summary["gate_c_farther_hybrid_continuation_reopen_required_now"])
    )
    pack_refresh_sync_followup_explicit = bool(
        prior_gate_summary["selected_secondary_completion_lane"]
        == "updated_pack_reserve_registry_gate_pack_refresh_sync"
    )
    target_surface_explicit = bool(
        selected
        and retry_mode
        and non_surrogate_guard
        and hybrid_reserve_refresh_surface_retained
        and unresolved_corrected_pack_refresh_state_registry_explicit
        and farther_hybrid_hold_registry_explicit
        and pack_refresh_sync_followup_explicit
    )
    machine_readable_now = bool(target_surface_explicit)
    exact_pack_refresh_sync_verdict_available_now = False
    exact_hybrid_reserve_judgement_available_now = False
    exact_reserve_registry_refresh_available_now = False
    reserve_registry_pack_refresh_primary_followup_required = bool(
        machine_readable_now and (not exact_reserve_registry_refresh_available_now)
    )
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])
    farther_hybrid = False

    rows = [
        sign_base.row(
            "updated_pack_corrected_reserve_registry_refresh_audit_selected",
            "pass" if selected else "reject",
            "updated-pack corrected reserve-registry refresh audit selected",
            sign_base.truth(selected),
            "Once corrected hybrid reserve policy is explicit but unresolved, the honest next move is to register that unresolved corrected state explicitly rather than reopen computation.",
        ),
        sign_base.row(
            "updated_pack_hybrid_reserve_refresh_surface_retained",
            "pass" if hybrid_reserve_refresh_surface_retained else "reject",
            "updated-pack hybrid-reserve refresh surface retained",
            sign_base.truth(hybrid_reserve_refresh_surface_retained),
            "The corrected registry inherits the already localized corrected reserve-policy surface instead of replacing it with a new surrogate bookkeeping rule.",
        ),
        sign_base.row(
            "updated_pack_unresolved_corrected_pack_refresh_state_registry_explicit",
            "pass" if unresolved_corrected_pack_refresh_state_registry_explicit else "reject",
            "updated-pack unresolved corrected pack-refresh state registry explicit",
            sign_base.truth(unresolved_corrected_pack_refresh_state_registry_explicit),
            "The registry records that the corrected mixed-kernel gap, corrected subtraction theorem gap, and corrected hybrid judgement all remain unresolved.",
        ),
        sign_base.row(
            "updated_pack_farther_hybrid_hold_registry_explicit",
            "pass" if farther_hybrid_hold_registry_explicit else "reject",
            "updated-pack farther hybrid hold registry explicit",
            sign_base.truth(farther_hybrid_hold_registry_explicit),
            "The registry keeps the farther hybrid lane in reserve-only status together with its controlled reopen condition.",
        ),
        sign_base.row(
            "updated_pack_reserve_registry_pack_refresh_followup_explicit",
            "pass" if pack_refresh_sync_followup_explicit else "reject",
            "updated-pack reserve-registry pack-refresh followup explicit",
            sign_base.truth(pack_refresh_sync_followup_explicit),
            "The corrected branch already knows that registry bookkeeping must feed back into one dedicated pack-refresh sync lane after this audit.",
        ),
        sign_base.row(
            "updated_pack_corrected_reserve_registry_refresh_target_surface_explicit",
            "pass" if target_surface_explicit else "reject",
            "updated-pack corrected reserve-registry refresh target surface explicit",
            sign_base.truth(target_surface_explicit),
            "The target surface is explicit: keep the unresolved corrected pack-refresh stack and the farther-hybrid hold rule on one corrected registry object.",
        ),
        sign_base.row(
            "updated_pack_reserve_registry_refresh_machine_readable_now",
            "pass" if machine_readable_now else "reject",
            "updated-pack reserve-registry refresh machine-readable now",
            sign_base.truth(machine_readable_now),
            "The corrected reserve registry now lives on one explicit machine-readable surface under the corrected pack-refresh ordering.",
        ),
        sign_base.row(
            "exact_pack_refresh_sync_verdict_available_now",
            "pass" if exact_pack_refresh_sync_verdict_available_now else "reject",
            "exact pack-refresh sync verdict available now",
            sign_base.truth(exact_pack_refresh_sync_verdict_available_now),
            "The corrected registry does not solve the missing corrected pack-refresh verdict by itself.",
        ),
        sign_base.row(
            "exact_hybrid_reserve_judgement_available_now",
            "pass" if exact_hybrid_reserve_judgement_available_now else "reject",
            "exact hybrid-reserve judgement available now",
            sign_base.truth(exact_hybrid_reserve_judgement_available_now),
            "The registry keeps the scientific judgement pending instead of pretending that the retained hybrid evidence is already adjudicated.",
        ),
        sign_base.row(
            "exact_reserve_registry_refresh_available_now",
            "pass" if exact_reserve_registry_refresh_available_now else "reject",
            "exact reserve-registry refresh available now",
            sign_base.truth(exact_reserve_registry_refresh_available_now),
            "This audit localizes the corrected registry state but does not yet close it canonically.",
        ),
        sign_base.row(
            "updated_pack_reserve_registry_pack_refresh_primary_followup_required",
            "pass" if reserve_registry_pack_refresh_primary_followup_required else "reject",
            "updated-pack reserve-registry pack-refresh primary followup required",
            sign_base.truth(reserve_registry_pack_refresh_primary_followup_required),
            "Once the unresolved corrected state is registered honestly, the next honest move is to sync that registry back into the corrected pack-refresh surface.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation remains blocked until theorem, normalization, corrected split, corrected kernel, corrected subtraction, and corrected reserve judgement all close honestly.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid),
            "The corrected registry explicitly keeps farther hybrid continuation closed.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_audit_summary["retained_scalar_residual_rel"]),
        "updated_pack_corrected_reserve_registry_refresh_audit_selected": selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_hybrid_reserve_refresh_surface_retained": hybrid_reserve_refresh_surface_retained,
        "updated_pack_unresolved_corrected_pack_refresh_state_registry_explicit": unresolved_corrected_pack_refresh_state_registry_explicit,
        "updated_pack_farther_hybrid_hold_registry_explicit": farther_hybrid_hold_registry_explicit,
        "updated_pack_reserve_registry_pack_refresh_followup_explicit": pack_refresh_sync_followup_explicit,
        "updated_pack_corrected_reserve_registry_refresh_target_surface_explicit": target_surface_explicit,
        "updated_pack_reserve_registry_refresh_machine_readable_now": machine_readable_now,
        "exact_pack_refresh_sync_verdict_available_now": exact_pack_refresh_sync_verdict_available_now,
        "exact_hybrid_reserve_judgement_available_now": exact_hybrid_reserve_judgement_available_now,
        "exact_reserve_registry_refresh_available_now": exact_reserve_registry_refresh_available_now,
        "updated_pack_reserve_registry_pack_refresh_primary_followup_required": reserve_registry_pack_refresh_primary_followup_required,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid,
        "recommended_next_route_or_none": "8.7.56.2883",
        "selected_followup_route_or_none": "8.7.56.2887",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.2881",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.2883",
                "followup_route": "8.7.56.2887",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_corrected_reserve_registry_refresh_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack corrected reserve-registry refresh audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
