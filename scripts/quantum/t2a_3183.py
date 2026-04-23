#!/usr/bin/env python3
"""Generate 8.7.56.3183-.3186 corrected mixed-kernel return refresh artifacts."""

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
        "8.7.56.3179-3182",
        "updated_pack_probe_split_return_gate_mixed_kernel_refresh",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_SPLIT_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.3175-3178",
        "updated_pack_corrected_probe_split_return_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
OLDER_KERNEL_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.3127-3130",
        "updated_pack_corrected_mixed_kernel_return_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
STEP_TAG = "8.7.56.3183-3186"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack corrected "
    "mixed-kernel return refresh audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_corrected_mixed_kernel_return_refresh_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_probe_split_return_audited_mixed_kernel_primary_"
    "hybrid_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_mixed_kernel_return_audited_vacuum_subtraction_primary_"
    "pack_refresh_secondary_gate"
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


# 関数: corrected mixed-kernel return refresh で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the corrected mixed-kernel return refresh audit."""
    return {
        "corrected_split": "P_mu(x) = Q_mu(x) + xi_mu(x),   A_mu(x): external probe",
        "mixed_probe_kernel": (
            "V^{mu nu}[Q](x,y) := delta^2 S_frozen / (delta xi_mu(x) delta A_nu(y))"
            " |_(Q,A=0)"
        ),
        "pure_probe_kernel": (
            "Pi^{mu nu}[Q](x,y) := delta^2 S_frozen / (delta A_mu(x) delta A_nu(y))"
            " |_(Q,A=0)"
        ),
        "ordering": (
            "corrected probe split return -> corrected mixed/pure kernel return -> "
            "corrected vacuum subtraction"
        ),
    }


# 関数: `.3183-.3186` を実行する。

def main() -> None:
    """Execute the updated-pack corrected mixed-kernel return refresh audit."""
    for path in (PRIOR_GATE, PRIOR_SPLIT_AUDIT, OLDER_KERNEL_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_split_summary = sign_base.read_json(PRIOR_SPLIT_AUDIT)["summary"]
    older_kernel_summary = sign_base.read_json(OLDER_KERNEL_AUDIT)["summary"]

    selected = bool(
        prior_gate_summary["gate_b_updated_pack_corrected_mixed_kernel_refresh_promoted_next"]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    corrected_probe_split_return_machine_readable = bool(
        prior_split_summary["updated_pack_corrected_probe_split_return_machine_readable_now"]
    )
    older_mixed_kernel_surface_retained = bool(
        older_kernel_summary["updated_pack_corrected_mixed_kernel_return_machine_readable_now"]
        and older_kernel_summary["updated_pack_corrected_mixed_kernel_return_target_surface_explicit"]
    )
    target_surface_explicit = bool(
        selected
        and retry_mode
        and non_surrogate_guard
        and corrected_probe_split_return_machine_readable
        and older_mixed_kernel_surface_retained
    )
    machine_readable_now = bool(target_surface_explicit)
    exact_corrected_mixed_probe_response_kernel_formula_available_now = False
    exact_corrected_pure_probe_response_kernel_formula_available_now = False
    exact_corrected_kernel_rank_match_available_now = False
    corrected_vacuum_subtraction_primary_followup_required = bool(
        machine_readable_now
        and (not exact_corrected_mixed_probe_response_kernel_formula_available_now)
    )
    corrected_pack_refresh_secondary_hold_retained = bool(
        corrected_vacuum_subtraction_primary_followup_required
        and (not exact_corrected_kernel_rank_match_available_now)
    )
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])
    farther_hybrid = False

    rows = [
        sign_base.row(
            "updated_pack_corrected_mixed_kernel_return_refresh_audit_selected",
            "pass" if selected else "reject",
            "updated-pack corrected mixed-kernel return refresh audit selected",
            sign_base.truth(selected),
            "Once the corrected split return is promoted again, the honest next move is to restate the corrected mixed-kernel blocker under that return.",
        ),
        sign_base.row(
            "updated_pack_corrected_probe_split_return_machine_readable_now",
            "pass" if corrected_probe_split_return_machine_readable else "reject",
            "updated-pack corrected probe-split return machine-readable now",
            sign_base.truth(corrected_probe_split_return_machine_readable),
            "The corrected mixed-kernel return inherits the already explicit corrected split return surface.",
        ),
        sign_base.row(
            "updated_pack_prior_corrected_mixed_kernel_surface_retained",
            "pass" if older_mixed_kernel_surface_retained else "reject",
            "updated-pack prior corrected mixed-kernel surface retained",
            sign_base.truth(older_mixed_kernel_surface_retained),
            "The retained corrected return-kernel surface survives the repeat-reset and can be reused as the canonical mixed-kernel target.",
        ),
        sign_base.row(
            "updated_pack_corrected_mixed_kernel_return_target_surface_explicit",
            "pass" if target_surface_explicit else "reject",
            "updated-pack corrected mixed-kernel return target surface explicit",
            sign_base.truth(target_surface_explicit),
            "The corrected split return and the retained kernel surface now sit on one explicit corrected mixed-kernel return target.",
        ),
        sign_base.row(
            "updated_pack_corrected_mixed_kernel_return_machine_readable_now",
            "pass" if machine_readable_now else "reject",
            "updated-pack corrected mixed-kernel return machine-readable now",
            sign_base.truth(machine_readable_now),
            "The corrected mixed-kernel blocker is now restated under the corrected return lane in machine-readable form.",
        ),
        sign_base.row(
            "exact_corrected_mixed_probe_response_kernel_formula_available_now",
            "pass" if exact_corrected_mixed_probe_response_kernel_formula_available_now else "reject",
            "exact corrected mixed probe-response kernel formula available now",
            sign_base.truth(exact_corrected_mixed_probe_response_kernel_formula_available_now),
            "The corrected return branch still does not derive the literal mixed probe-response kernel formula.",
        ),
        sign_base.row(
            "exact_corrected_pure_probe_response_kernel_formula_available_now",
            "pass" if exact_corrected_pure_probe_response_kernel_formula_available_now else "reject",
            "exact corrected pure probe-response kernel formula available now",
            sign_base.truth(exact_corrected_pure_probe_response_kernel_formula_available_now),
            "The pure probe-response kernel remains underived under the corrected return lane as well.",
        ),
        sign_base.row(
            "exact_corrected_kernel_rank_match_available_now",
            "pass" if exact_corrected_kernel_rank_match_available_now else "reject",
            "exact corrected kernel rank match available now",
            sign_base.truth(exact_corrected_kernel_rank_match_available_now),
            "Without literal xi/A formulas, the corrected kernel still lacks a closed rank-matched theorem.",
        ),
        sign_base.row(
            "updated_pack_corrected_vacuum_subtraction_primary_followup_required",
            "pass" if corrected_vacuum_subtraction_primary_followup_required else "reject",
            "updated-pack corrected vacuum-subtraction primary followup required",
            sign_base.truth(corrected_vacuum_subtraction_primary_followup_required),
            "After the corrected mixed-kernel return is restated, subtraction refresh remains the next honest blocker.",
        ),
        sign_base.row(
            "updated_pack_corrected_pack_refresh_secondary_hold_retained",
            "pass" if corrected_pack_refresh_secondary_hold_retained else "reject",
            "updated-pack corrected pack-refresh secondary hold retained",
            sign_base.truth(corrected_pack_refresh_secondary_hold_retained),
            "Pack-refresh sync stays secondary until corrected subtraction closes more honestly.",
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
            "Corrected mixed-kernel return still does not justify reopening extra q-range continuation.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_corrected_mixed_kernel_return_refresh_audit_selected": selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_corrected_probe_split_return_machine_readable_now": corrected_probe_split_return_machine_readable,
        "updated_pack_prior_corrected_mixed_kernel_surface_retained": older_mixed_kernel_surface_retained,
        "updated_pack_corrected_mixed_kernel_return_target_surface_explicit": target_surface_explicit,
        "updated_pack_corrected_mixed_kernel_return_machine_readable_now": machine_readable_now,
        "exact_corrected_mixed_probe_response_kernel_formula_available_now": exact_corrected_mixed_probe_response_kernel_formula_available_now,
        "exact_corrected_pure_probe_response_kernel_formula_available_now": exact_corrected_pure_probe_response_kernel_formula_available_now,
        "exact_corrected_kernel_rank_match_available_now": exact_corrected_kernel_rank_match_available_now,
        "updated_pack_corrected_vacuum_subtraction_primary_followup_required": corrected_vacuum_subtraction_primary_followup_required,
        "updated_pack_corrected_pack_refresh_secondary_hold_retained": corrected_pack_refresh_secondary_hold_retained,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid,
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_corrected_mixed_kernel_gate_vacuum_subtraction_refresh",
        "recommended_next_route_or_none": "8.7.56.3187",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_corrected_vacuum_subtraction_return_refresh_audit",
        "selected_followup_route_or_none": "8.7.56.3191",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.3185",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_split_audit": sign_base.display_path(PRIOR_SPLIT_AUDIT),
                "older_kernel_audit": sign_base.display_path(OLDER_KERNEL_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.3187",
                "followup_route": "8.7.56.3191",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_corrected_mixed_kernel_return_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack corrected mixed-kernel return refresh completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
