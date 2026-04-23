#!/usr/bin/env python3
"""Generate 8.7.56.2839-.2842 corrected probe-split rederivation artifacts."""

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
        "8.7.56.2835-2838",
        "updated_pack_pack_refresh_gate_probe_split_reset",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PURE_DERIVATION_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2767-2770",
        "updated_pack_pure_derivation_probe_split_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
CURRENT_VERTEX_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2775-2778",
        "updated_pack_exact_external_probe_current_vertex_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
STEP_TAG = "8.7.56.2839-2842"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack corrected "
    "probe-split rederivation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_corrected_probe_split_rederivation_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "pack_refresh_cycle_repeat_detected_probe_split_primary_kernel_secondary_"
    "hybrid_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_probe_split_rederivation_audited_mixed_kernel_primary_"
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


# 関数: corrected probe-split rederivation で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the corrected probe-split rederivation audit."""
    return {
        "dual_field_split": "P_mu(x) = Q_mu(x) + xi_mu(x),   A_mu(x): external probe",
        "background_stationarity": (
            "S_bg^(1)[Q;xi] = int d^4x xi_mu (delta S / delta P_mu)|_(P=Q) = 0"
        ),
        "external_probe_current": (
            "J_ext^mu[Q](x) := delta S_frozen[Q;A] / delta A_mu(x) |_(A=0)"
        ),
        "mixed_probe_kernel": (
            "V^{mu nu}[Q](x,y) := delta^2 S_frozen / (delta xi_mu(x) delta A_nu(y))"
            " |_(Q,A=0)"
        ),
        "ordering": (
            "corrected probe split -> mixed probe-response kernel -> vacuum subtraction"
        ),
    }


# 関数: `.2839-.2842` を実行する。

def main() -> None:
    """Execute the updated-pack corrected probe-split rederivation audit."""
    for path in (PRIOR_GATE, PURE_DERIVATION_AUDIT, CURRENT_VERTEX_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pure_derivation_summary = sign_base.read_json(PURE_DERIVATION_AUDIT)["summary"]
    current_vertex_summary = sign_base.read_json(CURRENT_VERTEX_AUDIT)["summary"]

    selected = bool(
        prior_gate_summary["gate_b_updated_pack_corrected_probe_split_rederivation_promoted_next"]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    pure_derivation_external_probe_split_required = bool(
        pure_derivation_summary["pure_derivation_external_probe_split_required"]
    )
    pure_derivation_single_split_only_explicit = bool(
        current_vertex_summary["pure_derivation_single_split_only_explicit"]
    )
    pure_derivation_on_shell_zero_explicit = bool(
        current_vertex_summary["pure_derivation_on_shell_zero_explicit"]
    )
    rank_matched_dual_field_probe_split_definition_explicit = bool(
        pure_derivation_external_probe_split_required
        and pure_derivation_single_split_only_explicit
        and pure_derivation_on_shell_zero_explicit
    )
    background_stationarity_only_for_self_fluctuation_explicit = bool(
        pure_derivation_on_shell_zero_explicit
    )
    corrected_probe_split_target_surface_explicit = bool(
        selected
        and retry_mode
        and non_surrogate_guard
        and rank_matched_dual_field_probe_split_definition_explicit
        and background_stationarity_only_for_self_fluctuation_explicit
    )
    updated_pack_corrected_probe_split_rederivation_machine_readable_now = bool(
        corrected_probe_split_target_surface_explicit
    )
    exact_corrected_probe_split_formula_available_now = False
    exact_external_probe_current_vertex_formula_available_now = False
    updated_pack_mixed_probe_response_kernel_primary_followup_required = bool(
        updated_pack_corrected_probe_split_rederivation_machine_readable_now
        and (not exact_corrected_probe_split_formula_available_now)
    )
    updated_pack_vacuum_subtraction_secondary_hold_retained = bool(
        updated_pack_mixed_probe_response_kernel_primary_followup_required
    )
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])
    farther_hybrid = False

    rows = [
        sign_base.row(
            "updated_pack_corrected_probe_split_rederivation_audit_selected",
            "pass" if selected else "reject",
            "updated-pack corrected probe-split rederivation audit selected",
            sign_base.truth(selected),
            "Once the pack-refresh cycle is declared exhausted, the honest next move is to restate the corrected dual-field split explicitly.",
        ),
        sign_base.row(
            "pure_derivation_external_probe_split_required",
            "pass" if pure_derivation_external_probe_split_required else "reject",
            "pure-derivation external-probe split required",
            sign_base.truth(pure_derivation_external_probe_split_required),
            "The pure-derivation route already showed that self fluctuation and external probe must be split to keep rank-matched observables honest.",
        ),
        sign_base.row(
            "pure_derivation_single_split_only_explicit",
            "pass" if pure_derivation_single_split_only_explicit else "reject",
            "pure-derivation single split only explicit",
            sign_base.truth(pure_derivation_single_split_only_explicit),
            "The note still collapses self fluctuation and external probe into one symbol, which is precisely the object being corrected here.",
        ),
        sign_base.row(
            "rank_matched_dual_field_probe_split_definition_explicit",
            "pass" if rank_matched_dual_field_probe_split_definition_explicit else "reject",
            "rank-matched dual-field probe split definition explicit",
            sign_base.truth(rank_matched_dual_field_probe_split_definition_explicit),
            "The corrected target is explicit: one field for self fluctuation and one separate external probe field.",
        ),
        sign_base.row(
            "background_stationarity_only_for_self_fluctuation_explicit",
            "pass" if background_stationarity_only_for_self_fluctuation_explicit else "reject",
            "background stationarity only for self fluctuation explicit",
            sign_base.truth(background_stationarity_only_for_self_fluctuation_explicit),
            "The on-shell vanishing first variation applies to the self fluctuation branch and must not be misread as the external probe current vanishing automatically.",
        ),
        sign_base.row(
            "updated_pack_corrected_probe_split_target_surface_explicit",
            "pass" if corrected_probe_split_target_surface_explicit else "reject",
            "updated-pack corrected probe-split target surface explicit",
            sign_base.truth(corrected_probe_split_target_surface_explicit),
            "The corrected dual-field split now sits on one explicit target surface for renewed computation-side derivation.",
        ),
        sign_base.row(
            "updated_pack_corrected_probe_split_rederivation_machine_readable_now",
            "pass" if updated_pack_corrected_probe_split_rederivation_machine_readable_now else "reject",
            "updated-pack corrected probe-split rederivation machine-readable now",
            sign_base.truth(updated_pack_corrected_probe_split_rederivation_machine_readable_now),
            "The reset from cycle bookkeeping to corrected split rederivation is now explicit and machine-readable.",
        ),
        sign_base.row(
            "exact_corrected_probe_split_formula_available_now",
            "pass" if exact_corrected_probe_split_formula_available_now else "reject",
            "exact corrected probe-split formula available now",
            sign_base.truth(exact_corrected_probe_split_formula_available_now),
            "The branch localizes the dual-field split target but does not yet derive the exact canonical formula.",
        ),
        sign_base.row(
            "exact_external_probe_current_vertex_formula_available_now",
            "pass" if exact_external_probe_current_vertex_formula_available_now else "reject",
            "exact external-probe current-vertex formula available now",
            sign_base.truth(exact_external_probe_current_vertex_formula_available_now),
            "Current-vertex completion still remains downstream of the corrected split itself.",
        ),
        sign_base.row(
            "updated_pack_mixed_probe_response_kernel_primary_followup_required",
            "pass" if updated_pack_mixed_probe_response_kernel_primary_followup_required else "reject",
            "updated-pack mixed probe-response kernel primary followup required",
            sign_base.truth(updated_pack_mixed_probe_response_kernel_primary_followup_required),
            "After the corrected split is restated, the next honest move is to refresh mixed kernel completion under that split.",
        ),
        sign_base.row(
            "updated_pack_vacuum_subtraction_secondary_hold_retained",
            "pass" if updated_pack_vacuum_subtraction_secondary_hold_retained else "reject",
            "updated-pack vacuum-subtraction secondary hold retained",
            sign_base.truth(updated_pack_vacuum_subtraction_secondary_hold_retained),
            "Vacuum subtraction remains downstream of the corrected split and kernel objects.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation remains downstream of the corrected split and kernel theorem stack.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid),
            "Cycle reset does not justify reopening extra q-range continuation.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(current_vertex_summary["retained_scalar_residual_rel"]),
        "updated_pack_corrected_probe_split_rederivation_audit_selected": selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "pure_derivation_external_probe_split_required": pure_derivation_external_probe_split_required,
        "pure_derivation_single_split_only_explicit": pure_derivation_single_split_only_explicit,
        "rank_matched_dual_field_probe_split_definition_explicit": rank_matched_dual_field_probe_split_definition_explicit,
        "background_stationarity_only_for_self_fluctuation_explicit": background_stationarity_only_for_self_fluctuation_explicit,
        "updated_pack_corrected_probe_split_target_surface_explicit": corrected_probe_split_target_surface_explicit,
        "updated_pack_corrected_probe_split_rederivation_machine_readable_now": updated_pack_corrected_probe_split_rederivation_machine_readable_now,
        "exact_corrected_probe_split_formula_available_now": exact_corrected_probe_split_formula_available_now,
        "exact_external_probe_current_vertex_formula_available_now": exact_external_probe_current_vertex_formula_available_now,
        "updated_pack_mixed_probe_response_kernel_primary_followup_required": updated_pack_mixed_probe_response_kernel_primary_followup_required,
        "updated_pack_vacuum_subtraction_secondary_hold_retained": updated_pack_vacuum_subtraction_secondary_hold_retained,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid,
        "updated_pack_corrected_probe_split_breakthrough_passed_now": False,
        "recommended_next_route_or_none": "8.7.56.2843",
        "selected_followup_route_or_none": "8.7.56.2847",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.2841",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "pure_derivation_audit": sign_base.display_path(PURE_DERIVATION_AUDIT),
                "current_vertex_audit": sign_base.display_path(CURRENT_VERTEX_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.2843",
                "followup_route": "8.7.56.2847",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_probe_split_rederivation_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack corrected probe-split audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
