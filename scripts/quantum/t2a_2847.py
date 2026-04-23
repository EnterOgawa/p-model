#!/usr/bin/env python3
"""Generate 8.7.56.2847-.2850 mixed probe-response kernel refresh artifacts."""

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
        "8.7.56.2843-2846",
        "updated_pack_probe_split_gate_mixed_kernel_refresh",
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
PRIOR_KERNEL_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2791-2794",
        "updated_pack_exact_mixed_probe_response_kernel_completion_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
STEP_TAG = "8.7.56.2847-2850"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack mixed "
    "probe-response kernel refresh audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_mixed_probe_response_kernel_refresh_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_probe_split_rederivation_audited_mixed_kernel_primary_"
    "hybrid_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "mixed_kernel_refresh_audited_vacuum_subtraction_primary_pack_refresh_"
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


# 関数: mixed-kernel refresh で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the mixed-kernel refresh audit."""
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
        "kernel_selection_rule": (
            "corrected probe split -> mixed/pure probe-response kernel -> vacuum subtraction"
        ),
    }


# 関数: `.2847-.2850` を実行する。

def main() -> None:
    """Execute the updated-pack mixed probe-response kernel refresh audit."""
    for path in (PRIOR_GATE, PRIOR_SPLIT_AUDIT, PRIOR_KERNEL_AUDIT):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_split_summary = sign_base.read_json(PRIOR_SPLIT_AUDIT)["summary"]
    prior_kernel_summary = sign_base.read_json(PRIOR_KERNEL_AUDIT)["summary"]

    selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_mixed_probe_response_kernel_refresh_promoted_next"
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
    single_split_only_explicit = bool(
        prior_split_summary["pure_derivation_single_split_only_explicit"]
    )
    forward_scattering_kernel_explicit = bool(
        prior_kernel_summary["pure_derivation_forward_scattering_kernel_explicit"]
    )
    transverse_projection_explicit = bool(
        prior_kernel_summary["pure_derivation_transverse_projection_surface_explicit"]
    )
    refresh_target_surface = bool(
        selected
        and retry_mode
        and non_surrogate_guard
        and corrected_split_machine_readable
        and single_split_only_explicit
        and forward_scattering_kernel_explicit
        and transverse_projection_explicit
    )
    refresh_machine_readable = bool(refresh_target_surface)
    exact_corrected_mixed_probe_response_kernel_formula_available_now = False
    exact_corrected_pure_probe_response_kernel_formula_available_now = False
    exact_corrected_kernel_rank_match_available_now = False
    mixed_kernel_completion_primary_followup_required = bool(
        refresh_machine_readable
        and (not exact_corrected_mixed_probe_response_kernel_formula_available_now)
    )
    vacuum_subtraction_secondary_followup_required = bool(
        mixed_kernel_completion_primary_followup_required
        and (not exact_corrected_kernel_rank_match_available_now)
    )
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])
    farther_hybrid = False

    rows = [
        sign_base.row(
            "updated_pack_mixed_probe_response_kernel_refresh_audit_selected",
            "pass" if selected else "reject",
            "updated-pack mixed probe-response kernel refresh audit selected",
            sign_base.truth(selected),
            "The corrected probe-split gate already promoted mixed probe-response kernel refresh as the next honest derivation lane.",
        ),
        sign_base.row(
            "updated_pack_corrected_probe_split_rederivation_machine_readable_now",
            "pass" if corrected_split_machine_readable else "reject",
            "updated-pack corrected probe-split rederivation machine-readable now",
            sign_base.truth(corrected_split_machine_readable),
            "The dual-field probe split target is already explicit and therefore can anchor a corrected mixed-kernel refresh.",
        ),
        sign_base.row(
            "pure_derivation_single_split_only_explicit",
            "pass" if single_split_only_explicit else "reject",
            "pure-derivation single split only explicit",
            sign_base.truth(single_split_only_explicit),
            "The note still collapses self fluctuation and external probe into one symbol, which is precisely why the corrected mixed-kernel formula is still absent.",
        ),
        sign_base.row(
            "pure_derivation_forward_scattering_kernel_explicit",
            "pass" if forward_scattering_kernel_explicit else "reject",
            "pure-derivation forward-scattering kernel explicit",
            sign_base.truth(forward_scattering_kernel_explicit),
            "The scattering-side kernel surface is already explicit and remains reusable under the corrected split reset.",
        ),
        sign_base.row(
            "pure_derivation_transverse_projection_surface_explicit",
            "pass" if transverse_projection_explicit else "reject",
            "pure-derivation transverse projection surface explicit",
            sign_base.truth(transverse_projection_explicit),
            "The transverse projector still supplies the observable-side kernel contraction surface.",
        ),
        sign_base.row(
            "updated_pack_mixed_probe_response_kernel_refresh_target_surface_explicit",
            "pass" if refresh_target_surface else "reject",
            "updated-pack mixed probe-response kernel refresh target surface explicit",
            sign_base.truth(refresh_target_surface),
            "The corrected split and the retained scattering kernel surfaces now sit on one explicit refresh target.",
        ),
        sign_base.row(
            "updated_pack_mixed_probe_response_kernel_refresh_machine_readable_now",
            "pass" if refresh_machine_readable else "reject",
            "updated-pack mixed probe-response kernel refresh machine-readable now",
            sign_base.truth(refresh_machine_readable),
            "The mixed-kernel blocker is now restated under the corrected probe split in machine-readable form.",
        ),
        sign_base.row(
            "exact_corrected_mixed_probe_response_kernel_formula_available_now",
            "pass" if exact_corrected_mixed_probe_response_kernel_formula_available_now else "reject",
            "exact corrected mixed probe-response kernel formula available now",
            sign_base.truth(exact_corrected_mixed_probe_response_kernel_formula_available_now),
            "The corrected split target is explicit, but the literal mixed probe-response kernel formula is still absent.",
        ),
        sign_base.row(
            "exact_corrected_pure_probe_response_kernel_formula_available_now",
            "pass" if exact_corrected_pure_probe_response_kernel_formula_available_now else "reject",
            "exact corrected pure probe-response kernel formula available now",
            sign_base.truth(exact_corrected_pure_probe_response_kernel_formula_available_now),
            "The pure probe-response kernel also remains underived after the corrected split reset.",
        ),
        sign_base.row(
            "exact_corrected_kernel_rank_match_available_now",
            "pass" if exact_corrected_kernel_rank_match_available_now else "reject",
            "exact corrected kernel rank match available now",
            sign_base.truth(exact_corrected_kernel_rank_match_available_now),
            "Without a literal xi/A split formula, the scattering kernel still lacks a closed rank-matched canonical theorem.",
        ),
        sign_base.row(
            "updated_pack_corrected_mixed_kernel_completion_primary_followup_required",
            "pass" if mixed_kernel_completion_primary_followup_required else "reject",
            "updated-pack corrected mixed-kernel completion primary followup required",
            sign_base.truth(mixed_kernel_completion_primary_followup_required),
            "The honest next move remains literal corrected mixed-kernel completion, not another reserve-policy cycle.",
        ),
        sign_base.row(
            "updated_pack_vacuum_subtraction_secondary_followup_required",
            "pass" if vacuum_subtraction_secondary_followup_required else "reject",
            "updated-pack vacuum-subtraction secondary followup required",
            sign_base.truth(vacuum_subtraction_secondary_followup_required),
            "Vacuum subtraction stays downstream of the corrected mixed-kernel blocker and therefore remains secondary.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation remains downstream of the corrected probe/current/kernel theorem stack.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid),
            "Extra q-range evidence remains reserve-only because the blocker is still corrected mixed-kernel completion.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_mixed_probe_response_kernel_refresh_audit_selected": selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_corrected_probe_split_rederivation_machine_readable_now": corrected_split_machine_readable,
        "pure_derivation_single_split_only_explicit": single_split_only_explicit,
        "pure_derivation_forward_scattering_kernel_explicit": forward_scattering_kernel_explicit,
        "pure_derivation_transverse_projection_surface_explicit": transverse_projection_explicit,
        "updated_pack_mixed_probe_response_kernel_refresh_target_surface_explicit": refresh_target_surface,
        "updated_pack_mixed_probe_response_kernel_refresh_machine_readable_now": refresh_machine_readable,
        "exact_corrected_mixed_probe_response_kernel_formula_available_now": exact_corrected_mixed_probe_response_kernel_formula_available_now,
        "exact_corrected_pure_probe_response_kernel_formula_available_now": exact_corrected_pure_probe_response_kernel_formula_available_now,
        "exact_corrected_kernel_rank_match_available_now": exact_corrected_kernel_rank_match_available_now,
        "updated_pack_corrected_mixed_kernel_completion_primary_followup_required": mixed_kernel_completion_primary_followup_required,
        "updated_pack_vacuum_subtraction_secondary_followup_required": vacuum_subtraction_secondary_followup_required,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid,
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_mixed_kernel_gate_vacuum_subtraction_refresh",
        "recommended_next_route_or_none": "8.7.56.2851",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_corrected_vacuum_subtraction_refresh_audit",
        "selected_followup_route_or_none": "8.7.56.2855",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.2849",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_split_audit": sign_base.display_path(PRIOR_SPLIT_AUDIT),
                "prior_kernel_audit": sign_base.display_path(PRIOR_KERNEL_AUDIT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.2851",
                "followup_route": "8.7.56.2855",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_mixed_kernel_refresh_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack mixed-kernel refresh completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
