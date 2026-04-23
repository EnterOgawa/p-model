#!/usr/bin/env python3
"""Generate 8.7.56.2791-.2794 exact mixed probe-response kernel completion audit artifacts."""

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
        "8.7.56.2787-2790",
        "updated_pack_mixed_probe_response_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2783-2786",
        "updated_pack_exact_mixed_probe_response_kernel_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PURE_DERIVATION_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_pure_derivation_20260330.md"
)
STEP_TAG = "8.7.56.2791-2794"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack exact mixed "
    "probe-response kernel completion audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_exact_mixed_probe_response_kernel_completion_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "mixed_probe_response_audited_exact_kernel_primary_vacuum_subtraction_"
    "secondary_hybrid_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "exact_mixed_probe_response_completion_audited_vacuum_subtraction_primary_"
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


# 関数: kernel-completion audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the kernel-completion audit."""
    return {
        "kernel_scattering_read": (
            "\\mathcal{M}=\\epsilon^{*\\mu}\\tilde{\\mathcal{K}}_{\\mu\\nu}\\epsilon'^\\nu"
        ),
        "transverse_projector": "\\Pi_T^{ij}=\\delta^{ij}-\\hat{k}^i\\hat{k}^j",
        "rank_match_requirement": "corrected xi/A split -> exact mixed/pure kernel",
    }


# 関数: `.2791-.2794` を実行する。

def main() -> None:
    """Execute the updated-pack exact mixed probe-response kernel completion audit."""
    for path in (PRIOR_GATE, PRIOR_AUDIT, PURE_DERIVATION_NOTE):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    note_text = sign_base.read_text(PURE_DERIVATION_NOTE)

    selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_exact_mixed_probe_response_kernel_completion_promoted_next"
        ]
        and prior_gate_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    mixed_kernel_surface = bool(
        prior_audit_summary["updated_pack_mixed_probe_response_kernel_machine_readable_now"]
    )
    second_variation = bool(
        sign_base.hit(note_text, "S^{(2)}") is not None
        or sign_base.hit(note_text, "S^(2)") is not None
    )
    scattering_kernel = bool(
        sign_base.hit(note_text, "photon scattering amplitude") is not None
        and sign_base.hit(note_text, "\\tilde{\\mathcal{K}}_{\\mu\\nu}") is not None
    )
    transverse_projection = bool(
        sign_base.hit(note_text, "\\Pi^{ij}_T") is not None
        or sign_base.hit(note_text, "\\Pi_T^{ij}") is not None
    )
    completion_surface = bool(
        selected
        and retry_mode
        and non_surrogate_guard
        and mixed_kernel_surface
        and second_variation
        and scattering_kernel
        and transverse_projection
    )
    corrected_probe_split_symbol_available_now = False
    exact_mixed_probe_response_kernel_formula_available_now = False
    exact_pure_probe_response_kernel_formula_available_now = False
    exact_kernel_rank_match_available_now = False
    exact_vacuum_subtraction_rule_available_now = False
    fully_localized = bool(
        completion_surface
        and (not corrected_probe_split_symbol_available_now)
        and (not exact_kernel_rank_match_available_now)
    )
    vacuum_followup = bool(fully_localized and (not exact_vacuum_subtraction_rule_available_now))
    breakthrough = False
    blind_blocked = bool(prior_gate_summary["blind_vector_observable_gate_still_blocked"])
    farther_hybrid = False

    rows = [
        sign_base.row(
            "updated_pack_exact_mixed_probe_response_kernel_completion_audit_selected",
            "pass" if selected else "reject",
            "updated-pack exact mixed probe-response kernel completion audit selected",
            sign_base.truth(selected),
            "The mixed probe-response gate already promoted exact kernel completion as the next honest derivation lane.",
        ),
        sign_base.row(
            "updated_pack_exact_mixed_probe_response_completion_machine_readable_now",
            "pass" if completion_surface else "reject",
            "updated-pack exact mixed probe-response completion machine-readable now",
            sign_base.truth(completion_surface),
            "The branch now localizes the honest completion target: corrected split plus literal mixed/pure kernel formulas.",
        ),
        sign_base.row(
            "corrected_probe_split_symbol_available_now",
            "pass" if corrected_probe_split_symbol_available_now else "reject",
            "corrected probe split symbol available now",
            sign_base.truth(corrected_probe_split_symbol_available_now),
            "The note still uses one a_mu, so the literal xi/A split needed for rank match is not yet written down.",
        ),
        sign_base.row(
            "exact_mixed_probe_response_kernel_formula_available_now",
            "pass" if exact_mixed_probe_response_kernel_formula_available_now else "reject",
            "exact mixed probe-response kernel formula available now",
            sign_base.truth(exact_mixed_probe_response_kernel_formula_available_now),
            "The exact mixed probe-response kernel is still underived once the self/probe split is corrected.",
        ),
        sign_base.row(
            "exact_pure_probe_response_kernel_formula_available_now",
            "pass" if exact_pure_probe_response_kernel_formula_available_now else "reject",
            "exact pure probe-response kernel formula available now",
            sign_base.truth(exact_pure_probe_response_kernel_formula_available_now),
            "The exact pure probe-response kernel also remains unavailable in this branch.",
        ),
        sign_base.row(
            "exact_kernel_rank_match_available_now",
            "pass" if exact_kernel_rank_match_available_now else "reject",
            "exact kernel rank match available now",
            sign_base.truth(exact_kernel_rank_match_available_now),
            "Without the corrected split, the scattering kernel is not yet the canonical rank-matched object.",
        ),
        sign_base.row(
            "exact_vacuum_subtraction_rule_available_now",
            "pass" if exact_vacuum_subtraction_rule_available_now else "reject",
            "exact vacuum subtraction rule available now",
            sign_base.truth(exact_vacuum_subtraction_rule_available_now),
            "Vacuum subtraction remains downstream of the corrected kernel selection and cannot honestly close here.",
        ),
        sign_base.row(
            "updated_pack_vacuum_subtraction_primary_followup_required",
            "pass" if vacuum_followup else "reject",
            "updated-pack vacuum subtraction primary followup required",
            sign_base.truth(vacuum_followup),
            "Once the kernel-completion gap is fully localized, the honest next followup is vacuum subtraction under that corrected ordering.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation remains downstream of the unresolved external-probe kernel theorem stack.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid),
            "Extra q-range evidence remains reserve-only because the blocker is still the exact mixed/pure probe-response kernel completion.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_gate_summary["retained_scalar_residual_rel"]),
        "updated_pack_exact_mixed_probe_response_kernel_completion_audit_selected": selected,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "updated_pack_mixed_probe_response_kernel_machine_readable_now": mixed_kernel_surface,
        "pure_derivation_second_variation_surface_explicit": second_variation,
        "pure_derivation_forward_scattering_kernel_explicit": scattering_kernel,
        "pure_derivation_transverse_projection_surface_explicit": transverse_projection,
        "updated_pack_exact_mixed_probe_response_completion_machine_readable_now": completion_surface,
        "corrected_probe_split_symbol_available_now": corrected_probe_split_symbol_available_now,
        "exact_mixed_probe_response_kernel_formula_available_now": exact_mixed_probe_response_kernel_formula_available_now,
        "exact_pure_probe_response_kernel_formula_available_now": exact_pure_probe_response_kernel_formula_available_now,
        "exact_kernel_rank_match_available_now": exact_kernel_rank_match_available_now,
        "exact_vacuum_subtraction_rule_available_now": exact_vacuum_subtraction_rule_available_now,
        "updated_pack_exact_mixed_probe_response_completion_fully_localized_now": fully_localized,
        "updated_pack_vacuum_subtraction_primary_followup_required": vacuum_followup,
        "updated_pack_exact_mixed_probe_response_breakthrough_passed_now": breakthrough,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid,
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_mixed_probe_response_gate_vacuum_subtraction_refresh",
        "recommended_next_route_or_none": "8.7.56.2795",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_vacuum_subtraction_refresh_audit",
        "selected_followup_route_or_none": "8.7.56.2799",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.2793",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "pure_derivation_note": sign_base.display_path(PURE_DERIVATION_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.2795",
                "followup_route": "8.7.56.2799",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_exact_mixed_probe_response_completion_declared",
            "branch_completed": True,
            "breakthrough_passed_now": breakthrough,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack exact mixed probe-response kernel completion audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
