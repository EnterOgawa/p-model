#!/usr/bin/env python3
"""Generate 8.7.56.5147-.5150 blind-vector direct-computation gate artifacts."""

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
        "8.7.56.5143-5146",
        "updated_pack_blind_vector_direct_computation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5147-5150"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack blind-vector "
    "direct computation gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_blind_vector_direct_computation_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_direct_computation_contract_derived_numeric_evaluation_"
    "primary_pack_refresh_secondary_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_direct_computation_audited_numeric_evaluation_primary_"
    "hybrid_reserve_secondary_next"
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


# 関数: gate で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the blind-vector direct-computation gate."""
    return {
        "gate_a": (
            "Gate A = selected-extension blind-vector direct-computation contract "
            "explicit and machine-readable"
        ),
        "gate_b": "Gate B = blind-vector numeric evaluation promoted next",
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.5147-.5150` を実行する。

def main() -> None:
    """Execute the blind-vector direct-computation decision gate."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    gate_a_updated_pack_blind_vector_direct_computation_contract_available_now = bool(
        prior_summary[
            "exact_blind_vector_selected_extension_checkpoint_contract_available_now"
        ]
        and prior_summary[
            "exact_blind_vector_selected_extension_residual_origin_discriminator_formula_available_now"
        ]
        and prior_summary["direct_blind_vector_computation_primary_admissible_now"]
    )
    gate_b_updated_pack_blind_vector_numeric_evaluation_promoted_next = bool(
        prior_summary["updated_pack_blind_vector_numeric_evaluation_followup_required"]
        and prior_summary["direct_blind_vector_computation_primary_admissible_now"]
    )
    gate_c_farther_hybrid_continuation_reopen_required_now = bool(
        prior_summary["farther_hybrid_continuation_reopen_required_now"]
    )
    exact_concrete_selected_extension_available_now = True
    blind_vector_observable_gate_still_blocked = bool(
        prior_summary["blind_vector_observable_gate_still_blocked"]
    )

    rows = [
        sign_base.row(
            "gate_a_updated_pack_blind_vector_direct_computation_contract_available_now",
            "pass"
            if gate_a_updated_pack_blind_vector_direct_computation_contract_available_now
            else "reject",
            "Gate A updated-pack blind-vector direct computation contract available now",
            sign_base.truth(
                gate_a_updated_pack_blind_vector_direct_computation_contract_available_now
            ),
            "The selected extension now carries one explicit and machine-readable blind-vector computation contract.",
        ),
        sign_base.row(
            "gate_b_updated_pack_blind_vector_numeric_evaluation_promoted_next",
            "pass"
            if gate_b_updated_pack_blind_vector_numeric_evaluation_promoted_next
            else "reject",
            "Gate B updated-pack blind-vector numeric evaluation promoted next",
            sign_base.truth(
                gate_b_updated_pack_blind_vector_numeric_evaluation_promoted_next
            ),
            "Once the computation contract is explicit, the honest next blocker is the numeric evaluation itself rather than another theorem-family refinement.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "pass" if gate_c_farther_hybrid_continuation_reopen_required_now else "reject",
            "Gate C farther hybrid continuation reopen required now",
            sign_base.truth(gate_c_farther_hybrid_continuation_reopen_required_now),
            "Extra q-range evidence remains reserve-only because the next live task is still blind-vector numeric evaluation on the selected extension.",
        ),
        sign_base.row(
            "exact_concrete_selected_extension_available_now",
            "pass" if exact_concrete_selected_extension_available_now else "reject",
            "exact concrete selected extension available now",
            sign_base.truth(exact_concrete_selected_extension_available_now),
            "The computation gate inherits the already closed selected extension Sigma_*^(pilot-HS) and does not reopen selector ambiguity.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_vector_observable_gate_still_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_vector_observable_gate_still_blocked),
            "False means the remaining work is numeric evaluation and residual-origin judgement, not a selector-side blocker.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_theory_over_m0": float(prior_summary["q_theory_over_m0"]),
        "retained_scalar_alpha_at_q_theory": float(
            prior_summary["retained_scalar_alpha_at_q_theory"]
        ),
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "gate_a_updated_pack_blind_vector_direct_computation_contract_available_now": gate_a_updated_pack_blind_vector_direct_computation_contract_available_now,
        "gate_b_updated_pack_blind_vector_numeric_evaluation_promoted_next": gate_b_updated_pack_blind_vector_numeric_evaluation_promoted_next,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c_farther_hybrid_continuation_reopen_required_now,
        "exact_concrete_selected_extension_available_now": exact_concrete_selected_extension_available_now,
        "blind_vector_observable_gate_still_blocked": blind_vector_observable_gate_still_blocked,
        "selected_primary_completion_lane": "updated_pack_blind_vector_numeric_evaluation_audit",
        "selected_secondary_completion_lane": "updated_pack_residual_origin_refresh_after_selected_extension_blind_vector",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_numeric_evaluation_audit",
        "recommended_next_route_or_none": "8.7.56.5151",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_numeric_evaluation_gate",
        "selected_followup_route_or_none": "8.7.56.5155",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5149",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_audit": sign_base.display_path(PRIOR_GATE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5151",
                "followup_route": "8.7.56.5155",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_blind_vector_direct_computation_gate_declared",
            "branch_completed": True,
            "direct_numeric_evaluation_ready_now": gate_b_updated_pack_blind_vector_numeric_evaluation_promoted_next,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} blind-vector direct computation gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
