#!/usr/bin/env python3
"""Generate 8.7.56.2963-.2966 corrected mixed-kernel gate / vacuum refresh artifacts."""

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
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2959-2962",
        "updated_pack_corrected_mixed_kernel_return_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
STEP_TAG = "8.7.56.2963-2966"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack corrected "
    "mixed-kernel gate / vacuum-subtraction refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_corrected_mixed_kernel_gate_vacuum_subtraction_refresh",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_mixed_kernel_return_audited_vacuum_subtraction_primary_"
    "pack_refresh_secondary_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_mixed_kernel_return_audited_vacuum_subtraction_primary_"
    "pack_refresh_secondary_hybrid_reserve_next"
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
    """Return formulas used in the corrected mixed-kernel gate."""
    return {
        "gate_a": "Gate A = exact corrected mixed probe-response kernel available now",
        "gate_b": "Gate B = corrected vacuum-subtraction refresh promoted next",
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.2963-.2966` を実行する。

def main() -> None:
    """Execute the updated-pack corrected mixed-kernel gate / vacuum refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary["exact_corrected_mixed_probe_response_kernel_formula_available_now"]
        and prior_summary["exact_corrected_pure_probe_response_kernel_formula_available_now"]
        and prior_summary["exact_corrected_kernel_rank_match_available_now"]
    )
    gate_b = bool(
        prior_summary["updated_pack_corrected_vacuum_subtraction_primary_followup_required"]
        and (not gate_a)
    )
    gate_c = False
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(prior_summary["failure_matrix_non_surrogate_guard_preserved"])
    blind_blocked = bool(prior_summary["blind_vector_observable_gate_still_blocked"])
    old_retry = False
    pack_update_required = bool(gate_b)

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_corrected_mixed_probe_response_kernel_available_now",
            "pass" if gate_a else "reject",
            "Gate A exact corrected mixed probe-response kernel available now",
            sign_base.truth(gate_a),
            "The corrected return branch still does not produce literal mixed/pure kernel formulas or a closed rank-matched theorem.",
        ),
        sign_base.row(
            "gate_b_updated_pack_corrected_vacuum_subtraction_refresh_promoted_next",
            "pass" if gate_b else "reject",
            "Gate B corrected vacuum-subtraction refresh promoted next",
            sign_base.truth(gate_b),
            "Because corrected mixed-kernel completion remains absent, the honest next move is corrected vacuum-subtraction refresh under that return ordering.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "pass" if gate_c else "reject",
            "Gate C farther hybrid continuation reopen required now",
            sign_base.truth(gate_c),
            "Extra q-range evidence remains reserve-only because the blocker is still theorem-side corrected kernel/subtraction closure.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The corrected gate keeps exhausted density/proxy/eigenvalue retries closed while subtraction is promoted.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation remains downstream of the unresolved corrected split, corrected kernel, and corrected subtraction theorem stack.",
        ),
        sign_base.row(
            "old_density_proxy_eigenvalue_retry_admissible_now",
            "pass" if old_retry else "reject",
            "old density/proxy/eigenvalue retry admissible now",
            sign_base.truth(old_retry),
            "The gate does not reopen any same-level surrogate retry family.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required else "reject",
            "updated-pack substantive pack update required now",
            sign_base.truth(pack_update_required),
            "The corrected theorem chain still needs subtraction-side work because the corrected mixed-kernel return is not yet literal.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "gate_a_updated_pack_exact_corrected_mixed_probe_response_kernel_available_now": gate_a,
        "gate_b_updated_pack_corrected_vacuum_subtraction_refresh_promoted_next": gate_b,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "old_density_proxy_eigenvalue_retry_admissible_now": old_retry,
        "pack_update_required_now": pack_update_required,
        "selected_primary_completion_lane": "updated_pack_corrected_vacuum_subtraction_return_refresh_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_vacuum_subtraction_gate_pack_refresh_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_corrected_vacuum_subtraction_return_refresh_audit",
        "recommended_next_route_or_none": "8.7.56.2967",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_corrected_vacuum_subtraction_gate_pack_refresh_sync",
        "selected_followup_route_or_none": "8.7.56.2971",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.2965",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.2967",
                "followup_route": "8.7.56.2971",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_corrected_mixed_kernel_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack corrected mixed-kernel gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
