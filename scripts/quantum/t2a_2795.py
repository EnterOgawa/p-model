#!/usr/bin/env python3
"""Generate 8.7.56.2795-.2798 mixed probe-response gate / vacuum-subtraction refresh artifacts."""

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
        "8.7.56.2791-2794",
        "updated_pack_exact_mixed_probe_response_kernel_completion_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
STEP_TAG = "8.7.56.2795-2798"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack mixed probe-"
    "response gate / vacuum-subtraction refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_mixed_probe_response_vacuum_subtraction_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "exact_mixed_probe_response_completion_audited_vacuum_subtraction_primary_"
    "hybrid_reserve_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "exact_kernel_completion_audited_vacuum_subtraction_primary_hybrid_"
    "reserve_next"
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
    """Return formulas used in the gate."""
    return {
        "gate_a": "Gate A = exact mixed probe-response completion available now",
        "gate_b": "Gate B = vacuum-subtraction refresh promoted next",
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.2795-.2798` を実行する。

def main() -> None:
    """Execute the updated-pack mixed probe-response gate / vacuum-subtraction refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary["exact_mixed_probe_response_kernel_formula_available_now"]
        and prior_summary["exact_pure_probe_response_kernel_formula_available_now"]
        and prior_summary["exact_kernel_rank_match_available_now"]
    )
    gate_b = bool(
        prior_summary["updated_pack_vacuum_subtraction_primary_followup_required"]
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
            "gate_a_updated_pack_exact_mixed_probe_response_completion_available_now",
            "pass" if gate_a else "reject",
            "Gate A exact mixed probe-response completion available now",
            sign_base.truth(gate_a),
            "The corrected split and exact mixed/pure kernel formulas are still absent, so exact completion is not yet available.",
        ),
        sign_base.row(
            "gate_b_updated_pack_vacuum_subtraction_refresh_promoted_next",
            "pass" if gate_b else "reject",
            "Gate B vacuum-subtraction refresh promoted next",
            sign_base.truth(gate_b),
            "Because the kernel-completion gap is fully localized but unresolved, the next honest refresh is vacuum subtraction under that corrected ordering.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "pass" if gate_c else "reject",
            "Gate C farther hybrid continuation reopen required now",
            sign_base.truth(gate_c),
            "Extra q-range evidence remains reserve-only because the blocker is still theorem-side kernel/subtraction completion.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation remains downstream of the unresolved probe-kernel theorem stack.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required else "reject",
            "updated-pack substantive pack update required now",
            sign_base.truth(pack_update_required),
            "The honest next move remains pack-update theorem work, now centered on vacuum-subtraction refresh under the corrected kernel ordering.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "gate_a_updated_pack_exact_mixed_probe_response_completion_available_now": gate_a,
        "gate_b_updated_pack_vacuum_subtraction_refresh_promoted_next": gate_b,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "old_density_proxy_eigenvalue_retry_admissible_now": old_retry,
        "pack_update_required_now": pack_update_required,
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_vacuum_subtraction_refresh_audit",
        "recommended_next_route_or_none": "8.7.56.2799",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_vacuum_subtraction_gate_pack_refresh_sync",
        "selected_followup_route_or_none": "8.7.56.2803",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.2797",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.2799",
                "followup_route": "8.7.56.2803",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_mixed_probe_response_vacuum_subtraction_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack mixed probe-response gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
