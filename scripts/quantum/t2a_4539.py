#!/usr/bin/env python3
"""Generate 8.7.56.4539-.4542 corrected vacuum theorem gate artifacts."""

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
        "8.7.56.4535-4538",
        "updated_pack_corrected_vacuum_subtraction_theorem_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
STEP_TAG = "8.7.56.4539-4542"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack corrected "
    "vacuum theorem gate / vacuum-state selector refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_corrected_vacuum_theorem_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_vacuum_nonuniqueness_and_subtracted_rank_preservation_theorem_"
    "derived_vacuum_state_selector_primary_pack_refresh_secondary_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "corrected_vacuum_nonuniqueness_and_subtracted_rank_preservation_theorem_"
    "derived_vacuum_state_selector_primary_pack_refresh_secondary_hybrid_"
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
    """Return formulas used in the corrected vacuum theorem gate."""
    return {
        "gate_a": "Gate A = exact corrected vacuum-subtraction rule available now",
        "gate_b": "Gate B = corrected vacuum-state selector theorem promoted next",
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.4539-.4542` を実行する。

def main() -> None:
    """Execute the corrected vacuum theorem gate."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(prior_summary["exact_corrected_vacuum_subtraction_rule_available_now"])
    gate_b = bool(
        prior_summary["updated_pack_corrected_vacuum_state_selector_primary_followup_required"]
    )
    gate_c = False
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(prior_summary["failure_matrix_non_surrogate_guard_preserved"])
    blind_blocked = bool(prior_summary["blind_vector_observable_gate_still_blocked"])
    old_retry = False
    pack_update_required = bool(gate_b)
    exact_corrected_subtracted_observable_rank_match_available_now = bool(
        prior_summary["exact_corrected_subtracted_observable_rank_match_available_now"]
    )
    exact_corrected_vacuum_state_nonuniqueness_theorem_available_now = bool(
        prior_summary["exact_corrected_vacuum_state_nonuniqueness_theorem_available_now"]
    )

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_corrected_vacuum_subtraction_rule_available_now",
            "pass" if gate_a else "reject",
            "Gate A exact corrected vacuum-subtraction rule available now",
            sign_base.truth(gate_a),
            "Rank-preserving subtraction is now explicit, but the canonical rule is still blocked by the missing vacuum-state selector theorem.",
        ),
        sign_base.row(
            "gate_b_updated_pack_corrected_vacuum_state_selector_theorem_promoted_next",
            "pass" if gate_b else "reject",
            "Gate B corrected vacuum-state selector theorem promoted next",
            sign_base.truth(gate_b),
            "Because the vacuum manifold is nonunique under the written action, the honest next blocker is the theorem that selects the subtraction vacuum rather than another pack-refresh loop.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "pass" if gate_c else "reject",
            "Gate C farther hybrid continuation reopen required now",
            sign_base.truth(gate_c),
            "Extra q-range evidence remains reserve-only because the blocker is theorem-side vacuum selection, not continuation range.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "This gate follows a real theorem closure and does not count a same-tag reserve loop as progress.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Promoting vacuum-state selection does not reopen the exhausted density/proxy/eigenvalue family.",
        ),
        sign_base.row(
            "exact_corrected_subtracted_observable_rank_match_available_now",
            "pass" if exact_corrected_subtracted_observable_rank_match_available_now else "reject",
            "exact corrected subtracted observable rank match available now",
            sign_base.truth(exact_corrected_subtracted_observable_rank_match_available_now),
            "The subtraction lane no longer fails on rank mismatch: the unresolved piece is vacuum selection, not observable rank.",
        ),
        sign_base.row(
            "exact_corrected_vacuum_state_nonuniqueness_theorem_available_now",
            "pass" if exact_corrected_vacuum_state_nonuniqueness_theorem_available_now else "reject",
            "exact corrected vacuum-state nonuniqueness theorem available now",
            sign_base.truth(exact_corrected_vacuum_state_nonuniqueness_theorem_available_now),
            "The written Mexican-hat vacuum manifold and the note's temporal assignment now conflict explicitly, so nonuniqueness is theorem-level rather than a vague caveat.",
        ),
        sign_base.row(
            "blind_vector_observable_gate_still_blocked",
            "pass" if blind_blocked else "reject",
            "blind-vector observable gate still blocked",
            sign_base.truth(blind_blocked),
            "Blind-vector direct computation still waits on canonical vacuum selection and downstream reserve closeout.",
        ),
        sign_base.row(
            "old_density_proxy_eigenvalue_retry_admissible_now",
            "pass" if old_retry else "reject",
            "old density/proxy/eigenvalue retry admissible now",
            sign_base.truth(old_retry),
            "The gate keeps the exhausted surrogate retry family closed.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required else "reject",
            "updated-pack substantive pack update required now",
            sign_base.truth(pack_update_required),
            "A real theorem object closed here, but the remaining blocker is now vacuum-state selection rather than another reserve bookkeeping sync.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "gate_a_updated_pack_exact_corrected_vacuum_subtraction_rule_available_now": gate_a,
        "gate_b_updated_pack_corrected_vacuum_state_selector_theorem_promoted_next": gate_b,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "exact_corrected_subtracted_observable_rank_match_available_now": exact_corrected_subtracted_observable_rank_match_available_now,
        "exact_corrected_vacuum_state_nonuniqueness_theorem_available_now": exact_corrected_vacuum_state_nonuniqueness_theorem_available_now,
        "blind_vector_observable_gate_still_blocked": blind_blocked,
        "old_density_proxy_eigenvalue_retry_admissible_now": old_retry,
        "pack_update_required_now": pack_update_required,
        "selected_primary_completion_lane": "updated_pack_corrected_vacuum_state_selector_theorem_audit",
        "selected_secondary_completion_lane": "updated_pack_corrected_pack_refresh_return_sync",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_corrected_vacuum_state_selector_theorem_audit",
        "recommended_next_route_or_none": "8.7.56.4543",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_corrected_vacuum_state_selector_gate_pack_refresh_sync",
        "selected_followup_route_or_none": "8.7.56.4547",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.4541",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.4543",
                "followup_route": "8.7.56.4547",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_updated_pack_corrected_vacuum_theorem_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulas": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} updated-pack corrected vacuum theorem gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
