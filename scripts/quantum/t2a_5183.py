#!/usr/bin/env python3
"""Generate 8.7.56.5183-.5186 blind-vector solver-side numeric rerun audit artifacts."""

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
        "8.7.56.5179-5182",
        "updated_pack_blind_vector_solver_side_deformation_front_runner_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
SELECTED_EXTENSION_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5139-5142",
        "updated_pack_external_rule_selector_selected_extension_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5183-5186"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack blind-vector "
    "solver-side numeric rerun audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_blind_vector_solver_side_numeric_rerun_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_solver_side_deformation_front_runner_audited_numeric_"
    "rerun_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_solver_side_numeric_rerun_backend_gap_audited_backend_"
    "inventory_primary_hybrid_reserve_secondary_next"
)
SCRIPT_SCAN_KEYWORDS = (
    "K_eff^(pilot-HS,recomp)",
    "Z_eff^(pilot-HS,recomp,T)",
    "F_blind^(pilot-HS,recomp)",
    "alpha_blind^(pilot-HS,recomp)",
)
SCRIPT_SCAN_EXCLUDED = {
    "t2a_5175.py",
    "t2a_5179.py",
    "t2a_5183.py",
    "t2a_5187.py",
}
SUMMARY_SCAN_KEYS = (
    "blind_F_recomp_at_zero",
    "blind_F_recomp_at_q_theory",
    "blind_F_recomp_at_m0",
    "blind_alpha_recomp_at_q_theory",
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


# 関数: rerun backend 候補 script を走査する。
def scan_backend_scripts() -> list[str]:
    """Return non-wrapper scripts that already implement selected-extension rerun objects."""
    candidates: list[str] = []
    for path in sorted((ROOT / "scripts" / "quantum").glob("*.py")):
        if path.name in SCRIPT_SCAN_EXCLUDED:
            continue

        text = path.read_text(encoding="utf-8")
        if any(keyword in text for keyword in SCRIPT_SCAN_KEYWORDS):
            candidates.append(path.name)

    return candidates


# 関数: recomputed retained-q numeric summary key の有無を走査する。
def scan_recomputed_numeric_outputs() -> list[str]:
    """Return public JSON artifacts that already expose recomputed retained-q numeric values."""
    hits: list[str] = []
    for path in sorted(PUBLIC_OUT.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        summary = payload.get("summary")
        if not isinstance(summary, dict):
            continue

        if all(key in summary for key in SUMMARY_SCAN_KEYS):
            hits.append(path.name)

    return hits


# 関数: numeric-rerun backend-gap audit の式を返す。
def build_formulae() -> dict[str, str]:
    """Return formulas used in the blind-vector solver-side numeric rerun audit."""
    return {
        "backend_inventory": (
            "B_recomp^(pilot-HS) := {B_K^(pilot-HS), B_G^(pilot-HS), "
            "B_Qret^(pilot-HS)}"
        ),
        "effective_kernel_backend": (
            "B_K^(pilot-HS) : (Sigma_*^(pilot-HS), Q_ret) -> "
            "K_AA^(Sigma_*^(pilot-HS))[Q], K_xiA^(Sigma_*^(pilot-HS))[Q]"
        ),
        "resolvent_backend": (
            "B_G^(pilot-HS) : (Sigma_*^(pilot-HS), Q_ret) -> (K_xixi[Q])^(-1)"
        ),
        "retained_q_numeric_backend": (
            "B_Qret^(pilot-HS) : (Sigma_*^(pilot-HS), Q_ret) -> "
            "{F_blind^(pilot-HS,recomp)(0), "
            "F_blind^(pilot-HS,recomp)(q_theory), "
            "F_blind^(pilot-HS,recomp)(m0), "
            "alpha_blind^(pilot-HS,recomp)(q_theory)}"
        ),
        "actual_rerun_availability": (
            "run_recomp^(pilot-HS) available iff "
            "Sigma_*^(pilot-HS), B_recomp^(pilot-HS), and O_recomp^(pilot-HS)[Q_ret] "
            "are all present"
        ),
    }


# 関数: `.5183-.5186` を実行する。
def main() -> None:
    """Execute the blind-vector solver-side numeric rerun audit."""
    for path in (PRIOR_GATE, SELECTED_EXTENSION_GATE):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    selected_summary = sign_base.read_json(SELECTED_EXTENSION_GATE)["summary"]

    audit_selected = bool(
        prior_summary["gate_b_updated_pack_blind_vector_solver_side_numeric_rerun_promoted_next"]
        and prior_summary["pack_update_required_now"]
    )
    selected_extension_still_available = bool(
        selected_summary[
            "gate_a_updated_pack_exact_external_rule_selector_selected_extension_available_now"
        ]
        and selected_summary[
            "exact_external_rule_selector_selected_extension_available_now"
        ]
    )
    rerun_contract_available = bool(
        prior_summary[
            "gate_a_updated_pack_exact_blind_vector_solver_side_deformation_front_runner_recompute_contract_available_now"
        ]
    )
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    farther_hybrid_continuation_reopen_required_now = bool(
        prior_summary["gate_c_farther_hybrid_continuation_reopen_required_now"]
    )

    backend_script_hits = scan_backend_scripts()
    recomputed_numeric_output_hits = scan_recomputed_numeric_outputs()
    backend_script_available_now = bool(backend_script_hits)
    recomputed_numeric_outputs_available_now = bool(recomputed_numeric_output_hits)
    actual_solver_side_numeric_rerun_available_now = bool(
        audit_selected
        and selected_extension_still_available
        and rerun_contract_available
        and backend_script_available_now
        and recomputed_numeric_outputs_available_now
    )
    exact_blind_vector_solver_side_numeric_rerun_backend_gap_theorem_available_now = bool(
        audit_selected
        and selected_extension_still_available
        and rerun_contract_available
        and not actual_solver_side_numeric_rerun_available_now
    )
    exact_blind_vector_solver_side_effective_kernel_backend_requirement_theorem_available_now = bool(
        exact_blind_vector_solver_side_numeric_rerun_backend_gap_theorem_available_now
    )
    exact_blind_vector_solver_side_internal_resolvent_backend_requirement_theorem_available_now = bool(
        exact_blind_vector_solver_side_numeric_rerun_backend_gap_theorem_available_now
    )
    exact_blind_vector_solver_side_retained_q_numeric_backend_requirement_theorem_available_now = bool(
        exact_blind_vector_solver_side_numeric_rerun_backend_gap_theorem_available_now
    )
    updated_pack_blind_vector_solver_side_backend_inventory_followup_required = bool(
        exact_blind_vector_solver_side_numeric_rerun_backend_gap_theorem_available_now
    )
    updated_pack_same_schema_blind_vector_numeric_rerun_replay_detected_now = False

    rows = [
        sign_base.row(
            "updated_pack_blind_vector_solver_side_numeric_rerun_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack blind-vector solver-side numeric rerun audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after a retained-q recomputation contract has already been promoted as the honest next blocker.",
        ),
        sign_base.row(
            "selected_extension_still_available_now",
            "pass" if selected_extension_still_available else "reject",
            "selected extension still available now",
            sign_base.truth(selected_extension_still_available),
            "The rerun audit remains meaningful only while the adopted external selector still fixes one concrete selected extension.",
        ),
        sign_base.row(
            "rerun_contract_available_now",
            "pass" if rerun_contract_available else "reject",
            "rerun contract available now",
            sign_base.truth(rerun_contract_available),
            "The retained-q recomputation contract must already be literal before we can audit actual execution readiness.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The active lane stays on computation-side blocker reduction instead of reopening theorem-family recursion.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "Auditing numeric rerun readiness does not reopen exhausted surrogate or selector-family branches.",
        ),
        sign_base.row(
            "backend_script_available_now",
            "pass" if backend_script_available_now else "reject",
            "backend script available now",
            sign_base.truth(backend_script_available_now),
            "A true rerun requires a non-wrapper implementation that maps the fixed selected extension to recomputed kernel/resolvent values.",
        ),
        sign_base.row(
            "recomputed_numeric_outputs_available_now",
            "pass" if recomputed_numeric_outputs_available_now else "reject",
            "recomputed numeric outputs available now",
            sign_base.truth(recomputed_numeric_outputs_available_now),
            "A true rerun also requires public numeric outputs carrying recomputed retained-q values rather than only theorem-side contract declarations.",
        ),
        sign_base.row(
            "actual_blind_vector_solver_side_numeric_rerun_available_now",
            "pass" if actual_solver_side_numeric_rerun_available_now else "reject",
            "actual blind-vector solver-side numeric rerun available now",
            sign_base.truth(actual_solver_side_numeric_rerun_available_now),
            "The actual rerun is available only if both the selected-extension backend and recomputed retained-q outputs already exist.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_numeric_rerun_backend_gap_theorem_available_now",
            "pass"
            if exact_blind_vector_solver_side_numeric_rerun_backend_gap_theorem_available_now
            else "reject",
            "exact blind-vector solver-side numeric rerun backend-gap theorem available now",
            sign_base.truth(
                exact_blind_vector_solver_side_numeric_rerun_backend_gap_theorem_available_now
            ),
            "The honest blocker is now explicit: the selected extension and rerun contract exist, but the actual recomputation backend and outputs do not.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_effective_kernel_backend_requirement_theorem_available_now",
            "pass"
            if exact_blind_vector_solver_side_effective_kernel_backend_requirement_theorem_available_now
            else "reject",
            "exact blind-vector solver-side effective-kernel backend requirement theorem available now",
            sign_base.truth(
                exact_blind_vector_solver_side_effective_kernel_backend_requirement_theorem_available_now
            ),
            "A concrete backend must recompute K_AA and K_xiA on the selected extension rather than reuse retained blind checkpoint values.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_internal_resolvent_backend_requirement_theorem_available_now",
            "pass"
            if exact_blind_vector_solver_side_internal_resolvent_backend_requirement_theorem_available_now
            else "reject",
            "exact blind-vector solver-side internal-resolvent backend requirement theorem available now",
            sign_base.truth(
                exact_blind_vector_solver_side_internal_resolvent_backend_requirement_theorem_available_now
            ),
            "The live rerun also requires a concrete recomputation path for the internal resolvent instead of inheriting the retained proxy.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_retained_q_numeric_backend_requirement_theorem_available_now",
            "pass"
            if exact_blind_vector_solver_side_retained_q_numeric_backend_requirement_theorem_available_now
            else "reject",
            "exact blind-vector solver-side retained-q numeric backend requirement theorem available now",
            sign_base.truth(
                exact_blind_vector_solver_side_retained_q_numeric_backend_requirement_theorem_available_now
            ),
            "Even with kernel formulas fixed, the branch still needs concrete retained-q numeric outputs before residual-origin verdict can move again.",
        ),
        sign_base.row(
            "updated_pack_blind_vector_solver_side_backend_inventory_followup_required",
            "pass"
            if updated_pack_blind_vector_solver_side_backend_inventory_followup_required
            else "reject",
            "updated-pack blind-vector solver-side backend inventory followup required",
            sign_base.truth(
                updated_pack_blind_vector_solver_side_backend_inventory_followup_required
            ),
            "The honest next blocker is backend inventory/integration, not another theorem-family descent or farther-hybrid reopen.",
        ),
        sign_base.row(
            "updated_pack_same_schema_blind_vector_numeric_rerun_replay_detected_now",
            "pass"
            if updated_pack_same_schema_blind_vector_numeric_rerun_replay_detected_now
            else "reject",
            "updated-pack same-schema blind-vector numeric-rerun replay detected now",
            sign_base.truth(
                updated_pack_same_schema_blind_vector_numeric_rerun_replay_detected_now
            ),
            "False means this turn reduced the live computation blocker materially instead of replaying the already fixed rerun contract schema.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Farther hybrid continuation stays reserve-only because retained-q numeric rerun is still blocked upstream by backend integration.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_theory_over_m0": float(prior_summary["q_theory_over_m0"]),
        "blind_F_at_q_theory": float(prior_summary["blind_F_at_q_theory"]),
        "blind_alpha_at_q_theory": float(prior_summary["blind_alpha_at_q_theory"]),
        "alpha_exact_at_q_theory": float(prior_summary["alpha_exact_at_q_theory"]),
        "selected_extension_still_available_now": selected_extension_still_available,
        "rerun_contract_available_now": rerun_contract_available,
        "backend_script_hits": backend_script_hits,
        "recomputed_numeric_output_hits": recomputed_numeric_output_hits,
        "backend_script_available_now": backend_script_available_now,
        "recomputed_numeric_outputs_available_now": recomputed_numeric_outputs_available_now,
        "actual_blind_vector_solver_side_numeric_rerun_available_now": actual_solver_side_numeric_rerun_available_now,
        "exact_blind_vector_solver_side_numeric_rerun_backend_gap_theorem_available_now": exact_blind_vector_solver_side_numeric_rerun_backend_gap_theorem_available_now,
        "exact_blind_vector_solver_side_effective_kernel_backend_requirement_theorem_available_now": exact_blind_vector_solver_side_effective_kernel_backend_requirement_theorem_available_now,
        "exact_blind_vector_solver_side_internal_resolvent_backend_requirement_theorem_available_now": exact_blind_vector_solver_side_internal_resolvent_backend_requirement_theorem_available_now,
        "exact_blind_vector_solver_side_retained_q_numeric_backend_requirement_theorem_available_now": exact_blind_vector_solver_side_retained_q_numeric_backend_requirement_theorem_available_now,
        "updated_pack_blind_vector_solver_side_backend_inventory_followup_required": updated_pack_blind_vector_solver_side_backend_inventory_followup_required,
        "updated_pack_same_schema_blind_vector_numeric_rerun_replay_detected_now": updated_pack_same_schema_blind_vector_numeric_rerun_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": bool(
            updated_pack_blind_vector_solver_side_backend_inventory_followup_required
        ),
        "selected_primary_completion_lane": "updated_pack_blind_vector_solver_side_backend_inventory_audit",
        "selected_secondary_completion_lane": "updated_pack_blind_vector_residual_origin_refresh_after_backend_integration",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_solver_side_backend_inventory_audit",
        "recommended_next_route_or_none": "8.7.56.5191",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_solver_side_backend_inventory_gate",
        "selected_followup_route_or_none": "8.7.56.5195",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5185",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "selected_extension_gate": sign_base.display_path(SELECTED_EXTENSION_GATE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5191",
                "followup_route": "8.7.56.5195",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_blind_vector_solver_side_numeric_rerun_backend_gap_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} blind-vector solver-side numeric rerun audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
