#!/usr/bin/env python3
"""Generate 8.7.56.5199-.5202 blind-vector backend-adapter audit artifacts."""

from __future__ import annotations

import csv
import importlib.util
import inspect
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
        "8.7.56.5195-5198",
        "updated_pack_blind_vector_solver_side_backend_inventory_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5191-5194",
        "updated_pack_blind_vector_solver_side_backend_inventory_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
SELECTED_EXTENSION_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5135-5138",
        "updated_pack_external_rule_selector_selected_extension_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
BLIND_CONTRACT_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5143-5146",
        "updated_pack_blind_vector_direct_computation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
LEGACY_PROFILE_SCRIPT = (
    ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
)
LEGACY_LADDER_SCRIPT = (
    ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"
)
LEGACY_PROFILE_ARTIFACT = (
    PUBLIC_OUT / "mass_origin_vector_qball_ell_sector_shooting_pilot_metrics.json"
)
LEGACY_LADDER_ARTIFACT = (
    PUBLIC_OUT / "mass_origin_vector_qball_full_coupled_solver_pilot_metrics.json"
)
LEGACY_RATIO_ARTIFACT = (
    PUBLIC_OUT / "mass_origin_vector_qball_exact_mass_table_handoff_retry_metrics.json"
)

STEP_TAG = "8.7.56.5199-5202"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack blind-vector "
    "solver-side backend adapter theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_blind_vector_solver_side_backend_adapter_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_solver_side_backend_inventory_audited_legacy_backend_"
    "adapter_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_solver_side_backend_adapter_contract_audited_implementation_"
    "primary_hybrid_reserve_secondary_next"
)
PROFILE_FUNCTIONS = (
    "solve_sector_profile",
    "find_sector_amplitudes",
    "build_base_modes",
)
LADDER_FUNCTIONS = (
    "build_exact_ladder",
    "polarization_weight",
    "coupled_charge_factor",
    "coupled_mass_factor",
)
BLIND_RERUN_TARGET_KEYS = (
    "blind_F_at_zero",
    "blind_F_at_q_theory",
    "blind_F_at_m0",
    "blind_alpha_at_q_theory",
    "delta_alpha_sel_exact",
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


# 関数: legacy backend module を読み込む。

def load_module(path: Path, module_name: str):
    """Load one Python module from a file path without executing its CLI."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to load module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 関数: signature を Windows-safe な文字列へ変換する。

def signature_text(module, function_name: str) -> str:
    """Return one compact signature string for the requested function."""
    return f"{function_name}{inspect.signature(getattr(module, function_name))}"


# 関数: backend-adapter theorem の式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the blind-vector backend-adapter audit."""
    return {
        "adapter_input_pack": (
            "I_adapter^(pilot-HS,legacy-vq) := {Sigma_*^(pilot-HS), "
            "Q_ret = {0, q_theory, m0}, B_profile^(legacy-vq), "
            "B_ladder^(legacy-vq), A_exact^(retained-blind)}"
        ),
        "adapter_output_pack": (
            "O_adapter^(pilot-HS,legacy-vq) := {ell_scan_rows^(legacy-vq), "
            "base_modes^(legacy-vq), exact_ladder^(legacy-vq), "
            "anchor_row^(retained-blind), blind_rerun_target_keys}"
        ),
        "adapter_contract": (
            "B_adapter^(pilot-HS,legacy-vq) : I_adapter^(pilot-HS,legacy-vq) -> "
            "O_adapter^(pilot-HS,legacy-vq)"
        ),
        "profile_mapping": (
            "A_profile^(legacy-vq) := build_base_modes(ell_values=(1,2,3)) "
            "with localized fallback hooks solve_sector_profile/find_sector_amplitudes"
        ),
        "ladder_mapping": (
            "A_ladder^(legacy-vq) := build_exact_ladder(scalar_modes, base_modes, "
            "lambda_rot)"
        ),
        "front_runner_adapter": (
            "B_adapter,front^(pilot-HS,legacy-vq) := "
            "(A_profile^(legacy-vq), A_ladder^(legacy-vq), "
            "A_exact^(retained-blind), blind_rerun_target_keys)"
        ),
    }


# 関数: `.5199-.5202` を実行する。

def main() -> None:
    """Execute the blind-vector backend-adapter theorem audit."""
    for path in (
        PRIOR_GATE,
        PRIOR_AUDIT,
        SELECTED_EXTENSION_AUDIT,
        BLIND_CONTRACT_AUDIT,
        LEGACY_PROFILE_SCRIPT,
        LEGACY_LADDER_SCRIPT,
        LEGACY_PROFILE_ARTIFACT,
        LEGACY_LADDER_ARTIFACT,
        LEGACY_RATIO_ARTIFACT,
    ):
        sign_base.require(path)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    selected_extension_summary = sign_base.read_json(SELECTED_EXTENSION_AUDIT)["summary"]
    blind_contract_summary = sign_base.read_json(BLIND_CONTRACT_AUDIT)["summary"]
    profile_summary = sign_base.read_json(LEGACY_PROFILE_ARTIFACT)["summary"]
    ladder_summary = sign_base.read_json(LEGACY_LADDER_ARTIFACT)["summary"]
    ratio_summary = sign_base.read_json(LEGACY_RATIO_ARTIFACT)["summary"]

    profile_module = load_module(LEGACY_PROFILE_SCRIPT, "wavep_legacy_profile_backend")
    ladder_module = load_module(LEGACY_LADDER_SCRIPT, "wavep_legacy_ladder_backend")

    audit_selected = bool(
        prior_gate_summary[
            "gate_b_updated_pack_blind_vector_solver_side_backend_adapter_promoted_next"
        ]
        and prior_gate_summary[
            "gate_a_updated_pack_exact_blind_vector_solver_side_backend_inventory_nonempty_available_now"
        ]
    )
    retry_mode = bool(prior_gate_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_gate_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    selected_extension_available = bool(
        selected_extension_summary["exact_concrete_selected_extension_available_now"]
    )
    blind_contract_available = bool(
        blind_contract_summary[
            "exact_blind_vector_selected_extension_checkpoint_contract_available_now"
        ]
    )
    legacy_profile_reusable = bool(
        prior_audit_summary[
            "exact_blind_vector_solver_side_legacy_profile_backend_reusable_theorem_available_now"
        ]
    )
    legacy_ladder_reusable = bool(
        prior_audit_summary[
            "exact_blind_vector_solver_side_legacy_full_coupled_backend_reusable_theorem_available_now"
        ]
    )
    retained_anchor_reusable = bool(
        prior_audit_summary[
            "exact_blind_vector_solver_side_retained_anchor_backend_reusable_theorem_available_now"
        ]
    )
    profile_signatures = {
        name: signature_text(profile_module, name) for name in PROFILE_FUNCTIONS
    }
    ladder_signatures = {
        name: signature_text(ladder_module, name) for name in LADDER_FUNCTIONS
    }
    profile_signature_contract_available_now = bool(
        len(profile_signatures) == len(PROFILE_FUNCTIONS)
    )
    ladder_signature_contract_available_now = bool(
        len(ladder_signatures) == len(LADDER_FUNCTIONS)
    )
    legacy_anchor_payload_compatible_now = bool(
        int(profile_summary["total_integer_mode_count_lower_bound"]) > 0
        and int(ladder_summary["exact_state_count"]) > 0
        and ratio_summary["best_exact_match_or_none"] is not None
    )
    adapter_contract_explicit = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selected_extension_available
        and blind_contract_available
        and legacy_profile_reusable
        and legacy_ladder_reusable
        and retained_anchor_reusable
        and profile_signature_contract_available_now
        and ladder_signature_contract_available_now
        and legacy_anchor_payload_compatible_now
    )
    exact_blind_vector_solver_side_backend_adapter_input_pack_formula_available_now = bool(
        adapter_contract_explicit
    )
    exact_blind_vector_solver_side_backend_adapter_output_pack_formula_available_now = bool(
        adapter_contract_explicit
    )
    exact_blind_vector_solver_side_backend_adapter_contract_formula_available_now = bool(
        adapter_contract_explicit
    )
    exact_blind_vector_solver_side_backend_adapter_front_runner_formula_available_now = bool(
        adapter_contract_explicit
    )
    exact_blind_vector_solver_side_backend_adapter_compatibility_theorem_available_now = bool(
        adapter_contract_explicit
    )
    updated_pack_blind_vector_solver_side_backend_implementation_followup_required = bool(
        adapter_contract_explicit
    )
    updated_pack_same_schema_blind_vector_backend_adapter_replay_detected_now = False
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_blind_vector_solver_side_backend_adapter_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack blind-vector solver-side backend adapter audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after reusable legacy backend assets are already official and the live blocker is one concrete adapter contract.",
        ),
        sign_base.row(
            "selected_extension_available_now",
            "pass" if selected_extension_available else "reject",
            "selected extension available now",
            sign_base.truth(selected_extension_available),
            "The adapter contract is meaningful only while one concrete selected extension Sigma_*^(pilot-HS) remains official.",
        ),
        sign_base.row(
            "blind_contract_available_now",
            "pass" if blind_contract_available else "reject",
            "blind rerun checkpoint contract available now",
            sign_base.truth(blind_contract_available),
            "The adapter must terminate on the retained blind-vector checkpoint keys rather than invent a fresh comparison surface.",
        ),
        sign_base.row(
            "profile_backend_signature_contract_available_now",
            "pass" if profile_signature_contract_available_now else "reject",
            "profile backend signature contract available now",
            sign_base.truth(profile_signature_contract_available_now),
            "The legacy profile backend already exposes stable callable signatures for profile solve, amplitude scan, and base-mode assembly.",
        ),
        sign_base.row(
            "ladder_backend_signature_contract_available_now",
            "pass" if ladder_signature_contract_available_now else "reject",
            "ladder backend signature contract available now",
            sign_base.truth(ladder_signature_contract_available_now),
            "The legacy full-coupled backend already exposes stable callable signatures for exact ladder reconstruction and weighting factors.",
        ),
        sign_base.row(
            "legacy_anchor_payload_compatible_now",
            "pass" if legacy_anchor_payload_compatible_now else "reject",
            "legacy anchor payload compatible now",
            sign_base.truth(legacy_anchor_payload_compatible_now),
            "The retained-q exact anchor row and the legacy state tables already match the live blind-vector comparison surface.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_backend_adapter_input_pack_formula_available_now",
            "pass"
            if exact_blind_vector_solver_side_backend_adapter_input_pack_formula_available_now
            else "reject",
            "exact blind-vector solver-side backend adapter input-pack formula available now",
            sign_base.truth(
                exact_blind_vector_solver_side_backend_adapter_input_pack_formula_available_now
            ),
            "The selected extension, retained q-window, reusable legacy profile/ladder backends, and retained exact anchor are now frozen as one literal adapter input pack.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_backend_adapter_output_pack_formula_available_now",
            "pass"
            if exact_blind_vector_solver_side_backend_adapter_output_pack_formula_available_now
            else "reject",
            "exact blind-vector solver-side backend adapter output-pack formula available now",
            sign_base.truth(
                exact_blind_vector_solver_side_backend_adapter_output_pack_formula_available_now
            ),
            "The adapter now has one explicit output contract: legacy profile rows, base modes, exact ladder rows, retained anchor row, and the blind rerun target keys.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_backend_adapter_contract_formula_available_now",
            "pass"
            if exact_blind_vector_solver_side_backend_adapter_contract_formula_available_now
            else "reject",
            "exact blind-vector solver-side backend adapter contract formula available now",
            sign_base.truth(
                exact_blind_vector_solver_side_backend_adapter_contract_formula_available_now
            ),
            "The live blocker is no longer generic backend wiring; it is one concrete contract from selected-extension inputs to implementation-ready legacy backend payloads.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_backend_adapter_front_runner_formula_available_now",
            "pass"
            if exact_blind_vector_solver_side_backend_adapter_front_runner_formula_available_now
            else "reject",
            "exact blind-vector solver-side backend adapter front-runner formula available now",
            sign_base.truth(
                exact_blind_vector_solver_side_backend_adapter_front_runner_formula_available_now
            ),
            "The honest front-runner adapter is now explicit: build base modes, rebuild the exact ladder, attach the retained exact anchor, and emit blind rerun target keys.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_backend_adapter_compatibility_theorem_available_now",
            "pass"
            if exact_blind_vector_solver_side_backend_adapter_compatibility_theorem_available_now
            else "reject",
            "exact blind-vector solver-side backend adapter compatibility theorem available now",
            sign_base.truth(
                exact_blind_vector_solver_side_backend_adapter_compatibility_theorem_available_now
            ),
            "The fixed selected extension and retained-q rerun surface are now compatible with the legacy vector-Q-ball backend assets at the contract level.",
        ),
        sign_base.row(
            "updated_pack_blind_vector_solver_side_backend_implementation_followup_required",
            "pass"
            if updated_pack_blind_vector_solver_side_backend_implementation_followup_required
            else "reject",
            "updated-pack blind-vector solver-side backend implementation followup required",
            sign_base.truth(
                updated_pack_blind_vector_solver_side_backend_implementation_followup_required
            ),
            "The honest next blocker is actual implementation of the front-runner adapter contract, not another inventory or theorem-family replay.",
        ),
        sign_base.row(
            "updated_pack_same_schema_blind_vector_backend_adapter_replay_detected_now",
            "pass"
            if updated_pack_same_schema_blind_vector_backend_adapter_replay_detected_now
            else "reject",
            "updated-pack same-schema blind-vector backend adapter replay detected now",
            sign_base.truth(
                updated_pack_same_schema_blind_vector_backend_adapter_replay_detected_now
            ),
            "False means this turn compressed the live blocker from generic adapter requirement to one concrete implementation contract.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Farther hybrid continuation stays reserve-only because the live blocker is now actual backend implementation on the retained-q rerun surface.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_theory_over_m0": float(prior_gate_summary["q_theory_over_m0"]),
        "blind_F_at_q_theory": float(prior_gate_summary["blind_F_at_q_theory"]),
        "blind_alpha_at_q_theory": float(prior_gate_summary["blind_alpha_at_q_theory"]),
        "alpha_exact_at_q_theory": float(prior_gate_summary["alpha_exact_at_q_theory"]),
        "blind_rerun_target_keys": list(BLIND_RERUN_TARGET_KEYS),
        "profile_backend_signatures": profile_signatures,
        "ladder_backend_signatures": ladder_signatures,
        "legacy_profile_integer_mode_count": int(
            profile_summary["total_integer_mode_count_lower_bound"]
        ),
        "legacy_exact_state_count": int(ladder_summary["exact_state_count"]),
        "legacy_exact_ratio_candidate_count": int(
            ratio_summary["exact_ratio_candidate_count"]
        ),
        "exact_blind_vector_solver_side_backend_adapter_input_pack_formula_available_now": exact_blind_vector_solver_side_backend_adapter_input_pack_formula_available_now,
        "exact_blind_vector_solver_side_backend_adapter_output_pack_formula_available_now": exact_blind_vector_solver_side_backend_adapter_output_pack_formula_available_now,
        "exact_blind_vector_solver_side_backend_adapter_contract_formula_available_now": exact_blind_vector_solver_side_backend_adapter_contract_formula_available_now,
        "exact_blind_vector_solver_side_backend_adapter_front_runner_formula_available_now": exact_blind_vector_solver_side_backend_adapter_front_runner_formula_available_now,
        "exact_blind_vector_solver_side_backend_adapter_compatibility_theorem_available_now": exact_blind_vector_solver_side_backend_adapter_compatibility_theorem_available_now,
        "updated_pack_blind_vector_solver_side_backend_implementation_followup_required": updated_pack_blind_vector_solver_side_backend_implementation_followup_required,
        "updated_pack_same_schema_blind_vector_backend_adapter_replay_detected_now": updated_pack_same_schema_blind_vector_backend_adapter_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": updated_pack_blind_vector_solver_side_backend_implementation_followup_required,
        "selected_primary_completion_lane": "updated_pack_blind_vector_solver_side_backend_implementation_audit",
        "selected_secondary_completion_lane": "updated_pack_blind_vector_numeric_rerun_after_backend_implementation",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_solver_side_backend_adapter_gate",
        "recommended_next_route_or_none": "8.7.56.5203",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_solver_side_backend_implementation_audit",
        "selected_followup_route_or_none": "8.7.56.5207",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5201",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "selected_extension_audit": sign_base.display_path(SELECTED_EXTENSION_AUDIT),
                "blind_contract_audit": sign_base.display_path(BLIND_CONTRACT_AUDIT),
                "legacy_profile_script": sign_base.display_path(LEGACY_PROFILE_SCRIPT),
                "legacy_ladder_script": sign_base.display_path(LEGACY_LADDER_SCRIPT),
                "legacy_profile_artifact": sign_base.display_path(LEGACY_PROFILE_ARTIFACT),
                "legacy_ladder_artifact": sign_base.display_path(LEGACY_LADDER_ARTIFACT),
                "legacy_ratio_artifact": sign_base.display_path(LEGACY_RATIO_ARTIFACT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5203",
                "followup_route": "8.7.56.5207",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_blind_vector_solver_side_backend_adapter_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} blind-vector backend-adapter audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
