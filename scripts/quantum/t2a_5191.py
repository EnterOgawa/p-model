#!/usr/bin/env python3
"""Generate 8.7.56.5191-.5194 blind-vector solver-side backend inventory artifacts."""

from __future__ import annotations

import csv
import importlib.util
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
        "8.7.56.5187-5190",
        "updated_pack_blind_vector_solver_side_numeric_rerun_gate",
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

STEP_TAG = "8.7.56.5191-5194"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack blind-vector "
    "solver-side backend inventory audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_blind_vector_solver_side_backend_inventory_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_solver_side_numeric_rerun_backend_gap_audited_backend_"
    "inventory_primary_hybrid_reserve_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "blind_vector_solver_side_backend_inventory_audited_legacy_backend_"
    "adapter_primary_hybrid_reserve_secondary_next"
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


# 関数: blind-vector backend inventory の式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the blind-vector solver-side backend inventory audit."""
    return {
        "backend_inventory": (
            "Inv_backend^(pilot-HS) := {B_profile^(legacy-vq), "
            "B_ladder^(legacy-vq), B_anchor^(retained-blind)}"
        ),
        "profile_backend": (
            "B_profile^(legacy-vq) := {solve_sector_profile, "
            "find_sector_amplitudes, build_base_modes}"
        ),
        "ladder_backend": (
            "B_ladder^(legacy-vq) := {build_exact_ladder, polarization_weight, "
            "coupled_charge_factor, coupled_mass_factor}"
        ),
        "anchor_backend": (
            "B_anchor^(retained-blind) := {Q_ret, q_theory, alpha_exact(q_theory), "
            "legacy exact-ladder checkpoint rows}"
        ),
        "adapter_requirement": (
            "B_adapter^(pilot-HS,legacy-vq) required because Inv_backend^(pilot-HS) "
            "contains reusable profile/ladder assets, but not yet a selected-extension "
            "Schur-complement wiring from {Sigma_*^(pilot-HS), Q_ret} to "
            "{K_AA, K_xiA, (K_xixi)^(-1), F_blind^(recomp), alpha_blind^(recomp)}"
        ),
    }


# 関数: `.5191-.5194` を実行する。

def main() -> None:
    """Execute the blind-vector solver-side backend inventory audit."""
    for path in (
        PRIOR_GATE,
        SELECTED_EXTENSION_GATE,
        LEGACY_PROFILE_SCRIPT,
        LEGACY_LADDER_SCRIPT,
        LEGACY_PROFILE_ARTIFACT,
        LEGACY_LADDER_ARTIFACT,
        LEGACY_RATIO_ARTIFACT,
    ):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    selected_summary = sign_base.read_json(SELECTED_EXTENSION_GATE)["summary"]
    profile_summary = sign_base.read_json(LEGACY_PROFILE_ARTIFACT)["summary"]
    ladder_summary = sign_base.read_json(LEGACY_LADDER_ARTIFACT)["summary"]
    ratio_summary = sign_base.read_json(LEGACY_RATIO_ARTIFACT)["summary"]

    profile_module = load_module(LEGACY_PROFILE_SCRIPT, "wavep_legacy_profile_backend")
    ladder_module = load_module(LEGACY_LADDER_SCRIPT, "wavep_legacy_ladder_backend")

    audit_selected = bool(
        prior_summary[
            "gate_b_updated_pack_blind_vector_solver_side_backend_inventory_promoted_next"
        ]
        and prior_summary["pack_update_required_now"]
    )
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(
        prior_summary["failure_matrix_non_surrogate_guard_preserved"]
    )
    selected_extension_still_available = bool(
        selected_summary[
            "gate_a_updated_pack_exact_external_rule_selector_selected_extension_available_now"
        ]
        and selected_summary[
            "exact_external_rule_selector_selected_extension_available_now"
        ]
    )
    rerun_backend_gap_still_available = bool(
        prior_summary[
            "gate_a_updated_pack_exact_blind_vector_solver_side_numeric_rerun_backend_gap_available_now"
        ]
        and not prior_summary["actual_blind_vector_solver_side_numeric_rerun_available_now"]
    )

    profile_backend_function_hits = [
        name for name in PROFILE_FUNCTIONS if hasattr(profile_module, name)
    ]
    ladder_backend_function_hits = [
        name for name in LADDER_FUNCTIONS if hasattr(ladder_module, name)
    ]
    legacy_profile_backend_script_available_now = bool(
        len(profile_backend_function_hits) == len(PROFILE_FUNCTIONS)
    )
    legacy_full_coupled_ladder_backend_script_available_now = bool(
        len(ladder_backend_function_hits) == len(LADDER_FUNCTIONS)
    )
    legacy_profile_backend_artifact_available_now = bool(
        int(profile_summary["localized_ell_sector_count"]) > 0
        and int(profile_summary["total_integer_mode_count_lower_bound"]) > 0
    )
    legacy_full_coupled_ladder_backend_artifact_available_now = bool(
        ladder_summary["exact_full_coupled_vector_ladder_available"]
        and int(ladder_summary["exact_state_count"]) > 0
    )
    retained_blind_anchor_backend_artifact_available_now = bool(
        ratio_summary["best_exact_match_or_none"] is not None
        and ratio_summary["hand_off_to_8_7_55_2_84"]
    )
    exact_blind_vector_solver_side_backend_inventory_nonempty_theorem_available_now = bool(
        audit_selected
        and retry_mode
        and non_surrogate_guard
        and selected_extension_still_available
        and rerun_backend_gap_still_available
        and legacy_profile_backend_script_available_now
        and legacy_full_coupled_ladder_backend_script_available_now
        and legacy_profile_backend_artifact_available_now
        and legacy_full_coupled_ladder_backend_artifact_available_now
        and retained_blind_anchor_backend_artifact_available_now
    )
    exact_blind_vector_solver_side_legacy_profile_backend_reusable_theorem_available_now = bool(
        exact_blind_vector_solver_side_backend_inventory_nonempty_theorem_available_now
    )
    exact_blind_vector_solver_side_legacy_full_coupled_backend_reusable_theorem_available_now = bool(
        exact_blind_vector_solver_side_backend_inventory_nonempty_theorem_available_now
    )
    exact_blind_vector_solver_side_retained_anchor_backend_reusable_theorem_available_now = bool(
        exact_blind_vector_solver_side_backend_inventory_nonempty_theorem_available_now
    )
    exact_blind_vector_solver_side_selected_extension_backend_adapter_requirement_theorem_available_now = bool(
        exact_blind_vector_solver_side_backend_inventory_nonempty_theorem_available_now
    )
    updated_pack_blind_vector_solver_side_backend_adapter_followup_required = bool(
        exact_blind_vector_solver_side_selected_extension_backend_adapter_requirement_theorem_available_now
    )
    updated_pack_same_schema_blind_vector_backend_inventory_replay_detected_now = False
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "updated_pack_blind_vector_solver_side_backend_inventory_audit_selected",
            "pass" if audit_selected else "reject",
            "updated-pack blind-vector solver-side backend inventory audit selected",
            sign_base.truth(audit_selected),
            "This branch is worth running only after the selected extension and rerun contract are fixed while actual numeric rerun remains blocked by a backend gap.",
        ),
        sign_base.row(
            "selected_extension_still_available_now",
            "pass" if selected_extension_still_available else "reject",
            "selected extension still available now",
            sign_base.truth(selected_extension_still_available),
            "Backend inventory matters only while one concrete selected extension Sigma_*^(pilot-HS) remains official.",
        ),
        sign_base.row(
            "rerun_backend_gap_still_available_now",
            "pass" if rerun_backend_gap_still_available else "reject",
            "rerun backend-gap theorem still available now",
            sign_base.truth(rerun_backend_gap_still_available),
            "The branch stays honest only if the live blocker is still missing backend integration rather than selector ambiguity or same-tag replay.",
        ),
        sign_base.row(
            "legacy_profile_backend_script_available_now",
            "pass" if legacy_profile_backend_script_available_now else "reject",
            "legacy profile backend script available now",
            sign_base.truth(legacy_profile_backend_script_available_now),
            "The old vector-Q-ball numerical solver already exposes localized profile builders that can seed a selected-extension backend path.",
        ),
        sign_base.row(
            "legacy_full_coupled_ladder_backend_script_available_now",
            "pass" if legacy_full_coupled_ladder_backend_script_available_now else "reject",
            "legacy full-coupled ladder backend script available now",
            sign_base.truth(legacy_full_coupled_ladder_backend_script_available_now),
            "The old full-coupled solver already exposes exact-ladder reconstruction helpers that can be reused as backend building blocks.",
        ),
        sign_base.row(
            "legacy_profile_backend_artifact_available_now",
            "pass" if legacy_profile_backend_artifact_available_now else "reject",
            "legacy profile backend artifact available now",
            sign_base.truth(legacy_profile_backend_artifact_available_now),
            "The ell-sector shooting pilot already published localized profiles and integer-charge base modes that can serve as profile seeds.",
        ),
        sign_base.row(
            "legacy_full_coupled_ladder_backend_artifact_available_now",
            "pass" if legacy_full_coupled_ladder_backend_artifact_available_now else "reject",
            "legacy full-coupled ladder backend artifact available now",
            sign_base.truth(legacy_full_coupled_ladder_backend_artifact_available_now),
            "The full-coupled ladder artifact already publishes exact vector states that can serve as ladder-level backend anchors.",
        ),
        sign_base.row(
            "retained_blind_anchor_backend_artifact_available_now",
            "pass" if retained_blind_anchor_backend_artifact_available_now else "reject",
            "retained blind-anchor backend artifact available now",
            sign_base.truth(retained_blind_anchor_backend_artifact_available_now),
            "The exact mass-table retry already publishes the retained-q exact anchor row needed to keep backend wiring tied to the live blind-vector comparison surface.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_backend_inventory_nonempty_theorem_available_now",
            "pass"
            if exact_blind_vector_solver_side_backend_inventory_nonempty_theorem_available_now
            else "reject",
            "exact blind-vector solver-side backend inventory nonempty theorem available now",
            sign_base.truth(
                exact_blind_vector_solver_side_backend_inventory_nonempty_theorem_available_now
            ),
            "The live backend blocker is no longer generic emptiness: the repo already contains reusable profile, ladder, and retained-anchor backend assets.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_legacy_profile_backend_reusable_theorem_available_now",
            "pass"
            if exact_blind_vector_solver_side_legacy_profile_backend_reusable_theorem_available_now
            else "reject",
            "exact blind-vector solver-side legacy profile backend reusable theorem available now",
            sign_base.truth(
                exact_blind_vector_solver_side_legacy_profile_backend_reusable_theorem_available_now
            ),
            "The selected-extension rerun may reuse the legacy localized-profile backend instead of inventing a fresh profile solver from scratch.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_legacy_full_coupled_backend_reusable_theorem_available_now",
            "pass"
            if exact_blind_vector_solver_side_legacy_full_coupled_backend_reusable_theorem_available_now
            else "reject",
            "exact blind-vector solver-side legacy full-coupled backend reusable theorem available now",
            sign_base.truth(
                exact_blind_vector_solver_side_legacy_full_coupled_backend_reusable_theorem_available_now
            ),
            "The selected-extension rerun may reuse the exact-ladder reconstruction backend rather than reopen the old vector mass-origin search as a separate lane.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_retained_anchor_backend_reusable_theorem_available_now",
            "pass"
            if exact_blind_vector_solver_side_retained_anchor_backend_reusable_theorem_available_now
            else "reject",
            "exact blind-vector solver-side retained-anchor backend reusable theorem available now",
            sign_base.truth(
                exact_blind_vector_solver_side_retained_anchor_backend_reusable_theorem_available_now
            ),
            "The retained blind/vector exact anchor surface already exists and can keep the backend wiring attached to q_theory / m0 checkpoints.",
        ),
        sign_base.row(
            "exact_blind_vector_solver_side_selected_extension_backend_adapter_requirement_theorem_available_now",
            "pass"
            if exact_blind_vector_solver_side_selected_extension_backend_adapter_requirement_theorem_available_now
            else "reject",
            "exact blind-vector solver-side selected-extension backend adapter requirement theorem available now",
            sign_base.truth(
                exact_blind_vector_solver_side_selected_extension_backend_adapter_requirement_theorem_available_now
            ),
            "What is missing is now specific: one adapter that wires the fixed selected extension into the reusable legacy backend assets and emits Schur-complement rerun outputs.",
        ),
        sign_base.row(
            "updated_pack_blind_vector_solver_side_backend_adapter_followup_required",
            "pass" if updated_pack_blind_vector_solver_side_backend_adapter_followup_required else "reject",
            "updated-pack blind-vector solver-side backend adapter followup required",
            sign_base.truth(
                updated_pack_blind_vector_solver_side_backend_adapter_followup_required
            ),
            "The honest next blocker is one concrete selected-extension backend adapter contract, not generic inventory search or farther-hybrid reopen.",
        ),
        sign_base.row(
            "updated_pack_same_schema_blind_vector_backend_inventory_replay_detected_now",
            "pass" if updated_pack_same_schema_blind_vector_backend_inventory_replay_detected_now else "reject",
            "updated-pack same-schema blind-vector backend inventory replay detected now",
            sign_base.truth(
                updated_pack_same_schema_blind_vector_backend_inventory_replay_detected_now
            ),
            "False means this branch reduced the live blocker from generic backend absence to one concrete adapter requirement.",
        ),
        sign_base.row(
            "farther_hybrid_continuation_reopen_required_now",
            "pass" if farther_hybrid_continuation_reopen_required_now else "reject",
            "farther hybrid continuation reopen required now",
            sign_base.truth(farther_hybrid_continuation_reopen_required_now),
            "Farther hybrid continuation stays reserve-only because the live blocker remains backend wiring on the retained-q rerun surface.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_theory_over_m0": float(prior_summary["q_theory_over_m0"]),
        "blind_F_at_q_theory": float(prior_summary["blind_F_at_q_theory"]),
        "blind_alpha_at_q_theory": float(prior_summary["blind_alpha_at_q_theory"]),
        "alpha_exact_at_q_theory": float(prior_summary["alpha_exact_at_q_theory"]),
        "profile_backend_function_hits": profile_backend_function_hits,
        "ladder_backend_function_hits": ladder_backend_function_hits,
        "legacy_profile_integer_mode_count": int(
            profile_summary["total_integer_mode_count_lower_bound"]
        ),
        "legacy_profile_k_positive_mode_count": int(
            profile_summary["k_positive_mode_count"]
        ),
        "legacy_exact_state_count": int(ladder_summary["exact_state_count"]),
        "legacy_exact_best_match_relative_error": float(
            ratio_summary["best_exact_match_or_none"]["relative_error"]
        ),
        "legacy_profile_backend_script_available_now": legacy_profile_backend_script_available_now,
        "legacy_full_coupled_ladder_backend_script_available_now": legacy_full_coupled_ladder_backend_script_available_now,
        "legacy_profile_backend_artifact_available_now": legacy_profile_backend_artifact_available_now,
        "legacy_full_coupled_ladder_backend_artifact_available_now": legacy_full_coupled_ladder_backend_artifact_available_now,
        "retained_blind_anchor_backend_artifact_available_now": retained_blind_anchor_backend_artifact_available_now,
        "exact_blind_vector_solver_side_backend_inventory_nonempty_theorem_available_now": exact_blind_vector_solver_side_backend_inventory_nonempty_theorem_available_now,
        "exact_blind_vector_solver_side_legacy_profile_backend_reusable_theorem_available_now": exact_blind_vector_solver_side_legacy_profile_backend_reusable_theorem_available_now,
        "exact_blind_vector_solver_side_legacy_full_coupled_backend_reusable_theorem_available_now": exact_blind_vector_solver_side_legacy_full_coupled_backend_reusable_theorem_available_now,
        "exact_blind_vector_solver_side_retained_anchor_backend_reusable_theorem_available_now": exact_blind_vector_solver_side_retained_anchor_backend_reusable_theorem_available_now,
        "exact_blind_vector_solver_side_selected_extension_backend_adapter_requirement_theorem_available_now": exact_blind_vector_solver_side_selected_extension_backend_adapter_requirement_theorem_available_now,
        "updated_pack_blind_vector_solver_side_backend_adapter_followup_required": updated_pack_blind_vector_solver_side_backend_adapter_followup_required,
        "updated_pack_same_schema_blind_vector_backend_inventory_replay_detected_now": updated_pack_same_schema_blind_vector_backend_inventory_replay_detected_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "pack_update_required_now": updated_pack_blind_vector_solver_side_backend_adapter_followup_required,
        "selected_primary_completion_lane": "updated_pack_blind_vector_solver_side_backend_adapter_audit",
        "selected_secondary_completion_lane": "updated_pack_blind_vector_numeric_rerun_after_backend_wiring",
        "selected_reserve_completion_lane": "farther_hybrid_extra_q_range_only",
        "selected_next_generation_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_solver_side_backend_inventory_gate",
        "recommended_next_route_or_none": "8.7.56.5195",
        "selected_followup_route": "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_blind_vector_solver_side_backend_adapter_audit",
        "selected_followup_route_or_none": "8.7.56.5199",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5193",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "selected_extension_gate": sign_base.display_path(SELECTED_EXTENSION_GATE),
                "legacy_profile_script": sign_base.display_path(LEGACY_PROFILE_SCRIPT),
                "legacy_ladder_script": sign_base.display_path(LEGACY_LADDER_SCRIPT),
                "legacy_profile_artifact": sign_base.display_path(LEGACY_PROFILE_ARTIFACT),
                "legacy_ladder_artifact": sign_base.display_path(LEGACY_LADDER_ARTIFACT),
                "legacy_ratio_artifact": sign_base.display_path(LEGACY_RATIO_ARTIFACT),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5195",
                "followup_route": "8.7.56.5199",
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_blind_vector_solver_side_backend_inventory_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} blind-vector solver-side backend inventory audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
