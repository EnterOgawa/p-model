#!/usr/bin/env python3
"""Generate 8.7.56.5463-.5466 Trial-2 numerical closeout inventory artifacts."""

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
ALPHA_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5375-5378",
        "updated_pack_scalar_proxy_alpha_q_curve_diagnosis_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
ROUTE_C_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5451-5454",
        "updated_pack_scalar_proxy_route_c_virial_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
SOURCE_MATERIALIZATION_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5459-5462",
        "updated_pack_selected_extension_independent_extra_q_range_source_materialization_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5463-5466"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "numerical closeout inventory audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_numerical_closeout_inventory_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_independent_extra_q_range_source_materialization_"
    "negative_closeout_completed_trial2_numerical_closeout_inventory_"
    "primary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_numerical_closeout_inventory_audited_target_free_blind_overlap_"
    "practical_close_primary_paper_sync_gate"
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


# 関数: numerical closeout inventory で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the Trial-2 numerical closeout inventory."""
    return {
        "scalar_proxy_formula": "alpha(q) = F(q)^2 / (4 pi)",
        "blind_overlap_match": "q_blind = q_exact up to machine precision",
        "practical_closeout": (
            "q_numeric := q_blind and alpha_numeric := F(q_numeric)^2 / (4 pi)"
        ),
        "closeout_reading": (
            "practical numerical closeout available now while target-free exact "
            "matching-law theorem remains absent"
        ),
    }


# 関数: `.5463-.5466` を実行する。

def main() -> None:
    """Execute the Trial-2 numerical closeout inventory audit."""
    for path in (ALPHA_AUDIT, ROUTE_C_GATE, SOURCE_MATERIALIZATION_GATE):
        sign_base.require(path)

    alpha_summary = sign_base.read_json(ALPHA_AUDIT)["summary"]
    route_c_summary = sign_base.read_json(ROUTE_C_GATE)["summary"]
    source_summary = sign_base.read_json(SOURCE_MATERIALIZATION_GATE)["summary"]

    inventory_selected = bool(
        alpha_summary["exact_scalar_proxy_alpha_q_curve_formula_available_now"]
        and alpha_summary["exact_scalar_proxy_q_exact_exists_on_retained_interval_now"]
        and alpha_summary["exact_scalar_proxy_q_exact_unique_on_retained_interval_now"]
        and route_c_summary[
            "gate_a_updated_pack_exact_scalar_proxy_route_c_virial_negative_closeout_available_now"
        ]
        and source_summary[
            "gate_a_updated_pack_exact_selected_extension_independent_extra_q_range_source_materialization_negative_closeout_available_now"
        ]
    )
    scalar_proxy_formula_alive_now = bool(
        alpha_summary["exact_scalar_proxy_alpha_q_curve_formula_available_now"]
        and not alpha_summary["exact_scalar_proxy_formula_failure_now"]
    )
    scalar_proxy_unique_q_exact_now = bool(
        alpha_summary["exact_scalar_proxy_q_exact_exists_on_retained_interval_now"]
        and alpha_summary["exact_scalar_proxy_q_exact_unique_on_retained_interval_now"]
    )
    scalar_proxy_q_exact_matches_blind_overlap_target_free_now = bool(
        abs(float(alpha_summary["q_exact_matches_prior_blind_crossing_abs_error"]))
        <= 1.0e-12
    )
    scalar_proxy_target_free_matching_law_closed_form_available_now = False
    selected_extension_source_materialization_negative_closeout_available_now = bool(
        source_summary[
            "gate_a_updated_pack_exact_selected_extension_independent_extra_q_range_source_materialization_negative_closeout_available_now"
        ]
    )
    trial2_practical_blind_overlap_numerical_closeout_available_now = bool(
        inventory_selected
        and scalar_proxy_formula_alive_now
        and scalar_proxy_unique_q_exact_now
        and scalar_proxy_q_exact_matches_blind_overlap_target_free_now
        and selected_extension_source_materialization_negative_closeout_available_now
    )
    trial2_exact_theorem_closeout_available_now = bool(
        trial2_practical_blind_overlap_numerical_closeout_available_now
        and scalar_proxy_target_free_matching_law_closed_form_available_now
    )
    updated_pack_trial2_numerical_closeout_paper_sync_followup_required_now = bool(
        trial2_practical_blind_overlap_numerical_closeout_available_now
    )
    updated_pack_same_schema_trial2_numerical_closeout_replay_detected_now = False

    rows = [
        sign_base.row(
            "exact_trial2_numerical_closeout_inventory_available_now",
            "pass" if inventory_selected else "reject",
            "exact Trial-2 numerical closeout inventory available now",
            sign_base.truth(inventory_selected),
            "The scalar-proxy alive formula, the exhausted theorem-side derivation program, and the exhausted selected-extension rescue program can now be read in one inventory.",
        ),
        sign_base.row(
            "trial2_scalar_proxy_formula_alive_now",
            "pass" if scalar_proxy_formula_alive_now else "reject",
            "Trial-2 scalar-proxy formula alive now",
            sign_base.truth(scalar_proxy_formula_alive_now),
            "The retained scalar proxy still supports alpha(q)=F(q)^2/(4 pi); formula failure is no longer the honest read.",
        ),
        sign_base.row(
            "trial2_scalar_proxy_unique_q_exact_now",
            "pass" if scalar_proxy_unique_q_exact_now else "reject",
            "Trial-2 scalar-proxy unique q_exact now",
            sign_base.truth(scalar_proxy_unique_q_exact_now),
            "The dense scalar-proxy curve gives one unique retained crossing, so no extra crossing-selection rule is needed numerically.",
        ),
        sign_base.row(
            "trial2_scalar_proxy_q_exact_matches_blind_overlap_target_free_now",
            "pass"
            if scalar_proxy_q_exact_matches_blind_overlap_target_free_now
            else "reject",
            "Trial-2 scalar-proxy q_exact matches blind-overlap target-free now",
            sign_base.truth(
                scalar_proxy_q_exact_matches_blind_overlap_target_free_now
            ),
            "The old blind overlap crossing already supplied the same numerical matching scale without importing alpha_target by hand.",
        ),
        sign_base.row(
            "trial2_scalar_proxy_target_free_matching_law_closed_form_available_now",
            "pass"
            if scalar_proxy_target_free_matching_law_closed_form_available_now
            else "reject",
            "Trial-2 scalar-proxy target-free matching-law closed form available now",
            sign_base.truth(
                scalar_proxy_target_free_matching_law_closed_form_available_now
            ),
            "Reject means the theorem-side derivation program exhausted Route B/A/D/C without closing one exact target-free matching law.",
        ),
        sign_base.row(
            "trial2_selected_extension_source_materialization_negative_closeout_available_now",
            "pass"
            if selected_extension_source_materialization_negative_closeout_available_now
            else "reject",
            "Trial-2 selected-extension source-materialization negative closeout available now",
            sign_base.truth(
                selected_extension_source_materialization_negative_closeout_available_now
            ),
            "The helper-backed extra-q surface preserves the retained failure and only materializes legacy Phase-3 sidebands.",
        ),
        sign_base.row(
            "trial2_practical_blind_overlap_numerical_closeout_available_now",
            "pass"
            if trial2_practical_blind_overlap_numerical_closeout_available_now
            else "reject",
            "Trial-2 practical blind-overlap numerical closeout available now",
            sign_base.truth(
                trial2_practical_blind_overlap_numerical_closeout_available_now
            ),
            "The honest numerical read is now explicit: q_blind=q_exact operationally closes the retained scalar proxy even though no exact closed-form law has been derived.",
        ),
        sign_base.row(
            "trial2_exact_theorem_closeout_available_now",
            "pass" if trial2_exact_theorem_closeout_available_now else "reject",
            "Trial-2 exact theorem closeout available now",
            sign_base.truth(trial2_exact_theorem_closeout_available_now),
            "Reject means Trial-2 can be closed numerically before it can be closed as a target-free analytic theorem.",
        ),
        sign_base.row(
            "updated_pack_trial2_numerical_closeout_paper_sync_followup_required_now",
            "pass"
            if updated_pack_trial2_numerical_closeout_paper_sync_followup_required_now
            else "reject",
            "updated-pack Trial-2 numerical closeout paper-sync followup required now",
            sign_base.truth(
                updated_pack_trial2_numerical_closeout_paper_sync_followup_required_now
            ),
            "Because the numerical closeout reading is now explicit, the next honest task is to sync paper-facing wording rather than reopen exhausted computation branches.",
        ),
        sign_base.row(
            "updated_pack_same_schema_trial2_numerical_closeout_replay_detected_now",
            "pass"
            if updated_pack_same_schema_trial2_numerical_closeout_replay_detected_now
            else "reject",
            "updated-pack same-schema Trial-2 numerical closeout replay detected now",
            sign_base.truth(
                updated_pack_same_schema_trial2_numerical_closeout_replay_detected_now
            ),
            "False means this branch did not spend one more turn replaying extra-q or theorem-side branches that are already exhausted.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_blind_over_m0": float(alpha_summary["q_blind_over_m0"]),
        "q_exact_over_m0": float(alpha_summary["primary_q_exact_over_m0"]),
        "q_star_over_m0": float(alpha_summary["q_star_over_m0"]),
        "q_theory_over_m0": float(source_summary["q_theory_over_m0"]),
        "q_exact_matches_prior_blind_crossing_abs_error": float(
            alpha_summary["q_exact_matches_prior_blind_crossing_abs_error"]
        ),
        "delta_q_over_q_star": float(alpha_summary["delta_q_over_q_star"]),
        "alpha_target": float(alpha_summary["alpha_target"]),
        "alpha_at_q_star": float(alpha_summary["alpha_at_q_star"]),
        "relative_residual_at_q_star": float(
            alpha_summary["relative_residual_at_q_star"]
        ),
        "alpha_exact_at_q_theory": float(source_summary["alpha_exact_at_q_theory"]),
        "q_theory_diagnostic": source_summary["q_theory_diagnostic"],
        "best_extra_label_vs_alpha_target": source_summary[
            "best_extra_label_vs_alpha_target"
        ],
        "best_extra_alpha_target_residual": float(
            source_summary["best_extra_alpha_target_residual"]
        ),
        "best_extra_q_exact_gap": float(source_summary["best_extra_q_exact_gap"]),
        "best_extra_label_diagnostic": source_summary["best_extra_label_diagnostic"],
        "exact_trial2_numerical_closeout_inventory_available_now": inventory_selected,
        "scalar_proxy_formula_alive_now": scalar_proxy_formula_alive_now,
        "scalar_proxy_unique_q_exact_now": scalar_proxy_unique_q_exact_now,
        "scalar_proxy_q_exact_matches_blind_overlap_target_free_now": (
            scalar_proxy_q_exact_matches_blind_overlap_target_free_now
        ),
        "scalar_proxy_target_free_matching_law_closed_form_available_now": (
            scalar_proxy_target_free_matching_law_closed_form_available_now
        ),
        "selected_extension_source_materialization_negative_closeout_available_now": (
            selected_extension_source_materialization_negative_closeout_available_now
        ),
        "trial2_practical_blind_overlap_numerical_closeout_available_now": (
            trial2_practical_blind_overlap_numerical_closeout_available_now
        ),
        "trial2_exact_theorem_closeout_available_now": (
            trial2_exact_theorem_closeout_available_now
        ),
        "updated_pack_trial2_numerical_closeout_paper_sync_followup_required_now": (
            updated_pack_trial2_numerical_closeout_paper_sync_followup_required_now
        ),
        "updated_pack_same_schema_trial2_numerical_closeout_replay_detected_now": (
            updated_pack_same_schema_trial2_numerical_closeout_replay_detected_now
        ),
        "selected_primary_completion_lane": (
            "updated_pack_trial2_numerical_closeout_gate"
        ),
        "selected_secondary_completion_lane": (
            "updated_pack_trial2_numerical_closeout_expert_share_sync"
        ),
        "selected_reserve_completion_lane": (
            "farther_hybrid_reserve_only_until_new_independent_source_exists"
        ),
        "selected_next_generation_route": (
            "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_"
            "trial2_numerical_closeout_gate"
        ),
        "recommended_next_route_or_none": "8.7.56.5467",
        "selected_followup_route": (
            "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_"
            "trial2_numerical_closeout_expert_share_sync"
        ),
        "selected_followup_route_or_none": "8.7.56.5471",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5465",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "alpha_audit": sign_base.display_path(ALPHA_AUDIT),
                "route_c_gate": sign_base.display_path(ROUTE_C_GATE),
                "source_materialization_gate": sign_base.display_path(
                    SOURCE_MATERIALIZATION_GATE
                ),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5467",
                "followup_route": "8.7.56.5471",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_numerical_closeout_inventory_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 numerical closeout inventory completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から inventory audit を実行する。

if __name__ == "__main__":
    main()
