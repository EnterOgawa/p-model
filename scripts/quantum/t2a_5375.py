#!/usr/bin/env python3
"""Generate 8.7.56.5375-.5378 scalar-proxy alpha(q) curve diagnosis artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.scalar_proxy_alpha_q_curve_backend import (
    build_scalar_proxy_alpha_q_curve_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths
from scripts.utils.windows_length_policy import ensure_windows_path_budget


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5371-5374",
        "updated_pack_selected_extension_independent_extra_q_range_source_materialization_implementation_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5375-5378"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor scalar-proxy alpha(q) "
    "curve diagnosis audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_scalar_proxy_alpha_q_curve_diagnosis_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "selected_extension_independent_extra_q_range_source_materialization_"
    "implementation_audited_numeric_rerun_primary_hybrid_reserve_secondary_"
    "next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_alpha_q_curve_diagnosed_matching_scale_redrive_primary_"
    "source_materialization_secondary_gate"
)


# Function: write one metrics payload as JSON and CSV.
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

    return {
        "json": sign_base.display_path(paths["json"]),
        "csv": sign_base.display_path(paths["csv"]),
    }


# Function: build the auxiliary curve artifact paths.

def build_curve_paths() -> dict[str, Path]:
    """Return one validated CSV/PDF path pair for the dense alpha(q) curve."""
    csv_path = ensure_windows_path_budget(PUBLIC_OUT / f"{STEM}_alpha_q_curve.csv")
    pdf_path = ensure_windows_path_budget(PUBLIC_OUT / f"{STEM}_alpha_q_curve.pdf")
    return {"csv": csv_path, "pdf": pdf_path}


# Function: write the dense curve CSV used for expert review.

def write_curve_csv(curve_paths: dict[str, Path], pack: dict) -> None:
    """Write the dense q/F/alpha curve as one CSV table."""
    with curve_paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["q_over_m0", "F_q", "alpha_q", "alpha_minus_target"],
        )
        writer.writeheader()
        for q_value, form_factor_value, alpha_value in zip(
            pack["q_values"],
            pack["form_factor_curve"],
            pack["alpha_curve"],
        ):
            writer.writerow(
                {
                    "q_over_m0": float(q_value),
                    "F_q": float(form_factor_value),
                    "alpha_q": float(alpha_value),
                    "alpha_minus_target": float(alpha_value - pack["alpha_target"]),
                }
            )


# Function: write the vector PDF used to visualize the scalar-proxy verdict.

def write_curve_pdf(curve_paths: dict[str, Path], pack: dict) -> None:
    """Write one vector PDF for the dense alpha(q) curve."""
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.plot(
        pack["q_values"],
        pack["alpha_curve"],
        color="#004c6d",
        linewidth=1.6,
        label=r"$\alpha(q)=F(q)^2/(4\pi)$",
    )
    axis.axhline(
        pack["alpha_target"],
        color="#d1495b",
        linewidth=1.1,
        linestyle="--",
        label=r"$\alpha_{\rm target}$",
    )
    axis.axvline(
        pack["q_star_over_m0"],
        color="#2f8f2f",
        linewidth=1.0,
        linestyle=":",
        label=r"$q_\ast$",
    )
    for root in pack["q_exact_list"]:
        axis.axvline(
            float(root),
            color="#a05a2c",
            linewidth=1.0,
            linestyle="-.",
            label=r"$q_{\rm exact}$" if root == pack["q_exact_list"][0] else None,
        )

    axis.set_xlabel(r"$q/m_0$")
    axis.set_ylabel(r"$\alpha(q)$")
    axis.set_xlim(float(pack["q_min_over_m0"]), float(pack["q_max_over_m0"]))
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(curve_paths["pdf"], format="pdf")
    plt.close(figure)


# Function: return formulas used by the computation-gate audit.

def build_formulae() -> dict[str, str]:
    """Return formulas used in the scalar-proxy alpha(q) audit."""
    return {
        "density_choice": "rho(r) = f_0(r)^2",
        "form_factor": "F(q) = int dr rho(r) r^2 sinc(q r) / int dr rho(r) r^2",
        "alpha_curve": "alpha(q) = F(q)^2 / (4 pi)",
        "target_crossing": "alpha(q_exact) = alpha_target",
        "matching_delta": "delta_q = q_exact - q_star",
    }


# Function: execute `.5375-.5378`.

def main() -> None:
    """Execute the scalar-proxy alpha(q) curve diagnosis audit."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(prior_summary["failure_matrix_non_surrogate_guard_preserved"])
    prior_numeric_rerun_promoted = bool(
        prior_summary[
            "gate_b_updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_promoted_next"
        ]
    )
    computation_gate_pivot_selected_now = bool(
        retry_mode and non_surrogate_guard and prior_numeric_rerun_promoted
    )

    pack = build_scalar_proxy_alpha_q_curve_pack()
    curve_paths = build_curve_paths()
    write_curve_csv(curve_paths, pack)
    write_curve_pdf(curve_paths, pack)

    q_exact_exists_now = bool(pack["q_exact_exists_now"])
    q_exact_unique_now = bool(pack["q_exact_unique_now"])
    formula_survives_now = bool(q_exact_exists_now)
    formula_failure_now = bool(pack["formula_failure_now"])
    matching_scale_primary_now = bool(pack["matching_scale_primary_now"])
    order_percent_matching_scale_correction_now = bool(
        q_exact_exists_now and abs(float(pack["delta_q_over_q_star"])) >= 3.0e-3
    )
    source_materialization_numeric_rerun_demoted_to_secondary_now = bool(
        computation_gate_pivot_selected_now and matching_scale_primary_now
    )
    same_schema_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_replayed_now = False
    farther_hybrid_continuation_reopen_required_now = False

    rows = [
        sign_base.row(
            "computation_gate_pivot_selected_now",
            "pass" if computation_gate_pivot_selected_now else "reject",
            "computation-gate pivot selected now",
            sign_base.truth(computation_gate_pivot_selected_now),
            "Repeated theory-extension and source-materialization branches reproduced the same failure surface, so this branch pivots to one higher-information scalar-proxy computation.",
        ),
        sign_base.row(
            "exact_scalar_proxy_alpha_q_curve_formula_available_now",
            "pass" if computation_gate_pivot_selected_now else "reject",
            "exact scalar-proxy alpha(q) curve formula available now",
            sign_base.truth(computation_gate_pivot_selected_now),
            "The dense scalar-proxy audit now fixes alpha(q)=F(q)^2/(4 pi) on the retained scalar profile without adding any parameter.",
        ),
        sign_base.row(
            "exact_scalar_proxy_q_exact_exists_on_retained_interval_now",
            "pass" if q_exact_exists_now else "reject",
            "exact scalar-proxy q_exact exists on retained interval now",
            sign_base.truth(q_exact_exists_now),
            "A target-crossing q_exact exists on the retained interval exactly when alpha(q) reaches alpha_target.",
        ),
        sign_base.row(
            "exact_scalar_proxy_q_exact_unique_on_retained_interval_now",
            "pass" if q_exact_unique_now else "reject",
            "exact scalar-proxy q_exact unique on retained interval now",
            sign_base.truth(q_exact_unique_now),
            "A unique crossing on the retained interval means the computation does not need an extra crossing-selection rule.",
        ),
        sign_base.row(
            "exact_scalar_proxy_formula_failure_now",
            "pass" if formula_failure_now else "reject",
            "exact scalar-proxy formula failure now",
            sign_base.truth(formula_failure_now),
            "False means alpha(q)=F(q)^2/(4 pi) survives the retained scalar-proxy audit because the curve actually reaches alpha_target.",
        ),
        sign_base.row(
            "exact_scalar_proxy_matching_scale_primary_verdict_available_now",
            "pass" if matching_scale_primary_now else "reject",
            "exact scalar-proxy matching-scale primary verdict available now",
            sign_base.truth(matching_scale_primary_now),
            "The retained scalar-proxy residual is now better read as a q_star matching-scale correction problem than as a formula failure problem.",
        ),
        sign_base.row(
            "scalar_proxy_order_percent_matching_scale_correction_now",
            "pass" if order_percent_matching_scale_correction_now else "reject",
            "scalar-proxy order-percent matching-scale correction now",
            sign_base.truth(order_percent_matching_scale_correction_now),
            "The retained crossing differs from q_star by an order-percent correction scale, not by an order-0.1% micro-shift and not by an order-1 mismatch.",
        ),
        sign_base.row(
            "scalar_proxy_primary_q_exact_over_m0_fixed",
            "pass" if q_exact_exists_now else "reject",
            "scalar-proxy primary q_exact over m0 fixed",
            float(pack["primary_q_exact_over_m0"]) if q_exact_exists_now else -1.0,
            "The first retained target-crossing scale is recorded exactly from the dense alpha(q) curve.",
        ),
        sign_base.row(
            "scalar_proxy_delta_q_over_q_star_fixed",
            "pass" if q_exact_exists_now else "reject",
            "scalar-proxy delta q over q_star fixed",
            float(pack["delta_q_over_q_star"]) if q_exact_exists_now else -1.0,
            "The retained mismatch between q_exact and q_star is recorded exactly as the primary scalar-proxy diagnostic.",
        ),
        sign_base.row(
            "scalar_proxy_alpha_max_over_target_fixed",
            "pass",
            "scalar-proxy alpha max over target fixed",
            float(pack["alpha_max_over_target"]),
            "The dense curve maximum shows whether the audited formula can ever reach alpha_target on the retained interval.",
        ),
        sign_base.row(
            "selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_demoted_to_secondary_now",
            "pass" if source_materialization_numeric_rerun_demoted_to_secondary_now else "reject",
            "selected-extension independent extra-q-range source-materialization numeric rerun demoted to secondary now",
            sign_base.truth(source_materialization_numeric_rerun_demoted_to_secondary_now),
            "Because q_exact exists and lies close to q_star, the higher-value blocker is now scalar-proxy matching-scale redrive, while extra-q source-materialization numeric rerun becomes a secondary lane.",
        ),
        sign_base.row(
            "same_schema_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_replayed_now",
            "pass" if same_schema_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_replayed_now else "reject",
            "same-schema selected-extension independent extra-q-range source-materialization numeric rerun replayed now",
            sign_base.truth(same_schema_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_replayed_now),
            "False means this branch did not spend one more turn replaying the same extra-q numeric surface once the computation gate reclassified the blocker.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "case_label": pack["case_label"],
        "alpha_target": float(pack["alpha_target"]),
        "beta1": float(pack["scalar_ground_state"]["beta_n"]),
        "q_min_over_m0": float(pack["q_min_over_m0"]),
        "q_max_over_m0": float(pack["q_max_over_m0"]),
        "q_count": int(pack["q_count"]),
        "q_blind_over_m0": float(pack["q_blind_over_m0"]),
        "q_star_over_m0": float(pack["q_star_over_m0"]),
        "q_exact_list": [float(value) for value in pack["q_exact_list"]],
        "primary_q_exact_over_m0": float(pack["primary_q_exact_over_m0"]),
        "delta_q_over_m0": float(pack["delta_q_over_m0"]),
        "delta_q_over_q_star": float(pack["delta_q_over_q_star"]),
        "q_exact_matches_prior_blind_crossing_abs_error": float(
            pack["q_exact_matches_prior_blind_crossing_abs_error"]
        ),
        "F_at_q_star": float(pack["F_at_q_star"]),
        "alpha_at_q_star": float(pack["alpha_at_q_star"]),
        "relative_residual_at_q_star": float(pack["relative_residual_at_q_star"]),
        "alpha_max": float(pack["alpha_max"]),
        "q_at_alpha_max": float(pack["q_at_alpha_max"]),
        "alpha_max_over_target": float(pack["alpha_max_over_target"]),
        "exact_scalar_proxy_alpha_q_curve_formula_available_now": computation_gate_pivot_selected_now,
        "exact_scalar_proxy_q_exact_exists_on_retained_interval_now": q_exact_exists_now,
        "exact_scalar_proxy_q_exact_unique_on_retained_interval_now": q_exact_unique_now,
        "exact_scalar_proxy_formula_failure_now": formula_failure_now,
        "exact_scalar_proxy_matching_scale_primary_verdict_available_now": matching_scale_primary_now,
        "scalar_proxy_order_percent_matching_scale_correction_now": order_percent_matching_scale_correction_now,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "computation_gate_pivot_selected_now": computation_gate_pivot_selected_now,
        "selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_demoted_to_secondary_now": source_materialization_numeric_rerun_demoted_to_secondary_now,
        "same_schema_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_replayed_now": same_schema_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_replayed_now,
        "farther_hybrid_continuation_reopen_required_now": farther_hybrid_continuation_reopen_required_now,
        "selected_primary_completion_lane": "updated_pack_scalar_proxy_matching_scale_redrive_audit",
        "selected_secondary_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun",
        "selected_reserve_completion_lane": "farther_hybrid_reserve_only_until_independent_need_reappears",
        "selected_next_generation_route": "trial2_numeric_alpha_scalar_proxy_matching_scale_redrive_audit",
        "recommended_next_route_or_none": "8.7.56.5379",
        "selected_followup_route": "trial2_numeric_alpha_scalar_proxy_alpha_q_curve_gate",
        "selected_followup_route_or_none": "8.7.56.5379",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5377",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "curve_artifacts": {
                "curve_csv": sign_base.display_path(curve_paths["csv"]),
                "curve_pdf": sign_base.display_path(curve_paths["pdf"]),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5379",
                "followup_route": "8.7.56.5383",
            },
        },
        rows,
        summary,
        {
            "overall_status": "scalar_proxy_alpha_q_curve_diagnosis_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "formulae": build_formulae(),
            "curve_samples": pack["curve_samples"],
            "curve_artifacts": {
                "curve_csv": sign_base.display_path(curve_paths["csv"]),
                "curve_pdf": sign_base.display_path(curve_paths["pdf"]),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} scalar-proxy alpha(q) curve diagnosis completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# Function: run the audit when invoked as one CLI script.

if __name__ == "__main__":
    main()
