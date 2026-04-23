#!/usr/bin/env python3
"""Generate 8.7.56.5783-.5786 symbolic common-root exactification artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_symbolic_common_root_exactification_backend import (
    build_trial2_symbolic_common_root_exactification_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5779-5782",
        "updated_pack_trial2_beta_sensitivity_source_weighted_full_operator_level_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5783-5786"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "symbolic common-root exactification audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_symbolic_common_root_exactification_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_first_principles_direct_alpha_closure_completed_"
    "full_v2_operator_level_continuum_refinement_completed_"
    "conditional_reopen_only_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_symbolic_common_root_exactification_audited_"
    "invariant_reduction_primary_exact_value_secondary_next"
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


# 関数: audit で使う式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the symbolic selector audit."""
    return {
        "alpha_qstar": "alpha_qstar(beta) = F_beta(q_star(beta))^2 / (4 pi)",
        "r8_exact": (
            "R8(beta) = [exact_cubic_numerator(beta) exact_total_numerator(beta)]"
            " / [36 (1 + beta^2)^2 I2(beta)^2]"
        ),
        "symbolic_selector": (
            "9 (1 + beta^2)^2 J(beta)^2 = pi exact_cubic_numerator(beta)"
            " exact_total_numerator(beta)"
        ),
        "sign_equivalence": (
            "selector_residual(beta) = 36 pi (1 + beta^2)^2 I2(beta)^2"
            " [alpha_qstar(beta) - R8_exact(beta)]"
        ),
    }


# 関数: `.5783-.5786` を実行する。

def main() -> None:
    """Execute the symbolic common-root exactification audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_symbolic_common_root_exactification_pack()
    common_row = pack["common_row"]
    symbolic_root_row = pack["symbolic_root_row"]
    retained_row = pack["retained_row"]
    prior_alpha_row = pack["prior_alpha_row"]
    prior_r8_row = pack["prior_r8_row"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    denominator_cancellation_available = bool(
        pack["exact_symbolic_common_root_denominator_cancellation_available_now"]
    )
    selector_available = bool(pack["exact_symbolic_common_root_selector_available_now"])
    low_negative = bool(pack["local_low_residual_negative_now"])
    high_positive = bool(pack["local_high_residual_positive_now"])
    exact_value_unavailable = not bool(
        pack["exact_symbolic_common_root_closed_form_value_available_now"]
    )
    invariant_reduction_required = bool(
        pack["updated_pack_trial2_invariant_reduction_refresh_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_symbolic_common_root_exactification_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 symbolic common-root exactification selected now",
            sign_base.truth(route_selected),
            "This audit starts only after the exact closed-form goal reset is official and symbolic common-root exactification is the declared primary blocker.",
        ),
        sign_base.row(
            "exact_trial2_symbolic_common_root_denominator_cancellation_available_now",
            "pass" if denominator_cancellation_available else "reject",
            "exact Trial-2 symbolic common-root denominator cancellation available now",
            sign_base.truth(denominator_cancellation_available),
            "Pass means the equality alpha_qstar(beta) = R8(beta) has been rewritten so that the common I2(beta)^2 denominator cancels exactly.",
        ),
        sign_base.row(
            "exact_trial2_symbolic_common_root_selector_available_now",
            "pass" if selector_available else "reject",
            "exact Trial-2 symbolic common-root selector available now",
            sign_base.truth(selector_available),
            "Pass means the retained common-root selector is now carried by one exact symbolic equation and one locally exactified symbolic root rather than by a purely numerical Delta_common(beta) root-search statement.",
        ),
        sign_base.row(
            "exact_trial2_symbolic_common_root_local_sign_support_now",
            "pass" if low_negative and high_positive else "reject",
            "exact Trial-2 symbolic common-root local sign support now",
            sign_base.truth(low_negative and high_positive),
            "The exactified selector residual keeps the same local sign orientation as Delta_common(beta): negative below the retained root and positive above it.",
        ),
        sign_base.row(
            "exact_trial2_symbolic_common_root_exact_value_unavailable_now",
            "pass" if exact_value_unavailable else "reject",
            "exact Trial-2 symbolic common-root exact value unavailable now",
            sign_base.truth(exact_value_unavailable),
            "This audit exactifies the selector equation only. It does not yet reduce the profile functionals to the finite invariant algebra required for alpha = 1/137 in residual-free closed form.",
        ),
        sign_base.row(
            "updated_pack_trial2_invariant_reduction_refresh_required_now",
            "pass" if invariant_reduction_required else "reject",
            "updated-pack Trial-2 invariant reduction refresh required now",
            sign_base.truth(invariant_reduction_required),
            "Once the selector equation is exact, the next honest blocker is invariant reduction rather than further selector-search replay.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(pack["beta_common_root"]),
        "beta_symbolic_root": float(pack["beta_symbolic_root"]),
        "beta_symbolic_root_rel_shift_vs_prior_common_root": float(
            pack["beta_symbolic_root_rel_shift_vs_prior_common_root"]
        ),
        "alpha_target": float(pack["alpha_target"]),
        "retained_beta_selector_residual": float(retained_row["selector_residual"]),
        "prior_alpha_beta_selector_residual": float(prior_alpha_row["selector_residual"]),
        "prior_r8_beta_selector_residual": float(prior_r8_row["selector_residual"]),
        "common_root_selector_lhs": float(common_row["selector_lhs"]),
        "common_root_selector_rhs": float(common_row["selector_rhs"]),
        "common_root_selector_residual": float(common_row["selector_residual"]),
        "common_root_selector_residual_abs": float(common_row["selector_residual_abs"]),
        "common_root_selector_residual_rel": float(common_row["selector_residual_rel"]),
        "common_root_positive_factor": float(common_row["positive_factor"]),
        "common_root_symbolic_j_beta": float(common_row["j_beta"]),
        "common_root_alpha_qstar": float(common_row["alpha_qstar"]),
        "common_root_alpha_r8_exact": float(common_row["alpha_r8_exact"]),
        "common_root_selector_weighted_eom_consistency_residual": float(
            common_row["selector_weighted_eom_consistency_residual"]
        ),
        "common_root_selector_integral_consistency_residual": float(
            common_row["selector_integral_consistency_residual"]
        ),
        "symbolic_root_selector_residual_abs": float(symbolic_root_row["selector_residual_abs"]),
        "symbolic_root_alpha_qstar": float(symbolic_root_row["alpha_qstar"]),
        "symbolic_root_alpha_r8_exact": float(symbolic_root_row["alpha_r8_exact"]),
        "local_low_selector_residual": float(pack["local_rows"][0]["selector_residual"]),
        "local_high_selector_residual": float(pack["local_rows"][-1]["selector_residual"]),
        "exact_trial2_symbolic_common_root_denominator_cancellation_available_now": bool(
            denominator_cancellation_available
        ),
        "exact_trial2_symbolic_common_root_selector_available_now": bool(
            selector_available
        ),
        "exact_trial2_symbolic_common_root_closed_form_value_available_now": False,
        "updated_pack_trial2_invariant_reduction_refresh_required_now": bool(
            invariant_reduction_required
        ),
        "selected_next_generation_route": "trial2_invariant_reduction_audit",
        "recommended_next_route_or_none": ".5791-.5794",
        "selected_followup_route": "trial2_symbolic_common_root_gate",
        "selected_followup_route_or_none": ".5787-.5790",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5785",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_symbolic_common_root_exactification_audited",
            "branch_completed": True,
            "breakthrough_passed_now": selector_available,
            "physical_reject_required": False,
        },
        {
            "beta_common_root": float(pack["beta_common_root"]),
            "beta_symbolic_root": float(pack["beta_symbolic_root"]),
            "symbolic_root_selector_residual_abs": float(symbolic_root_row["selector_residual_abs"]),
            "common_root_positive_factor": float(common_row["positive_factor"]),
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5783-5786 Trial-2 symbolic common-root exactification audit completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
