#!/usr/bin/env python3
"""Generate 8.7.56.5791-.5794 invariant-reduction artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_invariant_reduction_backend import (
    build_trial2_invariant_reduction_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5787-5790",
        "updated_pack_trial2_symbolic_common_root_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5791-5794"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "invariant reduction audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_invariant_reduction_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_symbolic_common_root_exactification_completed_"
    "invariant_reduction_primary_exact_value_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_invariant_reduction_audited_"
    "exact_alpha_extraction_primary_zero_residual_secondary_next"
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


# 関数: invariant reduction の式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the invariant-reduction audit."""
    return {
        "reduced_invariants": (
            "f(beta) = J(beta)/I2(beta), g(beta) = Ig(beta)/I2(beta), "
            "q(beta) = I4(beta)/I2(beta), b(beta) = B(beta)/I2(beta)"
        ),
        "reduced_selector": (
            "9 (1 + beta^2)^2 f(beta)^2 = pi [4(g + eps - b) - q] "
            "[2(5 + beta^2) + 10 g - q - 4 b]"
        ),
        "reduced_alpha": (
            "alpha(beta) = [4(g + eps - b) - q] "
            "[2(5 + beta^2) + 10 g - q - 4 b] / [36 (1 + beta^2)^2]"
        ),
    }


# 関数: `.5791-.5794` を実行する。

def main() -> None:
    """Execute the invariant-reduction audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_invariant_reduction_pack()
    retained_row = pack["retained_row"]
    symbolic_row = pack["symbolic_row"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    invariant_reduction_available = bool(
        pack["exact_trial2_selector_invariant_reduction_available_now"]
    )
    retained_continuity_support = bool(
        pack["exact_trial2_retained_continuity_support_available_now"]
    )
    finite_algebra_available = bool(
        pack["exact_trial2_finite_invariant_algebra_available_now"]
    )
    exact_alpha_form_available = bool(
        pack["exact_trial2_finite_invariant_alpha_form_available_now"]
    )
    zero_residual_unavailable = not bool(
        pack["exact_trial2_zero_residual_closed_form_available_now"]
    )
    exact_alpha_extraction_required = bool(
        pack["updated_pack_trial2_exact_alpha_extraction_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_invariant_reduction_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 invariant reduction selected now",
            sign_base.truth(route_selected),
            "This audit starts only after symbolic common-root exactification is completed and invariant reduction is the declared live blocker.",
        ),
        sign_base.row(
            "exact_trial2_selector_invariant_reduction_available_now",
            "pass" if invariant_reduction_available else "reject",
            "exact Trial-2 selector invariant reduction available now",
            sign_base.truth(invariant_reduction_available),
            "Pass means the exact selector has been rewritten without raw I2(beta)^2 scale dependence and now lives on reduced beta-native invariants only at the exact symbolic root.",
        ),
        sign_base.row(
            "exact_trial2_retained_continuity_support_available_now",
            "pass" if retained_continuity_support else "reject",
            "exact Trial-2 retained continuity support available now",
            sign_base.truth(retained_continuity_support),
            "The retained common-root row is allowed one small continuity gap relative to the exact symbolic root; this row checks that the gap stays sub-ppm and therefore does not break the reduced-invariant reading.",
        ),
        sign_base.row(
            "exact_trial2_finite_invariant_algebra_available_now",
            "pass" if finite_algebra_available else "reject",
            "exact Trial-2 finite invariant algebra available now",
            sign_base.truth(finite_algebra_available),
            "Pass means one finite invariant tuple (f, g, q, b) is sufficient to carry the exact selector relation.",
        ),
        sign_base.row(
            "exact_trial2_finite_invariant_alpha_form_available_now",
            "pass" if exact_alpha_form_available else "reject",
            "exact Trial-2 finite invariant alpha form available now",
            sign_base.truth(exact_alpha_form_available),
            "Pass means alpha itself is now carried by one exact finite-invariant expression rather than by separate form-factor and energy-partition readouts.",
        ),
        sign_base.row(
            "exact_trial2_zero_residual_closed_form_unavailable_now",
            "pass" if zero_residual_unavailable else "reject",
            "exact Trial-2 zero-residual closed form unavailable now",
            sign_base.truth(zero_residual_unavailable),
            "Invariant reduction alone does not yet collapse the exact alpha formula to the constant target 1/137 with zero residual.",
        ),
        sign_base.row(
            "updated_pack_trial2_exact_alpha_extraction_required_now",
            "pass" if exact_alpha_extraction_required else "reject",
            "updated-pack Trial-2 exact-alpha extraction required now",
            sign_base.truth(exact_alpha_extraction_required),
            "Once the finite invariant algebra is fixed, the next honest blocker is constant extraction rather than further selector reduction replay.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(pack["beta_common_root"]),
        "beta_symbolic_root": float(pack["beta_symbolic_root"]),
        "retained_f_beta": float(retained_row["f_beta"]),
        "retained_g_beta": float(retained_row["g_beta"]),
        "retained_q_beta": float(retained_row["q_beta"]),
        "retained_b_beta": float(retained_row["b_beta"]),
        "retained_alpha_from_reduced_invariants": float(
            retained_row["alpha_from_reduced_invariants"]
        ),
        "retained_alpha_reduced_minus_form_factor": float(
            retained_row["alpha_reduced_minus_form_factor"]
        ),
        "retained_selector_reduced_residual_abs": float(
            retained_row["selector_reduced_residual_abs"]
        ),
        "symbolic_alpha_from_reduced_invariants": float(
            symbolic_row["alpha_from_reduced_invariants"]
        ),
        "symbolic_selector_reduced_residual_abs": float(
            symbolic_row["selector_reduced_residual_abs"]
        ),
        "exact_trial2_selector_invariant_reduction_available_now": bool(
            invariant_reduction_available
        ),
        "exact_trial2_retained_continuity_support_available_now": bool(
            retained_continuity_support
        ),
        "exact_trial2_finite_invariant_algebra_available_now": bool(
            finite_algebra_available
        ),
        "exact_trial2_finite_invariant_alpha_form_available_now": bool(
            exact_alpha_form_available
        ),
        "exact_trial2_zero_residual_closed_form_available_now": False,
        "updated_pack_trial2_exact_alpha_extraction_required_now": bool(
            exact_alpha_extraction_required
        ),
        "selected_next_generation_route": "trial2_exact_alpha_closed_form_extraction_audit",
        "recommended_next_route_or_none": ".5795-.5798",
        "selected_followup_route": "trial2_zero_residual_final_theorem_gate",
        "selected_followup_route_or_none": ".5799-.5802",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5793",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_invariant_reduction_audited",
            "branch_completed": True,
            "breakthrough_passed_now": finite_algebra_available,
            "physical_reject_required": False,
        },
        {
            "beta_common_root": float(pack["beta_common_root"]),
            "beta_symbolic_root": float(pack["beta_symbolic_root"]),
            "retained_alpha_from_reduced_invariants": float(
                retained_row["alpha_from_reduced_invariants"]
            ),
            "symbolic_selector_reduced_residual_abs": float(
                symbolic_row["selector_reduced_residual_abs"]
            ),
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5791-5794 Trial-2 invariant reduction audit completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
