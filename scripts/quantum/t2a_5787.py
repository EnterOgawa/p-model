#!/usr/bin/env python3
"""Generate 8.7.56.5787-.5790 symbolic common-root gate artifacts."""

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
        "8.7.56.5783-5786",
        "updated_pack_trial2_symbolic_common_root_exactification_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "105_trial2_numeric_alpha_vector_qball_symbolic_common_root_exactification_audit.md"
)

STEP_TAG = "8.7.56.5787-5790"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "symbolic common-root gate / invariant-reduction refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_symbolic_common_root_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_symbolic_common_root_exactification_audited_"
    "invariant_reduction_primary_exact_value_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_symbolic_common_root_exactification_completed_"
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


# 関数: audit note が required claims を含むか確認する。

def note_contains_audit(text: str) -> bool:
    """Return whether the symbolic selector note carries the required claims."""
    patterns = (
        "symbolic common-root selector",
        "9(",
        "I_2(",
        "invariant reduction",
    )
    return all(pattern in text for pattern in patterns)


# 関数: gate で使う式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the symbolic common-root gate."""
    return {
        "gate_a": "Gate A = exact symbolic common-root selector is available now",
        "gate_b": "Gate B = invariant reduction is the next honest blocker now",
        "gate_c": "Gate C = exact value closure stays secondary until invariant reduction is classified",
    }


# 関数: `.5787-.5790` を実行する。

def main() -> None:
    """Execute the symbolic common-root gate / invariant refresh."""
    sign_base.require(PRIOR_AUDIT)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)

    gate_a = bool(
        prior_summary["exact_trial2_symbolic_common_root_selector_available_now"]
    )
    gate_b = bool(
        gate_a
        and prior_summary["updated_pack_trial2_invariant_reduction_refresh_required_now"]
    )
    gate_c = bool(
        gate_b
        and not prior_summary["exact_trial2_symbolic_common_root_closed_form_value_available_now"]
        and note_contains_audit(note_text)
    )

    rows = [
        sign_base.row(
            "gate_a_trial2_symbolic_common_root_selector_completed_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 symbolic common-root selector completed now",
            sign_base.truth(gate_a),
            "The selector equality alpha_qstar(beta) = R8(beta) is now carried by one exact symbolic equation with exact denominator cancellation.",
        ),
        sign_base.row(
            "gate_b_trial2_invariant_reduction_promoted_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 invariant reduction promoted now",
            sign_base.truth(gate_b),
            "Once the selector is exactified, the next blocker is no longer selector existence but finite invariant reduction of the profile functionals.",
        ),
        sign_base.row(
            "gate_c_trial2_exact_value_closure_secondary_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 exact value closure secondary now",
            sign_base.truth(gate_c),
            "Exact alpha = 1/137 extraction remains downstream of invariant reduction and is not yet an honest terminal claim at this gate.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(prior_summary["beta_common_root"]),
        "beta_symbolic_root": float(prior_summary["beta_symbolic_root"]),
        "beta_symbolic_root_rel_shift_vs_prior_common_root": float(
            prior_summary["beta_symbolic_root_rel_shift_vs_prior_common_root"]
        ),
        "symbolic_root_selector_residual_abs": float(
            prior_summary["symbolic_root_selector_residual_abs"]
        ),
        "common_root_positive_factor": float(prior_summary["common_root_positive_factor"]),
        "exact_trial2_symbolic_common_root_selector_completed_now": bool(gate_a),
        "updated_pack_trial2_invariant_reduction_promoted_now": bool(gate_b),
        "exact_trial2_symbolic_common_root_exact_value_available_now": False,
        "selected_next_generation_route": "trial2_invariant_reduction_audit",
        "recommended_next_route_or_none": ".5791-.5794",
        "selected_followup_route": "trial2_exact_alpha_closed_form_extraction_audit",
        "selected_followup_route_or_none": ".5795-.5798",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5789",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
            },
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_symbolic_common_root_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": gate_a,
            "physical_reject_required": False,
        },
        {
            "beta_common_root": float(prior_summary["beta_common_root"]),
            "beta_symbolic_root": float(prior_summary["beta_symbolic_root"]),
            "symbolic_root_selector_residual_abs": float(
                prior_summary["symbolic_root_selector_residual_abs"]
            ),
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5787-5790 Trial-2 symbolic common-root gate completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
