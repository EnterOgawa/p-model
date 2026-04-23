#!/usr/bin/env python3
"""Generate 8.7.56.5695-.5698 Trial-2 uniqueness-anchor followup artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_beta_sensitivity_uniqueness_anchor_followup_backend import (
    build_trial2_beta_sensitivity_uniqueness_anchor_followup_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5691-5694",
        "updated_pack_trial2_beta_sensitivity_derivative_chain_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "93_trial2_numeric_alpha_vector_qball_beta_sensitivity_uniqueness_anchor_followup_audit.md"
)

STEP_TAG = "8.7.56.5695-5698"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "beta-sensitivity uniqueness-anchor followup audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_uniqueness_anchor_followup_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_derivative_chain_sign_support_completed_"
    "uniqueness_anchor_followup_primary_conditional_hold_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_uniqueness_anchor_audited_"
    "final_closure_gate_next"
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


# 関数: audit note が expected claims を含むか確認する。

def note_contains_audit(text: str) -> bool:
    """Return whether the uniqueness-anchor note carries the expected claims."""
    patterns = (
        "lower anchor",
        "upper anchor",
        "final closure",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the uniqueness-anchor audit."""
    return {
        "difference": "Delta_common(beta) = alpha_qstar(beta) - R8(beta)",
        "anchors": "Delta_common(beta_lower) < 0 < Delta_common(beta_upper)",
        "selector": "beta_common_root in (beta_lower, beta_upper)",
    }


# 関数: `.5695-.5698` を実行する。

def main() -> None:
    """Execute the Trial-2 beta-sensitivity uniqueness-anchor followup audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_beta_sensitivity_uniqueness_anchor_followup_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    lower_anchor_negative_now = bool(pack["lower_anchor_negative_now"])
    upper_anchor_positive_now = bool(pack["upper_anchor_positive_now"])
    common_root_inside_anchor_interval_now = bool(
        pack["common_root_inside_anchor_interval_now"]
    )
    sampled_selector_monotone_now = bool(pack["sampled_selector_monotone_now"])
    sampled_selector_single_sign_change_now = bool(
        pack["sampled_selector_single_sign_change_now"]
    )
    local_delta_derivative_positive_now = bool(
        pack["local_delta_derivative_positive_now"]
    )
    uniqueness_anchor_support_available_now = bool(
        pack["uniqueness_anchor_support_available_now"]
    )
    exact_uniqueness_anchor_theorem_available_now = bool(
        pack["exact_trial2_beta_sensitivity_uniqueness_anchor_theorem_available_now"]
    )
    final_closure_followup_required_now = bool(
        pack["updated_pack_trial2_beta_sensitivity_final_closure_followup_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_beta_sensitivity_uniqueness_anchor_followup_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 beta-sensitivity uniqueness-anchor followup selected now",
            sign_base.truth(route_selected),
            "This branch starts only after derivative-chain sign support is already synchronized as the current retained surface.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_uniqueness_anchor_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 beta-sensitivity uniqueness-anchor note available now",
            sign_base.truth(note_available),
            "The note must record the lower / upper anchors, the retained common root inside the interval, and the remaining final-closure blocker.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_lower_anchor_negative_now",
            "pass" if lower_anchor_negative_now else "reject",
            "exact Trial-2 beta-sensitivity lower anchor negative now",
            sign_base.truth(lower_anchor_negative_now),
            "The uniqueness anchor requires one retained lower beta with Delta_common(beta) < 0.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_upper_anchor_positive_now",
            "pass" if upper_anchor_positive_now else "reject",
            "exact Trial-2 beta-sensitivity upper anchor positive now",
            sign_base.truth(upper_anchor_positive_now),
            "The uniqueness anchor requires one retained upper beta with Delta_common(beta) > 0.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_common_root_inside_anchor_interval_now",
            "pass" if common_root_inside_anchor_interval_now else "reject",
            "exact Trial-2 beta-sensitivity common root inside anchor interval now",
            sign_base.truth(common_root_inside_anchor_interval_now),
            "The retained common root must actually lie between the lower and upper sign anchors.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_sampled_selector_monotone_now",
            "pass" if sampled_selector_monotone_now else "reject",
            "exact Trial-2 beta-sensitivity sampled selector monotone now",
            sign_base.truth(sampled_selector_monotone_now),
            "The retained sampled family should keep Delta_common(beta) monotone increasing on the localized branch.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_sampled_selector_single_sign_change_now",
            "pass" if sampled_selector_single_sign_change_now else "reject",
            "exact Trial-2 beta-sensitivity sampled selector single sign change now",
            sign_base.truth(sampled_selector_single_sign_change_now),
            "A single sampled sign change is the retained non-ambiguity witness for the common-root selector.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_local_delta_derivative_positive_now",
            "pass" if local_delta_derivative_positive_now else "reject",
            "exact Trial-2 beta-sensitivity local Delta_common derivative positive now",
            sign_base.truth(local_delta_derivative_positive_now),
            "The uniqueness-anchor support still needs the already retained local transversality layer d Delta_common / d beta > 0.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_uniqueness_anchor_support_available_now",
            "pass" if uniqueness_anchor_support_available_now else "reject",
            "exact Trial-2 beta-sensitivity uniqueness-anchor support available now",
            sign_base.truth(uniqueness_anchor_support_available_now),
            "Pass means the retained anchors, sampled selector, and local transversality now live on one synchronized support surface.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_uniqueness_anchor_theorem_available_now",
            "pass" if exact_uniqueness_anchor_theorem_available_now else "reject",
            "exact Trial-2 beta-sensitivity uniqueness-anchor theorem available now",
            sign_base.truth(exact_uniqueness_anchor_theorem_available_now),
            "This audit still stops short of the final strict theorem; it only fixes uniqueness-anchor support as the remaining near-closure layer.",
        ),
        sign_base.row(
            "updated_pack_trial2_beta_sensitivity_final_closure_followup_required_now",
            "pass" if final_closure_followup_required_now else "reject",
            "updated-pack Trial-2 beta-sensitivity final closure followup required now",
            sign_base.truth(final_closure_followup_required_now),
            "Once uniqueness-anchor support is synchronized honestly, the only remaining blocker is one final closure verdict that decides whether the theorem can be declared complete.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_anchor_lower": float(pack["beta_anchor_lower"]),
        "beta_anchor_upper": float(pack["beta_anchor_upper"]),
        "delta_common_lower_anchor": float(pack["delta_common_lower_anchor"]),
        "delta_common_upper_anchor": float(pack["delta_common_upper_anchor"]),
        "sampled_anchor_gap_span": float(pack["sampled_anchor_gap_span"]),
        "lower_anchor_abs_margin": float(pack["lower_anchor_abs_margin"]),
        "upper_anchor_abs_margin": float(pack["upper_anchor_abs_margin"]),
        "beta_common_root": float(pack["beta_common_root"]),
        "alpha_common_value": float(pack["alpha_common_value"]),
        "alpha_common_rel_error_vs_target": float(
            pack["alpha_common_rel_error_vs_target"]
        ),
        "derivative_transversality_min": float(
            pack["derivative_transversality_min"]
        ),
        "derivative_transversality_max": float(
            pack["derivative_transversality_max"]
        ),
        "exact_trial2_beta_sensitivity_uniqueness_anchor_support_available_now": bool(
            uniqueness_anchor_support_available_now
        ),
        "exact_trial2_beta_sensitivity_uniqueness_anchor_theorem_available_now": bool(
            exact_uniqueness_anchor_theorem_available_now
        ),
        "updated_pack_trial2_beta_sensitivity_final_closure_followup_required_now": bool(
            final_closure_followup_required_now
        ),
        "selected_next_generation_route": (
            "trial2_beta_sensitivity_final_closure_followup"
        ),
        "recommended_next_route_or_none": (
            "trial2_beta_sensitivity_final_closure_followup"
        ),
    }

    payload = sign_base.payload(
        "8.7.56.5697",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
            },
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_beta_sensitivity_uniqueness_anchor_followup_completed",
            "branch_completed": True,
            "breakthrough_passed_now": uniqueness_anchor_support_available_now,
            "physical_reject_required": False,
        },
        {
            "common_root_anchor_interval": {
                "beta_lower": pack["beta_anchor_lower"],
                "beta_upper": pack["beta_anchor_upper"],
                "delta_lower": pack["delta_common_lower_anchor"],
                "delta_upper": pack["delta_common_upper_anchor"],
            }
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print(
        "[done] 8.7.56.5695-5698 Trial-2 beta-sensitivity uniqueness-anchor audit completed"
    )
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
