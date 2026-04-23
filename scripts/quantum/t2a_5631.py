#!/usr/bin/env python3
"""Generate 8.7.56.5631-.5634 Trial-2 strict-theorem followup audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_target_free_common_root_strict_theorem_followup_backend import (
    build_trial2_target_free_common_root_strict_theorem_followup_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5627-5630",
        "updated_pack_trial2_interaction_total_over_harmonic_sq_beta_root_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "84_trial2_numeric_alpha_vector_qball_target_free_common_root_strict_theorem_followup_audit.md"
)

STEP_TAG = "8.7.56.5631-5634"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "target-free common-root strict-theorem followup audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_target_free_common_root_strict_theorem_followup_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_target_free_common_root_direct_alpha_audited_practical_closeout_"
    "strict_theorem_followup_primary_conditional_hold_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_target_free_common_root_strict_theorem_followup_audited_"
    "negative_closeout_gate_next"
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
    """Return whether the strict-theorem followup note carries the expected claims."""
    patterns = (
        "difference_derivative",
        "strict theorem",
        "common-root",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the strict-theorem followup audit."""
    return {
        "selector": "Delta_common(beta) = alpha_qstar(beta) - alpha_R8(beta)",
        "root": "select beta from Delta_common(beta) = 0",
        "local_support": "Delta_common'(beta_common) > 0 from stable central differences",
        "verdict": (
            "Strong local numerical transversality support exists, but no exact "
            "monotone / uniqueness theorem is yet materialized."
        ),
    }


# 関数: `.5631-.5634` を実行する。

def main() -> None:
    """Execute the Trial-2 strict-theorem followup audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_target_free_common_root_strict_theorem_followup_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    local_transversality_support_available_now = bool(
        pack["local_transversality_support_available_now"]
    )
    exact_alpha_qstar_monotone_theorem_available_now = bool(
        pack["exact_alpha_qstar_monotone_theorem_available_now"]
    )
    exact_alpha_r8_monotone_theorem_available_now = bool(
        pack["exact_alpha_r8_monotone_theorem_available_now"]
    )
    exact_common_root_uniqueness_theorem_available_now = bool(
        pack["exact_common_root_uniqueness_theorem_available_now"]
    )
    strict_theorem_negative_closeout_available_now = bool(
        pack["strict_theorem_negative_closeout_available_now"]
    )
    conditional_hold_restored_primary_now = bool(
        pack["conditional_hold_restored_primary_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_target_free_common_root_strict_theorem_followup_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 target-free common-root strict-theorem followup selected now",
            sign_base.truth(route_selected),
            "The audit starts only after the practical target-free common-root direct-alpha closeout has been synchronized.",
        ),
        sign_base.row(
            "exact_trial2_target_free_common_root_strict_theorem_followup_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 target-free common-root strict-theorem followup note available now",
            sign_base.truth(note_available),
            "The note must record local derivative support and the non-materialized theorem surface explicitly.",
        ),
        sign_base.row(
            "exact_trial2_target_free_common_root_local_transversality_support_available_now",
            "pass" if local_transversality_support_available_now else "reject",
            "exact Trial-2 target-free common-root local transversality support available now",
            sign_base.truth(local_transversality_support_available_now),
            "Pass means alpha_qstar rises, alpha_R8 falls, and Delta_common'(beta_common) stays positive under stable central differences.",
        ),
        sign_base.row(
            "exact_trial2_target_free_common_root_alpha_qstar_monotone_theorem_available_now",
            "pass" if exact_alpha_qstar_monotone_theorem_available_now else "reject",
            "exact Trial-2 target-free common-root alpha_qstar monotone theorem available now",
            sign_base.truth(exact_alpha_qstar_monotone_theorem_available_now),
            "A strict theorem would require one analytic monotone law for alpha_qstar(beta), not only sampled derivative evidence.",
        ),
        sign_base.row(
            "exact_trial2_target_free_common_root_alpha_r8_monotone_theorem_available_now",
            "pass" if exact_alpha_r8_monotone_theorem_available_now else "reject",
            "exact Trial-2 target-free common-root alpha_R8 monotone theorem available now",
            sign_base.truth(exact_alpha_r8_monotone_theorem_available_now),
            "The exact R8 relation exists, but its monotone sign law is not yet promoted as an analytic theorem in the current pack.",
        ),
        sign_base.row(
            "exact_trial2_target_free_common_root_uniqueness_theorem_available_now",
            "pass" if exact_common_root_uniqueness_theorem_available_now else "reject",
            "exact Trial-2 target-free common-root uniqueness theorem available now",
            sign_base.truth(exact_common_root_uniqueness_theorem_available_now),
            "Strict theorem closeout would require an analytic uniqueness statement for the common-root selector, not only sampled sign-change uniqueness.",
        ),
        sign_base.row(
            "exact_trial2_target_free_common_root_strict_theorem_negative_closeout_available_now",
            "pass" if strict_theorem_negative_closeout_available_now else "reject",
            "exact Trial-2 target-free common-root strict-theorem negative closeout available now",
            sign_base.truth(strict_theorem_negative_closeout_available_now),
            "The honest verdict is negative once local numerical support is strong but the exact uniqueness theorem still does not materialize.",
        ),
        sign_base.row(
            "updated_pack_trial2_conditional_hold_restored_primary_now",
            "pass" if conditional_hold_restored_primary_now else "reject",
            "updated-pack Trial-2 conditional hold restored primary now",
            sign_base.truth(conditional_hold_restored_primary_now),
            "After the strict-theorem followup dead-ends honestly, the pack returns to conditional hold with no unconditional next branch.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "interaction_total_over_harmonic_sq_beta_common_root": float(
            pack["beta_common_root"]
        ),
        "interaction_total_over_harmonic_sq_alpha_common_value": float(
            pack["alpha_common_value"]
        ),
        "interaction_total_over_harmonic_sq_alpha_common_rel_error_vs_target": float(
            pack["alpha_common_rel_error_vs_target"]
        ),
        "common_root_difference_derivative_min": float(pack["difference_derivative_min"]),
        "common_root_difference_derivative_max": float(pack["difference_derivative_max"]),
        "common_root_difference_derivative_rel_spread": float(
            pack["difference_derivative_rel_spread"]
        ),
        "common_root_alpha_qstar_derivative_min": float(
            pack["alpha_qstar_derivative_min"]
        ),
        "common_root_alpha_qstar_derivative_max": float(
            pack["alpha_qstar_derivative_max"]
        ),
        "common_root_alpha_r8_derivative_min": float(pack["alpha_r8_derivative_min"]),
        "common_root_alpha_r8_derivative_max": float(pack["alpha_r8_derivative_max"]),
        "exact_trial2_target_free_common_root_local_transversality_support_available_now": (
            local_transversality_support_available_now
        ),
        "exact_trial2_target_free_common_root_alpha_qstar_monotone_theorem_available_now": (
            exact_alpha_qstar_monotone_theorem_available_now
        ),
        "exact_trial2_target_free_common_root_alpha_r8_monotone_theorem_available_now": (
            exact_alpha_r8_monotone_theorem_available_now
        ),
        "exact_trial2_target_free_common_root_uniqueness_theorem_available_now": (
            exact_common_root_uniqueness_theorem_available_now
        ),
        "exact_trial2_target_free_common_root_strict_theorem_negative_closeout_available_now": (
            strict_theorem_negative_closeout_available_now
        ),
        "updated_pack_trial2_conditional_hold_restored_primary_now": (
            conditional_hold_restored_primary_now
        ),
        "updated_pack_trial2_no_unconditional_next_official_branch_now": bool(
            pack["no_unconditional_next_official_branch_now"]
        ),
    }

    payload = sign_base.payload(
        "8.7.56.5633",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5635",
                "followup_route": "trial2_conditional_hold_only",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_target_free_common_root_strict_theorem_followup_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 strict-theorem followup audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
