#!/usr/bin/env python3
"""Generate 8.7.56.5623-.5626 Trial-2 beta-root followup audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_interaction_total_over_harmonic_sq_beta_root_followup_backend import (
    build_trial2_interaction_total_over_harmonic_sq_beta_root_followup_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5619-5622",
        "updated_pack_trial2_interaction_total_over_harmonic_sq_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "83_trial2_numeric_alpha_vector_qball_interaction_total_over_harmonic_sq_beta_root_followup_audit.md"
)

STEP_TAG = "8.7.56.5623-5626"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "interaction_total_over_harmonic_sq beta-root followup audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_interaction_total_over_harmonic_sq_beta_root_followup_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_interaction_total_over_harmonic_sq_exact_relation_audited_"
    "local_beta_root_followup_primary_conditional_hold_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_interaction_total_over_harmonic_sq_beta_root_followup_"
    "target_free_common_root_direct_alpha_gate_next"
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
    """Return whether the beta-root followup note carries the expected claims."""
    patterns = (
        "alpha_qstar",
        "alpha_R8",
        "beta_common_root",
        "target-free",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the target-free common-root audit."""
    return {
        "alpha_qstar": "alpha_qstar(beta) = F_beta(q_star(beta))^2 / (4*pi)",
        "alpha_r8": "alpha_R8(beta) = R8_exact(beta)",
        "selector": "select beta from alpha_qstar(beta) = alpha_R8(beta)",
    }


# 関数: `.5623-.5626` を実行する。

def main() -> None:
    """Execute the Trial-2 beta-root followup audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_interaction_total_over_harmonic_sq_beta_root_followup_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    monotone_now = bool(pack["difference_monotone_increasing_now"])
    single_sign_change_now = int(pack["difference_sign_change_count"]) == 1
    common_root_available_now = bool(pack["common_root_available_now"])
    target_free_beta_selector_available_now = bool(
        pack["target_free_beta_selector_available_now"]
    )
    practical_direct_alpha_closeout_available_now = bool(
        pack["practical_direct_alpha_closeout_available_now"]
    )
    strict_target_free_theorem_closeout_available_now = bool(
        pack["strict_target_free_theorem_closeout_available_now"]
    )
    strict_theorem_followup_required_now = bool(
        pack["strict_theorem_followup_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_interaction_total_over_harmonic_sq_beta_root_followup_route_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 interaction_total_over_harmonic_sq beta-root followup route selected now",
            sign_base.truth(route_selected),
            "The followup starts only from the synchronized exact-relation state where the current blocker is the target-free beta selector itself.",
        ),
        sign_base.row(
            "exact_trial2_interaction_total_over_harmonic_sq_beta_root_followup_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 interaction_total_over_harmonic_sq beta-root followup note available now",
            sign_base.truth(note_available),
            "The note must record alpha_qstar, alpha_R8, and the target-free common-root selector.",
        ),
        sign_base.row(
            "exact_trial2_interaction_total_over_harmonic_sq_common_root_difference_monotone_now",
            "pass" if monotone_now else "reject",
            "exact Trial-2 interaction_total_over_harmonic_sq common-root difference monotone now",
            sign_base.truth(monotone_now),
            "A sampled monotone difference makes the equality selector non-ambiguous on the retained localized beta family.",
        ),
        sign_base.row(
            "exact_trial2_interaction_total_over_harmonic_sq_common_root_single_sign_change_now",
            "pass" if single_sign_change_now else "reject",
            "exact Trial-2 interaction_total_over_harmonic_sq common-root single sign change now",
            sign_base.truth(single_sign_change_now),
            "A single sign change on the retained family is the minimal evidence that the equality selector yields one unique common root.",
        ),
        sign_base.row(
            "exact_trial2_interaction_total_over_harmonic_sq_common_root_available_now",
            "pass" if common_root_available_now else "reject",
            "exact Trial-2 interaction_total_over_harmonic_sq common root available now",
            sign_base.truth(common_root_available_now),
            "The target-free selector only becomes physical if the equality alpha_qstar(beta) = alpha_R8(beta) actually has one root.",
        ),
        sign_base.row(
            "exact_trial2_interaction_total_over_harmonic_sq_target_free_beta_selector_available_now",
            "pass" if target_free_beta_selector_available_now else "reject",
            "exact Trial-2 interaction_total_over_harmonic_sq target-free beta selector available now",
            sign_base.truth(target_free_beta_selector_available_now),
            "Pass means beta is selected by equality of two independent frozen-action readouts rather than by alpha_target comparison.",
        ),
        sign_base.row(
            "exact_trial2_interaction_total_over_harmonic_sq_practical_direct_alpha_closeout_available_now",
            "pass" if practical_direct_alpha_closeout_available_now else "reject",
            "exact Trial-2 interaction_total_over_harmonic_sq practical direct-alpha closeout available now",
            sign_base.truth(practical_direct_alpha_closeout_available_now),
            "The common-root alpha readout should land within one-per-mille of alpha_target before the route can be called a practical closeout.",
        ),
        sign_base.row(
            "exact_trial2_interaction_total_over_harmonic_sq_strict_target_free_theorem_closeout_available_now",
            "pass" if strict_target_free_theorem_closeout_available_now else "reject",
            "exact Trial-2 interaction_total_over_harmonic_sq strict target-free theorem closeout available now",
            sign_base.truth(strict_target_free_theorem_closeout_available_now),
            "Pass would require an analytic theorem for uniqueness, not only the sampled monotone equality selector.",
        ),
        sign_base.row(
            "updated_pack_trial2_target_free_common_root_strict_theorem_followup_primary_next_now",
            "pass" if strict_theorem_followup_required_now else "reject",
            "updated-pack Trial-2 target-free common-root strict-theorem followup primary next now",
            sign_base.truth(strict_theorem_followup_required_now),
            "Because the practical target-free selector is now available while the strict theorem is not, the next honest blocker is the strict-theorem followup.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "prior_retained_beta": float(pack["prior_retained_beta"]),
        "prior_alpha_beta_root": float(pack["prior_alpha_beta_root"]),
        "prior_r8_beta_root": float(pack["prior_r8_beta_root"]),
        "prior_q_exact_over_m0": float(pack["prior_q_exact_over_m0"]),
        "common_root_scan_beta_min": float(pack["scan_beta_min"]),
        "common_root_scan_beta_max": float(pack["scan_beta_max"]),
        "common_root_scan_row_count": int(pack["scan_row_count"]),
        "common_root_difference_monotone_increasing_now": monotone_now,
        "common_root_difference_sign_change_count": int(pack["difference_sign_change_count"]),
        "common_root_difference_first": float(pack["difference_first"]),
        "common_root_difference_last": float(pack["difference_last"]),
        "interaction_total_over_harmonic_sq_beta_common_root": float(pack["beta_common_root"]),
        "interaction_total_over_harmonic_sq_beta_common_root_rel_shift_vs_retained": float(
            pack["beta_common_root_rel_shift_vs_retained"]
        ),
        "interaction_total_over_harmonic_sq_beta_common_root_rel_shift_vs_prior_alpha_beta": float(
            pack["beta_common_root_rel_shift_vs_prior_alpha_beta"]
        ),
        "interaction_total_over_harmonic_sq_beta_common_root_rel_shift_vs_prior_r8_beta_root": float(
            pack["beta_common_root_rel_shift_vs_prior_r8_beta_root"]
        ),
        "interaction_total_over_harmonic_sq_alpha_common_value": float(pack["alpha_common_value"]),
        "interaction_total_over_harmonic_sq_alpha_common_rel_error_vs_target": float(
            pack["alpha_common_rel_error_vs_target"]
        ),
        "interaction_total_over_harmonic_sq_q_star_common_over_m0": float(
            pack["q_star_common_over_m0"]
        ),
        "interaction_total_over_harmonic_sq_q_star_common_rel_shift_vs_q_exact": float(
            pack["q_star_common_rel_shift_vs_q_exact"]
        ),
        "exact_trial2_interaction_total_over_harmonic_sq_target_free_beta_selector_available_now": (
            target_free_beta_selector_available_now
        ),
        "exact_trial2_interaction_total_over_harmonic_sq_practical_direct_alpha_closeout_available_now": (
            practical_direct_alpha_closeout_available_now
        ),
        "exact_trial2_interaction_total_over_harmonic_sq_strict_target_free_theorem_closeout_available_now": (
            strict_target_free_theorem_closeout_available_now
        ),
        "updated_pack_trial2_target_free_common_root_strict_theorem_followup_primary_next_now": (
            strict_theorem_followup_required_now
        ),
    }

    payload = sign_base.payload(
        "8.7.56.5625",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5627",
                "followup_route": "trial2_target_free_common_root_strict_theorem_followup",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_interaction_total_over_harmonic_sq_beta_root_followup_audited",
            "branch_completed": True,
            "breakthrough_passed_now": target_free_beta_selector_available_now,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 beta-root followup audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
