#!/usr/bin/env python3
"""Generate 8.7.56.5735-.5738 patched-tail pure-continuum closure audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_beta_sensitivity_patched_tail_pure_continuum_closure_refresh_backend import (
    build_trial2_beta_sensitivity_patched_tail_pure_continuum_closure_refresh_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_FINAL_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5707-5710",
        "updated_pack_trial2_beta_sensitivity_final_closure_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_PATCHED_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5731-5734",
        "updated_pack_trial2_beta_sensitivity_patched_tail_remainder_bound_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "98_trial2_numeric_alpha_vector_qball_patched_tail_pure_continuum_closure_refresh_audit.md"
)

STEP_TAG = "8.7.56.5735-5738"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "patched-tail pure-continuum closure refresh audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_patched_tail_pure_continuum_closure_refresh_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_admissible_positive_decay_tail_patch_analytic_remainder_bound_"
    "completed_pure_continuum_closure_refresh_primary_conditional_reopen_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_admissible_positive_decay_tail_patch_pure_continuum_closure_"
    "audited_v2_wording_gate_next"
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


# 関数: note が closure-refresh claims を含むか確認する。

def note_contains_refresh(text: str) -> bool:
    """Return whether the audit note carries the expected closure-refresh claims."""
    patterns = (
        "pure-continuum",
        "weighted-integral",
        "operator-level continuum refinement",
        "v3.0",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the patched-tail pure-continuum closure refresh."""
    return {
        "selector": "alpha_q*(beta) = F_beta(q*(beta))^2 / (4*pi) = R8(beta)",
        "patched_tail_closure": (
            "dI_n/dbeta on [0,+inf) = dI_n/dbeta on [0,X] + R_n^(tail)(X), with explicit |R_n^(tail)(X)| upper bound"
        ),
        "wording_refresh": (
            "first-principles direct-alpha closure completed + patched-tail pure-continuum weighted-integral closure completed + full operator-level continuum refinement deferred to v3.0"
        ),
    }


# 関数: `.5735-.5738` を実行する。

def main() -> None:
    """Execute the patched-tail pure-continuum closure refresh audit."""
    sign_base.require(PRIOR_FINAL_GATE)
    sign_base.require(PRIOR_PATCHED_GATE)
    sign_base.require(AUDIT_NOTE)

    final_summary = sign_base.read_json(PRIOR_FINAL_GATE)["summary"]
    patched_summary = sign_base.read_json(PRIOR_PATCHED_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = (
        build_trial2_beta_sensitivity_patched_tail_pure_continuum_closure_refresh_pack()
    )

    route_selected = (
        str(patched_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_refresh(note_text)

    rows = [
        sign_base.row(
            "updated_pack_trial2_patched_tail_pure_continuum_refresh_route_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 patched-tail pure-continuum refresh route selected now",
            sign_base.truth(route_selected),
            "This refresh starts only after the patched-tail analytic remainder bound is already official and the live blocker has reduced to wording closure.",
        ),
        sign_base.row(
            "exact_trial2_patched_tail_pure_continuum_refresh_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 patched-tail pure-continuum refresh note available now",
            sign_base.truth(note_available),
            "The note must explicitly state that one patched weighted-integral continuum layer is now closed, while the full operator-level continuum theorem remains open.",
        ),
        sign_base.row(
            "exact_trial2_first_principles_direct_alpha_closure_completed_now",
            "pass"
            if pack["exact_trial2_first_principles_direct_alpha_closure_completed_now"]
            else "reject",
            "exact Trial-2 first-principles direct-alpha closure completed now",
            sign_base.truth(
                pack["exact_trial2_first_principles_direct_alpha_closure_completed_now"]
            ),
            "The closure refresh must preserve the already-fixed direct-alpha verdict rather than reopen it.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_promotion_available_now",
            "pass"
            if pack[
                "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_promotion_available_now"
            ]
            else "reject",
            "exact Trial-2 beta-sensitivity patched-tail pure-continuum promotion available now",
            sign_base.truth(
                pack[
                    "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_promotion_available_now"
                ]
            ),
            "Pass means the admissible patched tail plus explicit remainder bound removes the finite-cutoff loophole at the weighted-integral level.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_closure_completed_now",
            "pass"
            if pack[
                "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_closure_completed_now"
            ]
            else "reject",
            "exact Trial-2 beta-sensitivity patched-tail pure-continuum closure completed now",
            sign_base.truth(
                pack[
                    "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_closure_completed_now"
                ]
            ),
            "Pass means the weighted-integral continuum statement is no longer merely finite-cutoff support and can be folded back into the theorem wording honestly.",
        ),
        sign_base.row(
            "exact_trial2_v2_theorem_wording_upgrade_available_now",
            "pass"
            if pack["exact_trial2_v2_theorem_wording_upgrade_available_now"]
            else "reject",
            "exact Trial-2 v2 theorem wording upgrade available now",
            sign_base.truth(pack["exact_trial2_v2_theorem_wording_upgrade_available_now"]),
            "Once the patched weighted-integral continuum layer is closed, v2 wording can distinguish it from the still-open operator-level refinement.",
        ),
        sign_base.row(
            "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now",
            "pass"
            if pack[
                "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now"
            ]
            else "reject",
            "exact Trial-2 pure analytic operator-level continuum refinement available now",
            sign_base.truth(
                pack[
                    "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now"
                ]
            ),
            "This branch is allowed to keep the full operator-level continuum theorem open; it only refreshes the theorem wording after one honest pure-continuum promotion.",
        ),
        sign_base.row(
            "updated_pack_trial2_patched_tail_pure_continuum_closure_gate_required_now",
            "pass"
            if pack[
                "updated_pack_trial2_patched_tail_pure_continuum_closure_gate_required_now"
            ]
            else "reject",
            "updated-pack Trial-2 patched-tail pure-continuum closure gate required now",
            sign_base.truth(
                pack[
                    "updated_pack_trial2_patched_tail_pure_continuum_closure_gate_required_now"
                ]
            ),
            "The next honest task is no longer another theorem-support replay but one official gate that synchronizes the refreshed v2 theorem wording.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(pack["beta_common_root"]),
        "alpha_common_value": float(pack["alpha_common_value"]),
        "alpha_common_rel_error_vs_target": float(
            pack["alpha_common_rel_error_vs_target"]
        ),
        "tail_match_x": float(pack["tail_match_x"]),
        "x_cutoff": float(pack["x_cutoff"]),
        "analytic_remainder_bound_n2": float(pack["analytic_remainder_bound_n2"]),
        "analytic_remainder_over_total_abs_min_n2": float(
            pack["analytic_remainder_over_total_abs_min_n2"]
        ),
        "exact_trial2_first_principles_direct_alpha_closure_completed_now": bool(
            pack["exact_trial2_first_principles_direct_alpha_closure_completed_now"]
        ),
        "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_promotion_available_now": bool(
            pack[
                "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_promotion_available_now"
            ]
        ),
        "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_closure_completed_now": bool(
            pack[
                "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_closure_completed_now"
            ]
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now": bool(
            pack["exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now"]
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_deferred_to_v3_now": bool(
            pack[
                "exact_trial2_pure_analytic_operator_level_continuum_refinement_deferred_to_v3_now"
            ]
        ),
        "exact_trial2_v2_theorem_wording_upgrade_available_now": bool(
            pack["exact_trial2_v2_theorem_wording_upgrade_available_now"]
        ),
        "updated_pack_trial2_patched_tail_pure_continuum_closure_gate_required_now": bool(
            pack[
                "updated_pack_trial2_patched_tail_pure_continuum_closure_gate_required_now"
            ]
        ),
    }

    payload = {
        "step_tag": STEP_TAG,
        "step_name": STEP_NAME,
        "summary": summary,
        "rows": rows,
        "formulae": build_formulae(),
        "notes": {
            "audit_meaning": (
                "The old v2 wording no longer needs to say that every pure analytic continuum layer is deferred. "
                "One patched weighted-integral continuum layer is now closed, while only the full operator-level theorem remains deferred."
            ),
        },
    }
    written = write_artifact("declaration_gate", payload)
    print(json.dumps({"ok": True, "written": written}, ensure_ascii=False))


if __name__ == "__main__":
    main()
