#!/usr/bin/env python3
"""Generate 8.7.56.2027-.2030 alias-harmonic spike gate artifacts.

`.2023-.2026` showed that the first and second alias-harmonic spike windows
are already dominated by the leading alias image of the exact overlap rather
than by same-level `h1` or `h2` boundary-term corrections. This branch
syncs that result into the official gate and promotes the next honest surface
to an alias-image signed-rule theorem.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
LONG_ROADMAP = ROOT / "doc" / "quantum" / "55_trial2_numeric_alpha_vector_qball_long_horizon_roadmap.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

PRIOR_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_2023_2026_alias_harmonic_spike_audit_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.2027-2030"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor alias-harmonic spike "
    "decision gate / registry"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "alias_harmonic_spike_gate",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_alias_harmonic_spike_audited_"
    "alias_image_reactivation_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_alias_image_signed_rule_"
    "reactivation_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_boundary_alias_image_"
    "signed_rule_reactivation"
)
NEXT_ROUTE = "8.7.56.2031"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_alias_image_signed_rule_"
    "decision_gate_registry"
)
FOLLOWUP_ROUTE = "8.7.56.2035"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


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

    return {
        "json": sign_base.display_path(paths["json"]),
        "csv": sign_base.display_path(paths["csv"]),
    }


# 関数: decision gate 用の公式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the alias-harmonic spike decision gate."""
    return {
        "retained_local_jet_rule": "G_jet(q)=(-h0 q^2 + h2) cos(q R_box) + h1 q sin(q R_box)=0",
        "alias_image_rule": "sigma_img^(n)(q)=(-1)^n sign(F_exact(|q_alias^(n)-q|))",
        "selection_logic": "if epsilon_1, epsilon_2 << 1 and sigma_img improves both spike windows, promote alias-image signed rule before any same-level boundary-term retry",
    }


# 関数: `.2027-.2030` を実行する。

def main() -> None:
    """Execute the alias-harmonic spike decision gate."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LONG_ROADMAP,
        PART5,
        PRIOR_GATE,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    inventory_ready = bool(prior_summary["alias_image_family_admissible"])

    gate_a_exact_spike_closeout_selected = False
    gate_b_alias_image_signed_rule_selected = bool(
        prior_summary["alias_image_family_admissible"]
    )
    gate_c_current_rule_blocked = False
    same_level_boundary_term_retry_admissible = bool(
        prior_summary["same_level_boundary_term_retry_admissible"]
    )
    alias_image_signed_rule_reactivation_admissible_now = True
    substantive_pack_update_required_now = False
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "alias-harmonic spike gate inventory ready",
            sign_base.truth(inventory_ready),
            "The gate starts only after the spike audit has shown that the leading alias image improves both retained spike windows.",
        ),
        sign_base.row(
            "gate_a_exact_spike_closeout_selected",
            "reject",
            "Gate A exact spike closeout selected",
            sign_base.truth(gate_a_exact_spike_closeout_selected),
            "The retained local-jet rule alone does not close the alias-harmonic spike family exactly.",
        ),
        sign_base.row(
            "gate_b_alias_image_signed_rule_selected",
            "pass" if gate_b_alias_image_signed_rule_selected else "reject",
            "Gate B alias-image signed rule selected",
            sign_base.truth(gate_b_alias_image_signed_rule_selected),
            "Once the spike windows are leading-carrier dominated, the honest next theorem surface is the alias-image signed rule.",
        ),
        sign_base.row(
            "gate_c_current_rule_blocked",
            "reject" if not gate_c_current_rule_blocked else "pass",
            "Gate C current rule blocked",
            sign_base.truth(gate_c_current_rule_blocked),
            "The current retained pack is not globally blocked because the alias-image surface is still internal to the present box pack.",
        ),
        sign_base.row(
            "same_level_boundary_term_retry_admissible",
            "reject" if not same_level_boundary_term_retry_admissible else "pass",
            "same-level boundary-term retry admissible",
            sign_base.truth(same_level_boundary_term_retry_admissible),
            "The spike audit already showed that h1/h2 corrections are subleading on both active windows.",
        ),
        sign_base.row(
            "alias_image_signed_rule_reactivation_admissible_now",
            "pass",
            "alias-image signed rule reactivation admissible now",
            sign_base.truth(alias_image_signed_rule_reactivation_admissible_now),
            "The next official branch is the boundary alias-image signed-rule reactivation.",
        ),
        sign_base.row(
            "substantive_pack_update_required_now",
            "reject" if not substantive_pack_update_required_now else "pass",
            "substantive pack update required now",
            sign_base.truth(substantive_pack_update_required_now),
            "The next move remains an internal signed-rule theorem audit rather than an immediate pack update.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "gate_a_exact_spike_closeout_selected": gate_a_exact_spike_closeout_selected,
        "gate_b_alias_image_signed_rule_selected": gate_b_alias_image_signed_rule_selected,
        "gate_c_current_rule_blocked": gate_c_current_rule_blocked,
        "same_level_boundary_term_retry_admissible": same_level_boundary_term_retry_admissible,
        "alias_image_signed_rule_reactivation_admissible_now": alias_image_signed_rule_reactivation_admissible_now,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "bulk_delta_r_over_m0": float(prior_summary["bulk_delta_r_over_m0"]),
        "bulk_uniform_cell_fraction": float(prior_summary["bulk_uniform_cell_fraction"]),
        "edge_cell_relative_gap": float(prior_summary["edge_cell_relative_gap"]),
        "first_alias_harmonic_over_m0": float(prior_summary["first_alias_harmonic_over_m0"]),
        "second_alias_harmonic_over_m0": float(prior_summary["second_alias_harmonic_over_m0"]),
        "fit_window_max_h1_over_h0q": float(prior_summary["fit_window_max_h1_over_h0q"]),
        "fit_window_max_h2_over_h0q2": float(prior_summary["fit_window_max_h2_over_h0q2"]),
        "edge_window_max_h1_over_h0q": float(prior_summary["edge_window_max_h1_over_h0q"]),
        "edge_window_max_h2_over_h0q2": float(prior_summary["edge_window_max_h2_over_h0q2"]),
        "fit_window_local_jet_sign_mismatch_fraction": float(
            prior_summary["fit_window_local_jet_sign_mismatch_fraction"]
        ),
        "fit_window_alias_image_sign_mismatch_fraction": float(
            prior_summary["fit_window_alias_image_sign_mismatch_fraction"]
        ),
        "fit_alias_image_gain_over_local_jet": float(
            prior_summary["fit_alias_image_gain_over_local_jet"]
        ),
        "edge_window_local_jet_sign_mismatch_fraction": float(
            prior_summary["edge_window_local_jet_sign_mismatch_fraction"]
        ),
        "edge_window_alias_image_sign_mismatch_fraction": float(
            prior_summary["edge_window_alias_image_sign_mismatch_fraction"]
        ),
        "edge_alias_image_gain_over_local_jet": float(
            prior_summary["edge_alias_image_gain_over_local_jet"]
        ),
        "fit_alias_image_sign_correlation": float(
            prior_summary["fit_alias_image_sign_correlation"]
        ),
        "edge_alias_image_sign_correlation": float(
            prior_summary["edge_alias_image_sign_correlation"]
        ),
        "fit_subleading_negligible": bool(prior_summary["fit_subleading_negligible"]),
        "edge_subleading_negligible": bool(prior_summary["edge_subleading_negligible"]),
        "fit_alias_image_supported": bool(prior_summary["fit_alias_image_supported"]),
        "edge_alias_image_supported": bool(prior_summary["edge_alias_image_supported"]),
        "alias_image_family_admissible": bool(
            prior_summary["alias_image_family_admissible"]
        ),
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2029",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "ai_context": sign_base.display_path(AI_CONTEXT),
                "work_history_recent": sign_base.display_path(WORK_HISTORY_RECENT),
                "current_problem": sign_base.display_path(CURRENT_PROBLEM),
                "current_status": sign_base.display_path(CURRENT_STATUS),
                "unified_roadmap": sign_base.display_path(UNIFIED_ROADMAP),
                "long_roadmap": sign_base.display_path(LONG_ROADMAP),
                "part5": sign_base.display_path(PART5),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
            },
            "constants": {
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_alias_image_signed_rule_gate_selected",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2023"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.2027-.2030"),
                "current_problem_hit": sign_base.hit(current_problem_text, "alias-harmonic spike"),
                "current_status_hit": sign_base.hit(current_status_text, "alias-harmonic spike"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2027-.2030"),
                "long_roadmap_hit": sign_base.hit(long_text, "alias-harmonic spike decision gate"),
                "part5_hit": sign_base.hit(part5_text, ".2015-.2022"),
            },
        },
    )

    route_payload = sign_base.payload(
        "8.7.56.2030",
        STEP_NAME + " route sync",
        {
            "declaration_source": sign_base.display_path(
                build_metrics_paths(PUBLIC_OUT, STEM, "declaration_gate")["json"]
            ),
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "selected_next_generation_route_or_none": NEXT_ROUTE,
            "selected_followup_route": FOLLOWUP_ROUTE_NAME,
            "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        },
        [
            sign_base.row(
                "gate_b_alias_image_signed_rule_selected",
                "pass" if gate_b_alias_image_signed_rule_selected else "reject",
                "Gate B alias-image signed rule selected",
                sign_base.truth(gate_b_alias_image_signed_rule_selected),
                "The next official branch is now the alias-image signed-rule reactivation.",
            ),
            sign_base.row(
                "same_level_boundary_term_retry_admissible",
                "reject" if not same_level_boundary_term_retry_admissible else "pass",
                "same-level boundary-term retry admissible",
                sign_base.truth(same_level_boundary_term_retry_admissible),
                "Once the spike family is reclassified, the old boundary-term retry remains closed.",
            ),
            sign_base.row(
                "next_route_fixed",
                "pass",
                "next route fixed",
                1.0,
                "The next official branch is the boundary alias-image signed-rule reactivation.",
            ),
        ],
        summary,
        {
            "overall_status": "vector_qball_form_factor_alias_image_signed_rule_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"formulas": build_formulae()},
    )

    declaration_paths = write_artifact("declaration_gate", declaration_payload)
    route_paths = write_artifact("route_sync", route_payload)
    print("[ok] 8.7.56.2027-.2030 alias-image gate artifacts generated")
    print(f"[ok] declaration: {declaration_paths['json']}")
    print(f"[ok] route sync:   {route_paths['json']}")


if __name__ == "__main__":
    main()
