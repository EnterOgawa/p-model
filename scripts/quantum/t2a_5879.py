#!/usr/bin/env python3
"""Generate 8.7.56.5879-.5882 self-consistent 4D selector audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_4d_self_consistent_selector_exact_goal_backend import (
    build_trial2_4d_self_consistent_selector_exact_goal_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5875-5878",
        "updated_pack_trial2_external_probe_weight_theorem_source",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5879-5882"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "4D self-consistent selector exact-goal audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_4d_self_consistent_selector_exact_goal_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_external_probe_weight_theorem_source_negative_closeout_completed_"
    "zero_residual_exact_goal_unavailable_current_pack_conditional_reopen_only_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_self_consistent_selector_audited_current_family_negative_partial_gate"
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


# 関数: audit で固定する式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the self-consistent 4D selector audit."""
    return {
        "self_consistent_selector": (
            "alpha_qstar^(4D)(beta) = alpha_R8^(4D)(beta)"
        ),
        "probe_vertex_mix": (
            "alpha_qstar^(4D,vertex) = alpha_qstar / (C_4D^eta_vertex M_4D^(2-eta_vertex))"
        ),
        "bulk_family": (
            "alpha_R8^(4D,det) is audited over {none, mass_sq, half_mix, all_mix, charge_mass}"
        ),
    }


# 関数: `.5879-.5882` を実行する。

def main() -> None:
    """Execute the self-consistent 4D selector audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_4d_self_consistent_selector_exact_goal_pack()
    best_row = dict(pack["best_row"])
    expert_row = dict(pack["expert_minimal_row"])

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    family_available = bool(
        pack["exact_trial2_4d_self_consistent_selector_family_available_now"]
    )
    expert_pair_available = bool(
        pack["exact_trial2_4d_self_consistent_selector_expert_minimal_pair_available_now"]
    )
    positive_partial = bool(
        pack["exact_trial2_4d_self_consistent_selector_positive_partial_now"]
    )
    beats_current_best = bool(
        pack["exact_trial2_4d_self_consistent_selector_beats_current_best_now"]
    )
    zero_residual_unavailable = not bool(
        pack["exact_trial2_4d_self_consistent_selector_zero_residual_exact_goal_available_now"]
    )
    followup_gate_required = bool(family_available and zero_residual_unavailable)

    rows = [
        sign_base.row(
            "updated_pack_trial2_external_probe_weight_theorem_source_negative_closeout_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 external-probe weight theorem-source negative closeout selected now",
            sign_base.truth(route_selected),
            "The self-consistent selector route only reopens after the deterministic weight theorem-source route closes honestly.",
        ),
        sign_base.row(
            "exact_trial2_4d_self_consistent_selector_family_available_now",
            "pass" if family_available else "reject",
            "exact Trial-2 4D self-consistent selector family available now",
            sign_base.truth(family_available),
            "The current 4D family does support deterministic self-consistent root solving once both readouts are corrected.",
        ),
        sign_base.row(
            "exact_trial2_4d_self_consistent_selector_expert_minimal_pair_available_now",
            "pass" if expert_pair_available else "reject",
            "exact Trial-2 4D self-consistent selector expert minimal pair available now",
            sign_base.truth(expert_pair_available),
            "The expert minimal route probe_vertex_mix / bulk_mass_sq does materialize one unique self-consistent root.",
        ),
        sign_base.row(
            "exact_trial2_4d_self_consistent_selector_positive_partial_now",
            "pass" if positive_partial else "reject",
            "exact Trial-2 4D self-consistent selector positive partial now",
            sign_base.truth(positive_partial),
            "The best deterministic self-consistent pair does improve the canonical 4D row, so the route is not empty replay.",
        ),
        sign_base.row(
            "exact_trial2_4d_self_consistent_selector_beats_current_best_now",
            "pass" if beats_current_best else "reject",
            "exact Trial-2 4D self-consistent selector beats current best now",
            sign_base.truth(beats_current_best),
            "Passing would mean the self-consistent 4D selector overtakes the retained external-probe current-vertex candidate.",
        ),
        sign_base.row(
            "exact_trial2_4d_self_consistent_selector_zero_residual_exact_goal_unavailable_now",
            "pass" if zero_residual_unavailable else "reject",
            "exact Trial-2 4D self-consistent selector zero-residual exact goal unavailable now",
            sign_base.truth(zero_residual_unavailable),
            "The corrected self-consistent family is still an audit route, not a zero-residual theorem.",
        ),
        sign_base.row(
            "updated_pack_trial2_4d_self_consistent_selector_gate_required_now",
            "pass" if followup_gate_required else "reject",
            "updated-pack Trial-2 4D self-consistent selector gate required now",
            sign_base.truth(followup_gate_required),
            "The next honest blocker is the route verdict itself: whether this self-consistent family is strong enough to promote or must close negatively.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "best_pair_q_label": str(best_row["q_label"]),
        "best_pair_r_label": str(best_row["r_label"]),
        "best_pair_beta_root": float(best_row["root_beta"]),
        "best_pair_alpha": float(best_row["alpha_self_consistent_at_root"]),
        "best_pair_rel_error_vs_exact_goal": float(
            best_row["alpha_self_consistent_rel_error_vs_exact_goal"]
        ),
        "best_pair_improvement_factor_vs_canonical": float(
            pack["best_pair_improvement_factor_vs_canonical"]
        ),
        "best_pair_improvement_factor_vs_current_best": float(
            pack["best_pair_improvement_factor_vs_current_best"]
        ),
        "expert_pair_q_label": str(expert_row["q_label"]),
        "expert_pair_r_label": str(expert_row["r_label"]),
        "expert_pair_beta_root": float(expert_row["root_beta"]),
        "expert_pair_alpha": float(expert_row["alpha_self_consistent_at_root"]),
        "expert_pair_rel_error_vs_exact_goal": float(
            expert_row["alpha_self_consistent_rel_error_vs_exact_goal"]
        ),
        "selected_next_generation_route": "trial2_4d_self_consistent_selector_gate",
        "recommended_next_route_or_none": ".5883-.5886",
        "selected_followup_route": "trial2_4d_self_consistent_selector_gate",
        "selected_followup_route_or_none": ".5883-.5886",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5881",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_4d_self_consistent_selector_exact_goal_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "pack": pack,
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5879-5882 self-consistent 4D selector audit completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
