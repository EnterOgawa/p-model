#!/usr/bin/env python3
"""Generate 8.7.56.5671-.5674 Trial-2 continuum spectral-projection artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_beta_sensitivity_continuum_spectral_projection_followup_backend import (
    build_trial2_beta_sensitivity_continuum_spectral_projection_followup_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5667-5670",
        "updated_pack_trial2_beta_sensitivity_spectral_projection_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "90_trial2_numeric_alpha_vector_qball_beta_sensitivity_continuum_spectral_projection_followup_audit.md"
)

STEP_TAG = "8.7.56.5671-5674"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "beta-sensitivity continuum spectral-projection followup audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_continuum_spectral_projection_followup_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_discrete_spectral_projection_theorem_completed_"
    "continuum_followup_primary_conditional_hold_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_continuum_open_interval_support_audited_"
    "operator_level_theorem_gate_next"
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
    """Return whether the continuum note carries the expected claims."""
    patterns = (
        "boundary layer",
        "interior window",
        "operator-level theorem",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the continuum-support audit."""
    return {
        "boundary_layer_ratio": "min_x M_h(x) / h",
        "interior_support": "inf_{x in [a+delta, b-delta]} M_h(x)",
        "richardson_estimate": "Q_inf = Q_hf + (Q_hf - Q_hc) / ((h_c / h_f)^2 - 1)",
    }


# 関数: `.5671-.5674` を実行する。

def main() -> None:
    """Execute the Trial-2 continuum spectral-projection followup audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_beta_sensitivity_continuum_spectral_projection_followup_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    continuum_boundary_layer_support_available_now = bool(
        pack[
            "exact_trial2_beta_sensitivity_continuum_boundary_layer_support_available_now"
        ]
    )
    continuum_gap_support_available_now = bool(
        pack["exact_trial2_beta_sensitivity_continuum_gap_support_available_now"]
    )
    continuum_open_interval_support_available_now = bool(
        pack["exact_trial2_beta_sensitivity_continuum_open_interval_support_available_now"]
    )
    operator_level_spectral_projection_theorem_available_now = bool(
        pack[
            "exact_trial2_beta_sensitivity_operator_level_spectral_projection_theorem_available_now"
        ]
    )
    operator_level_spectral_projection_followup_required_now = bool(
        pack[
            "updated_pack_trial2_beta_sensitivity_operator_level_spectral_projection_followup_required_now"
        ]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_beta_sensitivity_continuum_spectral_projection_followup_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 beta-sensitivity continuum spectral-projection followup selected now",
            sign_base.truth(route_selected),
            "This branch starts only after the discrete spectral-projection theorem has already been synchronized as the current positive closeout.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_continuum_spectral_projection_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 beta-sensitivity continuum spectral-projection note available now",
            sign_base.truth(note_available),
            "The note must record boundary-layer scaling, fixed interior-window positivity, and the remaining operator-level theorem gap.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_continuum_boundary_layer_support_available_now",
            "pass" if continuum_boundary_layer_support_available_now else "reject",
            "exact Trial-2 beta-sensitivity continuum boundary-layer support available now",
            sign_base.truth(continuum_boundary_layer_support_available_now),
            "Pass means the shrinking global margin is explained by stable O(h) boundary-layer scaling instead of genuine interior sign loss.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_continuum_gap_support_available_now",
            "pass" if continuum_gap_support_available_now else "reject",
            "exact Trial-2 beta-sensitivity continuum gap support available now",
            sign_base.truth(continuum_gap_support_available_now),
            "The sign pattern lambda_1 < 0 < lambda_2 must survive refinement before any continuum-support claim is honest.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_continuum_open_interval_support_available_now",
            "pass" if continuum_open_interval_support_available_now else "reject",
            "exact Trial-2 beta-sensitivity continuum open-interval support available now",
            sign_base.truth(continuum_open_interval_support_available_now),
            "Pass means every retained fixed interior window keeps a positive pointwise-dominance lower bound under refinement and under the Richardson continuum estimate.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_operator_level_spectral_projection_theorem_available_now",
            "pass" if operator_level_spectral_projection_theorem_available_now else "reject",
            "exact Trial-2 beta-sensitivity operator-level spectral-projection theorem available now",
            sign_base.truth(operator_level_spectral_projection_theorem_available_now),
            "The current audit still does not promote the final continuum theorem; it only establishes continuum-support numerics on fixed open interior windows.",
        ),
        sign_base.row(
            "updated_pack_trial2_beta_sensitivity_operator_level_spectral_projection_followup_required_now",
            "pass" if operator_level_spectral_projection_followup_required_now else "reject",
            "updated-pack Trial-2 beta-sensitivity operator-level spectral-projection followup required now",
            sign_base.truth(operator_level_spectral_projection_followup_required_now),
            "Once continuum-support numerics are fixed, the honest next blocker is the operator-level theorem that would remove the remaining discretization surface.",
        ),
    ]

    smallest_window = pack["interior_window_summaries"][0]
    largest_window = pack["interior_window_summaries"][-1]
    finest_row = pack["continuum_rows"][-1]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(pack["beta_common_root"]),
        "exact_trial2_beta_sensitivity_continuum_boundary_layer_support_available_now": bool(
            continuum_boundary_layer_support_available_now
        ),
        "exact_trial2_beta_sensitivity_continuum_gap_support_available_now": bool(
            continuum_gap_support_available_now
        ),
        "exact_trial2_beta_sensitivity_continuum_open_interval_support_available_now": bool(
            continuum_open_interval_support_available_now
        ),
        "exact_trial2_beta_sensitivity_operator_level_spectral_projection_theorem_available_now": bool(
            operator_level_spectral_projection_theorem_available_now
        ),
        "updated_pack_trial2_beta_sensitivity_operator_level_spectral_projection_followup_required_now": bool(
            operator_level_spectral_projection_followup_required_now
        ),
        "boundary_layer_rel_spread": float(pack["boundary_layer_rel_spread"]),
        "continuum_row_2400_global_margin_over_step": float(
            finest_row["global_pointwise_margin_over_step"]
        ),
        "continuum_row_2400_lambda_1": float(finest_row["lambda_1"]),
        "continuum_row_2400_lambda_2": float(finest_row["lambda_2"]),
        "lambda_1_continuum_estimate": float(pack["lambda_1_continuum_estimate"]),
        "lambda_2_continuum_estimate": float(pack["lambda_2_continuum_estimate"]),
        "smallest_interior_window": [
            float(smallest_window["x_min"]),
            float(smallest_window["x_max"]),
        ],
        "smallest_interior_window_continuum_margin_estimate": float(
            smallest_window["continuum_margin_estimate"]
        ),
        "smallest_interior_window_last_rel_spread": float(
            smallest_window["last_rel_spread"]
        ),
        "largest_interior_window": [
            float(largest_window["x_min"]),
            float(largest_window["x_max"]),
        ],
        "largest_interior_window_continuum_margin_estimate": float(
            largest_window["continuum_margin_estimate"]
        ),
        "largest_interior_window_last_rel_spread": float(
            largest_window["last_rel_spread"]
        ),
    }

    payload = {
        "step_tag": STEP_TAG,
        "step_name": STEP_NAME,
        "summary": summary,
        "rows": rows,
        "formulae": build_formulae(),
        "notes": {
            "audit_note": sign_base.display_path(AUDIT_NOTE),
            "continuum_windows": [
                [float(summary["x_min"]), float(summary["x_max"])]
                for summary in pack["interior_window_summaries"]
            ],
        },
    }
    written = write_artifact("declaration_gate", payload)
    print(json.dumps({"ok": True, "written": written}, ensure_ascii=False))


if __name__ == "__main__":
    main()
