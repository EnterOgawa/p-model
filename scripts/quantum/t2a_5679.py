#!/usr/bin/env python3
"""Generate 8.7.56.5679-.5682 Trial-2 operator-level followup artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_beta_sensitivity_operator_level_spectral_projection_followup_backend import (
    build_trial2_beta_sensitivity_operator_level_spectral_projection_followup_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5675-5678",
        "updated_pack_trial2_beta_sensitivity_continuum_spectral_projection_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "91_trial2_numeric_alpha_vector_qball_beta_sensitivity_operator_level_spectral_projection_followup_audit.md"
)

STEP_TAG = "8.7.56.5679-5682"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "beta-sensitivity operator-level spectral-projection followup audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_operator_level_spectral_projection_followup_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_continuum_open_interval_support_completed_"
    "operator_level_spectral_projection_followup_primary_"
    "conditional_hold_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_operator_level_spectral_projection_audited_"
    "weighted_integral_sign_gate_next"
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
    """Return whether the operator-level note carries the expected claims."""
    patterns = (
        "weighted-integral sign support",
        "boundary complement",
        "derivative-chain theorem",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the operator-level followup audit."""
    return {
        "weighted_integral": "dI_n / d beta = n ∫ y_beta^(n-1) u_beta x^2 dx",
        "control_split": "T_n = T_n^[0.10,19.90] + T_n^boundary",
        "difference_derivative": "Delta_common'(beta) = d alpha_qstar / d beta - d R8 / d beta",
    }


# 関数: `.5679-.5682` を実行する。

def main() -> None:
    """Execute the Trial-2 operator-level spectral-projection followup audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_beta_sensitivity_operator_level_spectral_projection_followup_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    control_window_continuum_support_available_now = bool(
        pack["control_window_continuum_support_available_now"]
    )
    weighted_integral_sign_support_available_now = bool(
        pack["exact_trial2_beta_sensitivity_weighted_integral_sign_support_available_now"]
    )
    delta_common_derivative_positive_local_support_now = bool(
        pack["delta_common_derivative_positive_local_support_now"]
    )
    operator_level_theorem_available_now = bool(
        pack[
            "exact_trial2_beta_sensitivity_operator_level_spectral_projection_theorem_available_now"
        ]
    )
    derivative_chain_followup_required_now = bool(
        pack["updated_pack_trial2_beta_sensitivity_derivative_chain_followup_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_beta_sensitivity_operator_level_spectral_projection_followup_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 beta-sensitivity operator-level spectral-projection followup selected now",
            sign_base.truth(route_selected),
            "This branch starts only after continuum open-interval support has already been fixed honestly as the prior positive closeout.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_operator_level_spectral_projection_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 beta-sensitivity operator-level spectral-projection note available now",
            sign_base.truth(note_available),
            "The note must distinguish weighted-integral sign support from the still-missing pure operator-level theorem and record the boundary-complement control.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_control_window_continuum_support_available_now",
            "pass" if control_window_continuum_support_available_now else "reject",
            "exact Trial-2 beta-sensitivity control-window continuum support available now",
            sign_base.truth(control_window_continuum_support_available_now),
            "The operator-level followup is only honest if the smallest retained interior window [0.10,19.90] still carries positive continuum-support margin under refinement.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_weighted_integral_sign_support_available_now",
            "pass" if weighted_integral_sign_support_available_now else "reject",
            "exact Trial-2 beta-sensitivity weighted-integral sign support available now",
            sign_base.truth(weighted_integral_sign_support_available_now),
            "Pass means dI_2/dbeta, dI_3/dbeta, and dI_4/dbeta stay negative after splitting the total integral into the retained interior window plus boundary complement.",
        ),
        sign_base.row(
            "trial2_delta_common_derivative_positive_local_support_now",
            "pass" if delta_common_derivative_positive_local_support_now else "reject",
            "Trial-2 Delta_common derivative positive local support now",
            sign_base.truth(delta_common_derivative_positive_local_support_now),
            "The operator-level sign route remains meaningful only if the retained local transversality support for d Delta_common / d beta > 0 stays stable together with the weighted-integral signs.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_operator_level_spectral_projection_theorem_available_now",
            "pass" if operator_level_theorem_available_now else "reject",
            "exact Trial-2 beta-sensitivity operator-level spectral-projection theorem available now",
            sign_base.truth(operator_level_theorem_available_now),
            "This audit still stops short of the final pure analytic continuum theorem; it only closes controlled weighted-integral sign support.",
        ),
        sign_base.row(
            "updated_pack_trial2_beta_sensitivity_derivative_chain_followup_required_now",
            "pass" if derivative_chain_followup_required_now else "reject",
            "updated-pack Trial-2 beta-sensitivity derivative-chain followup required now",
            sign_base.truth(derivative_chain_followup_required_now),
            "Once weighted-integral signs are controlled, the next blocker is one derivative-chain theorem that turns those signs into Delta_common'(beta) > 0 and then into uniqueness of beta_*.",
        ),
    ]

    weighted_by_order = pack["weighted_integral_by_order"]
    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(pack["beta_common_root"]),
        "control_window_x_min": float(pack["control_window_x_min"]),
        "control_window_x_max": float(pack["control_window_x_max"]),
        "control_window_continuum_margin_estimate": float(
            pack["smallest_window_summary"]["continuum_margin_estimate"]
        ),
        "control_window_last_rel_spread": float(
            pack["smallest_window_summary"]["last_rel_spread"]
        ),
        "exact_trial2_beta_sensitivity_weighted_integral_sign_support_available_now": bool(
            weighted_integral_sign_support_available_now
        ),
        "delta_common_derivative_positive_local_support_now": bool(
            delta_common_derivative_positive_local_support_now
        ),
        "exact_trial2_beta_sensitivity_operator_level_spectral_projection_theorem_available_now": bool(
            operator_level_theorem_available_now
        ),
        "updated_pack_trial2_beta_sensitivity_derivative_chain_followup_required_now": bool(
            derivative_chain_followup_required_now
        ),
        "d_i2_dbeta_min": float(
            weighted_by_order["2"]["d_integral_order_dbeta_min"]
        ),
        "d_i2_dbeta_max": float(
            weighted_by_order["2"]["d_integral_order_dbeta_max"]
        ),
        "d_i3_dbeta_min": float(
            weighted_by_order["3"]["d_integral_order_dbeta_min"]
        ),
        "d_i3_dbeta_max": float(
            weighted_by_order["3"]["d_integral_order_dbeta_max"]
        ),
        "d_i4_dbeta_min": float(
            weighted_by_order["4"]["d_integral_order_dbeta_min"]
        ),
        "d_i4_dbeta_max": float(
            weighted_by_order["4"]["d_integral_order_dbeta_max"]
        ),
        "boundary_complement_abs_fraction_max_n2": float(
            weighted_by_order["2"]["boundary_complement_abs_fraction_max"]
        ),
        "boundary_complement_abs_fraction_max_n3": float(
            weighted_by_order["3"]["boundary_complement_abs_fraction_max"]
        ),
        "boundary_complement_abs_fraction_max_n4": float(
            weighted_by_order["4"]["boundary_complement_abs_fraction_max"]
        ),
        "delta_common_derivative_min": float(pack["delta_common_derivative_min"]),
        "delta_common_derivative_max": float(pack["delta_common_derivative_max"]),
        "delta_common_derivative_rel_spread": float(
            pack["delta_common_derivative_rel_spread"]
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
            "retained_orders": [2, 3, 4],
        },
    }
    written = write_artifact("declaration_gate", payload)
    print(json.dumps({"ok": True, "written": written}, ensure_ascii=False))


if __name__ == "__main__":
    main()
