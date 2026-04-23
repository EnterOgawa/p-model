#!/usr/bin/env python3
"""Generate 8.7.56.5687-.5690 Trial-2 beta-sensitivity derivative-chain artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_beta_sensitivity_derivative_chain_followup_backend import (
    build_trial2_beta_sensitivity_derivative_chain_followup_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5683-5686",
        "updated_pack_trial2_beta_sensitivity_operator_level_spectral_projection_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "92_trial2_numeric_alpha_vector_qball_beta_sensitivity_derivative_chain_followup_audit.md"
)

STEP_TAG = "8.7.56.5687-5690"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "beta-sensitivity derivative-chain followup audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_derivative_chain_followup_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_weighted_integral_sign_support_completed_"
    "derivative_chain_followup_primary_conditional_hold_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_derivative_chain_audited_"
    "uniqueness_anchor_gate_next"
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
    """Return whether the derivative-chain note carries the expected claims."""
    patterns = (
        "alpha_qstar(beta)",
        "R8(beta)",
        "uniqueness-anchor",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the derivative-chain audit."""
    return {
        "alpha_chain": (
            "d alpha_qstar / d beta = alpha_profile_channel + alpha_qstar_channel"
        ),
        "r8_chain": (
            "d R8 / d beta = (partial_I2 R8) dI2/dbeta + (partial_Ig R8) dIg/dbeta "
            "+ (partial_I4 R8) dI4/dbeta + (partial_B R8) dB/dbeta + partial_beta R8"
        ),
        "delta_chain": (
            "Delta_common'(beta) = d alpha_qstar / d beta - d R8 / d beta"
        ),
    }


# 関数: `.5687-.5690` を実行する。

def main() -> None:
    """Execute the Trial-2 beta-sensitivity derivative-chain followup audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_beta_sensitivity_derivative_chain_followup_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    alpha_chain_available = bool(
        pack[
            "exact_trial2_alpha_qstar_derivative_chain_positive_local_support_available_now"
        ]
    )
    r8_chain_available = bool(
        pack["exact_trial2_r8_derivative_chain_negative_local_support_available_now"]
    )
    delta_chain_available = bool(
        pack[
            "exact_trial2_delta_common_derivative_chain_positive_local_support_available_now"
        ]
    )
    derivative_chain_theorem_available = bool(
        pack["exact_trial2_beta_sensitivity_derivative_chain_theorem_available_now"]
    )
    uniqueness_anchor_followup_required_now = bool(
        pack[
            "updated_pack_trial2_beta_sensitivity_uniqueness_anchor_followup_required_now"
        ]
    )

    alpha_row = pack["alpha_chain_rows"][-1]
    r8_row = pack["r8_chain_rows"][-1]

    rows = [
        sign_base.row(
            "updated_pack_trial2_beta_sensitivity_derivative_chain_followup_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 beta-sensitivity derivative-chain followup selected now",
            sign_base.truth(route_selected),
            "This branch starts only after weighted-integral sign support is already synchronized as the current positive operator-level surface.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_derivative_chain_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 beta-sensitivity derivative-chain note available now",
            sign_base.truth(note_available),
            "The note must record the alpha_qstar chain split, the exact R8 total-derivative channels, and the remaining uniqueness-anchor blocker.",
        ),
        sign_base.row(
            "exact_trial2_alpha_qstar_derivative_chain_positive_local_support_available_now",
            "pass" if alpha_chain_available else "reject",
            "exact Trial-2 alpha_qstar derivative-chain positive local support available now",
            sign_base.truth(alpha_chain_available),
            "Pass means the negative profile-response channel is overruled by the explicit q_star(beta) channel and the total alpha_qstar derivative stays positive across the retained h family.",
        ),
        sign_base.row(
            "exact_trial2_r8_derivative_chain_negative_local_support_available_now",
            "pass" if r8_chain_available else "reject",
            "exact Trial-2 R8 derivative-chain negative local support available now",
            sign_base.truth(r8_chain_available),
            "Pass means the exact R8 total derivative keeps the required partial-sign pattern and the negative channels dominate the positive channels across the retained h family.",
        ),
        sign_base.row(
            "exact_trial2_delta_common_derivative_chain_positive_local_support_available_now",
            "pass" if delta_chain_available else "reject",
            "exact Trial-2 Delta_common derivative-chain positive local support available now",
            sign_base.truth(delta_chain_available),
            "Pass means the retained local transversality survives the explicit derivative-chain split instead of only the raw finite-difference support.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_derivative_chain_theorem_available_now",
            "pass" if derivative_chain_theorem_available else "reject",
            "exact Trial-2 beta-sensitivity derivative-chain theorem available now",
            sign_base.truth(derivative_chain_theorem_available),
            "The current audit still does not promote the full uniqueness theorem; it only fixes derivative-chain sign support strongly enough to isolate the next anchor-level blocker.",
        ),
        sign_base.row(
            "updated_pack_trial2_beta_sensitivity_uniqueness_anchor_followup_required_now",
            "pass" if uniqueness_anchor_followup_required_now else "reject",
            "updated-pack Trial-2 beta-sensitivity uniqueness-anchor followup required now",
            sign_base.truth(uniqueness_anchor_followup_required_now),
            "Once the derivative-chain signs are fixed honestly, the next blocker is the anchor statement that turns local transversality into the unique common-root theorem.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(pack["beta_common_root"]),
        "alpha_total_derivative_min": float(pack["alpha_total_derivative_min"]),
        "alpha_total_derivative_max": float(pack["alpha_total_derivative_max"]),
        "alpha_total_derivative_rel_spread": float(
            pack["alpha_total_derivative_rel_spread"]
        ),
        "r8_total_derivative_min": float(pack["r8_total_derivative_min"]),
        "r8_total_derivative_max": float(pack["r8_total_derivative_max"]),
        "r8_total_derivative_rel_spread": float(
            pack["r8_total_derivative_rel_spread"]
        ),
        "delta_common_derivative_min": float(pack["delta_common_derivative_min"]),
        "delta_common_derivative_max": float(pack["delta_common_derivative_max"]),
        "delta_common_derivative_rel_spread": float(
            pack["delta_common_derivative_rel_spread"]
        ),
        "alpha_profile_channel_dbeta_h1e6": float(
            alpha_row["alpha_profile_channel_dbeta"]
        ),
        "alpha_qstar_channel_dbeta_h1e6": float(alpha_row["alpha_qstar_channel_dbeta"]),
        "r8_negative_channels_sum_dbeta_h1e6": float(
            r8_row["negative_channels_sum_dbeta"]
        ),
        "r8_positive_channels_sum_dbeta_h1e6": float(
            r8_row["positive_channels_sum_dbeta"]
        ),
        "exact_trial2_alpha_qstar_derivative_chain_positive_local_support_available_now": bool(
            alpha_chain_available
        ),
        "exact_trial2_r8_derivative_chain_negative_local_support_available_now": bool(
            r8_chain_available
        ),
        "exact_trial2_delta_common_derivative_chain_positive_local_support_available_now": bool(
            delta_chain_available
        ),
        "exact_trial2_beta_sensitivity_derivative_chain_theorem_available_now": bool(
            derivative_chain_theorem_available
        ),
        "updated_pack_trial2_beta_sensitivity_uniqueness_anchor_followup_required_now": bool(
            uniqueness_anchor_followup_required_now
        ),
        "selected_next_generation_route": (
            "trial2_beta_sensitivity_uniqueness_anchor_followup"
        ),
        "recommended_next_route_or_none": (
            "trial2_beta_sensitivity_uniqueness_anchor_followup"
        ),
    }

    payload = sign_base.payload(
        "8.7.56.5689",
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
            "overall_status": "trial2_beta_sensitivity_derivative_chain_followup_completed",
            "branch_completed": True,
            "breakthrough_passed_now": delta_chain_available,
            "physical_reject_required": False,
        },
        {
            "alpha_chain_row_h1e6": alpha_row,
            "r8_chain_row_h1e6": r8_row,
            "operator_pack": pack["operator_pack"],
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print(
        "[done] 8.7.56.5687-5690 Trial-2 beta-sensitivity derivative-chain audit completed"
    )
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
