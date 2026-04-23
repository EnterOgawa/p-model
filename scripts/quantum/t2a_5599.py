#!/usr/bin/env python3
"""Generate 8.7.56.5599-.5602 Trial-2 entropy audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_entropy_route_backend import build_trial2_entropy_pack
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5595-5598",
        "updated_pack_trial2_interaction_harmonic_exact_relation_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT / "doc" / "quantum" / "80_trial2_numeric_alpha_vector_qball_entropy_audit.md"
)

STEP_TAG = "8.7.56.5599-5602"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor Trial-2 entropy audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_entropy_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_interaction_harmonic_exact_relation_negative_closeout_completed_"
    "entropy_primary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_entropy_negative_closeout_completed_conditional_reopen_only_gate"
)
RETAINED_BETA = 0.9982557379261291
NEAREST_BETA = 0.9982996989044647
Q_EXACT = 0.2416825755115744


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


# 関数: note が expected claims を含むか確認する。

def note_contains_audit(text: str) -> bool:
    """Return whether the entropy audit note carries the expected claims."""
    patterns = (
        "entropy",
        "Shannon",
        "conditional reopen",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the entropy audit."""
    return {
        "probability_density": "p(r) = f(r)^2 r^2 / int dr f(r)^2 r^2",
        "entropy": "S = - int dr p(r) ln p(r)",
        "alpha_candidate": "alpha_ent = exp(-S) / (4*pi)",
        "form_factor_candidate": "F_ent = exp(-S/2)",
    }


# 関数: `.5599-.5602` を実行する。

def main() -> None:
    """Execute the Trial-2 entropy audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_entropy_pack(
        retained_beta=float(RETAINED_BETA),
        nearest_beta=float(NEAREST_BETA),
        q_exact=float(Q_EXACT),
    )

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    entropy_alpha_exact_route_available_now = bool(
        pack["entropy_alpha_exact_route_available_now"]
    )
    entropy_form_factor_exact_route_available_now = bool(
        pack["entropy_form_factor_exact_route_available_now"]
    )
    entropy_route_negative_closeout_available_now = bool(
        pack["entropy_route_negative_closeout_available_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_entropy_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 entropy selected now",
            sign_base.truth(route_selected),
            "Entropy becomes the next honest low-cost direct-alpha branch only after the interaction-over-harmonic exact-law route closes negatively.",
        ),
        sign_base.row(
            "exact_trial2_entropy_audit_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 entropy audit note available now",
            sign_base.truth(note_available),
            "The audit note records the retained Shannon entropy readout and the conditional-hold verdict.",
        ),
        sign_base.row(
            "exact_trial2_entropy_alpha_exact_route_available_now",
            "pass" if entropy_alpha_exact_route_available_now else "reject",
            "exact Trial-2 entropy alpha exact route available now",
            sign_base.truth(entropy_alpha_exact_route_available_now),
            "Pass would mean alpha = exp(-S)/(4*pi) reproduces the retained direct-alpha target.",
        ),
        sign_base.row(
            "exact_trial2_entropy_form_factor_exact_route_available_now",
            "pass" if entropy_form_factor_exact_route_available_now else "reject",
            "exact Trial-2 entropy form-factor exact route available now",
            sign_base.truth(entropy_form_factor_exact_route_available_now),
            "Pass would mean F(q_exact) = exp(-S/2) reproduces the retained crossing form factor directly.",
        ),
        sign_base.row(
            "exact_trial2_entropy_negative_closeout_available_now",
            "pass" if entropy_route_negative_closeout_available_now else "reject",
            "exact Trial-2 entropy negative closeout available now",
            sign_base.truth(entropy_route_negative_closeout_available_now),
            "Both retained entropy candidates miss badly enough that the entropy route closes honestly without a new alpha readout.",
        ),
    ]

    retained = dict(pack["retained_row"])
    near_row = dict(pack["nearest_row"])
    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_exact_over_m0": float(pack["q_exact_over_m0"]),
        "retained_beta1": float(pack["retained_beta1"]),
        "nearest_alpha_beta_root_to_retained": float(
            pack["nearest_alpha_beta_root_to_retained"]
        ),
        "retained_shannon_entropy": float(retained["shannon_entropy"]),
        "retained_alpha_from_entropy": float(retained["alpha_from_entropy"]),
        "retained_alpha_from_entropy_rel_error_vs_target": float(
            retained["alpha_from_entropy_rel_error_vs_target"]
        ),
        "retained_form_factor_from_entropy": float(retained["form_factor_from_entropy"]),
        "retained_form_factor_exact": float(retained["form_factor_exact"]),
        "retained_form_factor_from_entropy_rel_error_vs_exact": float(
            retained["form_factor_from_entropy_rel_error_vs_exact"]
        ),
        "nearest_shannon_entropy": float(near_row["shannon_entropy"]),
        "nearest_alpha_from_entropy_rel_error_vs_target": float(
            near_row["alpha_from_entropy_rel_error_vs_target"]
        ),
        "exact_trial2_entropy_alpha_exact_route_available_now": (
            entropy_alpha_exact_route_available_now
        ),
        "exact_trial2_entropy_form_factor_exact_route_available_now": (
            entropy_form_factor_exact_route_available_now
        ),
        "exact_trial2_entropy_negative_closeout_available_now": (
            entropy_route_negative_closeout_available_now
        ),
        "selected_primary_completion_lane": "conditional_reopen_only",
        "selected_secondary_completion_lane": "conditional_reopen_only",
        "selected_reserve_completion_lane": "conditional_reopen_only",
        "selected_next_generation_route": "conditional_reopen_only",
        "recommended_next_route_or_none": "8.7.56.5603",
        "selected_followup_route": "conditional_reopen_only",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5601",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
                "backend_helper": sign_base.display_path(
                    ROOT / "scripts" / "quantum" / "trial2_entropy_route_backend.py"
                ),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5603",
                "followup_route": "conditional_reopen_only",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_entropy_audit_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 entropy audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
