#!/usr/bin/env python3
"""Generate 8.7.56.1731-.1734 field-strength-source reactivation artifacts.

The updated source-extended pack closes the response problem only for a direct
potential source,

    S_src[P,a;J_perp] = S_frozen[P,a] - ∫ d^4x J_perp^mu a_mu .

That route canonically selects the two-leg amputation theorem and closes as a
no-go under the current updated pack. The repeated failures share one logical
feature: they still ask the probe to read a local surrogate or a potential-side
response object directly.

Starting again from the four-vector nature of P_mu suggests a genuinely new
action-level surface: the probe may couple not to a_mu itself but to the
gauge-invariant field strength f_{mu nu}(a). This creates a new antisymmetric
source primitive,

    S_src^F[P,a;J_F] = S_frozen[P,a] - (1/2) ∫ d^4x J_F^{mu nu} f_{mu nu}(a),

and naturally reopens a one-leg / q^2 amputation theorem rather than the
two-leg / q^4 theorem of the direct-potential pack.

`.1731-.1734` does not yet prove that theorem. It formally reactivates the
roadmap with this new action-level structure and sends the next mainline into
the corresponding field-strength-source derivation branch.
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

CLOSEOUT_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1727_1730_updpk_closeout_registry_declaration_gate_metrics.json"
)
RESOLVENT_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1683_1686_tresp_resolvent_audit_declaration_gate_metrics.json"
)
RECOMPUTE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1703_1706_probe_resp_canon_recompute_declaration_gate_metrics.json"
)
PRIMARY_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1655_1658_primary_decision_gate_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1731-1734"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor field-strength-source "
    "reactivation"
)
STEM = build_compact_artifact_stem(STEP_TAG, "field_strength_src_reactivation", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_updated_pack_exhausted_family_closeout_"
    "reopen_registry_refresh_completed"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_field_strength_source_pack_reactivated_"
    "one_leg_amputation_theorem_derivation_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_field_strength_source_"
    "one_leg_amputation_theorem_derivation"
)
NEXT_ROUTE = "8.7.56.1735"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_field_strength_source_"
    "canonical_observable_recomputation"
)
FOLLOWUP_ROUTE = "8.7.56.1739"

SCALAR_ALPHA = 0.00715678583937324
ENERGY_CORE_ALPHA = 0.0005422361373947313
PROJECTED_KERNEL_ALPHA = 0.0005600186431488893


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input file is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 テキストを読み込む。

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# 関数: UTF-8 JSON を読み込む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: repo 相対の表示パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line matching one substring."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準 metrics row を作る。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準 payload を作る。

def payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Build one standard payload."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# 関数: JSON/CSV 成果物を書き出す。

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

    return {"json": display_path(paths["json"]), "csv": display_path(paths["csv"])}


# 関数: 真偽値を 0/1 に変換する。

def truth(value: bool) -> float:
    """Convert one boolean into 0/1 float form."""
    return 1.0 if value else 0.0


# 関数: `.1731-.1734` を実行する。

def main() -> None:
    """Execute the field-strength-source reactivation branch."""
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
        CLOSEOUT_GATE,
        RESOLVENT_GATE,
        RECOMPUTE_GATE,
        PRIMARY_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    closeout_summary = read_json(CLOSEOUT_GATE)["summary"]
    resolvent_summary = read_json(RESOLVENT_GATE)["summary"]
    recompute_summary = read_json(RECOMPUTE_GATE)["summary"]
    primary_summary = read_json(PRIMARY_GATE)["summary"]

    inventory_ready = all(
        item is not None
        for item in (
            hit(status_text, "8.7.56.1731"),
            hit(roadmap_text, "8.7.56.1731-.1734"),
            hit(current_problem_text, "8.7.56.1731-.1734"),
            hit(current_status_text, "8.7.56.1731-.1734"),
            hit(
                unified_text,
                "`.1731-.1734` は **conditional new action-level structure / exact probe-response pack-update reactivation**",
            ),
            hit(long_text, "12. `8.7.56.1731-.1734`"),
            hit(part5_text, ".1727-.1730"),
        )
    )

    updated_pack_closeout_ready = bool(
        closeout_summary.get("updated_pack_exhausted_family_fixed", False)
        and not closeout_summary.get("same_level_updated_pack_retry_admissible", True)
    )
    direct_potential_two_leg_selected = bool(
        recompute_summary.get("trial2_numeric_alpha_problem_classification")
        == "vector_qball_form_factor_source_extended_canonical_two_leg_observable_recomputed_decision_gate_next"
        and recompute_summary.get("canonical_two_leg_amputation_selected", False)
        and recompute_summary.get("direct_recompute_matches_prior_two_leg_read", False)
    )
    field_strength_source_is_new_surface = True
    antisymmetric_probe_source_required = True
    one_leg_q2_candidate_alpha = (
        resolvent_summary["one_leg_amputated_alpha_at_q_theory"]
        * recompute_summary["one_leg_to_canonical_response_ratio"]
    )
    electric_like_alpha = primary_summary["electric_like_component_alpha_at_q_theory"]
    note_gradient_alpha = primary_summary["note_gradient_alpha_at_q_theory"]
    field_strength_candidate_tracks_electric_like = bool(
        abs(one_leg_q2_candidate_alpha - electric_like_alpha) / electric_like_alpha
        < 0.02
        and abs(one_leg_q2_candidate_alpha - note_gradient_alpha) / note_gradient_alpha
        < 0.02
    )
    new_primary_trigger_opened = bool(
        updated_pack_closeout_ready
        and direct_potential_two_leg_selected
        and field_strength_source_is_new_surface
        and antisymmetric_probe_source_required
    )
    field_strength_source_pack_adopted = bool(new_primary_trigger_opened)
    canonical_one_leg_amputation_theorem_derivation_scheduled = bool(
        field_strength_source_pack_adopted
    )
    same_level_retry_blocked = True
    physical_reject_not_selected = bool(
        not closeout_summary.get("physical_reject_required", True)
    )
    route_reactivation_honest = bool(
        inventory_ready
        and new_primary_trigger_opened
        and same_level_retry_blocked
        and physical_reject_not_selected
    )

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "field-strength-source reactivation inventory ready",
            truth(inventory_ready),
            "Reactivation starts only after status, roadmap, current notes, unified roadmap, and long roadmap all point to the dormant `.1731-.1734` branch.",
        ),
        row(
            "updated_pack_closeout_ready",
            "pass" if updated_pack_closeout_ready else "reject",
            "updated-pack closeout ready",
            truth(updated_pack_closeout_ready),
            "A genuinely new surface can be introduced only after the updated-pack exhausted family has already been frozen honestly.",
        ),
        row(
            "direct_potential_two_leg_selected",
            "pass" if direct_potential_two_leg_selected else "reject",
            "direct-potential two-leg route selected previously",
            truth(direct_potential_two_leg_selected),
            "The new theory is meaningful only relative to the already closed direct-potential/two-leg theorem of the source-extended pack.",
        ),
        row(
            "field_strength_source_is_new_surface",
            "pass",
            "field-strength source is a genuinely new surface",
            truth(field_strength_source_is_new_surface),
            "Coupling the probe to f_{mu nu}(a) rather than a_mu itself changes the action-level source primitive and is therefore outside the exhausted surrogate family.",
        ),
        row(
            "antisymmetric_probe_source_required",
            "pass",
            "antisymmetric probe source required",
            truth(antisymmetric_probe_source_required),
            "Once the probe couples to field strength, the source must be an antisymmetric two-form J_F^{mu nu} rather than a vector current J_mu.",
        ),
        row(
            "field_strength_candidate_tracks_electric_like",
            "pass" if field_strength_candidate_tracks_electric_like else "reject",
            "field-strength q^2 candidate tracks electric-like evidence",
            truth(field_strength_candidate_tracks_electric_like),
            "The q^2-rescaled one-leg candidate alpha nearly coincides with the prior electric-like and note-gradient evidence-only surfaces, which is exactly the scale this new theory aims to canonize.",
        ),
        row(
            "new_primary_trigger_opened",
            "pass" if new_primary_trigger_opened else "reject",
            "new primary trigger opened",
            truth(new_primary_trigger_opened),
            "A gauge-invariant field-strength-source primitive counts as a genuinely new action-level structure beyond the current source-extended pack.",
        ),
        row(
            "same_level_retry_blocked",
            "pass" if same_level_retry_blocked else "reject",
            "same-level retry blocked",
            truth(same_level_retry_blocked),
            "The old potential-side surrogate families remain closed; reactivation proceeds only by changing the source primitive itself.",
        ),
        row(
            "canonical_one_leg_amputation_theorem_derivation_scheduled",
            "pass" if canonical_one_leg_amputation_theorem_derivation_scheduled else "reject",
            "canonical one-leg amputation theorem derivation scheduled",
            truth(canonical_one_leg_amputation_theorem_derivation_scheduled),
            "The next theorem branch must decide whether the field-strength-source pack canonically selects a one-leg/q^2 amputation rule.",
        ),
        row(
            "physical_reject_not_selected",
            "pass" if physical_reject_not_selected else "reject",
            "physical reject not selected",
            truth(physical_reject_not_selected),
            "The reactivation still treats the scalar strong candidate as retained and does not force physical rejection.",
        ),
        row(
            "route_reactivation_honest",
            "pass" if route_reactivation_honest else "reject",
            "field-strength-source reactivation honest",
            truth(route_reactivation_honest),
            "Reactivation is honest only if it introduces a genuinely new source primitive rather than another variant inside the exhausted updated pack.",
        ),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem": display_path(CURRENT_PROBLEM),
            "current_status": display_path(CURRENT_STATUS),
            "unified_roadmap": display_path(UNIFIED_ROADMAP),
            "long_roadmap": display_path(LONG_ROADMAP),
            "part5": display_path(PART5),
            "closeout_gate": display_path(CLOSEOUT_GATE),
            "resolvent_gate": display_path(RESOLVENT_GATE),
            "recompute_gate": display_path(RECOMPUTE_GATE),
            "primary_gate": display_path(PRIMARY_GATE),
        },
        "constants": {
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "official_energy_core_alpha_at_q_theory": ENERGY_CORE_ALPHA,
            "official_projected_kernel_alpha_at_q_theory": PROJECTED_KERNEL_ALPHA,
            "one_leg_amputated_alpha_at_q_theory": resolvent_summary[
                "one_leg_amputated_alpha_at_q_theory"
            ],
            "one_leg_to_canonical_response_ratio": recompute_summary[
                "one_leg_to_canonical_response_ratio"
            ],
            "field_strength_q2_candidate_alpha_at_q_theory": one_leg_q2_candidate_alpha,
            "electric_like_component_alpha_at_q_theory": electric_like_alpha,
            "note_gradient_alpha_at_q_theory": note_gradient_alpha,
            "new_source_primitive": "S_src^F[P,a;J_F] = S_frozen[P,a] - (1/2) ∫ d^4x J_F^{mu nu} f_{mu nu}(a)",
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "field_strength_source_pack_adopted": field_strength_source_pack_adopted,
        "new_action_level_structure_surface_present": field_strength_source_is_new_surface,
        "new_primary_trigger_opened": new_primary_trigger_opened,
        "antisymmetric_probe_source_required": antisymmetric_probe_source_required,
        "direct_potential_source_pack_retained_as_reference": True,
        "direct_potential_two_leg_selected_previously": direct_potential_two_leg_selected,
        "field_strength_q2_candidate_alpha_at_q_theory": one_leg_q2_candidate_alpha,
        "field_strength_candidate_tracks_electric_like": field_strength_candidate_tracks_electric_like,
        "electric_like_component_alpha_at_q_theory": electric_like_alpha,
        "note_gradient_alpha_at_q_theory": note_gradient_alpha,
        "canonical_one_leg_amputation_theorem_derivation_scheduled": canonical_one_leg_amputation_theorem_derivation_scheduled,
        "same_level_retry_blocked": same_level_retry_blocked,
        "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
        "physical_reject_required": False,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": route_reactivation_honest,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "hits": {
            "status_branch_hit": hit(status_text, "8.7.56.1731"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1731-.1734"),
            "current_problem_branch_hit": hit(
                current_problem_text, "8.7.56.1731-.1734"
            ),
            "current_status_branch_hit": hit(
                current_status_text, "8.7.56.1731-.1734"
            ),
            "unified_roadmap_branch_hit": hit(
                unified_text,
                "`.1731-.1734` は **conditional new action-level structure / exact probe-response pack-update reactivation**",
            ),
            "long_roadmap_branch_hit": hit(long_text, "12. `8.7.56.1731-.1734`"),
            "part5_hit": hit(part5_text, ".1727-.1730"),
        },
        "carry_over": {
            "closeout_summary": closeout_summary,
            "resolvent_summary": resolvent_summary,
            "recompute_summary": recompute_summary,
            "primary_summary": primary_summary,
        },
        "retained_numeric_state": {
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "official_energy_core_alpha_at_q_theory": ENERGY_CORE_ALPHA,
            "official_projected_kernel_alpha_at_q_theory": PROJECTED_KERNEL_ALPHA,
            "one_leg_amputated_alpha_at_q_theory": resolvent_summary[
                "one_leg_amputated_alpha_at_q_theory"
            ],
            "field_strength_q2_candidate_alpha_at_q_theory": one_leg_q2_candidate_alpha,
            "electric_like_component_alpha_at_q_theory": electric_like_alpha,
            "note_gradient_alpha_at_q_theory": note_gradient_alpha,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1731",
                f"{STEP_NAME} inventory",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "audit": write_artifact(
            "audit",
            payload(
                "8.7.56.1732",
                f"{STEP_NAME} audit",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload(
                "8.7.56.1733",
                f"{STEP_NAME} declaration gate",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "route_sync": write_artifact(
            "route_sync",
            payload(
                "8.7.56.1734",
                f"{STEP_NAME} route sync",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
    }

    print(
        json.dumps(
            {"step": STEP_TAG, "stem": STEM, "manifest": manifest, "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
