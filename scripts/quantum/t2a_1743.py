#!/usr/bin/env python3
"""Generate 8.7.56.1743-.1746 field-strength decision-gate sync artifacts.

The field-strength-source theorem canonizes the old electric-like evidence
scale, but the direct fixed-q read still remains below the retained scalar
strong candidate. The decision question is therefore:

    does the new canonical field-strength observable promote the scalar
    candidate exactly, or does it close as a scalar-leaning partial under the
    current field-strength-source pack?

This branch fixes that official read and reorganizes the roadmap when the new
theory still fails to close exact scalar promotion.
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

THEOREM_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1735_1738_field_strength_amp_theorem_declaration_gate_metrics.json"
)
RECOMPUTE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1739_1742_field_strength_recompute_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1743-1746"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor field-strength-source "
    "decision gate / canonical promotion sync"
)
STEM = build_compact_artifact_stem(STEP_TAG, "field_strength_gate_sync", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_field_strength_source_canonical_one_leg_"
    "observable_recomputed_decision_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_field_strength_source_scalar_leaning_partial_"
    "canonical_promotion_closeout_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_field_strength_source_"
    "closeout_reopen_registry"
)
NEXT_ROUTE = "8.7.56.1747"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_internal_"
    "hamiltonian_surface_or_external_input_reactivation"
)
FOLLOWUP_ROUTE = "8.7.56.1751"


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


# 関数: `.1743-.1746` を実行する。

def main() -> None:
    """Execute the field-strength-source decision gate / sync branch."""
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
        THEOREM_GATE,
        RECOMPUTE_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    theorem_summary = read_json(THEOREM_GATE)["summary"]
    recompute_summary = read_json(RECOMPUTE_GATE)["summary"]

    canonical_field_strength_theorem_derived = bool(
        theorem_summary["canonical_field_strength_theorem_derived"]
    )
    updated_field_strength_supports_scalar_candidate = bool(
        recompute_summary["updated_field_strength_supports_scalar_candidate"]
    )
    updated_field_strength_tracks_scalar_side = bool(
        recompute_summary["updated_field_strength_tracks_scalar_side"]
    )
    updated_field_strength_canonizes_electric_like_evidence = bool(
        recompute_summary["updated_field_strength_canonizes_electric_like_evidence"]
    )

    gate_a_exact_promote_selected = updated_field_strength_supports_scalar_candidate
    gate_b_partial_scalar_leaning_selected = bool(
        canonical_field_strength_theorem_derived
        and not updated_field_strength_supports_scalar_candidate
        and updated_field_strength_tracks_scalar_side
        and updated_field_strength_canonizes_electric_like_evidence
    )
    gate_c_reject_selected = False
    field_strength_surface_exact_scalar_promotion_failed = bool(
        canonical_field_strength_theorem_derived
        and not updated_field_strength_supports_scalar_candidate
    )
    field_strength_surface_partial_promotion_retained = gate_b_partial_scalar_leaning_selected
    same_level_field_strength_retry_admissible = False
    closeout_reopen_registry_admissible_now = bool(
        field_strength_surface_exact_scalar_promotion_failed
    )

    rows = [
        row(
            "canonical_field_strength_theorem_derived",
            "pass" if canonical_field_strength_theorem_derived else "reject",
            "canonical field-strength theorem derived",
            truth(canonical_field_strength_theorem_derived),
            "The decision gate only opens after the new field-strength theorem closes canonically.",
        ),
        row(
            "gate_a_exact_promote_selected",
            "reject",
            "Gate A exact scalar promotion selected",
            truth(gate_a_exact_promote_selected),
            "The new canonical read remains substantially below the retained scalar strong candidate, so exact promotion is not honest.",
        ),
        row(
            "gate_b_partial_scalar_leaning_selected",
            "pass" if gate_b_partial_scalar_leaning_selected else "reject",
            "Gate B partial scalar-leaning canonical promotion selected",
            truth(gate_b_partial_scalar_leaning_selected),
            "The field-strength theorem canonizes the old electric-like evidence and moves the canonical read decisively onto the scalar side without closing exact promotion.",
        ),
        row(
            "gate_c_reject_selected",
            "reject",
            "Gate C reject selected",
            truth(gate_c_reject_selected),
            "Physical rejection is still not selected because the scalar candidate remains retained and the new surface yields a nontrivial scalar-leaning canonical response.",
        ),
        row(
            "field_strength_surface_exact_scalar_promotion_failed",
            "pass" if field_strength_surface_exact_scalar_promotion_failed else "reject",
            "field-strength exact scalar promotion failed",
            truth(field_strength_surface_exact_scalar_promotion_failed),
            "The new theorem does not close the final scalar-promotion gap under the current field-strength-source pack.",
        ),
        row(
            "field_strength_surface_partial_promotion_retained",
            "pass" if field_strength_surface_partial_promotion_retained else "reject",
            "field-strength partial canonical promotion retained",
            truth(field_strength_surface_partial_promotion_retained),
            "The honest read is no longer vector-no-go-like; it is a scalar-leaning partial canonical promotion.",
        ),
        row(
            "same_level_field_strength_retry_admissible",
            "reject",
            "same-level field-strength retry admissible",
            truth(same_level_field_strength_retry_admissible),
            "The external source theorem is now closed, so the next honest move is closeout / reopen registry rather than another same-level external-source variant.",
        ),
        row(
            "closeout_reopen_registry_admissible_now",
            "pass" if closeout_reopen_registry_admissible_now else "reject",
            "field-strength closeout / reopen registry admissible now",
            truth(closeout_reopen_registry_admissible_now),
            "Once exact promotion fails but partial scalar-leaning canonization is fixed, the next honest branch is closeout / reopen registry.",
        ),
        row(
            "physical_reject_not_selected",
            "pass",
            "physical reject not selected",
            1.0,
            "The retained scalar strong candidate remains live, so the branch closes as partial rather than reject.",
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
            "theorem_gate": display_path(THEOREM_GATE),
            "recompute_gate": display_path(RECOMPUTE_GATE),
        },
        "constants": {
            "updated_field_strength_alpha_at_q_theory": recompute_summary["updated_field_strength_alpha_at_q_theory"],
            "updated_field_strength_vs_scalar_alpha_rel_gap": recompute_summary["updated_field_strength_vs_scalar_alpha_rel_gap"],
            "updated_field_strength_vs_vector_alpha_rel_gap": recompute_summary["updated_field_strength_vs_vector_alpha_rel_gap"],
            "updated_field_strength_vs_electric_like_rel_gap": recompute_summary["updated_field_strength_vs_electric_like_rel_gap"],
            "updated_field_strength_vs_note_gradient_rel_gap": recompute_summary["updated_field_strength_vs_note_gradient_rel_gap"],
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "field_strength_source_pack_adopted": True,
        "canonical_field_strength_theorem_derived": canonical_field_strength_theorem_derived,
        "field_strength_surface_exact_scalar_promotion_failed": field_strength_surface_exact_scalar_promotion_failed,
        "field_strength_surface_partial_promotion_retained": field_strength_surface_partial_promotion_retained,
        "gate_a_exact_promote_selected": gate_a_exact_promote_selected,
        "gate_b_partial_scalar_leaning_selected": gate_b_partial_scalar_leaning_selected,
        "gate_c_reject_selected": gate_c_reject_selected,
        "selected_primary_decision_gate": "gate_b_partial_scalar_leaning_not_exact",
        "updated_field_strength_alpha_at_q_theory": recompute_summary["updated_field_strength_alpha_at_q_theory"],
        "updated_field_strength_vs_scalar_alpha_rel_gap": recompute_summary["updated_field_strength_vs_scalar_alpha_rel_gap"],
        "updated_field_strength_vs_vector_alpha_rel_gap": recompute_summary["updated_field_strength_vs_vector_alpha_rel_gap"],
        "updated_field_strength_canonizes_electric_like_evidence": updated_field_strength_canonizes_electric_like_evidence,
        "updated_field_strength_tracks_scalar_side": updated_field_strength_tracks_scalar_side,
        "same_level_field_strength_retry_admissible": same_level_field_strength_retry_admissible,
        "field_strength_closeout_reopen_registry_admissible_now": closeout_reopen_registry_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": closeout_reopen_registry_admissible_now,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "hits": {
            "status_current_branch": hit(status_text, "8.7.56.1743"),
            "roadmap_current_branch": hit(roadmap_text, "8.7.56.1743-.1746"),
            "current_problem_current_branch": hit(
                current_problem_text, "8.7.56.1743-.1746"
            ),
            "current_status_current_branch": hit(
                current_status_text, "8.7.56.1743-.1746"
            ),
            "unified_roadmap_current_branch": hit(
                unified_text,
                "`.1743-.1746` は **field-strength-source decision gate / canonical promotion sync**",
            ),
            "long_roadmap_current_branch": hit(long_text, "15. `8.7.56.1743-.1746`"),
            "part5_field_strength_hit": hit(part5_text, ".1731-.1734"),
        },
        "prior_summaries": {
            "theorem": theorem_summary,
            "recompute": recompute_summary,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1743",
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
                "8.7.56.1744",
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
                "8.7.56.1745",
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
                "8.7.56.1746",
                f"{STEP_NAME} route sync",
                inputs,
                rows,
                summary,
                decision,
                {"manifest": "written below"},
            ),
        ),
    }

    route_sync_path = PUBLIC_OUT / f"{STEM}_route_sync_metrics.json"
    route_sync_payload = read_json(route_sync_path)
    route_sync_payload["evidence"] = {
        "manifest": manifest,
        "prior_summaries": {
            "theorem": theorem_summary,
            "recompute": recompute_summary,
        },
    }
    route_sync_path.write_text(
        json.dumps(route_sync_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"stem": STEM, "artifacts": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
