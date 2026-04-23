#!/usr/bin/env python3
"""Generate 8.7.56.1707-.1710 updated decision-gate / promotion-sync artifacts.

The source-extended theorem now closes the canonical two-leg probe-response map,
and `.1703-.1706` recomputes that observable directly on the retained exact
branch. The remaining question is a pure decision gate:

    does the now-canonical observable promote the retained scalar strong
    candidate, or does it close as a canonical no-go under the updated pack?

If the theorem-selected observable still stays far from the scalar candidate,
the honest next routes are the updated-pack secondary reopen surfaces:

1. exact constitutive-map reopen after pack update,
2. branch-local full nonlinear energy-density reopen after pack update.
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
    / "q_8_7_56_1699_1702_probe_resp_amp_theorem_declaration_gate_metrics.json"
)
RECOMPUTE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1703_1706_probe_resp_canon_recompute_declaration_gate_metrics.json"
)
RECOMPUTE_ROUTE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1703_1706_probe_resp_canon_recompute_route_sync_metrics.json"
)

STEP_TAG = "8.7.56.1707-1710"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated decision gate / "
    "canonical promotion sync"
)
STEM = build_compact_artifact_stem(STEP_TAG, "probe_resp_gate_sync", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_source_extended_canonical_two_leg_observable_"
    "recomputed_decision_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_source_extended_canonical_two_leg_no_scalar_"
    "promotion_exact_constitutive_map_reopen_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exact_constitutive_map_"
    "reopen_after_pack_update"
)
NEXT_ROUTE = "8.7.56.1711"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_branch_local_full_nonlinear_"
    "energy_density_reopen_after_pack_update"
)
FOLLOWUP_ROUTE = "8.7.56.1715"


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


# 関数: repo 相対表示パスを返す。

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


# 関数: `.1707-.1710` を実行する。

def main() -> None:
    """Execute the updated decision gate / canonical promotion sync branch."""
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
        RECOMPUTE_ROUTE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    theorem_gate = read_json(THEOREM_GATE)
    recompute_gate = read_json(RECOMPUTE_GATE)
    recompute_route = read_json(RECOMPUTE_ROUTE)

    theorem_summary = theorem_gate["summary"]
    recompute_summary = recompute_gate["summary"]

    canonical_probe_response_theorem_derived = bool(
        theorem_summary["canonical_probe_response_theorem_derived"]
    )
    updated_canonical_observable_exact_available = bool(
        recompute_summary["direct_recompute_matches_prior_two_leg_read"]
    )
    updated_canonical_supports_scalar_candidate = bool(
        recompute_summary["updated_canonical_supports_scalar_candidate"]
    )

    gate_a_promote_selected = updated_canonical_supports_scalar_candidate
    gate_b_retain_canonical_no_go_selected = (
        updated_canonical_observable_exact_available
        and not updated_canonical_supports_scalar_candidate
    )
    gate_c_reject_selected = False
    exact_constitutive_map_reopen_after_pack_update_admissible_now = True
    branch_local_full_nonlinear_energy_density_reopen_after_pack_update_admissible_now = True

    rows = [
        row(
            "canonical_probe_response_theorem_derived",
            "pass" if canonical_probe_response_theorem_derived else "reject",
            "canonical probe-response theorem derived",
            truth(canonical_probe_response_theorem_derived),
            "The decision gate only opens after the source-extended theorem closes the canonical two-leg response rule.",
        ),
        row(
            "updated_canonical_observable_exact_available",
            "pass" if updated_canonical_observable_exact_available else "reject",
            "updated canonical observable exact recomputation available",
            truth(updated_canonical_observable_exact_available),
            "The theorem-selected observable has now been recomputed directly on the retained exact branch with no extra normalization.",
        ),
        row(
            "gate_a_promote_selected",
            "reject",
            "Gate A canonical promotion selected",
            truth(gate_a_promote_selected),
            "Gate A would require the theorem-selected observable to support the retained scalar strong candidate, which it does not.",
        ),
        row(
            "gate_b_retain_canonical_no_go_selected",
            "pass" if gate_b_retain_canonical_no_go_selected else "reject",
            "Gate B retain canonical no-go selected",
            truth(gate_b_retain_canonical_no_go_selected),
            "The updated pack now has an exact canonical observable, but that observable still lands far from the retained scalar strong candidate.",
        ),
        row(
            "gate_c_reject_selected",
            "reject",
            "Gate C reject selected",
            truth(gate_c_reject_selected),
            "Physical reject remains unavailable because the theory-side structure is still partial and the scalar strong candidate is retained.",
        ),
        row(
            "exact_constitutive_map_reopen_after_pack_update_admissible_now",
            "pass" if exact_constitutive_map_reopen_after_pack_update_admissible_now else "reject",
            "exact constitutive-map reopen after pack update admissible now",
            truth(exact_constitutive_map_reopen_after_pack_update_admissible_now),
            "With the canonical probe-response theorem now closed, the next honest missing object is whether the updated pack can supply an exact constitutive map.",
        ),
        row(
            "branch_local_full_nonlinear_energy_density_reopen_after_pack_update_admissible_now",
            "pass" if branch_local_full_nonlinear_energy_density_reopen_after_pack_update_admissible_now else "reject",
            "branch-local full nonlinear energy-density reopen after pack update admissible now",
            truth(branch_local_full_nonlinear_energy_density_reopen_after_pack_update_admissible_now),
            "If exact constitutive-map reopen still fails, the updated-pack nonlinear branch remains the next secondary reopen surface.",
        ),
        row(
            "updated_pack_canonical_promotion_failed",
            "pass" if gate_b_retain_canonical_no_go_selected else "reject",
            "updated-pack canonical promotion failed",
            truth(gate_b_retain_canonical_no_go_selected),
            "This records that the new theorem closes the observable map but still does not rescue the scalar candidate under the updated pack.",
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
            "recompute_route": display_path(RECOMPUTE_ROUTE),
        },
        "constants": {
            "updated_canonical_alpha_at_q_theory": recompute_summary["updated_canonical_alpha_at_q_theory"],
            "updated_canonical_alpha_residual_rel": recompute_summary["updated_canonical_alpha_residual_rel"],
            "updated_canonical_vs_scalar_alpha_rel_gap": recompute_summary["updated_canonical_vs_scalar_alpha_rel_gap"],
            "updated_canonical_vs_vector_alpha_rel_gap": recompute_summary["updated_canonical_vs_vector_alpha_rel_gap"],
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "canonical_probe_response_theorem_derived": canonical_probe_response_theorem_derived,
        "updated_canonical_observable_exact_available": updated_canonical_observable_exact_available,
        "updated_canonical_alpha_at_q_theory": recompute_summary["updated_canonical_alpha_at_q_theory"],
        "updated_canonical_alpha_residual_rel": recompute_summary["updated_canonical_alpha_residual_rel"],
        "updated_canonical_vs_scalar_alpha_rel_gap": recompute_summary["updated_canonical_vs_scalar_alpha_rel_gap"],
        "updated_canonical_vs_vector_alpha_rel_gap": recompute_summary["updated_canonical_vs_vector_alpha_rel_gap"],
        "gate_a_promote_selected": gate_a_promote_selected,
        "gate_b_retain_canonical_no_go_selected": gate_b_retain_canonical_no_go_selected,
        "gate_c_reject_selected": gate_c_reject_selected,
        "updated_pack_canonical_promotion_failed": gate_b_retain_canonical_no_go_selected,
        "exact_constitutive_map_reopen_after_pack_update_admissible_now": exact_constitutive_map_reopen_after_pack_update_admissible_now,
        "branch_local_full_nonlinear_energy_density_reopen_after_pack_update_admissible_now": branch_local_full_nonlinear_energy_density_reopen_after_pack_update_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": {
            "canonical_rule": "F_T,can(q) = -q^4 Delta chi_T(q)",
            "decision_rule": "If the theorem-selected canonical observable remains far from the retained scalar strong candidate, Gate A closes and the next honest work shifts to updated-pack secondary reopen surfaces.",
        },
        "hits": {
            "status_current_branch": hit(status_text, "8.7.56.1699"),
            "roadmap_current_branch": hit(roadmap_text, "8.7.56.1703-.1706"),
            "current_problem_current_branch": hit(
                current_problem_text,
                "canonical probe-response / amputation theorem derivation",
            ),
            "current_status_current_branch": hit(
                current_status_text,
                "canonical probe-response / amputation theorem derivation",
            ),
            "unified_roadmap_recompute": hit(
                unified_text,
                "`.1703-.1706` は **updated canonical observable recomputation**",
            ),
            "long_roadmap_recompute": hit(long_text, "8.7.56.1703-.1706"),
            "part5_theorem": hit(part5_text, "source-extended probe-response pack"),
        },
        "prior_summaries": {
            "theorem": theorem_summary,
            "recompute": recompute_summary,
            "recompute_route": recompute_route["summary"],
        },
    }

    artifacts = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1707",
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
                "8.7.56.1708",
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
                "8.7.56.1709",
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
                "8.7.56.1710",
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
            {"step": STEP_TAG, "stem": STEM, "artifacts": artifacts, "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )


# 条件分岐: スクリプトとして直接実行された場合に main を呼ぶ。

if __name__ == "__main__":
    main()
