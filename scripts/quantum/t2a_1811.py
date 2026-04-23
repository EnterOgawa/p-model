#!/usr/bin/env python3
"""Generate 8.7.56.1811-.1814 global-completion obstruction closeout artifacts.

`.1807-.1810` proved that the retained scalar-compatible full-q HH window
cannot be extended globally while simultaneously keeping

1. the vacuum-saturated carrier surface `A_FF(q)=|q|`, and
2. the same real-symmetric canonical eigenvalue read.

This branch freezes that sharpened reopen structure:

1. same-level post-window retries are blocked,
2. generic non-rank-one reopening is demoted,
3. the only honest next surfaces are carrier-breaking or canonical-rule-
   breaking updates beyond the retained obstruction theorem pack.
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

OBSTRUCTION_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1807_1810_global_hh_obstruction_theorem_declaration_gate_metrics.json"
OBSTRUCTION_ROUTE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1807_1810_global_hh_obstruction_theorem_route_sync_metrics.json"
WINDOW_CLOSEOUT_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1803_1806_full_q_hh_window_closeout_registry_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.1811-1814"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor global completion "
    "obstruction closeout / reopen registry"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "global_completion_obstruction_closeout_registry",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_global_completion_obstructed_under_vacuum_"
    "saturated_psd_mixed_pack_route_reset_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_global_completion_obstruction_closeout_"
    "carrier_or_canonical_rule_breaking_reopen_registry_completed"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_axiom_breaking_"
    "mixed_surface_reactivation"
)
NEXT_ROUTE = "8.7.56.1815"


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


# 関数: repo相対の表示パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative display path when possible."""
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


# 関数: JSON/CSV artifact を書き出す。

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one CSV rows file."""
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


# 関数: reopen ordering の説明式を返す。

def build_formulae() -> dict[str, str]:
    """Return the global-completion obstruction closeout formulas."""
    return {
        "retained_window_theorem": "A_FF(q) = |q|, A_HH(q) = F_exact(q) - |q| on 0 <= q <= q_HH,max",
        "obstruction_theorem": "lambda_+(q) >= A_FF(q) = |q| for the retained real-symmetric carrier pack",
        "primary_reopen_surface": "carrier-breaking or canonical-rule-breaking surface beyond the retained obstruction theorem pack",
        "secondary_reopen_surface": "substantive pack update changing the external carrier theorem or the canonical observable rule",
        "reserve_reopen_surface": "future external input guiding a carrier-breaking or canonical-rule-breaking global completion surface",
    }


# 関数: `.1811-.1814` を実行する。

def main() -> None:
    """Execute the global-completion obstruction closeout / reopen registry branch."""
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
        OBSTRUCTION_GATE,
        OBSTRUCTION_ROUTE,
        WINDOW_CLOSEOUT_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    obstruction_summary = read_json(OBSTRUCTION_GATE)["summary"]
    obstruction_route = read_json(OBSTRUCTION_ROUTE)["summary"]
    window_closeout_summary = read_json(WINDOW_CLOSEOUT_GATE)["summary"]

    inventory_ready = all(
        (
            hit(current_problem_text, "full-q exact HH surface"),
            hit(current_status_text, "full-q HH surface"),
            hit(unified_text, "`.1799-.1802`"),
            hit(long_text, "30. `8.7.56.1803-.1806`"),
            hit(part5_text, "`.1795-.1806`"),
            window_closeout_summary["full_q_window_retained"],
            obstruction_summary["psd_mixed_pack_lower_bound_theorem_derived"],
        )
    )
    full_q_window_retained = bool(
        window_closeout_summary["full_q_window_retained"]
        and window_closeout_summary["exact_scalar_promotion_selected"]
    )
    obstruction_retained = bool(
        obstruction_summary["psd_mixed_pack_lower_bound_theorem_derived"]
        and not obstruction_summary["global_exact_completion_under_vacuum_saturated_psd_pack"]
        and obstruction_summary["non_rank_one_mixed_surface_alone_not_sufficient"]
    )
    same_level_post_window_retry_admissible = False
    primary_reopen_surface_fixed = True
    secondary_reopen_surface_fixed = True
    reserve_reopen_surface_fixed = True
    branch_honest = all(
        (
            inventory_ready,
            full_q_window_retained,
            obstruction_retained,
            not same_level_post_window_retry_admissible,
            primary_reopen_surface_fixed,
            secondary_reopen_surface_fixed,
            reserve_reopen_surface_fixed,
        )
    )

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "global completion obstruction closeout inventory ready",
            truth(inventory_ready),
            "Closeout starts only after the obstruction theorem has already reset the route away from generic post-window retries.",
        ),
        row(
            "full_q_window_retained",
            "pass" if full_q_window_retained else "reject",
            "full-q HH window retained",
            truth(full_q_window_retained),
            "The scalar-compatible HH window remains a retained exact result after the obstruction theorem.",
        ),
        row(
            "obstruction_retained",
            "pass" if obstruction_retained else "reject",
            "obstruction theorem retained",
            truth(obstruction_retained),
            "The retained obstruction is that vacuum-saturated same-rule global completion is blocked even before a generic non-rank-one retry is considered.",
        ),
        row(
            "q_hh_max_over_m0",
            "watch",
            "retained scalar-compatible window edge q_HH,max/m0",
            float(obstruction_summary["q_hh_max_over_m0"]),
            "This is the edge up to which the exact HH window is retained before the global obstruction activates.",
        ),
        row(
            "deficit_at_q_probe",
            "watch",
            "carrier minus exact amplitude at the post-window probe",
            float(obstruction_summary["deficit_at_q_probe"]),
            "The positive deficit at the retained post-window probe is the concrete numerical witness for the obstruction theorem.",
        ),
        row(
            "non_rank_one_mixed_surface_alone_not_sufficient",
            "pass",
            "non-rank-one mixed surface alone not sufficient",
            truth(obstruction_summary["non_rank_one_mixed_surface_alone_not_sufficient"]),
            "Generic non-rank-one reopening is demoted because the obstruction theorem already survives without the rank-one assumption.",
        ),
        row(
            "same_level_post_window_retry_admissible",
            "reject",
            "same-level post-window retry admissible",
            truth(same_level_post_window_retry_admissible),
            "The next honest reopening must break the retained carrier or canonical-rule axioms, not merely retry the same post-window family.",
        ),
        row(
            "primary_reopen_surface_fixed",
            "pass",
            "primary reopen surface fixed",
            truth(primary_reopen_surface_fixed),
            "Primary reopen surface = carrier-breaking or canonical-rule-breaking surface beyond the retained obstruction theorem pack.",
        ),
        row(
            "secondary_reopen_surface_fixed",
            "pass",
            "secondary reopen surface fixed",
            truth(secondary_reopen_surface_fixed),
            "Secondary reopen surface = substantive pack update that changes the external carrier theorem or the canonical observable rule.",
        ),
        row(
            "reserve_reopen_surface_fixed",
            "pass",
            "reserve reopen surface fixed",
            truth(reserve_reopen_surface_fixed),
            "Reserve reopen surface = future external input guiding a carrier-breaking or canonical-rule-breaking global completion surface.",
        ),
        row(
            "branch_honest",
            "pass" if branch_honest else "reject",
            "global completion obstruction closeout honest",
            truth(branch_honest),
            "The closeout is honest only if it retains the window result while sharply narrowing future retries to carrier- or rule-breaking surfaces.",
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
            "obstruction_gate": display_path(OBSTRUCTION_GATE),
            "obstruction_route": display_path(OBSTRUCTION_ROUTE),
            "window_closeout_gate": display_path(WINDOW_CLOSEOUT_GATE),
        },
        "constants": {
            "q_hh_max_over_m0": float(obstruction_summary["q_hh_max_over_m0"]),
            "selected_primary_reopen_surface": "carrier_breaking_or_canonical_rule_breaking_surface_beyond_retained_obstruction_pack",
            "selected_secondary_reopen_surface": "substantive_pack_update_changing_external_carrier_theorem_or_canonical_observable_rule",
            "selected_reserve_reopen_surface": "future_external_input_guiding_carrier_breaking_or_rule_breaking_global_completion_surface",
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "full_q_window_retained": full_q_window_retained,
        "obstruction_retained": obstruction_retained,
        "q_hh_max_over_m0": float(obstruction_summary["q_hh_max_over_m0"]),
        "q_probe_over_m0": float(obstruction_summary["q_probe_over_m0"]),
        "deficit_at_q_probe": float(obstruction_summary["deficit_at_q_probe"]),
        "post_window_positive_gap_detected": bool(obstruction_summary["post_window_positive_gap_detected"]),
        "same_level_post_window_retry_admissible": same_level_post_window_retry_admissible,
        "selected_primary_reopen_surface": "carrier_breaking_or_canonical_rule_breaking_surface_beyond_retained_obstruction_pack",
        "selected_secondary_reopen_surface": "substantive_pack_update_changing_external_carrier_theorem_or_canonical_observable_rule",
        "selected_reserve_reopen_surface": "future_external_input_guiding_carrier_breaking_or_rule_breaking_global_completion_surface",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": branch_honest,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": build_formulae(),
        "hits": {
            "status_branch_hit": hit(status_text, "8.7.56.1795"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1795-.1798"),
            "current_problem_hit": hit(current_problem_text, "branch-local completion theorem"),
            "current_status_hit": hit(current_status_text, "fixed-q exact promotion succeeded"),
            "unified_roadmap_hit": hit(unified_text, "`.1799-.1802`"),
            "long_roadmap_hit": hit(long_text, "30. `8.7.56.1803-.1806`"),
            "part5_hit": hit(part5_text, "`.1791-.1794`"),
        },
        "carry_over": {
            "obstruction_summary": obstruction_summary,
            "obstruction_route": obstruction_route,
            "window_closeout_summary": window_closeout_summary,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload("8.7.56.1811", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence),
        ),
        "audit": write_artifact(
            "audit",
            payload("8.7.56.1812", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload("8.7.56.1813", f"{STEP_NAME} declaration gate", inputs, rows, summary, decision, evidence),
        ),
        "route_sync": write_artifact(
            "route_sync",
            payload("8.7.56.1814", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence),
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
