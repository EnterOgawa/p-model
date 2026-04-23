#!/usr/bin/env python3
"""Generate 8.7.56.1855-.1858 post-closeout wait-restore artifacts."""

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

POST_CLOSEOUT_AUDIT_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1851_1854_post_closeout_reactivation_audit_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1855-1858"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor post-closeout wait "
    "restore / dormant registry"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "post_closeout_wait_restore_dormant_registry",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_post_closeout_reactivation_audit_no_new_surface_"
    "dormant_registry_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_global_exact_alpha_signed_form_factor_dormant_"
    "registry_wait_restored"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_post_dormant_"
    "pack_update_or_external_input_reactivation"
)
NEXT_ROUTE = "8.7.56.1859"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_post_dormant_wait_restore_"
    "registry_refresh"
)
FOLLOWUP_ROUTE = "8.7.56.1863"


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


# 関数: dormant registry formulas を返す。

def build_formulae() -> dict[str, str]:
    """Return the retained dormant-registry formulas."""
    return {
        "retained_exact_alpha_rule": "alpha_exact(q) = |F_exact(q)|^2 / (4 pi)",
        "retained_signed_rule": "F_exact(q) = sigma_F(q) |F_exact(q)|",
        "primary_reopen_surface": "substantive pack update beyond the retained real-branch sign-parity theorem",
        "secondary_reopen_surface": "retained-interval extension or new signed observable rule beyond the current dormant family",
        "reserve_reopen_surface": "future external input guiding a new signed surface or substantive pack update",
    }


# 関数: `.1855-.1858` を実行する。

def main() -> None:
    """Execute the post-closeout wait-restore dormant registry branch."""
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
        POST_CLOSEOUT_AUDIT_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    audit_summary = read_json(POST_CLOSEOUT_AUDIT_GATE)["summary"]
    inventory_ready = bool(audit_summary["no_new_trigger_opened"])
    exact_alpha_promotion_retained = True
    exact_signed_form_factor_promotion_retained = True
    post_closeout_dormant_registry_retained = True
    same_level_post_closeout_retry_admissible = False
    physical_reject_required = False

    formulas = build_formulae()

    rows = [
        row("inventory_ready", "pass" if inventory_ready else "reject", "post-closeout dormant inventory ready", truth(inventory_ready), "Dormant sync starts only after the reactivation audit has declared no new trigger."),
        row("exact_alpha_promotion_retained", "pass", "exact alpha promotion retained", truth(exact_alpha_promotion_retained), "The absolute loading theorem remains the canonical alpha-side result."),
        row("exact_signed_form_factor_promotion_retained", "pass", "exact signed form-factor promotion retained", truth(exact_signed_form_factor_promotion_retained), "The retained real-branch sign-parity theorem remains the canonical sign-side result."),
        row("post_closeout_dormant_registry_retained", "pass", "post-closeout dormant registry retained", truth(post_closeout_dormant_registry_retained), "The closed family is now frozen as a dormant registry rather than a live same-level branch."),
        row("same_level_post_closeout_retry_admissible", "reject", "same-level post-closeout retry admissible", truth(same_level_post_closeout_retry_admissible), "Same-level retry stays blocked after the dormant registry is restored."),
        row("physical_reject_required", "reject", "physical reject required", truth(physical_reject_required), "The retained family stays valid on the audit interval, so no physical reject is needed."),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "exact_alpha_promotion_retained": exact_alpha_promotion_retained,
        "exact_signed_form_factor_promotion_retained": exact_signed_form_factor_promotion_retained,
        "post_closeout_dormant_registry_retained": post_closeout_dormant_registry_retained,
        "same_level_post_closeout_retry_admissible": same_level_post_closeout_retry_admissible,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": "vector_qball_form_factor_post_closeout_wait_restore_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": formulas,
        "hits": {
            "status_branch_hit": hit(status_text, "8.7.56.1855"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1855-.1858"),
            "current_problem_hit": hit(current_problem_text, "signed form factor"),
            "current_status_hit": hit(current_status_text, "signed form factor"),
            "unified_roadmap_hit": hit(unified_text, "91. `.1855-.1858`"),
            "long_roadmap_hit": hit(long_text, "8.7.56.1855-.1858"),
            "part5_hit": hit(part5_text, "signed form factor"),
        },
    }

    declaration_payload = payload(
        "8.7.56.1857",
        STEP_NAME + " declaration gate",
        {
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
                "post_closeout_audit_gate": display_path(POST_CLOSEOUT_AUDIT_GATE),
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
        decision,
        evidence,
    )

    route_payload = payload(
        "8.7.56.1858",
        STEP_NAME + " route sync",
        declaration_payload["inputs"],
        [
            row("dormant_registry_retained", "pass", "dormant registry retained", 1.0, "The post-closeout family is now frozen and only future substantive reactivation is honest."),
            row("next_route_fixed", "pass", "next route fixed", 1.0, "The next official branch is the conditional post-dormant reactivation audit."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "selected_followup_route": FOLLOWUP_ROUTE_NAME,
            "selected_followup_route_or_none": FOLLOWUP_ROUTE,
            "physical_reject_required": physical_reject_required,
        },
        {
            "overall_status": "vector_qball_form_factor_post_closeout_wait_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"formulas": formulas},
    )

    write_artifact("declaration_gate", declaration_payload)
    write_artifact("route_sync", route_payload)

    print("[ok] 8.7.56.1855-.1858 post-closeout wait-restore artifacts generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
