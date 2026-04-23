#!/usr/bin/env python3
"""Generate 8.7.56.1795-.1798 branch-local completion closeout artifacts.

`.1791-.1794` introduced a branch-local completion theorem that exactly fixes
the missing HH diagonal *at the retained matching point* q_theory:

    lambda_+(q_theory) = F_exact(q_theory),
    A_HH,exact(q_theory) = F_exact(q_theory) - A_FF(q_theory).

This closes exact scalar promotion at fixed q, but it still does not derive a
full-q HH surface. The present branch freezes that status honestly:

1. fixed-q exact scalar promotion is retained officially,
2. the remaining missing bridge becomes the full-q HH surface,
3. non-rank-one mixed surfaces remain secondary reopen routes,
4. same-level branch-local completion retries are blocked.
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

REACTIVATION_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1791_1794_hh_surface_reactivation_declaration_gate_metrics.json"
REACTIVATION_ROUTE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1791_1794_hh_surface_reactivation_route_sync_metrics.json"
MIXED_CLOSEOUT_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1787_1790_mixed_proxy_closeout_registry_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.1795-1798"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor branch-local completion "
    "closeout / reopen registry"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "branch_local_completion_closeout",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_branch_local_completion_theorem_exact_hh_point_"
    "fixed_exact_scalar_promotion_closeout_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_branch_local_completion_exact_scalar_promotion_"
    "closeout_full_q_hh_surface_reopen_registry_completed"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_full_q_hh_"
    "surface_or_non_rank_one_mixed_surface_generalization"
)
NEXT_ROUTE = "8.7.56.1799"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を検査する。

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
    """Return the closeout / reopen registry formulas."""
    return {
        "retained_branch_local_rule": "lambda_+(q_theory) = F_exact(q_theory),  A_HH,exact(q_theory) = F_exact(q_theory) - A_FF(q_theory)",
        "primary_reopen_surface": "full-q exact HH surface under the retained rank-one completion pack",
        "secondary_reopen_surface": "non-rank-one mixed surface beyond the branch-local rank-one completion pack",
        "reserve_reopen_surface": "future external input or pack update guiding full-q HH or non-rank-one mixed surfaces",
    }


# 関数: `.1795-.1798` を実行する。

def main() -> None:
    """Execute the branch-local completion closeout / reopen registry branch."""
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
        REACTIVATION_GATE,
        REACTIVATION_ROUTE,
        MIXED_CLOSEOUT_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    reactivation_summary = read_json(REACTIVATION_GATE)["summary"]
    reactivation_route = read_json(REACTIVATION_ROUTE)["summary"]
    mixed_closeout_summary = read_json(MIXED_CLOSEOUT_GATE)["summary"]

    inventory_ready = all(
        item is not None
        for item in (
            hit(status_text, "8.7.56.1795"),
            hit(roadmap_text, "次の公式 branch は `.1795-.1798`"),
            hit(current_problem_text, "branch-local completion closeout / reopen registry"),
            hit(current_status_text, "branch-local completion closeout / reopen registry"),
            hit(unified_text, "`.1791-.1794` は **conditional exact HH surface or non-rank-one mixed surface reactivation**"),
            hit(long_text, "28. `8.7.56.1795-.1798`"),
            hit(part5_text, "`.1791-.1794` の **conditional exact HH surface or non-rank-one mixed surface reactivation**"),
        )
    )
    fixed_q_exact_scalar_promotion_retained = bool(
        reactivation_summary["exact_scalar_promotion_selected"]
        and reactivation_summary["branch_local_completion_only"]
        and not reactivation_summary["full_q_exact_hh_surface_available"]
    )
    full_q_exact_hh_surface_missing = bool(
        not reactivation_summary["full_q_exact_hh_surface_available"]
    )
    same_level_branch_local_completion_retry_admissible = False
    primary_reopen_surface_fixed = True
    secondary_reopen_surface_fixed = True
    reserve_reopen_surface_fixed = True
    branch_local_closeout_honest = all(
        (
            inventory_ready,
            fixed_q_exact_scalar_promotion_retained,
            full_q_exact_hh_surface_missing,
            not same_level_branch_local_completion_retry_admissible,
            primary_reopen_surface_fixed,
            secondary_reopen_surface_fixed,
            reserve_reopen_surface_fixed,
        )
    )

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "branch-local completion closeout inventory ready",
            truth(inventory_ready),
            "Closeout starts only after the live docs already point to `.1795-.1798` as the next official branch.",
        ),
        row(
            "fixed_q_exact_scalar_promotion_retained",
            "pass" if fixed_q_exact_scalar_promotion_retained else "reject",
            "fixed-q exact scalar promotion retained",
            truth(fixed_q_exact_scalar_promotion_retained),
            "The closeout is only honest if the fixed-q completion theorem is retained explicitly as an exact result.",
        ),
        row(
            "exact_alpha_mix_at_q_theory",
            "watch",
            "exact mixed alpha at q_theory",
            reactivation_summary["exact_alpha_mix_at_q_theory"],
            "The retained branch-local canonical promotion exactly reproduces the scalar strong candidate at q_theory.",
        ),
        row(
            "exact_hh_amplitude_at_q_theory",
            "watch",
            "exact HH amplitude at q_theory",
            reactivation_summary["exact_hh_amplitude_at_q_theory"],
            "This is the fixed-q HH diagonal complement derived by the branch-local completion theorem.",
        ),
        row(
            "proxy_to_exact_hh_ratio",
            "watch",
            "proxy HH / exact HH ratio",
            reactivation_summary["proxy_to_exact_hh_ratio"],
            "The old HH proxy overshoots the exact fixed-q completion point by about 1.449x.",
        ),
        row(
            "full_q_exact_hh_surface_missing",
            "pass" if full_q_exact_hh_surface_missing else "reject",
            "full-q exact HH surface missing",
            truth(full_q_exact_hh_surface_missing),
            "The remaining unresolved bridge is the general q-dependent HH surface, not the fixed-q completion point.",
        ),
        row(
            "same_level_branch_local_completion_retry_admissible",
            "reject",
            "same-level branch-local completion retry admissible",
            truth(same_level_branch_local_completion_retry_admissible),
            "The fixed-q completion theorem should not be rerun again without a full-q HH theorem or a genuinely new mixed surface.",
        ),
        row(
            "primary_reopen_surface_fixed",
            "pass" if primary_reopen_surface_fixed else "reject",
            "primary reopen surface fixed",
            truth(primary_reopen_surface_fixed),
            "Primary reopen surface = full-q exact HH surface under the retained rank-one completion pack.",
        ),
        row(
            "secondary_reopen_surface_fixed",
            "pass" if secondary_reopen_surface_fixed else "reject",
            "secondary reopen surface fixed",
            truth(secondary_reopen_surface_fixed),
            "Secondary reopen surface = genuinely new non-rank-one mixed surface beyond the branch-local rank-one completion pack.",
        ),
        row(
            "reserve_reopen_surface_fixed",
            "pass" if reserve_reopen_surface_fixed else "reject",
            "reserve reopen surface fixed",
            truth(reserve_reopen_surface_fixed),
            "Reserve reopen surface = future external input or pack update that guides full-q HH or non-rank-one mixed generalization.",
        ),
        row(
            "branch_local_closeout_honest",
            "pass" if branch_local_closeout_honest else "reject",
            "branch-local completion closeout honest",
            truth(branch_local_closeout_honest),
            "The closeout is honest only if it retains exact fixed-q promotion while explicitly refusing to over-claim a full-q HH surface.",
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
            "reactivation_gate": display_path(REACTIVATION_GATE),
            "reactivation_route": display_path(REACTIVATION_ROUTE),
            "mixed_closeout_gate": display_path(MIXED_CLOSEOUT_GATE),
        },
        "constants": {
            "exact_hh_amplitude_at_q_theory": reactivation_summary["exact_hh_amplitude_at_q_theory"],
            "exact_hh_alpha_at_q_theory": reactivation_summary["exact_hh_alpha_at_q_theory"],
            "exact_fh_amplitude_at_q_theory": reactivation_summary["exact_fh_amplitude_at_q_theory"],
            "exact_lambda_plus_at_q_theory": reactivation_summary["exact_lambda_plus_at_q_theory"],
            "exact_alpha_mix_at_q_theory": reactivation_summary["exact_alpha_mix_at_q_theory"],
            "proxy_to_exact_hh_ratio": reactivation_summary["proxy_to_exact_hh_ratio"],
            "primary_reopen_surface": "full_q_exact_hh_surface_under_retained_rank_one_completion_pack",
            "secondary_reopen_surface": "non_rank_one_mixed_surface_beyond_branch_local_rank_one_completion_pack",
            "reserve_reopen_surface": "future_external_input_or_pack_update_guiding_full_q_hh_or_non_rank_one_surface",
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "fixed_q_exact_scalar_promotion_retained": fixed_q_exact_scalar_promotion_retained,
        "exact_hh_amplitude_at_q_theory": reactivation_summary["exact_hh_amplitude_at_q_theory"],
        "exact_hh_alpha_at_q_theory": reactivation_summary["exact_hh_alpha_at_q_theory"],
        "exact_fh_amplitude_at_q_theory": reactivation_summary["exact_fh_amplitude_at_q_theory"],
        "exact_lambda_plus_at_q_theory": reactivation_summary["exact_lambda_plus_at_q_theory"],
        "exact_alpha_mix_at_q_theory": reactivation_summary["exact_alpha_mix_at_q_theory"],
        "exact_scalar_promotion_selected": reactivation_summary["exact_scalar_promotion_selected"],
        "proxy_to_exact_hh_ratio": reactivation_summary["proxy_to_exact_hh_ratio"],
        "full_q_exact_hh_surface_missing": full_q_exact_hh_surface_missing,
        "same_level_branch_local_completion_retry_admissible": same_level_branch_local_completion_retry_admissible,
        "selected_primary_reopen_surface": "full_q_exact_hh_surface_under_retained_rank_one_completion_pack",
        "selected_secondary_reopen_surface": "non_rank_one_mixed_surface_beyond_branch_local_rank_one_completion_pack",
        "selected_reserve_reopen_surface": "future_external_input_or_pack_update_guiding_full_q_hh_or_non_rank_one_surface",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": branch_local_closeout_honest,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": build_formulae(),
        "hits": {
            "status_branch_hit": hit(status_text, "8.7.56.1795"),
            "roadmap_branch_hit": hit(roadmap_text, "次の公式 branch は `.1795-.1798`"),
            "current_problem_hit": hit(current_problem_text, "branch-local completion closeout / reopen registry"),
            "current_status_hit": hit(current_status_text, "branch-local completion closeout / reopen registry"),
            "unified_roadmap_hit": hit(unified_text, "`.1791-.1794` は **conditional exact HH surface or non-rank-one mixed surface reactivation**"),
            "long_roadmap_hit": hit(long_text, "28. `8.7.56.1795-.1798`"),
            "part5_hit": hit(part5_text, "`.1791-.1794` の **conditional exact HH surface or non-rank-one mixed surface reactivation**"),
        },
        "carry_over": {
            "reactivation_summary": reactivation_summary,
            "reactivation_route": reactivation_route,
            "mixed_closeout_summary": mixed_closeout_summary,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1795",
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
                "8.7.56.1796",
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
                "8.7.56.1797",
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
                "8.7.56.1798",
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
