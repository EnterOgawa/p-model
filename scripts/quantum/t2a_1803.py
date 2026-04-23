#!/usr/bin/env python3
"""Generate 8.7.56.1803-.1806 full-q HH window closeout artifacts.

`.1799-.1802` upgraded the fixed-q branch-local completion theorem into a
scalar-compatible full-q HH window:

    A_FF,exact(q) = |q|,
    A_HH,exact(q) = F_exact(q) - |q|,
    A_FH,exact(q) = sqrt(|q| (F_exact(q) - |q|)),

on the retained interval `0 <= q <= q_HH,max`, with `q_theory` inside that
window.  This closes exact scalar promotion on the retained branch and on the
entire scalar-compatible window, but it still does not derive either

1. a global all-q HH diagonal surface, or
2. a genuinely non-rank-one mixed surface.

This branch freezes that status machine-readably and blocks same-level retries.
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

WINDOW_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1799_1802_full_q_hh_window_generalization_declaration_gate_metrics.json"
WINDOW_ROUTE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1799_1802_full_q_hh_window_generalization_route_sync_metrics.json"
CLOSEOUT_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1795_1798_branch_local_completion_closeout_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.1803-1806"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor full-q HH window "
    "closeout / reopen registry"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "full_q_hh_window_closeout_registry",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_vacuum_saturated_full_q_hh_window_theorem_"
    "derived_closeout_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_vacuum_saturated_full_q_hh_window_closeout_"
    "global_hh_or_non_rank_one_reopen_registry_completed"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_global_hh_"
    "surface_or_non_rank_one_mixed_generalization"
)
NEXT_ROUTE = "8.7.56.1807"


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
    """Return the full-q HH window closeout formulas."""
    return {
        "retained_ff_surface": "A_FF,exact(q) = |q|",
        "retained_window_rule": "A_HH,exact(q) = F_exact(q) - |q| on 0 <= q <= q_HH,max",
        "retained_rank_one_rule": "A_FH,exact(q) = sqrt(|q| (F_exact(q) - |q|)), rho_exact(q) = 1",
        "primary_reopen_surface": "global all-q HH surface beyond the retained scalar-compatible window",
        "secondary_reopen_surface": "genuinely non-rank-one mixed surface beyond the retained rank-one window pack",
        "reserve_reopen_surface": "future external input or pack update guiding global HH or non-rank-one mixed generalization",
    }


# 関数: `.1803-.1806` を実行する。

def main() -> None:
    """Execute the full-q HH window closeout / reopen registry branch."""
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
        WINDOW_GATE,
        WINDOW_ROUTE,
        CLOSEOUT_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    window_summary = read_json(WINDOW_GATE)["summary"]
    window_route = read_json(WINDOW_ROUTE)["summary"]
    prior_closeout_summary = read_json(CLOSEOUT_GATE)["summary"]

    inventory_ready = all(
        item is not None
        for item in (
            hit(status_text, "8.7.56.1795"),
            hit(roadmap_text, "8.7.56.1795-.1798"),
            hit(current_problem_text, "branch-local completion theorem"),
            hit(current_status_text, "fixed-q exact promotion succeeded"),
            hit(unified_text, "`.1795-.1798`"),
            hit(long_text, "29. `8.7.56.1799-.1802`"),
            hit(part5_text, "`.1791-.1794`"),
        )
    )
    full_q_window_retained = bool(
        window_summary["full_q_exact_hh_surface_window_available"]
        and window_summary["window_covers_q_theory"]
        and window_summary["exact_scalar_promotion_selected"]
    )
    global_hh_surface_missing = bool(
        not window_summary["global_all_q_exact_hh_surface_available"]
    )
    non_rank_one_mixed_surface_missing = bool(
        not window_summary["non_rank_one_mixed_surface_available"]
    )
    same_level_window_retry_admissible = False
    primary_reopen_surface_fixed = True
    secondary_reopen_surface_fixed = True
    reserve_reopen_surface_fixed = True
    branch_honest = all(
        (
            inventory_ready,
            full_q_window_retained,
            global_hh_surface_missing,
            non_rank_one_mixed_surface_missing,
            not same_level_window_retry_admissible,
            primary_reopen_surface_fixed,
            secondary_reopen_surface_fixed,
            reserve_reopen_surface_fixed,
        )
    )

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "full-q HH window closeout inventory ready",
            truth(inventory_ready),
            "Closeout starts only after `.1799-.1802` has already derived the scalar-compatible HH window.",
        ),
        row(
            "full_q_window_retained",
            "pass" if full_q_window_retained else "reject",
            "full-q HH window retained",
            truth(full_q_window_retained),
            "The retained theorem must keep exact scalar promotion on the scalar-compatible window and at q_theory.",
        ),
        row(
            "scalar_compatible_window_upper_edge_over_m0",
            "watch",
            "scalar-compatible HH window upper edge q_HH,max/m0",
            float(window_summary["scalar_compatible_window_upper_edge_over_m0"]),
            "This is the retained upper edge of the exact HH window derived by the vacuum-saturated completion theorem.",
        ),
        row(
            "exact_alpha_mix_at_q_theory",
            "watch",
            "exact mixed alpha at q_theory",
            float(window_summary["exact_alpha_mix_at_q_theory"]),
            "The retained window theorem still reproduces the scalar strong candidate exactly at the matching scale.",
        ),
        row(
            "exact_hh_amplitude_at_q_theory",
            "watch",
            "windowed exact HH amplitude at q_theory",
            float(window_summary["exact_hh_amplitude_at_q_theory"]),
            "The new windowed HH diagonal is the canonical branch-local value carried forward into the closeout registry.",
        ),
        row(
            "global_hh_surface_missing",
            "pass" if global_hh_surface_missing else "reject",
            "global all-q HH surface missing",
            truth(global_hh_surface_missing),
            "The retained theorem still covers only the scalar-compatible window, not an unrestricted all-q HH surface.",
        ),
        row(
            "non_rank_one_mixed_surface_missing",
            "pass" if non_rank_one_mixed_surface_missing else "reject",
            "non-rank-one mixed surface missing",
            truth(non_rank_one_mixed_surface_missing),
            "No genuinely non-rank-one mixed extension has been derived under the retained window pack.",
        ),
        row(
            "same_level_window_retry_admissible",
            "reject",
            "same-level full-q HH window retry admissible",
            truth(same_level_window_retry_admissible),
            "The honest next step is a global HH or non-rank-one generalization, not another same-level window retry.",
        ),
        row(
            "primary_reopen_surface_fixed",
            "pass",
            "primary reopen surface fixed",
            truth(primary_reopen_surface_fixed),
            "Primary reopen surface = global all-q HH surface beyond the retained scalar-compatible window.",
        ),
        row(
            "secondary_reopen_surface_fixed",
            "pass",
            "secondary reopen surface fixed",
            truth(secondary_reopen_surface_fixed),
            "Secondary reopen surface = genuinely non-rank-one mixed surface beyond the retained rank-one window pack.",
        ),
        row(
            "reserve_reopen_surface_fixed",
            "pass",
            "reserve reopen surface fixed",
            truth(reserve_reopen_surface_fixed),
            "Reserve reopen surface = future external input or pack update that guides global HH or non-rank-one mixed generalization.",
        ),
        row(
            "branch_honest",
            "pass" if branch_honest else "reject",
            "full-q HH window closeout honest",
            truth(branch_honest),
            "The closeout is honest only if it retains the scalar-compatible window while refusing to over-claim a global HH or non-rank-one theorem.",
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
            "window_gate": display_path(WINDOW_GATE),
            "window_route": display_path(WINDOW_ROUTE),
            "prior_closeout_gate": display_path(CLOSEOUT_GATE),
        },
        "constants": {
            "scalar_compatible_window_upper_edge_over_m0": float(
                window_summary["scalar_compatible_window_upper_edge_over_m0"]
            ),
            "exact_ff_amplitude_at_q_theory": float(
                window_summary["exact_ff_amplitude_at_q_theory"]
            ),
            "exact_hh_amplitude_at_q_theory": float(
                window_summary["exact_hh_amplitude_at_q_theory"]
            ),
            "exact_fh_amplitude_at_q_theory": float(
                window_summary["exact_fh_amplitude_at_q_theory"]
            ),
            "exact_lambda_plus_at_q_theory": float(
                window_summary["exact_lambda_plus_at_q_theory"]
            ),
            "exact_alpha_mix_at_q_theory": float(
                window_summary["exact_alpha_mix_at_q_theory"]
            ),
            "selected_primary_reopen_surface": "global_all_q_hh_surface_beyond_retained_scalar_compatible_window",
            "selected_secondary_reopen_surface": "non_rank_one_mixed_surface_beyond_retained_rank_one_window_pack",
            "selected_reserve_reopen_surface": "future_external_input_or_pack_update_guiding_global_hh_or_non_rank_one_mixed_generalization",
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "full_q_window_retained": full_q_window_retained,
        "scalar_compatible_window_upper_edge_over_m0": float(
            window_summary["scalar_compatible_window_upper_edge_over_m0"]
        ),
        "exact_ff_amplitude_at_q_theory": float(
            window_summary["exact_ff_amplitude_at_q_theory"]
        ),
        "exact_hh_amplitude_at_q_theory": float(
            window_summary["exact_hh_amplitude_at_q_theory"]
        ),
        "exact_hh_alpha_at_q_theory": float(
            window_summary["exact_hh_alpha_at_q_theory"]
        ),
        "exact_fh_amplitude_at_q_theory": float(
            window_summary["exact_fh_amplitude_at_q_theory"]
        ),
        "exact_lambda_plus_at_q_theory": float(
            window_summary["exact_lambda_plus_at_q_theory"]
        ),
        "exact_alpha_mix_at_q_theory": float(
            window_summary["exact_alpha_mix_at_q_theory"]
        ),
        "exact_scalar_promotion_selected": bool(
            window_summary["exact_scalar_promotion_selected"]
        ),
        "global_hh_surface_missing": global_hh_surface_missing,
        "non_rank_one_mixed_surface_missing": non_rank_one_mixed_surface_missing,
        "same_level_window_retry_admissible": same_level_window_retry_admissible,
        "selected_primary_reopen_surface": "global_all_q_hh_surface_beyond_retained_scalar_compatible_window",
        "selected_secondary_reopen_surface": "non_rank_one_mixed_surface_beyond_retained_rank_one_window_pack",
        "selected_reserve_reopen_surface": "future_external_input_or_pack_update_guiding_global_hh_or_non_rank_one_mixed_generalization",
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
            "unified_roadmap_hit": hit(unified_text, "`.1795-.1798`"),
            "long_roadmap_hit": hit(long_text, "29. `8.7.56.1799-.1802`"),
            "part5_hit": hit(part5_text, "`.1791-.1794`"),
        },
        "carry_over": {
            "window_summary": window_summary,
            "window_route": window_route,
            "prior_closeout_summary": prior_closeout_summary,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload("8.7.56.1803", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence),
        ),
        "audit": write_artifact(
            "audit",
            payload("8.7.56.1804", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload("8.7.56.1805", f"{STEP_NAME} declaration gate", inputs, rows, summary, decision, evidence),
        ),
        "route_sync": write_artifact(
            "route_sync",
            payload("8.7.56.1806", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence),
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
