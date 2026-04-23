#!/usr/bin/env python3
"""Generate 8.7.56.1831-.1834 source-direction exact-loading closeout artifacts.

`.1827-.1830` replaced the old fixed-q proxy loading with a theorem-level
windowed loading surface

    kappa_exact(q) = sqrt(F_exact(q)-|q|) / (sqrt(F_exact(q)) + sqrt(|q|))

on the retained scalar-compatible window `0 <= q <= q_HH,max`.  That closes the
source-loading theorem on the exact window and promotes the source-direction
bilinear family from Gate B proxy to Gate A exact *within the window*.

This branch freezes that honest read and sharpens the remaining gap to the
global all-q loading surface side, so same-level ad-hoc loading scans are no
longer admissible.
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

SOURCE_LOADING_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1827_1830_windowed_exact_source_loading_reactivation_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1831-1834"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor source-direction exact "
    "loading closeout / reopen registry"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "source_direction_exact_loading_closeout",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_windowed_exact_source_loading_theorem_derived_"
    "source_direction_exact_promotion_closeout_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_source_direction_windowed_exact_loading_closeout_"
    "global_loading_reopen_registry_completed"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_global_source_"
    "loading_surface_or_substantive_pack_update_reactivation"
)
NEXT_ROUTE = "8.7.56.1835"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_global_source_loading_closeout_"
    "reopen_registry"
)
FOLLOWUP_ROUTE = "8.7.56.1839"


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


# 関数: closeout registry の主要式を返す。

def build_formulae() -> dict[str, str]:
    """Return the exact-loading closeout formulas."""
    return {
        "retained_rule": "F_src,k(q) = s_k^T A_mix(q) s_k,  s_k = (1, k)^T",
        "windowed_exact_loading_rule": "kappa_exact(q) = sqrt(F_exact(q)-|q|) / (sqrt(F_exact(q)) + sqrt(|q|))",
        "windowed_exact_promotion": "F_src,kappa_exact(q) = F_exact(q) on 0 <= q <= q_HH,max",
        "primary_reopen_surface": "global_all_q_source_loading_surface_beyond_scalar_compatible_window",
        "secondary_reopen_surface": "substantive_pack_update_changing_ff_carrier_hh_surface_or_source_direction_rule_beyond_windowed_loading_pack",
        "reserve_reopen_surface": "future_external_input_guiding_global_loading_surface_or_rule_update",
    }


# 関数: `.1831-.1834` を実行する。

def main() -> None:
    """Execute the source-direction exact-loading closeout / reopen registry branch."""
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
        SOURCE_LOADING_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    source_loading_payload = read_json(SOURCE_LOADING_GATE)
    source_loading_summary = source_loading_payload["summary"]
    source_loading_constants = source_loading_payload["inputs"]["constants"]

    inventory_ready = all(
        (
            bool(source_loading_summary["retained_source_direction_rule"]),
            bool(source_loading_summary["windowed_exact_source_loading_theorem_derived"]),
            bool(source_loading_summary["exact_source_loading_theorem_available"]),
            bool(source_loading_summary["q_dependent_loading_surface_available"]),
            bool(source_loading_summary["gate_a_windowed_exact_promote_selected"]),
        )
    )
    source_direction_rule_retained = bool(source_loading_summary["retained_source_direction_rule"])
    windowed_exact_source_loading_theorem_retained = bool(
        source_loading_summary["windowed_exact_source_loading_theorem_derived"]
    )
    q_dependent_loading_surface_retained = bool(
        source_loading_summary["q_dependent_loading_surface_available"]
    )
    gate_a_windowed_exact_promotion_retained = bool(
        source_loading_summary["gate_a_windowed_exact_promote_selected"]
    )
    global_all_q_loading_surface_missing = bool(
        not source_loading_summary["global_all_q_loading_surface_available"]
    )
    same_level_branch_local_loading_retry_admissible = False
    branch_honest = all(
        (
            inventory_ready,
            source_direction_rule_retained,
            windowed_exact_source_loading_theorem_retained,
            q_dependent_loading_surface_retained,
            gate_a_windowed_exact_promotion_retained,
            global_all_q_loading_surface_missing,
            not same_level_branch_local_loading_retry_admissible,
        )
    )

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "source-direction exact-loading closeout inventory ready",
            truth(inventory_ready),
            "The closeout starts only after `.1827-.1830` has already replaced proxy loading by the theorem-level windowed surface.",
        ),
        row(
            "source_direction_rule_retained",
            "pass" if source_direction_rule_retained else "reject",
            "source-direction bilinear rule retained",
            truth(source_direction_rule_retained),
            "The new observable rule itself survives and remains the canonical mixed family for future loading-surface generalization attempts.",
        ),
        row(
            "windowed_exact_source_loading_theorem_retained",
            "pass" if windowed_exact_source_loading_theorem_retained else "reject",
            "windowed exact source-loading theorem retained",
            truth(windowed_exact_source_loading_theorem_retained),
            "The theorem-level loading coefficient is now fixed on the scalar-compatible window and therefore replaces the old fixed-q proxy root.",
        ),
        row(
            "q_dependent_loading_surface_retained",
            "pass" if q_dependent_loading_surface_retained else "reject",
            "q-dependent loading surface retained",
            truth(q_dependent_loading_surface_retained),
            "The retained source-loading surface is now a genuine kappa(q), not a branch-local ad-hoc number.",
        ),
        row(
            "kappa_exact_at_q_theory",
            "watch",
            "retained theorem-level loading at q_theory",
            float(source_loading_summary["kappa_exact_at_q_theory"]),
            "This is the current theorem-level loading coefficient at the matching point.",
        ),
        row(
            "kappa_target_at_q_theory",
            "watch",
            "retained target loading at q_theory",
            float(source_loading_summary["kappa_target_at_q_theory"]),
            "The physical target still lives inside the same windowed loading family.",
        ),
        row(
            "exact_source_direction_alpha_at_q_theory",
            "watch",
            "retained exact source-direction alpha at q_theory",
            float(source_loading_summary["exact_source_direction_alpha_at_q_theory"]),
            "The theorem-level source-loading now reproduces the retained scalar strong candidate exactly at q_theory.",
        ),
        row(
            "gate_a_windowed_exact_promotion_retained",
            "pass" if gate_a_windowed_exact_promotion_retained else "reject",
            "Gate A windowed exact promotion retained",
            truth(gate_a_windowed_exact_promotion_retained),
            "Within the scalar-compatible window the source-direction family is no longer partial or proxy-only.",
        ),
        row(
            "global_all_q_loading_surface_missing",
            "pass" if global_all_q_loading_surface_missing else "reject",
            "global all-q loading surface missing",
            truth(global_all_q_loading_surface_missing),
            "The unresolved gap is now the extension of the theorem beyond the scalar-compatible window, not the loading theorem itself.",
        ),
        row(
            "same_level_branch_local_loading_retry_admissible",
            "reject",
            "same-level branch-local loading retry admissible",
            truth(same_level_branch_local_loading_retry_admissible),
            "Once kappa(q) is derived analytically on the window, ad-hoc branch-local loading scans are no longer honest.",
        ),
        row(
            "primary_reopen_surface_fixed",
            "pass",
            "primary reopen surface fixed",
            1.0,
            "Primary reopen surface = global_all_q_source_loading_surface_beyond_scalar_compatible_window.",
        ),
        row(
            "secondary_reopen_surface_fixed",
            "pass",
            "secondary reopen surface fixed",
            1.0,
            "Secondary reopen surface = substantive pack update changing FF carrier, HH surface, or source-direction rule beyond the windowed loading pack.",
        ),
        row(
            "reserve_reopen_surface_fixed",
            "pass",
            "reserve reopen surface fixed",
            1.0,
            "Reserve reopen surface = future external input guiding global loading-surface or rule update.",
        ),
        row(
            "branch_honest",
            "pass" if branch_honest else "reject",
            "source-direction exact-loading closeout honest",
            truth(branch_honest),
            "The closeout is honest only if it retains the windowed exact theorem and localizes the remaining gap to the global all-q loading surface.",
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
            "source_loading_gate": display_path(SOURCE_LOADING_GATE),
        },
        "constants": {
            "q_theory_over_m0": float(source_loading_constants["q_theory_over_m0"]),
            "q_hh_max_over_m0": float(source_loading_summary["q_hh_max_over_m0"]),
            "selected_primary_reopen_surface": "global_all_q_source_loading_surface_beyond_scalar_compatible_window",
            "selected_secondary_reopen_surface": "substantive_pack_update_changing_ff_carrier_hh_surface_or_source_direction_rule_beyond_windowed_loading_pack",
            "selected_reserve_reopen_surface": "future_external_input_guiding_global_loading_surface_or_rule_update",
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "source_direction_rule_retained": source_direction_rule_retained,
        "windowed_exact_source_loading_theorem_retained": windowed_exact_source_loading_theorem_retained,
        "q_dependent_loading_surface_retained": q_dependent_loading_surface_retained,
        "kappa_exact_at_q_theory": float(source_loading_summary["kappa_exact_at_q_theory"]),
        "kappa_target_at_q_theory": float(source_loading_summary["kappa_target_at_q_theory"]),
        "exact_source_direction_alpha_at_q_theory": float(source_loading_summary["exact_source_direction_alpha_at_q_theory"]),
        "gate_a_windowed_exact_promotion_retained": gate_a_windowed_exact_promotion_retained,
        "global_all_q_loading_surface_missing": global_all_q_loading_surface_missing,
        "same_level_branch_local_loading_retry_admissible": same_level_branch_local_loading_retry_admissible,
        "selected_primary_reopen_surface": "global_all_q_source_loading_surface_beyond_scalar_compatible_window",
        "selected_secondary_reopen_surface": "substantive_pack_update_changing_ff_carrier_hh_surface_or_source_direction_rule_beyond_windowed_loading_pack",
        "selected_reserve_reopen_surface": "future_external_input_guiding_global_loading_surface_or_rule_update",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
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
            "status_branch_hit": hit(status_text, "8.7.56.1831"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1831-.1834"),
            "current_problem_hit": hit(current_problem_text, "windowed exact source-loading theorem"),
            "current_status_hit": hit(current_status_text, "source-direction exact promotion closeout / global loading registry"),
            "unified_roadmap_hit": hit(unified_text, "85. `.1831-.1834`"),
            "long_roadmap_hit": hit(long_text, "37. `8.7.56.1831-.1834`"),
            "part5_hit": hit(part5_text, "next official branch は `.1831-.1834`"),
        },
        "carry_over": {
            "source_loading_summary": source_loading_summary,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload("8.7.56.1831", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence),
        ),
        "audit": write_artifact(
            "audit",
            payload("8.7.56.1832", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload("8.7.56.1833", f"{STEP_NAME} declaration gate", inputs, rows, summary, decision, evidence),
        ),
        "route_sync": write_artifact(
            "route_sync",
            payload("8.7.56.1834", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence),
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
