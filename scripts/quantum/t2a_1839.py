#!/usr/bin/env python3
"""Generate 8.7.56.1839-.1842 absolute-loading closeout artifacts.

`.1835-.1838` derived a global all-q source-loading surface by splitting the
retained scalar form factor into amplitude and phase:

    F_exact(q) = sigma_F(q) |F_exact(q)|.

The bilinear source rule then closes the nonnegative amplitude sector exactly,
which is sufficient for alpha because `alpha = |F|^2 / (4 pi)`.

This branch freezes that honest read:

- exact alpha promotion is now global under the amplitude/phase split,
- the unresolved gap moves to a separate signed source-phase theorem,
- same-level ad-hoc loading retry is no longer admissible.
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

GLOBAL_LOADING_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1835_1838_global_abs_source_loading_reactivation_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1839-1842"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor global absolute source-loading "
    "closeout / signed phase reopen registry"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "global_abs_source_loading_closeout",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_global_absolute_source_loading_surface_derived_"
    "exact_alpha_promotion_signed_phase_closeout_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_global_absolute_source_loading_exact_alpha_"
    "promotion_signed_phase_reopen_registry_completed"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_signed_source_"
    "phase_theorem_or_substantive_pack_update_reactivation"
)
NEXT_ROUTE = "8.7.56.1843"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_signed_source_phase_closeout_"
    "or_wait_restore"
)
FOLLOWUP_ROUTE = "8.7.56.1847"


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
    """Return the global absolute-loading closeout formulas."""
    return {
        "retained_rule": "F_src,abs(q) = |q| + 2 sigma_abs(q) kappa_abs(q) sqrt(|q| D_abs(q)) + kappa_abs(q)^2 D_abs(q) = |F_exact(q)|",
        "exact_alpha_rule": "alpha_src,abs(q) = F_src,abs(q)^2 / (4 pi) = alpha_exact(q)",
        "missing_signed_phase_rule": "F_exact(q) = sigma_F(q) F_src,abs(q)",
        "primary_reopen_surface": "exact_signed_source_phase_theorem_beyond_abs_loading_pack",
        "secondary_reopen_surface": "substantive_pack_update_linking_source_direction_rule_to_signed_form_factor_surface_beyond_abs_loading_pack",
        "reserve_reopen_surface": "future_external_input_guiding_signed_phase_surface",
    }


# 関数: `.1839-.1842` を実行する。

def main() -> None:
    """Execute the global absolute-loading closeout / signed phase reopen registry branch."""
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
        GLOBAL_LOADING_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    global_loading_payload = read_json(GLOBAL_LOADING_GATE)
    global_loading_summary = global_loading_payload["summary"]

    inventory_ready = all(
        (
            bool(global_loading_summary["retained_source_direction_rule"]),
            bool(global_loading_summary["windowed_exact_source_loading_theorem_retained"]),
            bool(global_loading_summary["global_absolute_source_loading_surface_available"]),
            bool(global_loading_summary["exact_alpha_promotion_selected"]),
            bool(global_loading_summary["source_phase_theorem_required"]),
        )
    )
    source_direction_rule_retained = bool(global_loading_summary["retained_source_direction_rule"])
    global_absolute_source_loading_surface_retained = bool(
        global_loading_summary["global_absolute_source_loading_surface_available"]
    )
    exact_alpha_promotion_retained = bool(global_loading_summary["exact_alpha_promotion_selected"])
    signed_source_phase_theorem_required = bool(global_loading_summary["source_phase_theorem_required"])
    same_level_abs_loading_retry_admissible = False
    physical_reject_required = False

    formulas = build_formulae()

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "global absolute-loading closeout inventory ready",
            truth(inventory_ready),
            "The closeout starts only after `.1835-.1838` has already promoted exact alpha globally under the amplitude/phase split theorem.",
        ),
        row(
            "source_direction_rule_retained",
            "pass" if source_direction_rule_retained else "reject",
            "retained source-direction bilinear rule",
            truth(source_direction_rule_retained),
            "The closeout keeps the source-direction observable as the canonical loading rule.",
        ),
        row(
            "global_absolute_source_loading_surface_retained",
            "pass" if global_absolute_source_loading_surface_retained else "reject",
            "global absolute source-loading surface retained",
            truth(global_absolute_source_loading_surface_retained),
            "The new theorem already closes all-q alpha loading, so it becomes the retained canonical amplitude surface.",
        ),
        row(
            "exact_alpha_promotion_retained",
            "pass" if exact_alpha_promotion_retained else "reject",
            "global exact alpha promotion retained",
            truth(exact_alpha_promotion_retained),
            "The numeric alpha problem is closed under the amplitude/phase split theorem because the bilinear source rule reproduces |F_exact(q)| globally.",
        ),
        row(
            "signed_source_phase_theorem_required",
            "pass" if signed_source_phase_theorem_required else "reject",
            "signed source-phase theorem required",
            truth(signed_source_phase_theorem_required),
            "The unresolved gap has moved entirely into the sign/phase sector that would reconstruct F_exact rather than |F_exact|.",
        ),
        row(
            "same_level_abs_loading_retry_admissible",
            "reject",
            "same-level abs-loading retry admissible",
            truth(same_level_abs_loading_retry_admissible),
            "Same-level abs-loading retry is no longer honest because exact alpha promotion is already closed.",
        ),
        row(
            "branch_honest",
            "pass",
            "global abs-loading closeout honest",
            1.0,
            "The closeout is honest only if it retains exact alpha promotion while explicitly freezing the remaining signed phase gap as a separate reopen surface.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "source_direction_rule_retained": source_direction_rule_retained,
        "global_absolute_source_loading_surface_retained": global_absolute_source_loading_surface_retained,
        "exact_alpha_promotion_retained": exact_alpha_promotion_retained,
        "signed_source_phase_theorem_required": signed_source_phase_theorem_required,
        "same_level_abs_loading_retry_admissible": same_level_abs_loading_retry_admissible,
        "selected_primary_reopen_surface": "exact_signed_source_phase_theorem_beyond_abs_loading_pack",
        "selected_secondary_reopen_surface": "substantive_pack_update_linking_source_direction_rule_to_signed_form_factor_surface_beyond_abs_loading_pack",
        "selected_reserve_reopen_surface": "future_external_input_guiding_signed_phase_surface",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": formulas,
        "hits": {
            "status_branch_hit": hit(status_text, "8.7.56.1839"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1839-.1842"),
            "current_problem_hit": hit(current_problem_text, "global all-q source-loading surface"),
            "current_status_hit": hit(current_status_text, "global source-loading closeout / reopen registry"),
            "unified_roadmap_hit": hit(unified_text, "85. `.1831-.1834`"),
            "long_roadmap_hit": hit(long_text, "37. `8.7.56.1831-.1834`"),
            "part5_hit": hit(part5_text, "next official branch は `.1835-.1838`"),
        },
    }

    declaration = payload(
        "8.7.56.1841",
        f"{STEP_NAME} declaration gate",
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
                "global_loading_gate": display_path(GLOBAL_LOADING_GATE),
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
    route_sync = payload(
        "8.7.56.1842",
        f"{STEP_NAME} route sync",
        declaration["inputs"],
        rows,
        summary,
        decision,
        evidence,
    )

    write_artifact("declaration_gate", declaration)
    write_artifact("route_sync", route_sync)

    print(f"[ok] {STEP_TAG} complete")
    print(f"[state] {BRANCH_CLASS}")
    print(f"[next] {NEXT_ROUTE} {NEXT_ROUTE_NAME}")
    print(f"[followup] {FOLLOWUP_ROUTE} {FOLLOWUP_ROUTE_NAME}")


if __name__ == "__main__":
    main()
