#!/usr/bin/env python3
"""Generate 8.7.56.1847-.1850 signed source-phase closeout artifacts.

`.1843-.1846` closed the retained sign sector exactly:

    F_exact(q) = sigma_F(q) |F_exact(q)|
               = sigma_F(q) F_src,abs(q)

with `sigma_F(q)` fixed by zero-count parity on the real overlap branch. This
branch freezes the honest post-closeout state:

1. exact alpha promotion stays retained,
2. exact signed form-factor promotion stays retained on the retained interval,
3. same-level signed-phase retries are blocked,
4. only substantive post-closeout pack updates or later external input remain
   as optional future surfaces.
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

SIGN_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1843_1846_signed_source_phase_reactivation_declaration_gate_metrics.json"
)
ABS_CLOSEOUT_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1839_1842_global_abs_source_loading_closeout_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1847-1850"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor signed source-phase "
    "closeout / wait restore"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "signed_source_phase_closeout_wait_restore",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_real_branch_sign_parity_theorem_derived_global_"
    "signed_form_factor_promotion_closeout_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_global_exact_alpha_and_signed_form_factor_"
    "promotion_closeout_wait_restore_completed"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_post_closeout_"
    "pack_update_or_external_input_reactivation"
)
NEXT_ROUTE = "8.7.56.1851"


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


# 関数: post-closeout reopen ordering の式を返す。

def build_formulae() -> dict[str, str]:
    """Return the retained post-closeout signed-source formulas."""
    return {
        "retained_amplitude_rule": "F_src,abs(q) = |F_exact(q)|",
        "retained_sign_rule": "sigma_F(q) = 0 at q_n and (-1)^{N_zero(q)} otherwise",
        "retained_signed_rule": "F_exact(q) = sigma_F(q) F_src,abs(q)",
        "primary_reopen_surface": "substantive pack update beyond the retained real-branch sign-parity theorem",
        "secondary_reopen_surface": "retained-interval extension or new signed observable rule beyond the current signed-phase pack",
        "reserve_reopen_surface": "future external input guiding a new signed surface or pack update",
    }


# 関数: `.1847-.1850` を実行する。

def main() -> None:
    """Execute the signed source-phase closeout / wait-restore branch."""
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
        SIGN_GATE,
        ABS_CLOSEOUT_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    sign_summary = read_json(SIGN_GATE)["summary"]
    abs_summary = read_json(ABS_CLOSEOUT_GATE)["summary"]

    inventory_ready = all(
        (
            bool(abs_summary["exact_alpha_promotion_retained"]),
            bool(sign_summary["exact_signed_source_phase_theorem_available"]),
            bool(sign_summary["exact_signed_form_factor_promotion_selected"]),
        )
    )
    exact_alpha_promotion_retained = bool(abs_summary["exact_alpha_promotion_retained"])
    exact_signed_source_phase_theorem_retained = bool(sign_summary["exact_signed_source_phase_theorem_available"])
    global_signed_form_factor_promotion_retained = bool(
        sign_summary["exact_signed_form_factor_promotion_selected"]
    )
    same_level_signed_phase_retry_admissible = False
    physical_reject_required = False

    formulas = build_formulae()

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "signed source-phase closeout inventory ready",
            truth(inventory_ready),
            "The branch starts only after alpha-space and sign-space have both been theorem-level closed on the retained interval.",
        ),
        row(
            "exact_alpha_promotion_retained",
            "pass" if exact_alpha_promotion_retained else "reject",
            "exact alpha promotion retained",
            truth(exact_alpha_promotion_retained),
            "The amplitude theorem remains the canonical alpha-side read.",
        ),
        row(
            "exact_signed_source_phase_theorem_retained",
            "pass" if exact_signed_source_phase_theorem_retained else "reject",
            "exact signed source-phase theorem retained",
            truth(exact_signed_source_phase_theorem_retained),
            "The real-branch parity theorem remains the canonical sign-side read.",
        ),
        row(
            "global_signed_form_factor_promotion_retained",
            "pass" if global_signed_form_factor_promotion_retained else "reject",
            "global signed form-factor promotion retained on 0<=q<=1",
            truth(global_signed_form_factor_promotion_retained),
            "Combining the amplitude and sign theorems now reproduces F_exact itself on the retained audit interval.",
        ),
        row(
            "same_level_signed_phase_retry_admissible",
            "reject",
            "same-level signed-phase retry admissible",
            truth(same_level_signed_phase_retry_admissible),
            "The signed family is closed on the retained interval, so same-level retry is no longer honest.",
        ),
        row(
            "physical_reject_required",
            "reject",
            "physical reject required",
            truth(physical_reject_required),
            "The family closes positively; no physical reject flag is needed.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "exact_alpha_promotion_retained": exact_alpha_promotion_retained,
        "exact_signed_source_phase_theorem_retained": exact_signed_source_phase_theorem_retained,
        "global_signed_form_factor_promotion_retained": global_signed_form_factor_promotion_retained,
        "same_level_signed_phase_retry_admissible": same_level_signed_phase_retry_admissible,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": "vector_qball_form_factor_signed_source_phase_closeout_wait_restore_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": formulas,
        "hits": {
            "status_branch_hit": hit(status_text, "8.7.56.1847"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1847-.1850"),
            "current_problem_hit": hit(current_problem_text, "signed source-phase theorem"),
            "current_status_hit": hit(current_status_text, "signed source-phase theorem"),
            "unified_roadmap_hit": hit(unified_text, "87. `.1839-.1842`"),
            "long_roadmap_hit": hit(long_text, "signed source-phase theorem"),
            "part5_hit": hit(part5_text, "signed source-phase theorem"),
        },
    }

    declaration_payload = payload(
        "8.7.56.1849",
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
                "sign_gate": display_path(SIGN_GATE),
                "abs_closeout_gate": display_path(ABS_CLOSEOUT_GATE),
            },
            "constants": {
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
            },
        },
        rows,
        summary,
        decision,
        evidence,
    )

    route_payload = payload(
        "8.7.56.1850",
        STEP_NAME + " route sync",
        declaration_payload["inputs"],
        [
            row(
                "post_closeout_wait_required",
                "pass",
                "post-closeout wait required",
                1.0,
                "The family is closed on the retained interval, so future work must come from a substantive pack update or later external input.",
            ),
            row(
                "same_level_signed_phase_retry_admissible",
                "reject",
                "same-level signed-phase retry admissible",
                truth(False),
                "Same-level retry remains blocked after closeout.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "physical_reject_required": physical_reject_required,
        },
        {
            "overall_status": "vector_qball_form_factor_signed_source_phase_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"formulas": formulas},
    )

    write_artifact("declaration_gate", declaration_payload)
    write_artifact("route_sync", route_payload)

    print("[ok] 8.7.56.1847-.1850 signed source-phase closeout artifacts generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
