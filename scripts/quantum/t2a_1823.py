#!/usr/bin/env python3
"""Generate 8.7.56.1823-.1826 source-direction closeout registry artifacts.

`.1819-.1822` opened a genuinely new mixed observable rule by replacing the old
largest-eigenvalue read with a physical source-direction bilinear contraction.
That branch achieved only Gate B partial promotion because the scalar-compatible
loading family is still proxy-local:

1. `kappa` is not derived canonically from the action, and
2. no q-dependent loading surface `kappa(q)` exists yet.

This closeout branch freezes that honest read and narrows the reopen ordering to
the exact loading theorem side, so same-level source-loading scans are not
reopened.
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

SOURCE_DIRECTION_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1819_1822_source_direction_mixed_reactivation_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1823-1826"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor source-direction bilinear "
    "closeout / exact loading registry"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "source_direction_closeout_registry",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_source_direction_bilinear_rule_scalar_compatible_"
    "proxy_loading_exact_source_loading_reopen_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_source_direction_bilinear_gate_b_partial_exact_"
    "loading_reopen_registry_completed"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_exact_source_"
    "loading_theorem_or_q_dependent_loading_surface_reactivation"
)
NEXT_ROUTE = "8.7.56.1827"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_source_direction_loading_"
    "closeout_reopen_registry"
)
FOLLOWUP_ROUTE = "8.7.56.1831"


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
    """Return the source-direction closeout formulas."""
    return {
        "retained_rule": "F_src,k(q) = s_k^T A_mix(q) s_k,  s_k = (1, k)^T",
        "fixed_q_proxy_rule": "F_src,k(q_theory) = A_FF + 2 k A_FH + k^2 A_HH,proxy",
        "primary_reopen_surface": "exact source-loading theorem under the retained source-direction bilinear pack",
        "secondary_reopen_surface": "q-dependent loading surface kappa(q) beyond the branch-local proxy pack",
        "reserve_reopen_surface": "future external input or pack update guiding exact or q-dependent loading closure",
    }


# 関数: `.1823-.1826` を実行する。

def main() -> None:
    """Execute the source-direction closeout / exact loading registry branch."""
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
        SOURCE_DIRECTION_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    source_direction_payload = read_json(SOURCE_DIRECTION_GATE)
    source_direction_summary = source_direction_payload["summary"]
    source_direction_constants = source_direction_payload["inputs"]["constants"]

    inventory_ready = all(
        (
            hit(status_text, "8.7.56.1823"),
            hit(roadmap_text, "8.7.56.1823-.1826"),
            hit(current_problem_text, "exact source-loading theorem / q-dependent loading surface"),
            hit(current_status_text, "source-direction bilinear closeout / exact loading registry"),
            hit(unified_text, "83. `.1823-.1826`"),
            hit(long_text, "35. `8.7.56.1823-.1826`"),
            hit(part5_text, "next official branch は `.1823-.1826`"),
            bool(source_direction_summary["gate_b_partial_source_direction_selected"]),
        )
    )
    source_direction_rule_retained = bool(
        source_direction_summary["source_direction_bilinear_rule_adopted"]
        and source_direction_summary["old_eigenvalue_rule_broken"]
    )
    gate_b_partial_retained = bool(source_direction_summary["gate_b_partial_source_direction_selected"])
    exact_source_loading_theorem_missing = bool(
        not source_direction_summary["exact_source_loading_theorem_available"]
    )
    q_dependent_loading_surface_missing = bool(
        not source_direction_summary["q_dependent_loading_surface_available"]
    )
    same_level_source_loading_scan_admissible = False
    primary_reopen_surface_fixed = True
    secondary_reopen_surface_fixed = True
    reserve_reopen_surface_fixed = True
    branch_honest = all(
        (
            inventory_ready,
            source_direction_rule_retained,
            gate_b_partial_retained,
            exact_source_loading_theorem_missing,
            q_dependent_loading_surface_missing,
            not same_level_source_loading_scan_admissible,
            primary_reopen_surface_fixed,
            secondary_reopen_surface_fixed,
            reserve_reopen_surface_fixed,
        )
    )

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "source-direction closeout inventory ready",
            truth(inventory_ready),
            "The closeout starts only after `.1819-.1822` has already fixed Gate B partial and sharpened the missing theorem to the loading side.",
        ),
        row(
            "source_direction_rule_retained",
            "pass" if source_direction_rule_retained else "reject",
            "source-direction bilinear rule retained",
            truth(source_direction_rule_retained),
            "The new physical observable rule itself survives and becomes the retained pack for future theorem-level reopen attempts.",
        ),
        row(
            "gate_b_partial_retained",
            "pass" if gate_b_partial_retained else "reject",
            "Gate B partial retained",
            truth(gate_b_partial_retained),
            "The honest current read is partial scalar-leaning promotion, not exact promotion or rejection.",
        ),
        row(
            "proxy_kappa_exact_at_q_theory",
            "watch",
            "retained proxy exact loading at q_theory",
            float(source_direction_summary["proxy_kappa_exact_at_q_theory"]),
            "This is the fixed-q constructive loading that reproduces the retained scalar strong candidate inside the proxy pack.",
        ),
        row(
            "proxy_kappa_target_at_q_theory",
            "watch",
            "retained proxy target loading at q_theory",
            float(source_direction_summary["proxy_kappa_target_at_q_theory"]),
            "The physical target sits in the same moderate loading family and therefore remains a live theorem-side goal.",
        ),
        row(
            "proxy_source_direction_alpha_exact_at_q_theory",
            "watch",
            "retained proxy alpha at exact loading",
            float(source_direction_summary["proxy_source_direction_alpha_exact_at_q_theory"]),
            "The proxy family reproduces the retained scalar strong candidate exactly at q_theory.",
        ),
        row(
            "proxy_source_direction_alpha_target_at_q_theory",
            "watch",
            "retained proxy alpha at target loading",
            float(source_direction_summary["proxy_source_direction_alpha_target_at_q_theory"]),
            "The same retained family reaches the physical target after only a small loading shift.",
        ),
        row(
            "exact_source_loading_theorem_missing",
            "pass" if exact_source_loading_theorem_missing else "reject",
            "exact source-loading theorem missing",
            truth(exact_source_loading_theorem_missing),
            "The remaining missing bridge is no longer amplitude size but a theorem that fixes the loading coefficient canonically.",
        ),
        row(
            "q_dependent_loading_surface_missing",
            "pass" if q_dependent_loading_surface_missing else "reject",
            "q-dependent loading surface missing",
            truth(q_dependent_loading_surface_missing),
            "A fixed-q proxy loading does not yet define the observable globally in q-space.",
        ),
        row(
            "same_level_source_loading_scan_admissible",
            "reject",
            "same-level source-loading scan admissible",
            truth(same_level_source_loading_scan_admissible),
            "The next honest move is theorem derivation, not more scans over kappa inside the same proxy family.",
        ),
        row(
            "primary_reopen_surface_fixed",
            "pass",
            "primary reopen surface fixed",
            truth(primary_reopen_surface_fixed),
            "Primary reopen surface = exact source-loading theorem under the retained source-direction bilinear pack.",
        ),
        row(
            "secondary_reopen_surface_fixed",
            "pass",
            "secondary reopen surface fixed",
            truth(secondary_reopen_surface_fixed),
            "Secondary reopen surface = q-dependent loading surface beyond the branch-local proxy pack.",
        ),
        row(
            "reserve_reopen_surface_fixed",
            "pass",
            "reserve reopen surface fixed",
            truth(reserve_reopen_surface_fixed),
            "Reserve reopen surface = future external input or pack update guiding exact or q-dependent loading closure.",
        ),
        row(
            "branch_honest",
            "pass" if branch_honest else "reject",
            "source-direction closeout honest",
            truth(branch_honest),
            "The closeout is honest only if it retains the new rule, freezes Gate B partial, and blocks same-level proxy loading scans.",
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
            "source_direction_gate": display_path(SOURCE_DIRECTION_GATE),
        },
        "constants": {
            "q_theory_over_m0": float(source_direction_constants["q_theory_over_m0"]),
            "proxy_kappa_exact_at_q_theory": float(source_direction_summary["proxy_kappa_exact_at_q_theory"]),
            "proxy_kappa_target_at_q_theory": float(source_direction_summary["proxy_kappa_target_at_q_theory"]),
            "selected_primary_reopen_surface": "exact_source_loading_theorem_under_retained_source_direction_bilinear_pack",
            "selected_secondary_reopen_surface": "q_dependent_loading_surface_beyond_branch_local_proxy_pack",
            "selected_reserve_reopen_surface": "future_external_input_or_pack_update_guiding_exact_or_q_dependent_loading_surface",
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
        "gate_b_partial_retained": gate_b_partial_retained,
        "proxy_kappa_exact_at_q_theory": float(source_direction_summary["proxy_kappa_exact_at_q_theory"]),
        "proxy_kappa_target_at_q_theory": float(source_direction_summary["proxy_kappa_target_at_q_theory"]),
        "proxy_source_direction_alpha_exact_at_q_theory": float(source_direction_summary["proxy_source_direction_alpha_exact_at_q_theory"]),
        "proxy_source_direction_alpha_target_at_q_theory": float(source_direction_summary["proxy_source_direction_alpha_target_at_q_theory"]),
        "exact_source_loading_theorem_missing": exact_source_loading_theorem_missing,
        "q_dependent_loading_surface_missing": q_dependent_loading_surface_missing,
        "same_level_source_loading_scan_admissible": same_level_source_loading_scan_admissible,
        "selected_primary_reopen_surface": "exact_source_loading_theorem_under_retained_source_direction_bilinear_pack",
        "selected_secondary_reopen_surface": "q_dependent_loading_surface_beyond_branch_local_proxy_pack",
        "selected_reserve_reopen_surface": "future_external_input_or_pack_update_guiding_exact_or_q_dependent_loading_surface",
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
            "status_branch_hit": hit(status_text, "8.7.56.1823"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1823-.1826"),
            "current_problem_hit": hit(current_problem_text, "exact source-loading theorem / q-dependent loading surface"),
            "current_status_hit": hit(current_status_text, "source-direction bilinear closeout / exact loading registry"),
            "unified_roadmap_hit": hit(unified_text, "83. `.1823-.1826`"),
            "long_roadmap_hit": hit(long_text, "35. `8.7.56.1823-.1826`"),
            "part5_hit": hit(part5_text, "next official branch は `.1823-.1826`"),
        },
        "carry_over": {
            "source_direction_summary": source_direction_summary,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload("8.7.56.1823", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence),
        ),
        "audit": write_artifact(
            "audit",
            payload("8.7.56.1824", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload("8.7.56.1825", f"{STEP_NAME} declaration gate", inputs, rows, summary, decision, evidence),
        ),
        "route_sync": write_artifact(
            "route_sync",
            payload("8.7.56.1826", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence),
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
