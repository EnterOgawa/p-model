#!/usr/bin/env python3
"""Generate 8.7.56.1787-.1790 mixed-proxy closeout / reopen registry artifacts.

`.1783-.1786` closed the internal coherence gap exactly by deriving the
rank-one identity

    A_FH^2 = A_FF A_HH,  rho_exact = 1,

inside a single-mediator mixed-source extension. That branch did *not* derive
the HH diagonal surface itself, so the canonical promotion gap is now sharply
localized to the exact HH surface alone.

This branch freezes that conclusion machine-readably:

1. Gate B partial proxy promotion remains the honest official read,
2. the exact HH diagonal surface becomes the primary reopen surface,
3. non-rank-one mixed extensions become the secondary reopen surface,
4. future external input / pack update remains reserve.
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

COHERENCE_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1783_1786_int_coh_hh_reactivation_declaration_gate_metrics.json"
DECISION_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1779_1782_mixed_proxy_decision_gate_declaration_gate_metrics.json"
PROXY_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1775_1778_mixed_proxy_recompute_declaration_gate_metrics.json"
FIELD_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1739_1742_field_strength_recompute_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.1787-1790"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor mixed proxy closeout / "
    "reopen registry"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "mixed_proxy_closeout_registry",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_exact_internal_rank_one_coherence_derived_"
    "exact_hh_surface_missing_mixed_proxy_closeout_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_rank_one_internal_coherence_closeout_exact_hh_"
    "surface_reopen_registry_completed"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_exact_hh_"
    "surface_or_non_rank_one_mixed_surface_reactivation"
)
NEXT_ROUTE = "8.7.56.1791"


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
    """Return the closeout / reopen registry formulas."""
    return {
        "retained_rank_one_rule": "A_FH^2 = A_FF A_HH,  rho_exact = 1",
        "retained_rank_one_eigenchannel": "lambda_+(q) = A_FF(q) + A_HH(q)",
        "primary_reopen_surface": "exact HH diagonal surface under the retained rank-one mixed pack",
        "secondary_reopen_surface": "non-rank-one mixed surface beyond the single internal mediator pack",
        "reserve_reopen_surface": "future external input or pack update guiding HH or non-rank-one mixed surfaces",
    }


# 関数: `.1787-.1790` を実行する。

def main() -> None:
    """Execute the mixed proxy closeout / reopen registry branch."""
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
        COHERENCE_GATE,
        DECISION_GATE,
        PROXY_GATE,
        FIELD_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    coherence_summary = read_json(COHERENCE_GATE)["summary"]
    decision_summary = read_json(DECISION_GATE)["summary"]
    proxy_summary = read_json(PROXY_GATE)["summary"]
    field_summary = read_json(FIELD_GATE)["summary"]

    inventory_ready = all(
        item is not None
        for item in (
            hit(status_text, "8.7.56.1787"),
            hit(roadmap_text, "次の公式 branch は `.1787-.1790`"),
            hit(current_problem_text, "exact HH diagonal surface"),
            hit(current_status_text, "mixed proxy closeout / reopen registry"),
            hit(unified_text, "`.1783-.1786` は **exact internal coherence or HH surface reactivation**"),
            hit(long_text, "26. `8.7.56.1787-.1790`"),
            hit(part5_text, "`.1783-.1786` の **exact internal coherence or HH surface reactivation**"),
        )
    )
    rank_one_internal_coherence_closed = bool(
        coherence_summary["exact_internal_rank_one_coherence_derived"]
        and coherence_summary["exact_rho_fixed_to_one"]
        and coherence_summary["exact_internal_coherence_gap_closed"]
    )
    gate_b_partial_proxy_promotion_retained = bool(
        decision_summary["gate_b_partial_proxy_promotion_selected"]
        and proxy_summary["partial_proxy_promotion_selected"]
    )
    exact_hh_diagonal_surface_missing = bool(
        not coherence_summary["exact_hh_surface_available"]
        and coherence_summary["remaining_missing_surface_is_exact_hh_diagonal"]
    )
    same_level_rank_one_retry_admissible = False
    primary_reopen_surface_fixed = True
    secondary_reopen_surface_fixed = True
    reserve_reopen_surface_fixed = True
    mixed_proxy_closeout_honest = all(
        (
            inventory_ready,
            rank_one_internal_coherence_closed,
            gate_b_partial_proxy_promotion_retained,
            exact_hh_diagonal_surface_missing,
            not same_level_rank_one_retry_admissible,
            primary_reopen_surface_fixed,
            secondary_reopen_surface_fixed,
            reserve_reopen_surface_fixed,
        )
    )

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "mixed proxy closeout inventory ready",
            truth(inventory_ready),
            "Closeout starts only after the live docs already point to `.1787-.1790` as the next official branch.",
        ),
        row(
            "rank_one_internal_coherence_closed",
            "pass" if rank_one_internal_coherence_closed else "reject",
            "rank-one internal coherence closed",
            truth(rank_one_internal_coherence_closed),
            "The closeout is only honest after `.1783-.1786` has already fixed rho_exact = 1 theorem-level.",
        ),
        row(
            "gate_b_partial_proxy_promotion_retained",
            "pass" if gate_b_partial_proxy_promotion_retained else "reject",
            "Gate B partial proxy promotion retained",
            truth(gate_b_partial_proxy_promotion_retained),
            "The official read remains partial proxy promotion; the new theorem did not yet justify exact canonical promotion.",
        ),
        row(
            "exact_hh_diagonal_surface_missing",
            "pass" if exact_hh_diagonal_surface_missing else "reject",
            "exact HH diagonal surface missing",
            truth(exact_hh_diagonal_surface_missing),
            "The unresolved gap is now localized to the HH diagonal alone.",
        ),
        row(
            "rank_one_proxy_alpha_reference",
            "watch",
            "rank-one proxy alpha reference",
            coherence_summary["rank_one_alpha_with_energy_proxy"],
            "This is the scalar-compatible proxy alpha that would result if the current HH proxy were promoted unchanged under the exact coherence theorem.",
        ),
        row(
            "scalar_alpha_reference",
            "watch",
            "retained scalar alpha reference",
            proxy_summary["alpha_rho_min"],
            "The retained scalar strong candidate stays as the comparison scale for future HH-surface derivations.",
        ),
        row(
            "field_strength_alpha_reference",
            "watch",
            "field-strength alpha reference",
            field_summary["updated_field_strength_alpha_at_q_theory"],
            "The canonical FF channel remains the fixed lower component inside the retained rank-one pack.",
        ),
        row(
            "same_level_rank_one_retry_admissible",
            "reject",
            "same-level rank-one retry admissible",
            truth(same_level_rank_one_retry_admissible),
            "The rank-one pack is now theorem-level closed on the coherence side and should not be retried without a new HH theorem surface.",
        ),
        row(
            "primary_reopen_surface_fixed",
            "pass" if primary_reopen_surface_fixed else "reject",
            "primary reopen surface fixed",
            truth(primary_reopen_surface_fixed),
            "Primary reopen surface = exact HH diagonal surface under the retained rank-one mixed pack.",
        ),
        row(
            "secondary_reopen_surface_fixed",
            "pass" if secondary_reopen_surface_fixed else "reject",
            "secondary reopen surface fixed",
            truth(secondary_reopen_surface_fixed),
            "Secondary reopen surface = genuinely new non-rank-one mixed surface beyond the single-mediator theorem.",
        ),
        row(
            "reserve_reopen_surface_fixed",
            "pass" if reserve_reopen_surface_fixed else "reject",
            "reserve reopen surface fixed",
            truth(reserve_reopen_surface_fixed),
            "Reserve reopen surface = future external input or pack update that guides either HH-surface closure or a non-rank-one mixed pack.",
        ),
        row(
            "mixed_proxy_closeout_honest",
            "pass" if mixed_proxy_closeout_honest else "reject",
            "mixed proxy closeout honest",
            truth(mixed_proxy_closeout_honest),
            "The closeout is honest only if it keeps Gate B, retains the exact coherence theorem, and freezes the remaining HH gap explicitly.",
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
            "coherence_gate": display_path(COHERENCE_GATE),
            "decision_gate": display_path(DECISION_GATE),
            "proxy_gate": display_path(PROXY_GATE),
            "field_gate": display_path(FIELD_GATE),
        },
        "constants": {
            "rank_one_proxy_alpha_reference": coherence_summary["rank_one_alpha_with_energy_proxy"],
            "scalar_alpha_reference": proxy_summary["alpha_rho_min"],
            "field_strength_alpha_reference": field_summary["updated_field_strength_alpha_at_q_theory"],
            "primary_reopen_surface": "exact_hh_diagonal_surface_under_retained_rank_one_mixed_pack",
            "secondary_reopen_surface": "non_rank_one_mixed_surface_beyond_single_internal_mediator_pack",
            "reserve_reopen_surface": "future_external_input_or_pack_update_guiding_hh_or_non_rank_one_surface",
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "rank_one_internal_coherence_closed": rank_one_internal_coherence_closed,
        "gate_b_partial_proxy_promotion_retained": gate_b_partial_proxy_promotion_retained,
        "exact_hh_diagonal_surface_missing": exact_hh_diagonal_surface_missing,
        "rank_one_proxy_alpha_reference": coherence_summary["rank_one_alpha_with_energy_proxy"],
        "scalar_alpha_reference": proxy_summary["alpha_rho_min"],
        "field_strength_alpha_reference": field_summary["updated_field_strength_alpha_at_q_theory"],
        "same_level_rank_one_retry_admissible": same_level_rank_one_retry_admissible,
        "selected_primary_reopen_surface": "exact_hh_diagonal_surface_under_retained_rank_one_mixed_pack",
        "selected_secondary_reopen_surface": "non_rank_one_mixed_surface_beyond_single_internal_mediator_pack",
        "selected_reserve_reopen_surface": "future_external_input_or_pack_update_guiding_hh_or_non_rank_one_surface",
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": mixed_proxy_closeout_honest,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": build_formulae(),
        "hits": {
            "status_branch_hit": hit(status_text, "8.7.56.1787"),
            "roadmap_branch_hit": hit(roadmap_text, "次の公式 branch は `.1787-.1790`"),
            "current_problem_hit": hit(current_problem_text, "exact HH diagonal surface"),
            "current_status_hit": hit(current_status_text, "mixed proxy closeout / reopen registry"),
            "unified_roadmap_hit": hit(unified_text, "`.1783-.1786` は **exact internal coherence or HH surface reactivation**"),
            "long_roadmap_hit": hit(long_text, "26. `8.7.56.1787-.1790`"),
            "part5_hit": hit(part5_text, "`.1783-.1786` の **exact internal coherence or HH surface reactivation**"),
        },
        "carry_over": {
            "coherence_summary": coherence_summary,
            "decision_summary": decision_summary,
            "proxy_summary": proxy_summary,
            "field_summary": field_summary,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1787",
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
                "8.7.56.1788",
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
                "8.7.56.1789",
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
                "8.7.56.1790",
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
