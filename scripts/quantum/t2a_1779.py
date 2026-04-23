#!/usr/bin/env python3
"""Generate 8.7.56.1779-.1782 mixed proxy decision-gate artifacts."""

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

PROXY_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1775_1778_mixed_proxy_recompute_declaration_gate_metrics.json"
FIELD_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1739_1742_field_strength_recompute_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.1779-1782"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor mixed proxy decision gate / internal coherence reopen"
STEM = build_compact_artifact_stem(STEP_TAG, "mixed_proxy_decision_gate", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_mixed_source_proxy_scalar_compatible_window_decision_gate_next"
BRANCH_CLASS = "vector_qball_form_factor_mixed_proxy_gate_b_partial_scalar_compatible_internal_coherence_reopen_next"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_exact_internal_coherence_or_hh_surface_reactivation"
NEXT_ROUTE = "8.7.56.1783"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_mixed_proxy_closeout_or_reopen_registry"
FOLLOWUP_ROUTE = "8.7.56.1787"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input file is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON を読み込む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: repo相対パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative display path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: metrics row を構築する。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {"row_id": row_id, "status": status, "metric": metric, "value": float(value), "note": note}


# 関数: payload を構築する。

def payload(step: str, name: str, inputs: dict, rows: list[dict], summary: dict, decision: dict, evidence: dict) -> dict:
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
    paths["json"].write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": display_path(paths["json"]), "csv": display_path(paths["csv"])}


# 関数: 真偽値を 0/1 に変換する。

def truth(value: bool) -> float:
    """Convert one boolean into 0/1 float form."""
    return 1.0 if value else 0.0


# 関数: `.1779-.1782` を実行する。

def main() -> None:
    """Execute the mixed proxy decision gate branch."""
    for path in (STATUS, ROADMAP, AI_CONTEXT, WORK_HISTORY_RECENT, CURRENT_PROBLEM, CURRENT_STATUS, UNIFIED_ROADMAP, LONG_ROADMAP, PART5, PROXY_GATE, FIELD_GATE):
        require(path)

    proxy_summary = read_json(PROXY_GATE)["summary"]
    field_summary = read_json(FIELD_GATE)["summary"]

    gate_a_exact_promote_selected = False
    gate_b_partial_proxy_promotion_selected = bool(proxy_summary["partial_proxy_promotion_selected"])
    gate_c_reject_selected = False
    scalar_window_open = bool(proxy_summary["proxy_window_open"])
    exact_canonical_promotion_selected = bool(proxy_summary["exact_canonical_promotion_selected"])
    internal_coherence_or_exact_hh_surface_required = bool(proxy_summary["internal_coherence_or_exact_hh_surface_required"])
    same_level_proxy_retry_without_new_internal_surface_admissible = False
    physical_reject_not_selected = True
    decision_gate_honest = all(
        (
            scalar_window_open,
            gate_b_partial_proxy_promotion_selected,
            not gate_a_exact_promote_selected,
            not gate_c_reject_selected,
            not exact_canonical_promotion_selected,
            internal_coherence_or_exact_hh_surface_required,
            not same_level_proxy_retry_without_new_internal_surface_admissible,
            physical_reject_not_selected,
        )
    )

    rows = [
        row("scalar_window_open", "pass" if scalar_window_open else "reject", "scalar-compatible proxy window open", truth(scalar_window_open), "The decision gate is only meaningful after the proxy family has already opened a scalar-compatible window."),
        row("gate_a_exact_promote_selected", "reject", "Gate A exact promote selected", truth(gate_a_exact_promote_selected), "Exact canonical promotion remains closed because the HH/coherence theorem is still missing."),
        row("gate_b_partial_proxy_promotion_selected", "pass" if gate_b_partial_proxy_promotion_selected else "reject", "Gate B partial proxy promotion selected", truth(gate_b_partial_proxy_promotion_selected), "The honest read is to retain the scalar-compatible proxy window without over-claiming exact canonical closure."),
        row("gate_c_reject_selected", "reject", "Gate C reject selected", truth(gate_c_reject_selected), "The mixed proxy family is constructive enough that outright rejection would be too strong."),
        row("field_strength_alpha_reference", "watch", "field-strength canonical alpha reference", field_summary["updated_field_strength_alpha_at_q_theory"], "The proxy decision gate should be read relative to the already canonized field-strength observable."),
        row("proxy_threshold_alpha", "watch", "threshold proxy alpha", proxy_summary["alpha_rho_min"], "The threshold proxy reproduces the retained scalar target exactly but only in a proxy, not exact, sense."),
        row("proxy_max_alpha", "watch", "maximally coherent proxy alpha", proxy_summary["alpha_rho_1"], "The rank-1 coherent proxy overshoots the scalar target, confirming that the remaining gap is theorem-level rather than amplitude-level."),
        row("internal_coherence_or_exact_hh_surface_required", "pass" if internal_coherence_or_exact_hh_surface_required else "reject", "internal coherence or exact HH surface required", truth(internal_coherence_or_exact_hh_surface_required), "The next missing surface is the exact theorem that fixes HH and FF/HH coherence inside the mixed channel."),
        row("same_level_proxy_retry_without_new_internal_surface_admissible", "reject", "same-level proxy retry without new internal surface admissible", truth(same_level_proxy_retry_without_new_internal_surface_admissible), "The proxy family should not be retried mechanically without a new internal theorem surface."),
        row("physical_reject_not_selected", "pass", "physical reject not selected", truth(physical_reject_not_selected), "The route remains open and does not force physical rejection."),
        row("decision_gate_honest", "pass" if decision_gate_honest else "reject", "mixed proxy decision gate honest", truth(decision_gate_honest), "The decision gate is honest only if it freezes Gate B and moves the roadmap to the missing internal-coherence surface."),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS), "roadmap": display_path(ROADMAP), "ai_context": display_path(AI_CONTEXT), "work_history_recent": display_path(WORK_HISTORY_RECENT), "current_problem": display_path(CURRENT_PROBLEM), "current_status": display_path(CURRENT_STATUS), "unified_roadmap": display_path(UNIFIED_ROADMAP), "long_roadmap": display_path(LONG_ROADMAP), "part5": display_path(PART5), "proxy_gate": display_path(PROXY_GATE), "field_gate": display_path(FIELD_GATE),
        },
        "constants": {
            "proxy_threshold_alpha": proxy_summary["alpha_rho_min"], "proxy_max_alpha": proxy_summary["alpha_rho_1"], "rho_min": proxy_summary["rho_min"], "next_route_name": NEXT_ROUTE_NAME, "next_route": NEXT_ROUTE, "followup_route_name": FOLLOWUP_ROUTE_NAME, "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "gate_a_exact_promote_selected": gate_a_exact_promote_selected,
        "gate_b_partial_proxy_promotion_selected": gate_b_partial_proxy_promotion_selected,
        "gate_c_reject_selected": gate_c_reject_selected,
        "scalar_window_open": scalar_window_open,
        "exact_canonical_promotion_selected": exact_canonical_promotion_selected,
        "proxy_threshold_alpha": proxy_summary["alpha_rho_min"],
        "proxy_max_alpha": proxy_summary["alpha_rho_1"],
        "internal_coherence_or_exact_hh_surface_required": internal_coherence_or_exact_hh_surface_required,
        "same_level_proxy_retry_without_new_internal_surface_admissible": same_level_proxy_retry_without_new_internal_surface_admissible,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    decision = {"overall_status": f"{BRANCH_CLASS}_declared", "branch_completed": decision_gate_honest, "next_required_artifacts": [NEXT_ROUTE_NAME]}
    evidence = {"carry_over": {"proxy_summary": proxy_summary, "field_summary": field_summary}}
    manifest = {
        "inventory": write_artifact("inventory", payload("8.7.56.1779", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence)),
        "audit": write_artifact("audit", payload("8.7.56.1780", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence)),
        "declaration_gate": write_artifact("declaration_gate", payload("8.7.56.1781", f"{STEP_NAME} declaration gate", inputs, rows, summary, decision, evidence)),
        "route_sync": write_artifact("route_sync", payload("8.7.56.1782", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence)),
    }
    print(json.dumps({"step": STEP_TAG, "stem": STEM, "manifest": manifest, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
