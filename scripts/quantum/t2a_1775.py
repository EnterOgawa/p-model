#!/usr/bin/env python3
"""Generate 8.7.56.1775-.1778 mixed-source proxy observable recomputation artifacts.

This branch recomputes the mixed canonical observable on a proxy family opened
by `.1771-.1774`. The proxy uses

    A_FF = F_F,can(q_theory)
    A_HH = |F_E(q_theory)|
    A_FH = rho sqrt(A_FF A_HH)

with rho varied across the scalar-compatible high-coherence window.

The result is intentionally labeled as a proxy family rather than as an exact
canonical promotion, because the HH theorem surface is still missing.
"""

from __future__ import annotations

import csv
import json
import math
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

THRESHOLD_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1771_1774_mixed_eigenchannel_threshold_audit_declaration_gate_metrics.json"
FIELD_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1739_1742_field_strength_recompute_declaration_gate_metrics.json"
ENERGY_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1635_1638_energy_density_closeout_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.1775-1778"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor mixed-source proxy canonical observable recomputation"
STEM = build_compact_artifact_stem(STEP_TAG, "mixed_proxy_recompute", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_mixed_eigenchannel_proxy_instantiation_window_opened_proxy_recomputation_next"
BRANCH_CLASS = "vector_qball_form_factor_mixed_source_proxy_scalar_compatible_window_decision_gate_next"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_mixed_proxy_decision_gate_or_internal_coherence_reopen"
NEXT_ROUTE = "8.7.56.1779"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_exact_internal_coherence_or_hh_surface_reactivation"
FOLLOWUP_ROUTE = "8.7.56.1783"


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


# 関数: mixed固有値を返す。

def lambda_plus(a_ff: float, a_hh: float, rho: float) -> float:
    """Return the largest eigenvalue of the proxy mixed matrix."""
    offdiag = rho * math.sqrt(a_ff * a_hh)
    disc = (a_ff - a_hh) ** 2 + 4.0 * offdiag * offdiag
    return 0.5 * (a_ff + a_hh + math.sqrt(disc))


# 関数: alpha を返す。

def alpha_from_amplitude(value: float) -> float:
    """Return alpha = F^2 / (4 pi)."""
    return value * value / (4.0 * math.pi)


# 関数: `.1775-.1778` を実行する。

def main() -> None:
    """Execute the mixed proxy canonical observable recomputation branch."""
    for path in (STATUS, ROADMAP, AI_CONTEXT, WORK_HISTORY_RECENT, CURRENT_PROBLEM, CURRENT_STATUS, UNIFIED_ROADMAP, LONG_ROADMAP, PART5, THRESHOLD_GATE, FIELD_GATE, ENERGY_GATE):
        require(path)

    threshold_summary = read_json(THRESHOLD_GATE)["summary"]
    field_gate = read_json(FIELD_GATE)
    energy_gate = read_json(ENERGY_GATE)
    field_summary = field_gate["summary"]
    field_constants = field_gate["inputs"]["constants"]
    energy_summary = energy_gate["summary"]

    a_ff = float(field_summary["updated_field_strength_response_at_q_theory"])
    alpha_field = float(field_summary["updated_field_strength_alpha_at_q_theory"])
    f_scalar = float(math.sqrt(4.0 * math.pi * field_constants["scalar_alpha_exact_at_q_theory"]))
    alpha_scalar = float(field_constants["scalar_alpha_exact_at_q_theory"])
    a_hh_proxy = abs(float(energy_summary["official_F_E_at_q_theory"]))
    alpha_hh_proxy = float(energy_summary["official_alpha_E_at_q_theory"])
    rho_min = float(threshold_summary["rho_min_for_hh_proxy"])

    lambda_rho_min = lambda_plus(a_ff, a_hh_proxy, rho_min)
    alpha_rho_min = alpha_from_amplitude(lambda_rho_min)
    lambda_rho_08 = lambda_plus(a_ff, a_hh_proxy, 0.8)
    alpha_rho_08 = alpha_from_amplitude(lambda_rho_08)
    lambda_rho_09 = lambda_plus(a_ff, a_hh_proxy, 0.9)
    alpha_rho_09 = alpha_from_amplitude(lambda_rho_09)
    lambda_rho_1 = lambda_plus(a_ff, a_hh_proxy, 1.0)
    alpha_rho_1 = alpha_from_amplitude(lambda_rho_1)

    scalar_window_open = bool(alpha_rho_08 >= alpha_scalar and alpha_rho_min >= alpha_scalar)
    threshold_reproduces_scalar_target = bool(abs(alpha_rho_min - alpha_scalar) <= 1.0e-12)
    maximally_coherent_proxy_exceeds_scalar = bool(alpha_rho_1 > alpha_scalar)
    exact_canonical_promotion_selected = False
    partial_proxy_promotion_selected = bool(scalar_window_open)
    internal_coherence_or_exact_hh_surface_required = True
    physical_reject_not_selected = True
    proxy_recompute_honest = all((scalar_window_open, threshold_reproduces_scalar_target, maximally_coherent_proxy_exceeds_scalar, not exact_canonical_promotion_selected, partial_proxy_promotion_selected, internal_coherence_or_exact_hh_surface_required, physical_reject_not_selected))

    rows = [
        row("proxy_window_open", "pass" if scalar_window_open else "reject", "scalar-compatible proxy window open", truth(scalar_window_open), "The mixed proxy family opens only if the high-coherence window already reaches the retained scalar candidate."),
        row("rho_min", "watch", "minimal proxy coherence for scalar compatibility", rho_min, "This is the exact coherence floor derived in `.1771-.1774` for the present HH proxy magnitude."),
        row("lambda_rho_min", "watch", "proxy eigenchannel amplitude at rho_min", lambda_rho_min, "By construction this is the threshold-saturating proxy amplitude."),
        row("alpha_rho_min", "watch", "proxy alpha at rho_min", alpha_rho_min, "The threshold-saturating proxy exactly reproduces the retained scalar alpha candidate."),
        row("threshold_reproduces_scalar_target", "pass" if threshold_reproduces_scalar_target else "reject", "threshold proxy reproduces scalar target", truth(threshold_reproduces_scalar_target), "This equality is algebraic and confirms that the threshold audit and proxy recomputation are internally consistent."),
        row("lambda_rho_08", "watch", "proxy eigenchannel amplitude at rho = 0.8", lambda_rho_08, "The 0.8-coherence point is the first simple reference point above the threshold floor."),
        row("alpha_rho_08", "watch", "proxy alpha at rho = 0.8", alpha_rho_08, "A modest step above the threshold already nudges the proxy family slightly above the scalar target."),
        row("lambda_rho_09", "watch", "proxy eigenchannel amplitude at rho = 0.9", lambda_rho_09, "This point quantifies how quickly the proxy family rises once coherence approaches one."),
        row("alpha_rho_09", "watch", "proxy alpha at rho = 0.9", alpha_rho_09, "The high-coherence proxy at rho = 0.9 already overshoots the scalar target by a visible margin."),
        row("lambda_rho_1", "watch", "proxy eigenchannel amplitude at rho = 1", lambda_rho_1, "The rank-1 coherent limit is the largest amplitude reachable with the present HH proxy magnitude."),
        row("alpha_rho_1", "watch", "proxy alpha at rho = 1", alpha_rho_1, "The rank-1 coherent limit sits above the scalar target, so the current gap is no longer algebraic but interpretational."),
        row("maximally_coherent_proxy_exceeds_scalar", "pass" if maximally_coherent_proxy_exceeds_scalar else "reject", "maximally coherent proxy exceeds scalar target", truth(maximally_coherent_proxy_exceeds_scalar), "The proxy family shows that the current scalar target is reachable inside the mixed FF/HH eigenchannel window if coherence is high enough."),
        row("field_strength_alpha_reference", "watch", "field-strength canonical alpha reference", alpha_field, "The proxy family should be compared against the retained canonical field-strength read rather than against the old vector no-go scale alone."),
        row("energy_proxy_alpha_reference", "watch", "energy-core proxy alpha reference", alpha_hh_proxy, "This is the HH proxy diagonal carried from the exact energy-core lane."),
        row("exact_canonical_promotion_selected", "reject", "exact canonical promotion selected", truth(exact_canonical_promotion_selected), "The present branch still depends on one imported HH proxy, so exact canonical promotion remains premature."),
        row("partial_proxy_promotion_selected", "pass" if partial_proxy_promotion_selected else "reject", "partial proxy promotion selected", truth(partial_proxy_promotion_selected), "The honest read is that the mixed proxy family opens a scalar-compatible window without yet becoming the exact canonical observable."),
        row("internal_coherence_or_exact_hh_surface_required", "pass", "internal coherence or exact HH surface required", truth(internal_coherence_or_exact_hh_surface_required), "The remaining missing piece is no longer amplitude size but the exact theorem that fixes the HH diagonal and FF/HH coherence canonically."),
        row("physical_reject_not_selected", "pass", "physical reject not selected", truth(physical_reject_not_selected), "The mixed proxy family remains constructive and does not force physical rejection."),
        row("proxy_recompute_honest", "pass" if proxy_recompute_honest else "reject", "mixed proxy recomputation honest", truth(proxy_recompute_honest), "The branch is honest only if it reports a scalar-compatible proxy window while keeping exact canonical promotion closed."),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS), "roadmap": display_path(ROADMAP), "ai_context": display_path(AI_CONTEXT), "work_history_recent": display_path(WORK_HISTORY_RECENT), "current_problem": display_path(CURRENT_PROBLEM), "current_status": display_path(CURRENT_STATUS), "unified_roadmap": display_path(UNIFIED_ROADMAP), "long_roadmap": display_path(LONG_ROADMAP), "part5": display_path(PART5), "threshold_gate": display_path(THRESHOLD_GATE), "field_gate": display_path(FIELD_GATE), "energy_gate": display_path(ENERGY_GATE),
        },
        "constants": {
            "field_strength_response_at_q_theory": a_ff, "field_strength_alpha_at_q_theory": alpha_field, "energy_proxy_response_abs_at_q_theory": a_hh_proxy, "energy_proxy_alpha_at_q_theory": alpha_hh_proxy, "scalar_response_exact_at_q_theory": f_scalar, "scalar_alpha_exact_at_q_theory": alpha_scalar, "rho_min": rho_min, "lambda_rho_min": lambda_rho_min, "alpha_rho_min": alpha_rho_min, "lambda_rho_08": lambda_rho_08, "alpha_rho_08": alpha_rho_08, "lambda_rho_09": lambda_rho_09, "alpha_rho_09": alpha_rho_09, "lambda_rho_1": lambda_rho_1, "alpha_rho_1": alpha_rho_1, "next_route_name": NEXT_ROUTE_NAME, "next_route": NEXT_ROUTE, "followup_route_name": FOLLOWUP_ROUTE_NAME, "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "proxy_window_open": scalar_window_open,
        "rho_min": rho_min,
        "lambda_rho_min": lambda_rho_min,
        "alpha_rho_min": alpha_rho_min,
        "lambda_rho_08": lambda_rho_08,
        "alpha_rho_08": alpha_rho_08,
        "lambda_rho_09": lambda_rho_09,
        "alpha_rho_09": alpha_rho_09,
        "lambda_rho_1": lambda_rho_1,
        "alpha_rho_1": alpha_rho_1,
        "threshold_reproduces_scalar_target": threshold_reproduces_scalar_target,
        "maximally_coherent_proxy_exceeds_scalar": maximally_coherent_proxy_exceeds_scalar,
        "exact_canonical_promotion_selected": exact_canonical_promotion_selected,
        "partial_proxy_promotion_selected": partial_proxy_promotion_selected,
        "internal_coherence_or_exact_hh_surface_required": internal_coherence_or_exact_hh_surface_required,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    decision = {"overall_status": f"{BRANCH_CLASS}_declared", "branch_completed": proxy_recompute_honest, "next_required_artifacts": [NEXT_ROUTE_NAME]}
    evidence = {"proxy_family_definition": {"a_ff": "F_F,can(q_theory)", "a_hh": "|F_E(q_theory)|", "a_fh": "rho sqrt(A_FF A_HH)"}, "carry_over": {"threshold_summary": threshold_summary, "field_summary": field_summary, "energy_summary": energy_summary}}
    manifest = {
        "inventory": write_artifact("inventory", payload("8.7.56.1775", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence)),
        "audit": write_artifact("audit", payload("8.7.56.1776", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence)),
        "declaration_gate": write_artifact("declaration_gate", payload("8.7.56.1777", f"{STEP_NAME} declaration gate", inputs, rows, summary, decision, evidence)),
        "route_sync": write_artifact("route_sync", payload("8.7.56.1778", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence)),
    }
    print(json.dumps({"step": STEP_TAG, "stem": STEM, "manifest": manifest, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
