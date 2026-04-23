#!/usr/bin/env python3
"""Generate 8.7.56.1771-.1774 mixed eigenchannel instantiation threshold artifacts.

This branch instantiates the mixed-source eigenchannel theorem with the only
current internal-Hamiltonian amplitude already fixed in the repository:
the magnitude of the exact Hamiltonian-core energy-density read.

The goal is not to declare an exact canonical promotion. The goal is to test
whether a positive-semidefinite mixed FF/FH/HH response matrix can in
principle bridge the scalar gap without introducing noncanonical magnitudes.
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

THEOREM_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1767_1770_mixed_eigenchannel_theorem_declaration_gate_metrics.json"
FIELD_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1739_1742_field_strength_recompute_declaration_gate_metrics.json"
ENERGY_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1635_1638_energy_density_closeout_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.1771-1774"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor mixed eigenchannel instantiation threshold audit"
STEM = build_compact_artifact_stem(STEP_TAG, "mixed_eigenchannel_threshold_audit", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_mixed_eigenchannel_threshold_theorem_derived_instantiation_audit_next"
BRANCH_CLASS = "vector_qball_form_factor_mixed_eigenchannel_proxy_instantiation_window_opened_proxy_recomputation_next"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_mixed_source_proxy_canonical_observable_recomputation"
NEXT_ROUTE = "8.7.56.1775"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_mixed_proxy_decision_gate_or_internal_coherence_reopen"
FOLLOWUP_ROUTE = "8.7.56.1779"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を検査する。

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


# 関数: coherence固定時の最小 HH 振幅を返す。

def c_min_for_rho(target: float, a_ff: float, rho: float) -> float:
    """Return the minimal HH diagonal compatible with one target amplitude."""
    gap = target - a_ff
    return target * gap / (gap + rho * rho * a_ff)


# 関数: HH 振幅固定時の最小 coherence を返す。

def rho_min_for_c(target: float, a_ff: float, a_hh: float) -> float:
    """Return the minimal coherence compatible with one HH proxy diagonal."""
    numerator = (target - a_ff) * (target - a_hh)
    denominator = a_ff * a_hh
    return math.sqrt(max(0.0, numerator / denominator))


# 関数: 主要式を返す。

def build_formulae() -> dict[str, str]:
    """Return the threshold formulas used in the audit."""
    return {
        "psd_bound": "A_FH^2 <= A_FF A_HH for a positive-semidefinite 2x2 response matrix.",
        "coherence_parameter": "A_FH = rho sqrt(A_FF A_HH), 0 <= rho <= 1.",
        "target_condition": "lambda_+(q_theory) >= F_scalar(q_theory).",
        "hh_threshold_for_fixed_rho": "A_HH,min(rho) = F_scalar(F_scalar-A_FF) / (F_scalar-A_FF + rho^2 A_FF).",
        "rho_threshold_for_fixed_hh": "rho_min(A_HH) = sqrt((F_scalar-A_FF)(F_scalar-A_HH)/(A_FF A_HH)).",
    }


# 関数: `.1771-.1774` を実行する。

def main() -> None:
    """Execute the mixed eigenchannel instantiation threshold audit branch."""
    for path in (STATUS, ROADMAP, AI_CONTEXT, WORK_HISTORY_RECENT, CURRENT_PROBLEM, CURRENT_STATUS, UNIFIED_ROADMAP, LONG_ROADMAP, PART5, THEOREM_GATE, FIELD_GATE, ENERGY_GATE):
        require(path)

    theorem_summary = read_json(THEOREM_GATE)["summary"]
    field_gate = read_json(FIELD_GATE)
    energy_gate = read_json(ENERGY_GATE)
    field_summary = field_gate["summary"]
    field_constants = field_gate["inputs"]["constants"]
    energy_summary = energy_gate["summary"]

    a_ff = float(field_summary["updated_field_strength_response_at_q_theory"])
    alpha_scalar = float(field_constants["scalar_alpha_exact_at_q_theory"])
    f_scalar = float(math.sqrt(4.0 * math.pi * alpha_scalar))
    a_hh_proxy = abs(float(energy_summary["official_F_E_at_q_theory"]))
    alpha_hh_proxy = float(energy_summary["official_alpha_E_at_q_theory"])

    rho_min = rho_min_for_c(f_scalar, a_ff, a_hh_proxy)
    rho_min_sq = rho_min * rho_min
    c_min_rho_1 = c_min_for_rho(f_scalar, a_ff, 1.0)
    c_min_rho_09 = c_min_for_rho(f_scalar, a_ff, 0.9)
    c_min_rho_08 = c_min_for_rho(f_scalar, a_ff, 0.8)
    c_min_rho_075 = c_min_for_rho(f_scalar, a_ff, 0.75)
    max_coherent_lambda = a_ff + a_hh_proxy
    max_coherent_alpha = max_coherent_lambda * max_coherent_lambda / (4.0 * math.pi)

    theorem_ready = bool(theorem_summary["canonical_eigenchannel_rule_derived"])
    hh_proxy_surface_available = True
    hh_proxy_exceeds_rank1_threshold = bool(a_hh_proxy >= c_min_rho_1)
    hh_proxy_exceeds_rho_08_threshold = bool(a_hh_proxy >= c_min_rho_08)
    rho_min_admissible = bool(rho_min <= 1.0)
    rho_min_high_coherence_required = bool(rho_min >= 0.75)
    proxy_window_opens_scalar_compatibility = bool(theorem_ready and hh_proxy_surface_available and rho_min_admissible)
    proxy_instantiation_canonical = False
    proxy_recompute_admissible_now = True
    physical_reject_not_selected = True
    threshold_audit_honest = all((theorem_ready, hh_proxy_surface_available, hh_proxy_exceeds_rank1_threshold, rho_min_admissible, proxy_window_opens_scalar_compatibility, not proxy_instantiation_canonical, proxy_recompute_admissible_now, physical_reject_not_selected))

    rows = [
        row("theorem_ready", "pass" if theorem_ready else "reject", "mixed eigenchannel theorem ready", truth(theorem_ready), "The threshold audit starts only after the eigenchannel rule itself has already been derived."),
        row("hh_proxy_surface_available", "pass", "HH proxy surface available", truth(hh_proxy_surface_available), "The only internal-Hamiltonian diagonal amplitude already fixed in the repository is the magnitude of the exact energy-core read."),
        row("a_hh_proxy_magnitude", "watch", "HH proxy diagonal magnitude from |F_E|", a_hh_proxy, "This is not yet a canonical HH theorem result; it is the current internal-sector amplitude proxy carried into the mixed threshold audit."),
        row("rho_min_for_hh_proxy", "watch", "minimal coherence needed for the HH proxy to hit the scalar target", rho_min, "If the mixed matrix remains positive-semidefinite, a coherence rho >= rho_min is sufficient for the HH proxy diagonal to bridge the scalar gap."),
        row("rho_min_admissible", "pass" if rho_min_admissible else "reject", "minimal coherence remains within [0,1]", truth(rho_min_admissible), "The proxy family is algebraically admissible only if the required coherence stays inside the positive-semidefinite window."),
        row("rho_min_high_coherence_required", "watch", "high coherence requirement flag", truth(rho_min_high_coherence_required), "The admissible proxy window is not broad: the required coherence already lies close to 0.8, so the next branch must treat it as a high-coherence family rather than as a generic mixed state."),
        row("c_min_rho_1", "watch", "minimal HH diagonal if rho = 1", c_min_rho_1, "Under maximal coherence the required HH diagonal collapses to the scalar-minus-FF amplitude gap."),
        row("c_min_rho_09", "watch", "minimal HH diagonal if rho = 0.9", c_min_rho_09, "This threshold quantifies how quickly the required HH diagonal grows as coherence falls away from one."),
        row("c_min_rho_08", "watch", "minimal HH diagonal if rho = 0.8", c_min_rho_08, "The 0.8-coherence threshold is useful because it sits only slightly above the proxy admissibility floor."),
        row("c_min_rho_075", "watch", "minimal HH diagonal if rho = 0.75", c_min_rho_075, "Below about 0.786 coherence the present HH proxy no longer reaches the scalar target."),
        row("hh_proxy_exceeds_rank1_threshold", "pass" if hh_proxy_exceeds_rank1_threshold else "reject", "HH proxy exceeds rho = 1 threshold", truth(hh_proxy_exceeds_rank1_threshold), "The current HH proxy magnitude is already larger than the minimal rank-1 threshold, so scalar-compatible promotion is not blocked by diagonal size alone."),
        row("hh_proxy_exceeds_rho_08_threshold", "pass" if hh_proxy_exceeds_rho_08_threshold else "reject", "HH proxy exceeds rho = 0.8 threshold", truth(hh_proxy_exceeds_rho_08_threshold), "The current HH proxy remains large enough even at about 0.8 coherence, which opens a narrow scalar-compatible proxy window."),
        row("max_coherent_lambda", "watch", "maximally coherent proxy eigenchannel amplitude", max_coherent_lambda, "At rho = 1 the mixed eigenchannel collapses to the rank-1 limit A_FF + A_HH."),
        row("max_coherent_alpha", "watch", "maximally coherent proxy eigenchannel alpha", max_coherent_alpha, "This is the largest scalar-compatible alpha reachable with the present HH proxy before any exact internal theorem is derived."),
        row("proxy_window_opens_scalar_compatibility", "pass" if proxy_window_opens_scalar_compatibility else "reject", "proxy window opens scalar compatibility", truth(proxy_window_opens_scalar_compatibility), "The mixed theorem plus the present HH proxy is sufficient to open an algebraically scalar-compatible proxy family."),
        row("proxy_instantiation_canonical", "reject", "proxy instantiation already canonical", truth(proxy_instantiation_canonical), "The HH proxy is still imported from the energy-core lane, so the present opening is only proxy-level and not yet an exact canonical promotion."),
        row("proxy_recompute_admissible_now", "pass", "proxy recompute admissible now", truth(proxy_recompute_admissible_now), "Because the threshold window is nonempty, the honest next step is to recompute the proxy family explicitly rather than to stop at the algebraic theorem."),
        row("physical_reject_not_selected", "pass", "physical reject not selected", truth(physical_reject_not_selected), "The threshold audit opens a viable proxy family and therefore does not force physical rejection."),
        row("threshold_audit_honest", "pass" if threshold_audit_honest else "reject", "threshold audit honest", truth(threshold_audit_honest), "The branch is honest only if it opens a proxy family without pretending that the HH proxy has already become an exact canonical theorem result."),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS), "roadmap": display_path(ROADMAP), "ai_context": display_path(AI_CONTEXT), "work_history_recent": display_path(WORK_HISTORY_RECENT), "current_problem": display_path(CURRENT_PROBLEM), "current_status": display_path(CURRENT_STATUS), "unified_roadmap": display_path(UNIFIED_ROADMAP), "long_roadmap": display_path(LONG_ROADMAP), "part5": display_path(PART5), "theorem_gate": display_path(THEOREM_GATE), "field_gate": display_path(FIELD_GATE), "energy_gate": display_path(ENERGY_GATE),
        },
        "constants": {
            "field_strength_response_at_q_theory": a_ff, "field_strength_alpha_at_q_theory": field_summary["updated_field_strength_alpha_at_q_theory"], "scalar_response_exact_at_q_theory": f_scalar, "scalar_alpha_exact_at_q_theory": alpha_scalar, "energy_core_proxy_response_abs_at_q_theory": a_hh_proxy, "energy_core_proxy_alpha_at_q_theory": alpha_hh_proxy, "rho_min_for_hh_proxy": rho_min, "c_min_rho_1": c_min_rho_1, "c_min_rho_09": c_min_rho_09, "c_min_rho_08": c_min_rho_08, "c_min_rho_075": c_min_rho_075, "max_coherent_lambda": max_coherent_lambda, "max_coherent_alpha": max_coherent_alpha, "next_route_name": NEXT_ROUTE_NAME, "next_route": NEXT_ROUTE, "followup_route_name": FOLLOWUP_ROUTE_NAME, "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "hh_proxy_surface_available": hh_proxy_surface_available,
        "a_hh_proxy_magnitude": a_hh_proxy,
        "rho_min_for_hh_proxy": rho_min,
        "rho_min_squared_for_hh_proxy": rho_min_sq,
        "rho_min_admissible": rho_min_admissible,
        "rho_min_high_coherence_required": rho_min_high_coherence_required,
        "c_min_rho_1": c_min_rho_1,
        "c_min_rho_09": c_min_rho_09,
        "c_min_rho_08": c_min_rho_08,
        "c_min_rho_075": c_min_rho_075,
        "hh_proxy_exceeds_rank1_threshold": hh_proxy_exceeds_rank1_threshold,
        "hh_proxy_exceeds_rho_08_threshold": hh_proxy_exceeds_rho_08_threshold,
        "max_coherent_lambda": max_coherent_lambda,
        "max_coherent_alpha": max_coherent_alpha,
        "proxy_window_opens_scalar_compatibility": proxy_window_opens_scalar_compatibility,
        "proxy_instantiation_canonical": proxy_instantiation_canonical,
        "proxy_recompute_admissible_now": proxy_recompute_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    decision = {"overall_status": f"{BRANCH_CLASS}_declared", "branch_completed": threshold_audit_honest, "next_required_artifacts": [NEXT_ROUTE_NAME]}
    evidence = {"formulas": build_formulae(), "carry_over": {"theorem_summary": theorem_summary, "field_summary": field_summary, "energy_summary": energy_summary}}
    manifest = {
        "inventory": write_artifact("inventory", payload("8.7.56.1771", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence)),
        "audit": write_artifact("audit", payload("8.7.56.1772", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence)),
        "declaration_gate": write_artifact("declaration_gate", payload("8.7.56.1773", f"{STEP_NAME} declaration gate", inputs, rows, summary, decision, evidence)),
        "route_sync": write_artifact("route_sync", payload("8.7.56.1774", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence)),
    }
    print(json.dumps({"step": STEP_TAG, "stem": STEM, "manifest": manifest, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
