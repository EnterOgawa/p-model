#!/usr/bin/env python3
"""Generate 8.7.56.1487-.1490 effective-source-theorem attempt artifacts.

This branch uses the restored corrected exact vector branch from 8.7.56.1483-.1486
and asks the next honest theorem-side question:

- does the current public pack already contain enough action-level structure to
  derive an exact `a_mu J_eff^mu[P^Qball]` source theorem?

If not, the missing piece should be localized before any observable-dictionary
work is allowed to proceed.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
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
CASE_GAMMA_ADVICE = ROOT / "doc" / "quantum" / "42_trial2_numeric_alpha_vector_qball_case_gamma_advice_request.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
NEXT_STEPS_NOTE = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")

ANCHOR_GATE = PUBLIC_OUT / "q_8_7_56_1483_1486_ell0_anchor_continuation_declaration_gate_metrics.json"
ANCHOR_EVAL = PUBLIC_OUT / "q_8_7_56_1483_1486_ell0_anchor_continuation_numeric_evaluation_metrics.json"
OLD_SOURCE_GATE = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_source_theorem_attempt_declaration_gate_metrics.json"
OLD_NO_GO_GATE = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_theorem_review_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.1487-1490"
STEM = build_compact_artifact_stem(STEP_TAG, "effective_source_theorem", prefix="q")
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor effective source theorem attempt"

PRIOR_CLASS = "vector_qball_form_factor_corrected_anchor_preserving_continuation_restored_blind_vector_no_go_retained"
FAIL_CLASS = "vector_qball_form_factor_effective_source_theorem_failed_noether_current_gap_retained"
PASS_CLASS = "vector_qball_form_factor_effective_source_theorem_derived_proxy_support_available"
FAIL_ROUTE = "trial2_numeric_alpha_vector_qball_form_factor_exact_charge_current_noether_closure_audit"
PASS_ROUTE = "trial2_numeric_alpha_vector_qball_form_factor_observable_dictionary_gate"
FAIL_STEP = "8.7.56.1491"
PASS_STEP = "8.7.56.1495"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Fail when one required input is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8テキストを読み込む。

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# 関数: UTF-8 JSONを読み込む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: 表示用の相対パス文字列へ変換する。

def display_path(path: Path) -> str:
    """Convert one path to repo-relative display form when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分文字列に一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first matching line for one substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: metrics rowを生成する。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {"row_id": row_id, "status": status, "metric": metric, "value": float(value), "note": note}


# 関数: payloadを生成する。

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


# 関数: compact stemでJSON/CSVを出力する。

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and CSV rows table."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": display_path(paths["json"]), "csv": display_path(paths["csv"])}


# 関数: 真偽値を0/1へ変換する。

def truth(value: bool) -> float:
    """Convert one boolean into 0/1 float form."""
    return 1.0 if value else 0.0


# 関数: `.1487-.1490` を実行する。

def main() -> None:
    """Execute the effective-source-theorem attempt branch."""
    for path in (
        STATUS, ROADMAP, AI_CONTEXT, WORK_HISTORY_RECENT, CURRENT_PROBLEM, CURRENT_STATUS, UNIFIED_ROADMAP,
        CASE_GAMMA_ADVICE, PART1, PART3A, PART5, NEXT_STEPS_NOTE, ANCHOR_GATE, ANCHOR_EVAL, OLD_SOURCE_GATE, OLD_NO_GO_GATE,
    ):
        require(path)

    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    note_text = read_text(NEXT_STEPS_NOTE)
    anchor_gate = read_json(ANCHOR_GATE)["summary"]
    anchor_eval = read_json(ANCHOR_EVAL)["summary"]
    old_source_gate = read_json(OLD_SOURCE_GATE)["summary"]
    old_no_go_gate = read_json(OLD_NO_GO_GATE)["summary"]

    part1_current = hit(part1_text, "J^\\mu_{\\mathrm{matter}}=(\\rho c,\\rho \\mathbf{v})")
    part1_interaction = hit(part1_text, "\\mathcal{L}_{\\mathrm{int}}=g_P\\,P_\\mu J^\\mu_{\\mathrm{matter}}")
    part1_total = hit(part1_text, "\\mathcal{L}_{P,\\mathrm{full}}")
    part1_energy_noether = hit(part1_text, "Noether保存則")
    charge_patterns = ("Noether電流", "電荷 Noether", "charge current", "保存電流")
    part1_charge_noether = any(hit(part1_text, pattern) is not None for pattern in charge_patterns)
    note_step_c = hit(note_text, "### Step C.")
    note_effective_source = hit(note_text, "\\mathcal L \\supset a_\\mu\\,J^{\\mu}_{\\rm eff}[P^{\\rm Qball}]")
    note_proxy = hit(note_text, "|f_0|^2 - |f_L|^2")
    note_low_order = hit(note_text, "J_eff^0")
    part3a_fail = hit(part3a_text, "effective source formula `J^\\mu_{\\rm eff}[P^{\\rm Qball}]`")
    part5_fail = hit(part5_text, "effective source formula `J^\\mu_{\\rm eff}[P^{\\rm Qball}]`")

    restored_branch = bool(anchor_gate.get("anchor_preserving_continuation_restored", False))
    blind_no_go = bool(anchor_gate.get("blind_vector_no_go_retained", False))
    explicit_qball_background = bool(old_source_gate.get("explicit_qball_background_expansion_available", False))
    explicit_effective_source = bool(old_source_gate.get("explicit_effective_source_formula_available", False))
    independent_connection = bool(old_source_gate.get("independent_connection_caveat_present", False))
    prior_route_local_no_go = bool(old_no_go_gate.get("route_local_no_go_theorem_honest", False))

    source_ready = all((restored_branch, part1_current, part1_interaction, part1_total, note_step_c, note_effective_source, note_proxy))
    exact_source_derived = all((source_ready, explicit_qball_background, explicit_effective_source, part1_charge_noether))
    low_order_proxy_executable = all((note_low_order, note_proxy, explicit_effective_source, part1_charge_noether))
    low_order_proxy_passed = exact_source_derived and low_order_proxy_executable
    noether_gap_required = source_ready and not exact_source_derived
    observable_dictionary_now = exact_source_derived

    branch_class = PASS_CLASS if exact_source_derived else FAIL_CLASS
    next_route = PASS_ROUTE if exact_source_derived else FAIL_ROUTE
    next_step = PASS_STEP if exact_source_derived else FAIL_STEP

    rows = [
        row("restored_exact_vector_branch_available", "pass" if restored_branch else "reject", "restored exact vector branch available", truth(restored_branch), "The source theorem attempt is only honest after anchor-preserving corrected continuation is restored."),
        row("explicit_matter_current_surface_available", "pass" if part1_current else "reject", "explicit matter current surface available", truth(part1_current is not None), "Part I still carries the explicit matter-current surface."),
        row("explicit_interaction_surface_available", "pass" if part1_interaction else "reject", "explicit interaction surface available", truth(part1_interaction is not None), "Part I still carries the explicit vector-current interaction term."),
        row("exact_total_vector_action_surface_available", "pass" if part1_total else "reject", "exact total vector action surface available", truth(part1_total is not None), "The free-plus-interaction backbone is visible in the current public pack."),
        row("energy_noether_surface_available", "pass" if part1_energy_noether else "reject", "energy Noether surface available", truth(part1_energy_noether is not None), "Energy-side Noether structure is present in Part I."),
        row("charge_noether_current_surface_available", "pass" if part1_charge_noether else "reject", "charge/current Noether surface available", truth(part1_charge_noether), "An exact charge-current closure would be needed to derive `J_eff^0` without a proxy leap."),
        row("explicit_qball_background_expansion_available", "pass" if explicit_qball_background else "reject", "explicit Q-ball background expansion available", truth(explicit_qball_background), "The old current-canon source-theorem audit already localized this missing surface."),
        row("explicit_effective_source_formula_available", "pass" if explicit_effective_source else "reject", "explicit effective source formula available", truth(explicit_effective_source), "The present pack still lacks an explicit `a_mu J_eff^mu[P^Qball]` formula."),
        row("effective_source_theorem_attempt_ready", "pass" if source_ready else "reject", "effective source theorem attempt ready", truth(source_ready), "Restored exact branch plus action/current surfaces are enough to ask the theorem question honestly."),
        row("exact_source_theorem_derived", "pass" if exact_source_derived else "reject", "exact source theorem derived", truth(exact_source_derived), "The theorem only passes if background expansion, effective source formula, and exact charge-current closure all surface."),
        row("exact_charge_current_noether_closure_required", "pass" if noether_gap_required else "reject", "exact charge-current / Noether closure required", truth(noether_gap_required), "This becomes the next honest blocker if the theorem attempt remains incomplete."),
        row("observable_dictionary_gate_admissible_now", "pass" if observable_dictionary_now else "reject", "observable dictionary gate admissible now", truth(observable_dictionary_now), "Observable-dictionary work stays downstream until the source theorem is actually derived."),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS), "roadmap": display_path(ROADMAP), "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT), "current_problem_note": display_path(CURRENT_PROBLEM),
            "current_status_note": display_path(CURRENT_STATUS), "unified_roadmap_note": display_path(UNIFIED_ROADMAP),
            "case_gamma_advice_note": display_path(CASE_GAMMA_ADVICE), "part1": display_path(PART1),
            "part3a": display_path(PART3A), "part5": display_path(PART5), "next_steps_note": display_path(NEXT_STEPS_NOTE),
        },
        "prior_metrics": {
            "anchor_gate": display_path(ANCHOR_GATE), "anchor_eval": display_path(ANCHOR_EVAL),
            "old_source_gate": display_path(OLD_SOURCE_GATE), "old_no_go_gate": display_path(OLD_NO_GO_GATE),
        },
        "constants": {"next_route_name": next_route, "next_route": next_step},
    }
    summary = {
        "trial2_numeric_alpha_problem_classification": branch_class,
        "prior_problem_classification": PRIOR_CLASS,
        "restored_exact_vector_branch_available": restored_branch,
        "blind_vector_no_go_retained": blind_no_go,
        "effective_source_theorem_attempt_ready": source_ready,
        "exact_source_theorem_derived": exact_source_derived,
        "low_order_proxy_reduction_executable": low_order_proxy_executable,
        "low_order_proxy_reduction_passed": low_order_proxy_passed,
        "exact_charge_current_noether_closure_required": noether_gap_required,
        "explicit_qball_background_expansion_available": explicit_qball_background,
        "explicit_effective_source_formula_available": explicit_effective_source,
        "independent_connection_caveat_present": independent_connection,
        "prior_route_local_no_go_theorem_honest": prior_route_local_no_go,
        "observable_dictionary_gate_admissible_now": observable_dictionary_now,
        "primary_residual_lane": "vector_qball_form_factor_exact_charge_current_noether_current_gap" if noether_gap_required else "vector_qball_form_factor_observable_dictionary_gate",
        "secondary_residual_lane": "vector_qball_form_factor_effective_source_formula_absent",
        "reserve_residual_lane": "vector_qball_form_factor_blind_fixed_q_no_go_retained",
        "selected_next_generation_route": next_route,
        "recommended_next_route_or_none": next_step,
        "physical_reject_required": False,
    }
    decision = {"overall_status": f"{branch_class}_declared", "branch_completed": True, "next_required_artifacts": [next_route]}
    evidence = {
        "part1_hits": {"matter_current": part1_current, "interaction": part1_interaction, "total_action": part1_total, "energy_noether": part1_energy_noether},
        "note_hits": {"step_c": note_step_c, "effective_source": note_effective_source, "proxy": note_proxy, "low_order_proxy": note_low_order},
        "carry_over_hits": {"part3a_source_fail": part3a_fail, "part5_source_fail": part5_fail},
        "retained_numeric_state": {
            "phase1_equivalent_max_abs_ratio": float(anchor_eval["phase1_equivalent_row"]["max_abs_ratio"]),
            "phase1_equivalent_F_at_q_theory": float(anchor_eval["phase1_equivalent_row"]["F_at_q_theory"]),
            "phase1_equivalent_alpha_at_q_theory": float(anchor_eval["phase1_equivalent_row"]["alpha_at_q_theory"]),
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    for kind in ("inventory", "audit", "declaration_gate", "numeric_evaluation"):
        write_artifact(kind, payload(STEP_TAG, STEP_NAME, inputs, rows, summary, decision, evidence))

    print(f"[ok] wrote compact artifacts for {STEP_TAG}: {STEM}")


if __name__ == "__main__":
    main()
