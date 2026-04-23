#!/usr/bin/env python3
"""Generate anchor-drift / branch-continuation audit artifacts for 8.7.56.1483-.1486.

This branch revisits the 8.7.56.1479-.1482 reinjection result carefully.
The previous grid proved that the corrected exact solver did not preserve a
localized mode-1 branch inside the sampled lambda window, and therefore the
roadmap moved to an anchor-drift / continuation audit before any source
theorem.

The key missing check is whether the reinjected exact solver can continue from
the scalar anchor amplitude to the retained Phase-1 exact pilot amplitude
(`amp_L = 1.25`) once the lambda grid is extended far enough. If that
continuation exists and remains localized, then the electron anchor is not
actually lost; only the blind fixed-q vector observable remains a no-go.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


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
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

NEXT_ACTION_NOTE = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_action_20260327.md")

BOOTSTRAP_EVAL = PUBLIC_OUT / "q_8_7_56_1475_1478_ell0_family_bootstrap_numeric_evaluation_metrics.json"
REINJECTION_AUDIT = PUBLIC_OUT / "q_8_7_56_1479_1482_ell0_exact_solver_reinjection_audit_metrics.json"
REINJECTION_GATE = PUBLIC_OUT / "q_8_7_56_1479_1482_ell0_exact_solver_reinjection_declaration_gate_metrics.json"
REINJECTION_EVAL = PUBLIC_OUT / "q_8_7_56_1479_1482_ell0_exact_solver_reinjection_numeric_evaluation_metrics.json"
PHASE1_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase1_exact_coupled_l0_solver_"
    "numeric_evaluation_metrics.json"
)
CASE_GAMMA_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_perturbative_fl_driven_ode_"
    "diagnostic_reopen_review_numeric_evaluation_metrics.json"
)

EXACT_REINJECTION_BRANCH = ROOT / "scripts" / "quantum" / "t2a_1479.py"

STEP_TAG = "8.7.56.1483-1486"
STEM = build_compact_artifact_stem(STEP_TAG, "ell0_anchor_continuation", prefix="q")
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor corrected exact-action-level ell=0 "
    "anchor-drift / branch-continuation audit"
)

PRIOR_CLASS = "vector_qball_form_factor_corrected_exact_solver_reinjection_mode1_anchor_lost_scalar_like_branch_only"
BRANCH_CLASS = "vector_qball_form_factor_corrected_anchor_preserving_continuation_restored_blind_vector_no_go_retained"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_effective_source_theorem_attempt"
NEXT_ROUTE = "8.7.56.1487"

TAIL_RATIO_THRESHOLD = 0.25
NONTRIVIAL_RATIO_THRESHOLD = 0.10
ANCHOR_RELATIVE_DRIFT_LIMIT = 0.02
ALPHA_TARGET = 1.0 / 137.035999084


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: fail immediately when one required path is missing.

def require(path: Path) -> None:
    """Fail immediately when one required path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: read one UTF-8 text file.

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# Function: read one UTF-8 JSON file.

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: convert one absolute path into repo-relative display text when possible.

def display_path(path: Path) -> str:
    """Convert one absolute path into repo-relative display text when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: return the first matching source line for one substring pattern.

def hit(text: str, pattern: str) -> dict | None:
    """Return the first matching source line for one substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: build one standard metrics row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: build one standard payload object.

def payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Build one standard payload object."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# Function: write one JSON payload and its rows CSV with Windows-safe paths.

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and its rows CSV with Windows-safe paths."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": display_path(paths["json"]), "csv": display_path(paths["csv"])}


# Function: dynamically load one local Python module.

def load_module(path: Path, module_name: str):
    """Dynamically load one local Python module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Function: build one sampled lambda path that covers the Phase-1 equivalent anchor.

def build_lambda_scale_path(phase1_lambda_scale: float) -> list[float]:
    """Build one sampled lambda path that covers the Phase-1 equivalent anchor."""
    ceiling = max(512.0, float(phase1_lambda_scale))
    grid = [0.0]
    grid.extend(float(value) for value in np.geomspace(0.05, ceiling, 40))
    grid.extend([64.0, 128.0, float(phase1_lambda_scale), 256.0, 384.0, 512.0])
    unique = sorted({float(value) for value in grid if float(value) >= 0.0})
    return unique


# Function: solve one anchor-preserving continuation sample along the corrected exact branch.

def solve_anchor_path_sample(exact_branch, pivot, numerical, beta: float, amp0: float, df_l0: float, lambda_scale: float) -> dict:
    """Solve one anchor-preserving continuation sample along the corrected exact branch."""
    amp_l = float(df_l0) * float(lambda_scale)
    solved = exact_branch.solve_reinjected_exact_profile(pivot, numerical, float(beta), float(amp0), float(amp_l))
    return {
        "lambda_scale": float(lambda_scale),
        "amp_l": float(amp_l),
        "localized": bool(
            solved["success"]
            and solved["tail_to_input_ratio"] is not None
            and float(solved["tail_to_input_ratio"]) <= float(TAIL_RATIO_THRESHOLD)
        ),
        "nontrivial": bool(float(solved["max_abs_ratio"]) >= float(NONTRIVIAL_RATIO_THRESHOLD)),
        **solved,
    }


# Function: return the first sampled row satisfying one predicate.

def first_match(rows: list[dict], predicate) -> dict | None:
    """Return the first sampled row satisfying one predicate."""
    for candidate in rows:
        if predicate(candidate):
            return candidate

    return None


# Function: summarize one anchored continuation path.

def summarize_anchor_path(rows: list[dict], phase1_lambda_scale: float) -> dict:
    """Summarize one anchored continuation path."""
    localized_rows = [candidate for candidate in rows if candidate["localized"]]
    if not localized_rows:
        raise SystemExit("[fail] no localized samples found on anchored continuation path")

    best_alpha = min(localized_rows, key=lambda candidate: float(candidate["alpha_relerr_vs_target"]))
    phase1_index = min(
        range(len(rows)),
        key=lambda index: abs(float(rows[index]["lambda_scale"]) - float(phase1_lambda_scale)),
    )
    phase1_row = rows[phase1_index]
    localized_until_phase1 = all(
        candidate["localized"] for candidate in rows if float(candidate["lambda_scale"]) <= float(phase1_lambda_scale)
    )
    localized_past_phase1 = all(candidate["localized"] for candidate in rows)
    first_nontrivial = first_match(
        rows,
        lambda candidate: candidate["localized"] and candidate["nontrivial"],
    )
    blind_vector_no_go_retained = bool(
        float(phase1_row["F_at_q_theory"]) < 0.0
        and float(phase1_row["alpha_relerr_vs_target"]) > 0.5
    )
    return {
        "sample_count": int(len(rows)),
        "localized_sample_count": int(len(localized_rows)),
        "phase1_equivalent_row": phase1_row,
        "localized_until_phase1_equivalent": localized_until_phase1,
        "localized_across_full_sampled_path": localized_past_phase1,
        "first_nontrivial_localized_row_or_none": first_nontrivial,
        "best_alpha_row": best_alpha,
        "blind_vector_no_go_retained": blind_vector_no_go_retained,
        "best_alpha_prefers_lambda_zero": bool(abs(float(best_alpha["lambda_scale"])) <= 1.0e-12),
        "max_localized_ratio_on_path": float(max(float(candidate["max_abs_ratio"]) for candidate in localized_rows)),
        "min_localized_tail_ratio_on_path": float(min(float(candidate["tail_to_input_ratio"]) for candidate in localized_rows)),
    }


# Function: return one numeric truth value.

def truth(flag: bool) -> float:
    """Return one numeric truth value."""
    return 1.0 if flag else 0.0


# Function: orchestrate the branch computation and artifact generation.

def main() -> None:
    """Orchestrate the branch computation and artifact generation."""
    required_paths = [
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        CASE_GAMMA_ADVICE,
        PART1,
        PART5,
        NEXT_ACTION_NOTE,
        BOOTSTRAP_EVAL,
        REINJECTION_AUDIT,
        REINJECTION_GATE,
        REINJECTION_EVAL,
        PHASE1_EVAL,
        CASE_GAMMA_EVAL,
        EXACT_REINJECTION_BRANCH,
    ]
    for path in required_paths:
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    case_gamma_text = read_text(CASE_GAMMA_ADVICE)
    part1_text = read_text(PART1)
    part5_text = read_text(PART5)
    next_action_text = read_text(NEXT_ACTION_NOTE)

    bootstrap_eval = read_json(BOOTSTRAP_EVAL)
    reinjection_audit = read_json(REINJECTION_AUDIT)
    reinjection_gate = read_json(REINJECTION_GATE)
    reinjection_eval = read_json(REINJECTION_EVAL)
    phase1_eval = read_json(PHASE1_EVAL)
    case_gamma_eval = read_json(CASE_GAMMA_EVAL)

    exact_branch = load_module(EXACT_REINJECTION_BRANCH, "t2a_1479_reuse")
    pivot = exact_branch.load_module(exact_branch.PIVOT_BRANCH, "pivot_branch_reuse")
    numerical = exact_branch.load_module(exact_branch.NUMERICAL_BRANCH, "numerical_branch_reuse")

    mode1_bootstrap = bootstrap_eval["evidence"]["bootstrap_mode_rows"][0]
    mode1_reinjection = reinjection_eval["evidence"]["mode1_row"]
    phase1_best = phase1_eval["summary"]["phase1_best_alpha_candidate"]

    phase1_equiv_amp_scale = float(phase1_best["amp0"] / mode1_bootstrap["central_amplitude"])
    phase1_equiv_lambda_scale = float(phase1_best["amp_l"] / mode1_bootstrap["df_l0"])
    min_localized_amp0 = float(mode1_bootstrap["central_amplitude"]) * float(mode1_reinjection["min_localized_amp_scale_or_none"])
    anchor_amp0_relative_drift = float(abs(min_localized_amp0 - float(phase1_best["amp0"])) / float(phase1_best["amp0"]))
    anchor_amp0_preserved_within_two_percent = bool(anchor_amp0_relative_drift <= float(ANCHOR_RELATIVE_DRIFT_LIMIT))

    lambda_scale_path = build_lambda_scale_path(phase1_equiv_lambda_scale)
    anchor_path_rows = [
        solve_anchor_path_sample(
            exact_branch,
            pivot,
            numerical,
            float(mode1_bootstrap["beta"]),
            float(phase1_best["amp0"]),
            float(mode1_bootstrap["df_l0"]),
            float(lambda_scale),
        )
        for lambda_scale in lambda_scale_path
    ]
    anchor_path_summary = summarize_anchor_path(anchor_path_rows, phase1_equiv_lambda_scale)
    phase1_equivalent_row = anchor_path_summary["phase1_equivalent_row"]
    first_nontrivial = anchor_path_summary["first_nontrivial_localized_row_or_none"]
    best_alpha_row = anchor_path_summary["best_alpha_row"]

    branch_continuation_success = bool(
        anchor_amp0_preserved_within_two_percent
        and anchor_path_summary["localized_until_phase1_equivalent"]
        and phase1_equivalent_row["localized"]
        and phase1_equivalent_row["nontrivial"]
    )
    source_theorem_attempt_admissible_now = bool(branch_continuation_success)
    blind_vector_no_go_retained = bool(anchor_path_summary["blind_vector_no_go_retained"])
    observable_dictionary_gate_admissible_now = False

    inventory_rows = [
        row(
            "mode1_min_localized_amp_scale",
            "watch",
            "mode 1 minimum localized amplitude scale from reinjection",
            float(mode1_reinjection["min_localized_amp_scale_or_none"]),
            "The continuation audit starts from the first localized mode-1 exact point found in 8.7.56.1479-.1482.",
        ),
        row(
            "phase1_equivalent_amp_scale",
            "watch",
            "Phase 1 exact pilot expressed as corrected bootstrap amplitude scale",
            phase1_equiv_amp_scale,
            "This converts the retained Phase-1 exact anchor back into the corrected bootstrap coordinates.",
        ),
        row(
            "phase1_equivalent_lambda_scale",
            "watch",
            "Phase 1 exact pilot expressed as corrected bootstrap lambda scale",
            phase1_equiv_lambda_scale,
            "The reinjection grid in 8.7.56.1479-.1482 stopped at 64, so the continuation audit must extend into this larger lambda range.",
        ),
        row(
            "anchor_amp0_relative_drift",
            "pass" if anchor_amp0_preserved_within_two_percent else "reject",
            "relative drift between first localized mode-1 amp0 and retained Phase-1 anchor",
            anchor_amp0_relative_drift,
            "If this drift is tiny, the apparent anchor loss was a grid-window artifact rather than a true amplitude-anchor failure.",
        ),
    ]

    inventory_payload = payload(
        "8.7.56.1483",
        f"{STEP_NAME} inventory",
        {
            "bootstrap_eval_json": display_path(BOOTSTRAP_EVAL),
            "reinjection_audit_json": display_path(REINJECTION_AUDIT),
            "reinjection_gate_json": display_path(REINJECTION_GATE),
            "reinjection_eval_json": display_path(REINJECTION_EVAL),
            "phase1_eval_json": display_path(PHASE1_EVAL),
            "case_gamma_eval_json": display_path(CASE_GAMMA_EVAL),
            "exact_reinjection_branch": display_path(EXACT_REINJECTION_BRANCH),
            "part1": display_path(PART1),
            "part5": display_path(PART5),
            "next_action_note": display_path(NEXT_ACTION_NOTE),
        },
        inventory_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "phase1_equivalent_amp_scale": phase1_equiv_amp_scale,
            "phase1_equivalent_lambda_scale": phase1_equiv_lambda_scale,
            "anchor_amp0_relative_drift": anchor_amp0_relative_drift,
            "anchor_amp0_preserved_within_two_percent": anchor_amp0_preserved_within_two_percent,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_corrected_anchor_drift_branch_continuation_inventory_completed",
            "branch_completed": False,
            "next_required_artifacts": [f"{STEM}_audit_metrics.json"],
        },
        {
            "status_next_step_hit": hit(status_text, "8.7.56.1483"),
            "roadmap_next_step_hit": hit(roadmap_text, "8.7.56.1483-.1486"),
            "current_problem_hit": hit(current_problem_text, "anchor-drift / branch-continuation audit"),
            "current_status_hit": hit(current_status_text, "anchor-drift / branch-continuation audit"),
            "unified_roadmap_hit": hit(unified_roadmap_text, "anchor-drift / branch-continuation audit"),
            "next_action_note_hit": hit(next_action_text, "wrong-branch suspicion"),
            "part1_free_backbone_hit": hit(part1_text, "post-photon"),
            "part5_phase1_hit": hit(part5_text, "8.7.56.1479-.1482"),
        },
    )
    inventory_paths = write_artifact("inventory", inventory_payload)

    audit_rows = [
        row(
            "anchor_amp0_preserved_within_two_percent",
            "pass" if anchor_amp0_preserved_within_two_percent else "reject",
            "first localized mode-1 amp0 stays within 2% of the retained Phase-1 anchor",
            anchor_amp0_relative_drift,
            "This checks whether the apparent anchor loss was merely caused by the previous reinjection coordinate choice.",
        ),
        row(
            "localized_until_phase1_equivalent",
            "pass" if anchor_path_summary["localized_until_phase1_equivalent"] else "reject",
            "sampled anchor-preserving continuation remains localized until the Phase-1 equivalent lambda",
            truth(anchor_path_summary["localized_until_phase1_equivalent"]),
            "The exact branch must stay localized across the sampled path, otherwise there is no honest continuation from the scalar anchor to the retained vector anchor.",
        ),
        row(
            "phase1_equivalent_nontrivial",
            "pass" if phase1_equivalent_row["nontrivial"] else "reject",
            "Phase-1 equivalent exact branch is nontrivial in max|fL/f0|",
            float(phase1_equivalent_row["max_abs_ratio"]),
            "A restored continuation only matters if the phase1-equivalent point truly carries vector weight rather than collapsing back to the scalar-like regime.",
        ),
        row(
            "fixed_q_blind_vector_no_go_retained",
            "reject" if blind_vector_no_go_retained else "pass",
            "blind fixed-q vector observable remains a no-go on the restored anchor path",
            float(phase1_equivalent_row["alpha_relerr_vs_target"]),
            "Restoring branch continuity does not by itself repair the blind vector observable if F(q_theory) stays negative and far from the target.",
        ),
        row(
            "source_theorem_attempt_admissible_now",
            "pass" if source_theorem_attempt_admissible_now else "reject",
            "effective source theorem attempt admissible now",
            truth(source_theorem_attempt_admissible_now),
            "Once the anchor-preserving exact branch is restored, the next honest reopen surface is the action-level source theorem rather than further branch-continuation bookkeeping.",
        ),
    ]

    audit_payload = payload(
        "8.7.56.1484",
        f"{STEP_NAME} audit",
        {
            "inventory_json": inventory_paths["json"],
            "reinjection_eval_json": display_path(REINJECTION_EVAL),
            "phase1_eval_json": display_path(PHASE1_EVAL),
            "case_gamma_eval_json": display_path(CASE_GAMMA_EVAL),
        },
        audit_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "phase1_equivalent_amp_scale": phase1_equiv_amp_scale,
            "phase1_equivalent_lambda_scale": phase1_equiv_lambda_scale,
            "anchor_amp0_relative_drift": anchor_amp0_relative_drift,
            "anchor_amp0_preserved_within_two_percent": anchor_amp0_preserved_within_two_percent,
            "localized_until_phase1_equivalent": anchor_path_summary["localized_until_phase1_equivalent"],
            "localized_across_full_sampled_path": anchor_path_summary["localized_across_full_sampled_path"],
            "phase1_equivalent_row": phase1_equivalent_row,
            "first_nontrivial_localized_row_or_none": first_nontrivial,
            "best_alpha_row": best_alpha_row,
            "blind_vector_no_go_retained": blind_vector_no_go_retained,
            "source_theorem_attempt_admissible_now": source_theorem_attempt_admissible_now,
            "observable_dictionary_gate_admissible_now": observable_dictionary_gate_admissible_now,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_corrected_anchor_drift_branch_continuation_audited",
            "branch_completed": False,
            "next_required_artifacts": [f"{STEM}_declaration_gate_metrics.json"],
        },
        {
            "anchor_path_rows": anchor_path_rows,
            "reinjection_summary": reinjection_eval["summary"],
            "phase1_summary": phase1_eval["summary"],
            "case_gamma_summary": case_gamma_eval["summary"],
        },
    )
    audit_paths = write_artifact("audit", audit_payload)

    gate_rows = [
        row(
            "anchor_preserving_continuation_restored",
            "pass" if branch_continuation_success else "reject",
            "anchor-preserving corrected exact branch is restored",
            truth(branch_continuation_success),
            "Success requires negligible anchor drift, localization through the sampled path, and a nontrivial phase1-equivalent vector weight.",
        ),
        row(
            "advance_to_effective_source_theorem_now",
            "pass" if source_theorem_attempt_admissible_now else "reject",
            "advance directly to effective source theorem now",
            truth(source_theorem_attempt_admissible_now),
            "Once branch continuity is restored, the next honest reopen surface is the action-level source theorem.",
        ),
        row(
            "advance_to_observable_dictionary_now",
            "pass" if observable_dictionary_gate_admissible_now else "reject",
            "advance directly to observable dictionary now",
            truth(observable_dictionary_gate_admissible_now),
            "Observable-dictionary work remains downstream until the source theorem has been checked on the restored exact branch.",
        ),
    ]

    gate_payload = payload(
        "8.7.56.1485",
        f"{STEP_NAME} declaration gate",
        {
            "inventory_json": inventory_paths["json"],
            "audit_json": audit_paths["json"],
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
        },
        gate_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "anchor_preserving_continuation_restored": branch_continuation_success,
            "blind_vector_no_go_retained": blind_vector_no_go_retained,
            "source_theorem_attempt_admissible_now": source_theorem_attempt_admissible_now,
            "observable_dictionary_gate_admissible_now": observable_dictionary_gate_admissible_now,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "physical_reject_required": False,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_corrected_anchor_drift_branch_continuation_completed",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "status_next_step_hit": hit(status_text, "8.7.56.1483"),
            "roadmap_next_step_hit": hit(roadmap_text, "8.7.56.1487-.1490"),
            "part5_reinjection_hit": hit(part5_text, "8.7.56.1479-.1482"),
        },
    )
    gate_paths = write_artifact("declaration_gate", gate_payload)

    numeric_rows = [
        row(
            "phase1_equivalent_tail_to_input_ratio",
            "watch",
            "Phase-1 equivalent tail-to-input ratio",
            float(phase1_equivalent_row["tail_to_input_ratio"]),
            "The restored branch stays localized at the retained vector-anchor point under the same localization criterion used in prior branches.",
        ),
        row(
            "phase1_equivalent_max_abs_ratio",
            "watch",
            "Phase-1 equivalent max|fL/f0|",
            float(phase1_equivalent_row["max_abs_ratio"]),
            "This shows that the restored anchor branch is no longer scalar-like at the retained Phase-1 point.",
        ),
        row(
            "phase1_equivalent_alpha_relerr_vs_target",
            "watch",
            "Phase-1 equivalent alpha relative error vs target",
            float(phase1_equivalent_row["alpha_relerr_vs_target"]),
            "Even with restored branch continuity, the blind vector observable remains far from the target at fixed q_theory.",
        ),
        row(
            "best_alpha_on_anchor_path_prefers_lambda_zero",
            "watch",
            "best alpha on anchored continuation path prefers lambda = 0",
            truth(anchor_path_summary["best_alpha_prefers_lambda_zero"]),
            "The best blind-alpha value still sits at the scalar-like end of the restored branch rather than at the vector-rich Phase-1 anchor.",
        ),
    ]

    numeric_payload = payload(
        "8.7.56.1486",
        f"{STEP_NAME} numeric evaluation",
        {
            "inventory_json": inventory_paths["json"],
            "audit_json": audit_paths["json"],
            "declaration_gate_json": gate_paths["json"],
            "reinjection_eval_json": display_path(REINJECTION_EVAL),
            "phase1_eval_json": display_path(PHASE1_EVAL),
            "case_gamma_eval_json": display_path(CASE_GAMMA_EVAL),
        },
        numeric_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "beta_1_scalar": float(mode1_bootstrap["beta"]),
            "q_theory_over_m0_scalar": float(mode1_bootstrap["q_theory_over_m0"]),
            "phase1_equivalent_amp_scale": phase1_equiv_amp_scale,
            "phase1_equivalent_lambda_scale": phase1_equiv_lambda_scale,
            "anchor_amp0_relative_drift": anchor_amp0_relative_drift,
            "anchor_preserving_continuation_restored": branch_continuation_success,
            "localized_until_phase1_equivalent": anchor_path_summary["localized_until_phase1_equivalent"],
            "localized_across_full_sampled_path": anchor_path_summary["localized_across_full_sampled_path"],
            "phase1_equivalent_row": phase1_equivalent_row,
            "first_nontrivial_localized_row_or_none": first_nontrivial,
            "best_alpha_row": best_alpha_row,
            "blind_vector_no_go_retained": blind_vector_no_go_retained,
            "source_theorem_attempt_admissible_now": source_theorem_attempt_admissible_now,
            "observable_dictionary_gate_admissible_now": observable_dictionary_gate_admissible_now,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "physical_reject_required": False,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_corrected_anchor_drift_branch_continuation_numeric_evaluation_completed",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "anchor_path_rows": anchor_path_rows,
            "phase1_equivalent_row": phase1_equivalent_row,
            "best_alpha_row": best_alpha_row,
        },
    )
    numeric_paths = write_artifact("numeric_evaluation", numeric_payload)

    print(
        json.dumps(
            {
                "status": "ok",
                "step": STEP_TAG,
                "inventory_json": inventory_paths["json"],
                "audit_json": audit_paths["json"],
                "declaration_gate_json": gate_paths["json"],
                "numeric_evaluation_json": numeric_paths["json"],
                "phase1_equivalent_amp_scale": phase1_equiv_amp_scale,
                "phase1_equivalent_lambda_scale": phase1_equiv_lambda_scale,
                "anchor_amp0_relative_drift": anchor_amp0_relative_drift,
                "phase1_equivalent_max_abs_ratio": phase1_equivalent_row["max_abs_ratio"],
                "phase1_equivalent_alpha_relerr_vs_target": phase1_equivalent_row["alpha_relerr_vs_target"],
                "source_theorem_attempt_admissible_now": source_theorem_attempt_admissible_now,
                "selected_next_generation_route": NEXT_ROUTE_NAME,
                "recommended_next_route_or_none": NEXT_ROUTE,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
