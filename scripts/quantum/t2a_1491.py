#!/usr/bin/env python3
"""Generate 8.7.56.1491-.1494 exact charge-current / Noether-current closure audit artifacts.

This branch follows the effective-source-theorem failure in 8.7.56.1487-.1490.
The honest next question is narrower:

- does the current public pack already contain an exact charge-current /
  Noether-current closure that can support `J_eff^0` on the restored exact
  vector branch?

The audit must distinguish three layers clearly:

1. generic U(1) continuity does exist,
2. adopted-U(1) / Q-ball identity statements do exist,
3. but the exact action-level closure needed for the restored vector branch
   may still be absent.
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

SOURCE_GATE = PUBLIC_OUT / "q_8_7_56_1487_1490_effective_source_theorem_declaration_gate_metrics.json"
SOURCE_EVAL = PUBLIC_OUT / "q_8_7_56_1487_1490_effective_source_theorem_numeric_evaluation_metrics.json"
ANCHOR_EVAL = PUBLIC_OUT / "q_8_7_56_1483_1486_ell0_anchor_continuation_numeric_evaluation_metrics.json"
VECTOR_REVIEW_INV = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_review_source_inventory_metrics.json"
ACTION_PRINCIPLE_AUDIT = PUBLIC_OUT / "action_principle_el_derivation_audit.json"
QBALL_CHARGE_MAPPING = PUBLIC_OUT / "mass_origin_qball_charge_mapping_statement_freeze_metrics.json"
QBALL_CHARGE_NORMALIZATION = PUBLIC_OUT / "mass_origin_qball_charge_operator_normalization_audit_metrics.json"

STEP_TAG = "8.7.56.1491-1494"
STEM = build_compact_artifact_stem(STEP_TAG, "charge_current_closure", prefix="q")
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor exact charge-current / Noether-current closure audit"

PRIOR_CLASS = "vector_qball_form_factor_effective_source_theorem_failed_noether_current_gap_retained"
FAIL_CLASS = "vector_qball_form_factor_exact_charge_current_noether_closure_failed_proxy_signed_density_only_retained"
PASS_CLASS = "vector_qball_form_factor_exact_charge_current_noether_closure_derived_observable_dictionary_next"
FAIL_ROUTE = "trial2_numeric_alpha_vector_qball_form_factor_exact_charge_current_noether_gap_closeout_sync"
PASS_ROUTE = "trial2_numeric_alpha_vector_qball_form_factor_observable_dictionary_gate"
NEXT_STEP = "8.7.56.1495"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Fail when one required input path is missing."""
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
    """Convert one absolute path into repo-relative display form when possible."""
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


# 関数: 標準形式の metrics row を生成する。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準形式の payload を生成する。

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


# 関数: compact stem で JSON / CSV を出力する。

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and its rows CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": display_path(paths["json"]), "csv": display_path(paths["csv"])}


# 関数: 真偽値を 0 / 1 に変換する。

def truth(value: bool) -> float:
    """Convert one boolean into 0/1 float form."""
    return 1.0 if value else 0.0


# 関数: `.1491-.1494` を実行する。

def main() -> None:
    """Execute the exact charge-current / Noether-current closure audit."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        CASE_GAMMA_ADVICE,
        PART1,
        PART3A,
        PART5,
        NEXT_STEPS_NOTE,
        SOURCE_GATE,
        SOURCE_EVAL,
        ANCHOR_EVAL,
        VECTOR_REVIEW_INV,
        ACTION_PRINCIPLE_AUDIT,
        QBALL_CHARGE_MAPPING,
        QBALL_CHARGE_NORMALIZATION,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    case_gamma_text = read_text(CASE_GAMMA_ADVICE)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    next_steps_text = read_text(NEXT_STEPS_NOTE)

    source_gate = read_json(SOURCE_GATE)
    source_eval = read_json(SOURCE_EVAL)
    anchor_eval = read_json(ANCHOR_EVAL)
    vector_review_inv = read_json(VECTOR_REVIEW_INV)
    action_principle_audit = read_json(ACTION_PRINCIPLE_AUDIT)
    qball_charge_mapping = read_json(QBALL_CHARGE_MAPPING)
    qball_charge_normalization = read_json(QBALL_CHARGE_NORMALIZATION)

    source_summary = source_gate["summary"]
    source_numeric = source_eval["evidence"]["retained_numeric_state"]
    anchor_summary = anchor_eval["summary"]
    vector_review_hits = vector_review_inv["evidence"]["note_hits"]

    part1_noether = hit(part1_text, "Noether保存則")
    part1_continuity = hit(part1_text, "\\partial_\\mu J^\\mu=0")
    part1_matter_current = hit(part1_text, "J^\\mu_{\\mathrm{matter}}=(\\rho c,\\rho \\mathbf{v})")
    part1_interaction = hit(part1_text, "\\mathcal{L}_{\\mathrm{int}}=g_P\\,P_\\mu J^\\mu_{\\mathrm{matter}}")
    part3a_qball_identity = hit(part3a_text, "Q-ball Noether charge = adopted U(1) charge")
    part5_gap = hit(part5_text, "exact charge-current / Noether-current closure")
    step_c_hit = hit(next_steps_text, "### Step C.")
    proxy_hit = hit(next_steps_text, "|f_0|^2 - |f_L|^2")

    generic_u1_continuity_surface_available = bool(
        part1_noether
        and part1_continuity
        and action_principle_audit.get("equations", {}).get("continuity")
    )
    qball_charge_mapping_statement_available = (
        qball_charge_mapping["summary"].get("u1_charge_quantization_to_qball_charge_mapping") == "available"
    )
    direct_qball_u1_identity_required = bool(
        qball_charge_normalization["summary"].get("direct_qball_u1_identity_required", False)
    )
    proxy_signed_density_available = bool(
        vector_review_hits.get("signed_density_line")
        and vector_review_hits.get("vector_noether_charge_density")
        and step_c_hit
        and proxy_hit
    )
    explicit_qball_background_expansion_available = bool(
        source_summary.get("explicit_qball_background_expansion_available", False)
    )
    explicit_effective_source_formula_available = bool(
        source_summary.get("explicit_effective_source_formula_available", False)
    )
    exact_source_theorem_derived = bool(source_summary.get("exact_source_theorem_derived", False))

    exact_charge_current_noether_closure_available = bool(
        generic_u1_continuity_surface_available
        and qball_charge_mapping_statement_available
        and direct_qball_u1_identity_required
        and explicit_qball_background_expansion_available
        and explicit_effective_source_formula_available
        and exact_source_theorem_derived
    )
    proxy_signed_density_only = bool(proxy_signed_density_available and not exact_charge_current_noether_closure_available)
    closure_audit_ready = bool(
        source_summary.get("effective_source_theorem_attempt_ready", False)
        and part1_matter_current
        and part1_interaction
        and part3a_qball_identity
        and part5_gap
    )
    noether_current_gap_retained = bool(closure_audit_ready and not exact_charge_current_noether_closure_available)
    observable_dictionary_gate_admissible_now = bool(exact_charge_current_noether_closure_available)

    branch_class = PASS_CLASS if exact_charge_current_noether_closure_available else FAIL_CLASS
    next_route = PASS_ROUTE if exact_charge_current_noether_closure_available else FAIL_ROUTE

    rows = [
        row(
            "closure_audit_ready",
            "pass" if closure_audit_ready else "reject",
            "exact charge-current / Noether-current closure audit ready",
            truth(closure_audit_ready),
            "The closure audit is only honest after the effective-source-theorem attempt has localized the missing piece.",
        ),
        row(
            "generic_u1_continuity_surface_available",
            "pass" if generic_u1_continuity_surface_available else "reject",
            "generic U(1) continuity surface available",
            truth(generic_u1_continuity_surface_available),
            "Part I and the older action-principle audit still expose generic continuity, but that alone does not yet identify J_eff^0 on the restored vector branch.",
        ),
        row(
            "qball_charge_mapping_statement_available",
            "pass" if qball_charge_mapping_statement_available else "reject",
            "Q-ball / adopted-U(1) charge mapping statement available",
            truth(qball_charge_mapping_statement_available),
            "The public pack already freezes the adopted-U(1) charge mapping for the Q-ball family.",
        ),
        row(
            "direct_qball_u1_identity_required",
            "pass" if direct_qball_u1_identity_required else "reject",
            "direct Q-ball / adopted-U(1) identity required",
            truth(direct_qball_u1_identity_required),
            "The charge-operator normalization audit already froze the direct identity, so any extra multiplicative freedom is not available under the current pack.",
        ),
        row(
            "proxy_signed_density_available",
            "pass" if proxy_signed_density_available else "reject",
            "proxy signed density available",
            truth(proxy_signed_density_available),
            "The vector route still has the proxy signed-density readout |f0|^2 - |fL|^2 available as a computation-side hint.",
        ),
        row(
            "explicit_qball_background_expansion_available",
            "pass" if explicit_qball_background_expansion_available else "reject",
            "explicit Q-ball background expansion available",
            truth(explicit_qball_background_expansion_available),
            "An exact closure would need a background expansion that is still absent from the current public pack.",
        ),
        row(
            "explicit_effective_source_formula_available",
            "pass" if explicit_effective_source_formula_available else "reject",
            "explicit effective source formula available",
            truth(explicit_effective_source_formula_available),
            "The restored vector branch still lacks an explicit a_mu J_eff^mu[P^Qball] formula.",
        ),
        row(
            "exact_charge_current_noether_closure_available",
            "pass" if exact_charge_current_noether_closure_available else "reject",
            "exact charge-current / Noether-current closure available",
            truth(exact_charge_current_noether_closure_available),
            "Generic continuity plus adopted-U(1) identity only suffice when they actually close the restored vector branch into J_eff^0; that closure is still missing here.",
        ),
        row(
            "proxy_signed_density_only",
            "pass" if proxy_signed_density_only else "reject",
            "proxy signed density remains proxy-only",
            truth(proxy_signed_density_only),
            "The current pack still supports the signed density only as a proxy hint rather than an exact action-level charge-current theorem.",
        ),
        row(
            "observable_dictionary_gate_admissible_now",
            "pass" if observable_dictionary_gate_admissible_now else "reject",
            "observable dictionary gate admissible now",
            truth(observable_dictionary_gate_admissible_now),
            "Observable-dictionary work remains downstream until exact charge-current / Noether-current closure is derived.",
        ),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem_note": display_path(CURRENT_PROBLEM),
            "current_status_note": display_path(CURRENT_STATUS),
            "unified_roadmap_note": display_path(UNIFIED_ROADMAP),
            "case_gamma_advice_note": display_path(CASE_GAMMA_ADVICE),
            "part1": display_path(PART1),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
            "next_steps_note": display_path(NEXT_STEPS_NOTE),
        },
        "prior_metrics": {
            "source_gate": display_path(SOURCE_GATE),
            "source_eval": display_path(SOURCE_EVAL),
            "anchor_eval": display_path(ANCHOR_EVAL),
            "vector_review_inventory": display_path(VECTOR_REVIEW_INV),
            "action_principle_audit": display_path(ACTION_PRINCIPLE_AUDIT),
            "qball_charge_mapping": display_path(QBALL_CHARGE_MAPPING),
            "qball_charge_normalization": display_path(QBALL_CHARGE_NORMALIZATION),
        },
        "constants": {
            "next_route_name": next_route,
            "next_route": NEXT_STEP,
        },
    }
    summary = {
        "trial2_numeric_alpha_problem_classification": branch_class,
        "prior_problem_classification": PRIOR_CLASS,
        "closure_audit_ready": closure_audit_ready,
        "generic_u1_continuity_surface_available": generic_u1_continuity_surface_available,
        "qball_charge_mapping_statement_available": qball_charge_mapping_statement_available,
        "direct_qball_u1_identity_required": direct_qball_u1_identity_required,
        "proxy_signed_density_available": proxy_signed_density_available,
        "proxy_signed_density_only": proxy_signed_density_only,
        "explicit_qball_background_expansion_available": explicit_qball_background_expansion_available,
        "explicit_effective_source_formula_available": explicit_effective_source_formula_available,
        "exact_source_theorem_derived": exact_source_theorem_derived,
        "exact_charge_current_noether_closure_available": exact_charge_current_noether_closure_available,
        "noether_current_gap_retained": noether_current_gap_retained,
        "observable_dictionary_gate_admissible_now": observable_dictionary_gate_admissible_now,
        "primary_residual_lane": "vector_qball_form_factor_exact_charge_current_noether_closure_missing",
        "secondary_residual_lane": "vector_qball_form_factor_effective_source_formula_absent",
        "reserve_residual_lane": "vector_qball_form_factor_proxy_signed_density_only",
        "selected_next_generation_route": next_route,
        "recommended_next_route_or_none": NEXT_STEP,
        "physical_reject_required": False,
    }
    decision = {
        "overall_status": f"{branch_class}_declared",
        "branch_completed": True,
        "next_required_artifacts": [next_route],
    }
    evidence = {
        "part_hits": {
            "part1_noether": part1_noether,
            "part1_continuity": part1_continuity,
            "part1_matter_current": part1_matter_current,
            "part1_interaction": part1_interaction,
            "part3a_qball_identity": part3a_qball_identity,
            "part5_gap": part5_gap,
        },
        "note_hits": {
            "step_c": step_c_hit,
            "proxy": proxy_hit,
            "vector_review_signed_density": vector_review_hits.get("signed_density_line"),
            "vector_review_charge_density_header": vector_review_hits.get("vector_noether_charge_density"),
        },
        "carry_over": {
            "action_principle_continuity": action_principle_audit.get("equations", {}).get("continuity"),
            "charge_mapping_summary": qball_charge_mapping["summary"],
            "charge_normalization_summary": qball_charge_normalization["summary"],
            "source_theorem_summary": source_summary,
        },
        "retained_numeric_state": {
            "phase1_equivalent_max_abs_ratio": float(source_numeric["phase1_equivalent_max_abs_ratio"]),
            "phase1_equivalent_F_at_q_theory": float(source_numeric["phase1_equivalent_F_at_q_theory"]),
            "phase1_equivalent_alpha_at_q_theory": float(source_numeric["phase1_equivalent_alpha_at_q_theory"]),
            "anchor_preserving_continuation_restored": bool(anchor_summary.get("anchor_preserving_continuation_restored", False)),
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    for kind in ("inventory", "audit", "declaration_gate", "numeric_evaluation"):
        write_artifact(kind, payload(STEP_TAG, STEP_NAME, inputs, rows, summary, decision, evidence))

    print(f"[ok] wrote compact artifacts for {STEP_TAG}: {STEM}")


if __name__ == "__main__":
    main()
