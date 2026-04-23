#!/usr/bin/env python3
"""Generate 8.7.56.735-.738 Trial-2 numeric alpha chi/proxy numeric-value artifacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"

ADVICE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_electron_identification.md")
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

PRIOR_INVENTORY = OUT / "mass_origin_v2_trial2_numeric_alpha_electron_identification_source_inventory_metrics.json"
PRIOR_AUDIT = OUT / "mass_origin_v2_trial2_numeric_alpha_electron_identification_audit_metrics.json"
PRIOR_GATE = OUT / "mass_origin_v2_trial2_numeric_alpha_electron_identification_declaration_gate_metrics.json"
PRIOR_ROUTE = OUT / "mass_origin_v2_t2_alpha_route_contract_eightieth_refresh_metrics.json"
CHI_PROXY_INVENTORY = OUT / "mass_origin_anchor_normalized_g3w_chi_proxy_inventory_metrics.json"
CHI_STAR_PROXY_SOURCE_INVENTORY = OUT / "mass_origin_chi_star_proxy_source_inventory_metrics.json"
CHI_STAR_PROXY_CLOSURE_RETRY = OUT / "mass_origin_chi_star_proxy_closure_retry_metrics.json"
SAME_SECTOR_PROXY_EQUIVALENCE_AUDIT = OUT / "mass_origin_same_sector_proxy_equivalence_audit_metrics.json"
QBALL_SPIN_ORBIT = OUT / "mass_origin_vector_qball_spin_orbit_mass_ratio_table_metrics.json"

CURRENT_ROUTE = "trial2_numeric_alpha_newton_limit_chi_star_or_same_sector_proxy_numeric_value_identification"
NEXT_ROUTE = "8.7.56.739"
NEXT_BRANCH = "8.7.56.739-.742"
NEXT_RESIDUAL_ROUTE = "trial2_numeric_alpha_newton_limit_same_sector_proxy_equivalence_rule_identification"
NEXT_MISSING_ARTIFACT = "trial2_numeric_alpha_same_sector_proxy_equivalence_rule"


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: stop execution when a required path is missing.

def require(path: Path) -> None:
    """Require an input path to exist before execution continues."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: read a UTF-8 text file.

def read_text(path: Path) -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# Function: read a UTF-8 JSON file.

def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: return a stable display path.

def display_path(path: Path) -> str:
    """Return a stable path relative to the repository root when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: locate the first line containing a substring pattern.

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line hit for the given substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: build a standard metrics row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build a standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: build a standard payload object.

def payload(
    step: str,
    name: str,
    inputs: dict,
    intent: str,
    formulas: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Build a standard metrics payload."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "intent": intent,
        "formulas": formulas,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# Function: write a JSON metrics artifact and the matching CSV rows table.

def write_artifact(stem: str, data: dict) -> None:
    """Write a metrics payload as JSON and CSV."""
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: build a standard inventory target record.

def target_record(file_key: str, path: Path, text: str, pattern: str, note: str) -> dict:
    """Build a standard inventory target record."""
    target_hit = hit(text, pattern)
    return {
        "file_key": file_key,
        "file": display_path(path),
        "pattern": pattern,
        "present": target_hit is not None,
        "note": note,
        "evidence": target_hit,
    }


# Function: normalize the first-close target into the current naming convention.

def normalized_first_route(raw_first_route: str, same_sector_proxy_rule_available: bool) -> str | None:
    """Normalize the historical first-close target into the current route naming."""
    if raw_first_route == "same_sector_equivalence_rule" and not same_sector_proxy_rule_available:
        return "same_sector_proxy_equivalence_rule"

    return raw_first_route or None


# Function: execute the chi/proxy numeric-value residual branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha chi/proxy numeric-value residual branch."""
    for path in (
        ADVICE,
        PART1,
        PART3A,
        PART5,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIOR_INVENTORY,
        PRIOR_AUDIT,
        PRIOR_GATE,
        PRIOR_ROUTE,
        CHI_PROXY_INVENTORY,
        CHI_STAR_PROXY_SOURCE_INVENTORY,
        CHI_STAR_PROXY_CLOSURE_RETRY,
        SAME_SECTOR_PROXY_EQUIVALENCE_AUDIT,
        QBALL_SPIN_ORBIT,
    ):
        require(path)

    advice_text = read_text(ADVICE)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    prior_inventory = read_json(PRIOR_INVENTORY)
    prior_audit = read_json(PRIOR_AUDIT)
    prior_gate = read_json(PRIOR_GATE)
    prior_route = read_json(PRIOR_ROUTE)
    chi_proxy_inventory = read_json(CHI_PROXY_INVENTORY)
    chi_star_proxy_source_inventory = read_json(CHI_STAR_PROXY_SOURCE_INVENTORY)
    chi_star_proxy_closure_retry = read_json(CHI_STAR_PROXY_CLOSURE_RETRY)
    same_sector_proxy_equivalence_audit = read_json(SAME_SECTOR_PROXY_EQUIVALENCE_AUDIT)
    qball_spin_orbit = read_json(QBALL_SPIN_ORBIT)

    prior_inventory_summary = prior_inventory["summary"]
    prior_audit_summary = prior_audit["summary"]
    prior_gate_summary = prior_gate["summary"]
    prior_route_summary = prior_route["summary"]
    chi_proxy_summary = chi_proxy_inventory["summary"]
    chi_star_proxy_source_summary = chi_star_proxy_source_inventory["summary"]
    chi_star_proxy_closure_summary = chi_star_proxy_closure_retry["summary"]
    same_sector_proxy_equivalence_summary = same_sector_proxy_equivalence_audit["summary"]
    qball_spin_orbit_summary = qball_spin_orbit["summary"]

    electron_identification_pivot_retained = bool(prior_gate_summary["electron_identification_pivot_adopted"]) and bool(
        prior_route_summary["electron_identification_pivot_retained"]
    )
    absolute_normalization_dictionary_ready = bool(
        prior_gate_summary["trial2_numeric_alpha_absolute_normalization_dictionary_ready"]
    )
    m0_numeric_from_electron_identification_ready = bool(
        prior_audit_summary["m0_numeric_from_electron_identification_ready"]
    )
    chi_proxy_inventory_ready = bool(chi_proxy_summary["chi_proxy_inventory_ready"])
    chi_star_proxy_source_inventory_ready = bool(chi_star_proxy_source_summary["proxy_source_inventory_ready"])
    chi_star_or_same_sector_proxy_numeric_value_available = bool(
        chi_star_proxy_closure_summary["chi_star_or_same_sector_proxy_available"]
    )
    same_sector_proxy_equivalence_rule_available = bool(
        same_sector_proxy_equivalence_summary["same_sector_proxy_rule_available"]
    )
    proxy_route_nonclosure_reason_or_none = chi_star_proxy_closure_summary[
        "proxy_route_retry_nonclosure_reason_or_none"
    ]
    raw_first_route_to_close_or_none = str(chi_star_proxy_source_summary["first_route_to_close_or_none"])
    first_route_to_close_or_none = normalized_first_route(
        raw_first_route_to_close_or_none,
        same_sector_proxy_equivalence_rule_available,
    )
    reference_state_public = qball_spin_orbit["formulas"]["reference_state"] == "M_(1,0,0,0)"
    reference_state_mass_proxy_available = float(qball_spin_orbit_summary["reference_state_mass_proxy"]) > 0.0
    qball_ground_state_proxy_evidence_retained = reference_state_public and reference_state_mass_proxy_available
    declaration_gate_consistent = prior_gate_summary["selected_residual_route"] == CURRENT_ROUTE
    route_contract_consistent = prior_route_summary["selected_next_generation_route"] == CURRENT_ROUTE
    dominant_blocker_is_same_sector_proxy_equivalence_rule_absence = (
        absolute_normalization_dictionary_ready
        and not chi_star_or_same_sector_proxy_numeric_value_available
        and not same_sector_proxy_equivalence_rule_available
        and proxy_route_nonclosure_reason_or_none == "same_sector_equivalence_rule_absent"
        and first_route_to_close_or_none == "same_sector_proxy_equivalence_rule"
    )

    common_inputs = {
        "expert_note_markdown": display_path(ADVICE),
        "part1_markdown": display_path(PART1),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "mass_origin_v2_trial2_numeric_alpha_electron_identification_source_inventory_json": display_path(
            PRIOR_INVENTORY
        ),
        "mass_origin_v2_trial2_numeric_alpha_electron_identification_audit_json": display_path(PRIOR_AUDIT),
        "mass_origin_v2_trial2_numeric_alpha_electron_identification_declaration_gate_json": display_path(
            PRIOR_GATE
        ),
        "mass_origin_v2_t2_alpha_route_contract_eightieth_refresh_json": display_path(PRIOR_ROUTE),
        "mass_origin_anchor_normalized_g3w_chi_proxy_inventory_json": display_path(CHI_PROXY_INVENTORY),
        "mass_origin_chi_star_proxy_source_inventory_json": display_path(CHI_STAR_PROXY_SOURCE_INVENTORY),
        "mass_origin_chi_star_proxy_closure_retry_json": display_path(CHI_STAR_PROXY_CLOSURE_RETRY),
        "mass_origin_same_sector_proxy_equivalence_audit_json": display_path(
            SAME_SECTOR_PROXY_EQUIVALENCE_AUDIT
        ),
        "mass_origin_vector_qball_spin_orbit_mass_ratio_table_json": display_path(QBALL_SPIN_ORBIT),
    }

    inventory_targets = [
        target_record(
            "advice_electron_identification_statement",
            ADVICE,
            advice_text,
            r"M_{(1,0,0,0)} = m_e",
            "The expert note still defines the absolute-normalization dictionary through electron identification.",
        ),
        target_record(
            "part1_electron_identification_statement",
            PART1,
            part1_text,
            r"M_{(1,0,0,0)} = m_e",
            "Part I still carries the absolute-normalization statement promoted by the pivot.",
        ),
        target_record(
            "part1_chi_proxy_open_question",
            PART1,
            part1_text,
            "same-sector proxy value が必要",
            "Part I still states that the remaining numeric route needs a chi_* / same-sector proxy value.",
        ),
        target_record(
            "part3a_numeric_value_blocker",
            PART3A,
            part3a_text,
            "chi_star_or_same_sector_proxy",
            "Part III-A still names the missing chi/proxy numeric-value family as the current numeric blocker.",
        ),
        target_record(
            "part5_numeric_value_branch",
            PART5,
            part5_text,
            "8.7.56.735-.738",
            "Part V still names the current official chi/proxy numeric-value residual branch.",
        ),
        target_record(
            "status_numeric_value_branch",
            STATUS,
            status_text,
            "chi-star-or-same-sector-proxy numeric-value residual branch",
            "STATUS still names the current official branch before this shrink step closes.",
        ),
        target_record(
            "roadmap_numeric_value_branch",
            ROADMAP,
            roadmap_text,
            "8.7.56.735-.738",
            "ROADMAP still names the current official branch before this shrink step closes.",
        ),
    ]
    inventory_ready = (
        all(target["present"] for target in inventory_targets)
        and electron_identification_pivot_retained
        and absolute_normalization_dictionary_ready
        and m0_numeric_from_electron_identification_ready
        and qball_ground_state_proxy_evidence_retained
        and chi_proxy_inventory_ready
        and chi_star_proxy_source_inventory_ready
        and declaration_gate_consistent
        and route_contract_consistent
    )

    inventory_payload = payload(
        "8.7.56.735",
        "Trial-2 numeric alpha chi-star-or-same-sector-proxy numeric-value source inventory",
        common_inputs,
        "Freeze the post-electron-identification residual pack and show that the next honest closure attempt starts from the missing same-sector proxy equivalence rule rather than from the old same-sector literal-fragment retry.",
        {
            "inventory_rule": "after electron identification fixes m_0, the remaining numeric alpha route stays open until the public pack exposes a usable chi_* or same-sector proxy numeric value",
            "normalization_rule": "M_(1,0,0,0) = m_e and m_0 = m_e / E(beta_1) close the absolute mass scale before the chi/proxy numeric-value inventory runs",
            "first_route_rule": "if the historical chi_* proxy source inventory still names same_sector_equivalence_rule as the first closure target and the closure retry remains non-closing for that reason, the numeric-value residual pack shrinks toward the same-sector proxy equivalence rule",
        },
        [
            row(
                "trial2_numeric_alpha_chi_proxy_numeric_value_inventory_complete",
                "pass" if inventory_ready else "reject",
                "chi-star or same-sector proxy numeric-value inventory complete",
                1 if inventory_ready else 0,
                "The electron-identification pivot, the computation formula, the chi/proxy inventories, and the current contracts are frozen as one residual pack.",
            ),
            row(
                "trial2_numeric_alpha_electron_identification_pivot_retained",
                "pass" if electron_identification_pivot_retained else "reject",
                "electron-identification pivot retained",
                1 if electron_identification_pivot_retained else 0,
                "The current residual still sits on the electron-identification absolute-normalization dictionary.",
            ),
            row(
                "trial2_numeric_alpha_absolute_normalization_dictionary_ready",
                "pass" if absolute_normalization_dictionary_ready else "reject",
                "absolute-normalization dictionary ready",
                1 if absolute_normalization_dictionary_ready else 0,
                "The current branch inherits an already-closed absolute mass scale.",
            ),
            row(
                "trial2_numeric_alpha_qball_ground_state_proxy_evidence_retained",
                "pass" if qball_ground_state_proxy_evidence_retained else "reject",
                "vector-Qball ground-state proxy evidence retained",
                1 if qball_ground_state_proxy_evidence_retained else 0,
                "The public ground-state proxy remains available for the electron-identification normalization.",
            ),
            row(
                "trial2_numeric_alpha_chi_proxy_inventory_ready",
                "pass" if chi_proxy_inventory_ready else "reject",
                "chi-proxy inventory ready",
                1 if chi_proxy_inventory_ready else 0,
                "The older anchor-normalized chi-proxy inventory remains usable as public evidence.",
            ),
            row(
                "trial2_numeric_alpha_chi_star_proxy_source_inventory_ready",
                "pass" if chi_star_proxy_source_inventory_ready else "reject",
                "chi-star proxy source inventory ready",
                1 if chi_star_proxy_source_inventory_ready else 0,
                "The historical chi-star proxy source inventory remains frozen and can be reinjected into the current mainline.",
            ),
            row(
                "trial2_numeric_alpha_first_route_to_close_is_same_sector_proxy_equivalence_rule",
                "pass" if first_route_to_close_or_none == "same_sector_proxy_equivalence_rule" else "reject",
                "first route to close is same-sector proxy equivalence rule",
                1 if first_route_to_close_or_none == "same_sector_proxy_equivalence_rule" else 0,
                "The post-pivot numeric-value residual now points to the same-sector proxy equivalence rule as the minimal next closure target.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "electron_identification_pivot_retained": electron_identification_pivot_retained,
            "absolute_normalization_dictionary_ready": absolute_normalization_dictionary_ready,
            "m0_numeric_from_electron_identification_ready": m0_numeric_from_electron_identification_ready,
            "qball_ground_state_proxy_evidence_retained": qball_ground_state_proxy_evidence_retained,
            "chi_proxy_inventory_ready": chi_proxy_inventory_ready,
            "chi_star_proxy_source_inventory_ready": chi_star_proxy_source_inventory_ready,
            "same_sector_proxy_equivalence_rule_available": same_sector_proxy_equivalence_rule_available,
            "chi_star_or_same_sector_proxy_numeric_value_available": chi_star_or_same_sector_proxy_numeric_value_available,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "declaration_gate_consistent": declaration_gate_consistent,
            "route_contract_consistent": route_contract_consistent,
        },
        {
            "overall_status": "trial2_numeric_alpha_chi_proxy_numeric_value_inventory_frozen",
            "advance_to_8_7_56_736": inventory_ready,
            "next_required_artifacts": [],
        },
        {
            "inventory_targets": inventory_targets,
            "prior_inventory_summary": prior_inventory_summary,
            "prior_audit_summary": prior_audit_summary,
            "prior_gate_summary": prior_gate_summary,
            "prior_route_summary": prior_route_summary,
            "chi_proxy_summary": chi_proxy_summary,
            "chi_star_proxy_source_inventory_summary": chi_star_proxy_source_summary,
            "chi_star_proxy_closure_retry_summary": chi_star_proxy_closure_summary,
            "same_sector_proxy_equivalence_summary": same_sector_proxy_equivalence_summary,
            "qball_spin_orbit_summary": qball_spin_orbit_summary,
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    audit_payload = payload(
        "8.7.56.736",
        "Trial-2 numeric alpha chi-star-or-same-sector-proxy numeric-value audit",
        common_inputs,
        "Audit whether the current public pack already exposes an honest numeric chi_* or same-sector proxy value and determine whether the blocker now shrinks to the same-sector proxy equivalence rule.",
        {
            "numeric_value_rule": "numeric alpha can close only if the public pack exposes a direct chi_* or same-sector proxy numeric value, or an already-public same-sector proxy equivalence rule that makes such a numeric value honest without a new fit",
            "shrink_rule": "if the historical closure retry remains non-closing because same_sector_equivalence_rule is absent, the post-pivot numeric-value blocker shrinks to same_sector_proxy_equivalence_rule itself",
            "mainline_rule": "electron identification keeps the absolute mass scale fixed, so failure to find a chi/proxy numeric value does not reopen the computation pivot",
        },
        [
            row(
                "trial2_numeric_alpha_chi_proxy_numeric_value_audit_complete",
                "pass",
                "chi-star or same-sector proxy numeric-value audit complete",
                1,
                "This step audits whether the current public pack already carries an honest numeric chi/proxy value.",
            ),
            row(
                "trial2_numeric_alpha_m0_numeric_from_electron_identification_retained",
                "pass" if m0_numeric_from_electron_identification_ready else "reject",
                "m0 numeric from electron identification retained",
                1 if m0_numeric_from_electron_identification_ready else 0,
                "The absolute mass scale remains closed during the chi/proxy numeric-value audit.",
            ),
            row(
                "trial2_numeric_alpha_chi_star_or_same_sector_proxy_numeric_value_available",
                "pass" if chi_star_or_same_sector_proxy_numeric_value_available else "reject",
                "chi-star or same-sector proxy numeric value available",
                1 if chi_star_or_same_sector_proxy_numeric_value_available else 0,
                "The current public pack still does not expose an honest numeric chi_* or same-sector proxy value.",
            ),
            row(
                "trial2_numeric_alpha_same_sector_proxy_equivalence_rule_available",
                "pass" if same_sector_proxy_equivalence_rule_available else "reject",
                "same-sector proxy equivalence rule available",
                1 if same_sector_proxy_equivalence_rule_available else 0,
                "The current public pack still lacks the equivalence rule that would make a same-sector proxy numeric value public-canonical.",
            ),
            row(
                "trial2_numeric_alpha_dominant_blocker_is_same_sector_proxy_equivalence_rule_absence",
                "pass" if dominant_blocker_is_same_sector_proxy_equivalence_rule_absence else "reject",
                "dominant blocker is same-sector proxy equivalence rule absence",
                1 if dominant_blocker_is_same_sector_proxy_equivalence_rule_absence else 0,
                "The current non-closure reason now shrinks from a generic chi/proxy numeric-value gap to the missing same-sector proxy equivalence rule.",
            ),
        ],
        {
            "audit_ready": True,
            "absolute_normalization_dictionary_ready": absolute_normalization_dictionary_ready,
            "m0_numeric_from_electron_identification_ready": m0_numeric_from_electron_identification_ready,
            "chi_star_or_same_sector_proxy_numeric_value_available": chi_star_or_same_sector_proxy_numeric_value_available,
            "same_sector_proxy_equivalence_rule_available": same_sector_proxy_equivalence_rule_available,
            "proxy_route_nonclosure_reason_or_none": proxy_route_nonclosure_reason_or_none,
            "dominant_blocker_is_same_sector_proxy_equivalence_rule_absence": dominant_blocker_is_same_sector_proxy_equivalence_rule_absence,
            "first_route_to_close_after_audit_or_none": "trial2_numeric_alpha_chi_star_or_same_sector_proxy_numeric_value_declaration_gate",
        },
        {
            "overall_status": "trial2_numeric_alpha_chi_proxy_numeric_value_audit_complete",
            "advance_to_8_7_56_737": True,
            "next_required_artifacts": [],
        },
        {
            "inventory_summary": inventory_payload["summary"],
            "prior_audit_summary": prior_audit_summary,
            "chi_star_proxy_closure_retry_summary": chi_star_proxy_closure_summary,
            "same_sector_proxy_equivalence_summary": same_sector_proxy_equivalence_summary,
        },
    )

    gate_payload = payload(
        "8.7.56.737",
        "Trial-2 numeric alpha chi-star-or-same-sector-proxy numeric-value declaration gate",
        common_inputs,
        "Close the numeric-value residual honestly: keep the computation route and electron-identification dictionary, record that numeric alpha still cannot be emitted, and freeze same-sector proxy equivalence rule as the next official blocker.",
        {
            "closure_rule": "the branch closes as numeric-open whenever the current public pack lacks the same-sector proxy equivalence rule needed to promote a public chi/proxy numeric value",
            "shrink_rule": "the generic chi/proxy numeric-value blocker becomes a same-sector-proxy-equivalence-rule blocker once the closure retry names that rule as the concrete non-closure reason",
            "mainline_rule": "precision-alpha remains on the mainline while the strong side stays on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_chi_proxy_numeric_value_gate_complete",
                "pass",
                "chi-star or same-sector proxy numeric-value declaration gate complete",
                1,
                "The numeric-value residual route is now officially closed as numeric-open.",
            ),
            row(
                "trial2_numeric_alpha_computation_formula_ready_after_numeric_value_gate",
                "pass" if prior_gate_summary["trial2_numeric_alpha_computation_formula_ready"] else "reject",
                "computation formula ready after numeric-value gate",
                1 if prior_gate_summary["trial2_numeric_alpha_computation_formula_ready"] else 0,
                "The Newton-limit alpha formula remains frozen after the numeric-value gate closes.",
            ),
            row(
                "trial2_numeric_alpha_absolute_normalization_dictionary_ready_after_numeric_value_gate",
                "pass" if absolute_normalization_dictionary_ready else "reject",
                "absolute-normalization dictionary ready after numeric-value gate",
                1 if absolute_normalization_dictionary_ready else 0,
                "Electron identification remains part of the official mainline.",
            ),
            row(
                "trial2_numeric_alpha_numeric_from_current_pack_ready_after_numeric_value_gate",
                "pass" if chi_star_or_same_sector_proxy_numeric_value_available else "reject",
                "numeric alpha from current pack ready after numeric-value gate",
                1 if chi_star_or_same_sector_proxy_numeric_value_available else 0,
                "Numeric alpha still cannot be emitted because the same-sector proxy equivalence rule is absent.",
            ),
            row(
                "trial2_numeric_alpha_blocker_shrunk_to_same_sector_proxy_equivalence_rule",
                "pass" if dominant_blocker_is_same_sector_proxy_equivalence_rule_absence else "reject",
                "blocker shrunk to same-sector proxy equivalence rule",
                1 if dominant_blocker_is_same_sector_proxy_equivalence_rule_absence else 0,
                "The next route no longer targets the generic chi/proxy numeric-value family; it targets the missing same-sector proxy equivalence rule directly.",
            ),
        ],
        {
            "trial2_numeric_alpha_computation_formula_ready": bool(
                prior_gate_summary["trial2_numeric_alpha_computation_formula_ready"]
            ),
            "trial2_numeric_alpha_absolute_normalization_dictionary_ready": absolute_normalization_dictionary_ready,
            "trial2_numeric_alpha_numeric_from_current_pack_ready": chi_star_or_same_sector_proxy_numeric_value_available,
            "trial2_numeric_alpha_closeout_ready": False,
            "dominant_blocker_shrunk_from_chi_star_or_same_sector_proxy_numeric_value_to_same_sector_proxy_equivalence_rule": dominant_blocker_is_same_sector_proxy_equivalence_rule_absence,
            "selected_residual_route": NEXT_RESIDUAL_ROUTE,
            "missing_v2_artifact": NEXT_MISSING_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_numeric_value_gate_closed_same_sector_proxy_equivalence_rule_open",
            "advance_to_8_7_56_738": True,
            "next_required_artifacts": [NEXT_MISSING_ARTIFACT],
        },
        {
            "audit_summary": audit_payload["summary"],
            "prior_gate_summary": prior_gate_summary,
            "prior_route_summary": prior_route_summary,
        },
    )

    route_payload = payload(
        "8.7.56.738",
        "Trial-2 numeric alpha next-generation route contract eighty-first refresh",
        common_inputs,
        "Refresh the next-generation contract after the chi/proxy numeric-value shrink: keep precision-alpha on the mainline, keep the strong side on reserve, and promote same-sector proxy equivalence rule as the next official blocker.",
        {
            "selected_route_rule": "the next official route is same_sector_proxy_equivalence_rule once the chi/proxy numeric-value route freezes its non-closure reason",
            "mainline_rule": "precision-alpha remains the next-generation mainline after the numeric-value shrink",
            "reserve_rule": "strong-side non-Abelian, running, and confinement gaps remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_numeric_value_gate_closed_before_refresh",
                "pass",
                "chi/proxy numeric-value declaration gate closed before route refresh",
                1,
                "The route contract is refreshed only after the numeric-value gate has frozen the next blocker.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_same_sector_proxy_equivalence_rule",
                "pass",
                "same-sector proxy equivalence rule route selected",
                1,
                "The next official route now targets the concrete same-sector proxy equivalence rule blocker.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_numeric_value_shrink",
                "pass" if prior_route_summary["precision_alpha_mainline_retained"] else "reject",
                "precision-alpha mainline retained after numeric-value shrink",
                1 if prior_route_summary["precision_alpha_mainline_retained"] else 0,
                "The precision-alpha route remains the official mainline.",
            ),
            row(
                "trial2_numeric_alpha_strong_side_route_state_retained_as_v3_hold_reserve",
                "pass" if prior_route_summary["strong_side_route_state"] == "v3_hold_reserve" else "reject",
                "strong-side route state retained as v3 hold reserve",
                1 if prior_route_summary["strong_side_route_state"] == "v3_hold_reserve" else 0,
                "The strong side remains exploratory and stays outside the promoted mainline.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_RESIDUAL_ROUTE,
            "strong_side_route_state": prior_route_summary["strong_side_route_state"],
            "precision_alpha_mainline_retained": bool(prior_route_summary["precision_alpha_mainline_retained"]),
            "electron_identification_pivot_retained": electron_identification_pivot_retained,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_eighty_first_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [
                "trial2_numeric_alpha_same_sector_proxy_equivalence_rule_source_inventory",
                "trial2_numeric_alpha_same_sector_proxy_equivalence_rule_audit",
            ],
        },
        {
            "gate_summary": gate_payload["summary"],
            "prior_route_summary": prior_route_summary,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_chi_star_or_same_sector_proxy_numeric_value_source_inventory",
        inventory_payload,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_chi_star_or_same_sector_proxy_numeric_value_audit",
        audit_payload,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_chi_star_or_same_sector_proxy_numeric_value_declaration_gate",
        gate_payload,
    )
    write_artifact("mass_origin_v2_t2_alpha_route_contract_eighty_first_refresh", route_payload)

    print("[done] 8.7.56.735-.738 artifacts generated:")
    print(" - mass_origin_v2_trial2_numeric_alpha_chi_star_or_same_sector_proxy_numeric_value_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_chi_star_or_same_sector_proxy_numeric_value_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_chi_star_or_same_sector_proxy_numeric_value_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_eighty_first_refresh_metrics.json")
    print(f" - next official branch should move to {NEXT_BRANCH}")


# Function: run the numeric-value residual branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the chi/proxy numeric-value residual branch."""
    main()


if __name__ == "__main__":
    run_cli()
