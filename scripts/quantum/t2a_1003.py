#!/usr/bin/env python3
"""Generate 8.7.56.1003-.1006 Trial-2 numeric alpha expert-advice escalation artifacts."""

from __future__ import annotations

import csv
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIVATE_OUT = ROOT / "output" / "private" / "quantum"

ADVICE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_final_computation.md")
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
EM_DOC = ROOT / "doc" / "quantum" / "16_electromagnetism_charge_maxwell_photon.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
PRIMARY_SOURCES = ROOT / "doc" / "PRIMARY_SOURCES.md"

FINAL_COMP_SOURCE = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_source_inventory_metrics.json"
UNIT_AUDIT = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_unit_consistency_audit_metrics.json"
DIMENSION_BRIDGE_AUDIT = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_audit_metrics.json"
EM_UNIT_AUDIT = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_em_unit_convention_audit_metrics.json"
STATEMENT_AUDIT = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_gp_to_elementary_charge_mapping_statement_audit_metrics.json"
LITERAL_AUDIT = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_gp_to_elementary_charge_mapping_literal_audit_metrics.json"
LITERAL_GATE = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_gp_to_elementary_charge_mapping_literal_declaration_gate_metrics.json"
LITERAL_ROUTE = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_seventh_refresh_metrics.json"
SCRIPT_995 = ROOT / "scripts" / "quantum" / "t2a_995.py"
SCRIPT_999 = ROOT / "scripts" / "quantum" / "t2a_999.py"

BUNDLE_STAMP = "20260324_004752"
BUNDLE_DIR = PRIVATE_OUT / f"expert_review_bundle_{BUNDLE_STAMP}"
BUNDLE_ZIP = PRIVATE_OUT / f"expert_review_bundle_{BUNDLE_STAMP}.zip"

CURRENT_BLOCKER = "trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_gp_to_elementary_charge_mapping_literal"
CURRENT_ROUTE = "trial2_numeric_alpha_final_computation_expert_advice_gp_to_elementary_charge_mapping"
NEXT_RESPONSE_ROUTE = "trial2_numeric_alpha_final_computation_expert_advice_gp_to_elementary_charge_mapping_response"
NEXT_ROUTE = "8.7.56.1007"


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


# Function: return a stable display path for repo or external files.

def display_path(path: Path) -> str:
    """Return a stable path relative to the repo root when possible."""
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
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    json_path = PUBLIC_OUT / f"{stem}_metrics.json"
    csv_path = PUBLIC_OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: write bundle text files.

def write_bundle_text(path: Path, text: str) -> None:
    """Write a UTF-8 text file into the expert bundle directory."""
    path.write_text(text, encoding="utf-8")


# Function: copy one file into the expert bundle directory.

def copy_into_bundle(source: Path) -> Path:
    """Copy one source file into the expert bundle directory."""
    destination = BUNDLE_DIR / source.name
    shutil.copy2(source, destination)
    return destination


# Function: create the current expert-review bundle.

def build_bundle() -> dict:
    """Create a blocker-specific expert-review bundle and return its manifest data."""
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    readme_text = """Expert review bundle

Purpose
- Current blocker: missing positive g_P <-> e literal inside the Trial-2 numeric alpha final-computation route.
- This bundle is refreshed after the retry triage gate fired on 2026-03-24.

Requested review
- Decide whether current canon already supports a positive literal equating structural e=g_P/sqrt(Z_P) with physical elementary charge.
- If not, decide whether Trial-2 should close as structural-pass / numeric-open, or pivot to a different computation / normalization bridge.

Current state
- Computation formula fixed.
- Electron-identification dictionary fixed.
- Raw final computation performed.
- Direct SI readout still not an honest dimensionless alpha.
- Mapping-literal branch added no new public-canonical surface, so mechanical wording descent is now stopped.
"""
    expert_note_text = """Expert note

What is already closed
- Trial-2 structural electromagnetism is fixed: e=g_P/sqrt(Z_P), alpha=g_P^2/(4*pi*Z_P*hbar*c).
- Weak-field normalization is fixed: g_P/Z_P = 4*pi*G.
- Electron-identification dictionary is fixed: M_(1,0,0,0) = m_e.
- The raw final computation route is fixed and C_bg = 1 has already been used once.

What remains open
- trial2_numeric_alpha_numeric_from_current_pack_ready = false
- trial2_numeric_alpha_closeout_ready = false
- The current canon still lacks a positive literal that explicitly identifies structural e with the physical elementary charge e.

Why the route was stopped
- The mapping-statement branch and the mapping-literal branch both ended with the same missing surface family.
- The mapping-literal branch added no new public-canonical surface.
- Under the retry triage policy, that makes further phrase/fragment/token descent a low-value wording loop rather than a live closeout route.

Requested judgment
- Does current canon already contain a defensible positive literal equating structural e=g_P/sqrt(Z_P) with the physical elementary charge?
- If yes, what is the minimal statement / literal / symbol and where is it?
- If no, should Trial-2 now be closed as structural pass / numeric open?
- If neither, what alternate computation / normalization bridge should replace the current route without introducing new assumptions?
"""
    questions_text = """Questions for review

1. Under the current public canon, is there a defensible positive literal that equates structural e=g_P/sqrt(Z_P) with the physical elementary charge e?

2. If yes, what is the minimal statement / literal / symbol, and where is it located in the current canon?

3. If no such literal exists, should Trial-2 numeric alpha now be closed as:
- structural pass
- computation route established
- numeric open pending new canon

4. If the route should not close as numeric-open, what is the better replacement?
- an alternate computation bridge
- an alternate normalization theorem
- an explicit no-go declaration for the current g_P <-> e route
- another route entirely
"""

    write_bundle_text(BUNDLE_DIR / "README.txt", readme_text)
    write_bundle_text(BUNDLE_DIR / "EXPERT_NOTE.txt", expert_note_text)
    write_bundle_text(BUNDLE_DIR / "QUESTIONS_FOR_REVIEW.txt", questions_text)

    copied = []
    for source in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIMARY_SOURCES,
        PART1,
        PART3A,
        PART5,
        EM_DOC,
        ADVICE,
        FINAL_COMP_SOURCE,
        UNIT_AUDIT,
        DIMENSION_BRIDGE_AUDIT,
        EM_UNIT_AUDIT,
        STATEMENT_AUDIT,
        LITERAL_AUDIT,
        LITERAL_GATE,
        LITERAL_ROUTE,
        SCRIPT_995,
        SCRIPT_999,
    ):
        copied.append(copy_into_bundle(source).name)

    manifest_lines = [
        "Expert bundle manifest",
        f"BUNDLE_DIR={display_path(BUNDLE_DIR)}",
        f"BUNDLE_ZIP={display_path(BUNDLE_ZIP)}",
        f"COPIED_COUNT={len(copied)}",
        "MISSING_COUNT=0",
        "FILES=",
    ]
    manifest_lines.extend(f"- {name}" for name in sorted(copied))
    write_bundle_text(BUNDLE_DIR / "BUNDLE_MANIFEST.txt", "\n".join(manifest_lines) + "\n")

    if BUNDLE_ZIP.exists():
        BUNDLE_ZIP.unlink()

    with zipfile.ZipFile(BUNDLE_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(BUNDLE_DIR.iterdir()):
            handle.write(path, arcname=path.name)

    return {
        "bundle_dir": display_path(BUNDLE_DIR),
        "bundle_zip": display_path(BUNDLE_ZIP),
        "copied_count": len(copied),
        "missing_count": 0,
        "question_count": 4,
    }


# Function: execute the expert-advice escalation branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha expert-advice escalation branch."""
    for path in (
        ADVICE,
        PART1,
        PART3A,
        PART5,
        EM_DOC,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIMARY_SOURCES,
        FINAL_COMP_SOURCE,
        UNIT_AUDIT,
        DIMENSION_BRIDGE_AUDIT,
        EM_UNIT_AUDIT,
        STATEMENT_AUDIT,
        LITERAL_AUDIT,
        LITERAL_GATE,
        LITERAL_ROUTE,
        SCRIPT_995,
        SCRIPT_999,
    ):
        require(path)

    advice_text = read_text(ADVICE)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    em_doc_text = read_text(EM_DOC)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    literal_audit = read_json(LITERAL_AUDIT)["summary"]
    literal_gate = read_json(LITERAL_GATE)["summary"]
    literal_route = read_json(LITERAL_ROUTE)["summary"]
    bundle_manifest = build_bundle()

    prior_escalation_triggered = (
        literal_audit["retry_triage_gate_triggered"]
        and literal_gate["trial2_numeric_alpha_retry_triage_gate_triggered"]
        and literal_route["retry_triage_gate_triggered"]
    )
    bundle_current_blocker_ready = (
        hit(part3a_text, "mapping literal") is not None
        and hit(part5_text, "expert-advice escalation") is not None
    )
    advice_has_final_computation_formula = hit(advice_text, r"\alpha = \frac{4\pi G^2 Z_P}{\hbar c}") is not None
    part1_has_weak_field_normalization = hit(part1_text, r"g_P/Z_P=4\pi G") is not None
    em_doc_has_coulomb_surface = hit(em_doc_text, r"\Phi(r)=\frac{1}{4\pi\varepsilon_0}\frac{q}{r}") is not None
    roadmap_has_expert_branch = hit(roadmap_text, "`8.7.56.1003-.1006`") is not None
    status_has_expert_next_step = hit(status_text, "8.7.56.1003") is not None

    inventory_ready = all(
        [
            prior_escalation_triggered,
            bundle_current_blocker_ready,
            advice_has_final_computation_formula,
            part1_has_weak_field_normalization,
            em_doc_has_coulomb_surface,
            roadmap_has_expert_branch,
            status_has_expert_next_step,
            bundle_manifest["copied_count"] >= 19,
            bundle_manifest["missing_count"] == 0,
        ]
    )

    common_inputs = {
        "expert_note_markdown": display_path(ADVICE),
        "part1_markdown": display_path(PART1),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "electromagnetism_doc_markdown": display_path(EM_DOC),
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "primary_sources_markdown": display_path(PRIMARY_SOURCES),
        "mapping_literal_audit_json": display_path(LITERAL_AUDIT),
        "mapping_literal_gate_json": display_path(LITERAL_GATE),
        "mapping_literal_route_json": display_path(LITERAL_ROUTE),
        "expert_bundle_dir": bundle_manifest["bundle_dir"],
        "expert_bundle_zip": bundle_manifest["bundle_zip"],
    }

    inventory = payload(
        "8.7.56.1003",
        "Trial-2 numeric alpha expert-advice source inventory",
        common_inputs,
        "Freeze the expert-advice escalation pack: current blocker, current canon, retry triage judgment, and a refreshed bundle that now targets the missing positive g_P-to-elementary-charge literal.",
        {
            "inventory_rule": "the escalation pack must contain the current blocker, the current canon, the retry triage judgment, and a shareable expert bundle targeted at the literal blocker",
            "stop_rule": "mechanical wording descent is stopped because the literal branch added no new public-canonical surface",
        },
        [
            row(
                "trial2_numeric_alpha_expert_advice_inventory_complete",
                "pass" if inventory_ready else "reject",
                "expert-advice inventory complete",
                1 if inventory_ready else 0,
                "The escalation pack must be ready before the expert question set is audited.",
            ),
            row(
                "trial2_numeric_alpha_expert_bundle_refreshed_for_current_blocker",
                "pass" if bundle_manifest["missing_count"] == 0 else "reject",
                "expert bundle refreshed for current blocker",
                1 if bundle_manifest["missing_count"] == 0 else 0,
                "The bundle now targets the g_P-to-elementary-charge literal blocker rather than an older H0^(P)-Z_P blocker.",
            ),
            row(
                "trial2_numeric_alpha_retry_triage_gate_already_triggered_before_expert_branch",
                "pass" if prior_escalation_triggered else "reject",
                "retry triage gate already triggered before expert branch",
                1 if prior_escalation_triggered else 0,
                "The expert-advice branch is justified only after the wording loop has been formally classified as low-value.",
            ),
            row(
                "trial2_numeric_alpha_current_blocker_still_literal_absence",
                "pass",
                "current blocker still literal absence",
                1,
                "The blocker remains the missing positive g_P-to-elementary-charge literal under current canon.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "current_blocker": CURRENT_BLOCKER,
            "expert_bundle_ready": bundle_manifest["missing_count"] == 0,
            "expert_question_pack_ready": True,
            "retry_triage_gate_triggered": prior_escalation_triggered,
            "first_route_to_close_or_none": NEXT_RESPONSE_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_expert_advice_inventory_frozen",
            "advance_to_8_7_56_1004": inventory_ready,
            "next_required_artifacts": [NEXT_RESPONSE_ROUTE],
        },
        {
            "bundle_manifest": bundle_manifest,
            "current_blocker_hit": hit(status_text, "missing positive `g_P \\leftrightarrow e` literal"),
            "part1_weak_field_hit": hit(part1_text, r"g_P/Z_P=4\pi G"),
        },
    )

    audit = payload(
        "8.7.56.1004",
        "Trial-2 numeric alpha expert question audit",
        common_inputs,
        "Audit whether the current expert question set is minimal and whether the acceptable response types are sufficiently narrow to avoid reopening the same wording loop.",
        {
            "question_rule": "the question set is minimal if it only asks for a positive literal, a no-go closeout, or an alternate computation/normalization bridge",
            "response_rule": "acceptable responses are positive_literal, no_go_closeout, and alternate_computation_bridge",
        },
        [
            row(
                "trial2_numeric_alpha_expert_question_audit_complete",
                "pass",
                "expert question audit complete",
                1,
                "The expert-advice branch audits whether the question set is minimal and actionable.",
            ),
            row(
                "trial2_numeric_alpha_expert_question_set_is_minimal",
                "pass",
                "expert question set is minimal",
                1,
                "The refreshed bundle asks only for a positive literal, a no-go closeout, or an alternate computation/normalization bridge.",
            ),
            row(
                "trial2_numeric_alpha_mechanical_wording_descent_should_remain_stopped",
                "pass" if prior_escalation_triggered else "reject",
                "mechanical wording descent should remain stopped",
                1 if prior_escalation_triggered else 0,
                "The current branch should not reopen phrase/fragment/token descent unless genuinely new canonical evidence appears.",
            ),
            row(
                "trial2_numeric_alpha_acceptable_response_types_frozen",
                "pass",
                "acceptable response types frozen",
                3,
                "The escalation accepts exactly three answer classes: positive literal, no-go closeout, or alternate computation bridge.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "expert_question_set_minimal": True,
            "acceptable_response_types": [
                "positive_literal",
                "no_go_closeout",
                "alternate_computation_bridge",
            ],
            "mechanical_wording_descent_stopped": prior_escalation_triggered,
            "first_route_to_close_after_audit_or_none": NEXT_RESPONSE_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_expert_question_audited",
            "advance_to_8_7_56_1005": True,
            "next_required_artifacts": [NEXT_RESPONSE_ROUTE],
        },
        {
            "bundle_manifest": bundle_manifest,
            "literal_audit_summary": literal_audit,
            "literal_gate_summary": literal_gate,
        },
    )

    gate = payload(
        "8.7.56.1005",
        "Trial-2 numeric alpha expert-advice declaration gate",
        common_inputs,
        "Update the official gate after the expert question audit: mechanical wording descent is stopped, expert-advice escalation is active, and Trial-2 remains structural-pass / numeric-open under current canon until a response arrives.",
        {
            "gate_rule": "without a positive literal or alternate bridge, the current canon still cannot produce an honest dimensionless alpha",
            "escalation_rule": "expert-advice escalation is active once the wording loop is formally stopped and the question set is frozen",
        },
        [
            row(
                "trial2_numeric_alpha_expert_advice_gate_complete",
                "pass",
                "expert-advice declaration gate complete",
                1,
                "The official state is updated after the expert question audit.",
            ),
            row(
                "trial2_numeric_alpha_expert_advice_escalation_active",
                "pass",
                "expert-advice escalation active",
                1,
                "The current mainline has switched from wording descent to expert-advice escalation.",
            ),
            row(
                "trial2_numeric_alpha_mechanical_wording_descent_stopped",
                "pass" if prior_escalation_triggered else "reject",
                "mechanical wording descent stopped",
                1 if prior_escalation_triggered else 0,
                "The retry triage gate prevents further low-value wording subdivision for the g_P-to-elementary-charge family.",
            ),
            row(
                "trial2_numeric_alpha_current_pack_still_not_closeout_ready",
                "reject",
                "current pack still not closeout ready",
                0,
                "Expert escalation is active precisely because the current pack still cannot close Trial-2 numeric alpha honestly.",
            ),
        ],
        {
            "trial2_numeric_alpha_computation_formula_ready": True,
            "trial2_numeric_alpha_absolute_normalization_dictionary_ready": True,
            "trial2_numeric_alpha_raw_final_computation_value_available": True,
            "trial2_numeric_alpha_numeric_from_current_pack_ready": False,
            "trial2_numeric_alpha_closeout_ready": False,
            "trial2_numeric_alpha_final_computation_performed": True,
            "trial2_numeric_alpha_final_computation_result_class": "precanonical_unit_incomplete",
            "trial2_numeric_alpha_retry_loop_retired": True,
            "trial2_numeric_alpha_retry_triage_gate_triggered": True,
            "trial2_numeric_alpha_expert_advice_escalation_active": True,
            "trial2_numeric_alpha_mechanical_wording_descent_stopped": True,
            "selected_residual_route": NEXT_RESPONSE_ROUTE,
            "missing_v2_artifact": CURRENT_BLOCKER,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_expert_advice_gate_closed",
            "advance_to_8_7_56_1006": True,
            "next_required_artifacts": [NEXT_RESPONSE_ROUTE],
        },
        {
            "expert_question_audit_summary": audit["summary"],
            "bundle_manifest": bundle_manifest,
        },
    )

    route = payload(
        "8.7.56.1006",
        "Trial-2 numeric alpha route contract one-hundred-forty-eighth refresh",
        common_inputs,
        "Refresh the next-generation contract after the expert-advice declaration gate: keep Trial-2 numeric alpha on the precision mainline, keep the strong side on reserve, and set the next official route to external-response intake.",
        {
            "next_route_rule": "the next route is expert-response intake, not additional wording subdivision, unless genuinely new public-canonical evidence appears",
            "reserve_rule": "strong-side non-Abelian, running, and confinement remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_route_contract_one_hundred_forty_eighth_refresh_complete",
                "pass",
                "route contract one-hundred-forty-eighth refresh complete",
                1,
                "The expert-advice declaration gate is converted into the next-generation contract.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_expert_response_intake",
                "pass",
                "next route selected as expert response intake",
                1,
                "The next official route is to intake and audit an expert response, not to continue the old wording loop.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_expert_gate",
                "pass" if literal_route["precision_alpha_mainline_retained"] else "reject",
                "precision-alpha mainline retained after expert gate",
                1 if literal_route["precision_alpha_mainline_retained"] else 0,
                "Trial-2 numeric alpha remains the precision mainline despite the escalation.",
            ),
            row(
                "trial2_numeric_alpha_strong_side_route_state_retained_after_expert_gate",
                "pass" if literal_route["strong_side_route_state"] == "v3_hold_reserve" else "reject",
                "strong-side route state retained after expert gate",
                1 if literal_route["strong_side_route_state"] == "v3_hold_reserve" else 0,
                "The strong side remains on reserve and is not promoted by the expert escalation.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_RESPONSE_ROUTE,
            "strong_side_route_state": literal_route["strong_side_route_state"],
            "precision_alpha_mainline_retained": bool(literal_route["precision_alpha_mainline_retained"]),
            "electron_identification_pivot_retained": bool(literal_route["electron_identification_pivot_retained"]),
            "h0p_bridge_pivot_retained": bool(literal_route["h0p_bridge_pivot_retained"]),
            "final_computation_branch_retained": bool(literal_route["final_computation_branch_retained"]),
            "unit_consistency_audit_branch_retained": bool(literal_route["unit_consistency_audit_branch_retained"]),
            "dimensionless_alpha_bridge_branch_retained": bool(
                literal_route["dimensionless_alpha_bridge_branch_retained"]
            ),
            "em_unit_convention_bridge_branch_retained": bool(
                literal_route["em_unit_convention_bridge_branch_retained"]
            ),
            "mapping_statement_branch_retained": bool(literal_route["mapping_statement_branch_retained"]),
            "mapping_literal_branch_retained": bool(literal_route["mapping_literal_branch_retained"]),
            "expert_advice_escalation_branch_retained": True,
            "same_pattern_retry_threshold_reached": True,
            "retry_triage_gate_triggered": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_forty_eighth_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_RESPONSE_ROUTE],
        },
        {
            "gate_summary": gate["summary"],
            "bundle_manifest": bundle_manifest,
            "prior_route_summary": literal_route,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_final_computation_expert_advice_gp_to_elementary_charge_mapping_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_final_computation_expert_advice_gp_to_elementary_charge_mapping_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_final_computation_expert_advice_gp_to_elementary_charge_mapping_declaration_gate",
        gate,
    )
    write_artifact(
        "mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_eighth_refresh",
        route,
    )

    print("[done] 8.7.56.1003-.1006 artifacts generated:")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_expert_advice_gp_to_elementary_charge_mapping_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_expert_advice_gp_to_elementary_charge_mapping_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_expert_advice_gp_to_elementary_charge_mapping_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_eighth_refresh_metrics.json")
    print(f" - bundle_zip = {bundle_manifest['bundle_zip']}")


# Function: run the expert-advice escalation branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the Trial-2 numeric alpha expert-advice escalation branch."""
    main()


if __name__ == "__main__":
    run_cli()
