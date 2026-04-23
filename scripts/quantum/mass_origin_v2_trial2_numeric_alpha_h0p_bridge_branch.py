#!/usr/bin/env python3
"""Generate 8.7.56.839-.842 Trial-2 numeric alpha H0^(P)-Z_P bridge pivot artifacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"

ADVICE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_h0p_bridge.md")
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART2 = ROOT / "doc" / "paper" / "11_part2_astrophysics.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

COMPUTATION_GATE = OUT / "mass_origin_v2_trial2_numeric_alpha_computation_declaration_gate_metrics.json"
ELECTRON_GATE = OUT / "mass_origin_v2_trial2_numeric_alpha_electron_identification_declaration_gate_metrics.json"
PRIOR_GATE = (
    OUT / "mass_origin_v2_trial2_numeric_alpha_same_sector_proxy_equivalence_terminal_glyph_declaration_gate_metrics.json"
)
PRIOR_ROUTE = OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_sixth_refresh_metrics.json"

CURRENT_RETRY_ROUTE = "trial2_numeric_alpha_newton_limit_same_sector_proxy_equivalence_symbol_fragment_identification"
CURRENT_RETRY_ARTIFACT = "trial2_numeric_alpha_same_sector_proxy_equivalence_symbol_fragment"
CURRENT_PIVOT_ROUTE = "trial2_numeric_alpha_newton_limit_h0p_zp_bridge_pivot"
NEXT_ROUTE = "8.7.56.843"
NEXT_BRANCH = "8.7.56.843-.846"
NEXT_RESIDUAL_ROUTE = "trial2_numeric_alpha_newton_limit_h0p_zp_bridge_statement_identification"
NEXT_MISSING_ARTIFACT = "trial2_numeric_alpha_h0p_zp_bridge_statement"


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


# Function: return the first hit among multiple patterns.

def first_hit(text: str, patterns: list[str]) -> dict | None:
    """Return the first available hit from the ordered list of patterns."""
    for pattern in patterns:
        candidate = hit(text, pattern)
        if candidate is not None:
            return candidate

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


# Function: execute the H0^(P)-Z_P bridge pivot branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha H0^(P)-Z_P bridge pivot branch."""
    for path in (
        ADVICE,
        PART1,
        PART2,
        PART3A,
        PART5,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        COMPUTATION_GATE,
        ELECTRON_GATE,
        PRIOR_GATE,
        PRIOR_ROUTE,
    ):
        require(path)

    advice_text = read_text(ADVICE)
    part1_text = read_text(PART1)
    part2_text = read_text(PART2)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    computation_gate = read_json(COMPUTATION_GATE)
    electron_gate = read_json(ELECTRON_GATE)
    prior_gate = read_json(PRIOR_GATE)
    prior_route = read_json(PRIOR_ROUTE)

    computation_gate_summary = computation_gate["summary"]
    electron_gate_summary = electron_gate["summary"]
    prior_gate_summary = prior_gate["summary"]
    prior_route_summary = prior_route["summary"]

    advice_bridge_candidate_hit = first_hit(
        advice_text,
        [
            r"Z_P = \frac{m_0^2 \cdot C_{\rm bg}}{(H_0^{(P)})^2}",
            r"Z_P = \frac{m_0^2}{(H_0^{(P)})^2}",
            r"Z_P \,(H_0^{(P)})^2 = m_0^2 \cdot C_{\rm bg}",
        ],
    )
    part1_m0_formula_hit = hit(part1_text, r"m_0^2 = \frac{4\lambda v^2}{Z_P}")
    part1_electron_identification_hit = hit(part1_text, r"M_{(1,0,0,0)} = m_e")
    part2_background_wave_hit = hit(part2_text, r"P_{\mathrm{bg}}(t)\propto\exp[-H_{0}^{(P)}(t-t_0)]")
    part2_h0p_scale_hit = first_hit(
        part2_text,
        [
            r"a_0=\frac{cH_{0}^{(P)}}{2\pi}",
            r"\omega_{\mathrm{bg}}=H_{0}^{(P)}",
        ],
    )

    explicit_h0p_zp_bridge_statement_hit = first_hit(
        "\n".join([part1_text, part2_text, part3a_text, part5_text]),
        [
            r"Z_P = \frac{m_0^2}{(H_0^{(P)})^2}",
            r"Z_P = \frac{m_0^2 \cdot C_{\rm bg}}{(H_0^{(P)})^2}",
            r"Z_P(H_0^{(P)})^2 = m_0^2",
            r"Z_P (H_0^{(P)})^2 = m_0^2",
            r"Z_P(H_0^{(P)})^2 = m_0^2 C_{\rm bg}",
            r"Z_P (H_0^{(P)})^2 = m_0^2 C_{\rm bg}",
        ],
    )
    explicit_background_factor_hit = first_hit(
        "\n".join([part1_text, part2_text, part3a_text, part5_text]),
        [r"C_{\rm bg}", "C_bg"],
    )

    computation_formula_ready = bool(computation_gate_summary["trial2_numeric_alpha_computation_formula_ready"])
    absolute_normalization_dictionary_ready = bool(
        electron_gate_summary["trial2_numeric_alpha_absolute_normalization_dictionary_ready"]
    )
    prior_same_sector_retry_active = (
        prior_gate_summary["selected_residual_route"] == CURRENT_RETRY_ROUTE
        and prior_gate_summary["missing_v2_artifact"] == CURRENT_RETRY_ARTIFACT
        and prior_route_summary["selected_next_generation_route"] == CURRENT_RETRY_ROUTE
    )
    h0p_background_law_ready = part2_background_wave_hit is not None and part2_h0p_scale_hit is not None
    h0p_bridge_candidate_ready = (
        advice_bridge_candidate_hit is not None
        and part1_m0_formula_hit is not None
        and part1_electron_identification_hit is not None
        and h0p_background_law_ready
        and computation_formula_ready
        and absolute_normalization_dictionary_ready
        and prior_same_sector_retry_active
    )
    explicit_h0p_zp_bridge_statement_available = explicit_h0p_zp_bridge_statement_hit is not None
    explicit_background_factor_available = explicit_background_factor_hit is not None
    dominant_blocker_is_h0p_zp_bridge_statement_absence = (
        h0p_bridge_candidate_ready and not explicit_h0p_zp_bridge_statement_available
    )

    common_inputs = {
        "expert_note_markdown": display_path(ADVICE),
        "part1_markdown": display_path(PART1),
        "part2_markdown": display_path(PART2),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "mass_origin_v2_trial2_numeric_alpha_computation_declaration_gate_json": display_path(
            COMPUTATION_GATE
        ),
        "mass_origin_v2_trial2_numeric_alpha_electron_identification_declaration_gate_json": display_path(
            ELECTRON_GATE
        ),
        "mass_origin_v2_trial2_numeric_alpha_same_sector_proxy_equivalence_terminal_glyph_declaration_gate_json": display_path(
            PRIOR_GATE
        ),
        "mass_origin_v2_t2_alpha_route_contract_one_hundred_sixth_refresh_json": display_path(
            PRIOR_ROUTE
        ),
    }

    inventory_targets = [
        target_record(
            "advice_h0p_bridge_candidate",
            ADVICE,
            advice_text,
            r"Z_P = \frac{m_0^2 \cdot C_{\rm bg}}{(H_0^{(P)})^2}",
            "The expert advice already proposes an H0^(P)-Z_P bridge candidate for numeric alpha closeout.",
        ),
        target_record(
            "part1_m0_formula",
            PART1,
            part1_text,
            r"m_0^2 = \frac{4\lambda v^2}{Z_P}",
            "Part I still carries the m0-Z_P relation needed before any H0^(P)-Z_P bridge can be evaluated.",
        ),
        target_record(
            "part1_electron_identification_statement",
            PART1,
            part1_text,
            r"M_{(1,0,0,0)} = m_e",
            "Part I still carries the electron-identification absolute-normalization dictionary.",
        ),
        target_record(
            "part2_background_wave_law",
            PART2,
            part2_text,
            r"P_{\mathrm{bg}}(t)\propto\exp[-H_{0}^{(P)}(t-t_0)]",
            "Part II still carries the late-time background-wave law from which H0^(P) is read.",
        ),
        target_record(
            "part2_h0p_scale_mapping",
            PART2,
            part2_text,
            r"\omega_{\mathrm{bg}}=H_{0}^{(P)}",
            "Part II still exposes H0^(P) as the late-time background-wave scale.",
        ),
        target_record(
            "part3a_current_same_sector_retry_wording",
            PART3A,
            part3a_text,
            "same_sector_proxy_equivalence_symbol_fragment",
            "Part III-A still names the current same-sector proxy equivalence symbol-fragment blocker before the pivot closes.",
        ),
        target_record(
            "part5_current_same_sector_retry_wording",
            PART5,
            part5_text,
            "same_sector_proxy_equivalence_symbol_fragment",
            "Part V still names the current same-sector proxy equivalence symbol-fragment blocker before the pivot closes.",
        ),
        target_record(
            "status_current_symbol_fragment_branch",
            STATUS,
            status_text,
            "8.7.56.839-.842",
            "STATUS still names the current official symbol-fragment branch before the pivot closes.",
        ),
        target_record(
            "roadmap_current_symbol_fragment_branch",
            ROADMAP,
            roadmap_text,
            "8.7.56.839-.842",
            "ROADMAP still names the current official symbol-fragment branch before the pivot closes.",
        ),
    ]
    inventory_ready = all(target["present"] for target in inventory_targets) and h0p_bridge_candidate_ready

    inventory_payload = payload(
        "8.7.56.839",
        "Trial-2 numeric alpha H0^(P)-Z_P bridge pivot source inventory",
        common_inputs,
        "Freeze the H0^(P)-Z_P bridge candidate pack and show that the current blocker is better described as an explicit bridge statement problem than as a same-sector proxy wording fragment loop.",
        {
            "pivot_rule": "if the computation formula, electron-identification dictionary, and late-time H0^(P) background law are all public while the advice supplies a concrete Z_P bridge candidate, the mainline should pivot away from the same-sector proxy wording retry",
            "bridge_rule": "the H0^(P)-Z_P bridge candidate combines m0 from electron identification with H0^(P) from the late-time background-wave law and therefore targets the remaining absolute-normalization freedom directly",
            "inventory_rule": "the pivot is admissible only if the current official route still points at the same-sector proxy equivalence symbol-fragment retry before the pivot closes it",
        },
        [
            row(
                "trial2_numeric_alpha_h0p_bridge_pivot_inventory_complete",
                "pass" if inventory_ready else "reject",
                "H0^(P)-Z_P bridge pivot inventory complete",
                1 if inventory_ready else 0,
                "The advice note, Part I m0 relation, Part II H0^(P) background law, and current symbol-fragment retry are frozen as one pivot pack.",
            ),
            row(
                "trial2_numeric_alpha_computation_formula_ready_before_h0p_pivot",
                "pass" if computation_formula_ready else "reject",
                "numeric alpha computation formula ready before H0^(P)-Z_P pivot",
                1 if computation_formula_ready else 0,
                "The computation formula remains the route basis through the pivot.",
            ),
            row(
                "trial2_numeric_alpha_absolute_normalization_dictionary_ready_before_h0p_pivot",
                "pass" if absolute_normalization_dictionary_ready else "reject",
                "absolute-normalization dictionary ready before H0^(P)-Z_P pivot",
                1 if absolute_normalization_dictionary_ready else 0,
                "Electron identification remains part of the mainline while the bridge candidate is evaluated.",
            ),
            row(
                "trial2_numeric_alpha_h0p_background_law_ready",
                "pass" if h0p_background_law_ready else "reject",
                "late-time H0^(P) background law ready",
                1 if h0p_background_law_ready else 0,
                "Part II already fixes H0^(P) from the background-wave law and its derived scale mapping.",
            ),
            row(
                "trial2_numeric_alpha_same_sector_retry_active_before_h0p_pivot",
                "pass" if prior_same_sector_retry_active else "reject",
                "same-sector proxy wording retry active before H0^(P)-Z_P pivot",
                1 if prior_same_sector_retry_active else 0,
                "The pivot is meaningful only because the current official blocker still sits on the same-sector proxy wording retry.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "computation_formula_ready": computation_formula_ready,
            "absolute_normalization_dictionary_ready": absolute_normalization_dictionary_ready,
            "h0p_background_law_ready": h0p_background_law_ready,
            "same_sector_retry_active_before_pivot": prior_same_sector_retry_active,
            "advice_h0p_bridge_candidate_ready": advice_bridge_candidate_hit is not None,
            "explicit_h0p_zp_bridge_statement_available": explicit_h0p_zp_bridge_statement_available,
            "explicit_background_factor_available": explicit_background_factor_available,
        },
        {
            "overall_status": "trial2_numeric_alpha_h0p_bridge_pivot_inventory_frozen",
            "advance_to_8_7_56_840": inventory_ready,
            "next_required_artifacts": [],
        },
        {
            "inventory_targets": inventory_targets,
            "advice_bridge_candidate_hit": advice_bridge_candidate_hit,
            "part2_h0p_scale_hit": part2_h0p_scale_hit,
            "current_ai_context_step": ai_context["current_step"],
            "prior_gate_summary": prior_gate_summary,
            "prior_route_summary": prior_route_summary,
        },
    )

    audit_payload = payload(
        "8.7.56.840",
        "Trial-2 numeric alpha H0^(P)-Z_P bridge pivot audit",
        common_inputs,
        "Audit whether the current public canon already contains an explicit H0^(P)-Z_P bridge statement or whether the honest blocker is now that explicit bridge itself.",
        {
            "availability_rule": "numeric alpha can use the H0^(P)-Z_P bridge only once an explicit bridge statement is public-canonical in the paper pack",
            "statement_rule": "the bridge statement may appear either as Z_P = m0^2/(H0^(P))^2, as Z_P = m0^2 C_bg/(H0^(P))^2, or as the equivalent dispersion form Z_P (H0^(P))^2 = ...",
            "pivot_rule": "if the bridge candidate pack exists but the explicit statement is absent, the mainline blocker is the missing H0^(P)-Z_P bridge statement rather than the same-sector proxy wording loop",
        },
        [
            row(
                "trial2_numeric_alpha_h0p_bridge_audit_complete",
                "pass",
                "H0^(P)-Z_P bridge pivot audit complete",
                1,
                "This step decides whether the new bridge candidate already exists in public canon or remains an explicit statement gap.",
            ),
            row(
                "trial2_numeric_alpha_explicit_h0p_zp_bridge_statement_available",
                "pass" if explicit_h0p_zp_bridge_statement_available else "reject",
                "explicit H0^(P)-Z_P bridge statement available",
                1 if explicit_h0p_zp_bridge_statement_available else 0,
                "Current public canon still does not state the H0^(P)-Z_P bridge explicitly.",
            ),
            row(
                "trial2_numeric_alpha_explicit_background_factor_available",
                "pass" if explicit_background_factor_available else "reject",
                "explicit background factor available",
                1 if explicit_background_factor_available else 0,
                "No explicit background damping factor is currently named in the public paper pack.",
            ),
            row(
                "trial2_numeric_alpha_h0p_bridge_candidate_ready_in_audit",
                "pass" if h0p_bridge_candidate_ready else "reject",
                "H0^(P)-Z_P bridge candidate ready in audit",
                1 if h0p_bridge_candidate_ready else 0,
                "The computation formula, electron dictionary, late-time H0^(P), and advice note already form a coherent bridge candidate pack.",
            ),
            row(
                "trial2_numeric_alpha_dominant_blocker_is_h0p_zp_bridge_statement_absence",
                "pass" if dominant_blocker_is_h0p_zp_bridge_statement_absence else "reject",
                "dominant blocker is H0^(P)-Z_P bridge statement absence",
                1 if dominant_blocker_is_h0p_zp_bridge_statement_absence else 0,
                "The same-sector proxy wording loop is no longer the honest blocker once the H0^(P)-Z_P bridge candidate exists without an explicit public statement.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "explicit_h0p_zp_bridge_statement_available": explicit_h0p_zp_bridge_statement_available,
            "explicit_background_factor_available": explicit_background_factor_available,
            "h0p_bridge_candidate_ready": h0p_bridge_candidate_ready,
            "dominant_blocker_is_h0p_zp_bridge_statement_absence": dominant_blocker_is_h0p_zp_bridge_statement_absence,
            "first_route_to_close_after_audit_or_none": "trial2_numeric_alpha_h0p_zp_bridge_pivot_declaration_gate",
        },
        {
            "overall_status": "trial2_numeric_alpha_h0p_bridge_pivot_audited",
            "advance_to_8_7_56_841": True,
            "next_required_artifacts": [],
        },
        {
            "inventory_summary": inventory_payload["summary"],
            "explicit_h0p_zp_bridge_statement_hit": explicit_h0p_zp_bridge_statement_hit,
            "explicit_background_factor_hit": explicit_background_factor_hit,
        },
    )

    gate_payload = payload(
        "8.7.56.841",
        "Trial-2 numeric alpha H0^(P)-Z_P bridge pivot declaration gate",
        common_inputs,
        "Close the pivot honestly: retire the same-sector proxy wording retry from the mainline and reclassify the open issue as a missing explicit H0^(P)-Z_P bridge statement.",
        {
            "pivot_gate_rule": "if the H0^(P)-Z_P bridge candidate pack is coherent but the explicit bridge statement is absent, the mainline should pivot to the bridge statement residual rather than continue the same-sector wording loop",
            "numeric_rule": "numeric alpha remains open until the explicit bridge statement becomes public-canonical",
            "structural_rule": "adopting the H0^(P)-Z_P bridge pivot does not reopen the computation formula or electron-identification absolute-normalization dictionary",
        },
        [
            row(
                "trial2_numeric_alpha_h0p_bridge_pivot_gate_complete",
                "pass",
                "H0^(P)-Z_P bridge pivot declaration gate complete",
                1,
                "The mainline is now allowed to pivot away from the same-sector wording retry.",
            ),
            row(
                "trial2_numeric_alpha_same_sector_retry_retired_from_mainline",
                "pass" if prior_same_sector_retry_active else "reject",
                "same-sector proxy wording retry retired from mainline",
                1 if prior_same_sector_retry_active else 0,
                "The prior blocker family is retired because the bridge candidate directly targets Z_P rather than wording fragments.",
            ),
            row(
                "trial2_numeric_alpha_h0p_bridge_pivot_adopted",
                "pass" if dominant_blocker_is_h0p_zp_bridge_statement_absence else "reject",
                "H0^(P)-Z_P bridge pivot adopted",
                1 if dominant_blocker_is_h0p_zp_bridge_statement_absence else 0,
                "The open issue is now the missing explicit bridge statement itself.",
            ),
            row(
                "trial2_numeric_alpha_numeric_from_current_pack_ready_after_h0p_pivot",
                "pass" if explicit_h0p_zp_bridge_statement_available else "reject",
                "numeric alpha from current pack ready after H0^(P)-Z_P pivot",
                1 if explicit_h0p_zp_bridge_statement_available else 0,
                "Numeric alpha stays open because the public pack still lacks the explicit bridge statement.",
            ),
            row(
                "trial2_numeric_alpha_structural_pass_retained_after_h0p_pivot",
                "pass" if computation_formula_ready and absolute_normalization_dictionary_ready else "reject",
                "structural Trial-2 pass retained after H0^(P)-Z_P pivot",
                1 if computation_formula_ready and absolute_normalization_dictionary_ready else 0,
                "The pivot changes the blocker classification, not the underlying structural pass.",
            ),
        ],
        {
            "trial2_numeric_alpha_computation_formula_ready": computation_formula_ready,
            "trial2_numeric_alpha_absolute_normalization_dictionary_ready": absolute_normalization_dictionary_ready,
            "trial2_numeric_alpha_numeric_from_current_pack_ready": False,
            "trial2_numeric_alpha_closeout_ready": False,
            "same_sector_proxy_equivalence_retry_retired_from_mainline": prior_same_sector_retry_active,
            "h0p_bridge_pivot_adopted": dominant_blocker_is_h0p_zp_bridge_statement_absence,
            "selected_residual_route": NEXT_RESIDUAL_ROUTE,
            "missing_v2_artifact": NEXT_MISSING_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_h0p_bridge_pivot_gate_closed_bridge_statement_open",
            "advance_to_8_7_56_842": True,
            "next_required_artifacts": [NEXT_MISSING_ARTIFACT],
        },
        {
            "audit_summary": audit_payload["summary"],
            "prior_gate_summary": prior_gate_summary,
            "prior_route_summary": prior_route_summary,
        },
    )

    route_payload = payload(
        "8.7.56.842",
        "Trial-2 numeric alpha next-generation route contract one-hundred-seventh refresh",
        common_inputs,
        "Refresh the next-generation contract after the H0^(P)-Z_P bridge pivot: keep precision-alpha on the mainline, keep the strong side on reserve, and promote the explicit bridge statement as the next official blocker.",
        {
            "selected_route_rule": "the next official route is the missing explicit H0^(P)-Z_P bridge statement",
            "mainline_rule": "precision-alpha remains the next-generation mainline after the H0^(P)-Z_P bridge pivot",
            "reserve_rule": "strong-side non-Abelian, running, and confinement gaps remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_h0p_bridge_pivot_gate_closed_before_refresh",
                "pass",
                "H0^(P)-Z_P bridge pivot declaration gate closed before route refresh",
                1,
                "The next-generation contract is refreshed only after the pivot gate closes.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_h0p_zp_bridge_statement",
                "pass",
                "H0^(P)-Z_P bridge statement route selected",
                1,
                "The next route now targets the explicit bridge statement instead of the same-sector wording retry.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_h0p_pivot",
                "pass" if prior_route_summary["precision_alpha_mainline_retained"] else "reject",
                "precision-alpha mainline retained after H0^(P)-Z_P pivot",
                1 if prior_route_summary["precision_alpha_mainline_retained"] else 0,
                "The mainline remains Trial-2 numeric alpha, not the strong-side reserve.",
            ),
            row(
                "trial2_numeric_alpha_strong_side_route_state_retained_after_h0p_pivot",
                "pass" if prior_route_summary["strong_side_route_state"] == "v3_hold_reserve" else "reject",
                "strong-side route state retained after H0^(P)-Z_P pivot",
                1 if prior_route_summary["strong_side_route_state"] == "v3_hold_reserve" else 0,
                "The strong side remains exploratory and is not promoted by the current alpha pivot.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_RESIDUAL_ROUTE,
            "strong_side_route_state": prior_route_summary["strong_side_route_state"],
            "precision_alpha_mainline_retained": bool(prior_route_summary["precision_alpha_mainline_retained"]),
            "electron_identification_pivot_retained": absolute_normalization_dictionary_ready,
            "h0p_bridge_pivot_retained": dominant_blocker_is_h0p_zp_bridge_statement_absence,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_seventh_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [
                "trial2_numeric_alpha_h0p_zp_bridge_statement_source_inventory",
                "trial2_numeric_alpha_h0p_zp_bridge_statement_audit",
            ],
        },
        {
            "gate_summary": gate_payload["summary"],
            "prior_route_summary": prior_route_summary,
        },
    )

    write_artifact("mass_origin_v2_trial2_numeric_alpha_h0p_bridge_source_inventory", inventory_payload)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_h0p_bridge_audit", audit_payload)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_h0p_bridge_declaration_gate", gate_payload)
    write_artifact("mass_origin_v2_t2_alpha_route_contract_one_hundred_seventh_refresh", route_payload)

    print("[done] 8.7.56.839-.842 artifacts generated:")
    print(" - mass_origin_v2_trial2_numeric_alpha_h0p_bridge_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_h0p_bridge_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_h0p_bridge_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_one_hundred_seventh_refresh_metrics.json")
    print(f" - next official branch should move to {NEXT_BRANCH}")


# Function: run the H0^(P)-Z_P bridge pivot branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the Trial-2 numeric alpha H0^(P)-Z_P bridge pivot branch."""
    main()


if __name__ == "__main__":
    run_cli()
