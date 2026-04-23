#!/usr/bin/env python3
"""Generate 8.7.56.431-.434 Part-III-A primary-surface-statement-wording artifacts.

The previous branch established that Part III-A 2.6.1 is the primary surface
for the missing Coulomb-normalization source statement, and that the statement
is absent not only there but also on the fallback and checkpoint surfaces.
This branch freezes the wording-level blocker more tightly by identifying the
current "numeric alpha open" clause as the exact line that still lacks the
needed Coulomb-normalization source link.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

PRIOR_INVENTORY = OUT / "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_source_inventory_metrics.json"
PRIOR_AUDIT = OUT / "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_audit_metrics.json"
PRIOR_GATE = OUT / "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_declaration_gate_metrics.json"
PRIOR_ROUTE = OUT / "mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_fourth_refresh_metrics.json"

STATUS_NEXT_STEP = "current official next step は `8.7.56.431`"
ROADMAP_BRANCH = "`8.7.56.431-.434` Trial-2 numeric $\\alpha$ Coulomb-normalization-source-surface Part-III-A primary-surface statement-wording residual branch"
PART5_NEXT_STEP = "8.7.56.431-.434"
PART3A_PRIMARY_HEADING = "#### 2.6.1 現行 canon で固定した source / structural route"
PART3A_FALLBACK_HEADING = "#### 2.6.2 未導出（近似検証と判定の固定）"
PART3A_NEXT_SECTION_HEADING = "### 2.7"
PART5_SECONDARY_HEADING = "### 3.2 v2.0 checkpoint：electromagnetism / weak-sector closeout（理論側 checkpoint）"
PART1_NO_SURFACE_PATTERN = "epsilon_0"
ALPHA_FORMULA_PATTERN = "$\\alpha=g_P^2/(4\\pi Z_P\\hbar c)$"
NUMERIC_ALPHA_OPEN_CLAUSE = "**foundational / structural pass (numeric α open)**"
SOURCE_LINK_GROUPS = (
    ("normalization", "normalize", "normalise"),
    ("source", "source-surface", "source surface"),
)
CLAUSE_LINK_GROUPS = (
    ("foundational / structural pass", "numeric α open", "numeric alpha open"),
    ("normalization", "normalize", "normalise"),
    ("source", "source-surface", "source surface"),
)

CURRENT_ROUTE = "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_identification"
NEXT_ROUTE = "8.7.56.435"
NEXT_ROUTE_LABEL = "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_identification"
NEXT_MISSING_ARTIFACT = "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause"


# 関数: UTC 現在時刻を ISO 8601 形式で返す。
def now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力 path の存在を確認する。

def req(path: Path) -> None:
    """Abort when a required input path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON artifact を読む。

def read_json(path: Path) -> dict:
    """Load a UTF-8 JSON artifact into a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: UTF-8 text source を読む。

def read_text(path: Path) -> str:
    """Read a UTF-8 text source into memory."""
    return path.read_text(encoding="utf-8")


# 関数: repo 相対 POSIX path を返す。

def rel(path: Path) -> str:
    """Return a repository-relative POSIX path string."""
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: 指定した部分文字列の最初の hit 行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line hit for a substring pattern, if any."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: start/end 見出しで挟まれた section text を返す。

def section_text(text: str, start_pattern: str, end_patterns: tuple[str, ...]) -> str:
    """Return the text contained inside a heading-delimited section."""
    start = text.find(start_pattern)
    if start < 0:
        return ""

    start += len(start_pattern)
    end_candidates = [text.find(pattern, start) for pattern in end_patterns]
    valid_ends = [candidate for candidate in end_candidates if candidate >= 0]
    end = min(valid_ends) if valid_ends else len(text)
    return text[start:end]


# 関数: 複数 token group を同時に満たす最初の hit 行を返す。

def multi_hit(text: str, groups: tuple[tuple[str, ...], ...]) -> dict | None:
    """Return the first line that contains at least one token from every group."""
    lowered_groups = tuple(tuple(token.lower() for token in group) for group in groups)
    for line_no, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        matched: list[str] = []
        for group, lowered_group in zip(groups, lowered_groups):
            token = next((src for src, lowered_token in zip(group, lowered_group) if lowered_token in lowered), None)
            if token is None:
                break

            matched.append(token)
        else:
            return {"groups": matched, "line": line_no, "text": line.strip()}

    return None


# 関数: 共通 schema の row を組み立てる。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build a standard metrics-row payload."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 共通 schema の payload を組み立てる。

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
    """Build the standard JSON metrics payload used across the roadmap."""
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


# 関数: JSON artifact と rows CSV を side-by-side で保存する。

def write_artifact(stem: str, data: dict) -> None:
    """Write the metrics payload as JSON and as a rows CSV sidecar."""
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# 関数: wording residual の source inventory を構築する。

def build_inventory(
    common_inputs: dict,
    prior_inventory: dict,
    prior_audit: dict,
    prior_gate: dict,
    prior_route: dict,
    status_text: str,
    roadmap_text: str,
    part1_text: str,
    part3a_text: str,
    part5_text: str,
) -> dict:
    """Freeze the source inventory for the wording-level residual."""
    primary_section = section_text(part3a_text, PART3A_PRIMARY_HEADING, (PART3A_FALLBACK_HEADING, PART3A_NEXT_SECTION_HEADING))
    fallback_section = section_text(part3a_text, PART3A_FALLBACK_HEADING, (PART3A_NEXT_SECTION_HEADING,))

    targets = [
        {"name": "status_next_step", "hit": hit(status_text, STATUS_NEXT_STEP), "note": "STATUS must point to 8.7.56.431."},
        {"name": "roadmap_branch", "hit": hit(roadmap_text, ROADMAP_BRANCH), "note": "ROADMAP must advertise 8.7.56.431-.434 as current branch."},
        {"name": "part5_next_step", "hit": hit(part5_text, PART5_NEXT_STEP), "note": "Part V must point to the wording branch."},
        {"name": "part3a_primary_heading", "hit": hit(part3a_text, PART3A_PRIMARY_HEADING), "note": "Part III-A 2.6.1 must remain available as primary surface."},
        {"name": "part3a_alpha_formula_line", "hit": hit(primary_section, ALPHA_FORMULA_PATTERN), "note": "The structural alpha formula line must remain present on the primary surface."},
        {"name": "part3a_numeric_alpha_open_clause", "hit": hit(primary_section, NUMERIC_ALPHA_OPEN_CLAUSE), "note": "The current primary-surface open-status clause must remain explicit."},
        {"name": "part3a_fallback_heading", "hit": hit(part3a_text, PART3A_FALLBACK_HEADING), "note": "Part III-A 2.6.2 must remain available as fallback surface."},
        {"name": "part5_secondary_heading", "hit": hit(part5_text, PART5_SECONDARY_HEADING), "note": "Part V 3.2 must remain available as checkpoint-only secondary surface."},
    ]

    inventory_ready = all(item["hit"] is not None for item in targets)
    part1_no_surface_preserved = bool(hit(part1_text, PART1_NO_SURFACE_PATTERN) is None)
    fallback_no_statement_preserved = bool(multi_hit(fallback_section, CLAUSE_LINK_GROUPS) is None)
    secondary_no_statement_preserved = bool(multi_hit(part5_text, CLAUSE_LINK_GROUPS) is None)
    route_contract_consistent = bool(
        prior_gate["summary"]["selected_residual_route"] == CURRENT_ROUTE
        and prior_route["summary"]["selected_next_generation_route"] == CURRENT_ROUTE
    )

    return payload(
        "8.7.56.431",
        "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_source_inventory",
        common_inputs,
        "Inventory the wording-level residual pack around the Part-III-A 2.6.1 primary section, the structural alpha formula line, the explicit 'numeric alpha open' clause, the fallback / secondary no-statement evidence, and the current route contract.",
        {
            "inventory_rule": "The branch starts only after the prior statement branch has frozen that the required statement is absent on every visible candidate surface.",
            "primary_surface_rule": "Part III-A 2.6.1 remains the primary technical surface because it already carries the structural alpha formula and the current open-status clause.",
            "wording_rule": "The remaining question is no longer whether the primary statement exists, but which wording element inside the existing open-status clause is still missing.",
        },
        [
            row(
                "trial2_numeric_alpha_part3a_primary_surface_statement_wording_inventory_targets_present",
                "pass" if inventory_ready else "reject",
                "primary-surface statement-wording inventory targets present",
                sum(1 for item in targets if item["hit"] is not None),
                "Control docs, primary section, formula line, and open-status clause must align on the wording branch.",
            ),
            row(
                "trial2_numeric_alpha_part1_no_surface_preserved",
                "pass" if part1_no_surface_preserved else "reject",
                "Part I no-surface status preserved",
                1 if part1_no_surface_preserved else 0,
                "Part I still carries no direct Coulomb-normalization source surface for numeric alpha.",
            ),
            row(
                "trial2_numeric_alpha_part3a_fallback_no_statement_preserved",
                "pass" if fallback_no_statement_preserved else "reject",
                "Part III-A fallback no-statement evidence preserved",
                1 if fallback_no_statement_preserved else 0,
                "Part III-A 2.6.2 still carries no normalization/source statement wording.",
            ),
            row(
                "trial2_numeric_alpha_part5_secondary_no_statement_preserved",
                "pass" if secondary_no_statement_preserved else "reject",
                "Part V secondary no-statement evidence preserved",
                1 if secondary_no_statement_preserved else 0,
                "Part V 3.2 still carries no normalization/source statement wording.",
            ),
            row(
                "trial2_numeric_alpha_part3a_primary_surface_statement_wording_route_contract_consistent",
                "pass" if route_contract_consistent else "reject",
                "primary-surface statement-wording route contract consistent",
                1 if route_contract_consistent else 0,
                "The prior declaration gate and route contract must agree on the current wording residual route.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "part1_no_surface_preserved": part1_no_surface_preserved,
            "part3a_fallback_no_statement_preserved": fallback_no_statement_preserved,
            "part5_secondary_no_statement_preserved": secondary_no_statement_preserved,
            "route_contract_consistent": route_contract_consistent,
            "first_route_to_close_or_none": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_audit",
        },
        {
            "overall_status": "trial2_numeric_alpha_part3a_primary_surface_statement_wording_inventory_frozen"
            if inventory_ready
            else "trial2_numeric_alpha_part3a_primary_surface_statement_wording_inventory_incomplete",
            "advance_to_8_7_56_432": inventory_ready,
            "next_required_artifacts": []
            if inventory_ready
            else ["trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_source_inventory"],
        },
        {
            "targets": targets,
            "prior_inventory_summary": prior_inventory["summary"],
            "prior_audit_summary": prior_audit["summary"],
            "prior_gate_summary": prior_gate["summary"],
            "prior_route_summary": prior_route["summary"],
        },
    )


# 関数: wording residual の audit を構築する。

def build_audit(
    common_inputs: dict,
    inventory: dict,
    part3a_text: str,
) -> dict:
    """Audit which exact clause still carries the missing wording blocker."""
    primary_section = section_text(part3a_text, PART3A_PRIMARY_HEADING, (PART3A_FALLBACK_HEADING, PART3A_NEXT_SECTION_HEADING))
    alpha_formula_line = hit(primary_section, ALPHA_FORMULA_PATTERN)
    numeric_alpha_open_clause_line = hit(primary_section, NUMERIC_ALPHA_OPEN_CLAUSE)
    open_clause_has_source_link = bool(
        numeric_alpha_open_clause_line is not None and multi_hit(numeric_alpha_open_clause_line["text"], SOURCE_LINK_GROUPS) is not None
    )
    open_clause_present = bool(numeric_alpha_open_clause_line is not None)
    alpha_formula_present = bool(alpha_formula_line is not None)
    dominant_blocker = bool(
        inventory["summary"]["inventory_ready"]
        and alpha_formula_present
        and open_clause_present
        and not open_clause_has_source_link
    )
    audit_ready = dominant_blocker

    return payload(
        "8.7.56.432",
        "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_audit",
        common_inputs,
        "Audit the exact wording-level blocker on the Part-III-A 2.6.1 primary surface by checking whether the existing 'numeric alpha open' clause already contains the missing Coulomb-normalization source link.",
        {
            "formula_rule": "The structural alpha formula line remains fixed and is not the blocker by itself.",
            "open_clause_rule": "The current 'foundational / structural pass (numeric alpha open)' clause is the exact line that still marks the unresolved precision state.",
            "source_link_rule": "The remaining blocker is whether that open-status clause already contains an explicit Coulomb-normalization source link.",
        },
        [
            row(
                "trial2_numeric_alpha_part3a_alpha_formula_line_present",
                "pass" if alpha_formula_present else "reject",
                "Part III-A alpha formula line present",
                1 if alpha_formula_present else 0,
                "The structural alpha formula remains present on the primary surface.",
            ),
            row(
                "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_present",
                "pass" if open_clause_present else "reject",
                "Part III-A numeric alpha open clause present",
                1 if open_clause_present else 0,
                "The current open-status clause remains explicit on the primary surface.",
            ),
            row(
                "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_contains_source_link",
                "pass" if open_clause_has_source_link else "reject",
                "numeric alpha open clause contains normalization-source link",
                1 if open_clause_has_source_link else 0,
                "The open-status clause still carries no explicit Coulomb-normalization source link.",
            ),
            row(
                "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_source_link_absence_dominant_blocker",
                "pass" if dominant_blocker else "reject",
                "numeric alpha open clause source-link absence is dominant blocker",
                1 if dominant_blocker else 0,
                "The residual now shrinks from generic wording absence to the missing source-link clause on the explicit open-status line.",
            ),
        ],
        {
            "audit_ready": audit_ready,
            "part3a_alpha_formula_line_present": alpha_formula_present,
            "part3a_numeric_alpha_open_clause_present": open_clause_present,
            "part3a_numeric_alpha_open_clause_contains_source_link": open_clause_has_source_link,
            "dominant_blocker_is_part3a_numeric_alpha_open_clause_source_link_absence": dominant_blocker,
            "first_route_to_close_or_none": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_declaration_gate",
        },
        {
            "overall_status": "trial2_numeric_alpha_part3a_primary_surface_statement_wording_audit_complete"
            if audit_ready
            else "trial2_numeric_alpha_part3a_primary_surface_statement_wording_audit_incomplete",
            "advance_to_8_7_56_433": audit_ready,
            "next_required_artifacts": []
            if audit_ready
            else ["trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_audit"],
        },
        {
            "inventory_summary": inventory["summary"],
            "alpha_formula_line": alpha_formula_line,
            "numeric_alpha_open_clause_line": numeric_alpha_open_clause_line,
        },
    )


# 関数: declaration gate artifact を構築する。

def build_gate(common_inputs: dict, audit: dict) -> dict:
    """Freeze the declaration gate after the wording-level audit."""
    audit_ready = bool(audit["summary"]["audit_ready"])

    return payload(
        "8.7.56.433",
        "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_declaration_gate",
        common_inputs,
        "Freeze the declaration gate after confirming that the explicit 'numeric alpha open' clause is the exact line still missing the Coulomb-normalization source link.",
        {
            "gate_rule": "The current branch closes once the dominant blocker is localized to the explicit open-status clause.",
            "residual_rule": "The next residual concerns the clause itself rather than the broader primary-surface wording family.",
        },
        [
            row(
                "trial2_numeric_alpha_part3a_primary_surface_statement_wording_gate_complete",
                "pass" if audit_ready else "reject",
                "primary-surface statement-wording gate complete",
                1 if audit_ready else 0,
                "The gate closes once the dominant blocker is narrowed to the explicit open-status clause.",
            ),
            row(
                "trial2_numeric_alpha_closeout_ready",
                "reject",
                "numeric-alpha closeout ready",
                0,
                "Numeric-alpha closeout still requires the missing source-link clause on the primary surface.",
            ),
            row(
                "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_missing",
                "pass" if audit_ready else "watch",
                "Part III-A numeric alpha open clause source link missing",
                1 if audit_ready else 0,
                "The narrowed blocker is the missing source-link clause on the explicit numeric-alpha-open line.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_preserved",
                "pass",
                "precision-alpha mainline preserved",
                1,
                "The branch does not reopen the structural EM pass or promote the strong-side reserve.",
            ),
        ],
        {
            "trial2_numeric_alpha_part3a_primary_surface_statement_wording_branch_closeable": audit_ready,
            "trial2_numeric_alpha_closeout_ready": False,
            "selected_residual_route": NEXT_ROUTE_LABEL,
            "missing_v2_artifact": NEXT_MISSING_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_part3a_primary_surface_statement_wording_gate_closed"
            if audit_ready
            else "trial2_numeric_alpha_part3a_primary_surface_statement_wording_gate_open",
            "advance_to_8_7_56_434": audit_ready,
            "next_required_artifacts": []
            if audit_ready
            else ["trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_declaration_gate"],
        },
        {
            "audit_summary": audit["summary"],
        },
    )


# 関数: strong-side reserve refresh / route contract fifth refresh を構築する。

def build_contract(common_inputs: dict, gate: dict, prior_route: dict) -> dict:
    """Refresh the strong-side reserve and freeze the next clause-level route."""
    gate_closed = bool(gate["summary"]["trial2_numeric_alpha_part3a_primary_surface_statement_wording_branch_closeable"])
    residual_route_selected = gate["summary"]["selected_residual_route"] == NEXT_ROUTE_LABEL

    return payload(
        "8.7.56.434",
        "trial2_numeric_alpha_next_generation_route_contract_fifth_refresh",
        common_inputs,
        "Refresh the strong-side reserve after the wording branch and freeze the next clause-level EM precision residual route.",
        {
            "contract_rule": "The EM precision mainline remains active while the blocker shrinks from primary-surface wording absence to the missing source-link clause on the explicit open-status line.",
            "reserve_rule": "Strong-side work remains on v3 hold reserve and does not outrank the EM precision route.",
        },
        [
            row(
                "trial2_numeric_alpha_part3a_primary_surface_statement_wording_gate_closed",
                "pass" if gate_closed else "reject",
                "primary-surface statement-wording gate closed",
                1 if gate_closed else 0,
                "The next route contract depends on the wording gate being frozen first.",
            ),
            row(
                "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_route_selected",
                "pass" if residual_route_selected else "reject",
                "numeric alpha open clause route selected",
                1 if residual_route_selected else 0,
                "The next official route stays inside the EM precision program.",
            ),
            row(
                "trial2_numeric_alpha_strong_side_reserve_retained",
                "pass",
                "strong-side reserve retained",
                1,
                "Strong-side non-Abelian/running/confinement gaps remain on reserve.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained",
                "pass",
                "precision-alpha mainline retained",
                1,
                "The first next-generation mainline remains the Trial-2 numeric-alpha program.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_ROUTE_LABEL if residual_route_selected else None,
            "strong_side_route_state": prior_route["summary"]["strong_side_route_state"],
            "precision_alpha_mainline_retained": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_next_route_contract_fifth_refresh_frozen"
            if gate_closed
            else "trial2_numeric_alpha_next_route_contract_fifth_refresh_pending",
            "advance_to_next_route": gate_closed,
            "next_required_artifacts": [
                "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_inventory",
                "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_audit",
            ]
            if gate_closed
            else ["trial2_numeric_alpha_next_generation_route_contract_fifth_refresh"],
        },
        {
            "gate_summary": gate["summary"],
            "prior_route_summary": prior_route["summary"],
        },
    )


# 関数: current branch を実行する。

def main() -> None:
    """Execute the Trial-2 numeric-alpha Part-III-A primary-surface-statement-wording branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PART1,
        PART3A,
        PART5,
        PRIOR_INVENTORY,
        PRIOR_AUDIT,
        PRIOR_GATE,
        PRIOR_ROUTE,
    ):
        req(path)

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "part1_core_theory_markdown": rel(PART1),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "part5_future_predictions_markdown": rel(PART5),
        "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_source_inventory_json": rel(PRIOR_INVENTORY),
        "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_audit_json": rel(PRIOR_AUDIT),
        "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_declaration_gate_json": rel(PRIOR_GATE),
        "mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_fourth_refresh_json": rel(PRIOR_ROUTE),
    }

    prior_inventory = read_json(PRIOR_INVENTORY)
    prior_audit = read_json(PRIOR_AUDIT)
    prior_gate = read_json(PRIOR_GATE)
    prior_route = read_json(PRIOR_ROUTE)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)

    inventory = build_inventory(
        common_inputs,
        prior_inventory,
        prior_audit,
        prior_gate,
        prior_route,
        status_text,
        roadmap_text,
        part1_text,
        part3a_text,
        part5_text,
    )
    audit = build_audit(common_inputs, inventory, part3a_text)
    gate = build_gate(common_inputs, audit)
    contract = build_contract(common_inputs, gate, prior_route)

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_declaration_gate",
        gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_fifth_refresh",
        contract,
    )

    print("[ok] generated Trial-2 numeric alpha Part-III-A primary-surface-statement-wording artifacts:")
    print(" - mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_declaration_gate_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_fifth_refresh_metrics.json")


# 関数: CLI 直実行時に branch main を起動する。

def run_cli() -> None:
    """CLI entry point for the Part-III-A primary-surface-statement-wording branch."""
    main()


if __name__ == "__main__":
    run_cli()
