#!/usr/bin/env python3
"""Generate 8.7.56.435-.438 numeric-alpha-open-clause residual artifacts.

The previous branch froze that Part III-A 2.6.1 already contains the explicit
"foundational / structural pass (numeric alpha open)" line, while that same
line still lacks an explicit Coulomb-normalization source link. This branch
promotes the clause itself into the official residual target and freezes the
next clause-level route contract.
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

PRIOR_INVENTORY = OUT / "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_source_inventory_metrics.json"
PRIOR_AUDIT = OUT / "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_audit_metrics.json"
PRIOR_GATE = OUT / "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_declaration_gate_metrics.json"
PRIOR_ROUTE = OUT / "mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_fifth_refresh_metrics.json"

STATUS_NEXT_STEP = "current official next step は `8.7.56.435`"
ROADMAP_BRANCH = "`8.7.56.435-.438` Trial-2 numeric $\\alpha$ Coulomb-normalization-source-surface Part-III-A numeric-α-open-clause residual branch"
PART5_NEXT_STEP = "8.7.56.435-.438"
PART3A_PRIMARY_HEADING = "#### 2.6.1 現行 canon で固定した source / structural route"
PART3A_FALLBACK_HEADING = "#### 2.6.2 未導出（近似検証と判定の固定）"
PART3A_NEXT_SECTION_HEADING = "### 2.7"
PART5_SECONDARY_HEADING = "### 3.2 v2.0 checkpoint：electromagnetism / weak-sector closeout（理論側 checkpoint）"
ALPHA_FORMULA_PATTERN = "$\\alpha=g_P^2/(4\\pi Z_P\\hbar c)$"
NUMERIC_ALPHA_OPEN_CLAUSE = "**foundational / structural pass (numeric α open)**"
PART1_NO_SURFACE_PATTERN = "epsilon_0"
SOURCE_LINK_GROUPS = (
    ("normalization", "normalize", "normalise"),
    ("source", "source-surface", "source surface"),
)
SOURCE_LINK_CLAUSE_GROUPS = (
    ("foundational / structural pass", "numeric α open", "numeric alpha open"),
    ("normalization", "normalize", "normalise"),
    ("source", "source-surface", "source surface"),
    ("fixed", "ready", "available", "machine-readable", "固定", "固定済み", "導出", "定義"),
)

CURRENT_ROUTE = "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_identification"
NEXT_ROUTE = "8.7.56.439"
NEXT_ROUTE_LABEL = "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_link_clause_identification"
NEXT_MISSING_ARTIFACT = "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_link_clause"


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


# 関数: clause residual の source inventory を構築する。

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
    """Freeze the source inventory for the numeric-alpha-open-clause residual."""
    primary_section = section_text(part3a_text, PART3A_PRIMARY_HEADING, (PART3A_FALLBACK_HEADING, PART3A_NEXT_SECTION_HEADING))
    fallback_section = section_text(part3a_text, PART3A_FALLBACK_HEADING, (PART3A_NEXT_SECTION_HEADING,))
    open_clause_line = hit(primary_section, NUMERIC_ALPHA_OPEN_CLAUSE)
    source_link_clause_line = multi_hit(primary_section, SOURCE_LINK_CLAUSE_GROUPS)

    targets = [
        {"name": "status_next_step", "hit": hit(status_text, STATUS_NEXT_STEP), "note": "STATUS must point to 8.7.56.435."},
        {"name": "roadmap_branch", "hit": hit(roadmap_text, ROADMAP_BRANCH), "note": "ROADMAP must advertise 8.7.56.435-.438 as current branch."},
        {"name": "part5_next_step", "hit": hit(part5_text, PART5_NEXT_STEP), "note": "Part V must point to the numeric-alpha-open-clause branch."},
        {"name": "part3a_primary_heading", "hit": hit(part3a_text, PART3A_PRIMARY_HEADING), "note": "Part III-A 2.6.1 must remain available as primary surface."},
        {"name": "part3a_alpha_formula_line", "hit": hit(primary_section, ALPHA_FORMULA_PATTERN), "note": "The structural alpha formula line must remain present on the primary surface."},
        {"name": "part3a_numeric_alpha_open_clause", "hit": open_clause_line, "note": "The current primary-surface open-status clause must remain explicit."},
        {"name": "part3a_missing_source_link_clause", "hit": source_link_clause_line, "note": "The primary surface still carries no explicit source-link clause on that open-status line."},
        {"name": "part3a_fallback_heading", "hit": hit(part3a_text, PART3A_FALLBACK_HEADING), "note": "Part III-A 2.6.2 must remain available as fallback surface."},
        {"name": "part5_secondary_heading", "hit": hit(part5_text, PART5_SECONDARY_HEADING), "note": "Part V 3.2 must remain available as checkpoint-only secondary surface."},
    ]

    inventory_ready = all(item["hit"] is not None for item in targets if item["name"] != "part3a_missing_source_link_clause") and source_link_clause_line is None
    part1_no_surface_preserved = hit(part1_text, PART1_NO_SURFACE_PATTERN) is None
    fallback_no_statement_preserved = multi_hit(fallback_section, SOURCE_LINK_CLAUSE_GROUPS) is None
    secondary_no_statement_preserved = multi_hit(part5_text, SOURCE_LINK_CLAUSE_GROUPS) is None
    route_contract_consistent = (
        prior_gate["summary"]["selected_residual_route"] == CURRENT_ROUTE
        and prior_route["summary"]["selected_next_generation_route"] == CURRENT_ROUTE
    )

    return payload(
        "8.7.56.435",
        "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_inventory",
        common_inputs,
        "Inventory the clause-level residual pack around the explicit Part-III-A 2.6.1 numeric-alpha-open line, the structural alpha formula line, the missing source-link clause, the fallback / secondary no-statement evidence, and the current route contract.",
        {
            "inventory_rule": "The branch starts only after the prior wording branch has frozen that the explicit numeric-alpha-open line exists while still lacking the source link.",
            "clause_rule": "The remaining question is no longer whether the open-status line exists, but which source-link clause is absent from that exact line.",
            "surface_rule": "Part III-A 2.6.1 remains the primary technical surface because it already carries both the structural alpha formula and the explicit open-status line.",
        },
        [
            row(
                "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_inventory_targets_present",
                "pass" if inventory_ready else "reject",
                "numeric-alpha-open-clause inventory targets present",
                sum(1 for item in targets if item["hit"] is not None),
                "Control docs, the explicit open-status line, and the no-source-link evidence must align on the clause branch.",
            ),
            row(
                "trial2_numeric_alpha_part3a_missing_source_link_clause_absent",
                "pass" if source_link_clause_line is None else "reject",
                "missing source-link clause absent on primary surface",
                1 if source_link_clause_line is None else 0,
                "The explicit numeric-alpha-open line still carries no Coulomb-normalization source-link clause.",
            ),
            row(
                "trial2_numeric_alpha_part3a_fallback_no_statement_preserved",
                "pass" if fallback_no_statement_preserved else "reject",
                "Part III-A fallback no-statement evidence preserved",
                1 if fallback_no_statement_preserved else 0,
                "Part III-A 2.6.2 still carries no normalization/source clause.",
            ),
            row(
                "trial2_numeric_alpha_part5_secondary_no_statement_preserved",
                "pass" if secondary_no_statement_preserved else "reject",
                "Part V secondary no-statement evidence preserved",
                1 if secondary_no_statement_preserved else 0,
                "Part V 3.2 still carries no normalization/source clause.",
            ),
            row(
                "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_route_contract_consistent",
                "pass" if route_contract_consistent else "reject",
                "numeric-alpha-open-clause route contract consistent",
                1 if route_contract_consistent else 0,
                "The prior declaration gate and route contract must agree on the current clause-level residual route.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "part1_no_surface_preserved": part1_no_surface_preserved,
            "part3a_fallback_no_statement_preserved": fallback_no_statement_preserved,
            "part5_secondary_no_statement_preserved": secondary_no_statement_preserved,
            "part3a_missing_source_link_clause_present": source_link_clause_line is not None,
            "route_contract_consistent": route_contract_consistent,
            "first_route_to_close_or_none": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_audit",
        },
        {
            "overall_status": "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_inventory_frozen"
            if inventory_ready
            else "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_inventory_incomplete",
            "advance_to_8_7_56_436": inventory_ready,
            "next_required_artifacts": []
            if inventory_ready
            else ["trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_inventory"],
        },
        {
            "targets": targets,
            "prior_inventory_summary": prior_inventory["summary"],
            "prior_audit_summary": prior_audit["summary"],
            "prior_gate_summary": prior_gate["summary"],
            "prior_route_summary": prior_route["summary"],
        },
    )


# 関数: clause residual の audit を構築する。

def build_audit(common_inputs: dict, inventory: dict, part3a_text: str) -> dict:
    """Audit the exact missing source-link clause on the explicit open-status line."""
    primary_section = section_text(part3a_text, PART3A_PRIMARY_HEADING, (PART3A_FALLBACK_HEADING, PART3A_NEXT_SECTION_HEADING))
    alpha_formula_line = hit(primary_section, ALPHA_FORMULA_PATTERN)
    numeric_alpha_open_clause_line = hit(primary_section, NUMERIC_ALPHA_OPEN_CLAUSE)
    source_link_clause_line = multi_hit(primary_section, SOURCE_LINK_CLAUSE_GROUPS)

    alpha_formula_present = alpha_formula_line is not None
    open_clause_present = numeric_alpha_open_clause_line is not None
    source_link_clause_present = source_link_clause_line is not None
    dominant_blocker = bool(
        inventory["summary"]["inventory_ready"]
        and alpha_formula_present
        and open_clause_present
        and not source_link_clause_present
    )
    audit_ready = dominant_blocker

    return payload(
        "8.7.56.436",
        "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_audit",
        common_inputs,
        "Audit the explicit Part-III-A 2.6.1 numeric-alpha-open line by checking whether the missing Coulomb-normalization source-link clause is present anywhere on that exact clause surface.",
        {
            "formula_rule": "The structural alpha formula line remains fixed and is not the blocker by itself.",
            "open_clause_rule": "The explicit 'foundational / structural pass (numeric alpha open)' line is the exact clause that still marks the unresolved precision state.",
            "clause_rule": "The remaining blocker is whether that clause already carries the missing Coulomb-normalization source-link wording.",
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
                "The explicit open-status clause remains present on the primary surface.",
            ),
            row(
                "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_source_link_clause_present",
                "pass" if source_link_clause_present else "reject",
                "numeric alpha open clause source-link clause present",
                1 if source_link_clause_present else 0,
                "The explicit open-status clause still carries no Coulomb-normalization source-link clause.",
            ),
            row(
                "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_source_link_clause_absence_dominant_blocker",
                "pass" if dominant_blocker else "reject",
                "numeric alpha open clause source-link clause absence is dominant blocker",
                1 if dominant_blocker else 0,
                "The residual now shrinks from generic source-link absence to the missing source-link clause on the explicit open-status line itself.",
            ),
        ],
        {
            "audit_ready": audit_ready,
            "part3a_alpha_formula_line_present": alpha_formula_present,
            "part3a_numeric_alpha_open_clause_present": open_clause_present,
            "part3a_numeric_alpha_open_clause_source_link_clause_present": source_link_clause_present,
            "dominant_blocker_is_part3a_numeric_alpha_open_clause_source_link_clause_absence": dominant_blocker,
            "first_route_to_close_or_none": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_declaration_gate",
        },
        {
            "overall_status": "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_audit_complete"
            if audit_ready
            else "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_audit_incomplete",
            "advance_to_8_7_56_437": audit_ready,
            "next_required_artifacts": []
            if audit_ready
            else ["trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_audit"],
        },
        {
            "inventory_summary": inventory["summary"],
            "alpha_formula_line": alpha_formula_line,
            "numeric_alpha_open_clause_line": numeric_alpha_open_clause_line,
            "source_link_clause_line": source_link_clause_line,
        },
    )


# 関数: declaration gate artifact を構築する。

def build_gate(common_inputs: dict, audit: dict) -> dict:
    """Freeze the declaration gate after the clause-level audit."""
    audit_ready = bool(audit["summary"]["audit_ready"])

    return payload(
        "8.7.56.437",
        "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_declaration_gate",
        common_inputs,
        "Freeze the declaration gate after confirming that the explicit numeric-alpha-open clause is still missing the Coulomb-normalization source-link clause.",
        {
            "gate_rule": "The current branch closes once the dominant blocker is localized to the missing source-link clause on the explicit open-status line.",
            "residual_rule": "The next residual concerns the source-link clause itself rather than the broader clause-level identification family.",
        },
        [
            row(
                "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_gate_complete",
                "pass" if audit_ready else "reject",
                "numeric-alpha-open-clause gate complete",
                1 if audit_ready else 0,
                "The gate closes once the dominant blocker is narrowed to the missing source-link clause on the explicit open-status line.",
            ),
            row(
                "trial2_numeric_alpha_closeout_ready",
                "reject",
                "numeric-alpha closeout ready",
                0,
                "Numeric-alpha closeout still requires the missing source-link clause on the primary surface.",
            ),
            row(
                "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_source_link_clause_missing",
                "pass" if audit_ready else "watch",
                "Part III-A numeric alpha open clause source-link clause missing",
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
            "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_branch_closeable": audit_ready,
            "trial2_numeric_alpha_closeout_ready": False,
            "selected_residual_route": NEXT_ROUTE_LABEL,
            "missing_v2_artifact": NEXT_MISSING_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_gate_closed"
            if audit_ready
            else "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_gate_open",
            "advance_to_8_7_56_438": audit_ready,
            "next_required_artifacts": []
            if audit_ready
            else ["trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_declaration_gate"],
        },
        {
            "audit_summary": audit["summary"],
        },
    )


# 関数: strong-side reserve refresh / route contract sixth refresh を構築する。

def build_contract(common_inputs: dict, gate: dict, prior_route: dict) -> dict:
    """Refresh the strong-side reserve and freeze the next source-link-clause route."""
    gate_closed = bool(gate["summary"]["trial2_numeric_alpha_part3a_numeric_alpha_open_clause_branch_closeable"])
    residual_route_selected = gate["summary"]["selected_residual_route"] == NEXT_ROUTE_LABEL

    return payload(
        "8.7.56.438",
        "trial2_numeric_alpha_next_generation_route_contract_sixth_refresh",
        common_inputs,
        "Refresh the strong-side reserve after the numeric-alpha-open-clause branch and freeze the next source-link-clause EM precision residual route.",
        {
            "contract_rule": "The EM precision mainline remains active while the blocker shrinks from generic clause-level identification to the missing source-link clause on the explicit open-status line.",
            "reserve_rule": "Strong-side work remains on v3 hold reserve and does not outrank the EM precision route.",
        },
        [
            row(
                "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_gate_closed",
                "pass" if gate_closed else "reject",
                "numeric-alpha-open-clause gate closed",
                1 if gate_closed else 0,
                "The next route contract depends on the clause gate being frozen first.",
            ),
            row(
                "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_source_link_clause_route_selected",
                "pass" if residual_route_selected else "reject",
                "numeric alpha open clause source-link-clause route selected",
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
            "overall_status": "trial2_numeric_alpha_next_route_contract_sixth_refresh_frozen"
            if gate_closed
            else "trial2_numeric_alpha_next_route_contract_sixth_refresh_pending",
            "advance_to_next_route": gate_closed,
            "next_required_artifacts": [
                "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_link_clause_source_inventory",
                "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_link_clause_audit",
            ]
            if gate_closed
            else ["trial2_numeric_alpha_next_generation_route_contract_sixth_refresh"],
        },
        {
            "gate_summary": gate["summary"],
            "prior_route_summary": prior_route["summary"],
        },
    )


# 関数: current branch を実行する。

def main() -> None:
    """Execute the Trial-2 numeric-alpha Part-III-A numeric-alpha-open-clause branch."""
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
        "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_source_inventory_json": rel(PRIOR_INVENTORY),
        "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_audit_json": rel(PRIOR_AUDIT),
        "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_wording_declaration_gate_json": rel(PRIOR_GATE),
        "mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_fifth_refresh_json": rel(PRIOR_ROUTE),
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
        "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_declaration_gate",
        gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_sixth_refresh",
        contract,
    )

    print("[ok] generated Trial-2 numeric alpha Part-III-A numeric-alpha-open-clause artifacts:")
    print(" - mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_declaration_gate_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_sixth_refresh_metrics.json")


# 関数: CLI 直実行時に branch main を起動する。

def run_cli() -> None:
    """CLI entry point for the Part-III-A numeric-alpha-open-clause branch."""
    main()


if __name__ == "__main__":
    run_cli()
