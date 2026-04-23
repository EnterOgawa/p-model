#!/usr/bin/env python3
"""Generate 8.7.56.443-.446 source-link-clause-wording residual artifacts."""

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

PRIOR_INVENTORY = OUT / "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_link_clause_source_inventory_metrics.json"
PRIOR_AUDIT = OUT / "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_link_clause_audit_metrics.json"
PRIOR_GATE = OUT / "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_link_clause_declaration_gate_metrics.json"
PRIOR_ROUTE = OUT / "mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_seventh_refresh_metrics.json"

STATUS_NEXT_STEP = "current official next step は `8.7.56.443`"
ROADMAP_BRANCH = "`8.7.56.443-.446` Trial-2 numeric $\\alpha$ Coulomb-normalization-source-surface Part-III-A numeric-α-open-clause source-link-clause wording residual branch"
PART5_NEXT_STEP = "8.7.56.443-.446"
PRIMARY_HEADING = "#### 2.6.1 現行 canon で固定した source / structural route"
FALLBACK_HEADING = "#### 2.6.2 未導出（近似検証と判定の固定）"
PART3A_NEXT = "### 2.7"
PART5_SECONDARY = "### 3.2 v2.0 checkpoint：electromagnetism / weak-sector closeout（理論側 checkpoint）"
PART5_NEXT = "## 4."
ALPHA_FORMULA = "$\\alpha=g_P^2/(4\\pi Z_P\\hbar c)$"
OPEN_CLAUSE = "**foundational / structural pass (numeric α open)**"
POSITIVE_GROUPS = (
    ("coulomb", "Coulomb"),
    ("normalization", "normalize", "normalise"),
    ("source", "source-surface", "source surface"),
    ("independent", "independently", "independent source", "独立"),
)

CURRENT_ROUTE = "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_link_clause_wording_identification"
NEXT_ROUTE = "8.7.56.447"
NEXT_ROUTE_LABEL = "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_family_identification"
NEXT_MISSING_ARTIFACT = "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_family"


# 関数: UTC 現在時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: UTF-8 テキストを読む。

def read_text(path: Path) -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# 関数: UTF-8 JSON を読む。

def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: 最初の部分文字列 hit を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line hit for a substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 見出しに挟まれた section text を返す。

def section_text(text: str, start_pattern: str, end_patterns: tuple[str, ...]) -> str:
    """Return the text inside a heading-delimited section."""
    start = text.find(start_pattern)
    if start < 0:
        return ""

    start += len(start_pattern)
    ends = [text.find(pattern, start) for pattern in end_patterns]
    valid = [end for end in ends if end >= 0]
    end = min(valid) if valid else len(text)
    return text[start:end]


# 関数: token group 全てを満たす最初の hit 行を返す。

def multi_hit(text: str, groups: tuple[tuple[str, ...], ...]) -> dict | None:
    """Return the first line containing at least one token from every group."""
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


# 関数: metrics row を組み立てる。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build a standard metrics row."""
    return {"row_id": row_id, "status": status, "metric": metric, "value": float(value), "note": note}


# 関数: JSON と rows CSV を書き出す。

def write_artifact(stem: str, data: dict) -> None:
    """Write a metrics payload as JSON and rows CSV."""
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# 関数: branch main を実行する。

def main() -> None:
    """Execute the source-link-clause-wording branch."""
    for path in (STATUS, ROADMAP, AI_CONTEXT, PART1, PART3A, PART5, PRIOR_INVENTORY, PRIOR_AUDIT, PRIOR_GATE, PRIOR_ROUTE):
        if not path.exists():
            raise SystemExit(f"[fail] missing required input: {path}")

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    prior_inventory = read_json(PRIOR_INVENTORY)
    prior_audit = read_json(PRIOR_AUDIT)
    prior_gate = read_json(PRIOR_GATE)
    prior_route = read_json(PRIOR_ROUTE)

    primary_section = section_text(part3a_text, PRIMARY_HEADING, (FALLBACK_HEADING, PART3A_NEXT))
    fallback_section = section_text(part3a_text, FALLBACK_HEADING, (PART3A_NEXT,))
    part5_section = section_text(part5_text, PART5_SECONDARY, (PART5_NEXT,))
    alpha_formula_line = hit(primary_section, ALPHA_FORMULA)
    open_clause_line = hit(primary_section, OPEN_CLAUSE)
    positive_wording_family_line = multi_hit(primary_section, POSITIVE_GROUPS)

    inventory_targets = [
        hit(status_text, STATUS_NEXT_STEP),
        hit(roadmap_text, ROADMAP_BRANCH),
        hit(part5_text, PART5_NEXT_STEP),
        hit(part3a_text, PRIMARY_HEADING),
        alpha_formula_line,
        open_clause_line,
        hit(part3a_text, FALLBACK_HEADING),
        hit(part5_text, PART5_SECONDARY),
    ]
    inventory_ready = all(item is not None for item in inventory_targets) and positive_wording_family_line is None
    part1_no_surface_preserved = hit(part1_text, "epsilon_0") is None
    fallback_no_wording_preserved = multi_hit(fallback_section, POSITIVE_GROUPS) is None
    secondary_no_wording_preserved = multi_hit(part5_section, POSITIVE_GROUPS) is None
    route_contract_consistent = (
        prior_gate["summary"]["selected_residual_route"] == CURRENT_ROUTE
        and prior_route["summary"]["selected_next_generation_route"] == CURRENT_ROUTE
    )

    inventory = {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": "8.7.56.443", "name": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_link_clause_wording_source_inventory"},
        "inputs": {
            "status_markdown": "doc/STATUS.md",
            "roadmap_markdown": "doc/ROADMAP.md",
            "ai_context_json": "doc/AI_CONTEXT_MIN.json",
            "part1_core_theory_markdown": "doc/paper/10_part1_core_theory.md",
            "part3a_quantum_foundations_markdown": "doc/paper/12_part3a_quantum_foundations.md",
            "part5_future_predictions_markdown": "doc/paper/14_part5_future_predictions.md",
            "prior_inventory_json": str(PRIOR_INVENTORY.relative_to(ROOT)).replace("\\", "/"),
            "prior_audit_json": str(PRIOR_AUDIT.relative_to(ROOT)).replace("\\", "/"),
            "prior_gate_json": str(PRIOR_GATE.relative_to(ROOT)).replace("\\", "/"),
            "prior_route_json": str(PRIOR_ROUTE.relative_to(ROOT)).replace("\\", "/"),
        },
        "intent": "Inventory the wording-level residual pack around the explicit numeric-alpha-open line and the missing positive source-link wording family.",
        "formulas": {
            "inventory_rule": "The explicit numeric-alpha-open line exists, but no positive Coulomb-normalization source-link wording family is present on that line family.",
            "surface_rule": "Part III-A 2.6.1 remains the primary surface because it already carries both the structural alpha formula and the explicit open-status line.",
        },
        "rows": [
            row("trial2_numeric_alpha_part3a_numeric_alpha_open_clause_source_link_clause_wording_inventory_targets_present", "pass" if inventory_ready else "reject", "numeric-alpha-open-clause source-link-clause wording inventory targets present", sum(1 for item in inventory_targets if item is not None), "Control docs, the explicit open-status line, and the missing-wording evidence must align on the wording branch."),
            row("trial2_numeric_alpha_part3a_positive_source_link_wording_family_absent", "pass" if positive_wording_family_line is None else "reject", "positive source-link wording family absent on primary surface", 1 if positive_wording_family_line is None else 0, "The explicit numeric-alpha-open line still carries no positive Coulomb-normalization source-link wording family."),
            row("trial2_numeric_alpha_part3a_fallback_no_wording_preserved", "pass" if fallback_no_wording_preserved else "reject", "Part III-A fallback no-wording evidence preserved", 1 if fallback_no_wording_preserved else 0, "Part III-A 2.6.2 still carries no positive source-link wording family."),
            row("trial2_numeric_alpha_part5_secondary_no_wording_preserved", "pass" if secondary_no_wording_preserved else "reject", "Part V secondary no-wording evidence preserved", 1 if secondary_no_wording_preserved else 0, "Part V 3.2 still carries no positive source-link wording family."),
            row("trial2_numeric_alpha_part3a_numeric_alpha_open_clause_source_link_clause_wording_route_contract_consistent", "pass" if route_contract_consistent else "reject", "source-link-clause wording route contract consistent", 1 if route_contract_consistent else 0, "The prior declaration gate and route contract must agree on the current wording residual route."),
        ],
        "summary": {
            "inventory_ready": inventory_ready,
            "part1_no_surface_preserved": part1_no_surface_preserved,
            "part3a_fallback_no_wording_preserved": fallback_no_wording_preserved,
            "part5_secondary_no_wording_preserved": secondary_no_wording_preserved,
            "part3a_positive_source_link_wording_family_present": positive_wording_family_line is not None,
            "route_contract_consistent": route_contract_consistent,
            "first_route_to_close_or_none": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_link_clause_wording_audit",
        },
        "decision": {
            "overall_status": "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_source_link_clause_wording_inventory_frozen" if inventory_ready else "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_source_link_clause_wording_inventory_incomplete",
            "advance_to_8_7_56_444": inventory_ready,
            "next_required_artifacts": [] if inventory_ready else ["trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_link_clause_wording_source_inventory"],
        },
        "evidence": {
            "alpha_formula_line": alpha_formula_line,
            "open_clause_line": open_clause_line,
            "positive_wording_family_line": positive_wording_family_line,
            "prior_inventory_summary": prior_inventory["summary"],
            "prior_audit_summary": prior_audit["summary"],
            "prior_gate_summary": prior_gate["summary"],
            "prior_route_summary": prior_route["summary"],
        },
    }

    wording_family_on_clause = multi_hit(open_clause_line["text"], POSITIVE_GROUPS) if open_clause_line is not None else None
    audit_ready = bool(inventory_ready and alpha_formula_line is not None and open_clause_line is not None and wording_family_on_clause is None)
    audit = {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": "8.7.56.444", "name": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_link_clause_wording_audit"},
        "inputs": inventory["inputs"],
        "intent": "Audit the sentence-level wording family required on the explicit numeric-alpha-open line for a positive Coulomb-normalization source link.",
        "formulas": {
            "open_clause_rule": "The explicit open-status line is the exact sentence surface that still marks the unresolved precision state.",
            "wording_family_rule": "The remaining blocker is whether that exact line already contains the positive Coulomb-normalization source-link wording family.",
        },
        "rows": [
            row("trial2_numeric_alpha_part3a_alpha_formula_line_present", "pass" if alpha_formula_line is not None else "reject", "Part III-A alpha formula line present", 1 if alpha_formula_line is not None else 0, "The structural alpha formula remains present on the primary surface."),
            row("trial2_numeric_alpha_part3a_numeric_alpha_open_clause_present", "pass" if open_clause_line is not None else "reject", "Part III-A numeric alpha open clause present", 1 if open_clause_line is not None else 0, "The explicit open-status clause remains present on the primary surface."),
            row("trial2_numeric_alpha_part3a_numeric_alpha_open_clause_contains_positive_source_link_wording_family", "pass" if wording_family_on_clause is not None else "reject", "numeric alpha open clause contains positive source-link wording family", 1 if wording_family_on_clause is not None else 0, "The open-status clause still carries no explicit positive Coulomb-normalization source-link wording family."),
            row("trial2_numeric_alpha_part3a_numeric_alpha_open_clause_positive_source_link_wording_family_absence_dominant_blocker", "pass" if audit_ready else "reject", "numeric alpha open clause positive source-link wording-family absence is dominant blocker", 1 if audit_ready else 0, "The residual now shrinks from generic source-link wording absence to the missing positive wording family on the explicit open-status line."),
        ],
        "summary": {
            "audit_ready": audit_ready,
            "part3a_alpha_formula_line_present": alpha_formula_line is not None,
            "part3a_numeric_alpha_open_clause_present": open_clause_line is not None,
            "part3a_numeric_alpha_open_clause_contains_positive_source_link_wording_family": wording_family_on_clause is not None,
            "dominant_blocker_is_part3a_numeric_alpha_open_clause_positive_source_link_wording_family_absence": audit_ready,
            "first_route_to_close_or_none": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_link_clause_wording_declaration_gate",
        },
        "decision": {
            "overall_status": "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_source_link_clause_wording_audit_complete" if audit_ready else "trial2_numeric_alpha_part3a_numeric_alpha_open_clause_source_link_clause_wording_audit_incomplete",
            "advance_to_8_7_56_445": audit_ready,
            "next_required_artifacts": [] if audit_ready else ["trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_link_clause_wording_audit"],
        },
        "evidence": {"inventory_summary": inventory["summary"], "wording_family_on_clause": wording_family_on_clause},
    }

    gate = {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": "8.7.56.445", "name": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_link_clause_wording_declaration_gate"},
        "inputs": inventory["inputs"],
        "intent": "Freeze the declaration gate after confirming that the explicit numeric-alpha-open clause is still missing the positive source-link wording family.",
        "formulas": {
            "gate_rule": "The current branch closes once the dominant blocker is localized to the missing positive wording family on the explicit open-status line.",
            "residual_rule": "The next residual concerns the wording family itself rather than the broader wording-branch identification family.",
        },
        "rows": [
            row("trial2_numeric_alpha_part3a_source_link_clause_wording_gate_complete", "pass" if audit_ready else "reject", "source-link-clause wording gate complete", 1 if audit_ready else 0, "The gate closes once the dominant blocker is narrowed to the missing positive wording family."),
            row("trial2_numeric_alpha_closeout_ready", "reject", "numeric-alpha closeout ready", 0, "Numeric-alpha closeout still requires the missing positive source-link wording family on the primary surface."),
            row("trial2_numeric_alpha_part3a_positive_source_link_wording_family_missing", "pass" if audit_ready else "watch", "Part III-A positive source-link wording family missing", 1 if audit_ready else 0, "The narrowed blocker is the missing positive source-link wording family on the explicit numeric-alpha-open line."),
            row("trial2_numeric_alpha_precision_mainline_preserved", "pass", "precision-alpha mainline preserved", 1, "The branch does not reopen the structural EM pass or promote the strong-side reserve."),
        ],
        "summary": {
            "trial2_numeric_alpha_part3a_source_link_clause_wording_branch_closeable": audit_ready,
            "trial2_numeric_alpha_closeout_ready": False,
            "selected_residual_route": NEXT_ROUTE_LABEL,
            "missing_v2_artifact": NEXT_MISSING_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        "decision": {
            "overall_status": "trial2_numeric_alpha_part3a_source_link_clause_wording_gate_closed" if audit_ready else "trial2_numeric_alpha_part3a_source_link_clause_wording_gate_open",
            "advance_to_8_7_56_446": audit_ready,
            "next_required_artifacts": [] if audit_ready else ["trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_link_clause_wording_declaration_gate"],
        },
        "evidence": {"audit_summary": audit["summary"]},
    }

    gate_closed = bool(gate["summary"]["trial2_numeric_alpha_part3a_source_link_clause_wording_branch_closeable"])
    contract = {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": "8.7.56.446", "name": "trial2_numeric_alpha_next_generation_route_contract_eighth_refresh"},
        "inputs": inventory["inputs"],
        "intent": "Refresh the strong-side reserve after the wording branch and freeze the next wording-family EM precision residual route.",
        "formulas": {
            "contract_rule": "The EM precision mainline remains active while the blocker shrinks from generic source-link-clause wording absence to the missing positive wording family on the explicit open-status line.",
            "reserve_rule": "Strong-side work remains on v3 hold reserve and does not outrank the EM precision route.",
        },
        "rows": [
            row("trial2_numeric_alpha_part3a_source_link_clause_wording_gate_closed", "pass" if gate_closed else "reject", "source-link-clause wording gate closed", 1 if gate_closed else 0, "The next route contract depends on the wording gate being frozen first."),
            row("trial2_numeric_alpha_part3a_positive_source_link_wording_family_route_selected", "pass" if gate_closed else "reject", "positive source-link wording-family route selected", 1 if gate_closed else 0, "The next official route stays inside the EM precision program."),
            row("trial2_numeric_alpha_strong_side_reserve_retained", "pass", "strong-side reserve retained", 1, "Strong-side non-Abelian/running/confinement gaps remain on reserve."),
            row("trial2_numeric_alpha_precision_mainline_retained", "pass", "precision-alpha mainline retained", 1, "The first next-generation mainline remains the Trial-2 numeric-alpha program."),
        ],
        "summary": {
            "selected_next_generation_route": NEXT_ROUTE_LABEL if gate_closed else None,
            "strong_side_route_state": prior_route["summary"]["strong_side_route_state"],
            "precision_alpha_mainline_retained": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        "decision": {
            "overall_status": "trial2_numeric_alpha_next_route_contract_eighth_refresh_frozen" if gate_closed else "trial2_numeric_alpha_next_route_contract_eighth_refresh_pending",
            "advance_to_next_route": gate_closed,
            "next_required_artifacts": [
                "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_family_source_inventory",
                "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_family_audit",
            ] if gate_closed else ["trial2_numeric_alpha_next_generation_route_contract_eighth_refresh"],
        },
        "evidence": {"gate_summary": gate["summary"], "prior_route_summary": prior_route["summary"]},
    }

    write_artifact("mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_link_clause_wording_source_inventory", inventory)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_link_clause_wording_audit", audit)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_source_link_clause_wording_declaration_gate", gate)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_eighth_refresh", contract)

    print("[ok] generated Trial-2 numeric alpha source-link-clause-wording artifacts")


# 関数: CLI 直実行時に branch main を起動する。

def run_cli() -> None:
    """CLI entry point for the source-link-clause-wording branch."""
    main()


if __name__ == "__main__":
    run_cli()
