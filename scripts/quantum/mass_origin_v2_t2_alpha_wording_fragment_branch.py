#!/usr/bin/env python3
"""Generate 8.7.56.451-.454 positive-source-link wording-fragment artifacts."""

from __future__ import annotations

import csv
import json
import re
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

PRIOR_INVENTORY = OUT / "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_family_source_inventory_metrics.json"
PRIOR_AUDIT = OUT / "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_family_audit_metrics.json"
PRIOR_GATE = OUT / "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_family_declaration_gate_metrics.json"
PRIOR_ROUTE = OUT / "mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_ninth_refresh_metrics.json"

STATUS_NEXT_STEP = "current official next step は `8.7.56.451`"
ROADMAP_BRANCH = "`8.7.56.451-.454` Trial-2 numeric $\\alpha$ Coulomb-normalization-source-surface Part-III-A numeric-α-open-clause positive-source-link-clause wording-fragment residual branch"
PART5_NEXT_STEP = "8.7.56.451-.454"
PRIMARY_HEADING = "#### 2.6.1 現行 canon で固定した source / structural route"
FALLBACK_HEADING = "#### 2.6.2 未導出（近似検証と判定の固定）"
PART3A_NEXT = "### 2.7"
PART5_SECONDARY = "### 3.2 v2.0 checkpoint：electromagnetism / weak-sector closeout（理論側 checkpoint）"
PART5_NEXT = "## 4."
OPEN_CLAUSE = "**foundational / structural pass (numeric α open)**"
WORDING_FRAGMENT_GROUPS = (
    ("coulomb", "Coulomb"),
    ("normalization", "normalize", "normalise"),
    ("source", "source-surface", "source surface"),
)

CURRENT_ROUTE = "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_identification"
NEXT_ROUTE = "8.7.56.455"
NEXT_ROUTE_LABEL = "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_token_set_identification"
NEXT_MISSING_ARTIFACT = "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_token_set"


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


# 関数: backtick 内の route/identifier を除去して判定面を簡約する。

def strip_code_spans(text: str) -> str:
    """Remove markdown code spans so route labels do not create false positives."""
    return re.sub(r"`[^`]*`", "", text)


# 関数: 各 token group の hit / miss 状態を返す。

def group_presence(text: str, groups: tuple[tuple[str, ...], ...]) -> list[dict]:
    """Return per-group presence information for the given text."""
    lowered = text.lower()
    result: list[dict] = []
    for group in groups:
        token = next((candidate for candidate in group if candidate.lower() in lowered), None)
        result.append(
            {
                "group": list(group),
                "matched_token_or_none": token,
                "present": token is not None,
            }
        )

    return result


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
    """Execute the positive-source-link wording-fragment branch."""
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
    fallback_section_plain = strip_code_spans(fallback_section)
    part5_section_plain = strip_code_spans(part5_section)
    open_clause_line = hit(primary_section, OPEN_CLAUSE)
    wording_fragment_line = multi_hit(primary_section, WORDING_FRAGMENT_GROUPS)
    clause_group_presence = group_presence(open_clause_line["text"], WORDING_FRAGMENT_GROUPS) if open_clause_line is not None else []
    missing_group_count = sum(1 for item in clause_group_presence if not item["present"])

    inventory_targets = [
        hit(status_text, STATUS_NEXT_STEP),
        hit(roadmap_text, ROADMAP_BRANCH),
        hit(part5_text, PART5_NEXT_STEP),
        hit(part3a_text, PRIMARY_HEADING),
        open_clause_line,
        hit(part3a_text, FALLBACK_HEADING),
        hit(part5_text, PART5_SECONDARY),
    ]
    inventory_ready = all(item is not None for item in inventory_targets) and wording_fragment_line is None
    part1_no_surface_preserved = hit(part1_text, "epsilon_0") is None
    fallback_no_wording_family_preserved = multi_hit(fallback_section_plain, WORDING_FRAGMENT_GROUPS) is None
    secondary_no_wording_family_preserved = multi_hit(part5_section_plain, WORDING_FRAGMENT_GROUPS) is None
    route_contract_consistent = (
        prior_gate["summary"]["selected_residual_route"] == CURRENT_ROUTE
        and prior_route["summary"]["selected_next_generation_route"] == CURRENT_ROUTE
    )

    inventory = {
        "generated_utc": now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.451",
            "name": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_source_inventory",
        },
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
        "intent": "Inventory the wording-fragment residual pack around the explicit numeric-alpha-open line and the missing positive source-link wording fragment.",
        "formulas": {
            "inventory_rule": "The explicit numeric-alpha-open line exists, but no positive Coulomb-normalization source-link wording fragment is present on that line.",
            "token_set_rule": "The next audit should cut that missing wording fragment into the token-set still absent on the explicit open-status line.",
        },
        "rows": [
            row("trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_inventory_targets_present", "pass" if inventory_ready else "reject", "positive-source-link wording-fragment inventory targets present", sum(1 for item in inventory_targets if item is not None), "Control docs, the explicit open-status line, and the missing-wording-fragment evidence must align on the wording-fragment branch."),
            row("trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_absent", "pass" if wording_fragment_line is None else "reject", "positive source-link wording fragment absent on primary surface", 1 if wording_fragment_line is None else 0, "The explicit numeric-alpha-open line still carries no positive Coulomb-normalization source-link wording fragment."),
            row("trial2_numeric_alpha_part3a_fallback_no_wording_family_preserved", "pass" if fallback_no_wording_family_preserved else "reject", "Part III-A fallback no-wording-family evidence preserved", 1 if fallback_no_wording_family_preserved else 0, "Part III-A 2.6.2 still carries no positive source-link wording fragment family."),
            row("trial2_numeric_alpha_part5_secondary_no_wording_family_preserved", "pass" if secondary_no_wording_family_preserved else "reject", "Part V secondary no-wording-family evidence preserved", 1 if secondary_no_wording_family_preserved else 0, "Part V 3.2 still carries no positive source-link wording fragment family."),
            row("trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_route_contract_consistent", "pass" if route_contract_consistent else "reject", "positive-source-link wording-fragment route contract consistent", 1 if route_contract_consistent else 0, "The prior declaration gate and route contract must agree on the current wording-fragment residual route."),
        ],
        "summary": {
            "inventory_ready": inventory_ready,
            "part1_no_surface_preserved": part1_no_surface_preserved,
            "part3a_fallback_no_wording_family_preserved": fallback_no_wording_family_preserved,
            "part5_secondary_no_wording_family_preserved": secondary_no_wording_family_preserved,
            "part3a_positive_source_link_wording_fragment_present": wording_fragment_line is not None,
            "route_contract_consistent": route_contract_consistent,
            "first_route_to_close_or_none": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_audit",
        },
        "decision": {
            "overall_status": "trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_inventory_frozen" if inventory_ready else "trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_inventory_incomplete",
            "advance_to_8_7_56_452": inventory_ready,
            "next_required_artifacts": [] if inventory_ready else ["trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_source_inventory"],
        },
        "evidence": {
            "open_clause_line": open_clause_line,
            "wording_fragment_line": wording_fragment_line,
            "clause_group_presence": clause_group_presence,
            "prior_inventory_summary": prior_inventory["summary"],
            "prior_audit_summary": prior_audit["summary"],
            "prior_gate_summary": prior_gate["summary"],
            "prior_route_summary": prior_route["summary"],
        },
    }

    fragment_token_set_present = all(item["present"] for item in clause_group_presence) if clause_group_presence else False
    audit_ready = bool(inventory_ready and open_clause_line is not None and not fragment_token_set_present)
    audit = {
        "generated_utc": now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.452",
            "name": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_audit",
        },
        "inputs": inventory["inputs"],
        "intent": "Audit the token-set still absent from the explicit numeric-alpha-open line after the wording-fragment residual was frozen.",
        "formulas": {
            "open_clause_rule": "The explicit open-status line is the exact sentence surface that still marks the unresolved precision state.",
            "token_set_rule": "The remaining blocker is whether that exact line already contains the Coulomb/normalization/source token set required for a positive Coulomb-normalization source-link wording fragment.",
        },
        "rows": [
            row("trial2_numeric_alpha_part3a_numeric_alpha_open_clause_present", "pass" if open_clause_line is not None else "reject", "Part III-A numeric alpha open clause present", 1 if open_clause_line is not None else 0, "The explicit open-status clause remains present on the primary surface."),
            row("trial2_numeric_alpha_part3a_numeric_alpha_open_clause_contains_positive_source_link_wording_fragment_token_set", "pass" if fragment_token_set_present else "reject", "numeric alpha open clause contains positive source-link wording fragment token set", 1 if fragment_token_set_present else 0, "The open-status clause still carries no explicit Coulomb-normalization/source token set."),
            row("trial2_numeric_alpha_part3a_numeric_alpha_open_clause_positive_source_link_wording_fragment_token_set_absence_dominant_blocker", "pass" if audit_ready else "reject", "numeric alpha open clause positive source-link wording-fragment token-set absence is dominant blocker", 1 if audit_ready else 0, "The residual now shrinks from generic wording-fragment absence to the missing token set on the explicit open-status line."),
        ],
        "summary": {
            "audit_ready": audit_ready,
            "part3a_numeric_alpha_open_clause_present": open_clause_line is not None,
            "part3a_numeric_alpha_open_clause_contains_positive_source_link_wording_fragment_token_set": fragment_token_set_present,
            "dominant_blocker_is_part3a_numeric_alpha_open_clause_positive_source_link_wording_fragment_token_set_absence": audit_ready,
            "missing_fragment_token_group_count": missing_group_count,
            "first_route_to_close_or_none": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_declaration_gate",
        },
        "decision": {
            "overall_status": "trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_audit_complete" if audit_ready else "trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_audit_incomplete",
            "advance_to_8_7_56_453": audit_ready,
            "next_required_artifacts": [] if audit_ready else ["trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_audit"],
        },
        "evidence": {
            "inventory_summary": inventory["summary"],
            "clause_group_presence": clause_group_presence,
        },
    }

    gate = {
        "generated_utc": now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.453",
            "name": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_declaration_gate",
        },
        "inputs": inventory["inputs"],
        "intent": "Freeze the declaration gate after confirming that the explicit numeric-alpha-open clause is still missing the positive source-link wording-fragment token set.",
        "formulas": {
            "gate_rule": "The current branch closes once the dominant blocker is localized to the missing wording-fragment token set on the explicit open-status line.",
            "residual_rule": "The next residual concerns the wording-fragment token set itself rather than the broader wording-fragment identification family.",
        },
        "rows": [
            row("trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_gate_complete", "pass" if audit_ready else "reject", "positive-source-link wording-fragment gate complete", 1 if audit_ready else 0, "The gate closes once the dominant blocker is narrowed to the missing wording-fragment token set."),
            row("trial2_numeric_alpha_closeout_ready", "reject", "numeric-alpha closeout ready", 0, "Numeric-alpha closeout still requires the missing positive source-link wording-fragment token set on the primary surface."),
            row("trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_token_set_missing", "pass" if audit_ready else "watch", "Part III-A positive source-link wording-fragment token set missing", 1 if audit_ready else 0, "The narrowed blocker is the missing wording-fragment token set on the explicit numeric-alpha-open line."),
            row("trial2_numeric_alpha_precision_mainline_preserved", "pass", "precision-alpha mainline preserved", 1, "The branch does not reopen the structural EM pass or promote the strong-side reserve."),
        ],
        "summary": {
            "trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_branch_closeable": audit_ready,
            "trial2_numeric_alpha_closeout_ready": False,
            "selected_residual_route": NEXT_ROUTE_LABEL,
            "missing_v2_artifact": NEXT_MISSING_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        "decision": {
            "overall_status": "trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_gate_closed" if audit_ready else "trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_gate_open",
            "advance_to_8_7_56_454": audit_ready,
            "next_required_artifacts": [] if audit_ready else ["trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_declaration_gate"],
        },
        "evidence": {"audit_summary": audit["summary"]},
    }

    gate_closed = bool(gate["summary"]["trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_branch_closeable"])
    contract = {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": "8.7.56.454", "name": "trial2_numeric_alpha_next_generation_route_contract_tenth_refresh"},
        "inputs": inventory["inputs"],
        "intent": "Refresh the strong-side reserve after the wording-fragment branch and freeze the next wording-fragment-token-set EM precision residual route.",
        "formulas": {
            "contract_rule": "The EM precision mainline remains active while the blocker shrinks from generic source-link wording-fragment absence to the missing wording-fragment token set on the explicit open-status line.",
            "reserve_rule": "Strong-side work remains on v3 hold reserve and does not outrank the EM precision route.",
        },
        "rows": [
            row("trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_gate_closed", "pass" if gate_closed else "reject", "positive-source-link wording-fragment gate closed", 1 if gate_closed else 0, "The next route contract depends on the wording-fragment gate being frozen first."),
            row("trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_token_set_route_selected", "pass" if gate_closed else "reject", "positive source-link wording-fragment token-set route selected", 1 if gate_closed else 0, "The next official route stays inside the EM precision program."),
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
            "overall_status": "trial2_numeric_alpha_next_route_contract_tenth_refresh_frozen" if gate_closed else "trial2_numeric_alpha_next_route_contract_tenth_refresh_pending",
            "advance_to_next_route": gate_closed,
            "next_required_artifacts": [
                "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_token_set_source_inventory",
                "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_token_set_audit",
            ] if gate_closed else ["trial2_numeric_alpha_next_generation_route_contract_tenth_refresh"],
        },
        "evidence": {"gate_summary": gate["summary"], "prior_route_summary": prior_route["summary"]},
    }

    write_artifact("mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_source_inventory", inventory)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_audit", audit)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_declaration_gate", gate)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_tenth_refresh", contract)

    print("[ok] generated Trial-2 numeric alpha positive-source-link wording-fragment artifacts")


# 関数: CLI 直実行時に branch main を起動する。

def run_cli() -> None:
    """CLI entry point for the positive-source-link wording-fragment branch."""
    main()


if __name__ == "__main__":
    run_cli()
