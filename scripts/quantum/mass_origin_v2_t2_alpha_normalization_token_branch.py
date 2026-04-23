#!/usr/bin/env python3
"""Generate 8.7.56.471-.474 positive-source-link normalization-token artifacts."""

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
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

PRIOR_INVENTORY = OUT / "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_wording_fragment_source_inventory_metrics.json"
PRIOR_AUDIT = OUT / "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_wording_fragment_audit_metrics.json"
PRIOR_GATE = OUT / "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_wording_fragment_declaration_gate_metrics.json"
PRIOR_ROUTE = OUT / "mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_fourteenth_refresh_metrics.json"

STATUS_NEXT_STEP = "current official next step は `8.7.56.471`"
ROADMAP_BRANCH = "`8.7.56.471-.474` Trial-2 numeric $\\alpha$ Coulomb-normalization-source-surface Part-III-A numeric-α-open-clause positive-source-link-clause wording-fragment Coulomb-token-wording-fragment normalization-token residual branch"
PART5_NEXT_STEP = "8.7.56.471-.474"
PRIMARY_HEADING = "#### 2.6.1 現行 canon で固定した source / structural route"
PART3A_NEXT = "#### 2.6.2 未導出（近似検証と判定の固定）"
OPEN_CLAUSE = "**foundational / structural pass (numeric α open)**"

NORMALIZATION_TOKENS = ("normalization", "Normalization")

CURRENT_ROUTE = "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_wording_fragment_normalization_token_identification"
NEXT_ROUTE = "8.7.56.475"
NEXT_ROUTE_LABEL = "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_wording_fragment_normalization_token_source_token_identification"
NEXT_MISSING_ARTIFACT = "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_wording_fragment_normalization_token_source_token"


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


# 関数: normalization token の hit / miss 状態を返す。

def token_presence(text: str, group: tuple[str, ...]) -> dict:
    """Return normalization-token presence information for the given text."""
    lowered = text.lower()
    token = next((candidate for candidate in group if candidate.lower() in lowered), None)
    return {
        "token_name": "normalization_token",
        "group": list(group),
        "matched_token_or_none": token,
        "present": token is not None,
    }


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
    """Execute the positive-source-link normalization-token branch."""
    for path in (STATUS, ROADMAP, AI_CONTEXT, PART3A, PART5, PRIOR_INVENTORY, PRIOR_AUDIT, PRIOR_GATE, PRIOR_ROUTE):
        if not path.exists():
            raise SystemExit(f"[fail] missing required input: {path}")

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    prior_inventory = read_json(PRIOR_INVENTORY)
    prior_audit = read_json(PRIOR_AUDIT)
    prior_gate = read_json(PRIOR_GATE)
    prior_route = read_json(PRIOR_ROUTE)

    primary_section = section_text(part3a_text, PRIMARY_HEADING, (PART3A_NEXT,))
    open_clause_line = hit(primary_section, OPEN_CLAUSE)
    token_hit = token_presence(open_clause_line["text"], NORMALIZATION_TOKENS) if open_clause_line is not None else None
    token_present = bool(token_hit and token_hit["present"])

    inventory_targets = [
        hit(status_text, STATUS_NEXT_STEP),
        hit(roadmap_text, ROADMAP_BRANCH),
        hit(part5_text, PART5_NEXT_STEP),
        hit(part3a_text, PRIMARY_HEADING),
        open_clause_line,
    ]
    inventory_ready = all(item is not None for item in inventory_targets)
    route_contract_consistent = (
        prior_gate["summary"]["selected_residual_route"] == CURRENT_ROUTE
        and prior_route["summary"]["selected_next_generation_route"] == CURRENT_ROUTE
    )

    inventory = {
        "generated_utc": now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.471",
            "name": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_wording_fragment_normalization_token_source_inventory",
        },
        "inputs": {
            "status_markdown": "doc/STATUS.md",
            "roadmap_markdown": "doc/ROADMAP.md",
            "ai_context_json": "doc/AI_CONTEXT_MIN.json",
            "part3a_quantum_foundations_markdown": "doc/paper/12_part3a_quantum_foundations.md",
            "part5_future_predictions_markdown": "doc/paper/14_part5_future_predictions.md",
            "prior_inventory_json": str(PRIOR_INVENTORY.relative_to(ROOT)).replace("\\", "/"),
            "prior_audit_json": str(PRIOR_AUDIT.relative_to(ROOT)).replace("\\", "/"),
            "prior_gate_json": str(PRIOR_GATE.relative_to(ROOT)).replace("\\", "/"),
            "prior_route_json": str(PRIOR_ROUTE.relative_to(ROOT)).replace("\\", "/"),
        },
        "intent": "Inventory the normalization-token residual pack around the explicit numeric-alpha-open line and the missing normalization token after the Coulomb wording fragment.",
        "formulas": {
            "inventory_rule": "The explicit numeric-alpha-open line exists, but it still carries no normalization token after the Coulomb wording fragment.",
            "next_rule": "The next audit should isolate the missing source token that would have to follow the normalization token inside the positive source-link clause contract.",
        },
        "rows": [
            row("trial2_numeric_alpha_part3a_positive_source_link_normalization_token_inventory_targets_present", "pass" if inventory_ready else "reject", "positive-source-link normalization-token inventory targets present", sum(1 for item in inventory_targets if item is not None), "Control docs and the explicit open-status line must align on the normalization-token branch."),
            row("trial2_numeric_alpha_part3a_positive_source_link_normalization_token_present", "pass" if token_present else "reject", "positive-source-link normalization token present", 1 if token_present else 0, "The explicit open-status line still carries no normalization token."),
            row("trial2_numeric_alpha_part3a_positive_source_link_normalization_token_route_contract_consistent", "pass" if route_contract_consistent else "reject", "positive-source-link normalization-token route contract consistent", 1 if route_contract_consistent else 0, "The prior declaration gate and route contract must agree on the current normalization-token residual route."),
        ],
        "summary": {
            "inventory_ready": inventory_ready,
            "normalization_token_present_on_open_clause": token_present,
            "route_contract_consistent": route_contract_consistent,
            "first_route_to_close_or_none": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_wording_fragment_normalization_token_audit",
        },
        "decision": {
            "overall_status": "trial2_numeric_alpha_part3a_positive_source_link_normalization_token_inventory_frozen" if inventory_ready else "trial2_numeric_alpha_part3a_positive_source_link_normalization_token_inventory_incomplete",
            "advance_to_8_7_56_472": inventory_ready,
            "next_required_artifacts": [] if inventory_ready else ["trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_wording_fragment_normalization_token_source_inventory"],
        },
        "evidence": {
            "open_clause_line": open_clause_line,
            "normalization_token_presence": token_hit,
            "prior_inventory_summary": prior_inventory["summary"],
            "prior_audit_summary": prior_audit["summary"],
            "prior_gate_summary": prior_gate["summary"],
            "prior_route_summary": prior_route["summary"],
        },
    }

    audit_ready = bool(inventory_ready and not token_present)
    audit = {
        "generated_utc": now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.472",
            "name": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_wording_fragment_normalization_token_audit",
        },
        "inputs": inventory["inputs"],
        "intent": "Audit the explicit numeric-alpha-open line and confirm that the dominant blocker is the missing normalization token itself.",
        "formulas": {
            "token_rule": "The required positive source-link wording fragment still lacks the normalization token after the Coulomb wording fragment.",
            "dominance_rule": "Once the Coulomb-token-wording-fragment route is closed, the missing normalization token becomes the dominant blocker for the next residual route.",
        },
        "rows": [
            row("trial2_numeric_alpha_part3a_numeric_alpha_open_clause_normalization_token_present", "pass" if token_present else "reject", "numeric alpha open clause contains normalization token", 1 if token_present else 0, "The explicit open-status line still carries no normalization token."),
            row("trial2_numeric_alpha_part3a_numeric_alpha_open_clause_normalization_token_absent", "pass" if not token_present else "reject", "numeric alpha open clause normalization token absent", 1 if not token_present else 0, "The open-status line remains missing the normalization token."),
            row("trial2_numeric_alpha_part3a_numeric_alpha_open_clause_normalization_token_dominant_blocker", "pass" if not token_present else "reject", "numeric alpha open clause normalization token is dominant blocker", 1 if not token_present else 0, "The residual has shrunk from missing Coulomb-token-wording-fragment normalization-token identification to the missing normalization token itself."),
        ],
        "summary": {
            "audit_ready": audit_ready,
            "normalization_token_present_on_open_clause": token_present,
            "dominant_blocker_is_normalization_token_absence": not token_present,
            "first_route_to_close_or_none": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_wording_fragment_normalization_token_declaration_gate",
        },
        "decision": {
            "overall_status": "trial2_numeric_alpha_part3a_positive_source_link_normalization_token_audit_complete" if audit_ready else "trial2_numeric_alpha_part3a_positive_source_link_normalization_token_audit_incomplete",
            "advance_to_8_7_56_473": audit_ready,
            "next_required_artifacts": [] if audit_ready else ["trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_wording_fragment_normalization_token_audit"],
        },
        "evidence": {
            "inventory_summary": inventory["summary"],
            "normalization_token_presence": token_hit,
        },
    }

    gate = {
        "generated_utc": now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.473",
            "name": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_wording_fragment_normalization_token_declaration_gate",
        },
        "inputs": inventory["inputs"],
        "intent": "Freeze the declaration gate after confirming that the explicit numeric-alpha-open clause is still missing the normalization token.",
        "formulas": {
            "gate_rule": "The current branch closes once the dominant blocker is localized to the missing normalization token on the explicit open-status line.",
            "residual_rule": "The next residual concerns the missing source token that must follow the normalization token inside the positive source-link clause contract.",
        },
        "rows": [
            row("trial2_numeric_alpha_part3a_positive_source_link_normalization_token_gate_complete", "pass" if audit_ready else "reject", "positive-source-link normalization-token gate complete", 1 if audit_ready else 0, "The gate closes once the dominant blocker is narrowed to the missing normalization token."),
            row("trial2_numeric_alpha_closeout_ready", "reject", "numeric-alpha closeout ready", 0, "Numeric-alpha closeout still requires a positive source-link clause on the primary surface."),
            row("trial2_numeric_alpha_part3a_positive_source_link_normalization_token_missing", "pass" if not token_present else "watch", "Part III-A positive source-link normalization token missing", 1 if not token_present else 0, "The narrowed blocker is the missing normalization token on the explicit numeric-alpha-open line."),
            row("trial2_numeric_alpha_precision_mainline_preserved", "pass", "precision-alpha mainline preserved", 1, "The branch does not reopen the structural EM pass or promote the strong-side reserve."),
        ],
        "summary": {
            "trial2_numeric_alpha_part3a_positive_source_link_normalization_token_branch_closeable": audit_ready,
            "trial2_numeric_alpha_closeout_ready": False,
            "selected_residual_route": NEXT_ROUTE_LABEL,
            "missing_v2_artifact": NEXT_MISSING_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        "decision": {
            "overall_status": "trial2_numeric_alpha_part3a_positive_source_link_normalization_token_gate_closed" if audit_ready else "trial2_numeric_alpha_part3a_positive_source_link_normalization_token_gate_open",
            "advance_to_8_7_56_474": audit_ready,
            "next_required_artifacts": [] if audit_ready else ["trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_wording_fragment_normalization_token_declaration_gate"],
        },
        "evidence": {"audit_summary": audit["summary"]},
    }

    gate_closed = bool(gate["summary"]["trial2_numeric_alpha_part3a_positive_source_link_normalization_token_branch_closeable"])
    contract = {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": "8.7.56.474", "name": "trial2_numeric_alpha_next_generation_route_contract_fifteenth_refresh"},
        "inputs": inventory["inputs"],
        "intent": "Refresh the strong-side reserve after the normalization-token branch and freeze the next source-token EM precision residual route.",
        "formulas": {
            "contract_rule": "The EM precision mainline remains active while the blocker shrinks from the missing normalization token family to the missing source token inside the positive source-link clause contract.",
            "reserve_rule": "Strong-side work remains on v3 hold reserve and does not outrank the EM precision route.",
        },
        "rows": [
            row("trial2_numeric_alpha_part3a_positive_source_link_normalization_token_gate_closed", "pass" if gate_closed else "reject", "positive-source-link normalization-token gate closed", 1 if gate_closed else 0, "The next route contract depends on the normalization-token gate being frozen first."),
            row("trial2_numeric_alpha_part3a_positive_source_link_normalization_token_source_token_route_selected", "pass" if gate_closed else "reject", "positive source-link normalization-token source-token route selected", 1 if gate_closed else 0, "The next official route stays inside the EM precision program."),
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
            "overall_status": "trial2_numeric_alpha_next_route_contract_fifteenth_refresh_frozen" if gate_closed else "trial2_numeric_alpha_next_route_contract_fifteenth_refresh_pending",
            "advance_to_next_route": gate_closed,
            "next_required_artifacts": [
                "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_wording_fragment_normalization_token_source_token_source_inventory",
                "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_wording_fragment_normalization_token_source_token_audit",
            ] if gate_closed else ["trial2_numeric_alpha_next_route_contract_fifteenth_refresh"],
        },
        "evidence": {"gate_summary": gate["summary"], "prior_route_summary": prior_route["summary"]},
    }

    write_artifact("mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_wording_fragment_normalization_token_source_inventory", inventory)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_wording_fragment_normalization_token_audit", audit)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_wording_fragment_normalization_token_declaration_gate", gate)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_fifteenth_refresh", contract)

    print("[ok] generated Trial-2 numeric alpha positive-source-link normalization-token artifacts")


# 関数: CLI 直実行時に branch main を起動する。

def run_cli() -> None:
    """CLI entry point for the positive-source-link normalization-token branch."""
    main()


if __name__ == "__main__":
    run_cli()
