#!/usr/bin/env python3
"""Generate 8.7.56.455-.458 positive-source-link wording-fragment-token-set artifacts."""

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

PRIOR_INVENTORY = OUT / "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_source_inventory_metrics.json"
PRIOR_AUDIT = OUT / "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_audit_metrics.json"
PRIOR_GATE = OUT / "mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_declaration_gate_metrics.json"
PRIOR_ROUTE = OUT / "mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_tenth_refresh_metrics.json"

STATUS_NEXT_STEP = "current official next step は `8.7.56.455`"
ROADMAP_BRANCH = "`8.7.56.455-.458` Trial-2 numeric $\\alpha$ Coulomb-normalization-source-surface Part-III-A numeric-α-open-clause positive-source-link-clause wording-fragment-token-set residual branch"
PART5_NEXT_STEP = "8.7.56.455-.458"
PRIMARY_HEADING = "#### 2.6.1 現行 canon で固定した source / structural route"
PART3A_NEXT = "#### 2.6.2 未導出（近似検証と判定の固定）"
PART5_CHECKPOINT = "### 3.2 v2.0 checkpoint：electromagnetism / weak-sector closeout（理論側 checkpoint）"
PART5_NEXT = "## 4."
OPEN_CLAUSE = "**foundational / structural pass (numeric α open)**"

TOKEN_GROUPS = (
    ("coulomb", "Coulomb"),
    ("normalization", "normalize", "normalise"),
    ("source", "source-surface", "source surface"),
)
TOKEN_NAMES = ("coulomb", "normalization", "source")

CURRENT_ROUTE = "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_token_set_identification"
NEXT_ROUTE = "8.7.56.459"
NEXT_ROUTE_LABEL = "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_identification"
NEXT_MISSING_ARTIFACT = "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token"


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


# 関数: 各 token group の hit / miss 状態を返す。

def group_presence(text: str, groups: tuple[tuple[str, ...], ...]) -> list[dict]:
    """Return per-group presence information for the given text."""
    lowered = text.lower()
    result: list[dict] = []
    for name, group in zip(TOKEN_NAMES, groups):
        token = next((candidate for candidate in group if candidate.lower() in lowered), None)
        result.append(
            {
                "token_name": name,
                "group": list(group),
                "matched_token_or_none": token,
                "present": token is not None,
            }
        )

    return result


# 関数: 最初に欠けている token 名を返す。

def first_missing_token(group_hits: list[dict]) -> str | None:
    """Return the first missing token name under the ordered token-set contract."""
    for item in group_hits:
        if not item["present"]:
            return str(item["token_name"])

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
    """Execute the positive-source-link wording-fragment-token-set branch."""
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
    clause_group_presence = group_presence(open_clause_line["text"], TOKEN_GROUPS) if open_clause_line is not None else []
    missing_group_count = sum(1 for item in clause_group_presence if not item["present"])
    first_missing = first_missing_token(clause_group_presence)

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
            "step": "8.7.56.455",
            "name": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_token_set_source_inventory",
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
        "intent": "Inventory the token-set residual pack around the explicit numeric-alpha-open line and the missing Coulomb/normalization/source token set.",
        "formulas": {
            "inventory_rule": "The explicit numeric-alpha-open line exists, but the ordered token set {Coulomb, normalization, source} is still absent there.",
            "first_missing_rule": "The next audit should isolate the first missing token under the ordered token-set contract.",
        },
        "rows": [
            row("trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_token_set_inventory_targets_present", "pass" if inventory_ready else "reject", "positive-source-link wording-fragment token-set inventory targets present", sum(1 for item in inventory_targets if item is not None), "Control docs and the explicit open-status line must align on the token-set branch."),
            row("trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_missing_token_group_count", "pass" if open_clause_line is not None else "reject", "missing wording-fragment token-group count", missing_group_count, "The explicit open-status line still lacks the ordered Coulomb/normalization/source token set."),
            row("trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_first_missing_token_is_coulomb", "pass" if first_missing == "coulomb" else "reject", "first missing wording-fragment token is Coulomb", 1 if first_missing == "coulomb" else 0, "The ordered token-set decomposition starts from the Coulomb token."),
            row("trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_token_set_route_contract_consistent", "pass" if route_contract_consistent else "reject", "positive-source-link wording-fragment token-set route contract consistent", 1 if route_contract_consistent else 0, "The prior declaration gate and route contract must agree on the current token-set residual route."),
        ],
        "summary": {
            "inventory_ready": inventory_ready,
            "missing_fragment_token_group_count": missing_group_count,
            "first_missing_fragment_token_or_none": first_missing,
            "route_contract_consistent": route_contract_consistent,
            "first_route_to_close_or_none": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_token_set_audit",
        },
        "decision": {
            "overall_status": "trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_token_set_inventory_frozen" if inventory_ready else "trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_token_set_inventory_incomplete",
            "advance_to_8_7_56_456": inventory_ready,
            "next_required_artifacts": [] if inventory_ready else ["trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_token_set_source_inventory"],
        },
        "evidence": {
            "open_clause_line": open_clause_line,
            "clause_group_presence": clause_group_presence,
            "prior_inventory_summary": prior_inventory["summary"],
            "prior_audit_summary": prior_audit["summary"],
            "prior_gate_summary": prior_gate["summary"],
            "prior_route_summary": prior_route["summary"],
        },
    }

    coulomb_missing = first_missing == "coulomb"
    audit_ready = bool(inventory_ready and coulomb_missing)
    audit = {
        "generated_utc": now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.456",
            "name": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_token_set_audit",
        },
        "inputs": inventory["inputs"],
        "intent": "Audit the ordered token-set still absent from the explicit numeric-alpha-open line and isolate the first missing token.",
        "formulas": {
            "token_set_rule": "The required positive source-link wording fragment is decomposed into the ordered token set {Coulomb, normalization, source}.",
            "dominance_rule": "The first missing token in that ordered set is treated as the dominant blocker for the next residual route.",
        },
        "rows": [
            row("trial2_numeric_alpha_part3a_numeric_alpha_open_clause_coulomb_token_present", "pass" if any(item["token_name"] == "coulomb" and item["present"] for item in clause_group_presence) else "reject", "numeric alpha open clause contains Coulomb token", 1 if any(item["token_name"] == "coulomb" and item["present"] for item in clause_group_presence) else 0, "The explicit open-status line still carries no Coulomb token."),
            row("trial2_numeric_alpha_part3a_numeric_alpha_open_clause_normalization_token_present", "pass" if any(item["token_name"] == "normalization" and item["present"] for item in clause_group_presence) else "reject", "numeric alpha open clause contains normalization token", 1 if any(item["token_name"] == "normalization" and item["present"] for item in clause_group_presence) else 0, "The explicit open-status line still carries no normalization token."),
            row("trial2_numeric_alpha_part3a_numeric_alpha_open_clause_source_token_present", "pass" if any(item["token_name"] == "source" and item["present"] for item in clause_group_presence) else "reject", "numeric alpha open clause contains source token", 1 if any(item["token_name"] == "source" and item["present"] for item in clause_group_presence) else 0, "The explicit open-status line still carries no source token."),
            row("trial2_numeric_alpha_part3a_numeric_alpha_open_clause_first_missing_token_is_coulomb", "pass" if coulomb_missing else "reject", "numeric alpha open clause first missing token is Coulomb", 1 if coulomb_missing else 0, "The ordered token-set decomposition localizes the next blocker to the first missing Coulomb token."),
        ],
        "summary": {
            "audit_ready": audit_ready,
            "coulomb_token_present_on_open_clause": any(item["token_name"] == "coulomb" and item["present"] for item in clause_group_presence),
            "normalization_token_present_on_open_clause": any(item["token_name"] == "normalization" and item["present"] for item in clause_group_presence),
            "source_token_present_on_open_clause": any(item["token_name"] == "source" and item["present"] for item in clause_group_presence),
            "dominant_blocker_is_first_missing_coulomb_token_absence": coulomb_missing,
            "first_missing_fragment_token_or_none": first_missing,
            "first_route_to_close_or_none": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_token_set_declaration_gate",
        },
        "decision": {
            "overall_status": "trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_token_set_audit_complete" if audit_ready else "trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_token_set_audit_incomplete",
            "advance_to_8_7_56_457": audit_ready,
            "next_required_artifacts": [] if audit_ready else ["trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_token_set_audit"],
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
            "step": "8.7.56.457",
            "name": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_token_set_declaration_gate",
        },
        "inputs": inventory["inputs"],
        "intent": "Freeze the declaration gate after confirming that the explicit numeric-alpha-open clause is still missing the first Coulomb token in the ordered wording-fragment token set.",
        "formulas": {
            "gate_rule": "The current branch closes once the dominant blocker is localized to the first missing Coulomb token on the explicit open-status line.",
            "residual_rule": "The next residual concerns the Coulomb token itself rather than the broader token-set identification family.",
        },
        "rows": [
            row("trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_token_set_gate_complete", "pass" if audit_ready else "reject", "positive-source-link wording-fragment token-set gate complete", 1 if audit_ready else 0, "The gate closes once the dominant blocker is narrowed to the missing Coulomb token."),
            row("trial2_numeric_alpha_closeout_ready", "reject", "numeric-alpha closeout ready", 0, "Numeric-alpha closeout still requires the missing Coulomb token on the primary surface."),
            row("trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_coulomb_token_missing", "pass" if coulomb_missing else "watch", "Part III-A positive source-link wording-fragment Coulomb token missing", 1 if coulomb_missing else 0, "The narrowed blocker is the missing Coulomb token on the explicit numeric-alpha-open line."),
            row("trial2_numeric_alpha_precision_mainline_preserved", "pass", "precision-alpha mainline preserved", 1, "The branch does not reopen the structural EM pass or promote the strong-side reserve."),
        ],
        "summary": {
            "trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_token_set_branch_closeable": audit_ready,
            "trial2_numeric_alpha_closeout_ready": False,
            "selected_residual_route": NEXT_ROUTE_LABEL,
            "missing_v2_artifact": NEXT_MISSING_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        "decision": {
            "overall_status": "trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_token_set_gate_closed" if audit_ready else "trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_token_set_gate_open",
            "advance_to_8_7_56_458": audit_ready,
            "next_required_artifacts": [] if audit_ready else ["trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_token_set_declaration_gate"],
        },
        "evidence": {"audit_summary": audit["summary"]},
    }

    gate_closed = bool(gate["summary"]["trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_token_set_branch_closeable"])
    contract = {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": "8.7.56.458", "name": "trial2_numeric_alpha_next_generation_route_contract_eleventh_refresh"},
        "inputs": inventory["inputs"],
        "intent": "Refresh the strong-side reserve after the token-set branch and freeze the next Coulomb-token EM precision residual route.",
        "formulas": {
            "contract_rule": "The EM precision mainline remains active while the blocker shrinks from generic wording-fragment token-set absence to the missing Coulomb token on the explicit open-status line.",
            "reserve_rule": "Strong-side work remains on v3 hold reserve and does not outrank the EM precision route.",
        },
        "rows": [
            row("trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_token_set_gate_closed", "pass" if gate_closed else "reject", "positive-source-link wording-fragment token-set gate closed", 1 if gate_closed else 0, "The next route contract depends on the token-set gate being frozen first."),
            row("trial2_numeric_alpha_part3a_positive_source_link_wording_fragment_coulomb_token_route_selected", "pass" if gate_closed else "reject", "positive source-link wording-fragment Coulomb-token route selected", 1 if gate_closed else 0, "The next official route stays inside the EM precision program."),
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
            "overall_status": "trial2_numeric_alpha_next_route_contract_eleventh_refresh_frozen" if gate_closed else "trial2_numeric_alpha_next_route_contract_eleventh_refresh_pending",
            "advance_to_next_route": gate_closed,
            "next_required_artifacts": [
                "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_source_inventory",
                "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_coulomb_token_audit",
            ] if gate_closed else ["trial2_numeric_alpha_next_route_contract_eleventh_refresh"],
        },
        "evidence": {"gate_summary": gate["summary"], "prior_route_summary": prior_route["summary"]},
    }

    write_artifact("mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_token_set_source_inventory", inventory)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_token_set_audit", audit)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_positive_source_link_clause_wording_fragment_token_set_declaration_gate", gate)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_eleventh_refresh", contract)

    print("[ok] generated Trial-2 numeric alpha positive-source-link wording-fragment-token-set artifacts")


# 関数: CLI 直実行時に branch main を起動する。

def run_cli() -> None:
    """CLI entry point for the positive-source-link wording-fragment-token-set branch."""
    main()


if __name__ == "__main__":
    run_cli()
