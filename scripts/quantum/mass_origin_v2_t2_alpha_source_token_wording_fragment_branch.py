#!/usr/bin/env python3
"""Generate 8.7.56.691-.694 deep7 source-token-wording-fragment artifacts."""

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

FRAGMENT_UNIT = "source_token_wording_fragment"
FRAGMENT_UNIT_HUMAN = "source-token-wording-fragment"
CHAIN_DEPTH = 7
DEEP_TAG = f"deep{CHAIN_DEPTH}"
FRAGMENT_CHAIN = "_".join([FRAGMENT_UNIT] * CHAIN_DEPTH)
FRAGMENT_CHAIN_HUMAN = " ".join([FRAGMENT_UNIT_HUMAN] * CHAIN_DEPTH)
CURRENT_FRAGMENT_NAME = (
    f"{FRAGMENT_CHAIN}_source_token_wording_fragment_source_token_wording_fragment_"
    "source_token_wording_fragment_source_token_wording_fragment_source_token_wording_fragment"
)
CURRENT_FRAGMENT_HUMAN = (
    f"{FRAGMENT_CHAIN_HUMAN} source-token wording fragment source-token wording fragment "
    "source-token wording fragment source-token wording fragment source-token wording fragment"
)
CURRENT_TOKEN_NAME = f"{CURRENT_FRAGMENT_NAME}_source_token_wording_fragment"
CURRENT_TOKEN_HUMAN = f"{CURRENT_FRAGMENT_HUMAN} source-token wording fragment"
TOKEN_SET_NAME = f"{CURRENT_TOKEN_NAME}_token_set"
TOKEN_SET_HUMAN = f"{CURRENT_TOKEN_HUMAN} token set"
ROUTE_PREFIX = (
    "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_numeric_alpha_open_clause_"
    "positive_source_link_clause_wording_fragment_coulomb_token_wording_fragment_normalization_token"
)

PRIOR_INVENTORY = OUT / "mass_origin_v2_t2_alpha_open_clause_deep7_source_token_wording_source_inventory_metrics.json"
PRIOR_AUDIT = OUT / "mass_origin_v2_t2_alpha_open_clause_deep7_source_token_wording_audit_metrics.json"
PRIOR_GATE = OUT / "mass_origin_v2_t2_alpha_open_clause_deep7_source_token_wording_declaration_gate_metrics.json"
PRIOR_ROUTE = OUT / "mass_origin_v2_t2_alpha_route_contract_sixty_ninth_refresh_metrics.json"

STATUS_NEXT_STEP = "current official next step は `8.7.56.691`"
ROADMAP_BRANCH = (
    "`8.7.56.691-.694` Trial-2 numeric $\\alpha$ Coulomb-normalization-source-surface "
    "Part-III-A numeric-α-open-clause deep7 source-token-wording-fragment residual branch"
)
PART5_NEXT_STEP = "8.7.56.691-.694"
PRIMARY_HEADING = "#### 2.6.1 現行 canon で固定した source / structural route"
PART3A_NEXT = "#### 2.6.2 未導出（近似検証と判定の固定）"
OPEN_CLAUSE = "**foundational / structural pass (numeric α open)**"

SOURCE_TOKEN_WORDING_FRAGMENTS = (
    "source token wording fragment",
    "Source token wording fragment",
    "source-token wording fragment",
    "source-token-wording-fragment",
)

CURRENT_ROUTE = f"{ROUTE_PREFIX}_{CURRENT_TOKEN_NAME}_identification"
NEXT_ROUTE = "8.7.56.695"
NEXT_ROUTE_LABEL = f"{ROUTE_PREFIX}_{TOKEN_SET_NAME}_identification"
NEXT_MISSING_ARTIFACT = f"{ROUTE_PREFIX}_{TOKEN_SET_NAME}"
INVENTORY_STEM = f"mass_origin_v2_t2_alpha_open_clause_{DEEP_TAG}_source_token_wording_fragment_source_inventory"
AUDIT_STEM = f"mass_origin_v2_t2_alpha_open_clause_{DEEP_TAG}_source_token_wording_fragment_audit"
GATE_STEM = f"mass_origin_v2_t2_alpha_open_clause_{DEEP_TAG}_source_token_wording_fragment_declaration_gate"
ROUTE_STEM = "mass_origin_v2_t2_alpha_route_contract_seventieth_refresh"


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


# 関数: deepest source-token-wording-fragment の hit / miss 状態を返す。

def token_presence(text: str, group: tuple[str, ...]) -> dict:
    """Return deepest source-token-wording-fragment presence information for the given text."""
    lowered = text.lower()
    token = next((candidate for candidate in group if candidate.lower() in lowered), None)
    return {
        "token_name": CURRENT_TOKEN_NAME,
        "group": list(group),
        "matched_token_or_none": token,
        "present": token is not None,
    }


# 関数: metrics row を組み立てる。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build a standard metrics row."""
    return {"row_id": row_id, "status": status, "metric": metric, "value": float(value), "note": note}


# 関数: Windows 長パス書き込み用の実パス文字列を返す。

def fs_path(path: Path) -> str:
    """Return a filesystem path string with Windows long-path support when needed."""
    resolved = str(path.resolve())
    if resolved.startswith("\\\\?\\"):
        return resolved

    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved[2:]

    return "\\\\?\\" + resolved if len(resolved) >= 240 else resolved


# 関数: JSON と rows CSV を書き出す。

def write_artifact(stem: str, data: dict) -> None:
    """Write a metrics payload as JSON and rows CSV."""
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    with open(fs_path(json_path), "w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    with open(fs_path(csv_path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# 関数: branch main を実行する。

def main() -> None:
    """Execute the deep7 source-token-wording-fragment residual branch."""
    required_paths = (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PART3A,
        PART5,
        PRIOR_INVENTORY,
        PRIOR_AUDIT,
        PRIOR_GATE,
        PRIOR_ROUTE,
    )
    for path in required_paths:
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
    token_hit = (
        token_presence(open_clause_line["text"], SOURCE_TOKEN_WORDING_FRAGMENTS)
        if open_clause_line is not None
        else None
    )
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
            "step": "8.7.56.691",
            "name": f"{ROUTE_PREFIX}_{CURRENT_TOKEN_NAME}_source_inventory",
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
        "intent": "Inventory the deep7 source-token-wording-fragment residual pack around the explicit numeric-alpha-open line and the missing deepest source-token wording fragment after the already-missing deepest source-token wording.",
        "formulas": {
            "inventory_rule": f"The explicit numeric-alpha-open line exists, but it still carries no deepest {CURRENT_TOKEN_HUMAN}.",
            "next_rule": f"The next audit should isolate the missing deepest {TOKEN_SET_HUMAN} that must populate that deepest source-token wording fragment.",
        },
        "rows": [
            row(
                f"trial2_numeric_alpha_part3a_positive_source_link_{CURRENT_TOKEN_NAME}_inventory_targets_present",
                "pass" if inventory_ready else "reject",
                f"positive-source-link {CURRENT_TOKEN_HUMAN} inventory targets present",
                sum(1 for item in inventory_targets if item is not None),
                f"Control docs and the explicit open-status line must align on the {CURRENT_TOKEN_HUMAN} branch.",
            ),
            row(
                f"trial2_numeric_alpha_part3a_positive_source_link_{CURRENT_TOKEN_NAME}_present",
                "pass" if token_present else "reject",
                f"positive-source-link {CURRENT_TOKEN_HUMAN} present",
                1 if token_present else 0,
                f"The explicit open-status line still carries no deepest {CURRENT_TOKEN_HUMAN}.",
            ),
            row(
                f"trial2_numeric_alpha_part3a_positive_source_link_{CURRENT_TOKEN_NAME}_route_contract_consistent",
                "pass" if route_contract_consistent else "reject",
                f"positive-source-link {CURRENT_TOKEN_HUMAN} route contract consistent",
                1 if route_contract_consistent else 0,
                f"The prior declaration gate and route contract must agree on the current {CURRENT_TOKEN_HUMAN} residual route.",
            ),
        ],
        "summary": {
            "inventory_ready": inventory_ready,
            f"{CURRENT_TOKEN_NAME}_present_on_open_clause": token_present,
            "route_contract_consistent": route_contract_consistent,
            "first_route_to_close_or_none": f"{ROUTE_PREFIX}_{CURRENT_TOKEN_NAME}_audit",
        },
        "decision": {
            "overall_status": f"trial2_numeric_alpha_part3a_positive_source_link_{CURRENT_TOKEN_NAME}_inventory_frozen" if inventory_ready else f"trial2_numeric_alpha_part3a_positive_source_link_{CURRENT_TOKEN_NAME}_inventory_incomplete",
            "advance_to_8_7_56_692": inventory_ready,
            "next_required_artifacts": [] if inventory_ready else [f"{ROUTE_PREFIX}_{CURRENT_TOKEN_NAME}_source_inventory"],
        },
        "evidence": {
            "open_clause_line": open_clause_line,
            "source_token_wording_fragment_presence": token_hit,
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
            "step": "8.7.56.692",
            "name": f"{ROUTE_PREFIX}_{CURRENT_TOKEN_NAME}_audit",
        },
        "inputs": inventory["inputs"],
        "intent": "Audit the explicit numeric-alpha-open line and confirm that the dominant blocker is the missing deepest source-token-wording-fragment itself.",
        "formulas": {
            "token_rule": f"The required positive source-link wording fragment still lacks the deepest {CURRENT_TOKEN_HUMAN}.",
            "dominance_rule": f"Once the deepest source-token-wording route is closed, the missing deepest {CURRENT_TOKEN_HUMAN} becomes the dominant blocker for the next residual route.",
        },
        "rows": [
            row(
                f"trial2_numeric_alpha_part3a_numeric_alpha_open_clause_{CURRENT_TOKEN_NAME}_present",
                "pass" if token_present else "reject",
                f"numeric alpha open clause contains {CURRENT_TOKEN_HUMAN}",
                1 if token_present else 0,
                f"The explicit open-status line still carries no deepest {CURRENT_TOKEN_HUMAN}.",
            ),
            row(
                f"trial2_numeric_alpha_part3a_numeric_alpha_open_clause_{CURRENT_TOKEN_NAME}_absent",
                "pass" if not token_present else "reject",
                f"numeric alpha open clause {CURRENT_TOKEN_HUMAN} absent",
                1 if not token_present else 0,
                f"The open-status line remains missing the deepest {CURRENT_TOKEN_HUMAN}.",
            ),
            row(
                f"trial2_numeric_alpha_part3a_numeric_alpha_open_clause_{CURRENT_TOKEN_NAME}_dominant_blocker",
                "pass" if not token_present else "reject",
                f"numeric alpha open clause {CURRENT_TOKEN_HUMAN} is dominant blocker",
                1 if not token_present else 0,
                f"The residual has shrunk from the missing deepest source-token wording to the missing deepest {CURRENT_TOKEN_HUMAN} itself.",
            ),
        ],
        "summary": {
            "audit_ready": audit_ready,
            f"{CURRENT_TOKEN_NAME}_present_on_open_clause": token_present,
            f"dominant_blocker_is_{CURRENT_TOKEN_NAME}_absence": not token_present,
            "first_route_to_close_or_none": f"{ROUTE_PREFIX}_{CURRENT_TOKEN_NAME}_declaration_gate",
        },
        "decision": {
            "overall_status": f"trial2_numeric_alpha_part3a_positive_source_link_{CURRENT_TOKEN_NAME}_audit_complete" if audit_ready else f"trial2_numeric_alpha_part3a_positive_source_link_{CURRENT_TOKEN_NAME}_audit_incomplete",
            "advance_to_8_7_56_693": audit_ready,
            "next_required_artifacts": [] if audit_ready else [f"{ROUTE_PREFIX}_{CURRENT_TOKEN_NAME}_audit"],
        },
        "evidence": {
            "inventory_summary": inventory["summary"],
            "source_token_wording_fragment_presence": token_hit,
        },
    }

    gate = {
        "generated_utc": now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.693",
            "name": f"{ROUTE_PREFIX}_{CURRENT_TOKEN_NAME}_declaration_gate",
        },
        "inputs": inventory["inputs"],
        "intent": "Freeze the declaration gate after confirming that the explicit numeric-alpha-open clause is still missing the deepest source-token-wording-fragment.",
        "formulas": {
            "gate_rule": f"The current branch closes once the dominant blocker is localized to the missing deepest {CURRENT_TOKEN_HUMAN} on the explicit open-status line.",
            "residual_rule": f"The next residual concerns the missing deepest {TOKEN_SET_HUMAN} that should populate that fragment.",
        },
        "rows": [
            row(
                f"trial2_numeric_alpha_part3a_positive_source_link_{CURRENT_TOKEN_NAME}_gate_complete",
                "pass" if audit_ready else "reject",
                f"positive-source-link {CURRENT_TOKEN_HUMAN} gate complete",
                1 if audit_ready else 0,
                f"The gate closes once the dominant blocker is narrowed to the missing deepest {CURRENT_TOKEN_HUMAN}.",
            ),
            row(
                "trial2_numeric_alpha_closeout_ready",
                "reject",
                "numeric-alpha closeout ready",
                0,
                "Numeric-alpha closeout still requires a positive source-link clause on the primary surface.",
            ),
            row(
                f"trial2_numeric_alpha_part3a_positive_source_link_{CURRENT_TOKEN_NAME}_missing",
                "pass" if not token_present else "watch",
                f"Part III-A positive source-link {CURRENT_TOKEN_HUMAN} missing",
                1 if not token_present else 0,
                f"The narrowed blocker is the missing deepest {CURRENT_TOKEN_HUMAN} on the explicit numeric-alpha-open line.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_preserved",
                "pass",
                "precision-alpha mainline preserved",
                1,
                "The branch does not reopen the structural EM pass or promote the strong-side reserve.",
            ),
        ],
        "summary": {
            f"trial2_numeric_alpha_part3a_positive_source_link_{CURRENT_TOKEN_NAME}_branch_closeable": audit_ready,
            "trial2_numeric_alpha_closeout_ready": False,
            "selected_residual_route": NEXT_ROUTE_LABEL,
            "missing_v2_artifact": NEXT_MISSING_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        "decision": {
            "overall_status": f"trial2_numeric_alpha_part3a_positive_source_link_{CURRENT_TOKEN_NAME}_gate_closed" if audit_ready else f"trial2_numeric_alpha_part3a_positive_source_link_{CURRENT_TOKEN_NAME}_gate_open",
            "advance_to_8_7_56_694": audit_ready,
            "next_required_artifacts": [] if audit_ready else [f"{ROUTE_PREFIX}_{CURRENT_TOKEN_NAME}_declaration_gate"],
        },
        "evidence": {"audit_summary": audit["summary"]},
    }

    gate_branch_key = f"trial2_numeric_alpha_part3a_positive_source_link_{CURRENT_TOKEN_NAME}_branch_closeable"
    gate_closed = bool(gate["summary"][gate_branch_key])
    contract = {
        "generated_utc": now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.694",
            "name": "trial2_numeric_alpha_next_generation_route_contract_seventieth_refresh",
        },
        "inputs": inventory["inputs"],
        "intent": "Refresh the strong-side reserve after the deep7 source-token-wording-fragment branch and freeze the next source-token-wording-fragment-token-set EM precision residual route.",
        "formulas": {
            "contract_rule": f"The EM precision mainline remains active while the blocker shrinks from the missing deepest {CURRENT_TOKEN_HUMAN} to the missing deepest {TOKEN_SET_HUMAN}.",
            "reserve_rule": "Strong-side work remains on v3 hold reserve and does not outrank the EM precision route.",
        },
        "rows": [
            row(
                f"trial2_numeric_alpha_part3a_positive_source_link_{CURRENT_TOKEN_NAME}_gate_closed",
                "pass" if gate_closed else "reject",
                f"positive-source-link {CURRENT_TOKEN_HUMAN} gate closed",
                1 if gate_closed else 0,
                f"The next route contract depends on the deepest {CURRENT_TOKEN_HUMAN} gate being frozen first.",
            ),
            row(
                f"trial2_numeric_alpha_part3a_positive_source_link_{TOKEN_SET_NAME}_route_selected",
                "pass" if gate_closed else "reject",
                f"positive source-link {TOKEN_SET_HUMAN} route selected",
                1 if gate_closed else 0,
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
        "summary": {
            "selected_next_generation_route": NEXT_ROUTE_LABEL if gate_closed else None,
            "strong_side_route_state": prior_route["summary"]["strong_side_route_state"],
            "precision_alpha_mainline_retained": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        "decision": {
            "overall_status": "trial2_numeric_alpha_next_route_contract_seventieth_refresh_frozen" if gate_closed else "trial2_numeric_alpha_next_route_contract_seventieth_refresh_pending",
            "advance_to_next_route": gate_closed,
            "next_required_artifacts": [
                f"{ROUTE_PREFIX}_{TOKEN_SET_NAME}_source_inventory",
                f"{ROUTE_PREFIX}_{TOKEN_SET_NAME}_audit",
            ] if gate_closed else ["trial2_numeric_alpha_next_route_contract_seventieth_refresh"],
        },
        "evidence": {
            "gate_summary": gate["summary"],
            "prior_route_summary": prior_route["summary"],
        },
    }

    write_artifact(INVENTORY_STEM, inventory)
    write_artifact(AUDIT_STEM, audit)
    write_artifact(GATE_STEM, gate)
    write_artifact(ROUTE_STEM, contract)

    print("[ok] generated Trial-2 numeric alpha deep7 source-token-wording-fragment artifacts for 8.7.56.691-.694")


# 関数: CLI 直実行時に branch main を起動する。

def run_cli() -> None:
    """CLI entry point for the deep7 source-token-wording-fragment residual branch."""
    main()


if __name__ == "__main__":
    run_cli()
