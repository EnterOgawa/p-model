#!/usr/bin/env python3
"""
Generate Trial-3 two-component beta-above-unity zero-kappa tail-statement artifacts for 8.7.56.375-.378.

This branch takes the post-reclassification blocker one step deeper than
8.7.56.371-.374. The exact same-family W/Z anchors are already numerically
identified as a zero-kappa clip branch, and the frozen boundary rule plus the
current solver implementation make that clip signature structurally explicit.
What current canon still does not contain is an explicit statement that this
clipped branch is physically admissible. The honest blocker therefore narrows
from a generic "tail statement pack" toward the missing clip-branch physical
statement itself.
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

PRIOR_SOURCE = OUT / "mass_origin_v2_t3_t2_zero_kappa_tail_reclassification_source_inventory_metrics.json"
PRIOR_AUDIT = OUT / "mass_origin_v2_t3_t2_zero_kappa_tail_reclassification_audit_metrics.json"
PRIOR_GATE = OUT / "mass_origin_v2_t3_t2_zero_kappa_tail_reclassification_declaration_gate_metrics.json"
PRIOR_DISPOSITION = OUT / "mass_origin_v2_t3_t2_paper_sync_trial4_disp_36th_refresh_metrics.json"
COUPLED_FREEZE = OUT / "mass_origin_vector_qball_coupled_constraint_freeze_audit_metrics.json"
FULL_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

ANCHOR_FAMILY = {"k": 17, "ell": 1, "s": 1}
TRIAL2_RESERVE_STATE = "unlocked_reserve_retained"
NEXT_ROUTE = "8.7.56.379"

ZERO_KAPPA_PATTERNS = (
    "zero-kappa",
    "zero kappa",
    "zero_kappa",
    "imaginary-kappa",
    "imaginary kappa",
    "imaginary_kappa",
)
CLIP_BRANCH_PATTERNS = (
    "clip branch",
    "clip-branch",
    "clipped branch",
    "zero-kappa clip",
    "zero_kappa clip",
)


# 関数: 現在の UTC 時刻を ISO 8601 形式で返す。
def now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力 artifact の存在を確認する。

def req(path: Path) -> None:
    """Abort when a required input artifact is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON artifact を辞書として読む。

def read_json(path: Path) -> dict:
    """Load a UTF-8 JSON artifact into a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: UTF-8 テキスト source を文字列として読む。

def read_text(path: Path) -> str:
    """Read a UTF-8 text source into memory."""
    return path.read_text(encoding="utf-8")


# 関数: 絶対パスを repo 相対表記へ変換する。

def rel(path: Path) -> str:
    """Return a repo-relative POSIX-style path string."""
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: source 内で最初に一致した pattern の行情報を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line hit for a substring pattern, if any."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 複数 pattern に対する最初の一致行を返す。

def first_any_hit(text: str, patterns: tuple[str, ...]) -> dict | None:
    """Return the first line hit for any substring pattern, if any."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pattern in patterns:
            if pattern in line:
                return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 複数 pattern に一致する行数を数える。

def count_line_hits(text: str, patterns: tuple[str, ...]) -> int:
    """Count the number of text lines that match at least one pattern."""
    total = 0
    for line in text.splitlines():
        if any(pattern in line for pattern in patterns):
            total += 1

    return total


# 関数: 共通 schema の metrics row を組み立てる。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build a standard metrics row payload."""
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


# 関数: compact state payload を読みやすい形へ整える。

def compact_state(state: dict | None) -> dict | None:
    """Return a compact subset of a state dictionary for evidence payloads."""
    if state is None:
        return None

    fields = (
        "n",
        "k",
        "ell",
        "s",
        "ratio_value",
        "relative_error",
        "passes_threshold",
        "beta_n",
        "polarization_weight",
        "coupled_charge_factor",
        "coupled_mass_factor",
        "mass_ratio_to_scalar_base",
    )
    return {field: state[field] for field in fields if field in state}


# 関数: exact-anchor state が zero-kappa clip signature を満たすかを判定する。

def has_zero_kappa_clip_signature(state: dict) -> bool:
    """Return True when a state sits on the clipped zero-kappa exact-anchor branch."""
    return bool(
        float(state["beta_n"]) > 1.0
        and float(state["polarization_weight"]) == 0.0
        and float(state["coupled_charge_factor"]) == 1.0
        and float(state["coupled_mass_factor"]) == 1.0
    )


# 関数: zero-kappa tail-statement branch を実行する。

def main() -> None:
    """Execute the zero-kappa tail-statement branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PART1,
        PART3A,
        PART5,
        PRIOR_SOURCE,
        PRIOR_AUDIT,
        PRIOR_GATE,
        PRIOR_DISPOSITION,
        COUPLED_FREEZE,
        FULL_SOLVER,
    ):
        req(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    prior_source = read_json(PRIOR_SOURCE)
    prior_audit = read_json(PRIOR_AUDIT)
    prior_gate = read_json(PRIOR_GATE)
    prior_disposition = read_json(PRIOR_DISPOSITION)
    coupled_freeze = read_json(COUPLED_FREEZE)
    full_text = read_text(FULL_SOLVER)

    exact_best_w = prior_source["evidence"]["exact_best_w_or_none"]
    exact_best_z = prior_source["evidence"]["exact_best_z_or_none"]
    subunity_best_w = prior_source["evidence"]["subunity_best_w_or_none"]
    subunity_best_z = prior_source["evidence"]["subunity_best_z_or_none"]
    subunity_best_pair = prior_source["evidence"]["subunity_best_pair_or_none"]

    exact_w_zero_kappa_signature_confirmed = has_zero_kappa_clip_signature(exact_best_w)
    exact_z_zero_kappa_signature_confirmed = has_zero_kappa_clip_signature(exact_best_z)
    exact_anchor_zero_kappa_clip_signature_confirmed = bool(
        exact_w_zero_kappa_signature_confirmed and exact_z_zero_kappa_signature_confirmed
    )
    clip_rule_line = hit(full_text, "localization = math.sqrt(max(0.0, 1.0 - beta_n * beta_n))")
    zero_kappa_tail_statement_structurally_inferable_from_current_canon = bool(
        exact_anchor_zero_kappa_clip_signature_confirmed
        and clip_rule_line is not None
        and "localized_boundary_rule" in prior_source["evidence"]["coupled_freeze_formulas"]
    )
    part1_zero_kappa_statement_hit = first_any_hit(part1_text, ZERO_KAPPA_PATTERNS)
    part3a_zero_kappa_statement_hit = first_any_hit(part3a_text, ZERO_KAPPA_PATTERNS)
    part5_zero_kappa_statement_hit = first_any_hit(part5_text, ZERO_KAPPA_PATTERNS)
    part1_clip_branch_statement_hit = first_any_hit(part1_text, CLIP_BRANCH_PATTERNS)
    part3a_clip_branch_statement_hit = first_any_hit(part3a_text, CLIP_BRANCH_PATTERNS)
    part5_clip_branch_statement_hit = first_any_hit(part5_text, CLIP_BRANCH_PATTERNS)
    current_canon_zero_kappa_statement_surface_line_count = (
        count_line_hits(part1_text, ZERO_KAPPA_PATTERNS)
        + count_line_hits(part3a_text, ZERO_KAPPA_PATTERNS)
        + count_line_hits(part5_text, ZERO_KAPPA_PATTERNS)
    )
    current_canon_clip_branch_statement_surface_line_count = (
        count_line_hits(part1_text, CLIP_BRANCH_PATTERNS)
        + count_line_hits(part3a_text, CLIP_BRANCH_PATTERNS)
        + count_line_hits(part5_text, CLIP_BRANCH_PATTERNS)
    )
    current_canon_has_explicit_zero_kappa_tail_statement = bool(
        current_canon_zero_kappa_statement_surface_line_count > 0
    )
    current_canon_has_explicit_clip_branch_physical_statement = bool(
        current_canon_clip_branch_statement_surface_line_count > 0
    )
    zero_kappa_tail_statement_pack_complete_under_current_canon = bool(
        current_canon_has_explicit_zero_kappa_tail_statement
        and current_canon_has_explicit_clip_branch_physical_statement
    )
    clip_branch_physical_statement_dominant_blocker = bool(
        zero_kappa_tail_statement_structurally_inferable_from_current_canon
        and not current_canon_has_explicit_clip_branch_physical_statement
    )
    branch_closeable = bool(
        exact_anchor_zero_kappa_clip_signature_confirmed
        and zero_kappa_tail_statement_pack_complete_under_current_canon
    )

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "part1_core_theory_markdown": rel(PART1),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "part5_future_predictions_markdown": rel(PART5),
        "mass_origin_v2_t3_t2_zero_kappa_tail_reclassification_source_inventory_json": rel(PRIOR_SOURCE),
        "mass_origin_v2_t3_t2_zero_kappa_tail_reclassification_audit_json": rel(PRIOR_AUDIT),
        "mass_origin_v2_t3_t2_zero_kappa_tail_reclassification_declaration_gate_json": rel(PRIOR_GATE),
        "mass_origin_v2_t3_t2_paper_sync_trial4_disp_36th_refresh_json": rel(PRIOR_DISPOSITION),
        "mass_origin_vector_qball_coupled_constraint_freeze_audit_json": rel(COUPLED_FREEZE),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_SOLVER),
    }

    source_inventory = payload(
        "8.7.56.375",
        "Trial-3 two-component beta-above-unity zero-kappa tail-statement source inventory",
        common_inputs,
        "Freeze the exact-anchor zero-kappa clip signature, the absence of explicit current-canon zero-kappa/clip-branch statements, and the surviving beta<=1 evidence in one statement-level source pack.",
        {
            "signature_rule": "the exact-anchor branch is numerically defined by beta_n > 1 together with polarization_weight = 0, coupled_charge_factor = 1, and coupled_mass_factor = 1",
            "statement_rule": "an honest current-canon closeout requires an explicit zero-kappa tail statement plus an explicit physical statement for the clipped branch",
            "split_rule": "if the clip signature is structurally inferable already but the physical statement is still absent, the residual narrows from a generic tail-statement pack to the missing clip-branch physical statement",
        },
        [
            row(
                "trial3_t2_zero_kappa_tail_statement_source_inventory_complete",
                "pass",
                "Trial-3 two-component beta-above-unity zero-kappa tail-statement source inventory complete",
                1,
                "The zero-kappa tail-statement source pack is frozen.",
            ),
            row(
                "trial3_t2_exact_anchor_zero_kappa_clip_signature_confirmed_in_statement_inventory",
                "pass" if exact_anchor_zero_kappa_clip_signature_confirmed else "reject",
                "exact same-family anchor zero-kappa clip signature confirmed",
                1 if exact_anchor_zero_kappa_clip_signature_confirmed else 0,
                "The current exact anchors are both carried by the same beta>1 zero-kappa clip signature.",
            ),
            row(
                "trial3_t2_zero_kappa_tail_statement_structurally_inferable_in_inventory",
                "pass" if zero_kappa_tail_statement_structurally_inferable_from_current_canon else "reject",
                "zero-kappa tail statement structurally inferable from frozen rule plus clip implementation",
                1 if zero_kappa_tail_statement_structurally_inferable_from_current_canon else 0,
                "The frozen imaginary-kappa rule and the current clip implementation already make the mathematical branch itself explicit enough to inspect.",
            ),
            row(
                "trial3_t2_current_canon_has_explicit_zero_kappa_tail_statement_in_inventory",
                "pass" if current_canon_has_explicit_zero_kappa_tail_statement else "reject",
                "current canon has explicit zero-kappa tail statement",
                1 if current_canon_has_explicit_zero_kappa_tail_statement else 0,
                "No explicit zero-kappa tail statement appears in Part I / III-A / V at present.",
            ),
            row(
                "trial3_t2_current_canon_has_explicit_clip_branch_physical_statement_in_inventory",
                "pass" if current_canon_has_explicit_clip_branch_physical_statement else "reject",
                "current canon has explicit clip-branch physical statement",
                1 if current_canon_has_explicit_clip_branch_physical_statement else 0,
                "No explicit sentence yet says that the clipped zero-kappa branch is physically admissible.",
            ),
            row(
                "trial3_t2_subunity_pair_preserved_in_statement_inventory",
                "pass" if subunity_best_pair["passes_threshold"] else "reject",
                "same-family beta<=1 pair preserved in statement inventory",
                1 if subunity_best_pair["passes_threshold"] else 0,
                "The near-exact beta<=1 pair remains preserved and is not the current blocker.",
            ),
            row(
                "trial3_t2_subunity_z_anchor_pass_preserved_in_statement_inventory",
                "pass" if subunity_best_z["passes_threshold"] else "reject",
                "same-family beta<=1 Z anchor pass preserved in statement inventory",
                1 if subunity_best_z["passes_threshold"] else 0,
                "The beta<=1 subset still preserves Z, so the open issue stays on the exact-anchor statement branch.",
            ),
            row(
                "trial3_t2_subunity_w_anchor_miss_preserved_in_statement_inventory",
                "reject" if not subunity_best_w["passes_threshold"] else "pass",
                "same-family beta<=1 W miss preserved in statement inventory",
                1 if not subunity_best_w["passes_threshold"] else 0,
                "The branch still cannot close honestly inside beta<=1, so the exact-anchor statement route remains active.",
            ),
        ],
        {
            "anchor_family_or_none": dict(ANCHOR_FAMILY),
            "exact_anchor_zero_kappa_clip_signature_confirmed": exact_anchor_zero_kappa_clip_signature_confirmed,
            "zero_kappa_tail_statement_structurally_inferable_from_current_canon": zero_kappa_tail_statement_structurally_inferable_from_current_canon,
            "current_canon_zero_kappa_statement_surface_line_count": current_canon_zero_kappa_statement_surface_line_count,
            "current_canon_clip_branch_statement_surface_line_count": current_canon_clip_branch_statement_surface_line_count,
            "current_canon_has_explicit_zero_kappa_tail_statement": current_canon_has_explicit_zero_kappa_tail_statement,
            "current_canon_has_explicit_clip_branch_physical_statement": current_canon_has_explicit_clip_branch_physical_statement,
            "next_required_route": "trial3_t2_zero_kappa_tail_statement_audit",
        },
        {
            "overall_status": "trial3_t2_zero_kappa_tail_statement_inventory_frozen",
            "advance_to_8_7_56_376": True,
            "next_required_artifacts": ["trial3_t2_zero_kappa_tail_statement_audit"],
        },
        {
            "status_next_step_line": hit(status_text, "current official next step は `8.7.56.375`"),
            "roadmap_branch_line": hit(
                roadmap_text,
                "`8.7.56.375-.378` 試練3 two-component beta-above-unity zero-kappa tail-statement residual branch",
            ),
            "clip_rule_line": clip_rule_line,
            "part1_zero_kappa_statement_hit": part1_zero_kappa_statement_hit,
            "part3a_zero_kappa_statement_hit": part3a_zero_kappa_statement_hit,
            "part5_zero_kappa_statement_hit": part5_zero_kappa_statement_hit,
            "part1_clip_branch_statement_hit": part1_clip_branch_statement_hit,
            "part3a_clip_branch_statement_hit": part3a_clip_branch_statement_hit,
            "part5_clip_branch_statement_hit": part5_clip_branch_statement_hit,
            "coupled_freeze_formulas": coupled_freeze["formulas"],
            "exact_best_w_or_none": compact_state(exact_best_w),
            "exact_best_z_or_none": compact_state(exact_best_z),
            "subunity_best_w_or_none": compact_state(subunity_best_w),
            "subunity_best_z_or_none": compact_state(subunity_best_z),
            "subunity_best_pair_or_none": subunity_best_pair,
        },
    )

    audit = payload(
        "8.7.56.376",
        "Trial-3 two-component beta-above-unity zero-kappa tail-statement audit",
        common_inputs,
        "Audit whether the current zero-kappa exact-anchor route still lacks a statement pack in canon, and whether the dominant remaining blocker has narrowed specifically to the missing clip-branch physical statement.",
        {
            "statement_pack_rule": "an honest current-canon closeout requires both an explicit zero-kappa tail statement and an explicit statement that the clipped branch is physically admissible",
            "dominant_blocker_rule": "if the mathematical branch is already structurally inferable from the frozen rule plus the clip implementation, the dominant remaining blocker is the missing physical-status statement for the clip branch",
        },
        [
            row(
                "trial3_t2_zero_kappa_tail_statement_audit_complete",
                "pass",
                "Trial-3 two-component beta-above-unity zero-kappa tail-statement audit complete",
                1,
                "The zero-kappa tail-statement audit is frozen.",
            ),
            row(
                "trial3_t2_zero_kappa_tail_statement_structurally_inferable_audit",
                "pass" if zero_kappa_tail_statement_structurally_inferable_from_current_canon else "reject",
                "zero-kappa tail statement structurally inferable from current canon plus solver",
                1 if zero_kappa_tail_statement_structurally_inferable_from_current_canon else 0,
                "The mathematical zero-kappa branch is already visible once the frozen rule is compared with the clip implementation.",
            ),
            row(
                "trial3_t2_current_canon_has_explicit_zero_kappa_tail_statement_audit",
                "pass" if current_canon_has_explicit_zero_kappa_tail_statement else "reject",
                "current canon has explicit zero-kappa tail statement",
                1 if current_canon_has_explicit_zero_kappa_tail_statement else 0,
                "The paper-side canon still lacks an explicit zero-kappa tail statement.",
            ),
            row(
                "trial3_t2_current_canon_has_explicit_clip_branch_physical_statement_audit",
                "pass" if current_canon_has_explicit_clip_branch_physical_statement else "reject",
                "current canon has explicit clip-branch physical statement",
                1 if current_canon_has_explicit_clip_branch_physical_statement else 0,
                "The paper-side canon still lacks an explicit physical-status statement for the clipped branch.",
            ),
            row(
                "trial3_t2_zero_kappa_tail_statement_pack_complete_under_current_canon",
                "pass" if zero_kappa_tail_statement_pack_complete_under_current_canon else "reject",
                "zero-kappa tail statement pack complete under current canon",
                1 if zero_kappa_tail_statement_pack_complete_under_current_canon else 0,
                "The statement pack remains incomplete until both explicit statements exist.",
            ),
            row(
                "trial3_t2_clip_branch_physical_statement_dominant_blocker",
                "pass" if clip_branch_physical_statement_dominant_blocker else "reject",
                "clip-branch physical statement is the dominant remaining blocker",
                1 if clip_branch_physical_statement_dominant_blocker else 0,
                "Because the mathematical branch is already inferable, the missing physical-status sentence is now the dominant blocker.",
            ),
        ],
        {
            "anchor_family_or_none": dict(ANCHOR_FAMILY),
            "exact_anchor_zero_kappa_clip_signature_confirmed": exact_anchor_zero_kappa_clip_signature_confirmed,
            "zero_kappa_tail_statement_structurally_inferable_from_current_canon": zero_kappa_tail_statement_structurally_inferable_from_current_canon,
            "current_canon_has_explicit_zero_kappa_tail_statement": current_canon_has_explicit_zero_kappa_tail_statement,
            "current_canon_has_explicit_clip_branch_physical_statement": current_canon_has_explicit_clip_branch_physical_statement,
            "zero_kappa_tail_statement_pack_complete_under_current_canon": zero_kappa_tail_statement_pack_complete_under_current_canon,
            "clip_branch_physical_statement_dominant_blocker": clip_branch_physical_statement_dominant_blocker,
            "next_required_route": "trial3_t2_zero_kappa_tail_statement_declaration_twelfth_gate",
        },
        {
            "overall_status": "trial3_t2_zero_kappa_tail_statement_audited",
            "advance_to_8_7_56_377": True,
            "next_required_artifacts": ["trial3_t2_zero_kappa_tail_statement_declaration_twelfth_gate"],
        },
        {
            "prior_audit_summary": prior_audit["summary"],
            "part1_zero_kappa_statement_hit": part1_zero_kappa_statement_hit,
            "part3a_zero_kappa_statement_hit": part3a_zero_kappa_statement_hit,
            "part5_zero_kappa_statement_hit": part5_zero_kappa_statement_hit,
            "part1_clip_branch_statement_hit": part1_clip_branch_statement_hit,
            "part3a_clip_branch_statement_hit": part3a_clip_branch_statement_hit,
            "part5_clip_branch_statement_hit": part5_clip_branch_statement_hit,
            "exact_best_w_or_none": compact_state(exact_best_w),
            "exact_best_z_or_none": compact_state(exact_best_z),
        },
    )

    declaration_gate = payload(
        "8.7.56.377",
        "Trial-3 two-component declaration twelfth gate",
        common_inputs,
        "Freeze whether the current exact-anchor gain closes Trial-3, or whether the honest next blocker is now the missing clip-branch physical statement for the structurally visible zero-kappa branch.",
        {
            "closeout_rule": "close Trial-3 only if the structurally visible zero-kappa branch is also explicitly licensed as physically admissible in current canon",
            "residual_rule": "if the branch is already structurally visible but still lacks a physical-status sentence, the next blocker is the missing clip-branch physical statement itself",
        },
        [
            row(
                "trial3_t2_declaration_twelfth_gate_complete",
                "pass",
                "Trial-3 two-component declaration twelfth gate complete",
                1,
                "The twelfth gate is frozen.",
            ),
            row(
                "trial3_t2_branch_closeable_twelfth_gate",
                "pass" if branch_closeable else "reject",
                "two-component branch closeable after zero-kappa tail-statement audit",
                1 if branch_closeable else 0,
                "The branch closes only if current canon explicitly licenses the structurally visible clipped branch physically.",
            ),
            row(
                "trial3_t2_residual_route_required_twelfth_gate",
                "reject" if branch_closeable else "pass",
                "two-component residual route still required after zero-kappa tail-statement audit",
                0 if branch_closeable else 1,
                "A residual route remains required while the clip-branch physical statement is absent.",
            ),
        ],
        {
            "solver_range_blocker_removed": True,
            "exact_anchor_zero_kappa_clip_signature_confirmed": exact_anchor_zero_kappa_clip_signature_confirmed,
            "zero_kappa_tail_statement_structurally_inferable_from_current_canon": zero_kappa_tail_statement_structurally_inferable_from_current_canon,
            "zero_kappa_tail_statement_pack_complete_under_current_canon": zero_kappa_tail_statement_pack_complete_under_current_canon,
            "trial3_current_branch_closeable": branch_closeable,
            "selected_residual_route": (
                None
                if branch_closeable
                else "trial3_two_component_beta_above_unity_clip_branch_physical_statement_identification"
            ),
            "missing_v2_artifact": (
                None
                if branch_closeable
                else "trial3_two_component_beta_above_unity_clip_branch_physical_statement"
            ),
            "recommended_next_route_or_none": None if branch_closeable else NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_t2_declaration_twelfth_gate_frozen",
            "trial3_branch_closeable": branch_closeable,
            "advance_to_8_7_56_378": True,
            "next_required_artifacts": [] if branch_closeable else [NEXT_ROUTE],
        },
        {
            "audit_summary": audit["summary"],
            "prior_gate_summary": prior_gate["summary"],
            "current_ai_context_step": ai_context["current_step"],
        },
    )

    disposition = payload(
        "8.7.56.378",
        "Trial-2 paper-side sync / Trial-4 disposition thirty-seventh refresh",
        common_inputs,
        "Refresh the reserve/deferred ordering after the zero-kappa tail-statement audit narrows the blocker to the missing clip-branch physical statement for the structurally visible beta>1 zero-kappa branch.",
        {
            "trial2_rule": "Trial-2 paper-side sync remains unlocked reserve retained while Trial-3 still has an honest current-canon residual route",
            "trial4_rule": "Trial-4 remains deferred while the two-component Trial-3 route is still scientifically live",
        },
        [
            row(
                "trial3_t2_trial2_trial4_thirty_seventh_refresh_complete",
                "pass",
                "Trial-2 paper-side sync / Trial-4 disposition thirty-seventh refresh complete",
                1,
                "The reserve/deferred ordering is refreshed after the zero-kappa tail-statement audit.",
            ),
            row(
                "trial3_t2_trial2_reserve_retained_thirty_seventh_refresh",
                "pass",
                "Trial-2 paper-side sync reserve retained",
                1,
                "Trial-2 paper sync remains reserve work while the Trial-3 clip-branch-physical-statement route is still open.",
            ),
            row(
                "trial3_t2_trial4_deferred_retained_thirty_seventh_refresh",
                "pass",
                "Trial-4 deferred retained",
                1,
                "Trial-4 stays deferred while the two-component Trial-3 route remains live.",
            ),
        ],
        {
            "selected_residual_route": declaration_gate["summary"]["selected_residual_route"],
            "missing_v2_artifact": declaration_gate["summary"]["missing_v2_artifact"],
            "trial2_paper_side_sync_state": TRIAL2_RESERVE_STATE,
            "trial4_deferred": True,
            "recommended_next_route_or_none": declaration_gate["summary"]["recommended_next_route_or_none"],
        },
        {
            "overall_status": "trial3_t2_trial2_trial4_thirty_seventh_refresh_frozen",
            "trial3_branch_closeable": branch_closeable,
            "advance_to_next_branch": not branch_closeable,
            "next_required_artifacts": [] if branch_closeable else [NEXT_ROUTE],
        },
        {
            "declaration_summary": declaration_gate["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
        },
    )

    write_artifact("mass_origin_v2_t3_t2_zero_kappa_tail_statement_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_t3_t2_zero_kappa_tail_statement_audit", audit)
    write_artifact("mass_origin_v2_t3_t2_zero_kappa_tail_statement_declaration_gate", declaration_gate)
    write_artifact("mass_origin_v2_t3_t2_paper_sync_trial4_disp_37th_refresh", disposition)

    print("[done] Trial-3 two-component zero-kappa tail-statement artifacts written:")
    print(" - mass_origin_v2_t3_t2_zero_kappa_tail_statement_source_inventory_metrics.json")
    print(" - mass_origin_v2_t3_t2_zero_kappa_tail_statement_audit_metrics.json")
    print(" - mass_origin_v2_t3_t2_zero_kappa_tail_statement_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t3_t2_paper_sync_trial4_disp_37th_refresh_metrics.json")


# 関数: CLI から branch を実行する。

def run_cli() -> None:
    """CLI entry point for the Trial-3 zero-kappa tail-statement branch."""
    main()


if __name__ == "__main__":
    run_cli()
