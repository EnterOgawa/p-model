#!/usr/bin/env python3
"""Generate 8.7.56.387-.390 clip-branch physical-admissibility-statement surface artifacts."""

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
FULL_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"
PRIOR_SOURCE = OUT / "mass_origin_v2_t3_t2_clip_branch_physical_admissibility_statement_source_inventory_metrics.json"
PRIOR_AUDIT = OUT / "mass_origin_v2_t3_t2_clip_branch_physical_admissibility_statement_audit_metrics.json"
PRIOR_GATE = OUT / "mass_origin_v2_t3_t2_clip_branch_physical_admissibility_statement_declaration_gate_metrics.json"
PRIOR_DISP = OUT / "mass_origin_v2_t3_t2_paper_sync_trial4_disp_39th_refresh_metrics.json"
ANCHOR = {"k": 17, "ell": 1, "s": 1}
NEXT_ROUTE = "8.7.56.391"
TRIAL2 = "unlocked_reserve_retained"
CLIP = ("clip branch", "clip-branch", "clipped branch", "zero-kappa clip", "zero_kappa clip")
PHYS = ("physical", "physically", "physical-status", "physical status", "物理")
ADMIT = ("admissible", "admissibility", "licensed", "license", "許容", "許可")
HEADERS = ("## 2.7", "## 4.16", "## 5.4", "## 5.5", "## 5.6")


# 関数: UTC 現在時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須ファイルの存在を確認する。

def req(path: Path) -> None:
    """Abort when a required input artifact is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: JSON を読む。

def read_json(path: Path) -> dict:
    """Load a UTF-8 JSON artifact into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: UTF-8 テキストを読む。

def read_text(path: Path) -> str:
    """Read a UTF-8 text source into memory."""
    return path.read_text(encoding="utf-8")


# 関数: repo 相対パスへ変換する。

def rel(path: Path) -> str:
    """Return a repo-relative POSIX-style path string."""
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: 部分文字列の最初のヒット行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line hit for a substring pattern, if any."""
    pattern_lower = pattern.lower()
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern_lower in line.lower():
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 複数 group を同時に満たす最初のヒット行を返す。

def multi_hit(text: str, groups: tuple[tuple[str, ...], ...]) -> dict | None:
    """Return the first line that contains at least one token from every group."""
    lowered_groups = tuple(tuple(token.lower() for token in group) for group in groups)
    for line_no, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        matched: list[str] = []
        for group, lowered_group in zip(groups, lowered_groups):
            chosen = next((token for token, lowered_token in zip(group, lowered_group) if lowered_token in lowered), None)
            if chosen is None:
                break

            matched.append(chosen)
        else:
            return {"groups": matched, "line": line_no, "text": line.strip()}

    return None


# 関数: 複数 group を同時に満たす行数を数える。

def multi_count(text: str, groups: tuple[tuple[str, ...], ...]) -> int:
    """Count lines that contain at least one token from every group."""
    lowered_groups = tuple(tuple(token.lower() for token in group) for group in groups)
    total = 0
    for line in text.splitlines():
        lowered = line.lower()
        if all(any(token in lowered for token in group) for group in lowered_groups):
            total += 1

    return total


# 関数: rows schema を作る。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build a standard metrics row payload."""
    return {"row_id": row_id, "status": status, "metric": metric, "value": float(value), "note": note}


# 関数: payload schema を作る。

def payload(step: str, name: str, inputs: dict, rows: list[dict], summary: dict, evidence: dict) -> dict:
    """Build the standard JSON metrics payload used across the roadmap."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": {"overall_status": name},
        "evidence": evidence,
    }


# 関数: JSON と CSV rows を保存する。

def write_artifact(stem: str, data: dict) -> None:
    """Write the metrics payload as JSON and as a rows CSV sidecar."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{stem}_metrics.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / f"{stem}_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# 関数: 状態辞書から重要項目だけを抜き出す。

def compact(state: dict | None) -> dict | None:
    """Return a compact subset of a state dictionary for evidence payloads."""
    if state is None:
        return None

    keys = (
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
    return {key: state[key] for key in keys if key in state}


# 関数: exact anchor が zero-kappa clip signature に乗っているか判定する。

def on_zero_kappa_clip(state: dict) -> bool:
    """Return True when a state sits on the clipped zero-kappa exact-anchor branch."""
    return bool(
        float(state["beta_n"]) > 1.0
        and float(state["polarization_weight"]) == 0.0
        and float(state["coupled_charge_factor"]) == 1.0
        and float(state["coupled_mass_factor"]) == 1.0
    )


# 関数: branch 本体を実行する。

def main() -> None:
    """Execute the clip-branch physical-admissibility-statement surface branch."""
    for path in (STATUS, ROADMAP, AI_CONTEXT, PART1, PART3A, PART5, FULL_SOLVER, PRIOR_SOURCE, PRIOR_AUDIT, PRIOR_GATE, PRIOR_DISP):
        req(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    full_text = read_text(FULL_SOLVER)
    prior_source = read_json(PRIOR_SOURCE)
    prior_audit = read_json(PRIOR_AUDIT)
    prior_gate = read_json(PRIOR_GATE)
    prior_disp = read_json(PRIOR_DISP)

    exact_w = prior_source["evidence"]["exact_best_w_or_none"]
    exact_z = prior_source["evidence"]["exact_best_z_or_none"]
    sub_w = prior_source["evidence"]["subunity_best_w_or_none"]
    sub_z = prior_source["evidence"]["subunity_best_z_or_none"]
    sub_pair = prior_source["evidence"]["subunity_best_pair_or_none"]

    exact_clip = bool(on_zero_kappa_clip(exact_w) and on_zero_kappa_clip(exact_z))
    clip_rule = hit(full_text, "localization = math.sqrt(max(0.0, 1.0 - beta_n * beta_n))")
    required = bool(prior_audit["summary"]["clip_branch_physical_admissibility_statement_surface_dominant_blocker"])

    phys_count = (
        multi_count(part1_text, (CLIP, PHYS))
        + multi_count(part3a_text, (CLIP, PHYS))
        + multi_count(part5_text, (CLIP, PHYS))
    )
    admit_count = (
        multi_count(part1_text, (CLIP, ADMIT))
        + multi_count(part3a_text, (CLIP, ADMIT))
        + multi_count(part5_text, (CLIP, ADMIT))
    )
    phys_admit_count = (
        multi_count(part1_text, (CLIP, PHYS, ADMIT))
        + multi_count(part3a_text, (CLIP, PHYS, ADMIT))
        + multi_count(part5_text, (CLIP, PHYS, ADMIT))
    )
    part1_header = next((hit(part1_text, header) for header in HEADERS if hit(part1_text, header) is not None), None)
    part3a_header = next((hit(part3a_text, header) for header in HEADERS if hit(part3a_text, header) is not None), None)
    part5_header = next((hit(part5_text, header) for header in HEADERS if hit(part5_text, header) is not None), None)
    part1_candidate_surface = bool(part1_header is not None)
    part3a_candidate_surface = bool(part3a_header is not None)
    part5_candidate_surface = bool(part5_header is not None)
    primary_surface_missing = bool(part1_candidate_surface and phys_admit_count == 0)
    placement_blocker = bool(required and primary_surface_missing)
    closeable = bool(exact_clip and phys_admit_count > 0)

    inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "part1_core_theory_markdown": rel(PART1),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "part5_future_predictions_markdown": rel(PART5),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_SOLVER),
        "mass_origin_v2_t3_t2_clip_branch_physical_admissibility_statement_source_inventory_json": rel(PRIOR_SOURCE),
        "mass_origin_v2_t3_t2_clip_branch_physical_admissibility_statement_audit_json": rel(PRIOR_AUDIT),
        "mass_origin_v2_t3_t2_clip_branch_physical_admissibility_statement_declaration_gate_json": rel(PRIOR_GATE),
        "mass_origin_v2_t3_t2_paper_sync_trial4_disp_39th_refresh_json": rel(PRIOR_DISP),
    }

    src = payload(
        "8.7.56.387",
        "trial3_t2_clip_branch_physical_admissibility_statement_surface_source_inventory",
        inputs,
        [
            row("inventory_complete", "pass", "inventory complete", 1, "source pack frozen"),
            row("exact_anchor_zero_kappa_clip_signature_confirmed", "pass" if exact_clip else "reject", "exact anchors on zero-kappa clip branch", 1 if exact_clip else 0, "same-family exact W/Z anchors remain on beta>1 clip branch"),
            row("clip_branch_physical_admissibility_statement_surface_required_for_honest_closeout", "pass" if required else "reject", "surface required", 1 if required else 0, "current blocker has narrowed to the missing statement surface"),
            row("current_canon_clip_branch_physical_statement_surface_present", "pass" if phys_count > 0 else "reject", "clip-branch physical surface present", 1 if phys_count > 0 else 0, "no explicit physical statement surface found"),
            row("current_canon_clip_branch_admissibility_statement_surface_present", "pass" if admit_count > 0 else "reject", "clip-branch admissibility surface present", 1 if admit_count > 0 else 0, "no explicit admissibility statement surface found"),
            row("current_canon_clip_branch_physical_admissibility_statement_surface_present", "pass" if phys_admit_count > 0 else "reject", "combined physical-admissibility surface present", 1 if phys_admit_count > 0 else 0, "no surface yet licenses the clipped branch as physically admissible"),
            row("part1_primary_candidate_surface_present", "pass" if part1_candidate_surface else "reject", "Part I primary candidate surface present", 1 if part1_candidate_surface else 0, "Part I still contains canonical theory sections that could host the sentence"),
            row("subunity_pair_preserved", "pass" if sub_pair["passes_threshold"] else "reject", "beta<=1 pair preserved", 1 if sub_pair["passes_threshold"] else 0, "pair remains near-exact"),
            row("subunity_z_pass_preserved", "pass" if sub_z["passes_threshold"] else "reject", "beta<=1 Z preserved", 1 if sub_z["passes_threshold"] else 0, "Z still passes below unity"),
            row("subunity_w_miss_preserved", "reject" if not sub_w["passes_threshold"] else "pass", "beta<=1 W miss preserved", 1 if not sub_w["passes_threshold"] else 0, "W still misses below unity"),
        ],
        {
            "anchor_family_or_none": dict(ANCHOR),
            "exact_anchor_zero_kappa_clip_signature_confirmed": exact_clip,
            "clip_branch_physical_admissibility_statement_surface_required_for_honest_closeout": required,
            "current_canon_clip_branch_physical_statement_surface_line_count": phys_count,
            "current_canon_clip_branch_admissibility_statement_surface_line_count": admit_count,
            "current_canon_clip_branch_physical_admissibility_statement_surface_line_count": phys_admit_count,
            "part1_primary_candidate_surface_present": part1_candidate_surface,
            "part3a_candidate_surface_present": part3a_candidate_surface,
            "part5_candidate_surface_present": part5_candidate_surface,
            "next_required_route": "trial3_t2_clip_branch_physical_admissibility_statement_surface_audit",
        },
        {
            "status_next_step_line": hit(status_text, "current official next step は `8.7.56.387`"),
            "roadmap_branch_line": hit(roadmap_text, "`8.7.56.387-.390` 試練3 two-component clip-branch physical-admissibility-statement surface residual branch"),
            "clip_rule_line": clip_rule,
            "part1_primary_candidate_surface_line": part1_header,
            "part3a_candidate_surface_line": part3a_header,
            "part5_candidate_surface_line": part5_header,
            "part1_clip_branch_physical_admissibility_hit": multi_hit(part1_text, (CLIP, PHYS, ADMIT)),
            "part3a_clip_branch_physical_admissibility_hit": multi_hit(part3a_text, (CLIP, PHYS, ADMIT)),
            "part5_clip_branch_physical_admissibility_hit": multi_hit(part5_text, (CLIP, PHYS, ADMIT)),
            "exact_best_w_or_none": compact(exact_w),
            "exact_best_z_or_none": compact(exact_z),
            "subunity_best_w_or_none": compact(sub_w),
            "subunity_best_z_or_none": compact(sub_z),
            "subunity_best_pair_or_none": sub_pair,
        },
    )

    audit = payload(
        "8.7.56.388",
        "trial3_t2_clip_branch_physical_admissibility_statement_surface_audit",
        inputs,
        [
            row("audit_complete", "pass", "audit complete", 1, "audit frozen"),
            row("clip_branch_physical_admissibility_statement_surface_required_for_honest_closeout", "pass" if required else "reject", "surface required", 1 if required else 0, "current blocker has narrowed to statement surface placement"),
            row("combined_surface_present", "pass" if phys_admit_count > 0 else "reject", "combined physical-admissibility surface present", 1 if phys_admit_count > 0 else 0, "no explicit combined surface found"),
            row("part1_primary_candidate_surface_present", "pass" if part1_candidate_surface else "reject", "Part I primary candidate surface present", 1 if part1_candidate_surface else 0, "Part I remains the primary canonical placement candidate"),
            row("clip_branch_physical_admissibility_statement_surface_complete_under_current_canon", "pass" if phys_admit_count > 0 else "reject", "statement surface complete", 1 if phys_admit_count > 0 else 0, "current canon still lacks the required surface"),
            row("clip_branch_physical_admissibility_statement_surface_placement_dominant_blocker", "pass" if placement_blocker else "reject", "surface placement is dominant blocker", 1 if placement_blocker else 0, "the remaining issue is where the sentence belongs, not whether the branch exists"),
        ],
        {
            "anchor_family_or_none": dict(ANCHOR),
            "exact_anchor_zero_kappa_clip_signature_confirmed": exact_clip,
            "clip_branch_physical_admissibility_statement_surface_required_for_honest_closeout": required,
            "clip_branch_physical_admissibility_statement_surface_complete_under_current_canon": bool(phys_admit_count > 0),
            "clip_branch_physical_admissibility_statement_surface_placement_dominant_blocker": placement_blocker,
            "part1_primary_candidate_surface_present": part1_candidate_surface,
            "next_required_route": "trial3_t2_clip_branch_physical_admissibility_statement_surface_declaration_fifteenth_gate",
        },
        {
            "prior_source_summary": prior_source["summary"],
            "prior_audit_summary": prior_audit["summary"],
        },
    )

    gate = payload(
        "8.7.56.389",
        "trial3_t2_declaration_fifteenth_gate",
        inputs,
        [
            row("gate_complete", "pass", "gate complete", 1, "fifteenth gate frozen"),
            row("trial3_current_branch_closeable", "pass" if closeable else "reject", "branch closeable", 1 if closeable else 0, "close only if a physical-admissibility sentence surface exists"),
            row("residual_route_required", "reject" if closeable else "pass", "residual still required", 0 if closeable else 1, "surface placement is still unresolved"),
        ],
        {
            "solver_range_blocker_removed": True,
            "exact_anchor_zero_kappa_clip_signature_confirmed": exact_clip,
            "clip_branch_physical_admissibility_statement_surface_complete_under_current_canon": bool(phys_admit_count > 0),
            "trial3_current_branch_closeable": closeable,
            "selected_residual_route": None if closeable else "trial3_two_component_beta_above_unity_clip_branch_physical_admissibility_statement_surface_placement_identification",
            "missing_v2_artifact": None if closeable else "trial3_two_component_beta_above_unity_clip_branch_physical_admissibility_statement_surface_placement_pack",
            "recommended_next_route_or_none": None if closeable else NEXT_ROUTE,
        },
        {
            "audit_summary": audit["summary"],
            "prior_gate_summary": prior_gate["summary"],
            "current_ai_context_step": ai_context["current_step"],
        },
    )

    disp = payload(
        "8.7.56.390",
        "trial3_t2_paper_sync_trial4_disp_40th_refresh",
        inputs,
        [
            row("refresh_complete", "pass", "refresh complete", 1, "reserve/deferred ordering refreshed"),
            row("trial2_paper_side_sync_reserve_retained", "pass", "Trial-2 reserve retained", 1, "Trial-2 remains reserve work"),
            row("trial4_deferred_retained", "pass", "Trial-4 deferred retained", 1, "Trial-4 remains deferred"),
        ],
        {
            "selected_residual_route": gate["summary"]["selected_residual_route"],
            "missing_v2_artifact": gate["summary"]["missing_v2_artifact"],
            "trial2_paper_side_sync_state": TRIAL2,
            "trial4_deferred": True,
            "recommended_next_route_or_none": gate["summary"]["recommended_next_route_or_none"],
        },
        {
            "declaration_summary": gate["summary"],
            "prior_disposition_summary": prior_disp["summary"],
        },
    )

    write_artifact("mass_origin_v2_t3_t2_clip_branch_physical_admissibility_statement_surface_source_inventory", src)
    write_artifact("mass_origin_v2_t3_t2_clip_branch_physical_admissibility_statement_surface_audit", audit)
    write_artifact("mass_origin_v2_t3_t2_clip_branch_physical_admissibility_statement_surface_declaration_gate", gate)
    write_artifact("mass_origin_v2_t3_t2_paper_sync_trial4_disp_40th_refresh", disp)
    print("[done] clip-branch physical-admissibility-statement surface artifacts written")


# 関数: CLI 入口を実行する。

def run_cli() -> None:
    """CLI entry point for the clip-branch physical-admissibility-statement surface branch."""
    main()


if __name__ == "__main__":
    run_cli()
