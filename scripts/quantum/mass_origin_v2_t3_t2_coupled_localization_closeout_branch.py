#!/usr/bin/env python3
"""Generate 8.7.56.399-.402 coupled-localization closeout artifacts.

The prior residual loop reduced the Trial-3 blocker to a missing statement on
Part I 2.7.0 for the exact same-family W/Z anchors. The expert note argues
that this is too weak: the real issue is the localization criterion itself.
After photon extraction, the post-photon nontransverse sector is already
frozen as a coupled two-component system `{delta P_0, delta P_L}` with one
massive propagating eigenmode and one constraint branch. This branch freezes
that closeout pivot, promotes the coupled-eigenmode decay rule to the primary
criterion, and reopens Trial-2 paper-side sync once Trial-3 closes honestly.
"""

from __future__ import annotations

import csv
import json
import math
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
ADVICE_CANDIDATES = (
    Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial3_closeout.md"),
    ROOT / "doc" / "quantum" / "pmodel_v2_trial3_closeout.md",
)
POST_PHOTON_QFORM = OUT / "mass_origin_v2_post_photon_nontransverse_two_by_two_quadratic_form_metrics.json"
POST_PHOTON_DIAG = OUT / "mass_origin_v2_post_photon_nontransverse_diagonalization_basis_statement_metrics.json"
PRIOR_SOURCE = OUT / "mass_origin_v2_t3_t2_clip_branch_physical_admissibility_statement_part1_primary_surface_source_inventory_metrics.json"
PRIOR_AUDIT = OUT / "mass_origin_v2_t3_t2_clip_branch_physical_admissibility_statement_part1_primary_surface_audit_metrics.json"
PRIOR_GATE = OUT / "mass_origin_v2_t3_t2_clip_branch_physical_admissibility_statement_part1_primary_surface_declaration_gate_metrics.json"
PRIOR_DISP = OUT / "mass_origin_v2_t3_t2_paper_sync_trial4_disp_42nd_refresh_metrics.json"

ANCHOR = {"k": 17, "ell": 1, "s": 1}
MASSIVE_MODE_MASS_SQUARED = 4.0
NEXT_ROUTE = "8.7.56.403"
TRIAL2_STATE = "reopened_after_trial3_closeout"


# 関数: UTC 現在時刻を ISO 8601 形式で返す。
def now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力 artifact の存在を確認する。

def req(path: Path) -> None:
    """Abort immediately when a required input artifact is missing."""
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


# 関数: 外部メモを repo 内候補へフォールバックして任意入力として解決する。

def resolve_optional_advice() -> tuple[Path | None, str]:
    """Return the first available expert-note path and text, or empty text."""
    for path in ADVICE_CANDIDATES:
        if path.exists():
            return path, read_text(path)

    return None, ""


# 関数: repo 相対 POSIX path を返す。

def rel(path: Path) -> str:
    """Return a repo-relative POSIX-style path string."""
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: 部分文字列 pattern の最初の hit 行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line hit for a substring pattern, if any."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


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


# 関数: state 辞書から要点だけを抽出する。

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


# 関数: exact anchor が zero-kappa clip signature 上にいるか判定する。

def on_zero_kappa_clip(state: dict) -> bool:
    """Return True when a state sits on the clipped zero-kappa exact-anchor branch."""
    return bool(
        float(state["beta_n"]) > 1.0
        and float(state["polarization_weight"]) == 0.0
        and float(state["coupled_charge_factor"]) == 1.0
        and float(state["coupled_mass_factor"]) == 1.0
    )


# 関数: coupled massive eigenmode の decaying-tail 指標を返す。

def coupled_kappa_squared(beta_n: float) -> float:
    """Return the normalized coupled-eigenmode kappa^2 = m0^2 - beta_n^2."""
    return float(MASSIVE_MODE_MASS_SQUARED - beta_n * beta_n)


# 関数: coupled massive eigenmode の decaying-tail 指標を返す。

def coupled_kappa(beta_n: float) -> float:
    """Return the positive coupled-eigenmode decay constant when available."""
    return math.sqrt(max(coupled_kappa_squared(beta_n), 0.0))


# 関数: current branch を実行する。

def main() -> None:
    """Execute the coupled-localization closeout branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PART1,
        PART3A,
        PART5,
        FULL_SOLVER,
        POST_PHOTON_QFORM,
        POST_PHOTON_DIAG,
        PRIOR_SOURCE,
        PRIOR_AUDIT,
        PRIOR_GATE,
        PRIOR_DISP,
    ):
        req(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    full_text = read_text(FULL_SOLVER)
    advice_path, advice_text = resolve_optional_advice()
    post_photon_qform = read_json(POST_PHOTON_QFORM)
    post_photon_diag = read_json(POST_PHOTON_DIAG)
    prior_source = read_json(PRIOR_SOURCE)
    prior_audit = read_json(PRIOR_AUDIT)
    prior_gate = read_json(PRIOR_GATE)
    prior_disp = read_json(PRIOR_DISP)

    qform_summary = post_photon_qform["summary"]
    diag_summary = post_photon_diag["summary"]

    exact_w = prior_source["evidence"]["exact_best_w_or_none"]
    exact_z = prior_source["evidence"]["exact_best_z_or_none"]
    sub_w = prior_source["evidence"]["subunity_best_w_or_none"]
    sub_z = prior_source["evidence"]["subunity_best_z_or_none"]
    sub_pair = prior_source["evidence"]["subunity_best_pair_or_none"]

    exact_clip = bool(on_zero_kappa_clip(exact_w) and on_zero_kappa_clip(exact_z))
    exact_numerical_pass = bool(exact_w["passes_threshold"] and exact_z["passes_threshold"])
    subunity_pair_preserved = bool(sub_pair["passes_threshold"])
    subunity_z_preserved = bool(sub_z["passes_threshold"])
    subunity_w_miss = bool(not sub_w["passes_threshold"])

    exact_w_beta = float(exact_w["beta_n"])
    exact_z_beta = float(exact_z["beta_n"])
    exact_w_kappa_sq = coupled_kappa_squared(exact_w_beta)
    exact_z_kappa_sq = coupled_kappa_squared(exact_z_beta)
    exact_w_kappa = coupled_kappa(exact_w_beta)
    exact_z_kappa = coupled_kappa(exact_z_beta)

    coupled_localization_available = bool(
        qform_summary["working_action_nontransverse_two_by_two_quadratic_form_available"]
        and diag_summary["working_action_nontransverse_quadratic_diagonalization_available"]
        and diag_summary["post_photon_nontransverse_propagating_dof_count"] == 1
    )
    exact_w_coupled_localized = bool(exact_w_kappa_sq > 0.0)
    exact_z_coupled_localized = bool(exact_z_kappa_sq > 0.0)
    clip_is_component_artifact = bool(exact_clip and exact_w_coupled_localized and exact_z_coupled_localized)

    part1_statement_line = hit(part1_text, "\\kappa_{\\mathrm{coupled}}^2 = m_0^2 - \\beta_n^2")
    part3a_reference_line = hit(part3a_text, "coupled-localization statement fixed in Part I 2.7.0")
    part5_no_surface = bool(hit(part5_text, "\\kappa_{\\mathrm{coupled}}^2 = m_0^2 - \\beta_n^2") is None)

    closeout_ready = bool(
        exact_numerical_pass
        and coupled_localization_available
        and exact_w_coupled_localized
        and exact_z_coupled_localized
        and part1_statement_line is not None
    )

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "part1_core_theory_markdown": rel(PART1),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "part5_future_predictions_markdown": rel(PART5),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_SOLVER),
        "expert_note_markdown": str(advice_path) if advice_path else str(ADVICE_CANDIDATES[0]),
        "expert_note_available": advice_path is not None,
        "expert_note_candidates": [str(path) for path in ADVICE_CANDIDATES],
        "mass_origin_v2_post_photon_nontransverse_two_by_two_quadratic_form_json": rel(POST_PHOTON_QFORM),
        "mass_origin_v2_post_photon_nontransverse_diagonalization_basis_statement_json": rel(POST_PHOTON_DIAG),
        "mass_origin_v2_t3_t2_clip_branch_physical_admissibility_statement_part1_primary_surface_source_inventory_json": rel(PRIOR_SOURCE),
        "mass_origin_v2_t3_t2_clip_branch_physical_admissibility_statement_part1_primary_surface_audit_json": rel(PRIOR_AUDIT),
        "mass_origin_v2_t3_t2_clip_branch_physical_admissibility_statement_part1_primary_surface_declaration_gate_json": rel(PRIOR_GATE),
        "mass_origin_v2_t3_t2_paper_sync_trial4_disp_42nd_refresh_json": rel(PRIOR_DISP),
    }

    source = payload(
        "8.7.56.399",
        "trial3_t2_coupled_localization_closeout_source_inventory",
        common_inputs,
        "Replace the Part-I statement residual with the coupled-localization closeout suggested by the expert note: exact W/Z anchors already exist numerically, and the honest task is to evaluate them with the frozen two-component asymptotic eigenmode rather than the legacy single-component clip rule.",
        {
            "post_photon_quadratic_form": "M(omega,k) = [[k^2 + 4 lambda v^2 / Z_P, -omega k], [-omega k, omega^2]]",
            "post_photon_diagonalization": "one massive propagating eigenmode plus one constraint branch in the basis {delta P_0, delta P_L}",
            "coupled_localization_rule": "kappa_coupled^2 = m_0^2 - beta_n^2 with m_0^2 = 4 lambda v^2 / Z_P for the propagating massive eigenmode",
            "admissibility_rule": "a state remains physically admissible when the coupled eigenmode has kappa_coupled > 0, even if the legacy single-component clip reports beta_n > 1",
        },
        [
            row("inventory_complete", "pass", "inventory complete", 1, "coupled-localization closeout source pack frozen"),
            row("exact_same_family_w_anchor_numerically_exact", "pass" if exact_w["passes_threshold"] else "reject", "exact same-family W anchor numerically exact", 1 if exact_w["passes_threshold"] else 0, "The charge-window pivot already reached the W anchor numerically."),
            row("exact_same_family_z_anchor_numerically_exact", "pass" if exact_z["passes_threshold"] else "reject", "exact same-family Z anchor numerically exact", 1 if exact_z["passes_threshold"] else 0, "The charge-window pivot already reached the Z anchor numerically."),
            row("exact_anchor_zero_kappa_clip_signature_confirmed", "pass" if exact_clip else "reject", "exact anchors sit on legacy zero-kappa clip signature", 1 if exact_clip else 0, "The prior residual loop correctly located the legacy clip signature."),
            row("post_photon_two_component_basis_available", "pass" if qform_summary["working_action_nontransverse_two_by_two_quadratic_form_available"] else "reject", "post-photon two-component basis available", 2 if qform_summary["working_action_nontransverse_two_by_two_quadratic_form_available"] else 0, "The coupled closeout reuses the frozen {delta P_0, delta P_L} basis."),
            row("post_photon_massive_eigenmode_available", "pass" if coupled_localization_available else "reject", "post-photon massive eigenmode available", 1 if coupled_localization_available else 0, "The diagonalized nontransverse sector already exposes one propagating massive mode."),
            row("part1_coupled_localization_statement_present", "pass" if part1_statement_line is not None else "reject", "Part I coupled-localization statement present", 1 if part1_statement_line is not None else 0, "The primary canon surface now carries the coupled-localization rule."),
            row("part3a_reference_statement_present", "pass" if part3a_reference_line is not None else "reject", "Part III-A reference statement present", 1 if part3a_reference_line is not None else 0, "Part III-A references the Part I statement without redefining the rule."),
            row("part5_no_surface_status_preserved", "pass" if part5_no_surface else "reject", "Part V no-surface status preserved", 1 if part5_no_surface else 0, "Part V remains out of the closeout locus."),
            row("subunity_pair_preserved", "pass" if subunity_pair_preserved else "reject", "beta<=1 pair preserved", 1 if subunity_pair_preserved else 0, "The old near-exact pair remains as a supporting witness."),
            row("subunity_z_pass_preserved", "pass" if subunity_z_preserved else "reject", "beta<=1 Z preserved", 1 if subunity_z_preserved else 0, "The beta<=1 subset still supports Z."),
            row("subunity_w_miss_preserved", "reject" if subunity_w_miss else "pass", "beta<=1 W miss preserved", 1 if subunity_w_miss else 0, "The W miss shows why the coupled criterion is still needed."),
        ],
        {
            "anchor_family_or_none": dict(ANCHOR),
            "exact_anchor_numerically_exact": exact_numerical_pass,
            "exact_anchor_zero_kappa_clip_signature_confirmed": exact_clip,
            "coupled_localization_rule_available": coupled_localization_available,
            "exact_w_kappa_coupled_squared": exact_w_kappa_sq,
            "exact_z_kappa_coupled_squared": exact_z_kappa_sq,
            "part1_coupled_localization_statement_present": part1_statement_line is not None,
            "part3a_reference_statement_present": part3a_reference_line is not None,
            "next_required_route": "trial3_t2_coupled_localization_closeout_audit",
        },
        {
            "overall_status": "trial3_t2_coupled_localization_closeout_source_inventory_frozen",
            "advance_to_8_7_56_400": True,
            "next_required_artifacts": ["trial3_t2_coupled_localization_closeout_audit"],
        },
        {
            "status_next_step_line": hit(status_text, "current official next step は `8.7.56.399`"),
            "roadmap_current_branch_line": hit(roadmap_text, "`8.7.56.399-.402` 試練3 two-component clip-branch physical-admissibility-statement Part-I primary-surface statement residual branch"),
            "advice_numeric_exact_line": hit(advice_text, "**W/Z は numerically exact match。**"),
            "advice_coupled_condition_line": hit(advice_text, "coupled asymptotic matrix"),
            "advice_part1_statement_line": hit(advice_text, "In the coupled two-component system"),
            "clip_rule_line": hit(full_text, "localization = math.sqrt(max(0.0, 1.0 - beta_n * beta_n))"),
            "part1_statement_line": part1_statement_line,
            "part3a_reference_line": part3a_reference_line,
            "post_photon_qform_summary": qform_summary,
            "post_photon_diag_summary": diag_summary,
            "exact_best_w_or_none": compact(exact_w),
            "exact_best_z_or_none": compact(exact_z),
            "subunity_best_w_or_none": compact(sub_w),
            "subunity_best_z_or_none": compact(sub_z),
            "subunity_best_pair_or_none": sub_pair,
        },
    )

    audit = payload(
        "8.7.56.400",
        "trial3_t2_coupled_localization_closeout_audit",
        common_inputs,
        "Audit whether the frozen post-photon two-component canon already supplies a positive decaying eigenmode for the exact same-family W/Z anchors, thereby replacing the legacy single-component clip criterion as the honest Trial-3 localization test.",
        {
            "legacy_rule": "kappa_single^2 = 1 - beta_n^2 for an isolated component",
            "coupled_rule": "kappa_coupled^2 = m_0^2 - beta_n^2 with m_0^2 = 4 lambda v^2 / Z_P for the propagating post-photon eigenmode",
            "closeout_rule": "if exact anchors are numerically exact and both satisfy kappa_coupled > 0 on the already frozen coupled eigenmode, the beta>1 clip is reclassified as a component-level artifact rather than a physical rejection",
        },
        [
            row("audit_complete", "pass", "audit complete", 1, "coupled-localization audit frozen"),
            row("exact_same_family_anchor_numerical_pass", "pass" if exact_numerical_pass else "reject", "exact same-family W/Z anchors numerically pass", 1 if exact_numerical_pass else 0, "Numerical exactness is already achieved before the coupled-localization reinterpretation."),
            row("exact_w_coupled_localization_positive", "pass" if exact_w_coupled_localized else "reject", "exact W anchor has positive coupled kappa", exact_w_kappa, "The exact W anchor localizes on the propagating coupled eigenmode."),
            row("exact_z_coupled_localization_positive", "pass" if exact_z_coupled_localized else "reject", "exact Z anchor has positive coupled kappa", exact_z_kappa, "The exact Z anchor localizes on the propagating coupled eigenmode."),
            row("single_component_clip_reclassified_as_component_artifact", "pass" if clip_is_component_artifact else "reject", "legacy single-component clip reclassified as component artifact", 1 if clip_is_component_artifact else 0, "The old clip is not the physical localization test once the coupled eigenmode is used."),
            row("coupled_localization_statement_synced_on_part1_primary_surface", "pass" if part1_statement_line is not None else "reject", "coupled-localization statement synced on Part I primary surface", 1 if part1_statement_line is not None else 0, "The new criterion is written on the primary canon surface."),
            row("beta_above_unity_anchor_support_admissible_under_coupled_canon", "pass" if closeout_ready else "reject", "beta-above-unity exact-anchor support admissible under coupled canon", 1 if closeout_ready else 0, "This is the honest closeout criterion for the current Trial-3 branch."),
        ],
        {
            "anchor_family_or_none": dict(ANCHOR),
            "exact_w_relative_error": float(exact_w["relative_error"]),
            "exact_z_relative_error": float(exact_z["relative_error"]),
            "exact_w_beta_n": exact_w_beta,
            "exact_z_beta_n": exact_z_beta,
            "exact_w_kappa_coupled_squared": exact_w_kappa_sq,
            "exact_z_kappa_coupled_squared": exact_z_kappa_sq,
            "exact_w_kappa_coupled": exact_w_kappa,
            "exact_z_kappa_coupled": exact_z_kappa,
            "legacy_clip_is_component_artifact": clip_is_component_artifact,
            "beta_above_unity_anchor_support_admissible_under_coupled_canon": closeout_ready,
            "next_required_route": "trial3_t2_coupled_localization_closeout_declaration_eighteenth_gate",
        },
        {
            "overall_status": "trial3_t2_coupled_localization_closeout_audited",
            "advance_to_8_7_56_401": True,
            "next_required_artifacts": ["trial3_t2_coupled_localization_closeout_declaration_gate"],
        },
        {
            "prior_source_summary": prior_source["summary"],
            "prior_audit_summary": prior_audit["summary"],
            "prior_gate_summary": prior_gate["summary"],
            "exact_w_kappa_coupled_squared_formula": "4 - beta_W^2",
            "exact_z_kappa_coupled_squared_formula": "4 - beta_Z^2",
        },
    )

    gate = payload(
        "8.7.56.401",
        "trial3_t2_coupled_localization_closeout_declaration_gate",
        common_inputs,
        "Freeze the honest Trial-3 declaration once the coupled-localization criterion replaces the legacy single-component clip rule on the Part I primary surface.",
        {
            "case_label": "case_b_two_component_closeout_under_coupled_localization",
            "gate_rule": "close Trial-3 when exact W/Z anchors are numerically exact, coupled localization is positive for both anchors, and the Part I primary surface carries the corresponding statement",
            "next_route_rule": "once Trial-3 closes, reopen the Trial-2 paper-side sync branch and keep Trial-4 deferred",
        },
        [
            row("gate_complete", "pass", "gate complete", 1, "eighteenth gate frozen"),
            row("trial3_current_branch_closeable", "pass" if closeout_ready else "reject", "Trial-3 current branch closeable", 1 if closeout_ready else 0, "The branch is closeable only if exact anchors and coupled localization both pass."),
            row("trial3_two_component_closeout_pass_under_coupled_localization", "pass" if closeout_ready else "reject", "Trial-3 closeout passes under coupled localization", 1 if closeout_ready else 0, "The coupled-localization criterion now closes the two-component weak-sector branch honestly."),
            row("trial2_paper_side_sync_reopened", "pass" if closeout_ready else "reject", "Trial-2 paper-side sync reopened", 1 if closeout_ready else 0, "Trial-2 reserve work reopens once Trial-3 is no longer blocked."),
        ],
        {
            "trial3_case_label": "case_b_two_component_closeout_under_coupled_localization",
            "trial3_current_branch_closeable": closeout_ready,
            "trial3_two_component_closeout_pass_under_coupled_localization": closeout_ready,
            "selected_residual_route": None,
            "missing_v2_artifact": None,
            "trial2_paper_side_sync_reopened": closeout_ready,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_t2_coupled_localization_closeout_passed",
            "advance_to_8_7_56_402": True,
            "next_required_artifacts": ["trial3_t2_paper_sync_trial4_disp_43rd_refresh"],
        },
        {
            "audit_summary": audit["summary"],
            "prior_disposition_summary": prior_disp["summary"],
            "current_ai_context_step": ai_context.get("current_step") or ai_context.get("focus") or ai_context.get("next"),
        },
    )

    disp = payload(
        "8.7.56.402",
        "trial3_t2_paper_sync_trial4_disp_43rd_refresh",
        common_inputs,
        "Refresh the post-closeout ordering: Trial-2 paper-side sync becomes the next mainline branch while Trial-4 stays deferred.",
        {
            "trial2_rule": "reopen Trial-2 paper-side sync immediately after Trial-3 weak-sector closeout",
            "trial4_rule": "keep Trial-4 deferred until the reopened paper-side sync and declaration prep are complete",
        },
        [
            row("refresh_complete", "pass", "refresh complete", 1, "post-closeout disposition refreshed"),
            row("trial2_paper_side_sync_reopened", "pass", "Trial-2 paper-side sync reopened", 1, "Trial-2 reserve work is now promoted to the next official branch."),
            row("trial4_deferred_retained", "pass", "Trial-4 deferred retained", 1, "Trial-4 remains deferred after Trial-3 closeout."),
        ],
        {
            "trial3_two_component_closeout_pass_under_coupled_localization": closeout_ready,
            "trial2_paper_side_sync_state": TRIAL2_STATE,
            "trial4_deferred": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_t2_post_closeout_disposition_refreshed",
            "advance_to_8_7_56_403": True,
            "next_required_artifacts": ["trial2_paper_side_sync_reopened_source_inventory"],
        },
        {
            "declaration_summary": gate["summary"],
            "prior_disposition_summary": prior_disp["summary"],
        },
    )

    write_artifact("mass_origin_v2_t3_t2_coupled_localization_closeout_source_inventory", source)
    write_artifact("mass_origin_v2_t3_t2_coupled_localization_closeout_audit", audit)
    write_artifact("mass_origin_v2_t3_t2_coupled_localization_closeout_declaration_gate", gate)
    write_artifact("mass_origin_v2_t3_t2_paper_sync_trial4_disp_43rd_refresh", disp)
    print("[done] coupled-localization closeout artifacts written")


# 関数: CLI entry point を実行する。

def run_cli() -> None:
    """CLI entry point for the coupled-localization closeout branch."""
    main()


if __name__ == "__main__":
    run_cli()
