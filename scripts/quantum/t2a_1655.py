#!/usr/bin/env python3
"""Generate 8.7.56.1655-.1658 primary decision gate / secondary audit artifacts.

This branch freezes the first-shot breakthrough instruction pack after both of
its primary computational surfaces have already been tested honestly:

1. the exact constitutive-map audit failed to close canonically,
2. the branch-local full nonlinear energy-density audit also stayed on the
   retained vector no-go scale.

The remaining task is therefore not another derivation layer. It is one honest
decision gate:

- Gate A: promote,
- Gate B: retain but not promote,
- Gate C: reserve.

Under the current frozen-action pack, Gate A does not open because no canonical
primary surface is available. Gate B is the honest result because
electric-like / note-gradient evidence-only surfaces do improve toward the
retained scalar candidate numerically, but they still do not close
canonically. The mainline therefore moves to the first fallback
`P_mu` transverse response / projected-kernel observable audit.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
LOCAL_RESPONSE = ROOT / "doc" / "quantum" / "50_trial2_vector_qball_breakthrough_instruction_response.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

EXTERNAL_NOTE = Path(
    r"C:\Users\ogawa\Downloads\50_trial2_numeric_alpha_vector_qball_breakthrough_instruction_pack.md"
)
ADVICE_PACK_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1643_1646_ed_reopen_advice_pack_declaration_gate_metrics.json"
)
CONSTITUTIVE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1647_1650_constitutive_map_audit_declaration_gate_metrics.json"
)
FULL_NL_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1651_1654_full_nl_energy_audit_declaration_gate_metrics.json"
)
ENERGY_FF_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1627_1630_energy_density_ff_audit_declaration_gate_metrics.json"
)
ENERGY_CASE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1631_1634_energy_density_alpha_case_class_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1655-1658"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor primary decision gate / "
    "secondary canonical-promotion audit"
)
STEM = build_compact_artifact_stem(STEP_TAG, "primary_decision_gate", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_branch_local_full_nonlinear_energy_candidates_"
    "track_vector_no_go_primary_decision_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_primary_gate_b_retain_not_promote_"
    "transverse_response_fallback_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_p_mu_transverse_response_"
    "projected_kernel_observable_audit"
)
NEXT_ROUTE = "8.7.56.1659"
SECOND_FALLBACK_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_constrained_ground_state_"
    "branch_selection_audit"
)
SECOND_FALLBACK_ROUTE = "8.7.56.1663"

SCALAR_ALPHA = 0.00715678583937324
TARGET_ALPHA = 1.0 / 137.035999084
VECTOR_ALPHA = 0.0005579616187042394


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 テキストを読み込む。

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# 関数: UTF-8 JSON を読み込む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: 表示用の相対パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分文字列に一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line matching one substring."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準 metrics row を構成する。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準 payload を構成する。

def payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Build one standard payload."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# 関数: JSON/CSV 成果物を書き出す。

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one rows CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "status", "metric", "value", "note"],
        )
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": display_path(paths["json"]), "csv": display_path(paths["csv"])}


# 関数: 真偽値を 0/1 に変換する。

def truth(value: bool) -> float:
    """Convert one boolean into 0/1 float form."""
    return 1.0 if value else 0.0


# 関数: 2つの alpha の絶対距離を返す。

def alpha_distance(value: float, reference: float) -> float:
    """Return the absolute alpha distance."""
    return float(abs(float(value) - float(reference)))


# 関数: `.1655-.1658` を実行する。

def main() -> None:
    """Execute the primary decision gate / secondary audit branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LOCAL_RESPONSE,
        PART5,
        EXTERNAL_NOTE,
        ADVICE_PACK_GATE,
        CONSTITUTIVE_GATE,
        FULL_NL_GATE,
        ENERGY_FF_GATE,
        ENERGY_CASE_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    local_text = read_text(LOCAL_RESPONSE)
    part5_text = read_text(PART5)
    external_text = read_text(EXTERNAL_NOTE)

    advice_pack_summary = read_json(ADVICE_PACK_GATE)["summary"]
    constitutive_summary = read_json(CONSTITUTIVE_GATE)["summary"]
    full_nl_summary = read_json(FULL_NL_GATE)["summary"]
    energy_ff_summary = read_json(ENERGY_FF_GATE)["summary"]
    energy_case_summary = read_json(ENERGY_CASE_GATE)["summary"]

    prior_pack_ready = bool(
        advice_pack_summary.get("breakthrough_instruction_pack_adopted", False)
        and constitutive_summary.get("trial2_numeric_alpha_problem_classification")
        == "vector_qball_form_factor_exact_constitutive_map_unavailable_branch_local_full_nonlinear_energy_audit_next"
        and full_nl_summary.get("trial2_numeric_alpha_problem_classification")
        == PRIOR_CLASS
    )
    primary_canonical_surface_available = bool(
        constitutive_summary.get("exact_constitutive_map_available", False)
        or full_nl_summary.get(
            "branch_local_full_nonlinear_energy_density_exact_available", False
        )
    )
    primary_scalar_support_available = bool(
        full_nl_summary.get("pilot_full_supports_scalar_candidate", False)
        or full_nl_summary.get("family_proxy_supports_scalar_candidate", False)
    )
    gate_a_promote_selected = bool(
        primary_canonical_surface_available and primary_scalar_support_available
    )

    electric_alpha = float(energy_ff_summary["electric_like_component_alpha_at_q_theory"])
    note_gradient_alpha = float(
        energy_ff_summary["note_gradient_alpha_at_q_theory"]
    )
    official_energy_alpha = float(energy_ff_summary["official_alpha_E_at_q_theory"])

    electric_d_scalar = alpha_distance(electric_alpha, SCALAR_ALPHA)
    electric_d_target = alpha_distance(electric_alpha, TARGET_ALPHA)
    electric_d_vec = alpha_distance(electric_alpha, VECTOR_ALPHA)
    note_gradient_d_scalar = alpha_distance(note_gradient_alpha, SCALAR_ALPHA)
    note_gradient_d_target = alpha_distance(note_gradient_alpha, TARGET_ALPHA)
    note_gradient_d_vec = alpha_distance(note_gradient_alpha, VECTOR_ALPHA)

    secondary_evidence_scalar_leaning = bool(
        electric_d_scalar < electric_d_vec and note_gradient_d_scalar < note_gradient_d_vec
    )
    secondary_canonical_promotion_supported = bool(
        secondary_evidence_scalar_leaning
        and energy_ff_summary.get("electric_like_improves_but_is_not_official", False)
        and constitutive_summary.get("exact_constitutive_map_available", False)
    )
    gate_b_retain_not_promote_selected = bool(
        not gate_a_promote_selected
        and secondary_evidence_scalar_leaning
        and not secondary_canonical_promotion_supported
    )
    gate_c_reserve_selected = bool(
        not gate_a_promote_selected and not gate_b_retain_not_promote_selected
    )
    primary_breakthrough_pack_failed = bool(not gate_a_promote_selected)
    transverse_response_fallback_required_now = bool(
        gate_b_retain_not_promote_selected or gate_c_reserve_selected
    )
    second_fallback_retained = bool(transverse_response_fallback_required_now)
    physical_reject_not_selected = bool(
        not advice_pack_summary.get("physical_reject_required", True)
        and not constitutive_summary.get("physical_reject_required", True)
        and not full_nl_summary.get("physical_reject_required", True)
        and not energy_ff_summary.get("physical_reject_required", True)
    )

    inventory_ready = all(
        item is not None
        for item in (
            hit(external_text, "#### Gate A: promote"),
            hit(external_text, "#### Gate B: retain but not promote"),
            hit(external_text, "#### Gate C: reserve"),
            hit(local_text, "primary decision gate / secondary canonical-promotion audit"),
            hit(status_text, "8.7.56.1655"),
            hit(roadmap_text, "8.7.56.1655-.1658"),
            hit(current_problem_text, "8.7.56.1655-.1658"),
            hit(current_status_text, "8.7.56.1655-.1658"),
            hit(
                unified_text,
                "`.1655-.1658` は **primary decision gate / secondary canonical-promotion audit**",
            ),
            hit(part5_text, "next mainline `.1655-.1658` is **primary decision gate / secondary canonical-promotion audit**"),
        )
    )
    primary_decision_gate_wording_honest = bool(
        inventory_ready
        and prior_pack_ready
        and not gate_a_promote_selected
        and gate_b_retain_not_promote_selected
        and not gate_c_reserve_selected
        and primary_breakthrough_pack_failed
        and transverse_response_fallback_required_now
        and physical_reject_not_selected
    )
    route_sync_ready = bool(primary_decision_gate_wording_honest)

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "primary decision gate inventory ready",
            truth(inventory_ready),
            "The decision gate only starts after the instruction pack, local response note, roadmap, and current notes all point to the same `.1655-.1658` branch.",
        ),
        row(
            "prior_pack_ready",
            "pass" if prior_pack_ready else "reject",
            "prior breakthrough pack ready",
            truth(prior_pack_ready),
            "The decision gate is only honest after the advice-pack adoption, constitutive-map no-go, and nonlinear-energy no-go have all been fixed explicitly.",
        ),
        row(
            "primary_canonical_surface_available",
            "pass" if primary_canonical_surface_available else "reject",
            "primary canonical surface available",
            truth(primary_canonical_surface_available),
            "Gate A requires either an exact constitutive map or an exact branch-local full nonlinear energy density. Under the current pack, both remain unavailable.",
        ),
        row(
            "primary_scalar_support_available",
            "pass" if primary_scalar_support_available else "reject",
            "primary scalar-support surface available",
            truth(primary_scalar_support_available),
            "Even before canonization, neither primary nonlinear candidate moves the blind fixed-q_theory read toward the retained scalar strong candidate.",
        ),
        row(
            "gate_a_promote_selected",
            "pass" if gate_a_promote_selected else "reject",
            "Gate A promote selected",
            truth(gate_a_promote_selected),
            "Gate A stays closed because the current frozen-action pack does not provide a canonical primary surface that supports the scalar strong candidate.",
        ),
        row(
            "secondary_evidence_scalar_leaning",
            "pass" if secondary_evidence_scalar_leaning else "reject",
            "secondary evidence surfaces are scalar-leaning",
            truth(secondary_evidence_scalar_leaning),
            "Electric-like and note-gradient evidence surfaces sit numerically closer to the retained scalar candidate than to the retained vector no-go scale, so they remain visible as secondary evidence.",
        ),
        row(
            "secondary_canonical_promotion_supported",
            "pass" if secondary_canonical_promotion_supported else "reject",
            "secondary canonical promotion supported",
            truth(secondary_canonical_promotion_supported),
            "Numerical improvement alone is insufficient. Without canonical closure, the secondary evidence surfaces cannot be promoted to an official observable under the current pack.",
        ),
        row(
            "gate_b_retain_not_promote_selected",
            "pass" if gate_b_retain_not_promote_selected else "reject",
            "Gate B retain-but-not-promote selected",
            truth(gate_b_retain_not_promote_selected),
            "Gate B is the honest current-pack read: the secondary evidence surfaces improve numerically, but the pack still lacks canonical promotion and therefore cannot claim breakthrough.",
        ),
        row(
            "gate_c_reserve_selected",
            "pass" if gate_c_reserve_selected else "reject",
            "Gate C reserve selected",
            truth(gate_c_reserve_selected),
            "Gate C stays closed because one internal current-pack fallback remains available before reserve-only wording becomes necessary.",
        ),
        row(
            "primary_breakthrough_pack_failed",
            "pass" if primary_breakthrough_pack_failed else "reject",
            "primary breakthrough pack failed",
            truth(primary_breakthrough_pack_failed),
            "The first-shot breakthrough pack fails under the current frozen-action pack once both primary surfaces close negatively and Gate A does not open.",
        ),
        row(
            "transverse_response_fallback_required_now",
            "pass" if transverse_response_fallback_required_now else "reject",
            "transverse-response fallback required now",
            truth(transverse_response_fallback_required_now),
            "Once Gate B is selected honestly, the next mainline is the projected-kernel / transverse-response observable audit.",
        ),
        row(
            "second_fallback_retained",
            "pass" if second_fallback_retained else "reject",
            "second fallback retained",
            truth(second_fallback_retained),
            "Ground-state / branch-selection remains available only after the projected-kernel fallback has also been tested honestly.",
        ),
        row(
            "physical_reject_not_selected",
            "pass" if physical_reject_not_selected else "reject",
            "physical reject not selected",
            truth(physical_reject_not_selected),
            "Even after the first-shot pack fails, the route remains local to the current pack and does not force physical rejection.",
        ),
        row(
            "primary_decision_gate_wording_honest",
            "pass" if primary_decision_gate_wording_honest else "reject",
            "primary decision gate wording honest",
            truth(primary_decision_gate_wording_honest),
            "The current decision is honest only if Gate A is closed, Gate B is selected, secondary evidence stays non-canonical, and the fallback ordering is made explicit.",
        ),
        row(
            "route_sync_ready",
            "pass" if route_sync_ready else "reject",
            "route sync ready",
            truth(route_sync_ready),
            "Once the Gate B read is frozen, the roadmap can move cleanly to the projected-kernel fallback without adding another density derivation loop.",
        ),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem": display_path(CURRENT_PROBLEM),
            "current_status": display_path(CURRENT_STATUS),
            "unified_roadmap": display_path(UNIFIED_ROADMAP),
            "local_response": display_path(LOCAL_RESPONSE),
            "part5": display_path(PART5),
            "external_note": display_path(EXTERNAL_NOTE),
            "advice_pack_gate": display_path(ADVICE_PACK_GATE),
            "constitutive_gate": display_path(CONSTITUTIVE_GATE),
            "full_nl_gate": display_path(FULL_NL_GATE),
            "energy_ff_gate": display_path(ENERGY_FF_GATE),
            "energy_case_gate": display_path(ENERGY_CASE_GATE),
        },
        "constants": {
            "scalar_alpha": SCALAR_ALPHA,
            "target_alpha": TARGET_ALPHA,
            "vector_alpha": VECTOR_ALPHA,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "second_fallback_route_name": SECOND_FALLBACK_ROUTE_NAME,
            "second_fallback_route": SECOND_FALLBACK_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_primary_decision_gate": "gate_b_retain_but_not_promote",
        "gate_a_promote_selected": gate_a_promote_selected,
        "gate_b_retain_not_promote_selected": gate_b_retain_not_promote_selected,
        "gate_c_reserve_selected": gate_c_reserve_selected,
        "primary_breakthrough_pack_failed": primary_breakthrough_pack_failed,
        "exact_constitutive_map_available": constitutive_summary[
            "exact_constitutive_map_available"
        ],
        "branch_local_full_nonlinear_energy_density_exact_available": full_nl_summary[
            "branch_local_full_nonlinear_energy_density_exact_available"
        ],
        "official_energy_core_alpha_at_q_theory": official_energy_alpha,
        "electric_like_component_alpha_at_q_theory": electric_alpha,
        "note_gradient_alpha_at_q_theory": note_gradient_alpha,
        "electric_like_d_scalar": electric_d_scalar,
        "electric_like_d_target": electric_d_target,
        "electric_like_d_vec": electric_d_vec,
        "note_gradient_d_scalar": note_gradient_d_scalar,
        "note_gradient_d_target": note_gradient_d_target,
        "note_gradient_d_vec": note_gradient_d_vec,
        "secondary_evidence_scalar_leaning": secondary_evidence_scalar_leaning,
        "secondary_canonical_promotion_supported": secondary_canonical_promotion_supported,
        "secondary_evidence_only_retained": gate_b_retain_not_promote_selected,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_second_fallback_route": SECOND_FALLBACK_ROUTE_NAME,
        "selected_second_fallback_route_or_none": SECOND_FALLBACK_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": route_sync_ready,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "hits": {
            "external_gate_a_hit": hit(external_text, "#### Gate A: promote"),
            "external_gate_b_hit": hit(
                external_text, "#### Gate B: retain but not promote"
            ),
            "external_gate_c_hit": hit(external_text, "#### Gate C: reserve"),
            "local_response_branch_hit": hit(
                local_text, "primary decision gate / secondary canonical-promotion audit"
            ),
            "status_branch_hit": hit(status_text, "8.7.56.1655"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1655-.1658"),
            "current_problem_branch_hit": hit(
                current_problem_text, "8.7.56.1655-.1658"
            ),
            "current_status_branch_hit": hit(
                current_status_text, "8.7.56.1655-.1658"
            ),
            "unified_roadmap_branch_hit": hit(
                unified_text,
                "`.1655-.1658` は **primary decision gate / secondary canonical-promotion audit**",
            ),
            "part5_branch_hit": hit(
                part5_text,
                "next mainline `.1655-.1658` is **primary decision gate / secondary canonical-promotion audit**",
            ),
        },
        "carry_over": {
            "advice_pack_summary": advice_pack_summary,
            "constitutive_summary": constitutive_summary,
            "full_nl_summary": full_nl_summary,
            "energy_ff_summary": energy_ff_summary,
            "energy_case_summary": energy_case_summary,
        },
        "retained_numeric_state": {
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "target_alpha": TARGET_ALPHA,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "official_energy_core_alpha_at_q_theory": official_energy_alpha,
            "electric_like_component_alpha_at_q_theory": electric_alpha,
            "note_gradient_alpha_at_q_theory": note_gradient_alpha,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1655",
                f"{STEP_NAME} inventory",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "audit": write_artifact(
            "audit",
            payload(
                "8.7.56.1656",
                f"{STEP_NAME} audit",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload(
                "8.7.56.1657",
                f"{STEP_NAME} declaration gate",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "route_sync": write_artifact(
            "route_sync",
            payload(
                "8.7.56.1658",
                f"{STEP_NAME} route sync",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
    }

    print(json.dumps({"stem": STEM, "manifest": manifest, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
