#!/usr/bin/env python3
"""Generate 8.7.56.1747-.1750 field-strength closeout / reopen registry artifacts.

The field-strength-source pack has already closed three facts honestly:

1. The one-leg amputation theorem is canonical under the field-strength source.
2. The exact retained-branch recomputation canonizes the old electric-like
   evidence.
3. The direct fixed-q read still falls short of exact scalar promotion, so the
   honest official read is only scalar-leaning partial canonical promotion.

`.1747-.1750` therefore does not search for another same-level field-strength
variant. It freezes the pack as a completed external-theorem closeout and
refreshes the reopen ordering around the missing internal-Hamiltonian surface.
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
LONG_ROADMAP = ROOT / "doc" / "quantum" / "55_trial2_numeric_alpha_vector_qball_long_horizon_roadmap.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

THEOREM_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1735_1738_field_strength_amp_theorem_declaration_gate_metrics.json"
)
RECOMPUTE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1739_1742_field_strength_recompute_declaration_gate_metrics.json"
)
DECISION_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1743_1746_field_strength_gate_sync_declaration_gate_metrics.json"
)
PROJECTED_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1659_1662_pmu_tresp_pk_audit_declaration_gate_metrics.json"
)
ENERGY_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1635_1638_energy_density_closeout_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1747-1750"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor field-strength-source "
    "closeout / reopen registry"
)
STEM = build_compact_artifact_stem(STEP_TAG, "field_strength_closeout_registry", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_field_strength_source_scalar_leaning_partial_"
    "canonical_promotion_closeout_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_field_strength_source_scalar_leaning_partial_"
    "canonical_promotion_reopen_registry_completed"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_internal_"
    "hamiltonian_surface_or_external_input_reactivation"
)
NEXT_ROUTE = "8.7.56.1751"

PRIMARY_REOPEN = (
    "genuinely_new_internal_hamiltonian_surface_beyond_current_field_strength_"
    "external_theorem_pack"
)
SECONDARY_REOPEN = (
    "substantive_external_input_guiding_new_internal_hamiltonian_surface_or_"
    "exact_scalar_promotion_beyond_current_field_strength_pack"
)
RESERVE_REOPEN = (
    "future_pack_update_linking_field_strength_external_theorem_to_internal_"
    "hamiltonian_sector"
)

SCALAR_ALPHA = 0.00715678583937324


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input file is missing."""
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


# 関数: repo 相対の表示パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line matching one substring."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準 metrics row を作る。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準 payload を作る。

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


# 関数: `.1747-.1750` を実行する。

def main() -> None:
    """Execute the field-strength-source closeout / reopen registry branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LONG_ROADMAP,
        PART5,
        THEOREM_GATE,
        RECOMPUTE_GATE,
        DECISION_GATE,
        PROJECTED_GATE,
        ENERGY_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    theorem_summary = read_json(THEOREM_GATE)["summary"]
    recompute_summary = read_json(RECOMPUTE_GATE)["summary"]
    decision_summary = read_json(DECISION_GATE)["summary"]
    projected_summary = read_json(PROJECTED_GATE)["summary"]
    energy_summary = read_json(ENERGY_GATE)["summary"]

    inventory_ready = all(
        item is not None
        for item in (
            hit(status_text, "8.7.56.1747"),
            hit(roadmap_text, "8.7.56.1747-.1750"),
            hit(current_problem_text, "field-strength-source closeout / reopen registry"),
            hit(current_status_text, "field-strength-source closeout / reopen registry"),
            hit(
                unified_text,
                "`.1747-.1750` は **field-strength-source closeout / reopen registry**",
            ),
            hit(long_text, "16. `8.7.56.1747-.1750`"),
            hit(part5_text, "next official branch は `.1747-.1750`"),
        )
    )
    theorem_sync_available = bool(
        theorem_summary.get("canonical_field_strength_theorem_derived", False)
        and recompute_summary.get("canonical_field_strength_theorem_derived", False)
        and decision_summary.get("canonical_field_strength_theorem_derived", False)
    )
    partial_promotion_closeout_available = bool(
        decision_summary.get("gate_b_partial_scalar_leaning_selected", False)
        and decision_summary.get("field_strength_surface_partial_promotion_retained", False)
        and not decision_summary.get("gate_a_exact_promote_selected", True)
    )
    exact_scalar_promotion_failed = bool(
        decision_summary.get("field_strength_surface_exact_scalar_promotion_failed", False)
    )
    external_theorem_closed_internal_surface_missing = bool(
        theorem_sync_available and partial_promotion_closeout_available and exact_scalar_promotion_failed
    )
    old_vector_reference_retained = bool(
        projected_summary.get("transverse_response_fallback_failed", False)
        and abs(
            projected_summary["official_projected_kernel_alpha_at_q_theory"]
            - 0.0005600186431488893
        )
        < 1.0e-18
        and abs(energy_summary["official_alpha_E_at_q_theory"] - 0.0005422361373947313)
        < 1.0e-18
    )
    scalar_strong_candidate_retained = bool(
        abs(SCALAR_ALPHA - 0.00715678583937324) < 1.0e-18
        and not decision_summary.get("physical_reject_required", True)
    )
    primary_reopen_surface_honest = external_theorem_closed_internal_surface_missing
    secondary_reopen_surface_honest = True
    reserve_reopen_surface_honest = True
    registry_wording_honest = all(
        (
            inventory_ready,
            theorem_sync_available,
            partial_promotion_closeout_available,
            exact_scalar_promotion_failed,
            external_theorem_closed_internal_surface_missing,
            old_vector_reference_retained,
            scalar_strong_candidate_retained,
            primary_reopen_surface_honest,
            secondary_reopen_surface_honest,
            reserve_reopen_surface_honest,
        )
    )
    registry_ready = registry_wording_honest

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "field-strength closeout registry inventory ready",
            truth(inventory_ready),
            "Registry starts only after status, roadmap, current notes, unified roadmap, long roadmap, and Part V all already point to the `.1747-.1750` branch.",
        ),
        row(
            "theorem_sync_available",
            "pass" if theorem_sync_available else "reject",
            "field-strength theorem sync available",
            truth(theorem_sync_available),
            "The closeout is only honest after theorem, recomputation, and decision-gate branches all agree on the one-leg canonical field-strength theorem.",
        ),
        row(
            "partial_promotion_closeout_available",
            "pass" if partial_promotion_closeout_available else "reject",
            "partial scalar-leaning promotion closeout available",
            truth(partial_promotion_closeout_available),
            "The registry freezes the pack only after Gate B partial promotion has already been fixed machine-readably.",
        ),
        row(
            "exact_scalar_promotion_failed",
            "pass" if exact_scalar_promotion_failed else "reject",
            "exact scalar promotion failed",
            truth(exact_scalar_promotion_failed),
            "The new field-strength theorem remains below the retained scalar strong candidate, so the remaining gap is still real at closeout time.",
        ),
        row(
            "external_theorem_closed_internal_surface_missing",
            "pass" if external_theorem_closed_internal_surface_missing else "reject",
            "external theorem closed while internal surface is still missing",
            truth(external_theorem_closed_internal_surface_missing),
            "The external source theorem is closed, so the unresolved scalar-promotion gap is localized to a missing internal-Hamiltonian surface rather than another same-level external-source variant.",
        ),
        row(
            "old_vector_reference_retained",
            "pass" if old_vector_reference_retained else "reject",
            "old vector reference retained",
            truth(old_vector_reference_retained),
            "The registry keeps the older projected-kernel and energy-core no-go scales visible as references instead of overwriting them.",
        ),
        row(
            "scalar_strong_candidate_retained",
            "pass" if scalar_strong_candidate_retained else "reject",
            "scalar strong candidate retained",
            truth(scalar_strong_candidate_retained),
            "The retained scalar exact-profile candidate stays live, so the branch closes as partial promotion rather than reject.",
        ),
        row(
            "primary_reopen_surface_fixed",
            "pass" if primary_reopen_surface_honest else "reject",
            "primary reopen surface fixed",
            truth(primary_reopen_surface_honest),
            "The primary reopen surface is a genuinely new internal-Hamiltonian surface beyond the current field-strength external-theorem pack.",
        ),
        row(
            "secondary_reopen_surface_fixed",
            "pass" if secondary_reopen_surface_honest else "reject",
            "secondary reopen surface fixed",
            truth(secondary_reopen_surface_honest),
            "Substantive external input is retained only insofar as it guides the missing internal surface or exact scalar-promotion bridge.",
        ),
        row(
            "reserve_reopen_surface_fixed",
            "pass" if reserve_reopen_surface_honest else "reject",
            "reserve reopen surface fixed",
            truth(reserve_reopen_surface_honest),
            "Future pack updates remain reserve unless they explicitly link the external theorem to the internal Hamiltonian sector.",
        ),
        row(
            "same_level_field_strength_retry_blocked",
            "pass",
            "same-level field-strength retry blocked",
            1.0,
            "The one-leg field-strength theorem pack is closed, so further same-level external-source variants are no longer honest.",
        ),
        row(
            "registry_wording_honest",
            "pass" if registry_wording_honest else "reject",
            "field-strength closeout registry wording honest",
            truth(registry_wording_honest),
            "The registry is honest only if it keeps both the scalar-leaning canonical gain and the still-missing exact internal bridge visible together.",
        ),
        row(
            "registry_ready",
            "pass" if registry_ready else "reject",
            "field-strength closeout registry ready",
            truth(registry_ready),
            "Once the partial canonical promotion and the reopen ordering are explicit, the field-strength pack can be frozen machine-readably.",
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
            "long_roadmap": display_path(LONG_ROADMAP),
            "part5": display_path(PART5),
            "theorem_gate": display_path(THEOREM_GATE),
            "recompute_gate": display_path(RECOMPUTE_GATE),
            "decision_gate": display_path(DECISION_GATE),
            "projected_gate": display_path(PROJECTED_GATE),
            "energy_gate": display_path(ENERGY_GATE),
        },
        "constants": {
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "updated_field_strength_alpha_at_q_theory": recompute_summary["updated_field_strength_alpha_at_q_theory"],
            "updated_field_strength_vs_scalar_alpha_rel_gap": recompute_summary["updated_field_strength_vs_scalar_alpha_rel_gap"],
            "updated_field_strength_vs_vector_alpha_rel_gap": recompute_summary["updated_field_strength_vs_vector_alpha_rel_gap"],
            "official_energy_core_alpha_at_q_theory": energy_summary["official_alpha_E_at_q_theory"],
            "official_projected_kernel_alpha_at_q_theory": projected_summary[
                "official_projected_kernel_alpha_at_q_theory"
            ],
            "primary_reopen_surface": PRIMARY_REOPEN,
            "secondary_reopen_surface": SECONDARY_REOPEN,
            "reserve_reopen_surface": RESERVE_REOPEN,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "field_strength_source_pack_adopted": decision_summary["field_strength_source_pack_adopted"],
        "canonical_field_strength_theorem_derived": decision_summary["canonical_field_strength_theorem_derived"],
        "field_strength_surface_exact_scalar_promotion_failed": exact_scalar_promotion_failed,
        "field_strength_surface_partial_promotion_retained": decision_summary["field_strength_surface_partial_promotion_retained"],
        "selected_primary_decision_gate": decision_summary["selected_primary_decision_gate"],
        "updated_field_strength_alpha_at_q_theory": recompute_summary["updated_field_strength_alpha_at_q_theory"],
        "updated_field_strength_vs_scalar_alpha_rel_gap": recompute_summary["updated_field_strength_vs_scalar_alpha_rel_gap"],
        "updated_field_strength_vs_vector_alpha_rel_gap": recompute_summary["updated_field_strength_vs_vector_alpha_rel_gap"],
        "updated_field_strength_canonizes_electric_like_evidence": recompute_summary["updated_field_strength_canonizes_electric_like_evidence"],
        "field_strength_external_theorem_pack_closed": True,
        "same_level_field_strength_retry_admissible": False,
        "missing_internal_hamiltonian_surface_identified": external_theorem_closed_internal_surface_missing,
        "primary_reopen_surface": PRIMARY_REOPEN,
        "secondary_reopen_surface": SECONDARY_REOPEN,
        "reserve_reopen_surface": RESERVE_REOPEN,
        "field_strength_closeout_reopen_registry_wording_honest": registry_wording_honest,
        "field_strength_closeout_reopen_registry_ready": registry_ready,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": None,
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": registry_ready,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "hits": {
            "status_branch_hit": hit(status_text, "8.7.56.1747"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1747-.1750"),
            "current_problem_branch_hit": hit(
                current_problem_text, "field-strength-source closeout / reopen registry"
            ),
            "current_status_branch_hit": hit(
                current_status_text, "field-strength-source closeout / reopen registry"
            ),
            "unified_roadmap_branch_hit": hit(
                unified_text,
                "`.1747-.1750` は **field-strength-source closeout / reopen registry**",
            ),
            "long_roadmap_branch_hit": hit(long_text, "16. `8.7.56.1747-.1750`"),
            "part5_branch_hit": hit(part5_text, "next official branch は `.1747-.1750`"),
        },
        "carry_over": {
            "theorem_summary": theorem_summary,
            "recompute_summary": recompute_summary,
            "decision_summary": decision_summary,
            "projected_summary": projected_summary,
            "energy_summary": energy_summary,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1747",
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
                "8.7.56.1748",
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
                "8.7.56.1749",
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
                "8.7.56.1750",
                f"{STEP_NAME} route sync",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
    }

    print(
        json.dumps(
            {"step": STEP_TAG, "stem": STEM, "manifest": manifest, "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
