#!/usr/bin/env python3
"""Generate 8.7.56.1635-.1638 energy-density disposition-sync / closeout artifacts.

This branch does not derive a new observable. The preceding branches already
fixed the essential current-pack read:

1. The exact Hamiltonian-core observable is available and canonical.
2. Its exact form factor tracks the retained vector no-go scale rather than the
   retained scalar strong candidate.
3. Electric-like / note-gradient improvements exist, but only as evidence-only
   non-canonical or norm-subleading surfaces.
4. Branch-local full nonlinear energy density is still unavailable.

The present task is therefore to synchronize one honest official disposition:

- Case II vector-no-go-like under the current pack is retained,
- evidence-only improvement surfaces are retained but not promoted,
- prior caseA worsen / ground-state no-go / caseB no-metric-rescue remain
  visible,
- the nonlinear energy-density / constitutive-map gap is frozen as the primary
  reopen surface.
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
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

CLASS_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1631_1634_energy_density_alpha_case_class_declaration_gate_metrics.json"
)
FF_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1627_1630_energy_density_ff_audit_declaration_gate_metrics.json"
)
DERIV_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1623_1626_energy_density_audit_declaration_gate_metrics.json"
)
CASEA_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1599_1602_v2_sub_exact_treat_declaration_gate_metrics.json"
)
GROUND_STATE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1615_1618_gs_nodeless_audit_declaration_gate_metrics.json"
)
CASEB_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1619_1622_eff_metric_v2_sub_restore_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1635-1638"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor energy-density disposition "
    "sync / closeout"
)
STEM = build_compact_artifact_stem(STEP_TAG, "energy_density_closeout", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_energy_density_case_ii_vector_no_go_like_"
    "disposition_sync_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_energy_density_case_ii_vector_no_go_like_"
    "closeout_sync_completed"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_energy_density_closeout_"
    "reopen_registry"
)
NEXT_ROUTE = "8.7.56.1639"
PRIMARY_REOPEN_SURFACE = (
    "branch_local_full_nonlinear_energy_density_or_exact_constitutive_map_gap"
)


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


# 関数: `.1635-.1638` を実行する。

def main() -> None:
    """Execute the energy-density disposition-sync / closeout branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PART5,
        CLASS_GATE,
        FF_GATE,
        DERIV_GATE,
        CASEA_GATE,
        GROUND_STATE_GATE,
        CASEB_GATE,
    ):
        require(path)

    class_payload = read_json(CLASS_GATE)
    ff_payload = read_json(FF_GATE)
    deriv_payload = read_json(DERIV_GATE)
    casea_payload = read_json(CASEA_GATE)
    ground_state_payload = read_json(GROUND_STATE_GATE)
    caseb_payload = read_json(CASEB_GATE)

    class_summary = class_payload["summary"]
    ff_summary = ff_payload["summary"]
    deriv_summary = deriv_payload["summary"]
    casea_summary = casea_payload["summary"]
    ground_state_summary = ground_state_payload["summary"]
    caseb_summary = caseb_payload["summary"]

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part5_text = read_text(PART5)

    prior_classification_ready = bool(
        class_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and class_summary.get("case_ii_vector_no_go_like_selected", False)
        and not class_summary.get("case_i_scalar_rescue_selected", True)
    )
    exact_energy_core_read_retained = bool(
        ff_summary.get("official_surface_name") == "epsilon_H_core"
        and ff_summary.get("energy_core_tracks_vector_no_go_scale", False)
        and not ff_summary.get("energy_core_supports_scalar_candidate", True)
        and not ff_summary.get("energy_core_exact_foundation_supported", True)
    )
    evidence_only_improvement_surfaces_retained = bool(
        class_summary.get("noncanonical_improvement_surfaces_retained", False)
        and ff_summary.get("electric_like_improves_but_is_not_official", False)
    )
    full_nonlinear_energy_density_reopen_retained = bool(
        class_summary.get("full_nonlinear_energy_density_reopen_retained", False)
        and not ff_summary.get("branch_local_full_energy_density_available", True)
    )
    prior_casea_worsen_retained = bool(casea_summary.get("worsen_selected", False))
    ground_state_note_no_go_retained = bool(
        not ground_state_summary.get(
            "ground_state_nodeless_hypothesis_supported_under_current_pack", True
        )
    )
    prior_caseb_no_metric_rescue_retained = bool(
        not caseb_summary.get("metric_artifact_rescue_supported", True)
    )
    physical_reject_not_selected = bool(
        not class_summary.get("physical_reject_required", True)
        and not ff_summary.get("physical_reject_required", True)
        and not deriv_summary.get("physical_reject_required", True)
        and not casea_summary.get("physical_reject_required", True)
        and not ground_state_summary.get("physical_reject_required", True)
        and not caseb_summary.get("physical_reject_required", True)
    )

    inventory_ready = all(
        item is not None
        for item in (
            hit(status_text, "energy-density disposition sync / closeout"),
            hit(roadmap_text, "energy-density disposition sync / closeout"),
            hit(current_problem_text, "energy-density disposition sync / closeout"),
            hit(current_status_text, "energy-density disposition sync / closeout"),
            hit(unified_roadmap_text, "`.1635-.1638` は **energy-density disposition sync / closeout**"),
            hit(part5_text, "**energy-density disposition sync / closeout**"),
            hit(status_text, "Case II vector-no-go-like"),
            hit(current_problem_text, "Case II vector-no-go-like"),
            hit(current_status_text, "Case II vector-no-go-like"),
        )
    )
    energy_density_closeout_wording_honest = bool(
        inventory_ready
        and prior_classification_ready
        and exact_energy_core_read_retained
        and evidence_only_improvement_surfaces_retained
        and full_nonlinear_energy_density_reopen_retained
        and prior_casea_worsen_retained
        and ground_state_note_no_go_retained
        and prior_caseb_no_metric_rescue_retained
        and physical_reject_not_selected
    )
    energy_density_closeout_sync_ready = bool(energy_density_closeout_wording_honest)
    energy_density_reopen_registry_admissible_now = bool(
        energy_density_closeout_sync_ready
    )

    official_f_value = float(class_summary["official_F_E_at_q_theory"])
    official_alpha = float(class_summary["official_alpha_E_at_q_theory"])
    official_residual = float(class_summary["official_alpha_E_residual_rel"])
    electric_like_alpha = float(
        class_summary["electric_like_component_alpha_at_q_theory"]
    )
    note_gradient_alpha = float(class_summary["note_gradient_alpha_at_q_theory"])

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "energy-density closeout inventory ready",
            truth(inventory_ready),
            "Closeout sync starts only after status, roadmap, current notes, and Part V all already point to the energy-density disposition branch.",
        ),
        row(
            "prior_classification_ready",
            "pass" if prior_classification_ready else "reject",
            "prior Case II classification ready",
            truth(prior_classification_ready),
            "The closeout branch is only honest after the energy-core read has already been classified as Case II vector-no-go-like under the current pack.",
        ),
        row(
            "exact_energy_core_read_retained",
            "pass" if exact_energy_core_read_retained else "reject",
            "exact Hamiltonian-core Case II read retained",
            truth(exact_energy_core_read_retained),
            "The official closeout must keep the exact energy-core observable as vector-no-go-like rather than reopening scalar rescue wording.",
        ),
        row(
            "evidence_only_improvement_surfaces_retained",
            "pass" if evidence_only_improvement_surfaces_retained else "reject",
            "evidence-only improvement surfaces retained",
            truth(evidence_only_improvement_surfaces_retained),
            "Electric-like and note-gradient improvements remain visible only as evidence-only surfaces and do not replace the official exact core read.",
        ),
        row(
            "full_nonlinear_energy_density_reopen_retained",
            "pass" if full_nonlinear_energy_density_reopen_retained else "reject",
            "branch-local full nonlinear energy-density reopen retained",
            truth(full_nonlinear_energy_density_reopen_retained),
            "The exact branch-local nonlinear energy density is still unavailable, so the honest closeout must keep it as the primary reopen surface.",
        ),
        row(
            "prior_casea_worsen_retained",
            "pass" if prior_casea_worsen_retained else "reject",
            "prior caseA worsen retained",
            truth(prior_casea_worsen_retained),
            "The energy-density closeout must not hide that the Minkowski-contracted subtraction lane already worsened badly.",
        ),
        row(
            "ground_state_note_no_go_retained",
            "pass" if ground_state_note_no_go_retained else "reject",
            "ground-state / nodeless no-go retained",
            truth(ground_state_note_no_go_retained),
            "The current pack still does not support the nodeless ground-state rescue, so that no-go remains part of the honest closeout pack.",
        ),
        row(
            "prior_caseb_no_metric_rescue_retained",
            "pass" if prior_caseb_no_metric_rescue_retained else "reject",
            "prior caseB no-metric-rescue retained",
            truth(prior_caseb_no_metric_rescue_retained),
            "The effective-metric recomputation also failed to rescue the scalar candidate and must stay explicit in the closeout wording.",
        ),
        row(
            "physical_reject_not_selected",
            "pass" if physical_reject_not_selected else "reject",
            "physical reject not selected",
            truth(physical_reject_not_selected),
            "Even after the energy-density branch closes as Case II vector-no-go-like, the route remains local and does not force physical rejection.",
        ),
        row(
            "energy_density_closeout_wording_honest",
            "pass" if energy_density_closeout_wording_honest else "reject",
            "energy-density closeout wording honest",
            truth(energy_density_closeout_wording_honest),
            "The closeout wording is honest only if Case II, the evidence-only surfaces, prior failed rescue lanes, and the nonlinear reopen gap all remain visible together.",
        ),
        row(
            "energy_density_closeout_sync_ready",
            "pass" if energy_density_closeout_sync_ready else "reject",
            "energy-density closeout sync ready",
            truth(energy_density_closeout_sync_ready),
            "Once the branch-level official read is aligned, the current-pack energy-density lane can be declared closed out.",
        ),
        row(
            "energy_density_reopen_registry_admissible_now",
            "pass" if energy_density_reopen_registry_admissible_now else "reject",
            "energy-density reopen registry admissible now",
            truth(energy_density_reopen_registry_admissible_now),
            "After the closeout wording is frozen, the next honest action is to register the retained reopen surface machine-readably.",
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
            "part5": display_path(PART5),
            "classification_gate": display_path(CLASS_GATE),
            "energy_ff_gate": display_path(FF_GATE),
            "energy_deriv_gate": display_path(DERIV_GATE),
            "casea_gate": display_path(CASEA_GATE),
            "ground_state_gate": display_path(GROUND_STATE_GATE),
            "caseb_gate": display_path(CASEB_GATE),
        },
        "constants": {
            "official_surface_name": "epsilon_H_core",
            "primary_reopen_surface": PRIMARY_REOPEN_SURFACE,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_disposition_case": "case_ii_vector_no_go_like_under_current_pack",
        "official_surface_name": "epsilon_H_core",
        "official_F_E_at_q_theory": official_f_value,
        "official_alpha_E_at_q_theory": official_alpha,
        "official_alpha_E_residual_rel": official_residual,
        "case_ii_vector_no_go_like_retained": exact_energy_core_read_retained,
        "evidence_only_improvement_surfaces_retained": evidence_only_improvement_surfaces_retained,
        "electric_like_component_alpha_at_q_theory": electric_like_alpha,
        "note_gradient_alpha_at_q_theory": note_gradient_alpha,
        "full_nonlinear_energy_density_reopen_retained": full_nonlinear_energy_density_reopen_retained,
        "prior_casea_worsen_retained": prior_casea_worsen_retained,
        "ground_state_note_no_go_retained": ground_state_note_no_go_retained,
        "prior_caseb_no_metric_rescue_retained": prior_caseb_no_metric_rescue_retained,
        "energy_density_closeout_wording_honest": energy_density_closeout_wording_honest,
        "energy_density_closeout_sync_ready": energy_density_closeout_sync_ready,
        "selected_primary_reopen_surface": PRIMARY_REOPEN_SURFACE,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": energy_density_closeout_sync_ready,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "hits": {
            "status_closeout": hit(status_text, "energy-density disposition sync / closeout"),
            "roadmap_closeout": hit(roadmap_text, "energy-density disposition sync / closeout"),
            "current_problem_closeout": hit(
                current_problem_text, "energy-density disposition sync / closeout"
            ),
            "current_status_closeout": hit(
                current_status_text, "energy-density disposition sync / closeout"
            ),
            "unified_roadmap_closeout": hit(
                unified_roadmap_text,
                "`.1635-.1638` は **energy-density disposition sync / closeout**",
            ),
            "part5_closeout": hit(
                part5_text, "**energy-density disposition sync / closeout**"
            ),
            "current_problem_case_ii": hit(
                current_problem_text, "Case II vector-no-go-like"
            ),
            "current_status_case_ii": hit(
                current_status_text, "Case II vector-no-go-like"
            ),
        },
        "carry_over": {
            "classification_summary": class_summary,
            "energy_ff_summary": ff_summary,
            "energy_deriv_summary": deriv_summary,
            "casea_summary": casea_summary,
            "ground_state_summary": ground_state_summary,
            "caseb_summary": caseb_summary,
        },
        "retained_numeric_state": {
            "scalar_alpha_exact_at_q_theory": 0.00715678583937324,
            "vector_alpha_at_q_theory": 0.0005579616187042394,
            "official_energy_core_alpha_at_q_theory": official_alpha,
            "electric_like_component_alpha_at_q_theory": electric_like_alpha,
            "note_gradient_alpha_at_q_theory": note_gradient_alpha,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1635",
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
                "8.7.56.1636",
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
                "8.7.56.1637",
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
                "8.7.56.1638",
                f"{STEP_NAME} route sync",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
    }

    print(json.dumps({"step": STEP_TAG, "stem": STEM, "artifacts": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
