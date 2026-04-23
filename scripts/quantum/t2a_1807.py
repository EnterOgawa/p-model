#!/usr/bin/env python3
"""Generate 8.7.56.1807-.1810 global HH obstruction theorem artifacts.

`.1799-.1802` derived a scalar-compatible full-q HH window under the retained
vacuum-saturated carrier surface

    A_FF(q) = |q|.

`.1803-.1806` then froze the remaining gap as either

1. a global all-q HH diagonal surface, or
2. a genuinely non-rank-one mixed surface.

This branch sharpens that reopen surface by proving a no-go theorem:

For any real symmetric mixed response matrix

    M(q) = [[A_FF(q), A_FH(q)],
            [A_FH(q), A_HH(q)]],

with `A_HH(q) >= 0`, the largest eigenvalue satisfies

    lambda_+(q) >= max(A_FF(q), A_HH(q)) >= A_FF(q).

Therefore, once `A_FF(q)=|q|` is retained, no positive-semidefinite mixed pack
can reproduce `lambda_+(q)=F_exact(q)` on any domain where `F_exact(q) < |q|`.
Since the full-q window edge is defined by `F_exact(q_HH,max)=q_HH,max`, the
global completion gap is not merely "non-rank-one": it requires breaking at
least one of the retained axioms (carrier saturation, PSD HH diagonal, or the
canonical Hermitian eigenvalue rule).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np


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

WINDOW_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1799_1802_full_q_hh_window_generalization_declaration_gate_metrics.json"
WINDOW_ROUTE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1799_1802_full_q_hh_window_generalization_route_sync_metrics.json"
CLOSEOUT_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1803_1806_full_q_hh_window_closeout_registry_declaration_gate_metrics.json"
QBALL_BRANCH_REFRESH = PUBLIC_OUT / "mass_origin_qball_charge_mapping_branch_refresh_metrics.json"
QBALL_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_qball_charge_mapping_branch.py"

STEP_TAG = "8.7.56.1807-1810"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor vacuum-saturated PSD "
    "mixed-pack global completion obstruction theorem"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "global_hh_obstruction_theorem",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_vacuum_saturated_full_q_hh_window_closeout_"
    "global_hh_or_non_rank_one_reopen_registry_completed"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_global_completion_obstructed_under_vacuum_"
    "saturated_psd_mixed_pack_route_reset_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_global_completion_"
    "obstruction_closeout_reopen_registry"
)
NEXT_ROUTE = "8.7.56.1811"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_axiom_breaking_"
    "mixed_surface_reactivation"
)
FOLLOWUP_ROUTE = "8.7.56.1815"


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


# 関数: repo相対の表示パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative display path when possible."""
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


# 関数: JSON/CSV artifact を書き出す。

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one CSV rows file."""
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


# 関数: solver module を読み込む。

def load_qball_module():
    """Load the retained scalar Q-ball solver as a reusable module."""
    spec = importlib.util.spec_from_file_location("wavep_qball_charge_mapping", QBALL_SOLVER)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to load module from {QBALL_SOLVER}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 関数: scalar ground-state row を取り出す。

def extract_scalar_ground_state(qball_branch_refresh: dict) -> dict:
    """Extract the scalar ground-state row from the retained branch refresh metrics."""
    for row_data in qball_branch_refresh["evidence"]["discrete_mode_rows"]:
        if int(row_data["mode_index"]) == 1:
            return {
                "beta_n": float(row_data["beta_n"]),
                "central_amplitude": float(row_data["central_amplitude"]),
            }

    raise SystemExit("[fail] missing scalar ground-state row in charge-mapping branch refresh metrics")


# 関数: retained exact profile overlap form factor を評価する。

def form_factor(radius: np.ndarray, weight: np.ndarray, norm: float, q_ratio: float) -> float:
    """Evaluate one normalized spherical-overlap form factor."""
    qx = float(q_ratio) * radius
    sinc = np.ones_like(qx)
    mask = np.abs(qx) > 1.0e-12
    sinc[mask] = np.sin(qx[mask]) / qx[mask]
    numerator = np.trapezoid(weight * sinc, radius)
    return float(numerator / norm)


# 関数: 主要 no-go 式を返す。

def build_formulae() -> dict[str, str]:
    """Return the global-completion obstruction formulas."""
    return {
        "carrier_surface": "A_FF(q) = |q|",
        "psd_mixed_matrix": "M(q) = [[A_FF(q), A_FH(q)], [A_FH(q), A_HH(q)]], A_HH(q) >= 0",
        "largest_eigenvalue": "lambda_+(q) = 0.5 * (A_FF + A_HH + sqrt((A_FF - A_HH)^2 + 4 A_FH^2))",
        "psd_lower_bound": "lambda_+(q) >= max(A_FF(q), A_HH(q)) >= A_FF(q) = |q|",
        "obstruction_rule": "If F_exact(q) < |q|, then no PSD Hermitian mixed pack with retained carrier saturation can satisfy lambda_+(q) = F_exact(q).",
    }


# 関数: `.1807-.1810` を実行する。

def main() -> None:
    """Execute the vacuum-saturated PSD mixed-pack obstruction theorem branch."""
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
        WINDOW_GATE,
        WINDOW_ROUTE,
        CLOSEOUT_GATE,
        QBALL_BRANCH_REFRESH,
        QBALL_SOLVER,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    window_summary = read_json(WINDOW_GATE)["summary"]
    window_route = read_json(WINDOW_ROUTE)["summary"]
    closeout_summary = read_json(CLOSEOUT_GATE)["summary"]
    qball_branch_refresh = read_json(QBALL_BRANCH_REFRESH)

    qball_module = load_qball_module()
    scalar_ground_state = extract_scalar_ground_state(qball_branch_refresh)
    radius, field, _field_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    weight = (field**2) * (radius**2)
    norm = float(np.trapezoid(weight, radius))

    q_window_edge = float(window_summary["scalar_compatible_window_upper_edge_over_m0"])
    q_probe = 0.3
    if q_probe <= q_window_edge:
        q_probe = q_window_edge + 0.02

    f_exact_at_q_probe = form_factor(radius, weight, norm, q_probe)
    carrier_at_q_probe = q_probe
    deficit_at_q_probe = carrier_at_q_probe - f_exact_at_q_probe

    q_scan = np.linspace(q_window_edge, 1.0, 2001)
    gap_scan = q_scan - np.array([form_factor(radius, weight, norm, float(q_val)) for q_val in q_scan])
    max_gap_on_post_window_scan = float(np.max(gap_scan))
    mean_gap_on_post_window_scan = float(np.mean(gap_scan))
    post_window_positive_gap_detected = bool(np.any(gap_scan > 1.0e-8))

    inventory_ready = all(
        item is not None
        for item in (
            hit(status_text, "8.7.56.1795"),
            hit(roadmap_text, "8.7.56.1795-.1798"),
            hit(current_problem_text, "branch-local completion theorem"),
            hit(current_status_text, "fixed-q exact promotion succeeded"),
            hit(unified_text, "`.1795-.1798`"),
            hit(long_text, "29. `8.7.56.1799-.1802`"),
            hit(part5_text, "`.1791-.1794`"),
        )
    )
    full_q_window_retained = bool(
        window_summary["full_q_exact_hh_surface_window_available"]
        and window_summary["window_covers_q_theory"]
        and window_summary["exact_scalar_promotion_selected"]
    )
    psd_mixed_pack_lower_bound_theorem_derived = True
    global_exact_completion_under_vacuum_saturated_psd_pack = False
    non_rank_one_mixed_surface_alone_not_sufficient = True
    new_surface_must_break_current_axiom = True
    same_level_post_window_psd_retry_admissible = False
    branch_honest = all(
        (
            inventory_ready,
            full_q_window_retained,
            psd_mixed_pack_lower_bound_theorem_derived,
            post_window_positive_gap_detected,
            not global_exact_completion_under_vacuum_saturated_psd_pack,
            non_rank_one_mixed_surface_alone_not_sufficient,
            new_surface_must_break_current_axiom,
            not same_level_post_window_psd_retry_admissible,
        )
    )

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "global HH obstruction inventory ready",
            truth(inventory_ready),
            "The obstruction theorem starts only after `.1803-.1806` has already frozen the full-q HH window and its remaining reopen surfaces.",
        ),
        row(
            "full_q_window_retained",
            "pass" if full_q_window_retained else "reject",
            "retained scalar-compatible HH window",
            truth(full_q_window_retained),
            "The no-go theorem is downstream of the retained full-q HH window, not a replacement for it.",
        ),
        row(
            "psd_mixed_pack_lower_bound_theorem_derived",
            "pass",
            "PSD mixed-pack lower-bound theorem derived",
            truth(psd_mixed_pack_lower_bound_theorem_derived),
            "For any real symmetric mixed pack with A_HH >= 0, the largest eigenvalue is bounded below by A_FF = |q| under the retained carrier axiom.",
        ),
        row(
            "q_probe_over_m0",
            "watch",
            "post-window probe q/m0",
            q_probe,
            "The probe point is chosen above the scalar-compatible window edge so the post-window deficit is explicit numerically.",
        ),
        row(
            "f_exact_at_q_probe",
            "watch",
            "retained exact profile amplitude at q_probe",
            f_exact_at_q_probe,
            "This is the exact scalar profile overlap at one representative post-window point.",
        ),
        row(
            "carrier_at_q_probe",
            "watch",
            "vacuum-saturated FF carrier at q_probe",
            carrier_at_q_probe,
            "The retained carrier surface equals |q| pointwise.",
        ),
        row(
            "deficit_at_q_probe",
            "watch",
            "carrier minus exact scalar amplitude at q_probe",
            deficit_at_q_probe,
            "Positive deficit means the retained exact scalar amplitude already lies below the FF carrier, so no PSD mixed completion can match it there.",
        ),
        row(
            "post_window_positive_gap_detected",
            "pass" if post_window_positive_gap_detected else "reject",
            "positive carrier gap detected on post-window scan",
            truth(post_window_positive_gap_detected),
            "A positive post-window gap confirms that the obstruction is not pointwise accidental but persists on the q > q_HH,max side.",
        ),
        row(
            "max_gap_on_post_window_scan",
            "watch",
            "max carrier gap on q >= q_HH,max scan",
            max_gap_on_post_window_scan,
            "This is the largest gap q - F_exact(q) seen on the retained post-window scan up to q/m0 = 1.",
        ),
        row(
            "mean_gap_on_post_window_scan",
            "watch",
            "mean carrier gap on q >= q_HH,max scan",
            mean_gap_on_post_window_scan,
            "The mean gap shows the obstruction is not confined to one exceptional point.",
        ),
        row(
            "global_exact_completion_under_vacuum_saturated_psd_pack",
            "reject",
            "global exact completion under vacuum-saturated PSD mixed pack",
            truth(global_exact_completion_under_vacuum_saturated_psd_pack),
            "Retaining A_FF = |q| and A_HH >= 0 blocks any global lambda_+ = F_exact completion once F_exact(q) drops below |q|.",
        ),
        row(
            "non_rank_one_mixed_surface_alone_not_sufficient",
            "pass",
            "non-rank-one mixed surface alone not sufficient",
            truth(non_rank_one_mixed_surface_alone_not_sufficient),
            "The obstruction theorem does not assume rank-one, so simply reopening non-rank-one coherence is not enough under the retained PSD carrier axioms.",
        ),
        row(
            "new_surface_must_break_current_axiom",
            "pass",
            "new surface must break current carrier/PSD/Hermitian axiom",
            truth(new_surface_must_break_current_axiom),
            "Any genuine global completion must now alter at least one retained axiom: FF saturation, PSD HH diagonal, or the canonical Hermitian eigenvalue rule.",
        ),
        row(
            "same_level_post_window_psd_retry_admissible",
            "reject",
            "same-level post-window PSD retry admissible",
            truth(same_level_post_window_psd_retry_admissible),
            "The next honest route is an axiom-breaking surface, not another same-level PSD mixed retry.",
        ),
        row(
            "branch_honest",
            "pass" if branch_honest else "reject",
            "global completion obstruction theorem honest",
            truth(branch_honest),
            "The theorem is honest only if it keeps the scalar-compatible window while proving that the retained PSD carrier pack cannot extend it globally.",
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
            "window_gate": display_path(WINDOW_GATE),
            "window_route": display_path(WINDOW_ROUTE),
            "closeout_gate": display_path(CLOSEOUT_GATE),
            "qball_branch_refresh": display_path(QBALL_BRANCH_REFRESH),
            "solver_module": display_path(QBALL_SOLVER),
        },
        "constants": {
            "q_hh_max_over_m0": q_window_edge,
            "q_probe_over_m0": q_probe,
            "selected_primary_reopen_surface": "axiom_breaking_surface_beyond_vacuum_saturated_psd_mixed_pack",
            "selected_secondary_reopen_surface": "substantive_pack_update_changing_ff_carrier_psd_hh_or_hermitian_eigenvalue_rule",
            "selected_reserve_reopen_surface": "future_external_input_guiding_axiom_breaking_global_completion_surface",
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "full_q_window_retained": full_q_window_retained,
        "q_hh_max_over_m0": q_window_edge,
        "q_probe_over_m0": q_probe,
        "f_exact_at_q_probe": f_exact_at_q_probe,
        "carrier_at_q_probe": carrier_at_q_probe,
        "deficit_at_q_probe": deficit_at_q_probe,
        "post_window_positive_gap_detected": post_window_positive_gap_detected,
        "max_gap_on_post_window_scan": max_gap_on_post_window_scan,
        "mean_gap_on_post_window_scan": mean_gap_on_post_window_scan,
        "psd_mixed_pack_lower_bound_theorem_derived": psd_mixed_pack_lower_bound_theorem_derived,
        "global_exact_completion_under_vacuum_saturated_psd_pack": global_exact_completion_under_vacuum_saturated_psd_pack,
        "non_rank_one_mixed_surface_alone_not_sufficient": non_rank_one_mixed_surface_alone_not_sufficient,
        "new_surface_must_break_current_axiom": new_surface_must_break_current_axiom,
        "same_level_post_window_psd_retry_admissible": same_level_post_window_psd_retry_admissible,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": branch_honest,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": build_formulae(),
        "hits": {
            "status_branch_hit": hit(status_text, "8.7.56.1795"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1795-.1798"),
            "current_problem_hit": hit(current_problem_text, "branch-local completion theorem"),
            "current_status_hit": hit(current_status_text, "fixed-q exact promotion succeeded"),
            "unified_roadmap_hit": hit(unified_text, "`.1795-.1798`"),
            "long_roadmap_hit": hit(long_text, "29. `8.7.56.1799-.1802`"),
            "part5_hit": hit(part5_text, "`.1791-.1794`"),
        },
        "carry_over": {
            "window_summary": window_summary,
            "window_route": window_route,
            "closeout_summary": closeout_summary,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload("8.7.56.1807", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence),
        ),
        "audit": write_artifact(
            "audit",
            payload("8.7.56.1808", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload("8.7.56.1809", f"{STEP_NAME} declaration gate", inputs, rows, summary, decision, evidence),
        ),
        "route_sync": write_artifact(
            "route_sync",
            payload("8.7.56.1810", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence),
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
